import os
import time
import json
import queue
import threading
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
from typing import List, Optional
import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import serial
import serial.tools.list_ports
from bleak import BleakScanner, BleakClient
import google.generativeai as genai
from dotenv import load_dotenv

# ---------------------------
# Config & Setup
# ---------------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# BLE Configuration
SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
DEVICE_NAME = "ESP32_EEG_Monitor"

# Serial Configuration
BAUD_RATE = 115200
MAX_DATA_POINTS = 500
EMOTION_SEQ = ["happy", "sad", "angry", "relaxed", "focused", "anxious", "neutral"]

# ---------------------------
# Data Models
# ---------------------------
@dataclass
class EEGReading:
    timestamp: datetime
    channels: List[float]
    raw_data: str

@dataclass
class EmotionResult:
    emotion: str
    confidence: float
    timestamp: datetime
    reasoning: str = ""

# ---------------------------
# Gemini Analyzer
# ---------------------------
class EEGAnalyzer:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def analyze_emotion(self, eeg_data: List[float]) -> EmotionResult:
        if not eeg_data or len(eeg_data) < 4:
            return EmotionResult("neutral", 0.0, datetime.now(), "Insufficient EEG data")

        # Calculate statistics
        mean_val = np.mean(eeg_data)
        std_val = np.std(eeg_data)
        max_val = np.max(eeg_data)
        min_val = np.min(eeg_data)

        prompt = f"""
You are an expert neuroscientist specializing in EEG emotion recognition.

Analyze this EEG data:
- Sample values (first 20): {eeg_data[:20]}
- Mean: {mean_val:.6f} V
- Std Dev: {std_val:.6f} V
- Range: {min_val:.6f} to {max_val:.6f} V
- Total samples: {len(eeg_data)}

EEG Emotion Guidelines:
- High amplitude (>0.03V): Strong activity - stress, excitement, focus
- Medium (0.01-0.03V): Normal - relaxed or neutral
- Low (<0.01V): Very relaxed, meditative, drowsy
- High variability: Unstable - anxiety, active thinking
- Low variability: Stable - calm, focused

Classify the emotion as ONE of: happy, sad, angry, relaxed, focused, anxious, neutral

Respond ONLY with this exact JSON format:
{{
  "emotion": "happy|sad|angry|relaxed|focused|anxious|neutral",
  "confidence": 0.85,
  "reasoning": "Brief explanation based on EEG patterns"
}}
"""

        try:
            resp = self.model.generate_content(prompt)
            text = (resp.text or "").strip()
            
            # Extract JSON
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = text[start_idx:end_idx]
                data = json.loads(json_str)
                emo = str(data.get("emotion", "neutral")).lower()
                conf = float(data.get("confidence", 0.6))
                reason = str(data.get("reasoning", "AI analysis"))
                
                if emo not in EMOTION_SEQ:
                    emo = "neutral"
                conf = min(max(conf, 0.0), 1.0)
                
                return EmotionResult(emo, conf, datetime.now(), reason)
            else:
                # Fallback parsing
                text_low = text.lower()
                for emo in EMOTION_SEQ:
                    if emo in text_low:
                        return EmotionResult(emo, 0.6, datetime.now(), "Parsed from AI response")
                return EmotionResult("neutral", 0.3, datetime.now(), "Could not parse AI response")
                
        except Exception as e:
            return EmotionResult("neutral", 0.2, datetime.now(), f"Analysis error: {str(e)}")

# ---------------------------
# BLE Data Manager
# ---------------------------
class BLEDataManager:
    def __init__(self):
        self.data_queue = queue.Queue(maxsize=1000)
        self.is_running = False
        self.ble_client: Optional[BleakClient] = None
        self.last_data_time: Optional[datetime] = None
        self.connection_status = "Not connected"
        
    def notification_handler(self, sender, data):
        """Handle incoming BLE notifications"""
        try:
            json_str = data.decode('utf-8')
            json_data = json.loads(json_str)
            
            # Extract 4 channels from JSON
            channels = [
                json_data.get('ain0', 0),
                json_data.get('ain1', 0),
                json_data.get('ain2', 0),
                json_data.get('ain3', 0)
            ]
            
            self.last_data_time = datetime.now()
            reading = EEGReading(self.last_data_time, channels, json_str)
            
            try:
                self.data_queue.put_nowait(reading)
            except queue.Full:
                try:
                    _ = self.data_queue.get_nowait()
                    self.data_queue.put_nowait(reading)
                except queue.Empty:
                    pass
                    
        except Exception as e:
            self.connection_status = f"Data error: {str(e)}"
    
    async def scan_for_device(self):
        """Scan for ESP32 BLE device"""
        try:
            devices = await BleakScanner.discover(timeout=10.0)
            for device in devices:
                if device.name and DEVICE_NAME in device.name:
                    return device
            return None
        except Exception as e:
            self.connection_status = f"Scan error: {str(e)}"
            return None
    
    async def connect(self):
        """Connect to ESP32 BLE device"""
        try:
            device = await self.scan_for_device()
            if not device:
                self.connection_status = "Device not found"
                return False
            
            self.ble_client = BleakClient(device.address, timeout=20.0)
            await self.ble_client.connect()
            
            await self.ble_client.start_notify(CHARACTERISTIC_UUID, self.notification_handler)
            
            self.is_running = True
            self.connection_status = "Connected"
            return True
            
        except Exception as e:
            self.connection_status = f"Connection failed: {str(e)}"
            return False
    
    async def disconnect(self):
        """Disconnect from BLE device"""
        if self.ble_client:
            try:
                await self.ble_client.stop_notify(CHARACTERISTIC_UUID)
                await self.ble_client.disconnect()
            except Exception:
                pass
            self.ble_client = None
        self.is_running = False
        self.connection_status = "Disconnected"
    
    def get_latest(self) -> Optional[EEGReading]:
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None
    
    def connected(self) -> bool:
        if not self.is_running or not self.ble_client:
            return False
        if self.last_data_time is None:
            return True
        return (datetime.now() - self.last_data_time) < timedelta(seconds=10)

# ---------------------------
# Serial Data Manager
# ---------------------------
class SerialDataManager:
    def __init__(self):
        self.data_queue = queue.Queue(maxsize=1000)
        self.is_running = False
        self.serial_connection: Optional[serial.Serial] = None
        self.thread: Optional[threading.Thread] = None
        self.last_data_time: Optional[datetime] = None
        self.error_count = 0

    def get_available_ports(self) -> List[str]:
        try:
            return [p.device for p in serial.tools.list_ports.comports()]
        except Exception:
            return []

    def start(self, port: str, baud: int = BAUD_RATE) -> bool:
        try:
            self.stop()
            self.serial_connection = serial.Serial(
                port=port,
                baudrate=baud,
                timeout=1,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
            )
            self.is_running = True
            self.error_count = 0
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
            return True
        except Exception as e:
            st.error(f"❌ Failed to open {port}: {e}")
            return False

    def stop(self):
        self.is_running = False
        if self.serial_connection:
            try:
                self.serial_connection.close()
            except Exception:
                pass
            self.serial_connection = None

    def _read_loop(self):
        while self.is_running and self.serial_connection:
            try:
                line = self.serial_connection.readline()
                decoded = line.decode("utf-8", errors="ignore").strip()
                if decoded:
                    self.last_data_time = datetime.now()
                    parts = decoded.replace(",", " ").split()
                    vals = []
                    for p in parts:
                        try:
                            vals.append(float(p))
                        except ValueError:
                            continue
                    if vals:
                        if len(vals) < 4:
                            vals = vals + [0.0] * (4 - len(vals))
                        else:
                            vals = vals[:4]
                        reading = EEGReading(self.last_data_time, vals, decoded)
                        try:
                            self.data_queue.put_nowait(reading)
                        except queue.Full:
                            try:
                                _ = self.data_queue.get_nowait()
                                self.data_queue.put_nowait(reading)
                            except queue.Empty:
                                pass
                time.sleep(0.005)
            except Exception:
                self.error_count += 1
                if self.error_count > 25:
                    break
                time.sleep(0.05)

    def get_latest(self) -> Optional[EEGReading]:
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None

    def connected(self) -> bool:
        if not self.is_running or not self.serial_connection:
            return False
        if self.last_data_time is None:
            return True
        return (datetime.now() - self.last_data_time) < timedelta(seconds=5)

# ---------------------------
# Plot Helpers
# ---------------------------
def create_realtime_plot(buffer: deque) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Channel 1 (Frontal)", "Channel 2 (Temporal)", 
                       "Channel 3 (Parietal)", "Channel 4 (Occipital)"],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )
    
    if not buffer:
        fig.update_layout(height=500, showlegend=False, title="EEG Signals (waiting for data...)")
        return fig

    recent = list(buffer)[-120:]
    timestamps = [r.timestamp for r in recent]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for i in range(4):
        row, col = (i // 2) + 1, (i % 2) + 1
        y = [r.channels[i] if i < len(r.channels) else 0 for r in recent]
        fig.add_trace(
            go.Scatter(x=timestamps, y=y, mode="lines", name=f"Ch{i+1}", 
                      line=dict(color=colors[i], width=2)),
            row=row, col=col
        )

    fig.update_layout(
        height=500, 
        showlegend=False, 
        title="Real-time EEG Signals (4 Channels)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    for r in range(1, 3):
        for c in range(1, 3):
            fig.update_xaxes(title_text="Time", row=r, col=c, showgrid=True, gridcolor='rgba(128,128,128,0.2)')
            fig.update_yaxes(title_text="Voltage (V)", row=r, col=c, showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    
    return fig

# ---------------------------
# Demo Mode Functions
# ---------------------------
def add_demo_sample():
    t = time.time()
    channels = [
        np.sin(0.9 * t + i * 0.6) * 0.018 + np.cos(0.15 * t + i) * 0.007 + np.random.normal(0, 0.0015)
        for i in range(4)
    ]
    reading = EEGReading(datetime.now(), channels, f"demo: {channels}")
    st.session_state.buffer.append(reading)

def append_random_demo_emotion():
    last_emo = st.session_state.history[-1].emotion if st.session_state.history else None
    choices = [e for e in EMOTION_SEQ if e != last_emo]
    emo = random.choice(choices)
    conf = random.uniform(0.75, 0.95)
    st.session_state.history.append(
        EmotionResult(emo, conf, datetime.now(), "Demo mode - simulated emotion")
    )

# ---------------------------
# Streamlit App
# ---------------------------
def init_state():
    if "connection_mode" not in st.session_state:
        st.session_state.connection_mode = "Demo"  # "Demo", "BLE", "Serial"
    if "ble_manager" not in st.session_state:
        st.session_state.ble_manager = BLEDataManager()
    if "serial_manager" not in st.session_state:
        st.session_state.serial_manager = SerialDataManager()
    if "buffer" not in st.session_state:
        st.session_state.buffer = deque(maxlen=MAX_DATA_POINTS)
    if "history" not in st.session_state:
        st.session_state.history: List[EmotionResult] = []
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = EEGAnalyzer(API_KEY) if API_KEY else None
    if "last_analysis_time" not in st.session_state:
        st.session_state.last_analysis_time = datetime.min

def app_header():
    st.markdown("""
    <div style="text-align:center; padding:20px; border-radius:15px;
         background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); color:#fff;
         box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
      <h1 style="margin:0; font-size:2.5rem;">🧠 EEG Emotion Analyzer</h1>
      <p style="margin:8px 0 0; font-size:1.1rem;">Real-time emotion monitoring with ESP32 + Gemini AI</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="EEG Emotion Analyzer", 
        layout="wide",
        page_icon="🧠"
    )
    init_state()
    app_header()

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Status
        if st.session_state.analyzer:
            st.success("✅ Gemini AI: Ready")
        else:
            st.error("❌ Gemini API: Not configured")
            st.info("Add GEMINI_API_KEY to .env file")
        
        st.divider()
        
        # Connection Mode Selection
        st.subheader("📡 Connection Mode")
        mode = st.radio(
            "Select Mode:",
            ["Demo", "BLE (Bluetooth)", "Serial (USB)"],
            index=0
        )
        
        # Mode-specific controls
        if mode == "BLE (Bluetooth)":
            st.caption("Connect to ESP32 via Bluetooth")
            if st.session_state.connection_mode != "BLE":
                st.session_state.serial_manager.stop()
                st.session_state.connection_mode = "BLE"
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔍 Scan & Connect", use_container_width=True):
                    with st.spinner("Scanning..."):
                        success = asyncio.run(st.session_state.ble_manager.connect())
                        if success:
                            st.success("✅ Connected!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"❌ {st.session_state.ble_manager.connection_status}")
            
            with col2:
                if st.button("🔌 Disconnect", use_container_width=True):
                    asyncio.run(st.session_state.ble_manager.disconnect())
                    st.rerun()
            
            st.caption(f"Status: {st.session_state.ble_manager.connection_status}")
            
        elif mode == "Serial (USB)":
            st.caption("Connect via USB Serial")
            if st.session_state.connection_mode != "Serial":
                asyncio.run(st.session_state.ble_manager.disconnect())
                st.session_state.connection_mode = "Serial"
            
            ports = st.session_state.serial_manager.get_available_ports()
            if ports:
                port = st.selectbox("Port", ports)
                baud = st.selectbox("Baud Rate", [115200, 57600, 9600], index=0)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("▶ Connect", use_container_width=True):
                        if st.session_state.serial_manager.start(port, baud):
                            st.success(f"✅ Connected to {port}")
                            time.sleep(1)
                            st.rerun()
                
                with col2:
                    if st.button("⏹ Stop", use_container_width=True):
                        st.session_state.serial_manager.stop()
                        st.rerun()
            else:
                st.warning("No serial ports found")
        
        else:  # Demo Mode
            st.caption("Simulated EEG data for testing")
            if st.session_state.connection_mode != "Demo":
                st.session_state.serial_manager.stop()
                asyncio.run(st.session_state.ble_manager.disconnect())
                st.session_state.connection_mode = "Demo"
        
        st.divider()
        
        # Statistics
        st.subheader("📊 Statistics")
        st.metric("Mode", st.session_state.connection_mode)
        st.metric("Data Points", len(st.session_state.buffer))
        st.metric("Emotions Detected", len(st.session_state.history))
        
        st.divider()
        
        # Controls
        col1, col2 = st.columns(2)
        with col1:
            auto_refresh = st.toggle("Auto-refresh", value=True)
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        
        refresh_interval = st.slider("Update interval (s)", 0.5, 3.0, 1.0, 0.5)
        
        if st.button("🗑️ Clear All Data", use_container_width=True):
            st.session_state.buffer.clear()
            st.session_state.history.clear()
            st.rerun()

    # Main Content Area
    
    # Data Collection based on mode
    if st.session_state.connection_mode == "Demo":
        add_demo_sample()
        if len(st.session_state.buffer) % 40 == 0 and len(st.session_state.buffer) > 0:
            append_random_demo_emotion()
    
    elif st.session_state.connection_mode == "BLE":
        while True:
            reading = st.session_state.ble_manager.get_latest()
            if reading is None:
                break
            st.session_state.buffer.append(reading)
    
    elif st.session_state.connection_mode == "Serial":
        while True:
            reading = st.session_state.serial_manager.get_latest()
            if reading is None:
                break
            st.session_state.buffer.append(reading)
    
    # Emotion Analysis
    if st.session_state.analyzer and st.session_state.buffer:
        time_since_last = (datetime.now() - st.session_state.last_analysis_time).total_seconds()
        if time_since_last >= 10:  # Analyze every 10 seconds
            recent_data = [r.channels for r in list(st.session_state.buffer)[-50:]]
            flat_data = [val for sublist in recent_data for val in sublist]
            
            if len(flat_data) >= 10:
                result = st.session_state.analyzer.analyze_emotion(flat_data)
                st.session_state.history.append(result)
                st.session_state.last_analysis_time = datetime.now()
                st.toast(f"🧠 {result.emotion.title()} ({result.confidence:.0%})", icon="🧠")
    
    # Display Layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Real-time EEG Signals")
        if st.session_state.buffer:
            fig = create_realtime_plot(st.session_state.buffer)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Waiting for EEG data...")
    
    with col2:
        st.subheader("🧠 Current Emotion")
        if st.session_state.history:
            last = st.session_state.history[-1]
            emoji_map = {
                "happy": "😊", "sad": "😢", "angry": "😠",
                "relaxed": "😌", "focused": "🤔", "anxious": "😰", "neutral": "😐"
            }
            color_map = {
                "happy": "#FFD93D", "sad": "#6C9BCF", "angry": "#FF6B6B",
                "relaxed": "#95E1D3", "focused": "#A78BFA", "anxious": "#FB8B24", "neutral": "#9CA3AF"
            }
            
            emoji = emoji_map.get(last.emotion, "😐")
            color = color_map.get(last.emotion, "#9CA3AF")
            
            st.markdown(f"""
            <div style='text-align:center; padding:30px; 
                 background:linear-gradient(135deg, {color}22 0%, {color}44 100%); 
                 border-radius:15px; border:3px solid {color};
                 box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <div style='font-size:100px;'>{emoji}</div>
                <h1 style='color:{color}; margin:10px 0; text-transform:uppercase;'>{last.emotion}</h1>
                <h2 style='margin:10px 0;'>{last.confidence*100:.1f}% Confidence</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.info(f"💭 {last.reasoning}")
            
            time_ago = (datetime.now() - last.timestamp).total_seconds()
            st.caption(f"⏱️ Analyzed {int(time_ago)}s ago")
        else:
            st.info("⏳ Waiting for emotion analysis...")
        
        # Latest values
        st.subheader("📊 Latest Channel Values")
        if st.session_state.buffer:
            latest = st.session_state.buffer[-1]
            for i, val in enumerate(latest.channels[:4]):
                st.metric(f"Channel {i+1}", f"{val:.6f} V")
    
    # Emotion History
    st.divider()
    st.subheader("📚 Emotion History (Latest 15)")
    if st.session_state.history:
        df = pd.DataFrame([
            {
                "⏰ Time": x.timestamp.strftime("%H:%M:%S"),
                "😊 Emotion": x.emotion.title(),
                "📊 Confidence": f"{x.confidence:.0%}",
                "💭 Reasoning": x.reasoning[:50] + "..." if len(x.reasoning) > 50 else x.reasoning
            }
            for x in reversed(st.session_state.history[-15:])
        ])
        st.dataframe(df, use_container_width=True, hide_index=True, height=300)
    else:
        st.info("No emotion history yet")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()