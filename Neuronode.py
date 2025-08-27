# app.py
import os
import time
import json
import queue
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import serial
import serial.tools.list_ports
import google.generativeai as genai
from dotenv import load_dotenv

# ---------------------------
# Config & Setup
# ---------------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")  # from .env

BAUD_RATE = 115200
MAX_DATA_POINTS = 500
EMOTION_SEQ = ["happy", "sad", "anger", "relaxed", "interested", "neutral"]

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
# Gemini Analyzer (sync)
# ---------------------------
class EEGAnalyzer:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        # Use a widely available model name
        self.model = genai.GenerativeModel('gemini-pro')

    def analyze_emotion(self, eeg_data: List[float]) -> EmotionResult:
        """Simple, robust emotion analysis. Falls back safely."""
        if not eeg_data:
            return EmotionResult("neutral", 0.0, datetime.now(), "No EEG data")

        prompt = f"""
        You are an EEG expert. Analyze this EEG sample (μV): {eeg_data[:32]} ...
        Classify primary emotion as one of:
        happy, sad, anger, relaxed, interested, neutral.
        Respond ONLY in JSON, exactly this shape:
        {{
          "emotion": "happy|sad|anger|relaxed|interested|neutral",
          "confidence": 0.82,
          "reasoning": "short explanation"
        }}
        """

        try:
            resp = self.model.generate_content(prompt)
            text = (resp.text or "").strip()
            # Try strict JSON parse first
            try:
                data = json.loads(text)
                emo = str(data.get("emotion", "neutral")).lower()
                conf = float(data.get("confidence", 0.6))
                reason = str(data.get("reasoning", ""))
                if emo not in EMOTION_SEQ:
                    emo = "neutral"
                conf = min(max(conf, 0.0), 1.0)
                return EmotionResult(emo, conf, datetime.now(), reason)
            except Exception:
                # Fallback: scan for a known label
                text_low = text.lower()
                for emo in EMOTION_SEQ:
                    if emo in text_low:
                        return EmotionResult(emo, 0.6, datetime.now(), "Heuristic parse")
                return EmotionResult("neutral", 0.3, datetime.now(), "Unstructured model response")
        except Exception as e:
            return EmotionResult("neutral", 0.2, datetime.now(), f"Gemini error: {e}")

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
            ports = []
            for port in serial.tools.list_ports.comports():
                ports.append(f"{port.device} - {port.description}")
            return ports
        except Exception as e:
            st.error(f"Error detecting ports: {e}")
            return []

    def extract_port_name(self, port_display: str) -> str:
        """Extract just the port name from the display string"""
        return port_display.split(" - ")[0] if " - " in port_display else port_display

    def start(self, port_display: str, baud: int = BAUD_RATE) -> bool:
        try:
            self.stop()  # in case a previous run is hanging
            
            # Extract actual port name from display string
            port = self.extract_port_name(port_display)
            
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
        if self.thread and self.thread.is_alive():
            # don't hard-block; allow thread to terminate
            pass

    def _read_loop(self):
        while self.is_running and self.serial_connection:
            try:
                line = self.serial_connection.readline()
                decoded = line.decode("utf-8", errors="ignore").strip()
                if decoded:
                    self.last_data_time = datetime.now()
                    # Accept comma/space separated floats
                    parts = decoded.replace(",", " ").split()
                    vals = []
                    for p in parts:
                        try:
                            vals.append(float(p))
                        except ValueError:
                            continue
                    # Keep at least 4 channels for display grid; pad/truncate to 8
                    if vals:
                        if len(vals) < 8:
                            vals = vals + [0.0] * (8 - len(vals))
                        else:
                            vals = vals[:8]
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
            return True  # just opened
        return (datetime.now() - self.last_data_time) < timedelta(seconds=5)

# ---------------------------
# Plot Helpers
# ---------------------------
def create_realtime_plot(buffer: deque) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=[f"Ch{i+1}" for i in range(8)],
        vertical_spacing=0.12,
        horizontal_spacing=0.06,
    )
    if not buffer:
        fig.update_layout(height=520, showlegend=False, title="EEG Signals (waiting...)")
        return fig

    recent = list(buffer)[-120:]
    timestamps = [r.timestamp for r in recent]
    for i in range(8):
        row, col = (i // 4) + 1, (i % 4) + 1
        y = [r.channels[i] if i < len(r.channels) else 0 for r in recent]
        fig.add_trace(go.Scatter(x=timestamps, y=y, mode="lines", name=f"Ch{i+1}"), row=row, col=col)

    fig.update_layout(height=520, showlegend=False, title="Real-time EEG Signals (μV)")
    for r in range(1, 3):
        for c in range(1, 5):
            fig.update_xaxes(title_text="Time", row=r, col=c)
            fig.update_yaxes(title_text="μV", row=r, col=c)
    return fig

# ---------------------------
# Streamlit App
# ---------------------------
def init_state():
    if "serial" not in st.session_state:
        st.session_state.serial = SerialDataManager()
    if "buffer" not in st.session_state:
        st.session_state.buffer = deque(maxlen=MAX_DATA_POINTS)
    if "history" not in st.session_state:
        st.session_state.history: List[EmotionResult] = []
    if "demo_mode" not in st.session_state:
        st.session_state.demo_mode = True
    if "demo_index" not in st.session_state:
        st.session_state.demo_index = 0
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = EEGAnalyzer(API_KEY) if API_KEY else None

def add_demo_sample():
    # Smooth demo waveform with slight randomness
    t = time.time()
    channels = [np.sin(0.9 * t + i * 0.6) * 18 + np.cos(0.15 * t + i) * 7 + np.random.normal(0, 1.5)
                for i in range(8)]
    reading = EEGReading(datetime.now(), channels, "demo," + ",".join(f"{v:.2f}" for v in channels))
    st.session_state.buffer.append(reading)

def append_sequential_demo_emotion():
    emo = EMOTION_SEQ[st.session_state.demo_index % len(EMOTION_SEQ)]
    st.session_state.demo_index += 1
    st.session_state.history.append(
        EmotionResult(emo, 0.95, datetime.now(), "Demo (sequential rotation)")
    )

def app_header():
    st.markdown("""
    <div style="text-align:center;padding:16px;border-radius:12px;
         background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);color:#fff;">
      <h2 style="margin:0;">EEG Emotion Analyzer</h2>
      <p style="margin:6px 0 0;">Real-time dashboard with Demo & Live ESP32 modes + Gemini AI</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="EEG Emotion Analyzer", layout="wide")
    init_state()
    app_header()

    # ------------- Sidebar -------------
    with st.sidebar:
        st.header("⚙️ Configuration")
        # Gemini status
        if st.session_state.analyzer:
            st.success("✅ Gemini API: Ready")
        else:
            st.warning("⚠️ Gemini API: Not configured (set GEMINI_API_KEY in .env)")

        # Get available ports and show them with descriptions
        available_ports = st.session_state.serial.get_available_ports()
        
        if available_ports:
            st.info(f"📋 Found {len(available_ports)} available port(s)")
            port_options = ["Demo Mode"] + available_ports
        else:
            st.warning("⚠️ No serial ports detected")
            port_options = ["Demo Mode"]
        
        port = st.selectbox("Serial Port", port_options, index=0)
        baud = st.selectbox("Baud Rate", [115200, 57600, 38400, 9600], index=0)

        # Show detected ports info
        if available_ports:
            with st.expander("🔍 Detected Ports"):
                for p in available_ports:
                    st.text(p)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("▶ Start", use_container_width=True):
                if port == "Demo Mode":
                    st.session_state.demo_mode = True
                    st.session_state.serial.stop()
                    st.toast("✅ Demo mode started", icon="✅")
                else:
                    ok = st.session_state.serial.start(port, baud)
                    if ok:
                        st.session_state.demo_mode = False
                        port_name = st.session_state.serial.extract_port_name(port)
                        st.toast(f"🔌 ESP32 connected on {port_name}", icon="🔌")
                    else:
                        st.session_state.demo_mode = True

        with col_b:
            if st.button("⏹ Stop", use_container_width=True):
                st.session_state.serial.stop()
                st.session_state.demo_mode = True
                st.toast("⏹️ Stopped. Back to Demo mode.", icon="⏹️")

        # Status badges
        is_live = not st.session_state.demo_mode
        conn_ok = st.session_state.serial.connected() if is_live else False
        st.markdown("---")
        st.markdown(f"**Mode:** {'🟢 Live (ESP32)' if is_live else '🟣 Demo'}")
        st.markdown(f"**ESP32:** {('✅ Connected' if conn_ok else '❌ Disconnected') if is_live else '—'}")
        st.markdown(f"**Last data:** {st.session_state.serial.last_data_time.strftime('%H:%M:%S') if st.session_state.serial.last_data_time else '—'}")
        st.markdown("---")

        # Update controls
        refresh_col1, refresh_col2 = st.columns(2)
        with refresh_col1:
            refresh_now = st.button("🔄 Refresh now", use_container_width=True)
        with refresh_col2:
            auto = st.toggle("Continuous update", value=True)

        interval = st.slider("Update interval (seconds)", 0.1, 2.0, 0.5, 0.1)

    # ------------- Data ingest -------------
    # DEMO MODE: push one sample every run
    if st.session_state.demo_mode:
        add_demo_sample()
        # Add a demo emotion every ~30 new samples
        if len(st.session_state.buffer) % 30 == 0:
            append_sequential_demo_emotion()
    else:
        # LIVE MODE: pull all queued samples this run
        pulled = 0
        while True:
            reading = st.session_state.serial.get_latest()
            if reading is None:
                break
            st.session_state.buffer.append(reading)
            pulled += 1

        # Periodically analyze an emotion if Gemini available
        if st.session_state.analyzer and st.session_state.buffer and (len(st.session_state.buffer) % 40 == 0):
            latest = st.session_state.buffer[-1]
            result = st.session_state.analyzer.analyze_emotion(latest.channels)
            st.session_state.history.append(result)
            st.toast(f"🧠 Emotion: {result.emotion.title()} ({result.confidence:.0%})", icon="🧠")

    # ------------- UI: Plots & Panels -------------
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📈 Real-time EEG")
        fig = create_realtime_plot(st.session_state.buffer)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🧠 Current Emotion")
        if st.session_state.history:
            last = st.session_state.history[-1]
            emoji_map = {
                "happy": "😊", "sad": "😢", "anger": "😠",
                "relaxed": "😌", "interested": "🤔", "neutral": "😐"
            }
            st.markdown(
                f"<div style='text-align:center;font-size:64px;'>{emoji_map.get(last.emotion,'😐')}</div>",
                unsafe_allow_html=True
            )
            st.metric("Emotion", last.emotion.title(), f"{last.confidence:.0%} confidence")
            if last.reasoning:
                st.caption(last.reasoning)
        else:
            st.info("No emotion detected yet.")

        st.subheader("🧾 Latest Raw")
        if st.session_state.buffer:
            lr = st.session_state.buffer[-1]
            st.code(
                f"Time: {lr.timestamp.strftime('%H:%M:%S')}\n"
                f"Channels: {np.array(lr.channels).round(2).tolist()}\n"
                f"Raw: {lr.raw_data}"
            )
        else:
            st.write("—")

    st.subheader("📚 Emotion History (latest 12)")
    if st.session_state.history:
        df = pd.DataFrame([
            {
                "Time": x.timestamp.strftime("%H:%M:%S"),
                "Emotion": x.emotion.title(),
                "Confidence": f"{x.confidence:.0%}",
                
            }
            for x in st.session_state.history[-12:]
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.write("—")

    # ------------- Optional Continuous Update -------------
    # Bounded, non-infinite loop to keep UI lively without freezing
    if auto:
        # Do a tiny bounded refresh loop
        # (keeps the app responsive and avoids infinite blocking)
        time.sleep(interval)
        st.rerun()
    elif refresh_now:
        st.rerun()

# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    main()
