import os
import time
import json
import queue
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
from typing import List, Optional
import random
import asyncio
import platform

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import serial
import serial.tools.list_ports
import google.generativeai as genai
from dotenv import load_dotenv

# Bluetooth imports
try:
    import bluetooth
    BLUETOOTH_AVAILABLE = True
except ImportError:
    BLUETOOTH_AVAILABLE = False
    st.warning("⚠️ PyBluez not installed. Install with: pip install pybluez")

try:
    import bleak
    BLEAK_AVAILABLE = True
except ImportError:
    BLEAK_AVAILABLE = False
    st.warning("⚠️ Bleak not installed. Install with: pip install bleak")

# ---------------------------
# Config & Setup
# ---------------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")  # from .env

BAUD_RATE = 115200
MAX_DATA_POINTS = 500
EMOTION_SEQ = ["happy", "sad", "anger", "relaxed", "interested", "neutral"]

# Bluetooth UUIDs for ESP32 (modify these based on your ESP32 implementation)
BLE_SERVICE_UUID = "12345678-1234-1234-1234-123456789abc"
BLE_CHARACTERISTIC_UUID = "87654321-4321-4321-4321-cba987654321"

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

@dataclass
class BluetoothDevice:
    name: str
    address: str
    device_type: str  # "classic" or "ble"

# ---------------------------
# Gemini Analyzer (sync)
# ---------------------------
class EEGAnalyzer:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    def analyze_emotion(self, eeg_data: List[float]) -> EmotionResult:
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
                text_low = text.lower()
                for emo in EMOTION_SEQ:
                    if emo in text_low:
                        return EmotionResult(emo, 0.6, datetime.now(), "Heuristic parse")
                return EmotionResult("neutral", 0.3, datetime.now(), "Unstructured model response")
        except Exception as e:
            return EmotionResult("neutral", 0.2, datetime.now(), f"Gemini error: {e}")

# ---------------------------
# Bluetooth Manager (Classic)
# ---------------------------
class BluetoothClassicManager:
    def __init__(self):
        self.data_queue = queue.Queue(maxsize=1000)
        self.is_running = False
        self.socket = None
        self.thread: Optional[threading.Thread] = None
        self.last_data_time: Optional[datetime] = None
        self.error_count = 0

    def scan_devices(self) -> List[BluetoothDevice]:
        """Scan for nearby Bluetooth Classic devices"""
        if not BLUETOOTH_AVAILABLE:
            return []
        
        try:
            devices = []
            nearby_devices = bluetooth.discover_devices(lookup_names=True, duration=8)
            for addr, name in nearby_devices:
                devices.append(BluetoothDevice(name or "Unknown", addr, "classic"))
            return devices
        except Exception as e:
            st.error(f"Bluetooth scan error: {e}")
            return []

    def connect(self, address: str) -> bool:
        try:
            self.disconnect()
            self.socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            self.socket.connect((address, 1))  # Channel 1 for RFCOMM
            self.socket.settimeout(1.0)
            
            self.is_running = True
            self.error_count = 0
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
            return True
        except Exception as e:
            st.error(f"❌ Failed to connect to {address}: {e}")
            return False

    def disconnect(self):
        self.is_running = False
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None

    def _read_loop(self):
        while self.is_running and self.socket:
            try:
                data = self.socket.recv(1024)
                decoded = data.decode("utf-8", errors="ignore").strip()
                if decoded:
                    self.last_data_time = datetime.now()
                    # Parse the data similar to serial
                    lines = decoded.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line:
                            parts = line.replace(",", " ").split()
                            vals = []
                            for p in parts:
                                try:
                                    vals.append(float(p))
                                except ValueError:
                                    continue
                            if vals:
                                if len(vals) < 8:
                                    vals = vals + [0.0] * (8 - len(vals))
                                else:
                                    vals = vals[:8]
                                reading = EEGReading(self.last_data_time, vals, line)
                                try:
                                    self.data_queue.put_nowait(reading)
                                except queue.Full:
                                    try:
                                        _ = self.data_queue.get_nowait()
                                        self.data_queue.put_nowait(reading)
                                    except queue.Empty:
                                        pass
                time.sleep(0.01)
            except Exception:
                self.error_count += 1
                if self.error_count > 25:
                    break
                time.sleep(0.1)

    def get_latest(self) -> Optional[EEGReading]:
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None

    def connected(self) -> bool:
        if not self.is_running or not self.socket:
            return False
        if self.last_data_time is None:
            return True
        return (datetime.now() - self.last_data_time) < timedelta(seconds=5)

    def send_emotion_result(self, emotion_result: EmotionResult):
        """Send emotion result back to ESP32 via Bluetooth"""
        if self.socket and self.is_running:
            try:
                message = f"EMOTION:{emotion_result.emotion},{emotion_result.confidence:.2f}\n"
                self.socket.send(message.encode())
            except Exception as e:
                st.error(f"Failed to send emotion result: {e}")

# ---------------------------
# BLE Manager (using bleak)
# ---------------------------
class BLEManager:
    def __init__(self):
        self.data_queue = queue.Queue(maxsize=1000)
        self.is_running = False
        self.client = None
        self.thread: Optional[threading.Thread] = None
        self.last_data_time: Optional[datetime] = None
        self.error_count = 0
        self.loop = None

    async def scan_devices_async(self) -> List[BluetoothDevice]:
        """Scan for BLE devices"""
        try:
            scanner = bleak.BleakScanner()
            devices = await scanner.discover(timeout=10.0)
            ble_devices = []
            for device in devices:
                name = device.name or "Unknown BLE Device"
                if "ESP32" in name or "EEG" in name or device.name:  # Filter for relevant devices
                    ble_devices.append(BluetoothDevice(name, device.address, "ble"))
            return ble_devices
        except Exception as e:
            st.error(f"BLE scan error: {e}")
            return []

    def scan_devices(self) -> List[BluetoothDevice]:
        """Synchronous wrapper for BLE scanning"""
        if not BLEAK_AVAILABLE:
            return []
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            devices = loop.run_until_complete(self.scan_devices_async())
            loop.close()
            return devices
        except Exception:
            return []

    def connect(self, address: str) -> bool:
        try:
            self.disconnect()
            self.is_running = True
            self.error_count = 0
            self.thread = threading.Thread(target=self._ble_thread, args=(address,), daemon=True)
            self.thread.start()
            return True
        except Exception as e:
            st.error(f"❌ Failed to start BLE connection to {address}: {e}")
            return False

    def _ble_thread(self, address: str):
        """Run BLE connection in separate thread with its own event loop"""
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._ble_connect_and_read(address))
        except Exception as e:
            st.error(f"BLE thread error: {e}")
        finally:
            if self.loop:
                self.loop.close()

    async def _ble_connect_and_read(self, address: str):
        """Connect to BLE device and read data"""
        try:
            self.client = bleak.BleakClient(address)
            await self.client.connect()
            
            # Start notifications for the characteristic
            await self.client.start_notify(BLE_CHARACTERISTIC_UUID, self._notification_handler)
            
            # Keep connection alive
            while self.is_running:
                if not self.client.is_connected:
                    break
                await asyncio.sleep(0.1)
                
        except Exception as e:
            st.error(f"BLE connection error: {e}")
        finally:
            if self.client:
                try:
                    await self.client.disconnect()
                except Exception:
                    pass

    def _notification_handler(self, sender, data):
        """Handle incoming BLE notifications"""
        try:
            decoded = data.decode("utf-8", errors="ignore").strip()
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
        except Exception:
            self.error_count += 1

    def get_latest(self) -> Optional[EEGReading]:
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None

    def connected(self) -> bool:
        if not self.is_running or not self.client:
            return False
        if self.last_data_time is None:
            return True
        return (datetime.now() - self.last_data_time) < timedelta(seconds=5)

    def disconnect(self):
        self.is_running = False
        # The async cleanup will happen in the thread

    async def send_emotion_result_async(self, emotion_result: EmotionResult):
        """Send emotion result back to BLE device"""
        if self.client and self.client.is_connected:
            try:
                message = f"EMOTION:{emotion_result.emotion},{emotion_result.confidence:.2f}"
                await self.client.write_gatt_char(BLE_CHARACTERISTIC_UUID, message.encode())
            except Exception as e:
                st.error(f"Failed to send BLE emotion result: {e}")

    def send_emotion_result(self, emotion_result: EmotionResult):
        """Synchronous wrapper for sending emotion result"""
        if self.loop and self.is_running:
            try:
                asyncio.run_coroutine_threadsafe(
                    self.send_emotion_result_async(emotion_result), 
                    self.loop
                )
            except Exception as e:
                st.error(f"Failed to send emotion result: {e}")

# ---------------------------
# Serial Data Manager (Original)
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
        if self.thread and self.thread.is_alive():
            pass

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
            return True
        return (datetime.now() - self.last_data_time) < timedelta(seconds=5)

    def send_emotion_result(self, emotion_result: EmotionResult):
        """Send emotion result back to ESP32 via Serial"""
        if self.serial_connection and self.is_running:
            try:
                message = f"EMOTION:{emotion_result.emotion},{emotion_result.confidence:.2f}\n"
                self.serial_connection.write(message.encode())
            except Exception as e:
                st.error(f"Failed to send emotion result: {e}")

# ---------------------------
# Connection Manager (Unified)
# ---------------------------
class ConnectionManager:
    def __init__(self):
        self.serial_manager = SerialDataManager()
        self.bt_classic_manager = BluetoothClassicManager() if BLUETOOTH_AVAILABLE else None
        self.ble_manager = BLEManager() if BLEAK_AVAILABLE else None
        self.current_mode = "demo"  # "demo", "serial", "bt_classic", "ble"
        self.current_device = None

    def get_available_connections(self) -> dict:
        connections = {"Demo Mode": "demo"}
        
        # Serial ports
        ports = self.serial_manager.get_available_ports()
        for port in ports:
            connections[f"Serial: {port}"] = f"serial:{port}"
        
        # Add Bluetooth options (but don't scan immediately for performance)
        if BLUETOOTH_AVAILABLE:
            connections["Scan Bluetooth Classic"] = "scan_bt_classic"
        if BLEAK_AVAILABLE:
            connections["Scan BLE Devices"] = "scan_ble"
            
        return connections

    def connect(self, connection_string: str) -> bool:
        self.disconnect_all()
        
        if connection_string == "demo":
            self.current_mode = "demo"
            return True
        elif connection_string.startswith("serial:"):
            port = connection_string.split(":", 1)[1]
            if self.serial_manager.start(port):
                self.current_mode = "serial"
                self.current_device = port
                return True
        elif connection_string.startswith("bt_classic:"):
            if self.bt_classic_manager:
                address = connection_string.split(":", 1)[1]
                if self.bt_classic_manager.connect(address):
                    self.current_mode = "bt_classic"
                    self.current_device = address
                    return True
        elif connection_string.startswith("ble:"):
            if self.ble_manager:
                address = connection_string.split(":", 1)[1]
                if self.ble_manager.connect(address):
                    self.current_mode = "ble"
                    self.current_device = address
                    return True
        
        self.current_mode = "demo"
        return False

    def disconnect_all(self):
        self.serial_manager.stop()
        if self.bt_classic_manager:
            self.bt_classic_manager.disconnect()
        if self.ble_manager:
            self.ble_manager.disconnect()

    def get_latest(self) -> Optional[EEGReading]:
        if self.current_mode == "serial":
            return self.serial_manager.get_latest()
        elif self.current_mode == "bt_classic" and self.bt_classic_manager:
            return self.bt_classic_manager.get_latest()
        elif self.current_mode == "ble" and self.ble_manager:
            return self.ble_manager.get_latest()
        return None

    def connected(self) -> bool:
        if self.current_mode == "demo":
            return True
        elif self.current_mode == "serial":
            return self.serial_manager.connected()
        elif self.current_mode == "bt_classic" and self.bt_classic_manager:
            return self.bt_classic_manager.connected()
        elif self.current_mode == "ble" and self.ble_manager:
            return self.ble_manager.connected()
        return False

    def send_emotion_result(self, emotion_result: EmotionResult):
        """Send emotion result back to device"""
        if self.current_mode == "serial":
            self.serial_manager.send_emotion_result(emotion_result)
        elif self.current_mode == "bt_classic" and self.bt_classic_manager:
            self.bt_classic_manager.send_emotion_result(emotion_result)
        elif self.current_mode == "ble" and self.ble_manager:
            self.ble_manager.send_emotion_result(emotion_result)

    def get_status(self) -> str:
        if self.current_mode == "demo":
            return "🟣 Demo Mode"
        elif self.current_mode == "serial":
            status = "✅ Connected" if self.connected() else "❌ Disconnected"
            return f"🔌 Serial ({self.current_device}): {status}"
        elif self.current_mode == "bt_classic":
            status = "✅ Connected" if self.connected() else "❌ Disconnected"
            return f"📶 Bluetooth Classic ({self.current_device}): {status}"
        elif self.current_mode == "ble":
            status = "✅ Connected" if self.connected() else "❌ Disconnected"
            return f"📡 BLE ({self.current_device}): {status}"
        return "❓ Unknown"

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
    if "connection_manager" not in st.session_state:
        st.session_state.connection_manager = ConnectionManager()
    if "buffer" not in st.session_state:
        st.session_state.buffer = deque(maxlen=MAX_DATA_POINTS)
    if "history" not in st.session_state:
        st.session_state.history: List[EmotionResult] = []
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = EEGAnalyzer(API_KEY) if API_KEY else None
    if "bt_devices" not in st.session_state:
        st.session_state.bt_devices: List[BluetoothDevice] = []
    if "scanning_bt" not in st.session_state:
        st.session_state.scanning_bt = False

# --- DEMO Helpers ---
def add_demo_sample():
    t = time.time()
    channels = [np.sin(0.9 * t + i * 0.6) * 18 + np.cos(0.15 * t + i) * 7 + np.random.normal(0, 1.5)
                for i in range(8)]
    reading = EEGReading(datetime.now(), channels, "demo," + ",".join(f"{v:.2f}" for v in channels))
    st.session_state.buffer.append(reading)

def append_random_demo_emotion():
    last_emo = st.session_state.history[-1].emotion if st.session_state.history else None
    choices = [e for e in EMOTION_SEQ if e != last_emo]
    emo = random.choice(choices)
    result = EmotionResult(emo, 0.95, datetime.now(), "Demo mode")
    st.session_state.history.append(result)
    # Send result back to device (demo mode won't actually send)
    st.session_state.connection_manager.send_emotion_result(result)

# --- UI Header ---
def app_header():
    st.markdown("""
    <div style="text-align:center;padding:16px;border-radius:12px;
         background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);color:#fff;">
      <h2 style="margin:0;">EEG Emotion Analyzer</h2>
      <p style="margin:6px 0 0;">Real-time dashboard with Serial, Bluetooth Classic, BLE + Gemini AI</p>
    </div>
    """, unsafe_allow_html=True)

# --- Bluetooth Device Scanner ---
def bluetooth_device_selector():
    st.subheader("📶 Bluetooth Devices")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Scan Classic", disabled=not BLUETOOTH_AVAILABLE):
            if BLUETOOTH_AVAILABLE:
                with st.spinner("Scanning Bluetooth Classic devices..."):
                    devices = st.session_state.connection_manager.bt_classic_manager.scan_devices()
                    st.session_state.bt_devices = [d for d in st.session_state.bt_devices if d.device_type != "classic"] + devices
                    st.success(f"Found {len(devices)} Classic devices")
            else:
                st.error("PyBluez not available")
    
    with col2:
        if st.button("🔍 Scan BLE", disabled=not BLEAK_AVAILABLE):
            if BLEAK_AVAILABLE:
                with st.spinner("Scanning BLE devices..."):
                    devices = st.session_state.connection_manager.ble_manager.scan_devices()
                    st.session_state.bt_devices = [d for d in st.session_state.bt_devices if d.device_type != "ble"] + devices
                    st.success(f"Found {len(devices)} BLE devices")
            else:
                st.error("Bleak not available")

    if st.session_state.bt_devices:
        st.write("**Available Devices:**")
        for device in st.session_state.bt_devices:
            icon = "📶" if device.device_type == "classic" else "📡"
            col_dev, col_btn = st.columns([3, 1])
            with col_dev:
                st.write(f"{icon} **{device.name}** ({device.address})")
            with col_btn:
                key = f"{device.device_type}:{device.address}"
                if st.button("Connect", key=f"connect_{key}"):
                    if st.session_state.connection_manager.connect(key):
                        st.success(f"✅ Connected to {device.name}")
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to connect to {device.name}")

# --- MAIN APP ---
def main():
    st.set_page_config(page_title="EEG Emotion Analyzer", layout="wide")
    init_state()
    app_header()

    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Status
        if st.session_state.analyzer:
            st.success("✅ Gemini API: Ready")
        else:
            st.warning("⚠️ Gemini API: Not configured (set GEMINI_API_KEY in .env)")

        # Connection Type Selection
        st.subheader("🔗 Connection Type")
        connection_type = st.radio(
            "Choose connection method:",
            ["Demo Mode", "Serial (USB)", "Bluetooth"],
            index=0
        )

        if connection_type == "Serial (USB)":
            ports = st.session_state.connection_manager.serial_manager.get_available_ports()
            if ports:
                port = st.selectbox("Serial Port", ports)
                baud = st.selectbox("Baud Rate", [115200, 57600, 38400, 9600], index=0)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("▶ Connect Serial", use_container_width=True):
                        if st.session_state.connection_manager.connect(f"serial:{port}"):
                            st.toast(f"🔌 Connected to {port}", icon="🔌")
                        else:
                            st.toast("❌ Connection failed", icon="❌")
                        st.rerun()
                with col_b:
                    if st.button("⏹ Disconnect", use_container_width=True):
                        # Stop any active connections
                        st.session_state.connection_manager.disconnect_all()
                        st.toast("⏹ Disconnected", icon="⏹")
                        st.rerun()