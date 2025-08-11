import streamlit as st
import serial
import threading
import time
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import google.generativeai as genai
from datetime import datetime, timedelta
import queue
import logging
from typing import List, Dict, Optional, Tuple
import serial.tools.list_ports
import asyncio
from dataclasses import dataclass
from collections import deque
import statistics
import os

# ############################################
# Full-featured, performance-optimized Streamlit
# EEG Emotion Analyzer (based on user's ref)
# ############################################

# -------------------------
# Configuration
# -------------------------
BAUD_RATE = 115200
MAX_DATA_POINTS = 1000            # circular buffer for plotting
EMOTION_CATEGORIES = ["happy", "sad", "anger", "relaxed", "interested", "neutral"]
CONFIDENCE_THRESHOLD = 0.6
ANALYSIS_INTERVAL_SECONDS = 3.0   # frequency of sending data to Gemini
PLOT_UPDATE_INTERVAL = 0.6       # seconds between UI refreshes
CSV_LOG_DIR = "./eeg_logs"
if not os.path.exists(CSV_LOG_DIR):
    os.makedirs(CSV_LOG_DIR, exist_ok=True)

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EEGApp")

# -------------------------
# Data classes
# -------------------------
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

# -------------------------
# Analyzer (Gemini integration)
# -------------------------
class EEGAnalyzer:
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.analysis_history = deque(maxlen=200)
        self._last_analysis_time = datetime.min

    def preprocess_eeg_data(self, eeg_values: List[float]) -> Dict:
        if not eeg_values or len(eeg_values) < 4:
            return {}
        features = {
            'mean': statistics.mean(eeg_values),
            'std': statistics.stdev(eeg_values) if len(eeg_values) > 1 else 0,
            'min': min(eeg_values),
            'max': max(eeg_values),
            'range': max(eeg_values) - min(eeg_values),
            'median': statistics.median(eeg_values),
        }
        features['high_frequency_activity'] = sum(abs(x) for x in eeg_values if abs(x) > features['mean'] + features['std'])
        features['stability'] = features['std'] / (abs(features['mean']) + 1)
        return features

    def _create_analysis_prompt(self, raw_data: List[float], features: Dict) -> str:
        return f"""
You are an expert EEG signal analyst. Analyze the following brain activity data and classify the primary emotion.

Raw EEG Data (channels snapshot): {raw_data}

Extracted Features:
- Mean amplitude: {features.get('mean', 0):.2f} μV
- Standard deviation: {features.get('std', 0):.2f} μV
- Signal range: {features.get('range', 0):.2f} μV
- Stability index: {features.get('stability', 0):.2f}
- High frequency activity: {features.get('high_frequency_activity', 0):.2f}

Classification Guidelines:
- Happy: Increased activity in frontal regions, positive amplitude patterns
- Sad: Reduced overall activity, negative amplitude bias
- Anger: High frequency spikes, increased variance
- Relaxed: Low variance, stable patterns, moderate amplitudes
- Interested: Moderate to high activity, balanced patterns
- Neutral: Baseline activity, moderate variance

Respond in this EXACT JSON format:
{
  "emotion": "one of: happy, sad, anger, relaxed, interested, neutral",
  "confidence": 0.85,
  "reasoning": "Brief explanation of the analysis"
}
"""

    def _parse_gemini_response(self, response_text: str) -> Dict:
        try:
            import re
            json_match = re.search(r'\{[^}]+\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
                emotion = result.get('emotion', 'neutral').lower()
                if emotion not in EMOTION_CATEGORIES:
                    emotion = 'neutral'
                confidence = float(result.get('confidence', 0.5))
                confidence = max(0.0, min(1.0, confidence))
                return {'emotion': emotion, 'confidence': confidence, 'reasoning': result.get('reasoning', '')}
        except Exception as e:
            logger.warning(f"Failed to parse Gemini response: {e}")
        response_lower = response_text.lower()
        for emotion in EMOTION_CATEGORIES:
            if emotion in response_lower:
                return {'emotion': emotion, 'confidence': 0.6, 'reasoning': 'Fallback text analysis'}
        return {'emotion': 'neutral', 'confidence': 0.3, 'reasoning': 'Unable to determine emotion'}

    def analyze_emotion(self, eeg_snapshot: List[float]) -> EmotionResult:
        now = datetime.now()
        # throttle requests
        if (now - self._last_analysis_time).total_seconds() < ANALYSIS_INTERVAL_SECONDS:
            return EmotionResult('neutral', 0.0, now, 'Throttled')

        features = self.preprocess_eeg_data(eeg_snapshot)
        if not features:
            return EmotionResult('neutral', 0.2, now, 'Insufficient data')

        prompt = self._create_analysis_prompt(eeg_snapshot, features)
        try:
            response = self.model.generate_content(prompt)
            parsed = self._parse_gemini_response(response.text.strip())
            result = EmotionResult(parsed['emotion'], parsed['confidence'], now, parsed.get('reasoning', ''))
            self.analysis_history.append(result)
            self._last_analysis_time = now
            return result
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            return EmotionResult('neutral', 0.2, now, f'Error: {e}')

# -------------------------
# Serial manager (high-performance)
# -------------------------
class SerialDataManager:
    def __init__(self, max_queue=2000):
        self.data_queue = queue.Queue(maxsize=max_queue)
        self.is_running = False
        self.serial_connection = None
        self.thread = None
        self.error_count = 0
        self.last_data_time = None

    def get_available_ports(self) -> List[str]:
        try:
            ports = serial.tools.list_ports.comports()
            return [port.device for port in ports]
        except Exception as e:
            logger.error(f"Error getting serial ports: {e}")
            return []

    def start_reading(self, port: str, baudrate: int = BAUD_RATE) -> bool:
        try:
            if self.is_running:
                self.stop_reading()
            self.serial_connection = serial.Serial(port=port, baudrate=baudrate, timeout=1)
            self.is_running = True
            self.error_count = 0
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()
            logger.info(f"Started reading from {port} at {baudrate} baud")
            return True
        except Exception as e:
            logger.error(f"Failed to start serial reading: {e}")
            return False

    def stop_reading(self):
        self.is_running = False
        try:
            if self.serial_connection and self.serial_connection.is_open:
                self.serial_connection.close()
        except Exception as e:
            logger.error(f"Error closing serial: {e}")
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
        logger.info("Serial reading stopped")

    def _read_loop(self):
        while self.is_running and self.serial_connection:
            try:
                if self.serial_connection.in_waiting > 0:
                    line = self.serial_connection.readline()
                    decoded_line = line.decode('utf-8', errors='ignore').strip()
                    if decoded_line:
                        self.last_data_time = datetime.now()
                        eeg_values = self._parse_eeg_line(decoded_line)
                        if eeg_values:
                            reading = EEGReading(self.last_data_time, eeg_values, decoded_line)
                            try:
                                self.data_queue.put_nowait(reading)
                            except queue.Full:
                                try:
                                    _ = self.data_queue.get_nowait()
                                    self.data_queue.put_nowait(reading)
                                except queue.Empty:
                                    pass
                else:
                    time.sleep(0.002)
            except Exception as e:
                self.error_count += 1
                logger.error(f"Serial read error: {e}")
                if self.error_count > 20:
                    logger.error("Too many serial errors, stopping")
                    break
                time.sleep(0.05)

    def _parse_eeg_line(self, line: str) -> Optional[List[float]]:
        try:
            parts = line.replace(',', ' ').split()
            values = []
            for part in parts:
                try:
                    v = float(part)
                    if -1000 <= v <= 1000:
                        values.append(v)
                except Exception:
                    continue
            return values if len(values) >= 4 else None
        except Exception as e:
            logger.warning(f"Failed to parse EEG line: {e}")
            return None

    def get_latest_readings_batch(self, max_items=200) -> List[EEGReading]:
        items = []
        while not self.data_queue.empty() and len(items) < max_items:
            try:
                items.append(self.data_queue.get_nowait())
            except queue.Empty:
                break
        return items

    def is_connected(self) -> bool:
        if not self.is_running or not self.serial_connection:
            return False
        if self.last_data_time:
            return (datetime.now() - self.last_data_time) < timedelta(seconds=5)
        return False

# -------------------------
# Plot helpers
# -------------------------

def create_realtime_plot(data_buffer: deque, channels=8) -> go.Figure:
    fig = make_subplots(rows=2, cols=4, subplot_titles=[f'Ch {i+1}' for i in range(8)],
                        vertical_spacing=0.06, horizontal_spacing=0.04)
    recent = list(data_buffer)[-200:]
    if not recent:
        return fig
    for i in range(channels):
        row = (i // 4) + 1
        col = (i % 4) + 1
        timestamps = [r.timestamp for r in recent]
        values = [r.channels[i] if i < len(r.channels) else 0 for r in recent]
        fig.add_trace(go.Scatter(x=timestamps, y=values, mode='lines', name=f'Ch{i+1}'), row=row, col=col)
        fig.update_yaxes(range=[-150, 150], row=row, col=col)
    fig.update_layout(height=600, showlegend=False, title_text='Real-time EEG (last ~200 samples)')
    return fig


def create_psd_plot(latest_channels: List[float]) -> go.Figure:
    # Quick FFT-based PSD for a single snapshot (approximative)
    fig = go.Figure()
    try:
        data = np.array(latest_channels)
        n = len(data)
        if n < 4:
            return fig
        # Window & FFT
        windowed = data - np.mean(data)
        freqs = np.fft.rfftfreq(n, d=1/256)  # assume 256Hz sampling (user should set correctly)
        psd = np.abs(np.fft.rfft(windowed))**2
        fig.add_trace(go.Bar(x=freqs, y=psd, name='PSD'))
        fig.update_layout(title='Approx. PSD (single-channel snapshot)', xaxis_title='Freq (Hz)', yaxis_title='Power')
    except Exception as e:
        logger.warning(f"PSD creation failed: {e}")
    return fig

# -------------------------
# CSV logging
# -------------------------

def append_to_csv(reading: EEGReading, filename: str):
    row = {'timestamp': reading.timestamp.isoformat()}
    for i, ch in enumerate(reading.channels):
        row[f'ch{i+1}'] = ch
    df = pd.DataFrame([row])
    header = not os.path.exists(filename)
    df.to_csv(filename, mode='a', header=header, index=False)

# -------------------------
# Streamlit app
# -------------------------

def main():
    st.set_page_config(page_title='EEG Emotion Analyzer (Fast)', layout='wide')
    st.title('🧠 EEG Emotion Analyzer — Full-featured + Fast')

    if 'serial_manager' not in st.session_state:
        st.session_state.serial_manager = SerialDataManager()
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'buffer' not in st.session_state:
        st.session_state.buffer = deque(maxlen=MAX_DATA_POINTS)
    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = []
    if 'demo_mode' not in st.session_state:
        st.session_state.demo_mode = False
    if 'csv_file' not in st.session_state:
        st.session_state.csv_file = os.path.join(CSV_LOG_DIR, f"eeg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    # Sidebar
    with st.sidebar:
        st.header('Configuration')
        api_key = st.text_input('Gemini API Key', type='password')
        if api_key and (st.session_state.analyzer is None or st.session_state.analyzer.api_key != api_key):
            st.session_state.analyzer = EEGAnalyzer(api_key)
            st.success('Gemini configured')

        ports = st.session_state.serial_manager.get_available_ports()
        if not ports:
            ports = ['Simulated (Demo)']
        port = st.selectbox('Serial Port', ports)
        baud = st.selectbox('Baud rate', [115200, 57600, 38400, 9600], index=0)

        start_col, stop_col = st.columns(2)
        with start_col:
            if st.button('Start'):
                if port == 'Simulated (Demo)':
                    st.session_state.demo_mode = True
                else:
                    ok = st.session_state.serial_manager.start_reading(port, baud)
                    if not ok:
                        st.error('Failed to open serial port')
        with stop_col:
            if st.button('Stop'):
                st.session_state.serial_manager.stop_reading()
                st.session_state.demo_mode = False

        st.checkbox('Enable CSV logging', value=True, key='enable_csv')
        st.markdown('---')
        connected = st.session_state.serial_manager.is_connected() or st.session_state.demo_mode
        st.markdown(f"**Status:** {'🟢 Connected' if connected else '🔴 Disconnected'}")
        st.metric('Buffered points', len(st.session_state.buffer))
        st.metric('Emotions logged', len(st.session_state.emotion_history))

    # Layout columns
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader('Real-time Signals')
        plot_area = st.empty()
        st.subheader('Power / Spectral View')
        psd_area = st.empty()
    with c2:
        st.subheader('Current Emotion')
        emotion_area = st.empty()
        st.subheader('Raw latest reading')
        raw_area = st.empty()

    history_area = st.empty()

    last_plot_time = datetime.min
    last_analysis_time = datetime.min

    try:
        while True:
            # 1) Gather data (batch for speed)
            if st.session_state.demo_mode:
                channels = [np.random.normal(0, 25) + np.sin(time.time()*0.5 + i) * 15 for i in range(8)]
                reading = EEGReading(datetime.now(), channels, ','.join(map(str, channels)))
                st.session_state.buffer.append(reading)
                if st.session_state.enable_csv:
                    append_to_csv(reading, st.session_state.csv_file)
            else:
                batch = st.session_state.serial_manager.get_latest_readings_batch(max_items=200)
                for r in batch:
                    st.session_state.buffer.append(r)
                    if st.session_state.enable_csv:
                        append_to_csv(r, st.session_state.csv_file)

            # 2) Periodic analysis (rate-limited)
            if st.session_state.analyzer and len(st.session_state.buffer) > 10:
                now = datetime.now()
                if (now - last_analysis_time).total_seconds() >= ANALYSIS_INTERVAL_SECONDS:
                    # Prepare snapshot: average across last N readings per channel
                    recent = list(st.session_state.buffer)[-120:]
                    # compute per-channel median to reduce noise
                    channels = []
                    for i in range(8):
                        vals = [r.channels[i] for r in recent if i < len(r.channels)]
                        if vals:
                            channels.append(float(np.median(vals)))
                        else:
                            channels.append(0.0)
                    # Run analysis in separate thread to avoid blocking UI
                    def do_analysis(snapshot):
                        res = st.session_state.analyzer.analyze_emotion(snapshot)
                        st.session_state.current_emotion = res
                        st.session_state.emotion_history.append(res)
                        logger.info(f"Analysis result: {res}")
                    threading.Thread(target=do_analysis, args=(channels,), daemon=True).start()
                    last_analysis_time = now

            # 3) Update UI plots at controlled rate
            now = datetime.now()
            if (now - last_plot_time).total_seconds() >= PLOT_UPDATE_INTERVAL:
                fig = create_realtime_plot(st.session_state.buffer)
                plot_area.plotly_chart(fig, use_container_width=True)

                # PSD for channel 1 (approx)
                latest = st.session_state.buffer[-1] if st.session_state.buffer else None
                if latest:
                    psd_fig = create_psd_plot(latest.channels[:128])
                    psd_area.plotly_chart(psd_fig, use_container_width=True)

                    raw_area.code(f"Time: {latest.timestamp.strftime('%H:%M:%S')}\nChannels: {latest.channels}\nRaw: {latest.raw_data}")

                # Emotion display
                current = getattr(st.session_state, 'current_emotion', EmotionResult('neutral', 0.0, datetime.now()))
                icons = {'happy':'😊','sad':'😢','anger':'😠','relaxed':'😌','interested':'🤔','neutral':'😐'}
                icon = icons.get(current.emotion, '😐')
                with emotion_area.container():
                    st.markdown(f"<div style='text-align:center;font-size:3.5rem'>{icon}</div>", unsafe_allow_html=True)
                    st.metric('Emotion', current.emotion.title(), f"{current.confidence:.1%} confidence")
                    if current.reasoning:
                        st.caption(current.reasoning)

                # Emotion history table
                if st.session_state.emotion_history:
                    hist_df = pd.DataFrame([{'time':r.timestamp.strftime('%H:%M:%S'),'emotion':r.emotion,'conf':r.confidence,'reason':r.reasoning} for r in st.session_state.emotion_history[-20:]])
                    history_area.dataframe(hist_df, use_container_width=True)

                last_plot_time = now

            # Small sleep to yield control
            time.sleep(0.08)

    except Exception as e:
        st.error(f"App error: {e}")
        logger.error(f"Main loop error: {e}")


if __name__ == '__main__':
    main()
