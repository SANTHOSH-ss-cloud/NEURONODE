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
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
BAUD_RATE = 115200
MAX_DATA_POINTS = 500
EMOTION_CATEGORIES = ["happy", "sad", "anger", "relaxed", "interested", "neutral"]
CONFIDENCE_THRESHOLD = 0.6

# Data structures
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

class EEGAnalyzer:
    """Enhanced EEG analyzer with improved Gemini AI integration"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        self.analysis_history = deque(maxlen=50)

    def preprocess_eeg_data(self, eeg_values: List[float]) -> Dict:
        """Preprocess EEG data to extract meaningful features"""
        if not eeg_values or len(eeg_values) < 4:
            return {}

        # Statistical features
        features = {
            'mean': statistics.mean(eeg_values),
            'std': statistics.stdev(eeg_values) if len(eeg_values) > 1 else 0,
            'min': min(eeg_values),
            'max': max(eeg_values),
            'range': max(eeg_values) - min(eeg_values),
            'median': statistics.median(eeg_values),
        }

        # Add frequency domain features (simplified)
        features['high_frequency_activity'] = sum(abs(x) for x in eeg_values if abs(x) > features['mean'] + features['std'])
        features['stability'] = features['std'] / (abs(features['mean']) + 1)

        return features

    async def analyze_emotion_advanced(self, eeg_data: List[float]) -> EmotionResult:
        """Advanced emotion analysis with feature extraction and contextual reasoning"""
        try:
            features = self.preprocess_eeg_data(eeg_data)
            if not features:
                return EmotionResult("neutral", 0.3, datetime.now(), "Insufficient data")

            # Create comprehensive prompt with features
            prompt = self._create_analysis_prompt(eeg_data, features)

            response = self.model.generate_content(prompt)
            result_text = response.text.strip()

            # Parse the structured response
            emotion_data = self._parse_gemini_response(result_text)

            return EmotionResult(
                emotion=emotion_data['emotion'],
                confidence=emotion_data['confidence'],
                timestamp=datetime.now(),
                reasoning=emotion_data.get('reasoning', '')
            )

        except Exception as e:
            logger.error(f"Emotion analysis error: {e}")
            return EmotionResult("neutral", 0.2, datetime.now(), f"Error: {str(e)}")

    def _create_analysis_prompt(self, raw_data: List[float], features: Dict) -> str:
        """Create a comprehensive prompt for Gemini AI"""
        return f"""
        You are an expert EEG signal analyst. Analyze the following brain activity data and classify the primary emotion.

        Raw EEG Data (8 channels): {raw_data}

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
        {{
            "emotion": "one of: happy, sad, anger, relaxed, interested, neutral",
            "confidence": 0.85,
            "reasoning": "Brief explanation of the analysis"
        }}
        """

    def _parse_gemini_response(self, response_text: str) -> Dict:
        """Parse Gemini's response and extract emotion data"""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'{{[^"]+}}', response_text)
            if json_match:
                result = json.loads(json_match.group())

                # Validate emotion
                emotion = result.get('emotion', 'neutral').lower()
                if emotion not in EMOTION_CATEGORIES:
                    emotion = 'neutral'

                # Validate confidence
                confidence = float(result.get('confidence', 0.5))
                confidence = max(0.0, min(1.0, confidence))

                return {
                    'emotion': emotion,
                    'confidence': confidence,
                    'reasoning': result.get('reasoning', '')
                }
        except Exception as e:
            logger.warning(f"Failed to parse Gemini response: {e}")

        # Fallback parsing
        response_lower = response_text.lower()
        for emotion in EMOTION_CATEGORIES:
            if emotion in response_lower:
                return {
                    'emotion': emotion,
                    'confidence': 0.6,
                    'reasoning': 'Fallback text analysis'
                }

        return {
            'emotion': 'neutral',
            'confidence': 0.3,
            'reasoning': 'Unable to determine emotion'
        }

class SerialDataManager:
    """Enhanced serial data management with buffering and error handling"""

    def __init__(self):
        self.data_queue = queue.Queue(maxsize=1000)
        self.is_running = False
        self.serial_connection = None
        self.thread = None
        self.error_count = 0
        self.last_data_time = None

    def get_available_ports(self) -> List[str]:
        """Get list of available serial ports"""
        try:
            ports = serial.tools.list_ports.comports()
            return [port.device for port in ports]
        except Exception as e:
            logger.error(f"Error getting serial ports: {e}")
            return []

    def start_reading(self, port: str, baudrate: int = BAUD_RATE) -> bool:
        """Start reading data from serial port"""
        try:
            if self.is_running:
                self.stop_reading()

            self.serial_connection = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=1,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
            )

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
        """Stop reading data from serial port"""
        self.is_running = False

        if self.serial_connection:
            try:
                self.serial_connection.close()
            except Exception as e:
                logger.error(f"Error closing serial connection: {e}")

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

        logger.info("Serial reading stopped")

    def _read_loop(self):
        """Main serial reading loop"""
        while self.is_running and self.serial_connection:
            try:
                if self.serial_connection.in_waiting > 0:
                    line = self.serial_connection.readline()
                    decoded_line = line.decode('utf-8', errors='ignore').strip()

                    if decoded_line:
                        self.last_data_time = datetime.now()

                        # Parse EEG data
                        eeg_values = self._parse_eeg_line(decoded_line)
                        if eeg_values:
                            reading = EEGReading(
                                timestamp=self.last_data_time,
                                channels=eeg_values,
                                raw_data=decoded_line
                            )

                            try:
                                self.data_queue.put_nowait(reading)
                                self.error_count = 0
                            except queue.Full:
                                # Remove oldest item and add new one
                                try:
                                    self.data_queue.get_nowait()
                                    self.data_queue.put_nowait(reading)
                                except queue.Empty:
                                    pass

                time.sleep(0.01)  # Small delay to prevent CPU overload

            except Exception as e:
                self.error_count += 1
                logger.error(f"Serial read error: {e}")

                if self.error_count > 10:
                    logger.error("Too many serial errors, stopping")
                    break

                time.sleep(0.1)

    def _parse_eeg_line(self, line: str) -> Optional[List[float]]:
        """Parse a line of EEG data"""
        try:
            # Handle different formats: comma-separated, space-separated, etc.
            values = []

            # Split by comma or space
            parts = line.replace(',', ' ').split()

            for part in parts:
                try:
                    value = float(part)
                    # Basic validation for EEG values (typically -100 to +100 μV)
                    if -1000 <= value <= 1000:
                        values.append(value)
                except ValueError:
                    continue

            return values if len(values) >= 4 else None

        except Exception as e:
            logger.warning(f"Failed to parse EEG line: {line} - {e}")
            return None

    def get_latest_reading(self) -> Optional[EEGReading]:
        """Get the latest EEG reading"""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None

    def is_connected(self) -> bool:
        """Check if serial connection is active"""
        if not self.is_running or not self.serial_connection:
            return False

        # Check if we've received data recently
        if self.last_data_time:
            time_since_last = datetime.now() - self.last_data_time
            return time_since_last < timedelta(seconds=5)

        return False

def create_realtime_plot(data_buffer: deque) -> go.Figure:
    """Create real-time EEG plot"""
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=[f'Channel {i+1}' for i in range(8)],
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )

    if not data_buffer:
        return fig

    # Get recent data
    recent_data = list(data_buffer)[-100:]  # Last 100 points

    for i in range(8):
        row = (i // 4) + 1
        col = (i % 4) + 1

        if recent_data:
            timestamps = [reading.timestamp for reading in recent_data]
            values = [reading.channels[i] if i < len(reading.channels) else 0 for reading in recent_data]

            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=values,
                    mode='lines',
                    name=f'Ch{i+1}',
                    line=dict(color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)])
                ),
                row=row, col=col
            )

    fig.update_layout(
        height=600,
        title="Real-time EEG Signals",
        showlegend=False
    )

    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Amplitude (μV)")

    return fig

def create_emotion_history_plot(emotion_history: List[EmotionResult]) -> go.Figure:
    """Create emotion history visualization"""
    if not emotion_history:
        return go.Figure()

    df = pd.DataFrame([
        {
            'timestamp': result.timestamp,
            'emotion': result.emotion,
            'confidence': result.confidence
        }
        for result in emotion_history
    ])

    fig = px.timeline(
        df,
        x_start='timestamp',
        x_end='timestamp',
        y='emotion',
        color='confidence',
        title='Emotion History',
        color_continuous_scale='viridis'
    )

    fig.update_layout(height=300)
    return fig

def main():
    """Main Streamlit application"""

    # Page configuration
    st.set_page_config(
        page_title="EEG Emotion Analyzer",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .status-connected {
        color: #28a745;
        font-weight: bold;
    }
    .status-disconnected {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="main-header">
        <h1> EEG Emotion Analyzer</h1>
        <p>Advanced real-time emotion detection using ESP32 and Gemini AI</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'serial_manager' not in st.session_state:
        st.session_state.serial_manager = SerialDataManager()

    if 'analyzer' not in st.session_state:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            st.session_state.analyzer = EEGAnalyzer(api_key)
            st.success("✅ Gemini AI configured")
        else:
            st.session_state.analyzer = None
            st.error("❌ Gemini API key not found in .env file")

    if 'data_buffer' not in st.session_state:
        st.session_state.data_buffer = deque(maxlen=MAX_DATA_POINTS)

    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = []

    if 'current_emotion' not in st.session_state:
        st.session_state.current_emotion = EmotionResult("neutral", 0.0, datetime.now())

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Serial port configuration
        st.subheader("Serial Port Setup")

        available_ports = st.session_state.serial_manager.get_available_ports()
        if not available_ports:
            st.warning("⚠️ No serial ports detected")
            available_ports = ["Simulated Port (Demo)"]

        selected_port = st.selectbox("Select Port", available_ports)
        baud_rate = st.selectbox("Baud Rate", [115200, 9600, 57600, 38400], index=0)

        # Control buttons
        col1, col2 = st.columns(2)

        with col1:
            if st.button(" Start", use_container_width=True):
                if not st.session_state.analyzer:
                    st.error("❌ Please configure the Gemini API key in the .env file")
                elif selected_port == "Simulated Port (Demo)":
                    st.info(" Starting demo mode...")
                    st.session_state.demo_mode = True
                else:
                    success = st.session_state.serial_manager.start_reading(selected_port, baud_rate)
                    if success:
                        st.success(f"✅ Connected to {selected_port}")
                    else:
                        st.error("❌ Failed to connect")

        with col2:
            if st.button("⏹️ Stop", use_container_width=True):
                st.session_state.serial_manager.stop_reading()
                st.session_state.demo_mode = False
                st.info(" Stopped")

        # Connection status
        is_connected = st.session_state.serial_manager.is_connected() or getattr(st.session_state, 'demo_mode', False)
        status_class = "status-connected" if is_connected else "status-disconnected"
        status_text = " Connected" if is_connected else " Disconnected"

        st.markdown(f'<p class="{status_class}">{status_text}</p>', unsafe_allow_html=True)

        # Statistics
        st.subheader(" Statistics")
        st.metric("Data Points", len(st.session_state.data_buffer))
        st.metric("Emotions Detected", len(st.session_state.emotion_history))

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(" Real-time EEG Signals")
        plot_container = st.empty()

    with col2:
        st.subheader(" Current Emotion")
        emotion_container = st.empty()

        st.subheader(" Raw Data")
        raw_data_container = st.empty()

    # Emotion history
    st.subheader(" Emotion History")
    history_container = st.empty()

    # Main processing loop
    if st.session_state.serial_manager.is_running or getattr(st.session_state, 'demo_mode', False):
        while True:
            try:
                # Get latest data
                if getattr(st.session_state, 'demo_mode', False):
                    # Demo mode - generate simulated data
                    channels = [np.random.normal(0, 20) + np.sin(time.time() * 0.1 + i) * 10 for i in range(8)]
                    reading = EEGReading(datetime.now(), channels, f"Demo: {','.join(map(str, channels))}")
                    st.session_state.data_buffer.append(reading)
                else:
                    reading = st.session_state.serial_manager.get_latest_reading()
                    if reading:
                        st.session_state.data_buffer.append(reading)

                # Update plots
                if st.session_state.data_buffer:
                    fig = create_realtime_plot(st.session_state.data_buffer)
                    plot_container.plotly_chart(fig, use_container_width=True)

                    # Show raw data
                    latest_reading = st.session_state.data_buffer[-1]
                    raw_data_container.code(
                        f"Timestamp: {latest_reading.timestamp.strftime('%H:%M:%S')}\n"
                        f"Channels: {latest_reading.channels}\n"
                        f"Raw: {latest_reading.raw_data}"
                    )

                    # Analyze emotion (every few seconds)
                    if (len(st.session_state.data_buffer) % 30 == 0 and
                        st.session_state.analyzer and
                        len(latest_reading.channels) >= 4):

                        with st.spinner(" Analyzing emotion..."):
                            emotion_result = asyncio.run(
                                st.session_state.analyzer.analyze_emotion_advanced(latest_reading.channels)
                            )

                            st.session_state.current_emotion = emotion_result
                            st.session_state.emotion_history.append(emotion_result)

                            # Keep only recent history
                            if len(st.session_state.emotion_history) > 20:
                                st.session_state.emotion_history = st.session_state.emotion_history[-20:]

                # Update emotion display
                current = st.session_state.current_emotion
                emotion_icons = {
                    'happy': '', 'sad': '', 'anger': '',
                    'relaxed': '', 'interested': '', 'neutral': ''
                }

                with emotion_container.container():
                    icon = emotion_icons.get(current.emotion, '')
                    st.markdown(f"<div style='text-align: center; font-size: 4rem;'>{icon}</div>",
                               unsafe_allow_html=True)
                    st.metric(
                        "Emotion",
                        current.emotion.title(),
                        f"{current.confidence:.1%} confidence"
                    )

                    if current.reasoning:
                        st.caption(f" {current.reasoning}")

                # Update history
                if st.session_state.emotion_history:
                    history_df = pd.DataFrame([
                        {
                            'Time': result.timestamp.strftime('%H:%M:%S'),
                            'Emotion': result.emotion.title(),
                            'Confidence': f"{result.confidence:.1%}",
                            'Reasoning': result.reasoning[:50] + "..." if len(result.reasoning) > 50 else result.reasoning
                        }
                        for result in st.session_state.emotion_history[-10:]  # Last 10
                    ])
                    history_container.dataframe(history_df, use_container_width=True)

                time.sleep(0.5)  # Update rate

            except Exception as e:
                st.error(f"Application error: {e}")
                logger.error(f"Main loop error: {e}")
                time.sleep(1)

if __name__ == "__main__":
    main()