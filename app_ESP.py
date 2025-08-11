import streamlit as st
import serial
import serial.tools.list_ports
import threading
import time
import queue
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import google.generativeai as genai
from datetime import datetime

# --- Config ---
BAUD_RATE = 115200
MAX_BUFFER_LEN = 500
EMOTION_CHECK_INTERVAL = 10  # seconds
GAIN = 6.0
VREF = 2.42
MAX_ADC_VALUE = 2**23 - 1  # 24-bit signed max positive

# Initialize global state
data_queue = queue.Queue()
data_buffer = []
emotion_history = []

# Gemini AI setup (must be set via UI)
gemini_model = None

def raw_to_voltage(raw):
    # Convert 24-bit signed raw ADC value to voltage in μV
    # Formula: voltage = (raw / MAX_ADC) * (Vref / Gain)
    return (raw / MAX_ADC_VALUE) * (VREF / GAIN) * 1e6  # in microvolts

def serial_reader(port_name):
    """Thread to read serial data continuously"""
    try:
        ser = serial.Serial(port_name, BAUD_RATE, timeout=1)
        st.session_state['serial_connection'] = ser
    except Exception as e:
        st.error(f"Failed to open serial port {port_name}: {e}")
        return

    while True:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line:
                try:
                    raw_val = int(line)
                    data_queue.put((datetime.now(), raw_val))
                except ValueError:
                    pass  # ignore bad lines
        except Exception as e:
            st.error(f"Serial read error: {e}")
            break

def analyze_emotion(voltage_data):
    """Call Gemini AI to analyze emotion from EEG voltage list"""
    global gemini_model
    if gemini_model is None:
        return {"emotion": "neutral", "confidence": 0.0, "reasoning": "No API key set"}

    # Prepare prompt with basic stats
    mean_val = np.mean(voltage_data)
    std_val = np.std(voltage_data)
    prompt = f"""
    You are an EEG expert analyzing brainwave data. Given this voltage data (μV): {voltage_data[:20]}...
    The data has mean {mean_val:.2f} μV and std {std_val:.2f} μV.
    Classify the primary emotion from this data in JSON format like:
    {{
      "emotion": "happy|sad|anger|relaxed|interested|neutral",
      "confidence": 0.85,
      "reasoning": "short explanation"
    }}
    """

    response = gemini_model.generate_content(prompt)
    text = response.text.lower()

    # Simple parsing, fallback neutral
    for emotion in ["happy", "sad", "anger", "relaxed", "interested", "neutral"]:
        if emotion in text:
            confidence = 0.6
            try:
                import re, json
                json_text = re.search(r"\{.*?\}", text)
                if json_text:
                    data = json.loads(json_text.group())
                    confidence = data.get("confidence", 0.6)
            except Exception:
                pass

            return {"emotion": emotion, "confidence": confidence, "reasoning": "Analyzed by Gemini AI"}

    return {"emotion": "neutral", "confidence": 0.0, "reasoning": "Unable to determine emotion"}

# Streamlit app starts here
def main():
    st.set_page_config(page_title="EEG Emotion Analyzer", layout="wide")

    st.title("🧠 EEG Emotion Analyzer")

    # Sidebar for serial port and API key
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("Google Gemini API Key", type="password")
        if api_key:
            global gemini_model
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel('gemini-pro')
            st.success("Gemini AI configured")

        ports = serial.tools.list_ports.comports()
        port_list = [p.device for p in ports]
        if not port_list:
            st.warning("No serial ports found!")
        port = st.selectbox("Select Serial Port", port_list)

        if st.button("Connect to ESP32"):
            if "serial_thread" not in st.session_state or not st.session_state.serial_thread.is_alive():
                st.session_state.serial_thread = threading.Thread(target=serial_reader, args=(port,), daemon=True)
                st.session_state.serial_thread.start()
                st.success(f"Started reading from {port}")

    # Data processing & visualization
    placeholder_plot = st.empty()
    placeholder_emotion = st.empty()
    placeholder_raw = st.empty()

    # Read from queue without blocking
    while not data_queue.empty():
        timestamp, raw_val = data_queue.get()
        voltage = raw_to_voltage(raw_val)
        data_buffer.append((timestamp, voltage))
        if len(data_buffer) > MAX_BUFFER_LEN:
            data_buffer.pop(0)

    if len(data_buffer) < 10:
        st.info("Waiting for EEG data...")
        return

    # Prepare plot
    times = [t for t, v in data_buffer]
    volts = [v for t, v in data_buffer]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=volts, mode="lines+markers", name="EEG Voltage (μV)"))
    fig.update_layout(title="Real-time EEG Signal", xaxis_title="Time", yaxis_title="Voltage (μV)", height=400)
    placeholder_plot.plotly_chart(fig, use_container_width=True, key="eeg_plot")

    # Analyze emotion every N seconds
    last_analysis_time = st.session_state.get("last_analysis_time", datetime.min)
    if (datetime.now() - last_analysis_time).total_seconds() > EMOTION_CHECK_INTERVAL:
        voltage_samples = [v for t, v in data_buffer[-100:]]
        with st.spinner("Analyzing emotion..."):
            result = analyze_emotion(voltage_samples)
            emotion_history.append(result)
            st.session_state["last_analysis_time"] = datetime.now()

    # Show last emotion
    if emotion_history:
        latest = emotion_history[-1]
        emoji_map = {
            "happy": "😊", "sad": "😢", "anger": "😠",
            "relaxed": "😌", "interested": "🤔", "neutral": "😐"
        }
        icon = emoji_map.get(latest["emotion"], "😐")
        confidence_pct = latest["confidence"] * 100

        placeholder_emotion.markdown(f"""
        ### Current Emotion  
        <div style='font-size: 5rem; text-align:center'>{icon}</div>
        **Emotion:** {latest["emotion"].title()}  
        **Confidence:** {confidence_pct:.1f}%  
        **Reasoning:** {latest["reasoning"]}
        """, unsafe_allow_html=True)

    # Show last raw value
    placeholder_raw.markdown(f"### Latest EEG Raw Voltage (μV)\n{volts[-1]:.2f} μV")

if __name__ == "__main__":
    main()
