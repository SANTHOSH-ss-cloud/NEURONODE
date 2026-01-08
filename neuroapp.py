import streamlit as st
import serial
import threading
import time
import google.generativeai as genai

import os
from dotenv import load_dotenv

# --- User Configuration ---
BAUD_RATE = 115200
load_dotenv()

# --- Setup Gemini API ---
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def analyze_emotion_with_gemini(eeg_data):
    model = genai.GenerativeModel('gemini-pro')
    prompt = (
        "Given the following raw EEG signal data from a brain-computer interface headset, "
        "classify the primary emotion into one of the following: happy, sad, anger, relaxed, interested.\n"
        f"EEG data: {eeg_data}\n"
        "Respond with only the emotion."
    )
    response = model.generate_content(prompt)
    return response.text.strip()

def get_available_serial_ports():
    import serial.tools.list_ports
    ports = [port.device for port in serial.tools.list_ports.comports()]
    return ports

# Shared variables for thread communication
eeg_line = ""
emotion = "N/A"
stop_thread = False

def serial_reader(port, baudrate):
    global eeg_line, emotion, stop_thread
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        while not stop_thread:
            if ser.in_waiting:
                line = ser.readline().decode(errors='ignore').strip()
                if line:
                    eeg_line = line
                    try:
                        eeg_values = [float(x) for x in line.split(",") if x.replace('.', '', 1).isdigit()]
                        if eeg_values:
                            emotion_result = analyze_emotion_with_gemini(eeg_values)
                            emotion = emotion_result.capitalize()
                    except Exception as e:
                        emotion = f"Error: {e}"
            time.sleep(0.1)
        ser.close()
    except Exception as e:
        emotion = f"Serial error: {e}"

# Streamlit UI

st.title("EEG Emotion Recognition (ESP32 + Gemini API)")

ports = get_available_serial_ports()
if not ports:
    st.warning("No serial ports found. Please connect your ESP32 device.")
    st.stop()

port = st.selectbox("Select Serial Port", ports)

start_button = st.button("Start")
stop_button = st.button("Stop")

# Placeholders to update UI
raw_data_placeholder = st.empty()
emotion_placeholder = st.empty()

# To keep track of thread
if "thread" not in st.session_state:
    st.session_state.thread = None

if start_button:
    if st.session_state.thread is None or not st.session_state.thread.is_alive():
        stop_thread = False
        # Start the background thread
        st.session_state.thread = threading.Thread(target=serial_reader, args=(port, BAUD_RATE), daemon=True)
        st.session_state.thread.start()
        st.success(f"Started reading from {port}")

if stop_button:
    stop_thread = True
    if st.session_state.thread and st.session_state.thread.is_alive():
        st.session_state.thread.join()
    st.session_state.thread = None
    st.info("Stopped reading EEG data.")

# Continuously update UI while thread is running
if st.session_state.thread and st.session_state.thread.is_alive():
    for _ in range(100):  # refresh ~100 times, adjust as needed
        raw_data_placeholder.write(f"Raw EEG: {eeg_line}")
        emotion_placeholder.metric("Detected Emotion", emotion)
        time.sleep(0.5)
        st.rerun()
else:
    raw_data_placeholder.write("No data yet.")
    emotion_placeholder.metric("Detected Emotion", "N/A")

st.write("---")
st.markdown("**Emotions to be detected: Happy, Sad, Anger, Relaxed, Interested**.")
