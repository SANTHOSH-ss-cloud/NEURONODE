import streamlit as st
import random
import time
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt

# -------------------------------
# Data model for emotions
# -------------------------------
class EmotionResult:
    def __init__(self, emotion, confidence, timestamp, source="Demo"):
        self.emotion = emotion
        self.confidence = confidence
        self.timestamp = timestamp
        self.source = source

# -------------------------------
# Session state setup
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "signal_data" not in st.session_state:
    st.session_state.signal_data = deque(maxlen=200)

# -------------------------------
# Demo emotion generator
# -------------------------------
def append_random_demo_emotion():
    last_emo = st.session_state.history[-1].emotion if st.session_state.history else None
    
    # Weighted pool (Happy & Neutral dominant, Sad rare, no Angry)
    weighted_choices = (
        ["happy"] * 4 +
        ["neutral"] * 4 +
        ["sad"] * 1 +
        ["relaxed"] * 2 +
        ["interested"] * 2
    )

    # Avoid repeating the same emotion back-to-back
    choices = [e for e in weighted_choices if e != last_emo]

    emo = random.choice(choices)
    st.session_state.history.append(
        EmotionResult(emo, random.uniform(0.85, 0.98), datetime.now(), "Demo")
    )

# -------------------------------
# Real-time plot
# -------------------------------
def create_realtime_plot():
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(list(st.session_state.signal_data), color="blue")
    ax.set_title("Real-Time EEG Signal (Demo Mode)")
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Amplitude (a.u.)")
    st.pyplot(fig)

# -------------------------------
# Main Streamlit app
# -------------------------------
st.set_page_config(page_title="EEG Emotion Detection Dashboard", layout="wide")

st.title("🧠 EEG Emotion Detection Dashboard")
st.caption("Demo Mode — Shows simulated EEG & Emotion outputs (Happy, Neutral, Sad).")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Controls")
    demo_mode = st.checkbox("Enable Demo Mode", value=True)
    if st.button("Clear History"):
        st.session_state.history.clear()
        st.session_state.signal_data.clear()
        st.success("History cleared!")

# Signal plotting
create_realtime_plot()

# Emotion history table
st.subheader("Emotion History")
if st.session_state.history:
    for emo in reversed(st.session_state.history[-10:]):  # show last 10
        st.write(
            f"**{emo.emotion.capitalize()}** "
            f"(Conf: {emo.confidence:.2f}) — "
            f"{emo.timestamp.strftime('%H:%M:%S')}"
        )
else:
    st.info("No emotions recorded yet.")

# -------------------------------
# Demo mode simulation
# -------------------------------
if demo_mode:
    for _ in range(5):  # update a few steps per refresh
        # Fake EEG signal
        st.session_state.signal_data.append(random.uniform(-1, 1))
        # Fake emotion classification
        append_random_demo_emotion()
    time.sleep(1)
    st.rerun()
