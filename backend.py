
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import serial
import serial.tools.list_ports
import threading
import time
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def analyze_emotion_with_gemini(eeg_data: str) -> str:
    model = genai.GenerativeModel('gemini-pro')
    prompt = (
        "Given the following raw EEG signal data from a brain-computer interface headset, "
        "classify the primary emotion into one of the following: happy, sad, anger, relaxed, interested.\n"
        f"EEG data: {eeg_data}\n"
        "Respond with only the emotion."
    )
    response = model.generate_content(prompt)
    return response.text.strip()

# Shared variables
serial_thread = None
stop_thread = False
eeg_data = "No data yet."
emotion = "N/A"

def serial_reader(port, baudrate):
    global stop_thread, eeg_data, emotion
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        while not stop_thread:
            if ser.in_waiting:
                line = ser.readline().decode(errors='ignore').strip()
                if line:
                    eeg_data = line
                    emotion = analyze_emotion_with_gemini(eeg_data)
            time.sleep(0.1)
        ser.close()
    except Exception as e:
        eeg_data = f"Serial error: {e}"
        emotion = "Error"

@app.route('/api/ports', methods=['GET'])
def get_ports():
    ports = [port.device for port in serial.tools.list_ports.comports()]
    return jsonify(ports)

@app.route('/api/start', methods=['POST'])
def start_reading():
    global serial_thread, stop_thread
    port = request.json.get('port')
    if not port:
        return jsonify({"error": "Port not specified"}), 400

    if serial_thread and serial_thread.is_alive():
        return jsonify({"message": "Already running"})

    stop_thread = False
    serial_thread = threading.Thread(target=serial_reader, args=(port, 115200), daemon=True)
    serial_thread.start()
    return jsonify({"message": "Started reading"})

@app.route('/api/stop', methods=['POST'])
def stop_reading():
    global stop_thread
    stop_thread = True
    return jsonify({"message": "Stopped reading"})

@app.route('/api/data', methods=['GET'])
def get_data():
    global eeg_data, emotion
    return jsonify({"eeg_data": eeg_data, "emotion": emotion})

if __name__ == '__main__':
    app.run(port=5000)
