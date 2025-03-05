import os
import cv2
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
import threading
import time

from Train_model import train_model
from Realtime_recognition import main as gesture_recognition_main
from Speech_recognition import VoiceToTextConverter

app = Flask(__name__)
socketio = SocketIO(app)

# WebSocket event to handle video stream for gesture recognition
@socketio.on('video_stream')
def handle_video_stream(data):
    # Here, you would capture the frame and perform gesture recognition using your existing code
    frame_data = data.get('frame')  # base64 or binary image data
    # Process the frame for gesture recognition (using your model and logic from Realtime_recognition.py)
    # You can return recognized gesture and confidence
    gesture, confidence = gesture_recognition_main(frame_data)
    emit('gesture_recognition', {'gesture': gesture, 'confidence': confidence})

# WebSocket event for speech recognition
@socketio.on('speech_stream')
def handle_speech_stream(data):
    # Handle real-time speech recognition using your existing VoiceToTextConverter
    audio_data = data.get('audio')
    converter = VoiceToTextConverter()
    text = converter.convert(audio_data)  # Assume you have a function for this
    emit('speech_recognition', {'text': text})

# Route to train gesture model
@app.route('/train_model', methods=['POST'])
def train_gesture_model():
    try:
        train_model()  # This function is from your Train_model.py
        return jsonify({"message": "Model trained successfully!"}), 200
    except Exception as e:
        return jsonify({"message": f"Error training model: {str(e)}"}), 500

# Route for clearing cache
@app.route('/clear_cache', methods=['GET'])
def clear_cache():
    # This function should clear the cache files
    # You can integrate the cache clearing logic here
    cache_dir = 'path_to_cache_directory'
    if os.path.exists(cache_dir):
        for f in os.listdir(cache_dir):
            os.remove(os.path.join(cache_dir, f))
    return jsonify({"message": "Cache cleared!"}), 200

# Route for speech recognition
@app.route('/speech_recognition', methods=['POST'])
def speech_recognition():
    try:
        audio = request.files['audio']  # Assume audio is uploaded
        converter = VoiceToTextConverter()
        text = converter.convert(audio)
        return jsonify({"text": text}), 200
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000)
