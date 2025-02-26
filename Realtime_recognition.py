import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf
import pyttsx3
import time
import threading
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

GESTURES = [
    "Bye",
    "Hey can you please help me",
    "Hey good job!",
    "Hey I need some water",
    "Hi how are you",
    "I love you",
    "What is your name",
    "Yes I agree"
]

engine = pyttsx3.init()
engine.setProperty("rate", 180)
engine.setProperty("volume", 0.9)

try:
    model = tf.keras.models.load_model("gesture_model.keras")
except FileNotFoundError:
    print("Model file 'gesture_model.keras' not found. Please train the model first.")
    exit()

# Initialize MediaPipe Hand Landmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

confidence_threshold = 0.7
verbose_mode = False

def collect_landmarks(frame_rgb):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    results = hand_landmarker.detect(mp_image)
    landmarks = []
    if results.hand_landmarks:
        for hand_landmarks_list in results.hand_landmarks:
            for landmark in hand_landmarks_list:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
    return landmarks

def speak_async(text):
    def speak():
        engine.say(text)
        engine.runAndWait()

    thread = threading.Thread(target=speak)
    thread.start()

def real_time_recognition():
    global confidence_threshold

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    last_gesture = None
    last_gesture_time = 0
    gesture_hold_duration = 2
    gesture_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (224, 224))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_preprocessed = preprocess_input(frame_rgb)
        frame_np = np.expand_dims(frame_preprocessed, axis=0)

        try:
            prediction = model.predict(frame_np)
            gesture_index = np.argmax(prediction)
            gesture = GESTURES[gesture_index]
            confidence = prediction[0][gesture_index]

            if confidence > confidence_threshold:
                current_time = time.time()
                if gesture == last_gesture and current_time - last_gesture_time >= gesture_hold_duration:
                    if last_gesture != "":
                        speak_async(gesture)
                        last_gesture = ""
                        gesture_history.insert(0, gesture)
                        if len(gesture_history) > 5:
                            gesture_history.pop()

                elif gesture != last_gesture:
                    last_gesture = gesture
                    last_gesture_time = current_time

                cv2.putText(frame, f"{gesture} ({confidence:.2f})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"Uncertain Gesture ({confidence:.2f})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if results.hand_landmarks:
                annotated_image = vision.DrawingUtils.draw_landmarks_on_image(frame, results)
                frame = annotated_image.numpy_view()

            if verbose_mode:
                print(f"Prediction: {gesture}, Confidence: {confidence:.2f}")

        except (ValueError, IndexError, TypeError) as e:
            print(f"Prediction error: {e}")
            cv2.putText(frame, "Error", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        y_offset = 100
        for g in gesture_history:
            cv2.putText(frame, f"Prev: {g}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25

        cv2.putText(frame, f"Conf: {confidence_threshold:.2f}", (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Gesture Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('v'):
            verbose_mode = not verbose_mode
            print(f"Verbose mode: {verbose_mode}")
        elif key == ord('c'):
            gesture_history.clear()
        elif key == ord('a'):
            confidence_threshold = min(0.95, confidence_threshold + 0.05)
            print(f"Confidence threshold: {confidence_threshold:.2f}")
        elif key == ord('s'):
            confidence_threshold = max(0.5, confidence_threshold - 0.05)
            print(f"Confidence threshold: {confidence_threshold:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    engine.stop()

if __name__ == "__main__":
    real_time_recognition()
