import os
import cv2
import mediapipe as mp
from mediapipe import solutions as mp_solutions
hands = mp_solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
drawing_utils = mp_solutions.drawing_utils

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imutils import paths
from sklearn.preprocessing import LabelEncoder
import time

# Constants
GESTURES = ["Hi, how are you?", "Hey, can you please help me?", "Hey, I need some water.",
            "Hey, good job!", "I love you.", "Bye.", "What is your name?", "Yes, I agree."]
IMAGE_SIZE = (224, 224)  # Size for MobileNetV2
NUM_IMAGES_PER_GESTURE = 100  # Increased dataset size
TEST_SIZE = 0.2
RANDOM_STATE = 42
LEARNING_RATE = 1e-4
EPOCHS = 20
BATCH_SIZE = 32

# Sanitization and Folder Creation
def sanitize_gesture_name(gesture_name):
    return gesture_name.replace(",", "").replace("?", "").replace(" ", "_")

def create_folders():
    os.makedirs("gesture_images", exist_ok=True)
    for gesture in GESTURES:
        sanitized_gesture = sanitize_gesture_name(gesture)
        os.makedirs(os.path.join("gesture_images", sanitized_gesture), exist_ok=True)

# Image Capture with Validation
def capture_images():
    cap = cv2.VideoCapture(0)
    print(f"Press a number (0-{len(GESTURES) - 1}) to save gesture image, 'q' to quit, 'd' to delete the last image.")
    image_count = {sanitize_gesture_name(gesture): len(os.listdir(os.path.join("gesture_images", sanitize_gesture_name(gesture)))) for gesture in GESTURES}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                drawing_utils.draw_landmarks(frame, hand_landmarks, mp_solutions.hands.HAND_CONNECTIONS)

        cv2.imshow("Capture Gestures", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in [ord(str(i)) for i in range(len(GESTURES))]:
            gesture_idx = int(chr(key))
            gesture_name = GESTURES[gesture_idx]
            sanitized_gesture_name = sanitize_gesture_name(gesture_name) if gesture_name else ""
            folder_path = os.path.join("gesture_images", sanitized_gesture_name)

            if image_count[sanitized_gesture_name] >= NUM_IMAGES_PER_GESTURE:
                print(f"Already collected {NUM_IMAGES_PER_GESTURE} images for '{gesture_name}'.")
                continue

            img_path = os.path.join(folder_path, f"{sanitized_gesture_name}_{image_count[sanitized_gesture_name]}.jpg") if sanitized_gesture_name else ""
            cv2.imwrite(img_path, frame)
            image_count[sanitized_gesture_name] += 1
            print(f"Saved: {img_path}")

            if image_count[sanitized_gesture_name] == NUM_IMAGES_PER_GESTURE:
                print(f"Collected {NUM_IMAGES_PER_GESTURE} images for '{gesture_name}'.")

        elif key == ord('q'):
            break
        elif key == ord('d'):  # Delete last image that was taken.
            if image_count[sanitized_gesture_name] > 0:
                image_count[sanitized_gesture_name] -= 1
                os.remove(img_path)
                print(f'deleted {img_path}')

    cap.release()
    cv2.destroyAllWindows()

# Model Training (MobileNetV2 with Transfer Learning)
def train_model():
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import AveragePooling2D, Dense, Flatten, Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras import backend as K
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.utils import to_categorical

    image_paths = list(paths.list_images("gesture_images"))
    data, labels = [], []
    label_encoder = LabelEncoder()

    for image_path in image_paths:
        label = image_path.split(os.path.sep)[-2]
        image = load_img(image_path, target_size=IMAGE_SIZE)
        image = img_to_array(image)
        image = preprocess_input(image)
        data.append(image)
        labels.append(label)

    labels = label_encoder.fit_transform(labels)
    labels = to_categorical(labels)
    data = np.array(data)

    base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    head_model = base_model.output
    head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(128, activation="relu")(head_model)
    head_model = Dense(len(GESTURES), activation="softmax")(head_model)
    model = Model(inputs=base_model.input, outputs=head_model)

    for layer in base_model.layers:
        layer.trainable = False

    opt = Adam(learning_rate=LEARNING_RATE)

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    train_datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")
    train_datagen.fit(data)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_STATE)

    model.fit(train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE), steps_per_epoch=len(X_train) // BATCH_SIZE, validation_data=(X_test, y_test), validation_steps=len(X_test) // BATCH_SIZE, epochs=EPOCHS)

    predictions = model.predict(X_test, batch_size=BATCH_SIZE)
    predictions = np.argmax(predictions, axis=1)
    y_test_decoded = np.argmax(y_test, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test_decoded, predictions, target_names=[sanitize_gesture_name(gesture) for gesture in GESTURES]))

    cm = confusion_matrix(y_test_decoded, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=[sanitize_gesture_name(gesture) for gesture in GESTURES], yticklabels=[sanitize_gesture_name(gesture) for gesture in GESTURES])
    plt.title("Confusion Matrix")
    plt.show()

    model.save("gesture_model.h5")
    print("Model saved as gesture_model.h5.")

if __name__ == "__main__":
    create_folders()
    capture_images()
    train_model()
