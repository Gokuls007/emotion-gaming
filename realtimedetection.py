"""Real-time emotion detection from the webcam using the CNN trained by train_emotion_cnn.py.

Run train_emotion_cnn.py first (it saves emotion_cnn.h5). Press q to quit.
"""

import os
import sys

import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "emotion_cnn.h5"
# Same order as train_emotion_cnn.py: sorted(os.listdir("train")).
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

if not os.path.exists(MODEL_PATH):
    sys.exit(f"{MODEL_PATH} not found. Train it first: python train_emotion_cnn.py")

model = load_model(MODEL_PATH, compile=False)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def extract_features(image):
    """48x48 grayscale face -> model input batch, scaled like the training data."""
    feature = np.array(image, dtype="float32").reshape(1, 48, 48, 1)
    return feature / 255.0


webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    sys.exit("Could not open the webcam.")

try:
    while True:
        ok, im = webcam.read()
        if not ok:
            print("Could not read a frame from the webcam.")
            break
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (p, q, r, s) in faces:
            face = cv2.resize(gray[q : q + s, p : p + r], (48, 48))
            pred = model.predict(extract_features(face), verbose=0)
            label = LABELS[int(pred.argmax())]
            cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
            cv2.putText(im, label, (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
        cv2.imshow("Output", im)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    webcam.release()
    cv2.destroyAllWindows()
