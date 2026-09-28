"""Webcam emotion demo (large window) using the CNN trained by train_emotion_cnn.py.

Run train_emotion_cnn.py first (it saves emotion_cnn.h5). Press q to quit.
"""

import os
import sys

import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "emotion_cnn.h5"
# Same order as train_emotion_cnn.py: sorted(os.listdir("train")).
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

if not os.path.exists(MODEL_PATH):
    sys.exit(f"{MODEL_PATH} not found. Train it first: python train_emotion_cnn.py")

loaded_model = load_model(MODEL_PATH, compile=False)
facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cv2.ocl.setUseOpenCL(False)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    sys.exit("Could not open the webcam.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray = cv2.resize(gray[y : y + h, x : x + w], (48, 48)).astype("float32") / 255.0
        cropped_img = roi_gray.reshape(1, 48, 48, 1)
        prediction = loaded_model.predict(cropped_img, verbose=0)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Video", cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
