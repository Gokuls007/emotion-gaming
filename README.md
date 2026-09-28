# 🎭 Emotion Gaming System

The **Emotion Gaming System** is a real-time computer vision platform that bridges human emotions with interactive digital experiences. By leveraging state-of-the-art facial recognition, the system dynamically influences game parameters (score, difficulty, and state) based on your detected mood.

---

## 🚀 Features

- **Real-Time Detection**: Analyzes facial expressions using high-accuracy deep learning models.
- **RetinaFace Integration**: Equipped with the **RetinaFace** backend for superior accuracy in challenging lighting and varied angles.
- **Unity Bridge**: Modular Unity integration via **UDP Sockets (Port 5065)**, allowing external game engines to react to player emotions.
- **Smoothing Engine**: Intelligent buffering (5-frame window) to prevent jittery state transitions.
- **Multi-threaded**: Dedicated detection thread to ensure zero lag in the video feed and game UI.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.9-3.13 (TensorFlow does not support Python 3.14 yet)
- A functional webcam
- (Optional) Unity Editor for the Gaming Edition

### 2. Environment Setup
```bash
# Create a virtual environment
python -m venv venv

# Activate on Windows
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** keep OpenCV below 5 (`requirements.txt` pins `opencv-python<5`). OpenCV 5 removed
> `cv2.CascadeClassifier` and the bundled Haar cascades, which breaks face detection here.

#### Custom CNN scripts
`emotion_game_cnn.py`, `realtimedetection.py` and `application.py` use a model you train yourself:
```bash
python train_emotion_cnn.py   # reads train/ and test/, saves emotion_cnn.h5
```

### 3. Execution

#### **Standalone Mode**
Launch the latest real-time edition to see the emotion-driven scoring system.
```bash
python emotion_game_v2.py
```

#### **Unity Integration Mode**
Launch the bridge to send data to the **3D Game Kit** or any other Unity project listening on port 5065.
```bash
python emotion_game_unity.py
```

---

## 🎮 How it Works

The system recognizes six primary emotions, each mapped to a specific game mode:

| Emotion | Game Mode | Impact |
| :--- | :--- | :--- |
| 😊 **Happy** | Bonus Mode | +100 Points, 1.3x Difficulty |
| 😠 **Angry** | Battle Mode | +150 Points, 1.5x Difficulty |
| 😲 **Surprise** | Mystery Mode | +200 Points |
| 😨 **Fear** | Stealth Mode | +75 Points, 0.8x Difficulty |
| 😢 **Sad** | Comfort Mode | +50 Points, 0.7x Difficulty |
| 🤢 **Disgust** | Defense Mode | +80 Points |

---

## 🔌 Unity Connection

The `emotion_game_unity.py` script broadcasts a JSON payload to `127.0.0.1:5065`:
```json
{
  "emotion": "happy",
  "confidence": 98.5,
  "score": 1200,
  "difficulty": 1.3,
  "game_state": "bonus_mode",
  "timestamp": 1618512345.0
}
```
You can import this into Unity using a standard UDP receiver script to control any game object or scene property.

---

## 🛡️ License
This project is for research and educational purposes. For usage in commercial environments, refer to standard MIT License guidelines.

---

*Built with ❤️ using DeepFace, OpenCV, and TensorFlow.*
