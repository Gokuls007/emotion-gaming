"""
Complete Emotion Recognition Gaming System
Using YOUR Custom CNN Model
Features: Real-time detection, Unity integration, Multi-threading, Emotion smoothing
"""

import cv2
import numpy as np
import time
import threading
from collections import deque
import socket
import json
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization

class EmotionGamingSystemCNN:
    def __init__(self, unity_enabled=False, unity_ip="127.0.0.1", unity_port=5065):
        # Load YOUR trained CNN model
        print("Loading custom CNN model...")
        self.model = self.load_cnn_model()
        print("✓ Model loaded successfully!")
        
        # Face detection setup
        haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(haar_file)
        
        # Emotion labels
        self.labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Game state
        self.score = 0
        self.difficulty = 1.0
        self.game_state = "normal"
        
        # Real-time detection variables
        self.current_emotion = "neutral"
        self.current_confidence = 0.0
        self.detection_active = True
        self.frame_for_analysis = None
        self.frame_lock = threading.Lock()
        
        # Emotion smoothing (5-frame buffer)
        self.emotion_buffer = deque(maxlen=5)
        self.last_detection_time = time.time()
        self.detection_interval = 2.0  # Detect every 2 seconds
        
        # Status message
        self.status_message = "Starting..."
        self.message_timer = 0
        
        # Unity UDP setup
        self.unity_enabled = unity_enabled
        if self.unity_enabled:
            self.unity_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.unity_ip = unity_ip
            self.unity_port = unity_port
            print(f"✓ Unity UDP enabled: {unity_ip}:{unity_port}")
    
    def load_cnn_model(self):
        """Load the CNN model with YOUR exact architecture"""
        try:
            # Try loading directly
            model = load_model("model.h5", compile=False)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            return model
        except:
            # Rebuild YOUR exact architecture from the notebook
            print("Rebuilding YOUR model architecture...")
            model = Sequential()
            
            # Block 1
            model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(Dropout(0.4))
            
            # Block 2
            model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(Dropout(0.4))
            
            # Block 3
            model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(Dropout(0.4))
            
            # Block 4
            model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(Dropout(0.4))
            
            # Dense layers
            model.add(Flatten())
            model.add(Dense(512, activation='relu'))
            model.add(Dropout(0.4))
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.3))
            
            # Output layer
            model.add(Dense(7, activation='softmax'))
            
            # Load the trained weights
            model.load_weights("model.h5")
            print("✓ Weights loaded into rebuilt architecture!")
            return model
    
    def detect_emotion(self, frame):
        """Detect emotion using YOUR trained CNN"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect face
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None, 0
        
        # Get first face
        (x, y, w, h) = faces[0]
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        
        # Prepare for CNN (normalize and reshape)
        img = face.reshape(1, 48, 48, 1) / 255.0
        
        # Predict with YOUR model
        pred = self.model.predict(img, verbose=0)
        emotion_idx = int(pred.argmax())
        emotion = self.labels[emotion_idx]
        confidence = float(pred.max() * 100)
        
        return emotion, confidence
    
    def smooth_emotion(self, new_emotion):
        """Smooth emotions using majority voting"""
        self.emotion_buffer.append(new_emotion)
        
        emotion_counts = {}
        for emotion in self.emotion_buffer:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        smoothed_emotion = max(emotion_counts, key=emotion_counts.get)
        return smoothed_emotion
    
    def process_emotion(self, emotion, confidence):
        """Apply game changes based on emotion"""
        old_state = self.game_state
        old_score = self.score
        
        if emotion == 'happy':
            self.score += 100
            self.difficulty = 1.3
            self.game_state = "bonus_mode"
            message = "😊 Happy detected! +100 points!"
            
        elif emotion == 'sad':
            self.score += 50
            self.difficulty = 0.7
            self.game_state = "comfort_mode"
            message = "😢 Comfort mode - easier gameplay"
            
        elif emotion == 'angry':
            self.score += 150
            self.difficulty = 1.5
            self.game_state = "battle_mode"
            message = "😠 RAGE MODE! Boss battle!"
            
        elif emotion == 'surprise':
            self.score += 200
            self.game_state = "mystery_mode"
            message = "😲 SURPRISE! Mystery bonus!"
            
        elif emotion == 'fear':
            self.score += 75
            self.difficulty = 0.8
            self.game_state = "stealth_mode"
            message = "😨 Stealth mode activated"
            
        elif emotion == 'disgust':
            self.score += 80
            self.game_state = "defense_mode"
            message = "🤢 Defense mode!"
            
        else:
            self.game_state = "normal"
            message = f"Detected: {emotion}"
        
        # Send to Unity if state changed
        if old_state != self.game_state or old_score != self.score:
            if self.unity_enabled:
                unity_data = {
                    "emotion": str(emotion),
                    "confidence": float(confidence),
                    "score": int(self.score),
                    "difficulty": float(self.difficulty),
                    "game_state": str(self.game_state),
                    "timestamp": float(time.time())
                }
                self.send_to_unity(unity_data)
            
            self.status_message = message
            self.message_timer = time.time()
        
        return message
    
    def send_to_unity(self, data):
        """Send data to Unity via UDP"""
        if not self.unity_enabled:
            return
        
        try:
            message = json.dumps(data)
            self.unity_socket.sendto(message.encode(), (self.unity_ip, self.unity_port))
        except Exception as e:
            print(f"Unity send error: {e}")
    
    def detection_thread(self):
        """Background thread for continuous emotion detection"""
        print("Detection thread started (using YOUR custom CNN)...")
        
        while self.detection_active:
            current_time = time.time()
            
            # Check if it's time for new detection
            if current_time - self.last_detection_time >= self.detection_interval:
                
                # Get current frame safely
                with self.frame_lock:
                    if self.frame_for_analysis is not None:
                        frame_copy = self.frame_for_analysis.copy()
                    else:
                        continue
                
                try:
                    # Detect emotion with YOUR CNN
                    result = self.detect_emotion(frame_copy)
                    
                    if result[0] is not None:
                        raw_emotion, confidence = result
                        
                        # Smooth the emotion
                        smoothed_emotion = self.smooth_emotion(raw_emotion)
                        
                        # Update current emotion
                        self.current_emotion = smoothed_emotion
                        self.current_confidence = confidence
                        
                        # Process game changes
                        self.process_emotion(smoothed_emotion, confidence)
                        
                        status = "→ Unity" if self.unity_enabled else ""
                        print(f"[{time.strftime('%H:%M:%S')}] {raw_emotion} → {smoothed_emotion} ({confidence:.1f}%) {status}")
                    else:
                        print(f"[{time.strftime('%H:%M:%S')}] No face detected")
                    
                except Exception as e:
                    print(f"Detection error: {e}")
                
                self.last_detection_time = current_time
            
            # Small sleep to prevent CPU overuse
            time.sleep(0.1)
    
    def draw_game_info(self, frame):
        """Draw all game information on the frame"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay for top info
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Score
        cv2.putText(frame, f"Score: {self.score}", (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Difficulty
        difficulty_color = (0, 255, 255) if self.difficulty > 1.0 else (255, 200, 0)
        cv2.putText(frame, f"Difficulty: {self.difficulty:.1f}x", (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, difficulty_color, 2)
        
        # Game Mode
        mode_colors = {
            "bonus_mode": (0, 255, 0),
            "battle_mode": (0, 0, 255),
            "comfort_mode": (255, 150, 0),
            "mystery_mode": (255, 0, 255),
            "stealth_mode": (200, 200, 200),
            "defense_mode": (0, 165, 255),
            "normal": (255, 255, 255)
        }
        color = mode_colors.get(self.game_state, (255, 255, 255))
        cv2.putText(frame, f"Mode: {self.game_state.replace('_', ' ').title()}", (10, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Unity status
        unity_status = "Unity: CONNECTED" if self.unity_enabled else "Unity: DISABLED"
        unity_color = (0, 255, 0) if self.unity_enabled else (128, 128, 128)
        cv2.putText(frame, unity_status, (10, 155),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, unity_color, 2)
        
        # Model info
        cv2.putText(frame, "Model: Your Custom CNN", (10, 190),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
        
        # Current emotion (top right)
        emotion_text = f"Emotion: {self.current_emotion.upper()}"
        confidence_text = f"Confidence: {self.current_confidence:.1f}%"
        cv2.putText(frame, emotion_text, (w - 400, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, confidence_text, (w - 400, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Status message (shows for 3 seconds after state change)
        if time.time() - self.message_timer < 3.0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h-60), (w, h), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            
            cv2.putText(frame, self.status_message, (10, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Press Q to quit", (w - 200, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return frame
    
    def run(self):
        """Main application loop"""
        print("="*70)
        print("    EMOTION RECOGNITION GAMING SYSTEM")
        print("         YOUR Custom CNN Edition")
        print("="*70)
        print("\nFeatures:")
        print("  ✓ YOUR trained CNN model (55-65% accuracy)")
        print("  ✓ Fast real-time detection (every 2 seconds)")
        print("  ✓ Emotion smoothing (5-frame buffer)")
        print("  ✓ Multi-threaded processing (no lag)")
        print("  ✓ Haar Cascade face detection")
        if self.unity_enabled:
            print(f"  ✓ Unity UDP communication (Port {self.unity_port})")
        print("\nControls:")
        print("  Q - Quit and see final score")
        print("-" * 70)
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot access webcam")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Start detection thread
        detection_thread = threading.Thread(target=self.detection_thread, daemon=True)
        detection_thread.start()
        
        print("\nSystem ready! Detecting emotions with YOUR CNN...")
        if self.unity_enabled:
            print("→ Sending data to Unity on port 5065...")
        print()
        
        # Main video loop
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update frame for analysis thread (thread-safe)
            with self.frame_lock:
                self.frame_for_analysis = frame.copy()
            
            # Draw game info overlay
            display_frame = self.draw_game_info(frame)
            
            # Show window
            cv2.imshow('Emotion Gaming System - Custom CNN', display_frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Cleanup
        self.detection_active = False
        cap.release()
        cv2.destroyAllWindows()
        if self.unity_enabled:
            self.unity_socket.close()
        
        # Final stats
        print("\n" + "="*70)
        print("                      GAME OVER")
        print("="*70)
        print(f"Final Score: {self.score}")
        print(f"Final Difficulty: {self.difficulty:.1f}x")
        print(f"Total Emotions Detected: {len(self.emotion_buffer)}")
        print(f"Model Used: Custom CNN (YOUR trained model)")
        print("="*70)

def main():
    print("\n" + "="*70)
    print("   EMOTION RECOGNITION GAMING SYSTEM")
    print("        Complete Custom CNN Edition")
    print("="*70)
    print("\nThis system uses YOUR custom-trained CNN model")
    print("for fast, accurate emotion recognition!")
    print()
    
    print("Enable Unity UDP communication?")
    print("1. Yes - Send data to Unity 3D Game (Port 5065)")
    print("2. No - Standalone mode (just emotion detection)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    unity_enabled = (choice == '1')
    """
Complete Emotion Recognition Gaming System
Using YOUR Custom CNN Model
Features: Real-time detection, Unity integration, Multi-threading, Emotion smoothing
"""

import cv2
import numpy as np
import time
import threading
from collections import deque
import socket
import json
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization

class EmotionGamingSystemCNN:
    def __init__(self, unity_enabled=False, unity_ip="127.0.0.1", unity_port=5065):
        # Load YOUR trained CNN model
        print("Loading custom CNN model...")
        self.model = self.load_cnn_model()
        print("✓ Model loaded successfully!")
        
        # Face detection setup
        haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(haar_file)
        
        # Emotion labels
        self.labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Game state
        self.score = 0
        self.difficulty = 1.0
        self.game_state = "normal"
        
        # Real-time detection variables
        self.current_emotion = "neutral"
        self.current_confidence = 0.0
        self.detection_active = True
        self.frame_for_analysis = None
        self.frame_lock = threading.Lock()
        
        # Emotion smoothing (5-frame buffer)
        self.emotion_buffer = deque(maxlen=5)
        self.last_detection_time = time.time()
        self.detection_interval = 2.0  # Detect every 2 seconds
        
        # Status message
        self.status_message = "Starting..."
        self.message_timer = 0
        
        # Unity UDP setup
        self.unity_enabled = unity_enabled
        if self.unity_enabled:
            self.unity_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.unity_ip = unity_ip
            self.unity_port = unity_port
            print(f"✓ Unity UDP enabled: {unity_ip}:{unity_port}")
    
    def load_cnn_model(self):
        """Load YOUR custom trained CNN"""
        print("Loading YOUR custom CNN...")
        from keras.models import load_model
        model = load_model('emotion_cnn.h5')
        print("✓ Model loaded!")
        return model
    
    def detect_emotion(self, frame):
        """Detect emotion using YOUR trained CNN"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect face
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None, 0
        
        # Get first face
        (x, y, w, h) = faces[0]
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        
        # Prepare for CNN (normalize and reshape)
        img = face.reshape(1, 48, 48, 1) / 255.0
        
        # Predict with YOUR model
        pred = self.model.predict(img, verbose=0)
        emotion_idx = int(pred.argmax())
        emotion = self.labels[emotion_idx]
        confidence = float(pred.max() * 100)
        
        return emotion, confidence
    
    def smooth_emotion(self, new_emotion):
        """Smooth emotions using majority voting"""
        self.emotion_buffer.append(new_emotion)
        
        emotion_counts = {}
        for emotion in self.emotion_buffer:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        smoothed_emotion = max(emotion_counts, key=emotion_counts.get)
        return smoothed_emotion
    
    def process_emotion(self, emotion, confidence):
        """Apply game changes based on emotion"""
        old_state = self.game_state
        old_score = self.score
        
        if emotion == 'happy':
            self.score += 100
            self.difficulty = 1.3
            self.game_state = "bonus_mode"
            message = "😊 Happy detected! +100 points!"
            
        elif emotion == 'sad':
            self.score += 50
            self.difficulty = 0.7
            self.game_state = "comfort_mode"
            message = "😢 Comfort mode - easier gameplay"
            
        elif emotion == 'angry':
            self.score += 150
            self.difficulty = 1.5
            self.game_state = "battle_mode"
            message = "😠 RAGE MODE! Boss battle!"
            
        elif emotion == 'surprise':
            self.score += 200
            self.game_state = "mystery_mode"
            message = "😲 SURPRISE! Mystery bonus!"
            
        elif emotion == 'fear':
            self.score += 75
            self.difficulty = 0.8
            self.game_state = "stealth_mode"
            message = "😨 Stealth mode activated"
            
        elif emotion == 'disgust':
            self.score += 80
            self.game_state = "defense_mode"
            message = "🤢 Defense mode!"
            
        else:
            self.game_state = "normal"
            message = f"Detected: {emotion}"
        
        # Send to Unity if state changed
        if old_state != self.game_state or old_score != self.score:
            if self.unity_enabled:
                unity_data = {
                    "emotion": str(emotion),
                    "confidence": float(confidence),
                    "score": int(self.score),
                    "difficulty": float(self.difficulty),
                    "game_state": str(self.game_state),
                    "timestamp": float(time.time())
                }
                self.send_to_unity(unity_data)
            
            self.status_message = message
            self.message_timer = time.time()
        
        return message
    
    def send_to_unity(self, data):
        """Send data to Unity via UDP"""
        if not self.unity_enabled:
            return
        
        try:
            message = json.dumps(data)
            self.unity_socket.sendto(message.encode(), (self.unity_ip, self.unity_port))
        except Exception as e:
            print(f"Unity send error: {e}")
    
    def detection_thread(self):
        """Background thread for continuous emotion detection"""
        print("Detection thread started (using YOUR custom CNN)...")
        
        while self.detection_active:
            current_time = time.time()
            
            # Check if it's time for new detection
            if current_time - self.last_detection_time >= self.detection_interval:
                
                # Get current frame safely
                with self.frame_lock:
                    if self.frame_for_analysis is not None:
                        frame_copy = self.frame_for_analysis.copy()
                    else:
                        continue
                
                try:
                    # Detect emotion with YOUR CNN
                    result = self.detect_emotion(frame_copy)
                    
                    if result[0] is not None:
                        raw_emotion, confidence = result
                        
                        # Smooth the emotion
                        smoothed_emotion = self.smooth_emotion(raw_emotion)
                        
                        # Update current emotion
                        self.current_emotion = smoothed_emotion
                        self.current_confidence = confidence
                        
                        # Process game changes
                        self.process_emotion(smoothed_emotion, confidence)
                        
                        status = "→ Unity" if self.unity_enabled else ""
                        print(f"[{time.strftime('%H:%M:%S')}] {raw_emotion} → {smoothed_emotion} ({confidence:.1f}%) {status}")
                    else:
                        print(f"[{time.strftime('%H:%M:%S')}] No face detected")
                    
                except Exception as e:
                    print(f"Detection error: {e}")
                
                self.last_detection_time = current_time
            
            # Small sleep to prevent CPU overuse
            time.sleep(0.1)
    
    def draw_game_info(self, frame):
        """Draw all game information on the frame"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay for top info
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Score
        cv2.putText(frame, f"Score: {self.score}", (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Difficulty
        difficulty_color = (0, 255, 255) if self.difficulty > 1.0 else (255, 200, 0)
        cv2.putText(frame, f"Difficulty: {self.difficulty:.1f}x", (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, difficulty_color, 2)
        
        # Game Mode
        mode_colors = {
            "bonus_mode": (0, 255, 0),
            "battle_mode": (0, 0, 255),
            "comfort_mode": (255, 150, 0),
            "mystery_mode": (255, 0, 255),
            "stealth_mode": (200, 200, 200),
            "defense_mode": (0, 165, 255),
            "normal": (255, 255, 255)
        }
        color = mode_colors.get(self.game_state, (255, 255, 255))
        cv2.putText(frame, f"Mode: {self.game_state.replace('_', ' ').title()}", (10, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Unity status
        unity_status = "Unity: CONNECTED" if self.unity_enabled else "Unity: DISABLED"
        unity_color = (0, 255, 0) if self.unity_enabled else (128, 128, 128)
        cv2.putText(frame, unity_status, (10, 155),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, unity_color, 2)
        
        # Model info
        cv2.putText(frame, "Model: Your Custom CNN", (10, 190),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
        
        # Current emotion (top right)
        emotion_text = f"Emotion: {self.current_emotion.upper()}"
        confidence_text = f"Confidence: {self.current_confidence:.1f}%"
        cv2.putText(frame, emotion_text, (w - 400, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, confidence_text, (w - 400, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Status message (shows for 3 seconds after state change)
        if time.time() - self.message_timer < 3.0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h-60), (w, h), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            
            cv2.putText(frame, self.status_message, (10, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Press Q to quit", (w - 200, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return frame
    
    def run(self):
        """Main application loop"""
        print("="*70)
        print("    EMOTION RECOGNITION GAMING SYSTEM")
        print("         YOUR Custom CNN Edition")
        print("="*70)
        print("\nFeatures:")
        print("  ✓ YOUR trained CNN model (55-65% accuracy)")
        print("  ✓ Fast real-time detection (every 2 seconds)")
        print("  ✓ Emotion smoothing (5-frame buffer)")
        print("  ✓ Multi-threaded processing (no lag)")
        print("  ✓ Haar Cascade face detection")
        if self.unity_enabled:
            print(f"  ✓ Unity UDP communication (Port {self.unity_port})")
        print("\nControls:")
        print("  Q - Quit and see final score")
        print("-" * 70)
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot access webcam")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Start detection thread
        detection_thread = threading.Thread(target=self.detection_thread, daemon=True)
        detection_thread.start()
        
        print("\nSystem ready! Detecting emotions with YOUR CNN...")
        if self.unity_enabled:
            print("→ Sending data to Unity on port 5065...")
        print()
        
        # Main video loop
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update frame for analysis thread (thread-safe)
            with self.frame_lock:
                self.frame_for_analysis = frame.copy()
            
            # Draw game info overlay
            display_frame = self.draw_game_info(frame)
            
            # Show window
            cv2.imshow('Emotion Gaming System - Custom CNN', display_frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Cleanup
        self.detection_active = False
        cap.release()
        cv2.destroyAllWindows()
        if self.unity_enabled:
            self.unity_socket.close()
        
        # Final stats
        print("\n" + "="*70)
        print("                      GAME OVER")
        print("="*70)
        print(f"Final Score: {self.score}")
        print(f"Final Difficulty: {self.difficulty:.1f}x")
        print(f"Total Emotions Detected: {len(self.emotion_buffer)}")
        print(f"Model Used: Custom CNN (YOUR trained model)")
        print("="*70)

def main():
    print("\n" + "="*70)
    print("   EMOTION RECOGNITION GAMING SYSTEM")
    print("        Complete Custom CNN Edition")
    print("="*70)
    print("\nThis system uses YOUR custom-trained CNN model")
    print("for fast, accurate emotion recognition!")
    print()
    
    print("Enable Unity UDP communication?")
    print("1. Yes - Send data to Unity 3D Game (Port 5065)")
    print("2. No - Standalone mode (just emotion detection)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    unity_enabled = (choice == '1')
    
    if unity_enabled:
        print("\n→ Unity mode enabled")
        print("→ Make sure Unity is running with EmotionManager")
        print("→ Press Play in Unity, then press Enter here")
        input("Press Enter when ready...")
    
    # Initialize and run the system
    game = EmotionGamingSystemCNN(unity_enabled=unity_enabled)
    game.run()

if __name__ == "__main__":
    main()
    if unity_enabled:
        print("\n→ Unity mode enabled")
        print("→ Make sure Unity is running with EmotionManager")
        print("→ Press Play in Unity, then press Enter here")
        input("Press Enter when ready...")
    
    # Initialize and run the system
    game = EmotionGamingSystemCNN(unity_enabled=unity_enabled)
    game.run()

if __name__ == "__main__":
    main()