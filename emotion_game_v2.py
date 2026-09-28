import cv2
from deepface import DeepFace
import time
import numpy as np
import threading
from collections import deque

class EmotionGamingSystem:
    def __init__(self):
        self.score = 0
        self.difficulty = 1.0
        self.game_state = "normal"
        
        # Real-time detection variables
        self.current_emotion = "neutral"
        self.current_confidence = 0.0
        self.detection_active = True
        self.frame_for_analysis = None
        self.frame_lock = threading.Lock()
        
        # Emotion smoothing buffer (stores last 5 emotions)
        self.emotion_buffer = deque(maxlen=5)
        self.last_detection_time = time.time()
        self.detection_interval = 2.0  # Analyze every 2 seconds
        
        # Status message
        self.status_message = "Starting..."
        self.message_timer = 0
        
    def smooth_emotion(self, new_emotion):
        """Smooth emotions by using majority vote from buffer"""
        self.emotion_buffer.append(new_emotion)
        
        # Count occurrences of each emotion
        emotion_counts = {}
        for emotion in self.emotion_buffer:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Return most common emotion
        smoothed_emotion = max(emotion_counts, key=emotion_counts.get)
        return smoothed_emotion
        
    def process_emotion(self, emotion, confidence):
        """Apply game changes based on emotion"""
        old_state = self.game_state
        
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
            message = "🤢 Defense mode - protect yourself!"
            
        else:
            self.game_state = "normal"
            message = f"Detected: {emotion}"
        
        # Only update message if state changed
        if old_state != self.game_state:
            self.status_message = message
            self.message_timer = time.time()
            
        return message
    
    def detection_thread(self):
        """Background thread for continuous emotion detection"""
        print("Detection thread started...")
        
        while self.detection_active:
            current_time = time.time()
            
            # Check if it's time for a new detection
            if current_time - self.last_detection_time >= self.detection_interval:
                
                # Get the current frame safely
                with self.frame_lock:
                    frame_copy = (
                        self.frame_for_analysis.copy()
                        if self.frame_for_analysis is not None
                        else None
                    )
                if frame_copy is None:
                    time.sleep(0.05)  # no frame yet: don't spin at 100% CPU
                    continue
                
                try:
                    # Save frame temporarily
                    cv2.imwrite('temp_capture.jpg', frame_copy)
                    
                    # Detect emotion
                    result = DeepFace.analyze('temp_capture.jpg', 
                                            actions=['emotion'], 
                                            enforce_detection=False,
                                            detector_backend='retinaface',
                                            align=True,
                                            silent=True)
                    
                    if isinstance(result, list):
                        result = result[0]
                    
                    raw_emotion = result['dominant_emotion']
                    emotions_dict = result['emotion']
                    confidence = emotions_dict[raw_emotion]
                    
                    # Smooth the emotion
                    smoothed_emotion = self.smooth_emotion(raw_emotion)
                    
                    # Update current emotion
                    self.current_emotion = smoothed_emotion
                    self.current_confidence = confidence
                    
                    # Process game changes
                    message = self.process_emotion(smoothed_emotion, confidence)
                    
                    print(f"[{time.strftime('%H:%M:%S')}] Raw: {raw_emotion} -> Smoothed: {smoothed_emotion} ({confidence:.1f}%)")
                    
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
        cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
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
        
        # Current emotion detection
        emotion_text = f"Emotion: {self.current_emotion.upper()}"
        confidence_text = f"({self.current_confidence:.1f}%)"
        cv2.putText(frame, emotion_text, (w - 350, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, confidence_text, (w - 350, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Status message (shows for 3 seconds after state change)
        if time.time() - self.message_timer < 3.0:
            # Create bottom banner for status
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h-60), (w, h), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            
            cv2.putText(frame, self.status_message, (10, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Press Q to quit", (w - 200, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return frame
    
    def run_webcam_mode(self):
        """Run real-time emotion detection with continuous analysis"""
        print("="*60)
        print("    EMOTION GAMING SYSTEM v2.0 - REAL-TIME MODE")
        print("="*60)
        print("\nFeatures:")
        print("  + Continuous emotion detection (every 2 seconds)")
        print("  + Emotion smoothing (prevents jittery changes)")
        print("  + Multi-threaded processing (no lag)")
        print("\nControls:")
        print("  Q - Quit and see final score")
        print("-" * 60)
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot access webcam")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Start detection thread
        detection_thread = threading.Thread(target=self.detection_thread, daemon=True)
        detection_thread.start()
        
        print("\nSystem ready! Detecting emotions...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update frame for analysis thread
            with self.frame_lock:
                self.frame_for_analysis = frame.copy()
            
            # Draw all game info
            display_frame = self.draw_game_info(frame)
            
            cv2.imshow('Emotion Gaming System v2.0', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
        
        # Cleanup
        self.detection_active = False
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("                      GAME OVER")
        print("="*60)
        print(f"Final Score: {self.score}")
        print(f"Final Difficulty: {self.difficulty:.1f}x")
        print(f"Emotions Detected: {len(self.emotion_buffer)}")
        print("="*60)

def main():
    print("\n" + "="*60)
    print("      EMOTION RECOGNITION GAMING SYSTEM v2.0")
    print("              Real-Time Edition")
    print("="*60)
    
    game = EmotionGamingSystem()
    game.run_webcam_mode()

if __name__ == "__main__":
    main()