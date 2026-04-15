import cv2
from deepface import DeepFace
import time
import numpy as np

class EmotionGamingSystem:
    def __init__(self):
        self.score = 0
        self.difficulty = 1.0
        self.game_state = "normal"
        
    def process_emotion(self, emotion, confidence):
        """Apply game changes based on emotion"""
        old_state = self.game_state
        
        if emotion == 'happy':
            self.score += 100
            self.difficulty = 1.3
            self.game_state = "bonus_mode"
            message = "Happy detected! +100 points, difficulty increased!"
            
        elif emotion == 'sad':
            self.score += 50
            self.difficulty = 0.7
            self.game_state = "comfort_mode"
            message = "Comfort mode activated - easier gameplay"
            
        elif emotion == 'angry':
            self.score += 150
            self.difficulty = 1.5
            self.game_state = "battle_mode"
            message = "RAGE MODE! Boss battle activated!"
            
        elif emotion == 'surprise':
            self.score += 200
            self.game_state = "mystery_mode"
            message = "SURPRISE! Mystery bonus unlocked!"
            
        else:
            self.game_state = "normal"
            message = f"Detected: {emotion}"
            
        return message
    
    def run_webcam_mode(self):
        """Run real-time emotion detection"""
        print("Starting Emotion Gaming System...")
        print("Controls:")
        print("  SPACE - Capture and detect emotion")
        print("  Q - Quit")
        print("-" * 40)
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot access webcam")
            print("Creating demo mode instead...")
            self.run_demo_mode()
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display game info on frame
            cv2.putText(frame, f"Score: {self.score}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Difficulty: {self.difficulty:.1f}x", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Mode: {self.game_state}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            
            cv2.putText(frame, "Press SPACE to detect emotion", (10, frame.shape[0]-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Emotion Gaming System', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # SPACE key
                print("\nAnalyzing emotion...")
                
                # Save frame temporarily
                cv2.imwrite('temp_capture.jpg', frame)
                
                try:
                    # Detect emotion
                    result = DeepFace.analyze('temp_capture.jpg', 
                                            actions=['emotion'], 
                                            enforce_detection=False)
                    
                    if isinstance(result, list):
                        result = result[0]
                    
                    emotion = result['dominant_emotion']
                    emotions_dict = result['emotion']
                    confidence = emotions_dict[emotion]
                    
                    print(f"\nDetected: {emotion} (confidence: {confidence:.1f}%)")
                    
                    # Process game changes
                    message = self.process_emotion(emotion, confidence)
                    print(message)
                    print(f"Current Score: {self.score}")
                    print("-" * 40)
                    
                except Exception as e:
                    print(f"Detection error: {e}")
            
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n=== GAME OVER ===")
        print(f"Final Score: {self.score}")
        
    def run_demo_mode(self):
        """Demo mode without webcam"""
        print("\n=== DEMO MODE ===")
        print("Creating test image...")
        
        # Create a simple test image
        img = np.ones((480, 640, 3), dtype=np.uint8) * 200
        cv2.putText(img, "DEMO MODE", (200, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.imwrite("demo.jpg", img)
        
        try:
            result = DeepFace.analyze("demo.jpg", 
                                    actions=['emotion'], 
                                    enforce_detection=False)
            print("DeepFace is working correctly!")
            print(f"Demo emotion: {result[0]['dominant_emotion'] if isinstance(result, list) else result['dominant_emotion']}")
        except Exception as e:
            print(f"Error: {e}")

def main():
    print("="*50)
    print("    EMOTION RECOGNITION GAMING SYSTEM")
    print("="*50)
    
    game = EmotionGamingSystem()
    
    print("\nSelect mode:")
    print("1. Webcam mode (real-time)")
    print("2. Demo mode (test without webcam)")
    
    choice = input("\nEnter choice (1 or 2): ")
    
    if choice == '1':
        game.run_webcam_mode()
    else:
        game.run_demo_mode()

if __name__ == "__main__":
    main()