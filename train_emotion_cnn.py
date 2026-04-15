"""
Train a Custom CNN for Emotion Recognition
Fast training version (~30-40 minutes)
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

print("="*70)
print("   TRAINING CUSTOM CNN FOR EMOTION RECOGNITION")
print("="*70)

# Load data from folders
def load_data_from_folders(train_dir='train', test_dir='test', max_per_class=1500):
    """Load images from train/test folders"""
    print("\nLoading training data...")
    
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    # Emotion mapping
    emotions = sorted(os.listdir(train_dir))
    emotion_map = {emotion: idx for idx, emotion in enumerate(emotions)}
    
    print(f"Emotion classes: {emotions}")
    
    # Load training data
    for emotion in emotions:
        emotion_path = os.path.join(train_dir, emotion)
        if not os.path.isdir(emotion_path):
            continue
        
        print(f"Loading {emotion}...", end=' ')
        count = 0
        
        for img_file in os.listdir(emotion_path):
            if count >= max_per_class:
                break
            
            try:
                img_path = os.path.join(emotion_path, img_file)
                img = load_img(img_path, color_mode='grayscale', target_size=(48, 48))
                img_array = img_to_array(img) / 255.0
                
                X_train.append(img_array)
                y_train.append(emotion_map[emotion])
                count += 1
            except:
                pass
        
        print(f"{count} images")
    
    # Load test data
    print("\nLoading test data...")
    for emotion in emotions:
        emotion_path = os.path.join(test_dir, emotion)
        if not os.path.isdir(emotion_path):
            continue
        
        print(f"Loading {emotion}...", end=' ')
        count = 0
        
        for img_file in os.listdir(emotion_path):
            try:
                img_path = os.path.join(emotion_path, img_file)
                img = load_img(img_path, color_mode='grayscale', target_size=(48, 48))
                img_array = img_to_array(img) / 255.0
                
                X_test.append(img_array)
                y_test.append(emotion_map[emotion])
                count += 1
            except:
                pass
        
        print(f"{count} images")
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # Convert labels to categorical
    y_train = keras.utils.to_categorical(y_train, 7)
    y_test = keras.utils.to_categorical(y_test, 7)
    
    print(f"\nTraining set: {len(X_train)} images")
    print(f"Test set: {len(X_test)} images")
    
    return X_train, y_train, X_test, y_test, emotions

# Build CNN model
def build_cnn_model():
    """Build a simple but effective CNN"""
    print("\nBuilding CNN architecture...")
    
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(7, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✓ Model built successfully!")
    model.summary()
    
    return model

# Main training function
def main():
    # Load data
    X_train, y_train, X_test, y_test, emotions = load_data_from_folders(
        max_per_class=1500  # Limit for faster training
    )
    
    # Build model
    model = build_cnn_model()
    
    # Train
    print("\n" + "="*70)
    print("   TRAINING CNN (This will take 30-40 minutes)")
    print("="*70)
    
    history = model.fit(
        X_train, y_train,
        batch_size=64,
        epochs=30,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate
    print("\n" + "="*70)
    print("   EVALUATING MODEL")
    print("="*70)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save model
    print("\nSaving model...")
    model.save('emotion_cnn.h5')
    print("✓ Model saved as: emotion_cnn.h5")
    
    print("\n" + "="*70)
    print("   TRAINING COMPLETE!")
    print("="*70)
    print(f"Final Accuracy: {test_acc*100:.2f}%")
    print("Model ready to use in emotion_game_cnn.py")
    print("="*70)

if __name__ == "__main__":
    main()