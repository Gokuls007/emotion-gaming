import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

print("="*60)
print("   TRAINING EMOTION CNN")
print("="*60)

# Load data
def load_data():
    X_train, y_train, X_test, y_test = [], [], [], []
    
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    print("\nLoading training data...")
    for idx, emotion in enumerate(emotions):
        path = f'train/{emotion}'
        count = 0
        for img_name in os.listdir(path):
            if count >= 1000:  # Limit for faster training
                break
            try:
                img = load_img(f'{path}/{img_name}', color_mode='grayscale', target_size=(48,48))
                arr = img_to_array(img) / 255.0
                X_train.append(arr)
                y_train.append(idx)
                count += 1
            except:
                pass
        print(f"  {emotion}: {count} images")
    
    print("\nLoading test data...")
    for idx, emotion in enumerate(emotions):
        path = f'test/{emotion}'
        count = 0
        for img_name in os.listdir(path):
            try:
                img = load_img(f'{path}/{img_name}', color_mode='grayscale', target_size=(48,48))
                arr = img_to_array(img) / 255.0
                X_test.append(arr)
                y_test.append(idx)
                count += 1
            except:
                pass
        print(f"  {emotion}: {count} images")
    
    from tensorflow.keras.utils import to_categorical
    return (np.array(X_train), to_categorical(y_train, 7),
            np.array(X_test), to_categorical(y_test, 7))

# Build model
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(48,48,1)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),
    
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.25),
    
    Conv2D(256, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.4),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\nModel architecture:")
model.summary()

# Train
X_train, y_train, X_test, y_test = load_data()

print("\n" + "="*60)
print("   TRAINING (30 epochs, ~25-30 minutes)")
print("="*60)

model.fit(X_train, y_train, batch_size=64, epochs=30,
         validation_data=(X_test, y_test), verbose=1)

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {acc*100:.2f}%")

# Save
model.save('my_emotion_cnn.h5')
print("\n✓ Model saved as: my_emotion_cnn.h5")
print("="*60)