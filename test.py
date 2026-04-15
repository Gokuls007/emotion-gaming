try:
    import cv2
    print("✓ OpenCV works")
except:
    print("✗ OpenCV failed")

try:
    from deepface import DeepFace
    print("✓ DeepFace works")
except Exception as e:
    print(f"✗ DeepFace failed: {e}")