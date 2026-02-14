from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Configuration ---
YOLO_MODEL_PATH = 'best_detect.pt'  # Detection-only YOLO
CLASSIFIER_MODEL_PATH = 'coffee_mobile_net_tuned.h5'  # MobileNetV2 for classification
DETECTION_CONFIDENCE = 0.25  # Lower threshold for detecting beans
CLASSIFICATION_THRESHOLD = 0.4  # MobileNet: >0.4 = Good, ≤0.4 = Bad (Lowered from 0.5 for better accuracy)

def preprocess_for_mobilenet(crop):
    """Preprocess cropped bean for MobileNetV2 classification."""
    # Resize to 224x224 (MobileNet input size)
    img_resized = cv2.resize(crop, (224, 224))
    
    # Convert to array and expand dims
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)
    
    # Normalize to [-1, 1] (Match training preprocessing)
    img_array = (img_array / 127.5) - 1.0
    
    return img_array

def main():
    # 1. Load Models
    print("Loading YOLO detection model...")
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        print(f"Error loading YOLO: {e}")
        return

    print("Loading MobileNetV2 classification model...")
    try:
        classifier = load_model(CLASSIFIER_MODEL_PATH)
    except Exception as e:
        print(f"Error loading classifier: {e}")
        return

    # 2. Open Webcam
    cap = cv2.VideoCapture(1)
    cap.set(3, 1280)
    cap.set(4, 720)

    actual_w = cap.get(3)
    actual_h = cap.get(4)
    print(f"Camera Resolution: {int(actual_w)}x{int(actual_h)}")

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cv2.namedWindow("Hybrid Bean Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hybrid Bean Detector", 1280, 720)  # Set window size
    print("Starting Hybrid Detection. Press 'q' to quit.")
    print("YOLO detects → MobileNet classifies")

    # Smoothing & Caching
    classification_cache = {}  # {track_id: (label, confidence)}
    frame_count = 0
    import time
    fps_time = time.time()

    while True:
        success, img = cap.read()
        if not success:
            break

        h_orig, w_orig = img.shape[:2]
        
        # Resize for faster YOLO inference (Reduced to 320 for speed!)
        inference_size = 320
        img_resized = cv2.resize(img, (inference_size, inference_size))

        # YOLO Detection (finds beans)
        results = yolo_model.track(img_resized, persist=True, verbose=False, 
                                   conf=DETECTION_CONFIDENCE, tracker="bytetrack.yaml")

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates (in 320x320)
                x1, y1, x2, y2 = box.xyxy[0]
                
                # Scale back to original frame
                scale_x = w_orig / inference_size
                scale_y = h_orig / inference_size
                
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                w, h = x2 - x1, y2 - y1
                
                # Get tracking ID
                track_id = int(box.id[0]) if box.id is not None else None

                # Check cache first (HUGE speed boost!)
                if track_id is not None and track_id in classification_cache:
                    label, confidence = classification_cache[track_id]
                else:
                    # Only run MobileNet for NEW beans
                    crop = img[max(0, y1):min(h_orig, y2), max(0, x1):min(w_orig, x2)]
                    
                    if crop.size == 0:
                        continue

                    # MobileNet Classification
                    processed_crop = preprocess_for_mobilenet(crop)
                    class_prediction = classifier.predict(processed_crop, verbose=0)[0][0]
                    
                    # Interpret result
                    if class_prediction > CLASSIFICATION_THRESHOLD:
                        label = "Good"
                        confidence = class_prediction
                    else:
                        label = "Bad"
                        confidence = 1 - class_prediction
                    
                    # Cache it
                    if track_id is not None:
                        classification_cache[track_id] = (label, confidence)

                # Draw results
                color = (0, 255, 0) if label == "Good" else (0, 0, 255)
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5, colorR=color)
                cvzone.putTextRect(img, f'{label} {confidence:.2f}', 
                                  (max(0, x1), max(35, y1)), 
                                  scale=1, thickness=1, colorR=color, offset=5)

        # FPS Counter
        frame_count += 1
        if frame_count % 10 == 0:
            fps = 10 / (time.time() - fps_time)
            fps_time = time.time()
            cvzone.putTextRect(img, f'FPS: {int(fps)}', (10, 30), 
                              scale=1.5, thickness=2, colorR=(255, 255, 0))

        cv2.imshow("Hybrid Bean Detector", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
