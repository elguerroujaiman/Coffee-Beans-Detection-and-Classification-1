from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Configuration ---
YOLO_MODEL_PATH = 'best_detect.pt'
CLASSIFIER_MODEL_PATH = 'coffee_mobile_net_tuned.h5'
DETECTION_CONFIDENCE = 0.25
CLASSIFICATION_THRESHOLD = 0.4  # LOWERED from 0.5 (less strict = more "Good" predictions)

def preprocess_for_mobilenet(crop):
    """Preprocess cropped bean for MobileNetV2 classification."""
    # Resize to 224x224 (MobileNet input size)
    img_resized = cv2.resize(crop, (224, 224))
    
    # Convert to array and expand dims
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)
    
    # Normalize to [-1, 1]
    img_array = (img_array / 127.5) - 1.0
    
    return img_array, img_resized  # Return both for debugging

def main():
    # Load Models
    print("Loading models...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    classifier = load_model(CLASSIFIER_MODEL_PATH)

    # Open Webcam
    cap = cv2.VideoCapture(1)
    cap.set(3, 1280)
    cap.set(4, 720)

    cv2.namedWindow("Main Feed", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Main Feed", 1000, 600)
    cv2.namedWindow("MobileNet View", cv2.WINDOW_NORMAL)  # Debug window
    cv2.resizeWindow("MobileNet View", 400, 400)
    
    print("Camera ready. Press 'q' to quit.")

    classification_cache = {}
    latest_crop = None  # For debug view

    while True:
        success, img = cap.read()
        if not success:
            break

        h_orig, w_orig = img.shape[:2]
        inference_size = 320
        img_resized = cv2.resize(img, (inference_size, inference_size))

        results = yolo_model.track(img_resized, persist=True, verbose=False, 
                                   conf=DETECTION_CONFIDENCE, tracker="bytetrack.yaml")

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                
                scale_x = w_orig / inference_size
                scale_y = h_orig / inference_size
                
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                w, h = x2 - x1, y2 - y1
                track_id = int(box.id[0]) if box.id is not None else None

                # Check cache
                if track_id is not None and track_id in classification_cache:
                    label, confidence, raw_score = classification_cache[track_id]
                else:
                    crop = img[max(0, y1):min(h_orig, y2), max(0, x1):min(w_orig, x2)]
                    
                    if crop.size == 0:
                        continue

                    processed_crop, debug_crop = preprocess_for_mobilenet(crop)
                    class_prediction = classifier.predict(processed_crop, verbose=0)[0][0]
                    
                    raw_score = class_prediction
                    
                    if class_prediction > CLASSIFICATION_THRESHOLD:
                        label = "Good"
                        confidence = class_prediction
                    else:
                        label = "Bad"
                        confidence = 1 - class_prediction
                    
                    if track_id is not None:
                        classification_cache[track_id] = (label, confidence, raw_score)
                    
                    # Save for debug window
                    latest_crop = debug_crop.copy()
                    cv2.putText(latest_crop, f'{label}', (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if label == "Good" else (0, 0, 255), 2)
                    cv2.putText(latest_crop, f'Score: {raw_score:.3f}', (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(latest_crop, f'Thresh: {CLASSIFICATION_THRESHOLD}', (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                color = (0, 255, 0) if label == "Good" else (0, 0, 255)
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5, colorR=color)
                cvzone.putTextRect(img, f'{label} {confidence:.2f}', 
                                  (max(0, x1), max(35, y1)), 
                                  scale=1, thickness=1, colorR=color, offset=5)

        cv2.imshow("Main Feed", img)
        
        # Show what MobileNet sees
        if latest_crop is not None:
            cv2.imshow("MobileNet View", latest_crop)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
