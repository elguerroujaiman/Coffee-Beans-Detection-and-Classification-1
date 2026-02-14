from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Configuration ---
YOLO_MODEL_PATH = 'best_640.pt' # Reverting to the model that detected beans better
CLASSIFIER_MODEL_PATH = 'coffee_mobile_net_tuned.h5'
DETECTION_CONFIDENCE = 0.25
CLASSIFICATION_THRESHOLD = 0.4

def preprocess_for_mobilenet(crop):
    """Preprocess cropped bean for MobileNetV2 classification."""
    img_resized = cv2.resize(crop, (224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)
    img_array = (img_array / 127.5) - 1.0
    return img_array

def main():
    # Load Models
    print("Loading models...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    classifier = load_model(CLASSIFIER_MODEL_PATH)

    # Open Webcam
    cap = cv2.VideoCapture(1)
    cap.set(3, 1280)
    cap.set(4, 720)

    actual_w = cap.get(3)
    actual_h = cap.get(4)
    print(f"Camera Resolution: {int(actual_w)}x{int(actual_h)}")

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cv2.namedWindow("Bean Counter", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Bean Counter", 1280, 720)
    print("Starting Bean Counter. Press 'q' to quit, 'r' to reset counters.")

    # Tracking & Counting
    classification_cache = {}  # {track_id: (label, confidence)}
    counted_ids = set()  # IDs that have been counted already
    good_count = 0
    bad_count = 0
    
    import time
    frame_count = 0
    fps_time = time.time()

    while True:
        success, img = cap.read()
        if not success:
            break

        h_orig, w_orig = img.shape[:2]
        inference_size = 320
        img_resized = cv2.resize(img, (inference_size, inference_size))

        # YOLO Detection
        results = yolo_model.track(img_resized, persist=True, verbose=False, 
                                   conf=DETECTION_CONFIDENCE, tracker="bytetrack.yaml")

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                
                # Scale to original frame
                scale_x = w_orig / inference_size
                scale_y = h_orig / inference_size
                
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                w, h = x2 - x1, y2 - y1
                track_id = int(box.id[0]) if box.id is not None else None

                # Skip if no tracking ID
                if track_id is None:
                    continue

                # --- 1. FILTER: Ignore "Ghost" Detections ---
                # The model mistakenly detects the whole screen as a bean because of the training data style.
                # Real beans should be small. Ignore if box is > 50% of width or height.
                if w > w_orig * 0.5 or h > h_orig * 0.5:
                    # Silently ignore ghosts (no print spam)
                    # if track_id not in counted_ids: 
                    #      print(f"⚠️ Ignoring huge ghost box (ID {track_id}): {w}x{h}")
                    continue

                # Check cache for classification
                if track_id in classification_cache:
                    label, confidence = classification_cache[track_id]
                else:
                    # New bean!
                    crop = img[max(0, y1):min(h_orig, y2), max(0, x1):min(w_orig, x2)]
                    
                    if crop.size == 0:
                        continue

                    processed_crop = preprocess_for_mobilenet(crop)
                    class_prediction = classifier.predict(processed_crop, verbose=0)[0][0]
                    
                    if class_prediction > CLASSIFICATION_THRESHOLD:
                        label = "Good"
                        confidence = class_prediction
                    else:
                        label = "Bad"
                        confidence = 1 - class_prediction
                    
                    # Cache the classification
                    classification_cache[track_id] = (label, confidence)
                    
                    # Count ONLY ONCE when first detected
                    if track_id not in counted_ids:
                        counted_ids.add(track_id)
                        if label == "Good":
                            good_count += 1
                            print(f"✅ Good bean detected! ID: {track_id} | Count: {good_count}")
                        else:
                            bad_count += 1
                            print(f"❌ Bad bean detected! ID: {track_id} | Count: {bad_count}")

                # Draw bounding box (Only for valid, small beans)
                color = (0, 255, 0) if label == "Good" else (0, 0, 255)
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5, colorR=color)
                cvzone.putTextRect(img, f'{label} {confidence:.2f}', 
                                  (max(0, x1), max(35, y1)), 
                                  scale=1, thickness=1, colorR=color, offset=5)

        # Display Counters (Top of screen)
        counter_bg_height = 80
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w_orig, counter_bg_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        # Good Counter (Left)
        cvzone.putTextRect(img, f'GOOD: {good_count}', (20, 50), 
                          scale=2, thickness=3, colorR=(0, 255, 0), 
                          colorT=(255, 255, 255), offset=10)
        
        # Bad Counter (Right)
        cvzone.putTextRect(img, f'BAD: {bad_count}', (w_orig - 250, 50), 
                          scale=2, thickness=3, colorR=(0, 0, 255), 
                          colorT=(255, 255, 255), offset=10)

        # FPS Counter
        frame_count += 1
        if frame_count % 10 == 0:
            fps = 10 / (time.time() - fps_time)
            fps_time = time.time()
        
        if frame_count >= 10:
            cvzone.putTextRect(img, f'FPS: {int(fps)}', (10, h_orig - 20), 
                              scale=1, thickness=2, colorR=(255, 255, 0))

        cv2.imshow("Bean Counter", img)

        # Key controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset counters
            good_count = 0
            bad_count = 0
            counted_ids.clear()
            classification_cache.clear()
            print("🔄 Counters reset!")

    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n📊 Final Results:")
    print(f"   Good Beans: {good_count}")
    print(f"   Bad Beans: {bad_count}")
    print(f"   Total: {good_count + bad_count}")

if __name__ == "__main__":
    main()
