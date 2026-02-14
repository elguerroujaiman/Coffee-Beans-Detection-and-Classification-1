from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Configuration ---
# CHANGE THIS to your image path
IMAGE_PATH = 'multiple beans 3.jfif' 

YOLO_MODEL_PATH = 'best_640.pt'
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
    print(f"Loading Detection Model: {YOLO_MODEL_PATH}...")
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
    except:
        print("❌ Error loading YOLO model. Check path.")
        return

    print(f"Loading Classifier: {CLASSIFIER_MODEL_PATH}...")
    try:
        classifier = load_model(CLASSIFIER_MODEL_PATH)
    except:
        print("❌ Error loading Classifier. Check path.")
        return

    print(f"\nAnalyzing Image: {IMAGE_PATH}...")
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"❌ Error: Could not read image '{IMAGE_PATH}'")
        return

    h_orig, w_orig = img.shape[:2]
    
    # 1. DETECT (YOLO)
    # We resize for consistency with real-time script, but you can try full size too
    inference_size = 640 
    img_resized = cv2.resize(img, (inference_size, inference_size))
    
    results = yolo_model.predict(img_resized, conf=DETECTION_CONFIDENCE, verbose=False)
    
    good_count = 0
    bad_count = 0
    detections = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            
            # Scale back to original image size
            scale_x = w_orig / inference_size
            scale_y = h_orig / inference_size
            
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            w, h = x2 - x1, y2 - y1
            
            # Logic: Crop -> Classify
            crop = img[max(0, y1):min(h_orig, y2), max(0, x1):min(w_orig, x2)]
            if crop.size == 0: continue

            # Classify
            processed_crop = preprocess_for_mobilenet(crop)
            score = classifier.predict(processed_crop, verbose=0)[0][0]
            
            if score > CLASSIFICATION_THRESHOLD:
                label = "Good"
                conf = score
                good_count += 1
                color = (0, 255, 0)
            else:
                label = "Bad"
                conf = 1 - score
                bad_count += 1
                color = (0, 0, 255)

            # Store for drawing
            detections.append({
                'label': label,
                'conf': conf,
                'box': (x1, y1, w, h),
                'color': color
            })

    # 2. DRAW
    print(f"\n✅ Analysis Complete!")
    print(f"   Total Beans Found: {len(detections)}")
    print(f"   Good: {good_count}")
    print(f"   Bad:  {bad_count}")

    for d in detections:
        x, y, w, h = d['box']
        color = d['color']
        label = d['label']
        conf = d['conf']
        
        cvzone.cornerRect(img, (x, y, w, h), l=9, rt=5, colorR=color)
        cvzone.putTextRect(img, f'{label} {conf:.2f}', (max(0, x), max(35, y)), 
                           scale=1, thickness=1, colorR=color, offset=5)

    # Show Image
    cv2.imshow("Multi-Bean Analysis", img)
    print("\nPress any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
