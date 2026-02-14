import cv2
import numpy as np
import tensorflow as tf
import sys
import json
import os

# --- Configuration ---
CLASSIFIER_MODEL_PATH = 'coffee_mobile_net_tuned.h5'

def main():
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: python detect_image_worker.py <input_path> <output_path>"}))
        return

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # 1. Load Model
    try:
        # Use standard loading - TF 2.18 handles Keras 3 models natively
        classifier = tf.keras.models.load_model(CLASSIFIER_MODEL_PATH)
    except Exception as e:
        print(json.dumps({"error": f"Failed to load model: {str(e)}"}))
        return

    # 2. Read Image
    img = cv2.imread(input_path)
    if img is None:
        print(json.dumps({"error": "Failed to read input image"}))
        return

    # 3. Detection Logic (Robust OTSU + Circularity)
    output = img.copy()
    good = 0
    bad = 0
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(thresh)
    
    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    used_boxes = []
    def overlaps(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return not (ax+aw < bx or bx+bw < ax or ay+ah < by or by+bh < ay)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500 or area > 500000: continue
        
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < 0.3: continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        current_box = (x, y, w, h)
        if any(overlaps(current_box, ub) for ub in used_boxes): continue
        used_boxes.append(current_box)
        
        bean_crop = img[y:y+h, x:x+w]
        if bean_crop.size == 0: continue

        # Classify
        img_resized = cv2.resize(bean_crop, (224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = tf.expand_dims(img_array, 0)
        img_array = (img_array / 127.5) - 1.0
        
        pred = classifier.predict(img_array, verbose=0)
        
        if pred.shape[-1] == 1:
            score = float(pred[0][0])
            label = "Good" if score > 0.5 else "Bad"
            conf = score if label == "Good" else 1 - score
        else:
            cls = int(np.argmax(pred))
            label = "Good" if cls == 1 else "Bad"
            conf = float(pred[0][cls])

        if label == "Good":
            good += 1
            color = (0, 255, 0)
        else:
            bad += 1
            color = (0, 0, 255)
            
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        cv2.putText(output, f"{label} {conf*100:.1f}%", (x, max(y - 10, 20)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 4. Save results
    cv2.imwrite(output_path, output)
    
    # 5. Output Result in JSON (No emojis to prevent encoding errors)
    print(json.dumps({
        "status": "success",
        "good": good,
        "bad": bad,
        "total": good + bad
    }))

if __name__ == "__main__":
    main()
