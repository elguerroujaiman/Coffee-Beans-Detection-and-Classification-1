from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import os
import json
import base64

# --- Configuration ---
app = Flask(__name__)
CLASSIFIER_MODEL_PATH = 'coffee_mobile_net_tuned.h5'
CLASSIFICATION_THRESHOLD = float(os.getenv("CLASSIFICATION_THRESHOLD", "0.60"))
MIN_AREA = 500
MAX_AREA = 500000
MIN_ASPECT_RATIO = 0.35
MAX_ASPECT_RATIO = 3.0
MIN_EXTENT = 0.20
MIN_CIRCULARITY = 0.12

print("--- Starting Inference Server ---")
print("Loading Model...")
try:
    model = tf.keras.models.load_model(CLASSIFIER_MODEL_PATH)
    print("Model loaded and ready.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def overlaps(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax+aw < bx or bx+bw < ax or ay+ah < by or by+bh < ay)

def is_valid_contour(cnt):
    area = cv2.contourArea(cnt)
    if area < MIN_AREA or area > MAX_AREA:
        return None

    x, y, w, h = cv2.boundingRect(cnt)
    if w <= 0 or h <= 0:
        return None

    aspect_ratio = w / float(h)
    if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
        return None

    extent = area / float(w * h)
    if extent < MIN_EXTENT:
        return None

    perimeter = cv2.arcLength(cnt, True)
    if perimeter <= 0:
        return None

    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    if circularity < MIN_CIRCULARITY:
        return None

    return x, y, w, h

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    data = request.json
    image_path = data.get('image_path')
    
    if not image_path or not os.path.exists(image_path):
        return jsonify({"error": "Invalid image path"}), 400

    # 1. Read Image
    img = cv2.imread(image_path)
    if img is None:
        return jsonify({"error": "Failed to read image"}), 400

    output = img.copy()
    good = 0
    bad = 0
    
    # 2. Detection Logic
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask_fill_ratio = cv2.countNonZero(thresh) / float(thresh.size)
    if mask_fill_ratio < 0.01 or mask_fill_ratio > 0.85:
        thresh = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            7
        )
    
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    used_boxes = []
    for cnt in contours:
        extracted = is_valid_contour(cnt)
        if extracted is None:
            continue
        x, y, w, h = extracted
        current_box = (x, y, w, h)
        if any(overlaps(current_box, ub) for ub in used_boxes):
            continue
        used_boxes.append(current_box)
        
        bean_crop = img[y:y+h, x:x+w]
        if bean_crop.size == 0:
            continue

        # Classify
        img_resized = cv2.resize(bean_crop, (224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = tf.expand_dims(img_array, 0)
        img_array = (img_array / 127.5) - 1.0
        
        pred = model.predict(img_array, verbose=0)
        
        if pred.shape[-1] == 1:
            score = float(pred[0][0])
            label = "Good" if score >= CLASSIFICATION_THRESHOLD else "Bad"
            conf = score if label == "Good" else 1 - score
        else:
            cls = int(np.argmax(pred))
            label = "Good" if cls == 1 else "Bad"
            conf = float(pred[0][cls])

        color = (0, 255, 0) if label == "Good" else (0, 0, 255)
        if label == "Good": good += 1
        else: bad += 1
            
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        cv2.putText(output, f"{label} {conf*100:.1f}%", (x, max(y - 10, 20)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 3. Encode result
    _, detect_encoded = cv2.imencode('.jpg', output)
    detect_b64 = base64.b64encode(detect_encoded).decode('utf-8')

    return jsonify({
        "status": "success",
        "good": good,
        "bad": bad,
        "total": good + bad,
        "detected_b64": detect_b64
    })

if __name__ == "__main__":
    app.run(port=5001, debug=False)
