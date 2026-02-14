import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Configuration ---
CLASSIFIER_MODEL_PATH = 'coffee_mobile_net_tuned.h5'
CLASSIFICATION_THRESHOLD = 0.4  # >0.4 = Good
MIN_AREA = 1000
MAX_AREA = 50000
MIN_CIRCULARITY = 0.4

def preprocess_for_mobilenet(crop):
    """Preprocess cropped bean for MobileNetV2 classification."""
    img_resized = cv2.resize(crop, (224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)
    img_array = (img_array / 127.5) - 1.0
    return img_array

def detect_beans_opencv(img):
    """Use OpenCV to find bean contours."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(thresh)
    
    kernel = np.ones((7,7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    beans = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_AREA < area < MAX_AREA:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity > MIN_CIRCULARITY:
                x, y, w, h = cv2.boundingRect(cnt)
                beans.append((x, y, w, h))
    return beans

def main():
    # --- Test on Static Image ---
    IMAGE_PATH = 'multiple beans 2.jfif'  # Change this
    
    print("Loading Classifier...")
    classifier = load_model(CLASSIFIER_MODEL_PATH)
    
    print(f"Analyzing: {IMAGE_PATH}")
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Error: Could not read image.")
        return
    
    h_orig, w_orig = img.shape[:2]
    
    # 1. DETECT with OpenCV
    beans = detect_beans_opencv(img)
    print(f"OpenCV found {len(beans)} beans.")
    
    # 2. CLASSIFY each bean with MobileNet
    good_count = 0
    bad_count = 0
    
    for (x, y, w, h) in beans:
        crop = img[y:y+h, x:x+w]
        if crop.size == 0: continue
        
        processed_crop = preprocess_for_mobilenet(crop)
        score = classifier.predict(processed_crop, verbose=0)[0][0]
        
        if score > CLASSIFICATION_THRESHOLD:
            label = "Good"
            confidence = score
            good_count += 1
            color = (0, 255, 0)
        else:
            label = "Bad"
            confidence = 1 - score
            bad_count += 1
            color = (0, 0, 255)
        
        # Draw
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
        cv2.putText(img, f'{label} {confidence:.2f}', (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # 3. SUMMARY
    print(f"\n✅ Analysis Complete!")
    print(f"   Good: {good_count}")
    print(f"   Bad:  {bad_count}")
    print(f"   Total: {good_count + bad_count}")
    
    cv2.imshow("OpenCV + MobileNet Result", img)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
