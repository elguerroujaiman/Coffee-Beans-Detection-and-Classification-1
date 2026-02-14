import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Configuration ---
# 1. Put your image path here!
IMAGE_PATH = 'testing_image4.jpg' 

# 2. Model Parameters
MODEL_PATH = 'coffee_mobile_net_tuned.h5'
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.5 

def preprocess_image(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None
    
    # Resize to model input size
    img_resized = cv2.resize(img, IMG_SIZE)
    
    # Convert to array
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    
    # Expand dims (1, 224, 224, 3)
    img_array = tf.expand_dims(img_array, 0)
    
    # Rescale to [-1, 1] (Match training!)
    img_array = (img_array / 127.5) - 1.0
    
    return img_array, img

def main():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print(f"Analyzing {IMAGE_PATH}...")
    processed_img, original_img = preprocess_image(IMAGE_PATH)
    
    if processed_img is None:
        return

    # Predict
    prediction = model.predict(processed_img, verbose=0)[0][0]
    
    # Interpret
    label = "GOOD Bean" if prediction > CONFIDENCE_THRESHOLD else "BAD Bean"
    confidence = prediction if label == "GOOD Bean" else (1 - prediction)
    
    print("\n" + "="*30)
    print(f"RESULT: {label}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"Raw Score: {prediction:.4f}")
    print("="*30)
    
    # Optional: Show the image with result
    cv2.putText(original_img, f"{label} ({confidence*100:.1f}%)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if label == "GOOD Bean" else (0, 0, 255), 2)
    cv2.imshow("Result", original_img)
    print("Press any key to close the image window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
