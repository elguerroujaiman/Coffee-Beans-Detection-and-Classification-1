from ultralytics import YOLO
import cv2

# --- Configuration ---
# 1. Put your image path here!
IMAGE_PATH = 'testing_image5.jpg'

# 2. Model Parameters
MODEL_PATH = 'best_640.pt'
CONFIDENCE_THRESHOLD = 0.4
CLASS_NAMES = ['Good', 'Bad']

def main():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print(f"Analyzing {IMAGE_PATH}...")
    img = cv2.imread(IMAGE_PATH)
    
    if img is None:
        print(f"Error: Could not read image at {IMAGE_PATH}")
        return

    # Run inference
    results = model.predict(img, verbose=False)
    
    # Process results
    detections_found = False
    for r in results:
        boxes = r.boxes
        if len(boxes) == 0:
            print("\n❌ No beans detected in this image!")
            continue
            
        print(f"\n✅ Found {len(boxes)} detection(s):\n")
        detections_found = True
        
        for i, box in enumerate(boxes):
            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get confidence and class
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            if cls < len(CLASS_NAMES):
                label = CLASS_NAMES[cls]
            else:
                label = f"Unknown (class {cls})"
            
            print(f"  Detection {i+1}:")
            print(f"    Class: {label}")
            print(f"    Confidence: {conf*100:.2f}%")
            print(f"    Box: ({x1}, {y1}) to ({x2}, {y2})")
            
            # Draw on image
            if conf > CONFIDENCE_THRESHOLD:
                color = (0, 255, 0) if label == 'Good' else (0, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f'{label} {conf:.2f}', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                print(f"    (Below confidence threshold {CONFIDENCE_THRESHOLD})")
    
    if detections_found:
        # Show result
        cv2.imshow("YOLO Test Result", img)
        print("\nPress any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("\nTry:")
        print("1. Using a clearer image")
        print("2. Lowering CONFIDENCE_THRESHOLD in the script")

if __name__ == '__main__':
    main()
