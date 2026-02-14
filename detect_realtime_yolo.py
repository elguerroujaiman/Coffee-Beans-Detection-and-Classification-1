from ultralytics import YOLO
import cv2
import cvzone
import math

# --- Configuration ---
MODEL_PATH = 'best_640.pt' # 640-trained model (better for distant beans!)
CONFIDENCE_THRESHOLD = 0.25  # Lower = detects distant beans better (but may get false positives)
CLASS_NAMES = ['Good', 'Bad'] # 0=Good, 1=Bad (Matches your data.yaml)

def main():
    # 1. Load the trained YOLO model
    print("Loading YOLO model...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Please make sure '{MODEL_PATH}' is in this directory!")
        return

    # 2. Open Webcam
    cap = cv2.VideoCapture(1) # Use 0 for Laptop Cam, 1 for USB if 0 fails
    cap.set(3, 1280) # Width
    cap.set(4, 720)  # Height

    actual_w = cap.get(3)
    actual_h = cap.get(4)
    print(f"Camera Resolution: {int(actual_w)}x{int(actual_h)}")

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cv2.namedWindow("YOLO Coffee Bean Detector", cv2.WINDOW_NORMAL)
    print("Starting Real-time Detection. Press 'q' to quit.")

    # Confidence smoothing (reduces flickering)
    smoothed_conf = {}  # {tracking_id: smoothed_confidence}
    SMOOTHING_FACTOR = 0.7  # Higher = more smoothing (0.5-0.9 recommended)

    while True:
        success, img = cap.read()
        if not success:
            break

        # Get original dimensions
        h_orig, w_orig = img.shape[:2]
        
        # Resize to 416x416 for faster inference (2.5x speed boost)
        # Model was trained at 640, so it still detects small beans well!
        inference_size = 416
        img_resized = cv2.resize(img, (inference_size, inference_size))

        # 3. Running Inference (Detection + Tracking) on RESIZED image
        results = model.track(img_resized, persist=True, verbose=False, tracker="bytetrack.yaml")

        # 4. Processing Results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box (in 416x416 coordinates)
                x1, y1, x2, y2 = box.xyxy[0]
                
                # Scale back to original frame size (1280x720)
                scale_x = w_orig / inference_size
                scale_y = h_orig / inference_size
                
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                w, h = x2 - x1, y2 - y1

                # Get tracking ID (if available)
                track_id = int(box.id[0]) if box.id is not None else None
                
                # Confidence (raw from model)
                conf_raw = float(box.conf[0])
                
                # Apply smoothing if we have a tracking ID
                if track_id is not None:
                    if track_id in smoothed_conf:
                        # Exponential moving average
                        smoothed_conf[track_id] = (SMOOTHING_FACTOR * smoothed_conf[track_id] + 
                                                   (1 - SMOOTHING_FACTOR) * conf_raw)
                    else:
                        smoothed_conf[track_id] = conf_raw
                    conf = smoothed_conf[track_id]
                else:
                    conf = conf_raw
                
                # Class Name
                cls = int(box.cls[0])
                if cls < len(CLASS_NAMES):
                    currentClass = CLASS_NAMES[cls]
                else:
                    currentClass = "Unknown"

                # Filter by Confidence
                if conf > CONFIDENCE_THRESHOLD:
                    # Draw Box using cvzone (looks nicer)
                    # Color: Green for Good, Red for Bad
                    color = (0, 255, 0) if currentClass == 'Good' else (0, 0, 255)
                    
                    cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5, colorR=color)
                    cvzone.putTextRect(img, f'{currentClass} {conf:.2f}', (max(0, x1), max(35, y1)), 
                                       scale=1, thickness=1, colorR=color, offset=5)

        # 5. Display on ORIGINAL frame (not resized)
        cv2.imshow("YOLO Coffee Bean Detector", img)

        # Quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
