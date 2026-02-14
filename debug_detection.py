import cv2
import numpy as np

# Configuration
VIDEO_PATH = 'vid2.mp4'  # Change to your video
MIN_AREA = 1000
MAX_AREA = 50000
MIN_CIRCULARITY = 0.4

# Try different thresholds
THRESHOLD_VALUE = 180  # Adjust this if needed

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Could not open {VIDEO_PATH}")
    exit()

print(f"Video Resolution: {int(cap.get(3))}x{int(cap.get(4))}")
print("Controls:")
print("  'q' - Quit")
print("  '+' - Increase threshold (lighter backgrounds)")
print("  '-' - Decrease threshold (darker backgrounds)")
print(f"\nStarting threshold: {THRESHOLD_VALUE}")

while True:
    success, img = cap.read()
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
        continue
    
    # Process
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # Threshold
    _, thresh = cv2.threshold(blur, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY_INV)
    
    # Morphology
    kernel = np.ones((7,7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and draw
    bean_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if MIN_AREA < area < MAX_AREA:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            if circularity > MIN_CIRCULARITY:
                x, y, w, h = cv2.boundingRect(cnt)
                bean_count += 1
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, f'Bean {bean_count}', (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Info overlay
    cv2.putText(img, f'Threshold: {THRESHOLD_VALUE}', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(img, f'Beans: {bean_count}', (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show
    cv2.imshow("Original + Detection", img)
    cv2.imshow("Threshold Mask", thresh)
    
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+') or key == ord('='):
        THRESHOLD_VALUE = min(255, THRESHOLD_VALUE + 5)
        print(f"Threshold: {THRESHOLD_VALUE}")
    elif key == ord('-') or key == ord('_'):
        THRESHOLD_VALUE = max(0, THRESHOLD_VALUE - 5)
        print(f"Threshold: {THRESHOLD_VALUE}")

cap.release()
cv2.destroyAllWindows()
print("\nFinal threshold value:", THRESHOLD_VALUE)
