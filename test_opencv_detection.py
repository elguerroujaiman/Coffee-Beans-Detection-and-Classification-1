import cv2
import numpy as np

# --- Configuration ---
IMAGE_PATH = 'multiple beans 3.jfif'  # Change this to test other images
MIN_AREA = 1000  # Minimum bean size (pixels)
MAX_AREA = 50000 # Maximum bean size (increased for larger images)

def main():
    print(f"Reading image: {IMAGE_PATH}")
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Error: Could not read image.")
        return

    # 1. Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0) # Stronger blur to remove texture
    
    # 2. Otsu's Thresholding (Better for solid objects)
    # This automatically finds the best separation between bean and background
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # INVERT so beans are WHITE (255) and background is BLACK (0)
    thresh = cv2.bitwise_not(thresh)
    
    # 3. Morphological Closing (Essential to fill those "speckled" holes)
    # This fills gaps inside the beans to make them solid white blobs
    kernel = np.ones((7,7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    thresh = cv2.dilate(thresh, kernel, iterations=1) # beef up the size slightly

    # 3. Find Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bean_count = 0
    detections = []

    print(f"Found {len(contours)} raw contours. Filtering...")

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        print(f"Contour - Area: {area:.0f}", end="")
        
        # Filter by size
        if MIN_AREA < area < MAX_AREA:
            # Filter by shape (Beans are roughly oval/round)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: 
                print(" → Rejected (perimeter=0)")
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            print(f", Circularity: {circularity:.2f}", end="")
            
            # Beans usually have circularity > 0.5 (1.0 is perfect circle)
            if circularity > 0.4:
                print(" → ✅ ACCEPTED")
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Store detection
                detections.append((x, y, w, h))
                bean_count += 1
                
                # Draw Box
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Draw Contour (optional, just to see shape)
                cv2.drawContours(img, [cnt], -1, (255, 0, 0), 1)
            else:
                print(f" → Rejected (circularity < 0.4)")
        else:
            print(f" → Rejected (area not in {MIN_AREA}-{MAX_AREA})")

    print(f"\n✅ Valid Beans Detected: {bean_count}")

    # Show result
    cv2.imshow('OpenCV Detection Test', img)
    # Also show the threshold mask (debug view)
    cv2.imshow('Threshold Mask', thresh)
    
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
