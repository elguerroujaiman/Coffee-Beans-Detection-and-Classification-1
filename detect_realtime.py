import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Configuration ---
MODEL_PATH = 'coffee_mobile_net_tuned.h5'
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.5  # < 0.5 is Bad, > 0.5 is Good
CAMERA_ID = 1  # 0 for default webcam, change if using iPhone over USB

# Detection Parameters
MIN_AREA = 1000  # Minimum size of a bean to be detected (tune this)
DETECTION_LINE_Height = 0.5 # Relative height of the line (0.5 = middle)

def load_trained_model():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_frame(roi):
    """
    Preprocess the Region of Interest (ROI) for the MobileNetV2 model.
    """
    # Resize to model input size
    img = cv2.resize(roi, IMG_SIZE)
    # Convert to standard array
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # Expand dimensions to create a batch of 1 (1, 224, 224, 3)
    img_array = tf.expand_dims(img_array, 0)
    # Rescale to [-1, 1] as we did in training
    # Note: We used Rescaling(1./127.5, offset=-1) in the dataset API.
    # We must match that manual calculation here:
    img_array = (img_array / 127.5) - 1.0
    return img_array

import math

class CentroidTracker:
    def __init__(self, maxDisappeared=10):
        self.nextObjectID = 0
        self.objects = {} # ID -> Centroid (x, y)
        self.disappeared = {} # ID -> Frames missing
        self.counted = {} # ID -> Boolean (True if already counted)
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.counted[self.nextObjectID] = False
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.counted[objectID]

    def update(self, rects):
        # Check if list of input rects is empty
        if len(rects) == 0:
            # Mark existing objects as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        # Initialize input centroids
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # If we are not tracking any objects, register all inputs
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            # Match input centroids to existing object centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # Calculate distances between each pair
            D = []
            for i in range(len(objectIDs)):
                row = []
                for j in range(len(inputCentroids)):
                     # Euclidean distance
                     dist = math.hypot(objectCentroids[i][0] - inputCentroids[j][0],
                                       objectCentroids[i][1] - inputCentroids[j][1])
                     row.append(dist)
                D.append(row)
            D = np.array(D)

            # Find smallest values to match
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                
                # If distance is too large, don't match (e.g. new object appeared far away)
                if D[row, col] > 100: # Max pixel distance to be considered same object
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            # Deal with disappearances
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # Deal with new objects
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            for col in unusedCols:
                self.register(inputCentroids[col])

        return self.objects

def main():
    model = load_trained_model()
    if model is None: return

    cap = cv2.VideoCapture("vid2.mp4")
    tracker = CentroidTracker(maxDisappeared=10)
    
    count_good = 0
    count_bad = 0
    
    print("Starting video stream... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        height, width, _ = frame.shape
        line_y = int(height * DETECTION_LINE_Height)

        # 1. Image Processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # OTSU might be failing if lighting is poor or contrast is low.
        # Let's add a fallback or ensure MIN_AREA is respected.
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rects = []
        
        # Draw Line
        cv2.line(frame, (0, line_y), (width, line_y), (255, 0, 0), 2)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > MIN_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                
                # --- FILTERING NOISE ---
                # 1. Aspect Ratio Filter
                aspect_ratio = float(w) / h
                if aspect_ratio > 3.0 or aspect_ratio < 0.3: 
                    continue
                
                # 2. Size Filter (Ignore things that are too big to be a single bean)
                # Beans are small. Text messages are usually wide/large.
                if w > 200: # If object is wider than 200px (approx 1/3 screen), ignore it
                    continue
                if area > 20000: # If object is massive, ignore it
                    continue

                rects.append((x, y, x+w, y+h))
                # Draw bounding box for visual debug
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 1)

        # 2. Update Tracker
        objects = tracker.update(rects)

        # 3. Process Tracked Objects
        for (objectID, centroid) in objects.items():
            cx, cy = centroid
            
            # --- Counting Logic ---
            # Check if this ID has already been counted
            if not tracker.counted[objectID]:
                # Check for crossing (simple proximity for now, or track history)
                # If it's close to the line
                if abs(cy - line_y) < 15: # 15px buffer
                    
                    # We need to classify it NOW
                    # We need to find the bbox associated with this centroid to crop the image
                    # Since tracker only stores centroids, we look at rects again to find match
                    # (Simplified: Just crop around centroid)
                    
                    # Crop size 100x100 around center (or dynamic if we saved w/h)
                    # Let's approximate from original rects loop for better accuracy? 
                    # For simplicity, we create a fixed crop or map back.
                    # Mapping back:
                    best_roi = None
                    for (rx, ry, rx2, ry2) in rects:
                        if rx <= cx <= rx2 and ry <= cy <= ry2:
                            best_roi = frame[ry:ry2, rx:rx2]
                            break
                    
                    if best_roi is not None and best_roi.size > 0:
                        processed_roi = preprocess_frame(best_roi)
                        prediction = model.predict(processed_roi, verbose=0)[0][0]
                        label = "Good" if prediction > CONFIDENCE_THRESHOLD else "Bad"
                        
                        if label == "Good": count_good += 1
                        else: count_bad += 1
                        
                        tracker.counted[objectID] = True # MARK AS COUNTED
                        
                        # Visual Feedback
                        cv2.circle(frame, (cx, cy), 10, (0, 255, 255), -1)
                        print(f"Counted Bean ID {objectID}: {label}")

            # Draw ID
            text = f"ID {objectID}"
            cv2.putText(frame, text, (cx - 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display Stats
        cv2.putText(frame, f"Good: {count_good}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Bad: {count_bad}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Coffee Bean Classifier', frame)
        cv2.imshow('Debug Threshold', thresh)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
