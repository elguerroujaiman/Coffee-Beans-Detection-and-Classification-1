import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import os
import datetime
from dotenv import load_dotenv
import google.generativeai as genai

# --- Configuration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("✅ Gemini API Configured")
    except Exception as e:
        print(f"⚠️ Warning: Gemini API Error: {e}")
else:
    print("⚠️ Warning: No Gemini API Key found in .env")

CLASSIFIER_MODEL_PATH = 'coffee_mobile_net_tuned.h5'
CLASSIFICATION_THRESHOLD = 0.20  # Very lenient: Captures beans with >30% Good score
VIDEO_PATH = "multiple_beans_vid.mp4" 

# Detection settings
MIN_AREA = 1500   # INCREASED: Filter out noise/fragments (Beans are large)
MAX_AREA = 999999
TRACKING_DISTANCE_THRESHOLD = 100  # Increased for faster moving beans

class BeanTracker:
    def __init__(self, bean_id, centroid, box):
        self.id = bean_id
        self.centroid = centroid  # (cx, cy)
        self.box = box            # (x, y, w, h)
        self.label = "Unknown"
        self.confidence = 0.0
        self.state = "WAITING_ENTRY" # WAITING_ENTRY -> TRACKING -> EXITING -> GONE
        self.frames_lost = 0
        self.color = (100, 100, 100) # Gray initially
        self.timestamp = None

def preprocess_for_mobilenet(crop):
    try:
        img_resized = cv2.resize(crop, (224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = tf.expand_dims(img_array, 0)
        img_array = (img_array / 127.5) - 1.0
        return img_array
    except:
        return None

def generate_llm_report(session_data, start_time):
    if not GEMINI_API_KEY:
        print("\n❌ Cannot generate report: No API Key.")
        return

    duration = time.time() - start_time
    total = len(session_data)
    if total == 0:
        print("\n⚠️ No data to report.")
        return

    good_count = sum(1 for b in session_data if b['label'] == 'Good')
    bad_count = sum(1 for b in session_data if b['label'] == 'Bad')
    avg_conf = np.mean([b['confidence'] for b in session_data])
    
    print("\n🤖 Generating AI Report (asking Gemini)... Please wait...")

    prompt = f"""
    You are a Data Analyst for a Coffee Production Line.
    Please analyze the following session statistics objectively.
    Do not criticize the classification model usage; simply report the trends found in the data.
    
    SESSION STATS:
    - Duration: {duration:.1f} seconds
    - Total Beans Processed: {total}
    - Throughput: {total/duration*60:.1f} beans/minute
    - Good Beans: {good_count} ({(good_count/total)*100:.1f}%)
    - Bad Beans: {bad_count} ({(bad_count/total)*100:.1f}%)
    - Average Classification Confidence: {avg_conf:.2f}

    RAW DATA (First 20 items):
    {session_data[:20]}

    Please provide:
    1. A neutral summary of the yield (Good vs Bad ratio).
    2. Observations on the confidence levels (e.g., "Confidence remains high/stable").
    3. Simple operational notes based on the throughput.
    Keep the tone helpful and professional. Under 150 words.
    """

    try:
        # Use Gemini 2.5 Flash (Verified Working!)
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        response = model.generate_content(prompt)
        report_text = response.text
        
        print("\n" + "="*40)
        print("     🧠 GEMINI INSPECTION REPORT      ")
        print("="*40)
        print(report_text)
        print("="*40 + "\n")
        
        # --- SAVE TO FILE ---
        end_dt = datetime.datetime.now()
        timestamp_str = end_dt.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{timestamp_str}_G{good_count}_B{bad_count}.txt"
        
        reports_dir = "reports"
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
            
        filepath = os.path.join(reports_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("COFFEE BEAN QC REPORT\n")
            f.write("=====================\n\n")
            f.write(f"Date: {end_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Stats: Good={good_count}, Bad={bad_count}, Total={total}\n")
            f.write("-" * 30 + "\n\n")
            f.write(report_text)
            
        print(f"✅ Report saved to: {filepath}")

    except Exception as e:
        print(f"\n❌ Gemini Error: {e}")

def main():
    print("Loading Classifier...")
    try:
        classifier = load_model(CLASSIFIER_MODEL_PATH)
    except:
        print("Error: Model not found")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Error opening video")
        return

    cv2.namedWindow("Multi-Bean Tracking", cv2.WINDOW_NORMAL)
    
    # Tracking State
    active_beans = []  # List of BeanTracker objects
    next_id = 1
    total_good = 0
    total_bad = 0
    paused = False
    
    # Session Data Logging
    session_data = []
    start_time = time.time()

    while True:
        if not paused:
            success, img = cap.read()
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                active_beans = [] # Clear trackers on loop
                continue
            
            h_orig, w_orig = img.shape[:2]
            
            # 1. ROI CROP (Full Frame - No Crop)
            roi = img 
            y_start = 0
            y_end = h_orig
            
            # 2. CREATE MASK
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (9, 9), 0)
            
            # LOWER THRESHOLD to 100: IGNORE light grey background noise
            _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)
            
            kernel = np.ones((5,5), np.uint8)
            
            # 1. OPEN: Remove small background noise (white specks)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
            
            # 2. CLOSE: Fill holes inside the bean
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)
            # Remove extra dilate (close is enough)
            
            # Show mask
            cv2.imshow("Mask", thresh)

            # 3. FIND CONTOURS (Candidates)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            current_blobs = [] # List of (centroid, box) tuples
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if MIN_AREA < area < MAX_AREA:
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Convert ROI coords to Original coords
                    detected_box = (x, y + y_start, w, h)
                    cx = x + w // 2
                    cy = y + y_start + h // 2
                    current_blobs.append(( (cx, cy), detected_box ))

            # 4. MULTI-OBJECT TRACKING LOGIC (Centroid Matching)
            
            # Separate matched and unmatched
            matched_blob_indices = set()
            active_beans_to_keep = []

            # A. Match existing beans to new blobs
            for bean in active_beans:
                best_blob_idx = -1
                min_dist = float('inf')
                
                # Find closest blob
                for i, (blob_centroid, blob_box) in enumerate(current_blobs):
                    if i in matched_blob_indices: continue # Skip already matched blobs
                    
                    dist = np.linalg.norm(np.array(bean.centroid) - np.array(blob_centroid))
                    if dist < min_dist:
                        min_dist = dist
                        best_blob_idx = i
                
                # Update if match found within threshold
                if best_blob_idx != -1 and min_dist < TRACKING_DISTANCE_THRESHOLD:
                    # MATCH FOUND
                    bean.centroid = current_blobs[best_blob_idx][0]
                    bean.box = current_blobs[best_blob_idx][1]
                    bean.frames_lost = 0
                    matched_blob_indices.add(best_blob_idx)
                    active_beans_to_keep.append(bean)
                else:
                    # NO MATCH (Lost)
                    bean.frames_lost += 1
                    if bean.frames_lost <= 5: # Tolerance before removing
                        active_beans_to_keep.append(bean)
                    else:
                        print(f"*** Bean {bean.id}: GONE (Lost Track) ***")

            active_beans = active_beans_to_keep

            # B. Create NEW beans for unmatched blobs (only on LEFT)
            for i, (blob_centroid, blob_box) in enumerate(current_blobs):
                if i not in matched_blob_indices:
                    cx, cy = blob_centroid
                    # Only create if in Entry Zone (Left 30%)
                    if cx < w_orig * 0.3:
                        new_bean = BeanTracker(next_id, blob_centroid, blob_box)
                        next_id += 1
                        active_beans.append(new_bean)
                        print(f"*** Bean {new_bean.id}: CREATED on Left ***")

            # 5. STATE MACHINE & CLASSIFICATION (For each bean)
            for bean in active_beans:
                bx, by, bw, bh = bean.box
                cx, cy = bean.centroid
                
                touching_left = bx < 10
                touching_right = (bx + bw) > (w_orig - 10)

                # WAITING -> TRACKING
                if bean.state == "WAITING_ENTRY":
                    if not touching_left:
                        print(f"*** Bean {bean.id}: ENTERED -> Classifying ***")
                        bean.state = "TRACKING"
                        
                        # CLASSIFY ONCE
                        crop = img[by:by+bh, bx:bx+bw]
                        processed = preprocess_for_mobilenet(crop)
                        if processed is not None:
                            score = classifier.predict(processed, verbose=0)[0][0]
                            if score > CLASSIFICATION_THRESHOLD:
                                bean.label = "Good"
                                bean.confidence = score
                                bean.color = (0, 255, 0)
                                total_good += 1
                            else:
                                bean.label = "Bad"
                                bean.confidence = 1 - score
                                bean.color = (0, 0, 255)
                                total_bad += 1
                            
                            # LOG DATA
                            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                            session_data.append({
                                "id": bean.id,
                                "time": timestamp,
                                "label": bean.label,
                                "confidence": float(bean.confidence)
                            })
                            print(f"    Bean {bean.id} Result: {bean.label} ({bean.confidence:.2f})")

                # TRACKING -> EXITING
                elif bean.state == "TRACKING":
                    if touching_right:
                        bean.state = "EXITING"
                        print(f"*** Bean {bean.id}: EXITING Right ***")
                
                # EXITING -> GONE (Handled by frame loss mostly, but can force remove)
                elif bean.state == "EXITING":
                    pass

            # 6. DRAW
            if active_beans:
                for bean in active_beans:
                    x, y, w, h = bean.box
                    cv2.rectangle(img, (x, y), (x+w, y+h), bean.color, 3)
                    
                    status_text = f"ID:{bean.id} {bean.label}" 
                    if bean.label != "Unknown":
                        status_text += f" {bean.confidence:.2f}"
                    
                    cv2.putText(img, status_text, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, bean.color, 2)

            # Overlay
            cv2.putText(img, f"Good: {total_good}  Bad: {total_bad}", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(img, f"Active: {len(active_beans)}", (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(img, "Press 'g' for AI Report", (20, h_orig - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            cv2.imshow("Multi-Bean Tracking", img)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'): break
        if key == ord(' '): paused = not paused
        if key == ord('r'): 
            total_good = 0; total_bad = 0; active_beans = []; session_data = []
            next_id = 1
        if key == ord('g'): 
            generate_llm_report(session_data, start_time)
            paused = True 

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
