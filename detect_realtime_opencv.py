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
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini API Configured")
else:
    print("⚠️ Warning: No Gemini API Key found in .env")

CLASSIFIER_MODEL_PATH = 'coffee_mobile_net_tuned.h5'
CLASSIFICATION_THRESHOLD = 0.60  # Balanced: Captures 0.62-0.69 beans as Good
VIDEO_PATH = "vid2.mp4"

# Detection settings - NO SIZE LIMIT
MIN_AREA = 50   
MAX_AREA = 999999

class BeanTracker:
    def __init__(self, bean_id, centroid, box):
        self.id = bean_id
        self.centroid = centroid
        self.box = box
        self.label = "Unknown"
        self.confidence = 0.0
        self.state = "WAITING_ENTRY"
        self.frames_lost = 0
        self.color = (100, 100, 100)
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

    # Softened, objective prompt
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

    cv2.namedWindow("Zone-Based Tracking", cv2.WINDOW_NORMAL)
    
    active_bean = None
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
                active_bean = None
                continue
            
            h_orig, w_orig = img.shape[:2]
            
            # 1. ROI CROP (Remove black bars)
            y_start = int(h_orig * 0.15)
            y_end = int(h_orig * 0.85)
            roi = img[y_start:y_end, :]
            
            # 2. CREATE MASK
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (9, 9), 0)
            _, thresh = cv2.threshold(blur, 75, 255, cv2.THRESH_BINARY_INV)
            
            kernel = np.ones((5,5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
            thresh = cv2.dilate(thresh, kernel, iterations=2)
            
            # Show mask
            cv2.imshow("Mask", thresh)

            # 3. FIND CONTOURS
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_blob = None
            max_area = 0
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if MIN_AREA < area < MAX_AREA:
                    if area > max_area:
                        max_area = area
                        x, y, w, h = cv2.boundingRect(cnt)
                        valid_blob = (x, y + y_start, w, h)

            # 4. TRACKING LOGIC
            if valid_blob:
                bx, by, bw, bh = valid_blob
                cx = bx + bw // 2
                cy = by + bh // 2
                
                touching_left = bx < 10
                touching_right = (bx + bw) > (w_orig - 10)
                
                # Teleportation Check
                if active_bean:
                     is_new_on_left = (cx < w_orig * 0.3)
                     is_old_on_right = (active_bean.centroid[0] > w_orig * 0.5)
                     if is_new_on_left and is_old_on_right:
                         print(f"*** NEW BEAN DETECTED (Override) ***")
                         active_bean = None 

                # CASE 1: No active bean
                if active_bean is None:
                    if cx < w_orig * 0.4:
                        active_bean = BeanTracker(next_id, (cx, cy), valid_blob)
                        next_id += 1
                        print(f"*** Bean {active_bean.id}: CREATED on Left ***")
                
                # CASE 2: Update active bean
                else:
                    active_bean.centroid = (cx, cy)
                    active_bean.box = valid_blob
                    active_bean.frames_lost = 0
                    
                    if active_bean.state == "WAITING_ENTRY":
                        if not touching_left:
                            print(f"*** Bean {active_bean.id}: ENTERED -> Classifying ***")
                            active_bean.state = "TRACKING"
                            
                            # CLASSIFY
                            crop = img[by:by+bh, bx:bx+bw]
                            processed = preprocess_for_mobilenet(crop)
                            if processed is not None:
                                score = classifier.predict(processed, verbose=0)[0][0]
                                if score > CLASSIFICATION_THRESHOLD:
                                    active_bean.label = "Good"
                                    active_bean.confidence = score
                                    active_bean.color = (0, 255, 0)
                                    total_good += 1
                                else:
                                    active_bean.label = "Bad"
                                    active_bean.confidence = 1 - score
                                    active_bean.color = (0, 0, 255)
                                    total_bad += 1
                                
                                # LOG DATA for REPORT
                                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                                session_data.append({
                                    "id": active_bean.id,
                                    "time": timestamp,
                                    "label": active_bean.label,
                                    "confidence": float(active_bean.confidence)
                                })
                                print(f"    Result: {active_bean.label} ({active_bean.confidence:.2f})")

                    elif active_bean.state == "TRACKING":
                        if touching_right:
                            active_bean.state = "EXITING"
                            print(f"*** Bean {active_bean.id}: EXITING Right ***")
                    
                    elif active_bean.state == "EXITING":
                        pass

            # CASE 3: NO blob
            else:
                if active_bean:
                    active_bean.frames_lost += 1
                    if active_bean.frames_lost > 10:
                        print(f"*** Bean {active_bean.id}: GONE ***")
                        active_bean = None

            # 5. DRAW
            if active_bean:
                x, y, w, h = active_bean.box
                cv2.rectangle(img, (x, y), (x+w, y+h), active_bean.color, 3)
                status_text = f"{active_bean.label} {active_bean.confidence:.2f}" if active_bean.label != "Unknown" else active_bean.state
                cv2.putText(img, status_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, active_bean.color, 2)

            cv2.putText(img, f"Good: {total_good}  Bad: {total_bad}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, "Press 'g' for AI Report", (20, h_orig - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Zone-Based Tracking", img)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'): break
        if key == ord(' '): paused = not paused
        if key == ord('r'): total_good = 0; total_bad = 0; active_bean = None; session_data = []
        if key == ord('g'): 
            generate_llm_report(session_data, start_time)
            paused = True # Pause while generating

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
