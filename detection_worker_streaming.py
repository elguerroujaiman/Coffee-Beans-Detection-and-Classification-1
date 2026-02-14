import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import os
import json
import datetime
import signal
import sys
import threading
from dotenv import load_dotenv
import google.generativeai as genai
import database

# --- Configuration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# DON'T configure Gemini here - only when generating report (saves API calls)

CLASSIFIER_MODEL_PATH = 'coffee_mobile_net_tuned.h5'
CLASSIFICATION_THRESHOLD = 0.20
VIDEO_PATH = "multiple_beans_vid.mp4" 
LIVE_STATS_FILE = "live_stats.json"

# Detection settings
MIN_AREA = 1500   
MAX_AREA = 999999
TRACKING_DISTANCE_THRESHOLD = 100

# Global variables for streaming
current_frame = None
frame_lock = threading.Lock()
is_running = False

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

def update_live_stats(good, bad, active, status="running"):
    """Write current stats to JSON file for web dashboard."""
    stats = {
        "status": status,
        "good_count": good,
        "bad_count": bad,
        "active_beans": active,
        "total": good + bad,
        "timestamp": datetime.datetime.now().isoformat()
    }
    with open(LIVE_STATS_FILE, 'w') as f:
        json.dump(stats, f)

def generate_llm_report(session_data, start_time):
    if not GEMINI_API_KEY:
        return None

    duration = time.time() - start_time
    total = len(session_data)
    if total == 0:
        return None

    good_count = sum(1 for b in session_data if b['label'] == 'Good')
    bad_count = sum(1 for b in session_data if b['label'] == 'Bad')
    avg_conf = np.mean([b['confidence'] for b in session_data])
    
    prompt = f"""
    You are a Data Analyst for a Coffee Production Line.
    Analyze these session statistics objectively.
    
    STATS:
    - Duration: {duration:.1f}s
    - Total: {total}, Good: {good_count}, Bad: {bad_count}
    - Avg Confidence: {avg_conf:.2f}
    
    Provide: Yield summary, confidence observations, operational notes. Under 150 words.
    """

    try:
        # Configure Gemini ONLY when generating report (saves startup time)
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('models/gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini Error: {e}")
        return f"Session completed. Good: {good_count}, Bad: {bad_count}"

def save_session_to_db(session_data, start_time):
    """Save session to database and update live stats."""
    update_live_stats(0, 0, 0, "stopped")
    
    if len(session_data) == 0:
        print("No data to save.")
        return
    
    print("\n🤖 Generating report...")
    report_text = generate_llm_report(session_data, start_time)
    
    if report_text:
        duration = time.time() - start_time
        good_count = sum(1 for b in session_data if b['label'] == 'Good')
        bad_count = sum(1 for b in session_data if b['label'] == 'Bad')
        avg_conf = np.mean([b['confidence'] for b in session_data])
        
        # Save to database
        session_id = database.save_session(
            timestamp=datetime.datetime.now(),
            total=len(session_data),
            good=good_count,
            bad=bad_count,
            avg_conf=avg_conf,
            duration=duration,
            report=report_text
        )
        print(f"✅ Session saved to database (ID: {session_id})")

def main():
    global current_frame, is_running
    
    # Initialize database
    database.init_db()
    
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
    
    active_beans = []
    next_id = 1
    total_good = 0
    total_bad = 0
    
    session_data = []
    start_time = time.time()
    last_stats_update = time.time()
    
    # Signal handler for graceful shutdown (Ctrl+C or Flask stop)
    def signal_handler(sig, frame):
        print("\n⚠️ Shutdown signal received. Saving session...")
        save_session_to_db(session_data, start_time)
        cap.release()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initial stats
    update_live_stats(0, 0, 0, "running")
    is_running = True

    while is_running:
        success, img = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            active_beans = []
            continue
        
        h_orig, w_orig = img.shape[:2]
        
        roi = img 
        y_start = 0
        y_end = h_orig
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)
        
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_blobs = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if MIN_AREA < area < MAX_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                detected_box = (x, y + y_start, w, h)
                cx = x + w // 2
                cy = y + y_start + h // 2
                current_blobs.append(( (cx, cy), detected_box ))

        matched_blob_indices = set()
        active_beans_to_keep = []

        for bean in active_beans:
            best_blob_idx = -1
            min_dist = float('inf')
            
            for i, (blob_centroid, blob_box) in enumerate(current_blobs):
                if i in matched_blob_indices: continue
                
                dist = np.linalg.norm(np.array(bean.centroid) - np.array(blob_centroid))
                if dist < min_dist:
                    min_dist = dist
                    best_blob_idx = i
            
            if best_blob_idx != -1 and min_dist < TRACKING_DISTANCE_THRESHOLD:
                bean.centroid = current_blobs[best_blob_idx][0]
                bean.box = current_blobs[best_blob_idx][1]
                bean.frames_lost = 0
                matched_blob_indices.add(best_blob_idx)
                active_beans_to_keep.append(bean)
            else:
                bean.frames_lost += 1
                if bean.frames_lost <= 5:
                    active_beans_to_keep.append(bean)

        active_beans = active_beans_to_keep

        for i, (blob_centroid, blob_box) in enumerate(current_blobs):
            if i not in matched_blob_indices:
                cx, cy = blob_centroid
                if cx < w_orig * 0.3:
                    new_bean = BeanTracker(next_id, blob_centroid, blob_box)
                    next_id += 1
                    active_beans.append(new_bean)

        for bean in active_beans:
            bx, by, bw, bh = bean.box
            cx, cy = bean.centroid
            
            touching_left = bx < 10
            touching_right = (bx + bw) > (w_orig - 10)

            if bean.state == "WAITING_ENTRY":
                if not touching_left:
                    bean.state = "TRACKING"
                    
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
                        
                        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                        session_data.append({
                            "id": bean.id,
                            "time": timestamp,
                            "label": bean.label,
                            "confidence": float(bean.confidence)
                        })

            elif bean.state == "TRACKING":
                if touching_right:
                    bean.state = "EXITING"

        if active_beans:
            for bean in active_beans:
                x, y, w, h = bean.box
                cv2.rectangle(img, (x, y), (x+w, y+h), bean.color, 3)
                
                status_text = f"ID:{bean.id} {bean.label}" 
                if bean.label != "Unknown":
                    status_text += f" {bean.confidence:.2f}"
                
                cv2.putText(img, status_text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, bean.color, 2)

        cv2.putText(img, f"Good: {total_good}  Bad: {total_bad}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, f"Active: {len(active_beans)}", (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Update live stats every 0.5s
        if time.time() - last_stats_update > 0.5:
            update_live_stats(total_good, total_bad, len(active_beans), "running")
            last_stats_update = time.time()
        
        # Update global frame for streaming
        with frame_lock:
            current_frame = img.copy()
        
        time.sleep(0.03)  # ~30 FPS

    # Clean exit (only reached if 'q' pressed)
    save_session_to_db(session_data, start_time)
    cap.release()

def get_current_frame():
    """Get the current frame for MJPEG streaming."""
    global current_frame
    with frame_lock:
        if current_frame is not None:
            return current_frame.copy()
    return None

def stop_detection():
    """Stop the detection loop."""
    global is_running
    is_running = False

if __name__ == "__main__":
    main()
