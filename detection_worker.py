import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import argparse
import os
import json
import datetime
import signal
import sys
import threading
from flask import Flask, Response
import logging

# Mute Flask startup
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

from dotenv import load_dotenv
import google.generativeai as genai
import database

# --- Windows Environment Fix ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# --- Configuration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# DON'T configure Gemini here - only when generating report (saves API calls)

CLASSIFIER_MODEL_PATH = 'coffee_mobile_net_tuned.h5'
CLASSIFICATION_THRESHOLD = float(os.getenv("CLASSIFICATION_THRESHOLD", "0.60"))
# Detection settings
MIN_AREA = 500
MAX_AREA = 500000
MIN_ASPECT_RATIO = 0.35
MAX_ASPECT_RATIO = 3.0
MIN_EXTENT = 0.20
MIN_CIRCULARITY = 0.12
TRACKING_DISTANCE_THRESHOLD = 100
ENTRY_REGION_RATIO = 0.30
EDGE_MARGIN = 10
MIN_TRACK_IOU = 0.03
MAX_FRAMES_LOST = 6
MIN_VISIBLE_FRAMES = int(os.getenv("MIN_VISIBLE_FRAMES", "2"))
MIN_STABLE_FRAMES = int(os.getenv("MIN_STABLE_FRAMES", "2"))
BOX_PADDING = 0
LIVE_STATS_FILE = "live_stats.json"
WORKER_RESULT_FILE = "worker_result.json"
STREAM_PORT = int(os.getenv("STREAM_PORT", "5002"))
DEBUG_OUTPUT_BASE = os.getenv("DEBUG_OUTPUT_BASE", "debug_outputs")
DEBUG_WINDOW_NAME = "BeanGuard Debug"

def env_enabled(name, default="0"):
    value = str(os.getenv(name, default)).strip().lower()
    return value in {"1", "true", "yes", "on"}

# Streaming Globals
output_lock = threading.Lock()
current_frame = None
stream_client_event = threading.Event()
app = Flask(__name__)

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/health')
def health():
    return ("ok", 200)

def generate():
    global current_frame
    # Mark that a browser is actively consuming the MJPEG stream.
    stream_client_event.set()
    while True:
        try:
            with output_lock:
                if current_frame is None:
                    # Create a placeholder loading frame
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    frame_to_send = placeholder
                else:
                    frame_to_send = current_frame

                flag, encoded_image = cv2.imencode(".jpg", frame_to_send)
                if not flag:
                    time.sleep(0.03)
                    continue

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +
                bytearray(encoded_image) +
                b'\r\n'
            )
            time.sleep(0.1 if current_frame is None else 0.03)
        except GeneratorExit:
            break
        except Exception:
            # Keep stream handler alive even if one frame fails.
            time.sleep(0.05)

def run_flask():
    print(f"Starting Flask MJPEG server on port {STREAM_PORT}...")
    try:
        # Local-only binding keeps the stream private and avoids firewall prompts.
        app.run(host='127.0.0.1', port=STREAM_PORT, debug=False, threaded=True, use_reloader=False)
    except Exception as e:
        print(f"Flask stream server failed: {e}")


def parse_worker_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("source", nargs="?", default="0")
    parser.add_argument("--stream-port", type=int, default=STREAM_PORT)
    parser.add_argument("--user-id", type=int, default=0)
    args, _ = parser.parse_known_args()
    return args


_worker_args = parse_worker_args()
STREAM_PORT = _worker_args.stream_port
SESSION_USER_ID = _worker_args.user_id if _worker_args.user_id > 0 else None
_source = str(_worker_args.source)
# Convert webcam indexes like "0", "1", ...
VIDEO_PATH = int(_source) if _source.isdigit() else _source

class BeanTracker:
    def __init__(self, bean_id, centroid, box):
        self.id = bean_id
        self.centroid = centroid
        self.box = box
        self.label = "Unknown"
        self.confidence = 0.0
        self.state = "WAITING_ENTRY"
        self.frames_lost = 0
        self.frames_seen = 1
        self.stable_frames = 0
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

def box_iou(box_a, box_b):
    """Compute IoU for two bounding boxes in (x, y, w, h) format."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    union_area = (aw * ah) + (bw * bh) - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def overlaps(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw < bx or bx + bw < ax or ay + ah < by or by + bh < ay)

def extract_blob(cnt):
    """Return (x, y, w, h) for bean-like contours."""
    area = cv2.contourArea(cnt)
    if area < MIN_AREA or area > MAX_AREA:
        return None

    x, y, w, h = cv2.boundingRect(cnt)
    if w <= 0 or h <= 0:
        return None

    aspect_ratio = w / float(h)
    if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
        return None

    extent = area / float(w * h)
    if extent < MIN_EXTENT:
        return None

    perimeter = cv2.arcLength(cnt, True)
    if perimeter <= 0:
        return None

    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    if circularity < MIN_CIRCULARITY:
        return None

    return x, y, w, h

def build_debug_visualization(frame, thresh, current_blobs, active_beans, mask_fill_ratio, total_good, total_bad):
    """Compose a 2x2 debug board showing detection/segmentation internals."""
    candidates_view = frame.copy()
    for idx, (_, box) in enumerate(current_blobs):
        x, y, w, h = box
        cv2.rectangle(candidates_view, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(
            candidates_view,
            f"C{idx + 1}",
            (x, max(y - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )

    tracks_view = frame.copy()
    for bean in active_beans:
        x, y, w, h = bean.box
        color = bean.color if bean.label != "Unknown" else (130, 130, 130)
        cv2.rectangle(tracks_view, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            tracks_view,
            f"ID{bean.id}:{bean.label}",
            (x, max(y - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    mask_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    isolated = cv2.bitwise_and(frame, frame, mask=thresh)

    panel_w, panel_h = 480, 270
    top_left = cv2.resize(candidates_view, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
    top_right = cv2.resize(mask_bgr, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)
    bottom_left = cv2.resize(isolated, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
    bottom_right = cv2.resize(tracks_view, (panel_w, panel_h), interpolation=cv2.INTER_AREA)

    board = np.zeros((panel_h * 2, panel_w * 2, 3), dtype=np.uint8)
    board[:panel_h, :panel_w] = top_left
    board[:panel_h, panel_w:] = top_right
    board[panel_h:, :panel_w] = bottom_left
    board[panel_h:, panel_w:] = bottom_right

    cv2.putText(board, "Candidates", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(board, "Mask", (panel_w + 12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(board, "Isolated", (12, panel_h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)
    cv2.putText(board, "Tracked", (panel_w + 12, panel_h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 0), 2)
    cv2.putText(
        board,
        f"mask_fill={mask_fill_ratio:.3f}  good={total_good}  bad={total_bad}  active={len(active_beans)}",
        (12, board.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )
    return board

def classify_bean_once(classifier, img, box, bean_id=None, frame_idx=None, debug_dir=None):
    """Classify a bean crop exactly once and return (label, confidence, color)."""
    h_img, w_img = img.shape[:2]
    x, y, w, h = box

    x1 = max(0, x - BOX_PADDING)
    y1 = max(0, y - BOX_PADDING)
    x2 = min(w_img, x + w + BOX_PADDING)
    y2 = min(h_img, y + h + BOX_PADDING)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    try:
        resized = cv2.resize(crop, (224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(resized)
        img_array = tf.expand_dims(img_array, 0)
        processed = (img_array / 127.5) - 1.0
    except Exception:
        return None

    score = float(classifier.predict(processed, verbose=0)[0][0])
    if score >= CLASSIFICATION_THRESHOLD:
        label, confidence, color = "Good", score, (0, 255, 0)
    else:
        label, confidence, color = "Bad", 1.0 - score, (0, 0, 255)

    if debug_dir:
        safe_label = label.lower()
        safe_score = f"{score:.4f}".replace(".", "p")
        bean_str = str(bean_id) if bean_id is not None else "na"
        frame_str = str(frame_idx) if frame_idx is not None else "na"
        base_name = f"bean_{bean_str}_frame_{frame_str}_{safe_label}_{safe_score}"
        try:
            os.makedirs(debug_dir, exist_ok=True)
            raw_path = os.path.join(debug_dir, f"{base_name}_raw.jpg")
            input_path = os.path.join(debug_dir, f"{base_name}_input224.jpg")
            meta_path = os.path.join(debug_dir, f"{base_name}.json")
            cv2.imwrite(raw_path, crop)
            cv2.imwrite(input_path, resized)
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump(
                    {
                        "bean_id": bean_id,
                        "frame_idx": frame_idx,
                        "box": [int(x), int(y), int(w), int(h)],
                        "threshold": float(CLASSIFICATION_THRESHOLD),
                        "raw_score_good": float(score),
                        "predicted_label": label,
                        "confidence": float(confidence),
                    },
                    mf,
                    indent=2,
                )
        except OSError:
            pass

    return label, confidence, color

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
    tmp_file = f"{LIVE_STATS_FILE}.tmp"
    for _ in range(4):
        try:
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f)
            os.replace(tmp_file, LIVE_STATS_FILE)
            return
        except PermissionError:
            # Browser polling can briefly lock the file on Windows.
            time.sleep(0.03)
        except OSError:
            time.sleep(0.03)
    # Last-resort fallback to keep worker alive.
    try:
        with open(LIVE_STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(stats, f)
    except OSError:
        pass

def write_worker_result(saved, reason, session_id=None):
    payload = {
        "saved": bool(saved),
        "reason": reason,
        "session_id": session_id,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    try:
        with open(WORKER_RESULT_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except OSError:
        pass

def generate_llm_report(session_data, start_time):
    if not GEMINI_API_KEY:
        return "Gemini API Key missing. Please check your .env file."

    duration = time.time() - start_time
    total = len(session_data)
    if total == 0:
        return "No beans detected during this session."

    good_count = sum(1 for b in session_data if b['label'] == 'Good')
    bad_count = sum(1 for b in session_data if b['label'] == 'Bad')
    good_pct = (good_count / total * 100)
    bad_pct = (bad_count / total * 100)
    avg_conf = np.mean([b['confidence'] for b in session_data])
    
    prompt = f"""
    You are an AI Quality Control Specialist for an industrial Coffee Bean processing plant.
    Your task is to provide a professional, data-driven Quality Analysis Report for the latest batch.
    
    SESSION METRICS:
    - Duration: {duration:.1f} seconds
    - Total Volume Scanned: {total} beans
    - Good Quality: {good_count} ({good_pct:.1f}%)
    - Defective Found: {bad_count} ({bad_pct:.1f}%)
    - Average Detection Confidence: {avg_conf:.2%}
    
    REPORT STRUCTURE:
    1. **Overview**: Executive summary of the batch quality.
    2. **Defect Analysis**: Observations based on the defective bean percentage.
    3. **Confidence Summary**: A brief note on the consistency of the detection results.
    4. **Operational Recommendation**: Actionable advice for the plant floor (e.g., "Batch approved" or "Halt line for manual sorting").

    IMPORTANT: Focus strictly on the coffee bean data and batch quality. 
    DO NOT suggest model enhancements, stricter thresholding, or any technical changes to the AI detection system itself.
    DO NOT add any disclaimer or meta-comment about AI availability, API limits, or missing insights.
    Keep the report professional, technical, and under 250 words.
    """

    try:
        # Configure Gemini ONLY when generating report
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini Error (Quota or 404): {e}")
        # Return a structured, professional-looking local report as fallback
        return f"""BATCH QUALITY REPORT

1. Executive Summary
A session of {duration:.1f} seconds was completed. A total of {total} beans were processed.

2. Quality Breakdown
- Good Beans: {good_count} ({good_pct:.1f}%)
- Defective Beans: {bad_count} ({bad_pct:.1f}%)

3. Detection Confidence
The average detection confidence for this batch was {avg_conf:.2%}.

4. Operational Recommendation
{"Batch quality is high. Proceed with standard processing." if bad_pct < 15 else "High defect rate detected. Manual inspection or recalibration recommended."}
"""

def save_session_to_db(session_data, start_time, user_id):
    """Save session to database and update live stats."""
    # Ensure we don't save empty/crash sessions twice
    if not hasattr(save_session_to_db, "completed"):
        save_session_to_db.completed = True
    else:
        return

    update_live_stats(0, 0, 0, "stopped")
    
    if len(session_data) == 0:
        print("No data to save.")
        write_worker_result(saved=False, reason="no_beans")
        return

    if user_id is None:
        print("No authenticated user id provided. Skipping DB save.")
        write_worker_result(saved=False, reason="missing_user")
        return
    
    print("\nGenerating professional AI report...")
    report_text = generate_llm_report(session_data, start_time)
    
    if report_text:
        duration = time.time() - start_time
        good_count = sum(1 for b in session_data if b['label'] == 'Good')
        bad_count = sum(1 for b in session_data if b['label'] == 'Bad')
        avg_conf = np.mean([b['confidence'] for b in session_data])
        
        # Save to database
        session_id = database.save_session(
            user_id=user_id,
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total=len(session_data),
            good=good_count,
            bad=bad_count,
            avg_conf=avg_conf,
            duration=duration,
            report=report_text
        )
        print(f"Session saved to database (ID: {session_id})")
        write_worker_result(saved=True, reason="saved", session_id=session_id)
    else:
        write_worker_result(saved=False, reason="report_unavailable")

def main():
    global current_frame
    # Initialize database
    database.init_db()
    
    # Session state
    session_data = []
    start_time = time.time()
    last_stats_update = time.time()
    cap = None
    debug_window_enabled = env_enabled("REALTIME_DEBUG_WINDOW", "0")
    save_classifier_crops = env_enabled("SAVE_CLASSIFIER_CROPS", "1")
    debug_root_dir = None
    classifier_debug_dir = None
    frame_idx = 0

    def handle_exit(sig, frame):
        print(f"\nShutdown signal ({sig}) received. Saving...")
        try:
            nonlocal session_data, start_time, cap
            save_session_to_db(session_data, start_time, SESSION_USER_ID)
            if cap: cap.release()
            cv2.destroyAllWindows()
        except:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_exit)
    signal.signal(signal.SIGINT, handle_exit)

    try:
        # Start streaming server IMMEDIATELY
        t = threading.Thread(target=run_flask, daemon=True)
        t.start()

        if os.path.exists(WORKER_RESULT_FILE):
            try:
                os.remove(WORKER_RESULT_FILE)
            except OSError:
                pass

        print("Loading Classifier...")
        classifier = load_model(CLASSIFIER_MODEL_PATH, compile=False)
        
        print(f"Opening Video: {VIDEO_PATH}")
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print("Error: Could not open video source.")
            update_live_stats(0, 0, 0, "error")
            return

        if debug_window_enabled or save_classifier_crops:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_root_dir = os.path.join(DEBUG_OUTPUT_BASE, f"debug_{timestamp}")
            os.makedirs(debug_root_dir, exist_ok=True)
            if save_classifier_crops:
                classifier_debug_dir = os.path.join(debug_root_dir, "classifier_inputs")
                os.makedirs(classifier_debug_dir, exist_ok=True)
            print(f"Debug outputs: {debug_root_dir}")

        source_is_webcam = isinstance(VIDEO_PATH, int)
        source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if source_fps <= 0 or source_fps > 240:
            source_fps = 30.0
        file_frame_interval = 1.0 / source_fps
        next_file_frame_deadline = time.perf_counter()
        file_read_failures = 0


        active_beans = []
        next_id = 1
        total_good = 0
        total_bad = 0
        paused = False
        
        # Cleanup old flags
        if os.path.exists('stop_worker.flag'):
            os.remove('stop_worker.flag')

        # Initial stats
        update_live_stats(0, 0, 0, "running")
        current_frame = None

        if not source_is_webcam:
            # For uploaded videos, wait until the browser actually starts
            # consuming /video_feed so detection and visualization start together.
            stream_client_event.clear()
            print("Waiting for stream client connection...")
            stream_client_event.wait(timeout=12)
            next_file_frame_deadline = time.perf_counter()

        while True:
            # Check for stop signal from Flask
            if os.path.exists('stop_worker.flag'):
                print("\nGraceful stop flag detected. Saving...")
                save_session_to_db(session_data, start_time, SESSION_USER_ID)
                break

            if not paused:
                success, img = cap.read()
                if not success:
                    if source_is_webcam:
                        # Webcam may drop frames occasionally; keep session alive.
                        time.sleep(0.02)
                        continue
                    # For file mode, tolerate brief decode hiccups before ending session.
                    file_read_failures += 1
                    current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                    near_eof = source_frame_count > 0 and current_pos >= max(source_frame_count - 2, 1)
                    if not near_eof and file_read_failures < 8:
                        time.sleep(0.02)
                        continue

                    print(
                        f"Video source ended (frame {current_pos}/{source_frame_count}). "
                        "Saving session..."
                    )
                    save_session_to_db(session_data, start_time, SESSION_USER_ID)
                    break
                file_read_failures = 0
                frame_idx += 1
                
                h_orig, w_orig = img.shape[:2]

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (9, 9), 0)
                _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                mask_fill_ratio = cv2.countNonZero(thresh) / float(thresh.size)
                if mask_fill_ratio < 0.01 or mask_fill_ratio > 0.85:
                    thresh = cv2.adaptiveThreshold(
                        blur,
                        255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY_INV,
                        31,
                        7
                    )
                kernel = np.ones((5, 5), np.uint8)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                current_blobs = []
                used_boxes = []
                
                for cnt in contours:
                    extracted = extract_blob(cnt)
                    if extracted is None:
                        continue
                    x, y, w, h = extracted
                    current_box = (x, y, w, h)
                    if any(overlaps(current_box, ub) for ub in used_boxes):
                        continue
                    used_boxes.append(current_box)

                    detected_box = current_box
                    cx = x + w // 2
                    cy = y + h // 2
                    current_blobs.append(((cx, cy), detected_box))

                matched_blob_indices = set()
                active_beans_to_keep = []

                for bean in active_beans:
                    best_blob_idx = -1
                    best_dist = float('inf')
                    best_iou = 0.0
                    
                    for i, (blob_centroid, blob_box) in enumerate(current_blobs):
                        if i in matched_blob_indices: continue
                        
                        dist = np.linalg.norm(np.array(bean.centroid) - np.array(blob_centroid))
                        iou_score = box_iou(bean.box, blob_box)
                        if (iou_score > best_iou) or (iou_score == best_iou and dist < best_dist):
                            best_iou = iou_score
                            best_dist = dist
                            best_blob_idx = i
                    
                    if best_blob_idx != -1 and (
                        best_dist < TRACKING_DISTANCE_THRESHOLD or best_iou >= MIN_TRACK_IOU
                    ):
                        new_centroid, new_box = current_blobs[best_blob_idx]
                        stability_iou = box_iou(bean.box, new_box)
                        bean.stable_frames = bean.stable_frames + 1 if stability_iou >= 0.45 else 0
                        bean.frames_seen += 1
                        bean.centroid = new_centroid
                        bean.box = new_box
                        bean.frames_lost = 0
                        matched_blob_indices.add(best_blob_idx)
                        active_beans_to_keep.append(bean)
                    else:
                        bean.frames_lost += 1
                        if bean.frames_lost <= MAX_FRAMES_LOST:
                            active_beans_to_keep.append(bean)

                active_beans = active_beans_to_keep

                for i, (blob_centroid, blob_box) in enumerate(current_blobs):
                    if i not in matched_blob_indices:
                        cx, cy = blob_centroid
                        enters_from_left = cx < w_orig * ENTRY_REGION_RATIO
                        enters_from_right = cx > w_orig * (1.0 - ENTRY_REGION_RATIO)
                        if enters_from_left or enters_from_right:
                            new_bean = BeanTracker(next_id, blob_centroid, blob_box)
                            next_id += 1
                            active_beans.append(new_bean)

                for bean in active_beans:
                    bx, by, bw, bh = bean.box
                    cx, cy = bean.centroid
                    
                    touching_left = bx < EDGE_MARGIN
                    touching_right = (bx + bw) > (w_orig - EDGE_MARGIN)

                    if bean.state == "WAITING_ENTRY":
                        stable_for_classification = (
                            bean.frames_seen >= MIN_VISIBLE_FRAMES and
                            bean.stable_frames >= MIN_STABLE_FRAMES
                        )
                        if (
                            bean.label == "Unknown" and
                            not touching_left and
                            not touching_right and
                            stable_for_classification
                        ):
                            classification = classify_bean_once(
                                classifier,
                                img,
                                bean.box,
                                bean_id=bean.id,
                                frame_idx=frame_idx,
                                debug_dir=classifier_debug_dir if save_classifier_crops else None,
                            )
                            if classification is not None:
                                bean.label, bean.confidence, bean.color = classification
                                bean.state = "TRACKING"
                                if bean.label == "Good":
                                    total_good += 1
                                else:
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
                        
                        # VISUALIZATION UPDATE: Only Green or Red, Simple Label
                        cv2.rectangle(img, (x, y), (x+w, y+h), bean.color, 3)
                        
                        # Simplified label: "Good" or "Bad"
                        label_text = bean.label
                        if bean.label == "Unknown":
                            label_text = "Tracking..."
                        
                        cv2.putText(img, label_text, (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, bean.color, 2)

                # Overlay stats
                cv2.putText(img, f"Good: {total_good}", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, f"Bad: {total_bad}", (20, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Update live stats every 0.5s
                if time.time() - last_stats_update > 0.5:
                    update_live_stats(total_good, total_bad, len(active_beans), "running")
                    last_stats_update = time.time()

                if debug_window_enabled:
                    try:
                        debug_board = build_debug_visualization(
                            img,
                            thresh,
                            current_blobs,
                            active_beans,
                            mask_fill_ratio,
                            total_good,
                            total_bad,
                        )
                        cv2.imshow(DEBUG_WINDOW_NAME, debug_board)
                        key = cv2.waitKey(1) & 0xFF
                        if key == 27:
                            cv2.destroyWindow(DEBUG_WINDOW_NAME)
                            debug_window_enabled = False
                            print("Debug window closed (ESC).")
                    except cv2.error as e:
                        print(f"Debug window disabled: {e}")
                        debug_window_enabled = False
                
                # UPDATE GLOBAL FRAME FOR STREAMING
                with output_lock:
                    current_frame = img.copy()

                if not source_is_webcam:
                    next_file_frame_deadline += file_frame_interval
                    sleep_time = next_file_frame_deadline - time.perf_counter()
                    if sleep_time > 0:
                        time.sleep(min(sleep_time, 0.2))
                    else:
                        # Catch up if processing lagged behind.
                        next_file_frame_deadline = time.perf_counter()
            
            # Allow clean exit via file flag primarily, but keep small sleep
            if source_is_webcam:
                time.sleep(0.01)

    except Exception as e:
        import traceback
        error_msg = f"Worker Crash: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        with open("worker_crash.log", "a") as f:
            f.write(f"\n[{datetime.datetime.now()}] {error_msg}\n")
        update_live_stats(0, 0, 0, "error")
    finally:
        if cap:
            cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

if __name__ == "__main__":
    main()
