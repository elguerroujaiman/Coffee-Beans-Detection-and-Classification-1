from flask import Flask, render_template, jsonify, redirect, url_for, request
import subprocess
import os
import sys
import json
import database
import requests
from pathlib import Path
import cv2
import numpy as np
import base64
import time

app = Flask(__name__)

# Global variables
detection_process = None
inference_process = None

def start_inference_server():
    """Launch the persistent inference server as a subprocess."""
    global inference_process
    if inference_process is None or inference_process.poll() is not None:
        print("\U0001F680 Launching Inference Server...")
        inference_process = subprocess.Popen(
            [sys.executable, 'inference_server.py'],
            cwd=os.getcwd()
        )
        # Give it a few seconds to load the model
        time.sleep(2)

@app.route('/')
def home():
    """Landing page with clear app description and navigation."""
    return render_template('index.html')

@app.route('/image_detection')
def image_detection_page():
    """Dedicated page for uploading and detecting beans in static images."""
    return render_template('image_detection.html')

@app.route('/realtime_detection')
def realtime_detection_page():
    """Page for video file or webcam stream detection."""
    return render_template('realtime_detection.html')

@app.route('/start')
def start_detection():
    """Start the real-time detection worker."""
    global detection_process
    
    source = request.args.get('source', '0') # '0' for webcam, or path to video
    
    if detection_process and detection_process.poll() is None:
        return jsonify({"status": "error", "message": "Detection already running"})
    
    # Launch detection_worker.py as subprocess
    # We pass the source (camera index or file path)
    detection_process = subprocess.Popen(
        [sys.executable, 'detection_worker.py', source],
        cwd=os.getcwd()
    )
    
    return jsonify({"status": "success", "message": "Detection started", "pid": detection_process.pid})

@app.route('/stop')
def stop_detection():
    """Stop the current detection process gracefully."""
    global detection_process
    
    # If no process ever started
    if not detection_process:
        return jsonify({"status": "error", "message": "No detection running"})

    # If already stopped (e.g. crashed or normally finished)
    if detection_process.poll() is not None:
        if os.path.exists('stop_worker.flag'):
            os.remove('stop_worker.flag')
        return jsonify({"status": "success", "message": "Already stopped"})

    # If running, try graceful stop
    with open('stop_worker.flag', 'w') as f:
        f.write('stop')
    
    try:
        detection_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        print("Worker didn't stop in time, forcing termination.")
        detection_process.terminate()
        
    if os.path.exists('stop_worker.flag'):
        os.remove('stop_worker.flag')
        
    return jsonify({"status": "success", "message": "Detection stopped"})

@app.route('/live_stats')
def live_stats():
    """Return live stats for AJAX polling."""
    try:
        if os.path.exists('live_stats.json'):
            with open('live_stats.json', 'r') as f:
                stats = json.load(f)
            return jsonify(stats)
        return jsonify({"status": "stopped", "good_count": 0, "bad_count": 0, "total": 0})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/detect_image', methods=['POST'])
def detect_image():
    """Route image analysis requests to the persistent inference server."""
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Ensuring inference server is up
    start_inference_server()
    
    # 1. Save temp image
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    input_path = temp_dir / "test_input.jpg"
    file.save(str(input_path))
    
    # 2. Call inference server
    try:
        response = requests.post(
            'http://localhost:5001/predict',
            json={"image_path": str(input_path.absolute())},
            timeout=10
        )
        
        if response.status_code != 200:
            return jsonify({"error": f"Inference error: {response.text}"}), 500
            
        result = response.json()
        
        # 3. Read original for display
        img_orig = cv2.imread(str(input_path))
        _, orig_encoded = cv2.imencode('.jpg', img_orig)
        orig_b64 = base64.b64encode(orig_encoded).decode('utf-8')
        
        return jsonify({
            "status": "success",
            "good_count": result["good"],
            "bad_count": result["bad"],
            "total": result["total"],
            "original": f"data:image/jpeg;base64,{orig_b64}",
            "detected": f"data:image/jpeg;base64,{result['detected_b64']}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/history')
def history():
    """View session history."""
    sessions = database.get_all_sessions()
    stats = database.get_stats()
    return render_template('history.html', sessions=sessions, stats=stats)

@app.route('/session/<int:session_id>')
def session_detail(session_id):
    """View session details."""
    session = database.get_session(session_id)
    return render_template('session.html', session=session) if session else ("Not found", 404)

@app.route('/delete/<int:session_id>')
def delete_session(session_id):
    """Remove a session."""
    database.delete_session(session_id)
    return redirect(url_for('history'))

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video file upload for detection."""
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    video_dir = Path("uploads/videos")
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / file.filename
    file.save(str(video_path))
    
    return jsonify({"status": "success", "video_path": str(video_path.absolute())})

if __name__ == '__main__':
    database.init_db()
    Path("temp_uploads").mkdir(exist_ok=True)
    start_inference_server()
    
    print("\n" + "="*50)
    print("  \U0001F310 Coffee Bean QC System Online")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
