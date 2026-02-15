from flask import Flask, render_template, jsonify, redirect, url_for, request, session, g
import subprocess
import os
import sys
import json
import hashlib
import secrets
import database
import requests
from pathlib import Path
import cv2
import base64
import time
import socket
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "bean-guard-local-secret-change-me")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_REFRESH_EACH_REQUEST"] = False

REMEMBER_COOKIE_NAME = "bean_guard_remember"
REMEMBER_HINT_COOKIE_NAME = "bean_guard_remember_hint"
REMEMBER_DAYS = 10
TAB_BOOT_COOKIE_NAME = "bean_guard_tab_boot"
TAB_BOOT_SECONDS = 20
WORKER_RESULT_FILE = "worker_result.json"
UPLOAD_RUNTIME_DIR = Path("uploads/runtime")

detection_process = None
inference_process = None
active_stream_port = None


def _hash_token(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _is_safe_next_path(path_value):
    if not path_value:
        return False
    return path_value.startswith("/") and not path_value.startswith("//")


def _is_api_request_path(path):
    protected_api_prefixes = (
        "/start",
        "/stop",
        "/live_stats",
        "/detect_image",
        "/upload_video",
    )
    return path.startswith(protected_api_prefixes)


def _set_logged_in_user(user):
    session.permanent = False
    session["user_id"] = user["id"]
    session["username"] = user["username"]


def _external_webcam_available(index=1):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return False
    success, _ = cap.read()
    cap.release()
    return bool(success)


def _clear_worker_result():
    if os.path.exists(WORKER_RESULT_FILE):
        try:
            os.remove(WORKER_RESULT_FILE)
        except OSError:
            pass


def _read_worker_result():
    if not os.path.exists(WORKER_RESULT_FILE):
        return None
    try:
        with open(WORKER_RESULT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _is_runtime_upload_path(path_value):
    if not path_value:
        return False
    try:
        file_path = Path(path_value).resolve()
        runtime_root = UPLOAD_RUNTIME_DIR.resolve()
        return runtime_root in file_path.parents
    except Exception:
        return False


def _delete_runtime_upload(path_value):
    if not _is_runtime_upload_path(path_value):
        return
    try:
        Path(path_value).unlink(missing_ok=True)
    except Exception:
        pass


def _stop_payload_from_worker_result():
    result = _read_worker_result()
    if not result:
        return {
            "status": "success",
            "message": "Session was not saved.",
            "saved": False,
            "reason": "no_result",
        }

    saved = bool(result.get("saved"))
    reason = result.get("reason")

    if not saved and reason == "no_beans":
        message = "No beans were detected; session was not saved."
    elif saved:
        message = "Session saved successfully."
    else:
        message = "Session was not saved."

    return {
        "status": "success",
        "message": message,
        "saved": saved,
        "reason": reason,
    }


def start_inference_server():
    """Launch the persistent inference server as a subprocess."""
    global inference_process
    if inference_process is None or inference_process.poll() is not None:
        print("Launching inference server...")
        inference_process = subprocess.Popen(
            [sys.executable, "inference_server.py"],
            cwd=os.getcwd(),
        )
        time.sleep(2)


def find_available_stream_port(start_port=5002, end_port=5100):
    """Find an available localhost port for the MJPEG stream."""
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return None


def wait_for_stream_ready(stream_port, timeout_sec=None):
    """Wait until detection worker stream server is reachable."""
    if timeout_sec is None:
        timeout_sec = float(os.getenv("STREAM_START_TIMEOUT_SEC", "25"))
    global detection_process
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if detection_process is not None and detection_process.poll() is not None:
            return False, "Detection worker exited before stream started."
        try:
            res = requests.get(f"http://127.0.0.1:{stream_port}/health", timeout=0.5)
            if res.status_code == 200:
                return True, None
        except requests.RequestException:
            pass
        time.sleep(0.2)
    return False, (
        f"Stream server did not become ready on port {stream_port} "
        f"within {timeout_sec:.0f}s."
    )


@app.before_request
def load_and_enforce_auth():
    """Load authenticated user and protect all routes except login/signup/static/logout."""
    g.user = None

    user_id = session.get("user_id")
    if user_id:
        user = database.get_user_by_id(user_id)
        if user:
            g.user = user
        else:
            session.clear()

    if g.user is None:
        remember_raw = request.cookies.get(REMEMBER_COOKIE_NAME)
        if remember_raw:
            user = database.get_user_by_valid_token(_hash_token(remember_raw))
            if user:
                g.user = user
                _set_logged_in_user(user)

    public_endpoints = {"login", "signup", "static"}
    if request.endpoint is None or request.endpoint in public_endpoints:
        return None

    if g.user is None:
        if _is_api_request_path(request.path):
            return jsonify({"status": "error", "message": "Authentication required"}), 401
        return redirect(url_for("login", next=request.path))

    return None


@app.context_processor
def inject_user_context():
    return {"current_user": g.user}


@app.route("/login", methods=["GET", "POST"])
def login():
    if g.user is not None:
        return redirect(url_for("home"))

    error = None
    created = request.args.get("created") == "1"
    next_path = request.args.get("next", "")

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        remember_me = request.form.get("remember_me") == "on"
        next_form = request.form.get("next") or ""

        user = database.get_user_by_username(username)
        if not user or not check_password_hash(user["password_hash"], password):
            error = "Invalid username or password."
        else:
            session.clear()
            _set_logged_in_user(user)
            redirect_target = next_form if _is_safe_next_path(next_form) else url_for("home")
            response = redirect(redirect_target)

            existing_raw = request.cookies.get(REMEMBER_COOKIE_NAME)
            if existing_raw:
                database.delete_remember_token(_hash_token(existing_raw))
                response.delete_cookie(REMEMBER_COOKIE_NAME)
            response.delete_cookie(REMEMBER_HINT_COOKIE_NAME)

            if remember_me:
                token_raw = secrets.token_urlsafe(48)
                expires_at = datetime.now(timezone.utc) + timedelta(days=REMEMBER_DAYS)
                database.save_remember_token(
                    user_id=user["id"],
                    token_hash=_hash_token(token_raw),
                    expires_at=expires_at.isoformat(),
                )
                response.set_cookie(
                    REMEMBER_COOKIE_NAME,
                    token_raw,
                    max_age=REMEMBER_DAYS * 24 * 60 * 60,
                    httponly=True,
                    samesite="Lax",
                )
                response.set_cookie(
                    REMEMBER_HINT_COOKIE_NAME,
                    "1",
                    max_age=REMEMBER_DAYS * 24 * 60 * 60,
                    httponly=False,
                    samesite="Lax",
                )
                response.delete_cookie(TAB_BOOT_COOKIE_NAME)
            else:
                response.set_cookie(
                    TAB_BOOT_COOKIE_NAME,
                    "1",
                    max_age=TAB_BOOT_SECONDS,
                    httponly=False,
                    samesite="Lax",
                )
            return response

    if not _is_safe_next_path(next_path):
        next_path = ""

    return render_template("login.html", error=error, created=created, next_path=next_path)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if g.user is not None:
        return redirect(url_for("home"))

    error = None

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        confirm_password = request.form.get("confirm_password") or ""

        if len(username) < 3:
            error = "Username must be at least 3 characters."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        elif password != confirm_password:
            error = "Passwords do not match."
        else:
            user_id, db_error = database.create_user(
                username=username,
                password_hash=generate_password_hash(password),
            )
            if db_error:
                error = db_error
            elif user_id:
                session.clear()
                _set_logged_in_user({"id": user_id, "username": username})
                response = redirect(url_for("home"))
                response.set_cookie(
                    TAB_BOOT_COOKIE_NAME,
                    "1",
                    max_age=TAB_BOOT_SECONDS,
                    httponly=False,
                    samesite="Lax",
                )
                return response
            else:
                error = "Could not create account. Please try again."

    return render_template("signup.html", error=error)


@app.route("/logout")
def logout():
    response = redirect(url_for("login"))
    remember_raw = request.cookies.get(REMEMBER_COOKIE_NAME)
    if remember_raw:
        database.delete_remember_token(_hash_token(remember_raw))
    response.delete_cookie(REMEMBER_COOKIE_NAME)
    response.delete_cookie(REMEMBER_HINT_COOKIE_NAME)
    response.delete_cookie(TAB_BOOT_COOKIE_NAME)
    session.clear()
    return response


@app.route("/")
def home():
    """Landing page with clear app description and navigation."""
    return render_template("index.html")


@app.route("/image_detection")
def image_detection_page():
    """Dedicated page for uploading and detecting beans in static images."""
    return render_template("image_detection.html")


@app.route("/realtime_detection")
def realtime_detection_page():
    """Page for video file or webcam stream detection."""
    return render_template("realtime_detection.html")


@app.route("/start")
def start_detection():
    """Start the real-time detection worker."""
    global detection_process, active_stream_port

    source = request.args.get("source", "1")

    if detection_process and detection_process.poll() is None:
        return jsonify({"status": "error", "message": "Detection already running"})

    if source.isdigit():
        if int(source) != 1:
            return jsonify({
                "status": "error",
                "message": "Camera source is fixed to webcam index 1.",
            }), 400
        if not _external_webcam_available(1):
            return jsonify({
                "status": "error",
                "message": "Try to connect a webcam (camera index 1) and start again.",
            }), 400

    _clear_worker_result()

    active_stream_port = find_available_stream_port()
    if active_stream_port is None:
        return jsonify({"status": "error", "message": "No free stream port found"}), 500

    worker_cmd = [
        sys.executable,
        "detection_worker.py",
        source,
        "--stream-port",
        str(active_stream_port),
        "--user-id",
        str(g.user["id"]),
    ]
    if _is_runtime_upload_path(source):
        worker_cmd.extend(["--delete-source-on-exit", "1"])

    detection_process = subprocess.Popen(worker_cmd, cwd=os.getcwd())

    ready, err = wait_for_stream_ready(active_stream_port)
    if not ready:
        if detection_process and detection_process.poll() is None:
            detection_process.terminate()
        _delete_runtime_upload(source)
        detection_process = None
        active_stream_port = None
        return jsonify({"status": "error", "message": err}), 500

    return jsonify({
        "status": "success",
        "message": "Detection started",
        "pid": detection_process.pid,
        "stream_url": f"http://127.0.0.1:{active_stream_port}/video_feed",
    })


@app.route("/stop")
def stop_detection():
    """Stop the current detection process gracefully."""
    global detection_process, active_stream_port

    if not detection_process:
        return jsonify({
            "status": "success",
            "message": "No detection running.",
            "saved": False,
            "reason": "no_process",
        })

    if detection_process.poll() is not None:
        if os.path.exists("stop_worker.flag"):
            os.remove("stop_worker.flag")
        detection_process = None
        active_stream_port = None
        return jsonify(_stop_payload_from_worker_result())

    with open("stop_worker.flag", "w", encoding="utf-8") as f:
        f.write("stop")

    try:
        detection_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        print("Worker didn't stop in time, forcing termination.")
        detection_process.terminate()

    if os.path.exists("stop_worker.flag"):
        os.remove("stop_worker.flag")
    detection_process = None
    active_stream_port = None
    return jsonify(_stop_payload_from_worker_result())


@app.route("/live_stats")
def live_stats():
    """Return live stats for AJAX polling."""
    try:
        if os.path.exists("live_stats.json"):
            with open("live_stats.json", "r", encoding="utf-8") as f:
                stats = json.load(f)
            return jsonify(stats)
        return jsonify({"status": "stopped", "good_count": 0, "bad_count": 0, "total": 0})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/detect_image", methods=["POST"])
def detect_image():
    """Route image analysis requests to the persistent inference server."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    start_inference_server()

    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    input_path = temp_dir / "test_input.jpg"
    file.save(str(input_path))

    try:
        response = requests.post(
            "http://127.0.0.1:5001/predict",
            json={"image_path": str(input_path.absolute())},
            timeout=10,
        )

        if response.status_code != 200:
            return jsonify({"error": f"Inference error: {response.text}"}), 500

        result = response.json()

        img_orig = cv2.imread(str(input_path))
        _, orig_encoded = cv2.imencode(".jpg", img_orig)
        orig_b64 = base64.b64encode(orig_encoded).decode("utf-8")

        return jsonify({
            "status": "success",
            "good_count": result["good"],
            "bad_count": result["bad"],
            "total": result["total"],
            "original": f"data:image/jpeg;base64,{orig_b64}",
            "detected": f"data:image/jpeg;base64,{result['detected_b64']}",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history")
def history():
    """View session history for logged-in user."""
    sessions = database.get_all_sessions(g.user["id"])
    stats = database.get_stats(g.user["id"])
    return render_template("history.html", sessions=sessions, stats=stats)


@app.route("/session/<int:session_id>")
def session_detail(session_id):
    """View session details for logged-in user."""
    session_row = database.get_session(g.user["id"], session_id)
    return render_template("session.html", session=session_row) if session_row else ("Not found", 404)


@app.route("/delete/<int:session_id>")
def delete_session(session_id):
    """Remove a session owned by logged-in user."""
    database.delete_session(g.user["id"], session_id)
    return redirect(url_for("history"))


@app.route("/upload_video", methods=["POST"])
def upload_video():
    """Handle video file upload for detection."""
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    video_dir = UPLOAD_RUNTIME_DIR
    video_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename).suffix.strip() or ".mp4"
    safe_name = f"upload_{int(time.time() * 1000)}_{uuid4().hex[:8]}{suffix}"
    video_path = video_dir / safe_name
    file.save(str(video_path))

    cap = cv2.VideoCapture(str(video_path))
    is_open = cap.isOpened()
    has_first_frame = False
    fps = 0.0
    frame_count = 0.0
    duration_sec = None
    if is_open:
        has_first_frame, _ = cap.read()
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        if fps > 0 and frame_count > 0:
            duration_sec = frame_count / fps
    cap.release()

    if not is_open or not has_first_frame:
        try:
            video_path.unlink(missing_ok=True)
        except Exception:
            pass
        return jsonify({
            "status": "error",
            "error": "Uploaded file is not a valid readable video.",
        }), 400

    return jsonify({
        "status": "success",
        "video_path": str(video_path.resolve()),
        "duration_sec": duration_sec,
    })


if __name__ == "__main__":
    database.init_db()
    Path("temp_uploads").mkdir(exist_ok=True)
    start_inference_server()

    print("\n" + "=" * 50)
    print("  Coffee Bean QC System Online")
    print("=" * 50 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
