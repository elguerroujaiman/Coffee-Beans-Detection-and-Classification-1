# CB Classifier

A local web application for detecting coffee beans in images and video, then classifying each detected bean as **Good** or **Bad**. It combines OpenCV image processing with a trained MobileNetV2 classifier and keeps each account's analysis history in SQLite.

## What it includes

- Image upload with annotated detections and quality counts.
- Video-file analysis with a live MJPEG stream, tracking, and session statistics.
- Optional webcam input (the current implementation uses camera index 1).
- Local signup/login and per-user analysis history.
- Optional Google Gemini summaries for completed real-time sessions. Without a Gemini key, the app still detects and classifies beans.
- The bundled model file **coffee_mobile_net_tuned.h5**.
- Docker Compose setup with persistent SQLite and upload volumes.

The app runs as a local demonstration. It uses Flask's development server and is not configured as an internet-facing production service.

## Run with Docker

Docker is the easiest way to get a consistent Python and TensorFlow environment.

In PowerShell, from the project folder:

    Copy-Item .env.example .env
    docker compose up --build

Open http://127.0.0.1:5000, create a local account, and upload an image or video. The first build downloads TensorFlow and can take several minutes.

Stop the app with Ctrl+C. Start it again with docker compose up. The database remains in a named volume. To delete the app's saved accounts, sessions, and uploaded files as well, run:

    docker compose down -v

That command permanently removes the local Docker volumes.

The Compose configuration binds the web app and stream ports to loopback on your computer. Docker publishes the live video stream on STREAM_PORT (default 5002). If that host port is already in use, set STREAM_PORT in .env to another free port. When running directly with Python and no STREAM_PORT configured, the app searches ports 5002 through 5100. Docker Desktop on Windows and macOS may not pass a host webcam into the container; video upload is the reliable container workflow. For webcam use, run the app directly on a machine where OpenCV can access the camera.

## Run directly with Python

Use Python 3.11 on Windows, macOS, or Linux.

    python -m venv .venv

On Windows PowerShell:

    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    Copy-Item .env.example .env
    python app.py

Then open http://127.0.0.1:5000. The first launch loads the bundled model and may take a while. Create a new account on the signup page; there is no pre-seeded account.

The .env file is local configuration and is ignored by Git. The app creates a random Flask session key for local development when FLASK_SECRET_KEY is blank. Set a strong value if you want sessions to survive app restarts. Set GEMINI_API_KEY only if you want Gemini-generated session reports. Never put a real key in this repository or in a submission archive.

## Using the app

1. Create an account from the signup page.
2. Use **Image Test** for a still image.
3. Use **Real-time** to upload a video, or use a compatible webcam when running locally.
4. Use **History** to review or delete saved analysis sessions.

Good results depend on lighting and background. The image-processing stage works best with beans spread out against a plain, light surface. This is a demonstration model; validate its predictions before using them for operational decisions.

## Project layout

- **app.py**: Flask UI, authentication, upload handling, and worker orchestration.
- **inference_server.py**: local image inference API on port 5001.
- **detection_worker.py**: video segmentation, tracking, classification, and MJPEG stream.
- **database.py**: SQLite schema and user/session persistence.
- **templates/** and **static/**: web interface.
- **coffee_mobile_net_tuned.h5**: trained classifier required for inference.
- **dataset/**: local training/validation/test images; ignored by Git and Docker and not needed for inference.
- **testing/**: local sample assets for manual evaluation.

## Data and privacy

SQLite data, uploaded videos, temporary images, logs, and .env configuration are local runtime files and are excluded from Git. Passwords are stored as Werkzeug password hashes. If GEMINI_API_KEY is configured, the app sends the completed session's summary data to Google Gemini to generate a report.

The raw training dataset was assembled from images found across multiple sources and is intentionally not included in this repository. Its source-specific redistribution terms and attribution requirements are not documented here, so do not redistribute those images unless you have confirmed the applicable terms. The dataset is not needed to run the app; inference uses the included trained model.

## Configuration

Copy .env.example to .env and edit values as needed. Important settings:

- FLASK_SECRET_KEY: optional for local development; required when APP_ENV=production.
- GEMINI_API_KEY: optional API key for generated reports.
- MAX_UPLOAD_MB: maximum image/video upload size, default 100.
- CLASSIFICATION_THRESHOLD: classification cutoff for a single-output model.

## License

The MIT license applies to the original application source code. The training images and trained model may have separate rights; confirm their terms and attribution before redistribution.
