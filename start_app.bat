@echo off
cd /d "%~dp0"
start "" /b powershell -NoProfile -Command "Start-Sleep -Seconds 3; Start-Process 'http://127.0.0.1:5000'"
python app.py
