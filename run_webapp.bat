@echo off
REM NeuroSim V2 - launch the Gradio web UI on Windows
cd /d "%~dp0"

echo Installing/updating dependencies (first run takes a while)...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed. Install Python 3.9+ and re-run.
    pause
    exit /b 1
)

echo.
echo Launching web UI... open http://127.0.0.1:7860 in your browser.
echo Press Ctrl+C in this window to stop.
echo.
python webapp\app.py

pause
