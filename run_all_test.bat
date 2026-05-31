@echo off
REM NeuroSim V2 - run the full automated test on Windows
cd /d "%~dp0"

echo ============================================================
echo  NeuroSim V2 - all_test
echo ============================================================
echo Installing/updating dependencies (first run takes a while)...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] pip install failed. Is Python installed and on PATH?
    echo         Install Python 3.9+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo.
echo Starting all_test... you will be asked for the good/bad device xlsx paths.
echo (analysis runs automatically when all runs finish.)
echo.
python all_test.py %*

echo.
pause
