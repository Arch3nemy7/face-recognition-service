@echo off
REM Face Recognition Service - Setup Script (Windows)
REM This script sets up the Python virtual environment and installs dependencies

echo ==========================================
echo Face Recognition Service Setup
echo ==========================================
echo.

echo Checking Python version...
python --version
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo.
echo Creating virtual environment...
python -m venv .venv

echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo ==========================================
echo Setup complete!
echo ==========================================
echo.
echo To activate the virtual environment, run:
echo   .venv\Scripts\activate.bat
echo.
echo To start the service, run:
echo   uvicorn face_recognition_service.main:app --host 0.0.0.0 --port 8000
echo.
echo Or simply:
echo   python -m face_recognition_service.main
echo.
echo API documentation will be available at:
echo   http://localhost:8000/docs
echo.
pause
