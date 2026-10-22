@echo off
title Nova AI Voice Assistant - Setup & Launch
echo ===================================================================
echo   Nova AI Voice Assistant - Automated Setup & Runner
echo ===================================================================
echo.

:: 1. Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

:: 2. Create virtual environment if not present
if not exist "venv" (
    echo [INFO] Creating Python virtual environment...
    python -m venv venv
)

:: 3. Activate virtual environment
call .\venv\Scripts\activate.bat

:: 4. Upgrade pip and install requirements
echo [INFO] Installing / updating required dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

:: 5. Check for .env file
if not exist ".env" (
    echo.
    echo [WARNING] .env file not found!
    if exist ".env.example" (
        copy .env.example .env
        echo [INFO] Created .env template from .env.example.
        echo [ACTION REQUIRED] Please open .env and add your GEMINI_API_KEY before continuing.
        pause
    )
)

:: 6. Launch options
echo.
echo ===================================================================
echo   Choose Launch Mode:
echo   [1] Web App Dashboard (Browser UI at http://localhost:8000) [Default]
echo   [2] Terminal Voice Assistant (Microphone + Wake Word "Hey Nova")
echo   [3] Streaming Voice Assistant (Threaded Fast TTS)
echo   [4] Run Verification Test Suite
echo ===================================================================
set /p choice="Enter choice [1-4] (default: 1): "

if "%choice%"=="2" (
    python voice_assistant.py
) else if "%choice%"=="3" (
    python voice_stream_assistant.py
) else if "%choice%"=="4" (
    python -c "import test_security, test_wake_word, test_memory_manager, test_voice_detector, test_web_app; test_security.test_security_suite(); test_wake_word.test_wake_word_detector(); test_memory_manager.test_memory_persistence(); test_voice_detector.test_voice_detector_algorithms(); test_web_app.test_web_app_suite()"
) else (
    echo [INFO] Starting Nova Web UI at http://localhost:8000 ...
    python web_app.py
)

pause
