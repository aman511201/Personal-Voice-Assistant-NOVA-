#!/usr/bin/env bash
# ==============================================================================
# Nova AI Voice Assistant - Automated Setup & Runner (macOS / Linux)
# ==============================================================================

set -e

echo "==================================================================="
echo "  Nova AI Voice Assistant - Automated Setup & Runner"
echo "==================================================================="

# 1. Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found. Please install Python 3.10+."
    exit 1
fi

# 2. Virtual Environment
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv venv
fi

# 3. Activate venv
source venv/bin/activate

# 4. Install requirements
echo "[INFO] Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# 5. Check .env
if [ ! -f ".env" ]; then
    echo "[WARNING] .env file not found."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "[INFO] Copied .env.example to .env. Please configure your GEMINI_API_KEY."
    fi
fi

# 6. Launch Web App
echo "[INFO] Starting Nova Web UI at http://localhost:8000 ..."
python3 web_app.py
