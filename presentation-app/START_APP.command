#!/bin/bash
# Double-click (macOS) to start the presentation app. First start installs
# the Python packages into a private folder (.venv) and takes 1-3 minutes.
cd "$(dirname "$0")"
PY=python3
if ! command -v $PY >/dev/null 2>&1; then
  echo "Python 3 is not installed. Download it from https://www.python.org/downloads/ and try again."
  read -p "Press Enter to close"; exit 1
fi
if [ ! -d ".venv" ]; then
  echo "First start: setting up (only once)..."
  $PY -m venv .venv || { echo "Could not create the virtual environment."; read -p "Press Enter"; exit 1; }
  ./.venv/bin/python -m pip install --upgrade pip -q
  ./.venv/bin/python -m pip install -r requirements.txt -q || { echo "Package installation failed - check the internet connection."; read -p "Press Enter"; exit 1; }
fi
echo "Starting the app - it opens in your browser. Keep this window open; close it to stop the app."
( sleep 4; open "http://localhost:8501" 2>/dev/null || xdg-open "http://localhost:8501" 2>/dev/null ) &
./.venv/bin/python -m streamlit run app.py --server.port 8501
