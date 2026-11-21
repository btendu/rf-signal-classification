@echo off
REM Double-click (Windows) to start the presentation app.
cd /d "%~dp0"
where python >nul 2>nul || (echo Python 3 is not installed. Get it from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^). & pause & exit /b 1)
if not exist ".venv" (
  echo First start: setting up ^(only once^)...
  python -m venv .venv || (echo Could not create the virtual environment. & pause & exit /b 1)
  .venv\Scripts\python -m pip install --upgrade pip -q
  .venv\Scripts\python -m pip install -r requirements.txt -q || (echo Package installation failed - check the internet connection. & pause & exit /b 1)
)
echo Starting the app - it opens in your browser. Keep this window open; close it to stop the app.
start "" /b cmd /c "timeout /t 4 >nul & start http://localhost:8501"
.venv\Scripts\python -m streamlit run app.py --server.port 8501
