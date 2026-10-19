@echo off
cd /d "%~dp0"
echo Setting up (first time takes a few minutes)...
if not exist venv python -m venv venv
call venv\Scripts\activate
pip install -q -r requirements.txt
echo Starting the app in your browser...
streamlit run app.py
pause
