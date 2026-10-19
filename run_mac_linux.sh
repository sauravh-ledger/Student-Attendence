#!/bin/bash
cd "$(dirname "$0")"
echo "Setting up (first time takes a few minutes)..."
[ -d venv ] || python3 -m venv venv
source venv/bin/activate
pip install -q -r requirements.txt
echo "Starting the app in your browser..."
streamlit run app.py
