@echo off
cd /d "%~dp0"
python -m pip install -r requirements.txt
python data_processing.py demo_data/demo_ball_by_ball.csv demo_data/demo_matches.csv --out output
streamlit run app.py
pause
