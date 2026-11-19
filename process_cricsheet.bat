@echo off
cd /d "%~dp0"
echo First download the Big Bash League JSON ZIP from Cricsheet and save it as data\bbl_json.zip.
if not exist data\bbl_json.zip (
  echo Archive not found: data\bbl_json.zip
  pause
  exit /b 1
)
python run_pipeline.py data\bbl_json.zip
streamlit run app.py
pause
