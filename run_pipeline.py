#!/usr/bin/env python3
"""Run the Cricsheet JSON flattening and analysis pipeline in one command."""
from pathlib import Path
import argparse, subprocess, sys
ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Build BBL tables and generate dashboard outputs")
parser.add_argument("archive", type=Path, help="Cricsheet JSON ZIP archive")
args = parser.parse_args()
archive = nargs.archive.resolve()
if not archive.exists():
    raise SystemExit(f"Archive not found: {archive}")
subprocess.run([sys.executable, str(ROOT / "build_dataset.py"), str(archive), "--out-dir", str(ROOT / "data")], check=True)
subprocess.run([sys.executable, str(ROOT / "data_processing.py"), "--out", str(ROOT / "output")], check=True)
print("Pipeline complete. Start the dashboard with: streamlit run app.py")
