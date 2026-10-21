"""
Root app launcher for Student-RAG-Knowledge-Base.
Allows running 'streamlit run app.py' directly from the workspace root directory.
"""
import runpy
from pathlib import Path

target_app = Path(__file__).resolve().parent / "Student-RAG-Knowledge-Base" / "app.py"
runpy.run_path(str(target_app), run_name="__main__")
