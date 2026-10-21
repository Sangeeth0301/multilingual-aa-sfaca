"""pytest bootstrap: make all member src/ folders (and the backend folder) importable."""
import glob
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import sfaca_paths  # noqa: E402,F401

for _d in glob.glob(os.path.join(ROOT, "[0-9][0-9]_*", "backend")):
    sys.path.insert(0, _d)
