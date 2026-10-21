"""Puts every team member's src/ folder on sys.path so modules can import each other.

The project is split into four member folders (01_..., 02_..., 03_..., 04_...), each with its own
src/. The code was written with flat imports (`from model import ...`), so one import of this file,
from any entry point, makes all of them importable regardless of which folder they live in.
"""
import glob
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
for _d in sorted(glob.glob(os.path.join(ROOT, "[0-9][0-9]_*", "src"))):
    if _d not in sys.path:
        sys.path.insert(0, _d)
