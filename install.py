"""Install hook — installs requirements.txt into the host ComfyUI env."""

import subprocess
import sys
from pathlib import Path

REQS = Path(__file__).resolve().parent / "requirements.txt"
if REQS.is_file():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(REQS)])
