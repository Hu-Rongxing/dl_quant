import os
import sys
import subprocess
from pathlib import Path

def start_jupyter_lab():
    work_dir = Path(__file__).parent.parent
    os.chdir(work_dir)
    os.environ['JUPYTER_CONFIG_DIR'] = str(work_dir)
    subprocess.run([sys.executable, "-m", "jupyter", "lab"])

if __name__ == "__main__":
    start_jupyter_lab()