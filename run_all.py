"""Re-run the whole analysis on the saved data (no download).

    python run_all.py              # analysis + figures + tests
    python run_all.py --download   # first download the public PDFs and SNB data again
"""

import os
import subprocess
import sys

STEPS = ["scripts/02_curve_and_inputs.py", "scripts/03_irrbb.py", "scripts/04_nmd_prepayment_hedge.py"]

if __name__ == "__main__":
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    if "--download" in sys.argv:
        STEPS.insert(0, "scripts/01_download_data.py")
    for step in STEPS:
        print(f"\n=== {step} ===", flush=True)
        subprocess.run([sys.executable, step], check=True, env=env)
    subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True, env=env)
