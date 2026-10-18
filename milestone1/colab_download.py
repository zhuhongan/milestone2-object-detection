"""Run the complete read-only acquisition on the allocated Colab VM."""
import subprocess
import sys

process = subprocess.Popen([sys.executable, "-u", "/content/download_dataset.py",
                            "--dest", "/content/milestone1/data/final", "--workers", "8"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in process.stdout:
    print(line, end="", flush=True)
if process.wait():
    raise RuntimeError(f"Dataset acquisition failed with exit code {process.returncode}")
