"""Read-only runtime diagnostics, invoked through colab exec."""
import importlib.metadata
import json
import platform
import subprocess

packages = {}
for name in ("torch", "torchvision", "Pillow", "numpy", "matplotlib", "gdown"):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
print(json.dumps({"python": platform.python_version(), "packages": packages}, indent=2))
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=True)
import torch
assert torch.cuda.is_available(), "A CUDA GPU is required for the real experiment."
print("CUDA ready:", torch.cuda.get_device_name(0))
