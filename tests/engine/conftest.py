import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "16")  # engine CPU tests share the host with GPU measurements
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# Engine tests run in .venv-engine (torch); the harness environment does not collect them.
collect_ignore_glob = [] if importlib.util.find_spec("torch") else ["test_*.py"]
if importlib.util.find_spec("torch"):
    import torch

    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
