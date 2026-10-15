import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# Engine tests run in .venv-engine (torch); the harness environment does not collect them.
collect_ignore_glob = [] if importlib.util.find_spec("torch") else ["test_*.py"]
