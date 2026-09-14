"""Make the repository-root module importable in local and CI pytest runs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
