"""
conftest.py
===========
Ensures the repository root is on sys.path so the top-level `config` and
`physics_engine` modules import cleanly under pytest regardless of the
directory pytest is invoked from (e.g. `pytest`, `pytest tests/`, or an
IDE test runner with a different working directory).
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
