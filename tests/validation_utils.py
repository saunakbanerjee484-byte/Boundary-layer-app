"""
tests/validation_utils.py
===========================
Shared utility for comparing model output against published reference
(benchmark/experimental) data, plus a small CSV loader for the stub
files under `validation/`.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict

VALIDATION_DIR = Path(__file__).resolve().parent.parent / "validation"


def relative_error_percent(q_model: float, q_reference: float) -> float:
    """
    Relative percentage error between a model's output and a published
    reference value:

        epsilon = |q_model - q_reference| / |q_reference| * 100%

    Raises ValueError if q_reference is zero, since relative error is
    undefined in that case (use an absolute-error check instead for
    reference values that can legitimately be zero).
    """
    if q_reference == 0:
        raise ValueError(
            "relative_error_percent is undefined for q_reference == 0; "
            "use an absolute-error comparison instead for this case."
        )
    return abs(q_model - q_reference) / abs(q_reference) * 100.0


def load_reference_csv(filename: str) -> List[Dict[str, str]]:
    """
    Loads a `validation/<filename>` reference CSV into a list of row
    dicts, skipping comment lines (leading '#') and the header. Returns
    an empty list for a file that has no data rows yet (e.g. the shipped
    stubs), so callers can `pytest.skip(...)` cleanly rather than fail
    when no reference data has been populated yet.
    """
    path = VALIDATION_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"No such validation reference file: {path}")

    rows: List[Dict[str, str]] = []
    with path.open(newline="") as f:
        # Filter out comment lines before handing to csv.DictReader, since
        # DictReader has no native comment-skipping support.
        data_lines = (line for line in f if not line.lstrip().startswith("#"))
        reader = csv.DictReader(data_lines)
        for row in reader:
            # Skip fully-blank trailing lines, if any.
            if any(v.strip() for v in row.values() if v is not None):
                rows.append(row)
    return rows
