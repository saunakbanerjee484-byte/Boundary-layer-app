"""
tests/test_validation_utils.py
================================
Tests for the relative-error utility itself, plus a validation-suite
skeleton that will automatically start checking model output against
`validation/thwaites_reference.csv` and `validation/head_reference.csv`
once those stub files are populated with real published benchmark data
(see the header comments in each CSV for suggested sources).
"""

from __future__ import annotations

import pytest

from physics_engine.base_model import FlowConditions
from physics_engine.models.laminar_thwaites import ThwaitesModel
from physics_engine.models.turbulent_head import HeadEntrainmentModel

from tests.validation_utils import load_reference_csv, relative_error_percent

# Engineering-acceptable tolerance for integral boundary-layer methods
# compared against exact/experimental references. Thwaites and Head are
# both approximate closures, not exact solutions, so a few percent
# agreement is the normal expectation in the literature.
ACCEPTABLE_RELATIVE_ERROR_PERCENT = 5.0


def test_relative_error_percent_basic():
    assert relative_error_percent(105.0, 100.0) == pytest.approx(5.0)
    assert relative_error_percent(95.0, 100.0) == pytest.approx(5.0)
    assert relative_error_percent(100.0, 100.0) == pytest.approx(0.0)


def test_relative_error_percent_rejects_zero_reference():
    with pytest.raises(ValueError):
        relative_error_percent(1.0, 0.0)


def test_thwaites_reference_csv_loads():
    """The stub CSV should load without error, even with zero data rows."""
    rows = load_reference_csv("thwaites_reference.csv")
    assert isinstance(rows, list)


def test_head_reference_csv_loads():
    """The stub CSV should load without error, even with zero data rows."""
    rows = load_reference_csv("head_reference.csv")
    assert isinstance(rows, list)


def test_thwaites_against_reference_data():
    """
    Validates Thwaites' momentum-thickness prediction against published
    reference data, once `validation/thwaites_reference.csv` has been
    populated (see that file's header for suggested sources, e.g.
    Falkner-Skan exact solutions). Skips cleanly while the stub is still
    empty, so the test suite documents the intended validation without
    failing on a deliberately-empty placeholder.
    """
    rows = load_reference_csv("thwaites_reference.csv")
    if not rows:
        pytest.skip(
            "validation/thwaites_reference.csv has no data rows yet -- "
            "populate it with published reference data to activate this "
            "validation check."
        )

    model = ThwaitesModel()
    for row in rows:
        conditions = FlowConditions(
            nu=1.5e-5,
            rho=1.225,
            U=float(row["U_ms"]),
            dpdx=0.0,
            x_max=float(row["x_m"]),
        )
        growth = model.compute_growth(conditions)
        theta_model = float(growth["delta"][-1])
        theta_ref = float(row["theta_ref_m"])

        error_pct = relative_error_percent(theta_model, theta_ref)
        assert error_pct <= ACCEPTABLE_RELATIVE_ERROR_PERCENT, (
            f"Thwaites momentum thickness at x={row['x_m']}m deviates from "
            f"reference '{row.get('source', '?')}' by {error_pct:.2f}% "
            f"(limit {ACCEPTABLE_RELATIVE_ERROR_PERCENT}%)."
        )


def test_head_against_reference_data():
    """
    Validates Head's shape-factor prediction against published reference
    data (e.g. the Stanford Olympics dataset), once
    `validation/head_reference.csv` has been populated. Skips cleanly
    while the stub is still empty.
    """
    rows = load_reference_csv("head_reference.csv")
    if not rows:
        pytest.skip(
            "validation/head_reference.csv has no data rows yet -- "
            "populate it with published reference data (e.g. the "
            "Stanford Olympics dataset) to activate this validation check."
        )

    model = HeadEntrainmentModel()
    for row in rows:
        conditions = FlowConditions(
            nu=1.5e-5,
            rho=1.225,
            U=float(row["U_ms"]),
            dpdx=0.0,
            x_max=float(row["x_m"]),
        )
        _x, _theta, H_arr, _x_sep = model._march(conditions)
        H_model = float(H_arr[-1])
        H_ref = float(row["H_ref"])

        error_pct = relative_error_percent(H_model, H_ref)
        assert error_pct <= ACCEPTABLE_RELATIVE_ERROR_PERCENT, (
            f"Head shape factor H at x={row['x_m']}m deviates from "
            f"reference '{row.get('source', '?')}' by {error_pct:.2f}% "
            f"(limit {ACCEPTABLE_RELATIVE_ERROR_PERCENT}%)."
        )
