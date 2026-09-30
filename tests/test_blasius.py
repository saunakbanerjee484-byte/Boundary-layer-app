"""
tests/test_blasius.py
======================
Scientific validation test for the Blasius similarity solution
(`physics_engine/models/zpg_blasius.py`).

Blasius' laminar flat-plate boundary-value problem,

    2 f''' + f f'' = 0,   f(0) = 0, f'(0) = 0, f'(eta -> inf) = 1

has a single well-known benchmark constant: the dimensionless wall-shear
parameter f''(0), first tabulated by Blasius (1908) and refined by later
numerical solutions (e.g. Howarth, 1938). The accepted reference value,
also used as this module's own internal fallback constant, is

    f''(0) ~= 0.33206

This test solves the BVP exactly as the app does (via
`_solve_blasius_similarity`, the same `scipy.integrate.solve_bvp`-based
solver `zpg_blasius.py` uses to drive the UI) and asserts the numerical
solve reproduces this benchmark to a tight tolerance -- i.e. that the
solver itself, not just a hardcoded fallback, is correct.
"""

from __future__ import annotations

import pytest

from physics_engine.models.zpg_blasius import (
    _BLASIUS_F_DOUBLE_PRIME_0_REFERENCE,
    _solve_blasius_similarity,
)

# Accepted literature benchmark for f''(0) (Howarth's refined tabulation
# of Blasius' 1908 similarity solution).
BLASIUS_F_DOUBLE_PRIME_0_BENCHMARK = 0.33206

# Tight absolute tolerance: solve_bvp with tol=1e-8 (as configured in
# zpg_blasius.py) should reproduce the benchmark to ~1e-4 or better.
TOLERANCE = 1.0e-4


def test_blasius_wall_shear_matches_benchmark():
    """
    f''(0), the dimensionless wall-shear parameter, must match the
    accepted Blasius/Howarth benchmark value of 0.33206 to within a tight
    tolerance.
    """
    _eta, _f, _fp, fpp = _solve_blasius_similarity()
    f_double_prime_0 = fpp[0]

    assert f_double_prime_0 == pytest.approx(
        BLASIUS_F_DOUBLE_PRIME_0_BENCHMARK, abs=TOLERANCE
    ), (
        f"Blasius wall-shear parameter f''(0) = {f_double_prime_0:.6f} "
        f"deviates from the benchmark {BLASIUS_F_DOUBLE_PRIME_0_BENCHMARK} "
        f"by more than {TOLERANCE}."
    )


def test_blasius_module_reference_constant_matches_benchmark():
    """
    The module's own internal reference constant
    (`_BLASIUS_F_DOUBLE_PRIME_0_REFERENCE`, used as an initial-guess seed
    and documentation anchor in zpg_blasius.py) should itself equal the
    accepted benchmark -- guards against the constant silently drifting
    out of sync with the literature value during future edits.
    """
    assert _BLASIUS_F_DOUBLE_PRIME_0_REFERENCE == pytest.approx(
        BLASIUS_F_DOUBLE_PRIME_0_BENCHMARK, abs=1e-6
    )


def test_blasius_boundary_conditions_satisfied():
    """
    Sanity-checks the solved profile actually satisfies the BVP's boundary
    conditions: no-slip/no-penetration at the wall (f(0)=0, f'(0)=0) and
    free-stream recovery far from the wall (f'(eta_max) -> 1).
    """
    _eta, f, fp, _fpp = _solve_blasius_similarity()

    assert f[0] == pytest.approx(0.0, abs=1e-6)
    assert fp[0] == pytest.approx(0.0, abs=1e-6)
    assert fp[-1] == pytest.approx(1.0, abs=1e-3)


def test_blasius_velocity_profile_is_monotonic():
    """
    Physical sanity check: u/U = f'(eta) should rise monotonically from 0
    at the wall to 1 in the free stream, with no overshoot -- a basic
    signature of a correctly-solved (non-oscillatory) BVP solution.
    """
    _eta, _f, fp, _fpp = _solve_blasius_similarity()

    assert (fp >= -1e-8).all(), "f' should never be negative."
    assert (fp <= 1.0 + 1e-6).all(), "f' should never overshoot the free-stream value."
    # Monotonic non-decreasing, allowing tiny numerical noise.
    diffs = fp[1:] - fp[:-1]
    assert (diffs >= -1e-6).all(), "f' should be monotonically non-decreasing."
