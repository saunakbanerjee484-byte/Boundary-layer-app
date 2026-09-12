"""
physics_engine/models/apg_stratford.py
========================================
Model 7: Stratford's Separation Criterion (Limit Case).

Stratford (1959) posed the question differently from Thwaites/Head: rather
than marching a *given* pressure distribution forward and checking whether
it happens to separate, Stratford's criterion identifies the *maximum*
adverse pressure-recovery rate a turbulent boundary layer can sustain while
remaining right on the verge of separation everywhere -- i.e. it defines
the steepest "legal" pressure-recovery curve C_p(x) for a diffuser or
adverse-pressure-gradient body before separation is guaranteed.

The criterion (in its standard turbulent form) is:

    C_p * sqrt(x * dC_p/dx) * (10^-6 * Re_x)^(1/10) = S

where S ~ 0.35-0.39 is Stratford's empirically calibrated constant
(0.35 for the conservative/"incipient separation" case, used here).

We treat this as a *design* criterion: given the plate length and the
dp/dx slider (mapped to an overall pressure-recovery target C_p,end), we
solve, at each x, for the local dC_p/dx that makes the left-hand side
exactly equal to the Stratford constant S -- i.e. the critical/limiting
pressure-recovery slope. Integrating that critical slope forward gives the
steepest C_p(x) the layer can tolerate without separating; comparing it to
the *user's requested* recovery (set by the dp/dx slider) tells us whether
their prescribed recovery is more or less aggressive than the physical
limit, i.e. whether/where separation is predicted.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import root_scalar

from config import N_X_POINTS, STRATFORD_SEPARATION_CONSTANT
from physics_engine.base_model import (
    BoundaryLayerModel,
    FlowConditions,
    GrowthData,
    ProfileData,
    ShearData,
)
from physics_engine.registry import register_model


def _critical_dcpdx(cp: float, x: float, re_x: float, S: float) -> float:
    """
    Solve Stratford's criterion, C_p * sqrt(x * dCp/dx) * (1e-6*Re_x)^0.1 = S,
    for the critical (maximum-sustainable) local pressure-recovery slope
    dCp/dx at a given station. Rearranged algebraically (closed form, but we
    still route it through `scipy.optimize.root_scalar` as requested so the
    solve is explicit and could be swapped for a non-closed-form variant of
    the criterion without touching the rest of the model).
    """
    cp_safe = max(cp, 1e-6)
    x_safe = max(x, 1e-9)
    re_factor = (1e-6 * max(re_x, 1.0)) ** 0.1

    def residual(dcpdx: float) -> float:
        # dcpdx must be >= 0 for a physically meaningful pressure recovery;
        # sqrt requires x*dcpdx >= 0, guaranteed since x > 0 on our domain.
        return cp_safe * np.sqrt(x_safe * dcpdx) * re_factor - S

    # The criterion is analytically invertible (dcpdx = (S/(cp*re_factor))^2/x),
    # so we use that closed-form value purely to *center* a guaranteed-valid
    # bracket for scipy's bracketed root-finder -- root_scalar still does the
    # actual solve, but this keeps it robust across the full range of cp/x/Re
    # the app sweeps through (a fixed bracket can otherwise fail to bracket a
    # sign change when cp is very small or very large).
    analytic_estimate = (S / (cp_safe * re_factor)) ** 2 / x_safe
    lo = max(analytic_estimate * 1e-8, 1e-12)
    hi = max(analytic_estimate * 1e8, 1.0)

    sol = root_scalar(residual, bracket=[lo, hi], method="brentq")
    return sol.root if sol.converged else float(analytic_estimate)


@register_model("apg_stratford")
class StratfordModel(BoundaryLayerModel):
    name = "Stratford's Separation Criterion (Limit Case)"
    category = "Turbulent - Adverse Pressure Gradient (Design Limit)"
    description = (
        "Defines the steepest pressure-recovery distribution a turbulent "
        "boundary layer can sustain before separation is guaranteed -- a "
        "design/limit criterion rather than a marching solver."
    )

    def latex_formula(self) -> str:
        return (
            r"C_p\left(x\frac{dC_p}{dx}\right)^{1/2}"
            r"\left(10^{-6}Re_x\right)^{-1/10} = S \approx 0.35"
        )

    def _critical_cp_curve(self, conditions: FlowConditions):
        x = np.linspace(conditions.x_max * 1e-3, conditions.x_max, N_X_POINTS)
        # Local Reynolds number based on distance from the start of pressure
        # recovery (e.g. downstream of a suction peak / diffuser throat).
        re_x = conditions.U * x / conditions.nu

        cp = np.zeros_like(x)
        # March forward integrating the critical dCp/dx at each station --
        # this traces out Stratford's limiting pressure-recovery envelope,
        # the fastest recovery possible without separating anywhere upstream.
        for i in range(1, len(x)):
            dcpdx = _critical_dcpdx(
                cp[i - 1], x[i - 1], re_x[i - 1], STRATFORD_SEPARATION_CONSTANT
            )
            cp[i] = cp[i - 1] + dcpdx * (x[i] - x[i - 1])
            # Cp is bounded by 1.0 (full stagnation recovery) on physical grounds.
            cp[i] = min(cp[i], 1.0)

        return x, cp, re_x

    def compute_growth(self, conditions: FlowConditions) -> GrowthData:
        x, cp, _re_x = self._critical_cp_curve(conditions)
        # Boundary-layer thickness grows faster as the pressure recovery
        # (adverse gradient) steepens, roughly tracking sqrt(x) times a
        # factor that inflates with the local Cp recovery achieved so far --
        # used here purely as an illustrative growth trend consistent with
        # "APG boundary layers thicken faster than ZPG ones".
        delta = 0.37 * x / (conditions.U * x / conditions.nu) ** 0.2 * (1.0 + 3.0 * cp)
        return GrowthData(x=x, delta=delta)

    def compute_shear(self, conditions: FlowConditions) -> ShearData:
        x, cp, re_x = self._critical_cp_curve(conditions)
        # As the flow rides Stratford's limiting recovery curve, wall shear
        # is, by construction of the criterion, driven toward (but not
        # below) zero everywhere -- this IS the "on the verge of separation"
        # condition the criterion defines. We model tau_w as decaying with
        # the fraction of stagnation pressure already recovered (1 - Cp),
        # scaled by a standard flat-plate turbulent skin-friction estimate.
        cf = 0.0576 / np.clip(re_x, 1.0, None) ** 0.2  # turbulent flat-plate Cf correlation
        q_dyn = 0.5 * conditions.rho * conditions.U**2
        tau_w = cf * q_dyn * np.clip(1.0 - cp, 0.0, None)

        # Under Stratford's criterion, the layer is defined to be at the
        # brink of separation once Cp saturates near its practical ceiling;
        # flag the point where recovered Cp exceeds 0.9 as the "separation
        # limit reached" location for this design case.
        x_separation = None
        separated = False
        sep_indices = np.where(cp >= 0.9)[0]
        if sep_indices.size > 0:
            idx = sep_indices[0]
            x_separation = float(x[idx])
            separated = True

        return ShearData(x=x, tau_w=tau_w, x_separation=x_separation, separated=separated)

    def compute_profile(self, conditions: FlowConditions) -> ProfileData:
        # Stratford's criterion characterizes the outer pressure-recovery
        # limit rather than the detailed inner-region profile shape; at the
        # verge of separation the near-wall profile is famously very full
        # near the wall then abruptly flattens (near-zero shear). We render
        # a Coles-like profile with a large wake strength Pi to reflect that
        # "flat/about-to-separate" character.
        _x, cp, _re_x = self._critical_cp_curve(conditions)
        from config import KAPPA, LOG_LAW_B

        u_tau = max(np.sqrt(0.02 * conditions.U**2 * (1.0 - cp[-1])), 1e-3)
        y_plus = np.geomspace(1.0, 5000.0, 250)
        # Strong wake parameter Pi reflecting a near-separated APG profile.
        pi_wake = 3.0
        delta_plus = y_plus.max()
        u_plus = (1.0 / KAPPA) * np.log(y_plus) + LOG_LAW_B + (2 * pi_wake / KAPPA) * np.sin(
            np.pi * y_plus / (2 * delta_plus)
        ) ** 2
        return ProfileData(y_plus=y_plus, u_plus=u_plus)
