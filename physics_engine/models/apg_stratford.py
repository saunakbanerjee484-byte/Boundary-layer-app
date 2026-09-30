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

The criterion (standard turbulent form, Cebeci & Bradshaw / Schlichting):

    C_p * sqrt(x * dC_p/dx) * (10^-6 * Re_x)^(-1/10) = S

where S ~ 0.35-0.39 is Stratford's empirically calibrated constant
(0.35 for the conservative/"incipient separation" case, used here). Note
the NEGATIVE 1/10 exponent on the Reynolds-number factor: higher local
Re_x means a more energetic, mixing-rich turbulent layer that can sustain
a *steeper* pressure-recovery slope for the same margin S, i.e. the
allowable dC_p/dx must *increase* with Re_x. Since the Re-factor here
*decreases* with Re_x (negative exponent), dividing S by a smaller number
correctly yields a larger allowable dC_p/dx as Re_x grows. (An earlier
revision of this module used a POSITIVE exponent in the code while
displaying the correct negative exponent in the on-screen LaTeX -- a
plus/minus contradiction between what was computed and what was shown.
Both are now the same, literature-consistent negative exponent.)

REFACTOR NOTE (physical-accuracy pass)
---------------------------------------
Two issues have been corrected in this module:

  1. The per-step solve previously clamped `cp_safe = max(cp, 1e-6)`
     before evaluating the criterion. Since dC_p/dx ~ 1/cp^2, this
     artificial floor produced a huge, unphysical initial pressure-
     recovery slope at the very first station (where the true value is
     C_p = 0, i.e. no recovery has occurred yet). That floor has been
     removed. Instead, the ODE is regularized analytically: writing
     w = C_p^3 and differentiating, Stratford's relation becomes

         dw/dx = d(C_p^3)/dx = 3 * C_p^2 * dC_p/dx
                = 3 * S^2 / (x * re_factor(x)^2)

     which no longer contains C_p on the right-hand side at all -- the
     apparent singularity at C_p = 0 was only an artifact of solving for
     dC_p/dx directly, not a true singularity of the underlying physics.
     This regularized form gives a well-posed, physically defensible
     initial slope starting exactly from C_p(x0) = 0 (the correct
     physical initial condition: pressure recovery has not yet begun at
     the start of the domain), with no arbitrary floor required. Every
     station after the first has C_p > 0 naturally and is solved with
     the original (now correctly-signed) root-finding formulation.

  2. The exponent sign contradiction described above.
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


def _re_factor(re_x: float, negative_tenth_power: bool = True) -> float:
    """
    The Reynolds-number factor (10^-6 * Re_x)^(-1/10) from Stratford's
    criterion. Kept as its own function so the sign is defined in exactly
    one place and both the numeric solve and the docstring/LaTeX above
    stay in agreement by construction.
    """
    base = 1e-6 * max(re_x, 1.0)
    exponent = -0.1 if negative_tenth_power else 0.1
    return base ** exponent


def _critical_dcpdx(cp: float, x: float, re_x: float, S: float) -> float:
    """
    Solve Stratford's criterion, C_p * sqrt(x * dCp/dx) * re_factor = S,
    for the critical (maximum-sustainable) local pressure-recovery slope
    dCp/dx at a given station, given the ALREADY-POSITIVE C_p carried
    forward from the previous station (the singular first step, where
    C_p = 0, is handled separately in `_critical_cp_curve` via an exact
    closed-form integral -- see that method's docstring).
    """
    x_safe = max(x, 1e-9)
    re_f = _re_factor(re_x)

    def residual(dcpdx: float) -> float:
        # dcpdx must be >= 0 for a physically meaningful pressure recovery;
        # sqrt requires x*dcpdx >= 0, guaranteed since x > 0 on our domain.
        return cp * np.sqrt(x_safe * dcpdx) * re_f - S

    # The criterion is analytically invertible (dcpdx = (S/(cp*re_f))^2/x),
    # so we use that closed-form value purely to *center* a guaranteed-valid
    # bracket for scipy's bracketed root-finder -- root_scalar still does
    # the actual solve, but this keeps it robust across the full range of
    # cp/x/Re the app sweeps through.
    analytic_estimate = (S / (cp * re_f)) ** 2 / x_safe
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
        x = np.geomspace(conditions.x_max * 1e-3, conditions.x_max, N_X_POINTS)
        # Local Reynolds number based on distance from the start of pressure
        # recovery (e.g. downstream of a suction peak / diffuser throat).
        re_x = conditions.U * x / conditions.nu

        cp = np.zeros_like(x)
        # Physical initial condition: no pressure recovery has occurred at
        # the start of the domain, C_p(x[0]) = 0. Since Re_x = U*x/nu is
        # linear in x, re_factor(x) = (1e-6*Re_x)^(-1/10) is an exact power
        # law in x, so dw/dx = 3*S^2/(x*re_factor(x)^2) is also an exact
        # power law (~x^-0.8) and integrates in closed form over the first
        # interval -- giving an EXACT first step rather than a crude Euler
        # estimate (a single Euler step here was found to badly
        # over-predict the initial Cp rise on a linear grid, since the
        # true solution is steepest exactly where x is smallest). Using a
        # log-spaced grid (geomspace) additionally keeps every subsequent
        # step proportionally small near x0, where the criterion is most
        # sensitive to x.
        S = STRATFORD_SEPARATION_CONSTANT
        K = (1e-6 * conditions.U / conditions.nu) ** (-0.2)
        w1 = (15.0 * S**2 / K) * (x[1] ** 0.2 - x[0] ** 0.2)  # C_p(x[1])^3
        cp[1] = np.cbrt(max(w1, 0.0))

        # March forward integrating the critical dCp/dx at each remaining
        # station -- this traces out Stratford's limiting pressure-recovery
        # envelope, the fastest recovery possible without separating
        # anywhere upstream.
        for i in range(2, len(x)):
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
