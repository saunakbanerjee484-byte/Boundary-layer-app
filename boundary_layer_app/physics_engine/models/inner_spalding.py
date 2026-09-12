"""
physics_engine/models/inner_spalding.py
==========================================
Model 3: Spalding's Single Formula (Seamless Inner-Region Profile).

Spalding (1961) proposed a single closed-form relation for y+(u+) that
smoothly blends the viscous sublayer (u+ = y+), the buffer layer, and the
logarithmic region into one continuous curve -- avoiding the need to stitch
together three separate piecewise laws. Because the formula is naturally
expressed as y+ = g(u+), and we want u+ as a function of y+ for plotting,
we invert it numerically per-point with Newton's method.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import newton

from config import KAPPA, LOG_LAW_B, N_X_POINTS, N_Y_POINTS, Y_PLUS_MAX
from physics_engine.base_model import (
    BoundaryLayerModel,
    FlowConditions,
    GrowthData,
    ProfileData,
    ShearData,
)
from physics_engine.registry import register_model


def _y_plus_from_u_plus(u_plus: float, kappa: float, B: float) -> float:
    """
    Spalding's single formula, evaluated forward (the direction it is
    naturally defined in):

        y+ = u+ + exp(-kappa B) * [ exp(kappa u+) - 1 - kappa u+
                                      - (kappa u+)^2 / 2 - (kappa u+)^3 / 6 ]

    The bracketed term is the first four terms of the Taylor expansion of
    exp(kappa u+) subtracted off, which is what makes the formula reduce to
    the linear viscous sublayer law (y+ = u+) for small u+, while
    asymptoting to the logarithmic law for large u+.
    """
    ku = kappa * u_plus
    correction = np.exp(-kappa * B) * (
        np.exp(ku) - 1.0 - ku - (ku**2) / 2.0 - (ku**3) / 6.0
    )
    return u_plus + correction


def _u_plus_from_y_plus(y_plus_target: float, kappa: float, B: float) -> float:
    """
    Inverts Spalding's formula for a single target y+ using Newton's
    method (`scipy.optimize.newton`), since y+(u+) has no closed-form
    inverse. The log-law value is used as the initial guess, which is
    accurate for y+ >= ~30 and still a reasonable starting point closer to
    the wall because Newton's method converges quickly for this smooth,
    monotonic function.
    """

    def residual(u_plus):
        return _y_plus_from_u_plus(u_plus, kappa, B) - y_plus_target

    # Log-law initial guess; for very small y+ this can go negative, so we
    # floor it at a small positive viscous-sublayer estimate instead.
    log_law_guess = (1.0 / kappa) * np.log(max(y_plus_target, 1e-6)) + B
    initial_guess = log_law_guess if log_law_guess > 0.5 else max(y_plus_target, 1e-3)

    return newton(residual, x0=initial_guess, tol=1e-8, maxiter=100)


@register_model("inner_spalding")
class SpaldingModel(BoundaryLayerModel):
    name = "Spalding's Single Formula (Inner Region)"
    category = "Turbulent - Inner Region (Universal)"
    description = (
        "A single continuous formula spanning the viscous sublayer, buffer "
        "layer, and log-law region -- no piecewise stitching required."
    )

    def latex_formula(self) -> str:
        return (
            r"y^+ = u^+ + e^{-\kappa B}\left["
            r"e^{\kappa u^+} - 1 - \kappa u^+ - \frac{(\kappa u^+)^2}{2} "
            r"- \frac{(\kappa u^+)^3}{6}\right]"
        )

    def compute_profile(self, conditions: FlowConditions) -> ProfileData:
        # Target y+ grid (log-spaced to resolve the near-wall region well),
        # then invert Spalding's formula pointwise to get u+.
        y_plus = np.geomspace(0.05, Y_PLUS_MAX, N_Y_POINTS)
        u_plus = np.array(
            [_u_plus_from_y_plus(yp, KAPPA, LOG_LAW_B) for yp in y_plus]
        )
        return ProfileData(y_plus=y_plus, u_plus=u_plus)

    def compute_growth(self, conditions: FlowConditions) -> GrowthData:
        # Spalding's formula describes the *inner-region shape*, not the
        # outer growth law; we pair it with the same flat-plate turbulent
        # growth correlation used by the power-law model so the "Boundary
        # Layer Growth" panel remains physically meaningful for this model.
        x = np.linspace(1e-3 * conditions.x_max, conditions.x_max, N_X_POINTS)
        Re_x = conditions.U * x / conditions.nu
        delta = 0.37 * x / np.power(Re_x, 1.0 / 5.0)
        return GrowthData(x=x, delta=delta)

    def compute_shear(self, conditions: FlowConditions) -> ShearData:
        x = np.linspace(1e-3 * conditions.x_max, conditions.x_max, N_X_POINTS)
        Re_x = conditions.U * x / conditions.nu
        # Schlichting's turbulent flat-plate skin-friction correlation,
        # paired here because it is derived from a log-law-consistent
        # velocity profile, matching Spalding's log-law asymptote.
        Cf = 0.0576 / np.power(Re_x, 1.0 / 5.0)
        tau_w = Cf * 0.5 * conditions.rho * conditions.U**2
        return ShearData(x=x, tau_w=tau_w, x_separation=None, separated=False)
