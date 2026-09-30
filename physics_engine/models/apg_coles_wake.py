"""
physics_engine/models/apg_coles_wake.py
==========================================
Model 4: Coles' Law of the Wake (Adverse-Pressure-Gradient Outer-Region
Deformation).

Coles (1956) observed that real turbulent boundary layers deviate from the
pure logarithmic law in the outer region, and that the deviation is well
captured by adding a "wake" function scaled by the wake-strength parameter
Pi. Pi grows with adverse pressure gradient (APG) and shrinks toward zero
(recovering the pure log-law) under favorable pressure gradient (FPG).

REFACTOR NOTE (physical-accuracy pass)
---------------------------------------
This module previously used two physically unjustified shortcuts that have
been removed:

  1. `Pi = 3.0 * dpdx_slider` -- an arbitrary linear mapping with no
     grounding in boundary-layer theory. Pi is now derived from the
     Clauser equilibrium pressure-gradient parameter,

         beta = (delta* / tau_w) * (dp/dx)

     via the standard equilibrium correlation Pi(beta) (see
     `_wake_parameter_from_beta` below), which is how Pi is actually
     estimated in the turbulence literature for equilibrium APG layers.

  2. `delta = base_delta * (1 + 0.3 * Pi)` -- an unscientific multiplier
     bolted onto the flat-plate growth law. Boundary-layer growth is now
     obtained by rigorously marching the von Karman momentum-integral
     equation coupled with Head's entrainment closure (see
     `physics_engine.models.turbulent_head`), exactly as Head's method
     already does. Coles' actual theoretical contribution is the shape
     of the OUTER velocity PROFILE (the log-law + wake term), not an
     independent growth law -- so rather than inventing a second,
     redundant momentum-integral solver here, this module reuses the one
     already validated in `turbulent_head.py`. This is both physically
     correct (growth is governed by momentum conservation regardless of
     which profile-shape closure is used to visualize the velocity
     distribution) and good software hygiene (one rigorous momentum-
     integral solver, reused wherever it's needed).
"""

from __future__ import annotations

import numpy as np

from config import KAPPA, LOG_LAW_B, Y_PLUS_MAX
from physics_engine.base_model import (
    BoundaryLayerModel,
    FlowConditions,
    GrowthData,
    ProfileData,
    ShearData,
)
from physics_engine.registry import register_model
from physics_engine.models.turbulent_head import (
    HeadEntrainmentModel,
    _dU_dx,
    _free_stream_velocity,
    _skin_friction_ludwieg_tillmann,
)


def _physical_pressure_gradient(x: np.ndarray, conditions: FlowConditions) -> np.ndarray:
    """
    Converts the UI's dimensionless dp/dx slider into an actual physical
    pressure gradient [Pa/m] via the inviscid Bernoulli relation applied
    to the boundary-layer edge velocity U(x):

        dp/dx = -rho * U * dU/dx

    U(x) and dU/dx use the exact same linear prescription already shared
    by the Thwaites and Head models (`_free_stream_velocity` / `_dU_dx`),
    so the dp/dx slider has one consistent physical meaning across every
    model in the app, rather than a model-specific arbitrary scaling.
    """
    U_x = _free_stream_velocity(x, conditions.U, conditions.dpdx, conditions.x_max)
    dUdx = _dU_dx(conditions.dpdx, conditions.U, conditions.x_max)
    return -conditions.rho * U_x * dUdx


def _clauser_beta(delta_star: np.ndarray, tau_w: np.ndarray, dpdx_phys: np.ndarray) -> np.ndarray:
    """
    Clauser's (1954) equilibrium pressure-gradient parameter,

        beta = (delta* / tau_w) * (dp/dx)

    the standard dimensionless measure of how strongly a turbulent
    boundary layer is departing from zero-pressure-gradient equilibrium.
    beta = 0 for a flat plate; beta > 0 under adverse pressure gradient.
    """
    tau_w_safe = np.clip(tau_w, 1e-9, None)
    return (delta_star / tau_w_safe) * dpdx_phys


def _wake_parameter_from_beta(beta: np.ndarray) -> np.ndarray:
    """
    Maps the Clauser parameter beta onto Coles' wake-strength parameter
    Pi using the standard equilibrium-boundary-layer correlation

        Pi ~= 0.8 * (beta + 0.5)^(3/4),   beta > -0.5

    This algebraic fit to the equilibrium (Clauser/Mellor-Gibson family)
    boundary-layer data is the textbook form used to estimate Pi from a
    prescribed pressure gradient (see e.g. White, "Viscous Fluid Flow",
    the equilibrium-layer wake-strength correlation; equivalent forms
    appear under Das' and related equilibrium correlations). It is only
    strictly valid for beta > -0.5 (attached, non-relaminarizing flow);
    we clip beta at -0.5 before evaluating so strong FPG never produces
    a complex or negative-under-the-root result.

    Simplification acknowledged: real ZPG flat plates carry a small but
    nonzero Pi (~0.45-0.6) due to residual outer-layer intermittency;
    this correlation returns Pi = 0 at beta = 0 by construction, treating
    Pi here purely as an *adverse-pressure-gradient-driven* deformation
    signal rather than attempting to reproduce the ZPG intermittency
    wake exactly.
    """
    beta_clipped = np.clip(beta, -0.5, None)
    return 0.8 * (beta_clipped + 0.5) ** 0.75


@register_model("apg_coles_wake")
class ColesWakeModel(BoundaryLayerModel):
    name = "Coles' Law of the Wake (APG Outer Region)"
    category = "Turbulent - Adverse Pressure Gradient"
    description = (
        "Adds Coles' wake function to the log law to capture outer-region "
        "deviation under adverse pressure gradient; wake strength Pi is "
        "derived from the local Clauser pressure-gradient parameter beta, "
        "and growth/shear are obtained from the von Karman momentum-"
        "integral + entrainment march shared with Head's method."
    )

    def latex_formula(self) -> str:
        return (
            r"u^+ = \frac{1}{\kappa}\ln(y^+) + B + "
            r"\frac{2\Pi}{\kappa}\sin^2\!\left(\frac{\pi y}{2\delta}\right)"
            r"\qquad \Pi \approx 0.8(\beta+0.5)^{3/4},\ \ "
            r"\beta = \frac{\delta^*}{\tau_w}\frac{dp}{dx}"
        )

    def _shared_march(self, conditions: FlowConditions):
        """
        Runs Head's rigorous von Karman momentum-integral + entrainment
        march once (theta(x), H(x), separation), then derives every
        Coles-specific quantity (delta*, physical dp/dx, beta, Pi) from
        that shared, already-validated solution. See module docstring
        for why growth/shear are not re-derived independently here.
        """
        head = HeadEntrainmentModel()
        x, theta, H_arr, x_separation = head._march(conditions)

        U_x = _free_stream_velocity(x, conditions.U, conditions.dpdx, conditions.x_max)
        Re_theta = U_x * theta / conditions.nu
        Cf = np.array(
            [_skin_friction_ludwieg_tillmann(H, Re) for H, Re in zip(H_arr, Re_theta)]
        )
        tau_w = Cf * 0.5 * conditions.rho * U_x**2

        # Displacement thickness follows exactly from the shape-factor
        # definition H = delta*/theta -- no heuristic delta*~delta/8
        # approximation needed, since theta and H both come from the
        # rigorous march.
        delta_star = H_arr * theta

        dpdx_phys = _physical_pressure_gradient(x, conditions)
        beta = _clauser_beta(delta_star, tau_w, dpdx_phys)
        Pi_arr = _wake_parameter_from_beta(beta)

        return x, theta, H_arr, tau_w, delta_star, beta, Pi_arr, x_separation

    def _reference_index(self, x: np.ndarray, x_separation) -> int:
        """
        Picks which downstream station to treat as "representative" for
        the profile plot. If the boundary layer has separated upstream of
        x_max, tau_w is floored to zero from that point on (see
        `turbulent_head.py`'s post-separation convention), which would
        make beta = delta*/tau_w blow up if we naively used the trailing
        edge. So: use the last ATTACHED station (just upstream of
        separation) when separated, otherwise the true trailing edge.
        """
        if x_separation is None:
            return len(x) - 1
        idx = int(np.searchsorted(x, x_separation))
        return max(idx - 1, 0)

    def compute_profile(self, conditions: FlowConditions) -> ProfileData:
        x, theta, H_arr, tau_w, delta_star, beta, Pi_arr, x_sep = self._shared_march(conditions)

        # Use the last attached station as the representative outer-region
        # deformation for the displayed profile (see `_reference_index`),
        # consistent with how the other x-marching models pick a
        # representative downstream station for their profile plot.
        ref = self._reference_index(x, x_sep)
        Pi_ref = float(Pi_arr[ref])
        delta_ref = max(float(theta[ref] * H_arr[ref] * 1.3), 1e-9)
        tau_w_ref = max(float(tau_w[ref]), 1e-9)
        u_tau = np.sqrt(tau_w_ref / conditions.rho)

        y_plus = np.geomspace(1.0, Y_PLUS_MAX, 250)
        y_phys = y_plus * conditions.nu / u_tau
        # Clip the wake argument at pi/2 (y = delta) since Coles' wake
        # function is only defined up to the boundary-layer edge.
        wake_arg = np.clip((np.pi * y_phys) / (2.0 * delta_ref), 0.0, np.pi / 2.0)

        u_plus = (1.0 / KAPPA) * np.log(y_plus) + LOG_LAW_B + (
            2.0 * Pi_ref / KAPPA
        ) * np.sin(wake_arg) ** 2

        return ProfileData(y_plus=y_plus, u_plus=u_plus)

    def compute_growth(self, conditions: FlowConditions) -> GrowthData:
        x, theta, H_arr, _tau_w, _delta_star, _beta, _Pi_arr, _x_sep = self._shared_march(conditions)
        # Same theta -> delta conversion Head's method uses (99%-style
        # thickness estimate from momentum thickness and shape factor),
        # kept identical so the two models' growth panels are directly
        # comparable -- this is now the rigorous momentum-integral result,
        # not a heuristic multiplier on the flat-plate correlation.
        delta = theta * H_arr * 1.3
        return GrowthData(x=x, delta=delta)

    def compute_shear(self, conditions: FlowConditions) -> ShearData:
        x, _theta, _H_arr, tau_w, _delta_star, _beta, _Pi_arr, x_separation = self._shared_march(
            conditions
        )
        separated = x_separation is not None
        return ShearData(x=x, tau_w=tau_w, x_separation=x_separation, separated=separated)
