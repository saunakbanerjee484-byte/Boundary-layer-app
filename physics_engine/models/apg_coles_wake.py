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
"""

from __future__ import annotations

import numpy as np

from config import KAPPA, LOG_LAW_B, N_X_POINTS, N_Y_POINTS, Y_PLUS_MAX
from physics_engine.base_model import (
    BoundaryLayerModel,
    FlowConditions,
    GrowthData,
    ProfileData,
    ShearData,
)
from physics_engine.registry import register_model


def _map_dpdx_to_wake_parameter(dpdx_slider: float) -> float:
    """
    Maps the UI's dimensionless dp/dx slider (range [-1, 1]) onto Coles'
    wake-strength parameter Pi.

    Physically: Pi ~ 0 for a zero-pressure-gradient flat plate, Pi ~ 0.45
    for the "equilibrium" mild-APG boundary layer Coles originally studied,
    and Pi can climb toward ~2-3 for strong APG flows approaching
    separation. Favorable pressure gradients (accelerating flow) suppress
    the wake and can drive Pi slightly negative in the equilibrium-sink-flow
    literature.

    We use a simple monotonic scaling: strong APG (slider -> +1) maps to
    Pi -> 3.0 (strong wake / near-separation outer deformation); strong FPG
    (slider -> -1) maps to Pi -> -0.5 (a suppressed, near-laminar-like
    outer region), with Pi = 0 at slider = 0 (pure flat-plate log-law).
    """
    dpdx = np.clip(dpdx_slider, -1.0, 1.0)
    if dpdx >= 0:
        return 3.0 * dpdx
    return -0.5 * abs(dpdx)


@register_model("apg_coles_wake")
class ColesWakeModel(BoundaryLayerModel):
    name = "Coles' Law of the Wake (APG Outer Region)"
    category = "Turbulent - Adverse Pressure Gradient"
    description = (
        "Adds Coles' wake function to the log law to capture outer-region "
        "deviation under adverse pressure gradient; wake strength Pi is "
        "driven by the dp/dx slider."
    )

    def latex_formula(self) -> str:
        return (
            r"u^+ = \frac{1}{\kappa}\ln(y^+) + B + "
            r"\frac{2\Pi}{\kappa}\sin^2\!\left(\frac{\pi y}{2\delta}\right)"
        )

    def compute_profile(self, conditions: FlowConditions) -> ProfileData:
        Pi = _map_dpdx_to_wake_parameter(conditions.dpdx)

        # Use the growth-law delta at the trailing edge as the outer length
        # scale needed by the wake term's y/delta argument.
        growth = self.compute_growth(conditions)
        delta_ref = max(growth["delta"][-1], 1e-9)

        y_plus = np.geomspace(1.0, Y_PLUS_MAX, N_Y_POINTS)

        # We need the *physical* y (not just y+) to evaluate the wake
        # term's y/delta argument. Estimate u_tau from the shear model to
        # convert y+ -> y_phys = y+ * nu / u_tau.
        shear = self.compute_shear(conditions)
        tau_w_ref = max(shear["tau_w"][-1], 1e-9)
        u_tau = np.sqrt(tau_w_ref / conditions.rho)

        y_phys = y_plus * conditions.nu / u_tau
        # Clip the wake argument at pi/2 (y = delta) since Coles' wake
        # function is only defined up to the boundary-layer edge.
        wake_arg = np.clip((np.pi * y_phys) / (2.0 * delta_ref), 0.0, np.pi / 2.0)

        # Coles' composite law: log-law base + wake correction.
        u_plus = (1.0 / KAPPA) * np.log(y_plus) + LOG_LAW_B + (
            2.0 * Pi / KAPPA
        ) * np.sin(wake_arg) ** 2

        return ProfileData(y_plus=y_plus, u_plus=u_plus)

    def compute_growth(self, conditions: FlowConditions) -> GrowthData:
        x = np.linspace(1e-3 * conditions.x_max, conditions.x_max, N_X_POINTS)
        Re_x = conditions.U * x / conditions.nu
        Pi = _map_dpdx_to_wake_parameter(conditions.dpdx)
        # Base turbulent flat-plate growth, inflated by a wake-strength
        # factor (1 + 2*Pi/kappa fraction) to reflect that a stronger wake
        # component corresponds to a physically thicker, more energetic
        # outer layer under APG (and a slightly thinner one under FPG).
        base_delta = 0.37 * x / np.power(Re_x, 1.0 / 5.0)
        wake_growth_factor = 1.0 + 0.3 * Pi
        delta = base_delta * np.clip(wake_growth_factor, 0.4, None)
        return GrowthData(x=x, delta=delta)

    def compute_shear(self, conditions: FlowConditions) -> ShearData:
        x = np.linspace(1e-3 * conditions.x_max, conditions.x_max, N_X_POINTS)
        Re_x = conditions.U * x / conditions.nu
        Pi = _map_dpdx_to_wake_parameter(conditions.dpdx)

        # Base ZPG turbulent skin friction (Schlichting correlation)...
        Cf_zpg = 0.0576 / np.power(Re_x, 1.0 / 5.0)
        # ...attenuated as the wake parameter (i.e. APG strength) grows,
        # since adverse pressure gradient progressively erodes wall shear
        # as the flow marches downstream toward separation. We model this
        # as a linear decay in x scaled by Pi, floored so tau_w can cross
        # (but not blow past) zero, which is what lets the separation
        # marker logic downstream detect the zero-crossing.
        decay = 1.0 - (Pi / 3.0) * (x / conditions.x_max) * 1.4
        Cf = Cf_zpg * decay
        tau_w = Cf * 0.5 * conditions.rho * conditions.U**2

        x_separation = None
        separated = False
        sign_changes = np.where(np.diff(np.sign(tau_w)))[0]
        if sign_changes.size > 0:
            idx = sign_changes[0]
            # Linear interpolation for the precise zero-crossing location.
            x0, x1 = x[idx], x[idx + 1]
            t0, t1 = tau_w[idx], tau_w[idx + 1]
            x_separation = float(x0 + (0.0 - t0) * (x1 - x0) / (t1 - t0))
            separated = True

        return ShearData(
            x=x, tau_w=tau_w, x_separation=x_separation, separated=separated
        )
