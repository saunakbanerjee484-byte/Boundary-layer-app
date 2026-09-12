"""
physics_engine/models/zpg_prandtl.py
======================================
Model 2: Prandtl's 1/7th Power Law (Turbulent ZPG Empirical Profile).

Purely algebraic NumPy array operations -- no ODE/BVP solve needed. This is
the classic engineering-correlation model for a turbulent flat-plate
boundary layer, valid over a useful (if moderate) Reynolds-number range.
"""

from __future__ import annotations

import numpy as np

from config import N_X_POINTS, N_Y_POINTS, Y_PLUS_MAX
from physics_engine.base_model import (
    BoundaryLayerModel,
    FlowConditions,
    GrowthData,
    ProfileData,
    ShearData,
)
from physics_engine.registry import register_model


@register_model("zpg_prandtl")
class PrandtlPowerLawModel(BoundaryLayerModel):
    name = "Prandtl's 1/7th Power Law (Turbulent ZPG)"
    category = "Turbulent - Zero Pressure Gradient"
    description = (
        "Empirical algebraic profile for a turbulent flat-plate boundary "
        "layer; a simple but historically foundational engineering fit."
    )

    def latex_formula(self) -> str:
        return r"\frac{\bar{u}}{U} = \left(\frac{y}{\delta}\right)^{1/7}"

    def compute_growth(self, conditions: FlowConditions) -> GrowthData:
        x = np.linspace(1e-3 * conditions.x_max, conditions.x_max, N_X_POINTS)
        Re_x = conditions.U * x / conditions.nu
        # Turbulent flat-plate growth correlation consistent with the 1/7th
        # power-law profile and a 1/7th-power wall-shear closure:
        # delta(x) = 0.37 * x / Re_x^(1/5)
        delta = 0.37 * x / np.power(Re_x, 1.0 / 5.0)
        return GrowthData(x=x, delta=delta)

    def compute_shear(self, conditions: FlowConditions) -> ShearData:
        x = np.linspace(1e-3 * conditions.x_max, conditions.x_max, N_X_POINTS)
        Re_x = conditions.U * x / conditions.nu
        # Local skin-friction coefficient consistent with the 1/7th power
        # law (Prandtl's original turbulent flat-plate closure):
        # Cf(x) = 0.0576 / Re_x^(1/5)
        Cf = 0.0576 / np.power(Re_x, 1.0 / 5.0)
        # Convert Cf back to a dimensional wall shear stress:
        # tau_w = Cf * 0.5 * rho * U^2
        tau_w = Cf * 0.5 * conditions.rho * conditions.U**2
        # A ZPG flat-plate flow driven purely by this empirical closure
        # never predicts separation (tau_w > 0 everywhere by construction).
        return ShearData(x=x, tau_w=tau_w, x_separation=None, separated=False)

    def compute_profile(self, conditions: FlowConditions) -> ProfileData:
        # Build the profile at the trailing edge (x = x_max) using the
        # local boundary-layer thickness and wall shear there, then convert
        # the physical y into wall units (y+) for the semi-log plot.
        growth = self.compute_growth(conditions)
        shear = self.compute_shear(conditions)
        delta_ref = growth["delta"][-1]
        tau_w_ref = shear["tau_w"][-1]

        u_tau = np.sqrt(max(tau_w_ref, 1e-12) / conditions.rho)

        # Sample the wall-normal coordinate geometrically so the log-x
        # profile plot is well-resolved near the wall.
        y_phys = np.geomspace(1e-6 * delta_ref, delta_ref, N_Y_POINTS)
        # 1/7th power-law velocity profile, clipped at y = delta (u = U
        # beyond the edge of the boundary layer).
        u_over_U = np.clip(y_phys / delta_ref, 0.0, 1.0) ** (1.0 / 7.0)
        u = u_over_U * conditions.U

        y_plus = y_phys * u_tau / conditions.nu
        u_plus = u / u_tau

        mask = (y_plus > 1e-3) & (y_plus <= Y_PLUS_MAX)
        return ProfileData(y_plus=y_plus[mask], u_plus=u_plus[mask])
