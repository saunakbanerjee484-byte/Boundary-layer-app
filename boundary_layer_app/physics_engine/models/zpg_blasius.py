"""
physics_engine/models/zpg_blasius.py
======================================
Model 1: Blasius Exact Solution (Laminar Baseline, Zero Pressure Gradient).

Solves the Blasius similarity ODE 2 f''' + f f'' = 0 via a boundary-value
solve (`scipy.integrate.solve_bvp`), which is the numerically robust modern
replacement for the classical "shooting method" (we still shoot in spirit --
solve_bvp iterates the missing initial slope f''(0) internally -- but we get
proper residual control and adaptive meshing for free).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_bvp

from config import BLASIUS_ETA_MAX, N_X_POINTS, N_Y_POINTS, Y_PLUS_MAX
from physics_engine.base_model import (
    BoundaryLayerModel,
    FlowConditions,
    GrowthData,
    ProfileData,
    ShearData,
)
from physics_engine.registry import register_model

# Blasius' own tabulated constant: f''(0), the dimensionless wall shear.
# This is the value the BVP solve should reproduce (used only as a sanity
# fallback if the numerical solve is ever degenerate).
_BLASIUS_F_DOUBLE_PRIME_0_REFERENCE = 0.33206


def _solve_blasius_similarity():
    """
    Solves the Blasius similarity ODE by rewriting it as a first-order
    system in state vector y = [f, f', f''] over similarity variable eta:

        y0' = y1                (definition: f'  = y1)
        y1' = y2                (definition: f'' = y2)
        y2' = -0.5 * y0 * y2    (the ODE itself: 2f''' + f f'' = 0
                                  =>  f''' = -0.5 f f'')

    Boundary conditions (no-slip + far-field free-stream recovery):
        f(0)  = 0      (wall is impermeable, no transpiration)
        f'(0) = 0      (no-slip: u = 0 at the wall)
        f'(eta_max) = 1  (u -> U far from the wall)

    Returns the eta grid and the solved f, f', f'' arrays.
    """

    def ode_system(eta, y):
        f, fp, fpp = y
        # Blasius ODE rearranged for f''' (third derivative):
        # 2 f''' = -f f''  =>  f''' = -0.5 * f * f''
        fppp = -0.5 * f * fpp
        return np.vstack([fp, fpp, fppp])

    def boundary_conditions(y_at_0, y_at_end):
        # y_at_0 = [f(0), f'(0), f''(0)]; y_at_end = [f(eta_max), f'(eta_max), f''(eta_max)]
        return np.array(
            [
                y_at_0[0],          # f(0)  = 0
                y_at_0[1],          # f'(0) = 0
                y_at_end[1] - 1.0,  # f'(eta_max) = 1  (free-stream recovery)
            ]
        )

    eta_mesh = np.linspace(0.0, BLASIUS_ETA_MAX, 400)
    # Initial guess: a smooth monotone ramp for f' from 0 to 1, f from 0
    # growing, f'' starting near the known reference wall-curvature value
    # and decaying -- this keeps solve_bvp's Newton iteration well-behaved.
    y_guess = np.zeros((3, eta_mesh.size))
    y_guess[0] = eta_mesh - (1 - np.exp(-eta_mesh))          # f
    y_guess[1] = 1 - np.exp(-eta_mesh)                        # f'
    y_guess[2] = _BLASIUS_F_DOUBLE_PRIME_0_REFERENCE * np.exp(-eta_mesh)  # f''

    solution = solve_bvp(ode_system, boundary_conditions, eta_mesh, y_guess, tol=1e-8)
    eta = np.linspace(0.0, BLASIUS_ETA_MAX, N_Y_POINTS)
    f, fp, fpp = solution.sol(eta)
    return eta, f, fp, fpp


@register_model("zpg_blasius")
class BlasiusModel(BoundaryLayerModel):
    name = "Blasius Exact Solution (Laminar ZPG)"
    category = "Laminar - Zero Pressure Gradient"
    description = (
        "Exact similarity solution of the laminar boundary-layer equations "
        "for a flat plate with no streamwise pressure gradient."
    )

    def latex_formula(self) -> str:
        return r"2f''' + f f'' = 0 \qquad f(0)=0,\; f'(0)=0,\; f'(\infty)=1"

    def compute_profile(self, conditions: FlowConditions) -> ProfileData:
        # We evaluate the similarity solution at the trailing edge (x = x_max)
        # to obtain a single representative wall-shear value, which we then
        # use to non-dimensionalize the profile into "wall units" (y+, u+)
        # so it can be shown on the same semi-log axes as the turbulent
        # models, even though laminar flow has no true log-law region.
        eta, f, fp, fpp = _solve_blasius_similarity()

        x_ref = max(conditions.x_max, 1e-6)
        # Local Reynolds number based on distance from the leading edge.
        Re_x = conditions.U * x_ref / conditions.nu
        # Wall shear stress from the similarity solution:
        # tau_w = mu * U * sqrt(U / (nu x)) * f''(0)
        mu = conditions.rho * conditions.nu
        f_pp_wall = fpp[0]
        tau_w = mu * conditions.U * np.sqrt(conditions.U / (conditions.nu * x_ref)) * f_pp_wall
        tau_w = max(tau_w, 1e-12)  # guard against div-by-zero downstream
        u_tau = np.sqrt(tau_w / conditions.rho)

        # Physical wall-normal coordinate corresponding to similarity
        # variable eta: y = eta * sqrt(nu x / U)
        y_phys = eta * np.sqrt(conditions.nu * x_ref / conditions.U)

        y_plus = y_phys * u_tau / conditions.nu
        u_plus = (fp * conditions.U) / u_tau  # u/u_tau, with u = U * f'

        # Clip to the standard plotting window and drop the y+=0 point
        # (log-x axis cannot render zero).
        mask = (y_plus > 1e-3) & (y_plus <= Y_PLUS_MAX)
        return ProfileData(y_plus=y_plus[mask], u_plus=u_plus[mask])

    def compute_growth(self, conditions: FlowConditions) -> GrowthData:
        x = np.linspace(1e-4 * conditions.x_max, conditions.x_max, N_X_POINTS)
        # Blasius 99%-thickness correlation: eta ~= 4.91 is where f' = 0.99
        # (tabulated result of the similarity solution), giving the classic
        # delta(x) = 4.91 * sqrt(nu x / U) growth law (delta ~ sqrt(x)).
        eta_99 = 4.91
        delta = eta_99 * np.sqrt(conditions.nu * x / conditions.U)
        return GrowthData(x=x, delta=delta)

    def compute_shear(self, conditions: FlowConditions) -> ShearData:
        x = np.linspace(1e-4 * conditions.x_max, conditions.x_max, N_X_POINTS)
        mu = conditions.rho * conditions.nu
        # tau_w(x) = 0.332 * mu * U * sqrt(U / (nu x))  (Blasius wall-shear law)
        tau_w = (
            _BLASIUS_F_DOUBLE_PRIME_0_REFERENCE
            * mu
            * conditions.U
            * np.sqrt(conditions.U / (conditions.nu * x))
        )
        # Pure ZPG laminar flow never separates (tau_w > 0 everywhere), so
        # we report no separation point regardless of x.
        return ShearData(x=x, tau_w=tau_w, x_separation=None, separated=False)
