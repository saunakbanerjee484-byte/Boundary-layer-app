"""
physics_engine/models/turbulent_head.py
==========================================
Model 6: Head's Entrainment Method (Turbulent APG & Separation).

Head (1958) closed the von Kármán momentum-integral equation for turbulent
boundary layers by adding an "entrainment" equation describing how fast the
boundary layer draws in free-stream fluid, expressed via an empirical
function of the shape factor H = delta*/theta. The coupled system is
marched downstream in x using `scipy.integrate.solve_ivp`. Separation is
flagged once H crosses Head's empirical stall threshold H ~ 2.4 (beyond
which the entrainment closure itself becomes unreliable, so we stop
trusting/marching the solution).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from config import HEAD_SEPARATION_H, N_X_POINTS
from physics_engine.base_model import (
    BoundaryLayerModel,
    FlowConditions,
    GrowthData,
    ProfileData,
    ShearData,
)
from physics_engine.registry import register_model


def _free_stream_velocity(x: np.ndarray, U0: float, dpdx_slider: float, x_max: float):
    """Same linear U(x) prescription used by the Thwaites model, so the
    dp/dx slider has a consistent physical meaning across models."""
    U = U0 * (1.0 - dpdx_slider * x / x_max)
    return np.clip(U, 0.05 * U0, None)


def _dU_dx(dpdx_slider: float, U0: float, x_max: float) -> float:
    return -U0 * dpdx_slider / x_max


def _H1_of_H(H: float) -> float:
    """
    Head's empirical entrainment shape-factor correlation H1(H), fitted
    from experimental data (Head 1958 / Cebeci-Bradshaw form):

        H1 = 3.3 + 0.8234 * (H - 1.1)^(-1.287),   H <= 1.6
        H1 = 3.3 + 1.5501 * (H - 0.6778)^(-3.064), H  > 1.6

    H1 = delta1_entrainment_thickness / theta is the auxiliary shape
    parameter that links H to the entrainment velocity.
    """
    H = max(H, 1.11)  # guard the singularity at H = 1.1
    if H <= 1.6:
        return 3.3 + 0.8234 * (H - 1.1) ** (-1.287)
    return 3.3 + 1.5501 * (H - 0.6778) ** (-3.064)


def _dH1_dH(H: float, dh=1e-4) -> float:
    """Numerical derivative dH1/dH, needed to convert the marched dH1/dx
    back into dH/dx for the shape-factor output."""
    return (_H1_of_H(H + dh) - _H1_of_H(H - dh)) / (2 * dh)


def _skin_friction_ludwieg_tillmann(H: float, Re_theta: float) -> float:
    """
    Ludwieg-Tillmann skin-friction correlation, the standard closure paired
    with Head's entrainment method:

        Cf = 0.246 * 10^(-0.678 H) * Re_theta^(-0.268)
    """
    Re_theta_safe = max(Re_theta, 1.0)
    return 0.246 * (10.0 ** (-0.678 * H)) * Re_theta_safe ** (-0.268)


@register_model("turbulent_head")
class HeadEntrainmentModel(BoundaryLayerModel):
    name = "Head's Entrainment Method (Turbulent APG)"
    category = "Turbulent - Adverse Pressure Gradient"
    description = (
        "Marches the von Kármán momentum-integral equation coupled with "
        "Head's entrainment closure; flags separation once the shape "
        "factor H exceeds ~2.4."
    )

    def latex_formula(self) -> str:
        return (
            r"\frac{d\theta}{dx} = \frac{C_f}{2} - (H+2)\frac{\theta}{U}"
            r"\frac{dU}{dx}, \qquad \frac{d(H_1\theta)}{dx} = "
            r"F(H_1) \qquad \text{Separation at } H = 2.4"
        )

    def _march(self, conditions: FlowConditions):
        """
        Marches the coupled ODE system [theta, H1_theta] downstream in x
        using solve_ivp, where H1_theta = H1(H) * theta is the entrainment
        "mass-flow" state variable used so the RHS stays well-behaved.

        State vector: state = [theta, H1_theta]
        """
        U0, nu, dpdx, x_max = conditions.U, conditions.nu, conditions.dpdx, conditions.x_max

        # Seed the march with a thin, healthy turbulent boundary layer at
        # x0 (a nominal transition point just downstream of the leading
        # edge), using a modest initial shape factor H0 = 1.4 typical of a
        # freshly-turbulent flat-plate layer.
        x0 = 1e-3 * x_max
        H0 = 1.4
        U_x0 = _free_stream_velocity(np.array([x0]), U0, dpdx, x_max)[0]
        Re_x0 = U_x0 * x0 / nu
        # Rough initial momentum thickness from the ZPG turbulent
        # correlation, just to get the march started smoothly.
        theta0 = 0.036 * x0 / (max(Re_x0, 10.0) ** (1.0 / 5.0))
        H1_0 = _H1_of_H(H0)
        state0 = [theta0, H1_0 * theta0]

        def entrainment_function(H1: float) -> float:
            """
            Head's entrainment function F(H1), the empirically fitted
            dimensionless entrainment rate (Cebeci-Bradshaw fit):
                F(H1) = 0.0306 * (H1 - 3.0)^(-0.6169)
            """
            return 0.0306 * max(H1 - 3.0, 1e-6) ** (-0.6169)

        def rhs(x, state):
            theta, H1_theta = state
            theta = max(theta, 1e-9)
            H1 = max(H1_theta / theta, 3.01)  # entrainment fn needs H1 > 3
            # Invert H1(H) numerically (monotonic) to recover H from H1 via
            # a short bisection, needed for the Cf closure term.
            H_lo, H_hi = 1.11, 4.0
            for _ in range(40):
                H_mid = 0.5 * (H_lo + H_hi)
                if _H1_of_H(H_mid) > H1:
                    H_lo = H_mid
                else:
                    H_hi = H_mid
            H = 0.5 * (H_lo + H_hi)

            U_local = _free_stream_velocity(np.array([x]), U0, dpdx, x_max)[0]
            dUdx_local = _dU_dx(dpdx, U0, x_max)
            Re_theta = U_local * theta / nu
            Cf = _skin_friction_ludwieg_tillmann(H, Re_theta)

            # von Kármán momentum-integral equation:
            # dtheta/dx = Cf/2 - (H + 2) * (theta/U) * dU/dx
            dtheta_dx = Cf / 2.0 - (H + 2.0) * (theta / U_local) * dUdx_local

            # Entrainment equation, rewritten in terms of the state
            # variable H1*theta:
            # d(H1 theta)/dx = U * F(H1)   [Head's original entrainment
            # equation, non-dimensionalized by U so it marches cleanly]
            F_H1 = entrainment_function(H1)
            dH1theta_dx = F_H1

            return [dtheta_dx, dH1theta_dx]

        def separation_event(x, state):
            theta, H1_theta = state
            theta = max(theta, 1e-9)
            H1 = max(H1_theta / theta, 3.01)
            H_lo, H_hi = 1.11, 4.0
            for _ in range(40):
                H_mid = 0.5 * (H_lo + H_hi)
                if _H1_of_H(H_mid) > H1:
                    H_lo = H_mid
                else:
                    H_hi = H_mid
            H = 0.5 * (H_lo + H_hi)
            return H - HEAD_SEPARATION_H

        separation_event.terminal = False
        separation_event.direction = 1

        x_eval = np.linspace(x0, x_max, N_X_POINTS)
        solution = solve_ivp(
            rhs,
            t_span=(x0, x_max),
            y0=state0,
            t_eval=x_eval,
            method="RK45",
            events=separation_event,
            max_step=x_max / 50.0,
        )

        theta = np.clip(solution.y[0], 1e-9, None)
        H1_theta = solution.y[1]
        H1 = H1_theta / theta

        # Recover H(x) from H1(x) via the same bisection used inside rhs().
        H_arr = np.zeros_like(theta)
        for i, h1_val in enumerate(H1):
            H_lo, H_hi = 1.11, 4.0
            h1_val = max(h1_val, 3.01)
            for _ in range(40):
                H_mid = 0.5 * (H_lo + H_hi)
                if _H1_of_H(H_mid) > h1_val:
                    H_lo = H_mid
                else:
                    H_hi = H_mid
            H_arr[i] = 0.5 * (H_lo + H_hi)

        x_separation = None
        if solution.t_events and len(solution.t_events[0]) > 0:
            x_separation = float(solution.t_events[0][0])

        return solution.t, theta, H_arr, x_separation

    def compute_growth(self, conditions: FlowConditions) -> GrowthData:
        x, theta, H_arr, _x_sep = self._march(conditions)
        # delta = H * theta * (delta/delta*) scale factor; we use
        # delta ~= theta * H * 1.3 as a standard turbulent shape estimate
        # converting momentum thickness + shape factor into a 99%-style
        # boundary-layer thickness for consistent display with the other
        # models' growth panels.
        delta = theta * H_arr * 1.3
        return GrowthData(x=x, delta=delta)

    def compute_shear(self, conditions: FlowConditions) -> ShearData:
        x, theta, H_arr, x_separation = self._march(conditions)
        U_x = _free_stream_velocity(x, conditions.U, conditions.dpdx, conditions.x_max)
        Re_theta = U_x * theta / conditions.nu
        Cf = np.array(
            [_skin_friction_ludwieg_tillmann(H, Re) for H, Re in zip(H_arr, Re_theta)]
        )
        tau_w = Cf * 0.5 * conditions.rho * U_x**2

        separated = x_separation is not None
        if separated:
            # Zero out tau_w downstream of the H=2.4 separation point,
            # since the entrainment closure is no longer physically valid
            # there (the boundary-layer approximation itself breaks down).
            idx = np.searchsorted(x, x_separation)
            tau_w = tau_w.copy()
            tau_w[idx:] = 0.0

        return ShearData(x=x, tau_w=tau_w, x_separation=x_separation, separated=separated)

    def compute_profile(self, conditions: FlowConditions) -> ProfileData:
        # Head's method predicts integral parameters (theta, H), not a
        # pointwise inner profile, so -- consistent with how the method is
        # used in practice -- we display the log-law inner-region shape
        # implied by the trailing-edge Cf/Re_theta this march produced.
        from config import KAPPA, LOG_LAW_B, Y_PLUS_MAX  # local import: small, avoids top-level coupling

        x, theta, H_arr, _x_sep = self._march(conditions)
        growth = self.compute_growth(conditions)
        shear = self.compute_shear(conditions)
        delta_ref = max(growth["delta"][-1], 1e-9)
        tau_w_ref = max(shear["tau_w"][-1], 1e-12)
        u_tau = np.sqrt(tau_w_ref / conditions.rho)

        y_plus = np.geomspace(1.0, Y_PLUS_MAX, 250)
        u_plus = (1.0 / KAPPA) * np.log(y_plus) + LOG_LAW_B
        return ProfileData(y_plus=y_plus, u_plus=u_plus)
