"""
physics_engine/models/laminar_thwaites.py
============================================
Model 5: Thwaites' Method (Laminar Separation Predictor).

Thwaites (1949) derived an approximate closed-form integral method for the
laminar momentum-thickness growth theta(x) under an arbitrary prescribed
free-stream velocity U(x), avoiding a full numerical solve of the boundary-
layer PDEs. The momentum-thickness Reynolds parameter

    lambda = (theta^2 / nu) * dU/dx

then predicts separation once lambda drops to -0.09 (Thwaites' empirically
calibrated threshold).

Because this app's sidebar only exposes a single scalar dp/dx slider, we
prescribe a simple linear free-stream velocity variation
U(x) = U0 * (1 - dpdx_slider * x / x_max), which is decelerating
(adverse gradient) for dpdx_slider > 0 and accelerating (favorable
gradient) for dpdx_slider < 0 -- exactly the qualitative behavior the
slider is meant to control.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.integrate import quad

from config import N_X_POINTS, THWAITES_SEPARATION_LAMBDA
from physics_engine.base_model import (
    BoundaryLayerModel,
    FlowConditions,
    GrowthData,
    ProfileData,
    ShearData,
)
from physics_engine.registry import register_model

# Thwaites' calibration constant relating theta^2 growth to the integral of
# U^5 along the plate (from curve-fitting a family of exact/near-exact
# laminar similarity solutions).
_THWAITES_CONSTANT = 0.45


def _free_stream_velocity(x: np.ndarray, U0: float, dpdx_slider: float, x_max: float):
    """
    Prescribed outer-flow velocity U(x): linear ramp controlled by the
    dp/dx slider. dpdx_slider > 0 decelerates the flow (adverse gradient,
    dp/dx > 0 via Bernoulli); dpdx_slider < 0 accelerates it (favorable
    gradient). Clipped at 5% of U0 to keep U(x) safely positive/nonzero for
    the numerics even under a strong nominal deceleration.
    """
    U = U0 * (1.0 - dpdx_slider * x / x_max)
    return np.clip(U, 0.05 * U0, None)


def _dU_dx(x: float, U0: float, dpdx_slider: float, x_max: float) -> float:
    """Analytic derivative of the linear U(x) law above (before clipping)."""
    return -U0 * dpdx_slider / x_max


def _theta_squared(x_eval: float, U0: float, nu: float, dpdx_slider: float, x_max: float) -> float:
    """
    Thwaites' closed-form integral for the momentum-thickness squared:

        theta(x)^2 = (0.45 * nu / U(x)^6) * integral_0^x U(x0)^5 dx0

    Evaluated with `scipy.integrate.quad` for the definite integral.
    """
    U_at_x = _free_stream_velocity(np.array([x_eval]), U0, dpdx_slider, x_max)[0]

    def integrand(x0):
        U0_local = _free_stream_velocity(np.array([x0]), U0, dpdx_slider, x_max)[0]
        return U0_local**5

    integral_value, _ = quad(integrand, 0.0, max(x_eval, 1e-9))
    theta_sq = (_THWAITES_CONSTANT * nu / (U_at_x**6)) * integral_value
    return max(theta_sq, 0.0)


def _thwaites_shear_function(lam: np.ndarray) -> np.ndarray:
    """
    Thwaites' empirical shear-correlation function l(lambda), tabulated
    from exact solutions and here approximated with his commonly-used
    polynomial fit:

        l(lambda) = 0.22 + 1.57*lambda - 1.8*lambda^2   for  -0.09 <= lambda <= 0.25

    Used to recover wall shear stress from theta and lambda via
    tau_w = mu * U / theta * l(lambda).
    """
    lam_clipped = np.clip(lam, -0.09, 0.25)
    return 0.22 + 1.57 * lam_clipped - 1.8 * lam_clipped**2


@register_model("laminar_thwaites")
class ThwaitesModel(BoundaryLayerModel):
    name = "Thwaites' Method (Laminar Separation Predictor)"
    category = "Laminar - Arbitrary Pressure Gradient"
    description = (
        "Approximate integral method for laminar boundary-layer growth "
        "under a prescribed pressure gradient; predicts separation via the "
        "momentum-thickness parameter lambda."
    )

    def latex_formula(self) -> str:
        return r"\lambda = \frac{\theta^2}{\nu}\frac{dU}{dx} \qquad \text{Separation at } \lambda = -0.09"

    def _theta_and_lambda(self, conditions: FlowConditions) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.linspace(1e-4 * conditions.x_max, conditions.x_max, N_X_POINTS)
        theta_sq = np.array(
            [
                _theta_squared(xi, conditions.U, conditions.nu, conditions.dpdx, conditions.x_max)
                for xi in x
            ]
        )
        theta = np.sqrt(theta_sq)
        dUdx = _dU_dx(x, conditions.U, conditions.dpdx, conditions.x_max)
        lam = (theta_sq / conditions.nu) * dUdx
        return x, theta, lam

    def compute_growth(self, conditions: FlowConditions) -> GrowthData:
        x, theta, _lam = self._theta_and_lambda(conditions)
        # Convert momentum thickness theta to a 99%-style boundary-layer
        # thickness delta using the classical laminar shape-factor ratio
        # delta / theta ~= 8 (typical for Blasius-like laminar profiles;
        # used here purely as a display scale, not as part of Thwaites'
        # separation calculation itself).
        delta = 8.0 * theta
        return GrowthData(x=x, delta=delta)

    def compute_shear(self, conditions: FlowConditions) -> ShearData:
        x, theta, lam = self._theta_and_lambda(conditions)
        U_x = _free_stream_velocity(x, conditions.U, conditions.dpdx, conditions.x_max)
        mu = conditions.rho * conditions.nu

        l_lambda = _thwaites_shear_function(lam)
        theta_safe = np.clip(theta, 1e-9, None)
        # Thwaites' wall-shear recovery formula: tau_w = mu * (U/theta) * l(lambda)
        tau_w = mu * (U_x / theta_safe) * l_lambda

        x_separation = None
        separated = False
        sep_indices = np.where(lam <= THWAITES_SEPARATION_LAMBDA)[0]
        if sep_indices.size > 0:
            idx = sep_indices[0]
            x_separation = float(x[idx])
            separated = True
            # Beyond the predicted separation point, Thwaites' method is no
            # longer valid (the flow has physically detached), so we zero
            # out tau_w downstream for a visually honest "flatline" rather
            # than extrapolating meaningless negative values.
            tau_w = tau_w.copy()
            tau_w[idx:] = 0.0

        return ShearData(x=x, tau_w=tau_w, x_separation=x_separation, separated=separated)

    def compute_profile(self, conditions: FlowConditions) -> ProfileData:
        # Thwaites' method characterizes integral momentum-thickness
        # growth, not the detailed inner-region velocity profile, so we
        # display a self-similar Blasius-like shape (Pohlhausen-style
        # quartic) evaluated at the trailing-edge station using this
        # model's own theta/tau_w there -- consistent in u_tau/delta, but
        # not re-deriving a full new profile theory.
        growth = self.compute_growth(conditions)
        shear = self.compute_shear(conditions)
        delta_ref = max(growth["delta"][-1], 1e-9)
        tau_w_ref = max(shear["tau_w"][-1], 1e-12)

        u_tau = np.sqrt(tau_w_ref / conditions.rho)
        y_phys = np.geomspace(1e-6 * delta_ref, delta_ref, 250)
        eta = np.clip(y_phys / delta_ref, 0.0, 1.0)
        # Pohlhausen quartic laminar velocity-profile shape:
        # u/U = 2*eta - 2*eta^3 + eta^4  (satisfies u(0)=0, u(delta)=U,
        # and zero curvature at the edge -- a standard laminar closure).
        u_over_U = 2 * eta - 2 * eta**3 + eta**4
        u = u_over_U * conditions.U

        y_plus = y_phys * u_tau / conditions.nu
        u_plus = u / u_tau
        mask = y_plus > 1e-3
        return ProfileData(y_plus=y_plus[mask], u_plus=u_plus[mask])
