"""
config.py
=========
Global constants and physics defaults shared across the physics engine and
the UI layer. Centralizing these values means every model and every plot
uses a single, consistent set of universal constants (e.g. the von Kármán
constant) and a single set of sane default slider values.
"""

from typing import Final

# ---------------------------------------------------------------------------
# Universal turbulence constants
# ---------------------------------------------------------------------------
# The von Kármán constant (kappa) governs the slope of the logarithmic
# region of a turbulent boundary layer: u+ = (1/kappa) * ln(y+) + B.
# The canonical experimentally-fitted value is ~0.41.
KAPPA: Final[float] = 0.41

# The log-law intercept constant B (sometimes called the "smooth wall"
# additive constant). Standard value for hydraulically smooth walls.
LOG_LAW_B: Final[float] = 5.0

# ---------------------------------------------------------------------------
# Numerical defaults (used to size arrays / integration ranges)
# ---------------------------------------------------------------------------
# Number of points used for spatial (x) marching grids.
N_X_POINTS: Final[int] = 200

# Number of points used for wall-normal (y or y+) profile plots.
N_Y_POINTS: Final[int] = 250

# Upper bound of y+ used when plotting the non-dimensional inner-region
# profile (u+ vs y+). 1e4 comfortably spans viscous sublayer -> log layer
# -> outer wake region for typical Re.
Y_PLUS_MAX: Final[float] = 1.0e4

# Blasius similarity variable (eta) truncation. f'(eta) asymptotes to 1.0
# well before eta = 10, so this is a numerically safe "infinity".
BLASIUS_ETA_MAX: Final[float] = 10.0

# ---------------------------------------------------------------------------
# Default fluid properties (air at ~20 C, 1 atm, in SI units)
# ---------------------------------------------------------------------------
DEFAULT_NU: Final[float] = 1.5e-5   # kinematic viscosity [m^2/s]
DEFAULT_RHO: Final[float] = 1.225   # density [kg/m^3]
DEFAULT_U: Final[float] = 20.0      # free-stream velocity [m/s]
DEFAULT_DPDX: Final[float] = 0.0    # dimensionless pressure-gradient slider [-1, 1]
DEFAULT_X_MAX: Final[float] = 1.0   # streamwise domain length [m]

# ---------------------------------------------------------------------------
# Separation / stability thresholds used by multiple models
# ---------------------------------------------------------------------------
# Thwaites' laminar separation criterion: the flow is taken to separate
# when the dimensionless pressure-gradient parameter lambda drops to -0.09.
THWAITES_SEPARATION_LAMBDA: Final[float] = -0.09

# Head's method: turbulent separation is conventionally flagged once the
# shape factor H (= delta* / theta) climbs above ~2.4, beyond which Head's
# entrainment correlation becomes unreliable and real flows are essentially
# stalled.
HEAD_SEPARATION_H: Final[float] = 2.4

# Stratford's universal separation constant. Stratford (1959) found that
# turbulent separation under a prescribed pressure recovery occurs when the
# dimensionless group below reaches ~0.35 (0.39 for the "zero skin
# friction at separation" idealization used here).
STRATFORD_SEPARATION_CONSTANT: Final[float] = 0.35

# ---------------------------------------------------------------------------
# Sidebar slider bounds (kept generous but numerically safe for the
# CPU-only solvers -- e.g. Blasius' shooting method and Spalding's implicit
# root-find both stay well-conditioned across these ranges).
# ---------------------------------------------------------------------------
NU_MIN: Final[float] = 1.0e-6   # m^2/s, denser-than-water liquids
NU_MAX: Final[float] = 1.0e-3   # m^2/s, viscous oils
RHO_MIN: Final[float] = 0.5     # kg/m^3, light gases
RHO_MAX: Final[float] = 1500.0  # kg/m^3, dense liquids
U_MIN: Final[float] = 0.5       # m/s
U_MAX: Final[float] = 100.0     # m/s
X_MAX_MIN: Final[float] = 0.1   # m, shortest plate/body length offered
X_MAX_MAX: Final[float] = 5.0   # m, longest plate/body length offered

# ---------------------------------------------------------------------------
# UI constants
# ---------------------------------------------------------------------------
APP_TITLE: Final[str] = "Turbulent & Laminar Boundary-Layer Studio"
APP_ICON: Final[str] = "〰️"
