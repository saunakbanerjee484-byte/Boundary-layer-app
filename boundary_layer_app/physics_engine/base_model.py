"""
physics_engine/base_model.py
=============================
Defines the abstract contract every boundary-layer model must satisfy, plus
the small `FlowConditions` data container that carries the user's sidebar
inputs down into the physics engine.

Keeping this contract narrow (profile / growth / shear, all in a common
shape) is what lets `app.py` and `ui/visualizations.py` stay completely
agnostic to *which* of the 7 models is currently selected -- they only ever
talk to `BoundaryLayerModel`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, TypedDict

import numpy as np


@dataclass(frozen=True)
class FlowConditions:
    """
    Immutable container for the sidebar-controlled flow state.

    Attributes:
        nu: Kinematic viscosity [m^2/s]. Controls how "thick" the viscous
            sublayer is relative to the outer flow.
        rho: Fluid density [kg/m^3]. Only enters via dynamic-pressure-based
            quantities (e.g. converting shear stress to a friction
            coefficient) -- the boundary-layer *shape* itself is a function
            of nu, U and dp/dx alone (incompressible assumption).
        U: Free-stream velocity [m/s] at the domain inlet (x = 0).
        dpdx: Dimensionless pressure-gradient slider in [-1, 1].
            0   -> Zero Pressure Gradient (ZPG), flat plate.
            >0  -> Adverse Pressure Gradient (APG), decelerating outer
                   flow, flow tends toward separation.
            <0  -> Favorable Pressure Gradient (FPG), accelerating outer
                   flow, stabilizing.
            Individual models map this slider onto their own physical
            pressure-gradient parameter (e.g. Coles' wake parameter Pi,
            or a prescribed U(x) profile for Thwaites/Head/Stratford).
        x_max: Streamwise domain length [m] used for spatial marching
            plots (growth, shear-vs-x).
    """

    nu: float
    rho: float
    U: float
    dpdx: float
    x_max: float = 1.0


class ProfileData(TypedDict):
    """Non-dimensional inner-region profile: u+ vs y+ (semi-log plot)."""

    y_plus: np.ndarray
    u_plus: np.ndarray


class GrowthData(TypedDict):
    """Physical boundary-layer thickness growth along the plate: delta(x)."""

    x: np.ndarray
    delta: np.ndarray


class ShearData(TypedDict, total=False):
    """
    Wall shear stress along the plate: tau_w(x), plus an optional flagged
    separation location where tau_w crosses zero (or the model's own
    separation criterion fires).
    """

    x: np.ndarray
    tau_w: np.ndarray
    x_separation: Optional[float]
    separated: bool


class BoundaryLayerModel(ABC):
    """
    Abstract base class that every physics model in `models/` must
    subclass and register (via `physics_engine.registry.register_model`).

    Subclasses supply:
      * Descriptive metadata (`name`, `category`, `description`) used to
        populate the sidebar selector.
      * `latex_formula()` -- the raw LaTeX string rendered via st.latex()
        so the user always sees the governing equation of the model that
        is currently driving the plots.
      * Three compute methods that all take the same `FlowConditions` and
        return plain-dict/np.ndarray payloads consumed by
        `ui/visualizations.py`. This uniform interface is what allows new
        models to be dropped into `models/` with zero changes to app.py.
    """

    #: Human-readable display name shown in the sidebar dropdown.
    name: str = "Unnamed Model"

    #: Physical regime bucket, purely for grouping/labeling in the UI
    #: (e.g. "Laminar - ZPG", "Turbulent - APG").
    category: str = "General"

    #: One or two sentence plain-English description shown as a caption.
    description: str = ""

    @abstractmethod
    def latex_formula(self) -> str:
        """Return the governing equation as a raw LaTeX string (no $ signs;
        st.latex() wraps it in display-math mode automatically)."""
        raise NotImplementedError

    @abstractmethod
    def compute_profile(self, conditions: FlowConditions) -> ProfileData:
        """Return the non-dimensional u+ vs y+ inner-region profile."""
        raise NotImplementedError

    @abstractmethod
    def compute_growth(self, conditions: FlowConditions) -> GrowthData:
        """Return the physical boundary-layer thickness delta(x)."""
        raise NotImplementedError

    @abstractmethod
    def compute_shear(self, conditions: FlowConditions) -> ShearData:
        """Return wall shear stress tau_w(x) and any separation point."""
        raise NotImplementedError
