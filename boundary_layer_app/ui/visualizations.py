"""
ui/visualizations.py
=====================
Builds the three transparent, interactive Plotly figures used across every
model:

  1. Non-dimensional inner-region profile: u+ vs y+ (semi-log x-axis).
  2. Boundary-layer growth: delta(x) along the plate.
  3. Wall shear stress tau_w(x), with the zero-crossing / separation point
     highlighted in red.

Kept model-agnostic: every function takes the plain-dict payloads returned
by a `BoundaryLayerModel`'s `compute_*` methods (see `physics_engine
.base_model`), never a model instance itself. This is what lets `app.py`
swap models via the registry without ever touching this file.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from physics_engine.base_model import GrowthData, ProfileData, ShearData

# Shared "premium academic glass" plot palette.
_ACCENT = "#4C6EF5"          # muted indigo, primary curve color
_ACCENT_SOFT = "rgba(76, 110, 245, 0.15)"  # fill under curves
_GRID_COLOR = "rgba(45, 55, 72, 0.12)"
_TEXT_COLOR = "#2D3748"
_SEPARATION_RED = "#E03131"


def _transparent_layout(fig: go.Figure, x_title: str, y_title: str, x_log: bool = False) -> go.Figure:
    """Applies the shared transparent, glass-friendly layout to any figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_TEXT_COLOR, family="Inter, Segoe UI, sans-serif"),
        margin=dict(l=50, r=20, t=10, b=50),
        height=340,
        showlegend=False,
        hovermode="x unified",
    )
    fig.update_xaxes(
        title=x_title,
        type="log" if x_log else "linear",
        gridcolor=_GRID_COLOR,
        zerolinecolor=_GRID_COLOR,
        showline=True,
        linecolor=_GRID_COLOR,
    )
    fig.update_yaxes(
        title=y_title,
        gridcolor=_GRID_COLOR,
        zerolinecolor=_GRID_COLOR,
        showline=True,
        linecolor=_GRID_COLOR,
    )
    return fig


def plot_velocity_profile(profile: ProfileData) -> go.Figure:
    """
    Figure 1: The non-dimensional inner-region profile, u+ vs y+, on a
    semi-log x-axis -- the classical way turbulence researchers visualize
    the viscous sublayer / log-law / wake regions on a single plot, since
    y+ spans several orders of magnitude near the wall.
    """
    y_plus = np.asarray(profile["y_plus"])
    u_plus = np.asarray(profile["u_plus"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_plus,
            y=u_plus,
            mode="lines",
            line=dict(color=_ACCENT, width=3),
            fill="tozeroy",
            fillcolor=_ACCENT_SOFT,
            name="u+ (model)",
        )
    )

    # Reference lines every CFD engineer expects on this plot: the viscous
    # sublayer law u+ = y+ (valid for y+ < ~5) and the canonical log-law
    # u+ = (1/kappa) ln(y+) + B (valid for y+ > ~30), both shown as thin
    # dashed guides for visual calibration against the active model.
    y_visc = np.geomspace(max(y_plus.min(), 0.5), 8.0, 30)
    fig.add_trace(
        go.Scatter(
            x=y_visc, y=y_visc, mode="lines",
            line=dict(color="rgba(45,55,72,0.35)", width=1.5, dash="dot"),
            name="Viscous sublayer u+=y+",
        )
    )
    y_log = np.geomspace(20.0, max(y_plus.max(), 30.0), 60)
    kappa, B = 0.41, 5.0
    fig.add_trace(
        go.Scatter(
            x=y_log, y=(1 / kappa) * np.log(y_log) + B, mode="lines",
            line=dict(color="rgba(45,55,72,0.35)", width=1.5, dash="dash"),
            name="Log-law reference",
        )
    )

    return _transparent_layout(fig, "y⁺ (wall units, log scale)", "u⁺", x_log=True)


def plot_boundary_layer_growth(growth: GrowthData) -> go.Figure:
    """
    Figure 2: Physical boundary-layer thickness delta(x) growing along the
    streamwise direction -- rendered as a filled "wedge" silhouette (mirrored
    about the wall line at y=0) so the growth reads immediately as a
    physical boundary-layer shape rather than an abstract line chart.
    """
    x = np.asarray(growth["x"])
    delta = np.asarray(growth["delta"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=delta, mode="lines",
            line=dict(color=_ACCENT, width=3),
            fill="tozeroy",
            fillcolor=_ACCENT_SOFT,
            name="δ(x)",
        )
    )
    return _transparent_layout(fig, "Distance downstream, x (m)", "Boundary-layer thickness, δ (m)")


def plot_shear_stress(shear: ShearData) -> go.Figure:
    """
    Figure 3: Wall shear stress tau_w(x). If the model reports a separation
    location (tau_w crossing zero, or its own separation criterion firing),
    it is marked with a red dot and annotation reading "Flow Separation".
    """
    x = np.asarray(shear["x"])
    tau_w = np.asarray(shear["tau_w"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=tau_w, mode="lines",
            line=dict(color=_ACCENT, width=3),
            name="τ_w(x)",
        )
    )
    # Zero-shear reference line -- physically, tau_w = 0 IS the separation
    # condition for an attached boundary layer.
    fig.add_hline(y=0, line=dict(color="rgba(45,55,72,0.4)", width=1, dash="dot"))

    x_sep = shear.get("x_separation")
    if shear.get("separated") and x_sep is not None:
        # Interpolate tau_w at the flagged separation station for marker placement.
        y_sep = float(np.interp(x_sep, x, tau_w))
        fig.add_trace(
            go.Scatter(
                x=[x_sep], y=[y_sep], mode="markers+text",
                marker=dict(color=_SEPARATION_RED, size=13, symbol="circle",
                            line=dict(color="white", width=2)),
                text=["Flow Separation"],
                textposition="top center",
                textfont=dict(color=_SEPARATION_RED, size=12),
                name="Separation",
            )
        )

    return _transparent_layout(fig, "Distance downstream, x (m)", "Wall shear stress, τ_w (Pa)")
