"""
app.py
======
Main Streamlit entry point for the Boundary-Layer Studio.

This file is intentionally "dumb" about physics: it never imports a
concrete model class, computes a formula, or knows what lambda/Pi/H mean.
Its only jobs are:

  1. Render the sidebar controls and collect them into a `FlowConditions`.
  2. Ask the registry for whichever model key is currently selected.
  3. Ask that model for its formula + profile/growth/shear data.
  4. Hand that data to `ui/visualizations.py` to draw the three Plotly cards.

That separation is exactly what the decorator-based registry pattern buys
us: dropping model #8 into `physics_engine/models/` requires zero edits to
this file.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

import config
from physics_engine.base_model import FlowConditions
from physics_engine.registry import get_model, list_model_display_names

# Importing this package runs every model's @register_model(...) decorator,
# populating the registry. Must happen before list_model_display_names()/
# get_model() are called anywhere below.
import physics_engine.models  # noqa: F401  (import-for-side-effects)

from ui.theme import inject_glassmorphism_theme, glass_card_open, glass_card_close
from ui.visualizations import (
    plot_velocity_profile,
    plot_boundary_layer_growth,
    plot_shear_stress,
)

# ---------------------------------------------------------------------------
# Page configuration & theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_glassmorphism_theme()


# ---------------------------------------------------------------------------
# Sidebar: fluid properties, free-stream conditions, and model selection
# ---------------------------------------------------------------------------
def render_sidebar() -> tuple[FlowConditions, str]:
    """
    Renders every sidebar control and packages the results into an
    immutable `FlowConditions` object plus the selected model's registry
    key. Kept as a single function so `main()` reads as a straight-line
    narrative of the app's data flow.
    """
    with st.sidebar:
        st.markdown("## ⚙️ Flow Configuration")

        display_names = list_model_display_names()
        # Sort by category prefix embedded in the name string isn't
        # available here, so we just present in registration order, which
        # already mirrors the physically-logical grouping (ZPG laminar ->
        # ZPG turbulent -> inner-region -> APG turbulent/laminar -> limits).
        model_key = st.selectbox(
            "Boundary-Layer Model",
            options=list(display_names.keys()),
            format_func=lambda k: display_names[k],
            help="Choose which theoretical/empirical model drives the plots below.",
        )

        st.markdown("### Fluid Properties")
        nu = st.number_input(
            "Kinematic viscosity, ν (m²/s)",
            min_value=config.NU_MIN,
            max_value=config.NU_MAX,
            value=config.DEFAULT_NU,
            format="%.2e",
            help="Momentum diffusivity of the fluid. Smaller ν (e.g. air) -> "
                 "thinner boundary layers than larger ν (e.g. oils).",
        )
        rho = st.number_input(
            "Density, ρ (kg/m³)",
            min_value=config.RHO_MIN,
            max_value=config.RHO_MAX,
            value=config.DEFAULT_RHO,
            help="Fluid density. Enters only through dynamic-pressure-based "
                 "quantities (e.g. wall shear); the boundary-layer shape "
                 "itself depends on ν, U and dp/dx (incompressible flow).",
        )

        st.markdown("### Free-Stream Conditions")
        U = st.number_input(
            "Free-stream velocity, U (m/s)",
            min_value=config.U_MIN,
            max_value=config.U_MAX,
            value=config.DEFAULT_U,
            help="Velocity of the undisturbed outer flow feeding the "
                 "boundary layer at the leading edge.",
        )
        dpdx = st.number_input(
            "Pressure gradient, dp/dx (normalized)",
            min_value=-1.0,
            max_value=1.0,
            value=config.DEFAULT_DPDX,
            step=0.05,
            help="0 = Zero Pressure Gradient (flat plate). Positive = "
                 "Adverse Pressure Gradient (decelerating outer flow, "
                 "pushes toward separation). Negative = Favorable Pressure "
                 "Gradient (accelerating, stabilizing).",
        )
        x_max = st.number_input(
            "Plate / body length, x_max (m)",
            min_value=config.X_MAX_MIN,
            max_value=config.X_MAX_MAX,
            value=max(config.DEFAULT_X_MAX, 1.0),
            help="Streamwise extent over which growth and shear are plotted.",
        )

        st.markdown("---")
        st.caption(
            "All computation is CPU-bound NumPy/SciPy — no neural networks, "
            "no GPU acceleration. Runs instantly on low-tier hardware."
        )

    conditions = FlowConditions(nu=nu, rho=rho, U=U, dpdx=dpdx, x_max=x_max)
    return conditions, model_key


# ---------------------------------------------------------------------------
# Main canvas
# ---------------------------------------------------------------------------
def render_main(conditions: FlowConditions, model_key: str) -> None:
    """Renders the header, governing-equation panel, and the three plot cards."""
    model = get_model(model_key)

    st.markdown(
        f"""
        <div class="hero-card">
            <h2 style="margin-bottom:0.2rem;">{config.APP_ICON} {config.APP_TITLE}</h2>
            <p style="margin-bottom:0.1rem; opacity:0.75;">
                <strong>{model.name}</strong> &nbsp;·&nbsp; {model.category}
            </p>
            <p style="margin-bottom:0; opacity:0.65; font-size:0.92rem;">
                {model.description}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Governing equation, always visible for whichever model is active --
    # this is the "Educational UX" requirement: the math on screen should
    # always match the physics currently driving the plots below it.
    glass_card_open("Governing Equation")
    st.latex(model.latex_formula())
    glass_card_close()

    # Compute all three payloads once, up front, so a solver hiccup in one
    # plot doesn't silently leave the others stale.
    try:
        profile = model.compute_profile(conditions)
        growth = model.compute_growth(conditions)
        shear = model.compute_shear(conditions)
    except Exception as exc:  # noqa: BLE001 - surface solver issues to the user, not a crash
        st.error(
            f"The solver could not converge for this parameter combination "
            f"({exc}). Try easing the pressure-gradient slider toward 0."
        )
        return

    if shear.get("separated"):
        x_sep = shear.get("x_separation")
        st.markdown(
            f'<div class="separation-alert">⚠️ Flow separation predicted at '
            f'x ≈ {x_sep:.3f} m under these conditions.</div>',
            unsafe_allow_html=True,
        )
        st.write("")

    col1, col2, col3 = st.columns(3)

    with col1:
        glass_card_open("Non-Dimensional Profile (u⁺ vs y⁺)")
        st.plotly_chart(plot_velocity_profile(profile), use_container_width=True)
        glass_card_close()

    with col2:
        glass_card_open("Boundary-Layer Growth δ(x)")
        st.plotly_chart(plot_boundary_layer_growth(growth), use_container_width=True)
        glass_card_close()

    with col3:
        glass_card_open("Wall Shear Stress & Separation")
        st.plotly_chart(plot_shear_stress(shear), use_container_width=True)
        glass_card_close()


def main() -> None:
    conditions, model_key = render_sidebar()
    render_main(conditions, model_key)


if __name__ == "__main__":
    main()
