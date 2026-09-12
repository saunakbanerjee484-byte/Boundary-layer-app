"""
ui/theme.py
===========
Injects the "Whitish Glassmorphism" design language into the Streamlit app
via a single raw CSS block. Kept isolated from app.py so the visual design
can be iterated on independently of the application logic.
"""

import streamlit as st

# Off-white canvas background for the whole app.
BACKGROUND_COLOR = "#F8F9FA"

# Dark slate-grey text color used throughout for strong readability against
# the light glass panels.
TEXT_COLOR = "#2D3748"

# A soft accent color (muted indigo/blue) used for headers, active widgets,
# and the separation markers so they read as "premium academic" rather than
# a loud primary blue.
ACCENT_COLOR = "#4C6EF5"


def inject_glassmorphism_theme() -> None:
    """
    Injects global CSS overrides to achieve a frosted-glass, off-white,
    academic-premium aesthetic across the sidebar, main canvas, metric
    cards, and widget containers.

    Called once, near the top of app.py, before any other Streamlit
    widgets are rendered.
    """
    st.markdown(
        f"""
        <style>
        /* ---------------------------------------------------------------
           Global canvas
        --------------------------------------------------------------- */
        .stApp {{
            background-color: {BACKGROUND_COLOR};
            color: {TEXT_COLOR};
        }}

        html, body, [class*="css"] {{
            font-family: 'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont,
                         sans-serif;
            color: {TEXT_COLOR};
        }}

        h1, h2, h3, h4, h5, h6 {{
            color: {TEXT_COLOR};
            font-weight: 700;
            letter-spacing: -0.02em;
        }}

        /* ---------------------------------------------------------------
           Sidebar: subtle glass panel over the off-white canvas
        --------------------------------------------------------------- */
        section[data-testid="stSidebar"] {{
            background: rgba(255, 255, 255, 0.55);
            backdrop-filter: blur(12px) saturate(150%);
            -webkit-backdrop-filter: blur(12px) saturate(150%);
            border-right: 1px solid rgba(255, 255, 255, 0.8);
        }}

        /* ---------------------------------------------------------------
           Generic "glass card" utility class, used to wrap plot groups
           and the formula panel via st.markdown(...unsafe_allow_html=True)
        --------------------------------------------------------------- */
        .glass-card {{
            background: rgba(255, 255, 255, 0.6);
            backdrop-filter: blur(12px) saturate(150%);
            -webkit-backdrop-filter: blur(12px) saturate(150%);
            border: 1px solid rgba(255, 255, 255, 0.8);
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(31, 41, 55, 0.08);
            padding: 1.4rem 1.6rem;
            margin-bottom: 1.2rem;
        }}

        .glass-card h4 {{
            margin-top: 0;
            color: {ACCENT_COLOR};
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}

        /* Header banner card */
        .hero-card {{
            background: rgba(255, 255, 255, 0.65);
            backdrop-filter: blur(14px) saturate(160%);
            -webkit-backdrop-filter: blur(14px) saturate(160%);
            border: 1px solid rgba(255, 255, 255, 0.85);
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(31, 41, 55, 0.08);
            padding: 1.6rem 2rem;
            margin-bottom: 1.4rem;
        }}

        /* ---------------------------------------------------------------
           Streamlit native widgets restyled to match the glass aesthetic
        --------------------------------------------------------------- */
        div[data-baseweb="select"] > div, .stTextInput > div > div,
        .stNumberInput > div > div {{
            background: rgba(255, 255, 255, 0.7) !important;
            border-radius: 10px !important;
            border: 1px solid rgba(255, 255, 255, 0.9) !important;
        }}

        .stSlider > div > div > div > div {{
            background: {ACCENT_COLOR} !important;
        }}

        div[data-testid="stMetric"] {{
            background: rgba(255, 255, 255, 0.6);
            backdrop-filter: blur(12px) saturate(150%);
            border: 1px solid rgba(255, 255, 255, 0.8);
            border-radius: 14px;
            padding: 0.8rem 1rem;
        }}

        /* Separation alert banner */
        .separation-alert {{
            background: rgba(255, 235, 235, 0.75);
            border: 1px solid rgba(220, 38, 38, 0.35);
            color: #9B1C1C;
            border-radius: 14px;
            padding: 0.9rem 1.2rem;
            font-weight: 600;
        }}

        /* Hide default Streamlit chrome for a cleaner "app" feel */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{background: transparent;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def glass_card_open(title: str) -> None:
    """Opens a `.glass-card` div with an optional uppercase title."""
    st.markdown(
        f'<div class="glass-card"><h4>{title}</h4>',
        unsafe_allow_html=True,
    )


def glass_card_close() -> None:
    """Closes a `.glass-card` div opened with `glass_card_open`."""
    st.markdown("</div>", unsafe_allow_html=True)
