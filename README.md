# 🌊 Turbulent & Laminar Boundary-Layer Studio

> A high-performance, CPU-only interactive computational fluid dynamics (CFD) sandbox built for engineers, researchers, and students. Explore, simulate, and visualize classical boundary-layer theory and separation criteria in real-time.

---

## 🚀 Quick Start

Ensure you have Python 3.9+ installed, then run the following commands in your terminal:

```bash
git clone https://github.com/your-username/boundary-layer-app.git
cd boundary-layer-app
pip install -r requirements.txt
streamlit run app.py
```

---

## 🏛️ System Architecture & Modular Design

The application is engineered with a strict separation of concerns between the computational physics backend and the Streamlit frontend. It utilizes a **Decorator-Based Registry Pattern**, meaning you can drop a new model into the system without ever touching the UI or routing code.

```text
boundary_layer_app/
├── app.py                     # Main Streamlit entry point (UI logic & wiring)
├── config.py                  # Global physical constants and slider ranges
├── ui/
│   ├── __init__.py
│   ├── theme.py               # "Whitish Glassmorphism" custom CSS injection
│   └── visualizations.py      # Transparent Plotly interactive graph generators
└── physics_engine/
    ├── __init__.py
    ├── registry.py            # Central model registry (@register_model)
    ├── base_model.py          # Abstract Base Class enforcing the simulation contract
    └── models/                # Plug-and-play directory for physics models
        ├── __init__.py        # Auto-imports models to trigger decorators
        ├── zpg_blasius.py     # Model 1: Blasius Exact Solution
        ├── zpg_prandtl.py     # Model 2: Prandtl's 1/7th Power Law
        ├── inner_spalding.py  # Model 3: Spalding's Single Formula
        ├── apg_coles_wake.py  # Model 4: Coles' Law of the Wake
        ├── laminar_thwaites.py# Model 5: Thwaites' Method
        ├── turbulent_head.py  # Model 6: Head's Entrainment Method
        └── apg_stratford.py   # Model 7: Stratford's Separation Criterion
```

---

## 📐 The Mathematical Engine & 7 Core Models

Every model is computed strictly on the CPU using optimized NumPy vectorization and SciPy numerical solvers (`solve_ivp`, `solve_bvp`, `newton`, `quad`, `root_scalar`). No GPUs or neural networks are used.

### 1. Blasius Exact Solution (Laminar Baseline)

The foundational laminar boundary-layer solution over a flat plate, derived by reducing the Navier-Stokes equations via similarity variables.

- **Governing Equation:** $2f''' + f f'' = 0$
- **Numerical Method:** Solved via a shooting method or `scipy.integrate.solve_bvp`.
- **Physical Insight:** Establishes the clean laminar baseline before momentum-mixing turbulent eddies alter the velocity distribution.

### 2. Prandtl's 1/7th Power Law (ZPG Empirical)

A lightweight algebraic approximation for turbulent flow under a Zero Pressure Gradient (ZPG).

- **Governing Equation:** $\dfrac{\bar{u}}{U} = \left(\dfrac{y}{\delta}\right)^{1/7}$
- **Numerical Method:** Direct vectorized NumPy evaluations.
- **Physical Insight:** Generates a blunter profile near the wall compared to laminar flow, reflecting increased momentum transfer.

### 3. Spalding's Single Formula (Seamless Inner Region)

Standard logarithmic profiles fail in the buffer layer ($5 < y^+ < 30$). Spalding formulated a single implicit equation that bridges the viscous sublayer, buffer layer, and log-law region seamlessly.

- **Governing Equation:** $y^+ = u^+ + e^{-\kappa B} \left[ e^{\kappa u^+} - 1 - \kappa u^+ - \dfrac{(\kappa u^+)^2}{2} - \dfrac{(\kappa u^+)^3}{6} \right]$
- **Numerical Method:** Inverted via `scipy.optimize.newton` to solve for $u^+$ given $y^+$.
- **Physical Insight:** Eliminates piecewise calculation errors and smooths out the transition zone near the solid boundary.

### 4. Coles' Law of the Wake (APG Outer Region Deformation)

When an Adverse Pressure Gradient ($\frac{dp}{dx} > 0$) opposes the flow, the outer region departs from the standard log-law. Coles added a wake parameter ($\Pi$) to capture this phenomenon.

- **Governing Equation:** $u^+ = \dfrac{1}{\kappa} \ln(y^+) + B + \dfrac{2\Pi}{\kappa} \sin^2\left(\dfrac{\pi y}{2\delta}\right)$
- **Numerical Method:** Algebraic evaluation where the user's pressure-gradient slider directly controls $\Pi$.
- **Physical Insight:** Visualizes how adverse pressure forces the upper half of the velocity profile to bulge outward and decelerate.

### 5. Thwaites' Method (Laminar Separation Predictor)

A robust integral method to track momentum thickness and predict laminar boundary-layer separation under arbitrary pressure gradients.

- **Governing Equation:** Uses the dimensionless momentum parameter $\lambda = \dfrac{\theta^2}{\nu} \dfrac{dU}{dx}$. Separation occurs precisely at $\lambda = -0.09$.
- **Numerical Method:** Evaluated using `scipy.integrate.quad`.
- **Physical Insight:** Identifies the precise coordinate where laminar fluid detaches from a surface.

### 6. Head's Entrainment Method (Turbulent APG & Separation)

The industry-standard integral approach for turbulent boundary layers, tracking how free-stream fluid is "entrained" into the turbulent layer.

- **Governing Equation:** Solves coupled ODEs of the von Kármán momentum integral and entrainment equations based on Shape Factor $H = \delta^* / \theta$.
- **Numerical Method:** `scipy.integrate.solve_ivp` marched along the stream coordinate $x$.
- **Physical Insight:** Triggers an automated UI alert when $H$ exceeds critical thresholds ($H \approx 2.4 - 3.0$), signaling turbulent detachment.

### 7. Stratford's Separation Criterion (The Limit Case)

An analytical limit-case criterion designed to calculate the maximum permissible adverse pressure gradient a system can sustain without inducing separation ($\tau_w = 0$).

- **Governing Equation:** $C_p \left( x \dfrac{dC_p}{dx} \right)^{1/2} (10^{-6} Re_x)^{-1/10} = \text{Constant}$
- **Numerical Method:** Root-finding via `scipy.optimize.root_scalar`.
- **Physical Insight:** Serves as a "Design Mode" for engineering diffusers and fluid channels safely beneath failure thresholds.

---

## 🎨 UI/UX Design Language

The app features a custom **"Whitish Glassmorphism"** theme injected via CSS:

- **Canvas:** Clean off-white background (`#F8F9FA`).
- **Glass Containers:** Semi-transparent frosted glass cards (`rgba(255, 255, 255, 0.6)`) with `backdrop-filter: blur(12px)` and subtle shadow layering.
- **Dynamic Equations:** Renders live LaTeX formulas (`st.latex()`) corresponding to the active model directly on the dashboard.
- **Interactive Visuals:** Zero-latency Plotly charts featuring semi-log velocity profiles, spatial growth curves ($\delta(x)$), and shear-stress profiles ($\tau_w(x)$) with live separation markers.

---

## 🔌 Extending the Engine (Adding Model #8)

Thanks to the registry pattern, adding a new model takes less than 3 minutes:

1. Create a new file under `physics_engine/models/my_new_model.py`.
2. Inherit from `BoundaryLayerModel` and implement the required methods.
3. Decorate the class with `@register_model("my_new_model")`.
4. Import your new module in `physics_engine/models/__init__.py`.

The UI dropdown updates automatically.
