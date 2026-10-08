# Turbulent & Laminar Boundary-Layer Studio

A CPU-only, GPU-free, neural-network-free interactive computational fluid
dynamics (CFD) sandbox for exploring seven classical laminar and turbulent
boundary-layer models behind a Streamlit front end. Built on NumPy and
SciPy deterministic solvers (`solve_bvp`, `solve_ivp`, `newton`, `quad`,
`root_scalar`) — no PINNs, no CUDA, no stochastic approximation.

---

## Problem

Predicting how a boundary layer grows, how its velocity profile is shaped,
and — critically — where and whether it separates from a surface is a
central problem in aerodynamics and internal-flow design (wings, diffusers,
turbine blades, nozzles). Full Navier–Stokes CFD is accurate but
computationally heavy and poorly suited to an interactive, real-time
teaching/design tool. This project instead implements seven classical
*reduced-order* boundary-layer theories — similarity solutions, empirical
inner/outer-region closures, and integral momentum methods — each valid
over a specific regime (laminar/turbulent, zero/adverse pressure gradient),
and lets a user compare their predictions of velocity profile, growth, wall
shear, and separation onset in real time on ordinary CPU hardware.

---

## Mathematical Formulation

| # | Model | Regime | Governing Relation |
|---|-------|--------|---------------------|
| 1 | Blasius Exact Solution | Laminar, ZPG | `2f''' + f f'' = 0`, `f(0)=0, f'(0)=0, f'(∞)=1` |
| 2 | Prandtl's 1/7th Power Law | Turbulent, ZPG | `ū/U = (y/δ)^(1/7)` |
| 3 | Spalding's Single Formula | Turbulent, inner region | `y⁺ = u⁺ + e^{-κB}[e^{κu⁺} - 1 - κu⁺ - (κu⁺)²/2 - (κu⁺)³/6]` |
| 4 | Coles' Law of the Wake | Turbulent, APG outer region | `u⁺ = (1/κ)ln(y⁺) + B + (2Π/κ)sin²(πy/2δ)`, `Π ≈ 0.8(β+0.5)^{3/4}`, `β = (δ*/τ_w)(dp/dx)` |
| 5 | Thwaites' Method | Laminar separation predictor | `λ = (θ²/ν)(dU/dx)`; separation at `λ = -0.09` |
| 6 | Head's Entrainment Method | Turbulent APG & separation | von Kármán momentum integral + entrainment closure on `H = δ*/θ`; separation flagged at `H ≈ 2.4` |
| 7 | Stratford's Separation Criterion | Turbulent, design-limit case | `C_p(x·dC_p/dx)^{1/2}(10⁻⁶Re_x)^{-1/10} = S ≈ 0.35` |

Universal constants (von Kármán constant `κ = 0.41`, log-law intercept
`B = 5.0`) are centralized in `config.py`.

---

## Numerical Methods

- **Blasius (Model 1):** third-order similarity ODE reduced to a first-order
  system and solved as a two-point boundary-value problem with
  `scipy.integrate.solve_bvp`.
- **Prandtl 1/7th (Model 2):** direct vectorized NumPy algebra, no solver.
- **Spalding (Model 3):** implicit in `u⁺`; inverted per-point with
  `scipy.optimize.newton`.
- **Coles (Model 4):** the wake-strength parameter `Π` is *derived*, not
  assumed — it is obtained from the Clauser pressure-gradient parameter
  `β = (δ*/τ_w)(dp/dx)` via the standard equilibrium-layer correlation
  `Π ≈ 0.8(β+0.5)^{3/4}`. Growth, momentum thickness `θ`, shape factor `H`,
  and wall shear `τ_w` are obtained by *reusing* Head's rigorous von
  Kármán momentum-integral + entrainment march (Model 6) rather than an
  independent, ad hoc growth law — Coles' distinct theoretical
  contribution is the outer-region *profile shape*, not a separate growth
  model.
- **Thwaites (Model 5):** the momentum-thickness integral is evaluated with
  `scipy.integrate.quad`.
- **Head (Model 6):** the coupled `[θ, H₁·θ]` ODE system is marched
  downstream with `scipy.integrate.solve_ivp` (RK45), with a terminal-free
  event watching for the `H = 2.4` separation threshold.
- **Stratford (Model 7):** at each downstream station, the critical
  pressure-recovery slope `dC_p/dx` is solved with
  `scipy.optimize.root_scalar` (Brent's method). The very first station
  (`C_p = 0` by definition — no recovery has yet occurred) is handled via
  an exact closed-form integral of the regularized substitution
  `w = C_p³` (under which the governing relation is no longer singular at
  `C_p = 0`), rather than an arbitrary numerical floor on `C_p`, and the
  march uses a log-spaced streamwise grid so step sizes stay proportionally
  small near the start of the recovery region.

---

## Validation

- **`tests/test_blasius.py`** — solves the Blasius BVP with the exact
  solver the app uses and asserts the dimensionless wall-shear parameter
  matches the accepted literature benchmark `f''(0) ≈ 0.33206` (Howarth's
  refined tabulation of Blasius' 1908 solution) to within `1e-4`, plus
  boundary-condition and monotonicity sanity checks.
- **`tests/validation_utils.py`** — a `relative_error_percent(q_model,
  q_reference)` utility implementing
  `ε = |q_model − q_reference| / |q_reference| × 100%`, plus a loader for
  the `validation/*.csv` reference files.
- **`validation/thwaites_reference.csv`** and **`validation/head_reference.csv`**
  — stub CSVs with a documented schema and suggested literature sources
  (Falkner–Skan exact solutions for Thwaites; the 1968 AFOSR-IFP-Stanford
  "Stanford Olympics" turbulent boundary-layer dataset for Head). Populate
  either file and the corresponding `test_*_against_reference_data` test in
  `tests/test_validation_utils.py` automatically activates (currently
  `SKIPPED` against the empty stubs, asserting ≤5% relative error once
  populated).

Run the suite with:

```bash
pip install -r requirements-dev.txt
pytest
```

---

## Model Limitations

- **Pressure-gradient closures are simplified U(x)/dp/dx models, not a
  coupled outer-flow solve.** Every APG/FPG model (Thwaites, Head, Coles,
  Stratford) maps the UI's single dimensionless `dp/dx` slider onto a
  *prescribed* linear edge-velocity distribution `U(x)`; there is no
  two-way coupling with an actual body geometry or outer potential-flow
  solution. The physical pressure gradient used by Coles'
  `β = (δ*/τ_w)(dp/dx)` is derived from this same prescribed `U(x)` via
  Bernoulli (`dp/dx = −ρU·dU/dx`), so it is only as physically meaningful
  as that linear `U(x)` assumption.
- **Coles' wake parameter Π is a simplification at ZPG.** The equilibrium
  correlation `Π ≈ 0.8(β+0.5)^{3/4}` returns `Π = 0` exactly at `β = 0`
  (flat plate); real ZPG layers carry a small residual wake (`Π ≈ 0.45–0.6`)
  from outer-layer intermittency that this model does not reproduce.
- **Separation criteria are empirical thresholds, not universal
  constants.** Thwaites' `λ = -0.09`, Head's `H ≈ 2.4`, and Stratford's
  `S ≈ 0.35` are each fitted to specific historical datasets; real
  separation onset for a given geometry can deviate from these thresholds,
  and all three lose validity as the boundary-layer approximation itself
  breaks down near/after separation.
- **Head's post-separation `τ_w = 0` is a visualization convention, not a
  physical claim.** Once `H` crosses 2.4, the entrainment closure's
  empirical correlations (fitted to attached-flow data) are no longer
  trustworthy, so the march is not continued; wall shear is floored to
  zero downstream purely so the plot communicates "the model stops here,"
  not that real post-stall wall shear is uniformly zero (in reality it is
  small, sign-indeterminate, and unsteady in the recirculating region).
- **Stratford is an inverse design-limit criterion, not a forward
  simulation.** It computes the steepest pressure-recovery curve a
  turbulent layer can sustain while remaining everywhere on the brink of
  separation (`τ_w ≈ 0`); it does not march a prescribed, independently
  specified pressure distribution the way Thwaites/Head do.
- **No transition model.** None of the seven models predicts
  laminar-to-turbulent transition; the user (or the model selection
  itself) determines which regime applies.
- **2D, incompressible, steady-state only.** No 3D cross-flow, no
  compressibility, no unsteadiness.

---

## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/saunakbanerjee484-byte/Boundary-layer-app.git
cd Boundary-layer-app
pip install -r requirements.txt
streamlit run app.py
```

For running the test suite:

```bash
pip install -r requirements-dev.txt
pytest
```

---

## References

- Blasius, H. (1908). *Grenzschichten in Flüssigkeiten mit kleiner Reibung*.
- Howarth, L. (1938). *On the Solution of the Laminar Boundary Layer
  Equations*. Proc. Roy. Soc. A.
- Prandtl, L. — the empirical 1/7th-power turbulent velocity profile, as
  presented in standard boundary-layer texts (e.g. Schlichting & Gersten,
  *Boundary-Layer Theory*).
- Spalding, D. B. (1961). *A Single Formula for the "Law of the Wall"*.
  J. Appl. Mech.
- Coles, D. (1956). *The Law of the Wake in the Turbulent Boundary Layer*.
  J. Fluid Mech.
- Clauser, F. H. (1954). *Turbulent Boundary Layers in Adverse Pressure
  Gradients*. J. Aeronaut. Sci. — source of the equilibrium parameter `β`.
- White, F. M. *Viscous Fluid Flow* — equilibrium-layer `Π(β)` correlation.
- Thwaites, B. (1949). *Approximate Calculation of the Laminar Boundary
  Layer*. Aeronaut. Quart.
- Head, M. R. (1958). *Entrainment in the Turbulent Boundary Layer*. ARC
  R&M 3152.
- Ludwieg, H. & Tillmann, W. (1950). *Investigations of the Wall-Shearing
  Stress in Turbulent Boundary Layers*. NACA TM 1285.
- Stratford, B. S. (1959). *The Prediction of Separation of the Turbulent
  Boundary Layer*. J. Fluid Mech.
- Coles, D. E. & Hirst, E. A. (eds., 1968). *Computation of Turbulent
  Boundary Layers — 1968 AFOSR-IFP-Stanford Conference*. (Source of the
  "Stanford Olympics" reference cases suggested for `head_reference.csv`.)

---

## Architecture

A decorator-based registry pattern decouples the physics engine from the
UI: `app.py` never imports a concrete model class, only
`physics_engine.registry.get_model(key)`. Adding model #8 requires only a
new file in `physics_engine/models/` implementing `BoundaryLayerModel`,
decorated with `@register_model("key")`, plus one import line in
`physics_engine/models/__init__.py` — no changes to `app.py` or
`ui/visualizations.py`.

```text
.
├── app.py
├── config.py
├── conftest.py
├── requirements.txt
├── requirements-dev.txt
├── ui/
│   ├── theme.py
│   └── visualizations.py
├── physics_engine/
│   ├── base_model.py
│   ├── registry.py
│   └── models/
│       ├── zpg_blasius.py
│       ├── zpg_prandtl.py
│       ├── inner_spalding.py
│       ├── apg_coles_wake.py
│       ├── laminar_thwaites.py
│       ├── turbulent_head.py
│       └── apg_stratford.py
├── tests/
│   ├── test_blasius.py
│   ├── test_validation_utils.py
│   └── validation_utils.py
└── validation/
    ├── thwaites_reference.csv
    └── head_reference.csv
```
TO DELVED DEEPER PLS FOLLOW DEVOLOPERS NOTES 