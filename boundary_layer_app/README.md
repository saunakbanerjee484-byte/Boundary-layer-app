# Turbulent & Laminar Boundary-Layer Studio

A CPU-only Streamlit app (NumPy + SciPy, no neural networks, no GPU) for
interactively exploring 7 classical boundary-layer models: Blasius,
Prandtl's 1/7th power law, Spalding's law of the wall, Coles' law of the
wake, Thwaites' method, Head's entrainment method, and Stratford's
separation criterion.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Architecture

- `config.py` — shared physical constants and default/slider ranges.
- `physics_engine/base_model.py` — the `BoundaryLayerModel` abstract
  contract every model implements (`latex_formula`, `compute_profile`,
  `compute_growth`, `compute_shear`).
- `physics_engine/registry.py` — decorator-based registry
  (`@register_model("key")`) so `app.py` never imports a model class
  directly.
- `physics_engine/models/*.py` — the 7 physics models.
- `ui/theme.py` — the "whitish glassmorphism" CSS injection.
- `ui/visualizations.py` — the 3 transparent Plotly figures (u+ vs y+,
  δ(x) growth, τ_w(x) with separation marker).
- `app.py` — sidebar controls + wiring; adding model #8 requires only a
  new file in `physics_engine/models/` plus one import line in that
  package's `__init__.py`.
