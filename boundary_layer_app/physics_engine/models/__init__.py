"""
physics_engine/models package
==============================
Importing this package has the side effect of importing every individual
model module below, which runs their `@register_model(...)` decorators and
populates `physics_engine.registry`. `app.py` only needs to do
`import physics_engine.models` once at startup; it never needs to know the
individual model class names.

To add model #8 in the future: create `models/my_new_model.py`, subclass
`BoundaryLayerModel`, decorate it with `@register_model("my_key")`, and add
one import line below. Nothing else in the codebase changes.
"""

from physics_engine.models import zpg_blasius       # noqa: F401
from physics_engine.models import zpg_prandtl        # noqa: F401
from physics_engine.models import inner_spalding      # noqa: F401
from physics_engine.models import apg_coles_wake       # noqa: F401
from physics_engine.models import laminar_thwaites      # noqa: F401
from physics_engine.models import turbulent_head          # noqa: F401
from physics_engine.models import apg_stratford             # noqa: F401
