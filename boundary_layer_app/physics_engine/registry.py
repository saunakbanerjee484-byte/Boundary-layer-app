"""
physics_engine/registry.py
===========================
A tiny decorator-based registry that decouples "which models exist" from
"how the UI lists/selects them". Each model module in `physics_engine/models`
calls `@register_model("some_key")` on its `BoundaryLayerModel` subclass;
`app.py` never imports a concrete model class directly -- it only ever asks
the registry for the currently selected key. This means a new model can be
added by dropping a new file into `models/` (and importing it once in
`models/__init__.py`) with **no changes to app.py or the UI layer**.
"""

from __future__ import annotations

from typing import Dict, Type

from physics_engine.base_model import BoundaryLayerModel

# Internal registry: maps a stable string key -> model class (not instance,
# so each lookup can construct a fresh, stateless instance).
_MODEL_REGISTRY: Dict[str, Type[BoundaryLayerModel]] = {}


def register_model(key: str):
    """
    Class decorator that registers a `BoundaryLayerModel` subclass under a
    stable string key.

    Usage:
        @register_model("zpg_blasius")
        class BlasiusModel(BoundaryLayerModel):
            ...
    """

    def _decorator(cls: Type[BoundaryLayerModel]) -> Type[BoundaryLayerModel]:
        if key in _MODEL_REGISTRY:
            raise ValueError(f"Duplicate model registry key: '{key}'")
        _MODEL_REGISTRY[key] = cls
        return cls

    return _decorator


def get_model(key: str) -> BoundaryLayerModel:
    """Instantiate and return the model registered under `key`."""
    try:
        model_cls = _MODEL_REGISTRY[key]
    except KeyError as exc:
        raise KeyError(
            f"No model registered under key '{key}'. "
            f"Available keys: {list(_MODEL_REGISTRY.keys())}"
        ) from exc
    return model_cls()


def list_model_keys() -> list[str]:
    """Return all registered model keys, in registration order."""
    return list(_MODEL_REGISTRY.keys())


def list_model_display_names() -> Dict[str, str]:
    """Return {key: display_name} for populating the sidebar selectbox."""
    return {key: cls().name for key, cls in _MODEL_REGISTRY.items()}
