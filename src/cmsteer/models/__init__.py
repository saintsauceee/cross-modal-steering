"""Model registry. Each model's dependencies are imported only when it is loaded, so the package itself (analysis,
judging) runs without any of them."""

from __future__ import annotations

import importlib

from cmsteer.models.base import SteerableModel

MODELS = {
    "uniar": "cmsteer.models.uniar:UniAR",
    "emu35": "cmsteer.models.emu35:Emu35",
    "liquid": "cmsteer.models.liquid:Liquid",
}


def register(name: str, target: str) -> None:
    """Register a model as "<module>:<class>"; the class must subclass SteerableModel."""
    MODELS[name] = target


def load_model(name: str, device: str = "cuda", **kwargs) -> SteerableModel:
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(MODELS)}")
    module, cls = MODELS[name].split(":")
    return getattr(importlib.import_module(module), cls)(device=device, **kwargs)


__all__ = ["MODELS", "SteerableModel", "load_model", "register"]
