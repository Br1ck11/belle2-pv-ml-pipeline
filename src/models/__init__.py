"""Explicit registry of trainable models, selected by the TOML [model].name.

Each model lives in its own folder, models/<name>/, containing:
    model.py      frozen hyperparameter dataclass, TrainableModel subclass
                  (network + compute_losses), build(), SPEC = ModelSpec(...)
    example.toml  a complete training configuration for this model

To add a model: create the folder, then add its SPEC to the tuple below.
The training machinery in src/train needs no changes.
"""

from .base import Batch, ForecastingModel, ModelSpec, TrainableModel
from .cnn_vae import SPEC as CNN_VAE

MODELS: dict[str, ModelSpec] = {spec.name: spec for spec in (CNN_VAE,)}


def get_model_spec(name: str) -> ModelSpec:
    try:
        return MODELS[name]
    except KeyError:
        raise ValueError(f"Unknown model {name!r}; registered models: {sorted(MODELS)}") from None


__all__ = ["Batch", "ForecastingModel", "ModelSpec", "TrainableModel", "MODELS", "get_model_spec"]
