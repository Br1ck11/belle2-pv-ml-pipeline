"""The contract between the shared training engine and individual models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import nn


# One DataLoader batch. "values" is float32 (B, C, T); "index" is int64 (B,)
# and links each window back to its row in the HDF5 split. Future objectives
# (e.g. supervised labels) add keys here without changing the engine.
Batch = dict[str, torch.Tensor]


class TrainableModel(nn.Module, ABC):
    """An nn.Module that defines its own training objective.

    The engine never computes a loss itself. It calls ``compute_losses`` in
    train mode (with gradients) and in eval mode (under no_grad), minimises
    ``result["loss"]`` and logs every entry. Each entry must be a finite
    scalar tensor averaged over the windows of the batch, so the engine can
    form exact epoch means by weighting with the batch size.
    """

    # True for models with a stochastic latent (e.g. VAEs). Testing then needs
    # an explicit latent_mode: "z_mean" (deterministic) or "sampling_z".
    is_variational = False

    @abstractmethod
    def compute_losses(self, batch: Batch) -> dict[str, torch.Tensor]:
        """Return {"loss": scalar to minimise, ...other scalar diagnostics}."""

    def evaluation_output(self, batch: Batch, *, latent_mode: str | None = None) -> dict[str, torch.Tensor]:
        """Model output used for testing, independent of the model type.

        Returns {"output": (B, C, T'), "target": (B, C, T')} and optionally
        "time_index": int64 (T',) giving the window sample of each output
        step (default: 0 .. T-1). The test score compares output and target.

        The default suits models whose forward pass returns a reconstruction
        with the input's shape (autoencoders). Forecasting models (e.g. LSTMs)
        override it to return their predictions, the matching target samples
        and their time_index. Variational models override it and honour
        latent_mode ("z_mean" or "sampling_z").
        """
        if latent_mode is not None:
            raise ValueError(f"{type(self).__name__} is not variational; latent_mode must not be set")
        values = batch["values"]
        output = self(values)
        if not isinstance(output, torch.Tensor) or output.shape != values.shape:
            raise NotImplementedError(
                f"{type(self).__name__}.forward does not return a reconstruction of shape {tuple(values.shape)}; "
                "override evaluation_output to define the model's test output")
        return {"output": output, "target": values}


class ForecastingModel(TrainableModel):
    """Base class for models that PREDICT samples instead of reconstructing them.

    Each window of T samples is split into a context (the first T - H
    samples, which the model sees) and a horizon (the last H samples, which
    it predicts). A subclass calls ``super().__init__(horizon=H, ...)`` and
    implements ``forecast(context) -> prediction``:

        context     (B, C, T - H)   values[:, :, :T - H]
        prediction  (B, C, H)       compared with values[:, :, T - H:]

    The base class provides
      * ``compute_losses``: MSE between prediction and the true horizon
        samples (override for another objective; keep "loss");
      * ``evaluation_output``: prediction, target and ``time_index`` =
        T - H ... T - 1, so the test score is the prediction error on the
        horizon and the plots place it at the right sample times.

    The window size T comes from dataset.h5; H is the model's own
    hyperparameter and must satisfy 1 <= H < T.
    """

    def __init__(self, *, horizon: int, n_channels: int, window_size: int) -> None:
        super().__init__()
        if n_channels <= 0 or window_size <= 0:
            raise ValueError("n_channels and window_size must be positive")
        if not 1 <= horizon < window_size:
            raise ValueError(f"horizon must satisfy 1 <= horizon < window_size = {window_size}, got {horizon}")
        self.horizon = horizon
        self.n_channels = n_channels
        self.window_size = window_size

    @property
    def context_size(self) -> int:
        return self.window_size - self.horizon

    @abstractmethod
    def forecast(self, context: torch.Tensor) -> torch.Tensor:
        """(B, C, T - H) context -> (B, C, H) prediction of the next H samples."""

    def split(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, C, T) window -> (context (B, C, T - H), true horizon (B, C, H))."""
        if values.ndim != 3 or tuple(values.shape[1:]) != (self.n_channels, self.window_size):
            raise ValueError(f"{type(self).__name__} was built for (C, T) = ({self.n_channels}, "
                             f"{self.window_size}), got windows of shape {tuple(values.shape[1:])}")
        return values[:, :, :self.context_size], values[:, :, self.context_size:]

    def _checked_forecast(self, context: torch.Tensor) -> torch.Tensor:
        prediction = self.forecast(context)
        expected = (context.shape[0], self.n_channels, self.horizon)
        if tuple(prediction.shape) != expected:
            raise ValueError(f"{type(self).__name__}.forecast returned {tuple(prediction.shape)}, expected {expected}")
        return prediction

    def compute_losses(self, batch: Batch) -> dict[str, torch.Tensor]:
        context, target = self.split(batch["values"])
        mse = (self._checked_forecast(context) - target).square().mean()
        return {"loss": mse, "mse": mse}

    def evaluation_output(self, batch: Batch, *, latent_mode: str | None = None) -> dict[str, torch.Tensor]:
        if latent_mode is not None and not self.is_variational:
            raise ValueError(f"{type(self).__name__} is not variational; latent_mode must not be set")
        context, target = self.split(batch["values"])
        return {"output": self._checked_forecast(context), "target": target,
                "time_index": torch.arange(self.context_size, self.window_size)}


@dataclass(frozen=True)
class ModelSpec:
    """Registry entry: how to validate [model] hyperparameters and build the model.

    hyperparameters: frozen dataclass parsed from the TOML [model] table
        (all keys except ``name``); its __post_init__ checks ranges.
    build: ``build(hparams, *, n_channels, window_size) -> TrainableModel``.
        n_channels and window_size come from dataset.h5, never from TOML.
    checkpoint_defaults: values for hyperparameters that did not exist when
        older checkpoints were written. A new hyperparameter may change the
        architecture by default (here: cnn_vae's output layer); an older
        checkpoint lacks the key and is rebuilt with this value instead, so
        its weights still load.
    """

    name: str
    hyperparameters: type
    build: Callable[..., TrainableModel]
    checkpoint_defaults: dict = field(default_factory=dict)
