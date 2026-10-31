"""Model-agnostic optimisation loop.

Knows nothing about files, TOML or a particular model: it moves batches to
the device, calls ``model.compute_losses(batch)``, steps the optimizer on
``"loss"`` and returns exact epoch means of every reported quantity.
"""

import math
import random
import time

import numpy as np
import torch

from models import TrainableModel

from .progress import BatchProgress


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch (all devices). Call before building the model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("run.device = 'cuda' but CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("run.device = 'mps' but MPS is not available")
    return torch.device(name)


def make_optimizer(model: torch.nn.Module, settings) -> torch.optim.Optimizer:
    """Build the [optimizer] table's optimizer over all model parameters."""
    optimizers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
    return optimizers[settings.name](
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay,
        betas=tuple(settings.betas), eps=settings.eps,
    )


def _checked(losses, keys, where):
    if not isinstance(losses, dict) or "loss" not in losses:
        raise TypeError("compute_losses must return a dict containing 'loss'")
    if keys is not None and set(losses) != keys:
        raise ValueError(f"compute_losses returned keys {sorted(losses)}, expected {sorted(keys)}")
    for name, value in losses.items():
        if not isinstance(value, torch.Tensor) or value.ndim != 0:
            raise TypeError(f"Loss {name!r} must be a scalar tensor")
        if not torch.isfinite(value):
            raise FloatingPointError(f"Non-finite {name!r} = {value.item()} at {where}")
    return set(losses)


def run_epoch(model: TrainableModel, loader, device, *, optimizer=None, gradient_clip_norm=None,
              label="epoch", verbose=0) -> dict[str, float]:
    """One pass over ``loader``; trains when an optimizer is given, else evaluates.

    Returns the mean of each loss entry over all windows (batch entries are
    per-window means, so they are weighted by batch size; a smaller last
    batch is therefore counted exactly).
    """
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    keys = None
    n_windows = 0
    live = BatchProgress(label, len(loader)) if verbose else None
    with torch.set_grad_enabled(training):
        for batch_number, batch in enumerate(loader, start=1):
            batch = {name: tensor.to(device, non_blocking=device.type == "cuda") for name, tensor in batch.items()}
            size = int(batch["values"].shape[0])
            losses = model.compute_losses(batch)
            keys = _checked(losses, keys, f"{label}, batch {batch_number}")
            if training:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * size
            n_windows += size
            if live is not None:
                live.update(batch_number, n_windows, totals["loss"] / n_windows)
    if live is not None:
        live.close()
    if n_windows == 0:
        raise ValueError(f"{label}: loader produced no windows")
    return {name: total / n_windows for name, total in totals.items()}


def fit(model, optimizer, train_loader, val_loader, device, settings, *, on_epoch_end, progress, verbose=0):
    """Train for settings.epochs with optional early stopping on validation loss.

    ``on_epoch_end(epoch, row, improved)`` is called after every epoch (the
    train script uses it to write history and checkpoints). ``improved``
    means the validation "loss" is strictly below the best so far.
    """
    best_loss, best_epoch, stale = math.inf, None, 0
    stopped_early = False
    epoch = 0
    for epoch in range(1, settings.epochs + 1):
        started = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, device, optimizer=optimizer,
                                  gradient_clip_norm=settings.gradient_clip_norm,
                                  label=f"Epoch {epoch}/{settings.epochs} train", verbose=verbose)
        val_metrics = run_epoch(model, val_loader, device, label=f"Epoch {epoch}/{settings.epochs} val  ",
                                verbose=verbose)
        row = {"epoch": epoch,
               **{f"train_{name}": value for name, value in train_metrics.items()},
               **{f"val_{name}": value for name, value in val_metrics.items()},
               "learning_rate": optimizer.param_groups[0]["lr"],
               "seconds": time.perf_counter() - started}
        improved = val_metrics["loss"] < best_loss
        if improved:
            best_loss, best_epoch, stale = val_metrics["loss"], epoch, 0
        else:
            stale += 1
        on_epoch_end(epoch, row, improved)
        progress(f"Epoch {epoch}/{settings.epochs}  train_loss={train_metrics['loss']:.6g}  "
                 f"val_loss={val_metrics['loss']:.6g}  ({row['seconds']:.1f} s)" + ("  *best" if improved else ""))
        if settings.early_stopping_patience is not None and stale >= settings.early_stopping_patience:
            stopped_early = True
            progress(f"Early stopping: no validation improvement for {stale} epochs")
            break
    return {"best_epoch": best_epoch, "best_val_loss": best_loss,
            "epochs_completed": epoch, "stopped_early": stopped_early}
