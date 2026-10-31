"""Files of one trained-model folder, loss plots, and reloading a trained model.

    <trained-model folder>/
      config.toml         exact copy of the training TOML
      run_metadata.json   status, resolved config (all defaults filled), PVs,
                          dataset identity + normalization, model source path
                          and hashes, training-code hashes, versions, summary
      history.csv         per epoch: train_*/val_* for every loss entry, lr, seconds
      loss.png            train vs validation curves, one panel per loss entry
      best.pt             weights at the lowest validation loss
      last.pt             weights + optimizer state after the last epoch

Checkpoints contain only tensors and plain Python values, so they load with
``torch.load(..., weights_only=True)`` (no pickled classes).
"""

import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

from matplotlib.figure import Figure

from .style import science_style
import torch

from models import get_model_spec


CHECKPOINT_FORMAT = 1


def sha256_files(directory: Path, pattern: str = "*.py") -> dict[str, str]:
    """Content hashes of the code that defines a run (model folder, training code)."""
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(Path(directory).glob(pattern))}


def _write_atomic(path: Path, write) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    write(temporary)
    os.replace(temporary, path)


class RunDirectory:
    def __init__(self, output_dir, config_bytes: bytes) -> None:
        self.path = Path(output_dir)
        # parents=True creates e.g. UnivariateTrainedModels/Pressure/D01_L01A;
        # exist_ok=False never reuses or overwrites a trained-model folder.
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "config.toml").write_bytes(config_bytes)
        self.history_path = self.path / "history.csv"
        self.history: list[dict] = []

    def write_metadata(self, metadata: dict) -> None:
        text = json.dumps(metadata, indent=2, allow_nan=False) + "\n"
        _write_atomic(self.path / "run_metadata.json", lambda p: p.write_text(text, encoding="utf-8"))

    def append_history(self, row: dict) -> None:
        if self.history and list(row) != list(self.history[0]):
            raise ValueError("History columns changed between epochs")
        self.history.append(dict(row))
        new_file = len(self.history) == 1
        with self.history_path.open("w" if new_file else "a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    def save_checkpoint(self, name: str, payload: dict) -> Path:
        path = self.path / name
        _write_atomic(path, lambda p: torch.save(payload, p))
        return path

    @science_style()
    def plot_history(self, *, title: str, best_epoch: int | None) -> Path:
        """Save loss.png: train vs val per epoch, one panel per loss entry ("loss" first)."""
        if not self.history:
            raise ValueError("No epochs to plot")
        names = [key[len("train_"):] for key in self.history[0] if key.startswith("train_")]
        epochs = [row["epoch"] for row in self.history]
        figure = Figure(figsize=(6.5, 2.6 * len(names)), layout="constrained")
        axes = figure.subplots(len(names), 1, sharex=True, squeeze=False)[:, 0]
        for axis, name in zip(axes, names):
            axis.plot(epochs, [row[f"train_{name}"] for row in self.history], marker="o", markersize=3, label="train")
            axis.plot(epochs, [row[f"val_{name}"] for row in self.history], marker="o", markersize=3, label="validation")
            if best_epoch is not None:
                axis.axvline(best_epoch, color="0.5", linestyle="--", linewidth=1, label=f"best epoch {best_epoch}")
            axis.set_ylabel(name)
            axis.grid(color="#D9D9D9", linewidth=0.5, alpha=0.72)
        axes[0].legend()
        axes[0].set_title(title)
        axes[-1].set_xlabel("Epoch")
        path = self.path / "loss.png"
        figure.savefig(path, dpi=150)
        return path


def checkpoint_payload(model, *, config, dataset_info, epoch, val_metrics, optimizer=None):
    payload = {
        "format_version": CHECKPOINT_FORMAT,
        "model_name": config.model_name,
        "hyperparameters": asdict(config.model),
        "n_channels": dataset_info.n_channels,
        "window_size": dataset_info.window_size,
        "step_ns": dataset_info.step_ns,
        "pvs": [pv.pv for pv in config.pvs],
        "channel_ids": list(dataset_info.channel_ids),
        "dataset_id": dataset_info.dataset_id,
        "values_state": dataset_info.values_state,
        "value_transform": dataset_info.value_transform,
        "epoch": epoch,
        "val_metrics": dict(val_metrics),
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    return payload


def load_checkpoint(path, *, device="cpu"):
    """Rebuild a model from best.pt/last.pt; return (model in eval mode, checkpoint dict).

    The model is rebuilt through the src/models registry from the saved name
    and hyperparameters (re-validated by its dataclass), then weights load
    strictly. Before scoring other data, compare checkpoint["pvs"] and
    checkpoint["dataset_id"]: test data must contain the same PVs and use the
    training dataset's normalization statistics.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported checkpoint format {checkpoint.get('format_version')!r}")
    spec = get_model_spec(checkpoint["model_name"])
    # Keys missing from older checkpoints take spec.checkpoint_defaults, not the current defaults.
    hyperparameters = spec.hyperparameters(**{**spec.checkpoint_defaults, **checkpoint["hyperparameters"]})
    model = spec.build(hyperparameters, n_channels=checkpoint["n_channels"], window_size=checkpoint["window_size"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval(), checkpoint
