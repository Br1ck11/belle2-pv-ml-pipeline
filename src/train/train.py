"""Train one registered model from a TOML file: python -m train --config X.toml.

The training machinery is model-independent; models live in src/models and
are selected by [model].name. Order of operations:

  1. Validate the TOML; [model] hyperparameters are parsed into the selected
     model's dataclass, the output folder follows from [[pvs]] and run.name.
  2. Inspect dataset.h5 (complete train_val build) and require that [[pvs]]
     equal its stored channel identities, in order. C and T come from the file.
  3. Seed, choose the device, build the model and optimizer.
  4. Create the trained-model folder (never overwritten) and copy the TOML.
  5. Shared epoch loop; history.csv, best.pt and last.pt after every epoch.
  6. When training finishes: plot train/val losses to loss.png and mark
     run_metadata.json "completed" (a failure is recorded as "failed").
"""

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import platform
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch

import models
from models import get_model_spec

from . import __version__
from .config import resolve_config
from .data import H5WindowDataset, check_pvs, inspect_dataset, make_loaders
from .engine import fit, make_optimizer, resolve_device, seed_everything
from .outputs import RunDirectory, checkpoint_payload, sha256_files


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def train_model(config_path, *, progress=print):
    """Train one model; return the trained-model folder Path. progress=None is quiet."""
    live = progress is not None                    # progress=None (tests, library use) is fully quiet
    progress = progress or (lambda message: None)
    started = _now()
    config_path = Path(config_path).expanduser().resolve()
    config_bytes = config_path.read_bytes()
    config = resolve_config(config_path)

    info = inspect_dataset(config.dataset_path)
    check_pvs(config.pvs, info)
    spec = get_model_spec(config.model_name)
    seed_everything(config.run.seed)
    device = resolve_device(config.run.device)
    model = spec.build(config.model, n_channels=info.n_channels, window_size=info.window_size).to(device)
    optimizer = make_optimizer(model, config.optimizer)
    n_parameters = sum(parameter.numel() for parameter in model.parameters())
    model_source = Path(inspect.getsourcefile(type(model))).resolve()
    dataset_stat = config.dataset_path.stat()

    progress(f"Dataset {info.path}: {info.split_counts['train']:,} train / {info.split_counts['val']:,} val "
             f"windows, (C, T) = ({info.n_channels}, {info.window_size}), values {info.values_state}"
             + (" log10" if info.value_transform == "log10" else ""))
    if info.values_state != "normalized":
        progress("Note: training on RAW (unnormalized) values, as stored in the dataset")
    progress(f"Model {config.model_name}: {n_parameters:,} parameters on {device}")
    progress(f"Output: {config.output_dir}")

    datasets = {split: H5WindowDataset(info.path, split, in_memory=config.data.in_memory)
                for split in ("train", "val")}
    try:
        train_loader, val_loader = make_loaders(
            datasets["train"], datasets["val"], batch_size=config.training.batch_size,
            num_workers=config.data.num_workers, seed=config.run.seed, device=device,
        )
        run = RunDirectory(config.output_dir, config_bytes)
        metadata = {
            "status": "running",
            "started_utc": started,
            "layout": config.layout,
            "pvs": [{"pv": pv.pv, "type": pv.type, "name": pv.name, "dataset_channel_id": channel}
                    for pv, channel in zip(config.pvs, info.channel_ids)],
            "config": {"original_path": str(config_path), "copy": "config.toml",
                       "sha256": hashlib.sha256(config_bytes).hexdigest(),
                       "note": "relative paths in config.toml resolve against original_path's folder"},
            "resolved_config": config.to_dict(),
            "dataset": {**info.to_dict(), "size_bytes": dataset_stat.st_size, "mtime_ns": dataset_stat.st_mtime_ns},
            "model": {"name": config.model_name, "source_path": str(model_source),
                      "source_sha256": sha256_files(model_source.parent),
                      "registry_sha256": sha256_files(Path(models.__file__).parent),
                      "parameters": n_parameters, "architecture": str(model)},
            "training_code_sha256": sha256_files(Path(__file__).parent),
            "device": str(device),
            "versions": {"belle2_train": __version__, "python": platform.python_version(),
                         "torch": torch.__version__, "numpy": np.__version__, "h5py": h5py.__version__,
                         "matplotlib": matplotlib.__version__},
            "files": {"config": "config.toml", "metadata": "run_metadata.json", "history": "history.csv",
                      "loss_plot": "loss.png", "best_checkpoint": "best.pt", "last_checkpoint": "last.pt"},
        }
        run.write_metadata(metadata)

        def on_epoch_end(epoch, row, improved):
            run.append_history(row)
            val_metrics = {key[len("val_"):]: value for key, value in row.items() if key.startswith("val_")}
            common = dict(config=config, dataset_info=info, epoch=epoch, val_metrics=val_metrics)
            if improved:
                run.save_checkpoint("best.pt", checkpoint_payload(model, **common))
            run.save_checkpoint("last.pt", checkpoint_payload(model, optimizer=optimizer, **common))

        try:
            summary = fit(model, optimizer, train_loader, val_loader, device, config.training,
                          on_epoch_end=on_epoch_end, progress=progress,
                          verbose=config.run.verbose if live else 0)
            pv_label = config.pvs[0].pv if config.layout == "univariate" else f"{len(config.pvs)} PVs"
            run.plot_history(title=f"{config.model_name} / {config.run.name} ({pv_label})",
                             best_epoch=summary["best_epoch"])
        except BaseException as error:
            metadata.update(status="failed", finished_utc=_now(), error=f"{type(error).__name__}: {error}",
                            epochs_recorded=len(run.history))
            run.write_metadata(metadata)
            raise
        best_row = next(row for row in run.history if row["epoch"] == summary["best_epoch"])
        metadata.update(status="completed", finished_utc=_now(),
                        summary={**summary, "best_epoch_metrics": best_row, "last_epoch_metrics": run.history[-1]})
        run.write_metadata(metadata)
    finally:
        for dataset in datasets.values():
            dataset.close()
    progress(f"TRAINED MODEL: {run.path}")
    return run.path


def main(argv=None):
    """CLI entry point; the TOML is the sole source of settings."""
    parser = argparse.ArgumentParser(description="Train a registered model (src/models) on a belle2-dataset HDF5 file.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        train_model(args.config)
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, FloatingPointError) as error:
        print(f"Training failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
