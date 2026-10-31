"""Test a trained model on a mode="test" dataset: python -m evaluation --config X.toml.

  1. Validate the TOML; load the checkpoint through the src/models registry.
  2. Check compatibility: channels, PV identities, value_transform and the
     normalization source must match (errors); window length / sample
     spacing may differ (warnings, the model is still applied).
  3. Score every test window: per-channel MSE of evaluation_output.
  4. Write <trained-model folder>/<run.name>/ (next to the checkpoint) with window_scores.csv,
     ModelReconstructions/, ScoreVsAbortDistance/, RegionStatistics/,
     config.toml (exact copy) and metadata.json.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import torch

from train.engine import resolve_device
from train.outputs import load_checkpoint

from . import __version__
from .config import resolve_test_config
from .data import REGIONS, check_compatibility, load_test_set, window_regions
from .outputs import (abort_rows, utc, write_abort_distance, write_exceedance, write_reconstructions,
                      write_region_statistics)
from .scoring import score_windows


def _code_sha256():
    import models
    import train
    hashes = {}
    for package in (Path(__file__).parent, Path(train.__file__).parent, Path(models.__file__).parent):
        for path in sorted(package.rglob("*.py")):
            hashes[f"{package.name}/{path.relative_to(package)}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def choose_reconstruction_rows(regions, settings, rng):
    """Per region: windows belonging only to that region first, then mixed ones."""
    chosen = {}
    for region in REGIONS:
        pure = [i for i, r in enumerate(regions) if r == {region}]
        mixed = [i for i, r in enumerate(regions) if region in r and len(r) > 1]
        picks = []
        for pool in (pure, mixed):
            need = settings.windows_per_region - len(picks)
            if need <= 0 or not pool:
                continue
            if settings.sample == "random":
                picks += sorted(int(i) for i in rng.choice(pool, size=min(need, len(pool)), replace=False))
            else:
                picks += pool[:need]
        chosen[region] = picks
    return chosen


def output_description(time_index, window_size):
    """Reconstruction (all T samples) or forecast (a subset, typically the last H)."""
    index = np.asarray(time_index)
    if np.array_equal(index, np.arange(window_size)):
        return {"kind": "reconstruction", "samples": int(window_size)}
    return {"kind": "forecast", "samples": int(index.size), "first_window_position": int(index[0]),
            "last_window_position": int(index[-1]), "time_index": index.tolist()}


def check_categories(test, categories):
    """Abort events (with windows in the test set) per category; every requested
    category must occur. Matching is case-insensitive; returns the counts."""
    referenced = {a["event_id"] for associations in test.associations for a in associations}
    counts = {}
    for event in test.events:
        if event["event_id"] in referenced:
            counts[event["category"]] = counts.get(event["category"], 0) + 1
    if categories:
        present = {category.casefold() for category in counts}
        missing = [c for c in categories if c.casefold() not in present]
        if missing:
            available = ", ".join(f"{c} ({n})" for c, n in sorted(counts.items())) or "none"
            raise ValueError(f"selection.categories {missing}: no such aborts in the test set "
                             f"{test.path}. Abort categories present (events): {available}")
    return counts


def choose_events(test, settings, categories, rng):
    referenced = {a["event_id"] for associations in test.associations for a in associations}
    wanted = None if not categories else {c.casefold() for c in categories}
    events = sorted((e for e in test.events if e["event_id"] in referenced
                     and (wanted is None or e["category"].casefold() in wanted)),
                    key=lambda e: (e["timestamp_ns"], e["event_id"]))
    if settings.event_ids is not None:
        by_id = {e["event_id"]: e for e in test.events}
        missing = [i for i in settings.event_ids if i not in by_id]
        if missing:
            raise ValueError(f"abort_distance.event_ids not in the test set: {missing}")
        if wanted is not None:
            other = [i for i in settings.event_ids if by_id[i]["category"].casefold() not in wanted]
            if other:
                raise ValueError(f"abort_distance.event_ids {other} are not of selection.categories "
                                 f"{list(categories)}")
        return [by_id[i] for i in settings.event_ids]
    if settings.selection == "random" and len(events) > settings.n_aborts:
        picks = rng.choice(len(events), size=settings.n_aborts, replace=False)
        return [events[i] for i in sorted(picks)]
    return events[:settings.n_aborts]


def evaluate_model(config_path, *, progress=print):
    """Run the test; return the results folder Path. progress=None is quiet."""
    live = progress is not None
    progress = progress or (lambda message: None)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    config_path = Path(config_path).expanduser().resolve()
    config_bytes = config_path.read_bytes()
    config = resolve_test_config(config_path)
    device = resolve_device(config.run.device)
    model, checkpoint = load_checkpoint(config.checkpoint_path, device=device)
    test = load_test_set(config.dataset_path)
    warnings = check_compatibility(checkpoint, test, vae=config.model.VAE, model=model)
    for warning in warnings:
        progress(f"WARNING: {warning}")

    selected_pvs = list(config.selection.pvs or test.pvs)
    unknown = [pv for pv in selected_pvs if pv not in test.pvs]
    if unknown:
        raise ValueError(f"selection.pvs {unknown} are not channels of the test set {test.pvs}")
    pv_indices = [test.pvs.index(pv) for pv in selected_pvs]
    categories = config.selection.categories
    category_counts = check_categories(test, categories)
    regions = [window_regions(l, a, categories) for l, a in zip(test.labels, test.associations)]
    rng = np.random.default_rng(config.run.seed)
    chosen = choose_reconstruction_rows(regions, config.reconstructions, rng)
    events = choose_events(test, config.abort_distance, categories, rng)

    progress(f"Test set {test.path}: {test.n_windows:,} windows, (C, T) = ({len(test.pvs)}, {test.window_size})")
    progress(f"Model {checkpoint['model_name']} (epoch {checkpoint['epoch']}) on {device}"
             + (f", latent_mode = {config.model.latent_mode}" if config.model.VAE else ""))
    torch.manual_seed(config.run.seed)              # makes sampling_z reproducible
    non_abort_ns = int(round(config.abort_distance.non_abort_seconds * 1e9))
    # Keep the model output of every plotted window: reconstruction examples
    # and all windows around the aborts of the score-vs-distance figures.
    keep = {row for rows in chosen.values() for row in rows}
    for event in events:
        keep.update(int(row) for row in abort_rows(test, event, non_abort_ns))
    scores, kept, time_index = score_windows(model, test, batch_size=config.data.batch_size, device=device,
                                 latent_mode=config.model.latent_mode, keep_rows=keep,
                                 verbose=config.run.verbose if live else 0)

    config.output_dir.mkdir(parents=True)
    (config.output_dir / "config.toml").write_bytes(config_bytes)
    table = pd.DataFrame({
        "row": np.arange(test.n_windows), "window_start_ns": test.starts, "window_end_ns": test.ends,
        "window_end_utc": [utc(t).isoformat() for t in test.ends],
        "labels": [";".join(l) for l in test.labels],
        "regions": [";".join(r for r in REGIONS if r in regs) for regs in regions],
        "event_ids": [";".join(a["event_id"] for a in assoc) for assoc in test.associations],
    })
    for c, pv in enumerate(test.pvs):
        table[f"mse {pv}"] = scores[:, c]
    table["mse mean selected PVs"] = scores[:, pv_indices].mean(axis=1)
    table.to_csv(config.output_dir / "window_scores.csv", index=False)

    non_abort = [i for i, r in enumerate(regions) if r == {"non_abort"}]
    reference = np.array([np.median(scores[non_abort, c]) if non_abort else np.nan for c in range(len(test.pvs))])
    reconstruction_records = write_reconstructions(config.output_dir, test, kept, chosen, scores)
    distance_records = write_abort_distance(config.output_dir, test, scores, kept, pv_indices, events,
                                            non_abort_ns=non_abort_ns, reference=reference)
    statistics = write_region_statistics(config.output_dir, test, scores, pv_indices, regions, categories=categories)
    exceedance = write_exceedance(config.output_dir, test, scores, pv_indices, non_abort,
                                  percentiles=config.exceedance.percentiles, categories=categories)
    described = output_description(time_index, test.window_size)
    progress(f"Model output: {described['kind']} of {described['samples']} of {test.window_size} samples per window")
    progress(f"Reconstructions: {len(reconstruction_records)} windows · score-vs-distance: {len(distance_records)} aborts")

    counts = {region: sum(region in r for r in regions) for region in REGIONS}
    metadata = {
        "status": "completed", "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {"original_path": str(config_path), "copy": "config.toml",
                   "sha256": hashlib.sha256(config_bytes).hexdigest()},
        "resolved_config": config.to_dict(), "warnings": warnings,
        "score": {"name": "mse", "definition": "per window and channel: mean over output steps of "
                  "(evaluation_output.output - target)^2, in the dataset's stored units",
                  "output": output_description(time_index, test.window_size),
                  "latent_mode": config.model.latent_mode, "selected_pvs": selected_pvs,
                  "standard_error_note": "standard error of the mean = std(ddof=1)/sqrt(n) assumes independent "
                  "windows; overlapping windows are correlated, so it is a lower bound on the uncertainty"},
        "checkpoint": {"path": str(config.checkpoint_path), "model_name": checkpoint["model_name"],
                       "epoch": checkpoint["epoch"], "hyperparameters": checkpoint["hyperparameters"],
                       "training_dataset_id": checkpoint["dataset_id"], "pvs": checkpoint["pvs"],
                       "window_size": checkpoint["window_size"], "step_ns": checkpoint.get("step_ns")},
        "test_dataset": {"path": test.path, "dataset_id": test.dataset_id, "values_state": test.values_state,
                         "value_transform": test.value_transform, "pvs": test.pvs, "window_size": test.window_size,
                         "step_ns": test.step_ns, "pre_abort_ns": test.pre_abort_ns,
                         "post_abort_ns": test.post_abort_ns, "normalization_source": test.normalization_source,
                         "windows": test.n_windows},
        "abort_categories": {"selected": list(categories) if categories else "all",
                             "events_in_test_set": category_counts},
        "region_window_counts": counts,
        "windows_in_several_regions": int(sum(len(r) > 1 for r in regions)),
        "windows_in_no_selected_region": int(sum(len(r) == 0 for r in regions)),
        "outputs": {"window_scores": "window_scores.csv", "reconstructions": reconstruction_records,
                    "abort_distance": distance_records,
                    "region_statistics": {k: v for k, v in statistics.items() if k != "table"},
                    "exceedance": exceedance},
        "versions": {"belle2_evaluation": __version__, "python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "h5py": h5py.__version__,
                     "matplotlib": matplotlib.__version__},
        "code_sha256": _code_sha256(),
    }
    (config.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n",
                                                     encoding="utf-8")
    progress(f"RESULTS: {config.output_dir}")
    return config.output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Test a trained model on a mode='test' belle2-dataset file.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        evaluate_model(args.config)
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, FloatingPointError) as error:
        print(f"Testing failed: {error}", file=sys.stderr)
        return 1
    return 0
