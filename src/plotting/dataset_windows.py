"""Plot stored windows of a belle2-dataset dataset.h5 (no archive access).

    python scripts/plot_dataset_windows.py --config <toml>

TOML tables (see examples/dataset_windows.toml):
    [plot]     name (-> Project/PVPlots/<name>/), optional project_root=".", dpi=200
    [dataset]  path (completed dataset.h5), split ("train", "val" or "test"),
               values = "stored" (as saved, usually normalized) or "raw"
               (saved normalization inverted: stored * scale + offset, float64)
    [select]   EITHER indices = [row, ...] OR any of label (window must carry it,
               e.g. "pre_abort_SBL", "SBL", "non_abort"), event_id; plus
               limit (required with filters), sample = "first" | "random", seed

One PNG per window: one panel per stored channel on the window's UTC
sample times. Associated catalogue events inside the window are drawn as
abort lines; all associations are listed with their offset from the
window's last sample.
"""

import argparse
import json
import sys

import h5py
from matplotlib.figure import Figure
import numpy as np

from .common import (PlotFolder, check_keys, code_sha256, file_fingerprint, folder_name, load_toml,
                     positive_int, resolve_path, resolve_root, text, utc, versions)
from .style import science_style, PV_COLORS, draw_abort_lines, tidy, utc_time_axis


def resolve_windows_config(config_path):
    path, data, raw = load_toml(config_path)
    check_keys(raw, "top level", required=("plot", "dataset", "select"))
    plot = check_keys(raw["plot"], "plot", required=("name",), optional=("project_root", "dpi"))
    dataset = check_keys(raw["dataset"], "dataset", required=("path", "split", "values"))
    select = check_keys(raw["select"], "select", optional=("indices", "label", "event_id", "limit", "sample", "seed"))
    root = resolve_root(path, plot.get("project_root", "."))
    if dataset["split"] not in ("train", "val", "test"):
        raise ValueError("dataset.split must be 'train', 'val' or 'test'")
    if dataset["values"] not in ("stored", "raw"):
        raise ValueError("dataset.values must be 'stored' or 'raw'")
    if "indices" in select:
        if set(select) - {"indices"}:
            raise ValueError("[select] indices cannot be combined with label/event_id/limit/sample/seed")
        indices = select["indices"]
        if not isinstance(indices, list) or not indices or any(type(i) is not int or i < 0 for i in indices):
            raise ValueError("select.indices must be a nonempty list of nonnegative row numbers")
    else:
        if "limit" not in select:
            raise ValueError("[select] needs indices, or filters with a limit")
        positive_int(select["limit"], "select.limit")
        if select.get("sample", "first") not in ("first", "random"):
            raise ValueError("select.sample must be 'first' or 'random'")
        if select.get("sample") == "random" and type(select.get("seed")) is not int:
            raise ValueError("select.sample = 'random' requires an integer select.seed")
        for key in ("label", "event_id"):
            if key in select:
                text(select[key], f"select.{key}")
    settings = {"project_root": str(root), "name": folder_name(plot["name"], "plot.name"),
                "dpi": positive_int(plot.get("dpi", 200), "plot.dpi"),
                "dataset": {"path": str(resolve_path(root, dataset["path"], "dataset.path")),
                            "split": dataset["split"], "values": dataset["values"]},
                "select": dict(select)}
    return path, data, settings


def select_rows(labels, associations, select):
    """Row numbers chosen by [select]; filters keep chronological (row) order."""
    n = len(labels)
    if "indices" in select:
        bad = [i for i in select["indices"] if i >= n]
        if bad:
            raise IndexError(f"select.indices {bad} outside the split's {n} windows")
        return list(dict.fromkeys(select["indices"]))
    rows = [i for i in range(n)
            if ("label" not in select or select["label"] in labels[i])
            and ("event_id" not in select or any(a["event_id"] == select["event_id"] for a in associations[i]))]
    if not rows:
        raise ValueError(f"No windows match {select}")
    if select.get("sample", "first") == "random":
        chosen = np.random.default_rng(select["seed"]).choice(len(rows), size=min(select["limit"], len(rows)), replace=False)
        return [rows[i] for i in sorted(chosen)]
    return rows[:select["limit"]]


@science_style()
def plot_dataset_windows(config_path, *, progress=print):
    progress = progress or (lambda message: None)
    path, data, settings = resolve_windows_config(config_path)
    split, dataset_path = settings["dataset"]["split"], settings["dataset"]["path"]
    with h5py.File(dataset_path, "r") as h5:
        if not bool(h5.attrs.get("complete", False)):
            raise ValueError(f"Dataset is incomplete: {dataset_path}")
        if split not in h5:
            raise ValueError(f"Split {split!r} not in dataset (mode {h5.attrs.get('mode')!r})")
        group = h5[split]
        labels = [json.loads(item) for item in group["labels_json"].asstr()[:]]
        associations = [json.loads(item) for item in group["event_associations_json"].asstr()[:]]
        rows = select_rows(labels, associations, settings["select"])
        values = np.stack([group["values"][row] for row in rows]).astype(np.float64)
        starts = group["window_start_ns"][rows]
        ends = group["window_end_ns"][rows]
        channels = h5["channel_ids"].asstr()[:].tolist()
        step_ns = int(h5.attrs["step_ns"])
        values_state = str(h5.attrs["values_state"])
        value_transform = str(h5.attrs.get("value_transform", "none"))   # absent in older builds
        normalization = json.loads(h5["normalization_json"].asstr()[()])
        identity = json.loads(h5["build_metadata_json"].asstr()[()])["channel_identity"]
        dataset_id = str(h5.attrs["dataset_id"])

    shown = f"stored ({values_state}" + (", log10)" if value_transform == "log10" else ")")
    if settings["dataset"]["values"] == "raw":
        inverted = []
        if values_state == "normalized":
            offset, scale = np.asarray(normalization["offset"]), np.asarray(normalization["scale"])
            values = values * scale[None, :, None] + offset[None, :, None]
            inverted.append(normalization["norm_mode"])
        if value_transform == "log10":
            values = 10.0 ** values
            inverted.append("log10")
        shown = f"raw (inverted {' + '.join(inverted) or 'nothing'}, float64 round-off)"
    pv_names = [entry["expected_pv"] for entry in identity]

    folder = PlotFolder(settings["project_root"], settings["name"], data)
    window_reports, outputs = [], []
    for k, row in enumerate(rows):
        t = np.asarray(starts[k] + np.arange(values.shape[2]) * step_ns, dtype="datetime64[ns]")
        figure = Figure(figsize=(12, 2.2 + 2.0 * len(channels)), layout="constrained")
        axes = figure.subplots(len(channels), 1, sharex=True, squeeze=False)[:, 0]
        inside = [a for a in associations[row] if starts[k] <= a["timestamp_ns"] <= ends[k]]
        for c, ax in enumerate(axes):
            ax.plot(t, values[k, c], color=PV_COLORS[c % len(PV_COLORS)], linewidth=1.3, marker=".", markersize=3)
            draw_abort_lines(ax, inside, alpha=0.8)
            ax.set_ylabel(f"{pv_names[c]}\n{shown.split(' ')[0]}", fontsize=9)
            tidy(ax)
        utc_time_axis(axes[-1], utc(starts[k]), utc(ends[k]))
        axes[-1].set_xlabel("UTC time")
        relation = "; ".join(f"{a['label']} ({a['event_id']}, abort {(a['timestamp_ns'] - ends[k]) / 1e9:+.1f} s "
                             "from last sample)" for a in associations[row]) or "no event association"
        axes[0].set_title(f"{split} window {row} · labels {', '.join(labels[row])} · "
                          f"{utc(starts[k]):%Y-%m-%d %H:%M:%S}–{utc(ends[k]):%H:%M:%S} UTC\n{relation}", fontsize=10)
        filename = f"window_{split}_{row}.png"
        figure.savefig(folder.path / filename, dpi=settings["dpi"], facecolor="white")
        outputs.append(filename)
        window_reports.append({"row": row, "window_start_ns": int(starts[k]), "window_end_ns": int(ends[k]),
                               "window_start_utc": utc(starts[k]).isoformat(), "labels": labels[row],
                               "event_associations": associations[row], "png": filename})
    folder.write_json("metadata.json", {
        "plot": "dataset_windows", "config": {"original_path": str(path), "copy": "config.toml"},
        "resolved": settings, "dataset": {**file_fingerprint(dataset_path), "dataset_id": dataset_id,
                                          "values_state": values_state, "channel_ids": channels,
                                          "channel_identity": identity, "normalization": normalization},
        "value_transform": value_transform, "values_shown": shown, "windows": window_reports, "versions": versions(),
        "code_sha256": code_sha256(), "outputs": outputs,
    })
    progress(f"PLOTS: {len(rows)} window(s) in {folder.path}")
    return folder.path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot stored dataset.h5 windows from a TOML.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    try:
        plot_dataset_windows(args.config)
    except (ValueError, TypeError, KeyError, IndexError, OSError) as error:
        print(f"Plot failed: {error}", file=sys.stderr)
        return 1
    return 0
