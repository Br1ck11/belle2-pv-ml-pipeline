"""Plots and the CSV files holding exactly the plotted data."""

from pathlib import Path
import re

import matplotlib
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from .style import science_style

from .data import REGIONS, relation


RECONSTRUCTIONS = "ModelReconstructions"
ABORT_DISTANCE = "ScoreVsAbortDistance"
REGION_STATISTICS = "RegionStatistics"
REGION_COLORS = {"abort": "#C0392B", "pre_abort": "#E67E22", "post_abort": "#2E86C1", "non_abort": "#4C956C",
                 "other_event": "#9CA3AF"}
REGION_NAMES = {"abort": "abort (window contains the abort)", "pre_abort": "pre-abort", "post_abort": "post-abort",
                "non_abort": "no abort", "other_event": "related to another abort"}


def utc(ns):
    return pd.Timestamp(int(ns), unit="ns", tz="UTC")


def _offset(delta_ns):
    """Human wording of (abort time - last window sample)."""
    seconds = delta_ns / 1e9
    if seconds == 0:
        return "abort at the last sample"
    return f"abort {abs(seconds):.0f} s {'after' if seconds > 0 else 'before'} the last sample"


def _tidy(ax):
    """Light horizontal grid on top of the science style (all four spines kept)."""
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.75)


# ---------------------------------------------------------------- reconstructions

@science_style()
def write_reconstructions(folder: Path, test, kept, chosen, scores, *, dpi=150):
    """One PNG + CSV per chosen window: target and model output on the window's UTC samples.

    Layout: ModelReconstructions/<region>/example_<k>_row<row>.{png,csv}, one
    subfolder per label region (abort, pre_abort, post_abort, non_abort),
    created even when the region has no windows so the layout is fixed.
    """
    root = Path(folder) / RECONSTRUCTIONS
    root.mkdir()
    records = []
    for region in REGIONS:
        out = root / region
        out.mkdir()
        for k, row in enumerate(chosen.get(region, []), start=1):
            result = kept[row]
            window_size = result["input"].shape[1]
            index = np.asarray(result["time_index"], dtype=np.int64)
            forecast = not np.array_equal(index, np.arange(window_size))
            times = test.starts[row] + np.arange(window_size, dtype=np.int64) * test.step_ns
            x = times.astype("datetime64[ns]")
            figure = Figure(figsize=(12, 2.4 + 2.2 * len(test.pvs)), layout="constrained")
            axes = figure.subplots(len(test.pvs), 1, sharex=True, squeeze=False)[:, 0]
            for c, ax in enumerate(axes):
                target_is_input = np.allclose(result["target"][c], result["input"][c, index])
                if forecast:
                    # Context (seen by the model) and horizon (predicted) of the same window.
                    ax.axvspan(x[index[0]], x[index[-1]], color="#9CA3AF", alpha=0.18, linewidth=0,
                               label="forecast horizon")
                    ax.plot(x, result["input"][c], color="#1F2937", linewidth=1.6,
                            label="input window (true values)")
                    if not target_is_input:
                        ax.plot(x[index], result["target"][c], color="#1F2937", linewidth=1.2, linestyle=":",
                                label="target")
                    ax.plot(x[index], result["output"][c], color=REGION_COLORS[region], linewidth=1.4,
                            linestyle="--", marker="o" if index.size <= 3 else None, markersize=3,
                            label="model prediction")
                else:
                    ax.plot(x, result["input"][c], color="#1F2937", linewidth=1.6, label="input (target)")
                    if not target_is_input:
                        ax.plot(x, result["target"][c], color="#1F2937", linewidth=1.2, linestyle=":", label="target")
                    ax.plot(x, result["output"][c], color=REGION_COLORS[region], linewidth=1.4, linestyle="--",
                            label="model output")
                for association in test.associations[row]:
                    if times[0] <= association["timestamp_ns"] <= times[-1]:
                        ax.axvline(np.datetime64(association["timestamp_ns"], "ns"), color="#C0392B",
                                   linewidth=1.0, alpha=0.8)
                ax.set_ylabel(f"{test.pvs[c]}\n[{test.values_state}]", fontsize=9)
                score_label = f"MSE over the {index.size} predicted samples" if forecast else "MSE"
                ax.set_title(f"{score_label} = {scores[row, c]:.4g}", fontsize=9, loc="right")
                _tidy(ax)
            axes[0].legend(frameon=False, fontsize=8, loc="best")      # the corner with least overlap
            locator = mdates.AutoDateLocator(tz=mdates.UTC)
            axes[-1].xaxis.set_major_locator(locator)
            axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=mdates.UTC, usetex=False))
            axes[-1].set_xlabel("UTC time")
            relations = "; ".join(f"{a['label']} ({_offset(a['timestamp_ns'] - test.ends[row])})"
                                  for a in test.associations[row]) or "no event association"
            kind = (f" · forecast of samples {index[0]}..{index[-1]} from the {index[0]} before"
                    if forecast else "")
            figure.suptitle(f"{REGION_NAMES[region]} · example {k} · test row {row} · "
                            f"{utc(test.starts[row]):%Y-%m-%d %H:%M:%S} UTC{kind}\n{relations}", fontsize=10)
            stem = f"example_{k}_row{row}"
            figure.savefig(out / f"{stem}.png", dpi=dpi, facecolor="white")
            # Whole window; target/output are empty (NaN) at samples the model did not output.
            predicted = np.zeros(window_size, dtype=bool)
            predicted[index] = True
            table = {"time_ns": times, "time_utc": [utc(t).isoformat() for t in times], "predicted": predicted}
            for c, pv in enumerate(test.pvs):
                target = np.full(window_size, np.nan)
                output = np.full(window_size, np.nan)
                target[index] = result["target"][c]
                output[index] = result["output"][c]
                table[f"{pv} input"] = result["input"][c]
                table[f"{pv} target"] = target
                table[f"{pv} output"] = output
            pd.DataFrame(table).to_csv(out / f"{stem}.csv", index=False)
            records.append({"region": region, "example": k, "row": int(row), "png": f"{RECONSTRUCTIONS}/{region}/{stem}.png",
                            "csv": f"{RECONSTRUCTIONS}/{region}/{stem}.csv", "labels": test.labels[row]})
    return records


# ---------------------------------------------------------------- score vs distance to an abort

def abort_window_range(test, non_abort_ns):
    """(low, high) of (window's last sample - abort time) shown around an abort, in ns:
    the no-abort stretch and pre-abort region before it, up to the post-abort end."""
    return -(test.pre_abort_ns + non_abort_ns), test.post_abort_ns


def abort_rows(test, event, non_abort_ns):
    """Test rows whose last sample lies in abort_window_range around this abort."""
    low, high = abort_window_range(test, non_abort_ns)
    distance_ns = test.ends - int(event["timestamp_ns"])
    return np.flatnonzero((distance_ns >= low) & (distance_ns <= high))


def file_name(pv):
    """PV identity -> file-system safe name (':' is not allowed everywhere)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", pv).strip("_") or "pv"


def _signal(test, kept, rows, c, event_ns):
    """Continuous PV data stitched from the windows (overlapping windows hold
    identical samples); NaN breaks the line where windows are missing."""
    samples = {}
    for row in rows:
        times = test.starts[row] + np.arange(kept[row]["input"].shape[1], dtype=np.int64) * test.step_ns
        for t, value in zip(times, kept[row]["input"][c]):
            samples.setdefault(int(t), float(value))
    times = np.array(sorted(samples), dtype=np.int64)
    values = np.array([samples[t] for t in times], dtype=np.float64)
    gaps = np.flatnonzero(np.diff(times) > test.step_ns)
    x = np.insert((times - event_ns) / 1e9, gaps + 1, np.nan)
    y = np.insert(values, gaps + 1, np.nan)
    return times, values, x, y


def _relations(test, rows, event):
    rel = []
    for row in rows:
        own = [a for a in test.associations[row] if a["event_id"] == event["event_id"]]
        rel.append(relation(own[0]["label"]) if own else
                   "non_abort" if test.labels[row] == ["non_abort"] else "other_event")
    return np.array(rel, dtype=object)


ORDER = ("non_abort", "pre_abort", "abort", "post_abort", "other_event")


def _region_lines(ax, pre_s, span_s, post_s, extra_s):
    ax.axvline(0, color=REGION_COLORS["abort"], linewidth=1.6)
    ax.axvline(-pre_s, color=REGION_COLORS["pre_abort"], linewidth=1.2, linestyle="--")
    ax.axvline(span_s, color=REGION_COLORS["abort"], linewidth=1.2, linestyle=":")
    ax.axvline(post_s, color=REGION_COLORS["post_abort"], linewidth=1.2, linestyle="--")
    if extra_s > 0:
        ax.axvline(-pre_s - extra_s, color=REGION_COLORS["non_abort"], linewidth=1.2, linestyle="--")


@science_style()
def write_abort_distance(folder: Path, test, scores, kept, pv_indices, events, *, non_abort_ns, reference, dpi=150):
    """Per abort and PV: a two-panel figure on one time axis (time - abort time).

    Top: the PV data (model input, stitched from the windows) and the model
    output of every window as its own segment, coloured by the window's
    region. Bottom: the score of each window at its last sample, i.e. how
    the score evolves around the abort. Windows from non_abort_ns before the
    pre-abort region up to the post-abort end are shown.

    Layout: ScoreVsAbortDistance/<abort>/<PV>.png with <PV>_scores.csv
    (bottom panel) and <PV>_signal.csv (top panel, long format).
    """
    out = Path(folder) / ABORT_DISTANCE
    out.mkdir()
    span_s = (test.window_size - 1) * test.step_ns / 1e9
    pre_s, post_s, extra_s = test.pre_abort_ns / 1e9, test.post_abort_ns / 1e9, non_abort_ns / 1e9
    names = {c: file_name(test.pvs[c]) for c in pv_indices}
    if len(set(names.values())) < len(names):
        names = {c: f"{name}_ch{c}" for c, name in names.items()}
    records = []
    for event in events:
        e = int(event["timestamp_ns"])
        rows = abort_rows(test, event, non_abort_ns)
        rel = _relations(test, rows, event)
        distance_s = (test.ends[rows] - e) / 1e9
        stem = f"{utc(e):%Y%m%dT%H%M%S}_{event['category']}_{event['event_id']}"
        event_dir = out / stem
        event_dir.mkdir()
        pv_records = []
        for c in pv_indices:
            figure = Figure(figsize=(11, 7.2), layout="constrained")
            signal_ax, score_ax = figure.subplots(2, 1, sharex=True, height_ratios=[1.15, 1])
            for ax in (signal_ax, score_ax):
                _region_lines(ax, pre_s, span_s, post_s, extra_s)

            # ---- top: PV data and the model output of each window
            outputs = []
            if len(rows):
                times, values, x, y = _signal(test, kept, rows, c, e)
                signal_ax.plot(x, y, color="#1F2937", linewidth=1.5, zorder=3)
            else:
                times, values = np.array([], dtype=np.int64), np.array([])
            for row, name in zip(rows, rel):
                result = kept[row]
                t = test.starts[row] + np.asarray(result["time_index"], dtype=np.int64) * test.step_ns
                signal_ax.plot((t - e) / 1e9, result["output"][c], color=REGION_COLORS[name], linewidth=0.9,
                               alpha=0.55, zorder=2)
                outputs.append(pd.DataFrame({"series": "output", "row": int(row), "time_s": (t - e) / 1e9,
                                             "time_utc": [utc(v).isoformat() for v in t],
                                             "value": result["output"][c]}))
            signal_ax.set_ylabel(f"{test.pvs[c]}\n[{test.values_state}]", fontsize=9)
            signal_ax.legend(handles=[
                Line2D([0], [0], color="#1F2937", linewidth=1.5, label="PV data (model input)"),
                Line2D([0], [0], color="#6B7280", linewidth=0.9,
                       label="model output of each window (colour = region)")],
                frameon=False, fontsize=8, loc="best")
            _tidy(signal_ax)

            # ---- bottom: score evolution around the abort
            score_ax.axvspan(-pre_s, 0, color=REGION_COLORS["pre_abort"], alpha=0.10, linewidth=0)
            score_ax.axvspan(0, span_s, color=REGION_COLORS["abort"], alpha=0.10, linewidth=0)
            if post_s > span_s:
                score_ax.axvspan(span_s, post_s, color=REGION_COLORS["post_abort"], alpha=0.10, linewidth=0)
            if extra_s > 0:
                score_ax.axvspan(-pre_s - extra_s, -pre_s, color=REGION_COLORS["non_abort"], alpha=0.08, linewidth=0)
            if np.isfinite(reference[c]):
                score_ax.axhline(reference[c], color=REGION_COLORS["non_abort"], linewidth=1.0, linestyle="-.")
            order = np.argsort(distance_s)
            score_ax.plot(distance_s[order], scores[rows[order], c], color="#6B7280", linewidth=0.8, alpha=0.6)
            for name in ORDER:
                mask = rel == name
                if mask.any():
                    score_ax.scatter(distance_s[mask], scores[rows[mask], c], s=22, color=REGION_COLORS[name],
                                     zorder=3)
            score_ax.set_ylabel(f"MSE · {test.pvs[c]}", fontsize=9)
            score_ax.set_xlabel("Time − abort time [s]   (score: at each window's last sample)")
            _tidy(score_ax)

            handles = [Line2D([0], [0], color=REGION_COLORS["abort"], linewidth=1.6, label="abort time (0)")]
            if extra_s > 0:
                handles.append(Line2D([0], [0], color=REGION_COLORS["non_abort"], linestyle="--",
                                      label=f"start of the shown no-abort time (−{pre_s + extra_s:g} s)"))
            handles += [
                Line2D([0], [0], color=REGION_COLORS["pre_abort"], linestyle="--", label=f"pre-abort start (−{pre_s:g} s)"),
                Line2D([0], [0], color=REGION_COLORS["abort"], linestyle=":",
                       label=f"last window containing the abort (+{span_s:g} s)"),
                Line2D([0], [0], color=REGION_COLORS["post_abort"], linestyle="--", label=f"post-abort end (+{post_s:g} s)"),
                Line2D([0], [0], color=REGION_COLORS["non_abort"], linestyle="-.", label="median no-abort score (test set)"),
                *[Patch(color=REGION_COLORS[name], alpha=0.6, label=REGION_NAMES[name])
                  for name in ORDER if (rel == name).any()],
            ]
            figure.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
            figure.suptitle(f"{event['category']} abort {event['event_id']} at {utc(e):%Y-%m-%d %H:%M:%S} UTC · "
                            f"{test.pvs[c]} · {len(rows)} windows", fontsize=11)

            name = names[c]
            figure.savefig(event_dir / f"{name}.png", dpi=dpi, facecolor="white")
            pd.DataFrame({"row": rows, "distance_s": distance_s,
                          "window_end_utc": [utc(t).isoformat() for t in test.ends[rows]],
                          "relation_to_this_abort": rel, "labels": [";".join(test.labels[r]) for r in rows],
                          f"mse {test.pvs[c]}": scores[rows, c]}).sort_values("distance_s").to_csv(
                event_dir / f"{name}_scores.csv", index=False)
            signal = pd.DataFrame({"series": "input", "row": -1, "time_s": (times - e) / 1e9,
                                   "time_utc": [utc(v).isoformat() for v in times], "value": values})
            pd.concat([signal, *outputs], ignore_index=True).to_csv(event_dir / f"{name}_signal.csv", index=False)
            base = f"{ABORT_DISTANCE}/{stem}/{name}"
            pv_records.append({"pv": test.pvs[c], "png": f"{base}.png", "scores_csv": f"{base}_scores.csv",
                               "signal_csv": f"{base}_signal.csv"})
        records.append({"event_id": event["event_id"], "category": event["category"],
                        "timestamp_utc": utc(e).isoformat(), "windows": int(len(rows)),
                        "folder": f"{ABORT_DISTANCE}/{stem}", "pvs": pv_records})
    return records


# ---------------------------------------------------------------- region statistics

def describe(values):
    values = np.asarray(values, dtype=np.float64)
    n = int(values.size)
    return {"n": n, "mean": float(values.mean()) if n else np.nan,
            "sem": float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
            "median": float(np.median(values)) if n else np.nan,
            "std": float(values.std(ddof=1)) if n > 1 else np.nan}


@science_style()
def write_region_statistics(folder: Path, test, scores, pv_indices, regions, *, categories=None, dpi=150):
    """Mean ± standard error of the mean, and median, of the score per region (and per category)."""
    out = Path(folder) / REGION_STATISTICS
    out.mkdir()
    rows, by_category = [], []
    wanted = None if not categories else {c.casefold() for c in categories}
    for c in pv_indices:
        for region in REGIONS:
            members = [i for i, r in enumerate(regions) if region in r]
            rows.append({"pv": test.pvs[c], "region": region, **describe(scores[members, c])})
            if region == "non_abort":
                continue
            per_category = {}
            for i in members:
                for a in test.associations[i]:
                    if relation(a["label"]) == region and (wanted is None or a["category"].casefold() in wanted):
                        per_category.setdefault(a["category"], set()).add(i)
            for category, member_set in sorted(per_category.items()):
                by_category.append({"pv": test.pvs[c], "region": region, "category": category,
                                    **describe(scores[sorted(member_set), c])})
    table = pd.DataFrame(rows)
    table.to_csv(out / "region_statistics.csv", index=False)
    pd.DataFrame(by_category, columns=["pv", "region", "category", "n", "mean", "sem", "median", "std"]).to_csv(
        out / "region_statistics_by_category.csv", index=False)

    figure = Figure(figsize=(9, 1.6 + 3.2 * len(pv_indices)), layout="constrained")
    axes = figure.subplots(len(pv_indices), 1, squeeze=False)[:, 0]
    x = np.arange(len(REGIONS))
    for ax, c in zip(axes, pv_indices):
        part = table[table["pv"] == test.pvs[c]].set_index("region").loc[list(REGIONS)]
        colors = [REGION_COLORS[r] for r in REGIONS]
        ax.bar(x - 0.2, part["mean"], width=0.38, color=colors, alpha=0.85, yerr=part["sem"], capsize=4,
               error_kw={"linewidth": 1.2})
        ax.bar(x + 0.2, part["median"], width=0.38, color=colors, alpha=0.35, hatch="//", edgecolor="#374151")
        for xi, n in zip(x, part["n"]):
            ax.annotate(f"n = {n}", (xi, 0), xytext=(0, -28), textcoords="offset points", ha="center", fontsize=8)
        ax.set_xticks(x, [REGION_NAMES[r].split(" (")[0] for r in REGIONS])
        ax.tick_params(axis="x", which="both", top=False)      # categorical axis: no minor/top ticks
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.set_ylabel(f"MSE · {test.pvs[c]}", fontsize=9)
        _tidy(ax)
    axes[0].legend(handles=[Patch(facecolor="#9CA3AF", alpha=0.85, label="mean ± standard error of the mean"),
                            Patch(facecolor="#9CA3AF", alpha=0.35, hatch="//", edgecolor="#374151", label="median")],
                   fontsize=8, loc="upper right")
    figure.suptitle("Score per label region (windows overlap: the standard error assumes independence)", fontsize=10)
    figure.savefig(out / "region_statistics.png", dpi=dpi, facecolor="white")
    return {"csv": f"{REGION_STATISTICS}/region_statistics.csv",
            "csv_by_category": f"{REGION_STATISTICS}/region_statistics_by_category.csv",
            "png": f"{REGION_STATISTICS}/region_statistics.png", "table": rows}


# ---------------------------------------------------------------- threshold exceedance per abort

EXCEEDANCE_REGIONS = ("pre_abort", "abort", "post_abort")


def abort_region_rows(test, categories=None):
    """[(event, {region: [rows]})] for every abort with at least one test window,
    chronologically; only the selected categories (case-insensitive)."""
    wanted = None if not categories else {c.casefold() for c in categories}
    per_event = {}
    for row, associations in enumerate(test.associations):
        for a in associations:
            if wanted is not None and a["category"].casefold() not in wanted:
                continue
            regions = per_event.setdefault(a["event_id"], {region: [] for region in EXCEEDANCE_REGIONS})
            regions[relation(a["label"])].append(row)
    by_id = {e["event_id"]: e for e in test.events}
    order = sorted(per_event, key=lambda event_id: (by_id[event_id]["timestamp_ns"], event_id))
    return [(by_id[event_id], per_event[event_id]) for event_id in order]


@science_style()
def write_exceedance(folder: Path, test, scores, pv_indices, non_abort_rows, *, percentiles, categories=None, dpi=150):
    """Fraction of aborts whose region has at least one window above a no-abort percentile.

    For each PV and percentile p: threshold s_p = p-th percentile of the
    scores of the test set's no-abort windows. For each region R of
    pre_abort, abort, post_abort:
        f_R = (aborts whose region R has >= 1 window with score > s_p) / N,
    with N = all aborts that have at least one window in the test set
    (selected categories), the same denominator for every region.

    Writes RegionStatistics/exceedance_<PV>.png, exceedance.csv (one row per
    PV, percentile and region) and exceedance_by_abort.csv (one row per PV,
    percentile, abort and region).
    """
    out = Path(folder) / REGION_STATISTICS
    out.mkdir(exist_ok=True)
    aborts = abort_region_rows(test, categories)
    n_aborts = len(aborts)
    non_abort_rows = np.asarray(non_abort_rows, dtype=np.int64)
    percentiles = [float(p) for p in percentiles]
    summary, by_abort, pngs = [], [], {}
    for c in pv_indices:
        pv = test.pvs[c]
        thresholds = (np.percentile(scores[non_abort_rows, c], percentiles) if non_abort_rows.size
                      else np.full(len(percentiles), np.nan))
        fractions = {}
        for p, threshold in zip(percentiles, thresholds):
            for region in EXCEEDANCE_REGIONS:
                hits = with_windows = windows = windows_above = 0
                for event, regions in aborts:
                    rows = regions[region]
                    region_scores = scores[rows, c]
                    above = int((region_scores > threshold).sum())
                    hit = above > 0
                    hits += hit
                    with_windows += bool(rows)
                    windows += len(rows)
                    windows_above += above
                    by_abort.append({"pv": pv, "percentile": p, "threshold": threshold,
                                     "event_id": event["event_id"], "category": event["category"],
                                     "timestamp_utc": utc(event["timestamp_ns"]).isoformat(), "region": region,
                                     "windows": len(rows), "windows_above": above, "exceeded": hit,
                                     "max_score": float(region_scores.max()) if rows else np.nan})
                fraction = hits / n_aborts if n_aborts else np.nan
                fractions[(p, region)] = (hits, fraction)
                summary.append({"pv": pv, "percentile": p, "threshold": threshold,
                                "no_abort_windows": int(non_abort_rows.size), "region": region,
                                "aborts_exceeded": hits, "aborts_total": n_aborts, "fraction": fraction,
                                "aborts_with_windows_in_region": with_windows, "windows_in_region": windows,
                                "windows_above": windows_above})
        if not n_aborts or not np.isfinite(thresholds).all():
            continue          # nothing to plot: no aborts or no no-abort windows

        figure = Figure(figsize=(8.5, 5.4), layout="constrained")
        ax = figure.subplots()
        x = np.arange(len(EXCEEDANCE_REGIONS))
        width = 0.8 / len(percentiles)
        shades = np.linspace(0.35, 0.85, len(percentiles))
        bottom = 0.5 / n_aborts
        for j, (p, threshold) in enumerate(zip(percentiles, thresholds)):
            offset = (j - (len(percentiles) - 1) / 2) * width
            heights = [fractions[(p, r)][1] for r in EXCEEDANCE_REGIONS]
            counts = [fractions[(p, r)][0] for r in EXCEEDANCE_REGIONS]
            color = matplotlib.colormaps["Blues"](shades[j])
            ax.bar(x + offset, [h if h > 0 else np.nan for h in heights], width=width * 0.92, color=color,
                   edgecolor="#1F2937", linewidth=0.4, label=f"{p:g}th percentile (threshold {threshold:.3g})")
            for xi, h, k in zip(x + offset, heights, counts):
                ax.annotate(f"{k}/{n_aborts}", (xi, max(h, bottom)), xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7, rotation=90 if len(percentiles) > 3 else 0)
        ax.axhline(1.0, color="#6B7280", linewidth=0.8, linestyle=":")
        ax.set_yscale("log")
        ax.set_ylim(bottom, 2.0)
        ax.set_xticks(x, [REGION_NAMES[r].split(" (")[0] for r in EXCEEDANCE_REGIONS])
        ax.tick_params(axis="x", which="both", top=False)
        ax.tick_params(axis="x", which="minor", bottom=False)
        for xi, region in zip(x, EXCEEDANCE_REGIONS):
            n_region = sum(bool(regions[region]) for _, regions in aborts)
            ax.annotate(f"{n_region} of {n_aborts} aborts\nhave windows here", (xi, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, -26), textcoords="offset points", ha="center", va="top", fontsize=7)
        ax.set_ylabel(r"$f_R$ = aborts with $\geq 1$ window above threshold / all aborts", fontsize=9)
        figure.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=min(len(percentiles), 3),
                      frameon=False, fontsize=8)
        _tidy(ax)
        figure.suptitle(f"Threshold exceedance per abort · {pv} · N = {n_aborts} aborts\n"
                        f"threshold = percentile of the {non_abort_rows.size} no-abort window scores (test set); "
                        "counted per abort, not per window", fontsize=10)
        name = f"exceedance_{file_name(pv)}"
        figure.savefig(out / f"{name}.png", dpi=dpi, facecolor="white")
        pngs[pv] = f"{REGION_STATISTICS}/{name}.png"
    pd.DataFrame(summary).to_csv(out / "exceedance.csv", index=False)
    pd.DataFrame(by_abort).to_csv(out / "exceedance_by_abort.csv", index=False)
    return {"aborts_total": n_aborts, "percentiles": percentiles, "png": pngs,
            "csv": f"{REGION_STATISTICS}/exceedance.csv", "csv_by_abort": f"{REGION_STATISTICS}/exceedance_by_abort.csv"}
