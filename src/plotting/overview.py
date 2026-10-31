"""Overview plot: beam currents (top) and arbitrary PVs (bottom) with aborts.

    python scripts/plot_pv_overview.py --config <toml>

TOML tables (see examples/pv_overview.toml):
    [plot]       name (-> Project/PVPlots/<name>/), start, end (explicit timezone,
                 interval [start, end)), summary_period (pandas offset, e.g. "1s",
                 "1min"), optional project_root=".", y_scale="auto"|"log"|"linear",
                 dpi=300, title
    [aborts]     filepath, optional categories=[] (empty = all; case-insensitive,
                 e.g. ["SBL", "BeamLoss"]), mark_pressure_bursts=true
    [currents]   optional; ler = {pv, filepaths}, her = {pv, filepaths}, unit="mA"
    [[pvs]]      one or more bottom-panel PVs sharing one unit: pv, filepaths,
                 optional unit="unknown", label

Display summary: raw measurements are binned per summary_period (min,
median, max, count); empty bins stay gaps. Medians are drawn as lines,
the PV bin maximum dotted (keeps short bursts visible), the current min-max
as a band. Abort lines use exact catalogue timestamps. A gold marker is
drawn for each (abort, pressure_burst location) pair whose location is a
':'-separated part of a plotted PV name (e.g. D07_L09 in
VALCCG:D07_L09:PRES), at the latest finite raw value at or before the abort.
"""

import argparse
import sys

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .aborts import load_events
from .archive import load_pv
from .common import (PlotFolder, check_keys, code_sha256, file_fingerprint, finite_or_none, folder_name,
                     load_toml, positive_int, resolve_path, resolve_root, text, utc, utc_ns, versions)
from .style import (science_style, BURST_FACE, CURRENT_STYLES, PV_COLORS, RING_MARKERS, abort_style, draw_abort_lines,
                    latest_finite_at_or_before, tidy, utc_time_axis)


def resolve_overview_config(config_path):
    """Validate the TOML; return (path, exact bytes, resolved settings dict)."""
    path, data, raw = load_toml(config_path)
    check_keys(raw, "top level", required=("plot", "aborts", "pvs"), optional=("currents",))
    plot = check_keys(raw["plot"], "plot", required=("name", "start", "end", "summary_period"),
                      optional=("project_root", "y_scale", "dpi", "title"))
    root = resolve_root(path, plot.get("project_root", "."))
    start_ns, end_ns = utc_ns(plot["start"], "plot.start"), utc_ns(plot["end"], "plot.end")
    if end_ns <= start_ns:
        raise ValueError("plot.end must be later than plot.start")
    period = text(plot["summary_period"], "plot.summary_period")
    try:
        if pd.Timedelta(pd.tseries.frequencies.to_offset(period)) <= pd.Timedelta(0):
            raise ValueError
    except (ValueError, TypeError) as error:
        raise ValueError(f"plot.summary_period {period!r} must be a positive pandas offset such as '1s'") from error
    y_scale = plot.get("y_scale", "auto")
    if y_scale not in ("auto", "log", "linear"):
        raise ValueError("plot.y_scale must be 'auto', 'log' or 'linear'")

    aborts = check_keys(raw["aborts"], "aborts", required=("filepath",),
                        optional=("categories", "mark_pressure_bursts"))
    categories = aborts.get("categories", [])
    if not isinstance(categories, list) or any(not isinstance(c, str) or not c.strip() for c in categories):
        raise ValueError("aborts.categories must be a list of category names ([] = all)")
    mark_bursts = aborts.get("mark_pressure_bursts", True)
    if type(mark_bursts) is not bool:
        raise ValueError("aborts.mark_pressure_bursts must be true or false")

    def pv_entry(table, section, optional):
        check_keys(table, section, required=("pv", "filepaths"), optional=optional)
        return {"pv": text(table["pv"], f"{section}.pv"), "filepaths": table["filepaths"], **{
            key: text(table[key], f"{section}.{key}") for key in optional if key in table}}

    currents = None
    if "currents" in raw:
        table = check_keys(raw["currents"], "currents", required=("ler", "her"), optional=("unit",))
        currents = {"unit": text(table.get("unit", "mA"), "currents.unit"),
                    "LER": pv_entry(table["ler"], "currents.ler", ()),
                    "HER": pv_entry(table["her"], "currents.her", ())}
        if currents["LER"]["pv"] == currents["HER"]["pv"]:
            raise ValueError("currents.ler and currents.her must be different PVs")

    if not isinstance(raw["pvs"], list) or not raw["pvs"]:
        raise ValueError("Provide at least one [[pvs]] entry for the bottom panel")
    pvs = [pv_entry(table, f"pvs[{i}]", ("unit", "label")) for i, table in enumerate(raw["pvs"])]
    for pv in pvs:
        pv.setdefault("unit", "unknown")
    if len({pv["pv"] for pv in pvs}) != len(pvs):
        raise ValueError("[[pvs]] must not repeat a PV")
    if len({pv["unit"] for pv in pvs}) != 1:
        raise ValueError("All bottom-panel [[pvs]] must share one unit (they share one y-axis)")

    settings = {
        "project_root": str(root), "name": folder_name(plot["name"], "plot.name"),
        "start": utc(start_ns).isoformat(), "end": utc(end_ns).isoformat(),
        "start_ns": start_ns, "end_ns": end_ns, "summary_period": period, "y_scale": y_scale,
        "dpi": positive_int(plot.get("dpi", 300), "plot.dpi"),
        "title": text(plot["title"], "plot.title") if "title" in plot else None,
        "aborts": {"filepath": str(resolve_path(root, aborts["filepath"], "aborts.filepath")),
                   "categories": categories, "mark_pressure_bursts": mark_bursts},
        "currents": currents, "pvs": pvs,
    }
    return path, data, settings


def summarize(times_ns, values, period):
    """Per-bin min/median/max/count of raw samples; duplicate timestamps keep the last value."""
    series = pd.Series(values, index=pd.to_datetime(times_ns, unit="ns", utc=True))
    series = series[~series.index.duplicated(keep="last")]
    return series.resample(period).agg(["min", "median", "max", "count"])


def _series_report(data):
    finite = np.isfinite(data.values)
    return {"pv": data.pv, "samples": int(data.values.size), "nonfinite": int((~finite).sum()),
            "first_sample": utc(data.times_ns[0]).isoformat() if data.values.size else None,
            "last_sample": utc(data.times_ns[-1]).isoformat() if data.values.size else None,
            "min": finite_or_none(data.values[finite].min()) if finite.any() else None,
            "max": finite_or_none(data.values[finite].max()) if finite.any() else None,
            "files": data.files}


@science_style()
def render_overview(settings, current_data, pv_data, events):
    """Draw the figure; return (Figure, counts of burst markers)."""
    start, end = utc(settings["start_ns"]), utc(settings["end_ns"])
    period = settings["summary_period"]
    has_currents = bool(current_data)
    if has_currents:
        figure = Figure(figsize=(17, 10))
        current_ax, pv_ax = figure.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [1.6, 2.4], "hspace": 0.10})
    else:
        figure = Figure(figsize=(17, 7))
        current_ax, pv_ax = None, figure.subplots()

    handles = []
    if has_currents:
        for ring, data in current_data.items():
            color, linestyle = CURRENT_STYLES[ring]
            summary = summarize(data.times_ns, data.values, period)
            x = summary.index.tz_convert("UTC").tz_localize(None).to_numpy()
            current_ax.fill_between(x, summary["min"].to_numpy(float), summary["max"].to_numpy(float),
                                    color=color, alpha=0.14, linewidth=0, zorder=2)
            handles.append(current_ax.plot(x, summary["median"].to_numpy(float), color=color, linestyle=linestyle,
                                           linewidth=1.2, label=f"{data.pv} · {period} median", zorder=3)[0])
            for event in events:
                if event["ring"] not in (ring, "Both"):
                    continue
                value = latest_finite_at_or_before(data.times_ns, data.values, event["timestamp_ns"])
                if value is not None:
                    current_ax.scatter(np.datetime64(event["timestamp_ns"], "ns"), value,
                                       marker=RING_MARKERS.get(event["ring"], "o"), s=42, facecolor=color,
                                       edgecolor="#111827", linewidth=0.75, zorder=5)
        draw_abort_lines(current_ax, events, alpha=0.52)
        current_ax.set_ylabel(f"Beam current [{settings['currents']['unit']}]")
        tidy(current_ax)

    colors = {pv["pv"]: PV_COLORS[i % len(PV_COLORS)] for i, pv in enumerate(settings["pvs"])}
    labels = {pv["pv"]: pv.get("label", pv["pv"]) for pv in settings["pvs"]}
    positive_only = True
    for name, data in pv_data.items():
        summary = summarize(data.times_ns, data.values, period)
        x = summary.index.tz_convert("UTC").tz_localize(None).to_numpy()
        median, maximum = summary["median"].to_numpy(float), summary["max"].to_numpy(float)
        finite = np.concatenate([median[np.isfinite(median)], maximum[np.isfinite(maximum)]])
        positive_only &= bool(finite.size == 0 or np.all(finite > 0))
        handles.append(pv_ax.plot(x, median, color=colors[name], linewidth=1.25,
                                  label=f"{labels[name]} · {period} median", zorder=3)[0])
        pv_ax.plot(x, maximum, color=colors[name], linewidth=0.85, alpha=0.58, linestyle=":", zorder=2)
    draw_abort_lines(pv_ax, events, alpha=0.34)

    references, drawn, burst_rings = 0, 0, []
    for event in events if settings["aborts"]["mark_pressure_bursts"] else ():
        for location in event["pressure_burst_locations"]:
            references += 1
            for name, data in pv_data.items():
                if location not in name.split(":"):
                    continue
                value = latest_finite_at_or_before(data.times_ns, data.values, event["timestamp_ns"])
                if value is None:
                    continue
                pv_ax.scatter(np.datetime64(event["timestamp_ns"], "ns"), value,
                              marker=RING_MARKERS.get(event["ring"], "o"), s=68, facecolor=BURST_FACE,
                              edgecolor=colors[name], linewidth=1.25, zorder=5)
                drawn += 1
                if event["ring"] not in burst_rings:
                    burst_rings.append(event["ring"])

    scale = settings["y_scale"]
    if scale == "log" or (scale == "auto" and positive_only):
        pv_ax.set_yscale("log")
    unit = settings["pvs"][0]["unit"]
    pv_ax.set_ylabel(f"{labels[settings['pvs'][0]['pv']]} [{unit}]" if len(pv_data) == 1 else f"[{unit}]")
    pv_ax.set_xlabel("UTC time")
    utc_time_axis(pv_ax, start, end)
    tidy(pv_ax, log_minor=True)

    encodings = []
    if has_currents:
        encodings.append(Line2D([0], [0], color="#6B7280", linewidth=7, alpha=0.18, label="Current within-bin min–max"))
    encodings.append(Line2D([0], [0], color="#555555", linewidth=0.9, linestyle=":", label=f"{period} maximum"))
    for category in dict.fromkeys(event["category"] for event in events):
        color, linestyle = abort_style(category)
        encodings.append(Line2D([0], [0], color=color, linewidth=1.2, linestyle=linestyle, label=f"Abort: {category}"))
    for ring in burst_rings:
        encodings.append(Line2D([0], [0], linestyle="None", marker=RING_MARKERS.get(ring, "o"), markersize=7,
                                markerfacecolor=BURST_FACE, markeredgecolor="#333333",
                                label=f"Mapped pressure event: {ring}"))
    all_handles = [*handles, *encodings]
    figure.legend(handles=all_handles, loc="upper center", bbox_to_anchor=(0.5, 0.91),
                  ncol=min(4, len(all_handles)), frameon=False, fontsize=8.3, columnspacing=1.2)

    categories = settings["aborts"]["categories"]
    default_title = "SuperKEKB currents, PVs, and aborts" if has_currents else "SuperKEKB PVs and aborts"
    figure.suptitle(settings["title"] or default_title, fontsize=16, y=0.985)
    subtitle = (f"{start:%Y-%m-%d %H:%M:%S} to {end:%Y-%m-%d %H:%M:%S} UTC · {len(events)} aborts"
                f" ({'all categories' if not categories else 'categories: ' + ', '.join(categories)})")
    if settings["aborts"]["mark_pressure_bursts"]:
        subtitle += f" · pressure-burst associations marked: {drawn} of {references} catalogue event–location references"
    figure.text(0.5, 0.947, subtitle, ha="center", va="top", fontsize=10, color="#444444")
    figure.subplots_adjust(left=0.085, right=0.985, bottom=0.09 if has_currents else 0.12, top=0.80)
    return figure, {"burst_references": references, "burst_markers_drawn": drawn}


@science_style()
def plot_overview(config_path, *, progress=print):
    """Build PVPlots/<name>/ with overview.png, config.toml and metadata.json; return the folder."""
    progress = progress or (lambda message: None)
    path, data, settings = resolve_overview_config(config_path)
    root = settings["project_root"]
    start_ns, end_ns = settings["start_ns"], settings["end_ns"]
    events, catalog_sha256 = load_events(settings["aborts"]["filepath"], start_ns, end_ns,
                                         categories=settings["aborts"]["categories"])
    current_data = {}
    for ring in ("LER", "HER") if settings["currents"] else ():
        entry = settings["currents"][ring]
        current_data[ring] = load_pv(entry["pv"], entry["filepaths"], root, start_ns, end_ns)
    pv_data = {entry["pv"]: load_pv(entry["pv"], entry["filepaths"], root, start_ns, end_ns)
               for entry in settings["pvs"]}
    for name, loaded in [*((d.pv, d) for d in current_data.values()), *pv_data.items()]:
        progress(f"LOADED {name}: {loaded.values.size:,} samples from {len(loaded.files)} file(s)")
        if loaded.values.size == 0:
            raise ValueError(f"No samples of {name} inside [{settings['start']}, {settings['end']})")
    progress(f"ABORTS: {len(events)} in interval"
             + (f" for categories {settings['aborts']['categories']}" if settings["aborts"]["categories"] else ""))

    figure, counts = render_overview(settings, current_data, pv_data, events)
    folder = PlotFolder(root, settings["name"], data)
    figure.savefig(folder.path / "overview.png", dpi=settings["dpi"], bbox_inches="tight", facecolor="white")
    folder.write_json("metadata.json", {
        "plot": "pv_overview", "config": {"original_path": str(path), "copy": "config.toml"},
        "resolved": settings, "catalog": {**file_fingerprint(settings["aborts"]["filepath"]), "sha256": catalog_sha256},
        "currents": {ring: _series_report(d) for ring, d in current_data.items()},
        "pvs": [_series_report(d) for d in pv_data.values()],
        "events": [{key: event[key] for key in ("event_id", "timestamp_ns", "category", "ring", "critical_trigger",
                                                "source_index", "pressure_burst", "pressure_burst_locations")}
                   | {"timestamp_utc": utc(event["timestamp_ns"]).isoformat()} for event in events],
        "markers": counts,
        "summary_binning": "pandas resample(summary_period) of raw samples, default origin; empty bins are gaps",
        "versions": versions(), "code_sha256": code_sha256(), "outputs": ["overview.png"],
    })
    progress(f"PLOT: {folder.path / 'overview.png'}")
    return folder.path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot beam currents and arbitrary PVs with aborts from a TOML.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    try:
        plot_overview(args.config)
    except (ValueError, TypeError, KeyError, OSError) as error:
        print(f"Plot failed: {error}", file=sys.stderr)
        return 1
    return 0
