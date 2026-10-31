"""Contract tests for the TOML-driven plots (overview, dataset windows, crosscheck)."""

import json
import shutil

import h5py
import numpy as np
import pandas as pd
import pytest

from plotting.aborts import load_events, pressure_burst_locations
from plotting.archive import file_time_range_ns, load_pv, overlapping_files
from plotting.crosscheck import crosscheck_window
from plotting.dataset_windows import plot_dataset_windows
from plotting.overview import plot_overview, summarize
from plotting_helpers import END, MIDNIGHT, PVS, START, overview_toml, write_pv_file

SECOND = 1_000_000_000


# ---------------------------------------------------------------- catalogue and archive

@pytest.mark.parametrize("value, expected", [
    (",D07_L09,abtkicker(1V)/injkicker(3H)/taper/GV(D07_L04)", ("D07_L09",)),
    ("D08_H23,taper/abtwindow&kicker,D08_H23,D02_L03A", ("D08_H23", "D02_L03A")),
    ("GV(D02_L03)", ()), ("", ()), (None, ()), (3.0, ()),
])
def test_pressure_burst_locations(value, expected):
    assert pressure_burst_locations(value) == expected


def test_events_window_categories_and_timezone(archive):
    root, _ = archive
    events, digest = load_events(root / "aborts.json", START.value, END.value)
    assert [e["category"] for e in events] == ["Manual", "SBL", "BeamLoss"]      # 2026-01-03 excluded
    assert events[1]["timestamp_ns"] == (MIDNIGHT + pd.Timedelta(minutes=10)).value  # naive = Asia/Tokyo
    assert events[1]["pressure_burst_locations"] == ["D01_L01"] and len(digest) == 64
    selected, _ = load_events(root / "aborts.json", START.value, END.value, categories=["sbl", "BEAMLOSS"])
    assert [e["category"] for e in selected] == ["SBL", "BeamLoss"]


def test_file_preselection_uses_parquet_statistics(archive, tmp_path):
    root, _ = archive
    files = sorted((root / "archive" / "a").glob("*.parquet"))
    first, last = file_time_range_ns(files[1])
    assert first == MIDNIGHT.value and last == END.value - SECOND
    inside_day2 = (MIDNIGHT + pd.Timedelta(minutes=5)).value
    assert overlapping_files(files, inside_day2, inside_day2 + SECOND) == [files[1]]
    assert overlapping_files(files, MIDNIGHT.value - SECOND, MIDNIGHT.value + SECOND) == files
    # A file without statistics is always read (conservative).
    blind = tmp_path / "blind.parquet"
    write_pv_file(blind, PVS["a"], [START.value], [1.0], statistics=False)
    assert file_time_range_ns(blind) is None and overlapping_files([blind], END.value, END.value + 1) == [blind]


def test_load_pv_is_half_open_and_checks_identity(archive):
    root, _ = archive
    start, end = MIDNIGHT.value - 5 * SECOND, MIDNIGHT.value + 5 * SECOND
    data = load_pv(PVS["a"], [str(root / "archive" / "a")], root, start, end)
    np.testing.assert_array_equal(data.times_ns, np.arange(start, end, SECOND))
    assert len(data.files) == 2
    with pytest.raises(ValueError, match="PV mismatch"):
        load_pv(PVS["b"], [str(root / "archive" / "a")], root, start, end)


def test_summarize_bins_and_gaps():
    base = MIDNIGHT.value
    times = np.array([0, 1, 2, 25, 25], dtype=np.int64) * SECOND + base
    summary = summarize(times, np.array([1., 3., 2., 7., 9.]), "10s")
    assert summary["count"].tolist() == [3, 0, 1]             # duplicate timestamp keeps the last value
    assert summary.iloc[0][["min", "median", "max"]].tolist() == [1., 2., 3.]
    assert np.isnan(summary.iloc[1]["median"]) and summary.iloc[2]["median"] == 9.


# ---------------------------------------------------------------- overview

def test_overview_folder_is_reproducible(archive, tmp_path):
    root, _ = archive
    config = overview_toml(tmp_path, root, name="ov_all")
    folder = plot_overview(config, progress=None)
    assert folder == tmp_path / "PVPlots" / "ov_all"
    assert {p.name for p in folder.iterdir()} == {"config.toml", "overview.png", "metadata.json"}
    assert (folder / "config.toml").read_bytes() == config.read_bytes()
    meta = json.loads((folder / "metadata.json").read_text())
    assert [e["category"] for e in meta["events"]] == ["Manual", "SBL", "BeamLoss"]
    # SBL names D01_L01 (PV a); BeamLoss names D02_H05 (PV b). GV(D02_H05) is not a location.
    assert meta["markers"] == {"burst_references": 2, "burst_markers_drawn": 2}
    assert [p["samples"] for p in meta["pvs"]] == [7200, 7200]
    assert {ring: c["pv"] for ring, c in meta["currents"].items()} == {"LER": PVS["ler"], "HER": PVS["her"]}
    assert "plotting/overview.py" in meta["code_sha256"] and "dataset/read_pv.py" in meta["code_sha256"]
    with pytest.raises(FileExistsError):
        plot_overview(config, progress=None)


def test_overview_category_selection_and_single_panel(archive, tmp_path):
    root, _ = archive
    folder = plot_overview(overview_toml(tmp_path, root, name="sbl", categories='["sbl"]', pvs=("b",),
                                         currents=False), progress=None)
    meta = json.loads((folder / "metadata.json").read_text())
    assert [e["category"] for e in meta["events"]] == ["SBL"]
    assert meta["markers"] == {"burst_references": 1, "burst_markers_drawn": 0}   # D01_L01 not plotted
    assert meta["currents"] == {}


@pytest.mark.parametrize("changes, error, message", [
    (dict(units={"a": "Pa", "b": "unknown"}), ValueError, "share one unit"),
    (dict(extra_plot="colour = 'red'"), ValueError, "Unknown \\[plot\\]"),
    (dict(start="2026-01-01T23:00:00"), ValueError, "timezone"),
    (dict(end="2026-01-01T22:00:00Z"), ValueError, "later"),
    (dict(start="2025-01-01T00:00:00Z", end="2025-01-02T00:00:00Z"), ValueError, "No samples"),
    (dict(name="../escape"), ValueError, "folder name"),
])
def test_overview_rejects_bad_configs(archive, tmp_path, changes, error, message):
    root, _ = archive
    with pytest.raises(error, match=message):
        plot_overview(overview_toml(tmp_path, root, **changes), progress=None)
    assert not (tmp_path / "PVPlots").exists()


# ---------------------------------------------------------------- dataset windows

def _windows_toml(tmp_path, dataset, select, *, split="test", values="stored", name="win"):
    path = tmp_path / f"{name}.toml"
    path.write_text(f'[plot]\nname = "{name}"\ndpi = 50\n\n[dataset]\npath = "{dataset}"\nsplit = "{split}"\n'
                    f'values = "{values}"\n\n[select]\n{select}\n', encoding="utf-8")
    return path


def test_dataset_windows_select_by_label(archive, tmp_path):
    _, datasets = archive
    folder = plot_dataset_windows(_windows_toml(tmp_path, datasets["test_raw"], 'label = "pre_abort_SBL"\nlimit = 2'),
                                  progress=None)
    meta = json.loads((folder / "metadata.json").read_text())
    assert len(meta["windows"]) == 2 and all("pre_abort_SBL" in w["labels"] for w in meta["windows"])
    assert sorted(meta["outputs"]) == sorted(p.name for p in folder.glob("window_*.png"))
    assert (folder / "config.toml").exists()


def test_dataset_windows_raw_inversion_matches_stored_statistics(archive, tmp_path):
    _, datasets = archive
    folder = plot_dataset_windows(_windows_toml(tmp_path, datasets["train_zscore"], "indices = [0]",
                                                split="train", values="raw"), progress=None)
    meta = json.loads((folder / "metadata.json").read_text())
    with h5py.File(datasets["train_zscore"]) as h5:
        stored = h5["train/values"][0]
        norm = json.loads(h5["normalization_json"].asstr()[()])
    raw = stored * np.asarray(norm["scale"])[:, None] + np.asarray(norm["offset"])[:, None]
    assert raw.min() > 0 and meta["values_shown"].startswith("raw")      # pressure-like values restored
    with pytest.raises(IndexError):
        plot_dataset_windows(_windows_toml(tmp_path, datasets["train_zscore"], "indices = [100000]",
                                           split="train", name="bad"), progress=None)


# ---------------------------------------------------------------- crosscheck

def _crosscheck_toml(tmp_path, dataset, *, split, index, name, prefix_map=None):
    path = tmp_path / f"{name}.toml"
    text = f'[check]\nname = "{name}"\ndpi = 50\n\n[dataset]\npath = "{dataset}"\nsplit = "{split}"\nindex = {index}\n'
    if prefix_map:
        text += f'\n[sources]\npath_prefix_map = [["{prefix_map[0]}", "{prefix_map[1]}"]]\n'
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("key, split, index", [("train_zscore", "train", 0), ("train_zscore", "val", 5),
                                               ("test_raw", "test", 40), ("train_log", "train", 7)])
def test_crosscheck_passes_exactly(archive, tmp_path, key, split, index):
    _, datasets = archive
    folder, passed = crosscheck_window(_crosscheck_toml(tmp_path, datasets[key], split=split, index=index,
                                                        name="cc"), progress=None)
    comparison = json.loads((folder / "comparison.json").read_text())
    assert passed and all(c["exact_equal"] for c in comparison["channels"])
    assert {p.name for p in folder.iterdir()} == {"config.toml", "crosscheck.png", "comparison.json", "metadata.json"}


def test_crosscheck_detects_tampering_and_moved_archive(archive, tmp_path):
    root, datasets = archive
    tampered = tmp_path / "tampered.h5"
    shutil.copy(datasets["train_zscore"], tampered)
    with h5py.File(tampered, "r+") as h5:
        h5["train/values"][3, 1, 7] += 1e-9
    _, passed = crosscheck_window(_crosscheck_toml(tmp_path, tampered, split="train", index=3, name="t"),
                                  progress=None)
    assert not passed

    moved = tmp_path / "moved"
    shutil.copytree(root / "archive", moved / "archive")
    copy = tmp_path / "copy.h5"
    shutil.copy(datasets["train_zscore"], copy)
    with h5py.File(copy, "r+") as h5:       # pretend the build was made in another folder
        text = h5["build_metadata_json"].asstr()[()].replace(str(root), "/old/location")
        del h5["build_metadata_json"]
        h5.create_dataset("build_metadata_json", data=text, dtype=h5py.string_dtype("utf-8"))
    with pytest.raises(FileNotFoundError, match="path_prefix_map"):
        crosscheck_window(_crosscheck_toml(tmp_path, copy, split="train", index=3, name="m1"), progress=None)
    _, passed = crosscheck_window(_crosscheck_toml(tmp_path, copy, split="train", index=3, name="m2",
                                                   prefix_map=("/old/location", str(moved))), progress=None)
    assert passed


def test_log_build_windows_invert_to_archive_values(archive, tmp_path):
    root, datasets = archive
    folder = plot_dataset_windows(_windows_toml(tmp_path, datasets["train_log"], "indices = [3]",
                                                split="train", values="raw", name="logwin"), progress=None)
    meta = json.loads((folder / "metadata.json").read_text())
    assert meta["value_transform"] == "log10" and "zscore + log10" in meta["values_shown"]
    with h5py.File(datasets["train_log"]) as h5:
        assert h5.attrs["value_transform"] == "log10"
        stored, start = h5["train/values"][3], int(h5["train/window_start_ns"][3])
        norm = json.loads(h5["normalization_json"].asstr()[()])
    raw = 10.0 ** (stored * np.asarray(norm["scale"])[:, None] + np.asarray(norm["offset"])[:, None])
    from plotting_helpers import signal
    seconds = (start + np.arange(stored.shape[1]) * SECOND) / 1e9
    expected = np.stack([signal("a", seconds), signal("b", seconds)])
    np.testing.assert_allclose(raw, expected, rtol=1e-12)
