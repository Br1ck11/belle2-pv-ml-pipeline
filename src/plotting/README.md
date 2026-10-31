# belle2-plotting

TOML-driven plots built on the `src/dataset` machinery (import name
**plotting**). Each run writes a new folder `Project/PVPlots/<name>/`
(an existing name is refused) containing:

- `config.toml`: an exact copy of the TOML;
- the PNG(s);
- `metadata.json`: resolved settings with defaults, every input file read
  (path, size, mtime), the plotted abort events, the catalogue sha256,
  package versions, and sha256 of the plotting and `src/dataset` code.

All figures of the project (these plots, training `loss.png`, model tests)
use one style, `plotting.style.science_style()`: SciencePlots `science` with
all text typeset by LaTeX. This needs a TeX distribution with `latex` and
`dvipng` (MacTeX / TeX Live; a system program, not a pip package). It is
found on `PATH` or in a standard location such as `/Library/TeX/texbin`.
Without it, a warning is printed and the `no-latex` look is used.

| Script (`Project/scripts/`) | Module | Example TOML | Output |
| --- | --- | --- | --- |
| `plot_pv_overview.py` | `overview.py` | `examples/pv_overview.toml` | `overview.png` |
| `plot_dataset_windows.py` | `dataset_windows.py` | `examples/dataset_windows.toml` | `window_<split>_<row>.png` |
| `crosscheck_dataset_windows.py` | `crosscheck.py` | `examples/crosscheck.toml` | `crosscheck.png`, `comparison.json` |

Run from `Project/` with `python scripts/<script> --config <toml>`. This
needs `pip install -e ./src/dataset -e ./src/plotting`, or `PYTHONPATH=src`.

## Package folder (development version)

In a clone of the repository the package is the folder `src/plotting/`; with an
editable install (`pip install -e ./src/plotting`) changes there take effect
immediately. Three plot types each have one module and one example TOML; the
other modules are shared helpers. `train` and `evaluation` keep their own copy of the style function, so they
do not need this package.

| File | Task |
| --- | --- |
| `overview.py` | Beam currents (top) and any PVs (bottom) over a time range, with the aborts marked (`scripts/plot_pv_overview.py`). |
| `dataset_windows.py` | Plots windows stored in a `dataset.h5`, without reading the archive (`scripts/plot_dataset_windows.py`). |
| `crosscheck.py` | Rebuilds one stored window independently from the raw Parquet files and compares it with `dataset.h5` (`scripts/crosscheck_dataset_windows.py`). |
| `archive.py` | Reads one PV over a time range with the `src/dataset` reader. |
| `aborts.py` | Loads the abort catalogue through the dataset package's abort handling and adds the pressure-burst annotations. |
| `common.py` | Shared TOML checks and the reproducible `PVPlots/<name>/` output folder (config copy, PNGs, `metadata.json`). |
| `style.py` | The figure style (SciencePlots `science` with LaTeX text, `science_style()`; `train` and `evaluation` hold copies) and the shared colours and line styles for aborts, currents and PVs. |
| `examples/` | One example TOML per plot type: `pv_overview.toml`, `dataset_windows.toml`, `crosscheck.toml`. |
| `__init__.py` | Package description. |
| `pyproject.toml` | Package metadata and dependencies. |

## PV overview

The top panel shows LER/HER currents: the bin median and the within-bin
min–max band, plus a ring marker at each abort. The bottom panel shows any
PV(s) sharing one unit: the bin median and the dotted bin maximum. Log
scale is used automatically if all values are positive. Abort lines use the
exact catalogue timestamps, with a category-specific colour and style.

`[aborts].categories` selects categories case-insensitively (`[]` = all).
A gold marker is drawn for each (abort, `pressure_burst` location) pair
whose location is a `:`-separated part of a plotted PV name (`D07_L09` in
`VALCCG:D07_L09:PRES`). It sits at the latest finite raw value at or before
the abort.

Locations are complete comma-separated tokens matching
`D\d{2}_[HL][0-9A-Z]+`; `GV(D07_L04)` is not one.

Reading uses `dataset.parse_config.resolve_pv_filepaths` and
`dataset.read_pv.read_pv_files`. Files are preselected by their Parquet
statistics (`t0` + min/max of `times`), so only files overlapping
`[start, end)` are read; the interval is then applied to the measured
timestamps. The catalogue is parsed by `dataset.filter_pv.WindowFilter`
(naive timestamps = Asia/Tokyo).

## Dataset windows

Selects rows of one split either by `indices`, or by `label` / `event_id`
with a `limit` (`first` or seeded `random`). `values = "raw"` inverts the
saved normalization, and then log10 for `value_transform = "log10"` builds.
The inversion has float64 round-off.

## Crosscheck

Independent of `dataset.read_pv` / `dataset.time_grid`: it uses pyarrow and
`pandas.merge_asof`. It re-reads the source files recorded in the build and
checks them against the recorded byte sizes. If the archive moved, use
`[sources].path_prefix_map`.

It verifies the stored window exactly: the reference, plus log10 and
normalization when the build used them. It also checks completeness and the
current thresholds for `non_abort` windows. Exit codes: 0 pass, 2 mismatch,
1 error.

Tests: `tests/test_plotting/` (a synthetic two-day archive in daily files).
