# belle2-dataset

Turns SuperKEKB/Belle II PV Parquet files and the abort catalogue into one
`dataset.h5` per build. It writes either abort-free train/validation windows
or test windows that mix "normal operation" (defined by the applied filters) 
and abort-related windows, each with labels, exact UTC times and training-only normalization.
All settings come from one TOML file. The import name is `dataset`.
Fundamentally, what is stored is an object of shape (N, C, T), meaning we have
N window each consisting of C channels (= number of distinct PVs) and T data points of a PV.

## Install

Python ≥ 3.11 is required. Dependencies (numpy, pandas, pyarrow, h5py) are
installed automatically when installing the package as described below.

**From GitHub or GitLab.** The git repository root is `Project/`, so the package
lives in `src/dataset`.

```bash
REPO=https://github.com/<user>/<repo>.git        # GitLab: https://gitlab.<host>/<group>/<repo>.git
REF=v0.1.0                                       # tag, branch or commit; pin it for reproducible builds
pip install "belle2-dataset @ git+$REPO@$REF#subdirectory=src/dataset"
```

- **Private repository:** use SSH, `REPO=ssh://git@github.com/<user>/<repo>.git`
  (or `ssh://git@gitlab.<host>/<group>/<repo>.git`), with your SSH key
  registered on the server.
- **Development (editable) install from a clone:**
  `git clone $REPO Belle2Project && cd Belle2Project && python -m pip install -e ./src/dataset`.
- **Check the installation:** `belle2-build-dataset --help` and
  `belle2-dataset-demo --output /tmp/belle2_demo`. The demo builds a train/val
  and a test dataset from synthetic data. Its destination must not exist yet.

## Package folder (development version)

In a clone of the repository the package is the folder `src/dataset/`; with an
editable install (`pip install -e ./src/dataset`) changes there take effect
immediately. The build runs top to bottom through these modules: configuration,
reading and sampling the PVs on a common time grid, splitting, cutting windows,
filtering and labelling them, normalizing, and writing `dataset.h5`.

| File | Task |
| --- | --- |
| `builder.py` | Runs the whole build from one TOML (`build_dataset`, CLI `main`) and records provenance, so every step below is called from here. |
| `parse_config.py` | Loads and validates the TOML, resolves `project_root`-relative paths and expands `filepaths` (files or folders of `*.parquet`) into an explicit file list. |
| `read_pv.py` | Reads the daily Parquet files of one PV, checks that each file holds the expected PV identity, and returns its measurements with UTC timestamps. |
| `time_grid.py` | Converts archive times to UTC nanoseconds, builds the regular sampling grid, and assigns each grid point the latest measurement no older than `max_age_seconds`. |
| `split.py` | Splits the time grid into the train and validation parts (`train_fraction`) before any windows are cut; test builds keep the whole grid. |
| `window_data.py` | Cuts gridded series into windows of T samples with a stride of S samples. |
| `filter_pv.py` | Selects windows (`WindowFilter`): complete data, both beam currents above threshold, outside the abort exclusion intervals; in test mode it also assigns the pre-abort, abort and post-abort labels. |
| `normalize.py` | Fits the per-channel statistics on the training windows only (z-score or min-max) and applies them, or reuses the statistics of another build (`statistics_filepath`). |
| `save_dataset.py` | Writes `dataset.h5` (values, times, labels, statistics, metadata) and marks it complete only after everything is written. |
| `demo.py` | Builds a small synthetic archive and both dataset modes, as a self-contained example. |
| `dataset.toml` | Starter configuration with every setting explained; copy and fill it for a real build. |
| `__init__.py`, `__main__.py` | Package exports and `python -m dataset --config <toml>`. |
| `pyproject.toml` | Package metadata and dependencies for `pip install`. |

## Run
**You may need to adapt the path to where the .toml is saved when running the command below:**
```bash
belle2-build-dataset --config <build>.toml       # or: python -m dataset --config <build>.toml
```

From Python: `from dataset import build_dataset; path = build_dataset("build.toml")`.
It returns the path of `dataset.h5`; pass `progress=None` for no progress output. An
annotated example template is in `src/dataset/dataset.toml`.

## TOML format
The file contains exactly one `[dataset]` table and one or more `[[pvs]]`
tables.
- Unknown or missing keys are errors.
- `int` must be a TOML integer. `float` also accepts integers.
- Durations in seconds must be ≥ 0 and exact in integer nanoseconds (`0.1` is
  fine; `1e-10` is not).
- Paths are absolute, or relative to the set `project_root`.


### `[dataset]`

| Key | Format | Meaning |
| --- | --- | --- |
| `mode` | `"train_val"` or `"test"` | `train_val`: abort-free training and validation splits. `test`: normal windows plus abort-related windows. |
| `start`, `end` | string, ISO time with timezone, e.g. Tokyo time: `"2026-01-30T09:00:00+09:00"` or unix time: `"...Z"` | Requested interval `[start, end)`. `end` must be later than `start`. |
| `project_root` | string path, default: the folder the TOML is saved in | Base for all paths. A relative value is resolved from the TOML's own folder. |
| `pvs` | non-empty array of `[[pvs]]` ids, no repeats | Model channels, in order. This fixes the channel axis C of the stored windows. |
| `sample_period_seconds` | float > 0 | Spacing of the shared UTC time grid. |
| `window_size_samples` | int > 0 | T, the number of grid points per window. |
| `stride_samples` | int > 0 | Grid points between consecutive window starts. Windows overlap when the stride is smaller than T. |
| `train_fraction` | float in (0, 1), required for `train_val` | The first ⌊fraction × grid points⌋ go to train, the rest to val. The split is made before filtering. It does not impact the test dataset mode |
| `ler_pv`, `her_pv` | `[[pvs]]` ids, different from each other | Current PVs used for selection. Current PV data is loaded even if not in `pvs`, and are never stored or normalized unless listed as [[pvs]]. |
| `ler_threshold`, `her_threshold` | float, mA | Normal windows need both currents strictly above these values at every point, defining normal operation. |
| `aborts_filepath` | string path to a JSON file | Abort catalogue file. |
| `exclude_before_seconds`, `exclude_after_seconds` | seconds | A normal window is rejected if it overlaps `[abort − before, abort + after]` of any catalogue event. |
| `pre_abort_seconds`, `post_abort_seconds` | seconds, required for `test` | How far before and after an abort a test window is still associated with it and labelled as "pre-abort" or "post-abort". |
| `norm_mode` | `"zscore"`, `"minmax"` or `"none"` | Per-channel normalization: z-score (**using population standard deviation**, ddof = 0, as no underlying population mean is intended to be estimated, but the one of the training data ), min-max, or raw values. |
| `log_transform` | bool, default `false` | `true`: take log10 of the model-channel measurements before sampling, statistics and normalization. Values ≤ 0 become invalid. Currents are never transformed. A log transform can reorganize relative distances wihtin the data if it spans over several orders of magnitude (like the pressure PVs). |
| `statistics_filepath` | string path, required for `test` with `norm_mode` ≠ `"none"`, not allowed otherwise | A complete `train_val` `dataset.h5` whose statistics are reused. Its channels and `log_transform` must match, and its intervals must not overlap the test interval. |
| `batch_size` | int > 0 | Candidate windows processed at once. This limits memory only; it is not the ML minibatch size. |
| `output_dir` | string path that must not exist yet | Created by the build. It receives `dataset.h5` and an exact copy of the TOML. |

### `[[pvs]]` (one table per PV definition; unused definitions are allowed)

| Key | Format | Meaning |
| --- | --- | --- |
| `id` | string, unique | Alias used in `pvs`, `ler_pv` and `her_pv`. |
| `expected_pv` | string, e.g. `"VALCCG:D01_L01A:PRES"` | Must equal the `pv` field of every file read. |
| `filepaths` | non-empty array of strings | Files or directories. A directory to a PV's `*.parquet` files inclduding the parquet file to be loaded. Order is kept; duplicate files are rejected. |
| `max_age_seconds` | seconds | A grid point takes the latest measurement at or before it (we interpolate if there is a gap of missing data points of maximum max\_age\_seconds), if that measurement is at most this old. Otherwise a point is invalid and a window, *made up of C channels*, containing it will be filtered out. |

## Input formats

- **Parquet files:** exactly one row with
  - `pv` (string),
  - `t0` (integer, Unix milliseconds),
  - `times` (list of seconds relative to `t0`, finite and non-decreasing),
  - `values` (list of numbers, same length as `times`).

  NaN and ±inf values are kept and treated as invalid.
- **Abort catalogue:** a JSON array of objects with these fields:
  - `timestamp` (string, required): a time without a timezone is read as Asia/Tokyo;
  - `category`: missing or empty becomes `"Unspecified"`;
  - `ring`: `LER`, `HER` or `Both`, case-insensitive;
  - `critical_trigger`.

  Event IDs are hashes of time, ring, category and trigger. Duplicate
  identities are an error.

## What a build does

1. **Grid and split.** One shared grid is made over `[start, end)`. In
   `train_val` mode it is split in time *before* sampling, and each split only
   uses measurements from inside its own interval.
2. **Sampling.** Each PV's values are sampled onto the time grid.
3. **Completeness.** A window is rejected if any model channel or either
   current is invalid at any point. Incomplete trailing windows are dropped.
4. **Filtering / Selection.** `train_val` keeps only normal windows: both currents above
   threshold and no overlap with an exclusion interval. `test` keeps normal
   windows plus every complete abort-related window; abort-related windows
   bypass the current thresholds.
5. **Normalization.** Statistics are fitted on the training grid points covered
   by kept windows, each point counted once. They are then applied to all
   splits. A constant channel uses scale 1 (otherwise the standard deviation is zero and normalization can be invalid), and this is recorded.

### Labels

For an abort event of category `x` at time `e`, and a window whose first and last
sample times are `s` and `t`:
 
| Label | Condition |
| --- | --- |
| `x` | `s ≤ e ≤ t` (the window contains the abort) |
| `pre_abort_x` | `e − pre ≤ t < e` |
| `post_abort_x` | `s > e` and `t ≤ e + post` |
| `non_abort` | no association (normal window) |

A window can carry several labels, specifically if overlapping windows are used. Every associated event is kept with its ID
and relation. Labels describe the input window.
## Minimal example

```toml
[dataset]
project_root = "../.."                       # this file lives in src/dataset/
mode = "train_val"
start = "2026-01-30T09:00:00+09:00"
end = "2026-04-30T09:00:00+09:00"
pvs = ["pressure"]
sample_period_seconds = 1.0
window_size_samples = 120
stride_samples = 15
train_fraction = 0.7
ler_pv = "ler"
her_pv = "her"
ler_threshold = 400.0
her_threshold = 400.0
aborts_filepath = "data/2026_aborts.json"
exclude_before_seconds = 300.0
exclude_after_seconds = 120.0
norm_mode = "zscore"
log_transform = false
batch_size = 256
output_dir = "datasets/D01_L01A_zscore"

[[pvs]]
id = "pressure"
expected_pv = "VALCCG:D01_L01A:PRES"
filepaths = ["data/2026ab/MTS_parquet/converted_flat2/vacuum/arch/mts/ArchiverStore/VALCCG/D01_L01A"]
max_age_seconds = 2.0

[[pvs]]
id = "ler"
expected_pv = "BMLDCCT:CURRENT"
filepaths = ["data/2026ab/MTS_parquet/converted_flat2/injection/arch/mts/ArchiverStore/BMLDCCT"]
max_age_seconds = 2.0

[[pvs]]
id = "her"
expected_pv = "BMHDCCT:CURRENT"
filepaths = ["data/2026ab/MTS_parquet/converted_flat2/injection/arch/mts/ArchiverStore/BMHDCCT"]
max_age_seconds = 2.0
```

For a test build, change these settings: set `mode = "test"`, give
non-overlapping `start`/`end`, set `pre_abort_seconds` and
`post_abort_seconds`, and point `statistics_filepath` at the training
`dataset.h5`. `train_fraction` is then ignored.

## Output: `<output_dir>/{config.toml, dataset.h5}`

- **Root attributes:**
  - `complete`: always check it is `True` before use;
  - `schema_version`, `dataset_id`, `mode`;
  - `window_size` (T) and `step_ns` (= sampling interval of time grid);
  - `values_state`: `normalized` or `raw`;
  - `value_transform`: `none` or `log10`; older files without it are `none`;
  - `time_reference` = Unix UTC nanoseconds: time elapsed after UNIX origin at 1970-01-01 00:00:00 UTC;
  - `window_end_convention` = `last_included_sample`.
- **Root datasets (JSON stored as UTF-8 strings):**
  - `channel_ids`: the model channels, in order;
  - `normalization_json`: offset, scale and raw scale per channel;
  - `build_metadata_json`: resolved config, PV identities, split intervals,
    per-PV source files with size and mtime, selection counts, and code,
    config and catalogue hashes;
  - `events_json`: the events referenced by the windows.
- **Groups `train`, `val` (or `test`):**
  - `values`: float64 `(N, C, T)`; N windows, each containing all C channels with T data points per channel, so `values[i]` has shape `(C, T)`. "C" is the total number of PVs in `pvs`, one channel per PV.
  - `window_start_ns`, `window_end_ns`: int64 `(N,)`, one pair per window.
    Each is the full number of nanoseconds since `1970-01-01 00:00:00 UTC`;
    the end is the last included sample. Sample `k` in window `i` is at
    `window_start_ns[i] + k * step_ns`, for `0 <= k < T`. The shared
    `step_ns` is saved once as a root attribute. Original irregular
    source-readout timestamps are not saved per sample;
  - `labels_json` and `event_associations_json`: one JSON list per window.

```python
import json, h5py
with h5py.File(path, "r") as data:
    assert data.attrs["complete"]
    x = data["train/values"][:32]                                       # (32, C, T)
    labels = [json.loads(s) for s in data["train/labels_json"].asstr()[:32]]
```

Raw data for one PV must fit in RAM. Disk must hold temporary sampled arrays
plus the uncompressed float64 output. A failed build stays marked
`complete = False` and cannot be resumed.

Tests (from `Project/`): `python -m pytest -q tests/test_dataset`.

## Deferred Features
- `supervised_train_val` mode: Save labels for each window (e.g. abort of category "..." and "no\_abort") that can be used to train a classifier model. This is ideally done by saving mutliple labels and training a multi-class classifier that predicts each abort type of a given window.
A variant of this can be: `supervised_forecasting_train_val` mode: Save labels of abort categories and then define a model that predicts the abort category of the next window (this includes predicting the class "no\_abort" for "normal" operation)
