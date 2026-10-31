"""dataset.log_transform: log10 before sampling, statistics and normalization."""

import json

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dataset import build_dataset
from dataset.filter_pv import training_coverage_mask
from dataset.normalize import log10_values
from dataset.parse_config import resolve_config


ORIGIN_MS = 1_767_225_600_000           # 2026-01-01T00:00:00Z
N = 600                                 # 10 minutes at 1 Hz
WINDOW, STRIDE = 20, 10
BAD = (305, 312)                        # zero and negative measurement (seconds)


def signal():
    """Spans six orders of magnitude, 1e-12 .. 1e-6, plus two non-positive values."""
    t = np.arange(N, dtype=np.float64)
    values = 10.0 ** (-9 + 3 * np.sin(2 * np.pi * t / 200))
    values[BAD[0]], values[BAD[1]] = 0.0, -1e-9
    return values


def write(path, pv, values):
    pq.write_table(pa.table({
        "pv": [pv], "t0": pa.array([ORIGIN_MS], type=pa.int64()),
        "times": pa.array([np.arange(values.size, dtype=np.float64).tolist()], type=pa.list_(pa.float64())),
        "values": pa.array([values.tolist()], type=pa.list_(pa.float64())),
    }), path)


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    root = tmp_path_factory.mktemp("log_archive")
    write(root / "p.parquet", "LOG:P", signal())
    write(root / "ler.parquet", "LOG:LER", np.full(N, 500.0))
    write(root / "her.parquet", "LOG:HER", np.full(N, 500.0))
    (root / "aborts.json").write_text("[]")
    return root


def config(root, name, *, log, norm="zscore", mode="train_val", statistics=None, end_time=None):
    extra = "train_fraction = 0.5\n" if mode == "train_val" else "pre_abort_seconds = 0.0\npost_abort_seconds = 0.0\n"
    if log is not None:
        extra += f"log_transform = {str(log).lower()}\n"
    if statistics:
        extra += f'statistics_filepath = "{statistics}"\n'
    start, end = ("00:00:00", end_time or "00:10:00") if mode == "train_val" else ("00:05:00", "00:10:00")
    path = root / f"{name}.toml"
    path.write_text(f"""
[dataset]
mode = "{mode}"
start = "2026-01-01T{start}Z"
end = "2026-01-01T{end}Z"
pvs = ["p"]
sample_period_seconds = 1.0
window_size_samples = {WINDOW}
stride_samples = {STRIDE}
ler_pv = "ler"
her_pv = "her"
ler_threshold = 400.0
her_threshold = 400.0
aborts_filepath = "aborts.json"
exclude_before_seconds = 0.0
exclude_after_seconds = 0.0
norm_mode = "{norm}"
batch_size = 7
output_dir = "{name}"
{extra}
[[pvs]]
id = "p"
expected_pv = "LOG:P"
filepaths = ["p.parquet"]
max_age_seconds = 0.0

[[pvs]]
id = "ler"
expected_pv = "LOG:LER"
filepaths = ["ler.parquet"]
max_age_seconds = 0.0

[[pvs]]
id = "her"
expected_pv = "LOG:HER"
filepaths = ["her.parquet"]
max_age_seconds = 0.0
""", encoding="utf-8")
    return path


def read(path, split="train"):
    with h5py.File(path) as data:
        assert data.attrs["complete"]
        metadata = json.loads(data["build_metadata_json"].asstr()[()])
        return {"values": data[f"{split}/values"][:, 0, :], "starts": data[f"{split}/window_start_ns"][:],
                "norm": json.loads(data["normalization_json"].asstr()[()]),
                "transform": str(data.attrs["value_transform"]), "metadata": metadata}


def expected_starts(first, last, *, exclude_bad):
    """Grid-index window starts in [first, last) for one split, optionally skipping BAD samples."""
    starts = np.arange(first, last - WINDOW + 1, STRIDE)
    if exclude_bad:
        starts = np.array([s for s in starts if not any(s <= bad < s + WINDOW for bad in BAD)])
    return starts


def test_log10_values_marks_non_positive():
    result, count = log10_values(np.array([1e-7, 0.0, -2.0, np.nan, np.inf, 100.0]))
    assert count == 2
    np.testing.assert_array_equal(result[[0, 5]], [-7.0, 2.0])
    assert np.isnan(result[1:4]).all() and result[4] == np.inf


def test_default_is_unchanged_and_option_is_boolean(archive):
    resolved = resolve_config(config(archive, "default_cfg", log=None))
    assert resolved["dataset"]["log_transform"] is False
    path = config(archive, "bad_cfg", log=None)
    path.write_text(path.read_text().replace('norm_mode = "zscore"', 'norm_mode = "zscore"\nlog_transform = 1'))
    with pytest.raises(ValueError, match="log_transform must be true or false"):
        resolve_config(path)


def test_raw_log_build_stores_exact_log10_and_rejects_non_positive(archive):
    plain = read(build_dataset(config(archive, "plain_none", log=False, norm="none"), progress=None))
    logged = read(build_dataset(config(archive, "log_none", log=True, norm="none"), progress=None))
    assert (plain["transform"], logged["transform"]) == ("none", "log10")
    raw = signal()
    # Without the transform the non-positive samples are ordinary finite values.
    np.testing.assert_array_equal((plain["starts"] - plain["starts"][0]) // 10**9, expected_starts(0, 300, exclude_bad=False))
    starts = expected_starts(0, 300, exclude_bad=True)          # BAD lie in val (>= 300)
    np.testing.assert_array_equal((logged["starts"] // 10**9) - ORIGIN_MS // 1000, starts)
    np.testing.assert_array_equal(logged["values"], np.log10(np.stack([raw[s:s + WINDOW] for s in starts])))
    val = read(build_dataset(config(archive, "log_none_b", log=True, norm="none"), progress=None), "val")
    val_starts = (val["starts"] // 10**9) - ORIGIN_MS // 1000
    np.testing.assert_array_equal(val_starts, expected_starts(300, 600, exclude_bad=True))
    sources = logged["metadata"]["provenance"]["sources"]
    assert sources["p"]["nonpositive_values_invalidated"] == 2 and sources["p"]["value_transform"] == "log10"
    assert "value_transform" not in sources["ler"]                 # currents stay in mA


def test_zscore_statistics_are_those_of_log_values(archive):
    built = read(build_dataset(config(archive, "log_z", log=True), progress=None))
    starts = expected_starts(0, 300, exclude_bad=True)
    covered = np.log10(signal()[:300])[training_coverage_mask(starts, WINDOW, 300)]
    mean, std = covered.mean(), covered.std()
    np.testing.assert_allclose(built["norm"]["offset"], [mean], rtol=1e-12)
    np.testing.assert_allclose(built["norm"]["scale"], [std], rtol=1e-12)
    windows = np.log10(np.stack([signal()[s:s + WINDOW] for s in starts]))
    np.testing.assert_allclose(built["values"], (windows - mean) / std, rtol=1e-12, atol=1e-12)
    # Distinguishable across magnitudes: window means cover several standard deviations.
    assert np.ptp(built["values"].mean(axis=1)) > 2.0


def test_test_build_requires_matching_transform(archive):
    # Statistics source covers 00:00-00:05 only; the test build covers 00:05-00:10.
    source = build_dataset(config(archive, "log_z_src", log=True, end_time="00:05:00"), progress=None)
    ok = build_dataset(config(archive, "log_test", log=True, mode="test", statistics=source), progress=None)
    assert read(ok, "test")["transform"] == "log10"
    with pytest.raises(ValueError, match="value_transform"):
        build_dataset(config(archive, "plain_test", log=False, mode="test", statistics=source), progress=None)
