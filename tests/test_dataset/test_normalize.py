"""Small synthetic checks for streamed, training-only normalization."""

import json

import numpy as np

from dataset.filter_pv import training_coverage_mask
from dataset.normalize import NormalizeWindows


def test_batches_match_numpy_and_constant_channel_metadata():
    # Unequal batches catch the error of averaging batch means equally.
    points = np.array([[1., 2., 4., 8., 16.], [5., 5., 5., 5., 5.]])
    points.flags.writeable = False
    windows = np.stack([points[:, :3], points[:, 2:]])
    for mode in ("minmax", "zscore"):
        normalizer = NormalizeWindows(mode, n_channels=2)
        normalizer.calculate_statistics(points[:, :1])
        normalizer.calculate_statistics(points[:, 1:1])  # Empty batch: no change.
        normalizer.calculate_statistics(points[:, 1:])
        assert normalizer.count == 5
        assert not normalizer.finalized
        normalizer.finalize()
        normalizer.finalize()  # Safe to repeat; no statistics are changed.

        if mode == "zscore":
            offset = points.mean(axis=1)
            raw_scale = points.std(axis=1, ddof=0)
            np.testing.assert_allclose(normalizer.mean, offset)
            np.testing.assert_allclose(
                normalizer.m2, np.sum((points - offset[:, None]) ** 2, axis=1)
            )
            assert normalizer.minimum is None and normalizer.maximum is None
        else:
            offset = points.min(axis=1)
            raw_scale = np.ptp(points, axis=1)
            np.testing.assert_array_equal(normalizer.minimum, offset)
            np.testing.assert_array_equal(normalizer.maximum, points.max(axis=1))
            assert normalizer.mean is None and normalizer.m2 is None

        scale = np.where(raw_scale == 0, 1., raw_scale)
        expected = (windows - offset[None, :, None]) / scale[None, :, None]
        actual = normalizer.apply_normalization(windows)
        np.testing.assert_allclose(actual, expected)
        assert actual.dtype == np.float64 and actual.shape == windows.shape
        assert not np.shares_memory(actual, windows)
        np.testing.assert_array_equal(actual[:, 1], np.zeros((2, 3)))

        # JSON round-trip verifies the exported fields can actually be saved.
        metadata = json.loads(json.dumps(normalizer.to_metadata(), allow_nan=False))
        assert metadata["norm_mode"] == mode
        assert metadata["fitted_on"] == "train" and metadata["count"] == 5
        assert metadata["constant_channel_policy"] == "scale_one"
        assert metadata["raw_scale"][1] == 0.
        assert metadata["scale"][1] == 1.
        assert metadata["scale_fallback_mask"] == [False, True]
        assert metadata["variance_ddof"] == (0 if mode == "zscore" else None)

        # Applying to later data must not refit, clip, or overwrite its input.
        later = np.array([[[31., 46.], [7., 3.]]])
        original = later.copy()
        normalized_later = normalizer.apply_normalization(later)
        np.testing.assert_array_equal(normalized_later[0, 1], [2., -2.])
        if mode == "minmax":
            np.testing.assert_allclose(normalized_later[0, 0], [2., 3.])
        np.testing.assert_array_equal(later, original)
        assert normalizer.to_metadata() == metadata
        metadata["scale"][0] = -1.
        assert normalizer.scale[0] > 0  # Export does not alias fitted arrays.


def test_unique_training_points_match_direct_statistics():
    starts = np.array([0, 2, 8], dtype=np.int64)
    points = np.array([
        [0., 1., 3., 7., 11., 17., 1e9, 1e9, 41., 47., 59., 71.],
        [5.] * 12,
        [12., 11., 9., 8., 6., 4., -1e9, -1e9, 3., 1., -2., -5.],
    ])
    coverage = training_coverage_mask(starts, window_size=4, grid_size=12)
    windows = np.stack([points[:, start:start + 4] for start in starts])
    # Direct statistics over the 10 UNIQUE covered grid points (indices 0-5
    # and 8-11). Replaces the former abort_windows TrainingStatistics reference.
    unique = points[:, coverage]
    assert unique.shape == (3, 10)
    direct = {
        "minmax": (unique.min(axis=1), np.ptp(unique, axis=1)),
        "zscore": (unique.mean(axis=1), unique.std(axis=1)),   # ddof=0
    }
    for mode in ("minmax", "zscore"):
        normalizer = NormalizeWindows(mode, n_channels=3)
        # Disjoint slices of the original grid, each with the global coverage
        # mask, prevent counting overlap twice across batch boundaries.
        for start, stop in ((0, 3), (3, 7), (7, 12)):
            normalizer.calculate_statistics(points[:, start:stop][:, coverage[start:stop]])
        normalizer.finalize()
        offset, raw_scale = direct[mode]
        scale = np.where(raw_scale == 0, 1.0, raw_scale)   # constant channel -> scale 1
        assert raw_scale[1] == 0 and normalizer.scale_fallback_mask.tolist() == [False, True, False]
        assert normalizer.count == 10  # 12 occurrences in windows, 10 unique points.
        np.testing.assert_allclose(normalizer.offset, offset, rtol=1e-12)
        np.testing.assert_allclose(normalizer.scale, scale, rtol=1e-12)
        np.testing.assert_allclose(
            normalizer.apply_normalization(windows),
            (windows - offset[None, :, None]) / scale[None, :, None], rtol=1e-12, atol=1e-12,
        )


def test_welford_single_points_large_offset_and_broadcasting():
    points = 1e9 + np.array([
        [1., 2., 3., 4., 5., 6., 7., 8.],
        [8., 6., 4., 2., 0., -2., -4., -6.],
        [-3., 0., 1., 4., 5., 8., 9., 12.],
    ])
    normalizer = NormalizeWindows("zscore", n_channels=3)
    for index in range(points.shape[1]):
        normalizer.streaming_Welford(points[:, index:index + 1])
    normalizer.finalize()
    np.testing.assert_allclose(normalizer.mean, points.mean(axis=1), rtol=0, atol=1e-7)
    np.testing.assert_allclose(normalizer.raw_scale, points.std(axis=1), rtol=1e-7)
    # N=2, C=3, T=4 catches accidental broadcasting along N or T.
    windows = np.stack([points[:, :4], points[:, 4:]])
    expected = (windows - points.mean(axis=1)[None, :, None]) / points.std(axis=1)[None, :, None]
    np.testing.assert_allclose(normalizer.apply_normalization(windows), expected, atol=1e-7)


def test_none_mode_and_empty_windows():
    normalizer = NormalizeWindows("none", n_channels=2)
    points = np.ones((2, 3), dtype=np.float32)
    normalizer.calculate_statistics(points)
    normalizer.finalize()
    assert normalizer.count == 0 and normalizer.mean is None
    windows = points[None, :, :]
    result = normalizer.apply_normalization(windows)
    np.testing.assert_array_equal(result, windows, strict=True)
    assert not np.shares_memory(result, windows)
    metadata = normalizer.to_metadata()
    assert metadata["raw_scale"] is None and metadata["fitted_on"] is None
    assert metadata["scale_fallback_mask"] == [False, False]
    json.dumps(metadata, allow_nan=False)

    for mode in ("none", "minmax", "zscore"):
        normalizer = NormalizeWindows(mode, n_channels=2)
        normalizer.calculate_statistics(np.array([[5], [7]], dtype=np.int64))
        normalizer.finalize()
        empty = np.empty((0, 2, 3), dtype=np.float32)
        assert normalizer.apply_normalization(empty).shape == (0, 2, 3)
        if mode != "none":
            np.testing.assert_array_equal(normalizer.scale_fallback_mask, [True, True])


def test_constant_decimal_channels_and_failed_update():
    # An ordinary mean over repeated 0.1 values can be slightly different
    # from 0.1, creating a spurious tiny variance unless centered first.
    points = np.repeat(np.array([[0.1], [-0.3]]), 1000, axis=1)
    normalizer = NormalizeWindows("zscore", n_channels=2)
    for start, stop in ((0, 3), (3, 20), (20, 1000)):
        normalizer.calculate_statistics(points[:, start:stop])
    normalizer.finalize()
    np.testing.assert_array_equal(normalizer.raw_scale, [0., 0.])
    np.testing.assert_array_equal(normalizer.scale_fallback_mask, [True, True])
    np.testing.assert_array_equal(
        normalizer.apply_normalization(points[None, :, :]), np.zeros((1, 2, 1000))
    )

    normalizer = NormalizeWindows("zscore", n_channels=1)
    normalizer.calculate_statistics(np.array([[1., 3.]]))
    with np.testing.assert_raises_regex(ValueError, "float64 range"):
        normalizer.calculate_statistics(np.array([[1e308, -1e308]]))
    assert normalizer.count == 2
    np.testing.assert_array_equal(normalizer.mean, [2.])
    np.testing.assert_array_equal(normalizer.m2, [2.])
    normalizer.finalize()
    np.testing.assert_array_equal(normalizer.raw_scale, [1.])


def test_input_validation_and_lifecycle():
    for mode in (None, "unknown", "", 0):
        with np.testing.assert_raises(ValueError):
            NormalizeWindows(mode, n_channels=2)
    for channels in (True, np.bool_(False), 2., None):
        with np.testing.assert_raises(TypeError):
            NormalizeWindows("zscore", n_channels=channels)
    for channels in (0, -1):
        with np.testing.assert_raises(ValueError):
            NormalizeWindows("zscore", n_channels=channels)

    for mode in ("minmax", "zscore", "none"):
        normalizer = NormalizeWindows(mode, n_channels=np.int64(2))
        with np.testing.assert_raises(TypeError):
            normalizer.calculate_statistics([[1., 2.], [3., 4.]])
        for batch in (
            np.ones((2,)), np.ones((3, 2)), np.ones((1, 2, 3)),
            np.ones((2, 2), dtype=bool), np.ones((2, 2), dtype=complex),
            np.array([["1"], ["2"]]), np.array([[np.nan], [1.]]),
            np.array([[1.], [np.inf]]),
        ):
            with np.testing.assert_raises(ValueError):
                normalizer.calculate_statistics(batch)
        assert normalizer.count == 0
        if mode != "none":
            normalizer.calculate_statistics(np.empty((2, 0)))
            with np.testing.assert_raises(ValueError):
                normalizer.finalize()
            with np.testing.assert_raises(RuntimeError):
                normalizer.apply_normalization(np.ones((1, 2, 3)))
            with np.testing.assert_raises(RuntimeError):
                normalizer.to_metadata()
            normalizer.calculate_statistics(np.ones((2, 3)))
            normalizer.finalize()
            with np.testing.assert_raises(RuntimeError):
                normalizer.calculate_statistics(np.ones((2, 1)))
        for windows in (np.ones((2, 3)), np.ones((1, 3, 2)), np.ones((1, 2, 0)),
                        np.full((1, 2, 3), np.nan)):
            with np.testing.assert_raises(ValueError):
                normalizer.apply_normalization(windows)
    with np.testing.assert_raises(ValueError):
        NormalizeWindows("minmax", n_channels=2).streaming_Welford(np.ones((2, 3)))
    with np.testing.assert_raises(RuntimeError):
        normalizer = NormalizeWindows("zscore", n_channels=2)
        normalizer.calculate_statistics(np.ones((2, 3)))
        normalizer.finalize()
        normalizer.streaming_Welford(np.ones((2, 1)))


if __name__ == "__main__":
    test_batches_match_numpy_and_constant_channel_metadata()
    test_unique_training_points_match_reference()
    test_welford_single_points_large_offset_and_broadcasting()
    test_none_mode_and_empty_windows()
    test_constant_decimal_channels_and_failed_update()
    test_input_validation_and_lifecycle()
    print("PASS: streamed statistics, unique coverage, reference, fallback metadata, and validation")
