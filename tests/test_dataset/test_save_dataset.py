"""Small checks for the dataset writer's constructor and file lifecycle."""

from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

from dataset.save_dataset import DatasetWriter


def test_file_layout_and_exact_config_copy():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        config = root / "input.toml"
        original = '# Preserve comments and accents: café\r\n[dataset]\r\nmode = "test"\r\n'.encode("utf-8")

        for mode, splits in (("train_val", ("train", "val")), ("test", ("test",))):
            config_bytes = original.replace(b'"test"', f'"{mode}"'.encode())
            config.write_bytes(config_bytes)
            output = root / mode
            channels = ["signal_b", "signal_a"]
            with DatasetWriter(
                output, config, mode=mode, channel_ids=channels,
                window_size=np.int64(3), step_ns=1_000_000_000,
            ) as writer:
                channels.reverse()  # Caller mutation must not reorder the file.
                assert writer.channel_ids == ("signal_b", "signal_a")
                assert writer.counts == {split: 0 for split in splits}
                assert set(writer.datasets) == set(splits)
                assert writer.file.id.valid

            assert not writer.file.id.valid
            writer.close()  # Closing a second time is harmless.
            assert {path.name for path in output.iterdir()} == {"dataset.h5", "config.toml"}
            assert (output / "config.toml").read_bytes() == config_bytes

            with h5py.File(output / "dataset.h5", "r") as saved:
                assert set(saved) == {"channel_ids", *splits}
                assert saved["channel_ids"].asstr()[:].tolist() == ["signal_b", "signal_a"]
                assert saved.attrs["mode"] == mode
                assert saved.attrs["step_ns"] == 1_000_000_000
                assert saved.attrs["window_size"] == 3
                assert saved.attrs["time_reference"] == "Unix UTC nanoseconds"
                assert not saved.attrs["complete"]
                for split in splits:
                    values = saved[split]["values"]
                    assert values.shape == (0, 2, 3)
                    assert values.maxshape == (None, 2, 3)
                    assert values.dtype == np.dtype("float64")
                    assert values.chunks is not None
                    for name in ("window_start_ns", "window_end_ns"):
                        times = saved[split][name]
                        assert times.shape == (0,)
                        assert times.maxshape == (None,)
                        assert times.dtype == np.dtype("int64")


def test_validation_no_overwrite_and_exception_cleanup():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        config = root / "input.toml"
        config.write_text('[dataset]\nmode = "test"\n', encoding="utf-8")
        output = root / "build"
        arguments = dict(mode="test", channel_ids=["signal"], window_size=3, step_ns=1)
        for replacement, error in (
            ({"mode": "other"}, ValueError),
            ({"channel_ids": "signal"}, TypeError),
            ({"channel_ids": []}, ValueError),
            ({"channel_ids": ["a", "a"]}, ValueError),
            ({"channel_ids": [""]}, ValueError),
            ({"window_size": True}, TypeError),
            ({"window_size": 0}, ValueError),
            ({"step_ns": 1.5}, TypeError),
            ({"step_ns": -1}, ValueError),
        ):
            with np.testing.assert_raises(error):
                DatasetWriter(output, config, **(arguments | replacement))
            assert not output.exists()

        with np.testing.assert_raises(FileNotFoundError):
            DatasetWriter(output, root / "missing.toml", **arguments)
        assert not output.exists()

        with np.testing.assert_raises_regex(RuntimeError, "synthetic failure"):
            with DatasetWriter(output, config, **arguments) as writer:
                raise RuntimeError("synthetic failure")
        assert not writer.file.id.valid
        before = (output / "dataset.h5").read_bytes()
        with np.testing.assert_raises(FileExistsError):
            DatasetWriter(output, config, **arguments)
        assert (output / "dataset.h5").read_bytes() == before
        with np.testing.assert_raises(RuntimeError):
            with writer:
                pass


if __name__ == "__main__":
    test_file_layout_and_exact_config_copy()
    test_validation_no_overwrite_and_exception_cleanup()
    print("PASS: writer layout, exact TOML copy, validation, no overwrite, and cleanup")
