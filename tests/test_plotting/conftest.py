"""Session fixtures for the plotting tests; helpers live in plotting_helpers.py."""

import pytest

from plotting_helpers import build, make_archive


@pytest.fixture(scope="session")
def archive(tmp_path_factory):
    """Read-only synthetic archive plus two dataset.h5 builds over it."""
    root = make_archive(tmp_path_factory.mktemp("plot_archive"))
    datasets = {
        "train_zscore": build(root, mode="train_val", norm_mode="zscore", output="train_zscore"),
        "test_raw": build(root, mode="test", norm_mode="none", output="test_raw"),
        "train_log": build(root, mode="train_val", norm_mode="zscore", output="train_log", log_transform=True),
    }
    return root, datasets
