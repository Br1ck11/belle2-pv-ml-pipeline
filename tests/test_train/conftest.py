"""Session fixtures for the training tests; helpers live in train_helpers.py."""

from pathlib import Path

import pytest

from train_helpers import make_dataset


@pytest.fixture(scope="session")
def dataset_1ch(tmp_path_factory) -> Path:
    return make_dataset(tmp_path_factory.mktemp("synthetic_1ch"), n_channels=1)


@pytest.fixture(scope="session")
def dataset_2ch(tmp_path_factory) -> Path:
    return make_dataset(tmp_path_factory.mktemp("synthetic_2ch"), n_channels=2)
