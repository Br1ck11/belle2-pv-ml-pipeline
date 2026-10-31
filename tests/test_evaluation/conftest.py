"""Session fixtures: synthetic archive, train/test datasets, a trained CNN VAE and a trained forecaster."""

import pytest

from evaluation_helpers import (build, make_archive, register_forecaster, train_forecaster, train_vae,
                                unregister_forecaster)


@pytest.fixture(scope="session")
def setup(tmp_path_factory):
    root = make_archive(tmp_path_factory.mktemp("evaluation"))
    train = build(root, "train", mode="train_val", start="00:00:00", end="01:00:00")
    test = build(root, "test", mode="test", start="01:00:00", end="02:00:00", statistics=train)
    test_t24 = build(root, "test_t24", mode="test", start="01:00:00", end="02:00:00", window=24, statistics=train)
    checkpoint = train_vae(root, train)
    return {"root": root, "train": train, "test": test, "test_t24": test_t24, "checkpoint": checkpoint}


@pytest.fixture(scope="session")
def forecaster(setup):
    """A ForecastingModel (horizon 8 of T = 32), registered only while the tests run."""
    register_forecaster()
    try:
        yield train_forecaster(setup["root"], setup["train"])
    finally:
        unregister_forecaster()
