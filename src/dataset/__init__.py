"""Belle II PV datasets: aligned windows, labels and train-only normalization."""

__version__ = "0.1.0"


def build_dataset(config_path, *, progress=print):
    """Build a dataset from TOML; return the completed HDF5 file's Path."""
    from .builder import build_dataset as build

    return build(config_path, progress=progress)


__all__ = ["build_dataset", "__version__"]
