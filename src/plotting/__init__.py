"""TOML-driven plots of SuperKEKB PV data, built on the src/dataset machinery.

overview         beam currents (top) + arbitrary PVs (bottom) with aborts
dataset_windows  stored windows of a dataset.h5
crosscheck       independent archive verification of one dataset.h5 window
Every plot writes Project/PVPlots/<name>/ with config.toml, PNG(s), metadata.json.
"""

__version__ = "0.1.0"
