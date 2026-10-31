"""Plot stored windows of a dataset.h5.

Usage (from Project/): python scripts/plot_dataset_windows.py --config <toml>
All settings come from the TOML; see src/plotting/examples/ and src/plotting/README.md.
"""

from plotting.dataset_windows import main


if __name__ == "__main__":
    raise SystemExit(main())
