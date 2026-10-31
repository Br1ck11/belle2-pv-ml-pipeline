"""Verify one dataset.h5 window against an independent raw-archive read.

Usage (from Project/): python scripts/crosscheck_dataset_windows.py --config <toml>
All settings come from the TOML; see src/plotting/examples/ and src/plotting/README.md.
"""

from plotting.crosscheck import main


if __name__ == "__main__":
    raise SystemExit(main())
