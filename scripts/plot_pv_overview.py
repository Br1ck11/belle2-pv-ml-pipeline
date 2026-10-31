"""Plot beam currents (top) and arbitrary PVs (bottom) with aborts.

Usage (from Project/): python scripts/plot_pv_overview.py --config <toml>
All settings come from the TOML; see src/plotting/examples/ and src/plotting/README.md.
"""

from plotting.overview import main


if __name__ == "__main__":
    raise SystemExit(main())
