"""Shared visual encoding (taken over from the former abort_windows renderer).

Every figure of this package is drawn inside ``science_style()``: the SciencePlots "science" style, with all
text typeset by LaTeX (text.usetex). This needs a TeX installation with
``latex`` and ``dvipng`` (e.g. MacTeX or TeX Live); LaTeX is a system
program, not a Python package. If neither is on PATH nor in a standard TeX
location, a warning is printed and the "no-latex" variant (matplotlib
mathtext, same look without LaTeX) is used instead.

train/style.py and evaluation/style.py hold copies of science_style() so
that those packages do not need this one; keep the three in sync.
"""

import glob
import os
import shutil
import warnings

import matplotlib
import matplotlib.dates as mdates
import matplotlib.style
import numpy as np


SCIENCE_STYLES = ("science",)                     # LaTeX text rendering
FALLBACK_STYLES = ("science", "no-latex")         # used only when LaTeX is unavailable
LATEX_PROGRAMS = ("latex", "dvipng")
# GUI-started processes on macOS often lack these on PATH.
STANDARD_TEX_DIRS = ("/Library/TeX/texbin", "/usr/local/texlive/*/bin/*", "/opt/homebrew/bin", "/usr/local/bin",
                     "/usr/bin")


def latex_available() -> bool:
    """True if latex and dvipng are runnable; adds a standard TeX bin folder to PATH if needed."""
    if all(shutil.which(program) for program in LATEX_PROGRAMS):
        return True
    for pattern in STANDARD_TEX_DIRS:
        for folder in sorted(glob.glob(pattern), reverse=True):     # newest TeX Live year first
            if all(os.access(os.path.join(folder, p), os.X_OK) for p in LATEX_PROGRAMS):
                os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
                return True
    return False


_warned = False


def science_style():
    """Context manager (also usable as a function decorator) applying the project style.

    Styles act on matplotlib's rcParams, which a Figure and its texts read
    when they are created and saved, so the whole figure is built inside it.
    """
    global _warned
    with warnings.catch_warnings():
        # SciencePlots 2.2 registers its styles through matplotlib helpers that
        # matplotlib 3.11 deprecates; harmless, so keep CLI output clean.
        warnings.simplefilter("ignore", matplotlib.MatplotlibDeprecationWarning)
        import scienceplots  # noqa: F401  (registers the SciencePlots styles)
    styles = SCIENCE_STYLES
    if not latex_available():
        styles = FALLBACK_STYLES
        if not _warned:
            warnings.warn("LaTeX (latex + dvipng) not found: figures use the SciencePlots 'no-latex' "
                          "fallback. Install a TeX distribution (e.g. MacTeX) for LaTeX text.", stacklevel=2)
            _warned = True
    return matplotlib.style.context(list(styles))


CATEGORY_STYLES = {
    "BeamLoss": ("#C05A22", "-"),
    "SBL": ("#D1495B", "--"),
    "Injection": ("#2A9D8F", ":"),
    "EQ": ("#7B2CBF", "-."),
    "RF": ("#6C757D", (0, (5, 2, 1, 2))),
    "Manual": ("#8C6D31", (0, (3, 2))),
    "MAG": ("#17A2B8", (0, (5, 3))),
    "Others": ("#7F7F7F", (0, (1, 2))),
    "Unspecified": ("#A0A0A0", (0, (1, 2))),
}
PV_COLORS = ("#3B5BA9", "#009E73", "#D55E00", "#7E57C2", "#CC79A7", "#8C6D31", "#2A9D8F", "#E69F00")
RING_MARKERS = {"LER": "v", "HER": "^", "Both": "D"}
CURRENT_STYLES = {"LER": ("#2459A6", "-"), "HER": ("#D55E00", "--")}
BURST_FACE = "#F4B400"


def abort_style(category):
    return CATEGORY_STYLES.get(category, ("#555555", "--"))


def draw_abort_lines(ax, events, *, alpha):
    """Exact catalogue timestamps, category-specific color and line style."""
    for event in events:
        color, linestyle = abort_style(event["category"])
        ax.axvline(np.datetime64(int(event["timestamp_ns"]), "ns"), color=color, linestyle=linestyle,
                   linewidth=0.9, alpha=alpha, zorder=1)


def latest_finite_at_or_before(times_ns, values, timestamp_ns):
    """Latest finite archived value with t <= timestamp_ns, or None."""
    index = int(np.searchsorted(times_ns, timestamp_ns, side="right")) - 1
    while index >= 0:
        if np.isfinite(values[index]):
            return float(values[index])
        index -= 1
    return None


def utc_time_axis(ax, start, end):
    ax.set_xlim(start.to_datetime64(), end.to_datetime64())
    locator = mdates.AutoDateLocator(tz=mdates.UTC)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=mdates.UTC, usetex=False))


def tidy(ax, *, log_minor=False):
    """Light horizontal grid on top of the science style (which keeps all four spines)."""
    ax.grid(which="both" if log_minor else "major", axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.72)
