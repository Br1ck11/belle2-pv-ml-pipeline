"""Figure style of the train package: SciencePlots "science" with LaTeX text.

Used for the training loss plot (loss.png). ``science_style()`` is a context manager (also usable as a
decorator) that applies the style while a figure is built and saved. All
text is typeset by LaTeX (text.usetex), which needs a TeX installation with
``latex`` and ``dvipng`` (e.g. MacTeX); LaTeX is a system program, not a pip
package. If neither is on PATH nor in a standard TeX location, a warning is
printed and the "no-latex" variant (same look, matplotlib mathtext) is used.

Kept as a copy in each package (train, evaluation, plotting) so that no
package needs another one for its figures; keep the three in sync.
"""

import glob
import os
import shutil
import warnings

import matplotlib
import matplotlib.style


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
