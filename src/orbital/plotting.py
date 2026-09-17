"""Shared figure styling for the report figures.

Palette: slots 1-3 of the reference categorical theme (blue, orange, aqua),
the documented all-pairs-safe subset. Series also differ by marker and dash
pattern, so figures survive greyscale printing and colour-vision deficiency.
Aqua is below 3:1 contrast on white, so its numbers are always printed too.
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt

from orbital.paths import WRITEUP_DIR

FIGURE_DIR = WRITEUP_DIR / "figures"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8880"
GRID = "#e3e2dd"
BAND = "#ecebe7"


def style() -> None:
    """Apply the report style to matplotlib's global state."""
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 300,
        "font.family": "serif", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "legend.fontsize": 8,
        "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.labelcolor": INK,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def despine(ax: Any) -> None:
    """Hide the top and right spines."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig: Any, stem: str) -> None:
    """Write a figure as vector PDF and PNG under ``writeup/figures``.

    The PDF creation date is omitted so that regenerating an unchanged figure
    leaves the file byte-identical and out of the git diff.

    Parameters
    ----------
    fig
        The matplotlib figure.
    stem
        File name without extension.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(FIGURE_DIR / f"{stem}.png", bbox_inches="tight")
