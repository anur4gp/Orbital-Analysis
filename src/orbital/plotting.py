"""Shared figure style for the report."""
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
    """Write ``stem.pdf`` and ``stem.png`` under ``writeup/figures``.

    The PDF creation date is dropped so unchanged figures stay byte-identical.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(FIGURE_DIR / f"{stem}.png", bbox_inches="tight")
