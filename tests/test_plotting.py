"""Shared figure styling.

Figures themselves are judged by looking at them; what is testable is that
the style applies, the palette is the documented one, and ``save`` writes
both formats reproducibly.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from orbital import plotting


@pytest.fixture
def figure():
    fig, ax = plt.subplots()
    yield fig, ax
    plt.close(fig)


def test_palette_is_the_documented_subset():
    """Slots 1-3 of the reference categorical theme, the all-pairs-safe set."""
    assert (plotting.BLUE, plotting.ORANGE, plotting.AQUA) == (
        "#2a78d6", "#eb6834", "#1baf7a"
    )


def test_style_sets_a_recessive_grid_and_white_surface():
    plotting.style()
    assert plt.rcParams["axes.grid"] is True
    assert plt.rcParams["axes.axisbelow"] is True      # grid behind the marks
    assert plt.rcParams["figure.facecolor"] == "white"
    assert plt.rcParams["savefig.dpi"] == 300


def test_despine_hides_only_top_and_right(figure):
    _, ax = figure
    plotting.despine(ax)
    visible = {side: ax.spines[side].get_visible() for side in
               ("top", "right", "left", "bottom")}
    assert visible == {"top": False, "right": False, "left": True, "bottom": True}


def test_save_writes_vector_and_raster(figure, tmp_path, monkeypatch):
    monkeypatch.setattr(plotting, "FIGURE_DIR", tmp_path / "figures")
    fig, ax = figure
    ax.plot([0, 1], [0, 1])
    plotting.save(fig, "smoke")
    assert (tmp_path / "figures" / "smoke.pdf").stat().st_size > 0
    assert (tmp_path / "figures" / "smoke.png").stat().st_size > 0


def test_saved_pdf_is_byte_identical_on_a_rerun(figure, tmp_path, monkeypatch):
    """The creation date is stripped, so regenerating an unchanged figure
    leaves the file out of the git diff.
    """
    monkeypatch.setattr(plotting, "FIGURE_DIR", tmp_path)
    fig, ax = figure
    ax.plot([0, 1], [1, 0])
    plotting.save(fig, "stable")
    first = (tmp_path / "stable.pdf").read_bytes()
    plotting.save(fig, "stable")
    assert (tmp_path / "stable.pdf").read_bytes() == first
