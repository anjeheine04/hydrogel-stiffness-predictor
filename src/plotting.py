"""Shared, colour-blind-safe matplotlib styling."""

import matplotlib.pyplot as plt

# First three slots of a validated categorical palette (CVD-safe as a set).
POLYMER_COLORS = {"GelMA": "#2a78d6", "PEGDA": "#eb6834", "Alginate": "#1baf7a"}

TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "text.color": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold",
    })


def style_axes(ax, grid_axis: str = "both") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
