"""Matplotlib style configuration for paper and presentation figures."""

from __future__ import annotations

import matplotlib.pyplot as plt

PAPER_STYLE = {
    # Font sizes optimized for ICML double-column (3.25in column width)
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    # Figure size for full-width figures (textwidth ~ 6.75in)
    "figure.figsize": (7, 3.5),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    # Grid and spines
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    # Lines
    "lines.linewidth": 2.5,
    "lines.markersize": 8,
    # Legend
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
}

PRESENTATION_STYLE = {
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
}


def apply_style(style: str = "paper") -> None:
    """Apply matplotlib style configuration.

    Args:
        style: Either 'paper' (ICML quality) or 'presentation'.
    """
    if style == "paper":
        plt.rcParams.update(PAPER_STYLE)
    elif style == "presentation":
        plt.rcParams.update(PRESENTATION_STYLE)
    else:
        raise ValueError(f"Unknown style: {style}")
