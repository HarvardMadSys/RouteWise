"""Shared plotting helpers used across paper figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    name: str,
    formats: list[str] | None = None,
    *,
    full_canvas: bool = False,
) -> None:
    """Save figure in multiple formats.

    Args:
        fig: Matplotlib figure.
        output_dir: Output directory.
        name: Base filename (without extension).
        formats: List of formats (default: ['png']).
        full_canvas: Save at the exact figure size instead of a tight bounding
            box. Use this when several figures must share identical pixel
            dimensions (e.g. a row of panels in the paper); the caller is then
            responsible for reserving margins so nothing is clipped.
    """
    if formats is None:
        formats = ["png"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_kwargs = {"bbox_inches": fig.bbox_inches, "pad_inches": 0.0} if full_canvas else {}
    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path, **save_kwargs)
        print(f"Saved: {path}")


def format_latency(ms: float) -> str:
    """Format latency value for display."""
    if ms >= 1000:
        return f"{ms / 1000:.1f}s"
    return f"{ms:.0f}ms"


def compute_ecdf(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute empirical CDF.

    Args:
        data: 1D array of values.

    Returns:
        (x, y) arrays for plotting ECDF.
    """
    sorted_data = np.sort(data)
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, ecdf


def compute_ccdf(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute complementary CDF (1 - CDF).

    Args:
        data: 1D array of values.

    Returns:
        (x, y) arrays for plotting CCDF.
    """
    x, y = compute_ecdf(data)
    return x, 1 - y


def add_light_grid(ax: plt.Axes, axis: str = "y") -> None:
    """Add subtle dashed grid lines to an axis.

    Args:
        ax: Matplotlib axes.
        axis: 'x', 'y', or 'both'.
    """
    if axis in ("y", "both"):
        ax.yaxis.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    if axis in ("x", "both"):
        ax.xaxis.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
