"""Common utilities and styles for plotting.

This module provides shared configurations for all plot scripts:
- Matplotlib style settings (ICML paper quality)
- Color palettes for providers, strategies, policies
- Common utility functions
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Matplotlib Style Configuration (ICML Paper Quality)
# =============================================================================

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


# =============================================================================
# Color Palettes
# =============================================================================

# Provider colors (for latency profiling)
PROVIDER_COLORS = {
    "Groq": "#2ecc71",  # Green (fast)
    "SambaNova": "#27ae60",  # Dark green
    "Cerebras": "#3498db",  # Blue
    "Fireworks": "#e74c3c",  # Red
    "Together": "#e67e22",  # Orange
    "Parasail": "#9b59b6",  # Purple
    "Friendli": "#1abc9c",  # Teal
    "Novita": "#f39c12",  # Yellow
    "Cloudflare": "#34495e",  # Dark gray
    "Nebius": "#16a085",  # Dark teal
    "Hyperbolic": "#8e44ad",  # Dark purple
    "Crusoe": "#d35400",  # Dark orange
}

# Hedging strategy colors (Phase 4)
STRATEGY_COLORS = {
    "never": "#1f77b4",  # Blue
    "always": "#2ca02c",  # Green
    "fixed_timeout": "#ff7f0e",  # Orange
    "smart_survival": "#d62728",  # Red
    "smart_residual": "#9467bd",  # Purple
}

STRATEGY_LABELS = {
    "never": "Never Hedge (Baseline)",
    "always": "Always Hedge",
    "fixed_timeout": "Fixed Timeout",
    "smart_survival": "Smart (Survival)",
    "smart_residual": "Smart (Residual)",
}

# Policy colors (Phase 5 online evaluation)
POLICY_COLORS = {
    "openrouter_auto": "#1f77b4",  # Blue
    "sort_price": "#17becf",  # Cyan
    "sort_throughput": "#bcbd22",  # Yellow-green
    "sort_latency": "#e377c2",  # Pink
    "lp_mix": "#2ca02c",  # Green
    "smart_hedge": "#ff7f0e",  # Orange
    "cheapest_fixed": "#9467bd",  # Purple
    "fastest_fixed": "#d62728",  # Red
}

POLICY_LABELS = {
    "openrouter_auto": "OpenRouter Auto",
    "sort_price": "sort=price",
    "sort_throughput": "sort=throughput",
    "sort_latency": "sort=latency",
    "lp_mix": "Latency-Aware (Ours)",
    "smart_hedge": "Smart Hedge (Ours)",
    "cheapest_fixed": "Cheapest Fixed",
    "fastest_fixed": "Fastest Fixed",
}

# Routing policy colors (Phase 3 simulation)
ROUTING_COLORS = {
    "oracle": "#2ecc71",  # Green
    "lp_mix": "#3498db",  # Blue
    "single_best": "#e74c3c",  # Red
    "cheapest": "#f39c12",  # Yellow
    "random": "#95a5a6",  # Gray
}


# =============================================================================
# Utility Functions
# =============================================================================


def get_provider_color(provider: str) -> str:
    """Get color for a provider, with fallback."""
    return PROVIDER_COLORS.get(provider, "#7f8c8d")


def get_strategy_color(strategy: str) -> str:
    """Get color for a hedging strategy, with fallback."""
    for key in STRATEGY_COLORS:
        if key in strategy:
            return STRATEGY_COLORS[key]
    return "#7f8c8d"


def get_policy_color(policy: str) -> str:
    """Get color for a policy, with fallback."""
    return POLICY_COLORS.get(policy, "#7f8c8d")


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    name: str,
    formats: list[str] | None = None,
) -> None:
    """Save figure in multiple formats.

    Args:
        fig: Matplotlib figure.
        output_dir: Output directory.
        name: Base filename (without extension).
        formats: List of formats (default: ['png']).
    """
    if formats is None:
        formats = ["png"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path)
        print(f"Saved: {path}")


def format_latency(ms: float) -> str:
    """Format latency value for display."""
    if ms >= 1000:
        return f"{ms/1000:.1f}s"
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
    """Add subtle grid lines to an axis.

    Args:
        ax: Matplotlib axes.
        axis: 'x', 'y', or 'both'.
    """
    if axis in ("y", "both"):
        ax.yaxis.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    if axis in ("x", "both"):
        ax.xaxis.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
