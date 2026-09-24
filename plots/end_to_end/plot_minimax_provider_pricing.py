"""Plot the MiniMax M2.5 OpenRouter provider pricing panel for the paper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from plots.end_to_end.frontier_plotting import PROVIDER_MIX_COLORS
from plots.style import apply_style


@dataclass(frozen=True)
class PriceRow:
    provider: str
    label: str
    input_price: float
    output_price: float


PRICE_ROWS = (
    PriceRow("OR_DeepInfra", "DeepInfra", 0.056, 1.15),
    PriceRow("OR_AtlasCloud", "Atlas", 0.254, 1.20),
    PriceRow("OR_Chutes", "Chutes", 0.110, 1.20),
    PriceRow("OR_AkashML", "Akash", 0.150, 1.15),
    PriceRow("OR_WandB", "W&B", 0.300, 1.20),
    PriceRow("OR_Novita", "Novita", 0.297, 1.20),
    PriceRow("OR_Phala", "Phala", 0.200, 1.38),
    PriceRow("OR_SiliconFlow", "SFlow", 0.271, 1.20),
)


def lighter(hex_color: str, factor: float = 0.55) -> tuple[float, float, float]:
    """Blend a hex color toward white by ``factor``."""
    raw = hex_color.lstrip("#")
    channels = [int(raw[index : index + 2], 16) / 255.0 for index in (0, 2, 4)]
    return tuple(channel + (1.0 - channel) * factor for channel in channels)


def plot(output_path: Path) -> None:
    apply_style("paper")
    plt.rcParams.update(
        {
            "font.size": 7.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "savefig.pad_inches": 0.01,
        }
    )

    rows = list(PRICE_ROWS)
    y = list(range(len(rows)))
    colors = [PROVIDER_MIX_COLORS[row.provider] for row in rows]
    input_prices = [row.input_price for row in rows]
    output_prices = [row.output_price for row in rows]

    fig, ax = plt.subplots(figsize=(3.35, 2.2), constrained_layout=False)
    ax.barh(
        [value - 0.17 for value in y],
        input_prices,
        height=0.28,
        color=[lighter(color) for color in colors],
        edgecolor="white",
        linewidth=0.3,
    )
    ax.barh(
        [value + 0.17 for value in y],
        output_prices,
        height=0.28,
        color=colors,
        edgecolor="white",
        linewidth=0.3,
    )

    ax.set_yticks(y, [row.label for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("USD / 1M tokens")
    ax.set_xlim(0, 1.55)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5], ["0", "0.5", "1.0", "1.5"])
    ax.grid(axis="x", color="#9a9a9a", alpha=0.28, linewidth=0.5)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.29, right=0.99, bottom=0.16, top=0.97)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> int:
    output_path = Path("../paper/figures/real_world_minimax_provider_pricing.pdf")
    plot(output_path)
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
