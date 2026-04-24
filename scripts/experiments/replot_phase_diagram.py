"""Re-plot phase diagram from cached cells.json (no rerun)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.synthetic_latency.phase_diagram import (
    P50_SPREADS,
    TAIL_RATIOS,
)
from scripts.experiments.run_phase_diagram import (
    OUT,
    plot_heatmap_p99,
    plot_heatmap_slo,
)


def main() -> None:
    cells_path = OUT / "cells.json"
    raw = json.load(cells_path.open())

    # Reconstruct agg dict with tuple keys.
    # Key format was f"{sp:.3f}_{tr:.3f}_{strategy}"
    agg = {}
    for k, v in raw.items():
        parts = k.split("_")
        # parts = [sp, tr, strategy_name_with_underscores]
        sp_val = float(parts[0])
        tr_val = float(parts[1])
        strategy = "_".join(parts[2:])
        agg[(sp_val, tr_val, strategy)] = v

    plot_heatmap_p99(agg, P50_SPREADS, TAIL_RATIOS, OUT / "heatmap_p99.png")
    plot_heatmap_slo(agg, P50_SPREADS, TAIL_RATIOS, OUT / "heatmap_slo.png")


if __name__ == "__main__":
    main()
