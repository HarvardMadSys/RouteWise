"""Phase-diagram sweep: where does joint beat two_layer?

Axes
----
x: p50_ratio = P50(S_Q) / P50(S_A). A value of 1 means the subscription is
   as fast as the API; larger values mean the subscription is slower.
y: saturation = workload_size / quota. 0.5 means the quota covers twice
   the workload; 5 means the workload is 5x the quota.

Per cell we instantiate a scenario with a single S_Q + single S_A provider,
run `two_layer` and `joint_nohedge`, and record three outcomes:

- cost_ratio(joint / two_layer)   (< 1 means joint is cheaper)
- slo_diff = two_layer_viol - joint_viol  (> 0 means joint has fewer viols)
- winner: which strategy has the lower expected dollar cost when an SLO
  violation is priced at V_penalty

The resulting heat maps tell the reader at a glance which regimes need
the joint extension and which are fine with two_layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..providers import LogNormal
from ..workload import generate_workload
from .providers import ProviderTier, QuotaState, TieredProvider
from .scenarios import TieredScenarioConfig
from .strategies import run_tiered_strategy


# ---------------------------------------------------------------------------
# Scenario instantiation
# ---------------------------------------------------------------------------

_SLO_MS = 2000.0
_BASE_SA_P50_MS = 100.0
_BASE_SIGMA = 0.5
_SA_COST_PER_TOKEN = 3.0e-6
_DURATION_SEC = 3600.0
_QUOTA_SIZE = 200          # fixed; workload scales with saturation
_TPS = LogNormal(mu=5.5, sigma=0.3)


def _lognormal_p50_sigma(p50_ms: float, sigma: float = _BASE_SIGMA) -> LogNormal:
    return LogNormal(mu=math.log(p50_ms), sigma=sigma)


def _make_cell_scenario(
    p50_ratio: float,
    saturation: float,
) -> TieredScenarioConfig:
    """Build a single-cell scenario for the grid."""
    sq_p50 = _BASE_SA_P50_MS * p50_ratio

    n_requests = int(_QUOTA_SIZE * saturation)

    return TieredScenarioConfig(
        name=f"cell_r{p50_ratio}_s{saturation}",
        description=(
            f"cell: p50_ratio={p50_ratio}, saturation={saturation}"
        ),
        providers=[
            TieredProvider(
                name="S_Q",
                cost_per_token=0.0,
                ttft_dist=_lognormal_p50_sigma(sq_p50),
                tps_dist=_TPS,
                tier=ProviderTier.S_Q,
                quota=QuotaState(size=_QUOTA_SIZE, window_sec=_DURATION_SEC * 2),
            ),
            TieredProvider(
                name="S_A",
                cost_per_token=_SA_COST_PER_TOKEN,
                ttft_dist=_lognormal_p50_sigma(_BASE_SA_P50_MS),
                tps_dist=_TPS,
                tier=ProviderTier.S_A,
            ),
        ],
        n_requests=n_requests,
        duration_seconds=_DURATION_SEC,
        primary_slo_ms=_SLO_MS,
        slo_thresholds_ms=[_SLO_MS],
    )


# ---------------------------------------------------------------------------
# Per-cell evaluation
# ---------------------------------------------------------------------------


@dataclass
class CellResult:
    p50_ratio: float
    saturation: float
    two_layer_cost: float
    two_layer_viol: float
    joint_cost: float
    joint_viol: float

    @property
    def cost_ratio(self) -> float:
        if self.two_layer_cost == 0:
            return float("inf") if self.joint_cost > 0 else 1.0
        return self.joint_cost / self.two_layer_cost

    @property
    def slo_diff_pp(self) -> float:
        """Positive = joint reduces SLO violations vs two_layer."""
        return (self.two_layer_viol - self.joint_viol) * 100.0


_DEFAULT_SEEDS: list[int] = [42, 43, 44, 45, 46]


def _evaluate_cell(
    p50_ratio: float,
    saturation: float,
    seeds: list[int] | None = None,
) -> CellResult:
    """Evaluate one grid cell, averaging across seeds to suppress sample noise.

    With ~100 requests per cell, the standard error on SLO violation rate is
    about 3 pp at an 8 % true rate; single-seed cells are visibly noisy in
    the phase diagram. Averaging over 5 seeds brings stderr to ~1.3 pp.
    """
    if seeds is None:
        seeds = _DEFAULT_SEEDS
    scenario = _make_cell_scenario(p50_ratio, saturation)
    requests = generate_workload(
        n_requests=scenario.n_requests,
        duration_seconds=scenario.duration_seconds,
        seed=0,
    )

    tl_costs: list[float] = []
    tl_viols: list[float] = []
    jt_costs: list[float] = []
    jt_viols: list[float] = []

    for seed in seeds:
        tl = run_tiered_strategy(scenario, requests, "two_layer", seed=seed)
        # Use the profile-based joint_ucb so the phase diagram reflects what a
        # production router (without oracle access to true P50/P95) would see.
        jt = run_tiered_strategy(scenario, requests, "joint_ucb", seed=seed)
        tl_costs.append(tl.mean_cost_usd())
        tl_viols.append(tl.slo_violation_rate(_SLO_MS))
        jt_costs.append(jt.mean_cost_usd())
        jt_viols.append(jt.slo_violation_rate(_SLO_MS))

    return CellResult(
        p50_ratio=p50_ratio,
        saturation=saturation,
        two_layer_cost=float(np.mean(tl_costs)),
        two_layer_viol=float(np.mean(tl_viols)),
        joint_cost=float(np.mean(jt_costs)),
        joint_viol=float(np.mean(jt_viols)),
    )


# ---------------------------------------------------------------------------
# Sweep + plot
# ---------------------------------------------------------------------------


_P50_RATIOS: list[float] = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]
_SATURATIONS: list[float] = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]


def run_sweep() -> list[CellResult]:
    cells: list[CellResult] = []
    n_total = len(_P50_RATIOS) * len(_SATURATIONS)
    n_done = 0
    for p in _P50_RATIOS:
        for s in _SATURATIONS:
            cells.append(_evaluate_cell(p, s))
            n_done += 1
            print(
                f"  [{n_done}/{n_total}] p50_ratio={p:.1f}  "
                f"saturation={s:.2f}  "
                f"joint_cost={cells[-1].joint_cost:.2e}  "
                f"tl_cost={cells[-1].two_layer_cost:.2e}  "
                f"slo_diff={cells[-1].slo_diff_pp:+.1f}pp"
            )
    return cells


def _cells_to_matrix(
    cells: list[CellResult],
    attr: str,
) -> np.ndarray:
    """Shape: (len(SATURATIONS), len(P50_RATIOS)) for imshow y=sat, x=ratio."""
    mat = np.zeros((len(_SATURATIONS), len(_P50_RATIOS)))
    idx = {(c.p50_ratio, c.saturation): c for c in cells}
    for j, p in enumerate(_P50_RATIOS):
        for i, s in enumerate(_SATURATIONS):
            c = idx[(p, s)]
            mat[i, j] = getattr(c, attr) if not callable(getattr(c, attr, None)) \
                else getattr(c, attr)()
    return mat


def plot_phase_diagrams(
    cells: list[CellResult],
    output_path: Path,
    v_penalty_usd: float = 1e-3,
) -> None:
    """Three-panel heatmap: cost ratio, SLO reduction, effective-cost ratio.

    The third panel combines cost and SLO into a single comparison by
    charging an SLO violation at `v_penalty_usd` dollars per violation.
    Effective cost = mean_cost_usd + v_penalty_usd * violation_rate.

    This collapses the cost / SLO tradeoff into a single "which strategy
    is cheaper in economic terms" ratio, answering the question the raw
    cost-ratio panel cannot answer on its own (it looks like joint is
    more expensive on the right, but it is purchasing SLO compliance).
    """
    idx = {(c.p50_ratio, c.saturation): c for c in cells}
    ratio_mat = np.array([
        [idx[(p, s)].cost_ratio for p in _P50_RATIOS]
        for s in _SATURATIONS
    ])
    slo_mat = np.array([
        [idx[(p, s)].slo_diff_pp for p in _P50_RATIOS]
        for s in _SATURATIONS
    ])
    # Effective cost ratio: includes SLO penalty.
    # Treat both-zero (fully-free, SLO-safe regime) as 1.0 (tied).
    def _eff_ratio(c: CellResult) -> float:
        jt_eff = c.joint_cost + v_penalty_usd * c.joint_viol
        tl_eff = c.two_layer_cost + v_penalty_usd * c.two_layer_viol
        if tl_eff <= 1e-12 and jt_eff <= 1e-12:
            return 1.0
        if tl_eff <= 1e-12:
            return float("inf")
        return jt_eff / tl_eff

    eff_mat = np.array([
        [_eff_ratio(idx[(p, s)]) for p in _P50_RATIOS]
        for s in _SATURATIONS
    ])

    fig, (ax_cost, ax_slo, ax_eff) = plt.subplots(1, 3, figsize=(16.5, 4.2))

    # Cost ratio: < 1 means joint cheaper. Use centered divergent colormap.
    vmax = max(1.0, float(np.nanmax(np.where(np.isfinite(ratio_mat), ratio_mat, 1.0))))
    im_cost = ax_cost.imshow(
        ratio_mat,
        aspect="auto",
        cmap="RdBu_r",
        origin="lower",
        vmin=0.0, vmax=2.0,
    )
    for i in range(ratio_mat.shape[0]):
        for j in range(ratio_mat.shape[1]):
            val = ratio_mat[i, j]
            txt = f"{val:.2f}" if np.isfinite(val) else "-"
            ax_cost.text(
                j, i, txt, ha="center", va="center",
                fontsize=8,
                color="white" if abs(val - 1.0) > 0.3 else "black",
            )
    ax_cost.set_xticks(range(len(_P50_RATIOS)))
    ax_cost.set_xticklabels([f"{r:.0f}x" for r in _P50_RATIOS])
    ax_cost.set_yticks(range(len(_SATURATIONS)))
    ax_cost.set_yticklabels([f"{s:.2f}x" for s in _SATURATIONS])
    ax_cost.set_xlabel("P50(S_Q) / P50(S_A)")
    ax_cost.set_ylabel("Workload / quota")
    ax_cost.set_title("Cost ratio (joint / two_layer); <1 means joint cheaper")
    plt.colorbar(im_cost, ax=ax_cost, label="ratio")

    # SLO difference (percentage points).
    vmax_slo = max(1.0, float(np.max(np.abs(slo_mat))))
    im_slo = ax_slo.imshow(
        slo_mat,
        aspect="auto",
        cmap="RdBu",
        origin="lower",
        vmin=-vmax_slo, vmax=vmax_slo,
    )
    for i in range(slo_mat.shape[0]):
        for j in range(slo_mat.shape[1]):
            val = slo_mat[i, j]
            txt = f"{val:+.1f}" if abs(val) >= 0.1 else "~0"
            ax_slo.text(
                j, i, txt, ha="center", va="center",
                fontsize=8,
                color="white" if abs(val) > vmax_slo * 0.4 else "black",
            )
    ax_slo.set_xticks(range(len(_P50_RATIOS)))
    ax_slo.set_xticklabels([f"{r:.0f}x" for r in _P50_RATIOS])
    ax_slo.set_yticks(range(len(_SATURATIONS)))
    ax_slo.set_yticklabels([f"{s:.2f}x" for s in _SATURATIONS])
    ax_slo.set_xlabel("P50(S_Q) / P50(S_A)")
    ax_slo.set_ylabel("Workload / quota")
    ax_slo.set_title("SLO violation reduction by joint (pp)")
    plt.colorbar(im_slo, ax=ax_slo, label="pp reduction")

    # Third panel: effective-cost ratio (cost + V*viol).
    im_eff = ax_eff.imshow(
        eff_mat,
        aspect="auto",
        cmap="RdBu_r",
        origin="lower",
        vmin=0.0, vmax=2.0,
    )
    for i in range(eff_mat.shape[0]):
        for j in range(eff_mat.shape[1]):
            val = eff_mat[i, j]
            txt = f"{val:.2f}" if np.isfinite(val) else "-"
            ax_eff.text(
                j, i, txt, ha="center", va="center",
                fontsize=8,
                color="white" if abs(val - 1.0) > 0.3 else "black",
            )
    ax_eff.set_xticks(range(len(_P50_RATIOS)))
    ax_eff.set_xticklabels([f"{r:.0f}x" for r in _P50_RATIOS])
    ax_eff.set_yticks(range(len(_SATURATIONS)))
    ax_eff.set_yticklabels([f"{s:.2f}x" for s in _SATURATIONS])
    ax_eff.set_xlabel("P50(S_Q) / P50(S_A)")
    ax_eff.set_ylabel("Workload / quota")
    ax_eff.set_title(
        f"Effective-cost ratio (V={v_penalty_usd:.0e}); <1 means joint wins"
    )
    plt.colorbar(im_eff, ax=ax_eff, label="ratio")

    fig.suptitle(
        f"Phase diagram: joint vs two_layer (SLO={_SLO_MS:.0f}ms, "
        f"quota={_QUOTA_SIZE}, averaged over {len(_DEFAULT_SEEDS)} seeds)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
