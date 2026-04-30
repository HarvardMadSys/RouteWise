"""CSV + JSON logging for real online evaluation.

Schema mirrors phase6_joint_online_evaluation's per-request log, plus a few
additional columns (``hedge_delay_ms``, ``lp_weights``) needed for the new
budget LP and Hedge-ProbTarget paper line.

Thread-safe: all writes hold a single ``threading.Lock`` so multiple
dispatch threads can append concurrently.
"""

from __future__ import annotations

import csv
import json
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.real_evaluation.executor import HedgedResult
from experiments.real_evaluation.policies import RoutingDecision
from experiments.real_evaluation.transports import SingleRequestResult

CSV_FIELDS: tuple[str, ...] = (
    "ts",
    "policy",
    "req_id",
    "prompt_tokens",
    "max_tokens",
    "primary_provider",
    "backup_provider",
    "hedge_triggered",
    "hedge_winner",
    "hedge_delay_ms",
    "actual_provider",
    "tier",
    "transport",
    "ttft_ms",
    "e2e_ms",
    "status",
    "error_message",
    "http_status",
    "retry_count",
    "retry_sleep_ms",
    "rate_limited",
    "billed_cost_usd",
    "primary_cost_usd",
    "backup_cost_usd",
    "lp_status",
    "lp_weights",
    "budget_usd",
    "reference_cost_usd",
    "tier_mix",
    "notes",
)


@dataclass
class RequestLogRow:
    """One CSV row, all fields populated."""

    ts: float
    policy: str
    req_id: str
    prompt_tokens: int
    max_tokens: int
    primary_provider: str | None
    backup_provider: str | None
    hedge_triggered: bool
    hedge_winner: str | None
    hedge_delay_ms: float | None
    actual_provider: str
    tier: str | None
    transport: str | None
    ttft_ms: float
    e2e_ms: float
    status: str
    error_message: str | None
    http_status: int | None
    retry_count: int
    retry_sleep_ms: float
    rate_limited: bool
    billed_cost_usd: float
    primary_cost_usd: float
    backup_cost_usd: float
    lp_status: str | None
    lp_weights: dict[str, float] | None
    budget_usd: float | None
    reference_cost_usd: float | None
    tier_mix: dict[str, float] | None
    notes: str | None

    def to_csv_dict(self) -> dict[str, str]:
        def _maybe_json(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, (dict, list)):
                return json.dumps(value, separators=(",", ":"))
            return str(value)

        return {
            "ts": f"{self.ts:.6f}",
            "policy": self.policy,
            "req_id": self.req_id,
            "prompt_tokens": str(self.prompt_tokens),
            "max_tokens": str(self.max_tokens),
            "primary_provider": self.primary_provider or "",
            "backup_provider": self.backup_provider or "",
            "hedge_triggered": "1" if self.hedge_triggered else "0",
            "hedge_winner": self.hedge_winner or "",
            "hedge_delay_ms": (
                f"{self.hedge_delay_ms:.3f}"
                if self.hedge_delay_ms is not None
                else ""
            ),
            "actual_provider": self.actual_provider,
            "tier": self.tier or "",
            "transport": self.transport or "",
            "ttft_ms": f"{self.ttft_ms:.3f}",
            "e2e_ms": f"{self.e2e_ms:.3f}",
            "status": self.status,
            "error_message": self.error_message or "",
            "http_status": str(self.http_status or ""),
            "retry_count": str(self.retry_count),
            "retry_sleep_ms": f"{self.retry_sleep_ms:.3f}",
            "rate_limited": "1" if self.rate_limited else "0",
            "billed_cost_usd": f"{self.billed_cost_usd:.8f}",
            "primary_cost_usd": f"{self.primary_cost_usd:.8f}",
            "backup_cost_usd": f"{self.backup_cost_usd:.8f}",
            "lp_status": self.lp_status or "",
            "lp_weights": _maybe_json(self.lp_weights),
            "budget_usd": (
                f"{self.budget_usd:.8f}" if self.budget_usd is not None else ""
            ),
            "reference_cost_usd": (
                f"{self.reference_cost_usd:.8f}"
                if self.reference_cost_usd is not None
                else ""
            ),
            "tier_mix": _maybe_json(self.tier_mix),
            "notes": self.notes or "",
        }


class Recorder:
    """Thread-safe CSV writer + summary aggregator."""

    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "requests.csv"
        self._lock = threading.Lock()
        self._rows: list[RequestLogRow] = []
        self._csv_handle = self.csv_path.open("w", newline="")
        self._writer = csv.DictWriter(self._csv_handle, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._csv_handle.flush()

    def write_row(self, row: RequestLogRow) -> None:
        with self._lock:
            self._rows.append(row)
            self._writer.writerow(row.to_csv_dict())
            self._csv_handle.flush()

    def write_request(
        self,
        *,
        policy: str,
        req_id: str,
        ctx_prompt_tokens: int,
        ctx_max_tokens: int,
        decision: RoutingDecision,
        primary_result: SingleRequestResult,
        backup_result: SingleRequestResult | None = None,
        hedge_triggered: bool = False,
        hedge_winner: str | None = None,
        hedge_delay_sec: float | None = None,
        chosen_result: SingleRequestResult | None = None,
        tier: str | None = None,
        transport: str | None = None,
        ts: float | None = None,
    ) -> None:
        """Convenience wrapper that builds a ``RequestLogRow`` from common parts."""
        chosen = chosen_result or primary_result
        primary_cost = primary_result.billed_cost_usd
        backup_cost = backup_result.billed_cost_usd if backup_result else 0.0
        billed_cost = primary_cost + backup_cost
        hedge_delay_ms = (
            hedge_delay_sec * 1000.0
            if hedge_delay_sec is not None and hedge_delay_sec != float("inf")
            else None
        )
        row = RequestLogRow(
            ts=ts if ts is not None else time.time(),
            policy=policy,
            req_id=req_id,
            prompt_tokens=ctx_prompt_tokens,
            max_tokens=ctx_max_tokens,
            primary_provider=decision.primary,
            backup_provider=decision.hedge,
            hedge_triggered=hedge_triggered,
            hedge_winner=hedge_winner,
            hedge_delay_ms=hedge_delay_ms,
            actual_provider=chosen.provider,
            tier=tier,
            transport=transport,
            ttft_ms=chosen.ttft_ms,
            e2e_ms=chosen.e2e_ms,
            status=chosen.status,
            error_message=chosen.error_message,
            http_status=chosen.http_status,
            retry_count=chosen.retry_count,
            retry_sleep_ms=chosen.retry_sleep_ms,
            rate_limited=chosen.rate_limited,
            billed_cost_usd=billed_cost,
            primary_cost_usd=primary_cost,
            backup_cost_usd=backup_cost,
            lp_status=decision.lp_status,
            lp_weights=decision.lp_weights,
            budget_usd=decision.budget_usd,
            reference_cost_usd=decision.reference_cost_usd,
            tier_mix=decision.tier_mix,
            notes=decision.notes,
        )
        self.write_row(row)

    def write_hedged(
        self,
        *,
        policy: str,
        req_id: str,
        ctx_prompt_tokens: int,
        ctx_max_tokens: int,
        decision: RoutingDecision,
        hedged: HedgedResult,
        hedge_delay_sec: float,
        tier: str | None = None,
        transport: str | None = None,
        ts: float | None = None,
    ) -> None:
        """Convenience wrapper that unpacks a ``HedgedResult``."""
        self.write_request(
            policy=policy,
            req_id=req_id,
            ctx_prompt_tokens=ctx_prompt_tokens,
            ctx_max_tokens=ctx_max_tokens,
            decision=decision,
            primary_result=hedged.primary_result,
            backup_result=hedged.backup_result,
            hedge_triggered=hedged.hedge_triggered,
            hedge_winner=hedged.winner if hedged.hedge_triggered else None,
            hedge_delay_sec=hedge_delay_sec,
            chosen_result=hedged.chosen_result,
            tier=tier,
            transport=transport,
            ts=ts,
        )

    def write_summary(self, slo_ms: float) -> Path:
        """Write per-policy aggregates to ``summary.json`` and return the path."""
        with self._lock:
            rows = list(self._rows)
        per_policy: dict[str, dict[str, Any]] = {}
        for row in rows:
            stats = per_policy.setdefault(
                row.policy,
                {
                    "n_total": 0,
                    "n_success": 0,
                    "n_slo_violation": 0,
                    "n_hedge_triggered": 0,
                    "n_hedge_won_by_backup": 0,
                    "n_rate_limited": 0,
                    "total_retry_count": 0,
                    "total_retry_sleep_ms": 0.0,
                    "total_cost_usd": 0.0,
                    "ttft_ms_success": [],
                    "e2e_ms_success": [],
                },
            )
            stats["n_total"] += 1
            stats["total_cost_usd"] += row.billed_cost_usd
            stats["total_retry_count"] += row.retry_count
            stats["total_retry_sleep_ms"] += row.retry_sleep_ms
            if row.rate_limited:
                stats["n_rate_limited"] += 1
            if row.hedge_triggered:
                stats["n_hedge_triggered"] += 1
                if row.hedge_winner == "backup":
                    stats["n_hedge_won_by_backup"] += 1
            if row.status == "success":
                stats["n_success"] += 1
                stats["ttft_ms_success"].append(row.ttft_ms)
                stats["e2e_ms_success"].append(row.e2e_ms)
                if row.ttft_ms > slo_ms:
                    stats["n_slo_violation"] += 1
            else:
                stats["n_slo_violation"] += 1

        summary: dict[str, dict[str, Any]] = {}
        for policy, stats in per_policy.items():
            n_total = stats["n_total"]
            ttfts = stats.pop("ttft_ms_success")
            stats.pop("e2e_ms_success")
            success_rate = stats["n_success"] / n_total if n_total else 0.0
            slo_violation_rate = stats["n_slo_violation"] / n_total if n_total else 0.0
            hedge_trigger_rate = (
                stats["n_hedge_triggered"] / n_total if n_total else 0.0
            )
            hedge_backup_win_rate = (
                stats["n_hedge_won_by_backup"] / stats["n_hedge_triggered"]
                if stats["n_hedge_triggered"]
                else 0.0
            )
            summary[policy] = {
                **stats,
                "success_rate": round(success_rate, 6),
                "slo_violation_rate": round(slo_violation_rate, 6),
                "hedge_trigger_rate": round(hedge_trigger_rate, 6),
                "hedge_backup_win_rate": round(hedge_backup_win_rate, 6),
                "mean_cost_usd": round(
                    stats["total_cost_usd"] / n_total if n_total else 0.0, 8
                ),
                "ttft_ms_p50": _percentile(ttfts, 50.0),
                "ttft_ms_p90": _percentile(ttfts, 90.0),
                "ttft_ms_p99": _percentile(ttfts, 99.0),
                "ttft_ms_mean": _mean(ttfts),
            }
        summary["_meta"] = {
            "slo_ms": slo_ms,
            "n_total_rows": len(rows),
            "csv_path": str(self.csv_path.relative_to(self.output_dir)),
        }
        path = self.output_dir / "summary.json"
        path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        return path

    def close(self) -> None:
        with self._lock:
            if not self._csv_handle.closed:
                self._csv_handle.flush()
                self._csv_handle.close()


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(statistics.mean(values)), 3)


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(float(values[0]), 3)
    s = sorted(values)
    rank = (pct / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return round(float(s[lo] + (s[hi] - s[lo]) * frac), 3)


__all__ = [
    "CSV_FIELDS",
    "Recorder",
    "RequestLogRow",
]
