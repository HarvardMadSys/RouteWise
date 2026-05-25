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
from typing import TYPE_CHECKING, Any

from rwsim.metrics import PerRequestRecord, Run, Status

if TYPE_CHECKING:
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
    "hedge_checkpoint_ts",
    "primary_start_ts",
    "backup_dispatch_ts",
    "backup_start_ts",
    "primary_first_token_ts",
    "backup_first_token_ts",
    "actual_dispatch_overhead_ms",
    "checkpoint_dispatch_overhead_ms",
    "backup_dispatch_overhead_ms",
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
    "cost_source",
    "primary_cached_input_tokens",
    "backup_cached_input_tokens",
    "primary_observed_cached_input_tokens",
    "backup_observed_cached_input_tokens",
    "primary_routing_estimated_cost_usd",
    "backup_routing_estimated_cost_usd",
    "billed_cost_usd",
    "physical_cost_usd",
    "primary_cost_usd",
    "backup_cost_usd",
    "primary_physical_cost_usd",
    "backup_physical_cost_usd",
    "lp_status",
    "lp_weights",
    "budget_usd",
    "reference_cost_usd",
    "tier_mix",
    "notes",
    "model",
    "hedge_algorithm",
    "hedge_schedule",
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
    hedge_checkpoint_ts: float | None
    primary_start_ts: float | None
    backup_dispatch_ts: float | None
    backup_start_ts: float | None
    primary_first_token_ts: float | None
    backup_first_token_ts: float | None
    actual_dispatch_overhead_ms: float | None
    checkpoint_dispatch_overhead_ms: float | None
    backup_dispatch_overhead_ms: float | None
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
    cost_source: str
    primary_cached_input_tokens: int
    backup_cached_input_tokens: int
    primary_observed_cached_input_tokens: int | None
    backup_observed_cached_input_tokens: int | None
    primary_routing_estimated_cost_usd: float | None
    backup_routing_estimated_cost_usd: float | None
    billed_cost_usd: float
    physical_cost_usd: float
    primary_cost_usd: float
    backup_cost_usd: float
    primary_physical_cost_usd: float
    backup_physical_cost_usd: float
    lp_status: str | None
    lp_weights: dict[str, float] | None
    budget_usd: float | None
    reference_cost_usd: float | None
    tier_mix: dict[str, float] | None
    notes: str | None
    # Policy-level routing identity. Persisted so a CSV reload can recover what
    # model and hedging rule the policy ran, instead of guessing from the row.
    model: str | None = None
    hedge_algorithm: str | None = None
    hedge_schedule: str | None = None

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
                f"{self.hedge_delay_ms:.3f}" if self.hedge_delay_ms is not None else ""
            ),
            "hedge_checkpoint_ts": _format_ts(self.hedge_checkpoint_ts),
            "primary_start_ts": _format_ts(self.primary_start_ts),
            "backup_dispatch_ts": _format_ts(self.backup_dispatch_ts),
            "backup_start_ts": _format_ts(self.backup_start_ts),
            "primary_first_token_ts": _format_ts(self.primary_first_token_ts),
            "backup_first_token_ts": _format_ts(self.backup_first_token_ts),
            "actual_dispatch_overhead_ms": _format_ms(self.actual_dispatch_overhead_ms),
            "checkpoint_dispatch_overhead_ms": _format_ms(
                self.checkpoint_dispatch_overhead_ms
            ),
            "backup_dispatch_overhead_ms": _format_ms(self.backup_dispatch_overhead_ms),
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
            "cost_source": self.cost_source,
            "primary_cached_input_tokens": str(self.primary_cached_input_tokens),
            "backup_cached_input_tokens": str(self.backup_cached_input_tokens),
            "primary_observed_cached_input_tokens": (
                ""
                if self.primary_observed_cached_input_tokens is None
                else str(self.primary_observed_cached_input_tokens)
            ),
            "backup_observed_cached_input_tokens": (
                ""
                if self.backup_observed_cached_input_tokens is None
                else str(self.backup_observed_cached_input_tokens)
            ),
            "primary_routing_estimated_cost_usd": (
                f"{self.primary_routing_estimated_cost_usd:.8f}"
                if self.primary_routing_estimated_cost_usd is not None
                else ""
            ),
            "backup_routing_estimated_cost_usd": (
                f"{self.backup_routing_estimated_cost_usd:.8f}"
                if self.backup_routing_estimated_cost_usd is not None
                else ""
            ),
            "billed_cost_usd": f"{self.billed_cost_usd:.8f}",
            "physical_cost_usd": f"{self.physical_cost_usd:.8f}",
            "primary_cost_usd": f"{self.primary_cost_usd:.8f}",
            "backup_cost_usd": f"{self.backup_cost_usd:.8f}",
            "primary_physical_cost_usd": f"{self.primary_physical_cost_usd:.8f}",
            "backup_physical_cost_usd": f"{self.backup_physical_cost_usd:.8f}",
            "lp_status": self.lp_status or "",
            "lp_weights": _maybe_json(self.lp_weights),
            "budget_usd": (f"{self.budget_usd:.8f}" if self.budget_usd is not None else ""),
            "reference_cost_usd": (
                f"{self.reference_cost_usd:.8f}" if self.reference_cost_usd is not None else ""
            ),
            "tier_mix": _maybe_json(self.tier_mix),
            "notes": self.notes or "",
            "model": self.model or "",
            "hedge_algorithm": self.hedge_algorithm or "",
            "hedge_schedule": self.hedge_schedule or "",
        }


class Recorder:
    """Thread-safe CSV writer + summary aggregator."""

    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "requests.csv"
        self._lock = threading.Lock()
        self._rows: list[RequestLogRow] = []
        self._records: list[PerRequestRecord] = []
        self._start_ts = time.time()
        self._csv_handle = self.csv_path.open("w", newline="")
        self._writer = csv.DictWriter(self._csv_handle, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._csv_handle.flush()

    def write_row(
        self,
        row: RequestLogRow,
        record: PerRequestRecord | None = None,
    ) -> None:
        with self._lock:
            self._rows.append(row)
            if record is not None:
                self._records.append(record)
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
        primary_tier: str | None = None,
        backup_tier: str | None = None,
        final_tier: str | None = None,
        slo_ms: float | None = None,
        transport: str | None = None,
        primary_cached_input_tokens: int = 0,
        backup_cached_input_tokens: int = 0,
        primary_routing_estimated_cost_usd: float | None = None,
        backup_routing_estimated_cost_usd: float | None = None,
        hedge_checkpoint_ts: float | None = None,
        backup_dispatch_ts: float | None = None,
        ts: float | None = None,
        hedge_algorithm: str | None = None,
        hedge_schedule: str | None = None,
        ctx_model: str | None = None,
    ) -> None:
        """Convenience wrapper that builds a ``RequestLogRow`` from common parts.

        ``hedge_algorithm`` and ``hedge_schedule`` describe what hedging rule
        and time-evaluation strategy the *policy* was configured to run, not
        whether this particular request hedged. The runner sets these once
        per policy (e.g. ``"probability_target"`` / ``"slo_relative_checkpoints"``
        for hedge policies, ``"disabled"`` / ``None`` for body-only policies).
        """
        chosen = chosen_result or primary_result
        row_ts = ts if ts is not None else time.time()
        primary_cost = primary_result.billed_cost_usd
        backup_cost = backup_result.billed_cost_usd if backup_result else 0.0
        billed_cost = primary_cost + backup_cost
        primary_physical_cost = float(primary_result.physical_cost_usd)
        backup_physical_cost = float(backup_result.physical_cost_usd) if backup_result else 0.0
        physical_cost = primary_physical_cost + backup_physical_cost
        cost_source = primary_result.cost_source
        if backup_result is not None:
            cost_source = f"{primary_result.cost_source}+{backup_result.cost_source}"
        hedge_delay_ms = (
            hedge_delay_sec * 1000.0
            if hedge_delay_sec is not None and hedge_delay_sec != float("inf")
            else None
        )
        primary_start_ts = primary_result.start_ts if primary_result.start_ts > 0 else None
        backup_start_ts = (
            backup_result.start_ts
            if backup_result is not None and backup_result.start_ts > 0
            else None
        )
        actual_dispatch_overhead_ms = _actual_dispatch_overhead_ms(
            primary_start_ts=primary_start_ts,
            backup_start_ts=backup_start_ts,
            hedge_delay_ms=hedge_delay_ms,
        )
        checkpoint_dispatch_overhead_ms = _checkpoint_dispatch_overhead_ms(
            hedge_checkpoint_ts=hedge_checkpoint_ts,
            backup_start_ts=backup_start_ts,
        )
        backup_dispatch_overhead_ms = _backup_dispatch_overhead_ms(
            backup_dispatch_ts=backup_dispatch_ts,
            backup_start_ts=backup_start_ts,
        )
        record_ttft_ms = _user_visible_ttft_ms(
            primary_result=primary_result,
            chosen_result=chosen,
            hedge_delay_ms=hedge_delay_ms,
        )
        record_e2e_ms = _user_visible_e2e_ms(
            primary_result=primary_result,
            chosen_result=chosen,
            hedge_delay_ms=hedge_delay_ms,
        )
        resolved_primary_tier = primary_tier or tier or ""
        resolved_backup_tier = backup_tier
        resolved_final_tier = final_tier or tier or resolved_primary_tier
        status = _canonical_status(chosen)
        slo_violated = (
            False if slo_ms is None else status != Status.SUCCESS or record_ttft_ms > slo_ms
        )
        row = RequestLogRow(
            ts=row_ts,
            policy=policy,
            req_id=req_id,
            prompt_tokens=ctx_prompt_tokens,
            max_tokens=ctx_max_tokens,
            primary_provider=decision.primary,
            backup_provider=decision.hedge,
            hedge_triggered=hedge_triggered,
            hedge_winner=hedge_winner,
            hedge_delay_ms=hedge_delay_ms,
            hedge_checkpoint_ts=hedge_checkpoint_ts,
            primary_start_ts=primary_start_ts,
            backup_dispatch_ts=backup_dispatch_ts,
            backup_start_ts=backup_start_ts,
            primary_first_token_ts=primary_result.first_token_ts,
            backup_first_token_ts=backup_result.first_token_ts if backup_result else None,
            actual_dispatch_overhead_ms=actual_dispatch_overhead_ms,
            checkpoint_dispatch_overhead_ms=checkpoint_dispatch_overhead_ms,
            backup_dispatch_overhead_ms=backup_dispatch_overhead_ms,
            actual_provider=chosen.provider,
            tier=resolved_final_tier,
            transport=transport,
            ttft_ms=record_ttft_ms,
            e2e_ms=record_e2e_ms,
            status=chosen.status,
            error_message=chosen.error_message,
            http_status=chosen.http_status,
            retry_count=chosen.retry_count,
            retry_sleep_ms=chosen.retry_sleep_ms,
            rate_limited=chosen.rate_limited,
            cost_source=cost_source,
            primary_cached_input_tokens=primary_cached_input_tokens,
            backup_cached_input_tokens=backup_cached_input_tokens,
            primary_observed_cached_input_tokens=(primary_result.cache_read_tokens_observed),
            backup_observed_cached_input_tokens=(
                backup_result.cache_read_tokens_observed if backup_result else None
            ),
            primary_routing_estimated_cost_usd=primary_routing_estimated_cost_usd,
            backup_routing_estimated_cost_usd=backup_routing_estimated_cost_usd,
            billed_cost_usd=billed_cost,
            physical_cost_usd=physical_cost,
            primary_cost_usd=primary_cost,
            backup_cost_usd=backup_cost,
            primary_physical_cost_usd=primary_physical_cost,
            backup_physical_cost_usd=backup_physical_cost,
            lp_status=decision.lp_status,
            lp_weights=decision.lp_weights,
            budget_usd=decision.budget_usd,
            reference_cost_usd=decision.reference_cost_usd,
            tier_mix=decision.tier_mix,
            notes=decision.notes,
            model=ctx_model,
            hedge_algorithm=hedge_algorithm,
            hedge_schedule=hedge_schedule,
        )
        # routing_estimated_cost_usd: sum of primary + backup LP estimates.
        # The recorder receives these as kwargs; sum them so default metrics can
        # ask "what did the LP think this request would cost?" without poking at
        # per-leg estimates.
        if (
            primary_routing_estimated_cost_usd is not None
            or backup_routing_estimated_cost_usd is not None
        ):
            routing_estimated_cost_usd: float | None = float(
                (primary_routing_estimated_cost_usd or 0.0)
                + (backup_routing_estimated_cost_usd or 0.0)
            )
        else:
            routing_estimated_cost_usd = None
        record = PerRequestRecord(
            request_id=req_id,
            elapsed_sec=max(row_ts - self._start_ts, 0.0),
            policy=policy,
            prompt_tokens=ctx_prompt_tokens,
            completion_tokens_budget=ctx_max_tokens,
            completion_tokens_actual=(
                chosen.completion_tokens if chosen.status == "success" else None
            ),
            primary_provider=decision.primary or "",
            primary_tier=resolved_primary_tier,
            final_provider=chosen.provider,
            final_tier=resolved_final_tier,
            backup_provider=decision.hedge,
            backup_tier=resolved_backup_tier,
            model=ctx_model,
            source="real_eval",
            timestamp_sec=row_ts,
            ttft_ms=record_ttft_ms,
            e2e_ms=record_e2e_ms,
            primary_local_ttft_ms=primary_result.ttft_ms,
            backup_local_ttft_ms=backup_result.ttft_ms if backup_result else None,
            slo_ms=slo_ms,
            slo_violated=slo_violated,
            # Canonical accounting cost (= billed for real-eval).
            total_cost_usd=billed_cost,
            primary_cost_usd=primary_cost,
            backup_cost_usd=backup_cost if backup_result else None,
            # Routing-time LP cost estimates (promoted from metadata).
            routing_estimated_cost_usd=routing_estimated_cost_usd,
            primary_routing_estimated_cost_usd=primary_routing_estimated_cost_usd,
            backup_routing_estimated_cost_usd=backup_routing_estimated_cost_usd,
            # Upstream/physical cost (promoted from metadata).
            physical_cost_usd=physical_cost,
            primary_physical_cost_usd=primary_physical_cost,
            backup_physical_cost_usd=backup_physical_cost if backup_result else None,
            hedge_triggered=hedge_triggered,
            hedge_delay_ms=hedge_delay_ms,
            hedge_winner=hedge_winner,
            hedge_algorithm=hedge_algorithm,
            hedge_schedule=hedge_schedule,
            # LP state (promoted from metadata; lp_status normalized to canonical enum).
            lp_weights=decision.lp_weights,
            lp_budget_usd=decision.budget_usd,
            lp_status=_normalize_lp_status(decision.lp_status),
            status=status,
            error_class=chosen.error_message or None,
            metadata={
                # Transport / HTTP layer
                "real_retry_count": chosen.retry_count,
                "real_rate_limit_count": 1 if chosen.rate_limited else 0,
                "real_http_status": chosen.http_status,
                "real_transport": transport,
                "real_retry_sleep_ms": chosen.retry_sleep_ms,
                "real_status": chosen.status,
                "real_cost_source": cost_source,
                # LP-side diagnostics not yet promoted to canonical
                "real_reference_cost_usd": decision.reference_cost_usd,
                "real_tier_mix": decision.tier_mix,
                # Cache tracking
                "real_primary_cached_input_tokens": primary_cached_input_tokens,
                "real_backup_cached_input_tokens": backup_cached_input_tokens,
                "real_primary_observed_cached_input_tokens": (
                    primary_result.cache_read_tokens_observed
                ),
                "real_backup_observed_cached_input_tokens": (
                    backup_result.cache_read_tokens_observed if backup_result else None
                ),
                # Dispatch timing diagnostics
                "real_hedge_checkpoint_ts": hedge_checkpoint_ts,
                "real_primary_start_ts": primary_start_ts,
                "real_backup_dispatch_ts": backup_dispatch_ts,
                "real_backup_start_ts": backup_start_ts,
                "real_primary_first_token_ts": primary_result.first_token_ts,
                "real_backup_first_token_ts": (
                    backup_result.first_token_ts if backup_result else None
                ),
                "real_actual_dispatch_overhead_ms": actual_dispatch_overhead_ms,
                "real_checkpoint_dispatch_overhead_ms": checkpoint_dispatch_overhead_ms,
                "real_backup_dispatch_overhead_ms": backup_dispatch_overhead_ms,
            },
        )
        self.write_row(row, record=record)

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
        primary_tier: str | None = None,
        backup_tier: str | None = None,
        final_tier: str | None = None,
        slo_ms: float | None = None,
        transport: str | None = None,
        primary_cached_input_tokens: int = 0,
        backup_cached_input_tokens: int = 0,
        primary_routing_estimated_cost_usd: float | None = None,
        backup_routing_estimated_cost_usd: float | None = None,
        ts: float | None = None,
        hedge_algorithm: str | None = "probability_target",
        hedge_schedule: str | None = "slo_relative_checkpoints",
        ctx_model: str | None = None,
    ) -> None:
        """Convenience wrapper that unpacks a ``HedgedResult``.

        Defaults to ``"probability_target"`` / ``"slo_relative_checkpoints"``
        since every call site for hedged paths today uses that combination.
        """
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
            hedge_delay_sec=(
                hedged.hedge_delay_sec
                if hedged.hedge_delay_sec is not None
                else hedge_delay_sec
            ),
            chosen_result=hedged.chosen_result,
            tier=tier,
            primary_tier=primary_tier,
            backup_tier=backup_tier,
            final_tier=final_tier,
            slo_ms=slo_ms,
            transport=transport,
            primary_cached_input_tokens=primary_cached_input_tokens,
            backup_cached_input_tokens=backup_cached_input_tokens,
            primary_routing_estimated_cost_usd=primary_routing_estimated_cost_usd,
            backup_routing_estimated_cost_usd=backup_routing_estimated_cost_usd,
            hedge_checkpoint_ts=hedged.hedge_checkpoint_ts,
            backup_dispatch_ts=hedged.backup_dispatch_ts,
            ts=ts,
            hedge_algorithm=hedge_algorithm,
            hedge_schedule=hedge_schedule,
            ctx_model=ctx_model,
        )

    def to_run(self, policy: str | None = None, *, slo_ms: float | None = None) -> Run:
        """Return recorded requests as a canonical ``Run``."""
        with self._lock:
            rows = list(self._rows)
            records = list(self._records)
        if len(records) != len(rows):
            records = [self._record_from_row(row, slo_ms=slo_ms) for row in rows]
        elif slo_ms is not None:
            records = [_with_slo(record, slo_ms) for record in records]
        if policy is not None:
            records = [record for record in records if record.policy == policy]
        return Run(records=records, policy=policy or "", source="real_eval")

    def _record_from_row(
        self,
        row: RequestLogRow,
        *,
        slo_ms: float | None = None,
    ) -> PerRequestRecord:
        status = _canonical_status_from_row(row)
        # routing_estimated_cost_usd: sum primary + backup LP estimates from row.
        if (
            row.primary_routing_estimated_cost_usd is not None
            or row.backup_routing_estimated_cost_usd is not None
        ):
            routing_estimated_cost_usd: float | None = float(
                (row.primary_routing_estimated_cost_usd or 0.0)
                + (row.backup_routing_estimated_cost_usd or 0.0)
            )
        else:
            routing_estimated_cost_usd = None
        return PerRequestRecord(
            request_id=row.req_id,
            elapsed_sec=max(row.ts - self._start_ts, 0.0),
            policy=row.policy,
            prompt_tokens=row.prompt_tokens,
            completion_tokens_budget=row.max_tokens,
            completion_tokens_actual=None,
            primary_provider=row.primary_provider or "",
            primary_tier=row.tier or "",
            final_provider=row.actual_provider,
            final_tier=row.tier or "",
            backup_provider=row.backup_provider,
            backup_tier=None,
            model=row.model,
            source="real_eval",
            timestamp_sec=row.ts,
            ttft_ms=row.ttft_ms,
            e2e_ms=row.e2e_ms,
            primary_local_ttft_ms=row.ttft_ms,
            backup_local_ttft_ms=None,
            slo_ms=slo_ms,
            slo_violated=(row.ttft_ms > slo_ms if slo_ms is not None else False)
            or status != Status.SUCCESS,
            # Canonical accounting cost (= billed for real-eval).
            total_cost_usd=row.billed_cost_usd,
            primary_cost_usd=row.primary_cost_usd,
            backup_cost_usd=row.backup_cost_usd if row.backup_provider else None,
            # Routing-time LP cost estimates.
            routing_estimated_cost_usd=routing_estimated_cost_usd,
            primary_routing_estimated_cost_usd=row.primary_routing_estimated_cost_usd,
            backup_routing_estimated_cost_usd=row.backup_routing_estimated_cost_usd,
            # Upstream / physical cost.
            physical_cost_usd=row.physical_cost_usd,
            primary_physical_cost_usd=row.primary_physical_cost_usd,
            backup_physical_cost_usd=(
                row.backup_physical_cost_usd if row.backup_provider else None
            ),
            hedge_triggered=row.hedge_triggered,
            hedge_delay_ms=row.hedge_delay_ms,
            hedge_winner=row.hedge_winner,
            # Policy-level hedge identity is now persisted on the row, so a CSV
            # reload recovers it exactly instead of guessing from backup presence.
            hedge_algorithm=row.hedge_algorithm,
            hedge_schedule=row.hedge_schedule,
            # LP state (lp_status normalized to canonical enum).
            lp_weights=row.lp_weights,
            lp_budget_usd=row.budget_usd,
            lp_status=_normalize_lp_status(row.lp_status),
            status=status,
            error_class=row.error_message,
            metadata={
                # Transport / HTTP layer
                "real_retry_count": row.retry_count,
                "real_rate_limit_count": 1 if row.rate_limited else 0,
                "real_http_status": row.http_status,
                "real_transport": row.transport,
                "real_retry_sleep_ms": row.retry_sleep_ms,
                "real_status": row.status,
                "real_cost_source": row.cost_source,
                # LP-side diagnostics not yet promoted to canonical
                "real_reference_cost_usd": row.reference_cost_usd,
                "real_tier_mix": row.tier_mix,
                # Cache tracking
                "real_primary_cached_input_tokens": row.primary_cached_input_tokens,
                "real_backup_cached_input_tokens": row.backup_cached_input_tokens,
                "real_primary_observed_cached_input_tokens": (
                    row.primary_observed_cached_input_tokens
                ),
                "real_backup_observed_cached_input_tokens": (
                    row.backup_observed_cached_input_tokens
                ),
                # Dispatch timing diagnostics
                "real_hedge_checkpoint_ts": row.hedge_checkpoint_ts,
                "real_primary_start_ts": row.primary_start_ts,
                "real_backup_dispatch_ts": row.backup_dispatch_ts,
                "real_backup_start_ts": row.backup_start_ts,
                "real_primary_first_token_ts": row.primary_first_token_ts,
                "real_backup_first_token_ts": row.backup_first_token_ts,
                "real_actual_dispatch_overhead_ms": row.actual_dispatch_overhead_ms,
                "real_checkpoint_dispatch_overhead_ms": (
                    row.checkpoint_dispatch_overhead_ms
                ),
                "real_backup_dispatch_overhead_ms": row.backup_dispatch_overhead_ms,
            },
        )

    def write_summary(
        self,
        slo_ms: float,
        *,
        fixed_cost_usd_by_policy: dict[str, float] | None = None,
    ) -> Path:
        """Write per-policy aggregates to ``summary.json`` and return the path."""
        with self._lock:
            rows = list(self._rows)
            records = list(self._records)
        if len(records) != len(rows):
            records = [self._record_from_row(row, slo_ms=slo_ms) for row in rows]
        else:
            records = [_with_slo(record, slo_ms) for record in records]

        per_policy_records: dict[str, list[PerRequestRecord]] = {}
        for record in records:
            per_policy_records.setdefault(record.policy, []).append(record)

        summary: dict[str, dict[str, Any]] = {}
        for policy, policy_records in per_policy_records.items():
            run = Run(records=policy_records, policy=policy, source="real_eval")
            ttfts = [record.ttft_ms for record in policy_records if record.status == Status.SUCCESS]
            e2es = [
                float(record.e2e_ms)
                for record in policy_records
                if record.status == Status.SUCCESS and record.e2e_ms is not None
            ]
            n_total = len(policy_records)
            n_hedge_triggered = sum(record.hedge_triggered for record in policy_records)
            n_hedge_won_by_backup = sum(
                record.hedge_winner == "backup"
                for record in policy_records
                if record.hedge_triggered
            )
            n_success = sum(record.status == Status.SUCCESS for record in policy_records)
            n_rate_limited = sum(
                int(record.metadata.get("real_rate_limit_count") or 0) > 0
                for record in policy_records
            )
            total_retry_count = sum(
                int(record.metadata.get("real_retry_count") or 0) for record in policy_records
            )
            total_retry_sleep_ms = sum(
                float(record.metadata.get("real_retry_sleep_ms") or 0.0)
                for record in policy_records
            )
            success_rate = n_success / n_total if n_total else 0.0
            slo_violation_rate = run.slo_violation_rate()
            hedge_trigger_rate = n_hedge_triggered / n_total if n_total else 0.0
            hedge_backup_win_rate = (
                n_hedge_won_by_backup / n_hedge_triggered if n_hedge_triggered else 0.0
            )
            total_physical_cost_usd = sum(
                float(record.physical_cost_usd or 0.0) for record in policy_records
            )
            actual_dispatch_overheads = [
                float(value)
                for record in policy_records
                for value in [record.metadata.get("real_actual_dispatch_overhead_ms")]
                if value is not None
            ]
            checkpoint_dispatch_overheads = [
                float(value)
                for record in policy_records
                for value in [record.metadata.get("real_checkpoint_dispatch_overhead_ms")]
                if value is not None
            ]
            backup_dispatch_overheads = [
                float(value)
                for record in policy_records
                for value in [record.metadata.get("real_backup_dispatch_overhead_ms")]
                if value is not None
            ]
            total_cost_usd = run.total_cost_usd() + (
                fixed_cost_usd_by_policy or {}
            ).get(policy, 0.0)
            summary[policy] = {
                "n_total": n_total,
                "n_success": n_success,
                "n_slo_violation": round(slo_violation_rate * n_total),
                "n_hedge_triggered": n_hedge_triggered,
                "n_hedge_won_by_backup": n_hedge_won_by_backup,
                "n_rate_limited": n_rate_limited,
                "total_retry_count": total_retry_count,
                "total_retry_sleep_ms": round(total_retry_sleep_ms, 3),
                "total_cost_usd": round(total_cost_usd, 8),
                "total_physical_cost_usd": round(total_physical_cost_usd, 8),
                "success_rate": round(success_rate, 6),
                "slo_violation_rate": round(slo_violation_rate, 6),
                "hedge_trigger_rate": round(hedge_trigger_rate, 6),
                "hedge_backup_win_rate": round(hedge_backup_win_rate, 6),
                "mean_cost_usd": round(total_cost_usd / n_total if n_total else 0.0, 8),
                "mean_physical_cost_usd": round(
                    total_physical_cost_usd / n_total if n_total else 0.0,
                    8,
                ),
                "ttft_ms_p50": _percentile(ttfts, 50.0),
                "ttft_ms_p90": _percentile(ttfts, 90.0),
                "ttft_ms_p99": _percentile(ttfts, 99.0),
                "ttft_ms_mean": _mean(ttfts),
                "e2e_ms_p50": _percentile(e2es, 50.0),
                "e2e_ms_p90": _percentile(e2es, 90.0),
                "e2e_ms_p99": _percentile(e2es, 99.0),
                "e2e_ms_mean": _mean(e2es),
                "actual_dispatch_overhead_ms_n": len(actual_dispatch_overheads),
                "actual_dispatch_overhead_ms_mean": _mean(actual_dispatch_overheads),
                "actual_dispatch_overhead_ms_p50": _percentile(actual_dispatch_overheads, 50.0),
                "actual_dispatch_overhead_ms_p90": _percentile(actual_dispatch_overheads, 90.0),
                "actual_dispatch_overhead_ms_p99": _percentile(actual_dispatch_overheads, 99.0),
                "checkpoint_dispatch_overhead_ms_n": len(checkpoint_dispatch_overheads),
                "checkpoint_dispatch_overhead_ms_mean": _mean(checkpoint_dispatch_overheads),
                "checkpoint_dispatch_overhead_ms_p50": _percentile(
                    checkpoint_dispatch_overheads,
                    50.0,
                ),
                "checkpoint_dispatch_overhead_ms_p90": _percentile(
                    checkpoint_dispatch_overheads,
                    90.0,
                ),
                "checkpoint_dispatch_overhead_ms_p99": _percentile(
                    checkpoint_dispatch_overheads,
                    99.0,
                ),
                "backup_dispatch_overhead_ms_n": len(backup_dispatch_overheads),
                "backup_dispatch_overhead_ms_mean": _mean(backup_dispatch_overheads),
                "backup_dispatch_overhead_ms_p50": _percentile(backup_dispatch_overheads, 50.0),
                "backup_dispatch_overhead_ms_p90": _percentile(backup_dispatch_overheads, 90.0),
                "backup_dispatch_overhead_ms_p99": _percentile(backup_dispatch_overheads, 99.0),
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


def _format_ts(value: float | None) -> str:
    return f"{value:.6f}" if value is not None else ""


def _format_ms(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else ""


def _actual_dispatch_overhead_ms(
    *,
    primary_start_ts: float | None,
    backup_start_ts: float | None,
    hedge_delay_ms: float | None,
) -> float | None:
    if primary_start_ts is None or backup_start_ts is None or hedge_delay_ms is None:
        return None
    return (backup_start_ts - primary_start_ts) * 1000.0 - hedge_delay_ms


def _checkpoint_dispatch_overhead_ms(
    *,
    hedge_checkpoint_ts: float | None,
    backup_start_ts: float | None,
) -> float | None:
    if hedge_checkpoint_ts is None or backup_start_ts is None:
        return None
    return (backup_start_ts - hedge_checkpoint_ts) * 1000.0


def _backup_dispatch_overhead_ms(
    *,
    backup_dispatch_ts: float | None,
    backup_start_ts: float | None,
) -> float | None:
    if backup_dispatch_ts is None or backup_start_ts is None:
        return None
    return (backup_start_ts - backup_dispatch_ts) * 1000.0


def _normalize_lp_status(raw: str | None) -> str | None:
    """Normalize policy-emitted ``lp_status`` to the canonical record enum.

    Canonical values are:
    ``"feasible" | "single_candidate" | "all_over_budget" | "no_candidates"``.

    Policy emits: ``"optimal"``, ``"fallback_in_budget_{label}"``,
    ``"fallback_no_budget_{label}"``, or default ``"n/a"``. The original raw
    value remains on the CSV row (``RequestLogRow.lp_status``) for forensic
    use; only the canonical ``PerRequestRecord.lp_status`` is normalized.
    """
    if raw is None or raw == "n/a":
        return None
    if raw == "optimal" or raw.startswith("fallback_in_budget"):
        return "feasible"
    if raw.startswith("fallback_no_budget"):
        return "all_over_budget"
    return None


def _canonical_status(result: SingleRequestResult) -> Status:
    if result.status == "success":
        return Status.SUCCESS
    if result.rate_limited or result.http_status == 429:
        return Status.RATE_LIMITED
    return Status.ERROR


def _canonical_status_from_row(row: RequestLogRow) -> Status:
    if row.status == "success":
        return Status.SUCCESS
    if row.rate_limited or row.http_status == 429:
        return Status.RATE_LIMITED
    return Status.ERROR


def _with_slo(record: PerRequestRecord, slo_ms: float) -> PerRequestRecord:
    return PerRequestRecord(
        **{
            **record.__dict__,
            "slo_ms": slo_ms,
            "slo_violated": record.status != Status.SUCCESS or record.ttft_ms > slo_ms,
        }
    )


def _user_visible_ttft_ms(
    *,
    primary_result: SingleRequestResult,
    chosen_result: SingleRequestResult,
    hedge_delay_ms: float | None,
) -> float:
    if primary_result.start_ts > 0 and chosen_result.first_token_ts:
        return max((chosen_result.first_token_ts - primary_result.start_ts) * 1000.0, 0.0)
    if chosen_result is not primary_result and hedge_delay_ms is not None:
        return hedge_delay_ms + chosen_result.ttft_ms
    return chosen_result.ttft_ms


def _user_visible_e2e_ms(
    *,
    primary_result: SingleRequestResult,
    chosen_result: SingleRequestResult,
    hedge_delay_ms: float | None,
) -> float:
    if primary_result.start_ts > 0 and chosen_result.start_ts > 0:
        chosen_end_ts = chosen_result.start_ts + chosen_result.e2e_ms / 1000.0
        return max((chosen_end_ts - primary_result.start_ts) * 1000.0, 0.0)
    if chosen_result is not primary_result and hedge_delay_ms is not None:
        return hedge_delay_ms + chosen_result.e2e_ms
    return chosen_result.e2e_ms


__all__ = [
    "CSV_FIELDS",
    "Recorder",
    "RequestLogRow",
]
