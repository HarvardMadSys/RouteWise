"""Streaming aggregation for run-level metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from routewise.metrics.histogram import TtftHistogram
from routewise.metrics.run import Run, RunAggregate

if TYPE_CHECKING:
    from routewise.metrics.record import PerRequestRecord


@dataclass
class RunAggregator:
    """Consume records once and build a compact Run.

    When retain_records=True this intentionally does both jobs: keep exact
    records for debugging/goldens and maintain the streaming aggregate for
    consistency with paper-run paths.
    """

    policy: str = ""
    scenario_name: str = ""
    source: str = "sim"
    retain_records: bool = True
    window_sec: float = 300.0
    records: list[PerRequestRecord] = field(default_factory=list)
    aggregate: RunAggregate = field(default_factory=RunAggregate)

    def observe(self, record: PerRequestRecord) -> None:
        """Update aggregate state from one record."""
        if self.retain_records:
            self.records.append(record)

        agg = self.aggregate
        agg.n += 1
        agg.ttft_histogram.add(record.ttft_ms)
        if record.e2e_ms is not None:
            if agg.e2e_histogram is None:
                agg.e2e_histogram = TtftHistogram.default()
            agg.e2e_histogram.add(float(record.e2e_ms))

        agg.total_cost_usd += float(record.total_cost_usd)
        agg.cost_count += 1
        agg.status_counts[record.status.value] += 1
        if record.slo_violated:
            agg.slo_violated_count += 1

        if record.primary_tier:
            agg.cost_by_tier[record.primary_tier] += float(record.primary_cost_usd)
        agg.cost_by_provider[record.primary_provider] += float(record.primary_cost_usd)

        if record.backup_cost_usd is not None and record.backup_provider:
            agg.cost_by_provider[record.backup_provider] += float(record.backup_cost_usd)
            if record.backup_tier:
                agg.cost_by_tier[record.backup_tier] += float(record.backup_cost_usd)

        agg.provider_counts[record.final_provider] += 1
        if record.final_tier:
            agg.tier_counts[record.final_tier] += 1

        agg.hedge_total_count += 1
        if record.hedge_triggered:
            agg.hedge_triggered_count += 1
            if record.hedge_winner:
                agg.hedge_winner_counts[record.hedge_winner] += 1

        window_id = int(float(record.elapsed_sec) // self.window_sec)
        agg.provider_window_counts.setdefault(window_id, Counter())[record.final_provider] += 1
        if record.final_tier:
            agg.tier_window_counts.setdefault(window_id, Counter())[record.final_tier] += 1

    def finalize(self) -> Run:
        """Build a Run from the accumulated state."""
        return Run(
            records=self.records if self.retain_records else [],
            policy=self.policy,
            scenario_name=self.scenario_name,
            source=self.source,
            aggregate=self.aggregate,
            window_sec=self.window_sec,
        )


def aggregate_records(
    records: list[PerRequestRecord],
    *,
    policy: str = "",
    scenario_name: str = "",
    source: str = "sim",
) -> Run:
    """Build a Run by streaming over an existing record list."""
    aggregator = RunAggregator(
        policy=policy,
        scenario_name=scenario_name,
        source=source,
        retain_records=True,
    )
    for record in records:
        aggregator.observe(record)
    return aggregator.finalize()

__all__ = ["RunAggregator", "aggregate_records"]
