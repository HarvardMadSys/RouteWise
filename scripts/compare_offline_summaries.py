"""Compare offline rows from two RouteWise summary.csv files.

This is a verification helper for refactors that should preserve cost-layer
offline baseline results. It intentionally compares only summary rows for one
policy, defaulting to ``offline``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_KEY_COLUMNS = ("scenario", "policy", "seeds")
DEFAULT_STRUCTURED_COLUMNS = ("provider_mix", "tier_mix")
DEFAULT_EXACT_COLUMNS = (
    "public_scenario",
    "artifact_label",
    "subscription_plan",
    "subscription_plan_display_name",
    "concurrency_plan",
    "concurrency_plan_display_name",
    "model",
    "model_class",
    "subscription_cost_known",
    "cost_claim_allowed",
    "trace_paper_grade",
    "quota_fits_in_trace",
    "concurrency_saturated_in_trace",
    "percentile_source",
)


@dataclass(frozen=True)
class Mismatch:
    """One comparison failure."""

    key: tuple[str, ...]
    column: str
    expected: str
    actual: str
    detail: str


@dataclass
class CompareStats:
    """Aggregated comparison counters."""

    rows: int = 0
    numeric_cells: int = 0
    structured_cells: int = 0
    exact_cells: int = 0
    max_abs_diff: float = 0.0
    max_rel_diff: float = 0.0
    worst_numeric: tuple[tuple[str, ...], str] | None = None


def _summary_path(path: Path) -> Path:
    if path.is_dir():
        path = path / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _read_rows(path: Path, *, policy: str) -> list[dict[str, str]]:
    with _summary_path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row.get("policy") == policy]


def _key(row: dict[str, str], key_columns: tuple[str, ...]) -> tuple[str, ...]:
    missing = [column for column in key_columns if column not in row]
    if missing:
        raise KeyError(f"missing key column {missing[0]!r}")
    return tuple(row[column] for column in key_columns)


def _index_rows(
    rows: list[dict[str, str]],
    *,
    key_columns: tuple[str, ...],
) -> dict[tuple[str, ...], dict[str, str]]:
    indexed: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = _key(row, key_columns)
        if key in indexed:
            raise ValueError(f"duplicate row key {key!r}")
        indexed[key] = row
    return indexed


def _parse_float(value: str) -> float | None:
    value = value.strip()
    if not value or value in {"True", "False"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _is_numeric_column(rows: list[dict[str, str]], column: str) -> bool:
    saw_numeric = False
    for row in rows:
        value = row.get(column, "").strip()
        if not value:
            continue
        parsed = _parse_float(value)
        if parsed is None:
            return False
        saw_numeric = True
    return saw_numeric


def _numeric_columns(
    baseline_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    *,
    key_columns: tuple[str, ...],
    structured_columns: tuple[str, ...],
    exact_columns: tuple[str, ...],
) -> tuple[str, ...]:
    common_columns = set().union(
        *(set(row) for row in baseline_rows + candidate_rows),
    )
    excluded = set(key_columns) | set(structured_columns) | set(exact_columns)
    rows = baseline_rows + candidate_rows
    return tuple(
        sorted(
            column
            for column in common_columns - excluded
            if _is_numeric_column(rows, column)
        )
    )


def _compare_float(
    expected: float,
    actual: float,
    *,
    rel_tol: float,
    abs_tol: float,
) -> tuple[bool, float, float]:
    if math.isnan(expected) and math.isnan(actual):
        return True, 0.0, 0.0
    abs_diff = abs(actual - expected)
    scale = max(abs(expected), abs(actual), 1e-300)
    rel_diff = abs_diff / scale
    return abs_diff <= abs_tol or rel_diff <= rel_tol, abs_diff, rel_diff


def _parse_json_map(value: str) -> dict[str, float]:
    parsed = json.loads(value or "{}")
    if not isinstance(parsed, dict):
        raise ValueError("expected JSON object")
    return {str(key): float(raw_value) for key, raw_value in parsed.items()}


def _compare_structured_map(
    *,
    key: tuple[str, ...],
    column: str,
    expected_raw: str,
    actual_raw: str,
    rel_tol: float,
    abs_tol: float,
) -> list[Mismatch]:
    expected = _parse_json_map(expected_raw)
    actual = _parse_json_map(actual_raw)
    if set(expected) != set(actual):
        return [
            Mismatch(
                key=key,
                column=column,
                expected=json.dumps(expected, sort_keys=True),
                actual=json.dumps(actual, sort_keys=True),
                detail="map keys differ",
            )
        ]
    mismatches: list[Mismatch] = []
    for map_key in sorted(expected):
        ok, abs_diff, rel_diff = _compare_float(
            expected[map_key],
            actual[map_key],
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
        if not ok:
            mismatches.append(
                Mismatch(
                    key=key,
                    column=f"{column}.{map_key}",
                    expected=f"{expected[map_key]:.17g}",
                    actual=f"{actual[map_key]:.17g}",
                    detail=f"abs_diff={abs_diff:.3g} rel_diff={rel_diff:.3g}",
                )
            )
    return mismatches


def compare_summaries(
    baseline_path: Path,
    candidate_path: Path,
    *,
    policy: str,
    key_columns: tuple[str, ...],
    structured_columns: tuple[str, ...],
    exact_columns: tuple[str, ...],
    rel_tol: float,
    abs_tol: float,
) -> tuple[CompareStats, list[Mismatch]]:
    baseline_rows = _read_rows(baseline_path, policy=policy)
    candidate_rows = _read_rows(candidate_path, policy=policy)
    baseline = _index_rows(baseline_rows, key_columns=key_columns)
    candidate = _index_rows(candidate_rows, key_columns=key_columns)
    stats = CompareStats(rows=len(baseline))
    mismatches: list[Mismatch] = []

    missing = sorted(set(baseline) - set(candidate))
    extra = sorted(set(candidate) - set(baseline))
    for key in missing:
        mismatches.append(
            Mismatch(key=key, column="<row>", expected="present", actual="missing", detail="")
        )
    for key in extra:
        mismatches.append(
            Mismatch(key=key, column="<row>", expected="missing", actual="present", detail="")
        )
    shared_keys = sorted(set(baseline) & set(candidate))

    numeric_columns = _numeric_columns(
        [baseline[key] for key in shared_keys],
        [candidate[key] for key in shared_keys],
        key_columns=key_columns,
        structured_columns=structured_columns,
        exact_columns=exact_columns,
    )

    for key in shared_keys:
        expected_row = baseline[key]
        actual_row = candidate[key]
        for column in numeric_columns:
            expected_raw = expected_row.get(column, "")
            actual_raw = actual_row.get(column, "")
            expected = _parse_float(expected_raw)
            actual = _parse_float(actual_raw)
            if expected is None or actual is None:
                if expected_raw != actual_raw:
                    mismatches.append(
                        Mismatch(
                            key=key,
                            column=column,
                            expected=expected_raw,
                            actual=actual_raw,
                            detail="numeric value missing on one side",
                        )
                    )
                continue
            ok, abs_diff, rel_diff = _compare_float(
                expected,
                actual,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
            stats.numeric_cells += 1
            if abs_diff > stats.max_abs_diff or rel_diff > stats.max_rel_diff:
                stats.max_abs_diff = max(stats.max_abs_diff, abs_diff)
                stats.max_rel_diff = max(stats.max_rel_diff, rel_diff)
                stats.worst_numeric = (key, column)
            if not ok:
                mismatches.append(
                    Mismatch(
                        key=key,
                        column=column,
                        expected=expected_raw,
                        actual=actual_raw,
                        detail=f"abs_diff={abs_diff:.3g} rel_diff={rel_diff:.3g}",
                    )
                )

        for column in structured_columns:
            if column not in expected_row or column not in actual_row:
                continue
            stats.structured_cells += 1
            mismatches.extend(
                _compare_structured_map(
                    key=key,
                    column=column,
                    expected_raw=expected_row[column],
                    actual_raw=actual_row[column],
                    rel_tol=rel_tol,
                    abs_tol=abs_tol,
                )
            )

        for column in exact_columns:
            if column not in expected_row or column not in actual_row:
                continue
            stats.exact_cells += 1
            if expected_row[column] != actual_row[column]:
                mismatches.append(
                    Mismatch(
                        key=key,
                        column=column,
                        expected=expected_row[column],
                        actual=actual_row[column],
                        detail="exact field differs",
                    )
                )

    return stats, mismatches


def _split_columns(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Baseline summary.csv or output directory.")
    parser.add_argument("candidate", type=Path, help="Candidate summary.csv or output directory.")
    parser.add_argument("--policy", default="offline", help="Policy rows to compare.")
    parser.add_argument("--rel-tol", type=float, default=1e-9)
    parser.add_argument("--abs-tol", type=float, default=1e-12)
    parser.add_argument(
        "--key-cols",
        help=f"Comma-separated row key columns. Default: {','.join(DEFAULT_KEY_COLUMNS)}",
    )
    parser.add_argument(
        "--structured-cols",
        help=(
            "Comma-separated JSON map columns to compare numerically. "
            f"Default: {','.join(DEFAULT_STRUCTURED_COLUMNS)}"
        ),
    )
    parser.add_argument(
        "--exact-cols",
        help=(
            "Comma-separated non-numeric columns to compare exactly. "
            f"Default: {','.join(DEFAULT_EXACT_COLUMNS)}"
        ),
    )
    parser.add_argument("--max-mismatches", type=int, default=20)
    args = parser.parse_args()

    stats, mismatches = compare_summaries(
        args.baseline,
        args.candidate,
        policy=args.policy,
        key_columns=_split_columns(args.key_cols, DEFAULT_KEY_COLUMNS),
        structured_columns=_split_columns(args.structured_cols, DEFAULT_STRUCTURED_COLUMNS),
        exact_columns=_split_columns(args.exact_cols, DEFAULT_EXACT_COLUMNS),
        rel_tol=args.rel_tol,
        abs_tol=args.abs_tol,
    )
    summary: dict[str, Any] = {
        "policy": args.policy,
        "rows": stats.rows,
        "numeric_cells": stats.numeric_cells,
        "structured_cells": stats.structured_cells,
        "exact_cells": stats.exact_cells,
        "max_abs_diff": stats.max_abs_diff,
        "max_rel_diff": stats.max_rel_diff,
        "worst_numeric": (
            {"key": list(stats.worst_numeric[0]), "column": stats.worst_numeric[1]}
            if stats.worst_numeric is not None
            else None
        ),
        "mismatch_count": len(mismatches),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if mismatches:
        for mismatch in mismatches[: args.max_mismatches]:
            print(
                "MISMATCH "
                f"key={mismatch.key} column={mismatch.column} "
                f"expected={mismatch.expected!r} actual={mismatch.actual!r} "
                f"{mismatch.detail}"
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
