"""Export the artifact's recorded evaluation data from the private run archives.

Four exports, all reduced to the fields the reproduction scripts read:

``real-eval-records``
    The 24-hour live-provider replay (10 policies x 14,233 requests). Keeps
    the 17 request-level columns that ``plots.end_to_end.plot_real_world_frontier``
    consumes, plus a minimal ``args.json`` per policy carrying the provider
    inventory reference and SLO. Free-text columns (notes, error messages),
    per-leg timing internals, and LP weights are not exported.

``hedge-legs``
    Per-leg hedging columns for the RouteWise hedging policies of an already
    exported record set: each leg's time to first token and the losing leg's
    charge. These are what a hedging analysis needs and the request-level
    release columns do not carry.

``router-state``
    The per-decision router state of the RouteWise policies of an already
    exported record set: the candidate set, latency objectives, effective
    costs and budget the LP saw, the quota and concurrency occupancy at that
    moment, and the predicted output length. These are what the mechanism
    analysis reads.

``prod-trace``
    The PROD agentic workload sample behind Figure 8. Keeps the numeric and
    categorical fields the simulator loader consumes; drops user names, user
    e-mails, and error strings; replaces the account identifier with a stable
    per-account pseudonym so account grouping is preserved. Cache discounts
    use the trace's cache_read_tokens counters, which are retained unchanged.

Every command writes a SHA256SUMS file next to the exported data. The
private source directories are not part of the repository.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

RECORD_FIELDS = (
    "ts",
    "policy",
    "req_id",
    "prompt_tokens",
    "max_tokens",
    "primary_provider",
    "backup_provider",
    "actual_provider",
    "tier",
    "status",
    "ttft_ms",
    "e2e_ms",
    "billed_cost_usd",
    "physical_cost_usd",
    "hedge_triggered",
    "hedge_winner",
    "rate_limited",
)
RECORD_ARGS_KEYS = ("policy", "inventory", "slo_ms")

HEDGE_LEG_FIELDS = (
    "req_id",
    "hedge_triggered",
    "hedge_winner",
    "hedge_delay_ms",
    "primary_ttft_ms",
    "backup_ttft_ms",
    "loser_provider",
    "loser_billed_cost_usd",
    "loser_physical_cost_usd",
    "loser_estimated_cost_usd",
    "loser_cached_input_tokens",
    "primary_estimated_cost_usd",
)

# Per-decision router state, present only for RouteWise policies. Everything
# the mechanism analysis needs and the release columns do not: what the LP saw
# (candidates, latency objectives, effective costs, budget), the capacity
# state it saw it under, and the output length it predicted. The per-provider
# maps become one column per provider (``latency_objective_ms:<name>`` and so
# on) so the file stays small; the header is built from the run's inventory.
ROUTER_STATE_SCALAR_FIELDS = (
    "req_id",
    "predicted_output_tokens",
    "quota_fraction_used",
    "concurrency_in_flight",
    "unavailable",
    "c_min_usd",
    "budget_usd",
    "c_max_usd",
)
ROUTER_STATE_MAP_FIELDS = (
    ("latency_objective_ms", "latency_objective_ms", ".0f"),
    ("c_eff", "c_eff", ".4e"),
    ("lp_weights", "weight", ".4f"),
)

# Archived aggregates kept next to the records; checksummed with them.
REFERENCE_SUMMARIES = (
    "reference_summary.json",
    "hedging_reference_summary.json",
    "mechanisms_reference_summary.json",
)

TRACE_FIELDS = (
    "request_id",
    "timestamp",
    "model_id",
    "provider",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "total_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "cost_usd",
    "latency_ms",
    "ttft_ms",
    "status_code",
)


def _policy_name(name: str) -> str:
    """Map the archived run's policy names onto the plotting scripts' naming.

    The archived run labeled RouteWise operating points ``budget_range_p<N>_hedge``;
    the analysis scripts identify them as ``budget_range_alpha<N>_hedge``.
    """
    if name.startswith("budget_range_p") and name.endswith("_hedge"):
        return "budget_range_alpha" + name[len("budget_range_p") : -len("_hedge")] + "_hedge"
    return name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_checksums(root: Path, files: list[Path]) -> None:
    lines = [f"{_sha256(path)}  {path.relative_to(root).as_posix()}" for path in sorted(files)]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_real_eval_records(source: Path, dest: Path, inventory: str | None = None) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    policies = sorted(p.parent.name for p in source.glob("*/requests.csv"))
    if not policies:
        raise SystemExit(f"no policy directories with requests.csv under {source}")
    for policy in policies:
        exported_name = _policy_name(policy)
        policy_dir = dest / exported_name
        policy_dir.mkdir(exist_ok=True)
        with (
            (source / policy / "requests.csv").open(newline="", encoding="utf-8") as src,
            (policy_dir / "requests.csv").open("w", newline="", encoding="utf-8") as out,
        ):
            reader = csv.DictReader(src)
            missing = set(RECORD_FIELDS).difference(reader.fieldnames or ())
            if missing:
                raise SystemExit(f"{policy}: source is missing columns {sorted(missing)}")
            writer = csv.DictWriter(out, fieldnames=RECORD_FIELDS)
            writer.writeheader()
            for row in reader:
                row["policy"] = _policy_name(row["policy"])
                writer.writerow({key: row[key] for key in RECORD_FIELDS})
        written.append(policy_dir / "requests.csv")

        args = json.loads((source / policy / "args.json").read_text(encoding="utf-8"))
        kept = {key: args[key] for key in RECORD_ARGS_KEYS if key in args}
        if "policy" in kept:
            kept["policy"] = exported_name
        if inventory is not None and "inventory" in kept:
            kept["inventory"] = inventory
        (policy_dir / "args.json").write_text(json.dumps(kept, indent=2) + "\n", encoding="utf-8")
        written.append(policy_dir / "args.json")

    reference = source / "summary.json"
    if reference.exists():
        rows = json.loads(reference.read_text(encoding="utf-8"))
        for row in rows:
            row["policy"] = _policy_name(row["policy"])
        target = dest / "reference_summary.json"
        target.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        written.append(target)
    _write_checksums(dest, written)
    print(f"exported {len(policies)} policies to {dest}")
    return 0


def _leg_ttft_ms(row: dict[str, str], leg: str) -> str:
    """Observed time to first token of one leg, blank when it produced none.

    A losing leg is canceled once the winner answers. The runs record the
    loser's first token when it arrives before that, so this is empty only for
    a leg that never produced one.
    """
    start, first = row.get(f"{leg}_start_ts"), row.get(f"{leg}_first_token_ts")
    if start in {"", None} or first in {"", None}:
        return ""
    return f"{(float(first) - float(start)) * 1000.0:.3f}"


def export_hedge_legs(source: Path, dest: Path, policies: list[str] | None = None) -> int:
    """Add hedging leg columns to an existing exported record set."""
    wanted = set(policies) if policies else None
    written = 0
    for source_csv in sorted(source.glob("*/requests.csv")):
        policy = source_csv.parent.name
        if wanted is not None and policy not in wanted:
            continue
        policy_dir = dest / _policy_name(policy)
        if not policy_dir.is_dir():
            raise SystemExit(f"{policy_dir} does not exist; export the records first")
        seen: set[str] = set()
        with (
            source_csv.open(newline="", encoding="utf-8") as src,
            (policy_dir / "hedge_legs.csv").open("w", newline="", encoding="utf-8") as out,
        ):
            writer = csv.DictWriter(out, fieldnames=HEDGE_LEG_FIELDS)
            writer.writeheader()
            for row in csv.DictReader(src):
                req_id = row["req_id"]
                if req_id in seen:
                    raise SystemExit(f"{policy}: duplicate req_id {req_id!r}")
                seen.add(req_id)
                winner = row.get("hedge_winner") or ""
                loser = "backup" if winner == "primary" else "primary" if winner else ""
                writer.writerow(
                    {
                        "req_id": req_id,
                        "hedge_triggered": row.get("hedge_triggered") or "",
                        "hedge_winner": winner,
                        "hedge_delay_ms": row.get("hedge_delay_ms") or "",
                        "primary_ttft_ms": _leg_ttft_ms(row, "primary"),
                        "backup_ttft_ms": _leg_ttft_ms(row, "backup"),
                        "loser_provider": (row.get(f"{loser}_provider") or "" if loser else ""),
                        "loser_billed_cost_usd": (
                            row.get(f"{loser}_cost_usd") or "" if loser else ""
                        ),
                        "loser_physical_cost_usd": (
                            row.get(f"{loser}_physical_cost_usd") or "" if loser else ""
                        ),
                        # What the router predicted that leg would cost if it
                        # ran to completion; the upper bound on what cancelling
                        # it saved.
                        "loser_estimated_cost_usd": (
                            row.get(f"{loser}_routing_estimated_cost_usd") or "" if loser else ""
                        ),
                        "loser_cached_input_tokens": (
                            row.get(f"{loser}_cached_input_tokens") or "" if loser else ""
                        ),
                        # Present on every request, hedged or not, so the same
                        # estimator can be calibrated against real charges.
                        "primary_estimated_cost_usd": (
                            row.get("primary_routing_estimated_cost_usd") or ""
                        ),
                    }
                )
        written += 1
    if not written:
        raise SystemExit(f"no matching policy directories with requests.csv under {source}")
    files = sorted(dest.glob("*/*.csv")) + sorted(dest.glob("*/args.json"))
    reference = dest / "reference_summary.json"
    if reference.exists():
        files.append(reference)
    _write_checksums(dest, files)
    print(f"exported hedging legs for {written} policies to {dest}")
    return 0


def _single_value(raw: str) -> str:
    """Return the one value of a recorded ``{provider: number}`` map."""
    if raw in {"", None}:
        return ""
    values = json.loads(raw)
    if len(values) != 1:
        raise SystemExit(f"expected one capacity-limited provider, got {sorted(values)}")
    return f"{next(iter(values.values())):g}"


def export_router_state(source: Path, dest: Path, policies: list[str] | None = None) -> int:
    """Add the per-decision router state of the RouteWise runs to a record set."""
    wanted = set(policies) if policies else None
    written = 0
    for source_csv in sorted(source.glob("*/requests.csv")):
        policy = source_csv.parent.name
        if wanted is not None and policy not in wanted:
            continue
        policy_dir = dest / _policy_name(policy)
        if not policy_dir.is_dir():
            raise SystemExit(f"{policy_dir} does not exist; export the records first")
        with source_csv.open(newline="", encoding="utf-8") as src:
            rows = list(csv.DictReader(src))
        # Baselines carry no router state; skip them silently unless asked for.
        if not any(row.get("candidates") for row in rows):
            if wanted is not None:
                raise SystemExit(f"{policy}: requests.csv has no router-state columns")
            continue
        providers: list[str] = []
        for row in rows:
            for name in json.loads(row["latency_objective_ms"] or "{}"):
                if name not in providers:
                    providers.append(name)
        fields = list(ROUTER_STATE_SCALAR_FIELDS) + [
            f"{prefix}:{name}" for _, prefix, _ in ROUTER_STATE_MAP_FIELDS for name in providers
        ]
        seen: set[str] = set()
        # mtime=0 keeps the archive byte-identical across exports so the
        # checksums are reproducible.
        with (
            (policy_dir / "router_state.csv.gz").open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
            io.TextIOWrapper(compressed, encoding="utf-8", newline="") as out,
        ):
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                req_id = row["req_id"]
                if req_id in seen:
                    raise SystemExit(f"{policy}: duplicate req_id {req_id!r}")
                seen.add(req_id)
                candidates = set(json.loads(row["candidates"]) if row.get("candidates") else [])
                out_row = {
                    "req_id": req_id,
                    "predicted_output_tokens": row.get("predicted_output_tokens") or "",
                    "quota_fraction_used": _single_value(row.get("quota_fraction_used") or ""),
                    "concurrency_in_flight": _single_value(row.get("concurrency_in_flight") or ""),
                    # Providers the LP did not see for this decision (no free
                    # slot, quota exhausted, or cooling down after an error).
                    "unavailable": "|".join(name for name in providers if name not in candidates),
                    "c_min_usd": row.get("c_min_usd") or "",
                    "budget_usd": row.get("budget_usd") or "",
                    "c_max_usd": row.get("reference_cost_usd") or "",
                }
                for source_field, prefix, fmt in ROUTER_STATE_MAP_FIELDS:
                    values = json.loads(row.get(source_field) or "{}")
                    for name in providers:
                        value = values.get(name)
                        out_row[f"{prefix}:{name}"] = (
                            "" if value is None else format(float(value), fmt)
                        )
                writer.writerow(out_row)
        written += 1
    if not written:
        raise SystemExit(f"no policy directories with router-state columns under {source}")
    files = (
        sorted(dest.glob("*/*.csv"))
        + sorted(dest.glob("*/*.csv.gz"))
        + sorted(dest.glob("*/args.json"))
    )
    for name in REFERENCE_SUMMARIES:
        if (dest / name).exists():
            files.append(dest / name)
    _write_checksums(dest, files)
    print(f"exported router state for {written} policies to {dest}")
    return 0


def export_prod_trace(source: Path, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    pseudonyms: dict[str, str] = {}
    count = 0
    with source.open(encoding="utf-8") as src, dest.open("w", encoding="utf-8") as out:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            exported = {key: record.get(key) for key in TRACE_FIELDS}
            user_id = record.get("user_id")
            if user_id is not None and str(user_id).strip():
                exported["user_id"] = pseudonyms.setdefault(
                    str(user_id), f"account-{len(pseudonyms) + 1:02d}"
                )
            out.write(json.dumps(exported, sort_keys=True) + "\n")
            count += 1
    files = [dest]
    reference = dest.parent / "figure8_reference_summary.csv"
    if reference.exists():
        files.append(reference)
    _write_checksums(dest.parent, files)
    print(f"exported {count} requests ({len(pseudonyms)} pseudonymous accounts) to {dest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    records = subparsers.add_parser("real-eval-records", help="Export the 24h replay records.")
    records.add_argument("--source", type=Path, required=True, help="Private run directory.")
    records.add_argument("--dest", type=Path, default=ROOT_DIR / "data" / "real_eval_records")
    records.add_argument(
        "--inventory",
        default=None,
        help="Repository inventory path written to the exported args.json in place of the run's.",
    )

    legs = subparsers.add_parser("hedge-legs", help="Add hedging leg columns to a record set.")
    legs.add_argument("--source", type=Path, required=True, help="Private run directory.")
    legs.add_argument("--dest", type=Path, default=ROOT_DIR / "data" / "real_eval_records")
    legs.add_argument(
        "--policy",
        action="append",
        dest="policies",
        help="Only export this policy. Repeatable; defaults to every policy in --source.",
    )

    state = subparsers.add_parser(
        "router-state", help="Add the RouteWise per-decision router state to a record set."
    )
    state.add_argument("--source", type=Path, required=True, help="Private run directory.")
    state.add_argument("--dest", type=Path, default=ROOT_DIR / "data" / "real_eval_records")
    state.add_argument(
        "--policy",
        action="append",
        dest="policies",
        help="Only export this policy. Repeatable; defaults to every RouteWise policy in --source.",
    )

    trace = subparsers.add_parser("prod-trace", help="Export the de-identified PROD trace.")
    trace.add_argument("--source", type=Path, required=True, help="Private trace JSONL.")
    trace.add_argument("--dest", type=Path, default=ROOT_DIR / "data" / "freeinference.jsonl")

    args = parser.parse_args(argv)
    if args.command == "real-eval-records":
        return export_real_eval_records(args.source, args.dest, inventory=args.inventory)
    if args.command == "hedge-legs":
        return export_hedge_legs(args.source, args.dest, policies=args.policies)
    if args.command == "router-state":
        return export_router_state(args.source, args.dest, policies=args.policies)
    return export_prod_trace(args.source, args.dest)


if __name__ == "__main__":
    raise SystemExit(main())
