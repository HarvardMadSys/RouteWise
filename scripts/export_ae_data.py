"""Export the artifact's recorded evaluation data from the private run archives.

Two exports, both reduced to the fields the reproduction scripts read:

``real-eval-records``
    The 24-hour live-provider replay (10 policies x 14,233 requests). Keeps
    the 17 request-level columns that ``plots.end_to_end.plot_real_world_frontier``
    consumes, plus a minimal ``args.json`` per policy carrying the provider
    inventory reference and SLO. Free-text columns (notes, error messages),
    per-leg timing internals, and LP weights are not exported.

``prod-trace``
    The PROD agentic workload sample behind Figure 8. Keeps the numeric and
    categorical fields the simulator loader consumes; drops user names, user
    e-mails, and error strings; replaces the account identifier with a stable
    per-account pseudonym so prefix-cache locality is preserved.

Both commands write a SHA256SUMS file next to the exported data. The
private source directories are not part of the repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
RECORD_ARGS_KEYS = ("policy", "inventory", "trace", "slo_ms", "max_requests", "duration_sec")

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


def export_real_eval_records(source: Path, dest: Path) -> int:
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
            if user_id:
                exported["user_id"] = pseudonyms.setdefault(
                    str(user_id), f"account-{len(pseudonyms) + 1:02d}"
                )
            out.write(json.dumps(exported, sort_keys=True) + "\n")
            count += 1
    _write_checksums(dest.parent, [dest])
    print(f"exported {count} requests ({len(pseudonyms)} pseudonymous accounts) to {dest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    records = subparsers.add_parser("real-eval-records", help="Export the 24h replay records.")
    records.add_argument("--source", type=Path, required=True, help="Private run directory.")
    records.add_argument("--dest", type=Path, default=ROOT_DIR / "data" / "real_eval_records")

    trace = subparsers.add_parser("prod-trace", help="Export the de-identified PROD trace.")
    trace.add_argument("--source", type=Path, required=True, help="Private trace JSONL.")
    trace.add_argument("--dest", type=Path, default=ROOT_DIR / "data" / "freeinference.jsonl")

    args = parser.parse_args(argv)
    if args.command == "real-eval-records":
        return export_real_eval_records(args.source, args.dest)
    return export_prod_trace(args.source, args.dest)


if __name__ == "__main__":
    raise SystemExit(main())
