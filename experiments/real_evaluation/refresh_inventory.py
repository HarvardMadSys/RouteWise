"""Refresh OpenRouter sub-provider pricing for a real_evaluation inventory.

Hits the OpenRouter endpoints API and rewrites the OR_* providers in an
inventory JSON file. Run before each paper-grade experiment so prices are
not stale (OR sub-providers add/drop and adjust prices over time).

Usage::

    uv run python -m experiments.real_evaluation.refresh_inventory \\
        --model minimax/minimax-m2.5 \\
        --output experiments/real_evaluation/data/minimax_m25_or_only.json

Behavior when the output file already exists:

* Non-OR providers (transport != "openrouter") are preserved verbatim
  so the joint S_A + S_Q + S_C inventory is safe to refresh too — only
  OR_* entries get rewritten.
* ``primary_slo_ms`` and ``slo_thresholds_ms`` are preserved from the
  existing file (treated as operator-controlled).
* If ``openrouter_model_id`` in the existing file disagrees with
  ``--model`` we fail fast (operator can pass ``--force-model-change``
  to override and re-derive ``model_family``). Without this guard you
  could silently mix MiniMax/StepFun configs in the same file.
* Each OR provider entry is written with an explicit ``model`` field
  (= ``--model``); without it ``transports.resolve_transport_config()``
  falls back to a hardcoded model id.

When OR returns multiple tiers per sub-provider (e.g. ``Minimax/fp8``
and ``Minimax/highspeed``) we keep the cheaper one. Tier ranking uses
``input_price + output_price`` so a tier that is cheap on input but
expensive on output cannot win — for typical chat workloads output
tokens dominate cost.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

OR_ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{model_id}/endpoints"


def fetch_endpoints(model_id: str, *, timeout: float = 30.0) -> list[dict]:
    url = OR_ENDPOINTS_URL.format(model_id=model_id)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            f"OR API returned HTTP {exc.code} for {url}: {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"OR API request failed: {exc.reason}") from exc
    return payload["data"]["endpoints"]


def dedup_by_cheapest(endpoints: list[dict]) -> dict[str, dict]:
    """Keep the cheapest tier per provider_name.

    Ranking key is ``input_price + output_price`` so output-heavy tiers
    (which dominate chat-completion cost) cannot win on input price alone.
    """
    by_name: dict[str, dict] = {}
    for ep in endpoints:
        name = ep["provider_name"]
        in_p = float(ep["pricing"]["prompt"]) * 1_000_000
        out_p = float(ep["pricing"]["completion"]) * 1_000_000
        cache_read_raw = ep["pricing"].get("input_cache_read")
        cache_read_p = (
            float(cache_read_raw) * 1_000_000
            if cache_read_raw is not None
            else None
        )
        rank = in_p + out_p
        if name not in by_name or rank < by_name[name]["_rank"]:
            by_name[name] = {
                "_input": in_p,
                "_cached_input": (
                    cache_read_p if cache_read_p is not None and cache_read_p > 0 else None
                ),
                "_output": out_p,
                "_rank": rank,
                "_uptime": ep.get("uptime_last_1d"),
                "_tag": ep.get("tag"),
            }
    return by_name


def to_provider_entries(by_name: dict[str, dict], model_id: str) -> list[dict]:
    out: list[dict] = []
    for name in sorted(by_name):
        info = by_name[name]
        uptime = info["_uptime"]
        uptime_str = (
            f"uptime_1d={uptime:.2f}%" if uptime is not None else "uptime_1d=unknown"
        )
        entry = {
            "name": f"OR_{name.replace(' ', '_')}",
            "tier": "api",
            "transport": "openrouter",
            "model": model_id,
            "provider_hint": name,
            "input_price_per_m": round(info["_input"], 4),
            "output_price_per_m": round(info["_output"], 4),
            "billing_mode": "metered",
            "notes": f"OR endpoint tag={info['_tag']!r}, {uptime_str}",
        }
        if info.get("_cached_input") is not None:
            entry["cached_input_price_per_m"] = round(info["_cached_input"], 4)
        out.append(entry)
    return out


def model_family_from_id(model_id: str) -> str:
    return model_id.split("/")[-1]


def load_existing(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def diff_or_providers(old: list[dict] | None, new: list[dict]) -> str:
    if not old:
        return "  (file does not exist; creating fresh)"
    old_or = {p["name"]: p for p in old if p.get("transport") == "openrouter"}
    new_or = {p["name"]: p for p in new}

    lines: list[str] = []
    added = sorted(set(new_or) - set(old_or))
    removed = sorted(set(old_or) - set(new_or))
    common = sorted(set(old_or) & set(new_or))

    for name in added:
        n = new_or[name]
        lines.append(
            f"  + {name:<22} "
            f"input=${n['input_price_per_m']:.3f}/M "
            f"cached_input={_fmt_cached_price(n)} "
            f"output=${n['output_price_per_m']:.3f}/M"
        )
    for name in removed:
        o = old_or[name]
        lines.append(
            f"  - {name:<22} "
            f"(was input=${o['input_price_per_m']:.3f}/M "
            f"cached_input={_fmt_cached_price(o)} "
            f"output=${o['output_price_per_m']:.3f}/M)"
        )
    for name in common:
        o, n = old_or[name], new_or[name]
        if (
            o.get("input_price_per_m") != n["input_price_per_m"]
            or o.get("cached_input_price_per_m") != n.get("cached_input_price_per_m")
            or o.get("output_price_per_m") != n["output_price_per_m"]
        ):
            lines.append(
                f"  ~ {name:<22} "
                f"input ${o.get('input_price_per_m')}→${n['input_price_per_m']}, "
                "cached_input "
                f"${o.get('cached_input_price_per_m')}→${n.get('cached_input_price_per_m')}, "
                f"output ${o.get('output_price_per_m')}→${n['output_price_per_m']}"
            )
    if not lines:
        return "  (no changes; OR_* providers and prices match)"
    return "\n".join(lines)


def _fmt_cached_price(provider: dict) -> str:
    value = provider.get("cached_input_price_per_m")
    return "n/a" if value is None else f"${float(value):.3f}/M"


def split_existing_providers(existing: dict | None) -> tuple[list[dict], list[dict]]:
    if not existing:
        return [], []
    non_or, or_only = [], []
    for p in existing.get("providers", []):
        if p.get("transport") == "openrouter":
            or_only.append(p)
        else:
            non_or.append(p)
    return non_or, or_only


def check_model_consistency(
    existing: dict | None,
    requested_model: str,
    *,
    force: bool,
) -> None:
    if not existing:
        return
    existing_model = existing.get("openrouter_model_id")
    if not existing_model:
        return
    if existing_model == requested_model:
        return
    if force:
        print(
            f"WARNING: existing openrouter_model_id={existing_model!r} "
            f"differs from --model {requested_model!r}; "
            "--force-model-change set, overwriting model_family too.",
            file=sys.stderr,
        )
        return
    raise SystemExit(
        f"refusing to refresh: existing file has openrouter_model_id="
        f"{existing_model!r} but --model={requested_model!r}. "
        "Pass --force-model-change to overwrite (also rewrites model_family). "
        "Otherwise use a different --output path."
    )


def build_inventory(
    *,
    model_id: str,
    new_or_providers: list[dict],
    existing: dict | None,
    slo_ms: int,
    force_model_change: bool,
) -> dict:
    non_or, _ = split_existing_providers(existing)
    if existing and not force_model_change:
        model_family = existing.get("model_family", model_family_from_id(model_id))
    else:
        model_family = model_family_from_id(model_id)
    primary_slo_ms = (
        existing.get("primary_slo_ms", slo_ms) if existing else slo_ms
    )
    slo_thresholds = (
        existing.get("slo_thresholds_ms", [1000, 2000, 3000, 5000])
        if existing
        else [1000, 2000, 3000, 5000]
    )
    inventory = {
        "model_family": model_family,
        "openrouter_model_id": model_id,
        "primary_slo_ms": primary_slo_ms,
        "slo_thresholds_ms": slo_thresholds,
        "providers": non_or + new_or_providers,
    }
    if existing:
        for key in ("openrouter_provider_only", "openrouter_provider_ignore"):
            if key in existing:
                inventory[key] = existing[key]
    return inventory


def verify_loadable(path: Path) -> bool:
    try:
        from experiments.real_evaluation.inventory import load_inventory

        inv = load_inventory(path)
        n_or = sum(
            1 for p in inv.providers if p.transport_cfg.transport == "openrouter"
        )
        n_other = len(inv.providers) - n_or
        print(
            f"  Verified: load_inventory() ok — {n_or} OR providers, "
            f"{n_other} non-OR (S_Q/S_C)"
        )
        return True
    except Exception as exc:
        print(f"  ERROR: load_inventory() failed: {exc}", file=sys.stderr)
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Refresh OR sub-provider pricing in a real_evaluation inventory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="OpenRouter model ID (e.g. minimax/minimax-m2.5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Inventory JSON path to write or update.",
    )
    parser.add_argument(
        "--slo-ms",
        type=int,
        default=2000,
        help="primary_slo_ms when creating a fresh file. Existing values preserved.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the diff but do not write.",
    )
    parser.add_argument(
        "--force-model-change",
        action="store_true",
        help=(
            "If existing file has a different openrouter_model_id, overwrite "
            "it (and re-derive model_family) instead of failing."
        ),
    )
    args = parser.parse_args(argv)

    existing = load_existing(args.output)
    check_model_consistency(
        existing, args.model, force=args.force_model_change
    )

    print(f"Fetching {args.model} endpoints from OpenRouter...")
    endpoints = fetch_endpoints(args.model)
    print(f"  Got {len(endpoints)} endpoints (before dedup)")

    by_name = dedup_by_cheapest(endpoints)
    print(f"  After dedup by cheapest tier: {len(by_name)} unique sub-providers")

    new_or_providers = to_provider_entries(by_name, args.model)

    print(f"\nDiff against {args.output}:")
    _, existing_or = split_existing_providers(existing)
    print(diff_or_providers(existing_or, new_or_providers))

    inventory = build_inventory(
        model_id=args.model,
        new_or_providers=new_or_providers,
        existing=existing,
        slo_ms=args.slo_ms,
        force_model_change=args.force_model_change,
    )

    if args.dry_run:
        print("\n--dry-run set; not writing")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2) + "\n")
    print(f"\nWrote {args.output}")
    n_or = sum(1 for p in inventory["providers"] if p.get("transport") == "openrouter")
    n_other = len(inventory["providers"]) - n_or
    print(f"  Total providers: {len(inventory['providers'])} ({n_or} OR + {n_other} S_Q/S_C)")

    return 0 if verify_loadable(args.output) else 1


if __name__ == "__main__":
    sys.exit(main())
