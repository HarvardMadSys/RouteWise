"""The launcher's ``__slo<ms>`` policy-label parsing.

A regression here mislabels a multi-day run (wrong SLO recorded against a
policy directory), so the two helpers are pinned directly.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

LAUNCHER = Path("scripts/run_real_eval_8h_policy_processes.sh")


def _parse(label: str) -> tuple[str, str]:
    """Return ``(base, slo)`` as the launcher's helpers compute them."""
    extract = (
        f"awk '/^policy_base\\(\\)/,/^}}/' {LAUNCHER}; "
        f"awk '/^policy_slo_override\\(\\)/,/^}}/' {LAUNCHER}"
    )
    script = (
        f'eval "$({extract})"\n'
        f'printf "%s\\n%s\\n" "$(policy_base "{label}")" "$(policy_slo_override "{label}")"'
    )
    out = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    return out[0], (out[1] if len(out) > 1 else "")


def test_suffixed_label_splits_into_base_policy_and_slo() -> None:
    assert _parse("budget_range_alpha50_hedge__slo2000") == (
        "budget_range_alpha50_hedge",
        "2000",
    )
    assert _parse("single_OR_Together__slo5000") == ("single_OR_Together", "5000")


def test_unsuffixed_label_is_unchanged_and_yields_no_override() -> None:
    for name in ("budget_range_alpha50_hedge", "greedy_cost", "or_auto"):
        assert _parse(name) == (name, "")
