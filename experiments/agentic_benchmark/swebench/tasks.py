"""Select which SWE-bench instances to run.

mini-SWE-agent's ``swebench-single`` loads the dataset itself (``--subset`` /
``--split`` / ``-i``); this module's job is only to decide the instance-id list
(a deterministic sample, or an explicit allowlist) so the runner can drive one
instance at a time with a per-task session tag.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

_SUBSET_TO_HF = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "lite": "princeton-nlp/SWE-bench_Lite",
    "full": "princeton-nlp/SWE-bench",
}


@dataclass(frozen=True)
class TaskSpec:
    """One SWE-bench instance to run."""

    instance_id: str
    subset: str
    split: str


def load_instance_ids(subset: str = "verified", split: str = "test") -> list[str]:
    """Load all instance ids for a subset/split via HuggingFace datasets."""
    from datasets import load_dataset

    name = _SUBSET_TO_HF.get(subset, subset)
    ds = load_dataset(name, split=split)
    return list(ds["instance_id"])


def sample_tasks(
    *,
    subset: str = "verified",
    split: str = "test",
    n: int = 5,
    seed: int = 0,
    instance_ids: list[str] | None = None,
) -> list[TaskSpec]:
    """Pick a deterministic sample of instances, or wrap an explicit allowlist."""
    if instance_ids:
        chosen = list(instance_ids)
    else:
        ids = load_instance_ids(subset, split)
        rng = random.Random(seed)
        chosen = sorted(rng.sample(ids, min(n, len(ids))))
    return [TaskSpec(instance_id=i, subset=subset, split=split) for i in chosen]
