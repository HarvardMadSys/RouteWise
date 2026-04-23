from __future__ import annotations


class FixedRatioStrategy:
    """Distribute weights between two groups by a fixed fraction."""

    def __init__(self, local_fraction: float) -> None:
        self.local_fraction = local_fraction

    def assign(
        self,
        local: list[tuple[object, str]],
        remote: list[tuple[object, str]],
    ) -> dict[object, float]:
        """Return per-adapter weights.

        Args:
            local: List of (adapter, model_id) in local group
            remote: List of (adapter, model_id) in remote group
        Returns:
            Mapping of adapter -> weight in [0,1]
        """
        weights: dict[object, float] = {}
        lf = self.local_fraction
        rf = max(0.0, 1.0 - lf)

        if local:
            per = lf / len(local)
            for a, _ in local:
                weights[a] = per
        if remote:
            per = rf / len(remote)
            for a, _ in remote:
                weights[a] = per

        # Edge cases: if one side is empty, allocate all to the other
        if not local and remote:
            per = 1.0 / len(remote)
            for a, _ in remote:
                weights[a] = per
        if not remote and local:
            per = 1.0 / len(local)
            for a, _ in local:
                weights[a] = per

        return weights
