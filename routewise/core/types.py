"""Shared RouteWise decision data contracts.

These types are intentionally dependency-free and side-effect-free. They
describe policy outputs, not provider state or execution lifecycle mutation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


@dataclass(frozen=True, init=False)
class RoutingDecision:
    """Primary routing decision returned by a RouteWise-compatible policy.

    ``hedge_checkpoints_sec`` stores elapsed seconds after primary dispatch.
    The legacy ``hedge_checkpoints`` keyword/property is kept as a compatibility
    alias for existing simulator policies and tests.
    """

    primary_provider: str
    hedge_checkpoints_sec: tuple[float, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        primary_provider: str,
        hedge_checkpoints_sec: Iterable[float] | None = None,
        metadata: Mapping[str, Any] | None = None,
        *,
        hedge_checkpoints: Iterable[float] | None = None,
    ) -> None:
        if hedge_checkpoints is not None:
            if hedge_checkpoints_sec is not None:
                raise TypeError(
                    "pass only one of hedge_checkpoints_sec or hedge_checkpoints"
                )
            hedge_checkpoints_sec = hedge_checkpoints
        object.__setattr__(self, "primary_provider", primary_provider)
        object.__setattr__(
            self,
            "hedge_checkpoints_sec",
            tuple(float(value) for value in (hedge_checkpoints_sec or ())),
        )
        object.__setattr__(self, "metadata", dict(metadata or {}))

    @property
    def hedge_checkpoints(self) -> tuple[float, ...]:
        """Compatibility alias for elapsed hedge checkpoint seconds."""
        return self.hedge_checkpoints_sec


@dataclass(frozen=True)
class HedgeDispatch:
    """In-flight hedge dispatch requested by a policy checkpoint."""

    backup_provider: str
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["HedgeDispatch", "RoutingDecision"]
