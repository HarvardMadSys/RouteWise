"""Round-robin provider-selection cost routers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar


T = TypeVar("T")


def provider_for_index(providers: Sequence[T], index: int) -> T:
    """Return the provider selected by deterministic round-robin order."""
    if not providers:
        raise ValueError("No providers available")
    return providers[index % len(providers)]


__all__ = ["provider_for_index"]
