"""Small typed registry used by migrated simulator components."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass
class Registry(Generic[T]):
    """Name-to-factory registry with explicit duplicate checks."""

    name: str
    _items: dict[str, Callable[..., T]] = field(default_factory=dict)

    def register(self, key: str, factory: Callable[..., T]) -> None:
        """Register a factory under a stable key."""
        if key in self._items:
            raise ValueError(f"{self.name} registry already has key {key!r}")
        self._items[key] = factory

    def get(self, key: str) -> Callable[..., T]:
        """Return the registered factory for a key."""
        try:
            return self._items[key]
        except KeyError as exc:
            known = ", ".join(sorted(self._items))
            raise KeyError(f"Unknown {self.name} key {key!r}. Known: {known}") from exc

    def keys(self) -> tuple[str, ...]:
        """Return registered keys."""
        return tuple(sorted(self._items))


__all__ = ["Registry"]
