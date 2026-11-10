#!/usr/bin/env python3
"""Internal cache-locality estimator for RouteWise.

This module provides a process-local estimator that learns destination-local
cache reuse from actual completion observations. It is used by the Router
to incorporate learned cache-locality evidence into routing decisions.

This is an internal implementation detail. It is not part of the public API.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class _CacheLocalityEvidence:
    """Blended cache-reuse evidence for a provider+affinity pair.

    This is NOT authoritative cache state. It records what was observed
    for one specific completed attempt. After a miss the entry holds
    blended state: the last positive observation's cached tokens and
    timestamp, scaled down by the accumulated confidence penalty.
    """

    last_cached_tokens: int
    observed_at: float
    confidence: float
    last_event_at: float


class _CacheLocalityEstimator:
    """Process-local cache-locality evidence store.

    This is the default estimator. It is suitable for single-router deployments.
    """

    def __init__(
        self,
        ttl_sec: float = 300.0,
    ) -> None:
        if ttl_sec <= 0:
            raise ValueError("ttl_sec must be positive")

        self._ttl_sec = ttl_sec
        # Derive half-life as TTL/2
        self._half_life_sec = ttl_sec / 2.0
        # Internal constants
        self._min_confidence = 0.01
        self._max_entries = 10_000
        self._miss_confidence_factor = 0.3
        # Ordered by most-recent observation. Refreshing an existing key moves
        # it to the end, so capacity eviction never scans the full store.
        self._evidence: OrderedDict[tuple[str, str], _CacheLocalityEvidence] = OrderedDict()
        # Keep ordering watermarks separately so a delayed hit cannot restore
        # evidence after a newer miss deleted the estimate itself.
        self._last_event_at: OrderedDict[tuple[str, str], float] = OrderedDict()
        self._lock = Lock()

    def record(
        self,
        provider: str,
        affinity_key: str,
        cached_tokens: int,
        input_tokens: int,
        now: float,
    ) -> None:
        """Record a cache-locality observation.

        ``now`` is when the cache was used, not when the caller reports it: a
        ``cached_tokens`` value that arrives late through ``settle(...)`` passes
        the completion time. An observation older than the newest one already
        recorded for the same provider and affinity key is dropped, so a
        delayed hit cannot undo a newer miss.

        ``cached_tokens`` is the authoritative observed cache-reuse count
        reported by the provider for this completed request:

        - ``cached_tokens > 0``: positive evidence that reusable prefix state
          exists for this provider+affinity pair.
        - ``cached_tokens == 0``: negative evidence -- reuse was not observed.
          Repeated misses degrade confidence but do not immediately delete
          evidence (transient cache eviction is possible). A subsequent hit
          restores confidence.

        Callers must distinguish ``0`` (provider explicitly reported no reuse)
        from ``None`` (no authoritative cache-use observation was available).
        ``None`` must not be passed here or treated as ``0``.
        """
        if input_tokens <= 0:
            return  # No meaningful evidence to record

        # Clamp observed cached tokens at input_tokens
        cached_tokens = min(cached_tokens, input_tokens)

        key = (provider, affinity_key)

        with self._lock:
            last_event_at = self._last_event_at.get(key)
            evidence = self._evidence.get(key)
            if (last_event_at is not None and now < last_event_at) or (
                evidence is not None and now < evidence.last_event_at
            ):
                return

            self._last_event_at[key] = now
            self._last_event_at.move_to_end(key)
            while len(self._last_event_at) > self._max_entries:
                self._last_event_at.popitem(last=False)

            existing = self._get_fresh_evidence(key, now)

            if cached_tokens > 0:
                # Positive observation: refresh evidence
                self._evidence[key] = _CacheLocalityEvidence(
                    last_cached_tokens=cached_tokens,
                    observed_at=now,
                    confidence=1.0,
                    last_event_at=now,
                )
                self._evidence.move_to_end(key)
            else:
                # Negative observation (miss): degrade confidence
                # Keep the last confirmed-hit timestamp. estimate() applies
                # time decay from that timestamp; refreshing it on a miss
                # would extend stale positive evidence and decay it twice.
                if existing is not None:
                    age = max(0.0, now - existing.observed_at)
                    new_confidence = existing.confidence * self._miss_confidence_factor
                    effective_confidence = new_confidence * (0.5 ** (age / self._half_life_sec))
                    if effective_confidence < self._min_confidence:
                        del self._evidence[key]
                    else:
                        self._evidence[key] = _CacheLocalityEvidence(
                            last_cached_tokens=existing.last_cached_tokens,
                            observed_at=existing.observed_at,
                            confidence=new_confidence,
                            last_event_at=now,
                        )
                # If no existing evidence, a miss creates no evidence

            self._enforce_capacity()

    def estimate(
        self,
        provider: str,
        affinity_key: str,
        current_input_tokens: int,
        now: float,
    ) -> int:
        """Return estimated cached tokens.

        Returns 0 if no valid evidence, expired, or below confidence threshold.
        Uses lazy expiration: stale entries are removed on lookup.
        """
        key = (provider, affinity_key)

        with self._lock:
            ev = self._get_fresh_evidence(key, now)
            if ev is None:
                return 0

            # A clock that steps backwards gives a negative age, which would
            # make the decay factor above 1.0 and inflate the confidence.
            age = max(0.0, now - ev.observed_at)
            # Confidence decay using exponential half-life
            decayed_confidence = ev.confidence * (0.5 ** (age / self._half_life_sec))
            if decayed_confidence < self._min_confidence:
                del self._evidence[key]
                return 0

            # Estimate: scale observed cached tokens by confidence
            estimated = int(ev.last_cached_tokens * decayed_confidence)

            # Conservative: never exceed observed or current input
            return max(0, min(estimated, ev.last_cached_tokens, current_input_tokens))

    def invalidate(self, provider: str, affinity_key: str) -> None:
        """Invalidate specific evidence entry."""
        key = (provider, affinity_key)
        with self._lock:
            if key in self._evidence:
                del self._evidence[key]

    def invalidate_provider(self, provider: str) -> None:
        """Invalidate all evidence for a provider."""
        with self._lock:
            keys_to_remove = [k for k in self._evidence if k[0] == provider]
            for k in keys_to_remove:
                del self._evidence[k]

    @property
    def evidence_count(self) -> int:
        """Current number of evidence entries."""
        with self._lock:
            return len(self._evidence)

    @property
    def ttl_sec(self) -> float:
        """Maximum age of a positive observation."""
        return self._ttl_sec

    def _enforce_capacity(self) -> None:
        """Evict oldest entries if over capacity. Must be called under lock."""
        while len(self._evidence) > self._max_entries:
            self._evidence.popitem(last=False)

    def _get_fresh_evidence(
        self,
        key: tuple[str, str],
        now: float,
    ) -> _CacheLocalityEvidence | None:
        """Return evidence unless its TTL elapsed; must be called under lock."""
        evidence = self._evidence.get(key)
        if evidence is not None and now - evidence.observed_at > self._ttl_sec:
            del self._evidence[key]
            return None
        return evidence
