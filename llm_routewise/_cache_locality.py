#!/usr/bin/env python3
"""Internal cache-locality estimator for RouteWise.

This module provides a process-local estimator that learns destination-local
cache reuse from actual completion observations. It is used by the Router
to incorporate learned cache-locality evidence into routing decisions.

This is an internal implementation detail. It is not part of the public API.
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class _CacheLocalityEvidence:
    """One observation of cache reuse for a provider+affinity pair.

    This is NOT authoritative cache state. It records what was observed
    for one specific completed attempt.
    """
    last_cached_tokens: int
    observed_at: float
    generation: int
    confidence: float


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
        self._evidence: dict[tuple[str, str], _CacheLocalityEvidence] = {}
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

        Positive cached_tokens values produce evidence of reusable prefix
        state. ``cached_tokens=0`` provides negative evidence that the prior
        reusable-state belief did not result in reuse on this request.

        Repeated misses degrade confidence but do not immediately delete
        evidence (transient cache eviction is possible). A subsequent hit
        restores confidence.
        """
        if input_tokens <= 0:
            return  # No meaningful evidence to record

        # Clamp observed cached tokens at input_tokens
        cached_tokens = min(cached_tokens, input_tokens)

        key = (provider, affinity_key)

        with self._lock:
            existing = self._evidence.get(key)
            generation = existing.generation + 1 if existing is not None else 1

            if cached_tokens > 0:
                # Positive observation: refresh evidence
                if existing is not None:
                    self._evidence[key] = _CacheLocalityEvidence(
                        last_cached_tokens=cached_tokens,
                        observed_at=now,
                        generation=generation,
                        confidence=1.0,
                    )
                else:
                    self._evidence[key] = _CacheLocalityEvidence(
                        last_cached_tokens=cached_tokens,
                        observed_at=now,
                        generation=generation,
                        confidence=1.0,
                    )
            else:
                # Negative observation (miss): degrade confidence
                # First apply time decay from previous observation to now,
                # then apply the miss penalty
                if existing is not None:
                    age = now - existing.observed_at
                    decayed_confidence = existing.confidence * (0.5 ** (age / self._half_life_sec))
                    new_confidence = decayed_confidence * self._miss_confidence_factor
                    self._evidence[key] = _CacheLocalityEvidence(
                        last_cached_tokens=existing.last_cached_tokens,
                        observed_at=now,
                        generation=generation,
                        confidence=new_confidence,
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
            ev = self._evidence.get(key)
            if ev is None:
                return 0

            age = now - ev.observed_at
            if age > self._ttl_sec:
                # Lazy expiration
                del self._evidence[key]
                return 0

            # Confidence decay using exponential half-life
            decayed_confidence = ev.confidence * (0.5 ** (age / self._half_life_sec))
            if decayed_confidence < self._min_confidence:
                del self._evidence[key]
                return 0

            # Estimate: scale observed cached tokens by confidence
            estimated = int(ev.last_cached_tokens * decayed_confidence)

            # Conservative: never exceed observed or current input
            return min(estimated, ev.last_cached_tokens, current_input_tokens)

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

    def _enforce_capacity(self) -> None:
        """Evict oldest entries if over capacity. Must be called under lock."""
        while len(self._evidence) > self._max_entries:
            oldest_key = min(self._evidence, key=lambda k: self._evidence[k].observed_at)
            del self._evidence[oldest_key]
