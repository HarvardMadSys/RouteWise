"""Streaming histograms for latency metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HISTOGRAM_MIN_MS = 1.0
HISTOGRAM_MAX_MS = 100_000.0
HISTOGRAM_BIN_COUNT = 256
DEFAULT_BIN_EDGES_MS = np.geomspace(
    HISTOGRAM_MIN_MS,
    HISTOGRAM_MAX_MS,
    HISTOGRAM_BIN_COUNT + 1,
)


@dataclass
class TtftHistogram:
    """Compact histogram for TTFT and latency-like values in milliseconds."""

    bin_edges_ms: np.ndarray = field(
        default_factory=lambda: DEFAULT_BIN_EDGES_MS.copy()
    )
    counts: np.ndarray | None = None
    sum_value: float = 0.0
    sum_sq: float = 0.0
    n: int = 0
    min_value: float = float("inf")
    max_value: float = float("-inf")

    def __post_init__(self) -> None:
        edges = np.asarray(self.bin_edges_ms, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("histogram bin edges must be a 1D array with at least two edges")
        if not np.all(np.isfinite(edges)):
            raise ValueError("histogram bin edges must be finite")
        if np.any(np.diff(edges) <= 0.0):
            raise ValueError("histogram bin edges must be strictly increasing")
        self.bin_edges_ms = edges
        if self.counts is None:
            self.counts = np.zeros(edges.size + 1, dtype=np.int64)
        else:
            counts = np.asarray(self.counts, dtype=np.int64)
            if counts.shape != (edges.size + 1,):
                raise ValueError(
                    "histogram counts must have len(bin_edges)+1 entries "
                    "(underflow + interior bins + overflow)"
                )
            self.counts = counts

    @classmethod
    def default(cls) -> TtftHistogram:
        """Return the default paper-facing TTFT histogram."""
        return cls()

    def copy(self) -> TtftHistogram:
        """Return a deep copy."""
        return TtftHistogram(
            bin_edges_ms=self.bin_edges_ms.copy(),
            counts=self.counts.copy(),
            sum_value=float(self.sum_value),
            sum_sq=float(self.sum_sq),
            n=int(self.n),
            min_value=float(self.min_value),
            max_value=float(self.max_value),
        )

    def add(self, value_ms: float) -> None:
        """Add one latency value."""
        self.add_array(np.asarray([value_ms], dtype=float))

    def add_array(self, values_ms: np.ndarray) -> None:
        """Add latency values."""
        values = np.asarray(values_ms, dtype=float)
        if values.size == 0:
            return
        values = values[np.isfinite(values)]
        if values.size == 0:
            return
        self.sum_value += float(np.sum(values))
        self.sum_sq += float(np.sum(values * values))
        self.n += int(values.size)
        self.min_value = min(self.min_value, float(np.min(values)))
        self.max_value = max(self.max_value, float(np.max(values)))

        edges = self.bin_edges_ms
        assert self.counts is not None
        self.counts[0] += int(np.sum(values < edges[0]))
        self.counts[-1] += int(np.sum(values >= edges[-1]))
        inside = values[(values >= edges[0]) & (values < edges[-1])]
        if inside.size:
            hist, _ = np.histogram(inside, bins=edges)
            self.counts[1:-1] += hist.astype(np.int64, copy=False)

    def merge(self, other: TtftHistogram) -> TtftHistogram:
        """Return a merged histogram."""
        if not np.array_equal(self.bin_edges_ms, other.bin_edges_ms):
            raise ValueError("cannot merge histograms with different bin edges")
        assert self.counts is not None
        assert other.counts is not None
        return TtftHistogram(
            bin_edges_ms=self.bin_edges_ms.copy(),
            counts=self.counts + other.counts,
            sum_value=self.sum_value + other.sum_value,
            sum_sq=self.sum_sq + other.sum_sq,
            n=self.n + other.n,
            min_value=min(self.min_value, other.min_value),
            max_value=max(self.max_value, other.max_value),
        )

    def mean(self) -> float:
        """Return the exact mean of observed values."""
        if self.n == 0:
            return float("nan")
        return self.sum_value / self.n

    def std(self) -> float:
        """Return the exact population standard deviation of observed values."""
        if self.n == 0:
            return float("nan")
        mean = self.mean()
        variance = max(self.sum_sq / self.n - mean * mean, 0.0)
        return float(np.sqrt(variance))

    def quantile(self, q: float) -> float:
        """Return an approximate quantile from histogram bins."""
        if not 0.0 < q < 1.0:
            raise ValueError(f"quantile q must be in (0, 1), got {q}")
        if self.n == 0:
            return float("nan")
        if self.min_value == self.max_value:
            return float(self.min_value)
        assert self.counts is not None
        target = q * self.n
        cumulative = 0
        edges = self.bin_edges_ms

        for idx, count in enumerate(self.counts):
            next_cumulative = cumulative + int(count)
            if target <= next_cumulative:
                if idx == 0:
                    return float(self.min_value)
                if idx == len(self.counts) - 1:
                    return float(self.max_value)
                lo = float(edges[idx - 1])
                hi = float(edges[idx])
                if count <= 1:
                    return 0.5 * (lo + hi)
                frac = (target - cumulative) / count
                return lo + frac * (hi - lo)
            cumulative = next_cumulative
        return float(edges[-1])

    def cdf(self, value_ms: float) -> float:
        """Return an approximate cumulative fraction at value_ms."""
        if self.n == 0:
            return 0.0
        value = float(value_ms)
        edges = self.bin_edges_ms
        assert self.counts is not None
        if value < edges[0]:
            return 0.0
        cumulative = int(self.counts[0])
        if value >= edges[-1]:
            return 1.0
        bin_idx = int(np.searchsorted(edges, value, side="right"))
        cumulative += int(np.sum(self.counts[1:bin_idx]))
        lo = float(edges[bin_idx - 1])
        hi = float(edges[bin_idx])
        count = int(self.counts[bin_idx])
        if count:
            cumulative += count * ((value - lo) / (hi - lo))
        return float(min(max(cumulative / self.n, 0.0), 1.0))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        assert self.counts is not None
        return {
            "bin_edges_ms": self.bin_edges_ms.tolist(),
            "counts": self.counts.astype(int).tolist(),
            "n": int(self.n),
            "mean_ms": self.mean(),
            "std_ms": self.std(),
            "min_ms": self.min_value if self.n else float("nan"),
            "max_ms": self.max_value if self.n else float("nan"),
        }


def merge_histograms(histograms: list[TtftHistogram]) -> TtftHistogram:
    """Merge histograms, returning an empty default histogram for an empty input."""
    if not histograms:
        return TtftHistogram.default()
    merged = histograms[0].copy()
    for histogram in histograms[1:]:
        merged = merged.merge(histogram)
    return merged


__all__ = [
    "DEFAULT_BIN_EDGES_MS",
    "HISTOGRAM_BIN_COUNT",
    "HISTOGRAM_MAX_MS",
    "HISTOGRAM_MIN_MS",
    "TtftHistogram",
    "merge_histograms",
]
