from __future__ import annotations

"""Small, low-cardinality error categorization helpers."""


_CATEGORIES: dict[str, tuple[str, ...]] = {
    "timeout": ("TimeoutError", "ReadTimeout", "WriteTimeout", "timed out"),
    "rate_limited": ("429", "RateLimit", "TooManyRequests"),
    "network": ("ConnectionError", "ConnectError", "DNSError", "Connection reset"),
    "server_error": ("500", "502", "503", "504", "InternalServerError"),
    "validation": ("ValidationError",),
}


def categorize_exception(exc: BaseException) -> str:
    """Map an exception to a coarse error category.

    Returns one of: timeout|rate_limited|network|server_error|validation|unknown
    """
    s = f"{type(exc).__name__}: {exc}"
    ls = s.lower()
    for cat, needles in _CATEGORIES.items():
        for needle in needles:
            if needle.lower() in ls:
                return cat
    return "unknown"


__all__ = ["categorize_exception"]
