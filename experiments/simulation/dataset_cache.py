"""Pre-process trace datasets into pickle caches for fast loading.

Raw CSV/JSONL parsing dominates wall-clock time on multi-cell grid runs:
rednote (14 GB CSV) takes ~90 s per cold load, freeinference (3.2 GB) ~25 s,
sharegpt (506 MB JSONL) ~8 s.  A pickle cache reduces reload to <2 s each
because ``pickle.load`` bypasses CSV parsing and string-to-int conversion
entirely — it deserialises pre-built ``Request`` dataclass instances.

**Source-fingerprint manifest:** every pickle is accompanied by a sidecar
``{dataset}.manifest.json`` that records the source path, file size, mtime,
and SHA-256 hash at build time.  On load, the manifest is compared against
the current source file; a mismatch raises ``CacheStalenessError`` so you
never silently run paper numbers against the wrong workload.

Usage
-----
Build caches for all three trace datasets::

    python -m experiments.simulation.dataset_cache build

Build for one dataset::

    python -m experiments.simulation.dataset_cache build --dataset rednote

Check cache status (includes staleness check)::

    python -m experiments.simulation.dataset_cache status

Clear caches::

    python -m experiments.simulation.dataset_cache clear
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rwsim.data import DataLoader
from rwsim.schemas import Request

TRACE_WORKLOAD_DATASETS = ("burstgpt", "freeinference", "rednote", "sharegpt")

_TRACE_DATA_ROOT = _ROOT / "data"
_DATASET_CACHE_ROOT = _ROOT / "outputs" / "cache" / "dataset"
_DATA_LOADER_CONFIG = {"dataset": {}}

_TRACE_DATASET_PATHS = {
    "freeinference": [
        _TRACE_DATA_ROOT / "freeinference.csv",
        _TRACE_DATA_ROOT / "freeinference_logs.csv",
    ],
    "rednote": [
        _TRACE_DATA_ROOT / "enterprise.csv",
        _TRACE_DATA_ROOT / "rednote_logs.csv",
    ],
    "burstgpt": [
        _TRACE_DATA_ROOT / "burstgpt_30d.jsonl",
    ],
    "sharegpt": [
        _TRACE_DATA_ROOT / "sharegpt_prompts_7d.jsonl",
        _TRACE_DATA_ROOT / "sharegpt_burstgpt" / "sharegpt_prompts_7d.jsonl",
        _TRACE_DATA_ROOT / "sharegpt_burstgpt" / "converted.csv",
    ],
}


def _resolve_trace_dataset_path(dataset_name: str) -> Path | None:
    """Return the first available canonical path for one trace dataset."""
    for candidate in _TRACE_DATASET_PATHS.get(dataset_name, []):
        if candidate.exists():
            return candidate
    return None


def _dataset_cache_path(dataset_name: str) -> Path:
    return _DATASET_CACHE_ROOT / f"{dataset_name}.pkl"


def _load_sharegpt_jsonl_requests(filepath: Path) -> list[Request]:
    """Load ShareGPT-style JSONL trace records into Request objects."""
    requests: list[Request] = []
    first_timestamp: float | None = None
    with filepath.open() as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            timestamp = float(record["arrived_at"])
            if first_timestamp is None:
                first_timestamp = timestamp
            request_tokens = int(record["num_prefill_tokens"])
            response_tokens = int(record["num_decode_tokens"])
            total_tokens = request_tokens + response_tokens
            if total_tokens <= 0:
                continue
            metadata = {
                key: record[key]
                for key in (
                    "session_id",
                    "prompt_text",
                    "response_text",
                    "sharegpt_conversation_id",
                    "sharegpt_turn_index",
                    "log_type",
                    "elapsed_time_sec",
                )
                if key in record
            }
            requests.append(
                Request(
                    id=idx,
                    timestamp=float(timestamp - first_timestamp),
                    request_tokens=request_tokens,
                    response_tokens=response_tokens,
                    total_tokens=total_tokens,
                    model=str(record.get("model") or "sharegpt"),
                    metadata=metadata,
                )
            )
    return requests


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class CacheStalenessError(RuntimeError):
    """Raised when the pickle cache does not match the current source file."""


# ---------------------------------------------------------------------------
# Source fingerprinting
# ---------------------------------------------------------------------------

# SHA-256 chunk size — 8 MB gives good throughput on SSD without
# allocating too much memory for the 14 GB rednote file.
_HASH_CHUNK = 8 * 1024 * 1024


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file, following symlinks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _source_fingerprint(source_path: Path) -> dict[str, object]:
    """Build the manifest fingerprint dict for one source file."""
    resolved = source_path.resolve()
    stat = resolved.stat()
    return {
        "source_path": str(source_path),
        "source_resolved": str(resolved),
        "source_size": stat.st_size,
        "source_mtime": stat.st_mtime,
        "source_sha256": _sha256_file(resolved),
    }


def _manifest_path(dataset_name: str) -> Path:
    """Return the path to the sidecar manifest JSON for one dataset."""
    return _dataset_cache_path(dataset_name).with_suffix(".manifest.json")


def _write_manifest(dataset_name: str, fingerprint: dict[str, object]) -> None:
    manifest = _manifest_path(dataset_name)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w") as f:
        json.dump(fingerprint, f, indent=2)


def _read_manifest(dataset_name: str) -> dict[str, object] | None:
    manifest = _manifest_path(dataset_name)
    if not manifest.exists():
        return None
    with manifest.open() as f:
        return json.load(f)


def verify_cache(dataset_name: str, *, quick: bool = False) -> bool:
    """Check that the cache matches the current source file.

    With ``quick=True``, only size + mtime are checked (fast but not
    cryptographically safe).  With ``quick=False`` (default), SHA-256 is
    recomputed and compared — takes ~10 s for rednote (14 GB).

    Returns True if the cache is valid.  Raises ``CacheStalenessError``
    if the manifest disagrees with the current source.  Returns False
    if the manifest or cache is missing.
    """
    cache_path = _dataset_cache_path(dataset_name)
    if not cache_path.exists():
        return False

    manifest = _read_manifest(dataset_name)
    if manifest is None:
        # Legacy cache without manifest — force rebuild.
        raise CacheStalenessError(
            f"Cache for {dataset_name!r} has no manifest.  Rebuild with: "
            f"python -m experiments.simulation.dataset_cache build "
            f"--dataset {dataset_name} --force"
        )

    source_path = _resolve_trace_dataset_path(dataset_name)
    if source_path is None:
        # Source is gone but cache exists — still usable.
        return True

    resolved = source_path.resolve()
    stat = resolved.stat()

    if stat.st_size != manifest.get("source_size"):
        raise CacheStalenessError(
            f"Source file size changed for {dataset_name!r}: "
            f"manifest={manifest.get('source_size')}, "
            f"current={stat.st_size}.  Rebuild the cache."
        )

    if stat.st_mtime != manifest.get("source_mtime"):
        raise CacheStalenessError(
            f"Source file mtime changed for {dataset_name!r}: "
            f"manifest={manifest.get('source_mtime')}, "
            f"current={stat.st_mtime}.  Rebuild the cache."
        )

    if not quick:
        current_sha = _sha256_file(resolved)
        if current_sha != manifest.get("source_sha256"):
            raise CacheStalenessError(
                f"Source SHA-256 changed for {dataset_name!r}: "
                f"manifest={manifest.get('source_sha256')[:16]}..., "
                f"current={current_sha[:16]}....  Rebuild the cache."
            )

    return True


# ---------------------------------------------------------------------------
# Core cache operations
# ---------------------------------------------------------------------------


def build_cache(dataset_name: str, *, force: bool = False) -> Path:
    """Parse a raw trace file and write a pickle cache with manifest.

    Returns the cache path on success.

    Raises
    ------
    FileNotFoundError
        If no raw source file is available for *dataset_name*.
    ValueError
        If *dataset_name* is not a known trace dataset.
    """
    if dataset_name not in TRACE_WORKLOAD_DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name!r}. "
            f"Expected one of {TRACE_WORKLOAD_DATASETS!r}."
        )

    cache_path = _dataset_cache_path(dataset_name)
    if cache_path.exists() and not force:
        # Even when skipping, verify the manifest is still valid.
        try:
            verify_cache(dataset_name, quick=True)
            print(f"  [skip] {dataset_name}: cache already exists at {cache_path}")
            return cache_path
        except CacheStalenessError as exc:
            print(f"  [stale] {dataset_name}: {exc} — rebuilding.")

    source_path = _resolve_trace_dataset_path(dataset_name)
    if source_path is None:
        raise FileNotFoundError(
            f"No raw source file found for {dataset_name!r}. "
            f"Searched: {list(_TRACE_DATASET_PATHS.get(dataset_name, []))}"
        )

    print(f"  [build] {dataset_name}: {source_path} -> {cache_path}")

    # --- fingerprint source BEFORE parsing (guards against mid-build edits)
    t0 = time.monotonic()
    fingerprint = _source_fingerprint(source_path)
    elapsed_hash = time.monotonic() - t0
    print(f"          source sha256={fingerprint['source_sha256'][:16]}... ({elapsed_hash:.1f}s)")

    t1 = time.monotonic()
    if source_path.suffix == ".jsonl":
        requests = _load_sharegpt_jsonl_requests(source_path)
    else:
        loader = DataLoader(_DATA_LOADER_CONFIG)
        requests = loader.load(source_path)

    requests = tuple(requests)
    elapsed_parse = time.monotonic() - t1
    print(
        f"          parsed {len(requests):,} requests in {elapsed_parse:.1f}s"
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a PID-specific temporary file, then atomic rename.
    # This prevents races when multiple processes try to build the same
    # cache simultaneously (e.g. parallel runner workers).
    tmp_path = cache_path.with_suffix(f".pkl.tmp.{os.getpid()}")
    t2 = time.monotonic()
    with tmp_path.open("wb") as handle:
        pickle.dump(requests, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.rename(cache_path)
    elapsed_write = time.monotonic() - t2

    # Write manifest (after pickle so the cache is never ahead of manifest).
    fingerprint["n_requests"] = len(requests)
    fingerprint["cache_size"] = cache_path.stat().st_size
    _write_manifest(dataset_name, fingerprint)

    cache_mb = cache_path.stat().st_size / (1024 * 1024)
    total = elapsed_hash + elapsed_parse + elapsed_write
    print(
        f"          wrote {cache_mb:.1f} MB pickle in {elapsed_write:.1f}s "
        f"(total {total:.1f}s)"
    )
    return cache_path


def load_cached(dataset_name: str) -> tuple:
    """Load a dataset from the pickle cache.

    Raises FileNotFoundError if the cache does not exist.
    """
    cache_path = _dataset_cache_path(dataset_name)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No cache for {dataset_name!r} at {cache_path}. "
            f"Run: python -m experiments.simulation.dataset_cache build "
            f"--dataset {dataset_name}"
        )
    with cache_path.open("rb") as handle:
        return tuple(pickle.load(handle))


def cache_status() -> dict[str, dict[str, object]]:
    """Return a status dict for every known trace dataset."""
    result = {}
    for name in TRACE_WORKLOAD_DATASETS:
        cache_path = _dataset_cache_path(name)
        source_path = _resolve_trace_dataset_path(name)
        entry: dict[str, object] = {
            "cache_path": str(cache_path),
            "cache_exists": cache_path.exists(),
            "source_path": str(source_path) if source_path else None,
            "source_exists": source_path is not None and source_path.exists(),
        }
        if cache_path.exists():
            stat = cache_path.stat()
            entry["cache_size_mb"] = round(stat.st_size / (1024 * 1024), 1)
            entry["cache_mtime"] = stat.st_mtime
            # Staleness check
            try:
                verify_cache(name, quick=True)
                entry["cache_valid"] = True
                entry["staleness_error"] = None
            except CacheStalenessError as exc:
                entry["cache_valid"] = False
                entry["staleness_error"] = str(exc)
        else:
            entry["cache_valid"] = None
        manifest = _read_manifest(name)
        if manifest:
            entry["manifest_sha256"] = manifest.get("source_sha256", "")[:16]
            entry["manifest_n_requests"] = manifest.get("n_requests")
        result[name] = entry
    return result


def clear_cache(dataset_name: str | None = None) -> list[str]:
    """Delete cache files and manifests.  If *dataset_name* is None, clear all."""
    cleared: list[str] = []
    targets = [dataset_name] if dataset_name else list(TRACE_WORKLOAD_DATASETS)
    for name in targets:
        pkl = _dataset_cache_path(name)
        manifest = _manifest_path(name)
        deleted = False
        if pkl.exists():
            pkl.unlink()
            deleted = True
        if manifest.exists():
            manifest.unlink()
            deleted = True
        if deleted:
            cleared.append(name)
            print(f"  [clear] {name}: deleted cache + manifest")
        else:
            print(f"  [skip]  {name}: no cache to delete")
    return cleared


def ensure_caches(
    datasets: list[str],
    *,
    force: bool = False,
    verify: bool = True,
) -> None:
    """Ensure pickle caches exist and are valid for the given datasets.

    This is the entry point called by the parallel runner's parent process
    before spawning workers.  It either builds missing caches or raises
    on staleness, ensuring every worker can safely load from pickle
    without touching the raw source.

    Parameters
    ----------
    datasets
        Subset of ``TRACE_WORKLOAD_DATASETS`` to prepare.
    force
        Rebuild caches unconditionally.
    verify
        Run quick staleness check (size + mtime) on existing caches.
        Set to ``False`` for test / CI environments where source files
        may not exist.
    """
    for name in datasets:
        if name not in TRACE_WORKLOAD_DATASETS:
            raise ValueError(
                f"Unknown dataset: {name!r}. "
                f"Expected one of {TRACE_WORKLOAD_DATASETS!r}."
            )
        cache_path = _dataset_cache_path(name)
        needs_build = force or not cache_path.exists()

        if not needs_build and verify:
            try:
                verify_cache(name, quick=True)
            except CacheStalenessError as exc:
                print(f"  [stale] {name}: {exc}")
                needs_build = True

        if needs_build:
            build_cache(name, force=True)
        else:
            print(f"  [ok] {name}: cache valid ({cache_path})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage trace dataset pickle caches."
    )
    sub = parser.add_subparsers(dest="command")

    build_p = sub.add_parser("build", help="Build pickle caches from raw data.")
    build_p.add_argument(
        "--dataset",
        choices=list(TRACE_WORKLOAD_DATASETS),
        default=None,
        help="Build cache for one dataset. Default: all.",
    )
    build_p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if cache already exists.",
    )

    status_p = sub.add_parser("status", help="Show cache status for all datasets.")
    status_p.add_argument(
        "--full-verify",
        action="store_true",
        help="Recompute SHA-256 for full staleness verification (slow for large files).",
    )

    clear_p = sub.add_parser("clear", help="Delete cache files.")
    clear_p.add_argument(
        "--dataset",
        choices=list(TRACE_WORKLOAD_DATASETS),
        default=None,
        help="Clear cache for one dataset. Default: all.",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.command == "build":
        datasets = [args.dataset] if args.dataset else list(TRACE_WORKLOAD_DATASETS)
        for name in datasets:
            try:
                build_cache(name, force=args.force)
            except FileNotFoundError as exc:
                print(f"  [error] {name}: {exc}", file=sys.stderr)
        print("\nDone.")

    elif args.command == "status":
        full = getattr(args, "full_verify", False)
        status = cache_status()
        for name, info in status.items():
            if info["cache_exists"]:
                valid_tag = "valid" if info.get("cache_valid") else "STALE"
                sha_tag = f" sha={info.get('manifest_sha256', '?')}..." if info.get("manifest_sha256") else ""
                n_req = f" {info.get('manifest_n_requests', '?'):,} reqs" if info.get("manifest_n_requests") else ""
                tag = f"cached ({info['cache_size_mb']} MB, {valid_tag}{sha_tag}{n_req})"
            else:
                tag = "missing"
            src = "source ok" if info["source_exists"] else "NO source"
            print(f"  {name:<20s} [{tag}]  {src}")
            if info.get("staleness_error"):
                print(f"                       !! {info['staleness_error']}")

            # Full SHA-256 verification if requested.
            if full and info["cache_exists"]:
                try:
                    verify_cache(name, quick=False)
                    print("                       SHA-256 verified OK")
                except CacheStalenessError as exc:
                    print(f"                       SHA-256 MISMATCH: {exc}")

    elif args.command == "clear":
        clear_cache(getattr(args, "dataset", None))

    else:
        print("Usage: python -m experiments.simulation.dataset_cache {build|status|clear}")
        sys.exit(1)


if __name__ == "__main__":
    main()
