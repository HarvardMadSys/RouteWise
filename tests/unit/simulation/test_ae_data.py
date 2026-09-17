"""Privacy, integrity, and replay-input checks for the released AE data."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path

import pytest

from experiments.simulation.dataset_cache import _load_freeinference_jsonl_requests
from scripts.export_ae_data import RECORD_FIELDS, TRACE_FIELDS, export_prod_trace
from scripts.reproduce_real_world import check_summary

ROOT = Path(__file__).resolve().parents[3]


def test_prod_export_removes_private_fields_without_changing_replay(tmp_path):
    base = {
        "timestamp": "2026-01-01T00:00:00Z",
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "cache_read_tokens": 80,
        "status_code": 200,
        "user_name": "Private Name",
        "user_email": "private@example.org",
        "error": "private diagnostic",
        "prompt": "private prompt",
        "response": "private response",
    }
    rows = [{**base, "user_id": uid} for uid in (0, 7, 0)]
    rows.extend([{**base, "status_code": 500}, {**base, "total_tokens": 0}])
    source, dest = tmp_path / "private.jsonl", tmp_path / "release" / "trace.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    export_prod_trace(source, dest)
    exported = [json.loads(line) for line in dest.read_text(encoding="utf-8").splitlines()]
    assert len(exported) == 5
    assert [row.get("user_id") for row in exported] == [
        "account-01",
        "account-02",
        "account-01",
        None,
        None,
    ]
    assert all(set(row) <= set(TRACE_FIELDS) | {"user_id"} for row in exported)
    assert "private" not in dest.read_text(encoding="utf-8").lower()
    before, after = (_load_freeinference_jsonl_requests(path) for path in (source, dest))
    assert len(before) == len(after) == 3
    for original, published in zip(before, after, strict=True):
        original, published = asdict(original), asdict(published)
        for record in (original, published):
            for field in ("user_id", "prefix_id", "error"):
                record["metadata"].pop(field, None)
            # The exporter writes explicit null for absent optional metadata.
            record["metadata"] = {k: v for k, v in record["metadata"].items() if v is not None}
        assert original == published


@pytest.mark.parametrize(
    "directory",
    [
        "data",
        "data/real_eval_records",
        "data/real_eval_records_m3",
        "data/real_eval_slo_sweep_m3",
    ],
)
def test_committed_data_checksums(directory):
    root = ROOT / directory
    entries = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert entries
    names = set()
    for entry in entries:
        digest, name = entry.split("  ", 1)
        assert name not in names
        names.add(name)
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    if directory == "data":
        assert names == {"freeinference.jsonl", "figure8_reference_summary.csv"}
    else:
        expected = {p.relative_to(root).as_posix() for p in root.glob("*/*.csv")}
        expected.update(p.relative_to(root).as_posix() for p in root.glob("*/args.json"))
        expected.update(p.name for p in root.glob("*.json"))
        assert names == expected


def test_committed_prod_schema_and_filtered_population():
    source = ROOT / "data/freeinference.jsonl"
    count = 0
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            assert set(row) <= set(TRACE_FIELDS) | {"user_id"}
            if "user_id" in row:
                assert re.fullmatch(r"account-\d+", row["user_id"])
            count += 1
    assert count == 24035
    requests = _load_freeinference_jsonl_requests(source)
    assert len(requests) == 21678
    assert requests[0].timestamp == 0.0
    assert requests[-1].timestamp == pytest.approx(608638.2867529392)


@pytest.mark.parametrize(
    ("directory", "n_policies"),
    [
        ("data/real_eval_records", 10),
        ("data/real_eval_records_m3", 11),
        ("data/real_eval_slo_sweep_m3", 4),
    ],
)
def test_committed_real_records_have_only_release_fields(directory, n_policies):
    root = ROOT / directory
    paths = sorted(root.glob("*/requests.csv"))
    assert len(paths) == n_policies
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            assert tuple(reader.fieldnames) == RECORD_FIELDS
            assert sum(1 for _ in reader) == 14233
        args = json.loads(path.with_name("args.json").read_text(encoding="utf-8"))
        assert set(args) <= {"policy", "inventory", "slo_ms"}
        assert (ROOT / args["inventory"]).is_file()


@pytest.mark.parametrize("metric", ["n", "total_cost_usd", "ttft_mean_ms", "slo_violation_rate"])
def test_real_summary_check_rejects_changed_results(tmp_path, metric):
    reference = ROOT / "data/real_eval_records/reference_summary.json"
    rows = json.loads(reference.read_text(encoding="utf-8"))
    rows[0][metric] += 1
    actual = tmp_path / "summary.json"
    actual.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError):
        check_summary(actual, reference)
