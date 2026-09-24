"""Shared runtime for the agentic benchmark.

Holds the gateway-facing pieces reused across SWE-bench and Terminal-Bench:
experiment config, request/record schemas, the gateway client, the recorder that
reads per-request cost/latency back, and the metrics/analysis helpers.
"""
