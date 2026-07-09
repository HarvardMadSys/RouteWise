"""End-to-end agentic benchmark for RouteWise.

Drives the live hybridInference gateway (freeinference.org) with a coding-agent
workload (SWE-bench via mini-SWE-agent) and measures the cost and latency of
RouteWise routing against single-provider baselines.

The gateway owns the routing decision; this package only generates the workload,
tags each request, and reads cost/latency back from the gateway's per-request log.
"""
