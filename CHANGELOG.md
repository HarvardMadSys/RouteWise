# Changelog

All notable changes to the published `llm-routewise` distribution are documented
here.

## 0.1.0 - Unreleased

`0.1.0` is planned as the first HarvardMadSys RouteWise library release under
the `llm-routewise` distribution. The `routewise` `0.1.x`--`0.2.0`
distributions currently on PyPI were published by an unaffiliated project and
are not part of this changelog.

### Added

- Cost- and latency-aware `Router`, `Decision`, and `Attempt` lifecycle.
- Stateless `route_once` routing for callers that already have cost and
  latency estimates.
- Typed outcome reporting, rolling latency profiles, cold-start exploration,
  cooldowns, hedging, billing provenance, and immutable stats snapshots.
- A narrow, typed wheel supporting Python 3.10--3.14 with no runtime
  dependencies.
- The `llm_routewise` import namespace, including the conventional
  `import llm_routewise as rw` alias.

### Package boundary

- RouteWise returns a routing decision and the caller performs provider I/O.
- The public surface deliberately excludes hosted-client symbols such as
  `RouteWiseClient`, `AuthError`, and `AllTiersFailedError`.
- The library has no runtime dependency and requires Python 3.10 or newer.
