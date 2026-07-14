# Changelog

All notable changes to the published `routewise` distribution are documented
here.

## 0.3.0 - Unreleased

`0.3.0` replaces the experimental hosted-service SDK published as
`routewise` `0.1.x`--`0.2.0` with RouteWise's local, dependency-free routing
library.

### Added

- Cost- and latency-aware `Router`, `Decision`, and `Attempt` lifecycle.
- Stateless `route_once` routing for callers that already have cost and
  latency estimates.
- Typed outcome reporting, rolling latency profiles, cold-start exploration,
  cooldowns, hedging, billing provenance, and immutable stats snapshots.
- A narrow, typed wheel supporting Python 3.10--3.14 with no runtime
  dependencies.

### Breaking changes

- Removed the hosted-service `RouteWiseClient` execution layer. RouteWise now
  returns a routing decision and the caller performs provider I/O.
- Removed `AuthError` and `AllTiersFailedError` from the package surface.
- Removed the `requests` dependency.
- Raised the minimum supported Python version from 3.8 to 3.10.

Applications that still require the hosted-service SDK must pin
`routewise<0.3` while that service remains supported.
