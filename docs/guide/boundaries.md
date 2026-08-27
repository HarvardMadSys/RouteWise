# API boundaries

RouteWise is deliberately narrow. This page states what it does not do, so you
can tell early whether it fits.

## Pricing

- Only on-demand, metered per-token provider prices are represented.
- Quotas, concurrency limits, reserved capacity, and subscription pricing are
  not part of this API.

## Transport

- There is no general LLM client, provider SDK adapter, network transport,
  authentication, or API-key management.
- Your application owns provider clients, credentials, dispatch, retries, and
  response handling.

## Selection

- RouteWise selects among the provider names you configured. Endpoint and model
  mapping stays with your application.
- It does not perform model selection.

## State

- Observations, cooldowns, leases, estimates, random state, and counters live
  in the current Python process.
- Nothing is persisted or shared across processes.
- `observe` cannot ingest historical timestamps.

## Research code

Paper-specific simulator and experiment tooling are maintained on the
[`eurosys27-ae`](https://github.com/HarvardMadSys/RouteWise/tree/eurosys27-ae)
artifact branch. They are not part of this library API.

## Why this matters

The boundary is the reason the wheel has no runtime dependencies and the reason
RouteWise can sit inside an application that already has its own client stack.
A router that owned transport would have to own credentials, retries, and
provider SDK versions too.
