# Changelog

All notable changes to the published `llm-routewise` distribution are documented
here.

## 0.3.0 - Unreleased

### Added

- Added a MkDocs Material documentation site under `docs/`, published in
  English and Simplified Chinese, with a `docs` dependency group and a
  `docs` workflow that builds it with `--strict`.

### Changed

- Moved the public API reference from `docs/public/API.md` and
  `docs/public/API.zh-CN.md` to `docs/reference/api.md` and
  `docs/reference/api.zh.md`. The `Documentation` project URL follows the move.
- The `main` branch now contains only the maintained public library, its
  documentation, examples, tests, and release tooling. Paper-specific code and
  reproduction material remain available on the
  [`eurosys27-ae`](https://github.com/HarvardMadSys/RouteWise/tree/eurosys27-ae)
  artifact branch.

### Removed

- Removed the repository-only simulator, offline pipeline, metrics package,
  experiments, plotting tools, research CLI, datasets, and research scripts
  from `main`.
- Removed `llm_routewise.capacity` and `llm_routewise.schemas`. These
  simulator-facing compatibility modules were not part of the documented
  top-level API; shared public decision types remain in `llm_routewise.core`.

## 0.2.0 - 2026-07-21

### Added

- `Router.route()` accepts a caller-provided `estimated_output_tokens` point
  estimate while retaining the internal online estimator as the default.

## 0.1.0 - 2026-07-20

`0.1.0` is the first HarvardMadSys RouteWise library release under
the `llm-routewise` distribution. The PyPI project `routewise` is a separate,
unaffiliated project and is not part of this changelog.

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
