<p align="center">
  <img
    src="https://raw.githubusercontent.com/HarvardMadSys/RouteWise/main/docs/assets/harvard-seas-wordmark.png"
    alt="Harvard John A. Paulson School of Engineering and Applied Sciences"
    width="270"
  >
</p>

<h1 align="center">RouteWise</h1>

<p align="center">
  <strong>Latency–Cost Optimization for Multi-Provider LLM Routing</strong>
  <br>
  <sub>Learn from recent outcomes. Route each request. Hedge selectively.</sub>
</p>

<p align="center">
  Developed and maintained by the
  <a href="https://juncheng.seas.harvard.edu/" title="Harvard Measurements and Design of Computer Systems Lab">Harvard MadSys Lab</a>
  at <a href="https://seas.harvard.edu/">Harvard SEAS</a>.
</p>

<p align="center">
  <a href="https://github.com/HarvardMadSys/RouteWise/stargazers"><img alt="GitHub Stars" src="https://img.shields.io/github/stars/HarvardMadSys/RouteWise?style=flat&amp;logo=github&amp;label=Stars"></a>
  <a href="https://github.com/HarvardMadSys/RouteWise/forks"><img alt="GitHub Forks" src="https://img.shields.io/github/forks/HarvardMadSys/RouteWise?style=flat&amp;logo=github&amp;label=Forks"></a>
</p>

<p align="center">
  <a href="https://github.com/HarvardMadSys/RouteWise/blob/main/docs/public/API.md">English API</a>
  ·
  <a href="https://github.com/HarvardMadSys/RouteWise/blob/main/docs/public/API.zh-CN.md">中文 API</a>
  ·
  <a href="https://github.com/HarvardMadSys/RouteWise/blob/main/docs/research/REPRODUCIBILITY.md">Reproducibility</a>
</p>

<p align="center">
  <img
    src="https://raw.githubusercontent.com/HarvardMadSys/RouteWise/main/docs/assets/routewise-routing-hero.svg"
    alt="RouteWise adaptive routing loop: request, RouteWise cost and latency policy, decision, application dispatch, and outcome feedback"
    width="1000"
  >
</p>

RouteWise is a dependency-free Python library for cost-aware,
latency-optimized routing across multiple LLM API providers. Applications
supply provider prices, dispatch the returned attempt, and report outcomes so
RouteWise can learn from them.

The `0.2.0` distribution is an API-provider-only preview. This repository also
contains the simulator and experiment harnesses used by the paper; those
research packages are not included in the wheel.

> **Package name:** The PyPI project `routewise` is an unaffiliated,
> incompatible project. Install the `llm-routewise` distribution and import
> the `llm_routewise` package for HarvardMadSys RouteWise.

## Installation

RouteWise requires Python 3.10 or later. The published wheel has no runtime
dependencies.

```bash
python -m pip install llm-routewise==0.2.0
```

For repository development and paper-artifact workflows:

```bash
git clone https://github.com/HarvardMadSys/RouteWise.git
cd RouteWise
uv sync
```

## Quickstart

```python
import llm_routewise as rw

router = rw.Router(
    [
        rw.Provider("fast", price_in=3.0, price_out=15.0),
        rw.Provider("cheap", price_in=0.15, price_out=0.60),
    ],
    alpha=0.25,  # Cost budget: 0 = cheapest; 1 = full range for latency optimization.
)

decision = router.route(input_tokens=800)
response = call_your_provider(decision.provider)
decision.completed(
    ttft_ms=response.ttft_ms,
    output_tokens=response.output_tokens,
)
```

If your application already predicts completion length, pass that point
estimate with the request. Omit it to use RouteWise's internal online estimate.

```python
predicted_tokens = predict_output_tokens(prompt)
decision = router.route(
    input_tokens=800,
    estimated_output_tokens=predicted_tokens,
)
```

The estimate affects route and hedge cost calculations only; it is not actual
usage. On completion, report the adopted attempt's actual `output_tokens` (or
an explicit `cost_usd`) for billing. Positive actual output tokens also update
RouteWise's internal estimator.

`Router` computes decisions but performs no network I/O and does not read API
keys. Your application owns provider clients, credentials, and dispatch. Read
the [English API reference](https://github.com/HarvardMadSys/RouteWise/blob/main/docs/public/API.md)
or [中文 API 参考](https://github.com/HarvardMadSys/RouteWise/blob/main/docs/public/API.zh-CN.md)
for the full contract.

## Repository Development

Run the fast test suite:

```bash
uv run pytest -q -m "not slow"
```

List the paper-facing simulator sections:

```bash
uv run python -m routewise_cli.main simulator list
```

The [reproducibility guide](https://github.com/HarvardMadSys/RouteWise/blob/main/docs/research/REPRODUCIBILITY.md)
covers datasets, live-evaluation credentials, experiment commands, and
regression checks.

## Documentation

### Library Users

- [Python API](https://github.com/HarvardMadSys/RouteWise/blob/main/docs/public/API.md)
- [Python API, Chinese](https://github.com/HarvardMadSys/RouteWise/blob/main/docs/public/API.zh-CN.md)

### Research Artifacts

- [Simulator architecture and algorithms](https://github.com/HarvardMadSys/RouteWise/blob/main/docs/research/ARCHITECTURE.md)
- [Experiment reproducibility](https://github.com/HarvardMadSys/RouteWise/blob/main/docs/research/REPRODUCIBILITY.md)

### Maintainers and Advanced Integrators

- [Core mathematical API](https://github.com/HarvardMadSys/RouteWise/blob/main/docs/maintainers/CORE_API.md)

## License

MIT. See the [license](https://github.com/HarvardMadSys/RouteWise/blob/main/LICENSE).
