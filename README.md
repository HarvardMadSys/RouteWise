# RouteWise

RouteWise is a dependency-free Python library for cost-aware,
latency-optimized routing across multiple LLM API providers. Applications
supply provider prices, dispatch the returned attempt, and report outcomes so
RouteWise can learn from them.

The `0.1.0` distribution is an API-provider-only preview. This repository also
contains the simulator and experiment harnesses used by the paper; those
research packages are not included in the wheel.

> **Package name:** The PyPI project `routewise` is an unaffiliated,
> incompatible project. Install the `llm-routewise` distribution and import
> the `llm_routewise` package for HarvardMadSys RouteWise.

## Installation

RouteWise requires Python 3.10 or later. The published wheel has no runtime
dependencies.

```bash
python -m pip install llm-routewise==0.1.0
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
    alpha=0.25,
)

decision = router.route(input_tokens=800)
response = call_your_provider(decision.provider)
decision.completed(
    ttft_ms=response.ttft_ms,
    output_tokens=response.output_tokens,
)
```

`Router` computes decisions but performs no network I/O and does not read API
keys. Your application owns provider clients, credentials, and dispatch. Read
the [English API reference](docs/public/API.md) or
[中文 API 参考](docs/public/API.zh-CN.md) for the full contract.

## Repository Development

Run the fast test suite:

```bash
uv run pytest -q -m "not slow"
```

List the paper-facing simulator sections:

```bash
uv run python -m routewise_cli.main simulator list
```

The [reproducibility guide](docs/research/REPRODUCIBILITY.md) covers datasets,
live-evaluation credentials, experiment commands, and regression checks.

## Documentation

### Library Users

- [Python API](docs/public/API.md)
- [Python API, Chinese](docs/public/API.zh-CN.md)

### Research Artifacts

- [Simulator architecture and algorithms](docs/research/ARCHITECTURE.md)
- [Experiment reproducibility](docs/research/REPRODUCIBILITY.md)

### Maintainers and Advanced Integrators

- [Core mathematical API](docs/maintainers/CORE_API.md)
- [Release procedure](docs/maintainers/RELEASING.md)
- [Published-package changes](CHANGELOG.md)

## License

MIT. See [LICENSE](LICENSE).
