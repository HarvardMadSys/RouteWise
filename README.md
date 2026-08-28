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
  <a href="https://pypi.org/project/llm-routewise/"><img alt="PyPI" src="https://img.shields.io/pypi/v/llm-routewise?style=flat-square&amp;label=PyPI&amp;color=A51C30"></a>
  <a href="https://github.com/HarvardMadSys/RouteWise/actions/workflows/package.yml"><img alt="Package CI" src="https://github.com/HarvardMadSys/RouteWise/actions/workflows/package.yml/badge.svg?branch=main"></a>
  <a href="https://pypi.org/project/llm-routewise/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/llm-routewise?style=flat-square&amp;color=4B5563"></a>
  <a href="https://github.com/HarvardMadSys/RouteWise/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/pypi/l/llm-routewise?style=flat-square&amp;color=4B5563"></a>
  <a href="https://github.com/HarvardMadSys/RouteWise/stargazers"><img alt="GitHub Stars" src="https://img.shields.io/github/stars/HarvardMadSys/RouteWise?style=flat&amp;logo=github&amp;label=Stars"></a>
  <a href="https://github.com/HarvardMadSys/RouteWise/forks"><img alt="GitHub Forks" src="https://img.shields.io/github/forks/HarvardMadSys/RouteWise?style=flat&amp;logo=github&amp;label=Forks"></a>
</p>

<p align="center">
  <a href="https://github.com/HarvardMadSys/RouteWise/blob/main/docs/public/API.md">English API</a>
  ·
  <a href="https://github.com/HarvardMadSys/RouteWise/blob/main/docs/public/API.zh-CN.md">中文 API</a>
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

The `main` branch contains the public library, its documentation, examples,
tests, and release tooling. The EuroSys paper artifact is maintained separately
on the [`eurosys27-ae`](https://github.com/HarvardMadSys/RouteWise/tree/eurosys27-ae)
branch so that its simulator, experiments, data preparation, and reproduction
environment remain self-contained.

> **Package name:** The PyPI project `routewise` is an unaffiliated,
> incompatible project. Install the `llm-routewise` distribution and import
> the `llm_routewise` package for HarvardMadSys RouteWise.

## News

- **August 2026:** Our paper, *RouteWise: Latency–Cost Optimization for
  Multi-Provider LLM Routing*, has been accepted to
  [EuroSys 2027](https://2027.eurosys.org/)! 🎉

## Installation

RouteWise requires Python 3.10 or later. The published wheel has no runtime
dependencies.

```bash
python -m pip install llm-routewise
```

For repository development:

```bash
git clone https://github.com/HarvardMadSys/RouteWise.git
cd RouteWise
uv sync --locked
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

Two offline examples use only the public API: a single decision, and a full
request lifecycle showing dispatch, failure reporting, and how outcomes change
routing. Run them with:

```bash
uv run python examples/basic.py
uv run python examples/simple_router.py
```

## Repository Development

Install the development tools and run the complete library checks:

```bash
uv sync --locked
uv run ruff check .
uv run pytest -q
uv run python examples/basic.py
uv run python examples/simple_router.py
uv run python -m build --wheel
uv run python scripts/check_wheel.py dist/*.whl
```

## Documentation

### Library Users

- [Python API](https://github.com/HarvardMadSys/RouteWise/blob/main/docs/public/API.md)
- [Python API, Chinese](https://github.com/HarvardMadSys/RouteWise/blob/main/docs/public/API.zh-CN.md)

### Maintainers and Advanced Integrators

- [Core mathematical API](https://github.com/HarvardMadSys/RouteWise/blob/main/docs/maintainers/CORE_API.md)
- [Release procedure](https://github.com/HarvardMadSys/RouteWise/blob/main/docs/maintainers/RELEASING.md)
- [Published-package changes](https://github.com/HarvardMadSys/RouteWise/blob/main/CHANGELOG.md)

## Citation

If you use RouteWise in your research, please cite our paper:

```bibtex
@inproceedings{tian2027routewise,
  title     = {{RouteWise}: Latency--Cost Optimization for Multi-Provider LLM Routing},
  author    = {Muxin Tian and Haoran Ni and Yiyan Zhai and Yangsun Park and Juncheng Yang},
  booktitle = {Proceedings of the 22nd European Conference on Computer Systems (EuroSys '27)},
  year      = {2027}
}
```

## License

MIT. See the [license](https://github.com/HarvardMadSys/RouteWise/blob/main/LICENSE).
