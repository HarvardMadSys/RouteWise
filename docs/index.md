# RouteWise

**Latency–Cost Optimization for Multi-Provider LLM Routing**

RouteWise is a dependency-free Python library that decides which LLM API
provider should serve each request. Your application supplies provider prices,
dispatches the returned attempt, and reports the outcome so RouteWise can learn
from it.

<div class="grid cards" markdown>

-   :material-download: **Installation**

    Python 3.10 or later, no runtime dependencies.

    [:octicons-arrow-right-24: Install](guide/installation.md)

-   :material-lightbulb-on: **Core concepts**

    Providers, decisions, outcome feedback, and the routing loop.

    [:octicons-arrow-right-24: Concepts](guide/concepts.md)

-   :material-api: **API reference**

    Every public type and method.

    [:octicons-arrow-right-24: Python API](reference/api.md)

-   :material-fence: **API boundaries**

    What the library does not do, stated explicitly.

    [:octicons-arrow-right-24: Boundaries](guide/boundaries.md)

</div>

## Install

```bash
python -m pip install llm-routewise
```

!!! warning "Package name"

    The PyPI project `routewise` is an unaffiliated, incompatible project.
    Install the `llm-routewise` distribution and import the `llm_routewise`
    package.

## Route a request

```python
import llm_routewise as rw

router = rw.Router(
    [
        rw.Provider("fast", price_in=3.0, price_out=15.0, price_cached=0.30),
        rw.Provider("cheap", price_in=0.15, price_out=0.60),
    ],
    alpha=0.25,  # Cost budget: 0 = cheapest; 1 = full range for latency.
)

decision = router.route(
    input_tokens=800,
    estimated_cached_tokens=600,  # Prompt prefix you expect to hit cache.
)
response = call_your_provider(decision.provider)
decision.completed(
    ttft_ms=response.ttft_ms,
    output_tokens=response.output_tokens,
    cached_tokens=response.cached_tokens,
)
```

Cached input is priced at the provider's cached rate when it has one — see
[Prefix cache](guide/cost-budget.md#prefix-cache).

!!! note "RouteWise decides; your application dispatches"

    `Router` computes decisions but performs no network I/O and does not read
    API keys. Your application owns provider clients, credentials, and
    dispatch. See [API boundaries](guide/boundaries.md) for the full list.

For a complete offline example using only the public API:

```bash
uv run python examples/basic.py
```

## Repository layout

The `main` branch holds the public library, its documentation, examples, tests,
and release tooling. The EuroSys paper artifact lives separately on the
[`eurosys27-ae`](https://github.com/HarvardMadSys/RouteWise/tree/eurosys27-ae)
branch so its simulator, experiments, data preparation, and reproduction
environment stay self-contained.

## Citation

```bibtex
@inproceedings{tian2027routewise,
  title     = {{RouteWise}: Latency--Cost Optimization for Multi-Provider LLM Routing},
  author    = {Muxin Tian and Haoran Ni and Yiyan Zhai and Yangsun Park and Juncheng Yang},
  booktitle = {Proceedings of the 22nd European Conference on Computer Systems (EuroSys '27)},
  year      = {2027}
}
```
