# RouteWise

**面向多供应商 LLM 路由的延迟–成本优化**

RouteWise 是一个零运行时依赖的 Python 库，用于决定每一次 LLM API 请求应该发给哪个供应商。
应用提供供应商价格、负责派发返回的尝试，并回报结果，RouteWise 据此学习。

<div class="grid cards" markdown>

-   :material-download: **安装**

    需要 Python 3.10 或更高版本，无运行时依赖。

    [:octicons-arrow-right-24: 安装](guide/installation.md)

-   :material-lightbulb-on: **核心概念**

    供应商、决策、结果回报与路由闭环。

    [:octicons-arrow-right-24: 核心概念](guide/concepts.md)

-   :material-api: **API 参考**

    全部公开类型与方法。

    [:octicons-arrow-right-24: Python API](reference/api.md)

-   :material-fence: **能力边界**

    明确列出这个库不做什么。

    [:octicons-arrow-right-24: 能力边界](guide/boundaries.md)

</div>

## 安装

```bash
python -m pip install llm-routewise
```

!!! warning "包名"

    PyPI 上的 `routewise` 是一个无关且不兼容的项目。请安装 `llm-routewise`
    发行包，并 import `llm_routewise`。

## 路由一次请求

```python
import llm_routewise as rw

router = rw.Router(
    [
        rw.Provider("fast", price_in=3.0, price_out=15.0, price_cached=0.30),
        rw.Provider("cheap", price_in=0.15, price_out=0.60),
    ],
    alpha=0.25,  # 成本预算：0 表示最便宜；1 表示放开整个价格区间以优化延迟。
)

decision = router.route(
    input_tokens=800,
    estimated_cached_tokens=600,  # 预计命中前缀缓存的 prompt 部分。
)
response = call_your_provider(decision.provider)
decision.completed(
    ttft_ms=response.ttft_ms,
    output_tokens=response.output_tokens,
    cached_tokens=response.cached_tokens,
)
```

供应商提供缓存价时，缓存输入按缓存价计费——见[前缀缓存](guide/cost-budget.md#prefix-cache)。

!!! note "RouteWise 只做决策，派发由应用负责"

    `Router` 计算决策，但不发起任何网络 I/O，也不读取 API key。供应商客户端、
    凭据和派发都归应用所有。完整清单见[能力边界](guide/boundaries.md)。

只使用公开 API 的完整离线示例：

```bash
uv run python examples/basic.py
```

## 仓库结构

`main` 分支保存公开库、文档、示例、测试和发布工具。EuroSys 论文 artifact 单独
维护在 [`eurosys27-ae`](https://github.com/HarvardMadSys/RouteWise/tree/eurosys27-ae)
分支上，使其模拟器、实验、数据准备和复现环境保持自包含。

## 引用

```bibtex
@inproceedings{tian2027routewise,
  title     = {{RouteWise}: Latency--Cost Optimization for Multi-Provider LLM Routing},
  author    = {Muxin Tian and Haoran Ni and Yiyan Zhai and Yangsun Park and Juncheng Yang},
  booktitle = {Proceedings of the 22nd European Conference on Computer Systems (EuroSys '27)},
  year      = {2027}
}
```
