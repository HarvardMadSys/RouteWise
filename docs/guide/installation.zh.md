# 安装

RouteWise 需要 Python 3.10 或更高版本。发布的 wheel 没有运行时依赖。

=== "pip"

    ```bash
    python -m pip install llm-routewise
    ```

=== "uv"

    ```bash
    uv add llm-routewise
    ```

=== "从源码"

    ```bash
    git clone https://github.com/HarvardMadSys/RouteWise.git
    cd RouteWise
    uv sync --locked
    ```

## 验证

```python
import llm_routewise as rw

print(rw.__version__)
```

## 名称

| 项目 | 值 |
| --- | --- |
| PyPI 发行包 | `llm-routewise` |
| Python import | `llm_routewise` |
| 无关项目 | PyPI 上的 `routewise` |

发行包名用连字符，import 名用下划线。

## 仓库开发

```bash
uv sync --locked
uv run ruff check .
uv run pytest -q
uv run python examples/basic.py
uv run python -m build --wheel
uv run python scripts/check_wheel.py dist/*.whl
```

## 构建本站

```bash
uv sync --group docs
uv run mkdocs serve
```
