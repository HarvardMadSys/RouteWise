# Installation

RouteWise requires Python 3.10 or later. The published wheel has no runtime
dependencies.

=== "pip"

    ```bash
    python -m pip install llm-routewise
    ```

=== "uv"

    ```bash
    uv add llm-routewise
    ```

=== "From source"

    ```bash
    git clone https://github.com/HarvardMadSys/RouteWise.git
    cd RouteWise
    uv sync --locked
    ```

## Verify

```python
import llm_routewise as rw

print(rw.__version__)
```

## Names

| What | Value |
| --- | --- |
| PyPI distribution | `llm-routewise` |
| Python import | `llm_routewise` |
| Unaffiliated project | `routewise` on PyPI |

The distribution name contains a hyphen; the import contains an underscore.

## Repository development

```bash
uv sync --locked
uv run ruff check .
uv run pytest -q
uv run python examples/basic.py
uv run python -m build --wheel
uv run python scripts/check_wheel.py dist/*.whl
```

## Building this site

```bash
uv sync --group docs
uv run mkdocs serve
```
