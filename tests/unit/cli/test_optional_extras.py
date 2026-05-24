"""CLI optional dependency diagnostics."""

from __future__ import annotations

import pytest

from routewise_cli import main as cli_main


def test_simulator_command_reports_missing_sim_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """A lightweight base install should fail with an install hint, not a traceback."""

    def fail_import(module_name: str):
        assert module_name == cli_main.SIMULATOR_SECTIONS["cost-layer"]
        raise ModuleNotFoundError("No module named 'numpy'", name="numpy")

    monkeypatch.setattr(cli_main.importlib, "import_module", fail_import)

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["simulator", "cost-layer"])

    message = str(exc_info.value)
    assert "routewise simulator" in message
    assert "routewise[sim]" in message
    assert "pip install -e '.[sim]'" in message
    assert "Missing module: numpy" in message


def test_ablation_command_reports_missing_sim_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ablation harnesses share simulator dependencies and should say so."""

    def fail_import(module_name: str):
        assert module_name == cli_main.ABLATION_COMMANDS["hedging"]
        raise ModuleNotFoundError("No module named 'numpy'", name="numpy")

    monkeypatch.setattr(cli_main.importlib, "import_module", fail_import)

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["ablation", "hedging"])

    message = str(exc_info.value)
    assert "routewise ablation" in message
    assert "routewise[sim]" in message
    assert "Missing module: numpy" in message
