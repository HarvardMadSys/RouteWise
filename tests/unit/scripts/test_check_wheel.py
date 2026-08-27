from __future__ import annotations

from typing import TYPE_CHECKING
from zipfile import ZipFile

import pytest

from scripts.check_wheel import ALLOWED_LIBRARY_MEMBERS, _expected_project_metadata, main

if TYPE_CHECKING:
    from pathlib import Path


def _write_wheel(
    path: Path,
    *,
    name: str = "llm-routewise",
    version: str | None = None,
    requires_python: str = ">=3.10",
    requires_dist: tuple[str, ...] = (),
    extra_members: tuple[str, ...] = (),
) -> None:
    if version is None:
        _, version, _ = _expected_project_metadata()
    metadata = [
        "Metadata-Version: 2.4",
        f"Name: {name}",
        f"Version: {version}",
        f"Requires-Python: {requires_python}",
    ]
    metadata.extend(f"Requires-Dist: {requirement}" for requirement in requires_dist)
    metadata.append("")
    dist_info = f"{name.replace('-', '_')}-{version}.dist-info"
    with ZipFile(path, "w") as archive:
        for member in ALLOWED_LIBRARY_MEMBERS:
            archive.writestr(member, "")
        for member in extra_members:
            archive.writestr(member, "")
        archive.writestr(f"{dist_info}/METADATA", "\n".join(metadata))
        archive.writestr(f"{dist_info}/WHEEL", "Wheel-Version: 1.0\n")
        archive.writestr(f"{dist_info}/RECORD", "")


def test_accepts_exact_dependency_free_metadata(tmp_path: Path) -> None:
    wheel = tmp_path / "artifact.whl"
    _write_wheel(wheel)

    assert main(["check_wheel.py", str(wheel)]) == 0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"name": "other"}, "METADATA Name"),
        ({"version": "999.0.0"}, "METADATA Version"),
        ({"requires_python": ">=3.8"}, "METADATA Requires-Python"),
        ({"requires_dist": ("requests",)}, "runtime dependencies"),
    ],
)
def test_rejects_incorrect_or_dependent_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    overrides: dict[str, object],
    message: str,
) -> None:
    wheel = tmp_path / "artifact.whl"
    _write_wheel(wheel, **overrides)  # type: ignore[arg-type]

    assert main(["check_wheel.py", str(wheel)]) == 1
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "member",
    [
        "llm_routewise/capacity.py",
        "llm_routewise/schemas.py",
    ],
)
def test_rejects_research_only_library_members(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    member: str,
) -> None:
    wheel = tmp_path / "artifact.whl"
    _write_wheel(wheel, extra_members=(member,))

    assert main(["check_wheel.py", str(wheel)]) == 1
    assert member in capsys.readouterr().err
