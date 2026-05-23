from __future__ import annotations

from importlib.metadata import version as package_version
from pathlib import Path
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_setuptools_discovery_only_packages_vmec_jax_namespace() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    package_find = data["tool"]["setuptools"]["packages"]["find"]

    assert package_find["where"] == ["."]
    assert package_find["include"] == ["vmec_jax*"]
    for pattern in ("tests*", "docs*", "examples*", "tools*", "validation*", "results*", "build*", "dist*"):
        assert pattern in package_find["exclude"]


def test_package_exposes_installed_version() -> None:
    import vmec_jax

    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert vmec_jax.__version__ == data["project"]["version"]
    if not Path(vmec_jax.__file__).resolve().is_relative_to(ROOT):
        assert vmec_jax.__version__ == package_version("vmec-jax")


def test_project_metadata_has_public_package_links() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = data["project"]

    assert "stellarator" in project["keywords"]
    assert "Topic :: Scientific/Engineering :: Physics" in project["classifiers"]
    assert project["urls"]["Documentation"] == "https://vmec-jax.readthedocs.io/en/latest/"
    assert project["urls"]["Repository"] == "https://github.com/uwplasma/vmec_jax"
    assert project["urls"]["Changelog"] == "https://github.com/uwplasma/vmec_jax/releases"
