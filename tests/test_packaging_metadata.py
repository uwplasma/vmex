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


def test_project_exposes_vmec_console_aliases() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    scripts = data["project"]["scripts"]

    assert scripts["vmec"] == "vmec_jax.cli:main"
    assert scripts["vmec-jax"] == "vmec_jax.cli:main"
    assert scripts["vmec_jax"] == "vmec_jax.cli:main"
    assert scripts["xvmec_jax"] == "vmec_jax.cli:main"


def test_plain_install_includes_plotting_and_qi_dependencies() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project_dependencies = set(data["project"]["dependencies"])
    optional_dependencies = data.get("project", {}).get("optional-dependencies", {})

    assert "matplotlib>=3.8" in project_dependencies
    assert "booz_xform_jax>=0.1.1" in project_dependencies
    assert "packaging>=24.2" in project_dependencies
    assert "numpy>=1.26,<3" in project_dependencies
    assert "plots" not in optional_dependencies
    assert "plot" not in optional_dependencies
    assert "qi" not in optional_dependencies
    assert "booz" not in optional_dependencies


def test_build_system_declares_setuptools_license_validation_dependency() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    build_requires = set(data["build-system"]["requires"])

    assert "setuptools>=77" in build_requires
    assert "packaging>=24.2" in build_requires
