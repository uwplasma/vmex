from __future__ import annotations

from importlib.metadata import version as package_version

from packaging.requirements import Requirement
from packaging.version import Version
from pathlib import Path
import re
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_setuptools_discovery_only_packages_vmex_namespace() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    package_find = data["tool"]["setuptools"]["packages"]["find"]

    assert package_find["where"] == ["."]
    assert package_find["include"] == ["vmex*"]
    for pattern in ("tests*", "docs*", "examples*", "tools*", "validation*", "results*", "build*", "dist*"):
        assert pattern in package_find["exclude"]


def test_package_exposes_installed_version() -> None:
    import vmex

    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert vmex.__version__ == data["project"]["version"]
    if not Path(vmex.__file__).resolve().is_relative_to(ROOT):
        assert vmex.__version__ == package_version("vmex")


def test_citation_version_matches_the_package_version() -> None:
    """CITATION.cff is part of the release, so it has to move with it.

    Nothing else compares the two, and a stale citation version is invisible
    until someone cites the wrong release.
    """
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    citation = (ROOT / "CITATION.cff").read_text()

    declared = next(
        line.split(":", 1)[1].strip()
        for line in citation.splitlines()
        if line.startswith("version:")
    )
    assert declared == data["project"]["version"], (
        f"CITATION.cff says {declared!r}, pyproject.toml says "
        f"{data['project']['version']!r}"
    )


def test_project_metadata_has_public_package_links() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = data["project"]

    assert "stellarator" in project["keywords"]
    assert "Topic :: Scientific/Engineering :: Physics" in project["classifiers"]
    assert project["urls"]["Documentation"] == "https://vmex.readthedocs.io/en/latest/"
    assert project["urls"]["Repository"] == "https://github.com/uwplasma/vmex"
    assert project["urls"]["Changelog"] == "https://github.com/uwplasma/vmex/releases"


def test_project_exposes_vmec_console_aliases() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    scripts = data["project"]["scripts"]

    # VMEX CLI: primary `vmex` + the legacy `vmec` alias.
    assert scripts["vmex"] == "vmex.core.cli:main"
    assert scripts["vmec"] == "vmex.core.cli:main"
    assert set(scripts) == {"vmex", "vmec"}


def test_plain_install_includes_plotting_and_qi_dependencies() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project_dependencies = set(data["project"]["dependencies"])
    # Compare parsed requirement NAMES, not raw strings: a version floor on
    # any dependency (booz_xform_jax>=0.1.1 broke the old string match) must
    # not trip the plain-install guard.
    dependency_names = {
        Requirement(dep).name for dep in project_dependencies
    }
    optional_dependencies = data.get("project", {}).get("optional-dependencies", {})

    assert "matplotlib" in dependency_names
    assert "booz_xform_jax" in dependency_names
    assert "packaging" in dependency_names
    assert "numpy" in dependency_names
    assert "solvax>=0.27.0" in project_dependencies
    assert "gkx>=2.5.0" in optional_dependencies["turbulence"]
    assert "virtual-casing-jax>=0.0.9" in optional_dependencies["freeb"]
    assert "plots" not in optional_dependencies
    assert "plot" not in optional_dependencies
    assert "qi" not in optional_dependencies
    assert "booz" not in optional_dependencies


def test_build_system_declares_setuptools_license_validation_dependency() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    build_requires = set(data["build-system"]["requires"])

    assert "setuptools" in build_requires
    assert "packaging" in build_requires


def test_import_guard_floors_match_pyproject() -> None:
    import vmex

    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert data["project"]["requires-python"] == ">=3.11"
    floors = {}
    for requirement in map(Requirement, data["project"]["dependencies"]):
        if requirement.name in vmex._MINIMUM_VERSIONS:
            (spec,) = requirement.specifier
            assert spec.operator == ">="
            floors[requirement.name] = Version(spec.version).release
    assert floors == vmex._MINIMUM_VERSIONS


def test_optional_floors_match_pyproject() -> None:
    from vmex._compat import OPTIONAL_MINIMUMS

    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    floors = {}
    for extra in data["project"]["optional-dependencies"].values():
        for requirement in map(Requirement, extra):
            if requirement.name in OPTIONAL_MINIMUMS:
                (spec,) = requirement.specifier
                floors[requirement.name] = spec.version
    assert floors == OPTIONAL_MINIMUMS


@pytest.mark.parametrize("found, message", [
    (None, 'essos>=0.20.0; run: pip install "essos>=0.20.0"'),
    ("0.19.2", 'essos>=0.20.0 (found 0.19.2); run: pip install -U "essos>=0.20.0"'),
])
def test_require_optional_names_the_fix(monkeypatch, found, message) -> None:
    from vmex import _compat

    def version(name):
        if found is None:
            raise _compat.importlib_metadata.PackageNotFoundError(name)
        return found

    monkeypatch.setattr(_compat.importlib_metadata, "version", version)
    with pytest.raises(ImportError, match=re.escape("alpha-particle tracing needs " + message)):
        _compat.require_optional("essos", "alpha-particle tracing")


def test_import_guard_names_found_required_and_fix(monkeypatch) -> None:
    import vmex

    monkeypatch.setattr(
        vmex, "_package_version",
        lambda name: "1.15.3" if name == "scipy" else "0.11.1")
    with pytest.raises(ImportError, match=(
            r'scipy >= 1\.16 \(found 1\.15\.3\).*pip install -U "scipy>=1\.16"')):
        vmex._check_supported_versions()


def test_import_guard_rejects_python_310(monkeypatch) -> None:
    from types import SimpleNamespace

    import vmex

    monkeypatch.setattr(vmex, "_sys", SimpleNamespace(
        version_info=(3, 10, 14), version="3.10.14 (main, Mar 1 2026)"))
    with pytest.raises(ImportError, match=r"Python >= 3\.11 \(found 3\.10\.14\)"):
        vmex._check_supported_versions()


def test_import_guard_skips_packages_without_metadata(monkeypatch) -> None:
    import vmex

    def version(name):
        if name == "scipy":
            raise vmex._PackageNotFoundError(name)
        return "99.0"  # above every floor

    monkeypatch.setattr(vmex, "_package_version", version)
    assert vmex._check_supported_versions() is None


def test_nightly_floors_lane_pins_the_declared_floors() -> None:
    """The nightly "floors" lane installs exactly the floors pyproject declares."""
    import re

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    extras = project["optional-dependencies"]
    floors = {}
    for dep in [*project["dependencies"], *(r for name in
                ("coils", "freeb", "neoclassical", "turbulence") for r in extras[name])]:
        requirement = Requirement(dep)
        for spec in requirement.specifier:
            floors[requirement.name] = max(floors.get(requirement.name, spec.version),
                                           spec.version, key=Version)
    workflow = (ROOT / ".github/workflows/nightly.yml").read_text()
    lane = workflow[workflow.index("if: matrix.install == 'floors'"):]
    lane = lane[:lane.index("- name:")]
    pins = dict(re.findall(r"([A-Za-z0-9_.-]+)==(\S+)", lane))
    for name, floor in floors.items():
        assert name in pins and Version(pins[name]) == Version(floor), (name, floor)
