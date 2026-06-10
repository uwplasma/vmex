from __future__ import annotations

import builtins
import subprocess
from importlib import metadata
from types import SimpleNamespace

import vmec_jax.doctor as doctor
from vmec_jax.doctor import DoctorReport, format_report


def test_doctor_format_report_shows_clean_status() -> None:
    report = DoctorReport(
        python="3.11.0",
        executable="/venv/bin/python",
        prefix="/venv",
        base_prefix="/usr",
        platform="Darwin 0 arm64",
        in_virtualenv=True,
        conda_prefix=None,
        user_site="/Users/example/Library/Python/3.11/lib/python/site-packages",
        user_site_on_path=False,
        pip_report="pip 26 from /venv/lib/python/site-packages/pip (python 3.11)",
        versions={
            "vmec-jax": "0.0.14",
            "numpy": "2.0",
            "jax": "0.9",
            "jaxlib": "0.9",
            "scipy": "1.14",
            "netCDF4": "1.7",
            "matplotlib": "3.10",
            "booz_xform_jax": "0.1.1",
            "setuptools": "82.0.1",
            "packaging": "26.2",
            "pip": "26.1",
        },
        jax_backend="cpu",
        jax_devices=("TFRT_CPU_0",),
        warnings=(),
    )

    text = format_report(report)

    assert "vmec_jax installation doctor" in text
    assert "Status: no obvious installation problems detected." in text
    assert "JAX backend: cpu" in text
    assert "booz_xform_jax" in text


def test_doctor_format_report_shows_recommended_clean_install_for_warnings() -> None:
    report = DoctorReport(
        python="3.11.0",
        executable="/opt/homebrew/bin/python",
        prefix="/opt/homebrew",
        base_prefix="/opt/homebrew",
        platform="Darwin 0 arm64",
        in_virtualenv=False,
        conda_prefix=None,
        user_site="/Users/example/Library/Python/3.11/lib/python/site-packages",
        user_site_on_path=True,
        pip_report="pip 26 from /Users/example/Library/Python/3.11/lib/python/site-packages/pip (python 3.11)",
        versions={
            "vmec-jax": "not installed",
            "numpy": "2.2",
            "jax": "0.9",
            "jaxlib": "0.9",
            "scipy": "1.14",
            "netCDF4": "1.7",
            "matplotlib": "3.10",
            "booz_xform_jax": "0.1.1",
            "setuptools": "82.0.1",
            "packaging": "24.0",
            "pip": "26.0",
        },
        jax_backend=None,
        jax_devices=(),
        warnings=("packaging may be too old for current setuptools license validation.",),
    )

    text = format_report(report)

    assert "Warnings:" in text
    assert "packaging may be too old" in text
    assert "Recommended clean install:" in text
    assert "pip install vmec-jax" in text


def test_doctor_package_version_and_version_helpers_cover_error_paths(monkeypatch) -> None:
    def fake_version(name: str) -> str:
        if name == "missing":
            raise metadata.PackageNotFoundError(name)
        if name == "broken":
            raise RuntimeError("metadata database unavailable")
        return "24.2"

    monkeypatch.setattr(doctor.metadata, "version", fake_version)

    assert doctor._package_version("missing") == "not installed"
    assert doctor._package_version("broken") == "error: metadata database unavailable"
    assert doctor._package_version("packaging") == "24.2"
    assert doctor._version_at_least("24.2", "24.2")
    assert not doctor._version_at_least("not installed", "24.2")


def test_doctor_pip_and_user_site_helpers_cover_local_failures(monkeypatch) -> None:
    monkeypatch.setattr(
        doctor.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="", stderr="pip failed", returncode=2),
    )
    assert doctor._pip_report() == "pip failed"

    monkeypatch.setattr(
        doctor.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="", stderr="", returncode=9),
    )
    assert doctor._pip_report() == "pip exited with code 9"

    def raise_timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("pip", timeout=15)

    monkeypatch.setattr(doctor.subprocess, "run", raise_timeout)
    assert doctor._pip_report().startswith("unavailable:")

    monkeypatch.setattr(doctor.site, "getusersitepackages", lambda: (_ for _ in ()).throw(RuntimeError("site broken")))
    assert doctor._user_site() is None


def test_doctor_jax_info_reports_import_failure(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "jax":
            raise RuntimeError("jax backend unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    backend, devices, error = doctor._jax_info()

    assert backend is None
    assert devices == ()
    assert error == "jax backend unavailable"


def test_doctor_collect_report_detects_mixed_prefix_and_missing_packages(monkeypatch) -> None:
    versions = {name: "1.0" for name in doctor._CORE_PACKAGES}
    versions.update({"setuptools": "not installed", "packaging": "24.0", "pip": "not installed"})

    monkeypatch.setattr(doctor, "_package_version", lambda name: versions[name])
    monkeypatch.setattr(doctor, "_user_site", lambda: "/Users/example/Library/Python/3.11/lib/python/site-packages")
    monkeypatch.setattr(
        doctor,
        "_pip_report",
        lambda: "pip 26 from /Users/example/Library/Python/3.11/lib/python/site-packages/pip (python 3.11)",
    )
    monkeypatch.setattr(doctor, "_jax_info", lambda: (None, (), "no jax backend"))
    monkeypatch.setattr(doctor.sys, "prefix", "/opt/homebrew")
    monkeypatch.setattr(doctor.sys, "base_prefix", "/opt/homebrew")
    monkeypatch.setattr(doctor.sys, "executable", "/opt/homebrew/bin/python")
    monkeypatch.setattr(doctor.sys, "path", ["/Users/example/Library/Python/3.11/lib/python/site-packages"])
    monkeypatch.delenv("CONDA_PREFIX", raising=False)

    report = doctor.collect_report()

    assert report.in_virtualenv is False
    assert report.user_site_on_path is True
    assert report.jax_backend is None
    assert any("setuptools is not installed" in warning for warning in report.warnings)
    assert any("packaging may be too old" in warning for warning in report.warnings)
    assert any("pip is not installed" in warning for warning in report.warnings)
    assert any("user-site packages are on sys.path" in warning for warning in report.warnings)
    assert any("pip appears to come from a different prefix" in warning for warning in report.warnings)
    assert any("JAX import/backend check failed" in warning for warning in report.warnings)


def test_doctor_collect_report_accepts_conda_prefix_and_main_prints(monkeypatch, capsys) -> None:
    versions = {name: "26.0" for name in doctor._CORE_PACKAGES}

    monkeypatch.setattr(doctor, "_package_version", lambda name: versions[name])
    monkeypatch.setattr(doctor, "_user_site", lambda: "/conda/env/lib/python/site-packages")
    monkeypatch.setattr(doctor, "_pip_report", lambda: "pip 26 from /conda/env/lib/python/site-packages/pip (python 3.11)")
    monkeypatch.setattr(doctor, "_jax_info", lambda: ("cpu", ("TFRT_CPU_0",), None))
    monkeypatch.setattr(doctor.sys, "prefix", "/conda/env")
    monkeypatch.setattr(doctor.sys, "base_prefix", "/usr")
    monkeypatch.setattr(doctor.sys, "executable", "/conda/env/bin/python")
    monkeypatch.setattr(doctor.sys, "path", ["/conda/env/lib/python/site-packages"])
    monkeypatch.setenv("CONDA_PREFIX", "/conda/env")

    report = doctor.collect_report()
    assert report.in_virtualenv is True
    assert report.conda_prefix == "/conda/env"
    assert report.warnings == ()

    monkeypatch.setattr(doctor, "collect_report", lambda: report)
    assert doctor.main() == 0
    out = capsys.readouterr().out
    assert "Conda env:   /conda/env" in out
    assert "Status: no obvious installation problems detected." in out
