from __future__ import annotations

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
        warnings=("packaging>=24.2 is required by setuptools>=77 license validation.",),
    )

    text = format_report(report)

    assert "Warnings:" in text
    assert "packaging>=24.2" in text
    assert "Recommended clean install:" in text
    assert "python -m pip install -U pip setuptools wheel packaging" in text

