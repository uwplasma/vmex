from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_example_module():
    script = ROOT / "examples" / "diagnostics" / "plot_glasser_qa_finite_beta.py"
    spec = importlib.util.spec_from_file_location("plot_glasser_qa_finite_beta_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_glasser_finite_beta_example_profiles_and_plot(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    mod = _load_example_module()
    wout = SimpleNamespace(
        ns=5,
        DMerc=np.asarray([0.0, 0.12, 0.18, 0.04, 0.0]),
        D_R=np.asarray([0.0, -0.08, 0.03, -0.01, 0.0]),
        glasser_shear_valid=np.asarray([False, True, True, True, False]),
    )

    s, dmerc, d_r, valid = mod.glasser_profiles(wout)
    np.testing.assert_allclose(s, np.linspace(0.0, 1.0, 5))
    np.testing.assert_allclose(dmerc, wout.DMerc)
    np.testing.assert_allclose(d_r, wout.D_R)
    np.testing.assert_array_equal(valid, wout.glasser_shear_valid)

    figure = mod.plot_glasser_profiles(wout, figure_path=tmp_path / "glasser.png")
    assert figure.exists()
    assert figure.stat().st_size > 0


def test_glasser_finite_beta_example_can_reuse_existing_wout(monkeypatch, tmp_path: Path) -> None:
    mod = _load_example_module()
    wout_path = tmp_path / "wout.nc"
    wout_path.write_text("placeholder")
    expected = object()
    monkeypatch.setattr(mod, "RUN_VMEC", False)
    monkeypatch.setattr(mod, "WOUT_FILE", wout_path)
    monkeypatch.setattr(mod, "read_wout", lambda path: expected if path == wout_path else None)
    monkeypatch.setattr(mod.vj, "run_fixed_boundary", lambda *args, **kwargs: pytest.fail("should reuse wout"))

    assert mod.load_or_run_wout() is expected


@pytest.mark.py311_slow_coverage
def test_finite_beta_qa_glasser_profile_physics_gate(tmp_path: Path, monkeypatch) -> None:
    """Run the finite-beta QA deck and lock down the promoted D_R profile."""

    jax = pytest.importorskip("jax")
    pytest.importorskip("netCDF4")
    jax.config.update("jax_disable_jit", False)
    mod = _load_example_module()

    monkeypatch.setattr(mod, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(mod, "WOUT_FILE", tmp_path / "wout_nfp2_QA_finite_beta.nc")
    monkeypatch.setattr(mod, "FIGURE_FILE", tmp_path / "qa_finite_beta_dmerc_dr.png")
    monkeypatch.setattr(mod, "RUN_VMEC", True)

    wout = mod.load_or_run_wout()
    figure = mod.plot_glasser_profiles(wout, figure_path=mod.FIGURE_FILE)
    assert figure.exists()

    s, dmerc, d_r, valid = mod.glasser_profiles(wout)
    interior = slice(1, -1)
    assert s.shape == (45,)
    assert valid.shape == (45,)
    assert int(np.count_nonzero(valid)) == 43
    assert float(wout.betatotal) > 2.0e-2
    assert np.all(np.isfinite(dmerc))
    assert np.all(np.isfinite(d_r))

    np.testing.assert_allclose(float(np.nanmin(dmerc[interior])), 4.796676303078426e-06, rtol=2.0e-2, atol=5.0e-8)
    np.testing.assert_allclose(float(np.nanmax(d_r[interior])), 1.970029492766792e-04, rtol=2.0e-2, atol=5.0e-7)
    assert float(np.nanmin(d_r[interior])) < -1.0e-5
    assert float(np.nanmax(d_r[interior])) > 1.0e-4
