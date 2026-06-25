from __future__ import annotations

from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.driver import run_free_boundary, write_wout_from_fixed_boundary_run
from vmec_jax.solvers.free_boundary.validation import (
    free_boundary_response_metrics,
    wout_beta_percent,
    wout_fsq_total,
    wout_mean_iota,
)
from vmec_jax.namelist import read_indata, write_indata
from vmec_jax.wout import read_wout


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_free_boundary_response_metrics_are_zero_for_same_wout() -> None:
    wout = REPO_ROOT / "examples" / "data" / "wout_circular_tokamak.nc"
    metrics = free_boundary_response_metrics(wout, wout, ntheta=48, nphi=8)

    assert metrics.beta_delta_percent == pytest.approx(0.0)
    assert metrics.aspect_delta == pytest.approx(0.0)
    assert metrics.mean_iota_delta == pytest.approx(0.0)
    assert metrics.lcfs_rms_displacement == pytest.approx(0.0)
    assert metrics.lcfs_max_displacement == pytest.approx(0.0)
    assert metrics.lcfs_b_rel_rms_delta == pytest.approx(0.0)


def test_free_boundary_response_metrics_detect_geometry_and_field_changes() -> None:
    """The WOUT-native metric must not hide real geometry/profile changes."""

    reference = REPO_ROOT / "examples" / "data" / "wout_circular_tokamak.nc"
    shaped = REPO_ROOT / "examples" / "data" / "wout_shaped_tokamak_pressure.nc"
    metrics = free_boundary_response_metrics(reference, shaped, ntheta=48, nphi=8)

    assert np.isfinite(list(metrics.to_dict().values())).all()
    assert metrics.lcfs_max_displacement > 0.05
    assert abs(metrics.mean_iota_delta) > 1.0e-3
    assert metrics.lcfs_b_rel_rms_delta > 1.0e-3


def test_free_boundary_response_scalar_helpers_and_nfp_guard() -> None:
    """Validation helpers should be backend-neutral for WOUT-like objects."""

    reference = SimpleNamespace(nfp=2, betatotal=0.012, iotaf=np.asarray([0.0, 0.2, 0.4]), fsqr=1.0, fsqz=2.0, fsql=3.0)
    fallback = SimpleNamespace(nfp=2, beta_total=0.034, iotas=np.asarray([0.0, -0.5]), fsqr=0.5, fsqz=0.25, fsql=0.125)
    no_profiles = SimpleNamespace(nfp=2)
    wrong_period = SimpleNamespace(nfp=3)

    assert wout_beta_percent(reference) == pytest.approx(1.2)
    assert wout_beta_percent(fallback) == pytest.approx(3.4)
    assert np.isnan(wout_beta_percent(no_profiles))
    assert wout_mean_iota(reference) == pytest.approx(0.3)
    assert wout_mean_iota(fallback) == pytest.approx(-0.5)
    assert np.isnan(wout_mean_iota(no_profiles))
    assert wout_fsq_total(reference) == pytest.approx(6.0)
    assert wout_fsq_total(fallback) == pytest.approx(0.875)
    assert np.isnan(wout_fsq_total(no_profiles))

    with pytest.raises(ValueError, match="nfp mismatch"):
        free_boundary_response_metrics(reference, wrong_period, ntheta=8, nphi=4)


def _write_cth_pressure_case(tmp_path: Path, pressure_scale: float) -> Path:
    input_src = REPO_ROOT / "examples" / "data" / "input.cth_like_free_bdy_lasym_small"
    mgrid_src = REPO_ROOT / "examples" / "data" / "mgrid_cth_like_lasym_small.nc"
    if not input_src.exists() or not mgrid_src.exists():
        pytest.skip("Bundled CTH-like free-boundary fixture is unavailable")
    shutil.copy2(mgrid_src, tmp_path / mgrid_src.name)

    indata = read_indata(input_src)
    indata.scalars.update(
        {
            "PRES_SCALE": float(pressure_scale),
            # Bounded CI-friendly schedule: this is a response/coupling smoke,
            # not the high-resolution DIII-D publication gate.
            "NS_ARRAY": [7],
            "NITER_ARRAY": [80],
            "FTOL_ARRAY": [1.0e-6],
            "NITER": 80,
            "FTOL": 1.0e-6,
            "NZETA": 10,
        }
    )
    path = tmp_path / f"input.cth_pressure_{pressure_scale:g}"
    write_indata(path, indata)
    return path


@pytest.mark.full
def test_free_boundary_pressure_scale_changes_bundled_lcfs_and_field(tmp_path: Path) -> None:
    """CI physics gate: finite pressure must activate a measurable free-boundary response."""

    wouts = []
    for pressure_scale in (0.0, 432.29080924603676):
        input_path = _write_cth_pressure_case(tmp_path, pressure_scale)
        run = run_free_boundary(
            input_path,
            verbose=False,
            solver="vmec2000_iter",
            solver_mode="parity",
            multigrid_use_input_niter=True,
        )
        wout_path = tmp_path / f"wout_cth_pressure_{pressure_scale:g}.nc"
        write_wout_from_fixed_boundary_run(wout_path, run, include_fsq=True)
        wouts.append(wout_path)

    vacuum = read_wout(wouts[0])
    finite_beta = read_wout(wouts[1])
    metrics = free_boundary_response_metrics(vacuum, finite_beta, ntheta=48, nphi=12)

    assert metrics.candidate_beta_percent > 0.05
    assert metrics.lcfs_max_displacement > 1.0e-4
    assert metrics.lcfs_b_rel_rms_delta > 1.0e-3
