from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import vmec_jax as vj
from vmec_jax.quasisymmetry import (
    quasisymmetry_ratio_residual_from_state,
    quasisymmetry_ratio_residual_from_wout,
)
from vmec_jax.wout import state_from_wout


pytestmark = [
    pytest.mark.simsopt,
    pytest.mark.skipif(
        os.environ.get("RUN_SIMSOPT_VALIDATION") != "1",
        reason="Set RUN_SIMSOPT_VALIDATION=1 to run optional SIMSOPT validation",
    ),
]


def _simsopt_qs_modules():
    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")
    simsopt_vmec = pytest.importorskip("simsopt.mhd.vmec")
    simsopt_diag = pytest.importorskip("simsopt.mhd.vmec_diagnostics")
    return simsopt_vmec, simsopt_diag


def _case_paths(input_name: str, wout_name: str) -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    input_path = root / "examples" / "data" / input_name
    wout_path = root / "examples" / "data" / wout_name
    if not input_path.exists():
        pytest.skip(f"Missing bundled fixture: {input_path}")
    if not wout_path.exists():
        pytest.skip(f"Missing bundled fixture: {wout_path}")
    return input_path, wout_path


def _qh_warm_start_paths() -> tuple[Path, Path]:
    return _case_paths("input.nfp4_QH_warm_start", "wout_nfp4_QH_warm_start.nc")


def _qs_family_cases():
    return (
        pytest.param(
            "qh_warm_start",
            "input.nfp4_QH_warm_start",
            "wout_nfp4_QH_warm_start.nc",
            1,
            -1,
            id="qh",
        ),
        pytest.param(
            "qa_landreman_paul_lowres",
            "input.LandremanPaul2021_QA_lowres",
            "wout_LandremanPaul2021_QA_lowres.nc",
            1,
            0,
            id="qa",
        ),
    )


def _simsopt_qs_reference(
    simsopt_vmec,
    simsopt_diag,
    wout_path: Path,
    *,
    surfaces,
    helicity_m,
    helicity_n,
    ntheta,
    nphi,
):
    vmec = simsopt_vmec.Vmec(str(wout_path), verbose=False)
    return simsopt_diag.QuasisymmetryRatioResidual(
        vmec,
        surfaces=surfaces,
        helicity_m=helicity_m,
        helicity_n=helicity_n,
        ntheta=ntheta,
        nphi=nphi,
    )


def test_qh_quasisymmetry_residual_matches_simsopt_wout_formula():
    """Compare the VMEC-only QS residual formula against SIMSOPT on a real wout."""

    simsopt_vmec, simsopt_diag = _simsopt_qs_modules()

    _input_path, wout_path = _qh_warm_start_paths()

    surfaces = np.linspace(0.0, 1.0, 3)
    helicity_m = 1
    helicity_n = -1
    ntheta = 15
    nphi = 16

    wout = vj.load_wout(wout_path)
    ours = quasisymmetry_ratio_residual_from_wout(
        wout,
        surfaces=surfaces,
        helicity_m=helicity_m,
        helicity_n=helicity_n,
        ntheta=ntheta,
        nphi=nphi,
    )

    ref = _simsopt_qs_reference(
        simsopt_vmec,
        simsopt_diag,
        wout_path,
        surfaces=surfaces,
        helicity_m=helicity_m,
        helicity_n=helicity_n,
        ntheta=ntheta,
        nphi=nphi,
    )

    np.testing.assert_allclose(
        np.asarray(ours["residuals1d"]),
        ref.residuals(),
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(float(ours["total"]), ref.total(), rtol=1.0e-12, atol=1.0e-13)


def test_qh_quasisymmetry_state_diagnostic_matches_simsopt_converged_wout():
    """Compare state-derived QS diagnostics against SIMSOPT on a converged fixture."""

    simsopt_vmec, simsopt_diag = _simsopt_qs_modules()
    input_path, wout_path = _qh_warm_start_paths()

    surfaces = np.linspace(0.0, 1.0, 3)
    helicity_m = 1
    helicity_n = -1
    ntheta = 15
    nphi = 16

    cfg, indata = vj.load_config(str(input_path))
    static = vj.build_static(cfg)
    wout = vj.load_wout(wout_path)
    state = state_from_wout(wout)
    ours = quasisymmetry_ratio_residual_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=int(wout.signgs),
        surfaces=surfaces,
        helicity_m=helicity_m,
        helicity_n=helicity_n,
        ntheta=ntheta,
        nphi=nphi,
    )

    ref = _simsopt_qs_reference(
        simsopt_vmec,
        simsopt_diag,
        wout_path,
        surfaces=surfaces,
        helicity_m=helicity_m,
        helicity_n=helicity_n,
        ntheta=ntheta,
        nphi=nphi,
    )

    np.testing.assert_allclose(
        np.asarray(ours["residuals1d"]),
        ref.residuals(),
        rtol=5.0e-5,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(float(ours["total"]), ref.total(), rtol=2.0e-5, atol=1.0e-7)


@pytest.mark.parametrize(
    "case,input_name,wout_name,helicity_m,helicity_n",
    _qs_family_cases(),
)
def test_quasisymmetry_residual_family_matches_simsopt_wout_formula(
    case: str,
    input_name: str,
    wout_name: str,
    helicity_m: int,
    helicity_n: int,
) -> None:
    """Optional family gate for SIMSOPT's public QS residual convention."""

    del case
    simsopt_vmec, simsopt_diag = _simsopt_qs_modules()

    _input_path, wout_path = _case_paths(input_name, wout_name)
    surfaces = np.linspace(0.0, 1.0, 3)
    ntheta = 15
    nphi = 16

    wout = vj.load_wout(wout_path)
    ours = quasisymmetry_ratio_residual_from_wout(
        wout,
        surfaces=surfaces,
        helicity_m=helicity_m,
        helicity_n=helicity_n,
        ntheta=ntheta,
        nphi=nphi,
    )
    ref = _simsopt_qs_reference(
        simsopt_vmec,
        simsopt_diag,
        wout_path,
        surfaces=surfaces,
        helicity_m=helicity_m,
        helicity_n=helicity_n,
        ntheta=ntheta,
        nphi=nphi,
    )

    np.testing.assert_allclose(np.asarray(ours["residuals1d"]), ref.residuals(), rtol=1.0e-11, atol=1.0e-12)
    np.testing.assert_allclose(float(ours["total"]), ref.total(), rtol=1.0e-12, atol=1.0e-13)


@pytest.mark.parametrize(
    "case,input_name,wout_name,helicity_m,helicity_n",
    _qs_family_cases(),
)
def test_quasisymmetry_state_diagnostic_family_matches_simsopt_converged_wout(
    case: str,
    input_name: str,
    wout_name: str,
    helicity_m: int,
    helicity_n: int,
) -> None:
    """Optional gate for state-derived QS diagnostics on QA and QH fixtures."""

    del case
    simsopt_vmec, simsopt_diag = _simsopt_qs_modules()

    input_path, wout_path = _case_paths(input_name, wout_name)
    surfaces = np.linspace(0.0, 1.0, 3)
    ntheta = 15
    nphi = 16

    cfg, indata = vj.load_config(str(input_path))
    static = vj.build_static(cfg)
    wout = vj.load_wout(wout_path)
    state = state_from_wout(wout)
    ours = quasisymmetry_ratio_residual_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=int(wout.signgs),
        surfaces=surfaces,
        helicity_m=helicity_m,
        helicity_n=helicity_n,
        ntheta=ntheta,
        nphi=nphi,
    )
    ref = _simsopt_qs_reference(
        simsopt_vmec,
        simsopt_diag,
        wout_path,
        surfaces=surfaces,
        helicity_m=helicity_m,
        helicity_n=helicity_n,
        ntheta=ntheta,
        nphi=nphi,
    )

    np.testing.assert_allclose(np.asarray(ours["residuals1d"]), ref.residuals(), rtol=7.5e-5, atol=2.5e-6)
    np.testing.assert_allclose(float(ours["total"]), ref.total(), rtol=3.0e-5, atol=2.0e-7)
