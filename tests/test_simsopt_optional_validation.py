from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import vmec_jax as vj
from vmec_jax.quasisymmetry import quasisymmetry_ratio_residual_from_wout


pytestmark = pytest.mark.simsopt


if os.environ.get("RUN_SIMSOPT_VALIDATION") != "1":
    pytest.skip("Set RUN_SIMSOPT_VALIDATION=1 to run optional SIMSOPT validation", allow_module_level=True)


def test_qh_quasisymmetry_residual_matches_simsopt_wout_formula():
    """Compare the VMEC-only QS residual formula against SIMSOPT on a real wout."""

    pytest.importorskip("jax")
    simsopt_vmec = pytest.importorskip("simsopt.mhd.vmec")
    simsopt_diag = pytest.importorskip("simsopt.mhd.vmec_diagnostics")

    root = Path(__file__).resolve().parents[1]
    wout_path = root / "examples" / "data" / "wout_nfp4_QH_warm_start.nc"
    if not wout_path.exists():
        pytest.skip(f"Missing bundled fixture: {wout_path}")

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

    vmec = simsopt_vmec.Vmec(str(wout_path), verbose=False)
    ref = simsopt_diag.QuasisymmetryRatioResidual(
        vmec,
        surfaces=surfaces,
        helicity_m=helicity_m,
        helicity_n=helicity_n,
        ntheta=ntheta,
        nphi=nphi,
    )

    np.testing.assert_allclose(np.asarray(ours["residuals1d"]), ref.residuals(), rtol=1.0e-11, atol=1.0e-12)
    np.testing.assert_allclose(float(ours["total"]), ref.total(), rtol=1.0e-12, atol=1.0e-13)
