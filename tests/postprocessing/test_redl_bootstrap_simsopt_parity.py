from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import vmec_jax as vj
from vmec_jax.wout import state_from_wout


pytestmark = [
    pytest.mark.simsopt,
    pytest.mark.skipif(
        os.environ.get("RUN_SIMSOPT_VALIDATION") != "1",
        reason="Set RUN_SIMSOPT_VALIDATION=1 to run optional SIMSOPT validation",
    ),
]


def test_redl_bootstrap_mismatch_matches_simsopt_redlgeomvmec_on_shaped_tokamak():
    """Compare vmec_jax Redl residuals against SIMSOPT on a real wout fixture."""

    pytest.importorskip("netCDF4")
    simsopt_bootstrap = pytest.importorskip("simsopt.mhd.bootstrap")
    simsopt_profiles = pytest.importorskip("simsopt.mhd.profiles")
    simsopt_vmec = pytest.importorskip("simsopt.mhd.vmec")

    root = Path(__file__).resolve().parents[2]
    input_path = root / "examples" / "data" / "input.shaped_tokamak_pressure"
    wout_path = root / "examples" / "data" / "wout_shaped_tokamak_pressure.nc"
    cfg, indata = vj.load_config(str(input_path))
    static = vj.build_static(cfg)
    wout = vj.load_wout(wout_path)
    state = state_from_wout(wout)
    signgs = int(getattr(wout, "signgs", 1))

    surfaces = tuple(np.linspace(0.2, 0.8, 4))
    ne_coeffs = [3.0e20, 0.0, -2.0e20]
    te_coeffs = [8.0e3, -5.0e3]
    ours = vj.redl_bootstrap_mismatch_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=signgs,
        helicity_n=0,
        ne_coeffs=ne_coeffs,
        Te_coeffs=te_coeffs,
        surfaces=surfaces,
    )

    vmec = simsopt_vmec.Vmec(str(wout_path), verbose=False)
    geom = simsopt_bootstrap.RedlGeomVmec(
        vmec,
        np.asarray(surfaces),
        ntheta=max(int(static.cfg.ntheta), 8),
        nphi=max(int(static.cfg.nzeta), 5),
    )
    ne = simsopt_profiles.ProfilePolynomial(ne_coeffs)
    te = simsopt_profiles.ProfilePolynomial(te_coeffs)
    ref_redl, _details = simsopt_bootstrap.j_dot_B_Redl(ne, te, te, 1.0, helicity_n=0, geom=geom)
    ref_obj = simsopt_bootstrap.VmecRedlBootstrapMismatch(geom, ne, te, te, 1.0, helicity_n=0)
    ref_residuals = ref_obj.residuals()

    # With the same SIMSOPT Redl geometry, the residual normalization and
    # state-derived VMEC jdotB path should agree tightly.
    ours_with_ref_redl = vj.redl_bootstrap_mismatch_from_profiles(
        jdotB_vmec=ours["jdotB_vmec"],
        jdotB_redl=ref_redl,
    )
    np.testing.assert_allclose(np.asarray(ours_with_ref_redl), ref_residuals, rtol=2.0e-6, atol=2.0e-7)

    # The public objective uses a differentiable state-geometry approximation
    # instead of SIMSOPT's spline-refined RedlGeomVmec post-processing. Keep a
    # physics regression envelope here, not a bitwise gate.
    np.testing.assert_allclose(np.asarray(ours["residuals1d"]), ref_residuals, rtol=2.0e-2, atol=5.0e-3)
