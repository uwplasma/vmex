from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vmec_jax.config import load_config
from vmec_jax.static import build_static
from vmec_jax.kernels.bcovar import vmec_bcovar_half_mesh_from_wout
from vmec_jax.kernels.tomnsp import vmec_angle_grid
from vmec_jax.wout import _compute_eqfor_beta, _vmec_wint_from_trig, read_wout, state_from_wout


BETA_CASES = (
    (
        "finite_beta_axisym",
        "examples/data/input.shaped_tokamak_pressure",
        "examples/data/wout_shaped_tokamak_pressure.nc",
    ),
    (
        "finite_beta_3d",
        "examples/data/input.li383_low_res",
        "examples/data/wout_li383_low_res.nc",
    ),
    (
        "finite_beta_lasym_3d",
        "examples/data/single_grid/input.basic_non_stellsym_pressure",
        "examples/data/single_grid/wout_basic_non_stellsym_pressure_reference.nc",
    ),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(("case_name", "input_rel", "wout_rel"), BETA_CASES, ids=[case[0] for case in BETA_CASES])
def test_bundled_finite_beta_wout_scalars_match_eqfor_decomposition(
    case_name: str,
    input_rel: str,
    wout_rel: str,
) -> None:
    """VMEC2000 beta scalars should close against the eqfor field-energy split."""
    pytest.importorskip("jax")
    pytest.importorskip("netCDF4")

    repo_root = _repo_root()
    input_path = repo_root / input_rel
    wout_path = repo_root / wout_rel
    if not input_path.exists() or not wout_path.exists():
        pytest.skip(f"Missing bundled finite-beta fixture: {case_name}")

    cfg, _indata = load_config(str(input_path))
    wout = read_wout(wout_path)
    cfg = replace(
        cfg,
        ns=int(wout.ns),
        mpol=int(wout.mpol),
        ntor=int(wout.ntor),
        nfp=int(wout.nfp),
        lasym=bool(wout.lasym),
        lthreed=bool(int(wout.ntor) > 0),
    )
    grid = vmec_angle_grid(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(wout.nfp),
        lasym=bool(wout.lasym),
    )
    static = build_static(cfg, grid=grid)

    bc = vmec_bcovar_half_mesh_from_wout(
        state=state_from_wout(wout),
        static=static,
        wout=wout,
        pres=wout.pres,
    )
    betapol, betator, betatotal, betaxis = _compute_eqfor_beta(
        pres=np.asarray(wout.pres, dtype=float),
        vp=np.asarray(wout.vp, dtype=float),
        bsq=np.asarray(bc.bsq, dtype=float),
        r12=np.asarray(bc.jac.r12, dtype=float),
        bsupv=np.asarray(bc.bsupv, dtype=float),
        sqrtg=np.asarray(bc.jac.sqrtg, dtype=float),
        wint=np.asarray(_vmec_wint_from_trig(static.trig_vmec), dtype=float),
        signgs=int(wout.signgs),
    )

    assert float(wout.wp) > 0.0, case_name
    assert float(wout.wb) > 0.0, case_name
    np.testing.assert_allclose(betapol, float(wout.betapol), rtol=2.0e-12, atol=1.0e-15, err_msg=case_name)
    np.testing.assert_allclose(betator, float(wout.betator), rtol=2.0e-12, atol=1.0e-15, err_msg=case_name)
    np.testing.assert_allclose(betatotal, float(wout.betatotal), rtol=2.0e-12, atol=1.0e-15, err_msg=case_name)
    np.testing.assert_allclose(betaxis, float(wout.betaxis), rtol=2.0e-12, atol=1.0e-15, err_msg=case_name)
