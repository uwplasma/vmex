from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


from vmec_jax.config import load_config
from vmec_jax.modes import ModeTable
from vmec_jax.static import build_static
from vmec_jax.kernels.jacobian import vmec_half_mesh_jacobian_from_state
from vmec_jax.kernels.realspace import vmec_realspace_synthesis
from vmec_jax.kernels.residue import vmec_wint_from_trig
from vmec_jax.kernels.tomnsp import vmec_trig_tables
from vmec_jax.wout import read_wout, state_from_wout
pytestmark = pytest.mark.full


_CASES = [
    (
        "circular_tokamak",
        "examples/data/input.circular_tokamak",
        "examples/data/wout_circular_tokamak_reference.nc",
    ),
]


def _hi_res_cfg(cfg, *, mpol: int, ntor: int):
    ntheta = max(int(cfg.ntheta), 4 * int(mpol) + 32)
    ntheta = 2 * (ntheta // 2)
    nzeta = max(int(cfg.nzeta), 4 * int(ntor) + 32)
    if nzeta <= 0:
        nzeta = 1
    return replace(cfg, ntheta=int(ntheta), nzeta=int(nzeta))


def _rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    num = float(np.sqrt(np.mean((a - b) ** 2)))
    den = float(np.sqrt(np.mean(b**2)))
    return num / den if den != 0.0 else float("inf")


@pytest.mark.parametrize("case_name,input_rel,wout_rel", _CASES)
def test_vmec_halfmesh_jacobian_matches_wout_gmnc(case_name: str, input_rel: str, wout_rel: str):
    pytest.importorskip("netCDF4")

    root = Path(__file__).resolve().parents[1]
    input_path = root / input_rel
    wout_path = root / wout_rel
    assert input_path.exists()
    assert wout_path.exists()

    cfg, _indata = load_config(str(input_path))
    wout = read_wout(wout_path)
    cfg_hi = _hi_res_cfg(cfg, mpol=wout.mpol, ntor=wout.ntor)
    static = build_static(cfg_hi)
    trig = static.trig_vmec
    if trig is None:
        trig = vmec_trig_tables(
            ntheta=int(cfg_hi.ntheta),
            nzeta=int(cfg_hi.nzeta),
            nfp=int(cfg_hi.nfp),
            mmax=int(cfg_hi.mpol) - 1,
            nmax=int(cfg_hi.ntor),
            lasym=bool(cfg_hi.lasym),
        )

    st = state_from_wout(wout)
    jh = vmec_half_mesh_jacobian_from_state(
        state=st,
        modes=static.modes,
        trig=trig,
        s=static.s,
        lconm1=bool(cfg_hi.lconm1),
        lthreed=bool(cfg_hi.ntor > 0),
    )
    sqrtg_calc = np.asarray(jh.sqrtg)

    # Reference sqrt(g) from wout Nyquist gmnc/gmns.
    modes_nyq = ModeTable(m=wout.xm_nyq, n=(wout.xn_nyq // wout.nfp))
    mmax_nyq = int(np.max(np.asarray(wout.xm_nyq)))
    nmax_nyq = int(np.max(np.abs(np.asarray(wout.xn_nyq)) // int(wout.nfp)))
    trig_nyq = vmec_trig_tables(
        ntheta=int(cfg_hi.ntheta),
        nzeta=int(cfg_hi.nzeta),
        nfp=int(cfg_hi.nfp),
        mmax=mmax_nyq,
        nmax=nmax_nyq,
        lasym=bool(cfg_hi.lasym),
    )
    sqrtg_ref = np.asarray(
        vmec_realspace_synthesis(
            coeff_cos=wout.gmnc,
            coeff_sin=wout.gmns,
            modes=modes_nyq,
            trig=trig_nyq,
            coeffs_internal=False,
        )
    )

    # Exclude axis.
    err = _rel_rms(sqrtg_calc[1:], sqrtg_ref[1:])
    # This parity kernel matches VMEC's half-mesh staggering and odd-m handling.
    # Remaining discrepancy is dominated by VMEC's endpoint-weighted theta grids
    # (ntheta2/ntheta3) and nscale/mscale normalization.
    assert err < 0.08

    # Volume derivative `vp` is an angular average of sqrt(g), and should match
    # much more tightly even if pointwise sqrt(g) differs.
    w_ang = np.asarray(vmec_wint_from_trig(trig, nzeta=sqrtg_calc.shape[2]))
    vp_calc = np.sum((int(wout.signgs) * sqrtg_calc) * w_ang[None, :, :], axis=(1, 2))
    vp_ref = np.asarray(wout.vp)
    vp_err = _rel_rms(vp_calc[1:], vp_ref[1:])
    assert vp_err < 1e-2
