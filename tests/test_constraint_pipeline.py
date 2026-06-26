from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest


from vmec_jax.config import load_config
from vmec_jax.static import build_static
from vmec_jax.kernels.constraints import tcon_from_bcovar_precondn_diag
from vmec_jax.kernels.forces import vmec_forces_rz_from_wout_reference_fields
from vmec_jax.kernels.tomnsp import vmec_angle_grid, vmec_trig_tables
from vmec_jax.wout import read_wout, state_from_wout
pytestmark = pytest.mark.full


def _load_case(input_rel: str, wout_rel: str):
    root = Path(__file__).resolve().parents[1]
    input_path = root / input_rel
    wout_path = root / wout_rel
    assert input_path.exists()
    assert wout_path.exists()
    cfg, _ = load_config(str(input_path))
    wout = read_wout(wout_path)
    # These tests only validate the constraint pipeline wiring (zero/nonzero),
    # not spectral parity. Use a very small grid to keep runtime tiny.
    ntheta = min(int(cfg.ntheta), 12)
    ntheta = 2 * (ntheta // 2)
    nzeta = 1
    cfg = replace(cfg, ntheta=int(ntheta), nzeta=int(nzeta))
    grid = vmec_angle_grid(ntheta=int(cfg.ntheta), nzeta=int(cfg.nzeta), nfp=int(wout.nfp), lasym=bool(wout.lasym))
    static = build_static(cfg, grid=grid)
    st = state_from_wout(wout)
    return static, st, wout


def test_constraint_pipeline_zero_tcon0_yields_zero_gcon():
    pytest.importorskip("netCDF4")

    static, st, wout = _load_case(
        "examples/data/input.circular_tokamak",
        "examples/data/wout_circular_tokamak_reference.nc",
    )

    k = vmec_forces_rz_from_wout_reference_fields(
        state=st,
        static=static,
        wout=wout,
        constraint_tcon0=0.0,
    )

    assert np.allclose(np.asarray(k.gcon), 0.0)
    assert np.allclose(np.asarray(k.arcon_e), 0.0)
    assert np.allclose(np.asarray(k.azcon_e), 0.0)


def test_constraint_pipeline_nonzero_tcon0_produces_gcon():
    pytest.importorskip("netCDF4")

    static, st, wout = _load_case(
        "examples/data/input.circular_tokamak",
        "examples/data/wout_circular_tokamak_reference.nc",
    )

    k = vmec_forces_rz_from_wout_reference_fields(
        state=st,
        static=static,
        wout=wout,
        constraint_tcon0=0.3,
    )

    gcon_norm = float(np.linalg.norm(np.asarray(k.gcon)))
    assert gcon_norm > 0.0


def test_tcon_axisym_matches_precondn_diag_numpy():
    pytest.importorskip("netCDF4")
    pytest.importorskip("jax")

    static, st, wout = _load_case(
        "examples/data/input.circular_tokamak",
        "examples/data/wout_circular_tokamak_reference.nc",
    )

    k = vmec_forces_rz_from_wout_reference_fields(
        state=st,
        static=static,
        wout=wout,
        constraint_tcon0=0.7,
    )

    ns = int(static.s.shape[0])
    s = np.asarray(static.s, dtype=float)
    trig = vmec_trig_tables(
        ntheta=int(static.cfg.ntheta),
        nzeta=int(static.cfg.nzeta),
        nfp=int(wout.nfp),
        mmax=int(wout.mpol) - 1,
        nmax=int(wout.ntor),
        lasym=bool(wout.lasym),
        dtype=np.float64,
    )

    psqrts = np.sqrt(np.maximum(s, 0.0))[:, None, None]
    ru0 = np.asarray(k.pru_even) + psqrts * np.asarray(k.pru_odd)
    zu0 = np.asarray(k.pzu_even) + psqrts * np.asarray(k.pzu_odd)

    tcon_jax = tcon_from_bcovar_precondn_diag(
        tcon0=0.7,
        trig=trig,
        s=static.s,
        signgs=int(getattr(wout, "signgs", 1)),
        lasym=bool(wout.lasym),
        bsq=np.asarray(k.bc.bsq),
        r12=np.asarray(k.bc.jac.r12),
        sqrtg=np.asarray(k.bc.jac.sqrtg),
        ru12=np.asarray(k.bc.jac.ru12),
        zu12=np.asarray(k.bc.jac.zu12),
        ru0=np.asarray(ru0),
        zu0=np.asarray(zu0),
    )

    w_theta = np.asarray(trig.cosmui3[:, 0]) / float(np.asarray(trig.mscale[0]))
    wint = w_theta[:, None] * np.ones((int(trig.cosnv.shape[0]),), dtype=float)[None, :]
    wint3 = wint[None, :, :]
    arnorm = np.sum((np.asarray(ru0) * np.asarray(ru0)) * wint3, axis=(1, 2))
    aznorm = np.sum((np.asarray(zu0) * np.asarray(zu0)) * wint3, axis=(1, 2))
    arnorm = np.where(arnorm != 0.0, arnorm, 1.0)
    aznorm = np.where(aznorm != 0.0, aznorm, 1.0)

    hs = float(s[1] - s[0])
    ohs = 0.0 if hs == 0.0 else 1.0 / hs
    pfactor = -4.0 * float(trig.r0scale) ** 2

    r12 = np.asarray(k.bc.jac.r12, dtype=float)
    sqrtg = np.asarray(k.bc.jac.sqrtg, dtype=float)
    bsq = np.asarray(k.bc.bsq, dtype=float)
    ru12 = np.asarray(k.bc.jac.ru12, dtype=float)
    zu12 = np.asarray(k.bc.jac.zu12, dtype=float)
    gs = np.where(sqrtg != 0.0, sqrtg, 1.0)
    ptau = (pfactor * (r12 * r12) * bsq * wint3) / gs
    ax_r = np.sum(ptau * ((zu12 * ohs) ** 2), axis=(1, 2))
    ax_z = np.sum(ptau * ((ru12 * ohs) ** 2), axis=(1, 2))
    if ax_r.size:
        ax_r[0] = 0.0
        ax_z[0] = 0.0
    ard1 = ax_r + np.concatenate([ax_r[1:], np.zeros((1,), dtype=float)], axis=0)
    azd1 = ax_z + np.concatenate([ax_z[1:], np.zeros((1,), dtype=float)], axis=0)

    tcon0 = min(abs(float(0.7)), 1.0)
    ns_f = float(ns)
    tcon_mul = tcon0 * (1.0 + ns_f * (1.0 / 60.0 + ns_f / (200.0 * 120.0)))
    tcon_mul = tcon_mul / ((4.0 * (float(trig.r0scale) ** 2)) ** 2)

    tcon_np = np.zeros((ns,), dtype=float)
    tcon_np[0] = tcon0
    if ns >= 3:
        js = np.arange(ns)
        mask = (js >= 1) & (js <= (ns - 2))
        ratio_r = np.abs(ard1) / arnorm
        ratio_z = np.abs(azd1) / aznorm
        core = np.minimum(ratio_r, ratio_z) * (tcon_mul * (32.0 * (s[1] - s[0])) ** 2)
        tcon_np = np.where(mask, core, tcon_np)
        tcon_np[-1] = 0.5 * tcon_np[-2]

    np.testing.assert_allclose(np.asarray(tcon_jax), tcon_np, rtol=1e-10, atol=1e-12)
