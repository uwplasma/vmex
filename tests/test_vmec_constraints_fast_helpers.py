from __future__ import annotations

from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.kernels.constraints import (
    alias_gcon,
    faccon_from_signgs,
    precondn_diag_axd1_from_bcovar,
    tcon_from_cached_precondn_diag,
    tcon_from_bcovar_precondn_diag,
    tcon_from_precondn_axisym,
    tcon_from_tcon0_heuristic,
)
from vmec_jax.kernels.tomnsp import vmec_trig_tables


Cfg = namedtuple("Cfg", "mpol ntor ntheta nzeta nfp lasym lthreed")


def test_faccon_from_signgs_matches_fixaray_indexing_and_guards():
    fac = np.asarray(faccon_from_signgs(mpol=6, signgs=-1))

    expected = np.zeros((6,), dtype=float)
    for m in range(1, 5):
        expected[m] = 0.25 / (((m + 1) * m) ** 2)
    np.testing.assert_allclose(fac, expected, rtol=0.0, atol=0.0)

    assert np.asarray(faccon_from_signgs(mpol=1, signgs=1)).tolist() == [0.0]
    with pytest.raises(ValueError, match="mpol must be positive"):
        faccon_from_signgs(mpol=0, signgs=1)


def test_tcon_heuristic_clamps_axis_and_edge_like_vmec_profile():
    trig = vmec_trig_tables(ntheta=8, nzeta=3, nfp=1, mmax=4, nmax=1, lasym=False)

    empty = np.asarray(tcon_from_tcon0_heuristic(tcon0=3.0, s=np.array([0.0]), trig=trig, lasym=False))
    np.testing.assert_allclose(empty, np.zeros((1,)))

    s = np.linspace(0.0, 1.0, 5)
    tcon = np.asarray(tcon_from_tcon0_heuristic(tcon0=3.0, s=s, trig=trig, lasym=True))

    hs = s[1] - s[0]
    tcon0 = 1.0
    ns = float(s.size)
    tcon_mul = tcon0 * (1.0 + ns * (1.0 / 60.0 + ns / (200.0 * 120.0)))
    tcon_mul = tcon_mul / ((4.0 * (float(trig.r0scale) ** 2)) ** 2)
    core = tcon_mul * (32.0 * hs) ** 2

    np.testing.assert_allclose(tcon[0], 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(tcon[1:-1], core, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(tcon[-1], 0.5 * core, rtol=1e-14, atol=1e-14)


def test_precondn_diag_short_mesh_returns_zero_tcon_and_diagonals():
    trig = vmec_trig_tables(ntheta=8, nzeta=2, nfp=1, mmax=3, nmax=1, lasym=False)
    shape = (1, int(trig.ntheta3), 2)
    ones = np.ones(shape)
    s = np.array([0.0])

    ard1, azd1 = precondn_diag_axd1_from_bcovar(
        trig=trig,
        s=s,
        bsq=ones,
        r12=ones,
        sqrtg=ones,
        ru12=ones,
        zu12=ones,
    )
    tcon = tcon_from_bcovar_precondn_diag(
        tcon0=0.5,
        trig=trig,
        s=s,
        signgs=1,
        lasym=False,
        bsq=ones,
        r12=ones,
        sqrtg=ones,
        ru12=ones,
        zu12=ones,
        ru0=ones,
        zu0=ones,
    )

    np.testing.assert_allclose(np.asarray(ard1), np.zeros((1,)))
    np.testing.assert_allclose(np.asarray(azd1), np.zeros((1,)))
    np.testing.assert_allclose(np.asarray(tcon), np.zeros((1,)))


def test_precondn_diag_falls_back_to_dnorm3_when_weight_shape_differs():
    trig_base = vmec_trig_tables(ntheta=8, nzeta=4, nfp=1, mmax=3, nmax=1, lasym=False)
    trig = SimpleNamespace(
        cosmu=trig_base.cosmu,
        cosmui3=trig_base.cosmui3,
        cosnv=trig_base.cosnv,
        mscale=trig_base.mscale,
        r0scale=trig_base.r0scale,
        dnorm3=np.full((2, 1), 0.25),
    )
    s = np.linspace(0.0, 1.0, 3)
    shape = (3, 2, 1)
    bsq = np.full(shape, 2.0)
    r12 = np.full(shape, 1.5)
    sqrtg = np.full(shape, 0.75)
    ru12 = np.full(shape, 0.4)
    zu12 = np.full(shape, 0.6)

    ard1, azd1 = precondn_diag_axd1_from_bcovar(
        trig=trig,
        s=s,
        bsq=bsq,
        r12=r12,
        sqrtg=sqrtg,
        ru12=ru12,
        zu12=zu12,
    )

    hs = s[1] - s[0]
    pfactor = -4.0 * float(trig.r0scale) ** 2
    ptau = (pfactor * r12**2 * bsq * np.asarray(trig.dnorm3)[None, :, :]) / sqrtg
    ax_r = np.sum(ptau * (zu12 / hs) ** 2, axis=(1, 2))
    ax_z = np.sum(ptau * (ru12 / hs) ** 2, axis=(1, 2))
    ax_r[0] = 0.0
    ax_z[0] = 0.0
    expected_ard = ax_r + np.concatenate([ax_r[1:], np.zeros((1,))])
    expected_azd = ax_z + np.concatenate([ax_z[1:], np.zeros((1,))])

    np.testing.assert_allclose(np.asarray(ard1), expected_ard, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.asarray(azd1), expected_azd, rtol=1e-13, atol=1e-13)


def test_alias_gcon_rejects_inconsistent_grid_shapes():
    trig = vmec_trig_tables(ntheta=8, nzeta=4, nfp=1, mmax=3, nmax=1, lasym=False)
    ztemp = np.zeros((2, int(trig.ntheta3), 4))
    tcon = np.ones((2,))

    with pytest.raises(ValueError, match="theta size"):
        alias_gcon(
            ztemp=ztemp[:, :-1, :],
            trig=trig,
            ntor=1,
            mpol=4,
            signgs=1,
            tcon=tcon,
            lasym=False,
        )

    trig_bad_lasym = SimpleNamespace(
        **{
            name: getattr(trig, name)
            for name in (
                "cosmu",
                "sinmu",
                "cosmui",
                "sinmui",
                "cosnv",
                "sinnv",
                "ntheta1",
                "ntheta2",
                "ntheta3",
            )
        }
    )
    trig_bad_lasym.ntheta1 = int(trig.ntheta3) + 1
    with pytest.raises(ValueError, match="lasym=True requires"):
        alias_gcon(
            ztemp=ztemp,
            trig=trig_bad_lasym,
            ntor=1,
            mpol=4,
            signgs=1,
            tcon=tcon,
            lasym=True,
        )

    with pytest.raises(ValueError, match="nzeta"):
        alias_gcon(
            ztemp=ztemp[:, :, :-1],
            trig=trig,
            ntor=1,
            mpol=4,
            signgs=1,
            tcon=tcon,
            lasym=False,
        )


def test_tcon_cached_precondn_diag_clamps_uses_norm_fallbacks_and_short_mesh():
    trig_base = vmec_trig_tables(ntheta=8, nzeta=4, nfp=1, mmax=3, nmax=1, lasym=False)
    trig = SimpleNamespace(
        cosmu=trig_base.cosmu,
        cosmui3=trig_base.cosmui3,
        cosnv=trig_base.cosnv,
        mscale=trig_base.mscale,
        r0scale=trig_base.r0scale,
        dnorm3=np.array([[0.25], [0.75]]),
    )

    short = tcon_from_cached_precondn_diag(
        tcon0=2.0,
        trig=trig,
        s=np.array([0.0]),
        lasym=False,
        ard1=np.array([0.0]),
        azd1=np.array([0.0]),
        ru0=np.zeros((1, 2, 1)),
        zu0=np.zeros((1, 2, 1)),
    )
    np.testing.assert_allclose(np.asarray(short), np.zeros((1,)))

    s = np.linspace(0.0, 1.0, 4)
    ard1 = np.array([0.0, -2.0, -4.0, -8.0])
    azd1 = np.array([0.0, -12.0, -8.0, -4.0])
    ru0 = np.zeros((4, 2, 1))
    zu0 = np.full((4, 2, 1), 2.0)
    tcon = np.asarray(
        tcon_from_cached_precondn_diag(
            tcon0=3.0,
            trig=trig,
            s=s,
            lasym=True,
            ard1=ard1,
            azd1=azd1,
            ru0=ru0,
            zu0=zu0,
        )
    )

    hs = s[1] - s[0]
    ns_f = float(s.size)
    tcon_mul = 1.0 * (1.0 + ns_f * (1.0 / 60.0 + ns_f / (200.0 * 120.0)))
    tcon_mul = tcon_mul / ((4.0 * (float(trig.r0scale) ** 2)) ** 2)
    scale = tcon_mul * (32.0 * hs) ** 2
    aznorm = 4.0 * float(np.sum(trig.dnorm3))
    expected_core = np.minimum(np.abs(ard1) / 1.0, np.abs(azd1) / aznorm) * scale

    np.testing.assert_allclose(tcon[0], 1.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(tcon[1:3], expected_core[1:3], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(tcon[-1], 0.5 * expected_core[2], rtol=1e-13, atol=1e-13)


def test_tcon_from_precondn_axisym_matches_cached_diagonal_scaling_and_guards():
    from vmec_jax import preconditioner_1d_jax as p1d

    trig = vmec_trig_tables(ntheta=8, nzeta=3, nfp=1, mmax=3, nmax=0, lasym=False)
    cfg = Cfg(mpol=3, ntor=0, ntheta=8, nzeta=3, nfp=1, lasym=False, lthreed=False)
    s = np.linspace(0.0, 1.0, 4)
    ns = s.size
    shape = (ns, int(trig.ntheta3), int(trig.cosnv.shape[0]))
    radial = np.arange(ns, dtype=float)[:, None, None]
    theta = np.arange(shape[1], dtype=float)[None, :, None]
    zeta = np.arange(shape[2], dtype=float)[None, None, :]
    base = 1.0 + 0.05 * radial + 0.02 * theta + 0.01 * zeta
    bc = SimpleNamespace(
        bsq=2.0 + 0.1 * base,
        bsupv=0.3 + 0.02 * base,
        jac=SimpleNamespace(
            r12=1.1 + 0.03 * base,
            tau=np.ones(shape),
            sqrtg=1.4 + 0.04 * base,
            zs=0.2 + 0.01 * base,
            zu12=0.5 + 0.02 * base,
            rs=0.3 + 0.03 * base,
            ru12=0.4 + 0.01 * base,
        ),
    )
    k = SimpleNamespace(
        pzu_even=0.6 + 0.02 * base,
        pzu_odd=0.2 + 0.01 * base,
        pz1_odd=0.1 + 0.03 * base,
        pru_even=0.5 + 0.01 * base,
        pru_odd=0.3 + 0.02 * base,
        pr1_odd=0.2 + 0.01 * base,
    )
    ru0 = 0.7 + 0.01 * base
    zu0 = 0.8 + 0.02 * base

    got = np.asarray(
        tcon_from_precondn_axisym(
            tcon0=0.8,
            bc=bc,
            k=k,
            cfg=cfg,
            s=s,
            trig=trig,
            ru0=ru0,
            zu0=zu0,
        )
    )

    dtype = np.asarray(trig.cosmu).dtype
    w_int = p1d._wint_from_config(cfg=cfg, dtype=dtype)
    sqrt_sf, sqrt_sh = p1d._sqrt_profiles_from_ns(ns, dtype=dtype)
    sm, sp = p1d._sm_sp_from_profiles(sqrt_sf, sqrt_sh)
    axd_r = p1d._compute_preconditioning_matrix(
        xs=bc.jac.zs[1:],
        xu12=bc.jac.zu12[1:],
        xu_e=k.pzu_even,
        xu_o=k.pzu_odd,
        x1_o=k.pz1_odd,
        r12=bc.jac.r12[1:],
        total_pressure=bc.bsq[1:],
        tau=bc.jac.tau[1:],
        bsupv=bc.bsupv[1:],
        sqrtg=bc.jac.sqrtg[1:],
        w_int=w_int,
        sqrt_sh=sqrt_sh,
        sm=sm,
        sp=sp,
        delta_s=s[1] - s[0],
        ns_full=ns,
    )[1]
    axd_z = p1d._compute_preconditioning_matrix(
        xs=bc.jac.rs[1:],
        xu12=bc.jac.ru12[1:],
        xu_e=k.pru_even,
        xu_o=k.pru_odd,
        x1_o=k.pr1_odd,
        r12=bc.jac.r12[1:],
        total_pressure=bc.bsq[1:],
        tau=bc.jac.tau[1:],
        bsupv=bc.bsupv[1:],
        sqrtg=bc.jac.sqrtg[1:],
        w_int=w_int,
        sqrt_sh=sqrt_sh,
        sm=sm,
        sp=sp,
        delta_s=s[1] - s[0],
        ns_full=ns,
    )[1]
    expected = np.asarray(
        tcon_from_cached_precondn_diag(
            tcon0=0.8,
            trig=trig,
            s=s,
            lasym=False,
            ard1=np.asarray(axd_r)[:, 0],
            azd1=np.asarray(axd_z)[:, 0],
            ru0=ru0,
            zu0=zu0,
        )
    )
    np.testing.assert_allclose(got[1:], expected[1:], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(got[0], 0.0, rtol=0.0, atol=0.0)

    bad_cfg = cfg._replace(lthreed=True)
    with pytest.raises(ValueError, match="only supports axisym"):
        tcon_from_precondn_axisym(tcon0=0.8, bc=bc, k=k, cfg=bad_cfg, s=s, trig=trig, ru0=ru0, zu0=zu0)

    short = tcon_from_precondn_axisym(
        tcon0=0.8,
        bc=bc,
        k=k,
        cfg=cfg,
        s=np.array([0.0]),
        trig=trig,
        ru0=ru0[:1],
        zu0=zu0[:1],
    )
    np.testing.assert_allclose(np.asarray(short), np.zeros((1,)))


def test_alias_gcon_lasym_path_is_signgs_antisymmetric_and_axis_scaled():
    trig = vmec_trig_tables(ntheta=8, nzeta=4, nfp=1, mmax=4, nmax=1, lasym=True)
    ns = 3
    shape = (ns, int(trig.ntheta3), int(trig.cosnv.shape[0]))
    radial = np.arange(ns, dtype=float)[:, None, None]
    theta = np.arange(shape[1], dtype=float)[None, :, None]
    zeta = np.arange(shape[2], dtype=float)[None, None, :]
    ztemp = np.sin(0.7 * theta + 0.3 * zeta) + 0.2 * radial
    tcon = np.array([0.0, 1.1, 0.7])

    gcon_pos = np.asarray(
        alias_gcon(ztemp=ztemp, trig=trig, ntor=1, mpol=4, signgs=1, tcon=tcon, lasym=True)
    )
    gcon_neg = np.asarray(
        alias_gcon(ztemp=ztemp, trig=trig, ntor=1, mpol=4, signgs=-1, tcon=tcon, lasym=True)
    )

    assert gcon_pos.shape == shape
    np.testing.assert_allclose(gcon_pos[0], 0.0, atol=1e-13)
    np.testing.assert_allclose(gcon_neg, -gcon_pos, rtol=1e-13, atol=1e-13)
    assert np.linalg.norm(gcon_pos[1:]) > 0.0
