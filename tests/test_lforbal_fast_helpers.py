from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.kernels.lforbal import (
    MU0,
    TWOPI,
    VmecLforbalFactors,
    _eqfactor_from_precondn_like_vmec,
    _pwint_from_trig,
    _sm_sp_from_s,
    apply_lforbal_to_tomnsps,
    currents_from_bcovar,
    equif_from_bcovar,
    lforbal_factors_from_state,
    plascur_edge_from_bcovar,
)
from vmec_jax.kernels.tomnsp import vmec_trig_tables


def _synthetic_trig():
    return vmec_trig_tables(ntheta=4, nzeta=3, nfp=1, mmax=1, nmax=0, lasym=False, cache=False)


def _synthetic_bcovar(*, ns: int = 4, ntheta: int = 3, nzeta: int = 3):
    radial = np.arange(ns, dtype=float)[:, None, None]
    theta = np.arange(ntheta, dtype=float)[None, :, None]
    zeta = np.arange(nzeta, dtype=float)[None, None, :]
    bsubu = 1.0 + 0.5 * radial + 0.1 * theta + 0.01 * zeta
    bsubv = -0.25 + 0.3 * radial - 0.05 * theta + 0.02 * zeta
    return SimpleNamespace(bsubu=bsubu, bsubv=bsubv)


def test_lforbal_radial_weights_match_vmec_indexing() -> None:
    s = np.linspace(0.0, 1.0, 4)
    sm, sp = (np.asarray(x) for x in _sm_sp_from_s(s))

    hs = s[1] - s[0]
    i = np.arange(s.size + 1, dtype=float)
    psqrts = np.where(i >= 1, np.sqrt(np.maximum(hs * (i - 1.0), 0.0)), 0.0)
    psqrts[-1] = 1.0
    pshalf = np.where(i >= 1, np.sqrt(np.maximum(hs * np.abs(i - 1.5), 0.0)), 0.0)

    expected_sm = np.zeros_like(psqrts)
    expected_sp = np.zeros_like(psqrts)
    expected_sm[2:] = pshalf[2:] / psqrts[2:]
    expected_sp[2:-1] = pshalf[3:] / psqrts[2:-1]
    expected_sp[-1] = 1.0 / psqrts[-1]
    expected_sp[1] = expected_sm[2]

    np.testing.assert_allclose(sm, expected_sm)
    np.testing.assert_allclose(sp, expected_sp)

    one_sm, one_sp = (np.asarray(x) for x in _sm_sp_from_s(np.asarray([0.0])))
    np.testing.assert_allclose(one_sm, [0.0, 0.0])
    np.testing.assert_allclose(one_sp, [0.0, 0.0])


def test_lforbal_current_equif_and_edge_current_formulas() -> None:
    trig = _synthetic_trig()
    bc = _synthetic_bcovar(ntheta=trig.ntheta3, nzeta=trig.cosnv.shape[0])
    s = np.linspace(0.0, 1.0, 4)
    wout = SimpleNamespace(
        signgs=-1,
        vp=np.asarray([1.0, 1.2, 1.4, 1.6]),
        pres=np.asarray([3.0, 2.5, 1.5, 1.0]),
        phipf=np.asarray([2.0, 2.1, 2.2, 2.3]) * TWOPI * -1.0,
        chipf=np.asarray([0.4, 0.5, 0.6, 0.7]) * TWOPI * -1.0,
    )

    buco, bvco, jcuru, jcurv = (np.asarray(x) for x in currents_from_bcovar(bc=bc, trig=trig, wout=wout, s=s))

    pwint = np.asarray(_pwint_from_trig(trig, nzeta=trig.cosnv.shape[0], dtype=float))
    radial_weight = (np.arange(s.size) + 1 >= 2).astype(float)[:, None, None]
    expected_buco = np.sum(bc.bsubu * radial_weight * pwint[None, :, :], axis=(1, 2))
    expected_bvco = np.sum(bc.bsubv * radial_weight * pwint[None, :, :], axis=(1, 2))
    hs = s[1] - s[0]
    expected_jcurv = -1.0 * (np.r_[expected_buco[1:], 0.0] - expected_buco) / hs / MU0
    expected_jcuru = (np.r_[expected_bvco[1:], 0.0] - expected_bvco) / hs / MU0
    np.testing.assert_allclose(buco, expected_buco)
    np.testing.assert_allclose(bvco, expected_bvco)
    np.testing.assert_allclose(jcurv, expected_jcurv)
    np.testing.assert_allclose(jcuru, expected_jcuru)

    ctor = np.asarray(plascur_edge_from_bcovar(bc=bc, trig=trig, wout=wout, s=s))
    buco_edge = np.sum(bc.bsubu[-2:] * pwint[None, :, :], axis=(1, 2))
    np.testing.assert_allclose(ctor, TWOPI * (1.5 * buco_edge[-1] - 0.5 * buco_edge[-2]))

    equif = np.asarray(equif_from_bcovar(bc=bc, trig=trig, wout=wout, s=s))
    vpphi = 0.5 * (np.r_[wout.vp[1:], wout.vp[-1]] + wout.vp)
    presgrad = (np.r_[wout.pres[1:], wout.pres[-1]] - wout.pres) / hs
    phipf = wout.phipf / (TWOPI * wout.signgs)
    chipf = wout.chipf / (TWOPI * wout.signgs)
    expected_equif = ((-phipf * expected_jcuru + chipf * expected_jcurv) / vpphi) + presgrad
    expected_equif[0] = 0.0
    expected_equif[-1] = 0.0
    np.testing.assert_allclose(equif, expected_equif)


def test_apply_lforbal_correction_only_changes_interior_m1n0() -> None:
    trig = _synthetic_trig()
    frcc = np.arange(16, dtype=float).reshape(4, 2, 2)
    fzsc = 100.0 + np.arange(16, dtype=float).reshape(4, 2, 2)
    factors = VmecLforbalFactors(
        rzu_fac=np.asarray([0.0, 2.0, 3.0, 0.0]),
        rru_fac=np.asarray([0.0, 5.0, 7.0, 0.0]),
        frcc_fac=np.asarray([0.0, 0.25, 0.5, 0.0]),
        fzsc_fac=np.asarray([0.0, -0.125, -0.25, 0.0]),
        equif=np.asarray([0.0, 11.0, 13.0, 0.0]),
    )

    fr_out, fz_out = (np.asarray(x) for x in apply_lforbal_to_tomnsps(frcc=frcc, fzsc=fzsc, factors=factors, trig=trig))

    expected_fr = frcc.copy()
    expected_fz = fzsc.copy()
    for js in (1, 2):
        work = factors.frcc_fac[js] * frcc[js, 1, 0] + factors.fzsc_fac[js] * fzsc[js, 1, 0]
        expected_fr[js, 1, 0] = factors.rzu_fac[js] * (trig.r0scale * factors.equif[js] + work)
        expected_fz[js, 1, 0] = factors.rru_fac[js] * (trig.r0scale * factors.equif[js] - work)

    np.testing.assert_allclose(fr_out, expected_fr)
    np.testing.assert_allclose(fz_out, expected_fz)

    short_fr, short_fz = apply_lforbal_to_tomnsps(
        frcc=np.zeros((1, 1, 1)),
        fzsc=np.ones((1, 1, 1)),
        factors=factors,
        trig=trig,
    )
    np.testing.assert_allclose(np.asarray(short_fr), np.zeros((1, 1, 1)))
    np.testing.assert_allclose(np.asarray(short_fz), np.ones((1, 1, 1)))


def test_lforbal_factor_guards_and_eqfactor_shape_validation() -> None:
    trig = _synthetic_trig()
    wout = SimpleNamespace(mpol=1, signgs=1)
    factors = lforbal_factors_from_state(
        bc=SimpleNamespace(),
        trig=trig,
        wout=wout,
        s=np.asarray([0.0, 1.0]),
        pru_even=np.zeros((2, trig.ntheta3, trig.cosnv.shape[0])),
        pru_odd=np.zeros((2, trig.ntheta3, trig.cosnv.shape[0])),
        pzu_even=np.zeros((2, trig.ntheta3, trig.cosnv.shape[0])),
        pzu_odd=np.zeros((2, trig.ntheta3, trig.cosnv.shape[0])),
        pr1_odd=np.zeros((2, trig.ntheta3, trig.cosnv.shape[0])),
        pz1_odd=np.zeros((2, trig.ntheta3, trig.cosnv.shape[0])),
    )
    np.testing.assert_allclose(np.asarray(factors.equif), [0.0, 0.0])
    np.testing.assert_allclose(np.asarray(factors.rzu_fac), [0.0, 0.0])

    shape = (4, trig.ntheta3, trig.cosnv.shape[0])
    ones = np.ones(shape)
    with pytest.raises(ValueError, match="trigmult must have shape"):
        _eqfactor_from_precondn_like_vmec(
            bsq=ones,
            sqrtg=ones,
            r12=ones,
            xu12=ones,
            xue=ones,
            xuo=np.zeros(shape),
            trigmult=np.ones((1, 1)),
            trig=trig,
            wout=SimpleNamespace(vp=np.ones(4), signgs=1),
            s=np.linspace(0.0, 1.0, 4),
        )


def test_lforbal_factors_from_synthetic_state_are_finite_on_interior_surfaces() -> None:
    trig = _synthetic_trig()
    s = np.linspace(0.0, 1.0, 4)
    shape = (s.size, trig.ntheta3, trig.cosnv.shape[0])
    radial = np.arange(s.size, dtype=float)[:, None, None]
    theta = np.arange(trig.ntheta3, dtype=float)[None, :, None]
    zeta = np.arange(trig.cosnv.shape[0], dtype=float)[None, None, :]

    bc = SimpleNamespace(
        bsubu=1.0 + 0.2 * radial + 0.01 * theta + 0.001 * zeta,
        bsubv=0.5 + 0.1 * radial - 0.02 * theta + 0.003 * zeta,
        bsq=np.ones(shape) * 2.0,
        jac=SimpleNamespace(
            sqrtg=np.ones(shape) * 1.5,
            r12=np.ones(shape) * 1.2,
            zu12=np.ones(shape) * 0.7,
            ru12=np.ones(shape) * 0.8,
        ),
    )
    wout = SimpleNamespace(
        mpol=2,
        signgs=1,
        vp=np.ones(s.size) * 2.0,
        pres=np.linspace(1.0, 0.0, s.size),
        phipf=np.ones(s.size) * TWOPI,
        chipf=np.ones(s.size) * 0.5 * TWOPI,
    )

    factors = lforbal_factors_from_state(
        bc=bc,
        trig=trig,
        wout=wout,
        s=s,
        pru_even=np.ones(shape),
        pru_odd=np.zeros(shape),
        pzu_even=np.ones(shape) * 1.1,
        pzu_odd=np.zeros(shape),
        pr1_odd=np.zeros(shape),
        pz1_odd=np.zeros(shape),
    )

    for arr in (factors.rzu_fac, factors.rru_fac, factors.frcc_fac, factors.fzsc_fac, factors.equif):
        assert np.all(np.isfinite(np.asarray(arr)))
    np.testing.assert_allclose(np.asarray(factors.rzu_fac)[[0, -1]], [0.0, 0.0])
    np.testing.assert_allclose(np.asarray(factors.rru_fac)[[0, -1]], [0.0, 0.0])
    assert np.any(np.asarray(factors.rzu_fac)[1:-1] != 0.0)
    assert np.any(np.asarray(factors.rru_fac)[1:-1] != 0.0)
