from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.integrals import dvds_from_sqrtg_zeta
from vmec_jax.preconditioner_1d import (
    _sm_sp_from_profiles,
    _sm_sp_from_s,
    _tridiagonal_solve,
    lambda_preconditioner,
)
from vmec_jax.solve_scan_math_helpers import _ptau_minmax_from_k_host, _state_jacobian
from vmec_jax.vmec_lforbal import (
    _eqfactor_from_precondn_like_vmec,
    _pshalf_from_s,
    currents_from_bcovar,
    equif_from_bcovar,
    plascur_edge_from_bcovar,
)
from vmec_jax.vmec_residue import (
    vmec_apply_m1_constraints,
    vmec_fsq_sums_from_tomnsps,
    vmec_gcx2_from_tomnsps,
    vmec_zero_m1_zforce,
)
from vmec_jax.vmec_tomnsp import TomnspsRZL, vmec_trig_tables


def _ptau_kernel(ns: int = 3) -> SimpleNamespace:
    zeros = np.zeros((ns, 1, 1), dtype=float)
    pz1_even = np.asarray([0.0, 1.0, 3.0], dtype=float)[:ns, None, None]
    return SimpleNamespace(
        pru_even=np.full((ns, 1, 1), 2.0),
        pru_odd=zeros,
        pzu_even=zeros,
        pzu_odd=zeros,
        pr1_even=zeros,
        pr1_odd=zeros,
        pz1_even=pz1_even,
        pz1_odd=zeros,
    )


def _tomnsps(*, ns: int, mpol: int, value: float = 1.0, optional: bool = False) -> TomnspsRZL:
    shape = (ns, mpol, 1)
    base = np.full(shape, value, dtype=float)
    opt = np.full(shape, value + 1.0, dtype=float) if optional else None
    return TomnspsRZL(
        frcc=base,
        frss=opt,
        fzsc=2.0 * base,
        fzcs=opt,
        flsc=3.0 * base,
        flcs=opt,
        frsc=opt,
        frcs=opt,
        fzcc=opt,
        fzss=opt,
        flcc=opt,
        flss=opt,
    )


def test_scan_ptau_host_fallbacks_and_absolute_vmec_tolerance() -> None:
    ptau_min, ptau_max = _ptau_minmax_from_k_host(
        _ptau_kernel(),
        pshalf=np.asarray([1.0]),  # resized to the radial mesh length
        ohs=2.0,
    )
    assert (ptau_min, ptau_max) == pytest.approx((4.0, 8.0))

    short = _ptau_minmax_from_k_host(
        _ptau_kernel(ns=1),
        pshalf=np.ones(1),
        ohs=1.0,
        compute_jit=lambda *_args: pytest.fail("short kernels should not call the JIT path"),
        pshalf_jax=np.ones(1),
        ohs_jax=1.0,
    )
    assert short == (None, None)

    bad_arrays = SimpleNamespace(
        pru_even=object(),
        pru_odd=object(),
        pzu_even=object(),
        pzu_odd=object(),
        pr1_even=object(),
        pr1_odd=object(),
        pz1_even=object(),
        pz1_odd=object(),
    )
    assert _ptau_minmax_from_k_host(bad_arrays, pshalf=np.ones(2), ohs=1.0) == (None, None)

    decision = _state_jacobian(
        np.asarray([99.0, -2.0e-6, 3.0e-6]),
        vmec2000_control=True,
        ptau_tol=1.0e-6,
    )
    assert bool(np.asarray(decision.bad_jacobian))
    assert float(np.asarray(decision.min_tau)) == pytest.approx(-2.0e-6)
    assert float(np.asarray(decision.max_tau)) == pytest.approx(3.0e-6)


def test_residue_single_mode_and_default_scalxc_edge_policies() -> None:
    single_m = _tomnsps(ns=2, mpol=1, optional=True)
    assert vmec_apply_m1_constraints(frzl=single_m, lconm1=True) is single_m
    assert vmec_zero_m1_zforce(frzl=single_m, enabled=True) is single_m

    one_surface = TomnspsRZL(
        frcc=np.asarray([[[2.0]]]),
        frss=None,
        fzsc=np.asarray([[[3.0]]]),
        fzcs=None,
        flsc=np.asarray([[[4.0]]]),
        flcs=None,
    )
    gcr2, gcz2, gcl2 = vmec_gcx2_from_tomnsps(
        frzl=one_surface,
        apply_m1_constraints=False,
        apply_scalxc=False,
        include_edge=False,
    )
    np.testing.assert_allclose(np.asarray([gcr2, gcz2, gcl2]), [4.0, 9.0, 16.0])

    sums = vmec_fsq_sums_from_tomnsps(
        frzl=_tomnsps(ns=3, mpol=2, value=1.0, optional=False),
        apply_m1_constraints=False,
        apply_scalxc=True,
        include_edge=False,
    )
    assert sums.gcr2_blocks == {"frcc": pytest.approx(6.0)}
    assert sums.gcz2_blocks == {"fzsc": pytest.approx(24.0)}
    assert sums.gcl2_blocks == {"flsc": pytest.approx(72.0)}


def test_preconditioner_degenerate_axis_and_faclam_contracts() -> None:
    sm, sp = _sm_sp_from_profiles(np.asarray([1.0]), np.asarray([]))
    assert sm.shape == (0,)
    assert sp.shape == (0,)
    sm_s, sp_s = _sm_sp_from_s(np.asarray([0.0]))
    assert sm_s.shape == (0,)
    assert sp_s.shape == (0,)

    cfg = SimpleNamespace(mpol=2, ntor=1, nfp=1, ntheta=4, nzeta=2, lasym=False, lthreed=False)
    short_bc = SimpleNamespace(
        guu=np.ones((1, 3, 2)),
        guv=np.zeros((1, 3, 2)),
        gvv=np.ones((1, 3, 2)),
        jac=SimpleNamespace(sqrtg=np.ones((1, 3, 2))),
        lamscale=1.0,
    )
    lam_short, debug = lambda_preconditioner(
        bc=short_bc,
        trig=None,
        s=np.asarray([0.0]),
        cfg=cfg,
        return_debug=True,
    )
    assert lam_short.shape == (1, 2, 2)
    np.testing.assert_allclose(debug["blam_pre"], [0.0])
    np.testing.assert_allclose(debug["dlam_post"], [0.0])

    bc = SimpleNamespace(
        guu=np.full((2, 3, 2), 2.0),
        guv=np.zeros((2, 3, 2)),
        gvv=np.full((2, 3, 2), 3.0),
        jac=SimpleNamespace(sqrtg=np.ones((2, 3, 2))),
        lamscale=1.0,
    )
    lam_prec, faclam = lambda_preconditioner(
        bc=bc,
        trig=None,
        s=np.asarray([0.0, 1.0]),
        cfg=cfg,
        return_faclam=True,
        r0scale=1.0,
    )
    np.testing.assert_allclose(faclam, lam_prec)
    assert lam_prec[0, 0, 0] == pytest.approx(0.25)
    np.testing.assert_allclose(lam_prec[0, 0, 1:], 0.0)
    np.testing.assert_allclose(lam_prec[0, 1, :], 0.0)

    tri = _tridiagonal_solve(
        a=np.zeros(4),
        d=np.asarray([1.0, 0.0, 0.0, 1.0]),
        b=np.zeros(4),
        rhs=np.asarray([1.0, 2.0, 3.0, 4.0]),
        jmin=0,
        jmax=4,
    )
    assert np.all(np.isfinite(tri))


def test_lforbal_short_radial_mesh_helpers_return_zero_physics_quantities() -> None:
    trig = vmec_trig_tables(ntheta=4, nzeta=2, nfp=1, mmax=1, nmax=0, lasym=False, cache=False)
    shape = (1, trig.ntheta3, trig.cosnv.shape[0])
    bc = SimpleNamespace(bsubu=np.ones(shape), bsubv=np.ones(shape))
    wout = SimpleNamespace(
        signgs=1,
        vp=np.ones(1),
        pres=np.ones(1),
        phipf=np.ones(1),
        chipf=np.ones(1),
    )
    s = np.asarray([0.25])

    np.testing.assert_allclose(np.asarray(_pshalf_from_s(s)), [0.5])
    for arr in currents_from_bcovar(bc=bc, trig=trig, wout=wout, s=s):
        np.testing.assert_allclose(np.asarray(arr), [0.0])
    assert float(np.asarray(plascur_edge_from_bcovar(bc=bc, trig=trig, wout=wout, s=s))) == 0.0
    np.testing.assert_allclose(np.asarray(equif_from_bcovar(bc=bc, trig=trig, wout=wout, s=s)), [0.0])

    eqfactor = _eqfactor_from_precondn_like_vmec(
        bsq=np.ones(shape),
        sqrtg=np.ones(shape),
        r12=np.ones(shape),
        xu12=np.ones(shape),
        xue=np.ones(shape),
        xuo=np.ones(shape),
        trigmult=np.ones(shape[1:]),
        trig=trig,
        wout=wout,
        s=s,
    )
    np.testing.assert_allclose(np.asarray(eqfactor), [0.0])


def test_zeta_volume_guard_rejects_empty_periodic_grid() -> None:
    with pytest.raises(ValueError, match="theta and zeta must be non-empty"):
        dvds_from_sqrtg_zeta(
            np.zeros((1, 2, 0)),
            np.asarray([0.0, np.pi]),
            np.asarray([]),
            signgs=1,
        )
