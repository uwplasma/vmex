from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.driver as driver
import vmec_jax.wout as wout
from vmec_jax.boundary import BoundaryCoeffs
from vmec_jax.config import VMECConfig
from vmec_jax.init_guess import initial_guess_from_boundary
from vmec_jax.modes import ModeTable, vmec_mode_table
from vmec_jax.namelist import InData
from vmec_jax.static import build_static
from vmec_jax.state import StateLayout, VMECState
from vmec_jax.kernels.bcovar import vmec_bcovar_half_mesh_from_wout
from vmec_jax.kernels.forces import VmecRZResidualCoeffs, rz_residual_scalars_like_vmec
from vmec_jax.kernels.realspace import (
    vmec_realspace_analysis,
    vmec_realspace_synthesis,
    vmec_realspace_synthesis_multi,
)
from vmec_jax.kernels.tomnsp import vmec_trig_tables


def _k_index(modes, m: int, n: int) -> int:
    for k, (mm, nn) in enumerate(zip(modes.m, modes.n)):
        if int(mm) == int(m) and int(nn) == int(n):
            return k
    raise KeyError((m, n))


def _circular_axisymmetric_case(*, ns: int = 4):
    cfg = VMECConfig(
        mpol=2,
        ntor=0,
        ns=ns,
        nfp=1,
        lasym=False,
        lthreed=False,
        lconm1=False,
        ntheta=10,
        nzeta=1,
    )
    static = build_static(cfg)
    modes = static.modes
    k00 = _k_index(modes, 0, 0)
    k10 = _k_index(modes, 1, 0)
    s = np.asarray(static.s, dtype=float)
    zeros = np.zeros((ns, int(modes.K)), dtype=float)
    rcos = zeros.copy()
    zsin = zeros.copy()
    rcos[:, k00] = 3.0
    rcos[:, k10] = np.sqrt(s) / np.sqrt(2.0)
    zsin[:, k10] = np.sqrt(s) / np.sqrt(2.0)
    state = SimpleNamespace(
        Rcos=rcos,
        Rsin=zeros.copy(),
        Zcos=zeros.copy(),
        Zsin=zsin,
        Lcos=zeros.copy(),
        Lsin=zeros.copy(),
    )
    wout_like = SimpleNamespace(
        phipf=np.ones(ns),
        phips=np.r_[0.0, np.ones(ns - 1)],
        chipf=np.zeros(ns),
        iotaf=np.zeros(ns),
        iotas=np.zeros(ns),
        signgs=1,
        nfp=1,
        mpol=2,
        ntor=0,
        lasym=False,
        flux_is_internal=True,
        ncurr=0,
        lcurrent=False,
        icurv=np.zeros(ns),
        pres=np.zeros(ns),
    )
    return static, state, wout_like


def test_wout_eqfor_profiles_beta_equif_and_ctor_branches() -> None:
    trig = vmec_trig_tables(ntheta=4, nzeta=1, nfp=1, mmax=1, nmax=0, lasym=False, cache=False)
    wint = wout._vmec_wint_from_trig(trig)
    ns = 4
    shape = (ns, *wint.shape)
    pres = np.asarray([0.0, 0.2, 0.3, 0.4])
    vp = np.asarray([1.0, 1.2, 1.3, 1.4])
    bsq = np.full(shape, 2.0)
    r12 = np.full(shape, 3.0)
    bsupv = np.full(shape, 0.25)
    sqrtg = np.ones(shape)

    betapol, betator, betatot, betaxis = wout._compute_eqfor_beta(
        pres=pres,
        vp=vp,
        bsq=bsq,
        r12=r12,
        bsupv=bsupv,
        sqrtg=sqrtg,
        wint=wint,
        signgs=1,
    )

    assert all(np.isfinite(v) for v in (betapol, betator, betatot, betaxis))
    assert betatot > 0.0
    assert betaxis == pytest.approx(
        wout._compute_eqfor_betaxis(pres=pres, vp=vp, bsq=bsq, sqrtg=sqrtg, wint=wint, signgs=1)
    )
    assert wout._compute_eqfor_beta(
        pres=pres[:2],
        vp=vp[:2],
        bsq=bsq[:2],
        r12=r12[:2],
        bsupv=bsupv[:2],
        sqrtg=sqrtg[:2],
        wint=wint,
        signgs=1,
    ) == (0.0, 0.0, 0.0, 0.0)

    weight_sum = float(np.sum(wint))
    target_buco = np.asarray([0.0, 1.0, 3.0, 6.0])
    bsubu = np.zeros(shape)
    bsubv = np.ones(shape)
    for js in range(1, ns):
        bsubu[js] = target_buco[js] / weight_sum

    buco, bvco, jcuru, jcurv, equif = wout._compute_equif_wout(
        bsubu=bsubu,
        bsubv=bsubv,
        pres=np.zeros(ns),
        vp=np.ones(ns),
        phipf=np.ones(ns),
        chipf=np.ones(ns),
        signgs=1,
        trig=trig,
        s=np.linspace(0.0, 1.0, ns),
    )
    np.testing.assert_allclose(buco, target_buco, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(bvco[1:], 1.0, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(jcuru, 0.0, rtol=0.0, atol=1.0e-12)
    assert np.all(jcurv > 0.0)
    np.testing.assert_allclose(equif, 1.0, rtol=1.0e-13, atol=1.0e-13)

    indata_default = InData(scalars={}, indexed={})
    indata_prec2d = InData(scalars={"LFREEB": True, "ICTRL_PREC2D": 1, "LHESS_EXACT": True}, indexed={})
    extrapolated = wout._compute_ctor_from_buco(buco=target_buco, signgs=1, indata=indata_default)
    edge_direct = wout._compute_ctor_from_buco(buco=target_buco, signgs=1, indata=indata_prec2d)
    assert extrapolated == pytest.approx((2.0 * np.pi * (1.5 * 6.0 - 0.5 * 3.0)) / wout.MU0)
    assert edge_direct == pytest.approx((2.0 * np.pi * 6.0) / wout.MU0)


def test_init_guess_zero_axis_branch_and_surface_regular_scaling() -> None:
    cfg = VMECConfig(
        mpol=3,
        ntor=1,
        ns=4,
        nfp=1,
        lasym=True,
        lthreed=True,
        lconm1=False,
        ntheta=8,
        nzeta=4,
    )
    static = build_static(cfg)
    K = int(static.modes.K)
    Rcos = np.zeros(K)
    Rsin = np.zeros(K)
    Zcos = np.zeros(K)
    Zsin = np.zeros(K)
    k00 = _k_index(static.modes, 0, 0)
    k01 = _k_index(static.modes, 0, 1)
    k20 = _k_index(static.modes, 2, 0)
    Rcos[k00] = 9.0
    Rsin[k01] = 0.6
    Zcos[k01] = -0.4
    Rcos[k20] = 1.2
    Zsin[k20] = 0.8
    boundary = BoundaryCoeffs(R_cos=Rcos, R_sin=Rsin, Z_cos=Zcos, Z_sin=Zsin)

    state = initial_guess_from_boundary(
        static,
        boundary,
        InData(scalars={}, indexed={}),
        infer_axis_if_missing=False,
        vmec_project=False,
    )

    s = np.asarray(static.s)
    mode_scale = np.asarray(static.mode_scale_internal)
    np.testing.assert_allclose(np.asarray(state.Rcos)[:, k00], s * Rcos[k00] * mode_scale[k00])
    np.testing.assert_allclose(np.asarray(state.Rsin)[:, k01], s * Rsin[k01] * mode_scale[k01])
    np.testing.assert_allclose(np.asarray(state.Zcos)[:, k01], s * Zcos[k01] * mode_scale[k01])
    np.testing.assert_allclose(np.asarray(state.Rcos)[:, k20], s * Rcos[k20] * mode_scale[k20])
    np.testing.assert_allclose(np.asarray(state.Zsin)[:, k20], s * Zsin[k20] * mode_scale[k20])
    np.testing.assert_allclose(np.asarray(state.Rcos)[0, [k00, k01, k20]], 0.0)
    np.testing.assert_allclose(np.asarray(state.Rcos)[-1, k00], Rcos[k00] * mode_scale[k00])


def test_realspace_parity_filters_and_multiderivative_invariant() -> None:
    modes = vmec_mode_table(mpol=3, ntor=1)
    trig = vmec_trig_tables(ntheta=10, nzeta=5, nfp=2, mmax=2, nmax=1, lasym=False, cache=False)
    ns = 3
    surfaces = np.arange(ns, dtype=float)[:, None]
    mode_id = np.arange(int(modes.K), dtype=float)[None, :]
    coeff_cos = 0.2 + 0.03 * surfaces + 0.01 * mode_id
    coeff_sin = -0.1 + 0.02 * surfaces - 0.005 * mode_id
    coeff_sin[:, 0] = 0.0

    field = vmec_realspace_synthesis(coeff_cos=coeff_cos, coeff_sin=coeff_sin, modes=modes, trig=trig)
    cos_only, sin_zero = vmec_realspace_analysis(f=field, modes=modes, trig=trig, parity="cos")
    cos_zero, sin_only = vmec_realspace_analysis(f=field, modes=modes, trig=trig, parity="sin")
    both_cos, both_sin = vmec_realspace_analysis(f=field, modes=modes, trig=trig, parity="both")

    np.testing.assert_allclose(np.asarray(cos_only), np.asarray(both_cos), rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(np.asarray(sin_only), np.asarray(both_sin), rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(np.asarray(sin_zero), 0.0, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(cos_zero), 0.0, rtol=0.0, atol=1.0e-14)
    with pytest.raises(ValueError, match="parity must be one of"):
        vmec_realspace_analysis(f=field, modes=modes, trig=trig, parity="bad")

    base_only = vmec_realspace_synthesis_multi(
        coeff_cos=coeff_cos,
        coeff_sin=coeff_sin,
        modes=modes,
        trig=trig,
        derivs=("base",),
    )
    assert len(base_only) == 1
    np.testing.assert_allclose(np.asarray(base_only[0]), np.asarray(field), rtol=1.0e-13, atol=1.0e-13)


def test_bcovar_free_boundary_edge_override_and_force_scalar_normalization() -> None:
    static, state, wout_like = _circular_axisymmetric_case()
    base = vmec_bcovar_half_mesh_from_wout(state=state, static=static, wout=wout_like)
    vac_edge = np.full(np.asarray(base.bsq).shape[1:], 7.5)
    freeb = vmec_bcovar_half_mesh_from_wout(
        state=state,
        static=static,
        wout=wout_like,
        freeb_bsqvac_edge=vac_edge,
    )

    np.testing.assert_allclose(np.asarray(freeb.bsq)[:-1], np.asarray(base.bsq)[:-1], rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(np.asarray(freeb.bsq)[-1], vac_edge, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(np.asarray(freeb.lu_e)[-1], vac_edge * np.asarray(freeb.jac.r12)[-1])
    with pytest.raises(ValueError, match="freeb_bsqvac_edge must have shape"):
        vmec_bcovar_half_mesh_from_wout(
            state=state,
            static=static,
            wout=wout_like,
            freeb_bsqvac_edge=np.zeros((1, 1, 1)),
        )

    coeffs = VmecRZResidualCoeffs(
        gcr_cos=np.asarray([[100.0, 100.0], [1.0, 2.0], [3.0, 4.0]]),
        gcr_sin=np.asarray([[100.0, 100.0], [0.5, 1.5], [2.5, 3.5]]),
        gcz_cos=np.asarray([[100.0, 100.0], [2.0, 1.0], [4.0, 3.0]]),
        gcz_sin=np.asarray([[100.0, 100.0], [1.5, 0.5], [3.5, 2.5]]),
    )
    bc = SimpleNamespace(
        jac=SimpleNamespace(r12=np.ones((3, 2, 1))),
        gij_b_uu=np.full((3, 2, 1), 2.0),
    )
    scalars = rz_residual_scalars_like_vmec(
        coeffs,
        bc=bc,
        wout=SimpleNamespace(volume_p=8.0 * np.pi**2, wb=6.0, wp=4.0),
        s=np.asarray([0.0, 0.5, 1.0]),
    )
    expected_gnorm = 0.25 / (2.0 * (6.0 / 2.0) ** 2)
    expected_r = expected_gnorm * float(np.sum(coeffs.gcr_cos[1:] ** 2 + coeffs.gcr_sin[1:] ** 2))
    expected_z = expected_gnorm * float(np.sum(coeffs.gcz_cos[1:] ** 2 + coeffs.gcz_sin[1:] ** 2))
    assert scalars.fsqr_like == pytest.approx(expected_r)
    assert scalars.fsqz_like == pytest.approx(expected_z)
    assert np.isnan(rz_residual_scalars_like_vmec(coeffs, bc=bc, wout=SimpleNamespace(), s=np.asarray([0.0])).fsqr_like)


def test_driver_policy_list_parsing_and_scan_selection_edges() -> None:
    assert driver._as_float_list(("1.5", np.float64(2.5))) == [1.5, 2.5]
    assert driver._as_float_list(object()) is None
    assert driver._as_list_like(np.asarray([1, 2])) == [1, 2]
    assert driver._as_list_like(np.int64(7)) == [np.int64(7)]
    assert driver._as_list_like(iter((3, 4))) == [3, 4]

    indata = SimpleNamespace()
    assert driver._default_use_scan_for_backend(indata, backend="cpu", solver_mode=None) is True
    assert driver._default_use_scan_for_backend(indata, backend="gpu", solver_mode="accelerated") is True
    with pytest.raises(ValueError, match="Unknown solver_mode"):
        driver._default_use_scan_for_backend(indata, backend="cpu", solver_mode="invalid")


def test_bcovar_pytree_roundtrip_preserves_field_order() -> None:
    children = tuple(np.asarray([float(i)]) for i in range(41))
    aux = None
    from vmec_jax.kernels.bcovar import VmecHalfMeshBcovar

    bc = VmecHalfMeshBcovar.tree_unflatten(aux, children)
    flattened, flattened_aux = bc.tree_flatten()

    assert flattened_aux is None
    assert len(flattened) == len(children)
    for actual, expected in zip(flattened, children, strict=True):
        np.testing.assert_allclose(actual, expected)


def test_state_layout_synthetic_mode_count_matches_realspace_shapes() -> None:
    modes = ModeTable(m=np.asarray([0, 1]), n=np.asarray([0, 0]))
    layout = StateLayout(ns=2, K=2, lasym=False)
    state = VMECState(
        layout=layout,
        Rcos=np.ones((2, 2)),
        Rsin=np.zeros((2, 2)),
        Zcos=np.zeros((2, 2)),
        Zsin=np.ones((2, 2)),
        Lcos=np.zeros((2, 2)),
        Lsin=np.zeros((2, 2)),
    )
    trig = vmec_trig_tables(ntheta=6, nzeta=1, nfp=1, mmax=1, nmax=0, lasym=False, cache=False)
    field = vmec_realspace_synthesis(coeff_cos=state.Rcos, coeff_sin=state.Rsin, modes=modes, trig=trig)

    assert int(state.layout.K) == int(modes.K)
    assert np.asarray(field).shape == (2, int(trig.ntheta2), 1)
