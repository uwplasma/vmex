from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vmec_jax.config import VMECConfig
from vmec_jax.static import build_static
from vmec_jax.kernels.bcovar import (
    _apply_vmec_lambda_axis_closure,
    _half_mesh_from_even_odd,
    _metric_cross_even_odd,
    _metric_even_odd,
    _pshalf_from_s,
    vmec_bcovar_half_mesh_from_wout,
)
from vmec_jax.kernels.residue import vmec_pwint_from_trig


def _circular_axisymmetric_case(*, ns: int = 4):
    cfg = VMECConfig(
        mpol=2,
        ntor=0,
        ns=ns,
        nfp=1,
        lasym=False,
        lthreed=False,
        lconm1=False,
        ntheta=12,
        nzeta=1,
    )
    static = build_static(cfg)
    modes = static.modes
    idx_m0 = int(np.flatnonzero((modes.m == 0) & (modes.n == 0))[0])
    idx_m1 = int(np.flatnonzero((modes.m == 1) & (modes.n == 0))[0])
    nmodes = int(modes.K)
    s = np.asarray(static.s)

    zeros = np.zeros((ns, nmodes), dtype=float)
    rcos = zeros.copy()
    rsin = zeros.copy()
    zcos = zeros.copy()
    zsin = zeros.copy()
    lcos = zeros.copy()
    lsin = zeros.copy()
    rcos[:, idx_m0] = 3.0
    rcos[:, idx_m1] = np.sqrt(s) / np.sqrt(2.0)
    zsin[:, idx_m1] = np.sqrt(s) / np.sqrt(2.0)
    state = SimpleNamespace(Rcos=rcos, Rsin=rsin, Zcos=zcos, Zsin=zsin, Lcos=lcos, Lsin=lsin)
    wout = SimpleNamespace(
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
    return static, state, wout


def test_pshalf_and_half_mesh_staggering_match_vmec_rules():
    s = np.array([0.0, 0.25, 1.0])
    pshalf = np.asarray(_pshalf_from_s(s))
    expected_pshalf = np.sqrt([0.125, 0.125, 0.625])
    np.testing.assert_allclose(pshalf, expected_pshalf)

    even = np.array([2.0, 4.0, 8.0])[:, None, None]
    odd = np.array([10.0, 20.0, 40.0])[:, None, None]
    half = np.asarray(_half_mesh_from_even_odd(even, odd, s=s))[:, 0, 0]

    expected_first = 0.5 * (4.0 + 2.0 + expected_pshalf[1] * (20.0 + 10.0))
    expected_second = 0.5 * (8.0 + 4.0 + expected_pshalf[2] * (40.0 + 20.0))
    np.testing.assert_allclose(half, [expected_first, expected_first, expected_second])

    one_surface_even = np.array([7.0])[:, None, None]
    one_surface_odd = np.array([11.0])[:, None, None]
    np.testing.assert_allclose(
        np.asarray(_half_mesh_from_even_odd(one_surface_even, one_surface_odd, s=np.array([0.5]))),
        one_surface_even,
    )
    np.testing.assert_allclose(np.asarray(_pshalf_from_s(np.array([-0.25]))), [0.0])


def test_metric_even_odd_decompositions_match_explicit_products():
    s = np.array([0.0, 0.25])
    a0 = np.array([[[2.0]], [[3.0]]])
    a1 = np.array([[[5.0]], [[7.0]]])
    b0 = np.array([[[11.0]], [[13.0]]])
    b1 = np.array([[[17.0]], [[19.0]]])

    even, odd = _metric_even_odd(a0=a0, a1=a1, b0=b0, b1=b1, s=s)
    expected_even = a0 * a0 + b0 * b0 + s[:, None, None] * (a1 * a1 + b1 * b1)
    expected_odd = 2.0 * (a0 * a1 + b0 * b1)
    np.testing.assert_allclose(np.asarray(even), expected_even)
    np.testing.assert_allclose(np.asarray(odd), expected_odd)

    cross_even, cross_odd = _metric_cross_even_odd(a0=a0, a1=a1, b0=b0, b1=b1, s=s)
    np.testing.assert_allclose(np.asarray(cross_even), a0 * b0 + s[:, None, None] * (a1 * b1))
    np.testing.assert_allclose(np.asarray(cross_odd), a0 * b1 + a1 * b0)


def test_lambda_axis_closure_copies_m0_npos_modes():
    lsin = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
            [100.0, 200.0, 300.0, 400.0],
        ],
        dtype=float,
    )
    m_modes = np.array([0, 0, 1, 0], dtype=int)
    n_modes = np.array([0, 1, 1, 2], dtype=int)

    out = np.asarray(
        _apply_vmec_lambda_axis_closure(
            Lsin=lsin,
            m_modes=m_modes,
            n_modes=n_modes,
            lthreed=True,
            ntor=2,
        )
    )

    np.testing.assert_allclose(out[0, 0], lsin[0, 0])  # n=0 unchanged
    np.testing.assert_allclose(out[0, 1], lsin[1, 1])  # m=0,n>0 copied
    np.testing.assert_allclose(out[0, 2], lsin[0, 2])  # m!=0 unchanged
    np.testing.assert_allclose(out[0, 3], lsin[1, 3])  # m=0,n>0 copied
    np.testing.assert_allclose(out[1:], lsin[1:])  # interior unchanged


def test_lambda_axis_closure_disabled_for_axisymmetric_or_ntor_zero():
    lsin = np.array([[1.0, 2.0], [10.0, 20.0]], dtype=float)
    m_modes = np.array([0, 0], dtype=int)
    n_modes = np.array([0, 1], dtype=int)

    out_axis = np.asarray(
        _apply_vmec_lambda_axis_closure(
            Lsin=lsin,
            m_modes=m_modes,
            n_modes=n_modes,
            lthreed=False,
            ntor=1,
        )
    )
    out_ntor0 = np.asarray(
        _apply_vmec_lambda_axis_closure(
            Lsin=lsin,
            m_modes=m_modes,
            n_modes=n_modes,
            lthreed=True,
            ntor=0,
        )
    )
    np.testing.assert_allclose(out_axis, lsin)
    np.testing.assert_allclose(out_ntor0, lsin)


def test_lambda_axis_closure_noop_when_ns_one():
    lsin = np.array([[1.0, 2.0, 3.0]], dtype=float)
    m_modes = np.array([0, 0, 1], dtype=int)
    n_modes = np.array([0, 2, 1], dtype=int)

    out = np.asarray(
        _apply_vmec_lambda_axis_closure(
            Lsin=lsin,
            m_modes=m_modes,
            n_modes=n_modes,
            lthreed=True,
            ntor=2,
        )
    )
    np.testing.assert_allclose(out, lsin)


def test_circular_axisymmetric_bcovar_pure_toroidal_field_identities():
    static, state, wout = _circular_axisymmetric_case()
    bc = vmec_bcovar_half_mesh_from_wout(state=state, static=static, wout=wout)
    pshalf = np.asarray(_pshalf_from_s(static.s))[:, None, None]
    bsubu_recombined = np.asarray(bc.bsubu_parity_even) + pshalf * np.asarray(bc.bsubu_parity_odd)
    bsubv_recombined = np.asarray(bc.bsubv_parity_even) + pshalf * np.asarray(bc.bsubv_parity_odd)

    np.testing.assert_allclose(np.asarray(bc.guv), 0.0, rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(np.asarray(bc.bsupu), 0.0, rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(np.asarray(bc.bsubu), 0.0, rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(np.asarray(bc.bsubu), bsubu_recombined, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(np.asarray(bc.bsubv), bsubv_recombined, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(np.asarray(bc.bsupu * bc.bsubu + bc.bsupv * bc.bsubv)[1:], 4.0, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(np.asarray(bc.bsq)[0], 0.0, rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(np.asarray(bc.bsq)[1:], 2.0, rtol=1.0e-13, atol=1.0e-13)


def test_current_driven_branch_recomputes_chips_to_match_target_current():
    static, state, wout = _circular_axisymmetric_case()
    target_icurv = np.array([0.0, 0.25, 0.5, 0.75])
    wout_current = SimpleNamespace(**{**wout.__dict__, "ncurr": 1, "lcurrent": True, "icurv": target_icurv})

    bc = vmec_bcovar_half_mesh_from_wout(state=state, static=static, wout=wout_current)

    pwint = vmec_pwint_from_trig(static.trig_vmec, ns=static.s.size, nzeta=bc.bsupu.shape[2]).astype(float)
    if pwint.shape[1:] != bc.bsupu.shape[1:]:
        pwint = np.broadcast_to(float(static.trig_vmec.dnorm3), bc.bsupu.shape).copy()
        pwint[0] = 0.0
    surface_current = np.sum(pwint * np.asarray(bc.bsubu), axis=(1, 2))

    np.testing.assert_allclose(surface_current[0], 0.0, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(surface_current[1:], target_icurv[1:], rtol=1.0e-12, atol=1.0e-12)


def test_pressure_override_and_mass_reconstruction_control_bsq_without_changing_bfield():
    static, state, wout = _circular_axisymmetric_case()
    pressure = np.array([0.0, 0.125, 0.25, 0.5])

    bc_zero = vmec_bcovar_half_mesh_from_wout(state=state, static=static, wout=wout)
    bc_pres = vmec_bcovar_half_mesh_from_wout(state=state, static=static, wout=wout, pres=pressure)

    np.testing.assert_allclose(np.asarray(bc_pres.bsupu), np.asarray(bc_zero.bsupu), rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(bc_pres.bsupv), np.asarray(bc_zero.bsupv), rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        np.asarray(bc_pres.bsq - bc_zero.bsq),
        np.broadcast_to(pressure[:, None, None], bc_pres.bsq.shape),
        rtol=1.0e-13,
        atol=1.0e-13,
    )

    gamma = 2.0
    mass = np.array([0.0, 2.0, 3.0, 4.0])
    wout_mass = SimpleNamespace(**{**wout.__dict__, "gamma": gamma, "mass": mass, "pres": 100.0 + pressure})
    bc_zero_vmec = vmec_bcovar_half_mesh_from_wout(
        state=state,
        static=static,
        wout=wout,
        use_vmec_synthesis=True,
    )
    bc_mass = vmec_bcovar_half_mesh_from_wout(
        state=state,
        static=static,
        wout=wout_mass,
        pres=pressure,
        use_vmec_synthesis=True,
    )

    pwint = vmec_pwint_from_trig(static.trig_vmec, ns=static.s.size, nzeta=bc_mass.jac.sqrtg.shape[2])
    vp = np.sum(np.asarray(pwint) * np.asarray(bc_mass.jac.sqrtg), axis=(1, 2))
    expected_pressure = np.zeros_like(mass)
    nonzero_vp = vp != 0.0
    expected_pressure[nonzero_vp] = mass[nonzero_vp] / (vp[nonzero_vp] ** gamma)
    np.testing.assert_allclose(
        np.asarray(bc_mass.bsq - bc_zero_vmec.bsq),
        np.broadcast_to(expected_pressure[:, None, None], bc_mass.bsq.shape),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_lambda_full_mesh_branch_averages_half_mesh_and_scales_interior_only():
    static, state, wout = _circular_axisymmetric_case()

    bc = vmec_bcovar_half_mesh_from_wout(state=state, static=static, wout=wout)

    expected_bsubu_e = np.zeros_like(np.asarray(bc.bsubu))
    expected_bsubu_e[:-1] = 0.5 * (np.asarray(bc.bsubu[:-1]) + np.asarray(bc.bsubu[1:]))
    expected_bsubu_e[-1] = 0.5 * np.asarray(bc.bsubu[-1])
    np.testing.assert_allclose(np.asarray(bc.bsubu_e), expected_bsubu_e, rtol=1.0e-13, atol=1.0e-13)

    np.testing.assert_allclose(np.asarray(bc.clmn_even[0]), np.asarray(bc.bsubu_e[0]), rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        np.asarray(bc.clmn_even[1:]),
        -np.asarray(bc.lamscale) * np.asarray(bc.bsubu_e[1:]),
        rtol=1.0e-13,
        atol=1.0e-13,
    )

    bdamp = (0.1 * (1.0 - np.asarray(static.s)))[:, None, None]
    expected_bsubv_avg = np.zeros_like(np.asarray(bc.bsubv))
    expected_bsubv_avg[:-1] = 0.5 * (np.asarray(bc.bsubv[:-1]) + np.asarray(bc.bsubv[1:]))
    expected_bsubv_avg[-1] = 0.5 * np.asarray(bc.bsubv[-1])
    expected_bsubv_e = bdamp * np.asarray(bc.bsubv_preblend) + (1.0 - bdamp) * expected_bsubv_avg
    np.testing.assert_allclose(np.asarray(bc.bsubv_avg), expected_bsubv_avg, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(np.asarray(bc.bsubv_e), expected_bsubv_e, rtol=1.0e-13, atol=1.0e-13)


def test_wout_nyquist_reference_branches_override_bsup_bsub_and_bmag():
    static, state, wout = _circular_axisymmetric_case()
    ns = static.s.size
    shape = (ns, 1)
    bsupu_ref = np.arange(ns, dtype=float)[:, None]
    bsupv_ref = (10.0 + np.arange(ns, dtype=float))[:, None]
    bsubu_ref = (20.0 + np.arange(ns, dtype=float))[:, None]
    bsubv_ref = (30.0 + np.arange(ns, dtype=float))[:, None]
    bmag_ref = (40.0 + np.arange(ns, dtype=float))[:, None]
    bmag_ref[0] = 0.0
    wout_ref = SimpleNamespace(
        **{
            **wout.__dict__,
            "xm_nyq": np.array([0]),
            "xn_nyq": np.array([0]),
            "bsupumnc": bsupu_ref,
            "bsupumns": np.zeros(shape),
            "bsupvmnc": bsupv_ref,
            "bsupvmns": np.zeros(shape),
            "bsubumnc": bsubu_ref,
            "bsubumns": np.zeros(shape),
            "bsubvmnc": bsubv_ref,
            "bsubvmns": np.zeros(shape),
            "bmnc": bmag_ref,
            "bmns": np.zeros(shape),
        }
    )

    bc = vmec_bcovar_half_mesh_from_wout(
        state=state,
        static=static,
        wout=wout_ref,
        use_wout_bsup=True,
        use_wout_bsub_for_lambda=True,
        use_wout_bmag_for_bsq=True,
    )

    expected_bsupu = np.broadcast_to(bsupu_ref[:, :, None], bc.bsupu.shape).copy()
    expected_bsupv = np.broadcast_to(bsupv_ref[:, :, None], bc.bsupv.shape).copy()
    expected_bsupu[0] = 0.0
    expected_bsupv[0] = 0.0
    np.testing.assert_allclose(np.asarray(bc.bsupu), expected_bsupu, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(bc.bsupv), expected_bsupv, rtol=0.0, atol=1.0e-14)

    bsubu_grid = np.broadcast_to(bsubu_ref[:, :, None], bc.bsubu_e.shape)
    bsubv_grid = np.broadcast_to(bsubv_ref[:, :, None], bc.bsubv_e.shape)
    expected_bsubu_e = np.zeros_like(np.asarray(bc.bsubu_e))
    expected_bsubv_e = np.zeros_like(np.asarray(bc.bsubv_e))
    expected_bsubu_e[:-1] = 0.5 * (bsubu_grid[:-1] + bsubu_grid[1:])
    expected_bsubu_e[-1] = 0.5 * bsubu_grid[-1]
    expected_bsubv_e[:-1] = 0.5 * (bsubv_grid[:-1] + bsubv_grid[1:])
    expected_bsubv_e[-1] = 0.5 * bsubv_grid[-1]
    np.testing.assert_allclose(np.asarray(bc.bsubu_e), expected_bsubu_e, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(bc.bsubv_e), expected_bsubv_e, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(np.asarray(bc.bsubv_avg), expected_bsubv_e, rtol=0.0, atol=1.0e-14)

    expected_bsq = 0.5 * np.broadcast_to((bmag_ref * bmag_ref)[:, :, None], bc.bsq.shape)
    np.testing.assert_allclose(np.asarray(bc.bsq), expected_bsq, rtol=0.0, atol=1.0e-14)
