from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.solve import (
    _can_reassemble_precond_mats,
    _finite_float_or_zero,
    _format_axis_coeff,
    _format_checkpoint_log_row,
    _format_evolve_trace_row,
    _format_freeb_control_trace_row,
    _format_time_control_log_row,
    _format_time_control_trace_row,
    _format_vmec2000_iter_row,
    _free_boundary_iter_controls,
    _grad_rms_state,
    _half_mesh_from_full_mesh,
    _jit_cache_get,
    _jit_cache_limit,
    _jit_cache_put,
    _mask_grad_for_constraints,
    _materialize_adjoint_trace_array,
    _merge_axis_reset_state,
    _metric_surface_precond_scales_jax,
    _metric_surface_precond_scales_np,
    _normalize_adjoint_trace_mode,
    _normalize_resume_state_mode,
    _pack_resume_state_record,
    _pshalf_from_s_jax,
    _pshalf_from_s_np,
    _radial_tridi_smooth_dirichlet,
    _replace_mode_slice,
    _replace_mode_slice_np,
    _resolve_cg_tol,
    _resolve_grad_tol,
    _resolve_lbfgs_curvature_tol,
    _resolve_lm_damping,
    _s_half_from_full_mesh_s,
    _scale_mode_slice,
    _scale_mode_slice_np,
    _sm_sp_from_s_np,
    _update_state_gd,
    _should_print_vmec2000_row,
    _vmec_force_flux_profiles,
    _vmec_scale_m1_factors_from_mats,
    _vmec2000_cadence_selected,
    _zero_coeff_column,
    _zero_coeff_column_np,
    first_step_diagnostics,
)
from vmec_jax.vmec_tomnsp import TomnspsRZL
from vmec_jax.state import StateLayout, VMECState


def _state_from_value(value: float, *, ns: int = 3, k: int = 3) -> VMECState:
    layout = StateLayout(ns=ns, K=k, lasym=False)
    arr = np.full((ns, k), float(value), dtype=float)
    return VMECState(
        layout=layout,
        Rcos=arr.copy(),
        Rsin=arr.copy(),
        Zcos=arr.copy(),
        Zsin=arr.copy(),
        Lcos=arr.copy(),
        Lsin=arr.copy(),
    )


def test_jit_cache_limit_put_and_lru_policy(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_TEST_CACHE", "-4")
    assert _jit_cache_limit("VMEC_JAX_TEST_CACHE", 3) == 0

    monkeypatch.setenv("VMEC_JAX_TEST_CACHE", "not-an-int")
    assert _jit_cache_limit("VMEC_JAX_TEST_CACHE", 3) == 3

    cache: OrderedDict[tuple, object] = OrderedDict()
    monkeypatch.setenv("VMEC_JAX_TEST_CACHE", "0")
    value = object()
    assert _jit_cache_put(cache, ("disabled",), value, env_name="VMEC_JAX_TEST_CACHE", default=2) is value
    assert cache == {}

    monkeypatch.setenv("VMEC_JAX_TEST_CACHE", "2")
    _jit_cache_put(cache, ("a",), "A", env_name="VMEC_JAX_TEST_CACHE", default=2)
    _jit_cache_put(cache, ("b",), "B", env_name="VMEC_JAX_TEST_CACHE", default=2)
    assert _jit_cache_get(cache, ("a",)) == "A"
    _jit_cache_put(cache, ("c",), "C", env_name="VMEC_JAX_TEST_CACHE", default=2)
    assert list(cache.keys()) == [("a",), ("c",)]
    assert _jit_cache_get(cache, ("missing",)) is None


def test_mode_slice_helpers_cover_invalid_none_and_singleton_branches():
    arr = np.arange(6, dtype=float).reshape(2, 3)
    np.testing.assert_allclose(np.asarray(_zero_coeff_column(arr, idx=-1)), arr)
    np.testing.assert_allclose(np.asarray(_zero_coeff_column(arr, idx=3)), arr)
    np.testing.assert_allclose(np.asarray(_zero_coeff_column(np.ones((2, 1)), idx=0)), np.zeros((2, 1)))
    np.testing.assert_allclose(_zero_coeff_column_np(arr, idx=1), np.array([[0.0, 0.0, 2.0], [3.0, 0.0, 5.0]]))

    cube = np.arange(2 * 3 * 2, dtype=float).reshape(2, 3, 2)
    repl = np.full((2, 2), -5.0)
    assert _replace_mode_slice(None, mode_idx=0, replacement=repl) is None
    assert _scale_mode_slice(None, mode_idx=0, scale=np.ones(2)) is None
    np.testing.assert_allclose(np.asarray(_replace_mode_slice(cube, mode_idx=9, replacement=repl)), cube)
    np.testing.assert_allclose(np.asarray(_scale_mode_slice(cube, mode_idx=-1, scale=np.ones(2))), cube)

    one_mode = cube[:, :1, :]
    np.testing.assert_allclose(np.asarray(_replace_mode_slice(one_mode, mode_idx=0, replacement=repl)), repl[:, None, :])
    np.testing.assert_allclose(_replace_mode_slice_np(one_mode, mode_idx=0, replacement=repl), repl[:, None, :])
    np.testing.assert_allclose(_scale_mode_slice_np(one_mode, mode_idx=0, scale=np.array([2.0, 3.0])), one_mode * np.array([2.0, 3.0])[:, None, None])


def test_state_update_mask_and_rms_helpers_are_componentwise():
    pytest.importorskip("jax")

    state = _state_from_value(2.0)
    grad = _state_from_value(1.0)
    updated = _update_state_gd(state, grad, step=0.25, scale_rz=2.0, scale_l=4.0)
    for field in ("Rcos", "Rsin", "Zcos", "Zsin"):
        np.testing.assert_allclose(np.asarray(getattr(updated, field)), 1.5)
    for field in ("Lcos", "Lsin"):
        np.testing.assert_allclose(np.asarray(getattr(updated, field)), 1.0)

    assert _grad_rms_state(grad) == pytest.approx(np.sqrt(6.0))

    static = SimpleNamespace(modes=SimpleNamespace(m=np.array([0, 1, 2])))
    masked = _mask_grad_for_constraints(grad, static, idx00=0, mask_lambda_axis=False)
    for field in ("Rcos", "Rsin", "Zcos", "Zsin"):
        got = np.asarray(getattr(masked, field))
        np.testing.assert_allclose(got[-1, :], 0.0)
        np.testing.assert_allclose(got[0, :], np.array([1.0, 0.0, 0.0]))
    for field in ("Lcos", "Lsin"):
        got = np.asarray(getattr(masked, field))
        np.testing.assert_allclose(got[:, 0], 0.0)
        np.testing.assert_allclose(got[0, 1:], 1.0)


def test_tolerance_resolvers_validate_explicit_values_and_scale_by_dtype():
    assert _resolve_grad_tol(0.0, grad_rms0=10.0, dtype=np.float64) == 0.0
    with pytest.raises(ValueError, match="grad_tol"):
        _resolve_grad_tol(-1.0, grad_rms0=10.0, dtype=np.float64)
    assert _resolve_grad_tol(None, grad_rms0=4.0, dtype=np.float32) == pytest.approx(
        np.sqrt(np.finfo(np.float32).eps) * 4.0
    )

    with pytest.raises(ValueError, match="cg_tol"):
        _resolve_cg_tol(0.0, current_obj=1.0, initial_obj=1.0, target_obj=0.0, dtype=np.float64)
    assert _resolve_cg_tol(None, current_obj=1.0, initial_obj=3.0, target_obj=0.0, dtype=np.float64) == pytest.approx(0.25)

    with pytest.raises(ValueError, match="damping"):
        _resolve_lm_damping(-1.0, curvature_scale=2.0, dtype=np.float64)
    assert _resolve_lm_damping(None, curvature_scale=2.0, dtype=np.float64) == pytest.approx(
        np.sqrt(np.finfo(np.float64).eps) * 2.0
    )

    assert _resolve_lbfgs_curvature_tol(np.array([3.0, 4.0]), np.array([0.0, 6.0])) == pytest.approx(
        np.finfo(float).eps * 30.0
    )


def test_mesh_flux_and_free_boundary_cadence_helpers():
    np.testing.assert_allclose(np.asarray(_s_half_from_full_mesh_s(np.array([0.0]))), np.array([0.0]))
    np.testing.assert_allclose(
        np.asarray(_s_half_from_full_mesh_s(np.array([0.0, 0.25, 1.0]))),
        np.array([0.0, 0.125, 0.625]),
    )
    np.testing.assert_allclose(
        np.asarray(_half_mesh_from_full_mesh(np.array([2.0, 4.0, 10.0]))),
        np.array([2.0, 3.0, 7.0]),
    )

    phipf_internal, chipf_internal, chips_eff = _vmec_force_flux_profiles(
        phipf=np.array([2.0, 4.0]),
        chipf=None,
        signgs=1,
        flux_is_internal=True,
    )
    np.testing.assert_allclose(np.asarray(phipf_internal), np.array([2.0, 4.0]))
    assert chipf_internal is None
    np.testing.assert_allclose(np.asarray(chips_eff), np.zeros(2))

    phipf_external, _, chips_iota = _vmec_force_flux_profiles(
        phipf=np.array([2.0 * np.pi, 4.0 * np.pi]),
        chipf=None,
        signgs=1,
        flux_is_internal=False,
        iotaf=np.array([3.0, 5.0]),
    )
    np.testing.assert_allclose(np.asarray(phipf_external), np.array([1.0, 2.0]))
    np.testing.assert_allclose(np.asarray(chips_iota), np.array([3.0, 10.0]))

    assert _free_boundary_iter_controls(iter2=5, iter1=1, nvacskip=0) == (1, 0)
    assert _free_boundary_iter_controls(iter2=6, iter1=1, nvacskip=4) == (2, 1)


def test_resume_state_mode_and_payload_packing(monkeypatch):
    monkeypatch.setenv("VMEC_JAX_RESUME_STATE_MODE", "light")
    assert _normalize_resume_state_mode(None) == "minimal"
    assert _normalize_resume_state_mode(" compact ") == "minimal"
    assert _normalize_resume_state_mode("off") == "none"
    assert _normalize_resume_state_mode("") == "full"
    with pytest.raises(ValueError, match="resume_state_mode"):
        _normalize_resume_state_mode("huge")

    base = {"time_step": 0.1, "iter1": 3}
    heavy = {"cache": object(), "iter1": 9}
    assert _pack_resume_state_record(base=base, heavy=heavy, mode="minimal") == base
    assert _pack_resume_state_record(base=base, heavy=heavy, mode="none") is None

    full = _pack_resume_state_record(base=base, heavy=heavy, mode="full")
    assert full is not None
    assert full["time_step"] == 0.1
    assert full["iter1"] == 9
    assert "cache" in full
    assert base["iter1"] == 3


def test_vmec2000_cadence_and_row_formatting_helpers():
    assert _vmec2000_cadence_selected(iter_idx=1, max_iter=20, nstep_screen=7)
    assert _vmec2000_cadence_selected(iter_idx=20, max_iter=20, nstep_screen=7)
    assert _vmec2000_cadence_selected(iter_idx=14, max_iter=20, nstep_screen=7)
    assert not _vmec2000_cadence_selected(iter_idx=13, max_iter=20, nstep_screen=7)
    assert not _should_print_vmec2000_row(
        iter_idx=1,
        max_iter=20,
        nstep_screen=7,
        verbose=False,
        vmec2000_control=True,
        verbose_vmec2000_table=True,
    )

    row = _format_vmec2000_iter_row(
        iter_idx=12,
        fsqr=1.25,
        fsqz=2.5,
        fsql=3.75,
        delt0r=0.125,
        r00=4.5,
        w_mhd=6.25,
        lasym=False,
    )
    assert row == "   12  1.25E+00  2.50E+00  3.75E+00  4.500E+00  1.25E-01  6.2500E+00"

    row_lasym = _format_vmec2000_iter_row(
        iter_idx=2,
        fsqr=1.0,
        fsqz=2.0,
        fsql=3.0,
        delt0r=4.0,
        r00=5.0,
        z00=None,
        w_mhd=6.0,
        lasym=True,
    )
    assert "NAN" in row_lasym
    assert row_lasym.startswith("    2  1.00E+00")


def test_trace_formatting_and_scalar_guard_helpers():
    assert _format_axis_coeff(1.0e-5) == "1E-05"
    assert _finite_float_or_zero(3.5) == 3.5
    assert _finite_float_or_zero(np.asarray(np.nan)) == 0.0
    assert _finite_float_or_zero(np.asarray(np.inf)) == 0.0

    assert _format_time_control_log_row(
        iter_idx=4,
        fsq=1.0,
        fsq0=2.0,
        res0=3.0,
        res1=4.0,
        time_step=0.5,
    ) == "iter=4 fsq=1.000000e+00 fsq0=2.000000e+00 res0=3.000000e+00 res1=4.000000e+00 time_step=5.000000e-01\n"

    trace_row = _format_time_control_trace_row(
        stage="restart",
        iter2=9,
        iter1=3,
        fsq=1.0,
        fsq0=2.0,
        res0=3.0,
        res1=4.0,
        time_step=0.5,
        irst=2,
    )
    assert trace_row.endswith("   2 restart\n")
    assert " 1.0000000000000000e+00" in trace_row

    assert _format_checkpoint_log_row(iter_idx=8, fsq=1.0, fsq0=2.0, res0=3.0, res1=4.0).startswith("iter=8")
    assert _format_freeb_control_trace_row(
        iter2=2,
        iter1=1,
        ivac=3,
        ivacskip=0,
        nvacskip=5,
        fsq_rz_prev=0.25,
        cached=True,
    ).endswith(" 1\n")
    assert _format_evolve_trace_row(
        iter2=2,
        iter1=1,
        ns=3,
        stage="pre",
        fsq1=1.0,
        fsq_prev=2.0,
        time_step=0.5,
        dtau=0.25,
        b1=0.75,
        fac=0.8,
        xc_norm=10.0,
        v_norm=11.0,
        g_norm=12.0,
    ).startswith("       2        1        3 pre")


def test_radial_tridi_smoothing_matches_dense_dirichlet_reference():
    pytest.importorskip("jax")

    rhs = np.array(
        [
            [1.0, 2.0],
            [4.0, 8.0],
            [9.0, 18.0],
            [16.0, 32.0],
        ]
    )
    alpha = 0.25
    system = np.array(
        [
            [1.0 + 2.0 * alpha, -alpha],
            [-alpha, 1.0 + 2.0 * alpha],
        ]
    )
    interior_rhs = rhs[1:-1].copy()
    interior_rhs[0] += alpha * rhs[0]
    interior_rhs[-1] += alpha * rhs[-1]
    expected = rhs.copy()
    expected[1:-1] = np.linalg.solve(system, interior_rhs)

    smoothed = _radial_tridi_smooth_dirichlet(rhs, alpha=alpha)
    np.testing.assert_allclose(np.asarray(smoothed), expected)

    rhs3 = rhs.reshape(4, 2, 1)
    smoothed3 = _radial_tridi_smooth_dirichlet(rhs3, alpha=alpha)
    np.testing.assert_allclose(np.asarray(smoothed3), expected.reshape(4, 2, 1))

    assert _radial_tridi_smooth_dirichlet(rhs, alpha=0.0, skip_nonpositive=True) is rhs
    with pytest.raises(ValueError, match="ndim>=2"):
        _radial_tridi_smooth_dirichlet(np.arange(3.0), alpha=alpha)
    with pytest.raises(ValueError, match="expected \\(ns,K\\) or \\(ns,M,N\\)"):
        _radial_tridi_smooth_dirichlet(np.zeros((3, 1, 1, 1)), alpha=alpha)


def test_first_step_metric_mesh_and_adjoint_trace_helper_branches():
    pytest.importorskip("jax")

    guu = np.array([[[4.0], [1.0]], [[0.0], [0.0]], [[1.0e-12], [1.0e-12]]])
    r12 = np.ones_like(guu)
    bsubu = np.array([[[3.0], [4.0]], [[0.0], [0.0]], [[1.0e12], [1.0e12]]])
    bsubv = np.zeros_like(bsubu)
    w_ang = np.ones((2, 1))

    rz_np, l_np = _metric_surface_precond_scales_np(guu=guu, r12=r12, bsubu=bsubu, bsubv=bsubv, w_ang=w_ang)
    np.testing.assert_allclose(rz_np, np.array([1.0 / np.sqrt(5.0), 1.0, 100.0]))
    np.testing.assert_allclose(l_np, np.array([0.2, 1.0, 1.0e-4]))

    rz_jax, l_jax = _metric_surface_precond_scales_jax(guu=guu, r12=r12, bsubu=bsubu, bsubv=bsubv, w_ang=w_ang)
    np.testing.assert_allclose(np.asarray(rz_jax), rz_np)
    np.testing.assert_allclose(np.asarray(l_jax), l_np)

    s = np.array([0.0, 0.25, 1.0])
    np.testing.assert_allclose(_pshalf_from_s_np(s), np.sqrt(np.array([0.125, 0.125, 0.625])))
    np.testing.assert_allclose(np.asarray(_pshalf_from_s_jax(s, np.float64)), _pshalf_from_s_np(s))
    sm, sp = _sm_sp_from_s_np(s)
    assert sm.shape == (4,)
    assert sp.shape == (4,)
    np.testing.assert_allclose(_sm_sp_from_s_np(np.array([0.0]))[0], np.zeros(2))

    arr = np.array([1.0, 2.0])
    assert _normalize_adjoint_trace_mode(" dynamic ") == "dynamic"
    assert _materialize_adjoint_trace_array(arr, mode="dynamic") is arr
    np.testing.assert_allclose(_materialize_adjoint_trace_array([1.0, 2.0], mode="full"), arr)
    with pytest.raises(ValueError, match="adjoint_trace_mode"):
        _normalize_adjoint_trace_mode("summary")


def test_axis_reset_state_merge_replaces_only_m0_geometry_and_preserves_lambda():
    state = _state_from_value(1.0, ns=2, k=3)
    axis_state = _state_from_value(9.0, ns=2, k=3)
    state = VMECState(
        layout=state.layout,
        Rcos=state.Rcos,
        Rsin=state.Rsin + 1.0,
        Zcos=state.Zcos + 2.0,
        Zsin=state.Zsin + 3.0,
        Lcos=state.Lcos + 4.0,
        Lsin=state.Lsin + 5.0,
    )
    axis_state = VMECState(
        layout=axis_state.layout,
        Rcos=axis_state.Rcos,
        Rsin=axis_state.Rsin + 1.0,
        Zcos=axis_state.Zcos + 2.0,
        Zsin=axis_state.Zsin + 3.0,
        Lcos=axis_state.Lcos + 4.0,
        Lsin=axis_state.Lsin + 5.0,
    )
    static = SimpleNamespace(modes=SimpleNamespace(m=np.array([0, 1, 0])))

    merged = _merge_axis_reset_state(st=state, st_axis=axis_state, static=static, full_reset=False)
    np.testing.assert_allclose(np.asarray(merged.Rcos), np.array([[9.0, 1.0, 9.0], [9.0, 1.0, 9.0]]))
    np.testing.assert_allclose(np.asarray(merged.Rsin), np.array([[10.0, 2.0, 10.0], [10.0, 2.0, 10.0]]))
    np.testing.assert_allclose(np.asarray(merged.Zcos), np.array([[11.0, 3.0, 11.0], [11.0, 3.0, 11.0]]))
    np.testing.assert_allclose(np.asarray(merged.Zsin), np.array([[12.0, 4.0, 12.0], [12.0, 4.0, 12.0]]))
    np.testing.assert_allclose(np.asarray(merged.Lcos), np.asarray(state.Lcos))
    np.testing.assert_allclose(np.asarray(merged.Lsin), np.asarray(state.Lsin))

    full = _merge_axis_reset_state(st=state, st_axis=axis_state, static=static, full_reset=True)
    assert full is axis_state


def test_vmec_scale_m1_factors_jax_and_reassembly_contract():
    pytest.importorskip("jax")

    parity_mats = {
        "ard_parity": np.array([[1.0, 2.0], [1.0, 0.0]]),
        "brd_parity": np.array([[1.0, 4.0], [1.0, 0.0]]),
        "azd_parity": np.array([[1.0, 6.0], [1.0, 0.0]]),
        "bzd_parity": np.array([[1.0, 8.0], [1.0, 0.0]]),
    }
    fac_r, fac_z = _vmec_scale_m1_factors_from_mats(parity_mats)
    np.testing.assert_allclose(np.asarray(fac_r), np.array([6.0 / 20.0, 1.0]))
    np.testing.assert_allclose(np.asarray(fac_z), np.array([14.0 / 20.0, 1.0]))

    fallback_mats = {
        "dr": -np.array([[[0.0], [3.0]], [[0.0], [0.0]]]),
        "dz": -np.array([[[0.0], [9.0]], [[0.0], [0.0]]]),
    }
    fac_r, fac_z = _vmec_scale_m1_factors_from_mats(fallback_mats)
    np.testing.assert_allclose(np.asarray(fac_r), np.array([0.25, 1.0]))
    np.testing.assert_allclose(np.asarray(fac_z), np.array([0.75, 1.0]))

    assert not _can_reassemble_precond_mats(None)
    complete = {key: object() for key in (
        "arm_parity",
        "ard_parity",
        "brm_parity",
        "brd_parity",
        "azm_parity",
        "azd_parity",
        "bzm_parity",
        "bzd_parity",
        "cxd_full",
        "delta_s",
    )}
    assert _can_reassemble_precond_mats(complete)


def test_first_step_diagnostics_synthetic_default_and_axisymmetric_paths(monkeypatch):
    pytest.importorskip("jax")

    import vmec_jax.boundary as boundary_mod
    import vmec_jax.energy as energy_mod
    import vmec_jax.preconditioner_1d_jax as precond_mod
    import vmec_jax.solve as solve_mod
    import vmec_jax.static as static_mod
    import vmec_jax.vmec_forces as forces_mod
    import vmec_jax.vmec_residue as residue_mod
    import vmec_jax.vmec_tomnsp as tomnsp_mod

    s = np.array([0.0, 0.5, 1.0])
    modes = SimpleNamespace(m=np.array([0, 1]), n=np.array([0, 0]))
    shape = (3, 2, 1)
    ones = np.ones(shape)

    class DummyInData:
        scalars = {}
        indexed = {}

        def get_float(self, name, default=0.0):
            return {"DELT": 0.125, "TCON0": 1.75, "GAMMA": 0.0}.get(name, default)

        def get_bool(self, name, default=False):
            return {"LFORBAL": True, "LRFP": False}.get(name, default)

        def get_int(self, name, default=0):
            return {"NCURR": 0}.get(name, default)

    def make_static(*, lthreed: bool):
        cfg = SimpleNamespace(
            ns=3,
            mpol=2,
            ntor=0,
            nfp=1,
            ntheta=2,
            nzeta=1,
            lasym=False,
            lthreed=lthreed,
            lconm1=True,
        )
        return SimpleNamespace(cfg=cfg, s=s, modes=modes)

    def make_frzl(scale=1.0):
        return TomnspsRZL(
            frcc=scale * ones,
            frss=2.0 * scale * ones,
            fzsc=3.0 * scale * ones,
            fzcs=4.0 * scale * ones,
            flsc=5.0 * scale * ones,
            flcs=6.0 * scale * ones,
        )

    def fake_build_static(cfg, grid):
        return SimpleNamespace(cfg=cfg, s=s, modes=modes, trig_vmec=None, tomnsps_masks={"mask": True})

    bc = SimpleNamespace(
        guu=2.0 * ones,
        bsubu=ones,
        bsubv=2.0 * ones,
        jac=SimpleNamespace(r12=ones),
    )
    k = SimpleNamespace(bc=bc)

    monkeypatch.setattr(tomnsp_mod, "vmec_angle_grid", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(tomnsp_mod, "vmec_trig_tables", lambda **kwargs: SimpleNamespace(wint3_precond=np.ones((2, 1))))
    monkeypatch.setattr(static_mod, "build_static", fake_build_static)
    monkeypatch.setattr(
        energy_mod,
        "flux_profiles_from_indata",
        lambda indata, s, signgs: SimpleNamespace(
            chipf=np.array([0.0, 0.25, 0.5]),
            phips=np.array([9.0, 8.0, 7.0]),
            phipf=np.array([1.0, 1.5, 2.0]),
        ),
    )
    monkeypatch.setattr(boundary_mod, "boundary_from_indata", lambda indata, modes: SimpleNamespace(R_cos=np.array([2.0, 0.0])))
    monkeypatch.setattr(solve_mod, "_mass_half_mesh_from_indata", lambda **kwargs: np.array([0.0, 1.0, 2.0]))
    monkeypatch.setattr(solve_mod, "_pressure_half_mesh_from_indata", lambda **kwargs: np.array([0.0, 3.0, 4.0]))
    monkeypatch.setattr(solve_mod, "_icurv_full_mesh_from_indata", lambda **kwargs: np.array([0.0, 0.0, 0.0]))
    monkeypatch.setattr(
        solve_mod,
        "_vmec_force_flux_profiles",
        lambda **kwargs: (np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
    )
    monkeypatch.setattr(forces_mod, "vmec_forces_rz_from_wout", lambda **kwargs: k)

    def fake_residual_internal_from_kernels(*args, **kwargs):
        assert kwargs["apply_lforbal"] is True
        assert kwargs["masks"] == {"mask": True}
        return make_frzl()

    monkeypatch.setattr(forces_mod, "vmec_residual_internal_from_kernels", fake_residual_internal_from_kernels)
    monkeypatch.setattr(residue_mod, "vmec_apply_scalxc_to_tomnsps", lambda *, frzl, s: frzl)
    monkeypatch.setattr(residue_mod, "vmec_apply_m1_constraints", lambda *, frzl, lconm1: frzl)
    monkeypatch.setattr(residue_mod, "vmec_zero_m1_zforce", lambda *, frzl, enabled: frzl)
    monkeypatch.setattr(residue_mod, "vmec_gcx2_from_tomnsps", lambda **kwargs: (np.array(1.0), np.array(2.0), np.array(3.0)))
    monkeypatch.setattr(
        residue_mod,
        "vmec_force_norms_from_bcovar_dynamic",
        lambda **kwargs: SimpleNamespace(r1=np.array(2.0), fnorm=np.array(3.0), fnormL=np.array(4.0)),
    )
    monkeypatch.setattr(residue_mod, "vmec_rz_norm_from_state", lambda **kwargs: np.array(5.0))
    monkeypatch.setattr(residue_mod, "vmec_scalxc_from_s", lambda **kwargs: np.array([1.0, 2.0, 3.0]))
    monkeypatch.setattr(residue_mod, "vmec_wint_from_trig", lambda trig, nzeta: np.ones((2, nzeta)))
    monkeypatch.setattr(precond_mod, "lambda_preconditioner", lambda **kwargs: 2.0 * ones)
    monkeypatch.setattr(
        precond_mod,
        "rz_preconditioner",
        lambda **kwargs: TomnspsRZL(
            frcc=np.full(shape, np.nan),
            frss=ones,
            fzsc=ones,
            fzcs=ones,
            flsc=ones,
            flcs=ones,
        ),
    )

    state0 = _state_from_value(0.5, ns=3, k=2)
    indata = DummyInData()

    default_diag = first_step_diagnostics(
        state0,
        make_static(lthreed=True),
        indata=indata,
        signgs=-1,
        step_size=0.25,
        include_constraint_force=True,
        use_axisymmetric_preconditioner=False,
    )
    assert default_diag["fsqr"] == pytest.approx(6.0)
    assert default_diag["fsql"] == pytest.approx(12.0)
    assert default_diag["time_step"] == pytest.approx(0.25)
    assert default_diag["frcc_u"].shape == shape
    np.testing.assert_allclose(default_diag["rz_scale"], 0.5)

    axis_diag = first_step_diagnostics(
        state0,
        make_static(lthreed=False),
        indata=indata,
        signgs=1,
        step_size=None,
        include_constraint_force=False,
        use_axisymmetric_preconditioner=True,
    )
    assert axis_diag["time_step"] == pytest.approx(0.125)
    np.testing.assert_allclose(axis_diag["frcc_u"], np.array([[[1.0], [0.5]], [[1.0], [0.5]], [[0.0], [0.0]]]))
