from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import vmec_jax.solve as solve
from vmec_jax.solvers.fixed_boundary.profiles import build_wout_like_profiles_from_indata


class _InData:
    def __init__(self, *, ints=None, floats=None, bools=None):
        self._ints = dict(ints or {})
        self._floats = dict(floats or {})
        self._bools = dict(bools or {})

    def get_int(self, key, default=0):
        return self._ints.get(str(key).upper(), default)

    def get_float(self, key, default=0.0):
        return self._floats.get(str(key).upper(), default)

    def get_bool(self, key, default=False):
        return self._bools.get(str(key).upper(), default)


def test_radial_smoothing_metric_scales_and_scan_restart_payload_edges():
    rhs = np.arange(6.0).reshape(3, 2)
    assert solve._radial_tridi_smooth_dirichlet(rhs, alpha=0.0, skip_nonpositive=True) is rhs
    np.testing.assert_allclose(np.asarray(solve._radial_tridi_smooth_dirichlet(rhs[:2], alpha=1.0)), rhs[:2])
    smoothed = np.asarray(solve._radial_tridi_smooth_dirichlet(rhs, alpha=0.5))
    assert smoothed.shape == rhs.shape
    np.testing.assert_allclose(smoothed[0], rhs[0])
    np.testing.assert_allclose(smoothed[-1], rhs[-1])

    rhs3 = np.arange(12.0).reshape(3, 2, 2)
    assert np.asarray(solve._radial_tridi_smooth_dirichlet(rhs3, alpha=0.25)).shape == rhs3.shape
    rhs4 = np.arange(12.0).reshape(3, 2, 2, 1)
    assert np.asarray(solve._radial_tridi_smooth_dirichlet(rhs4, alpha=0.25)).shape == rhs4.shape
    with pytest.raises(ValueError, match="ndim>=2"):
        solve._radial_tridi_smooth_dirichlet(np.ones(3), alpha=1.0)
    with pytest.raises(ValueError, match="ndim>=2"):
        solve._radial_tridi_smooth_dirichlet(np.ones((3, 2, 2)), alpha=1.0, allow_3d=False)

    zeros = np.zeros((2, 2, 2))
    rz_np, l_np = solve._metric_surface_precond_scales_np(
        guu=zeros,
        r12=zeros,
        bsubu=zeros,
        bsubv=zeros,
        w_ang=np.ones((2, 2)),
    )
    np.testing.assert_allclose(rz_np, np.ones(2))
    np.testing.assert_allclose(l_np, np.ones(2))

    rz_jax, l_jax = solve._metric_surface_precond_scales_jax(
        guu=zeros + 4.0,
        r12=zeros + 2.0,
        bsubu=zeros + 3.0,
        bsubv=zeros + 4.0,
        w_ang=np.ones((2, 2)),
    )
    assert np.all(np.asarray(rz_jax) < 1.0)
    assert np.all(np.asarray(l_jax) < 1.0)

    blocks, valid = solve._mask_scan_restart_force_payload(
        force_blocks=(np.asarray([1.0, 2.0]), np.asarray([3.0, 4.0])),
        cache_valid=True,
        do_restart=False,
    )
    np.testing.assert_allclose(np.asarray(blocks[0]), [1.0, 2.0])
    assert bool(np.asarray(valid))
    blocks, valid = solve._mask_scan_restart_force_payload(
        force_blocks=(np.asarray([1.0, 2.0]),),
        cache_valid=True,
        do_restart=True,
    )
    np.testing.assert_allclose(np.asarray(blocks[0]), [0.0, 0.0])
    assert not bool(np.asarray(valid))


def test_force_profile_helpers_cover_external_flux_mass_current_and_tree_contract(monkeypatch):
    phipf_internal, chipf_internal, chips_eff = solve._vmec_force_flux_profiles(
        phipf=np.asarray([2.0 * np.pi, 4.0 * np.pi]),
        chipf=np.asarray([6.0 * np.pi, 8.0 * np.pi]),
        signgs=-1,
        flux_is_internal=False,
    )
    np.testing.assert_allclose(np.asarray(phipf_internal), [-1.0, -2.0])
    np.testing.assert_allclose(np.asarray(chipf_internal), [-3.0, -4.0])
    assert np.all(np.isfinite(np.asarray(chips_eff)))

    _, _, chips_from_iotas = solve._vmec_force_flux_profiles(
        phipf=np.asarray([2.0, 4.0]),
        chipf=None,
        signgs=1,
        flux_is_internal=True,
        iotas=np.asarray([0.25, 0.5]),
    )
    np.testing.assert_allclose(np.asarray(chips_from_iotas), [0.5, 2.0])

    import vmec_jax.profiles as profiles

    def fake_eval_profiles(_indata, s):
        s = np.asarray(s, dtype=float)
        return {"pressure": 1.0 + s, "current": 2.0 * s}

    monkeypatch.setattr(profiles, "eval_profiles", fake_eval_profiles)
    indata = _InData(
        ints={"NCURR": 1},
        floats={"CURTOR": 2.0, "GAMMA": 2.0},
        bools={"LRFP": True},
    )
    s_full = np.asarray([0.0, 0.5, 1.0])
    mass = solve._mass_half_mesh_from_indata(
        indata=indata,
        s_full=s_full,
        phips=np.asarray([0.0, 0.5, 1.0]),
        r00=2.0,
        gamma=2.0,
        lrfp=True,
        chips=np.asarray([0.0, 0.25, 0.75]),
    )
    assert float(np.asarray(mass)[0]) == 0.0
    assert float(np.asarray(mass)[-1]) > 0.0
    icurv = solve._icurv_full_mesh_from_indata(indata=indata, s_full=s_full, signgs=-1)
    assert float(np.asarray(icurv)[0]) == 0.0
    assert float(np.asarray(icurv)[-1]) < 0.0
    np.testing.assert_allclose(
        np.asarray(solve._icurv_full_mesh_from_indata(indata=_InData(ints={"NCURR": 0}), s_full=s_full, signgs=1)),
        np.zeros_like(s_full),
    )
    np.testing.assert_allclose(
        np.asarray(
            solve._icurv_full_mesh_from_indata(
                indata=_InData(ints={"NCURR": 1}, floats={"CURTOR": 0.0}),
                s_full=s_full,
                signgs=1,
            )
        ),
        np.zeros_like(s_full),
    )

    wout_like = solve._WoutLikeVmecForces(
        nfp=2,
        mpol=3,
        ntor=1,
        lasym=True,
        signgs=-1,
        phipf=np.asarray([1.0, 2.0]),
        phips=np.asarray([0.0, 1.0]),
        chipf=np.asarray([0.0, 0.5]),
        pres=np.asarray([0.0, 0.1]),
        mass=np.asarray([0.0, 0.2]),
        gamma=1.5,
        ncurr=1,
        lcurrent=False,
        icurv=np.asarray([0.0, 0.3]),
        flux_is_internal=False,
        phipf_internal=np.asarray([1.0, 2.0]),
        chipf_internal=np.asarray([0.0, 0.5]),
        chips_eff=np.asarray([0.0, 1.0]),
    )
    children, aux = wout_like.tree_flatten()
    rebuilt = solve._WoutLikeVmecForces.tree_unflatten(aux, children)
    assert rebuilt.nfp == 2
    assert rebuilt.lasym is True
    assert rebuilt.signgs == -1
    assert rebuilt.gamma == pytest.approx(1.5)
    np.testing.assert_allclose(np.asarray(rebuilt.chips_eff), [0.0, 1.0])


def test_preconditioner_capability_and_small_mesh_shape_helpers():
    assert not solve._can_reassemble_precond_mats(None)
    assert not solve._can_reassemble_precond_mats({"arm_parity": 1})
    complete = {
        "arm_parity": 1,
        "ard_parity": 1,
        "brm_parity": 1,
        "brd_parity": 1,
        "azm_parity": 1,
        "azd_parity": 1,
        "bzm_parity": 1,
        "bzd_parity": 1,
        "cxd_full": 1,
        "delta_s": 1,
    }
    assert solve._can_reassemble_precond_mats(complete)
    np.testing.assert_allclose(solve._pshalf_from_s_np(np.asarray([0.25])), [0.5])
    np.testing.assert_allclose(np.asarray(solve._pshalf_from_s_jax(np.asarray([0.25]), np.float64)), [0.5])
    sm, sp = solve._sm_sp_from_s_np(np.asarray([0.0]))
    np.testing.assert_allclose(sm, [0.0, 0.0])
    np.testing.assert_allclose(sp, [0.0, 0.0])


def test_residual_iter_precompile_setup_branches(load_case_circular_tokamak, monkeypatch):
    pytest.importorskip("jax")

    _cfg, indata, static, _boundary, state0 = load_case_circular_tokamak

    result = solve.solve_fixed_boundary_residual_iter(
        state0,
        static,
        indata=indata,
        signgs=1,
        max_iter=1,
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        jit_forces=False,
        use_scan=False,
        precompile_only=True,
        verbose=False,
        verbose_vmec2000_table=False,
    )

    assert result.state is state0
    assert result.diagnostics == {"precompile_only": True}
    assert result.w_history.shape == (0,)

    monkeypatch.setenv("VMEC_JAX_FREEB_SAMPLE_EXTERNAL", "0")
    freeb_static = SimpleNamespace(
        **{
            name: getattr(static, name)
            for name in (
                "cfg",
                "modes",
                "grid",
                "s",
            )
        }
    )
    freeb_static.cfg = SimpleNamespace(**vars(static.cfg), lfreeb=True, nvacskip=2)
    freeb_static.mgrid_metadata = None
    freeb_static.free_boundary_extcur = None
    freeb_static.trig_vmec = getattr(static, "trig_vmec", None)
    freeb_static.m_np = getattr(static, "m_np", None)
    freeb_static.n_np = getattr(static, "n_np", None)
    freeb_static.lambda_axis_copy_mask = getattr(static, "lambda_axis_copy_mask", None)
    freeb_static.tomnsps_masks = getattr(static, "tomnsps_masks", None)
    freeb_static.tomnsps_masks_edge = getattr(static, "tomnsps_masks_edge", None)
    freeb_static.signed_maps = getattr(static, "signed_maps", None)
    for name in (
        "m_is_m0",
        "m_is_even",
        "m_is_odd",
        "m_is_m1",
        "m_is_odd_rest",
        "mn_idx_m",
        "mn_idx_n",
        "mn_idx_kp",
        "mn_idx_kn",
        "mn_has_kn",
    ):
        if hasattr(static, name):
            setattr(freeb_static, name, getattr(static, name))

    freeb_result = solve.solve_fixed_boundary_residual_iter(
        state0,
        freeb_static,
        indata=indata,
        signgs=1,
        max_iter=1,
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        jit_forces=False,
        use_scan=True,
        precompile_only=True,
        verbose=False,
        verbose_vmec2000_table=False,
    )

    assert freeb_result.diagnostics == {"precompile_only": True}


def test_residual_iter_vmec2000_scan_minimal_one_step(load_case_circular_tokamak, monkeypatch):
    pytest.importorskip("jax")

    _cfg, indata, static, _boundary, state0 = load_case_circular_tokamak
    monkeypatch.setenv("VMEC_JAX_SCAN_PRINT", "0")
    monkeypatch.setenv("VMEC_JAX_SCAN_LIGHT", "0")
    monkeypatch.setenv("VMEC_JAX_SCAN_MINIMAL", "1")
    monkeypatch.setenv("VMEC_JAX_SCAN_PRECOND_PRECOMPUTE", "0")
    monkeypatch.setenv("VMEC_JAX_TIMING", "1")

    result = solve.solve_fixed_boundary_residual_iter(
        state0,
        static,
        indata=indata,
        signgs=1,
        max_iter=1,
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        strict_update=True,
        backtracking=False,
        jit_forces=False,
        use_scan=True,
        scan_minimal_default=True,
        verbose=False,
        verbose_vmec2000_table=False,
    )

    assert result.n_iter == 1
    assert result.w_history.shape == (1,)
    assert result.diagnostics["use_scan"] is True
    assert result.diagnostics["vmec2000_scan"] is True
    assert result.diagnostics["scan_minimal"] is True
    assert result.diagnostics["probe_count"] == 1
    assert np.isfinite(result.diagnostics["final_fsqr"])
    assert result.diagnostics["timing"]["scan_total_s"] >= 0.0


def test_residual_iter_vmec2000_scan_state_only(load_case_circular_tokamak, monkeypatch):
    pytest.importorskip("jax")

    _cfg, indata, static, _boundary, state0 = load_case_circular_tokamak
    monkeypatch.setenv("VMEC_JAX_SCAN_PRINT", "0")
    monkeypatch.setenv("VMEC_JAX_SCAN_MINIMAL", "1")

    result = solve.solve_fixed_boundary_residual_iter(
        state0,
        static,
        indata=indata,
        signgs=1,
        max_iter=1,
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        strict_update=True,
        backtracking=False,
        jit_forces=False,
        use_scan=True,
        state_only=True,
        scan_minimal_default=True,
        verbose=False,
        verbose_vmec2000_table=False,
    )

    assert result.n_iter == 1
    assert result.w_history.shape == (0,)
    assert result.diagnostics["state_only"] is True
    assert result.diagnostics["history_none"] is True
    assert result.diagnostics["vmec2000_scan"] is True


def test_wout_like_profile_setup_uses_real_input_profiles(load_case_circular_tokamak):
    _cfg, indata, static, _boundary, _state0 = load_case_circular_tokamak
    modes_m = np.asarray(static.modes.m, dtype=int)
    modes_n = np.asarray(static.modes.n, dtype=int)
    idx_candidates = np.nonzero((modes_m == 0) & (modes_n == 0))[0]
    idx00 = int(idx_candidates[0]) if idx_candidates.size else 0

    setup = build_wout_like_profiles_from_indata(
        indata=indata,
        static=static,
        s_profile=static.s,
        signgs=1,
        idx00=idx00,
        prefer_host_default_profiles=True,
        s_profile_has_tracer=False,
    )

    assert setup.wout_like.nfp == int(static.cfg.nfp)
    assert setup.wout_like.mpol == int(static.cfg.mpol)
    assert setup.wout_like.ntor == int(static.cfg.ntor)
    assert setup.wout_like.lasym is bool(static.cfg.lasym)
    assert setup.ncurr == int(indata.get_int("NCURR", 0))
    assert np.asarray(setup.phips).shape == np.asarray(static.s).shape
    assert float(np.asarray(setup.phips)[0]) == pytest.approx(0.0)
    assert np.asarray(setup.wout_like.mass).shape == np.asarray(static.s).shape
    assert np.asarray(setup.wout_like.chips_eff).shape == np.asarray(static.s).shape
