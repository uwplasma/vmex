from __future__ import annotations

import os

import numpy as np
import pytest


def _mode_index(modes, m: int, n: int) -> int:
    for k, (mm, nn) in enumerate(zip(np.asarray(modes.m), np.asarray(modes.n))):
        if int(mm) == int(m) and int(nn) == int(n):
            return k
    raise KeyError((m, n))


def _require_slow() -> None:
    if os.environ.get("RUN_SLOW", "") != "1":
        pytest.skip("Set RUN_SLOW=1 to run slow QH derivative checks")


def test_replay_tridi_policy_helpers_and_static_flags():
    import vmec_jax.discrete_adjoint as da

    assert da._tridi_policy_cache_value(None) == -1
    assert da._tridi_policy_cache_value(False) == 0
    assert da._tridi_policy_cache_value(True) == 1
    assert da._trace_preconditioner_use_precomputed_tridi({}) is None
    assert da._trace_preconditioner_use_precomputed_tridi({"preconditioner_use_precomputed_tridi": True}) is True
    assert da._trace_preconditioner_use_lax_tridi({}) is None
    assert da._trace_preconditioner_use_lax_tridi({"preconditioner_use_lax_tridi": True}) is True
    assert (
        da._trace_preconditioner_use_precomputed_tridi(
            {"preconditioner_use_precomputed_tridi": False},
            {"preconditioner_use_precomputed_tridi": True},
        )
        is True
    )
    assert (
        da._trace_preconditioner_use_lax_tridi(
            {"preconditioner_use_lax_tridi": False},
            {"preconditioner_use_lax_tridi": True},
        )
        is True
    )

    base_trace = {
        key: 1
        for key in (
            "apply_lforbal",
            "include_edge_residual",
            "apply_m1_constraints",
            "limit_update_rms",
            "limit_dt_from_force",
            "vmec2000_control",
            "divide_by_scalxc_for_update",
            "signgs",
        )
    }
    base_trace["precond_jmax"] = 4
    base_trace["preconditioner_use_precomputed_tridi"] = True
    base_trace["preconditioner_use_lax_tridi"] = False
    flags = da._static_flags_from_replay_step_traces((dict(base_trace), dict(base_trace)))
    assert flags["preconditioner_use_precomputed_tridi"] is True
    assert flags["preconditioner_use_lax_tridi"] is False
    bad = dict(base_trace, preconditioner_use_precomputed_tridi=False)
    with pytest.raises(ValueError, match="tridiagonal policy"):
        da._static_flags_from_replay_step_traces((base_trace, bad))
    bad_lax = dict(base_trace, preconditioner_use_lax_tridi=True)
    with pytest.raises(ValueError, match="lax tridiagonal policy"):
        da._static_flags_from_replay_step_traces((base_trace, bad_lax))


def test_qh_warm_start_fixture_loads_expected_case(load_case_qh_warm_start):
    _cfg, _indata, static, boundary, _state0 = load_case_qh_warm_start

    assert int(static.cfg.nfp) == 4
    assert not bool(static.cfg.lasym)
    assert _mode_index(static.modes, 0, 1) >= 0
    assert float(np.max(np.abs(np.asarray(boundary.R_cos)))) > 0.0
    assert float(np.max(np.abs(np.asarray(boundary.Z_sin)))) > 0.0


def test_qh_warm_start_implicit_aspect_directional_derivative_matches_fd(load_case_qh_warm_start):
    pytest.importorskip("jax")
    _require_slow()

    from vmec_jax._compat import enable_x64, jax, jnp
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.implicit import solve_fixed_boundary_state_implicit_vmec_residual
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.wout import equilibrium_aspect_ratio_from_state

    enable_x64(True)

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))

    k_rc01 = _mode_index(static.modes, 0, 1)
    edge_Rcos0 = np.asarray(boundary.R_cos, dtype=float)
    edge_Rsin0 = np.asarray(boundary.R_sin, dtype=float)
    edge_Zcos0 = np.asarray(boundary.Z_cos, dtype=float)
    edge_Zsin0 = np.asarray(boundary.Z_sin, dtype=float)
    alpha0 = float(edge_Rcos0[k_rc01])
    step_size = float(indata.get_float("DELT", 1.0))
    ftol = float(indata.get_float("FTOL", 1e-14))

    def _aspect_from_alpha(alpha):
        edge_Rcos = jnp.asarray(edge_Rcos0).at[k_rc01].set(alpha)
        state = solve_fixed_boundary_state_implicit_vmec_residual(
            state_guess,
            static,
            indata=indata,
            signgs=signgs,
            state0_host=state_guess,
            max_iter=1,
            step_size=step_size,
            ftol=ftol,
            edge_Rcos=edge_Rcos,
            edge_Rsin=jnp.asarray(edge_Rsin0),
            edge_Zcos=jnp.asarray(edge_Zcos0),
            edge_Zsin=jnp.asarray(edge_Zsin0),
        )
        return equilibrium_aspect_ratio_from_state(state=state, static=static)

    eps = 1.0e-5
    grad_ad = float(np.asarray(jax.grad(_aspect_from_alpha)(alpha0)))
    grad_fd = float(
        (
            np.asarray(_aspect_from_alpha(alpha0 + eps))
            - np.asarray(_aspect_from_alpha(alpha0 - eps))
        )
        / (2.0 * eps)
    )

    assert np.isfinite(grad_ad)
    assert np.isfinite(grad_fd)
    assert grad_ad == pytest.approx(grad_fd, rel=1.0e-1, abs=5.0e-4)


def test_qh_projected_initial_guess_boundary_derivative_matches_fd_with_frozen_axis(load_case_qh_warm_start):
    pytest.importorskip("jax")
    _require_slow()

    from vmec_jax._compat import enable_x64, jax, jnp
    from vmec_jax.boundary import BoundaryCoeffs
    from vmec_jax.init_guess import extract_axis_override_from_state, initial_guess_from_boundary

    enable_x64(True)

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    base_state = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    axis_override = extract_axis_override_from_state(base_state, static)
    k_rc01 = _mode_index(static.modes, 0, 1)
    alpha0 = float(np.asarray(boundary.R_cos, dtype=float)[k_rc01])

    def _rcos_mid(alpha):
        boundary_alpha = BoundaryCoeffs(
            R_cos=jnp.asarray(boundary.R_cos).at[k_rc01].set(alpha),
            R_sin=jnp.asarray(boundary.R_sin),
            Z_cos=jnp.asarray(boundary.Z_cos),
            Z_sin=jnp.asarray(boundary.Z_sin),
        )
        state = initial_guess_from_boundary(
            static,
            boundary_alpha,
            indata,
            vmec_project=True,
            axis_override=axis_override,
        )
        return state.Rcos[5, k_rc01]

    eps = 1.0e-5
    grad_ad = float(np.asarray(jax.grad(_rcos_mid)(alpha0)))
    grad_fd = float(
        (
            np.asarray(_rcos_mid(alpha0 + eps))
            - np.asarray(_rcos_mid(alpha0 - eps))
        )
        / (2.0 * eps)
    )

    assert np.isfinite(grad_ad)
    assert np.isfinite(grad_fd)
    assert grad_ad == pytest.approx(grad_fd, rel=1.0e-6, abs=1.0e-8)


def test_qh_projected_initial_guess_boundary_derivative_matches_fd_moving_axis(load_case_qh_warm_start):
    pytest.importorskip("jax")
    _require_slow()

    from vmec_jax._compat import enable_x64, jax, jnp
    from vmec_jax.boundary import BoundaryCoeffs
    from vmec_jax.init_guess import initial_guess_from_boundary

    enable_x64(True)

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    k_rc01 = _mode_index(static.modes, 0, 1)
    alpha0 = float(np.asarray(boundary.R_cos, dtype=float)[k_rc01])

    def _rcos_mid(alpha):
        boundary_alpha = BoundaryCoeffs(
            R_cos=jnp.asarray(boundary.R_cos).at[k_rc01].set(alpha),
            R_sin=jnp.asarray(boundary.R_sin),
            Z_cos=jnp.asarray(boundary.Z_cos),
            Z_sin=jnp.asarray(boundary.Z_sin),
        )
        state = initial_guess_from_boundary(
            static,
            boundary_alpha,
            indata,
            vmec_project=True,
        )
        return state.Rcos[5, k_rc01]

    eps = 1.0e-5
    grad_ad = float(np.asarray(jax.grad(_rcos_mid)(alpha0)))
    grad_fd = float(
        (
            np.asarray(_rcos_mid(alpha0 + eps))
            - np.asarray(_rcos_mid(alpha0 - eps))
        )
        / (2.0 * eps)
    )

    assert np.isfinite(grad_ad)
    assert np.isfinite(grad_fd)
    assert grad_ad == pytest.approx(grad_fd, rel=1.0e-6, abs=1.0e-8)


def test_residual_iteration_trace_extracts_qh_history(load_case_qh_warm_start):
    pytest.importorskip("jax")

    from vmec_jax.discrete_adjoint import residual_iteration_trace_from_result
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.solve import solve_fixed_boundary_residual_iter

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))

    result = solve_fixed_boundary_residual_iter(
        state_guess,
        static,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        max_iter=1,
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
    )

    trace = residual_iteration_trace_from_result(result)
    n = int(trace.iter2.shape[0])

    assert n >= 1
    assert trace.step_status.shape == (n,)
    assert trace.restart_reason.shape == (n,)
    assert trace.time_step.shape == (n,)
    assert trace.dt_eff.shape == (n,)
    assert trace.fsq_curr.shape == (n,)
    assert trace.state_advanced.shape == (n,)
    assert np.all(np.isfinite(trace.fsq_curr))


def test_residual_checkpoint_tape_matches_direct_one_step_qh(load_case_qh_warm_start):
    pytest.importorskip("jax")

    from vmec_jax.discrete_adjoint import build_residual_checkpoint_tape
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.solve import solve_fixed_boundary_residual_iter
    from vmec_jax.state import pack_state

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    common_kwargs = dict(
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        light_history=True,
        resume_state_mode="minimal",
    )

    direct = solve_fixed_boundary_residual_iter(
        state_guess,
        static,
        max_iter=1,
        **common_kwargs,
    )
    tape = build_residual_checkpoint_tape(
        state_guess,
        static,
        max_iter=1,
        solver_kwargs=common_kwargs,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
        light_history=True,
        resume_state_mode="minimal",
    )

    assert tape.packed_states.shape[0] == 1
    assert tape.trace.iter2.shape[0] >= 1
    assert tape.resume_states[0] is not None
    assert len(tape.step_traces) == 1
    assert np.asarray(tape.packed_states[-1]) == pytest.approx(
        np.asarray(pack_state(direct.state)),
        rel=0.0,
        abs=1.0e-12,
    )


def test_residual_checkpoint_tape_can_skip_debug_storage_qh(load_case_qh_warm_start):
    pytest.importorskip("jax")
    _require_slow()

    from vmec_jax.discrete_adjoint import build_residual_checkpoint_tape
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.solve import solve_fixed_boundary_residual_iter
    from vmec_jax.state import pack_state

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    common_kwargs = dict(
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        light_history=True,
        resume_state_mode="minimal",
    )

    direct = solve_fixed_boundary_residual_iter(
        state_guess,
        static,
        max_iter=1,
        **common_kwargs,
    )
    tape = build_residual_checkpoint_tape(
        state_guess,
        static,
        max_iter=1,
        solver_kwargs=common_kwargs,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
        light_history=True,
        resume_state_mode="minimal",
        store_packed_states=False,
        store_trace=False,
        store_resume_states=False,
    )

    assert tape.packed_states.shape[0] == 0
    assert tape.trace.iter2.shape[0] == 0
    assert tape.resume_states == ()
    assert len(tape.step_traces) == 1
    assert np.asarray(tape.final_packed_state) == pytest.approx(
        np.asarray(pack_state(direct.state)),
        rel=0.0,
        abs=1.0e-12,
    )


def test_residual_checkpoint_tape_direct_matches_two_step_qh(load_case_qh_warm_start):
    pytest.importorskip("jax")
    _require_slow()

    from vmec_jax.discrete_adjoint import (
        build_residual_checkpoint_tape,
        build_residual_checkpoint_tape_direct,
    )
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    common_kwargs = dict(
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        light_history=True,
        resume_state_mode="full",
    )

    replay_tape = build_residual_checkpoint_tape(
        state_guess,
        static,
        max_iter=2,
        solver_kwargs=common_kwargs,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
        light_history=True,
        resume_state_mode="full",
        store_packed_states=False,
        store_trace=False,
        store_resume_states=False,
    )
    direct_tape = build_residual_checkpoint_tape_direct(
        state_guess,
        static,
        max_iter=2,
        solver_kwargs=common_kwargs,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
        light_history=True,
        store_trace=False,
    )

    assert np.asarray(direct_tape.final_packed_state) == pytest.approx(
        np.asarray(replay_tape.final_packed_state),
        rel=0.0,
        abs=1.0e-12,
    )
    assert len(direct_tape.step_traces) == len(replay_tape.step_traces)


def test_residual_checkpoint_tape_direct_dynamic_only_matches_two_step_qh(load_case_qh_warm_start):
    pytest.importorskip("jax")
    _require_slow()

    from vmec_jax._compat import jnp
    from vmec_jax.discrete_adjoint import (
        build_residual_checkpoint_tape_direct,
        checkpoint_tape_state_jvp_columns,
    )
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    common_kwargs = dict(
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        light_history=True,
        resume_state_mode="full",
    )

    direct_full = build_residual_checkpoint_tape_direct(
        state_guess,
        static,
        max_iter=2,
        solver_kwargs=common_kwargs,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
        light_history=True,
        store_trace=False,
    )
    direct_lean = build_residual_checkpoint_tape_direct(
        state_guess,
        static,
        max_iter=2,
        solver_kwargs=common_kwargs,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
        light_history=True,
        store_trace=False,
        store_full_step_traces=False,
    )

    assert np.asarray(direct_lean.final_packed_state) == pytest.approx(
        np.asarray(direct_full.final_packed_state),
        rel=0.0,
        abs=1.0e-12,
    )
    assert direct_lean.step_traces == ()
    assert direct_lean.dynamic_initial_carry is not None

    tangent = jnp.asarray(np.eye(int(direct_full.final_packed_state.size), dtype=float)[:1])
    full_jvp = checkpoint_tape_state_jvp_columns(
        tape=direct_full,
        static=static,
        initial_tangents=tangent,
        rebuild_preconditioner=True,
    )
    lean_jvp = checkpoint_tape_state_jvp_columns(
        tape=direct_lean,
        static=static,
        initial_tangents=tangent,
        rebuild_preconditioner=True,
    )
    np.testing.assert_allclose(np.asarray(lean_jvp), np.asarray(full_jvp), rtol=1.0e-10, atol=1.0e-10)


def test_residual_checkpoint_tape_direct_buckets_dynamic_shapes_for_nearby_lengths(load_case_qh_warm_start):
    pytest.importorskip("jax")
    _require_slow()

    from vmec_jax._compat import jax
    from vmec_jax.discrete_adjoint import build_residual_checkpoint_tape_direct
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    common_kwargs = dict(
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        light_history=True,
        resume_state_mode="full",
    )

    tape2 = build_residual_checkpoint_tape_direct(
        state_guess,
        static,
        max_iter=2,
        solver_kwargs=common_kwargs,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
        light_history=True,
        store_trace=False,
        store_full_step_traces=False,
    )
    tape3 = build_residual_checkpoint_tape_direct(
        state_guess,
        static,
        max_iter=3,
        solver_kwargs=common_kwargs,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
        light_history=True,
        store_trace=False,
        store_full_step_traces=False,
    )

    assert tape2.stacked_step_traces is not None
    assert tape3.stacked_step_traces is not None
    assert tape2.dynamic_base_carries_stacked is not None
    assert tape3.dynamic_base_carries_stacked is not None

    trace2_leaves = jax.tree_util.tree_leaves(tape2.stacked_step_traces)
    trace3_leaves = jax.tree_util.tree_leaves(tape3.stacked_step_traces)
    carry2_leaves = jax.tree_util.tree_leaves(tape2.dynamic_base_carries_stacked)
    carry3_leaves = jax.tree_util.tree_leaves(tape3.dynamic_base_carries_stacked)

    assert trace2_leaves[0].shape[0] == trace3_leaves[0].shape[0]
    assert carry2_leaves[0].shape[0] == carry3_leaves[0].shape[0]
    assert trace2_leaves[0].shape[0] >= 3


def test_residual_checkpoint_tape_matches_direct_two_step_qh(load_case_qh_warm_start):
    pytest.importorskip("jax")
    _require_slow()

    from vmec_jax.discrete_adjoint import build_residual_checkpoint_tape
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.solve import solve_fixed_boundary_residual_iter
    from vmec_jax.state import pack_state

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    common_kwargs = dict(
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        light_history=False,
        resume_state_mode="minimal",
    )

    direct = solve_fixed_boundary_residual_iter(
        state_guess,
        static,
        max_iter=2,
        **common_kwargs,
    )
    tape = build_residual_checkpoint_tape(
        state_guess,
        static,
        max_iter=2,
        solver_kwargs=common_kwargs,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
        light_history=False,
        resume_state_mode="minimal",
    )

    assert tape.packed_states.shape[0] == 2
    assert tape.trace.iter2.shape[0] >= 2
    assert tape.resume_states[-1] is not None
    assert len(tape.step_traces) == 2
    assert np.asarray(tape.packed_states[-1]) == pytest.approx(
        np.asarray(pack_state(direct.state)),
        rel=0.0,
        abs=1.0e-12,
    )


def test_checkpoint_tape_state_vjp_matches_direct_two_step_qh(load_case_qh_warm_start):
    pytest.importorskip("jax")
    _require_slow()

    from vmec_jax._compat import enable_x64, jax, jnp
    from vmec_jax.discrete_adjoint import build_residual_checkpoint_tape, checkpoint_tape_state_vjp
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.state import pack_state, unpack_state

    enable_x64(True)

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    common_kwargs = dict(
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        light_history=True,
        resume_state_mode="full",
    )
    tape = build_residual_checkpoint_tape(
        state_guess,
        static,
        max_iter=2,
        solver_kwargs=common_kwargs,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
        light_history=True,
        resume_state_mode="full",
    )
    assert len(tape.step_traces) == 2

    x0 = jnp.asarray(pack_state(tape.step_traces[0]["state_pre"]))
    cotangent = jnp.linspace(-0.15, 0.15, int(x0.size), dtype=x0.dtype)

    def _forward_two_step(x):
        from vmec_jax.discrete_adjoint import strict_update_one_step_from_state

        state = unpack_state(x, tape.step_traces[0]["state_pre"].layout)
        for trace in tape.step_traces:
            out = strict_update_one_step_from_state(
                state,
                static,
                wout_like=trace["wout_like"],
                trig=trace["trig"],
                apply_lforbal=trace["apply_lforbal"],
                include_edge_residual=trace["include_edge_residual"],
                apply_m1_constraints=trace["apply_m1_constraints"],
                zero_m1=trace["zero_m1"],
                mats=trace["precond_mats"],
                jmax=trace["precond_jmax"],
                lam_prec=trace["lam_prec"],
                w_mode_mn=trace["w_mode_mn"],
                lambda_update_scale=trace["lambda_update_scale"],
                dt_eff=trace["dt_eff"],
                b1=trace["b1"],
                fac=trace["fac"],
                force_scale=trace["force_scale"],
                flip_sign=trace["flip_sign"],
                vRcc_before=trace["vRcc_before"],
                vRss_before=trace["vRss_before"],
                vZsc_before=trace["vZsc_before"],
                vZcs_before=trace["vZcs_before"],
                vLsc_before=trace["vLsc_before"],
                vLcs_before=trace["vLcs_before"],
                max_update_rms=trace["max_update_rms_pre"],
                limit_update_rms=trace["limit_update_rms"],
                divide_by_scalxc_for_update=trace["divide_by_scalxc_for_update"],
            )
            state = out["step"]["state_post"]
        return pack_state(state)

    _, vjp_fun = jax.vjp(_forward_two_step, x0)
    direct = vjp_fun(cotangent)[0]
    replay = checkpoint_tape_state_vjp(
        tape=tape,
        static=static,
        final_cotangent=cotangent,
    )
    assert np.asarray(replay) == pytest.approx(np.asarray(direct), rel=1.0e-10, abs=1.0e-10)


def test_checkpoint_tape_state_jvp_matches_direct_two_step_qh(load_case_qh_warm_start):
    pytest.importorskip("jax")
    _require_slow()

    from vmec_jax._compat import enable_x64, jax, jnp
    from vmec_jax.discrete_adjoint import build_residual_checkpoint_tape, checkpoint_tape_state_jvp
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.state import pack_state, unpack_state

    enable_x64(True)

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    common_kwargs = dict(
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        light_history=True,
        resume_state_mode="full",
    )
    tape = build_residual_checkpoint_tape(
        state_guess,
        static,
        max_iter=2,
        solver_kwargs=common_kwargs,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
        light_history=True,
        resume_state_mode="full",
    )
    assert len(tape.step_traces) == 2

    x0 = jnp.asarray(pack_state(tape.step_traces[0]["state_pre"]))
    tangent = jnp.linspace(-0.05, 0.05, int(x0.size), dtype=x0.dtype)

    def _forward_two_step(x):
        from vmec_jax.discrete_adjoint import strict_update_one_step_from_state

        state = unpack_state(x, tape.step_traces[0]["state_pre"].layout)
        for trace in tape.step_traces:
            out = strict_update_one_step_from_state(
                state,
                static,
                wout_like=trace["wout_like"],
                trig=trace["trig"],
                apply_lforbal=trace["apply_lforbal"],
                include_edge_residual=trace["include_edge_residual"],
                apply_m1_constraints=trace["apply_m1_constraints"],
                zero_m1=trace["zero_m1"],
                mats=trace["precond_mats"],
                jmax=trace["precond_jmax"],
                lam_prec=trace["lam_prec"],
                w_mode_mn=trace["w_mode_mn"],
                lambda_update_scale=trace["lambda_update_scale"],
                dt_eff=trace["dt_eff"],
                b1=trace["b1"],
                fac=trace["fac"],
                force_scale=trace["force_scale"],
                flip_sign=trace["flip_sign"],
                vRcc_before=trace["vRcc_before"],
                vRss_before=trace["vRss_before"],
                vZsc_before=trace["vZsc_before"],
                vZcs_before=trace["vZcs_before"],
                vLsc_before=trace["vLsc_before"],
                vLcs_before=trace["vLcs_before"],
                max_update_rms=trace["max_update_rms_pre"],
                limit_update_rms=trace["limit_update_rms"],
                divide_by_scalxc_for_update=trace["divide_by_scalxc_for_update"],
            )
            state = out["step"]["state_post"]
        return pack_state(state)

    direct = jax.jvp(_forward_two_step, (x0,), (tangent,))[1]
    replay = checkpoint_tape_state_jvp(
        tape=tape,
        static=static,
        initial_tangent=tangent,
    )
    assert np.asarray(replay) == pytest.approx(np.asarray(direct), rel=1.0e-10, abs=1.0e-10)


def test_checkpoint_tape_state_jvp_columns_matches_single_column_qh(load_case_qh_warm_start):
    pytest.importorskip("jax")
    _require_slow()

    from vmec_jax._compat import enable_x64, jnp
    from vmec_jax.discrete_adjoint import (
        build_residual_checkpoint_tape,
        checkpoint_tape_state_jvp,
        checkpoint_tape_state_jvp_columns,
    )
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.state import pack_state

    enable_x64(True)

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    common_kwargs = dict(
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        light_history=True,
        resume_state_mode="full",
    )
    tape = build_residual_checkpoint_tape(
        state_guess,
        static,
        max_iter=2,
        solver_kwargs=common_kwargs,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
        light_history=True,
        resume_state_mode="full",
    )
    assert len(tape.step_traces) == 2

    x0 = jnp.asarray(pack_state(tape.step_traces[0]["state_pre"]))
    tangents = jnp.stack(
        (
            jnp.linspace(-0.05, 0.05, int(x0.size), dtype=x0.dtype),
            jnp.linspace(0.03, -0.03, int(x0.size), dtype=x0.dtype),
        ),
        axis=0,
    )

    batch = checkpoint_tape_state_jvp_columns(
        tape=tape,
        static=static,
        initial_tangents=tangents,
    )
    single = jnp.stack(
        [
            checkpoint_tape_state_jvp(
                tape=tape,
                static=static,
                initial_tangent=tangents[i],
            )
            for i in range(int(tangents.shape[0]))
        ],
        axis=0,
    )
    assert np.asarray(batch) == pytest.approx(np.asarray(single), rel=1.0e-10, abs=1.0e-10)


def test_checkpoint_tape_state_jvp_columns_matches_single_column_qh_rebuild_preconditioner(
    load_case_qh_warm_start,
):
    pytest.importorskip("jax")
    _require_slow()

    from vmec_jax._compat import enable_x64, jnp
    from vmec_jax.discrete_adjoint import (
        build_residual_checkpoint_tape,
        checkpoint_tape_state_jvp,
        checkpoint_tape_state_jvp_columns,
    )
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.state import pack_state

    enable_x64(True)

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    common_kwargs = dict(
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        light_history=True,
        resume_state_mode="full",
    )
    tape = build_residual_checkpoint_tape(
        state_guess,
        static,
        max_iter=2,
        solver_kwargs=common_kwargs,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
        light_history=True,
        resume_state_mode="full",
    )
    assert len(tape.step_traces) == 2

    x0 = jnp.asarray(pack_state(tape.step_traces[0]["state_pre"]))
    tangents = jnp.stack(
        (
            jnp.linspace(-0.05, 0.05, int(x0.size), dtype=x0.dtype),
            jnp.linspace(0.03, -0.03, int(x0.size), dtype=x0.dtype),
        ),
        axis=0,
    )

    batch = checkpoint_tape_state_jvp_columns(
        tape=tape,
        static=static,
        initial_tangents=tangents,
        rebuild_preconditioner=True,
    )
    single = jnp.stack(
        [
            checkpoint_tape_state_jvp(
                tape=tape,
                static=static,
                initial_tangent=tangents[i],
                rebuild_preconditioner=True,
            )
            for i in range(int(tangents.shape[0]))
        ],
        axis=0,
    )
    assert np.asarray(batch) == pytest.approx(np.asarray(single), rel=1.0e-10, abs=1.0e-10)


def test_dynamic_replay_scan_matches_primal_qh_full_inner(load_case_qh_warm_start):
    pytest.importorskip("jax")
    _require_slow()

    import vmec_jax.discrete_adjoint as da
    from vmec_jax._compat import enable_x64
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary

    enable_x64(True)

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    common_kwargs = dict(
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        light_history=True,
        resume_state_mode="full",
    )
    tape = da.build_residual_checkpoint_tape_direct(
        state_guess,
        static,
        max_iter=20,
        solver_kwargs=common_kwargs,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
        light_history=True,
        store_trace=False,
    )
    assert da._dynamic_replay_supported(tape=tape, rebuild_preconditioner=True)
    carry0 = da._dynamic_replay_initial_carry(tape.step_traces[0])
    run_scan = da._checkpoint_tape_dynamic_scan_runner(
        static=static,
        stacked=tape.stacked_step_traces,
        static_flags=tape.step_trace_static_flags,
    )
    carryf = run_scan(carry0, tape.stacked_step_traces)
    final_state_linf = float(np.max(np.abs(np.asarray(carryf[0]) - np.asarray(tape.final_packed_state))))
    assert final_state_linf < 1.0e-5


def test_checkpoint_tape_state_jvp_columns_matches_single_column_circular(load_case_circular_tokamak):
    pytest.importorskip("jax")

    from vmec_jax._compat import enable_x64, jnp
    from vmec_jax.discrete_adjoint import (
        build_residual_checkpoint_tape,
        checkpoint_tape_state_jvp,
        checkpoint_tape_state_jvp_columns,
    )
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.state import pack_state

    enable_x64(True)

    _cfg, indata, static, boundary, _state0 = load_case_circular_tokamak
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    common_kwargs = dict(
        indata=indata,
        signgs=signgs,
        ftol=1.0e-10,
        step_size=1.0,
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        light_history=True,
        resume_state_mode="full",
    )
    tape = build_residual_checkpoint_tape(
        state_guess,
        static,
        max_iter=1,
        solver_kwargs=common_kwargs,
        indata=indata,
        signgs=signgs,
        ftol=1.0e-10,
        step_size=1.0,
        light_history=True,
        resume_state_mode="full",
    )
    assert len(tape.step_traces) == 1

    x0 = jnp.asarray(pack_state(tape.step_traces[0]["state_pre"]))
    tangents = jnp.stack(
        (
            jnp.linspace(-0.02, 0.02, int(x0.size), dtype=x0.dtype),
            jnp.linspace(0.01, -0.01, int(x0.size), dtype=x0.dtype),
        ),
        axis=0,
    )

    batch = checkpoint_tape_state_jvp_columns(
        tape=tape,
        static=static,
        initial_tangents=tangents,
        rebuild_preconditioner=True,
    )
    single = jnp.stack(
        [
            checkpoint_tape_state_jvp(
                tape=tape,
                static=static,
                initial_tangent=tangents[i],
                rebuild_preconditioner=True,
            )
            for i in range(int(tangents.shape[0]))
        ],
        axis=0,
    )
    assert np.asarray(batch) == pytest.approx(np.asarray(single), rel=1.0e-10, abs=1.0e-10)


def test_checkpoint_tape_param_vjp_matches_direct_two_step_qh(load_case_qh_warm_start):
    pytest.importorskip("jax")
    _require_slow()

    from vmec_jax._compat import enable_x64, jax, jnp
    from vmec_jax.discrete_adjoint import build_residual_checkpoint_tape, checkpoint_tape_param_vjp
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import extract_axis_override_from_state, initial_guess_from_boundary
    from vmec_jax.optimization import apply_boundary_params, boundary_param_specs
    from vmec_jax.state import pack_state, unpack_state

    enable_x64(True)

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    axis_override = extract_axis_override_from_state(state_guess, static)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    common_kwargs = dict(
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        light_history=True,
        resume_state_mode="full",
    )
    tape = build_residual_checkpoint_tape(
        state_guess,
        static,
        max_iter=2,
        solver_kwargs=common_kwargs,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
        light_history=True,
        resume_state_mode="full",
    )

    specs = boundary_param_specs(
        boundary,
        static.modes,
        max_mode=1,
        min_coeff=0.0,
        include=("rc",),
        fix=(),
    )
    specs = [spec for spec in specs if spec.m == 0 and spec.n == 1]
    assert len(specs) == 1
    params0 = jnp.zeros((1,), dtype=jnp.float64)
    cotangent = jnp.linspace(-0.2, 0.2, int(state_guess.layout.size), dtype=jnp.float64)

    def _forward_from_params(p):
        from vmec_jax.discrete_adjoint import strict_update_one_step_from_state

        boundary_p = apply_boundary_params(boundary, specs, p)
        state = initial_guess_from_boundary(
            static,
            boundary_p,
            indata,
            vmec_project=True,
            axis_override=axis_override,
        )
        x = jnp.asarray(pack_state(state))
        state = unpack_state(x, state.layout)
        for trace in tape.step_traces:
            out = strict_update_one_step_from_state(
                state,
                static,
                wout_like=trace["wout_like"],
                trig=trace["trig"],
                apply_lforbal=trace["apply_lforbal"],
                include_edge_residual=trace["include_edge_residual"],
                apply_m1_constraints=trace["apply_m1_constraints"],
                zero_m1=trace["zero_m1"],
                lambda_update_scale=trace["lambda_update_scale"],
                dt_eff=trace["dt_eff"],
                b1=trace["b1"],
                fac=trace["fac"],
                force_scale=trace["force_scale"],
                flip_sign=trace["flip_sign"],
                vRcc_before=trace["vRcc_before"],
                vRss_before=trace["vRss_before"],
                vZsc_before=trace["vZsc_before"],
                vZcs_before=trace["vZcs_before"],
                vLsc_before=trace["vLsc_before"],
                vLcs_before=trace["vLcs_before"],
                max_update_rms=trace["max_update_rms_pre"],
                limit_update_rms=trace["limit_update_rms"],
                divide_by_scalxc_for_update=trace["divide_by_scalxc_for_update"],
            )
            state = out["step"]["state_post"]
        return pack_state(state)

    _, vjp_fun = jax.vjp(_forward_from_params, params0)
    direct = vjp_fun(cotangent)[0]
    replay = checkpoint_tape_param_vjp(
        tape=tape,
        static=static,
        boundary=boundary,
        indata=indata,
        specs=specs,
        params=params0,
        axis_override=axis_override,
        final_cotangent=cotangent,
        vmec_project=True,
        rebuild_preconditioner=True,
    )
    assert np.asarray(replay) == pytest.approx(np.asarray(direct), rel=1.0e-9, abs=1.0e-9)


def test_checkpoint_tape_param_jvp_matches_two_step_aspect_direction_qh(load_case_qh_warm_start):
    pytest.importorskip("jax")
    _require_slow()

    from vmec_jax._compat import enable_x64, jax, jnp
    from vmec_jax.discrete_adjoint import build_residual_checkpoint_tape, checkpoint_tape_param_jvp
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import extract_axis_override_from_state, initial_guess_from_boundary
    from vmec_jax.optimization import apply_boundary_params, boundary_param_specs
    from vmec_jax.state import pack_state, unpack_state
    from vmec_jax.wout import equilibrium_aspect_ratio_from_state

    enable_x64(True)

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    axis_override = extract_axis_override_from_state(state_guess, static)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    common_kwargs = dict(
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        light_history=True,
        resume_state_mode="full",
    )
    tape = build_residual_checkpoint_tape(
        state_guess,
        static,
        max_iter=2,
        solver_kwargs=common_kwargs,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
        light_history=True,
        resume_state_mode="full",
    )

    specs = boundary_param_specs(
        boundary,
        static.modes,
        max_mode=1,
        min_coeff=0.0,
        include=("rc",),
        fix=(),
    )
    specs = [spec for spec in specs if spec.m == 0 and spec.n == 1]
    assert len(specs) == 1
    params0 = jnp.zeros((1,), dtype=jnp.float64)
    params_tangent = jnp.asarray([1.0], dtype=jnp.float64)

    def _final_state_from_params(p):
        from vmec_jax.discrete_adjoint import strict_update_one_step_from_state

        boundary_p = apply_boundary_params(boundary, specs, p)
        state = initial_guess_from_boundary(
            static,
            boundary_p,
            indata,
            vmec_project=True,
            axis_override=axis_override,
        )
        for trace in tape.step_traces:
            out = strict_update_one_step_from_state(
                state,
                static,
                wout_like=trace["wout_like"],
                trig=trace["trig"],
                apply_lforbal=trace["apply_lforbal"],
                include_edge_residual=trace["include_edge_residual"],
                apply_m1_constraints=trace["apply_m1_constraints"],
                zero_m1=trace["zero_m1"],
                lambda_update_scale=trace["lambda_update_scale"],
                dt_eff=trace["dt_eff"],
                b1=trace["b1"],
                fac=trace["fac"],
                force_scale=trace["force_scale"],
                flip_sign=trace["flip_sign"],
                vRcc_before=trace["vRcc_before"],
                vRss_before=trace["vRss_before"],
                vZsc_before=trace["vZsc_before"],
                vZcs_before=trace["vZcs_before"],
                vLsc_before=trace["vLsc_before"],
                vLcs_before=trace["vLcs_before"],
                max_update_rms=trace["max_update_rms_pre"],
                limit_update_rms=trace["limit_update_rms"],
                divide_by_scalxc_for_update=trace["divide_by_scalxc_for_update"],
            )
            state = out["step"]["state_post"]
        return state

    def _aspect_from_params(p):
        state = _final_state_from_params(p)
        return equilibrium_aspect_ratio_from_state(state=state, static=static)

    direct = jax.jvp(_aspect_from_params, (params0,), (params_tangent,))[1]
    final_state = _final_state_from_params(params0)
    final_x = jnp.asarray(pack_state(final_state))

    def _aspect_from_packed(x):
        state = unpack_state(x, final_state.layout)
        return equilibrium_aspect_ratio_from_state(state=state, static=static)

    final_state_tangent = checkpoint_tape_param_jvp(
        tape=tape,
        static=static,
        boundary=boundary,
        indata=indata,
        specs=specs,
        params=params0,
        axis_override=axis_override,
        params_tangent=params_tangent,
        vmec_project=True,
        rebuild_preconditioner=True,
    )
    replay = jax.jvp(_aspect_from_packed, (final_x,), (final_state_tangent,))[1]
    eps = 1.0e-5
    fd = (
        float(np.asarray(_aspect_from_params(params0 + eps * params_tangent)))
        - float(np.asarray(_aspect_from_params(params0 - eps * params_tangent)))
    ) / (2.0 * eps)

    assert float(np.asarray(replay)) == pytest.approx(float(np.asarray(direct)), rel=1.0e-9, abs=1.0e-9)
    assert float(np.asarray(replay)) == pytest.approx(fd, rel=1.0e-5, abs=1.0e-8)


def test_checkpoint_tape_param_vjp_matches_two_step_aspect_gradient_qh(load_case_qh_warm_start):
    pytest.importorskip("jax")
    _require_slow()

    from vmec_jax._compat import enable_x64, jax, jnp
    from vmec_jax.discrete_adjoint import build_residual_checkpoint_tape, checkpoint_tape_param_vjp
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import extract_axis_override_from_state, initial_guess_from_boundary
    from vmec_jax.optimization import apply_boundary_params, boundary_param_specs
    from vmec_jax.state import pack_state, unpack_state
    from vmec_jax.wout import equilibrium_aspect_ratio_from_state

    enable_x64(True)

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    axis_override = extract_axis_override_from_state(state_guess, static)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    common_kwargs = dict(
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        light_history=True,
        resume_state_mode="full",
    )
    tape = build_residual_checkpoint_tape(
        state_guess,
        static,
        max_iter=2,
        solver_kwargs=common_kwargs,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
        light_history=True,
        resume_state_mode="full",
    )

    specs = boundary_param_specs(
        boundary,
        static.modes,
        max_mode=1,
        min_coeff=0.0,
        include=("rc",),
        fix=(),
    )
    specs = [spec for spec in specs if spec.m == 0 and spec.n == 1]
    assert len(specs) == 1
    params0 = jnp.zeros((1,), dtype=jnp.float64)

    def _final_state_from_params(p):
        from vmec_jax.discrete_adjoint import strict_update_one_step_from_state

        boundary_p = apply_boundary_params(boundary, specs, p)
        state = initial_guess_from_boundary(
            static,
            boundary_p,
            indata,
            vmec_project=True,
            axis_override=axis_override,
        )
        for trace in tape.step_traces:
            out = strict_update_one_step_from_state(
                state,
                static,
                wout_like=trace["wout_like"],
                trig=trace["trig"],
                apply_lforbal=trace["apply_lforbal"],
                include_edge_residual=trace["include_edge_residual"],
                apply_m1_constraints=trace["apply_m1_constraints"],
                zero_m1=trace["zero_m1"],
                lambda_update_scale=trace["lambda_update_scale"],
                dt_eff=trace["dt_eff"],
                b1=trace["b1"],
                fac=trace["fac"],
                force_scale=trace["force_scale"],
                flip_sign=trace["flip_sign"],
                vRcc_before=trace["vRcc_before"],
                vRss_before=trace["vRss_before"],
                vZsc_before=trace["vZsc_before"],
                vZcs_before=trace["vZcs_before"],
                vLsc_before=trace["vLsc_before"],
                vLcs_before=trace["vLcs_before"],
                max_update_rms=trace["max_update_rms_pre"],
                limit_update_rms=trace["limit_update_rms"],
                divide_by_scalxc_for_update=trace["divide_by_scalxc_for_update"],
            )
            state = out["step"]["state_post"]
        return state

    def _aspect_from_params(p):
        state = _final_state_from_params(p)
        return equilibrium_aspect_ratio_from_state(state=state, static=static)

    direct = jax.grad(_aspect_from_params)(params0)
    final_state = _final_state_from_params(params0)
    final_x = jnp.asarray(pack_state(final_state))

    def _aspect_from_packed(x):
        state = unpack_state(x, final_state.layout)
        return equilibrium_aspect_ratio_from_state(state=state, static=static)

    final_cotangent = jax.grad(_aspect_from_packed)(final_x)
    replay = checkpoint_tape_param_vjp(
        tape=tape,
        static=static,
        boundary=boundary,
        indata=indata,
        specs=specs,
        params=params0,
        axis_override=axis_override,
        final_cotangent=final_cotangent,
        vmec_project=True,
        rebuild_preconditioner=True,
    )
    eps = 1.0e-5
    fd = (
        float(np.asarray(_aspect_from_params(params0 + jnp.asarray([eps]))))
        - float(np.asarray(_aspect_from_params(params0 - jnp.asarray([eps]))))
    ) / (2.0 * eps)

    assert np.asarray(replay) == pytest.approx(np.asarray(direct), rel=1.0e-9, abs=1.0e-9)
    assert float(np.asarray(replay[0])) == pytest.approx(fd, rel=1.0e-5, abs=1.0e-8)


def test_replay_residual_checkpoint_step_matches_second_direct_step_qh(load_case_qh_warm_start):
    pytest.importorskip("jax")
    _require_slow()

    from vmec_jax.discrete_adjoint import replay_residual_checkpoint_step
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.solve import solve_fixed_boundary_residual_iter
    from vmec_jax.state import pack_state

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    solve_kwargs = dict(
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        host_update_assembly=False,
        light_history=False,
        resume_state_mode="full",
    )

    step1 = solve_fixed_boundary_residual_iter(state_guess, static, max_iter=1, **solve_kwargs)
    step2 = replay_residual_checkpoint_step(
        step1.state,
        static,
        resume_state=step1.diagnostics.get("resume_state"),
        solve_kwargs=solve_kwargs,
    )
    direct2 = solve_fixed_boundary_residual_iter(state_guess, static, max_iter=2, **solve_kwargs)

    assert np.asarray(pack_state(step2.state)) == pytest.approx(
        np.asarray(pack_state(direct2.state)),
        rel=0.0,
        abs=1.0e-12,
    )


def test_strict_update_velocity_state_advance_reconstructs_first_qh_step(load_case_qh_warm_start):
    pytest.importorskip("jax")

    from vmec_jax.discrete_adjoint import strict_update_velocity_state_advance
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.solve import solve_fixed_boundary_residual_iter
    from vmec_jax.state import pack_state

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    result = solve_fixed_boundary_residual_iter(
        state_guess,
        static,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        max_iter=1,
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        host_update_assembly=False,
        light_history=False,
        resume_state_mode="full",
    )
    assert result.diagnostics["step_status_history"][0] == "momentum"

    resume = result.diagnostics["resume_state"]
    reconstructed = strict_update_velocity_state_advance(
        state_guess,
        static,
        dt_eff=float(result.diagnostics["dt_eff_history"][0]),
        vRcc=resume["vRcc"],
        vRss=resume["vRss"],
        vZsc=resume["vZsc"],
        vZcs=resume["vZcs"],
        vLsc=resume["vLsc"],
        vLcs=resume["vLcs"],
        edge_Rcos=np.asarray(state_guess.Rcos)[-1, :],
        edge_Rsin=np.asarray(state_guess.Rsin)[-1, :],
        edge_Zcos=np.asarray(state_guess.Zcos)[-1, :],
        edge_Zsin=np.asarray(state_guess.Zsin)[-1, :],
    )
    assert np.asarray(pack_state(reconstructed)) == pytest.approx(
        np.asarray(pack_state(result.state)),
        rel=0.0,
        abs=1.0e-12,
    )


def test_strict_update_velocity_block_reconstructs_first_qh_step(load_case_qh_warm_start):
    pytest.importorskip("jax")

    from vmec_jax.discrete_adjoint import strict_update_velocity_block
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.solve import solve_fixed_boundary_residual_iter

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    result = solve_fixed_boundary_residual_iter(
        state_guess,
        static,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        max_iter=1,
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        host_update_assembly=False,
        light_history=False,
        resume_state_mode="full",
        adjoint_trace=True,
    )
    trace = result.diagnostics["adjoint_step_trace"][0]
    block = strict_update_velocity_block(
        b1=trace["b1"],
        fac=trace["fac"],
        force_scale=trace["force_scale"],
        flip_sign=trace["flip_sign"],
        vRcc_before=trace["vRcc_before"],
        vRss_before=trace["vRss_before"],
        vZsc_before=trace["vZsc_before"],
        vZcs_before=trace["vZcs_before"],
        vLsc_before=trace["vLsc_before"],
        vLcs_before=trace["vLcs_before"],
        frcc_u=trace["frcc_u"],
        frss_u=trace["frss_u"],
        fzsc_u=trace["fzsc_u"],
        fzcs_u=trace["fzcs_u"],
        flsc_u=trace["flsc_u"],
        flcs_u=trace["flcs_u"],
    )
    for key in ["vRcc_after", "vRss_after", "vZsc_after", "vZcs_after", "vLsc_after", "vLcs_after"]:
        assert np.asarray(block[key]) == pytest.approx(np.asarray(trace[key]), rel=0.0, abs=1.0e-12)


def test_strict_update_velocity_state_advance_vjp_identity_qh(load_case_qh_warm_start):
    pytest.importorskip("jax")

    from vmec_jax._compat import enable_x64, jax, jnp
    from vmec_jax.discrete_adjoint import strict_update_velocity_state_advance
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.solve import solve_fixed_boundary_residual_iter
    from vmec_jax.state import pack_state

    enable_x64(True)

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    result = solve_fixed_boundary_residual_iter(
        state_guess,
        static,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        max_iter=1,
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        light_history=False,
        resume_state_mode="full",
    )
    resume = result.diagnostics["resume_state"]
    dt_eff = float(result.diagnostics["dt_eff_history"][0])
    edge_Rcos = jnp.asarray(np.asarray(state_guess.Rcos)[-1, :])
    edge_Rsin = jnp.asarray(np.asarray(state_guess.Rsin)[-1, :])
    edge_Zcos = jnp.asarray(np.asarray(state_guess.Zcos)[-1, :])
    edge_Zsin = jnp.asarray(np.asarray(state_guess.Zsin)[-1, :])

    pieces0 = [
        jnp.asarray(resume["vRcc"]),
        jnp.asarray(resume["vRss"]),
        jnp.asarray(resume["vZsc"]),
        jnp.asarray(resume["vZcs"]),
        jnp.asarray(resume["vLsc"]),
        jnp.asarray(resume["vLcs"]),
    ]
    sizes = [int(p.size) for p in pieces0]
    offsets = np.cumsum([0] + sizes)
    x0 = jnp.concatenate([jnp.ravel(p) for p in pieces0], axis=0)
    tangent = jnp.linspace(0.1, 1.0, int(x0.size), dtype=x0.dtype)
    cotangent = jnp.linspace(-0.7, 0.3, int(state_guess.layout.size), dtype=x0.dtype)

    def _unpack(x):
        vals = []
        for start, stop, ref in zip(offsets[:-1], offsets[1:], pieces0):
            vals.append(jnp.reshape(x[start:stop], ref.shape))
        return vals

    def _f(x):
        vRcc, vRss, vZsc, vZcs, vLsc, vLcs = _unpack(x)
        state = strict_update_velocity_state_advance(
            state_guess,
            static,
            dt_eff=dt_eff,
            vRcc=vRcc,
            vRss=vRss,
            vZsc=vZsc,
            vZcs=vZcs,
            vLsc=vLsc,
            vLcs=vLcs,
            edge_Rcos=edge_Rcos,
            edge_Rsin=edge_Rsin,
            edge_Zcos=edge_Zcos,
            edge_Zsin=edge_Zsin,
        )
        return pack_state(state)

    y0, jvp = jax.jvp(_f, (x0,), (tangent,))
    _, vjp_fun = jax.vjp(_f, x0)
    vjp = vjp_fun(cotangent)[0]

    lhs = float(jnp.vdot(jvp, cotangent))
    rhs = float(jnp.vdot(tangent, vjp))
    assert np.asarray(y0).shape == (state_guess.layout.size,)
    assert lhs == pytest.approx(rhs, rel=1.0e-10, abs=1.0e-10)


def test_strict_update_velocity_state_advance_taylor_qh(load_case_qh_warm_start):
    pytest.importorskip("jax")

    from vmec_jax._compat import enable_x64, jax, jnp
    from vmec_jax.discrete_adjoint import strict_update_velocity_state_advance
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.solve import solve_fixed_boundary_residual_iter
    from vmec_jax.state import pack_state

    enable_x64(True)

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    result = solve_fixed_boundary_residual_iter(
        state_guess,
        static,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        max_iter=1,
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        host_update_assembly=False,
        light_history=False,
        resume_state_mode="full",
    )
    resume = result.diagnostics["resume_state"]
    dt_eff = float(result.diagnostics["dt_eff_history"][0])
    edge_Rcos = jnp.asarray(np.asarray(state_guess.Rcos)[-1, :])
    edge_Rsin = jnp.asarray(np.asarray(state_guess.Rsin)[-1, :])
    edge_Zcos = jnp.asarray(np.asarray(state_guess.Zcos)[-1, :])
    edge_Zsin = jnp.asarray(np.asarray(state_guess.Zsin)[-1, :])

    pieces0 = [
        jnp.asarray(resume["vRcc"]),
        jnp.asarray(resume["vRss"]),
        jnp.asarray(resume["vZsc"]),
        jnp.asarray(resume["vZcs"]),
        jnp.asarray(resume["vLsc"]),
        jnp.asarray(resume["vLcs"]),
    ]
    sizes = [int(p.size) for p in pieces0]
    offsets = np.cumsum([0] + sizes)
    x0 = jnp.concatenate([jnp.ravel(p) for p in pieces0], axis=0)
    direction = jnp.linspace(-0.3, 0.4, int(x0.size), dtype=x0.dtype)

    def _unpack(x):
        vals = []
        for start, stop, ref in zip(offsets[:-1], offsets[1:], pieces0):
            vals.append(jnp.reshape(x[start:stop], ref.shape))
        return vals

    def _f(x):
        vRcc, vRss, vZsc, vZcs, vLsc, vLcs = _unpack(x)
        state = strict_update_velocity_state_advance(
            state_guess,
            static,
            dt_eff=dt_eff,
            vRcc=vRcc,
            vRss=vRss,
            vZsc=vZsc,
            vZcs=vZcs,
            vLsc=vLsc,
            vLcs=vLcs,
            edge_Rcos=edge_Rcos,
            edge_Rsin=edge_Rsin,
            edge_Zcos=edge_Zcos,
            edge_Zsin=edge_Zsin,
        )
        return pack_state(state)

    y0, jvp = jax.jvp(_f, (x0,), (direction,))
    eps = 1.0e-6
    y1 = _f(x0 + eps * direction)
    residual = np.asarray(y1 - y0 - eps * jvp)
    assert np.max(np.abs(residual)) < 1.0e-8


def test_strict_update_velocity_block_vjp_identity_qh(load_case_qh_warm_start):
    pytest.importorskip("jax")

    from vmec_jax._compat import enable_x64, jax, jnp
    from vmec_jax.discrete_adjoint import strict_update_velocity_block
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.solve import solve_fixed_boundary_residual_iter

    enable_x64(True)

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    result = solve_fixed_boundary_residual_iter(
        state_guess,
        static,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        max_iter=1,
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        host_update_assembly=False,
        light_history=False,
        resume_state_mode="full",
        adjoint_trace=True,
    )
    trace = result.diagnostics["adjoint_step_trace"][0]
    pieces0 = [
        jnp.asarray(trace["vRcc_before"]),
        jnp.asarray(trace["vRss_before"]),
        jnp.asarray(trace["vZsc_before"]),
        jnp.asarray(trace["vZcs_before"]),
        jnp.asarray(trace["vLsc_before"]),
        jnp.asarray(trace["vLcs_before"]),
        jnp.asarray(trace["frcc_u"]),
        jnp.asarray(trace["frss_u"]),
        jnp.asarray(trace["fzsc_u"]),
        jnp.asarray(trace["fzcs_u"]),
        jnp.asarray(trace["flsc_u"]),
        jnp.asarray(trace["flcs_u"]),
    ]
    sizes = [int(p.size) for p in pieces0]
    offsets = np.cumsum([0] + sizes)
    x0 = jnp.concatenate([jnp.ravel(p) for p in pieces0], axis=0)
    tangent = jnp.linspace(0.2, 1.3, int(x0.size), dtype=x0.dtype)
    out_ref = strict_update_velocity_block(
        b1=trace["b1"],
        fac=trace["fac"],
        force_scale=trace["force_scale"],
        flip_sign=trace["flip_sign"],
        vRcc_before=trace["vRcc_before"],
        vRss_before=trace["vRss_before"],
        vZsc_before=trace["vZsc_before"],
        vZcs_before=trace["vZcs_before"],
        vLsc_before=trace["vLsc_before"],
        vLcs_before=trace["vLcs_before"],
        frcc_u=trace["frcc_u"],
        frss_u=trace["frss_u"],
        fzsc_u=trace["fzsc_u"],
        fzcs_u=trace["fzcs_u"],
        flsc_u=trace["flsc_u"],
        flcs_u=trace["flcs_u"],
    )
    y_ref = jnp.concatenate(
        [jnp.ravel(out_ref[k]) for k in ["vRcc_after", "vRss_after", "vZsc_after", "vZcs_after", "vLsc_after", "vLcs_after"]],
        axis=0,
    )
    cotangent = jnp.linspace(-0.5, 0.4, int(y_ref.size), dtype=x0.dtype)

    def _unpack(x):
        vals = []
        for start, stop, ref in zip(offsets[:-1], offsets[1:], pieces0):
            vals.append(jnp.reshape(x[start:stop], ref.shape))
        return vals

    def _f(x):
        vals = _unpack(x)
        out = strict_update_velocity_block(
            b1=trace["b1"],
            fac=trace["fac"],
            force_scale=trace["force_scale"],
            flip_sign=trace["flip_sign"],
            vRcc_before=vals[0],
            vRss_before=vals[1],
            vZsc_before=vals[2],
            vZcs_before=vals[3],
            vLsc_before=vals[4],
            vLcs_before=vals[5],
            frcc_u=vals[6],
            frss_u=vals[7],
            fzsc_u=vals[8],
            fzcs_u=vals[9],
            flsc_u=vals[10],
            flcs_u=vals[11],
        )
        return jnp.concatenate(
            [jnp.ravel(out[k]) for k in ["vRcc_after", "vRss_after", "vZsc_after", "vZcs_after", "vLsc_after", "vLcs_after"]],
            axis=0,
        )

    _, jvp = jax.jvp(_f, (x0,), (tangent,))
    _, vjp_fun = jax.vjp(_f, x0)
    vjp = vjp_fun(cotangent)[0]
    lhs = float(jnp.vdot(jvp, cotangent))
    rhs = float(jnp.vdot(tangent, vjp))
    assert lhs == pytest.approx(rhs, rel=1.0e-10, abs=1.0e-10)


def test_strict_update_velocity_limit_clips_and_vjp_identity():
    pytest.importorskip("jax")

    from vmec_jax._compat import enable_x64, jax, jnp
    from vmec_jax.discrete_adjoint import strict_update_velocity_limit

    enable_x64(True)

    dt_eff = 0.4
    max_update_rms = 0.02
    pieces0 = [
        jnp.ones((3, 2), dtype=jnp.float64) * 0.7,
        jnp.ones((3, 2), dtype=jnp.float64) * -0.5,
        jnp.ones((3, 2), dtype=jnp.float64) * 0.3,
        jnp.ones((3, 2), dtype=jnp.float64) * -0.4,
        jnp.ones((3, 2), dtype=jnp.float64) * 0.2,
        jnp.ones((3, 2), dtype=jnp.float64) * -0.6,
    ]
    sizes = [int(p.size) for p in pieces0]
    offsets = np.cumsum([0] + sizes)
    x0 = jnp.concatenate([jnp.ravel(p) for p in pieces0], axis=0)
    tangent = jnp.linspace(-0.2, 0.5, int(x0.size), dtype=x0.dtype)

    def _unpack(x):
        vals = []
        for start, stop, ref in zip(offsets[:-1], offsets[1:], pieces0):
            vals.append(jnp.reshape(x[start:stop], ref.shape))
        return vals

    def _f(x):
        vals = _unpack(x)
        out = strict_update_velocity_limit(
            dt_eff=dt_eff,
            max_update_rms=max_update_rms,
            limit_update_rms=True,
            vRcc=vals[0],
            vRss=vals[1],
            vZsc=vals[2],
            vZcs=vals[3],
            vLsc=vals[4],
            vLcs=vals[5],
        )
        return jnp.concatenate(
            [
                jnp.ravel(out["vRcc"]),
                jnp.ravel(out["vRss"]),
                jnp.ravel(out["vZsc"]),
                jnp.ravel(out["vZcs"]),
                jnp.ravel(out["vLsc"]),
                jnp.ravel(out["vLcs"]),
            ],
            axis=0,
        )

    out = strict_update_velocity_limit(
        dt_eff=dt_eff,
        max_update_rms=max_update_rms,
        limit_update_rms=True,
        vRcc=pieces0[0],
        vRss=pieces0[1],
        vZsc=pieces0[2],
        vZcs=pieces0[3],
        vLsc=pieces0[4],
        vLcs=pieces0[5],
    )
    assert float(np.asarray(out["update_rms_scale"])) < 1.0

    no_diag = strict_update_velocity_limit(
        dt_eff=dt_eff,
        max_update_rms=max_update_rms,
        limit_update_rms=False,
        need_update_rms=False,
        vRcc=pieces0[0],
        vRss=pieces0[1],
        vZsc=pieces0[2],
        vZcs=pieces0[3],
        vLsc=pieces0[4],
        vLcs=pieces0[5],
    )
    assert float(np.asarray(no_diag["update_rms_preclip"])) == pytest.approx(0.0)
    assert float(np.asarray(no_diag["update_rms_scale"])) == pytest.approx(1.0)
    np.testing.assert_allclose(np.asarray(no_diag["vRcc"]), np.asarray(pieces0[0]))

    @jax.jit
    def _traced_limit(limit_flag, need_flag):
        return strict_update_velocity_limit(
            dt_eff=dt_eff,
            max_update_rms=max_update_rms,
            limit_update_rms=limit_flag,
            need_update_rms=need_flag,
            vRcc=pieces0[0],
            vRss=pieces0[1],
            vZsc=pieces0[2],
            vZcs=pieces0[3],
            vLsc=pieces0[4],
            vLcs=pieces0[5],
        )

    traced_limited = _traced_limit(jnp.asarray(True), jnp.asarray(True))
    assert float(np.asarray(traced_limited["update_rms_scale"])) < 1.0
    traced_skipped = _traced_limit(jnp.asarray(False), jnp.asarray(False))
    assert float(np.asarray(traced_skipped["update_rms_preclip"])) == pytest.approx(0.0)
    assert float(np.asarray(traced_skipped["update_rms_scale"])) == pytest.approx(1.0)

    y_ref = _f(x0)
    cotangent = jnp.linspace(-0.4, 0.3, int(y_ref.size), dtype=x0.dtype)
    _, jvp = jax.jvp(_f, (x0,), (tangent,))
    _, vjp_fun = jax.vjp(_f, x0)
    vjp = vjp_fun(cotangent)[0]
    lhs = float(jnp.vdot(jvp, cotangent))
    rhs = float(jnp.vdot(tangent, vjp))
    assert lhs == pytest.approx(rhs, rel=1.0e-10, abs=1.0e-10)


def test_preconditioned_force_channels_from_rz_output_reconstructs_first_qh_step(load_case_qh_warm_start):
    pytest.importorskip("jax")

    from vmec_jax.discrete_adjoint import preconditioned_force_channels_from_rz_output
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.solve import solve_fixed_boundary_residual_iter
    from vmec_jax.vmec_tomnsp import TomnspsRZL

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    result = solve_fixed_boundary_residual_iter(
        state_guess,
        static,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        max_iter=1,
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        host_update_assembly=False,
        light_history=False,
        resume_state_mode="full",
        adjoint_trace=True,
    )
    trace = result.diagnostics["adjoint_step_trace"][0]
    frzl_rz = TomnspsRZL(
        frcc=trace["frzl_rz_frcc"],
        frss=trace["frzl_rz_frss"],
        fzsc=trace["frzl_rz_fzsc"],
        fzcs=trace["frzl_rz_fzcs"],
        flsc=trace["frzl_rz_flsc"],
        flcs=trace["frzl_rz_flcs"],
        frsc=trace["frzl_rz_frsc"],
        frcs=trace["frzl_rz_frcs"],
        fzcc=trace["frzl_rz_fzcc"],
        fzss=trace["frzl_rz_fzss"],
        flcc=trace["frzl_rz_flcc"],
        flss=trace["frzl_rz_flss"],
    )
    out = preconditioned_force_channels_from_rz_output(
        frzl_rz=frzl_rz,
        lam_prec=trace["lam_prec"],
        w_mode_mn=trace["w_mode_mn"],
        lambda_update_scale=trace["lambda_update_scale"],
    )
    for key in ["frcc_u", "frss_u", "fzsc_u", "fzcs_u", "flsc_u", "flcs_u"]:
        assert np.asarray(out[key]) == pytest.approx(np.asarray(trace[key]), rel=0.0, abs=1.0e-12)


def test_preconditioned_force_channels_from_rz_output_vjp_identity_qh(load_case_qh_warm_start):
    pytest.importorskip("jax")

    from vmec_jax._compat import enable_x64, jax, jnp
    from vmec_jax.discrete_adjoint import preconditioned_force_channels_from_rz_output
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.solve import solve_fixed_boundary_residual_iter
    from vmec_jax.vmec_tomnsp import TomnspsRZL

    enable_x64(True)

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    result = solve_fixed_boundary_residual_iter(
        state_guess,
        static,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        max_iter=1,
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        host_update_assembly=False,
        light_history=False,
        resume_state_mode="full",
        adjoint_trace=True,
    )
    trace = result.diagnostics["adjoint_step_trace"][0]
    pieces0 = [
        jnp.asarray(trace["frzl_rz_frcc"]),
        jnp.asarray(trace["frzl_rz_frss"]),
        jnp.asarray(trace["frzl_rz_fzsc"]),
        jnp.asarray(trace["frzl_rz_fzcs"]),
        jnp.asarray(trace["frzl_rz_flsc"]),
        jnp.asarray(trace["frzl_rz_flcs"]),
        jnp.asarray(trace["lam_prec"]),
    ]
    sizes = [int(p.size) for p in pieces0]
    offsets = np.cumsum([0] + sizes)
    x0 = jnp.concatenate([jnp.ravel(p) for p in pieces0], axis=0)
    tangent = jnp.linspace(-0.15, 0.65, int(x0.size), dtype=x0.dtype)

    def _unpack(x):
        vals = []
        for start, stop, ref in zip(offsets[:-1], offsets[1:], pieces0):
            vals.append(jnp.reshape(x[start:stop], ref.shape))
        return vals

    def _f(x):
        vals = _unpack(x)
        frzl_rz = TomnspsRZL(
            frcc=vals[0],
            frss=vals[1],
            fzsc=vals[2],
            fzcs=vals[3],
            flsc=vals[4],
            flcs=vals[5],
        )
        out = preconditioned_force_channels_from_rz_output(
            frzl_rz=frzl_rz,
            lam_prec=vals[6],
            w_mode_mn=trace["w_mode_mn"],
            lambda_update_scale=trace["lambda_update_scale"],
        )
        return jnp.concatenate(
            [
                jnp.ravel(out["frcc_u"]),
                jnp.ravel(out["frss_u"]),
                jnp.ravel(out["fzsc_u"]),
                jnp.ravel(out["fzcs_u"]),
                jnp.ravel(out["flsc_u"]),
                jnp.ravel(out["flcs_u"]),
            ],
            axis=0,
        )

    y_ref = _f(x0)
    cotangent = jnp.linspace(-0.35, 0.25, int(y_ref.size), dtype=x0.dtype)
    _, jvp = jax.jvp(_f, (x0,), (tangent,))
    _, vjp_fun = jax.vjp(_f, x0)
    vjp = vjp_fun(cotangent)[0]
    lhs = float(jnp.vdot(jvp, cotangent))
    rhs = float(jnp.vdot(tangent, vjp))
    assert lhs == pytest.approx(rhs, rel=1.0e-10, abs=1.0e-10)


def test_rz_output_to_accepted_step_reconstructs_first_qh_step(load_case_qh_warm_start):
    pytest.importorskip("jax")

    from vmec_jax.discrete_adjoint import preconditioned_force_channels_from_rz_output, strict_update_accepted_step
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.solve import solve_fixed_boundary_residual_iter
    from vmec_jax.state import pack_state
    from vmec_jax.vmec_tomnsp import TomnspsRZL

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    result = solve_fixed_boundary_residual_iter(
        state_guess,
        static,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        max_iter=1,
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        host_update_assembly=False,
        light_history=False,
        resume_state_mode="full",
        adjoint_trace=True,
    )
    trace = result.diagnostics["adjoint_step_trace"][0]
    frzl_rz = TomnspsRZL(
        frcc=trace["frzl_rz_frcc"],
        frss=trace["frzl_rz_frss"],
        fzsc=trace["frzl_rz_fzsc"],
        fzcs=trace["frzl_rz_fzcs"],
        flsc=trace["frzl_rz_flsc"],
        flcs=trace["frzl_rz_flcs"],
        frsc=trace["frzl_rz_frsc"],
        frcs=trace["frzl_rz_frcs"],
        fzcc=trace["frzl_rz_fzcc"],
        fzss=trace["frzl_rz_fzss"],
        flcc=trace["frzl_rz_flcc"],
        flss=trace["frzl_rz_flss"],
    )
    force_out = preconditioned_force_channels_from_rz_output(
        frzl_rz=frzl_rz,
        lam_prec=trace["lam_prec"],
        w_mode_mn=trace["w_mode_mn"],
        lambda_update_scale=trace["lambda_update_scale"],
    )
    out = strict_update_accepted_step(
        trace["state_pre"],
        static,
        dt_eff=trace["dt_eff"],
        b1=trace["b1"],
        fac=trace["fac"],
        force_scale=trace["force_scale"],
        flip_sign=trace["flip_sign"],
        vRcc_before=trace["vRcc_before"],
        vRss_before=trace["vRss_before"],
        vZsc_before=trace["vZsc_before"],
        vZcs_before=trace["vZcs_before"],
        vLsc_before=trace["vLsc_before"],
        vLcs_before=trace["vLcs_before"],
        frcc_u=force_out["frcc_u"],
        frss_u=force_out["frss_u"],
        fzsc_u=force_out["fzsc_u"],
        fzcs_u=force_out["fzcs_u"],
        flsc_u=force_out["flsc_u"],
        flcs_u=force_out["flcs_u"],
        max_update_rms=trace["max_update_rms_pre"],
        limit_update_rms=trace["limit_update_rms"],
        divide_by_scalxc_for_update=trace["divide_by_scalxc_for_update"],
    )
    assert np.asarray(pack_state(out["state_post"])) == pytest.approx(
        np.asarray(pack_state(trace["state_post"])),
        rel=0.0,
        abs=1.0e-12,
    )


def test_preconditioned_force_channels_from_raw_forces_reconstructs_first_qh_step(load_case_qh_warm_start):
    pytest.importorskip("jax")

    from vmec_jax.discrete_adjoint import preconditioned_force_channels_from_raw_forces
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.solve import solve_fixed_boundary_residual_iter
    from vmec_jax.vmec_tomnsp import TomnspsRZL

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    result = solve_fixed_boundary_residual_iter(
        state_guess,
        static,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        max_iter=1,
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        host_update_assembly=False,
        light_history=False,
        resume_state_mode="full",
        adjoint_trace=True,
    )
    trace = result.diagnostics["adjoint_step_trace"][0]
    frzl = TomnspsRZL(
        frcc=trace["frzl_frcc"],
        frss=trace["frzl_frss"],
        fzsc=trace["frzl_fzsc"],
        fzcs=trace["frzl_fzcs"],
        flsc=trace["frzl_flsc"],
        flcs=trace["frzl_flcs"],
        frsc=trace["frzl_frsc"],
        frcs=trace["frzl_frcs"],
        fzcc=trace["frzl_fzcc"],
        fzss=trace["frzl_fzss"],
        flcc=trace["frzl_flcc"],
        flss=trace["frzl_flss"],
    )
    out = preconditioned_force_channels_from_raw_forces(
        frzl=frzl,
        mats=trace["precond_mats"],
        jmax=trace["precond_jmax"],
        cfg=static.cfg,
        lam_prec=trace["lam_prec"],
        w_mode_mn=trace["w_mode_mn"],
        lambda_update_scale=trace["lambda_update_scale"],
    )
    assert np.asarray(out["frcc_u"]) == pytest.approx(np.asarray(trace["frcc_u"]), rel=0.0, abs=1.0e-12)
    assert np.asarray(out["flsc_u"]) == pytest.approx(np.asarray(trace["flsc_u"]), rel=0.0, abs=1.0e-12)


def test_preconditioned_force_channels_from_raw_forces_vjp_identity_qh(load_case_qh_warm_start):
    pytest.importorskip("jax")

    from vmec_jax._compat import enable_x64, jax, jnp
    from vmec_jax.discrete_adjoint import preconditioned_force_channels_from_raw_forces
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.solve import solve_fixed_boundary_residual_iter
    from vmec_jax.vmec_tomnsp import TomnspsRZL

    enable_x64(True)

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    result = solve_fixed_boundary_residual_iter(
        state_guess,
        static,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        max_iter=1,
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        host_update_assembly=False,
        light_history=False,
        resume_state_mode="full",
        adjoint_trace=True,
    )
    trace = result.diagnostics["adjoint_step_trace"][0]
    pieces0 = [
        jnp.asarray(trace["frzl_frcc"]),
        jnp.asarray(trace["frzl_frss"]),
        jnp.asarray(trace["frzl_fzsc"]),
        jnp.asarray(trace["frzl_fzcs"]),
        jnp.asarray(trace["frzl_flsc"]),
        jnp.asarray(trace["frzl_flcs"]),
    ]
    sizes = [int(p.size) for p in pieces0]
    offsets = np.cumsum([0] + sizes)
    x0 = jnp.concatenate([jnp.ravel(p) for p in pieces0], axis=0)
    tangent = jnp.linspace(-0.1, 0.55, int(x0.size), dtype=x0.dtype)

    def _unpack(x):
        vals = []
        for start, stop, ref in zip(offsets[:-1], offsets[1:], pieces0):
            vals.append(jnp.reshape(x[start:stop], ref.shape))
        return vals

    def _f(x):
        vals = _unpack(x)
        frzl = TomnspsRZL(
            frcc=vals[0],
            frss=vals[1],
            fzsc=vals[2],
            fzcs=vals[3],
            flsc=vals[4],
            flcs=vals[5],
        )
        out = preconditioned_force_channels_from_raw_forces(
            frzl=frzl,
            mats=trace["precond_mats"],
            jmax=trace["precond_jmax"],
            cfg=static.cfg,
            lam_prec=trace["lam_prec"],
            w_mode_mn=trace["w_mode_mn"],
            lambda_update_scale=trace["lambda_update_scale"],
        )
        return jnp.concatenate(
            [
                jnp.ravel(out["frcc_u"]),
                jnp.ravel(out["frss_u"]),
                jnp.ravel(out["fzsc_u"]),
                jnp.ravel(out["fzcs_u"]),
                jnp.ravel(out["flsc_u"]),
                jnp.ravel(out["flcs_u"]),
            ],
            axis=0,
        )

    y_ref = _f(x0)
    cotangent = jnp.linspace(-0.25, 0.2, int(y_ref.size), dtype=x0.dtype)
    _, jvp = jax.jvp(_f, (x0,), (tangent,))
    _, vjp_fun = jax.vjp(_f, x0)
    vjp = vjp_fun(cotangent)[0]
    lhs = float(jnp.vdot(jvp, cotangent))
    rhs = float(jnp.vdot(tangent, vjp))
    assert lhs == pytest.approx(rhs, rel=1.0e-10, abs=1.0e-10)


def test_raw_force_residual_from_state_reconstructs_first_qh_step(load_case_qh_warm_start):
    pytest.importorskip("jax")

    from vmec_jax.discrete_adjoint import raw_force_residual_from_state
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.solve import solve_fixed_boundary_residual_iter

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    result = solve_fixed_boundary_residual_iter(
        state_guess,
        static,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        max_iter=1,
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        host_update_assembly=False,
        light_history=False,
        resume_state_mode="full",
        adjoint_trace=True,
    )
    trace = result.diagnostics["adjoint_step_trace"][0]
    out = raw_force_residual_from_state(
        trace["state_pre"],
        static,
        wout_like=trace["wout_like"],
        trig=trace["trig"],
        apply_lforbal=trace["apply_lforbal"],
        include_edge_residual=trace["include_edge_residual"],
        apply_m1_constraints=trace["apply_m1_constraints"],
        zero_m1=trace["zero_m1"],
    )
    frzl = out["frzl"]
    assert np.asarray(frzl.frcc) == pytest.approx(np.asarray(trace["frzl_frcc"]), rel=0.0, abs=1.0e-12)
    assert np.asarray(frzl.fzsc) == pytest.approx(np.asarray(trace["frzl_fzsc"]), rel=0.0, abs=1.0e-12)
    assert np.asarray(frzl.flsc) == pytest.approx(np.asarray(trace["frzl_flsc"]), rel=0.0, abs=1.0e-12)


def test_raw_force_to_accepted_step_reconstructs_first_qh_step(load_case_qh_warm_start):
    pytest.importorskip("jax")

    from vmec_jax.discrete_adjoint import (
        preconditioned_force_channels_from_raw_forces,
        raw_force_residual_from_state,
        strict_update_accepted_step,
    )
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.solve import solve_fixed_boundary_residual_iter
    from vmec_jax.state import pack_state

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    result = solve_fixed_boundary_residual_iter(
        state_guess,
        static,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        max_iter=1,
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        host_update_assembly=False,
        light_history=False,
        resume_state_mode="full",
        adjoint_trace=True,
    )
    trace = result.diagnostics["adjoint_step_trace"][0]
    raw = raw_force_residual_from_state(
        trace["state_pre"],
        static,
        wout_like=trace["wout_like"],
        trig=trace["trig"],
        apply_lforbal=trace["apply_lforbal"],
        include_edge_residual=trace["include_edge_residual"],
        apply_m1_constraints=trace["apply_m1_constraints"],
        zero_m1=trace["zero_m1"],
    )
    force_out = preconditioned_force_channels_from_raw_forces(
        frzl=raw["frzl"],
        mats=trace["precond_mats"],
        jmax=trace["precond_jmax"],
        cfg=static.cfg,
        lam_prec=trace["lam_prec"],
        w_mode_mn=trace["w_mode_mn"],
        lambda_update_scale=trace["lambda_update_scale"],
    )
    out = strict_update_accepted_step(
        trace["state_pre"],
        static,
        dt_eff=trace["dt_eff"],
        b1=trace["b1"],
        fac=trace["fac"],
        force_scale=trace["force_scale"],
        flip_sign=trace["flip_sign"],
        vRcc_before=trace["vRcc_before"],
        vRss_before=trace["vRss_before"],
        vZsc_before=trace["vZsc_before"],
        vZcs_before=trace["vZcs_before"],
        vLsc_before=trace["vLsc_before"],
        vLcs_before=trace["vLcs_before"],
        frcc_u=force_out["frcc_u"],
        frss_u=force_out["frss_u"],
        fzsc_u=force_out["fzsc_u"],
        fzcs_u=force_out["fzcs_u"],
        flsc_u=force_out["flsc_u"],
        flcs_u=force_out["flcs_u"],
        max_update_rms=trace["max_update_rms_pre"],
        limit_update_rms=trace["limit_update_rms"],
        divide_by_scalxc_for_update=trace["divide_by_scalxc_for_update"],
    )
    assert np.asarray(pack_state(out["state_post"])) == pytest.approx(
        np.asarray(pack_state(trace["state_post"])),
        rel=0.0,
        abs=1.0e-12,
    )


def test_strict_update_one_step_from_state_reconstructs_first_qh_step(load_case_qh_warm_start):
    pytest.importorskip("jax")

    from vmec_jax.discrete_adjoint import strict_update_one_step_from_state
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.solve import solve_fixed_boundary_residual_iter
    from vmec_jax.state import pack_state

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    result = solve_fixed_boundary_residual_iter(
        state_guess,
        static,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        max_iter=1,
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        host_update_assembly=False,
        light_history=False,
        resume_state_mode="full",
        adjoint_trace=True,
    )
    trace = result.diagnostics["adjoint_step_trace"][0]
    out = strict_update_one_step_from_state(
        trace["state_pre"],
        static,
        wout_like=trace["wout_like"],
        trig=trace["trig"],
        apply_lforbal=trace["apply_lforbal"],
        include_edge_residual=trace["include_edge_residual"],
        apply_m1_constraints=trace["apply_m1_constraints"],
        zero_m1=trace["zero_m1"],
        mats=trace["precond_mats"],
        jmax=trace["precond_jmax"],
        lam_prec=trace["lam_prec"],
        w_mode_mn=trace["w_mode_mn"],
        lambda_update_scale=trace["lambda_update_scale"],
        dt_eff=trace["dt_eff"],
        b1=trace["b1"],
        fac=trace["fac"],
        force_scale=trace["force_scale"],
        flip_sign=trace["flip_sign"],
        vRcc_before=trace["vRcc_before"],
        vRss_before=trace["vRss_before"],
        vZsc_before=trace["vZsc_before"],
        vZcs_before=trace["vZcs_before"],
        vLsc_before=trace["vLsc_before"],
        vLcs_before=trace["vLcs_before"],
        max_update_rms=trace["max_update_rms_pre"],
        limit_update_rms=trace["limit_update_rms"],
        divide_by_scalxc_for_update=trace["divide_by_scalxc_for_update"],
    )
    assert np.asarray(pack_state(out["step"]["state_post"])) == pytest.approx(
        np.asarray(pack_state(trace["state_post"])),
        rel=0.0,
        abs=1.0e-12,
    )
    out_rebuilt = strict_update_one_step_from_state(
        trace["state_pre"],
        static,
        wout_like=trace["wout_like"],
        trig=trace["trig"],
        apply_lforbal=trace["apply_lforbal"],
        include_edge_residual=trace["include_edge_residual"],
        apply_m1_constraints=trace["apply_m1_constraints"],
        zero_m1=trace["zero_m1"],
        lambda_update_scale=trace["lambda_update_scale"],
        dt_eff=trace["dt_eff"],
        b1=trace["b1"],
        fac=trace["fac"],
        force_scale=trace["force_scale"],
        flip_sign=trace["flip_sign"],
        vRcc_before=trace["vRcc_before"],
        vRss_before=trace["vRss_before"],
        vZsc_before=trace["vZsc_before"],
        vZcs_before=trace["vZcs_before"],
        vLsc_before=trace["vLsc_before"],
        vLcs_before=trace["vLcs_before"],
        max_update_rms=trace["max_update_rms_pre"],
        limit_update_rms=trace["limit_update_rms"],
        divide_by_scalxc_for_update=trace["divide_by_scalxc_for_update"],
    )
    rebuilt_diff = float(
        np.max(
            np.abs(
                np.asarray(pack_state(out_rebuilt["step"]["state_post"]))
                - np.asarray(pack_state(trace["state_post"]))
            )
        )
    )
    assert rebuilt_diff < 2.0e-5


def test_strict_update_one_step_from_state_vjp_identity_qh(load_case_qh_warm_start):
    pytest.importorskip("jax")

    from vmec_jax._compat import enable_x64, jax, jnp
    from vmec_jax.discrete_adjoint import strict_update_one_step_from_state
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.solve import solve_fixed_boundary_residual_iter
    from vmec_jax.state import pack_state, unpack_state

    enable_x64(True)

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    result = solve_fixed_boundary_residual_iter(
        state_guess,
        static,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        max_iter=1,
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        host_update_assembly=False,
        light_history=False,
        resume_state_mode="full",
        adjoint_trace=True,
    )
    trace = result.diagnostics["adjoint_step_trace"][0]
    x0 = jnp.asarray(pack_state(trace["state_pre"]))
    tangent = jnp.linspace(-0.05, 0.05, int(x0.size), dtype=x0.dtype)
    cotangent = jnp.linspace(-0.2, 0.2, int(x0.size), dtype=x0.dtype)

    def _f(x):
        state = unpack_state(x, trace["state_pre"].layout)
        out = strict_update_one_step_from_state(
            state,
            static,
            wout_like=trace["wout_like"],
            trig=trace["trig"],
            apply_lforbal=trace["apply_lforbal"],
            include_edge_residual=trace["include_edge_residual"],
            apply_m1_constraints=trace["apply_m1_constraints"],
            zero_m1=trace["zero_m1"],
            mats=trace["precond_mats"],
            jmax=trace["precond_jmax"],
            lam_prec=trace["lam_prec"],
            w_mode_mn=trace["w_mode_mn"],
            lambda_update_scale=trace["lambda_update_scale"],
            dt_eff=trace["dt_eff"],
            b1=trace["b1"],
            fac=trace["fac"],
            force_scale=trace["force_scale"],
            flip_sign=trace["flip_sign"],
            vRcc_before=trace["vRcc_before"],
            vRss_before=trace["vRss_before"],
            vZsc_before=trace["vZsc_before"],
            vZcs_before=trace["vZcs_before"],
            vLsc_before=trace["vLsc_before"],
            vLcs_before=trace["vLcs_before"],
            max_update_rms=trace["max_update_rms_pre"],
            limit_update_rms=trace["limit_update_rms"],
            divide_by_scalxc_for_update=trace["divide_by_scalxc_for_update"],
        )
        return pack_state(out["step"]["state_post"])

    _, jvp = jax.jvp(_f, (x0,), (tangent,))
    _, vjp_fun = jax.vjp(_f, x0)
    vjp = vjp_fun(cotangent)[0]
    lhs = float(jnp.vdot(jvp, cotangent))
    rhs = float(jnp.vdot(tangent, vjp))
    assert lhs == pytest.approx(rhs, rel=1.0e-9, abs=1.0e-9)


def test_strict_update_one_step_boundary_derivative_matches_fd_with_frozen_axis(load_case_qh_warm_start):
    pytest.importorskip("jax")
    _require_slow()

    from vmec_jax._compat import enable_x64, jax, jnp
    from vmec_jax.boundary import BoundaryCoeffs
    from vmec_jax.discrete_adjoint import strict_update_one_step_from_state
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import extract_axis_override_from_state, initial_guess_from_boundary
    from vmec_jax.solve import solve_fixed_boundary_residual_iter

    enable_x64(True)

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    axis_override = extract_axis_override_from_state(state_guess, static)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    result = solve_fixed_boundary_residual_iter(
        state_guess,
        static,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        max_iter=1,
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        host_update_assembly=False,
        light_history=False,
        resume_state_mode="full",
        adjoint_trace=True,
    )
    trace = result.diagnostics["adjoint_step_trace"][0]
    k_rc01 = _mode_index(static.modes, 0, 1)
    alpha0 = float(np.asarray(boundary.R_cos, dtype=float)[k_rc01])

    def _lambda_scalar(alpha):
        boundary_alpha = BoundaryCoeffs(
            R_cos=jnp.asarray(boundary.R_cos).at[k_rc01].set(alpha),
            R_sin=jnp.asarray(boundary.R_sin),
            Z_cos=jnp.asarray(boundary.Z_cos),
            Z_sin=jnp.asarray(boundary.Z_sin),
        )
        state_pre = initial_guess_from_boundary(
            static,
            boundary_alpha,
            indata,
            vmec_project=True,
            axis_override=axis_override,
        )
        out = strict_update_one_step_from_state(
            state_pre,
            static,
            wout_like=trace["wout_like"],
            trig=trace["trig"],
            apply_lforbal=trace["apply_lforbal"],
            include_edge_residual=trace["include_edge_residual"],
            apply_m1_constraints=trace["apply_m1_constraints"],
            zero_m1=trace["zero_m1"],
            lambda_update_scale=trace["lambda_update_scale"],
            dt_eff=trace["dt_eff"],
            b1=trace["b1"],
            fac=trace["fac"],
            force_scale=trace["force_scale"],
            flip_sign=trace["flip_sign"],
            vRcc_before=trace["vRcc_before"],
            vRss_before=trace["vRss_before"],
            vZsc_before=trace["vZsc_before"],
            vZcs_before=trace["vZcs_before"],
            vLsc_before=trace["vLsc_before"],
            vLcs_before=trace["vLcs_before"],
            max_update_rms=trace["max_update_rms_pre"],
            limit_update_rms=trace["limit_update_rms"],
            divide_by_scalxc_for_update=trace["divide_by_scalxc_for_update"],
        )
        return out["step"]["state_post"].Lsin[5, k_rc01]

    eps = 1.0e-5
    grad_ad = float(np.asarray(jax.grad(_lambda_scalar)(alpha0)))
    grad_fd = float(
        (
            np.asarray(_lambda_scalar(alpha0 + eps))
            - np.asarray(_lambda_scalar(alpha0 - eps))
        )
        / (2.0 * eps)
    )

    assert np.isfinite(grad_ad)
    assert np.isfinite(grad_fd)
    assert grad_ad == pytest.approx(grad_fd, rel=1.0e-1, abs=1.0e-6)


def test_strict_update_accepted_step_reconstructs_first_qh_step(load_case_qh_warm_start):
    pytest.importorskip("jax")

    from vmec_jax.discrete_adjoint import strict_update_accepted_step
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.solve import solve_fixed_boundary_residual_iter
    from vmec_jax.state import pack_state

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    result = solve_fixed_boundary_residual_iter(
        state_guess,
        static,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        max_iter=1,
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        host_update_assembly=False,
        light_history=False,
        resume_state_mode="full",
        adjoint_trace=True,
    )
    trace = result.diagnostics["adjoint_step_trace"][0]
    out = strict_update_accepted_step(
        trace["state_pre"],
        static,
        dt_eff=trace["dt_eff"],
        b1=trace["b1"],
        fac=trace["fac"],
        force_scale=trace["force_scale"],
        flip_sign=trace["flip_sign"],
        vRcc_before=trace["vRcc_before"],
        vRss_before=trace["vRss_before"],
        vZsc_before=trace["vZsc_before"],
        vZcs_before=trace["vZcs_before"],
        vLsc_before=trace["vLsc_before"],
        vLcs_before=trace["vLcs_before"],
        frcc_u=trace["frcc_u"],
        frss_u=trace["frss_u"],
        fzsc_u=trace["fzsc_u"],
        fzcs_u=trace["fzcs_u"],
        flsc_u=trace["flsc_u"],
        flcs_u=trace["flcs_u"],
        max_update_rms=trace["max_update_rms_pre"],
        limit_update_rms=trace["limit_update_rms"],
        divide_by_scalxc_for_update=trace["divide_by_scalxc_for_update"],
    )
    assert np.asarray(pack_state(out["state_post"])) == pytest.approx(
        np.asarray(pack_state(trace["state_post"])),
        rel=0.0,
        abs=1.0e-12,
    )
    for key in ["vRcc_after", "vRss_after", "vZsc_after", "vZcs_after", "vLsc_after", "vLcs_after"]:
        assert np.asarray(out[key]) == pytest.approx(np.asarray(trace[key]), rel=0.0, abs=1.0e-12)


def test_strict_update_accepted_step_vjp_identity_qh(load_case_qh_warm_start):
    pytest.importorskip("jax")

    from vmec_jax._compat import enable_x64, jax, jnp
    from vmec_jax.discrete_adjoint import strict_update_accepted_step
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.solve import solve_fixed_boundary_residual_iter
    from vmec_jax.state import pack_state

    enable_x64(True)

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    result = solve_fixed_boundary_residual_iter(
        state_guess,
        static,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        max_iter=1,
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        host_update_assembly=False,
        light_history=False,
        resume_state_mode="full",
        adjoint_trace=True,
    )
    trace = result.diagnostics["adjoint_step_trace"][0]
    pieces0 = [
        jnp.asarray(trace["vRcc_before"]),
        jnp.asarray(trace["vRss_before"]),
        jnp.asarray(trace["vZsc_before"]),
        jnp.asarray(trace["vZcs_before"]),
        jnp.asarray(trace["vLsc_before"]),
        jnp.asarray(trace["vLcs_before"]),
        jnp.asarray(trace["frcc_u"]),
        jnp.asarray(trace["frss_u"]),
        jnp.asarray(trace["fzsc_u"]),
        jnp.asarray(trace["fzcs_u"]),
        jnp.asarray(trace["flsc_u"]),
        jnp.asarray(trace["flcs_u"]),
    ]
    sizes = [int(p.size) for p in pieces0]
    offsets = np.cumsum([0] + sizes)
    x0 = jnp.concatenate([jnp.ravel(p) for p in pieces0], axis=0)
    tangent = jnp.linspace(-0.25, 0.75, int(x0.size), dtype=x0.dtype)

    def _unpack(x):
        vals = []
        for start, stop, ref in zip(offsets[:-1], offsets[1:], pieces0):
            vals.append(jnp.reshape(x[start:stop], ref.shape))
        return vals

    def _f(x):
        vals = _unpack(x)
        out = strict_update_accepted_step(
            trace["state_pre"],
            static,
            dt_eff=trace["dt_eff"],
            b1=trace["b1"],
            fac=trace["fac"],
            force_scale=trace["force_scale"],
            flip_sign=trace["flip_sign"],
            vRcc_before=vals[0],
            vRss_before=vals[1],
            vZsc_before=vals[2],
            vZcs_before=vals[3],
            vLsc_before=vals[4],
            vLcs_before=vals[5],
            frcc_u=vals[6],
            frss_u=vals[7],
            fzsc_u=vals[8],
            fzcs_u=vals[9],
            flsc_u=vals[10],
            flcs_u=vals[11],
            max_update_rms=trace["max_update_rms_pre"],
            limit_update_rms=trace["limit_update_rms"],
            divide_by_scalxc_for_update=trace["divide_by_scalxc_for_update"],
        )
        y_pieces = [
            jnp.ravel(out["vRcc_after"]),
            jnp.ravel(out["vRss_after"]),
            jnp.ravel(out["vZsc_after"]),
            jnp.ravel(out["vZcs_after"]),
            jnp.ravel(out["vLsc_after"]),
            jnp.ravel(out["vLcs_after"]),
            pack_state(out["state_post"]),
        ]
        return jnp.concatenate(y_pieces, axis=0)

    y_ref = _f(x0)
    cotangent = jnp.linspace(-0.6, 0.5, int(y_ref.size), dtype=x0.dtype)
    _, jvp = jax.jvp(_f, (x0,), (tangent,))
    _, vjp_fun = jax.vjp(_f, x0)
    vjp = vjp_fun(cotangent)[0]
    lhs = float(jnp.vdot(jvp, cotangent))
    rhs = float(jnp.vdot(tangent, vjp))
    assert lhs == pytest.approx(rhs, rel=1.0e-10, abs=1.0e-10)


def test_strict_update_accepted_step_taylor_qh(load_case_qh_warm_start):
    pytest.importorskip("jax")

    from vmec_jax._compat import enable_x64, jax, jnp
    from vmec_jax.discrete_adjoint import strict_update_accepted_step
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.solve import solve_fixed_boundary_residual_iter
    from vmec_jax.state import pack_state

    enable_x64(True)

    _cfg, indata, static, boundary, _state0 = load_case_qh_warm_start
    state_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess, static).sqrtg), axis_index=1))
    result = solve_fixed_boundary_residual_iter(
        state_guess,
        static,
        indata=indata,
        signgs=signgs,
        ftol=float(indata.get_float("FTOL", 1e-14)),
        max_iter=1,
        step_size=float(indata.get_float("DELT", 1.0)),
        vmec2000_control=True,
        reference_mode=False,
        backtracking=True,
        limit_dt_from_force=True,
        limit_update_rms=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces="auto",
        use_scan=False,
        host_update_assembly=False,
        light_history=False,
        resume_state_mode="full",
        adjoint_trace=True,
    )
    trace = result.diagnostics["adjoint_step_trace"][0]
    pieces0 = [
        jnp.asarray(trace["vRcc_before"]),
        jnp.asarray(trace["vRss_before"]),
        jnp.asarray(trace["vZsc_before"]),
        jnp.asarray(trace["vZcs_before"]),
        jnp.asarray(trace["vLsc_before"]),
        jnp.asarray(trace["vLcs_before"]),
        jnp.asarray(trace["frcc_u"]),
        jnp.asarray(trace["frss_u"]),
        jnp.asarray(trace["fzsc_u"]),
        jnp.asarray(trace["fzcs_u"]),
        jnp.asarray(trace["flsc_u"]),
        jnp.asarray(trace["flcs_u"]),
    ]
    sizes = [int(p.size) for p in pieces0]
    offsets = np.cumsum([0] + sizes)
    x0 = jnp.concatenate([jnp.ravel(p) for p in pieces0], axis=0)
    direction = jnp.linspace(-0.4, 0.6, int(x0.size), dtype=x0.dtype)

    def _unpack(x):
        vals = []
        for start, stop, ref in zip(offsets[:-1], offsets[1:], pieces0):
            vals.append(jnp.reshape(x[start:stop], ref.shape))
        return vals

    def _f(x):
        vals = _unpack(x)
        out = strict_update_accepted_step(
            trace["state_pre"],
            static,
            dt_eff=trace["dt_eff"],
            b1=trace["b1"],
            fac=trace["fac"],
            force_scale=trace["force_scale"],
            flip_sign=trace["flip_sign"],
            vRcc_before=vals[0],
            vRss_before=vals[1],
            vZsc_before=vals[2],
            vZcs_before=vals[3],
            vLsc_before=vals[4],
            vLcs_before=vals[5],
            frcc_u=vals[6],
            frss_u=vals[7],
            fzsc_u=vals[8],
            fzcs_u=vals[9],
            flsc_u=vals[10],
            flcs_u=vals[11],
            max_update_rms=trace["max_update_rms_pre"],
            limit_update_rms=trace["limit_update_rms"],
            divide_by_scalxc_for_update=trace["divide_by_scalxc_for_update"],
        )
        return jnp.concatenate(
            [
                jnp.ravel(out["vRcc_after"]),
                jnp.ravel(out["vRss_after"]),
                jnp.ravel(out["vZsc_after"]),
                jnp.ravel(out["vZcs_after"]),
                jnp.ravel(out["vLsc_after"]),
                jnp.ravel(out["vLcs_after"]),
                pack_state(out["state_post"]),
            ],
            axis=0,
        )

    y0, jvp = jax.jvp(_f, (x0,), (direction,))
    eps = 1.0e-6
    y1 = _f(x0 + eps * direction)
    residual = np.asarray(y1 - y0 - eps * jvp)
    assert np.max(np.abs(residual)) < 1.0e-8
