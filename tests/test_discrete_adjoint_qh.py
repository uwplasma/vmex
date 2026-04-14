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
