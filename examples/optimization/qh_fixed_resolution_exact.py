#!/usr/bin/env python

"""Optimize a VMEC-JAX equilibrium for quasi-helical symmetry.

This standalone example mirrors the SIMSOPT fixed-resolution QH workflow, but
stays entirely inside vmec_jax. It uses the recovered exact discrete-adjoint
Jacobian path, not finite differences.

Usage
-----
Run with defaults (max_mode=1, max_nfev=20, ftol=gtol=xtol=1e-3)::

    python examples/optimization/qh_fixed_resolution_exact.py

Write wout files and objective history to a custom directory::

    python examples/optimization/qh_fixed_resolution_exact.py \\
        --output-dir results/qh_opt \\
        --max-mode 1 --max-nfev 20

The script saves:
  - ``results/qh_opt/wout_initial.nc``   — equilibrium at the start point
  - ``results/qh_opt/wout_final.nc``     — equilibrium at the optimized point
  - ``results/qh_opt/history.json``      — objective/cost history per iteration
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import vmec_jax as vj


def _parse_args():
    p = argparse.ArgumentParser(
        description="QH fixed-resolution exact discrete-adjoint optimization"
    )
    p.add_argument("--max-nfev", type=int, default=20,
                   help="Maximum number of function evaluations (default: 20)")
    p.add_argument("--max-mode", type=int, default=1,
                   help="Maximum |m|,|n| mode numbers for boundary DOFs (default: 1)")
    p.add_argument("--ftol", type=float, default=1e-3,
                   help="Relative cost reduction tolerance (default: 1e-3)")
    p.add_argument("--gtol", type=float, default=1e-3,
                   help="Gradient infinity-norm tolerance (default: 1e-3)")
    p.add_argument("--xtol", type=float, default=1e-3,
                   help="Step norm tolerance (default: 1e-3)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Directory for wout_initial.nc, wout_final.nc, history.json")
    return p.parse_args()


def _state_to_run(state, static, indata, flux, signgs):
    """Wrap a bare VMECState in a minimal FixedBoundaryRun for wout writing."""
    from vmec_jax.driver import FixedBoundaryRun
    return FixedBoundaryRun(
        cfg=static.cfg,
        indata=indata,
        static=static,
        state=state,
        result=None,
        flux=flux,
        profiles={},
        signgs=signgs,
    )


def _write_wout(path: Path, state, static, indata, flux, signgs) -> None:
    """Write a wout NetCDF file for the given converged state.

    Uses fast_bcovar=True which calls vmec_bcovar_half_mesh instead of the
    full force-constraint path, avoiding a grid-size mismatch on some inputs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    run = _state_to_run(state, static, indata, flux, signgs)
    vj.write_wout_from_fixed_boundary_run(str(path), run, include_fsq=False, fast_bcovar=True)
    print(f"  Wrote {path}")


def main() -> None:
    args = _parse_args()
    max_nfev = int(args.max_nfev)
    max_mode = int(args.max_mode)
    ftol = float(args.ftol)
    gtol = float(args.gtol)
    xtol = float(args.xtol)
    outdir = Path(args.output_dir) if args.output_dir else None

    from vmec_jax._compat import enable_x64, jax, jnp
    from vmec_jax.field import signgs_from_sqrtg
    from vmec_jax.geom import eval_geom
    from vmec_jax.init_guess import extract_axis_override_from_state, initial_guess_from_boundary
    from vmec_jax.state import pack_state, unpack_state

    enable_x64(True)
    os.environ.setdefault("VMEC_JAX_DYNAMIC_REPLAY_BUCKET", "1024")

    print("Running examples/optimization/qh_fixed_resolution_exact.py")
    print("==========================================================")
    print(f"  max_mode={max_mode}  max_nfev={max_nfev}  "
          f"ftol={ftol:.0e}  gtol={gtol:.0e}  xtol={xtol:.0e}")

    filename = Path(__file__).resolve().parents[1] / "data" / "input.nfp4_QH_warm_start"
    cfg, indata = vj.load_config(str(filename))
    static = vj.build_static(cfg)
    boundary = vj.boundary_from_indata(indata, static.modes)

    def _last_array_value(key: str, scalar_key: str, default, cast):
        value = indata.get(key, None)
        if isinstance(value, list) and value:
            return cast(value[-1])
        return cast(indata.get(scalar_key, default))

    inner_max_iter = int(_last_array_value("NITER_ARRAY", "NITER", 1500, int))
    inner_ftol = float(_last_array_value("FTOL_ARRAY", "FTOL", 1e-13, float))
    step_size = float(indata.get_float("DELT", 1.0))

    # Define parameter space:
    specs = vj.boundary_param_specs(
        boundary,
        static.modes,
        max_mode=max_mode,
        min_coeff=0.0,
        include=("rc", "zs"),
        fix=("rc00",),
    )
    params0 = jnp.zeros((len(specs),), dtype=jnp.float64)
    print("Parameter space:", vj.boundary_param_names(specs))

    state_guess0 = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
    signgs = int(signgs_from_sqrtg(np.asarray(eval_geom(state_guess0, static).sqrtg), axis_index=1))
    flux = vj.flux_profiles_from_indata(indata, static.s, signgs=signgs)
    pressure = jnp.zeros_like(jnp.asarray(static.s))
    layout = state_guess0.layout

    def residuals_from_state(state):
        aspect = vj.equilibrium_aspect_ratio_from_state(state=state, static=static)
        qs = vj.quasisymmetry_ratio_residual_from_state(
            state=state,
            static=static,
            indata=indata,
            signgs=signgs,
            flux_local=flux,
            prof_local={"pressure": pressure},
            pressure_local=pressure,
            surfaces=np.arange(0, 1.01, 0.1),
            helicity_m=1,
            helicity_n=-1,
        )
        aspect_residual = jnp.asarray([aspect - 7.0], dtype=jnp.float64)
        return jnp.concatenate([aspect_residual, jnp.asarray(qs["residuals1d"], dtype=jnp.float64)])

    def total_from_residual(residual):
        residual = np.asarray(residual, dtype=float).reshape(-1)
        return float(np.dot(residual, residual))

    def _boundary_from_params(params):
        return vj.apply_boundary_params(boundary, specs, jnp.asarray(params, dtype=jnp.float64))

    # ── Solver keyword arguments ──────────────────────────────────────────────
    # These mirror the vmec_jax driver's vmec2000_iter path: strict_update=True,
    # backtracking=False (standard VMEC algorithm without a line-search accept
    # step — the adaptive time-step handles oscillations).  Using backtracking=True
    # here causes the step to get stuck at dt ~ machine-epsilon for this problem,
    # resulting in 1500 non-convergent iterations instead of ~500 tight ones.
    _base_solver_kwargs = dict(
        indata=indata,
        signgs=signgs,
        step_size=step_size,
        include_constraint_force=True,
        apply_m1_constraints=True,
        precond_radial_alpha=0.5,
        precond_lambda_alpha=0.5,
        mode_diag_exponent=0.0,
        auto_flip_force=False,
        divide_by_scalxc_for_update=False,
        lambda_update_scale=1.0,
        enforce_vmec_lambda_axis=True,
        vmec2000_control=True,
        strict_update=True,
        backtracking=False,
        reference_mode=False,
        use_restart_triggers=True,
        verbose=False,
        verbose_vmec2000_table=False,
        jit_forces=True,
        use_scan=False,
        light_history=True,
        resume_state_mode="full",
    )
    # Tight settings for the Jacobian / accepted-step solves.
    # NOTE: max_iter and ftol are NOT included here — they are passed as explicit
    # arguments to solve_fixed_boundary_residual_iter and
    # build_residual_checkpoint_tape_direct.  Including them in solver_kwargs would
    # cause a "multiple values" TypeError when the tape builder also passes max_iter.
    _exact_solver_kwargs = dict(_base_solver_kwargs)
    _exact_max_iter = inner_max_iter
    _exact_ftol = inner_ftol
    # Relaxed settings for forward-only trial residuals used in line search.
    # The line search only needs a good enough equilibrium, not a fully converged
    # one.  Using fewer iterations here halves the line-search overhead.
    _trial_max_iter = min(inner_max_iter, 800)
    _trial_ftol = max(inner_ftol, 1e-10)
    _trial_solver_kwargs = dict(
        _base_solver_kwargs,
        jit_forces=False,  # Line-search trials don't need JIT warmup overhead
    )

    def solve_forward_state(params, *, trial: bool = False):
        """Solve equilibrium for given boundary params.

        When ``trial=True`` uses a relaxed budget suitable for Gauss-Newton
        line-search forward evaluations.
        """
        boundary_now = _boundary_from_params(params)
        state0 = initial_guess_from_boundary(static, boundary_now, indata, vmec_project=True)
        if trial:
            result = vj.solve_fixed_boundary_residual_iter(
                state0, static, max_iter=_trial_max_iter, ftol=_trial_ftol, **_trial_solver_kwargs
            )
        else:
            result = vj.solve_fixed_boundary_residual_iter(
                state0, static, max_iter=_exact_max_iter, ftol=_exact_ftol, **_exact_solver_kwargs
            )
        return result.state

    # Cache for the last exact solve — avoids building the tape twice per accepted
    # Gauss-Newton step (once in residual_fun and once in jacobian_fun at the same x).
    _exact_cache: dict = {}

    def solve_exact_state(params, *, return_payload: bool = False):
        params_arr = np.asarray(params, dtype=float)
        cache_key = params_arr.tobytes()
        if cache_key in _exact_cache:
            state, payload = _exact_cache[cache_key]
            return (state, payload) if return_payload else state

        boundary_now = _boundary_from_params(params)
        state0 = initial_guess_from_boundary(static, boundary_now, indata, vmec_project=True)
        axis_override = extract_axis_override_from_state(state0, static)
        tape = vj.build_residual_checkpoint_tape_direct(
            state0,
            static,
            max_iter=inner_max_iter,
            solver_kwargs=_exact_solver_kwargs,
            indata=indata,
            signgs=signgs,
            ftol=inner_ftol,
            step_size=step_size,
            light_history=True,
            store_trace=False,
            store_full_step_traces=False,
        )
        state = unpack_state(jnp.asarray(tape.final_packed_state, dtype=jnp.float64), layout)
        payload = {"tape": tape, "axis_override": axis_override}
        _exact_cache.clear()  # keep only the last entry to bound memory usage
        _exact_cache[cache_key] = (state, payload)
        return (state, payload) if return_payload else state

    def residual_fun(params):
        state = solve_exact_state(params)
        return np.asarray(residuals_from_state(state), dtype=float)

    def forward_residual_fun(params):
        # Used for line-search trial points: relaxed solve budget is sufficient.
        state = solve_forward_state(params, trial=True)
        return np.asarray(residuals_from_state(state), dtype=float)

    def jacobian_fun(params):
        params = jnp.asarray(params, dtype=jnp.float64)
        state, payload = solve_exact_state(params, return_payload=True)
        packed_final = jnp.asarray(pack_state(state), dtype=jnp.float64)

        def _initial_state_packed(p):
            boundary_now = _boundary_from_params(p)
            state0 = initial_guess_from_boundary(
                static,
                boundary_now,
                indata,
                vmec_project=True,
                axis_override=payload["axis_override"],
            )
            return jnp.asarray(pack_state(state0), dtype=jnp.float64)

        def _residuals_from_packed(packed):
            return residuals_from_state(unpack_state(packed, layout))

        directions = jnp.eye(int(params.size), dtype=params.dtype)
        _, initial_linear = jax.linearize(_initial_state_packed, params)
        initial_tangents = jax.vmap(initial_linear)(directions)
        final_tangents = vj.checkpoint_tape_state_jvp_columns(
            tape=payload["tape"],
            static=static,
            initial_tangents=initial_tangents,
            rebuild_preconditioner=True,
        )
        _, residual_linear = jax.linearize(_residuals_from_packed, packed_final)
        columns = jax.vmap(residual_linear)(final_tangents)
        return np.asarray(columns, dtype=float).T

    # ── Objective history tracking ─────────────────────────────────────────────
    _history: list[dict] = []
    _wall_t0 = time.perf_counter()

    def _post_jacobian_clear():
        """Release replay/preconditioner JIT caches between Jacobian calls."""
        vj.clear_replay_scan_caches()
        vj.clear_preconditioner_jit_caches()

    _last_jacobian_key: list = [None]

    def _qs_total_from_residual(residual) -> float:
        residual = np.asarray(residual, dtype=float).reshape(-1)
        return float(np.dot(residual[1:], residual[1:]))

    def _history_entry(residual, *, aspect: float, wall_time_s: float) -> dict:
        objective = total_from_residual(residual)
        return {
            "wall_time_s": float(wall_time_s),
            "cost": 0.5 * objective,
            "objective": objective,
            "qs_objective": _qs_total_from_residual(residual),
            "aspect": float(aspect),
        }

    def _jacobian_fun_tracked(params):
        _last_jacobian_key[0] = np.asarray(params, dtype=float).tobytes()
        jac = jacobian_fun(params)
        # Record accepted-step info in history after Jacobian is computed.
        key = _last_jacobian_key[0]
        if key is not None and key in _exact_cache:
            cached_state, _ = _exact_cache[key]
            res = np.asarray(residuals_from_state(cached_state), dtype=float)
            aspect = float(np.asarray(
                vj.equilibrium_aspect_ratio_from_state(state=cached_state, static=static)
            ))
            _history.append(_history_entry(res, aspect=aspect, wall_time_s=time.perf_counter() - _wall_t0))
        return jac

    def _exact_residual_after_jacobian():
        key = _last_jacobian_key[0]
        if key is None or key not in _exact_cache:
            return None
        cached_state, _ = _exact_cache[key]
        return np.asarray(residuals_from_state(cached_state), dtype=float)

    # ── Initial evaluation ─────────────────────────────────────────────────────
    residual0 = residual_fun(params0)
    # Reuse the exact state from the cache (residual_fun already built the tape).
    state0, _ = solve_exact_state(np.asarray(params0, dtype=float), return_payload=True)
    aspect0 = float(np.asarray(vj.equilibrium_aspect_ratio_from_state(state=state0, static=static)))
    cost0 = total_from_residual(residual0)
    qs_total0 = _qs_total_from_residual(residual0)

    print(f"Aspect ratio before optimization:        {aspect0:.4f}")
    print(f"Quasisymmetry objective before:          {qs_total0:.6f}")
    print(f"Total objective before optimization:     {cost0:.6f}")

    # Record the initial point.
    _history.append(_history_entry(residual0, aspect=aspect0, wall_time_s=0.0))

    # Write initial wout if requested.
    if outdir is not None:
        _write_wout(outdir / "wout_initial.nc", state0, static, indata, flux, signgs)

    # ── Gauss-Newton optimization ──────────────────────────────────────────────
    t_opt_start = time.perf_counter()

    result = vj.gauss_newton_least_squares(
        residual_fun,
        _jacobian_fun_tracked,
        np.asarray(params0, dtype=float),
        forward_residual_fun=forward_residual_fun,
        post_jacobian_callback=_post_jacobian_clear,
        exact_residual_after_jacobian_fun=_exact_residual_after_jacobian,
        max_nfev=max_nfev,
        ftol=ftol,
        gtol=gtol,
        xtol=xtol,
        verbose=1,
    )

    t_opt_total = time.perf_counter() - t_opt_start

    # Final cache clear after the GN loop.
    _post_jacobian_clear()

    # Try to reuse the cached exact state for the final display.  The last
    # accepted GN step will have stored the final x in _exact_cache.  If the
    # cache key matches we skip the extra forward solve entirely; otherwise we
    # fall back to a fresh forward solve (non-trial, tight budget).
    _final_key = np.asarray(result["x"], dtype=float).tobytes()
    if _final_key in _exact_cache:
        state_final = _exact_cache[_final_key][0]
    else:
        state_final = solve_forward_state(result["x"], trial=False)
    residual_final = np.asarray(residuals_from_state(state_final), dtype=float)
    aspect_final = float(np.asarray(
        vj.equilibrium_aspect_ratio_from_state(state=state_final, static=static)
    ))
    cost_final = total_from_residual(residual_final)
    qs_total_final = _qs_total_from_residual(residual_final)

    print()
    print(f"Optimization complete in {t_opt_total:.1f} s  "
          f"({result['nfev']} residual evals, {result['njev']} Jacobian evals)")
    print(f"Termination: {result['message']}")
    print(f"Aspect ratio after optimization:         {aspect_final:.4f}")
    print(f"Quasisymmetry objective after:           {qs_total_final:.6f}")
    print(f"Total objective after optimization:      {cost_final:.6f}")
    print(f"Objective reduction:                     "
          f"{100.0 * (1.0 - cost_final / cost0):.1f}%")
    print("End of examples/optimization/qh_fixed_resolution_exact.py")
    print("=======================================================")

    # Write final wout and history if requested.
    if outdir is not None:
        _write_wout(outdir / "wout_final.nc", state_final, static, indata, flux, signgs)

        # Append final point to history (may duplicate last jacobian entry if
        # the last accepted step is already there, but that's harmless).
        _history.append(_history_entry(residual_final, aspect=aspect_final, wall_time_s=t_opt_total))

        hist_path = outdir / "history.json"
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(hist_path, "w") as f:
            json.dump(
                {
                    "max_mode": max_mode,
                    "max_nfev": max_nfev,
                    "ftol": ftol,
                    "gtol": gtol,
                    "xtol": xtol,
                    "total_wall_time_s": t_opt_total,
                    "nfev": result["nfev"],
                    "njev": result["njev"],
                    "success": result["success"],
                    "message": result["message"],
                    "objective_initial": cost0,
                    "objective_final": cost_final,
                    "qs_initial": qs_total0,
                    "qs_final": qs_total_final,
                    "aspect_initial": aspect0,
                    "aspect_final": aspect_final,
                    "history": _history,
                },
                f,
                indent=2,
            )
        print(f"  Wrote {hist_path}")


if __name__ == "__main__":
    main()
