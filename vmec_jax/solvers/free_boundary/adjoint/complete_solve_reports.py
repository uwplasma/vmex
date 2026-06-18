"""Complete-solve and fixed accepted-trace free-boundary validation reports."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

from vmec_jax._compat import jax, jnp
from vmec_jax.solvers.free_boundary.adjoint.boundary_replay import free_boundary_boundary_geometry_jax
from vmec_jax.solvers.free_boundary.adjoint.branch_metadata import (
    direct_coil_accepted_trace_branch_metadata,
)
from vmec_jax.solvers.free_boundary.adjoint.direct_coil_replay import (
    direct_coil_boundary_bsqvac_from_trace_jax,
)
from vmec_jax.solvers.free_boundary.adjoint.objectives import (
    tree_weighted_half_norm as _tree_weighted_half_norm,
    weighted_half_norm as _weighted_half_norm,
)
from vmec_jax.solvers.free_boundary.adjoint.replay_context import (
    direct_coil_boundary_replay_context,
    direct_coil_boundary_replay_context_for_shape,
    direct_coil_trace_boundary_shape as _direct_coil_trace_boundary_shape,
)
from vmec_jax.solvers.free_boundary.adjoint.replay_plan import (
    complete_solve_objective_values as _complete_solve_objective_values,
    extract_adjoint_step_trace as _extract_adjoint_step_trace,
    stackability_probe as _stackability_probe,
)
from vmec_jax.solvers.free_boundary.adjoint.runtime import jax_named_scope as _runtime_jax_named_scope
from vmec_jax.solvers.free_boundary.adjoint.trace_controls import _accepted_trace_reset_flags
from vmec_jax.solvers.free_boundary.adjoint.trace_stack import (
    direct_coil_accepted_trace_array_controls_jax,
    direct_coil_accepted_trace_preconditioner_controls_jax,
    direct_coil_accepted_trace_scalar_controls_jax,
)
from vmec_jax.solvers.free_boundary.adjoint.trace_fingerprint import (
    direct_coil_accepted_trace_fingerprint,
    direct_coil_accepted_trace_fingerprint_delta,
)
from vmec_jax.solvers.free_boundary.adjoint.trace_metadata import _json_safe_fingerprint_value


def _jax_named_scope(name: str) -> Any:
    return _runtime_jax_named_scope(name, jax_module=jax, nullcontext_factory=nullcontext)


def direct_coil_accepted_trace_replay_objective_jax(
    params: Any,
    initial_state: Any,
    *,
    static: Any,
    traces: Any,
    signgs: int,
    max_steps: int | None = None,
    sample_nzeta: int | None = None,
    include_analytic: bool = True,
    enforce_edge: bool = False,
    state_weight: Any = 1.0,
    force_weight: Any = 0.0,
    bsqvac_weight: Any = 0.0,
    coil_geometry: Any | None = None,
) -> dict[str, Any]:
    """Replay fixed accepted free-boundary traces with differentiable coils."""

    from vmec_jax.discrete_adjoint import strict_update_one_step_from_trace
    from vmec_jax.state import pack_state

    trace_seq = list(traces)
    if max_steps is not None:
        trace_seq = trace_seq[: int(max_steps)]
    if not trace_seq:
        raise ValueError("at least one accepted trace is required")
    reset_flags = _accepted_trace_reset_flags(trace_seq)

    state = initial_state
    objective_components: dict[str, Any] = {
        "state": jnp.asarray(0.0),
        "force": jnp.asarray(0.0),
        "bsqvac": jnp.asarray(0.0),
    }
    context_cache: dict[tuple[int, int], dict[str, Any]] = {}

    def _precomputed_context_for_trace(trace: Mapping[str, Any]) -> dict[str, Any] | None:
        shape = _direct_coil_trace_boundary_shape(trace)
        if shape is None:
            return None
        if shape not in context_cache:
            context_cache[shape] = direct_coil_boundary_replay_context_for_shape(
                static,
                ntheta=shape[0],
                nzeta=shape[1],
            )
        return context_cache[shape]

    steps: list[dict[str, Any]] = []
    bsqvac_values: list[Any] = []
    for trace, reset_to_trace_pre in zip(trace_seq, reset_flags, strict=True):
        if reset_to_trace_pre:
            # VMEC free-boundary turn-on/restart control can reset the working
            # state between accepted trace entries. Preserve that fixed host
            # control transition instead of incorrectly chaining state_post.
            state = trace["state_pre"]
        has_active_freeb_replay = trace.get("freeb_bsqvac_half") is not None and trace.get("freeb_nestor_trace") is not None
        if has_active_freeb_replay:
            with _jax_named_scope("vmec_jax.free_boundary.boundary_geometry"):
                geometry = free_boundary_boundary_geometry_jax(
                    state,
                    static,
                    sample_nzeta=sample_nzeta,
                )
            context = _precomputed_context_for_trace(trace)
            if context is None or tuple(int(v) for v in geometry["R"].shape) != (
                int(context["ntheta"]),
                int(context["nzeta"]),
            ):
                with _jax_named_scope("vmec_jax.free_boundary.replay_context"):
                    context = direct_coil_boundary_replay_context(static, geometry)
            with _jax_named_scope("vmec_jax.free_boundary.direct_coil_bsqvac_replay"):
                replay = direct_coil_boundary_bsqvac_from_trace_jax(
                    params,
                    geometry,
                    trace,
                    basis=context["basis"],
                    tables=context["tables"],
                    signgs=int(signgs),
                    nvper=int(context["nvper"]),
                    wint=jnp.asarray(context["wint"]),
                    include_analytic=bool(include_analytic),
                    coil_geometry=coil_geometry,
                )
            freeb_bsqvac_half = replay["bsqvac"]
        else:
            # Full accepted-trace replay must preserve non-vacuum/setup steps.
            # These steps do not have enough NESTOR metadata to resample coils,
            # so replay the original trace payload and keep coil derivatives
            # zero for that step.
            replay = None
            freeb_bsqvac_half = trace.get("freeb_bsqvac_half", None)
        with _jax_named_scope("vmec_jax.free_boundary.strict_update_one_step_from_trace"):
            step = strict_update_one_step_from_trace(
                state,
                static,
                trace,
                freeb_bsqvac_half=freeb_bsqvac_half,
                enforce_edge=bool(enforce_edge),
            )
        state = step["step"]["state_post"]
        steps.append(step)
        bsqvac_values.append(freeb_bsqvac_half)
        objective_components["force"] = objective_components["force"] + _tree_weighted_half_norm(
            step["force"],
            force_weight,
        )
        if replay is not None:
            objective_components["bsqvac"] = objective_components["bsqvac"] + _weighted_half_norm(
                replay["bsqvac"],
                bsqvac_weight,
            )

    objective_components["state"] = _weighted_half_norm(
        pack_state(state),
        state_weight,
    )
    objective = sum(objective_components.values())
    return {
        "objective": objective,
        "objective_components": objective_components,
        "state": state,
        "steps": steps,
        "bsqvac": bsqvac_values,
        "state_reset_flags": tuple(reset_flags),
    }


def free_boundary_adjoint_trace_replay_diagnostics(
    source: Any,
    *,
    accept_mask: Any | None = None,
    done_mask: Any | None = None,
    max_steps: int | None = None,
    json_safe: bool = False,
) -> dict[str, Any]:
    """Return diagnostics for fixed accepted-trace free-boundary replay."""

    traces = _extract_adjoint_step_trace(source)
    if max_steps is not None:
        traces = traces[: int(max_steps)]
    if not traces:
        raise RuntimeError(
            "adjoint_step_trace is empty. Run the residual solver with "
            "adjoint_trace=True and adjoint_trace_mode='full'."
        )
    metadata = direct_coil_accepted_trace_branch_metadata(
        traces,
        accept_mask=accept_mask,
        done_mask=done_mask,
        max_steps=max_steps,
        json_safe=False,
    )
    scalar_ok, scalar_error = _stackability_probe(
        "scalar_controls",
        direct_coil_accepted_trace_scalar_controls_jax,
        traces,
    )
    array_ok, array_error = _stackability_probe(
        "array_controls",
        direct_coil_accepted_trace_array_controls_jax,
        traces,
    )
    preconditioner_ok, preconditioner_error = _stackability_probe(
        "preconditioner_controls",
        direct_coil_accepted_trace_preconditioner_controls_jax,
        traces,
    )
    errors = {
        key: value
        for key, value in {
            "scalar_controls": scalar_error,
            "array_controls": array_error,
            "preconditioner_controls": preconditioner_error,
        }.items()
        if value is not None
    }
    diagnostics = {
        "contract": "fixed accepted-trace replay diagnostics only",
        "differentiates_adaptive_controller": False,
        "n_steps": metadata["n_steps"],
        "branch_fingerprint": metadata["fingerprint"],
        "masks": metadata["masks"],
        "replay_diagnostics": {
            "preconditioner_policy_n_segments": len(metadata["preconditioner_policy_segments"]),
            "preconditioner_policy_segment_summary": metadata["preconditioner_policy_segment_summary"],
            "scalar_controls_stackable": bool(scalar_ok),
            "array_controls_stackable": bool(array_ok),
            "preconditioner_controls_stackable": bool(preconditioner_ok),
            "errors": errors,
        },
    }
    if json_safe:
        return _json_safe_fingerprint_value(diagnostics)
    return diagnostics


def direct_coil_complete_solve_trace(
    input_path: Any,
    params: Any,
    *,
    init_kwargs: dict[str, Any] | None = None,
    solve_kwargs: dict[str, Any] | None = None,
    require_active_trace: bool = True,
) -> dict[str, Any]:
    """Run a direct-coil free-boundary solve and return accepted traces."""

    from vmec_jax.driver import run_free_boundary
    from vmec_jax.solve import solve_fixed_boundary_residual_iter

    init_options: dict[str, Any] = {
        "use_initial_guess": True,
        "verbose": False,
        "external_field_provider_kind": "direct_coils",
        "external_field_provider_params": params,
    }
    if init_kwargs:
        init_options.update(init_kwargs)
    init = run_free_boundary(input_path, **init_options)

    solve_options: dict[str, Any] = {
        "max_iter": 2,
        "ftol": 1.0e-8,
        "vmec2000_control": True,
        "auto_flip_force": False,
        "use_direct_fallback": True,
        "verbose": False,
        "verbose_vmec2000_table": False,
        "jit_forces": False,
        "use_scan": False,
        "host_update_assembly": False,
        "adjoint_trace": True,
        "adjoint_trace_mode": "full",
        "external_field_provider_kind": "direct_coils",
        "external_field_provider_params": params,
        "free_boundary_activate_fsq": 1.0e99,
    }
    if solve_kwargs:
        solve_options.update(solve_kwargs)
    solve_options["external_field_provider_params"] = params
    result = solve_fixed_boundary_residual_iter(
        init.state,
        init.static,
        indata=init.indata,
        signgs=init.signgs,
        **solve_options,
    )
    traces = list(result.diagnostics.get("adjoint_step_trace", []))
    if not traces:
        raise RuntimeError("direct-coil solve did not record adjoint_step_trace")
    active_trace = any(trace.get("freeb_bsqvac_half") is not None for trace in traces)
    if bool(require_active_trace) and not active_trace:
        raise RuntimeError("direct-coil solve did not record an active free-boundary trace")
    return {
        "init": init,
        "result": result,
        "traces": traces,
        "params": params,
        "active_trace": bool(active_trace),
    }


def direct_coil_same_branch_complete_solve_fd_report(
    input_path: Any,
    base_params: Any,
    *,
    params_for: Any,
    objective_fn: Any,
    eps: float = 1.0e-4,
    init_kwargs: dict[str, Any] | None = None,
    solve_kwargs: dict[str, Any] | None = None,
    fingerprint_rtol: float = 1.0e-6,
    fingerprint_atol: float = 1.0e-9,
    require_active_trace: bool = True,
) -> dict[str, Any]:
    """Return same-branch complete-solve finite-difference diagnostics."""

    from vmec_jax.discrete_adjoint import residual_branch_fingerprint

    eps_f = float(eps)
    if eps_f == 0.0:
        raise ValueError("eps must be nonzero")
    base = direct_coil_complete_solve_trace(
        input_path,
        base_params,
        init_kwargs=init_kwargs,
        solve_kwargs=solve_kwargs,
        require_active_trace=require_active_trace,
    )
    plus = direct_coil_complete_solve_trace(
        input_path,
        params_for(eps_f),
        init_kwargs=init_kwargs,
        solve_kwargs=solve_kwargs,
        require_active_trace=require_active_trace,
    )
    minus = direct_coil_complete_solve_trace(
        input_path,
        params_for(-eps_f),
        init_kwargs=init_kwargs,
        solve_kwargs=solve_kwargs,
        require_active_trace=require_active_trace,
    )
    plus_branch = direct_coil_accepted_trace_fingerprint_delta(
        base["traces"],
        plus["traces"],
        rtol=float(fingerprint_rtol),
        atol=float(fingerprint_atol),
    )
    minus_branch = direct_coil_accepted_trace_fingerprint_delta(
        base["traces"],
        minus["traces"],
        rtol=float(fingerprint_rtol),
        atol=float(fingerprint_atol),
    )
    base_fingerprint = direct_coil_accepted_trace_fingerprint(base["traces"])
    plus_fingerprint = direct_coil_accepted_trace_fingerprint(plus["traces"])
    minus_fingerprint = direct_coil_accepted_trace_fingerprint(minus["traces"])
    base_residual_fingerprint = residual_branch_fingerprint(base["result"])
    plus_residual_fingerprint = residual_branch_fingerprint(plus["result"])
    minus_residual_fingerprint = residual_branch_fingerprint(minus["result"])
    same_residual_branch = bool(
        base_residual_fingerprint == plus_residual_fingerprint
        and base_residual_fingerprint == minus_residual_fingerprint
    )
    trace_replay_diagnostics = {
        "base": free_boundary_adjoint_trace_replay_diagnostics(base["traces"]),
        "plus": free_boundary_adjoint_trace_replay_diagnostics(plus["traces"]),
        "minus": free_boundary_adjoint_trace_replay_diagnostics(minus["traces"]),
    }
    base_values = _complete_solve_objective_values(objective_fn(base))
    plus_values = _complete_solve_objective_values(objective_fn(plus))
    minus_values = _complete_solve_objective_values(objective_fn(minus))
    if base_values.keys() != plus_values.keys() or base_values.keys() != minus_values.keys():
        raise ValueError("objective_fn returned different scalar keys for base/plus/minus solves")
    primary_key = "objective" if "objective" in base_values else next(iter(base_values))
    objective_values = {
        key: {
            "base": float(base_values[key]),
            "plus": float(plus_values[key]),
            "minus": float(minus_values[key]),
            "central_fd_directional": float((plus_values[key] - minus_values[key]) / (2.0 * eps_f)),
        }
        for key in base_values
    }
    return {
        "base": base,
        "plus": plus,
        "minus": minus,
        "branch_compatibility": {
            "same_branch": bool(plus_branch["compatible"] and minus_branch["compatible"] and same_residual_branch),
            "same_accepted_trace_branch": bool(plus_branch["compatible"] and minus_branch["compatible"]),
            "same_residual_branch": same_residual_branch,
            "plus": plus_branch,
            "minus": minus_branch,
            "base_fingerprint": base_fingerprint,
            "plus_fingerprint": plus_fingerprint,
            "minus_fingerprint": minus_fingerprint,
            "base_residual_fingerprint": base_residual_fingerprint,
            "plus_residual_fingerprint": plus_residual_fingerprint,
            "minus_residual_fingerprint": minus_residual_fingerprint,
        },
        "trace_replay_diagnostics": trace_replay_diagnostics,
        "primary_objective": primary_key,
        "values": objective_values[primary_key],
        "objective_values": objective_values,
    }
