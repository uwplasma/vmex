"""Validated branch-local derivatives for direct-coil free-boundary solves.

The public functions in this module expose the current production derivative
contract for single-stage coil optimization:

``coil parameters -> direct Biot-Savart field -> complete free-boundary solve -> physical scalars``.

Values always come from complete ``run_free_boundary`` solves.  Derivatives are
computed by replaying the same accepted free-boundary branch and should be used
for optimizer proposals only; complete solves remain the acceptance authority.
Central finite-difference validation is available through the same public API
and checks that the accepted/rejected branch fingerprint did not change.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from vmec_jax._compat import jax, jnp
from vmec_jax.external_fields import CoilFieldParams
from vmec_jax.solvers.free_boundary.adjoint.branch_local_derivatives import (
    direct_coil_branch_local_scalars_report_from_complete_fd,
    direct_coil_run_free_boundary_branch_local_scalars_value_and_jacobian_jax,
    direct_coil_same_branch_complete_solve_fd_report,
)
from vmec_jax.solvers.free_boundary.coil_optimization import (
    SUPPORTED_SAME_BRANCH_VECTOR_KEYS,
    same_branch_scalar_function_registry,
)
from vmec_jax.quasisymmetry import quasisymmetry_angle_cache_from_static


PUBLIC_OUTPUT_ALIASES: dict[str, str] = {
    "aspect": "aspect",
    "mean_iota": "mean_iota",
    "iota": "mean_iota",
    "qs": "qs_total",
    "qs_residual": "qs_total",
    "qs_total": "qs_total",
    "bnormal_rms": "accepted_bnormal_rms",
    "accepted_bnormal_rms": "accepted_bnormal_rms",
    "boundary_displacement": "lcfs_boundary_moment",
    "boundary_moment": "lcfs_boundary_moment",
    "lcfs_boundary_moment": "lcfs_boundary_moment",
    "state_norm": "state_norm",
    "betatotal": "betatotal",
    "boozer_qs_total": "boozer_qs_total",
}

DEFAULT_FREE_BOUNDARY_DERIVATIVE_OUTPUTS = (
    "aspect",
    "mean_iota",
    "boundary_displacement",
    "bnormal_rms",
    "qs_residual",
)


@dataclass(frozen=True)
class FreeBoundaryDerivativeOptions:
    """Physics and replay options for branch-local free-boundary derivatives.

    Parameters configure the physical scalar definitions and the fixed-branch
    replay path.  They do not change the production claim: adaptive branch
    changes are validated by finite differences but are not differentiated.
    """

    helicity_m: int = 1
    helicity_n: int = 0
    qs_surfaces: tuple[float, ...] = (0.5,)
    qs_ntheta: int = 16
    qs_nphi: int = 16
    same_branch_boozer_mboz: int = 8
    same_branch_boozer_nboz: int = 8
    same_branch_boozer_normalize: bool = True
    replay_ad_mode: str = "direct"
    require_active_trace: bool = True
    replay_kwargs: Mapping[str, Any] | None = None


def canonical_free_boundary_output_keys(outputs: Sequence[str]) -> tuple[tuple[str, ...], dict[str, str]]:
    """Return internal scalar keys and a public-to-internal alias map.

    The public names are intentionally readable in examples.  Internally they
    map to the scalar registry shared with the same-branch validation reports.
    """

    public_to_internal: dict[str, str] = {}
    internal_keys: list[str] = []
    for output in outputs:
        public = str(output).strip()
        if not public:
            raise ValueError("free-boundary derivative output names must be non-empty")
        internal = PUBLIC_OUTPUT_ALIASES.get(public)
        if internal is None:
            supported = ", ".join(sorted(PUBLIC_OUTPUT_ALIASES))
            raise ValueError(f"unsupported free-boundary derivative output {public!r}; supported: {supported}")
        if internal not in SUPPORTED_SAME_BRANCH_VECTOR_KEYS:
            supported_internal = ", ".join(SUPPORTED_SAME_BRANCH_VECTOR_KEYS)
            raise ValueError(f"internal output {internal!r} is not supported; supported: {supported_internal}")
        public_to_internal[public] = internal
        if internal not in internal_keys:
            internal_keys.append(internal)
    return tuple(internal_keys), public_to_internal


def _params_for_direction(base_params: CoilFieldParams, direction_params: CoilFieldParams):
    """Return a central-FD callback along one coil-parameter direction."""

    def params_for(scale: float) -> CoilFieldParams:
        return base_params.with_arrays(
            base_curve_dofs=jnp.asarray(base_params.base_curve_dofs)
            + float(scale) * jnp.asarray(direction_params.base_curve_dofs),
            base_currents=jnp.asarray(base_params.base_currents)
            + float(scale) * jnp.asarray(direction_params.base_currents),
        )

    return params_for


def _qs_angle_cache_factory(args: Any):
    """Return a per-static quasisymmetry angle cache for repeated scalar calls."""

    cache: dict[tuple[int, ...], dict[str, object]] = {}

    def qs_angle_cache_for_static(static: Any) -> dict[str, object]:
        cfg = static.cfg
        key = (
            int(cfg.nfp),
            int(cfg.mpol),
            int(cfg.ntor),
            int(cfg.ntheta),
            int(cfg.nzeta),
            int(args.qs_ntheta),
            int(args.qs_nphi),
        )
        if key not in cache:
            cache[key] = quasisymmetry_angle_cache_from_static(
                static,
                ntheta=int(args.qs_ntheta),
                nphi=int(args.qs_nphi),
            )
        return cache[key]

    return qs_angle_cache_for_static


def _options_args(options: FreeBoundaryDerivativeOptions) -> SimpleNamespace:
    """Return the small namespace expected by the shared scalar registry."""

    return SimpleNamespace(
        helicity_m=int(options.helicity_m),
        helicity_n=int(options.helicity_n),
        qs_ntheta=int(options.qs_ntheta),
        qs_nphi=int(options.qs_nphi),
        same_branch_boozer_mboz=int(options.same_branch_boozer_mboz),
        same_branch_boozer_nboz=int(options.same_branch_boozer_nboz),
        same_branch_boozer_normalize=bool(options.same_branch_boozer_normalize),
    )


def _public_projection(report_map: Mapping[str, Any], public_to_internal: Mapping[str, str]) -> dict[str, Any]:
    """Project an internal scalar mapping back onto requested public names."""

    projected: dict[str, Any] = {}
    for public, internal in public_to_internal.items():
        if internal in report_map:
            projected[public] = report_map[internal]
    return projected


def free_boundary_value_and_jvp(
    input_path: Any,
    params: CoilFieldParams,
    *,
    direction_params: CoilFieldParams | None = None,
    outputs: Sequence[str] = DEFAULT_FREE_BOUNDARY_DERIVATIVE_OUTPUTS,
    options: FreeBoundaryDerivativeOptions | None = None,
    complete_payload: Mapping[str, Any] | None = None,
    init_kwargs: Mapping[str, Any] | None = None,
    solve_kwargs: Mapping[str, Any] | None = None,
    replay_payload: Mapping[str, Any] | None = None,
    replay_plan: Mapping[str, Any] | None = None,
    production_values: Mapping[str, Any] | None = None,
    validate_fd: bool = False,
    fd_epsilon: float = 1.0e-4,
    fingerprint_rtol: float = 1.0e-6,
    fingerprint_atol: float = 1.0e-9,
    rtol: Mapping[str, float] | float | None = None,
    atol: Mapping[str, float] | float | None = None,
    base_value_atol: Mapping[str, float] | float | None = None,
    include_payload: bool = False,
    include_replay_graph_metadata: bool = True,
) -> dict[str, Any]:
    """Return complete-solve values and branch-local derivatives.

    If ``direction_params`` is supplied the report contains directional JVPs.
    Otherwise it contains a row-stacked pytree Jacobian and per-output VJP
    gradients.  Setting ``validate_fd=True`` adds a complete-solve central-FD
    comparison along ``direction_params`` and a same-branch pass/fail report.
    """

    if validate_fd and direction_params is None:
        raise ValueError("validate_fd=True requires direction_params for the central-FD path")
    options = options or FreeBoundaryDerivativeOptions()
    internal_keys, public_to_internal = canonical_free_boundary_output_keys(outputs)
    args = _options_args(options)
    scalar_value_fns, scalar_replay_fns = same_branch_scalar_function_registry(
        args=args,
        qs_surfaces=tuple(float(value) for value in options.qs_surfaces),
        qs_angle_cache_for_static=_qs_angle_cache_factory(args),
    )
    replay_kwargs = dict(options.replay_kwargs or {})
    complete_solve_kwargs = dict(solve_kwargs or {})

    branch_local = direct_coil_run_free_boundary_branch_local_scalars_value_and_jacobian_jax(
        input_path=input_path,
        params=params,
        direction_params=direction_params,
        complete_payload=complete_payload,
        scalar_keys=internal_keys,
        production_values=production_values,
        replay_payload=replay_payload,
        replay_plan=replay_plan,
        scalar_fn=lambda payload: {key: scalar_value_fns[key](payload) for key in internal_keys},
        replay_scalar_fns={key: scalar_replay_fns[key] for key in internal_keys},
        init_kwargs=dict(init_kwargs or {}),
        solve_kwargs=complete_solve_kwargs,
        replay_kwargs=replay_kwargs,
        replay_ad_mode=str(options.replay_ad_mode),
        include_trace_replay_diagnostics=bool(validate_fd),
        include_payload=bool(include_payload),
        include_replay_graph_metadata=bool(include_replay_graph_metadata),
        require_active_trace=bool(options.require_active_trace),
    )

    result = {
        "contract": "complete-solve values with same-branch branch-local derivatives",
        "differentiates_adaptive_controller": False,
        "differentiates_fixed_accepted_branch": True,
        "scalar_keys": internal_keys,
        "requested_outputs": tuple(str(output) for output in outputs),
        "public_to_internal": dict(public_to_internal),
        "values": _public_projection(branch_local["values"], public_to_internal),
        "internal_values": branch_local["values"],
        "replay_values": _public_projection(branch_local["replay_value_map"], public_to_internal),
        "internal_replay_values": branch_local["replay_value_map"],
        "base_abs_delta": _public_projection(branch_local["base_abs_delta"], public_to_internal),
        "internal_base_abs_delta": branch_local["base_abs_delta"],
        "base_rel_delta": _public_projection(branch_local["base_rel_delta"], public_to_internal),
        "internal_base_rel_delta": branch_local["base_rel_delta"],
        "directional_derivatives": None
        if branch_local.get("directional_derivatives") is None
        else _public_projection(branch_local["directional_derivatives"], public_to_internal),
        "internal_directional_derivatives": branch_local.get("directional_derivatives"),
        "jacobian": branch_local.get("jacobian"),
        "grads": branch_local.get("grads"),
        "derivative_mode": branch_local.get("derivative_mode"),
        "branch_local_report": branch_local,
        "fd_validation": None,
    }

    if validate_fd:
        params_for = _params_for_direction(params, direction_params)  # type: ignore[arg-type]
        complete_report = direct_coil_same_branch_complete_solve_fd_report(
            input_path,
            params,
            params_for=params_for,
            objective_fn=lambda payload: {key: scalar_value_fns[key](payload) for key in internal_keys},
            eps=float(fd_epsilon),
            init_kwargs=dict(init_kwargs or {}),
            solve_kwargs=complete_solve_kwargs,
            fingerprint_rtol=float(fingerprint_rtol),
            fingerprint_atol=float(fingerprint_atol),
            require_active_trace=bool(options.require_active_trace),
        )
        scalar_report = direct_coil_branch_local_scalars_report_from_complete_fd(
            complete_report,
            branch_local,
            scalar_keys=internal_keys,
            rtol=1.0e-2 if rtol is None else rtol,
            atol=5.0e-8 if atol is None else atol,
            base_value_atol=2.0e-3 if base_value_atol is None else base_value_atol,
            json_safe=True,
        )
        result["fd_validation"] = {
            "complete_solve_report": complete_report,
            "scalar_report": scalar_report,
            "public_scalar_report": {
                public: scalar_report["scalars"][internal]
                for public, internal in public_to_internal.items()
                if internal in scalar_report.get("scalars", {})
            },
        }

    return result


def free_boundary_value_and_jacobian(
    input_path: Any,
    params: CoilFieldParams,
    *,
    outputs: Sequence[str] = DEFAULT_FREE_BOUNDARY_DERIVATIVE_OUTPUTS,
    options: FreeBoundaryDerivativeOptions | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Return complete-solve values plus a branch-local Jacobian/VJP report."""

    return free_boundary_value_and_jvp(
        input_path,
        params,
        direction_params=None,
        outputs=outputs,
        options=options,
        **kwargs,
    )


def coil_direction(
    params: CoilFieldParams,
    *,
    current: float = 0.0,
    curve_dof: float = 0.0,
    curve_index: tuple[int, int, int] | None = None,
) -> CoilFieldParams:
    """Return a simple coil tangent for examples and finite-difference checks.

    ``current`` perturbs the first current by an additive amount.  ``curve_dof``
    perturbs one Fourier coefficient selected by ``curve_index``.
    """

    currents = np.zeros_like(np.asarray(params.base_currents, dtype=float))
    dofs = np.zeros_like(np.asarray(params.base_curve_dofs, dtype=float))
    if currents.size:
        currents.reshape(-1)[0] = float(current)
    if curve_index is not None:
        dofs[curve_index] = float(curve_dof)
    return params.with_arrays(base_curve_dofs=jnp.asarray(dofs), base_currents=jnp.asarray(currents))


def contract_free_boundary_vjp(jacobian: Any, cotangent: Sequence[float]) -> Any:
    """Contract a row-stacked pytree Jacobian with an output cotangent vector."""

    weights = jnp.asarray(cotangent)
    n_outputs = int(weights.size)

    def contract_leaf(leaf: Any) -> Any:
        array = jnp.asarray(leaf)
        return jnp.tensordot(weights, jnp.reshape(array, (n_outputs, -1)), axes=(0, 0)).reshape(array.shape[1:])

    return jax.tree_util.tree_map(contract_leaf, jacobian)


__all__ = [
    "DEFAULT_FREE_BOUNDARY_DERIVATIVE_OUTPUTS",
    "FreeBoundaryDerivativeOptions",
    "PUBLIC_OUTPUT_ALIASES",
    "canonical_free_boundary_output_keys",
    "coil_direction",
    "contract_free_boundary_vjp",
    "free_boundary_value_and_jacobian",
    "free_boundary_value_and_jvp",
]
