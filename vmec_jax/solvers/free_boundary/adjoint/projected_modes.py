"""Projected-mode fixed-point helpers for free-boundary adjoint validation."""

from __future__ import annotations

from typing import Any

from vmec_jax._compat import jnp

from .boundary_replay import vacuum_boundary_fields_from_cylindrical_jax
from .dense import dense_fixed_point_solve_jax
from .mode_operator import mode_rhs_from_gsource_jax
from .mode_solve import dense_mode_vacuum_solve_jax
from .objectives import weighted_half_norm


def direct_coil_projected_mode_fixed_point_jax(
    params: Any,
    initial_state: Any,
    *,
    boundary_from_state: Any,
    update_from_response: Any,
    mode_matrix: Any,
    sin_basis: Any,
    xmpot: Any,
    n_raw: Any,
    imirr: Any | None = None,
    imirr_full: Any | None = None,
    cos_basis: Any | None = None,
    onp: float = 1.0,
    lasym: bool = False,
    nuv3: int | None = None,
    nuv_full: int | None = None,
    max_iter: int = 10,
    damping: float = 1.0,
    symmetric: bool = False,
) -> dict[str, Any]:
    """Solve a small direct-coil free-boundary fixed-point validation loop."""

    from vmec_jax.external_fields import sample_coil_field_cylindrical

    required = ("R", "Z", "phi", "Ru", "Zu", "Rv", "Zv")

    def _mode_response_for_state(state, coil_params):
        boundary = boundary_from_state(state)
        missing = [name for name in required if name not in boundary]
        if missing:
            raise ValueError(f"boundary_from_state missing keys: {missing}")
        br, bp, bz = sample_coil_field_cylindrical(
            coil_params,
            jnp.asarray(boundary["R"]),
            jnp.asarray(boundary["Z"]),
            jnp.asarray(boundary["phi"]),
        )
        vac = vacuum_boundary_fields_from_cylindrical_jax(
            br=br,
            bp=bp,
            bz=bz,
            R=boundary["R"],
            Ru=boundary["Ru"],
            Zu=boundary["Zu"],
            Rv=boundary["Rv"],
            Zv=boundary["Zv"],
        )
        rhs_mode = mode_rhs_from_gsource_jax(
            vac["bnormal"],
            sin_basis=sin_basis,
            cos_basis=cos_basis,
            xmpot=xmpot,
            n_raw=n_raw,
            onp=float(onp),
            lasym=bool(lasym),
            imirr=imirr,
            imirr_full=imirr_full,
            nuv3=nuv3,
            nuv_full=nuv_full,
        )
        response = dense_mode_vacuum_solve_jax(
            mode_matrix,
            rhs_mode,
            sin_basis,
            cos_basis,
            symmetric=bool(symmetric),
        )
        response = {**response, "rhs_mode": rhs_mode}
        return boundary, vac, response

    def _update(state, coil_params):
        boundary, vac, response = _mode_response_for_state(state, coil_params)
        return update_from_response(state, response, vac, boundary, coil_params)

    root = dense_fixed_point_solve_jax(
        _update,
        initial_state,
        params,
        max_iter=max_iter,
        damping=damping,
    )
    boundary, vac, response = _mode_response_for_state(root, params)
    fixed_update = _update(root, params)
    return {
        "state": root,
        "fixed_point_residual": root - fixed_update,
        "update": fixed_update,
        "boundary": boundary,
        "vac": vac,
        "response": response,
    }


def direct_coil_projected_mode_fixed_point_objective_jax(
    params: Any,
    initial_state: Any,
    *,
    boundary_from_state: Any,
    update_from_response: Any,
    mode_matrix: Any,
    sin_basis: Any,
    xmpot: Any,
    n_raw: Any,
    imirr: Any | None = None,
    imirr_full: Any | None = None,
    cos_basis: Any | None = None,
    onp: float = 1.0,
    lasym: bool = False,
    nuv3: int | None = None,
    nuv_full: int | None = None,
    max_iter: int = 10,
    damping: float = 1.0,
    symmetric: bool = False,
    state_weights: Any = 1.0,
    update_weights: Any = 0.0,
    mode_weights: Any = 0.0,
    rhs_mode_weights: Any = 0.0,
    bnormal_weight: float = 0.0,
    fixed_point_residual_weight: float = 1.0,
) -> dict[str, Any]:
    """Return a scalar objective for the projected-mode fixed-point helper."""

    solved = direct_coil_projected_mode_fixed_point_jax(
        params,
        initial_state,
        boundary_from_state=boundary_from_state,
        update_from_response=update_from_response,
        mode_matrix=mode_matrix,
        sin_basis=sin_basis,
        xmpot=xmpot,
        n_raw=n_raw,
        imirr=imirr,
        imirr_full=imirr_full,
        cos_basis=cos_basis,
        onp=onp,
        lasym=lasym,
        nuv3=nuv3,
        nuv_full=nuv_full,
        max_iter=max_iter,
        damping=damping,
        symmetric=symmetric,
    )
    components = {
        "state": weighted_half_norm(solved["state"], state_weights),
        "update": weighted_half_norm(solved["update"], update_weights),
        "mode": weighted_half_norm(solved["response"]["mode_coeffs"], mode_weights),
        "rhs_mode": weighted_half_norm(solved["response"]["rhs_mode"], rhs_mode_weights),
        "bnormal": weighted_half_norm(solved["vac"]["bnormal"], bnormal_weight),
        "fixed_point_residual": weighted_half_norm(
            solved["fixed_point_residual"],
            fixed_point_residual_weight,
        ),
    }
    objective = sum(components.values())
    return {
        **solved,
        "objective": objective,
        "objective_components": components,
    }


def direct_coil_projected_mode_fixed_point_directional_check_jax(
    params: Any,
    direction: Any,
    initial_state: Any,
    *,
    directional_check_func: Any,
    eps: float = 1.0e-4,
    **objective_kwargs: Any,
) -> dict[str, Any]:
    """Validate projected-mode fixed-point coil gradients by central FD."""

    def objective(coil_params):
        """Evaluate objective for direct-coil free-boundary solve and branch-local adjoint validation."""
        solved = direct_coil_projected_mode_fixed_point_objective_jax(
            coil_params,
            initial_state,
            **objective_kwargs,
        )
        return solved["objective"]

    check = directional_check_func(
        objective,
        params,
        direction,
        eps=eps,
    )
    solved = direct_coil_projected_mode_fixed_point_objective_jax(
        params,
        initial_state,
        **objective_kwargs,
    )
    return {
        **check,
        "solved": solved,
        "objective_components": solved["objective_components"],
    }


__all__ = [
    "direct_coil_projected_mode_fixed_point_directional_check_jax",
    "direct_coil_projected_mode_fixed_point_jax",
    "direct_coil_projected_mode_fixed_point_objective_jax",
]
