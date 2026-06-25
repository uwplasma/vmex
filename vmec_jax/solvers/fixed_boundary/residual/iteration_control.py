"""Scalar control decisions for one fixed-boundary residual iteration.

The residual iteration loop is intentionally close to VMEC2000, but the scalar
branch decisions are easier to validate when they are pure functions.  This
module holds those decisions so the hot loop can focus on force evaluation and
state updates.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple


class ResidualIterationControlSample(NamedTuple):
    """Host-side branch decisions sampled at the start of an iteration."""

    iter_since_restart: int
    zero_m1_value: float
    include_edge: bool
    include_edge_residual: bool
    precond_jmax_override: int | None
    precond_expected_jmax: int
    need_bcovar_update: bool
    use_cached_precond: bool


class ConstraintPreconditionerChannels(NamedTuple):
    """Constraint preconditioner arrays and activity flags for force assembly."""

    precond_diag: Any
    tcon: Any
    precond_active: Any
    tcon_active: Any


def resolve_residual_iteration_control_sample(
    *,
    iter2: int,
    iter1: int,
    vmec2000_control: bool,
    free_boundary_enabled: bool,
    freeb_ivac_effective: int,
    prev_rz_fsq: float,
    fsqz2_history: Sequence[float],
    env_freeb_include_edge: bool,
    env_force_edge_residual: str,
    precond_cache_valid: bool,
    force_bcovar_update: bool,
    preconditioner_update_interval: int,
    ns: int,
) -> ResidualIterationControlSample:
    """Resolve VMEC-compatible scalar controls for the current iteration.

    These branches govern edge residual rows, m=1 force suppression, and whether
    VMEC2000-style cached one-dimensional preconditioner channels may be reused.
    Keeping this logic outside the loop makes branch-local AD/FD gates easier to
    audit because the fingerprint inputs are explicit.
    """

    iter_since_restart = int(iter2) - int(iter1)
    if bool(vmec2000_control):
        fsqz_prev = float(fsqz2_history[-1]) if fsqz2_history else 1.0
        zero_m1_value = 1.0 if (int(iter2) < 2) or (fsqz_prev < 1.0e-6) else 0.0
        include_edge = bool(env_freeb_include_edge)
    else:
        zero_m1_value = (
            1.0
            if (iter_since_restart < 2) or (len(fsqz2_history) > 0 and float(fsqz2_history[-1]) < 1.0e-6)
            else 0.0
        )
        include_edge = bool(iter_since_restart < 50) and (float(prev_rz_fsq) < 1.0e-6)

    include_edge_residual = bool(include_edge)
    if bool(free_boundary_enabled) and int(freeb_ivac_effective) >= 1:
        include_edge_residual = True
    if str(env_force_edge_residual).strip().lower() in ("1", "true", "yes"):
        include_edge_residual = True

    precond_jmax_override = None
    if bool(vmec2000_control) and bool(free_boundary_enabled) and int(freeb_ivac_effective) >= 1:
        precond_jmax_override = int(ns)
    precond_expected_jmax = int(precond_jmax_override) if precond_jmax_override is not None else max(int(ns) - 1, 1)

    need_bcovar_update = bool(vmec2000_control) and (
        (not bool(precond_cache_valid))
        or bool(force_bcovar_update)
        or (iter_since_restart % int(preconditioner_update_interval) == 0)
    )
    use_cached_precond = bool(vmec2000_control) and bool(precond_cache_valid) and (not bool(need_bcovar_update))

    return ResidualIterationControlSample(
        iter_since_restart=iter_since_restart,
        zero_m1_value=zero_m1_value,
        include_edge=include_edge,
        include_edge_residual=include_edge_residual,
        precond_jmax_override=precond_jmax_override,
        precond_expected_jmax=precond_expected_jmax,
        need_bcovar_update=need_bcovar_update,
        use_cached_precond=use_cached_precond,
    )


def constraint_preconditioner_channels(
    *,
    use_cached_precond: bool,
    cached_precond_diag: Any,
    cached_tcon: Any,
    zero_precond_diag: Any,
    zero_tcon: Any,
    host_update_assembly: bool,
    jnp_true_bool: Any,
    jnp_false_bool: Any,
    jnp_module: Any,
) -> ConstraintPreconditionerChannels:
    """Select cached or zero constraint channels for the force kernel."""

    precond_diag = (
        cached_precond_diag if (bool(use_cached_precond) and cached_precond_diag is not None) else zero_precond_diag
    )
    tcon = cached_tcon if (bool(use_cached_precond) and cached_tcon is not None) else zero_tcon
    if bool(host_update_assembly) and jnp_true_bool is not None:
        precond_active = jnp_true_bool if bool(use_cached_precond) else jnp_false_bool
        tcon_active = jnp_true_bool if bool(use_cached_precond) else jnp_false_bool
    else:
        precond_active = jnp_module.asarray(bool(use_cached_precond), dtype=bool)
        tcon_active = jnp_module.asarray(bool(use_cached_precond), dtype=bool)

    return ConstraintPreconditionerChannels(
        precond_diag=precond_diag,
        tcon=tcon,
        precond_active=precond_active,
        tcon_active=tcon_active,
    )
