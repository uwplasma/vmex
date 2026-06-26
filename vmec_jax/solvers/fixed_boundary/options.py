"""Pure option validation helpers for solver entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LambdaGDOptions:
    """Validated controls for lambda-only gradient descent."""

    max_iter: int
    max_backtracks: int
    bt_factor: float
    preconditioner: str
    precond_exponent: float


@dataclass(frozen=True)
class FixedBoundaryGDOptions:
    """Validated controls for full fixed-boundary gradient descent."""

    max_iter: int
    max_backtracks: int
    bt_factor: float
    gamma: float


@dataclass(frozen=True)
class FixedBoundaryLBFGSOptions:
    """Validated controls for fixed-boundary L-BFGS optimization."""

    history_size: int
    max_iter: int
    max_backtracks: int
    bt_factor: float
    gamma: float


@dataclass(frozen=True)
class ResidualLBFGSOptions:
    """Validated controls for L-BFGS on the VMEC force residual objective."""

    w_rz: float
    w_l: float
    objective_scale: float | None
    scale_rz: float
    scale_l: float
    history_size: int
    max_iter: int
    max_backtracks: int
    bt_factor: float


@dataclass(frozen=True)
class ResidualGNOptions:
    """Validated controls for Gauss-Newton/CG VMEC residual minimization."""

    damping: float | None
    damping_increase: float
    damping_decrease: float
    max_damping_eff: float
    max_retries: int
    zero_m1_iters_eff: int
    zero_m1_fsqz_thresh: float | None
    w_rz: float
    w_l: float
    max_iter: int
    cg_maxiter: int
    max_backtracks: int
    bt_factor: float
    objective_scale: float | None


@dataclass(frozen=True)
class ResidualIterationOptions:
    """Validated controls for the VMEC2000-style residual iteration loop."""

    max_iter: int
    step_size: float
    precompile_only: bool
    signgs: int
    lambda_update_scale: float
    enforce_vmec_lambda_axis: bool
    vmec2000_control: bool
    reference_mode: bool
    limit_dt_from_force: bool
    limit_update_rms: bool
    backtracking: bool
    strict_update: bool
    jit_precompile: bool
    use_scan: bool


def _validate_line_search_options(*, max_iter: Any, max_backtracks: Any, bt_factor: Any) -> tuple[int, int, float]:
    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    max_backtracks = int(max_backtracks)
    if max_backtracks < 0:
        raise ValueError("max_backtracks must be >= 0")
    bt_factor = float(bt_factor)
    if not (0.0 < bt_factor < 1.0):
        raise ValueError("bt_factor must be in (0, 1)")
    return max_iter, max_backtracks, bt_factor


def _validate_gamma(gamma: Any) -> float:
    gamma = float(gamma)
    if abs(gamma - 1.0) < 1e-14:
        raise ValueError("gamma=1 makes wp/(gamma-1) singular")
    return gamma


def _validate_history_size(history_size: Any) -> int:
    history_size = int(history_size)
    if history_size < 1:
        raise ValueError("history_size must be >= 1")
    return history_size


def _validate_residual_weights(*, w_rz: Any, w_l: Any) -> tuple[float, float]:
    w_rz = float(w_rz)
    w_l = float(w_l)
    if w_rz < 0.0 or w_l < 0.0:
        raise ValueError("w_rz and w_l must be nonnegative")
    return w_rz, w_l


def _validate_objective_scale(objective_scale: Any | None) -> float | None:
    if objective_scale is not None and float(objective_scale) <= 0.0:
        raise ValueError("objective_scale must be positive when provided")
    return float(objective_scale) if objective_scale is not None else None


def validate_pressure_shape(actual_shape: tuple[int, ...], expected_shape: tuple[int, ...]) -> None:
    """Validate the pressure vector shape without importing array libraries."""
    if actual_shape != expected_shape:
        raise ValueError(f"pressure must have shape {expected_shape}, got {actual_shape}")


def validate_lambda_gd_options(
    *,
    max_iter: Any,
    max_backtracks: Any,
    bt_factor: Any,
    preconditioner: Any,
    precond_exponent: Any,
) -> LambdaGDOptions:
    """Validate validate lambda gd options for fixed-boundary VMEC solve and implicit differentiation."""
    max_iter, max_backtracks, bt_factor = _validate_line_search_options(
        max_iter=max_iter,
        max_backtracks=max_backtracks,
        bt_factor=bt_factor,
    )
    preconditioner = str(preconditioner).strip().lower()
    if preconditioner not in ("none", "mode_diag"):
        raise ValueError(f"Unknown preconditioner kind={preconditioner!r}")
    precond_exponent = float(precond_exponent)
    if preconditioner != "none" and precond_exponent <= 0.0:
        raise ValueError("precond_exponent must be > 0 when using a preconditioner")
    return LambdaGDOptions(
        max_iter=max_iter,
        max_backtracks=max_backtracks,
        bt_factor=bt_factor,
        preconditioner=preconditioner,
        precond_exponent=precond_exponent,
    )


def validate_fixed_boundary_gd_options(
    *,
    max_iter: Any,
    max_backtracks: Any,
    bt_factor: Any,
    gamma: Any,
) -> FixedBoundaryGDOptions:
    """Validate validate fixed boundary gd options for fixed-boundary VMEC solve and implicit differentiation."""
    max_iter, max_backtracks, bt_factor = _validate_line_search_options(
        max_iter=max_iter,
        max_backtracks=max_backtracks,
        bt_factor=bt_factor,
    )
    return FixedBoundaryGDOptions(
        max_iter=max_iter,
        max_backtracks=max_backtracks,
        bt_factor=bt_factor,
        gamma=_validate_gamma(gamma),
    )


def validate_fixed_boundary_lbfgs_options(
    *,
    history_size: Any,
    max_iter: Any,
    max_backtracks: Any,
    bt_factor: Any,
    gamma: Any,
) -> FixedBoundaryLBFGSOptions:
    """Validate validate fixed boundary lbfgs options for fixed-boundary VMEC solve and implicit differentiation."""
    history_size = _validate_history_size(history_size)
    max_iter, max_backtracks, bt_factor = _validate_line_search_options(
        max_iter=max_iter,
        max_backtracks=max_backtracks,
        bt_factor=bt_factor,
    )
    return FixedBoundaryLBFGSOptions(
        history_size=history_size,
        max_iter=max_iter,
        max_backtracks=max_backtracks,
        bt_factor=bt_factor,
        gamma=_validate_gamma(gamma),
    )


def validate_residual_lbfgs_options(
    *,
    w_rz: Any,
    w_l: Any,
    objective_scale: Any | None,
    scale_rz: Any,
    scale_l: Any,
    history_size: Any,
    max_iter: Any,
    max_backtracks: Any,
    bt_factor: Any,
) -> ResidualLBFGSOptions:
    """Validate validate residual lbfgs options for fixed-boundary VMEC solve and implicit differentiation."""
    w_rz, w_l = _validate_residual_weights(w_rz=w_rz, w_l=w_l)
    objective_scale = _validate_objective_scale(objective_scale)
    scale_rz = float(scale_rz)
    scale_l = float(scale_l)
    if scale_rz <= 0.0 or scale_l <= 0.0:
        raise ValueError("scale_rz and scale_l must be positive")
    history_size = _validate_history_size(history_size)
    max_iter, max_backtracks, bt_factor = _validate_line_search_options(
        max_iter=max_iter,
        max_backtracks=max_backtracks,
        bt_factor=bt_factor,
    )
    return ResidualLBFGSOptions(
        w_rz=w_rz,
        w_l=w_l,
        objective_scale=objective_scale,
        scale_rz=scale_rz,
        scale_l=scale_l,
        history_size=history_size,
        max_iter=max_iter,
        max_backtracks=max_backtracks,
        bt_factor=bt_factor,
    )


def validate_residual_gn_options(
    *,
    damping: Any | None,
    damping_increase: Any,
    damping_decrease: Any,
    max_damping: Any | None,
    max_retries: Any,
    zero_m1_iters: Any | None,
    zero_m1_fsqz_thresh: Any | None,
    w_rz: Any,
    w_l: Any,
    max_iter: Any,
    cg_maxiter: Any,
    max_backtracks: Any,
    bt_factor: Any,
    objective_scale: Any | None,
) -> ResidualGNOptions:
    """Validate validate residual gn options for fixed-boundary VMEC solve and implicit differentiation."""
    damping = None if damping is None else float(damping)
    if damping is not None and damping < 0.0:
        raise ValueError("damping must be nonnegative")
    damping_increase = float(damping_increase)
    damping_decrease = float(damping_decrease)
    if damping_increase <= 1.0:
        raise ValueError("damping_increase must be > 1")
    if not (0.0 < damping_decrease <= 1.0):
        raise ValueError("damping_decrease must be in (0, 1]")
    max_damping_eff = float("inf") if max_damping is None else float(max_damping)
    if max_damping_eff <= 0.0:
        raise ValueError("max_damping must be positive")
    max_retries = int(max_retries)
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    zero_m1_iters_eff = 0 if zero_m1_iters is None else int(zero_m1_iters)
    if zero_m1_iters_eff < 0:
        raise ValueError("zero_m1_iters must be >= 0")
    zero_m1_fsqz_thresh = None if zero_m1_fsqz_thresh is None else float(zero_m1_fsqz_thresh)
    if zero_m1_fsqz_thresh is not None and zero_m1_fsqz_thresh < 0.0:
        raise ValueError("zero_m1_fsqz_thresh must be >= 0")
    w_rz, w_l = _validate_residual_weights(w_rz=w_rz, w_l=w_l)
    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    cg_maxiter = int(cg_maxiter)
    if cg_maxiter < 1:
        raise ValueError("cg_maxiter must be >= 1")
    max_backtracks = int(max_backtracks)
    bt_factor = float(bt_factor)
    if not (0.0 < bt_factor < 1.0):
        raise ValueError("bt_factor must be in (0, 1)")
    objective_scale = _validate_objective_scale(objective_scale)
    return ResidualGNOptions(
        damping=damping,
        damping_increase=damping_increase,
        damping_decrease=damping_decrease,
        max_damping_eff=max_damping_eff,
        max_retries=max_retries,
        zero_m1_iters_eff=zero_m1_iters_eff,
        zero_m1_fsqz_thresh=zero_m1_fsqz_thresh,
        w_rz=w_rz,
        w_l=w_l,
        max_iter=max_iter,
        cg_maxiter=cg_maxiter,
        max_backtracks=max_backtracks,
        bt_factor=bt_factor,
        objective_scale=objective_scale,
    )


def validate_residual_iteration_options(
    *,
    max_iter: Any,
    step_size: Any,
    precompile_only: Any,
    signgs: Any,
    lambda_update_scale: Any,
    enforce_vmec_lambda_axis: Any,
    vmec2000_control: Any,
    reference_mode: Any,
    limit_dt_from_force: Any,
    limit_update_rms: Any,
    backtracking: Any,
    strict_update: Any,
    jit_precompile: Any,
    use_scan: Any,
) -> ResidualIterationOptions:
    """Validate validate residual iteration options for fixed-boundary VMEC solve and implicit differentiation."""
    max_iter = int(max_iter)
    precompile_only = bool(precompile_only)
    if max_iter < 1 and not precompile_only:
        raise ValueError("max_iter must be >= 1")
    if max_iter < 1 and precompile_only:
        max_iter = 1
    step_size = float(step_size)
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    return ResidualIterationOptions(
        max_iter=max_iter,
        step_size=step_size,
        precompile_only=precompile_only,
        signgs=int(signgs),
        lambda_update_scale=float(lambda_update_scale),
        enforce_vmec_lambda_axis=bool(enforce_vmec_lambda_axis),
        vmec2000_control=bool(vmec2000_control),
        reference_mode=bool(reference_mode),
        limit_dt_from_force=bool(limit_dt_from_force),
        limit_update_rms=bool(limit_update_rms),
        backtracking=bool(backtracking),
        strict_update=bool(strict_update),
        jit_precompile=bool(jit_precompile),
        use_scan=bool(use_scan),
    )
