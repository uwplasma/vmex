"""High-level helpers for VMEC driver scripts.

These functions provide a thin, convenient layer over the core modules so
simple scripts can be written with minimal boilerplate, while still allowing
power users to drop down to lower-level APIs.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
import os
import time
from typing import Any, Optional

import numpy as np
from .boundary import boundary_from_indata
from .config import VMECConfig, load_config
from .energy import _iotaf_from_iotas, flux_profiles_from_indata, flux_profiles_from_indata_host_default
from .init_guess import initial_guess_from_boundary
from .multigrid import interp_vmec_state
from .profiles import eval_profiles
from .drivers import flux as _driver_flux_helpers
from .drivers import debug as _driver_debug_helpers
from .drivers import dynamic_scan as _driver_dynamic_scan_helpers
from .drivers import finish as _driver_finish_helpers
from .drivers import io as _driver_io_helpers
from .drivers import output as _driver_output_helpers
from .drivers import policy as _driver_policy_helpers
from .drivers import results as _driver_result_helpers
from .drivers import runtime as _driver_runtime_helpers
from .drivers import solve as _driver_solve_helpers
from .drivers import staging as _driver_staging_helpers
from .solve import (
    SolveVmecResidualResult,
    solve_fixed_boundary_gd,
    solve_fixed_boundary_lbfgs,
    solve_fixed_boundary_residual_iter,
)
from .static import VMECStatic, build_static
from .wout import WoutData, read_wout, state_from_wout

_FSQ_COMPONENT_NAMES = _driver_policy_helpers.FSQ_COMPONENT_NAMES
_VALID_SOLVER_MODES = _driver_policy_helpers.VALID_SOLVER_MODES
_accelerated_cli_budgeted_stage_iters = _driver_policy_helpers.accelerated_cli_budgeted_stage_iters
_accelerated_cli_budgeted_total_iters = _driver_policy_helpers.accelerated_cli_budgeted_total_iters
_accelerated_fsq_total_target_from_ftol = _driver_policy_helpers.accelerated_fsq_total_target_from_ftol
_allocate_integer_budget = _driver_policy_helpers.allocate_integer_budget
_as_float_list = _driver_policy_helpers.as_float_list
_as_list_like = _driver_policy_helpers.as_list_like
_default_non_autodiff_solver_policy_for_backend = _driver_policy_helpers.default_non_autodiff_solver_policy_for_backend
_default_preconditioner_use_lax_tridi = _driver_policy_helpers.default_preconditioner_use_lax_tridi
_default_preconditioner_use_precomputed_tridi = _driver_policy_helpers.default_preconditioner_use_precomputed_tridi
_default_use_scan_for_backend = _driver_policy_helpers.default_use_scan_for_backend
_distribute_stage_iters = _driver_policy_helpers.distribute_stage_iters
_dynamic_scan_probe_settings_for_backend = _driver_policy_helpers.dynamic_scan_probe_settings
_host_update_assembly_driver_default = _driver_policy_helpers.host_update_assembly_driver_default
_normalize_solver_mode = _driver_policy_helpers.normalize_solver_mode
_policy_backend_for_requested_device = _driver_policy_helpers.policy_backend_for_requested_device
_requested_solver_device_name = _driver_policy_helpers.requested_solver_device_name
_requested_final_ftol = _driver_policy_helpers.requested_final_ftol
_resolve_initial_fixed_boundary_policy = _driver_policy_helpers.resolve_initial_fixed_boundary_policy
_resolve_axis_infer_missing_policy = _driver_policy_helpers.resolve_axis_infer_missing_policy
_resolve_driver_signgs = _driver_policy_helpers.resolve_driver_signgs
_resolve_driver_step_size = _driver_policy_helpers.resolve_driver_step_size
_resolve_fixed_boundary_solver_device_name = _driver_policy_helpers.resolve_fixed_boundary_solver_device_name
_resolve_jit_forces_auto_policy = _driver_policy_helpers.resolve_jit_forces_auto_policy
_resolve_stage_jit_settings = _driver_policy_helpers.resolve_stage_jit_settings
_resolve_vmec2000_jit_forces_policy = _driver_policy_helpers.resolve_vmec2000_jit_forces_policy
_resolve_vmec2000_stage_controls = _driver_policy_helpers.resolve_vmec2000_stage_controls
_result_final_fsq = _driver_policy_helpers.result_final_fsq
_result_final_residuals = _driver_policy_helpers.result_final_residuals
_result_hits_total_target = _driver_policy_helpers.result_hits_total_target
_result_meets_requested_ftol = _driver_policy_helpers.result_meets_requested_ftol
_sanitize_minimal_resume_state_for_finish = _driver_policy_helpers.sanitize_minimal_resume_state_for_finish
_sanitize_resume_state_for_grid_change = _driver_policy_helpers.sanitize_resume_state_for_grid_change
_sanitize_resume_state_for_same_grid = _driver_policy_helpers.sanitize_resume_state_for_same_grid
_aggregate_stage_chunk_timing = _driver_result_helpers.aggregate_stage_chunk_timing
_cat_result_history = _driver_result_helpers.cat_result_history
_copy_final_force_payload = _driver_result_helpers.copy_final_force_payload
_merge_stage_chunk_results = _driver_result_helpers.merge_stage_chunk_results
_result_with_diag = _driver_result_helpers.result_with_diag
_stage_switch_reason_from_progress = _driver_result_helpers.stage_switch_reason_from_progress
_timing_solve_total_s = _driver_result_helpers.timing_solve_total_s
_vmec_histories_match = _driver_result_helpers.vmec_histories_match
_vmec_history_relerr = _driver_result_helpers.vmec_history_relerr


def _free_boundary_module():
    from importlib import import_module

    return import_module(".free_boundary", __package__)


def __getattr__(name: str):
    if name in {"MGridMetadata", "PreparedMGrid"}:
        value = getattr(_free_boundary_module(), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def validate_free_boundary_config(cfg: VMECConfig, *, strict: bool = False) -> None:
    """Lazily validate free-boundary inputs.

    This wrapper preserves the historical driver-level monkeypatch point while
    keeping fixed-boundary CLI startup from importing NESTOR/free-boundary code.
    """

    return _free_boundary_module().validate_free_boundary_config(cfg, strict=strict)


def prepare_mgrid_for_config(cfg: VMECConfig, *, load_fields: bool = True, strict: bool = False):
    """Lazily prepare mgrid metadata/fields for free-boundary inputs."""

    return _free_boundary_module().prepare_mgrid_for_config(cfg, load_fields=load_fields, strict=strict)


def _free_boundary_static_inputs(
    cfg: VMECConfig,
    *,
    load_fields: bool,
    strict: bool,
) -> tuple[object | None, tuple[float, ...] | None]:
    if not bool(getattr(cfg, "lfreeb", False)):
        return None, None
    validate_free_boundary_config(cfg, strict=strict)
    prepared_fb = prepare_mgrid_for_config(cfg, load_fields=load_fields, strict=strict)
    PreparedMGrid = getattr(_free_boundary_module(), "PreparedMGrid")
    if isinstance(prepared_fb, PreparedMGrid):
        return prepared_fb.metadata, prepared_fb.extcur
    return None, None


@dataclass(frozen=True)
class ExampleData:
    input_path: Path
    wout_path: Optional[Path]
    cfg: VMECConfig
    indata: any
    static: VMECStatic
    wout: Optional[WoutData]
    state: Optional[any]


@dataclass(frozen=True)
class FixedBoundaryRun:
    """Container returned by ``run_fixed_boundary``."""

    cfg: VMECConfig
    indata: any
    static: VMECStatic
    state: any
    result: any | None
    flux: any
    profiles: dict
    signgs: int


@dataclass(frozen=True)
class _FixedBoundaryStartupContext:
    """Resolved startup policy and restart state for one fixed-boundary run."""

    cfg: VMECConfig
    indata: Any
    requested_solver_device: str
    policy_backend: str
    initial_policy: Any
    solver_mode_explicit: bool
    solver_mode_eff: str
    accelerated_mode: bool
    performance_mode: bool
    use_scan: bool
    cli_fixed_boundary_mode: bool
    restart_state: Any | None
    restart_wout: Any | None
    restart_solver_state: dict | None
    solver_lower: str
    axis_infer_missing: bool
    routed_run: Any | None


def _default_backend_name() -> str:
    try:
        import jax

        return str(jax.default_backend()).strip().lower() or "cpu"
    except Exception:
        return "cpu"


def _dynamic_scan_probe_settings(niter_i: int) -> tuple[int, bool, str]:
    return _dynamic_scan_probe_settings_for_backend(
        niter_i,
        backend_name_func=_default_backend_name,
        getenv=os.getenv,
    )


def default_non_autodiff_solver_policy(indata) -> tuple[str, bool]:
    """Choose the ordinary non-autodiff solver policy from input structure."""

    return _default_non_autodiff_solver_policy_for_backend(indata, _default_backend_name())


def _final_flux_profiles_from_state(**kwargs):
    """Compatibility wrapper around the extracted post-solve flux helper."""

    return _driver_flux_helpers.final_flux_profiles_from_state(
        **kwargs,
        boundary_from_indata_func=boundary_from_indata,
        iotaf_from_iotas_func=_iotaf_from_iotas,
    )


residual_scalars_from_state = _driver_output_helpers.residual_scalars_from_state


def solve_fixed_boundary_from_boundary(
    *,
    boundary,
    static: VMECStatic,
    indata,
    flux,
    pressure,
    signgs: int,
    max_iter: int = 2,
    step_size: float = 5e-3,
    jacobian_penalty: float = 1e3,
    jit_grad: bool = False,
    differentiable: bool = True,
    stop_grad_in_update: bool = True,
    verbose: bool = False,
    vmec_project: bool = False,
):
    """Solve VMEC fixed-boundary starting from a boundary coefficient set.

    This helper wraps `initial_guess_from_boundary` and `solve_fixed_boundary_gd`
    so optimization scripts can call a single function.
    """
    return _driver_solve_helpers.solve_fixed_boundary_from_boundary(
        boundary=boundary,
        static=static,
        indata=indata,
        flux=flux,
        pressure=pressure,
        signgs=signgs,
        max_iter=max_iter,
        step_size=step_size,
        jacobian_penalty=jacobian_penalty,
        jit_grad=jit_grad,
        differentiable=differentiable,
        stop_grad_in_update=stop_grad_in_update,
        verbose=verbose,
        vmec_project=vmec_project,
        initial_guess_from_boundary_func=initial_guess_from_boundary,
        solve_fixed_boundary_gd_func=solve_fixed_boundary_gd,
    )


def wout_from_fixed_boundary_run(
    run: FixedBoundaryRun,
    *,
    include_fsq: bool = True,
    path: str | Path | None = None,
    fast_bcovar: bool | None = None,
) -> WoutData:
    """Build a minimal VMEC-style ``WoutData`` from a fixed-boundary run."""

    return _driver_output_helpers.wout_from_fixed_boundary_run(
        run,
        include_fsq=include_fsq,
        path=path,
        fast_bcovar=fast_bcovar,
        residual_scalars_func=residual_scalars_from_state,
    )


def write_wout_from_fixed_boundary_run(
    path: str | Path,
    run: FixedBoundaryRun,
    *,
    include_fsq: bool = True,
    fast_bcovar: bool | None = None,
):
    """Write a minimal VMEC-style `wout_*.nc` from a fixed-boundary run."""
    return _driver_output_helpers.write_wout_from_fixed_boundary_run(
        path,
        run,
        include_fsq=include_fsq,
        fast_bcovar=fast_bcovar,
        wout_from_run_func=wout_from_fixed_boundary_run,
    )


def example_paths(case: str, *, root: str | Path | None = None) -> tuple[Path, Optional[Path]]:
    """Return (input_path, wout_path) for a bundled example case."""
    return _driver_io_helpers.example_paths(case, root=root, package_file=__file__)


def load_example(
    case: str,
    *,
    root: str | Path | None = None,
    with_wout: bool = True,
    grid=None,
) -> ExampleData:
    """Load a bundled example case (config + static + optional wout/state)."""
    return _driver_io_helpers.load_example(
        case,
        root=root,
        with_wout=with_wout,
        grid=grid,
        example_data_type=ExampleData,
        example_paths_func=example_paths,
        load_config_func=load_config,
        free_boundary_static_inputs_func=_free_boundary_static_inputs,
        build_static_func=build_static,
        read_wout_func=read_wout,
        state_from_wout_func=state_from_wout,
    )


def load_input(path: str | Path):
    """Convenience wrapper around `load_config`."""
    return _driver_io_helpers.load_input(path, load_config_func=load_config)


def load_wout(path: str | Path) -> WoutData:
    """Convenience wrapper around `read_wout`."""
    return _driver_io_helpers.load_wout(path, read_wout_func=read_wout)


save_npz = _driver_io_helpers.save_npz


_STEP_SIZE_SENTINEL = object()
_MAX_ITER_SENTINEL = object()


def _stage_array_list(value):
    """Return VMEC stage arrays as Python lists with legacy driver semantics."""

    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        if isinstance(value, np.ndarray):
            return list(value.tolist())
    except Exception:
        pass
    if isinstance(value, (int, float, np.integer, np.floating)):
        return [value]
    return None


def _driver_resume_step_size_value(*, step_size, indata) -> float:
    """Resolve the step size used to sanitize resumable VMEC time-step state."""

    if step_size is not _STEP_SIZE_SENTINEL and step_size is not None:
        return float(step_size)
    try:
        return float(indata.get_float("DELT", 5e-3))
    except Exception:
        return 5e-3


def _sanitize_resume_state_for_driver_stage(resume_state, *, step_size, indata):
    return _sanitize_resume_state_for_grid_change(
        resume_state,
        step_size=_driver_resume_step_size_value(step_size=step_size, indata=indata),
    )


def _sanitize_resume_state_for_driver_same_stage(resume_state, *, step_size, indata):
    return _sanitize_resume_state_for_same_grid(
        resume_state,
        step_size=_driver_resume_step_size_value(step_size=step_size, indata=indata),
    )


def _resolve_fixed_boundary_startup_context(
    *,
    input_path,
    solver,
    solver_mode,
    max_iter,
    step_size,
    history_size,
    gn_damping,
    gn_cg_tol,
    gn_cg_maxiter,
    use_initial_guess,
    vmec_project,
    use_restart_triggers,
    vmecpp_restart,
    use_direct_fallback,
    multigrid,
    multigrid_use_input_niter,
    verbose,
    jit_forces,
    jit_precompile,
    use_scan,
    performance_mode,
    scan_wout_corrector,
    stage_transition_heuristic,
    stage_transition_factor,
    stage_transition_scale,
    grid,
    ns_override,
    restart_state,
    restart_wout_path,
    restart_solver_state,
    cli_fixed_boundary_mode,
    solver_device,
    auto_cli_fixed_boundary_mode,
    solver_device_context_active,
    run_fixed_boundary_func,
) -> _FixedBoundaryStartupContext:
    """Resolve input, device, restart, and first-pass policy for the driver."""
    try:
        from ._compat import enable_x64

        enable_x64(True)
    except Exception:
        pass
    requested_solver_device = _requested_solver_device_name(solver_device)
    policy_backend = _policy_backend_for_requested_device(
        requested_solver_device=requested_solver_device,
        default_backend=_default_backend_name(),
    )
    if not bool(solver_device_context_active):
        from ._compat import _default_compilation_cache_dir

        _driver_runtime_helpers.maybe_enable_compilation_cache(
            accelerator_requested=str(policy_backend).strip().lower() in ("gpu", "cuda", "rocm", "tpu"),
            default_compilation_cache_dir=_default_compilation_cache_dir,
            path_cls=Path,
        )
    cfg, indata = load_config(str(input_path))
    initial_policy = _resolve_initial_fixed_boundary_policy(
        requested_solver_device=requested_solver_device,
        policy_backend=policy_backend,
        indata=indata,
        cfg=cfg,
        solver=solver,
        solver_mode=solver_mode,
        performance_mode=bool(performance_mode),
        use_scan=use_scan,
        verbose=bool(verbose),
        grid=grid,
        cli_fixed_boundary_mode=bool(cli_fixed_boundary_mode),
        auto_cli_fixed_boundary_mode=bool(auto_cli_fixed_boundary_mode),
        default_non_autodiff_policy_func=_default_non_autodiff_solver_policy_for_backend,
        default_use_scan_func=_default_use_scan_for_backend,
    )
    restart_context = _driver_runtime_helpers.resolve_restart_context(
        cfg=cfg,
        restart_state=restart_state,
        restart_wout_path=restart_wout_path,
        restart_solver_state=restart_solver_state,
        ns_override=ns_override,
        read_wout_func=read_wout,
        state_from_wout_func=state_from_wout,
        replace_func=replace,
        path_cls=Path,
    )
    cfg = restart_context.cfg
    restart_state_eff = restart_context.restart_state
    restart_solver_state_eff = restart_context.restart_solver_state
    solver_lower = str(solver).lower()
    performance_mode_eff = bool(initial_policy.performance_mode)
    accelerated_mode = bool(initial_policy.accelerated_mode)
    cli_fixed_boundary_mode_eff = bool(initial_policy.cli_fixed_boundary_mode)
    axis_infer_missing = _resolve_axis_infer_missing_policy(
        solver_lower=solver_lower,
        performance_mode=performance_mode_eff,
    )
    routed_run = _driver_solve_helpers.maybe_run_fixed_boundary_in_solver_device_context(
        input_path=input_path,
        solver_device=solver_device,
        solver_device_context_active=bool(solver_device_context_active),
        cfg=cfg,
        indata=indata,
        solver_lower=solver_lower,
        cli_fixed_boundary_mode=cli_fixed_boundary_mode_eff,
        accelerated_mode=accelerated_mode,
        restart_state_present=(restart_state_eff is not None) or (restart_wout_path is not None),
        restart_solver_state_present=restart_solver_state_eff is not None,
        recursive_run_kwargs={
            "solver": solver,
            "solver_mode": str(initial_policy.solver_mode_eff),
            "max_iter": max_iter,
            "step_size": step_size,
            "history_size": int(history_size),
            "gn_damping": gn_damping,
            "gn_cg_tol": gn_cg_tol,
            "gn_cg_maxiter": int(gn_cg_maxiter),
            "use_initial_guess": bool(use_initial_guess),
            "vmec_project": bool(vmec_project),
            "use_restart_triggers": use_restart_triggers,
            "vmecpp_restart": bool(vmecpp_restart),
            "use_direct_fallback": use_direct_fallback,
            "multigrid": multigrid,
            "multigrid_use_input_niter": bool(multigrid_use_input_niter),
            "verbose": bool(verbose),
            "jit_forces": jit_forces,
            "jit_precompile": jit_precompile,
            "use_scan": bool(initial_policy.use_scan),
            "performance_mode": performance_mode_eff,
            "scan_wout_corrector": scan_wout_corrector,
            "stage_transition_heuristic": stage_transition_heuristic,
            "stage_transition_factor": float(stage_transition_factor),
            "stage_transition_scale": float(stage_transition_scale),
            "grid": grid,
            "ns_override": ns_override,
            "restart_state": restart_state,
            "restart_wout_path": restart_wout_path,
            "restart_solver_state": restart_solver_state_eff,
            "cli_fixed_boundary_mode": cli_fixed_boundary_mode_eff,
            "_auto_cli_fixed_boundary_mode": bool(auto_cli_fixed_boundary_mode),
        },
        run_fixed_boundary_func=run_fixed_boundary_func,
        replace_func=replace,
        as_list_like_func=_as_list_like,
        default_backend_name_func=_default_backend_name,
        resolve_solver_device_name_func=_resolve_fixed_boundary_solver_device_name,
        getenv=os.getenv,
    )
    return _FixedBoundaryStartupContext(
        cfg=cfg,
        indata=indata,
        requested_solver_device=requested_solver_device,
        policy_backend=policy_backend,
        initial_policy=initial_policy,
        solver_mode_explicit=bool(initial_policy.solver_mode_explicit),
        solver_mode_eff=str(initial_policy.solver_mode_eff),
        accelerated_mode=accelerated_mode,
        performance_mode=performance_mode_eff,
        use_scan=bool(initial_policy.use_scan),
        cli_fixed_boundary_mode=cli_fixed_boundary_mode_eff,
        restart_state=restart_state_eff,
        restart_wout=restart_context.restart_wout,
        restart_solver_state=restart_solver_state_eff,
        solver_lower=solver_lower,
        axis_infer_missing=bool(axis_infer_missing),
        routed_run=routed_run,
    )


def run_fixed_boundary(
    input_path: str | Path,
    *,
    solver: str = "vmec2000_iter",
    solver_mode: str | None = None,
    max_iter: int | object = _MAX_ITER_SENTINEL,
    step_size: float | object = _STEP_SIZE_SENTINEL,
    history_size: int = 10,
    # vmec_gn tuning (Gauss-Newton on VMEC residual vector)
    gn_damping: float | None = None,
    gn_cg_tol: float | None = None,
    gn_cg_maxiter: int = 80,
    use_initial_guess: bool = False,
    vmec_project: bool = True,
    use_restart_triggers: bool | None = None,
    vmecpp_restart: bool = False,
    use_direct_fallback: bool | None = None,
    multigrid: bool | None = None,
    multigrid_use_input_niter: bool = True,
    verbose: bool = True,
    jit_forces: bool | str = True,
    jit_precompile: bool | None = None,
    use_scan: bool | None = None,
    performance_mode: bool = True,
    scan_wout_corrector: bool | None = None,
    stage_transition_heuristic: bool | None = None,
    stage_transition_factor: float = 50.0,
    stage_transition_scale: float = 0.5,
    grid=None,
    ns_override: int | None = None,
    restart_state: any | None = None,
    restart_wout_path: str | Path | None = None,
    restart_solver_state: dict | None = None,
    cli_fixed_boundary_mode: bool = False,
    solver_device: str | None = None,
    external_field_provider_kind: str | None = None,
    external_field_provider_static: Any = None,
    external_field_provider_params: Any = None,
    free_boundary_activate_fsq: float | None = None,
    limit_update_rms: bool | None = None,
    _auto_cli_fixed_boundary_mode: bool = True,
    _solver_device_context_active: bool = False,
):
    """Run a vmec_jax solve from an ``input.*`` file.

    This is the main public driver and remains backward compatible with older
    scripts that called :func:`run_fixed_boundary` for both fixed-boundary and
    free-boundary decks. If ``LFREEB = T`` in the input namelist, the shared
    free-boundary path is used automatically. New code that wants to make the
    operating mode explicit should prefer :func:`run_free_boundary` for
    free-boundary decks.

    Parameters
    ----------
    input_path:
        Path to a VMEC-style ``input.*`` file.
    solver:
        ``"vmec2000_iter"`` (VMEC-style multigrid iteration; default),
        ``"gd"`` (gradient descent), ``"lbfgs"``, ``"vmec_lbfgs"``, or
        ``"vmec_gn"`` (VMEC residual objective).
    use_initial_guess:
        If True, skip the solve and return the initialized state.
    ns_override:
        If provided, overrides the radial resolution (ns) used to build the state.
    restart_state:
        If provided, use this VMECState as the initial condition instead of
        building a new boundary-based guess. This disables multigrid staging.
    restart_wout_path:
        If provided, load the `wout_*.nc` file and use its state as the initial
        condition (same effect as `restart_state`). This disables multigrid
        staging.
    restart_solver_state:
        Optional solver-state dictionary returned by ``solve_fixed_boundary_residual_iter``
        (``diagnostics["resume_state"]``). When supplied with ``solver="vmec2000_iter"``,
        the time-step/momentum/preconditioner cache is resumed. This disables multigrid
        staging.
    cli_fixed_boundary_mode:
        Internal CLI-only flag for non-differentiable fixed-boundary policy
        overrides. Library callers should leave this as False.
    solver_device:
        Optional JAX default-device override for the solver body. ``None`` uses
        the automatic policy, which routes known CPU-shaped conservative paths
        away from a GPU default backend. Use ``"default"`` to opt out of
        automatic rerouting, or ``"cpu"``/``"gpu"`` to force a device context.
    vmec_project:
        If True (default), re-project the initial guess through the VMEC
        internal grid/weights before returning or solving.
    verbose:
        If True (default), print VMEC-style iteration progress and a summary.
    jit_forces:
        If True (default), JIT the force kernels. If ``"auto"``, disable JIT
        for very small workloads to reduce first-iteration latency.
    performance_mode:
        If True, allow the optimized fixed-boundary policy instead of strict
        VMEC2000 parity. Auto-selected public runs use the VMEC-control
        non-scan loop on CPU and the scan-lifted loop on GPU/CUDA/ROCm; explicit
        accelerated/fast-mode requests keep the scan path unless ``use_scan`` is
        set to False.
    solver_mode:
        Optional explicit solver policy. Supported values:
        ``"default"`` (current parity-guarded fast path),
        ``"parity"`` (strict VMEC2000-style control path), and
        ``"accelerated"`` (experimental non-parity path that prioritizes
        final residual/quality and device residency).

    Returns
    -------
    FixedBoundaryRun
        Shared run container for both fixed-boundary and free-boundary solves.
    """
    t_start = time.perf_counter()
    max_iter_overridden = max_iter is not _MAX_ITER_SENTINEL
    startup = _resolve_fixed_boundary_startup_context(
        input_path=input_path,
        solver=solver,
        solver_mode=solver_mode,
        max_iter=max_iter,
        step_size=step_size,
        history_size=int(history_size),
        gn_damping=gn_damping,
        gn_cg_tol=gn_cg_tol,
        gn_cg_maxiter=int(gn_cg_maxiter),
        use_initial_guess=bool(use_initial_guess),
        vmec_project=bool(vmec_project),
        use_restart_triggers=use_restart_triggers,
        vmecpp_restart=bool(vmecpp_restart),
        use_direct_fallback=use_direct_fallback,
        multigrid=multigrid,
        multigrid_use_input_niter=bool(multigrid_use_input_niter),
        verbose=bool(verbose),
        jit_forces=jit_forces,
        jit_precompile=jit_precompile,
        use_scan=use_scan,
        performance_mode=bool(performance_mode),
        scan_wout_corrector=scan_wout_corrector,
        stage_transition_heuristic=stage_transition_heuristic,
        stage_transition_factor=float(stage_transition_factor),
        stage_transition_scale=float(stage_transition_scale),
        grid=grid,
        ns_override=ns_override,
        restart_state=restart_state,
        restart_wout_path=restart_wout_path,
        restart_solver_state=restart_solver_state,
        cli_fixed_boundary_mode=bool(cli_fixed_boundary_mode),
        solver_device=solver_device,
        auto_cli_fixed_boundary_mode=bool(_auto_cli_fixed_boundary_mode),
        solver_device_context_active=bool(_solver_device_context_active),
        run_fixed_boundary_func=run_fixed_boundary,
    )
    if startup.routed_run is not None:
        return startup.routed_run
    cfg = startup.cfg
    indata = startup.indata
    policy_backend = startup.policy_backend
    solver_mode_eff = startup.solver_mode_eff
    accelerated_mode = startup.accelerated_mode
    performance_mode = startup.performance_mode
    use_scan = startup.use_scan
    cli_fixed_boundary_mode = startup.cli_fixed_boundary_mode
    restart_state_eff = startup.restart_state
    restart_solver_state = startup.restart_solver_state
    solver_lower = startup.solver_lower
    axis_infer_missing = startup.axis_infer_missing
    if grid is None and solver_lower in ("vmec_lbfgs", "vmec_gn", "vmec2000_iter"):
        from .vmec_tomnsp import vmec_angle_grid

        grid = vmec_angle_grid(
            ntheta=int(cfg.ntheta),
            nzeta=int(cfg.nzeta),
            nfp=int(cfg.nfp),
            lasym=bool(cfg.lasym),
        )
    stage_policy = _driver_policy_helpers.resolve_fixed_boundary_stage_policy(
        cfg=cfg,
        indata=indata,
        solver_lower=solver_lower,
        cli_fixed_boundary_mode=bool(cli_fixed_boundary_mode),
        accelerated_mode=bool(accelerated_mode),
        multigrid=multigrid,
        max_iter=max_iter,
        max_iter_sentinel=_MAX_ITER_SENTINEL,
        max_iter_overridden=bool(max_iter_overridden),
        restart_state_present=restart_state_eff is not None,
        restart_solver_state_present=restart_solver_state is not None,
        ns_override=ns_override,
        stage_transition_heuristic=stage_transition_heuristic,
        stage_array_list_func=_stage_array_list,
        getenv=os.getenv,
    )
    ns_list_input = stage_policy.ns_list_input
    niter_list_input = stage_policy.niter_list_input
    ftol_list_input = stage_policy.ftol_list_input
    cli_budgeted_multigrid_requested = bool(stage_policy.cli_budgeted_multigrid_requested)
    cli_fixed_boundary_finish_enabled = bool(stage_policy.cli_fixed_boundary_finish_enabled)
    multigrid = bool(stage_policy.multigrid)
    multigrid_user_provided = bool(stage_policy.multigrid_user_provided)
    accelerated_single_grid_default = bool(stage_policy.accelerated_single_grid_default)
    direct_staged_current_driven_3d_cli = bool(stage_policy.direct_staged_current_driven_3d_cli)
    deferred_staged_current_driven_3d_cli = bool(stage_policy.deferred_staged_current_driven_3d_cli)
    max_iter = int(stage_policy.max_iter)
    stage_transition_heuristic = bool(stage_policy.stage_transition_heuristic)
    ns_stages = list(stage_policy.ns_stages)

    sanitize_resume_state_for_stage = partial(
        _sanitize_resume_state_for_driver_stage,
        step_size=step_size,
        indata=indata,
    )
    sanitize_resume_state_for_same_stage = partial(
        _sanitize_resume_state_for_driver_same_stage,
        step_size=step_size,
        indata=indata,
    )
    stage_results: list[SolveVmecResidualResult] = []
    stage_statics: list[VMECStatic] = []

    def _run_cli_explicit_staged_followup(
        *,
        ns_stage_list: list[int],
        niter_stage_list: list[int],
        ftol_stage_list: list[float],
        start_stage_index: int = 0,
        restart_state=None,
        restart_static_prev=None,
        restart_resume_state=None,
        stage_mode_override: str | None = None,
        use_scan_override: bool | None = None,
        performance_mode_override: bool | None = None,
        policy_name: str = "input_multigrid",
    ) -> FixedBoundaryRun:
        return _driver_staging_helpers.run_cli_explicit_staged_followup(
            _stage_runner_context(),
            ns_stage_list=ns_stage_list,
            niter_stage_list=niter_stage_list,
            ftol_stage_list=ftol_stage_list,
            start_stage_index=int(start_stage_index),
            restart_state=restart_state,
            restart_static_prev=restart_static_prev,
            restart_resume_state=restart_resume_state,
            stage_mode_override=stage_mode_override,
            use_scan_override=use_scan_override,
            performance_mode_override=performance_mode_override,
            policy_name=str(policy_name),
        )

    def _stage_runner_context() -> _driver_staging_helpers.FixedBoundaryStageRunnerContext:
        _ = (
            input_path, cfg, step_size, gn_damping, gn_cg_tol, use_restart_triggers, use_direct_fallback, jit_forces,
            jit_precompile, scan_wout_corrector, stage_transition_heuristic, grid,
        )
        return _driver_staging_helpers.FixedBoundaryStageRunnerContext.from_namespace(
            locals(),
            history_size=int(history_size),
            gn_cg_maxiter=int(gn_cg_maxiter),
            vmec_project=bool(vmec_project),
            vmecpp_restart=bool(vmecpp_restart),
            verbose=bool(verbose),
            use_scan=bool(use_scan),
            stage_transition_factor=float(stage_transition_factor),
            stage_transition_scale=float(stage_transition_scale),
            solver_mode_eff=str(solver_mode_eff),
            cli_fixed_boundary_finish_enabled=bool(cli_fixed_boundary_finish_enabled),
            run_fixed_boundary=run_fixed_boundary,
            interp_vmec_state=interp_vmec_state,
            maybe_finish_cli_fixed_boundary_run=_maybe_finish_cli_fixed_boundary_run,
            sanitize_resume_state_for_stage=sanitize_resume_state_for_stage,
            timing_solve_total_s=_timing_solve_total_s,
            accelerated_cli_budgeted_stage_iters=_accelerated_cli_budgeted_stage_iters,
        )

    def _finish_context() -> _driver_finish_helpers.FixedBoundaryFinishContext:
        _ = (
            input_path, cfg, indata, ftol_list_input, ns_list_input, niter_list_input, step_size,
            gn_damping, gn_cg_tol, use_restart_triggers, use_direct_fallback, jit_forces, jit_precompile, use_scan,
            scan_wout_corrector, stage_transition_heuristic, grid,
        )
        return _driver_finish_helpers.FixedBoundaryFinishContext.from_namespace(
            locals(),
            solver_mode_eff=str(solver_mode_eff),
            accelerated_mode=bool(accelerated_mode),
            deferred_staged_current_driven_3d_cli=bool(deferred_staged_current_driven_3d_cli),
            max_iter=int(max_iter),
            max_iter_overridden=bool(max_iter_overridden),
            step_size_sentinel=_STEP_SIZE_SENTINEL,
            history_size=int(history_size),
            gn_cg_maxiter=int(gn_cg_maxiter),
            vmec_project=bool(vmec_project),
            vmecpp_restart=bool(vmecpp_restart),
            multigrid=bool(multigrid),
            multigrid_use_input_niter=bool(multigrid_use_input_niter),
            multigrid_user_provided=bool(multigrid_user_provided),
            verbose=bool(verbose),
            stage_transition_factor=float(stage_transition_factor),
            stage_transition_scale=float(stage_transition_scale),
            policy_backend=str(policy_backend),
            direct_external_provider=bool(direct_external_provider),
            accelerated_single_grid_default=bool(accelerated_single_grid_default),
            run_fixed_boundary=run_fixed_boundary,
            run_cli_explicit_staged_followup=_run_cli_explicit_staged_followup,
            get_stage_results=lambda: stage_results,
            get_stage_statics=lambda: stage_statics,
            sanitize_resume_state_for_stage=sanitize_resume_state_for_stage,
            solve_fixed_boundary_residual_iter=solve_fixed_boundary_residual_iter,
            default_backend_name=_default_backend_name,
            host_update_assembly_driver_default=_host_update_assembly_driver_default,
            default_preconditioner_use_precomputed_tridi=_default_preconditioner_use_precomputed_tridi,
            default_preconditioner_use_lax_tridi=_default_preconditioner_use_lax_tridi,
            resolve_jit_forces_auto_policy=_resolve_jit_forces_auto_policy,
            requested_final_ftol=_requested_final_ftol,
            accelerated_fsq_total_target_from_ftol=_accelerated_fsq_total_target_from_ftol,
            result_final_fsq=_result_final_fsq,
            result_final_residuals=_result_final_residuals,
            result_hits_total_target=_result_hits_total_target,
            result_meets_requested_ftol=_result_meets_requested_ftol,
            sanitize_minimal_resume_state_for_finish=_sanitize_minimal_resume_state_for_finish,
        )

    def _maybe_finish_cli_fixed_boundary_run(
        run_in: FixedBoundaryRun,
        *,
        initial_policy: str,
        enabled: bool,
    ) -> FixedBoundaryRun:
        return _driver_finish_helpers.maybe_finish_cli_fixed_boundary_run(
            run_in,
            initial_policy=initial_policy,
            enabled=bool(enabled),
            context=_finish_context(),
        )

    multigrid_use_input_niter = bool(multigrid_use_input_niter)

    fb_strict_env = os.getenv("VMEC_JAX_FREEB_STRICT", "1").strip().lower()
    fb_strict = fb_strict_env not in ("", "0", "false", "no")
    external_provider_context = _driver_runtime_helpers.resolve_external_field_provider_context(
        external_field_provider_kind=external_field_provider_kind,
        external_field_provider_static=external_field_provider_static,
        external_field_provider_params=external_field_provider_params,
    )
    direct_external_provider = bool(external_provider_context.direct_external_provider)
    external_field_provider_static_eff = external_provider_context.provider_static
    if not bool(direct_external_provider):
        fb_meta, fb_extcur = _free_boundary_static_inputs(cfg, load_fields=False, strict=fb_strict)
    else:
        fb_meta, fb_extcur = None, None

    if bool(cli_budgeted_multigrid_requested):
        budget_total = _accelerated_cli_budgeted_total_iters(total_budget=int(max_iter), ns_stages=ns_stages)
        return _driver_staging_helpers.run_cli_accelerated_budgeted_multigrid(
            _stage_runner_context(),
            ns_stage_list=list(ns_stages),
            warm_start_budget=int(budget_total),
            final_stage_budget=int(max_iter),
        )

    from .modes import vmec_mode_table

    # Precompute boundary coefficients without triggering JAX initialization.
    boundary_coeffs = None
    if restart_state_eff is None:
        boundary_modes = vmec_mode_table(cfg.mpol, cfg.ntor)
        boundary_coeffs = boundary_from_indata(indata, boundary_modes)

    signgs = _resolve_driver_signgs(solver_lower=solver_lower, indata=indata)
    jit_forces = _resolve_vmec2000_jit_forces_policy(solver_lower=solver_lower, jit_forces=jit_forces)

    gamma = indata.get_float("GAMMA", 0.0)
    profiles_from_static = partial(
        _driver_flux_helpers.profiles_from_static,
        indata=indata,
        signgs=signgs,
        flux_profiles_from_indata_host_default_func=flux_profiles_from_indata_host_default,
        flux_profiles_from_indata_func=flux_profiles_from_indata,
        eval_profiles_func=eval_profiles,
    )
    static_profile_cache = _driver_runtime_helpers.StaticProfileCache(
        cfg=cfg,
        indata=indata,
        grid=grid,
        signgs=signgs,
        free_boundary_metadata=fb_meta,
        free_boundary_extcur=fb_extcur,
        build_static_func=build_static,
        boundary_from_indata_func=boundary_from_indata,
        profiles_from_static_func=profiles_from_static,
    )
    static = None
    bdy = None
    flux = None
    prof = None
    pressure = None

    step_size_val = _resolve_driver_step_size(
        step_size=step_size,
        step_size_sentinel=_STEP_SIZE_SENTINEL,
        solver_lower=solver_lower,
        indata=indata,
    )

    if verbose and (solver_lower != "vmec2000_iter" or use_initial_guess):
        _driver_io_helpers.print_fixed_boundary_intro(
            input_path=input_path,
            cfg=cfg,
            solver=solver,
            use_initial_guess=bool(use_initial_guess),
            max_iter=int(max_iter),
            step_size=float(step_size_val),
            history_size=int(history_size),
        )
    elif verbose and (solver_lower == "vmec2000_iter") and (not use_initial_guess):
        _driver_io_helpers.print_vmec2000_run_header(
            input_path=input_path,
            version=os.getenv("VMEC_JAX_VMEC2000_VERSION", "vmec_jax"),
        )

    _initial_guess_with_optional_nojit = partial(
        _driver_solve_helpers.initial_guess_with_optional_nojit,
        indata=indata,
        vmec_project=bool(vmec_project),
        infer_axis_if_missing=bool(axis_infer_missing),
        performance_mode=bool(performance_mode),
        initial_guess_from_boundary_func=initial_guess_from_boundary,
        default_backend_name_func=_default_backend_name,
    )

    if use_initial_guess:
        static, bdy, flux, prof, pressure = static_profile_cache.ensure()
        return _driver_solve_helpers.fixed_boundary_initial_guess_run(
            cfg=cfg,
            indata=indata,
            static=static,
            boundary=bdy,
            flux=flux,
            profiles=prof,
            signgs=signgs,
            restart_state=restart_state_eff,
            initial_guess_func=_initial_guess_with_optional_nojit,
            maybe_dump_xc_init=_driver_debug_helpers.maybe_dump_xc_init,
            run_container=FixedBoundaryRun,
        )

    if performance_mode:
        if solver_lower == "vmec2000_iter":
            solver_lower = "vmec2000_iter_fast"

    # Fast mode keeps minimal history only when not printing (verbose=False).
    scan_minimal_default = True if (bool(performance_mode) and (not bool(verbose))) else None

    solver = solver_lower
    if solver in ("vmec2000_iter_fast", "vmec2000_scan"):
        # Respect an explicitly-passed use_scan=False (e.g. CPU CLI fast path
        # that uses the Python loop instead of lax.scan).  Only default to
        # scan=True when the caller did not explicitly opt out.
        if use_scan is not False:
            use_scan = True
        solver = "vmec2000_iter"
    # Parity mode defaults to the VMEC2000 non-scan control path unless
    # explicitly forced via environment variables.
    if solver == "vmec2000_iter" and (not bool(performance_mode)):
        use_scan = False
    if os.getenv("VMEC_JAX_USE_SCAN", "") not in ("", "0"):
        use_scan = True
    if solver in _driver_solve_helpers.FIXED_BOUNDARY_OPTIMIZER_SOLVERS:
        static, bdy, flux, prof, pressure = static_profile_cache.ensure()
        res = _driver_solve_helpers.run_fixed_boundary_optimizer_solver(
            solver=solver,
            restart_state=restart_state_eff,
            static=static,
            boundary=bdy,
            indata=indata,
            flux=flux,
            pressure=pressure,
            signgs=signgs,
            gamma=gamma,
            max_iter=int(max_iter),
            step_size=float(step_size_val),
            history_size=int(history_size),
            gn_damping=gn_damping,
            gn_cg_tol=gn_cg_tol,
            gn_cg_maxiter=int(gn_cg_maxiter),
            verbose=bool(verbose),
            initial_guess_func=_initial_guess_with_optional_nojit,
            solve_fixed_boundary_gd_func=solve_fixed_boundary_gd,
            solve_fixed_boundary_lbfgs_func=solve_fixed_boundary_lbfgs,
        )
    elif solver == "vmec2000_iter":
        # Stage controls.
        nstep = len(ns_stages)
        niter_array = indata.get("NITER_ARRAY", None)
        ftol_array = indata.get("FTOL_ARRAY", None)
        niter_list = _stage_array_list(niter_array)
        ftol_list = _stage_array_list(ftol_array)
        niter_stages, ftol_stages, niter_stages_input, _ftol_stages_input = _resolve_vmec2000_stage_controls(
            nstep=int(nstep),
            niter_list=niter_list,
            ftol_list=ftol_list,
            max_iter=int(max_iter),
            max_iter_overridden=bool(max_iter_overridden),
            multigrid_use_input_niter=bool(multigrid_use_input_niter),
            multigrid_user_provided=bool(multigrid_user_provided),
            accelerated_single_grid_default=bool(accelerated_single_grid_default),
            indata=indata,
        )

        env_precompile_stages = os.getenv("VMEC_JAX_PRECOMPILE_STAGES", "0")
        precompile_stages = env_precompile_stages.strip().lower() not in ("", "0", "false", "no")

        staged = _driver_staging_helpers.run_vmec2000_staged_solve(
            _driver_staging_helpers.Vmec2000StagedSolveContext.from_namespace(
                locals(),
                solver_mode_eff=str(solver_mode_eff),
                accelerated_mode=bool(accelerated_mode),
                performance_mode=bool(performance_mode),
                use_scan=bool(use_scan),
                cli_fixed_boundary_mode=bool(cli_fixed_boundary_mode),
                direct_staged_current_driven_3d_cli=bool(direct_staged_current_driven_3d_cli),
                multigrid=bool(multigrid),
                multigrid_user_provided=bool(multigrid_user_provided),
                accelerated_single_grid_default=bool(accelerated_single_grid_default),
                direct_external_provider=bool(direct_external_provider),
                policy_backend=str(policy_backend),
                stage_transition_heuristic=bool(stage_transition_heuristic),
                stage_transition_factor=float(stage_transition_factor),
                stage_transition_scale=float(stage_transition_scale),
                precompile_stages=bool(precompile_stages),
                vmecpp_restart=bool(vmecpp_restart),
                external_field_provider_static=external_field_provider_static_eff,
                verbose=bool(verbose),
                signgs=signgs,
                step_size_val=float(step_size_val),
                ns_stages=list(ns_stages),
                niter_stages=list(niter_stages),
                ftol_stages=list(ftol_stages),
                t_start=float(t_start),
                build_static_cfg=static_profile_cache.build_static_cfg,
                initial_guess_with_optional_nojit=_initial_guess_with_optional_nojit,
                resolve_jit_forces=_resolve_jit_forces_auto_policy,
                interp_vmec_state=interp_vmec_state,
                mode_table_func=vmec_mode_table,
                maybe_dump_xc_init=_driver_debug_helpers.maybe_dump_xc_init,
                maybe_disable_scan_by_parity_guard=_driver_dynamic_scan_helpers.maybe_disable_scan_by_parity_guard,
                resolve_stage_jit_settings=_resolve_stage_jit_settings,
                accelerated_fsq_total_target_from_ftol=_accelerated_fsq_total_target_from_ftol,
                host_update_assembly_driver_default=_host_update_assembly_driver_default,
                default_preconditioner_use_precomputed_tridi=_default_preconditioner_use_precomputed_tridi,
                default_preconditioner_use_lax_tridi=_default_preconditioner_use_lax_tridi,
                solve_fixed_boundary_residual_iter=solve_fixed_boundary_residual_iter,
                maybe_select_dynamic_scan_mode=_driver_dynamic_scan_helpers.maybe_select_dynamic_scan_mode,
                dynamic_scan_probe_settings=_dynamic_scan_probe_settings,
                vmec_histories_match=_vmec_histories_match,
                vmec_history_relerr=_vmec_history_relerr,
                maybe_precompile_fixed_boundary_stage=_driver_solve_helpers.maybe_precompile_fixed_boundary_stage,
                run_fixed_boundary_stage_solve=_driver_solve_helpers.run_fixed_boundary_stage_solve,
                result_meets_requested_ftol=_result_meets_requested_ftol,
                stage_switch_reason_from_progress=_stage_switch_reason_from_progress,
                merge_stage_chunk_results=_merge_stage_chunk_results,
                result_with_diag=_result_with_diag,
                maybe_rerun_scan_abort_stage=_driver_solve_helpers.maybe_rerun_scan_abort_stage,
                assemble_multigrid_stage_result=_driver_result_helpers.assemble_multigrid_stage_result,
                maybe_apply_scan_wout_corrector=_driver_solve_helpers.maybe_apply_scan_wout_corrector,
                copy_final_force_payload=_copy_final_force_payload,
                timing_solve_total_s=_timing_solve_total_s,
                requested_final_ftol=_requested_final_ftol,
                result_final_residuals=_result_final_residuals,
                result_hits_total_target=_result_hits_total_target,
                finalize_fixed_boundary_convergence_result=_driver_result_helpers.finalize_fixed_boundary_convergence_result,
                print_vmec2000_run_summary=_driver_io_helpers.print_vmec2000_run_summary,
                default_backend_name=_default_backend_name,
                deepcopy_func=deepcopy,
                getenv=os.getenv,
                perf_counter=time.perf_counter,
            )
        )
        res = staged.result
        static = staged.static
        stage_results = staged.stage_results
        stage_statics = staged.stage_statics
    else:
        raise ValueError(
            f"Unknown solver: {solver!r} (expected 'gd', 'lbfgs', 'vmec_lbfgs', 'vmec_gn', or 'vmec2000_iter')"
        )

    if verbose and solver != "vmec2000_iter":
        n_iter = int(getattr(res, "n_iter", -1))
        w_final = float(res.w_history[-1]) if getattr(res, "w_history", None) is not None else float("nan")
        if getattr(res, "grad_rms_history", None) is not None and len(res.grad_rms_history) > 0:
            grad_final = float(res.grad_rms_history[-1])
        else:
            grad_final = float("nan")
        print(f"[vmec_jax] finished: n_iter={n_iter} w={w_final:.8e} grad_rms={grad_final:.3e}")

    if flux is None or prof is None or pressure is None:
        if static is None:
            static = static_profile_cache.build_static_cfg(cfg)
        flux, prof, pressure = static_profile_cache.profiles_for_static(static)
    flux, prof = _final_flux_profiles_from_state(
        indata=indata,
        static_in=static,
        state=res.state,
        signgs=signgs,
        flux_local=flux,
        prof_local=prof,
        pressure_local=pressure,
    )

    run_out = FixedBoundaryRun(
        cfg=cfg,
        indata=indata,
        static=static,
        state=res.state,
        result=res,
        flux=flux,
        profiles=prof,
        signgs=signgs,
    )
    cli_initial_policy = "multigrid" if bool(multigrid) and (len(ns_stages) > 1) else "single_grid"
    return _maybe_finish_cli_fixed_boundary_run(
        run_out,
        initial_policy=cli_initial_policy,
        enabled=bool(cli_fixed_boundary_finish_enabled),
    )


def run_free_boundary(input_path: str | Path, **kwargs):
    """Run a free-boundary vmec_jax solve.

    Parameters
    ----------
    input_path:
        Path to a VMEC-style ``input.*`` file with ``LFREEB = T`` and a valid
        ``MGRID_FILE`` entry.
    **kwargs:
        Forwarded directly to :func:`run_fixed_boundary`. Common options include
        ``max_iter``, ``verbose``, ``use_initial_guess``, ``vmec_project``,
        ``solver_mode``, ``jit_forces``, and ``limit_update_rms``.

    Returns
    -------
    FixedBoundaryRun
        Run container with the parsed input, static data, final state, and
        solver diagnostics.

    Raises
    ------
    ValueError
        If the input deck is not a free-boundary case.

    Notes
    -----
    This wrapper intentionally shares the internal implementation with
    :func:`run_fixed_boundary`. The only behavioral difference is that
    ``run_free_boundary`` validates the mode up front, which makes scripts and
    examples clearer and avoids silently running the wrong branch.
    """
    cfg, _ = load_config(str(input_path))
    if not bool(cfg.lfreeb):
        raise ValueError(
            f"Input {input_path!s} is not a free-boundary case (LFREEB=F). "
            "Use run_fixed_boundary(...) instead."
        )
    return run_fixed_boundary(input_path, **kwargs)
