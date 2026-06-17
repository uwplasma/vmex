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
_STAGE_CHUNK_DIAG_KEYS = _driver_result_helpers.STAGE_CHUNK_DIAG_KEYS
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


def residual_scalars_from_state(
    *,
    state,
    static,
    indata,
    signgs: int,
    wout=None,
    use_vmec_synthesis: bool = True,
):
    """Compatibility wrapper for VMEC-style residual scalar construction."""

    return _driver_output_helpers.residual_scalars_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=signgs,
        wout=wout,
        use_vmec_synthesis=use_vmec_synthesis,
    )


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


def save_npz(path: str | Path, **arrays) -> Path:
    """Save arrays into a NumPy `.npz` file and return the path."""
    return _driver_io_helpers.save_npz(path, **arrays)


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
    t_start = time.perf_counter()
    max_iter_overridden = max_iter is not _MAX_ITER_SENTINEL
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
    # Default to 64-bit for VMEC parity; users can opt out via JAX_ENABLE_X64=0.
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
    if not bool(_solver_device_context_active):
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
        auto_cli_fixed_boundary_mode=bool(_auto_cli_fixed_boundary_mode),
        default_non_autodiff_policy_func=_default_non_autodiff_solver_policy_for_backend,
        default_use_scan_func=_default_use_scan_for_backend,
    )
    solver_mode_explicit = bool(initial_policy.solver_mode_explicit)
    solver_mode_eff = str(initial_policy.solver_mode_eff)
    accelerated_mode = bool(initial_policy.accelerated_mode)
    performance_mode = bool(initial_policy.performance_mode)
    use_scan = bool(initial_policy.use_scan)
    cli_fixed_boundary_mode = bool(initial_policy.cli_fixed_boundary_mode)
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
    restart_wout = restart_context.restart_wout
    restart_solver_state = restart_context.restart_solver_state
    solver_lower = str(solver).lower()
    # VMEC starts from the input axis coefficients and only recomputes the
    # axis (guess_axis) after a bad-Jacobian trigger. For vmec2000_iter we
    # follow that behavior by default and allow opt-in axis inference via env.
    axis_infer_missing = _resolve_axis_infer_missing_policy(
        solver_lower=solver_lower,
        performance_mode=bool(performance_mode),
    )

    routed_run = _driver_solve_helpers.maybe_run_fixed_boundary_in_solver_device_context(
        input_path=input_path,
        solver_device=solver_device,
        solver_device_context_active=bool(_solver_device_context_active),
        cfg=cfg,
        indata=indata,
        solver_lower=solver_lower,
        cli_fixed_boundary_mode=bool(cli_fixed_boundary_mode),
        accelerated_mode=bool(accelerated_mode),
        restart_state_present=(restart_state_eff is not None) or (restart_wout_path is not None),
        restart_solver_state_present=restart_solver_state is not None,
        recursive_run_kwargs={
            "solver": solver,
            "solver_mode": solver_mode_eff,
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
            "use_scan": bool(use_scan),
            "performance_mode": bool(performance_mode),
            "scan_wout_corrector": scan_wout_corrector,
            "stage_transition_heuristic": stage_transition_heuristic,
            "stage_transition_factor": float(stage_transition_factor),
            "stage_transition_scale": float(stage_transition_scale),
            "grid": grid,
            "ns_override": ns_override,
            "restart_state": restart_state,
            "restart_wout_path": restart_wout_path,
            "restart_solver_state": restart_solver_state,
            "cli_fixed_boundary_mode": bool(cli_fixed_boundary_mode),
            "_auto_cli_fixed_boundary_mode": bool(_auto_cli_fixed_boundary_mode),
        },
        run_fixed_boundary_func=run_fixed_boundary,
        replace_func=replace,
        as_list_like_func=_as_list_like,
        default_backend_name_func=_default_backend_name,
        resolve_solver_device_name_func=_resolve_fixed_boundary_solver_device_name,
        getenv=os.getenv,
    )
    if routed_run is not None:
        return routed_run
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
    user_explicitly_staged_cli = bool(stage_policy.user_explicitly_staged_cli)
    cli_fixed_boundary_finish_enabled = bool(stage_policy.cli_fixed_boundary_finish_enabled)
    multigrid = bool(stage_policy.multigrid)
    multigrid_user_provided = bool(stage_policy.multigrid_user_provided)
    accelerated_single_grid_default = bool(stage_policy.accelerated_single_grid_default)
    current_driven_3d_cli = bool(stage_policy.current_driven_3d_cli)
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

    def _run_cli_accelerated_budgeted_multigrid(
        *,
        ns_stage_list: list[int],
        warm_start_budget: int,
        final_stage_budget: int,
    ):
        return _driver_staging_helpers.run_cli_accelerated_budgeted_multigrid(
            _stage_runner_context(),
            ns_stage_list=ns_stage_list,
            warm_start_budget=int(warm_start_budget),
            final_stage_budget=int(final_stage_budget),
        )

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

    def _finish_stage_results():
        try:
            return stage_results
        except NameError:
            return ()

    def _stage_runner_context() -> _driver_staging_helpers.FixedBoundaryStageRunnerContext:
        return _driver_staging_helpers.FixedBoundaryStageRunnerContext(
            input_path=input_path,
            cfg=cfg,
            step_size=step_size,
            history_size=int(history_size),
            gn_damping=gn_damping,
            gn_cg_tol=gn_cg_tol,
            gn_cg_maxiter=int(gn_cg_maxiter),
            vmec_project=bool(vmec_project),
            use_restart_triggers=use_restart_triggers,
            vmecpp_restart=bool(vmecpp_restart),
            use_direct_fallback=use_direct_fallback,
            verbose=bool(verbose),
            jit_forces=jit_forces,
            jit_precompile=jit_precompile,
            use_scan=bool(use_scan),
            scan_wout_corrector=scan_wout_corrector,
            stage_transition_heuristic=stage_transition_heuristic,
            stage_transition_factor=float(stage_transition_factor),
            stage_transition_scale=float(stage_transition_scale),
            grid=grid,
            solver_mode_eff=str(solver_mode_eff),
            cli_fixed_boundary_finish_enabled=bool(cli_fixed_boundary_finish_enabled),
            run_fixed_boundary=run_fixed_boundary,
            interp_vmec_state=interp_vmec_state,
            maybe_finish_cli_fixed_boundary_run=_maybe_finish_cli_fixed_boundary_run,
            sanitize_resume_state_for_stage=sanitize_resume_state_for_stage,
            timing_solve_total_s=_timing_solve_total_s,
            accelerated_cli_budgeted_stage_iters=_accelerated_cli_budgeted_stage_iters,
        )

    def _finish_stage_statics():
        try:
            return stage_statics
        except NameError:
            return ()

    def _finish_context() -> _driver_finish_helpers.FixedBoundaryFinishContext:
        return _driver_finish_helpers.FixedBoundaryFinishContext(
            input_path=input_path,
            cfg=cfg,
            indata=indata,
            solver_mode_eff=str(solver_mode_eff),
            accelerated_mode=bool(accelerated_mode),
            ftol_list_input=ftol_list_input,
            ns_list_input=ns_list_input,
            niter_list_input=niter_list_input,
            deferred_staged_current_driven_3d_cli=bool(deferred_staged_current_driven_3d_cli),
            max_iter=int(max_iter),
            max_iter_overridden=bool(max_iter_overridden),
            step_size=step_size,
            step_size_sentinel=_STEP_SIZE_SENTINEL,
            history_size=int(history_size),
            gn_damping=gn_damping,
            gn_cg_tol=gn_cg_tol,
            gn_cg_maxiter=int(gn_cg_maxiter),
            vmec_project=bool(vmec_project),
            use_restart_triggers=use_restart_triggers,
            vmecpp_restart=bool(vmecpp_restart),
            use_direct_fallback=use_direct_fallback,
            multigrid=bool(multigrid),
            multigrid_use_input_niter=bool(multigrid_use_input_niter),
            multigrid_user_provided=bool(multigrid_user_provided),
            verbose=bool(verbose),
            jit_forces=jit_forces,
            jit_precompile=jit_precompile,
            use_scan=use_scan,
            scan_wout_corrector=scan_wout_corrector,
            stage_transition_heuristic=stage_transition_heuristic,
            stage_transition_factor=float(stage_transition_factor),
            stage_transition_scale=float(stage_transition_scale),
            grid=grid,
            policy_backend=str(policy_backend),
            direct_external_provider=bool(direct_external_provider),
            accelerated_single_grid_default=bool(accelerated_single_grid_default),
            run_fixed_boundary=run_fixed_boundary,
            run_cli_explicit_staged_followup=_run_cli_explicit_staged_followup,
            get_stage_results=_finish_stage_results,
            get_stage_statics=_finish_stage_statics,
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
        return _run_cli_accelerated_budgeted_multigrid(
            ns_stage_list=list(ns_stages),
            warm_start_budget=int(budget_total),
            final_stage_budget=int(max_iter),
        )

    # Precompute boundary coefficients without triggering JAX initialization.
    boundary_coeffs = None
    if restart_state_eff is None:
        from .modes import vmec_mode_table

        boundary_modes = vmec_mode_table(cfg.mpol, cfg.ntor)
        boundary_coeffs = boundary_from_indata(indata, boundary_modes)

    signgs = _resolve_driver_signgs(solver_lower=solver_lower, indata=indata)
    jit_forces = _resolve_vmec2000_jit_forces_policy(solver_lower=solver_lower, jit_forces=jit_forces)

    gamma = indata.get_float("GAMMA", 0.0)
    static = None
    static_final = None
    bdy = None
    flux = None
    prof = None
    pressure = None

    def _build_static_cfg(cfg_in: VMECConfig) -> VMECStatic:
        if bool(cfg_in.lfreeb):
            return build_static(
                cfg_in,
                grid=grid,
                mgrid_metadata=fb_meta,
                free_boundary_extcur=fb_extcur,
            )
        return build_static(cfg_in, grid=grid)

    def _profiles_from_static(static_in: VMECStatic):
        return _driver_flux_helpers.profiles_from_static(
            indata=indata,
            static_in=static_in,
            signgs=signgs,
            flux_profiles_from_indata_host_default_func=flux_profiles_from_indata_host_default,
            flux_profiles_from_indata_func=flux_profiles_from_indata,
            eval_profiles_func=eval_profiles,
        )

    def _ensure_static_profiles() -> None:
        nonlocal static, bdy, flux, prof, pressure
        if static is None:
            static = _build_static_cfg(cfg)
        if bdy is None:
            bdy = boundary_from_indata(indata, static.modes)
        if flux is None or prof is None or pressure is None:
            flux, prof, pressure = _profiles_from_static(static)

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

    def _initial_guess_with_optional_nojit(static_in, bdy_in, *, force_disable_jit: bool = False):
        return _driver_solve_helpers.initial_guess_with_optional_nojit(
            static_in,
            bdy_in,
            indata,
            vmec_project=bool(vmec_project),
            infer_axis_if_missing=bool(axis_infer_missing),
            performance_mode=bool(performance_mode),
            force_disable_jit=bool(force_disable_jit),
            initial_guess_from_boundary_func=initial_guess_from_boundary,
            default_backend_name_func=_default_backend_name,
        )

    if use_initial_guess:
        _ensure_static_profiles()
        if restart_state_eff is not None:
            st0 = restart_state_eff
        else:
            st0 = _initial_guess_with_optional_nojit(static, bdy)
            _driver_debug_helpers.maybe_dump_xc_init(state=st0, static=static, label="init")
        return FixedBoundaryRun(
            cfg=cfg,
            indata=indata,
            static=static,
            state=st0,
            result=None,
            flux=flux,
            profiles=prof,
            signgs=signgs,
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
        _ensure_static_profiles()
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

        # Run coarse -> fine stages with VMEC `interp.f` interpolation.
        stage_results: list[SolveVmecResidualResult] = []
        stage_statics: list[VMECStatic] = []
        stage_offsets: list[int] = []
        from .modes import vmec_mode_table

        header_modes = vmec_mode_table(cfg.mpol, cfg.ntor)
        nmodes_header = int(np.asarray(header_modes.m).size)

        state = restart_state_eff
        static_prev = None
        static_final = None
        resume_state_stage = restart_solver_state
        multigrid_resume = False
        if multigrid:
            # Default to VMEC2000 behavior (reset time-step state per stage).
            env_resume = os.getenv("VMEC_JAX_MULTIGRID_RESUME", "0")
            multigrid_resume = env_resume.strip().lower() not in ("", "0", "false", "no")

        def _resolve_jit_forces(flag: bool | str, static_i: VMECStatic, niter_i: int) -> bool:
            return _resolve_jit_forces_auto_policy(flag, static_i, niter_i)

        env_precompile_stages = os.getenv("VMEC_JAX_PRECOMPILE_STAGES", "0")
        precompile_stages = env_precompile_stages.strip().lower() not in ("", "0", "false", "no")

        prev_stage_fsq = None
        stage_mode_history: list[str] = []
        stage_wall_s: list[float] = []
        stage_solve_total_s: list[float] = []
        ftol_last = None
        step_size_last = None
        for i, (ns_i, niter_i, ftol_i) in enumerate(zip(ns_stages, niter_stages, ftol_stages)):
            if int(niter_i) <= 0:
                continue
            stage_t0 = time.perf_counter()
            stage_accelerated_mode = bool(accelerated_mode)
            if (
                bool(stage_accelerated_mode)
                and bool(direct_staged_current_driven_3d_cli)
                and bool(cfg.lasym)
            ):
                # LASYM current-driven 3D staged runs remain noticeably more
                # sensitive in lambda than in geometry. The mixed accelerated
                # controller was slightly faster here, but it consistently
                # degraded the final lambda channels versus the conservative
                # staged baseline. Keep this class fully on the conservative
                # controller until the lambda mismatch is closed numerically.
                stage_accelerated_mode = False
            stage_mode_i = "accelerated" if bool(stage_accelerated_mode) else "parity"
            stage_mode_history.append("accelerated" if bool(stage_accelerated_mode) else "parity")
            if verbose:
                print(
                    f"  NS = {int(ns_i):4d} NO. FOURIER MODES = {nmodes_header:4d} "
                    f"FTOLV = {float(ftol_i):10.3E} NITER = {int(niter_i):6d}",
                    flush=True,
                )
                print("  PROCESSOR COUNT - RADIAL:    1", flush=True)
                print("", flush=True)
                if bool(cfg.lasym):
                    print(
                        "  ITER    FSQR      FSQZ      FSQL    RAX(v=0)  ZAX(v=0)    DELT       WMHD",
                        flush=True,
                    )
                else:
                    print(
                        "  ITER    FSQR      FSQZ      FSQL    RAX(v=0)    DELT       WMHD",
                        flush=True,
                    )

            cfg_i = replace(cfg, ns=int(ns_i))
            static_i = _build_static_cfg(cfg_i)
            scan_mode = bool(use_scan) if bool(stage_accelerated_mode) else False
            if stage_accelerated_mode and bool(use_scan):
                # In accelerated mode the default is to use scan (lax.scan is
                # faster on GPU; on CPU the caller can override via use_scan=False).
                scan_mode = not bool(cfg_i.lfreeb)
            if bool(cfg.lasym):
                # For LASYM fixed-boundary stages, allow scan as a candidate in
                # the default fast path and let the automatic selector decide
                # whether the warmed scan route is both safe and worthwhile.
                lasym_scan_env = os.getenv("VMEC_JAX_LASYM_USE_SCAN", "auto").strip().lower()
                if lasym_scan_env in ("0", "false", "no", "off"):
                    scan_mode = False
                elif lasym_scan_env not in ("", "auto"):
                    scan_mode = True
            # Note: scan is now enabled for current_driven_3d_cli on CPU as well.
            # Benchmarks show lax.scan is faster than the Python-loop NumPy hot-path
            # (26s cold vs 36s cold for LandremanPaul2021_QA_lowres), with identical
            # numerical results.
            # Optional scan-parity guard: probe a few iterations and disable scan
            # if it diverges from the non-scan VMEC2000 path.
            scan_guard_default = "0"
            scan_guard_env = os.getenv("VMEC_JAX_SCAN_PARITY_GUARD", scan_guard_default).strip().lower()
            scan_guard_enabled = scan_guard_env not in ("", "0", "false", "no")
            if (not accelerated_mode) and scan_mode and scan_guard_enabled and int(niter_i) >= 3:
                probe_iters = min(10, int(niter_i))
                try:
                    guard_rtol = float(os.getenv("VMEC_JAX_SCAN_GUARD_RTOL", "1e-3"))
                    guard_atol = float(os.getenv("VMEC_JAX_SCAN_GUARD_ATOL", "1e-12"))
                    probe_kwargs = dict(
                        indata=indata,
                        signgs=signgs,
                        ftol=float(ftol_i),
                        max_iter=int(probe_iters),
                        step_size=float(step_size_val),
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
                        use_restart_triggers=True if use_restart_triggers is None else bool(use_restart_triggers),
                        vmecpp_restart=bool(vmecpp_restart),
                        use_direct_fallback=False,
                        stage_prev_fsq=None,
                        stage_transition_factor=float(stage_transition_factor),
                        stage_transition_scale=float(stage_transition_scale),
                        resume_state=None,
                        verbose=False,
                        verbose_vmec2000_table=False,
                        jit_precompile=False,
                        jit_warmup_iters=0,
                        scan_minimal_default=scan_minimal_default,
                    )
                    res_probe_scan = solve_fixed_boundary_residual_iter(
                        state,
                        static_i,
                        jit_forces=_resolve_jit_forces(jit_forces, static_i, int(probe_iters)),
                        use_scan=True,
                        **probe_kwargs,
                    )
                    res_probe_direct = solve_fixed_boundary_residual_iter(
                        state,
                        static_i,
                        jit_forces=_resolve_jit_forces(jit_forces, static_i, int(probe_iters)),
                        use_scan=False,
                        **probe_kwargs,
                    )
                    fsqr_scan = np.asarray(res_probe_scan.fsqr2_history)
                    fsqz_scan = np.asarray(res_probe_scan.fsqz2_history)
                    fsql_scan = np.asarray(res_probe_scan.fsql2_history)
                    fsqr_ref = np.asarray(res_probe_direct.fsqr2_history)
                    fsqz_ref = np.asarray(res_probe_direct.fsqz2_history)
                    fsql_ref = np.asarray(res_probe_direct.fsql2_history)
                    mismatch = False
                    if fsqr_scan.size == fsqr_ref.size == probe_iters:
                        if not np.allclose(fsqr_scan, fsqr_ref, rtol=guard_rtol, atol=guard_atol):
                            mismatch = True
                        if not np.allclose(fsqz_scan, fsqz_ref, rtol=guard_rtol, atol=guard_atol):
                            mismatch = True
                        if not np.allclose(fsql_scan, fsql_ref, rtol=guard_rtol, atol=guard_atol):
                            mismatch = True
                    else:
                        mismatch = True
                    if mismatch:
                        scan_mode = False
                        if bool(verbose):
                            print(
                                "[vmec_jax] scan parity guard: disabling scan for this stage (probe mismatch)",
                                flush=True,
                            )
                except Exception as exc:
                    # If probe fails, fall back to the safe (non-scan) path.
                    scan_mode = False
                    if bool(verbose):
                        print(
                            f"[vmec_jax] scan parity guard probe failed ({type(exc).__name__}); using non-scan for this stage.",
                            flush=True,
                        )
            jit_forces_base = _resolve_jit_forces(jit_forces, static_i, int(niter_i))
            jit_settings = _resolve_stage_jit_settings(
                jit_forces_base=bool(jit_forces_base),
                scan_mode=bool(scan_mode),
                solver=solver,
                performance_mode=bool(performance_mode),
                jit_precompile=jit_precompile,
            )
            jit_forces_eff = bool(jit_settings.jit_forces_eff)
            jit_precompile_eff = bool(jit_settings.jit_precompile_eff)
            jit_warmup_iters = int(jit_settings.jit_warmup_iters)
            jit_precompile_noscan = bool(jit_settings.jit_precompile_noscan)
            jit_warmup_noscan = int(jit_settings.jit_warmup_noscan)
            if i == 0:
                if state is None:
                    if boundary_coeffs is None:
                        raise ValueError("boundary_coeffs missing; cannot build initial guess")
                    state = _initial_guess_with_optional_nojit(
                        static_i,
                        boundary_coeffs,
                        force_disable_jit=bool(jit_warmup_iters > 0),
                    )
                    _driver_debug_helpers.maybe_dump_xc_init(state=state, static=static_i, label="stage0")
            else:
                state = interp_vmec_state(
                    state,
                    m=static_prev.modes.m,
                    n=static_prev.modes.n,
                    lthreed=bool(static_prev.cfg.lthreed),
                    lconm1=bool(getattr(static_prev.cfg, "lconm1", True)),
                    ns_new=int(ns_i),
                )
            state_stage_start = state
            static_prev = static_i
            static_final = static_i

            stage_offsets.append(sum(int(np.asarray(r.w_history).size) for r in stage_results))
            vmec2000_ctrl = True
            stage_prev_fsq = prev_stage_fsq if bool(stage_transition_heuristic) else None
            stage_light_history = (
                True
                if (
                    bool(performance_mode)
                    and (not bool(verbose))
                    and ((not bool(cfg.lfreeb)) or bool(direct_external_provider))
                )
                else None
            )
            stage_resume_state_mode = "minimal" if stage_accelerated_mode else None
            is_last_stage = (i == len(ns_stages) - 1)
            _final_cpu_scan_env = os.getenv("VMEC_JAX_FINAL_STAGE_CPU_SCAN", "1").strip().lower()
            _final_cpu_scan_disabled = _final_cpu_scan_env in ("0", "false", "no")
            if (
                bool(cli_fixed_boundary_mode)
                and scan_mode
                and (_default_backend_name() == "cpu")
                and _final_cpu_scan_disabled
            ):
                # lax.scan on CPU CLI is consistently faster than the NumPy
                # hot-path when the JAX compilation disk cache is warm (which it
                # is after the first CLI run).  Benchmarks show 2-2.5× speedup
                # for small cases (circular/shaped tokamak, QH warm-start) and
                # ~5% speedup for medium cases (QA_lowres NS=50) when using scan.
                # The scan path also benefits GPU runs maximally (10-100×).
                # Disable via VMEC_JAX_FINAL_STAGE_CPU_SCAN=0 to revert to the
                # NumPy hot-path (useful for debugging or first-run profiling).
                scan_mode = False
            stage_fsq_total_target = (
                _accelerated_fsq_total_target_from_ftol(float(ftol_i))
                if (stage_accelerated_mode and not is_last_stage)
                else None
            )
            stage_host_update_assembly = _host_update_assembly_driver_default(
                cfg=cfg_i,
                performance_mode=bool(performance_mode),
                backend=_default_backend_name(),
                use_scan=bool(scan_mode),
            )
            stage_preconditioner_use_precomputed_tridi = _default_preconditioner_use_precomputed_tridi(
                cfg=cfg_i,
                backend=policy_backend,
                performance_mode=bool(performance_mode),
                use_scan=bool(scan_mode),
                direct_external_provider=bool(direct_external_provider),
            )
            stage_preconditioner_use_lax_tridi = _default_preconditioner_use_lax_tridi(
                cfg=cfg_i,
                backend=policy_backend,
                performance_mode=bool(performance_mode),
                use_scan=bool(scan_mode),
                direct_external_provider=bool(direct_external_provider),
            )
            stage_limit_update_rms = False if limit_update_rms is None else bool(limit_update_rms)
            solve_kwargs = dict(
                indata=indata,
                signgs=signgs,
                ftol=float(ftol_i),
                max_iter=int(niter_i),
                step_size=float(step_size_val),
                include_constraint_force=True,
                apply_m1_constraints=True,
                precond_radial_alpha=0.5,
                precond_lambda_alpha=0.5,
                mode_diag_exponent=0.0,
                auto_flip_force=False,
                divide_by_scalxc_for_update=False,
                lambda_update_scale=1.0,
                enforce_vmec_lambda_axis=True,
                vmec2000_control=vmec2000_ctrl,
                strict_update=True,
                backtracking=False,
                limit_update_rms=stage_limit_update_rms,
                reference_mode=False,
                use_restart_triggers=True if use_restart_triggers is None else bool(use_restart_triggers),
                vmecpp_restart=bool(vmecpp_restart),
                use_direct_fallback=False,
                stage_prev_fsq=stage_prev_fsq,
                stage_transition_factor=float(stage_transition_factor),
                stage_transition_scale=float(stage_transition_scale),
                resume_state=resume_state_stage,
                verbose=bool(verbose),
                verbose_vmec2000_table=bool(verbose),
                use_scan=bool(scan_mode),
                jit_warmup_iters=int(jit_warmup_iters),
                jit_precompile=bool(jit_precompile_eff),
                scan_minimal_default=scan_minimal_default,
                light_history=stage_light_history,
                resume_state_mode=stage_resume_state_mode,
                fsq_total_target=stage_fsq_total_target,
                host_update_assembly=stage_host_update_assembly,
                preconditioner_use_precomputed_tridi=stage_preconditioner_use_precomputed_tridi,
                preconditioner_use_lax_tridi=stage_preconditioner_use_lax_tridi,
                external_field_provider_kind=external_field_provider_kind,
                external_field_provider_static=external_field_provider_static_eff,
                external_field_provider_params=external_field_provider_params,
                free_boundary_activate_fsq=free_boundary_activate_fsq,
                return_final_force_payload=True,
            )
            dynamic_scan_default = "1" if bool(cfg.lasym) else "0"
            dynamic_scan_env = os.getenv("VMEC_JAX_DYNAMIC_SCAN", dynamic_scan_default).strip().lower()
            dynamic_scan = dynamic_scan_env not in ("", "0", "false", "no")
            if (
                (not accelerated_mode)
                and
                dynamic_scan
                and bool(performance_mode)
                and bool(scan_mode)
                and bool(vmec2000_ctrl)
                and int(niter_i) > 1
            ):
                pre_iters, timed_probe, probe_backend = _dynamic_scan_probe_settings(int(niter_i))
                if pre_iters > 0:
                    fsq_tol_env = os.getenv("VMEC_JAX_DYNAMIC_SCAN_FSQ_RTOL", "1e-6").strip()
                    try:
                        fsq_tol = float(fsq_tol_env)
                    except Exception:
                        fsq_tol = 1e-6
                    hist_atol_env = os.getenv("VMEC_JAX_DYNAMIC_SCAN_ATOL", "1e-12").strip()
                    try:
                        hist_atol = float(hist_atol_env)
                    except Exception:
                        hist_atol = 1e-12
                    pre_kwargs = dict(solve_kwargs)
                    pre_kwargs.update(
                        {
                            "max_iter": int(pre_iters),
                            "verbose": False,
                            "verbose_vmec2000_table": False,
                            "jit_warmup_iters": 0,
                            "jit_precompile": False,
                            # Keep full histories in the probe so we can compare
                            # the warmed scan/non-scan traces, not just a single
                            # terminal residual scalar.
                            "scan_minimal_default": False,
                        }
                    )

                    def _run_pref(*, use_scan_flag: bool):
                        kwargs = dict(pre_kwargs)
                        kwargs["use_scan"] = bool(use_scan_flag)
                        kwargs["resume_state"] = deepcopy(resume_state_stage)
                        state_probe = deepcopy(state_stage_start)
                        if not bool(jit_forces_base):
                            try:
                                import jax
                                with jax.disable_jit():
                                    return solve_fixed_boundary_residual_iter(
                                        state_probe,
                                        static_i,
                                        jit_forces=False,
                                        **kwargs,
                                    )
                            except Exception:
                                return solve_fixed_boundary_residual_iter(
                                    state_probe,
                                    static_i,
                                    jit_forces=False,
                                    **kwargs,
                                )
                        return solve_fixed_boundary_residual_iter(
                            state_probe,
                            static_i,
                            jit_forces=True,
                            **kwargs,
                        )

                    if timed_probe:
                        # Warm both variants before timing so the selector compares
                        # steady-state iteration cost rather than one-off compile cost.
                        _ = _run_pref(use_scan_flag=False)
                        _ = _run_pref(use_scan_flag=True)

                        t0 = time.perf_counter()
                        res_pref_noscan = _run_pref(use_scan_flag=False)
                        t_noscan = time.perf_counter() - t0
                        t0 = time.perf_counter()
                        res_pref_scan = _run_pref(use_scan_flag=True)
                        t_scan = time.perf_counter() - t0
                    else:
                        t_noscan = None
                        t_scan = None
                        res_pref_scan = _run_pref(use_scan_flag=True)
                        res_pref_noscan = _run_pref(use_scan_flag=False)

                    fsq_ok = _vmec_histories_match(
                        res_pref_scan,
                        res_pref_noscan,
                        rtol=float(fsq_tol),
                        atol=float(hist_atol),
                    )
                    choose_scan = bool(fsq_ok) and ((not timed_probe) or (t_scan < t_noscan))
                    scan_mode = bool(choose_scan)
                    solve_kwargs["use_scan"] = bool(scan_mode)
                    if bool(verbose):
                        if not bool(fsq_ok):
                            print(
                                "[vmec_jax] dynamic scan probe mismatch: "
                                f"w={_vmec_history_relerr(res_pref_scan.w_history, res_pref_noscan.w_history):.3e} "
                                f"fsqr={_vmec_history_relerr(res_pref_scan.fsqr2_history, res_pref_noscan.fsqr2_history):.3e} "
                                f"fsqz={_vmec_history_relerr(res_pref_scan.fsqz2_history, res_pref_noscan.fsqz2_history):.3e} "
                                f"fsql={_vmec_history_relerr(res_pref_scan.fsql2_history, res_pref_noscan.fsql2_history):.3e}",
                                flush=True,
                            )
                        if timed_probe:
                            print(
                                "[vmec_jax] dynamic scan selection: "
                                f"backend={probe_backend} scan={t_scan:.3f}s noscan={t_noscan:.3f}s "
                                f"fsq_ok={fsq_ok} -> use_scan={scan_mode}",
                                flush=True,
                            )
                        else:
                            print(
                                "[vmec_jax] dynamic scan parity probe: "
                                f"backend={probe_backend} iters={pre_iters} "
                                f"fsq_ok={fsq_ok} -> use_scan={scan_mode}",
                                flush=True,
                            )
            if bool(precompile_stages) and bool(jit_forces_eff):
                try:
                    precompile_kwargs = dict(solve_kwargs)
                    precompile_kwargs.update(
                        {
                            "precompile_only": True,
                            "verbose": False,
                            "verbose_vmec2000_table": False,
                            "jit_warmup_iters": 0,
                            "jit_precompile": True,
                            "max_iter": 1,
                        }
                    )
                    solve_fixed_boundary_residual_iter(
                        state,
                        static_i,
                        jit_forces=True,
                        **precompile_kwargs,
                    )
                except Exception:
                    pass
            def _run_stage_solve(
                *,
                state_i,
                kwargs_i: dict,
                jit_forces_flag: bool,
            ) -> SolveVmecResidualResult:
                if not bool(jit_forces_flag):
                    try:
                        import jax
                        with jax.disable_jit():
                            return solve_fixed_boundary_residual_iter(
                                state_i,
                                static_i,
                                jit_forces=False,
                                **kwargs_i,
                            )
                    except Exception:
                        return solve_fixed_boundary_residual_iter(
                            state_i,
                            static_i,
                            jit_forces=False,
                            **kwargs_i,
                        )
                return solve_fixed_boundary_residual_iter(
                    state_i,
                    static_i,
                    jit_forces=True,
                    **kwargs_i,
                )

            explicit_stage_monitor = (
                bool(stage_accelerated_mode)
                and (niter_stages_input is not None)
                and int(nstep) > 1
                and int(i) > 0
            )
            explicit_stage_chunk = min(int(niter_i), max(int(indata.get_int("NSTEP", 1)), 200))
            explicit_stage_target = _accelerated_fsq_total_target_from_ftol(float(ftol_i))
            explicit_stage_monitor_jit_forces = bool(jit_forces_base)

            if bool(explicit_stage_monitor) and int(explicit_stage_chunk) < int(niter_i):
                chunk_results: list[SolveVmecResidualResult] = []
                chunk_state = state
                chunk_resume_state = resume_state_stage
                stage_switch_reason = None
                stage_monitor_used = True
                stage_monitor_scan = bool(scan_mode) and str(policy_backend).lower() in (
                    "gpu",
                    "cuda",
                    "rocm",
                    "tpu",
                )
                remaining_budget = int(niter_i)
                stage_first_chunk = True

                chunk_budget = min(int(explicit_stage_chunk), int(remaining_budget))
                chunk_kwargs = dict(solve_kwargs)
                chunk_kwargs.update(
                    {
                        "max_iter": int(chunk_budget),
                        "resume_state": chunk_resume_state,
                        "stage_prev_fsq": stage_prev_fsq if bool(stage_first_chunk) else None,
                        "use_scan": bool(stage_monitor_scan),
                        "jit_warmup_iters": int(jit_warmup_noscan),
                        "jit_precompile": bool(jit_precompile_noscan),
                    }
                )
                res_chunk = _run_stage_solve(
                    state_i=chunk_state,
                    kwargs_i=chunk_kwargs,
                    jit_forces_flag=bool(explicit_stage_monitor_jit_forces),
                )
                chunk_results.append(res_chunk)
                stage_first_chunk = False

                completed_chunk_iters = min(int(chunk_budget), int(res_chunk.n_iter) + 1)
                remaining_budget = max(0, int(remaining_budget) - int(completed_chunk_iters))
                chunk_state = res_chunk.state
                chunk_resume_state = sanitize_resume_state_for_same_stage(
                    res_chunk.diagnostics.get("resume_state")
                )

                strict_chunk = bool(_result_meets_requested_ftol(res_chunk, ftol=float(ftol_i)))
                if (not bool(strict_chunk)) and int(remaining_budget) > 0:
                    try:
                        chunk_w = np.asarray(res_chunk.w_history, dtype=float).reshape(-1)
                        if chunk_w.size > 0:
                            stage_switch_reason = _stage_switch_reason_from_progress(
                                start_total_fsq=float(chunk_w[0]),
                                best_total_fsq=float(np.min(chunk_w)),
                                target_total_fsq=float(explicit_stage_target),
                                chunk_iters=int(completed_chunk_iters),
                                remaining_budget=int(remaining_budget),
                            )
                    except Exception:
                        stage_switch_reason = None

                if (stage_switch_reason is None) and (not bool(strict_chunk)) and int(remaining_budget) > 0:
                    tail_kwargs = dict(solve_kwargs)
                    tail_kwargs.update(
                        {
                            "max_iter": int(remaining_budget),
                            "resume_state": chunk_resume_state,
                            "stage_prev_fsq": None,
                            "use_scan": bool(stage_monitor_scan),
                            "jit_warmup_iters": 0,
                            "jit_precompile": False,
                        }
                    )
                    res_tail = _run_stage_solve(
                        state_i=chunk_state,
                        kwargs_i=tail_kwargs,
                        jit_forces_flag=bool(explicit_stage_monitor_jit_forces),
                    )
                    chunk_results.append(res_tail)
                elif (stage_switch_reason is None) and (not bool(strict_chunk)):
                    stage_switch_reason = "budget_exhausted"

                if stage_switch_reason is not None:
                    if bool(verbose):
                        print(
                            "[vmec_jax] accelerated staged solve cannot meet requested FTOL; "
                            f"switching stage ns={int(ns_i)} to parity mode "
                            f"({stage_switch_reason}).",
                            flush=True,
                        )
                    fallback_kwargs = dict(solve_kwargs)
                    fallback_kwargs.update(
                        {
                            "use_scan": False,
                            "resume_state": resume_state_stage,
                            "max_iter": int(niter_i),
                            "jit_warmup_iters": int(jit_warmup_noscan),
                            "jit_precompile": bool(jit_precompile_noscan),
                            "light_history": None,
                            "resume_state_mode": None,
                            "fsq_total_target": None,
                            "host_update_assembly": False,
                        }
                    )
                    res_i = _run_stage_solve(
                        state_i=state_stage_start,
                        kwargs_i=fallback_kwargs,
                        jit_forces_flag=bool(jit_forces_base),
                    )
                    res_i = _result_with_diag(
                        res_i,
                        accelerated_stage_chunked=bool(stage_monitor_used or len(chunk_results) > 0),
                        accelerated_stage_early_switch=True,
                        accelerated_stage_switch_reason=str(stage_switch_reason),
                        accelerated_stage_probe_chunk_iters=np.asarray(
                            [int(r.n_iter) + 1 for r in chunk_results],
                            dtype=int,
                        ),
                        accelerated_stage_effective_mode="parity",
                    )
                    stage_mode_i = "parity"
                else:
                    res_i = _merge_stage_chunk_results(
                        chunk_results,
                        mode_i=str(stage_mode_i),
                    )
            else:
                res_i = _run_stage_solve(
                    state_i=state,
                    kwargs_i=solve_kwargs,
                    jit_forces_flag=bool(jit_forces_eff),
                )
                # Auto-fast fallback: if scan hits a bad-Jacobian path, rerun the stage
                # in the parity-safe non-scan mode.
                if (not accelerated_mode) and bool(performance_mode) and bool(scan_mode):
                    try:
                        if bool(res_i.diagnostics.get("vmec2000_scan", False)) and bool(
                            res_i.diagnostics.get("abort_scan", False)
                        ):
                            if bool(verbose):
                                print(
                                    "[vmec_jax] scan abort detected; rerunning stage in parity mode.",
                                    flush=True,
                                )
                            solve_kwargs_fallback = dict(solve_kwargs)
                            solve_kwargs_fallback.update(
                                {
                                    "use_scan": False,
                                    "resume_state": resume_state_stage,
                                    "jit_warmup_iters": int(jit_warmup_noscan),
                                    "jit_precompile": bool(jit_precompile_noscan),
                                }
                            )
                            res_i = _run_stage_solve(
                                state_i=state_stage_start,
                                kwargs_i=solve_kwargs_fallback,
                                jit_forces_flag=bool(jit_forces_base),
                            )
                    except Exception:
                        pass
            stage_mode_history[-1] = str(stage_mode_i)
            stage_wall_s.append(float(time.perf_counter() - stage_t0))
            try:
                stage_timing = res_i.diagnostics.get("timing", {})
            except Exception:
                stage_timing = {}
            stage_solve_total_s.append(_timing_solve_total_s(stage_timing))
            stage_results.append(res_i)
            stage_statics.append(static_i)
            try:
                w_hist = np.asarray(res_i.w_history)
                prev_stage_fsq = float(w_hist[-1]) if w_hist.size else None
            except Exception:
                prev_stage_fsq = None
            if multigrid_resume and i < (nstep - 1):
                resume_state_stage = sanitize_resume_state_for_stage(res_i.diagnostics.get("resume_state"))
            state = stage_results[-1].state
            static_prev = static_i
            ftol_last = float(ftol_i)
            step_size_last = float(step_size_val)

        diag = dict(stage_results[-1].diagnostics)
        diag["solver_mode"] = str(solver_mode_eff)
        diag["accelerated_mode"] = bool(accelerated_mode)
        diag["accelerated_scan"] = bool(accelerated_mode) and bool(diag.get("use_scan", False))
        diag["multigrid_user_provided"] = bool(multigrid_user_provided)
        diag["accelerated_single_grid_default"] = bool(accelerated_single_grid_default)
        diag["multigrid_ns_stages"] = np.asarray(ns_stages, dtype=int)
        diag["multigrid_niter_stages"] = np.asarray(niter_stages, dtype=int)
        diag["multigrid_ftol_stages"] = np.asarray(ftol_stages, dtype=float)
        diag["multigrid_stage_offsets"] = np.asarray(stage_offsets, dtype=int)
        diag["multigrid_stage_modes"] = np.asarray(stage_mode_history, dtype=object)
        diag["multigrid_stage_wall_s"] = np.asarray(stage_wall_s, dtype=float)
        diag["multigrid_stage_solve_total_s"] = np.asarray(stage_solve_total_s, dtype=float)
        # Record whether the final stage exhausted its NITER budget (matches
        # xvmec2000 behavior: terminate normally when NITER is reached).
        # n_iter is 0-indexed (999 means 1000 iterations completed).
        # The +1 correction is only applied when NITER_ARRAY was explicitly provided
        # by the user (niter_stages_input is not None); when the per-stage budget is
        # derived from a single NITER value, exhausting it doesn't constitute a
        # "EXECUTION TERMINATED NORMALLY" signal and the parity finisher should still
        # run if convergence hasn't been reached.
        try:
            _final_stage_niter = int(stage_results[-1].n_iter)
            _final_stage_budget = int(niter_stages[-1]) if niter_stages else 0
            if niter_stages_input is not None:
                # NITER_ARRAY was explicitly given: use 0-indexed check
                _exhausted = bool(_final_stage_niter + 1 >= _final_stage_budget)
            else:
                # Budget derived from NITER (one value per stage): use plain >=
                _exhausted = bool(_final_stage_niter >= _final_stage_budget)
            diag["multigrid_final_stage_niter_exhausted"] = _exhausted
        except Exception:
            diag["multigrid_final_stage_niter_exhausted"] = False

        # Concatenate the common history keys that are useful for parity debugging.
        for k in (
            "step_status_history",
            "restart_reason_history",
            "pre_restart_reason_history",
            "time_step_history",
            "res0_history",
            "res1_history",
            "fsq_prev_history",
            "bad_growth_streak_history",
            "iter1_history",
            "bcovar_update_history",
            "include_edge_history",
            "zero_m1_history",
            "dt_eff_history",
            "update_rms_history",
            "w_curr_history",
            "w_try_history",
            "w_try_ratio_history",
            "restart_path_history",
            "min_tau_history",
            "max_tau_history",
            "bad_jacobian_history",
            "fsq1_history",
            "fsqr1_history",
            "fsqz1_history",
            "fsql1_history",
            "r00_history",
            "z00_history",
            "wb_history",
            "wp_history",
            "w_vmec_history",
            "rz_norm_history",
            "f_norm1_history",
            "gcr2_p_history",
            "gcz2_p_history",
            "gcl2_p_history",
        ):
            if any(k in r.diagnostics for r in stage_results):
                diag[k] = np.concatenate(
                    [np.asarray(r.diagnostics.get(k, np.zeros((0,), dtype=float))) for r in stage_results]
                )

        res = _copy_final_force_payload(SolveVmecResidualResult(
            state=state,
            n_iter=int(sum(int(r.n_iter) + 1 for r in stage_results) - 1),
            w_history=_cat_result_history(stage_results, "w_history"),
            fsqr2_history=_cat_result_history(stage_results, "fsqr2_history"),
            fsqz2_history=_cat_result_history(stage_results, "fsqz2_history"),
            fsql2_history=_cat_result_history(stage_results, "fsql2_history"),
            grad_rms_history=_cat_result_history(stage_results, "grad_rms_history"),
            step_history=_cat_result_history(stage_results, "step_history"),
            diagnostics=diag,
        ), stage_results[-1])
        # Optional scan corrector: run a single non-scan VMEC2000 step to
        # re-anchor the final state before writing wout outputs.
        try:
            use_scan_any = any(bool(r.diagnostics.get("vmec2000_scan", False)) for r in stage_results)
        except Exception:
            use_scan_any = False
        if accelerated_mode and scan_wout_corrector is None:
            scan_wout_corrector = False
        if scan_wout_corrector is None:
            scan_wout_env = os.getenv("VMEC_JAX_SCAN_WOUT_CORRECTOR", "0").strip().lower()
            scan_wout_corrector = scan_wout_env not in ("", "0", "false", "no")
        if use_scan_any and bool(scan_wout_corrector):
            try:
                resume_state_corr = res.diagnostics.get("resume_state", None)
                static_corr = static_prev if static_prev is not None else _build_static_cfg(cfg)
                ftol_corr = float(ftol_last) if ftol_last is not None else float(indata.get_float("FTOL", 1e-13))
                step_corr = float(step_size_last) if step_size_last is not None else 1.0
                corr_kwargs = dict(
                    indata=indata,
                    signgs=signgs,
                    ftol=ftol_corr,
                    max_iter=1,
                    step_size=step_corr,
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
                    use_restart_triggers=True if use_restart_triggers is None else bool(use_restart_triggers),
                    vmecpp_restart=bool(vmecpp_restart),
                    stage_prev_fsq=None,
                    stage_transition_factor=float(stage_transition_factor),
                    stage_transition_scale=float(stage_transition_scale),
                    use_direct_fallback=False,
                    resume_state=resume_state_corr,
                    verbose=False,
                    verbose_vmec2000_table=False,
                    jit_precompile=False,
                    jit_warmup_iters=0,
                    use_scan=False,
                    scan_minimal_default=scan_minimal_default,
                    light_history=True if accelerated_mode else None,
                    resume_state_mode="minimal" if accelerated_mode else None,
                    fsq_total_target=(
                        _accelerated_fsq_total_target_from_ftol(float(ftol_corr)) if accelerated_mode else None
                    ),
                    return_final_force_payload=True,
                )
                res_corr = solve_fixed_boundary_residual_iter(
                    res.state,
                    static_corr,
                    jit_forces=_resolve_jit_forces(jit_forces, static_corr, 1),
                    **corr_kwargs,
                )
                diag = dict(res.diagnostics)
                diag["scan_wout_corrector"] = True
                diag["scan_wout_corrector_iters"] = int(res_corr.n_iter)
                res = _copy_final_force_payload(SolveVmecResidualResult(
                    state=res_corr.state,
                    n_iter=res.n_iter,
                    w_history=res.w_history,
                    fsqr2_history=res.fsqr2_history,
                    fsqz2_history=res.fsqz2_history,
                    fsql2_history=res.fsql2_history,
                    grad_rms_history=res.grad_rms_history,
                    step_history=res.step_history,
                    diagnostics=diag,
                ), res_corr)
            except Exception:
                pass
        final_requested_ftol = _requested_final_ftol(indata=indata, ftol_list_input=ftol_list_input)
        final_target_fsq = _accelerated_fsq_total_target_from_ftol(float(final_requested_ftol))
        final_residuals = _result_final_residuals(res)
        final_diag = dict(res.diagnostics)
        final_diag["requested_ftol"] = float(final_requested_ftol)
        final_diag["fsq_total_target"] = (
            final_target_fsq if (final_diag.get("fsq_total_target", None) is not None or bool(accelerated_mode)) else None
        )
        if final_residuals is not None:
            final_diag["final_fsqr"] = float(final_residuals[0])
            final_diag["final_fsqz"] = float(final_residuals[1])
            final_diag["final_fsql"] = float(final_residuals[2])
        final_diag["converged_strict"] = bool(_result_meets_requested_ftol(res, ftol=float(final_requested_ftol)))
        final_diag["converged_by_total_fsq"] = bool(
            _result_hits_total_target(res, fsq_total_target=float(final_target_fsq))
        )
        final_diag["converged"] = bool(final_diag["converged_strict"])
        res = _copy_final_force_payload(SolveVmecResidualResult(
            state=res.state,
            n_iter=int(res.n_iter),
            w_history=np.asarray(res.w_history),
            fsqr2_history=np.asarray(res.fsqr2_history),
            fsqz2_history=np.asarray(res.fsqz2_history),
            fsql2_history=np.asarray(res.fsql2_history),
            grad_rms_history=np.asarray(res.grad_rms_history),
            step_history=np.asarray(res.step_history),
            diagnostics=final_diag,
        ), res)
        # Use the static from the last executed stage (static_prev) when
        # available.  This ensures that static.cfg.ns matches the actual
        # solved state's ns even when the final NS_ARRAY stage is skipped
        # because the iteration budget (max_iter) was exhausted by earlier
        # stages — e.g. max_iter=1500 with NITER_ARRAY=[600,1000,1000]
        # only reaches ns=31, so static_prev.cfg.ns=31 while cfg.ns=50.
        # Falling back to _build_static_cfg(cfg) when static_prev is None
        # preserves the existing behavior for single-stage solves.
        static = static_prev if static_prev is not None else _build_static_cfg(cfg)
        if verbose and solver == "vmec2000_iter":
            _driver_io_helpers.print_vmec2000_run_summary(
                input_path=input_path,
                result=res,
                niter_stage=int(niter_i),
                total_time=time.perf_counter() - t_start,
            )
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
            static = _build_static_cfg(cfg)
        flux, prof, pressure = _profiles_from_static(static)
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
