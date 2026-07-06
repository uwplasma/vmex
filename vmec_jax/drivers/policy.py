"""Backend and solver-policy helpers for :mod:`vmec_jax.driver`.

The public driver is intentionally kept thin enough to read as a workflow:
parse inputs, build static data, choose solver policy, run stages, and emit
results.  This module holds the small policy and budget helpers used by that
workflow so they can be unit-tested without importing the whole driver stack.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np


VALID_SOLVER_MODES = frozenset(("default", "parity", "accelerated"))
VALID_FIXED_BOUNDARY_FINISH_POLICIES = frozenset(("auto", "none", "converge"))
FSQ_COMPONENT_NAMES = ("fsqr", "fsqz", "fsql")
_BUNDLED_ACCELERATED_SCAN_PROFILE = Path(__file__).resolve().parents[1] / "resources" / "accelerated_scan_profile.json"


@dataclass(frozen=True)
class InitialFixedBoundaryPolicy:
    """Resolved driver policy before static-data construction."""

    requested_solver_device: str
    policy_backend: str
    solver_mode_explicit: bool
    solver_mode_eff: str
    performance_mode: bool
    accelerated_mode: bool
    use_scan: bool
    use_scan_policy_source: str
    use_scan_policy_detail: str
    cli_fixed_boundary_mode: bool


@dataclass(frozen=True)
class ScanDefaultDecision:
    """Default scan-loop decision and provenance for accelerated VMEC solves."""

    use_scan: bool
    source: str
    detail: str


@dataclass(frozen=True)
class StageJitSettings:
    """Resolved JIT/precompile/warmup policy for one VMEC2000 stage."""

    jit_forces_eff: bool
    jit_precompile_eff: bool
    jit_warmup_iters: int
    jit_precompile_noscan: bool
    jit_warmup_noscan: int


@dataclass(frozen=True)
class FixedBoundaryStagePolicy:
    """Resolved multigrid/staging policy for one fixed-boundary solve."""

    ns_list_input: list | None
    niter_list_input: list | None
    ftol_list_input: list | None
    cli_budgeted_multigrid_requested: bool
    user_explicitly_staged_cli: bool
    cli_fixed_boundary_finish_enabled: bool
    multigrid: bool
    multigrid_user_provided: bool
    accelerated_single_grid_default: bool
    current_driven_3d_cli: bool
    direct_staged_current_driven_3d_cli: bool
    deferred_staged_current_driven_3d_cli: bool
    max_iter: int
    stage_transition_heuristic: bool
    ns_stages: list[int]


@dataclass(frozen=True)
class FixedBoundarySolverDispatchPolicy:
    """Final solver alias and scan policy used by the public driver."""

    solver: str
    use_scan: bool
    scan_minimal_default: bool | None


def host_update_assembly_driver_default(
    *,
    cfg,
    performance_mode: bool,
    backend: str,
    use_scan: bool,
) -> bool:
    """Resolve the public driver default for CPU host-update assembly."""

    backend_name = str(backend).strip().lower()
    # Host NumPy update assembly is fastest for low/moderate CPU solves because
    # it avoids per-step JAX dispatch. On large spectral/radial grids repeated
    # host state assembly can dominate; let the fused strict-update JIT take
    # those cases instead. The default covers the bundled finite-beta QH row
    # (ns=51, mpol=5, ntor=5) where profiling shows the host path is much faster.
    nrange = int(getattr(cfg, "ntor", 0)) + 1
    if bool(getattr(cfg, "lasym", False)):
        nrange = 2 * int(getattr(cfg, "ntor", 0)) + 1
    update_work = int(getattr(cfg, "ns", 0)) * int(getattr(cfg, "mpol", 0)) * int(nrange)
    try:
        work_limit = int(os.getenv("VMEC_JAX_HOST_UPDATE_CPU_WORK_LIMIT", "4096"))
    except Exception:
        work_limit = 4096
    use_host_update_default = update_work < work_limit
    default = bool(performance_mode) and (backend_name == "cpu") and (not bool(use_scan)) and use_host_update_default
    env = os.getenv("VMEC_JAX_HOST_UPDATE_ASSEMBLY", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return (backend_name == "cpu") and (not bool(use_scan))
    if env in ("0", "false", "no", "off"):
        return False
    return bool(default)


def resolve_fixed_boundary_solver_device_name(
    *,
    solver_device: str | None,
    backend: str,
    cfg,
    indata,
    solver_lower: str,
    cli_fixed_boundary_mode: bool,
    accelerated_mode: bool,
    ns_list_input,
    niter_list_input,
    restart_state_present: bool,
    restart_solver_state_present: bool,
) -> str | None:
    """Return an optional JAX default-device override for fixed-boundary runs.

    ``solver_device=None`` / ``"auto"`` / ``"default"`` inherit JAX's active
    default device. Pass ``"cpu"`` or ``"gpu"`` to explicitly run the solver
    under that device context. In particular, GPU-enabled JAX installations are
    not silently routed back to CPU.
    """

    del (
        backend,
        cfg,
        indata,
        solver_lower,
        cli_fixed_boundary_mode,
        accelerated_mode,
        ns_list_input,
        niter_list_input,
        restart_state_present,
        restart_solver_state_present,
    )
    name = "auto" if solver_device is None else str(solver_device).strip().lower()
    if name in ("", "none", "auto", "default"):
        return None
    return name


def normalize_solver_mode(*, solver_mode: str | None, performance_mode: bool) -> str:
    """Normalize public solver-mode aliases to a supported policy name."""

    if solver_mode is None:
        return "default" if bool(performance_mode) else "parity"
    mode = str(solver_mode).strip().lower()
    aliases = {
        "fast": "default",
        "safe": "parity",
        "reference": "parity",
        "memory": "parity",
        "low-memory": "parity",
        "low_memory": "parity",
        "perf": "accelerated",
    }
    mode = aliases.get(mode, mode)
    if mode not in VALID_SOLVER_MODES:
        valid = ", ".join(sorted(VALID_SOLVER_MODES))
        raise ValueError(f"Unknown solver_mode {solver_mode!r}. Expected one of: {valid}.")
    return mode


def normalize_fixed_boundary_finish_policy(finish_policy: str | None) -> str:
    """Normalize fixed-boundary post-solve finish policy aliases.

    ``"auto"`` records finish diagnostics but does not spend hidden extra
    iteration budgets after the input deck has been exhausted. ``"none"``
    disables even that diagnostic finisher, which is useful for exact-budget
    profiling. ``"converge"`` explicitly enables VMEC-style finish attempts for
    fixed-boundary ``vmec2000_iter`` runs.
    """

    if finish_policy is None:
        return "auto"
    policy = str(finish_policy).strip().lower().replace("_", "-")
    aliases = {
        "": "auto",
        "default": "auto",
        "bounded": "none",
        "budgeted": "none",
        "exact-budget": "none",
        "no-finish": "none",
        "off": "none",
        "false": "none",
        "0": "none",
        "finish": "converge",
        "finished": "converge",
        "converged": "converge",
        "on": "converge",
        "true": "converge",
        "1": "converge",
    }
    policy = aliases.get(policy, policy)
    if policy not in VALID_FIXED_BOUNDARY_FINISH_POLICIES:
        valid = ", ".join(sorted(VALID_FIXED_BOUNDARY_FINISH_POLICIES))
        raise ValueError(f"Unknown finish_policy {finish_policy!r}. Expected one of: {valid}.")
    return policy


def requested_solver_device_name(solver_device: str | None) -> str:
    """Normalize the public solver-device request without selecting a device."""

    return "auto" if solver_device is None else str(solver_device).strip().lower()


def policy_backend_for_requested_device(*, requested_solver_device: str, default_backend: str) -> str:
    """Choose the backend name used for policy decisions before device routing."""

    requested = str(requested_solver_device).strip().lower()
    if requested in ("cpu", "gpu"):
        return requested
    return str(default_backend).strip().lower()


def resolve_initial_fixed_boundary_policy(
    *,
    requested_solver_device: str,
    policy_backend: str,
    indata,
    cfg,
    solver: str,
    solver_mode: str | None,
    performance_mode: bool,
    use_scan: bool | None,
    verbose: bool,
    grid,
    cli_fixed_boundary_mode: bool,
    auto_cli_fixed_boundary_mode: bool,
    default_non_autodiff_policy_func=None,
    default_scan_decision_func=None,
    default_use_scan_func=None,
) -> InitialFixedBoundaryPolicy:
    """Resolve the initial run policy shared by CLI and API entry points.

    This helper is intentionally pure policy logic: it does not import JAX,
    build grids, prepare free-boundary metadata, or enter solver loops. Keeping
    it isolated makes the front of ``run_fixed_boundary`` testable without
    accidentally changing VMEC control-flow semantics.
    """

    solver_mode_explicit = solver_mode is not None
    performance_mode_eff = bool(performance_mode)
    solver_mode_input = solver_mode
    if default_non_autodiff_policy_func is None:
        default_non_autodiff_policy_func = default_non_autodiff_solver_policy_for_backend
    if default_scan_decision_func is None:
        default_scan_decision_func = default_scan_decision_for_backend
    if default_use_scan_func is None:
        default_use_scan_func = default_use_scan_for_backend
    if solver_mode_input is None and performance_mode_eff:
        solver_mode_input, performance_mode_eff = default_non_autodiff_policy_func(
            indata,
            policy_backend,
        )
    solver_mode_eff = normalize_solver_mode(
        solver_mode=solver_mode_input,
        performance_mode=bool(performance_mode_eff),
    )

    if use_scan is None:
        if bool(solver_mode_explicit):
            use_scan_eff = True
            use_scan_policy_source = "solver_mode_explicit"
            use_scan_policy_detail = "explicit_solver_mode_defaults_to_scan"
        else:
            try:
                scan_decision = default_scan_decision_func(indata, policy_backend, solver_mode_eff)
                use_scan_eff = bool(scan_decision.use_scan)
                use_scan_policy_source = str(scan_decision.source)
                use_scan_policy_detail = str(scan_decision.detail)
            except Exception:
                use_scan_eff = bool(default_use_scan_func(indata, policy_backend, solver_mode_eff))
                use_scan_policy_source = "default_callback"
                use_scan_policy_detail = "boolean_default_use_scan_func"
    else:
        use_scan_eff = bool(use_scan)
        use_scan_policy_source = "explicit"
        use_scan_policy_detail = "user_supplied_use_scan"

    accelerated_mode = solver_mode_eff == "accelerated"
    performance_mode_eff = solver_mode_eff != "parity"
    cli_fixed_boundary_mode_eff = bool(cli_fixed_boundary_mode) or (
        bool(auto_cli_fixed_boundary_mode)
        and (not bool(solver_mode_explicit))
        and (not bool(getattr(cfg, "lfreeb", False)))
        and bool(performance_mode_eff)
        and (grid is None)
        and str(solver).strip().lower() == "vmec2000_iter"
    )

    return InitialFixedBoundaryPolicy(
        requested_solver_device=str(requested_solver_device),
        policy_backend=str(policy_backend),
        solver_mode_explicit=bool(solver_mode_explicit),
        solver_mode_eff=str(solver_mode_eff),
        performance_mode=bool(performance_mode_eff),
        accelerated_mode=bool(accelerated_mode),
        use_scan=bool(use_scan_eff),
        use_scan_policy_source=str(use_scan_policy_source),
        use_scan_policy_detail=str(use_scan_policy_detail),
        cli_fixed_boundary_mode=bool(cli_fixed_boundary_mode_eff),
    )


def resolve_fixed_boundary_solver_dispatch(
    *,
    solver_lower: str,
    performance_mode: bool,
    verbose: bool,
    use_scan: bool | None,
    getenv=os.getenv,
) -> FixedBoundarySolverDispatchPolicy:
    """Normalize final fixed-boundary solver aliases and scan policy."""

    solver_eff = str(solver_lower)
    use_scan_eff = use_scan
    if bool(performance_mode) and solver_eff == "vmec2000_iter":
        solver_eff = "vmec2000_iter_fast"

    scan_minimal_default = True if (bool(performance_mode) and (not bool(verbose))) else None

    if solver_eff in ("vmec2000_iter_fast", "vmec2000_scan"):
        # Respect explicitly-passed use_scan=False. Only default to scan=True
        # when the caller did not explicitly opt out.
        if use_scan_eff is not False:
            use_scan_eff = True
        solver_eff = "vmec2000_iter"
    # Parity mode defaults to the VMEC2000 non-scan control path unless
    # explicitly forced via environment variables.
    if solver_eff == "vmec2000_iter" and (not bool(performance_mode)):
        use_scan_eff = False
    if getenv("VMEC_JAX_USE_SCAN", "") not in ("", "0"):
        use_scan_eff = True

    return FixedBoundarySolverDispatchPolicy(
        solver=str(solver_eff),
        use_scan=bool(use_scan_eff),
        scan_minimal_default=scan_minimal_default,
    )


def resolve_axis_infer_missing_policy(
    *,
    solver_lower: str,
    performance_mode: bool,
    getenv=os.getenv,
) -> bool:
    """Resolve whether missing axis coefficients are inferred before iteration.

    VMEC2000 parity starts from the input axis and lets the bad-Jacobian path
    improve it. Performance mode may infer the axis up front to avoid that
    extra first-stage reset. This helper keeps the env-controlled decision
    visible and testable without moving solver math.
    """

    solver_name = str(solver_lower).strip().lower()
    axis_infer_missing = solver_name != "vmec2000_iter"
    if solver_name != "vmec2000_iter":
        return bool(axis_infer_missing)

    enable_axis_infer = str(getenv("VMEC_JAX_ENABLE_AXIS_INFER", "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    disable_axis_infer = str(getenv("VMEC_JAX_DISABLE_AXIS_INFER", "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if enable_axis_infer:
        axis_infer_missing = True
    if disable_axis_infer:
        axis_infer_missing = False
    if (not disable_axis_infer) and bool(performance_mode):
        # Conservative VMEC-style raw-axis start remains the parity default.
        axis_infer_missing = True
    return bool(axis_infer_missing)


def accelerated_fsq_total_target_from_ftol(ftol: float) -> float:
    """Collapse per-component FTOL into an equivalent total-residual target."""

    return max(0.0, float(ftol)) * float(len(FSQ_COMPONENT_NAMES))


def resolve_driver_signgs(*, solver_lower: str, indata) -> int:
    """Resolve the driver sign convention, preserving VMEC2000 parity."""

    solver_name = str(solver_lower).strip().lower()
    if solver_name in ("vmec2000_iter", "vmec2000_scan", "vmec2000_iter_fast"):
        # VMEC readin.f initializes signgs=-1 and flips theta later if needed.
        return -1
    signgs = int(indata.get_int("SIGNGS", -1))
    return signgs if signgs in (-1, 1) else -1


def resolve_vmec2000_jit_forces_policy(*, solver_lower: str, jit_forces, getenv=os.getenv):
    """Apply VMEC2000-specific env overrides to the force-JIT policy."""

    solver_name = str(solver_lower).strip().lower()
    if solver_name not in ("vmec2000_iter", "vmec2000_scan", "vmec2000_iter_fast"):
        return jit_forces
    force_jit_env = str(getenv("VMEC_JAX_VMEC2000_FORCE_JIT", "")).strip().lower()
    force_nojit_env = str(getenv("VMEC_JAX_VMEC2000_FORCE_NOJIT", "")).strip().lower()
    if force_jit_env not in ("", "0", "false", "no"):
        return True
    if force_nojit_env not in ("", "0", "false", "no"):
        return False
    if isinstance(jit_forces, str) and jit_forces.strip().lower() == "auto":
        return True
    return jit_forces


def resolve_fixed_boundary_stage_policy(
    *,
    cfg,
    indata,
    solver_lower: str,
    cli_fixed_boundary_mode: bool,
    accelerated_mode: bool,
    multigrid,
    max_iter,
    max_iter_sentinel,
    max_iter_overridden: bool,
    restart_state_present: bool,
    restart_solver_state_present: bool,
    ns_override,
    stage_transition_heuristic,
    stage_array_list_func=None,
    getenv=os.getenv,
) -> FixedBoundaryStagePolicy:
    """Resolve fixed-boundary multigrid, stage, and iteration-budget policy.

    This helper contains only policy derived from the input deck and public
    driver flags.  It does not build VMEC static data, run a solve, or touch
    free-boundary provider state, which keeps the long driver workflow easier
    to audit against VMEC2000 staging semantics.
    """

    if stage_array_list_func is None:
        stage_array_list_func = as_list_like

    solver_name = str(solver_lower).strip().lower()
    vmec2000_solver = solver_name in ("vmec2000_iter", "vmec2000_scan", "vmec2000_iter_fast")
    ns_list_input = stage_array_list_func(indata.get("NS_ARRAY", None))
    niter_list_input = stage_array_list_func(indata.get("NITER_ARRAY", None))
    ftol_list_input = stage_array_list_func(indata.get("FTOL_ARRAY", None))
    multigrid_user_provided = multigrid is not None

    cli_budgeted_multigrid_requested = (
        bool(cli_fixed_boundary_mode)
        and bool(accelerated_mode)
        and bool(vmec2000_solver)
        and (not bool(cfg.lfreeb))
        and (not bool(restart_state_present))
        and (not bool(restart_solver_state_present))
        and (multigrid is None)
        and (ns_list_input is not None)
        and (len(ns_list_input) > 1)
        and (niter_list_input is None)
    )
    user_explicitly_staged_cli = (
        bool(cli_fixed_boundary_mode)
        and bool(accelerated_mode)
        and bool(vmec2000_solver)
        and (not bool(cfg.lfreeb))
        and (not bool(restart_state_present))
        and (not bool(restart_solver_state_present))
        and (multigrid is None)
        and (ns_list_input is not None)
        and (len(ns_list_input) > 1)
        and (niter_list_input is not None)
        and (len(niter_list_input) == len(ns_list_input))
    )
    cli_fixed_boundary_finish_enabled = (
        bool(cli_fixed_boundary_mode)
        and (solver_name == "vmec2000_iter")
        and (not bool(cfg.lfreeb))
    )
    current_driven_3d_cli = (
        bool(cli_fixed_boundary_mode)
        and bool(accelerated_mode)
        and (not bool(cfg.lfreeb))
        and bool(cfg.lthreed)
        and (ns_list_input is not None)
        and (len(ns_list_input) > 1)
        and (niter_list_input is not None)
        and (len(niter_list_input) == len(ns_list_input))
        and (int(indata.get_int("NCURR", 0)) != 0)
    )
    direct_staged_current_driven_3d_cli = bool(current_driven_3d_cli)
    deferred_staged_current_driven_3d_cli = bool(current_driven_3d_cli) and (
        not bool(direct_staged_current_driven_3d_cli)
    )

    multigrid_eff = multigrid
    accelerated_single_grid_default = False
    if multigrid_eff is None:
        multigrid_eff = solver_name == "vmec2000_iter"
        if bool(cli_budgeted_multigrid_requested):
            multigrid_eff = True
        elif bool(direct_staged_current_driven_3d_cli):
            multigrid_eff = True
        elif bool(user_explicitly_staged_cli):
            multigrid_eff = True
        elif bool(accelerated_mode) and (not bool(cfg.lfreeb)):
            multigrid_eff = False
            accelerated_single_grid_default = True

    max_iter_eff = max_iter
    if max_iter_eff is max_iter_sentinel:
        if bool(vmec2000_solver):
            if niter_list_input:
                max_iter_eff = int(sum(int(v) for v in niter_list_input))
            else:
                max_iter_eff = int(indata.get_int("NITER", 10))
        else:
            max_iter_eff = 10
    max_iter_eff = int(max_iter_eff)

    if bool(restart_state_present) or bool(restart_solver_state_present):
        multigrid_eff = False
    multigrid_eff = bool(multigrid_eff) and (ns_override is None)

    if stage_transition_heuristic is None:
        env_stage = str(getenv("VMEC_JAX_STAGE_HEURISTIC", "")).strip().lower()
        if env_stage in ("1", "true", "yes"):
            stage_transition_heuristic_eff = True
        elif env_stage in ("0", "false", "no"):
            stage_transition_heuristic_eff = False
        else:
            stage_transition_heuristic_eff = False
    else:
        stage_transition_heuristic_eff = bool(stage_transition_heuristic)

    ns_stages = [int(cfg.ns)]
    if bool(multigrid_eff) and ns_list_input:
        ns_stages = [int(v) for v in ns_list_input]

    if niter_list_input:
        niter_sum = int(sum(int(v) for v in niter_list_input))
        niter_default = int(indata.get_int("NITER", max_iter_eff))
        if (not bool(max_iter_overridden)) and int(max_iter_eff) == niter_default:
            max_iter_eff = niter_sum

    return FixedBoundaryStagePolicy(
        ns_list_input=ns_list_input,
        niter_list_input=niter_list_input,
        ftol_list_input=ftol_list_input,
        cli_budgeted_multigrid_requested=bool(cli_budgeted_multigrid_requested),
        user_explicitly_staged_cli=bool(user_explicitly_staged_cli),
        cli_fixed_boundary_finish_enabled=bool(cli_fixed_boundary_finish_enabled),
        multigrid=bool(multigrid_eff),
        multigrid_user_provided=bool(multigrid_user_provided),
        accelerated_single_grid_default=bool(accelerated_single_grid_default),
        current_driven_3d_cli=bool(current_driven_3d_cli),
        direct_staged_current_driven_3d_cli=bool(direct_staged_current_driven_3d_cli),
        deferred_staged_current_driven_3d_cli=bool(deferred_staged_current_driven_3d_cli),
        max_iter=int(max_iter_eff),
        stage_transition_heuristic=bool(stage_transition_heuristic_eff),
        ns_stages=ns_stages,
    )


def resolve_driver_step_size(*, step_size, step_size_sentinel, solver_lower: str, indata) -> float:
    """Resolve the public driver step-size default for the selected solver."""

    if step_size is not step_size_sentinel and step_size is not None:
        return float(step_size)
    solver_name = str(solver_lower).strip().lower()
    if solver_name in ("vmec2000_iter", "vmec2000_scan", "vmec2000_iter_fast"):
        return float(indata.get_float("DELT", 5e-3))
    return 5e-3


def requested_final_ftol(*, indata, ftol_list_input) -> float:
    """Return the final requested residual tolerance for a staged run."""

    ftol_list = as_float_list(ftol_list_input)
    if ftol_list:
        return max(0.0, float(ftol_list[-1]))
    return max(0.0, float(indata.get_float("FTOL", 1.0e-13)))


def as_float_list(value) -> list[float] | None:
    """Best-effort conversion of a scalar/list-like value to floats."""

    if value is None:
        return None
    try:
        return [float(v) for v in value]
    except Exception:
        return None


def as_list_like(value):
    """Best-effort conversion of VMEC namelist values to a Python list."""

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
    if isinstance(value, str):
        return None
    try:
        return list(value)
    except Exception:
        return None


def profile_guided_scan_decision_for_indata(indata, *, getenv=os.getenv) -> bool | None:
    """Return an opt-in scan decision from measured benchmark provenance.

    This is the no-probe dynamic selector hook for accelerated fixed-boundary
    solves.  It does not inspect pressure, current, spectral size, or radial
    grid size.  By default it consults the small bundled benchmark profile.
    Deployments can set ``VMEC_JAX_ACCELERATED_SCAN_PROFILE`` to another
    benchmark/provenance JSON with per-case ``recommended_use_scan`` or
    ``prefer_use_scan`` values, or set it to ``off`` to disable profile-guided
    selection.  That keeps cold-short-row policy decisions measured and
    reproducible without spending more time probing than solving.
    """

    decision = profile_guided_scan_decision_with_detail_for_indata(indata, getenv=getenv)
    return None if decision is None else bool(decision.use_scan)


def profile_guided_scan_decision_with_detail_for_indata(
    indata,
    *,
    getenv=os.getenv,
) -> ScanDefaultDecision | None:
    """Return a measured scan decision with provenance, or ``None``.

    The selector is intentionally profile-based instead of physics- or
    size-threshold based: benchmark provenance says whether a known cold row
    should use the fused scan runner, while unknown rows keep the backend
    default.  This makes the startup policy easy to audit in result
    diagnostics without paying for a runtime probe.
    """

    profile_path = str(getenv("VMEC_JAX_ACCELERATED_SCAN_PROFILE", "")).strip()
    if profile_path.lower() in ("0", "false", "no", "off", "none"):
        return None
    if profile_path:
        profile = Path(profile_path).expanduser()
        profile_label = str(profile)
    else:
        profile = _BUNDLED_ACCELERATED_SCAN_PROFILE
        profile_label = "bundled"
        if not profile.exists():
            return None
    candidate_ids = _profile_candidate_case_ids(getattr(indata, "source_path", None))
    if not candidate_ids:
        return None
    try:
        payload = json.loads(profile.read_text(encoding="utf-8"))
    except Exception:
        return None
    for record in _iter_profile_records(payload):
        case_id = str(record.get("case_id", record.get("id", ""))).strip()
        if case_id not in candidate_ids:
            continue
        backend = str(record.get("backend", "vmec_jax")).strip().lower()
        if backend not in ("", "vmec_jax", "cpu"):
            continue
        decision = _profile_record_scan_decision(record)
        if decision is not None:
            detail = str(record.get("classification", "")).strip()
            suffix = f":{detail}" if detail else ""
            return ScanDefaultDecision(
                use_scan=bool(decision),
                source="profile",
                detail=f"{profile_label}:{case_id}{suffix}",
            )
    return None


def _profile_candidate_case_ids(source_path) -> set[str]:
    """Return benchmark-style case ids for one VMEC input path."""

    if source_path is None:
        return set()
    path = Path(str(source_path))
    names = {path.name, path.stem}
    if path.name.startswith("input."):
        names.add(path.name[len("input."):])
    if path.stem.startswith("input."):
        names.add(path.stem[len("input."):])
    return {name for name in names if name}


def _iter_profile_records(payload):
    """Yield records from benchmark summaries or classification sidecars."""

    if isinstance(payload, dict):
        for key in ("records", "results", "overrides"):
            records = payload.get(key)
            if isinstance(records, list):
                yield from (record for record in records if isinstance(record, dict))
                return
            if isinstance(records, dict):
                for case_id, record in records.items():
                    if not isinstance(record, dict):
                        continue
                    item = dict(record)
                    item.setdefault("case_id", str(case_id).split("|", 1)[0])
                    yield item
                return
    elif isinstance(payload, list):
        yield from (record for record in payload if isinstance(record, dict))


def _profile_record_scan_decision(record: dict) -> bool | None:
    """Extract an explicit scan recommendation from one profile record."""

    for key in ("recommended_use_scan", "prefer_use_scan", "use_scan"):
        if key in record:
            return bool(record[key])
    classification = str(record.get("classification", "")).lower()
    if "cold_scan_compile_amortization" in classification and bool(record.get("cold_latency_prefer_noscan", False)):
        return False
    return None


def default_non_autodiff_solver_policy_for_backend(indata, backend: str) -> tuple[str, bool]:
    """Choose the fast-first default without classifying input physics or size."""

    if bool(indata.get_bool("LFREEB", False)):
        return "default", True
    _ = backend
    return "accelerated", True


def default_use_scan_for_backend(indata, backend: str, solver_mode: str | None) -> bool:
    """Choose the fused scan loop when the selected backend can execute it."""

    return bool(default_scan_decision_for_backend(indata, backend, solver_mode).use_scan)


def default_scan_decision_for_backend(indata, backend: str, solver_mode: str | None) -> ScanDefaultDecision:
    """Choose scan/non-scan and report why the default was selected."""

    mode = normalize_solver_mode(solver_mode=solver_mode, performance_mode=True)
    backend_l = str(backend).strip().lower()
    if mode == "parity" and str(backend).strip().lower() == "cpu":
        return ScanDefaultDecision(False, "solver_mode", "cpu_parity_uses_loop")
    _ = indata
    if backend_l == "cpu":
        profile_decision = profile_guided_scan_decision_with_detail_for_indata(indata)
        if profile_decision is not None:
            return profile_decision
    use_scan = backend_l in ("cpu", "gpu", "cuda", "rocm", "tpu")
    detail = f"{backend_l or 'unknown'}_{'supports_scan' if use_scan else 'uses_loop'}"
    return ScanDefaultDecision(bool(use_scan), "backend_default", detail)


def resolve_jit_forces_auto_policy(flag: bool | str, static_i, niter_i: int) -> bool:
    """Resolve ``jit_forces='auto'`` for one stage from workload size."""

    if isinstance(flag, str):
        if flag.strip().lower() != "auto":
            return True
        try:
            nmodes_i = int(np.asarray(static_i.modes.m).size)
            nrzt = int(static_i.cfg.ns) * int(static_i.cfg.ntheta) * int(static_i.cfg.nzeta)
            work = nmodes_i * nrzt
        except Exception:
            return True
        # Heuristic: avoid JIT for very small workloads unless the stage will run
        # long enough to amortize compilation cost.
        if int(niter_i) >= 5:
            return True
        return bool(work >= 2_000_000)
    return bool(flag)


def _truthy_env(value: str | None) -> bool:
    return str(value or "").strip().lower() not in ("", "0", "false", "no")


def _optional_bool_env(value: str | None) -> bool | None:
    if value is None:
        return None
    return _truthy_env(value)


def _warmup_iters_from_env(value: str | None, *, precompile_enabled: bool) -> int:
    if value is not None:
        try:
            return max(0, int(value))
        except Exception:
            return 2
    return 0 if bool(precompile_enabled) else 2


def resolve_stage_jit_settings(
    *,
    jit_forces_base: bool,
    scan_mode: bool,
    solver: str,
    performance_mode: bool,
    jit_precompile: bool | None,
    getenv=os.getenv,
) -> StageJitSettings:
    """Resolve per-stage force-kernel JIT and warmup policy.

    This helper preserves the driver defaults:
    scan keeps JIT in fast/performance mode, parity scan disables JIT unless
    explicitly overridden, and non-scan stages optionally precompile before the
    actual iteration loop.
    """

    jit_forces_eff = bool(jit_forces_base)
    if bool(scan_mode) and str(solver).strip().lower() == "vmec2000_iter":
        scan_jit = _optional_bool_env(getenv("VMEC_JAX_SCAN_JIT_FORCES", None))
        if scan_jit is None:
            if not bool(performance_mode):
                jit_forces_eff = False
        else:
            jit_forces_eff = bool(scan_jit)

    jit_precompile_eff = False
    if bool(jit_forces_eff) and (not bool(scan_mode)):
        if jit_precompile is None:
            jit_precompile_eff = _truthy_env(getenv("VMEC_JAX_JIT_PRECOMPILE", "1"))
        else:
            jit_precompile_eff = bool(jit_precompile)

    jit_warmup_iters = 0
    if bool(jit_forces_eff) and (not bool(scan_mode)):
        jit_warmup_iters = _warmup_iters_from_env(
            getenv("VMEC_JAX_JIT_WARMUP_ITERS", None),
            precompile_enabled=bool(jit_precompile_eff),
        )

    jit_precompile_noscan = False
    if bool(jit_forces_base):
        if jit_precompile is None:
            jit_precompile_noscan = _truthy_env(getenv("VMEC_JAX_JIT_PRECOMPILE", "1"))
        else:
            jit_precompile_noscan = bool(jit_precompile)

    jit_warmup_noscan = 0
    if bool(jit_forces_base):
        jit_warmup_noscan = _warmup_iters_from_env(
            getenv("VMEC_JAX_JIT_WARMUP_ITERS", None),
            precompile_enabled=bool(jit_precompile_noscan),
        )

    return StageJitSettings(
        jit_forces_eff=bool(jit_forces_eff),
        jit_precompile_eff=bool(jit_precompile_eff),
        jit_warmup_iters=int(jit_warmup_iters),
        jit_precompile_noscan=bool(jit_precompile_noscan),
        jit_warmup_noscan=int(jit_warmup_noscan),
    )


def dynamic_scan_probe_settings(
    niter_i: int,
    *,
    backend_name_func,
    getenv=os.getenv,
) -> tuple[int, bool, str]:
    """Resolve dynamic scan-probe budget and timing mode for one stage.

    Timed probes run four short prefixes (warm loop, warm scan, timed loop,
    timed scan), while parity-only probes run two.  The default cap keeps the
    total prefix-iteration budget no larger than the production stage budget so
    dynamic selection cannot cost more iterations than the solve it is trying
    to accelerate.  ``VMEC_JAX_DYNAMIC_SCAN_MAX_SOLVE_FRACTION`` can relax this
    for explicit diagnostics without adding physics- or input-size thresholds.
    """

    backend = str(backend_name_func()).strip().lower() or "cpu"
    timed_env = str(getenv("VMEC_JAX_DYNAMIC_SCAN_TIMED", "")).strip().lower()
    if timed_env in ("1", "true", "yes", "on"):
        timed_probe = True
    elif timed_env in ("0", "false", "no", "off"):
        timed_probe = False
    else:
        timed_probe = backend == "cpu"

    default_probe_iters = 10 if timed_probe else 3
    try:
        pre_iters_env = str(getenv("VMEC_JAX_DYNAMIC_SCAN_ITERS", "")).strip()
        pre_iters = max(1, int(pre_iters_env)) if pre_iters_env else default_probe_iters
    except Exception:
        pre_iters = default_probe_iters

    try:
        max_fraction = max(0.0, float(str(getenv("VMEC_JAX_DYNAMIC_SCAN_MAX_SOLVE_FRACTION", "1.0")).strip()))
    except Exception:
        max_fraction = 1.0
    probe_runs = 4 if bool(timed_probe) else 2
    max_probe_iters = max(1, int((max(1, int(niter_i)) * max_fraction) // probe_runs))
    pre_iters = min(int(pre_iters), int(max_probe_iters))
    if pre_iters >= int(niter_i):
        pre_iters = max(1, int(niter_i) - 1)
    return pre_iters, timed_probe, backend


def default_preconditioner_use_precomputed_tridi(
    *,
    cfg,
    backend: str,
    performance_mode: bool,
    use_scan: bool,
    direct_external_provider: bool = False,
) -> bool | None:
    """Choose the R/Z preconditioner tridiagonal-solver policy for public runs."""

    if os.getenv("VMEC_JAX_TRIDI_PRECOMPUTE") is not None:
        return None
    backend_l = str(backend).strip().lower()
    if not bool(performance_mode):
        return None
    if bool(use_scan):
        return True if backend_l in ("gpu", "cuda", "rocm", "tpu") else None
    if backend_l == "cpu" and bool(getattr(cfg, "lfreeb", False)) and bool(direct_external_provider):
        return True
    if backend_l not in ("gpu", "cuda", "rocm", "tpu"):
        return None
    if bool(getattr(cfg, "lasym", False)):
        return True
    try:
        mpol = int(getattr(cfg, "mpol", 0))
        ntor = int(getattr(cfg, "ntor", 0))
        signed_mode_count = mpol * (2 * ntor + 1) - ntor if ntor > 0 else mpol
    except Exception:
        signed_mode_count = 0
    return True if signed_mode_count >= 32 else None


def default_preconditioner_use_lax_tridi(
    *,
    cfg,
    backend: str,
    performance_mode: bool,
    use_scan: bool,
    direct_external_provider: bool = False,
) -> bool | None:
    """Choose the CPU R/Z preconditioner tridiagonal-solve primitive."""

    if os.getenv("VMEC_JAX_TRIDI_SOLVE") is not None:
        return None
    del backend, cfg, direct_external_provider, use_scan, performance_mode
    return None


def result_final_residuals(result) -> tuple[float, float, float] | None:
    """Extract final ``FSQR/FSQZ/FSQL`` residuals from a solver result."""

    if result is None:
        return None
    diag = getattr(result, "diagnostics", {}) or {}
    explicit = (
        diag.get("final_fsqr", None),
        diag.get("final_fsqz", None),
        diag.get("final_fsql", None),
    )
    if all(val is not None for val in explicit):
        try:
            return tuple(float(val) for val in explicit)
        except Exception:
            pass
    try:
        fsqr = np.asarray(getattr(result, "fsqr2_history"))
        fsqz = np.asarray(getattr(result, "fsqz2_history"))
        fsql = np.asarray(getattr(result, "fsql2_history"))
        if fsqr.size > 0 and fsqz.size > 0 and fsql.size > 0:
            return (
                float(fsqr.reshape(-1)[-1]),
                float(fsqz.reshape(-1)[-1]),
                float(fsql.reshape(-1)[-1]),
            )
    except Exception:
        pass
    try:
        fsqr = np.asarray(diag.get("fsqr_full", []))
        fsqz = np.asarray(diag.get("fsqz_full", []))
        fsql = np.asarray(diag.get("fsql_full", []))
        if fsqr.size > 0 and fsqz.size > 0 and fsql.size > 0:
            return (
                float(fsqr.reshape(-1)[-1]),
                float(fsqz.reshape(-1)[-1]),
                float(fsql.reshape(-1)[-1]),
            )
    except Exception:
        pass
    return None


def result_final_fsq(result) -> float:
    """Return the final total residual used by staged-driver decisions."""

    if result is None:
        return float("inf")
    try:
        w_hist = np.asarray(getattr(result, "w_history"))
        if w_hist.size > 0:
            return float(w_hist.reshape(-1)[-1])
    except Exception:
        pass
    residuals = result_final_residuals(result)
    if residuals is not None:
        return float(sum(residuals))
    return float("inf")


def result_meets_requested_ftol(result, *, ftol: float) -> bool:
    """Return whether a solver result satisfies the requested tolerance."""

    if result is None:
        return False
    diag = getattr(result, "diagnostics", {}) or {}
    strict = diag.get("converged_strict", None)
    if strict is not None:
        return bool(strict)
    if ("ftol" not in diag) and ("requested_ftol" not in diag):
        return bool(diag.get("converged", False))
    residuals = result_final_residuals(result)
    if residuals is None:
        return False
    target = max(0.0, float(ftol))
    return all(float(val) <= target for val in residuals)


def result_hits_total_target(result, *, fsq_total_target: float | None) -> bool:
    """Return whether a solver result reached an optional total-FSQ target."""

    if result is None or fsq_total_target is None:
        return False
    return bool(result_final_fsq(result) <= max(0.0, float(fsq_total_target)))


def allocate_integer_budget(*, total: int, weights: list[int]) -> list[int]:
    """Allocate an integer iteration budget in proportion to integer weights."""

    total = max(0, int(total))
    if total <= 0:
        return [0] * len(weights)
    if not weights:
        return []
    weights_eff = [max(0, int(w)) for w in weights]
    if sum(weights_eff) <= 0:
        out = [0] * len(weights_eff)
        out[-1] = total
        return out
    raw = [float(total) * float(w) / float(sum(weights_eff)) for w in weights_eff]
    out = [int(v) for v in raw]
    remaining = int(total - sum(out))
    if remaining > 0:
        order = sorted(range(len(raw)), key=lambda idx: (raw[idx] - float(out[idx])), reverse=True)
        for idx in order[:remaining]:
            out[idx] += 1
    elif remaining < 0:  # pragma: no cover - floors of nonnegative shares cannot overspend.
        order = sorted(range(len(raw)), key=lambda idx: (raw[idx] - float(out[idx])))
        for idx in order[: abs(remaining)]:
            if out[idx] > 0:
                out[idx] -= 1
    return out


def accelerated_cli_budgeted_total_iters(*, total_budget: int, ns_stages: list[int]) -> int:
    """Reduce oversized parity-era budgets for CLI accelerated warm starts."""

    total_budget = max(1, int(total_budget))
    if not ns_stages:
        return total_budget
    ns0 = max(1, int(ns_stages[0]))
    nsf = max(ns0, int(ns_stages[-1]))
    return max(1, int(round(float(total_budget) * float(np.sqrt(float(ns0) / float(nsf))))))


def accelerated_cli_budgeted_stage_iters(*, total_budget: int, ns_stages: list[int]) -> list[int]:
    """Distribute a reduced CLI accelerated budget across multigrid stages."""

    if not ns_stages:
        return [max(1, int(total_budget))]
    ns_int = [max(1, int(v)) for v in ns_stages]
    increments = [ns_int[0]]
    for prev_ns, curr_ns in zip(ns_int[:-1], ns_int[1:]):
        increments.append(max(1, int(curr_ns - prev_ns)))
    weights = [int(v) * int(v) for v in increments]
    out = allocate_integer_budget(total=max(1, int(total_budget)), weights=weights)
    if out:
        out[-1] = max(1, int(out[-1]))
    return out


def distribute_stage_iters(*, iters: int, nstep: int) -> list[int]:
    """Split an iteration budget across equal-sized stage chunks."""

    iters = int(iters)
    nstep = int(nstep)
    if iters <= 0:
        return [0]
    if nstep <= 1:
        return [iters]
    base, rem = divmod(iters, nstep)
    if base == 0:
        return [iters]
    return [base + (1 if i < rem else 0) for i in range(nstep)]


def sanitize_resume_state_for_grid_change(resume_state, *, step_size: float):
    """Keep only resume-state fields that remain valid after a grid change."""

    if resume_state is None:
        return None
    time_step = resume_state.get("time_step", None)
    if time_step is None:
        return None
    try:
        time_step = min(float(time_step), float(step_size))
    except Exception:
        time_step = float(time_step)
    inv_tau = [0.15 / float(time_step)] * 10
    out = {
        "time_step": float(time_step),
        "inv_tau": list(inv_tau),
    }
    if "flip_sign" in resume_state:
        out["flip_sign"] = float(resume_state["flip_sign"])
    out["iter_offset"] = 0
    out["vmec2000_cache_valid"] = False
    return out


def sanitize_resume_state_for_same_grid(resume_state, *, step_size: float):
    """Normalize resume-state fields before continuing on the same grid."""

    if resume_state is None:
        return None
    time_step = resume_state.get("time_step", None)
    if time_step is None:
        return None
    try:
        time_step = min(float(time_step), float(step_size))
    except Exception:
        time_step = float(time_step)
    out = {
        "time_step": float(time_step),
        "inv_tau": list(resume_state.get("inv_tau", [0.15 / max(float(time_step), 1.0e-12)] * 10)),
        "iter_offset": int(resume_state.get("iter_offset", 0)),
        "vmec2000_cache_valid": False,
    }
    if "flip_sign" in resume_state:
        out["flip_sign"] = float(resume_state["flip_sign"])
    return out


def sanitize_minimal_resume_state_for_finish(resume_state):
    """Reduce a resume payload to the fields needed by final finish passes."""

    if not isinstance(resume_state, dict):
        return resume_state
    time_step = resume_state.get("time_step", None)
    if time_step is None:
        return resume_state
    try:
        time_step_f = float(time_step)
    except Exception:
        return resume_state
    out = {
        "time_step": float(time_step_f),
        "inv_tau": list(resume_state.get("inv_tau", [0.15 / max(abs(time_step_f), 1.0e-16)] * 10)),
        "iter_offset": int(resume_state.get("iter_offset", 0)),
        "vmec2000_cache_valid": bool(resume_state.get("vmec2000_cache_valid", False)),
    }
    if "flip_sign" in resume_state:
        try:
            out["flip_sign"] = float(resume_state["flip_sign"])
        except Exception:
            pass
    return out


def resolve_vmec2000_stage_controls(
    *,
    nstep: int,
    niter_list,
    ftol_list,
    max_iter: int,
    max_iter_overridden: bool,
    multigrid_use_input_niter: bool,
    multigrid_user_provided: bool,
    accelerated_single_grid_default: bool,
    indata,
) -> tuple[list[int], list[float], list[int] | None, list[float] | None]:
    """Resolve VMEC2000-style per-stage iteration and tolerance budgets."""

    nstep = int(nstep)
    niter_stages_input = [int(v) for v in niter_list] if niter_list and len(niter_list) == nstep else None
    ftol_stages_input = [float(v) for v in ftol_list] if ftol_list and len(ftol_list) == nstep else None
    accelerated_single_grid_budget = (
        bool(accelerated_single_grid_default)
        and (not bool(multigrid_user_provided))
        and int(nstep) == 1
        and bool(niter_list)
        and (not bool(max_iter_overridden))
    )
    if multigrid_use_input_niter:
        niter_stages = niter_stages_input
        ftol_stages = ftol_stages_input
        if niter_stages is None:
            if accelerated_single_grid_budget:
                niter_stages = [int(max_iter)]
            elif max_iter_overridden:
                niter_stages = distribute_stage_iters(iters=int(max_iter), nstep=int(nstep))
            else:
                niter_stage = int(indata.get_int("NITER", int(max_iter)))
                niter_stages = [niter_stage] * nstep
        else:
            budget = int(max_iter)
            remaining = max(0, budget)
            out = [0] * nstep
            for i in range(nstep):
                if remaining <= 0:
                    break
                cap = max(0, int(niter_stages[i]))
                take = min(cap, remaining)
                out[i] = take
                remaining -= take
            niter_stages = out
        if ftol_stages is None:
            if bool(accelerated_single_grid_default) and int(nstep) == 1 and bool(ftol_list):
                ftol_stages = [float(ftol_list[-1])]
            else:
                ftol_stages = [float(indata.get_float("FTOL", 1e-13))] * nstep
    else:
        niter_stages = distribute_stage_iters(iters=int(max_iter), nstep=int(nstep))
        if ftol_stages_input is not None:
            ftol_stages = ftol_stages_input
        else:
            ftol_stages = [float(indata.get_float("FTOL", 1e-13))] * nstep
    return niter_stages, ftol_stages, niter_stages_input, ftol_stages_input
