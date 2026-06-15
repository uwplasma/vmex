"""Backend and solver-policy helpers for :mod:`vmec_jax.driver`.

The public driver is intentionally kept thin enough to read as a workflow:
parse inputs, build static data, choose solver policy, run stages, and emit
results.  This module holds the small policy and budget helpers used by that
workflow so they can be unit-tested without importing the whole driver stack.
"""

from __future__ import annotations

import os

import numpy as np


VALID_SOLVER_MODES = frozenset(("default", "parity", "accelerated"))
FSQ_COMPONENT_NAMES = ("fsqr", "fsqz", "fsql")


def host_update_assembly_driver_default(
    *,
    cfg,
    performance_mode: bool,
    backend: str,
    use_scan: bool,
) -> bool:
    """Resolve the public driver default for CPU host-update assembly."""

    backend_name = str(backend).strip().lower()
    # Host NumPy update assembly is fastest for low-mode CPU solves because it
    # avoids per-step JAX dispatch. On larger spectral/radial grids the repeated
    # host state assembly dominates; let solve.py's fused strict-update JIT take
    # those cases instead.
    nrange = int(getattr(cfg, "ntor", 0)) + 1
    if bool(getattr(cfg, "lasym", False)):
        nrange = 2 * int(getattr(cfg, "ntor", 0)) + 1
    update_work = int(getattr(cfg, "ns", 0)) * int(getattr(cfg, "mpol", 0)) * int(nrange)
    try:
        work_limit = int(os.getenv("VMEC_JAX_HOST_UPDATE_CPU_WORK_LIMIT", "1000"))
    except Exception:
        work_limit = 1000
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
    if solver_mode is None:
        return "default" if bool(performance_mode) else "parity"
    mode = str(solver_mode).strip().lower()
    aliases = {
        "fast": "default",
        "safe": "parity",
        "reference": "parity",
        "perf": "accelerated",
    }
    mode = aliases.get(mode, mode)
    if mode not in VALID_SOLVER_MODES:
        valid = ", ".join(sorted(VALID_SOLVER_MODES))
        raise ValueError(f"Unknown solver_mode {solver_mode!r}. Expected one of: {valid}.")
    return mode


def accelerated_fsq_total_target_from_ftol(ftol: float) -> float:
    """Collapse per-component FTOL into an equivalent total-residual target."""

    return max(0.0, float(ftol)) * float(len(FSQ_COMPONENT_NAMES))


def requested_final_ftol(*, indata, ftol_list_input) -> float:
    ftol_list = as_float_list(ftol_list_input)
    if ftol_list:
        return max(0.0, float(ftol_list[-1]))
    return max(0.0, float(indata.get_float("FTOL", 1.0e-13)))


def as_float_list(value) -> list[float] | None:
    if value is None:
        return None
    try:
        return [float(v) for v in value]
    except Exception:
        return None


def as_list_like(value):
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
    try:
        return list(value)
    except Exception:
        return None


def default_non_autodiff_solver_policy_for_backend(indata, backend: str) -> tuple[str, bool]:
    if bool(indata.get_bool("LFREEB", False)):
        return "default", True
    ns_array = as_list_like(indata.get("NS_ARRAY", None))
    niter_array = as_list_like(indata.get("NITER_ARRAY", None))
    if (ns_array is not None) and (len(ns_array) > 1) and (niter_array is None):
        return "parity", False

    if str(backend).strip().lower() == "cpu":
        # LASYM and current-driven multigrid inputs still use the accelerated
        # solver mode for stricter finish behavior, but the backend-aware scan
        # selector keeps auto-selected CPU solves on the VMEC-control loop.
        if bool(indata.get_bool("LASYM", False)):
            return "accelerated", True
        ncurr = int(indata.get_int("NCURR", 0))
        if ncurr == 1 and (ns_array is not None) and (len(ns_array) > 1):
            return "accelerated", True
        return "default", True
    return "accelerated", True


def default_use_scan_for_backend(indata, backend: str, solver_mode: str | None) -> bool:
    """Choose the public fixed-boundary iteration loop for ordinary runs."""

    _ = (indata, normalize_solver_mode(solver_mode=solver_mode, performance_mode=True))
    backend_l = str(backend).strip().lower()
    if backend_l in ("gpu", "cuda", "rocm"):
        return True
    if backend_l != "cpu":
        return False

    def _get_int(name: str, default: int) -> int:
        try:
            return int(indata.get_int(name, default))
        except Exception:
            try:
                return int(indata.get(name, default))
            except Exception:
                return int(default)

    ns_values = as_list_like(getattr(indata, "get", lambda *_args: None)("NS_ARRAY", None))
    niter_values = as_list_like(getattr(indata, "get", lambda *_args: None)("NITER_ARRAY", None))
    ns_max = max([_get_int("NS", 0), *[int(v) for v in (ns_values or []) if v is not None]], default=0)
    niter_max = max(
        [_get_int("NITER", 0), *[int(v) for v in (niter_values or []) if v is not None]],
        default=0,
    )
    mpol = max(1, _get_int("MPOL", 1))
    ntor = max(0, _get_int("NTOR", 0))
    signed_mode_count = mpol * (2 * ntor + 1) - ntor if ntor > 0 else mpol
    work = int(ns_max) * int(niter_max) * int(max(1, signed_mode_count))
    return bool(work >= 2_000_000)


def resolve_jit_forces_auto_policy(flag: bool | str, static_i, niter_i: int) -> bool:
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


def dynamic_scan_probe_settings(
    niter_i: int,
    *,
    backend_name_func,
    getenv=os.getenv,
) -> tuple[int, bool, str]:
    """Resolve dynamic scan-probe budget and timing mode for one stage."""

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
    if result is None or fsq_total_target is None:
        return False
    return bool(result_final_fsq(result) <= max(0.0, float(fsq_total_target)))


def allocate_integer_budget(*, total: int, weights: list[int]) -> list[int]:
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
