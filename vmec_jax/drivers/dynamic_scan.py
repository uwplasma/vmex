"""Dynamic scan selection for fixed-boundary VMEC2000-style stages."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable
import os
import time

import numpy as np


def maybe_select_dynamic_scan_mode(
    *,
    cfg: Any,
    accelerated_mode: bool,
    performance_mode: bool,
    scan_mode: bool,
    vmec2000_control: bool,
    niter: int,
    solve_kwargs: dict[str, Any],
    state_stage_start: Any,
    static_stage: Any,
    resume_state_stage: Any,
    jit_forces_base: bool,
    solve_fixed_boundary_residual_iter: Callable[..., Any],
    dynamic_scan_probe_settings: Callable[[int], tuple[int, bool, str]],
    vmec_histories_match: Callable[..., bool],
    vmec_history_relerr: Callable[[Any, Any], float],
    verbose: bool,
    print_func: Callable[..., None] = print,
    getenv: Callable[[str, str], str] = os.getenv,
    deepcopy_func: Callable[[Any], Any] = deepcopy,
) -> bool:
    """Return the selected scan mode after an optional dynamic scan probe.

    The probe is a runtime policy gate, not solver mathematics.  It compares a
    short warmed scan/non-scan prefix and keeps scan only when histories match
    and, for timed probes, scan is faster.  All solver arguments are passed in
    from the driver so this helper stays a pure orchestration seam.
    """

    dynamic_scan_default = "1" if bool(cfg.lasym) else "0"
    dynamic_scan_env = getenv("VMEC_JAX_DYNAMIC_SCAN", dynamic_scan_default).strip().lower()
    dynamic_scan = dynamic_scan_env not in ("", "0", "false", "no")
    if not (
        (not bool(accelerated_mode))
        and bool(dynamic_scan)
        and bool(performance_mode)
        and bool(scan_mode)
        and bool(vmec2000_control)
        and int(niter) > 1
    ):
        return bool(scan_mode)

    pre_iters, timed_probe, probe_backend = dynamic_scan_probe_settings(int(niter))
    if int(pre_iters) <= 0:
        return bool(scan_mode)

    fsq_tol_env = getenv("VMEC_JAX_DYNAMIC_SCAN_FSQ_RTOL", "1e-6").strip()
    try:
        fsq_tol = float(fsq_tol_env)
    except Exception:
        fsq_tol = 1.0e-6
    hist_atol_env = getenv("VMEC_JAX_DYNAMIC_SCAN_ATOL", "1e-12").strip()
    try:
        hist_atol = float(hist_atol_env)
    except Exception:
        hist_atol = 1.0e-12

    pre_kwargs = dict(solve_kwargs)
    pre_kwargs.update(
        {
            "max_iter": int(pre_iters),
            "verbose": False,
            "verbose_vmec2000_table": False,
            "jit_warmup_iters": 0,
            "jit_precompile": False,
            # Keep full histories in the probe so we compare traces, not just
            # one terminal residual scalar.
            "scan_minimal_default": False,
        }
    )

    def _run_pref(*, use_scan_flag: bool):
        kwargs = dict(pre_kwargs)
        kwargs["use_scan"] = bool(use_scan_flag)
        kwargs["resume_state"] = deepcopy_func(resume_state_stage)
        state_probe = deepcopy_func(state_stage_start)
        if not bool(jit_forces_base):
            try:
                import jax

                with jax.disable_jit():
                    return solve_fixed_boundary_residual_iter(
                        state_probe,
                        static_stage,
                        jit_forces=False,
                        **kwargs,
                    )
            except Exception:
                return solve_fixed_boundary_residual_iter(
                    state_probe,
                    static_stage,
                    jit_forces=False,
                    **kwargs,
                )
        return solve_fixed_boundary_residual_iter(
            state_probe,
            static_stage,
            jit_forces=True,
            **kwargs,
        )

    if bool(timed_probe):
        # Warm both variants before timing so the selector compares steady-state
        # iteration cost rather than one-off compile cost.
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

    fsq_ok = vmec_histories_match(
        res_pref_scan,
        res_pref_noscan,
        rtol=float(fsq_tol),
        atol=float(hist_atol),
    )
    choose_scan = bool(fsq_ok) and ((not bool(timed_probe)) or (t_scan < t_noscan))
    selected_scan = bool(choose_scan)
    if bool(verbose):
        if not bool(fsq_ok):
            print_func(
                "[vmec_jax] dynamic scan probe mismatch: "
                f"w={vmec_history_relerr(res_pref_scan.w_history, res_pref_noscan.w_history):.3e} "
                f"fsqr={vmec_history_relerr(res_pref_scan.fsqr2_history, res_pref_noscan.fsqr2_history):.3e} "
                f"fsqz={vmec_history_relerr(res_pref_scan.fsqz2_history, res_pref_noscan.fsqz2_history):.3e} "
                f"fsql={vmec_history_relerr(res_pref_scan.fsql2_history, res_pref_noscan.fsql2_history):.3e}",
                flush=True,
            )
        if bool(timed_probe):
            print_func(
                "[vmec_jax] dynamic scan selection: "
                f"backend={probe_backend} scan={t_scan:.3f}s noscan={t_noscan:.3f}s "
                f"fsq_ok={fsq_ok} -> use_scan={selected_scan}",
                flush=True,
            )
        else:
            print_func(
                "[vmec_jax] dynamic scan parity probe: "
                f"backend={probe_backend} iters={pre_iters} "
                f"fsq_ok={fsq_ok} -> use_scan={selected_scan}",
                flush=True,
            )
    return selected_scan


def maybe_disable_scan_by_parity_guard(
    *,
    accelerated_mode: bool,
    performance_mode: bool,
    scan_mode: bool,
    niter: int,
    state_stage_start: Any,
    static_stage: Any,
    indata: Any,
    signgs: int,
    ftol: float,
    step_size: float,
    use_restart_triggers: bool | None,
    vmecpp_restart: bool,
    stage_transition_factor: float,
    stage_transition_scale: float,
    scan_minimal_default: bool | None,
    jit_forces: Any,
    resolve_jit_forces: Callable[[Any, Any, int], bool],
    solve_fixed_boundary_residual_iter: Callable[..., Any],
    verbose: bool,
    print_func: Callable[..., None] = print,
    getenv: Callable[[str, str], str] = os.getenv,
) -> bool:
    """Optionally disable scan when a short parity probe diverges.

    In ``auto`` mode this guard runs only for performance scan paths.  It
    compares a short scan and non-scan VMEC2000-style prefix and keeps scan only
    when the residual histories agree.  Reference/parity solves skip the probe
    because they already choose the VMEC-control branch for accuracy.
    """

    scan_guard_env = getenv("VMEC_JAX_SCAN_PARITY_GUARD", "auto").strip().lower()
    if scan_guard_env in ("", "auto", "default"):
        scan_guard_enabled = bool(performance_mode)
    else:
        scan_guard_enabled = scan_guard_env not in ("0", "false", "no", "off")
    if bool(accelerated_mode) or (not bool(scan_mode)) or (not bool(scan_guard_enabled)) or int(niter) < 3:
        return bool(scan_mode)

    probe_iters = min(10, int(niter))
    try:
        guard_rtol = float(getenv("VMEC_JAX_SCAN_GUARD_RTOL", "1e-3"))
        guard_atol = float(getenv("VMEC_JAX_SCAN_GUARD_ATOL", "1e-12"))
        probe_kwargs = dict(
            indata=indata,
            signgs=signgs,
            ftol=float(ftol),
            max_iter=int(probe_iters),
            step_size=float(step_size),
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
            state_stage_start,
            static_stage,
            jit_forces=resolve_jit_forces(jit_forces, static_stage, int(probe_iters)),
            use_scan=True,
            **probe_kwargs,
        )
        res_probe_direct = solve_fixed_boundary_residual_iter(
            state_stage_start,
            static_stage,
            jit_forces=resolve_jit_forces(jit_forces, static_stage, int(probe_iters)),
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
            if bool(verbose):
                print_func(
                    "[vmec_jax] scan parity guard: disabling scan for this stage (probe mismatch)",
                    flush=True,
                )
            return False
    except Exception as exc:
        if bool(verbose):
            print_func(
                f"[vmec_jax] scan parity guard probe failed ({type(exc).__name__}); "
                "using non-scan for this stage.",
                flush=True,
            )
        return False
    return bool(scan_mode)
