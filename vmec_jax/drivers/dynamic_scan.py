"""Dynamic scan selection for fixed-boundary VMEC2000-style stages."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable
import os
import time


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
