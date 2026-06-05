#!/usr/bin/env python
"""Write same-branch direct-coil free-boundary adjoint evidence to JSON.

This diagnostic is intentionally small and optional. It runs a tiny
forced-active direct-coil free-boundary solve at a base coil set and at
``base +/- eps * direction``. If the accepted-step fingerprints stay
compatible, it compares

1. central finite differences of the complete accepted solve,
2. the fixed-trace custom-VJP directional derivative, and
3. optionally, the stacked-controller custom-VJP directional derivative.

The result is a reviewer-facing JSON payload for the current phase-2 adjoint
seam. It is not a claim that arbitrary host-controller branch changes are
differentiable.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "freeb_same_branch_adjoint_report.json",
        help="JSON output path.",
    )
    p.add_argument("--workdir", type=Path, default=None, help="Directory for the generated tiny input.")
    p.add_argument("--lasym", action="store_true", help="Run the LASYM=T tiny case instead of stellarator-symmetric.")
    p.add_argument("--eps", type=float, default=1.0e-4, help="Central finite-difference step.")
    p.add_argument("--niter", type=int, default=2, help="Tiny solve iteration count.")
    p.add_argument("--mpol", type=int, default=3, help="Tiny solve MPOL.")
    p.add_argument("--ntheta", type=int, default=6, help="Tiny solve NTHETA.")
    p.add_argument("--n-segments", type=int, default=64, help="Circular coil quadrature segments.")
    p.add_argument("--current", type=float, default=3.0e7, help="Base circular coil current.")
    p.add_argument("--radius", type=float, default=1.8, help="Base circular coil radius.")
    p.add_argument("--current-direction", type=float, default=0.02, help="Relative current perturbation direction.")
    p.add_argument("--radius-direction", type=float, default=5.0e-3, help="Fourier radius perturbation direction.")
    p.add_argument("--rtol", type=float, default=2.0e-3, help="Relative tolerance for slope agreement.")
    p.add_argument("--atol", type=float, default=1.0e-8, help="Absolute tolerance for slope agreement.")
    p.add_argument(
        "--aspect-rtol",
        type=float,
        default=5.0e-3,
        help="Relative tolerance for final aspect-ratio slope agreement.",
    )
    p.add_argument(
        "--aspect-atol",
        type=float,
        default=5.0e-8,
        help="Absolute tolerance for final aspect-ratio slope agreement.",
    )
    p.add_argument("--fingerprint-rtol", type=float, default=1.0e-6)
    p.add_argument("--fingerprint-atol", type=float, default=1.0e-9)
    p.add_argument(
        "--activate-fsq",
        type=float,
        default=1.0e99,
        help="Force active free-boundary coupling in the tiny diagnostic.",
    )
    p.add_argument("--jit-forces", action="store_true", help="Enable JIT force kernels for the tiny solve.")
    p.add_argument(
        "--include-controller-vjp",
        action="store_true",
        help="Also evaluate the stacked-controller custom-VJP slope. This is slower in cold processes.",
    )
    p.add_argument(
        "--include-aspect-scalar-vjp",
        action="store_true",
        help="Also evaluate the final aspect-ratio custom-VJP scalar slope. This is slower in cold processes.",
    )
    p.add_argument(
        "--fail-on-mismatch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit nonzero if branch compatibility or slope checks fail.",
    )
    return p


def _write_tiny_direct_freeb_input(
    path: Path,
    *,
    lasym: bool,
    niter: int,
    mpol: int,
    ntheta: int,
) -> Path:
    lasym_flag = "T" if bool(lasym) else "F"
    path.write_text(
        f"""
&INDATA
  LFREEB = T
  MGRID_FILE = 'DIRECT_COILS'
  EXTCUR = 1.0
  LASYM = {lasym_flag}
  NFP = 1
  MPOL = {int(mpol)}
  NTOR = 0
  NS = 7
  NZETA = 2
  NTHETA = {int(ntheta)}
  NS_ARRAY = 7
  FTOL_ARRAY = 1.0E-8
  NITER_ARRAY = {int(niter)}
  NITER = {int(niter)}
  FTOL = 1.0E-8
  NSTEP = 20
  NVACSKIP = 1
  GAMMA = 0.0
  PHIEDGE = 1.0
  CURTOR = 0.0
  SPRES_PED = 1.0
  NCURR = 0
  PRES_SCALE = 1.0E4
  AM = 1.0 -1.0
  AI = 0.4 0.0
  AC = 0.0
  RAXIS = 1.0
  ZAXIS = 0.0
  RBC(0,0) = 1.0  ZBS(0,0) = 0.0
  RBC(0,1) = 0.25 ZBS(0,1) = 0.25
  RBC(0,2) = 0.03 ZBS(0,2) = 0.00
/
""".lstrip()
    )
    return path


def _circle_coil_params(*, current: float, radius: float, n_segments: int):
    from vmec_jax._compat import jnp
    from vmec_jax.external_fields import CoilFieldParams

    dofs = jnp.zeros((1, 3, 3), dtype=float)
    dofs = dofs.at[0, 0, 2].set(float(radius))
    dofs = dofs.at[0, 1, 1].set(float(radius))
    return CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([float(current)], dtype=float),
        n_segments=int(n_segments),
        nfp=1,
        stellsym=False,
    )


def _configure_validation_nestor_path() -> dict[str, str | None]:
    values = {
        "VMEC_JAX_FREEB_NESTOR_MODE": "dense",
        "VMEC_JAX_FREEB_DENSE_SOLVE_MODE": "mode",
        "VMEC_JAX_FREEB_USE_GREENF_SOURCE": "1",
        "VMEC_JAX_FREEB_EXPERIMENTAL_FOURI_MATRIX": "1",
        "VMEC_JAX_FREEB_ADD_ANALYTIC_BVEC": "1",
        "VMEC_JAX_FREEB_JAX_NESTOR_OPERATOR": "1",
        "VMEC_JAX_FREEB_JAX_NESTOR_JIT_OPERATOR": "0",
    }
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    return previous


def _restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _state_norm_objective(state: Any) -> float:
    from vmec_jax.state import pack_state

    packed = np.asarray(pack_state(state), dtype=float)
    return float(0.5 * np.vdot(packed, packed))


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _block_until_ready(value: Any) -> Any:
    """Synchronize JAX values in nested diagnostic payloads before timing."""

    if hasattr(value, "block_until_ready"):
        return value.block_until_ready()
    if isinstance(value, dict):
        return {key: _block_until_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_block_until_ready(item) for item in value)
    return value


def _slope_report(*, exact: float, fd: float, rtol: float, atol: float) -> dict[str, Any]:
    abs_error = abs(float(exact) - float(fd))
    denom = max(1.0, abs(float(fd)))
    rel_error = abs_error / denom
    passed = bool(np.isfinite(exact) and np.isfinite(fd) and abs_error <= float(atol) + float(rtol) * abs(float(fd)))
    return {
        "exact_directional": float(exact),
        "fd_directional": float(fd),
        "abs_error": float(abs_error),
        "rel_error": float(rel_error),
        "passed": passed,
    }


def _directional_dot(grad: Any, direction: Any):
    from vmec_jax._compat import jax, jnp

    return sum(
        jnp.vdot(grad_leaf, direction_leaf)
        for grad_leaf, direction_leaf in zip(
            jax.tree_util.tree_leaves(grad),
            jax.tree_util.tree_leaves(direction),
            strict=True,
        )
    )


def _solve_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_iter": int(args.niter),
        "ftol": 1.0e-8,
        "vmec2000_control": True,
        "auto_flip_force": False,
        "use_direct_fallback": True,
        "verbose": False,
        "verbose_vmec2000_table": False,
        "jit_forces": bool(args.jit_forces),
        "use_scan": False,
        "host_update_assembly": False,
        "adjoint_trace": True,
        "adjoint_trace_mode": "full",
        "external_field_provider_kind": "direct_coils",
        "free_boundary_activate_fsq": float(args.activate_fsq),
    }


def _run_trace(input_path: Path, params: Any, *, args: argparse.Namespace):
    """Run the tiny direct-coil solve used by optional replay diagnostics."""

    from vmec_jax.free_boundary_adjoint import direct_coil_complete_solve_trace

    payload = direct_coil_complete_solve_trace(
        input_path,
        params,
        solve_kwargs=_solve_kwargs_from_args(args),
        require_active_trace=True,
    )
    return payload["init"], payload["result"], payload["traces"]


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    from vmec_jax._compat import enable_x64, jax, jnp
    from vmec_jax.free_boundary_adjoint import (
        direct_coil_accepted_trace_preconditioner_policy_segment_summary,
        direct_coil_same_branch_complete_solve_fd_report,
        direct_coil_same_branch_controller_scalars_custom_vjp_report,
        direct_coil_fixed_trace_custom_vjp_objective_jax,
    )
    from vmec_jax.state import pack_state
    from vmec_jax.wout import equilibrium_aspect_ratio_from_state

    if jax is None:
        raise RuntimeError("JAX is required for same-branch adjoint diagnostics.")
    enable_x64(True)

    workdir = Path(args.workdir).expanduser().resolve() if args.workdir else Path(args.out).expanduser().resolve().parent
    workdir.mkdir(parents=True, exist_ok=True)
    input_path = _write_tiny_direct_freeb_input(
        workdir / ("input.direct_same_branch_lasym" if args.lasym else "input.direct_same_branch_stellsym"),
        lasym=bool(args.lasym),
        niter=int(args.niter),
        mpol=int(args.mpol),
        ntheta=int(args.ntheta),
    )

    base_params = _circle_coil_params(
        current=float(args.current),
        radius=float(args.radius),
        n_segments=int(args.n_segments),
    )
    base_dofs = jnp.asarray(base_params.base_curve_dofs)
    base_currents = jnp.asarray(base_params.base_currents)
    direction = base_params.with_arrays(
        base_curve_dofs=jnp.zeros_like(base_dofs).at[0, 0, 2].set(float(args.radius_direction)),
        base_currents=base_currents * float(args.current_direction),
    )

    def params_for(scale: float):
        return base_params.with_arrays(
            base_curve_dofs=base_dofs.at[0, 0, 2].add(float(args.radius_direction) * float(scale)),
            base_currents=base_currents * (1.0 + float(args.current_direction) * float(scale)),
        )

    previous_env = _configure_validation_nestor_path()
    build_t0 = time.perf_counter()
    timings: dict[str, float] = {}
    try:
        t0 = time.perf_counter()
        def complete_objectives(payload: dict[str, Any]) -> dict[str, float]:
            aspect = float(
                np.asarray(
                    equilibrium_aspect_ratio_from_state(
                        state=payload["result"].state,
                        static=payload["init"].static,
                    )
                )
            )
            return {
                "state_norm": _state_norm_objective(payload["result"].state),
                "aspect": aspect,
            }

        complete_report = direct_coil_same_branch_complete_solve_fd_report(
            input_path,
            base_params,
            params_for=params_for,
            objective_fn=complete_objectives,
            eps=float(args.eps),
            solve_kwargs=_solve_kwargs_from_args(args),
            fingerprint_rtol=float(args.fingerprint_rtol),
            fingerprint_atol=float(args.fingerprint_atol),
        )
        wall_s = time.perf_counter() - t0
        timings["complete_solve_fd_wall_s"] = float(wall_s)
    finally:
        _restore_env(previous_env)

    base_init = complete_report["base"]["init"]
    base_result = complete_report["base"]["result"]
    base_traces = complete_report["base"]["traces"]
    plus_result = complete_report["plus"]["result"]
    minus_result = complete_report["minus"]["result"]
    plus_branch = _json_ready(complete_report["branch_compatibility"]["plus"])
    minus_branch = _json_ready(complete_report["branch_compatibility"]["minus"])
    same_branch = bool(complete_report["branch_compatibility"]["same_branch"])
    preconditioner_segment_summary = direct_coil_accepted_trace_preconditioner_policy_segment_summary(base_traces)

    base_complete = float(complete_report["values"]["base"])
    plus_complete = float(complete_report["values"]["plus"])
    minus_complete = float(complete_report["values"]["minus"])
    complete_fd = float(complete_report["values"]["central_fd_directional"])
    base_aspect = float(
        np.asarray(equilibrium_aspect_ratio_from_state(state=base_result.state, static=base_init.static))
    )
    plus_aspect = float(
        np.asarray(equilibrium_aspect_ratio_from_state(state=plus_result.state, static=base_init.static))
    )
    minus_aspect = float(
        np.asarray(equilibrium_aspect_ratio_from_state(state=minus_result.state, static=base_init.static))
    )
    complete_aspect_fd = (plus_aspect - minus_aspect) / (2.0 * float(args.eps))

    replay_kwargs = {
        "static": base_init.static,
        "traces": base_traces,
        "signgs": int(base_init.signgs),
        "state_weight": 1.0,
        "bsqvac_weight": 0.0,
        "force_weight": 0.0,
        "enforce_edge": False,
    }
    controller_replay_kwargs = {
        **replay_kwargs,
        # The promoted same-branch gates use stacked step controls.  Keeping
        # this diagnostic on the same path avoids the global per-step
        # trace-switch closure that makes optional controller reports much more
        # expensive in cold processes.
        "use_stacked_step_controls": True,
    }

    def fixed_objective(params):
        return direct_coil_fixed_trace_custom_vjp_objective_jax(
            params,
            base_traces[0]["state_pre"],
            **replay_kwargs,
        )

    t0 = time.perf_counter()
    fixed_value_jax, fixed_grad = jax.value_and_grad(fixed_objective)(base_params)
    fixed_value = float(np.asarray(fixed_value_jax))
    fixed_exact = float(np.asarray(_directional_dot(fixed_grad, direction)))
    timings["fixed_trace_value_and_grad_wall_s"] = float(time.perf_counter() - t0)

    fixed_report = _slope_report(exact=fixed_exact, fd=complete_fd, rtol=float(args.rtol), atol=float(args.atol))
    base_value_report = {
        "complete_state_norm": float(base_complete),
        "fixed_trace_objective": float(fixed_value),
        "fixed_trace_abs_delta": float(abs(fixed_value - base_complete)),
    }

    base_value_report.update(
        {
            "complete_aspect": float(base_aspect),
        }
    )
    aspect_report: dict[str, Any] = {"status": "skipped", "reason": "pass --include-aspect-scalar-vjp"}
    controller_report: dict[str, Any] = {"status": "skipped", "reason": "pass --include-controller-vjp"}
    fixed_controller_agree: dict[str, Any] = {"status": "skipped", "reason": "pass --include-controller-vjp"}
    controller_scalar_reports = None
    if bool(args.include_aspect_scalar_vjp) or bool(args.include_controller_vjp):
        replay_scalar_fns: dict[str, Any] = {}
        rtol_by_key: dict[str, float] = {}
        atol_by_key: dict[str, float] = {}
        base_value_atol_by_key: dict[str, float] = {}
        if bool(args.include_controller_vjp):
            replay_scalar_fns["state_norm"] = lambda replay, _base: 0.5 * jnp.vdot(
                pack_state(replay["state"]),
                pack_state(replay["state"]),
            )
            rtol_by_key["state_norm"] = float(args.rtol)
            atol_by_key["state_norm"] = float(args.atol)
            base_value_atol_by_key["state_norm"] = 2.0e-3
        if bool(args.include_aspect_scalar_vjp):
            replay_scalar_fns["aspect"] = lambda replay, base: equilibrium_aspect_ratio_from_state(
                state=replay["state"],
                static=base["init"].static,
            )
            rtol_by_key["aspect"] = float(args.aspect_rtol)
            atol_by_key["aspect"] = float(args.aspect_atol)
            base_value_atol_by_key["aspect"] = 2.0e-3
        t0 = time.perf_counter()
        controller_scalar_reports = direct_coil_same_branch_controller_scalars_custom_vjp_report(
            complete_report,
            base_params,
            direction,
            replay_scalar_fns=replay_scalar_fns,
            eps=float(args.eps),
            replay_kwargs=controller_replay_kwargs,
            rtol=rtol_by_key,
            atol=atol_by_key,
            base_value_atol=base_value_atol_by_key,
            compute_frozen_fd=False,
        )
        _block_until_ready(
            {
                "values": controller_scalar_reports["values"],
                "exact_directionals": controller_scalar_reports["exact_directionals"],
                "jacobian": controller_scalar_reports["jacobian"],
            }
        )
        timings["controller_scalar_vjp_wall_s"] = float(time.perf_counter() - t0)
    if bool(args.include_aspect_scalar_vjp):
        assert controller_scalar_reports is not None
        aspect_helper_report = controller_scalar_reports["scalar_reports"]["aspect"]
        aspect_value = float(np.asarray(aspect_helper_report["base_value"], dtype=float))
        aspect_report = _slope_report(
            exact=float(np.asarray(aspect_helper_report["exact_directional"], dtype=float)),
            fd=float(aspect_helper_report["complete_fd_directional"]),
            rtol=float(args.aspect_rtol),
            atol=float(args.aspect_atol),
        )
        aspect_report["frozen_trace_fd_directional"] = aspect_helper_report["frozen_trace_fd_directional"]
        aspect_report["base_abs_delta"] = aspect_helper_report["base_abs_delta"]
        aspect_report["compute_frozen_fd"] = False
        base_value_report.update(
            {
                "controller_trace_aspect": float(aspect_value),
                "controller_trace_aspect_abs_delta": float(abs(aspect_value - base_aspect)),
            }
        )
    if bool(args.include_controller_vjp):
        assert controller_scalar_reports is not None
        state_norm_report = controller_scalar_reports["scalar_reports"]["state_norm"]
        controller_value = float(np.asarray(state_norm_report["base_value"], dtype=float))
        controller_exact = float(np.asarray(state_norm_report["exact_directional"], dtype=float))
        controller_report = _slope_report(
            exact=controller_exact,
            fd=float(state_norm_report["complete_fd_directional"]),
            rtol=float(args.rtol),
            atol=float(args.atol),
        )
        fixed_controller_agree = _slope_report(
            exact=controller_exact,
            fd=fixed_exact,
            rtol=float(args.rtol),
            atol=float(args.atol),
        )
        base_value_report.update(
            {
                "controller_trace_objective": float(controller_value),
                "controller_trace_abs_delta": float(abs(controller_value - base_complete)),
            }
        )
    timings["build_report_wall_s"] = float(time.perf_counter() - build_t0)
    passed = bool(
        same_branch
        and fixed_report["passed"]
        and base_value_report["fixed_trace_abs_delta"] < 2.0e-3
    )
    if bool(args.include_aspect_scalar_vjp):
        passed = bool(
            passed
            and aspect_report["passed"]
            and base_value_report["controller_trace_aspect_abs_delta"] < 2.0e-3
        )
    if bool(args.include_controller_vjp):
        passed = bool(
            passed
            and controller_report["passed"]
            and fixed_controller_agree["passed"]
            and base_value_report["controller_trace_abs_delta"] < 2.0e-3
        )
    return _json_ready(
        {
            "status": "passed" if passed else "failed",
            "passed": passed,
            "metadata": {
                "diagnostic": "direct_coil_same_branch_adjoint_report",
                "lasym": bool(args.lasym),
                "eps": float(args.eps),
                "niter": int(args.niter),
                "mpol": int(args.mpol),
                "ntheta": int(args.ntheta),
                "n_segments": int(args.n_segments),
                "workdir": str(workdir),
                "input": str(input_path),
                "wall_s": float(wall_s),
                "timings": timings,
                "include_controller_vjp": bool(args.include_controller_vjp),
                "include_aspect_scalar_vjp": bool(args.include_aspect_scalar_vjp),
                "note": (
                    "Same-branch phase-2 evidence only. Adaptive host-controller "
                    "branch changes are guarded by fingerprints and are not claimed differentiable."
                ),
            },
            "branch_compatibility": {
                "same_branch": same_branch,
                "plus": plus_branch,
                "minus": minus_branch,
            },
            "accepted_trace_controls": {
                "preconditioner_policy_segment_summary": preconditioner_segment_summary,
                "preconditioner_policy_n_segments": len(preconditioner_segment_summary),
            },
            "complete_solve_values": {
                "base": float(base_complete),
                "plus": float(plus_complete),
                "minus": float(minus_complete),
                "central_fd_directional": float(complete_fd),
            },
            "complete_solve_objective_values": complete_report["objective_values"],
            "complete_solve_primary_objective": complete_report["primary_objective"],
            "complete_solve_aspect": {
                "base": float(base_aspect),
                "plus": float(plus_aspect),
                "minus": float(minus_aspect),
                "central_fd_directional": float(complete_aspect_fd),
            },
            "base_value_consistency": base_value_report,
            "checks": {
                "fixed_trace_custom_vjp_vs_complete_fd": fixed_report,
                "controller_custom_vjp_aspect_vs_complete_fd": aspect_report,
                "controller_custom_vjp_vs_complete_fd": controller_report,
                "controller_vs_fixed_trace_custom_vjp": fixed_controller_agree,
            },
        }
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_report(args)
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"Wrote {out}")
    print(f"status={report['status']}")
    if args.fail_on_mismatch and not bool(report["passed"]):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point.
    raise SystemExit(main())
