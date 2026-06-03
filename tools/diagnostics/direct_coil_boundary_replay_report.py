#!/usr/bin/env python
"""Isolate direct-coil boundary/NESTOR replay timing.

This optional diagnostic complements the strict-update replay report.  It
builds a tiny forced-active direct-coil free-boundary trace, selects one
accepted trace with active free-boundary replay, and times:

1. direct-coil/NESTOR replay on a fixed accepted boundary geometry,
2. accepted-boundary geometry synthesis plus direct-coil/NESTOR replay.

The measured path excludes the strict VMEC force/preconditioner/update map, so
it helps separate boundary-vacuum replay cost from accepted-update cost.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.diagnostics.freeb_replay_diagnostic_utils import json_ready as _json_ready
from tools.diagnostics.freeb_replay_diagnostic_utils import timed_call as _timed


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "freeb_boundary_replay_report.json",
        help="JSON output path.",
    )
    p.add_argument("--workdir", type=Path, default=None, help="Directory for generated tiny input.")
    p.add_argument("--niter", type=int, default=4, help="Tiny solve iteration count.")
    p.add_argument("--mpol", type=int, default=3, help="Tiny solve MPOL.")
    p.add_argument("--ntheta", type=int, default=6, help="Tiny solve NTHETA.")
    p.add_argument("--n-segments", type=int, default=64, help="Circular coil quadrature segments.")
    p.add_argument("--current", type=float, default=3.0e7, help="Base circular coil current.")
    p.add_argument("--radius", type=float, default=1.8, help="Base circular coil radius.")
    p.add_argument(
        "--activate-fsq",
        type=float,
        default=1.0e99,
        help="Force active free-boundary coupling in the tiny diagnostic.",
    )
    p.add_argument("--jit-forces", action="store_true", help="Enable JIT force kernels for the tiny solve.")
    p.add_argument(
        "--trace-index",
        type=int,
        default=-1,
        help="Accepted trace index to time. Negative indices follow Python indexing.",
    )
    p.add_argument("--warm-repeats", type=int, default=2, help="Warm timing repeats after first JIT call.")
    p.add_argument("--rtol", type=float, default=1.0e-10, help="Fixed-geometry/full-geometry parity rtol.")
    p.add_argument("--atol", type=float, default=1.0e-10, help="Fixed-geometry/full-geometry parity atol.")
    p.add_argument(
        "--fail-on-mismatch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit nonzero if fixed-geometry and full-geometry replay values differ.",
    )
    return p


def _select_active_trace(traces: list[dict[str, Any]], trace_index: int) -> tuple[int, dict[str, Any]]:
    if not traces:
        raise RuntimeError("tiny direct-coil solve produced no accepted traces")
    index = int(trace_index)
    if index < 0:
        index += len(traces)
    if index < 0 or index >= len(traces):
        raise IndexError(f"trace-index {trace_index} is out of range for {len(traces)} traces")
    trace = traces[index]
    if trace.get("freeb_bsqvac_half") is not None and trace.get("freeb_nestor_trace") is not None:
        return index, trace
    for fallback_index in range(len(traces) - 1, -1, -1):
        fallback = traces[fallback_index]
        if fallback.get("freeb_bsqvac_half") is not None and fallback.get("freeb_nestor_trace") is not None:
            return fallback_index, fallback
    raise RuntimeError("no accepted trace contains active free-boundary replay metadata")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    from tools.diagnostics.direct_coil_same_branch_adjoint_report import (
        _circle_coil_params,
        _configure_validation_nestor_path,
        _restore_env,
        _run_trace,
        _write_tiny_direct_freeb_input,
    )
    from vmec_jax._compat import enable_x64, jax, jnp
    from vmec_jax.free_boundary_adjoint import (
        direct_coil_boundary_bsqvac_from_trace_jax,
        direct_coil_boundary_replay_context,
        free_boundary_boundary_geometry_jax,
    )

    enable_x64(True)
    if jax is None:  # pragma: no cover - JAX is required for this diagnostic.
        raise RuntimeError("JAX is required for boundary replay diagnostics.")

    workdir = Path(args.workdir).expanduser().resolve() if args.workdir else Path(args.out).expanduser().resolve().parent
    workdir.mkdir(parents=True, exist_ok=True)
    input_path = _write_tiny_direct_freeb_input(
        workdir / "input.direct_boundary_replay",
        lasym=False,
        niter=int(args.niter),
        mpol=int(args.mpol),
        ntheta=int(args.ntheta),
    )
    params = _circle_coil_params(
        current=float(args.current),
        radius=float(args.radius),
        n_segments=int(args.n_segments),
    )

    previous_env = _configure_validation_nestor_path()
    try:
        t0 = time.perf_counter()
        init, _result, traces = _run_trace(input_path, params, args=args)
        trace_generation_wall_s = time.perf_counter() - t0
    finally:
        _restore_env(previous_env)

    trace_index, trace = _select_active_trace(traces, int(args.trace_index))
    t0 = time.perf_counter()
    geometry = free_boundary_boundary_geometry_jax(trace["state_pre"], init.static)
    context = direct_coil_boundary_replay_context(init.static, geometry)
    geometry_context_wall_s = time.perf_counter() - t0

    def _half_norm_bsqvac(replay: dict[str, Any]):
        bsqvac = jnp.asarray(replay["bsqvac"])
        return 0.5 * jnp.vdot(bsqvac, bsqvac)

    def _fixed_geometry_objective(coil_params):
        replay = direct_coil_boundary_bsqvac_from_trace_jax(
            coil_params,
            geometry,
            trace,
            basis=context["basis"],
            tables=context["tables"],
            signgs=int(init.signgs),
            nvper=int(context["nvper"]),
            wint=jnp.asarray(context["wint"]),
            include_analytic=True,
        )
        return _half_norm_bsqvac(replay)

    def _geometry_plus_boundary_objective(coil_params, state):
        geom = free_boundary_boundary_geometry_jax(state, init.static)
        ctx = direct_coil_boundary_replay_context(init.static, geom)
        replay = direct_coil_boundary_bsqvac_from_trace_jax(
            coil_params,
            geom,
            trace,
            basis=ctx["basis"],
            tables=ctx["tables"],
            signgs=int(init.signgs),
            nvper=int(ctx["nvper"]),
            wint=jnp.asarray(ctx["wint"]),
            include_analytic=True,
        )
        return _half_norm_bsqvac(replay)

    fixed_jit = jax.jit(_fixed_geometry_objective)
    full_jit = jax.jit(_geometry_plus_boundary_objective)

    fixed_value, fixed_first, fixed_warm = _timed(
        fixed_jit,
        params,
        warm_repeats=int(args.warm_repeats),
    )
    full_value, full_first, full_warm = _timed(
        full_jit,
        params,
        trace["state_pre"],
        warm_repeats=int(args.warm_repeats),
    )

    fixed_float = float(np.asarray(fixed_value))
    full_float = float(np.asarray(full_value))
    value_delta = abs(full_float - fixed_float)
    passed = bool(np.allclose(full_float, fixed_float, rtol=float(args.rtol), atol=float(args.atol)))

    return _json_ready(
        {
            "status": "passed" if passed else "failed",
            "passed": passed,
            "metadata": {
                "diagnostic": "direct_coil_boundary_replay_report",
                "input": str(input_path),
                "workdir": str(workdir),
                "niter": int(args.niter),
                "trace_index": int(trace_index),
                "n_traces": len(traces),
                "precond_jmax": int(trace.get("precond_jmax", -1)),
                "trace_generation_wall_s": float(trace_generation_wall_s),
                "geometry_context_wall_s": float(geometry_context_wall_s),
                "boundary_shape": tuple(int(v) for v in np.asarray(geometry["R"]).shape),
                "note": (
                    "This diagnostic times direct-coil/NESTOR boundary replay and "
                    "excludes the strict VMEC force/preconditioner/update map."
                ),
            },
            "timings": {
                "fixed_geometry_first_s": float(fixed_first),
                "fixed_geometry_warm_s": fixed_warm,
                "geometry_plus_boundary_first_s": float(full_first),
                "geometry_plus_boundary_warm_s": full_warm,
                "geometry_plus_over_fixed_first": float(full_first / fixed_first) if fixed_first > 0.0 else None,
            },
            "parity": {
                "fixed_geometry_value": fixed_float,
                "geometry_plus_boundary_value": full_float,
                "value_delta": float(value_delta),
                "passed": passed,
            },
        }
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_report(args)
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {out}")
    print(
        "status="
        f"{report['status']} "
        f"fixed_geometry_first_s={report['timings']['fixed_geometry_first_s']:.6g} "
        f"geometry_plus_boundary_first_s={report['timings']['geometry_plus_boundary_first_s']:.6g}"
    )
    if bool(args.fail_on_mismatch) and not bool(report["passed"]):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point.
    raise SystemExit(main())
