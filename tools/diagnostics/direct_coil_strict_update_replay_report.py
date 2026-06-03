#!/usr/bin/env python
"""Isolate strict accepted-update replay timing for direct-coil traces.

This optional diagnostic builds the same tiny forced-active direct-coil
free-boundary trace used by the phase-2 reports, selects one accepted trace,
and times only ``strict_update_one_step_from_trace``.  It intentionally reuses
the stored ``freeb_bsqvac_half`` channel, so the measured path excludes
Biot-Savart boundary resampling and NESTOR replay.  The goal is to separate
force/preconditioner/update compile cost from external-field replay cost.
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


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "freeb_strict_update_replay_report.json",
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
    p.add_argument("--rtol", type=float, default=1.0e-10, help="Trace-static/dynamic parity relative tolerance.")
    p.add_argument("--atol", type=float, default=1.0e-10, help="Trace-static/dynamic parity absolute tolerance.")
    p.add_argument(
        "--fail-on-mismatch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit nonzero if trace-static and dynamic-control update values differ.",
    )
    return p


def _block(value: Any) -> Any:
    if hasattr(value, "block_until_ready"):
        return value.block_until_ready()
    return value


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


def _first_slice(tree: Any) -> Any:
    from vmec_jax._compat import jnp, tree_util

    return tree_util.tree_map(lambda value: jnp.asarray(value)[0], tree)


def _timed(fn: Any, *args: Any, warm_repeats: int) -> tuple[Any, float, list[float]]:
    t0 = time.perf_counter()
    value = fn(*args)
    _block(value)
    first = time.perf_counter() - t0
    warm: list[float] = []
    for _ in range(max(0, int(warm_repeats))):
        t0 = time.perf_counter()
        value = fn(*args)
        _block(value)
        warm.append(time.perf_counter() - t0)
    return value, first, warm


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
        direct_coil_accepted_trace_array_controls_jax,
        direct_coil_accepted_trace_preconditioner_controls_jax,
        direct_coil_accepted_trace_scalar_controls_jax,
    )
    from vmec_jax.discrete_adjoint import strict_update_one_step_from_trace
    from vmec_jax.state import pack_state

    enable_x64(True)
    if jax is None:  # pragma: no cover - JAX is required for this diagnostic.
        raise RuntimeError("JAX is required for strict-update replay diagnostics.")

    workdir = Path(args.workdir).expanduser().resolve() if args.workdir else Path(args.out).expanduser().resolve().parent
    workdir.mkdir(parents=True, exist_ok=True)
    input_path = _write_tiny_direct_freeb_input(
        workdir / "input.direct_strict_update_replay",
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

    if not traces:
        raise RuntimeError("tiny direct-coil solve produced no accepted traces")
    trace_index = int(args.trace_index)
    if trace_index < 0:
        trace_index += len(traces)
    if trace_index < 0 or trace_index >= len(traces):
        raise IndexError(f"trace-index {args.trace_index} is out of range for {len(traces)} traces")
    trace = traces[trace_index]

    scalar_controls_all = _first_slice(direct_coil_accepted_trace_scalar_controls_jax([trace]))
    scalar_controls = {
        key: value
        for key, value in scalar_controls_all.items()
        if key
        not in (
            "preconditioner_use_precomputed_tridi",
            "preconditioner_use_lax_tridi",
        )
    }
    array_controls = _first_slice(direct_coil_accepted_trace_array_controls_jax([trace]))
    preconditioner_controls = _first_slice(direct_coil_accepted_trace_preconditioner_controls_jax([trace]))
    state0 = trace["state_pre"]
    static = init.static

    def _objective_from_step(step_out):
        packed = jnp.asarray(pack_state(step_out["step"]["state_post"]))
        force_leaf_sum = sum(
            jnp.sum(jnp.asarray(leaf) * jnp.asarray(leaf))
            for leaf in jax.tree_util.tree_leaves(step_out["force"])
            if leaf is not None
        )
        return 0.5 * jnp.vdot(packed, packed) + jnp.asarray(1.0e-18, dtype=packed.dtype) * force_leaf_sum

    def _trace_static_objective(state):
        step_out = strict_update_one_step_from_trace(
            state,
            static,
            trace,
            freeb_bsqvac_half=trace.get("freeb_bsqvac_half"),
            enforce_edge=False,
        )
        return _objective_from_step(step_out)

    def _dynamic_controls_objective(state, step_scalars, step_arrays, step_preconditioner):
        step_out = strict_update_one_step_from_trace(
            state,
            static,
            trace,
            scalar_controls=step_scalars,
            array_controls=step_arrays,
            preconditioner_controls=step_preconditioner,
            freeb_bsqvac_half=trace.get("freeb_bsqvac_half"),
            enforce_edge=False,
        )
        return _objective_from_step(step_out)

    trace_static_jit = jax.jit(_trace_static_objective)
    dynamic_controls_jit = jax.jit(_dynamic_controls_objective)

    trace_static_value, trace_static_first, trace_static_warm = _timed(
        trace_static_jit,
        state0,
        warm_repeats=int(args.warm_repeats),
    )
    dynamic_value, dynamic_first, dynamic_warm = _timed(
        dynamic_controls_jit,
        state0,
        scalar_controls,
        array_controls,
        preconditioner_controls,
        warm_repeats=int(args.warm_repeats),
    )

    trace_static_float = float(np.asarray(trace_static_value))
    dynamic_float = float(np.asarray(dynamic_value))
    value_delta = abs(dynamic_float - trace_static_float)
    passed = bool(np.allclose(dynamic_float, trace_static_float, rtol=float(args.rtol), atol=float(args.atol)))

    return _json_ready(
        {
            "status": "passed" if passed else "failed",
            "passed": passed,
            "metadata": {
                "diagnostic": "direct_coil_strict_update_replay_report",
                "input": str(input_path),
                "workdir": str(workdir),
                "niter": int(args.niter),
                "trace_index": int(trace_index),
                "n_traces": len(traces),
                "precond_jmax": int(trace.get("precond_jmax", -1)),
                "has_active_freeb_replay": bool(
                    trace.get("freeb_bsqvac_half") is not None and trace.get("freeb_nestor_trace") is not None
                ),
                "trace_generation_wall_s": float(trace_generation_wall_s),
                "note": (
                    "This diagnostic reuses stored freeb_bsqvac_half and therefore "
                    "measures strict force/preconditioner/update replay, not direct-coil "
                    "boundary resampling."
                ),
            },
            "timings": {
                "trace_static_first_s": float(trace_static_first),
                "trace_static_warm_s": trace_static_warm,
                "dynamic_controls_first_s": float(dynamic_first),
                "dynamic_controls_warm_s": dynamic_warm,
                "dynamic_over_trace_static_first": (
                    float(dynamic_first / trace_static_first) if trace_static_first > 0.0 else None
                ),
            },
            "parity": {
                "trace_static_value": trace_static_float,
                "dynamic_controls_value": dynamic_float,
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
        f"trace_static_first_s={report['timings']['trace_static_first_s']:.6g} "
        f"dynamic_controls_first_s={report['timings']['dynamic_controls_first_s']:.6g}"
    )
    if bool(args.fail_on_mismatch) and not bool(report["passed"]):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point.
    raise SystemExit(main())
