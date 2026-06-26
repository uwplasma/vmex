#!/usr/bin/env python
"""Benchmark monolithic versus segmented accepted-trace controller replay.

This diagnostic is intentionally optional and JSON-producing.  It builds the
same tiny forced-active direct-coil free-boundary traces used by the phase-2
same-branch adjoint report, then replays those accepted traces through
``direct_coil_accepted_trace_controller_replay_objective_jax`` in two modes:

1. monolithic accepted-controller scan,
2. segmented accepted-controller scans split by static preconditioner policy.

By default the script flips a preconditioner policy flag on alternating traces
to synthesize a multi-policy replay with identical trace array shapes.  This is
not production-physics evidence; it is a bounded performance/control-flow
diagnostic for the segment machinery.  Use ``--no-synthetic-multi-policy`` to
time the unmodified production traces.
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

from tools.diagnostics.freeb_replay_diagnostic_utils import block_until_ready as _block
from tools.diagnostics.freeb_replay_diagnostic_utils import json_ready as _json_ready


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "freeb_segmented_replay_report.json",
        help="JSON output path.",
    )
    p.add_argument("--workdir", type=Path, default=None, help="Directory for generated tiny input.")
    p.add_argument("--niter", type=int, default=2, help="Tiny solve iteration count.")
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
        "--synthetic-multi-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Flip alternating preconditioner policy flags to create multiple static-policy segments.",
    )
    p.add_argument(
        "--segment-local-preconditioner-controls",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When segmented replay is enabled and global preconditioner controls "
            "cannot be stacked, try stacking controls independently inside each segment."
        ),
    )
    p.add_argument(
        "--use-stacked-step-controls",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Replay through full step-policy segments with stacked step controls. "
            "This is a stricter GPU-performance diagnostic; it preserves parity "
            "only when every segment can stack the required controls."
        ),
    )
    p.add_argument("--warm-repeats", type=int, default=1, help="Number of repeated replay timings per mode.")
    p.add_argument("--rtol", type=float, default=1.0e-10, help="Replay objective/state parity relative tolerance.")
    p.add_argument("--atol", type=float, default=1.0e-10, help="Replay objective/state parity absolute tolerance.")
    p.add_argument(
        "--fail-on-mismatch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit nonzero if segmented replay differs from monolithic replay.",
    )
    return p


def _with_synthetic_policy_segments(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return trace copies with alternating static preconditioner policies."""

    if len(traces) < 2:
        return [dict(trace) for trace in traces]
    out = [dict(trace) for trace in traces]
    base_lax = bool(out[0].get("preconditioner_use_lax_tridi", False))
    base_precomputed = bool(out[0].get("preconditioner_use_precomputed_tridi", False))
    for index in range(1, len(out), 2):
        out[index] = dict(out[index])
        out[index]["preconditioner_use_lax_tridi"] = not base_lax
        out[index]["preconditioner_use_precomputed_tridi"] = base_precomputed
    return out


def _timed_replay(
    *,
    params: Any,
    initial_state: Any,
    static: Any,
    traces: list[dict[str, Any]],
    signgs: int,
    use_segments: bool,
    use_segment_preconditioner_controls: bool,
    use_stacked_step_controls: bool,
    use_accepted_only_fast_path: bool,
    repeats: int,
) -> tuple[dict[str, Any], list[float]]:
    from vmec_jax.solvers.free_boundary.adjoint.branch_local_derivatives import direct_coil_accepted_trace_controller_replay_objective_jax

    timings: list[float] = []
    replay: dict[str, Any] | None = None
    for _ in range(max(1, int(repeats))):
        t0 = time.perf_counter()
        replay = direct_coil_accepted_trace_controller_replay_objective_jax(
            params,
            initial_state,
            static=static,
            traces=traces,
            signgs=int(signgs),
            state_weight=1.0,
            bsqvac_weight=0.0,
            force_weight=0.0,
            enforce_edge=False,
            use_preconditioner_policy_segments=bool(use_segments),
            use_segment_preconditioner_controls=bool(use_segment_preconditioner_controls),
            use_stacked_step_controls=bool(use_stacked_step_controls),
            use_accepted_only_fast_path=bool(use_accepted_only_fast_path),
        )
        _block(replay["objective"])
        timings.append(time.perf_counter() - t0)
    assert replay is not None
    return replay, timings


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    from tools.diagnostics.direct_coil_same_branch_adjoint_report import (
        _circle_coil_params,
        _configure_validation_nestor_path,
        _restore_env,
        _run_trace,
        _write_tiny_direct_freeb_input,
    )
    from vmec_jax._compat import enable_x64
    from vmec_jax.solvers.free_boundary.adjoint.branch_local_derivatives import direct_coil_accepted_trace_preconditioner_policy_segment_summary
    from vmec_jax.state import pack_state

    enable_x64(True)
    workdir = Path(args.workdir).expanduser().resolve() if args.workdir else Path(args.out).expanduser().resolve().parent
    workdir.mkdir(parents=True, exist_ok=True)
    input_path = _write_tiny_direct_freeb_input(
        workdir / "input.direct_segmented_replay",
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

    replay_traces = _with_synthetic_policy_segments(traces) if bool(args.synthetic_multi_policy) else [dict(t) for t in traces]
    segment_summary = direct_coil_accepted_trace_preconditioner_policy_segment_summary(replay_traces)

    monolithic, monolithic_times = _timed_replay(
        params=params,
        initial_state=replay_traces[0]["state_pre"],
        static=init.static,
        traces=replay_traces,
        signgs=int(init.signgs),
        use_segments=False,
        use_segment_preconditioner_controls=False,
        use_stacked_step_controls=bool(args.use_stacked_step_controls),
        use_accepted_only_fast_path=True,
        repeats=int(args.warm_repeats),
    )
    monolithic_fallback, monolithic_fallback_times = _timed_replay(
        params=params,
        initial_state=replay_traces[0]["state_pre"],
        static=init.static,
        traces=replay_traces,
        signgs=int(init.signgs),
        use_segments=False,
        use_segment_preconditioner_controls=False,
        use_stacked_step_controls=bool(args.use_stacked_step_controls),
        use_accepted_only_fast_path=False,
        repeats=int(args.warm_repeats),
    )
    segmented, segmented_times = _timed_replay(
        params=params,
        initial_state=replay_traces[0]["state_pre"],
        static=init.static,
        traces=replay_traces,
        signgs=int(init.signgs),
        use_segments=True,
        use_segment_preconditioner_controls=bool(args.segment_local_preconditioner_controls),
        use_stacked_step_controls=bool(args.use_stacked_step_controls),
        use_accepted_only_fast_path=True,
        repeats=int(args.warm_repeats),
    )
    segmented_fallback, segmented_fallback_times = _timed_replay(
        params=params,
        initial_state=replay_traces[0]["state_pre"],
        static=init.static,
        traces=replay_traces,
        signgs=int(init.signgs),
        use_segments=True,
        use_segment_preconditioner_controls=bool(args.segment_local_preconditioner_controls),
        use_stacked_step_controls=bool(args.use_stacked_step_controls),
        use_accepted_only_fast_path=False,
        repeats=int(args.warm_repeats),
    )

    monolithic_state = np.asarray(pack_state(monolithic["state"]), dtype=float)
    segmented_state = np.asarray(pack_state(segmented["state"]), dtype=float)
    monolithic_fallback_state = np.asarray(pack_state(monolithic_fallback["state"]), dtype=float)
    segmented_fallback_state = np.asarray(pack_state(segmented_fallback["state"]), dtype=float)
    objective_delta = abs(float(np.asarray(segmented["objective"])) - float(np.asarray(monolithic["objective"])))
    state_max_abs_delta = float(np.max(np.abs(segmented_state - monolithic_state)))
    monolithic_fast_fallback_objective_delta = abs(
        float(np.asarray(monolithic["objective"])) - float(np.asarray(monolithic_fallback["objective"]))
    )
    segmented_fast_fallback_objective_delta = abs(
        float(np.asarray(segmented["objective"])) - float(np.asarray(segmented_fallback["objective"]))
    )
    monolithic_fast_fallback_state_delta = float(np.max(np.abs(monolithic_state - monolithic_fallback_state)))
    segmented_fast_fallback_state_delta = float(np.max(np.abs(segmented_state - segmented_fallback_state)))
    objective_close = bool(
        np.allclose(
            np.asarray(segmented["objective"]),
            np.asarray(monolithic["objective"]),
            rtol=float(args.rtol),
            atol=float(args.atol),
        )
    )
    state_close = bool(np.allclose(segmented_state, monolithic_state, rtol=float(args.rtol), atol=float(args.atol)))
    passed = bool(objective_close and state_close)

    return _json_ready(
        {
            "status": "passed" if passed else "failed",
            "passed": passed,
            "metadata": {
                "diagnostic": "direct_coil_segmented_replay_report",
                "input": str(input_path),
                "workdir": str(workdir),
                "niter": int(args.niter),
                "mpol": int(args.mpol),
                "ntheta": int(args.ntheta),
                "n_segments": int(args.n_segments),
                "synthetic_multi_policy": bool(args.synthetic_multi_policy),
                "segment_local_preconditioner_controls": bool(args.segment_local_preconditioner_controls),
                "use_stacked_step_controls": bool(args.use_stacked_step_controls),
                "trace_generation_wall_s": float(trace_generation_wall_s),
                "note": (
                    "Synthetic multi-policy mode flips static preconditioner policy "
                    "flags to exercise segmented replay control flow; it is not a "
                    "claim that the modified trace sequence came from production."
                ),
            },
            "trace_summary": {
                "n_traces": len(replay_traces),
                "preconditioner_policy_n_segments": len(segment_summary),
                "preconditioner_policy_segment_summary": segment_summary,
                "monolithic_preconditioner_controls_stacked": bool(monolithic["preconditioner_controls_stacked"]),
                "segmented_preconditioner_controls_stacked": bool(segmented["preconditioner_controls_stacked"]),
                "segmented_preconditioner_controls_segment_stacked": tuple(
                    bool(value) for value in segmented["preconditioner_controls_segment_stacked"]
                ),
                "monolithic_used_accepted_only_fast_path": bool(monolithic["used_accepted_only_fast_path"]),
                "segmented_used_accepted_only_fast_path": bool(segmented["used_accepted_only_fast_path"]),
                "monolithic_used_stacked_step_controls": bool(monolithic["used_stacked_step_controls"]),
                "segmented_used_stacked_step_controls": bool(segmented["used_stacked_step_controls"]),
                "monolithic_fallback_used_accepted_only_fast_path": bool(
                    monolithic_fallback["used_accepted_only_fast_path"]
                ),
                "segmented_fallback_used_accepted_only_fast_path": bool(
                    segmented_fallback["used_accepted_only_fast_path"]
                ),
                "monolithic_step_policy_n_segments": int(monolithic["step_policy_n_segments"]),
                "segmented_step_policy_n_segments": int(segmented["step_policy_n_segments"]),
                "monolithic_accepted_only_fast_path_segments": tuple(
                    bool(value) for value in monolithic["accepted_only_fast_path_segments"]
                ),
                "segmented_accepted_only_fast_path_segments": tuple(
                    bool(value) for value in segmented["accepted_only_fast_path_segments"]
                ),
            },
            "timings": {
                "monolithic_replay_s": monolithic_times,
                "segmented_replay_s": segmented_times,
                "monolithic_fallback_replay_s": monolithic_fallback_times,
                "segmented_fallback_replay_s": segmented_fallback_times,
                "monolithic_first_s": float(monolithic_times[0]),
                "segmented_first_s": float(segmented_times[0]),
                "monolithic_fallback_first_s": float(monolithic_fallback_times[0]),
                "segmented_fallback_first_s": float(segmented_fallback_times[0]),
                "speedup_first": (
                    float(monolithic_times[0] / segmented_times[0])
                    if segmented_times and segmented_times[0] > 0.0
                    else None
                ),
                "accepted_only_monolithic_speedup_first": (
                    float(monolithic_fallback_times[0] / monolithic_times[0])
                    if monolithic_times and monolithic_times[0] > 0.0
                    else None
                ),
                "accepted_only_segmented_speedup_first": (
                    float(segmented_fallback_times[0] / segmented_times[0])
                    if segmented_times and segmented_times[0] > 0.0
                    else None
                ),
            },
            "parity": {
                "objective_close": objective_close,
                "state_close": state_close,
                "objective_delta": float(objective_delta),
                "state_max_abs_delta": state_max_abs_delta,
                "monolithic_fast_fallback_objective_delta": float(monolithic_fast_fallback_objective_delta),
                "segmented_fast_fallback_objective_delta": float(segmented_fast_fallback_objective_delta),
                "monolithic_fast_fallback_state_delta": monolithic_fast_fallback_state_delta,
                "segmented_fast_fallback_state_delta": segmented_fast_fallback_state_delta,
                "monolithic_objective": float(np.asarray(monolithic["objective"])),
                "segmented_objective": float(np.asarray(segmented["objective"])),
                "monolithic_fallback_objective": float(np.asarray(monolithic_fallback["objective"])),
                "segmented_fallback_objective": float(np.asarray(segmented_fallback["objective"])),
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
    print(
        "segments="
        f"{report['trace_summary']['preconditioner_policy_n_segments']} "
        f"monolithic_first_s={report['timings']['monolithic_first_s']:.6g} "
        f"segmented_first_s={report['timings']['segmented_first_s']:.6g}"
    )
    if bool(args.fail_on_mismatch) and not bool(report["passed"]):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point.
    raise SystemExit(main())
