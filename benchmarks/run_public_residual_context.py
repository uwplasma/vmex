#!/usr/bin/env python3
"""Benchmark the public implicit residual/Jacobian context at QI scale.

The production preset mirrors the retained four-block QI discretization used
by ``SVD/single_stage_vacuum_jax``: four flux surfaces, ``mboz=nboz=18``,
``nphi=141``, ``nalpha=27``, and ``n_bounce=51``.  The benchmark runs in one
fresh process and separates:

* context construction;
* the first residual and Jacobian calls, including compilation;
* one complete stabilization cycle for lazy warm-start branches;
* repeated steady-state calls at the identical parameter vector.

JAX compilation-monitoring events are recorded per phase.  A steady-state
phase that reports a new backend compilation is flagged as an unexpected
recompilation.  Peak RSS is cumulative because ``getrusage`` exposes the
portable process high-water mark rather than instantaneous memory.

Examples::

    python benchmarks/run_public_residual_context.py --preset quick
    python benchmarks/run_public_residual_context.py \
        --preset production \
        --out benchmarks/public_residual_context_qi.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "examples" / "data"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import jax  # noqa: E402
import numpy as np  # noqa: E402

import vmex  # noqa: E402
from benchmarks._provenance import (  # noqa: E402
    assert_repo_vmex,
    file_sha256,
    git_state,
)
from vmex.core import optimize  # noqa: E402
from vmex.core.input import VmecInput  # noqa: E402


COMPILE_EVENTS = (
    "/jax/core/compile/jaxpr_trace_duration",
    "/jax/core/compile/jaxpr_to_mlir_module_duration",
    "/jax/core/compile/backend_compile_duration",
)


def _peak_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if platform.system() == "Darwin" else raw * 1024


class CompilationMonitor:
    """Count and time JAX compilation events, grouped by benchmark phase."""

    def __init__(self) -> None:
        self.phase = "startup"
        self._counts: dict[str, collections.Counter[str]] = collections.defaultdict(
            collections.Counter
        )
        self._durations: dict[str, collections.Counter[str]] = collections.defaultdict(
            collections.Counter
        )

    def callback(self, event: str, duration: float, **metadata) -> None:
        del metadata
        self._counts[self.phase][event] += 1
        self._durations[self.phase][event] += float(duration)

    def snapshot(self, phase: str) -> dict[str, object]:
        counts = dict(self._counts.get(phase, {}))
        durations = dict(self._durations.get(phase, {}))
        return {
            "counts": counts,
            "duration_s": durations,
            "backend_compile_count": int(
                counts.get("/jax/core/compile/backend_compile_duration", 0)
            ),
        }


def _timed_phase(
    name: str,
    fn: Callable[[], np.ndarray],
    monitor: CompilationMonitor,
) -> tuple[np.ndarray, dict[str, object]]:
    monitor.phase = name
    rss_before = _peak_rss_bytes()
    started = time.perf_counter()
    value = np.asarray(fn(), dtype=float)
    wall = time.perf_counter() - started
    rss_after = _peak_rss_bytes()
    row = {
        "wall_s": wall,
        "peak_rss_bytes_before": rss_before,
        "peak_rss_bytes_after": rss_after,
        "peak_rss_increment_bytes": max(rss_after - rss_before, 0),
        "compilation": monitor.snapshot(name),
    }
    print(
        f"[{name}] wall={wall:.3f}s peak_rss={rss_after / 2**30:.3f} GiB "
        f"backend_compiles={row['compilation']['backend_compile_count']}",
        flush=True,
    )
    return value, row


def _configuration(preset: str) -> dict[str, object]:
    if preset == "quick":
        return {
            "input": DATA / "input.solovev",
            "max_mode": 1,
            "surfaces": [0.5],
            "mboz": 4,
            "nboz": 3,
            "oversample": 1,
            "nphi": 17,
            "nalpha": 5,
            "n_bounce": 7,
            "include_bounce_endpoints": False,
            "softness": 2.0e-2,
            "width_weight": 1.0,
            "branch_width_weight": 0.5,
            "branch_width_softness": 1.0e-2,
            "profile_weight": 0.1,
            "shuffle_profile_weight": 1.0,
            "shuffle_profile_softness": 2.0e-2,
        }
    return {
        "input": DATA / "input.nfp1_QI",
        "max_mode": 4,
        "surfaces": [1 / 16, 5 / 16, 9 / 16, 13 / 16],
        "mboz": 18,
        "nboz": 18,
        "oversample": 2,
        "nphi": 141,
        "nalpha": 27,
        "n_bounce": 51,
        "include_bounce_endpoints": True,
        "softness": 2.0e-2,
        "width_weight": 1.0,
        "branch_width_weight": 0.5,
        "branch_width_softness": 2.0e-2,
        "profile_weight": 0.1,
        "shuffle_profile_weight": 1.0,
        "shuffle_profile_softness": 2.0e-2,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    assert_repo_vmex(vmex.__file__, REPO)
    config = _configuration(args.preset)
    input_path = Path(config.pop("input"))
    inp = VmecInput.from_file(input_path)
    max_mode = int(config.pop("max_mode"))
    canonical_dofs = int(optimize.pack_boundary(inp, max_mode).size)
    parameter_indices = np.arange(0, canonical_dofs, args.parameter_stride)
    residual = optimize.LegacyQIResidual(**config)
    terms = [(residual.residuals_state, 0.0, 1.0)]

    monitor = CompilationMonitor()
    jax.monitoring.register_event_duration_secs_listener(monitor.callback)
    try:
        monitor.phase = "context_construction"
        rss_before = _peak_rss_bytes()
        started = time.perf_counter()
        context = optimize.ImplicitResidualContext(
            inp,
            terms,
            max_mode=max_mode,
            parameter_indices=parameter_indices,
            jac_chunk_size=args.jac_chunk_size,
            jac_solver=args.jac_solver,
            recycle=False,
            warm_start=args.warm_start,
            solve_kwargs={"mode": "cli", "lconm1": True, "use_fft": False},
            device=args.device,
        )
        construction_wall = time.perf_counter() - started
        construction_rss = _peak_rss_bytes()
        construction = {
            "wall_s": construction_wall,
            "peak_rss_bytes_before": rss_before,
            "peak_rss_bytes_after": construction_rss,
            "peak_rss_increment_bytes": max(construction_rss - rss_before, 0),
            "compilation": monitor.snapshot("context_construction"),
        }
        print(
            f"[context_construction] wall={construction_wall:.3f}s "
            f"dofs={context.x0.size}",
            flush=True,
        )

        x = context.x0
        residual0, cold_residual = _timed_phase(
            "cold_residual", lambda: context.residuals(x), monitor
        )
        jacobian0, cold_jacobian = _timed_phase(
            "cold_jacobian", lambda: context.jacobian(x), monitor
        )
        if jacobian0.shape != (residual0.size, x.size):
            raise RuntimeError(
                f"unexpected Jacobian shape {jacobian0.shape}; "
                f"expected {(residual0.size, x.size)}"
            )
        if not np.all(np.isfinite(residual0)) or not np.all(np.isfinite(jacobian0)):
            raise RuntimeError("benchmark residual or Jacobian contains nonfinite values")

        # The first residual after the first Jacobian can initialize a lazy
        # warm-start cache branch. Keep that one-time work out of the measured
        # steady-state recompilation gate.
        residual_stable, stabilization_residual = _timed_phase(
            "stabilization_residual", lambda: context.residuals(x), monitor
        )
        jacobian_stable, stabilization_jacobian = _timed_phase(
            "stabilization_jacobian", lambda: context.jacobian(x), monitor
        )
        np.testing.assert_allclose(residual_stable, residual0, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(jacobian_stable, jacobian0, rtol=0.0, atol=0.0)

        repeats = []
        for index in range(args.repeats):
            residual_i, residual_row = _timed_phase(
                f"warm_residual_{index + 1}", lambda: context.residuals(x), monitor
            )
            jacobian_i, jacobian_row = _timed_phase(
                f"warm_jacobian_{index + 1}", lambda: context.jacobian(x), monitor
            )
            np.testing.assert_allclose(residual_i, residual0, rtol=0.0, atol=0.0)
            np.testing.assert_allclose(jacobian_i, jacobian0, rtol=0.0, atol=0.0)
            repeats.append({"residual": residual_row, "jacobian": jacobian_row})
    finally:
        jax.monitoring.unregister_event_duration_listener(monitor.callback)

    steady_backend_compiles = sum(
        int(call[kind]["compilation"]["backend_compile_count"])
        for call in repeats
        for kind in ("residual", "jacobian")
    )
    input_relative = input_path.resolve().relative_to(REPO.resolve()).as_posix()
    record = {
        "schema_version": 1,
        "benchmark": "public_implicit_residual_context_qi",
        "preset": args.preset,
        "configuration": {
            "input": input_relative,
            "input_sha256": file_sha256(input_path),
            "max_mode": max_mode,
            "canonical_dofs": canonical_dofs,
            "parameter_stride": args.parameter_stride,
            "parameter_indices": parameter_indices.tolist(),
            **config,
            "jac_solver": args.jac_solver,
            "jac_chunk_size": args.jac_chunk_size,
            "warm_start": args.warm_start,
            "repeats": args.repeats,
        },
        "dimensions": {
            "residual_rows": int(residual0.size),
            "active_dofs": int(x.size),
            "jacobian_shape": list(jacobian0.shape),
            "jacobian_bytes": int(jacobian0.nbytes),
        },
        "numerics": {
            "residual_norm": float(np.linalg.norm(residual0)),
            "jacobian_norm": float(np.linalg.norm(jacobian0)),
            "residual_sha256": hashlib.sha256(residual0.tobytes()).hexdigest(),
            "jacobian_sha256": hashlib.sha256(jacobian0.tobytes()).hexdigest(),
            "solve_stats": context.solve_stats,
        },
        "phases": {
            "context_construction": construction,
            "cold_residual": cold_residual,
            "cold_jacobian": cold_jacobian,
            "stabilization_residual": stabilization_residual,
            "stabilization_jacobian": stabilization_jacobian,
            "steady_state_repeats": repeats,
        },
        "steady_state": {
            "unexpected_backend_compilations": steady_backend_compiles,
            "recompilation_gate_passed": steady_backend_compiles == 0,
            "residual_wall_s": [call["residual"]["wall_s"] for call in repeats],
            "jacobian_wall_s": [call["jacobian"]["wall_s"] for call in repeats],
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": jax.__version__,
            "jax_backend": jax.default_backend(),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "jax_devices": [str(device) for device in jax.devices()],
            "vmex_module": assert_repo_vmex(vmex.__file__, REPO),
            "final_peak_rss_bytes": _peak_rss_bytes(),
        },
        "provenance": git_state(REPO),
    }
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("quick", "production"), default="quick")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--parameter-stride", type=int, default=1)
    parser.add_argument("--jac-solver", choices=("block", "gmres", "reverse"), default="block")
    parser.add_argument("--jac-chunk-size", default="auto")
    parser.add_argument("--warm-start", choices=("perturbation", "state"), default="perturbation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.parameter_stride < 1:
        parser.error("--parameter-stride must be at least 1")
    if args.jac_chunk_size != "auto":
        args.jac_chunk_size = int(args.jac_chunk_size)
    if args.device == "none":
        args.device = None

    record = run(args)
    print(json.dumps(record["dimensions"], indent=2), flush=True)
    print(json.dumps(record["steady_state"], indent=2), flush=True)
    if args.out:
        output = Path(args.out)
        output.write_text(json.dumps(record, indent=2) + "\n")
        print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
