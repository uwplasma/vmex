#!/usr/bin/env python3
"""Profile vmec_jax fixed-boundary iterations with JAX profiler.

Example:
  python tools/diagnostics/profile_fixed_boundary.py --input /path/to/input.qa_signgs1 --iters 3 --outdir tmp/jax_profile
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="VMEC input file")
    p.add_argument("--iters", type=int, default=3, help="Number of iterations to run")
    p.add_argument("--outdir", type=str, default="tmp/jax_profile", help="Profiler trace output directory")
    p.add_argument("--no-warmup", action="store_true", help="Disable warmup run")
    p.add_argument("--simple-profile", action="store_true", help="Use a timing-only profiler (no TensorBoard trace)")
    p.add_argument("--jit-forces", action="store_true", help="Enable jit_forces (default)")
    p.add_argument("--no-jit-forces", action="store_true", help="Disable jit_forces")
    p.add_argument("--use-input-niter", action="store_true", help="Use NITER from input for staging")
    p.add_argument("--use-scan", action="store_true", help="Run the lax.scan iteration path")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        import jax
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"JAX is required for profiling: {exc}") from exc

    import vmec_jax.api as vj

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.jit_forces and args.no_jit_forces:
        raise SystemExit("Specify at most one of --jit-forces or --no-jit-forces.")
    jit_forces = True if args.jit_forces else False if args.no_jit_forces else True

    if not args.no_warmup:
        warm = vj.run_fixed_boundary(
            args.input,
            solver="vmec2000_iter",
            max_iter=int(args.iters),
            multigrid_use_input_niter=bool(args.use_input_niter),
            verbose=False,
            jit_forces=bool(jit_forces),
            use_scan=bool(args.use_scan),
        )
        try:
            res = warm.result
            if res is not None and hasattr(res, "fsqr2_history"):
                _ = float(np.asarray(res.fsqr2_history)[-1])
        except Exception:
            pass

    use_simple = bool(args.simple_profile)
    if not use_simple:
        try:
            jax.profiler.start_trace(str(outdir))
        except Exception as exc:
            print(f"[profile_fixed_boundary] profiler start failed: {exc}")
            print("[profile_fixed_boundary] falling back to timing-only profile (no trace).")
            use_simple = True

    if use_simple:
        import time

        t0 = time.perf_counter()
        run = vj.run_fixed_boundary(
            args.input,
            solver="vmec2000_iter",
            max_iter=int(args.iters),
            multigrid_use_input_niter=bool(args.use_input_niter),
            verbose=False,
            jit_forces=bool(jit_forces),
            use_scan=bool(args.use_scan),
        )
        res = run.result
        if res is not None and hasattr(res, "fsqr2_history"):
            _ = float(np.asarray(res.fsqr2_history)[-1])
        t1 = time.perf_counter()
        print(f"[profile_fixed_boundary] total wall time: {t1 - t0:.3f}s")
        return 0

    run = vj.run_fixed_boundary(
        args.input,
        solver="vmec2000_iter",
        max_iter=int(args.iters),
        multigrid_use_input_niter=bool(args.use_input_niter),
        verbose=False,
        jit_forces=bool(jit_forces),
        use_scan=bool(args.use_scan),
    )
    try:
        res = run.result
        if res is not None and hasattr(res, "fsqr2_history"):
            _ = float(np.asarray(res.fsqr2_history)[-1])
    finally:
        jax.profiler.stop_trace()

    print(f"[profile_fixed_boundary] trace saved to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
