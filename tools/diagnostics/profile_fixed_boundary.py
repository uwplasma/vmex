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
    p.add_argument("--dump-hlo", action="store_true", help="Dump tomnsps_rzl HLO to the output directory")
    return p.parse_args()


def _dump_tomnsps_hlo(input_path: str, outdir: Path) -> None:
    try:
        import jax
        from jax import numpy as jnp
    except Exception as exc:  # pragma: no cover
        print(f"[profile_fixed_boundary] HLO dump skipped (JAX missing): {exc}")
        return
    try:
        from vmec_jax.config import load_config
        from vmec_jax.vmec_tomnsp import vmec_trig_tables, tomnsps_rzl
    except Exception as exc:  # pragma: no cover
        print(f"[profile_fixed_boundary] HLO dump skipped (vmec_jax import failed): {exc}")
        return

    cfg, _ = load_config(input_path)
    trig = vmec_trig_tables(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(cfg.nfp),
        mmax=int(cfg.mpol) - 1,
        nmax=int(cfg.ntor),
        lasym=bool(cfg.lasym),
        dtype=jnp.float64,
        cache=True,
    )
    ns = int(cfg.ns)
    ntheta3 = int(trig.ntheta3)
    nzeta = int(np.asarray(trig.cosnv).shape[0])
    shape = (ns, ntheta3, nzeta)
    aval = jax.ShapeDtypeStruct(shape, jnp.float64)

    def _tomnsps_core(
        armn_even,
        armn_odd,
        brmn_even,
        brmn_odd,
        crmn_even,
        crmn_odd,
        azmn_even,
        azmn_odd,
        bzmn_even,
        bzmn_odd,
        czmn_even,
        czmn_odd,
    ):
        return tomnsps_rzl(
            armn_even=armn_even,
            armn_odd=armn_odd,
            brmn_even=brmn_even,
            brmn_odd=brmn_odd,
            crmn_even=crmn_even,
            crmn_odd=crmn_odd,
            azmn_even=azmn_even,
            azmn_odd=azmn_odd,
            bzmn_even=bzmn_even,
            bzmn_odd=bzmn_odd,
            czmn_even=czmn_even,
            czmn_odd=czmn_odd,
            mpol=int(cfg.mpol),
            ntor=int(cfg.ntor),
            nfp=int(cfg.nfp),
            lasym=bool(cfg.lasym),
            trig=trig,
            include_edge=False,
            masks=None,
        )

    try:
        hlo_ir = (
            jax.jit(_tomnsps_core)
            .lower(aval, aval, aval, aval, aval, aval, aval, aval, aval, aval, aval, aval)
            .compiler_ir(dialect="hlo")
        )
        if hasattr(hlo_ir, "as_text"):
            hlo = hlo_ir.as_text()
        elif hasattr(hlo_ir, "as_hlo_text"):
            hlo = hlo_ir.as_hlo_text()
        else:
            hlo = str(hlo_ir)
    except Exception as exc:  # pragma: no cover
        print(f"[profile_fixed_boundary] HLO dump failed: {exc}")
        return

    out_path = outdir / "tomnsps_rzl.hlo.txt"
    out_path.write_text(hlo, encoding="utf-8")
    print(f"[profile_fixed_boundary] tomnsps_rzl HLO written to {out_path}")


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

    if bool(args.dump_hlo):
        _dump_tomnsps_hlo(str(args.input), outdir)

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
