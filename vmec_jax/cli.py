"""Command-line interface for vmec_jax (VMEC2000-like executable)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .driver import run_fixed_boundary, write_wout_from_fixed_boundary_run
from .namelist import read_indata


def _case_from_input(path: Path) -> str:
    name = path.name
    if name.startswith("input."):
        case = name.split("input.", 1)[-1]
    elif name.startswith("input_"):
        case = name.split("input_", 1)[-1]
    else:
        case = path.stem
    return case or path.stem


def resolve_wout_path(*, input_path: Path, outdir: Path | None, output: Path | None) -> Path:
    """Return the default wout path for an input file."""
    if output is not None:
        return Path(output)
    base_dir = Path(outdir) if outdir is not None else input_path.parent
    case = _case_from_input(input_path)
    return base_dir / f"wout_{case}.nc"


def _parse_jit_forces(value: str):
    val = str(value).strip().lower()
    if val in ("auto", ""):
        return "auto"
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid --jit-forces value: {value}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vmec_jax",
        description=(
            "Run vmec_jax in a VMEC2000-like fixed-boundary mode. "
            "Provide a single input.* file and a wout_*.nc will be written."
        ),
    )
    p.add_argument("input", type=str, help="Path to VMEC input file (input.*).")
    p.add_argument("--outdir", type=str, default=None, help="Directory for wout_*.nc output.")
    p.add_argument("--output", type=str, default=None, help="Explicit wout_*.nc path.")
    p.add_argument("--max-iter", type=int, default=None, help="Total iteration budget (default: input NITER).")
    p.add_argument("--solver", type=str, default="vmec2000_iter", help="Solver to use (default: vmec2000_iter).")
    p.add_argument("--step-size", type=float, default=None, help="Time step (DELT). Defaults to input DELT.")
    p.add_argument("--history-size", type=int, default=10, help="History size (LBFGS-style solvers).")
    p.add_argument("--multigrid", action="store_true", help="Enable multigrid staging.")
    p.add_argument("--no-multigrid", dest="multigrid", action="store_false", help="Disable multigrid staging.")
    p.set_defaults(multigrid=None)
    p.add_argument(
        "--parity",
        action="store_true",
        help="Use VMEC2000 parity loop (slower, exact time-step control).",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Use scan-based fast loop (default).",
    )
    p.add_argument(
        "--vmecpp-restart",
        dest="vmecpp_restart",
        action="store_true",
        help="Enable VMEC++ bad-progress restart heuristic (fast path).",
    )
    p.add_argument(
        "--no-vmecpp-restart",
        dest="vmecpp_restart",
        action="store_false",
        help="Disable VMEC++ bad-progress restarts.",
    )
    p.set_defaults(vmecpp_restart=None)
    p.add_argument(
        "--use-input-niter",
        dest="use_input_niter",
        action="store_true",
        help="Honor NITER_ARRAY/FTOL_ARRAY from the input (default).",
    )
    p.add_argument(
        "--no-use-input-niter",
        dest="use_input_niter",
        action="store_false",
        help="Ignore NITER_ARRAY/FTOL_ARRAY and distribute max_iter uniformly.",
    )
    p.set_defaults(use_input_niter=True)
    p.add_argument("--jit-forces", type=str, default="auto", help="JIT force kernels: auto|true|false.")
    p.add_argument("--quiet", action="store_true", help="Silence VMEC-style stdout.")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        parser.error(f"input file not found: {input_path}")

    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else None
    output = Path(args.output).expanduser().resolve() if args.output else None
    wout_path = resolve_wout_path(input_path=input_path, outdir=outdir, output=output)
    wout_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        indata = read_indata(input_path)
    except Exception as exc:  # pragma: no cover - parser already handles invalid input
        parser.error(f"failed to read &INDATA from {input_path}: {exc}")
        return 2

    def _as_list(value):
        if value is None:
            return None
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                return list(value.tolist())
        except Exception:
            pass
        if isinstance(value, (int, float)):
            return [value]
        return None

    max_iter = args.max_iter
    if max_iter is None:
        niter_array = _as_list(indata.get("NITER_ARRAY", None))
        ns_array = _as_list(indata.get("NS_ARRAY", None))
        if bool(args.use_input_niter) and niter_array and (not ns_array or len(niter_array) == len(ns_array)):
            max_iter = int(sum(int(v) for v in niter_array))
        else:
            max_iter = int(indata.get_int("NITER", 10))

    try:
        jit_forces = _parse_jit_forces(args.jit_forces)
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    if bool(args.parity) and bool(args.fast):
        parser.error("--parity and --fast are mutually exclusive")
        return 2
    performance_mode = True
    if bool(args.parity):
        performance_mode = False
    elif bool(args.fast):
        performance_mode = True
    if args.vmecpp_restart is None:
        vmecpp_restart = False
    else:
        vmecpp_restart = bool(args.vmecpp_restart)

    profile_dir = os.getenv("VMEC_JAX_PROFILE_DIR", "")
    profile_window = os.getenv("VMEC_JAX_PROFILE_WINDOW", "")
    profile_server = os.getenv("VMEC_JAX_PROFILE_SERVER", "")
    server_handle = None
    if profile_server and profile_server not in ("0", "false", "False"):
        try:
            import jax

            port_env = os.getenv("VMEC_JAX_PROFILE_SERVER_PORT", "9999")
            server_handle = jax.profiler.start_server(int(port_env))
        except Exception:
            server_handle = None

    if profile_dir and not profile_window:
        try:
            import jax

            Path(profile_dir).mkdir(parents=True, exist_ok=True)
            perfetto_env = os.getenv("VMEC_JAX_PROFILE_PERFETTO", "1")
            perfetto_trace = perfetto_env.strip().lower() not in ("", "0", "false", "no")
            jax.profiler.start_trace(
                profile_dir,
                create_perfetto_trace=perfetto_trace,
            )
        except Exception:
            profile_dir = ""

    try:
        run = run_fixed_boundary(
            str(input_path),
            solver=str(args.solver),
            max_iter=int(max_iter),
            step_size=args.step_size,
            history_size=int(args.history_size),
            multigrid=args.multigrid,
            multigrid_use_input_niter=bool(args.use_input_niter),
            verbose=not bool(args.quiet),
            jit_forces=jit_forces,
            performance_mode=bool(performance_mode),
            vmecpp_restart=bool(vmecpp_restart),
        )
        if profile_dir and not profile_window:
            try:
                import jax

                # Ensure pending device work is finished before stopping the trace.
                jax.block_until_ready(run.state.Rcos)
            except Exception:
                pass
    finally:
        if profile_dir and not profile_window:
            try:
                import jax

                jax.profiler.stop_trace()
            except Exception:
                pass
        if server_handle is not None:
            try:
                import jax

                jax.profiler.stop_server()
            except Exception:
                pass

    write_wout_from_fixed_boundary_run(wout_path, run, include_fsq=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
