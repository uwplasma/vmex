"""Command-line interface for vmec_jax (VMEC2000-like executable)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .driver import (
    _default_non_autodiff_solver_policy_for_backend,
    _default_use_scan_for_backend,
    default_non_autodiff_solver_policy,
    run_fixed_boundary,
    write_wout_from_fixed_boundary_run,
)
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


def _start_wout_io_warmup():
    """Optionally import the heavy netCDF writer dependency during the solve."""
    env = os.getenv("VMEC_JAX_WOUT_IO_WARMUP", "0").strip().lower()
    if env in ("", "0", "false", "no", "off"):
        return None
    try:
        import importlib
        import threading

        def _warmup() -> None:
            try:
                importlib.import_module("netCDF4")
            except Exception:
                pass

        thread = threading.Thread(target=_warmup, name="vmec-jax-wout-io-warmup", daemon=True)
        thread.start()
        return thread
    except Exception:
        return None


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
            "Run vmec_jax equilibrium solver or plot a wout file.\n\n"
            "  vmec_jax input.*           — run the solver\n"
            "  vmec_jax --plot wout_*.nc  — generate diagnostic plots"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "input",
        type=str,
        nargs="?",
        default=None,
        help=(
            "Path to VMEC input file (input.*) when running the solver, "
            "or omit when using --plot."
        ),
    )
    p.add_argument(
        "--plot",
        metavar="wout.nc",
        type=str,
        default=None,
        help="Generate diagnostic plots from a wout_*.nc file (skips the solver).",
    )
    p.add_argument("--outdir", type=str, default=None, help="Directory for wout_*.nc output.")
    p.add_argument("--output", type=str, default=None, help="Explicit wout_*.nc path.")
    p.add_argument("--max-iter", type=int, default=None, help="Total iteration budget (default: input NITER).")
    p.add_argument("--solver", type=str, default="vmec2000_iter", help="Solver to use (default: vmec2000_iter).")
    p.add_argument(
        "--solver-mode",
        type=str,
        default=None,
        help="Solver policy: default|parity|accelerated (default: current default path).",
    )
    p.add_argument(
        "--solver-device",
        type=str,
        default=None,
        help="JAX solver device override: auto|default|cpu|gpu (default: auto).",
    )
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
        help="Use scan-based fast loop.",
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

    # ── --plot mode: generate diagnostic plots from a wout file ────────────────
    if args.plot is not None:
        wout_path = Path(args.plot).expanduser().resolve()
        if not wout_path.exists():
            parser.error(f"wout file not found: {wout_path}")
        outdir = Path(args.outdir).expanduser().resolve() if getattr(args, "outdir", None) else wout_path.parent
        from .plotting import plot_wout
        print(f"Plotting {wout_path.name} → {outdir}/")
        plot_wout(wout_path, outdir=outdir)
        return 0

    if args.input is None:
        parser.error("provide a VMEC input file or use --plot wout.nc")

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        parser.error(f"input file not found: {input_path}")

    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else None
    output = Path(args.output).expanduser().resolve() if args.output else None
    wout_path = resolve_wout_path(input_path=input_path, outdir=outdir, output=output)
    wout_path.parent.mkdir(parents=True, exist_ok=True)
    wout_warmup_thread = _start_wout_io_warmup()

    try:
        indata = read_indata(input_path)
    except Exception as exc:  # pragma: no cover - parser already handles invalid input
        parser.error(f"failed to read &INDATA from {input_path}: {exc}")
        return 2

    # Preserve driver semantics:
    # - default (--use-input-niter): let run_fixed_boundary infer stage budgets
    #   directly from the input (NITER_ARRAY if present, else NITER).
    # - --no-use-input-niter: fall back to a single total budget from NITER
    #   unless --max-iter is explicitly provided.
    max_iter_arg: int | None = args.max_iter
    if max_iter_arg is None and (not bool(args.use_input_niter)):
        max_iter_arg = int(indata.get_int("NITER", 10))

    try:
        jit_forces = _parse_jit_forces(args.jit_forces)
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    if bool(args.parity) and bool(args.fast):
        parser.error("--parity and --fast are mutually exclusive")
        return 2
    if args.solver_mode is not None and (bool(args.parity) or bool(args.fast)):
        parser.error("--solver-mode cannot be combined with --parity/--fast")
        return 2
    solver_mode = args.solver_mode
    default_policy = solver_mode is None and (not bool(args.parity)) and (not bool(args.fast))
    default_policy_backend = None
    if default_policy:
        solver_device_arg = "" if args.solver_device is None else str(args.solver_device).strip().lower()
        if solver_device_arg == "cpu":
            default_policy_backend = "cpu"
            solver_mode, performance_mode = _default_non_autodiff_solver_policy_for_backend(indata, "cpu")
        elif solver_device_arg == "gpu":
            default_policy_backend = "gpu"
            solver_mode, performance_mode = _default_non_autodiff_solver_policy_for_backend(indata, "gpu")
        else:
            default_policy_backend = None
            solver_mode, performance_mode = default_non_autodiff_solver_policy(indata)
    else:
        # Preserve explicit CLI override semantics:
        # - default to the fast scan loop,
        # - use --parity to force the reference path.
        performance_mode = True
        if bool(args.parity):
            performance_mode = False
        if bool(args.fast):
            performance_mode = True
        if solver_mode is None:
            solver_mode = "default" if bool(performance_mode) else "parity"
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
        run_kwargs = dict(
            solver=str(args.solver),
            step_size=args.step_size,
            history_size=int(args.history_size),
            multigrid=args.multigrid,
            multigrid_use_input_niter=bool(args.use_input_niter),
            verbose=not bool(args.quiet),
            jit_forces=jit_forces,
            solver_mode=str(solver_mode),
            solver_device=args.solver_device,
            performance_mode=bool(performance_mode),
            vmecpp_restart=bool(vmecpp_restart),
            cli_fixed_boundary_mode=True,
        )
        if default_policy:
            if default_policy_backend is None:
                try:
                    import jax

                    default_policy_backend = str(jax.default_backend()).strip().lower() or "cpu"
                except Exception:
                    default_policy_backend = "cpu"
            use_scan_default = _default_use_scan_for_backend(
                indata,
                str(default_policy_backend),
                str(solver_mode),
            )
            if str(default_policy_backend).strip().lower() == "cpu":
                use_scan_default = False
            run_kwargs["use_scan"] = bool(use_scan_default)
        if max_iter_arg is not None:
            run_kwargs["max_iter"] = int(max_iter_arg)
        run = run_fixed_boundary(str(input_path), **run_kwargs)
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

    if wout_warmup_thread is not None:
        try:
            wout_warmup_thread.join()
        except Exception:
            pass
    write_wout_from_fixed_boundary_run(wout_path, run, include_fsq=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
