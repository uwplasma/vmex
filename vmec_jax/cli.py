"""Command-line interface for vmec_jax (VMEC2000-like executable)."""

from __future__ import annotations

import argparse
import os
import re
import shutil
from importlib import resources
from pathlib import Path

from .driver import (
    _default_non_autodiff_solver_policy_for_backend,
    _default_use_scan_for_backend,
    default_non_autodiff_solver_policy,
    run_fixed_boundary,
    write_wout_from_fixed_boundary_run,
)
from .namelist import read_indata

_TEST_INPUT_NAME = "input.nfp4_QH_warm_start"
_TEST_FTOL = 1.0e-12
_PLOT_AUTO = "__vmec_jax_plot_auto__"


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


def _is_wout_path(path: Path) -> bool:
    lower = path.name.lower()
    return path.suffix.lower() == ".nc" and not lower.startswith("boozmn_")


def _is_boozmn_path(path: Path) -> bool:
    return path.name.lower().startswith("boozmn_") and path.suffix.lower() == ".nc"


def _plot_wout_file(wout_path: Path, outdir: Path) -> None:
    from .plotting import plot_wout

    print(f"Plotting VMEC WOUT {wout_path.name} → {outdir}/")
    plot_wout(wout_path, outdir=outdir)


def _plot_boozmn_file(boozmn_path: Path, outdir: Path) -> None:
    from .plotting import plot_boozmn

    print(f"Plotting Boozer output {boozmn_path.name} → {outdir}/")
    paths = plot_boozmn(boozmn_path, outdir=outdir)
    for path in paths.values():
        print(f"  Saved {path}")


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
    """Build the command-line parser for the ``vmec`` executable."""

    p = argparse.ArgumentParser(
        prog="vmec",
        description=(
            "Run vmec_jax equilibrium solver or plot a wout file.\n\n"
            "  vmec --doctor        — print installation and JAX backend diagnostics\n"
            "  vmec --test          — run and plot the bundled quick-start case\n"
            "  vmec input.*         — run the solver\n"
            "  vmec --plot wout_*.nc  — generate diagnostic plots\n"
            "  vmec --booz wout_*.nc  — run booz_xform_jax and write boozmn_*.nc\n\n"
            "Compatibility aliases: vmec_jax, vmec-jax, xvmec_jax."
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
        metavar="PATH",
        type=str,
        nargs="?",
        const=_PLOT_AUTO,
        default=None,
        help=(
            "Generate plots. With a wout_*.nc file, plot WOUT diagnostics. "
            "With a boozmn_*.nc file, plot Boozer diagnostics. With an input.* "
            "file, solve first and then plot the generated WOUT. If PATH is "
            "omitted, the positional input path is used."
        ),
    )
    p.add_argument(
        "--booz",
        action="store_true",
        help=(
            "Run booz_xform_jax after solving an input file, or directly from "
            "a wout_*.nc file. Defaults are mbooz=nbooz=32 and all surfaces."
        ),
    )
    p.add_argument("--mbooz", type=int, default=None, help="Boozer poloidal resolution (default: input &BOOZ_XFORM_JAX MBOOZ or 32).")
    p.add_argument("--nbooz", type=int, default=None, help="Boozer toroidal resolution (default: input &BOOZ_XFORM_JAX NBOOZ or 32).")
    p.add_argument(
        "--booz-surfaces",
        type=str,
        default=None,
        help="Boozer surfaces as comma/space-separated normalized s values, or 'all' (default).",
    )
    p.add_argument("--booz-output", type=str, default=None, help="Explicit boozmn_*.nc output path.")
    p.add_argument(
        "--jit-booz",
        action="store_true",
        help="JIT the booz_xform_jax transform. Accurate but can add compile time for one-off CLI runs.",
    )
    p.add_argument(
        "--doctor",
        action="store_true",
        help="Print installation, Python, pip, package, and JAX backend diagnostics.",
    )
    p.add_argument(
        "--test",
        action="store_true",
        help=(
            "Run the bundled input.nfp4_QH_warm_start quick-start case, write "
            "the input and wout files, and plot the resulting wout."
        ),
    )
    p.add_argument("--outdir", type=str, default=None, help="Directory for wout_*.nc output.")
    p.add_argument("--output", type=str, default=None, help="Explicit wout_*.nc path.")
    p.add_argument("--max-iter", type=int, default=None, help="Total iteration budget (default: input NITER).")
    p.add_argument("--solver", type=str, default="vmec2000_iter", help="Solver to use (default: vmec2000_iter).")
    p.add_argument(
        "--solver-mode",
        type=str,
        default=None,
        help=(
            "Solver policy: default|parity|accelerated|memory "
            "(memory is a low-peak-memory alias for parity; default uses current production policy)."
        ),
    )
    p.add_argument(
        "--solver-device",
        type=str,
        default=None,
        help="JAX solver device override: auto|default|cpu|gpu (default: auto).",
    )
    p.add_argument("--step-size", type=float, default=None, help="Time step (DELT). Defaults to input DELT.")
    p.add_argument("--history-size", type=int, default=10, help="History size (LBFGS-style solvers).")
    p.add_argument(
        "--finish-policy",
        type=str,
        default="auto",
        choices=("auto", "none", "bounded", "converge"),
        help=(
            "Fixed-boundary post-solve policy: auto preserves default converged CLI behavior; "
            "none/bounded runs only the requested budget; converge forces the VMEC-style finish stage."
        ),
    )
    p.add_argument(
        "--no-finish",
        dest="finish_policy",
        action="store_const",
        const="none",
        help="Alias for --finish-policy none; useful for exact-budget profiling and quick bounded runs.",
    )
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


def _booz_config_for_path(input_path: Path | None, args: argparse.Namespace):
    from .booz import BoozConfig, parse_booz_surfaces, read_booz_config

    cfg = read_booz_config(input_path) if input_path is not None and input_path.exists() else BoozConfig()
    cli_surfaces = parse_booz_surfaces(args.booz_surfaces) if args.booz_surfaces is not None else None
    return BoozConfig(
        enabled=bool(args.booz) or bool(cfg.enabled),
        mbooz=int(args.mbooz if args.mbooz is not None else cfg.mbooz),
        nbooz=int(args.nbooz if args.nbooz is not None else cfg.nbooz),
        surfaces=cli_surfaces if args.booz_surfaces is not None else cfg.surfaces,
        jit=bool(args.jit_booz) or bool(cfg.jit),
    )


def _run_booz_for_wout(
    wout_path: Path,
    *,
    source_input_path: Path | None,
    args: argparse.Namespace,
    plot: bool,
    outdir: Path,
) -> Path:
    from .booz import resolve_boozmn_path, run_booz_xform

    cfg = _booz_config_for_path(source_input_path, args)
    booz_output = Path(args.booz_output).expanduser().resolve() if args.booz_output else None
    boozmn_path = resolve_boozmn_path(
        source_path=wout_path,
        outdir=outdir,
        output=booz_output,
    )
    print(
        "Running booz_xform_jax "
        f"(mbooz={cfg.mbooz}, nbooz={cfg.nbooz}, surfaces={'all' if cfg.surfaces is None else cfg.surfaces})"
    )
    boozmn_path = run_booz_xform(
        wout_path,
        output_path=boozmn_path,
        mbooz=cfg.mbooz,
        nbooz=cfg.nbooz,
        surfaces=cfg.surfaces,
        jit=cfg.jit,
        verbose=not bool(args.quiet),
    )
    print(f"Wrote Boozer output: {boozmn_path}")
    if plot:
        _plot_boozmn_file(boozmn_path, outdir)
    return boozmn_path


def _copy_test_input(outdir: Path) -> Path:
    """Copy the packaged quick-start input into ``outdir`` and return its path."""
    outdir.mkdir(parents=True, exist_ok=True)
    dst = outdir / _TEST_INPUT_NAME
    resource = resources.files("vmec_jax").joinpath("data", _TEST_INPUT_NAME)
    with resources.as_file(resource) as src:
        shutil.copyfile(src, dst)
    _set_test_input_ftol(dst, ftol=_TEST_FTOL)
    return dst


def _set_test_input_ftol(path: Path, *, ftol: float) -> None:
    """Use a quick-start tolerance without changing the packaged reference deck."""
    text = path.read_text()
    replacement = f"FTOL_ARRAY  = {float(ftol):.0e}"
    new_text, count = re.subn(r"(?im)^\s*FTOL_ARRAY\s*=.*$", f"  {replacement}", text, count=1)
    if count == 0:
        new_text = re.sub(r"(?im)^(\s*NITER_ARRAY\s*=.*)$", rf"\1\n  {replacement}", text, count=1)
    path.write_text(new_text)


def _run_bundled_test(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Run the packaged quick-start case and plot the resulting WOUT file."""
    if args.input is not None:
        parser.error("--test does not take an input path")
    if args.plot is not None:
        parser.error("--test and --plot are mutually exclusive")

    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else Path.cwd().resolve() / "vmec_jax_test"
    input_path = _copy_test_input(outdir)
    wout_path = resolve_wout_path(
        input_path=input_path,
        outdir=outdir,
        output=Path(args.output).expanduser().resolve() if args.output else None,
    )
    plot_dir = outdir / "figures"

    print("vmec bundled test")
    print("-----------------")
    print("This quick-start run uses the packaged fixed-boundary QH warm-start input.")
    print(f"1. Wrote the VMEC input file to: {input_path}")
    print(f"   The copied test input uses FTOL_ARRAY = {_TEST_FTOL:.0e} for a faster first run.")
    print("2. Running vmec to solve the equilibrium and write a WOUT file.")
    print()
    print("Equivalent manual command:")
    print(f"  vmec {input_path} --output {wout_path}")
    print()

    solver_argv = [str(input_path), "--output", str(wout_path)]
    if args.solver_device is not None:
        solver_argv.extend(["--solver-device", str(args.solver_device)])
    if bool(args.parity):
        solver_argv.append("--parity")
    if bool(args.fast):
        solver_argv.append("--fast")
    if args.solver_mode is not None:
        solver_argv.extend(["--solver-mode", str(args.solver_mode)])
    if args.max_iter is not None:
        solver_argv.extend(["--max-iter", str(args.max_iter)])
    if args.jit_forces != "auto":
        solver_argv.extend(["--jit-forces", str(args.jit_forces)])
    if bool(args.quiet):
        solver_argv.append("--quiet")

    rc = main(solver_argv)
    if rc != 0:
        return rc

    print()
    print(f"3. Wrote the WOUT file to: {wout_path}")
    print(f"4. Plotting the WOUT file into: {plot_dir}")
    print()
    print("Equivalent manual plotting command:")
    print(f"  vmec --plot {wout_path} --outdir {plot_dir}")
    print()

    from .plotting import plot_wout

    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_wout(wout_path, outdir=plot_dir)
    print()
    print("Bundled test complete. You can now inspect the input, WOUT file, and figures above.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the ``vmec`` command-line entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if bool(args.doctor):
        if args.input is not None:
            parser.error("--doctor does not take an input path")
        from .doctor import main as doctor_main

        return doctor_main()

    if bool(args.test):
        return _run_bundled_test(args, parser)

    plot_requested = args.plot is not None
    target_arg = args.input
    if args.plot not in (None, _PLOT_AUTO):
        target_arg = args.plot
        if args.input is not None:
            parser.error("provide the target either as --plot PATH or as a positional input, not both")
    if args.plot == _PLOT_AUTO and target_arg is None:
        parser.error("--plot requires a PATH or a positional input")
    if target_arg is None:
        parser.error("provide a VMEC input file, wout_*.nc with --booz, or use --plot PATH")

    input_path = Path(target_arg).expanduser().resolve()
    if not input_path.exists():
        parser.error(f"input file not found: {input_path}")

    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else None
    plot_outdir = outdir if outdir is not None else input_path.parent

    if _is_boozmn_path(input_path):
        if not plot_requested:
            parser.error("boozmn_*.nc inputs are plot-only; use --plot boozmn_*.nc")
        _plot_boozmn_file(input_path, plot_outdir)
        return 0

    if _is_wout_path(input_path):
        if plot_requested:
            _plot_wout_file(input_path, plot_outdir)
        if bool(args.booz):
            _run_booz_for_wout(input_path, source_input_path=None, args=args, plot=plot_requested, outdir=plot_outdir)
        if (not plot_requested) and (not bool(args.booz)):
            parser.error("wout_*.nc inputs require --plot and/or --booz")
        return 0

    # From here onward the target is a VMEC input file.
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
            finish_policy=str(args.finish_policy),
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
    print(f"Wrote WOUT file: {wout_path}")

    if plot_requested:
        _plot_wout_file(wout_path, plot_outdir)

    booz_cfg = _booz_config_for_path(input_path, args)
    if booz_cfg.enabled:
        _run_booz_for_wout(wout_path, source_input_path=input_path, args=args, plot=plot_requested, outdir=plot_outdir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
