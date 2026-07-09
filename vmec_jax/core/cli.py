"""``vmec`` command-line entry point (new core, fixed-boundary).

VMEC2000 counterpart: the ``xvmec2000`` executable driver
(``Sources/TimeStep/vmec.f`` / ``runvmec.f``): parse the input deck, run the
``NS_ARRAY`` multigrid ladder with VMEC2000-format console output, write the
``wout_<case>.nc`` file, and print the ``fileout.f`` termination summary.

The solve path is the clean-room core end to end:
:class:`vmec_jax.core.input.VmecInput` (INDATA or VMEC++-style JSON) ->
:func:`vmec_jax.core.multigrid.solve_multigrid` ->
:func:`vmec_jax.core.wout.wout_from_state` -> :func:`vmec_jax.core.wout.write_wout`,
plus the core plotting (``--plot``) and Boozer (``--booz``) drivers.

Zero-crash policy (plan.md §2.5): every failure maps to a typed
:class:`vmec_jax.core.errors.VmecError`; the CLI prints the VMEC2000
``werror`` message plus a one-line hint and exits with the matching
``ier_flag`` code.  Free-boundary decks are not yet served by the core: a
missing mgrid file falls back to a fixed-boundary solve with a warning
(VMEC2000 behavior), any other ``LFREEB = T`` deck is redirected to the
legacy entry point (``python -m vmec_jax``).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import time
from importlib import resources
from pathlib import Path

# Suppress noisy C++ logging from XLA/PjRt before any JAX import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

from .errors import (
    INPUT_ERROR_FLAG,
    WERROR_MESSAGES,
    VmecError,
    VmecInputError,
)

_TEST_INPUT_NAME = "input.nfp4_QH_warm_start"
_TEST_FTOL = 1.0e-12
_PLOT_AUTO = "__vmec_jax_plot_auto__"


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("vmec-jax")
    except Exception:  # pragma: no cover - metadata unavailable in-place
        return "unknown"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def case_from_input(path: Path) -> str:
    """Case name from an input path (``input.solovev`` -> ``solovev``)."""
    name = path.name
    if name.startswith("input."):
        case = name.split("input.", 1)[-1]
    elif name.startswith("input_"):
        case = name.split("input_", 1)[-1]
    else:
        case = path.stem
    case = case[: -len(".json")] if case.endswith(".json") else case
    return case or path.stem


def resolve_wout_path(*, input_path: Path, outdir: Path | None) -> Path:
    """Default ``wout_<case>.nc`` path for an input file."""
    base_dir = Path(outdir) if outdir is not None else input_path.parent
    return base_dir / f"wout_{case_from_input(input_path)}.nc"


def _is_wout_path(path: Path) -> bool:
    lower = path.name.lower()
    return path.suffix.lower() == ".nc" and not lower.startswith("boozmn_")


def _is_boozmn_path(path: Path) -> bool:
    return path.name.lower().startswith("boozmn_") and path.suffix.lower() == ".nc"


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the ``vmec`` executable."""
    p = argparse.ArgumentParser(
        prog="vmec",
        description=(
            "vmec_jax equilibrium solver (fixed-boundary core).\n\n"
            "  vmec input.X           — solve (INDATA or VMEC++ JSON), write wout_X.nc\n"
            "  vmec --plot wout_*.nc  — diagnostic plots from a WOUT file\n"
            "  vmec --booz wout_*.nc  — run booz_xform_jax, write boozmn_*.nc\n"
            "  vmec --plot boozmn_*.nc— Boozer contour/spectrum plots\n"
            "  vmec --doctor          — installation and JAX backend diagnostics\n"
            "  vmec --test            — run and plot the bundled quick-start case\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "input",
        type=str,
        nargs="?",
        default=None,
        help=(
            "VMEC input file (input.* namelist or VMEC++ .json) to solve, or a "
            "wout_*.nc/boozmn_*.nc file for --plot/--booz."
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
            "Generate plots. With a wout_*.nc file, plot WOUT diagnostics; with a "
            "boozmn_*.nc file, plot Boozer diagnostics; with an input file, solve "
            "first and plot the resulting WOUT. If PATH is omitted, the positional "
            "input path is used."
        ),
    )
    p.add_argument(
        "--booz",
        action="store_true",
        help=(
            "Run booz_xform_jax after solving an input file, or directly from a "
            "wout_*.nc file, and write boozmn_*.nc."
        ),
    )
    p.add_argument("--mbooz", type=int, default=32, help="Boozer poloidal resolution (default: 32).")
    p.add_argument("--nbooz", type=int, default=32, help="Boozer toroidal resolution (default: 32).")
    p.add_argument(
        "--booz-surfaces",
        type=str,
        default=None,
        help="Boozer surfaces: comma/space-separated normalized s values, or 'all' (default).",
    )
    p.add_argument("--outdir", type=str, default=None, help="Directory for wout/boozmn/figure output (default: alongside the input).")
    p.add_argument("--quiet", action="store_true", help="Silence the VMEC-style stdout.")
    p.add_argument(
        "--mode",
        type=str,
        default="cli",
        choices=("cli", "jit"),
        help=(
            "Solver lane: 'cli' (jitted blocks with host residual checks, live "
            "printing, exact-ftol exit) or 'jit' (single lax.while_loop)."
        ),
    )
    p.add_argument("--ftol", type=float, default=None, help="Override the final-stage FTOL_ARRAY tolerance.")
    p.add_argument("--max-iter", type=int, default=None, help="Override the final-stage NITER_ARRAY iteration cap.")
    p.add_argument(
        "--doctor",
        action="store_true",
        help="Print installation, Python, package, and JAX backend diagnostics.",
    )
    p.add_argument(
        "--test",
        action="store_true",
        help=(
            "Run the bundled input.nfp4_QH_warm_start quick-start case: solve, "
            "write the wout file, and plot it."
        ),
    )
    p.add_argument("--version", action="version", version=f"vmec-jax {_package_version()}")
    return p


def _parse_booz_surfaces(text: str | None):
    """Parse ``--booz-surfaces``: ``None``/'all' -> all, else float list."""
    if text is None:
        return None
    cleaned = text.strip().lower()
    if cleaned in ("", "all"):
        return None
    tokens = [tok for tok in re.split(r"[,\s]+", text.strip()) if tok]
    try:
        return [float(tok) for tok in tokens]
    except ValueError as exc:
        raise VmecInputError(
            WERROR_MESSAGES[INPUT_ERROR_FLAG],
            hint=f"--booz-surfaces expects 'all' or numbers, got {text!r}",
        ) from exc


# ---------------------------------------------------------------------------
# VMEC2000-format console output around the solve
# ---------------------------------------------------------------------------


def _preamble(case: str) -> str:
    """Run header block (vmec.f banner, structural match to xvmec2000)."""
    import platform

    now = time.localtime()
    date = time.strftime("%b %d,%Y", now)
    clock = time.strftime("%H:%M:%S", now)
    return (
        " - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n"
        "  SEQ =    1 TIME SLICE  0.0000E+00\n"
        f"  PROCESSING INPUT.{case}\n"
        f"  THIS IS VMEC_JAX, VERSION {_package_version()}\n"
        "  Lambda: Full Radial Mesh. L-Force: hybrid full/half.\n"
        "\n"
        f"  COMPUTER: {platform.node()}   OS: {platform.system()}   "
        f"RELEASE: {platform.release()}   DATE = {date}  TIME = {clock}\n"
    )


def _threed1_summary(wout) -> str:
    """Compact threed1-style equilibrium summary from the WoutData."""
    lines = [
        "",
        f" Aspect Ratio          = {float(wout.aspect):14.6f}",
        f" Plasma Volume         = {float(wout.volume_p):14.6f} [M**3]",
        f" Major Radius          = {float(wout.Rmajor_p):14.6f} [M]",
        f" Minor Radius          = {float(wout.Aminor_p):14.6f} [M]",
        f" Volume Average B      = {float(wout.volavgB):14.6f} [T]",
        f" beta total            = {float(wout.betatotal):14.6E}",
        f" MHD Energy (wb + wp)  = {float(wout.wb) + float(wout.wp):14.6E}",
        "",
    ]
    return "\n".join(lines)


def _timing_block(read_s: float, solve_s: float, wout_s: float) -> str:
    """IO/solver timing breakdown (fileout.f layout)."""
    return (
        f"    TIME TO INPUT/OUTPUT           {read_s + wout_s:12.2f}\n"
        f"       READ IN DATA                {read_s:12.2f}\n"
        f"       WRITE OUT DATA TO WOUT      {wout_s:12.2f}\n"
        f"    TIME IN SOLVER                 {solve_s:12.2f}\n"
    )


# ---------------------------------------------------------------------------
# Solve + wout
# ---------------------------------------------------------------------------


def _read_input(input_path: Path):
    """Parse the deck into a :class:`VmecInput` (typed error on failure)."""
    from .input import VmecInput

    try:
        return VmecInput.from_file(input_path)
    except VmecError:
        raise
    except Exception as exc:
        raise VmecInputError(
            WERROR_MESSAGES[INPUT_ERROR_FLAG],
            hint=f"{input_path.name}: {exc}",
        ) from exc


def _check_free_boundary(inp, input_path: Path, *, emit) -> None:
    """Free-boundary policy: mgrid fallback or redirect to the legacy CLI.

    VMEC2000 falls back to a fixed-boundary solve when the mgrid file cannot
    be read (a behavior VMEC++ dropped); decks with a readable mgrid need the
    free-boundary solver, which is not in the core yet.
    """
    if not bool(inp.lfreeb):
        return
    mgrid = Path(str(inp.mgrid_file)).expanduser()
    if not mgrid.is_absolute():
        mgrid = (input_path.parent / mgrid).resolve()
    if not mgrid.exists():
        emit(
            f" WARNING: mgrid file not found: {mgrid}\n"
            "          proceeding with a FIXED-BOUNDARY solve (VMEC2000 fallback)."
        )
        return
    raise VmecError(
        "free-boundary runs are not yet served by the vmec_jax core",
        hint=f"use the legacy entry point: python -m vmec_jax {input_path}",
        ier_flag=INPUT_ERROR_FLAG,
    )


def _stage_overrides(inp, *, ftol: float | None, max_iter: int | None):
    """Final-stage ftol/niter overrides for the multigrid ladder."""
    import numpy as np

    ftol_array = niter_array = None
    n_stages = int(np.atleast_1d(np.asarray(inp.ns_array)).size)
    if ftol is not None:
        ftol_array = np.resize(np.asarray(inp.ftol_array, dtype=float), n_stages)
        ftol_array[-1] = float(ftol)
    if max_iter is not None:
        niter_array = np.resize(np.asarray(inp.niter_array, dtype=np.int64), n_stages)
        niter_array[-1] = int(max_iter)
    return ftol_array, niter_array


def _legacy_run_objects(inp, input_path: Path, *, ns: int):
    """Bridge to the legacy (cfg, static, indata) trio the wout engine needs.

    The parity-proven wout post-processing engine consumes the legacy
    ``VMECStatic``/``InData`` containers; both are pure functions of the deck.
    JSON decks round-trip through :meth:`VmecInput.to_indata` (exact writer)
    so the legacy namelist reader sees identical values.  Temporary until the
    legacy wout engine internals move into the core (plan.md §5, sweep step).
    """
    import dataclasses
    import tempfile

    from ..config import load_config
    from ..static import build_static

    source = input_path
    tmp_name = None
    try:
        text = input_path.read_text()
    except Exception:
        text = ""
    if input_path.suffix.lower() == ".json" or text.lstrip()[:1] == "{":
        with tempfile.NamedTemporaryFile(
            "w", prefix="input.", suffix=".vmec_jax_indata", delete=False
        ) as handle:
            tmp_name = handle.name
        inp.to_indata(tmp_name)
        source = Path(tmp_name)
    try:
        cfg, indata = load_config(source)
    finally:
        if tmp_name is not None:
            Path(tmp_name).unlink(missing_ok=True)
    if int(cfg.ns) != int(ns):  # final ladder stage (runvmec.f skips decreasing ns)
        cfg = dataclasses.replace(cfg, ns=int(ns))
    # The wout engine synthesizes on VMEC's internal ntheta1/2/3 grid.
    from ..kernels.tomnsp import vmec_angle_grid

    grid = vmec_angle_grid(
        ntheta=int(cfg.ntheta), nzeta=int(cfg.nzeta),
        nfp=int(cfg.nfp), lasym=bool(cfg.lasym),
    )
    return cfg, build_static(cfg, grid=grid), indata


def _write_wout_from_result(inp, input_path: Path, result, wout_path: Path):
    """Build the full VMEC2000-compatible wout dataset and write it."""
    import numpy as np

    from ..state import StateLayout, VMECState
    from .wout import wout_from_state, write_wout

    ns, mnmax = (int(v) for v in np.shape(result.state.R_cos))
    cfg, static, indata = _legacy_run_objects(inp, input_path, ns=ns)
    layout = StateLayout(ns=ns, K=mnmax, lasym=bool(cfg.lasym))
    state = VMECState(
        layout=layout,
        Rcos=np.asarray(result.state.R_cos), Rsin=np.asarray(result.state.R_sin),
        Zcos=np.asarray(result.state.Z_cos), Zsin=np.asarray(result.state.Z_sin),
        Lcos=np.asarray(result.state.L_cos), Lsin=np.asarray(result.state.L_sin),
    )

    # fsqt history (wrout.f nstore_seq subsampling of fsqr + fsqz).
    history = np.asarray(result.fsq_history, dtype=float)
    fsqt = None
    if history.size:
        total = history[:, 0] + history[:, 1]
        stride = total.size // 100 + 1
        fsqt = total[stride - 1 :: stride][:100]

    wout = wout_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=int(indata.get_int("SIGNGS", -1)),
        fsqr=float(result.fsqr), fsqz=float(result.fsqz), fsql=float(result.fsql),
        fsqt=fsqt,
        niter=int(result.iterations),
        converged=bool(result.converged),
        input_extension=case_from_input(input_path),
        path=wout_path,
    )
    write_wout(wout_path, wout)
    return wout


def _solve_input_file(args, input_path: Path, outdir: Path | None, *, emit) -> int:
    """Full solve pipeline for one input deck (fixed-boundary core)."""
    from .multigrid import solve_multigrid

    case = case_from_input(input_path)
    verbose = not bool(args.quiet)

    t0 = time.perf_counter()
    inp = _read_input(input_path)
    read_s = time.perf_counter() - t0

    if verbose:
        emit(_preamble(case))
    _check_free_boundary(inp, input_path, emit=emit)

    ftol_array, niter_array = _stage_overrides(inp, ftol=args.ftol, max_iter=args.max_iter)

    t1 = time.perf_counter()
    result = solve_multigrid(
        inp,
        ftol_array=ftol_array,
        niter_array=niter_array,
        mode=str(args.mode),
        verbose=verbose,
        emit=emit,
    )
    solve_s = time.perf_counter() - t1

    wout_path = resolve_wout_path(input_path=input_path, outdir=outdir)
    wout_path.parent.mkdir(parents=True, exist_ok=True)
    t2 = time.perf_counter()
    wout = _write_wout_from_result(inp, input_path, result, wout_path)
    wout_s = time.perf_counter() - t2

    if verbose:
        from .printing import termination_summary

        emit(_threed1_summary(wout), end="")
        emit(
            termination_summary(
                int(result.ier_flag), case, int(result.jacobian_resets),
                time.perf_counter() - t0,
            ),
            end="",
        )
        emit(_timing_block(read_s, solve_s, wout_s), end="")
        emit(f"\n Wrote WOUT file: {wout_path}")

    plot_dir = outdir if outdir is not None else input_path.parent
    if args.plot is not None:
        _plot_wout_file(wout_path, plot_dir, emit=emit, quiet=bool(args.quiet))
    if bool(args.booz):
        _run_booz(
            wout_path, args, plot_dir,
            plot=args.plot is not None, emit=emit, quiet=bool(args.quiet),
        )
    return 0


# ---------------------------------------------------------------------------
# Plotting / Boozer drivers
# ---------------------------------------------------------------------------


def _plot_wout_file(wout_path: Path, outdir: Path, *, emit, quiet: bool) -> None:
    from .plotting import plot_wout

    if not quiet:
        emit(f" Plotting WOUT {wout_path.name} -> {outdir}/")
    for key, path in plot_wout(wout_path, outdir=outdir).items():
        if not quiet:
            emit(f"   Saved {key}: {path}")


def _plot_boozmn_file(boozmn_path: Path, outdir: Path, *, emit, quiet: bool) -> None:
    from .plotting import plot_boozmn

    if not quiet:
        emit(f" Plotting Boozer output {boozmn_path.name} -> {outdir}/")
    for key, path in plot_boozmn(boozmn_path, outdir=outdir).items():
        if not quiet:
            emit(f"   Saved {key}: {path}")


def _run_booz(wout_path: Path, args, outdir: Path, *, plot: bool, emit, quiet: bool) -> Path:
    from .boozer import resolve_boozmn_path, run_booz_xform

    surfaces = _parse_booz_surfaces(args.booz_surfaces)
    boozmn_path = resolve_boozmn_path(wout_path, outdir=outdir)
    if not quiet:
        emit(
            f" Running booz_xform_jax (mbooz={int(args.mbooz)}, nbooz={int(args.nbooz)}, "
            f"surfaces={'all' if surfaces is None else surfaces})"
        )
    boozmn_path = run_booz_xform(
        wout_path,
        mbooz=int(args.mbooz),
        nbooz=int(args.nbooz),
        surfaces=surfaces,
        output_path=boozmn_path,
        verbose=not quiet,
    )
    if not quiet:
        emit(f" Wrote Boozer output: {boozmn_path}")
    if plot:
        _plot_boozmn_file(boozmn_path, outdir, emit=emit, quiet=quiet)
    return boozmn_path


# ---------------------------------------------------------------------------
# Bundled quick-start case (--test)
# ---------------------------------------------------------------------------


def _copy_test_input(outdir: Path) -> Path:
    """Copy the packaged quick-start deck into ``outdir`` (quick-start ftol)."""
    outdir.mkdir(parents=True, exist_ok=True)
    dst = outdir / _TEST_INPUT_NAME
    resource = resources.files("vmec_jax").joinpath("resources", _TEST_INPUT_NAME)
    with resources.as_file(resource) as src:
        shutil.copyfile(src, dst)
    text = dst.read_text()
    replacement = f"FTOL_ARRAY  = {_TEST_FTOL:.0e}"
    new_text, count = re.subn(r"(?im)^\s*FTOL_ARRAY\s*=.*$", f"  {replacement}", text, count=1)
    if count:
        dst.write_text(new_text)
    return dst


def _run_bundled_test(args, parser: argparse.ArgumentParser, *, emit) -> int:
    """Solve + wout + plots for the packaged quick-start case."""
    if args.input is not None:
        parser.error("--test does not take an input path")
    if args.plot is not None:
        parser.error("--test and --plot are mutually exclusive")

    outdir = (
        Path(args.outdir).expanduser().resolve()
        if args.outdir
        else Path.cwd().resolve() / "vmec_jax_test"
    )
    input_path = _copy_test_input(outdir)
    wout_path = resolve_wout_path(input_path=input_path, outdir=outdir)
    plot_dir = outdir / "figures"
    quiet = bool(args.quiet)

    if not quiet:
        emit("vmec bundled test")
        emit("-----------------")
        emit("Quick-start run of the packaged fixed-boundary QH warm-start input.")
        emit(f"1. Wrote the VMEC input file to: {input_path}")
        emit("2. Solving the equilibrium and writing the WOUT file.")
        emit("")
        emit("Equivalent manual command:")
        emit(f"  vmec {input_path} --outdir {outdir}")
        emit("")

    rc = _solve_input_file(args, input_path, outdir, emit=emit)
    if rc != 0:  # pragma: no cover - _solve_input_file raises instead
        return rc

    if not quiet:
        emit("")
        emit(f"3. Wrote the WOUT file to: {wout_path}")
        emit(f"4. Plotting the WOUT file into: {plot_dir}")
        emit("")
        emit("Equivalent manual plotting command:")
        emit(f"  vmec --plot {wout_path} --outdir {plot_dir}")
        emit("")
    plot_dir.mkdir(parents=True, exist_ok=True)
    _plot_wout_file(wout_path, plot_dir, emit=emit, quiet=quiet)
    if not quiet:
        emit("")
        emit("Bundled test complete: inspect the input, WOUT file, and figures above.")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _dispatch(args, parser: argparse.ArgumentParser, *, emit) -> int:
    if bool(args.doctor):
        if args.input is not None:
            parser.error("--doctor does not take an input path")
        from ..doctor import main as doctor_main

        return int(doctor_main())

    if bool(args.test):
        return _run_bundled_test(args, parser, emit=emit)

    plot_requested = args.plot is not None
    target_arg = args.input
    if args.plot not in (None, _PLOT_AUTO):
        if args.input is not None:
            parser.error("provide the target either as --plot PATH or as a positional input, not both")
        target_arg = args.plot
    if args.plot == _PLOT_AUTO and target_arg is None:
        parser.error("--plot requires a PATH or a positional input")
    if target_arg is None:
        parser.error("provide a VMEC input file, a wout_*.nc/boozmn_*.nc file, or --test/--doctor")

    input_path = Path(target_arg).expanduser().resolve()
    if not input_path.exists():
        raise VmecInputError(
            WERROR_MESSAGES[INPUT_ERROR_FLAG],
            hint=f"input file not found: {input_path}",
        )

    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else None
    plot_outdir = outdir if outdir is not None else input_path.parent
    quiet = bool(args.quiet)

    if _is_boozmn_path(input_path):
        if not plot_requested:
            parser.error("boozmn_*.nc inputs are plot-only; use --plot boozmn_*.nc")
        _plot_boozmn_file(input_path, plot_outdir, emit=emit, quiet=quiet)
        return 0

    if _is_wout_path(input_path):
        if not plot_requested and not bool(args.booz):
            parser.error("wout_*.nc inputs require --plot and/or --booz")
        if plot_requested:
            _plot_wout_file(input_path, plot_outdir, emit=emit, quiet=quiet)
        if bool(args.booz):
            _run_booz(input_path, args, plot_outdir, plot=plot_requested, emit=emit, quiet=quiet)
        return 0

    return _solve_input_file(args, input_path, outdir, emit=emit)


def main(argv: list[str] | None = None) -> int:
    """Run the ``vmec`` command-line entry point (zero-crash)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    emit = print

    try:
        return _dispatch(args, parser, emit=emit)
    except VmecError as err:
        emit(f"\n {err.message}\n")
        if err.hint:
            emit(f" HINT : {err.hint}")
        code = int(err.ier_flag)
        return code if code != 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
