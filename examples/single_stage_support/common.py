"""Shared file and command-line interface for the two scalar examples.

Importing this module does not initialize JAX, create files, or run a solve.
The examples supply physics/optimizer settings; public VMEX APIs own solvers.
"""

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time


DATA = Path(__file__).resolve().parent / "data"


def parse_options(argv, *, parameters, description, formulation):
    p = parameters
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--wout", type=Path, help="WOUT restart; requires its corresponding input deck")
    if formulation == "free":
        start = parser.add_mutually_exclusive_group()
        start.add_argument("--qualification", type=Path, help="passing report for this code and case")
        start.add_argument("--seed", type=Path, help="authenticated step-zero seed manifest; no derivative qualification claim")
    else:
        parser.set_defaults(qualification=None, seed=None)
    coils = parser.add_mutually_exclusive_group()
    coils.add_argument("--coils", type=Path, help="reuse fitted coils without stage two")
    coils.add_argument("--initial-coils", type=Path, help="fit these coils instead of generating circles")
    parser.add_argument("--coil-fit-maxiter", type=int, default=p["COIL_FIT_MAXITER"])
    parser.add_argument("--output", type=Path, default=p["HERE"] / "runs" / f"{formulation}-boundary-{time.time_ns()}")
    parser.add_argument("--device", choices=("cpu", "gpu"))
    parser.add_argument("--accepted-steps", "--maxiter", type=int, default=p["ACCEPTED_STEPS"],
                        help="step/iteration budget; optimization can stop earlier")
    parser.add_argument("--ftol", type=float, help="equilibrium force tolerance")
    parser.add_argument("--resolution", type=int, nargs=3, metavar=("MPOL", "NTOR", "NS"))
    parser.add_argument("--grid", type=int, nargs=2, metavar=("NTHETA", "NZETA"))
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plots", dest="no_plots", action="store_false")
    parser.add_argument("--movie", action=argparse.BooleanOptionalAction, default=p["MAKE_MOVIE"])
    parser.add_argument("--dry-run", action="store_true")
    # Compatibility with older fixed-boundary command lines; SLSQP is now default.
    parser.add_argument("--constrained", action="store_true", default=True, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    custom_input = args.input is not None
    bundle = args.qualification or args.seed
    if bundle is not None:
        report = json.loads(bundle.read_text())
        for name in ("input", "wout", "resolution", "grid", "ftol", "device"):
            if getattr(args, name) is None:
                value = report["configuration"][name]
                setattr(args, name, Path(value) if name in ("input", "wout") and value else value)
        if args.initial_coils is not None:
            parser.error("a saved seed reuses fitted coils; --initial-coils requires a fresh start")
    elif args.wout is not None and args.input is None:
        parser.error("--wout requires --input to define the pressure/current profiles and solver settings")
    if args.input is None:
        args.input = p["INPUT"]
    if not custom_input and args.wout is None and bundle is None:
        args.resolution = p["RESOLUTION"] if args.resolution is None else args.resolution
        args.grid = p["GRID"] if args.grid is None else args.grid
    args.ftol = p["EQUILIBRIUM_FTOL"] if args.ftol is None else args.ftol
    args.device = args.device or "gpu"
    if args.initial_coils is not None:
        args.coils = None  # Explicit initial coils override the editable COILS default.
    elif args.coils is None and bundle is None:
        args.coils = p["COILS"]
    if args.coil_fit_maxiter < 1:
        parser.error("positive coil-fit iteration budget required")
    if args.accepted_steps < 1 or not math.isfinite(args.ftol) or args.ftol <= 0:
        parser.error("positive step budget and finite positive force tolerance required")
    if ((args.resolution is not None and (args.resolution[0] < 1 or args.resolution[1] < 0 or args.resolution[2] < 3))
            or (args.grid is not None and min(args.grid) < 4)):
        parser.error("invalid equilibrium resolution or grid")
    if not 0 <= p["RADIUS_MARGIN"] < p["RADIUS_TOLERANCE"] < p["RADIUS_TARGET"] or p["IOTA_MARGIN"] < 0:
        parser.error("invalid physical constraint margins")
    if p["MOVIE_SURFACE_COLOR"] not in ("absB", "B.n/B", None):
        parser.error("movie surface color must be absB, B.n/B or None")
    # Resolve before the fixed driver enters its output directory.
    for name in ("input", "wout", "coils", "initial_coils", "qualification", "seed", "output"):
        if getattr(args, name) is not None:
            setattr(args, name, Path(getattr(args, name)).resolve())
    return args


def read_seed(path, *, contract):
    """Authenticate a seed's case and artifacts without claiming derivative checks.

    Conversion from an older checkpoint is an explicit preparation step. The
    public FreeBoundaryProblem API freshly certifies the stored equilibrium.
    """
    path = Path(path).resolve()
    report = json.loads(path.read_text())
    if report.get("schema") != "vmex.single-stage-seed/v1" or report.get("accepted_step") != 0:
        raise ValueError("a step-zero seed manifest is required")
    if report.get("contract") != contract:
        raise ValueError("seed differs from current code, input, physics, resolution or runtime")
    assets = {}
    for name in ("coils", "checkpoint"):
        item = report["artifacts"][name]
        filename = item["file"]
        if not filename or Path(filename).name != filename or filename in (".", ".."):
            raise ValueError("seed artifacts must be files beside the manifest")
        asset = path.parent / filename
        if asset.resolve().parent != path.parent:
            raise ValueError("seed artifacts must remain beside the manifest")
        if hashlib.sha256(asset.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"seed {name} hash mismatch")
        assets[name] = asset
    return report, assets


def load_input(args, *, restore_state=True):
    """Read input/WOUT through public VMEX APIs, preserving explicit deck grids."""
    import numpy as np
    import vmex as vj
    from vmex import optimize as opt

    inp = vj.VmecInput.from_file(args.input)
    has_pressure = any(values is not None and np.any(np.asarray(values) != 0)
                       for values in (inp.am, inp.am_aux_f))
    if (inp.pres_scale != 0 and has_pressure) or inp.curtor != 0:
        raise ValueError("these vacuum examples require zero pressure and plasma current; "
                         "finite beta needs plasma-aware coil fitting and interface diagnostics")
    mpol, ntor, ns = args.resolution or (inp.mpol, inp.ntor, int(inp.ns_array[-1]))
    ntheta, nzeta = args.grid or (inp.ntheta, inp.nzeta)
    inp = inp.change_resolution(mpol=mpol, ntor=ntor, ntheta=ntheta, nzeta=nzeta)
    inp = replace(inp, ns_array=np.array([ns]), ftol_array=np.array([args.ftol]), lfreeb=False)
    if inp.lasym:
        raise ValueError("these examples require stellarator symmetry")
    seed = None
    if args.wout is not None:
        wout = vj.read_wout(args.wout)
        if int(wout.nfp) != inp.nfp or bool(wout.lasym):
            raise ValueError("WOUT symmetry differs from the input")
        rbc, zbs, rbs, zbc = opt.boundary_from_wout(wout, mpol=mpol, ntor=ntor)
        inp = replace(inp, rbc=rbc, zbs=zbs, rbs=rbs, zbc=zbc)
        if restore_state:
            seed = vj.state_from_wout(wout, inp=inp, ns=ns)
    return inp, seed


def initial_coils(inp, path, *, parameters, resize=False):
    """Load ESSOS coils, or generate the same circular seed in either example."""
    import jax.numpy as jnp
    from essos.coils import Coils, CreateEquallySpacedCurves

    p = parameters
    if path is not None:
        coils = Coils.from_json(str(path))
        if resize:
            coils = resize_coils(coils, p["COIL_ORDER"], p["N_SEGMENTS"])
    else:
        curves = CreateEquallySpacedCurves(p["N_COILS"], p["COIL_ORDER"], p["COIL_MAJOR_RADIUS"],
            p["COIL_MINOR_RADIUS"], n_segments=p["N_SEGMENTS"], nfp=inp.nfp, stellsym=True)
        coils = Coils(curves, jnp.full(p["N_COILS"], p["COIL_CURRENT"]))
    if (coils.nfp != inp.nfp or not coils.stellsym or coils.n_segments != p["N_SEGMENTS"]
            or coils.dofs_curves.shape != (p["N_COILS"], 3, 2*p["COIL_ORDER"]+1)):
        raise ValueError("coils must match symmetry, count, Fourier order and quadrature")
    return coils


def physical_constraint(problem, *, parameters):
    """Use the same public iota/radius bounds and conditioning in both arms."""
    p = parameters
    width = p["RADIUS_TOLERANCE"] - p["RADIUS_MARGIN"]
    return problem.nonlinear_constraint(
        [p["IOTA_FLOOR"] + p["IOTA_MARGIN"], p["RADIUS_TARGET"] - width],
        [math.inf, p["RADIUS_TARGET"] + width],
        scales=[p["IOTA_FLOOR"], p["RADIUS_TOLERANCE"]])


def setup_run(args):
    """Create a fresh output directory and keep every runtime cache inside it."""
    import os
    args.output.mkdir(parents=True, exist_ok=False)
    for key in ("TMPDIR", "XDG_CACHE_HOME", "MPLCONFIGDIR", "CUDA_CACHE_PATH"):
        path = args.output / "cache" / key
        path.mkdir(parents=True)
        os.environ[key] = str(path)
    os.environ.update(JAX_ENABLE_X64="1", JAX_PLATFORMS="cuda" if args.device == "gpu" else "cpu",
                      VMEX_COMPILATION_CACHE="disabled", JAX_ENABLE_COMPILATION_CACHE="false", MPLBACKEND="Agg")
    return args.output


def run_fixed(args, *, settings, build_problem, run_optimizer, verify_endpoint):
    """One production pipeline; qualification is a separate command."""
    import os
    args.constrained = settings.METHOD == "SLSQP"
    if args.dry_run:
        parameters = {k: v for k, v in vars(settings).items() if k.isupper() and k not in ("HERE", "P")}
        print(json.dumps(dict(optimizer=settings.METHOD, parameters=parameters,
                             arguments=vars(args)), indent=2, default=str))
        return 0
    setup_run(args)
    previous_directory = Path.cwd()
    try:
        os.chdir(args.output)
        stage = build_problem(args)
        result = run_optimizer(stage, args)
        return verify_endpoint(stage, args, result)
    finally:
        os.chdir(previous_directory)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resize_coils(coils, order, n_segments):
    """Zero-pad higher modes, preserve scaling, and choose field quadrature.

    Refuse truncation: lowering order requires a separate, explicitly fitted seed.
    """
    import jax.numpy as jnp
    from essos.coils import Coils, Curves

    if order < coils.order:
        raise ValueError(f"cannot truncate order-{coils.order} seed to order {order}")
    if order == coils.order and n_segments == coils.n_segments:
        return coils
    old = coils.curves
    raw = old.dofs / old.scaling
    raw = jnp.pad(raw, ((0, 0), (0, 0), (0, 2*(order-coils.order))))
    curves = Curves(raw, n_segments=n_segments, nfp=coils.nfp, stellsym=coils.stellsym,
                    scaling_type=old.scaling_type, scaling_factor=old.scaling_factor,
                    scale_fixed=old.scale_fixed)
    return Coils(curves, coils.dofs_currents_raw, currents_scale=coils.currents_scale)
