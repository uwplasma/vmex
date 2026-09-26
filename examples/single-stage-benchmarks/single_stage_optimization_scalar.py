#!/usr/bin/env python
"""Fixed-boundary single-stage optimization: edit physics and optimizer settings below.

The shared driver owns coil fitting, accepted-state bookkeeping and output.
Run verify_single_stage.py separately for derivative qualification.
"""
from pathlib import Path
import sys
from types import SimpleNamespace
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[1])]
from single_stage_support import common, fixed

INPUT = common.DATA / "input.rotating_ellipse"
COILS = None  # --coils skips fitting; --initial-coils chooses the fitting seed
MAKE_MOVIE = True  # set True for a compact GIF of accepted iterates
# Surface colors: None, "absB", or "B.n/B".
MOVIE_SURFACE_COLOR = "absB"

QA_SURFACES = tuple(i / 10 for i in range(1, 11))
MAX_MODE = 3  # boundary optimization modes; distinct from equilibrium resolution
ACCEPTED_STEPS = 100
COIL_FIT_MAXITER = 200  # preliminary coil-only fit on the frozen seed boundary
ASPECT_TARGET = 5.0
ASPECT_WEIGHT = 1.0
IOTA_FLOOR = 0.19  # matches the current qa_optimization.py
IOTA_WEIGHT = 10.0
VARY_MAJOR_RADIUS = False  # set True to optimize RBC(0,0) instead of fixing it
PARAMETER_STEP = 0.1
COIL_STEP = 0.05
ESS_ALPHA = 1.2

N_COILS = 3
COIL_ORDER = 5
COIL_MAJOR_RADIUS = 1.0
COIL_MINOR_RADIUS = 0.5
COIL_CURRENT = 2.7e5
N_SEGMENTS = 64

NORMAL_FIELD_WEIGHT = 1.0e3
NORMAL_FIELD_LIMIT = 0.01
NORMAL_FIELD_OBJECTIVE_LIMIT = 0.008  # margin for the independent final grid
NORMAL_FIELD_LIMIT_WEIGHT = 2.0e5
LENGTH_TARGET = 5
LENGTH_WEIGHT = 1.0
CURVATURE_LIMIT = 7.0
CURVATURE_OBJECTIVE_LIMIT = 6.9  # margin for the independent final grid
CURVATURE_WEIGHT = 10.0
COIL_DISTANCE_LIMIT = 0.15
COIL_DISTANCE_WEIGHT = 1.0e3
COIL_SURFACE_DISTANCE_LIMIT = 0.20
COIL_SURFACE_DISTANCE_WEIGHT = 1.0e3

# A toroidal grid commensurate with the coil count can alias narrow B.n/B structure.
NPHI, NTHETA = 37, 32
METHOD = "SLSQP"  # also accepts "BFGS" or "L-BFGS-B"
PARAMETER_BOUND = 5.0  # used only when METHOD is L-BFGS-B
RESOLUTION = (8, 8, 51)  # MPOL, NTOR, NS; explicit --input preserves its deck
GRID = (64, 64)  # NTHETA, NZETA
EQUILIBRIUM_FTOL = 1e-15
OPTIMIZER_FTOL, OPTIMIZER_GTOL = 1e-10, 1e-10
RADIUS_TARGET, RADIUS_TOLERANCE = 1.0, 0.01
IOTA_MARGIN, RADIUS_MARGIN = 0.0005, 0.001
VERIFY_NS, VERIFY_FTOL, VERIFY_MAXITER = 201, 1e-15, 12000


def parse_args(argv=None):
    args = common.parse_options(argv, parameters=globals(), description=__doc__, formulation="fixed")
    args.constrained = METHOD == "SLSQP"
    return args


def build_problem(args):
    """Prepare the objective and constraints without joint optimization."""
    return fixed.build_problem(args, settings=SimpleNamespace(**globals()))


def run_optimizer(stage, args):
    """Use the same VMEX optimizer and acceptance policy as the free arm."""
    from vmex import optimize as opt
    return opt.minimize(stage.joint_problem, method=METHOD,
        bounds=[(-PARAMETER_BOUND, PARAMETER_BOUND)] * stage.x0.size if METHOD == "L-BFGS-B" else None,
        constraints=stage.constraints, callback=stage.record_step, options=stage.options)


def verify_endpoint(stage, args, result):
    return fixed.verify_endpoint(stage, args, result)


def run(args):
    """Build, optimize, then independently solve the final configuration."""
    stage = build_problem(args)
    result = run_optimizer(stage, args)
    return verify_endpoint(stage, args, result)


def main(argv=None):
    return common.run_fixed(parse_args(argv), settings=SimpleNamespace(**globals()),
        build_problem=build_problem, run_optimizer=run_optimizer, verify_endpoint=verify_endpoint)


if __name__ == "__main__":
    raise SystemExit(main())
