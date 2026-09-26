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

import parameters as P

INPUT = common.DATA / "input.rotating_ellipse"
COILS = common.DATA / "coils.fitted.json"  # --coils skips fitting; --initial-coils chooses the fitting seed
MAKE_MOVIE = True  # set True for a compact GIF of accepted iterates
# Surface colors: None, "absB", or "B.n/B".
MOVIE_SURFACE_COLOR = "absB"

QA_SURFACES = tuple(i / 10 for i in range(1, 11))
MAX_MODE = P.RESOLUTION[0]  # boundary optimization modes; distinct from equilibrium resolution
ACCEPTED_STEPS = P.ACCEPTED_STEPS
COIL_FIT_MAXITER = 200  # preliminary coil-only fit on the frozen seed boundary
ASPECT_TARGET = P.ASPECT_TARGET
ASPECT_WEIGHT = P.ASPECT_WEIGHT
IOTA_FLOOR = P.IOTA_FLOOR  # matches the current qa_optimization.py
IOTA_WEIGHT = P.IOTA_WEIGHT
VARY_MAJOR_RADIUS = False  # set True to optimize RBC(0,0) instead of fixing it
PARAMETER_STEP = 0.1
COIL_STEP = P.COIL_STEP
ESS_ALPHA = 1.2

N_COILS = P.N_COILS
COIL_ORDER = P.COIL_ORDER
N_SEGMENTS = P.N_SEGMENTS

NORMAL_FIELD_WEIGHT = 1.0e3
NORMAL_FIELD_LIMIT = 0.01
NORMAL_FIELD_OBJECTIVE_LIMIT = 0.008  # margin for the independent final grid
NORMAL_FIELD_LIMIT_WEIGHT = 2.0e5
CURVATURE_LIMIT = P.CURVATURE_LIMIT
COIL_DISTANCE_LIMIT = P.COIL_DISTANCE_LIMIT
COIL_SURFACE_DISTANCE_LIMIT = P.COIL_SURFACE_DISTANCE_LIMIT

# A toroidal grid commensurate with the coil count can alias narrow B.n/B structure.
NPHI, NTHETA = 37, 32
METHOD = "SLSQP"  # enforces every nonlinear coil and plasma inequality
PARAMETER_BOUND = 5.0  # used only when METHOD is L-BFGS-B
RESOLUTION = P.RESOLUTION  # MPOL, NTOR, NS; explicit --input preserves its deck
GRID = P.GRID  # NTHETA, NZETA
EQUILIBRIUM_FTOL = P.EQUILIBRIUM_FTOL
OPTIMIZER_FTOL, OPTIMIZER_GTOL = P.OPTIMIZER_FTOL, 1e-10
RADIUS_TARGET, RADIUS_TOLERANCE = P.RADIUS_TARGET, P.RADIUS_TOLERANCE
IOTA_MARGIN, RADIUS_MARGIN = P.IOTA_MARGIN, P.RADIUS_MARGIN
VERIFY_NS, VERIFY_FTOL, VERIFY_MAXITER = P.VERIFY_NS, P.VERIFY_FTOL, 8000


DISTANCE_MARGIN = P.DISTANCE_MARGIN

def parse_args(argv=None):
    if METHOD != "SLSQP":
        raise ValueError("coil constraints require SLSQP")
    return common.parse_options(argv, parameters=globals(), description=__doc__, formulation="fixed")


def build_problem(args):
    """Prepare the objective and constraints without joint optimization."""
    import _coil_constraints as coil_limits
    return fixed.build_problem(args, settings=SimpleNamespace(**globals()), coil_limits=coil_limits)


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
