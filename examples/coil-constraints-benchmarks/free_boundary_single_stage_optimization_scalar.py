#!/usr/bin/env python
"""Scalar free-boundary single-stage optimization through the public VMEX API.

Edit parameters.py, then follow build_problem -> run_optimizer ->
verify_endpoint -> postprocess. Like the fixed-boundary examples, the loss is
configured below; opt.minimize owns scaling and optimizer acceptance. VMEX owns
solves, total derivatives and prediction from the last accepted equilibrium.
Derivative tests live in verify_free_boundary_single_stage.py and run separately.
Use --qualification for a matching report, or --seed for a prepared step-zero
checkpoint. Neither repeats qualification; --dry-run only prints settings.
This benchmark is vacuum-only.
"""

from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[1])]
from single_stage_support import common, free
import parameters as P
COIL_CONSTRAINTS = True
DISTANCE_MARGIN = P.DISTANCE_MARGIN
INPUT = common.DATA / "input.rotating_ellipse"
COILS = common.DATA / "coils.fitted.json"
COIL_FIT_MAXITER = 200
COIL_MAJOR_RADIUS, COIL_MINOR_RADIUS, COIL_CURRENT = 1.0, 0.5, 2.7e5
NORMAL_FIELD_WEIGHT, NORMAL_FIELD_OBJECTIVE_LIMIT, NORMAL_FIELD_LIMIT_WEIGHT = 1e3, 0.008, 2e5

# Optimization and equilibrium accuracy (different stopping criteria).
ACCEPTED_STEPS = P.ACCEPTED_STEPS
OPTIMIZER_FTOL = P.OPTIMIZER_FTOL
COIL_STEP = P.COIL_STEP
MAX_TRIALS = 500
RESOLUTION = P.RESOLUTION
GRID = P.GRID
EQUILIBRIUM_FTOL = P.EQUILIBRIUM_FTOL
ROOT_TOLERANCE = 2e-6
ROOT_POLISH_TOLERANCE = 1e-12
INITIALIZATION_SECONDS = 3600
OPTIMIZATION_SECONDS = 43200
VERIFICATION_SECONDS = 1800

# Shared plasma objective; coil geometry uses hard inequalities.
QA_SURFACES = tuple(i / 10 for i in range(1, 11))
ASPECT_TARGET, ASPECT_WEIGHT = P.ASPECT_TARGET, P.ASPECT_WEIGHT
IOTA_FLOOR, IOTA_WEIGHT = P.IOTA_FLOOR, P.IOTA_WEIGHT
RADIUS_TARGET, RADIUS_TOLERANCE = P.RADIUS_TARGET, P.RADIUS_TOLERANCE
IOTA_MARGIN, RADIUS_MARGIN = P.IOTA_MARGIN, P.RADIUS_MARGIN
N_COILS, COIL_ORDER, N_SEGMENTS = P.N_COILS, P.COIL_ORDER, P.N_SEGMENTS
NPHI, NTHETA = 37, 32
LENGTH_TARGET = P.LENGTH_LIMIT
CURVATURE_LIMIT = P.CURVATURE_LIMIT
COIL_DISTANCE_LIMIT = P.COIL_DISTANCE_LIMIT
COIL_SURFACE_DISTANCE_LIMIT = P.COIL_SURFACE_DISTANCE_LIMIT
# Free-boundary equilibrium determines B.n; it is a diagnostic, not a penalty.
NORMAL_FIELD_LIMIT = 0.01

# Checked matrix-free adjoints. VMEX shares a root-recovery LU with the trial's
# adjoints and promotes it only on acceptance; the dense fallback stays enabled.
ADJOINT_RESIDUAL_RTOL = 1e-9
ADJOINT_BATCH_SIZE, ADJOINT_MAX_DOFS = 32, 20000
MATRIXFREE_RTOL = 1e-11
MATRIXFREE_RESTART, MATRIXFREE_MAX_CYCLES, MATRIXFREE_RHS_BATCH_SIZE = 100, 3, 3
LU_REFRESH_HORIZON = 10  # rebuild when estimated savings repay its measured cost
VERIFY_NS, VERIFY_FTOL, VERIFY_MAXITER = P.VERIFY_NS, P.VERIFY_FTOL, 12000

# Match the scalar reference: figures, accepted-iterate movie and WOUT plots.
MAKE_MOVIE = True
MOVIE_SURFACE_COLOR = "absB"  # total equilibrium field; alternatively "B.n/B"
POSTPROCESSING_SECONDS = 1800


def parse_args(argv=None):
    return common.parse_options(argv, parameters=globals(), description=__doc__, formulation="free")

# Shared implementation keeps file handling, root setup and restart checks out of this example.
sha, write_json, setup_run = common.sha, free.write_json, free.setup_run

def qualification_contract(args):
    return free.qualification_contract(args, case=SimpleNamespace(**globals()))

def read_qualification(args):
    return free.read_qualification(args, case=SimpleNamespace(**globals()))

def build_problem(args, *, event=None, qualified=None):
    """Build the weighted QA/aspect/iota objective and the selected coil constraints."""
    import _coil_constraints as coil_limits
    return free.build_problem(args, case=SimpleNamespace(**globals()),
        event=event, qualified=qualified, coil_limits=coil_limits)

def configure_solver(problem, args):
    return free.configure_solver(problem, args, case=SimpleNamespace(**globals()))

def run_optimizer(stage, args, record_step, *, method):
    """Choose the optimizer and physical bounds; VMEX owns scaling and acceptance."""
    import numpy as np
    from vmex import optimize as opt

    if method != "SLSQP":
        raise ValueError("this benchmark requires explicit nonlinear constraints and SLSQP")
    problem = stage.problem
    width = RADIUS_TOLERANCE - RADIUS_MARGIN
    constraints = [problem.nonlinear_constraint(
        [IOTA_FLOOR + IOTA_MARGIN, RADIUS_TARGET - width,
         P.COIL_SURFACE_DISTANCE_LIMIT + P.DISTANCE_MARGIN],
        [np.inf, RADIUS_TARGET + width, np.inf],
        scales=[IOTA_FLOOR, RADIUS_TOLERANCE, P.COIL_SURFACE_DISTANCE_LIMIT]),
        stage.coil_constraint]
    return opt.minimize(problem, method=method, constraints=constraints,
        callback=lambda x: record_step(),
        options=dict(maxiter=args.accepted_steps, ftol=OPTIMIZER_FTOL))

def verify_endpoint(stage, args, summary, history, initial_equilibrium):
    import _coil_constraints as coil_limits
    return free.verify_endpoint(SimpleNamespace(**globals()), coil_limits, stage, args, summary, history, initial_equilibrium)


def postprocess(stage, args, summary, monitor, history, initial_equilibrium, verified):
    return free.postprocess(SimpleNamespace(**globals()), stage, args, summary, monitor, history, initial_equilibrium, verified)


def main(argv=None, *, method="SLSQP"):
    """Prepare the case, optimize, and independently solve the endpoint."""
    if method != "SLSQP":
        raise ValueError("coil constraints require SLSQP")
    return free.run(parse_args(argv), case=SimpleNamespace(**globals()), method=method)


if __name__ == "__main__":
    raise SystemExit(main())
