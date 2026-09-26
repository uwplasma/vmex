#!/usr/bin/env python
"""Scalar free-boundary single-stage optimization through the public VMEX API.

Edit the parameters below, then follow build_problem -> run_optimizer ->
verify_endpoint -> postprocess. Like the fixed-boundary examples, the loss is
configured below; opt.minimize owns scaling and optimizer acceptance. VMEX owns
solves, total derivatives and prediction from the last accepted equilibrium.
Derivative tests live in verify_free_boundary_single_stage.py and run separately.
Use --qualification for a matching report, or --seed for a prepared step-zero
checkpoint. Neither repeats qualification; --dry-run only prints settings.
This example is vacuum-only; finite-beta API requirements are in the README.
"""

from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
# Also support importlib-based tooling without initializing VMEX.
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[1])]
from single_stage_support import common, free

INPUT = common.DATA / "input.rotating_ellipse"
COILS = None  # supply --coils to reuse a fitted set; otherwise perform stage two
COIL_FIT_MAXITER = 200
COIL_MAJOR_RADIUS, COIL_MINOR_RADIUS, COIL_CURRENT = 1.0, 0.5, 2.7e5
NORMAL_FIELD_WEIGHT, NORMAL_FIELD_OBJECTIVE_LIMIT, NORMAL_FIELD_LIMIT_WEIGHT = 1e3, 0.008, 2e5

# Optimization and equilibrium accuracy (different stopping criteria).
ACCEPTED_STEPS = 100
OPTIMIZER_FTOL = 1e-10
OPTIMIZER_GTOL = 1e-10
PARAMETER_BOUND = 5.0  # L-BFGS-B bounds in scaled coil coordinates
LBFGSB_MAXCOR, LBFGSB_MAXLS = 20, 20
COIL_STEP = 0.05
MAX_TRIALS = 500
RESOLUTION = (8, 8, 51)  # MPOL, NTOR, NS
GRID = (64, 64)  # NTHETA, NZETA
EQUILIBRIUM_FTOL = 1e-15
ROOT_TOLERANCE = 2e-6  # ordinary-root admission before coupled Newton refinement
ROOT_POLISH_TOLERANCE = 1e-12
INITIALIZATION_SECONDS = 3600
OPTIMIZATION_SECONDS = 43200
VERIFICATION_SECONDS = 1800

# Same QA, aspect, iota and coil penalties as the scalar reference.
QA_SURFACES = tuple(i / 10 for i in range(1, 11))
ASPECT_TARGET, ASPECT_WEIGHT = 5.0, 1.0
IOTA_FLOOR, IOTA_WEIGHT = 0.19, 10.0
RADIUS_TARGET, RADIUS_TOLERANCE = 1.0, 0.01
IOTA_MARGIN, RADIUS_MARGIN = 0.0005, 0.001
N_COILS, COIL_ORDER, N_SEGMENTS = 3, 5, 64
NPHI, NTHETA = 37, 32
LENGTH_TARGET, LENGTH_WEIGHT = 5.0, 1.0
CURVATURE_OBJECTIVE_LIMIT, CURVATURE_LIMIT, CURVATURE_WEIGHT = 6.9, 7.0, 10.0
COIL_DISTANCE_LIMIT, COIL_DISTANCE_WEIGHT = 0.15, 1e3
COIL_SURFACE_DISTANCE_LIMIT, COIL_SURFACE_DISTANCE_WEIGHT = 0.20, 1e3
# Free-boundary equilibrium determines B.n; it is a diagnostic, not a penalty.
NORMAL_FIELD_LIMIT = 0.01

# Checked matrix-free adjoints; dense recovery refreshes only on acceptance.
ADJOINT_RESIDUAL_RTOL = 1e-9
ADJOINT_BATCH_SIZE, ADJOINT_MAX_DOFS = 32, 20000
MATRIXFREE_RTOL = 1e-11
MATRIXFREE_RESTART, MATRIXFREE_MAX_CYCLES, MATRIXFREE_RHS_BATCH_SIZE = 100, 3, 3
LU_REFRESH_HORIZON = 10  # rebuild when estimated savings repay its measured cost
VERIFY_NS, VERIFY_FTOL, VERIFY_MAXITER = 201, 1e-15, 12000

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
    return free.build_problem(args, case=SimpleNamespace(**globals()),
        event=event, qualified=qualified)

def configure_solver(problem, args):
    return free.configure_solver(problem, args, case=SimpleNamespace(**globals()))

def run_optimizer(stage, args, record_step, *, method):
    """Choose the optimizer and physical bounds; VMEX owns scaling and acceptance."""
    from scipy.optimize import Bounds
    from vmex import optimize as opt

    problem = stage.problem
    options = dict(maxiter=args.accepted_steps, ftol=OPTIMIZER_FTOL)
    constraints, bounds = (), None
    if method == "SLSQP":
        constraints = common.physical_constraint(problem, parameters=globals())
    else:
        bounds = Bounds(-PARAMETER_BOUND*problem.scales, PARAMETER_BOUND*problem.scales)
        options.update(maxfun=MAX_TRIALS, gtol=OPTIMIZER_GTOL,
                       maxcor=LBFGSB_MAXCOR, maxls=LBFGSB_MAXLS)
    return opt.minimize(problem, method=method, bounds=bounds, constraints=constraints,
                        callback=lambda x: record_step(), options=options)

def verify_endpoint(stage, args, summary, history, initial_equilibrium):

    return free.verify_endpoint(SimpleNamespace(**globals()), None, stage, args, summary, history, initial_equilibrium)


def postprocess(stage, args, summary, monitor, history, initial_equilibrium, verified):
    return free.postprocess(SimpleNamespace(**globals()), stage, args, summary, monitor, history, initial_equilibrium, verified)


def main(argv=None, *, method="SLSQP"):
    """Prepare the case, optimize, and independently solve the endpoint."""

    return free.run(parse_args(argv), case=SimpleNamespace(**globals()), method=method)


if __name__ == "__main__":
    raise SystemExit(main())
