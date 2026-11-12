#!/usr/bin/env python
"""Optimize a boundary for piecewise omnigenity at finite beta, from the plain seed.

The pressure is p proportional to 1-s. Each stage names its boundary mode
number and its volume-averaged beta, so one run climbs the mode ladder in
vacuum and then ramps beta to the target: the circular seed needs no
vacuum optimization first. SciPy nonlinear least squares varies the boundary,
and the residual vector holds the mapped pwO variance, aspect ratio,
rotational transform floor, mirror ratio and shaping limits. The magnetic well,
Mercier stability (both hinges), GammaC and bootstrap-current control are
independent switches below. NEO compares the vacuum seed with the result.
Requires vmex[neoclassical].
"""

import os
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt
from vmex.core.neoclassical import screen_transport
from vmex.core.pwo import FreeMapProblem, MappedPwOResidual
from vmex.core.restart import restart_state

# Number of field periods, and the seed deck the boundary is shaped from:
NFP = 3
print(f"finite-beta pwO optimization, NFP = {NFP}")
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"

# Deterministic seed: boundary modes (n, m) given a rotating-ellipse perturbation of
# this amplitude, RBC(n, m) = -a and ZBS(n, m) = +a. One or two modes kickstart the
# optimizer from the circular seed; None leaves INPUT_FILE as it is, to continue a
# saved stage:
SEED_MODES = {(-1, 1): 0.05}

# Volume-averaged beta of the last stage, and the beta each stage runs at as a
# fraction of it (see BETA_FRACTIONS below):
TARGET_BETA = 0.03
BETA_WEIGHT = 100.0               # on the relative beta error

# Flux surfaces the pwO residual is evaluated on, and its Boozer resolution:
SURFACES = np.linspace(0.1, 0.9, 5)
MBOZ = 16

# Initial values of the pwO map (a, b, c, eta_min), and which of them the
# optimizer may change; remove a name to hold it fixed:
MAP_PARAMETERS = dict(a=0.3, b=0.0, c=-2.0, eta_min=2.8)
FREE_MAP = ("a", "b", "c", "eta_min")

# Physics terms, each independent. A floor of None (or a bootstrap term of None)
# removes the term. The well and Mercier hinges act at finite beta only: the
# pressure itself deepens the well, so the vacuum stages keep the pwO geometry:
MAGNETIC_WELL_FLOOR = 0.01        # hinge on the magnetic well
MAGNETIC_WELL_WEIGHT = 1.0e3
MERCIER_MARGIN = 0.3              # DMerc floor on the optimizer grid; the final grid differs
MERCIER_S_MIN = 0.2               # DMerc inward of this is set by the radial grid
MERCIER_WEIGHT = 100.0
# Zero bootstrap current. "band" flattens B across the unmapped band where B_max
# lives (cheap, geometric); the last DKX_FINAL_STAGES stages also drive the DKX
# bootstrap <J.B> on DKX_SURFACES to zero (0 = off). None leaves the current free:
BOOTSTRAP_TERM = None
BOOTSTRAP_WEIGHT = 1.0e3 if NFP == 3 else 1.0e4
DKX_FINAL_STAGES = 0
DKX_SURFACES = (0.25, 0.55, 0.85)
DKX_WEIGHT = 30.0
DKX_RESOLUTION = dict(Ntheta=15, Nzeta=21, Nxi=48, NL=4, Nx=5)
ARIES_MINOR_RADIUS = 1.7044       # metres
ARIES_DENSITY, ARIES_TEMPERATURE = 2.38e20, 9.45e3  # ne (1 - s^5) in m^-3, Te = Ti (1 - s) in eV

# The last GAMMA_C_FINAL_STAGES stages also minimize the smooth fast-ion proxy
# GammaC on GAMMA_C_SURFACES (0 = off):
GAMMA_C_FINAL_STAGES = 0
GAMMA_C_SURFACES = tuple(SURFACES[1:])
GAMMA_C_WEIGHT = 100.0

# Stage ladder: highest boundary mode number varied in each stage, its beta as a
# fraction of TARGET_BETA, and the residual evaluations it may spend:
MAX_MODES = [1, 2, 3, 3, 3]
BETA_FRACTIONS = [0.0, 0.0, 0.0, 0.5, 1.0]
MAX_NFEV = [15, 15, 15, 20, 30]
NS_ARRAY = np.array([35])
NITER_ARRAY = np.array([20000])
FTOL_ARRAY = np.array([1e-12])

# Targets:
ASPECT_TARGET = 6.0
ASPECT_WEIGHT = 10.0
IOTA_FLOOR = 0.31 if NFP == 1 else 0.42  # minimum |iota| over the profile
IOTA_WEIGHT = 1.0e4
MIRROR_LIMIT = 0.16               # maximum (Bmax-Bmin)/(Bmax+Bmin) on the pwO surfaces
MIRROR_WEIGHT = 1.0e3
AXIS_LIMIT = 0.25                 # maximum RMS helical excursion of the magnetic axis, metres
ELONGATION_LIMIT = 6.0
EDGE_JACOBIAN_FLOOR = 0.25        # minimum min/max Jacobian ratio in each of the three outer cells
PWO_WEIGHT = 1.0e3                # the seed's raw pwO variance is of order 0.01

# Step control. One scaled variable moves a low-order coefficient by
# PARAMETER_STEP metres, and a stage may move it MAX_PARAMETER_CHANGE steps:
PARAMETER_STEP = 0.05
MAX_PARAMETER_CHANGE = 5.0
ESS_ALPHA = 1.2                   # smaller values let high Fourier modes move more
VARY_MAJOR_RADIUS = False         # True optimizes RBC(0,0) instead of fixing it
HOLD_VOLUME = False               # True fixes the volume at the target aspect ratio's; it stalls NFP 4

# Equilibrium resolution: poloidal and toroidal mode numbers are max_mode + 2,
# but never below MINIMUM_MPOL. DMerc needs more modes than the boundary:
MINIMUM_MPOL = 13 if NFP == 1 else 7

# Verification solve of the optimized boundary:
FINAL_NS = [37, 45, 55, 71]        # restarted in small steps: a large jump flips the Jacobian
FINAL_FTOL = 1.0e-13
FINAL_NITER = 30000

# Every output file name contains this:
OUTPUT_NAME = f"pwO_finite_beta_nfp{NFP}"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODES, BETA_FRACTIONS, MAX_NFEV, DKX_FINAL_STAGES, GAMMA_C_FINAL_STAGES = [1], [0.0], [4], 0, 0
    FINAL_NS, FINAL_FTOL = [31], 1.0e-10

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

# VmecInput is frozen, so copy its arrays before shaping the seed boundary.
inp = vj.VmecInput.from_file(INPUT_FILE)
mpol = max(inp.mpol, MINIMUM_MPOL)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
for (n, m), amplitude in (SEED_MODES or {}).items():
    rbc[inp.ntor + n, m], zbs[inp.ntor + n, m] = -amplitude, amplitude
major_radius = float(inp.rbc[inp.ntor, 0])
# Optionally hold the plasma volume that gives the target aspect ratio at the seed's major radius:
volume = 2 * np.pi**2 * major_radius**3 / ASPECT_TARGET**2 if HOLD_VOLUME else 0.0
inp = replace(inp, rbc=rbc, zbs=zbs, am=np.array([1.0, -1.0]), pres_scale=0.0, ns_array=NS_ARRAY,
              ftol_array=FTOL_ARRAY, niter_array=NITER_ARRAY, lvolume_rfix=HOLD_VOLUME, delt=0.5,
              tvolume=volume).change_resolution(
    mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)

if DKX_FINAL_STAGES:
    from dkx.bootstrap import KineticBootstrapMismatch
    from vmex.core.bootstrap import KineticProfiles
    size = major_radius / ASPECT_TARGET / ARIES_MINOR_RADIUS  # seed minor / ARIES-CS
    kinetic = KineticBootstrapMismatch(
        KineticProfiles(ARIES_DENSITY / size * np.array([1, 0, 0, 0, 0, -1.]),
                        ARIES_TEMPERATURE * np.array([1, -1.]), ARIES_TEMPERATURE * np.array([1, -1.])),
        DKX_SURFACES, mismatch=False, collision_model="pas+momentum_correction",
        resolution=DKX_RESOLUTION, spectral_angles=True)

    def dkx_j_dot_B(equilibrium_state, solver_context):
        """DKX <J.B> in MA T/m^2 at ARIES-CS size."""
        return kinetic.residuals_state(equilibrium_state, solver_context) * size**2

    kinetic._setup(int(NS_ARRAY[-1]), int(inp.nfp))  # DKX builds its host-side operators eagerly, not under a trace

### Set up the objective ######################################################

def max_mirror(equilibrium_state, solver_context):
    """Full-surface mirror ratio on the pwO surfaces, which the mapped variance does not see."""
    return pwo.diagnostics_state(equilibrium_state, solver_context)[3]


def boozer_spectrum(equilibrium_state, solver_context):
    """The field and iota the map reads; FreeMapProblem turns these rows into the pwO residual."""
    return pwo.spectrum_state(equilibrium_state, solver_context)


def dmerc(equilibrium_state, solver_context):
    """DMerc from MERCIER_S_MIN to inside the edge; positive is stable."""
    values = opt.d_merc_state(equilibrium_state, solver_context)
    s = np.linspace(0.0, 1.0, values.shape[0])
    return values[(s >= MERCIER_S_MIN) & (s < 1.0)]


def mercier(equilibrium_state, solver_context):
    """DMerc hinge, log-bounded: DMerc diverges at zero shear and would swamp every other row."""
    return jnp.log1p(opt.Hinge(dmerc, floor=MERCIER_MARGIN)(equilibrium_state, solver_context))


def stage_terms(beta, with_dkx, with_gamma_c):
    """The objective rows of one stage; the beta, well and Mercier rows exist only at finite beta."""
    terms = [
        (boozer_spectrum, 0.0, 1.0),  # must stay first: FreeMapProblem reads these rows
        (opt.aspect_ratio, ASPECT_TARGET, ASPECT_WEIGHT),
        (opt.Hinge(opt.min_abs_iota, floor=IOTA_FLOOR), 0.0, IOTA_WEIGHT),
        (opt.Hinge(max_mirror, ceiling=MIRROR_LIMIT), 0.0, MIRROR_WEIGHT),
        (opt.Hinge(opt.axis_excursion, ceiling=AXIS_LIMIT), 0.0, 100.0),
        (opt.Hinge(opt.edge_jacobian_ratio, floor=EDGE_JACOBIAN_FLOOR), 0.0, 100.0),
        (opt.Hinge(opt.max_elongation, ceiling=ELONGATION_LIMIT), 0.0, 10.0),
    ]
    if beta > 0:
        terms.append((opt.volume_average_beta, beta, BETA_WEIGHT / beta**2))
        if MAGNETIC_WELL_FLOOR is not None:
            terms.append((opt.Hinge(opt.magnetic_well, floor=MAGNETIC_WELL_FLOOR), 0.0, MAGNETIC_WELL_WEIGHT))
        if MERCIER_MARGIN is not None:
            terms.append((mercier, 0.0, MERCIER_WEIGHT))
    if with_gamma_c:
        terms.append((opt.GammaCSmooth(GAMMA_C_SURFACES), 0.0, GAMMA_C_WEIGHT))
    if with_dkx:
        terms.append((dkx_j_dot_B, 0.0, DKX_WEIGHT))
    return terms


pwo = MappedPwOResidual(SURFACES, mboz=MBOZ, nboz=MBOZ, oversample=2, limit_map=True,
                        **MAP_PARAMETERS)
report = opt.EquilibriumReporter(
    ("pwO total", pwo.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("beta", opt.volume_average_beta, ".3%"), ("min |iota|", opt.min_abs_iota, ".4f"),
    ("magnetic well", opt.magnetic_well, ".4f"),
    ("min DMerc", lambda state, rt: jnp.min(dmerc(state, rt)), ".3e"),
    ("axis excursion", opt.axis_excursion, ".4f"),
    ("max mirror", lambda state, rt: jnp.max(max_mirror(state, rt)), ".3f"))
monitor = opt.OptimizationMonitor()

### Run the optimization ######################################################

# The plain seed carries no pressure: it cannot hold the target beta, so the NEO comparison is in vacuum.
seed_input = inp
equilibrium = seed_vacuum = opt.solve_equilibrium(inp)
report("seed", equilibrium)
# Stages that share a boundary resolution and beta share ONE problem, as in
# QA_optimization.py: every jitted graph is compiled once for the whole group.
n_stage = len(MAX_MODES)
dkx_stage = [i >= n_stage - DKX_FINAL_STAGES for i in range(n_stage)]
gamma_c_stage = [i >= n_stage - GAMMA_C_FINAL_STAGES for i in range(n_stage)]
resolution_of = [max(max_mode + 2, inp.mpol) for max_mode in MAX_MODES]
group_of = [(resolution_of[i], BETA_FRACTIONS[i], dkx_stage[i], gamma_c_stage[i]) for i in range(n_stage)]
problem, group = None, None
for i, (max_mode, max_nfev) in enumerate(zip(MAX_MODES, MAX_NFEV)):
    beta = BETA_FRACTIONS[i] * TARGET_BETA
    print(f"\n===== finite-beta pwO stage {i}, max_mode = {max_mode}, beta = {beta:.2%} =====")
    if group_of[i] != group:
        group = group_of[i]
        mpol = group[0]
        inp = inp.change_resolution(mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
        if beta > 0:  # pressure that gives this beta on the current boundary
            unit = opt.solve_equilibrium(replace(inp, pres_scale=1.0), initial_state=restart_state(equilibrium.wout, inp))
            inp = replace(inp, pres_scale=beta / float(unit.wout.betatotal))
        last = max(j for j in range(n_stage) if group_of[j] == group)
        problem = opt.VmecProblem.from_tuples(
            inp, stage_terms(beta, dkx_stage[i], gamma_c_stage[i]), max_mode=max(MAX_MODES[i:last + 1]),
            vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True,
            ess_alpha=ESS_ALPHA, restart_from=equilibrium, progress=True)
        x = problem.x0
        monitor.problem = problem
        if not ci_smoke:
            problem.compile_residual_and_jacobian()
    stage = problem.subproblem(max_mode=max_mode, x=x)
    print(f"dof_names = {stage.dof_names}")
    joint = FreeMapProblem(stage, pwo, equilibrium, MAP_PARAMETERS, FREE_MAP, weight=PWO_WEIGHT,
                           band_weight=BOOTSTRAP_WEIGHT if BOOTSTRAP_TERM == "band" else 0.)
    # Map variables are sigmoid-scaled, so a unit step spans their bounds.
    step = np.concatenate((PARAMETER_STEP * stage.scales, np.ones(len(FREE_MAP))))
    change = np.concatenate((MAX_PARAMETER_CHANGE * step[:joint.nx], np.full(len(FREE_MAP), np.inf)))
    result = least_squares(
        joint.residual, joint.x0, jac=joint.residual_jac,
        x_scale=step, max_nfev=max_nfev, bounds=(joint.x0 - change, joint.x0 + change),
        ftol=1e-6, xtol=1e-10, verbose=2, callback=monitor)
    x = stage.embed(result.x[:joint.nx])
    MAP_PARAMETERS = joint.values(result.x)
    pwo.options.update(MAP_PARAMETERS)
    print(f"map = {MAP_PARAMETERS}")
    inp = problem.input_from_x(x)
    equilibrium = problem.equilibrium_from_x(x)
    report(f"stage {i}", equilibrium)
    inp.to_indata(f"input.{OUTPUT_NAME}_stage{i}")

### Check the result ##########################################################

# The optimizer's grid is not the certificate: re-solve the optimized boundary
# and the seed on a finer radial grid to a tighter tolerance and quote that.
final_resolution = dict(ftol_array=np.array([FINAL_FTOL]), niter_array=np.array([FINAL_NITER]))
final_equilibrium, seed = equilibrium, seed_vacuum
for ns in FINAL_NS:
    final_input = replace(inp, ns_array=np.array([ns]), **final_resolution)
    final_equilibrium = opt.solve_equilibrium(
        final_input, initial_state=final_equilibrium.solution,
        verbose=not ci_smoke, raise_on_max_iterations=True)
    seed = opt.solve_equilibrium(
        replace(seed_input, ns_array=np.array([ns]), **final_resolution),
        initial_state=seed.solution, verbose=False, raise_on_max_iterations=True)

# Matched NEO: the seed on the same radial grid, the same half-mesh surfaces.
half_mesh = (np.arange(FINAL_NS[-1] - 1) + 0.5) / (FINAL_NS[-1] - 1)
neo_surfaces = half_mesh[np.abs(half_mesh[:, None] - SURFACES).argmin(axis=0)]
seed_Q = screen_transport(seed.wout, neo_surfaces)
final_Q = screen_transport(final_equilibrium.wout, neo_surfaces)

### Print, plot and save ######################################################

final = report("final", final_equilibrium)
beta_bounds = [("beta", final["beta"], 0.9 * TARGET_BETA, "min"), ("beta", final["beta"], 1.1 * TARGET_BETA, "max")]
mercier_bound = [] if MERCIER_MARGIN is None else [("min DMerc", final["min DMerc"], 0.0, "min")]
opt.report_targets(final_equilibrium, aspect=ASPECT_TARGET, iota_floor=IOTA_FLOOR,
                   well_floor=MAGNETIC_WELL_FLOOR, extra=mercier_bound + beta_bounds)
print(f"\npwO total {final['pwO total']:.3e}")
print(f"epsilon_eff^(3/2) at s = {neo_surfaces}: seed {seed_Q}, final {final_Q}")

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {input_path}\nWrote {wout_path}")
print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")
