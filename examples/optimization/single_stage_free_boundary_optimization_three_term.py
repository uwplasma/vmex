#!/usr/bin/env python
r"""Free-boundary single-stage coil optimization on sheet-current-free equilibria.

The free arm of ``single_stage_free_boundary_optimization_coil_constraints.py``
with its equilibrium replaced: every trial is the free boundary that satisfies
all three plasma-vacuum interface conditions (B.n = 0, pressure balance and no
sheet current), not a VMEC + NESTOR solve, which balances only |B| and leaves
the tangential field jump free.

The cases (``COIL_CASE``; the case table is in that script's docstring), the
design variables (coil shapes, PHIEDGE and, with ``--bootstrap``, the current
spline values and CURTOR), the loss, the constraint rows and their bounds, and
the run directory are that script's. The exception is ``BOOTSTRAP_IN_SOLVE``
(Redl): then the current is solved with the boundary, Redl's on every half-grid
surface, and neither its values nor the bootstrap-mismatch row enter the
optimization.

The coil currents are design variables too (``FREE_COIL_CURRENTS``, no bounds),
with a row holding the axis |B| at ``B0``, and where the case sets
``VACUUM_IOTA_FLOOR`` a row floors the vacuum |iota| of the plasma boundary
(``vacuum_iota``).

The problem is ``FreeBoundaryProblem.from_loss(..., boundary_condition="three_term")``
(:class:`vmex.core.freeboundary_vc.ThreeTermFreeBoundaryProblem`): warm-started
trial boundaries and design gradients by the implicit function theorem of the
boundary fit, without recompiling when the coils or plasma parameters change.

    COIL_CASE=qa4-beta python single_stage_free_boundary_optimization_three_term.py --beta 0.025 --bootstrap \
        --modes 8 8 --ns 51 --output runs/three-term-qa4

``--seed-run <run>`` starts from another run's calibrated deck and fitted coils.
``metrics.jsonl`` adds the three interface residuals (``bn``,
``pressure_balance``, ``sheet_current``), each step's trial and boundary-fit
counts and the peak GPU memory.

After a run, ``benchmarks/three_term_highres.py`` re-solves the final coils at
another resolution and exports them, ``benchmarks/three_term_postprocess.py``
re-solves the final boundary densely for the WOUT figures and alpha losses, and
``benchmarks/coil_constraints_evolution_gifs.py`` draws the coils and LCFS of
the saved steps.
"""

import argparse
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

# JAX, ESSOS and VMEX are imported inside the functions, so that main() can set JAX_ENABLE_X64 and
# JAX_PLATFORMS before the first JAX import.

# ---- cases -----------------------------------------------------------------------------------------------------------
# COIL_CASE selects the case. The values are ellipse5's; the blocks after them override them per case.
DATA = Path(__file__).resolve().parents[1] / "data"
CASE = os.environ.get("COIL_CASE", "ellipse5")
RESOLUTION = (8, 8, 51)            # MPOL, NTOR, NS of the optimization solves
GRID = (64, 64)                    # NTHETA, NZETA
EQUILIBRIUM_FTOL = 1e-15
QA_SURFACES = tuple(i / 10 for i in range(1, 11))
INPUT_FILE = DATA / "input.rotating_ellipse_nfp2"
COILS_FILE = DATA / "ESSOS_coils_ellipse5.json"
BETA_DEFINITION = "volume"         # --beta is <beta>; "axis": WOUT betaxis
# T: axis |B| (opt.axis_field_strength) the seed is calibrated to and the B-axis row holds
B0 = 1.0
FREE_PHIEDGE = True                # PHIEDGE is a design variable (False: fixed PHIEDGE)
FREE_COIL_CURRENTS = True          # every base-coil current is a design variable, with no bounds
COIL_CURRENT_STEP = 0.06           # coordinate scale of the relative coil-current change
PHIEDGE_STEP = 0.05                # coordinate scale of the relative PHIEDGE change
OPTIMIZER_FTOL = 1e-10             # SLSQP ftol

# --bootstrap: Landreman-Buller-Drevlak kinetic profiles ne ~ 1 - s^5, Te = Ti ~ 1 - s, at the
# beta and collisionality of a Helios-like reactor (n T ~ B^2 and nu* ~ n R / T^2 held), and a
# self-consistent bootstrap current: CURRENT_KNOTS spline values (the last one fixed) and CURTOR
# are design variables, and the mismatch sum_j R_j^2 against the BOOTSTRAP_MODEL current (Redl
# or DKX, in Redl's normalized form) is held under REDL_TOLERANCE. The seed's Picard loop is Redl's.
# DKX: the current is a line segment through a value on every half-mesh surface (those values,
# the axis one and CURTOR are design variables), its seed DKX's on the same surfaces.
REACTOR_R0, REACTOR_B0, REACTOR_N0, REACTOR_T0 = 8.0, 6.0, 1.5e20, 15.0e3   # m, T, 1/m^3, eV
PRESSURE_SHAPE = (1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0)  # AM of p ~ (1 - s)(1 - s^5), the kinetic profiles' pressure
REDL_SURFACES = None              # None: every VMEC half-grid surface, as simsopt's RedlGeomVmec
REDL_N_LAMBDA, REDL_TOLERANCE = 32, 1e-3
PICARD_ITERATIONS, PICARD_TOLERANCE, PICARD_RELAX = 10, 1e-3, 1.0
BOOTSTRAP_BETA_STEP = 0.01        # a larger --beta is ramped in with its bootstrap current, in steps of at most this
BOOTSTRAP_BETA_START = None       # set: the ramp first doubles beta from this, so a seed with little vacuum
                                  # iota stays under its eps iota^2 limit while the bootstrap current grows
OHMIC_CURRENT = None              # set (A): the seed ramp's prescribed current, blended into the bootstrap one
CURRENT_KNOTS, CURRENT_STEP = 8, 0.05   # step relative to the largest knot value and to |CURTOR|
# Redl cases: the current is no design variable but solved with every equilibrium (ThreeTermFreeBoundaryModel
# bootstrap=), Redl-self-consistent on every half-grid surface, with I'(0) = 0 (a quasisymmetric bootstrap
# current density vanishes on the axis with the trapped fraction, f_t ~ eps^(1/2)).
BOOTSTRAP_IN_SOLVE = False
SEED = None                        # (nfp, aspect, b / a_eff): rotating ellipse replacing the deck's boundary
HELICITY = (1, 0)                  # quasisymmetry (M, N); None minimizes the constructed QI residual
TARGET_NAME = "QA"
QI_SURFACES = tuple(i / 5 for i in range(1, 6))
QI_OPTIONS = dict(mboz=12, nboz=12, nphi=61, nalpha=18, n_bounce=21)  # examples/optimization/QI_optimization.py
MIRROR_LIMIT, MIRROR_MARGIN = None, 0.001  # upper limit on the edge mirror ratio (Bmax - Bmin) / (Bmax + Bmin)
BOOTSTRAP_MODEL = "redl"          # "dkx": the DKX kinetic <j.B> replaces Redl in the self-consistency row
DKX_SURFACES, DKX_COLLISION_OPERATOR = None, 0  # None: every half-mesh surface; 0: momentum-conserving Fokker-Planck

# Physical targets, imposed as hard inequalities.
IOTA_FLOOR, IOTA_MARGIN = 0.41, 0.0005
IOTA_CEILING = None                # upper limit on max |iota|
IOTA_EDGE = False                  # True: floor and ceiling also bound the edge iotaf
VACUUM_IOTA_FLOOR = None           # set: floor on the vacuum |iota| (no pressure or current) of the plasma boundary
ASPECT_RANGE = (4.9, 5.1)
RADIUS_TARGET, RADIUS_TOLERANCE, RADIUS_MARGIN = 1.0, 0.01, 0.001
B_AXIS_TOLERANCE, B_AXIS_MARGIN = 1e-3, 1e-4  # T, the band of the axis |B| row about B0

# Coils and their hard engineering limits.
COIL_ORDER, N_SEGMENTS = 16, 256
COIL_STEP = 0.05                   # coordinate scale of the coil Fourier modes
LENGTH_LIMIT = 5.0                 # m, each independent coil
# Set: the limits follow the widest plasma allowed, a = R / aspect_min, and the clearance d:
# LENGTH_LIMIT = c_L 2 pi (a + d), CURVATURE_LIMIT = c_k / (a + d), MSC_LIMIT = c_m / (a + d)^2.
COIL_LIMIT_FACTORS = None          # (c_L, c_k, c_m)
COIL_FIT_MAXITER = 200             # L-BFGS-B iterations of the finite-beta seed coil refit
# Finite-beta seed coil refit: B.n/|B| unit and penalty weight of the scaled coil rows.
COIL_FIT_NORMAL_SCALE, COIL_FIT_WEIGHT = 1.0e-3, 1.0e3
CURVATURE_LIMIT = 5.0              # 1/m, everywhere along each coil
MSC_LIMIT = 5.0                    # 1/m^2, each coil
COIL_DISTANCE_LIMIT = 0.15         # m, including symmetry copies
COIL_SURFACE_DISTANCE_LIMIT = 0.20 # m, to the current plasma boundary
# Interior margins of the sampled constraints; endpoint checks use the limits.
CURVATURE_MARGIN, MSC_MARGIN, LENGTH_MARGIN, DISTANCE_MARGIN = 0.10, 0.02, 1e-5, 0.001

if CASE == "ellipse5-beta7":
    # Helios-like: a low iota floor, --beta on axis, the bootstrap current supplying the rest of the transform.
    BETA_DEFINITION = "axis"
    IOTA_FLOOR = 0.16
    # damped: the current dominates iota (bootstrap.self_consistent_bootstrap)
    PICARD_ITERATIONS, PICARD_RELAX = 30, 0.5
elif CASE in ("qa3", "qh", "qi"):
    COILS_FILE = DATA / f"ESSOS_coils_{CASE.replace('-', '_')}.json"
    COIL_ORDER = 8
    LENGTH_LIMIT, CURVATURE_LIMIT, MSC_LIMIT = 3.5, 8.0, 10.0
    COIL_DISTANCE_LIMIT, COIL_SURFACE_DISTANCE_LIMIT = 0.08, 0.15
    if CASE == "qa3":
        SEED, IOTA_FLOOR, ASPECT_RANGE = (3, 6.0, 0.5), 0.41, (5.9, 6.1)
    elif CASE == "qh":
        SEED, HELICITY, TARGET_NAME, ASPECT_RANGE = (4, 6.0, 0.9), (1, -1), "QH", (5.9, 6.1)
        IOTA_FLOOR = 1.1           # between the iota = 1 and 8/7 resonances
    else:
        SEED, HELICITY, TARGET_NAME, ASPECT_RANGE, MIRROR_LIMIT = (4, 8.0, 0.5), None, "QI", (7.9, 8.1), 0.21
        IOTA_FLOOR = 0.51          # above the iota = 1/2 resonance
elif CASE.removesuffix("-tok") in ("qa4-beta", "qi6-beta", "qh4-beta"):
    # Finite-beta, self-consistent bootstrap cases (Redl for the QA and QH, DKX for the QI), 4 order-12 coils per
    # half period (qh4-beta: the 3 QH coils).
    COILS_FILE = DATA / f"ESSOS_coils_{CASE.replace('-', '_') if CASE != 'qh4-beta' else 'qh'}.json"
    # (1.8, 2.5, 1.2): mid-range of Wechsung et al. (2022), Jorge et al. (2023) and Wiedman et al. (2024)
    COIL_ORDER, COIL_LIMIT_FACTORS = 12, (1.8, 2.5, 1.2)
    COIL_FIT_MAXITER = 3000  # at 200 the qa4-beta refit left B.n/|B| ~3e-3 and the first free solve could fail
    PICARD_ITERATIONS, PICARD_RELAX, BOOTSTRAP_BETA_STEP = 30, 0.5, 0.005
    # The total iota floor bounds the half-mesh surfaces and the edge; R0 is held to 1 mm.
    IOTA_EDGE, RADIUS_TOLERANCE, RADIUS_MARGIN = True, 1e-3, 1e-4
    if CASE.startswith("qa4-beta"):
        SEED, IOTA_FLOOR, ASPECT_RANGE = (2, 4.0, 0.5), 0.27, (3.5, 4.5)
        # The bootstrap current can carry the whole transform of an axisymmetric boundary, so its vacuum iota is
        # floored too, at LBD22's optimized QA (0.23): the boundary keeps its stellarator shaping.
        VACUUM_IOTA_FLOOR = 0.23
        COIL_DISTANCE_LIMIT, COIL_SURFACE_DISTANCE_LIMIT = 0.10, 0.20
        BOOTSTRAP_IN_SOLVE = True
    elif CASE.startswith("qh4-beta"):
        SEED, HELICITY, TARGET_NAME, ASPECT_RANGE = (4, 6.0, 0.9), (1, -1), "QH", (5.9, 6.1)
        IOTA_FLOOR = 1.1           # between the iota = 1 and 8/7 resonances, as qh
        COIL_DISTANCE_LIMIT, COIL_SURFACE_DISTANCE_LIMIT = 0.08, 0.15
        BOOTSTRAP_IN_SOLVE = True
    else:
        SEED, HELICITY, TARGET_NAME, ASPECT_RANGE, MIRROR_LIMIT = (4, 6.0, 0.7), None, "QI", (5.9, 6.1), 0.21
        # Stellaris (Lion et al. 2025): iota 0.86 on axis to 0.98 at the edge, below the 4/4 islands
        IOTA_FLOOR, IOTA_CEILING, BOOTSTRAP_MODEL = 0.86, 0.98, "dkx"
        COIL_DISTANCE_LIMIT, COIL_SURFACE_DISTANCE_LIMIT = 0.08, 0.15
elif CASE != "ellipse5":
    raise ValueError(f"unknown COIL_CASE {CASE!r}")
if CASE.endswith("-tok"):
    # A circular tokamak with a 1% helical ripple b / a, as vmex examples/data/input.minimal_seed_nfp*.
    # It has no vacuum transform: an Ohmic current of about the final bootstrap current carries the beta ramp.
    SEED = (SEED[0], SEED[1], 0.01)
    # Same sign as the bootstrap current it hands over to (negative for the (1, -1) QH), so the blend
    # never passes through zero current, where the seed has no transform.
    OHMIC_CURRENT = 1.0e5 if CASE.startswith("qa4-beta") else -6.0e4 if CASE.startswith("qh4-beta") else 6.0e4
REDL_HELICITY = 0 if HELICITY is None else HELICITY[1]  # Redl's quasisymmetry N (simsopt convention)
# SLSQP's first step has an identity Hessian, so it grows with the target residual, which starts near 1 for
# a QH or QI seed (|q| ~ 1.6) against ~0.3 for a QA one; their design coordinates are scaled down, as the
# coil-constraints script's fixed arm scales its seeded boundary steps (0.1 -> 0.02).
DESIGN_STEP_SCALE = 1.0 if TARGET_NAME == "QA" else 0.2
if COIL_LIMIT_FACTORS is not None:
    _radius = RADIUS_TARGET / ASPECT_RANGE[0] + COIL_SURFACE_DISTANCE_LIMIT
    LENGTH_LIMIT = COIL_LIMIT_FACTORS[0] * 2 * math.pi * _radius
    CURVATURE_LIMIT, MSC_LIMIT = COIL_LIMIT_FACTORS[1] / _radius, COIL_LIMIT_FACTORS[2] / _radius**2
CURVATURE_POINTS = max(512, 64 * COIL_ORDER)  # quadrature points per coil of the length and curvature rows
DISTANCE_POINTS = max(128, 16 * COIL_ORDER)   # quadrature points per coil of the distance rows
SURFACE_GRID = (61, 64)                       # (nphi, ntheta) of the plasma surface in the clearance row
SELF_CLEARANCE = 1e-6                         # m: numerical nonintersection guard


# ---- seeds, targets and restarts -------------------------------------------------------------------------------------
def seed_input():
    """The case's fixed-boundary seed at the optimization resolution.

    With ``SEED = (nfp, aspect, ratio)`` the boundary is the rotating ellipse
    R = R0 + a cos(theta) - b cos(theta + nfp phi), Z = a sin(theta) + b sin(theta + nfp phi)
    with a^2 - b^2 = (R0 / aspect)^2, b = ratio R0 / aspect and PHIEDGE for B0 ~ 1 T;
    otherwise it is ``INPUT_FILE``'s. ``finite_beta_input`` then calibrates PHIEDGE.
    """
    import vmex as vj

    mpol, ntor, ns = RESOLUTION
    inp = vj.VmecInput.from_file(INPUT_FILE)
    if SEED is not None:
        inp = replace(inp, nfp=SEED[0])
    inp = inp.change_resolution(mpol=mpol, ntor=ntor, ntheta=GRID[0], nzeta=GRID[1])
    if SEED is not None:
        _, aspect, ratio = SEED
        minor = RADIUS_TARGET / aspect
        rbc, zbs = np.zeros_like(inp.rbc), np.zeros_like(inp.zbs)
        rbc[ntor, 0] = RADIUS_TARGET
        rbc[ntor, 1] = zbs[ntor, 1] = minor * np.sqrt(1.0 + ratio**2)
        rbc[ntor - 1, 1], zbs[ntor - 1, 1] = -ratio * minor, ratio * minor
        inp = replace(inp, rbc=rbc, zbs=zbs, phiedge=np.pi * minor**2)
    return replace(inp, ns_array=np.array([ns]), ftol_array=np.array([EQUILIBRIUM_FTOL]), lfreeb=False)


def target_residual():
    """Residual vector of the case's target: quasisymmetry of ``HELICITY``, or constructed QI."""
    from vmex import optimize as opt
    from vmex.core.qi import ConstructedQIResidual

    if HELICITY is None:
        return ConstructedQIResidual(np.asarray(QI_SURFACES), **QI_OPTIONS)
    return opt.QuasisymmetryRatioResidual(np.asarray(QA_SURFACES), *HELICITY)


def finite_beta_input(inp, beta, device, am=(1.0, -1.0)):
    """Give ``inp`` a pressure p ~ ``am`` (power series in s, default 1 - s) whose fixed-boundary seed has ``beta``.

    ``BETA_DEFINITION`` "volume": ``beta`` is <beta>; "axis": WOUT ``betaxis``.
    The pressure is ramped in with hot restarts, then three corrections rescale
    PHIEDGE until the axis |B| (``opt.axis_field_strength``) is ``B0``, and the
    pressure to ``beta``. Returns the input and the last fixed-boundary solve.
    """
    from vmex import optimize as opt

    axis = BETA_DEFINITION == "axis"
    pressure = beta * B0**2 / (8e-7 * np.pi) if axis else beta / (4e-7 * np.pi)
    shape = np.zeros_like(np.asarray(inp.am, dtype=float))
    shape[: len(am)] = am
    inp = replace(inp, am=shape)
    fixed = None
    ramp = (0.25, 0.5, 0.75, 1.0) if beta > 0 else (1.0,)
    for index in range(len(ramp) + 3):
        if index < len(ramp):
            inp = replace(inp, pres_scale=ramp[index] * pressure)
        else:
            # beta ~ p / PHIEDGE^2 at fixed shape, so a flux rescale carries its pressure along
            measured = float(fixed.wout.betaxis if axis else fixed.wout.betatotal)
            flux = B0 / float(opt.axis_field_strength(fixed.state, fixed.runtime))
            inp = replace(inp, phiedge=float(inp.phiedge) * flux,
                          pres_scale=inp.pres_scale * flux**2 * (beta / measured if beta > 0 else 1.0))
        fixed = opt.solve_equilibrium(inp, initial_state=None if fixed is None else fixed.state, device=device,
                                      raise_on_max_iterations=True, polish_force_balance=False)
    w = fixed.wout
    print(f"PRES_SCALE = {inp.pres_scale:.6e} Pa, PHIEDGE = {float(inp.phiedge):.6f} Wb: betaxis = "
          f"{float(w.betaxis):.4%}, <beta> = {float(w.betatotal):.4%}, axis |B| = "
          f"{float(opt.axis_field_strength(fixed.state, fixed.runtime)):.5f} T")
    return inp, fixed


def redl_surfaces():
    """Surfaces of the Redl self-consistency check: ``REDL_SURFACES``, or by default every VMEC
    half-grid surface s = (j - 1/2) / (ns - 1) at the run's ns, as simsopt's ``RedlGeomVmec``."""
    if REDL_SURFACES is not None:
        return np.asarray(REDL_SURFACES)
    ns = int(RESOLUTION[2])
    return (np.arange(1, ns) - 0.5) / (ns - 1)


def redl_profiles(inp):
    """The kinetic profiles of ``bootstrap_input`` for a deck's calibrated pressure, and their Redl mismatch."""
    from vmex.core.bootstrap import ELEMENTARY_CHARGE, KineticProfiles, RedlBootstrapMismatch

    r0 = float(inp.rbc[inp.ntor, 0])
    t0 = REACTOR_T0 * (B0 / REACTOR_B0) ** (2 / 3) * (r0 / REACTOR_R0) ** (1 / 3)
    n0 = REACTOR_N0 * (B0 / REACTOR_B0) ** (4 / 3) * (REACTOR_R0 / r0) ** (1 / 3)
    scale = float(inp.pres_scale) / (2 * ELEMENTARY_CHARGE * n0 * t0)
    n0, t0 = n0 * scale ** (2 / 3), t0 * scale ** (1 / 3)
    profiles = KineticProfiles(n0 * np.array([1.0, 0, 0, 0, 0, -1.0]), t0 * np.array([1.0, -1.0]),
                               t0 * np.array([1.0, -1.0]))
    return profiles, RedlBootstrapMismatch(profiles, REDL_HELICITY, redl_surfaces(), n_lambda=REDL_N_LAMBDA)


def bootstrap_input(inp, beta, device):
    """``finite_beta_input`` with reactor-like kinetic profiles and their self-consistent Redl current.

    ne = n0 (1 - s^5) and Te = Ti = T0 (1 - s), so p = 2 e ne Te ~ (1 - s)(1 - s^5). n0 and T0
    start at the Helios-like reactor's collisionality nu* ~ n R / T^2 and beta ~ n T / B^2 moved
    to this R0 and B0, and follow the beta calibration as n ~ p^(2/3), T ~ p^(1/3), which keeps
    nu*. A Picard loop then makes the current Redl's, and it is resampled onto
    ``CURRENT_KNOTS`` spline knots (DKX: instead DKX's on every half-mesh
    surface, ``dkx_current``); with ``BOOTSTRAP_IN_SOLVE`` (Redl) it is instead
    solved in the run's own representation, Redl's on every half-mesh surface
    (``bootstrap.HalfMeshCurrent``), so the first step starts self-consistent.

    Above ``BOOTSTRAP_BETA_STEP`` beta is ramped in steps of at most that size,
    each step's pressure ramp carrying the previous step's bootstrap current, so
    its transform holds the equilibrium together as beta rises; with
    ``BOOTSTRAP_BETA_START`` the ramp first doubles beta from that value. With
    ``OHMIC_CURRENT`` (a near-axisymmetric seed) beta is instead ramped at that
    prescribed current and the current then blended into Redl's.

    Returns the input, the equilibrium and the Redl mismatch.
    """
    from vmex import optimize as opt
    from vmex.core.bootstrap import self_consistent_bootstrap
    from vmex.core.freeboundary_vc import ThreeTermFreeBoundaryModel

    ac = np.zeros_like(np.asarray(inp.ac, dtype=float))
    ac[0] = 1.0
    inp = replace(inp, ncurr=1, pcurr_type="power_series", ac=ac, curtor=0.0)
    steps = max(1, math.ceil(round(beta / BOOTSTRAP_BETA_STEP, 9)))
    betas = [beta * k / steps for k in range(1, steps + 1)]
    start = BOOTSTRAP_BETA_START
    while start is not None and start < betas[-steps]:  # double up to the first linear stage
        betas.insert(len(betas) - steps, start)
        start *= 2
    stages = len(betas)

    def picard_at(inp, n_iter):
        """Redl Picard loop of ``n_iter`` steps from ``inp``."""
        return self_consistent_bootstrap(inp, redl_profiles(inp)[0], REDL_HELICITY, n_iter=n_iter,
                                         tol=PICARD_TOLERANCE, relax=PICARD_RELAX, degree=CURRENT_KNOTS - 1,
                                         s_eval=redl_surfaces(), solve_kwargs=dict(device=device))

    if OHMIC_CURRENT:
        # A near-axisymmetric seed has no vacuum transform and, at low beta, little bootstrap current: an
        # Ohmic current I' = 2 I (1 - s) holds it together while beta rises, then is blended into the Redl
        # current at full beta, so the seed ends with the bootstrap current alone.
        ohmic = np.zeros_like(ac)
        ohmic[:2] = 2.0 * OHMIC_CURRENT * np.array([1.0, -1.0])
        inp = replace(inp, ac=ohmic, curtor=OHMIC_CURRENT)
        for stage_beta in betas:
            inp, _ = finite_beta_input(inp, stage_beta, device, am=PRESSURE_SHAPE)
        for weight in (0.25, 0.5, 0.75):
            boot = picard_at(inp, 1).input  # one Picard step: the Redl current of the current equilibrium
            ac_boot = np.zeros(max(boot.ac.size, ohmic.size))
            ac_boot[: boot.ac.size] = boot.ac
            ac_mix = (1 - weight) * np.pad(ohmic, (0, ac_boot.size - ohmic.size)) + weight * ac_boot
            inp = replace(boot, ac=ac_mix, curtor=(1 - weight) * OHMIC_CURRENT + weight * float(boot.curtor))
            print(f"Ohmic-to-bootstrap blend {weight:.2f}: CURTOR = {float(inp.curtor):.1f} A", flush=True)
        betas = []
        picard = picard_at(inp, PICARD_ITERATIONS)
        profiles, redl = redl_profiles(inp)
        inp = picard.input
    for stage, stage_beta in enumerate(betas, 1):
        inp, _ = finite_beta_input(inp, stage_beta, device, am=PRESSURE_SHAPE)
        profiles, redl = redl_profiles(inp)
        picard = picard_at(inp, PICARD_ITERATIONS)
        inp = picard.input
        if stages > 1:
            print(f"bootstrap ramp {stage}/{stages}: beta {stage_beta:.4f}, CURTOR = "
                  f"{float(inp.curtor):.1f} A, Picard {picard.iterations} iterations (converged {picard.converged})",
                  flush=True)
    n0, t0 = float(profiles.ne_coeffs[0]), float(profiles.Te_coeffs[0])
    if BOOTSTRAP_IN_SOLVE and BOOTSTRAP_MODEL == "redl":
        model = ThreeTermFreeBoundaryModel(inp, bootstrap=profiles, bootstrap_helicity=REDL_HELICITY,
                                           fixed_boundary=True)
        current = model.solve_boundary(model.params0, None, ftol=1e-10, max_nfev=30)
        state, _, params, _ = current["aux"]
        print(f"Redl current on its {current['x'].size} knots: max relative mismatch "
              f"{model.bootstrap_residual(state, params):.1e}", flush=True)
        inp = model.bootstrap.deck_with(model.fixed, current["x"])
    elif BOOTSTRAP_MODEL == "dkx":
        inp = dkx_current(inp, picard.equilibrium, device)
    else:
        inp = opt.resample_current_profile(inp, CURRENT_KNOTS)
    fixed = opt.solve_equilibrium(inp, initial_state=picard.equilibrium.state, device=device,
                                  raise_on_max_iterations=True, polish_force_balance=False)
    w, state, runtime = fixed.wout, fixed.state, fixed.runtime
    print(f"Redl seed: n0 = {n0:.4e} 1/m^3, T0 = {t0:.1f} eV, Picard {picard.iterations} iterations "
          f"(converged {picard.converged}), CURTOR = {float(inp.curtor):.1f} A, mismatch = "
          f"{float(redl.total(w)):.3e}; <beta> = {float(w.betatotal):.4%}, iota = "
          f"{float(min_abs_iota(state, runtime)):.4f}..{float(max_abs_iota(state, runtime)):.4f}"
          f", edge R B_phi = {abs(float(w.rbtor)):.5f} T m")
    return inp, fixed, redl


def abs_iota(state, runtime):
    """|iota| bounded by the floor and ceiling rows.

    The half-mesh surfaces (axis slot excluded), as ``opt.min_abs_iota``; with
    ``IOTA_EDGE`` also VMEC's edge iotaf[-1] = 1.5 iotas[-1] - 0.5 iotas[-2].
    The axis is left out: there a bootstrap current's steep part of iota
    (~ s^(1/4)) makes VMEC's extrapolated iotaf[0] drift with ns, and the
    vacuum iota row bounds the shaping's share (``vacuum_iota``).
    """
    import jax.numpy as jnp
    from vmex.core.statephysics import _iotas_half  # private: opt exposes only the half-mesh minimum

    half = _iotas_half(state, runtime)[1:]
    if IOTA_EDGE:
        half = jnp.concatenate([half, 1.5 * half[-1:] - 0.5 * half[-2:-1]])
    return jnp.abs(half)


def vacuum_iota(inp):
    """(state, runtime) -> the vacuum |iota| of the state's plasma boundary: axis, half-mesh surfaces and edge.

    The fixed-boundary equilibrium of that boundary without pressure or current
    (``inp`` otherwise, hot-restarted), differentiable through its implicit
    adjoint; the axis value is ``opt.axis_iota``, the edge VMEC's iotaf[-1].
    """
    import jax
    import jax.numpy as jnp
    from vmex import optimize as opt
    from vmex.core import implicit as im
    from vmex.core.statephysics import _iotas_half

    vacuum = replace(inp, pres_scale=0.0, curtor=0.0, ncurr=1, pcurr_type="power_series",
                     ac=np.zeros_like(np.asarray(inp.ac, dtype=float)), ac_aux_s=None, ac_aux_f=None, lfreeb=False)
    # the model's device (the GPU when present); unpinned, its certification ran on the host CPU
    device = jax.devices()[0]
    cfg = im.make_config(vacuum, multigrid=True, hot_restart=True, device=device)
    base = im.params_from_input(vacuum, device=device)
    im.runtime_from_params(base, cfg)  # its setup, built here: run_setup cannot be traced inside a jit

    def iota(state, runtime):
        rmnc, _, _, zmns = im._edge_physical(state, runtime)  # private: no public traced LCFS of a state
        rmnc, zmns = jax.device_put((rmnc, zmns), device)  # a fixed-boundary state may live on the host
        rows, cols = np.asarray(runtime.modes.n) + int(inp.ntor), np.asarray(runtime.modes.m)
        params = replace(base, rbc=jnp.zeros_like(base.rbc).at[rows, cols].set(rmnc),
                         zbs=jnp.zeros_like(base.zbs).at[rows, cols].set(zmns))
        vstate = im.solve_implicit(params, cfg)
        vrt = im.runtime_from_params(params, cfg)
        half = _iotas_half(vstate, vrt)[1:]
        return jnp.abs(jnp.concatenate([opt.axis_iota(vstate, vrt)[None], half, 1.5 * half[-1:] - 0.5 * half[-2:-1]]))

    return iota


def min_abs_iota(state, runtime):
    """Smallest |iota| of ``abs_iota``: ``opt.min_abs_iota``, with ``IOTA_EDGE`` including the edge."""
    import jax.numpy as jnp

    return jnp.min(abs_iota(state, runtime))


def max_abs_iota(state, runtime):
    """Largest |iota| of ``abs_iota``, the counterpart of ``min_abs_iota``."""
    import jax.numpy as jnp

    return jnp.max(abs_iota(state, runtime))


def bootstrap_mismatch(inp, redl, device):
    """(state, runtime) -> the bootstrap self-consistency mismatch held under ``REDL_TOLERANCE``.

    Redl's, or with ``BOOTSTRAP_MODEL = "dkx"`` the equilibrium's <j.B> against DKX's
    drift-kinetic one on ``DKX_SURFACES``, in Redl's normalized form (Redl assumes quasisymmetry).
    """
    from vmex import optimize as opt

    if BOOTSTRAP_MODEL == "redl":
        return redl.total_state
    mismatch = dkx_mismatch(inp)
    # DKX builds its per-surface operators and Boozer plan on the host at the first
    # call and caches them; make that call on a concrete equilibrium, so the
    # optimizer's traced calls only find the cache.
    seed = opt.solve_equilibrium(replace(inp, lfreeb=False), device=device, raise_on_max_iterations=True,
                                 polish_force_balance=False)
    print(f"DKX bootstrap mismatch of the fixed-boundary seed: {float(mismatch(seed.state, seed.solver_context)):.3e}",
          flush=True)
    return mismatch


def dkx_kinetic(inp):
    """DKX's bootstrap ``<j.B>`` for ``inp``'s kinetic profiles on the ``DKX_SURFACES`` rows (needs ``dkx``)."""
    from dkx.bootstrap import KineticBootstrapMismatch

    surfaces = redl_surfaces() if DKX_SURFACES is None else DKX_SURFACES
    return KineticBootstrapMismatch(redl_profiles(inp)[0], surfaces=surfaces, collision_operator=DKX_COLLISION_OPERATOR,
                                    mboz=QI_OPTIONS["mboz"], nboz=QI_OPTIONS["nboz"])


def on_grid(runtime):
    """``runtime`` with a concrete radial grid: DKX reads it on the host, and under jit it is a tracer
    (always linspace(0, 1, ns))."""
    grid = np.linspace(0.0, 1.0, runtime.setup.s_full.shape[0])
    return replace(runtime, setup=replace(runtime.setup, s_full=grid))


def dkx_mismatch(inp):
    """(state, runtime) -> DKX's bootstrap mismatch for ``inp``'s kinetic profiles (needs the ``dkx`` package)."""
    kinetic = dkx_kinetic(inp)
    return lambda state, runtime: kinetic.total(state, on_grid(runtime))


def dkx_current(inp, fixed, device):
    """``inp`` with DKX's self-consistent bootstrap current, a value on every half-mesh surface.

    Picard steps I' <- I'_DKX from the equilibrium ``fixed``: DKX's ``<j.B>`` on its rows,
    interpolated onto every half-mesh surface and inverted for dI/ds there
    (``bootstrap.current_derivative``), under-relaxed by ``PICARD_RELAX``. The profile is
    ``line_segment_ip`` through those values, the axis and edge ones extrapolated linearly,
    and CURTOR its integral: no spline fit.
    """
    import jax
    import jax.numpy as jnp
    from vmex import optimize as opt
    from vmex.core.bootstrap import current_derivative
    from vmex.core.profiles import current

    kinetic, ns = dkx_kinetic(inp), int(np.asarray(inp.ns_array)[-1])
    s = (np.arange(1, ns) - 0.5) / (ns - 1)
    rows = kinetic.rows(ns)[1]
    knots = np.r_[0.0, s, 1.0]
    enclosed = lambda z: current(inp.pcurr_type, inp.ac, inp.ac_aux_s, inp.ac_aux_f, z, bloat=inp.bloat)  # noqa: E731
    v = float(inp.curtor) / float(enclosed(1.0)) * np.asarray(jax.vmap(jax.grad(enclosed))(jnp.asarray(s)))
    kinetic.total(fixed.state, on_grid(fixed.runtime))  # DKX builds its operators on the host at a concrete call
    kinetic_j = jax.jit(lambda state, runtime: kinetic.current_profiles(state, on_grid(runtime))[2])
    derivative = jax.jit(current_derivative)
    for iteration in range(PICARD_ITERATIONS):
        values = np.r_[1.5 * v[0] - 0.5 * v[1], v, 1.5 * v[-1] - 0.5 * v[-2]]
        inp = replace(inp, pcurr_type="line_segment_ip", ac_aux_s=knots, ac_aux_f=values,
                      curtor=float(current("line_segment_ip", inp.ac, knots, values, 1.0)))
        fixed = opt.solve_equilibrium(inp, initial_state=fixed.state, device=device, raise_on_max_iterations=True,
                                      polish_force_balance=False)
        jk = np.interp(s, rows, np.asarray(kinetic_j(fixed.state, fixed.runtime)))
        target = np.asarray(derivative(jnp.asarray(jk), fixed.state, fixed.runtime)[1])
        delta = float(np.max(np.abs(target - v)) / np.max(np.abs(target)))
        print(f"DKX Picard {iteration}: CURTOR = {float(inp.curtor):.1f} A, max|I' - I'_DKX| / max|I'_DKX| = "
              f"{delta:.2e}", flush=True)
        if delta <= PICARD_TOLERANCE:
            break
        v = (1.0 - PICARD_RELAX) * v + PICARD_RELAX * target
    return inp


def boundary_from_wout(inp, wout):
    """``inp`` with the boundary of ``wout``'s last surface, truncated to its resolution."""
    rbc, zbs = np.zeros_like(np.asarray(inp.rbc)), np.zeros_like(np.asarray(inp.zbs))
    for m, n, r, z in zip(np.asarray(wout.xm, int), np.asarray(wout.xn, int) // int(wout.nfp),
                          np.asarray(wout.rmnc)[-1], np.asarray(wout.zmns)[-1]):
        if m < inp.mpol and abs(n) <= inp.ntor:
            rbc[n + inp.ntor, m], zbs[n + inp.ntor, m] = r, z
    return replace(inp, rbc=rbc, zbs=zbs)


def current_from_wout(inp, w):
    """``inp`` with ``w``'s current profile -- its type, knots or coefficients -- and CURTOR, when ``inp``
    prescribes the current (a run that solved the current changes its type and knots: line_segment_ip)."""
    if int(inp.ncurr) == 1:
        kind = str(w.pcurr_type).strip()
        if "spline" in kind or "line_segment" in kind:  # VmecInput trims both to the knots before the -1 padding
            inp = replace(inp, pcurr_type=kind, ac_aux_s=np.asarray(w.ac_aux_s, dtype=float),
                          ac_aux_f=np.asarray(w.ac_aux_f, dtype=float), curtor=float(w.ctor))
        else:
            inp = replace(inp, pcurr_type=kind, ac=np.asarray(w.ac, dtype=float)[: np.size(inp.ac)],
                          curtor=float(w.ctor))
    return inp


def restart_input(run):
    """A finished run's ``input.run`` with its final WOUT's boundary, PHIEDGE and current (``--restart``)."""
    import vmex as vj

    inp, w = vj.VmecInput.from_file(run / "input.run"), vj.read_wout(run / "wout.nc")
    return current_from_wout(replace(boundary_from_wout(inp, w), phiedge=float(w.phi[-1]), lfreeb=False), w)


# ---- coils and their field -------------------------------------------------------------------------------------------
def resize_coils(coils, order, n_segments):
    """Zero-pad higher Fourier modes without changing the curves or currents."""
    import jax.numpy as jnp
    from essos.coils import Coils, Curves

    if order < coils.order:
        raise ValueError(f"cannot truncate order-{coils.order} coils to order {order}")
    old = coils.curves
    raw = jnp.pad(old.dofs / old.scaling, ((0, 0), (0, 0), (0, 2 * (order - coils.order))))
    curves = Curves(raw, n_segments=n_segments, nfp=coils.nfp, stellsym=coils.stellsym,
                    scaling_type=old.scaling_type, scaling_factor=old.scaling_factor, scale_fixed=old.scale_fixed)
    return Coils(curves, coils.dofs_currents_raw, currents_scale=coils.currents_scale)


def scale_coil_currents(coils, rbtor):
    """Coils with every current scaled so the linked mu0 I / 2 pi is ``rbtor``.

    mu0 I / 2 pi is the mean of R B_phi on the loop R = R0, Z = 0, with R0 the
    case's ``RADIUS_TARGET`` (the seed's RBC(0,0)).
    """
    import jax
    import jax.numpy as jnp
    from essos.coils import Coils
    from essos.fields import BiotSavart

    r0, phi = RADIUS_TARGET, np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    points = jnp.asarray(r0 * np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=-1))
    field = np.asarray(jax.vmap(BiotSavart(coils).B)(points))
    linked = abs(r0 * float(np.mean(-np.sin(phi) * field[:, 0] + np.cos(phi) * field[:, 1])))
    return Coils(coils.curves, coils.dofs_currents_raw * (rbtor / linked), currents_scale=coils.currents_scale)


def coil_field(coils):
    """points (..., 3) -> the coils' Biot-Savart field B (..., 3)."""
    import jax
    from essos.fields import BiotSavart

    biot_savart = BiotSavart(coils)
    return lambda points: jax.vmap(biot_savart.B)(points.reshape(-1, 3)).reshape(points.shape)


def weighted_rms(weights, values):
    """sqrt(sum weights values^2): the RMS for area weights that sum to one."""
    import jax.numpy as jnp

    return jnp.sqrt(jnp.sum(weights * values**2))


def total_normal_field(interface, field):
    """(B_coils + B_plasma).n/|B| on a ``vmex.PlasmaVacuumInterface``, B_coils = ``field``."""
    import jax.numpy as jnp

    return interface.bnormal_residual(field) / jnp.linalg.norm(interface.total_B_out(field), axis=0)


# ---- coil chart and coil inequalities -------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class DirectCoilField:
    """Biot-Savart field of filament coils, as a differentiable pytree.

    ESSOS' ``BiotSavart`` jit-compiles its methods with the coil object as a
    static ``self``, so every new coil geometry recompiles and no derivative
    flows back to the coil arrays. This pytree carries the arrays as leaves
    instead: one compiled program serves every trial and JAX differentiates
    the field with respect to the coil parameters. The field is the mean over
    each coil's quadrature points of the filament Biot-Savart integrand.
    ``gamma``/``gamma_dash`` are ESSOS' points and tangents, shape
    ``(coils, points, 3)`` in metres; ``currents`` are in amperes.
    """

    gamma: Any
    gamma_dash: Any
    currents: Any

    def b_cyl(self, r, phi, z):
        """Evaluate the filament field in cylindrical coordinates, in tesla."""
        import jax.numpy as jnp

        rr, pp, zz = jnp.broadcast_arrays(jnp.asarray(r), jnp.asarray(phi), jnp.asarray(z))
        cosine, sine = jnp.cos(pp), jnp.sin(pp)
        xyz = jnp.stack((rr * cosine, rr * sine, zz), axis=-1)
        displacement = xyz[..., None, None, :] - jnp.asarray(self.gamma)
        radius2 = jnp.sum(displacement * displacement, axis=-1)
        inv_radius3 = jnp.maximum(radius2, 1.0e-30) ** -1.5
        differential = jnp.cross(jnp.asarray(self.gamma_dash), displacement)
        differential = differential * inv_radius3[..., None]
        current_shape = (1,) * (xyz.ndim - 1) + (-1, 1, 1)
        weighted = differential * jnp.reshape(jnp.asarray(self.currents), current_shape)
        bxyz = 1.0e-7 * jnp.mean(jnp.sum(weighted, axis=-3), axis=-2)
        br = cosine * bxyz[..., 0] + sine * bxyz[..., 1]
        bphi = -sine * bxyz[..., 0] + cosine * bxyz[..., 1]
        return br, bphi, bxyz[..., 2]


class CoilChart:
    """Design vector -> ESSOS coils and plasma parameters: ``FreeBoundaryProblem``'s ``field_from_parameters``.

    ``x`` holds, in order: with ``phiedge`` (the deck's PHIEDGE in Wb) a relative
    PHIEDGE change, PHIEDGE = phiedge * (1 + x[0]); with ``plasma_current`` (the
    deck's leading ``k`` AC coefficients, or AC_AUX_F spline values when
    ``plasma_current_spline``, then CURTOR in A; ``ncurr = 1``) ``k + 1``
    additive changes of those values in units of the largest nominal shape value
    and of ``abs(CURTOR)``; relative changes of the selected base-coil currents
    (``current[i] = nominal[i] * (1 + x[k])``); and additive changes of every
    base coil's Cartesian Fourier coefficients in metres. ``x0 = 0`` is the
    nominal state. Calling the chart gives the :class:`DirectCoilField` of
    ``coils_from_x(x)``; ``plasma_params_at`` is ``FreeBoundaryProblem``'s
    ``plasma_from_parameters``. ``scales`` holds one coordinate scale per entry of ``x``.
    """

    def __init__(self, coils, *, current_dofs, scales, phiedge=None, plasma_current=None, plasma_current_spline=False):
        """Chart about the nominal ``coils``, varying the currents of the base coils ``current_dofs``."""
        import jax

        try:  # the field is a pytree of its arrays; registered here, after main() has configured JAX
            jax.tree_util.register_dataclass(DirectCoilField, data_fields=["gamma", "gamma_dash", "currents"],
                                             meta_fields=[])
        except ValueError:  # registered by an earlier chart
            pass
        self.coefficients = np.asarray(coils.dofs_curves) / np.asarray(coils.curves.scaling)[None, None, :]
        self.currents = np.asarray(coils.dofs_currents_raw, dtype=float)
        self.current_dofs = tuple(int(i) for i in current_dofs)
        if len(set(self.current_dofs)) != len(self.current_dofs) or any(
                not 0 <= i < len(self.currents) or self.currents[i] == 0 for i in self.current_dofs):
            raise ValueError("current_dofs must be unique indices of base coils with nonzero currents")
        self.nfp, self.stellsym, self.n_segments = int(coils.nfp), bool(coils.stellsym), int(coils.n_segments)
        if phiedge is not None and not (np.isfinite(phiedge) and phiedge != 0):
            raise ValueError("phiedge must be finite and nonzero")
        self.phiedge = None if phiedge is None else float(phiedge)
        self.nphiedge = int(phiedge is not None)
        self.plasma_current = None if plasma_current is None else np.asarray(plasma_current, dtype=float)
        self.plasma_current_spline = bool(plasma_current_spline)
        self.nplasma = 0 if self.plasma_current is None else self.plasma_current.size
        if self.plasma_current is not None:
            nominal = self.plasma_current
            if nominal.ndim != 1 or nominal.size < 2:
                raise ValueError("plasma_current is [k >= 1 shape values..., CURTOR]")
            self.plasma_current_units = np.r_[np.full(nominal.size - 1, np.max(np.abs(nominal[:-1]))),
                                              abs(nominal[-1])]
            if not np.all(np.isfinite(self.plasma_current_units)) or np.any(self.plasma_current_units == 0):
                raise ValueError("plasma_current needs finite values, a nonzero shape value and a nonzero CURTOR")
        self._coil0 = self.nphiedge + self.nplasma
        self.size = self._coil0 + len(self.current_dofs) + self.coefficients.size
        self.x0 = np.zeros(self.size)
        order = (self.coefficients.shape[2] - 1) // 2
        modes = ["constant"] + [f"{kind}({k})" for k in range(1, order + 1) for kind in ("sin", "cos")]
        self.dof_names = tuple(["phiedge/nominal"] * self.nphiedge
                               + [f"plasma_current[{j}]/unit" for j in range(self.nplasma - 1)]
                               + ["curtor/nominal"] * bool(self.nplasma)
                               + [f"current[{i}]/nominal" for i in self.current_dofs]
                               + [f"coil[{i}].{axis}.{mode}" for i in range(len(self.currents))
                                  for axis in "xyz" for mode in modes])
        self.scales = np.asarray(scales, dtype=float)
        if self.scales.shape != self.x0.shape or not np.all(np.isfinite(self.scales)) or np.any(self.scales <= 0):
            raise ValueError("one positive finite scale per coordinate required")

    def phiedge_at(self, x):
        """PHIEDGE [Wb] at ``x``; only for a chart built with ``phiedge``."""
        import jax.numpy as jnp

        if self.phiedge is None:
            raise ValueError("this chart does not vary PHIEDGE")
        return self.phiedge * (1.0 + jnp.asarray(x)[0])

    def plasma_params_at(self, params, x):
        """``params`` (``ImplicitParams``) with the PHIEDGE and current profile of ``x``."""
        import jax.numpy as jnp

        x = jnp.asarray(x)
        if self.phiedge is not None:
            params = replace(params, phiedge=self.phiedge_at(x))
        if self.plasma_current is not None:
            values = jnp.asarray(self.plasma_current) + x[self.nphiedge:self._coil0] * self.plasma_current_units
            name = "ac_aux_f" if self.plasma_current_spline else "ac"
            profile = getattr(params, name).at[: self.nplasma - 1].set(values[:-1])
            params = replace(params, curtor=values[-1], **{name: profile})
        return params

    def base_currents_at(self, x):
        """Physical base-coil currents in amperes."""
        import jax.numpy as jnp

        x = jnp.asarray(x)
        currents = jnp.asarray(self.currents)
        for local, base in enumerate(self.current_dofs):
            currents = currents.at[base].add(x[self._coil0 + local] * self.currents[base])
        return currents

    def coils_from_x(self, x):
        """ESSOS coils at ``x``, without changing the nominal coils."""
        import jax.numpy as jnp
        from essos.coils import Coils, Curves

        x = jnp.asarray(x)
        raw = jnp.asarray(self.coefficients) + x[self._coil0 + len(self.current_dofs):].reshape(
            self.coefficients.shape)
        return Coils(Curves(raw, self.n_segments, self.nfp, self.stellsym), self.base_currents_at(x))

    def __call__(self, x):
        """The differentiable filament field of ``coils_from_x(x)``."""
        import jax.numpy as jnp

        coils = self.coils_from_x(x)
        return DirectCoilField(jnp.asarray(coils.gamma), jnp.asarray(coils.gamma_dash), jnp.asarray(coils.currents))


def resampled(coils, points, *, symmetric=True):
    """ESSOS curves of ``coils`` on ``points`` quadrature points; base curves only unless ``symmetric``."""
    curves = coils.curves.copy()
    curves.n_segments = points
    if not symmetric:
        curves.nfp, curves.stellsym = 1, False
    return curves


def segment_distances(a, b):
    """All distances between two closed polygons' segments, including interiors."""
    import jax.numpy as jnp

    u, v = jnp.roll(a, -1, axis=0) - a, jnp.roll(b, -1, axis=0) - b
    w = a[:, None] - b[None, :]
    aa, bb = jnp.sum(u * u, axis=-1)[:, None], jnp.sum(v * v, axis=-1)[None, :]
    uv = jnp.einsum("ik,jk->ij", u, v)
    uw, vw = jnp.sum(u[:, None] * w, axis=-1), jnp.sum(v[None, :] * w, axis=-1)
    aa, bb = jnp.maximum(aa, 1e-30), jnp.maximum(bb, 1e-30)
    den = aa * bb - uv * uv
    safe = jnp.where(den > 1e-24, den, 1.0)
    s, t = (uv * vw - bb * uw) / safe, (aa * vw - uv * uw) / safe

    def distance(s, t):
        """Squared distance between the points at segment parameters ``s`` and ``t``."""
        q = w + s[..., None] * u[:, None] - t[..., None] * v[None, :]
        return jnp.sum(q * q, axis=-1)

    candidates = [distance(jnp.zeros_like(uw), jnp.clip(vw / bb, 0, 1)),
                  distance(jnp.ones_like(uw), jnp.clip((vw + uv) / bb, 0, 1)),
                  distance(jnp.clip(-uw / aa, 0, 1), jnp.zeros_like(uw)),
                  distance(jnp.clip((uv - uw) / aa, 0, 1), jnp.ones_like(uw)),
                  jnp.where((den > 1e-24) & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1), distance(s, t), jnp.inf)]
    return jnp.sqrt(jnp.min(jnp.stack(candidates), axis=0) + 1e-30)


def separations(points):
    """Minimum intercoil and nonadjacent self-segment distance, all symmetry copies."""
    import jax
    import jax.numpy as jnp

    pairs = jnp.asarray([(i, j) for i in range(len(points)) for j in range(i + 1, len(points))])
    inter = jax.lax.map(lambda ij: jnp.min(segment_distances(points[ij[0]], points[ij[1]])), pairs)
    n = points.shape[1]
    diff = jnp.abs(jnp.arange(n)[:, None] - jnp.arange(n)[None, :])
    nonadjacent = jnp.minimum(diff, n - diff) > 1
    own = jax.lax.map(lambda p: jnp.min(jnp.where(nonadjacent, segment_distances(p, p), jnp.inf)), points)
    return jnp.min(inter), jnp.min(own)


def coil_metrics(coils):
    """ESSOS length, peak curvature and speed of each base coil, plus what ESSOS lacks."""
    import jax.numpy as jnp

    base = resampled(coils, CURVATURE_POINTS, symmetric=False)
    speed = jnp.linalg.norm(base.gamma_dash, axis=-1)
    curvature = base.curvature
    cc, own = separations(resampled(coils, DISTANCE_POINTS).gamma)
    return dict(length=base.length, peak=jnp.max(curvature, axis=1),
                msc=jnp.sum(curvature**2 * speed, axis=1) / jnp.sum(speed, axis=1),
                coil_distance=cc, self_distance=own, min_speed=jnp.min(speed, axis=1))


def coil_inequalities(coils):
    """Scaled coil-engineering rows, c >= 0 feasible: length, curvature, MSC, distances, speed."""
    import jax.numpy as jnp

    m = coil_metrics(coils)
    return jnp.concatenate(((LENGTH_LIMIT - LENGTH_MARGIN - m["length"]) / LENGTH_LIMIT,
                            (CURVATURE_LIMIT - CURVATURE_MARGIN - m["peak"]) / CURVATURE_LIMIT,
                            (MSC_LIMIT - MSC_MARGIN - m["msc"]) / MSC_LIMIT,
                            jnp.atleast_1d((m["coil_distance"] - COIL_DISTANCE_LIMIT - DISTANCE_MARGIN)
                                           / COIL_DISTANCE_LIMIT),
                            jnp.atleast_1d((m["self_distance"] - SELF_CLEARANCE) / COIL_DISTANCE_LIMIT),
                            (m["min_speed"] - 1e-4) / LENGTH_LIMIT))


def surface_distance(coils, surface):
    """Sampled coil-to-moving-surface clearance; JAX differentiates both sides."""
    import jax
    import jax.numpy as jnp

    points = resampled(coils, DISTANCE_POINTS).gamma
    targets = surface.gamma.reshape(-1, 3)
    # Mapping bounds memory and preserves exact differentiation of the active min.
    return jnp.min(jax.lax.map(lambda p: jnp.sqrt(jnp.min(jnp.sum((p - targets)**2, axis=1)) + 1e-30),
                               points.reshape(-1, 3)))


def coil_constraint(coils_from_x):
    """Pure-coil rows require no equilibrium solves or adjoint right-hand sides."""
    import jax
    import jax.numpy as jnp
    from scipy.optimize import NonlinearConstraint

    fun = jax.jit(lambda x: coil_inequalities(coils_from_x(x)))
    jac = jax.jit(jax.jacrev(fun))
    return NonlinearConstraint(lambda x: np.asarray(fun(jnp.asarray(x))), 0, np.inf,
                               jac=lambda x: np.asarray(jac(jnp.asarray(x))))


# ---- the optimization ------------------------------------------------------------------------------------------------
def parse_args(argv=None):
    """Command-line options; ``--restart`` and ``--seed-run`` replace ``--coils`` by their run's coils."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="new run directory")
    parser.add_argument("--steps", type=int, default=100, help="accepted SLSQP steps")
    parser.add_argument("--coils", type=Path, default=COILS_FILE, help="initial coils (default: the case's COILS_FILE)")
    parser.add_argument("--no-coil-fit", action="store_true", help="use --coils as given, without the seed refit")
    parser.add_argument("--save-every", type=int, default=1, help="save coils and WOUT every this many steps")
    parser.add_argument("--ns", type=int, help="radial resolution (default: RESOLUTION)")
    parser.add_argument("--modes", type=int, nargs=2, metavar=("MPOL", "NTOR"), help="default: RESOLUTION")
    parser.add_argument("--beta", type=float, default=0.0, help="seed beta (BETA_DEFINITION: <beta> or betaxis)")
    parser.add_argument("--bootstrap", action="store_true", help="self-consistent bootstrap current (needs --beta > 0)")
    parser.add_argument("--restart", type=Path, help="continue a run from its input.run, coils.json and wout.nc")
    parser.add_argument("--seed-run", type=Path, help="start from another run's input.run and coils.initial.json "
                        "(its calibrated deck and fitted coils), skipping the bootstrap ramp and the coil fit")
    parser.add_argument("--vc-grid", type=int, default=48, help="virtual-casing grid per field period")
    parser.add_argument("--chunk", type=int, default=8, help="Jacobian columns per batch (memory)")
    parser.add_argument("--quadrature", type=int, nargs=2, metavar=("NT", "NP"),
                        help="virtual-casing singular quadrature (default 4 nfp grid x 2 grid: 384 x 96 for nfp 2; "
                        "grid x grid leaves a ~3e-4 tangential plasma-field error; see ThreeTermFreeBoundaryModel)")
    parser.add_argument("--max-iterations", type=int, help="VMEC iteration cap of every solve (default: the deck's)")
    parser.add_argument("--design-step-scale", type=float, default=DESIGN_STEP_SCALE,
                        help="factor on the design-coordinate scales (default: DESIGN_STEP_SCALE)")
    parser.add_argument("--trial-ftol", type=float, default=1e-3,
                        help="relative cost change at which a trial's boundary steps stop")
    parser.add_argument("--max-boundary-residual", type=float, default=0.05,
                        help="largest interface RMS (B.n, pressure balance or K, over |B|) of a solved trial")
    parser.add_argument("--trial-forward-ftol", type=float, default=1e-11,
                        help="force residual of the trials' equilibria (each accepted point is re-solved to the "
                        "deck's tolerance before its gradient)")
    args = parser.parse_args(argv)
    if args.restart is not None:
        args.coils = args.restart / "coils.json"
    elif args.seed_run is not None:
        args.coils = args.seed_run / "coils.initial.json"
    if args.bootstrap and not args.beta > 0:
        parser.error("--bootstrap needs --beta > 0")
    return args


def fit_coils_to_plasma(coils, wout, inp):
    """Coils whose field with the plasma's own is tangent to a fixed-boundary finite-beta seed."""
    import jax
    import jax.numpy as jnp
    from essos.surfaces import surfacerzfourier_from_boundary
    from scipy.optimize import minimize
    import vmex as vj

    interface = vj.PlasmaVacuumInterface.from_wout(wout, nphi=37, ntheta=32)
    surface = surfacerzfourier_from_boundary(jnp.asarray(inp.rbc), jnp.asarray(inp.zbs), inp.nfp,
                                             nphi=SURFACE_GRID[0], ntheta=SURFACE_GRID[1])
    x0 = jnp.asarray(coils.curves.dofs).ravel()

    def coils_from_u(u):
        return coils.with_dofs(jnp.concatenate((x0 + COIL_STEP * u, coils.dofs_currents)))

    def normal_field_rms(new):
        return weighted_rms(interface.weights, total_normal_field(interface, coil_field(new)))

    def objective(u):
        new = coils_from_u(u)
        rows = jnp.concatenate([coil_inequalities(new), jnp.atleast_1d(
            (surface_distance(new, surface) - COIL_SURFACE_DISTANCE_LIMIT - DISTANCE_MARGIN)
            / COIL_SURFACE_DISTANCE_LIMIT)])
        return (0.5 * (normal_field_rms(new) / COIL_FIT_NORMAL_SCALE)**2
                + 0.5 * COIL_FIT_WEIGHT * jnp.sum(jnp.minimum(rows, 0.0)**2))

    value_and_grad = jax.jit(jax.value_and_grad(objective))
    before = float(normal_field_rms(coils))
    fit = minimize(lambda u: tuple(map(np.asarray, value_and_grad(jnp.asarray(u)))), np.zeros(x0.size), jac=True,
                   method="L-BFGS-B", bounds=[(-5.0, 5.0)] * x0.size,
                   options=dict(maxiter=COIL_FIT_MAXITER, maxcor=20, ftol=1e-15, gtol=1e-12))
    fitted = coils_from_u(jnp.asarray(fit.x))
    print(f"[coil fit] {fit.nit} L-BFGS-B iterations ({fit.message}): (B_coils + B_plasma).n/|B| RMS "
          f"{before:.3e} -> {float(normal_field_rms(fitted)):.3e} on the fixed-boundary seed", flush=True)
    return fitted


def main(argv=None):
    """Build the seed, the coils and the three-term problem, then run SLSQP and save the run."""
    global RESOLUTION
    args = parse_args(argv)
    RESOLUTION = (*(args.modes or RESOLUTION[:2]), args.ns or RESOLUTION[2])
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ["JAX_ENABLE_X64"] = "1"
    # The vacuum-iota solve runs inside a jax.pure_callback; a GPU program XLA compiles there (its Krylov
    # certification fallback) deadlocks in the autotuner, which waits on the device the callback holds.
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_autotune_level=0").strip()
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

    import jax
    import jax.numpy as jnp
    from essos.coils import Coils
    from essos.surfaces import surfacerzfourier_from_boundary
    import vmex as vj
    from vmex import optimize as opt
    from vmex.core import implicit as im

    started = time.monotonic()
    mpol, ntor, _ = RESOLUTION
    inp, redl = seed_input(), None
    if args.max_iterations:  # before the seed is built: its beta ramp and current blend solve with the same cap
        inp = replace(inp, niter_array=np.full(np.size(inp.niter_array), args.max_iterations))
    if args.restart is not None:
        inp = restart_input(args.restart)
        redl = redl_profiles(inp)[1] if args.bootstrap else None
    elif args.seed_run is not None:
        inp = replace(vj.VmecInput.from_file(str(args.seed_run / "input.run")), lfreeb=False)
        redl = redl_profiles(inp)[1] if args.bootstrap else None
    elif args.bootstrap:
        inp, fixed, redl = bootstrap_input(inp, args.beta, "gpu")
    else:
        inp, fixed = finite_beta_input(inp, args.beta, "gpu")
    current_in_solve = args.bootstrap and BOOTSTRAP_IN_SOLVE and BOOTSTRAP_MODEL == "redl"
    profiles = redl_profiles(inp)[0] if current_in_solve else None
    redl = None if current_in_solve else redl
    if args.max_iterations:  # again: a seed-run or restart deck replaced inp
        inp = replace(inp, niter_array=np.full(np.size(inp.niter_array), args.max_iterations))
    inp.to_indata(out / "input.run")

    coils = resize_coils(Coils.from_json(str(args.coils)), COIL_ORDER, N_SEGMENTS)
    if args.restart is None and args.seed_run is None:
        coils = scale_coil_currents(coils, abs(float(fixed.wout.rbtor)))
        if args.beta > 0 and not args.no_coil_fit:
            coils = fit_coils_to_plasma(coils, fixed.wout, inp)
    coils.to_json(str(out / "coils.initial.json"))
    coils0 = Coils.from_json(str(out / "coils.initial.json"))
    current = np.r_[np.asarray(inp.ac_aux_f)[:-1], inp.curtor] if args.bootstrap and not current_in_solve else None
    current_dofs = tuple(np.flatnonzero(np.asarray(coils0.dofs_currents_raw))) if FREE_COIL_CURRENTS else ()
    scales = args.design_step_scale * np.r_[
        [PHIEDGE_STEP] * FREE_PHIEDGE, [CURRENT_STEP] * (0 if current is None else current.size),
        [COIL_CURRENT_STEP] * len(current_dofs),
        COIL_STEP / np.broadcast_to(np.asarray(coils0.curves.scaling), coils0.dofs_curves.shape).ravel()]
    chart = CoilChart(coils0, current_dofs=current_dofs, scales=scales,
                      phiedge=float(inp.phiedge) if FREE_PHIEDGE else None,
                      plasma_current=current, plasma_current_spline=True)
    qs = target_residual()
    mirror = (opt.mirror_ratio,) if MIRROR_LIMIT else ()
    ceiling = (max_abs_iota,) if IOTA_CEILING else ()
    vacuum = vacuum_iota(inp)
    vacuum_floor = ((lambda state, runtime: jnp.min(vacuum(state, runtime))),) if VACUUM_IOTA_FLOOR else ()

    def boundary_surface(state, runtime):
        rmnc, _, _, zmns = im._edge_physical(state, runtime)  # private: no public traced LCFS of a state
        rows, cols = np.asarray(runtime.modes.n) + ntor, np.asarray(runtime.modes.m)
        rbc = jnp.zeros((2 * ntor + 1, mpol)).at[rows, cols].set(rmnc)
        zbs = jnp.zeros((2 * ntor + 1, mpol)).at[rows, cols].set(zmns)
        return surfacerzfourier_from_boundary(rbc, zbs, inp.nfp, nphi=SURFACE_GRID[0], ntheta=SURFACE_GRID[1])

    def loss(state, runtime, x):
        residuals = qs.residuals_state(state, runtime)
        return 0.5 * jnp.vdot(residuals, residuals)

    def clearance(state, runtime, x):
        return surface_distance(chart.coils_from_x(x), boundary_surface(state, runtime))

    def aspect(state, runtime, x):
        return opt.aspect_ratio(state, runtime)

    problem = opt.FreeBoundaryProblem.from_loss(
        inp, loss, chart.x0, field_from_parameters=chart, plasma_from_parameters=chart.plasma_params_at,
        scales=chart.scales, names=chart.dof_names,
        quantities=(min_abs_iota, opt.major_radius, opt.axis_field_strength, *mirror, *ceiling, *vacuum_floor,
                    *([bootstrap_mismatch(inp, redl, "gpu")] if redl is not None else [])),
        parameter_quantities=(clearance, aspect), boundary_condition="three_term",
        three_term_options=dict(nphi=args.vc_grid, ntheta=args.vc_grid, chunk=args.chunk,
                                quadrature=args.quadrature or (4 * int(inp.nfp) * args.vc_grid, 2 * args.vc_grid),
                                trial_ftol=args.trial_forward_ftol, bootstrap=profiles,
                                bootstrap_helicity=REDL_HELICITY, boundary_ftol=args.trial_ftol,
                                boundary_max_residual=args.max_boundary_residual))
    model = problem.model
    print(f"[model] {model.n_boundary} boundary and {model.x0.size - model.n_boundary} bootstrap-current coordinates, "
          f"{chart.size} design coordinates; built in {time.monotonic() - started:.0f} s", flush=True)

    aspect_lower, aspect_upper = ASPECT_RANGE
    width, b_width = RADIUS_TOLERANCE - RADIUS_MARGIN, B_AXIS_TOLERANCE - B_AXIS_MARGIN
    nredl = int(redl is not None)
    lower, upper, row_scales = zip(
        (IOTA_FLOOR + IOTA_MARGIN, np.inf, IOTA_FLOOR),
        (RADIUS_TARGET - width, RADIUS_TARGET + width, RADIUS_TOLERANCE),
        (B0 - b_width, B0 + b_width, B_AXIS_TOLERANCE),
        *([(-np.inf, MIRROR_LIMIT - MIRROR_MARGIN, MIRROR_LIMIT)] if mirror else []),
        *([(-np.inf, IOTA_CEILING - IOTA_MARGIN, IOTA_FLOOR)] if ceiling else []),
        *([(VACUUM_IOTA_FLOOR + IOTA_MARGIN, np.inf, VACUUM_IOTA_FLOOR)] if vacuum_floor else []),
        *[(-np.inf, REDL_TOLERANCE, REDL_TOLERANCE)] * nredl,
        (COIL_SURFACE_DISTANCE_LIMIT + DISTANCE_MARGIN, np.inf, COIL_SURFACE_DISTANCE_LIMIT),
        (aspect_lower, aspect_upper, 0.5 * (aspect_upper - aspect_lower)))
    coil_rows = coil_constraint(chart.coils_from_x)
    constraints = [problem.nonlinear_constraint(list(lower), list(upper), scales=list(row_scales)), coil_rows]

    def save(tag):
        x = problem.accepted.parameters
        chart.coils_from_x(jnp.asarray(x)).to_json(str(out / f"coils{tag}.json"))
        vj.write_wout(str(out / f"wout{tag}.nc"), problem.equilibrium_from_x(x).wout)

    last = dict(time=time.monotonic())

    def log_step():
        x = problem.accepted.parameters
        h = problem.constraint_values(x)
        res = problem.boundary_residual(x)
        now = time.monotonic()
        state_x, params_x = problem.state(x)
        runtime_x = im.runtime_from_params(params_x, model.cfg)
        vac = np.asarray(vacuum(state_x, runtime_x))
        redl_row = 3 + len(mirror) + len(ceiling) + len(vacuum_floor)
        # "qa" holds the target residual of every case (QA, QH or QI), kept for compatibility
        row = dict(step=problem.accepted_step, qa=2 * problem.fun(x), min_abs_iota=float(h[0]),
                   geometric_axis_iota=float(abs(opt.axis_iota(state_x, runtime_x))),
                   vacuum_iota_axis=float(vac[0]), vacuum_iota_min=float(vac.min()), vacuum_iota_edge=float(vac[-1]),
                   major_radius_m=float(h[1]), b_axis_t=float(h[2]),
                   beta=float(opt.volume_average_beta(state_x, runtime_x)),
                   coil_currents_a=np.asarray(chart.base_currents_at(jnp.asarray(x))).tolist(),
                   **({"redl_mismatch": float(h[redl_row])} if nredl else {}),
                   **({"redl_max_relative": model.bootstrap_residual(state_x, params_x)} if current_in_solve else {}),
                   coil_surface_distance_m=float(h[-2]), aspect=float(h[-1]),
                   coil_minimum_scaled_slack=float(np.min(coil_rows.fun(x))),
                   phiedge_factor=float(chart.phiedge_at(jnp.asarray(x)) / chart.phiedge) if FREE_PHIEDGE else 1.0,
                   bn=res.normal, pressure_balance=res.pressure, sheet_current=res.sheet_current,
                   step_seconds=now - last["time"], elapsed_seconds=now - started, **problem.stats,
                   peak_gpu_gib=(jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / 2**30)
        last.update(time=now)
        with open(out / "metrics.jsonl", "a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(f"[step {row['step']}] {TARGET_NAME}={row['qa']:.6e} iota={row['min_abs_iota']:.5f} "
              f"vacuum iota={row['vacuum_iota_min']:.4f} (axis {row['vacuum_iota_axis']:.4f}) "
              f"B_axis={row['b_axis_t']:.5f} R={row['major_radius_m']:.5f} aspect={row['aspect']:.4f} "
              f"clearance={row['coil_surface_distance_m']:.4f} "
              f"coil_slack={row['coil_minimum_scaled_slack']:.4f} "
              + (f"redl={row['redl_mismatch']:.2e} " if nredl else "")
              + (f"redl(max rel)={row['redl_max_relative']:.1e} " if current_in_solve else "")
              + f"B.n={res.normal:.1e} K={res.sheet_current:.1e} {row['step_seconds']:.1f}s "
              f"(trials {row['trials']}, boundary evaluations {row['boundary_evaluations']}, "
              f"peak {row['peak_gpu_gib']:.1f} GiB)", flush=True)
        if row["step"] % args.save_every == 0:
            save(f".step{row['step']}")

    log_step()
    while True:  # a rejected trial ends an SLSQP call; restart it from the accepted point
        before = problem.accepted_step
        result = opt.minimize(problem, x0=problem.accepted.parameters, method="SLSQP", constraints=constraints,
                              callback=lambda x: log_step(),
                              options=dict(maxiter=args.steps - before, ftol=OPTIMIZER_FTOL))
        if (result.stop_reason != "equilibrium_trial_rejected" or problem.accepted_step == before
                or problem.accepted_step >= args.steps):
            break
    save("")
    summary = dict(accepted_steps=problem.accepted_step, success=bool(result.success), message=str(result.message),
                   stop_reason=result.stop_reason,
                   failed_trials=problem.metadata["holder"]["failed_trials"],
                   elapsed_seconds=time.monotonic() - started)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
