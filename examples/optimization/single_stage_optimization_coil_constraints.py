#!/usr/bin/env python
r"""Fixed-boundary single-stage coil optimization with hard coil constraints.

The fixed-boundary arm of ``single_stage_free_boundary_optimization_coil_constraints.py``:
the same cases (``COIL_CASE``, default ``ellipse5``; the case table is in that
script's docstring), limits and seeds. The design variables are the boundary
modes up to ``MAX_MODE`` (RBC(0,0) fixed), the coil currents (``FREE_COIL_CURRENTS``,
no bounds) and shapes and, at ``--beta`` > 0 with ``FREE_PHIEDGE``, PHIEDGE; with ``--bootstrap`` also
the current spline values and CURTOR. Every trial solves the fixed-boundary
equilibrium, and SLSQP minimizes

    J = (1/2) |r|^2 + (1/2) NORMAL_FIELD_WEIGHT rms(B.n/|B|)^2

with r the quasisymmetry or constructed QI residual of the case. The B.n term
keeps the prescribed boundary close to a flux surface of the coils. The hard
inequalities are, in ``main``'s row order:

- plasma rows: min |iota| >= ``IOTA_FLOOR``, the aspect band, the major-radius
  band, the mirror ratio, max |iota| <= ``IOTA_CEILING`` and the boundary's
  vacuum |iota| >= ``VACUUM_IOTA_FLOOR`` where the case sets them; at beta > 0
  the total-field B.n limit, the field-strength band and the axis |B| band;
  with ``--bootstrap`` the bootstrap mismatch <= ``REDL_TOLERANCE``
  (``run_bootstrap_in_solve`` orders its plasma rows differently and solves the
  current instead of holding this row);
- coil rows: ``coil_inequalities`` (per-coil length, curvature and mean
  squared curvature, coil-coil distance, self-clearance and minimum speed), the
  coil-to-plasma clearance and, in vacuum, rms B.n/|B| <= ``NORMAL_FIELD_CONSTRAINT``.

A rejected equilibrium trial counts the plasma rows as violated. In vacuum
PHIEDGE only scales the field, so the coils may enclose any flux; their
toroidal flux through the boundary over PHIEDGE is logged as ``flux_ratio``
(``benchmarks/coil_constraints_postprocess.py --match-flux`` rescales the
currents by it).

``--beta`` uses the free arm's fixed pressure p ~ 1 - s (<beta>, or on-axis
beta for ``ellipse5-beta7``) with zero net current. The plasma then carries a
field of its own, so the B.n limit and objective term apply to the total
(B_coils + B_plasma).n/|B|, with B_plasma from virtual casing on every trial,
differentiated through the equilibrium. Outside the plasma R B_phi is set by
the coils alone (Ampere's law), so a band of ``FIELD_STRENGTH_TOLERANCE``
holds the edge R B_phi at the coils' linked mu0 I / 2 pi, and a band of
``B_AXIS_TOLERANCE`` holds the axis |B| at ``B0``. Pressure balance and the
total and coil-only B.n are checked at the end on a 61 x 64 grid.
``--bootstrap`` (with ``--beta``) adds the free arm's self-consistent bootstrap
current (Redl, or DKX for ``qi6-beta*``); with ``BOOTSTRAP_IN_SOLVE`` (Redl)
the current is solved with every equilibrium instead (``run_bootstrap_in_solve``).

It needs ESSOS (``pip install "vmex[coils]"``), virtual casing for ``--beta``
(``vmex[freeb]``), DKX (``vmex[kinetic]``) for ``qi6-beta*`` with
``--bootstrap``, and a GPU for the default resolution:

    python single_stage_optimization_coil_constraints.py --steps 5 --output runs/fixed
    COIL_CASE=qi python single_stage_optimization_coil_constraints.py --steps 5 --output runs/fixed-qi

Outputs in ``--output``: ``input.run``, ``coils.initial.json``, one
``metrics.jsonl`` line per SLSQP iteration, ``coils.stepN.json`` /
``wout.stepN.nc`` every ``--save-every`` iterations, and the final
``coils.json``, ``wout.nc`` and ``summary.json`` (with the endpoint
diagnostics). Continue a run with ``--restart <run>``, or with ``--coils`` and
``--wout`` (the boundary restarts from the WOUT, SLSQP from an identity Hessian).
"""

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import time

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
B0 = 1.0                           # T: |B| on the magnetic axis (opt.axis_field_strength), the seed's and,
                                   # at --beta > 0, a row
FREE_PHIEDGE = True                # PHIEDGE is a design variable at --beta > 0 if set, fixed otherwise
FREE_COIL_CURRENTS = True          # every base-coil current is a design variable, with no bounds
COIL_CURRENT_STEP = 0.06           # coordinate scale of the relative coil-current change

# --bootstrap: Landreman-Buller-Drevlak kinetic profiles ne ~ 1 - s^5, Te = Ti ~ 1 - s, at the
# beta and collisionality of a Helios-like reactor (n T ~ B^2 and nu* ~ n R / T^2 held), and a
# self-consistent bootstrap current: CURRENT_KNOTS spline values (the last one fixed) and CURTOR
# are design variables, and the mismatch sum_j R_j^2 against the BOOTSTRAP_MODEL current (Redl
# or DKX, in Redl's normalized form) is held under REDL_TOLERANCE. The seed's Picard loop is Redl's.
# DKX: the current is a line segment through a value on every half-mesh surface (those values,
# the axis one and CURTOR are design variables), its seed DKX's on the same surfaces.
REACTOR_R0, REACTOR_B0, REACTOR_N0, REACTOR_T0 = 8.0, 6.0, 1.5e20, 15.0e3   # m, T, 1/m^3, eV
REDL_SURFACES = None              # None: every VMEC half-grid surface, as simsopt's RedlGeomVmec
REDL_N_LAMBDA, REDL_TOLERANCE = 32, 1e-3
PICARD_ITERATIONS, PICARD_TOLERANCE, PICARD_RELAX = 10, 1e-3, 1.0
BOOTSTRAP_BETA_STEP = 0.01        # a larger --beta is ramped in with its bootstrap current, in steps of at most this
BOOTSTRAP_BETA_START = None       # set: the ramp first doubles beta from this, so a seed with little vacuum
                                  # iota stays under its eps iota^2 limit while the bootstrap current grows
OHMIC_CURRENT = None              # set (A): the seed ramp's prescribed current, blended into the bootstrap one
CURRENT_KNOTS, CURRENT_STEP = 8, 0.05   # step relative to the largest knot value and to |CURTOR|
# Redl cases: the current is no design variable but solved with every equilibrium (ThreeTermFreeBoundaryModel
# bootstrap=, both arms), Redl-self-consistent on every half-grid surface, with I'(0) = 0 (a quasisymmetric bootstrap
# current density vanishes on the axis with the trapped fraction, f_t ~ eps^(1/2)).
BOOTSTRAP_IN_SOLVE = False
SEED = None                        # (nfp, aspect, b / a_eff): rotating ellipse replacing the deck's boundary
HELICITY = (1, 0)                  # quasisymmetry (M, N); None minimizes the constructed QI residual
TARGET_NAME = "QA"
QI_SURFACES = tuple(i / 5 for i in range(1, 6))
QI_OPTIONS = dict(mboz=12, nboz=12, nphi=61, nalpha=18, n_bounce=21)  # as examples/optimization/QI_optimization.py
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
    # Damped: the current dominates iota (bootstrap.self_consistent_bootstrap).
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
if COIL_LIMIT_FACTORS is not None:
    _radius = RADIUS_TARGET / ASPECT_RANGE[0] + COIL_SURFACE_DISTANCE_LIMIT
    LENGTH_LIMIT = COIL_LIMIT_FACTORS[0] * 2 * math.pi * _radius
    CURVATURE_LIMIT, MSC_LIMIT = COIL_LIMIT_FACTORS[1] / _radius, COIL_LIMIT_FACTORS[2] / _radius**2

NORMAL_FIELD_CONSTRAINT = 0.008   # limit on the area-weighted RMS B.n/|B| (total B at beta > 0)
FIELD_STRENGTH_TOLERANCE = 0.005  # beta > 0: relative band on edge R B_phi around the coils' mu0 I / 2 pi
# Sampling of the coil inequalities, after the per-case COIL_ORDER.
CURVATURE_POINTS = max(512, 64 * COIL_ORDER)  # quadrature points per base coil: length, curvature, speed
DISTANCE_POINTS = max(128, 16 * COIL_ORDER)   # points per coil of the coil-coil and coil-surface distances
SELF_CLEARANCE = 1e-6                         # m, numerical nonintersection guard
MIN_SPEED = 1e-4                              # lower limit on each coil's speed |gamma'|

MAX_MODE = 8                       # boundary modes varied; RBC(0,0) stays fixed
ESS_ALPHA = 1.2
# SLSQP's first step has an identity Hessian, so the seeded (SEED) cases scale their boundary coordinates down:
# the QH and QI residuals start near 1, 50 times the QA seed's.
BOUNDARY_STEP = 0.1 if SEED is None else 0.02
NORMAL_FIELD_WEIGHT = 1.0e3
OPTIMIZER_FTOL = 1e-10
VC_DIGITS = 4                      # significant digits of the virtual-casing plasma field
NPHI, NTHETA = 37, 32              # toroidal x poloidal points of the B.n and clearance surfaces
VC_GRID_BOOTSTRAP = 48             # BOOTSTRAP_IN_SOLVE: the three-term arm's virtual-casing grid
                                   # (quadrature 4 nfp 48 x 96)

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

    ne = n0 (1 - s^5) and Te = Ti = T0 (1 - s), so p = 2 e ne Te ~ (1 - s)(1 - s^5). n0 and T0 start at the
    Helios-like reactor's collisionality nu* ~ n R / T^2 and beta ~ n T / B^2 moved to this R0 and B0, and follow
    the beta calibration as n ~ p^(2/3), T ~ p^(1/3), which keeps nu*. A Picard loop then makes the current
    Redl's, and it is resampled onto ``CURRENT_KNOTS`` spline knots (DKX: instead DKX's on every half-mesh
    surface, ``dkx_current``); with ``BOOTSTRAP_IN_SOLVE`` (Redl) it is solved in the run's own representation,
    Redl's on every half-mesh surface (``bootstrap.HalfMeshCurrent``), so the first step starts self-consistent.
    Above ``BOOTSTRAP_BETA_STEP`` beta is ramped in steps of at most that size, each step's pressure ramp carrying
    the previous step's bootstrap current, so its transform holds the equilibrium together as beta rises; with
    ``BOOTSTRAP_BETA_START`` the ramp first doubles beta from that value. With ``OHMIC_CURRENT`` (a
    near-axisymmetric seed) beta is instead ramped at that prescribed current and the current then blended into
    Redl's. Returns the input, the equilibrium and the Redl mismatch.
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
        """Redl's self-consistent current for ``inp`` after at most ``n_iter`` Picard steps."""
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
            inp, _ = finite_beta_input(inp, stage_beta, device, am=(1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0))
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
        inp, _ = finite_beta_input(inp, stage_beta, device, am=(1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0))
        profiles, redl = redl_profiles(inp)
        picard = picard_at(inp, PICARD_ITERATIONS)
        inp = picard.input
        if stages > 1:
            print(f"bootstrap ramp {stage}/{stages}: beta {stage_beta:.4f}, CURTOR = {float(inp.curtor):.1f} A, "
                  f"Picard {picard.iterations} iterations (converged {picard.converged})", flush=True)
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
    w = fixed.wout
    print(f"Redl seed: n0 = {n0:.4e} 1/m^3, T0 = {t0:.1f} eV, Picard {picard.iterations} iterations "
          f"(converged {picard.converged}), CURTOR = {float(inp.curtor):.1f} A, mismatch = "
          f"{float(redl.total(w)):.3e}; <beta> = {float(w.betatotal):.4%}, iota = "
          f"{float(min_abs_iota(fixed.state, fixed.runtime)):.4f}.."
          f"{float(max_abs_iota(fixed.state, fixed.runtime)):.4f}, edge R B_phi = {abs(float(w.rbtor)):.5f} T m")
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
    device = jax.devices()[0]  # pinned, as ThreeTermFreeBoundaryModel: unpinned, its certification ran on the host CPU
    cfg = im.make_config(vacuum, multigrid=True, hot_restart=True, device=device)
    base = im.params_from_input(vacuum, device=device)
    im.runtime_from_params(base, cfg)  # its setup, built here: run_setup cannot be traced inside a jit

    def iota(state, ctx):
        """The vacuum |iota| of ``state``'s plasma boundary: axis, half-mesh surfaces and edge."""
        rmnc, _, _, zmns = im._edge_physical(state, ctx)  # private: no public traced LCFS of a state
        rmnc, zmns = jax.device_put((rmnc, zmns), device)  # a fixed-boundary state may live on the host
        rows, cols = np.asarray(ctx.modes.n) + int(inp.ntor), np.asarray(ctx.modes.m)
        params = replace(base, rbc=jnp.zeros_like(base.rbc).at[rows, cols].set(rmnc),
                         zbs=jnp.zeros_like(base.zbs).at[rows, cols].set(zmns))
        vstate = im.solve_implicit(params, cfg)
        vctx = im.runtime_from_params(params, cfg)
        half = _iotas_half(vstate, vctx)[1:]
        return jnp.abs(jnp.concatenate([opt.axis_iota(vstate, vctx)[None], half,
                                        1.5 * half[-1:] - 0.5 * half[-2:-1]]))

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
    print(f"DKX bootstrap mismatch of the fixed-boundary seed: "
          f"{float(mismatch(seed.state, seed.solver_context)):.3e}", flush=True)
    return mismatch


def dkx_kinetic(inp):
    """DKX's bootstrap ``<j.B>`` for ``inp``'s kinetic profiles on the ``DKX_SURFACES`` rows (needs ``dkx``)."""
    from dkx.bootstrap import KineticBootstrapMismatch

    surfaces = redl_surfaces() if DKX_SURFACES is None else DKX_SURFACES
    return KineticBootstrapMismatch(redl_profiles(inp)[0], surfaces=surfaces,
                                    collision_operator=DKX_COLLISION_OPERATOR, mboz=QI_OPTIONS["mboz"],
                                    nboz=QI_OPTIONS["nboz"])


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

    def enclosed(z):
        """The enclosed current profile of ``inp`` at ``z``."""
        return current(inp.pcurr_type, inp.ac, inp.ac_aux_s, inp.ac_aux_f, z, bloat=inp.bloat)

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


def restart_input(run):
    """A finished run's ``input.run`` with its final WOUT's boundary, PHIEDGE and current (``--restart``)."""
    import vmex as vj

    inp, w = vj.VmecInput.from_file(run / "input.run"), vj.read_wout(run / "wout.nc")
    return current_from_wout(replace(boundary_from_wout(inp, w), phiedge=float(w.phi[-1]), lfreeb=False), w)


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
    import jax.numpy as jnp
    from essos.coils import Coils

    r0, phi = RADIUS_TARGET, np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    points = jnp.asarray(r0 * np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=-1))
    field = np.asarray(coil_field(coils)(points))
    linked = abs(r0 * float(np.mean(-np.sin(phi) * field[:, 0] + np.cos(phi) * field[:, 1])))
    return Coils(coils.curves, coils.dofs_currents_raw * (rbtor / linked), currents_scale=coils.currents_scale)


def coil_field(coils):
    """points (..., 3) -> the coils' Biot-Savart field B (..., 3)."""
    import jax
    from essos.fields import BiotSavart

    biot_savart = BiotSavart(coils)
    return lambda points: jax.vmap(biot_savart.B)(points.reshape(-1, 3)).reshape(points.shape)


def linked_rbtor_of(r0):
    r"""coils -> the poloidal current they link, mu0 I / 2 pi = (1/2 pi) \oint R B_phi dphi on R = ``r0``, Z = 0 [T m].

    Differentiable in the coil currents and shapes.
    """
    import jax.numpy as jnp

    phi = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    loop = jnp.asarray(r0 * np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=-1))
    tangent = jnp.asarray(np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=-1))
    return lambda coils: jnp.abs(r0 * jnp.mean(jnp.sum(coil_field(coils)(loop) * tangent, axis=-1)))


def edge_rbtor(state, ctx):
    """The equilibrium's |edge R B_phi| [T m], from its covariant fields."""
    import jax.numpy as jnp
    from vmex.core.fields import surface_currents
    from vmex.core.statephysics import _field_chain  # private: the covariant fields behind the edge R B_phi

    fields = _field_chain(state, ctx)[3]
    currents = surface_currents(bsubu=fields.bsubu, bsubv=fields.bsubv, trig=ctx.trig,
                                s=jnp.asarray(ctx.setup.s_full), signgs=ctx.setup.signgs)
    return jnp.abs(currents.rbtor)


def weighted_rms(weights, values):
    """sqrt(sum weights values^2): the RMS for area weights that sum to one."""
    import jax.numpy as jnp

    return jnp.sqrt(jnp.sum(weights * values**2))


def total_normal_field(interface, field):
    """(B_coils + B_plasma).n/|B| on a ``vmex.PlasmaVacuumInterface``, B_coils = ``field``."""
    import jax.numpy as jnp

    return interface.bnormal_residual(field) / jnp.linalg.norm(interface.total_B_out(field), axis=0)


def normal_field_rms(coils, surface):
    """Area-weighted RMS of the coil-only B.n/|B| on an ESSOS surface."""
    import jax.numpy as jnp

    field = coil_field(coils)(surface.gamma)
    normal = jnp.sum(field * surface.unitnormal, axis=2) / jnp.linalg.norm(field, axis=2)
    return weighted_rms(surface.area_element / jnp.sum(surface.area_element), normal)


def boundary_diagnostics(wout, coils, nphi=61, ntheta=64):
    """Interface diagnostics of a WOUT in its coils' field; virtual casing is planned on this grid.

    Returns beta, the RMS/max of (B_coils + B_plasma).n/|B|, the RMS of the
    coil-only B.n/|B|, and the RMS of the pressure-balance residual
    (|B_out|^2 - |B_in|^2 - 2 mu0 p) / |B_in|^2.
    """
    import jax.numpy as jnp
    import vmex as vj

    field = coil_field(coils)
    interface = vj.PlasmaVacuumInterface.from_wout(wout, nphi=nphi, ntheta=ntheta)
    weights = interface.weights
    total = total_normal_field(interface, field)
    coil_only = interface.external_Bn(field) / jnp.linalg.norm(interface.external_B(field), axis=0)
    balance = interface.pressure_balance_residual(field) / interface.Bin_mag2
    return dict(beta=float(wout.betatotal), normal_field_rms=float(weighted_rms(weights, total)),
                normal_field_max=float(jnp.max(jnp.abs(total))),
                coil_normal_field_rms=float(weighted_rms(weights, coil_only)),
                pressure_balance_rms=float(weighted_rms(weights, balance)))


# ---- coil inequalities -----------------------------------------------------------------------------------------------
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
    uv = jnp.einsum('ik,jk->ij', u, v)
    uw, vw = jnp.sum(u[:, None] * w, axis=-1), jnp.sum(v[None, :] * w, axis=-1)
    aa, bb = jnp.maximum(aa, 1e-30), jnp.maximum(bb, 1e-30)
    den = aa * bb - uv * uv
    safe = jnp.where(den > 1e-24, den, 1.0)
    s, t = (uv * vw - bb * uw) / safe, (aa * vw - uv * uw) / safe

    def distance(s, t):
        """Squared distance between the points at parameters ``s`` and ``t`` of each segment pair."""
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
    """Scaled coil rows (c >= 0 feasible): length, peak curvature, MSC, coil-coil and self distance, speed."""
    import jax.numpy as jnp

    m = coil_metrics(coils)
    return jnp.concatenate(((LENGTH_LIMIT - LENGTH_MARGIN - m['length']) / LENGTH_LIMIT,
                            (CURVATURE_LIMIT - CURVATURE_MARGIN - m['peak']) / CURVATURE_LIMIT,
                            (MSC_LIMIT - MSC_MARGIN - m['msc']) / MSC_LIMIT,
                            jnp.atleast_1d((m['coil_distance'] - COIL_DISTANCE_LIMIT - DISTANCE_MARGIN)
                                           / COIL_DISTANCE_LIMIT),
                            jnp.atleast_1d((m['self_distance'] - SELF_CLEARANCE) / COIL_DISTANCE_LIMIT),
                            (m['min_speed'] - MIN_SPEED) / LENGTH_LIMIT))


def surface_distance(coils, surface):
    """Sampled coil-to-moving-surface clearance; JAX differentiates both sides."""
    import jax
    import jax.numpy as jnp

    points = resampled(coils, DISTANCE_POINTS).gamma
    targets = surface.gamma.reshape(-1, 3)
    # Mapping bounds memory and preserves exact differentiation of the active min.
    return jnp.min(jax.lax.map(lambda p: jnp.sqrt(jnp.min(jnp.sum((p - targets)**2, axis=1)) + 1e-30),
                               points.reshape(-1, 3)))


# ---- the bootstrap current solved in every equilibrium --------------------------------------------------------------
def run_bootstrap_in_solve(args, inp, coils0, out, started):
    """The single stage with the bootstrap current solved in every equilibrium (``BOOTSTRAP_IN_SOLVE``).

    The design variables are the boundary modes up to ``MAX_MODE`` (RBC(0,0) fixed), PHIEDGE (with ``FREE_PHIEDGE``),
    the coil currents and shapes, as in ``main``; the current is no design variable and no constraint row: every trial
    solves the fixed-boundary equilibrium together with a current that is Redl's on every half-grid surface
    (``ThreeTermFreeBoundaryModel(fixed_boundary=True, bootstrap=)``, ``I'(0) = 0``). The objective and the
    other rows are ``main``'s. A row ``h`` then has the design gradient

        dh/dp = dh/dp|_current - lambda^T dR/dp,   lambda = J_c (J_c^T J_c)^-1 dh/dc,

    with ``R`` the current's self-consistency rows, ``J_c`` their Jacobian in the current values and
    ``dR/dp`` along the boundary and PHIEDGE (state tangents); coil-only rows are differentiated directly.
    """
    import jax
    import jax.numpy as jnp
    from essos.surfaces import surfacerzfourier_from_boundary
    from scipy.optimize import minimize
    import vmex as vj
    from vmex import optimize as opt
    from vmex.core import implicit as im
    from vmex.core import virtual_casing as vc
    from vmex.core.freeboundary_vc import ThreeTermFreeBoundaryModel
    from vmex.core.optimize import _ess_scale, boundary_arrays_from_x, pack_boundary

    checking = bool(args.check_gradient)  # central differences need every trial solved tight
    model = ThreeTermFreeBoundaryModel(inp, bootstrap=redl_profiles(inp)[0], bootstrap_helicity=REDL_HELICITY,
                                       fixed_boundary=True, chunk=args.chunk, trial_ftol=None if checking else 1e-11)
    current_ftol = 1e-14 if checking else 1e-8
    fixed, nc = model.fixed, model.x0.size
    vary_phiedge = FREE_PHIEDGE
    phiedge0 = float(fixed.phiedge)
    xb0 = pack_boundary(fixed, MAX_MODE)
    nb = xb0.size
    npl = nb + int(vary_phiedge)                    # plasma design coordinates: boundary, then PHIEDGE
    ncur = np.size(coils0.dofs_currents) * FREE_COIL_CURRENTS  # relative base-coil current changes, after PHIEDGE
    x_coils0 = np.r_[np.zeros(ncur), np.asarray(coils0.curves.dofs).ravel()]
    x0 = np.concatenate([xb0, np.zeros(int(vary_phiedge)), x_coils0])
    scales = np.concatenate([BOUNDARY_STEP * _ess_scale(fixed, MAX_MODE, ESS_ALPHA), [0.05] * int(vary_phiedge),
                             np.full(ncur, COIL_CURRENT_STEP), np.full(x_coils0.size - ncur, COIL_STEP)])
    print(f"[model] {nc} bootstrap-current coordinates, {npl} plasma and {x_coils0.size} coil design coordinates",
          flush=True)

    def plasma_params(params, x):
        """``params`` with the boundary and PHIEDGE of the design vector ``x``."""
        rbc, zbs = boundary_arrays_from_x(fixed, x[:nb], MAX_MODE)
        params = replace(params, rbc=rbc, zbs=zbs)
        return replace(params, phiedge=phiedge0 * (1.0 + x[nb])) if vary_phiedge else params

    def coils_from_x(x):
        """The coils of the design vector ``x``."""
        currents = coils0.dofs_currents * (1.0 + x[npl:npl + ncur]) if ncur else coils0.dofs_currents
        return coils0.with_dofs(jnp.concatenate((x[npl + ncur:], currents)))

    def surface_from_x(x):
        """The ESSOS plasma boundary of the design vector ``x``."""
        rbc, zbs = boundary_arrays_from_x(fixed, x[:nb], MAX_MODE)
        return surfacerzfourier_from_boundary(rbc, zbs, fixed.nfp, nphi=VC_GRID_BOOTSTRAP, ntheta=VC_GRID_BOOTSTRAP)

    state0, _ = model.seed
    ctx0 = im.runtime_from_params(model.params0, model.cfg)
    # A fixed singular quadrature (4 nfp nphi x 2 ntheta, the three-term arm's default): the planner's error estimate
    # alone can exhaust a GPU on an optimized boundary.
    precision = vc.plan_vc_precision(vc.surface_field_data_from_state(fixed, state0, runtime=ctx0,
                                                                      nphi=VC_GRID_BOOTSTRAP,
                                                                      ntheta=VC_GRID_BOOTSTRAP),
                                     digits=VC_DIGITS, quad_nt=4 * int(fixed.nfp) * VC_GRID_BOOTSTRAP,
                                     quad_np=2 * VC_GRID_BOOTSTRAP)
    linked_rbtor = linked_rbtor_of(float(fixed.rbc[fixed.ntor, 0]))
    vacuum = vacuum_iota(fixed)

    qs = target_residual()
    aspect_lower, aspect_upper = ASPECT_RANGE
    aspect_scale = 0.5 * (aspect_upper - aspect_lower)
    width, b_width = RADIUS_TOLERANCE - RADIUS_MARGIN, B_AXIS_TOLERANCE - B_AXIS_MARGIN

    def total_normal_field_rms(coils, state, ctx):
        """Area-weighted RMS of (B_coils + B_plasma).n/|B| on the boundary."""
        data = vc.surface_field_data_from_state(fixed, state, runtime=ctx, nphi=VC_GRID_BOOTSTRAP,
                                                ntheta=VC_GRID_BOOTSTRAP)
        interface = vc.PlasmaVacuumInterface.from_surface_data(data, digits=VC_DIGITS, precision=precision)
        normal = interface.bnormal_residual(coil_field(coils)) / jnp.linalg.norm(data.B_total, axis=0)
        return weighted_rms(interface.weights, normal)

    def rbtor_ratio(state, ctx, coils):
        """Equilibrium edge R B_phi over the coils' linked mu0 I / 2 pi."""
        return edge_rbtor(state, ctx) / linked_rbtor(coils)

    def normal_field(state, params, x):
        """The total B.n rms of the design vector ``x``, as a 1-vector."""
        return total_normal_field_rms(coils_from_x(x), state, im.runtime_from_params(params, model.cfg))[None]

    def constraint_rows(state, params, x, nf, with_vacuum=True):
        """The plasma rows (scaled so c >= 0 is feasible), the vacuum-iota row only ``with_vacuum``."""
        ctx = im.runtime_from_params(params, model.cfg)
        iota, aspect, radius = min_abs_iota(state, ctx), opt.aspect_ratio(state, ctx), opt.major_radius(state, ctx)
        strength = rbtor_ratio(state, ctx, coils_from_x(x)) - 1.0
        b_axis = opt.axis_field_strength(state, ctx)
        rows = [(iota - IOTA_FLOOR - IOTA_MARGIN) / IOTA_FLOOR,
                (aspect - aspect_lower) / aspect_scale, (aspect_upper - aspect) / aspect_scale,
                (radius - RADIUS_TARGET + width) / RADIUS_TOLERANCE,
                (RADIUS_TARGET + width - radius) / RADIUS_TOLERANCE,
                (b_axis - B0 + b_width) / B_AXIS_TOLERANCE, (B0 + b_width - b_axis) / B_AXIS_TOLERANCE]
        if MIRROR_LIMIT:
            rows.append((MIRROR_LIMIT - MIRROR_MARGIN - opt.mirror_ratio(state, ctx)) / MIRROR_LIMIT)
        if IOTA_CEILING:
            rows.append((IOTA_CEILING - IOTA_MARGIN - max_abs_iota(state, ctx)) / IOTA_FLOOR)
        if VACUUM_IOTA_FLOOR and with_vacuum:
            rows.append(vacuum_row(state, params))
        rows += [1.0 - nf / NORMAL_FIELD_CONSTRAINT, (FIELD_STRENGTH_TOLERANCE - strength) / FIELD_STRENGTH_TOLERANCE,
                 (FIELD_STRENGTH_TOLERANCE + strength) / FIELD_STRENGTH_TOLERANCE]
        return rows

    def vacuum_row(state, params):
        """The scaled vacuum-iota floor row."""
        ctx = im.runtime_from_params(params, model.cfg)
        return (jnp.min(vacuum(state, ctx)) - VACUUM_IOTA_FLOOR - IOTA_MARGIN) / VACUUM_IOTA_FLOOR

    def quantities(state, params, x, nf=None):
        """``[target residuals..., B.n rms, plasma rows...]`` (``nf`` fixes B.n)."""
        ctx = im.runtime_from_params(params, model.cfg)
        nf = normal_field(state, params, x)[0] if nf is None else nf
        return jnp.concatenate([qs.residuals_state(state, ctx),
                                jnp.stack([nf] + constraint_rows(state, params, x, nf))])

    nq = int(np.asarray(qs.residuals_state(state0, ctx0)).size)
    vacuum_index = 7 + bool(MIRROR_LIMIT) + bool(IOTA_CEILING)  # the vacuum-iota row among the plasma rows
    nrows = vacuum_index + bool(VACUUM_IOTA_FLOOR) + 3
    quantities_jit = jax.jit(quantities)
    # order maps quantity_pullback's parts (the objective, the other plasma rows, the vacuum-iota row) to row order.
    others = [j for j in range(nrows) if not (VACUUM_IOTA_FLOOR and j == vacuum_index)]
    order = np.argsort(np.r_[0, 1 + np.asarray(others), [1 + vacuum_index] * bool(VACUUM_IOTA_FLOOR)])

    @jax.jit
    def quantity_pullback(state, mask, params, objective, x):
        """Each row's pullback (the objective's cotangent ``objective``, then a unit one per plasma row) with B.n
        held fixed, and each row's weight on B.n.

        Every row runs only its own part's reverse pass: the quasisymmetry residuals for the objective, the
        vacuum-iota solve for its row, the other plasma rows together.  B.n goes forward along the linearization's
        columns instead (``normal_field_columns``): the virtual casing's reverse pass stores its whole
        target-by-source kernel (25 GiB at the 4 nfp 48 x 96 quadrature).
        """
        nf = normal_field(state, params, x)[0]
        parts = [lambda s, p, xx: qs.residuals_state(s, im.runtime_from_params(p, model.cfg)),
                 lambda s, p, xx: jnp.stack(constraint_rows(s, p, xx, nf, with_vacuum=False))]
        blocks = [objective[None, :nq], jnp.eye(len(others))]
        if VACUUM_IOTA_FLOOR:
            parts.append(lambda s, p, xx: vacuum_row(s, p)[None])
            blocks.append(jnp.ones((1, 1)))
        rest = jax.tree.map(lambda a: a[order], model.pullback(tuple(parts), state, mask, params, tuple(blocks), x))
        cots = jnp.concatenate([objective[None], jnp.eye(nrows, nq + 1 + nrows, nq + 1)])
        return rest, cots @ jax.jvp(lambda n: quantities(state, params, x, n), (nf,), (jnp.ones_like(nf),))[1]

    normal_field_columns = jax.jit(lambda state, mask, params, dz, batch, x: model.push(
        lambda s, p: normal_field(s, p, x), state, mask, params, dz, batch)[:, 0])

    @jax.jit
    def direct_gradient(state, params, x, cot):
        """The pullback of ``cot`` through the quantities' direct dependence on ``x``."""
        return jax.vjp(lambda xx: quantities(state, params, xx), x)[1](cot)[0]

    @jax.jit
    def to_coordinates(params, v, x, params_bar):
        """Pull ``params_bar`` back to the current values and the plasma design coordinates."""
        _, current_vjp = jax.vjp(lambda z: model.with_boundary(params, z), v)
        _, plasma_vjp = jax.vjp(lambda z: plasma_params(params, z), x)
        return jax.vmap(lambda bar: jnp.concatenate([current_vjp(bar)[0], plasma_vjp(bar)[0][:npl]]))(params_bar)

    plasma_directions = jax.jit(lambda params, x: jax.vmap(lambda e: jax.jvp(
        lambda xx: plasma_params(params, xx), (x,), (e,))[1])(jnp.eye(x.size)[:npl]))

    def coil_rows(x):
        """``coil_inequalities`` and the scaled coil-to-plasma clearance of the design vector ``x``."""
        coils, surface = coils_from_x(x), surface_from_x(x)
        clearance = surface_distance(coils, surface)
        return jnp.concatenate([coil_inequalities(coils), jnp.atleast_1d(
            (clearance - COIL_SURFACE_DISTANCE_LIMIT - DISTANCE_MARGIN) / COIL_SURFACE_DISTANCE_LIMIT)])

    coil_rows_jit, coil_rows_jac = jax.jit(coil_rows), jax.jit(jax.jacrev(coil_rows))

    cache, anchor = {}, {}
    counters = dict(trials=0, failed=0, evaluations=0, linearizations=0)

    def solve(x):
        """The (cached) equilibrium and quantities at ``x``, or None for a failed trial."""
        key = np.asarray(x, dtype=float).tobytes()
        if key in cache:
            return cache[key]
        counters["trials"] += 1
        params = plasma_params(model.params0, jnp.asarray(x))
        try:
            result = model.solve_boundary(params, None, x0=anchor.get("v"), jacobian=anchor.get("J_c"),
                                          ftol=current_ftol, max_nfev=30)
        except (vj.VmecError, RuntimeError) as error:
            print(f"[trial] failed: {str(error)[:300]}", flush=True)
            counters["failed"] += 1
            cache[key] = None
            return None
        counters["evaluations"] += result["nfev"]
        state, mask, params_x, _ = result["aux"]
        values = np.asarray(quantities_jit(state, params_x, jnp.asarray(x)))
        cache[key] = dict(result, values=values, state=state, mask=mask, params=params_x)
        return cache[key]

    def linearize(x):
        """Design gradients of the objective and every plasma row at ``x``'s solution."""
        sol = solve(x)
        if "gradients" in sol:
            return sol["gradients"]
        counters["linearizations"] += 1
        xj = jnp.asarray(x)
        extra = plasma_directions(sol["params"], xj)
        J, dz, batch, aux = model.linearize(sol["x"], sol["aux"], None, extra=extra)
        # polish the current with this Jacobian, and linearize again where it went if it gained
        polished = model.solve_boundary(plasma_params(model.params0, xj), None, x0=sol["x"], jacobian=J[:, :nc],
                                        ftol=1e-10, max_nfev=10)
        if 0.5 * polished["rows"] @ polished["rows"] < 0.9 * 0.5 * sol["rows"] @ sol["rows"]:
            sol.update(x=polished["x"], aux=polished["aux"])
            J, dz, batch, aux = model.linearize(sol["x"], sol["aux"], None, extra=extra)
        state, mask, params_x, _ = aux
        sol.update(aux=aux, state=state, mask=mask, params=params_x,
                   values=np.asarray(quantities_jit(state, params_x, xj)))
        J_c, J_p = J[:, :nc], J[:, nc:]
        values = sol["values"]
        q, nf = values[:nq], values[nq]
        # Row 0 pulls back the objective 0.5 |q|^2 + 0.5 w nf^2, each further row a unit cotangent.
        cots = np.zeros((1 + nrows, values.size))
        cots[0, :nq], cots[0, nq] = q, NORMAL_FIELD_WEIGHT * nf
        cots[1:, nq + 1:] = np.eye(nrows)
        params_bar, through = quantity_pullback(state, mask, params_x, jnp.asarray(cots[0]), xj)
        G = np.asarray(to_coordinates(params_x, jnp.asarray(sol["x"]), xj, params_bar))
        G = G + np.outer(through, np.asarray(normal_field_columns(state, mask, params_x, dz, batch, xj)))
        Q, R = np.linalg.qr(J_c)
        lam = Q @ np.linalg.solve(R.T, G[:, :nc].T)
        gradients = np.zeros((1 + nrows, x.size))
        for i in range(1 + nrows):
            gradients[i] = np.asarray(direct_gradient(state, params_x, xj, jnp.asarray(cots[i])))
            gradients[i, :npl] += G[i, nc:] - lam[:, i] @ J_p
        anchor.update(v=np.asarray(sol["x"]).copy(), J_c=J_c)
        sol["gradients"] = gradients
        return gradients

    class Rejected(Exception):
        """A trial without a certified equilibrium: SLSQP restarts from the accepted point."""

    def objective(u):
        """SLSQP's objective at the scaled coordinates ``u``."""
        sol = solve(x0 + scales * u)
        if sol is None:
            raise Rejected
        q, nf = sol["values"][:nq], sol["values"][nq]
        return 0.5 * float(q @ q) + 0.5 * NORMAL_FIELD_WEIGHT * float(nf) ** 2

    def objective_gradient(u):
        """Gradient of ``objective`` in ``u``."""
        return linearize(x0 + scales * u)[0] * scales

    def plasma_values(u):
        """Plasma rows; a rejected equilibrium counts as violated."""
        sol = solve(x0 + scales * u)
        return np.full(nrows, -1.0) if sol is None else sol["values"][nq + 1:]

    def plasma_jacobian(u):
        """Jacobian of ``plasma_values`` in ``u`` (zero at a rejected equilibrium)."""
        if solve(x0 + scales * u) is None:
            return np.zeros((nrows, u.size))
        return linearize(x0 + scales * u)[1:] * scales

    constraints = [dict(type="ineq", fun=plasma_values, jac=plasma_jacobian),
                   dict(type="ineq", fun=lambda u: np.asarray(coil_rows_jit(jnp.asarray(x0 + scales * u))),
                        jac=lambda u: np.asarray(coil_rows_jac(jnp.asarray(x0 + scales * u))) * scales)]
    last = dict(time=time.monotonic(), step=0, u=np.zeros_like(x0))

    def save(tag, x):
        """Write ``coils{tag}.json`` and ``wout{tag}.nc`` at ``x``."""
        # The gradient first (cached for SLSQP, which asks for it next): it solves the state tight, so the
        # WOUT re-solve starts converged instead of iterating from the trial tolerance (1000-2000 iterations).
        linearize(x)
        sol = solve(x)
        coils_from_x(jnp.asarray(x)).to_json(str(out / f"coils{tag}.json"))
        deck = im.input_with_params(fixed, sol["params"])
        vj.write_wout(str(out / f"wout{tag}.nc"), opt.solve_equilibrium(deck, initial_state=sol["state"]).wout)

    def log_step(u):
        """SLSQP callback: log the accepted point to ``metrics.jsonl`` and save every ``--save-every`` steps."""
        x = x0 + scales * np.asarray(u, dtype=float)
        sol = solve(x)
        if sol is None:
            raise RuntimeError("the accepted point has no certified equilibrium")
        state, params = sol["state"], sol["params"]
        ctx = im.runtime_from_params(params, model.cfg)
        values = sol["values"]
        coils = coils_from_x(jnp.asarray(x))
        vac = np.asarray(vacuum(state, ctx))
        now = time.monotonic()
        # coil_minimum_scaled_slack includes the clearance row here (main's: coil_inequalities only).
        row = dict(step=last["step"], qa=float(values[:nq] @ values[:nq]),
                   objective=objective(np.asarray(u, dtype=float)), min_abs_iota=float(min_abs_iota(state, ctx)),
                   geometric_axis_iota=float(abs(opt.axis_iota(state, ctx))),
                   vacuum_iota_axis=float(vac[0]), vacuum_iota_min=float(vac.min()), vacuum_iota_edge=float(vac[-1]),
                   b_axis_t=float(opt.axis_field_strength(state, ctx)),
                   coil_currents_a=np.asarray(coils.dofs_currents_raw).tolist(),
                   aspect=float(opt.aspect_ratio(state, ctx)), major_radius_m=float(opt.major_radius(state, ctx)),
                   total_normal_field_rms=float(values[nq]), rbtor_ratio=float(rbtor_ratio(state, ctx, coils)),
                   coil_surface_distance_m=float(surface_distance(coils, surface_from_x(jnp.asarray(x)))),
                   coil_minimum_scaled_slack=float(np.min(np.asarray(coil_rows_jit(jnp.asarray(x))))),
                   plasma_minimum_scaled_slack=float(np.min(values[nq + 1:])),
                   redl_max_relative=model.bootstrap_residual(state, params), curtor=float(params.curtor),
                   beta=float(opt.volume_average_beta(state, ctx)), **counters,
                   step_seconds=now - last["time"], elapsed_seconds=now - started,
                   peak_gpu_gib=(jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / 2**30)
        for k in counters:
            counters[k] = 0
        with open(out / "metrics.jsonl", "a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(f"[step {row['step']}] {TARGET_NAME}={row['qa']:.6e} iota={row['min_abs_iota']:.5f} "
              f"vacuum iota={row['vacuum_iota_min']:.4f} (axis {row['vacuum_iota_axis']:.4f}) "
              f"B_axis={row['b_axis_t']:.5f} aspect={row['aspect']:.4f} R={row['major_radius_m']:.5f} "
              f"B.n={row['total_normal_field_rms']:.2e} RBphi={row['rbtor_ratio']:.5f} "
              f"coil_slack={row['coil_minimum_scaled_slack']:.4f} "
              f"plasma_slack={row['plasma_minimum_scaled_slack']:.4f} redl(max rel)={row['redl_max_relative']:.1e} "
              f"CURTOR={row['curtor']:.0f}A {row['step_seconds']:.1f}s", flush=True)
        if args.save_every and row["step"] % args.save_every == 0:
            save(f".step{row['step']}", x)
        last.update(time=now, step=last["step"] + 1, u=np.asarray(u, dtype=float).copy())
        if len(cache) > 64:
            keep = np.asarray(x, dtype=float).tobytes()
            for k in [k for k in cache if k != keep]:
                del cache[k]

    log_step(last["u"])
    if args.check_gradient:
        g0 = linearize(x0) * scales
        rng = np.random.default_rng(0)
        for k in range(args.check_gradient):
            d = rng.normal(size=x0.size)
            if k == 0:
                d[npl:] = 0.0  # first along the plasma coordinates only
            d /= np.linalg.norm(d)
            for h in (1e-2, 3e-3):
                vals = [np.r_[objective(s * h * d), plasma_values(s * h * d)] for s in (1, -1)]
                fd = (vals[0] - vals[1]) / (2 * h)
                exact = g0 @ d
                print(f"[gradient check] direction {k} h {h:g}: objective {exact[0]:+.6e} vs {fd[0]:+.6e}; rows rel. "
                      "err " + " ".join(f"{abs(e - f) / max(abs(f), 1e-14):.1e}" for e, f in zip(exact[1:], fd[1:])),
                      flush=True)
        return
    u, stop = last["u"], dict(message="", success=False)
    while last["step"] <= args.steps:
        before = last["step"]
        try:
            result = minimize(objective, u, jac=objective_gradient, method="SLSQP", constraints=constraints,
                              callback=log_step, options=dict(maxiter=args.steps + 1 - before, ftol=OPTIMIZER_FTOL))
            stop = dict(message=str(result.message), success=bool(result.success))
            break
        except Rejected:
            print("[slsqp] trial rejected: restarting from the accepted point", flush=True)
            u = last["u"]
            if last["step"] == before:
                stop = dict(message="trial rejected before any progress", success=False)
                break
    x = x0 + scales * last["u"]
    save("", x)
    coils = coils_from_x(jnp.asarray(x))
    coils.to_json(str(out / "coils.json"))
    summary = dict(accepted_steps=last["step"] - 1, **stop, elapsed_seconds=time.monotonic() - started,
                   endpoint=boundary_diagnostics(vj.read_wout(str(out / "wout.nc")), coils))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


# ---- the optimization ------------------------------------------------------------------------------------------------
def parse_args(argv=None):
    """The command-line arguments; ``--restart`` takes its coils from the run."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="new run directory")
    parser.add_argument("--steps", type=int, default=100, help="SLSQP iterations")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu", help="device of the equilibrium solves")
    parser.add_argument("--coils", type=Path, default=COILS_FILE, help="initial ESSOS coils (default: the case's)")
    parser.add_argument("--beta", type=float, default=0.0,
                        help="seed beta: on-axis (WOUT betaxis) for COIL_CASE=ellipse5-beta7, else volume-average")
    parser.add_argument("--wout", type=Path, help="restart the boundary from this WOUT's last surface")
    parser.add_argument("--save-every", type=int, default=25, help="save coils and WOUT every N steps")
    parser.add_argument("--ns", type=int, help="radial resolution of the optimization solves (default: RESOLUTION)")
    parser.add_argument("--modes", type=int, nargs=2, metavar=("MPOL", "NTOR"),
                        help="poloidal and toroidal modes of the optimization solves (default: RESOLUTION)")
    parser.add_argument("--bootstrap", action="store_true",
                        help="reactor-like kinetic profiles and a self-consistent bootstrap current (BOOTSTRAP_MODEL)")
    parser.add_argument("--restart", type=Path, help="continue a finished run from its input.run, final coils and "
                        "WOUT boundary (no seed calibration); pass the run's --beta/--bootstrap")
    parser.add_argument("--chunk", type=int, default=8,
                        help="Jacobian columns per batch of the bootstrap-current solve (BOOTSTRAP_IN_SOLVE)")
    parser.add_argument("--check-gradient", type=int, default=0, metavar="N",
                        help="with BOOTSTRAP_IN_SOLVE: compare the gradients with central differences along N "
                        "directions, then stop")
    args = parser.parse_args(argv)
    if args.restart is not None:
        args.coils = args.restart / "coils.json"
    if args.bootstrap and not args.beta > 0:
        parser.error("--bootstrap needs --beta > 0")
    return args


def main(argv=None):
    """Seed, then optimize the boundary and coils with SLSQP under the hard constraints."""
    global RESOLUTION
    args = parse_args(argv)
    RESOLUTION = (*(args.modes or RESOLUTION[:2]), args.ns or RESOLUTION[2])
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ["JAX_ENABLE_X64"] = "1"
    # The vacuum-iota solve runs inside a jax.pure_callback; a GPU program XLA compiles there (its Krylov
    # certification fallback) deadlocks in the autotuner, which waits on the device the callback holds.
    # CUDA graphs hold device memory outside JAX's pool: with the DKX rows' Jacobian (qi6-beta) the kernels of a
    # later executable no longer load at the 0.8 memory fraction.
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                               + " --xla_gpu_autotune_level=0 --xla_gpu_enable_command_buffer=").strip()
    if args.device == "cpu":
        # On a GPU host JAX_PLATFORMS would count as a user placement and disable VMEX's CPU implicit default.
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import jax
    import jax.numpy as jnp
    from essos.coils import Coils
    from essos.fields import BiotSavart
    from essos.surfaces import surfacerzfourier_from_boundary
    from scipy.optimize import minimize
    import vmex as vj
    from vmex import optimize as opt
    from vmex.core import virtual_casing as vc

    started = time.monotonic()
    inp = seed_input()
    if args.wout is not None:
        inp = boundary_from_wout(inp, vj.read_wout(args.wout))
    redl = fixed = None
    if args.restart is not None:
        inp = restart_input(args.restart)
        redl = redl_profiles(inp)[1] if args.bootstrap else None
    elif args.bootstrap:
        inp, fixed, redl = bootstrap_input(inp, args.beta, args.device)
    else:  # also in vacuum: it sets PHIEDGE for B0
        inp, fixed = finite_beta_input(inp, args.beta, args.device)
    phiedge0 = abs(float(inp.phiedge))
    inp.to_indata(out / "input.run")  # the deck as run: resolution, pressure and seed PHIEDGE
    coils = resize_coils(Coils.from_json(str(args.coils)), COIL_ORDER, N_SEGMENTS)
    if args.restart is None:  # the seed's edge R B_phi, which the field-strength band holds; a restart keeps its own
        coils = scale_coil_currents(coils, abs(float(fixed.wout.rbtor)))
    coils.to_json(str(out / "coils.initial.json"))
    coils0 = Coils.from_json(str(out / "coils.initial.json"))
    if redl is not None and BOOTSTRAP_IN_SOLVE and BOOTSTRAP_MODEL == "redl":
        # The current solved with every equilibrium instead of designed (run_bootstrap_in_solve).
        return run_bootstrap_in_solve(args, inp, coils0, out, started)
    mismatch = None if redl is None else bootstrap_mismatch(inp, redl, args.device)

    qs = target_residual()
    # In vacuum PHIEDGE only scales the plasma field, a null direction; at finite beta the
    # field-strength band ties it to the boundary.
    vary_phiedge = FREE_PHIEDGE and args.beta > 0
    plasma_problem = opt.VmecProblem.from_tuples(
        inp, [(qs.residuals_state, 0.0, 1.0)], max_mode=MAX_MODE, use_ess=True, ess_alpha=ESS_ALPHA,
        vary_phiedge=vary_phiedge, current_dofs=np.size(inp.ac_aux_f) - 1 if redl is not None else None)

    x_boundary0 = plasma_problem.x0
    ncur = np.size(coils0.dofs_currents) * FREE_COIL_CURRENTS  # relative base-coil current changes
    x_coils0 = np.r_[np.zeros(ncur), np.asarray(coils0.curves.dofs).ravel()]
    x0 = np.concatenate([x_boundary0, x_coils0])
    scales = np.concatenate([BOUNDARY_STEP * plasma_problem.scales, np.full(ncur, COIL_CURRENT_STEP),
                             np.full(x_coils0.size - ncur, COIL_STEP)])
    n_boundary = x_boundary0.size  # plasma variables, with the PHIEDGE dof last when it varies
    if redl is not None:  # the current knot values (in units of the largest) and CURTOR/1e6, before PHIEDGE
        knots = np.size(inp.ac_aux_f)
        block = slice(n_boundary - int(vary_phiedge) - knots, n_boundary - int(vary_phiedge))
        assert plasma_problem.names[block.stop - 1].startswith("CURTOR")
        scales[block] = CURRENT_STEP * np.r_[np.ones(knots - 1), abs(float(inp.curtor)) / 1e6]

    def phiedge_at(x):
        """PHIEDGE of the design vector ``x``."""
        return phiedge0 * (1.0 + x[n_boundary - 1]) if vary_phiedge else phiedge0

    def objects_from_x(x):
        """Boundary RBC and ZBS, ESSOS surface and coils of the design vector ``x``."""
        rbc, zbs = plasma_problem.boundary_from_x(x[:n_boundary])
        surface = surfacerzfourier_from_boundary(rbc, zbs, inp.nfp, nphi=NPHI, ntheta=NTHETA)
        currents = coils0.dofs_currents * (1.0 + x[n_boundary:n_boundary + ncur]) if ncur else coils0.dofs_currents
        coils = coils0.with_dofs(jnp.concatenate((x[n_boundary + ncur:], currents)))
        return rbc, zbs, surface, coils

    # Toroidal flux of the coil field through the boundary's phi = 0 cross-section:
    # Gauss-Legendre in radius, uniform in angle, differentiable in boundary and coils.
    rho, rho_weights = np.polynomial.legendre.leggauss(24)
    rho, rho_weights = 0.5 * (rho + 1.0), 0.5 * rho_weights
    theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)

    def toroidal_flux(rbc, zbs, coils):
        """The coils' toroidal flux through the boundary's phi = 0 cross-section."""
        m = np.arange(rbc.shape[1])
        cos_m, sin_m = np.cos(np.outer(theta, m)), np.sin(np.outer(theta, m))
        rbc0, zbs0 = rbc.sum(axis=0), zbs.sum(axis=0)
        r_edge, z_edge = cos_m @ rbc0, sin_m @ zbs0
        dr_edge, dz_edge = -sin_m @ (m * rbc0), cos_m @ (m * zbs0)
        r = rbc0[0] + rho[:, None] * (r_edge - rbc0[0])
        z = rho[:, None] * z_edge
        area = rho[:, None] * ((r_edge - rbc0[0]) * dz_edge - z_edge * dr_edge)
        points = jnp.stack([r, jnp.zeros_like(r), z], axis=-1).reshape(-1, 3)
        b_phi = jax.vmap(BiotSavart(coils).B)(points)[:, 1].reshape(r.shape)
        return jnp.sum(rho_weights[:, None] * b_phi * area) * (2.0 * np.pi / theta.size)

    if args.beta > 0:
        # Virtual casing picks its quadrature once, on the concrete seed, so the
        # plasma field stays differentiable in the boundary on every trial.
        seed = plasma_problem.equilibrium_from_x(x_boundary0)
        precision = vc.plan_vc_precision(vc.surface_field_data_from_state(
            inp, seed.solution, runtime=seed.solver_context, nphi=NPHI, ntheta=NTHETA), digits=VC_DIGITS)

    linked_rbtor = linked_rbtor_of(float(inp.rbc[inp.ntor, 0]))  # the coils' poloidal current, on R = RBC(0,0)
    print(f"coils link mu0 I / 2 pi = {float(linked_rbtor(coils0)):.6f} T m")
    vacuum = vacuum_iota(inp)

    def rbtor_ratio(state, ctx, coils):
        """Equilibrium edge R B_phi over the coils' linked mu0 I / 2 pi."""
        return edge_rbtor(state, ctx) / linked_rbtor(coils)

    def total_normal_field_rms(coils, state, ctx):
        """Area-weighted RMS of (B_coils + B_plasma).n/|B| on the boundary."""
        data = vc.surface_field_data_from_state(inp, state, runtime=ctx, nphi=NPHI, ntheta=NTHETA)
        interface = vc.PlasmaVacuumInterface.from_surface_data(data, digits=VC_DIGITS, precision=precision)
        normal = interface.bnormal_residual(coil_field(coils)) / jnp.linalg.norm(data.B_total, axis=0)
        return weighted_rms(interface.weights, normal)

    aspect_lower, aspect_upper = ASPECT_RANGE
    aspect_scale = 0.5 * (aspect_upper - aspect_lower)
    width, b_width = RADIUS_TOLERANCE - RADIUS_MARGIN, B_AXIS_TOLERANCE - B_AXIS_MARGIN

    def shape_rows(state, ctx, coils):
        """The scaled iota floor, aspect, major-radius, mirror and iota-ceiling rows."""
        iota, aspect, radius = min_abs_iota(state, ctx), opt.aspect_ratio(state, ctx), opt.major_radius(state, ctx)
        rows = [(iota - IOTA_FLOOR - IOTA_MARGIN) / IOTA_FLOOR,
                (aspect - aspect_lower) / aspect_scale, (aspect_upper - aspect) / aspect_scale,
                (radius - RADIUS_TARGET + width) / RADIUS_TOLERANCE,
                (RADIUS_TARGET + width - radius) / RADIUS_TOLERANCE]
        if MIRROR_LIMIT:
            rows.append((MIRROR_LIMIT - MIRROR_MARGIN - opt.mirror_ratio(state, ctx)) / MIRROR_LIMIT)
        if IOTA_CEILING:
            rows.append((IOTA_CEILING - IOTA_MARGIN - max_abs_iota(state, ctx)) / IOTA_FLOOR)
        return jnp.stack(rows)

    def field_rows(state, ctx, coils):
        """The scaled edge R B_phi and axis |B| band rows."""
        strength = rbtor_ratio(state, ctx, coils) - 1.0
        b_axis = opt.axis_field_strength(state, ctx)
        return jnp.stack([(FIELD_STRENGTH_TOLERANCE - strength) / FIELD_STRENGTH_TOLERANCE,
                          (FIELD_STRENGTH_TOLERANCE + strength) / FIELD_STRENGTH_TOLERANCE,
                          (b_axis - B0 + b_width) / B_AXIS_TOLERANCE, (B0 + b_width - b_axis) / B_AXIS_TOLERANCE])

    # Rows that need the equilibrium (scaled so c >= 0 is feasible), in groups: a row's gradient differentiates
    # only its group, so the vacuum-iota solve, the virtual casing and the bootstrap mismatch each run one
    # reverse pass, not one per row.
    plasma_groups, group_sizes = [shape_rows], [5 + bool(MIRROR_LIMIT) + bool(IOTA_CEILING)]
    if VACUUM_IOTA_FLOOR:
        plasma_groups.append(lambda state, ctx, coils: (
            (jnp.min(vacuum(state, ctx)) - VACUUM_IOTA_FLOOR - IOTA_MARGIN) / VACUUM_IOTA_FLOOR)[None])
        group_sizes.append(1)
    if args.beta > 0:
        plasma_groups += [lambda state, ctx, coils: (
            1.0 - total_normal_field_rms(coils, state, ctx) / NORMAL_FIELD_CONSTRAINT)[None], field_rows]
        group_sizes += [1, 4]
    if redl is not None:
        plasma_groups.append(lambda state, ctx, coils: (1.0 - mismatch(state, ctx) / REDL_TOLERANCE)[None])
        group_sizes.append(1)

    def plasma_constraint(u, groups=None):
        """The rows of ``groups`` (default: every plasma group) at the scaled coordinates ``u``."""
        x = jnp.asarray(x0) + jnp.asarray(scales) * u
        coils = objects_from_x(x)[3]
        rows, _ = plasma_problem.jax_quantity_from_state(x[:n_boundary], lambda state, ctx: jnp.concatenate(
            [group(state, ctx, coils) for group in (plasma_groups if groups is None else groups)]))
        return rows

    def coil_rows(u):
        """The coil rows at the scaled coordinates ``u``."""
        x = jnp.asarray(x0) + jnp.asarray(scales) * u
        _, _, surface, coils = objects_from_x(x)
        clearance = surface_distance(coils, surface)
        rows = [coil_inequalities(coils),
                jnp.atleast_1d((clearance - COIL_SURFACE_DISTANCE_LIMIT - DISTANCE_MARGIN)
                               / COIL_SURFACE_DISTANCE_LIMIT),
                # At finite beta the normal-field limit is on the total field: plasma_groups.
                *([] if args.beta > 0 else [
                    jnp.atleast_1d(1.0 - normal_field_rms(coils, surface) / NORMAL_FIELD_CONSTRAINT)])]
        return jnp.concatenate(rows)

    coil_rows_jit = jax.jit(coil_rows)
    plasma_rows_jit = jax.jit(plasma_constraint)
    # One adjoint per plasma row: the gradient of w . rows at a unit w, the forward solve reused.
    group_grads = [jax.jit(jax.grad(lambda u, w, g=g: jnp.vdot(w, plasma_constraint(u, (g,)))))
                   for g in plasma_groups]
    coil_rows_jac = jax.jit(jax.jacrev(coil_rows))

    def half_target(u):
        """(1/2) |r|^2 of the target residual at the scaled coordinates ``u``."""
        x = jnp.asarray(x0) + jnp.asarray(scales) * u
        return plasma_problem.jax_objective_from_state(x[:n_boundary], lambda state, ctx: jnp.zeros(1),
                                                       n_extra_terms=1)[0]

    target_value_and_grad = jax.jit(jax.value_and_grad(half_target))

    def normal_field_cost(u):
        """(1/2) ``NORMAL_FIELD_WEIGHT`` rms(B.n/|B|)^2 at the scaled coordinates ``u``."""
        x = jnp.asarray(x0) + jnp.asarray(scales) * u
        _, _, surface, coils = objects_from_x(x)
        if args.beta > 0:  # the total field, through the equilibrium
            return plasma_problem.jax_extra_costs_from_state(x[:n_boundary], lambda state, ctx: (
                0.5 * NORMAL_FIELD_WEIGHT * total_normal_field_rms(coils, state, ctx)**2)[None], n_extra_terms=1)[0]
        return 0.5 * NORMAL_FIELD_WEIGHT * normal_field_rms(coils, surface)**2

    normal_field_value_and_grad = jax.jit(jax.value_and_grad(normal_field_cost))
    cache = {}

    def objective(u):
        """J and its gradient at the scaled coordinates ``u``, cached for the logging."""
        u = np.asarray(u, dtype=float)
        if cache.get("key") != u.tobytes():
            half, gradient = target_value_and_grad(jnp.asarray(u))
            extra, extra_gradient = normal_field_value_and_grad(jnp.asarray(u))
            cache.update(key=u.tobytes(), qa=2 * float(half), value=float(half + extra),
                         gradient=np.asarray(gradient + extra_gradient))
        return cache["value"], cache["gradient"].copy()

    def plasma_values(u):
        """Plasma rows; a rejected equilibrium (NaN rows) counts as violated."""
        rows = np.asarray(plasma_rows_jit(jnp.asarray(np.asarray(u, dtype=float))))
        return np.where(np.isfinite(rows), rows, -1.0)

    def plasma_jacobian(u):
        """Jacobian of ``plasma_values``, one adjoint per row (non-finite entries zeroed)."""
        u = jnp.asarray(np.asarray(u, dtype=float))
        jac = np.stack([np.asarray(grad(u, jnp.eye(size)[i])) for grad, size in zip(group_grads, group_sizes)
                        for i in range(size)])
        return np.where(np.isfinite(jac), jac, 0.0)

    last = dict(time=time.monotonic(), step=0)

    def log_step(u):
        """Log the point ``u`` to ``metrics.jsonl``."""
        u = np.asarray(u, dtype=float)
        value, _ = objective(u)
        x = x0 + scales * u
        equilibrium = plasma_problem.equilibrium_from_x(x[:n_boundary])
        state, ctx = equilibrium.solution, equilibrium.runtime
        rbc, zbs, surface, coils = objects_from_x(jnp.asarray(x))
        coil_slack = np.asarray(coil_inequalities(coils))
        vac = np.asarray(vacuum(state, ctx))
        now = time.monotonic()
        row = dict(step=last["step"], qa=cache["qa"], objective=value,
                   min_abs_iota=float(min_abs_iota(state, ctx)),
                   geometric_axis_iota=float(abs(opt.axis_iota(state, ctx))),
                   vacuum_iota_axis=float(vac[0]), vacuum_iota_min=float(vac.min()), vacuum_iota_edge=float(vac[-1]),
                   b_axis_t=float(opt.axis_field_strength(state, ctx)),
                   coil_currents_a=np.asarray(coils.dofs_currents_raw).tolist(),
                   aspect=float(opt.aspect_ratio(state, ctx)),
                   major_radius_m=float(opt.major_radius(state, ctx)),
                   **({"mirror_ratio": float(opt.mirror_ratio(state, ctx))} if MIRROR_LIMIT else {}),
                   **({"max_abs_iota": float(max_abs_iota(state, ctx))} if IOTA_CEILING else {}),
                   coil_surface_distance_m=float(surface_distance(coils, surface)),
                   coil_minimum_scaled_slack=float(np.min(coil_slack)),
                   normal_field_rms=float(normal_field_rms(coils, surface)),
                   plasma_minimum_scaled_slack=float(np.min(plasma_values(u))),
                   flux_ratio=float(abs(toroidal_flux(rbc, zbs, coils)) / phiedge_at(x)),
                   phiedge=float(phiedge_at(x)),
                   beta=float(opt.volume_average_beta(state, ctx)),
                   step_seconds=now - last["time"], elapsed_seconds=now - started)
        if args.beta > 0:
            row.update(total_normal_field_rms=float(total_normal_field_rms(coils, state, ctx)),
                       rbtor_ratio=float(rbtor_ratio(state, ctx, coils)))
        if redl is not None:
            row.update(redl_mismatch=float(mismatch(state, ctx)), curtor=float(equilibrium.inp.curtor))
        last.update(time=now, step=last["step"] + 1)
        with open(out / "metrics.jsonl", "a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(f"[step {row['step']}] {TARGET_NAME}={row['qa']:.6e} iota={row['min_abs_iota']:.5f} "
              f"vacuum iota={row['vacuum_iota_min']:.4f} (axis {row['vacuum_iota_axis']:.4f}) "
              f"B_axis={row['b_axis_t']:.5f} "
              + (f"max_iota={row['max_abs_iota']:.5f} " if IOTA_CEILING else "")
              + f"aspect={row['aspect']:.4f} R={row['major_radius_m']:.5f} flux={row['flux_ratio']:.5f} "
              f"B.n={row.get('total_normal_field_rms', row['normal_field_rms']):.2e} beta={row['beta']:.4%} "
              f"RBphi={row.get('rbtor_ratio', float('nan')):.5f} coil_slack={row['coil_minimum_scaled_slack']:.4f} "
              f"plasma_slack={row['plasma_minimum_scaled_slack']:.4f} "
              + (f"{BOOTSTRAP_MODEL}={row['redl_mismatch']:.2e} CURTOR={row['curtor']:.0f}A "
                 if redl is not None else "")
              + f"{row['step_seconds']:.1f}s", flush=True)

    def save(tag, u):
        """Write ``coils{tag}.json`` and ``wout{tag}.nc`` at ``u``."""
        x = x0 + scales * np.asarray(u, dtype=float)
        objects_from_x(jnp.asarray(x))[3].to_json(str(out / f"coils{tag}.json"))
        vj.write_wout(str(out / f"wout{tag}.nc"), plasma_problem.equilibrium_from_x(x[:n_boundary]).wout)

    def checkpoint(u):
        """SLSQP callback: ``log_step`` and, every ``--save-every`` steps, ``save``."""
        # SLSQP can accept a point whose equilibrium re-solve fails; log that and keep optimizing.
        try:
            log_step(u)
        except RuntimeError as error:
            last.update(time=time.monotonic(), step=last["step"] + 1)
            print(f"[step {last['step'] - 1}] not logged: {error}", flush=True)
            return
        if args.save_every and (last["step"] - 1) % args.save_every == 0:
            try:
                save(f".step{last['step'] - 1}", u)
            except RuntimeError as error:
                print(f"[save] step {last['step'] - 1} not saved: {error}", flush=True)

    u0 = np.zeros_like(x0)
    checkpoint(u0)
    constraints = [dict(type="ineq", fun=plasma_values, jac=plasma_jacobian),
                   dict(type="ineq", fun=lambda u: np.asarray(coil_rows_jit(jnp.asarray(u))),
                        jac=lambda u: np.asarray(coil_rows_jac(jnp.asarray(u))))]
    result = minimize(objective, u0, jac=True, method="SLSQP", callback=checkpoint,
                      constraints=constraints, options={"maxiter": args.steps, "ftol": OPTIMIZER_FTOL})
    x = x0 + scales * result.x
    coils = objects_from_x(jnp.asarray(x))[3]
    coils.to_json(str(out / "coils.json"))
    wout = plasma_problem.equilibrium_from_x(x[:n_boundary]).wout
    vj.write_wout(str(out / "wout.nc"), wout)
    summary = dict(iterations=int(result.nit), evaluations=int(result.nfev), success=bool(result.success),
                   message=str(result.message), elapsed_seconds=time.monotonic() - started,
                   beta_target=args.beta, pres_scale=float(inp.pres_scale), endpoint=boundary_diagnostics(wout, coils))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
