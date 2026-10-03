"""Case shared by the free- and fixed-boundary coil-constraint benchmarks.

Default: vacuum QA from a rotating ellipse, 1 m major radius, three independent
order-16 coils. MSC is mean SQUARED curvature, in inverse square metres.

``COIL_CASE=qa3``, ``qh`` and ``qi`` start from rotating ellipses at R = 1 m
and B0 ~ 1 T: nfp 3 at aspect 6 for QA, nfp 4 at aspect 6 for QH, helicity
(1, -1), and nfp 4 at aspect 8 for the constructed QI residual with a
mirror-ratio limit. Their iota floors keep the profile off the low-order
rationals a vacuum field breaks into islands at: QH above iota = 1 (its seed,
b = 0.9 a_eff, starts at 1.016), QI above 1/2. Each has its own stage-two
coils, three order-8 coils per half period.
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASE = os.environ.get("COIL_CASE", "ellipse5")
RESOLUTION = (8, 8, 51)            # MPOL, NTOR, NS of the optimization solves
GRID = (64, 64)                    # NTHETA, NZETA
EQUILIBRIUM_FTOL = 1e-15
QA_SURFACES = tuple(i / 10 for i in range(1, 11))
COILS_FILE = HERE / "coils.initial.json"
SEED = None                        # (nfp, aspect, b / a_eff): rotating ellipse replacing the deck's boundary
HELICITY = (1, 0)                  # quasisymmetry (M, N); None minimizes the constructed QI residual
TARGET_NAME = "QA"
QI_SURFACES = tuple(i / 5 for i in range(1, 6))
QI_OPTIONS = dict(mboz=12, nboz=12, nphi=61, nalpha=18, n_bounce=21)  # examples/optimization/QI_optimization.py
MIRROR_LIMIT, MIRROR_MARGIN = None, 0.001  # upper limit on the edge mirror ratio (Bmax - Bmin) / (Bmax + Bmin)

# Physical targets. The free-boundary arm imposes them as hard inequalities.
IOTA_FLOOR, IOTA_MARGIN = 0.41, 0.0005
ASPECT_RANGE = (4.9, 5.1)
RADIUS_TARGET, RADIUS_TOLERANCE, RADIUS_MARGIN = 1.0, 0.01, 0.001

# Coils and their hard engineering limits.
N_COILS, COIL_ORDER, N_SEGMENTS = 3, 16, 256
COIL_STEP = 0.05                   # coordinate scale of the coil Fourier modes
LENGTH_LIMIT = 5.0                 # m, each independent coil
CURVATURE_LIMIT = 5.0              # 1/m, everywhere along each coil
MSC_LIMIT = 5.0                    # 1/m^2, each coil
COIL_DISTANCE_LIMIT = 0.15         # m, including symmetry copies
COIL_SURFACE_DISTANCE_LIMIT = 0.20 # m, to the current plasma boundary
# Interior margins of the sampled constraints; endpoint checks use the limits.
CURVATURE_MARGIN, MSC_MARGIN, LENGTH_MARGIN, DISTANCE_MARGIN = 0.10, 0.02, 1e-5, 0.001

if CASE in ("qa3", "qh", "qi"):
    COILS_FILE = HERE / f"coils.{CASE}.json"
    N_COILS, COIL_ORDER = 3, 8
    LENGTH_LIMIT, CURVATURE_LIMIT, MSC_LIMIT = 3.5, 8.0, 10.0
    COIL_DISTANCE_LIMIT, COIL_SURFACE_DISTANCE_LIMIT = 0.08, 0.15
if CASE == "qa3":
    SEED, IOTA_FLOOR, ASPECT_RANGE = (3, 6.0, 0.5), 0.41, (5.9, 6.1)
elif CASE == "qh":
    SEED, HELICITY, TARGET_NAME, ASPECT_RANGE = (4, 6.0, 0.9), (1, -1), "QH", (5.9, 6.1)
    IOTA_FLOOR = 1.1               # between the iota = 1 and 8/7 resonances
elif CASE == "qi":
    SEED, HELICITY, TARGET_NAME, ASPECT_RANGE, MIRROR_LIMIT = (4, 8.0, 0.5), None, "QI", (7.9, 8.1), 0.21
    IOTA_FLOOR = 0.51              # above the iota = 1/2 resonance
elif CASE != "ellipse5":
    raise ValueError(f"unknown COIL_CASE {CASE!r}")
