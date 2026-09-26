"""Shared physical and optimization parameters for the two benchmark arms.

Reference size: 1 m plasma major radius (Wechsung 2022 / Jorge 2023).
Three independent coils are retained; this is not a reproduction of either
paper's coil count. MSC is mean SQUARED curvature, in inverse square metres.
"""
RESOLUTION = (8, 8, 51)
GRID = (64, 64)
EQUILIBRIUM_FTOL = 1e-15
VERIFY_NS, VERIFY_FTOL = 201, 1e-15
ACCEPTED_STEPS = 100
OPTIMIZER_FTOL = 1e-10
COIL_STEP = 0.05
ASPECT_TARGET, ASPECT_WEIGHT = 5.0, 1.0
IOTA_FLOOR, IOTA_WEIGHT = 0.19, 10.0
RADIUS_TARGET, RADIUS_TOLERANCE = 1.0, 0.01
IOTA_MARGIN, RADIUS_MARGIN = 0.0005, 0.001
N_COILS, COIL_ORDER, N_SEGMENTS = 3, 16, 256
LENGTH_LIMIT = 5.0                 # m, each independent coil; upper bound
CURVATURE_LIMIT = 5.0              # 1/m, everywhere along each coil
MSC_LIMIT = 5.0                    # 1/m^2, each coil; RMS limit sqrt(5)
COIL_DISTANCE_LIMIT = 0.15         # m, including symmetry copies
COIL_SURFACE_DISTANCE_LIMIT = 0.20 # m, relative to the current plasma boundary

# Interior margins for the sampled optimization constraints. Final checks use
# physical bounds above, with independently refined geometry.
CURVATURE_MARGIN = 0.10
MSC_MARGIN = 0.02
LENGTH_MARGIN = 1e-5
DISTANCE_MARGIN = 0.001
