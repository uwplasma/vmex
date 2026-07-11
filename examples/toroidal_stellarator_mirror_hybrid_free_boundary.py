"""Solve and plot the 16-coil toroidal mirror hybrid in free-boundary mode.

The production schedule requests beta through 50%, but plots and files are
written only for genuinely converged coupled equilibria.  The present Fourier
corrector is validated through target beta 0.775% at ``ftol=1e-8`` and is
expected to report its conditioning barrier before 1%; see ``plan.md`` M8.
"""

import json
import os
from pathlib import Path

import jax

from vmec_jax import (
    plot_hybrid_free_boundary_scan,
    plot_wout,
    solve_square_coil_free_boundary_scan,
    write_wout,
)

# Inputs: edit these values, then run this file directly.
OUTPUT_DIR = Path("results/toroidal_stellarator_mirror_hybrid_free_boundary")
BETA_TARGETS = (
    0.0005,
    0.001,
    0.0015,
    0.002,
    0.0025,
    0.003,
    0.0035,
    0.004,
    0.00475,
    0.005,
    0.00525,
    0.0055625,
    0.005953125,
    0.006153125,
    0.006403125,
    0.006715625,
    0.006815625,
    0.006915625,
    0.006978125,
    0.007040625,
    0.0071,
    0.0072,
    0.0073,
    0.00735,
    0.0074,
    0.0075,
    0.0076,
    0.00775,
    0.008,
    0.00825,
    0.0085,
    0.00875,
    0.009,
    0.00925,
    0.0095,
    0.00975,
    0.01,
    0.03,
    0.10,
    0.25,
    0.50,
)
N_COILS_PER_SIDE = 4
COIL_SIDE_LENGTH = 3.0
COIL_RADIUS = 0.5
COIL_CURRENT = 8.0e5
TOROIDAL_CURRENT = 3.0e3
MINOR_RADIUS = 0.10
MPOL, NTOR = 6, 20
NS_ARRAY = (3, 5)
NTHETA, NZETA = 48, 256
FTOL = 1.0e-8
MAX_ITERATIONS = 5000
CORRECTOR_ITERATIONS = 1000

CI = os.environ.get("VMEC_JAX_EXAMPLES_CI") == "1"
if CI:
    BETA_TARGETS = ()
    MPOL, NTOR = 3, 4
    NS_ARRAY = (3,)
    NTHETA, NZETA = 16, 32
    FTOL = 1.0e-5
    MAX_ITERATIONS = 800
    CORRECTOR_ITERATIONS = 800

jax.config.update("jax_enable_x64", True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

scan = solve_square_coil_free_boundary_scan(
    BETA_TARGETS,
    n_coils_per_side=N_COILS_PER_SIDE,
    side_length=COIL_SIDE_LENGTH,
    coil_radius=COIL_RADIUS,
    coil_current=COIL_CURRENT,
    toroidal_current=TOROIDAL_CURRENT,
    minor_radius=MINOR_RADIUS,
    mpol=MPOL,
    ntor=NTOR,
    ns_array=NS_ARRAY,
    ntheta=NTHETA,
    nzeta=NZETA,
    ftol=FTOL,
    max_iterations=MAX_ITERATIONS,
    corrector_iterations=CORRECTOR_ITERATIONS,
    coil_segments=32 if CI else 96,
    mgrid_shape=(17, 15, 24) if CI else (49, 41, 96),
)

rows = []
for point in scan.points:
    label = f"beta_{100 * point.target_beta:.6f}pct".replace(".", "p")
    write_wout(OUTPUT_DIR / f"wout_hybrid_free_{label}.nc", point.wout)
    row = {
        "target_beta": point.target_beta,
        "achieved_beta": point.achieved_beta,
        "residuals": [point.result.fsqr, point.result.fsqz, point.result.fsql],
        "iterations": [point.predictor_iterations, point.corrector_iterations, point.free_iterations],
        "volume_m3": point.wout.volume_p,
        "aspect": point.wout.aspect,
        "wall_seconds": point.wall_seconds,
    }
    rows.append(row)
    print(
        f"target={100 * point.target_beta:8.5f}% achieved={100 * point.achieved_beta:8.5f}% "
        f"fsq={point.maximum_residual:.3e} iterations={sum(row['iterations'])}"
    )

summary = {
    "points": rows,
    "failed_target_beta": scan.failed_target_beta,
    "failure": scan.failure,
    "failed_corrector_residuals": None
    if scan.failed_corrector is None
    else [scan.failed_corrector.fsqr, scan.failed_corrector.fsqz, scan.failed_corrector.fsql],
    "failed_corrector_iterations": None if scan.failed_corrector is None else scan.failed_corrector.iterations,
}
(OUTPUT_DIR / "hybrid_free_boundary_scan.json").write_text(json.dumps(summary, indent=2) + "\n")
plot_wout(scan.points[-1].wout, OUTPUT_DIR, name="hybrid_free_endpoint")
plot_hybrid_free_boundary_scan(scan, OUTPUT_DIR)

if scan.failed_target_beta is not None:
    print(f"STOPPED at target beta={100 * scan.failed_target_beta:.6f}%: {scan.failure}")
print(f"Wrote {len(scan.points)} converged equilibria and plots under {OUTPUT_DIR}")
