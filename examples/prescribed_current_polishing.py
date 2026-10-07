"""Forward 3D force polishing with a geometry-dependent prescribed current.

Run with ``python examples/prescribed_current_polishing.py``. The shipped
finite-beta QA deck is a Landreman–Buller–Drevlak case despite its legacy
filename. This example does not certify an equilibrium or export a WOUT:
the field requires functional chi(rho), not the geometry's chipf placeholder.
"""
import os
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import vmex
from vmex.core.polish import (
    evaluate_prescribed_current_force, make_variational_plan, polish_prescribed_current,
)
from vmex.core.solver import prepare_runtime
from vmex.core.strong_force import lift_high_order_state

INPUT_FILE = Path(__file__).parent / "data" / "input.LandremanPaul2021_QA_beta0p5_bootstrap"
OUTPUT_DIR = Path("output_prescribed_current_polishing")
# Laptop defaults (a few minutes).  The research settings are
#   MPOL, NTOR, NTHETA, NZETA = 9, 6, 80, 32
#   NS, SPANS, STEPS = 241, 8, 3000
#   ANGLES, CURRENT_ANGLES = (48, 40), (64, 48)
MPOL, NTOR, NTHETA, NZETA = 5, 4, 32, 24
NS, SPANS, STEPS = 51, 4, 150
ANGLES, CURRENT_ANGLES = (24, 20), (32, 24)
PRINT_EVERY = 10
# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
if os.environ.get("VMEX_EXAMPLES_CI") == "1":
    MPOL, NTOR, NTHETA, NZETA = 3, 1, 12, 8
    NS, SPANS, STEPS = 9, 2, 3
    ANGLES, CURRENT_ANGLES = (12, 8), (16, 12)

###############################################################################
# End of input parameters.
###############################################################################

inp = vmex.VmecInput.from_file(INPUT_FILE).change_resolution(
    mpol=MPOL, ntor=NTOR, ntheta=NTHETA, nzeta=NZETA)
inp = replace(inp, ns_array=np.array([NS]))
started = perf_counter()
solve = vmex.solve(inp, polish=False, verbose=True)
solve_seconds = perf_counter() - started
if not solve.converged:
    raise RuntimeError("the discrete solve must converge before polishing")
native = lift_high_order_state(solve.state, prepare_runtime(inp), inp=inp, degree=5, max_spans=SPANS)

iteration = 0
def progress(_geometry):
    """Report every PRINT_EVERY accepted L-BFGS iterations."""
    global iteration
    iteration += 1
    if iteration % PRINT_EVERY == 0 or iteration == 1:
        print(f"polish iteration {iteration}/{STEPS}  {perf_counter() - started:.1f} s", flush=True)

print(f"polishing (setup compiles first, then up to {STEPS} iterations)", flush=True)
started = perf_counter()
result = polish_prescribed_current(native, inp, max_iterations=STEPS,
    ntheta=ANGLES[0], nzeta=ANGLES[1],
    current_ntheta=CURRENT_ANGLES[0], current_nzeta=CURRENT_ANGLES[1], callback=progress)

### Held-out check on a finer grid ############################################

heldout = make_variational_plan(native, radial_order=9, ntheta=2 * ANGLES[0], nzeta=2 * ANGLES[1])
current_grid = make_variational_plan(native, radial_order=9,
    ntheta=CURRENT_ANGLES[0], nzeta=CURRENT_ANGLES[1])
before = evaluate_prescribed_current_force(native, heldout, inp, current_plan=current_grid)
after = result.samples(heldout)
regions = (("axis", 0.0, 0.1), ("bulk", 0.1, 0.9), ("edge", 0.9, 1.0))
ratios = np.zeros((2, len(regions)))
for j, (label, lo, hi) in enumerate(regions):
    for i, samples in enumerate((before, after)):
        weights = native.jacobian_sign * np.asarray(samples.sqrt_g) * heldout.quadrature_weights
        weights *= ((heldout.rho**2 >= lo) & (heldout.rho**2 < hi))[:, None, None]
        ratios[i, j] = np.sqrt(np.sum(weights * np.sum(np.asarray(samples.force)**2, axis=-1))
                               / np.sum(weights * np.asarray(samples.grad_pressure_norm)**2))
    print(f"{label} force/pressure-gradient RMS: {ratios[0, j]:.6g} -> {ratios[1, j]:.6g}")
print(f"discrete solve {solve_seconds:.2f} s; polish including setup {result.seconds:.2f} s; "
      f"{result.iterations} iterations; optimizer_converged={result.optimizer_converged}")
print("Forward result only; current quadrature, refinement, stationarity and edge checks remain required.")

### Plot and save #############################################################

OUTPUT_DIR.mkdir(exist_ok=True)
figure, axis = plt.subplots(figsize=(5.5, 3.6), layout="constrained")
x = np.arange(len(regions))
for i, (label, color) in enumerate((("discrete solve", "#eda100"), ("polished", "#2a78d6"))):
    axis.bar(x + (i - 0.5) * 0.4, ratios[i], 0.38, color=color, label=label)
axis.set_yscale("log")
axis.set_xticks(x, [f"{name}\n$s\\in[{lo},{hi}]$" for name, lo, hi in regions])
axis.set_ylabel(r"RMS $|\mathbf{J}\times\mathbf{B}-\nabla p|$ / RMS $|\nabla p|$")
axis.legend(frameon=False)
figure_path = OUTPUT_DIR / "prescribed_current_polishing.png"
figure.savefig(figure_path, dpi=150)
print(f"Wrote {figure_path}")
