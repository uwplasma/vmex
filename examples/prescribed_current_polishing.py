"""Polish a 3D finite-beta equilibrium with a prescribed toroidal current.

The discrete VMEX solve converges its spectral force residual, but the
continuous force ``J x B - grad p`` it leaves can still be large.  This
script lifts the converged state to a high-order radial representation and
lowers the continuous force with ``polish_prescribed_current``: L-BFGS on the
weighted force, with the enclosed current ``I(s)`` of the deck imposed through
an angular-mean Ampere closure for ``chi'(rho)``.  The deck is the finite-beta
QA of Landreman, Buller and Drevlak (its file keeps a legacy name).

It reports the force over pressure-gradient RMS on a finer held-out grid, by
region, before and after, and saves a bar chart.  This is a forward
experiment: there is no stationarity certificate (unlike the axisymmetric
polish), and no WOUT is exported, because the field needs the functional
``chi(rho)``, not the geometry's ``chipf`` placeholder.
"""
import os
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import vmex
from _polish_figure import polish_figure
from vmex.core.polish import (
    evaluate_prescribed_current_force, make_variational_plan, polish_prescribed_current,
)
from vmex.core.solver import prepare_runtime
from vmex.core.strong_force import lift_high_order_state

INPUT_FILE = Path(__file__).parent / "data" / "input.LandremanPaul2021_QA_beta0p5_bootstrap"
OUTPUT_DIR = Path("output_prescribed_current_polishing")
FIGURES = Path(__file__).resolve().parents[1] / "docs" / "_static" / "figures"
if os.environ.get("VMEX_EXAMPLES_CI") == "1":
    FIGURES = OUTPUT_DIR
# Laptop defaults.  The research settings are
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
print(f"Forward result only: no stationarity certificate (L-BFGS stopped at |grad| = "
      f"{result.gradient_norm:.2e}); quadrature, refinement and edge checks remain open.")

### Plot and save #############################################################

OUTPUT_DIR.mkdir(exist_ok=True)
rho = np.asarray(heldout.rho)
profiles = {}
for label, samples in (("VMEC solve", before), ("polished", after)):
    weights = native.jacobian_sign * np.asarray(samples.sqrt_g) * heldout.quadrature_weights
    force = np.sum(weights * np.sum(np.asarray(samples.force)**2, axis=-1), axis=(1, 2))
    gradient = np.sum(weights * np.asarray(samples.grad_pressure_norm)**2, axis=(1, 2))
    profiles[label] = (rho, np.sqrt(force / gradient))
names = [f"{name}\n$s\\in[{lo},{hi}]$" for name, lo, hi in regions]
figure_path = FIGURES / ("prescribed_current_polishing.webp" if FIGURES == OUTPUT_DIR else "readme_polish_stellarator.webp")
ratio_label = r"RMS $|\mathbf{J}\times\mathbf{B}-\nabla p|$ / RMS $|\nabla p|$"
polish_figure(figure_path, profiles, {"VMEC solve": ratios[0], "polished": ratios[1]}, names,
              ratio_label, ratio_label, bar_format="%.2g", headroom=4.0)
print(f"Wrote {figure_path}")
