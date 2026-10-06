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

import numpy as np
import vmex
from vmex.core.polish import (
    evaluate_prescribed_current_force, make_variational_plan, polish_prescribed_current,
)
from vmex.core.solver import prepare_runtime
from vmex.core.strong_force import lift_high_order_state

INPUT_FILE = Path(__file__).parent / "data" / "input.LandremanPaul2021_QA_beta0p5_bootstrap"
NS, SPANS, STEPS = 241, 8, 3000
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
inp = vmex.VmecInput.from_file(INPUT_FILE).change_resolution(mpol=9, ntor=6, ntheta=80, nzeta=32)
if ci_smoke:
    inp = inp.change_resolution(mpol=3, ntor=1, ntheta=12, nzeta=8)
    NS, SPANS, STEPS = 9, 2, 3
inp = replace(inp, ns_array=np.array([NS]))
started = perf_counter()
solve = vmex.solve(inp, polish=False, verbose=True)
solve_seconds = perf_counter() - started
if not solve.converged:
    raise RuntimeError("the discrete solve must converge before polishing")
native = lift_high_order_state(solve.state, prepare_runtime(inp), inp=inp, degree=5, max_spans=SPANS)
angles = (12, 8) if ci_smoke else (48, 40)
current_angles = (16, 12) if ci_smoke else (64, 48)
result = polish_prescribed_current(native, inp, max_iterations=STEPS,
    ntheta=angles[0], nzeta=angles[1],
    current_ntheta=current_angles[0], current_nzeta=current_angles[1])
heldout = make_variational_plan(native, radial_order=9,
    ntheta=2 * angles[0], nzeta=2 * angles[1])
current_grid = make_variational_plan(native, radial_order=9,
    ntheta=current_angles[0], nzeta=current_angles[1])
before = evaluate_prescribed_current_force(native, heldout, inp, current_plan=current_grid)
after = result.samples(heldout)
for label, lo, hi in (("axis", 0.0, 0.1), ("bulk", 0.1, 0.9), ("edge", 0.9, 1.0)):
    ratios = []
    for samples in (before, after):
        weights = native.jacobian_sign * np.asarray(samples.sqrt_g) * heldout.quadrature_weights
        weights *= ((heldout.rho**2 >= lo) & (heldout.rho**2 < hi))[:, None, None]
        ratios.append(float(np.sqrt(np.sum(weights * np.sum(np.asarray(samples.force)**2, axis=-1))
                                    / np.sum(weights * np.asarray(samples.grad_pressure_norm)**2))))
    print(f"{label} force/pressure-gradient RMS: {ratios[0]:.6g} -> {ratios[1]:.6g}")
print(f"discrete solve {solve_seconds:.2f} s; polish including setup {result.seconds:.2f} s; "
      f"{result.iterations} iterations; optimizer_converged={result.optimizer_converged}")
print("Forward result only; current quadrature, refinement, stationarity and edge checks remain required.")
