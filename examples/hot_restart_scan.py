#!/usr/bin/env python
"""Scan a parameter with hot restarts, and count what that saves.

A parameter scan solves a sequence of nearby equilibria. Starting each one from
the previous converged state -- a hot restart -- means the solver begins a hair
from the answer and converges in a handful of iterations instead of hundreds.
Because VMEX caches one compiled executable per solver structure, every scan
point at fixed resolution reuses it and nothing recompiles.

This solves a base case cold, then scans the edge toroidal flux and solves each
point warm-started from its predecessor. It also re-solves each point cold, so
the saving is measured rather than asserted. The iteration counts are
machine-independent; the wall times are printed too but are only indicative on
a shared CPU.

The deck is a circular tokamak, one field period, fixed boundary, on a single
radial grid so the seed shape matches across the scan.
"""

import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import vmex as vj

# Input deck:
INPUT_FILE = Path(__file__).resolve().parent / "data" / "input.circular_tokamak"

# Multipliers of the deck's PHIEDGE, one equilibrium each:
SCAN = np.linspace(0.95, 1.05, 5)

# One radial grid for the whole scan, so every point reuses one compiled solver:
NS = 25
FTOL = 1e-11
NITER = 5000

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    NS = 15

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

base = replace(vj.VmecInput.from_file(INPUT_FILE),
               ns_array=[NS], ftol_array=[FTOL], niter_array=[NITER])
phiedge = float(base.phiedge)


def solve(inp, seed):
    """Solve one scan point, returning the result and its wall time."""
    started = time.perf_counter()
    result = vj.solve_multigrid(inp, initial_state=seed, verbose=False)
    return result, time.perf_counter() - started


### Run the scan ##############################################################

base_result, base_seconds = solve(base, None)
print(f"cold base solve: {int(base_result.iterations):4d} iters, {base_seconds:6.2f} s "
      f"(phiedge = {phiedge:.4f})")
print(f"\n{'phiedge':>10s} {'iters':>6s} {'wall_s':>8s}   restart")
print(f"{'-'*10} {'-'*6} {'-'*8}   {'-'*12}")

state = base_result.state
cold_iterations = []
for multiplier in SCAN:
    inp = replace(base, phiedge=phiedge * float(multiplier))
    result, seconds = solve(inp, state)     # warm: seed from the previous point
    state = result.state                    # carry the state forward
    # Reference: how many iterations the SAME point needs cold, with no seed.
    cold_result, _ = solve(inp, None)
    cold_iterations.append(int(cold_result.iterations))
    print(f"{phiedge * multiplier:10.4f} {int(result.iterations):6d} {seconds:8.2f}   "
          f"warm ({int(cold_result.iterations)} cold)")

### Print, plot and save ######################################################

print(f"\nWarm restarts converge in far fewer iterations than cold "
      f"(cold ~{int(np.mean(cold_iterations))} iters/point); the compiled executable "
      "is reused across every scan point.")
