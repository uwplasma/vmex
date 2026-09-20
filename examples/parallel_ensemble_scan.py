#!/usr/bin/env python
"""Solve an ensemble of independent equilibria concurrently, and time it.

A parameter scan or an ensemble optimization solves N independent equilibria.
Each VMEX forward solve runs on the host and releases the GIL while XLA
executes its compiled iteration lanes, so a plain thread pool overlaps them and
gives real wall-clock speedup -- with every result byte-identical to solving
that input alone.

This builds a balanced scan whose members share one compiled executable, times
it serially and through ``vmex.parallel.solve_ensemble`` at several worker
counts, and prints the strong-scaling curve. It also checks the threaded
results against the serial ones: that is the correctness contract, not the
speed.

The speedup is machine-dependent -- it scales with the free cores on the box --
and sub-linear scaling is expected, because XLA already multithreads within
each solve, so ensemble workers and intra-solve threads share the same cores.
See ``docs/howto/parallel-ensembles.md`` for the mechanism study.
"""

import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import vmex as vj

# Input deck:
INPUT_FILE = Path(__file__).resolve().parent / "data" / "input.nfp2_QA"

# One radial grid, so every ensemble member shares one compiled executable:
NS = 35
FTOL = 1e-11
NITER = 3000

# Ensemble size, the thread counts to time, and how many times to repeat each
# configuration (the best of the repeats is reported):
ENSEMBLE_SIZE = 8
WORKER_COUNTS = (1, 2, 4)
REPEATS = 2

# Half-width of the PHIEDGE spread the ensemble covers, as a fraction:
PHIEDGE_SPREAD = 0.01

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    NS, ENSEMBLE_SIZE, WORKER_COUNTS, REPEATS = 15, 4, (1, 2), 1

###############################################################################
# End of input parameters.
###############################################################################

### Set up the ensemble #######################################################

base = replace(vj.VmecInput.from_file(INPUT_FILE),
               ns_array=[NS], ftol_array=[FTOL], niter_array=[NITER])
phiedge = float(base.phiedge)

# A balanced ensemble: a small PHIEDGE scan around the base value.
inputs = [
    replace(base, phiedge=phiedge * (1.0 + PHIEDGE_SPREAD
                                     * (i - ENSEMBLE_SIZE / 2) / ENSEMBLE_SIZE))
    for i in range(ENSEMBLE_SIZE)
]


def solve(inp):
    """One serial solve, used for the baseline timing and the correctness check."""
    return vj.solve(inp, verbose=False)


def max_state_difference(a, b):
    return max(
        float(np.max(np.abs(np.asarray(getattr(a, f)) - np.asarray(getattr(b, f)))))
        for f in ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")
    )


### Run the ensemble ##########################################################

# Compile the shared executable once, so the timings measure execution rather
# than the one-off XLA compile.
_ = solve(base)

serial_results = [solve(x) for x in inputs]


def time_serial():
    """Wall time of solving the whole ensemble one member at a time."""
    started = time.perf_counter()
    for x in inputs:
        solve(x)
    return time.perf_counter() - started


def time_workers(workers):
    """Wall time of solving the whole ensemble on a thread pool."""
    started = time.perf_counter()
    vj.parallel.solve_ensemble(inputs, workers=workers, multigrid=False, verbose=False)
    return time.perf_counter() - started


serial_seconds = min(time_serial() for _ in range(REPEATS))

### Check the result ##########################################################

# The whole point: threaded results are byte-identical to the serial solve.
ensemble = vj.parallel.solve_ensemble(inputs, multigrid=False, verbose=False)
max_diff = max(max_state_difference(a.state, b.state)
               for a, b in zip(serial_results, ensemble))
iters_match = all(int(a.iterations) == int(b.iterations)
                  for a, b in zip(serial_results, ensemble))
print(f"correctness: max|state diff| vs serial = {max_diff:.1e}  "
      f"(iterations identical: {iters_match})")
assert max_diff == 0.0 and iters_match, "ensemble result must be bit-identical to serial"

### Print, plot and save ######################################################

print(f"\nstrong scaling  (N={ENSEMBLE_SIZE} solves, ns={NS}, "
      f"{os.cpu_count()} logical CPUs, best-of-{REPEATS})")
print(f"{'workers':>8} {'wall_s':>8} {'speedup':>8} {'eff%':>6}")
print(f"{'serial':>8} {serial_seconds:8.2f} {'1.00x':>8} {'100':>6}")
for workers in WORKER_COUNTS:
    seconds = min(time_workers(workers) for _ in range(REPEATS))
    print(f"{workers:8d} {seconds:8.2f} {serial_seconds / seconds:7.2f}x "
          f"{100 * serial_seconds / seconds / workers:5.0f}")
