#!/usr/bin/env python
"""Ramp the pressure of a fixed-boundary tokamak: beta, Shafranov shift, Mercier.

Raising the plasma pressure raises beta, pushes the magnetic axis outward (the
Shafranov shift), and eventually threatens interchange (Mercier) stability.
This script ramps the pressure scale of a shaped tokamak, hot-restarting each
point from the previous equilibrium, and reads three diagnostics straight from
the wout:

- ``betatotal``, the volume-averaged beta;
- ``raxis_cc[0]``, the magnetic-axis major radius, whose growth over the
  zero-pressure axis is the Shafranov shift;
- ``DMerc``, the Mercier criterion profile (positive is stable), reported as
  its interior minimum.

The deck is a shaped tokamak with ``p(s) = PRES_SCALE * (1 - s)`` and a
prescribed rotational transform (``NCURR = 0``).
"""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np

import vmex as vj

# Input deck:
INPUT_FILE = Path(__file__).resolve().parent / "data" / "input.shaped_tokamak_pressure"

# Multipliers of the deck's PRES_SCALE, one equilibrium each; 20 gives about 1.5 % beta:
PRES_MULTIPLIERS = [0.0, 5.0, 10.0, 15.0, 20.0]

# One radial grid for the whole scan, so every point reuses one compiled solver:
NS = 25
FTOL = 1e-11
NITER = 5000

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    NS = 15
    PRES_MULTIPLIERS = [0.0, 10.0, 20.0]

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

base = replace(vj.VmecInput.from_file(INPUT_FILE),
               ns_array=[NS], ftol_array=[FTOL], niter_array=[NITER])
pres_scale = float(base.pres_scale)

### Run the scan ##############################################################

print(f"{'pres_scale':>11s} {'beta_tot':>10s} {'R_axis(m)':>10s} "
      f"{'Shafranov':>10s} {'min DMerc':>11s}")
print(f"{'-'*11} {'-'*10} {'-'*10} {'-'*10} {'-'*11}")

state = None
axis_radius_vacuum = None
for multiplier in PRES_MULTIPLIERS:
    inp = replace(base, pres_scale=pres_scale * multiplier)
    result = vj.solve_multigrid(inp, initial_state=state, verbose=False)
    state = result.state  # hot restart for the next, higher-pressure point
    wout = vj.wout_from_state(
        inp=inp, state=result.state,
        fsqr=float(result.fsqr), fsqz=float(result.fsqz), fsql=float(result.fsql),
        niter=int(result.iterations), converged=bool(result.converged))

    beta = float(wout.betatotal)
    axis_radius = float(np.asarray(wout.raxis_cc)[0])
    if axis_radius_vacuum is None:
        axis_radius_vacuum = axis_radius  # the zero-pressure axis
    shafranov_shift = axis_radius - axis_radius_vacuum
    minimum_dmerc = float(np.min(np.asarray(wout.DMerc)[2:-1]))  # interior surfaces only
    print(f"{pres_scale * multiplier:11.3e} {beta:10.3e} {axis_radius:10.4f} "
          f"{shafranov_shift:+10.4f} {minimum_dmerc:+11.3e}")

### Print, plot and save ######################################################

print("\nAs pressure rises: beta grows, the magnetic axis shifts outward "
      "(Shafranov), and DMerc (>0 stable) tracks Mercier stability.")
