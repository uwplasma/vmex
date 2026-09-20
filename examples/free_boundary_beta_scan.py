#!/usr/bin/env python
"""Ramp the pressure of a free-boundary equilibrium at fixed coil currents.

A free-boundary plasma responds to pressure both internally, through the
Shafranov shift, and at its edge: the last closed flux surface moves as the
plasma pushes against the fixed external field. This ramps the pressure of the
CTH-like case with the coil currents held fixed and reports beta and the plasma
volume at each step.

Unlike the fixed-boundary ramp in ``finite_beta_scan.py``, the boundary here is
recomputed by the NESTOR vacuum solve at every pressure, so each point is a
full free-boundary solve. Each converged state hot-starts the next.

The deck is the CTH-like torsatron, five field periods, with a parabolic
pressure; the scan reaches about 2.6 % beta.
"""

import os
from dataclasses import replace
from pathlib import Path

import vmex as vj

# Input deck and the tabulated vacuum field its coils produce:
DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_FILE = DATA_DIR / "input.cth_like_free_bdy"
MGRID_FILE = DATA_DIR / "mgrid_cth_like.nc"

# Pressure scale of the parabolic profile, in Pa, one equilibrium each:
PRES_SCALES = [0.0, 2000.0, 4000.0, 6000.0]

# Solver budget for every point of the ramp:
FTOL = 1e-9
NITER = 3000

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    PRES_SCALES = [0.0, 3000.0, 6000.0]

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

if not MGRID_FILE.is_file():  # release asset, not tracked in git
    raise SystemExit(f"{MGRID_FILE} is missing; fetch it with "
                     "`python tools/fetch_assets.py --bundle reference-nc`")

# Parabolic pressure p(s) = PRES_SCALE * (1 - s); coil currents held fixed.
base = vj.VmecInput.from_file(INPUT_FILE)
base = replace(
    base, ftol_array=[FTOL], niter_array=[NITER],
    pmass_type="power_series", am=[1.0, -1.0] + [0.0] * (len(base.am) - 2))

### Run the scan ##############################################################

print(f"free-boundary beta scan (EXTCUR held at {list(map(float, base.extcur[:2]))} A-turns)")
print(f"\n{'pres_scale':>11s} {'beta_tot':>10s} {'volume(m^3)':>12s} {'iters':>6s}")
print(f"{'-'*11} {'-'*10} {'-'*12} {'-'*6}")

betas = []
state = None
for ps in PRES_SCALES:
    inp = replace(base, pres_scale=ps)
    res = vj.solve_free_boundary(
        inp, mgrid_path=MGRID_FILE, initial_state=state,
        error_on_no_convergence=False)
    state = res.state
    wout = vj.wout_from_state(
        inp=inp, state=res.state, fsqr=float(res.fsqr), fsqz=float(res.fsqz),
        fsql=float(res.fsql), niter=int(res.iterations),
        converged=bool(res.converged), vacuum_output=res.vacuum)
    beta = float(wout.betatotal)
    betas.append(beta)
    print(f"{ps:11.1f} {beta:10.3e} {float(wout.volume_p):12.4f} {int(res.iterations):6d}")

### Print, plot and save ######################################################

print(f"\nbeta ramps 0 -> {max(betas) * 100:.1f}% at fixed coil currents; the "
      "free boundary is re-solved (NESTOR) at every pressure.")
