#!/usr/bin/env python
"""Solve a free-boundary equilibrium from coil currents and an mgrid file.

In a fixed-boundary run you prescribe the last closed flux surface. In a
free-boundary run you prescribe the coils instead -- their currents (``EXTCUR``)
and the vacuum field they produce on a grid (an ``mgrid`` file) -- and VMEC
solves for the plasma boundary that balances against that external field. Each
iteration the NESTOR vacuum solver recomputes the field outside the plasma, so
the surface at the end is an output, not an input.

CLI equivalent: ``vmex examples/data/input.cth_like_free_bdy``.

The deck is the CTH-like torsatron, five field periods, with two coil circuits.
The NESTOR solve makes this heavier than a fixed-boundary run.
"""

import os
from dataclasses import replace
from pathlib import Path

import vmex as vj

# Input deck and the tabulated vacuum field its coils produce:
DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_FILE = DATA_DIR / "input.cth_like_free_bdy"
MGRID_FILE = DATA_DIR / "mgrid_cth_like.nc"

# Fields of the deck to replace before solving; empty runs it as written:
INPUT_OVERRIDES = {}

# Diagnostic figures of the found equilibrium:
MAKE_PLOTS = True

# Directory that receives every output file:
OUTPUT_DIR = Path("output_free_boundary_mgrid")

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs.  The looser
# tolerance still converges fully; the figures are what the smoke pass skips:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    INPUT_OVERRIDES = dict(ftol_array=[1e-9], niter_array=[3000])
    MAKE_PLOTS = False

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

if not MGRID_FILE.is_file():  # release asset, not tracked in git
    raise SystemExit(f"{MGRID_FILE} is missing; fetch it with "
                     "`python tools/fetch_assets.py --bundle reference-nc`")

inp = replace(vj.VmecInput.from_file(INPUT_FILE), **INPUT_OVERRIDES)
case = INPUT_FILE.name.removeprefix("input.")
print(f"free boundary: nfp={inp.nfp}  EXTCUR={list(map(float, inp.extcur[:2]))} A-turns")
print(f"external field: {MGRID_FILE.name}")

### Solve the equilibrium #####################################################

result = vj.solve_free_boundary(inp, mgrid_path=MGRID_FILE, verbose=not ci_smoke)
print(f"\nconverged = {result.converged} after {int(result.iterations)} "
      f"iterations; fsqr = {float(result.fsqr):.3e}")

### Print, plot and save ######################################################

wout = vj.wout_from_state(
    inp=inp, state=result.state, fsqr=float(result.fsqr), fsqz=float(result.fsqz),
    fsql=float(result.fsql), niter=int(result.iterations),
    converged=bool(result.converged), vacuum_output=result.vacuum)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
wout_path = vj.write_wout(OUTPUT_DIR / f"wout_{case}.nc", wout)
print(f"aspect = {float(wout.aspect):.4f}, volume = {float(wout.volume_p):.4f} m^3 "
      "(the boundary was solved for, not prescribed)")
print(f"Wrote {wout_path}")

if MAKE_PLOTS:
    for key, path in vj.plot_wout(wout_path, OUTPUT_DIR).items():
        print(f"Wrote {path}")
