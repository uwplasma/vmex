#!/usr/bin/env python
"""Solve a fixed-boundary equilibrium from an input file, then write and plot it.

The three steps every new user needs: read an ``&INDATA`` file, converge the
equilibrium with VMEC2000-style progress printing, and write and plot the wout
file. The command line does the same with
``vmex examples/data/input.li383_low_res --plot --booz``.

The deck is the LI383 (NCSX-class, three field periods) boundary at low
resolution and about 4 % beta. The first run compiles the solver and caches
the result, so later runs are much shorter. The Boozer step needs the optional
``booz_xform_jax`` package and is skipped with a message when it is absent.
"""

import os
from dataclasses import replace
from pathlib import Path

import vmex as vj

# Input deck; the output files are named after it:
INPUT_FILE = Path(__file__).resolve().parent / "data" / "input.li383_low_res"

# Fields of the deck to replace before solving, for example
# dict(ns_array=[25, 51], ftol_array=[1e-9, 1e-12]); empty runs it as written:
INPUT_OVERRIDES = {}

# Boozer spectrum of the converged equilibrium:
RUN_BOOZER = True

# Directory that receives every output file:
OUTPUT_DIR = Path("output_fixed_boundary_run")

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    INPUT_OVERRIDES = dict(ns_array=[13], ftol_array=[1e-8], niter_array=[2000])
    RUN_BOOZER = False

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

inp = replace(vj.VmecInput.from_file(INPUT_FILE), **INPUT_OVERRIDES)
case = INPUT_FILE.name.removeprefix("input.")
print(f"input: {INPUT_FILE.name}")
print(f"  nfp={inp.nfp}  mpol={inp.mpol}  ntor={inp.ntor}  lasym={inp.lasym}")
print(f"  ns_array={list(map(int, inp.ns_array))}  "
      f"ftol_array={[f'{f:.0e}' for f in inp.ftol_array]}")
print(f"  major radius RBC(0,0) = {inp.rbc[inp.ntor, 0]:.3f} m, "
      f"phiedge = {inp.phiedge:.3f} Wb")

### Solve the equilibrium #####################################################

# verbose=True prints the VMEC2000-format iteration tables.
result = vj.solve_multigrid(inp, verbose=True)
print(f"\nconverged = {result.converged} after {int(result.iterations)} "
      f"iterations; fsqr = {float(result.fsqr):.3e}, "
      f"fsqz = {float(result.fsqz):.3e}, fsql = {float(result.fsql):.3e}")

### Print, plot and save ######################################################

wout = vj.wout_from_state(
    inp=inp, state=result.state,
    fsqr=float(result.fsqr), fsqz=float(result.fsqz), fsql=float(result.fsql),
    niter=int(result.iterations), converged=bool(result.converged))
print(f"\nwout scalars: aspect = {float(wout.aspect):.4f}, "
      f"volume = {float(wout.volume_p):.4f} m^3, "
      f"B0 = {float(wout.b0):.4f} T, betatotal = {float(wout.betatotal):.3e}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
wout_path = vj.write_wout(OUTPUT_DIR / f"wout_{case}.nc", wout)
print(f"Wrote {wout_path}")
# plot_wout writes the summary, surfaces, |B|, profiles, stability and 3-D figures.
for path in vj.plot_wout(wout_path, OUTPUT_DIR).values():
    print(f"Wrote {path}")

if RUN_BOOZER:
    try:
        boozmn_path = vj.run_booz_xform(wout_path, outdir=OUTPUT_DIR)
        for path in vj.plot_boozmn(boozmn_path, OUTPUT_DIR).values():
            print(f"Wrote {path}")
    except ImportError as error:
        print(f"Skipping the Boozer step ({error}); pip install booz_xform_jax")
