#!/usr/bin/env python
"""Solve one equilibrium and produce every built-in diagnostic figure.

VMEX ships its own plotting and a Boozer-transform wrapper, so a single
converged equilibrium gives the whole diagnostic set without a separate
post-processing chain. Two steps do the work:

- ``vj.plot_wout`` writes the figures named in FIGURES and returns
  ``{key: path}``;
- ``vj.run_booz_xform`` and ``vj.plot_boozmn`` transform to straight-field-line
  Boozer coordinates and plot the |B| spectrum on the last closed surface,
  which is the view that shows quasisymmetry.

CLI equivalent: ``vmex examples/data/input.li383_low_res --plot --booz``.

The deck is LI383 (NCSX-class, three field periods) at finite beta -- it
carries a seven-term pressure polynomial and the solve reports betatotal
4.262e-02. The Boozer step needs the optional ``booz_xform_jax`` package and is
skipped with a message when it is absent.
"""

import os
from dataclasses import replace
from pathlib import Path

import vmex as vj

# Input deck; the output files are named after it:
INPUT_FILE = Path(__file__).resolve().parent / "data" / "input.li383_low_res"

# Fields of the deck to replace before solving; empty runs it as written:
INPUT_OVERRIDES = {}

# Which plot_wout figures to write:
FIGURES = ("summary", "surfaces", "modB", "stability", "3d")

# Boozer spectrum of the converged equilibrium:
RUN_BOOZER = True

# Directory that receives every output file:
OUTPUT_DIR = Path("output_plot_and_boozer")

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    INPUT_OVERRIDES = dict(ns_array=[13], ftol_array=[1e-8], niter_array=[2000])

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

inp = replace(vj.VmecInput.from_file(INPUT_FILE), **INPUT_OVERRIDES)
case = INPUT_FILE.name.removeprefix("input.")

### Solve the equilibrium #####################################################

result = vj.solve_multigrid(inp, verbose=not ci_smoke)
wout = vj.wout_from_result(inp, result)

### Print, plot and save ######################################################

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
wout_path = vj.write_wout(OUTPUT_DIR / f"wout_{case}.nc", wout)
print(f"converged = {result.converged}; wrote {wout_path}")

# plot_wout accepts a WoutData or a path and returns {key: written_png_path}.
figures = vj.plot_wout(wout_path, OUTPUT_DIR, which=FIGURES)
for key, path in figures.items():
    print(f"  [{key:9s}] {path}")

# run_booz_xform raises ImportError without the optional booz_xform_jax
# package; the figures above are already written, so skip with a message.
if RUN_BOOZER:
    try:
        boozmn_path = vj.run_booz_xform(wout_path, outdir=OUTPUT_DIR)
        print(f"Wrote {boozmn_path}")
        for key, path in vj.plot_boozmn(boozmn_path, OUTPUT_DIR).items():
            print(f"  [booz:{key:9s}] {path}")
    except ImportError as exc:
        print(f"skipping Boozer step ({exc}); pip install booz_xform_jax")
