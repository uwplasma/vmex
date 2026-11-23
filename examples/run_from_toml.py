#!/usr/bin/env python
"""Solve an equilibrium from a TOML input deck.

A ``.toml`` deck holds the keys of the structured JSON deck, with arrays and
``{m, n, value}`` tables for the boundary harmonics, so it is easy to read and
to edit by hand.  ``vmex circular_tokamak.toml`` does the same from the
command line.  Python 3.10 reads TOML through ``tomli``, which VMEX installs.

The deck is a circular tokamak of major radius 6 and minor radius 2, so the
aspect ratio is 3.
"""

from pathlib import Path

import vmex as vj

# Input deck:
INPUT_FILE = Path(__file__).resolve().parent / "data" / "circular_tokamak.toml"

# Directory that receives every output file:
OUTPUT_DIR = Path("output_run_from_toml")

###############################################################################
# End of input parameters.
###############################################################################

### Solve #####################################################################

inp = vj.VmecInput.from_file(INPUT_FILE)
res = vj.solve_multigrid(inp, verbose=False)
wout = vj.wout_from_state(
    inp=inp, state=res.state, fsqr=float(res.fsqr), fsqz=float(res.fsqz),
    fsql=float(res.fsql), niter=int(res.iterations), converged=bool(res.converged))

### Print and save ############################################################

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"read {INPUT_FILE.name}: nfp={inp.nfp} mpol={inp.mpol} ntor={inp.ntor}")
print(f"converged={bool(res.converged)}  aspect={float(wout.aspect):.6f}  (R0/a = 3)")
print(f"Wrote {vj.write_wout(OUTPUT_DIR / 'wout_circular_tokamak.nc', wout)}")
