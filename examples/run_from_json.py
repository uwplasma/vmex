#!/usr/bin/env python
"""Convert an &INDATA deck to structured JSON, read it back, and solve both.

VMEX reads the classic Fortran ``&INDATA`` namelist and its structured JSON
schema, and writes the JSON form, so it is a drop-in for either ecosystem:
``vmex input.json`` and ``vmex input.circular_tokamak`` both work.

The steps: read an INDATA deck, write JSON with ``inp.to_json``, read that back
with ``VmecInput.from_file`` (which dispatches on the suffix or a leading brace),
solve both, and compare. The two representations describe one equilibrium, so
the aspect ratios agree to solver tolerance.

The deck is a circular tokamak, one field period, zero pressure.
"""

import os
from dataclasses import replace
from pathlib import Path

import vmex as vj

# Input deck:
INPUT_FILE = Path(__file__).resolve().parent / "data" / "input.circular_tokamak"

# Fields of the deck to replace before solving; empty runs it as written:
INPUT_OVERRIDES = {}

# Directory that receives every output file:
OUTPUT_DIR = Path("output_run_from_json")

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    INPUT_OVERRIDES = dict(ns_array=[15], ftol_array=[1e-10], niter_array=[2000])

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

def solve(inp):
    """Converge one deck and return its wout; called once per representation."""
    inp = replace(inp, **INPUT_OVERRIDES)
    res = vj.solve_multigrid(inp, verbose=False)
    wout = vj.wout_from_state(
        inp=inp, state=res.state, fsqr=float(res.fsqr), fsqz=float(res.fsqz),
        fsql=float(res.fsql), niter=int(res.iterations),
        converged=bool(res.converged))
    return wout


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
case = INPUT_FILE.name.removeprefix("input.")
inp_indata = vj.VmecInput.from_file(INPUT_FILE)
json_path = inp_indata.to_json(OUTPUT_DIR / f"{case}.json")
print(f"read {INPUT_FILE.name} (&INDATA) -> wrote {json_path} (structured JSON)")

# from_file dispatches on the .json suffix (or a leading '{') to the JSON parser.
inp_json = vj.VmecInput.from_file(json_path)
print(f"read {json_path.name} back: nfp={inp_json.nfp} mpol={inp_json.mpol} "
      f"ntor={inp_json.ntor} lasym={inp_json.lasym}")

### Solve both representations ################################################

wout_indata = solve(inp_indata)
wout_json = solve(inp_json)
d_aspect = abs(float(wout_indata.aspect) - float(wout_json.aspect))
print(f"aspect: INDATA={float(wout_indata.aspect):.6f}  "
      f"JSON={float(wout_json.aspect):.6f}  |diff|={d_aspect:.2e}")

### Print, plot and save ######################################################

wout_path = vj.write_wout(OUTPUT_DIR / f"wout_{case}.nc", wout_json)
print(f"Wrote {wout_path} from the JSON input "
      "(the two input formats describe one equilibrium)")
