#!/usr/bin/env python
"""Solve one equilibrium twice, from power-series and from spline profiles.

VMEC prescribes the pressure and either the rotational transform (``NCURR=0``)
or the toroidal current (``NCURR=1``) as radial profiles in the normalized flux
``s``. Each can be given as polynomial coefficients (``power_series``) or as
spline knots (``cubic_spline`` / ``akima_spline``).

This solves the same shaped tokamak twice -- once with the deck's power-series
profiles, once with cubic-spline knots sampled from those very polynomials --
and shows the two converge to the same equilibrium. That is the point: the
representation is a modelling convenience, not different physics.

The deck uses ``NCURR=0`` with a parabolic pressure and a linear iota. To
prescribe the current instead, set ``ncurr=1`` with a ``pcurr_type`` and an
``ac`` shape, and give the total through ``CURTOR``; VMEC then solves for iota.
"""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np

import vmex as vj

# Input deck:
INPUT_FILE = Path(__file__).resolve().parent / "data" / "input.shaped_tokamak_pressure"

# Where the spline samples the power-series profiles, in normalized flux:
KNOTS = np.array([0.0, 0.25, 0.50, 0.75, 1.0])

# Fields of the deck to replace before solving; empty runs it as written:
INPUT_OVERRIDES = {}

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    INPUT_OVERRIDES = dict(ns_array=[13], ftol_array=[1e-10], niter_array=[3000])

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

def polyval(coeffs, s):
    """VMEC power-series convention: profile(s) = sum_k coeffs[k] * s**k."""
    return sum(c * s**k for k, c in enumerate(coeffs))


def nonzero_terms(coeffs):
    """The meaningful (non-zero-padded) power-series coefficients as floats."""
    arr = np.asarray(coeffs, dtype=float)
    last = int(np.max(np.nonzero(arr))) if np.any(arr) else 0
    return arr[: last + 1].tolist()


def solve(inp, label):
    """Converge one profile representation and print its wout scalars."""
    inp = replace(inp, **INPUT_OVERRIDES)
    res = vj.solve_multigrid(inp, verbose=False)
    wout = vj.wout_from_state(
        inp=inp, state=res.state, fsqr=float(res.fsqr), fsqz=float(res.fsqz),
        fsql=float(res.fsql), niter=int(res.iterations),
        converged=bool(res.converged))
    print(f"  {label:14s} converged={bool(res.converged)!s:5s} "
          f"aspect={float(wout.aspect):.4f} volume={float(wout.volume_p):.4f} "
          f"beta={float(wout.betatotal):.4e}")
    return wout


### Solve both representations ################################################

inp_power = vj.VmecInput.from_file(INPUT_FILE)
print(f"power-series profiles: AM={nonzero_terms(inp_power.am)}  "
      f"AI={nonzero_terms(inp_power.ai)}")
wout_power = solve(inp_power, "power_series")

# Sample the SAME polynomials at the knot locations, then hand VMEC the knots.
am_knots = polyval(np.asarray(inp_power.am), KNOTS)
ai_knots = polyval(np.asarray(inp_power.ai), KNOTS)
inp_spline = replace(
    inp_power,
    pmass_type="cubic_spline", am_aux_s=KNOTS.copy(), am_aux_f=am_knots,
    piota_type="cubic_spline", ai_aux_s=KNOTS.copy(), ai_aux_f=ai_knots,
)
print(f"cubic-spline profiles: AM_AUX_F={np.round(am_knots, 4).tolist()}  "
      f"AI_AUX_F={np.round(ai_knots, 4).tolist()}")
wout_spline = solve(inp_spline, "cubic_spline")

### Print, plot and save ######################################################

d_aspect = abs(float(wout_power.aspect) - float(wout_spline.aspect))
d_beta = abs(float(wout_power.betatotal) - float(wout_spline.betatotal))
print(f"\nrepresentation-independent: |d aspect| = {d_aspect:.2e}, "
      f"|d beta| = {d_beta:.2e} (the two profile forms describe one equilibrium)")

