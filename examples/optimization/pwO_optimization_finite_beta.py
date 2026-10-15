#!/usr/bin/env python
"""Stage-1 finite-beta pwO fitting with pressure continuation and zero current.

The shared, side-effect-free driver in pwO_optimization supplies the seed,
fixed-target canonical / squeezed-mapping selection, boundary mode continuation,
constraints and final refined solve/rescore. Set VMEX_PWO_FAMILY=canonical or
mapped and VMEX_PWO_METHOD=scalar or least_squares, as in the vacuum example.

p(s)=PRES_SCALE*(1-s) vanishes at the edge. NCURR=1, CURTOR=0 and AC=0 specify
zero enclosed toroidal current. PRES_SCALE is estimated from a unit-pressure
seed solve for each beta rung, then fixed during that rung; the actual beta is
constrained and reported. This is not a self-consistent bootstrap calculation.
"""

from pwO_optimization import run

BETA_TARGETS = (0.0, 0.0025, 0.005)

if __name__ == "__main__":
    run(beta_targets=BETA_TARGETS, output_name="pwO_finite_beta_optimized")
