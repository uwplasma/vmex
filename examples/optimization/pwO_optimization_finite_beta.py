#!/usr/bin/env python
"""Fit pwO targets with pressure continuation and zero enclosed current.

Uses the vacuum driver's settings and validation. Pressure follows
p(s)=PRES_SCALE*(1-s); each stage fixes PRES_SCALE after seed calibration.
The prescribed current is not bootstrap-consistent.
"""

if __package__:
    from .pwO_optimization import run
else:
    from pwO_optimization import run

BETA_TARGETS = (0.0, 0.0025, 0.005)

if __name__ == "__main__":
    run(beta_targets=BETA_TARGETS, output_name="pwO_finite_beta_optimized")
