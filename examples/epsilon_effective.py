#!/usr/bin/env python
"""Compute the NEO_JAX effective ripple of a solved equilibrium.

``epsilon_eff^(3/2)`` measures the neoclassical transport a stellarator's field
ripple drives in the 1/nu regime. VMEX computes it straight from a solved
equilibrium through NEO_JAX, with the Boozer transform done internally, so no
``boozmn`` file is written.

The radial trend is the result worth reading: a quasi-axisymmetric field has a
small ripple that rises toward the boundary. Raise the NEO controls for
anything beyond that trend.

Needs the optional ``neoclassical`` extra: ``pip install "vmex[neoclassical]"``.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from neo_jax import NeoConfig

import vmex as vj
from vmex import optimize as opt

# Input deck:
INPUT_FILE = Path(__file__).resolve().parent / "data" / "input.LandremanPaul2021_QA_lowres"

# Flux surfaces the effective ripple is evaluated on:
SURFACES = np.linspace(0.15, 0.95, 5)

# NEO controls.  This compact set keeps the example responsive while retaining
# the radial trend; raise them for a publication calculation.  Setting
# max_rational_field_periods to 0 asks for an unlimited exact rational
# correction, which on a near-rational surface does not finish in an example's
# budget, so it is bounded here:
NEO_CONFIG_ARGS = dict(theta_n=24, phi_n=24, npart=12, multra=1, no_bins=20,
                       nstep_per=6, nstep_min=30, nstep_max=60, acc_req=0.1,
                       max_rational_field_periods=100000)

# Figure written to the working directory:
FIGURE_PATH = Path("epsilon_effective.png")

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs.  The Boozer
# transform per surface is the cost, so the smoke pass coarsens NEO and samples
# fewer surfaces:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
# SURFACES is deliberately unchanged: the radial trend is the result, and
# tests/test_examples.py checks it over all five.
if ci_smoke:
    NEO_CONFIG_ARGS = dict(theta_n=12, phi_n=12, npart=6, multra=1, no_bins=10,
                           nstep_per=3, nstep_min=15, nstep_max=30, acc_req=0.3,
                           max_rational_field_periods=100000)

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

print("Solving the VMEX equilibrium...")
equilibrium = opt.solve_equilibrium(vj.VmecInput.from_file(INPUT_FILE))

### Compute the effective ripple ##############################################

print("Transforming to Boozer coordinates and calculating epsilon_eff^(3/2)...")
NEO_CONFIG = NeoConfig(**NEO_CONFIG_ARGS)
s, epsilon_effective_3_2 = vj.epsilon_effective_from_wout(
    equilibrium.wout, surfaces=SURFACES, config=NEO_CONFIG)
### Print, plot and save ######################################################

print("s =", np.asarray(s))
print("epsilon_eff^(3/2) =", np.asarray(epsilon_effective_3_2))

figure, axis = plt.subplots(figsize=(5.2, 3.8))
axis.semilogy(s, epsilon_effective_3_2, "o-")
axis.set(xlabel=r"$s=\psi/\psi_b$", ylabel=r"$\epsilon_{\rm eff}^{3/2}$")
axis.grid(alpha=0.25, which="both")
figure.tight_layout()
figure.savefig(FIGURE_PATH, dpi=200)
plt.close(figure)
print(f"Wrote {FIGURE_PATH}")
