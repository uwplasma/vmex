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

# This example has no VMEX_EXAMPLES_CI smoke path, deliberately.  Coarsening
# the NEO controls enough to matter destroys the result: at theta_n/phi_n 12,
# npart 6 and acc_req 0.3 the profile came back about four times too large and
# no longer rose outward, for a saving of 91 s to 54 s.  Its test is gated on
# the optional neoclassical extra rather than on runtime, so a cheaper lane
# would buy nothing and cost the radial trend this example exists to show.

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
