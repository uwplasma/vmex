#!/usr/bin/env python
"""Compute the effective ripple of a solved equilibrium, differentiably.

``epsilon_eff^(3/2)`` measures the neoclassical transport a stellarator's field
ripple drives in the 1/nu regime. VMEX evaluates Nemov's bounce integrals along
Boozer field lines (NEO_JAX) directly from the solver state (Boozer transform included, no
``boozmn`` or WOUT file), so the same call is a differentiable objective: see
``test_neoclassical.py`` for its gradient with respect to boundary coefficients.

A quasi-axisymmetric field has a very small ripple (about 1e-8 for this deck),
so the profile there is a statement about the residual symmetry breaking and is
not monotonic in s.  The default ``neo_jax.RippleBudget`` holds 1 % on non-quasi-symmetric
fields (raise it below about 1e-7); it agrees with converged NEO_JAX, xneo and DESC runs to
0.2 to 3 %.

Needs the optional ``neoclassical`` extra: ``pip install "vmex[neoclassical]"``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import vmex as vj
from vmex import optimize as opt

# Input deck:
INPUT_FILE = Path(__file__).resolve().parent / "data" / "input.LandremanPaul2021_QA_lowres"

# Flux surfaces the effective ripple is evaluated on (linear interpolation in s
# between the two bracketing half-mesh rows):
SURFACES = np.linspace(0.15, 0.95, 5)

# Figure written to the working directory:
FIGURE_PATH = Path("epsilon_effective.png")

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

print("Solving the VMEX equilibrium...")
equilibrium = opt.solve_equilibrium(vj.VmecInput.from_file(INPUT_FILE))

### Compute the effective ripple ##############################################

print("Transforming to Boozer coordinates and calculating epsilon_eff^(3/2)...")
epsilon_effective_3_2 = vj.epsilon_effective_3_2_from_state(
    equilibrium.state, equilibrium.runtime, SURFACES)

### Print, plot and save ######################################################

print("s =", SURFACES)
print("epsilon_eff^(3/2) =", np.asarray(epsilon_effective_3_2))

figure, axis = plt.subplots(figsize=(5.2, 3.8))
axis.semilogy(SURFACES, epsilon_effective_3_2, "o-")
axis.set(xlabel=r"$s=\psi/\psi_b$", ylabel=r"$\epsilon_{\rm eff}^{3/2}$")
axis.grid(alpha=0.25, which="both")
figure.tight_layout()
figure.savefig(FIGURE_PATH, dpi=200)
plt.close(figure)
print(f"Wrote {FIGURE_PATH}")
