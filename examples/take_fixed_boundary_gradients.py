#!/usr/bin/env python
"""Exact fixed-boundary gradients of a converged equilibrium, checked against FD.

The fixed-boundary companion to ``take_free_boundary_gradients.py``, with no
coils and no ESSOS. ``vj.implicit.run`` converges the equilibrium and exposes
the wout scalars (``aspect``, ``wb`` magnetic energy, ``volume``,
``iota_edge``, ...) as JAX values, so ``jax.grad`` returns their derivatives
with respect to the boundary Fourier coefficients and the profile and flux
parameters.

The gradient comes from the implicit function theorem at the converged fixed
point: one adjoint linear solve per scalar output, not a backward pass through
the iteration. Its memory does not grow with the iteration count and it has no
step size to tune. The script checks two derivatives against central finite
differences of independent re-solves:

- ``d(aspect)/d(RBC(0,1))``, a boundary shape coefficient;
- ``d(wb)/d(phiedge)``, the enclosed toroidal flux.

The deck is the Solovev analytic tokamak at ns = 11, which takes seconds once
the solver is compiled.
"""

import dataclasses
from pathlib import Path

import jax
import numpy as np

import vmex as vj

# Input deck:
INPUT_FILE = Path(__file__).resolve().parent / "data" / "input.solovev"

# Forward solve; the adjoint is exact at the fixed point, so solve tightly:
FTOL = 1e-12
MAX_ITERATIONS = 5000

# Central finite-difference steps for the boundary coefficient and phiedge:
STEP_RBC = 3e-5
STEP_PHIEDGE = 1e-5

###############################################################################
# End of input parameters.
###############################################################################

### Set up the parameters #####################################################

im = vj.implicit  # the differentiable fixed-point solver and its wout scalars
inp = vj.VmecInput.from_file(INPUT_FILE)
ntor = int(inp.ntor)
p0 = im.params_from_input(inp)  # pytree of rbc, zbs, phiedge, pres_scale, am, ...


def scalar(params, name):
    """Converge the equilibrium and return one differentiable wout scalar."""
    solution = im.run(inp, params, ftol=FTOL, max_iterations=MAX_ITERATIONS)
    return getattr(solution, name)


### Adjoint gradients #########################################################

# Each jax.grad is one forward solve and one adjoint solve. The aspect ratio
# depends strongly on the m = 1 boundary mode, the magnetic energy on phiedge.
aspect, gradient_aspect = jax.value_and_grad(lambda p: scalar(p, "aspect"))(p0)
wb, gradient_wb = jax.value_and_grad(lambda p: scalar(p, "wb"))(p0)
print(f"solovev ns={int(inp.ns_array[-1])}, ftol={FTOL:g}:  "
      f"aspect = {float(aspect):.8f}   wb = {float(wb):.8e}")
adjoint_rbc = float(np.asarray(gradient_aspect.rbc)[ntor, 1])
adjoint_phiedge = float(np.asarray(gradient_wb.phiedge))

### Finite-difference check ###################################################


def central_difference(name, perturb, step):
    """Central difference of a wout scalar over two independent re-solves."""
    plus = float(scalar(perturb(+step), name))
    minus = float(scalar(perturb(-step), name))
    return (plus - minus) / (2.0 * step)


def with_rbc_01(delta):
    rbc = np.array(p0.rbc)
    rbc[ntor, 1] += delta
    return dataclasses.replace(p0, rbc=rbc)


def with_phiedge(delta):
    return dataclasses.replace(p0, phiedge=p0.phiedge + delta)


fd_rbc = central_difference("aspect", with_rbc_01, STEP_RBC)
fd_phiedge = central_difference("wb", with_phiedge, STEP_PHIEDGE)
print(f"\nd(aspect)/d(RBC(0,1))  AD={adjoint_rbc:+.10e}  FD={fd_rbc:+.10e}  "
      f"rel={abs(adjoint_rbc / fd_rbc - 1.0):.2e}")
print(f"d(wb)/d(phiedge)       AD={adjoint_phiedge:+.10e}  FD={fd_phiedge:+.10e}  "
      f"rel={abs(adjoint_phiedge / fd_phiedge - 1.0):.2e}")

# Several outputs of one solve at once: jax.jacrev of a stacked vector costs
# one forward solve plus one adjoint solve per output row, for example
#   def outputs(p):
#       solution = im.run(inp, p, ftol=FTOL, max_iterations=MAX_ITERATIONS)
#       return jax.numpy.stack([solution.aspect, solution.wb])
#   jacobian = jax.jacrev(outputs)(p0)
