#!/usr/bin/env python
"""Differentiate a true NESTOR free-boundary solve through its coupled root.

"""

from dataclasses import replace
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import vmex as vj
from vmex.core import implicit as im
from vmex.core.freeboundary_implicit import (
    make_free_boundary_config,
    solve_free_boundary_implicit,
)

from essos.coils import Coils
from essos.fields import BiotSavart

DATA = Path(__file__).resolve().parent / "data"
NS, MPOL, NTOR, NITER, FTOL = 25, 5, 5, 12000, 1.0e-10
CI = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if CI:
    NS, MPOL, NTOR, NITER, FTOL = 12, 2, 2, 3000, 1.0e-7

print("Loading the direct ESSOS Biot-Savart field (no mgrid file)...")
coils = Coils.from_json(str(DATA / "ESSOS_biot_savart_LandremanPaulQA.json"))
if CI:
    coils.n_segments = 24
biot_savart = BiotSavart(coils)
base_dofs = coils.dofs
n_curve_dofs = coils.dofs_curves.size
scales = jnp.concatenate((jnp.full(n_curve_dofs, 0.02),
                          0.20 * jnp.abs(base_dofs[n_curve_dofs:])))


def field_from_parameters(parameters):
    return BiotSavart(coils.with_dofs(base_dofs + scales * parameters))

# The fixed-boundary surface is only the initial guess. NESTOR evolves its
# edge against the ESSOS field, while no mgrid file is written or read.
inp = vj.VmecInput.from_file(DATA / "input.LandremanPaul2021_QA_lowres")
inp = inp.change_resolution(
    mpol=MPOL, ntor=NTOR, ntheta=2 * MPOL + 6, nzeta=16)
inp = replace(inp, lfreeb=True, mgrid_file="direct ESSOS field",
              phiedge=-0.025, ns_array=np.array([NS]),
              niter_array=np.array([NITER]), ftol_array=np.array([FTOL]))
params = im.params_from_input(inp)


def configured(**overrides):
    config = make_free_boundary_config(
        inp, biot_savart, ns=NS, ftol=FTOL, max_iterations=NITER,
        adjoint_tol=1.0e-9,
        field_from_parameters=field_from_parameters, **overrides)
    context = im.runtime_from_params(params, config.implicit)

    def aspect_from_coils(parameters):
        state = solve_free_boundary_implicit(params, parameters, config)
        return im.aspect_ratio(state, context)

    return aspect_from_coils


aspect_from_coils = configured()

print("Solving the free boundary and its implicit adjoint...")
parameters = jnp.zeros(base_dofs.size)
aspect, gradient = jax.value_and_grad(aspect_from_coils)(parameters)
# One normalized direction changes a curve Fourier coefficient and all base
# currents. Independent re-solves therefore certify both ESSOS derivative paths.
direction = jnp.zeros_like(parameters).at[2].set(0.1)
direction = direction.at[-coils.curves.n_base_curves:].set(
    1.0 / coils.curves.n_base_curves)
direction /= jnp.linalg.norm(direction)
autodiff = jnp.vdot(gradient, direction)

# The certificate is the second adjoint, not a difference quotient. A free
# boundary re-solve is path dependent, so a central difference here has no
# usable step: above about 1e-2 truncation dominates and below it the solve's
# own floor makes the quotient change sign. The coupled GCROT adjoint and the
# edge Schur adjoint are independent solvers of the same linear system, so
# agreement between them certifies both. How close they agree is set by how
# tight a root the equilibrium is: 1.6e-04 at the settings above and 1.6e-02
# at the smoke-test settings, where ftol is 1e-7.
schur_gradient = jax.grad(configured(adjoint_solver="boundary_schur"))(parameters)
schur = jnp.vdot(schur_gradient, direction)
disagreement = jnp.abs(autodiff - schur) / jnp.abs(schur)

step = 1.0e-1
finite_difference = (aspect_from_coils(parameters + step * direction)
                     - aspect_from_coils(parameters - step * direction)) / (2 * step)

print(f"aspect = {float(aspect):.6f}")
print(f"directional d(aspect)/d(coils): coupled GCROT = {float(autodiff):.8e}, "
      f"edge Schur = {float(schur):.8e}, they differ by {float(disagreement):.2e}")
print(f"central FD at step {step:g} = {float(finite_difference):.6e} "
      "(step-limited; see the comment above)")
