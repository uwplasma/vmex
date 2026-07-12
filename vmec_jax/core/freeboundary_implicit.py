"""Coupled residual for implicit differentiation of free-boundary solves.

The existing :mod:`vmec_jax.core.freeboundary_diff` module differentiates a
coil objective on a fixed trial surface.  This module instead linearizes the
discrete equations that determine the plasma state: a NESTOR vacuum solve is
performed for the current edge, its magnetic pressure is inserted in the MHD
edge force, and the same preconditioned force used by the VMEC iteration is
returned.  The edge coefficients remain active.

This focused layer owns only differentiation orchestration.  Vacuum and MHD
physics stay in :mod:`vmec_jax.core.freeboundary` and
:mod:`vmec_jax.core.solver`, respectively.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .freeboundary import (
    FreeBoundaryState,
    _presf_ns_scale,
    _vacuum_scalars,
    _vacuum_executables,
)
from .input import VmecInput
from .mgrid import MgridField
from .solver import (
    ForceResiduals,
    SolveResult,
    SolverRuntime,
    SpectralState,
    evaluate_forces,
    prepare_runtime,
    resolution_from_input,
)

__all__ = ["CoupledFreeBoundaryProblem"]


@dataclass(frozen=True, eq=False)
class CoupledFreeBoundaryProblem:
    """Pure-JAX free-boundary fixed-point residual around a solved state.

    Construct with :meth:`from_result`.  :meth:`residual` is differentiable
    with respect to both ``state`` and mgrid ``extcur`` and therefore supplies
    the two Jacobian actions required by an implicit-function adjoint.  A
    linear solver and scalar-output adjoint are intentionally separate from
    this physics contract.
    """

    runtime: SolverRuntime
    external_field: MgridField
    fused_vacuum: Any
    reference_state: SpectralState

    @classmethod
    def from_result(
        cls,
        inp: VmecInput,
        result: SolveResult,
        external_field: MgridField,
    ) -> "CoupledFreeBoundaryProblem":
        """Reconstruct the discrete residual used by a converged solve.

        Parameters
        ----------
        inp:
            The free-boundary input used for ``result``.
        result:
            A converged :func:`~vmec_jax.core.freeboundary.solve_free_boundary`
            result.  Its retained NESTOR state supplies the final constraint
            baselines.
        external_field:
            The mgrid field used by the solve, including its current vector.
        """
        if not bool(inp.lfreeb):
            raise ValueError("CoupledFreeBoundaryProblem requires LFREEB=T")
        if not result.converged:
            raise ValueError("implicit differentiation requires a converged result")
        vacuum = result.vacuum_state
        if not isinstance(vacuum, FreeBoundaryState) or vacuum.rcon0 is None:
            raise ValueError("result does not retain final free-boundary constraint state")
        if not isinstance(external_field, MgridField):
            raise TypeError("coupled extcur derivatives currently require MgridField")

        resolution = resolution_from_input(inp)
        rt = prepare_runtime(inp, resolution)
        dtype = rt.setup.s_full.dtype
        ns = int(resolution.ns)
        rt = replace(
            rt,
            lfreeb=True,
            jmax=ns,
            rcon0=jnp.asarray(vacuum.rcon0, dtype=dtype),
            zcon0=jnp.asarray(vacuum.zcon0, dtype=dtype),
            bsqvac_edge=jnp.zeros((resolution.ntheta3, resolution.nzeta), dtype=dtype),
            presf_ns_scale=jnp.asarray(_presf_ns_scale(inp, ns), dtype=dtype),
        )
        axis_r, axis_z = _vacuum_scalars(result.state, rt)[2:4]
        _basis, fused = _vacuum_executables(
            resolution,
            mf=int(inp.mpol) + 1,
            nf=int(inp.ntor),
            signgs=int(rt.setup.signgs),
            wint=np.asarray(rt.trig.wint, dtype=float),
            modes=rt.modes,
            axis_r0=axis_r,
            axis_z0=axis_z,
        )
        return cls(rt, external_field, fused, result.state)

    def _forces(
        self, state: SpectralState, extcur: Any
    ) -> tuple[SpectralState, ForceResiduals]:
        field = replace(self.external_field, extcur=jnp.asarray(extcur))
        vacuum = self.fused_vacuum.full(state, self.runtime, field)
        runtime = replace(self.runtime, bsqvac_edge=vacuum["bsqvac"])
        force, residuals, _diagnostics = evaluate_forces(state, runtime)
        return force, residuals

    def residual(self, state: SpectralState, extcur: Any) -> SpectralState:
        """Return the coupled NESTOR-MHD preconditioned force pytree."""
        return self._forces(state, extcur)[0]

    def force_residuals(self, state: SpectralState, extcur: Any) -> ForceResiduals:
        """Return VMEC's normalized ``fsqr/fsqz/fsql`` diagnostics."""
        return self._forces(state, extcur)[1]

    def residual_objective(self, state: SpectralState, extcur: Any) -> jax.Array:
        """Mean squared coupled residual over all packed state entries."""
        leaves = jax.tree.leaves(self.residual(state, extcur))
        total = sum(jnp.sum(x * x) for x in leaves)
        count = sum(int(x.size) for x in leaves)
        return total / float(count)
