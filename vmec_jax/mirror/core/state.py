"""State containers for mirror equilibria."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vmec_jax._compat import tree_util


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MirrorStateAxisym:
    """Axisymmetric nodal mirror state.

    ``a`` has shape ``(ns, nxi)`` and defines the physical radius by
    ``r = sqrt(s) * a``.  ``lam`` carries the VMEC-like field-line stream
    function for later field/energy phases.
    """

    a: np.ndarray
    lam: np.ndarray

    def __post_init__(self):
        a = np.asarray(self.a)
        lam = np.asarray(self.lam)
        if a.ndim != 2:
            raise ValueError("axisymmetric mirror a must have shape (ns, nxi)")
        if lam.shape != a.shape:
            raise ValueError("axisymmetric mirror lam must have the same shape as a")
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "lam", lam)

    @classmethod
    def from_boundary(cls, grid, boundary, *, lam=None, project: bool = True) -> "MirrorStateAxisym":
        """Build the radial initial guess ``r = sqrt(s) * r_b(xi)``."""
        boundary_radius = boundary.radius_on_grid(grid)
        a = np.broadcast_to(boundary_radius[None, :], (grid.ns, grid.nxi)).copy()
        if lam is None:
            lam = np.zeros_like(a)
        state = cls(a=a, lam=np.asarray(lam, dtype=a.dtype))
        if not project:
            return state
        from ..kernels.constraints import project_axisym_state

        return project_axisym_state(state, grid, boundary)

    @property
    def shape(self) -> tuple[int, int]:
        return self.a.shape

    def tree_flatten(self):
        return (self.a, self.lam), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        a, lam = children
        return cls(a=a, lam=lam)
