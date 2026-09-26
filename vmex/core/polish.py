"""Native coefficient representation shared by the polish.

The continuous state stores clamped B-spline coefficients of the regularized
``rho**abs(m) q(s)`` Fourier amplitudes.  This module packs the free polish
coordinates (R/Z/lambda with the fixed edge and the lambda gauge removed),
applies a correction, and samples a native state on a VMEX radial mesh for
WOUT export (the inverse of ``lift_high_order_state``'s representation
changes).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .residuals import m1_physical_to_constrained
from .solver import SpectralState
from .strong_force import HighOrderEquilibriumState
from .transforms import physical_to_internal_scale

Array = Any
_FIELDS = ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")


@dataclass(frozen=True)
class HighOrderCorrection:
    """Regularized spline coefficients for one geometry/lambda correction."""

    R_cos: Array
    R_sin: Array
    Z_cos: Array
    Z_sin: Array
    L_cos: Array
    L_sin: Array


jax.tree_util.register_dataclass(
    HighOrderCorrection,
    data_fields=list(_FIELDS),
    meta_fields=[],
)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class NativeCorrectionLayout:
    """Direct structural coordinates for native spline corrections.

    Unlike :class:`StrongRootLayout`, this layout is not defined as the image
    of a legacy restriction/prolongation pair.  It removes only exact
    structural nulls, the fixed R/Z edge coefficients, stellarator-asymmetric
    channels when requested, and the angle-independent lambda gauge.  In
    particular, independent R and Z normal displacements remain present.
    """

    mnmax: int
    nbasis: int
    active_indices: np.ndarray
    lasym: bool

    def tree_flatten(self):
        """Expose active indices as data and topology as metadata."""

        return (jnp.asarray(self.active_indices),), (
            int(self.mnmax),
            int(self.nbasis),
            bool(self.lasym),
        )

    @classmethod
    def tree_unflatten(cls, metadata, children):
        """Rebuild a native layout from its pytree representation."""

        mnmax, nbasis, lasym = metadata
        (active_indices,) = children
        return cls(
            mnmax=mnmax,
            nbasis=nbasis,
            active_indices=active_indices,
            lasym=lasym,
        )

    @property
    def size(self) -> int:
        """Number of structurally free native coefficients."""

        return int(self.active_indices.size)

    def pack(self, correction: HighOrderCorrection) -> Array:
        """Select free coefficients from a native correction."""

        flat = _flatten_high(correction)
        return flat[jnp.asarray(self.active_indices)]

    def unpack(self, vector: Array) -> HighOrderCorrection:
        """Insert a free vector into the six native coefficient tables."""

        vector = jnp.asarray(vector)
        if vector.shape != (self.size,):
            raise ValueError(
                f"free vector has shape {vector.shape}; expected {(self.size,)}"
            )
        flat = jnp.zeros(
            (len(_FIELDS) * self.mnmax * self.nbasis,), dtype=vector.dtype
        )
        flat = flat.at[jnp.asarray(self.active_indices)].set(vector)
        return _unflatten_high(flat, self.mnmax, self.nbasis)


def make_native_correction_layout(
    native: HighOrderEquilibriumState,
    *,
    lasym: bool = False,
) -> NativeCorrectionLayout:
    """Construct full native R/Z/lambda structural coordinates.

    A clamped spline's final coefficient is its value at ``rho=1``; removing
    that coefficient from R and Z corrections enforces the fixed boundary
    algebraically.  Lambda remains free at the edge, except for its complete
    ``(m,n)=(0,0)`` gauge family.
    """

    mnmax = int(np.asarray(native.m).size)
    nbasis = int(native.radial_basis.size)
    active = np.ones((len(_FIELDS), mnmax, nbasis), dtype=bool)
    if not lasym:
        for name in ("R_sin", "Z_cos", "L_cos"):
            active[_FIELDS.index(name)] = False
    mode_zero = (np.asarray(native.m, dtype=int) == 0) & (
        np.asarray(native.n, dtype=int) == 0
    )
    for name in ("R_sin", "Z_sin", "L_sin"):
        active[_FIELDS.index(name), mode_zero] = False
    for name in ("R_cos", "R_sin", "Z_cos", "Z_sin"):
        active[_FIELDS.index(name), :, -1] = False
    for name in ("L_cos", "L_sin"):
        active[_FIELDS.index(name), mode_zero] = False
    return NativeCorrectionLayout(
        mnmax=mnmax,
        nbasis=nbasis,
        active_indices=np.flatnonzero(active.reshape(-1)).astype(np.int32),
        lasym=bool(lasym),
    )


def _flatten_high(correction: HighOrderCorrection) -> Array:
    return jnp.concatenate(
        tuple(jnp.ravel(jnp.asarray(getattr(correction, name))) for name in _FIELDS)
    )


def _unflatten_high(vector: Array, mnmax: int, nbasis: int) -> HighOrderCorrection:
    vector = jnp.asarray(vector)
    block = int(mnmax) * int(nbasis)
    values = [
        vector[index * block : (index + 1) * block].reshape((mnmax, nbasis))
        for index in range(len(_FIELDS))
    ]
    return HighOrderCorrection(*values)




def _mode_table(m: np.ndarray, n: np.ndarray):
    """Construct only the mode metadata needed by the m=1 linear maps."""

    from .fourier import ModeTable

    return ModeTable(m=np.asarray(m), n=np.asarray(n))


def sample_high_order_state(
    native: HighOrderEquilibriumState,
    runtime: Any,
) -> SpectralState:
    """Evaluate a continuous native state on ``runtime``'s legacy full mesh.

    The exact inverse of the representation changes performed by
    :func:`~vmex.core.strong_force.lift_high_order_state`: clamped-spline
    evaluation with the ``rho**abs(m)`` regularity factor, VMEX Fourier
    normalization, the m=1 constrained variables, and the internal
    ``phipf/lamscale`` lambda scaling.  Unlike :meth:`HighLowTransfer.restrict`
    this samples a full equilibrium rather than a correction, so the boundary
    row and the non-evolved degrees of freedom are kept, not zeroed.  The
    target mesh is whatever ``runtime`` was prepared with — it does not have
    to match the mesh the native state was lifted from.
    """

    modes = runtime.modes
    m = np.asarray(modes.m, dtype=int)
    n = np.asarray(modes.n, dtype=int)
    if not np.array_equal(m, np.asarray(native.m)) or not np.array_equal(
        n, np.asarray(native.n)
    ):
        raise ValueError("native and legacy mode tables must match")
    setup = runtime.setup
    s = np.asarray(setup.s_full, dtype=float)
    rho = np.sqrt(np.maximum(s, 0.0))
    basis_values = np.asarray(native.radial_basis.basis_matrix(s), dtype=float)
    evaluation = jnp.asarray(
        rho[None, :, None] ** np.abs(m)[:, None, None] * basis_values[None]
    )

    def sample(coefficients: Array) -> Array:
        return jnp.einsum(
            "msk,mk->sm",
            evaluation,
            jnp.asarray(coefficients),
            precision=jax.lax.Precision.HIGHEST,
        )

    scale = jnp.asarray(physical_to_internal_scale(modes, runtime.trig))[None, :]
    R_cos, Z_sin, R_sin, Z_cos = m1_physical_to_constrained(
        sample(native.R_cos) * scale,
        sample(native.Z_sin) * scale,
        sample(native.R_sin) * scale,
        sample(native.Z_cos) * scale,
        modes=_mode_table(m, n),
        lthreed=bool(setup.lthreed),
        lasym=bool(setup.lasym),
        lconm1=bool(setup.lconm1),
    )
    lambda_scale = (
        scale * jnp.asarray(setup.phipf)[:, None] / jnp.asarray(setup.lamscale)
    )
    return SpectralState(
        R_cos=R_cos,
        R_sin=R_sin,
        Z_cos=Z_cos,
        Z_sin=Z_sin,
        L_cos=sample(native.L_cos) * lambda_scale,
        L_sin=sample(native.L_sin) * lambda_scale,
    )


def apply_high_order_correction(
    native: HighOrderEquilibriumState,
    correction: HighOrderCorrection,
) -> HighOrderEquilibriumState:
    """Add a constrained geometry correction while leaving profiles fixed."""

    return replace(
        native,
        R_cos=native.R_cos + correction.R_cos,
        R_sin=native.R_sin + correction.R_sin,
        Z_cos=native.Z_cos + correction.Z_cos,
        Z_sin=native.Z_sin + correction.Z_sin,
        L_cos=native.L_cos + correction.L_cos,
        L_sin=native.L_sin + correction.L_sin,
        source=f"{native.source}; strong-root correction",
    )


__all__ = [
    "HighOrderCorrection",
    "NativeCorrectionLayout",
    "apply_high_order_correction",
    "make_native_correction_layout",
    "sample_high_order_state",
]
