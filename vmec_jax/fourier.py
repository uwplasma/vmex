"""Helical Fourier basis utilities.

VMEC represents surfaces using Fourier series in the helical phase::

    phase = m*theta - n*zeta

where ``zeta`` is the *field-period* toroidal angle (one field period spans
``[0, 2π)``).

This module provides:

- basis matrices (cos/sin of phase) for a set of (m,n) modes,
- evaluation of a scalar field from (cos, sin) Fourier coefficients,
- evaluation of angular derivatives, including the physical-toroidal derivative
  that includes the NFP scaling used in VMEC.

All functions are backend-agnostic: they use ``jax.numpy`` if available,
otherwise NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np

from ._compat import jnp, jit, has_jax, einsum
from .modes import ModeTable
from .grids import AngleGrid


_HELICAL_BASIS_CACHE: dict[tuple, "HelicalBasis"] = {}


def _cache_allowed() -> bool:
    if not has_jax():
        return True
    try:
        from jax import core

        return bool(core.trace_ctx.is_top_level())
    except Exception:
        return False


def _basis_cache_key(modes: ModeTable, grid: AngleGrid) -> tuple:
    m = np.asarray(modes.m)
    n = np.asarray(modes.n)
    theta = np.asarray(grid.theta)
    zeta = np.asarray(grid.zeta)
    return (
        int(grid.nfp),
        str(m.dtype),
        str(n.dtype),
        m.tobytes(),
        n.tobytes(),
        str(theta.dtype),
        str(zeta.dtype),
        theta.shape,
        zeta.shape,
        theta.tobytes(),
        zeta.tobytes(),
    )

# Make HelicalBasis a JAX PyTree so it can be passed through jitted functions.
# Registration is made idempotent to avoid duplicate-registration errors
# in notebook/reload workflows.
if has_jax():
    try:
        from jax.tree_util import register_pytree_node_class as _register_pytree_node_class  # type: ignore

        def register_pytree_node_class(cls):  # type: ignore
            """Register this type as a JAX pytree for JIT and automatic differentiation."""
            try:
                return _register_pytree_node_class(cls)
            except ValueError as e:
                if "Duplicate custom PyTreeDef type registration" in str(e):
                    return cls
                raise
    except Exception:  # pragma: no cover
        import jax.tree_util as _tu

        def register_pytree_node_class(cls):  # type: ignore
            """Register this type as a JAX pytree for JIT and automatic differentiation."""
            try:
                _tu.register_pytree_node(
                    cls,
                    lambda x: x.tree_flatten(),
                    lambda aux, children: cls.tree_unflatten(aux, children),
                )
            except ValueError as e:
                if "Duplicate custom PyTreeDef type registration" not in str(e):
                    raise
            return cls
else:

    def register_pytree_node_class(cls):  # type: ignore
        """Register this type as a JAX pytree for JIT and automatic differentiation."""
        return cls


@register_pytree_node_class
@dataclass(frozen=True)
class HelicalBasis:
    """Precomputed basis on a theta×zeta grid for a given mode table."""

    cos_phase: any  # (K, ntheta, nzeta)
    sin_phase: any  # (K, ntheta, nzeta)
    phase_stack: any | None  # (2K, ntheta, nzeta) or None
    m: any  # (K,)
    n: any  # (K,)
    nfp: int

    def tree_flatten(self):
        # Arrays are dynamic leaves; nfp is static auxiliary data.
        """Return JAX pytree leaves and static metadata for transformations."""
        children = (self.cos_phase, self.sin_phase, self.phase_stack, self.m, self.n)
        aux = (int(self.nfp),)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Rebuild the object from JAX pytree metadata and leaves."""
        (nfp,) = aux
        cos_phase, sin_phase, phase_stack, m, n = children
        return cls(cos_phase=cos_phase, sin_phase=sin_phase, phase_stack=phase_stack, m=m, n=n, nfp=int(nfp))


def build_helical_basis(modes: ModeTable, grid: AngleGrid, *, cache: bool = True) -> HelicalBasis:
    """Precompute cos/sin(phase) tensors.

    Notes:
    - This is O(K*ntheta*nzeta) memory; fine for VMEC defaults (low mpol, ntor) on a laptop.
    - Later we can switch to factored/FFT-based transforms once parity is validated.
    """
    if cache and _cache_allowed():
        key = _basis_cache_key(modes, grid)
        cached = _HELICAL_BASIS_CACHE.get(key)
        if cached is not None:
            return cached

    # Build static basis tables on the host.  Creating these with jnp at
    # top-level eagerly compiles many tiny GPU programs before the real solver
    # scan is even lowered.  NumPy leaves are promoted to device constants at
    # the enclosing JIT boundary, which is the intended staging point.
    m = np.asarray(modes.m)
    n = np.asarray(modes.n)
    theta = np.asarray(grid.theta)
    zeta = np.asarray(grid.zeta)

    # phase[K, ntheta, nzeta] via broadcasting
    phase = m[:, None, None] * theta[None, :, None] - n[:, None, None] * zeta[None, None, :]
    cos_phase = np.cos(phase)
    sin_phase = np.sin(phase)
    phase_stack = np.concatenate([cos_phase, sin_phase], axis=0)
    basis = HelicalBasis(
        cos_phase=cos_phase,
        sin_phase=sin_phase,
        phase_stack=phase_stack,
        m=m,
        n=n,
        nfp=grid.nfp,
    )
    if cache and _cache_allowed():
        _HELICAL_BASIS_CACHE[key] = basis
    return basis


def _internal_mode_scale(m, n, *, dtype):
    """Return mscale*nscale factors (VMEC internal -> physical)."""
    m = jnp.asarray(m)
    n = jnp.asarray(n)
    sqrt2 = jnp.sqrt(jnp.asarray(2.0, dtype=dtype))
    mscale = jnp.where(m == 0, jnp.asarray(1.0, dtype=dtype), sqrt2)
    nscale = jnp.where(jnp.abs(n) == 0, jnp.asarray(1.0, dtype=dtype), sqrt2)
    return mscale * nscale


@partial(jit, static_argnames=("coeffs_internal",))
def eval_fourier(
    coeff_cos,
    coeff_sin,
    basis: HelicalBasis,
    *,
    coeffs_internal: bool = False,
):
    """Evaluate f(theta,zeta) = Σ_k [c_k cos(phase_k) + s_k sin(phase_k)].

    coeff_cos, coeff_sin: shape (..., K)

    Returns: shape (..., ntheta, nzeta)
    """
    coeff_cos = jnp.asarray(coeff_cos)
    coeff_sin = jnp.asarray(coeff_sin)
    if coeffs_internal:
        scale = _internal_mode_scale(basis.m, basis.n, dtype=coeff_cos.dtype)
        coeff_cos = coeff_cos * scale
        coeff_sin = coeff_sin * scale

    # Einsum works in both numpy and jax.numpy
    phase_stack = getattr(basis, "phase_stack", None)
    if phase_stack is not None:
        coeff = jnp.concatenate([coeff_cos, coeff_sin], axis=-1)
        return einsum("...k,kij->...ij", coeff, phase_stack)
    return einsum("...k,kij->...ij", coeff_cos, basis.cos_phase) + einsum(
        "...k,kij->...ij", coeff_sin, basis.sin_phase
    )


@partial(jit, static_argnames=("coeffs_internal",))
def eval_fourier_dtheta(coeff_cos, coeff_sin, basis: HelicalBasis, *, coeffs_internal: bool = False):
    """∂/∂theta of the helical Fourier series."""
    coeff_cos = jnp.asarray(coeff_cos)
    coeff_sin = jnp.asarray(coeff_sin)
    if coeffs_internal:
        scale = _internal_mode_scale(basis.m, basis.n, dtype=coeff_cos.dtype)
        coeff_cos = coeff_cos * scale
        coeff_sin = coeff_sin * scale

    m = basis.m
    # d/dtheta cos = -m sin; d/dtheta sin = +m cos
    phase_stack = getattr(basis, "phase_stack", None)
    if phase_stack is not None:
        coeff = jnp.concatenate([coeff_sin * m, coeff_cos * (-m)], axis=-1)
        return einsum("...k,kij->...ij", coeff, phase_stack)
    return einsum("...k,kij->...ij", coeff_cos * (-m), basis.sin_phase) + einsum(
        "...k,kij->...ij", coeff_sin * m, basis.cos_phase
    )


@partial(jit, static_argnames=("coeffs_internal",))
def eval_fourier_dzeta_phys(coeff_cos, coeff_sin, basis: HelicalBasis, *, coeffs_internal: bool = False):
    """∂/∂phi_phys where phi_phys is the physical toroidal angle.

    VMEC uses zeta on one field period, so phi_phys = zeta / NFP and
    ∂/∂phi_phys = NFP * ∂/∂zeta.

    In the helical phase m*theta - n*zeta, ∂/∂zeta introduces factor (+n).
    Therefore ∂/∂phi_phys introduces factor (+n*NFP).
    """
    coeff_cos = jnp.asarray(coeff_cos)
    coeff_sin = jnp.asarray(coeff_sin)
    if coeffs_internal:
        scale = _internal_mode_scale(basis.m, basis.n, dtype=coeff_cos.dtype)
        coeff_cos = coeff_cos * scale
        coeff_sin = coeff_sin * scale

    n_phys = basis.n * basis.nfp
    # d/dzeta cos = +n sin; d/dzeta sin = -n cos  (because phase has -n*zeta)
    phase_stack = getattr(basis, "phase_stack", None)
    if phase_stack is not None:
        coeff = jnp.concatenate([coeff_sin * (-n_phys), coeff_cos * n_phys], axis=-1)
        return einsum("...k,kij->...ij", coeff, phase_stack)
    return einsum("...k,kij->...ij", coeff_cos * n_phys, basis.sin_phase) + einsum(
        "...k,kij->...ij", coeff_sin * (-n_phys), basis.cos_phase
    )


def project_to_modes(
    f: Array,
    basis: HelicalBasis,
    *,
    normalize: bool = True,
):
    """Project a real-space field `f(theta,zeta)` onto the helical Fourier basis.

    This is the inverse operation of `eval_fourier` *in the discrete sense* for a
    uniform tensor-product grid.

    Parameters
    ----------
    f:
        Real-space field on the angular grid. Shape (..., ntheta, nzeta).
    basis:
        Precomputed basis from `build_helical_basis`.
    normalize:
        If True, use uniform-grid orthogonality normalization (fast, no solves).
        If False, return raw inner products.

    Returns
    -------
    (coeff_cos, coeff_sin):
        Arrays of coefficients with shape (..., nmodes).

    Notes
    -----
    - This assumes the grid spans a full 2pi period in both theta and zeta.
    - For later parity-optimized grids (ntheta2/ntheta3) we'll provide a separate
      routine matching VMEC's normalization exactly.
    """
    # f: (..., i, j), basis: (k, i, j)
    inner_cos = einsum("...ij,kij->...k", f, basis.cos_phase)
    inner_sin = einsum("...ij,kij->...k", f, basis.sin_phase)

    if not normalize:
        return inner_cos, inner_sin

    ntheta = basis.cos_phase.shape[1]
    nzeta = basis.cos_phase.shape[2]
    norm = 2.0 / (ntheta * nzeta)

    coeff_cos = norm * inner_cos
    coeff_sin = norm * inner_sin

    # (m,n)=(0,0) mode should not get factor 2.
    # We detect it via (m==0 and n==0) in the basis metadata.
    if hasattr(basis, "m") and hasattr(basis, "n"):
        mask00 = jnp.logical_and(basis.m == 0, basis.n == 0)
        # Convert to float mask for broadcasting.
        mask00f = mask00.astype(coeff_cos.dtype)
        # Replace coefficients where mask00 = 1.
        coeff_cos = coeff_cos * (1.0 - mask00f) + (inner_cos / (ntheta * nzeta)) * mask00f
        coeff_sin = coeff_sin * (1.0 - mask00f) + 0.0 * mask00f

    return coeff_cos, coeff_sin
