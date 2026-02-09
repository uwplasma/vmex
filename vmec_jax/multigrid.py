"""VMEC multigrid staging helpers (fixed-boundary parity work).

VMEC2000 typically uses multigrid in the radial direction via `NS_ARRAY`,
solving on a coarse `ns` first and then interpolating the Fourier coefficients
onto the next grid. The interpolation has a VMEC-specific convention:

- interpolate the **scaled** coefficients `x_old * scalxc_old`, where `scalxc`
  converts odd-m harmonics into VMEC's internal 1/sqrt(s) representation,
- then divide by `scalxc_new` to return to physical coefficients on the new grid,
- extrapolate odd-m values to the axis on the *scaled* array before interpolating,
  and zero odd-m coefficients on the axis on output.

This module ports the core of `STELLOPT/VMEC2000/Sources/TimeStep/interp.f`.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ._compat import jnp
from .state import StateLayout, VMECState


def _scalxc_vmec(*, ns: int, m: Any, dtype) -> Any:
    """VMEC `scalxc(js,m)` factors for each stored mode.

    Parameters
    ----------
    ns:
        Number of radial surfaces.
    m:
        Poloidal mode numbers per coefficient, shape (K,).
    dtype:
        Output dtype.
    """
    ns = int(ns)
    m = jnp.asarray(m)
    if ns <= 0:
        return jnp.zeros((0, int(m.shape[0])), dtype=dtype)

    s = jnp.linspace(0.0, 1.0, ns, dtype=dtype)
    sqrts = jnp.sqrt(jnp.maximum(s, 0.0))
    # VMEC sets sqrts(ns)=1 explicitly.
    if ns >= 1:
        sqrts = jnp.where(jnp.arange(ns, dtype=jnp.int32) == (ns - 1), jnp.asarray(1.0, dtype=dtype), sqrts)
    sq2 = sqrts[1] if ns >= 2 else jnp.asarray(1.0, dtype=dtype)
    scal_odd = 1.0 / jnp.maximum(sqrts, sq2)

    is_odd = ((m.astype(jnp.int32) % 2) == 1).astype(dtype)
    return jnp.where(is_odd[None, :] > 0, scal_odd[:, None], jnp.ones((ns, int(m.shape[0])), dtype=dtype))


def interp_vmec_radial_coeffs(
    x_old: Any,
    *,
    m: Any,
    ns_new: int,
) -> Any:
    """Interpolate a (ns_old, K) coefficient array onto a new VMEC radial grid.

    This reproduces VMEC2000's `interp.f` convention described in the module
    docstring.
    """
    x_old = jnp.asarray(x_old)
    ns_old, K = int(x_old.shape[0]), int(x_old.shape[1])
    ns_new = int(ns_new)
    if ns_old <= 0 or ns_new <= 0:
        return jnp.zeros((max(ns_new, 0), K), dtype=x_old.dtype)

    m = jnp.asarray(m)
    if int(m.shape[0]) != K:
        raise ValueError(f"m has shape {m.shape}, expected (K,) with K={K}")

    # Degenerate grids: fall back to a direct copy/truncate.
    if ns_old == ns_new:
        return x_old
    if ns_new == 1:
        return x_old[:1]
    if ns_old == 1:
        return jnp.broadcast_to(x_old[:1], (ns_new, K))

    dtype = x_old.dtype
    scal_old = _scalxc_vmec(ns=ns_old, m=m, dtype=dtype)
    scal_new = _scalxc_vmec(ns=ns_new, m=m, dtype=dtype)

    # Work in scaled (internal odd-m) representation.
    x_scaled = x_old * scal_old

    # Extrapolate odd-m modes over sqrt(s) to the axis on the scaled array:
    #   x(1) = 2*x(2) - x(3)   (Fortran, 1-based)
    if ns_old >= 3:
        is_odd = ((m.astype(jnp.int32) % 2) == 1).astype(dtype)
        axis_extrap = 2.0 * x_scaled[1] - x_scaled[2]
        axis_row = jnp.where(is_odd > 0, axis_extrap, x_scaled[0])
        x_scaled = jnp.concatenate([axis_row[None, :], x_scaled[1:]], axis=0)

    # Uniform-grid interpolation matching interp.f's js1/js2/xint construction.
    j = jnp.arange(ns_new, dtype=jnp.int32)
    num = j.astype(jnp.int64) * int(ns_old - 1)
    den = int(ns_new - 1)
    j1 = (num // den).astype(jnp.int32)
    j2 = jnp.minimum(j1 + 1, int(ns_old - 1))

    # xint = (sj - s1)/hsold with sj=j/(ns_new-1), s1=j1/(ns_old-1), hsold=1/(ns_old-1)
    # => xint = j*(ns_old-1)/(ns_new-1) - j1
    xint = (j.astype(dtype) * float(ns_old - 1) / float(ns_new - 1)) - j1.astype(dtype)
    xint = jnp.clip(xint, 0.0, 1.0)

    x1 = x_scaled[j1]
    x2 = x_scaled[j2]
    x_new_scaled = (1.0 - xint)[:, None] * x1 + xint[:, None] * x2

    # Unscale by scalxc on the new grid to return physical coefficients.
    x_new = x_new_scaled / scal_new

    # Zero odd-m modes on the axis (physical coefficients).
    is_odd = ((m.astype(jnp.int32) % 2) == 1).astype(dtype)
    axis_row = jnp.where(is_odd > 0, jnp.asarray(0.0, dtype=dtype), x_new[0])
    x_new = jnp.concatenate([axis_row[None, :], x_new[1:]], axis=0)
    return x_new


def interp_vmec_state(
    state_old: VMECState,
    *,
    m: Sequence[int] | np.ndarray | Any,
    ns_new: int,
) -> VMECState:
    """Interpolate a VMECState to a new radial resolution (ns_new)."""
    ns_new = int(ns_new)
    ns_old = int(state_old.layout.ns)
    K = int(state_old.layout.K)
    lasym = bool(state_old.layout.lasym)
    if ns_new <= 0:
        layout = StateLayout(ns=0, K=K, lasym=lasym)
        z = jnp.zeros((0, K), dtype=jnp.asarray(state_old.Rcos).dtype)
        return VMECState(layout=layout, Rcos=z, Rsin=z, Zcos=z, Zsin=z, Lcos=z, Lsin=z)

    m_arr = jnp.asarray(np.asarray(m, dtype=np.int32))
    if int(m_arr.shape[0]) != K:
        raise ValueError(f"m has shape {m_arr.shape}, expected (K,) with K={K}")

    layout = StateLayout(ns=ns_new, K=K, lasym=lasym)
    return VMECState(
        layout=layout,
        Rcos=interp_vmec_radial_coeffs(state_old.Rcos, m=m_arr, ns_new=ns_new),
        Rsin=interp_vmec_radial_coeffs(state_old.Rsin, m=m_arr, ns_new=ns_new),
        Zcos=interp_vmec_radial_coeffs(state_old.Zcos, m=m_arr, ns_new=ns_new),
        Zsin=interp_vmec_radial_coeffs(state_old.Zsin, m=m_arr, ns_new=ns_new),
        Lcos=interp_vmec_radial_coeffs(state_old.Lcos, m=m_arr, ns_new=ns_new),
        Lsin=interp_vmec_radial_coeffs(state_old.Lsin, m=m_arr, ns_new=ns_new),
    )

