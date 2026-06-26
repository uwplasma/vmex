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

from ._compat import jnp, has_jax
from .state import StateLayout, VMECState


_INTERP_CACHE: dict[tuple[int, int, str], tuple[Any, Any, Any]] = {}
_SCALXC_CACHE: dict[tuple[int, bytes, str], Any] = {}


def _contains_jax_tracer(value: Any) -> bool:
    """Return True when *value* contains a JAX tracer."""
    if not has_jax():
        return False
    try:
        from jax import core, tree_util

        return any(isinstance(leaf, core.Tracer) for leaf in tree_util.tree_leaves(value))
    except Exception:
        return False


def _should_use_numpy_interp(value: Any) -> bool:
    return bool(has_jax()) and not _contains_jax_tracer(value)


def _cache_allowed() -> bool:
    if not has_jax():
        return True
    try:
        from jax import core

        return bool(core.trace_ctx.is_top_level())
    except Exception:
        return False


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
    cache_key = None
    if _cache_allowed():
        try:
            m_bytes = np.asarray(m).tobytes()
            cache_key = (ns, m_bytes, str(np.dtype(dtype)))
            cached = _SCALXC_CACHE.get(cache_key)
            if cached is not None:
                return cached
        except Exception:
            cache_key = None
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
    out = jnp.where(is_odd[None, :] > 0, scal_odd[:, None], jnp.ones((ns, int(m.shape[0])), dtype=dtype))
    if cache_key is not None:
        _SCALXC_CACHE[cache_key] = out
    return out


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
    if _should_use_numpy_interp((x_old, m)):
        try:
            from .kernels.numpy_forces import _numpy_module_patch

            with _numpy_module_patch():
                return interp_vmec_radial_coeffs(x_old, m=m, ns_new=ns_new)
        except Exception:
            pass

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
    cache_key = None
    if _cache_allowed():
        try:
            cache_key = (int(ns_old), int(ns_new), str(np.dtype(dtype)))
            cached = _INTERP_CACHE.get(cache_key)
            if cached is not None:
                j1, j2, xint = cached
            else:
                j = jnp.arange(ns_new, dtype=jnp.int32)
                num = j.astype(jnp.int64) * int(ns_old - 1)
                den = int(ns_new - 1)
                j1 = (num // den).astype(jnp.int32)
                j2 = jnp.minimum(j1 + 1, int(ns_old - 1))
                xint = (j.astype(dtype) * float(ns_old - 1) / float(ns_new - 1)) - j1.astype(dtype)
                xint = jnp.clip(xint, 0.0, 1.0)
                _INTERP_CACHE[cache_key] = (j1, j2, xint)
        except Exception:
            cache_key = None
    if cache_key is None:
        j = jnp.arange(ns_new, dtype=jnp.int32)
        num = j.astype(jnp.int64) * int(ns_old - 1)
        den = int(ns_new - 1)
        j1 = (num // den).astype(jnp.int32)
        j2 = jnp.minimum(j1 + 1, int(ns_old - 1))
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
    n: Sequence[int] | np.ndarray | Any | None = None,
    lthreed: bool = True,
    lconm1: bool = True,
    ns_new: int,
) -> VMECState:
    """Interpolate a VMECState to a new radial resolution (ns_new)."""
    if _should_use_numpy_interp((state_old, m, n)):
        try:
            from .kernels.numpy_forces import _numpy_module_patch

            with _numpy_module_patch():
                return interp_vmec_state(
                    state_old,
                    m=m,
                    n=n,
                    lthreed=lthreed,
                    lconm1=lconm1,
                    ns_new=ns_new,
                )
        except Exception:
            pass

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

    n_arr_np = None
    if n is not None:
        n_arr_np = np.asarray(n, dtype=np.int32)
        if int(n_arr_np.shape[0]) != K:
            raise ValueError(f"n has shape {n_arr_np.shape}, expected (K,) with K={K}")

    layout = StateLayout(ns=ns_new, K=K, lasym=lasym)

    # For non-axisymmetric lasym=False stages, VMEC interpolates in internal
    # (m,n>=0) storage (interp.f) rather than directly on signed coefficients.
    # This avoids n-sign convention drift at stage boundaries.
    if (n_arr_np is not None) and (not lasym) and (int(np.max(np.abs(n_arr_np))) > 0):
        m_np = np.asarray(m_arr, dtype=np.int32)
        n_np = n_arr_np
        ns_old = int(state_old.layout.ns)
        from types import SimpleNamespace
        from .kernels.parity import (
            signed_maps_from_modes,
            _mn_cos_to_signed_cached as _mn_cos_to_signed_cached,
            _mn_sin_to_signed_cached as _mn_sin_to_signed_cached,
            _signed_to_mn_cos_cached as _signed_to_mn_cos_cached,
            _signed_to_mn_sin_cached as _signed_to_mn_sin_cached,
        )
        signed_maps = signed_maps_from_modes(SimpleNamespace(m=m_np, n=n_np))
        mpol = int(signed_maps.mpol)
        nrange = int(signed_maps.nrange)
        basis_norm = jnp.ones((mpol, nrange), dtype=jnp.asarray(state_old.Rcos).dtype)

        def _signed_to_mn_cos(coeffs: Any):
            return _signed_to_mn_cos_cached(jnp.asarray(coeffs), maps=signed_maps)

        def _signed_to_mn_sin(coeffs: Any):
            return _signed_to_mn_sin_cached(jnp.asarray(coeffs), maps=signed_maps)

        def _mn_cos_to_signed(rcc, rss):
            return _mn_cos_to_signed_cached(jnp.asarray(rcc), jnp.asarray(rss), maps=signed_maps, ncoeff=K)

        def _mn_sin_to_signed(zsc, zcs):
            return _mn_sin_to_signed_cached(jnp.asarray(zsc), jnp.asarray(zcs), maps=signed_maps, ncoeff=K)

        m_flat = jnp.repeat(jnp.arange(mpol, dtype=jnp.int32), nrange)

        def _interp_block(block):
            interp = interp_vmec_radial_coeffs(jnp.asarray(block).reshape(ns_old, -1), m=m_flat, ns_new=ns_new)
            return interp.reshape(ns_new, mpol, nrange)

        rcc, rss = _signed_to_mn_cos(state_old.Rcos)
        zsc, zcs = _signed_to_mn_sin(state_old.Zsin)
        lsc, lcs = _signed_to_mn_sin(state_old.Lsin)

        if bool(lthreed) and bool(lconm1) and mpol > 1:
            rss_m1 = rss[:, 1, :]
            zcs_m1 = zcs[:, 1, :]
            rss = rss.at[:, 1, :].set(0.5 * (rss_m1 + zcs_m1))
            zcs = zcs.at[:, 1, :].set(0.5 * (rss_m1 - zcs_m1))

        rcc = rcc * basis_norm[None, :, :]
        rss = rss * basis_norm[None, :, :]
        zsc = zsc * basis_norm[None, :, :]
        zcs = zcs * basis_norm[None, :, :]
        lsc = lsc * basis_norm[None, :, :]
        lcs = lcs * basis_norm[None, :, :]

        rcc_i = _interp_block(rcc)
        rss_i = _interp_block(rss)
        zsc_i = _interp_block(zsc)
        zcs_i = _interp_block(zcs)
        lsc_i = _interp_block(lsc)
        lcs_i = _interp_block(lcs)

        if bool(lthreed) and bool(lconm1) and mpol > 1:
            rss_m1 = rss_i[:, 1, :]
            zcs_m1 = zcs_i[:, 1, :]
            rss_i = rss_i.at[:, 1, :].set(rss_m1 + zcs_m1)
            zcs_i = zcs_i.at[:, 1, :].set(rss_m1 - zcs_m1)

        Rcos_new = _mn_cos_to_signed(rcc_i, rss_i)
        Zsin_new = _mn_sin_to_signed(zsc_i, zcs_i)
        Lsin_new = _mn_sin_to_signed(lsc_i, lcs_i)

        return VMECState(
            layout=layout,
            Rcos=jnp.asarray(Rcos_new),
            Rsin=interp_vmec_radial_coeffs(state_old.Rsin, m=m_arr, ns_new=ns_new),
            Zcos=interp_vmec_radial_coeffs(state_old.Zcos, m=m_arr, ns_new=ns_new),
            Zsin=jnp.asarray(Zsin_new),
            Lcos=interp_vmec_radial_coeffs(state_old.Lcos, m=m_arr, ns_new=ns_new),
            Lsin=jnp.asarray(Lsin_new),
        )

    return VMECState(
        layout=layout,
        Rcos=interp_vmec_radial_coeffs(state_old.Rcos, m=m_arr, ns_new=ns_new),
        Rsin=interp_vmec_radial_coeffs(state_old.Rsin, m=m_arr, ns_new=ns_new),
        Zcos=interp_vmec_radial_coeffs(state_old.Zcos, m=m_arr, ns_new=ns_new),
        Zsin=interp_vmec_radial_coeffs(state_old.Zsin, m=m_arr, ns_new=ns_new),
        Lcos=interp_vmec_radial_coeffs(state_old.Lcos, m=m_arr, ns_new=ns_new),
        Lsin=interp_vmec_radial_coeffs(state_old.Lsin, m=m_arr, ns_new=ns_new),
    )
