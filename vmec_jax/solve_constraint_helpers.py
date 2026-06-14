"""Fixed-boundary, axis-regularity, and lambda-gauge helper functions."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._compat import jnp
from .state import VMECState


def mode00_index(modes) -> Optional[int]:
    """Return the first Fourier-mode index for ``(m,n)=(0,0)``, if present."""

    m = np.asarray(modes.m)
    n = np.asarray(modes.n)
    idx = np.where((m == 0) & (n == 0))[0]
    if idx.size == 0:
        return None
    return int(idx[0])


def enforce_lambda_gauge(Lcos, Lsin, *, idx00: Optional[int]):
    """Fix the ``(m,n)=(0,0)`` lambda gauge mode to zero."""

    if idx00 is None:
        return Lcos, Lsin
    Lcos = zero_coeff_column(Lcos, idx=int(idx00))
    Lsin = zero_coeff_column(Lsin, idx=int(idx00))
    return Lcos, Lsin


def apply_vmec_lambda_axis_rules_to_state(
    st: VMECState,
    *,
    enforce_vmec_lambda_axis: bool,
    host_update_assembly: bool,
    idx00: Optional[int],
) -> VMECState:
    """Enforce the VMEC lambda gauge while preserving stored axis coefficients."""

    if not bool(enforce_vmec_lambda_axis):
        return st
    if bool(host_update_assembly):
        Lcos = np.array(np.asarray(st.Lcos))
        Lsin = np.array(np.asarray(st.Lsin))
        if idx00 is not None:
            ncols = Lcos.shape[1]
            if 0 <= int(idx00) < ncols:
                Lcos[:, int(idx00)] = 0.0
                Lsin[:, int(idx00)] = 0.0
        return VMECState(
            layout=st.layout,
            Rcos=st.Rcos,
            Rsin=st.Rsin,
            Zcos=st.Zcos,
            Zsin=st.Zsin,
            Lcos=Lcos,
            Lsin=Lsin,
        )
    Lcos = jnp.asarray(st.Lcos)
    Lsin = jnp.asarray(st.Lsin)
    Lcos, Lsin = enforce_lambda_gauge(Lcos, Lsin, idx00=idx00)
    return VMECState(
        layout=st.layout,
        Rcos=st.Rcos,
        Rsin=st.Rsin,
        Zcos=st.Zcos,
        Zsin=st.Zsin,
        Lcos=Lcos,
        Lsin=Lsin,
    )


def axis_m0_mask(static, *, dtype):
    """Return a mask that keeps only m=0 Fourier coefficients on the axis."""

    if getattr(static, "m_is_m0", None) is not None:
        return jnp.asarray(static.m_is_m0, dtype=dtype)
    m = jnp.asarray(static.modes.m)
    return (m == 0).astype(dtype)


def zero_coeff_column(arr, *, idx: int):
    """Zero one Fourier coefficient column with concatenation instead of masking."""

    arr = jnp.asarray(arr)
    ncols = int(arr.shape[1])
    if idx < 0 or idx >= ncols:
        return arr
    zero = jnp.zeros_like(arr[:, :1])
    if idx == 0:
        if ncols == 1:
            return zero
        return jnp.concatenate([zero, arr[:, 1:]], axis=1)
    if idx == ncols - 1:
        return jnp.concatenate([arr[:, :idx], zero], axis=1)
    return jnp.concatenate([arr[:, :idx], zero, arr[:, idx + 1 :]], axis=1)


def replace_mode_slice(arr, *, mode_idx: int, replacement):
    """Replace one ``(m, :)`` slice of a ``(ns, mpol, nrange)`` array."""

    if arr is None:
        return None
    arr = jnp.asarray(arr)
    nmodes = int(arr.shape[1])
    if mode_idx < 0 or mode_idx >= nmodes:
        return arr
    repl = jnp.asarray(replacement, dtype=arr.dtype)[:, None, :]
    if mode_idx == 0:
        if nmodes == 1:
            return repl
        return jnp.concatenate([repl, arr[:, 1:, :]], axis=1)
    if mode_idx == nmodes - 1:
        return jnp.concatenate([arr[:, :mode_idx, :], repl], axis=1)
    return jnp.concatenate([arr[:, :mode_idx, :], repl, arr[:, mode_idx + 1 :, :]], axis=1)


def scale_mode_slice(arr, *, mode_idx: int, scale):
    """Scale one ``(m, :)`` slice of a ``(ns, mpol, nrange)`` array."""

    if arr is None:
        return None
    arr = jnp.asarray(arr)
    nmodes = int(arr.shape[1])
    if mode_idx < 0 or mode_idx >= nmodes:
        return arr
    scaled = arr[:, mode_idx, :] * jnp.asarray(scale, dtype=arr.dtype)[:, None]
    return replace_mode_slice(arr, mode_idx=mode_idx, replacement=scaled)


def zero_coeff_column_np(arr, *, idx: int) -> np.ndarray:
    """NumPy in-place version of :func:`zero_coeff_column`."""

    arr = np.array(np.asarray(arr))
    ncols = int(arr.shape[1])
    if 0 <= idx < ncols:
        arr[:, idx] = 0.0
    return arr


def replace_mode_slice_np(arr, *, mode_idx: int, replacement):
    """NumPy in-place version of :func:`replace_mode_slice`."""

    if arr is None:
        return None
    arr = np.array(np.asarray(arr))
    nmodes = int(arr.shape[1])
    if 0 <= mode_idx < nmodes:
        arr[:, mode_idx, :] = np.asarray(replacement)
    return arr


def scale_mode_slice_np(arr, *, mode_idx: int, scale):
    """NumPy in-place version of :func:`scale_mode_slice`."""

    if arr is None:
        return None
    arr = np.array(np.asarray(arr))
    nmodes = int(arr.shape[1])
    if 0 <= mode_idx < nmodes:
        arr[:, mode_idx, :] *= np.asarray(scale)[:, None]
    return arr


def enforce_field_rows(arr, *, axis_mask=None, edge_row=None, zero_axis: bool = False):
    """Apply axis/edge row constraints with at most one concatenation."""

    arr = jnp.asarray(arr)
    ns = int(arr.shape[0])
    if ns == 0:
        return arr

    first = arr[:1, :]
    if zero_axis:
        first = jnp.zeros_like(first)
    elif axis_mask is not None:
        first = first * jnp.asarray(axis_mask, dtype=arr.dtype)[None, :]

    last = arr[-1:, :]
    if edge_row is not None:
        last = jnp.asarray(edge_row, dtype=arr.dtype)[None, :]

    if ns == 1:
        row = last
        if zero_axis:
            row = jnp.zeros_like(row)
        elif axis_mask is not None:
            row = row * jnp.asarray(axis_mask, dtype=arr.dtype)[None, :]
        return row

    if (zero_axis or axis_mask is not None) and (edge_row is not None):
        return jnp.concatenate([first, arr[1:-1, :], last], axis=0)
    if zero_axis or axis_mask is not None:
        return jnp.concatenate([first, arr[1:, :]], axis=0)
    if edge_row is not None:
        return jnp.concatenate([arr[:-1, :], last], axis=0)
    return arr


def enforce_fixed_boundary_and_axis(
    state: VMECState,
    static,
    *,
    edge_Rcos,
    edge_Rsin,
    edge_Zcos,
    edge_Zsin,
    enforce_axis: bool = True,
    enforce_edge: bool = True,
    enforce_lambda_axis: bool = True,
    idx00: Optional[int],
) -> VMECState:
    """Apply VMEC axis regularity, fixed-boundary, and lambda-gauge constraints."""

    Rcos = jnp.asarray(state.Rcos)
    Rsin = jnp.asarray(state.Rsin)
    Zcos = jnp.asarray(state.Zcos)
    Zsin = jnp.asarray(state.Zsin)
    Lcos = jnp.asarray(state.Lcos)
    Lsin = jnp.asarray(state.Lsin)

    mask_m0 = axis_m0_mask(static, dtype=Rcos.dtype) if enforce_axis else None
    edge_Rcos_arr = edge_Rcos if enforce_edge else None
    edge_Rsin_arr = edge_Rsin if enforce_edge else None
    edge_Zcos_arr = edge_Zcos if enforce_edge else None
    edge_Zsin_arr = edge_Zsin if enforce_edge else None

    Rcos = enforce_field_rows(Rcos, axis_mask=mask_m0, edge_row=edge_Rcos_arr)
    Rsin = enforce_field_rows(Rsin, axis_mask=mask_m0, edge_row=edge_Rsin_arr)
    Zcos = enforce_field_rows(Zcos, axis_mask=mask_m0, edge_row=edge_Zcos_arr)
    Zsin = enforce_field_rows(Zsin, axis_mask=mask_m0, edge_row=edge_Zsin_arr)
    Lcos = enforce_field_rows(Lcos, zero_axis=bool(enforce_lambda_axis))
    Lsin = enforce_field_rows(Lsin, zero_axis=bool(enforce_lambda_axis))

    Lcos, Lsin = enforce_lambda_gauge(Lcos, Lsin, idx00=idx00)

    return VMECState(
        layout=state.layout,
        Rcos=Rcos,
        Rsin=Rsin,
        Zcos=Zcos,
        Zsin=Zsin,
        Lcos=Lcos,
        Lsin=Lsin,
    )


def enforce_field_rows_np(arr, *, axis_mask=None, edge_row=None, zero_axis: bool = False):
    """NumPy in-place version of :func:`enforce_field_rows`."""

    arr = np.array(arr)  # writable copy
    ns = arr.shape[0]
    if ns == 0:
        return arr
    if ns == 1:
        if edge_row is not None:
            arr[0] = np.asarray(edge_row)
        if zero_axis:
            arr[0] = 0.0
        elif axis_mask is not None:
            arr[0] *= np.asarray(axis_mask)
        return arr
    if edge_row is not None:
        arr[-1] = np.asarray(edge_row)
    if zero_axis:
        arr[0] = 0.0
    elif axis_mask is not None:
        arr[0] *= np.asarray(axis_mask)
    return arr


def enforce_fixed_boundary_and_axis_np(
    state: VMECState,
    static,
    *,
    edge_Rcos,
    edge_Rsin,
    edge_Zcos,
    edge_Zsin,
    enforce_axis: bool = True,
    enforce_edge: bool = True,
    enforce_lambda_axis: bool = True,
    idx00: Optional[int],
    precomputed_axis_mask: Optional[np.ndarray] = None,
) -> VMECState:
    """NumPy version of :func:`enforce_fixed_boundary_and_axis`."""

    Rcos = np.array(state.Rcos)
    Rsin = np.array(state.Rsin)
    Zcos = np.array(state.Zcos)
    Zsin = np.array(state.Zsin)
    Lcos = np.array(state.Lcos)
    Lsin = np.array(state.Lsin)

    if precomputed_axis_mask is not None:
        mask_m0 = precomputed_axis_mask if enforce_axis else None
    else:
        mask_m0 = np.asarray(axis_m0_mask(static, dtype=Rcos.dtype)) if enforce_axis else None
    edge_Rcos_np = np.asarray(edge_Rcos) if enforce_edge else None
    edge_Rsin_np = np.asarray(edge_Rsin) if enforce_edge else None
    edge_Zcos_np = np.asarray(edge_Zcos) if enforce_edge else None
    edge_Zsin_np = np.asarray(edge_Zsin) if enforce_edge else None

    Rcos = enforce_field_rows_np(Rcos, axis_mask=mask_m0, edge_row=edge_Rcos_np)
    Rsin = enforce_field_rows_np(Rsin, axis_mask=mask_m0, edge_row=edge_Rsin_np)
    Zcos = enforce_field_rows_np(Zcos, axis_mask=mask_m0, edge_row=edge_Zcos_np)
    Zsin = enforce_field_rows_np(Zsin, axis_mask=mask_m0, edge_row=edge_Zsin_np)
    Lcos = enforce_field_rows_np(Lcos, zero_axis=bool(enforce_lambda_axis))
    Lsin = enforce_field_rows_np(Lsin, zero_axis=bool(enforce_lambda_axis))

    if idx00 is not None and 0 <= int(idx00) < Lcos.shape[1]:
        Lcos[:, int(idx00)] = 0.0
        Lsin[:, int(idx00)] = 0.0

    return VMECState(
        layout=state.layout,
        Rcos=Rcos,
        Rsin=Rsin,
        Zcos=Zcos,
        Zsin=Zsin,
        Lcos=Lcos,
        Lsin=Lsin,
    )


def grad_rms_state(grad: VMECState) -> float:
    """Return RMS norm over all coefficient arrays in a ``VMECState``."""

    g = np.asarray(grad.Rcos) ** 2
    g = g + np.asarray(grad.Rsin) ** 2
    g = g + np.asarray(grad.Zcos) ** 2
    g = g + np.asarray(grad.Zsin) ** 2
    g = g + np.asarray(grad.Lcos) ** 2
    g = g + np.asarray(grad.Lsin) ** 2
    return float(np.sqrt(np.mean(g)))
