"""VMEC profile evaluation.

This module implements the VMEC2000 profile logic used by the bundled examples:
power-series, two-power, and tabulated spline profiles for pressure, iota, and
toroidal current.  The spline boundary conditions follow STELLOPT/VMEC2000's
``spline_cubic.f`` convention: endpoint derivatives are fixed by a quadratic
fit to the first/last three knots.

We intentionally keep this code:
- dependency-light (NumPy for parsing, JAX-compatible math via ``vmec_jax._compat.jnp``)
- pure (no I/O, no global state)
- easy to extend later (additional VMEC profile families, pedestal variants, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from ._compat import jnp
from .namelist import InData

MU0 = 4e-7 * np.pi  # N/A^2


def _is_traced_array(x: Any) -> bool:
    """Return True if *x* is a JAX array (possibly traced inside jit)."""
    try:
        import jax
        return isinstance(x, jax.Array)
    except Exception:
        return False


def _is_jax_tracer(x: Any) -> bool:
    """Return True for values that cannot be materialized with NumPy."""

    try:
        import jax

        return isinstance(x, jax.core.Tracer)
    except Exception:
        return False


def _as_float_list(x: Any) -> Any:
    """Convert *x* to a list of floats, or return it unchanged if it is a JAX array.

    When called inside ``jax.jit`` with a traced array the value cannot be
    converted to a Python list.  In that case we return it as-is so that
    :func:`_coeff_array` can handle it through the JAX code-path.
    """
    if x is None:
        return []
    if _is_traced_array(x):
        return x           # keep as traced array; _coeff_array handles padding
    if hasattr(x, "shape") and hasattr(x, "dtype"):
        # numpy / concrete JAX array — try converting, fall back to as-is
        try:
            return [float(v) for v in np.asarray(x).ravel()]
        except Exception:
            return x
    if isinstance(x, list):
        return [float(v) for v in x]
    try:
        return [float(x)]
    except Exception:
        return x


def _coeff_array(
    coeffs,
    *,
    nmin: int = 21,
    dtype: Any = np.float64,
) -> Any:
    """Convert coefficients into a dense 1D JAX array of length ≥ ``nmin``.

    Handles three cases:

    * Python list / NumPy array  — converted via NumPy (static path).
    * Concrete JAX array          — same as NumPy path (values are known).
    * Traced JAX array (inside jit) — padding is done with JAX ops so that
      the shape is static but the values may be traced.
    """
    if _is_traced_array(coeffs):
        coeffs = jnp.asarray(coeffs)
        k = coeffs.shape[0]        # static (known at trace time)
        n = max(int(nmin), int(k))
        if k < n:
            coeffs = jnp.concatenate(
                [coeffs, jnp.zeros(n - k, dtype=coeffs.dtype)]
            )
        return coeffs
    coeffs_list = list(coeffs)
    n = max(int(nmin), len(coeffs_list))
    out = np.zeros((n,), dtype=dtype)
    if coeffs_list:
        out[: len(coeffs_list)] = np.asarray(coeffs_list, dtype=dtype)
    return jnp.asarray(out)


def _lower(s: Any, default: str) -> str:
    if s is None:
        return default
    if isinstance(s, list):
        s = s[0] if s else default
    s = str(s).strip().lower()
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        s = s[1:-1].strip().lower()
    return s


def _coeffs_static_or_jax(coeffs):
    """Prefer NumPy scalars for static coeff loops, but stay tracer-safe."""
    try:
        return np.asarray(coeffs)
    except Exception:
        return jnp.asarray(coeffs)


def _power_series(coeffs, x):
    """Evaluate Σ_i coeffs[i] * x**i using Horner (coeffs in ascending order)."""
    # Prefer NumPy scalars so coeffs[i] in the Python loop is a plain scalar
    # rather than a JAX dynamic_slice op, but fall back to JAX arrays if the
    # coefficients are traced.
    coeffs = _coeffs_static_or_jax(coeffs)
    x = jnp.asarray(x)
    y = jnp.zeros_like(x, dtype=coeffs.dtype)
    # Horner: iterate from high order to low order.
    for i in range(len(coeffs) - 1, -1, -1):
        y = y * x + coeffs[i]
    return y


def _pcurr_power_series_ip(ac, x):
    """VMEC `pcurr` default: parameterize I'(x) as a power series, return I(x).

    In VMEC2000 (`profile_functions.f`, case 'power_series'):
      I'(x) = Σ_i ac(i) * x**i
      I(x)  = ∫_0^x I'(t) dt = Σ_i ac(i)/(i+1) * x**(i+1)
    """
    # Same rationale as _power_series, with a tracer-safe fallback.
    ac = _coeffs_static_or_jax(ac)
    x = jnp.asarray(x)
    y = jnp.zeros_like(x, dtype=ac.dtype)
    for i in range(len(ac) - 1, -1, -1):
        y = y * x + ac[i] / (i + 1)
    return x * y


def _pcurr_power_series_i(ac, x):
    """VMEC ``power_series_i``: parameterize enclosed current I(x) directly.

    VMEC writes this branch as ``I(s) = sum_i ac[i] * s**(i + 1)``, so the
    enclosed current vanishes at the magnetic axis.
    """
    ac = _coeffs_static_or_jax(ac)
    x = jnp.asarray(x)
    y = jnp.zeros_like(x, dtype=ac.dtype)
    for i in range(len(ac) - 1, -1, -1):
        y = (y + ac[i]) * x
    return y


def _two_power(b, x):
    """VMEC `two_power` profile: b0 * (1 - x**b1)**b2."""
    b = jnp.asarray(b)
    x = jnp.asarray(x)
    b0 = b[0]
    b1 = b[1]
    b2 = b[2]
    core = jnp.maximum(1.0 - x**b1, 0.0)
    return b0 * core**b2


# Fixed-order Gauss-Legendre quadrature on [-1, 1], used to integrate I'(x) -> I(x).
_GL_N = 16
_GL_X_NP, _GL_W_NP = np.polynomial.legendre.leggauss(_GL_N)
_GL_X = jnp.asarray(_GL_X_NP)
_GL_W = jnp.asarray(_GL_W_NP)


def _pcurr_two_power_ip(ac, x):
    """VMEC `two_power` pcurr: parameterize I'(x) by `two_power`, return I(x)=∫ I'(t) dt."""
    ac = jnp.asarray(ac)
    x = jnp.asarray(x)
    # Map [-1, 1] -> [0, x]: t_i = 0.5 * x * (xi + 1)
    t = 0.5 * x[..., None] * (_GL_X[None, :] + 1.0)
    ip = _two_power(ac, t)
    return 0.5 * x * jnp.sum(_GL_W[None, :] * ip, axis=-1)


def _aux_profile_arrays(indata: InData, prefix: str) -> tuple[Any, Any]:
    """Return trimmed ``<prefix>_AUX_S/F`` arrays from an input namelist."""
    s_values = _as_float_list(indata.get(f"{prefix}_AUX_S", []))
    f_values = _as_float_list(indata.get(f"{prefix}_AUX_F", []))
    try:
        s_arr = np.asarray(s_values, dtype=np.float64).ravel()
        f_arr = np.asarray(f_values, dtype=np.float64).ravel()
    except Exception:
        return jnp.asarray(s_values), jnp.asarray(f_values)
    n = min(int(s_arr.size), int(f_arr.size))
    if n <= 0:
        return jnp.asarray([]), jnp.asarray([])
    s_arr = s_arr[:n]
    f_arr = f_arr[:n]
    n_valid = n
    for idx in range(1, n):
        if s_arr[idx] <= s_arr[idx - 1]:
            n_valid = idx
            break
    return jnp.asarray(s_arr[:n_valid]), jnp.asarray(f_arr[:n_valid])


def _vmec_cubic_endpoint_derivatives(x_knots, y_knots):
    """Endpoint slopes used by VMEC's ``spline_cubic`` routines.

    VMEC fixes the first derivative at each endpoint using a quadratic fit
    through the first/last three spline knots instead of using a natural
    spline.  For two knots this reduces to the secant slope.
    """
    x_knots = jnp.asarray(x_knots)
    y_knots = jnp.asarray(y_knots)
    n = int(x_knots.shape[0])
    if n <= 1:
        return jnp.asarray(0.0, dtype=y_knots.dtype), jnp.asarray(0.0, dtype=y_knots.dtype)
    if n == 2:
        slope = (y_knots[1] - y_knots[0]) / (x_knots[1] - x_knots[0])
        return slope, slope

    c_left = (
        (y_knots[2] - y_knots[0]) / (x_knots[2] - x_knots[0])
        - (y_knots[1] - y_knots[0]) / (x_knots[1] - x_knots[0])
    ) / (x_knots[2] - x_knots[1])
    yp1 = (y_knots[1] - y_knots[0]) / (x_knots[1] - x_knots[0]) - c_left * (
        x_knots[1] - x_knots[0]
    )

    c_right = (
        (y_knots[-3] - y_knots[-1]) / (x_knots[-3] - x_knots[-1])
        - (y_knots[-2] - y_knots[-1]) / (x_knots[-2] - x_knots[-1])
    ) / (x_knots[-3] - x_knots[-2])
    ypn = (y_knots[-2] - y_knots[-1]) / (x_knots[-2] - x_knots[-1]) - c_right * (
        x_knots[-2] - x_knots[-1]
    )
    return yp1, ypn


def _cubic_second_derivatives(x_knots, y_knots):
    """VMEC cubic-spline second derivatives at knots."""
    x_knots = jnp.asarray(x_knots)
    y_knots = jnp.asarray(y_knots)
    n = int(x_knots.shape[0])
    if n <= 2:
        return jnp.zeros_like(y_knots)
    h = x_knots[1:] - x_knots[:-1]
    yp1, ypn = _vmec_cubic_endpoint_derivatives(x_knots, y_knots)
    dtype = y_knots.dtype
    mat = jnp.zeros((n, n), dtype=dtype)
    rhs = jnp.zeros((n,), dtype=dtype)
    mat = mat.at[0, 0].set(2.0 * h[0])
    mat = mat.at[0, 1].set(h[0])
    rhs = rhs.at[0].set(6.0 * ((y_knots[1] - y_knots[0]) / h[0] - yp1))
    mat = mat.at[-1, -2].set(h[-1])
    mat = mat.at[-1, -1].set(2.0 * h[-1])
    rhs = rhs.at[-1].set(6.0 * (ypn - (y_knots[-1] - y_knots[-2]) / h[-1]))
    for idx in range(1, n - 1):
        hm = h[idx - 1]
        hp = h[idx]
        mat = mat.at[idx, idx - 1].set(hm)
        mat = mat.at[idx, idx].set(2.0 * (hm + hp))
        mat = mat.at[idx, idx + 1].set(hp)
        slope_p = (y_knots[idx + 1] - y_knots[idx]) / hp
        slope_m = (y_knots[idx] - y_knots[idx - 1]) / hm
        rhs = rhs.at[idx].set(6.0 * (slope_p - slope_m))
    return jnp.linalg.solve(mat, rhs)


def _cubic_interval_integral(y0, y1, m0, m1, h, dx):
    """Integral of one cubic-spline interval from the left knot to ``dx``."""
    a = y0 - m0 * h * h / 6.0
    b = y1 - m1 * h * h / 6.0
    return (
        m0 * (h**4 - (h - dx) ** 4) / (24.0 * h)
        + m1 * dx**4 / (24.0 * h)
        + a * (dx - dx**2 / (2.0 * h))
        + b * dx**2 / (2.0 * h)
    )


def _cubic_spline_profile(x_knots, y_knots, x, *, integrate: bool):
    """Evaluate or integrate a VMEC-style cubic spline through static knots."""
    x_knots = jnp.asarray(x_knots)
    y_knots = jnp.asarray(y_knots)
    x = jnp.asarray(x)
    n = int(x_knots.shape[0])
    if n == 0:
        return jnp.zeros_like(x)
    if n == 1:
        return y_knots[0] * x if integrate else jnp.broadcast_to(y_knots[0], x.shape)

    x_clipped = jnp.clip(x, x_knots[0], x_knots[-1])
    idx_hi = jnp.searchsorted(x_knots, x_clipped, side="right")
    idx_hi = jnp.clip(idx_hi, 1, n - 1)
    idx = idx_hi - 1
    h = x_knots[1:] - x_knots[:-1]
    dx = x_clipped - x_knots[idx]
    m = _cubic_second_derivatives(x_knots, y_knots)
    h_i = h[idx]
    y0 = y_knots[idx]
    y1 = y_knots[idx + 1]
    m0 = m[idx]
    m1 = m[idx + 1]
    if not integrate:
        left = m0 * (h_i - dx) ** 3 / (6.0 * h_i)
        right = m1 * dx**3 / (6.0 * h_i)
        linear_left = (y0 - m0 * h_i * h_i / 6.0) * (h_i - dx) / h_i
        linear_right = (y1 - m1 * h_i * h_i / 6.0) * dx / h_i
        return left + right + linear_left + linear_right

    full_interval = _cubic_interval_integral(
        y_knots[:-1],
        y_knots[1:],
        m[:-1],
        m[1:],
        h,
        h,
    )
    prefix = jnp.concatenate([jnp.zeros((1,), dtype=full_interval.dtype), jnp.cumsum(full_interval)])
    partial = _cubic_interval_integral(y0, y1, m0, m1, h_i, dx)
    return prefix[idx] + partial


def _line_segment_profile(x_knots, y_knots, x, *, integrate: bool):
    """Evaluate or integrate VMEC's line-segment profile through static knots."""
    x_knots = jnp.asarray(x_knots)
    y_knots = jnp.asarray(y_knots)
    x = jnp.asarray(x)
    n = int(x_knots.shape[0])
    if n == 0:
        return jnp.zeros_like(x)
    if n == 1:
        return y_knots[0] * x if integrate else jnp.broadcast_to(y_knots[0], x.shape)

    x_clipped = jnp.clip(x, x_knots[0], x_knots[-1])
    idx_hi = jnp.searchsorted(x_knots, x_clipped, side="right")
    idx_hi = jnp.clip(idx_hi, 1, n - 1)
    idx = idx_hi - 1
    h = x_knots[1:] - x_knots[:-1]
    dx = x_clipped - x_knots[idx]
    slopes = (y_knots[1:] - y_knots[:-1]) / h
    y = y_knots[idx] + slopes[idx] * dx
    if not integrate:
        return y
    full_interval = 0.5 * h * (y_knots[:-1] + y_knots[1:])
    prefix = jnp.concatenate([jnp.zeros((1,), dtype=full_interval.dtype), jnp.cumsum(full_interval)])
    partial = y_knots[idx] * dx + 0.5 * slopes[idx] * dx * dx
    return prefix[idx] + partial


def _spline_profile(profile_type: str, x_knots, y_knots, x, *, integrate: bool = False):
    """Evaluate a supported tabulated VMEC profile."""
    if profile_type == "cubic_spline":
        return _cubic_spline_profile(x_knots, y_knots, x, integrate=integrate)
    if profile_type == "line_segment":
        return _line_segment_profile(x_knots, y_knots, x, integrate=integrate)
    raise NotImplementedError(
        f"profile_type={profile_type!r} not implemented "
        "(supported tabulated profiles: cubic_spline, line_segment)"
    )


def _profile_coeffs_np(coeffs) -> np.ndarray:
    """Return profile coefficients as a concrete NumPy vector."""

    if coeffs is None:
        return np.asarray([], dtype=np.float64)
    try:
        return np.asarray(coeffs, dtype=np.float64).reshape(-1)
    except Exception:
        return np.asarray([], dtype=np.float64)


def _jnp_size_or_zero(values) -> int:
    if values is None:
        return 0
    return int(jnp.size(values))


def _power_series_np(coeffs, x: np.ndarray) -> np.ndarray:
    coeffs = _profile_coeffs_np(coeffs)
    y = np.zeros_like(x, dtype=np.result_type(x, coeffs, np.float64))
    for i in range(len(coeffs) - 1, -1, -1):
        y = y * x + coeffs[i]
    return y


def _pcurr_power_series_ip_np(coeffs, x: np.ndarray) -> np.ndarray:
    coeffs = _profile_coeffs_np(coeffs)
    y = np.zeros_like(x, dtype=np.result_type(x, coeffs, np.float64))
    for i in range(len(coeffs) - 1, -1, -1):
        y = y * x + coeffs[i] / float(i + 1)
    return x * y


def _two_power_np(coeffs, x: np.ndarray) -> np.ndarray:
    coeffs = _profile_coeffs_np(coeffs)
    if coeffs.size < 3:
        coeffs = np.pad(coeffs, (0, 3 - coeffs.size))
    b0, b1, b2 = coeffs[:3]
    core = np.maximum(1.0 - x**b1, 0.0)
    return b0 * core**b2


def _pcurr_two_power_ip_np(coeffs, x: np.ndarray) -> np.ndarray:
    t = 0.5 * x[..., None] * (_GL_X_NP[None, :] + 1.0)
    ip = _two_power_np(coeffs, t)
    return 0.5 * x * np.sum(_GL_W_NP[None, :] * ip, axis=-1)


def _vmec_cubic_endpoint_derivatives_np(x_knots: np.ndarray, y_knots: np.ndarray) -> tuple[float, float]:
    n = int(x_knots.shape[0])
    if n <= 1:
        return 0.0, 0.0
    if n == 2:
        slope = float((y_knots[1] - y_knots[0]) / (x_knots[1] - x_knots[0]))
        return slope, slope
    c_left = (
        (y_knots[2] - y_knots[0]) / (x_knots[2] - x_knots[0])
        - (y_knots[1] - y_knots[0]) / (x_knots[1] - x_knots[0])
    ) / (x_knots[2] - x_knots[1])
    yp1 = (y_knots[1] - y_knots[0]) / (x_knots[1] - x_knots[0]) - c_left * (
        x_knots[1] - x_knots[0]
    )
    c_right = (
        (y_knots[-3] - y_knots[-1]) / (x_knots[-3] - x_knots[-1])
        - (y_knots[-2] - y_knots[-1]) / (x_knots[-2] - x_knots[-1])
    ) / (x_knots[-3] - x_knots[-2])
    ypn = (y_knots[-2] - y_knots[-1]) / (x_knots[-2] - x_knots[-1]) - c_right * (
        x_knots[-2] - x_knots[-1]
    )
    return float(yp1), float(ypn)


def _cubic_second_derivatives_np(x_knots: np.ndarray, y_knots: np.ndarray) -> np.ndarray:
    n = int(x_knots.shape[0])
    if n <= 2:
        return np.zeros_like(y_knots)
    h = x_knots[1:] - x_knots[:-1]
    yp1, ypn = _vmec_cubic_endpoint_derivatives_np(x_knots, y_knots)
    mat = np.zeros((n, n), dtype=np.result_type(x_knots, y_knots, np.float64))
    rhs = np.zeros((n,), dtype=mat.dtype)
    mat[0, 0] = 2.0 * h[0]
    mat[0, 1] = h[0]
    rhs[0] = 6.0 * ((y_knots[1] - y_knots[0]) / h[0] - yp1)
    mat[-1, -2] = h[-1]
    mat[-1, -1] = 2.0 * h[-1]
    rhs[-1] = 6.0 * (ypn - (y_knots[-1] - y_knots[-2]) / h[-1])
    for idx in range(1, n - 1):
        hm = h[idx - 1]
        hp = h[idx]
        mat[idx, idx - 1] = hm
        mat[idx, idx] = 2.0 * (hm + hp)
        mat[idx, idx + 1] = hp
        slope_p = (y_knots[idx + 1] - y_knots[idx]) / hp
        slope_m = (y_knots[idx] - y_knots[idx - 1]) / hm
        rhs[idx] = 6.0 * (slope_p - slope_m)
    return np.linalg.solve(mat, rhs)


def _cubic_spline_profile_np(x_knots, y_knots, x: np.ndarray, *, integrate: bool) -> np.ndarray:
    x_knots = _profile_coeffs_np(x_knots)
    y_knots = _profile_coeffs_np(y_knots)
    n = int(x_knots.shape[0])
    if n == 0:
        return np.zeros_like(x)
    if n == 1:
        return y_knots[0] * x if integrate else np.broadcast_to(y_knots[0], x.shape)

    x_clipped = np.clip(x, x_knots[0], x_knots[-1])
    idx_hi = np.searchsorted(x_knots, x_clipped, side="right")
    idx_hi = np.clip(idx_hi, 1, n - 1)
    idx = idx_hi - 1
    h = x_knots[1:] - x_knots[:-1]
    dx = x_clipped - x_knots[idx]
    m = _cubic_second_derivatives_np(x_knots, y_knots)
    h_i = h[idx]
    y0 = y_knots[idx]
    y1 = y_knots[idx + 1]
    m0 = m[idx]
    m1 = m[idx + 1]
    if not integrate:
        left = m0 * (h_i - dx) ** 3 / (6.0 * h_i)
        right = m1 * dx**3 / (6.0 * h_i)
        linear_left = (y0 - m0 * h_i * h_i / 6.0) * (h_i - dx) / h_i
        linear_right = (y1 - m1 * h_i * h_i / 6.0) * dx / h_i
        return left + right + linear_left + linear_right

    full_interval = _cubic_interval_integral(
        y_knots[:-1],
        y_knots[1:],
        m[:-1],
        m[1:],
        h,
        h,
    )
    prefix = np.concatenate([np.zeros((1,), dtype=np.asarray(full_interval).dtype), np.cumsum(full_interval)])
    partial = _cubic_interval_integral(y0, y1, m0, m1, h_i, dx)
    return np.asarray(prefix[idx] + partial)


def _line_segment_profile_np(x_knots, y_knots, x: np.ndarray, *, integrate: bool) -> np.ndarray:
    x_knots = _profile_coeffs_np(x_knots)
    y_knots = _profile_coeffs_np(y_knots)
    n = int(x_knots.shape[0])
    if n == 0:
        return np.zeros_like(x)
    if n == 1:
        return y_knots[0] * x if integrate else np.broadcast_to(y_knots[0], x.shape)
    x_clipped = np.clip(x, x_knots[0], x_knots[-1])
    idx_hi = np.searchsorted(x_knots, x_clipped, side="right")
    idx_hi = np.clip(idx_hi, 1, n - 1)
    idx = idx_hi - 1
    h = x_knots[1:] - x_knots[:-1]
    dx = x_clipped - x_knots[idx]
    slopes = (y_knots[1:] - y_knots[:-1]) / h
    y = y_knots[idx] + slopes[idx] * dx
    if not integrate:
        return y
    full_interval = 0.5 * h * (y_knots[:-1] + y_knots[1:])
    prefix = np.concatenate([np.zeros((1,), dtype=full_interval.dtype), np.cumsum(full_interval)])
    partial = y_knots[idx] * dx + 0.5 * slopes[idx] * dx * dx
    return np.asarray(prefix[idx] + partial)


def _spline_profile_np(profile_type: str, x_knots, y_knots, x: np.ndarray, *, integrate: bool = False) -> np.ndarray:
    if profile_type == "cubic_spline":
        return _cubic_spline_profile_np(x_knots, y_knots, x, integrate=integrate)
    if profile_type == "line_segment":
        return _line_segment_profile_np(x_knots, y_knots, x, integrate=integrate)
    raise NotImplementedError(
        f"profile_type={profile_type!r} not implemented "
        "(supported tabulated profiles: cubic_spline, line_segment)"
    )


def _pcurr_power_series_i_np(coeffs, x: np.ndarray) -> np.ndarray:
    coeffs = _profile_coeffs_np(coeffs)
    y = np.zeros_like(x, dtype=np.result_type(x, coeffs, np.float64))
    for i in range(len(coeffs) - 1, -1, -1):
        y = (y + coeffs[i]) * x
    return y


def _can_use_numpy_profile_eval(s_grid: Any) -> bool:
    """Return True for concrete host profile grids outside JAX tracing."""

    if _is_jax_tracer(s_grid):
        return False
    try:
        arr = np.asarray(s_grid)
    except Exception:
        return False
    return arr.dtype != object


def _eval_profiles_numpy(cfg: ProfileInputs, s_grid) -> Dict[str, Any]:
    s = np.asarray(s_grid, dtype=np.float64)
    x = np.minimum(np.abs(s * float(cfg.bloat)), 1.0)
    out: Dict[str, Any] = {"ncurr": int(cfg.ncurr)}

    if cfg.pmass_type == "power_series":
        p_pa = float(cfg.pres_scale) * _power_series_np(cfg.am, x)
    elif cfg.pmass_type == "two_power":
        p_pa = float(cfg.pres_scale) * _two_power_np(cfg.am, x)
    elif cfg.pmass_type in ("cubic_spline", "line_segment"):
        if _profile_coeffs_np(cfg.am_aux_s).size == 0 or _profile_coeffs_np(cfg.am_aux_f).size == 0:
            p_pa = np.zeros_like(x)
        else:
            p_pa = float(cfg.pres_scale) * _spline_profile_np(
                cfg.pmass_type,
                cfg.am_aux_s,
                cfg.am_aux_f,
                x,
                integrate=False,
            )
    else:
        raise NotImplementedError(
            f"pmass_type={cfg.pmass_type!r} not implemented "
            "(supported: power_series, two_power, cubic_spline, line_segment)"
        )
    if float(cfg.spres_ped) < 1.0:
        x_ped = min(abs(float(cfg.spres_ped) * float(cfg.bloat)), 1.0)
        if cfg.pmass_type == "power_series":
            p_ped = float(cfg.pres_scale) * _power_series_np(cfg.am, np.asarray(x_ped))
        elif cfg.pmass_type == "two_power":
            p_ped = float(cfg.pres_scale) * _two_power_np(cfg.am, np.asarray(x_ped))
        else:
            p_ped = float(cfg.pres_scale) * _spline_profile_np(
                cfg.pmass_type,
                cfg.am_aux_s,
                cfg.am_aux_f,
                np.asarray(x_ped),
                integrate=False,
            )
        p_pa = np.where(s > float(cfg.spres_ped), p_ped, p_pa)
    out["pressure_pa"] = p_pa
    out["pressure"] = (MU0 * p_pa).astype(np.asarray(p_pa).dtype)

    ai = _profile_coeffs_np(cfg.ai)
    if ai.size > 0:
        if cfg.piota_type != "power_series":
            if cfg.piota_type in ("cubic_spline", "line_segment"):
                if _profile_coeffs_np(cfg.ai_aux_s).size == 0 or _profile_coeffs_np(cfg.ai_aux_f).size == 0:
                    iota = np.zeros_like(x)
                else:
                    iota = _spline_profile_np(
                        cfg.piota_type,
                        cfg.ai_aux_s,
                        cfg.ai_aux_f,
                        x,
                        integrate=False,
                    )
            else:
                raise NotImplementedError(
                    f"piota_type={cfg.piota_type!r} not implemented "
                    "(supported: power_series, cubic_spline, line_segment)"
                )
        else:
            iota = _power_series_np(ai, x)
        if cfg.lrfp:
            iota = np.divide(1.0, iota, out=np.full_like(iota, np.inf), where=iota != 0)
        out["iota"] = iota

    ac = _profile_coeffs_np(cfg.ac)
    if ac.size > 0:
        if cfg.pcurr_type == "power_series":
            out["current"] = _pcurr_power_series_ip_np(ac, x)
        elif cfg.pcurr_type in ("power_series_i", "power_series_I"):
            out["current"] = _pcurr_power_series_i_np(ac, x)
        elif cfg.pcurr_type == "two_power":
            out["current"] = _pcurr_two_power_ip_np(ac, x)
        elif cfg.pcurr_type.endswith("_ip") and cfg.pcurr_type.rsplit("_", 1)[0] in ("cubic_spline", "line_segment"):
            if _profile_coeffs_np(cfg.ac_aux_s).size == 0 or _profile_coeffs_np(cfg.ac_aux_f).size == 0:
                out["current"] = np.zeros_like(x)
            else:
                out["current"] = _spline_profile_np(
                    cfg.pcurr_type.rsplit("_", 1)[0],
                    cfg.ac_aux_s,
                    cfg.ac_aux_f,
                    x,
                    integrate=True,
                )
        elif cfg.pcurr_type.endswith("_i") and cfg.pcurr_type.rsplit("_", 1)[0] in ("cubic_spline", "line_segment"):
            if _profile_coeffs_np(cfg.ac_aux_s).size == 0 or _profile_coeffs_np(cfg.ac_aux_f).size == 0:
                out["current"] = np.zeros_like(x)
            else:
                out["current"] = _spline_profile_np(
                    cfg.pcurr_type.rsplit("_", 1)[0],
                    cfg.ac_aux_s,
                    cfg.ac_aux_f,
                    x,
                    integrate=False,
                )
        else:
            raise NotImplementedError(
                f"pcurr_type={cfg.pcurr_type!r} not implemented "
                "(supported: power_series, power_series_i, two_power, "
                "cubic_spline_i, cubic_spline_ip, line_segment_i, line_segment_ip)"
            )
    return out


@dataclass(frozen=True)
class ProfileInputs:
    """Profile-related inputs extracted from &INDATA."""

    pmass_type: str
    piota_type: str
    pcurr_type: str

    am: Any  # (n_am,)
    ai: Any  # (n_ai,)
    ac: Any  # (n_ac,)
    ac_aux_s: Any  # current-profile spline knots
    ac_aux_f: Any  # current-profile spline values

    pres_scale: float
    bloat: float
    spres_ped: float
    lrfp: bool
    ncurr: int
    am_aux_s: Any = None  # pressure-profile spline knots
    am_aux_f: Any = None  # pressure-profile spline values
    ai_aux_s: Any = None  # iota-profile spline knots
    ai_aux_f: Any = None  # iota-profile spline values


def profiles_from_indata(indata: InData) -> ProfileInputs:
    """Extract and normalize profile inputs from an :class:`~vmec_jax.namelist.InData`."""
    pmass_type = _lower(indata.get("PMASS_TYPE", "power_series"), "power_series")
    piota_type = _lower(indata.get("PIOTA_TYPE", "power_series"), "power_series")
    pcurr_type = _lower(indata.get("PCURR_TYPE", "power_series"), "power_series")

    am = _coeff_array(_as_float_list(indata.get("AM", [])))
    ai = _coeff_array(_as_float_list(indata.get("AI", [])))
    ac = _coeff_array(_as_float_list(indata.get("AC", [])))
    am_aux_s, am_aux_f = _aux_profile_arrays(indata, "AM")
    ai_aux_s, ai_aux_f = _aux_profile_arrays(indata, "AI")
    ac_aux_s, ac_aux_f = _aux_profile_arrays(indata, "AC")

    pres_scale = float(indata.get_float("PRES_SCALE", 1.0))
    bloat = float(indata.get_float("BLOAT", 1.0))
    spres_ped = float(abs(indata.get_float("SPRES_PED", 1.0)))
    lrfp = bool(indata.get_bool("LRFP", False))
    ncurr = int(indata.get_int("NCURR", 0))

    return ProfileInputs(
        pmass_type=pmass_type,
        piota_type=piota_type,
        pcurr_type=pcurr_type,
        am=am,
        ai=ai,
        ac=ac,
        ac_aux_s=ac_aux_s,
        ac_aux_f=ac_aux_f,
        pres_scale=pres_scale,
        bloat=bloat,
        spres_ped=spres_ped,
        lrfp=lrfp,
        ncurr=ncurr,
        am_aux_s=am_aux_s,
        am_aux_f=am_aux_f,
        ai_aux_s=ai_aux_s,
        ai_aux_f=ai_aux_f,
    )


def eval_profiles(cfg: ProfileInputs | InData, s_grid) -> Dict[str, Any]:
    """Evaluate VMEC profiles on a radial grid.

    Parameters
    ----------
    cfg:
        Either a :class:`ProfileInputs` or an :class:`~vmec_jax.namelist.InData`.
    s_grid:
        1D array of `s` values in [0, 1]. (In VMEC, s is normalized toroidal flux.)

    Returns
    -------
    dict
        Keys include:
        - ``pressure`` (VMEC internal units, ``mu0 * Pa``)
        - ``pressure_pa`` (Pa)
        - ``iota`` (dimensionless) if AI present
        - ``current`` (VMEC's I(s) function) if AC present
        - ``ncurr`` (0: iota-driven, 1: current-driven)

    Notes
    -----
    - Supported profile families are documented in :mod:`vmec_jax.profiles`
      and the user guide. Unsupported VMEC2000 profile families raise
      :class:`NotImplementedError` rather than silently falling back.
    - In VMEC, the input pressure coefficients ``AM`` and ``PRES_SCALE`` are in
      Pascals, but VMEC's internal pressure variable is in ``mu0 * Pa`` (i.e.
      the same units as ``B^2``). For solver/energy parity, we return:

      - ``pressure`` in ``mu0 * Pa`` (VMEC internal units),
      - ``pressure_pa`` in ``Pa`` (physical units).
    """
    if isinstance(cfg, InData):
        cfg = profiles_from_indata(cfg)
    if _can_use_numpy_profile_eval(s_grid):
        return _eval_profiles_numpy(cfg, s_grid)

    s = jnp.asarray(s_grid)
    x = jnp.minimum(jnp.abs(s * cfg.bloat), 1.0)

    out: Dict[str, Any] = {"ncurr": int(cfg.ncurr)}

    # --- Pressure (pmass) ---
    if cfg.pmass_type == "power_series":
        p_pa = cfg.pres_scale * _power_series(cfg.am, x)
    elif cfg.pmass_type == "two_power":
        p_pa = cfg.pres_scale * _two_power(cfg.am, x)
    elif cfg.pmass_type in ("cubic_spline", "line_segment"):
        if _jnp_size_or_zero(cfg.am_aux_s) == 0 or _jnp_size_or_zero(cfg.am_aux_f) == 0:
            p_pa = jnp.zeros_like(x)
        else:
            p_pa = cfg.pres_scale * _spline_profile(
                cfg.pmass_type,
                cfg.am_aux_s,
                cfg.am_aux_f,
                x,
                integrate=False,
            )
    else:
        raise NotImplementedError(
            f"pmass_type={cfg.pmass_type!r} not implemented "
            "(supported: power_series, two_power, cubic_spline, line_segment)"
        )
    if cfg.spres_ped < 1.0:
        x_ped = jnp.minimum(jnp.abs(jnp.asarray(cfg.spres_ped) * cfg.bloat), 1.0)
        if cfg.pmass_type == "power_series":
            p_ped = cfg.pres_scale * _power_series(cfg.am, x_ped)
        elif cfg.pmass_type == "two_power":
            p_ped = cfg.pres_scale * _two_power(cfg.am, x_ped)
        else:
            p_ped = cfg.pres_scale * _spline_profile(
                cfg.pmass_type,
                cfg.am_aux_s,
                cfg.am_aux_f,
                x_ped,
                integrate=False,
            )
        p_pa = jnp.where(s > cfg.spres_ped, p_ped, p_pa)
    out["pressure_pa"] = p_pa
    out["pressure"] = (MU0 * p_pa).astype(p_pa.dtype)

    # --- Iota / q (piota) ---
    if cfg.ai is not None and int(jnp.size(cfg.ai)) > 0:
        if cfg.piota_type == "power_series":
            iota = _power_series(cfg.ai, x)
        elif cfg.piota_type in ("cubic_spline", "line_segment"):
            if _jnp_size_or_zero(cfg.ai_aux_s) == 0 or _jnp_size_or_zero(cfg.ai_aux_f) == 0:
                iota = jnp.zeros_like(x)
            else:
                iota = _spline_profile(
                    cfg.piota_type,
                    cfg.ai_aux_s,
                    cfg.ai_aux_f,
                    x,
                    integrate=False,
                )
        else:
            raise NotImplementedError(
                f"piota_type={cfg.piota_type!r} not implemented "
                "(supported: power_series, cubic_spline, line_segment)"
            )
        if cfg.lrfp:
            iota = jnp.where(iota != 0, 1.0 / iota, jnp.asarray(np.inf, dtype=iota.dtype))
        out["iota"] = iota

    # --- Toroidal current function (pcurr) ---
    if cfg.ac is not None and int(jnp.size(cfg.ac)) > 0:
        if cfg.pcurr_type == "power_series":
            out["current"] = _pcurr_power_series_ip(cfg.ac, x)
        elif cfg.pcurr_type == "power_series_i":
            out["current"] = _pcurr_power_series_i(cfg.ac, x)
        elif cfg.pcurr_type == "two_power":
            out["current"] = _pcurr_two_power_ip(cfg.ac, x)
        elif cfg.pcurr_type.endswith("_ip") and cfg.pcurr_type.rsplit("_", 1)[0] in ("cubic_spline", "line_segment"):
            if _jnp_size_or_zero(cfg.ac_aux_s) == 0 or _jnp_size_or_zero(cfg.ac_aux_f) == 0:
                out["current"] = jnp.zeros_like(x)
            else:
                out["current"] = _spline_profile(
                    cfg.pcurr_type.rsplit("_", 1)[0],
                    cfg.ac_aux_s,
                    cfg.ac_aux_f,
                    x,
                    integrate=True,
                )
        elif cfg.pcurr_type.endswith("_i") and cfg.pcurr_type.rsplit("_", 1)[0] in ("cubic_spline", "line_segment"):
            if _jnp_size_or_zero(cfg.ac_aux_s) == 0 or _jnp_size_or_zero(cfg.ac_aux_f) == 0:
                out["current"] = jnp.zeros_like(x)
            else:
                out["current"] = _spline_profile(
                    cfg.pcurr_type.rsplit("_", 1)[0],
                    cfg.ac_aux_s,
                    cfg.ac_aux_f,
                    x,
                    integrate=False,
                )
        else:
            raise NotImplementedError(
                f"pcurr_type={cfg.pcurr_type!r} not implemented "
                "(supported: power_series, power_series_i, two_power, "
                "cubic_spline_i, cubic_spline_ip, line_segment_i, line_segment_ip)"
            )

    return out
