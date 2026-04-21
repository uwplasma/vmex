"""VMEC profile evaluation.

This module implements a small subset of VMEC2000's profile logic as found in
`profile_functions.f`, starting with the common `power_series` parameterization.

We intentionally keep this code:
- dependency-light (NumPy for parsing, JAX-compatible math via ``vmec_jax._compat.jnp``)
- pure (no I/O, no global state)
- easy to extend later (splines, pedestal, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

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
_GL_X, _GL_W = np.polynomial.legendre.leggauss(_GL_N)
_GL_X = jnp.asarray(_GL_X)
_GL_W = jnp.asarray(_GL_W)


def _pcurr_two_power_ip(ac, x):
    """VMEC `two_power` pcurr: parameterize I'(x) by `two_power`, return I(x)=∫ I'(t) dt."""
    ac = jnp.asarray(ac)
    x = jnp.asarray(x)
    # Map [-1, 1] -> [0, x]: t_i = 0.5 * x * (xi + 1)
    t = 0.5 * x[..., None] * (_GL_X[None, :] + 1.0)
    ip = _two_power(ac, t)
    return 0.5 * x * jnp.sum(_GL_W[None, :] * ip, axis=-1)


@dataclass(frozen=True)
class ProfileInputs:
    """Profile-related inputs extracted from &INDATA."""

    pmass_type: str
    piota_type: str
    pcurr_type: str

    am: Any  # (n_am,)
    ai: Any  # (n_ai,)
    ac: Any  # (n_ac,)

    pres_scale: float
    bloat: float
    spres_ped: float
    lrfp: bool
    ncurr: int


def profiles_from_indata(indata: InData) -> ProfileInputs:
    """Extract and normalize profile inputs from an :class:`~vmec_jax.namelist.InData`."""
    pmass_type = _lower(indata.get("PMASS_TYPE", "power_series"), "power_series")
    piota_type = _lower(indata.get("PIOTA_TYPE", "power_series"), "power_series")
    pcurr_type = _lower(indata.get("PCURR_TYPE", "power_series"), "power_series")

    am = _coeff_array(_as_float_list(indata.get("AM", [])))
    ai = _coeff_array(_as_float_list(indata.get("AI", [])))
    ac = _coeff_array(_as_float_list(indata.get("AC", [])))

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
        pres_scale=pres_scale,
        bloat=bloat,
        spres_ped=spres_ped,
        lrfp=lrfp,
        ncurr=ncurr,
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
        - ``pressure`` (Pa)
        - ``iota`` (dimensionless) if AI present
        - ``current`` (VMEC's I(s) function) if AC present
        - ``ncurr`` (0: iota-driven, 1: current-driven)

    Notes
    -----
    - For now we implement only the common ``power_series`` profile type.
    - In VMEC, the input pressure coefficients ``AM`` and ``PRES_SCALE`` are in
      Pascals, but VMEC's internal pressure variable is in ``mu0 * Pa`` (i.e.
      the same units as ``B^2``). For solver/energy parity, we return:

      - ``pressure`` in ``mu0 * Pa`` (VMEC internal units),
      - ``pressure_pa`` in ``Pa`` (physical units).
    """
    if isinstance(cfg, InData):
        cfg = profiles_from_indata(cfg)

    s = jnp.asarray(s_grid)
    x = jnp.minimum(jnp.abs(s * cfg.bloat), 1.0)

    out: Dict[str, Any] = {"ncurr": int(cfg.ncurr)}

    # --- Pressure (pmass) ---
    if cfg.pmass_type == "power_series":
        p_pa = cfg.pres_scale * _power_series(cfg.am, x)
    elif cfg.pmass_type == "two_power":
        p_pa = cfg.pres_scale * _two_power(cfg.am, x)
    else:
        raise NotImplementedError(
            f"pmass_type={cfg.pmass_type!r} not implemented (only 'power_series' and 'two_power')"
        )
    if cfg.spres_ped < 1.0:
        x_ped = jnp.minimum(jnp.abs(jnp.asarray(cfg.spres_ped) * cfg.bloat), 1.0)
        if cfg.pmass_type == "power_series":
            p_ped = cfg.pres_scale * _power_series(cfg.am, x_ped)
        else:
            p_ped = cfg.pres_scale * _two_power(cfg.am, x_ped)
        p_pa = jnp.where(s > cfg.spres_ped, p_ped, p_pa)
    out["pressure_pa"] = p_pa
    out["pressure"] = (MU0 * p_pa).astype(p_pa.dtype)

    # --- Iota / q (piota) ---
    if cfg.ai is not None and int(jnp.size(cfg.ai)) > 0:
        if cfg.piota_type != "power_series":
            raise NotImplementedError(f"piota_type={cfg.piota_type!r} not implemented (only 'power_series')")
        iota = _power_series(cfg.ai, x)
        if cfg.lrfp:
            iota = jnp.where(iota != 0, 1.0 / iota, jnp.asarray(np.inf, dtype=iota.dtype))
        out["iota"] = iota

    # --- Toroidal current function (pcurr) ---
    if cfg.ac is not None and int(jnp.size(cfg.ac)) > 0:
        if cfg.pcurr_type == "power_series":
            out["current"] = _pcurr_power_series_ip(cfg.ac, x)
        elif cfg.pcurr_type == "two_power":
            out["current"] = _pcurr_two_power_ip(cfg.ac, x)
        else:
            raise NotImplementedError(
                f"pcurr_type={cfg.pcurr_type!r} not implemented (only 'power_series' and 'two_power')"
            )

    return out
