"""Bootstrap-current profile updates for finite-beta VMEC inputs.

The Redl bootstrap-current formula returns a flux-surface averaged parallel
current, ``<J.B>``.  VMEC inputs instead take a toroidal-current profile
through ``CURTOR`` and ``PCURR_TYPE``/``AC_AUX_*``.  This module implements the
pure profile-conversion layer needed for a deterministic VMEC/Redl fixed-point
iteration:

``VMEC solve -> Redl <J.B> -> I(s) or I'(s) -> VMEC current profile``.

The helpers here do not run VMEC.  They are small, differentiable building
blocks that can be tested against manufactured profiles before being used by a
solve-in-the-loop driver.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ._compat import jnp
from .integrals import cumtrapz_s
from .namelist import InData

MU0 = 4e-7 * np.pi
CurrentUpdatePolicy = Literal["low_beta", "lagged_pressure", "integrating_factor"]


@dataclass(frozen=True)
class BootstrapCurrentOptions:
    """Configuration for Redl-to-VMEC current-profile fixed-point updates."""

    helicity_n: int
    surfaces: tuple[float, ...] | None = None
    n_current: int = 50
    policy: CurrentUpdatePolicy = "integrating_factor"
    damping: float = 0.5
    anderson_depth: int = 0
    mismatch_tol: float = 1.0e-3
    current_tol: float = 1.0e-3
    max_fixed_point_iter: int = 8
    pcurr_type: str = "cubic_spline_ip"


@dataclass(frozen=True)
class BootstrapCurrentIteration:
    """Serializable per-iteration diagnostics for bootstrap fixed points."""

    iteration: int
    mismatch_norm: float
    current_update_norm: float
    curtor: float
    ac_aux_s: tuple[float, ...]
    ac_aux_f: tuple[float, ...]
    beta_total: float | None = None
    aspect: float | None = None
    mean_iota: float | None = None
    fsq_total: float | None = None


def _as_1d(name: str, value: Any):
    arr = jnp.asarray(value, dtype=jnp.float64)
    if int(arr.ndim) != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}")
    return arr


def _validate_profile_shapes(s, *arrays):
    n = int(s.shape[0])
    if n < 2:
        raise ValueError("s must contain at least two radial points")
    for name, arr in arrays:
        if int(arr.shape[0]) != n:
            raise ValueError(f"{name} length {arr.shape[0]} does not match s length {n}")


def dpsi_ds_from_vmec_phiedge(phiedge: Any, *, signgs: int) -> Any:
    r"""Return the flux derivative convention used in the bootstrap update.

    Landreman's VMEC-current note gives

    .. math::

       d\psi/ds = \mathrm{signgs}\,\Phi_\mathrm{edge}/(2\pi).

    ``PHIEDGE`` in VMEC input is the toroidal flux at the LCFS, not divided by
    ``2*pi``.
    """

    return float(1 if int(signgs) >= 0 else -1) * jnp.asarray(phiedge, dtype=jnp.float64) / (2.0 * np.pi)


def redl_current_rhs(*, jdotB_redl: Any, bdotb: Any, dpsi_ds: Any) -> Any:
    r"""Return the Redl source term for the VMEC current ODE.

    .. math::

       R(s) = 2\pi \psi'(s)\,\langle J.B\rangle_\mathrm{Redl}/\langle B^2\rangle.
    """

    jdotB_redl = _as_1d("jdotB_redl", jdotB_redl)
    bdotb = _as_1d("bdotb", bdotb)
    _validate_profile_shapes(jdotB_redl, ("bdotb", bdotb))
    bdotb_safe = jnp.where(jnp.abs(bdotb) > 0.0, bdotb, jnp.asarray(1.0, dtype=jnp.float64))
    return 2.0 * np.pi * jnp.asarray(dpsi_ds, dtype=jnp.float64) * jdotB_redl / bdotb_safe


def redl_current_derivative_update(
    *,
    s: Any,
    jdotB_redl: Any,
    bdotb: Any,
    dpsi_ds: Any,
    dpds: Any | None = None,
    previous_current: Any | None = None,
    policy: Literal["low_beta", "lagged_pressure"] = "low_beta",
) -> Any:
    r"""Return a VMEC current-derivative update ``I'(s)``.

    ``low_beta`` applies the common approximation in which the
    pressure-gradient correction is neglected.  ``lagged_pressure`` evaluates
    the pressure-gradient term using the previous equilibrium/current profile:

    .. math::

       I'_{k+1} = R_k - \mu_0 I_k p'_k/\langle B^2\rangle_k.
    """

    s = _as_1d("s", s)
    jdotB_redl = _as_1d("jdotB_redl", jdotB_redl)
    bdotb = _as_1d("bdotb", bdotb)
    _validate_profile_shapes(s, ("jdotB_redl", jdotB_redl), ("bdotb", bdotb))
    rhs = redl_current_rhs(jdotB_redl=jdotB_redl, bdotb=bdotb, dpsi_ds=dpsi_ds)
    if policy == "low_beta":
        return rhs
    if policy != "lagged_pressure":
        raise ValueError(f"unsupported derivative update policy {policy!r}")
    if dpds is None or previous_current is None:
        raise ValueError("lagged_pressure update requires dpds and previous_current")
    dpds = _as_1d("dpds", dpds)
    previous_current = _as_1d("previous_current", previous_current)
    _validate_profile_shapes(s, ("dpds", dpds), ("previous_current", previous_current))
    bdotb_safe = jnp.where(jnp.abs(bdotb) > 0.0, bdotb, jnp.asarray(1.0, dtype=jnp.float64))
    return rhs - MU0 * previous_current * dpds / bdotb_safe


def redl_current_integrating_factor_update(
    *,
    s: Any,
    jdotB_redl: Any,
    bdotb: Any,
    dpsi_ds: Any,
    dpds: Any,
) -> dict[str, Any]:
    r"""Solve the Redl/VMEC current ODE with an integrating factor.

    The equation is

    .. math::

       I'(s) + a(s) I(s) = R(s),\qquad
       a(s)=\mu_0 p'(s)/\langle B^2\rangle.

    With ``I(0)=0``,

    .. math::

       I(s) = \exp[-A(s)] \int_0^s \exp[A(t)] R(t)\,dt,
       \quad A(s)=\int_0^s a(t)\,dt.

    The derivative returned is computed from the ODE, ``I' = R - a I``.
    """

    s = _as_1d("s", s)
    jdotB_redl = _as_1d("jdotB_redl", jdotB_redl)
    bdotb = _as_1d("bdotb", bdotb)
    dpds = _as_1d("dpds", dpds)
    _validate_profile_shapes(s, ("jdotB_redl", jdotB_redl), ("bdotb", bdotb), ("dpds", dpds))
    bdotb_safe = jnp.where(jnp.abs(bdotb) > 0.0, bdotb, jnp.asarray(1.0, dtype=jnp.float64))
    rhs = redl_current_rhs(jdotB_redl=jdotB_redl, bdotb=bdotb_safe, dpsi_ds=dpsi_ds)
    a = MU0 * dpds / bdotb_safe
    A = cumtrapz_s(a, s)
    expA = jnp.exp(jnp.clip(A, -100.0, 100.0))
    current = jnp.exp(jnp.clip(-A, -100.0, 100.0)) * cumtrapz_s(expA * rhs, s)
    current_derivative = rhs - a * current
    return {
        "s": s,
        "current": current,
        "current_derivative": current_derivative,
        "rhs": rhs,
        "a": a,
        "integrating_factor_exponent": A,
    }


def integrate_current_derivative(s: Any, current_derivative: Any) -> Any:
    """Integrate a current-derivative profile with ``I(0)=0``."""

    s = _as_1d("s", s)
    current_derivative = _as_1d("current_derivative", current_derivative)
    _validate_profile_shapes(s, ("current_derivative", current_derivative))
    return cumtrapz_s(current_derivative, s)


def damp_current_profile(old: Any, new: Any, damping: float) -> Any:
    """Return a damped current-profile update."""

    if not (0.0 <= float(damping) <= 1.0):
        raise ValueError("damping must be in [0, 1]")
    old = _as_1d("old", old)
    new = _as_1d("new", new)
    _validate_profile_shapes(old, ("new", new))
    alpha = jnp.asarray(float(damping), dtype=jnp.float64)
    return (1.0 - alpha) * old + alpha * new


def vmec_current_profile_from_bootstrap_update(
    *,
    s: Any,
    current_derivative: Any | None = None,
    current: Any | None = None,
    signgs: int,
    pcurr_type: str = "cubic_spline_ip",
) -> dict[str, Any]:
    """Return VMEC current-profile arrays and ``CURTOR`` from bootstrap data."""

    s = _as_1d("s", s)
    pcurr_type_l = str(pcurr_type).strip().lower()
    if pcurr_type_l not in {"cubic_spline_ip", "cubic_spline_i"}:
        raise ValueError("only cubic_spline_ip and cubic_spline_i are supported")
    if pcurr_type_l == "cubic_spline_ip":
        if current_derivative is None:
            raise ValueError("current_derivative is required for cubic_spline_ip")
        values = _as_1d("current_derivative", current_derivative)
        _validate_profile_shapes(s, ("current_derivative", values))
        integrated_current = integrate_current_derivative(s, values) if current is None else _as_1d("current", current)
    else:
        if current is None:
            raise ValueError("current is required for cubic_spline_i")
        values = _as_1d("current", current)
        _validate_profile_shapes(s, ("current", values))
        integrated_current = values

    _validate_profile_shapes(s, ("integrated_current", integrated_current))
    edge_current = integrated_current[-1]
    curtor = float(1 if int(signgs) >= 0 else -1) * edge_current
    return {
        "pcurr_type": pcurr_type_l,
        "ac_aux_s": s,
        "ac_aux_f": values,
        "current": integrated_current,
        "curtor": curtor,
    }


def apply_current_profile_to_indata(
    indata: InData,
    *,
    ac_aux_s: Any,
    ac_aux_f: Any,
    curtor: Any,
    pcurr_type: str = "cubic_spline_ip",
) -> InData:
    """Return a copy of ``indata`` with a VMEC current profile applied."""

    s = np.asarray(ac_aux_s, dtype=np.float64).reshape(-1)
    f = np.asarray(ac_aux_f, dtype=np.float64).reshape(-1)
    if s.size != f.size:
        raise ValueError(f"AC_AUX_S/F size mismatch: {s.size} != {f.size}")
    if s.size < 2:
        raise ValueError("current profile requires at least two knots")
    if np.any(np.diff(s) <= 0.0):
        raise ValueError("AC_AUX_S must be strictly increasing")
    out = deepcopy(indata)
    out.scalars["NCURR"] = 1
    out.scalars["CURTOR"] = float(np.asarray(curtor, dtype=np.float64))
    out.scalars["PCURR_TYPE"] = str(pcurr_type).strip().lower()
    # Keep AC non-empty so profile evaluation enters the current-profile branch.
    out.scalars["AC"] = [1.0]
    out.scalars["AC_AUX_S"] = [float(x) for x in s]
    out.scalars["AC_AUX_F"] = [float(x) for x in f]
    return out


def bootstrap_current_update_to_indata(
    indata: InData,
    *,
    s: Any,
    current_derivative: Any | None = None,
    current: Any | None = None,
    signgs: int,
    pcurr_type: str = "cubic_spline_ip",
) -> tuple[InData, dict[str, Any]]:
    """Convert a bootstrap-current update and apply it to an ``InData`` copy."""

    profile = vmec_current_profile_from_bootstrap_update(
        s=s,
        current_derivative=current_derivative,
        current=current,
        signgs=signgs,
        pcurr_type=pcurr_type,
    )
    out = apply_current_profile_to_indata(
        indata,
        ac_aux_s=profile["ac_aux_s"],
        ac_aux_f=profile["ac_aux_f"],
        curtor=profile["curtor"],
        pcurr_type=profile["pcurr_type"],
    )
    return out, profile


__all__ = [
    "BootstrapCurrentIteration",
    "BootstrapCurrentOptions",
    "CurrentUpdatePolicy",
    "apply_current_profile_to_indata",
    "bootstrap_current_update_to_indata",
    "damp_current_profile",
    "dpsi_ds_from_vmec_phiedge",
    "integrate_current_derivative",
    "redl_current_derivative_update",
    "redl_current_integrating_factor_update",
    "redl_current_rhs",
    "vmec_current_profile_from_bootstrap_update",
]
