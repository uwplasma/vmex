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
import tempfile
from typing import Any, Literal

import numpy as np

from ._compat import jnp
from .integrals import cumtrapz_s
from .namelist import InData
from .profiles import ELEMENTARY_CHARGE
from .redl_bootstrap import polynomial_profile_and_derivative

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
    max_current_update_norm: float | None = None
    return_best_evaluated_on_max_iter: bool = False
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
    effective_damping: float | None = None
    current_update_limited: bool = False
    unlimited_current_update_norm: float | None = None
    max_current_update_norm: float | None = None


@dataclass(frozen=True)
class BootstrapCurrentResult:
    """Result returned by :func:`bootstrap_current_fixed_point`."""

    indata: InData
    history: tuple[BootstrapCurrentIteration, ...]
    converged: bool
    reason: str
    last_run: Any | None = None
    last_diagnostics: dict[str, Any] | None = None
    returned_best_evaluated: bool = False
    best_evaluated_iteration: int | None = None
    best_evaluated_mismatch_norm: float | None = None


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


def _pressure_derivative_pa_from_profile_coeffs(*, s, ne_coeffs, Te_coeffs, Ti_coeffs=None, Zeff_coeffs=1.0):
    """Return ``d(e*(ne*Te + ni*Ti))/ds`` in Pascals for Redl profiles."""

    s = _as_1d("s", s)
    Ti_coeffs = Te_coeffs if Ti_coeffs is None else Ti_coeffs
    ne_s, dne_ds = polynomial_profile_and_derivative(ne_coeffs, s)
    Te_s, dTe_ds = polynomial_profile_and_derivative(Te_coeffs, s)
    Ti_s, dTi_ds = polynomial_profile_and_derivative(Ti_coeffs, s)
    Zeff_s, dZeff_ds = polynomial_profile_and_derivative(jnp.atleast_1d(jnp.asarray(Zeff_coeffs)), s)
    Zeff_s = jnp.maximum(Zeff_s, jnp.asarray(1.0, dtype=jnp.float64))
    ni_s = ne_s / Zeff_s
    dni_ds = (dne_ds * Zeff_s - ne_s * dZeff_ds) / (Zeff_s * Zeff_s)
    return ELEMENTARY_CHARGE * (dne_ds * Te_s + ne_s * dTe_ds + dni_ds * Ti_s + ni_s * dTi_ds)


def _current_derivative_from_indata(indata: InData, s: Any, pcurr_type: str) -> Any:
    """Return the current derivative represented by a VMEC input when available."""

    s = _as_1d("s", s)
    pcurr_type_l = str(pcurr_type).strip().lower()
    if pcurr_type_l == "cubic_spline_ip":
        knots = jnp.asarray(indata.scalars.get("AC_AUX_S", []), dtype=jnp.float64)
        values = jnp.asarray(indata.scalars.get("AC_AUX_F", []), dtype=jnp.float64)
        if int(knots.size) >= 2 and int(values.size) == int(knots.size):
            return jnp.interp(s, knots, values)
    if pcurr_type_l == "power_series":
        coeffs = jnp.asarray(indata.scalars.get("AC", []), dtype=jnp.float64).reshape(-1)
        if int(coeffs.size) > 0:
            out = jnp.zeros_like(s, dtype=jnp.float64)
            for coeff in coeffs[::-1]:
                out = out * s + coeff
            return out
    return jnp.zeros_like(s, dtype=jnp.float64)


def _current_grid_update_samples(*, options: BootstrapCurrentOptions, s: Any, jdotB_redl: Any, bdotb: Any, dpds: Any):
    """Return Redl update channels on a current grid that spans ``[0, 1]``."""

    s = _as_1d("s", s)
    jdotB_redl = _as_1d("jdotB_redl", jdotB_redl)
    bdotb = _as_1d("bdotb", bdotb)
    dpds = _as_1d("dpds", dpds)
    _validate_profile_shapes(s, ("jdotB_redl", jdotB_redl), ("bdotb", bdotb), ("dpds", dpds))
    if int(options.n_current) < 2:
        raise ValueError("n_current must be at least 2")

    # Redl geometry should avoid exactly s=0 and s=1 for robustness, while
    # VMEC's current spline should still cover the full normalized-flux domain.
    left_needed = bool(float(np.asarray(s[0])) > 0.0)
    right_needed = bool(float(np.asarray(s[-1])) < 1.0)
    if left_needed:
        s = jnp.concatenate([jnp.asarray([0.0], dtype=s.dtype), s])
        jdotB_redl = jnp.concatenate([jdotB_redl[:1], jdotB_redl])
        bdotb = jnp.concatenate([bdotb[:1], bdotb])
        dpds = jnp.concatenate([dpds[:1], dpds])
    if right_needed:
        s = jnp.concatenate([s, jnp.asarray([1.0], dtype=s.dtype)])
        jdotB_redl = jnp.concatenate([jdotB_redl, jdotB_redl[-1:]])
        bdotb = jnp.concatenate([bdotb, bdotb[-1:]])
        dpds = jnp.concatenate([dpds, dpds[-1:]])

    if int(options.n_current) == int(s.shape[0]):
        return {"s": s, "jdotB_redl": jdotB_redl, "bdotb": bdotb, "dpds": dpds}
    grid = jnp.linspace(0.0, 1.0, int(options.n_current), dtype=s.dtype)
    return {
        "s": grid,
        "jdotB_redl": jnp.interp(grid, s, jdotB_redl),
        "bdotb": jnp.interp(grid, s, bdotb),
        "dpds": jnp.interp(grid, s, dpds),
    }


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


def _default_bootstrap_solve_fn(indata: InData, *, run_kwargs: dict[str, Any] | None = None):
    """Run VMEC from an in-memory input by writing a temporary input deck."""

    from .driver import run_fixed_boundary
    from .namelist import write_indata

    kwargs = {} if run_kwargs is None else dict(run_kwargs)
    kwargs.setdefault("verbose", False)
    with tempfile.TemporaryDirectory(prefix="vmec_jax_bootstrap_current_") as tmp:
        path = f"{tmp}/input.bootstrap_current"
        write_indata(path, indata)
        return run_fixed_boundary(path, **kwargs)


def _default_redl_diagnostics_fn(
    run,
    indata: InData,
    *,
    options: BootstrapCurrentOptions,
    ne_coeffs,
    Te_coeffs,
    Ti_coeffs=None,
    Zeff_coeffs=1.0,
):
    """Return Redl/VMEC diagnostics from a completed vmec_jax run."""

    from .finite_beta import (
        redl_bootstrap_mismatch_from_state,
    )
    from .wout import equilibrium_aspect_ratio_from_state, equilibrium_iota_profiles_from_state

    redl = redl_bootstrap_mismatch_from_state(
        state=run.state,
        static=run.static,
        indata=run.indata,
        signgs=int(run.signgs),
        helicity_n=int(options.helicity_n),
        ne_coeffs=ne_coeffs,
        Te_coeffs=Te_coeffs,
        Ti_coeffs=Ti_coeffs,
        Zeff_coeffs=Zeff_coeffs,
        surfaces=options.surfaces,
    )
    geom = redl["geometry"]
    s = jnp.asarray(geom["s"], dtype=jnp.float64)
    out = {
        "s": s,
        "jdotB_redl": redl["jdotB_redl"],
        "bdotb": geom["fsa_B2"],
        "dpds": _pressure_derivative_pa_from_profile_coeffs(
            s=s,
            ne_coeffs=ne_coeffs,
            Te_coeffs=Te_coeffs,
            Ti_coeffs=Ti_coeffs,
            Zeff_coeffs=Zeff_coeffs,
        ),
        "dpsi_ds": dpsi_ds_from_vmec_phiedge(run.indata.get_float("PHIEDGE", 1.0), signgs=int(run.signgs)),
        "signgs": int(run.signgs),
        "mismatch_norm": jnp.linalg.norm(jnp.asarray(redl["residuals1d"], dtype=jnp.float64))
        / jnp.sqrt(jnp.maximum(jnp.asarray(jnp.size(redl["residuals1d"]), dtype=jnp.float64), 1.0)),
    }
    try:
        out["aspect"] = float(equilibrium_aspect_ratio_from_state(state=run.state, static=run.static))
    except Exception:
        pass
    try:
        _chips, iotas, _iotaf = equilibrium_iota_profiles_from_state(
            state=run.state,
            static=run.static,
            indata=run.indata,
            signgs=int(run.signgs),
        )
        out["mean_iota"] = float(np.nanmean(np.asarray(iotas, dtype=float)))
    except Exception:
        pass
    diag = getattr(getattr(run, "result", None), "diagnostics", {}) or {}
    fsqr = diag.get("final_fsqr", diag.get("fsqr"))
    fsqz = diag.get("final_fsqz", diag.get("fsqz"))
    fsql = diag.get("final_fsql", diag.get("fsql"))
    if fsqr is not None and fsqz is not None and fsql is not None:
        try:
            out["fsq_total"] = float(fsqr) + float(fsqz) + float(fsql)
        except Exception:
            pass
    return out


def bootstrap_current_fixed_point(
    indata: InData,
    *,
    options: BootstrapCurrentOptions,
    solve_fn=None,
    diagnostics_fn=None,
    ne_coeffs=None,
    Te_coeffs=None,
    Ti_coeffs=None,
    Zeff_coeffs=1.0,
    run_kwargs: dict[str, Any] | None = None,
) -> BootstrapCurrentResult:
    """Run a VMEC/Redl fixed-point iteration for a self-consistent current.

    The loop is deliberately callback-friendly: production use can rely on the
    default vmec_jax solve and Redl diagnostic callbacks, while tests and
    workflow studies can inject cheap deterministic callbacks.  The optimized
    quantity is the VMEC current profile only; plasma boundary coefficients are
    not touched by this helper.
    """

    if int(options.max_fixed_point_iter) < 1:
        raise ValueError("max_fixed_point_iter must be at least 1")
    if options.max_current_update_norm is not None and float(options.max_current_update_norm) <= 0.0:
        raise ValueError("max_current_update_norm must be positive when provided")
    if int(options.anderson_depth) != 0:
        raise NotImplementedError("Anderson acceleration is planned but not implemented yet")
    if solve_fn is None:
        solve_fn = lambda current_indata: _default_bootstrap_solve_fn(current_indata, run_kwargs=run_kwargs)
    if diagnostics_fn is None:
        if ne_coeffs is None or Te_coeffs is None:
            raise ValueError("ne_coeffs and Te_coeffs are required when diagnostics_fn is not provided")

        def diagnostics_fn(run, current_indata):
            return _default_redl_diagnostics_fn(
                run,
                current_indata,
                options=options,
                ne_coeffs=ne_coeffs,
                Te_coeffs=Te_coeffs,
                Ti_coeffs=Ti_coeffs,
                Zeff_coeffs=Zeff_coeffs,
            )

    current_indata = deepcopy(indata)
    history: list[BootstrapCurrentIteration] = []
    last_run = None
    last_diag: dict[str, Any] | None = None
    best_evaluated_indata = deepcopy(current_indata)
    best_evaluated_iteration: int | None = None
    best_evaluated_mismatch_norm = float("inf")
    converged = False
    reason = "max_fixed_point_iter"

    for iteration in range(1, int(options.max_fixed_point_iter) + 1):
        last_run = solve_fn(current_indata)
        diag = diagnostics_fn(last_run, current_indata)
        last_diag = dict(diag)
        signgs = int(diag.get("signgs", getattr(last_run, "signgs", 1)))
        dpsi_ds = diag.get("dpsi_ds", dpsi_ds_from_vmec_phiedge(current_indata.get_float("PHIEDGE", 1.0), signgs=signgs))
        update_samples = _current_grid_update_samples(
            options=options,
            s=diag["s"],
            jdotB_redl=diag["jdotB_redl"],
            bdotb=diag["bdotb"],
            dpds=diag["dpds"],
        )
        s = update_samples["s"]
        if options.policy == "integrating_factor":
            update = redl_current_integrating_factor_update(
                s=s,
                jdotB_redl=update_samples["jdotB_redl"],
                bdotb=update_samples["bdotb"],
                dpsi_ds=dpsi_ds,
                dpds=update_samples["dpds"],
            )
            proposed_derivative = update["current_derivative"]
        else:
            previous_current = diag.get("previous_current")
            if previous_current is None:
                previous_current = integrate_current_derivative(
                    s,
                    _current_derivative_from_indata(current_indata, s, options.pcurr_type),
                )
            proposed_derivative = redl_current_derivative_update(
                s=s,
                jdotB_redl=update_samples["jdotB_redl"],
                bdotb=update_samples["bdotb"],
                dpsi_ds=dpsi_ds,
                dpds=update_samples["dpds"],
                previous_current=previous_current,
                policy=options.policy,
            )

        old_derivative = _current_derivative_from_indata(current_indata, s, options.pcurr_type)
        effective_damping = float(options.damping)
        new_derivative = damp_current_profile(old_derivative, proposed_derivative, effective_damping)
        update_denominator = (
            jnp.linalg.norm(old_derivative) + jnp.linalg.norm(proposed_derivative) + jnp.asarray(1.0e-300)
        )
        update_norm = jnp.linalg.norm(new_derivative - old_derivative) / update_denominator
        unlimited_update_norm = float(np.asarray(update_norm, dtype=np.float64))
        current_update_limited = False
        if options.max_current_update_norm is not None:
            max_update = float(options.max_current_update_norm)
            if unlimited_update_norm > max_update:
                effective_damping *= max_update / unlimited_update_norm
                new_derivative = damp_current_profile(old_derivative, proposed_derivative, effective_damping)
                update_norm = jnp.linalg.norm(new_derivative - old_derivative) / update_denominator
                current_update_limited = True
        new_current = integrate_current_derivative(s, new_derivative)
        next_indata, profile = bootstrap_current_update_to_indata(
            current_indata,
            s=s,
            current_derivative=new_derivative,
            current=new_current,
            signgs=signgs,
            pcurr_type=options.pcurr_type,
        )
        mismatch_norm = float(np.asarray(diag.get("mismatch_norm", np.nan), dtype=np.float64))
        if np.isfinite(mismatch_norm) and mismatch_norm < best_evaluated_mismatch_norm:
            best_evaluated_indata = deepcopy(current_indata)
            best_evaluated_iteration = int(iteration)
            best_evaluated_mismatch_norm = float(mismatch_norm)
        current_update_norm = float(np.asarray(update_norm, dtype=np.float64))
        history.append(
            BootstrapCurrentIteration(
                iteration=int(iteration),
                mismatch_norm=mismatch_norm,
                current_update_norm=current_update_norm,
                curtor=float(profile["curtor"]),
                ac_aux_s=tuple(float(x) for x in np.asarray(profile["ac_aux_s"], dtype=np.float64)),
                ac_aux_f=tuple(float(x) for x in np.asarray(profile["ac_aux_f"], dtype=np.float64)),
                beta_total=None if diag.get("beta_total") is None else float(diag["beta_total"]),
                aspect=None if diag.get("aspect") is None else float(diag["aspect"]),
                mean_iota=None if diag.get("mean_iota") is None else float(diag["mean_iota"]),
                fsq_total=None if diag.get("fsq_total") is None else float(diag["fsq_total"]),
                effective_damping=float(effective_damping),
                current_update_limited=bool(current_update_limited),
                unlimited_current_update_norm=float(unlimited_update_norm),
                max_current_update_norm=(
                    None if options.max_current_update_norm is None else float(options.max_current_update_norm)
                ),
            )
        )
        current_indata = next_indata
        if current_update_norm <= float(options.current_tol) and (
            not np.isfinite(mismatch_norm) or mismatch_norm <= float(options.mismatch_tol)
        ):
            converged = True
            reason = "current_and_mismatch_tolerances"
            break

    returned_best_evaluated = False
    return_indata = current_indata
    if (
        not converged
        and bool(options.return_best_evaluated_on_max_iter)
        and best_evaluated_iteration is not None
    ):
        return_indata = best_evaluated_indata
        returned_best_evaluated = True

    return BootstrapCurrentResult(
        indata=return_indata,
        history=tuple(history),
        converged=bool(converged),
        reason=reason,
        last_run=last_run,
        last_diagnostics=last_diag,
        returned_best_evaluated=bool(returned_best_evaluated),
        best_evaluated_iteration=best_evaluated_iteration,
        best_evaluated_mismatch_norm=(
            None if not np.isfinite(best_evaluated_mismatch_norm) else float(best_evaluated_mismatch_norm)
        ),
    )


__all__ = [
    "BootstrapCurrentIteration",
    "BootstrapCurrentOptions",
    "BootstrapCurrentResult",
    "CurrentUpdatePolicy",
    "apply_current_profile_to_indata",
    "bootstrap_current_fixed_point",
    "bootstrap_current_update_to_indata",
    "damp_current_profile",
    "dpsi_ds_from_vmec_phiedge",
    "integrate_current_derivative",
    "redl_current_derivative_update",
    "redl_current_integrating_factor_update",
    "redl_current_rhs",
    "vmec_current_profile_from_bootstrap_update",
]
