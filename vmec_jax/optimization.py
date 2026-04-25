"""Optimization-oriented helpers for vmec_jax workflows."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, fields, is_dataclass, replace
import json
import math
import os
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ._compat import jnp
from .boundary import BoundaryCoeffs
from .booz_input import BoozXformInputs, booz_xform_inputs_from_state
from .energy import FluxProfiles, flux_profiles_from_indata
from .field import signgs_from_sqrtg
from .geom import eval_geom
from .init_guess import initial_guess_from_boundary
from .namelist import InData, write_indata
from .modes import ModeTable
from .state import VMECState
from .static import VMECStatic


@dataclass(frozen=True)
class BoundaryParamSpec:
    """Descriptor for a boundary Fourier coefficient parameter."""

    name: str
    kind: str
    index: int
    m: int
    n: int


@dataclass(frozen=True)
class FixedBoundaryContext:
    """Bundled inputs for repeated fixed-boundary solves."""

    st_guess: VMECState
    signgs: int
    flux: FluxProfiles
    pressure: jnp.ndarray
    booz_inputs: BoozXformInputs


def _coeff_label(prefix: str, m: int, n: int) -> str:
    n_str = f"{n:+d}".replace("+", "")
    return f"{prefix}{m}{n_str}"


def extend_boundary_for_max_mode(
    indata,
    static: "VMECStatic",
    boundary: "BoundaryCoeffs",
    max_mode: int,
) -> tuple:
    """Extend *indata*, *static*, and *boundary* to support ``max_mode`` DOFs.

    When ``max_mode`` exceeds the modes already present in the VMEC input (i.e.
    when ``mpol <= max_mode`` or ``ntor < max_mode``), this helper rebuilds the
    static grid and boundary with a larger mode table — mirroring SIMSOPT's
    behaviour where ``surf.fixed_range(mmax=max_mode, ...)`` silently unfixes
    zero-valued harmonics that already exist in the surface's full Fourier grid.

    The returned *boundary* has the same non-zero values as the original, with
    all new high-mode entries initialised to zero.

    Parameters
    ----------
    indata:
        VMEC namelist input.  *Copied* before modification; the original is
        untouched.
    static:
        Current :class:`~vmec_jax.static.VMECStatic`.  Replaced if extension
        is needed.
    boundary:
        Current boundary coefficients.  Replaced if extension is needed.
    max_mode:
        Desired maximum mode number.  The extended mode table will have
        ``mpol = ntor = max(max(mpol_cur, ntor_cur), max(5, max_mode + 2))``
        so that the VMEC solver resolution (and hence the QS metric
        normalisation) is independent of *max_mode*.

    Returns
    -------
    tuple
        ``(new_indata, new_static, new_boundary)`` — identical to the inputs
        when no extension is required.
    """
    from .config import config_from_indata
    from .static import build_static
    from .boundary import boundary_from_indata
    from .namelist import InData

    cur_mpol = int(indata.get_int("MPOL", 6))
    cur_ntor = int(indata.get_int("NTOR", 0))
    # Use at least mpol=ntor=5 so the VMEC solver resolution (and hence the
    # QS metric normalisation) is independent of max_mode.  Without this floor
    # max_mode=1 would run with mpol=3 and give a different initial QS value
    # than max_mode=2/3, making cross-mode comparisons misleading.
    need_mpol = max(5, max_mode + 2)   # VMEC mpol = max_m + 1; add extra headroom
    need_ntor = max(5, max_mode + 2)

    if need_mpol <= cur_mpol and need_ntor <= cur_ntor:
        return indata, static, boundary   # nothing to do

    new_mpol = max(cur_mpol, need_mpol)
    new_ntor = max(cur_ntor, need_ntor)

    # Shallow-copy the scalars dict so we don't mutate the caller's indata.
    new_scalars = dict(indata.scalars)
    new_scalars["MPOL"] = new_mpol
    new_scalars["NTOR"] = new_ntor
    new_indata = InData(
        scalars=new_scalars,
        indexed=indata.indexed,
        source_path=indata.source_path,
    )

    cfg = config_from_indata(new_indata)
    new_static = build_static(cfg)
    new_boundary = boundary_from_indata(new_indata, new_static.modes)

    print(
        f"  [extend_boundary_for_max_mode] extended mpol {cur_mpol}→{new_mpol}, "
        f"ntor {cur_ntor}→{new_ntor}  "
        f"(modes table size: {len(new_static.modes.m)})"
    )
    return new_indata, new_static, new_boundary


def rebuild_indata_with_resolution(indata, *, mpol: int, ntor: int):
    """Return a copy of ``indata`` with updated VMEC spectral resolution."""
    from .namelist import InData

    new_scalars = dict(indata.scalars)
    new_scalars["MPOL"] = int(mpol)
    new_scalars["NTOR"] = int(ntor)
    return InData(
        scalars=new_scalars,
        indexed=indata.indexed,
        source_path=indata.source_path,
    )


def boundary_param_specs(
    boundary: BoundaryCoeffs,
    modes: ModeTable,
    *,
    max_mode: int | None = None,
    max_m: int | None = None,
    max_n: int | None = None,
    min_coeff: float = 0.0,
    include: Sequence[str] = ("rc", "zs"),
    fix: Sequence[str] = ("rc00",),
    include_axis: bool = False,
) -> list[BoundaryParamSpec]:
    """Build parameter specifications for boundary optimization.

    Parameters
    ----------
    boundary:
        Boundary coefficients aligned with ``modes``.
    modes:
        Mode table describing (m, n) pairs.
    max_mode:
        Convenience limit applied to both ``max_m`` and ``max_n`` when provided.
    max_m, max_n:
        Limits for m and n mode numbers. If ``None``, no limit is applied.
    min_coeff:
        Minimum absolute coefficient magnitude to include.
    include:
        Iterable of coefficient families to include. Supported values are
        ``"rc"``, ``"rs"``, ``"zc"``, ``"zs"``.
    fix:
        Iterable of parameter names to exclude (e.g. ``["rc00"]``).
    include_axis:
        If ``True``, include the (m=0,n=0) mode. By default it is excluded.
    """
    max_m = max_m if max_m is not None else max_mode
    max_n = max_n if max_n is not None else max_mode
    include_set = {item.lower() for item in include}
    fix_set = {item.lower() for item in fix}

    r_cos = np.asarray(boundary.R_cos)
    r_sin = np.asarray(boundary.R_sin)
    z_cos = np.asarray(boundary.Z_cos)
    z_sin = np.asarray(boundary.Z_sin)

    specs: list[BoundaryParamSpec] = []
    for k, (m_i, n_i) in enumerate(zip(np.asarray(modes.m), np.asarray(modes.n))):
        m_i = int(m_i)
        n_i = int(n_i)
        if m_i < 0:
            continue
        if max_m is not None and abs(m_i) > int(max_m):
            continue
        if max_n is not None and abs(n_i) > int(max_n):
            continue

        if not include_axis and m_i == 0 and n_i == 0:
            continue

        if "rc" in include_set and abs(float(r_cos[k])) >= float(min_coeff):
            name = _coeff_label("rc", m_i, n_i)
            if name.lower() not in fix_set:
                specs.append(BoundaryParamSpec(name, "rc", k, m_i, n_i))
        if "rs" in include_set and abs(float(r_sin[k])) >= float(min_coeff):
            name = _coeff_label("rs", m_i, n_i)
            if name.lower() not in fix_set:
                specs.append(BoundaryParamSpec(name, "rs", k, m_i, n_i))
        if "zc" in include_set and abs(float(z_cos[k])) >= float(min_coeff):
            name = _coeff_label("zc", m_i, n_i)
            if name.lower() not in fix_set:
                specs.append(BoundaryParamSpec(name, "zc", k, m_i, n_i))
        if "zs" in include_set and abs(float(z_sin[k])) >= float(min_coeff):
            name = _coeff_label("zs", m_i, n_i)
            if name.lower() not in fix_set:
                specs.append(BoundaryParamSpec(name, "zs", k, m_i, n_i))

    return specs


def boundary_param_names(specs: Sequence[BoundaryParamSpec]) -> list[str]:
    """Return the parameter names for a list of specs."""
    return [spec.name for spec in specs]


def lift_boundary_params(
    source_specs: Sequence[BoundaryParamSpec],
    source_params,
    target_specs: Sequence[BoundaryParamSpec],
) -> np.ndarray:
    """Lift a parameter vector defined on one boundary basis to another.

    Parameters
    ----------
    source_specs:
        Parameter specification list describing ``source_params``.
    source_params:
        1-D parameter vector aligned with ``source_specs``.
    target_specs:
        Target parameter specification list.

    Returns
    -------
    np.ndarray
        Parameter vector aligned with ``target_specs``. Parameters present in
        both lists are copied by name; all others are initialised to zero.
    """
    source_vals = {
        spec.name: float(value) for spec, value in zip(source_specs, np.asarray(source_params, dtype=float))
    }
    return np.asarray([source_vals.get(spec.name, 0.0) for spec in target_specs], dtype=float)


def apply_boundary_params(
    boundary: BoundaryCoeffs,
    specs: Sequence[BoundaryParamSpec],
    params: jnp.ndarray,
) -> BoundaryCoeffs:
    """Apply parameter updates to a boundary coefficient set."""
    r_cos = jnp.asarray(boundary.R_cos)
    r_sin = jnp.asarray(boundary.R_sin)
    z_cos = jnp.asarray(boundary.Z_cos)
    z_sin = jnp.asarray(boundary.Z_sin)

    for idx, spec in enumerate(specs):
        if spec.kind == "rc":
            r_cos = r_cos.at[spec.index].add(params[idx])
        elif spec.kind == "rs":
            r_sin = r_sin.at[spec.index].add(params[idx])
        elif spec.kind == "zc":
            z_cos = z_cos.at[spec.index].add(params[idx])
        elif spec.kind == "zs":
            z_sin = z_sin.at[spec.index].add(params[idx])
        else:
            raise ValueError(f"Unknown boundary parameter kind '{spec.kind}'")

    return BoundaryCoeffs(R_cos=r_cos, R_sin=r_sin, Z_cos=z_cos, Z_sin=z_sin)


def _indexed_boundary_maps_from_boundary(
    boundary: BoundaryCoeffs,
    modes: ModeTable,
) -> dict[str, dict[tuple[int, int], float]]:
    """Build sparse VMEC namelist boundary maps from dense boundary coefficients."""
    maps = {"RBC": {}, "RBS": {}, "ZBC": {}, "ZBS": {}}
    seen: set[tuple[int, int]] = set()
    m_arr = np.asarray(modes.m, dtype=int)
    n_arr = np.asarray(modes.n, dtype=int)
    r_cos = np.asarray(boundary.R_cos, dtype=float)
    r_sin = np.asarray(boundary.R_sin, dtype=float)
    z_cos = np.asarray(boundary.Z_cos, dtype=float)
    z_sin = np.asarray(boundary.Z_sin, dtype=float)
    for idx, (m_i, n_i) in enumerate(zip(m_arr, n_arr)):
        m_i = int(m_i)
        n_i = int(n_i)
        if m_i < 0:
            continue
        key = (n_i, m_i)
        if key in seen:
            continue
        seen.add(key)
        maps["RBC"][key] = float(r_cos[idx])
        maps["RBS"][key] = float(r_sin[idx])
        maps["ZBC"][key] = float(z_cos[idx])
        maps["ZBS"][key] = float(z_sin[idx])
    return maps


def surface_indices_from_s(
    s_half: np.ndarray,
    surfaces: Sequence[int | float],
) -> tuple[list[int], np.ndarray]:
    """Map surface requests to half-mesh indices."""
    indices: list[int] = []
    for val in surfaces:
        if isinstance(val, float) and 0.0 <= val <= 1.0:
            indices.append(int(np.argmin(np.abs(s_half - val))))
        else:
            indices.append(int(val) - 1)
    return indices, s_half[np.asarray(indices)]


def surface_indices_from_static(
    static: VMECStatic,
    surfaces: Sequence[int | float],
) -> tuple[list[int], np.ndarray]:
    """Map surface requests to indices using a VMEC static object."""
    s_half = 0.5 * (np.asarray(static.s[:-1]) + np.asarray(static.s[1:]))
    return surface_indices_from_s(s_half, surfaces)


def parse_surface_list(text: str) -> list[float | int]:
    """Parse a comma-separated surface list into floats/ints.

    Integers are treated as 1-based indices; floats in [0, 1] are treated as
    normalized toroidal flux ``s`` values.
    """
    items: list[float | int] = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if any(ch in raw for ch in (".", "e", "E")):
            items.append(float(raw))
        else:
            items.append(int(raw))
    return items


def prepare_fixed_boundary_context(
    *,
    static: VMECStatic,
    indata,
    boundary: BoundaryCoeffs,
    vmec_project: bool = False,
) -> FixedBoundaryContext:
    """Precompute common fixed-boundary inputs for optimization loops."""
    st_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=vmec_project)
    geom = eval_geom(st_guess, static)
    signgs = signgs_from_sqrtg(np.asarray(geom.sqrtg), axis_index=1)
    flux = flux_profiles_from_indata(indata, static.s, signgs=signgs)
    pressure = jnp.zeros_like(jnp.asarray(static.s))
    booz_inputs = booz_xform_inputs_from_state(
        state=st_guess,
        static=static,
        indata=indata,
        signgs=signgs,
    )
    return FixedBoundaryContext(
        st_guess=st_guess,
        signgs=signgs,
        flux=flux,
        pressure=pressure,
        booz_inputs=booz_inputs,
    )


def gauss_newton_least_squares(
    residual_fun,
    jacobian_fun,
    x0,
    *,
    max_nfev: int = 10,
    ftol: float = 1e-4,
    gtol: float = 1e-4,
    xtol: float = 1e-4,
    x_scale=None,
    forward_residual_fun=None,
    post_jacobian_callback=None,
    exact_residual_after_jacobian_fun=None,
    damping_factors: Sequence[float] | None = None,
    verbose: int = 1,
):
    """Solve a nonlinear least-squares problem with a concrete Gauss-Newton loop.

    This helper is intended for small- to medium-sized outer optimization
    problems that already provide concrete residual and Jacobian callbacks.
    Unlike generic traced solvers, it never asks JAX to differentiate through
    the outer least-squares algorithm itself.

    Parameters
    ----------
    residual_fun:
        Callable ``(x) -> residuals`` for accepted steps. May be expensive
        (e.g. builds a discrete-adjoint tape).
    jacobian_fun:
        Callable ``(x) -> J`` where J has shape ``(n_residuals, n_params)``.
    x0:
        Initial parameter vector.
    max_nfev:
        Maximum total number of residual/Jacobian evaluations.
    ftol, gtol, xtol:
        Convergence tolerances on cost reduction, gradient, and step norm.
    x_scale:
        Optional per-parameter scaling vector.
    forward_residual_fun:
        Optional cheaper residual callback for line-search trial evaluations.
        When provided, line-search trial points are evaluated with this
        function instead of ``residual_fun``.
    post_jacobian_callback:
        Optional zero-argument callable invoked immediately after each
        ``jacobian_fun`` call.  Useful for releasing JIT caches between
        expensive Jacobian evaluations, e.g.
        ``post_jacobian_callback=jax.clear_caches``.
    exact_residual_after_jacobian_fun:
        Optional zero-argument callable returning an exact residual vector
        that corresponds to the state used by the most recent ``jacobian_fun``
        call.  When provided, the residual used for gradient computation and
        convergence checks is replaced by this exact value after each Jacobian
        evaluation.  This is useful when ``forward_residual_fun`` is a relaxed
        solver and the exact state can be extracted from a side-effect cache
        (e.g. the ``_exact_cache`` in the discrete-adjoint QH example).
    damping_factors:
        Optional Levenberg damping factors tried after the undamped
        Gauss-Newton step fails the relaxed trial line search.  The default
        keeps the zero-damping fast path first and only pays for damped normal
        equations on difficult steps.
    verbose:
        Verbosity level (0 = silent, 1 = iteration table).
    """
    x = np.asarray(x0, dtype=float).copy()
    scale = np.ones_like(x) if x_scale is None else np.asarray(x_scale, dtype=float).copy()
    scale[scale == 0.0] = 1.0
    trial_residual_fun = residual_fun if forward_residual_fun is None else forward_residual_fun
    damping_schedule = (
        (1e-6, 1e-4, 1e-2, 1.0, 100.0)
        if damping_factors is None
        else tuple(float(value) for value in damping_factors)
    )

    nfev = 0
    njev = 0
    alpha_prev = 1.0
    x_prev = None
    cost_prev = None
    accepted_residual = None
    accepted_cost = None
    accepted_step_norm = None
    success = False
    message = "maximum function evaluations exceeded"

    if verbose:
        print("   Iteration     Total nfev        Cost      Cost reduction    Step norm     Optimality")

    iteration = 0
    while nfev < int(max_nfev):
        if accepted_residual is None:
            residual = np.asarray(residual_fun(x), dtype=float).reshape(-1)
            nfev += 1
        else:
            residual = accepted_residual
            accepted_residual = None
        cost = 0.5 * float(np.dot(residual, residual))

        jacobian = np.asarray(jacobian_fun(x), dtype=float)
        njev += 1
        if exact_residual_after_jacobian_fun is not None:
            _exact_res = exact_residual_after_jacobian_fun()
            if _exact_res is not None:
                residual = np.asarray(_exact_res, dtype=float).reshape(-1)
                cost = 0.5 * float(np.dot(residual, residual))
        if post_jacobian_callback is not None:
            post_jacobian_callback()
        gradient = jacobian.T @ residual
        optimality = float(np.linalg.norm(gradient, ord=np.inf))
        if not np.isfinite(optimality):
            message = "non-finite optimality encountered"
            break
        if optimality <= float(gtol):
            success = True
            message = "`gtol` termination condition is satisfied."
            accepted_cost = cost
            accepted_step_norm = 0.0
            if verbose:
                print(
                    f"{iteration:12d}{nfev:16d}{cost:13.4e}{0.0:18.2e}{0.0:16.2e}{optimality:16.2e}"
                )
            break

        jacobian_scaled = jacobian * scale[None, :]
        normal = jacobian_scaled.T @ jacobian_scaled
        rhs = -(jacobian_scaled.T @ residual)
        try:
            step_y, *_ = np.linalg.lstsq(jacobian_scaled, -residual, rcond=None)
        except np.linalg.LinAlgError:
            message = "linear least-squares solve failed"
            break
        step = scale * np.asarray(step_y, dtype=float)
        step_norm = float(np.linalg.norm(step))
        if not np.all(np.isfinite(step)):
            message = "non-finite Gauss-Newton step encountered"
            break
        if step_norm <= float(xtol):
            success = True
            message = "`xtol` termination condition is satisfied."
            accepted_cost = cost
            accepted_step_norm = step_norm
            if verbose:
                print(
                    f"{iteration:12d}{nfev:16d}{cost:13.4e}{0.0:18.2e}{step_norm:16.2e}{optimality:16.2e}"
                )
            break

        accepted = False
        cost_trial = math.inf
        residual_trial = None
        x_trial = None
        alpha_accepted = 0.0
        step_accepted = None

        def _try_step(candidate_step, initial_alpha):
            nonlocal nfev, accepted, cost_trial, residual_trial, x_trial
            nonlocal alpha_accepted, step_accepted
            alpha = min(max(float(initial_alpha), 1.0 / 128.0), 1.0)
            for _ in range(8):
                x_candidate = x + alpha * candidate_step
                residual_candidate = np.asarray(trial_residual_fun(x_candidate), dtype=float).reshape(-1)
                nfev += 1
                cost_candidate = 0.5 * float(np.dot(residual_candidate, residual_candidate))
                if np.isfinite(cost_candidate) and cost_candidate < cost:
                    x_trial = x_candidate
                    residual_trial = residual_candidate
                    cost_trial = cost_candidate
                    alpha_accepted = alpha
                    step_accepted = candidate_step
                    accepted = True
                    break
                alpha *= 0.5
                if nfev >= int(max_nfev):
                    break
            return accepted

        _try_step(step, alpha_prev)
        if (not accepted) and nfev < int(max_nfev):
            diag = np.maximum(np.diag(normal), 1.0)
            for damping in damping_schedule:
                if damping <= 0.0:
                    continue
                try:
                    damped_y = np.linalg.solve(
                        normal + float(damping) * np.diag(diag),
                        rhs,
                    )
                except np.linalg.LinAlgError:
                    continue
                damped_step = scale * np.asarray(damped_y, dtype=float)
                if not np.all(np.isfinite(damped_step)):
                    continue
                _try_step(damped_step, 1.0)
                if accepted or nfev >= int(max_nfev):
                    break

        if not accepted:
            message = "line search failed to reduce the objective"
            accepted_cost = cost
            accepted_step_norm = 0.0
            if verbose:
                print(
                    f"{iteration:12d}{nfev:16d}{cost:13.4e}{0.0:18.2e}{0.0:16.2e}{optimality:16.2e}"
                )
            break

        cost_reduction = cost - cost_trial
        step_norm_trial = float(np.linalg.norm(alpha_accepted * step_accepted))
        if verbose:
            print(
                f"{iteration:12d}{nfev:16d}{cost_trial:13.4e}{cost_reduction:18.2e}{step_norm_trial:16.2e}{optimality:16.2e}"
            )

        x_prev = x
        cost_prev = cost
        x = x_trial
        accepted_residual = residual_trial
        accepted_cost = cost_trial
        accepted_step_norm = step_norm_trial
        alpha_prev = alpha_accepted
        iteration += 1

        if cost_prev is not None and cost_prev > 0.0 and cost_reduction <= float(ftol) * cost_prev:
            success = True
            message = "`ftol` termination condition is satisfied."
            break

    if accepted_cost is None:
        residual_final = np.asarray(residual_fun(x), dtype=float).reshape(-1)
        nfev += 1
        accepted_cost = 0.5 * float(np.dot(residual_final, residual_final))
        accepted_step_norm = 0.0

    return {
        "x": x,
        "cost": float(accepted_cost),
        "objective": float(2.0 * accepted_cost),
        "nfev": int(nfev),
        "njev": int(njev),
        "nit": int(iteration),
        "success": bool(success),
        "status": 1 if success else 0,
        "message": str(message),
        "step_norm": float(accepted_step_norm if accepted_step_norm is not None else 0.0),
        "x_prev": None if x_prev is None else np.asarray(x_prev, dtype=float),
        "cost_prev": None if cost_prev is None else float(cost_prev),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Exponential spectral scaling helpers
# ─────────────────────────────────────────────────────────────────────────────


def create_x_scale(
    specs: Sequence[BoundaryParamSpec],
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """Compute per-parameter exponential spectral scaling weights.

    Assigns smaller weights to high-mode-number boundary DOFs so that the
    optimizer penalises large perturbations in fine-scale modes more than in
    coarse-scale modes.  The weight for parameter *i* is

    .. math::

        w_i = \\exp(-\\alpha \\cdot \\max(|m_i|, |n_i|)) \\;/\\; \\exp(-\\alpha)

    so that the lowest non-trivial mode level (``max(|m|, |n|) = 1``) has
    weight 1 and higher modes have decreasing weights.

    Parameters
    ----------
    specs:
        Parameter specification list from :func:`boundary_param_specs`.
    alpha:
        Decay rate.  Larger values suppress high modes more aggressively.
        ``alpha=0`` gives equal weights (no scaling).

    Returns
    -------
    np.ndarray
        1-D array of shape ``(len(specs),)`` containing the per-DOF scales.
        Pass this as ``x_scale`` to
        :meth:`FixedBoundaryExactOptimizer.run`.
    """
    scales = np.empty(len(specs), dtype=float)
    norm = math.exp(-alpha) if alpha > 0.0 else 1.0
    for i, spec in enumerate(specs):
        level = max(abs(spec.m), abs(spec.n))
        scales[i] = math.exp(-alpha * level) / norm if alpha > 0.0 else 1.0
    return scales


# ─────────────────────────────────────────────────────────────────────────────
# QH/QA residuals factories
# ─────────────────────────────────────────────────────────────────────────────


def make_qh_residuals_fn(
    static: VMECStatic,
    indata,
    *,
    signgs: int | None = None,
    helicity_m: int = 1,
    helicity_n: int = -1,
    target_aspect: float = 7.0,
    surfaces=None,
    aspect_weight: float = 1.0,
    qs_weight: float = 1.0,
) -> Callable:
    """Build a ``residuals_from_state`` callable for quasi-helical symmetry.

    The returned function takes a :class:`~vmec_jax.state.VMECState` and
    returns a 1-D residual vector suitable for nonlinear least-squares
    optimisation.  The residuals are:

    * One aspect-ratio residual: ``aspect_weight * (aspect - target_aspect)``
    * One QS residual per selected flux surface (from
      :func:`~vmec_jax.quasisymmetry.quasisymmetry_ratio_residual_from_state`).

    Parameters
    ----------
    static:
        Pre-built :class:`~vmec_jax.static.VMECStatic`.
    indata:
        VMEC input namelist object (used to derive flux profiles and for the
        QS kernel).
    signgs:
        Sign of the Jacobian.  Computed automatically from the initial guess
        when ``None``.
    helicity_m, helicity_n:
        Helicity of the target quasi-symmetry.  Default ``(1, -1)`` gives QH.
    target_aspect:
        Target aspect ratio.
    surfaces:
        Surface coordinates (``s ∈ [0, 1]``) to evaluate quasisymmetry on.
        Defaults to ``np.arange(0, 1.01, 0.1)``.
    aspect_weight, qs_weight:
        Scalar weights applied to the aspect and QS residual blocks.
    """
    from .init_guess import initial_guess_from_boundary
    from .boundary import boundary_from_indata
    from .quasisymmetry import quasisymmetry_ratio_residual_from_state
    from .wout import equilibrium_aspect_ratio_from_state

    if surfaces is None:
        surfaces = np.arange(0.0, 1.01, 0.1)
    surfaces = np.asarray(surfaces, dtype=float)

    if signgs is None:
        try:
            boundary_init = boundary_from_indata(indata, static.modes)
            state0 = initial_guess_from_boundary(static, boundary_init, indata)
            from .geom import eval_geom as _eval_geom
            geom = _eval_geom(state0, static)
            signgs = int(signgs_from_sqrtg(np.asarray(geom.sqrtg), axis_index=1))
        except Exception:
            signgs = 1

    flux = flux_profiles_from_indata(indata, static.s, signgs=signgs)
    pressure = jnp.zeros_like(jnp.asarray(static.s))

    def _qs_eval_from_state(state: VMECState):
        return quasisymmetry_ratio_residual_from_state(
            state=state,
            static=static,
            indata=indata,
            signgs=signgs,
            flux_local=flux,
            prof_local={"pressure": pressure},
            pressure_local=pressure,
            surfaces=surfaces,
            helicity_m=helicity_m,
            helicity_n=helicity_n,
        )

    def residuals_from_state(state: VMECState) -> jnp.ndarray:
        aspect = equilibrium_aspect_ratio_from_state(state=state, static=static)
        qs = _qs_eval_from_state(state)
        aspect_residual = jnp.asarray([float(aspect_weight) * (aspect - target_aspect)],
                                      dtype=jnp.float64)
        qs_residual = jnp.asarray(qs["residuals1d"], dtype=jnp.float64) * float(qs_weight)
        return jnp.concatenate([aspect_residual, qs_residual])

    def state_cotangent_operator_from_packed(packed_state, layout):
        from ._compat import jax, jnp as _jnp
        from .state import unpack_state

        packed_state = _jnp.asarray(packed_state, dtype=_jnp.float64)

        def _aspect_from_packed(packed):
            state = unpack_state(packed, layout)
            aspect = equilibrium_aspect_ratio_from_state(state=state, static=static)
            return float(aspect_weight) * (aspect - target_aspect)

        def _qs_from_packed(packed):
            state = unpack_state(packed, layout)
            qs = _qs_eval_from_state(state)
            return jnp.asarray(qs["residuals1d"], dtype=jnp.float64) * float(qs_weight)

        _, aspect_vjp = jax.vjp(_aspect_from_packed, packed_state)
        _, qs_vjp = jax.vjp(_qs_from_packed, packed_state)

        def _apply(residual_cotangent):
            residual_cotangent = _jnp.asarray(residual_cotangent, dtype=_jnp.float64).reshape(-1)
            total = _jnp.zeros_like(packed_state)
            aspect_cot = residual_cotangent[0]
            total = total + jax.lax.cond(
                _jnp.any(aspect_cot != 0.0),
                lambda cot: aspect_vjp(cot)[0],
                lambda cot: _jnp.zeros_like(packed_state),
                aspect_cot,
            )
            qs_cot = residual_cotangent[1:]
            total = total + jax.lax.cond(
                _jnp.any(qs_cot != 0.0),
                lambda cot: qs_vjp(cot)[0],
                lambda cot: _jnp.zeros_like(packed_state),
                qs_cot,
            )
            return total

        return _apply

    def state_cotangent_from_packed(packed_state, layout, residual_cotangent):
        return state_cotangent_operator_from_packed(packed_state, layout)(residual_cotangent)

    residuals_from_state._n_non_qs = 1
    residuals_from_state._qs_total_from_state = (
        lambda state: float(_qs_eval_from_state(state)["total"]) * float(qs_weight) ** 2
    )
    residuals_from_state._state_cotangent_from_packed = state_cotangent_from_packed
    residuals_from_state._state_cotangent_operator_from_packed = state_cotangent_operator_from_packed

    return residuals_from_state


def make_qs_residuals_fn(
    static: VMECStatic,
    indata,
    *,
    signgs: int | None = None,
    helicity_m: int = 1,
    helicity_n: int = 0,
    target_aspect: float | None = None,
    target_iota: float | None = None,
    surfaces=None,
    aspect_weight: float = 1.0,
    qs_weight: float = 1.0,
    iota_weight: float = 1.0,
) -> Callable:
    """General quasisymmetry residuals factory supporting QH and QA objectives.

    Builds a combined residual vector with optional aspect-ratio and mean-iota
    targets.  This is the recommended factory for new workflows; use it for QA
    (``helicity_m=1, helicity_n=0``) or QH (``helicity_m=1, helicity_n=-1``).

    Parameters
    ----------
    static:
        Pre-built :class:`~vmec_jax.static.VMECStatic`.
    indata:
        VMEC input namelist (used to derive flux profiles and for the QS kernel).
    signgs:
        Sign of the Jacobian.  Computed automatically when ``None``.
    helicity_m, helicity_n:
        Helicity of the target quasisymmetry.
        QA: ``(1, 0)``, QH: ``(1, -1)`` or ``(1, 1)``.
    target_aspect:
        If given, adds one aspect-ratio residual
        ``aspect_weight * (aspect - target_aspect)``.
    target_iota:
        If given, adds one mean-iota residual
        ``iota_weight * (mean_iota - target_iota)``.
    surfaces:
        Surface coordinates (``s ∈ [0, 1]``) to evaluate quasisymmetry on.
        Defaults to ``np.arange(0, 1.01, 0.1)``.
    aspect_weight, qs_weight, iota_weight:
        Scalar weights applied to the corresponding residual blocks.
    """
    from .boundary import boundary_from_indata
    from .init_guess import initial_guess_from_boundary
    from .quasisymmetry import quasisymmetry_ratio_residual_from_state
    from .wout import equilibrium_aspect_ratio_from_state, equilibrium_iota_profiles_from_state

    if surfaces is None:
        surfaces = np.arange(0.0, 1.01, 0.1)
    surfaces = np.asarray(surfaces, dtype=float)

    if signgs is None:
        try:
            boundary_init = boundary_from_indata(indata, static.modes)
            state0 = initial_guess_from_boundary(static, boundary_init, indata)
            from .geom import eval_geom as _eval_geom
            geom = _eval_geom(state0, static)
            signgs = int(signgs_from_sqrtg(np.asarray(geom.sqrtg), axis_index=1))
        except Exception:
            signgs = 1

    flux = flux_profiles_from_indata(indata, static.s, signgs=signgs)
    pressure = jnp.zeros_like(jnp.asarray(static.s))
    _signgs = signgs
    _indata = indata

    def _qs_eval_from_state(state: VMECState):
        return quasisymmetry_ratio_residual_from_state(
            state=state,
            static=static,
            indata=indata,
            signgs=signgs,
            flux_local=flux,
            prof_local={"pressure": pressure},
            pressure_local=pressure,
            surfaces=surfaces,
            helicity_m=helicity_m,
            helicity_n=helicity_n,
        )

    def residuals_from_state(state: VMECState) -> jnp.ndarray:
        parts: list[jnp.ndarray] = []

        if target_aspect is not None:
            aspect = equilibrium_aspect_ratio_from_state(state=state, static=static)
            parts.append(jnp.asarray(
                [float(aspect_weight) * (aspect - target_aspect)], dtype=jnp.float64
            ))

        if target_iota is not None:
            _chips, _iotas, iotaf = equilibrium_iota_profiles_from_state(
                state=state, static=static, indata=_indata, signgs=_signgs,
            )
            iotas = jnp.asarray(_iotas, dtype=jnp.float64)
            mean_iota = (
                jnp.asarray(0.0, dtype=iotas.dtype)
                if int(iotas.shape[0]) <= 1
                else jnp.mean(iotas[1:])
            )
            parts.append(jnp.asarray(
                [float(iota_weight) * (mean_iota - target_iota)], dtype=jnp.float64
            ))

        qs = _qs_eval_from_state(state)
        parts.append(jnp.asarray(qs["residuals1d"], dtype=jnp.float64) * float(qs_weight))

        return jnp.concatenate(parts)

    def state_cotangent_operator_from_packed(packed_state, layout):
        from ._compat import jax, jnp as _jnp
        from .state import unpack_state

        packed_state = _jnp.asarray(packed_state, dtype=_jnp.float64)
        blocks: list[tuple[slice | int, Callable, bool]] = []
        offset = 0

        if target_aspect is not None:
            block_index = offset
            offset += 1

            def _aspect_from_packed(packed):
                state = unpack_state(packed, layout)
                aspect = equilibrium_aspect_ratio_from_state(state=state, static=static)
                return float(aspect_weight) * (aspect - target_aspect)

            _, aspect_vjp = jax.vjp(_aspect_from_packed, packed_state)
            blocks.append((block_index, aspect_vjp, False))

        if target_iota is not None:
            block_index = offset
            offset += 1

            def _iota_from_packed(packed):
                state = unpack_state(packed, layout)
                _chips, _iotas, _iotaf = equilibrium_iota_profiles_from_state(
                    state=state, static=static, indata=_indata, signgs=_signgs,
                )
                del _chips, _iotaf
                iotas = _jnp.asarray(_iotas, dtype=_jnp.float64)
                mean_iota = (
                    _jnp.asarray(0.0, dtype=iotas.dtype)
                    if int(iotas.shape[0]) <= 1
                    else _jnp.mean(iotas[1:])
                )
                return float(iota_weight) * (mean_iota - target_iota)

            _, iota_vjp = jax.vjp(_iota_from_packed, packed_state)
            blocks.append((block_index, iota_vjp, True))

        qs_slice = slice(offset, None)

        def _qs_from_packed(packed):
            state = unpack_state(packed, layout)
            qs = _qs_eval_from_state(state)
            return _jnp.asarray(qs["residuals1d"], dtype=_jnp.float64) * float(qs_weight)

        _, qs_vjp = jax.vjp(_qs_from_packed, packed_state)
        blocks.append((qs_slice, qs_vjp, False))

        def _apply(residual_cotangent):
            residual_cotangent = _jnp.asarray(residual_cotangent, dtype=_jnp.float64).reshape(-1)
            total = _jnp.zeros_like(packed_state)
            for selector, vjp_fun, sanitize in blocks:
                cot = residual_cotangent[selector]

                def _active(cot_block):
                    contribution = vjp_fun(cot_block)[0]
                    if sanitize:
                        # The current-driven iota path has axis/near-axis gauge-null
                        # cotangent entries. Dense JVP columns are finite there;
                        # zeroing the null reverse entries gives the matching
                        # transpose on the boundary-parameter subspace.
                        contribution = _jnp.nan_to_num(
                            contribution, nan=0.0, posinf=0.0, neginf=0.0
                        )
                    return contribution

                total = total + jax.lax.cond(
                    _jnp.any(cot != 0.0),
                    _active,
                    lambda cot_block: _jnp.zeros_like(packed_state),
                    cot,
                )
            return total

        return _apply

    def state_cotangent_from_packed(packed_state, layout, residual_cotangent):
        return state_cotangent_operator_from_packed(packed_state, layout)(residual_cotangent)

    residuals_from_state._n_non_qs = int(target_aspect is not None) + int(target_iota is not None)
    residuals_from_state._qs_total_from_state = (
        lambda state: float(_qs_eval_from_state(state)["total"]) * float(qs_weight) ** 2
    )
    residuals_from_state._state_cotangent_from_packed = state_cotangent_from_packed
    residuals_from_state._state_cotangent_operator_from_packed = state_cotangent_operator_from_packed
    return residuals_from_state


# ─────────────────────────────────────────────────────────────────────────────
# FixedBoundaryExactOptimizer
# ─────────────────────────────────────────────────────────────────────────────

class FixedBoundaryExactOptimizer:
    """End-to-end optimizer for fixed-boundary VMEC equilibria.

    Wraps the discrete-adjoint Jacobian machinery into a clean interface
    analogous to SIMSOPT's ``Vmec + QuasisymmetryRatioResidual +
    LeastSquaresProblem`` trio — but stays entirely within vmec_jax and
    requires no finite differences.

    Parameters
    ----------
    static:
        Pre-built :class:`~vmec_jax.static.VMECStatic`.
    indata:
        VMEC input namelist (passed to the solver and wout writer).
    boundary:
        Reference boundary Fourier coefficients.
    specs:
        Parameter descriptors from :func:`boundary_param_specs`.
    residuals_fn:
        Callable ``(VMECState) -> jnp.ndarray`` returning the residual vector
        to minimise.  Build with :func:`make_qh_residuals_fn` or supply your
        own.
    boundary_input:
        Optional boundary coefficients in VMEC input convention. When
        provided, optimization parameters are applied in that convention and
        then converted internally with ``apply_m1_constraint=False``.
    inner_max_iter, inner_ftol:
        Accepted-point VMEC residual solve budget. ``inner_max_iter <= 0``
        means "use the VMEC input-deck NITER / NITER_ARRAY budget", and
        ``inner_ftol <= 0`` means "use the VMEC input-deck FTOL / FTOL_ARRAY".
    trial_max_iter, trial_ftol:
        Trial-point VMEC residual solve budget used by the relaxed forward
        callback inside the optimizer. ``trial_max_iter <= 0`` means "use the
        same budget selected from the VMEC input deck / accepted-point solve"
        instead of forcing a separate override", and ``trial_ftol <= 0``
        means "use that same accepted-point FTOL" instead of forcing a
        separate relaxed tolerance.
    solver_device:
        Device for the exact optimizer's inner solves and Jacobian callbacks.
        ``None`` / ``"auto"`` / ``"default"`` inherit JAX's active default
        device. Pass ``"cpu"`` or ``"gpu"`` to explicitly run callbacks under
        that device context.

    Example
    -------
    .. code-block:: python

        import numpy as np
        import vmec_jax as vj

        cfg, indata = vj.load_config("input.nfp4_QH_warm_start")
        static       = vj.build_static(cfg)
        boundary     = vj.boundary_from_indata(indata, static.modes)

        specs        = vj.boundary_param_specs(boundary, static.modes, max_mode=2)
        residuals_fn = vj.make_qh_residuals_fn(static, indata)

        opt    = vj.FixedBoundaryExactOptimizer(static, indata, boundary, specs, residuals_fn)
        result = opt.run(np.zeros(len(specs)), max_nfev=15)

        opt.save_wout("wout_final.nc", result["x"])
        opt.save_history("history.json", result)
    """

    def __init__(
        self,
        static: VMECStatic,
        indata,
        boundary: BoundaryCoeffs,
        specs: Sequence[BoundaryParamSpec],
        residuals_fn: Callable,
        boundary_input: BoundaryCoeffs | None = None,
        *,
        inner_max_iter: int | None = None,
        inner_ftol: float | None = None,
        trial_max_iter: int | None = None,
        trial_ftol: float | None = None,
        solver_device: str | None = None,
    ) -> None:
        self._solver_device_name = self._resolve_solver_device(solver_device)
        self._inside_solver_device_context = False
        if self._solver_device_name is not None:
            static = self._move_to_solver_device(static)
            boundary = self._move_to_solver_device(boundary)
            if boundary_input is not None:
                boundary_input = self._move_to_solver_device(boundary_input)

        self._static = static
        self._indata = indata
        self._boundary = boundary
        self._boundary_input = boundary_input
        self._specs = list(specs)
        self._residuals_fn = residuals_fn
        self._n_qs: int | None = getattr(residuals_fn, "_n_qs", None)
        self._n_non_qs: int = int(getattr(residuals_fn, "_n_non_qs", 1))
        self._qs_total_from_state_fn = getattr(residuals_fn, "_qs_total_from_state", None)

        # Derive signgs from the initial guess.
        state0 = initial_guess_from_boundary(static, boundary, indata, vmec_project=True)
        geom0 = eval_geom(state0, static)
        self._signgs = int(signgs_from_sqrtg(np.asarray(geom0.sqrtg), axis_index=1))
        self._flux = flux_profiles_from_indata(indata, static.s, signgs=self._signgs)

        self._layout = state0.layout

        # Solver settings derived from indata.
        self._inner_max_iter = self._read_last_array("NITER_ARRAY", "NITER", 1500, int)
        self._inner_ftol = self._read_last_array("FTOL_ARRAY", "FTOL", 1e-13, float)
        self._step_size = float(indata.get_float("DELT", 1.0))
        if inner_max_iter is not None and int(inner_max_iter) > 0:
            self._inner_max_iter = int(inner_max_iter)
        if inner_ftol is not None and float(inner_ftol) > 0.0:
            self._inner_ftol = float(inner_ftol)

        _base = dict(
            indata=indata,
            signgs=self._signgs,
            step_size=self._step_size,
            include_constraint_force=True,
            apply_m1_constraints=True,
            precond_radial_alpha=0.5,
            precond_lambda_alpha=0.5,
            mode_diag_exponent=0.0,
            auto_flip_force=False,
            divide_by_scalxc_for_update=False,
            lambda_update_scale=1.0,
            enforce_vmec_lambda_axis=True,
            vmec2000_control=True,
            strict_update=True,
            backtracking=False,
            reference_mode=False,
            use_restart_triggers=True,
            verbose=False,
            verbose_vmec2000_table=False,
            jit_forces="auto",
            use_scan=False,
            light_history=True,
            # The optimizer only ever consumes `result.state` from these inner
            # solves. Keeping the full resume_state payload alive in diagnostics
            # needlessly retains large cached arrays/checkpoints across SciPy
            # callbacks and is a major source of RSS growth on converged runs.
            resume_state_mode="none",
        )
        self._exact_solver_kwargs = dict(_base)
        self._trial_solver_kwargs = dict(
            _base,
            # Trial-point residuals do not need an adjoint tape.  On CPU the
            # VMEC2000-style Python loop is still faster, while on GPU the scan
            # loop avoids thousands of small host-dispatched kernels.
            jit_forces="auto",
            use_scan=self._use_scan_for_trial_solves(),
        )
        self._trial_max_iter = min(self._inner_max_iter, 800)
        if trial_max_iter is not None:
            if int(trial_max_iter) > 0:
                self._trial_max_iter = int(trial_max_iter)
            else:
                self._trial_max_iter = int(self._inner_max_iter)
        if trial_ftol is None:
            self._trial_ftol = max(self._inner_ftol, 1e-10)
        elif float(trial_ftol) > 0.0:
            self._trial_ftol = float(trial_ftol)
        else:
            self._trial_ftol = float(self._inner_ftol)

        # Single-entry caches: keep the heavy adjoint tape only while the
        # current accepted-point Jacobian needs it, but retain the much smaller
        # solved state so final metrics/wout writing do not rerun VMEC.
        self._exact_cache: dict = {}
        self._exact_state_cache: dict = {}
        self._discrete_jacobian_helper_cache: dict = {}
        self._scan_exact_helper_cache: dict = {}
        self._scan_exact_path = self._select_exact_path()
        self._last_jacobian_residual: np.ndarray | None = None
        self._profile: dict[str, dict[str, float | int]] = {}

        # History collected during optimisation.
        self._history: list[dict] = []
        self._wall_t0: float = 0.0
        self._last_jacobian_key: list = [None]
        self._iota_fn = None  # set by run() when iota tracking is requested

    # ── private helpers ───────────────────────────────────────────────────────

    def _resolve_solver_device(self, solver_device: str | None) -> str | None:
        name = "auto" if solver_device is None else str(solver_device).strip().lower()
        if name in ("", "none", "auto", "default"):
            return None
        return name

    def _select_exact_path(self) -> str:
        """Choose the accepted-point differentiation path.

        The established non-scan discrete-adjoint tape is currently the fastest
        cold path on both CPU and GPU for QA/QH optimization callbacks.  The
        scan-differentiated path remains available via
        ``VMEC_JAX_OPT_EXACT_PATH=scan`` for diagnostics and future GPU work.
        """
        forced = os.getenv("VMEC_JAX_OPT_EXACT_PATH", "").strip().lower()
        if forced in ("scan", "tape"):
            return forced
        if self._solver_device_name == "cpu":
            return "tape"
        if self._solver_device_name in ("gpu", "tpu", "cuda", "rocm"):
            return "tape"
        try:
            from ._compat import jax as _jax

            backend = str(_jax.default_backend()).strip().lower() if _jax is not None else "cpu"
        except Exception:
            backend = "cpu"
        return "tape"

    def _use_scan_for_trial_solves(self) -> bool:
        """Use the scan loop for residual-only trial solves on accelerators."""
        if self._solver_device_name == "cpu":
            return False
        if self._solver_device_name in ("gpu", "tpu", "cuda", "rocm"):
            return True
        try:
            from ._compat import jax as _jax

            backend = str(_jax.default_backend()).strip().lower() if _jax is not None else "cpu"
        except Exception:
            backend = "cpu"
        return backend not in ("cpu", "")

    def _solver_device_context(self):
        if self._solver_device_name is None:
            return nullcontext()
        try:
            from ._compat import jax as _jax

            if _jax is None:
                return nullcontext()
            devices = _jax.devices(self._solver_device_name)
            if not devices:
                return nullcontext()
            return _jax.default_device(devices[0])
        except Exception:
            return nullcontext()

    def _move_to_solver_device(self, value):
        if self._solver_device_name is None:
            return value
        try:
            from ._compat import jax as _jax

            if _jax is None:
                return value
            device = _jax.devices(self._solver_device_name)[0]
            jax_array_type = _jax.Array
        except Exception:
            return value

        def _move(obj):
            if obj is None or isinstance(obj, (str, bytes, int, float, complex, bool)):
                return obj
            if isinstance(obj, (np.ndarray, jax_array_type)):
                return _jax.device_put(obj, device)
            if is_dataclass(obj) and not isinstance(obj, type):
                return replace(
                    obj,
                    **{field.name: _move(getattr(obj, field.name)) for field in fields(obj)},
                )
            if isinstance(obj, dict):
                return {key: _move(val) for key, val in obj.items()}
            if isinstance(obj, list):
                return [_move(item) for item in obj]
            if isinstance(obj, tuple):
                moved = tuple(_move(item) for item in obj)
                if hasattr(obj, "_fields"):
                    return type(obj)(*moved)
                return moved
            return obj

        return _move(value)

    def _run_in_solver_device_context(self, fn, *args, **kwargs):
        if self._solver_device_name is None or self._inside_solver_device_context:
            return fn(*args, **kwargs)
        with self._solver_device_context():
            self._inside_solver_device_context = True
            try:
                return fn(*args, **kwargs)
            finally:
                self._inside_solver_device_context = False

    def _read_last_array(self, array_key: str, scalar_key: str, default, cast):
        value = self._indata.get(array_key, None)
        if value is not None:
            if isinstance(value, (list, tuple)):
                if value:
                    return cast(value[-1])
            elif isinstance(value, np.ndarray):
                if int(value.size) > 0:
                    return cast(np.asarray(value).reshape(-1)[-1])
            else:
                return cast(value)
        return cast(self._indata.get(scalar_key, default))

    def _profile_add(self, name: str, dt: float) -> None:
        if not hasattr(self, "_profile"):
            self._profile = {}
        rec = self._profile.setdefault(name, {"count": 0, "wall_time_s": 0.0})
        rec["count"] = int(rec["count"]) + 1
        rec["wall_time_s"] = float(rec["wall_time_s"]) + float(dt)

    def _profile_dump(self) -> dict[str, dict[str, float | int]]:
        out: dict[str, dict[str, float | int]] = {}
        for name, rec in sorted(self._profile.items()):
            count = int(rec.get("count", 0))
            total = float(rec.get("wall_time_s", 0.0))
            out[name] = {
                "count": count,
                "wall_time_s": total,
                "mean_wall_time_s": total / count if count else 0.0,
            }
        return out

    def _exact_cache_key(self, params) -> bytes:
        return np.asarray(params, dtype=float).reshape(-1).tobytes()

    def _remember_exact_state(self, cache_key: bytes, state: VMECState) -> None:
        self._exact_state_cache = {cache_key: state}

    def _cached_exact_state(self, params):
        cache_key = self._exact_cache_key(params)
        if cache_key in self._exact_cache:
            state = self._exact_cache[cache_key][0]
            self._remember_exact_state(cache_key, state)
            self._profile_add("exact_cache_hit", 0.0)
            return state
        if cache_key in getattr(self, "_exact_state_cache", {}):
            self._profile_add("exact_state_cache_hit", 0.0)
            return self._exact_state_cache[cache_key]
        return None

    def _boundary_from_params(self, params):
        from ._compat import jnp as _jnp
        boundary = apply_boundary_params(
            self._boundary_input if self._boundary_input is not None else self._boundary,
            self._specs,
            _jnp.asarray(params, dtype=_jnp.float64),
        )
        if self._boundary_input is None:
            return boundary
        from .boundary import boundary_from_input_convention
        return boundary_from_input_convention(
            boundary,
            self._static.modes,
            lasym=bool(self._static.cfg.lasym),
            apply_m1_constraint=False,
        )

    def _boundary_input_from_params(self, params) -> BoundaryCoeffs:
        """Boundary coefficients in VMEC input convention for ``params``."""
        from ._compat import jnp as _jnp

        base_boundary = self._boundary_input if self._boundary_input is not None else self._boundary
        return apply_boundary_params(
            base_boundary,
            self._specs,
            _jnp.asarray(params, dtype=_jnp.float64),
        )

    def _indata_from_params(self, params) -> InData:
        """Return a VMEC namelist with boundary coefficients updated for ``params``."""
        boundary_input = self._boundary_input_from_params(params)
        indexed = {name: dict(values) for name, values in self._indata.indexed.items()}
        indexed.update(_indexed_boundary_maps_from_boundary(boundary_input, self._static.modes))
        return InData(
            scalars=dict(self._indata.scalars),
            indexed=indexed,
            source_path=self._indata.source_path,
        )

    def _base_params_vector(self) -> np.ndarray:
        """Return the reference free coefficients aligned with ``self._specs``."""
        boundary = self._boundary_input if self._boundary_input is not None else self._boundary
        base = np.empty(len(self._specs), dtype=float)
        for idx, spec in enumerate(self._specs):
            if spec.kind == "rc":
                base[idx] = float(boundary.R_cos[spec.index])
            elif spec.kind == "rs":
                base[idx] = float(boundary.R_sin[spec.index])
            elif spec.kind == "zc":
                base[idx] = float(boundary.Z_cos[spec.index])
            elif spec.kind == "zs":
                base[idx] = float(boundary.Z_sin[spec.index])
            else:  # pragma: no cover - guarded by boundary_param_specs
                raise ValueError(f"Unknown boundary parameter kind '{spec.kind}'")
        return base

    def _solve_forward(self, params, *, trial: bool = False):
        """Run a forward equilibrium solve."""
        if self._solver_device_name is not None and not self._inside_solver_device_context:
            return self._run_in_solver_device_context(self._solve_forward, params, trial=trial)
        from .solve import solve_fixed_boundary_residual_iter  # noqa: PLC0415
        t_total = time.perf_counter()
        boundary_now = self._boundary_from_params(params)
        state0 = initial_guess_from_boundary(
            self._static, boundary_now, self._indata, vmec_project=True
        )
        self._profile_add(
            "initial_guess_trial" if trial else "initial_guess_forward",
            time.perf_counter() - t_total,
        )
        t_solve = time.perf_counter()
        if trial:
            result = solve_fixed_boundary_residual_iter(
                state0, self._static,
                max_iter=self._trial_max_iter,
                ftol=self._trial_ftol,
                **self._trial_solver_kwargs,
            )
        else:
            result = solve_fixed_boundary_residual_iter(
                state0, self._static,
                max_iter=self._inner_max_iter,
                ftol=self._inner_ftol,
                **self._exact_solver_kwargs,
            )
        self._profile_add(
            "solve_forward_trial" if trial else "solve_forward_exact",
            time.perf_counter() - t_solve,
        )
        self._profile_add(
            "solve_forward_trial_total" if trial else "solve_forward_exact_total",
            time.perf_counter() - t_total,
        )
        return result.state

    def _scan_exact_helpers(self):
        """Return JIT-compiled scan residual/Jacobian helpers for accelerator solves."""
        if self._solver_device_name is not None and not self._inside_solver_device_context:
            return self._run_in_solver_device_context(self._scan_exact_helpers)
        from ._compat import jax, jnp as _jnp
        from .solve import solve_fixed_boundary_residual_iter

        cache_key = (
            int(len(self._specs)),
            int(self._layout.size),
            id(self._residuals_fn),
            int(self._inner_max_iter),
            float(self._inner_ftol),
            self._solver_device_name or "default",
        )
        helper_cache = self._scan_exact_helper_cache.get(cache_key)
        if helper_cache is not None:
            return helper_cache

        scan_solver_kwargs = dict(self._exact_solver_kwargs)
        scan_solver_kwargs.update(
            use_scan=True,
            light_history=True,
            resume_state_mode="none",
        )

        def _scan_state_from_params(p):
            boundary_now = self._boundary_from_params(p)
            state0 = initial_guess_from_boundary(
                self._static, boundary_now, self._indata, vmec_project=True
            )
            result = solve_fixed_boundary_residual_iter(
                state0,
                self._static,
                max_iter=self._inner_max_iter,
                ftol=self._inner_ftol,
                **scan_solver_kwargs,
            )
            return result.state

        def _scan_residuals_from_params(p):
            return _jnp.asarray(
                self._residuals_fn(_scan_state_from_params(p)),
                dtype=_jnp.float64,
            )

        @jax.jit
        def _residual_impl(p):
            return _scan_residuals_from_params(p)

        @jax.jit
        def _residual_and_jacobian_impl(p):
            residuals, linear = jax.linearize(_scan_residuals_from_params, p)
            directions = _jnp.eye(int(p.size), dtype=p.dtype)
            columns = jax.vmap(linear)(directions)
            return residuals, columns.T

        helper_cache = {
            "state": _scan_state_from_params,
            "residual": _residual_impl,
            "residual_and_jacobian": _residual_and_jacobian_impl,
        }
        self._scan_exact_helper_cache[cache_key] = helper_cache
        return helper_cache

    def _solve_scan_exact_state(self, params):
        """Run the scan accepted-point solve and remember the final state."""
        if self._solver_device_name is not None and not self._inside_solver_device_context:
            return self._run_in_solver_device_context(self._solve_scan_exact_state, params)
        from ._compat import jnp as _jnp

        cache_key = self._exact_cache_key(params)
        if cache_key in getattr(self, "_exact_state_cache", {}):
            self._profile_add("scan_exact_state_cache_hit", 0.0)
            return self._exact_state_cache[cache_key]
        helpers = self._scan_exact_helpers()
        t0 = time.perf_counter()
        state = helpers["state"](_jnp.asarray(params, dtype=_jnp.float64))
        self._remember_exact_state(cache_key, state)
        self._profile_add("scan_exact_state_solve", time.perf_counter() - t0)
        return state

    def _solve_exact_with_tape(self, params, *, return_payload: bool = False):
        """Run exact solve + build adjoint tape, with caching."""
        if self._solver_device_name is not None and not self._inside_solver_device_context:
            return self._run_in_solver_device_context(
                self._solve_exact_with_tape, params, return_payload=return_payload
            )
        from ._compat import jnp as _jnp
        from .discrete_adjoint import build_residual_checkpoint_tape_direct
        from .init_guess import extract_axis_override_from_state
        from .state import unpack_state

        cache_key = self._exact_cache_key(params)
        if cache_key in self._exact_cache:
            self._profile_add("exact_cache_hit", 0.0)
            state, payload = self._exact_cache[cache_key]
            self._remember_exact_state(cache_key, state)
            return (state, payload) if return_payload else state

        t_total = time.perf_counter()
        t_guess = time.perf_counter()
        boundary_now = self._boundary_from_params(params)
        state0 = initial_guess_from_boundary(
            self._static, boundary_now, self._indata, vmec_project=True
        )
        axis_override = extract_axis_override_from_state(state0, self._static)
        self._profile_add("initial_guess_exact", time.perf_counter() - t_guess)
        t_tape = time.perf_counter()
        tape = build_residual_checkpoint_tape_direct(
            state0,
            self._static,
            max_iter=self._inner_max_iter,
            solver_kwargs=self._exact_solver_kwargs,
            indata=self._indata,
            signgs=self._signgs,
            ftol=self._inner_ftol,
            step_size=self._step_size,
            light_history=True,
            store_trace=False,
            store_full_step_traces=False,
        )
        self._profile_add("exact_tape_build", time.perf_counter() - t_tape)
        t_unpack = time.perf_counter()
        state = unpack_state(
            _jnp.asarray(tape.final_packed_state, dtype=_jnp.float64), self._layout
        )
        payload = {"tape": tape, "axis_override": axis_override}
        self._exact_cache.clear()
        self._exact_cache[cache_key] = (state, payload)
        self._remember_exact_state(cache_key, state)
        self._profile_add("exact_unpack_cache", time.perf_counter() - t_unpack)
        self._profile_add("exact_solve_with_tape_total", time.perf_counter() - t_total)
        return (state, payload) if return_payload else state

    # ── public residual / Jacobian interface ──────────────────────────────────

    def residual_fun(self, params) -> np.ndarray:
        """Exact residual at *params* (builds adjoint tape, cached)."""
        if self._solver_device_name is not None and not self._inside_solver_device_context:
            return self._run_in_solver_device_context(self.residual_fun, params)
        if self._scan_exact_path == "scan":
            # Avoid compiling a second residual-only scan executable.  The exact
            # optimizer immediately needs the same accepted-point state for
            # history/cached residuals, so solve once and evaluate residuals from
            # that state.
            state = self._solve_scan_exact_state(params)
            t0 = time.perf_counter()
            out = np.asarray(self._residuals_fn(state), dtype=float)
            self._profile_add("scan_residual_eval_exact", time.perf_counter() - t0)
            return out
        state = self._solve_exact_with_tape(params)
        t_res = time.perf_counter()
        out = np.asarray(self._residuals_fn(state), dtype=float)
        self._profile_add("residual_eval_exact", time.perf_counter() - t_res)
        return out

    def forward_residual_fun(self, params) -> np.ndarray:
        """Relaxed residual for line-search trial evaluations."""
        if self._solver_device_name is not None and not self._inside_solver_device_context:
            return self._run_in_solver_device_context(self.forward_residual_fun, params)
        state = self._solve_forward(params, trial=True)
        t_res = time.perf_counter()
        out = np.asarray(self._residuals_fn(state), dtype=float)
        self._profile_add("residual_eval_trial", time.perf_counter() - t_res)
        return out

    def jacobian_fun(self, params) -> np.ndarray:
        """Exact discrete-adjoint Jacobian at *params*."""
        if self._solver_device_name is not None and not self._inside_solver_device_context:
            return self._run_in_solver_device_context(self.jacobian_fun, params)
        if self._scan_exact_path == "scan":
            from ._compat import jnp as _jnp

            helpers = self._scan_exact_helpers()
            t0 = time.perf_counter()
            residuals, jac = helpers["residual_and_jacobian"](
                _jnp.asarray(params, dtype=_jnp.float64)
            )
            self._last_jacobian_residual = np.asarray(residuals, dtype=float)
            self._solve_scan_exact_state(params)
            out = np.asarray(jac, dtype=float)
            self._profile_add("scan_jacobian_total", time.perf_counter() - t0)
            return out
        from ._compat import jax, jnp as _jnp
        from .discrete_adjoint import checkpoint_tape_state_jvp_columns
        from .init_guess import initial_guess_from_boundary as _ig
        from .state import pack_state, unpack_state

        t_total = time.perf_counter()
        params = _jnp.asarray(params, dtype=_jnp.float64)
        state, payload = self._solve_exact_with_tape(params, return_payload=True)
        tape = payload["tape"]
        axis_override = {
            key: _jnp.asarray(value, dtype=params.dtype)
            for key, value in payload["axis_override"].items()
        }
        packed_final = _jnp.asarray(pack_state(state), dtype=_jnp.float64)

        def _initial_state_packed(p, axis_override_arg):
            bdy = self._boundary_from_params(p)
            s0 = _ig(
                self._static, bdy, self._indata,
                vmec_project=True,
                axis_override=axis_override_arg,
            )
            return _jnp.asarray(pack_state(s0), dtype=_jnp.float64)

        def _residuals_from_packed(packed):
            return self._residuals_fn(unpack_state(packed, self._layout))

        directions = _jnp.eye(int(params.size), dtype=params.dtype)
        cache_key = (
            int(params.size),
            int(self._layout.size),
            id(self._residuals_fn),
        )
        helper_cache = self._discrete_jacobian_helper_cache.get(cache_key)
        if helper_cache is None:
            @jax.jit
            def _initial_tangent_columns(xf, directions, axis_override_arg):
                def _initial_state_at_axis(p):
                    return _initial_state_packed(p, axis_override_arg)

                _, initial_state_linear = jax.linearize(_initial_state_at_axis, xf)
                return jax.vmap(initial_state_linear)(directions)

            @jax.jit
            def _residual_tangent_columns(packed_state, packed_tangents):
                _, residual_linear = jax.linearize(_residuals_from_packed, packed_state)
                return jax.vmap(residual_linear)(packed_tangents)

            helper_cache = {
                "initial_tangent_columns": _initial_tangent_columns,
                "residual_tangent_columns": _residual_tangent_columns,
            }
            self._discrete_jacobian_helper_cache[cache_key] = helper_cache

        t_initial = time.perf_counter()
        initial_tangents = helper_cache["initial_tangent_columns"](params, directions, axis_override)
        self._profile_add("jacobian_initial_tangents", time.perf_counter() - t_initial)
        t_replay = time.perf_counter()
        final_tangents = checkpoint_tape_state_jvp_columns(
            tape=tape,
            static=self._static,
            initial_tangents=initial_tangents,
            rebuild_preconditioner=True,
        )
        self._profile_add("jacobian_tape_replay", time.perf_counter() - t_replay)
        t_res = time.perf_counter()
        columns = helper_cache["residual_tangent_columns"](packed_final, final_tangents)
        self._profile_add("jacobian_residual_tangents", time.perf_counter() - t_res)
        out = np.asarray(columns, dtype=float).T
        self._profile_add("jacobian_total", time.perf_counter() - t_total)
        return out

    def objective_and_gradient_fun(self, params) -> tuple[float, np.ndarray]:
        """Exact scalar objective and reverse-discrete-adjoint gradient.

        This computes the gradient of ``0.5 * ||residual_fun(params)||**2``
        by one reverse replay through the VMEC iteration tape, instead of
        replaying one forward tangent column per boundary parameter.  It is
        intentionally exposed separately from :meth:`jacobian_fun` so we can
        profile and validate the adjoint-gradient path before changing the
        default least-squares optimizer.
        """
        if self._solver_device_name is not None and not self._inside_solver_device_context:
            return self._run_in_solver_device_context(self.objective_and_gradient_fun, params)
        from ._compat import jax, jnp as _jnp
        from .discrete_adjoint import checkpoint_tape_state_vjp
        from .init_guess import initial_guess_from_boundary as _ig
        from .state import pack_state, unpack_state

        t_total = time.perf_counter()
        params = _jnp.asarray(params, dtype=_jnp.float64)
        state, payload = self._solve_exact_with_tape(params, return_payload=True)
        tape = payload["tape"]
        axis_override = {
            key: _jnp.asarray(value, dtype=params.dtype)
            for key, value in payload["axis_override"].items()
        }
        packed_final = _jnp.asarray(pack_state(state), dtype=_jnp.float64)

        def _residuals_from_packed(packed):
            return self._residuals_fn(unpack_state(packed, self._layout))

        t_res_vjp = time.perf_counter()
        residuals = self._residuals_fn(state)
        residuals = _jnp.asarray(residuals, dtype=_jnp.float64)
        cost = 0.5 * _jnp.vdot(residuals, residuals)
        state_cotangent_operator_factory = getattr(
            self._residuals_fn, "_state_cotangent_operator_from_packed", None
        )
        if state_cotangent_operator_factory is not None:
            final_cotangent = state_cotangent_operator_factory(packed_final, self._layout)(residuals)
        else:
            _, residual_vjp = jax.vjp(_residuals_from_packed, packed_final)
            final_cotangent = residual_vjp(residuals)[0]
        # Some state directions are intentionally inactive/gauge-null for a
        # given VMEC symmetry. Reverse-mode rules for sqrt/atan2-style geometry
        # kernels can return NaN cotangents in those unused directions even
        # though all parameter JVP columns are finite. Treat these inactive
        # cotangents as zero before replaying back to boundary parameters.
        final_cotangent = _jnp.nan_to_num(final_cotangent, nan=0.0, posinf=0.0, neginf=0.0)
        self._profile_add("gradient_residual_vjp", time.perf_counter() - t_res_vjp)

        t_replay = time.perf_counter()
        initial_cotangent = checkpoint_tape_state_vjp(
            tape=tape,
            static=self._static,
            final_cotangent=final_cotangent,
            rebuild_preconditioner=True,
        )
        initial_cotangent = _jnp.nan_to_num(initial_cotangent, nan=0.0, posinf=0.0, neginf=0.0)
        self._profile_add("gradient_tape_replay", time.perf_counter() - t_replay)

        def _initial_state_packed(p, axis_override_arg):
            bdy = self._boundary_from_params(p)
            s0 = _ig(
                self._static,
                bdy,
                self._indata,
                vmec_project=True,
                axis_override=axis_override_arg,
            )
            return _jnp.asarray(pack_state(s0), dtype=_jnp.float64)

        t_initial = time.perf_counter()
        _, initial_vjp = jax.vjp(
            lambda p: _initial_state_packed(p, axis_override),
            params,
        )
        grad = initial_vjp(_jnp.asarray(initial_cotangent, dtype=_jnp.float64))[0]
        self._profile_add("gradient_initial_vjp", time.perf_counter() - t_initial)
        self._profile_add("gradient_total", time.perf_counter() - t_total)
        return float(np.asarray(cost, dtype=float)), np.asarray(grad, dtype=float)

    def gradient_fun(self, params) -> np.ndarray:
        """Exact reverse-discrete-adjoint gradient of the scalar objective."""
        return self.objective_and_gradient_fun(params)[1]

    def residual_linear_operator(self, params):
        """Return a matrix-free exact residual Jacobian at ``params``.

        The returned :class:`scipy.sparse.linalg.LinearOperator` implements
        ``J @ v`` with one forward tangent replay and ``J.T @ w`` with one
        reverse replay through the same converged VMEC iteration tape.  This is
        the trust-region counterpart to :meth:`objective_and_gradient_fun` and
        avoids materializing the dense ``n_residuals x n_parameters`` Jacobian.
        """
        if self._solver_device_name is not None and not self._inside_solver_device_context:
            return self._run_in_solver_device_context(self.residual_linear_operator, params)
        try:
            from scipy.sparse.linalg import LinearOperator
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("residual_linear_operator requires scipy") from exc

        from ._compat import jax, jnp as _jnp
        from .discrete_adjoint import (
            checkpoint_tape_state_jvp,
            checkpoint_tape_state_jvp_columns,
            checkpoint_tape_state_vjp,
        )
        from .init_guess import initial_guess_from_boundary as _ig
        from .state import pack_state, unpack_state

        t_total = time.perf_counter()
        params = _jnp.asarray(params, dtype=_jnp.float64)
        state, payload = self._solve_exact_with_tape(params, return_payload=True)
        tape = payload["tape"]
        axis_override = {
            key: _jnp.asarray(value, dtype=params.dtype)
            for key, value in payload["axis_override"].items()
        }
        packed_final = _jnp.asarray(pack_state(state), dtype=_jnp.float64)

        def _initial_state_packed(p):
            bdy = self._boundary_from_params(p)
            s0 = _ig(
                self._static,
                bdy,
                self._indata,
                vmec_project=True,
                axis_override=axis_override,
            )
            return _jnp.asarray(pack_state(s0), dtype=_jnp.float64)

        def _residuals_from_packed(packed):
            return self._residuals_fn(unpack_state(packed, self._layout))

        t_setup = time.perf_counter()
        _, initial_linear = jax.linearize(_initial_state_packed, params)
        residuals, residual_linear = jax.linearize(_residuals_from_packed, packed_final)
        _, initial_vjp = jax.vjp(_initial_state_packed, params)
        state_cotangent_from_packed = getattr(
            self._residuals_fn, "_state_cotangent_from_packed", None
        )
        residual_cotangent_helper = None
        if state_cotangent_from_packed is not None:
            residual_cotangent_key = (
                "linear_operator_residual_cotangent",
                int(self._layout.size),
                int(residuals.size),
                id(self._residuals_fn),
            )
            helper_cache = self._discrete_jacobian_helper_cache.get(residual_cotangent_key)
            if helper_cache is None:
                @jax.jit
                def _residual_cotangent_helper(packed_state_arg, cotangent_arg):
                    return state_cotangent_from_packed(
                        packed_state_arg, self._layout, cotangent_arg
                    )

                helper_cache = {"residual_cotangent": _residual_cotangent_helper}
                self._discrete_jacobian_helper_cache[residual_cotangent_key] = helper_cache
            residual_cotangent_helper = helper_cache["residual_cotangent"]
        residual_vjp = None
        if state_cotangent_from_packed is None:
            _, residual_vjp = jax.vjp(_residuals_from_packed, packed_final)
        residuals_np = np.asarray(residuals, dtype=float)
        self._profile_add("linear_operator_setup", time.perf_counter() - t_setup)

        n_res = int(residuals_np.size)
        n_params = int(params.size)

        def _matvec(direction):
            t_mv = time.perf_counter()
            direction_j = _jnp.asarray(np.asarray(direction, dtype=float).reshape(-1), dtype=params.dtype)
            initial_tangent = initial_linear(direction_j)
            final_tangent = checkpoint_tape_state_jvp(
                tape=tape,
                static=self._static,
                initial_tangent=initial_tangent,
                rebuild_preconditioner=True,
            )
            out = residual_linear(final_tangent)
            self._profile_add("linear_operator_matvec", time.perf_counter() - t_mv)
            return np.asarray(out, dtype=float)

        def _matmat(directions):
            t_mm = time.perf_counter()
            directions_arr = np.asarray(directions, dtype=float)
            if directions_arr.ndim != 2:
                directions_arr = directions_arr.reshape((n_params, -1))
            directions_j = _jnp.asarray(directions_arr.T, dtype=params.dtype)
            initial_tangents = jax.vmap(initial_linear)(directions_j)
            final_tangents = checkpoint_tape_state_jvp_columns(
                tape=tape,
                static=self._static,
                initial_tangents=initial_tangents,
                rebuild_preconditioner=True,
            )
            out_columns = jax.vmap(residual_linear)(final_tangents)
            self._profile_add("linear_operator_matmat", time.perf_counter() - t_mm)
            return np.asarray(out_columns, dtype=float).T

        def _rmatvec(cotangent):
            t_rmv = time.perf_counter()
            cotangent_j = _jnp.asarray(np.asarray(cotangent, dtype=float).reshape(-1), dtype=_jnp.float64)
            t_res_cot = time.perf_counter()
            if residual_cotangent_helper is not None:
                final_cotangent = residual_cotangent_helper(packed_final, cotangent_j)
            else:
                final_cotangent = residual_vjp(cotangent_j)[0]
            self._profile_add("linear_operator_residual_vjp", time.perf_counter() - t_res_cot)
            final_cotangent = _jnp.nan_to_num(final_cotangent, nan=0.0, posinf=0.0, neginf=0.0)
            t_tape_vjp = time.perf_counter()
            initial_cotangent = checkpoint_tape_state_vjp(
                tape=tape,
                static=self._static,
                final_cotangent=final_cotangent,
                rebuild_preconditioner=True,
            )
            self._profile_add("linear_operator_tape_vjp", time.perf_counter() - t_tape_vjp)
            initial_cotangent = _jnp.nan_to_num(initial_cotangent, nan=0.0, posinf=0.0, neginf=0.0)
            t_initial_vjp = time.perf_counter()
            grad = initial_vjp(_jnp.asarray(initial_cotangent, dtype=_jnp.float64))[0]
            self._profile_add("linear_operator_initial_vjp", time.perf_counter() - t_initial_vjp)
            self._profile_add("linear_operator_rmatvec", time.perf_counter() - t_rmv)
            return np.asarray(grad, dtype=float)

        self._profile_add("linear_operator_total", time.perf_counter() - t_total)
        return LinearOperator(
            shape=(n_res, n_params),
            matvec=_matvec,
            rmatvec=_rmatvec,
            matmat=_matmat,
            dtype=np.dtype(float),
        )

    # ── tracked Jacobian for history + cache callbacks ────────────────────────

    def _jacobian_fun_tracked(self, params):
        self._last_jacobian_key[0] = self._exact_cache_key(params)
        jac = self.jacobian_fun(params)
        key = self._last_jacobian_key[0]
        if self._scan_exact_path == "scan" and key is not None and key in self._exact_state_cache:
            cached_state = self._exact_state_cache[key]
            res = (
                np.asarray(self._last_jacobian_residual, dtype=float)
                if self._last_jacobian_residual is not None
                else np.asarray(self._residuals_fn(cached_state), dtype=float)
            )
            cost = float(0.5 * np.dot(res, res))
            qs_total = self._qs_total_from_state(cached_state, res)
            from .wout import equilibrium_aspect_ratio_from_state
            aspect = float(np.asarray(
                equilibrium_aspect_ratio_from_state(state=cached_state, static=self._static)
            ))
            entry: dict = {
                "wall_time_s": time.perf_counter() - self._wall_t0,
                "cost": cost,
                "objective": 2.0 * cost,
                "qs_objective": qs_total,
                "aspect": aspect,
            }
            iota_fn = getattr(self, "_iota_fn", None)
            if iota_fn is not None:
                entry["iota"] = float(iota_fn(cached_state))
            self._history.append(entry)
        elif key is not None and key in self._exact_cache:
            cached_state, _ = self._exact_cache[key]
            res = np.asarray(self._residuals_fn(cached_state), dtype=float)
            cost = float(0.5 * np.dot(res, res))
            qs_total = self._qs_total_from_state(cached_state, res)
            from .wout import equilibrium_aspect_ratio_from_state
            aspect = float(np.asarray(
                equilibrium_aspect_ratio_from_state(state=cached_state, static=self._static)
            ))
            entry: dict = {
                "wall_time_s": time.perf_counter() - self._wall_t0,
                "cost": cost,
                "objective": 2.0 * cost,
                "qs_objective": qs_total,
                "aspect": aspect,
            }
            iota_fn = getattr(self, "_iota_fn", None)
            if iota_fn is not None:
                entry["iota"] = float(iota_fn(cached_state))
            self._history.append(entry)
        return jac

    def _exact_residual_after_jacobian(self):
        if self._scan_exact_path == "scan" and self._last_jacobian_residual is not None:
            return np.asarray(self._last_jacobian_residual, dtype=float)
        key = self._last_jacobian_key[0]
        if key is None or key not in self._exact_cache:
            return None
        cached_state, _ = self._exact_cache[key]
        return np.asarray(self._residuals_fn(cached_state), dtype=float)

    def _post_jacobian_clear(self, *, clear_compiled: bool = False):
        """Optionally release compiled replay helpers.

        Exact tapes and solved states are managed by the optimizer caches.  The
        replay/preconditioner JIT helpers are shape-keyed and LRU-bounded, so
        keeping them across accepted points avoids repeated CPU/GPU
        recompilation in long optimizations.  Full release is still available
        through clear_caches().
        """
        from .preconditioner_1d_jax import clear_preconditioner_jit_caches
        from .discrete_adjoint import clear_replay_scan_caches
        from .vmec_numpy_forces import clear_numpy_force_caches
        if clear_compiled:
            clear_replay_scan_caches()
            clear_preconditioner_jit_caches()
            clear_numpy_force_caches()

    # ── utilities ─────────────────────────────────────────────────────────────

    def clear_caches(self) -> None:
        """Release JIT and exact-solve caches."""
        self._exact_cache.clear()
        self._exact_state_cache.clear()
        self._last_jacobian_residual = None
        self._post_jacobian_clear(clear_compiled=True)

    def aspect_ratio(self, params) -> float:
        """Return the aspect ratio at *params* (uses exact solve cache)."""
        from .wout import equilibrium_aspect_ratio_from_state
        state = (
            self._solve_scan_exact_state(params)
            if self._scan_exact_path == "scan"
            else self._solve_exact_with_tape(params)
        )
        return float(np.asarray(
            equilibrium_aspect_ratio_from_state(state=state, static=self._static)
        ))

    def _qs_from_res(self, res: np.ndarray) -> float:
        """Sum of squared QS residuals only (excludes aspect and iota)."""
        if self._n_qs is not None:
            return float(np.dot(res[-self._n_qs:], res[-self._n_qs:]))
        start = max(0, min(int(self._n_non_qs), int(res.shape[0])))
        return float(np.dot(res[start:], res[start:]))

    def _qs_total_from_state(self, state: VMECState, res: np.ndarray | None = None) -> float:
        """QS-only objective from a solved state, with metadata-aware fallback."""
        if self._qs_total_from_state_fn is not None:
            return float(self._qs_total_from_state_fn(state))
        if res is None:
            res = np.asarray(self._residuals_fn(state), dtype=float)
        return self._qs_from_res(np.asarray(res, dtype=float))

    def quasisymmetry_objective(self, params) -> float:
        """Return the total QS objective at *params*."""
        state = (
            self._solve_scan_exact_state(params)
            if self._scan_exact_path == "scan"
            else self._solve_exact_with_tape(params)
        )
        res = np.asarray(self._residuals_fn(state), dtype=float)
        return self._qs_total_from_state(state, res)

    def save_wout(self, path, params=None, *, state: VMECState | None = None) -> None:
        """Write a wout NetCDF file for the equilibrium at *params*.

        Parameters
        ----------
        path:
            Output path for the ``.nc`` file.
        params:
            Boundary parameter vector (zeros = reference boundary). Optional
            when ``state`` is provided.
        state:
            Already-solved VMEC state to write. Passing this avoids rerunning
            the equilibrium solve and is the preferred path immediately after
            :meth:`run`.

        Notes
        -----
        Uses the exact-solve cache when *params* was previously evaluated.
        On a cache miss the trial solver (slightly relaxed tolerances) is used
        to avoid OOM after long optimization runs that have filled the JAX heap.
        """
        t0 = time.perf_counter()
        from .driver import FixedBoundaryRun
        from .driver import write_wout_from_fixed_boundary_run
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if state is None:
            if params is None:
                raise ValueError("save_wout requires either params or state")
            # Use cached state when available; fall back to trial solver (cheaper)
            # to avoid OOM when the JAX heap is saturated after a long run.
            state = self._cached_exact_state(params)
            if state is None:
                state = self._solve_forward(params, trial=True)
        run = FixedBoundaryRun(
            cfg=self._static.cfg,
            indata=self._indata,
            static=self._static,
            state=state,
            result=None,
            flux=self._flux,
            profiles={},
            signgs=self._signgs,
        )
        write_wout_from_fixed_boundary_run(
            str(path), run, include_fsq=False, fast_bcovar=True
        )
        self._profile_add("write_wout", time.perf_counter() - t0)
        print(f"  Wrote {path}")

    def save_input(self, path, params) -> None:
        """Write a VMEC ``input.*`` namelist for the boundary at ``params``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_indata(path, self._indata_from_params(params))
        print(f"  Wrote {path}")

    def save_history(self, path, result: dict) -> None:
        """Persist the optimisation history to a JSON file.

        Parameters
        ----------
        path:
            Output JSON path.
        result:
            Dict returned by :meth:`run`.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(result["_history_dump"], f, indent=2)
        print(f"  Wrote {path}")

    # ── main entry point ──────────────────────────────────────────────────────

    def run(
        self,
        params0,
        *,
        method: str = "gauss_newton",
        max_nfev: int = 10,
        ftol: float = 1e-3,
        gtol: float = 1e-3,
        xtol: float = 1e-3,
        x_scale=None,
        verbose: int = 1,
        iota_fn=None,
        target_iota: float | None = None,
        target_aspect: float | None = None,
        scipy_tr_solver: str | None = "lsmr",
        scipy_lsmr_maxiter: int | None = None,
    ) -> dict:
        """Run exact least-squares optimisation.

        Parameters
        ----------
        params0:
            Initial parameter vector (usually ``np.zeros(len(specs))``).
        method:
            Outer least-squares method. Supported values are ``"gauss_newton"``
            and ``"scipy"``. ``"scipy"`` uses ``scipy.optimize.least_squares``
            with the exact residual and discrete-adjoint Jacobian callbacks,
            which is more robust on some QA/QH examples.
            ``"scipy_matrix_free"`` uses the same SciPy trust-region solver
            with a matrix-free exact ``LinearOperator`` Jacobian.  It applies
            ``Jv`` and ``J.Tv`` products by replaying the converged VMEC tape
            without materializing the dense Jacobian. ``"lbfgs_adjoint"``
            minimizes the same scalar objective using one reverse discrete
            adjoint gradient per callback; it is experimental but scales much
            better with boundary-parameter count on mode-2/3 diagnostics.
        max_nfev:
            Maximum residual/Jacobian evaluations.
        ftol, gtol, xtol:
            Convergence tolerances.
        x_scale:
            Optional per-parameter scale vector.  When provided, parameter
            *i* is divided by ``x_scale[i]`` in the internal optimisation
            space.  Use :func:`create_x_scale` to build an exponential
            spectral-scaling vector.  ``None`` (default) treats all
            parameters uniformly.
        verbose:
            Verbosity (0 = silent, 1 = iteration table).
        iota_fn:
            Optional callable ``iota_fn(state) -> float`` that returns the
            mean rotational transform for a solved state.  When provided,
            the iota value is recorded in the per-iteration history under
            the key ``"iota"`` and saved by :meth:`save_history`.  Use this
            for QA runs where iota is a target quantity.
        target_iota:
            If provided alongside *iota_fn*, saved to the history dump
            under ``"target_iota"`` so plotting code can draw the target
            line.
        scipy_tr_solver:
            Trust-region linear solver passed through to
            :func:`scipy.optimize.least_squares` when ``method="scipy"``.
            Use ``"exact"`` for SciPy's dense SVD/QR-style path, ``"lsmr"``
            for the iterative path, or ``None`` for SciPy's default.
        scipy_lsmr_maxiter:
            Optional maximum number of LSMR iterations for SciPy's iterative
            trust-region linear solve.  This is primarily useful for the
            matrix-free path, where every LSMR iteration costs one or more
            exact ``Jv``/``J.Tv`` products.

        Returns
        -------
        dict
            Result dict from :func:`gauss_newton_least_squares` extended with
            ``_history_dump`` (the full per-iteration history suitable for
            :meth:`save_history`).
        """
        from .wout import equilibrium_aspect_ratio_from_state
        os.environ.setdefault("VMEC_JAX_DYNAMIC_REPLAY_BUCKET", "1024")

        self._history = []
        self._profile = {}
        self._wall_t0 = time.perf_counter()
        self._iota_fn = iota_fn  # stored so _jacobian_fun_tracked can use it

        params0_arr = np.asarray(params0, dtype=float)

        # ── initial evaluation ──────────────────────────────────────────────
        res0 = self.residual_fun(params0_arr)
        if self._scan_exact_path == "scan":
            state0 = self._solve_scan_exact_state(params0_arr)
        else:
            state0, _ = self._solve_exact_with_tape(params0_arr, return_payload=True)
        aspect0 = float(np.asarray(
            equilibrium_aspect_ratio_from_state(state=state0, static=self._static)
        ))
        cost0 = float(0.5 * np.dot(res0, res0))
        qs_total0 = self._qs_total_from_state(state0, res0)

        entry0: dict = {
            "wall_time_s": 0.0,
            "cost": cost0,
            "objective": 2.0 * cost0,
            "qs_objective": qs_total0,
            "aspect": aspect0,
        }
        if iota_fn is not None:
            entry0["iota"] = float(iota_fn(state0))
        self._history.append(entry0)

        # ── outer least-squares loop ────────────────────────────────────────
        t_start = time.perf_counter()
        method_key = str(method).strip().lower()
        if method_key == "gauss_newton":
            result = gauss_newton_least_squares(
                self.residual_fun,
                self._jacobian_fun_tracked,
                params0_arr,
                forward_residual_fun=self.forward_residual_fun,
                post_jacobian_callback=self._post_jacobian_clear,
                exact_residual_after_jacobian_fun=self._exact_residual_after_jacobian,
                max_nfev=max_nfev,
                ftol=ftol,
                gtol=gtol,
                xtol=xtol,
                x_scale=x_scale,
                verbose=verbose,
            )
        elif method_key in ("lbfgs", "lbfgs_adjoint"):
            try:
                from scipy.optimize import minimize as _scipy_minimize
            except Exception as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "method='lbfgs_adjoint' requires scipy.optimize.minimize"
                ) from exc

            scale = np.ones_like(params0_arr) if x_scale is None else np.asarray(x_scale, dtype=float)
            scale[scale == 0.0] = 1.0
            base_params = self._base_params_vector()
            y0 = (params0_arr + base_params) / scale
            last_history_key = [self._exact_cache_key(params0_arr)]

            def _record_history_from_cached_state(x, cost):
                key = self._exact_cache_key(x)
                if key == last_history_key[0] or key not in self._exact_cache:
                    return
                cached_state, _ = self._exact_cache[key]
                res = np.asarray(self._residuals_fn(cached_state), dtype=float)
                qs_total = self._qs_total_from_state(cached_state, res)
                aspect = float(np.asarray(
                    equilibrium_aspect_ratio_from_state(state=cached_state, static=self._static)
                ))
                entry: dict = {
                    "wall_time_s": time.perf_counter() - self._wall_t0,
                    "cost": float(cost),
                    "objective": float(2.0 * cost),
                    "qs_objective": qs_total,
                    "aspect": aspect,
                }
                if iota_fn is not None:
                    entry["iota"] = float(iota_fn(cached_state))
                self._history.append(entry)
                last_history_key[0] = key

            def _objective_and_gradient_y(y):
                x = np.asarray(y, dtype=float) * scale - base_params
                cost, grad_x = self.objective_and_gradient_fun(x)
                _record_history_from_cached_state(x, cost)
                return float(cost), np.asarray(grad_x, dtype=float) * scale

            minimize_result = _scipy_minimize(
                _objective_and_gradient_y,
                y0,
                jac=True,
                method="L-BFGS-B",
                options={
                    "maxiter": int(max_nfev),
                    "maxfun": int(max_nfev),
                    "ftol": float(ftol),
                    "gtol": float(gtol),
                    "disp": bool(int(verbose) > 0),
                },
            )
            x_result = np.asarray(minimize_result.x, dtype=float) * scale - base_params
            result = {
                "x": x_result,
                "cost": float(minimize_result.fun),
                "objective": float(2.0 * minimize_result.fun),
                "nfev": int(getattr(minimize_result, "nfev", 0)),
                "njev": int(getattr(minimize_result, "njev", 0)),
                "nit": int(getattr(minimize_result, "nit", 0)),
                "success": bool(minimize_result.success),
                "status": int(minimize_result.status),
                "message": str(minimize_result.message),
                "step_norm": 0.0,
                "x_prev": None,
                "cost_prev": None,
            }
        elif method_key in ("scipy_matrix_free", "matrix_free", "scipy_mf"):
            try:
                from scipy.optimize import least_squares as _scipy_least_squares
            except Exception as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "method='scipy_matrix_free' requires scipy.optimize.least_squares"
                ) from exc

            scale = np.ones_like(params0_arr) if x_scale is None else np.asarray(x_scale, dtype=float)
            scale[scale == 0.0] = 1.0
            base_params = self._base_params_vector()
            y0 = (params0_arr + base_params) / scale
            last_history_key = [self._exact_cache_key(params0_arr)]

            def _record_history_from_cached_state(x):
                key = self._exact_cache_key(x)
                if key == last_history_key[0] or key not in self._exact_cache:
                    return
                cached_state, _ = self._exact_cache[key]
                res = np.asarray(self._residuals_fn(cached_state), dtype=float)
                cost = float(0.5 * np.dot(res, res))
                qs_total = self._qs_total_from_state(cached_state, res)
                aspect = float(np.asarray(
                    equilibrium_aspect_ratio_from_state(state=cached_state, static=self._static)
                ))
                entry: dict = {
                    "wall_time_s": time.perf_counter() - self._wall_t0,
                    "cost": cost,
                    "objective": 2.0 * cost,
                    "qs_objective": qs_total,
                    "aspect": aspect,
                }
                if iota_fn is not None:
                    entry["iota"] = float(iota_fn(cached_state))
                self._history.append(entry)
                last_history_key[0] = key

            def _residuals_y(y):
                x = np.asarray(y, dtype=float) * scale - base_params
                cached_state = self._cached_exact_state(x)
                if cached_state is not None:
                    return np.asarray(self._residuals_fn(cached_state), dtype=float)
                return self.forward_residual_fun(x)

            def _jacobian_y(y):
                x = np.asarray(y, dtype=float) * scale - base_params
                op_x = self.residual_linear_operator(x)
                _record_history_from_cached_state(x)

                def _matvec(v):
                    v_arr = np.asarray(v, dtype=float).reshape(-1)
                    return op_x.matvec(v_arr * scale)

                def _matmat(v):
                    v_arr = np.asarray(v, dtype=float)
                    if v_arr.ndim != 2:
                        v_arr = v_arr.reshape((scale.size, -1))
                    return op_x.matmat(v_arr * scale[:, None])

                def _rmatvec(w):
                    w_arr = np.asarray(w, dtype=float).reshape(-1)
                    return op_x.rmatvec(w_arr) * scale

                try:
                    from scipy.sparse.linalg import LinearOperator
                except Exception as exc:  # pragma: no cover - optional dependency
                    raise ImportError("method='scipy_matrix_free' requires scipy") from exc

                return LinearOperator(
                    shape=op_x.shape,
                    matvec=_matvec,
                    matmat=_matmat,
                    rmatvec=_rmatvec,
                    dtype=np.dtype(float),
                )

            scipy_result = _scipy_least_squares(
                _residuals_y,
                y0,
                jac=_jacobian_y,
                method="trf",
                tr_solver="lsmr",
                tr_options=(
                    {"maxiter": int(scipy_lsmr_maxiter)}
                    if scipy_lsmr_maxiter is not None
                    else None
                ),
                max_nfev=max_nfev,
                ftol=ftol,
                gtol=gtol,
                xtol=xtol,
                verbose=2 if int(verbose) > 0 else 0,
            )
            x_result = np.asarray(scipy_result.x, dtype=float) * scale - base_params
            result = {
                "x": x_result,
                "cost": float(scipy_result.cost),
                "objective": float(2.0 * scipy_result.cost),
                "nfev": int(scipy_result.nfev),
                "njev": 0 if scipy_result.njev is None else int(scipy_result.njev),
                "nit": 0,
                "success": bool(scipy_result.success),
                "status": int(scipy_result.status),
                "message": str(scipy_result.message),
                "step_norm": 0.0,
                "x_prev": None,
                "cost_prev": None,
            }
        elif method_key == "scipy":
            try:
                from scipy.optimize import least_squares as _scipy_least_squares
            except Exception as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "method='scipy' requires scipy.optimize.least_squares"
                ) from exc

            scale = np.ones_like(params0_arr) if x_scale is None else np.asarray(x_scale, dtype=float)
            scale[scale == 0.0] = 1.0
            base_params = self._base_params_vector()
            y0 = (params0_arr + base_params) / scale

            def _forward_residual_exact(x):
                # For SciPy trial-point residual callbacks, use the trial solver
                # configuration even when the user requests deck-controlled
                # budgets via TRIAL_MAX_ITER=0 / TRIAL_FTOL=0. This keeps the
                # solve on the lighter forward path without building an exact
                # adjoint tape for every trust-region trial point, which
                # materially lowers RSS on QA/QH exact-optimization runs.
                return self.forward_residual_fun(x)

            def _residuals_y(y):
                x = np.asarray(y, dtype=float) * scale - base_params
                cached_state = self._cached_exact_state(x)
                if cached_state is not None:
                    return np.asarray(self._residuals_fn(cached_state), dtype=float)
                # Residual-only callbacks do not need an adjoint tape. Building one
                # for every SciPy trial point bloats memory badly on converged QA/QH
                # runs. Keep the Jacobian exact, but evaluate residuals through the
                # converged forward solve only.
                return _forward_residual_exact(x)

            def _jacobian_y(y):
                x = np.asarray(y, dtype=float) * scale - base_params
                jac = np.asarray(self._jacobian_fun_tracked(x), dtype=float) * scale[None, :]
                # SciPy residual callbacks above no longer consume the exact-tape cache.
                # Drop the retained tape immediately after the Jacobian/history entry is
                # materialized, otherwise later converged QA iterations keep a multi-GB
                # exact tape alive between callbacks and get killed by RSS.
                self._exact_cache.clear()
                return jac

            scipy_result = _scipy_least_squares(
                _residuals_y,
                y0,
                jac=_jacobian_y,
                method="trf",
                # The exact-optimizer Jacobians are extremely tall
                # (tens of thousands of residuals, tens of parameters).
                # `lsmr` gives materially smaller direct-start trial steps on
                # the QA/QH mode-3 cases than the default dense path, which
                # avoids wasting minutes in hopeless VMEC trial solves.
                tr_solver=scipy_tr_solver,
                tr_options=(
                    {"maxiter": int(scipy_lsmr_maxiter)}
                    if scipy_lsmr_maxiter is not None and scipy_tr_solver == "lsmr"
                    else None
                ),
                max_nfev=max_nfev,
                ftol=ftol,
                gtol=gtol,
                xtol=xtol,
                verbose=2 if int(verbose) > 0 else 0,
            )
            x_result = np.asarray(scipy_result.x, dtype=float) * scale - base_params
            result = {
                "x": x_result,
                "cost": float(scipy_result.cost),
                "objective": float(2.0 * scipy_result.cost),
                "nfev": int(scipy_result.nfev),
                "njev": 0 if scipy_result.njev is None else int(scipy_result.njev),
                "nit": 0,
                "success": bool(scipy_result.success),
                "status": int(scipy_result.status),
                "message": str(scipy_result.message),
                "step_norm": 0.0,
                "x_prev": None,
                "cost_prev": None,
            }
        else:
            raise ValueError(f"Unknown optimization method '{method}'.")
        t_total = time.perf_counter() - t_start
        self._post_jacobian_clear()

        # ── final evaluation ────────────────────────────────────────────────
        # Use the exact cache when available (avoids a fresh full VMEC solve
        # that can OOM after a long optimization session).  Fall back to the
        # trial (cheaper) solver when the cache doesn't hold result["x"].
        state_final = self._cached_exact_state(result["x"])
        if state_final is None:
            try:
                state_final = (
                    self._solve_scan_exact_state(result["x"])
                    if self._scan_exact_path == "scan"
                    else self._solve_exact_with_tape(result["x"])
                )
            except Exception:
                state_final = self._solve_forward(result["x"], trial=True)

        res_final = np.asarray(self._residuals_fn(state_final), dtype=float)
        aspect_final = float(np.asarray(
            equilibrium_aspect_ratio_from_state(state=state_final, static=self._static)
        ))
        cost_final = float(0.5 * np.dot(res_final, res_final))
        qs_total_final = self._qs_total_from_state(state_final, res_final)

        entry_final: dict = {
            "wall_time_s": t_total,
            "cost": cost_final,
            "objective": 2.0 * cost_final,
            "qs_objective": qs_total_final,
            "aspect": aspect_final,
        }
        if iota_fn is not None:
            entry_final["iota"] = float(iota_fn(state_final))
        self._history.append(entry_final)

        # ── assemble history dump ───────────────────────────────────────────
        history_dump: dict = {
            "label": "Optimisation",
            "max_nfev": max_nfev,
            "ftol": ftol,
            "gtol": gtol,
            "xtol": xtol,
            "method": method_key,
            "exact_path": self._scan_exact_path,
            "scipy_tr_solver": (
                scipy_tr_solver
                if method_key == "scipy"
                else "lsmr"
                if method_key in ("scipy_matrix_free", "matrix_free", "scipy_mf")
                else None
            ),
            "scipy_lsmr_maxiter": (
                None if scipy_lsmr_maxiter is None else int(scipy_lsmr_maxiter)
            ),
            "solver_device": self._solver_device_name or "default",
            "total_wall_time_s": t_total,
            "nfev": result["nfev"],
            "njev": result["njev"],
            "success": result["success"],
            "message": result["message"],
            "objective_initial": 2.0 * cost0,
            "objective_final": 2.0 * cost_final,
            "qs_initial": qs_total0,
            "qs_final": qs_total_final,
            "aspect_initial": aspect0,
            "aspect_final": aspect_final,
            "history": self._history,
            "profile": self._profile_dump(),
        }
        if iota_fn is not None:
            history_dump["iota_initial"] = float(entry0["iota"])
            history_dump["iota_final"] = float(entry_final["iota"])
        if target_iota is not None:
            history_dump["target_iota"] = float(target_iota)
        if target_aspect is not None:
            history_dump["target_aspect"] = float(target_aspect)

        # Private, non-serializable convenience payload for scripts that want
        # to write wout files without rerunning the VMEC solve immediately after
        # optimization. save_history() only persists `_history_dump`.
        result["_state_initial"] = state0
        result["_state_final"] = state_final
        result["_profile"] = self._profile_dump()
        result["_history_dump"] = history_dump
        return result
