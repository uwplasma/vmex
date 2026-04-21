"""Optimization-oriented helpers for vmec_jax workflows."""

from __future__ import annotations

from dataclasses import dataclass
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
    verbose:
        Verbosity level (0 = silent, 1 = iteration table).
    """
    x = np.asarray(x0, dtype=float).copy()
    scale = np.ones_like(x) if x_scale is None else np.asarray(x_scale, dtype=float).copy()
    scale[scale == 0.0] = 1.0
    trial_residual_fun = residual_fun if forward_residual_fun is None else forward_residual_fun

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

        alpha = min(max(alpha_prev, 1.0 / 128.0), 1.0)
        accepted = False
        cost_trial = math.inf
        residual_trial = None
        x_trial = None
        for _ in range(8):
            x_candidate = x + alpha * step
            residual_candidate = np.asarray(trial_residual_fun(x_candidate), dtype=float).reshape(-1)
            nfev += 1
            cost_candidate = 0.5 * float(np.dot(residual_candidate, residual_candidate))
            if np.isfinite(cost_candidate) and cost_candidate < cost:
                x_trial = x_candidate
                residual_trial = residual_candidate
                cost_trial = cost_candidate
                accepted = True
                break
            alpha *= 0.5
            if nfev >= int(max_nfev):
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
        step_norm_trial = float(np.linalg.norm(alpha * step))
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
        alpha_prev = alpha
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

    def residuals_from_state(state: VMECState) -> jnp.ndarray:
        aspect = equilibrium_aspect_ratio_from_state(state=state, static=static)
        qs = quasisymmetry_ratio_residual_from_state(
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
        aspect_residual = jnp.asarray([float(aspect_weight) * (aspect - target_aspect)],
                                      dtype=jnp.float64)
        qs_residual = jnp.asarray(qs["residuals1d"], dtype=jnp.float64) * float(qs_weight)
        return jnp.concatenate([aspect_residual, qs_residual])

    residuals_from_state._n_qs = int(len(surfaces))

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

        qs = quasisymmetry_ratio_residual_from_state(
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
        parts.append(jnp.asarray(qs["residuals1d"], dtype=jnp.float64) * float(qs_weight))

        return jnp.concatenate(parts)

    residuals_from_state._n_qs = int(len(surfaces))
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
        Accepted-point VMEC residual solve budget.
    trial_max_iter, trial_ftol:
        Trial-point VMEC residual solve budget used by the relaxed forward
        callback inside the optimizer.

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
    ) -> None:
        self._static = static
        self._indata = indata
        self._boundary = boundary
        self._boundary_input = boundary_input
        self._specs = list(specs)
        self._residuals_fn = residuals_fn
        # Number of QS residuals (last _n_qs entries of the residual vector).
        # Stored so quasisymmetry_objective correctly excludes aspect/iota entries.
        self._n_qs: int | None = getattr(residuals_fn, "_n_qs", None)

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
        if inner_max_iter is not None:
            self._inner_max_iter = int(inner_max_iter)
        if inner_ftol is not None:
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
            jit_forces=True,
            use_scan=False,
            light_history=True,
            resume_state_mode="full",
        )
        self._exact_solver_kwargs = dict(_base)
        self._trial_solver_kwargs = dict(_base, jit_forces=False)
        self._trial_max_iter = min(self._inner_max_iter, 800) if trial_max_iter is None else int(trial_max_iter)
        self._trial_ftol = max(self._inner_ftol, 1e-10) if trial_ftol is None else float(trial_ftol)

        # Single-entry cache: avoids building the tape twice at the same x.
        self._exact_cache: dict = {}

        # History collected during optimisation.
        self._history: list[dict] = []
        self._wall_t0: float = 0.0
        self._last_jacobian_key: list = [None]
        self._iota_fn = None  # set by run() when iota tracking is requested

    # ── private helpers ───────────────────────────────────────────────────────

    def _read_last_array(self, array_key: str, scalar_key: str, default, cast):
        value = self._indata.get(array_key, None)
        if isinstance(value, list) and value:
            return cast(value[-1])
        return cast(self._indata.get(scalar_key, default))

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

    def _solve_forward(self, params, *, trial: bool = False):
        """Run a forward equilibrium solve."""
        from .solve import solve_fixed_boundary_residual_iter  # noqa: PLC0415
        boundary_now = self._boundary_from_params(params)
        state0 = initial_guess_from_boundary(
            self._static, boundary_now, self._indata, vmec_project=True
        )
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
        return result.state

    def _solve_exact_with_tape(self, params, *, return_payload: bool = False):
        """Run exact solve + build adjoint tape, with caching."""
        from ._compat import jnp as _jnp
        from .discrete_adjoint import build_residual_checkpoint_tape_direct
        from .init_guess import extract_axis_override_from_state
        from .state import unpack_state

        params_arr = np.asarray(params, dtype=float)
        cache_key = params_arr.tobytes()
        if cache_key in self._exact_cache:
            state, payload = self._exact_cache[cache_key]
            return (state, payload) if return_payload else state

        boundary_now = self._boundary_from_params(params)
        state0 = initial_guess_from_boundary(
            self._static, boundary_now, self._indata, vmec_project=True
        )
        axis_override = extract_axis_override_from_state(state0, self._static)
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
        state = unpack_state(
            _jnp.asarray(tape.final_packed_state, dtype=_jnp.float64), self._layout
        )
        payload = {"tape": tape, "axis_override": axis_override}
        self._exact_cache.clear()
        self._exact_cache[cache_key] = (state, payload)
        return (state, payload) if return_payload else state

    # ── public residual / Jacobian interface ──────────────────────────────────

    def residual_fun(self, params) -> np.ndarray:
        """Exact residual at *params* (builds adjoint tape, cached)."""
        state = self._solve_exact_with_tape(params)
        return np.asarray(self._residuals_fn(state), dtype=float)

    def forward_residual_fun(self, params) -> np.ndarray:
        """Relaxed residual for line-search trial evaluations."""
        state = self._solve_forward(params, trial=True)
        return np.asarray(self._residuals_fn(state), dtype=float)

    def jacobian_fun(self, params) -> np.ndarray:
        """Exact discrete-adjoint Jacobian at *params*."""
        from ._compat import jax, jnp as _jnp
        from .discrete_adjoint import checkpoint_tape_state_jvp_columns
        from .init_guess import initial_guess_from_boundary as _ig
        from .state import pack_state, unpack_state

        params = _jnp.asarray(params, dtype=_jnp.float64)
        state, payload = self._solve_exact_with_tape(params, return_payload=True)
        packed_final = _jnp.asarray(pack_state(state), dtype=_jnp.float64)

        def _initial_state_packed(p):
            bdy = self._boundary_from_params(p)
            s0 = _ig(
                self._static, bdy, self._indata,
                vmec_project=True,
                axis_override=payload["axis_override"],
            )
            return _jnp.asarray(pack_state(s0), dtype=_jnp.float64)

        def _residuals_from_packed(packed):
            return self._residuals_fn(unpack_state(packed, self._layout))

        directions = _jnp.eye(int(params.size), dtype=params.dtype)
        _, initial_linear = jax.linearize(_initial_state_packed, params)
        initial_tangents = jax.vmap(initial_linear)(directions)
        final_tangents = checkpoint_tape_state_jvp_columns(
            tape=payload["tape"],
            static=self._static,
            initial_tangents=initial_tangents,
            rebuild_preconditioner=True,
        )
        _, residual_linear = jax.linearize(_residuals_from_packed, packed_final)
        columns = jax.vmap(residual_linear)(final_tangents)
        return np.asarray(columns, dtype=float).T

    # ── tracked Jacobian for history + cache callbacks ────────────────────────

    def _jacobian_fun_tracked(self, params):
        self._last_jacobian_key[0] = np.asarray(params, dtype=float).tobytes()
        jac = self.jacobian_fun(params)
        key = self._last_jacobian_key[0]
        if key is not None and key in self._exact_cache:
            cached_state, _ = self._exact_cache[key]
            res = np.asarray(self._residuals_fn(cached_state), dtype=float)
            cost = float(0.5 * np.dot(res, res))
            qs_total = self._qs_from_res(res)
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
        key = self._last_jacobian_key[0]
        if key is None or key not in self._exact_cache:
            return None
        cached_state, _ = self._exact_cache[key]
        return np.asarray(self._residuals_fn(cached_state), dtype=float)

    def _post_jacobian_clear(self):
        from .preconditioner_1d_jax import clear_preconditioner_jit_caches
        from .discrete_adjoint import clear_replay_scan_caches
        clear_replay_scan_caches()
        clear_preconditioner_jit_caches()

    # ── utilities ─────────────────────────────────────────────────────────────

    def clear_caches(self) -> None:
        """Release JIT and exact-solve caches."""
        self._exact_cache.clear()
        self._post_jacobian_clear()

    def aspect_ratio(self, params) -> float:
        """Return the aspect ratio at *params* (uses exact solve cache)."""
        from .wout import equilibrium_aspect_ratio_from_state
        state = self._solve_exact_with_tape(params)
        return float(np.asarray(
            equilibrium_aspect_ratio_from_state(state=state, static=self._static)
        ))

    def _qs_from_res(self, res: np.ndarray) -> float:
        """Sum of squared QS residuals only (excludes aspect and iota)."""
        if self._n_qs is not None:
            return float(np.dot(res[-self._n_qs:], res[-self._n_qs:]))
        # Fallback for externally-supplied residuals_fn without _n_qs tag
        return float(np.dot(res[1:], res[1:]))

    def quasisymmetry_objective(self, params) -> float:
        """Return the total QS objective at *params*."""
        res = np.asarray(self.residual_fun(params), dtype=float)
        return self._qs_from_res(res)

    def save_wout(self, path, params) -> None:
        """Write a wout NetCDF file for the equilibrium at *params*.

        Parameters
        ----------
        path:
            Output path for the ``.nc`` file.
        params:
            Boundary parameter vector (zeros = reference boundary).

        Notes
        -----
        Uses the exact-solve cache when *params* was previously evaluated.
        On a cache miss the trial solver (slightly relaxed tolerances) is used
        to avoid OOM after long optimization runs that have filled the JAX heap.
        """
        from .driver import FixedBoundaryRun
        from .driver import write_wout_from_fixed_boundary_run
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Use cached state when available; fall back to trial solver (cheaper)
        # to avoid OOM when the JAX heap is saturated after a long run.
        cache_key = np.asarray(params, dtype=float).tobytes()
        if cache_key in self._exact_cache:
            state = self._exact_cache[cache_key][0]
        else:
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

        Returns
        -------
        dict
            Result dict from :func:`gauss_newton_least_squares` extended with
            ``_history_dump`` (the full per-iteration history suitable for
            :meth:`save_history`).
        """
        from .wout import equilibrium_aspect_ratio_from_state
        from .quasisymmetry import quasisymmetry_ratio_residual_from_state

        os.environ.setdefault("VMEC_JAX_DYNAMIC_REPLAY_BUCKET", "1024")

        self._history = []
        self._wall_t0 = time.perf_counter()
        self._iota_fn = iota_fn  # stored so _jacobian_fun_tracked can use it

        params0_arr = np.asarray(params0, dtype=float)

        # ── initial evaluation ──────────────────────────────────────────────
        res0 = self.residual_fun(params0_arr)
        state0, _ = self._solve_exact_with_tape(params0_arr, return_payload=True)
        aspect0 = float(np.asarray(
            equilibrium_aspect_ratio_from_state(state=state0, static=self._static)
        ))
        cost0 = float(0.5 * np.dot(res0, res0))
        qs_total0 = self._qs_from_res(res0)

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
        elif method_key == "scipy":
            try:
                from scipy.optimize import least_squares as _scipy_least_squares
            except Exception as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "method='scipy' requires scipy.optimize.least_squares"
                ) from exc

            scale = np.ones_like(params0_arr) if x_scale is None else np.asarray(x_scale, dtype=float)
            scale[scale == 0.0] = 1.0
            scipy_result = _scipy_least_squares(
                lambda x: np.asarray(self.residual_fun(x), dtype=float),
                params0_arr,
                jac=lambda x: np.asarray(self._jacobian_fun_tracked(x), dtype=float),
                method="trf",
                x_scale=scale,
                max_nfev=max_nfev,
                ftol=ftol,
                gtol=gtol,
                xtol=xtol,
                verbose=2 if int(verbose) > 0 else 0,
            )
            result = {
                "x": np.asarray(scipy_result.x, dtype=float),
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
        final_key = np.asarray(result["x"], dtype=float).tobytes()
        if final_key in self._exact_cache:
            state_final = self._exact_cache[final_key][0]
        else:
            state_final = self._solve_forward(result["x"], trial=True)

        res_final = np.asarray(self._residuals_fn(state_final), dtype=float)
        aspect_final = float(np.asarray(
            equilibrium_aspect_ratio_from_state(state=state_final, static=self._static)
        ))
        cost_final = float(0.5 * np.dot(res_final, res_final))
        qs_total_final = self._qs_from_res(res_final)

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
        }
        if target_iota is not None:
            history_dump["target_iota"] = float(target_iota)

        result["_history_dump"] = history_dump
        return result
