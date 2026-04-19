"""Optimization-oriented helpers for vmec_jax workflows."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

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
