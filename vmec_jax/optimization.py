"""Optimization-oriented helpers for vmec_jax workflows."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import ExitStack, nullcontext
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
from .profiles import eval_profiles
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


_EXACT_TAPE_BUILD_TIMING_PROFILE_NAMES = (
    ("tape_solve_call_s", "exact_tape_build_solve_call"),
    ("tape_final_state_pack_s", "exact_tape_build_final_state_pack"),
    ("tape_step_trace_extract_s", "exact_tape_build_step_trace_extract"),
    ("tape_dynamic_payload_build_s", "exact_tape_build_dynamic_payload"),
    ("tape_trace_stack_s", "exact_tape_build_trace_stack"),
)


def _linear_operator_vector_arg(value, *, size: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if int(arr.size) != int(size):
        raise ValueError(f"{name} expected {int(size)} entries, got {int(arr.size)}.")
    return arr


def _linear_operator_matrix_arg(value, *, rows: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    rows = int(rows)
    if arr.ndim != 2:
        if rows <= 0:
            if arr.size != 0:
                raise ValueError(f"{name} expected 0 rows, got {int(arr.size)} entries.")
            return arr.reshape((0, 0))
        if int(arr.size) % rows != 0:
            raise ValueError(f"{name} with {int(arr.size)} entries cannot be reshaped to {rows} rows.")
        arr = arr.reshape((rows, -1))
    if int(arr.shape[0]) != rows:
        raise ValueError(f"{name} expected {rows} rows, got {int(arr.shape[0])}.")
    return arr


def _skip_exhausted_gauss_newton_jacobian() -> bool:
    flag = os.getenv("VMEC_JAX_OPT_SKIP_EXHAUSTED_GN_JACOBIAN", "").strip().lower()
    return flag in ("1", "true", "yes", "on")


def _optimizer_backend_name(solver_device_name: str | None) -> str:
    """Return the active optimizer backend name without changing device policy."""

    backend = str(solver_device_name or "").strip().lower()
    if backend:
        return backend
    try:
        from ._compat import jax as _jax

        return str(_jax.default_backend()).strip().lower() if _jax is not None else "cpu"
    except Exception:
        return "cpu"


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
    need_mpol = max(5, max_mode + 2)  # VMEC mpol = max_m + 1; add extra headroom
    need_ntor = max(5, max_mode + 2)

    if need_mpol <= cur_mpol and need_ntor <= cur_ntor:
        return indata, static, boundary  # nothing to do

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


def truncate_indata_boundary_modes(indata, *, max_mode: int | None):
    """Return a copy of ``indata`` with boundary modes above ``max_mode`` zeroed.

    VMEC inputs can contain non-zero harmonics outside the active optimization
    space.  Parameter specs only decide which coefficients are free; they do
    not alter the fixed coefficients already present in the input.  Use this
    helper when a ``max_mode=N`` optimization should start from the boundary
    projected onto ``max(abs(m), abs(n)) <= N`` rather than keeping higher
    harmonics fixed in the background.
    """
    from .namelist import InData

    if max_mode is None:
        return indata
    limit = int(max_mode)
    boundary_names = {"RBC", "RBS", "ZBC", "ZBS"}
    indexed = {}
    for name, values in indata.indexed.items():
        upper = str(name).upper()
        copied = dict(values)
        if upper in boundary_names:
            copied = {
                tuple(key): float(value)
                for key, value in copied.items()
                if len(tuple(key)) >= 2 and max(abs(int(tuple(key)[0])), abs(int(tuple(key)[1]))) <= limit
            }
        indexed[name] = copied
    return InData(
        scalars=dict(indata.scalars),
        indexed=indexed,
        source_path=indata.source_path,
    )


def smooth_min_abs_iota_residual(
    iota,
    minimum: float,
    *,
    softness: float = 1.0e-3,
    abs_epsilon: float = 1.0e-12,
):
    """Smooth residual for the differentiable constraint ``abs(iota) >= minimum``.

    The returned residual is approximately zero when ``abs(iota)`` is above the
    requested lower bound and approximately ``minimum - abs(iota)`` below it.
    A softplus shortfall avoids the non-differentiable kink of a hard hinge,
    which is important when this term is used inside exact JAX Jacobians.
    """

    iota = jnp.asarray(iota, dtype=jnp.float64)
    minimum = jnp.asarray(minimum, dtype=iota.dtype)
    softness = jnp.maximum(
        jnp.asarray(softness, dtype=iota.dtype),
        jnp.asarray(1.0e-15, dtype=iota.dtype),
    )
    abs_epsilon = jnp.asarray(abs_epsilon, dtype=iota.dtype)
    smooth_abs_iota = jnp.sqrt(iota * iota + abs_epsilon * abs_epsilon)
    shortfall = minimum - smooth_abs_iota
    return softness * jnp.logaddexp(jnp.asarray(0.0, dtype=iota.dtype), shortfall / softness)


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
    source_vals = {spec.name: float(value) for spec, value in zip(source_specs, np.asarray(source_params, dtype=float))}
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


def _apply_boundary_params_numpy(
    boundary: BoundaryCoeffs,
    specs: Sequence[BoundaryParamSpec],
    params: np.ndarray,
) -> BoundaryCoeffs:
    """Apply parameter updates on the host for branch/cache-key logic."""
    params = np.asarray(params, dtype=float).reshape(-1)
    r_cos = np.asarray(boundary.R_cos, dtype=float).copy()
    r_sin = np.asarray(boundary.R_sin, dtype=float).copy()
    z_cos = np.asarray(boundary.Z_cos, dtype=float).copy()
    z_sin = np.asarray(boundary.Z_sin, dtype=float).copy()

    for idx, spec in enumerate(specs):
        if idx >= int(params.size):
            break
        if spec.kind == "rc":
            r_cos[spec.index] += float(params[idx])
        elif spec.kind == "rs":
            r_sin[spec.index] += float(params[idx])
        elif spec.kind == "zc":
            z_cos[spec.index] += float(params[idx])
        elif spec.kind == "zs":
            z_sin[spec.index] += float(params[idx])
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
    pressure = _pressure_profile_for_static(indata, static)
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


def _pressure_profile_for_static(indata, static: VMECStatic):
    """Evaluate the VMEC pressure profile on the optimization radial mesh."""
    prof = eval_profiles(indata, jnp.asarray(static.s))
    return jnp.asarray(
        prof.get("pressure", jnp.zeros_like(jnp.asarray(static.s))),
        dtype=jnp.asarray(static.s).dtype,
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
        (1e-6, 1e-4, 1e-2, 1.0, 100.0) if damping_factors is None else tuple(float(value) for value in damping_factors)
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
        if cost == 0.0:
            success = True
            message = "`gtol` termination condition is satisfied."
            accepted_cost = cost
            accepted_step_norm = 0.0
            if verbose:
                print(f"{iteration:12d}{nfev:16d}{cost:13.4e}{0.0:18.2e}{0.0:16.2e}{0.0:16.2e}")
            break
        if nfev >= int(max_nfev) and _skip_exhausted_gauss_newton_jacobian():
            accepted_cost = cost
            accepted_step_norm = 0.0
            if verbose:
                print(f"{iteration:12d}{nfev:16d}{cost:13.4e}{0.0:18.2e}{0.0:16.2e}{float('nan'):16.2e}")
            break

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
                print(f"{iteration:12d}{nfev:16d}{cost:13.4e}{0.0:18.2e}{0.0:16.2e}{optimality:16.2e}")
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
                print(f"{iteration:12d}{nfev:16d}{cost:13.4e}{0.0:18.2e}{step_norm:16.2e}{optimality:16.2e}")
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
                print(f"{iteration:12d}{nfev:16d}{cost:13.4e}{0.0:18.2e}{0.0:16.2e}{optimality:16.2e}")
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
    from .modes import nyquist_mode_table_from_grid
    from .quasisymmetry import (
        _quasisymmetry_angle_cache,
        quasisymmetry_ratio_residual_from_state,
    )
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
    pressure = _pressure_profile_for_static(indata, static)
    nyq_modes = nyquist_mode_table_from_grid(
        mpol=int(static.cfg.mpol),
        ntor=int(static.cfg.ntor),
        ntheta=int(static.cfg.ntheta),
        nzeta=int(static.cfg.nzeta),
    )
    angle_cache = _quasisymmetry_angle_cache(
        nfp=int(static.cfg.nfp),
        xm_nyq=nyq_modes.m,
        xn_nyq=nyq_modes.n * int(static.cfg.nfp),
    )

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
            angle_cache=angle_cache,
        )

    def residuals_from_state(state: VMECState) -> jnp.ndarray:
        aspect = equilibrium_aspect_ratio_from_state(state=state, static=static)
        qs = _qs_eval_from_state(state)
        aspect_residual = jnp.asarray([float(aspect_weight) * (aspect - target_aspect)], dtype=jnp.float64)
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

    def state_objective_value_and_cotangent_from_packed(packed_state, layout):
        from ._compat import jax, jnp as _jnp
        from .state import unpack_state

        packed_state = _jnp.asarray(packed_state, dtype=_jnp.float64)

        def _objective(packed):
            state = unpack_state(packed, layout)
            aspect = equilibrium_aspect_ratio_from_state(state=state, static=static)
            aspect_residual = float(aspect_weight) * (aspect - target_aspect)
            qs = _qs_eval_from_state(state)
            qs_total = _jnp.asarray(qs["total"], dtype=_jnp.float64) * float(qs_weight) ** 2
            return 0.5 * aspect_residual * aspect_residual + 0.5 * qs_total

        return jax.value_and_grad(_objective)(packed_state)

    residuals_from_state._n_non_qs = 1
    residuals_from_state._aspect_target = float(target_aspect)
    residuals_from_state._aspect_weight = float(aspect_weight)
    residuals_from_state._objective_family = "qs"
    residuals_from_state._helicity_m = int(helicity_m)
    residuals_from_state._helicity_n = int(helicity_n)
    residuals_from_state._qs_total_from_state = (
        lambda state: float(_qs_eval_from_state(state)["total"]) * float(qs_weight) ** 2
    )
    residuals_from_state._state_cotangent_from_packed = state_cotangent_from_packed
    residuals_from_state._state_cotangent_operator_from_packed = state_cotangent_operator_from_packed
    residuals_from_state._state_objective_value_and_cotangent_from_packed = (
        state_objective_value_and_cotangent_from_packed
    )

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
    min_abs_iota: float | None = None,
    surfaces=None,
    aspect_weight: float = 1.0,
    qs_weight: float = 1.0,
    iota_weight: float = 1.0,
    iota_floor_softness: float = 1.0e-3,
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
    min_abs_iota:
        If given and ``target_iota`` is not given, adds one smooth lower-bound
        residual enforcing ``abs(mean_iota) >= min_abs_iota``.  This is a
        differentiable softplus hinge, not a hard target.
    surfaces:
        Surface coordinates (``s ∈ [0, 1]``) to evaluate quasisymmetry on.
        Defaults to ``np.arange(0, 1.01, 0.1)``.
    aspect_weight, qs_weight, iota_weight:
        Scalar weights applied to the corresponding residual blocks.
    """
    from .boundary import boundary_from_indata
    from .init_guess import initial_guess_from_boundary
    from .modes import nyquist_mode_table_from_grid
    from .quasisymmetry import (
        _quasisymmetry_angle_cache,
        quasisymmetry_ratio_residual_from_state,
    )
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
    pressure = _pressure_profile_for_static(indata, static)
    nyq_modes = nyquist_mode_table_from_grid(
        mpol=int(static.cfg.mpol),
        ntor=int(static.cfg.ntor),
        ntheta=int(static.cfg.ntheta),
        nzeta=int(static.cfg.nzeta),
    )
    angle_cache = _quasisymmetry_angle_cache(
        nfp=int(static.cfg.nfp),
        xm_nyq=nyq_modes.m,
        xn_nyq=nyq_modes.n * int(static.cfg.nfp),
    )
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
            angle_cache=angle_cache,
        )

    def residuals_from_state(state: VMECState) -> jnp.ndarray:
        parts: list[jnp.ndarray] = []

        if target_aspect is not None:
            aspect = equilibrium_aspect_ratio_from_state(state=state, static=static)
            parts.append(jnp.asarray([float(aspect_weight) * (aspect - target_aspect)], dtype=jnp.float64))

        if target_iota is not None or min_abs_iota is not None:
            _chips, _iotas, iotaf = equilibrium_iota_profiles_from_state(
                state=state,
                static=static,
                indata=_indata,
                signgs=_signgs,
            )
            iotas = jnp.asarray(_iotas, dtype=jnp.float64)
            mean_iota = jnp.asarray(0.0, dtype=iotas.dtype) if int(iotas.shape[0]) <= 1 else jnp.mean(iotas[1:])
            if target_iota is not None:
                iota_residual = mean_iota - target_iota
            else:
                iota_residual = smooth_min_abs_iota_residual(
                    mean_iota,
                    float(min_abs_iota),
                    softness=float(iota_floor_softness),
                )
            parts.append(jnp.asarray([float(iota_weight) * iota_residual], dtype=jnp.float64))

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

        if target_iota is not None or min_abs_iota is not None:
            block_index = offset
            offset += 1

            def _iota_from_packed(packed):
                state = unpack_state(packed, layout)
                _chips, _iotas, _iotaf = equilibrium_iota_profiles_from_state(
                    state=state,
                    static=static,
                    indata=_indata,
                    signgs=_signgs,
                )
                del _chips, _iotaf
                iotas = _jnp.asarray(_iotas, dtype=_jnp.float64)
                mean_iota = _jnp.asarray(0.0, dtype=iotas.dtype) if int(iotas.shape[0]) <= 1 else _jnp.mean(iotas[1:])
                if target_iota is not None:
                    iota_residual = mean_iota - target_iota
                else:
                    iota_residual = smooth_min_abs_iota_residual(
                        mean_iota,
                        float(min_abs_iota),
                        softness=float(iota_floor_softness),
                    )
                return float(iota_weight) * iota_residual

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
                        contribution = _jnp.nan_to_num(contribution, nan=0.0, posinf=0.0, neginf=0.0)
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

    def state_objective_value_and_cotangent_from_packed(packed_state, layout):
        from ._compat import jax, jnp as _jnp
        from .state import unpack_state

        packed_state = _jnp.asarray(packed_state, dtype=_jnp.float64)

        def _objective(packed):
            state = unpack_state(packed, layout)
            total = _jnp.asarray(0.0, dtype=_jnp.float64)
            if target_aspect is not None:
                aspect = equilibrium_aspect_ratio_from_state(state=state, static=static)
                aspect_residual = float(aspect_weight) * (aspect - target_aspect)
                total = total + 0.5 * aspect_residual * aspect_residual
            if target_iota is not None or min_abs_iota is not None:
                _chips, _iotas, _iotaf = equilibrium_iota_profiles_from_state(
                    state=state,
                    static=static,
                    indata=_indata,
                    signgs=_signgs,
                )
                del _chips, _iotaf
                iotas = _jnp.asarray(_iotas, dtype=_jnp.float64)
                mean_iota = _jnp.asarray(0.0, dtype=iotas.dtype) if int(iotas.shape[0]) <= 1 else _jnp.mean(iotas[1:])
                if target_iota is not None:
                    iota_residual = mean_iota - target_iota
                else:
                    iota_residual = smooth_min_abs_iota_residual(
                        mean_iota,
                        float(min_abs_iota),
                        softness=float(iota_floor_softness),
                    )
                iota_residual = float(iota_weight) * iota_residual
                total = total + 0.5 * iota_residual * iota_residual
            qs = _qs_eval_from_state(state)
            qs_total = _jnp.asarray(qs["total"], dtype=_jnp.float64) * float(qs_weight) ** 2
            return total + 0.5 * qs_total

        value, cotangent = jax.value_and_grad(_objective)(packed_state)
        if target_iota is not None or min_abs_iota is not None:
            # Match state_cotangent_operator_from_packed: the current-driven
            # iota path has gauge-null state entries that can produce NaNs in
            # reverse mode but do not contribute on the boundary-parameter
            # tangent subspace.
            cotangent = _jnp.nan_to_num(cotangent, nan=0.0, posinf=0.0, neginf=0.0)
        return value, cotangent

    residuals_from_state._n_non_qs = int(target_aspect is not None) + int(
        target_iota is not None or min_abs_iota is not None
    )
    residuals_from_state._aspect_target = None if target_aspect is None else float(target_aspect)
    residuals_from_state._aspect_weight = float(aspect_weight)
    residuals_from_state._objective_family = "qs"
    residuals_from_state._helicity_m = int(helicity_m)
    residuals_from_state._helicity_n = int(helicity_n)
    residuals_from_state._qs_total_from_state = (
        lambda state: float(_qs_eval_from_state(state)["total"]) * float(qs_weight) ** 2
    )
    residuals_from_state._state_cotangent_from_packed = state_cotangent_from_packed
    residuals_from_state._state_cotangent_operator_from_packed = state_cotangent_operator_from_packed
    residuals_from_state._state_objective_value_and_cotangent_from_packed = (
        state_objective_value_and_cotangent_from_packed
    )
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
        self._residuals_eval_fn = self._make_residuals_eval_fn(residuals_fn)
        self._n_qs: int | None = getattr(residuals_fn, "_n_qs", None)
        self._n_non_qs: int = int(getattr(residuals_fn, "_n_non_qs", 1))
        self._has_residual_block_metadata = hasattr(residuals_fn, "_n_qs") or hasattr(residuals_fn, "_n_non_qs")
        self._qs_total_from_state_fn = getattr(residuals_fn, "_qs_total_from_state", None)
        self._aspect_target = getattr(residuals_fn, "_aspect_target", None)
        self._aspect_weight = float(getattr(residuals_fn, "_aspect_weight", 1.0))
        self._objective_family = getattr(residuals_fn, "_objective_family", None)
        self._helicity_m = getattr(residuals_fn, "_helicity_m", None)
        self._helicity_n = getattr(residuals_fn, "_helicity_n", None)

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
        self._exact_solver_kwargs = dict(
            _base,
            preconditioner_use_precomputed_tridi=self._use_precomputed_tridi_for_exact_tape(),
        )
        self._trial_solver_kwargs = dict(
            _base,
            # Trial-point residuals do not need an adjoint tape.  Use a
            # backend-aware policy: CPU stays on the VMEC-control loop for
            # convergence/control parity, while accelerator backends use scan
            # to reduce launch overhead. VMEC_JAX_OPT_TRIAL_SCAN overrides this
            # for diagnostics.
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
        self._exact_state_key_by_id: dict[int, object] = {}
        self._exact_residual_cache: dict = {}
        self._exact_jacobian_cache: dict = {}
        self._discrete_jacobian_helper_cache: dict = {}
        self._scan_exact_helper_cache: dict = {}
        self._scan_exact_path = self._select_exact_path()
        self._initial_state_cache: OrderedDict[bytes, VMECState] = OrderedDict()
        self._initial_state_cache_max = 4
        self._remember_initial_state(np.zeros(len(self._specs), dtype=float), state0)
        self._initial_state_packed_helper = None
        self._initial_tangent_cache: dict = {}
        self._initial_tangent_direction_cache: dict = {}
        self._last_jacobian_residual: np.ndarray | None = None
        self._last_jacobian_source = "exact_tape_replay"
        self._trial_residual_cache: OrderedDict[bytes, np.ndarray] = OrderedDict()
        self._trial_residual_cache_max = 8
        self._profile: dict[str, dict[str, float | int]] = {}
        self._callback_trace_enabled = False
        self._callback_trace: list[dict] = []
        self._callback_point_ids: dict[bytes, int] = {}
        self._callback_previous_key: bytes | None = None

        # History collected during optimisation.
        self._history: list[dict] = []
        self._wall_t0: float = 0.0
        self._last_jacobian_key: list = [None]
        self._iota_fn = None  # set by run() when iota tracking is requested
        self._best_exact_params: np.ndarray | None = None
        self._best_exact_state: VMECState | None = None
        self._best_exact_residual: np.ndarray | None = None
        self._best_exact_cost: float = math.inf
        self._exact_history_rejected_count: int = 0

    @property
    def static(self):
        """VMEC static configuration used by this optimizer stage."""

        return self._static

    @property
    def indata(self):
        """VMEC input data used by this optimizer stage."""

        return self._indata

    @property
    def signgs(self) -> int:
        """VMEC Jacobian sign used for profile and Boozer adapters."""

        return int(self._signgs)

    @property
    def flux(self):
        """Half/full-mesh flux-profile data used by objective callbacks."""

        return self._flux

    # ── private helpers ───────────────────────────────────────────────────────

    def _resolve_solver_device(self, solver_device: str | None) -> str | None:
        name = "auto" if solver_device is None else str(solver_device).strip().lower()
        if name in ("", "none", "auto", "default"):
            return None
        try:
            from ._compat import jax as _jax

            current_backend = str(_jax.default_backend()).strip().lower() if _jax is not None else ""
        except Exception:
            current_backend = ""
        aliases = {
            "gpu": {"gpu", "cuda", "rocm", "tpu"},
            "cuda": {"gpu", "cuda"},
            "rocm": {"gpu", "rocm"},
            "tpu": {"tpu"},
            "cpu": {"cpu"},
        }
        if current_backend in aliases.get(name, {name}):
            # Explicitly requesting the already-active backend should not wrap
            # every callback in a default_device context or move static data a
            # second time.  On GPU this path is materially slower for new
            # accepted-point exact tapes.
            return None
        return name

    def _spec_max_mode(self) -> int:
        if not self._specs:
            return 0
        return max(max(abs(int(spec.m)), abs(int(spec.n))) for spec in self._specs)

    def _has_stellarator_asymmetric_parameter_specs(self) -> bool:
        return any(str(spec.kind).lower() in ("rs", "zc") for spec in self._specs)

    def _has_stellarator_asymmetric_configuration(self) -> bool:
        if self._has_stellarator_asymmetric_parameter_specs():
            return True
        get_bool = getattr(self._indata, "get_bool", None)
        if callable(get_bool):
            try:
                return bool(get_bool("LASYM", False))
            except Exception:
                pass
        return bool(getattr(getattr(self._static, "cfg", None), "lasym", False))

    def _resolve_optimizer_method(self, method: str, scipy_lsmr_maxiter: int | None) -> tuple[str, int | None, str | None]:
        """Resolve optimizer method aliases and the opt-in automatic policy.

        ``method="auto"`` is intentionally conservative and device-preserving:
        it chooses the matrix-free trust-region path for profiled high-mode,
        stellarator-symmetric QS/QI CPU/default-backend lanes where
        cold-process and memory-pressure profiles motivated the option. It does
        not guarantee the fastest warm wall time for every run, and it never
        moves work between CPU and GPU; explicit device choices are preserved.
        """

        method_key = str(method).strip().lower().replace("-", "_")
        aliases = {
            "matrix_free": "scipy_matrix_free",
            "scipy_mf": "scipy_matrix_free",
            "trf": "scipy",
        }
        method_key = aliases.get(method_key, method_key)
        if method_key not in ("auto", "adaptive"):
            return method_key, scipy_lsmr_maxiter, None

        backend = _optimizer_backend_name(getattr(self, "_solver_device_name", None))
        if backend in ("gpu", "cuda", "rocm", "tpu", "metal"):
            return "scipy", scipy_lsmr_maxiter, f"auto:dense-preserves-{backend}"
        if self._has_stellarator_asymmetric_configuration():
            return "scipy", scipy_lsmr_maxiter, "auto:dense-lasym"

        helicity_m = None if self._helicity_m is None else int(self._helicity_m)
        helicity_n = None if self._helicity_n is None else int(self._helicity_n)
        if self._spec_max_mode() >= 3 and self._objective_family in ("qs", "qi"):
            lsmr_maxiter = 4 if scipy_lsmr_maxiter is None else scipy_lsmr_maxiter
            if self._objective_family == "qi":
                family = "qi"
            elif helicity_m == 1 and helicity_n == 0:
                family = "qa"
            elif helicity_m == 0 and helicity_n not in (None, 0):
                family = "qp"
            elif helicity_m == 1 and helicity_n not in (None, 0):
                family = "qh"
            else:
                family = "qs"
            return "scipy_matrix_free", lsmr_maxiter, f"auto:{family}-high-mode-matrix-free"

        return "scipy", scipy_lsmr_maxiter, "auto:dense-default"

    def _select_exact_path(self) -> str:
        """Choose the accepted-point differentiation path.

        The established non-scan discrete-adjoint tape is the default on CPU
        and GPU. May 2026 cold and warm ``office`` RTX A4000 profiling showed
        the scan-differentiated exact path can be useful for targeted parity
        studies but is not a robust GPU default for accepted-point Jacobians.
        The environment override ``VMEC_JAX_OPT_EXACT_PATH={tape,scan}``
        remains available for profiling and parity studies.
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
        if backend in ("gpu", "cuda", "tpu", "rocm"):
            return "tape"
        return "tape"

    def _use_precomputed_tridi_for_exact_tape(self) -> bool | None:
        """Use precomputed Thomas coefficients for accepted GPU tape solves.

        This is deliberately scoped to accepted-point exact solves. May 2026
        office RTX A4000 profiles show it reduces dense-Jacobian tape cost for
        small-DOF tapes, while larger parameter spaces can lose more in replay
        payload cost than they gain in preconditioner cost. ``None`` preserves
        the solver's legacy environment-controlled default for CPU/default
        backends.
        """

        forced = os.getenv("VMEC_JAX_OPT_EXACT_TRIDI_PRECOMPUTE", "").strip().lower()
        if forced in ("1", "true", "yes", "on"):
            return True
        if forced in ("0", "false", "no", "off"):
            return False
        backend = str(self._solver_device_name or "").strip().lower()
        if not backend:
            try:
                from ._compat import jax as _jax

                backend = str(_jax.default_backend()).strip().lower() if _jax is not None else "cpu"
            except Exception:
                backend = "cpu"
        if backend not in ("gpu", "cuda", "tpu", "rocm"):
            return None
        try:
            max_dofs = int(os.getenv("VMEC_JAX_OPT_EXACT_TRIDI_PRECOMPUTE_MAX_DOFS", "12"))
        except ValueError:
            max_dofs = 12
        if max_dofs < 0:
            return False
        return True if len(self._specs) <= max_dofs else None

    def _use_scan_for_trial_solves(self) -> bool:
        """Return whether trial residual solves should use the scan loop.

        Exact-optimizer trial residuals are short, trace-compatible VMEC solves
        called repeatedly by SciPy's trust-region line search.  They do not need
        an adjoint tape.  May 2026 callback profiles showed CPU scan can be
        slower and less well behaved for QH mode-2 trial residuals, while the
        current ``office`` GPU path is faster on scan for the same residual
        norm.  Environment overrides always win.
        """
        forced = os.getenv("VMEC_JAX_OPT_TRIAL_SCAN", "").strip().lower()
        if forced in ("1", "true", "yes", "on", "scan"):
            return True
        if forced in ("0", "false", "no", "off", "loop", "none"):
            return False
        backend = str(getattr(self, "_solver_device_name", None) or "").strip().lower()
        if not backend:
            try:
                from ._compat import jax as _jax

                backend = str(_jax.default_backend()).strip().lower() if _jax is not None else "cpu"
            except Exception:
                backend = "cpu"
        return backend in ("gpu", "cuda", "tpu", "rocm")

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
        from .vmec_tomnsp import tomnsps_fft_policy_override

        backend_name = str(self._solver_device_name).strip().lower()
        tomnsps_fft_override = (
            backend_name in ("gpu", "cuda", "rocm", "tpu")
            if os.getenv("VMEC_JAX_TOMNSPS_FFT") is None
            else None
        )
        with ExitStack() as stack:
            stack.enter_context(self._solver_device_context())
            stack.enter_context(tomnsps_fft_policy_override(tomnsps_fft_override))
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

    def _profile_add_counter(self, name: str, value: int | float) -> None:
        """Record a diagnostic counter in the profile schema without timing it."""
        if not hasattr(self, "_profile"):
            self._profile = {}
        rec = self._profile.setdefault(name, {"count": 0, "wall_time_s": 0.0})
        rec["count"] = int(rec["count"]) + 1
        rec["wall_time_s"] = float(rec["wall_time_s"]) + float(value)

    def _profile_solver_timing(
        self,
        diagnostics,
        *,
        profile_prefix: str,
        phase_wall_s: float,
        unattributed_name: str | None,
    ) -> float:
        if not isinstance(diagnostics, dict):
            return 0.0
        timing = diagnostics.get("timing")
        if not isinstance(timing, dict):
            return 0.0
        solver_total = 0.0
        timing_keys = (
            ("solve_total_s", "solve_total"),
            ("setup_total_s", "setup_total"),
            ("setup_axis_reset_s", "setup_axis_reset"),
            ("setup_axis_reset_compute_forces_s", "setup_axis_reset_compute_forces"),
            ("setup_axis_reset_unattributed_s", "setup_axis_reset_unattributed"),
            ("setup_unattributed_s", "setup_unattributed"),
            ("iteration_loop_s", "iteration_loop"),
            ("iteration_prepare_s", "iteration_prepare"),
            ("compute_forces_s", "compute_forces"),
            ("compute_forces_first_s", "compute_forces_first"),
            ("compute_forces_rest_s", "compute_forces_rest"),
            ("iteration_residual_metrics_s", "iteration_residual_metrics"),
            ("preconditioner_s", "preconditioner"),
            ("precond_refresh_s", "precond_refresh"),
            ("precond_apply_s", "preconditioner_apply"),
            ("precond_mode_scale_s", "preconditioner_mode_scale"),
            ("update_s", "update"),
            ("update_state_s", "update_state"),
            ("update_trace_build_s", "update_trace_build"),
            ("update_trace_finalize_s", "update_trace_finalize"),
            ("iteration_post_update_s", "iteration_post_update"),
            ("iteration_loop_unattributed_s", "iteration_loop_unattributed"),
            ("finalize_s", "finalize"),
            ("scan_total_s", "scan_total"),
            ("scan_setup_s", "scan_setup"),
            ("scan_initial_compute_forces_s", "scan_initial_compute_forces"),
            ("scan_axis_reset_compute_forces_s", "scan_axis_reset_compute_forces"),
            ("scan_run_setup_s", "scan_run_setup"),
            ("scan_runner_cache_lookup_s", "scan_runner_cache_lookup"),
            ("scan_runner_cache_build_s", "scan_runner_cache_build"),
            ("scan_preflight_s", "scan_preflight"),
            ("scan_device_run_s", "scan_device_run"),
            ("scan_device_dispatch_s", "scan_device_dispatch"),
            ("scan_device_ready_s", "scan_device_ready"),
            ("scan_runner_cache_hit_device_run_s", "scan_runner_cache_hit_device_run"),
            ("scan_runner_cache_hit_dispatch_s", "scan_runner_cache_hit_dispatch"),
            ("scan_runner_cache_hit_ready_s", "scan_runner_cache_hit_ready"),
            ("scan_runner_cache_miss_device_run_s", "scan_runner_cache_miss_device_run"),
            ("scan_runner_cache_miss_dispatch_s", "scan_runner_cache_miss_dispatch"),
            ("scan_runner_cache_miss_ready_s", "scan_runner_cache_miss_ready"),
            ("scan_runner_cache_bypass_device_run_s", "scan_runner_cache_bypass_device_run"),
            ("scan_runner_cache_bypass_dispatch_s", "scan_runner_cache_bypass_dispatch"),
            ("scan_runner_cache_bypass_ready_s", "scan_runner_cache_bypass_ready"),
            ("scan_host_materialize_s", "scan_host_materialize"),
            ("scan_postprocess_s", "scan_postprocess"),
            ("scan_unattributed_s", "scan_unattributed"),
        )
        counter_keys = (
            ("scan_runner_cache_hit_count", "scan_runner_cache_hit_count"),
            ("scan_runner_cache_miss_count", "scan_runner_cache_miss_count"),
            ("scan_runner_cache_bypass_count", "scan_runner_cache_bypass_count"),
        )
        outer_solver_total_keys = {"setup_total_s", "iteration_loop_s", "finalize_s", "scan_total_s"}
        fallback_solver_total_keys = {"compute_forces_s", "preconditioner_s", "update_s", "scan_total_s"}
        has_outer_solver_total = any(key in timing for key in outer_solver_total_keys)
        for key, suffix in timing_keys:
            if key not in timing:
                continue
            try:
                value = float(timing.get(key, 0.0))
            except Exception:
                continue
            self._profile_add(f"{profile_prefix}_{suffix}", value)
            if key in (outer_solver_total_keys if has_outer_solver_total else fallback_solver_total_keys):
                solver_total += max(0.0, value)
        for key, suffix in counter_keys:
            if key not in timing:
                continue
            try:
                value = int(timing.get(key, 0))
            except Exception:
                continue
            self._profile_add_counter(f"{profile_prefix}_{suffix}", value)
        for key, value_raw in sorted(timing.items()):
            if not (str(key).startswith("scan_runner_cache_miss_category_") and str(key).endswith("_count")):
                continue
            try:
                value = int(value_raw)
            except Exception:
                continue
            self._profile_add_counter(f"{profile_prefix}_{key}", value)
        if unattributed_name is not None:
            self._profile_add(unattributed_name, max(0.0, float(phase_wall_s) - solver_total))
        return solver_total

    def _profile_exact_tape_solver_timing(self, tape, tape_build_wall_s: float) -> None:
        diagnostics = getattr(tape, "diagnostics", None)
        solver_total = self._profile_solver_timing(
            diagnostics,
            profile_prefix="exact_tape_solver",
            phase_wall_s=tape_build_wall_s,
            unattributed_name=None,
        )
        timing = diagnostics.get("timing") if isinstance(diagnostics, dict) else None
        build_leaf_total = 0.0
        has_solve_call_timer = False
        if isinstance(timing, dict):
            for key, profile_name in _EXACT_TAPE_BUILD_TIMING_PROFILE_NAMES:
                if key not in timing:
                    continue
                try:
                    value = float(timing.get(key, 0.0))
                except Exception:
                    continue
                self._profile_add(profile_name, value)
                build_leaf_total += max(0.0, value)
                if key == "tape_solve_call_s":
                    has_solve_call_timer = True
        attributed = build_leaf_total if has_solve_call_timer else solver_total + build_leaf_total
        self._profile_add("exact_tape_build_unattributed", max(0.0, float(tape_build_wall_s) - attributed))

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

    def _sync_replay_timing_enabled(self) -> bool:
        flag = os.getenv("VMEC_JAX_OPT_SYNC_REPLAY_TIMING", "").strip().lower()
        return flag not in ("", "0", "false", "no", "off")

    def _profile_async_phase(self, name: str, start: float, value):
        """Record dispatch time, optionally synchronizing for device-ready timing."""

        dispatch_s = time.perf_counter() - float(start)
        self._profile_add(f"{name}_dispatch", dispatch_s)
        total_s = dispatch_s
        if self._sync_replay_timing_enabled():
            try:
                from ._compat import jax as _jax

                t_ready = time.perf_counter()
                value = _jax.block_until_ready(value)
                ready_s = time.perf_counter() - t_ready
            except Exception:
                ready_s = 0.0
            self._profile_add(f"{name}_ready", ready_s)
            total_s += ready_s
        self._profile_add(name, total_s)
        return value

    def _profile_blocking_phase(self, name: str, start: float, value):
        """Record dispatch and mandatory device-ready timing for a blocking callback phase."""

        dispatch_s = time.perf_counter() - float(start)
        self._profile_add(f"{name}_dispatch", dispatch_s)
        try:
            from ._compat import jax as _jax

            t_ready = time.perf_counter()
            value = _jax.block_until_ready(value)
            ready_s = time.perf_counter() - t_ready
        except Exception:
            ready_s = 0.0
        self._profile_add(f"{name}_ready", ready_s)
        self._profile_add(name, dispatch_s + ready_s)
        return value

    def _make_residuals_eval_fn(self, residuals_fn: Callable) -> Callable:
        """Return the non-differentiating residual evaluator used by callbacks."""
        flag = os.getenv("VMEC_JAX_OPT_JIT_RESIDUALS", "1").strip().lower()
        if flag in ("", "0", "false", "no", "off"):
            return residuals_fn

        from ._compat import jax, jnp as _jnp

        @jax.jit
        def _eval(state):
            return _jnp.asarray(residuals_fn(state), dtype=_jnp.float64)

        return _eval

    def _evaluate_residuals_from_state(self, state: VMECState) -> np.ndarray:
        fn = getattr(self, "_residuals_eval_fn", self._residuals_fn)
        return np.asarray(fn(state), dtype=float)

    def _callback_point_id(self, cache_key: bytes) -> int:
        point_ids = getattr(self, "_callback_point_ids", None)
        if point_ids is None:
            self._callback_point_ids = {}
            point_ids = self._callback_point_ids
        point_id = point_ids.get(cache_key)
        if point_id is None:
            point_id = len(point_ids)
            point_ids[cache_key] = point_id
        return int(point_id)

    def _trace_callback_event(
        self,
        kind: str,
        params,
        *,
        source: str,
        wall_time_s: float,
    ) -> None:
        if not getattr(self, "_callback_trace_enabled", False):
            return
        cache_key = self._exact_cache_key(params)
        previous_key = getattr(self, "_callback_previous_key", None)
        event = {
            "index": len(self._callback_trace),
            "kind": str(kind),
            "source": str(source),
            "point_id": self._callback_point_id(cache_key),
            "same_as_previous": bool(previous_key == cache_key),
            "wall_time_s": float(wall_time_s),
        }
        self._callback_trace.append(event)
        self._callback_previous_key = cache_key

    def _callback_trace_dump(self) -> dict:
        events = list(getattr(self, "_callback_trace", []))
        counts: dict[str, int] = {}
        wall_time: dict[str, float] = {}
        for event in events:
            key = f"{event['kind']}:{event['source']}"
            counts[key] = counts.get(key, 0) + 1
            wall_time[key] = wall_time.get(key, 0.0) + float(event["wall_time_s"])
        return {
            "enabled": bool(getattr(self, "_callback_trace_enabled", False)),
            "events": events,
            "summary": {key: {"count": counts[key], "wall_time_s": wall_time[key]} for key in sorted(counts)},
        }

    def _exact_cache_key(self, params) -> bytes:
        return np.asarray(params, dtype=float).reshape(-1).tobytes()

    def _remember_initial_state(self, params, state: VMECState) -> None:
        cache = getattr(self, "_initial_state_cache", None)
        if cache is None:
            self._initial_state_cache = OrderedDict()
            cache = self._initial_state_cache
        cache_key = self._exact_cache_key(params)
        cache[cache_key] = state
        cache.move_to_end(cache_key)
        max_size = max(0, int(getattr(self, "_initial_state_cache_max", 0)))
        while max_size and len(cache) > max_size:
            cache.popitem(last=False)
        if max_size == 0:
            cache.clear()

    def _initial_state_from_params(self, params, *, profile_name: str) -> VMECState:
        cache_key = self._exact_cache_key(params)
        cache = getattr(self, "_initial_state_cache", None)
        if cache is not None and cache_key in cache:
            state0 = cache.pop(cache_key)
            cache[cache_key] = state0
            self._profile_add(f"{profile_name}_cache_hit", 0.0)
            return state0

        t_guess = time.perf_counter()
        state0 = self._initial_state_from_params_jit(params)
        if state0 is None:
            boundary_now = self._boundary_from_params(params)
            state0 = initial_guess_from_boundary(self._static, boundary_now, self._indata, vmec_project=True)
        self._remember_initial_state(params, state0)
        self._profile_add(profile_name, time.perf_counter() - t_guess)
        return state0

    def _use_jit_initial_state(self) -> bool:
        flag = os.getenv("VMEC_JAX_OPT_JIT_INITIAL_STATE")
        if flag is not None:
            return flag.strip().lower() not in ("", "0", "false", "no", "off")
        # The projected initial-state map is small enough that JIT compile and
        # dispatch overhead dominates cold CPU exact callbacks.  Keep the JIT
        # helper opt-in until a workload has enough same-shape reuse to amortize
        # compilation.
        return False

    def _initial_state_from_params_jit(self, params) -> VMECState | None:
        """Return the projected initial state using a cached JIT helper when safe."""

        if not self._use_jit_initial_state():
            return None
        try:
            from ._compat import jax, jnp as _jnp
            from .init_guess import initial_guess_from_boundary as _ig
            from .state import pack_state, unpack_state
        except Exception:
            return None

        helper = getattr(self, "_initial_state_packed_helper", None)
        if helper is None:

            @jax.jit
            def _packed_initial_state(p):
                bdy = self._boundary_from_params(p)
                state = _ig(
                    self._static,
                    bdy,
                    self._indata,
                    vmec_project=True,
                )
                return _jnp.asarray(pack_state(state), dtype=_jnp.float64)

            helper = _packed_initial_state
            self._initial_state_packed_helper = helper

        try:
            packed = helper(_jnp.asarray(params, dtype=_jnp.float64))
            if self._sync_initial_state_projection_enabled():
                packed = jax.block_until_ready(packed)
            return unpack_state(packed, self._layout)
        except Exception:
            return None

    def _sync_initial_state_projection_enabled(self) -> bool:
        """Return whether the JIT initial-state projection should synchronize."""

        flag = os.getenv("VMEC_JAX_OPT_SYNC_INITIAL_STATE", "").strip().lower()
        return flag in ("1", "true", "yes", "on")

    def _remember_exact_state(self, cache_key: bytes, state: VMECState) -> None:
        self._exact_state_cache = {cache_key: state}
        if not hasattr(self, "_exact_state_key_by_id"):
            self._exact_state_key_by_id = {}
        self._exact_state_key_by_id[id(state)] = cache_key
        residual_cache = getattr(self, "_exact_residual_cache", None)
        if residual_cache is not None and cache_key not in residual_cache:
            residual_cache.clear()

    def _state_matches_params(self, state: VMECState, params) -> bool:
        """Return true when *state* is a known exact solve for *params*."""

        state_keys = getattr(self, "_exact_state_key_by_id", {})
        return state_keys.get(id(state)) == self._exact_cache_key(params)

    def _remember_exact_residual(self, cache_key: bytes, residual: np.ndarray) -> None:
        self._exact_residual_cache = {cache_key: np.asarray(residual, dtype=float).reshape(-1).copy()}

    def _remember_exact_jacobian(self, cache_key: bytes, jacobian: np.ndarray, residual: np.ndarray) -> None:
        """Keep the most recent dense accepted-point Jacobian for same-point callbacks."""

        self._exact_jacobian_cache = {
            cache_key: (
                np.asarray(jacobian, dtype=float).copy(),
                np.asarray(residual, dtype=float).reshape(-1).copy(),
            )
        }

    def _remember_best_exact_point(
        self,
        params,
        residual: np.ndarray,
        cost: float | None = None,
        *,
        state: VMECState | None = None,
    ) -> None:
        """Track the best exact accepted-point residual seen during one run."""

        residual_arr = np.asarray(residual, dtype=float).reshape(-1)
        if cost is None:
            cost = 0.5 * float(np.dot(residual_arr, residual_arr))
        if not np.isfinite(float(cost)) or not np.all(np.isfinite(residual_arr)):
            return
        if float(cost) < float(getattr(self, "_best_exact_cost", math.inf)):
            cache_key = self._exact_cache_key(params)
            self._best_exact_cost = float(cost)
            self._best_exact_params = np.asarray(params, dtype=float).reshape(-1).copy()
            self._best_exact_residual = residual_arr.copy()
            best_state = state
            if best_state is not None and not self._state_matches_params(best_state, params):
                best_state = None
            if best_state is None:
                exact_cache = getattr(self, "_exact_cache", {})
                if cache_key in exact_cache:
                    best_state = exact_cache[cache_key][0]
                else:
                    best_state = getattr(self, "_exact_state_cache", {}).get(cache_key)
            self._best_exact_state = best_state

    def _exact_history_accepts(self, cost: float) -> bool:
        """Return whether an exact callback row should enter accepted history."""

        if not np.isfinite(float(cost)):
            return False
        best_cost = float(getattr(self, "_best_exact_cost", math.inf))
        if not np.isfinite(best_cost):
            return True
        tol = max(1.0e-14, 1.0e-9 * max(1.0, abs(best_cost), abs(float(cost))))
        return float(cost) <= best_cost + tol

    def _cached_exact_residual(
        self,
        params=None,
        *,
        cache_key: bytes | None = None,
    ) -> np.ndarray | None:
        if cache_key is None:
            if params is None:
                return None
            cache_key = self._exact_cache_key(params)
        last_key = getattr(self, "_last_jacobian_key", [None])[0]
        if last_key == cache_key and getattr(self, "_last_jacobian_residual", None) is not None:
            return np.asarray(self._last_jacobian_residual, dtype=float).reshape(-1)
        cache = getattr(self, "_exact_residual_cache", None)
        if cache is not None and cache_key in cache:
            self._profile_add("exact_residual_cache_hit", 0.0)
            return np.asarray(cache[cache_key], dtype=float).reshape(-1)
        return None

    def _cached_exact_state(self, params):
        cache_key = self._exact_cache_key(params)
        if cache_key in self._exact_cache:
            state = self._exact_cache[cache_key][0]
            self._remember_exact_state(cache_key, state)
            self._profile_add("exact_cache_hit", 0.0)
            return state
        if cache_key in getattr(self, "_exact_state_cache", {}):
            self._profile_add("exact_state_cache_hit", 0.0)
            state = self._exact_state_cache[cache_key]
            self._remember_exact_state(cache_key, state)
            return state
        return None

    def _cached_trial_residual(self, params) -> np.ndarray | None:
        cache_key = self._exact_cache_key(params)
        cache = getattr(self, "_trial_residual_cache", None)
        if cache is None or cache_key not in cache:
            return None
        residual = cache.pop(cache_key)
        cache[cache_key] = residual
        self._profile_add("trial_residual_cache_hit", 0.0)
        return np.asarray(residual, dtype=float)

    def _remember_trial_residual(self, params, residual: np.ndarray) -> None:
        cache_key = self._exact_cache_key(params)
        cache = getattr(self, "_trial_residual_cache", None)
        if cache is None:
            self._trial_residual_cache = OrderedDict()
            cache = self._trial_residual_cache
        cache[cache_key] = np.asarray(residual, dtype=float).copy()
        cache.move_to_end(cache_key)
        max_size = max(0, int(getattr(self, "_trial_residual_cache_max", 0)))
        while max_size and len(cache) > max_size:
            cache.popitem(last=False)
        if max_size == 0:
            cache.clear()

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

    def _boundary_from_params_numpy(self, params) -> BoundaryCoeffs:
        """Boundary coefficients with parameters applied on the host."""
        boundary = _apply_boundary_params_numpy(
            self._boundary_input if self._boundary_input is not None else self._boundary,
            self._specs,
            np.asarray(params, dtype=float),
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

    def _initial_tangent_cache_key(self, params):
        """Cache key for affine initial-state tangent maps.

        With the accepted-point magnetic axis frozen, VMEC's initial state is
        affine in the boundary coefficients except for the discrete theta-flip
        branch.  Keep one tangent map per flip branch so Jacobian callbacks do
        not re-linearize the same initialization graph at every accepted point.
        """
        from .init_guess import _vmec_lflip_from_boundary

        try:
            boundary = self._boundary_from_params_numpy(np.asarray(params, dtype=float))
        except Exception:
            try:
                boundary = self._boundary_from_params(params)
            except Exception:
                return None
        try:
            lflip = _vmec_lflip_from_boundary(self._static, boundary)
        except Exception:
            return None
        if lflip is None:
            lflip = False
        return (
            int(np.asarray(params).size),
            bool(lflip),
            bool(self._boundary_input is not None),
            bool(self._static.cfg.lasym),
            int(self._static.cfg.ns),
            int(self._static.modes.K),
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
        state0 = self._initial_state_from_params(
            params,
            profile_name="initial_guess_trial" if trial else "initial_guess_forward",
        )
        t_solve = time.perf_counter()
        if trial:
            trial_solver_kwargs = dict(self._trial_solver_kwargs)
            trial_solver_kwargs.setdefault("state_only", bool(trial_solver_kwargs.get("use_scan", False)))
            result = solve_fixed_boundary_residual_iter(
                state0,
                self._static,
                max_iter=self._trial_max_iter,
                ftol=self._trial_ftol,
                **trial_solver_kwargs,
            )
        else:
            result = solve_fixed_boundary_residual_iter(
                state0,
                self._static,
                max_iter=self._inner_max_iter,
                ftol=self._inner_ftol,
                **self._exact_solver_kwargs,
            )
        solve_wall_s = time.perf_counter() - t_solve
        self._profile_solver_timing(
            getattr(result, "diagnostics", None),
            profile_prefix="trial_solver" if trial else "forward_exact_solver",
            phase_wall_s=solve_wall_s,
            unattributed_name="solve_forward_trial_unattributed" if trial else "solve_forward_exact_unattributed",
        )
        self._profile_add(
            "solve_forward_trial" if trial else "solve_forward_exact",
            solve_wall_s,
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
            state_only=True,
            light_history=True,
            resume_state_mode="none",
        )

        def _scan_state_from_params(p):
            boundary_now = self._boundary_from_params(p)
            state0 = initial_guess_from_boundary(self._static, boundary_now, self._indata, vmec_project=True)
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

    def _solve_exact_with_tape(self, params, *, return_payload: bool = False, jvp_only: bool = False):
        """Run exact solve + build adjoint tape, with caching."""
        if self._solver_device_name is not None and not self._inside_solver_device_context:
            return self._run_in_solver_device_context(
                self._solve_exact_with_tape,
                params,
                return_payload=return_payload,
                jvp_only=jvp_only,
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
        state0 = self._initial_state_from_params(params, profile_name="initial_guess_exact")
        axis_override = extract_axis_override_from_state(state0, self._static)
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
            jvp_only=bool(jvp_only),
        )
        tape_build_wall_s = time.perf_counter() - t_tape
        self._profile_add("exact_tape_build", tape_build_wall_s)
        if jvp_only:
            self._profile_add("exact_tape_build_jvp_only", tape_build_wall_s)
        self._profile_exact_tape_solver_timing(tape, tape_build_wall_s)
        t_unpack = time.perf_counter()
        state = unpack_state(_jnp.asarray(tape.final_packed_state, dtype=_jnp.float64), self._layout)
        payload = {"tape": tape, "axis_override": axis_override}
        self._exact_cache.clear()
        if not jvp_only:
            self._exact_cache[cache_key] = (state, payload)
        self._remember_exact_state(cache_key, state)
        self._profile_add("exact_unpack_cache", time.perf_counter() - t_unpack)
        self._profile_add("exact_solve_with_tape_total", time.perf_counter() - t_total)
        if jvp_only:
            self._profile_add("exact_solve_with_tape_jvp_only_total", time.perf_counter() - t_total)
        return (state, payload) if return_payload else state

    # ── public residual / Jacobian interface ──────────────────────────────────

    def residual_fun(self, params) -> np.ndarray:
        """Exact residual at *params* (builds adjoint tape, cached)."""
        if self._solver_device_name is not None and not self._inside_solver_device_context:
            return self._run_in_solver_device_context(self.residual_fun, params)
        cache_key = self._exact_cache_key(params)
        cached = self._cached_exact_residual(params, cache_key=cache_key)
        if cached is not None:
            self._profile_add("residual_exact_cache_hit", 0.0)
            return cached
        if self._scan_exact_path == "scan":
            # Avoid compiling a second residual-only scan executable.  The exact
            # optimizer immediately needs the same accepted-point state for
            # history/cached residuals, so solve once and evaluate residuals from
            # that state.
            state = self._solve_scan_exact_state(params)
            t0 = time.perf_counter()
            out = self._evaluate_residuals_from_state(state)
            self._profile_add("scan_residual_eval_exact", time.perf_counter() - t0)
            self._remember_exact_residual(cache_key, out)
            return out
        state = self._solve_exact_with_tape(params)
        t_res = time.perf_counter()
        out = self._evaluate_residuals_from_state(state)
        self._profile_add("residual_eval_exact", time.perf_counter() - t_res)
        self._remember_exact_residual(cache_key, out)
        return out

    def forward_residual_fun(self, params) -> np.ndarray:
        """Relaxed residual for line-search trial evaluations."""
        if self._solver_device_name is not None and not self._inside_solver_device_context:
            return self._run_in_solver_device_context(self.forward_residual_fun, params)
        cached = self._cached_trial_residual(params)
        if cached is not None:
            return cached
        exact_cached = self._cached_exact_residual(params)
        if exact_cached is not None:
            self._profile_add("trial_residual_exact_cache_hit", 0.0)
            self._remember_trial_residual(params, exact_cached)
            return np.asarray(exact_cached, dtype=float).reshape(-1)
        state = self._solve_forward(params, trial=True)
        t_res = time.perf_counter()
        out = self._evaluate_residuals_from_state(state)
        self._profile_add("residual_eval_trial", time.perf_counter() - t_res)
        self._remember_trial_residual(params, out)
        return out

    def _state_and_tangent_columns(self, params, *, profile_prefix: str):
        """Return accepted-point state and packed tangent columns as JAX arrays."""
        from ._compat import jnp as _jnp
        from .discrete_adjoint import checkpoint_tape_state_jvp_columns

        params = _jnp.asarray(params, dtype=_jnp.float64)
        state, payload = self._solve_exact_with_tape_for_jvp(params)
        if int(params.size) == 0:
            empty = _jnp.zeros((0, int(self._layout.size)), dtype=_jnp.float64)
            return state, empty

        initial_tangents = self._initial_tangent_columns(
            params,
            payload["axis_override"],
            profile_prefix=profile_prefix,
        )
        column_chunk = self._lasym_replay_column_chunk(int(params.size))
        if column_chunk is not None:
            self._profile_add(f"{profile_prefix}_replay_column_chunk_{column_chunk}", 0.0)
        t_replay = time.perf_counter()
        final_tangents = checkpoint_tape_state_jvp_columns(
            tape=payload["tape"],
            static=self._static,
            initial_tangents=initial_tangents,
            rebuild_preconditioner=True,
            column_chunk=column_chunk,
        )
        final_tangents = self._profile_async_phase(
            f"{profile_prefix}_tape_replay",
            t_replay,
            final_tangents,
        )
        return state, final_tangents

    def _solve_exact_with_tape_for_jvp(self, params):
        """Build an exact tape optimized for forward tangent-column replay."""
        solve = self._solve_exact_with_tape
        if not self._jvp_only_exact_tape_enabled():
            return solve(params, return_payload=True)
        try:
            from inspect import Parameter, signature

            parameters = signature(solve).parameters
            accepts_jvp_only = "jvp_only" in parameters or any(
                parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values()
            )
        except (TypeError, ValueError):
            accepts_jvp_only = True
        if accepts_jvp_only:
            env_name = "VMEC_JAX_JVP_ONLY_EXACT_TAPE_BASEPOINT_CARRIES"
            set_basepoint_default = self._jvp_only_basepoint_carries_default_enabled()
            old_basepoint_env = os.environ.get(env_name)
            if set_basepoint_default:
                os.environ[env_name] = "1"
            try:
                return solve(params, return_payload=True, jvp_only=True)
            finally:
                if set_basepoint_default:
                    if old_basepoint_env is None:
                        os.environ.pop(env_name, None)
                    else:
                        os.environ[env_name] = old_basepoint_env
        return solve(params, return_payload=True)

    def _jvp_only_exact_tape_enabled(self) -> bool:
        flag = os.getenv("VMEC_JAX_OPT_JVP_ONLY_EXACT_TAPE", "").strip().lower()
        if flag:
            return flag in ("1", "true", "yes", "on")
        backend = _optimizer_backend_name(getattr(self, "_solver_device_name", None))
        return backend in ("gpu", "cuda", "rocm")

    def _jvp_only_basepoint_carries_default_enabled(self) -> bool:
        """Preserve JVP-only base carries by default on GPU exact callbacks."""

        flag = os.getenv("VMEC_JAX_JVP_ONLY_EXACT_TAPE_BASEPOINT_CARRIES", "").strip().lower()
        if flag:
            return False
        backend = _optimizer_backend_name(getattr(self, "_solver_device_name", None))
        return backend in ("gpu", "cuda", "rocm")

    def _initial_tangent_columns(self, params, axis_override, *, profile_prefix: str):
        """Return cached packed initial-state tangents for boundary parameters."""
        from ._compat import jax, jnp as _jnp
        from .init_guess import initial_guess_from_boundary as _ig
        from .state import pack_state

        params = _jnp.asarray(params, dtype=_jnp.float64)
        if int(params.size) == 0:
            return _jnp.zeros((0, int(self._layout.size)), dtype=_jnp.float64)

        t_initial = time.perf_counter()
        t_key = time.perf_counter()
        cache_key = self._initial_tangent_cache_key(params)
        self._profile_add(
            f"{profile_prefix}_initial_tangents_cache_key",
            time.perf_counter() - t_key,
        )
        initial_tangents = self._initial_tangent_cache.get(cache_key) if cache_key is not None else None
        if initial_tangents is None:
            self._profile_add(f"{profile_prefix}_initial_tangents_cache_miss", 0.0)
            axis_override = {
                key: _jnp.asarray(value, dtype=params.dtype) for key, value in axis_override.items()
            }

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

            t_linearize = time.perf_counter()
            _, initial_state_linear = jax.linearize(_initial_state_packed, params)
            self._profile_add(
                f"{profile_prefix}_initial_tangents_linearize",
                time.perf_counter() - t_linearize,
            )
            if int(params.size) == 1:
                t_jvp = time.perf_counter()
                initial_tangents = initial_state_linear(_jnp.ones_like(params))[None, :]
                initial_tangents = self._profile_async_phase(
                    f"{profile_prefix}_initial_tangents_single_jvp",
                    t_jvp,
                    initial_tangents,
                )
            else:
                directions = self._initial_tangent_directions(params, profile_prefix=profile_prefix)
                t_vmap = time.perf_counter()
                initial_tangents = jax.vmap(initial_state_linear)(directions)
                initial_tangents = self._profile_async_phase(
                    f"{profile_prefix}_initial_tangents_vmap",
                    t_vmap,
                    initial_tangents,
                )
            if cache_key is not None:
                t_store = time.perf_counter()
                self._initial_tangent_cache[cache_key] = initial_tangents
                self._profile_add(
                    f"{profile_prefix}_initial_tangents_cache_store",
                    time.perf_counter() - t_store,
                )
        else:
            self._profile_add(f"{profile_prefix}_initial_tangents_cache_hit", 0.0)
        self._profile_add(
            f"{profile_prefix}_initial_tangents",
            time.perf_counter() - t_initial,
        )
        return initial_tangents

    def _initial_tangent_directions(self, params, *, profile_prefix: str):
        """Return cached identity directions used for dense initial-state JVPs."""
        from ._compat import jnp as _jnp

        if not hasattr(self, "_initial_tangent_direction_cache"):
            self._initial_tangent_direction_cache = {}
        dtype = _jnp.asarray(params).dtype
        backend = _optimizer_backend_name(getattr(self, "_solver_device_name", None))
        cache_key = (int(_jnp.asarray(params).size), str(dtype), backend)
        directions = self._initial_tangent_direction_cache.get(cache_key)
        if directions is not None:
            self._profile_add(f"{profile_prefix}_initial_tangents_eye_cache_hit", 0.0)
            return directions

        self._profile_add(f"{profile_prefix}_initial_tangents_eye_cache_miss", 0.0)
        t_eye = time.perf_counter()
        directions = _jnp.eye(cache_key[0], dtype=dtype)
        self._initial_tangent_direction_cache[cache_key] = directions
        self._profile_add(f"{profile_prefix}_initial_tangents_eye", time.perf_counter() - t_eye)
        return directions

    def _lasym_replay_column_chunk(self, n_params: int) -> int | None:
        """Replay-column chunk heuristic for dense exact Jacobians."""

        env_override = os.environ.get("VMEC_JAX_LASYM_REPLAY_COLUMN_CHUNK")
        if env_override is not None:
            from .discrete_adjoint import _replay_column_chunk_override

            handled, requested = _replay_column_chunk_override(env_override)
            if handled:
                return requested
        if os.environ.get("VMEC_JAX_REPLAY_COLUMN_CHUNK") is not None:
            return None
        backend_name = None
        if self._solver_device_name is not None:
            backend_name = str(self._solver_device_name).lower()
        else:
            try:
                from ._compat import jax as _jax

                backend_name = str(_jax.default_backend()).lower()
            except Exception:
                backend_name = None
        if backend_name in ("gpu", "cuda", "rocm"):
            # GPU exact replay is launch/transpose dominated.  Office A4000
            # profiles on JAX 0.6.2 showed 8-column replay chunks reduce both
            # QH mode-2 24-DOF and QH mode-3 48-DOF cold/new-point callback
            # wall time materially. Explicit environment overrides remain
            # authoritative.
            if int(n_params) >= 24:
                return 8
            return None
        if backend_name == "tpu":
            return None
        if not bool(getattr(self._static.cfg, "lasym", False)):
            return None
        if int(n_params) >= 64:
            return 8
        if int(n_params) >= 32:
            return 4
        return None

    def _projected_replay_residuals_enabled(self, n_params: int | None = None) -> bool:
        """Whether dense Jacobians should project replayed tangents without an intermediate sync."""

        flag = os.getenv("VMEC_JAX_OPT_PROJECTED_REPLAY_RESIDUALS")
        if flag is not None:
            return flag.strip().lower() in ("1", "true", "yes", "on")
        backend = _optimizer_backend_name(getattr(self, "_solver_device_name", None))
        if backend not in ("gpu", "cuda", "rocm"):
            return False
        if n_params is None:
            return False
        if bool(getattr(getattr(self._static, "cfg", None), "lasym", False)):
            return False
        # May 2026 office profiles now show a small but repeatable win for
        # non-LASYM QH mode-2 (24 columns) and larger wins for mode-3+ because
        # the residual projection avoids an intermediate tangent synchronization.
        # LASYM mode-2 profiles also have 48+ columns, but projected replay is
        # currently much slower there, so keep LASYM on the conservative path.
        return int(n_params) >= 24

    def _fused_projected_replay_enabled(self) -> bool:
        """Whether projected replay should fuse replay and residual projection when possible."""

        flag = os.getenv("VMEC_JAX_OPT_FUSED_PROJECTED_REPLAY", "").strip().lower()
        if flag:
            return flag in ("1", "true", "yes", "on")
        # Office GPU profiles on 2026-05-28 show the fused mode-2 QH callback
        # is slower than the regular projected replay path. Keep fusion opt-in
        # for diagnostics until a broader matrix shows a reproducible win.
        return False

    def _discrete_jacobian_residual_helper(self, params_size: int, residuals_from_packed, *, jax):
        """Return cached residual/Jacobian projection helper for packed tangents."""

        helper_key = (
            int(params_size),
            int(self._layout.size),
            id(self._residuals_fn),
        )
        t_helper = time.perf_counter()
        helper_cache = self._discrete_jacobian_helper_cache.get(helper_key)
        if helper_cache is None:

            @jax.jit
            def _residual_tangent_jacobian(packed_state, packed_tangents):
                residuals, residual_linear = jax.linearize(residuals_from_packed, packed_state)
                # Keep the residual Jacobian transpose on device. Materializing
                # columns on the host and transposing there is especially costly
                # for GPU exact callbacks.
                return residuals, jax.vmap(residual_linear)(packed_tangents).T

            helper_cache = {
                "residual_tangent_jacobian": _residual_tangent_jacobian,
            }
            self._discrete_jacobian_helper_cache[helper_key] = helper_cache
            self._profile_add("jacobian_residual_tangent_helper_build", time.perf_counter() - t_helper)
        else:
            self._profile_add("jacobian_residual_tangent_helper_cache_hit", time.perf_counter() - t_helper)
        return helper_cache

    def _fused_dynamic_basepoint_projected_replay_helper(
        self,
        *,
        tape,
        params_size: int,
        residuals_from_packed,
        initial_tangents,
        column_chunk: int | None,
        jax,
    ):
        """Return a fused dynamic-basepoint replay/projection helper when eligible."""

        if column_chunk is not None or not self._fused_projected_replay_enabled():
            return None
        stacked = getattr(tape, "stacked_step_traces", None)
        stacked_base_carries = getattr(tape, "dynamic_base_carries_stacked", None)
        static_flags = getattr(tape, "step_trace_static_flags", None)
        if stacked is None or stacked_base_carries is None or static_flags is None:
            return None

        from .discrete_adjoint import (
            _checkpoint_tape_dynamic_basepoint_scan_runner,
            _dynamic_basepoint_payload_shapes_match,
            _replay_column_chunk_default,
            _tridi_policy_cache_value,
            _stacked_trace_signature,
        )

        if not _dynamic_basepoint_payload_shapes_match(stacked, stacked_base_carries):
            return None
        # The fused path intentionally bypasses checkpoint_tape_state_jvp_columns,
        # so defer to the standard replay path whenever explicit or automatic
        # chunking would be active.
        if os.environ.get("VMEC_JAX_REPLAY_COLUMN_CHUNK") is not None:
            return None
        from ._compat import jnp as _jnp

        auto_chunk = _replay_column_chunk_default(
            tape=tape,
            tangents=_jnp.asarray(initial_tangents),
        )
        if auto_chunk is not None and int(params_size) > int(auto_chunk):
            return None

        helper_key = (
            "fused_dynamic_basepoint_projected_replay",
            int(params_size),
            int(self._layout.size),
            id(self._residuals_fn),
            id(self._static),
            _stacked_trace_signature(stacked),
            _stacked_trace_signature(stacked_base_carries),
            bool(static_flags["apply_lforbal"]),
            bool(static_flags["include_edge_residual"]),
            bool(static_flags["apply_m1_constraints"]),
            bool(static_flags["limit_update_rms"]),
            bool(static_flags["limit_dt_from_force"]),
            bool(static_flags["vmec2000_control"]),
            bool(static_flags["divide_by_scalxc_for_update"]),
            int(static_flags["signgs"]),
            int(static_flags["precond_jmax"]),
            _tridi_policy_cache_value(static_flags.get("preconditioner_use_precomputed_tridi", None)),
            _tridi_policy_cache_value(static_flags.get("preconditioner_use_lax_tridi", None)),
        )
        t_helper = time.perf_counter()
        helper_cache = self._discrete_jacobian_helper_cache.get(helper_key)
        if helper_cache is not None:
            self._profile_add(
                "jacobian_fused_projected_replay_helper_cache_hit",
                time.perf_counter() - t_helper,
            )
            return helper_cache

        run_scan = _checkpoint_tape_dynamic_basepoint_scan_runner(
            static=self._static,
            stacked=stacked,
            stacked_base_carries=stacked_base_carries,
            static_flags=static_flags,
        )

        @jax.jit
        def _fused_project(initial_tangents, packed_state, stacked_base_carries_in, stacked_traces_in):
            tangents = _jnp.asarray(initial_tangents)
            carry0 = jax.tree_util.tree_map(lambda x: x[0], stacked_base_carries_in)

            def _zeros_like(arr):
                arr = _jnp.asarray(arr)
                return _jnp.zeros((tangents.shape[0],) + arr.shape, dtype=arr.dtype)

            carry_tangents0 = (tangents,) + tuple(_zeros_like(arr) for arr in carry0[1:])
            final_carry_tangents = run_scan(
                carry_tangents0,
                stacked_base_carries_in,
                stacked_traces_in,
            )
            residuals, residual_linear = jax.linearize(residuals_from_packed, packed_state)
            return residuals, jax.vmap(residual_linear)(final_carry_tangents[0]).T

        helper_cache = {
            "fused_project": _fused_project,
        }
        self._discrete_jacobian_helper_cache[helper_key] = helper_cache
        self._profile_add("jacobian_fused_projected_replay_helper_build", time.perf_counter() - t_helper)
        return helper_cache

    def _jacobian_fun_projected_replay(self, params, exact_param_key, *, t_total: float) -> np.ndarray:
        """Dense exact Jacobian path that avoids synchronizing full state tangents."""

        from ._compat import jax, jnp as _jnp
        from .discrete_adjoint import checkpoint_tape_state_jvp_columns
        from .state import pack_state, unpack_state

        state, payload = self._solve_exact_with_tape_for_jvp(params)
        packed_final = _jnp.asarray(pack_state(state), dtype=_jnp.float64)

        def _residuals_from_packed(packed):
            return self._residuals_fn(unpack_state(packed, self._layout))

        if int(params.size) == 0:
            helper_cache = self._discrete_jacobian_residual_helper(
                int(params.size),
                _residuals_from_packed,
                jax=jax,
            )
            residuals = helper_cache["residual_tangent_jacobian"](
                packed_final,
                _jnp.zeros((0, int(self._layout.size)), dtype=_jnp.float64),
            )[0]
            residuals = jax.block_until_ready(residuals)
            self._last_jacobian_residual = np.asarray(residuals, dtype=float)
            self._remember_exact_residual(exact_param_key, self._last_jacobian_residual)
            out = np.zeros((int(self._last_jacobian_residual.size), 0), dtype=float)
            self._remember_exact_jacobian(exact_param_key, out, self._last_jacobian_residual)
            self._profile_add("jacobian_total", time.perf_counter() - t_total)
            return out

        initial_tangents = self._initial_tangent_columns(
            params,
            payload["axis_override"],
            profile_prefix="jacobian",
        )
        column_chunk = self._lasym_replay_column_chunk(int(params.size))
        if column_chunk is not None:
            self._profile_add(f"jacobian_projected_replay_column_chunk_{column_chunk}", 0.0)
        fused_helper = self._fused_dynamic_basepoint_projected_replay_helper(
            tape=payload["tape"],
            params_size=int(params.size),
            residuals_from_packed=_residuals_from_packed,
            initial_tangents=initial_tangents,
            column_chunk=column_chunk,
            jax=jax,
        )
        if fused_helper is not None:
            t_replay = time.perf_counter()
            residuals, jac = fused_helper["fused_project"](
                initial_tangents,
                packed_final,
                payload["tape"].dynamic_base_carries_stacked,
                payload["tape"].stacked_step_traces,
            )
            residuals, jac = self._profile_blocking_phase(
                "jacobian_fused_projected_replay_total",
                t_replay,
                (residuals, jac),
            )
            self._last_jacobian_residual = np.asarray(residuals, dtype=float)
            self._remember_exact_residual(exact_param_key, self._last_jacobian_residual)
            t_host = time.perf_counter()
            out = np.asarray(jac, dtype=float)
            self._profile_add("jacobian_host_materialize", time.perf_counter() - t_host)
            self._remember_exact_jacobian(exact_param_key, out, self._last_jacobian_residual)
            self._last_jacobian_source = "exact_tape_fused_projected_replay"
            self._profile_add("jacobian_total", time.perf_counter() - t_total)
            return out
        helper_cache = self._discrete_jacobian_residual_helper(
            int(params.size),
            _residuals_from_packed,
            jax=jax,
        )
        t_replay = time.perf_counter()
        final_tangents = checkpoint_tape_state_jvp_columns(
            tape=payload["tape"],
            static=self._static,
            initial_tangents=initial_tangents,
            rebuild_preconditioner=True,
            column_chunk=column_chunk,
        )
        # Intentionally do not block here.  Let the residual projection consume
        # the device value and block once after projection so GPU callbacks avoid
        # an extra host synchronization between replay and residual tangents.
        self._profile_add("jacobian_projected_tape_replay_dispatch", time.perf_counter() - t_replay)

        t_res = time.perf_counter()
        residuals, jac = helper_cache["residual_tangent_jacobian"](packed_final, final_tangents)
        residuals, jac = self._profile_blocking_phase(
            "jacobian_projected_replay_residual_tangents",
            t_res,
            (residuals, jac),
        )
        self._profile_add("jacobian_projected_replay_total", time.perf_counter() - t_replay)
        self._last_jacobian_residual = np.asarray(residuals, dtype=float)
        self._remember_exact_residual(exact_param_key, self._last_jacobian_residual)
        t_host = time.perf_counter()
        out = np.asarray(jac, dtype=float)
        self._profile_add("jacobian_host_materialize", time.perf_counter() - t_host)
        self._remember_exact_jacobian(exact_param_key, out, self._last_jacobian_residual)
        self._last_jacobian_source = "exact_tape_projected_replay"
        self._profile_add("jacobian_total", time.perf_counter() - t_total)
        return out

    def jacobian_fun(self, params) -> np.ndarray:
        """Exact discrete-adjoint Jacobian at *params*."""
        if self._solver_device_name is not None and not self._inside_solver_device_context:
            return self._run_in_solver_device_context(self.jacobian_fun, params)
        self._last_jacobian_source = "exact_tape_replay"
        exact_param_key = self._exact_cache_key(params)
        if self._scan_exact_path == "scan":
            from ._compat import jnp as _jnp

            helpers = self._scan_exact_helpers()
            t0 = time.perf_counter()
            residuals, jac = helpers["residual_and_jacobian"](_jnp.asarray(params, dtype=_jnp.float64))
            self._last_jacobian_residual = np.asarray(residuals, dtype=float)
            self._remember_exact_residual(exact_param_key, self._last_jacobian_residual)
            # Avoid a second accepted-point scan solve when the history metrics
            # can be reconstructed from the residual vector.  This is the common
            # QA/QH/QP/QI fixed-boundary optimization path; the state cache is
            # still populated for custom residuals or iota-tracked histories.
            if not self._can_build_history_from_residuals():
                self._solve_scan_exact_state(params)
            out = np.asarray(jac, dtype=float)
            self._last_jacobian_source = "scan_exact_replay"
            self._profile_add("scan_jacobian_total", time.perf_counter() - t0)
            return out
        from ._compat import jax, jnp as _jnp
        from .state import pack_state, unpack_state

        t_total = time.perf_counter()
        cached_jacobian = getattr(self, "_exact_jacobian_cache", {}).get(exact_param_key)
        if cached_jacobian is not None:
            jac_cached, residual_cached = cached_jacobian
            self._last_jacobian_residual = np.asarray(residual_cached, dtype=float).reshape(-1)
            self._remember_exact_residual(exact_param_key, self._last_jacobian_residual)
            self._last_jacobian_source = "jacobian_cache_hit"
            self._profile_add("jacobian_cache_hit", 0.0)
            self._profile_add("jacobian_total", time.perf_counter() - t_total)
            return np.asarray(jac_cached, dtype=float).copy()

        params = _jnp.asarray(params, dtype=_jnp.float64)
        if self._projected_replay_residuals_enabled(int(params.size)):
            return self._jacobian_fun_projected_replay(params, exact_param_key, t_total=t_total)

        state, final_tangents = self._state_and_tangent_columns(
            params,
            profile_prefix="jacobian",
        )
        packed_final = _jnp.asarray(pack_state(state), dtype=_jnp.float64)

        def _residuals_from_packed(packed):
            return self._residuals_fn(unpack_state(packed, self._layout))

        helper_cache = self._discrete_jacobian_residual_helper(
            int(params.size),
            _residuals_from_packed,
            jax=jax,
        )

        t_res = time.perf_counter()
        residuals, jac = helper_cache["residual_tangent_jacobian"](packed_final, final_tangents)
        residuals, jac = self._profile_blocking_phase(
            "jacobian_residual_tangents",
            t_res,
            (residuals, jac),
        )
        self._last_jacobian_residual = np.asarray(residuals, dtype=float)
        self._remember_exact_residual(exact_param_key, self._last_jacobian_residual)
        t_host = time.perf_counter()
        out = np.asarray(jac, dtype=float)
        self._profile_add("jacobian_host_materialize", time.perf_counter() - t_host)
        self._remember_exact_jacobian(exact_param_key, out, self._last_jacobian_residual)
        self._last_jacobian_source = "exact_tape_replay"
        self._profile_add("jacobian_total", time.perf_counter() - t_total)
        return out

    def state_tangent_columns_fun(self, params) -> tuple[VMECState, np.ndarray]:
        """Return the accepted-point state and packed state tangent columns.

        The tangent columns use the same frozen-axis initial-state convention
        and checkpoint tape replay as :meth:`jacobian_fun`. The returned array
        has shape ``(n_parameters, state.layout.size)``.
        """
        if self._solver_device_name is not None and not self._inside_solver_device_context:
            return self._run_in_solver_device_context(
                self.state_tangent_columns_fun,
                params,
            )

        t_total = time.perf_counter()
        state, final_tangents = self._state_and_tangent_columns(
            params,
            profile_prefix="state_tangent",
        )
        out = np.asarray(final_tangents, dtype=float)
        self._profile_add("state_tangent_columns_total", time.perf_counter() - t_total)
        return state, out

    def b_cartesian_tangent_columns_fun(
        self,
        params,
        static: VMECStatic | None = None,
        *,
        s_index: int = -1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return boundary Cartesian field values and exact tangent columns.

        ``static`` supplies the angular grid for the field evaluation. If it is
        omitted, the optimizer's solve grid is used. The field has shape
        ``(ntheta, nzeta, 3)`` and the tangent columns have shape
        ``(ntheta, nzeta, 3, n_parameters)``.
        """
        if self._solver_device_name is not None and not self._inside_solver_device_context:
            return self._run_in_solver_device_context(
                self.b_cartesian_tangent_columns_fun,
                params,
                static,
                s_index=s_index,
            )
        from ._compat import jax, jnp as _jnp
        from .field import b_cartesian_from_state
        from .state import pack_state, unpack_state

        if static is None:
            static = self._static
        params = _jnp.asarray(params, dtype=_jnp.float64)
        state, state_tangents = self._state_and_tangent_columns(
            params,
            profile_prefix="b_cartesian_tangent",
        )
        packed_final = _jnp.asarray(pack_state(state), dtype=_jnp.float64)

        def _field_from_packed(packed):
            state_arg = unpack_state(packed, self._layout)
            field = b_cartesian_from_state(
                state_arg,
                static,
                indata=self._indata,
                signgs=self._signgs,
                s_index=s_index,
            )
            return _jnp.ravel(field)

        field_flat, field_linear = jax.linearize(_field_from_packed, packed_final)
        nparams = int(params.size)
        if nparams == 0:
            columns = _jnp.zeros((0, field_flat.size), dtype=field_flat.dtype)
        else:
            columns = jax.vmap(field_linear)(state_tangents)

        ntheta = int(static.grid.ntheta)
        nzeta = int(static.grid.nzeta)
        field = np.asarray(field_flat).reshape((ntheta, nzeta, 3))
        tangent_columns = np.asarray(columns, dtype=float).reshape((nparams, ntheta, nzeta, 3))
        tangent_columns = np.transpose(tangent_columns, (1, 2, 3, 0))
        return field, tangent_columns

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
        from .state import pack_state, unpack_state

        t_total = time.perf_counter()
        params = _jnp.asarray(params, dtype=_jnp.float64)
        state, payload = self._solve_exact_with_tape(params, return_payload=True)
        tape = payload["tape"]
        axis_override = payload["axis_override"]
        packed_final = _jnp.asarray(pack_state(state), dtype=_jnp.float64)

        def _residuals_from_packed(packed):
            return self._residuals_fn(unpack_state(packed, self._layout))

        t_res_vjp = time.perf_counter()
        objective_cotangent_factory = getattr(
            self._residuals_fn,
            "_state_objective_value_and_cotangent_from_packed",
            None,
        )
        if objective_cotangent_factory is not None:
            helper_key = (
                "objective_value_and_cotangent",
                int(self._layout.size),
                id(self._residuals_fn),
            )
            helper_cache = self._discrete_jacobian_helper_cache.get(helper_key)
            if helper_cache is None:

                @jax.jit
                def _objective_value_and_cotangent_helper(packed_state_arg):
                    return objective_cotangent_factory(packed_state_arg, self._layout)

                helper_cache = {"objective_value_and_cotangent": _objective_value_and_cotangent_helper}
                self._discrete_jacobian_helper_cache[helper_key] = helper_cache
            try:
                cost, final_cotangent = helper_cache["objective_value_and_cotangent"](packed_final)
            except getattr(jax.errors, "TracerArrayConversionError", Exception):
                # Custom/test hooks may contain host-side NumPy assertions or
                # diagnostics. Keep the fast jitted helper for pure-JAX hooks,
                # but fall back to the Python callable when newer JAX refuses
                # host conversion during tracing.
                helper_cache["objective_value_and_cotangent"] = objective_cotangent_factory
                cost, final_cotangent = objective_cotangent_factory(packed_final, self._layout)
        else:
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
        initial_cotangent = self._profile_async_phase(
            "gradient_tape_replay",
            t_replay,
            initial_cotangent,
        )
        initial_cotangent = _jnp.nan_to_num(initial_cotangent, nan=0.0, posinf=0.0, neginf=0.0)

        t_initial = time.perf_counter()
        cache_key = None
        initial_tangents = None
        try:
            cache_key = self._initial_tangent_cache_key(params)
            initial_tangents = self._initial_tangent_cache.get(cache_key) if cache_key is not None else None
        except Exception:
            cache_key = None
            initial_tangents = None
        if initial_tangents is not None:
            self._profile_add("gradient_initial_tangents_cache_hit", 0.0)
            grad = _jnp.tensordot(
                _jnp.asarray(initial_tangents, dtype=_jnp.float64),
                _jnp.asarray(initial_cotangent, dtype=_jnp.float64),
                axes=([1], [0]),
            )
            self._profile_add("gradient_initial_projection", time.perf_counter() - t_initial)
        else:
            # For a fixed axis/flip branch the initial-state map is affine, so
            # the same cache key used for tangent columns is valid for its VJP.
            initial_vjp_key = None if cache_key is None else ("gradient_initial_vjp", cache_key)
            initial_vjp = (
                self._discrete_jacobian_helper_cache.get(initial_vjp_key)
                if initial_vjp_key is not None
                else None
            )
            if initial_vjp is not None:
                self._profile_add("gradient_initial_vjp_cache_hit", 0.0)
            else:
                from .init_guess import initial_guess_from_boundary as _ig

                axis_override = {
                    key: _jnp.asarray(value, dtype=params.dtype) for key, value in axis_override.items()
                }

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

                _, initial_vjp = jax.vjp(_initial_state_packed, params)
                if initial_vjp_key is not None:
                    self._discrete_jacobian_helper_cache[initial_vjp_key] = initial_vjp
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
        from .state import pack_state, unpack_state

        t_total = time.perf_counter()
        params = _jnp.asarray(params, dtype=_jnp.float64)
        state, payload = self._solve_exact_with_tape(params, return_payload=True)
        tape = payload["tape"]
        axis_override = {
            key: _jnp.asarray(value, dtype=params.dtype) for key, value in payload["axis_override"].items()
        }
        packed_final = _jnp.asarray(pack_state(state), dtype=_jnp.float64)

        def _residuals_from_packed(packed):
            return self._residuals_fn(unpack_state(packed, self._layout))

        t_setup = time.perf_counter()
        initial_tangent_cache_key = None
        initial_tangent_columns = None
        try:
            initial_tangent_cache_key = self._initial_tangent_cache_key(params)
            initial_tangent_columns = (
                self._initial_tangent_cache.get(initial_tangent_cache_key)
                if initial_tangent_cache_key is not None
                else None
            )
        except Exception:
            initial_tangent_cache_key = None
            initial_tangent_columns = None
        if initial_tangent_columns is not None:
            initial_tangent_columns = _jnp.asarray(initial_tangent_columns, dtype=_jnp.float64)
            initial_linear = None
            initial_transpose = None
            self._profile_add("linear_operator_initial_tangents_cache_hit", 0.0)
        else:
            from .init_guess import initial_guess_from_boundary as _ig

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

            _, initial_linear = jax.linearize(_initial_state_packed, params)
            initial_transpose = jax.linear_transpose(initial_linear, params)
            self._profile_add("linear_operator_initial_tangents_cache_miss", 0.0)
        residuals, residual_linear = jax.linearize(_residuals_from_packed, packed_final)
        state_cotangent_from_packed = getattr(self._residuals_fn, "_state_cotangent_from_packed", None)
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
                    return state_cotangent_from_packed(packed_state_arg, self._layout, cotangent_arg)

                helper_cache = {"residual_cotangent": _residual_cotangent_helper}
                self._discrete_jacobian_helper_cache[residual_cotangent_key] = helper_cache
            residual_cotangent_helper = helper_cache["residual_cotangent"]
        residual_vjp = None
        if state_cotangent_from_packed is None:
            _, residual_vjp = jax.vjp(_residuals_from_packed, packed_final)
        residuals_np = np.asarray(residuals, dtype=float)
        self._remember_exact_residual(self._exact_cache_key(params), residuals_np)
        self._profile_add("linear_operator_setup", time.perf_counter() - t_setup)

        n_res = int(residuals_np.size)
        n_params = int(params.size)

        def _matvec(direction):
            t_mv = time.perf_counter()
            direction_j = _jnp.asarray(
                _linear_operator_vector_arg(direction, size=n_params, name="matvec direction"),
                dtype=params.dtype,
            )
            if initial_tangent_columns is not None:
                initial_tangent = _jnp.tensordot(direction_j, initial_tangent_columns, axes=([0], [0]))
            else:
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
            directions_arr = _linear_operator_matrix_arg(
                directions,
                rows=n_params,
                name="matmat directions",
            )
            directions_j = _jnp.asarray(directions_arr.T, dtype=params.dtype)
            if initial_tangent_columns is not None:
                initial_tangents = _jnp.tensordot(directions_j, initial_tangent_columns, axes=([1], [0]))
            else:
                initial_tangents = jax.vmap(initial_linear)(directions_j)
            final_tangents = checkpoint_tape_state_jvp_columns(
                tape=tape,
                static=self._static,
                initial_tangents=initial_tangents,
                rebuild_preconditioner=True,
                column_chunk=self._lasym_replay_column_chunk(int(directions_j.shape[0])),
            )
            out_columns = jax.vmap(residual_linear)(final_tangents)
            self._profile_add("linear_operator_matmat", time.perf_counter() - t_mm)
            return np.asarray(out_columns, dtype=float).T

        def _rmatvec(cotangent):
            t_rmv = time.perf_counter()
            cotangent_j = _jnp.asarray(
                _linear_operator_vector_arg(cotangent, size=n_res, name="rmatvec cotangent"),
                dtype=_jnp.float64,
            )
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
            initial_cotangent = self._profile_async_phase(
                "linear_operator_tape_vjp",
                t_tape_vjp,
                initial_cotangent,
            )
            initial_cotangent = _jnp.nan_to_num(initial_cotangent, nan=0.0, posinf=0.0, neginf=0.0)
            t_initial_transpose = time.perf_counter()
            # The frozen-axis initial-state map is linear for a fixed flip
            # branch. Reuse cached tangent columns when available; otherwise
            # reuse the transpose of the setup linearization instead of tracing
            # a second VJP through the same initialization graph.
            if initial_tangent_columns is not None:
                grad = _jnp.tensordot(initial_tangent_columns, initial_cotangent, axes=([1], [0]))
            else:
                grad = initial_transpose(_jnp.asarray(initial_cotangent, dtype=_jnp.float64))[0]
            self._profile_add("linear_operator_initial_transpose", time.perf_counter() - t_initial_transpose)
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
        self._last_jacobian_residual = None
        self._last_jacobian_source = "exact_tape_replay"
        jac = self.jacobian_fun(params)
        key = self._last_jacobian_key[0]
        exact_residual = (
            np.asarray(self._last_jacobian_residual, dtype=float)
            if self._last_jacobian_residual is not None
            else self._cached_exact_residual(cache_key=key)
        )
        cached_state = None
        if self._last_jacobian_residual is not None and self._can_build_history_from_residuals():
            entry = self._history_entry_from_residuals(
                self._last_jacobian_residual,
                wall_time_s=time.perf_counter() - self._wall_t0,
            )
        elif self._scan_exact_path == "scan" and key is not None and key in self._exact_state_cache:
            cached_state = self._exact_state_cache[key]
            entry = self._history_entry_from_state_or_residual(
                cached_state,
                exact_residual,
                wall_time_s=time.perf_counter() - self._wall_t0,
                cache_key=key,
            )
        elif key is not None and key in self._exact_cache:
            cached_state, _ = self._exact_cache[key]
            entry = self._history_entry_from_state_or_residual(
                cached_state,
                exact_residual,
                wall_time_s=time.perf_counter() - self._wall_t0,
                cache_key=key,
            )
        else:
            entry = None
        if entry is not None and self._exact_history_accepts(float(entry["cost"])):
            self._history.append(entry)
            if exact_residual is not None:
                self._remember_best_exact_point(params, exact_residual, float(entry["cost"]), state=cached_state)
        elif entry is not None:
            self._exact_history_rejected_count += 1
        elif exact_residual is not None:
            self._remember_best_exact_point(params, exact_residual)
        return jac

    def _exact_residual_after_jacobian(self):
        key = self._last_jacobian_key[0]
        if key is None and self._last_jacobian_residual is not None:
            return np.asarray(self._last_jacobian_residual, dtype=float)
        cached_residual = self._cached_exact_residual(cache_key=key)
        if cached_residual is not None:
            return cached_residual
        if key is None or key not in self._exact_cache:
            return None
        cached_state, _ = self._exact_cache[key]
        residual = self._evaluate_residuals_from_state(cached_state)
        self._remember_exact_residual(key, residual)
        return residual

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
        if hasattr(self, "_exact_state_key_by_id"):
            self._exact_state_key_by_id.clear()
        if hasattr(self, "_exact_residual_cache"):
            self._exact_residual_cache.clear()
        if hasattr(self, "_exact_jacobian_cache"):
            self._exact_jacobian_cache.clear()
        self._trial_residual_cache.clear()
        if hasattr(self, "_initial_state_cache"):
            self._initial_state_cache.clear()
        self._initial_state_packed_helper = None
        self._initial_tangent_cache.clear()
        if hasattr(self, "_initial_tangent_direction_cache"):
            self._initial_tangent_direction_cache.clear()
        if hasattr(self, "_discrete_jacobian_helper_cache"):
            self._discrete_jacobian_helper_cache.clear()
        if hasattr(self, "_scan_exact_helper_cache"):
            self._scan_exact_helper_cache.clear()
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
        return float(np.asarray(equilibrium_aspect_ratio_from_state(state=state, static=self._static)))

    def _qs_from_res(self, res: np.ndarray) -> float:
        """Sum of squared QS residuals only (excludes aspect and iota)."""
        n_qs = getattr(self, "_n_qs", None)
        if n_qs is not None:
            return float(np.dot(res[-n_qs:], res[-n_qs:]))
        start = max(0, min(int(getattr(self, "_n_non_qs", 1)), int(res.shape[0])))
        return float(np.dot(res[start:], res[start:]))

    def _has_qs_residual_block_metadata(self) -> bool:
        flag = getattr(self, "_has_residual_block_metadata", None)
        if flag is not None:
            return bool(flag)
        return hasattr(self, "_n_non_qs") or getattr(self, "_n_qs", None) is not None

    def _can_build_qs_from_residuals(self) -> bool:
        """Return true when residual block metadata identifies QS/objective blocks."""
        return self._has_qs_residual_block_metadata()

    def _can_build_aspect_from_residuals(self) -> bool:
        """Return true when the first residual encodes weighted aspect error."""
        if getattr(self, "_aspect_target", None) is None:
            return False
        aspect_weight = float(getattr(self, "_aspect_weight", 1.0))
        if not np.isfinite(aspect_weight) or aspect_weight == 0.0:
            return False
        return True

    def _can_build_history_from_residuals(self) -> bool:
        """Return true when residual metadata is enough for history metrics."""
        if getattr(self, "_iota_fn", None) is not None:
            return False
        if not self._can_build_aspect_from_residuals():
            return False
        if not self._can_build_qs_from_residuals():
            return False
        return True

    def _history_entry_from_residuals(self, res: np.ndarray, *, wall_time_s: float) -> dict:
        """Build a history row without re-solving the accepted scan state."""
        res = np.asarray(res, dtype=float).reshape(-1)
        aspect = float(self._aspect_target) + float(res[0]) / float(self._aspect_weight)
        cost = float(0.5 * np.dot(res, res))
        return {
            "wall_time_s": float(wall_time_s),
            "cost": cost,
            "objective": 2.0 * cost,
            "qs_objective": self._qs_from_res(res),
            "aspect": aspect,
        }

    def _qs_total_from_residual_or_state(
        self,
        state: VMECState,
        res: np.ndarray | None = None,
    ) -> float:
        """Use residual block metadata for QS totals before expensive state callbacks."""
        if res is not None and self._can_build_qs_from_residuals():
            return self._qs_from_res(np.asarray(res, dtype=float).reshape(-1))
        return self._qs_total_from_state(state, res)

    def _history_entry_from_state_or_residual(
        self,
        state: VMECState,
        res: np.ndarray | None = None,
        *,
        wall_time_s: float,
        cost: float | None = None,
        cache_key: bytes | None = None,
    ) -> dict:
        """Build a history row, preferring exact cached residual data when safe."""
        res_arr = None if res is None else np.asarray(res, dtype=float).reshape(-1)
        if res_arr is not None and self._can_build_history_from_residuals():
            return self._history_entry_from_residuals(res_arr, wall_time_s=wall_time_s)

        if res_arr is None:
            res_arr = self._evaluate_residuals_from_state(state)
            if cache_key is not None:
                self._remember_exact_residual(cache_key, res_arr)

        cost_val = float(0.5 * np.dot(res_arr, res_arr)) if cost is None else float(cost)
        if self._can_build_aspect_from_residuals():
            aspect = float(getattr(self, "_aspect_target")) + float(res_arr[0]) / float(
                getattr(self, "_aspect_weight", 1.0)
            )
        else:
            from .wout import equilibrium_aspect_ratio_from_state

            aspect = float(np.asarray(equilibrium_aspect_ratio_from_state(state=state, static=self._static)))

        entry: dict = {
            "wall_time_s": float(wall_time_s),
            "cost": cost_val,
            "objective": 2.0 * cost_val,
            "qs_objective": self._qs_total_from_residual_or_state(state, res_arr),
            "aspect": aspect,
        }
        iota_fn = getattr(self, "_iota_fn", None)
        if iota_fn is not None:
            entry["iota"] = float(iota_fn(state))
        return entry

    def _qs_total_from_state(self, state: VMECState, res: np.ndarray | None = None) -> float:
        """QS-only objective from a solved state, with metadata-aware fallback."""
        if self._qs_total_from_state_fn is not None:
            return float(self._qs_total_from_state_fn(state))
        if res is not None:
            return self._qs_from_res(np.asarray(res, dtype=float))
        res = self._evaluate_residuals_from_state(state)
        return self._qs_from_res(np.asarray(res, dtype=float))

    def quasisymmetry_objective(self, params) -> float:
        """Return the total QS objective at *params*."""
        state = (
            self._solve_scan_exact_state(params)
            if self._scan_exact_path == "scan"
            else self._solve_exact_with_tape(params)
        )
        res = self._evaluate_residuals_from_state(state)
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
            Already-solved exact VMEC state to write. Passing this avoids
            rerunning the equilibrium solve when the optimizer can verify that
            the state was solved for ``params``.

        Notes
        -----
        Uses the exact-solve cache when *params* was previously evaluated.  On
        a cache miss the accepted-point solver settings are used; the relaxed
        trial solver is never used for persisted wout artifacts.
        """
        t0 = time.perf_counter()
        from .driver import FixedBoundaryRun
        from .driver import write_wout_from_fixed_boundary_run

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if state is None:
            if params is None:
                raise ValueError("save_wout requires either params or state")
            state = self._cached_exact_state(params)
            if state is None:
                state = self._solve_forward(params, trial=False)
                self._remember_exact_state(self._exact_cache_key(params), state)
        elif params is not None and not self._state_matches_params(state, params):
            cached_state = self._cached_exact_state(params)
            if cached_state is not None:
                state = cached_state
            else:
                state = self._solve_forward(params, trial=False)
                self._remember_exact_state(self._exact_cache_key(params), state)
        run = FixedBoundaryRun(
            cfg=self._static.cfg,
            indata=self._indata,
            static=self._static,
            state=state,
            result=None,
            flux=self._flux if hasattr(self._flux, "chipf") else None,
            profiles=None,
            signgs=self._signgs,
        )
        write_wout_from_fixed_boundary_run(str(path), run, include_fsq=False, fast_bcovar=True)
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
        lbfgs_step_bound: float | None = 0.01,
        scalar_step_bound: float | None = 0.01,
        scalar_cost_only_trials: bool | None = None,
        trace_callbacks: bool | None = None,
    ) -> dict:
        """Run exact least-squares optimisation.

        Parameters
        ----------
        params0:
            Initial parameter vector (usually ``np.zeros(len(specs))``).
        method:
            Outer least-squares method. Supported values are ``"gauss_newton"``
            and ``"scipy"``. ``"auto"`` keeps the current device selection and
            resolves to a conservative device-preserving method for known cases:
            currently matrix-free SciPy for high-mode, stellarator-symmetric
            QS/QI on CPU/default CPU, otherwise dense SciPy. This is an opt-in
            policy and not a guarantee that every warm run is fastest.
            ``"scipy"`` uses ``scipy.optimize.least_squares``
            with the exact residual and discrete-adjoint Jacobian callbacks,
            which is more robust on some QA/QH examples.
            ``"scipy_matrix_free"`` uses the same SciPy trust-region solver
            with a matrix-free exact ``LinearOperator`` Jacobian.  It applies
            ``Jv`` and ``J.Tv`` products by replaying the converged VMEC tape
            without materializing the dense Jacobian. ``"lbfgs_adjoint"``
            minimizes the same scalar objective using one reverse discrete
            adjoint gradient per callback; it is experimental but scales much
            better with boundary-parameter count on mode-2/3 diagnostics.
            ``"scalar_trust"`` is a safeguarded scalar-adjoint path with
            monotone accepted steps, limited-memory inverse-Hessian directions,
            aggressive backtracking, and a hard evaluation budget.  It is
            intended for profiling high-parameter-count cases before a full
            matrix-free least-squares trust-region implementation is available.
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
        lbfgs_step_bound:
            Optional half-width of the L-BFGS-B trust box in scaled parameter
            space when ``method="lbfgs_adjoint"``. The scalar-adjoint path is
            not a least-squares trust-region method; this bound prevents the
            line search from probing extremely distorted boundaries. Set to
            ``None`` or a non-positive value to run unbounded L-BFGS-B.
        scalar_step_bound:
            Initial and maximum trust radius in scaled parameter space when
            ``method="scalar_trust"``. Set to ``None`` or a non-positive value
            to use a unit initial radius.
        scalar_cost_only_trials:
            When true with ``method="scalar_trust"``, evaluate trial points with
            the lighter forward residual path before building a full exact
            scalar-adjoint tape for accepted candidates.  This can reduce
            accepted-point tape builds in rugged high-mode cases, at the cost
            of additional forward solves. ``None`` preserves the legacy
            environment/private-attribute controls for profiling scripts.
        trace_callbacks:
            When true, include a lightweight SciPy callback trace in the
            history dump.  This is intended for CPU/GPU profiling of repeated
            trial residuals, exact-state cache hits, and accepted-point
            Jacobian replay.  ``None`` enables tracing only when
            ``VMEC_JAX_OPT_TRACE_CALLBACKS`` is set to a truthy value.

        Returns
        -------
        dict
            Result dict from :func:`gauss_newton_least_squares` extended with
            ``_history_dump`` (the full per-iteration history suitable for
            :meth:`save_history`).
        """
        self._history = []
        self._profile = {}
        self._trial_residual_cache.clear()
        if not hasattr(self, "_exact_jacobian_cache"):
            self._exact_jacobian_cache = {}
        else:
            self._exact_jacobian_cache.clear()
        self._callback_trace_enabled = (
            os.getenv("VMEC_JAX_OPT_TRACE_CALLBACKS", "").strip().lower() in ("1", "true", "yes", "on")
            if trace_callbacks is None
            else bool(trace_callbacks)
        )
        self._callback_trace = []
        self._callback_point_ids = {}
        self._callback_previous_key = None
        self._wall_t0 = time.perf_counter()
        self._iota_fn = iota_fn  # stored so _jacobian_fun_tracked can use it
        self._best_exact_params = None
        self._best_exact_state = None
        self._best_exact_residual = None
        self._best_exact_cost = math.inf
        self._exact_history_rejected_count = 0

        params0_arr = np.asarray(params0, dtype=float)
        scalar_cost_only_trials_used: bool | None = None

        # ── initial evaluation ──────────────────────────────────────────────
        res0 = self.residual_fun(params0_arr)
        if self._scan_exact_path == "scan":
            state0 = self._solve_scan_exact_state(params0_arr)
        else:
            state0, _ = self._solve_exact_with_tape(params0_arr, return_payload=True)
        entry0 = self._history_entry_from_state_or_residual(
            state0,
            res0,
            wall_time_s=0.0,
            cache_key=self._exact_cache_key(params0_arr),
        )
        cost0 = float(entry0["cost"])
        qs_total0 = float(entry0["qs_objective"])
        aspect0 = float(entry0["aspect"])
        self._history.append(entry0)
        self._remember_best_exact_point(params0_arr, res0, cost0, state=state0)

        # ── outer least-squares loop ────────────────────────────────────────
        method_requested = str(method).strip().lower().replace("-", "_")
        method_key, scipy_lsmr_maxiter, method_auto_reason = self._resolve_optimizer_method(
            method_requested,
            scipy_lsmr_maxiter,
        )
        if method_auto_reason is not None:
            self._profile_add(f"method_auto_{method_key}", 0.0)
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
        elif method_key in ("scalar_trust", "adjoint_trust", "gradient_trust"):
            scale = np.ones_like(params0_arr) if x_scale is None else np.asarray(x_scale, dtype=float)
            scale[scale == 0.0] = 1.0
            base_params = self._base_params_vector()
            y_current = (params0_arr + base_params) / scale
            x_current = params0_arr.copy()
            last_history_key = [self._exact_cache_key(params0_arr)]
            max_scalar_evals = max(1, int(max_nfev))
            initial_radius = (
                1.0 if scalar_step_bound is None or float(scalar_step_bound) <= 0.0 else float(scalar_step_bound)
            )
            radius = initial_radius
            min_radius = max(1.0e-12, initial_radius * 1.0e-8)
            eval_count = 0
            accepted_steps = 0
            best_eval: dict[str, object] = {
                "cost": float("inf"),
                "x": x_current.copy(),
                "y": y_current.copy(),
                "state": None,
                "grad_x": np.zeros_like(params0_arr),
                "grad_y": np.zeros_like(y_current),
            }

            def _record_history_from_cached_state(x, cost):
                key = self._exact_cache_key(x)
                if key == last_history_key[0] or key not in self._exact_cache:
                    return
                cached_state, _ = self._exact_cache[key]
                exact_residual = self._cached_exact_residual(cache_key=key)
                entry = self._history_entry_from_state_or_residual(
                    cached_state,
                    exact_residual,
                    wall_time_s=time.perf_counter() - self._wall_t0,
                    cost=float(cost),
                    cache_key=key,
                )
                entry_cost = float(entry["cost"])
                if self._exact_history_accepts(entry_cost):
                    self._history.append(entry)
                    if exact_residual is None:
                        exact_residual = self._cached_exact_residual(cache_key=key)
                    if exact_residual is not None:
                        self._remember_best_exact_point(x, exact_residual, entry_cost, state=cached_state)
                    last_history_key[0] = key
                else:
                    self._exact_history_rejected_count += 1

            def _evaluate_y(y):
                nonlocal eval_count
                eval_count += 1
                x = np.asarray(y, dtype=float) * scale - base_params
                cost, grad_x = self.objective_and_gradient_fun(x)
                grad_y = np.asarray(grad_x, dtype=float) * scale
                if float(cost) < float(best_eval["cost"]):
                    best_state = self._cached_exact_state(x)
                    best_eval.update(
                        {
                            "cost": float(cost),
                            "x": np.asarray(x, dtype=float).copy(),
                            "y": np.asarray(y, dtype=float).copy(),
                            "state": best_state,
                            "grad_x": np.asarray(grad_x, dtype=float).copy(),
                            "grad_y": grad_y.copy(),
                        }
                    )
                return float(cost), np.asarray(x, dtype=float), grad_y

            def _trial_cost_y(y):
                x = np.asarray(y, dtype=float) * scale - base_params
                t0 = time.perf_counter()
                residual = np.asarray(self.forward_residual_fun(x), dtype=float).reshape(-1)
                cost = 0.5 * float(np.dot(residual, residual))
                self._profile_add("scalar_trust_cost_only_trial", time.perf_counter() - t0)
                return cost, x

            cost_current, x_current, grad_y = _evaluate_y(y_current)
            grad_norm = float(np.linalg.norm(grad_y, ord=np.inf))
            success_result = bool(grad_norm <= float(gtol))
            status_result = 1 if success_result else 0
            message_result = (
                "`gtol` termination condition is satisfied."
                if success_result
                else "maximum number of scalar objective evaluations is exceeded"
            )
            lbfgs_pairs: list[tuple[np.ndarray, np.ndarray, float]] = []
            max_lbfgs_pairs = 8
            armijo_c1 = 1.0e-4
            backtrack_factor = 0.1
            if scalar_cost_only_trials is None:
                cost_only_trial_flag = os.getenv("VMEC_JAX_OPT_SCALAR_COST_ONLY_TRIALS")
                cost_only_trials = (
                    bool(getattr(self, "_scalar_trust_cost_only_trials", False))
                    if cost_only_trial_flag is None
                    else cost_only_trial_flag.strip().lower() in ("1", "true", "yes", "on")
                )
            else:
                cost_only_trials = bool(scalar_cost_only_trials)
            scalar_cost_only_trials_used = bool(cost_only_trials)

            def _scalar_trust_direction(grad):
                grad = np.asarray(grad, dtype=float)
                if not lbfgs_pairs:
                    self._profile_add("scalar_trust_gradient_direction", 0.0)
                    return -grad

                q = grad.copy()
                alphas: list[float] = []
                for s_vec, y_vec, rho in reversed(lbfgs_pairs):
                    alpha = float(rho * np.dot(s_vec, q))
                    alphas.append(alpha)
                    q = q - alpha * y_vec

                s_last, y_last, _rho_last = lbfgs_pairs[-1]
                yy_last = float(np.dot(y_last, y_last))
                sy_last = float(np.dot(s_last, y_last))
                h0 = sy_last / yy_last if yy_last > 0.0 and sy_last > 0.0 else 1.0
                h0 = min(1.0e6, max(1.0e-12, h0))
                r = h0 * q
                for (s_vec, y_vec, rho), alpha in zip(lbfgs_pairs, reversed(alphas)):
                    beta = float(rho * np.dot(y_vec, r))
                    r = r + s_vec * (alpha - beta)

                direction = -r
                if (
                    not np.all(np.isfinite(direction))
                    or float(np.dot(direction, grad)) >= -1.0e-14 * max(1.0, float(np.linalg.norm(grad) ** 2))
                ):
                    self._profile_add("scalar_trust_gradient_direction", 0.0)
                    return -grad
                self._profile_add("scalar_trust_lbfgs_direction", 0.0)
                return direction

            while not success_result and eval_count < max_scalar_evals:
                grad_norm_2 = float(np.linalg.norm(grad_y))
                if not np.isfinite(grad_norm_2) or grad_norm_2 <= 0.0:
                    message_result = "zero or non-finite scalar-adjoint gradient"
                    break

                direction_y = _scalar_trust_direction(grad_y)
                direction_norm = float(np.linalg.norm(direction_y))
                if not np.isfinite(direction_norm) or direction_norm <= 0.0:
                    message_result = "zero or non-finite scalar-adjoint search direction"
                    break
                base_step_y = direction_y * min(1.0, radius / direction_norm)
                directional_decrease = -float(np.dot(grad_y, base_step_y))
                if directional_decrease <= 0.0 or not np.isfinite(directional_decrease):
                    base_step_y = -grad_y * min(1.0, radius / grad_norm_2)
                    directional_decrease = -float(np.dot(grad_y, base_step_y))
                    lbfgs_pairs.clear()

                accepted = False
                shrink = 1.0
                while eval_count < max_scalar_evals:
                    step_y = shrink * base_step_y
                    if float(np.linalg.norm(step_y)) < min_radius:
                        break
                    y_trial = y_current + step_y
                    armijo_limit = cost_current - armijo_c1 * shrink * max(0.0, directional_decrease)
                    if cost_only_trials:
                        cost_trial_estimate, x_trial = _trial_cost_y(y_trial)
                        passes_trial_filter = np.isfinite(cost_trial_estimate) and (
                            cost_trial_estimate <= armijo_limit or cost_trial_estimate < cost_current
                        )
                        if not passes_trial_filter:
                            self._profile_add("scalar_trust_rejected_step", 0.0)
                            shrink *= backtrack_factor
                            continue
                        cost_trial, x_trial, grad_trial = _evaluate_y(y_trial)
                        if not (
                            np.isfinite(cost_trial)
                            and (cost_trial <= armijo_limit or cost_trial < cost_current)
                        ):
                            self._profile_add("scalar_trust_exact_validation_rejected_step", 0.0)
                            shrink *= backtrack_factor
                            continue
                    else:
                        cost_trial, x_trial, grad_trial = _evaluate_y(y_trial)
                    if np.isfinite(cost_trial) and (
                        cost_trial <= armijo_limit or cost_trial < cost_current
                    ):
                        y_current = y_trial
                        x_current = x_trial
                        cost_previous = cost_current
                        cost_current = cost_trial
                        step_accepted = np.asarray(step_y, dtype=float)
                        grad_delta = np.asarray(grad_trial, dtype=float) - np.asarray(grad_y, dtype=float)
                        grad_y = grad_trial
                        grad_norm = float(np.linalg.norm(grad_y, ord=np.inf))
                        accepted_steps += 1
                        accepted = True
                        sy = float(np.dot(step_accepted, grad_delta))
                        curvature_floor = 1.0e-12 * max(
                            1.0,
                            float(np.linalg.norm(step_accepted)) * float(np.linalg.norm(grad_delta)),
                        )
                        if sy > curvature_floor and np.all(np.isfinite(grad_delta)):
                            lbfgs_pairs.append((step_accepted, grad_delta, 1.0 / sy))
                            if len(lbfgs_pairs) > max_lbfgs_pairs:
                                del lbfgs_pairs[0]
                        _record_history_from_cached_state(x_current, cost_current)
                        step_norm = float(np.linalg.norm(step_accepted))
                        if shrink < 1.0:
                            # Backtracking is a local globalization choice, not
                            # evidence that future quasi-Newton directions need
                            # a permanently tiny trust radius.  Re-expand to the
                            # previous rejected scale so the next accepted-point
                            # callback can still make useful progress.
                            radius = min(
                                initial_radius,
                                max(2.0 * step_norm, step_norm / backtrack_factor),
                            )
                            self._profile_add("scalar_trust_backtracked_accept", 0.0)
                        elif step_norm >= 0.8 * radius:
                            radius = min(initial_radius, max(radius * 1.5, radius))
                        else:
                            radius = min(initial_radius, max(radius, 2.0 * step_norm))
                        if abs(cost_previous - cost_current) <= float(ftol) * max(1.0, abs(cost_current)):
                            success_result = True
                            status_result = 2
                            message_result = "`ftol` termination condition is satisfied."
                        elif grad_norm <= float(gtol):
                            success_result = True
                            status_result = 1
                            message_result = "`gtol` termination condition is satisfied."
                        break
                    self._profile_add("scalar_trust_rejected_step", 0.0)
                    shrink *= backtrack_factor

                if success_result:
                    break
                if not accepted:
                    message_result = "scalar trust-region radius became too small"
                    break

            if not success_result and eval_count >= max_scalar_evals:
                message_result = "maximum number of scalar objective evaluations is exceeded"

            x_result = np.asarray(best_eval["x"], dtype=float)
            best_state = best_eval.get("state")
            if best_state is not None:
                self._remember_exact_state(self._exact_cache_key(x_result), best_state)
            cost_result = float(best_eval["cost"])
            result = {
                "x": x_result,
                "cost": cost_result,
                "objective": 2.0 * cost_result,
                "nfev": int(eval_count),
                "njev": int(eval_count),
                "nit": int(accepted_steps),
                "success": success_result,
                "status": status_result,
                "message": message_result,
                "step_norm": float(np.linalg.norm(x_result - params0_arr)),
                "x_prev": None,
                "cost_prev": None,
            }
        elif method_key in ("lbfgs", "lbfgs_adjoint"):
            try:
                from scipy.optimize import minimize as _scipy_minimize
            except Exception as exc:  # pragma: no cover - optional dependency
                raise ImportError("method='lbfgs_adjoint' requires scipy.optimize.minimize") from exc

            scale = np.ones_like(params0_arr) if x_scale is None else np.asarray(x_scale, dtype=float)
            scale[scale == 0.0] = 1.0
            base_params = self._base_params_vector()
            y0 = (params0_arr + base_params) / scale
            last_history_key = [self._exact_cache_key(params0_arr)]
            max_scalar_evals = max(1, int(max_nfev))
            eval_count = [0]
            best_eval: dict[str, object] = {
                "cost": float("inf"),
                "x": params0_arr.copy(),
                "y": y0.copy(),
                "grad_x": np.zeros_like(params0_arr),
                "grad_y": np.zeros_like(y0),
            }

            class _LBFGSBudgetExceeded(RuntimeError):
                pass

            def _record_history_from_cached_state(x, cost):
                key = self._exact_cache_key(x)
                if key == last_history_key[0] or key not in self._exact_cache:
                    return
                cached_state, _ = self._exact_cache[key]
                exact_residual = self._cached_exact_residual(cache_key=key)
                entry = self._history_entry_from_state_or_residual(
                    cached_state,
                    exact_residual,
                    wall_time_s=time.perf_counter() - self._wall_t0,
                    cost=float(cost),
                    cache_key=key,
                )
                entry_cost = float(entry["cost"])
                if self._exact_history_accepts(entry_cost):
                    self._history.append(entry)
                    if exact_residual is None:
                        exact_residual = self._cached_exact_residual(cache_key=key)
                    if exact_residual is not None:
                        self._remember_best_exact_point(x, exact_residual, entry_cost, state=cached_state)
                    last_history_key[0] = key
                else:
                    self._exact_history_rejected_count += 1

            def _objective_and_gradient_y(y):
                if eval_count[0] >= max_scalar_evals:
                    raise _LBFGSBudgetExceeded
                eval_count[0] += 1
                x = np.asarray(y, dtype=float) * scale - base_params
                cost, grad_x = self.objective_and_gradient_fun(x)
                grad_y = np.asarray(grad_x, dtype=float) * scale
                if float(cost) < float(best_eval["cost"]):
                    best_eval.update(
                        {
                            "cost": float(cost),
                            "x": np.asarray(x, dtype=float).copy(),
                            "y": np.asarray(y, dtype=float).copy(),
                            "grad_x": np.asarray(grad_x, dtype=float).copy(),
                            "grad_y": grad_y.copy(),
                        }
                    )
                _record_history_from_cached_state(x, cost)
                return float(cost), grad_y

            try:
                lbfgs_bounds = None
                if lbfgs_step_bound is not None and float(lbfgs_step_bound) > 0.0:
                    bound = float(lbfgs_step_bound)
                    lbfgs_bounds = [
                        (float(center) - bound, float(center) + bound) for center in np.asarray(y0, dtype=float)
                    ]
                minimize_result = _scipy_minimize(
                    _objective_and_gradient_y,
                    y0,
                    jac=True,
                    method="L-BFGS-B",
                    bounds=lbfgs_bounds,
                    options={
                        "maxiter": int(max_nfev),
                        "maxfun": int(max_nfev),
                        "ftol": float(ftol),
                        "gtol": float(gtol),
                        "disp": bool(int(verbose) > 0),
                    },
                )
                x_result = np.asarray(minimize_result.x, dtype=float) * scale - base_params
                cost_result = float(minimize_result.fun)
                success_result = bool(minimize_result.success)
                status_result = int(minimize_result.status)
                message_result = str(minimize_result.message)
                nit_result = int(getattr(minimize_result, "nit", 0))
            except _LBFGSBudgetExceeded:
                x_result = np.asarray(best_eval["x"], dtype=float)
                cost_result = float(best_eval["cost"])
                success_result = False
                status_result = 0
                message_result = "maximum number of scalar objective evaluations is exceeded"
                nit_result = 0
            result = {
                "x": x_result,
                "cost": cost_result,
                "objective": 2.0 * cost_result,
                "nfev": int(eval_count[0]),
                "njev": int(eval_count[0]),
                "nit": nit_result,
                "success": success_result,
                "status": status_result,
                "message": message_result,
                "step_norm": 0.0,
                "x_prev": None,
                "cost_prev": None,
            }
        elif method_key in ("scipy_matrix_free", "matrix_free", "scipy_mf"):
            try:
                from scipy.optimize import least_squares as _scipy_least_squares
            except Exception as exc:  # pragma: no cover - optional dependency
                raise ImportError("method='scipy_matrix_free' requires scipy.optimize.least_squares") from exc

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
                exact_residual = self._cached_exact_residual(cache_key=key)
                entry = self._history_entry_from_state_or_residual(
                    cached_state,
                    exact_residual,
                    wall_time_s=time.perf_counter() - self._wall_t0,
                    cache_key=key,
                )
                entry_cost = float(entry["cost"])
                if self._exact_history_accepts(entry_cost):
                    self._history.append(entry)
                    if exact_residual is None:
                        exact_residual = self._cached_exact_residual(cache_key=key)
                    if exact_residual is not None:
                        self._remember_best_exact_point(x, exact_residual, entry_cost, state=cached_state)
                    last_history_key[0] = key
                else:
                    self._exact_history_rejected_count += 1

            def _residuals_y(y):
                x = np.asarray(y, dtype=float) * scale - base_params
                cached_residual = self._cached_exact_residual(x)
                if cached_residual is not None:
                    return cached_residual
                cached_state = self._cached_exact_state(x)
                if cached_state is not None:
                    return self._evaluate_residuals_from_state(cached_state)
                return self.forward_residual_fun(x)

            def _jacobian_y(y):
                x = np.asarray(y, dtype=float) * scale - base_params
                op_x = self.residual_linear_operator(x)
                _record_history_from_cached_state(x)

                def _matvec(v):
                    v_arr = _linear_operator_vector_arg(v, size=int(scale.size), name="scaled matvec direction")
                    return op_x.matvec(v_arr * scale)

                def _matmat(v):
                    v_arr = _linear_operator_matrix_arg(v, rows=int(scale.size), name="scaled matmat directions")
                    return op_x.matmat(v_arr * scale[:, None])

                def _rmatvec(w):
                    w_arr = _linear_operator_vector_arg(w, size=int(op_x.shape[0]), name="scaled rmatvec cotangent")
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

            try:
                scipy_result = _scipy_least_squares(
                    _residuals_y,
                    y0,
                    jac=_jacobian_y,
                    method="trf",
                    tr_solver="lsmr",
                    tr_options=({"maxiter": int(scipy_lsmr_maxiter)} if scipy_lsmr_maxiter is not None else None),
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
            except Exception as exc:
                best_exact_params = getattr(self, "_best_exact_params", None)
                best_exact_cost = float(getattr(self, "_best_exact_cost", math.inf))
                if best_exact_params is None or not np.isfinite(best_exact_cost):
                    raise
                x_result = np.asarray(best_exact_params, dtype=float).copy()
                result = {
                    "x": x_result,
                    "cost": best_exact_cost,
                    "objective": 2.0 * best_exact_cost,
                    "nfev": max(1, len(getattr(self, "_history", []))),
                    "njev": max(0, len(getattr(self, "_history", [])) - 1),
                    "nit": 0,
                    "success": False,
                    "status": -1,
                    "message": f"scipy matrix-free least_squares failed; returning best exact accepted point: {exc}",
                    "step_norm": float(np.linalg.norm(x_result - params0_arr)),
                    "x_prev": None,
                    "cost_prev": None,
                    "_selected_best_exact_point": True,
                    "_optimizer_exception": repr(exc),
                }
        elif method_key == "scipy":
            try:
                from scipy.optimize import least_squares as _scipy_least_squares
            except Exception as exc:  # pragma: no cover - optional dependency
                raise ImportError("method='scipy' requires scipy.optimize.least_squares") from exc

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
                t_cb = time.perf_counter()
                cache_key = self._exact_cache_key(x)
                cached_residual = self._cached_exact_residual(cache_key=cache_key)
                if cached_residual is not None:
                    self._trace_callback_event(
                        "residual",
                        x,
                        source="exact_residual_cache",
                        wall_time_s=time.perf_counter() - t_cb,
                    )
                    return cached_residual
                cached_state = self._cached_exact_state(x)
                if cached_state is not None:
                    out = self._evaluate_residuals_from_state(cached_state)
                    self._trace_callback_event(
                        "residual",
                        x,
                        source="exact_state_cache",
                        wall_time_s=time.perf_counter() - t_cb,
                    )
                    return out
                cached_trial = self._cached_trial_residual(x)
                if cached_trial is not None:
                    self._trace_callback_event(
                        "residual",
                        x,
                        source="trial_residual_cache",
                        wall_time_s=time.perf_counter() - t_cb,
                    )
                    return cached_trial
                # Residual-only callbacks do not need an adjoint tape. Building one
                # for every SciPy trial point bloats memory badly on converged QA/QH
                # runs. Keep the Jacobian exact, but evaluate residuals through the
                # converged forward solve only.
                out = _forward_residual_exact(x)
                self._trace_callback_event(
                    "residual",
                    x,
                    source="trial_solve",
                    wall_time_s=time.perf_counter() - t_cb,
                )
                return out

            def _jacobian_y(y):
                x = np.asarray(y, dtype=float) * scale - base_params
                t_cb = time.perf_counter()
                self._last_jacobian_source = "exact_tape_replay"
                jac = np.asarray(self._jacobian_fun_tracked(x), dtype=float) * scale[None, :]
                self._trace_callback_event(
                    "jacobian",
                    x,
                    source=getattr(self, "_last_jacobian_source", "exact_tape_replay"),
                    wall_time_s=time.perf_counter() - t_cb,
                )
                # SciPy residual callbacks above no longer consume the exact-tape cache.
                # Drop the retained tape immediately after the Jacobian/history entry is
                # materialized, otherwise later converged QA iterations keep a multi-GB
                # exact tape alive between callbacks and get killed by RSS.
                self._exact_cache.clear()
                return jac

            try:
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
            except Exception as exc:
                best_exact_params = getattr(self, "_best_exact_params", None)
                best_exact_cost = float(getattr(self, "_best_exact_cost", math.inf))
                if best_exact_params is None or not np.isfinite(best_exact_cost):
                    raise
                x_result = np.asarray(best_exact_params, dtype=float).copy()
                result = {
                    "x": x_result,
                    "cost": best_exact_cost,
                    "objective": 2.0 * best_exact_cost,
                    "nfev": max(1, len(getattr(self, "_history", []))),
                    "njev": max(0, len(getattr(self, "_history", [])) - 1),
                    "nit": 0,
                    "success": False,
                    "status": -1,
                    "message": f"scipy least_squares failed; returning best exact accepted point: {exc}",
                    "step_norm": float(np.linalg.norm(x_result - params0_arr)),
                    "x_prev": None,
                    "cost_prev": None,
                    "_selected_best_exact_point": True,
                    "_optimizer_exception": repr(exc),
                }
        else:
            raise ValueError(f"Unknown optimization method '{method}'.")
        result["method"] = method_key
        result["method_requested"] = method_requested
        result["method_auto_reason"] = method_auto_reason
        self._post_jacobian_clear()

        # ── final evaluation ────────────────────────────────────────────────
        # Use the exact cache when available (avoids a fresh full VMEC solve
        # that can OOM after a long optimization session).  If the optimizer's
        # final point cannot be exactly replayed, prefer a prior exact accepted
        # point; never use a relaxed trial solve for final artifacts.
        selected_best_exact = bool(result.pop("_selected_best_exact_point", False))
        optimizer_exception = result.pop("_optimizer_exception", None)
        best_exact_params = getattr(self, "_best_exact_params", None)
        best_exact_state = getattr(self, "_best_exact_state", None)
        best_exact_residual = getattr(self, "_best_exact_residual", None)
        best_exact_cost = float(getattr(self, "_best_exact_cost", math.inf))

        final_key = self._exact_cache_key(result["x"])
        res_final = self._cached_exact_residual(cache_key=final_key)
        if (
            res_final is None
            and best_exact_params is not None
            and best_exact_residual is not None
            and final_key == self._exact_cache_key(best_exact_params)
        ):
            res_final = np.asarray(best_exact_residual, dtype=float).reshape(-1)
            self._remember_exact_residual(final_key, res_final)
        state_final = self._cached_exact_state(result["x"])
        if state_final is None:
            try:
                state_final = (
                    self._solve_scan_exact_state(result["x"])
                    if self._scan_exact_path == "scan"
                    else self._solve_exact_with_tape(result["x"])
                )
            except Exception as exc:
                if (
                    best_exact_params is not None
                    and best_exact_residual is not None
                    and np.isfinite(best_exact_cost)
                ):
                    selected_best_exact = True
                    result["x"] = np.asarray(best_exact_params, dtype=float).copy()
                    final_key = self._exact_cache_key(result["x"])
                    res_final = np.asarray(best_exact_residual, dtype=float).reshape(-1)
                    state_final = self._cached_exact_state(result["x"])
                    if state_final is None:
                        state_final = best_exact_state
                    if state_final is None:
                        state_final = (
                            self._solve_scan_exact_state(result["x"])
                            if self._scan_exact_path == "scan"
                            else self._solve_exact_with_tape(result["x"])
                        )
                else:
                    raise RuntimeError(
                        "Final exact accepted-point solve failed and no prior exact "
                        "accepted point is available for final output."
                    ) from exc

        if state_final is not None:
            self._remember_exact_state(final_key, state_final)

        final_wall_time_s = time.perf_counter() - self._wall_t0
        if self._history:
            final_wall_time_s = max(final_wall_time_s, float(self._history[-1].get("wall_time_s", 0.0)))
        entry_final = self._history_entry_from_state_or_residual(
            state_final,
            res_final,
            wall_time_s=final_wall_time_s,
            cache_key=final_key,
        )
        cost_final = float(entry_final["cost"])
        qs_total_final = float(entry_final["qs_objective"])
        aspect_final = float(entry_final["aspect"])

        exact_improvement_tol = max(
            1.0e-14,
            1.0e-9
            * max(
                1.0,
                abs(cost_final) if np.isfinite(cost_final) else 1.0,
                abs(best_exact_cost) if np.isfinite(best_exact_cost) else 1.0,
            ),
        )
        if (
            best_exact_params is not None
            and best_exact_residual is not None
            and np.isfinite(best_exact_cost)
            and (not np.isfinite(cost_final) or best_exact_cost < cost_final - exact_improvement_tol)
        ):
            selected_best_exact = True
            result["x"] = np.asarray(best_exact_params, dtype=float).copy()
            final_key = self._exact_cache_key(result["x"])
            res_final = np.asarray(best_exact_residual, dtype=float).reshape(-1)
            state_final = self._cached_exact_state(result["x"])
            if state_final is None:
                state_final = best_exact_state
            if state_final is None:
                try:
                    state_final = (
                        self._solve_scan_exact_state(result["x"])
                        if self._scan_exact_path == "scan"
                        else self._solve_exact_with_tape(result["x"])
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "Best exact accepted point was selected for final output, "
                        "but its exact state could not be reconstructed."
                    ) from exc
            final_wall_time_s = time.perf_counter() - self._wall_t0
            if self._history:
                final_wall_time_s = max(final_wall_time_s, float(self._history[-1].get("wall_time_s", 0.0)))
            entry_final = self._history_entry_from_state_or_residual(
                state_final,
                res_final,
                wall_time_s=final_wall_time_s,
                cache_key=final_key,
            )
            cost_final = float(entry_final["cost"])
            qs_total_final = float(entry_final["qs_objective"])
            aspect_final = float(entry_final["aspect"])

        if state_final is not None:
            self._remember_exact_state(final_key, state_final)

        result["cost"] = float(cost_final)
        result["objective"] = float(2.0 * cost_final)
        self._history.append(entry_final)

        # ── assemble history dump ───────────────────────────────────────────
        history_dump: dict = {
            "label": "Optimisation",
            "max_nfev": max_nfev,
            "ftol": ftol,
            "gtol": gtol,
            "xtol": xtol,
            "method": method_key,
            "method_requested": method_requested,
            "method_auto_reason": method_auto_reason,
            "exact_path": self._scan_exact_path,
            "scipy_tr_solver": (
                scipy_tr_solver
                if method_key == "scipy"
                else "lsmr"
                if method_key in ("scipy_matrix_free", "matrix_free", "scipy_mf")
                else None
            ),
            "scipy_lsmr_maxiter": (None if scipy_lsmr_maxiter is None else int(scipy_lsmr_maxiter)),
            "lbfgs_step_bound": (None if lbfgs_step_bound is None else float(lbfgs_step_bound)),
            "scalar_step_bound": (None if scalar_step_bound is None else float(scalar_step_bound)),
            "scalar_cost_only_trials": scalar_cost_only_trials_used,
            "solver_device": self._solver_device_name or "default",
            "inner_max_iter": int(self._inner_max_iter),
            "inner_ftol": float(self._inner_ftol),
            "trial_max_iter": int(self._trial_max_iter),
            "trial_ftol": float(self._trial_ftol),
            "total_wall_time_s": final_wall_time_s,
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
            "selected_best_exact_point": bool(selected_best_exact),
            "rejected_trial_exact_history_count": int(self._exact_history_rejected_count),
        }
        if optimizer_exception is not None:
            history_dump["optimizer_exception"] = str(optimizer_exception)
        if iota_fn is not None:
            history_dump["iota_initial"] = float(entry0["iota"])
            history_dump["iota_final"] = float(entry_final["iota"])
        if target_iota is not None:
            history_dump["target_iota"] = float(target_iota)
        if target_aspect is not None:
            history_dump["target_aspect"] = float(target_aspect)
        if self._callback_trace_enabled:
            history_dump["callback_trace"] = self._callback_trace_dump()

        # Private, non-serializable convenience payload for scripts that want
        # to write wout files without rerunning the VMEC solve immediately after
        # optimization. save_history() only persists `_history_dump`.
        result["_state_initial"] = state0
        result["_state_final"] = state_final
        result["_profile"] = self._profile_dump()
        result["_history_dump"] = history_dump
        return result
