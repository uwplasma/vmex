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
from .optimizers.fixed_boundary.linear_guards import finite_linear_operator_output
from .optimizers.fixed_boundary.linear_guards import linear_operator_matrix_arg
from .optimizers.fixed_boundary.linear_guards import linear_operator_vector_arg
from .optimizers.fixed_boundary.history import ResidualHistoryPolicy
from .optimizers.fixed_boundary.history import build_run_history_dump
from .optimizers.fixed_boundary.history import history_entry_from_residuals
from .optimizers.fixed_boundary.history import monotone_final_wall_time
from .optimizers.fixed_boundary.history import qs_objective_from_residuals
from .optimizers.fixed_boundary.gauss_newton import gauss_newton_least_squares
from .optimizers.fixed_boundary.matrix_free import build_residual_linear_operator
from .optimizers.fixed_boundary.parameterization import BoundaryParamSpec
from .optimizers.fixed_boundary.parameterization import apply_boundary_params
from .optimizers.fixed_boundary.parameterization import apply_boundary_params_numpy
from .optimizers.fixed_boundary.parameterization import boundary_param_names
from .optimizers.fixed_boundary.parameterization import boundary_param_specs
from .optimizers.fixed_boundary.parameterization import coeff_label
from .optimizers.fixed_boundary.parameterization import create_x_scale
from .optimizers.fixed_boundary.parameterization import extend_boundary_for_max_mode
from .optimizers.fixed_boundary.parameterization import indexed_boundary_maps_from_boundary
from .optimizers.fixed_boundary.parameterization import lift_boundary_params
from .optimizers.fixed_boundary.parameterization import rebuild_indata_with_resolution
from .optimizers.fixed_boundary.parameterization import truncate_indata_boundary_modes
from .optimizers.fixed_boundary.scalar_gradient import exact_objective_and_gradient
from .optimizers.fixed_boundary.scalar_lbfgs import run_lbfgs_adjoint_exact_optimizer
from .optimizers.fixed_boundary.scalar_trust import run_scalar_trust_exact_optimizer
from .optimizers.fixed_boundary.scipy_least_squares import run_scipy_dense_exact_optimizer
from .optimizers.fixed_boundary.scipy_least_squares import run_scipy_matrix_free_exact_optimizer
from .profiles import eval_profiles
from .state import VMECState
from .static import VMECStatic

# Backwards-compatible private helper names used by older tests and profiling
# scripts.  Implementations live in optimizers.fixed_boundary.linear_guards.
_finite_linear_operator_output = finite_linear_operator_output
_linear_operator_matrix_arg = linear_operator_matrix_arg
_linear_operator_vector_arg = linear_operator_vector_arg
_apply_boundary_params_numpy = apply_boundary_params_numpy
_coeff_label = coeff_label
_indexed_boundary_maps_from_boundary = indexed_boundary_maps_from_boundary

__all__ = [
    "BoundaryParamSpec",
    "FixedBoundaryContext",
    "FixedBoundaryExactOptimizer",
    "apply_boundary_params",
    "boundary_param_names",
    "boundary_param_specs",
    "create_x_scale",
    "extend_boundary_for_max_mode",
    "gauss_newton_least_squares",
    "lift_boundary_params",
    "prepare_fixed_boundary_context",
    "rebuild_indata_with_resolution",
    "smooth_min_abs_iota_residual",
    "truncate_indata_boundary_modes",
]


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
    exact_path:
        Accepted-point differentiation path. ``None`` / ``"auto"`` keeps the
        default tape path and honors ``VMEC_JAX_OPT_EXACT_PATH``. Pass
        ``"tape"`` for the low-cold-cost discrete-adjoint tape path, or
        ``"scan"`` for the high-compile-cost scan-differentiated path that can
        be faster for long GPU runs after compilation is amortized.
    freeze_initial_axis:
        Reuse the initial magnetic-axis branch from the optimizer base point in
        residual and exact-replay callbacks.  This is useful for optimization
        stages whose input deck omits explicit axis coefficients because VMEC's
        inferred-axis search contains nonsmooth extrema/branch choices.

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
        exact_path: str | None = None,
        freeze_initial_axis: bool = False,
    ) -> None:
        self._solver_device_name = self._resolve_solver_device(solver_device)
        self._exact_path_request = self._resolve_exact_path_request(exact_path)
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
        self._freeze_initial_axis = bool(freeze_initial_axis)
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
        if self._freeze_initial_axis:
            from .init_guess import extract_axis_override_from_state

            self._initial_axis_override = extract_axis_override_from_state(state0, static)
        else:
            self._initial_axis_override = None

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

    def _resolve_exact_path_request(self, exact_path: str | None) -> str | None:
        """Validate the optional accepted-point differentiation path request."""

        if exact_path is None:
            return None
        name = str(exact_path).strip().lower().replace("-", "_")
        if name in ("", "none", "auto", "default"):
            return None
        if name not in ("tape", "scan"):
            raise ValueError("exact_path must be one of None, 'auto', 'tape', or 'scan'")
        return name

    def _spec_max_mode(self) -> int:
        if not self._specs:
            return 0
        return max(max(abs(int(spec.m)), abs(int(spec.n))) for spec in self._specs)

    def _has_stellarator_asymmetric_parameter_specs(self) -> bool:
        return any(str(spec.kind).lower() in ("rs", "zc") for spec in getattr(self, "_specs", ()))

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
        stellarator-symmetric QS CPU/default-backend lanes where cold-process
        and memory-pressure profiles motivated the option. QI currently stays
        on dense SciPy unless matrix-free is requested explicitly because QI
        Boozer/bounce residual JVPs can be non-finite in cleanup stages. It does
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
        scalar_auto_requested = method_key in ("auto_scalar", "auto_adjoint", "adaptive_scalar", "adaptive_adjoint")
        if method_key not in ("auto", "adaptive") and not scalar_auto_requested:
            return method_key, scipy_lsmr_maxiter, None

        if self._has_stellarator_asymmetric_configuration():
            prefix = "auto_scalar" if scalar_auto_requested else "auto"
            return "scipy", scipy_lsmr_maxiter, f"{prefix}:dense-lasym"

        backend = _optimizer_backend_name(getattr(self, "_solver_device_name", None))
        helicity_m = None if self._helicity_m is None else int(self._helicity_m)
        helicity_n = None if self._helicity_n is None else int(self._helicity_n)
        if self._spec_max_mode() >= 3 and self._objective_family in ("qs", "qi"):
            if scalar_auto_requested:
                suffix = f"{backend}-" if backend in ("gpu", "cuda", "rocm", "tpu", "metal") else ""
                return "scalar_trust", scipy_lsmr_maxiter, f"auto_scalar:{suffix}high-mode-scalar-trust"
            if backend in ("gpu", "cuda", "rocm", "tpu", "metal"):
                return "scipy", scipy_lsmr_maxiter, f"auto:dense-preserves-{backend}"
            if self._objective_family == "qi":
                return "scipy", scipy_lsmr_maxiter, "auto:qi-dense-default"
            lsmr_maxiter = 4 if scipy_lsmr_maxiter is None else scipy_lsmr_maxiter
            if helicity_m == 1 and helicity_n == 0:
                family = "qa"
            elif helicity_m == 0 and helicity_n not in (None, 0):
                family = "qp"
            elif helicity_m == 1 and helicity_n not in (None, 0):
                family = "qh"
            else:
                family = "qs"
            return "scipy_matrix_free", lsmr_maxiter, f"auto:{family}-high-mode-matrix-free"

        if backend in ("gpu", "cuda", "rocm", "tpu", "metal"):
            prefix = "auto_scalar" if scalar_auto_requested else "auto"
            return "scipy", scipy_lsmr_maxiter, f"{prefix}:dense-preserves-{backend}"
        prefix = "auto_scalar" if scalar_auto_requested else "auto"
        return "scipy", scipy_lsmr_maxiter, f"{prefix}:dense-default"

    def _select_exact_path(self) -> str:
        """Choose the accepted-point differentiation path.

        The established non-scan discrete-adjoint tape is the default on CPU
        and GPU. May 2026 cold and warm ``office`` RTX A4000 profiling showed
        the scan-differentiated exact path can be useful for targeted parity
        studies but is not a robust GPU default for accepted-point Jacobians.
        The environment override ``VMEC_JAX_OPT_EXACT_PATH={tape,scan}``
        remains available for profiling and parity studies.
        """
        requested = getattr(self, "_exact_path_request", None)
        if requested in ("scan", "tape"):
            return str(requested)
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
        mode-2 and mode-3 stellarator-symmetric tapes (24 and 48 DOFs), while
        larger parameter spaces can lose more in replay payload cost than they
        gain in preconditioner cost. ``None`` preserves the solver's legacy
        environment-controlled default for CPU/default backends.
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
            max_dofs = int(os.getenv("VMEC_JAX_OPT_EXACT_TRIDI_PRECOMPUTE_MAX_DOFS", "48"))
        except ValueError:
            max_dofs = 48
        if max_dofs < 0:
            return False
        return True if len(self._specs) <= max_dofs else None

    def _use_scan_for_trial_solves(self) -> bool:
        """Return whether trial residual solves should use the scan loop.

        Exact-optimizer trial residuals are short VMEC solves called repeatedly
        by SciPy's trust-region line search.  They do not need an adjoint tape.
        CPU and current ``office`` GPU/CUDA profiles showed the non-scan loop is
        materially faster for high-mode QS trial points because scan pays a
        large cold compile/dispatch cost.  Environment overrides always win.
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
        return backend in ("tpu",)

    def _exact_tape_backend_name(self) -> str:
        """Return the backend name used for exact-tape optimization policy."""

        backend = str(getattr(self, "_solver_device_name", None) or "").strip().lower()
        if backend:
            return backend
        try:
            from ._compat import jax as _jax

            return str(_jax.default_backend()).strip().lower() if _jax is not None else "cpu"
        except Exception:
            return "cpu"

    @staticmethod
    def _env_bool_override(name: str) -> bool | None:
        flag = os.getenv(name, "").strip().lower()
        if flag in ("1", "true", "yes", "on"):
            return True
        if flag in ("0", "false", "no", "off"):
            return False
        return None

    def _gpu_like_exact_tape_backend(self) -> bool:
        return self._exact_tape_backend_name() in ("gpu", "cuda", "tpu", "rocm")

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

    def _profile_solver_free_boundary_timing(self, diagnostics, *, profile_prefix: str) -> None:
        if not isinstance(diagnostics, dict):
            return

        def _sum_time(key: str) -> float | None:
            if key not in diagnostics:
                return None
            try:
                arr = np.asarray(diagnostics.get(key), dtype=float).reshape(-1)
            except Exception:
                return None
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return None
            return float(np.sum(arr))

        def _count_nonzero(key: str) -> int | None:
            if key not in diagnostics:
                return None
            try:
                arr = np.asarray(diagnostics.get(key), dtype=int).reshape(-1)
            except Exception:
                return None
            if arr.size == 0:
                return None
            return int(np.count_nonzero(arr))

        for key, suffix in (
            ("freeb_nestor_sample_time_history", "freeb_nestor_sample"),
            ("freeb_nestor_solve_time_history", "freeb_nestor_solve"),
            ("freeb_nestor_trial_sample_time_history", "freeb_nestor_trial_sample"),
            ("freeb_nestor_trial_solve_time_history", "freeb_nestor_trial_solve"),
        ):
            value = _sum_time(key)
            if value is not None:
                self._profile_add(f"{profile_prefix}_{suffix}", value)
        for key, suffix in (
            ("freeb_full_update_history", "freeb_nestor_full_update_count"),
            ("freeb_nestor_reused_history", "freeb_nestor_reused_count"),
            ("freeb_nestor_trial_reused_history", "freeb_nestor_trial_reused_count"),
            ("freeb_nestor_trial_failed_history", "freeb_nestor_trial_failed_count"),
        ):
            value = _count_nonzero(key)
            if value is not None:
                self._profile_add_counter(f"{profile_prefix}_{suffix}", value)

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
            self._profile_solver_free_boundary_timing(diagnostics, profile_prefix=profile_prefix)
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
            ("iteration_control_s", "iteration_control"),
            ("iteration_control_fsq1_s", "iteration_control_fsq1"),
            ("iteration_control_fsq1_precond_norm_s", "iteration_control_fsq1_precond_norm"),
            ("iteration_control_fsq1_scalar_build_s", "iteration_control_fsq1_scalar_build"),
            ("iteration_control_fsq1_payload_get_s", "iteration_control_fsq1_payload_get"),
            ("iteration_control_fsq1_direct_get_s", "iteration_control_fsq1_direct_get"),
            ("iteration_control_fsq1_unattributed_s", "iteration_control_fsq1_unattributed"),
            ("iteration_control_badjac_s", "iteration_control_badjac"),
            ("iteration_control_badjac_ptau_get_s", "iteration_control_badjac_ptau_get"),
            ("iteration_control_badjac_state_jacobian_s", "iteration_control_badjac_state_jacobian"),
            ("iteration_control_badjac_unattributed_s", "iteration_control_badjac_unattributed"),
            ("iteration_control_vmec_time_s", "iteration_control_vmec_time"),
            ("iteration_control_restart_s", "iteration_control_restart"),
            ("iteration_control_evolve_s", "iteration_control_evolve"),
            ("iteration_control_unattributed_s", "iteration_control_unattributed"),
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
        self._profile_solver_free_boundary_timing(diagnostics, profile_prefix=profile_prefix)
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
            axis_override = getattr(self, "_initial_axis_override", None)
            if axis_override is None:
                state0 = initial_guess_from_boundary(
                    self._static,
                    boundary_now,
                    self._indata,
                    vmec_project=True,
                )
            else:
                state0 = initial_guess_from_boundary(
                    self._static,
                    boundary_now,
                    self._indata,
                    vmec_project=True,
                    axis_override=axis_override,
                )
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
                axis_override = getattr(self, "_initial_axis_override", None)
                if axis_override is None:
                    state = _ig(
                        self._static,
                        bdy,
                        self._indata,
                        vmec_project=True,
                    )
                else:
                    state = _ig(
                        self._static,
                        bdy,
                        self._indata,
                        vmec_project=True,
                        axis_override=axis_override,
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
        """Host-side boundary update for cache keys and other non-AD logic."""
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
            axis_override = getattr(self, "_initial_axis_override", None)
            if axis_override is None:
                state0 = initial_guess_from_boundary(
                    self._static,
                    boundary_now,
                    self._indata,
                    vmec_project=True,
                )
            else:
                state0 = initial_guess_from_boundary(
                    self._static,
                    boundary_now,
                    self._indata,
                    vmec_project=True,
                    axis_override=axis_override,
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
        axis_override = (
            getattr(self, "_initial_axis_override", None)
            if getattr(self, "_initial_axis_override", None) is not None
            else extract_axis_override_from_state(state0, self._static)
        )
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
        packed_final = _jnp.asarray(tape.final_packed_state, dtype=_jnp.float64)
        state = unpack_state(packed_final, self._layout)
        payload = {"tape": tape, "axis_override": axis_override, "packed_final": packed_final}
        self._exact_cache.clear()
        if not jvp_only:
            self._exact_cache[cache_key] = (state, payload)
        self._remember_exact_state(cache_key, state)
        self._profile_add("exact_unpack_cache", time.perf_counter() - t_unpack)
        self._profile_add("exact_solve_with_tape_total", time.perf_counter() - t_total)
        if jvp_only:
            self._profile_add("exact_solve_with_tape_jvp_only_total", time.perf_counter() - t_total)
        return (state, payload) if return_payload else state

    def _packed_final_from_exact_payload(self, state, payload):
        """Return the accepted packed state already carried by an exact tape payload."""

        from ._compat import jnp as _jnp
        from .state import pack_state

        packed = None
        if isinstance(payload, dict):
            packed = payload.get("packed_final")
            if packed is None:
                tape = payload.get("tape")
                packed = getattr(tape, "final_packed_state", None)
        if packed is None:
            packed = pack_state(state)
        return _jnp.asarray(packed, dtype=_jnp.float64)

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
            previous = os.environ.get(env_name)
            use_basepoint_carries = self._jvp_only_basepoint_carries_enabled()
            if previous is None and use_basepoint_carries:
                os.environ[env_name] = "1"
                self._profile_add("exact_tape_jvp_only_basepoint_carries_auto", 0.0)
            try:
                return solve(params, return_payload=True, jvp_only=True)
            finally:
                if previous is None and use_basepoint_carries:
                    os.environ.pop(env_name, None)
        return solve(params, return_payload=True)

    def _jvp_only_exact_tape_enabled(self) -> bool:
        forced = self._env_bool_override("VMEC_JAX_OPT_JVP_ONLY_EXACT_TAPE")
        if forced is not None:
            return bool(forced)
        enabled = self._gpu_like_exact_tape_backend()
        if enabled:
            self._profile_add("exact_tape_jvp_only_auto_gpu", 0.0)
        return bool(enabled)

    def _jvp_only_basepoint_carries_enabled(self) -> bool:
        forced = self._env_bool_override("VMEC_JAX_JVP_ONLY_EXACT_TAPE_BASEPOINT_CARRIES")
        if forced is not None:
            return bool(forced)
        return self._gpu_like_exact_tape_backend()

    def _initial_tangent_columns(self, params, axis_override, *, profile_prefix: str):
        """Return cached packed initial-state tangents for boundary parameters."""
        from ._compat import jax, jnp as _jnp

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

            t_linearize = time.perf_counter()
            _, initial_state_linear = jax.linearize(
                lambda p: self._solver_initial_state_packed_from_params(p, axis_override),
                params,
            )
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

    def _solver_initial_state_packed_from_params(self, params, axis_override):
        """Packed initial state after the solver's setup-time constraints.

        ``solve_fixed_boundary_residual_iter`` applies fixed-boundary edge
        enforcement, axis regularity, and VMEC lambda-axis rules before it
        records the first adjoint trace.  Replay tangents must enter at that
        same ``state_pre`` point rather than at the raw initial-guess state.
        """

        from ._compat import jnp as _jnp
        from .init_guess import initial_guess_from_boundary as _ig
        from .solve import (
            _apply_vmec_lambda_axis_rules_to_state,
            _enforce_fixed_boundary_and_axis,
            _mode00_index,
        )
        from .state import pack_state

        bdy = self._boundary_from_params(params)
        state0 = _ig(
            self._static,
            bdy,
            self._indata,
            vmec_project=True,
            axis_override=axis_override,
        )
        modes = getattr(self._static, "modes", None)
        if modes is None:
            return _jnp.asarray(pack_state(state0), dtype=_jnp.float64)

        idx00 = _mode00_index(modes)
        state0 = _enforce_fixed_boundary_and_axis(
            state0,
            self._static,
            edge_Rcos=_jnp.asarray(state0.Rcos)[-1, :],
            edge_Rsin=_jnp.asarray(state0.Rsin)[-1, :],
            edge_Zcos=_jnp.asarray(state0.Zcos)[-1, :],
            edge_Zsin=_jnp.asarray(state0.Zsin)[-1, :],
            enforce_edge=True,
            enforce_lambda_axis=True,
            idx00=idx00,
        )
        state0 = _apply_vmec_lambda_axis_rules_to_state(
            state0,
            enforce_vmec_lambda_axis=True,
            host_update_assembly=False,
            idx00=idx00,
        )
        return _jnp.asarray(pack_state(state0), dtype=_jnp.float64)

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
            if int(n_params) < 24:
                return None
            if bool(getattr(self._static.cfg, "lasym", False)):
                # LASYM doubles the boundary columns and remains more memory
                # sensitive on GPU; keep the older conservative replay chunks.
                return 8
            # Non-LASYM GPU projected replay is launch/transpose dominated.
            # Fresh office RTX A4000/JAX 0.6.2 profiles in June 2026 showed
            # larger chunks reduce cold and warm callback time for QH mode-2
            # through mode-4 without increasing host materialization cost.
            if int(n_params) <= 64:
                return int(n_params)
            if int(n_params) <= 128:
                return max(24, int(n_params) // 2)
            return 64
        if backend_name == "tpu":
            return None
        if not bool(getattr(self._static.cfg, "lasym", False)):
            return None
        if int(n_params) >= 64:
            return 8
        if int(n_params) >= 32:
            return 4
        return None

    def _precompute_linear_operator_initial_tangents_enabled(self, n_params: int) -> bool:
        """Whether matrix-free operators should cache initial-state tangent columns.

        Matrix-free least-squares avoids materializing the full residual
        Jacobian, but every ``J.T @ w`` still needs the transpose of the frozen
        initial-state map.  For stellarator-symmetric CPU/default high-mode
        production runs this transpose was a repeat offender in accepted-point
        profiles.  Precomputing the small ``n_params x state_size`` initial
        tangent block once per accepted point lets both ``J @ v`` and
        ``J.T @ w`` reuse fast tensor contractions while preserving the larger
        matrix-free residual projection.  The default starts at 64 DOFs because
        lower-DOF probes usually perform too few transpose products to amortize
        the tangent-column build.
        """

        if int(n_params) <= 0:
            return False
        flag = os.getenv("VMEC_JAX_OPT_LINEAR_OPERATOR_INITIAL_TANGENTS")
        if flag is not None:
            return flag.strip().lower() not in ("", "0", "false", "no", "off")
        backend = _optimizer_backend_name(getattr(self, "_solver_device_name", None))
        if backend in ("gpu", "cuda", "rocm", "tpu", "metal"):
            return False
        if self._has_stellarator_asymmetric_configuration():
            return False
        min_dofs = int(os.getenv("VMEC_JAX_OPT_LINEAR_OPERATOR_INITIAL_TANGENT_MIN_DOFS", "64"))
        max_dofs = int(os.getenv("VMEC_JAX_OPT_LINEAR_OPERATOR_INITIAL_TANGENT_MAX_DOFS", "128"))
        return min_dofs <= int(n_params) <= max_dofs

    def _scalar_gradient_initial_tangents_enabled(self, n_params: int) -> bool:
        """Whether scalar-adjoint gradients should project cached initial tangents.

        The reverse scalar-adjoint path needs one VMEC-tape VJP plus the
        transpose of the initial-state map.  On GPU high-mode optimizations the
        repeated initial VJP is a cold/warm callback hotspot.  The initial map is
        affine for a fixed axis/flip branch, so precomputing its tangent columns
        once per branch lets scalar gradients use a device-side dot product on
        subsequent accepted points.  Keep this default narrow because CPU
        single-gradient probes usually do not amortize the tangent build.
        """

        if int(n_params) <= 0:
            return False
        flag = os.getenv("VMEC_JAX_OPT_SCALAR_GRADIENT_INITIAL_TANGENTS")
        if flag is not None:
            return flag.strip().lower() not in ("", "0", "false", "no", "off")
        backend = _optimizer_backend_name(getattr(self, "_solver_device_name", None))
        if backend not in ("gpu", "cuda", "rocm", "tpu", "metal"):
            return False
        if self._has_stellarator_asymmetric_configuration():
            return False
        min_dofs = int(os.getenv("VMEC_JAX_OPT_SCALAR_GRADIENT_INITIAL_TANGENT_MIN_DOFS", "24"))
        max_dofs = int(os.getenv("VMEC_JAX_OPT_SCALAR_GRADIENT_INITIAL_TANGENT_MAX_DOFS", "256"))
        return min_dofs <= int(n_params) <= max_dofs

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
        # June 2026 office profiles show projected replay is slower for QH
        # mode-2 (24 columns): dispatch dominates the projected path, while the
        # standard replay path is about 16% faster over two perturbed GPU
        # Jacobian callbacks. Keep projected replay for larger non-LASYM dense
        # Jacobians where avoiding the intermediate tangent synchronization can
        # still amortize the extra dispatch cost.
        return int(n_params) >= 48

    def _fused_projected_replay_enabled(self) -> bool:
        """Whether projected replay should fuse replay and residual projection when possible."""

        flag = os.getenv("VMEC_JAX_OPT_FUSED_PROJECTED_REPLAY", "").strip().lower()
        if flag:
            return flag in ("1", "true", "yes", "on")
        # Office GPU profiles on 2026-05-28 show the fused mode-2 QH callback
        # is slower than the regular projected replay path. Keep fusion opt-in
        # for diagnostics until a broader matrix shows a reproducible win.
        return False

    def _chunked_projected_replay_projection_enabled(self, column_chunk: int | None, n_params: int) -> bool:
        """Whether to project residual tangents immediately after each replay chunk."""

        if column_chunk is None:
            return False
        if int(n_params) <= int(column_chunk):
            return False
        flag = os.getenv("VMEC_JAX_OPT_CHUNKED_PROJECTED_REPLAY_PROJECTION", "").strip().lower()
        if flag:
            return flag in ("1", "true", "yes", "on")
        backend = _optimizer_backend_name(getattr(self, "_solver_device_name", None))
        if backend not in ("gpu", "cuda", "rocm"):
            return False
        if bool(getattr(getattr(self._static, "cfg", None), "lasym", False)):
            return False
        return True

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
        from .state import unpack_state

        state, payload = self._solve_exact_with_tape_for_jvp(params)
        packed_final = self._packed_final_from_exact_payload(state, payload)

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
        if self._chunked_projected_replay_projection_enabled(column_chunk, int(params.size)):
            t_replay = time.perf_counter()
            jac_blocks = []
            residuals = None
            for start in range(0, int(params.size), int(column_chunk)):
                stop = min(start + int(column_chunk), int(params.size))
                final_tangents_chunk = checkpoint_tape_state_jvp_columns(
                    tape=payload["tape"],
                    static=self._static,
                    initial_tangents=initial_tangents[start:stop],
                    rebuild_preconditioner=True,
                    column_chunk=column_chunk,
                    _allow_chunking=False,
                )
                residuals, jac_chunk = helper_cache["residual_tangent_jacobian"](
                    packed_final,
                    final_tangents_chunk,
                )
                jac_blocks.append(jac_chunk)
            jac = _jnp.concatenate(jac_blocks, axis=1)
            residuals, jac = self._profile_blocking_phase(
                "jacobian_chunked_projected_replay_projection_total",
                t_replay,
                (residuals, jac),
            )
            self._last_jacobian_residual = np.asarray(residuals, dtype=float)
            self._remember_exact_residual(exact_param_key, self._last_jacobian_residual)
            t_host = time.perf_counter()
            out = np.asarray(jac, dtype=float)
            self._profile_add("jacobian_host_materialize", time.perf_counter() - t_host)
            self._remember_exact_jacobian(exact_param_key, out, self._last_jacobian_residual)
            self._last_jacobian_source = "exact_tape_chunked_projected_replay_projection"
            self._profile_add("jacobian_total", time.perf_counter() - t_total)
            return out

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
        return exact_objective_and_gradient(self, params)

    def gradient_fun(self, params) -> np.ndarray:
        """Exact reverse-discrete-adjoint gradient of the scalar objective."""
        return self.objective_and_gradient_fun(params)[1]

    def residual_linear_operator(self, params):
        """Return a matrix-free exact residual Jacobian at ``params``.

        The returned :class:`scipy.sparse.linalg.LinearOperator` implements
        ``J @ v`` with one forward tangent replay and ``J.T @ w`` with one
        reverse replay through the same converged VMEC iteration tape. This is
        the trust-region counterpart to :meth:`objective_and_gradient_fun` and
        avoids materializing the dense ``n_residuals x n_parameters`` Jacobian.
        """

        return build_residual_linear_operator(self, params)

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
        return qs_objective_from_residuals(res, self._residual_history_policy())

    def _residual_history_policy(self) -> ResidualHistoryPolicy:
        """Return the residual-block metadata used for history reconstruction."""

        return ResidualHistoryPolicy(
            aspect_target=getattr(self, "_aspect_target", None),
            aspect_weight=float(getattr(self, "_aspect_weight", 1.0)),
            n_non_qs=int(getattr(self, "_n_non_qs", 1)),
            n_qs=getattr(self, "_n_qs", None),
            has_residual_block_metadata=getattr(self, "_has_residual_block_metadata", None),
            has_iota_callback=getattr(self, "_iota_fn", None) is not None,
        )

    def _has_qs_residual_block_metadata(self) -> bool:
        return self._residual_history_policy().has_qs_residual_block_metadata()

    def _can_build_qs_from_residuals(self) -> bool:
        """Return true when residual block metadata identifies QS/objective blocks."""
        return self._residual_history_policy().can_build_qs_from_residuals()

    def _can_build_aspect_from_residuals(self) -> bool:
        """Return true when the first residual encodes weighted aspect error."""
        return self._residual_history_policy().can_build_aspect_from_residuals()

    def _can_build_history_from_residuals(self) -> bool:
        """Return true when residual metadata is enough for history metrics."""
        return self._residual_history_policy().can_build_history_from_residuals()

    def _history_entry_from_residuals(self, res: np.ndarray, *, wall_time_s: float) -> dict:
        """Build a history row without re-solving the accepted scan state."""
        return history_entry_from_residuals(
            res,
            wall_time_s=wall_time_s,
            policy=self._residual_history_policy(),
        )

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

    def _reset_run_state(self, *, trace_callbacks: bool | None, iota_fn) -> None:
        """Reset mutable per-run caches, traces, and accepted-point bookkeeping."""

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
        self._iota_fn = iota_fn
        self._best_exact_params = None
        self._best_exact_state = None
        self._best_exact_residual = None
        self._best_exact_cost = math.inf
        self._exact_history_rejected_count = 0

    def _build_run_history_dump(
        self,
        *,
        max_nfev: int,
        ftol: float,
        gtol: float,
        xtol: float,
        method_key: str,
        method_requested: str,
        method_auto_reason: str | None,
        scipy_tr_solver: str | None,
        scipy_lsmr_maxiter: int | None,
        lbfgs_step_bound: float | None,
        scalar_step_bound: float | None,
        scalar_cost_only_trials_used: bool | None,
        final_wall_time_s: float,
        result: dict,
        cost0: float,
        cost_final: float,
        qs_total0: float,
        qs_total_final: float,
        aspect0: float,
        aspect_final: float,
        entry0: dict,
        entry_final: dict,
        iota_fn,
        target_iota: float | None,
        target_aspect: float | None,
        selected_best_exact: bool,
        optimizer_exception,
    ) -> dict:
        """Assemble the serializable optimization history payload."""

        return build_run_history_dump(
            label="Optimisation",
            max_nfev=max_nfev,
            ftol=ftol,
            gtol=gtol,
            xtol=xtol,
            method_key=method_key,
            method_requested=method_requested,
            method_auto_reason=method_auto_reason,
            exact_path=self._scan_exact_path,
            scipy_tr_solver=scipy_tr_solver,
            scipy_lsmr_maxiter=scipy_lsmr_maxiter,
            lbfgs_step_bound=lbfgs_step_bound,
            scalar_step_bound=scalar_step_bound,
            scalar_cost_only_trials_used=scalar_cost_only_trials_used,
            solver_device=self._solver_device_name or "default",
            inner_max_iter=int(self._inner_max_iter),
            inner_ftol=float(self._inner_ftol),
            trial_max_iter=int(self._trial_max_iter),
            trial_ftol=float(self._trial_ftol),
            final_wall_time_s=final_wall_time_s,
            result=result,
            cost0=cost0,
            cost_final=cost_final,
            qs_total0=qs_total0,
            qs_total_final=qs_total_final,
            aspect0=aspect0,
            aspect_final=aspect_final,
            history=self._history,
            profile=self._profile_dump(),
            selected_best_exact=selected_best_exact,
            rejected_trial_exact_history_count=int(self._exact_history_rejected_count),
            optimizer_exception=optimizer_exception,
            iota_fn_present=iota_fn is not None,
            entry0=entry0,
            entry_final=entry_final,
            target_iota=target_iota,
            target_aspect=target_aspect,
            callback_trace=(self._callback_trace_dump() if self._callback_trace_enabled else None),
        )

    def _attach_run_private_payload(
        self,
        result: dict,
        *,
        state_initial,
        state_final,
        history_dump: dict,
    ) -> dict:
        """Attach non-serializable state/profile payloads used by examples."""

        result["_state_initial"] = state_initial
        result["_state_final"] = state_final
        result["_profile"] = self._profile_dump()
        result["_history_dump"] = history_dump
        return result

    def _initial_run_evaluation(self, params0_arr: np.ndarray):
        """Evaluate and record the exact initial point for an optimization run."""

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
        return res0, state0, entry0, cost0, qs_total0, aspect0

    def _record_cached_exact_history_entry(
        self,
        params,
        *,
        last_history_key: list,
        cost: float | None = None,
    ) -> bool:
        """Append history from an exact cached state when this is a new accepted point."""

        key = self._exact_cache_key(params)
        if key == last_history_key[0] or key not in self._exact_cache:
            return False
        cached_state, _ = self._exact_cache[key]
        exact_residual = self._cached_exact_residual(cache_key=key)
        kwargs = {
            "wall_time_s": time.perf_counter() - self._wall_t0,
            "cache_key": key,
        }
        if cost is not None:
            kwargs["cost"] = float(cost)
        entry = self._history_entry_from_state_or_residual(
            cached_state,
            exact_residual,
            **kwargs,
        )
        entry_cost = float(entry["cost"])
        if self._exact_history_accepts(entry_cost):
            self._history.append(entry)
            if exact_residual is None:
                exact_residual = self._cached_exact_residual(cache_key=key)
            if exact_residual is not None:
                self._remember_best_exact_point(params, exact_residual, entry_cost, state=cached_state)
            last_history_key[0] = key
            return True
        self._exact_history_rejected_count += 1
        return False

    def _wall_time_for_final_history_entry(self) -> float:
        """Return monotone wall time for a final optimization-history entry."""

        return monotone_final_wall_time(now_s=time.perf_counter() - self._wall_t0, history=self._history)

    def _evaluate_and_record_final_exact_point(
        self,
        result: dict,
        *,
        selected_best_exact: bool,
    ):
        """Select the final exact accepted point and append its history entry.

        Final artifacts must come from an exact accepted solve.  If the optimizer's
        nominal final point cannot be reconstructed, or if a prior exact accepted
        point has a lower exact cost, use that best exact point instead of a
        relaxed trial solve.
        """

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
                if best_exact_params is not None and best_exact_residual is not None and np.isfinite(best_exact_cost):
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

        final_wall_time_s = self._wall_time_for_final_history_entry()
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
            final_wall_time_s = self._wall_time_for_final_history_entry()
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
        return (
            state_final,
            entry_final,
            cost_final,
            qs_total_final,
            aspect_final,
            final_wall_time_s,
            selected_best_exact,
        )

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
            ``"auto_scalar"``/``"auto_adjoint"`` keep the same safeguards but
            choose ``"scalar_trust"`` for high-mode, stellarator-symmetric
            QS/QI CPU/default-backend cases, enabling scalar-adjoint production
            tests without environment variables.
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
            exact ``Jv``/``J.Tv`` products.  For ``method="scipy_matrix_free"``,
            ``None`` uses vmec_jax's bounded default of 4 to avoid unbounded
            inner Krylov work; pass an explicit integer to tighten or relax
            that cap.
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
        self._reset_run_state(trace_callbacks=trace_callbacks, iota_fn=iota_fn)

        params0_arr = np.asarray(params0, dtype=float)
        scalar_cost_only_trials_used: bool | None = None

        # ── initial evaluation ──────────────────────────────────────────────
        res0, state0, entry0, cost0, qs_total0, aspect0 = self._initial_run_evaluation(params0_arr)

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
            result, scalar_cost_only_trials_used = run_scalar_trust_exact_optimizer(
                self,
                params0_arr,
                x_scale=x_scale,
                max_nfev=max_nfev,
                ftol=ftol,
                gtol=gtol,
                scalar_step_bound=scalar_step_bound,
                scalar_cost_only_trials=scalar_cost_only_trials,
            )
        elif method_key in ("lbfgs", "lbfgs_adjoint"):
            result = run_lbfgs_adjoint_exact_optimizer(
                self,
                params0_arr,
                x_scale=x_scale,
                max_nfev=max_nfev,
                ftol=ftol,
                gtol=gtol,
                verbose=verbose,
                lbfgs_step_bound=lbfgs_step_bound,
            )
        elif method_key in ("scipy_matrix_free", "matrix_free", "scipy_mf"):
            result, scipy_lsmr_maxiter = run_scipy_matrix_free_exact_optimizer(
                self,
                params0_arr,
                x_scale=x_scale,
                max_nfev=max_nfev,
                ftol=ftol,
                gtol=gtol,
                xtol=xtol,
                verbose=verbose,
                scipy_lsmr_maxiter=scipy_lsmr_maxiter,
            )
        elif method_key == "scipy":
            result = run_scipy_dense_exact_optimizer(
                self,
                params0_arr,
                x_scale=x_scale,
                max_nfev=max_nfev,
                ftol=ftol,
                gtol=gtol,
                xtol=xtol,
                verbose=verbose,
                scipy_tr_solver=scipy_tr_solver,
                scipy_lsmr_maxiter=scipy_lsmr_maxiter,
            )
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
        (
            state_final,
            entry_final,
            cost_final,
            qs_total_final,
            aspect_final,
            final_wall_time_s,
            selected_best_exact,
        ) = self._evaluate_and_record_final_exact_point(
            result,
            selected_best_exact=selected_best_exact,
        )

        # ── assemble history dump ───────────────────────────────────────────
        history_dump = self._build_run_history_dump(
            max_nfev=max_nfev,
            ftol=ftol,
            gtol=gtol,
            xtol=xtol,
            method_key=method_key,
            method_requested=method_requested,
            method_auto_reason=method_auto_reason,
            scipy_tr_solver=scipy_tr_solver,
            scipy_lsmr_maxiter=scipy_lsmr_maxiter,
            lbfgs_step_bound=lbfgs_step_bound,
            scalar_step_bound=scalar_step_bound,
            scalar_cost_only_trials_used=scalar_cost_only_trials_used,
            final_wall_time_s=final_wall_time_s,
            result=result,
            cost0=cost0,
            cost_final=cost_final,
            qs_total0=qs_total0,
            qs_total_final=qs_total_final,
            aspect0=aspect0,
            aspect_final=aspect_final,
            entry0=entry0,
            entry_final=entry_final,
            iota_fn=iota_fn,
            target_iota=target_iota,
            target_aspect=target_aspect,
            selected_best_exact=selected_best_exact,
            optimizer_exception=optimizer_exception,
        )

        # Private, non-serializable convenience payload for scripts that want
        # to write wout files without rerunning the VMEC solve immediately after
        # optimization. save_history() only persists `_history_dump`.
        return self._attach_run_private_payload(
            result,
            state_initial=state0,
            state_final=state_final,
            history_dump=history_dump,
        )
