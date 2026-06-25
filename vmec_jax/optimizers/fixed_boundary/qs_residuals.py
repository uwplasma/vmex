"""Quasisymmetry residual factories for fixed-boundary exact optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from vmec_jax._compat import jnp
from vmec_jax.boundary import BoundaryCoeffs, boundary_from_indata
from vmec_jax.booz_input import BoozXformInputs, booz_xform_inputs_from_state
from vmec_jax.energy import FluxProfiles, flux_profiles_from_indata
from vmec_jax.field import signgs_from_sqrtg
from vmec_jax.geom import eval_geom
from vmec_jax.init_guess import initial_guess_from_boundary
import vmec_jax.modes as modes_module
from vmec_jax.profiles import eval_profiles
import vmec_jax.quasisymmetry as quasisymmetry_module
from vmec_jax.state import VMECState
from vmec_jax.static import VMECStatic
import vmec_jax.wout as wout_module


@dataclass(frozen=True)
class FixedBoundaryContext:
    """Bundled inputs for repeated fixed-boundary solves."""

    st_guess: VMECState
    signgs: int
    flux: FluxProfiles
    pressure: jnp.ndarray
    booz_inputs: BoozXformInputs


def smooth_min_abs_iota_residual(
    iota,
    minimum: float,
    *,
    softness: float = 1.0e-3,
    abs_epsilon: float = 1.0e-12,
):
    """Smooth residual for the differentiable constraint ``abs(iota) >= minimum``."""

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
    """Parse a comma-separated surface list into floats or one-based indices."""
    items: list[float | int] = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        items.append(float(raw) if any(ch in raw for ch in (".", "e", "E")) else int(raw))
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
    prof = eval_profiles(indata, jnp.asarray(static.s))
    return jnp.asarray(
        prof.get("pressure", jnp.zeros_like(jnp.asarray(static.s))),
        dtype=jnp.asarray(static.s).dtype,
    )


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
    **deps,
) -> Callable:
    """Build a QH/QS residual with a required aspect-ratio target."""

    return make_qs_residuals_fn(
        static,
        indata,
        signgs=signgs,
        helicity_m=helicity_m,
        helicity_n=helicity_n,
        target_aspect=target_aspect,
        target_iota=None,
        min_abs_iota=None,
        surfaces=surfaces,
        aspect_weight=aspect_weight,
        qs_weight=qs_weight,
        iota_weight=1.0,
        **deps,
    )


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
    boundary_from_indata_func=None,
    initial_guess_from_boundary_func=None,
    eval_geom_func=None,
    signgs_from_sqrtg_func=None,
    flux_profiles_from_indata_func=None,
    pressure_profile_for_static_func=None,
    smooth_min_abs_iota_residual_func=None,
) -> Callable:
    """General quasisymmetry residual factory supporting QH, QA, and QP."""

    if boundary_from_indata_func is None:
        boundary_from_indata_func = boundary_from_indata
    if initial_guess_from_boundary_func is None:
        initial_guess_from_boundary_func = initial_guess_from_boundary
    if eval_geom_func is None:
        eval_geom_func = eval_geom
    if signgs_from_sqrtg_func is None:
        signgs_from_sqrtg_func = signgs_from_sqrtg
    if flux_profiles_from_indata_func is None:
        flux_profiles_from_indata_func = flux_profiles_from_indata
    if pressure_profile_for_static_func is None:
        pressure_profile_for_static_func = _pressure_profile_for_static
    if surfaces is None:
        surfaces = np.arange(0.0, 1.01, 0.1)
    surfaces = np.asarray(surfaces, dtype=float)

    if signgs is None:
        try:
            boundary_init = boundary_from_indata_func(indata, static.modes)
            state0 = initial_guess_from_boundary_func(static, boundary_init, indata)
            geom = eval_geom_func(state0, static)
            signgs = int(signgs_from_sqrtg_func(np.asarray(geom.sqrtg), axis_index=1))
        except Exception:
            signgs = 1

    flux = flux_profiles_from_indata_func(indata, static.s, signgs=signgs)
    pressure = pressure_profile_for_static_func(indata, static)
    nyq_modes = modes_module.nyquist_mode_table_from_grid(
        mpol=int(static.cfg.mpol),
        ntor=int(static.cfg.ntor),
        ntheta=int(static.cfg.ntheta),
        nzeta=int(static.cfg.nzeta),
    )
    angle_cache = quasisymmetry_module._quasisymmetry_angle_cache(
        nfp=int(static.cfg.nfp),
        xm_nyq=nyq_modes.m,
        xn_nyq=nyq_modes.n * int(static.cfg.nfp),
    )
    _signgs = signgs
    _indata = indata

    if smooth_min_abs_iota_residual_func is None:
        smooth_min_abs_iota_residual_func = smooth_min_abs_iota_residual

    def _qs_eval_from_state(state: VMECState):
        return quasisymmetry_module.quasisymmetry_ratio_residual_from_state(
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
            aspect = wout_module.equilibrium_aspect_ratio_from_state(state=state, static=static)
            parts.append(jnp.asarray([float(aspect_weight) * (aspect - target_aspect)], dtype=jnp.float64))

        if target_iota is not None or min_abs_iota is not None:
            _chips, _iotas, _iotaf = wout_module.equilibrium_iota_profiles_from_state(
                state=state,
                static=static,
                indata=_indata,
                signgs=_signgs,
            )
            del _chips, _iotaf
            iotas = jnp.asarray(_iotas, dtype=jnp.float64)
            mean_iota = jnp.asarray(0.0, dtype=iotas.dtype) if int(iotas.shape[0]) <= 1 else jnp.mean(iotas[1:])
            if target_iota is not None:
                iota_residual = mean_iota - target_iota
            else:
                iota_residual = smooth_min_abs_iota_residual_func(
                    mean_iota,
                    float(min_abs_iota),
                    softness=float(iota_floor_softness),
                )
            parts.append(jnp.asarray([float(iota_weight) * iota_residual], dtype=jnp.float64))

        qs = _qs_eval_from_state(state)
        parts.append(jnp.asarray(qs["residuals1d"], dtype=jnp.float64) * float(qs_weight))
        return jnp.concatenate(parts)

    def state_cotangent_operator_from_packed(packed_state, layout):
        from vmec_jax._compat import jax, jnp as _jnp
        from vmec_jax.state import unpack_state

        packed_state = _jnp.asarray(packed_state, dtype=_jnp.float64)
        blocks: list[tuple[slice | int, Callable, bool]] = []
        offset = 0

        if target_aspect is not None:
            block_index = offset
            offset += 1

            def _aspect_from_packed(packed):
                state = unpack_state(packed, layout)
                aspect = wout_module.equilibrium_aspect_ratio_from_state(state=state, static=static)
                return float(aspect_weight) * (aspect - target_aspect)

            _, aspect_vjp = jax.vjp(_aspect_from_packed, packed_state)
            blocks.append((block_index, aspect_vjp, False))

        if target_iota is not None or min_abs_iota is not None:
            block_index = offset
            offset += 1

            def _iota_from_packed(packed):
                state = unpack_state(packed, layout)
                _chips, _iotas, _iotaf = wout_module.equilibrium_iota_profiles_from_state(
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
                    iota_residual = smooth_min_abs_iota_residual_func(
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
        from vmec_jax._compat import jax, jnp as _jnp
        from vmec_jax.state import unpack_state

        packed_state = _jnp.asarray(packed_state, dtype=_jnp.float64)

        def _objective(packed):
            state = unpack_state(packed, layout)
            total = _jnp.asarray(0.0, dtype=_jnp.float64)
            if target_aspect is not None:
                aspect = wout_module.equilibrium_aspect_ratio_from_state(state=state, static=static)
                aspect_residual = float(aspect_weight) * (aspect - target_aspect)
                total = total + 0.5 * aspect_residual * aspect_residual
            if target_iota is not None or min_abs_iota is not None:
                _chips, _iotas, _iotaf = wout_module.equilibrium_iota_profiles_from_state(
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
                    iota_residual = smooth_min_abs_iota_residual_func(
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
