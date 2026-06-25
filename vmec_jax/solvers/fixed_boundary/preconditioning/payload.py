"""Cached JIT helpers for residual-loop preconditioner payload assembly."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable

from ...._compat import has_jax, jax, jit, jnp
from ..jit_cache import (
    jit_cache_get,
    jit_cache_put,
    strict_update_static_cache_key,
)
from ....vmec_residue import vmec_gcx2_from_tomnsps
from ....vmec_tomnsp import TomnspsRZL
from ..residual.force_norms import lambda_preconditioned_full_norm


STRICT_UPDATE_STEP_JIT_CACHE: OrderedDict[tuple, Any] = OrderedDict()
PRECOND_OUTPUT_SCALE_JIT_CACHE: OrderedDict[tuple, Any] = OrderedDict()
PRECOND_OUTPUT_PAYLOAD_JIT_CACHE: OrderedDict[tuple, Any] = OrderedDict()
PRECOND_APPLY_PAYLOAD_JIT_CACHE: OrderedDict[tuple, Any] = OrderedDict()
ACCEPTED_CONTROL_PAYLOAD_JIT_CACHE: OrderedDict[tuple, Any] = OrderedDict()


def strict_update_step_jit(
    static,
    *,
    limit_update_rms: bool,
    need_update_rms: bool,
    divide_by_scalxc_for_update: bool,
    enforce_edge: bool = True,
):
    """Return a cached fused strict-update step for accelerator exact solves."""

    if not has_jax():
        return None
    key = (
        strict_update_static_cache_key(static),
        bool(limit_update_rms),
        bool(need_update_rms),
        bool(divide_by_scalxc_for_update),
        bool(enforce_edge),
    )
    cached = jit_cache_get(STRICT_UPDATE_STEP_JIT_CACHE, key)
    if cached is not None:
        return cached

    from ....discrete_adjoint import strict_update_accepted_step

    def _step(
        state_pre,
        dt_eff,
        b1,
        fac,
        force_scale,
        flip_sign,
        vRcc_before,
        vRss_before,
        vZsc_before,
        vZcs_before,
        vLsc_before,
        vLcs_before,
        vRsc_before,
        vRcs_before,
        vZcc_before,
        vZss_before,
        vLcc_before,
        vLss_before,
        frcc_u,
        frss_u,
        fzsc_u,
        fzcs_u,
        flsc_u,
        flcs_u,
        frsc_u,
        frcs_u,
        fzcc_u,
        fzss_u,
        flcc_u,
        flss_u,
        max_update_rms,
    ):
        return strict_update_accepted_step(
            state_pre,
            static,
            dt_eff=dt_eff,
            b1=b1,
            fac=fac,
            force_scale=force_scale,
            flip_sign=flip_sign,
            vRcc_before=vRcc_before,
            vRss_before=vRss_before,
            vZsc_before=vZsc_before,
            vZcs_before=vZcs_before,
            vLsc_before=vLsc_before,
            vLcs_before=vLcs_before,
            vRsc_before=vRsc_before,
            vRcs_before=vRcs_before,
            vZcc_before=vZcc_before,
            vZss_before=vZss_before,
            vLcc_before=vLcc_before,
            vLss_before=vLss_before,
            frcc_u=frcc_u,
            frss_u=frss_u,
            fzsc_u=fzsc_u,
            fzcs_u=fzcs_u,
            flsc_u=flsc_u,
            flcs_u=flcs_u,
            frsc_u=frsc_u,
            frcs_u=frcs_u,
            fzcc_u=fzcc_u,
            fzss_u=fzss_u,
            flcc_u=flcc_u,
            flss_u=flss_u,
            max_update_rms=max_update_rms,
            limit_update_rms=bool(limit_update_rms),
            need_update_rms=bool(need_update_rms),
            divide_by_scalxc_for_update=bool(divide_by_scalxc_for_update),
            enforce_edge=bool(enforce_edge),
        )

    compiled = jax.jit(_step)
    return jit_cache_put(
        STRICT_UPDATE_STEP_JIT_CACHE,
        key,
        compiled,
        env_name="VMEC_JAX_STRICT_UPDATE_CACHE_SIZE",
        default=16,
    )


def preconditioner_output_scaling_jit(*, apply_lambda_update_scale: bool):
    """Return a cached fused scaler for R/Z/lambda preconditioner outputs."""

    if not has_jax():
        return None
    key = (bool(apply_lambda_update_scale),)
    cached = jit_cache_get(PRECOND_OUTPUT_SCALE_JIT_CACHE, key)
    if cached is not None:
        return cached

    def _scale(frzl_rz, lam_prec, w_mode_mn, lambda_update_scale_j):
        w = jnp.asarray(w_mode_mn)[None, :, :]
        lam_prec_j = jnp.asarray(lam_prec)

        frcc = jnp.asarray(frzl_rz.frcc)
        frss = frzl_rz.frss
        fzsc = jnp.asarray(frzl_rz.fzsc)
        fzcs = frzl_rz.fzcs

        flsc = jnp.asarray(frzl_rz.flsc) * lam_prec_j
        flcs = None if frzl_rz.flcs is None else (jnp.asarray(frzl_rz.flcs) * lam_prec_j)

        frsc = (
            jnp.asarray(frzl_rz.frsc)
            if getattr(frzl_rz, "frsc", None) is not None
            else jnp.zeros_like(frcc)
        )
        frcs = (
            jnp.asarray(frzl_rz.frcs)
            if getattr(frzl_rz, "frcs", None) is not None
            else jnp.zeros_like(frcc)
        )
        fzcc = (
            jnp.asarray(frzl_rz.fzcc)
            if getattr(frzl_rz, "fzcc", None) is not None
            else jnp.zeros_like(fzsc)
        )
        fzss = (
            jnp.asarray(frzl_rz.fzss)
            if getattr(frzl_rz, "fzss", None) is not None
            else jnp.zeros_like(fzsc)
        )
        flcc = (
            jnp.asarray(frzl_rz.flcc) * lam_prec_j
            if getattr(frzl_rz, "flcc", None) is not None
            else jnp.zeros_like(flsc)
        )
        flss = (
            jnp.asarray(frzl_rz.flss) * lam_prec_j
            if getattr(frzl_rz, "flss", None) is not None
            else jnp.zeros_like(flsc)
        )

        frcc_u = frcc * w
        frss_u = (frss if frss is not None else jnp.zeros_like(frcc_u)) * w
        fzsc_u = fzsc * w
        fzcs_u = (fzcs if fzcs is not None else jnp.zeros_like(fzsc_u)) * w
        flsc_u = flsc * w
        flcs_u = (flcs if flcs is not None else jnp.zeros_like(flsc_u)) * w
        frsc_u = frsc * w
        frcs_u = frcs * w
        fzcc_u = fzcc * w
        fzss_u = fzss * w
        flcc_u = flcc * w
        flss_u = flss * w

        if bool(apply_lambda_update_scale):
            lambda_update_scale_j = jnp.asarray(lambda_update_scale_j, dtype=flsc_u.dtype)
            flsc_u = flsc_u * lambda_update_scale_j
            flcs_u = flcs_u * lambda_update_scale_j
            flcc_u = flcc_u * lambda_update_scale_j
            flss_u = flss_u * lambda_update_scale_j

        return (
            (frcc, frss, fzsc, fzcs, flsc, flcs, frsc, frcs, fzcc, fzss, flcc, flss),
            (frcc_u, frss_u, fzsc_u, fzcs_u, flsc_u, flcs_u, frsc_u, frcs_u, fzcc_u, fzss_u, flcc_u, flss_u),
        )

    compiled = jax.jit(_scale)
    return jit_cache_put(
        PRECOND_OUTPUT_SCALE_JIT_CACHE,
        key,
        compiled,
        env_name="VMEC_JAX_PRECOND_OUTPUT_SCALE_CACHE_SIZE",
        default=4,
    )


def preconditioner_output_payload_jit(
    *,
    apply_lambda_update_scale: bool,
    vmec2000_control: bool,
    lconm1: bool,
    scaling_func: Callable[..., Any] = preconditioner_output_scaling_jit,
):
    """Return a cached GPU payload builder for preconditioner outputs and fsq1.

    This keeps the VMEC2000 convention that ``lambda_update_scale`` only changes
    coefficient updates; the preconditioned lambda residual diagnostic still uses
    the unscaled ``faclam*gcl`` norm.
    """

    if not has_jax():
        return None
    key = (bool(apply_lambda_update_scale), bool(vmec2000_control), bool(lconm1), id(scaling_func))
    cached = jit_cache_get(PRECOND_OUTPUT_PAYLOAD_JIT_CACHE, key)
    if cached is not None:
        return cached

    def _payload(frzl_rz, lam_prec, w_mode_mn, lambda_update_scale_j, f_norm1, delta_s, s):
        (pre_blocks, update_blocks) = scaling_func(
            apply_lambda_update_scale=bool(apply_lambda_update_scale)
        )(frzl_rz, lam_prec, w_mode_mn, lambda_update_scale_j)
        (frcc, frss, fzsc, fzcs, flsc, flcs, frsc, frcs, fzcc, fzss, flcc, flss) = pre_blocks
        frzl_pre = TomnspsRZL(
            frcc=frcc,
            frss=frss,
            fzsc=fzsc,
            fzcs=fzcs,
            flsc=flsc,
            flcs=flcs,
            frsc=frsc,
            frcs=frcs,
            fzcc=fzcc,
            fzss=fzss,
            flcc=flcc,
            flss=flss,
        )
        gcr2_p, gcz2_p, gcl2_p = vmec_gcx2_from_tomnsps(
            frzl=frzl_pre,
            lconm1=bool(lconm1),
            apply_m1_constraints=False,
            include_edge=True,
            apply_scalxc=False,
            s=s,
        )
        f_norm1_j = jnp.asarray(f_norm1)
        finite_fnorm1 = jnp.isfinite(f_norm1_j)
        fsqr1 = jnp.where(finite_fnorm1, gcr2_p * f_norm1_j, jnp.asarray(0.0, dtype=jnp.asarray(gcr2_p).dtype))
        fsqz1 = jnp.where(finite_fnorm1, gcz2_p * f_norm1_j, jnp.asarray(0.0, dtype=jnp.asarray(gcz2_p).dtype))
        if bool(vmec2000_control):
            gcl2_full = lambda_preconditioned_full_norm(frzl_pre, use_jax=True)
            fsql1 = gcl2_full * delta_s
        else:
            fsql1 = gcl2_p * delta_s
        fsqr1_safe = jnp.where(jnp.isfinite(fsqr1), fsqr1, jnp.asarray(0.0, dtype=jnp.asarray(fsqr1).dtype))
        fsqz1_safe = jnp.where(jnp.isfinite(fsqz1), fsqz1, jnp.asarray(0.0, dtype=jnp.asarray(fsqz1).dtype))
        fsql1_safe = jnp.where(jnp.isfinite(fsql1), fsql1, jnp.asarray(0.0, dtype=jnp.asarray(fsql1).dtype))
        fsq1_safe = fsqr1_safe + fsqz1_safe + fsql1_safe
        return (
            pre_blocks,
            update_blocks,
            (gcr2_p, gcz2_p, gcl2_p, fsqr1_safe, fsqz1_safe, fsql1_safe, fsq1_safe),
        )

    compiled = jax.jit(_payload)
    return jit_cache_put(
        PRECOND_OUTPUT_PAYLOAD_JIT_CACHE,
        key,
        compiled,
        env_name="VMEC_JAX_PRECOND_OUTPUT_PAYLOAD_CACHE_SIZE",
        default=4,
    )


def preconditioner_apply_payload_jit(
    *,
    jmax: int,
    lthreed: bool,
    lasym: bool,
    use_precomputed: bool,
    use_lax_tridi: bool,
    has_lax_t: bool,
    has_frss: bool,
    has_fzcs: bool,
    has_frsc: bool,
    has_frcs: bool,
    has_fzcc: bool,
    has_fzss: bool,
    has_flcs: bool,
    has_flcc: bool,
    has_flss: bool,
    apply_lambda_update_scale: bool,
    vmec2000_control: bool,
    lconm1: bool,
    include_control_ptau: bool,
):
    """Return a cached fused GPU preconditioner-apply/payload kernel."""

    if not has_jax():
        return None
    key = (
        int(jmax),
        bool(lthreed),
        bool(lasym),
        bool(use_precomputed),
        bool(use_lax_tridi and has_lax_t),
        bool(has_lax_t),
        bool(has_frss),
        bool(has_fzcs),
        bool(has_frsc),
        bool(has_frcs),
        bool(has_fzcc),
        bool(has_fzss),
        bool(has_flcs),
        bool(has_flcc),
        bool(has_flss),
        bool(apply_lambda_update_scale),
        bool(vmec2000_control),
        bool(lconm1),
        bool(include_control_ptau),
    )
    cached = jit_cache_get(PRECOND_APPLY_PAYLOAD_JIT_CACHE, key)
    if cached is not None:
        return cached

    from ....preconditioner_1d_jax import _rz_preconditioner_apply_arrays

    use_rss = bool(lthreed)
    use_rsc = bool(lasym)
    use_rcs = bool(lthreed and lasym)
    use_zcs = bool(lthreed)
    use_zcc = bool(lasym)
    use_zss = bool(lthreed and lasym)

    def _payload_from_rz(
        frcc_rz,
        frss_rz,
        fzsc_rz,
        fzcs_rz,
        frsc_rz,
        frcs_rz,
        fzcc_rz,
        fzss_rz,
        flsc,
        flcs,
        flcc,
        flss,
        lam_prec,
        w_mode_mn,
        lambda_update_scale_j,
        f_norm1,
        delta_s,
        s,
        control_args,
    ):
        w = jnp.asarray(w_mode_mn)[None, :, :]
        lam_prec_j = jnp.asarray(lam_prec)

        frcc = jnp.asarray(frcc_rz)
        frss = jnp.asarray(frss_rz) if bool(has_frss) else None
        fzsc = jnp.asarray(fzsc_rz)
        fzcs = jnp.asarray(fzcs_rz) if bool(has_fzcs) else None
        flsc_pre = jnp.asarray(flsc) * lam_prec_j
        flcs_pre = (jnp.asarray(flcs) * lam_prec_j) if bool(has_flcs) else None

        frsc = jnp.asarray(frsc_rz) if bool(has_frsc) else jnp.zeros_like(frcc)
        frcs = jnp.asarray(frcs_rz) if bool(has_frcs) else jnp.zeros_like(frcc)
        fzcc = jnp.asarray(fzcc_rz) if bool(has_fzcc) else jnp.zeros_like(fzsc)
        fzss = jnp.asarray(fzss_rz) if bool(has_fzss) else jnp.zeros_like(fzsc)
        flcc_pre = (jnp.asarray(flcc) * lam_prec_j) if bool(has_flcc) else jnp.zeros_like(flsc_pre)
        flss_pre = (jnp.asarray(flss) * lam_prec_j) if bool(has_flss) else jnp.zeros_like(flsc_pre)

        frcc_u = frcc * w
        frss_u = (frss if frss is not None else jnp.zeros_like(frcc_u)) * w
        fzsc_u = fzsc * w
        fzcs_u = (fzcs if fzcs is not None else jnp.zeros_like(fzsc_u)) * w
        flsc_u = flsc_pre * w
        flcs_u = (flcs_pre if flcs_pre is not None else jnp.zeros_like(flsc_u)) * w
        frsc_u = frsc * w
        frcs_u = frcs * w
        fzcc_u = fzcc * w
        fzss_u = fzss * w
        flcc_u = flcc_pre * w
        flss_u = flss_pre * w

        if bool(apply_lambda_update_scale):
            lambda_update_scale_j = jnp.asarray(lambda_update_scale_j, dtype=flsc_u.dtype)
            flsc_u = flsc_u * lambda_update_scale_j
            flcs_u = flcs_u * lambda_update_scale_j
            flcc_u = flcc_u * lambda_update_scale_j
            flss_u = flss_u * lambda_update_scale_j

        frzl_pre = TomnspsRZL(
            frcc=frcc,
            frss=frss,
            fzsc=fzsc,
            fzcs=fzcs,
            flsc=flsc_pre,
            flcs=flcs_pre,
            frsc=frsc,
            frcs=frcs,
            fzcc=fzcc,
            fzss=fzss,
            flcc=flcc_pre,
            flss=flss_pre,
        )
        gcr2_p, gcz2_p, gcl2_p = vmec_gcx2_from_tomnsps(
            frzl=frzl_pre,
            lconm1=bool(lconm1),
            apply_m1_constraints=False,
            include_edge=True,
            apply_scalxc=False,
            s=s,
        )
        f_norm1_j = jnp.asarray(f_norm1)
        finite_fnorm1 = jnp.isfinite(f_norm1_j)
        fsqr1 = jnp.where(finite_fnorm1, gcr2_p * f_norm1_j, jnp.asarray(0.0, dtype=jnp.asarray(gcr2_p).dtype))
        fsqz1 = jnp.where(finite_fnorm1, gcz2_p * f_norm1_j, jnp.asarray(0.0, dtype=jnp.asarray(gcz2_p).dtype))
        if bool(vmec2000_control):
            gcl2_full = lambda_preconditioned_full_norm(frzl_pre, use_jax=True)
            fsql1 = gcl2_full * delta_s
        else:
            fsql1 = gcl2_p * delta_s
        fsqr1_safe = jnp.where(jnp.isfinite(fsqr1), fsqr1, jnp.asarray(0.0, dtype=jnp.asarray(fsqr1).dtype))
        fsqz1_safe = jnp.where(jnp.isfinite(fsqz1), fsqz1, jnp.asarray(0.0, dtype=jnp.asarray(fsqz1).dtype))
        fsql1_safe = jnp.where(jnp.isfinite(fsql1), fsql1, jnp.asarray(0.0, dtype=jnp.asarray(fsql1).dtype))
        fsq1_safe = fsqr1_safe + fsqz1_safe + fsql1_safe
        if bool(include_control_ptau):
            (
                pru_even,
                pru_odd,
                pzu_even,
                pzu_odd,
                pr1_even,
                pr1_odd,
                pz1_even,
                pz1_odd,
                pshalf,
                ohs,
            ) = control_args
            ptau_min, ptau_max = ptau_compute_jit(
                pru_even,
                pru_odd,
                pzu_even,
                pzu_odd,
                pr1_even,
                pr1_odd,
                pz1_even,
                pz1_odd,
                pshalf,
                ohs,
            )
            return (
                (frcc, frss, fzsc, fzcs, flsc_pre, flcs_pre, frsc, frcs, fzcc, fzss, flcc_pre, flss_pre),
                (frcc_u, frss_u, fzsc_u, fzcs_u, flsc_u, flcs_u, frsc_u, frcs_u, fzcc_u, fzss_u, flcc_u, flss_u),
                (gcr2_p, gcz2_p, gcl2_p, fsqr1_safe, fsqz1_safe, fsql1_safe, fsq1_safe),
                (fsq1_safe, ptau_min, ptau_max),
            )
        return (
            (frcc, frss, fzsc, fzcs, flsc_pre, flcs_pre, frsc, frcs, fzcc, fzss, flcc_pre, flss_pre),
            (frcc_u, frss_u, fzsc_u, fzcs_u, flsc_u, flcs_u, frsc_u, frcs_u, fzcc_u, fzss_u, flcc_u, flss_u),
            (gcr2_p, gcz2_p, gcl2_p, fsqr1_safe, fsqz1_safe, fsql1_safe, fsq1_safe),
        )

    @jit
    def _apply_payload(
        frcc,
        fzsc,
        frss,
        fzcs,
        frsc,
        frcs,
        fzcc,
        fzss,
        ar,
        br,
        dr,
        cr,
        ir,
        az,
        bz,
        dz,
        cz,
        iz,
        dlr_t,
        dr_t,
        dur_t,
        dlz_t,
        dz_t,
        duz_t,
        flsc,
        flcs,
        flcc,
        flss,
        lam_prec,
        w_mode_mn,
        lambda_update_scale_j,
        f_norm1,
        delta_s,
        s,
        *control_args,
    ):
        frcc_rz, frss_rz, fzsc_rz, fzcs_rz, frsc_rz, frcs_rz, fzcc_rz, fzss_rz = (
            _rz_preconditioner_apply_arrays(
                ar=ar,
                br=br,
                dr=dr,
                cr=cr,
                ir=ir,
                az=az,
                bz=bz,
                dz=dz,
                cz=cz,
                iz=iz,
                dlr_t=dlr_t,
                dr_t=dr_t,
                dur_t=dur_t,
                dlz_t=dlz_t,
                dz_t=dz_t,
                duz_t=duz_t,
                frcc=frcc,
                frss=frss,
                fzsc=fzsc,
                fzcs=fzcs,
                frsc=frsc,
                frcs=frcs,
                fzcc=fzcc,
                fzss=fzss,
                jmax=int(jmax),
                use_precomputed=bool(use_precomputed),
                use_lax_tridi=bool(use_lax_tridi and has_lax_t),
                use_rss=use_rss,
                use_rsc=use_rsc,
                use_rcs=use_rcs,
                use_zcs=use_zcs,
                use_zcc=use_zcc,
                use_zss=use_zss,
            )
        )
        return _payload_from_rz(
            frcc_rz,
            frss_rz,
            fzsc_rz,
            fzcs_rz,
            frsc_rz,
            frcs_rz,
            fzcc_rz,
            fzss_rz,
            flsc,
            flcs,
            flcc,
            flss,
            lam_prec,
            w_mode_mn,
            lambda_update_scale_j,
            f_norm1,
            delta_s,
            s,
            control_args,
        )

    return jit_cache_put(
        PRECOND_APPLY_PAYLOAD_JIT_CACHE,
        key,
        _apply_payload,
        env_name="VMEC_JAX_PRECOND_APPLY_PAYLOAD_CACHE_SIZE",
        default=8,
    )


def accepted_control_payload_jit():
    """Return a cached JIT helper for accepted-step scalar control payloads."""

    if not has_jax():
        return None
    key: tuple[Any, ...] = ()
    cached = jit_cache_get(ACCEPTED_CONTROL_PAYLOAD_JIT_CACHE, key)
    if cached is not None:
        return cached

    @jax.jit
    def _payload(
        fsq1_safe,
        pru_even,
        pru_odd,
        pzu_even,
        pzu_odd,
        pr1_even,
        pr1_odd,
        pz1_even,
        pz1_odd,
        pshalf,
        ohs,
    ):
        ptau_min, ptau_max = ptau_compute_jit(
            pru_even,
            pru_odd,
            pzu_even,
            pzu_odd,
            pr1_even,
            pr1_odd,
            pz1_even,
            pz1_odd,
            pshalf,
            ohs,
        )
        return jnp.asarray(fsq1_safe), ptau_min, ptau_max

    return jit_cache_put(
        ACCEPTED_CONTROL_PAYLOAD_JIT_CACHE,
        key,
        _payload,
        env_name="VMEC_JAX_ACCEPTED_CONTROL_PAYLOAD_CACHE_SIZE",
        default=2,
    )


def preconditioner_apply_payload_fused(
    *,
    frzl_in: TomnspsRZL,
    mats: dict[str, Any],
    jmax: int,
    cfg,
    lam_prec,
    w_mode_mn,
    lambda_update_scale_j,
    f_norm1,
    delta_s,
    s,
    use_precomputed: bool | None,
    use_lax_tridi: bool | None,
    apply_lambda_update_scale: bool,
    vmec2000_control: bool,
    lconm1: bool,
    include_control_ptau: bool = False,
    control_ptau_arrays: tuple[Any, ...] | None = None,
    control_ptau_pshalf: Any = None,
    control_ptau_ohs: Any = None,
    apply_payload_jit_func: Callable[..., Any] = preconditioner_apply_payload_jit,
):
    """Apply R/Z preconditioning and build update/diagnostic payload in one dispatch."""

    from ....preconditioner_1d_jax import _get_env_tridi_flags

    lthreed = bool(getattr(cfg, "lthreed", False))
    lasym = bool(getattr(cfg, "lasym", False))
    if use_precomputed is None or use_lax_tridi is None:
        env_pre, env_lax = _get_env_tridi_flags()
        if use_precomputed is None:
            use_precomputed = env_pre
        if use_lax_tridi is None:
            use_lax_tridi = env_lax
    has_cr_ir = ("cr" in mats) and ("ir" in mats) and ("cz" in mats) and ("iz" in mats)
    if not has_cr_ir:
        use_precomputed = False

    has_frss = frzl_in.frss is not None
    has_fzcs = frzl_in.fzcs is not None
    has_frsc = getattr(frzl_in, "frsc", None) is not None
    has_frcs = getattr(frzl_in, "frcs", None) is not None
    has_fzcc = getattr(frzl_in, "fzcc", None) is not None
    has_fzss = getattr(frzl_in, "fzss", None) is not None
    has_flcs = frzl_in.flcs is not None
    has_flcc = getattr(frzl_in, "flcc", None) is not None
    has_flss = getattr(frzl_in, "flss", None) is not None
    include_control_ptau = (
        bool(include_control_ptau)
        and control_ptau_arrays is not None
        and len(tuple(control_ptau_arrays)) == 8
    )

    has_lax_t = (
        ("dlr_t" in mats)
        and ("dr_t" in mats)
        and ("dur_t" in mats)
        and ("dlz_t" in mats)
        and ("dz_t" in mats)
        and ("duz_t" in mats)
    )
    apply_payload = apply_payload_jit_func(
        jmax=int(jmax),
        lthreed=lthreed,
        lasym=lasym,
        use_precomputed=bool(use_precomputed),
        use_lax_tridi=bool(use_lax_tridi),
        has_lax_t=has_lax_t,
        has_frss=has_frss,
        has_fzcs=has_fzcs,
        has_frsc=has_frsc,
        has_frcs=has_frcs,
        has_fzcc=has_fzcc,
        has_fzss=has_fzss,
        has_flcs=has_flcs,
        has_flcc=has_flcc,
        has_flss=has_flss,
        apply_lambda_update_scale=bool(apply_lambda_update_scale),
        vmec2000_control=bool(vmec2000_control),
        lconm1=bool(lconm1),
        include_control_ptau=bool(include_control_ptau),
    )

    frcc = frzl_in.frcc
    fzsc = frzl_in.fzsc
    frss = frzl_in.frss if has_frss else frcc
    fzcs = frzl_in.fzcs if has_fzcs else fzsc
    frsc = getattr(frzl_in, "frsc", None) if has_frsc else frcc
    frcs = getattr(frzl_in, "frcs", None) if has_frcs else frcc
    fzcc = getattr(frzl_in, "fzcc", None) if has_fzcc else fzsc
    fzss = getattr(frzl_in, "fzss", None) if has_fzss else fzsc

    ar = mats["ar"]
    br = mats["br"]
    dr = mats["dr"]
    cr = mats.get("cr", ar)
    ir = mats.get("ir", dr)
    az = mats["az"]
    bz = mats["bz"]
    dz = mats["dz"]
    cz = mats.get("cz", az)
    iz = mats.get("iz", dz)
    dlr_t = mats.get("dlr_t", ar)
    dr_t = mats.get("dr_t", ar)
    dur_t = mats.get("dur_t", ar)
    dlz_t = mats.get("dlz_t", az)
    dz_t = mats.get("dz_t", az)
    duz_t = mats.get("duz_t", az)

    args = (
        frcc,
        fzsc,
        frss,
        fzcs,
        frsc,
        frcs,
        fzcc,
        fzss,
        ar,
        br,
        dr,
        cr,
        ir,
        az,
        bz,
        dz,
        cz,
        iz,
        dlr_t,
        dr_t,
        dur_t,
        dlz_t,
        dz_t,
        duz_t,
        frzl_in.flsc,
        frzl_in.flcs,
        getattr(frzl_in, "flcc", None),
        getattr(frzl_in, "flss", None),
        lam_prec,
        w_mode_mn,
        lambda_update_scale_j,
        f_norm1,
        delta_s,
        s,
    )
    if bool(include_control_ptau):
        args = (
            *args,
            *control_ptau_arrays,
            control_ptau_pshalf,
            control_ptau_ohs,
        )
    return apply_payload(*args)


if has_jax():

    @jax.jit
    def ptau_compute_jit(
        pru_even,
        pru_odd,
        pzu_even,
        pzu_odd,
        pr1_even,
        pr1_odd,
        pz1_even,
        pz1_odd,
        pshalf,
        ohs,
    ):
        """Compute ptau min/max without redefining a hot JIT helper per solve."""

        pshalf = pshalf.astype(pru_even.dtype)
        ohs = ohs.astype(pru_even.dtype)
        dphids = jnp.asarray(0.25, dtype=pru_even.dtype)
        psh = pshalf[1:][:, None, None]
        psh_safe = jnp.where(psh != 0.0, psh, jnp.ones_like(psh))
        ru12 = 0.5 * (pru_even[1:] + pru_even[:-1] + psh * (pru_odd[1:] + pru_odd[:-1]))
        pzs = ohs * ((pz1_even[1:] - pz1_even[:-1]) + psh * (pz1_odd[1:] - pz1_odd[:-1]))
        ptau = ru12 * pzs + dphids * (
            pru_odd[1:] * pz1_odd[1:]
            + pru_odd[:-1] * pz1_odd[:-1]
            + (pru_even[1:] * pz1_odd[1:] + pru_even[:-1] * pz1_odd[:-1]) / psh_safe
        )
        pzu12 = 0.5 * (pzu_even[1:] + pzu_even[:-1] + psh * (pzu_odd[1:] + pzu_odd[:-1]))
        prs = ohs * ((pr1_even[1:] - pr1_even[:-1]) + psh * (pr1_odd[1:] - pr1_odd[:-1]))
        ptau = (
            ptau
            - prs * pzu12
            - dphids
            * (
                pzu_odd[1:] * pr1_odd[1:]
                + pzu_odd[:-1] * pr1_odd[:-1]
                + (pzu_even[1:] * pr1_odd[1:] + pzu_even[:-1] * pr1_odd[:-1]) / psh_safe
            )
        )
        return jnp.min(ptau), jnp.max(ptau)
else:
    ptau_compute_jit = None
