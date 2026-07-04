"""Preconditioner helpers for fixed-boundary VMEC solve paths."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import os
from typing import Any, Callable, Mapping, NamedTuple

import numpy as np

from ...._compat import jax, jnp
from ....state import VMECState
from ....kernels.tomnsp import TomnspsRZL
from ..optimization.constraints import scale_mode_slice_np


@partial(jax.jit, static_argnames=("has_frss", "has_fzcs", "has_frsc", "has_fzcc"))
def _scale_m1_preconditioner_channels_jit(
    frss,
    fzcs,
    frsc,
    fzcc,
    fac_r,
    fac_z,
    *,
    has_frss: bool,
    has_fzcs: bool,
    has_frsc: bool,
    has_fzcc: bool,
):
    """Scale VMEC m=1 R/Z RHS channels in one compiled operation."""

    fac_r = jnp.asarray(fac_r)
    fac_z = jnp.asarray(fac_z)

    def _scale(arr, fac):
        return jnp.asarray(arr).at[:, 1, :].multiply(jnp.asarray(fac, dtype=jnp.asarray(arr).dtype)[:, None])

    return (
        _scale(frss, fac_r) if bool(has_frss) else None,
        _scale(fzcs, fac_z) if bool(has_fzcs) else None,
        _scale(frsc, fac_r) if bool(has_frsc) else None,
        _scale(fzcc, fac_z) if bool(has_fzcc) else None,
    )


def resolve_preconditioner_tridi_policies(
    *, use_precomputed: bool | None, use_lax_tridi: bool | None
) -> tuple[bool, bool]:
    """Resolve preconditioner tridiagonal solver policy from explicit flags/env."""

    env_precomputed = os.getenv("VMEC_JAX_TRIDI_PRECOMPUTE", "0").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
    )
    env_lax_tridi = os.getenv("VMEC_JAX_TRIDI_SOLVE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "lax",
        "force",
    )
    return (
        bool(env_precomputed) if use_precomputed is None else bool(use_precomputed),
        bool(env_lax_tridi) if use_lax_tridi is None else bool(use_lax_tridi),
    )


class PreconditionerCacheDecision(NamedTuple):
    """Cache/reassembly decision for residual-iteration 1D preconditioners."""

    need_prec_reassemble: bool
    can_reuse_bcovar_seeded_precond: bool
    need_prec_refresh: bool


class LambdaPreconditionerOutputs(NamedTuple):
    """Resolved lambda-preconditioner payload for optional debug dumps."""

    lam_prec: Any
    faclam_dump: Any | None
    lam_debug: Any | None


class PreconditionerCacheSnapshot(NamedTuple):
    """Residual-iteration cache fields for the 1D preconditioner state."""

    valid: bool
    precond_diag: Any
    tcon: Any
    norms: Any
    rz_scale: Any
    l_scale: Any
    rz_norm: Any
    f_norm1: Any
    prec_rz_mats: Any
    prec_rz_jmax: int | None
    prec_lam_prec: Any
    prec_faclam: Any | None
    prec_lam_debug: Any | None


_PRECONDITIONER_CACHE_RESUME_KEYS = {
    "vmec2000_cache_valid": "valid",
    "cache_precond_diag": "precond_diag",
    "cache_tcon": "tcon",
    "cache_norms": "norms",
    "cache_rz_scale": "rz_scale",
    "cache_l_scale": "l_scale",
    "cache_rz_norm": "rz_norm",
    "cache_f_norm1": "f_norm1",
    "cache_prec_rz_mats": "prec_rz_mats",
    "cache_prec_rz_jmax": "prec_rz_jmax",
    "cache_prec_lam_prec": "prec_lam_prec",
    "cache_prec_faclam": "prec_faclam",
    "cache_prec_lam_debug": "prec_lam_debug",
}


@dataclass
class PreconditionerCacheState:
    """Mutable residual-loop cache for VMEC2000 1D preconditioner payloads."""

    valid: bool = False
    precond_diag: Any = None
    tcon: Any = None
    norms: Any = None
    rz_scale: Any = None
    l_scale: Any = None
    rz_norm: Any = None
    f_norm1: Any = None
    prec_rz_mats: Any = None
    prec_rz_jmax: int | None = None
    prec_lam_prec: Any = None
    prec_faclam: Any | None = None
    prec_lam_debug: Any | None = None

    def clear(self) -> None:
        """Invalidate all cached preconditioner payloads in place."""

        self.valid = False
        self.precond_diag = None
        self.tcon = None
        self.norms = None
        self.rz_scale = None
        self.l_scale = None
        self.rz_norm = None
        self.f_norm1 = None
        self.prec_rz_mats = None
        self.prec_rz_jmax = None
        self.prec_lam_prec = None
        self.prec_faclam = None
        self.prec_lam_debug = None

    def update_from_resume_state(self, resume_state: Mapping[str, Any]) -> None:
        """Restore legacy resume-state cache fields into the mutable cache."""

        for resume_key, attr_name in _PRECONDITIONER_CACHE_RESUME_KEYS.items():
            if resume_key in resume_state:
                value = resume_state[resume_key]
                if attr_name == "valid":
                    value = bool(value)
                setattr(self, attr_name, value)

    def legacy_resume_payload(self) -> dict[str, Any]:
        """Return the public resume-state keys expected by existing drivers."""

        return {
            resume_key: getattr(self, attr_name)
            for resume_key, attr_name in _PRECONDITIONER_CACHE_RESUME_KEYS.items()
        }


class PreconditionerCacheUpdate(NamedTuple):
    """Resolved current preconditioner payload plus updated cache entries."""

    decision: PreconditionerCacheDecision
    lam_prec: Any
    faclam_dump: Any | None
    lam_debug: Any | None
    mats: Any
    jmax: Any
    cache_prec_lam_prec: Any
    cache_prec_faclam: Any | None
    cache_prec_lam_debug: Any | None
    cache_prec_rz_mats: Any
    cache_prec_rz_jmax: int | None


def empty_preconditioner_cache_snapshot() -> PreconditionerCacheSnapshot:
    """Return the invalid/empty residual-iteration preconditioner cache."""

    return PreconditionerCacheSnapshot(
        valid=False,
        precond_diag=None,
        tcon=None,
        norms=None,
        rz_scale=None,
        l_scale=None,
        rz_norm=None,
        f_norm1=None,
        prec_rz_mats=None,
        prec_rz_jmax=None,
        prec_lam_prec=None,
        prec_faclam=None,
        prec_lam_debug=None,
    )


def lambda_preconditioner_outputs(
    bc: Any,
    *,
    need_lam_prec: bool,
    need_lamcal: bool,
    lambda_preconditioner_func: Callable[..., Any],
) -> LambdaPreconditionerOutputs:
    """Call the lambda preconditioner with the dump payloads actually needed."""

    if bool(need_lamcal):
        if bool(need_lam_prec):
            lam_prec, faclam_dump, lam_debug = lambda_preconditioner_func(
                bc,
                return_faclam=True,
                return_debug=True,
            )
        else:
            lam_prec, lam_debug = lambda_preconditioner_func(bc, return_debug=True)
            faclam_dump = None
    else:
        if bool(need_lam_prec):
            lam_prec, faclam_dump = lambda_preconditioner_func(bc, return_faclam=True)
        else:
            lam_prec = lambda_preconditioner_func(bc)
            faclam_dump = None
        lam_debug = None

    return LambdaPreconditionerOutputs(
        lam_prec=lam_prec,
        faclam_dump=faclam_dump,
        lam_debug=lam_debug,
    )


def resolve_preconditioner_cache_decision(
    *,
    precond_traced: bool,
    vmec2000_cache_valid: bool,
    need_bcovar_update: bool,
    precond_cache_seeded_from_bcovar_update: bool,
    need_lam_prec: bool,
    need_lamcal: bool,
    cache_prec_lam_prec: Any,
    cache_prec_rz_mats: Any,
    cache_prec_rz_jmax: int | None,
    precond_expected_jmax: int,
    can_reassemble_func: Callable[[Any], bool],
) -> PreconditionerCacheDecision:
    """Resolve cached preconditioner reuse, refresh, and reassembly policy."""

    need_prec_reassemble = (
        (not bool(precond_traced))
        and (cache_prec_rz_jmax is not None)
        and (int(cache_prec_rz_jmax) != int(precond_expected_jmax))
        and bool(can_reassemble_func(cache_prec_rz_mats))
    )
    can_reuse_bcovar_seeded_precond = (
        bool(precond_cache_seeded_from_bcovar_update)
        and (not bool(precond_traced))
        and (not bool(need_lam_prec))
        and (not bool(need_lamcal))
        and (cache_prec_lam_prec is not None)
        and (cache_prec_rz_mats is not None)
        and (cache_prec_rz_jmax is not None)
    )
    need_prec_refresh = (
        bool(precond_traced)
        or (not bool(vmec2000_cache_valid))
        or (cache_prec_lam_prec is None)
        or (cache_prec_rz_mats is None)
        or (cache_prec_rz_jmax is None)
        or (bool(need_bcovar_update) and (not bool(can_reuse_bcovar_seeded_precond)))
        or (
            (cache_prec_rz_jmax is not None)
            and (int(cache_prec_rz_jmax) != int(precond_expected_jmax))
            and (not bool(need_prec_reassemble))
        )
    )
    return PreconditionerCacheDecision(
        need_prec_reassemble=bool(need_prec_reassemble),
        can_reuse_bcovar_seeded_precond=bool(can_reuse_bcovar_seeded_precond),
        need_prec_refresh=bool(need_prec_refresh),
    )


def update_preconditioner_cache(
    *,
    bc: Any,
    k: Any,
    cfg: Any,
    precond_traced: bool,
    vmec2000_cache_valid: bool,
    need_bcovar_update: bool,
    precond_cache_seeded_from_bcovar_update: bool,
    need_lam_prec: bool,
    need_lamcal: bool,
    cache_prec_lam_prec: Any,
    cache_prec_faclam: Any | None,
    cache_prec_lam_debug: Any | None,
    cache_prec_rz_mats: Any,
    cache_prec_rz_jmax: int | None,
    precond_expected_jmax: int,
    precond_jmax_override: int | None,
    preconditioner_use_precomputed_tridi: bool,
    preconditioner_use_lax_tridi: bool,
    lambda_preconditioner_func: Callable[..., Any],
    rz_preconditioner_matrices_func: Callable[..., Any],
    rz_preconditioner_matrices_reassemble_func: Callable[..., Any],
    can_reassemble_func: Callable[[Any], bool],
) -> PreconditionerCacheUpdate:
    """Refresh, reassemble, or reuse cached 1D preconditioner payloads."""

    decision = resolve_preconditioner_cache_decision(
        precond_traced=bool(precond_traced),
        vmec2000_cache_valid=bool(vmec2000_cache_valid),
        need_bcovar_update=bool(need_bcovar_update),
        precond_cache_seeded_from_bcovar_update=bool(precond_cache_seeded_from_bcovar_update),
        need_lam_prec=bool(need_lam_prec),
        need_lamcal=bool(need_lamcal),
        cache_prec_lam_prec=cache_prec_lam_prec,
        cache_prec_rz_mats=cache_prec_rz_mats,
        cache_prec_rz_jmax=cache_prec_rz_jmax,
        precond_expected_jmax=int(precond_expected_jmax),
        can_reassemble_func=can_reassemble_func,
    )

    if decision.need_prec_refresh:
        lam_outputs = lambda_preconditioner_outputs(
            bc,
            need_lam_prec=bool(need_lam_prec),
            need_lamcal=bool(need_lamcal),
            lambda_preconditioner_func=lambda_preconditioner_func,
        )
        mats, _jmin, jmax = rz_preconditioner_matrices_func(
            bc=bc,
            k=k,
            jmax_override=precond_jmax_override,
            use_precomputed=preconditioner_use_precomputed_tridi,
            use_lax_tridi=preconditioner_use_lax_tridi,
        )
        return PreconditionerCacheUpdate(
            decision=decision,
            lam_prec=lam_outputs.lam_prec,
            faclam_dump=lam_outputs.faclam_dump,
            lam_debug=lam_outputs.lam_debug,
            mats=mats,
            jmax=jmax,
            cache_prec_lam_prec=lam_outputs.lam_prec,
            cache_prec_faclam=lam_outputs.faclam_dump,
            cache_prec_lam_debug=lam_outputs.lam_debug,
            cache_prec_rz_mats=mats,
            cache_prec_rz_jmax=None if bool(precond_traced) else int(jmax),
        )

    lam_prec = cache_prec_lam_prec
    faclam_dump = cache_prec_faclam if bool(need_lam_prec) else None
    lam_debug = cache_prec_lam_debug if bool(need_lamcal) else None
    if decision.need_prec_reassemble:
        mats, _jmin, jmax = rz_preconditioner_matrices_reassemble_func(
            mats=cache_prec_rz_mats,
            cfg=cfg,
            jmax_override=precond_jmax_override,
        )
        cache_prec_rz_mats = mats
        cache_prec_rz_jmax = None if bool(precond_traced) else int(jmax)
    else:
        mats = cache_prec_rz_mats
        jmax = cache_prec_rz_jmax

    return PreconditionerCacheUpdate(
        decision=decision,
        lam_prec=lam_prec,
        faclam_dump=faclam_dump,
        lam_debug=lam_debug,
        mats=mats,
        jmax=jmax,
        cache_prec_lam_prec=cache_prec_lam_prec,
        cache_prec_faclam=cache_prec_faclam,
        cache_prec_lam_debug=cache_prec_lam_debug,
        cache_prec_rz_mats=cache_prec_rz_mats,
        cache_prec_rz_jmax=cache_prec_rz_jmax,
    )


def radial_tridi_smooth_dirichlet(
    rhs,
    *,
    alpha: float,
    skip_nonpositive: bool = False,
    allow_3d: bool = True,
):
    """Solve the Dirichlet tri-diagonal smoothing system along the radial axis."""

    if skip_nonpositive and alpha <= 0.0:
        return rhs
    rhs = jnp.asarray(rhs)
    if rhs.ndim == 2:
        rhs2 = rhs
        orig_shape = None
    elif rhs.ndim < 2:
        raise ValueError(f"expected (ns,...) with ndim>=2, got {rhs.shape}")
    elif allow_3d and rhs.ndim == 3:
        ns = int(rhs.shape[0])
        rhs2 = rhs.reshape(ns, -1)
        orig_shape = rhs.shape
    elif allow_3d:
        raise ValueError(f"expected (ns,K) or (ns,M,N), got {rhs.shape}")
    else:
        raise ValueError(f"expected (ns,...) with ndim>=2, got {rhs.shape}")
    ns = int(rhs2.shape[0])
    if ns < 3:
        return rhs
    alpha_arr = jnp.asarray(alpha, dtype=rhs2.dtype)
    a = -alpha_arr
    b = 1.0 + 2.0 * alpha_arr
    c = -alpha_arr

    x0 = rhs2[0]
    xN = rhs2[-1]
    d = rhs2[1:-1]
    d = d.at[0].add(alpha_arr * x0)
    d = d.at[-1].add(alpha_arr * xN)

    n = int(d.shape[0])
    if n == 1:
        x_int = d / b
    else:
        cp0 = c / b
        dp0 = d[0] / b

        def fwd(carry, di):
            """Run the forward rule for the custom derivative."""
            cp_prev, dp_prev = carry
            denom = b - a * cp_prev
            cp = c / denom
            dp = (di - a * dp_prev) / denom
            return (cp, dp), (cp, dp)

        (_cp_last, dp_last), (cp_rest, dp_rest) = jax.lax.scan(fwd, (cp0, dp0), d[1:])
        cp = jnp.concatenate([jnp.asarray([cp0]), cp_rest], axis=0)
        dp = jnp.concatenate([dp0[None, :], dp_rest], axis=0)

        def bwd(x_next, items):
            """Run the transpose rule for the custom derivative."""
            cpi, dpi = items
            xi = dpi - cpi * x_next
            return xi, xi

        _x0, x_rev = jax.lax.scan(bwd, dp_last, (cp[:-1], dp[:-1]), reverse=True)
        x_int = jnp.concatenate([x_rev, dp_last[None, :]], axis=0)

    out = jnp.concatenate([x0[None, :], x_int, xN[None, :]], axis=0)
    if orig_shape is not None:
        out = out.reshape(orig_shape)
    return out


def apply_preconditioner(
    grad: VMECState,
    static,
    *,
    kind: str,
    exponent: float = 1.0,
    radial_alpha: float = 0.0,
) -> VMECState:
    """Apply a simple Fourier/radial preconditioner to all state blocks."""

    kind = str(kind).strip().lower()
    if kind == "none":
        return grad

    kinds = [k.strip() for k in kind.replace("+", ",").split(",") if k.strip()]
    if not kinds:
        return grad

    exponent = float(exponent)
    if ("mode_diag" in kinds) and exponent <= 0.0:
        raise ValueError("preconditioner exponent must be > 0 for mode_diag")
    radial_alpha = float(radial_alpha)
    if ("radial_tridi" in kinds) and radial_alpha <= 0.0:
        raise ValueError("radial_alpha must be > 0 for radial_tridi")

    def _apply_mode_diag(g: VMECState) -> VMECState:
        m = jnp.asarray(static.modes.m)
        n = jnp.asarray(static.modes.n)
        nfp = float(static.cfg.nfp)
        k2 = m.astype(jnp.float64) ** 2 + (n.astype(jnp.float64) * nfp) ** 2
        # (1 + k2)^(-exponent) avoids singularity at (m,n)=(0,0).
        w = (1.0 + k2) ** (-exponent)
        w = w.astype(jnp.asarray(g.Rcos).dtype)

        def _scale(a):
            a = jnp.asarray(a)
            return a * w[None, :]

        return VMECState(
            layout=g.layout,
            Rcos=_scale(g.Rcos),
            Rsin=_scale(g.Rsin),
            Zcos=_scale(g.Zcos),
            Zsin=_scale(g.Zsin),
            Lcos=_scale(g.Lcos),
            Lsin=_scale(g.Lsin),
        )

    def _apply_radial_tridi(g: VMECState) -> VMECState:
        return VMECState(
            layout=g.layout,
            Rcos=radial_tridi_smooth_dirichlet(g.Rcos, alpha=radial_alpha),
            Rsin=radial_tridi_smooth_dirichlet(g.Rsin, alpha=radial_alpha),
            Zcos=radial_tridi_smooth_dirichlet(g.Zcos, alpha=radial_alpha),
            Zsin=radial_tridi_smooth_dirichlet(g.Zsin, alpha=radial_alpha),
            Lcos=radial_tridi_smooth_dirichlet(g.Lcos, alpha=radial_alpha),
            Lsin=radial_tridi_smooth_dirichlet(g.Lsin, alpha=radial_alpha),
        )

    g = grad
    for k in kinds:
        if k == "mode_diag":
            g = _apply_mode_diag(g)
        elif k == "radial_tridi":
            g = _apply_radial_tridi(g)
        else:
            raise ValueError(f"Unknown preconditioner kind={k!r}")
    return g


def metric_surface_precond_scales_jax(*, guu, r12, bsubu, bsubv, w_ang):
    """Approximate radial/lambda preconditioner scales with tracer-safe ops."""

    w3 = jnp.asarray(w_ang, dtype=jnp.asarray(guu).dtype)[None, :, :]
    rz_denom = jnp.sum((guu * (r12 * r12)) * w3, axis=(1, 2))
    rz_scale = jnp.where(rz_denom > 0.0, 1.0 / jnp.sqrt(jnp.maximum(rz_denom, 1e-300)), 1.0)
    l_denom = jnp.sum(((bsubu * bsubu) + (bsubv * bsubv)) * w3, axis=(1, 2))
    l_scale = jnp.where(l_denom > 0.0, 1.0 / jnp.sqrt(jnp.maximum(l_denom, 1e-300)), 1.0)
    return jnp.clip(rz_scale, 1e-4, 1e2), jnp.clip(l_scale, 1e-4, 1e2)


def metric_surface_precond_scales_np(*, guu, r12, bsubu, bsubv, w_ang) -> tuple[np.ndarray, np.ndarray]:
    """Host NumPy variant of the first-step metric preconditioner scales."""

    guu_arr = np.asarray(guu)
    r12_arr = np.asarray(r12)
    bsubu_arr = np.asarray(bsubu)
    bsubv_arr = np.asarray(bsubv)
    w3 = np.asarray(w_ang, dtype=guu_arr.dtype)[None, :, :]
    rz_denom = np.sum((guu_arr * (r12_arr * r12_arr)) * w3, axis=(1, 2))
    rz_scale = np.where(rz_denom > 0.0, 1.0 / np.sqrt(np.maximum(rz_denom, 1e-300)), 1.0)
    l_denom = np.sum(((bsubu_arr * bsubu_arr) + (bsubv_arr * bsubv_arr)) * w3, axis=(1, 2))
    l_scale = np.where(l_denom > 0.0, 1.0 / np.sqrt(np.maximum(l_denom, 1e-300)), 1.0)
    return np.clip(rz_scale, 1e-4, 1e2), np.clip(l_scale, 1e-4, 1e2)


def metric_surface_precond_from_bcovar_jax(*, bc, trig, wint_from_trig_func=None, scales_func=None):
    """Return traced metric preconditioner scales from a bcovar payload."""

    if wint_from_trig_func is None:
        from ....kernels.residue import vmec_wint_from_trig as wint_from_trig_func
    if scales_func is None:
        scales_func = metric_surface_precond_scales_jax

    guu = bc.guu
    r12 = bc.jac.r12
    bsubu = bc.bsubu
    bsubv = bc.bsubv
    nzeta = int(guu.shape[2])
    w_ang = jnp.asarray(wint_from_trig_func(trig, nzeta=nzeta), dtype=guu.dtype)
    return scales_func(guu=guu, r12=r12, bsubu=bsubu, bsubv=bsubv, w_ang=w_ang)


def metric_surface_precond_from_bcovar_np(
    *,
    bc,
    trig,
    wint_from_trig_func=None,
    scales_func=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return host metric preconditioner scales from a bcovar payload."""

    if wint_from_trig_func is None:
        from ....kernels.residue import vmec_wint_from_trig as wint_from_trig_func
    if scales_func is None:
        scales_func = metric_surface_precond_scales_np

    guu = np.asarray(bc.guu)
    r12 = np.asarray(bc.jac.r12)
    bsubu = np.asarray(bc.bsubu)
    bsubv = np.asarray(bc.bsubv)
    nzeta = int(guu.shape[2])
    w_ang = np.asarray(wint_from_trig_func(trig, nzeta=nzeta), dtype=guu.dtype)
    return scales_func(guu=guu, r12=r12, bsubu=bsubu, bsubv=bsubv, w_ang=w_ang)


def pshalf_from_s_np(s_arr) -> np.ndarray:
    """Return VMEC-style half-mesh square-root radial coordinate from full mesh."""

    s_arr = np.asarray(s_arr, dtype=float)
    if s_arr.size < 2:
        return np.sqrt(np.maximum(s_arr, 0.0))
    sh = 0.5 * (s_arr[1:] + s_arr[:-1])
    p = np.concatenate([sh[:1], sh], axis=0)
    return np.sqrt(np.maximum(p, 0.0))


def pshalf_from_s_jax(s_arr, dtype):
    """JAX variant of :func:`pshalf_from_s_np`."""

    s_arr = jnp.asarray(s_arr, dtype=dtype)
    if int(s_arr.size) < 2:
        return jnp.sqrt(jnp.maximum(s_arr, jnp.asarray(0.0, dtype=dtype)))
    sh = 0.5 * (s_arr[1:] + s_arr[:-1])
    p = jnp.concatenate([sh[:1], sh], axis=0)
    return jnp.sqrt(jnp.maximum(p, jnp.asarray(0.0, dtype=dtype)))


def sm_sp_from_s_np(s_arr) -> tuple[np.ndarray, np.ndarray]:
    """Return VMEC radial finite-difference scale factors on the full mesh."""

    s_arr = np.asarray(s_arr, dtype=float)
    ns = int(s_arr.shape[0])
    if ns < 2:
        z = np.zeros((ns + 1,), dtype=float)
        return z, z
    hs = s_arr[1] - s_arr[0]
    i = np.arange(ns + 1, dtype=float)
    psqrts = np.where(i >= 1, np.sqrt(np.maximum(hs * (i - 1.0), 0.0)), 0.0)
    psqrts[-1] = 1.0
    pshalf = np.where(i >= 1, np.sqrt(np.maximum(hs * np.abs(i - 1.5), 0.0)), 0.0)
    sm = np.zeros((ns + 1,), dtype=float)
    sp = np.zeros((ns + 1,), dtype=float)
    idx = np.arange(2, ns + 1)
    sm[idx] = np.where(psqrts[idx] != 0, pshalf[idx] / psqrts[idx], 0.0)
    sm[1] = 0.0
    idx2 = np.arange(2, ns)
    sp[idx2] = np.where(psqrts[idx2] != 0, pshalf[idx2 + 1] / psqrts[idx2], 0.0)
    sp[ns] = np.where(psqrts[ns] != 0, 1.0 / psqrts[ns], 0.0)
    sp[0] = 0.0
    sp[1] = sm[2] if ns >= 2 else 0.0
    return sm, sp


def vmec_scale_m1_factors_from_mats(mats: dict[str, Any]) -> tuple[Any, Any]:
    """Return VMEC ``scale_m1_par`` R/Z factors from cached preconditioner data."""

    fac_r = mats.get("m1_fac_r")
    fac_z = mats.get("m1_fac_z")
    if fac_r is not None and fac_z is not None:
        return jnp.asarray(fac_r), jnp.asarray(fac_z)

    ard = mats.get("ard_parity")
    brd = mats.get("brd_parity")
    azd = mats.get("azd_parity")
    bzd = mats.get("bzd_parity")
    if ard is not None and brd is not None and azd is not None and bzd is not None:
        ard_arr = jnp.asarray(ard)
        brd_arr = jnp.asarray(brd)
        azd_arr = jnp.asarray(azd)
        bzd_arr = jnp.asarray(bzd)
        if (
            ard_arr.ndim == 2
            and brd_arr.shape == ard_arr.shape
            and azd_arr.shape == ard_arr.shape
            and bzd_arr.shape == ard_arr.shape
            and ard_arr.shape[1] > 1
        ):
            sr = ard_arr[:, 1] + brd_arr[:, 1]
            sz = azd_arr[:, 1] + bzd_arr[:, 1]
            denom = sr + sz
            fac_r = jnp.where(denom != 0.0, sr / denom, 1.0)
            fac_z = jnp.where(denom != 0.0, sz / denom, 1.0)
            return fac_r, fac_z

    dr = jnp.asarray(mats["dr"])
    dz = jnp.asarray(mats["dz"])
    sr = -dr[:, 1, 0]
    sz = -dz[:, 1, 0]
    denom = sr + sz
    fac_r = jnp.where(denom != 0.0, sr / denom, 1.0)
    fac_z = jnp.where(denom != 0.0, sz / denom, 1.0)
    return fac_r, fac_z


def vmec_scale_m1_factors_from_mats_np(mats: dict) -> tuple[np.ndarray, np.ndarray]:
    """NumPy version of :func:`vmec_scale_m1_factors_from_mats`."""

    fac_r = mats.get("m1_fac_r")
    fac_z = mats.get("m1_fac_z")
    if fac_r is not None and fac_z is not None:
        return np.asarray(fac_r), np.asarray(fac_z)

    ard = mats.get("ard_parity")
    brd = mats.get("brd_parity")
    azd = mats.get("azd_parity")
    bzd = mats.get("bzd_parity")
    if ard is not None and brd is not None and azd is not None and bzd is not None:
        ard_arr = np.asarray(ard)
        brd_arr = np.asarray(brd)
        azd_arr = np.asarray(azd)
        bzd_arr = np.asarray(bzd)
        if (
            ard_arr.ndim == 2
            and brd_arr.shape == ard_arr.shape
            and azd_arr.shape == ard_arr.shape
            and bzd_arr.shape == ard_arr.shape
            and ard_arr.shape[1] > 1
        ):
            sr = ard_arr[:, 1] + brd_arr[:, 1]
            sz = azd_arr[:, 1] + bzd_arr[:, 1]
            denom = sr + sz
            fac_r = np.where(denom != 0.0, sr / np.where(denom != 0.0, denom, 1.0), 1.0)
            fac_z = np.where(denom != 0.0, sz / np.where(denom != 0.0, denom, 1.0), 1.0)
            return fac_r, fac_z
    dr = np.asarray(mats["dr"])
    dz = np.asarray(mats["dz"])
    sr = -dr[:, 1, 0]
    sz = -dz[:, 1, 0]
    denom = sr + sz
    fac_r = np.where(denom != 0.0, sr / np.where(denom != 0.0, denom, 1.0), 1.0)
    fac_z = np.where(denom != 0.0, sz / np.where(denom != 0.0, denom, 1.0), 1.0)
    return fac_r, fac_z


def scale_m1_precond_rhs_from_mats(
    frzl_in,
    mats: dict[str, Any],
    *,
    lconm1: bool,
    mpol: int,
    host_update_assembly: bool,
):
    """Apply VMEC ``scale_m1_par`` factors before the radial preconditioner solve."""

    if (not bool(lconm1)) or (int(mpol) <= 1):
        return frzl_in

    if bool(host_update_assembly):
        fac_r_arr, fac_z_arr = vmec_scale_m1_factors_from_mats_np(mats)
        if fac_r_arr.size == 0:
            return frzl_in
        ns_full = int(np.asarray(frzl_in.frcc).shape[0])
        nsolve = min(ns_full, int(fac_r_arr.shape[0]))
        if nsolve == ns_full:
            fac_r_full = fac_r_arr[:nsolve]
            fac_z_full = fac_z_arr[:nsolve]
        else:
            ones = np.ones((ns_full - nsolve,), dtype=fac_r_arr.dtype)
            fac_r_full = np.concatenate([fac_r_arr[:nsolve], ones])
            fac_z_full = np.concatenate([fac_z_arr[:nsolve], ones])
        frss = scale_mode_slice_np(frzl_in.frss, mode_idx=1, scale=fac_r_full)
        fzcs = scale_mode_slice_np(frzl_in.fzcs, mode_idx=1, scale=fac_z_full)
        frsc = scale_mode_slice_np(getattr(frzl_in, "frsc", None), mode_idx=1, scale=fac_r_full)
        fzcc = scale_mode_slice_np(getattr(frzl_in, "fzcc", None), mode_idx=1, scale=fac_z_full)
    else:
        fac_r_jax, fac_z_jax = vmec_scale_m1_factors_from_mats(mats)
        if fac_r_jax.size == 0:
            return frzl_in
        fac_r = jnp.asarray(fac_r_jax, dtype=jnp.asarray(frzl_in.frcc).dtype)
        fac_z = jnp.asarray(fac_z_jax, dtype=jnp.asarray(frzl_in.fzsc).dtype)
        ns_full = int(jnp.asarray(frzl_in.frcc).shape[0])
        nsolve = min(ns_full, int(fac_r.shape[0]))
        ones_r = jnp.ones((max(ns_full - nsolve, 0),), dtype=jnp.asarray(frzl_in.frcc).dtype)
        ones_z = jnp.ones((max(ns_full - nsolve, 0),), dtype=jnp.asarray(frzl_in.fzsc).dtype)
        fac_r_full = fac_r[:nsolve] if nsolve == ns_full else jnp.concatenate([fac_r[:nsolve], ones_r], axis=0)
        fac_z_full = fac_z[:nsolve] if nsolve == ns_full else jnp.concatenate([fac_z[:nsolve], ones_z], axis=0)
        frsc_in = getattr(frzl_in, "frsc", None)
        fzcc_in = getattr(frzl_in, "fzcc", None)
        frss, fzcs, frsc, fzcc = _scale_m1_preconditioner_channels_jit(
            frzl_in.frss,
            frzl_in.fzcs,
            frsc_in,
            fzcc_in,
            fac_r_full,
            fac_z_full,
            has_frss=frzl_in.frss is not None,
            has_fzcs=frzl_in.fzcs is not None,
            has_frsc=frsc_in is not None,
            has_fzcc=fzcc_in is not None,
        )

    return TomnspsRZL(
        frcc=frzl_in.frcc,
        frss=frss,
        fzsc=frzl_in.fzsc,
        fzcs=fzcs,
        flsc=frzl_in.flsc,
        flcs=frzl_in.flcs,
        frsc=frsc,
        frcs=getattr(frzl_in, "frcs", None),
        fzcc=fzcc,
        fzss=getattr(frzl_in, "fzss", None),
        flcc=getattr(frzl_in, "flcc", None),
        flss=getattr(frzl_in, "flss", None),
    )


def can_reassemble_precond_mats(mats: Any) -> bool:
    """Return whether cached preconditioner matrices contain all reassembly channels."""

    if not isinstance(mats, dict):
        return False
    required = (
        "arm_parity",
        "ard_parity",
        "brm_parity",
        "brd_parity",
        "azm_parity",
        "azd_parity",
        "bzm_parity",
        "bzd_parity",
        "cxd_full",
        "delta_s",
    )
    return all(key in mats for key in required)
