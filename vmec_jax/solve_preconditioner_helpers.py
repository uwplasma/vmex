"""Preconditioner helpers for fixed-boundary VMEC solve paths."""

from __future__ import annotations

import os

import numpy as np

from ._compat import jax, jnp
from .state import VMECState


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
            cp_prev, dp_prev = carry
            denom = b - a * cp_prev
            cp = c / denom
            dp = (di - a * dp_prev) / denom
            return (cp, dp), (cp, dp)

        (_cp_last, dp_last), (cp_rest, dp_rest) = jax.lax.scan(fwd, (cp0, dp0), d[1:])
        cp = jnp.concatenate([jnp.asarray([cp0]), cp_rest], axis=0)
        dp = jnp.concatenate([dp0[None, :], dp_rest], axis=0)

        def bwd(x_next, items):
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
