"""Preconditioner helpers for fixed-boundary VMEC solve paths."""

from __future__ import annotations

from ._compat import jax, jnp
from .state import VMECState


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
