"""Lightweight diagnostic helpers.

These utilities are intentionally dependency-free (NumPy-only) and meant to
print *useful* debugging information that you can copy/paste into chat.

We keep this module small and stable so it can be used from examples and tests
without pulling in plotting libraries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Summary:
    name: str
    shape: Tuple[int, ...]
    dtype: str
    min: float
    max: float
    mean: float
    std: float
    n_nan: int
    n_inf: int
    n_zero: int
    n_neg: int
    q: Tuple[float, float, float, float, float]


def _as_array(x: Any) -> np.ndarray:
    """Convert x to a NumPy array (safe for JAX arrays too)."""
    return np.asarray(x)


def summarize_array(name: str, x: Any, *, q: Sequence[float] = (0.0, 0.01, 0.5, 0.99, 1.0)) -> Summary:
    """Return basic stats + quantiles for an array-like."""
    a = _as_array(x)
    af = a.reshape(-1)
    # Handle empty arrays defensively
    if af.size == 0:
        return Summary(
            name=name,
            shape=tuple(a.shape),
            dtype=str(a.dtype),
            min=float("nan"),
            max=float("nan"),
            mean=float("nan"),
            std=float("nan"),
            n_nan=0,
            n_inf=0,
            n_zero=0,
            n_neg=0,
            q=(float("nan"),) * 5,
        )

    n_nan = int(np.sum(np.isnan(af))) if np.issubdtype(af.dtype, np.floating) else 0
    n_inf = int(np.sum(np.isinf(af))) if np.issubdtype(af.dtype, np.floating) else 0
    finite = af
    if np.issubdtype(af.dtype, np.floating):
        finite = af[np.isfinite(af)]
        if finite.size == 0:
            finite = af

    qvals = tuple(float(np.quantile(finite, qq)) for qq in q)

    return Summary(
        name=name,
        shape=tuple(a.shape),
        dtype=str(a.dtype),
        min=float(np.min(finite)),
        max=float(np.max(finite)),
        mean=float(np.mean(finite)),
        std=float(np.std(finite)),
        n_nan=n_nan,
        n_inf=n_inf,
        n_zero=int(np.sum(finite == 0)),
        n_neg=int(np.sum(finite < 0)),
        q=qvals,
    )


def print_summary(s: Summary, *, indent: str = "") -> None:
    """Pretty-print a Summary."""
    q0, q1, q50, q99, q100 = s.q
    print(
        f"{indent}{s.name}: shape={s.shape} dtype={s.dtype} "
        f"min={s.min:.6g} max={s.max:.6g} mean={s.mean:.6g} std={s.std:.6g}"
    )
    print(
        f"{indent}  q[0%]={q0:.6g} q[1%]={q1:.6g} q[50%]={q50:.6g} q[99%]={q99:.6g} q[100%]={q100:.6g}"
    )
    if s.n_nan or s.n_inf or s.n_zero or s.n_neg:
        print(
            f"{indent}  counts: nan={s.n_nan} inf={s.n_inf} zero={s.n_zero} neg={s.n_neg}"
        )


def summarize_many(names_and_arrays: Iterable[Tuple[str, Any]], *, indent: str = "") -> None:
    """Summarize many arrays."""
    for name, arr in names_and_arrays:
        print_summary(summarize_array(name, arr), indent=indent)


def print_jacobian_stats(sqrtg: Any, *, indent: str = "") -> None:
    """Print useful statistics for the Jacobian sqrt(g)."""
    a = _as_array(sqrtg)
    print_summary(summarize_array("sqrtg", a), indent=indent)
    print_summary(summarize_array("|sqrtg|", np.abs(a)), indent=indent)


def slice_excluding_axis(a: Any, axis_dim: int = 0) -> np.ndarray:
    """Return a[1:] along the chosen axis (used to avoid s=0 degeneracy)."""
    x = _as_array(a)
    if x.ndim == 0 or x.shape[axis_dim] <= 1:
        return x
    slc = [slice(None)] * x.ndim
    slc[axis_dim] = slice(1, None)
    return x[tuple(slc)]


def _vmec_basis_norm(*, mpol: int, ntor: int) -> np.ndarray:
    """Return 1/(mscale*nscale) factors for VMEC's internal Fourier basis."""
    mpol = int(mpol)
    ntor = int(ntor)
    # Match VMEC `fixaray` scaling: mscale(1:) = nscale(1:) = sqrt(2).
    mscale = np.ones((mpol,), dtype=float)
    nscale = np.ones((ntor + 1,), dtype=float)
    if mpol > 1:
        mscale[1:] = np.sqrt(2.0)
    if ntor > 0:
        nscale[1:] = np.sqrt(2.0)
    return 1.0 / (mscale[:, None] * nscale[None, :])


def _signed_to_mn_cos(coeffs: Any, *, modes, mpol: int, ntor: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert signed (m,n) cos coefficients to (rcc, rss) in VMEC (m,n>=0) storage."""
    coeffs = _as_array(coeffs)
    ns, ncoeff = coeffs.shape
    nrange = int(ntor) + 1
    idx_pos = -np.ones((mpol, nrange), dtype=int)
    idx_neg = -np.ones((mpol, nrange), dtype=int)
    m_arr = np.asarray(modes.m, dtype=int)
    n_arr = np.asarray(modes.n, dtype=int)
    for k in range(ncoeff):
        m_k = int(m_arr[k])
        n_k = int(n_arr[k])
        if n_k >= 0:
            idx_pos[m_k, n_k] = k
        else:
            idx_neg[m_k, -n_k] = k
    rcc = np.zeros((ns, mpol, nrange), dtype=coeffs.dtype)
    rss = np.zeros_like(rcc)
    for m_i in range(mpol):
        for n_i in range(nrange):
            kp = idx_pos[m_i, n_i]
            if kp < 0:
                continue
            pos = coeffs[:, kp]
            kn = idx_neg[m_i, n_i]
            neg = coeffs[:, kn] if kn >= 0 else 0.0
            rcc[:, m_i, n_i] = pos + neg
            if n_i == 0 or m_i == 0:
                rss[:, m_i, n_i] = 0.0
            else:
                rss[:, m_i, n_i] = pos - neg
    return rcc, rss


def _signed_to_mn_sin(coeffs: Any, *, modes, mpol: int, ntor: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert signed (m,n) sin coefficients to (zsc, zcs) in VMEC (m,n>=0) storage."""
    coeffs = _as_array(coeffs)
    ns, ncoeff = coeffs.shape
    nrange = int(ntor) + 1
    idx_pos = -np.ones((mpol, nrange), dtype=int)
    idx_neg = -np.ones((mpol, nrange), dtype=int)
    m_arr = np.asarray(modes.m, dtype=int)
    n_arr = np.asarray(modes.n, dtype=int)
    for k in range(ncoeff):
        m_k = int(m_arr[k])
        n_k = int(n_arr[k])
        if n_k >= 0:
            idx_pos[m_k, n_k] = k
        else:
            idx_neg[m_k, -n_k] = k
    zsc = np.zeros((ns, mpol, nrange), dtype=coeffs.dtype)
    zcs = np.zeros_like(zsc)
    for m_i in range(mpol):
        for n_i in range(nrange):
            kp = idx_pos[m_i, n_i]
            if kp < 0:
                continue
            pos = coeffs[:, kp]
            kn = idx_neg[m_i, n_i]
            neg = coeffs[:, kn] if kn >= 0 else 0.0
            zsc_val = pos + neg
            if n_i == 0:
                zcs[:, m_i, n_i] = 0.0
            else:
                zcs[:, m_i, n_i] = neg - pos
                if m_i == 0:
                    zsc_val = 0.0
            zsc[:, m_i, n_i] = zsc_val
    return zsc, zcs


def vmec_internal_mn_from_state(state: Any, static: Any, *, apply_basis_norm: bool = True) -> dict[str, np.ndarray]:
    """Return VMEC (m,n>=0) coefficient blocks from a signed-coefficient state.

    The returned arrays are in VMEC's internal basis (mscale/nscale removed)
    when ``apply_basis_norm`` is True.
    """
    cfg = static.cfg
    if bool(getattr(cfg, "lasym", False)):
        raise NotImplementedError("lasym coefficient decomposition not implemented yet")
    mpol = int(cfg.mpol)
    ntor = int(cfg.ntor)
    basis_norm = _vmec_basis_norm(mpol=mpol, ntor=ntor)

    rcc, rss = _signed_to_mn_cos(state.Rcos, modes=static.modes, mpol=mpol, ntor=ntor)
    zsc, zcs = _signed_to_mn_sin(state.Zsin, modes=static.modes, mpol=mpol, ntor=ntor)
    lsc, lcs = _signed_to_mn_sin(state.Lsin, modes=static.modes, mpol=mpol, ntor=ntor)

    # VMEC stores m=1 (rss,zcs) in an internal constrained basis when lconm1:
    #   rss_int = 0.5*(rss_phys + zcs_phys)
    #   zcs_int = 0.5*(rss_phys - zcs_phys)
    if bool(getattr(cfg, "lthreed", True)) and bool(getattr(cfg, "lconm1", True)) and mpol > 1:
        rss_m1 = rss[:, 1, :].copy()
        zcs_m1 = zcs[:, 1, :].copy()
        rss[:, 1, :] = 0.5 * (rss_m1 + zcs_m1)
        zcs[:, 1, :] = 0.5 * (rss_m1 - zcs_m1)

    if apply_basis_norm:
        rcc = rcc * basis_norm[None, :, :]
        rss = rss * basis_norm[None, :, :]
        zsc = zsc * basis_norm[None, :, :]
        zcs = zcs * basis_norm[None, :, :]
        lsc = lsc * basis_norm[None, :, :]
        lcs = lcs * basis_norm[None, :, :]

    return {
        "rcc": np.asarray(rcc),
        "rss": np.asarray(rss),
        "zsc": np.asarray(zsc),
        "zcs": np.asarray(zcs),
        "lsc": np.asarray(lsc),
        "lcs": np.asarray(lcs),
    }


def vmec_xc_from_mn_blocks(
    *,
    rcc: Any,
    rss: Any,
    zsc: Any,
    zcs: Any,
    lsc: Any,
    lcs: Any,
    cfg: Any,
) -> np.ndarray:
    """Pack VMEC (m,n>=0) coefficient blocks into the 1D xc vector."""
    if bool(getattr(cfg, "lasym", False)):
        raise NotImplementedError("lasym xc packing not implemented yet")
    mpol = int(cfg.mpol)
    ntor = int(cfg.ntor)
    lthreed = bool(getattr(cfg, "lthreed", True))
    ntmax = 2 if lthreed else 1
    rcc = _as_array(rcc)
    ns = int(rcc.shape[0])
    nrange = int(ntor) + 1
    mnsize = mpol * nrange
    mns = ns * mnsize

    def _flat(a: Any) -> np.ndarray:
        a = _as_array(a)
        if a.size == 0:
            return np.zeros((mns,), dtype=float)
        # VMEC serial order (after Parallel2Serial4X) packs with radial index
        # (js) fastest: idx = js + ns*mn (Fortran 1-based).
        return a.reshape((ns, mnsize)).T.reshape(-1)

    xc = np.zeros((3 * ntmax * mns,), dtype=rcc.dtype)
    if ntmax == 1:
        blocks = (rcc, zsc, lsc)
    elif ntmax == 2:
        # VMEC serial layout uses variable-major blocks:
        # [R(ntype=1..ntmax), Z(ntype=1..ntmax), L(ntype=1..ntmax)].
        # For symmetric 3D (ntmax=2): [rcc, rss, zsc, zcs, lsc, lcs].
        blocks = (rcc, rss, zsc, zcs, lsc, lcs)
    else:
        raise NotImplementedError("xc packing not implemented for ntmax > 2")
    for i, blk in enumerate(blocks):
        xc[i * mns : (i + 1) * mns] = _flat(blk)
    return xc
