"""VMEC Fourier transform conventions (`fixaray` + `tomnsps`) for parity work.

VMEC does not use a plain unweighted DFT for its force/residual transforms.
Instead it uses:

- symmetry-aware theta grids (`ntheta1/2/3`) and endpoint weights,
- mode normalization scalings (`mscale`, `nscale`),
- precomputed trig tables (`cosmu/sinmu`, `cosnv/sinnv`) and their derivative
  companions (`cosmum/sinmum`, `cosnvn/sinnvn`).

This module implements the *core* pieces needed for parity diagnostics:

- `vmec_trig_tables(...)`  : build VMEC-style trig and weight tables
- `tomnsps_rzl(...)`       : real-space -> Fourier-space force transform

Scope
-----
This is intended for diagnostics/regressions against VMEC2000 `wout` outputs.
It currently implements the `lasym=False` and `lasym=True` tables, but the
transform itself is primarily exercised for the parity kernel work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import os
import numpy as np

from ._compat import jnp, tree_util, has_jax
from .grids import AngleGrid
from ._compat import has_jax


@dataclass(frozen=True)
class VmecTrigTables:
    """Trig/weight tables consistent with VMEC `fixaray.f`."""

    # VMEC theta grid sizes.
    ntheta1: int
    ntheta2: int
    ntheta3: int

    # Integration weight normalization used by VMEC in fixaray (dnorm).
    dnorm: float
    # Surface-integration normalization (dnorm3 in fixaray).
    dnorm3: float

    # Mode normalization scalings.
    mscale: Any  # (mmax+1,)
    nscale: Any  # (nmax+1,)
    r0scale: float

    # Theta tables. Shapes (ntheta3, mmax+1).
    cosmu: Any
    sinmu: Any
    cosmum: Any
    sinmum: Any
    cosmui: Any
    sinmui: Any
    cosmumi: Any
    sinmumi: Any
    cosmui3: Any
    cosmumi3: Any
    cosmui_nt2: Any | None = None
    sinmui_nt2: Any | None = None
    cosmumi_nt2: Any | None = None
    sinmumi_nt2: Any | None = None

    # Zeta tables. Shapes (nzeta, nmax+1).
    cosnv: Any
    sinnv: Any
    cosnvn: Any
    sinnvn: Any

    # Optional cached phase stacks for vmec_realspace synthesis.
    # These are populated by VMECStatic when enabled.
    phase_stack: Any | None = None
    phase_dtheta_stack: Any | None = None
    phase_dzeta_stack: Any | None = None
    phase_stack_m: Any | None = None
    phase_stack_n: Any | None = None


@dataclass(frozen=True)
class TomnspsMasks:
    """Precomputed parity/evolution masks for tomnsps/tomnspa."""

    ns: int
    mpol: int
    include_edge: bool
    mask_even: Any  # (mpol,)
    mask_rz: Any  # (ns, mpol, 1)
    mask_l: Any  # (ns, mpol, 1)
    xmpq1: Any | None = None  # (1, mpol, 1)


def vmec_theta_sizes(ntheta: int, *, lasym: bool) -> tuple[int, int, int]:
    """Reproduce VMEC `read_indata.f` theta sizes."""
    ntheta = int(ntheta)
    ntheta1 = 2 * (ntheta // 2)
    ntheta2 = 1 + ntheta1 // 2  # includes u=pi point
    ntheta3 = ntheta1 if bool(lasym) else ntheta2
    return int(ntheta1), int(ntheta2), int(ntheta3)


_TRIG_CACHE: dict[tuple[int, int, int, int, int, bool, str], VmecTrigTables] = {}


def vmec_trig_tables(
    *,
    ntheta: int,
    nzeta: int,
    nfp: int,
    mmax: int,
    nmax: int,
    lasym: bool,
    dtype=jnp.float64,
    cache: bool = True,
) -> VmecTrigTables:
    """Build VMEC-style trig and weight tables.

    This follows `fixaray.f` closely (with `mscale(0)=nscale(0)=1`).
    """
    ntheta1, ntheta2, ntheta3 = vmec_theta_sizes(ntheta, lasym=lasym)
    nzeta = int(nzeta)
    nfp = int(nfp)
    mmax = int(mmax)
    nmax = int(nmax)
    if ntheta1 <= 0 or ntheta2 <= 0 or ntheta3 <= 0:
        raise ValueError("Invalid theta sizes")
    if nzeta <= 0:
        raise ValueError("nzeta must be positive")
    if nfp <= 0:
        raise ValueError("nfp must be positive")
    if mmax < 0 or nmax < 0:
        raise ValueError("mmax/nmax must be nonnegative")
    if cache:
        try:
            dtype_key = str(np.dtype(dtype))
        except Exception:
            dtype_key = str(dtype)
        cache_key = (ntheta, nzeta, nfp, mmax, nmax, bool(lasym), dtype_key)
        cached = _TRIG_CACHE.get(cache_key)
        if cached is not None:
            return cached

    # dnorm normalization in fixaray (always on reduced interval [0, pi]).
    dnorm = 1.0 / (nzeta * (ntheta2 - 1))
    # dnorm3 normalization for surface averages.
    if lasym:
        dnorm3 = 1.0 / (nzeta * ntheta1)
    else:
        dnorm3 = dnorm

    # VMEC uses `osqrt2 = 1/sqrt(2)` and sets:
    #   mscale(1:) = mscale(0)/osqrt2  => sqrt(2) for m>=1 (with mscale(0)=1)
    # Likewise for `nscale`. This ensures the trig tables implement the
    # orthonormal-like scaling VMEC assumes in `fixaray.f`.
    osqrt2 = 1.0 / np.sqrt(2.0)
    mscale = np.ones((mmax + 1,), dtype=float)
    nscale = np.ones((nmax + 1,), dtype=float)
    if mmax >= 1:
        mscale[1:] = mscale[0] / osqrt2
    if nmax >= 1:
        nscale[1:] = nscale[0] / osqrt2
    r0scale = float(mscale[0] * nscale[0])

    # Theta tables use argi = 2π*(i-1)/ntheta1, for i=1..ntheta3.
    # Follow fixaray.f exactly, including the explicit symmetry enforcement
    # for i > ntheta2 and the special handling of the pi endpoint.
    m = np.arange(mmax + 1, dtype=float)  # (mmax+1,)
    cosmu = np.zeros((ntheta3, mmax + 1), dtype=float)
    sinmu = np.zeros_like(cosmu)
    for i in range(ntheta3):
        if (not lasym) and i == (ntheta2 - 1):
            # Special case theta = pi (i == ntheta2 in Fortran) on the reduced grid.
            signs = np.where((m.astype(int) % 2) == 0, 1.0, -1.0)
            cosmu[i, :] = signs * mscale
            sinmu[i, :] = 0.0
        elif (not lasym) and i >= ntheta2:
            # Force symmetry for indices over ntheta2 (lasym=False only).
            ir = 2 * ntheta2 - i - 2
            cosmu[i, :] = cosmu[ir, :]
            sinmu[i, :] = -sinmu[ir, :]
        else:
            arg = (2.0 * np.pi) * float(i) / float(ntheta1)
            cosmu[i, :] = np.cos(arg * m) * mscale
            sinmu[i, :] = np.sin(arg * m) * mscale

    cosmum = cosmu * m[None, :]
    sinmum = -sinmu * m[None, :]

    cosmui_base = dnorm * cosmu
    sinmui_base = dnorm * sinmu
    cosmui = cosmui_base.copy()
    sinmui = sinmui_base.copy()
    cosmui3 = (dnorm3 * cosmu).copy()

    # Endpoint half-weights in theta for i==1 or i==ntheta2.
    # Note: in VMEC, `ntheta2` is the u=pi index even when `lasym=True`.
    if ntheta2 >= 1 and ntheta2 <= ntheta3:
        cosmui[0, :] *= 0.5
        cosmui[ntheta2 - 1, :] *= 0.5
        if not lasym and ntheta2 == ntheta3:
            # When `ntheta3==ntheta2` (lasym=False), VMEC reuses the half-interval
            # integration weights for full-interval integrations too.
            cosmui3 = cosmui.copy()

    cosmumi = cosmui * m[None, :]
    sinmumi = -sinmui * m[None, :]
    cosmumi3 = cosmui3 * m[None, :]

    cosmui_nt2 = cosmui[:ntheta2, :]
    sinmui_nt2 = sinmui[:ntheta2, :]
    cosmumi_nt2 = cosmumi[:ntheta2, :]
    sinmumi_nt2 = sinmumi[:ntheta2, :]

    # Zeta tables use argj = 2π*(j-1)/nzeta, for j=1..nzeta.
    j = np.arange(nzeta, dtype=float)
    zeta = (2.0 * np.pi) * j / float(nzeta)
    n = np.arange(nmax + 1, dtype=float)
    argn = zeta[:, None] * n[None, :]
    cosnv = np.cos(argn) * nscale[None, :]
    sinnv = np.sin(argn) * nscale[None, :]

    cosnvn = cosnv * (n[None, :] * float(nfp))
    sinnvn = -sinnv * (n[None, :] * float(nfp))

    # Convert to backend arrays.
    tables = VmecTrigTables(
        ntheta1=ntheta1,
        ntheta2=ntheta2,
        ntheta3=ntheta3,
        dnorm=float(dnorm),
        dnorm3=float(dnorm3),
        mscale=jnp.asarray(mscale, dtype=dtype),
        nscale=jnp.asarray(nscale, dtype=dtype),
        r0scale=float(r0scale),
        cosmu=jnp.asarray(cosmu, dtype=dtype),
        sinmu=jnp.asarray(sinmu, dtype=dtype),
        cosmum=jnp.asarray(cosmum, dtype=dtype),
        sinmum=jnp.asarray(sinmum, dtype=dtype),
        cosmui=jnp.asarray(cosmui, dtype=dtype),
        sinmui=jnp.asarray(sinmui, dtype=dtype),
        cosmumi=jnp.asarray(cosmumi, dtype=dtype),
        sinmumi=jnp.asarray(sinmumi, dtype=dtype),
        cosmui3=jnp.asarray(cosmui3, dtype=dtype),
        cosmumi3=jnp.asarray(cosmumi3, dtype=dtype),
        cosmui_nt2=jnp.asarray(cosmui_nt2, dtype=dtype),
        sinmui_nt2=jnp.asarray(sinmui_nt2, dtype=dtype),
        cosmumi_nt2=jnp.asarray(cosmumi_nt2, dtype=dtype),
        sinmumi_nt2=jnp.asarray(sinmumi_nt2, dtype=dtype),
        cosnv=jnp.asarray(cosnv, dtype=dtype),
        sinnv=jnp.asarray(sinnv, dtype=dtype),
        cosnvn=jnp.asarray(cosnvn, dtype=dtype),
        sinnvn=jnp.asarray(sinnvn, dtype=dtype),
    )
    if cache:
        _TRIG_CACHE[cache_key] = tables
    return tables


_GRID_CACHE: dict[tuple[int, int, int, bool], AngleGrid] = {}

_TOMNSPS_MASK_CACHE: dict[tuple[int, int, bool, str], TomnspsMasks] = {}


def vmec_angle_grid(*, ntheta: int, nzeta: int, nfp: int, lasym: bool, cache: bool = True) -> AngleGrid:
    """Build the VMEC internal (theta,zeta) grid implied by `read_indata.f`.

    - If `lasym=False`, VMEC uses `ntheta3=ntheta2` points covering `[0,π]`
      including the endpoint `π`.
    - If `lasym=True`, VMEC uses `ntheta3=ntheta1` points covering `[0,2π)`
      endpoint-free.

    In both cases, `zeta` spans one field period `[0,2π)` endpoint-free.
    """
    ntheta1, ntheta2, ntheta3 = vmec_theta_sizes(ntheta, lasym=lasym)
    nzeta = int(nzeta)
    if nzeta <= 0:
        nzeta = 1
    cache_key = (int(ntheta), int(nzeta), int(nfp), bool(lasym))
    if cache:
        cached = _GRID_CACHE.get(cache_key)
        if cached is not None:
            return cached
    if lasym:
        i = np.arange(ntheta3, dtype=float)
        theta = (2.0 * np.pi) * i / float(ntheta1)
    else:
        i = np.arange(ntheta3, dtype=float)  # 0..ntheta2-1
        theta = (2.0 * np.pi) * i / float(ntheta1)  # includes pi at i=ntheta2-1
    j = np.arange(nzeta, dtype=float)
    zeta = (2.0 * np.pi) * j / float(nzeta)
    grid = AngleGrid(theta=theta.astype(float), zeta=zeta.astype(float), nfp=int(nfp))
    if cache:
        _GRID_CACHE[cache_key] = grid
    return grid


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class TomnspsRZL:
    """Fourier-space force arrays in VMEC's internal (n,m) packing (subset)."""

    # R components: frcc is always present; frss only if lthreed.
    frcc: Any  # (ns, mpol, ntor+1)
    frss: Any | None  # (ns, mpol, ntor+1)

    # Z components: fzsc always present; fzcs only if lthreed.
    fzsc: Any  # (ns, mpol, ntor+1)
    fzcs: Any | None  # (ns, mpol, ntor+1)

    # Lambda components (for later): flsc always present; flcs only if lthreed.
    flsc: Any  # (ns, mpol, ntor+1)
    flcs: Any | None  # (ns, mpol, ntor+1)

    # Asymmetric components (only meaningful when lasym=True). These correspond
    # to `tomnspa_par` outputs:
    # - R: frsc (sin(mu)cos(nv)) and frcs (cos(mu)sin(nv), if lthreed)
    # - Z: fzcc (cos(mu)cos(nv)) and fzss (sin(mu)sin(nv), if lthreed)
    # - L: flcc (cos(mu)cos(nv)) and flss (sin(mu)sin(nv), if lthreed)
    frsc: Any | None = None  # (ns, mpol, ntor+1)
    frcs: Any | None = None  # (ns, mpol, ntor+1)
    fzcc: Any | None = None  # (ns, mpol, ntor+1)
    fzss: Any | None = None  # (ns, mpol, ntor+1)
    flcc: Any | None = None  # (ns, mpol, ntor+1)
    flss: Any | None = None  # (ns, mpol, ntor+1)

    def tree_flatten(self):
        children = (
            self.frcc,
            self.frss,
            self.fzsc,
            self.fzcs,
            self.flsc,
            self.flcs,
            self.frsc,
            self.frcs,
            self.fzcc,
            self.fzss,
            self.flcc,
            self.flss,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


_MPARITY_CACHE: dict[tuple[int, str], jnp.ndarray] = {}
_JNP_EINSUM = jnp.einsum
_DETERMINISTIC_REDUCE = bool(int(os.environ.get("VMEC_JAX_DETERMINISTIC_REDUCE", "0")))


def _einsum(expr: str, *operands):
    if has_jax():
        try:
            from jax import lax

            return _JNP_EINSUM(expr, *operands, precision=lax.Precision.HIGHEST)
        except TypeError:
            return _JNP_EINSUM(expr, *operands)
    return _JNP_EINSUM(expr, *operands)


def _theta_contract(arr, mat):
    """Deterministic theta reduction matching VMEC loop order when enabled."""
    if (not _DETERMINISTIC_REDUCE) or (not has_jax()):
        return _einsum("apsik,im->apsmk", arr, mat)
    from jax import lax

    arr = jnp.asarray(arr)
    mat = jnp.asarray(mat)
    a, p, s, i_size, k = arr.shape
    m = mat.shape[1]
    acc = jnp.zeros((a, p, s, m, k), dtype=arr.dtype)

    def body(i, acc_i):
        arr_i = arr[:, :, :, i, :]  # (a, p, s, k)
        mat_i = mat[i, :]  # (m,)
        return acc_i + arr_i[..., None, :] * mat_i[None, None, None, :, None]

    return lax.fori_loop(0, i_size, body, acc)


def _zeta_contract(arr, mat):
    """Deterministic zeta reduction matching VMEC loop order when enabled."""
    if (not _DETERMINISTIC_REDUCE) or (not has_jax()):
        return _einsum("psmk,kn->psmn", arr, mat)
    from jax import lax

    arr = jnp.asarray(arr)
    mat = jnp.asarray(mat)
    p, s, m, k_size = arr.shape
    n = mat.shape[1]
    acc = jnp.zeros((p, s, m, n), dtype=arr.dtype)

    def body(k, acc_k):
        arr_k = arr[:, :, :, k]  # (p, s, m)
        mat_k = mat[k, :]  # (n,)
        return acc_k + arr_k[..., None] * mat_k[None, None, None, :]

    return lax.fori_loop(0, k_size, body, acc)


def _cache_allowed() -> bool:
    if not has_jax():
        return True
    try:
        from jax import core

        return bool(core.trace_ctx.is_top_level())
    except Exception:
        return False


def _mparity_mask(mpol: int, *, dtype) -> jnp.ndarray:
    key = (int(mpol), str(np.dtype(dtype)))
    if _cache_allowed():
        cached = _MPARITY_CACHE.get(key)
        if cached is not None:
            return cached
    m = jnp.arange(int(mpol))
    mask_even = jnp.asarray((m % 2) == 0, dtype=dtype)
    if _cache_allowed():
        _MPARITY_CACHE[key] = mask_even
    return mask_even


def _select_mparity(a_even, a_odd, mask_even: jnp.ndarray):
    mask = mask_even[None, :, None]
    return mask * a_even + (1.0 - mask) * a_odd


def tomnsps_masks(
    *,
    ns: int,
    mpol: int,
    include_edge: bool,
    dtype=jnp.float64,
    cache: bool = True,
) -> TomnspsMasks:
    """Precompute parity/evolution masks for tomnsps/tomnspa."""
    ns = int(ns)
    mpol = int(mpol)
    include_edge = bool(include_edge)
    if ns < 1:
        raise ValueError("ns must be positive")
    if mpol < 1:
        raise ValueError("mpol must be positive")
    try:
        dtype_key = str(np.dtype(dtype))
    except Exception:
        dtype_key = str(dtype)
    cache_key = (ns, mpol, include_edge, dtype_key)
    if cache and _cache_allowed():
        cached = _TOMNSPS_MASK_CACHE.get(cache_key)
        if cached is not None:
            return cached

    mask_even = _mparity_mask(mpol, dtype=dtype)
    js_fortran = jnp.arange(ns, dtype=jnp.int32) + 1  # 1..ns
    m_fortran = jnp.arange(mpol, dtype=jnp.int32)  # 0..mpol-1
    jmin2 = jnp.where(m_fortran == 0, 1, 2)[None, :]
    jlam = jnp.full((1, mpol), 2, dtype=jnp.int32)
    jsmax_rz = int(ns if include_edge else (ns - 1))
    mask_rz = (js_fortran[:, None] >= jmin2) & (js_fortran[:, None] <= jsmax_rz)
    mask_l = js_fortran[:, None] >= jlam
    mask_rz = mask_rz.astype(jnp.asarray(mask_even).dtype)[:, :, None]
    mask_l = mask_l.astype(jnp.asarray(mask_even).dtype)[:, :, None]
    m_fortran_f = jnp.asarray(m_fortran, dtype=jnp.asarray(mask_even).dtype)
    xmpq1 = (m_fortran_f * (m_fortran_f - 1.0))[None, :, None]
    masks = TomnspsMasks(
        ns=ns,
        mpol=mpol,
        include_edge=include_edge,
        mask_even=mask_even,
        mask_rz=mask_rz,
        mask_l=mask_l,
        xmpq1=xmpq1,
    )
    if cache and _cache_allowed():
        _TOMNSPS_MASK_CACHE[cache_key] = masks
    return masks


def tomnsps_rzl(
    *,
    armn_even,
    armn_odd,
    brmn_even,
    brmn_odd,
    crmn_even,
    crmn_odd,
    azmn_even,
    azmn_odd,
    bzmn_even,
    bzmn_odd,
    czmn_even,
    czmn_odd,
    blmn_even=None,
    blmn_odd=None,
    clmn_even=None,
    clmn_odd=None,
    arcon_even=None,
    arcon_odd=None,
    azcon_even=None,
    azcon_odd=None,
    mpol: int,
    ntor: int,
    nfp: int,
    lasym: bool,
    trig: VmecTrigTables,
    include_edge: bool = False,
    masks: TomnspsMasks | None = None,
) -> TomnspsRZL:
    """VMEC real-space -> Fourier-space force transform (core of `tomnsps`).

    Parameters
    ----------
    *_even, *_odd:
        Real-space kernels on the VMEC angular grid, with shapes:
            (ns, ntheta3, nzeta)
        representing `mparity=0` and `mparity=1` pieces.
    mpol, ntor:
        VMEC spectral truncation (`m=0..mpol-1`, `n=0..ntor`).
    lasym:
        Included for signature completeness; the transform uses `trig.ntheta2`
        and `trig.ntheta3` as built by `vmec_trig_tables`.

    Notes
    -----
    This mirrors the structure of `tomnsp_mod.f:tomnsps_par` but returns only
    the primary R/Z/L blocks needed for residual norms and parity work.

    Parameters
    ----------
    include_edge:
        If True, keep the boundary surface (js=ns) in the fixed-boundary masks.
        This is useful for diagnostics that emulate the `jedge=1` branch in
        `getfsq`, but should be left False for standard VMEC fixed-boundary norms.
    """
    # Shapes.
    armn_even = jnp.asarray(armn_even)
    armn_odd = jnp.asarray(armn_odd)
    ns, ntheta3, nzeta = armn_even.shape
    mpol = int(mpol)
    ntor = int(ntor)
    nfp = int(nfp)
    if mpol <= 0:
        raise ValueError("mpol must be positive")
    if ntor < 0:
        raise ValueError("ntor must be nonnegative")
    if trig.ntheta3 != int(ntheta3) or int(trig.cosnv.shape[0]) != int(nzeta):
        raise ValueError("Input grid does not match trig tables")

    # Restrict theta integration to VMEC's ntheta2 (u in [0,pi]).
    nt2 = int(trig.ntheta2)
    armn_even = armn_even[:, :nt2, :]
    armn_odd = armn_odd[:, :nt2, :]
    brmn_even = jnp.asarray(brmn_even)[:, :nt2, :]
    brmn_odd = jnp.asarray(brmn_odd)[:, :nt2, :]
    crmn_even = jnp.asarray(crmn_even)[:, :nt2, :]
    crmn_odd = jnp.asarray(crmn_odd)[:, :nt2, :]
    azmn_even = jnp.asarray(azmn_even)[:, :nt2, :]
    azmn_odd = jnp.asarray(azmn_odd)[:, :nt2, :]
    bzmn_even = jnp.asarray(bzmn_even)[:, :nt2, :]
    bzmn_odd = jnp.asarray(bzmn_odd)[:, :nt2, :]
    czmn_even = jnp.asarray(czmn_even)[:, :nt2, :]
    czmn_odd = jnp.asarray(czmn_odd)[:, :nt2, :]

    if arcon_even is None:
        arcon_even = jnp.zeros_like(armn_even)
    if arcon_odd is None:
        arcon_odd = jnp.zeros_like(armn_odd)
    if azcon_even is None:
        azcon_even = jnp.zeros_like(armn_even)
    if azcon_odd is None:
        azcon_odd = jnp.zeros_like(armn_odd)

    if blmn_even is None:
        blmn_even = jnp.zeros_like(armn_even)
    if blmn_odd is None:
        blmn_odd = jnp.zeros_like(armn_odd)
    if clmn_even is None:
        clmn_even = jnp.zeros_like(armn_even)
    if clmn_odd is None:
        clmn_odd = jnp.zeros_like(armn_odd)
    blmn_even = jnp.asarray(blmn_even)[:, :nt2, :]
    blmn_odd = jnp.asarray(blmn_odd)[:, :nt2, :]
    clmn_even = jnp.asarray(clmn_even)[:, :nt2, :]
    clmn_odd = jnp.asarray(clmn_odd)[:, :nt2, :]
    arcon_even = jnp.asarray(arcon_even)[:, :nt2, :]
    arcon_odd = jnp.asarray(arcon_odd)[:, :nt2, :]
    azcon_even = jnp.asarray(azcon_even)[:, :nt2, :]
    azcon_odd = jnp.asarray(azcon_odd)[:, :nt2, :]

    # Tables for m=0..mpol-1 and n=0..ntor.
    cosmui = getattr(trig, "cosmui_nt2", None)
    sinmui = getattr(trig, "sinmui_nt2", None)
    cosmumi = getattr(trig, "cosmumi_nt2", None)
    sinmumi = getattr(trig, "sinmumi_nt2", None)
    if cosmui is None or int(cosmui.shape[0]) != nt2:
        cosmui = trig.cosmui[:nt2, :]
    if sinmui is None or int(sinmui.shape[0]) != nt2:
        sinmui = trig.sinmui[:nt2, :]
    if cosmumi is None or int(cosmumi.shape[0]) != nt2:
        cosmumi = trig.cosmumi[:nt2, :]
    if sinmumi is None or int(sinmumi.shape[0]) != nt2:
        sinmumi = trig.sinmumi[:nt2, :]
    cosmui = cosmui[:, :mpol]
    sinmui = sinmui[:, :mpol]
    cosmumi = cosmumi[:, :mpol]
    sinmumi = sinmumi[:, :mpol]

    cosnv = trig.cosnv[:, : (ntor + 1)]  # (nzeta, ntor+1)
    sinnv = trig.sinnv[:, : (ntor + 1)]
    cosnvn = trig.cosnvn[:, : (ntor + 1)]
    sinnvn = trig.sinnvn[:, : (ntor + 1)]

    # VMEC constraint operator multiplier: xmpq(m,1)=m*(m-1).
    xmpq1 = None
    if masks is not None:
        try:
            if (int(masks.ns) == int(ns)) and (int(masks.mpol) == int(mpol)):
                xmpq1 = getattr(masks, "xmpq1", None)
        except Exception:
            xmpq1 = None
    if xmpq1 is None:
        m = np.arange(mpol, dtype=int)
        xmpq1 = (
            jnp.asarray(m, dtype=jnp.asarray(armn_even).dtype)
            * (jnp.asarray(m, dtype=jnp.asarray(armn_even).dtype) - 1.0)
        )[None, :, None]
    else:
        xmpq1 = jnp.asarray(xmpq1, dtype=jnp.asarray(armn_even).dtype)

    # Theta integration: compute work arrays for even-parity and odd-parity pieces.
    # Each is (ns, mpol, nzeta).
    # work1 indices follow tomnsp_mod.f numbering but we compute only needed combos.
    armn = jnp.stack([armn_even, armn_odd], axis=0)
    brmn = jnp.stack([brmn_even, brmn_odd], axis=0)
    crmn = jnp.stack([crmn_even, crmn_odd], axis=0)
    azmn = jnp.stack([azmn_even, azmn_odd], axis=0)
    bzmn = jnp.stack([bzmn_even, bzmn_odd], axis=0)
    czmn = jnp.stack([czmn_even, czmn_odd], axis=0)
    blmn = jnp.stack([blmn_even, blmn_odd], axis=0)
    clmn = jnp.stack([clmn_even, clmn_odd], axis=0)
    arcon = jnp.stack([arcon_even, arcon_odd], axis=0)
    azcon = jnp.stack([azcon_even, azcon_odd], axis=0)

    def _theta_einsum_stack(arr, mat):
        return _theta_contract(arr, mat)

    stack_cosmui = jnp.stack([armn, crmn, azmn, czmn, arcon, azcon, clmn], axis=0)
    stack_sinmumi = jnp.stack([brmn, bzmn, blmn], axis=0)

    cosmui_out = _theta_einsum_stack(stack_cosmui, cosmui)
    sinmui_out = _theta_einsum_stack(stack_cosmui, sinmui)
    sinmumi_out = _theta_einsum_stack(stack_sinmumi, sinmumi)
    cosmumi_out = _theta_einsum_stack(stack_sinmumi, cosmumi)

    armn_cos, crmn_cos, azmn_cos, czmn_cos, arcon_cos, azcon_cos, clmn_cos = cosmui_out
    armn_sin, crmn_sin, azmn_sin, czmn_sin, arcon_sin, azcon_sin, clmn_sin = sinmui_out
    brmn_sin, bzmn_sin, blmn_sin = sinmumi_out
    brmn_cos, bzmn_cos, blmn_cos = cosmumi_out

    xmpq1 = xmpq1[None, ...]

    # R:
    w1 = armn_cos + brmn_sin + xmpq1 * arcon_cos
    w2 = -crmn_cos
    w3 = armn_sin + brmn_cos + xmpq1 * arcon_sin
    w4 = -crmn_sin

    # Z:
    w7 = azmn_sin + bzmn_cos + xmpq1 * azcon_sin
    w8 = -czmn_sin
    w5 = azmn_cos + bzmn_sin + xmpq1 * azcon_cos
    w6 = -czmn_cos

    # Lambda:
    w11 = blmn_cos
    w12 = -clmn_sin
    w9 = blmn_sin
    w10 = -clmn_cos

    w1_e, w1_o = w1[0], w1[1]
    w2_e, w2_o = w2[0], w2[1]
    w3_e, w3_o = w3[0], w3[1]
    w4_e, w4_o = w4[0], w4[1]
    w5_e, w5_o = w5[0], w5[1]
    w6_e, w6_o = w6[0], w6[1]
    w7_e, w7_o = w7[0], w7[1]
    w8_e, w8_o = w8[0], w8[1]
    w9_e, w9_o = w9[0], w9[1]
    w10_e, w10_o = w10[0], w10[1]
    w11_e, w11_o = w11[0], w11[1]
    w12_e, w12_o = w12[0], w12[1]

    # Select parity per m (mparity = mod(m,2)).
    mask_even = None
    if masks is not None:
        if (int(masks.ns) == int(ns)) and (int(masks.mpol) == int(mpol)) and (bool(masks.include_edge) == bool(include_edge)):
            mask_even = jnp.asarray(masks.mask_even, dtype=jnp.asarray(armn_even).dtype)
    if mask_even is None:
        mask_even = _mparity_mask(mpol, dtype=jnp.asarray(armn_even).dtype)
    w1 = _select_mparity(w1_e, w1_o, mask_even)
    w2 = _select_mparity(w2_e, w2_o, mask_even)
    w3 = _select_mparity(w3_e, w3_o, mask_even)
    w4 = _select_mparity(w4_e, w4_o, mask_even)
    w5 = _select_mparity(w5_e, w5_o, mask_even)
    w6 = _select_mparity(w6_e, w6_o, mask_even)
    w7 = _select_mparity(w7_e, w7_o, mask_even)
    w8 = _select_mparity(w8_e, w8_o, mask_even)
    w9 = _select_mparity(w9_e, w9_o, mask_even)
    w10 = _select_mparity(w10_e, w10_o, mask_even)
    w11 = _select_mparity(w11_e, w11_o, mask_even)
    w12 = _select_mparity(w12_e, w12_o, mask_even)

    lthreed = bool(ntor > 0)

    # Zeta integration. Result arrays are (ns, mpol, ntor+1).
    w_cosnv = jnp.stack([w1, w7, w11], axis=0)
    out_cosnv = _zeta_contract(w_cosnv, cosnv)
    frcc, fzsc, flsc = out_cosnv[0], out_cosnv[1], out_cosnv[2]

    if lthreed:
        w_sinnvn = jnp.stack([w2, w8, w12], axis=0)
        out_sinnvn = _zeta_contract(w_sinnvn, sinnvn)
        frcc = frcc + out_sinnvn[0]
        fzsc = fzsc + out_sinnvn[1]
        flsc = flsc + out_sinnvn[2]

        w_sinnv = jnp.stack([w3, w5, w9], axis=0)
        w_cosnvn = jnp.stack([w4, w6, w10], axis=0)
        out_sinnv = _zeta_contract(w_sinnv, sinnv)
        out_cosnvn = _zeta_contract(w_cosnvn, cosnvn)

        frss = out_sinnv[0] + out_cosnvn[0]
        fzcs = out_sinnv[1] + out_cosnvn[1]
        flcs = out_sinnv[2] + out_cosnvn[2]
    else:
        frss = None
        fzcs = None
        flcs = None

    # Apply VMEC's radial evolution masks (jmin2/jlam + fixed-boundary edge).
    # For parity work we use the default vmec_params values:
    #   jmin2(m=0)=1, jmin2(m>=1)=2; jlam(m)=2.
    if masks is not None and (int(masks.ns) == int(ns)) and (int(masks.mpol) == int(mpol)) and (bool(masks.include_edge) == bool(include_edge)):
        mask_rz = jnp.asarray(masks.mask_rz, dtype=jnp.asarray(frcc).dtype)
        mask_l = jnp.asarray(masks.mask_l, dtype=jnp.asarray(flsc).dtype)
    else:
        js_fortran = jnp.arange(ns, dtype=jnp.int32) + 1  # 1..ns
        m_fortran = jnp.arange(mpol, dtype=jnp.int32)  # 0..mpol-1
        jmin2 = jnp.where(m_fortran == 0, 1, 2)[None, :]  # (1, mpol)
        jlam = jnp.full((1, mpol), 2, dtype=jmin2.dtype)

        # Fixed-boundary convention: R/Z not evolved on the boundary surface (js=ns).
        # `include_edge=True` reproduces the `jedge=1` branch in `getfsq` diagnostics.
        jsmax_rz = int(ns if include_edge else (ns - 1))
        mask_rz = (js_fortran[:, None] >= jmin2) & (js_fortran[:, None] <= jsmax_rz)
        mask_l = js_fortran[:, None] >= jlam

        mask_rz = mask_rz.astype(jnp.asarray(frcc).dtype)[:, :, None]
        mask_l = mask_l.astype(jnp.asarray(flsc).dtype)[:, :, None]
    frcc = frcc * mask_rz
    fzsc = fzsc * mask_rz
    if frss is not None:
        frss = frss * mask_rz
    if fzcs is not None:
        fzcs = fzcs * mask_rz
    flsc = flsc * mask_l
    if flcs is not None:
        flcs = flcs * mask_l

    return TomnspsRZL(frcc=frcc, frss=frss, fzsc=fzsc, fzcs=fzcs, flsc=flsc, flcs=flcs)


def tomnspa_rzl(
    *,
    armn_even,
    armn_odd,
    brmn_even,
    brmn_odd,
    crmn_even,
    crmn_odd,
    azmn_even,
    azmn_odd,
    bzmn_even,
    bzmn_odd,
    czmn_even,
    czmn_odd,
    blmn_even=None,
    blmn_odd=None,
    clmn_even=None,
    clmn_odd=None,
    arcon_even=None,
    arcon_odd=None,
    azcon_even=None,
    azcon_odd=None,
    mpol: int,
    ntor: int,
    nfp: int,
    lasym: bool,
    trig: VmecTrigTables,
    include_edge: bool = False,
    masks: TomnspsMasks | None = None,
) -> TomnspsRZL:
    """VMEC `tomnspa` asymmetric force transform (real-space -> Fourier-space).

    This mirrors `tomnsp_mod.f:tomnspa_par`. It is only used when `lasym=True`,
    but is safe to call regardless; it returns the asymmetric blocks (frsc/fzcc/...)
    and leaves the symmetric blocks as zeros.
    """
    # Shapes.
    armn_even = jnp.asarray(armn_even)
    armn_odd = jnp.asarray(armn_odd)
    ns, ntheta3, nzeta = armn_even.shape
    mpol = int(mpol)
    ntor = int(ntor)
    nfp = int(nfp)
    if mpol <= 0:
        raise ValueError("mpol must be positive")
    if ntor < 0:
        raise ValueError("ntor must be nonnegative")
    if trig.ntheta3 != int(ntheta3) or int(trig.cosnv.shape[0]) != int(nzeta):
        raise ValueError("Input grid does not match trig tables")

    # Restrict theta integration to VMEC's ntheta2 (u in [0,pi]).
    nt2 = int(trig.ntheta2)
    armn_even = armn_even[:, :nt2, :]
    armn_odd = armn_odd[:, :nt2, :]
    brmn_even = jnp.asarray(brmn_even)[:, :nt2, :]
    brmn_odd = jnp.asarray(brmn_odd)[:, :nt2, :]
    crmn_even = jnp.asarray(crmn_even)[:, :nt2, :]
    crmn_odd = jnp.asarray(crmn_odd)[:, :nt2, :]
    azmn_even = jnp.asarray(azmn_even)[:, :nt2, :]
    azmn_odd = jnp.asarray(azmn_odd)[:, :nt2, :]
    bzmn_even = jnp.asarray(bzmn_even)[:, :nt2, :]
    bzmn_odd = jnp.asarray(bzmn_odd)[:, :nt2, :]
    czmn_even = jnp.asarray(czmn_even)[:, :nt2, :]
    czmn_odd = jnp.asarray(czmn_odd)[:, :nt2, :]

    if arcon_even is None:
        arcon_even = jnp.zeros_like(armn_even)
    if arcon_odd is None:
        arcon_odd = jnp.zeros_like(armn_odd)
    if azcon_even is None:
        azcon_even = jnp.zeros_like(armn_even)
    if azcon_odd is None:
        azcon_odd = jnp.zeros_like(armn_odd)

    if blmn_even is None:
        blmn_even = jnp.zeros_like(armn_even)
    if blmn_odd is None:
        blmn_odd = jnp.zeros_like(armn_odd)
    if clmn_even is None:
        clmn_even = jnp.zeros_like(armn_even)
    if clmn_odd is None:
        clmn_odd = jnp.zeros_like(armn_odd)
    blmn_even = jnp.asarray(blmn_even)[:, :nt2, :]
    blmn_odd = jnp.asarray(blmn_odd)[:, :nt2, :]
    clmn_even = jnp.asarray(clmn_even)[:, :nt2, :]
    clmn_odd = jnp.asarray(clmn_odd)[:, :nt2, :]
    arcon_even = jnp.asarray(arcon_even)[:, :nt2, :]
    arcon_odd = jnp.asarray(arcon_odd)[:, :nt2, :]
    azcon_even = jnp.asarray(azcon_even)[:, :nt2, :]
    azcon_odd = jnp.asarray(azcon_odd)[:, :nt2, :]

    cosmui = getattr(trig, "cosmui_nt2", None)
    sinmui = getattr(trig, "sinmui_nt2", None)
    cosmumi = getattr(trig, "cosmumi_nt2", None)
    sinmumi = getattr(trig, "sinmumi_nt2", None)
    if cosmui is None or int(cosmui.shape[0]) != nt2:
        cosmui = trig.cosmui[:nt2, :]
    if sinmui is None or int(sinmui.shape[0]) != nt2:
        sinmui = trig.sinmui[:nt2, :]
    if cosmumi is None or int(cosmumi.shape[0]) != nt2:
        cosmumi = trig.cosmumi[:nt2, :]
    if sinmumi is None or int(sinmumi.shape[0]) != nt2:
        sinmumi = trig.sinmumi[:nt2, :]
    cosmui = cosmui[:, :mpol]
    sinmui = sinmui[:, :mpol]
    cosmumi = cosmumi[:, :mpol]
    sinmumi = sinmumi[:, :mpol]

    cosnv = trig.cosnv[:, : (ntor + 1)]  # (nzeta, ntor+1)
    sinnv = trig.sinnv[:, : (ntor + 1)]
    cosnvn = trig.cosnvn[:, : (ntor + 1)]
    sinnvn = trig.sinnvn[:, : (ntor + 1)]

    xmpq1 = None
    if masks is not None:
        try:
            if (int(masks.ns) == int(ns)) and (int(masks.mpol) == int(mpol)):
                xmpq1 = getattr(masks, "xmpq1", None)
        except Exception:
            xmpq1 = None
    if xmpq1 is None:
        m = np.arange(mpol, dtype=int)
        xmpq1 = (
            jnp.asarray(m, dtype=jnp.asarray(armn_even).dtype)
            * (jnp.asarray(m, dtype=jnp.asarray(armn_even).dtype) - 1.0)
        )[None, :, None]
    else:
        xmpq1 = jnp.asarray(xmpq1, dtype=jnp.asarray(armn_even).dtype)

    # Theta integration work arrays (indices per tomnspa_par).
    # Base (present always):
    #   work1(3): frsc cosnv
    #   work1(5): fzcc cosnv
    #   work1(9): flcc cosnv
    armn = jnp.stack([armn_even, armn_odd], axis=0)
    brmn = jnp.stack([brmn_even, brmn_odd], axis=0)
    crmn = jnp.stack([crmn_even, crmn_odd], axis=0)
    azmn = jnp.stack([azmn_even, azmn_odd], axis=0)
    bzmn = jnp.stack([bzmn_even, bzmn_odd], axis=0)
    czmn = jnp.stack([czmn_even, czmn_odd], axis=0)
    blmn = jnp.stack([blmn_even, blmn_odd], axis=0)
    clmn = jnp.stack([clmn_even, clmn_odd], axis=0)
    arcon = jnp.stack([arcon_even, arcon_odd], axis=0)
    azcon = jnp.stack([azcon_even, azcon_odd], axis=0)

    def _theta_einsum_stack(arr, mat):
        return _theta_contract(arr, mat)

    stack_cosmui = jnp.stack([armn, crmn, azmn, czmn, arcon, azcon, clmn], axis=0)
    stack_sinmumi = jnp.stack([brmn, bzmn, blmn], axis=0)

    cosmui_out = _theta_einsum_stack(stack_cosmui, cosmui)
    sinmui_out = _theta_einsum_stack(stack_cosmui, sinmui)
    sinmumi_out = _theta_einsum_stack(stack_sinmumi, sinmumi)
    cosmumi_out = _theta_einsum_stack(stack_sinmumi, cosmumi)

    armn_cos, crmn_cos, azmn_cos, czmn_cos, arcon_cos, azcon_cos, clmn_cos = cosmui_out
    armn_sin, crmn_sin, azmn_sin, czmn_sin, arcon_sin, azcon_sin, clmn_sin = sinmui_out
    brmn_sin, bzmn_sin, blmn_sin = sinmumi_out
    brmn_cos, bzmn_cos, blmn_cos = cosmumi_out

    xmpq1 = xmpq1[None, ...]

    w3 = armn_sin + brmn_cos + xmpq1 * arcon_sin
    w5 = azmn_cos + bzmn_sin + xmpq1 * azcon_cos
    w9 = blmn_sin

    # 3D-only:
    #   work1(1/2): frcs (sinnv/cosnvn)
    #   work1(4): frsc sinnvn
    #   work1(6): fzcc sinnvn
    #   work1(7/8): fzss (sinnv/cosnvn)
    #   work1(10): flcc sinnvn
    #   work1(11/12): flss (sinnv/cosnvn)
    w1 = armn_cos + brmn_sin + xmpq1 * arcon_cos
    w2 = -crmn_cos
    w4 = -crmn_sin
    w6 = -czmn_cos
    w7 = azmn_sin + bzmn_cos + xmpq1 * azcon_sin
    w8 = -czmn_sin
    w10 = -clmn_cos
    w11 = blmn_cos
    w12 = -clmn_sin

    w1_e, w1_o = w1[0], w1[1]
    w2_e, w2_o = w2[0], w2[1]
    w3_e, w3_o = w3[0], w3[1]
    w4_e, w4_o = w4[0], w4[1]
    w5_e, w5_o = w5[0], w5[1]
    w6_e, w6_o = w6[0], w6[1]
    w7_e, w7_o = w7[0], w7[1]
    w8_e, w8_o = w8[0], w8[1]
    w9_e, w9_o = w9[0], w9[1]
    w10_e, w10_o = w10[0], w10[1]
    w11_e, w11_o = w11[0], w11[1]
    w12_e, w12_o = w12[0], w12[1]

    mask_even = _mparity_mask(mpol, dtype=jnp.asarray(armn_even).dtype)
    w1 = _select_mparity(w1_e, w1_o, mask_even)
    w2 = _select_mparity(w2_e, w2_o, mask_even)
    w3 = _select_mparity(w3_e, w3_o, mask_even)
    w4 = _select_mparity(w4_e, w4_o, mask_even)
    w5 = _select_mparity(w5_e, w5_o, mask_even)
    w6 = _select_mparity(w6_e, w6_o, mask_even)
    w7 = _select_mparity(w7_e, w7_o, mask_even)
    w8 = _select_mparity(w8_e, w8_o, mask_even)
    w9 = _select_mparity(w9_e, w9_o, mask_even)
    w10 = _select_mparity(w10_e, w10_o, mask_even)
    w11 = _select_mparity(w11_e, w11_o, mask_even)
    w12 = _select_mparity(w12_e, w12_o, mask_even)

    lthreed = bool(ntor > 0)

    # Zeta integration.
    w_cosnv = jnp.stack([w3, w5, w9], axis=0)
    out_cosnv = _zeta_contract(w_cosnv, cosnv)
    frsc, fzcc, flcc = out_cosnv[0], out_cosnv[1], out_cosnv[2]

    if lthreed:
        w_sinnvn = jnp.stack([w4, w6, w10], axis=0)
        out_sinnvn = _zeta_contract(w_sinnvn, sinnvn)
        frsc = frsc + out_sinnvn[0]
        fzcc = fzcc + out_sinnvn[1]
        flcc = flcc + out_sinnvn[2]

        w_sinnv = jnp.stack([w1, w7, w11], axis=0)
        w_cosnvn = jnp.stack([w2, w8, w12], axis=0)
        out_sinnv = _zeta_contract(w_sinnv, sinnv)
        out_cosnvn = _zeta_contract(w_cosnvn, cosnvn)

        frcs = out_sinnv[0] + out_cosnvn[0]
        fzss = out_sinnv[1] + out_cosnvn[1]
        flss = out_sinnv[2] + out_cosnvn[2]
    else:
        frcs = None
        fzss = None
        flss = None

    # VMEC `tomnspa` note (tomnsp_mod.f): the antisymmetric transform is
    # performed on a restricted theta interval after `symforce`. For the
    # 3D+lasym lambda blocks, VMEC's conventions imply an additional √2 scaling
    # compared to the symmetric (`tomnsps`) lambda blocks. This improves `fsql`
    # parity on lasym+3D reference equilibria.
    if bool(lthreed):
        s2 = jnp.asarray(np.sqrt(2.0), dtype=jnp.asarray(flcc).dtype)
        flcc = flcc * s2
        if flss is not None:
            flss = flss * s2

    # Apply radial evolution masks (same as tomnsps): fixed-boundary edge.
    if masks is not None and (int(masks.ns) == int(ns)) and (int(masks.mpol) == int(mpol)) and (bool(masks.include_edge) == bool(include_edge)):
        mask_rz = jnp.asarray(masks.mask_rz, dtype=jnp.asarray(frsc).dtype)
        mask_l = jnp.asarray(masks.mask_l, dtype=jnp.asarray(flcc).dtype)
    else:
        js_fortran = jnp.arange(ns, dtype=jnp.int32) + 1  # 1..ns
        m_fortran = jnp.arange(mpol, dtype=jnp.int32)  # 0..mpol-1
        jmin2 = jnp.where(m_fortran == 0, 1, 2)[None, :]  # (1, mpol)
        jlam = jnp.full((1, mpol), 2, dtype=jmin2.dtype)

        jsmax_rz = int(ns if include_edge else (ns - 1))
        mask_rz = (js_fortran[:, None] >= jmin2) & (js_fortran[:, None] <= jsmax_rz)
        mask_l = js_fortran[:, None] >= jlam
        mask_rz = mask_rz.astype(jnp.asarray(frsc).dtype)[:, :, None]
        mask_l = mask_l.astype(jnp.asarray(flcc).dtype)[:, :, None]

    frsc = frsc * mask_rz
    fzcc = fzcc * mask_rz
    if frcs is not None:
        frcs = frcs * mask_rz
    if fzss is not None:
        fzss = fzss * mask_rz

    flcc = flcc * mask_l
    if flss is not None:
        flss = flss * mask_l

    # Return with symmetric blocks zeroed; caller is expected to merge with tomnsps.
    z_sym = jnp.zeros_like(frsc)
    z3 = None if not lthreed else jnp.zeros_like(frsc)
    return TomnspsRZL(
        frcc=z_sym,
        frss=z3,
        fzsc=z_sym,
        fzcs=z3,
        flsc=z_sym,
        flcs=z3,
        frsc=frsc,
        frcs=frcs,
        fzcc=fzcc,
        fzss=fzss,
        flcc=flcc,
        flss=flss,
    )
