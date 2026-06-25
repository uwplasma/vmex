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

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import os
import numpy as np

from ._compat import jnp, tree_util, has_jax
from .grids import AngleGrid


@tree_util.register_pytree_node_class
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

    # Zeta tables. Shapes (nzeta, nmax+1).
    cosnv: Any
    sinnv: Any
    cosnvn: Any
    sinnvn: Any

    # Precomputed angular weights for the 1D preconditioner (wint3).
    wint3_precond: Any | None = None

    # Cached theta slices (ntheta2, mmax+1) to reduce per-call slicing cost.
    cosmui_nt2: Any | None = None
    sinmui_nt2: Any | None = None
    cosmumi_nt2: Any | None = None
    sinmumi_nt2: Any | None = None
    # Cached fused theta/zeta bases to reduce per-call concatenation.
    basis_theta_cs_nt2: Any | None = None  # (ntheta2, 2*(mmax+1))
    basis_theta_mu_nt2: Any | None = None  # (ntheta2, 2*(mmax+1))
    basis_zeta_cs: Any | None = None  # (nzeta, 2*(nmax+1))
    basis_zeta_all: Any | None = None  # (nzeta, 4*(nmax+1))

    # Optional cached phase stacks for vmec_realspace synthesis.
    # These are populated by VMECStatic when enabled.
    phase_stack: Any | None = None
    phase_dtheta_stack: Any | None = None
    phase_dzeta_stack: Any | None = None
    phase_stack_m: Any | None = None
    phase_stack_n: Any | None = None

    def tree_flatten(self):
        children = (
            self.mscale,
            self.nscale,
            self.cosmu,
            self.sinmu,
            self.cosmum,
            self.sinmum,
            self.cosmui,
            self.sinmui,
            self.cosmumi,
            self.sinmumi,
            self.cosmui3,
            self.cosmumi3,
            self.cosnv,
            self.sinnv,
            self.cosnvn,
            self.sinnvn,
            self.wint3_precond,
            self.cosmui_nt2,
            self.sinmui_nt2,
            self.cosmumi_nt2,
            self.sinmumi_nt2,
            self.basis_theta_cs_nt2,
            self.basis_theta_mu_nt2,
            self.basis_zeta_cs,
            self.basis_zeta_all,
            self.phase_stack,
            self.phase_dtheta_stack,
            self.phase_dzeta_stack,
            self.phase_stack_m,
            self.phase_stack_n,
        )
        aux = (
            int(self.ntheta1),
            int(self.ntheta2),
            int(self.ntheta3),
            float(self.dnorm),
            float(self.dnorm3),
            float(self.r0scale),
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            ntheta1,
            ntheta2,
            ntheta3,
            dnorm,
            dnorm3,
            r0scale,
        ) = aux_data
        return cls(
            ntheta1=int(ntheta1),
            ntheta2=int(ntheta2),
            ntheta3=int(ntheta3),
            dnorm=float(dnorm),
            dnorm3=float(dnorm3),
            r0scale=float(r0scale),
            mscale=children[0],
            nscale=children[1],
            cosmu=children[2],
            sinmu=children[3],
            cosmum=children[4],
            sinmum=children[5],
            cosmui=children[6],
            sinmui=children[7],
            cosmumi=children[8],
            sinmumi=children[9],
            cosmui3=children[10],
            cosmumi3=children[11],
            cosnv=children[12],
            sinnv=children[13],
            cosnvn=children[14],
            sinnvn=children[15],
            wint3_precond=children[16],
            cosmui_nt2=children[17],
            sinmui_nt2=children[18],
            cosmumi_nt2=children[19],
            sinmumi_nt2=children[20],
            basis_theta_cs_nt2=children[21],
            basis_theta_mu_nt2=children[22],
            basis_zeta_cs=children[23],
            basis_zeta_all=children[24],
            phase_stack=children[25],
            phase_dtheta_stack=children[26],
            phase_dzeta_stack=children[27],
            phase_stack_m=children[28],
            phase_stack_n=children[29],
        )


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
    # Cached JAX arrays to avoid per-call host->device copies.
    mask_even_j: Any | None = None
    mask_rz_j: Any | None = None
    mask_l_j: Any | None = None
    xmpq1_j: Any | None = None


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

    # VMEC fixaray.f normalizes LASYM transforms on the full theta grid:
    #   dnorm = 1/(nzeta*ntheta3)  ! SPH012314
    # while stellarator-symmetric transforms use endpoint-weighted [0, pi].
    if lasym:
        dnorm = 1.0 / (nzeta * ntheta3)
    else:
        dnorm = 1.0 / (nzeta * (ntheta2 - 1))
    # dnorm3 normalization for surface averages.
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
    basis_theta_cs_nt2 = np.concatenate([cosmui_nt2, sinmui_nt2], axis=1)
    basis_theta_mu_nt2 = np.concatenate([sinmumi_nt2, cosmumi_nt2], axis=1)

    # Zeta tables use argj = 2π*(j-1)/nzeta, for j=1..nzeta.
    j = np.arange(nzeta, dtype=float)
    zeta = (2.0 * np.pi) * j / float(nzeta)
    n = np.arange(nmax + 1, dtype=float)
    argn = zeta[:, None] * n[None, :]
    cosnv = np.cos(argn) * nscale[None, :]
    sinnv = np.sin(argn) * nscale[None, :]

    cosnvn = cosnv * (n[None, :] * float(nfp))
    sinnvn = -sinnv * (n[None, :] * float(nfp))
    basis_zeta_cs = np.concatenate([cosnv, sinnv], axis=1)
    basis_zeta_all = np.concatenate([cosnv, sinnvn, sinnv, cosnvn], axis=1)

    # Preconditioner angular weights (wint3) on the VMEC internal grid.
    w_theta = cosmui3[:, 0] / float(mscale[0])
    wint = w_theta[:, None] * np.ones((nzeta,), dtype=float)[None, :]
    wint3_precond = wint[None, :, :]

    # Store as plain NumPy arrays.  Every field was computed with np.* above,
    # so wrapping with jnp.asarray here triggers 26 separate eager XLA
    # compilations (one copy-to-device primitive per field, ~2 ms each).
    # JAX automatically promotes NumPy arrays to device arrays when the trig
    # tables are captured as closure constants inside _run_scan at JIT boundary,
    # so the explicit conversion is both redundant and expensive on cold start.
    _np_dtype = np.dtype(dtype) if not isinstance(dtype, np.dtype) else dtype
    tables = VmecTrigTables(
        ntheta1=ntheta1,
        ntheta2=ntheta2,
        ntheta3=ntheta3,
        dnorm=float(dnorm),
        dnorm3=float(dnorm3),
        mscale=np.asarray(mscale, dtype=_np_dtype),
        nscale=np.asarray(nscale, dtype=_np_dtype),
        r0scale=float(r0scale),
        cosmu=np.asarray(cosmu, dtype=_np_dtype),
        sinmu=np.asarray(sinmu, dtype=_np_dtype),
        cosmum=np.asarray(cosmum, dtype=_np_dtype),
        sinmum=np.asarray(sinmum, dtype=_np_dtype),
        cosmui=np.asarray(cosmui, dtype=_np_dtype),
        sinmui=np.asarray(sinmui, dtype=_np_dtype),
        cosmumi=np.asarray(cosmumi, dtype=_np_dtype),
        sinmumi=np.asarray(sinmumi, dtype=_np_dtype),
        cosmui3=np.asarray(cosmui3, dtype=_np_dtype),
        cosmumi3=np.asarray(cosmumi3, dtype=_np_dtype),
        cosmui_nt2=np.asarray(cosmui_nt2, dtype=_np_dtype),
        sinmui_nt2=np.asarray(sinmui_nt2, dtype=_np_dtype),
        cosmumi_nt2=np.asarray(cosmumi_nt2, dtype=_np_dtype),
        sinmumi_nt2=np.asarray(sinmumi_nt2, dtype=_np_dtype),
        basis_theta_cs_nt2=np.asarray(basis_theta_cs_nt2, dtype=_np_dtype),
        basis_theta_mu_nt2=np.asarray(basis_theta_mu_nt2, dtype=_np_dtype),
        cosnv=np.asarray(cosnv, dtype=_np_dtype),
        sinnv=np.asarray(sinnv, dtype=_np_dtype),
        cosnvn=np.asarray(cosnvn, dtype=_np_dtype),
        sinnvn=np.asarray(sinnvn, dtype=_np_dtype),
        basis_zeta_cs=np.asarray(basis_zeta_cs, dtype=_np_dtype),
        basis_zeta_all=np.asarray(basis_zeta_all, dtype=_np_dtype),
        wint3_precond=np.asarray(wint3_precond, dtype=_np_dtype),
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
    nfp = int(nfp)
    if ntheta1 <= 0 or ntheta2 <= 0 or ntheta3 <= 0:
        raise ValueError("Invalid theta sizes")
    if nfp <= 0:
        raise ValueError("nfp must be positive")
    if nzeta <= 0:
        nzeta = 1
    cache_key = (int(ntheta), int(nzeta), nfp, bool(lasym))
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
    grid = AngleGrid(theta=theta.astype(float), zeta=zeta.astype(float), nfp=nfp)
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


_MPARITY_CACHE: dict[tuple[int, str], np.ndarray] = {}
_JNP_EINSUM = jnp.einsum
_DETERMINISTIC_REDUCE = bool(int(os.environ.get("VMEC_JAX_DETERMINISTIC_REDUCE", "0")))
# FFT path for stellarator-symmetric cases.
# Default: auto-detect — ON for GPU/TPU (cuFFT is fast), OFF for CPU (DFT-GEMM wins
# for typical small ntor grids). Override with VMEC_JAX_TOMNSPS_FFT=0/1.
_TOMNSPS_FFT_ENV: str = os.environ.get("VMEC_JAX_TOMNSPS_FFT", "").strip().lower()
_TOMNSPS_FFT_CACHE: list[bool] = []  # populated on first call


@contextmanager
def tomnsps_fft_policy_override(enabled: bool | None):
    """Temporarily override auto TOMNSPS FFT policy when the solver device is explicit.

    JAX's ``default_device`` context does not change ``jax.default_backend()``.
    Without this override, a process imported on CPU can keep the CPU DFT/GEMM
    policy even when a public solve is explicitly routed to GPU.
    """

    global _TOMNSPS_FFT_ENV

    if enabled is None or _TOMNSPS_FFT_ENV in ("1", "true", "yes", "0", "false", "no"):
        yield
        return

    old_env = _TOMNSPS_FFT_ENV
    old_cache = list(_TOMNSPS_FFT_CACHE)
    _TOMNSPS_FFT_ENV = "1" if bool(enabled) else "0"
    _TOMNSPS_FFT_CACHE.clear()
    try:
        yield
    finally:
        _TOMNSPS_FFT_ENV = old_env
        _TOMNSPS_FFT_CACHE.clear()
        _TOMNSPS_FFT_CACHE.extend(old_cache)


def _get_tomnsps_fft() -> bool:
    """Return whether to use the FFT synthesis path (lazy, cached after first call)."""
    if _TOMNSPS_FFT_CACHE:
        return _TOMNSPS_FFT_CACHE[0]
    if _TOMNSPS_FFT_ENV in ("1", "true", "yes"):
        result = True
    elif _TOMNSPS_FFT_ENV in ("0", "false", "no"):
        result = False
    else:
        # Auto: GPU/TPU → FFT (cuFFT / XLA FFT is fast);  CPU → DFT-GEMM.
        try:
            import jax as _jax

            result = _jax.default_backend() not in ("cpu",)
        except Exception:
            result = False
    _TOMNSPS_FFT_CACHE.append(result)
    return result
_TOMNSPS_THETA_FUSED = os.environ.get("VMEC_JAX_TOMNSPS_THETA_FUSED", "1").strip().lower() not in (
    "",
    "0",
    "false",
    "no",
)
_TOMNSPS_ZETA_FUSED = os.environ.get("VMEC_JAX_TOMNSPS_ZETA_FUSED", "1").strip().lower() not in (
    "",
    "0",
    "false",
    "no",
)
# NOTE: Keep tomnspa zeta fusion off by default for LASYM parity. In practice,
# the large fused GEMM introduces cancellation-order drift in flcc/flss for
# low-(m,n) channels (notably stage-3 iter-2 in LASYM parity traces). Users can
# still opt in explicitly when they want to trade parity for speed.
_TOMNSPA_ZETA_FUSED = os.environ.get("VMEC_JAX_TOMNSPA_ZETA_FUSED", "0").strip().lower() not in (
    "",
    "0",
    "false",
    "no",
)


def _einsum(expr: str, *operands):
    if has_jax():
        try:
            from jax import lax

            return _JNP_EINSUM(expr, *operands, precision=lax.Precision.HIGHEST)
        except TypeError:
            return _JNP_EINSUM(expr, *operands)
    return _JNP_EINSUM(expr, *operands)


def _theta_contract(arr, mat):
    """Theta reduction using precomputed basis; dot_general for GEMM fusion."""
    if (not has_jax()) or _DETERMINISTIC_REDUCE:
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
    from jax import lax

    arr = jnp.asarray(arr)
    mat = jnp.asarray(mat)
    a, p, s, i_size, k = arr.shape
    arr_t = jnp.moveaxis(arr, 3, -1)  # (a, p, s, k, i)
    arr2 = arr_t.reshape((a * p * s * k, i_size))
    out2 = lax.dot_general(
        arr2,
        mat,
        dimension_numbers=(((1,), (0,)), ((), ())),
        precision=lax.Precision.HIGHEST,
    )
    out = out2.reshape((a, p, s, k, mat.shape[1]))
    return jnp.moveaxis(out, -1, -2)  # (a, p, s, m, k)


def _zeta_contract(arr, mat):
    """Zeta reduction using precomputed basis; dot_general for GEMM fusion."""
    if (not has_jax()) or _DETERMINISTIC_REDUCE:
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
    from jax import lax

    arr = jnp.asarray(arr)
    mat = jnp.asarray(mat)
    p, s, m, k_size = arr.shape
    arr2 = arr.reshape((p * s * m, k_size))
    out2 = lax.dot_general(
        arr2,
        mat,
        dimension_numbers=(((1,), (0,)), ((), ())),
        precision=lax.Precision.HIGHEST,
    )
    return out2.reshape((p, s, m, mat.shape[1]))



def _theta_transform_fft(arr, *, mpol: int, dnorm: float, mscale, want_sin: bool):
    """FFT-based theta transform for lasym=False (DCT-I / DST-I via FFT)."""
    arr = jnp.asarray(arr)
    nt2 = int(arr.shape[-2])
    if nt2 < 2:
        out_shape = arr.shape[:-2] + (int(mpol), int(arr.shape[-1]))
        return jnp.zeros(out_shape, dtype=arr.dtype)
    if want_sin:
        # Enforce zero endpoints for sine series.
        arr = arr.at[..., 0, :].set(0.0)
        arr = arr.at[..., nt2 - 1, :].set(0.0)
        ext = jnp.concatenate([arr, -arr[..., nt2 - 2 : 0 : -1, :]], axis=-2)
        fft = jnp.fft.rfft(ext, axis=-2)
        coeff = -jnp.imag(fft[..., :mpol, :])
    else:
        ext = jnp.concatenate([arr, arr[..., nt2 - 2 : 0 : -1, :]], axis=-2)
        fft = jnp.fft.rfft(ext, axis=-2)
        coeff = jnp.real(fft[..., :mpol, :])
    # Match VMEC's normalization: dnorm and mscale (sqrt2 for m>0).
    mscale = jnp.asarray(mscale[:mpol], dtype=coeff.dtype)
    mshape = (1,) * (coeff.ndim - 2) + (int(mpol), 1)
    coeff = 0.5 * coeff * mscale.reshape(mshape) * jnp.asarray(dnorm, dtype=coeff.dtype)
    return coeff


def _zeta_transform_fft(arr, *, ntor: int, nscale, want_sin: bool):
    """FFT-based zeta transform for periodic zeta grid."""
    arr = jnp.asarray(arr)
    nzeta = int(arr.shape[-1])
    if nzeta == 0:
        out_shape = arr.shape[:-1] + (int(ntor) + 1,)
        return jnp.zeros(out_shape, dtype=arr.dtype)
    fft = jnp.fft.rfft(arr, axis=-1)
    coeff = -jnp.imag(fft[..., : (ntor + 1)]) if want_sin else jnp.real(fft[..., : (ntor + 1)])
    nscale = jnp.asarray(nscale[: (ntor + 1)], dtype=coeff.dtype)
    nshape = (1,) * (coeff.ndim - 1) + (int(ntor) + 1,)
    return coeff * nscale.reshape(nshape)


def _cache_allowed() -> bool:
    if not has_jax():
        return True
    try:
        from jax import core

        return bool(core.trace_ctx.is_top_level())
    except Exception:
        return False


def _mparity_mask(mpol: int, *, dtype) -> np.ndarray:
    key = (int(mpol), str(np.dtype(dtype)))
    if _cache_allowed():
        cached = _MPARITY_CACHE.get(key)
        if cached is not None:
            return cached
    m = np.arange(int(mpol))
    mask_even = np.asarray((m % 2) == 0, dtype=np.dtype(dtype))
    if _cache_allowed():
        _MPARITY_CACHE[key] = mask_even
    return mask_even


def _select_mparity(a_even, a_odd, mask_even: jnp.ndarray):
    mask = jnp.asarray(mask_even)[None, :, None] > 0
    return jnp.where(mask, a_even, a_odd)


def _select_mparity_pairs(mask_even: jnp.ndarray, *pairs):
    return tuple(_select_mparity(pair[0], pair[1], mask_even) for pair in pairs)


def _slice_theta2(a, nt2: int):
    return jnp.asarray(a)[:, : int(nt2), :]


def _slice_theta2_many(nt2: int, *values):
    return tuple(_slice_theta2(value, nt2) for value in values)


def _optional_theta2(a, like, nt2: int):
    return jnp.zeros_like(like) if a is None else _slice_theta2(a, nt2)


def _optional_theta2_pairs(nt2: int, even_like, odd_like, *pairs):
    out = []
    for even, odd in pairs:
        out.extend((_optional_theta2(even, even_like, nt2), _optional_theta2(odd, odd_like, nt2)))
    return tuple(out)


def _stack_even_odd_pairs(*pairs):
    return tuple(jnp.stack(pair, axis=0) for pair in pairs)


def _theta2_tables(trig: VmecTrigTables, nt2: int, mpol: int):
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
    return cosmui[:, :mpol], sinmui[:, :mpol], cosmumi[:, :mpol], sinmumi[:, :mpol]


def _masks_match(masks: TomnspsMasks | None, ns: int, mpol: int, include_edge: bool | None = None) -> bool:
    try:
        ok = masks is not None and (int(masks.ns) == int(ns)) and (int(masks.mpol) == int(mpol))
        if include_edge is not None:
            ok = ok and (bool(masks.include_edge) == bool(include_edge))
        return bool(ok)
    except Exception:
        return False


def _xmpq1_from_masks(masks: TomnspsMasks | None, ns: int, mpol: int, dtype_like):
    if _masks_match(masks, ns, mpol):
        xmpq1 = getattr(masks, "xmpq1_j", None)
        if xmpq1 is None:
            xmpq1 = getattr(masks, "xmpq1", None)
        if xmpq1 is not None:
            return jnp.asarray(xmpq1, dtype=jnp.asarray(dtype_like).dtype)
    m = np.arange(mpol, dtype=int)
    m = jnp.asarray(m, dtype=jnp.asarray(dtype_like).dtype)
    return (m * (m - 1.0))[None, :, None]


def _mparity_from_masks(masks: TomnspsMasks | None, ns: int, mpol: int, include_edge: bool, dtype_like):
    if _masks_match(masks, ns, mpol, include_edge):
        mask_even = getattr(masks, "mask_even_j", None)
        if mask_even is None:
            mask_even = getattr(masks, "mask_even", None)
        if mask_even is not None:
            return jnp.asarray(mask_even, dtype=jnp.asarray(dtype_like).dtype)
    return _mparity_mask(mpol, dtype=jnp.asarray(dtype_like).dtype)


def _radial_masks_from_masks(masks: TomnspsMasks | None, ns: int, mpol: int, include_edge: bool, rz_like, l_like):
    if _masks_match(masks, ns, mpol, include_edge):
        mask_rz = getattr(masks, "mask_rz_j", None)
        mask_l = getattr(masks, "mask_l_j", None)
        if mask_rz is None:
            mask_rz = jnp.asarray(masks.mask_rz, dtype=jnp.asarray(rz_like).dtype)
        if mask_l is None:
            mask_l = jnp.asarray(masks.mask_l, dtype=jnp.asarray(l_like).dtype)
        return mask_rz, mask_l

    js_fortran = jnp.arange(ns, dtype=jnp.int32) + 1  # 1..ns
    m_fortran = jnp.arange(mpol, dtype=jnp.int32)  # 0..mpol-1
    jmin2 = jnp.where(m_fortran == 0, 1, 2)[None, :]
    jlam = jnp.full((1, mpol), 2, dtype=jmin2.dtype)
    jsmax_rz = int(ns if include_edge else (ns - 1))
    mask_rz = (js_fortran[:, None] >= jmin2) & (js_fortran[:, None] <= jsmax_rz)
    mask_l = js_fortran[:, None] >= jlam
    return (
        mask_rz.astype(jnp.asarray(rz_like).dtype)[:, :, None],
        mask_l.astype(jnp.asarray(l_like).dtype)[:, :, None],
    )


def _apply_tomnsp_radial_masks(*, masks, ns: int, mpol: int, include_edge: bool, rz_fields, l_fields):
    """Apply VMEC fixed-boundary radial masks to R/Z and lambda blocks."""

    mask_rz, mask_l = _radial_masks_from_masks(
        masks,
        ns=ns,
        mpol=mpol,
        include_edge=include_edge,
        rz_like=rz_fields[0],
        l_like=l_fields[0],
    )
    return (
        tuple(None if field is None else field * mask_rz for field in rz_fields),
        tuple(None if field is None else field * mask_l for field in l_fields),
    )


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

    np_dtype = np.dtype(dtype)
    mask_even = _mparity_mask(mpol, dtype=np_dtype)
    js_fortran = np.arange(ns, dtype=np.int32) + 1  # 1..ns
    m_fortran = np.arange(mpol, dtype=np.int32)  # 0..mpol-1
    jmin2 = np.where(m_fortran == 0, 1, 2)[None, :]
    jlam = np.full((1, mpol), 2, dtype=np.int32)
    jsmax_rz = int(ns if include_edge else (ns - 1))
    mask_rz = (js_fortran[:, None] >= jmin2) & (js_fortran[:, None] <= jsmax_rz)
    mask_l = js_fortran[:, None] >= jlam
    mask_rz = mask_rz.astype(mask_even.dtype)[:, :, None]
    mask_l = mask_l.astype(mask_even.dtype)[:, :, None]
    m_fortran_f = np.asarray(m_fortran, dtype=mask_even.dtype)
    xmpq1 = (m_fortran_f * (m_fortran_f - 1.0))[None, :, None]
    masks = TomnspsMasks(
        ns=ns,
        mpol=mpol,
        include_edge=include_edge,
        mask_even=mask_even,
        mask_rz=mask_rz,
        mask_l=mask_l,
        xmpq1=xmpq1,
        mask_even_j=mask_even,
        mask_rz_j=mask_rz,
        mask_l_j=mask_l,
        xmpq1_j=xmpq1,
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
    (
        armn_even, armn_odd, brmn_even, brmn_odd, crmn_even, crmn_odd,
        azmn_even, azmn_odd, bzmn_even, bzmn_odd, czmn_even, czmn_odd,
    ) = _slice_theta2_many(
        nt2,
        armn_even, armn_odd, brmn_even, brmn_odd, crmn_even, crmn_odd,
        azmn_even, azmn_odd, bzmn_even, bzmn_odd, czmn_even, czmn_odd,
    )
    (
        arcon_even, arcon_odd, azcon_even, azcon_odd,
        blmn_even, blmn_odd, clmn_even, clmn_odd,
    ) = _optional_theta2_pairs(
        nt2,
        armn_even,
        armn_odd,
        (arcon_even, arcon_odd),
        (azcon_even, azcon_odd),
        (blmn_even, blmn_odd),
        (clmn_even, clmn_odd),
    )

    # Tables for m=0..mpol-1 and n=0..ntor.
    cosmui, sinmui, cosmumi, sinmumi = _theta2_tables(trig, nt2, mpol)

    cosnv = trig.cosnv[:, : (ntor + 1)]  # (nzeta, ntor+1)
    sinnv = trig.sinnv[:, : (ntor + 1)]

    # VMEC constraint operator multiplier: xmpq(m,1)=m*(m-1).
    xmpq1 = _xmpq1_from_masks(masks, ns=ns, mpol=mpol, dtype_like=armn_even)

    # Theta integration: compute work arrays for even-parity and odd-parity pieces.
    # Each is (ns, mpol, nzeta).
    # work1 indices follow tomnsp_mod.f numbering but we compute only needed combos.
    armn, brmn, crmn, azmn, bzmn, czmn, blmn, clmn, arcon, azcon = _stack_even_odd_pairs(
        (armn_even, armn_odd),
        (brmn_even, brmn_odd),
        (crmn_even, crmn_odd),
        (azmn_even, azmn_odd),
        (bzmn_even, bzmn_odd),
        (czmn_even, czmn_odd),
        (blmn_even, blmn_odd),
        (clmn_even, clmn_odd),
        (arcon_even, arcon_odd),
        (azcon_even, azcon_odd),
    )

    stack_cosmui = jnp.stack([armn, crmn, azmn, czmn, arcon, azcon, clmn], axis=0)
    stack_sinmumi = jnp.stack([brmn, bzmn, blmn], axis=0)

    use_fft = _get_tomnsps_fft() and (not bool(lasym)) and has_jax()
    use_fft_fused = True
    if use_fft:
        env_fused = os.getenv("VMEC_JAX_TOMNSPS_FFT_FUSED", "1").strip().lower()
        use_fft_fused = env_fused not in ("", "0", "false", "no")
    if use_fft:
        dnorm = float(getattr(trig, "dnorm", 1.0))
        mscale = getattr(trig, "mscale", None)
        if mscale is None:
            mscale = jnp.ones((mpol,), dtype=jnp.asarray(armn_even).dtype)
        if use_fft_fused:
            stack_all = jnp.concatenate([stack_cosmui, stack_sinmumi], axis=0)
            cos_all = _theta_transform_fft(stack_all, mpol=mpol, dnorm=dnorm, mscale=mscale, want_sin=False)
            sin_all = _theta_transform_fft(stack_all, mpol=mpol, dnorm=dnorm, mscale=mscale, want_sin=True)
            n_cos = int(stack_cosmui.shape[0])
            cosmui_out = cos_all[:n_cos]
            sinmui_out = sin_all[:n_cos]
            cos_coeff = cos_all[n_cos:]
            sin_coeff = sin_all[n_cos:]
        else:
            cosmui_out = _theta_transform_fft(stack_cosmui, mpol=mpol, dnorm=dnorm, mscale=mscale, want_sin=False)
            sinmui_out = _theta_transform_fft(stack_cosmui, mpol=mpol, dnorm=dnorm, mscale=mscale, want_sin=True)
            sin_coeff = _theta_transform_fft(stack_sinmumi, mpol=mpol, dnorm=dnorm, mscale=mscale, want_sin=True)
            cos_coeff = _theta_transform_fft(stack_sinmumi, mpol=mpol, dnorm=dnorm, mscale=mscale, want_sin=False)
        m = jnp.arange(int(mpol), dtype=cos_coeff.dtype)
        mshape = (1,) * (cos_coeff.ndim - 2) + (int(mpol), 1)
        m = m.reshape(mshape)
        sinmumi_out = -sin_coeff * m
        cosmumi_out = cos_coeff * m
    else:
        # DFT path: use a single cos/sin basis transform for both stacks,
        # then apply m-derivative scaling for the sinmumi/cosmumi blocks.
        stack_all = jnp.concatenate([stack_cosmui, stack_sinmumi], axis=0)
        use_theta_fused = bool(_TOMNSPS_THETA_FUSED)
        if use_theta_fused:
            basis_theta = getattr(trig, "basis_theta_cs_nt2", None)
            if basis_theta is None or int(basis_theta.shape[0]) != nt2:
                basis_theta = jnp.concatenate([cosmui, sinmui], axis=1)
            basis_theta = basis_theta[:, : (2 * int(mpol))]
            out_all = _theta_contract(stack_all, basis_theta)
            mpol_i = int(mpol)
            cos_all = out_all[..., :mpol_i, :]
            sin_all = out_all[..., mpol_i:, :]
        else:
            cos_all = _theta_contract(stack_all, cosmui)
            sin_all = _theta_contract(stack_all, sinmui)
        n_cos = int(stack_cosmui.shape[0])
        cosmui_out = cos_all[:n_cos]
        sinmui_out = sin_all[:n_cos]
        cos_sinmumi = cos_all[n_cos:]
        sin_sinmumi = sin_all[n_cos:]
        m = jnp.arange(int(mpol), dtype=cos_sinmumi.dtype)
        mshape = (1,) * (cos_sinmumi.ndim - 2) + (int(mpol), 1)
        m = m.reshape(mshape)
        sinmumi_out = -sin_sinmumi * m
        cosmumi_out = cos_sinmumi * m

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

    # Select parity per m (mparity = mod(m,2)).
    mask_even = _mparity_from_masks(
        masks,
        ns=ns,
        mpol=mpol,
        include_edge=include_edge,
        dtype_like=armn_even,
    )
    w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12 = _select_mparity_pairs(
        mask_even, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12
    )

    lthreed = bool(ntor > 0)

    # Zeta integration. Result arrays are (ns, mpol, ntor+1).
    use_fft_zeta = _get_tomnsps_fft() and has_jax()
    if use_fft_zeta:
        env_fused = os.getenv("VMEC_JAX_TOMNSPS_FFT_FUSED", "1").strip().lower()
        use_fft_fused = env_fused not in ("", "0", "false", "no")
        nscale = getattr(trig, "nscale", None)
        if nscale is None:
            nscale = jnp.ones((ntor + 1,), dtype=jnp.asarray(w1).dtype)
        n = jnp.arange(ntor + 1, dtype=jnp.asarray(w1).dtype)
        nfac = (n * float(nfp)).reshape((1, 1, 1, ntor + 1))
        w_cosnv = jnp.stack([w1, w7, w11], axis=0)
        if use_fft_fused:
            if lthreed:
                w_sinnvn = jnp.stack([w2, w8, w12], axis=0)
                w_sinnv = jnp.stack([w3, w5, w9], axis=0)
                w_cosnvn = jnp.stack([w4, w6, w10], axis=0)
                w_stack = jnp.stack([w_cosnv, w_sinnvn, w_sinnv, w_cosnvn], axis=0)
            else:
                w_stack = w_cosnv[None, ...]

            fft = jnp.fft.rfft(w_stack, axis=-1)
            coeff = fft[..., : (ntor + 1)]
            real_dtype = jnp.asarray(w1).dtype
            nscale = jnp.asarray(nscale[: (int(ntor) + 1)], dtype=real_dtype)
            nshape = (1,) * (coeff.ndim - 1) + (int(ntor) + 1,)
            coeff_real = jnp.real(coeff) * nscale.reshape(nshape)
            coeff_imag = -jnp.imag(coeff) * nscale.reshape(nshape)

            out_cosnv = coeff_real[0]
            frcc, fzsc, flsc = out_cosnv[0], out_cosnv[1], out_cosnv[2]

            if lthreed:
                out_sinnvn = -coeff_imag[1] * nfac
                frcc = frcc + out_sinnvn[0]
                fzsc = fzsc + out_sinnvn[1]
                flsc = flsc + out_sinnvn[2]

                out_sinnv = coeff_imag[2]
                out_cosnvn = coeff_real[3] * nfac
                frss = out_sinnv[0] + out_cosnvn[0]
                fzcs = out_sinnv[1] + out_cosnvn[1]
                flcs = out_sinnv[2] + out_cosnvn[2]
            else:
                frss = None
                fzcs = None
                flcs = None
        else:
            out_cosnv = _zeta_transform_fft(w_cosnv, ntor=ntor, nscale=nscale, want_sin=False)
            frcc, fzsc, flsc = out_cosnv[0], out_cosnv[1], out_cosnv[2]

            if lthreed:
                w_sinnvn = jnp.stack([w2, w8, w12], axis=0)
                out_sinnvn = -_zeta_transform_fft(w_sinnvn, ntor=ntor, nscale=nscale, want_sin=True) * nfac
                frcc = frcc + out_sinnvn[0]
                fzsc = fzsc + out_sinnvn[1]
                flsc = flsc + out_sinnvn[2]

                w_sinnv = jnp.stack([w3, w5, w9], axis=0)
                w_cosnvn = jnp.stack([w4, w6, w10], axis=0)
                out_sinnv = _zeta_transform_fft(w_sinnv, ntor=ntor, nscale=nscale, want_sin=True)
                out_cosnvn = _zeta_transform_fft(w_cosnvn, ntor=ntor, nscale=nscale, want_sin=False) * nfac

                frss = out_sinnv[0] + out_cosnvn[0]
                fzcs = out_sinnv[1] + out_cosnvn[1]
                flcs = out_sinnv[2] + out_cosnvn[2]
            else:
                frss = None
                fzcs = None
                flcs = None
    else:
        if lthreed:
            n = jnp.arange(ntor + 1, dtype=jnp.asarray(w1).dtype)
            nfac = (n * float(nfp)).reshape((1, 1, 1, int(ntor) + 1))

            use_zeta_fused = bool(_TOMNSPS_ZETA_FUSED)
            if use_zeta_fused:
                w_cos_stack = jnp.stack([w1, w7, w11, w4, w6, w10], axis=0)
                w_sin_stack = jnp.stack([w2, w8, w12, w3, w5, w9], axis=0)
                w_stack = jnp.concatenate([w_cos_stack, w_sin_stack], axis=0)
                basis_cs = getattr(trig, "basis_zeta_cs", None)
                if basis_cs is None or int(basis_cs.shape[0]) != int(nzeta):
                    basis_cs = jnp.concatenate([cosnv, sinnv], axis=1)
                basis_cs = basis_cs[:, : (2 * (int(ntor) + 1))]
                out = _zeta_contract(w_stack, basis_cs)
                nsize = int(ntor) + 1
                out_cos = out[..., :nsize]
                out_sin = out[..., nsize:]
                out_cos_w = out_cos[:6]
                out_sin_w = out_sin[6:]

                frcc, fzsc, flsc = out_cos_w[0], out_cos_w[1], out_cos_w[2]
                out_cosnvn = out_cos_w[3:] * nfac
                out_sinnvn = -out_sin_w[:3] * nfac
                out_sinnv = out_sin_w[3:]
            else:
                w_cos_stack = jnp.stack([w1, w7, w11, w4, w6, w10], axis=0)
                out_cos = _zeta_contract(w_cos_stack, cosnv)
                frcc, fzsc, flsc = out_cos[0], out_cos[1], out_cos[2]
                out_cosnvn = out_cos[3:] * nfac

                w_sin_stack = jnp.stack([w2, w8, w12, w3, w5, w9], axis=0)
                out_sin = _zeta_contract(w_sin_stack, sinnv)
                out_sinnvn = -out_sin[:3] * nfac
                out_sinnv = out_sin[3:]

            frcc = frcc + out_sinnvn[0]
            fzsc = fzsc + out_sinnvn[1]
            flsc = flsc + out_sinnvn[2]

            frss = out_sinnv[0] + out_cosnvn[0]
            fzcs = out_sinnv[1] + out_cosnvn[1]
            flcs = out_sinnv[2] + out_cosnvn[2]
        else:
            w_cosnv = jnp.stack([w1, w7, w11], axis=0)
            out_cosnv = _zeta_contract(w_cosnv, cosnv)
            frcc, fzsc, flsc = out_cosnv[0], out_cosnv[1], out_cosnv[2]
            frss = None
            fzcs = None
            flcs = None

    # Apply VMEC's radial evolution masks (jmin2/jlam + fixed-boundary edge).
    # For parity work we use the default vmec_params values:
    #   jmin2(m=0)=1, jmin2(m>=1)=2; jlam(m)=2.
    (frcc, fzsc, frss, fzcs), (flsc, flcs) = _apply_tomnsp_radial_masks(
        masks=masks,
        ns=ns,
        mpol=mpol,
        include_edge=include_edge,
        rz_fields=(frcc, fzsc, frss, fzcs),
        l_fields=(flsc, flcs),
    )

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
    (
        armn_even, armn_odd, brmn_even, brmn_odd, crmn_even, crmn_odd,
        azmn_even, azmn_odd, bzmn_even, bzmn_odd, czmn_even, czmn_odd,
    ) = _slice_theta2_many(
        nt2,
        armn_even, armn_odd, brmn_even, brmn_odd, crmn_even, crmn_odd,
        azmn_even, azmn_odd, bzmn_even, bzmn_odd, czmn_even, czmn_odd,
    )
    (
        arcon_even, arcon_odd, azcon_even, azcon_odd,
        blmn_even, blmn_odd, clmn_even, clmn_odd,
    ) = _optional_theta2_pairs(
        nt2,
        armn_even,
        armn_odd,
        (arcon_even, arcon_odd),
        (azcon_even, azcon_odd),
        (blmn_even, blmn_odd),
        (clmn_even, clmn_odd),
    )

    cosmui, sinmui, cosmumi, sinmumi = _theta2_tables(trig, nt2, mpol)

    cosnv = trig.cosnv[:, : (ntor + 1)]  # (nzeta, ntor+1)
    sinnv = trig.sinnv[:, : (ntor + 1)]
    cosnvn = trig.cosnvn[:, : (ntor + 1)]
    sinnvn = trig.sinnvn[:, : (ntor + 1)]

    xmpq1 = _xmpq1_from_masks(masks, ns=ns, mpol=mpol, dtype_like=armn_even)

    # Theta integration work arrays (indices per tomnspa_par).
    # Base (present always):
    #   work1(3): frsc cosnv
    #   work1(5): fzcc cosnv
    #   work1(9): flcc cosnv
    armn, brmn, crmn, azmn, bzmn, czmn, blmn, clmn, arcon, azcon = _stack_even_odd_pairs(
        (armn_even, armn_odd),
        (brmn_even, brmn_odd),
        (crmn_even, crmn_odd),
        (azmn_even, azmn_odd),
        (bzmn_even, bzmn_odd),
        (czmn_even, czmn_odd),
        (blmn_even, blmn_odd),
        (clmn_even, clmn_odd),
        (arcon_even, arcon_odd),
        (azcon_even, azcon_odd),
    )

    stack_cosmui = jnp.stack([armn, crmn, azmn, czmn, arcon, azcon, clmn], axis=0)
    stack_sinmumi = jnp.stack([brmn, bzmn, blmn], axis=0)

    use_theta_fused = bool(_TOMNSPS_THETA_FUSED)
    if use_theta_fused:
        mpol_i = int(mpol)
        basis_cs = getattr(trig, "basis_theta_cs_nt2", None)
        if basis_cs is None or int(basis_cs.shape[0]) != nt2:
            basis_cs = jnp.concatenate([cosmui, sinmui], axis=1)
        basis_cs = basis_cs[:, : (2 * mpol_i)]
        out_cs = _theta_contract(stack_cosmui, basis_cs)
        cosmui_out = out_cs[..., :mpol_i, :]
        sinmui_out = out_cs[..., mpol_i:, :]

        basis_mu = getattr(trig, "basis_theta_mu_nt2", None)
        if basis_mu is None or int(basis_mu.shape[0]) != nt2:
            basis_mu = jnp.concatenate([sinmumi, cosmumi], axis=1)
        basis_mu = basis_mu[:, : (2 * mpol_i)]
        out_mu = _theta_contract(stack_sinmumi, basis_mu)
        sinmumi_out = out_mu[..., :mpol_i, :]
        cosmumi_out = out_mu[..., mpol_i:, :]
    else:
        cosmui_out = _theta_contract(stack_cosmui, cosmui)
        sinmui_out = _theta_contract(stack_cosmui, sinmui)
        sinmumi_out = _theta_contract(stack_sinmumi, sinmumi)
        cosmumi_out = _theta_contract(stack_sinmumi, cosmumi)

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

    mask_even = _mparity_from_masks(
        masks,
        ns=ns,
        mpol=mpol,
        include_edge=include_edge,
        dtype_like=armn_even,
    )
    w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12 = _select_mparity_pairs(
        mask_even, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12
    )

    lthreed = bool(ntor > 0)
    use_zeta_fused = bool(lthreed) and bool(_TOMNSPA_ZETA_FUSED)

    # Zeta integration.
    if use_zeta_fused:
        nsize = int(ntor) + 1
        w_stack = jnp.stack(
            [w3, w5, w9, w4, w6, w10, w1, w7, w11, w2, w8, w12],
            axis=0,
        )
        basis_all = getattr(trig, "basis_zeta_all", None)
        if basis_all is None or int(basis_all.shape[0]) != int(nzeta):
            basis_all = jnp.concatenate([cosnv, sinnvn, sinnv, cosnvn], axis=1)
        basis_all = basis_all[:, : (4 * (int(ntor) + 1))]
        out = _zeta_contract(w_stack, basis_all)
        out_cosnv = out[:3, ..., :nsize]
        out_sinnvn = out[3:6, ..., nsize : 2 * nsize]
        out_sinnv = out[6:9, ..., 2 * nsize : 3 * nsize]
        out_cosnvn = out[9:12, ..., 3 * nsize : 4 * nsize]

        frsc, fzcc, flcc = out_cosnv[0], out_cosnv[1], out_cosnv[2]
        frsc = frsc + out_sinnvn[0]
        fzcc = fzcc + out_sinnvn[1]
        flcc = flcc + out_sinnvn[2]

        frcs = out_sinnv[0] + out_cosnvn[0]
        fzss = out_sinnv[1] + out_cosnvn[1]
        flss = out_sinnv[2] + out_cosnvn[2]
    else:
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

    # VMEC `tomnspa` integrates over the restricted theta interval without
    # additional scaling for the asymmetric lambda blocks. Keep parity by
    # default; a manual override can be enabled via env if needed.
    if bool(lthreed):
        scale_env = os.getenv("VMEC_JAX_TOMNSPA_LAM_SCALE", "").strip().lower()
        if scale_env not in ("", "0", "false", "no", "1", "true", "yes"):
            try:
                scale_val = float(scale_env)
            except ValueError:
                scale_val = 1.0
        elif scale_env in ("1", "true", "yes"):
            scale_val = np.sqrt(2.0)
        else:
            scale_val = 1.0
        if scale_val != 1.0:
            flcc = flcc * jnp.asarray(scale_val, dtype=jnp.asarray(flcc).dtype)
            if flss is not None:
                flss = flss * jnp.asarray(scale_val, dtype=jnp.asarray(flss).dtype)

    # Apply radial evolution masks (same as tomnsps): fixed-boundary edge.
    (frsc, fzcc, frcs, fzss), (flcc, flss) = _apply_tomnsp_radial_masks(
        masks=masks,
        ns=ns,
        mpol=mpol,
        include_edge=include_edge,
        rz_fields=(frsc, fzcc, frcs, fzss),
        l_fields=(flcc, flss),
    )

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
