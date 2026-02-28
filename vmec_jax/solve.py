"""Fixed-boundary solvers.

The first solver milestone is a robust "inner solve" for the VMEC ``lambda`` field
with R/Z held fixed. This is useful for:

- validating the magnetic energy objective against VMEC2000 `wout` files,
- building toward a full fixed-boundary equilibrium solve.

Notes
-----
This module intentionally avoids optional dependencies (e.g. jaxopt). The current
implementation uses gradient descent with a simple backtracking line search.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import time
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, NamedTuple

import numpy as np

from ._compat import has_jax, jax, jnp, jit
from .field import TWOPI, b2_from_bsup, bsup_from_geom, bsup_from_sqrtg_lambda
from .fourier import eval_fourier_dtheta, eval_fourier_dzeta_phys
from .geom import eval_geom
from .grids import angle_steps
from .state import VMECState, pack_state, unpack_state


_SCAN_RUNNER_CACHE: dict[tuple, Any] = {}
_COMPUTE_FORCES_CACHE: dict[tuple, Any] = {}


def _hash_array_bytes(a: Any) -> str:
    arr = np.asarray(a)
    h = hashlib.blake2b(digest_size=16)
    h.update(arr.tobytes())
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    return h.hexdigest()


@dataclass(frozen=True)
class SolveLambdaResult:
    state: VMECState
    n_iter: int
    wb_history: np.ndarray
    grad_rms_history: np.ndarray
    step_history: np.ndarray
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class SolveFixedBoundaryResult:
    state: VMECState
    n_iter: int
    w_history: np.ndarray
    wb_history: np.ndarray
    wp_history: np.ndarray
    grad_rms_history: np.ndarray
    step_history: np.ndarray
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class SolveVmecResidualResult:
    state: VMECState
    n_iter: int
    w_history: np.ndarray
    fsqr2_history: np.ndarray
    fsqz2_history: np.ndarray
    fsql2_history: np.ndarray
    grad_rms_history: np.ndarray
    step_history: np.ndarray
    diagnostics: Dict[str, Any]


class _ScanCarry(NamedTuple):
    state: VMECState
    time_step: Any
    inv_tau: Any
    fsq_prev: Any
    fsq0_prev: Any
    accepted_count: Any
    probe_count: Any
    probe_bad_jac: Any
    probe_accept: Any
    probe_fsq_min: Any
    probe_fsq_max: Any
    probe_fsq_start: Any
    fallback_active: Any
    abort_scan: Any
    skip_timecontrol: Any
    vRcc: Any
    vRss: Any
    vZsc: Any
    vZcs: Any
    vLsc: Any
    vLcs: Any
    vRsc: Any
    vRcs: Any
    vZcc: Any
    vZss: Any
    vLcc: Any
    vLss: Any
    flip_sign: Any
    iter_offset: Any
    iter1: Any
    res0: Any
    res1: Any
    state_checkpoint: VMECState
    cache_valid: Any
    cache_precond_diag: Any
    cache_tcon: Any
    cache_norms: Any
    cache_rz_scale: Any
    cache_l_scale: Any
    cache_rz_norm: Any
    cache_f_norm1: Any
    cache_prec_rz_mats: Any
    cache_prec_lam_prec: Any
    force_bcovar_update: Any
    ijacob: Any
    bad_resets: Any
    bad_growth: Any
    fsqz_prev: Any
    r00_prev: Any
    z00_prev: Any
    w_mhd_prev: Any
    converged: Any
    fsqr_prev_phys: Any
    fsqz_prev_phys: Any
    fsql_prev_phys: Any
    fsqr1_prev: Any
    fsqz1_prev: Any
    fsql1_prev: Any
    fsqr_checkpoint: Any
    fsqz_checkpoint: Any
    fsql_checkpoint: Any
    fsqr1_checkpoint: Any
    fsqz1_checkpoint: Any
    fsql1_checkpoint: Any


def _parse_iter_list(val: str) -> set[int] | None:
    if not val:
        return None
    out: set[int] = set()
    for chunk in val.replace(" ", "").split(","):
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            try:
                lo = int(a)
                hi = int(b)
            except ValueError:
                continue
            if hi < lo:
                lo, hi = hi, lo
            out.update(range(lo, hi + 1))
        else:
            try:
                out.add(int(chunk))
            except ValueError:
                continue
    return out if out else None


def _maybe_dump_tomnsps(*, frzl, static, iter_idx: int, label: str = "raw") -> None:
    env = os.getenv("VMEC_JAX_DUMP_TOMNSPS", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns = int(static.cfg.ns)
    path = outdir / f"tomnsps_{label}_ns{ns}_iter{int(iter_idx)}.npz"

    def _arr(x):
        return np.asarray(x) if x is not None else np.zeros((0,), dtype=float)

    np.savez(
        path,
        frcc=_arr(frzl.frcc),
        frss=_arr(getattr(frzl, "frss", None)),
        fzsc=_arr(frzl.fzsc),
        fzcs=_arr(getattr(frzl, "fzcs", None)),
        flsc=_arr(frzl.flsc),
        flcs=_arr(getattr(frzl, "flcs", None)),
        frsc=_arr(getattr(frzl, "frsc", None)),
        frcs=_arr(getattr(frzl, "frcs", None)),
        fzcc=_arr(getattr(frzl, "fzcc", None)),
        fzss=_arr(getattr(frzl, "fzss", None)),
        flcc=_arr(getattr(frzl, "flcc", None)),
        flss=_arr(getattr(frzl, "flss", None)),
        ns=int(static.cfg.ns),
        mpol=int(static.cfg.mpol),
        ntor=int(static.cfg.ntor),
        lasym=bool(static.cfg.lasym),
    )


def _maybe_dump_force_kernels(*, k, static, iter_idx: int, label: str = "raw") -> None:
    env = os.getenv("VMEC_JAX_DUMP_FORCE_KERNELS", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns = int(static.cfg.ns)
    path = outdir / f"force_kernels_{label}_ns{ns}_iter{int(iter_idx)}.npz"

    def _arr(x):
        return np.asarray(x) if x is not None else np.zeros((0,), dtype=float)

    np.savez(
        path,
        armn_e=_arr(getattr(k, "armn_e", None)),
        armn_o=_arr(getattr(k, "armn_o", None)),
        brmn_e=_arr(getattr(k, "brmn_e", None)),
        brmn_o=_arr(getattr(k, "brmn_o", None)),
        crmn_e=_arr(getattr(k, "crmn_e", None)),
        crmn_o=_arr(getattr(k, "crmn_o", None)),
        azmn_e=_arr(getattr(k, "azmn_e", None)),
        azmn_o=_arr(getattr(k, "azmn_o", None)),
        bzmn_e=_arr(getattr(k, "bzmn_e", None)),
        bzmn_o=_arr(getattr(k, "bzmn_o", None)),
        czmn_e=_arr(getattr(k, "czmn_e", None)),
        czmn_o=_arr(getattr(k, "czmn_o", None)),
        arcon_e=_arr(getattr(k, "arcon_e", None)),
        arcon_o=_arr(getattr(k, "arcon_o", None)),
        azcon_e=_arr(getattr(k, "azcon_e", None)),
        azcon_o=_arr(getattr(k, "azcon_o", None)),
        gcon=_arr(getattr(k, "gcon", None)),
        tcon=_arr(getattr(k, "tcon", None)),
        blmn_e=_arr(getattr(getattr(k, "bc", None), "blmn_even", None)),
        blmn_o=_arr(getattr(getattr(k, "bc", None), "blmn_odd", None)),
        clmn_e=_arr(getattr(getattr(k, "bc", None), "clmn_even", None)),
        clmn_o=_arr(getattr(getattr(k, "bc", None), "clmn_odd", None)),
        bsubu_e=_arr(getattr(getattr(k, "bc", None), "bsubu_e", None)),
        bsubv_e=_arr(getattr(getattr(k, "bc", None), "bsubv_e", None)),
        bsubu=_arr(getattr(getattr(k, "bc", None), "bsubu", None)),
        bsubv=_arr(getattr(getattr(k, "bc", None), "bsubv", None)),
        bsupu=_arr(getattr(getattr(k, "bc", None), "bsupu", None)),
        bsupv=_arr(getattr(getattr(k, "bc", None), "bsupv", None)),
        guu_metric=_arr(getattr(getattr(k, "bc", None), "guu", None)),
        guv_metric=_arr(getattr(getattr(k, "bc", None), "guv", None)),
        gvv_metric=_arr(getattr(getattr(k, "bc", None), "gvv", None)),
        sqrtg=_arr(getattr(getattr(getattr(k, "bc", None), "jac", None), "sqrtg", None)),
        r12=_arr(getattr(getattr(getattr(k, "bc", None), "jac", None), "r12", None)),
        tau=_arr(getattr(getattr(getattr(k, "bc", None), "jac", None), "tau", None)),
        ru12=_arr(getattr(getattr(getattr(k, "bc", None), "jac", None), "ru12", None)),
        zu12=_arr(getattr(getattr(getattr(k, "bc", None), "jac", None), "zu12", None)),
        rs=_arr(getattr(getattr(getattr(k, "bc", None), "jac", None), "rs", None)),
        zs=_arr(getattr(getattr(getattr(k, "bc", None), "jac", None), "zs", None)),
        bsubu_e_scaled=_arr(
            getattr(getattr(k, "bc", None), "bsubu_e_scaled", None)
            if getattr(getattr(k, "bc", None), "bsubu_e_scaled", None) is not None
            else getattr(getattr(k, "bc", None), "clmn_even", None)
        ),
        bsubv_e_scaled=_arr(
            getattr(getattr(k, "bc", None), "bsubv_e_scaled", None)
            if getattr(getattr(k, "bc", None), "bsubv_e_scaled", None) is not None
            else getattr(getattr(k, "bc", None), "blmn_even", None)
        ),
        bsubu_tmp=_arr(getattr(getattr(k, "bc", None), "bsubu_tmp", None)),
        bsubv_preblend=_arr(getattr(getattr(k, "bc", None), "bsubv_preblend", None)),
        bsubv_avg=_arr(getattr(getattr(k, "bc", None), "bsubv_avg", None)),
        lamscale=_arr(getattr(getattr(k, "bc", None), "lamscale", None)),
        lu0_full=_arr(getattr(getattr(k, "bc", None), "lu0_full", None)),
        lu0_force=_arr(getattr(getattr(k, "bc", None), "lu0_force", None)),
        lu1_full=_arr(getattr(getattr(k, "bc", None), "lu1_full", None)),
        lvv=_arr(getattr(getattr(k, "bc", None), "lvv", None)),
        lvv_sh=_arr(getattr(getattr(k, "bc", None), "lvv_sh", None)),
        phip_full=_arr(getattr(getattr(k, "bc", None), "phip_full", None)),
        phip_internal=_arr(getattr(getattr(k, "bc", None), "phip_internal", None)),
        pr1_even=_arr(getattr(k, "pr1_even", None)),
        pr1_odd=_arr(getattr(k, "pr1_odd", None)),
        pz1_even=_arr(getattr(k, "pz1_even", None)),
        pz1_odd=_arr(getattr(k, "pz1_odd", None)),
        pru_even=_arr(getattr(k, "pru_even", None)),
        pru_odd=_arr(getattr(k, "pru_odd", None)),
        pzu_even=_arr(getattr(k, "pzu_even", None)),
        pzu_odd=_arr(getattr(k, "pzu_odd", None)),
        prv_even=_arr(getattr(k, "prv_even", None)),
        prv_odd=_arr(getattr(k, "prv_odd", None)),
        pzv_even=_arr(getattr(k, "pzv_even", None)),
        pzv_odd=_arr(getattr(k, "pzv_odd", None)),
        ns=int(static.cfg.ns),
        ntheta=int(static.cfg.ntheta),
        nzeta=int(static.cfg.nzeta),
        lasym=bool(static.cfg.lasym),
    )


def _maybe_dump_scalars(*, norms, iter_idx: int, ns: int) -> None:
    env = os.getenv("VMEC_JAX_DUMP_SCALARS", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"scalars_ns{int(ns)}_iter{int(iter_idx)}.dat"

    wb = float(np.asarray(getattr(norms, "wb", np.nan)))
    wp = float(np.asarray(getattr(norms, "wp", np.nan)))
    volume = float(np.asarray(getattr(norms, "volume", np.nan)))
    r2 = float(np.asarray(getattr(norms, "r2", np.nan)))
    fnorm = float(np.asarray(getattr(norms, "fnorm", np.nan)))
    fnormL = float(np.asarray(getattr(norms, "fnormL", np.nan)))
    fnorm1 = float("nan")
    with path.open("w") as f:
        f.write("# bcovar scalars dump\n")
        f.write("cols: iter wb wp vol r2 fnorm\n")
        f.write("      fn1 fnL\n")
        f.write(
            f"{int(iter_idx):6d}"
            f"{wb:24.16e}{wp:24.16e}{volume:24.16e}{r2:24.16e}"
            f"{fnorm:24.16e}{fnorm1:24.16e}{fnormL:24.16e}\n"
        )


def _maybe_dump_gcx2(*, gcr2, gcz2, gcl2, iter_idx: int, include_edge: bool, ns: int) -> None:
    env = os.getenv("VMEC_JAX_DUMP_GCX2", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"gcx2_ns{int(ns)}_iter{int(iter_idx)}.dat"
    with path.open("w") as f:
        f.write("# gcx2 dump (post-scalxc, post-m1)\n")
        f.write("columns: iter include_edge gcr2 gcz2 gcl2\n")
        f.write(
            f"{int(iter_idx):6d} {int(bool(include_edge)):3d}"
            f"{float(np.asarray(gcr2)):24.16e}"
            f"{float(np.asarray(gcz2)):24.16e}"
            f"{float(np.asarray(gcl2)):24.16e}\n"
        )


def _gc_from_frzl(*, frzl, cfg):
    frcc = np.asarray(frzl.frcc)
    ns, mpol, nrange = frcc.shape
    lthreed = bool(getattr(cfg, "lthreed", True))
    lasym = bool(getattr(cfg, "lasym", False))
    if lasym:
        ntmax = 4 if lthreed else 2
    else:
        ntmax = 2 if lthreed else 1

    gcr = np.zeros((ns, mpol, nrange, ntmax), dtype=frcc.dtype)
    gcz = np.zeros_like(gcr)
    gcl = np.zeros_like(gcr)

    gcr[:, :, :, 0] = frcc
    gcz[:, :, :, 0] = np.asarray(frzl.fzsc)
    gcl[:, :, :, 0] = np.asarray(frzl.flsc)

    if lasym:
        if lthreed:
            if frzl.frss is not None:
                gcr[:, :, :, 1] = np.asarray(frzl.frss)
            if frzl.fzcs is not None:
                gcz[:, :, :, 1] = np.asarray(frzl.fzcs)
            if frzl.flcs is not None:
                gcl[:, :, :, 1] = np.asarray(frzl.flcs)
            if getattr(frzl, "frsc", None) is not None:
                gcr[:, :, :, 2] = np.asarray(frzl.frsc)
            if getattr(frzl, "fzcc", None) is not None:
                gcz[:, :, :, 2] = np.asarray(frzl.fzcc)
            if getattr(frzl, "flcc", None) is not None:
                gcl[:, :, :, 2] = np.asarray(frzl.flcc)
            if getattr(frzl, "frcs", None) is not None:
                gcr[:, :, :, 3] = np.asarray(frzl.frcs)
            if getattr(frzl, "fzss", None) is not None:
                gcz[:, :, :, 3] = np.asarray(frzl.fzss)
            if getattr(frzl, "flss", None) is not None:
                gcl[:, :, :, 3] = np.asarray(frzl.flss)
        else:
            if getattr(frzl, "frsc", None) is not None:
                gcr[:, :, :, 1] = np.asarray(frzl.frsc)
            if getattr(frzl, "fzcc", None) is not None:
                gcz[:, :, :, 1] = np.asarray(frzl.fzcc)
            if getattr(frzl, "flcc", None) is not None:
                gcl[:, :, :, 1] = np.asarray(frzl.flcc)
    else:
        if lthreed:
            if frzl.frss is not None:
                gcr[:, :, :, 1] = np.asarray(frzl.frss)
            if frzl.fzcs is not None:
                gcz[:, :, :, 1] = np.asarray(frzl.fzcs)
            if frzl.flcs is not None:
                gcl[:, :, :, 1] = np.asarray(frzl.flcs)

    return gcr, gcz, gcl


def _maybe_dump_gc(*, frzl, static, iter_idx: int, label: str) -> None:
    env = os.getenv("VMEC_JAX_DUMP_GC", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_GC_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    stage = os.getenv("VMEC_JAX_DUMP_GC_STAGE", "precond").lower()
    if stage not in {"raw", "precond", "both"}:
        stage = "precond"
    if stage != "both" and stage != label:
        return

    outdir = Path(os.getenv("VMEC_JAX_DUMP_GC_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns = int(static.cfg.ns)
    path = outdir / f"gc_{label}_ns{ns}_iter{int(iter_idx)}.npz"
    gcr, gcz, gcl = _gc_from_frzl(frzl=frzl, cfg=static.cfg)
    np.savez(
        path,
        gcr=gcr,
        gcz=gcz,
        gcl=gcl,
        ns=int(static.cfg.ns),
        mpol=int(static.cfg.mpol),
        ntor=int(static.cfg.ntor),
        lthreed=bool(static.cfg.lthreed),
        lasym=bool(static.cfg.lasym),
    )


def _maybe_dump_lam_prec(*, lam_prec, faclam, static, iter_idx: int) -> None:
    env = os.getenv("VMEC_JAX_DUMP_LAM", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_LAM_ITER", ""))
    if iters is None:
        iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = os.getenv("VMEC_JAX_DUMP_LAM_DIR", "")
    if not outdir:
        outdir = os.getenv("VMEC_JAX_DUMP_DIR", ".")
    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns = int(static.cfg.ns)
    path = outdir / f"lam_prec_ns{ns}_iter{int(iter_idx)}.npz"
    lthreed = bool(static.cfg.lthreed)
    lasym = bool(static.cfg.lasym)
    ntmax = 4 if (lasym and lthreed) else (2 if lthreed else 1)
    lam_arr = np.asarray(lam_prec)
    if lam_arr.ndim != 3:
        raise ValueError(f"lam_prec expected 3D (ns,mpol,ntor+1), got {lam_arr.shape}")
    # VMEC dumps use (ns, n, m, t) with t=1..ntmax.
    pfaclam = np.zeros((ns, lam_arr.shape[2], lam_arr.shape[1], ntmax), dtype=lam_arr.dtype)
    pfaclam[:, :, :, 0] = np.transpose(lam_arr, (0, 2, 1))
    if ntmax > 1:
        pfaclam[:, :, :, 1:ntmax] = pfaclam[:, :, :, :1]
        # VMEC updates (m,n)=(0,0) only for t=1, leaving t>1 at zero.
        pfaclam[:, 0, 0, 1:ntmax] = 0.0
    data = {
        "pfaclam": pfaclam,
        "ns": ns,
        "mpol": int(static.cfg.mpol),
        "ntor": int(static.cfg.ntor),
        "lthreed": lthreed,
        "lasym": lasym,
    }
    if faclam is not None:
        fac_arr = np.asarray(faclam)
        faclam_out = np.zeros_like(pfaclam)
        if fac_arr.shape == lam_arr.shape:
            faclam_out[:, :, :, 0] = np.transpose(fac_arr, (0, 2, 1))
            if ntmax > 1:
                faclam_out[:, :, :, 1:ntmax] = faclam_out[:, :, :, :1]
                faclam_out[:, 0, 0, 1:ntmax] = 0.0
        else:
            faclam_out = fac_arr
        data["faclam"] = faclam_out
    np.savez(path, **data)


def _maybe_dump_lam_fsql1(*, fsql1_pre, fsql1_post, static, iter_idx: int) -> None:
    env = os.getenv("VMEC_JAX_DUMP_LAM", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_LAM_ITER", ""))
    if iters is None:
        iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = os.getenv("VMEC_JAX_DUMP_LAM_DIR", "")
    if not outdir:
        outdir = os.getenv("VMEC_JAX_DUMP_DIR", ".")
    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns = int(static.cfg.ns)
    path = outdir / f"lam_fsql1_ns{ns}_iter{int(iter_idx)}.dat"
    with path.open("w", encoding="utf-8") as f:
        f.write("# lambda fsql1 dump (pre/post faclam)\n")
        f.write("columns: iter fsql1_pre fsql1_post\n")
        f.write(f"{int(iter_idx):6d} {float(np.asarray(fsql1_pre)):24.16e} {float(np.asarray(fsql1_post)):24.16e}\n")


def _maybe_dump_lamcal(*, lam_debug: dict[str, np.ndarray], static, iter_idx: int) -> None:
    env = os.getenv("VMEC_JAX_DUMP_LAMCAL", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns = int(static.cfg.ns)
    path = outdir / f"lamcal_ns{ns}_iter{int(iter_idx)}.npz"
    np.savez(
        path,
        blam_pre=np.asarray(lam_debug.get("blam_pre")),
        clam_pre=np.asarray(lam_debug.get("clam_pre")),
        dlam_pre=np.asarray(lam_debug.get("dlam_pre")),
        blam_post=np.asarray(lam_debug.get("blam_post")),
        clam_post=np.asarray(lam_debug.get("clam_post")),
        dlam_post=np.asarray(lam_debug.get("dlam_post")),
    )


def _maybe_dump_lam_gcl(
    *,
    frzl_pre,
    frzl_post,
    static,
    iter_idx: int,
    delta_s,
) -> None:
    env = os.getenv("VMEC_JAX_DUMP_LAM", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_LAM_ITER", ""))
    if iters is None:
        iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = os.getenv("VMEC_JAX_DUMP_LAM_DIR", "")
    if not outdir:
        outdir = os.getenv("VMEC_JAX_DUMP_DIR", ".")
    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns = int(static.cfg.ns)
    path = outdir / f"lam_gcl_ns{ns}_iter{int(iter_idx)}.npz"

    _gcr_pre, _gcz_pre, gcl_pre = _gc_from_frzl(frzl=frzl_pre, cfg=static.cfg)
    _gcr_post, _gcz_post, gcl_post = _gc_from_frzl(frzl=frzl_post, cfg=static.cfg)

    gcl_pre = np.asarray(gcl_pre)
    gcl_post = np.asarray(gcl_post)
    delta_s_f = float(np.asarray(delta_s))
    # VMEC excludes the axis surface (js=1) from fsql1 sums.
    fsql1_pre = float(np.sum(gcl_pre[1:] * gcl_pre[1:]) * delta_s_f)
    fsql1_post = float(np.sum(gcl_post[1:] * gcl_post[1:]) * delta_s_f)

    _maybe_dump_lam_fsql1(
        fsql1_pre=fsql1_pre,
        fsql1_post=fsql1_post,
        static=static,
        iter_idx=int(iter_idx),
    )

    np.savez(
        path,
        gcl_pre=gcl_pre,
        gcl_post=gcl_post,
        fsql1_pre=fsql1_pre,
        fsql1_post=fsql1_post,
        delta_s=delta_s_f,
        ns=int(static.cfg.ns),
        mpol=int(static.cfg.mpol),
        ntor=int(static.cfg.ntor),
        lthreed=bool(static.cfg.lthreed),
        lasym=bool(static.cfg.lasym),
    )


_HLO_DUMPED_KEYS: set[tuple[str, int, int, int, int, int, bool]] = set()


def _maybe_dump_hlo_kernel(
    *,
    label: str,
    fn,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    static: Any,
    wout_like: Any,
    force: bool = False,
) -> None:
    env_dir = os.getenv("VMEC_JAX_DUMP_HLO_DIR", "").strip()
    if not env_dir:
        return
    env_all = os.getenv("VMEC_JAX_DUMP_HLO", "").strip().lower()
    enabled_all = env_all not in ("", "0", "false", "no")
    env_label = os.getenv(f"VMEC_JAX_DUMP_HLO_{label.upper()}", "").strip().lower()
    enabled_label = env_label not in ("", "0", "false", "no")
    if not force and not (enabled_all or enabled_label):
        return
    if not has_jax():
        return
    try:
        ns = int(getattr(static.cfg, "ns", 0))
        key = (
            str(label),
            ns,
            int(getattr(wout_like, "mpol", 0)),
            int(getattr(wout_like, "ntor", 0)),
            int(getattr(wout_like, "nfp", 0)),
            int(getattr(static.cfg, "ntheta", 0)),
            bool(getattr(wout_like, "lasym", False)),
        )
    except Exception:
        key = (str(label), 0, 0, 0, 0, 0, False)
    if key in _HLO_DUMPED_KEYS:
        return

    try:
        import jax
    except Exception:
        return

    outdir = Path(env_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    fname = f"hlo_{label}_ns{key[1]}_mpol{key[2]}_ntor{key[3]}.txt"
    outpath = outdir / fname

    hlo_text = None
    err_text = None
    try:
        jitted = jax.jit(fn)
        hlo = jitted.lower(*args, **kwargs).compiler_ir(dialect="hlo")
        if hasattr(hlo, "as_hlo_text"):
            hlo_text = hlo.as_hlo_text()
        elif hasattr(hlo, "as_text"):
            hlo_text = hlo.as_text()
        else:
            hlo_text = str(hlo)
    except Exception as exc:
        err_text = f"jit.lower failed: {exc!r}"
        try:
            hlo = jax.xla_computation(fn)(*args, **kwargs)
            if hasattr(hlo, "as_hlo_text"):
                hlo_text = hlo.as_hlo_text()
            else:
                hlo_text = str(hlo)
        except Exception as exc2:
            err_text = f"{err_text}\n xla_computation failed: {exc2!r}"
            hlo_text = None

    if hlo_text is None:
        if os.getenv("VMEC_JAX_DUMP_HLO_VERBOSE", "").strip().lower() not in ("", "0", "false", "no"):
            try:
                errpath = outdir / f"hlo_{label}_error_ns{key[1]}_mpol{key[2]}_ntor{key[3]}.txt"
                errpath.write_text(err_text or "unknown error")
            except Exception:
                pass
        return
    try:
        outpath.write_text(hlo_text)
        _HLO_DUMPED_KEYS.add(key)
    except Exception:
        return

def _maybe_dump_bsube(*, bc, static, iter_idx: int) -> None:
    env = os.getenv("VMEC_JAX_DUMP_BSUBE", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns = int(static.cfg.ns)
    path = outdir / f"bsube_ns{ns}_iter{int(iter_idx)}.dat"

    bsubu = np.asarray(bc.bsubu_e_scaled)
    bsubv = np.asarray(bc.bsubv_e_scaled)
    ns, ntheta, nzeta = bsubu.shape

    with path.open("w", encoding="utf-8") as f:
        f.write("# bcovar bsube dump (scaled)\n")
        f.write(f"ns={ns}\n")
        f.write(f"ntheta3={ntheta}\n")
        f.write(f"nzeta={nzeta}\n")
        f.write(f"lamscale={float(np.asarray(bc.lamscale)):.16e}\n")
        f.write("columns: js lt lz bsubu_e bsubv_e\n")
        for lt in range(ntheta):
            for lz in range(nzeta):
                for js in range(ns):
                    f.write(
                        f"{js + 1:6d}{lt + 1:6d}{lz + 1:6d}"
                        f"{bsubu[js, lt, lz]:24.16e}{bsubv[js, lt, lz]:24.16e}\n"
                    )


def _maybe_dump_bsube_terms(*, bc, static, iter_idx: int) -> None:
    env = os.getenv("VMEC_JAX_DUMP_BSUBE_TERMS", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns = int(static.cfg.ns)
    path = outdir / f"bsube_terms_ns{ns}_iter{int(iter_idx)}.dat"

    lvv_sh = np.asarray(getattr(bc, "lvv_sh"))
    lu0 = np.asarray(getattr(bc, "lu0_force"))
    lu1 = np.asarray(getattr(bc, "lu1_full"))
    phip = np.asarray(getattr(bc, "phip_internal"))
    bsubu_tmp = np.asarray(getattr(bc, "bsubu_tmp"))
    bsubv_pre = np.asarray(getattr(bc, "bsubv_preblend"))

    ns, ntheta, nzeta = lvv_sh.shape
    with path.open("w", encoding="utf-8") as f:
        f.write("# bcovar bsube terms dump\n")
        f.write(f"ns={ns}\n")
        f.write(f"ntheta3={ntheta}\n")
        f.write(f"nzeta={nzeta}\n")
        f.write("columns: js lt lz lvv_sh lu0 lu1 phipf bsubu_tmp bsubv_pre\n")
        for lt in range(ntheta):
            for lz in range(nzeta):
                for js in range(ns):
                    f.write(
                        f"{js + 1:6d}{lt + 1:6d}{lz + 1:6d}"
                        f"{lvv_sh[js, lt, lz]:24.16e}{lu0[js, lt, lz]:24.16e}{lu1[js, lt, lz]:24.16e}"
                    f"{phip[js]:24.16e}{bsubu_tmp[js, lt, lz]:24.16e}{bsubv_pre[js, lt, lz]:24.16e}\n"
                )


def _maybe_dump_bsubh(*, bc, static, iter_idx: int) -> None:
    env = os.getenv("VMEC_JAX_DUMP_BSUBH", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns = int(static.cfg.ns)
    path = outdir / f"bsubh_ns{ns}_iter{int(iter_idx)}.dat"

    bsubu = np.asarray(getattr(bc, "bsubu"))
    bsubv = np.asarray(getattr(bc, "bsubv"))

    ns, ntheta, nzeta = bsubu.shape
    with path.open("w", encoding="utf-8") as f:
        f.write("# bcovar bsubh dump (half mesh)\n")
        f.write(f"ns={ns}\n")
        f.write(f"ntheta3={ntheta}\n")
        f.write(f"nzeta={nzeta}\n")
        f.write("columns: js lt lz bsubuh bsubvh\n")
        for lt in range(ntheta):
            for lz in range(nzeta):
                for js in range(ns):
                    f.write(
                        f"{js + 1:6d}{lt + 1:6d}{lz + 1:6d}"
                        f"{bsubu[js, lt, lz]:24.16e}{bsubv[js, lt, lz]:24.16e}\n"
                    )


def _maybe_dump_bsubs(*, bc, state, static, trig, iter_idx: int, kernels=None) -> None:
    env = os.getenv("VMEC_JAX_DUMP_BSUBS", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns = int(static.cfg.ns)
    path = outdir / f"bsubs_ns{ns}_iter{int(iter_idx)}.npz"

    from .wout import _compute_bsubs_half_mesh, _vmec_symforce_apply

    s = np.asarray(static.s, dtype=float)
    bsupu = np.asarray(bc.bsupu)
    bsupv = np.asarray(bc.bsupv)
    force_rs = None
    force_zs = None
    force_ru12 = None
    force_zu12 = None
    _force_bss_env = os.getenv("VMEC_JAX_WOUT_FORCE_BSS", "").strip().lower()
    if _force_bss_env == "":
        use_force_bss = not bool(static.cfg.lasym)
    else:
        use_force_bss = _force_bss_env not in ("0", "false", "no")
    def _force_sym(arr, kind: str):
        arr_np = np.asarray(arr, dtype=float)
        if not bool(static.cfg.lasym):
            return arr_np
        return _vmec_symforce_apply(f=arr_np, trig=trig, kind=kind)
    if use_force_bss and kernels is not None:
        if hasattr(kernels, "crmn_e"):
            bsupu = _force_sym(getattr(kernels, "crmn_e"), "crs")
        if hasattr(kernels, "czmn_e"):
            bsupv = _force_sym(getattr(kernels, "czmn_e"), "czs")
        if hasattr(kernels, "bzmn_e"):
            force_rs = _force_sym(getattr(kernels, "bzmn_e"), "bzs")
        if hasattr(kernels, "brmn_e"):
            force_zs = _force_sym(getattr(kernels, "brmn_e"), "brs")
        if hasattr(kernels, "azmn_e"):
            force_ru12 = _force_sym(getattr(kernels, "azmn_e"), "azs")
        if hasattr(kernels, "armn_e"):
            force_zu12 = _force_sym(getattr(kernels, "armn_e"), "ars")
    bsubu = np.asarray(bc.bsubu)
    bsubv = np.asarray(bc.bsubv)
    sqrtg = np.asarray(bc.jac.sqrtg)

    geom_terms = {}
    if kernels is not None:
        for name in (
            "pr1_even",
            "pr1_odd",
            "pz1_even",
            "pz1_odd",
            "pru_even",
            "pru_odd",
            "pzu_even",
            "pzu_odd",
            "prv_even",
            "prv_odd",
            "pzv_even",
            "pzv_odd",
        ):
            if hasattr(kernels, name):
                geom_terms[name] = np.asarray(getattr(kernels, name), dtype=float)

    bsubs_half = _compute_bsubs_half_mesh(
        state=state,
        geom_modes=static.modes,
        s=s,
        lconm1=bool(getattr(static.cfg, "lconm1", True)),
        lthreed=bool(static.cfg.ntor > 0),
        lasym=bool(static.cfg.lasym),
        bsupu=bsupu,
        bsupv=bsupv,
        trig=trig,
        geom=geom_terms,
        jac_half=bc.jac,
        force_rs=force_rs,
        force_zs=force_zs,
        force_ru12=force_ru12,
        force_zu12=force_zu12,
    )
    bsubs_full = np.asarray(bsubs_half, dtype=float).copy()
    if ns > 2:
        bsubs_full[1:-1] = 0.5 * (bsubs_full[1:-1] + bsubs_full[2:])
    if ns > 0:
        bsubs_full[0] = 0.0
        bsubs_full[-1] = 0.0

    # JXBFORCE-style full-mesh bsupu/bsupv averages (for comparison with jxbout).
    bsupu1 = np.zeros_like(bsupu)
    bsupv1 = np.zeros_like(bsupv)
    if ns > 1:
        sqrtg_half = 0.5 * (sqrtg[1:] + sqrtg[:-1])
        denom = np.where(sqrtg_half != 0.0, sqrtg_half, 1.0)
        if ns > 2:
            # VMEC jxbforce: bsupu1(js) = 0.5*(bsupu(js)*gsqrt(js) + bsupu(js+1)*gsqrt(js+1)) / sqrtg_half
            bsupu1[1:-1] = 0.5 * (bsupu[1:-1] * sqrtg[1:-1] + bsupu[2:] * sqrtg[2:]) / denom[1:]
            bsupv1[1:-1] = 0.5 * (bsupv[1:-1] * sqrtg[1:-1] + bsupv[2:] * sqrtg[2:]) / denom[1:]
        bsupu1[0] = 0.0
        bsupu1[-1] = 0.0
        bsupv1[0] = 0.0
        bsupv1[-1] = 0.0

    np.savez(
        path,
        bsubs_half=np.asarray(bsubs_half, dtype=float),
        bsubs_full=np.asarray(bsubs_full, dtype=float),
        bsupu=np.asarray(bsupu, dtype=float),
        bsupv=np.asarray(bsupv, dtype=float),
        bsupu1=np.asarray(bsupu1, dtype=float),
        bsupv1=np.asarray(bsupv1, dtype=float),
        bsubu=np.asarray(bsubu, dtype=float),
        bsubv=np.asarray(bsubv, dtype=float),
        sqrtg=np.asarray(sqrtg, dtype=float),
        s=np.asarray(s, dtype=float),
    )

def _maybe_dump_lulv(
    *,
    bc,
    static,
    iter_idx: int,
    state: VMECState | None = None,
    trig: VmecTrigTables | None = None,
) -> None:
    env = os.getenv("VMEC_JAX_DUMP_LULV", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns = int(static.cfg.ns)
    path = outdir / f"lulv_ns{ns}_iter{int(iter_idx)}.npz"
    data = {
        "lu0_full": np.asarray(getattr(bc, "lu0_full")),
        "lu1_full": np.asarray(getattr(bc, "lu1_full")),
        "lv0_full": np.asarray(getattr(bc, "lv0_full")),
        "lv1_full": np.asarray(getattr(bc, "lv1_full")),
    }
    if state is not None:
        data["Lcos"] = np.asarray(state.Lcos)
        data["Lsin"] = np.asarray(state.Lsin)
        data["m_modes"] = np.asarray(static.modes.m, dtype=int)
        data["n_modes"] = np.asarray(static.modes.n, dtype=int)
        if trig is not None:
            # Debug lambda odd-m synthesis inputs (physical odd pieces).
            from .vmec_realspace import vmec_realspace_synthesis_dtheta, vmec_realspace_synthesis_dzeta_phys

            m_modes = np.asarray(static.modes.m, dtype=int)
            mask_m1 = (m_modes == 1).astype(np.asarray(state.Lsin).dtype)
            mask_odd_rest = ((m_modes % 2 == 1) & (m_modes != 1)).astype(np.asarray(state.Lsin).dtype)
            lu_m1 = vmec_realspace_synthesis_dtheta(
                coeff_cos=jnp.asarray(state.Lcos) * mask_m1,
                coeff_sin=jnp.asarray(state.Lsin) * mask_m1,
                modes=static.modes,
                trig=trig,
                coeffs_internal=True,
                apply_scalxc=True,
                s=static.s,
            )
            lu_rest = vmec_realspace_synthesis_dtheta(
                coeff_cos=jnp.asarray(state.Lcos) * mask_odd_rest,
                coeff_sin=jnp.asarray(state.Lsin) * mask_odd_rest,
                modes=static.modes,
                trig=trig,
                coeffs_internal=True,
                apply_scalxc=True,
                s=static.s,
            )
            lv_m1 = vmec_realspace_synthesis_dzeta_phys(
                coeff_cos=jnp.asarray(state.Lcos) * mask_m1,
                coeff_sin=jnp.asarray(state.Lsin) * mask_m1,
                modes=static.modes,
                trig=trig,
                coeffs_internal=True,
                apply_scalxc=True,
                s=static.s,
            )
            lv_rest = vmec_realspace_synthesis_dzeta_phys(
                coeff_cos=jnp.asarray(state.Lcos) * mask_odd_rest,
                coeff_sin=jnp.asarray(state.Lsin) * mask_odd_rest,
                modes=static.modes,
                trig=trig,
                coeffs_internal=True,
                apply_scalxc=True,
                s=static.s,
            )
            data["lu_phys_m1"] = np.asarray(lu_m1)
            data["lu_phys_rest"] = np.asarray(lu_rest)
            data["lv_phys_m1"] = np.asarray(lv_m1)
            data["lv_phys_rest"] = np.asarray(lv_rest)
    np.savez(path, **data)


def _maybe_dump_precond_inputs(*, bc, trig, static, iter_idx: int) -> None:
    env = os.getenv("VMEC_JAX_DUMP_PRECOND", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"precond_inputs_iter{int(iter_idx)}.dat"

    try:
        r12 = np.asarray(bc.jac.r12)
        sqrtg = np.asarray(bc.jac.sqrtg)
        bsq = np.asarray(bc.bsq)
        ru12 = np.asarray(bc.jac.ru12)
        zu12 = np.asarray(bc.jac.zu12)
    except Exception:
        return

    wint3 = getattr(trig, "wint3_precond", None) if trig is not None else None
    if wint3 is None:
        # Fallback to uniform weights if trig tables are missing.
        wint3 = np.ones((1, r12.shape[1], r12.shape[2]), dtype=float)
    wint3 = np.asarray(wint3)
    if wint3.ndim != 3:
        return
    if wint3.shape[0] == 1:
        wint_full = np.broadcast_to(wint3, r12.shape)
    elif wint3.shape[0] == r12.shape[0]:
        wint_full = wint3
    else:
        wint_full = np.broadcast_to(wint3[:1, :, :], r12.shape)

    ns = int(r12.shape[0])
    ntheta3 = int(r12.shape[1])
    nzeta = int(r12.shape[2])

    with path.open("w", encoding="utf-8") as f:
        f.write("# precond inputs (vmec_jax)\n")
        f.write(f"ns={ns}\n")
        f.write(f"nzeta={nzeta}\n")
        f.write(f"ntheta3={ntheta3}\n")
        f.write("columns: js lt lz\n")
        f.write("         r12 sqrtg bsq\n")
        f.write("         ru12 zu12 wint\n")
        for lt in range(ntheta3):
            for lz in range(nzeta):
                for j in range(1, ns):
                    f.write(
                        f"{j+1:6d}{lt+1:6d}{lz+1:6d}"
                        f"{r12[j, lt, lz]:24.16E}"
                        f"{sqrtg[j, lt, lz]:24.16E}"
                        f"{bsq[j, lt, lz]:24.16E}"
                        f"{ru12[j, lt, lz]:24.16E}"
                        f"{zu12[j, lt, lz]:24.16E}"
                        f"{wint_full[j, lt, lz]:24.16E}\n"
                    )

def _maybe_dump_xc(
    *,
    state: VMECState,
    vRcc,
    vRss,
    vZsc,
    vZcs,
    vLsc,
    vLcs,
    static,
    iter_idx: int,
) -> None:
    env = os.getenv("VMEC_JAX_DUMP_XC", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns_val = int(static.cfg.ns)
    path = outdir / f"xc_ns{ns_val}_iter{int(iter_idx)}.npz"
    from .diagnostics import vmec_internal_mn_from_state, vmec_xc_from_mn_blocks

    blocks = vmec_internal_mn_from_state(state, static, apply_basis_norm=False, apply_m1_constraint=False)
    xc_kwargs = dict(
        rcc=blocks["rcc"],
        rss=blocks["rss"],
        zsc=blocks["zsc"],
        zcs=blocks["zcs"],
        lsc=blocks["lsc"],
        lcs=blocks["lcs"],
    )
    if "rsc" in blocks:
        xc_kwargs.update(
            rsc=blocks.get("rsc"),
            rcs=blocks.get("rcs"),
            zcc=blocks.get("zcc"),
            zss=blocks.get("zss"),
            lcc=blocks.get("lcc"),
            lss=blocks.get("lss"),
        )
    xc = vmec_xc_from_mn_blocks(cfg=static.cfg, **xc_kwargs)

    xcdot_kwargs = dict(
        rcc=np.asarray(vRcc),
        rss=np.asarray(vRss),
        zsc=np.asarray(vZsc),
        zcs=np.asarray(vZcs),
        lsc=np.asarray(vLsc),
        lcs=np.asarray(vLcs),
    )
    # Asymmetric force channels are optional in dumps; default to zeros.
    xcdot = vmec_xc_from_mn_blocks(cfg=static.cfg, **xcdot_kwargs)
    np.savez(
        path,
        xc=np.asarray(xc),
        xcdot=np.asarray(xcdot),
        v=np.asarray(xcdot),
        ns=int(static.cfg.ns),
        mpol=int(static.cfg.mpol),
        ntor=int(static.cfg.ntor),
        lthreed=bool(static.cfg.lthreed),
        lasym=bool(static.cfg.lasym),
    )


def _mode00_index(modes) -> Optional[int]:
    m = np.asarray(modes.m)
    n = np.asarray(modes.n)
    idx = np.where((m == 0) & (n == 0))[0]
    if idx.size == 0:
        return None
    return int(idx[0])


def _enforce_lambda_gauge(Lcos, Lsin, *, idx00: Optional[int]):
    """Fix the (m,n)=(0,0) gauge mode to 0 (it is a nullspace)."""
    if idx00 is None:
        return Lcos, Lsin
    mask = jnp.asarray(np.arange(int(jnp.asarray(Lcos).shape[1])) == int(idx00))
    mask = mask[None, :]
    Lcos = jnp.where(mask, jnp.asarray(0.0, dtype=jnp.asarray(Lcos).dtype), jnp.asarray(Lcos))
    Lsin = jnp.where(mask, jnp.asarray(0.0, dtype=jnp.asarray(Lsin).dtype), jnp.asarray(Lsin))
    return Lcos, Lsin


def _axis_m0_mask(static, *, dtype):
    if getattr(static, "m_is_m0", None) is not None:
        return jnp.asarray(static.m_is_m0, dtype=dtype)
    m = jnp.asarray(static.modes.m)
    return (m == 0).astype(dtype)


def _enforce_fixed_boundary_and_axis(
    state: VMECState,
    static,
    *,
    edge_Rcos,
    edge_Rsin,
    edge_Zcos,
    edge_Zsin,
    enforce_axis: bool = True,
    enforce_edge: bool = True,
    enforce_lambda_axis: bool = True,
    idx00: Optional[int],
) -> VMECState:
    """Apply minimal VMEC regularity + fixed-boundary constraints.

    - Fix R/Z at the outer surface (s=1) to preserve the prescribed boundary.
    - Enforce axis regularity by zeroing all m>0 Fourier coefficients at s=0.
    - Enforce lambda gauge (m,n)=(0,0) = 0 everywhere.
    """
    Rcos = jnp.asarray(state.Rcos)
    Rsin = jnp.asarray(state.Rsin)
    Zcos = jnp.asarray(state.Zcos)
    Zsin = jnp.asarray(state.Zsin)
    Lcos = jnp.asarray(state.Lcos)
    Lsin = jnp.asarray(state.Lsin)

    if enforce_edge:
        Rcos = jnp.concatenate([Rcos[:-1, :], jnp.asarray(edge_Rcos)[None, :]], axis=0)
        Rsin = jnp.concatenate([Rsin[:-1, :], jnp.asarray(edge_Rsin)[None, :]], axis=0)
        Zcos = jnp.concatenate([Zcos[:-1, :], jnp.asarray(edge_Zcos)[None, :]], axis=0)
        Zsin = jnp.concatenate([Zsin[:-1, :], jnp.asarray(edge_Zsin)[None, :]], axis=0)

    if enforce_axis:
        mask_m0 = _axis_m0_mask(static, dtype=Rcos.dtype)
        Rcos = jnp.concatenate([Rcos[:1, :] * mask_m0[None, :], Rcos[1:, :]], axis=0)
        Rsin = jnp.concatenate([Rsin[:1, :] * mask_m0[None, :], Rsin[1:, :]], axis=0)
        Zcos = jnp.concatenate([Zcos[:1, :] * mask_m0[None, :], Zcos[1:, :]], axis=0)
        Zsin = jnp.concatenate([Zsin[:1, :] * mask_m0[None, :], Zsin[1:, :]], axis=0)

    if enforce_lambda_axis:
        Lcos = jnp.concatenate([jnp.zeros_like(Lcos[:1, :]), Lcos[1:, :]], axis=0)
        Lsin = jnp.concatenate([jnp.zeros_like(Lsin[:1, :]), Lsin[1:, :]], axis=0)

    Lcos, Lsin = _enforce_lambda_gauge(Lcos, Lsin, idx00=idx00)

    return VMECState(
        layout=state.layout,
        Rcos=Rcos,
        Rsin=Rsin,
        Zcos=Zcos,
        Zsin=Zsin,
        Lcos=Lcos,
        Lsin=Lsin,
    )


def _grad_rms_state(grad: VMECState) -> float:
    g = np.asarray(grad.Rcos) ** 2
    g = g + np.asarray(grad.Rsin) ** 2
    g = g + np.asarray(grad.Zcos) ** 2
    g = g + np.asarray(grad.Zsin) ** 2
    g = g + np.asarray(grad.Lcos) ** 2
    g = g + np.asarray(grad.Lsin) ** 2
    return float(np.sqrt(np.mean(g)))


def _update_state_gd(state: VMECState, grad: VMECState, *, step: float, scale_rz: float, scale_l: float) -> VMECState:
    step = jnp.asarray(step, dtype=jnp.asarray(state.Rcos).dtype)
    scale_rz = jnp.asarray(scale_rz, dtype=step.dtype)
    scale_l = jnp.asarray(scale_l, dtype=step.dtype)
    return VMECState(
        layout=state.layout,
        Rcos=jnp.asarray(state.Rcos) - step * scale_rz * jnp.asarray(grad.Rcos),
        Rsin=jnp.asarray(state.Rsin) - step * scale_rz * jnp.asarray(grad.Rsin),
        Zcos=jnp.asarray(state.Zcos) - step * scale_rz * jnp.asarray(grad.Zcos),
        Zsin=jnp.asarray(state.Zsin) - step * scale_rz * jnp.asarray(grad.Zsin),
        Lcos=jnp.asarray(state.Lcos) - step * scale_l * jnp.asarray(grad.Lcos),
        Lsin=jnp.asarray(state.Lsin) - step * scale_l * jnp.asarray(grad.Lsin),
    )


def _mask_grad_for_constraints(
    grad: VMECState,
    static,
    *,
    idx00: Optional[int],
    mask_lambda_axis: bool = True,
) -> VMECState:
    """Project gradients onto the feasible set implied by our constraints."""
    gRcos = jnp.asarray(grad.Rcos)
    gRsin = jnp.asarray(grad.Rsin)
    gZcos = jnp.asarray(grad.Zcos)
    gZsin = jnp.asarray(grad.Zsin)
    gLcos = jnp.asarray(grad.Lcos)
    gLsin = jnp.asarray(grad.Lsin)

    # Fixed-boundary: don't update the edge surface for R/Z.
    gRcos = gRcos.at[-1, :].set(0.0)
    gRsin = gRsin.at[-1, :].set(0.0)
    gZcos = gZcos.at[-1, :].set(0.0)
    gZsin = gZsin.at[-1, :].set(0.0)

    # Axis regularity: don't update m>0 coefficients at s=0 for R/Z.
    m = jnp.asarray(static.modes.m)
    mask_m0 = (m == 0).astype(gRcos.dtype)
    gRcos = gRcos.at[0, :].set(gRcos[0, :] * mask_m0)
    gRsin = gRsin.at[0, :].set(gRsin[0, :] * mask_m0)
    gZcos = gZcos.at[0, :].set(gZcos[0, :] * mask_m0)
    gZsin = gZsin.at[0, :].set(gZsin[0, :] * mask_m0)

    # Lambda: optionally fix the axis row.
    if bool(mask_lambda_axis):
        gLcos = gLcos.at[0, :].set(0.0)
        gLsin = gLsin.at[0, :].set(0.0)

    # Lambda gauge: (m,n)=(0,0) stays 0 everywhere.
    if idx00 is not None:
        gLcos = gLcos.at[:, idx00].set(0.0)
        gLsin = gLsin.at[:, idx00].set(0.0)

    return VMECState(
        layout=grad.layout,
        Rcos=gRcos,
        Rsin=gRsin,
        Zcos=gZcos,
        Zsin=gZsin,
        Lcos=gLcos,
        Lsin=gLsin,
    )


def _apply_preconditioner(
    grad: VMECState,
    static,
    *,
    kind: str,
    exponent: float = 1.0,
    radial_alpha: float = 0.0,
) -> VMECState:
    """Apply a simple diagonal preconditioner in (m,n) Fourier space.

    Parameters
    ----------
    kind:
        - ``"none"``: no preconditioning
        - ``"mode_diag"``: scale each (m,n) mode by ~(m^2 + (n*NFP)^2)^(-exponent)
        - ``"radial_tridi"``: apply a simple Dirichlet tri-diagonal smoother in s
        - ``"mode_diag+radial_tridi"``: apply both (order: mode, then radial)
    """
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

    def _tridi_smooth_dirichlet(rhs, *, alpha: float):
        """Solve a simple tri-diagonal smoothing system along s for each mode.

        This applies a Dirichlet-boundary operator in s:

            (-α) x_{i-1} + (1+2α) x_i + (-α) x_{i+1} = rhs_i

        on interior points i=1..ns-2, treating x_0 and x_{ns-1} as fixed to rhs
        at those endpoints. This preserves any constraint-masked gradients at
        the endpoints while still coupling interior surfaces.
        """
        rhs = jnp.asarray(rhs)
        if rhs.ndim < 2:
            raise ValueError(f"expected (ns,...) with ndim>=2, got {rhs.shape}")
        ns = int(rhs.shape[0])
        if rhs.ndim == 2:
            rhs2 = rhs
            orig_shape = None
        else:
            rhs2 = rhs.reshape(ns, -1)
            orig_shape = rhs.shape
        ns = int(rhs2.shape[0])
        if ns < 3:
            return rhs
        alpha = jnp.asarray(alpha, dtype=rhs.dtype)
        a = -alpha
        b = 1.0 + 2.0 * alpha
        c = -alpha

        x0 = rhs2[0]
        xN = rhs2[-1]
        d = rhs2[1:-1]
        d = d.at[0].add(alpha * x0)
        d = d.at[-1].add(alpha * xN)

        n = int(d.shape[0])
        if n == 1:
            x_int = d / b
        else:
            # Forward sweep (Thomas algorithm), vectorized over modes K.
            cp0 = c / b
            dp0 = d[0] / b

            def fwd(carry, di):
                cp_prev, dp_prev = carry
                denom = b - a * cp_prev
                cp = c / denom
                dp = (di - a * dp_prev) / denom
                return (cp, dp), (cp, dp)

            (cp_last, dp_last), (cp_rest, dp_rest) = jax.lax.scan(fwd, (cp0, dp0), d[1:])
            cp = jnp.concatenate([jnp.asarray([cp0]), cp_rest], axis=0)
            dp = jnp.concatenate([dp0[None, :], dp_rest], axis=0)
            # Back substitution.
            x_last = dp_last

            def bwd(x_next, items):
                cpi, dpi = items
                xi = dpi - cpi * x_next
                return xi, xi

            _x0, x_rev = jax.lax.scan(bwd, x_last, (cp[:-1], dp[:-1]), reverse=True)
            x_int = jnp.concatenate([x_rev, x_last[None, :]], axis=0)

        return jnp.concatenate([x0[None, :], x_int, xN[None, :]], axis=0)

    def _apply_radial_tridi(g: VMECState) -> VMECState:
        return VMECState(
            layout=g.layout,
            Rcos=_tridi_smooth_dirichlet(g.Rcos, alpha=radial_alpha),
            Rsin=_tridi_smooth_dirichlet(g.Rsin, alpha=radial_alpha),
            Zcos=_tridi_smooth_dirichlet(g.Zcos, alpha=radial_alpha),
            Zsin=_tridi_smooth_dirichlet(g.Zsin, alpha=radial_alpha),
            Lcos=_tridi_smooth_dirichlet(g.Lcos, alpha=radial_alpha),
            Lsin=_tridi_smooth_dirichlet(g.Lsin, alpha=radial_alpha),
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


def solve_lambda_gd(
    state0: VMECState,
    static,
    *,
    phipf,
    chipf,
    signgs: int,
    lamscale,
    sqrtg: Any | None = None,
    max_iter: int = 50,
    step_size: float = 0.05,
    grad_tol: float = 1e-10,
    max_backtracks: int = 16,
    bt_factor: float = 0.5,
    jit_grad: bool = False,
    preconditioner: str = "none",
    precond_exponent: float = 1.0,
    precond_radial_alpha: float = 0.0,
    verbose: bool = True,
) -> SolveLambdaResult:
    """Solve for VMEC lambda (scaled coefficients) with fixed R/Z.

    Parameters
    ----------
    state0:
        Initial state. Only the lambda coefficients are updated.
    static:
        VMECStatic from :func:`vmec_jax.static.build_static`.
    phipf, chipf:
        1D flux functions (ns,) matching VMEC's `wout` meaning.
    signgs:
        Orientation (+1 or -1).
    lamscale:
        VMEC lambda scaling factor (see :func:`vmec_jax.field.lamscale_from_phips`).
    sqrtg:
        Optional signed Jacobian on the 3D grid. If provided (e.g. reconstructed from
        `wout` Nyquist coefficients), it is used for the objective and field formulas.
        Otherwise we use :func:`vmec_jax.geom.eval_geom`'s sqrtg.
    """
    if not has_jax():
        raise ImportError("solve_lambda_gd requires JAX (jax + jaxlib)")

    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if max_backtracks < 0:
        raise ValueError("max_backtracks must be >= 0")
    if not (0.0 < bt_factor < 1.0):
        raise ValueError("bt_factor must be in (0, 1)")

    idx00 = _mode00_index(static.modes)
    preconditioner = str(preconditioner).strip().lower()
    if preconditioner not in ("none", "mode_diag"):
        raise ValueError(f"Unknown preconditioner kind={preconditioner!r}")
    precond_exponent = float(precond_exponent)
    if preconditioner != "none" and precond_exponent <= 0.0:
        raise ValueError("precond_exponent must be > 0 when using a preconditioner")

    # Metric depends only on R/Z, so compute it once.
    g0 = eval_geom(state0, static)
    gtt = jnp.asarray(g0.g_tt)
    gtp = jnp.asarray(g0.g_tp)
    gpp = jnp.asarray(g0.g_pp)

    sqrtg_use = jnp.asarray(g0.sqrtg if sqrtg is None else sqrtg)

    phipf = jnp.asarray(phipf)
    chipf = jnp.asarray(chipf)
    lamscale = jnp.asarray(lamscale)
    signgs = int(signgs)
    nfp = int(static.cfg.nfp)

    s = jnp.asarray(static.s)
    dtype_state = jnp.asarray(state0.Rcos).dtype
    zero_precond_diag = (
        jnp.zeros((int(s.shape[0]),), dtype=dtype_state),
        jnp.zeros((int(s.shape[0]),), dtype=dtype_state),
    )
    zero_tcon = jnp.zeros((int(s.shape[0]),), dtype=dtype_state)
    constraint_active_false = jnp.asarray(False)
    constraint_active_false = jnp.asarray(False)
    constraint_active_false = jnp.asarray(False)
    theta = jnp.asarray(static.grid.theta)
    zeta = jnp.asarray(static.grid.zeta)
    if s.shape[0] < 2:
        ds = jnp.asarray(1.0, dtype=s.dtype)
    else:
        ds = s[1] - s[0]
    dtheta_f, dzeta_f = angle_steps(ntheta=int(theta.shape[0]), nzeta=int(zeta.shape[0]))
    dtheta = jnp.asarray(dtheta_f, dtype=s.dtype)
    dzeta = jnp.asarray(dzeta_f, dtype=s.dtype)
    weight = ds * dtheta * dzeta

    def _wb_from_L(Lcos, Lsin):
        lam_u = eval_fourier_dtheta(Lcos, Lsin, static.basis, coeffs_internal=True)
        lam_v = eval_fourier_dzeta_phys(Lcos, Lsin, static.basis, coeffs_internal=True) / nfp
        bsupu, bsupv = bsup_from_sqrtg_lambda(
            sqrtg=sqrtg_use,
            lam_u=lam_u,
            lam_v=lam_v,
            phipf=phipf,
            chipf=chipf,
            signgs=signgs,
            lamscale=lamscale,
        )
        B2 = gtt * bsupu**2 + 2.0 * gtp * bsupu * bsupv + gpp * bsupv**2
        jac = signgs * sqrtg_use
        E_total = jnp.sum(0.5 * B2 * jac) * weight
        return E_total / (TWOPI * TWOPI)

    wb_and_grad = jax.value_and_grad(_wb_from_L, argnums=(0, 1))
    wb_only = _wb_from_L
    if jit_grad:
        wb_and_grad = jit(wb_and_grad)
        wb_only = jit(wb_only)

    Lcos = jnp.asarray(state0.Lcos)
    Lsin = jnp.asarray(state0.Lsin)
    Lcos, Lsin = _enforce_lambda_gauge(Lcos, Lsin, idx00=idx00)

    wb0, (gcos, gsin) = wb_and_grad(Lcos, Lsin)
    wb_history = [float(np.asarray(wb0))]
    grad_rms_history = []
    step_history = []

    for it in range(max_iter):
        # Optional mode-diagonal preconditioning for the lambda subproblem.
        if preconditioner == "mode_diag":
            m = jnp.asarray(static.modes.m)
            n = jnp.asarray(static.modes.n)
            k2 = m.astype(jnp.float64) ** 2 + (n.astype(jnp.float64) * float(static.cfg.nfp)) ** 2
            w = (1.0 + k2) ** (-precond_exponent)
            w = w.astype(jnp.asarray(Lcos).dtype)
            gcos_p = gcos * w[None, :]
            gsin_p = gsin * w[None, :]
        else:
            gcos_p = gcos
            gsin_p = gsin

        grad_rms = float(np.sqrt(np.mean(np.asarray(gcos_p) ** 2 + np.asarray(gsin_p) ** 2)))
        grad_rms_history.append(grad_rms)

        if verbose:
            print(f"[solve_lambda_gd] iter={it:03d} wb={wb_history[-1]:.8e} grad_rms={grad_rms:.3e}")

        if grad_rms < grad_tol:
            break

        step = float(step_size)
        accepted = False

        for bt in range(max_backtracks + 1):
            if bt > 0:
                step *= bt_factor
            Lcos_t = Lcos - step * gcos_p
            Lsin_t = Lsin - step * gsin_p
            Lcos_t, Lsin_t = _enforce_lambda_gauge(Lcos_t, Lsin_t, idx00=idx00)
            wb_t = wb_only(Lcos_t, Lsin_t)
            if float(np.asarray(wb_t)) < wb_history[-1]:
                accepted = True
                Lcos, Lsin, wb0 = Lcos_t, Lsin_t, wb_t
                break

        step_history.append(step)

        if not accepted:
            if verbose:
                print("[solve_lambda_gd] line search failed to improve objective; stopping")
            break

        wb_history.append(float(np.asarray(wb0)))
        wb0, (gcos, gsin) = wb_and_grad(Lcos, Lsin)

    st = VMECState(
        layout=state0.layout,
        Rcos=state0.Rcos,
        Rsin=state0.Rsin,
        Zcos=state0.Zcos,
        Zsin=state0.Zsin,
        Lcos=Lcos,
        Lsin=Lsin,
    )
    diag: Dict[str, Any] = {"idx00": idx00}
    return SolveLambdaResult(
        state=st,
        n_iter=len(wb_history) - 1,
        wb_history=np.asarray(wb_history, dtype=float),
        grad_rms_history=np.asarray(grad_rms_history, dtype=float),
        step_history=np.asarray(step_history, dtype=float),
        diagnostics=diag,
    )


def solve_fixed_boundary_gd(
    state0: VMECState,
    static,
    *,
    phipf,
    chipf,
    signgs: int,
    lamscale,
    edge_Rcos: Any | None = None,
    edge_Rsin: Any | None = None,
    edge_Zcos: Any | None = None,
    edge_Zsin: Any | None = None,
    pressure: Any | None = None,
    gamma: float = 0.0,
    jacobian_penalty: float = 1e3,
    max_iter: int = 25,
    step_size: float = 5e-3,
    scale_rz: float = 1.0,
    scale_l: float = 1.0,
    grad_tol: float = 1e-10,
    max_backtracks: int = 16,
    bt_factor: float = 0.5,
    jit_grad: bool = False,
    preconditioner: str = "none",
    precond_exponent: float = 1.0,
    precond_radial_alpha: float = 0.0,
    differentiable: bool = False,
    stop_grad_in_update: bool = False,
    verbose: bool = True,
) -> SolveFixedBoundaryResult:
    """Minimize a VMEC-style energy objective over (R,Z,lambda) coefficients.

    This is the first "full" fixed-boundary solver step:
    - R/Z are evolved on interior surfaces only; the outer surface is held fixed.
    - Lambda gauge mode (0,0) is fixed to 0.

    The objective is::

        W = wb + wp/(gamma - 1)

    where ``wb`` is VMEC's normalized magnetic energy and
    ``wp = ∫ p dV /(2π)^2``.
    A soft penalty enforces a consistent Jacobian sign away from the axis.
    """
    if not has_jax():
        raise ImportError("solve_fixed_boundary_gd requires JAX (jax + jaxlib)")

    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if max_backtracks < 0:
        raise ValueError("max_backtracks must be >= 0")
    if not (0.0 < bt_factor < 1.0):
        raise ValueError("bt_factor must be in (0, 1)")

    gamma = float(gamma)
    if abs(gamma - 1.0) < 1e-14:
        raise ValueError("gamma=1 makes wp/(gamma-1) singular")

    idx00 = _mode00_index(static.modes)

    phipf = jnp.asarray(phipf)
    chipf = jnp.asarray(chipf)
    lamscale = jnp.asarray(lamscale)
    signgs = int(signgs)
    nfp = int(static.cfg.nfp)

    s = jnp.asarray(static.s)
    theta = jnp.asarray(static.grid.theta)
    zeta = jnp.asarray(static.grid.zeta)
    if s.shape[0] < 2:
        ds = jnp.asarray(1.0, dtype=s.dtype)
    else:
        ds = s[1] - s[0]
    dtheta_f, dzeta_f = angle_steps(ntheta=int(theta.shape[0]), nzeta=int(zeta.shape[0]))
    dtheta = jnp.asarray(dtheta_f, dtype=s.dtype)
    dzeta = jnp.asarray(dzeta_f, dtype=s.dtype)
    weight = ds * dtheta * dzeta

    if pressure is None:
        pressure = jnp.zeros_like(s)
    pressure = jnp.asarray(pressure)
    if pressure.shape != s.shape:
        raise ValueError(f"pressure must have shape {s.shape}, got {pressure.shape}")

    edge_Rcos = jnp.asarray(edge_Rcos) if edge_Rcos is not None else jnp.asarray(state0.Rcos)[-1, :]
    edge_Rsin = jnp.asarray(edge_Rsin) if edge_Rsin is not None else jnp.asarray(state0.Rsin)[-1, :]
    edge_Zcos = jnp.asarray(edge_Zcos) if edge_Zcos is not None else jnp.asarray(state0.Zcos)[-1, :]
    edge_Zsin = jnp.asarray(edge_Zsin) if edge_Zsin is not None else jnp.asarray(state0.Zsin)[-1, :]

    def _wb_wp_from_geom(g) -> Tuple[Any, Any]:
        bsupu, bsupv = bsup_from_geom(g, phipf=phipf, chipf=chipf, nfp=nfp, signgs=signgs, lamscale=lamscale)
        B2 = b2_from_bsup(g, bsupu, bsupv)
        jac = signgs * g.sqrtg
        wb = (jnp.sum(0.5 * B2 * jac) * weight) / (TWOPI * TWOPI)
        wp = (jnp.sum(pressure[:, None, None] * jac) * weight) / (TWOPI * TWOPI)
        return wb, wp

    def _w_total_from_wb_wp(wb, wp) -> Any:
        return wb + wp / (gamma - 1.0)

    def _objective(state: VMECState) -> Any:
        # Softly enforce a consistent Jacobian sign away from the axis (s=0).
        g = eval_geom(state, static)
        wb, wp = _wb_wp_from_geom(g)
        w = _w_total_from_wb_wp(wb, wp)
        jac = signgs * g.sqrtg
        jac = jac.at[0, :, :].set(0.0)
        neg = jnp.minimum(jac, 0.0)
        penalty = float(jacobian_penalty) * jnp.mean(neg * neg)
        return w + penalty

    def _w_terms(state: VMECState) -> Tuple[Any, Any, Any]:
        g = eval_geom(state, static)
        wb, wp = _wb_wp_from_geom(g)
        return wb, wp, _w_total_from_wb_wp(wb, wp)

    obj_and_grad = jax.value_and_grad(_objective)
    w_terms = _w_terms
    if jit_grad:
        obj_and_grad = jit(obj_and_grad)
        w_terms = jit(w_terms)

    # Start from a constraint-satisfying state.
    state = _enforce_fixed_boundary_and_axis(
        state0,
        static,
        edge_Rcos=edge_Rcos,
        edge_Rsin=edge_Rsin,
        edge_Zcos=edge_Zcos,
        edge_Zsin=edge_Zsin,
        enforce_lambda_axis=False,
        idx00=idx00,
    )

    if differentiable:
        wb_history = []
        wp_history = []
        w_history = []
        grad_rms_history = []
        step_history = []

        def _grad_rms_jax(grad_state: VMECState):
            g = (
                jnp.asarray(grad_state.Rcos) ** 2
                + jnp.asarray(grad_state.Rsin) ** 2
                + jnp.asarray(grad_state.Zcos) ** 2
                + jnp.asarray(grad_state.Zsin) ** 2
                + jnp.asarray(grad_state.Lcos) ** 2
                + jnp.asarray(grad_state.Lsin) ** 2
            )
            return jnp.sqrt(jnp.mean(g))

        for _ in range(max_iter):
            wb_t, wp_t, w_t = w_terms(state)
            w_history.append(w_t)
            wb_history.append(wb_t)
            wp_history.append(wp_t)

            obj_t, grad_t = obj_and_grad(state)
            grad_t = _mask_grad_for_constraints(grad_t, static, idx00=idx00)
            grad_t = _apply_preconditioner(
                grad_t,
                static,
                kind=preconditioner,
                exponent=precond_exponent,
                radial_alpha=precond_radial_alpha,
            )
            if stop_grad_in_update:
                grad_t = jax.lax.stop_gradient(grad_t)
            grad_rms_history.append(_grad_rms_jax(grad_t))
            step_history.append(jnp.asarray(step_size, dtype=jnp.asarray(state.Rcos).dtype))

            state = _update_state_gd(state, grad_t, step=step_size, scale_rz=scale_rz, scale_l=scale_l)
            state = _enforce_fixed_boundary_and_axis(
                state,
                static,
                edge_Rcos=edge_Rcos,
                edge_Rsin=edge_Rsin,
                edge_Zcos=edge_Zcos,
                edge_Zsin=edge_Zsin,
                idx00=idx00,
            )
    else:
        wb0, wp0, w0 = w_terms(state)
        wb0 = float(np.asarray(wb0))
        wp0 = float(np.asarray(wp0))
        w0 = float(np.asarray(w0))
        wb_history = [wb0]
        wp_history = [wp0]
        grad_rms_history = []
        step_history = []

        obj0, grad0 = obj_and_grad(state)
        obj0 = float(np.asarray(obj0))
        w_history = [obj0]

        for it in range(max_iter):
            grad0m = _mask_grad_for_constraints(grad0, static, idx00=idx00)
            grad_raw = grad0m
            grad0m = _apply_preconditioner(
                grad0m,
                static,
                kind=preconditioner,
                exponent=precond_exponent,
                radial_alpha=precond_radial_alpha,
            )
            grad_rms = _grad_rms_state(grad0m)
            grad_rms_history.append(grad_rms)

            if verbose:
                print(f"[solve_fixed_boundary_gd] iter={it:03d} w={w_history[-1]:.8e} grad_rms={grad_rms:.3e}")

            if grad_rms < grad_tol:
                break

            step = float(step_size)
            accepted = False

            def _try_line_search(grad_step):
                step_local = float(step_size)
                for bt in range(max_backtracks + 1):
                    if bt > 0:
                        step_local *= bt_factor
                    trial = _update_state_gd(state, grad_step, step=step_local, scale_rz=scale_rz, scale_l=scale_l)
                    trial = _enforce_fixed_boundary_and_axis(
                        trial,
                        static,
                        edge_Rcos=edge_Rcos,
                        edge_Rsin=edge_Rsin,
                        edge_Zcos=edge_Zcos,
                        edge_Zsin=edge_Zsin,
                        idx00=idx00,
                    )
                    obj_t = _objective(trial)
                    obj_t = float(np.asarray(obj_t))
                    if np.isfinite(obj_t) and obj_t < w_history[-1]:
                        return True, trial, obj_t, step_local
                return False, None, None, step_local

            accepted, trial, obj_t, step = _try_line_search(grad0m)
            if not accepted and preconditioner != "none":
                accepted, trial, obj_t, step = _try_line_search(grad_raw)
                if accepted and verbose:
                    print("[solve_fixed_boundary_gd] fallback to unpreconditioned gradient")

            step_history.append(step)

            if not accepted:
                if verbose:
                    print("[solve_fixed_boundary_gd] line search failed to improve objective; stopping")
                break

            state = trial
            obj0 = obj_t

            wb_t, wp_t, _w_t = w_terms(state)
            w_history.append(obj0)
            wb_history.append(float(np.asarray(wb_t)))
            wp_history.append(float(np.asarray(wp_t)))

            obj0, grad0 = obj_and_grad(state)

    diag: Dict[str, Any] = {
        "idx00": idx00,
        "signgs": signgs,
        "gamma": gamma,
        "jacobian_penalty": float(jacobian_penalty),
        "scale_rz": float(scale_rz),
        "scale_l": float(scale_l),
        "preconditioner": str(preconditioner),
        "precond_exponent": float(precond_exponent),
        "precond_radial_alpha": float(precond_radial_alpha),
    }
    if differentiable:
        return SolveFixedBoundaryResult(
            state=state,
            n_iter=len(w_history),
            w_history=jnp.asarray(w_history),
            wb_history=jnp.asarray(wb_history),
            wp_history=jnp.asarray(wp_history),
            grad_rms_history=jnp.asarray(grad_rms_history),
            step_history=jnp.asarray(step_history),
            diagnostics=diag,
        )
    return SolveFixedBoundaryResult(
        state=state,
        n_iter=len(w_history) - 1,
        w_history=np.asarray(w_history, dtype=float),
        wb_history=np.asarray(wb_history, dtype=float),
        wp_history=np.asarray(wp_history, dtype=float),
        grad_rms_history=np.asarray(grad_rms_history, dtype=float),
        step_history=np.asarray(step_history, dtype=float),
        diagnostics=diag,
    )


def solve_fixed_boundary_lbfgs(
    state0: VMECState,
    static,
    *,
    phipf,
    chipf,
    signgs: int,
    lamscale,
    edge_Rcos: Any | None = None,
    edge_Rsin: Any | None = None,
    edge_Zcos: Any | None = None,
    edge_Zsin: Any | None = None,
    pressure: Any | None = None,
    gamma: float = 0.0,
    history_size: int = 10,
    max_iter: int = 40,
    step_size: float = 1.0,
    grad_tol: float = 1e-10,
    max_backtracks: int = 12,
    bt_factor: float = 0.5,
    jit_grad: bool = False,
    preconditioner: str = "none",
    precond_exponent: float = 1.0,
    precond_radial_alpha: float = 0.0,
    verbose: bool = True,
) -> SolveFixedBoundaryResult:
    """Fixed-boundary solve using L-BFGS (no external deps).

    This solver minimizes::

        W = wb + wp/(gamma - 1)

    with:

    - fixed R/Z edge coefficients (prescribed boundary),
    - simple axis regularity,
    - lambda gauge (0,0)=0.
    """
    if not has_jax():
        raise ImportError("solve_fixed_boundary_lbfgs requires JAX (jax + jaxlib)")

    history_size = int(history_size)
    if history_size < 1:
        raise ValueError("history_size must be >= 1")
    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if max_backtracks < 0:
        raise ValueError("max_backtracks must be >= 0")
    if not (0.0 < bt_factor < 1.0):
        raise ValueError("bt_factor must be in (0, 1)")

    gamma = float(gamma)
    if abs(gamma - 1.0) < 1e-14:
        raise ValueError("gamma=1 makes wp/(gamma-1) singular")

    idx00 = _mode00_index(static.modes)

    phipf = jnp.asarray(phipf)
    chipf = jnp.asarray(chipf)
    lamscale = jnp.asarray(lamscale)
    signgs = int(signgs)
    nfp = int(static.cfg.nfp)

    s = jnp.asarray(static.s)
    theta = jnp.asarray(static.grid.theta)
    zeta = jnp.asarray(static.grid.zeta)
    if s.shape[0] < 2:
        ds = jnp.asarray(1.0, dtype=s.dtype)
    else:
        ds = s[1] - s[0]
    dtheta_f, dzeta_f = angle_steps(ntheta=int(theta.shape[0]), nzeta=int(zeta.shape[0]))
    dtheta = jnp.asarray(dtheta_f, dtype=s.dtype)
    dzeta = jnp.asarray(dzeta_f, dtype=s.dtype)
    weight = ds * dtheta * dzeta

    if pressure is None:
        pressure = jnp.zeros_like(s)
    pressure = jnp.asarray(pressure)
    if pressure.shape != s.shape:
        raise ValueError(f"pressure must have shape {s.shape}, got {pressure.shape}")

    edge_Rcos = jnp.asarray(edge_Rcos) if edge_Rcos is not None else jnp.asarray(state0.Rcos)[-1, :]
    edge_Rsin = jnp.asarray(edge_Rsin) if edge_Rsin is not None else jnp.asarray(state0.Rsin)[-1, :]
    edge_Zcos = jnp.asarray(edge_Zcos) if edge_Zcos is not None else jnp.asarray(state0.Zcos)[-1, :]
    edge_Zsin = jnp.asarray(edge_Zsin) if edge_Zsin is not None else jnp.asarray(state0.Zsin)[-1, :]

    def _wb_wp_from_geom(g) -> Tuple[Any, Any]:
        bsupu, bsupv = bsup_from_geom(g, phipf=phipf, chipf=chipf, nfp=nfp, signgs=signgs, lamscale=lamscale)
        B2 = b2_from_bsup(g, bsupu, bsupv)
        jac = signgs * g.sqrtg
        wb = (jnp.sum(0.5 * B2 * jac) * weight) / (TWOPI * TWOPI)
        wp = (jnp.sum(pressure[:, None, None] * jac) * weight) / (TWOPI * TWOPI)
        return wb, wp

    def _w_total_from_wb_wp(wb, wp) -> Any:
        return wb + wp / (gamma - 1.0)

    def _w_only(state: VMECState) -> Any:
        g = eval_geom(state, static)
        wb, wp = _wb_wp_from_geom(g)
        return _w_total_from_wb_wp(wb, wp)

    def _w_terms_and_jacmin(state: VMECState) -> Tuple[Any, Any, Any, Any]:
        g = eval_geom(state, static)
        wb, wp = _wb_wp_from_geom(g)
        w = _w_total_from_wb_wp(wb, wp)
        jac = signgs * g.sqrtg
        if jac.shape[0] <= 1:
            jac_min = jnp.min(jac)
        else:
            jac_min = jnp.min(jac[1:, :, :])
        return wb, wp, w, jac_min

    w_and_grad = jax.value_and_grad(_w_only)
    w_terms = _w_terms_and_jacmin
    if jit_grad:
        w_and_grad = jit(w_and_grad)
        w_terms = jit(w_terms)

    def _lbfgs_direction(g_flat, s_hist, y_hist):
        if not s_hist:
            return -g_flat
        q = g_flat
        alpha = []
        rho = []
        for s_i, y_i in zip(reversed(s_hist), reversed(y_hist)):
            ys = jnp.dot(y_i, s_i)
            rho_i = jnp.where(ys != 0, 1.0 / ys, 0.0)
            a_i = rho_i * jnp.dot(s_i, q)
            q = q - a_i * y_i
            alpha.append(a_i)
            rho.append(rho_i)

        # Initial inverse-Hessian scaling (common L-BFGS choice)
        s0 = s_hist[-1]
        y0 = y_hist[-1]
        ys0 = jnp.dot(y0, s0)
        yy0 = jnp.dot(y0, y0)
        gamma0 = jnp.where(yy0 != 0, ys0 / yy0, 1.0)
        r = gamma0 * q

        for s_i, y_i, a_i, rho_i in zip(s_hist, y_hist, reversed(alpha), reversed(rho)):
            beta = rho_i * jnp.dot(y_i, r)
            r = r + s_i * (a_i - beta)

        return -r

    # Start from a constraint-satisfying state.
    state = _enforce_fixed_boundary_and_axis(
        state0,
        static,
        edge_Rcos=edge_Rcos,
        edge_Rsin=edge_Rsin,
        edge_Zcos=edge_Zcos,
        edge_Zsin=edge_Zsin,
        enforce_lambda_axis=False,
        idx00=idx00,
    )

    wb0, wp0, w0, jacmin0 = w_terms(state)
    w0 = float(np.asarray(w0))
    wb0 = float(np.asarray(wb0))
    wp0 = float(np.asarray(wp0))
    jacmin0 = float(np.asarray(jacmin0))
    if not np.isfinite(w0) or jacmin0 <= 0.0:
        raise ValueError("Initial state has invalid Jacobian sign or non-finite energy")

    w_history = [w0]
    wb_history = [wb0]
    wp_history = [wp0]
    grad_rms_history = []
    step_history = []

    w_val, grad = w_and_grad(state)
    grad = _mask_grad_for_constraints(grad, static, idx00=idx00, mask_lambda_axis=False)
    grad = _apply_preconditioner(
        grad,
        static,
        kind=preconditioner,
        exponent=precond_exponent,
        radial_alpha=precond_radial_alpha,
    )

    x = pack_state(state)
    g_flat = pack_state(grad)

    s_hist: list[Any] = []
    y_hist: list[Any] = []

    step0 = float(step_size)

    for it in range(max_iter):
        grad_rms = _grad_rms_state(grad)
        grad_rms_history.append(grad_rms)

        if verbose:
            print(f"[solve_fixed_boundary_lbfgs] iter={it:03d} w={w_history[-1]:.8e} grad_rms={grad_rms:.3e}")

        if grad_rms < grad_tol:
            break

        p_flat = _lbfgs_direction(g_flat, s_hist, y_hist)
        # Ensure descent direction; otherwise fall back to steepest descent.
        gtp = float(np.asarray(jnp.dot(g_flat, p_flat)))
        if not np.isfinite(gtp) or gtp >= 0.0:
            p_flat = -g_flat

        accepted = False
        step = step0

        x_old = x
        g_old = g_flat

        for bt in range(max_backtracks + 1):
            if bt > 0:
                step *= bt_factor
            x_try = x_old + jnp.asarray(step, dtype=x_old.dtype) * p_flat
            st_try = unpack_state(x_try, state.layout)
            st_try = _enforce_fixed_boundary_and_axis(
                st_try,
                static,
                edge_Rcos=edge_Rcos,
                edge_Rsin=edge_Rsin,
                edge_Zcos=edge_Zcos,
                edge_Zsin=edge_Zsin,
                enforce_lambda_axis=False,
                idx00=idx00,
            )

            wb_t, wp_t, w_t, jacmin_t = w_terms(st_try)
            w_tf = float(np.asarray(w_t))
            jacmin_tf = float(np.asarray(jacmin_t))
            if np.isfinite(w_tf) and jacmin_tf > 0.0 and w_tf < w_history[-1]:
                state = st_try
                x = pack_state(state)
                accepted = True
                break

        step_history.append(step)

        if not accepted:
            if verbose:
                print("[solve_fixed_boundary_lbfgs] line search failed; stopping")
            break

        # New value/grad at accepted state.
        wb_t, wp_t, w_t, _jacmin_t = w_terms(state)
        w_history.append(float(np.asarray(w_t)))
        wb_history.append(float(np.asarray(wb_t)))
        wp_history.append(float(np.asarray(wp_t)))

        w_val, grad_new = w_and_grad(state)
        grad_new = _mask_grad_for_constraints(grad_new, static, idx00=idx00)
        grad_new = _apply_preconditioner(
            grad_new,
            static,
            kind=preconditioner,
            exponent=precond_exponent,
            radial_alpha=precond_radial_alpha,
        )
        g_flat_new = pack_state(grad_new)

        s_k = x - x_old
        y_k = g_flat_new - g_old
        ys = float(np.asarray(jnp.dot(y_k, s_k)))
        if np.isfinite(ys) and ys > 1e-14:
            s_hist.append(s_k)
            y_hist.append(y_k)
            if len(s_hist) > history_size:
                s_hist.pop(0)
                y_hist.pop(0)

        grad = grad_new
        g_flat = g_flat_new
        step0 = float(step)

    diag: Dict[str, Any] = {
        "idx00": idx00,
        "signgs": signgs,
        "gamma": gamma,
        "history_size": int(history_size),
        "preconditioner": str(preconditioner),
        "precond_exponent": float(precond_exponent),
        "precond_radial_alpha": float(precond_radial_alpha),
    }
    return SolveFixedBoundaryResult(
        state=state,
        n_iter=len(w_history) - 1,
        w_history=np.asarray(w_history, dtype=float),
        wb_history=np.asarray(wb_history, dtype=float),
        wp_history=np.asarray(wp_history, dtype=float),
        grad_rms_history=np.asarray(grad_rms_history, dtype=float),
        step_history=np.asarray(step_history, dtype=float),
        diagnostics=diag,
    )


@dataclass(frozen=True)
class _WoutLikeVmecForces:
    """Minimal `wout`-like container for VMEC force/residual kernels."""

    nfp: int
    mpol: int
    ntor: int
    lasym: bool
    signgs: int

    phipf: Any  # (ns,)
    phips: Any  # (ns,)
    chipf: Any  # (ns,)  (VMEC `wout` half-mesh averaged convention)
    pres: Any  # (ns,)  (half mesh, VMEC internal units mu0*Pa)
    mass: Any | None = None  # (ns,) mass profile on half mesh (VMEC internal units)
    gamma: float | None = None
    ncurr: int = 0
    lcurrent: bool = True
    icurv: Any | None = None  # (ns,) integrated toroidal current profile
    flux_is_internal: bool = True


def _s_half_from_full_mesh_s(s):
    s = jnp.asarray(s)
    if int(s.shape[0]) < 2:
        return s
    return jnp.concatenate([s[:1], 0.5 * (s[1:] + s[:-1])], axis=0)


def _half_mesh_from_full_mesh(x):
    x = jnp.asarray(x)
    if int(x.shape[0]) < 2:
        return x
    return jnp.concatenate([x[:1], 0.5 * (x[1:] + x[:-1])], axis=0)


def _pressure_half_mesh_from_indata(*, indata, s_full):
    from .profiles import eval_profiles

    s_half = _s_half_from_full_mesh_s(s_full)
    prof = eval_profiles(indata, s_half)
    return jnp.asarray(prof.get("pressure", jnp.zeros_like(s_half)))


def _mass_half_mesh_from_indata(*, indata, s_full, phips, r00, gamma, lrfp: bool = False, chips=None):
    """Compute VMEC mass profile on half mesh: mass = pmass * (|vnorm|*r00)^gamma."""
    from .profiles import eval_profiles

    s_half = _s_half_from_full_mesh_s(s_full)
    prof = eval_profiles(indata, s_half)
    pmass = jnp.asarray(prof.get("pressure", jnp.zeros_like(s_half)))
    vnorm = jnp.asarray(phips)
    if lrfp and (chips is not None):
        vnorm = jnp.asarray(chips)
    mass = pmass * (jnp.abs(vnorm) * jnp.asarray(r00, dtype=pmass.dtype)) ** jnp.asarray(gamma, dtype=pmass.dtype)
    if int(mass.shape[0]) > 0:
        mass = mass.at[0].set(jnp.asarray(0.0, dtype=mass.dtype))
    return mass


def _icurv_full_mesh_from_indata(*, indata, s_full, signgs: int):
    from .profiles import eval_profiles

    s_full = jnp.asarray(s_full)
    ncurr = int(indata.get_int("NCURR", 0))
    if ncurr != 1:
        return jnp.zeros_like(s_full)

    curtor = float(indata.get_float("CURTOR", 0.0))
    if abs(curtor) <= np.finfo(float).eps:
        return jnp.zeros_like(s_full)

    # VMEC stores icurv on the half mesh (same indexing as phips/chips/iotas),
    # evaluated at s = (i-1.5)*hs for i>=2. Mirror that here.
    s_half = _s_half_from_full_mesh_s(s_full)
    prof = eval_profiles(indata, s_half)
    icurv_raw = jnp.asarray(prof.get("current", jnp.zeros_like(s_half)))
    if int(icurv_raw.shape[0]) != int(s_full.shape[0]):
        icurv_raw = jnp.zeros_like(s_half)

    # VMEC scales by pcurr(1) (edge), not the last half-mesh value.
    pedge_prof = eval_profiles(indata, jnp.asarray([1.0], dtype=s_full.dtype))
    pedge = float(np.asarray(pedge_prof.get("current", jnp.asarray([0.0], dtype=s_full.dtype)))[0])
    if abs(pedge) <= abs(np.finfo(float).eps * curtor):
        return jnp.zeros_like(s_full)

    mu0 = 4e-7 * np.pi
    currv = mu0 * curtor
    scale = float(signgs) * currv / (2.0 * np.pi * pedge)
    icurv = jnp.asarray(scale, dtype=icurv_raw.dtype) * icurv_raw
    if int(icurv.shape[0]) > 0:
        icurv = icurv.at[0].set(0.0)
    return icurv


def solve_fixed_boundary_lbfgs_vmec_residual(
    state0: VMECState,
    static,
    *,
    indata,
    signgs: int,
    w_rz: float = 1.0,
    w_l: float = 1.0,
    include_constraint_force: bool = True,
    objective_scale: float | None = None,
    apply_m1_constraints: bool = True,
    history_size: int = 10,
    max_iter: int = 40,
    step_size: float = 1.0,
    scale_rz: float = 1.0,
    scale_l: float = 1.0,
    grad_tol: float = 1e-10,
    max_backtracks: int = 12,
    bt_factor: float = 0.5,
    jit_grad: bool = False,
    preconditioner: str = "none",
    precond_exponent: float = 1.0,
    precond_radial_alpha: float = 0.0,
    verbose: bool = True,
) -> SolveVmecResidualResult:
    """Fixed-boundary solve by minimizing a VMEC-style force-residual objective.

    The objective follows the parity pipeline
    ``bcovar -> forces -> tomnsps -> sum-of-squares of Fourier residual blocks``,
    using VMEC's ``getfsq`` conventions (post-``tomnsps`` ``scalxc`` scaling,
    optional converged-iteration m=1 constraints, and R/Z edge exclusion).

    For parity, build ``static`` with ``vmec_angle_grid(...)`` (see
    ``vmec_jax.vmec_tomnsp``). This solver does not include VMEC's
    iteration-dependent switching logic (e.g. ``lforbal`` triggering); it
    provides a differentiable objective suitable for regression and initial
    end-to-end parity.

    """
    if not has_jax():
        raise ImportError("solve_fixed_boundary_lbfgs_vmec_residual requires JAX (jax + jaxlib)")

    w_rz = float(w_rz)
    w_l = float(w_l)
    if w_rz < 0.0 or w_l < 0.0:
        raise ValueError("w_rz and w_l must be nonnegative")
    if objective_scale is not None and float(objective_scale) <= 0.0:
        raise ValueError("objective_scale must be positive when provided")
    scale_rz = float(scale_rz)
    scale_l = float(scale_l)
    if scale_rz <= 0.0 or scale_l <= 0.0:
        raise ValueError("scale_rz and scale_l must be positive")

    history_size = int(history_size)
    if history_size < 1:
        raise ValueError("history_size must be >= 1")
    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if max_backtracks < 0:
        raise ValueError("max_backtracks must be >= 0")
    if not (0.0 < bt_factor < 1.0):
        raise ValueError("bt_factor must be in (0, 1)")

    idx00 = _mode00_index(static.modes)
    signgs = int(signgs)

    from .energy import flux_profiles_from_indata
    from .static import build_static
    from .vmec_forces import vmec_forces_rz_from_wout, vmec_residual_internal_from_kernels
    from .vmec_residue import (
        vmec_force_norms_from_bcovar_dynamic,
        vmec_gcx2_from_tomnsps,
        vmec_zero_m1_zforce,
    )
    from .vmec_tomnsp import vmec_trig_tables

    s = jnp.asarray(static.s)

    flux = flux_profiles_from_indata(indata, s, signgs=signgs)
    chipf_wout = jnp.asarray(flux.chipf)

    phips = jnp.asarray(flux.phips)
    if phips.shape[0] >= 1:
        phips = phips.at[0].set(0.0)

    # VMEC mass profile uses the boundary r00 coefficient and phips scaling.
    from .boundary import boundary_from_indata

    boundary = boundary_from_indata(indata, static.modes)
    r00 = float(np.asarray(boundary.R_cos)[int(idx00)]) if int(idx00) >= 0 else float(np.asarray(boundary.R_cos)[0])
    gamma = float(indata.get_float("GAMMA", 5.0 / 3.0))
    lrfp = bool(indata.get_bool("LRFP", False))
    chips = _half_mesh_from_full_mesh(chipf_wout) if lrfp else None
    mass = _mass_half_mesh_from_indata(
        indata=indata,
        s_full=s,
        phips=phips,
        r00=r00,
        gamma=gamma,
        lrfp=lrfp,
        chips=chips,
    )

    pres = _pressure_half_mesh_from_indata(indata=indata, s_full=s)
    ncurr = int(indata.get_int("NCURR", 0))
    icurv = _icurv_full_mesh_from_indata(indata=indata, s_full=s, signgs=signgs)

    wout_like = _WoutLikeVmecForces(
        nfp=int(static.cfg.nfp),
        mpol=int(static.cfg.mpol),
        ntor=int(static.cfg.ntor),
        lasym=bool(static.cfg.lasym),
        signgs=signgs,
        phipf=jnp.asarray(flux.phipf),
        phips=phips,
        chipf=chipf_wout,
        pres=pres,
        mass=mass,
        gamma=gamma,
        ncurr=ncurr,
        lcurrent=True,
        icurv=icurv,
    )

    trig = getattr(static, "trig_vmec", None)
    if trig is None:
        trig = vmec_trig_tables(
            ntheta=int(static.cfg.ntheta),
            nzeta=int(static.cfg.nzeta),
            nfp=int(wout_like.nfp),
            mmax=int(wout_like.mpol) - 1,
            nmax=int(wout_like.ntor),
            lasym=bool(wout_like.lasym),
            dtype=jnp.asarray(state0.Rcos).dtype,
        )
    objective_scale_f = float(objective_scale) if objective_scale is not None else None

    constraint_tcon0: float | None = None
    if bool(include_constraint_force):
        # VMEC2000 default is `TCON0=1` (readin.f).
        constraint_tcon0 = float(indata.get_float("TCON0", 1.0))
    apply_lforbal = bool(indata.get_bool("LFORBAL", False)) if indata is not None else False
    apply_lforbal = bool(indata.get_bool("LFORBAL", False)) if indata is not None else False
    mask_pack = getattr(static, "tomnsps_masks", None)

    def _fsq2_terms_and_jacmin(state: VMECState, zero_m1_zforce: Any):
        k = vmec_forces_rz_from_wout(
            state=state,
            static=static,
            wout=wout_like,
            indata=None,
            constraint_tcon0=constraint_tcon0,
            use_vmec_synthesis=True,
            trig=trig,
        )
        rzl = vmec_residual_internal_from_kernels(
            k,
            cfg_ntheta=int(static.cfg.ntheta),
            cfg_nzeta=int(static.cfg.nzeta),
            wout=wout_like,
            trig=trig,
            apply_lforbal=apply_lforbal,
            include_edge=False,
            masks=mask_pack,
        )
        rzl = vmec_zero_m1_zforce(frzl=rzl, enabled=zero_m1_zforce)
        gcr2, gcz2, gcl2 = vmec_gcx2_from_tomnsps(
            frzl=rzl,
            lconm1=bool(getattr(static.cfg, "lconm1", True)),
            apply_m1_constraints=bool(apply_m1_constraints),
            include_edge=False,
            apply_scalxc=True,
            s=s,
        )
        norms = vmec_force_norms_from_bcovar_dynamic(bc=k.bc, trig=trig, s=s, signgs=signgs)
        fsqr2 = norms.r1 * norms.fnorm * gcr2
        fsqz2 = norms.r1 * norms.fnorm * gcz2
        fsql2 = norms.fnormL * gcl2

        w = (w_rz * (fsqr2 + fsqz2)) + (w_l * fsql2)
        if objective_scale_f is not None:
            w = jnp.asarray(objective_scale_f, dtype=jnp.asarray(w).dtype) * w

        jac = signgs * jnp.asarray(k.bc.jac.sqrtg)
        jac_min = jnp.min(jac) if jac.shape[0] <= 1 else jnp.min(jac[1:, :, :])
        return fsqr2, fsqz2, fsql2, w, jac_min

    def _w_only(state: VMECState, zero_m1_zforce: Any):
        return _fsq2_terms_and_jacmin(state, zero_m1_zforce)[3]

    w_and_grad = jax.value_and_grad(_w_only)
    w_terms = _fsq2_terms_and_jacmin
    if jit_grad:
        w_and_grad = jit(w_and_grad)
        w_terms = jit(w_terms)

    edge_Rcos = jnp.asarray(state0.Rcos)[-1, :]
    edge_Rsin = jnp.asarray(state0.Rsin)[-1, :]
    edge_Zcos = jnp.asarray(state0.Zcos)[-1, :]
    edge_Zsin = jnp.asarray(state0.Zsin)[-1, :]

    state = _enforce_fixed_boundary_and_axis(
        state0,
        static,
        edge_Rcos=edge_Rcos,
        edge_Rsin=edge_Rsin,
        edge_Zcos=edge_Zcos,
        edge_Zsin=edge_Zsin,
        idx00=idx00,
    )

    zero_m1 = jnp.asarray(1.0, dtype=jnp.asarray(state0.Rcos).dtype)
    fsqr2_0, fsqz2_0, fsql2_0, w0, jacmin0 = w_terms(state, zero_m1)
    w0 = float(np.asarray(w0))
    jacmin0 = float(np.asarray(jacmin0))
    if not np.isfinite(w0):
        raise ValueError("Initial state has non-finite residual objective")
    if jacmin0 <= 0.0 and verbose:
        print("[solve_fixed_boundary_lbfgs_vmec_residual] warning: initial Jacobian has non-positive entries")

    if objective_scale_f is None:
        # Auto-scale the objective to be O(1) on the initial iterate.
        objective_scale_f = 1.0 / max(abs(w0), 1.0)
        # Rebuild the objective closures with the now-fixed scale.
        def _fsq2_terms_and_jacmin(state: VMECState, zero_m1_zforce: Any):  # type: ignore[no-redef]
            k = vmec_forces_rz_from_wout(
                state=state,
                static=static,
                wout=wout_like,
                indata=None,
                constraint_tcon0=constraint_tcon0,
                use_vmec_synthesis=True,
                trig=trig,
            )
            rzl = vmec_residual_internal_from_kernels(
                k,
                cfg_ntheta=int(static.cfg.ntheta),
                cfg_nzeta=int(static.cfg.nzeta),
                wout=wout_like,
                trig=trig,
                apply_lforbal=apply_lforbal,
                include_edge=False,
                masks=mask_pack,
            )
            rzl = vmec_zero_m1_zforce(frzl=rzl, enabled=zero_m1_zforce)
            gcr2, gcz2, gcl2 = vmec_gcx2_from_tomnsps(
                frzl=rzl,
                lconm1=bool(getattr(static.cfg, "lconm1", True)),
                apply_m1_constraints=bool(apply_m1_constraints),
                include_edge=False,
                apply_scalxc=True,
                s=s,
            )
            norms = vmec_force_norms_from_bcovar_dynamic(bc=k.bc, trig=trig, s=s, signgs=signgs)
            fsqr2 = norms.r1 * norms.fnorm * gcr2
            fsqz2 = norms.r1 * norms.fnorm * gcz2
            fsql2 = norms.fnormL * gcl2

            w = (w_rz * (fsqr2 + fsqz2)) + (w_l * fsql2)
            w = jnp.asarray(objective_scale_f, dtype=jnp.asarray(w).dtype) * w

            jac = signgs * jnp.asarray(k.bc.jac.sqrtg)
            jac_min = jnp.min(jac) if jac.shape[0] <= 1 else jnp.min(jac[1:, :, :])
            return fsqr2, fsqz2, fsql2, w, jac_min

        def _w_only(state: VMECState, zero_m1_zforce: Any):  # type: ignore[no-redef]
            return _fsq2_terms_and_jacmin(state, zero_m1_zforce)[3]

        w_and_grad = jax.value_and_grad(_w_only)
        w_terms = _fsq2_terms_and_jacmin
        if jit_grad:
            w_and_grad = jit(w_and_grad)
            w_terms = jit(w_terms)

        fsqr2_0, fsqz2_0, fsql2_0, w0, jacmin0 = w_terms(state, zero_m1)
        w0 = float(np.asarray(w0))

    w_history = [w0]
    fsqr2_history = [float(np.asarray(fsqr2_0))]
    fsqz2_history = [float(np.asarray(fsqz2_0))]
    fsql2_history = [float(np.asarray(fsql2_0))]
    grad_rms_history = []
    step_history = []

    w_val, grad = w_and_grad(state, zero_m1)
    grad = _mask_grad_for_constraints(grad, static, idx00=idx00, mask_lambda_axis=False)
    grad = _apply_preconditioner(
        grad,
        static,
        kind=preconditioner,
        exponent=precond_exponent,
        radial_alpha=precond_radial_alpha,
    )
    sr = jnp.asarray(scale_rz, dtype=jnp.asarray(grad.Rcos).dtype)
    sl = jnp.asarray(scale_l, dtype=jnp.asarray(grad.Lcos).dtype)
    grad = VMECState(
        layout=grad.layout,
        Rcos=jnp.asarray(grad.Rcos) * sr,
        Rsin=jnp.asarray(grad.Rsin) * sr,
        Zcos=jnp.asarray(grad.Zcos) * sr,
        Zsin=jnp.asarray(grad.Zsin) * sr,
        Lcos=jnp.asarray(grad.Lcos) * sl,
        Lsin=jnp.asarray(grad.Lsin) * sl,
    )

    x = pack_state(state)
    g_flat = pack_state(grad)

    s_hist: list[Any] = []
    y_hist: list[Any] = []

    step0 = float(step_size)

    def _lbfgs_direction(g_flat, s_hist, y_hist):
        if not s_hist:
            return -g_flat
        q = g_flat
        alpha = []
        rho = []
        for s_i, y_i in zip(reversed(s_hist), reversed(y_hist)):
            ys = jnp.dot(y_i, s_i)
            rho_i = jnp.where(ys != 0, 1.0 / ys, 0.0)
            a_i = rho_i * jnp.dot(s_i, q)
            q = q - a_i * y_i
            alpha.append(a_i)
            rho.append(rho_i)

        s0 = s_hist[-1]
        y0 = y_hist[-1]
        ys0 = jnp.dot(y0, s0)
        yy0 = jnp.dot(y0, y0)
        gamma0 = jnp.where(yy0 != 0, ys0 / yy0, 1.0)
        r = gamma0 * q

        for s_i, y_i, a_i, rho_i in zip(s_hist, y_hist, reversed(alpha), reversed(rho)):
            beta = rho_i * jnp.dot(y_i, r)
            r = r + s_i * (a_i - beta)

        return -r

    for it in range(max_iter):
        grad_rms = _grad_rms_state(grad)
        grad_rms_history.append(grad_rms)

        if verbose:
            print(f"[solve_fixed_boundary_lbfgs_vmec_residual] iter={it:03d} w={w_history[-1]:.8e} grad_rms={grad_rms:.3e}")

        if grad_rms < grad_tol:
            break

        p_flat = _lbfgs_direction(g_flat, s_hist, y_hist)
        gtp = float(np.asarray(jnp.dot(g_flat, p_flat)))
        if not np.isfinite(gtp) or gtp >= 0.0:
            p_flat = -g_flat

        accepted = False
        step = step0
        best_w = np.inf
        best_state = None
        best_step = None
        best_fsqr2 = None
        best_fsqz2 = None
        best_fsql2 = None

        x_old = x
        g_old = g_flat

        zero_m1 = jnp.asarray(1.0 if (it < 2) or (fsqz2_history[-1] < 1e-6) else 0.0, dtype=jnp.asarray(state.Rcos).dtype)
        for bt in range(max_backtracks + 1):
            if bt > 0:
                step *= bt_factor
            x_try = x_old + jnp.asarray(step, dtype=x_old.dtype) * p_flat
            st_try = unpack_state(x_try, state.layout)
            st_try = _enforce_fixed_boundary_and_axis(
                st_try,
                static,
                edge_Rcos=edge_Rcos,
                edge_Rsin=edge_Rsin,
                edge_Zcos=edge_Zcos,
                edge_Zsin=edge_Zsin,
                idx00=idx00,
            )

            fsqr2_t, fsqz2_t, fsql2_t, w_t, jacmin_t = w_terms(st_try, zero_m1)
            w_tf = float(np.asarray(w_t))
            jacmin_tf = float(np.asarray(jacmin_t))
            if np.isfinite(w_tf) and w_tf < best_w:
                best_w = w_tf
                best_state = st_try
                best_step = step
                best_fsqr2 = float(np.asarray(fsqr2_t))
                best_fsqz2 = float(np.asarray(fsqz2_t))
                best_fsql2 = float(np.asarray(fsql2_t))
            if np.isfinite(w_tf) and jacmin_tf > 0.0 and w_tf < w_history[-1]:
                state = st_try
                x = pack_state(state)
                accepted = True
                fsqr2_accept = float(np.asarray(fsqr2_t))
                fsqz2_accept = float(np.asarray(fsqz2_t))
                fsql2_accept = float(np.asarray(fsql2_t))
                break

        step_history.append(step)

        if not accepted:
            if best_state is not None and np.isfinite(best_w):
                if verbose:
                    print(
                        "[solve_fixed_boundary_lbfgs_vmec_residual] line search failed; "
                        "accepting best finite step"
                    )
                state = best_state
                x = pack_state(state)
                w_t = best_w
                fsqr2_accept = best_fsqr2 if best_fsqr2 is not None else float(np.asarray(fsqr2_t))
                fsqz2_accept = best_fsqz2 if best_fsqz2 is not None else float(np.asarray(fsqz2_t))
                fsql2_accept = best_fsql2 if best_fsql2 is not None else float(np.asarray(fsql2_t))
                step_history[-1] = best_step
            else:
                if verbose:
                    print("[solve_fixed_boundary_lbfgs_vmec_residual] line search failed; stopping")
                break

        w_history.append(float(np.asarray(w_t)))
        fsqr2_history.append(fsqr2_accept)
        fsqz2_history.append(fsqz2_accept)
        fsql2_history.append(fsql2_accept)

        w_val, grad_new = w_and_grad(state, zero_m1)
        grad_new = _mask_grad_for_constraints(grad_new, static, idx00=idx00, mask_lambda_axis=False)
        grad_new = _apply_preconditioner(
            grad_new,
            static,
            kind=preconditioner,
            exponent=precond_exponent,
            radial_alpha=precond_radial_alpha,
        )
        g_flat_new = pack_state(grad_new)

        s_k = x - x_old
        y_k = g_flat_new - g_old
        ys = float(np.asarray(jnp.dot(y_k, s_k)))
        if np.isfinite(ys) and ys > 1e-14:
            s_hist.append(s_k)
            y_hist.append(y_k)
            if len(s_hist) > history_size:
                s_hist.pop(0)
                y_hist.pop(0)

        grad = grad_new
        g_flat = g_flat_new
        step0 = float(step)

    diag: Dict[str, Any] = {
        "idx00": idx00,
        "signgs": signgs,
        "w_rz": float(w_rz),
        "w_l": float(w_l),
        "objective_scale": float(objective_scale_f),
        "include_constraint_force": bool(include_constraint_force),
        "scale_rz": float(scale_rz),
        "scale_l": float(scale_l),
        "apply_m1_constraints": bool(apply_m1_constraints),
        "history_size": int(history_size),
        "preconditioner": str(preconditioner),
        "precond_exponent": float(precond_exponent),
        "precond_radial_alpha": float(precond_radial_alpha),
    }
    return SolveVmecResidualResult(
        state=state,
        n_iter=len(w_history) - 1,
        w_history=np.asarray(w_history, dtype=float),
        fsqr2_history=np.asarray(fsqr2_history, dtype=float),
        fsqz2_history=np.asarray(fsqz2_history, dtype=float),
        fsql2_history=np.asarray(fsql2_history, dtype=float),
        grad_rms_history=np.asarray(grad_rms_history, dtype=float),
        step_history=np.asarray(step_history, dtype=float),
        diagnostics=diag,
    )


def solve_fixed_boundary_gn_vmec_residual(
    state0: VMECState,
    static,
    *,
    indata,
    signgs: int,
    w_rz: float = 1.0,
    w_l: float = 1.0,
    include_constraint_force: bool = True,
    apply_m1_constraints: bool = True,
    objective_scale: float | None = None,
    damping: float = 1e-3,
    damping_increase: float = 10.0,
    damping_decrease: float = 0.5,
    max_damping: float = 1e6,
    max_retries: int = 6,
    zero_m1_iters: int = 50,
    zero_m1_fsqz_thresh: float = 1e-6,
    max_iter: int = 20,
    cg_tol: float = 1e-6,
    cg_maxiter: int = 80,
    step_size: float = 1.0,
    max_backtracks: int = 12,
    bt_factor: float = 0.5,
    jit_kernels: bool = True,
    verbose: bool = True,
) -> SolveVmecResidualResult:
    """Fixed-boundary solve using a Gauss-Newton (normal-equations) step on VMEC residuals.

    This treats the VMEC residual blocks returned by `tomnsps` as a least-squares
    problem and solves (approximately) for a step `dx` using conjugate gradients:

        (Jᵀ J + damping * I) dx = -Jᵀ r

    where `r(state)` is the stacked residual vector and `J` is its Jacobian.

    The residual vector uses the same conventions as `vmec_jax.vmec_residue`
    (post-`tomnsps` `scalxc` scaling, optional m=1 constraints, and R/Z edge
    exclusion) so the objective is consistent with the scalar residual definitions.
    """
    if not has_jax():
        raise ImportError("solve_fixed_boundary_gn_vmec_residual requires JAX (jax + jaxlib)")
    if damping < 0.0:
        raise ValueError("damping must be nonnegative")
    damping_increase = float(damping_increase)
    damping_decrease = float(damping_decrease)
    max_damping = float(max_damping)
    max_retries = int(max_retries)
    if damping_increase <= 1.0:
        raise ValueError("damping_increase must be > 1")
    if not (0.0 < damping_decrease <= 1.0):
        raise ValueError("damping_decrease must be in (0, 1]")
    if max_damping <= 0.0:
        raise ValueError("max_damping must be positive")
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    zero_m1_iters = int(zero_m1_iters)
    zero_m1_fsqz_thresh = float(zero_m1_fsqz_thresh)
    if zero_m1_iters < 0:
        raise ValueError("zero_m1_iters must be >= 0")
    if zero_m1_fsqz_thresh < 0.0:
        raise ValueError("zero_m1_fsqz_thresh must be >= 0")
    w_rz = float(w_rz)
    w_l = float(w_l)
    if w_rz < 0.0 or w_l < 0.0:
        raise ValueError("w_rz and w_l must be nonnegative")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if cg_maxiter < 1:
        raise ValueError("cg_maxiter must be >= 1")
    if not (0.0 < bt_factor < 1.0):
        raise ValueError("bt_factor must be in (0, 1)")
    if objective_scale is not None and float(objective_scale) <= 0.0:
        raise ValueError("objective_scale must be positive when provided")

    constraint_tcon0: float | None = None
    if bool(include_constraint_force):
        constraint_tcon0 = float(indata.get_float("TCON0", 1.0))
    apply_lforbal = bool(indata.get_bool("LFORBAL", False)) if indata is not None else False
    apply_lforbal = bool(indata.get_bool("LFORBAL", False)) if indata is not None else False
    apply_lforbal = bool(indata.get_bool("LFORBAL", False)) if indata is not None else False
    apply_lforbal = bool(indata.get_bool("LFORBAL", False)) if indata is not None else False

    signgs = int(signgs)
    idx00 = _mode00_index(static.modes)

    from .energy import flux_profiles_from_indata
    from .static import build_static
    from .vmec_forces import vmec_forces_rz_from_wout, vmec_residual_internal_from_kernels
    from .vmec_residue import (
        vmec_apply_m1_constraints,
        vmec_apply_scalxc_to_tomnsps,
        vmec_force_norms_from_bcovar_dynamic,
        vmec_scalxc_from_s,
        vmec_zero_m1_zforce,
    )
    from .vmec_tomnsp import TomnspsRZL, vmec_angle_grid, vmec_trig_tables

    try:
        from jax.scipy.sparse.linalg import cg  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("solve_fixed_boundary_gn_vmec_residual requires jax.scipy.sparse.linalg.cg") from e

    s = jnp.asarray(static.s)
    flux = flux_profiles_from_indata(indata, s, signgs=signgs)
    chipf_wout = jnp.asarray(flux.chipf)

    phips = jnp.asarray(flux.phips)
    if phips.shape[0] >= 1:
        phips = phips.at[0].set(0.0)

    from .boundary import boundary_from_indata

    boundary = boundary_from_indata(indata, static.modes)
    r00 = float(np.asarray(boundary.R_cos)[int(idx00)]) if int(idx00) >= 0 else float(np.asarray(boundary.R_cos)[0])
    gamma = float(indata.get_float("GAMMA", 5.0 / 3.0))
    lrfp = bool(indata.get_bool("LRFP", False))
    chips = _half_mesh_from_full_mesh(chipf_wout) if lrfp else None
    mass = _mass_half_mesh_from_indata(
        indata=indata,
        s_full=s,
        phips=phips,
        r00=r00,
        gamma=gamma,
        lrfp=lrfp,
        chips=chips,
    )

    pres = _pressure_half_mesh_from_indata(indata=indata, s_full=s)
    ncurr = int(indata.get_int("NCURR", 0))
    icurv = _icurv_full_mesh_from_indata(indata=indata, s_full=s, signgs=signgs)

    wout_like = _WoutLikeVmecForces(
        nfp=int(static.cfg.nfp),
        mpol=int(static.cfg.mpol),
        ntor=int(static.cfg.ntor),
        lasym=bool(static.cfg.lasym),
        signgs=signgs,
        phipf=jnp.asarray(flux.phipf),
        phips=phips,
        chipf=chipf_wout,
        pres=pres,
        mass=mass,
        gamma=gamma,
        ncurr=ncurr,
        lcurrent=True,
        icurv=icurv,
    )

    trig = getattr(static, "trig_vmec", None)
    if trig is None:
        trig = vmec_trig_tables(
            ntheta=int(static.cfg.ntheta),
            nzeta=int(static.cfg.nzeta),
            nfp=int(wout_like.nfp),
            mmax=int(wout_like.mpol) - 1,
            nmax=int(wout_like.ntor),
            lasym=bool(wout_like.lasym),
            dtype=jnp.asarray(state0.Rcos).dtype,
        )

    # VMEC updates the *unscaled* coefficients using scalxc-weighted residuals.
    # By default we keep that behavior (no division by scalxc). The optional
    # `divide_by_scalxc_for_update` hook exists for experiments only.
    scalxc = vmec_scalxc_from_s(s=s, mpol=int(static.cfg.mpol))  # (ns, mpol)
    scalxc_mn = scalxc[:, :, None]  # (ns, mpol, ntor+1)

    edge_Rcos = jnp.asarray(state0.Rcos)[-1, :]
    edge_Rsin = jnp.asarray(state0.Rsin)[-1, :]
    edge_Zcos = jnp.asarray(state0.Zcos)[-1, :]
    edge_Zsin = jnp.asarray(state0.Zsin)[-1, :]

    def _project_step(d: VMECState) -> VMECState:
        return _mask_grad_for_constraints(d, static, idx00=idx00, mask_lambda_axis=True)

    def _enforce_state(st: VMECState) -> VMECState:
        return _enforce_fixed_boundary_and_axis(
            st,
            static,
            edge_Rcos=edge_Rcos,
            edge_Rsin=edge_Rsin,
            edge_Zcos=edge_Zcos,
            edge_Zsin=edge_Zsin,
            enforce_lambda_axis=True,
            idx00=idx00,
        )

    def _zero_edge_rz(a):
        a = None if a is None else jnp.asarray(a)
        if a is None:
            return None
        if a.shape[0] < 2:
            return a
        return a.at[-1].set(jnp.zeros_like(a[-1]))

    mask_pack = getattr(static, "tomnsps_masks", None)

    def _residual_blocks(state: VMECState, zero_m1_zforce: Any):
        k = vmec_forces_rz_from_wout(
            state=state,
            static=static,
            wout=wout_like,
            indata=None,
            constraint_tcon0=constraint_tcon0,
            use_vmec_synthesis=True,
            trig=trig,
        )
        rzl = vmec_residual_internal_from_kernels(
            k,
            cfg_ntheta=int(static.cfg.ntheta),
            cfg_nzeta=int(static.cfg.nzeta),
            wout=wout_like,
            trig=trig,
            apply_lforbal=apply_lforbal,
            include_edge=False,
            masks=mask_pack,
        )
        frzl = rzl
        if bool(apply_m1_constraints):
            frzl = vmec_apply_m1_constraints(frzl=frzl, lconm1=bool(getattr(static.cfg, "lconm1", True)))
        frzl = vmec_zero_m1_zforce(frzl=frzl, enabled=zero_m1_zforce)

        # VMEC convention: after tomnsps, scale Fourier-space forces by `scalxc`
        # before forming sums-of-squares/scalars (funct3d.f).
        frzl = vmec_apply_scalxc_to_tomnsps(frzl=frzl, s=s)

        # VMEC convention: R/Z sums exclude the edge surface; enforce that by
        # zeroing R/Z blocks at js=ns (lambda blocks are left untouched).
        frzl = TomnspsRZL(
            frcc=_zero_edge_rz(frzl.frcc),
            frss=_zero_edge_rz(frzl.frss),
            fzsc=_zero_edge_rz(frzl.fzsc),
            fzcs=_zero_edge_rz(frzl.fzcs),
            flsc=frzl.flsc,
            flcs=frzl.flcs,
            frsc=_zero_edge_rz(getattr(frzl, "frsc", None)),
            frcs=_zero_edge_rz(getattr(frzl, "frcs", None)),
            fzcc=_zero_edge_rz(getattr(frzl, "fzcc", None)),
            fzss=_zero_edge_rz(getattr(frzl, "fzss", None)),
            flcc=getattr(frzl, "flcc", None),
            flss=getattr(frzl, "flss", None),
        )

        gcr2 = jnp.sum(jnp.asarray(frzl.frcc) ** 2)
        gcz2 = jnp.sum(jnp.asarray(frzl.fzsc) ** 2)
        gcl2 = jnp.sum(jnp.asarray(frzl.flsc) ** 2)
        if frzl.frss is not None:
            gcr2 = gcr2 + jnp.sum(jnp.asarray(frzl.frss) ** 2)
        if frzl.fzcs is not None:
            gcz2 = gcz2 + jnp.sum(jnp.asarray(frzl.fzcs) ** 2)
        if frzl.flcs is not None:
            gcl2 = gcl2 + jnp.sum(jnp.asarray(frzl.flcs) ** 2)

        if getattr(frzl, "frsc", None) is not None:
            gcr2 = gcr2 + jnp.sum(jnp.asarray(frzl.frsc) ** 2)
        if getattr(frzl, "fzcc", None) is not None:
            gcz2 = gcz2 + jnp.sum(jnp.asarray(frzl.fzcc) ** 2)
        if getattr(frzl, "flcc", None) is not None:
            gcl2 = gcl2 + jnp.sum(jnp.asarray(frzl.flcc) ** 2)

        if getattr(frzl, "frcs", None) is not None:
            gcr2 = gcr2 + jnp.sum(jnp.asarray(frzl.frcs) ** 2)
        if getattr(frzl, "fzss", None) is not None:
            gcz2 = gcz2 + jnp.sum(jnp.asarray(frzl.fzss) ** 2)
        if getattr(frzl, "flss", None) is not None:
            gcl2 = gcl2 + jnp.sum(jnp.asarray(frzl.flss) ** 2)

        norms = vmec_force_norms_from_bcovar_dynamic(bc=k.bc, trig=trig, s=s, signgs=signgs)
        fsqr2 = norms.r1 * norms.fnorm * gcr2
        fsqz2 = norms.r1 * norms.fnorm * gcz2
        fsql2 = norms.fnormL * gcl2
        return frzl, fsqr2, fsqz2, fsql2, norms

    def _residual_vec(state: VMECState, zero_m1_zforce: Any) -> Any:
        frzl, *_vals = _residual_blocks(state, zero_m1_zforce)
        norms = _vals[-1]
        scale_rz = jnp.sqrt(jnp.asarray(w_rz)) * jnp.sqrt(norms.r1 * norms.fnorm)
        scale_l = jnp.sqrt(jnp.asarray(w_l)) * jnp.sqrt(norms.fnormL)
        scale_rz = jnp.asarray(scale_rz, dtype=jnp.asarray(frzl.frcc).dtype)
        scale_l = jnp.asarray(scale_l, dtype=jnp.asarray(frzl.frcc).dtype)

        parts = [scale_rz * frzl.frcc, scale_rz * frzl.fzsc, scale_l * frzl.flsc]
        if frzl.frss is not None:
            parts.append(scale_rz * frzl.frss)
        if frzl.fzcs is not None:
            parts.append(scale_rz * frzl.fzcs)
        if frzl.flcs is not None:
            parts.append(scale_l * frzl.flcs)
        for name in ["frsc", "fzcc", "flcc", "frcs", "fzss", "flss"]:
            a = getattr(frzl, name, None)
            if a is not None:
                if name.startswith("fl"):
                    parts.append(scale_l * a)
                else:
                    parts.append(scale_rz * a)
        return jnp.concatenate([jnp.ravel(jnp.asarray(p)) for p in parts], axis=0)

    def _obj_terms(state: VMECState, zero_m1_zforce: Any):
        _frzl, fsqr2, fsqz2, fsql2, _norms = _residual_blocks(state, zero_m1_zforce)
        w = (w_rz * (fsqr2 + fsqz2)) + (w_l * fsql2)
        return fsqr2, fsqz2, fsql2, w

    if bool(jit_kernels):
        _residual_vec_jit = jit(_residual_vec)
        _obj_terms_jit = jit(_obj_terms)
    else:
        _residual_vec_jit = _residual_vec
        _obj_terms_jit = _obj_terms

    state = _enforce_state(state0)
    zero_m1 = jnp.asarray(1.0, dtype=jnp.asarray(state0.Rcos).dtype)
    fsqr2_0, fsqz2_0, fsql2_0, w0 = _obj_terms_jit(state, zero_m1)
    w0_f = float(np.asarray(w0))
    if not np.isfinite(w0_f):
        raise ValueError("Initial state has non-finite residual objective")

    scale_f = float(objective_scale) if objective_scale is not None else (1.0 / max(abs(w0_f), 1.0))

    w_history = [float(scale_f * w0_f)]
    fsqr2_history = [float(np.asarray(fsqr2_0))]
    fsqz2_history = [float(np.asarray(fsqz2_0))]
    fsql2_history = [float(np.asarray(fsql2_0))]
    grad_rms_history = []
    step_history = []

    damping_it = float(damping)
    for it in range(int(max_iter)):
        zero_m1 = jnp.asarray(
            1.0 if (it < zero_m1_iters) or (fsqz2_history[-1] < zero_m1_fsqz_thresh) else 0.0,
            dtype=jnp.asarray(state.Rcos).dtype,
        )
        r, pullback = jax.vjp(_residual_vec_jit, state, zero_m1)
        # Gradient of 0.5*||r||^2 is J^T r.
        g_state = pullback(r)[0]
        g_state = _project_step(g_state)
        grad_rms_history.append(_grad_rms_state(g_state))

        b_flat = -pack_state(g_state)

        accepted = False
        step = float(step_size)
        w_curr = w_history[-1]
        retry = 0
        while True:
            dmp = float(damping_it)

            def _matvec(v_flat):
                v_state = unpack_state(v_flat, state.layout)
                v_state = _project_step(v_state)
                zero_tangent = jnp.zeros_like(zero_m1)
                jv = jax.jvp(_residual_vec_jit, (state, zero_m1), (v_state, zero_tangent))[1]
                jt_jv = pullback(jv)[0]
                jt_jv = _project_step(jt_jv)
                if dmp != 0.0:
                    jt_jv = VMECState(
                        layout=jt_jv.layout,
                        Rcos=jt_jv.Rcos + dmp * v_state.Rcos,
                        Rsin=jt_jv.Rsin + dmp * v_state.Rsin,
                        Zcos=jt_jv.Zcos + dmp * v_state.Zcos,
                        Zsin=jt_jv.Zsin + dmp * v_state.Zsin,
                        Lcos=jt_jv.Lcos + dmp * v_state.Lcos,
                        Lsin=jt_jv.Lsin + dmp * v_state.Lsin,
                    )
                return pack_state(jt_jv)

            dx_flat, _info = cg(_matvec, b_flat, tol=float(cg_tol), maxiter=int(cg_maxiter))
            dx_state = unpack_state(dx_flat, state.layout)
            dx_state = _project_step(dx_state)

            step = float(step_size)
            for bt in range(int(max_backtracks) + 1):
                if bt > 0:
                    step *= float(bt_factor)
                st_try = VMECState(
                    layout=state.layout,
                    Rcos=jnp.asarray(state.Rcos) + jnp.asarray(step, dtype=jnp.asarray(state.Rcos).dtype) * jnp.asarray(dx_state.Rcos),
                    Rsin=jnp.asarray(state.Rsin) + jnp.asarray(step, dtype=jnp.asarray(state.Rsin).dtype) * jnp.asarray(dx_state.Rsin),
                    Zcos=jnp.asarray(state.Zcos) + jnp.asarray(step, dtype=jnp.asarray(state.Zcos).dtype) * jnp.asarray(dx_state.Zcos),
                    Zsin=jnp.asarray(state.Zsin) + jnp.asarray(step, dtype=jnp.asarray(state.Zsin).dtype) * jnp.asarray(dx_state.Zsin),
                    Lcos=jnp.asarray(state.Lcos) + jnp.asarray(step, dtype=jnp.asarray(state.Lcos).dtype) * jnp.asarray(dx_state.Lcos),
                    Lsin=jnp.asarray(state.Lsin) + jnp.asarray(step, dtype=jnp.asarray(state.Lsin).dtype) * jnp.asarray(dx_state.Lsin),
                )
                st_try = _enforce_state(st_try)
                fsqr2_t, fsqz2_t, fsql2_t, w_t = _obj_terms_jit(st_try, zero_m1)
                w_tf = float(np.asarray(w_t))
                w_scaled = float(scale_f * w_tf)
                if np.isfinite(w_scaled) and w_scaled < w_curr:
                    state = st_try
                    accepted = True
                    w_history.append(w_scaled)
                    fsqr2_history.append(float(np.asarray(fsqr2_t)))
                    fsqz2_history.append(float(np.asarray(fsqz2_t)))
                    fsql2_history.append(float(np.asarray(fsql2_t)))
                    break

            if accepted:
                # Levenberg-Marquardt style: relax damping after success.
                damping_it = max(damping_it * damping_decrease, 0.0)
                break

            if retry >= max_retries or damping_it >= max_damping:
                break
            # Increase damping and try again from the same state.
            damping_it = min(max_damping, damping_it * damping_increase)
            retry += 1

        if not accepted:
            # Robust fallback: take a small steepest-descent step on 0.5*||r||^2
            # using the already-computed gradient g_state = J^T r.
            dx_state = unpack_state(b_flat, state.layout)  # b_flat = -grad_flat
            dx_state = _project_step(dx_state)
            step = float(step_size)
            for bt in range(int(max_backtracks) + 1):
                if bt > 0:
                    step *= float(bt_factor)
                st_try = VMECState(
                    layout=state.layout,
                    Rcos=jnp.asarray(state.Rcos) + jnp.asarray(step, dtype=jnp.asarray(state.Rcos).dtype) * jnp.asarray(dx_state.Rcos),
                    Rsin=jnp.asarray(state.Rsin) + jnp.asarray(step, dtype=jnp.asarray(state.Rsin).dtype) * jnp.asarray(dx_state.Rsin),
                    Zcos=jnp.asarray(state.Zcos) + jnp.asarray(step, dtype=jnp.asarray(state.Zcos).dtype) * jnp.asarray(dx_state.Zcos),
                    Zsin=jnp.asarray(state.Zsin) + jnp.asarray(step, dtype=jnp.asarray(state.Zsin).dtype) * jnp.asarray(dx_state.Zsin),
                    Lcos=jnp.asarray(state.Lcos) + jnp.asarray(step, dtype=jnp.asarray(state.Lcos).dtype) * jnp.asarray(dx_state.Lcos),
                    Lsin=jnp.asarray(state.Lsin) + jnp.asarray(step, dtype=jnp.asarray(state.Lsin).dtype) * jnp.asarray(dx_state.Lsin),
                )
                st_try = _enforce_state(st_try)
                fsqr2_t, fsqz2_t, fsql2_t, w_t = _obj_terms_jit(st_try, zero_m1)
                w_tf = float(np.asarray(w_t))
                w_scaled = float(scale_f * w_tf)
                if np.isfinite(w_scaled) and w_scaled < w_curr:
                    state = st_try
                    accepted = True
                    w_history.append(w_scaled)
                    fsqr2_history.append(float(np.asarray(fsqr2_t)))
                    fsqz2_history.append(float(np.asarray(fsqz2_t)))
                    fsql2_history.append(float(np.asarray(fsql2_t)))
                    break

        step_history.append(step)
        if verbose:
            print(
                f"[solve_fixed_boundary_gn_vmec_residual] iter={it:03d} w={w_history[-1]:.8e} "
                f"step={step:.3e} accepted={accepted} damping={damping_it:.3e} retries={retry}"
            )

        if not accepted:
            break

    diag = {
        "idx00": idx00,
        "signgs": signgs,
        "w_rz": float(w_rz),
        "w_l": float(w_l),
        "objective_scale": float(scale_f),
        "apply_m1_constraints": bool(apply_m1_constraints),
        "damping": float(damping),
        "cg_tol": float(cg_tol),
        "cg_maxiter": int(cg_maxiter),
    }
    return SolveVmecResidualResult(
        state=state,
        n_iter=len(w_history) - 1,
        w_history=np.asarray(w_history, dtype=float),
        fsqr2_history=np.asarray(fsqr2_history, dtype=float),
        fsqz2_history=np.asarray(fsqz2_history, dtype=float),
        fsql2_history=np.asarray(fsql2_history, dtype=float),
        grad_rms_history=np.asarray(grad_rms_history, dtype=float),
        step_history=np.asarray(step_history, dtype=float),
        diagnostics=diag,
    )


def solve_fixed_boundary_residual_iter(
    state0: VMECState,
    static,
    *,
    indata,
    signgs: int,
    ftol: float | None = None,
    max_iter: int = 50,
    step_size: float = 1.0,
    initial_flip_sign: float = 1.0,
    include_constraint_force: bool = True,
    apply_m1_constraints: bool = True,
    precond_radial_alpha: float = 0.5,
    precond_lambda_alpha: float = 0.5,
    mode_diag_exponent: float = 0.0,
    auto_flip_force: bool = True,
    divide_by_scalxc_for_update: bool = False,
    lambda_update_scale: float = 1.0,
    enforce_vmec_lambda_axis: bool = False,
    vmec2000_control: bool = False,
    strict_update: bool = True,
    backtracking: bool = False,
    limit_dt_from_force: bool = False,
    limit_update_rms: bool = False,
    reference_mode: bool = False,
    use_restart_triggers: bool | None = None,
    vmecpp_restart: bool = False,
    stage_prev_fsq: float | None = None,
    stage_transition_factor: float = 50.0,
    stage_transition_scale: float = 0.5,
    use_direct_fallback: bool | None = None,
    verbose: bool = True,
    verbose_vmec2000_table: bool = True,
    jit_forces: bool = True,
    jit_warmup_iters: int = 0,
    jit_precompile: bool = False,
    use_scan: bool = False,
    precompile_only: bool = False,
    resume_state: dict | None = None,
    scan_minimal_default: bool | None = None,
) -> SolveVmecResidualResult:
    """VMEC-style fixed-point update loop using preconditioned force residuals."""
    if not has_jax():
        raise ImportError("solve_fixed_boundary_residual_iter requires JAX (jax + jaxlib)")

    def _device_get_floats(*vals):
        """Batch host materialization for scalar diagnostics.

        In the VMEC2000 parity (non-scan) loop we still need host scalars to
        drive Python control flow (TimeStepControl / restarts / printing).
        Pulling them one-by-one forces repeated synchronization; batch them.
        """

        return tuple(float(v) for v in jax.device_get(vals))

    max_iter = int(max_iter)
    precompile_only = bool(precompile_only)
    if max_iter < 1 and not precompile_only:
        raise ValueError("max_iter must be >= 1")
    if max_iter < 1 and precompile_only:
        max_iter = 1
    step_size = float(step_size)
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")

    signgs = int(signgs)
    lambda_update_scale = float(lambda_update_scale)
    enforce_vmec_lambda_axis = bool(enforce_vmec_lambda_axis)
    vmec2000_control = bool(vmec2000_control)
    badjac_mode_env = os.getenv("VMEC_JAX_BADJAC_MODE", "ptau").strip().lower()
    if badjac_mode_env not in ("ptau", "state"):
        badjac_mode_env = "ptau"
    badjac_mode = badjac_mode_env
    badjac_use_state = badjac_mode == "state"
    dump_ptau_state_env = os.getenv("VMEC_JAX_DUMP_PTAU_STATE", "0").strip().lower()
    dump_ptau_state = dump_ptau_state_env not in ("", "0", "false", "no")
    light_hist_env = os.getenv("VMEC_JAX_LIGHT_HISTORY", "0").strip().lower()
    light_history = light_hist_env not in ("", "0", "false", "no")
    badjac_state_probe_env = os.getenv("VMEC_JAX_BADJAC_STATE_PROBE", "0").strip().lower()
    badjac_state_probe = badjac_state_probe_env not in ("", "0", "false", "no")
    ptau_tol_env = os.getenv("VMEC_JAX_PTAU_TOL", "").strip()
    ptau_tol_rel_env = os.getenv("VMEC_JAX_PTAU_TOL_REL", "1.0e-6").strip()
    try:
        ptau_tol = float(ptau_tol_env) if ptau_tol_env else 0.0
    except Exception:
        ptau_tol = 0.0
    try:
        ptau_tol_rel = float(ptau_tol_rel_env) if ptau_tol_rel_env else 0.0
    except Exception:
        ptau_tol_rel = 0.0
    if ptau_tol_rel < 0.0:
        ptau_tol_rel = 0.0
    reference_mode = bool(reference_mode)
    jit_precompile = bool(jit_precompile)
    if use_restart_triggers is None:
        # Restart triggers are generally stabilizing. Keep them on by default
        # so the fixed-point update loop is robust during parity work.
        use_restart_triggers = True
    if use_direct_fallback is None:
        use_direct_fallback = False
    use_restart_triggers = bool(use_restart_triggers)
    use_direct_fallback = bool(use_direct_fallback)
    vmecpp_restart = bool(vmecpp_restart)
    verbose_vmec2000_table = bool(verbose_vmec2000_table)
    # Allow automatic fallback to the non-scan path when scan diverges.
    scan_fallback_env = os.getenv("VMEC_JAX_SCAN_FALLBACK", "1").strip().lower()
    scan_fallback_enabled = scan_fallback_env not in ("", "0", "false", "no")
    scan_fallback_iters_env = os.getenv("VMEC_JAX_SCAN_FALLBACK_ITERS", "50").strip()
    scan_fallback_badjac_env = os.getenv("VMEC_JAX_SCAN_FALLBACK_BJAC_LIMIT", "10").strip()
    try:
        scan_fallback_iters = max(1, int(scan_fallback_iters_env))
    except Exception:
        scan_fallback_iters = 20
    try:
        scan_fallback_badjac_limit = max(0, int(scan_fallback_badjac_env))
    except Exception:
        scan_fallback_badjac_limit = 10
    scan_fallback_fsq_abs_env = os.getenv("VMEC_JAX_SCAN_FALLBACK_FSQ_ABS", "1.0e-2").strip()
    try:
        scan_fallback_fsq_abs = float(scan_fallback_fsq_abs_env)
    except Exception:
        scan_fallback_fsq_abs = 1.0e-2
    if scan_fallback_fsq_abs < 0.0:
        scan_fallback_fsq_abs = 0.0
    scan_fallback_accept_env = os.getenv("VMEC_JAX_SCAN_FALLBACK_ACCEPTED_FRAC", "0.5").strip()
    scan_fallback_fsq_env = os.getenv("VMEC_JAX_SCAN_FALLBACK_FSQ_FACTOR", "50").strip()
    scan_fallback_improve_env = os.getenv("VMEC_JAX_SCAN_FALLBACK_IMPROVE", "0.1").strip()
    try:
        scan_fallback_accept_frac = float(scan_fallback_accept_env)
    except Exception:
        scan_fallback_accept_frac = 0.5
    try:
        scan_fallback_fsq_factor = float(scan_fallback_fsq_env)
    except Exception:
        scan_fallback_fsq_factor = 50.0
    try:
        scan_fallback_improve = float(scan_fallback_improve_env)
    except Exception:
        scan_fallback_improve = 0.9
    if scan_fallback_accept_frac < 0.0:
        scan_fallback_accept_frac = 0.0
    if scan_fallback_accept_frac > 1.0:
        scan_fallback_accept_frac = 1.0
    if scan_fallback_fsq_factor < 1.0:
        scan_fallback_fsq_factor = 1.0
    if scan_fallback_improve <= 0.0 or scan_fallback_improve >= 1.0:
        scan_fallback_improve = 0.9
    stage_transition_factor = float(stage_transition_factor)
    stage_transition_scale = float(stage_transition_scale)
    if stage_transition_factor <= 0.0 or stage_transition_scale <= 0.0:
        stage_prev_fsq = None

    if use_scan and vmec2000_control and auto_flip_force:
        auto_flip_force = False
    jit_forces = bool(jit_forces)
    use_scan = bool(use_scan)
    # Default to chunked scan to reduce per-iteration host sync overhead.
    # Respect explicit non-scan requests (e.g., parity comparators).
    chunked_device_env = os.getenv("VMEC_JAX_VMEC2000_CHUNKED", "1").strip().lower()
    force_chunked_scan = chunked_device_env not in ("", "0", "false", "no")
    if force_chunked_scan and (not use_scan):
        force_chunked_scan = False
    limit_dt_from_force = bool(limit_dt_from_force)
    limit_update_rms = bool(limit_update_rms)
    backtracking = bool(backtracking)
    strict_update = bool(strict_update)
    light_dump_envs = (
        "VMEC_JAX_DUMP_SCALARS",
        "VMEC_JAX_DUMP_GCX2",
        "VMEC_JAX_DUMP_FSQ1",
        "VMEC_JAX_DUMP_TIMECONTROL",
        "VMEC_JAX_DUMP_CHECKPOINT",
    )
    heavy_dump_envs = (
        "VMEC_JAX_DUMP_TOMNSPS",
        "VMEC_JAX_DUMP_TOMNSPS_KERNELS",
        "VMEC_JAX_DUMP_FORCE_KERNELS",
        "VMEC_JAX_DUMP_GC",
        "VMEC_JAX_DUMP_BSUBE",
        "VMEC_JAX_DUMP_BSUBE_TERMS",
        "VMEC_JAX_DUMP_LULV",
        "VMEC_JAX_DUMP_XC",
        "VMEC_JAX_DUMP_LAM",
        "VMEC_JAX_DUMP_LAMCAL",
        "VMEC_JAX_DUMP_LAM_FSQL1",
        "VMEC_JAX_DUMP_LAM_GCL",
    )
    dumps_enabled = any(os.getenv(name, "") not in ("", "0") for name in heavy_dump_envs)
    dump_any = dumps_enabled or any(os.getenv(name, "") not in ("", "0") for name in light_dump_envs)
    if dumps_enabled and jit_forces:
        if verbose:
            print("[solve_fixed_boundary_residual_iter] jit_forces disabled (debug dumps enabled)")
        jit_forces = False
    if dump_any:
        # Force full histories when parity dumps are enabled.
        light_history = False
    track_history = not light_history

    from .energy import flux_profiles_from_indata
    from .energy import magnetic_wb_from_state
    from .static import build_static
    from .boundary import boundary_from_indata
    from .init_guess import (
        _boundary_cross_section_areas,
        _recompute_axis_from_boundary,
        _recompute_axis_from_state_vmec,
        _read_axis_coeffs,
        initial_guess_from_boundary,
    )
    from .vmec_forces import vmec_forces_rz_from_wout, vmec_residual_internal_from_kernels
    from .vmec_residue import (
        vmec_apply_m1_constraints,
        vmec_apply_scalxc_to_tomnsps,
        vmec_force_norms_from_bcovar_dynamic,
        vmec_gcx2_from_tomnsps,
        vmec_scalxc_from_s,
        vmec_wint_from_trig,
        vmec_zero_m1_zforce,
    )
    from .vmec_jacobian import vmec_half_mesh_jacobian_from_state
    from .vmec_tomnsp import TomnspsRZL, vmec_angle_grid, vmec_trig_tables

    # VMEC2000 evaluates the force kernels on VMEC's internal
    # angle grid. In particular, when `lasym=False`, VMEC uses a reduced theta
    # grid (stellarator symmetry) for the force pipeline. Rebuild `static`
    # using `vmec_angle_grid(...)` so the force terms do not mix full-grid and
    # VMEC-grid arrays (which triggers broadcasting errors and parity drift).
    cfg = static.cfg
    grid_vmec = vmec_angle_grid(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(cfg.nfp),
        lasym=bool(cfg.lasym),
    )
    reuse_static = False
    try:
        theta_curr = np.asarray(static.grid.theta)
        zeta_curr = np.asarray(static.grid.zeta)
        reuse_static = (
            int(static.grid.nfp) == int(grid_vmec.nfp)
            and theta_curr.shape == np.asarray(grid_vmec.theta).shape
            and zeta_curr.shape == np.asarray(grid_vmec.zeta).shape
            and np.allclose(theta_curr, np.asarray(grid_vmec.theta))
            and np.allclose(zeta_curr, np.asarray(grid_vmec.zeta))
        )
    except Exception:
        reuse_static = False
    if not reuse_static:
        static = build_static(cfg, grid=grid_vmec)

    idx00 = _mode00_index(static.modes)
    m_modes = np.asarray(getattr(static, "m_np", None) if getattr(static, "m_np", None) is not None else static.modes.m, dtype=int)
    n_modes = np.asarray(getattr(static, "n_np", None) if getattr(static, "n_np", None) is not None else static.modes.n, dtype=int)
    axis_copy_mask_np = (
        np.asarray(getattr(static, "lambda_axis_copy_mask", None), dtype=bool)
        if getattr(static, "lambda_axis_copy_mask", None) is not None
        else (m_modes == 0) & (n_modes > 0)
    )
    lambda_axis_copy_mask = jnp.asarray(axis_copy_mask_np, dtype=jnp.asarray(state0.Rcos).dtype)
    s = jnp.asarray(static.s)
    dtype_state = jnp.asarray(state0.Rcos).dtype
    zero_precond_diag = (
        jnp.zeros((int(s.shape[0]),), dtype=dtype_state),
        jnp.zeros((int(s.shape[0]),), dtype=dtype_state),
    )
    zero_tcon = jnp.zeros((int(s.shape[0]),), dtype=dtype_state)
    constraint_active_false = jnp.asarray(False)

    # Boundary + axis recompute helpers (for VMEC-style bad-Jacobian reset).
    boundary_for_axis = (
        boundary_from_indata(indata, static.modes, apply_m1_constraint=True) if indata is not None else None
    )
    axis_reset_done = bool(resume_state is not None)
    lmove_axis = True if indata is None else bool(indata.get_bool("LMOVE_AXIS", True))
    force_axis_reset = False if indata is None else bool(indata.get_bool("LFREEB", False))
    axis_reset_env = os.getenv("VMEC_JAX_AXIS_RESET_ALWAYS_3D", "0").strip().lower()
    axis_reset_always_3d = axis_reset_env not in ("", "0", "false", "no")
    axis_reset_fsq_env = os.getenv("VMEC_JAX_AXIS_RESET_FSQ_MIN", "1.0").strip()
    try:
        axis_reset_fsq_min = float(axis_reset_fsq_env) if axis_reset_fsq_env else 0.0
    except Exception:
        axis_reset_fsq_min = 0.0
    if axis_reset_fsq_min < 0.0:
        axis_reset_fsq_min = 0.0

    def _apply_vmec_lambda_axis_rules(st: VMECState) -> VMECState:
        """Enforce VMEC lambda gauge without mutating stored axis coefficients.

        VMEC applies the m=0 lambda axis-closure during real-space synthesis
        (totzsps) but does not overwrite the stored `xc` coefficients. Keep
        the state axis row intact and only enforce the (m,n)=(0,0) gauge here.
        """
        if not enforce_vmec_lambda_axis:
            return st
        Lcos = jnp.asarray(st.Lcos)
        Lsin = jnp.asarray(st.Lsin)
        Lcos, Lsin = _enforce_lambda_gauge(Lcos, Lsin, idx00=idx00)
        return VMECState(
            layout=st.layout,
            Rcos=st.Rcos,
            Rsin=st.Rsin,
            Zcos=st.Zcos,
            Zsin=st.Zsin,
            Lcos=Lcos,
            Lsin=Lsin,
        )

    axis_reset_coeffs = None

    def _reset_axis_from_boundary(
        st: VMECState,
        *,
        k_guess=None,
        full_reset: bool = False,
        refine_axis_guess: bool = True,
    ) -> VMECState:
        nonlocal axis_reset_coeffs
        if boundary_for_axis is None:
            return st
        ntor = int(static.cfg.ntor)
        raxis_cc = np.zeros((ntor + 1,), dtype=float)
        raxis_cs = np.zeros((ntor + 1,), dtype=float)
        zaxis_cc = np.zeros((ntor + 1,), dtype=float)
        zaxis_cs = np.zeros((ntor + 1,), dtype=float)

        used_state_guess = False
        if k_guess is not None:
            try:
                raxis_cc, raxis_cs, zaxis_cc, zaxis_cs = _recompute_axis_from_state_vmec(
                    static,
                    pr1_even=k_guess.pr1_even,
                    pr1_odd=k_guess.pr1_odd,
                    pz1_even=k_guess.pz1_even,
                    pz1_odd=k_guess.pz1_odd,
                    pru_even=k_guess.pru_even,
                    pru_odd=k_guess.pru_odd,
                    pzu_even=k_guess.pzu_even,
                    pzu_odd=k_guess.pzu_odd,
                    signgs=int(signgs),
                    trig=trig,
                )
                used_state_guess = True
            except Exception:
                used_state_guess = False

        def _state_from_axis_coeffs(
            rcc: np.ndarray,
            rcs: np.ndarray,
            zcc: np.ndarray,
            zcs: np.ndarray,
            *,
            dtype,
        ) -> VMECState:
            scalars_local = dict(indata.scalars)
            scalars_local["RAXIS_CC"] = [float(v) for v in np.ravel(rcc)]
            scalars_local["RAXIS_CS"] = [float(v) for v in np.ravel(rcs)]
            scalars_local["ZAXIS_CC"] = [float(v) for v in np.ravel(zcc)]
            scalars_local["ZAXIS_CS"] = [float(v) for v in np.ravel(zcs)]
            indata_local = type(indata)(scalars=scalars_local, indexed=indata.indexed)
            return initial_guess_from_boundary(
                static,
                boundary_for_axis,
                indata_local,
                dtype=dtype,
                infer_axis_if_missing=False,
            )

        # One refinement pass on the VMEC state-based axis estimate stabilizes
        # non-axis starts where the first guess is still too far off.
        if used_state_guess and bool(refine_axis_guess):
            try:
                st_tmp = _state_from_axis_coeffs(
                    raxis_cc,
                    raxis_cs,
                    zaxis_cc,
                    zaxis_cs,
                    dtype=jnp.asarray(st.Rcos).dtype,
                )
                k_tmp, _, _, _, _, _, _, _ = _compute_forces_iter(
                    st_tmp,
                    include_edge=False,
                    zero_m1=jnp.asarray(1.0, dtype=jnp.asarray(st.Rcos).dtype),
                    constraint_precond_diag=zero_precond_diag,
                    constraint_tcon=zero_tcon,
                    constraint_precond_active=constraint_active_false,
                    constraint_tcon_active=constraint_active_false,
                    iter_idx=None,
                    iter2=1,
                )
                raxis_cc, raxis_cs, zaxis_cc, zaxis_cs = _recompute_axis_from_state_vmec(
                    static,
                    pr1_even=k_tmp.pr1_even,
                    pr1_odd=k_tmp.pr1_odd,
                    pz1_even=k_tmp.pz1_even,
                    pz1_odd=k_tmp.pz1_odd,
                    pru_even=k_tmp.pru_even,
                    pru_odd=k_tmp.pru_odd,
                    pzu_even=k_tmp.pzu_even,
                    pzu_odd=k_tmp.pzu_odd,
                    signgs=int(signgs),
                    trig=trig,
                )
            except Exception:
                pass

        if not used_state_guess:
            axis_vals = _read_axis_coeffs(indata)
            raxis_cc = np.asarray(axis_vals.get("RAXIS_CC", 0.0), dtype=float)
            zaxis_cs = np.asarray(axis_vals.get("ZAXIS_CS", 0.0), dtype=float)
            if raxis_cc.ndim == 0:
                raxis_cc = np.asarray([float(raxis_cc)], dtype=float)
            if zaxis_cs.ndim == 0:
                zaxis_cs = np.asarray([float(zaxis_cs)], dtype=float)
            if raxis_cc.size < ntor + 1:
                raxis_cc = np.pad(raxis_cc, (0, ntor + 1 - raxis_cc.size))
            if zaxis_cs.size < ntor + 1:
                zaxis_cs = np.pad(zaxis_cs, (0, ntor + 1 - zaxis_cs.size))
            raxis_cc, zaxis_cs = _recompute_axis_from_boundary(
                static,
                boundary_for_axis,
                raxis_cc=raxis_cc,
                zaxis_cs=zaxis_cs,
                signgs=int(signgs),
            )

        axis_dump_dir = os.environ.get("VMEC_JAX_DUMP_AXIS_DIR", "").strip()
        if axis_dump_dir:
            try:
                p = Path(axis_dump_dir).expanduser().resolve()
                p.mkdir(parents=True, exist_ok=True)
                out = p / f"axis_reset_ns{int(static.cfg.ns)}.dat"
                with out.open("w", encoding="utf-8") as f:
                    f.write(f"# used_state_guess={int(used_state_guess)}\n")
                    f.write("n raxis_cc raxis_cs zaxis_cc zaxis_cs\n")
                    for n in range(int(static.cfg.ntor) + 1):
                        f.write(
                            f"{n:4d} "
                            f"{float(raxis_cc[n]): .16e} "
                            f"{float(raxis_cs[n]): .16e} "
                            f"{float(zaxis_cc[n]): .16e} "
                            f"{float(zaxis_cs[n]): .16e}\n"
                        )
            except Exception:
                pass

        st_axis = _state_from_axis_coeffs(
            raxis_cc,
            raxis_cs,
            zaxis_cc,
            zaxis_cs,
            dtype=jnp.asarray(st.Rcos).dtype,
        )
        axis_reset_coeffs = (raxis_cc, raxis_cs, zaxis_cc, zaxis_cs)
        if full_reset:
            st_out = st_axis
        else:
            # Preserve non-axis coefficients (including lambda) when resetting axis.
            if getattr(static, "m_is_m0", None) is None:
                mask_m0 = jnp.asarray(np.asarray(static.modes.m, dtype=int) == 0, dtype=jnp.asarray(st.Rcos).dtype)
            else:
                mask_m0 = jnp.asarray(static.m_is_m0, dtype=jnp.asarray(st.Rcos).dtype)
            Rcos = jnp.where(mask_m0[None, :] != 0, jnp.asarray(st_axis.Rcos), jnp.asarray(st.Rcos))
            Rsin = jnp.where(mask_m0[None, :] != 0, jnp.asarray(st_axis.Rsin), jnp.asarray(st.Rsin))
            Zcos = jnp.where(mask_m0[None, :] != 0, jnp.asarray(st_axis.Zcos), jnp.asarray(st.Zcos))
            Zsin = jnp.where(mask_m0[None, :] != 0, jnp.asarray(st_axis.Zsin), jnp.asarray(st.Zsin))
            st_out = VMECState(
                layout=st.layout,
                Rcos=Rcos,
                Rsin=Rsin,
                Zcos=Zcos,
                Zsin=Zsin,
                Lcos=st.Lcos,
                Lsin=st.Lsin,
            )
        return _apply_vmec_lambda_axis_rules(st_out)
    flux = flux_profiles_from_indata(indata, s, signgs=signgs)
    chipf_wout = jnp.asarray(flux.chipf)

    phips = jnp.asarray(flux.phips)
    if phips.shape[0] >= 1:
        phips = phips.at[0].set(0.0)

    from .boundary import boundary_from_indata

    boundary = boundary_from_indata(indata, static.modes)
    r00 = float(np.asarray(boundary.R_cos)[int(idx00)]) if int(idx00) >= 0 else float(np.asarray(boundary.R_cos)[0])
    gamma = float(indata.get_float("GAMMA", 5.0 / 3.0))
    lrfp = bool(indata.get_bool("LRFP", False))
    chips = _half_mesh_from_full_mesh(chipf_wout) if lrfp else None
    mass = _mass_half_mesh_from_indata(
        indata=indata,
        s_full=s,
        phips=phips,
        r00=r00,
        gamma=gamma,
        lrfp=lrfp,
        chips=chips,
    )

    pres = _pressure_half_mesh_from_indata(indata=indata, s_full=s)
    ncurr = int(indata.get_int("NCURR", 0))
    icurv = _icurv_full_mesh_from_indata(indata=indata, s_full=s, signgs=signgs)

    wout_like = _WoutLikeVmecForces(
        nfp=int(static.cfg.nfp),
        mpol=int(static.cfg.mpol),
        ntor=int(static.cfg.ntor),
        lasym=bool(static.cfg.lasym),
        signgs=signgs,
        phipf=jnp.asarray(flux.phipf),
        phips=phips,
        chipf=chipf_wout,
        pres=pres,
        mass=mass,
        gamma=gamma,
        ncurr=ncurr,
        lcurrent=True,
        icurv=icurv,
    )

    trig = getattr(static, "trig_vmec", None)
    if trig is None:
        trig = vmec_trig_tables(
            ntheta=int(static.cfg.ntheta),
            nzeta=int(static.cfg.nzeta),
            nfp=int(wout_like.nfp),
            mmax=int(wout_like.mpol) - 1,
            nmax=int(wout_like.ntor),
            lasym=bool(wout_like.lasym),
            dtype=jnp.asarray(state0.Rcos).dtype,
        )
    else:
        if (
            int(trig.ntheta1) != int(static.cfg.ntheta)
            or int(trig.cosnv.shape[0]) != int(static.cfg.nzeta)
            or int(trig.cosmu.shape[1]) != int(wout_like.mpol)
            or int(trig.cosnv.shape[1]) != int(wout_like.ntor) + 1
        ):
            trig = vmec_trig_tables(
                ntheta=int(static.cfg.ntheta),
                nzeta=int(static.cfg.nzeta),
                nfp=int(wout_like.nfp),
                mmax=int(wout_like.mpol) - 1,
                nmax=int(wout_like.ntor),
                lasym=bool(wout_like.lasym),
                dtype=jnp.asarray(state0.Rcos).dtype,
            )
    modes = static.modes
    m_idx = jnp.asarray(modes.m, dtype=jnp.int32)
    n_idx = jnp.asarray(modes.n, dtype=jnp.int32)
    mscale = jnp.asarray(trig.mscale)
    nscale = jnp.asarray(trig.nscale)
    idx00 = _mode00_index(static.modes)
    lambda_update_scale_j = jnp.asarray(lambda_update_scale, dtype=jnp.asarray(state0.Rcos).dtype)

    # VMEC stores Fourier coefficients in an internal (mscale/nscale) basis and
    # uses `scalxc` to represent odd-m modes in 1/sqrt(s) form. The force pipeline
    # applies `scalxc` after `tomnsps` (see `funct3d.f: gc = gc*scalxc`) so the
    # residual/preconditioner updates operate in the same internal coefficient
    # space as `VMECState`.

    edge_Rcos = jnp.asarray(state0.Rcos)[-1, :]
    edge_Rsin = jnp.asarray(state0.Rsin)[-1, :]
    edge_Zcos = jnp.asarray(state0.Zcos)[-1, :]
    edge_Zsin = jnp.asarray(state0.Zsin)[-1, :]

    constraint_tcon0: float | None = None
    if bool(include_constraint_force):
        constraint_tcon0 = float(indata.get_float("TCON0", 1.0))
    apply_lforbal = bool(indata.get_bool("LFORBAL", False)) if indata is not None else False

    static_key = (
        int(static.cfg.mpol),
        int(static.cfg.ntor),
        int(static.cfg.ntheta),
        int(static.cfg.nzeta),
        int(static.cfg.nfp),
        int(static.cfg.ns),
        bool(static.cfg.lasym),
        _hash_array_bytes(static.modes.m),
        _hash_array_bytes(static.modes.n),
        _hash_array_bytes(static.grid.theta),
        _hash_array_bytes(static.grid.zeta),
    )
    wout_key = (
        int(wout_like.nfp),
        int(wout_like.mpol),
        int(wout_like.ntor),
        bool(wout_like.lasym),
        int(wout_like.signgs),
        _hash_array_bytes(wout_like.phipf),
        _hash_array_bytes(wout_like.phips),
        _hash_array_bytes(wout_like.chipf),
        _hash_array_bytes(wout_like.pres),
        _hash_array_bytes(wout_like.icurv) if getattr(wout_like, "icurv", None) is not None else None,
        float(constraint_tcon0) if constraint_tcon0 is not None else None,
    )
    edge_key = (
        _hash_array_bytes(edge_Rcos),
        _hash_array_bytes(edge_Rsin),
        _hash_array_bytes(edge_Zcos),
        _hash_array_bytes(edge_Zsin),
    )

    def _zero_edge_rz(a):
        a = None if a is None else jnp.asarray(a)
        if a is None:
            return None
        if a.shape[0] < 2:
            return a
        return a.at[-1].set(jnp.zeros_like(a[-1]))

    def _apply_radial_tridi(a, alpha: float):
        if alpha <= 0.0:
            return a
        return _tridi_smooth_dirichlet(jnp.asarray(a), alpha=alpha)

    def _apply_radial_tridi_batched(arrs, alpha: float):
        if alpha <= 0.0:
            return tuple(arrs)
        # Stack directly into (ns, B, ...) to avoid swapaxes.
        stack = jnp.stack(arrs, axis=1)
        smooth = _tridi_smooth_dirichlet(stack, alpha=alpha)
        return tuple(smooth[:, i] for i in range(int(smooth.shape[1])))

    def _tridi_smooth_dirichlet(rhs, *, alpha: float):
        """Dirichlet tridiagonal smoother along s for fixed-point updates."""
        rhs = jnp.asarray(rhs)
        if rhs.ndim < 2:
            raise ValueError(f"expected (ns,...) with ndim>=2, got {rhs.shape}")
        ns = int(rhs.shape[0])
        if rhs.ndim == 2:
            rhs2 = rhs
            orig_shape = None
        else:
            rhs2 = rhs.reshape(ns, -1)
            orig_shape = rhs.shape
        ns = int(rhs2.shape[0])
        if ns < 3:
            return rhs
        alpha = jnp.asarray(alpha, dtype=rhs2.dtype)
        a = -alpha
        b = 1.0 + 2.0 * alpha
        c = -alpha

        x0 = rhs2[0]
        xN = rhs2[-1]
        d = rhs2[1:-1]
        d = d.at[0].add(alpha * x0)
        d = d.at[-1].add(alpha * xN)

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

            (cp_last, dp_last), (cp, dp) = jax.lax.scan(fwd, (cp0, dp0), d[1:])

            def bwd(carry, cp_dp):
                x_next = carry
                cp_i, dp_i = cp_dp
                x_i = dp_i - cp_i * x_next
                return x_i, x_i

            _, x_rev = jax.lax.scan(bwd, dp_last, (cp, dp), reverse=True)
            x_int = jnp.concatenate([x_rev, dp_last[None, :]], axis=0)

        out = jnp.concatenate([x0[None, :], x_int, xN[None, :]], axis=0)
        if orig_shape is not None:
            out = out.reshape(orig_shape)
        return out

    def _metric_surface_precond_from_bcovar(bc):
        """Approximate radial preconditioner scaling from bcovar metrics."""
        guu = jnp.asarray(bc.guu)
        r12 = jnp.asarray(bc.jac.r12)
        bsubu = jnp.asarray(bc.bsubu)
        bsubv = jnp.asarray(bc.bsubv)
        nzeta = int(guu.shape[2])
        w_ang = vmec_wint_from_trig(trig, nzeta=nzeta).astype(guu.dtype)
        w3 = w_ang[None, :, :]

        # R/Z preconditioner proxy: VMEC force-norm denominator integrand.
        rz_denom = jnp.sum((guu * (r12 * r12)) * w3, axis=(1, 2))
        rz_scale = jnp.where(rz_denom > 0.0, 1.0 / jnp.sqrt(rz_denom), 1.0)

        # Lambda preconditioner proxy: VMEC lambda norm denominator integrand.
        l_denom = jnp.sum(((bsubu * bsubu) + (bsubv * bsubv)) * w3, axis=(1, 2))
        l_scale = jnp.where(l_denom > 0.0, 1.0 / jnp.sqrt(l_denom), 1.0)

        # Keep updates bounded and avoid axis/boundary blowups.
        rz_scale = jnp.clip(rz_scale, 1e-4, 1e2)
        l_scale = jnp.clip(l_scale, 1e-4, 1e2)
        return rz_scale, l_scale

    def _pshalf_from_s(s_arr):
        s_arr = np.asarray(s_arr, dtype=float)
        if s_arr.size < 2:
            return np.sqrt(np.maximum(s_arr, 0.0))
        sh = 0.5 * (s_arr[1:] + s_arr[:-1])
        p = np.concatenate([sh[:1], sh], axis=0)
        return np.sqrt(np.maximum(p, 0.0))

    def _pshalf_from_s_jax(s_arr, dtype):
        s_arr = jnp.asarray(s_arr, dtype=dtype)
        if int(s_arr.size) < 2:
            return jnp.sqrt(jnp.maximum(s_arr, jnp.asarray(0.0, dtype=dtype)))
        sh = 0.5 * (s_arr[1:] + s_arr[:-1])
        p = jnp.concatenate([sh[:1], sh], axis=0)
        return jnp.sqrt(jnp.maximum(p, jnp.asarray(0.0, dtype=dtype)))

    def _ptau_minmax_from_k(k) -> tuple[Any | None, Any | None]:
        """Compute VMEC `ptau` min/max (jacobian_par) for sign-change detection."""
        try:
            pru_even = jnp.asarray(getattr(k, "pru_even"))
            pru_odd = jnp.asarray(getattr(k, "pru_odd"))
            pzu_even = jnp.asarray(getattr(k, "pzu_even"))
            pzu_odd = jnp.asarray(getattr(k, "pzu_odd"))
            pr1_even = jnp.asarray(getattr(k, "pr1_even"))
            pr1_odd = jnp.asarray(getattr(k, "pr1_odd"))
            pz1_even = jnp.asarray(getattr(k, "pz1_even"))
            pz1_odd = jnp.asarray(getattr(k, "pz1_odd"))
        except Exception:
            return None, None

    def _ptau_minmax_from_k_jax(k):
        pru_even = jnp.asarray(getattr(k, "pru_even"))
        pru_odd = jnp.asarray(getattr(k, "pru_odd"))
        pzu_even = jnp.asarray(getattr(k, "pzu_even"))
        pzu_odd = jnp.asarray(getattr(k, "pzu_odd"))
        pr1_even = jnp.asarray(getattr(k, "pr1_even"))
        pr1_odd = jnp.asarray(getattr(k, "pr1_odd"))
        pz1_even = jnp.asarray(getattr(k, "pz1_even"))
        pz1_odd = jnp.asarray(getattr(k, "pz1_odd"))

        ns = jnp.asarray(pru_even).shape[0]
        nan_val = jnp.asarray(jnp.nan, dtype=pru_even.dtype)

        def _compute(_):
            pshalf = _pshalf_from_s_jax(s, dtype=pru_even.dtype)
            if int(pshalf.shape[0]) != int(ns):
                pshalf_loc = jnp.resize(pshalf, (ns,))
            else:
                pshalf_loc = pshalf
            hs = jnp.asarray(s[1] - s[0]) if int(jnp.asarray(s).shape[0]) > 1 else jnp.asarray(1.0, dtype=pru_even.dtype)
            ohs = jnp.where(hs != 0.0, 1.0 / hs, jnp.asarray(0.0, dtype=pru_even.dtype))
            dphids = jnp.asarray(0.25, dtype=pru_even.dtype)

            psh = pshalf_loc[1:]
            psh = psh[:, None, None]
            psh_safe = jnp.where(psh != 0.0, psh, jnp.asarray(1.0, dtype=pru_even.dtype))

            pru_e = pru_even[1:]
            pru_o = pru_odd[1:]
            pru_e_m1 = pru_even[:-1]
            pru_o_m1 = pru_odd[:-1]
            pz1_e = pz1_even[1:]
            pz1_o = pz1_odd[1:]
            pz1_e_m1 = pz1_even[:-1]
            pz1_o_m1 = pz1_odd[:-1]
            pzu_e = pzu_even[1:]
            pzu_o = pzu_odd[1:]
            pzu_e_m1 = pzu_even[:-1]
            pzu_o_m1 = pzu_odd[:-1]
            pr1_e = pr1_even[1:]
            pr1_o = pr1_odd[1:]
            pr1_e_m1 = pr1_even[:-1]
            pr1_o_m1 = pr1_odd[:-1]

            ru12 = 0.5 * (pru_e + pru_e_m1 + psh * (pru_o + pru_o_m1))
            pzs = ohs * ((pz1_e - pz1_e_m1) + psh * (pz1_o - pz1_o_m1))
            ptau = ru12 * pzs + dphids * (
                pru_o * pz1_o
                + pru_o_m1 * pz1_o_m1
                + (pru_e * pz1_o + pru_e_m1 * pz1_o_m1) / psh_safe
            )
            pzu12 = 0.5 * (pzu_e + pzu_e_m1 + psh * (pzu_o + pzu_o_m1))
            prs = ohs * ((pr1_e - pr1_e_m1) + psh * (pr1_o - pr1_o_m1))
            ptau = ptau - prs * pzu12 - dphids * (
                pzu_o * pr1_o
                + pzu_o_m1 * pr1_o_m1
                + (pzu_e * pr1_o + pzu_e_m1 * pr1_o_m1) / psh_safe
            )
            return jnp.min(ptau), jnp.max(ptau)

        def _nan(_):
            return nan_val, nan_val

        return jax.lax.cond(ns >= 2, _compute, _nan, operand=None)

    def _ptau_minmax(k):
        if has_jax():
            return _ptau_minmax_from_k_jax(k)
        return _ptau_minmax_from_k(k)
        try:
            ns = int(jnp.asarray(pru_even).shape[0])
            if ns < 2:
                return None, None
            pshalf = _pshalf_from_s_jax(s, dtype=pru_even.dtype)
            if int(pshalf.shape[0]) != ns:
                pshalf = jnp.resize(pshalf, (ns,))
            hs = jnp.asarray(s[1] - s[0]) if int(jnp.asarray(s).shape[0]) > 1 else jnp.asarray(1.0, dtype=pru_even.dtype)
            ohs = jnp.where(hs != 0.0, 1.0 / hs, jnp.asarray(0.0, dtype=pru_even.dtype))
            dphids = jnp.asarray(0.25, dtype=pru_even.dtype)

            psh = pshalf[1:]
            psh = psh[:, None, None]
            psh_safe = jnp.where(psh != 0.0, psh, jnp.asarray(1.0, dtype=pru_even.dtype))

            pru_e = pru_even[1:]
            pru_o = pru_odd[1:]
            pru_e_m1 = pru_even[:-1]
            pru_o_m1 = pru_odd[:-1]
            pz1_e = pz1_even[1:]
            pz1_o = pz1_odd[1:]
            pz1_e_m1 = pz1_even[:-1]
            pz1_o_m1 = pz1_odd[:-1]
            pzu_e = pzu_even[1:]
            pzu_o = pzu_odd[1:]
            pzu_e_m1 = pzu_even[:-1]
            pzu_o_m1 = pzu_odd[:-1]
            pr1_e = pr1_even[1:]
            pr1_o = pr1_odd[1:]
            pr1_e_m1 = pr1_even[:-1]
            pr1_o_m1 = pr1_odd[:-1]

            ru12 = 0.5 * (pru_e + pru_e_m1 + psh * (pru_o + pru_o_m1))
            pzs = ohs * ((pz1_e - pz1_e_m1) + psh * (pz1_o - pz1_o_m1))
            ptau = ru12 * pzs + dphids * (
                pru_o * pz1_o
                + pru_o_m1 * pz1_o_m1
                + (pru_e * pz1_o + pru_e_m1 * pz1_o_m1) / psh_safe
            )
            pzu12 = 0.5 * (pzu_e + pzu_e_m1 + psh * (pzu_o + pzu_o_m1))
            prs = ohs * ((pr1_e - pr1_e_m1) + psh * (pr1_o - pr1_o_m1))
            ptau = ptau - prs * pzu12 - dphids * (
                pzu_o * pr1_o
                + pzu_o_m1 * pr1_o_m1
                + (pzu_e * pr1_o + pzu_e_m1 * pr1_o_m1) / psh_safe
            )
            ptau_min = jnp.min(ptau)
            ptau_max = jnp.max(ptau)
            return ptau_min, ptau_max
        except Exception:
            return None, None

    def _sm_sp_from_s(s_arr):
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

    def _maybe_dump_jacobian_terms(*, k, iter_idx: int) -> None:
        env = os.getenv("VMEC_JAX_DUMP_JACOBIAN_TERMS", "").strip()
        if not env or env in ("0", "false", "no", "False"):
            return
        dump_iter = os.getenv("VMEC_JAX_DUMP_ITER", "").strip()
        if dump_iter:
            try:
                if int(dump_iter) != int(iter_idx):
                    return
            except Exception:
                pass
        outdir = os.getenv("VMEC_JAX_DUMP_DIR", "").strip() or "."
        outpath = Path(outdir).expanduser().resolve()
        outpath.mkdir(parents=True, exist_ok=True)
        fname = outpath / f"jacobian_terms_iter{int(iter_idx)}.dat"

        pr1_even = np.asarray(getattr(k, "pr1_even"))
        pr1_odd = np.asarray(getattr(k, "pr1_odd"))
        pz1_even = np.asarray(getattr(k, "pz1_even"))
        pz1_odd = np.asarray(getattr(k, "pz1_odd"))
        pru_even = np.asarray(getattr(k, "pru_even"))
        pru_odd = np.asarray(getattr(k, "pru_odd"))
        pzu_even = np.asarray(getattr(k, "pzu_even"))
        pzu_odd = np.asarray(getattr(k, "pzu_odd"))
        prv_even = np.asarray(getattr(k, "prv_even"))
        prv_odd = np.asarray(getattr(k, "prv_odd"))
        pzv_even = np.asarray(getattr(k, "pzv_even"))
        pzv_odd = np.asarray(getattr(k, "pzv_odd"))

        ns, ntheta3, nzeta = pr1_even.shape
        pshalf = _pshalf_from_s(np.asarray(s))
        if pshalf.shape[0] != ns:
            pshalf = np.resize(pshalf, (ns,))
        hs = float(np.asarray(s[1] - s[0])) if ns > 1 else 1.0
        ohs = 1.0 / hs if hs != 0.0 else 0.0
        dphids = 0.25

        with fname.open("w", encoding="utf-8") as f:
            f.write("# jacobian term dump\n")
            f.write(f"ns={ns}\n")
            f.write(f"ntheta3={ntheta3}\n")
            f.write(f"nzeta={nzeta}\n")
            f.write("columns: js lt lz pshalf\n")
            f.write(" pru_e pru_o pru_e_m1 pru_o_m1\n")
            f.write(" pz1_e pz1_o pz1_e_m1 pz1_o_m1\n")
            f.write(" pzu_e pzu_o pzu_e_m1 pzu_o_m1\n")
            f.write(" pr1_e pr1_o pr1_e_m1 pr1_o_m1\n")
            f.write(" prv_e prv_o prv_e_m1 prv_o_m1\n")
            f.write(" pzv_e pzv_o pzv_e_m1 pzv_o_m1\n")
            f.write(" ru12 pzs pzu12 prs pr12 ptau\n")
            f.write(" rv12 zv12\n")
            for lt in range(ntheta3):
                for lz in range(nzeta):
                    for j in range(1, ns):
                        jm1 = j - 1
                        psh = pshalf[j]
                        psh_safe = psh if psh != 0.0 else 1.0
                        pru_e = pru_even[j, lt, lz]
                        pru_o = pru_odd[j, lt, lz]
                        pru_e_m1 = pru_even[jm1, lt, lz]
                        pru_o_m1 = pru_odd[jm1, lt, lz]
                        pz1_e = pz1_even[j, lt, lz]
                        pz1_o = pz1_odd[j, lt, lz]
                        pz1_e_m1 = pz1_even[jm1, lt, lz]
                        pz1_o_m1 = pz1_odd[jm1, lt, lz]
                        pzu_e = pzu_even[j, lt, lz]
                        pzu_o = pzu_odd[j, lt, lz]
                        pzu_e_m1 = pzu_even[jm1, lt, lz]
                        pzu_o_m1 = pzu_odd[jm1, lt, lz]
                        pr1_e = pr1_even[j, lt, lz]
                        pr1_o = pr1_odd[j, lt, lz]
                        pr1_e_m1 = pr1_even[jm1, lt, lz]
                        pr1_o_m1 = pr1_odd[jm1, lt, lz]
                        prv_e = prv_even[j, lt, lz]
                        prv_o = prv_odd[j, lt, lz]
                        prv_e_m1 = prv_even[jm1, lt, lz]
                        prv_o_m1 = prv_odd[jm1, lt, lz]
                        pzv_e = pzv_even[j, lt, lz]
                        pzv_o = pzv_odd[j, lt, lz]
                        pzv_e_m1 = pzv_even[jm1, lt, lz]
                        pzv_o_m1 = pzv_odd[jm1, lt, lz]

                        ru12 = 0.5 * (pru_e + pru_e_m1 + psh * (pru_o + pru_o_m1))
                        pzs = ohs * ((pz1_e - pz1_e_m1) + psh * (pz1_o - pz1_o_m1))
                        ptau = ru12 * pzs + dphids * (
                            pru_o * pz1_o
                            + pru_o_m1 * pz1_o_m1
                            + (pru_e * pz1_o + pru_e_m1 * pz1_o_m1) / psh_safe
                        )
                        pzu12 = 0.5 * (pzu_e + pzu_e_m1 + psh * (pzu_o + pzu_o_m1))
                        prs = ohs * ((pr1_e - pr1_e_m1) + psh * (pr1_o - pr1_o_m1))
                        pr12 = 0.5 * (pr1_e + pr1_e_m1 + psh * (pr1_o + pr1_o_m1))
                        ptau = ptau - prs * pzu12 - dphids * (
                            pzu_o * pr1_o
                            + pzu_o_m1 * pr1_o_m1
                            + (pzu_e * pr1_o + pzu_e_m1 * pr1_o_m1) / psh_safe
                        )
                        rv12 = 0.5 * (prv_e + prv_e_m1 + psh * (prv_o + prv_o_m1))
                        zv12 = 0.5 * (pzv_e + pzv_e_m1 + psh * (pzv_o + pzv_o_m1))

                        f.write(
                            f"{j+1:6d}{lt+1:6d}{lz+1:6d}"
                            f"{psh:24.16E}"
                            f"{pru_e:24.16E}{pru_o:24.16E}{pru_e_m1:24.16E}{pru_o_m1:24.16E}"
                            f"{pz1_e:24.16E}{pz1_o:24.16E}{pz1_e_m1:24.16E}{pz1_o_m1:24.16E}"
                            f"{pzu_e:24.16E}{pzu_o:24.16E}{pzu_e_m1:24.16E}{pzu_o_m1:24.16E}"
                            f"{pr1_e:24.16E}{pr1_o:24.16E}{pr1_e_m1:24.16E}{pr1_o_m1:24.16E}"
                            f"{prv_e:24.16E}{prv_o:24.16E}{prv_e_m1:24.16E}{prv_o_m1:24.16E}"
                            f"{pzv_e:24.16E}{pzv_o:24.16E}{pzv_e_m1:24.16E}{pzv_o_m1:24.16E}"
                            f"{ru12:24.16E}{pzs:24.16E}{pzu12:24.16E}{prs:24.16E}{pr12:24.16E}{ptau:24.16E}"
                            f"{rv12:24.16E}{zv12:24.16E}\n"
                        )

    def _maybe_dump_ptau(
        *,
        iter_idx: int,
        ptau_min: float,
        ptau_max: float,
        tau_min_state: float | None,
        tau_max_state: float | None,
        badjac_ptau: bool | None,
        badjac_state: bool | None,
        badjac_used: bool,
        mode: str,
        label: str,
    ) -> None:
        if os.getenv("VMEC_JAX_DUMP_PTAU", "") in ("", "0"):
            return
        dump_dir = os.getenv("VMEC_JAX_DUMP_DIR", "").strip()
        if not dump_dir:
            return
        try:
            path = Path(dump_dir) / "ptau_minmax.log"
            if not path.exists():
                with path.open("w", encoding="utf-8") as f:
                    f.write(
                        "iter label mode ptau_min ptau_max state_min state_max bad_ptau bad_state bad_used\n"
                    )
            with path.open("a", encoding="utf-8") as f:
                f.write(
                    f"{int(iter_idx)} {label} {mode} "
                    f"{float(ptau_min):.16e} {float(ptau_max):.16e} "
                    f"{float(tau_min_state if tau_min_state is not None else float('nan')):.16e} "
                    f"{float(tau_max_state if tau_max_state is not None else float('nan')):.16e} "
                    f"{int(badjac_ptau) if badjac_ptau is not None else -1} "
                    f"{int(badjac_state) if badjac_state is not None else -1} "
                    f"{int(bool(badjac_used))}\n"
                )
        except Exception:
            return

    def _lambda_preconditioner(bc, *, return_faclam: bool = False, return_debug: bool = False):
        from .preconditioner_1d_jax import lambda_preconditioner_cached

        lam_r0scale = float(getattr(trig, "r0scale", 1.0)) if trig is not None else 1.0
        return lambda_preconditioner_cached(
            bc=bc,
            trig=trig,
            s=s,
            cfg=cfg,
            return_faclam=return_faclam,
            return_debug=return_debug,
            r0scale=lam_r0scale,
        )

    def _rz_preconditioner(frzl_in: TomnspsRZL, bc, k):
        from .preconditioner_1d_jax import rz_preconditioner

        return rz_preconditioner(
            frzl_in=frzl_in,
            bc=bc,
            k=k,
            trig=trig,
            s=s,
            cfg=cfg,
        )

    def _compute_forces(
        state: VMECState,
        *,
        include_edge: bool,
        zero_m1: Any,
        constraint_precond_diag: tuple[Any, Any] | None = None,
        constraint_tcon: Any | None = None,
        constraint_precond_active: Any | None = None,
        constraint_tcon_active: Any | None = None,
        iter_idx: int | None = None,
    ):
        k = vmec_forces_rz_from_wout(
            state=state,
            static=static,
            wout=wout_like,
            indata=None,
            constraint_tcon0=constraint_tcon0,
            constraint_tcon=constraint_tcon,
            constraint_precond_diag=constraint_precond_diag,
            constraint_precond_active=constraint_precond_active,
            constraint_tcon_active=constraint_tcon_active,
            use_vmec_synthesis=True,
            trig=trig,
            iter_idx=iter_idx,
        )
        if iter_idx is not None:
            _maybe_dump_bsube(bc=k.bc, static=static, iter_idx=int(iter_idx))
            _maybe_dump_bsube_terms(bc=k.bc, static=static, iter_idx=int(iter_idx))
            _maybe_dump_bsubh(bc=k.bc, static=static, iter_idx=int(iter_idx))
            _maybe_dump_bsubs(
                bc=k.bc,
                state=state,
                static=static,
                trig=trig,
                iter_idx=int(iter_idx),
                kernels=k,
            )
            _maybe_dump_lulv(bc=k.bc, static=static, iter_idx=int(iter_idx), state=state, trig=trig)
            _maybe_dump_jacobian_terms(k=k, iter_idx=int(iter_idx))
            _maybe_dump_precond_inputs(bc=k.bc, trig=trig, static=static, iter_idx=int(iter_idx))
        if iter_idx is not None:
            _maybe_dump_force_kernels(k=k, static=static, iter_idx=int(iter_idx), label="raw")
        mask_pack = None
        if getattr(static, "tomnsps_masks", None) is not None:
            mask_pack = static.tomnsps_masks_edge if bool(include_edge) else static.tomnsps_masks
        frzl = vmec_residual_internal_from_kernels(
            k,
            cfg_ntheta=int(static.cfg.ntheta),
            cfg_nzeta=int(static.cfg.nzeta),
            wout=wout_like,
            trig=trig,
            apply_lforbal=apply_lforbal,
            include_edge=bool(include_edge),
            masks=mask_pack,
        )
        if os.getenv("VMEC_JAX_SCAN_DEBUG_FORCE", "") not in ("", "0"):
            try:
                from jax import debug as _jax_debug  # type: ignore
            except Exception:
                _jax_debug = None  # type: ignore
            if _jax_debug is not None:
                fzsc2_raw = jnp.sum(frzl.fzsc * frzl.fzsc)
                fzcs2_raw = (
                    jnp.sum(frzl.fzcs * frzl.fzcs)
                    if frzl.fzcs is not None
                    else jnp.asarray(0.0, dtype=jnp.asarray(frzl.fzsc).dtype)
                )
                _jax_debug.print(
                    "[scan-debug-raw] fzsc2_raw={fzsc:.6e} fzcs2_raw={fzcs:.6e}",
                    fzsc=fzsc2_raw,
                    fzcs=fzcs2_raw,
                )
        if os.getenv("VMEC_JAX_DUMP_HLO_FORCE_TOMNSPS", "").strip().lower() not in ("", "0", "false", "no"):
            try:
                def _tomnsps_only(k_in):
                    frzl_hlo = vmec_residual_internal_from_kernels(
                        k_in,
                        cfg_ntheta=int(static.cfg.ntheta),
                        cfg_nzeta=int(static.cfg.nzeta),
                        wout=wout_like,
                        trig=trig,
                        apply_lforbal=apply_lforbal,
                        include_edge=bool(include_edge),
                        masks=mask_pack,
                    )
                    return (
                        frzl_hlo.frcc,
                        frzl_hlo.frss,
                        frzl_hlo.fzsc,
                        frzl_hlo.fzcs,
                        frzl_hlo.flsc,
                        frzl_hlo.flcs,
                    )
                _maybe_dump_hlo_kernel(
                    label="tomnsps",
                    fn=_tomnsps_only,
                    args=(k,),
                    kwargs={},
                    static=static,
                    wout_like=wout_like,
                    force=True,
                )
            except Exception:
                pass
        if iter_idx is not None:
            _maybe_dump_tomnsps(frzl=frzl, static=static, iter_idx=int(iter_idx), label="raw")
        if bool(apply_m1_constraints):
            frzl = vmec_apply_m1_constraints(frzl=frzl, lconm1=bool(getattr(static.cfg, "lconm1", True)))
            if os.getenv("VMEC_JAX_SCAN_DEBUG_FORCE", "") not in ("", "0"):
                try:
                    from jax import debug as _jax_debug  # type: ignore
                except Exception:
                    _jax_debug = None  # type: ignore
                if _jax_debug is not None:
                    fzsc2_c = jnp.sum(frzl.fzsc * frzl.fzsc)
                    fzcs2_c = (
                        jnp.sum(frzl.fzcs * frzl.fzcs)
                        if frzl.fzcs is not None
                        else jnp.asarray(0.0, dtype=jnp.asarray(frzl.fzsc).dtype)
                    )
                    _jax_debug.print(
                        "[scan-debug-m1] fzsc2={fzsc:.6e} fzcs2={fzcs:.6e}",
                        fzsc=fzsc2_c,
                        fzcs=fzcs2_c,
                    )
        frzl = vmec_zero_m1_zforce(frzl=frzl, enabled=zero_m1)
        if os.getenv("VMEC_JAX_SCAN_DEBUG_FORCE", "") not in ("", "0"):
            try:
                from jax import debug as _jax_debug  # type: ignore
            except Exception:
                _jax_debug = None  # type: ignore
            if _jax_debug is not None:
                fzsc2_z = jnp.sum(frzl.fzsc * frzl.fzsc)
                fzcs2_z = (
                    jnp.sum(frzl.fzcs * frzl.fzcs)
                    if frzl.fzcs is not None
                    else jnp.asarray(0.0, dtype=jnp.asarray(frzl.fzsc).dtype)
                )
                _jax_debug.print(
                    "[scan-debug-zero] fzsc2={fzsc:.6e} fzcs2={fzcs:.6e}",
                    fzsc=fzsc2_z,
                    fzcs=fzcs2_z,
                )
        frzl = vmec_apply_scalxc_to_tomnsps(frzl=frzl, s=s)
        # Materialize tomnsps blocks after scalxc to avoid XLA aliasing.
        def _nan_guard(x):
            x = jnp.asarray(x)
            return jnp.where(jnp.isnan(x), x, x)
        try:
            from dataclasses import replace as _dc_replace
            frzl = _dc_replace(
                frzl,
                frcc=_nan_guard(frzl.frcc),
                frss=None if frzl.frss is None else _nan_guard(frzl.frss),
                fzsc=_nan_guard(frzl.fzsc),
                fzcs=None if frzl.fzcs is None else _nan_guard(frzl.fzcs),
                flsc=_nan_guard(frzl.flsc),
                flcs=None if frzl.flcs is None else _nan_guard(frzl.flcs),
                frsc=None if getattr(frzl, "frsc", None) is None else _nan_guard(frzl.frsc),
                frcs=None if getattr(frzl, "frcs", None) is None else _nan_guard(frzl.frcs),
                fzcc=None if getattr(frzl, "fzcc", None) is None else _nan_guard(frzl.fzcc),
                fzss=None if getattr(frzl, "fzss", None) is None else _nan_guard(frzl.fzss),
                flcc=None if getattr(frzl, "flcc", None) is None else _nan_guard(frzl.flcc),
                flss=None if getattr(frzl, "flss", None) is None else _nan_guard(frzl.flss),
            )
        except Exception:
            frzl = TomnspsRZL(
                frcc=_nan_guard(frzl.frcc),
                frss=None if frzl.frss is None else _nan_guard(frzl.frss),
                fzsc=_nan_guard(frzl.fzsc),
                fzcs=None if frzl.fzcs is None else _nan_guard(frzl.fzcs),
                flsc=_nan_guard(frzl.flsc),
                flcs=None if frzl.flcs is None else _nan_guard(frzl.flcs),
                frsc=None if getattr(frzl, "frsc", None) is None else _nan_guard(frzl.frsc),
                frcs=None if getattr(frzl, "frcs", None) is None else _nan_guard(frzl.frcs),
                fzcc=None if getattr(frzl, "fzcc", None) is None else _nan_guard(frzl.fzcc),
                fzss=None if getattr(frzl, "fzss", None) is None else _nan_guard(frzl.fzss),
                flcc=None if getattr(frzl, "flcc", None) is None else _nan_guard(frzl.flcc),
                flss=None if getattr(frzl, "flss", None) is None else _nan_guard(frzl.flss),
            )
        z_force_dummy = jnp.sum(frzl.fzsc)
        if frzl.fzcs is not None:
            z_force_dummy = z_force_dummy + jnp.sum(frzl.fzcs)
        if os.getenv("VMEC_JAX_SCAN_DEBUG_FORCE", "") not in ("", "0"):
            try:
                from jax import debug as _jax_debug  # type: ignore
            except Exception:
                _jax_debug = None  # type: ignore
            if _jax_debug is not None:
                fzsc2_s = jnp.sum(frzl.fzsc * frzl.fzsc)
                fzcs2_s = (
                    jnp.sum(frzl.fzcs * frzl.fzcs)
                    if frzl.fzcs is not None
                    else jnp.asarray(0.0, dtype=jnp.asarray(frzl.fzsc).dtype)
                )
                _jax_debug.print(
                    "[scan-debug-scalxc] fzsc2={fzsc:.6e} fzcs2={fzcs:.6e}",
                    fzsc=fzsc2_s,
                    fzcs=fzcs2_s,
                )
        if iter_idx is not None:
            _maybe_dump_gc(frzl=frzl, static=static, iter_idx=int(iter_idx), label="raw")

        # Optionally remove the LCFS contribution from the R/Z force arrays
        # before forming gcr2/gcz2 (lambda always uses the full-domain residual).
        def _mask_edge(frzl_in: TomnspsRZL) -> TomnspsRZL:
            return TomnspsRZL(
                frcc=_zero_edge_rz(frzl_in.frcc),
                frss=_zero_edge_rz(frzl_in.frss),
                fzsc=_zero_edge_rz(frzl_in.fzsc),
                fzcs=_zero_edge_rz(frzl_in.fzcs),
                flsc=frzl_in.flsc,
                flcs=frzl_in.flcs,
                frsc=_zero_edge_rz(getattr(frzl_in, "frsc", None)),
                frcs=_zero_edge_rz(getattr(frzl_in, "frcs", None)),
                fzcc=_zero_edge_rz(getattr(frzl_in, "fzcc", None)),
                fzss=_zero_edge_rz(getattr(frzl_in, "fzss", None)),
                flcc=getattr(frzl_in, "flcc", None),
                flss=getattr(frzl_in, "flss", None),
            )

        if not bool(include_edge):
            frzl = _mask_edge(frzl)

        gcr2, gcz2, gcl2 = vmec_gcx2_from_tomnsps(
            frzl=frzl,
            lconm1=bool(getattr(static.cfg, "lconm1", True)),
            apply_m1_constraints=False,
            include_edge=include_edge,
            apply_scalxc=False,
            s=s,
        )
        z_guard = jnp.where(
            jnp.isnan(z_force_dummy),
            z_force_dummy,
            jnp.asarray(0.0, dtype=jnp.asarray(z_force_dummy).dtype),
        )
        gcz2 = gcz2 + z_guard
        if iter_idx is not None:
            _maybe_dump_gcx2(
                gcr2=gcr2,
                gcz2=gcz2,
                gcl2=gcl2,
                iter_idx=int(iter_idx),
                include_edge=bool(np.asarray(include_edge)),
                ns=int(static.cfg.ns),
            )
        norms_current = vmec_force_norms_from_bcovar_dynamic(bc=k.bc, trig=trig, s=s, signgs=signgs)
        if iter_idx is not None:
            _maybe_dump_scalars(norms=norms_current, iter_idx=int(iter_idx), ns=int(static.cfg.ns))
        rz_scale, l_scale = _metric_surface_precond_from_bcovar(k.bc)
        return k, frzl, gcr2, gcz2, gcl2, rz_scale, l_scale, norms_current

    if os.getenv("VMEC_JAX_DUMP_HLO_DIR", "").strip():
        try:
            def _bcovar_only(st):
                from .vmec_bcovar import vmec_bcovar_half_mesh_from_wout

                return vmec_bcovar_half_mesh_from_wout(
                    state=st,
                    static=static,
                    wout=wout_like,
                    pres=None,
                    use_wout_bsup=False,
                    use_wout_bsub_for_lambda=False,
                    use_wout_bmag_for_bsq=False,
                    use_vmec_synthesis=True,
                    trig=trig,
                )

            _maybe_dump_hlo_kernel(
                label="bcovar",
                fn=_bcovar_only,
                args=(state0,),
                kwargs={},
                static=static,
                wout_like=wout_like,
            )
        except Exception:
            pass
        try:
            from .vmec_forces import vmec_forces_rz_from_wout
            from .vmec_forces import vmec_residual_internal_from_kernels

            k_hlo = vmec_forces_rz_from_wout(
                state=state0,
                static=static,
                wout=wout_like,
                indata=None,
                constraint_tcon0=constraint_tcon0,
                constraint_tcon=None,
                constraint_precond_diag=None,
                constraint_precond_active=None,
                constraint_tcon_active=None,
                use_wout_bsup=False,
                use_vmec_synthesis=True,
                trig=trig,
                iter_idx=None,
            )
            mask_pack_hlo = static.tomnsps_masks if getattr(static, "tomnsps_masks", None) is not None else None

            def _tomnsps_only(k_in):
                frzl = vmec_residual_internal_from_kernels(
                    k_in,
                    cfg_ntheta=int(static.cfg.ntheta),
                    cfg_nzeta=int(static.cfg.nzeta),
                    wout=wout_like,
                    trig=trig,
                    apply_lforbal=apply_lforbal,
                    include_edge=False,
                    masks=mask_pack_hlo,
                )
                return (frzl.frcc, frzl.frss, frzl.fzsc, frzl.fzcs, frzl.flsc, frzl.flcs)

            _maybe_dump_hlo_kernel(
                label="tomnsps",
                fn=_tomnsps_only,
                args=(k_hlo,),
                kwargs={},
                static=static,
                wout_like=wout_like,
            )
        except Exception:
            pass

    _compute_forces_impl = _compute_forces
    compute_cache_key = (
        "compute_forces_v1",
        static_key,
        wout_key,
        int(signgs),
        bool(apply_m1_constraints),
    )
    if jit_forces:
        def _compute_forces_nodump(
            state: VMECState,
            *,
            include_edge: bool,
            zero_m1: Any,
            constraint_precond_diag: tuple[Any, Any] | None = None,
            constraint_tcon: Any | None = None,
            constraint_precond_active: Any | None = None,
            constraint_tcon_active: Any | None = None,
            iter_idx: int | None = None,
        ):
            return _compute_forces_impl(
                state,
                include_edge=include_edge,
                zero_m1=zero_m1,
                constraint_precond_diag=constraint_precond_diag,
                constraint_tcon=constraint_tcon,
                constraint_precond_active=constraint_precond_active,
                constraint_tcon_active=constraint_tcon_active,
                iter_idx=None,
            )

        cached = _COMPUTE_FORCES_CACHE.get(compute_cache_key)
        if cached is None:
            cached = jit(_compute_forces_nodump, static_argnames=("include_edge",))
            _COMPUTE_FORCES_CACHE[compute_cache_key] = cached
        _compute_forces = cached

    if bool(jit_forces) and bool(jit_precompile) and has_jax() and (jax is not None):
        try:
            zero_m1_pre = jnp.asarray(1.0, dtype=dtype_state)
            for include_edge_flag in (False, True):
                _compute_forces.lower(
                    state0,
                    include_edge=include_edge_flag,
                    zero_m1=zero_m1_pre,
                    constraint_precond_diag=zero_precond_diag,
                    constraint_tcon=zero_tcon,
                    constraint_precond_active=constraint_active_false,
                    constraint_tcon_active=constraint_active_false,
                    iter_idx=None,
                ).compile()
        except Exception:
            pass

    if precompile_only:
        empty = np.zeros((0,), dtype=float)
        return SolveVmecResidualResult(
            state=state0,
            n_iter=0,
            w_history=empty,
            fsqr2_history=empty,
            fsqz2_history=empty,
            fsql2_history=empty,
            grad_rms_history=empty,
            step_history=empty,
            diagnostics={"precompile_only": True},
        )

    def _iter_idx_for_dump(it: int | None) -> int | None:
        return None if jit_forces else it

    warmup_iters = int(jit_warmup_iters) if bool(jit_forces) else 0

    def _compute_forces_iter(
        state: VMECState,
        *,
        include_edge: bool,
        zero_m1: Any,
        constraint_precond_diag: tuple[Any, Any] | None = None,
        constraint_tcon: Any | None = None,
        constraint_precond_active: Any | None = None,
        constraint_tcon_active: Any | None = None,
        iter_idx: int | None = None,
        iter2: int | None = None,
    ):
        if warmup_iters > 0 and (iter2 is not None) and (int(iter2) <= warmup_iters):
            if has_jax():
                import jax

                with jax.disable_jit():
                    return _compute_forces_impl(
                    state,
                    include_edge=include_edge,
                    zero_m1=zero_m1,
                    constraint_precond_diag=constraint_precond_diag,
                    constraint_tcon=constraint_tcon,
                    constraint_precond_active=constraint_precond_active,
                    constraint_tcon_active=constraint_tcon_active,
                    iter_idx=iter_idx,
                )
            return _compute_forces_impl(
                state,
                include_edge=include_edge,
                zero_m1=zero_m1,
                constraint_precond_diag=constraint_precond_diag,
                constraint_tcon=constraint_tcon,
                constraint_precond_active=constraint_precond_active,
                constraint_tcon_active=constraint_tcon_active,
                iter_idx=iter_idx,
            )
        return _compute_forces(
            state,
            include_edge=include_edge,
            zero_m1=zero_m1,
            constraint_precond_diag=constraint_precond_diag,
            constraint_tcon=constraint_tcon,
            constraint_precond_active=constraint_precond_active,
            constraint_tcon_active=constraint_tcon_active,
            iter_idx=iter_idx,
        )

    def _fsq_from_norms(norms_in, *, gcr2_in, gcz2_in, gcl2_in):
        fsqr_out = norms_in.r1 * norms_in.fnorm * gcr2_in
        fsqz_out = norms_in.r1 * norms_in.fnorm * gcz2_in
        fsql_out = norms_in.fnormL * gcl2_in
        return fsqr_out, fsqz_out, fsql_out

    mpol = int(static.cfg.mpol)
    ntor = int(static.cfg.ntor)
    nrange = ntor + 1
    nfp = float(static.cfg.nfp)
    ncoeff = int(jnp.asarray(state0.Rcos).shape[1])

    from .vmec_parity import signed_maps_from_modes

    signed_maps = (
        static.signed_maps if getattr(static, "signed_maps", None) is not None else signed_maps_from_modes(static.modes)
    )
    idx_pos = np.asarray(signed_maps.idx_pos, dtype=np.int32)
    idx_neg = np.asarray(signed_maps.idx_neg, dtype=np.int32)
    idx_pos_flat_np = np.asarray(signed_maps.idx_pos_flat, dtype=np.int32)
    idx_neg_flat_np = np.asarray(signed_maps.idx_neg_flat, dtype=np.int32)
    mask_pos_flat_np = np.asarray(signed_maps.mask_pos_flat)
    mask_neg_flat_np = np.asarray(signed_maps.mask_neg_flat)
    idx_pos_safe_np = np.asarray(signed_maps.idx_pos_safe_flat, dtype=np.int32)
    idx_neg_safe_np = np.asarray(signed_maps.idx_neg_safe_flat, dtype=np.int32)

    if getattr(static, "mn_idx_m", None) is not None:
        m_idx = jnp.asarray(np.asarray(static.mn_idx_m, dtype=np.int32))
        n_idx = jnp.asarray(np.asarray(static.mn_idx_n, dtype=np.int32))
        kp_idx = jnp.asarray(np.asarray(static.mn_idx_kp, dtype=np.int32))
        kn_idx_np = np.asarray(static.mn_idx_kn, dtype=np.int32)
        has_kn_np = np.asarray(static.mn_has_kn, dtype=bool) if static.mn_has_kn is not None else (kn_idx_np >= 0)
    else:
        m_idx_list = []
        n_idx_list = []
        kp_idx_list = []
        kn_idx_list = []
        for m_i in range(mpol):
            for n_i in range(nrange):
                kp = int(idx_pos[m_i, n_i])
                if kp < 0:
                    continue
                m_idx_list.append(m_i)
                n_idx_list.append(n_i)
                kp_idx_list.append(kp)
                kn_idx_list.append(int(idx_neg[m_i, n_i]))

        m_idx = jnp.asarray(np.asarray(m_idx_list, dtype=np.int32))
        n_idx = jnp.asarray(np.asarray(n_idx_list, dtype=np.int32))
        kp_idx = jnp.asarray(np.asarray(kp_idx_list, dtype=np.int32))
        kn_idx_np = np.asarray(kn_idx_list, dtype=np.int32)
        has_kn_np = kn_idx_np >= 0
    kn_idx = jnp.asarray(kn_idx_np)
    has_kn = jnp.asarray(has_kn_np)
    has_kn_any = bool(np.any(has_kn_np))
    m0_mask = np.asarray(getattr(static, "m_is_m0", None) if getattr(static, "m_is_m0", None) is not None else (np.asarray(static.modes.m) == 0))
    m0 = jnp.asarray((np.arange(mpol)[:, None] == 0))
    n0 = jnp.asarray((np.arange(nrange)[None, :] == 0))
    from .vmec_parity import _mn_cos_to_signed_cached as _mn_cos_to_signed_block
    from .vmec_parity import _mn_sin_to_signed_cached as _mn_sin_to_signed_block

    def _mn_cos_to_signed(cc, ss):
        cc = jnp.asarray(cc)
        ss = jnp.asarray(ss) if ss is not None else jnp.zeros_like(cc)
        return _mn_cos_to_signed_block(cc, ss, maps=signed_maps, ncoeff=ncoeff)

    def _mn_sin_to_signed(sc, cs):
        sc = jnp.asarray(sc)
        cs = jnp.asarray(cs) if cs is not None else jnp.zeros_like(sc)
        return _mn_sin_to_signed_block(sc, cs, maps=signed_maps, ncoeff=ncoeff)

    if has_jax():
        def _mn_sin_to_signed_batch(sc, cs):
            sc = jnp.asarray(sc)
            cs = jnp.asarray(cs) if cs is not None else jnp.zeros_like(sc)
            return jax.vmap(
                lambda sc_i, cs_i: _mn_sin_to_signed_block(sc_i, cs_i, maps=signed_maps, ncoeff=ncoeff)
            )(sc, cs)
    else:
        def _mn_sin_to_signed_batch(sc, cs):
            sc = jnp.asarray(sc)
            cs = jnp.asarray(cs) if cs is not None else jnp.zeros_like(sc)
            out = [
                _mn_sin_to_signed_block(sc[i], cs[i], maps=signed_maps, ncoeff=ncoeff)
                for i in range(int(sc.shape[0]))
            ]
            return jnp.stack(out, axis=0)

    use_m1_pair_convert = bool(getattr(static.cfg, "lthreed", True)) and bool(getattr(static.cfg, "lconm1", True)) and int(static.cfg.mpol) > 1

    def _m1_internal_to_physical_pair(rss, zcs):
        """Convert VMEC internal m=1 (rss,zcs) pair to physical coefficients."""
        if rss is None and zcs is None:
            return None, None
        if rss is None:
            zcs_arr = jnp.asarray(zcs)
            rss_arr = jnp.zeros_like(zcs_arr)
        else:
            rss_arr = jnp.asarray(rss)
        if zcs is None:
            zcs_arr = jnp.zeros_like(rss_arr)
        else:
            zcs_arr = jnp.asarray(zcs)
        if not use_m1_pair_convert:
            return rss_arr, zcs_arr
        tmp = rss_arr[:, 1, :]
        rss_arr = rss_arr.at[:, 1, :].set(tmp + zcs_arr[:, 1, :])
        zcs_arr = zcs_arr.at[:, 1, :].set(tmp - zcs_arr[:, 1, :])
        return rss_arr, zcs_arr

    scalxc_mn = vmec_scalxc_from_s(s=s, mpol=int(static.cfg.mpol)).astype(jnp.asarray(state0.Rcos).dtype)[:, :, None]
    if not bool(divide_by_scalxc_for_update):
        scalxc_mn = jnp.ones_like(scalxc_mn)

    def _mn_cos_to_signed_physical(cc, ss):
        cc = jnp.asarray(cc) / scalxc_mn
        ss = jnp.asarray(ss) / scalxc_mn if ss is not None else None
        return _mn_cos_to_signed(cc, ss)

    def _mn_sin_to_signed_physical(sc, cs):
        sc = jnp.asarray(sc) / scalxc_mn
        cs = jnp.asarray(cs) / scalxc_mn if cs is not None else None
        return _mn_sin_to_signed(sc, cs)

    def _mn_sin_to_signed_physical_lambda(sc, cs):
        """Map lambda updates onto signed physical coefficients (VMEC scalxc)."""
        sc = jnp.asarray(sc) / scalxc_mn
        cs = jnp.asarray(cs) / scalxc_mn if cs is not None else None
        return _mn_sin_to_signed(sc, cs)

    def _mn_cos_to_signed_physical_lambda(cc, ss):
        """Map asymmetric lambda updates onto signed physical coefficients (VMEC scalxc)."""
        cc = jnp.asarray(cc) / scalxc_mn
        ss = jnp.asarray(ss) / scalxc_mn if ss is not None else None
        return _mn_cos_to_signed(cc, ss)

    def _mn_sin_to_signed_physical_batch(sc, cs):
        sc = jnp.asarray(sc) / scalxc_mn
        if cs is None:
            cs = jnp.zeros_like(sc)
        else:
            cs = jnp.asarray(cs) / scalxc_mn
        return _mn_sin_to_signed_batch(sc, cs)

    def _rz_norm(state: VMECState) -> Any:
        """R/Z norm (exclude R(0,0) offset) in (m,n>=0) storage.

        This is a plain sum-of-squares over geometry Fourier coefficients in
        (m,n>=0) storage, excluding the R(0,0) offset term. For parity with the
        reference executable's norm conventions, do not apply `scalxc` here.
        """
        rpos = jnp.asarray(state.Rcos)[:, kp_idx]
        zpos = jnp.asarray(state.Zsin)[:, kp_idx]
        rneg = jnp.zeros_like(rpos)
        zneg = jnp.zeros_like(zpos)
        if has_kn_any:
            rneg = rneg.at[:, has_kn].set(jnp.asarray(state.Rcos)[:, kn_idx[has_kn]])
            zneg = zneg.at[:, has_kn].set(jnp.asarray(state.Zsin)[:, kn_idx[has_kn]])

        has_kn_mask = has_kn[None, :]
        is_m0 = (m_idx == 0)[None, :]
        rcc = rpos + jnp.where(has_kn_mask, rneg, 0.0)
        zsc = jnp.where(has_kn_mask, zpos + zneg, zpos)
        is_n0 = (n_idx == 0)[None, :]
        # VMEC m=0 uses only (rcc, zcs) for n>0; rss and zsc are canonicalized
        # to zero in internal storage.
        rss = jnp.where(is_n0 | is_m0, 0.0, jnp.where(has_kn_mask, rpos - rneg, 0.0))
        zsc = jnp.where((~is_n0) & is_m0, 0.0, zsc)
        zcs = jnp.where(is_n0, 0.0, jnp.where(has_kn_mask, zneg - zpos, -zpos))
        # Note: VMEC builds fnorm1 directly from the internal xc vector without
        # applying m=1 constraints or mscale/nscale basis normalization.

        # VMEC `bcovar_par` accumulates fnorm1 over l=2..ns (excludes axis).
        sl = slice(1, None)

        include_rcc = ((m_idx > 0) | (n_idx > 0))[None, :].astype(rcc.dtype)
        rz_norm = jnp.sum(zsc[sl] * zsc[sl]) + jnp.sum(include_rcc * (rcc[sl] * rcc[sl]))
        if bool(getattr(static.cfg, "lthreed", True)):
            rz_norm = rz_norm + jnp.sum(rss[sl] * rss[sl]) + jnp.sum(zcs[sl] * zcs[sl])
        if bool(getattr(static.cfg, "lasym", False)):
            # Asymmetric terms: include Rsin/Zcos internal components.
            rs_pos = jnp.asarray(state.Rsin)[:, kp_idx]
            zc_pos = jnp.asarray(state.Zcos)[:, kp_idx]
            rs_neg = jnp.zeros_like(rs_pos)
            zc_neg = jnp.zeros_like(zc_pos)
            if has_kn_any:
                rs_neg = rs_neg.at[:, has_kn].set(jnp.asarray(state.Rsin)[:, kn_idx[has_kn]])
                zc_neg = zc_neg.at[:, has_kn].set(jnp.asarray(state.Zcos)[:, kn_idx[has_kn]])

            # Internal sin/cos blocks from signed coefficients.
            rsc = jnp.where(has_kn_mask, rs_pos + rs_neg, jnp.where(is_n0, rs_pos, jnp.where(is_m0, 0.0, rs_pos)))
            rcs = jnp.where(has_kn_mask, rs_neg - rs_pos, jnp.where(is_n0, 0.0, jnp.where(is_m0, -rs_pos, 0.0)))

            zcc = zc_pos + jnp.where(has_kn_mask, zc_neg, 0.0)
            zss = jnp.where(is_n0 | is_m0, 0.0, jnp.where(has_kn_mask, zc_pos - zc_neg, 0.0))

            rz_norm = rz_norm + jnp.sum(rsc[sl] * rsc[sl]) + jnp.sum(rcs[sl] * rcs[sl])
            rz_norm = rz_norm + jnp.sum(zcc[sl] * zcc[sl]) + jnp.sum(zss[sl] * zss[sl])
        return rz_norm

    def _mode_diag_weights_mn(dtype):
        m = jnp.arange(mpol, dtype=jnp.float64)
        n = jnp.arange(nrange, dtype=jnp.float64) * nfp
        k2 = (m[:, None] * m[:, None]) + (n[None, :] * n[None, :])
        w = (1.0 + k2) ** (-float(mode_diag_exponent))
        return w.astype(dtype)

    # Precompute per-iteration constants once.
    w_mode_mn = _mode_diag_weights_mn(jnp.asarray(state0.Rcos).dtype)
    delta_s = (
        jnp.asarray(s[1] - s[0], dtype=jnp.asarray(state0.Rcos).dtype)
        if int(jnp.asarray(s).shape[0]) > 1
        else jnp.asarray(1.0, dtype=jnp.asarray(state0.Rcos).dtype)
    )

    state = _enforce_fixed_boundary_and_axis(
        state0,
        static,
        edge_Rcos=edge_Rcos,
        edge_Rsin=edge_Rsin,
        edge_Zcos=edge_Zcos,
        edge_Zsin=edge_Zsin,
        enforce_lambda_axis=True,
        idx00=idx00,
    )
    state = _apply_vmec_lambda_axis_rules(state)

    ftol = float(indata.get_float("FTOL", 1e-13)) if ftol is None else float(ftol)
    gamma = float(indata.get_float("GAMMA", 0.0))
    if abs(gamma - 1.0) < 1e-14:
        raise ValueError("GAMMA=1 makes wp/(gamma-1) singular (VMEC objective undefined)")

    stage_prev_fsq_j = None
    if stage_prev_fsq is not None:
        try:
            stage_prev_fsq_j = jnp.asarray(float(stage_prev_fsq), dtype=dtype)
        except Exception:
            stage_prev_fsq_j = None

    def _run_vmec2000_scan(state_init: VMECState) -> SolveVmecResidualResult:
        if backtracking or limit_dt_from_force or limit_update_rms or use_direct_fallback or reference_mode:
            raise ValueError(
                "vmec2000 scan requires backtracking=False, limit_dt_from_force=False, "
                "limit_update_rms=False, use_direct_fallback=False, reference_mode=False."
            )
        if not bool(strict_update):
            raise ValueError("vmec2000 scan requires strict_update=True.")
        if bool(auto_flip_force):
            raise ValueError("vmec2000 scan does not yet support auto_flip_force=True.")

        k_preconditioner_update_interval = 25
        restart_badjac_factor = 0.9
        restart_badprog_factor = 1.03
        vmec2000_fact = 1.0e4
        iter_offset0 = 0
        nstep_screen = int(indata.get_int("NSTEP", 1)) if indata is not None else 1
        if nstep_screen < 1:
            nstep_screen = 1
        # Live scan printing is on by default. To keep compilation caching
        # effective, we use chunked host-side printing unless explicitly
        # disabled.
        scan_print_env = os.getenv("VMEC_JAX_SCAN_PRINT", "1").strip().lower()
        scan_print_mode = os.getenv("VMEC_JAX_SCAN_PRINT_MODE", "debug_callback").strip().lower()
        scan_print_ordered = os.getenv("VMEC_JAX_SCAN_PRINT_ORDERED", "0").strip().lower() not in ("", "0", "false", "no")
        scan_print_chunked_env = os.getenv("VMEC_JAX_SCAN_PRINT_CHUNKED", "1").strip().lower()
        scan_print_chunked = scan_print_chunked_env not in ("", "0", "false", "no")
        scan_light_env = os.getenv("VMEC_JAX_SCAN_LIGHT", "0").strip().lower()
        scan_light = (scan_light_env not in ("", "0", "false", "no")) or bool(light_history)
        scan_minimal_env = os.getenv("VMEC_JAX_SCAN_MINIMAL", "").strip().lower()
        if scan_minimal_env:
            scan_minimal = scan_minimal_env not in ("", "0", "false", "no")
        elif scan_minimal_default is not None:
            scan_minimal = bool(scan_minimal_default)
        else:
            # Quiet runs default to the minimal scan history to reduce host traffic.
            scan_minimal = not bool(verbose)
        if dump_any:
            scan_minimal = False
            scan_light = False
        scan_collect_scalars = not scan_minimal
        scan_collect_print = (
            bool(verbose)
            and bool(vmec2000_control)
            and bool(verbose_vmec2000_table)
            and scan_collect_scalars
        )
        scan_core_env = os.getenv("VMEC_JAX_SCAN_CORE", "0").strip().lower()
        scan_core = scan_core_env not in ("", "0", "false", "no")
        scan_trace_env = os.getenv("VMEC_JAX_SCAN_TRACE", "0").strip().lower()
        scan_trace = scan_trace_env not in ("", "0", "false", "no")
        abort_scan_env = os.getenv("VMEC_JAX_SCAN_ABORT_ON_BADJAC", "0").strip().lower()
        abort_scan_on_badjac = abort_scan_env not in ("", "0", "false", "no")
        scan_precompute_env = os.getenv("VMEC_JAX_SCAN_PRECOND_PRECOMPUTE", "").strip().lower()
        if scan_precompute_env:
            scan_use_precomputed = scan_precompute_env not in ("", "0", "false", "no")
        else:
            scan_use_precomputed = os.getenv("VMEC_JAX_TRIDI_PRECOMPUTE", "0").strip().lower() not in (
                "",
                "0",
                "false",
                "no",
            )
        scan_lax_env = os.getenv("VMEC_JAX_SCAN_PRECOND_LAXTRIDI", "").strip().lower()
        if scan_lax_env:
            scan_use_lax_tridi = scan_lax_env not in ("", "0", "false", "no")
        else:
            scan_use_lax_tridi = os.getenv("VMEC_JAX_TRIDI_SOLVE", "").strip().lower() in (
                "1",
                "true",
                "yes",
                "lax",
                "force",
            )
        dump_timecontrol_scan = os.getenv("VMEC_JAX_DUMP_TIMECONTROL", "") not in ("", "0")
        scan_timecontrol_callback = None
        if dump_timecontrol_scan:
            try:
                from jax.experimental import io_callback as _io_callback

                scan_timecontrol_callback = _io_callback
            except Exception:
                dump_timecontrol_scan = False
                scan_timecontrol_callback = None
        print_in_scan = (
            bool(verbose)
            and bool(vmec2000_control)
            and bool(verbose_vmec2000_table)
            and scan_print_env not in ("", "0", "false", "no")
        )
        if scan_minimal:
            print_in_scan = False
        chunked_print = False
        _jax_debug = None
        _jax_debug_print = None
        if print_in_scan and scan_print_chunked:
            # Avoid host callbacks inside the scan: we'll print per chunk on host.
            chunked_print = True
            print_in_scan = False
        if force_chunked_scan:
            chunked_print = True
            print_in_scan = False
        if print_in_scan:
            try:
                from jax import debug as _jax_debug
                _jax_debug_print = _jax_debug.print
            except Exception:
                print_in_scan = False
        if scan_print_mode not in ("debug_print", "debug_callback", "io_callback"):
            scan_print_mode = "debug_print"
        if scan_print_mode == "io_callback":
            try:
                from jax.experimental import io_callback as _io_callback
            except Exception:
                scan_print_mode = "debug_print"
                _io_callback = None  # type: ignore[assignment]
        scan_trace_ctx = None
        if scan_trace:
            try:
                from jax import profiler as _jax_profiler

                def scan_trace_ctx(label: str):  # type: ignore[misc]
                    return _jax_profiler.TraceAnnotation(label)
            except Exception:
                scan_trace = False
                scan_trace_ctx = None
        def _maybe_trace(label: str):
            if scan_trace and scan_trace_ctx is not None:
                return scan_trace_ctx(label)
            return nullcontext()
        _timecontrol_path = None
        if dump_timecontrol_scan:
            dump_dir = os.getenv("VMEC_JAX_DUMP_DIR", "").strip()
            if dump_dir:
                try:
                    _timecontrol_path = Path(dump_dir) / "time_control_trace.log"
                except Exception:
                    _timecontrol_path = None
            else:
                _timecontrol_path = None
            if _timecontrol_path is None:
                dump_timecontrol_scan = False
                scan_timecontrol_callback = None
        if resume_state is not None:
            try:
                iter_offset0 = int(resume_state.get("iter_offset", iter_offset0))
            except Exception:
                pass
        axis_reset_enabled = bool(vmec2000_control) and (not axis_reset_done) and bool(lmove_axis)
        axis_reset_repeat = False

        def _scan_hist_light(
            fsqr,
            fsqz,
            fsql,
            accepted,
            r00,
            z00,
            w_mhd,
            time_step,
            bad_jacobian,
        ):
            return (
                fsqr,
                fsqz,
                fsql,
                accepted,
                r00,
                z00,
                w_mhd,
                time_step,
                bad_jacobian,
            )

        def _scan_hist_min(fsqr, fsqz, fsql):
            return (fsqr, fsqz, fsql)

        def _should_print_vmec2000_local(iter_idx: int, max_iter_local: int) -> bool:
            if not (bool(verbose) and bool(vmec2000_control) and bool(verbose_vmec2000_table)):
                return False
            if iter_idx <= 1:
                return True
            if iter_idx >= max_iter_local:
                return True
            return (iter_idx % nstep_screen) == 0

        def _print_vmec2000_row_local(
            *,
            iter_idx: int,
            fsqr: float,
            fsqz: float,
            fsql: float,
            delt0r: float,
            r00: float,
            w_mhd: float,
            z00: float | None = None,
        ) -> None:
            if not (bool(verbose) and bool(vmec2000_control) and bool(verbose_vmec2000_table)):
                return
            if bool(cfg.lasym):
                z_val = float("nan") if z00 is None else float(z00)
                print(
                    f"{int(iter_idx):5d}"
                    f"{float(fsqr):10.2E}{float(fsqz):10.2E}{float(fsql):10.2E}"
                    f"{float(r00):11.3E}{z_val:11.3E}{float(delt0r):10.2E}{float(w_mhd):12.4E}",
                    flush=True,
                )
            else:
                print(
                    f"{int(iter_idx):5d}"
                    f"{float(fsqr):10.2E}{float(fsqz):10.2E}{float(fsql):10.2E}"
                    f"{float(r00):11.3E}{float(delt0r):10.2E}{float(w_mhd):12.4E}",
                    flush=True,
                )

        def _fmt_axis_coeff_local(val: float) -> str:
            s = f"{float(val):.16g}"
            if "e" in s:
                s = s.replace("e", "E")
            return s

        def _print_axis_guess_local(raxis_cc, zaxis_cs) -> None:
            try:
                r_line = "      RAXIS_CC =    " + "   ".join(_fmt_axis_coeff_local(v) for v in np.ravel(raxis_cc))
                z_line = "      ZAXIS_CS =    " + "   ".join(_fmt_axis_coeff_local(v) for v in np.ravel(zaxis_cs))
                print("  ---- Improved AXIS Guess ----", flush=True)
                print(r_line, flush=True)
                print(z_line, flush=True)
                print("  -----------------------------", flush=True)
            except Exception:
                pass
        dtype = jnp.asarray(state_init.Rcos).dtype

        def _maybe_dump_timecontrol_scan(*, cond, stage_id, iter2, iter1, fsq, fsq0, res0, res1, time_step, irst):
            if not dump_timecontrol_scan or scan_timecontrol_callback is None or _timecontrol_path is None:
                return jnp.asarray(0, dtype=jnp.int32)

            def _emit(args):
                (iter2_v, iter1_v, fsq_v, fsq0_v, res0_v, res1_v, time_step_v, irst_v, stage_id_v) = args
                try:
                    stage_map = {0: "init", 1: "pre", 2: "checkpoint", 3: "restart"}
                    stage = stage_map.get(int(stage_id_v), "pre")
                    with _timecontrol_path.open("a", encoding="utf-8") as f:
                        f.write(
                            f"{int(iter2_v):8d} {int(iter1_v):8d} "
                            f"{float(fsq_v): .16e} {float(fsq0_v): .16e} "
                            f"{float(res0_v): .16e} {float(res1_v): .16e} "
                            f"{float(time_step_v): .16e} {int(irst_v):3d} {stage}\n"
                        )
                except Exception:
                    pass
                return np.int32(0)

            def _call(_):
                return scan_timecontrol_callback(
                    _emit,
                    jax.ShapeDtypeStruct((), jnp.int32),
                    (
                        iter2,
                        iter1,
                        fsq,
                        fsq0,
                        res0,
                        res1,
                        time_step,
                        irst,
                        stage_id,
                    ),
                    ordered=True,
                )

            return jax.lax.cond(cond, _call, lambda _: jnp.asarray(0, dtype=jnp.int32), operand=None)

        time_step0 = jnp.asarray(float(step_size), dtype=dtype)
        flip_sign0 = jnp.asarray(float(initial_flip_sign), dtype=dtype)
        k_ndamp = 10
        inv_tau0 = jnp.full((k_ndamp,), jnp.asarray(0.15, dtype=dtype) / time_step0)
        fsq_prev0 = jnp.asarray(1.0, dtype=dtype)
        fsq0_prev0 = jnp.asarray(1.0, dtype=dtype)
        res0_0 = jnp.asarray(-1.0, dtype=dtype)
        res1_0 = jnp.asarray(-1.0, dtype=dtype)
        iter1_0 = jnp.asarray(1, dtype=jnp.int32)
        ijacob0 = jnp.asarray(0, dtype=jnp.int32)
        bad_resets0 = jnp.asarray(0, dtype=jnp.int32)
        bad_growth0 = jnp.asarray(0, dtype=jnp.int32)
        fsqz_prev0 = jnp.asarray(1.0, dtype=dtype)
        force_bcovar0 = jnp.asarray(False)

        vRcc0 = jnp.zeros((int(state_init.Rcos.shape[0]), mpol, nrange), dtype=dtype)
        vRss0 = jnp.zeros_like(vRcc0)
        vZsc0 = jnp.zeros_like(vRcc0)
        vZcs0 = jnp.zeros_like(vRcc0)
        vLsc0 = jnp.zeros_like(vRcc0)
        vLcs0 = jnp.zeros_like(vRcc0)
        vRsc0 = jnp.zeros_like(vRcc0)
        vRcs0 = jnp.zeros_like(vRcc0)
        vZcc0 = jnp.zeros_like(vRcc0)
        vZss0 = jnp.zeros_like(vRcc0)
        vLcc0 = jnp.zeros_like(vRcc0)
        vLss0 = jnp.zeros_like(vRcc0)
        r00_prev0 = jnp.asarray(0.0, dtype=dtype)
        z00_prev0 = jnp.asarray(0.0, dtype=dtype)
        w_mhd_prev0 = jnp.asarray(0.0, dtype=dtype)
        state_checkpoint0 = state_init

        if resume_state is not None:
            try:
                time_step0 = jnp.asarray(float(resume_state.get("time_step", time_step0)), dtype=dtype)
            except Exception:
                time_step0 = jnp.asarray(time_step0, dtype=dtype)
            try:
                flip_sign0 = jnp.asarray(float(resume_state.get("flip_sign", flip_sign0)), dtype=dtype)
            except Exception:
                pass
            inv_tau_val = resume_state.get("inv_tau", None)
            if inv_tau_val is not None:
                inv_tau0 = jnp.asarray(inv_tau_val, dtype=dtype)
            else:
                inv_tau0 = jnp.full((k_ndamp,), jnp.asarray(0.15, dtype=dtype) / time_step0)
            try:
                fsq_prev0 = jnp.asarray(float(resume_state.get("fsq_prev", fsq_prev0)), dtype=dtype)
            except Exception:
                pass
            try:
                fsq0_prev0 = jnp.asarray(float(resume_state.get("fsq0_prev", fsq0_prev0)), dtype=dtype)
            except Exception:
                pass
            try:
                res0_0 = jnp.asarray(float(resume_state.get("res0", res0_0)), dtype=dtype)
                res1_0 = jnp.asarray(float(resume_state.get("res1", res1_0)), dtype=dtype)
            except Exception:
                pass
            try:
                iter1_0 = jnp.asarray(int(resume_state.get("iter1", int(iter1_0))), dtype=jnp.int32)
            except Exception:
                pass
            try:
                ijacob0 = jnp.asarray(int(resume_state.get("ijacob", int(ijacob0))), dtype=jnp.int32)
            except Exception:
                pass
            try:
                bad_resets0 = jnp.asarray(int(resume_state.get("bad_resets", int(bad_resets0))), dtype=jnp.int32)
            except Exception:
                pass
            try:
                bad_growth0 = jnp.asarray(int(resume_state.get("bad_growth_streak", int(bad_growth0))), dtype=jnp.int32)
            except Exception:
                pass
            try:
                fsqz_prev0 = jnp.asarray(float(resume_state.get("fsqz_prev", fsqz_prev0)), dtype=dtype)
            except Exception:
                pass
            if "vRcc" in resume_state:
                vRcc0 = jnp.asarray(resume_state["vRcc"], dtype=dtype)
                vRss0 = jnp.asarray(resume_state.get("vRss", vRss0), dtype=dtype)
                vZsc0 = jnp.asarray(resume_state.get("vZsc", vZsc0), dtype=dtype)
                vZcs0 = jnp.asarray(resume_state.get("vZcs", vZcs0), dtype=dtype)
                vLsc0 = jnp.asarray(resume_state.get("vLsc", vLsc0), dtype=dtype)
                vLcs0 = jnp.asarray(resume_state.get("vLcs", vLcs0), dtype=dtype)
                vRsc0 = jnp.asarray(resume_state.get("vRsc", vRsc0), dtype=dtype)
                vRcs0 = jnp.asarray(resume_state.get("vRcs", vRcs0), dtype=dtype)
                vZcc0 = jnp.asarray(resume_state.get("vZcc", vZcc0), dtype=dtype)
                vZss0 = jnp.asarray(resume_state.get("vZss", vZss0), dtype=dtype)
                vLcc0 = jnp.asarray(resume_state.get("vLcc", vLcc0), dtype=dtype)
                vLss0 = jnp.asarray(resume_state.get("vLss", vLss0), dtype=dtype)
            try:
                force_bcovar0 = jnp.asarray(
                    bool(resume_state.get("force_bcovar_update", bool(force_bcovar0))), dtype=bool
                )
            except Exception:
                pass
            if "r00_prev" in resume_state:
                r00_prev0 = jnp.asarray(resume_state.get("r00_prev", r00_prev0), dtype=dtype)
            if "z00_prev" in resume_state:
                z00_prev0 = jnp.asarray(resume_state.get("z00_prev", z00_prev0), dtype=dtype)
            if "w_mhd_prev" in resume_state:
                w_mhd_prev0 = jnp.asarray(resume_state.get("w_mhd_prev", w_mhd_prev0), dtype=dtype)
            state_checkpoint0 = resume_state.get("state_checkpoint", state_checkpoint0)

        def _scale_m1_precond_rhs(frzl_in: TomnspsRZL, mats: dict[str, Any]) -> TomnspsRZL:
            if (not bool(getattr(cfg, "lconm1", True))) or (int(cfg.mpol) <= 1):
                return frzl_in
            dr = jnp.asarray(mats["dr"])
            dz = jnp.asarray(mats["dz"])
            if dr.shape[0] == 0:
                return frzl_in
            sr = -dr[:, 1, 0]
            sz = -dz[:, 1, 0]
            denom = sr + sz
            fac_r = jnp.where(denom != 0.0, sr / denom, jnp.ones_like(sr))
            fac_z = jnp.where(denom != 0.0, sz / denom, jnp.ones_like(sz))

            ns_full = int(jnp.asarray(frzl_in.frcc).shape[0])
            nsolve = min(ns_full, int(sr.shape[0]))
            fac_r_full = jnp.ones((ns_full,), dtype=jnp.asarray(frzl_in.frcc).dtype).at[:nsolve].set(fac_r[:nsolve])
            fac_z_full = jnp.ones((ns_full,), dtype=jnp.asarray(frzl_in.fzsc).dtype).at[:nsolve].set(fac_z[:nsolve])

            frss = frzl_in.frss
            fzcs = frzl_in.fzcs
            frsc = getattr(frzl_in, "frsc", None)
            fzcc = getattr(frzl_in, "fzcc", None)
            if frss is not None:
                frss = jnp.asarray(frss)
                frss = frss.at[:, 1, :].set(frss[:, 1, :] * fac_r_full[:, None])
            if fzcs is not None:
                fzcs = jnp.asarray(fzcs)
                fzcs = fzcs.at[:, 1, :].set(fzcs[:, 1, :] * fac_z_full[:, None])
            if frsc is not None:
                frsc = jnp.asarray(frsc)
                frsc = frsc.at[:, 1, :].set(frsc[:, 1, :] * fac_r_full[:, None])
            if fzcc is not None:
                fzcc = jnp.asarray(fzcc)
                fzcc = fzcc.at[:, 1, :].set(fzcc[:, 1, :] * fac_z_full[:, None])
            return TomnspsRZL(
                frcc=frzl_in.frcc,
                frss=frss,
                fzsc=frzl_in.fzsc,
                fzcs=fzcs,
                flsc=frzl_in.flsc,
                flcs=frzl_in.flcs,
                frsc=frsc,
                frcs=frzl_in.frcs,
                fzcc=fzcc,
                fzss=frzl_in.fzss,
                flcc=getattr(frzl_in, "flcc", None),
                flss=getattr(frzl_in, "flss", None),
            )

        # Avoid nested JIT inside the scan by default; allow opt-in for testing.
        # Some cases benefit from a separately-jitted force kernel.
        scan_jit_env = os.getenv("VMEC_JAX_SCAN_JIT_FORCES")
        if scan_jit_env is None:
            jit_forces_scan = bool(jit_forces)
        else:
            jit_forces_scan = scan_jit_env.strip().lower() not in ("", "0", "false", "no")
        _compute_forces_scan = _compute_forces if jit_forces_scan else _compute_forces_impl
        with _maybe_trace("scan/compute_forces:init"):
            with _maybe_trace("scan/compute_forces:init"):
                k0, frzl0, gcr2_0, gcz2_0, gcl2_0, rz_scale0, l_scale0, norms0 = _compute_forces_scan(
                    state_init,
                    include_edge=False,
                    zero_m1=jnp.asarray(1.0, dtype=dtype),
                    constraint_precond_diag=zero_precond_diag,
                    constraint_tcon=zero_tcon,
                    constraint_precond_active=constraint_active_false,
                    constraint_tcon_active=constraint_active_false,
                    iter_idx=None,
                )
        fsq_phys0_val = None
        try:
            fsqr0 = norms0.r1 * norms0.fnorm * gcr2_0
            fsqz0 = norms0.r1 * norms0.fnorm * gcz2_0
            fsql0 = norms0.fnormL * gcl2_0
            fsq_phys0_val = float(np.asarray(fsqr0 + fsqz0 + fsql0))
        except Exception:
            fsq_phys0_val = None
        bad_jacobian0 = False
        if axis_reset_enabled:
            axis_reset_debug = os.getenv("VMEC_JAX_AXIS_RESET_DEBUG", "").strip().lower() not in ("", "0", "false", "no")
            try:
                ptau_min0, ptau_max0 = _ptau_minmax(k0)
            except Exception:
                ptau_min0, ptau_max0 = None, None
            bad_jacobian_ptau = None
            if (ptau_min0 is not None) and (ptau_max0 is not None):
                try:
                    min_tau_ptau0 = float(np.asarray(ptau_min0))
                    max_tau_ptau0 = float(np.asarray(ptau_max0))
                    ptau_scale0 = max(abs(min_tau_ptau0), abs(max_tau_ptau0))
                    tau_tol_ptau0 = max(abs(ptau_tol), ptau_tol_rel * ptau_scale0)
                    bad_jacobian_ptau = (min_tau_ptau0 < -tau_tol_ptau0) and (max_tau_ptau0 > tau_tol_ptau0)
                except Exception:
                    bad_jacobian_ptau = None
            bad_jacobian_state = False
            if badjac_use_state:
                try:
                    jac0 = vmec_half_mesh_jacobian_from_state(
                        state=state_init,
                        modes=static.modes,
                        trig=trig,
                        s=s,
                        lconm1=bool(getattr(static.cfg, "lconm1", True)),
                        lthreed=bool(getattr(static.cfg, "lthreed", True)),
                        mask_even=getattr(static, "m_is_even", None),
                        mask_odd=getattr(static, "m_is_odd", None),
                    )
                    tau0 = jnp.asarray(jac0.tau)
                    tau0_use = tau0[1:] if int(tau0.shape[0]) > 1 else tau0
                    min_tau_state0 = float(np.asarray(jnp.min(tau0_use)))
                    max_tau_state0 = float(np.asarray(jnp.max(tau0_use)))
                    tau_scale_state0 = max(abs(min_tau_state0), abs(max_tau_state0))
                    tau_tol_state0 = max(1.0e-12, 1.0e-2 * tau_scale_state0)
                    bad_jacobian_state = (min_tau_state0 < -tau_tol_state0) and (max_tau_state0 > tau_tol_state0)
                except Exception:
                    bad_jacobian_state = False

            if bad_jacobian_ptau is None:
                bad_jacobian0 = bad_jacobian_state
            else:
                if badjac_use_state:
                    # Require both ptau and state sign changes to avoid
                    # false-positive axis resets on benign cases.
                    bad_jacobian0 = bool(bad_jacobian_ptau) and bool(bad_jacobian_state)
                else:
                    bad_jacobian0 = bool(bad_jacobian_ptau)
            if bad_jacobian0 and axis_reset_fsq_min > 0.0:
                if (fsq_phys0_val is None) or (not np.isfinite(fsq_phys0_val)):
                    bad_jacobian0 = False
                elif fsq_phys0_val < axis_reset_fsq_min:
                    bad_jacobian0 = False
            if axis_reset_debug:
                try:
                    fsq_debug_val = float("nan") if fsq_phys0_val is None else float(fsq_phys0_val)
                    print(
                        "[axis_reset] fsq0="
                        f"{fsq_debug_val:.6e} "
                        f"axis_reset_fsq_min={axis_reset_fsq_min:.3e} "
                        f"badjac_ptau={bad_jacobian_ptau} badjac_state={bad_jacobian_state} "
                        f"badjac_used={bad_jacobian0}",
                        flush=True,
                    )
                except Exception:
                    pass
        force_axis_reset_init = bool(force_axis_reset) or (
            bool(vmec2000_control) and bool(lmove_axis) and bool(getattr(static.cfg, "lthreed", True)) and axis_reset_always_3d
        )
        if axis_reset_enabled and (bad_jacobian0 or force_axis_reset_init):
            if bool(verbose) and bool(vmec2000_control) and bool(verbose_vmec2000_table):
                if bad_jacobian0 or force_axis_reset_init:
                    print(" INITIAL JACOBIAN CHANGED SIGN!", flush=True)
                print(" TRYING TO IMPROVE INITIAL MAGNETIC AXIS GUESS", flush=True)
            state_init = _reset_axis_from_boundary(state_init, k_guess=k0, full_reset=False, refine_axis_guess=False)
            if bool(verbose) and bool(vmec2000_control) and bool(verbose_vmec2000_table):
                if axis_reset_coeffs is not None:
                    raxis_cc, _raxis_cs, _zaxis_cc, zaxis_cs = axis_reset_coeffs
                    _print_axis_guess_local(raxis_cc, zaxis_cs)
            ijacob0 = jnp.asarray(1, dtype=jnp.int32)
            state_checkpoint0 = state_init
            axis_reset_enabled = False
            axis_reset_repeat = True
            k0, frzl0, gcr2_0, gcz2_0, gcl2_0, rz_scale0, l_scale0, norms0 = _compute_forces_scan(
                state_init,
                include_edge=False,
                zero_m1=jnp.asarray(1.0, dtype=dtype),
                constraint_precond_diag=zero_precond_diag,
                constraint_tcon=zero_tcon,
                constraint_precond_active=constraint_active_false,
                constraint_tcon_active=constraint_active_false,
                iter_idx=None,
            )
        # Axis reset handled before scan; avoid per-iteration callbacks.
        axis_reset_enabled = False
        cache_valid0 = jnp.asarray(False)
        if constraint_tcon0 is None or float(constraint_tcon0) == 0.0:
            cache_precond_diag0 = zero_precond_diag
            cache_tcon0 = zero_tcon
        else:
            from .vmec_constraints import precondn_diag_axd1_from_bcovar

            ard1_0, azd1_0 = precondn_diag_axd1_from_bcovar(
                trig=trig,
                s=s,
                bsq=k0.bc.bsq,
                r12=k0.bc.jac.r12,
                sqrtg=k0.bc.jac.sqrtg,
                ru12=k0.bc.jac.ru12,
                zu12=k0.bc.jac.zu12,
            )
            cache_precond_diag0 = (ard1_0, azd1_0)
            cache_tcon0 = jnp.asarray(k0.tcon)
        cache_norms0 = norms0
        cache_rz_scale0 = rz_scale0
        cache_l_scale0 = l_scale0
        cache_rz_norm0 = _rz_norm(state_init)
        cache_f_norm1_0 = jnp.where(cache_rz_norm0 != 0.0, 1.0 / cache_rz_norm0, jnp.asarray(float("inf"), dtype=dtype))
        from .preconditioner_1d_jax import rz_preconditioner_matrices

        cache_lam_prec0 = _lambda_preconditioner(k0.bc)
        cache_rz_mats0, _jmin0, jmax0 = rz_preconditioner_matrices(
            bc=k0.bc,
            k=k0,
            trig=trig,
            s=s,
            cfg=cfg,
        )
        jmax0 = int(jmax0)

        if resume_state is not None:
            try:
                cache_valid0 = jnp.asarray(bool(resume_state.get("vmec2000_cache_valid", bool(cache_valid0))), dtype=bool)
            except Exception:
                cache_valid0 = jnp.asarray(cache_valid0, dtype=bool)
            if "cache_precond_diag" in resume_state:
                cache_precond_diag0 = resume_state.get("cache_precond_diag", cache_precond_diag0)
            if "cache_tcon" in resume_state:
                cache_tcon0 = resume_state.get("cache_tcon", cache_tcon0)
            if "cache_norms" in resume_state:
                cache_norms0 = resume_state.get("cache_norms", cache_norms0)
            if "cache_rz_scale" in resume_state:
                cache_rz_scale0 = resume_state.get("cache_rz_scale", cache_rz_scale0)
            if "cache_l_scale" in resume_state:
                cache_l_scale0 = resume_state.get("cache_l_scale", cache_l_scale0)
            if "cache_rz_norm" in resume_state:
                try:
                    cache_rz_norm0 = jnp.asarray(resume_state.get("cache_rz_norm", cache_rz_norm0), dtype=dtype)
                except Exception:
                    pass
            if "cache_f_norm1" in resume_state:
                try:
                    cache_f_norm1_0 = jnp.asarray(resume_state.get("cache_f_norm1", cache_f_norm1_0), dtype=dtype)
                except Exception:
                    pass
            if "cache_prec_rz_mats" in resume_state:
                cache_rz_mats0 = resume_state.get("cache_prec_rz_mats", cache_rz_mats0)
            if "cache_prec_lam_prec" in resume_state:
                cache_lam_prec0 = resume_state.get("cache_prec_lam_prec", cache_lam_prec0)

        def _tree_select(cond, t_true, t_false):
            return jax.tree_util.tree_map(lambda a, b: jnp.where(cond, a, b), t_true, t_false)

        ftol_j = jnp.asarray(float(ftol), dtype=dtype)
        scan_fallback_iters_j = jnp.asarray(int(scan_fallback_iters), dtype=jnp.int32)
        scan_fallback_badjac_limit_j = jnp.asarray(int(scan_fallback_badjac_limit), dtype=jnp.int32)
        scan_fallback_accept_frac_j = jnp.asarray(float(scan_fallback_accept_frac), dtype=dtype)
        scan_fallback_fsq_factor_j = jnp.asarray(float(scan_fallback_fsq_factor), dtype=dtype)
        scan_fallback_fsq_abs_j = jnp.asarray(float(scan_fallback_fsq_abs), dtype=dtype)
        scan_fallback_improve_j = jnp.asarray(float(scan_fallback_improve), dtype=dtype)

        scan_jax_debug = _jax_debug
        scan_jax_debug_print = _jax_debug_print
        scan_debug_force = os.getenv("VMEC_JAX_SCAN_DEBUG_FORCE", "") not in ("", "0")
        debug_iter_env = os.getenv("VMEC_JAX_SCAN_DEBUG_ITER", "").strip()
        try:
            scan_debug_iter = int(debug_iter_env) if debug_iter_env else -1
        except Exception:
            scan_debug_iter = -1

        def _scan_step(carry: _ScanCarry, it):
            def _hold_step(carry_hold: _ScanCarry):
                carry_hold_out = carry_hold
                fsqr_h = carry_hold.fsqr_prev_phys
                fsqz_h = carry_hold.fsqz_prev_phys
                fsql_h = carry_hold.fsql_prev_phys
                fsqr1_h = carry_hold.fsqr1_prev
                fsqz1_h = carry_hold.fsqz1_prev
                fsql1_h = carry_hold.fsql1_prev
                accepted_h = jnp.asarray(False)
                zero_m1_h = jnp.asarray(0.0, dtype=dtype)
                include_edge_h = jnp.asarray(False)
                if scan_minimal:
                    return carry_hold_out, _scan_hist_min(fsqr_h, fsqz_h, fsql_h)
                if scan_light:
                    return carry_hold_out, _scan_hist_light(
                        fsqr_h,
                        fsqz_h,
                        fsql_h,
                        accepted_h,
                        carry_hold.r00_prev,
                        carry_hold.z00_prev,
                        carry_hold.w_mhd_prev,
                        carry_hold.time_step,
                        jnp.asarray(False),
                    )
                return carry_hold_out, (
                    fsqr_h,
                    fsqz_h,
                    fsql_h,
                    fsqr1_h,
                    fsqz1_h,
                    fsql1_h,
                    accepted_h,
                    carry_hold.r00_prev,
                    carry_hold.z00_prev,
                    carry_hold.w_mhd_prev,
                    carry_hold.time_step,
                    zero_m1_h,
                    include_edge_h,
                    carry_hold.res0,
                    carry_hold.res1,
                    carry_hold.iter1,
                    jnp.asarray(False),
                    jnp.asarray(0.0, dtype=dtype),
                    jnp.asarray(0.0, dtype=dtype),
                    jnp.asarray(jnp.nan, dtype=dtype),
                    jnp.asarray(jnp.nan, dtype=dtype),
                    jnp.asarray(jnp.nan, dtype=dtype),
                    jnp.asarray(jnp.nan, dtype=dtype),
                    jnp.asarray(False),
                    jnp.asarray(False),
                )

            def _advance_step(carry_adv: _ScanCarry):
                iter2 = jnp.asarray(it + 1, dtype=jnp.int32) + jnp.asarray(carry_adv.iter_offset, dtype=jnp.int32)
                fsq_prev_before = carry_adv.fsq_prev
                fsq0_prev_before = carry_adv.fsq0_prev
                skip_timecontrol = carry_adv.skip_timecontrol
                iter_since_restart = iter2 - carry_adv.iter1
                time_step_report = carry_adv.time_step
                # VMEC `constrain_m1`: zero gcz(m=1) on the first global
                # iteration, and again when the previous fsqz drops below the tolerance.
                zero_m1 = jnp.where(
                    (iter2 < 2) | (carry_adv.fsqz_prev < 1.0e-6),
                    jnp.asarray(1.0, dtype=dtype),
                    jnp.asarray(0.0, dtype=dtype),
                )
                prev_rz_fsq = carry_adv.fsqr_prev_phys + carry_adv.fsqz_prev_phys
                include_edge = (iter_since_restart < 50) & (prev_rz_fsq < jnp.asarray(1.0e-6, dtype=prev_rz_fsq.dtype))

                need_bcovar_update = (~carry_adv.cache_valid) | carry_adv.force_bcovar_update | (
                    ((iter2 - carry_adv.iter1) % k_preconditioner_update_interval) == 0
                )
                use_cached_precond = carry_adv.cache_valid & (~need_bcovar_update)
                constraint_precond_diag = _tree_select(use_cached_precond, carry_adv.cache_precond_diag, zero_precond_diag)
                constraint_tcon_override = jnp.where(use_cached_precond, carry_adv.cache_tcon, zero_tcon)
                constraint_precond_active = use_cached_precond
                constraint_tcon_active = use_cached_precond

                with _maybe_trace("scan/compute_forces"):
                    k, frzl, gcr2, gcz2, gcl2, rz_scale, l_scale, norms_current = _compute_forces_scan(
                        carry_adv.state,
                        include_edge=False,
                        zero_m1=zero_m1,
                        constraint_precond_diag=constraint_precond_diag,
                        constraint_tcon=constraint_tcon_override,
                        constraint_precond_active=constraint_precond_active,
                        constraint_tcon_active=constraint_tcon_active,
                        iter_idx=None,
                    )
                norms_used = jax.lax.cond(
                    use_cached_precond,
                    lambda _: carry_adv.cache_norms,
                    lambda _: norms_current,
                    operand=None,
                )
                fsqr = norms_used.r1 * norms_used.fnorm * gcr2
                fsqz = norms_used.r1 * norms_used.fnorm * gcz2
                fsql = norms_used.fnormL * gcl2
                if scan_debug_force:
                    try:
                        from jax import debug as _jax_debug  # type: ignore
                    except Exception:
                        _jax_debug = None  # type: ignore
                    if _jax_debug is not None:
                        def _dbg(_):
                            fzsc2 = jnp.sum(frzl.fzsc * frzl.fzsc)
                            fzcs2 = jnp.sum(frzl.fzcs * frzl.fzcs) if frzl.fzcs is not None else jnp.asarray(0.0, dtype=fsqz.dtype)
                            fzcs_m1 = (
                                jnp.sum(frzl.fzcs[:, 1, :] * frzl.fzcs[:, 1, :])
                                if frzl.fzcs is not None and int(jnp.asarray(frzl.fzcs).shape[1]) > 1
                                else jnp.asarray(0.0, dtype=fsqz.dtype)
                            )
                            rcos_sum = jnp.sum(carry_adv.state.Rcos)
                            zsin_sum = jnp.sum(carry_adv.state.Zsin)
                            use_cached_flag = jnp.asarray(use_cached_precond, dtype=jnp.int32)
                            need_bcovar_flag = jnp.asarray(need_bcovar_update, dtype=jnp.int32)
                            if _jax_debug_print is not None:
                                _jax_debug_print(
                                "[scan-debug] iter={i} gcr2={gcr:.6e} gcz2={gcz:.6e} fzsc2={fzsc2:.6e} fzcs2={fzcs2:.6e} fzcs_m1={fzcsm1:.6e} rcos_sum={rcsum:.6e} zsin_sum={zssum:.6e} use_cached={uc} need_bcovar={nb} fnorm={fn:.6e} r1={r1:.6e} fsqr={fsqr:.6e} fsqz={fsqz:.6e}",
                                i=iter2,
                                gcr=gcr2,
                                gcz=gcz2,
                                fzsc2=fzsc2,
                                fzcs2=fzcs2,
                                fzcsm1=fzcs_m1,
                                rcsum=rcos_sum,
                                zssum=zsin_sum,
                                uc=use_cached_flag,
                                nb=need_bcovar_flag,
                                fn=norms_used.fnorm,
                                r1=norms_used.r1,
                                fsqr=fsqr,
                                fsqz=fsqz,
                            )
                            return 0
                        _ = jax.lax.cond(iter2 == 1, _dbg, lambda _: 0, operand=None)
                if scan_debug_iter > 0:
                    try:
                        from jax import debug as _jax_debug_state  # type: ignore
                    except Exception:
                        _jax_debug_state = None  # type: ignore
                    if _jax_debug_state is not None:
                        def _dbg_state(_):
                            rcos_sum = jnp.sum(carry_adv.state.Rcos)
                            zsin_sum = jnp.sum(carry_adv.state.Zsin)
                            lsin_sum = jnp.sum(carry_adv.state.Lsin)
                            rcos_ck = jnp.sum(carry_adv.state_checkpoint.Rcos)
                            zsin_ck = jnp.sum(carry_adv.state_checkpoint.Zsin)
                            lsin_ck = jnp.sum(carry_adv.state_checkpoint.Lsin)
                            fsqr_dbg = norms_used.r1 * norms_used.fnorm * gcr2
                            fsqz_dbg = norms_used.r1 * norms_used.fnorm * gcz2
                            fsql_dbg = norms_used.fnormL * gcl2
                            _jax_debug_state.print(
                                "[scan-state] iter={i} rcos_sum={rc:.6e} zsin_sum={zs:.6e} lsin_sum={ls:.6e} "
                                "rcos_ck={rck:.6e} zsin_ck={zck:.6e} lsin_ck={lck:.6e} "
                                "use_cached={uc} need_bcovar={nb} gcr2={gcr:.6e} gcz2={gcz:.6e} gcl2={gcl:.6e} "
                                "fnorm={fn:.6e} r1={r1:.6e} fsqr={fsqr:.6e} fsqz={fsqz:.6e} fsql={fsql:.6e}",
                                i=iter2,
                                rc=rcos_sum,
                                zs=zsin_sum,
                                ls=lsin_sum,
                                rck=rcos_ck,
                                zck=zsin_ck,
                                lck=lsin_ck,
                                uc=jnp.asarray(use_cached_precond, dtype=jnp.int32),
                                nb=jnp.asarray(need_bcovar_update, dtype=jnp.int32),
                                gcr=gcr2,
                                gcz=gcz2,
                                gcl=gcl2,
                                fn=norms_used.fnorm,
                                r1=norms_used.r1,
                                fsqr=fsqr_dbg,
                                fsqz=fsqz_dbg,
                                fsql=fsql_dbg,
                            )
                            return 0
                        _ = jax.lax.cond(iter2 == scan_debug_iter, _dbg_state, lambda _: 0, operand=None)
                conv_now = (fsqr <= ftol_j) & (fsqz <= ftol_j) & (fsql <= ftol_j)
                # Scalars for VMEC-style screen output (sampled on NSTEP cadence + convergence).
                sample_vmec = (
                    (iter2 <= 1) | (iter2 >= int(max_iter)) | ((iter2 % nstep_screen) == 0) | conv_now
                )
                sample_vmec = sample_vmec & jnp.asarray(scan_collect_scalars, dtype=bool)

                def _compute_scalars(_):
                    r00_j = jnp.asarray(k.pr1_even)[0, 0, 0]
                    if bool(cfg.lasym):
                        z00_j = jnp.asarray(k.pz1_even)[0, 0, 0]
                    else:
                        z00_j = jnp.asarray(0.0, dtype=r00_j.dtype)
                    # `norms_current` already reflects the current bcovar state.
                    wb_val = jnp.asarray(norms_current.wb)
                    wp_val = jnp.asarray(norms_current.wp)
                    w_mhd = (wb_val + wp_val / (gamma - 1.0)) * jnp.asarray(float(TWOPI * TWOPI), dtype=wb_val.dtype)
                    return r00_j, z00_j, w_mhd

                def _reuse_scalars(_):
                    return carry_adv.r00_prev, carry_adv.z00_prev, carry_adv.w_mhd_prev

                r00_j, z00_j, w_mhd = jax.lax.cond(sample_vmec, _compute_scalars, _reuse_scalars, operand=None)
                def _refresh_cache(_):
                    if constraint_tcon0 is None or float(constraint_tcon0) == 0.0:
                        cache_precond_diag = zero_precond_diag
                        cache_tcon = zero_tcon
                    else:
                        from .vmec_constraints import precondn_diag_axd1_from_bcovar

                        ard1, azd1 = precondn_diag_axd1_from_bcovar(
                            trig=trig,
                            s=s,
                            bsq=k.bc.bsq,
                            r12=k.bc.jac.r12,
                            sqrtg=k.bc.jac.sqrtg,
                            ru12=k.bc.jac.ru12,
                            zu12=k.bc.jac.zu12,
                        )
                        cache_precond_diag = (ard1, azd1)
                        cache_tcon = jnp.asarray(k.tcon)
                    cache_norms = norms_used
                    cache_rz_scale = rz_scale
                    cache_l_scale = l_scale
                    cache_rz_norm = _rz_norm(carry_adv.state)
                    cache_f_norm1 = jnp.where(
                        cache_rz_norm != 0.0, 1.0 / cache_rz_norm, jnp.asarray(float("inf"), dtype=dtype)
                    )
                    from .preconditioner_1d_jax import rz_preconditioner_matrices

                    cache_lam_prec = _lambda_preconditioner(k.bc)
                    mats, _jmin, jmax = rz_preconditioner_matrices(
                        bc=k.bc,
                        k=k,
                        trig=trig,
                        s=s,
                        cfg=cfg,
                    )
                    # jmax is constant for fixed ns; reuse the static jmax0
                    return (
                        cache_precond_diag,
                        cache_tcon,
                        cache_norms,
                        cache_rz_scale,
                        cache_l_scale,
                        cache_rz_norm,
                        cache_f_norm1,
                        cache_lam_prec,
                        mats,
                        jnp.asarray(True),
                    )

                def _keep_cache(_):
                    return (
                        carry_adv.cache_precond_diag,
                        carry_adv.cache_tcon,
                        carry_adv.cache_norms,
                        carry_adv.cache_rz_scale,
                        carry_adv.cache_l_scale,
                        carry_adv.cache_rz_norm,
                        carry_adv.cache_f_norm1,
                        carry_adv.cache_prec_lam_prec,
                        carry_adv.cache_prec_rz_mats,
                        carry_adv.cache_valid,
                    )

                (
                    cache_precond_diag,
                    cache_tcon,
                    cache_norms,
                    cache_rz_scale,
                    cache_l_scale,
                    cache_rz_norm,
                    cache_f_norm1,
                    cache_lam_prec,
                    cache_rz_mats,
                    cache_valid,
                ) = jax.lax.cond(need_bcovar_update, _refresh_cache, _keep_cache, operand=None)

                frzl_rhs = _scale_m1_precond_rhs(frzl, cache_rz_mats)
                from .preconditioner_1d_jax import rz_preconditioner_apply

                frzl_rz = rz_preconditioner_apply(
                    frzl_in=frzl_rhs,
                    mats=cache_rz_mats,
                    jmax=jmax0,
                    cfg=cfg,
                    use_precomputed=bool(scan_use_precomputed),
                    use_lax_tridi=bool(scan_use_lax_tridi),
                )
                frcc = jnp.asarray(frzl_rz.frcc)
                frss = frzl_rz.frss if frzl_rz.frss is not None else jnp.zeros_like(frcc)
                fzsc = jnp.asarray(frzl_rz.fzsc)
                fzcs = frzl_rz.fzcs if frzl_rz.fzcs is not None else jnp.zeros_like(fzsc)
                flsc = jnp.asarray(frzl_rz.flsc) * jnp.asarray(cache_lam_prec)
                flcs = None if frzl_rz.flcs is None else (jnp.asarray(frzl_rz.flcs) * jnp.asarray(cache_lam_prec))
                frsc = jnp.zeros_like(frcc)
                frcs = jnp.zeros_like(frcc)
                fzcc = jnp.zeros_like(fzsc)
                fzss = jnp.zeros_like(fzsc)
                flcc = jnp.zeros_like(flsc)
                flss = jnp.zeros_like(flsc)
                if getattr(frzl_rz, "frsc", None) is not None:
                    frsc = jnp.asarray(frzl_rz.frsc)
                if getattr(frzl_rz, "frcs", None) is not None:
                    frcs = jnp.asarray(frzl_rz.frcs)
                if getattr(frzl_rz, "fzcc", None) is not None:
                    fzcc = jnp.asarray(frzl_rz.fzcc)
                if getattr(frzl_rz, "fzss", None) is not None:
                    fzss = jnp.asarray(frzl_rz.fzss)
                if getattr(frzl_rz, "flcc", None) is not None:
                    flcc = jnp.asarray(frzl_rz.flcc) * jnp.asarray(cache_lam_prec)
                if getattr(frzl_rz, "flss", None) is not None:
                    flss = jnp.asarray(frzl_rz.flss) * jnp.asarray(cache_lam_prec)

                frcc_u = frcc * w_mode_mn[None, :, :]
                frss_u = frss * w_mode_mn[None, :, :]
                fzsc_u = fzsc * w_mode_mn[None, :, :]
                fzcs_u = fzcs * w_mode_mn[None, :, :]
                flsc_u = flsc * w_mode_mn[None, :, :]
                flcs_u = (flcs if flcs is not None else jnp.zeros_like(flsc_u)) * w_mode_mn[None, :, :]
                frsc_u = frsc * w_mode_mn[None, :, :]
                frcs_u = frcs * w_mode_mn[None, :, :]
                fzcc_u = fzcc * w_mode_mn[None, :, :]
                fzss_u = fzss * w_mode_mn[None, :, :]
                flcc_u = flcc * w_mode_mn[None, :, :]
                flss_u = flss * w_mode_mn[None, :, :]
                if lambda_update_scale != 1.0:
                    flsc_u = flsc_u * lambda_update_scale_j
                    flcs_u = flcs_u * lambda_update_scale_j
                    flcc_u = flcc_u * lambda_update_scale_j
                    flss_u = flss_u * lambda_update_scale_j

                gcr2_p, gcz2_p, gcl2_p = vmec_gcx2_from_tomnsps(
                    frzl=TomnspsRZL(
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
                    ),
                    lconm1=bool(getattr(static.cfg, "lconm1", True)),
                    apply_m1_constraints=False,
                    include_edge=True,
                    apply_scalxc=False,
                    s=s,
                )
                rz_norm = jnp.where(cache_valid, cache_rz_norm, _rz_norm(carry_adv.state))
                f_norm1 = jnp.where(
                    cache_valid,
                    cache_f_norm1,
                    jnp.where(rz_norm != 0.0, 1.0 / rz_norm, jnp.asarray(float("inf"), dtype=dtype)),
                )
                fsqr1 = gcr2_p * f_norm1
                fsqz1 = gcz2_p * f_norm1
                # VMEC excludes the axis surface (js=1) from fsql1 sums.
                gcl2_full = jnp.sum(flsc[1:] * flsc[1:])
                if flcs is not None:
                    gcl2_full = gcl2_full + jnp.sum(flcs[1:] * flcs[1:])
                if getattr(frzl, "flcc", None) is not None:
                    flcc = jnp.asarray(frzl.flcc)
                    gcl2_full = gcl2_full + jnp.sum(flcc[1:] * flcc[1:])
                if getattr(frzl, "flss", None) is not None:
                    flss = jnp.asarray(frzl.flss)
                    gcl2_full = gcl2_full + jnp.sum(flss[1:] * flss[1:])
                fsql1 = gcl2_full * delta_s
                fsq1 = fsqr1 + fsqz1 + fsql1

                fsq0 = fsqr + fsqz + fsql
                if bool(vmec2000_control):
                    ptau_min, ptau_max = _ptau_minmax(k)
                    if (ptau_min is None) or (ptau_max is None):
                        min_tau_ptau = jnp.asarray(jnp.nan, dtype=dtype)
                        max_tau_ptau = jnp.asarray(jnp.nan, dtype=dtype)
                        bad_jacobian_ptau = jnp.asarray(False)
                    else:
                        min_tau_ptau = jnp.asarray(ptau_min)
                        max_tau_ptau = jnp.asarray(ptau_max)
                        tau_tol_ptau = jnp.asarray(abs(ptau_tol), dtype=dtype)
                        bad_jacobian_ptau = (min_tau_ptau < -tau_tol_ptau) & (max_tau_ptau > tau_tol_ptau)
                    ptau_valid = jnp.isfinite(min_tau_ptau) & jnp.isfinite(max_tau_ptau)
                    state_probe = badjac_state_probe & (iter2 <= 2)
                    need_state_jac = badjac_use_state | dump_ptau_state | state_probe | (~ptau_valid) | bad_jacobian_ptau

                    def _state_jacobian():
                        jac_scan = vmec_half_mesh_jacobian_from_state(
                            state=carry_adv.state,
                            modes=static.modes,
                            trig=trig,
                            s=s,
                            lconm1=bool(getattr(static.cfg, "lconm1", True)),
                            lthreed=bool(getattr(static.cfg, "lthreed", True)),
                            mask_even=getattr(static, "m_is_even", None),
                            mask_odd=getattr(static, "m_is_odd", None),
                        )
                        tau = jnp.asarray(jac_scan.tau)
                        tau_use = tau[1:] if int(tau.shape[0]) > 1 else tau
                        min_tau_state = jnp.min(tau_use)
                        max_tau_state = jnp.max(tau_use)
                        tau_scale_state = jnp.maximum(jnp.abs(min_tau_state), jnp.abs(max_tau_state))
                        tau_tol_state = jnp.maximum(
                            jnp.asarray(1.0e-12, dtype=min_tau_state.dtype),
                            jnp.asarray(1.0e-2, dtype=min_tau_state.dtype) * tau_scale_state,
                        )
                        bad_state = (min_tau_state < -tau_tol_state) & (max_tau_state > tau_tol_state)
                        return bad_state, min_tau_state, max_tau_state

                    def _ptau_only():
                        return jnp.asarray(False), jnp.asarray(jnp.nan, dtype=dtype), jnp.asarray(jnp.nan, dtype=dtype)

                    badjac_state, min_tau_state, max_tau_state = jax.lax.cond(
                        need_state_jac, _state_jacobian, _ptau_only
                    )
                    badjac_ptau = bad_jacobian_ptau
                    min_tau = jnp.where(
                        badjac_use_state,
                        min_tau_state,
                        min_tau_ptau,
                    )
                    max_tau = jnp.where(
                        badjac_use_state,
                        max_tau_state,
                        max_tau_ptau,
                    )
                    bad_jacobian = jnp.where(
                        badjac_use_state,
                        badjac_state,
                        badjac_ptau,
                    )
                else:
                    use_state_jac = os.getenv("VMEC_JAX_SCAN_JAC_FROM_STATE", "0").strip().lower() not in (
                        "",
                        "0",
                        "false",
                        "no",
                    )
                    if use_state_jac:
                        jac_scan = vmec_half_mesh_jacobian_from_state(
                            state=carry_adv.state,
                            modes=static.modes,
                            trig=trig,
                            s=s,
                            lconm1=bool(getattr(static.cfg, "lconm1", True)),
                            lthreed=bool(getattr(static.cfg, "lthreed", True)),
                            mask_even=getattr(static, "m_is_even", None),
                            mask_odd=getattr(static, "m_is_odd", None),
                        )
                        tau = jnp.asarray(jac_scan.tau)
                    else:
                        tau = jnp.asarray(k.bc.jac.tau)
                    tau_use = tau[1:] if int(tau.shape[0]) > 1 else tau
                    min_tau = jnp.min(tau_use)
                    max_tau = jnp.max(tau_use)
                    tau_scale = jnp.maximum(jnp.abs(min_tau), jnp.abs(max_tau))
                    tau_tol = jnp.maximum(jnp.asarray(1.0e-12, dtype=tau_scale.dtype), jnp.asarray(1.0e-3, dtype=tau_scale.dtype) * tau_scale)
                    bad_jacobian = (min_tau < -tau_tol) & (max_tau > tau_tol)
                    badjac_ptau = jnp.asarray(False)
                    badjac_state = bad_jacobian
                    min_tau_state = min_tau
                    max_tau_state = max_tau
                    min_tau_ptau = min_tau
                    max_tau_ptau = max_tau
                if os.getenv("VMEC_JAX_SCAN_IGNORE_BADJAC", "") not in ("", "0"):
                    bad_jacobian = jnp.asarray(False)
                # Axis reset handled before entering the scan loop.

                fsq_phys = fsq0
                if bool(vmec2000_control):
                    fsq_phys = jnp.where(
                        bad_jacobian & (iter2 > carry_adv.iter1),
                        fsq0_prev_before,
                        fsq_phys,
                    )
                if bool(vmec2000_control):
                    # VMEC2000 TimeStepControl uses the *previous* preconditioned residual.
                    fsq = carry_adv.fsq_prev
                    fsq_res = fsq
                else:
                    fsq_res = jnp.where(jnp.asarray(reference_mode), fsq_phys, fsq1)
                    fsq = fsq_res
                init_mask = (iter2 == carry_adv.iter1) | (carry_adv.res0 < 0.0) | (carry_adv.res1 < 0.0)

                def _timecontrol_update(_):
                    res0_tc = jnp.where(init_mask, fsq_res, carry_adv.res0)
                    res1_tc = jnp.where(init_mask, fsq_phys, carry_adv.res1)
                    # Important for scan performance: avoid element-wise `where`
                    # over the full state arrays. VMEC-style checkpointing is a
                    # discrete choice (keep old checkpoint vs overwrite with
                    # current state), so model it as a conditional.
                    state_checkpoint_tc = jax.lax.cond(
                        init_mask,
                        lambda _: carry_adv.state,
                        lambda _: carry_adv.state_checkpoint,
                        operand=None,
                    )
                    if bool(vmec2000_control):
                        res0_tc = jnp.minimum(res0_tc, fsq)
                    else:
                        res0_tc = jnp.minimum(res0_tc, fsq_res)
                    res1_tc = jnp.minimum(res1_tc, fsq_phys)
                    if dump_timecontrol_scan:
                        _maybe_dump_timecontrol_scan(
                            cond=init_mask,
                            stage_id=jnp.asarray(0, dtype=jnp.int32),
                            iter2=iter2,
                            iter1=carry_adv.iter1,
                            fsq=fsq,
                            fsq0=fsq_phys,
                            res0=res0_tc,
                            res1=res1_tc,
                            time_step=carry_adv.time_step,
                            irst=jnp.asarray(1, dtype=jnp.int32),
                        )
                        _maybe_dump_timecontrol_scan(
                            cond=jnp.asarray(True),
                            stage_id=jnp.asarray(1, dtype=jnp.int32),
                            iter2=iter2,
                            iter1=carry_adv.iter1,
                            fsq=fsq,
                            fsq0=fsq_phys,
                            res0=res0_tc,
                            res1=res1_tc,
                            time_step=carry_adv.time_step,
                            irst=jnp.asarray(1, dtype=jnp.int32),
                        )
                    checkpoint_mask_tc = (fsq <= res0_tc) & (fsq_phys <= res1_tc) & (~bad_jacobian)
                    state_checkpoint_tc = jax.lax.cond(
                        checkpoint_mask_tc,
                        lambda _: carry_adv.state,
                        lambda _: state_checkpoint_tc,
                        operand=None,
                    )
                    fsqr_checkpoint_tc = jnp.where(checkpoint_mask_tc, fsqr, carry_adv.fsqr_checkpoint)
                    fsqz_checkpoint_tc = jnp.where(checkpoint_mask_tc, fsqz, carry_adv.fsqz_checkpoint)
                    fsql_checkpoint_tc = jnp.where(checkpoint_mask_tc, fsql, carry_adv.fsql_checkpoint)
                    fsqr1_checkpoint_tc = jnp.where(checkpoint_mask_tc, fsqr1, carry_adv.fsqr1_checkpoint)
                    fsqz1_checkpoint_tc = jnp.where(checkpoint_mask_tc, fsqz1, carry_adv.fsqz1_checkpoint)
                    fsql1_checkpoint_tc = jnp.where(checkpoint_mask_tc, fsql1, carry_adv.fsql1_checkpoint)
                    if dump_timecontrol_scan:
                        _maybe_dump_timecontrol_scan(
                            cond=checkpoint_mask_tc,
                            stage_id=jnp.asarray(2, dtype=jnp.int32),
                            iter2=iter2,
                            iter1=carry_adv.iter1,
                            fsq=fsq,
                            fsq0=fsq_phys,
                            res0=res0_tc,
                            res1=res1_tc,
                            time_step=carry_adv.time_step,
                            irst=jnp.asarray(1, dtype=jnp.int32),
                        )
                    return (
                        res0_tc,
                        res1_tc,
                        state_checkpoint_tc,
                        fsqr_checkpoint_tc,
                        fsqz_checkpoint_tc,
                        fsql_checkpoint_tc,
                        fsqr1_checkpoint_tc,
                        fsqz1_checkpoint_tc,
                        fsql1_checkpoint_tc,
                        checkpoint_mask_tc,
                    )

                def _skip_timecontrol_update(_):
                    # Mimic the non-scan skip_time_control behavior: keep res1
                    # and checkpoint, but allow res0 to tighten when fsq1 improves.
                    res0_tc = jnp.where(
                        (fsq1 <= fsq_prev_before) & jnp.isfinite(fsq1),
                        jnp.minimum(carry_adv.res0, fsq1),
                        carry_adv.res0,
                    )
                    return (
                        res0_tc,
                        carry_adv.res1,
                        carry_adv.state_checkpoint,
                        carry_adv.fsqr_checkpoint,
                        carry_adv.fsqz_checkpoint,
                        carry_adv.fsql_checkpoint,
                        carry_adv.fsqr1_checkpoint,
                        carry_adv.fsqz1_checkpoint,
                        carry_adv.fsql1_checkpoint,
                        jnp.asarray(False),
                    )

                (
                    res0,
                    res1,
                    state_checkpoint,
                    fsqr_checkpoint,
                    fsqz_checkpoint,
                    fsql_checkpoint,
                    fsqr1_checkpoint,
                    fsqz1_checkpoint,
                    fsql1_checkpoint,
                    checkpoint_mask,
                ) = jax.lax.cond(skip_timecontrol, _skip_timecontrol_update, _timecontrol_update, operand=None)

                restart_time = jnp.where(
                    skip_timecontrol,
                    jnp.asarray(False),
                    (~bad_jacobian)
                    & ((iter2 - carry_adv.iter1) > 10)
                    & (
                        (fsq > vmec2000_fact * jnp.maximum(res0, 1.0e-30))
                        | (fsq_phys > vmec2000_fact * jnp.maximum(res1, 1.0e-30))
                    ),
                )
                vmecpp_bad_progress = jnp.asarray(False)
                if vmecpp_restart:
                    vmecpp_bad_progress = (
                        (iter2 - carry_adv.iter1) > (k_preconditioner_update_interval // 2)
                    ) & (iter2 > 2 * k_preconditioner_update_interval) & ((fsqr + fsqz) > 1.0e-2)
                stage_spike = jnp.asarray(False)
                if stage_prev_fsq_j is not None:
                    stage_spike = (iter2 == 1) & (fsq_phys > (stage_prev_fsq_j * stage_transition_factor))

                RESTART_NONE = jnp.asarray(0, dtype=jnp.int32)
                RESTART_BADJAC = jnp.asarray(1, dtype=jnp.int32)
                RESTART_STAGE = jnp.asarray(2, dtype=jnp.int32)
                RESTART_BADPROG_VMEC = jnp.asarray(3, dtype=jnp.int32)
                RESTART_TIME = jnp.asarray(4, dtype=jnp.int32)

                if bool(vmec2000_control):
                    # Match non-scan VMEC2000 restart logic (pre_restart_reason).
                    pre_reason = jnp.where(stage_spike, RESTART_STAGE, RESTART_NONE)
                    pre_reason = jnp.where(
                        bad_jacobian & (iter2 > carry_adv.iter1),
                        RESTART_BADJAC,
                        pre_reason,
                    )
                    pre_reason = jnp.where(
                        (pre_reason == RESTART_NONE) & vmecpp_bad_progress,
                        RESTART_BADPROG_VMEC,
                        pre_reason,
                    )
                    do_restart = restart_time | (use_restart_triggers & (pre_reason != RESTART_NONE))
                    restart_reason = jnp.where(restart_time, RESTART_TIME, pre_reason)
                else:
                    restart_badjac = use_restart_triggers & bad_jacobian & (iter2 > carry_adv.iter1)
                    restart_vmecpp = use_restart_triggers & vmecpp_bad_progress
                    do_restart = restart_time | restart_badjac | restart_vmecpp
                    restart_reason = jnp.where(
                        restart_time,
                        jnp.asarray(4, dtype=jnp.int32),
                        jnp.where(restart_badjac, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(3, dtype=jnp.int32)),
                    )
                # Non-scan VMEC2000 skips all restart logic on the iteration
                # immediately after a restart. Mirror that behavior here.
                do_restart = jnp.where(skip_timecontrol, jnp.asarray(False), do_restart)
                restart_reason = jnp.where(skip_timecontrol, RESTART_NONE, restart_reason)
                if dump_timecontrol_scan:
                    irst_restart = jnp.where(
                        restart_reason == RESTART_BADJAC,
                        jnp.asarray(2, dtype=jnp.int32),
                        jnp.where(
                            (restart_reason == RESTART_TIME) | (restart_reason == RESTART_BADPROG_VMEC),
                            jnp.asarray(3, dtype=jnp.int32),
                            jnp.asarray(1, dtype=jnp.int32),
                        ),
                    )
                    _maybe_dump_timecontrol_scan(
                        cond=do_restart,
                        stage_id=jnp.asarray(3, dtype=jnp.int32),
                        iter2=iter2,
                        iter1=carry_adv.iter1,
                        fsq=fsq,
                        fsq0=fsq_phys,
                        res0=res0,
                        res1=res1,
                        time_step=carry_adv.time_step,
                        irst=irst_restart,
                    )

                def _restart_updates(_):
                    time_step = carry_adv.time_step
                    if bool(vmec2000_control):
                        # Map reason -> time-step scaling.
                        time_step = jnp.where(
                            restart_reason == RESTART_BADJAC,
                            restart_badjac_factor * time_step,
                            time_step,
                        )
                        time_step = jnp.where(
                            (restart_reason == RESTART_TIME) | (restart_reason == RESTART_BADPROG_VMEC),
                            time_step / restart_badprog_factor,
                            time_step,
                        )
                        time_step = jnp.where(
                            restart_reason == RESTART_STAGE,
                            time_step * jnp.asarray(stage_transition_scale, dtype=dtype),
                            time_step,
                        )
                    else:
                        time_step = jnp.where(
                            restart_reason == jnp.asarray(1, dtype=jnp.int32),
                            restart_badjac_factor * time_step,
                            time_step / restart_badprog_factor,
                        )
                    time_step = jnp.maximum(time_step, jnp.asarray(1.0e-12, dtype=dtype))
                    if bool(vmec2000_control):
                        iter_offset = carry_adv.iter_offset
                    else:
                        iter_offset = carry_adv.iter_offset - jnp.asarray(1, dtype=jnp.int32)
                    iter1 = iter2
                    inv_tau = jnp.full((k_ndamp,), jnp.asarray(0.15, dtype=dtype) / time_step)
                    fsq_prev = fsq_prev_before
                    if bool(vmec2000_control):
                        ijacob = jnp.where(restart_reason == RESTART_BADJAC, carry_adv.ijacob + 1, carry_adv.ijacob)
                        # VMEC2000 special scaling at ijacob 25/50.
                        ijacob25 = ijacob == jnp.asarray(25, dtype=ijacob.dtype)
                        ijacob50 = ijacob == jnp.asarray(50, dtype=ijacob.dtype)
                        time_step = jnp.where(
                            ijacob25 & (restart_reason == RESTART_BADJAC),
                            jnp.asarray(0.98, dtype=dtype) * jnp.asarray(float(step_size), dtype=dtype),
                            time_step,
                        )
                        time_step = jnp.where(
                            ijacob50 & (restart_reason == RESTART_BADJAC),
                            jnp.asarray(0.96, dtype=dtype) * jnp.asarray(float(step_size), dtype=dtype),
                            time_step,
                        )
                    else:
                        ijacob = jnp.where(restart_reason == jnp.asarray(1, dtype=jnp.int32), carry_adv.ijacob + 1, carry_adv.ijacob)
                    bad_resets = carry_adv.bad_resets + 1
                    bad_growth = jnp.asarray(0, dtype=jnp.int32)
                    force_bcovar_update = jnp.asarray(True)
                    vRcc = jnp.zeros_like(carry_adv.vRcc)
                    vRss = jnp.zeros_like(carry_adv.vRss)
                    vZsc = jnp.zeros_like(carry_adv.vZsc)
                    vZcs = jnp.zeros_like(carry_adv.vZcs)
                    vLsc = jnp.zeros_like(carry_adv.vLsc)
                    vLcs = jnp.zeros_like(carry_adv.vLcs)
                    vRsc = jnp.zeros_like(carry_adv.vRcc)
                    vRcs = jnp.zeros_like(carry_adv.vRcc)
                    vZcc = jnp.zeros_like(carry_adv.vRcc)
                    vZss = jnp.zeros_like(carry_adv.vRcc)
                    vLcc = jnp.zeros_like(carry_adv.vRcc)
                    vLss = jnp.zeros_like(carry_adv.vRcc)
                    state = state_checkpoint
                    return (
                        state,
                        time_step,
                        inv_tau,
                        fsq_prev,
                        vRcc,
                        vRss,
                        vZsc,
                        vZcs,
                        vLsc,
                        vLcs,
                        vRsc,
                        vRcs,
                        vZcc,
                        vZss,
                        vLcc,
                        vLss,
                        iter_offset,
                        iter1,
                        ijacob,
                        bad_resets,
                        bad_growth,
                        force_bcovar_update,
                    )

                def _no_restart_updates(_):
                    return (
                        carry_adv.state,
                        carry_adv.time_step,
                        carry_adv.inv_tau,
                        carry_adv.fsq_prev,
                        carry_adv.vRcc,
                        carry_adv.vRss,
                        carry_adv.vZsc,
                        carry_adv.vZcs,
                        carry_adv.vLsc,
                        carry_adv.vLcs,
                        carry_adv.vRsc,
                        carry_adv.vRcs,
                        carry_adv.vZcc,
                        carry_adv.vZss,
                        carry_adv.vLcc,
                        carry_adv.vLss,
                        carry_adv.iter_offset,
                        carry_adv.iter1,
                        carry_adv.ijacob,
                        carry_adv.bad_resets,
                        carry_adv.bad_growth,
                        jnp.asarray(False),
                    )

                (
                    state_post,
                    time_step_post,
                    inv_tau_post,
                    fsq_prev_post,
                    vRcc_post,
                    vRss_post,
                    vZsc_post,
                    vZcs_post,
                    vLsc_post,
                    vLcs_post,
                    vRsc_post,
                    vRcs_post,
                    vZcc_post,
                    vZss_post,
                    vLcc_post,
                    vLss_post,
                    iter_offset_post,
                    iter1_post,
                    ijacob_post,
                    bad_resets_post,
                    bad_growth_post,
                    force_bcovar_post,
                ) = jax.lax.cond(do_restart, _restart_updates, _no_restart_updates, operand=None)

                fsq0_prev_post = jnp.where(do_restart, fsq0_prev_before, fsq_phys)

                if stage_prev_fsq_j is not None:
                    time_step_post = jnp.where(
                        stage_spike,
                        jnp.maximum(time_step_post * jnp.asarray(stage_transition_scale, dtype=dtype), jnp.asarray(1.0e-12, dtype=dtype)),
                        time_step_post,
                    )
                    inv_tau_post = jnp.where(
                        stage_spike,
                        jnp.full((k_ndamp,), jnp.asarray(0.15, dtype=dtype) / time_step_post),
                        inv_tau_post,
                    )
                    vRcc_post = jnp.where(stage_spike, jnp.zeros_like(vRcc_post), vRcc_post)
                    vRss_post = jnp.where(stage_spike, jnp.zeros_like(vRss_post), vRss_post)
                    vZsc_post = jnp.where(stage_spike, jnp.zeros_like(vZsc_post), vZsc_post)
                    vZcs_post = jnp.where(stage_spike, jnp.zeros_like(vZcs_post), vZcs_post)
                    vLsc_post = jnp.where(stage_spike, jnp.zeros_like(vLsc_post), vLsc_post)
                    vLcs_post = jnp.where(stage_spike, jnp.zeros_like(vLcs_post), vLcs_post)
                    vRsc_post = jnp.where(stage_spike, jnp.zeros_like(vRsc_post), vRsc_post)
                    vRcs_post = jnp.where(stage_spike, jnp.zeros_like(vRcs_post), vRcs_post)
                    vZcc_post = jnp.where(stage_spike, jnp.zeros_like(vZcc_post), vZcc_post)
                    vZss_post = jnp.where(stage_spike, jnp.zeros_like(vZss_post), vZss_post)
                    vLcc_post = jnp.where(stage_spike, jnp.zeros_like(vLcc_post), vLcc_post)
                    vLss_post = jnp.where(stage_spike, jnp.zeros_like(vLss_post), vLss_post)
                    iter1_post = jnp.where(stage_spike, iter2, iter1_post)

                def _restart_payload(_):
                    with _maybe_trace("scan/compute_forces:restart"):
                        k_r, frzl_r, gcr2_r, gcz2_r, gcl2_r, rz_scale_r, l_scale_r, norms_current_r = _compute_forces_scan(
                            state_post,
                            include_edge=False,
                            zero_m1=zero_m1,
                            constraint_precond_diag=zero_precond_diag,
                            constraint_tcon=zero_tcon,
                            constraint_precond_active=constraint_active_false,
                            constraint_tcon_active=constraint_active_false,
                            iter_idx=None,
                        )
                    norms_used_r = norms_current_r
                    fsqr_r = norms_used_r.r1 * norms_used_r.fnorm * gcr2_r
                    fsqz_r = norms_used_r.r1 * norms_used_r.fnorm * gcz2_r
                    fsql_r = norms_used_r.fnormL * gcl2_r

                    rz_norm_r = _rz_norm(state_post)
                    f_norm1_r = jnp.where(
                        rz_norm_r != 0.0, 1.0 / rz_norm_r, jnp.asarray(float("inf"), dtype=dtype)
                    )

                    if constraint_tcon0 is None or float(constraint_tcon0) == 0.0:
                        cache_precond_diag_r = zero_precond_diag
                        cache_tcon_r = zero_tcon
                    else:
                        from .vmec_constraints import precondn_diag_axd1_from_bcovar

                        ard1_r, azd1_r = precondn_diag_axd1_from_bcovar(
                            trig=trig,
                            s=s,
                            bsq=k_r.bc.bsq,
                            r12=k_r.bc.jac.r12,
                            sqrtg=k_r.bc.jac.sqrtg,
                            ru12=k_r.bc.jac.ru12,
                            zu12=k_r.bc.jac.zu12,
                        )
                        cache_precond_diag_r = (ard1_r, azd1_r)
                        cache_tcon_r = jnp.asarray(k_r.tcon)
                    cache_norms_r = norms_used_r
                    cache_rz_scale_r = rz_scale_r
                    cache_l_scale_r = l_scale_r
                    cache_rz_norm_r = rz_norm_r
                    cache_f_norm1_r = f_norm1_r
                    cache_lam_prec_r = _lambda_preconditioner(k_r.bc)
                    from .preconditioner_1d_jax import rz_preconditioner_matrices

                    mats_r, _jmin, jmax_r = rz_preconditioner_matrices(bc=k_r.bc, k=k_r, trig=trig, s=s, cfg=cfg)
                    cache_valid_r = jnp.asarray(True)

                    frzl_rhs_r = _scale_m1_precond_rhs(frzl_r, mats_r)
                    from .preconditioner_1d_jax import rz_preconditioner_apply

                    frzl_rz_r = rz_preconditioner_apply(
                        frzl_in=frzl_rhs_r,
                        mats=mats_r,
                        jmax=jmax0,
                        cfg=cfg,
                        use_precomputed=bool(scan_use_precomputed),
                        use_lax_tridi=bool(scan_use_lax_tridi),
                    )
                    frcc_r = jnp.asarray(frzl_rz_r.frcc)
                    frss_r = frzl_rz_r.frss if frzl_rz_r.frss is not None else jnp.zeros_like(frcc_r)
                    fzsc_r = jnp.asarray(frzl_rz_r.fzsc)
                    fzcs_r = frzl_rz_r.fzcs if frzl_rz_r.fzcs is not None else jnp.zeros_like(fzsc_r)
                    flsc_r = jnp.asarray(frzl_rz_r.flsc) * jnp.asarray(cache_lam_prec_r)
                    flcs_r = None if frzl_rz_r.flcs is None else (jnp.asarray(frzl_rz_r.flcs) * jnp.asarray(cache_lam_prec_r))
                    frsc_r = jnp.zeros_like(frcc_r)
                    frcs_r = jnp.zeros_like(frcc_r)
                    fzcc_r = jnp.zeros_like(fzsc_r)
                    fzss_r = jnp.zeros_like(fzsc_r)
                    flcc_r = jnp.zeros_like(flsc_r)
                    flss_r = jnp.zeros_like(flsc_r)
                    if getattr(frzl_rz_r, "frsc", None) is not None:
                        frsc_r = jnp.asarray(frzl_rz_r.frsc)
                    if getattr(frzl_rz_r, "frcs", None) is not None:
                        frcs_r = jnp.asarray(frzl_rz_r.frcs)
                    if getattr(frzl_rz_r, "fzcc", None) is not None:
                        fzcc_r = jnp.asarray(frzl_rz_r.fzcc)
                    if getattr(frzl_rz_r, "fzss", None) is not None:
                        fzss_r = jnp.asarray(frzl_rz_r.fzss)
                    if getattr(frzl_rz_r, "flcc", None) is not None:
                        flcc_r = jnp.asarray(frzl_rz_r.flcc) * jnp.asarray(cache_lam_prec_r)
                    if getattr(frzl_rz_r, "flss", None) is not None:
                        flss_r = jnp.asarray(frzl_rz_r.flss) * jnp.asarray(cache_lam_prec_r)

                    frzl_pre_r = TomnspsRZL(
                        frcc=frcc_r,
                        frss=frss_r,
                        fzsc=fzsc_r,
                        fzcs=fzcs_r,
                        flsc=flsc_r,
                        flcs=flcs_r,
                        frsc=frsc_r,
                        frcs=frcs_r,
                        fzcc=fzcc_r,
                        fzss=fzss_r,
                        flcc=flcc_r,
                        flss=flss_r,
                    )
                    gcr2_p_r, gcz2_p_r, gcl2_p_r = vmec_gcx2_from_tomnsps(
                        frzl=frzl_pre_r,
                        lconm1=bool(getattr(static.cfg, "lconm1", True)),
                        apply_m1_constraints=False,
                        include_edge=True,
                        apply_scalxc=False,
                        s=s,
                    )
                    fsqr1_r = gcr2_p_r * f_norm1_r
                    fsqz1_r = gcz2_p_r * f_norm1_r
                    gcl2_full_r = jnp.sum(flsc_r[1:] * flsc_r[1:])
                    if flcs_r is not None:
                        flcs_r_arr = jnp.asarray(flcs_r)
                        gcl2_full_r = gcl2_full_r + jnp.sum(flcs_r_arr[1:] * flcs_r_arr[1:])
                    if getattr(frzl_pre_r, "flcc", None) is not None:
                        flcc_r = jnp.asarray(frzl_pre_r.flcc)
                        gcl2_full_r = gcl2_full_r + jnp.sum(flcc_r[1:] * flcc_r[1:])
                    if getattr(frzl_pre_r, "flss", None) is not None:
                        flss_r = jnp.asarray(frzl_pre_r.flss)
                        gcl2_full_r = gcl2_full_r + jnp.sum(flss_r[1:] * flss_r[1:])
                    fsql1_r = gcl2_full_r * delta_s

                    frcc_u_r = frcc_r * w_mode_mn[None, :, :]
                    frss_u_r = frss_r * w_mode_mn[None, :, :]
                    fzsc_u_r = fzsc_r * w_mode_mn[None, :, :]
                    fzcs_u_r = fzcs_r * w_mode_mn[None, :, :]
                    flsc_u_r = flsc_r * w_mode_mn[None, :, :]
                    flcs_u_r = (flcs_r if flcs_r is not None else jnp.zeros_like(flsc_u_r)) * w_mode_mn[None, :, :]
                    frsc_u_r = frsc_r * w_mode_mn[None, :, :]
                    frcs_u_r = frcs_r * w_mode_mn[None, :, :]
                    fzcc_u_r = fzcc_r * w_mode_mn[None, :, :]
                    fzss_u_r = fzss_r * w_mode_mn[None, :, :]
                    flcc_u_r = flcc_r * w_mode_mn[None, :, :]
                    flss_u_r = flss_r * w_mode_mn[None, :, :]
                    if lambda_update_scale != 1.0:
                        flsc_u_r = flsc_u_r * lambda_update_scale_j
                        flcs_u_r = flcs_u_r * lambda_update_scale_j
                        flcc_u_r = flcc_u_r * lambda_update_scale_j
                        flss_u_r = flss_u_r * lambda_update_scale_j

                    return (
                        frcc_u_r,
                        frss_u_r,
                        fzsc_u_r,
                        fzcs_u_r,
                        flsc_u_r,
                        flcs_u_r,
                        frsc_u_r,
                        frcs_u_r,
                        fzcc_u_r,
                        fzss_u_r,
                        flcc_u_r,
                        flss_u_r,
                        fsqr_r,
                        fsqz_r,
                        fsql_r,
                        fsqr1_r,
                        fsqz1_r,
                        fsql1_r,
                        cache_precond_diag_r,
                        cache_tcon_r,
                        cache_norms_r,
                        cache_rz_scale_r,
                        cache_l_scale_r,
                        cache_rz_norm_r,
                        cache_f_norm1_r,
                        mats_r,
                        cache_lam_prec_r,
                        cache_valid_r,
                    )

                def _current_payload(_):
                    return (
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
                        fsqr,
                        fsqz,
                        fsql,
                        fsqr1,
                        fsqz1,
                        fsql1,
                        cache_precond_diag,
                        cache_tcon,
                        cache_norms,
                        cache_rz_scale,
                        cache_l_scale,
                        cache_rz_norm,
                        cache_f_norm1,
                        cache_rz_mats,
                        cache_lam_prec,
                        cache_valid,
                    )

                do_restart_payload = do_restart
                (
                    frcc_u_use,
                    frss_u_use,
                    fzsc_u_use,
                    fzcs_u_use,
                    flsc_u_use,
                    flcs_u_use,
                    frsc_u_use,
                    frcs_u_use,
                    fzcc_u_use,
                    fzss_u_use,
                    flcc_u_use,
                    flss_u_use,
                    fsqr_use,
                    fsqz_use,
                    fsql_use,
                    fsqr1_use,
                    fsqz1_use,
                    fsql1_use,
                    cache_precond_diag_use,
                    cache_tcon_use,
                    cache_norms_use,
                    cache_rz_scale_use,
                    cache_l_scale_use,
                    cache_rz_norm_use,
                    cache_f_norm1_use,
                    cache_rz_mats_use,
                    cache_lam_prec_use,
                    cache_valid_use,
                ) = jax.lax.cond(do_restart_payload, _restart_payload, _current_payload, operand=None)

                frcc_u = frcc_u_use
                frss_u = frss_u_use
                fzsc_u = fzsc_u_use
                fzcs_u = fzcs_u_use
                flsc_u = flsc_u_use
                flcs_u = flcs_u_use
                frsc_u = frsc_u_use
                frcs_u = frcs_u_use
                fzcc_u = fzcc_u_use
                fzss_u = fzss_u_use
                flcc_u = flcc_u_use
                flss_u = flss_u_use
                fsqr = fsqr_use
                fsqz = fsqz_use
                fsql = fsql_use
                fsqr1 = fsqr1_use
                fsqz1 = fsqz1_use
                fsql1 = fsql1_use
                fsq1 = fsqr1 + fsqz1 + fsql1

                def _accept_step(_):
                    inv_tau_reset = jnp.full((k_ndamp,), jnp.asarray(0.15, dtype=dtype) / time_step_post)
                    invtau_num = jnp.where(
                        fsq1 == 0.0,
                        0.0,
                        jnp.minimum(jnp.abs(jnp.log(fsq1 / jnp.maximum(fsq_prev_post, 1.0e-30))), 0.15),
                    )
                    inv_tau_next = jnp.concatenate([inv_tau_post[1:], invtau_num[None] / time_step_post], axis=0)
                    inv_tau = jnp.where(iter2 == iter1_post, inv_tau_reset, inv_tau_next)
                    fsq_prev = fsq1
                    otav = jnp.sum(inv_tau) / float(k_ndamp)
                    dtau = time_step_post * otav / 2.0
                    b1 = 1.0 - dtau
                    fac = 1.0 / (1.0 + dtau)
                    force_scale = time_step_post
                    vRcc = fac * (b1 * vRcc_post + force_scale * (flip_sign0 * frcc_u))
                    vRss = fac * (b1 * vRss_post + force_scale * (flip_sign0 * frss_u))
                    vRsc = fac * (b1 * vRsc_post + force_scale * (flip_sign0 * frsc_u))
                    vRcs = fac * (b1 * vRcs_post + force_scale * (flip_sign0 * frcs_u))
                    vZsc = fac * (b1 * vZsc_post + force_scale * (flip_sign0 * fzsc_u))
                    vZcs = fac * (b1 * vZcs_post + force_scale * (flip_sign0 * fzcs_u))
                    vZcc = fac * (b1 * vZcc_post + force_scale * (flip_sign0 * fzcc_u))
                    vZss = fac * (b1 * vZss_post + force_scale * (flip_sign0 * fzss_u))
                    vLsc = fac * (b1 * vLsc_post + force_scale * (flip_sign0 * flsc_u))
                    vLcs = fac * (b1 * vLcs_post + force_scale * (flip_sign0 * flcs_u))
                    vLcc = fac * (b1 * vLcc_post + force_scale * (flip_sign0 * flcc_u))
                    vLss = fac * (b1 * vLss_post + force_scale * (flip_sign0 * flss_u))
                    dR = time_step_post * _mn_cos_to_signed_physical(vRcc, vRss)
                    dZ = time_step_post * _mn_sin_to_signed_physical(vZsc, vZcs)
                    dL = time_step_post * _mn_sin_to_signed_physical_lambda(vLsc, vLcs)
                    dR_sin = time_step_post * _mn_sin_to_signed_physical(vRsc, vRcs)
                    dZ_cos = time_step_post * _mn_cos_to_signed_physical(vZcc, vZss)
                    dL_cos = time_step_post * _mn_cos_to_signed_physical_lambda(vLcc, vLss)
                    if not bool(cfg.lasym):
                        dR_sin = jnp.zeros_like(dR_sin)
                        dZ_cos = jnp.zeros_like(dZ_cos)
                        dL_cos = jnp.zeros_like(dL_cos)
                    state_new = VMECState(
                        layout=state_post.layout,
                        Rcos=jnp.asarray(state_post.Rcos) + dR,
                        Rsin=jnp.asarray(state_post.Rsin) + dR_sin,
                        Zcos=jnp.asarray(state_post.Zcos) + dZ_cos,
                        Zsin=jnp.asarray(state_post.Zsin) + dZ,
                        Lcos=jnp.asarray(state_post.Lcos) + dL_cos,
                        Lsin=jnp.asarray(state_post.Lsin) + dL,
                    )
                    state_new = _enforce_fixed_boundary_and_axis(
                        state_new,
                        static,
                        edge_Rcos=edge_Rcos,
                        edge_Rsin=edge_Rsin,
                        edge_Zcos=edge_Zcos,
                        edge_Zsin=edge_Zsin,
                        enforce_lambda_axis=True,
                        idx00=idx00,
                    )
                    state_new = _apply_vmec_lambda_axis_rules(state_new)
                    return (
                        state_new,
                        vRcc,
                        vRss,
                        vZsc,
                        vZcs,
                        vLsc,
                        vLcs,
                        vRsc,
                        vRcs,
                        vZcc,
                        vZss,
                        vLcc,
                        vLss,
                        inv_tau,
                        fsq_prev,
                    )

                def _reject_step(_):
                    return (
                        state_post,
                        vRcc_post,
                        vRss_post,
                        vZsc_post,
                        vZcs_post,
                        vLsc_post,
                        vLcs_post,
                        vRsc_post,
                        vRcs_post,
                        vZcc_post,
                        vZss_post,
                        vLcc_post,
                        vLss_post,
                        inv_tau_post,
                        fsq_prev_post,
                    )

                if bool(vmec2000_control):
                    # VMEC2000 restarts retry the same iteration immediately.
                    # Use the restart payload but still accept the step.
                    (
                        state_new,
                        vRcc_new,
                        vRss_new,
                        vZsc_new,
                        vZcs_new,
                        vLsc_new,
                        vLcs_new,
                        vRsc_new,
                        vRcs_new,
                        vZcc_new,
                        vZss_new,
                        vLcc_new,
                        vLss_new,
                        inv_tau_new,
                        fsq_prev_new,
                    ) = _accept_step(None)
                else:
                    (
                        state_new,
                        vRcc_new,
                        vRss_new,
                        vZsc_new,
                        vZcs_new,
                        vLsc_new,
                        vLcs_new,
                        vRsc_new,
                        vRcs_new,
                        vZcc_new,
                        vZss_new,
                        vLcc_new,
                        vLss_new,
                        inv_tau_new,
                        fsq_prev_new,
                    ) = jax.lax.cond(do_restart, _reject_step, _accept_step, operand=None)
                fsq0_prev_new = fsq0_prev_post

                fsqr_out = fsqr
                fsqz_out = fsqz
                fsql_out = fsql
                fsqr1_out = fsqr1
                fsqz1_out = fsqz1
                fsql1_out = fsql1

                restart_effective = do_restart if not bool(vmec2000_control) else jnp.asarray(False)
                fsqz_prev = jnp.where(restart_effective, carry_adv.fsqz_prev, fsqz_out)

                accepted = jnp.logical_not(do_restart)
                accepted_count_new = jnp.where(
                    jnp.asarray(scan_core),
                    carry_adv.accepted_count,
                    carry_adv.accepted_count + jnp.asarray(accepted, dtype=jnp.int32),
                )
                nan_fsq = (~jnp.isfinite(fsq_phys)) | (~jnp.isfinite(fsq1))
                fallback_active = carry_adv.fallback_active
                if scan_fallback_enabled and (not bool(scan_core)):
                    # Track the first N probe steps independent of iter2/iter_offset.
                    probe_active = (carry_adv.probe_count < scan_fallback_iters_j) & fallback_active
                    probe_inc = jnp.where(
                        probe_active,
                        jnp.asarray(1, dtype=jnp.int32),
                        jnp.asarray(0, dtype=jnp.int32),
                    )
                    probe_count_new = carry_adv.probe_count + probe_inc
                    probe_bad_jac_new = carry_adv.probe_bad_jac + jnp.where(
                        probe_active & bad_jacobian,
                        jnp.asarray(1, dtype=jnp.int32),
                        jnp.asarray(0, dtype=jnp.int32),
                    )
                    probe_accept_new = carry_adv.probe_accept + jnp.where(
                        probe_active & accepted,
                        jnp.asarray(1, dtype=jnp.int32),
                        jnp.asarray(0, dtype=jnp.int32),
                    )
                    probe_fsq_start_new = jnp.where(
                        probe_active & (carry_adv.probe_count == 0),
                        fsq_phys,
                        carry_adv.probe_fsq_start,
                    )
                    probe_fsq_min_new = jnp.where(
                        probe_active,
                        jnp.minimum(carry_adv.probe_fsq_min, fsq_phys),
                        carry_adv.probe_fsq_min,
                    )
                    probe_fsq_max_new = jnp.where(
                        probe_active,
                        jnp.maximum(carry_adv.probe_fsq_max, fsq_phys),
                        carry_adv.probe_fsq_max,
                    )
                    has_probe = (probe_count_new >= scan_fallback_iters_j) & fallback_active
                    accepted_frac = probe_accept_new.astype(dtype) / jnp.maximum(
                        probe_count_new.astype(dtype), jnp.asarray(1.0, dtype=dtype)
                    )
                    probe_start = jnp.maximum(probe_fsq_start_new, jnp.asarray(1.0e-30, dtype=dtype))
                    probe_ratio = probe_fsq_max_new / probe_start
                    bad_progress = probe_ratio > scan_fallback_fsq_factor_j
                    probe_improve = probe_fsq_min_new <= (probe_fsq_start_new * scan_fallback_improve_j)
                    stagnation_trigger = (
                        has_probe
                        & (~probe_improve)
                        & (probe_fsq_min_new > scan_fallback_fsq_abs_j)
                        & (bad_progress | (accepted_frac < scan_fallback_accept_frac_j))
                    )
                    accepted_trigger = (
                        has_probe
                        & (accepted_frac < scan_fallback_accept_frac_j)
                        & (probe_fsq_min_new > scan_fallback_fsq_abs_j)
                        & bad_progress
                    )
                    bad_jac_trigger = (
                        probe_bad_jac_new > scan_fallback_badjac_limit_j
                    ) & (probe_fsq_min_new > scan_fallback_fsq_abs_j) & (probe_count_new > 0) & bad_progress
                    scan_fallback_abort = (bad_jac_trigger | accepted_trigger | stagnation_trigger) & fallback_active
                    abort_scan_new = (
                        carry_adv.abort_scan
                        | nan_fsq
                        | (bad_jacobian & jnp.asarray(abort_scan_on_badjac))
                        | scan_fallback_abort
                    )
                else:
                    probe_count_new = carry_adv.probe_count
                    probe_bad_jac_new = carry_adv.probe_bad_jac
                    probe_accept_new = carry_adv.probe_accept
                    probe_fsq_start_new = carry_adv.probe_fsq_start
                    probe_fsq_min_new = carry_adv.probe_fsq_min
                    probe_fsq_max_new = carry_adv.probe_fsq_max
                    abort_scan_new = (
                        carry_adv.abort_scan | nan_fsq | (bad_jacobian & jnp.asarray(abort_scan_on_badjac))
                    )

                if bool(vmec2000_control):
                    cache_valid_out = cache_valid_use
                else:
                    cache_valid_out = jnp.where(do_restart, jnp.asarray(False), cache_valid)
                # VMEC prints the updated time-step (post TimeStepControl/restart),
                # so report the post-update value on this iteration.
                time_step_report = time_step_post
                if print_in_scan:
                    def _do_print(_):
                        if scan_print_mode == "debug_print":
                            if scan_jax_debug_print is not None:
                                scan_jax_debug_print(
                                    "{i:5d}{fsqr:10.2E}{fsqz:10.2E}{fsql:10.2E}{r00:11.3E}{dt:10.2E}{w:12.4E}",
                                    i=iter2,
                                    fsqr=fsqr,
                                    fsqz=fsqz,
                                    fsql=fsql,
                                    r00=r00_j,
                                    dt=time_step_report,
                                    w=w_mhd,
                                    ordered=bool(scan_print_ordered),
                                )
                        elif scan_print_mode == "debug_callback":
                            if scan_jax_debug is not None:
                                def _cb(i, fsqr_v, fsqz_v, fsql_v, r00_v, dt_v, w_v):
                                    print(
                                        f"{int(i):5d}"
                                        f"{float(fsqr_v):10.2E}{float(fsqz_v):10.2E}{float(fsql_v):10.2E}"
                                        f"{float(r00_v):11.3E}{float(dt_v):10.2E}{float(w_v):12.4E}",
                                        flush=True,
                                    )
                                    return None
                                scan_jax_debug.callback(
                                    _cb,
                                    iter2,
                                    fsqr,
                                    fsqz,
                                    fsql,
                                    r00_j,
                                    time_step_report,
                                    w_mhd,
                                    ordered=bool(scan_print_ordered),
                                )
                        else:
                            def _cb_io(i, fsqr_v, fsqz_v, fsql_v, r00_v, dt_v, w_v):
                                print(
                                    f"{int(i):5d}"
                                    f"{float(fsqr_v):10.2E}{float(fsqz_v):10.2E}{float(fsql_v):10.2E}"
                                    f"{float(r00_v):11.3E}{float(dt_v):10.2E}{float(w_v):12.4E}",
                                    flush=True,
                                )
                                return ()
                            _io_callback(  # type: ignore[misc]
                                _cb_io,
                                None,
                                iter2,
                                fsqr,
                                fsqz,
                                fsql,
                                r00_j,
                                time_step_report,
                                w_mhd,
                                ordered=bool(scan_print_ordered),
                            )
                        return 0

                    _ = jax.lax.cond(sample_vmec, _do_print, lambda _: 0, operand=None)
                new_carry = _ScanCarry(
                    state=state_new,
                    time_step=time_step_post,
                    inv_tau=inv_tau_new,
                    fsq_prev=fsq_prev_new,
                    fsq0_prev=fsq0_prev_new,
                    accepted_count=accepted_count_new,
                    abort_scan=abort_scan_new,
                    skip_timecontrol=jnp.asarray(False) if bool(vmec2000_control) else jnp.asarray(do_restart),
                    vRcc=vRcc_new,
                    vRss=vRss_new,
                    vZsc=vZsc_new,
                    vZcs=vZcs_new,
                    vLsc=vLsc_new,
                    vLcs=vLcs_new,
                    vRsc=vRsc_new,
                    vRcs=vRcs_new,
                    vZcc=vZcc_new,
                    vZss=vZss_new,
                    vLcc=vLcc_new,
                    vLss=vLss_new,
                    flip_sign=flip_sign0,
                    iter_offset=iter_offset_post,
                    iter1=iter1_post,
                    res0=res0,
                    res1=res1,
                    state_checkpoint=state_checkpoint,
                    cache_valid=cache_valid_out,
                    cache_precond_diag=cache_precond_diag,
                    cache_tcon=cache_tcon,
                    cache_norms=cache_norms,
                    cache_rz_scale=cache_rz_scale,
                    cache_l_scale=cache_l_scale,
                    cache_rz_norm=cache_rz_norm,
                    cache_f_norm1=cache_f_norm1,
                    cache_prec_rz_mats=cache_rz_mats,
                    cache_prec_lam_prec=cache_lam_prec,
                    force_bcovar_update=jnp.asarray(False) if bool(vmec2000_control) else force_bcovar_post,
                    ijacob=ijacob_post,
                    bad_resets=bad_resets_post,
                    bad_growth=bad_growth_post,
                    fsqz_prev=fsqz_prev,
                    r00_prev=r00_j,
                    z00_prev=z00_j,
                    w_mhd_prev=w_mhd,
                    converged=carry_adv.converged | conv_now,
                    probe_count=probe_count_new,
                    probe_bad_jac=probe_bad_jac_new,
                    probe_accept=probe_accept_new,
                    probe_fsq_min=probe_fsq_min_new,
                    probe_fsq_max=probe_fsq_max_new,
                    probe_fsq_start=probe_fsq_start_new,
                    fallback_active=fallback_active,
                    fsqr_prev_phys=jnp.where(restart_effective, carry_adv.fsqr_prev_phys, fsqr_out),
                    fsqz_prev_phys=jnp.where(restart_effective, carry_adv.fsqz_prev_phys, fsqz_out),
                    fsql_prev_phys=jnp.where(restart_effective, carry_adv.fsql_prev_phys, fsql_out),
                    fsqr1_prev=jnp.where(restart_effective, carry_adv.fsqr1_prev, fsqr1_out),
                    fsqz1_prev=jnp.where(restart_effective, carry_adv.fsqz1_prev, fsqz1_out),
                    fsql1_prev=jnp.where(restart_effective, carry_adv.fsql1_prev, fsql1_out),
                    fsqr_checkpoint=fsqr_checkpoint,
                    fsqz_checkpoint=fsqz_checkpoint,
                    fsql_checkpoint=fsql_checkpoint,
                    fsqr1_checkpoint=fsqr1_checkpoint,
                    fsqz1_checkpoint=fsqz1_checkpoint,
                    fsql1_checkpoint=fsql1_checkpoint,
                )
                if scan_minimal:
                    return new_carry, _scan_hist_min(fsqr_out, fsqz_out, fsql_out)
                if scan_light:
                    return new_carry, _scan_hist_light(
                        fsqr_out,
                        fsqz_out,
                        fsql_out,
                        accepted,
                        r00_j,
                        z00_j,
                        w_mhd,
                        time_step_report,
                        bad_jacobian,
                    )
                return new_carry, (
                    fsqr_out,
                    fsqz_out,
                    fsql_out,
                    fsqr1_out,
                    fsqz1_out,
                    fsql1_out,
                    accepted,
                    r00_j,
                    z00_j,
                    w_mhd,
                    time_step_report,
                    zero_m1,
                    include_edge,
                    res0,
                    res1,
                    iter1_post,
                    bad_jacobian,
                    min_tau,
                    max_tau,
                    min_tau_ptau,
                    max_tau_ptau,
                    min_tau_state,
                    max_tau_state,
                    badjac_ptau,
                    badjac_state,
                )

            iter2_hold = jnp.asarray(it + 1, dtype=jnp.int32) + jnp.asarray(carry.iter_offset, dtype=jnp.int32)
            hold_cond = carry.converged | carry.abort_scan | (
                iter2_hold > jnp.asarray(int(max_iter), dtype=jnp.int32)
            )
            return jax.lax.cond(hold_cond, _hold_step, _advance_step, operand=carry)

        carry0 = _ScanCarry(
            state=state_init,
            time_step=time_step0,
            inv_tau=inv_tau0,
            fsq_prev=fsq_prev0,
            fsq0_prev=fsq0_prev0,
            accepted_count=jnp.asarray(0, dtype=jnp.int32),
            probe_count=jnp.asarray(0, dtype=jnp.int32),
            probe_bad_jac=jnp.asarray(0, dtype=jnp.int32),
            probe_accept=jnp.asarray(0, dtype=jnp.int32),
            probe_fsq_min=jnp.asarray(jnp.inf, dtype=dtype),
            probe_fsq_max=jnp.asarray(-jnp.inf, dtype=dtype),
            probe_fsq_start=jnp.asarray(jnp.inf, dtype=dtype),
            fallback_active=jnp.asarray(True),
            abort_scan=jnp.asarray(False),
            skip_timecontrol=jnp.asarray(False),
            vRcc=vRcc0,
            vRss=vRss0,
            vZsc=vZsc0,
            vZcs=vZcs0,
            vLsc=vLsc0,
            vLcs=vLcs0,
            vRsc=vRsc0,
            vRcs=vRcs0,
            vZcc=vZcc0,
            vZss=vZss0,
            vLcc=vLcc0,
            vLss=vLss0,
            flip_sign=flip_sign0,
            iter_offset=jnp.asarray(iter_offset0, dtype=jnp.int32),
            iter1=iter1_0,
            res0=res0_0,
            res1=res1_0,
            state_checkpoint=state_checkpoint0,
            cache_valid=cache_valid0,
            cache_precond_diag=cache_precond_diag0,
            cache_tcon=cache_tcon0,
            cache_norms=cache_norms0,
            cache_rz_scale=cache_rz_scale0,
            cache_l_scale=cache_l_scale0,
            cache_rz_norm=cache_rz_norm0,
            cache_f_norm1=cache_f_norm1_0,
            cache_prec_rz_mats=cache_rz_mats0,
            cache_prec_lam_prec=cache_lam_prec0,
            force_bcovar_update=force_bcovar0,
            ijacob=ijacob0,
            bad_resets=bad_resets0,
            bad_growth=bad_growth0,
            fsqz_prev=fsqz_prev0,
            r00_prev=r00_prev0,
            z00_prev=z00_prev0,
            w_mhd_prev=w_mhd_prev0,
            converged=jnp.asarray(False),
            fsqr_prev_phys=jnp.asarray(2.0, dtype=dtype),
            fsqz_prev_phys=jnp.asarray(0.0, dtype=dtype),
            fsql_prev_phys=jnp.asarray(0.0, dtype=dtype),
            fsqr1_prev=jnp.asarray(0.0, dtype=dtype),
            fsqz1_prev=jnp.asarray(0.0, dtype=dtype),
            fsql1_prev=jnp.asarray(0.0, dtype=dtype),
            fsqr_checkpoint=jnp.asarray(0.0, dtype=dtype),
            fsqz_checkpoint=jnp.asarray(0.0, dtype=dtype),
            fsql_checkpoint=jnp.asarray(0.0, dtype=dtype),
            fsqr1_checkpoint=jnp.asarray(0.0, dtype=dtype),
            fsqz1_checkpoint=jnp.asarray(0.0, dtype=dtype),
            fsql1_checkpoint=jnp.asarray(0.0, dtype=dtype),
        )

        preflight_default = "0" if bool(jit_forces_scan) else "1"
        preflight_env = os.getenv("VMEC_JAX_SCAN_PREFLIGHT", preflight_default).strip()
        preflight_env_l = preflight_env.lower()
        preflight_enabled = preflight_env_l not in ("", "0", "false", "no")
        if preflight_enabled and bool(vmec2000_control) and int(max_iter) > 0:
            try:
                preflight_iters = max(1, int(preflight_env))
            except Exception:
                preflight_iters = 1
        else:
            preflight_iters = 0
        if axis_reset_repeat and int(max_iter) > 0:
            preflight_iters = max(1, int(preflight_iters))
        extra_iters_default = "0" if bool(vmec2000_control) else "10"
        extra_iters_env = os.getenv("VMEC_JAX_SCAN_EXTRA_ITERS", extra_iters_default).strip()
        try:
            extra_iters = max(0, int(extra_iters_env))
        except Exception:
            extra_iters = 0
        max_iter_scan = int(max_iter) + int(extra_iters)
        if max_iter_scan <= 0:
            preflight_iters = 0
        elif preflight_iters > max_iter_scan:
            preflight_iters = int(max_iter_scan)
        max_iter_tail = int(max_iter_scan) - int(preflight_iters)

        iter_offset_preflight = iter_offset0
        if axis_reset_repeat:
            iter_offset_preflight = 0
            iter_offset0 = -1
            carry0 = carry0._replace(iter_offset=jnp.asarray(iter_offset0, dtype=jnp.int32))

        scan_cache_key = (
            "vmec2000_scan_v5",
            static_key,
            wout_key,
            edge_key,
            int(max_iter_tail),
            int(preflight_iters),
            int(iter_offset0),
            float(step_size),
            float(initial_flip_sign),
            float(lambda_update_scale),
            float(ftol),
            int(nstep_screen),
            bool(use_restart_triggers),
            bool(vmecpp_restart),
            None if stage_prev_fsq is None else float(stage_prev_fsq),
            float(stage_transition_factor),
            float(stage_transition_scale),
            bool(jit_forces_scan),
            bool(scan_light),
            bool(scan_minimal),
            int(scan_fallback_iters),
            float(scan_fallback_accept_frac),
            float(scan_fallback_fsq_factor),
            int(scan_fallback_badjac_limit),
            float(scan_fallback_fsq_abs),
        )

        def _run_scan(carry_init, it_seq):
            return jax.lax.scan(_scan_step, carry_init, it_seq)

        def _get_scan_runner(seq_len: int):
            key = scan_cache_key + (int(seq_len),)
            cached_run = _SCAN_RUNNER_CACHE.get(key)
            if cached_run is None:
                runner = jit(_run_scan)
                _SCAN_RUNNER_CACHE[key] = runner
                return runner
            return cached_run

        def _emit_scan_prints(
            *,
            hist_np,
            it_start: int,
            max_iter_local: int,
        ) -> bool:
            if scan_minimal:
                return False
            if scan_light:
                (
                    fsqr_h,
                    fsqz_h,
                    fsql_h,
                    _accepted_h,
                    r00_h,
                    z00_h,
                    w_mhd_h,
                    dt_h,
                    _bad_jac_h,
                ) = hist_np
            else:
                (
                    fsqr_h,
                    fsqz_h,
                    fsql_h,
                    _fsqr1_h,
                    _fsqz1_h,
                    _fsql1_h,
                    _accepted_h,
                    r00_h,
                    z00_h,
                    w_mhd_h,
                    dt_h,
                    _zero_m1_h,
                    _include_edge_h,
                    _res0_h,
                    _res1_h,
                    _iter1_h,
                    _bad_jac_h,
                    _min_tau_h,
                    _max_tau_h,
                    _ptau_min_h,
                    _ptau_max_h,
                    _tau_min_state_h,
                    _tau_max_state_h,
                    _badjac_ptau_h,
                    _badjac_state_h,
                ) = hist_np
            conv_mask = (fsqr_h <= float(ftol)) & (fsqz_h <= float(ftol)) & (fsql_h <= float(ftol))
            conv_idx = int(np.argmax(conv_mask)) if bool(np.any(conv_mask)) else None
            n_iter_local = int(fsqr_h.shape[0])
            for i in range(n_iter_local):
                iter2 = int(it_start + i + 1 + int(iter_offset0))
                if iter2 > int(max_iter_local):
                    break
                conv_now = (conv_idx is not None) and (i == conv_idx)
                if _should_print_vmec2000_local(iter2, int(max_iter_local)) or conv_now:
                    r00_val = float(r00_h[i])
                    z00_val = float(z00_h[i])
                    # Match VMEC precision (E11.3) for r00/z00.
                    r00_val = float(f"{r00_val:.3E}")
                    z00_val = float(f"{z00_val:.3E}")
                    _print_vmec2000_row_local(
                        iter_idx=int(iter2),
                        fsqr=float(fsqr_h[i]),
                        fsqz=float(fsqz_h[i]),
                        fsql=float(fsql_h[i]),
                        delt0r=float(dt_h[i]),
                        r00=r00_val,
                        w_mhd=float(w_mhd_h[i]),
                        z00=z00_val,
                    )
                if conv_now:
                    return True
            return False

        carry_init = carry0._replace(state=state_init, state_checkpoint=state_init)
        if chunked_print:
            hist_parts = []
            start_idx = 0
            carry = carry_init
            abort_scan_host = False
            fsq_min_global_j = jnp.asarray(jnp.inf, dtype=dtype)
            early_fallback = bool(scan_fallback_enabled) and (int(cfg.ns) > int(scan_fallback_iters))
            need_print = bool(scan_collect_print)
            chunk_size_env = os.getenv("VMEC_JAX_SCAN_CHUNK_SIZE", "").strip()
            if chunk_size_env:
                try:
                    chunk_size = max(1, int(chunk_size_env))
                except Exception:
                    chunk_size = int(nstep_screen)
            else:
                chunk_size = int(nstep_screen)
            if preflight_iters > 0:
                if iter_offset_preflight is not None:
                    carry = carry._replace(iter_offset=jnp.asarray(iter_offset_preflight, dtype=jnp.int32))
                try:
                    with jax.disable_jit():
                        carry, hist_pre = _scan_step(carry, jnp.asarray(0, dtype=jnp.int32))
                except Exception:
                    carry, hist_pre = _scan_step(carry, jnp.asarray(0, dtype=jnp.int32))
                fsq_min_global_j = jnp.minimum(
                    fsq_min_global_j,
                    jnp.min(hist_pre[0] + hist_pre[1] + hist_pre[2]),
                )
                if need_print:
                    hist_pre_np = jax.tree_util.tree_map(lambda a: np.asarray(a)[None], hist_pre)
                    hist_parts.append(hist_pre_np)
                    _ = _emit_scan_prints(hist_np=hist_pre_np, it_start=0, max_iter_local=int(max_iter))
                else:
                    hist_parts.append(jax.tree_util.tree_map(lambda a: a[None], hist_pre))
                start_idx = int(preflight_iters)
                if axis_reset_repeat:
                    carry = carry._replace(iter_offset=jnp.asarray(iter_offset0, dtype=jnp.int32))
            while start_idx < int(max_iter_scan):
                # Fixed-length chunk to avoid retracing for varying tail sizes.
                # Extra iterations are masked by the in-scan hold condition.
                chunk_len = int(chunk_size)
                it_seq = jnp.arange(start_idx, start_idx + int(chunk_len), dtype=jnp.int32)
                runner = _get_scan_runner(int(chunk_len))
                carry, hist_chunk = runner(carry, it_seq)
                fsq_min_global_j = jnp.minimum(
                    fsq_min_global_j,
                    jnp.min(hist_chunk[0] + hist_chunk[1] + hist_chunk[2]),
                )
                if need_print:
                    hist_chunk_np = jax.tree_util.tree_map(lambda a: np.asarray(a), hist_chunk)
                    hist_parts.append(hist_chunk_np)
                    converged_now = _emit_scan_prints(
                        hist_np=hist_chunk_np,
                        it_start=int(start_idx),
                        max_iter_local=int(max_iter),
                    )
                else:
                    hist_parts.append(hist_chunk)
                    converged_now = False
                start_idx = int(start_idx + int(chunk_len))
                if (
                    scan_fallback_enabled
                    and int(scan_fallback_iters) > 0
                    and start_idx >= int(scan_fallback_iters)
                    and bool(np.asarray(carry.fallback_active))
                ):
                    # Disable fallback logic after the probe window completes.
                    carry = carry._replace(fallback_active=jnp.asarray(False))
                if converged_now:
                    break
                if scan_fallback_enabled and start_idx >= int(scan_fallback_iters):
                    # Defer host sync for fsq_min_global until after the loop.
                    pass
                if bool(np.asarray(carry.converged)) or bool(np.asarray(carry.abort_scan)):
                    break
            if scan_fallback_enabled and start_idx >= int(scan_fallback_iters):
                try:
                    fsq_min_global = float(jax.device_get(fsq_min_global_j))
                except Exception:
                    fsq_min_global = None
                if fsq_min_global is not None and fsq_min_global > float(scan_fallback_fsq_abs):
                    abort_scan_host = True
            if need_print:
                hist = jax.tree_util.tree_map(lambda *parts: np.concatenate(parts, axis=0), *hist_parts)
            else:
                hist = jax.tree_util.tree_map(lambda *parts: jnp.concatenate(parts, axis=0), *hist_parts)
                hist = jax.tree_util.tree_map(lambda a: np.asarray(a), hist)
            carry_final = carry
            if abort_scan_host:
                carry_final = carry_final._replace(abort_scan=jnp.asarray(True))
        else:
            runner = _get_scan_runner(int(max_iter_tail) if int(max_iter_tail) > 0 else int(max_iter_scan))
            if preflight_iters > 0:
                # Preflight the first iteration outside the jitted scan to avoid
                # XLA aliasing issues in the initial tomnsps pass.
                carry_pre = carry_init
                if iter_offset_preflight is not None:
                    carry_pre = carry_pre._replace(iter_offset=jnp.asarray(iter_offset_preflight, dtype=jnp.int32))
                try:
                    with jax.disable_jit():
                        carry_pre, hist_pre = _scan_step(carry_pre, jnp.asarray(0, dtype=jnp.int32))
                except Exception:
                    carry_pre, hist_pre = _scan_step(carry_pre, jnp.asarray(0, dtype=jnp.int32))
                if (
                    scan_fallback_enabled
                    and int(scan_fallback_iters) > 0
                    and int(preflight_iters) >= int(scan_fallback_iters)
                ):
                    carry_pre = carry_pre._replace(fallback_active=jnp.asarray(False))
                if max_iter_tail > 0:
                    it_seq = jnp.arange(preflight_iters, int(max_iter_scan), dtype=jnp.int32)
                    if axis_reset_repeat:
                        carry_pre = carry_pre._replace(iter_offset=jnp.asarray(iter_offset0, dtype=jnp.int32))
                    carry_final, hist_tail = runner(carry_pre, it_seq)
                    hist = jax.tree_util.tree_map(
                        lambda a, b: jnp.concatenate([a[None], b], axis=0),
                        hist_pre,
                        hist_tail,
                    )
                else:
                    carry_final = carry_pre
                    hist = jax.tree_util.tree_map(lambda a: a[None], hist_pre)
            else:
                it_seq = jnp.arange(int(max_iter_scan), dtype=jnp.int32)
                carry_final, hist = runner(carry_init, it_seq)
        if scan_minimal:
            fsqr_hist, fsqz_hist, fsql_hist = hist
            accepted = None
            r00_hist = None
            z00_hist = None
            w_mhd_hist = None
            dt_hist = None
            bad_jac_hist = None
            fsqr1_hist = None
            fsqz1_hist = None
            fsql1_hist = None
            zero_m1_hist = None
            include_edge_hist = None
            res0_hist = None
            res1_hist = None
            iter1_hist = None
            min_tau_hist = None
            max_tau_hist = None
            ptau_min_hist = None
            ptau_max_hist = None
            tau_min_state_hist = None
            tau_max_state_hist = None
            badjac_ptau_hist = None
            badjac_state_hist = None
        elif scan_light:
            (
                fsqr_hist,
                fsqz_hist,
                fsql_hist,
                accepted,
                r00_hist,
                z00_hist,
                w_mhd_hist,
                dt_hist,
                bad_jac_hist,
            ) = hist
            fsqr1_hist = None
            fsqz1_hist = None
            fsql1_hist = None
            zero_m1_hist = None
            include_edge_hist = None
            res0_hist = None
            res1_hist = None
            iter1_hist = None
            min_tau_hist = None
            max_tau_hist = None
            ptau_min_hist = None
            ptau_max_hist = None
            tau_min_state_hist = None
            tau_max_state_hist = None
            badjac_ptau_hist = None
            badjac_state_hist = None
        else:
            (
                fsqr_hist,
                fsqz_hist,
                fsql_hist,
                fsqr1_hist,
                fsqz1_hist,
                fsql1_hist,
                accepted,
                r00_hist,
                z00_hist,
                w_mhd_hist,
                dt_hist,
                zero_m1_hist,
                include_edge_hist,
                res0_hist,
                res1_hist,
                iter1_hist,
                bad_jac_hist,
                min_tau_hist,
                max_tau_hist,
                ptau_min_hist,
                ptau_max_hist,
                tau_min_state_hist,
                tau_max_state_hist,
                badjac_ptau_hist,
                badjac_state_hist,
            ) = hist
        fsqr_full = np.asarray(fsqr_hist)
        fsqz_full = np.asarray(fsqz_hist)
        fsql_full = np.asarray(fsql_hist)
        if bool(vmec2000_control) or (accepted is None):
            accepted_mask = np.ones_like(fsqr_full, dtype=bool)
        else:
            accepted_mask = np.asarray(accepted).astype(bool)
        conv_mask = (fsqr_full <= float(ftol)) & (fsqz_full <= float(ftol)) & (fsql_full <= float(ftol))
        if bool(np.any(conv_mask)):
            conv_idx = int(np.argmax(conv_mask)) + 1
            conv_idx_print = conv_idx
        else:
            conv_idx = int(fsqr_full.size)
            conv_idx_print = int(max_iter)
        # Drop iterations beyond the first convergence point.
        if conv_idx < int(fsqr_full.size):
            accepted_mask = accepted_mask & (np.arange(int(fsqr_full.size)) < int(conv_idx))
        accepted_idx = np.flatnonzero(accepted_mask)
        if accepted_idx.size > int(max_iter):
            accepted_idx = accepted_idx[: int(max_iter)]
        fsqr_hist_np = fsqr_full[accepted_idx]
        fsqz_hist_np = fsqz_full[accepted_idx]
        fsql_hist_np = fsql_full[accepted_idx]
        w_hist = fsqr_hist_np + fsqz_hist_np + fsql_hist_np

        if scan_minimal or scan_light:
            fsqr1_hist_np = np.zeros((0,), dtype=float)
            fsqz1_hist_np = np.zeros((0,), dtype=float)
            fsql1_hist_np = np.zeros((0,), dtype=float)
            zero_m1_hist_np = np.zeros((0,), dtype=int)
            include_edge_hist_np = np.zeros((0,), dtype=int)
        else:
            fsqr1_hist_np = np.asarray(fsqr1_hist)[accepted_idx]
            fsqz1_hist_np = np.asarray(fsqz1_hist)[accepted_idx]
            fsql1_hist_np = np.asarray(fsql1_hist)[accepted_idx]
            zero_m1_hist_np = np.asarray(zero_m1_hist)[accepted_idx]
            include_edge_hist_np = np.asarray(include_edge_hist)[accepted_idx]
        dt_hist_np = (
            np.asarray(dt_hist)[accepted_idx] if dt_hist is not None else np.zeros((0,), dtype=float)
        )
        r00_hist_np = (
            np.asarray(r00_hist)[accepted_idx] if r00_hist is not None else np.zeros((0,), dtype=float)
        )
        w_mhd_hist_np = (
            np.asarray(w_mhd_hist)[accepted_idx] if w_mhd_hist is not None else np.zeros((0,), dtype=float)
        )

        if (
            (not scan_minimal)
            and (not print_in_scan)
            and (not chunked_print)
            and verbose
            and bool(vmec2000_control)
            and bool(verbose_vmec2000_table)
        ):
            r00_full = np.asarray(r00_hist)
            z00_full = np.asarray(z00_hist)
            w_mhd_full = np.asarray(w_mhd_hist)
            dt_full = np.asarray(dt_hist)
            last_iter = int(conv_idx_print) if int(conv_idx_print) > 0 else int(max_iter)
            for i in range(last_iter):
                iter2 = i + 1
                if _should_print_vmec2000_local(int(iter2), int(last_iter)):
                    r00_val = float(r00_full[i])
                    z00_val = float(z00_full[i])
                    # Match VMEC precision (E11.3) for r00/z00.
                    r00_val = float(f"{r00_val:.3E}")
                    z00_val = float(f"{z00_val:.3E}")
                    _print_vmec2000_row_local(
                        iter_idx=int(iter2),
                        fsqr=float(fsqr_full[i]),
                        fsqz=float(fsqz_full[i]),
                        fsql=float(fsql_full[i]),
                        delt0r=float(dt_full[i]),
                        r00=r00_val,
                        w_mhd=float(w_mhd_full[i]),
                        z00=z00_val,
                    )
        if (not scan_light) and (not scan_minimal) and os.getenv("VMEC_JAX_DUMP_PTAU", "") not in ("", "0"):
            last_iter = int(conv_idx_print) if int(conv_idx_print) > 0 else int(max_iter)
            ptau_min_full = np.asarray(ptau_min_hist)
            ptau_max_full = np.asarray(ptau_max_hist)
            tau_min_state_full = np.asarray(tau_min_state_hist)
            tau_max_state_full = np.asarray(tau_max_state_hist)
            badjac_ptau_full = np.asarray(badjac_ptau_hist).astype(int)
            badjac_state_full = np.asarray(badjac_state_hist).astype(int)
            for i in range(last_iter):
                iter2 = i + 1 + int(iter_offset0)
                _maybe_dump_ptau(
                    iter_idx=int(iter2),
                    ptau_min=float(ptau_min_full[i]),
                    ptau_max=float(ptau_max_full[i]),
                    tau_min_state=float(tau_min_state_full[i]) if np.isfinite(tau_min_state_full[i]) else None,
                    tau_max_state=float(tau_max_state_full[i]) if np.isfinite(tau_max_state_full[i]) else None,
                    badjac_ptau=bool(badjac_ptau_full[i]),
                    badjac_state=bool(badjac_state_full[i]),
                    badjac_used=bool(np.asarray(bad_jac_hist)[i]),
                    mode=badjac_mode,
                    label="scan",
                )
        res0_full = np.asarray(res0_hist) if res0_hist is not None else np.zeros((0,), dtype=float)
        res1_full = np.asarray(res1_hist) if res1_hist is not None else np.zeros((0,), dtype=float)
        iter1_full = np.asarray(iter1_hist) if iter1_hist is not None else np.zeros((0,), dtype=int)
        min_tau_full = np.asarray(min_tau_hist) if min_tau_hist is not None else np.zeros((0,), dtype=float)
        max_tau_full = np.asarray(max_tau_hist) if max_tau_hist is not None else np.zeros((0,), dtype=float)
        ptau_min_full = np.asarray(ptau_min_hist) if ptau_min_hist is not None else np.zeros((0,), dtype=float)
        ptau_max_full = np.asarray(ptau_max_hist) if ptau_max_hist is not None else np.zeros((0,), dtype=float)
        tau_min_state_full = (
            np.asarray(tau_min_state_hist) if tau_min_state_hist is not None else np.zeros((0,), dtype=float)
        )
        tau_max_state_full = (
            np.asarray(tau_max_state_hist) if tau_max_state_hist is not None else np.zeros((0,), dtype=float)
        )
        badjac_ptau_full = np.asarray(badjac_ptau_hist) if badjac_ptau_hist is not None else np.zeros((0,), dtype=int)
        badjac_state_full = np.asarray(badjac_state_hist) if badjac_state_hist is not None else np.zeros((0,), dtype=int)
        bad_jac_full = (
            np.asarray(bad_jac_hist).astype(int) if bad_jac_hist is not None else np.zeros((0,), dtype=int)
        )
        probe_count_final = int(np.asarray(carry_final.probe_count))
        probe_bad_jac_final = int(np.asarray(carry_final.probe_bad_jac))
        probe_accept_final = int(np.asarray(carry_final.probe_accept))
        probe_fsq_start_final = float(np.asarray(carry_final.probe_fsq_start))
        probe_fsq_min_final = float(np.asarray(carry_final.probe_fsq_min))
        probe_fsq_max_final = float(np.asarray(carry_final.probe_fsq_max))
        probe_ratio_final = (
            probe_fsq_max_final / max(probe_fsq_start_final, 1.0e-30)
            if probe_count_final > 0
            else float("nan")
        )
        probe_accept_frac_final = (
            float(probe_accept_final) / max(float(probe_count_final), 1.0)
            if probe_count_final > 0
            else float("nan")
        )
        fsqr_full_diag = fsqr_full if not scan_minimal else np.zeros((0,), dtype=float)
        fsqz_full_diag = fsqz_full if not scan_minimal else np.zeros((0,), dtype=float)
        fsql_full_diag = fsql_full if not scan_minimal else np.zeros((0,), dtype=float)
        n_iter_hist = int(np.asarray(w_hist).shape[0])
        resume_iter_offset = int(np.asarray(carry_final.iter_offset)) + n_iter_hist

        return SolveVmecResidualResult(
            state=carry_final.state,
            n_iter=int(w_hist.shape[0]),
            w_history=np.asarray(w_hist),
            fsqr2_history=np.asarray(fsqr_hist_np),
            fsqz2_history=np.asarray(fsqz_hist_np),
            fsql2_history=np.asarray(fsql_hist_np),
            grad_rms_history=np.asarray([], dtype=float),
            step_history=np.asarray([], dtype=float),
            diagnostics={
                "use_scan": True,
                "vmec2000_scan": True,
                "light_history": bool(scan_light),
                "scan_minimal": bool(scan_minimal),
                "badjac_use_state": bool(badjac_use_state),
                "badjac_mode": badjac_mode,
                "fsqr_full": fsqr_full_diag,
                "fsqz_full": fsqz_full_diag,
                "fsql_full": fsql_full_diag,
                "accepted_mask": np.asarray(accepted_mask),
                "converged_iter": int(conv_idx),
                "converged": bool(np.any(conv_mask)),
                "ijacob": int(np.asarray(carry_final.ijacob)),
                "fsqr1_history": fsqr1_hist_np,
                "fsqz1_history": fsqz1_hist_np,
                "fsql1_history": fsql1_hist_np,
                "time_step_history": dt_hist_np,
                "r00_history": r00_hist_np,
                "w_vmec_history": w_mhd_hist_np,
                "zero_m1_history": zero_m1_hist_np.astype(int),
                "include_edge_history": include_edge_hist_np.astype(int),
                "res0_full": res0_full,
                "res1_full": res1_full,
                "iter1_full": iter1_full,
                "bad_jacobian_full": bad_jac_full,
                "min_tau_full": min_tau_full,
                "max_tau_full": max_tau_full,
                "ptau_min_full": ptau_min_full,
                "ptau_max_full": ptau_max_full,
                "tau_min_state_full": tau_min_state_full,
                "tau_max_state_full": tau_max_state_full,
                "badjac_ptau_full": badjac_ptau_full.astype(int),
                "badjac_state_full": badjac_state_full.astype(int),
                "abort_scan": bool(np.asarray(carry_final.abort_scan)),
                "probe_count": probe_count_final,
                "probe_bad_jac": probe_bad_jac_final,
                "probe_accept": probe_accept_final,
                "probe_fsq_start": probe_fsq_start_final,
                "probe_fsq_min": probe_fsq_min_final,
                "probe_fsq_max": probe_fsq_max_final,
                "probe_ratio": probe_ratio_final,
                "probe_accept_frac": probe_accept_frac_final,
                "resume_state": {
                    "state_checkpoint": carry_final.state_checkpoint,
                    "time_step": float(np.asarray(carry_final.time_step)),
                    "inv_tau": np.asarray(carry_final.inv_tau),
                    "fsq_prev": float(np.asarray(carry_final.fsq_prev)),
                    "fsq0_prev": float(np.asarray(carry_final.fsq0_prev)),
                    "flip_sign": float(np.asarray(carry_final.flip_sign)),
                    "iter1": int(np.asarray(carry_final.iter1)),
                    "iter_offset": int(resume_iter_offset),
                    "res0": float(np.asarray(carry_final.res0)),
                    "res1": float(np.asarray(carry_final.res1)),
                    "prev_rz_fsq": float(
                        np.asarray(carry_final.fsqr_prev_phys + carry_final.fsqz_prev_phys)
                    ),
                    "vmec2000_cache_valid": bool(np.asarray(carry_final.cache_valid)),
                    "ijacob": int(np.asarray(carry_final.ijacob)),
                    "bad_resets": int(np.asarray(carry_final.bad_resets)),
                    "bad_growth_streak": int(np.asarray(carry_final.bad_growth)),
                    "fsqz_prev": float(np.asarray(carry_final.fsqz_prev)),
                    "vRcc": np.asarray(carry_final.vRcc),
                    "vRss": np.asarray(carry_final.vRss),
                    "vZsc": np.asarray(carry_final.vZsc),
                    "vZcs": np.asarray(carry_final.vZcs),
                    "vLsc": np.asarray(carry_final.vLsc),
                    "vLcs": np.asarray(carry_final.vLcs),
                    "cache_precond_diag": carry_final.cache_precond_diag,
                    "cache_tcon": carry_final.cache_tcon,
                    "cache_norms": carry_final.cache_norms,
                    "cache_rz_scale": carry_final.cache_rz_scale,
                    "cache_l_scale": carry_final.cache_l_scale,
                    "cache_rz_norm": np.asarray(carry_final.cache_rz_norm),
                    "cache_f_norm1": np.asarray(carry_final.cache_f_norm1),
                    "cache_prec_rz_mats": carry_final.cache_prec_rz_mats,
                    "cache_prec_lam_prec": np.asarray(carry_final.cache_prec_lam_prec),
                    "r00_prev": float(np.asarray(carry_final.r00_prev)),
                    "z00_prev": float(np.asarray(carry_final.z00_prev)),
                    "w_mhd_prev": float(np.asarray(carry_final.w_mhd_prev)),
                    "force_bcovar_update": bool(np.asarray(carry_final.force_bcovar_update)),
                },
            },
        )

    if use_scan:
        if vmec2000_control:
            scan_result = _run_vmec2000_scan(state)
            if scan_fallback_enabled:
                try:
                    bad_jac_full = scan_result.diagnostics.get("bad_jacobian_full", None)
                except Exception:
                    bad_jac_full = None
                try:
                    abort_scan_flag = bool(scan_result.diagnostics.get("abort_scan", False))
                except Exception:
                    abort_scan_flag = False
                bad_jac_count = 0
                if bad_jac_full is not None:
                    try:
                        bad_jac_arr = np.asarray(bad_jac_full).astype(int)
                        probe_iters = min(int(max_iter), int(scan_fallback_iters))
                        if probe_iters > 0:
                            bad_jac_count = int(np.sum(bad_jac_arr[:probe_iters]))
                    except Exception:
                        bad_jac_count = 0
                accepted_frac = None
                try:
                    accepted_mask = scan_result.diagnostics.get("accepted_mask", None)
                except Exception:
                    accepted_mask = None
                if accepted_mask is not None:
                    try:
                        accepted_arr = np.asarray(accepted_mask).astype(float)
                        probe_iters = min(int(max_iter), int(scan_fallback_iters))
                        if probe_iters > 0 and accepted_arr.size >= probe_iters:
                            accepted_frac = float(np.mean(accepted_arr[:probe_iters]))
                    except Exception:
                        accepted_frac = None
                fallback_reasons = []
                fsq_min_full = None
                fsq_max_full = None
                fsq_all_finite = True
                try:
                    fsqr_diag = scan_result.diagnostics.get("fsqr_full", None)
                    fsqz_diag = scan_result.diagnostics.get("fsqz_full", None)
                    fsql_diag = scan_result.diagnostics.get("fsql_full", None)
                    if fsqr_diag is None or np.asarray(fsqr_diag).size == 0:
                        fsqr_diag = scan_result.fsqr2_history
                        fsqz_diag = scan_result.fsqz2_history
                        fsql_diag = scan_result.fsql2_history
                    fsq_full_arr = np.asarray(fsqr_diag) + np.asarray(fsqz_diag) + np.asarray(fsql_diag)
                    fsq_min_full = float(np.min(fsq_full_arr))
                    fsq_max_full = float(np.max(fsq_full_arr))
                    fsq_all_finite = bool(np.all(np.isfinite(fsq_full_arr)))
                except Exception:
                    fsq_min_full = None
                    fsq_max_full = None
                    fsq_all_finite = False
                fsq_min_ok = (
                    True
                    if fsq_min_full is None
                    else bool(fsq_min_full > float(scan_fallback_fsq_abs))
                )
                fsq_ratio_ok = True
                if fsq_min_full is not None and fsq_max_full is not None:
                    if fsq_min_full > 0.0:
                        fsq_ratio_ok = (fsq_max_full / fsq_min_full) > float(scan_fallback_fsq_factor)
                if abort_scan_flag and (not fsq_all_finite or fsq_ratio_ok):
                    fallback_reasons.append("abort_scan")

                if bad_jac_count > scan_fallback_badjac_limit and fsq_min_ok and fsq_ratio_ok:
                    fallback_reasons.append(
                        f"bad_jac_count={bad_jac_count} > {scan_fallback_badjac_limit}"
                    )
                # Note: ijacob alone is not a reliable failure signal; do not
                # fall back solely on ijacob growth.
                if (
                    accepted_frac is not None
                    and accepted_frac < scan_fallback_accept_frac
                    and fsq_min_ok
                    and fsq_ratio_ok
                ):
                    fallback_reasons.append(
                        f"accepted_frac={accepted_frac:.2f} < {scan_fallback_accept_frac:.2f}"
                    )

                if fallback_reasons:
                    if verbose:
                        reason_str = ", ".join(fallback_reasons)
                        probe_msg = ""
                        try:
                            probe_count = int(scan_result.diagnostics.get("probe_count", 0))
                            if probe_count > 0:
                                probe_msg = (
                                    " "
                                    f"(probe_count={probe_count} "
                                    f"probe_accept_frac={scan_result.diagnostics.get('probe_accept_frac', float('nan')):.2f} "
                                    f"probe_ratio={scan_result.diagnostics.get('probe_ratio', float('nan')):.2f} "
                                    f"probe_fsq_min={scan_result.diagnostics.get('probe_fsq_min', float('nan')):.3e})"
                                )
                        except Exception:
                            probe_msg = ""
                        print(
                            "[solve_fixed_boundary_residual_iter] scan fallback -> non-scan "
                            f"({reason_str}){probe_msg}",
                            flush=True,
                        )
                    use_scan = False
                    resume_state = None
                    state = state0
                else:
                    return scan_result
            else:
                return scan_result

        if use_scan:
            if backtracking or use_restart_triggers or auto_flip_force or limit_dt_from_force or limit_update_rms or strict_update or use_direct_fallback or reference_mode:
                raise ValueError(
                    "use_scan requires vmec2000_control=False, backtracking=False, "
                    "use_restart_triggers=False, auto_flip_force=False, "
                    "limit_dt_from_force=False, limit_update_rms=False, strict_update=False, "
                    "use_direct_fallback=False, reference_mode=False."
                )

            dtype = jnp.asarray(state0.Rcos).dtype
            time_step_j = jnp.asarray(float(step_size), dtype=dtype)
            flip_sign_j = jnp.asarray(float(initial_flip_sign), dtype=dtype)

            include_edge_scan = False
            _compute_forces_scan = _compute_forces if jit_forces else _compute_forces_impl

            scan_cache_key = (
                "scan_v1",
                static_key,
                wout_key,
                edge_key,
                int(max_iter),
                float(step_size),
                float(initial_flip_sign),
                float(lambda_update_scale),
                float(precond_radial_alpha),
                float(precond_lambda_alpha),
                bool(apply_m1_constraints),
                bool(jit_forces),
            )

            def _scan_step(state, it):
                it = jnp.asarray(it, dtype=jnp.int32)
                iter_since_restart = it + 1
                zero_m1 = jnp.where(iter_since_restart < 2, jnp.asarray(1.0, dtype=dtype), jnp.asarray(0.0, dtype=dtype))

                k, frzl, fsqr, fsqz, fsql, rz_scale, l_scale, _norms = _compute_forces_scan(
                    state,
                    include_edge=include_edge_scan,
                    zero_m1=zero_m1,
                    iter_idx=None,
                )

                frss_in = (frzl.frss if frzl.frss is not None else jnp.zeros_like(frzl.frcc)) * rz_scale[:, None, None]
                fzcs_in = (frzl.fzcs if frzl.fzcs is not None else jnp.zeros_like(frzl.fzsc)) * rz_scale[:, None, None]
                frcc, frss, fzsc, fzcs = _apply_radial_tridi_batched(
                    [
                        frzl.frcc * rz_scale[:, None, None],
                        frss_in,
                        frzl.fzsc * rz_scale[:, None, None],
                        fzcs_in,
                    ],
                    precond_radial_alpha,
                )
                flcs_in = (frzl.flcs if frzl.flcs is not None else jnp.zeros_like(frzl.flsc)) * l_scale[:, None, None]
                flsc, flcs = _apply_radial_tridi_batched(
                    [
                        frzl.flsc * l_scale[:, None, None],
                        flcs_in,
                    ],
                    precond_lambda_alpha,
                )

                frzl_pre = TomnspsRZL(
                    frcc=frcc,
                    frss=frss,
                    fzsc=fzsc,
                    fzcs=fzcs,
                    flsc=flsc,
                    flcs=flcs,
                    frsc=getattr(frzl, "frsc", None),
                    frcs=getattr(frzl, "frcs", None),
                    fzcc=getattr(frzl, "fzcc", None),
                    fzss=getattr(frzl, "fzss", None),
                    flcc=getattr(frzl, "flcc", None),
                    flss=getattr(frzl, "flss", None),
                )

                frcc_u = frcc * w_mode_mn[None, :, :]
                frss_u = frss * w_mode_mn[None, :, :]
                fzsc_u = fzsc * w_mode_mn[None, :, :]
                fzcs_u = fzcs * w_mode_mn[None, :, :]
                flsc_u = flsc * w_mode_mn[None, :, :]
                flcs_u = flcs * w_mode_mn[None, :, :]
                frsc_u = (jnp.asarray(getattr(frzl, "frsc", None)) if getattr(frzl, "frsc", None) is not None else jnp.zeros_like(frcc_u)) * w_mode_mn[None, :, :]
                frcs_u = (jnp.asarray(getattr(frzl, "frcs", None)) if getattr(frzl, "frcs", None) is not None else jnp.zeros_like(frcc_u)) * w_mode_mn[None, :, :]
                fzcc_u = (jnp.asarray(getattr(frzl, "fzcc", None)) if getattr(frzl, "fzcc", None) is not None else jnp.zeros_like(fzsc_u)) * w_mode_mn[None, :, :]
                fzss_u = (jnp.asarray(getattr(frzl, "fzss", None)) if getattr(frzl, "fzss", None) is not None else jnp.zeros_like(fzsc_u)) * w_mode_mn[None, :, :]
                flcc_u = (jnp.asarray(getattr(frzl, "flcc", None)) if getattr(frzl, "flcc", None) is not None else jnp.zeros_like(flsc_u)) * w_mode_mn[None, :, :]
                flss_u = (jnp.asarray(getattr(frzl, "flss", None)) if getattr(frzl, "flss", None) is not None else jnp.zeros_like(flsc_u)) * w_mode_mn[None, :, :]

                if lambda_update_scale != 1.0:
                    flsc_u = flsc_u * lambda_update_scale_j
                    flcs_u = flcs_u * lambda_update_scale_j
                    flcc_u = flcc_u * lambda_update_scale_j
                    flss_u = flss_u * lambda_update_scale_j

                dR = (time_step_j * flip_sign_j) * _mn_cos_to_signed_physical(frcc_u, frss_u)
                sin_updates = _mn_sin_to_signed_batch(
                    jnp.stack([fzsc_u, flsc_u], axis=0),
                    jnp.stack([fzcs_u, flcs_u], axis=0),
                )
                dZ = (time_step_j * flip_sign_j) * sin_updates[0]
                dL = (time_step_j * flip_sign_j) * sin_updates[1]
                dR_sin = (time_step_j * flip_sign_j) * _mn_sin_to_signed_physical(frsc_u, frcs_u)
                dZ_cos = (time_step_j * flip_sign_j) * _mn_cos_to_signed_physical(fzcc_u, fzss_u)
                dL_cos = (time_step_j * flip_sign_j) * _mn_cos_to_signed_physical_lambda(flcc_u, flss_u)
                if not bool(cfg.lasym):
                    dR_sin = jnp.zeros_like(dR_sin)
                    dZ_cos = jnp.zeros_like(dZ_cos)
                    dL_cos = jnp.zeros_like(dL_cos)

                state_new = VMECState(
                    layout=state.layout,
                    Rcos=jnp.asarray(state.Rcos) + dR,
                    Rsin=jnp.asarray(state.Rsin) + dR_sin,
                    Zcos=jnp.asarray(state.Zcos) + dZ_cos,
                    Zsin=jnp.asarray(state.Zsin) + dZ,
                    Lcos=jnp.asarray(state.Lcos) + dL_cos,
                    Lsin=jnp.asarray(state.Lsin) + dL,
                )
                state_new = _enforce_fixed_boundary_and_axis(
                    state_new,
                    static,
                    edge_Rcos=edge_Rcos,
                    edge_Rsin=edge_Rsin,
                    edge_Zcos=edge_Zcos,
                    edge_Zsin=edge_Zsin,
                    enforce_lambda_axis=True,
                    idx00=idx00,
                )
                state_new = _apply_vmec_lambda_axis_rules(state_new)

                return state_new, (fsqr, fsqz, fsql)

            def _run_scan(state_init):
                return jax.lax.scan(_scan_step, state_init, jnp.arange(max_iter, dtype=jnp.int32))

            cached_run = _SCAN_RUNNER_CACHE.get(scan_cache_key)
            if cached_run is None:
                _run_scan = jit(_run_scan)
                _SCAN_RUNNER_CACHE[scan_cache_key] = _run_scan
            else:
                _run_scan = cached_run

            state_final, hist = _run_scan(state)
            fsqr_hist, fsqz_hist, fsql_hist = hist
            w_hist = fsqr_hist + fsqz_hist + fsql_hist
            return SolveVmecResidualResult(
                state=state_final,
                n_iter=int(max_iter),
                w_history=np.asarray(w_hist),
                fsqr2_history=np.asarray(fsqr_hist),
                fsqz2_history=np.asarray(fsqz_hist),
                fsql2_history=np.asarray(fsql_hist),
                grad_rms_history=np.asarray([], dtype=float),
                step_history=np.asarray([], dtype=float),
                diagnostics={"use_scan": True},
            )

    profile_window = os.getenv("VMEC_JAX_PROFILE_WINDOW", "").strip().lower()
    profile_dir_env = os.getenv("VMEC_JAX_PROFILE_DIR", "").strip()
    profile_started = False
    profile_active = False
    profile_start_iter = None
    profile_dir = ""
    if profile_window and profile_dir_env:
        if profile_window in ("pre", "iter1", "1"):
            profile_start_iter = 1
        else:
            window_str = profile_window
            if window_str.startswith("iter"):
                window_str = window_str[4:]
            try:
                profile_start_iter = max(1, int(window_str))
            except Exception:
                profile_start_iter = None
        if profile_start_iter is not None:
            profile_dir = str(Path(profile_dir_env) / f"window_{profile_window}")
            profile_active = True
    perfetto_env = os.getenv("VMEC_JAX_PROFILE_PERFETTO", "1")
    profile_perfetto = perfetto_env.strip().lower() not in ("", "0", "false", "no")

    timing_env = os.getenv("VMEC_JAX_TIMING", "").strip().lower()
    timing_enabled = timing_env not in ("", "0", "false", "no")
    timing_stats = {
        "compute_forces": 0.0,
        "preconditioner": 0.0,
        "update": 0.0,
        "precond_refresh": 0.0,
        "iterations": 0,
    }

    w_history = []
    fsqr2_history = []
    fsqz2_history = []
    fsql2_history = []
    r00_history: list[float] = []
    z00_history: list[float] = []
    wb_history: list[float] = []
    wp_history: list[float] = []
    w_vmec_history: list[float] = []
    fsqr1_history = []
    fsqz1_history = []
    fsql1_history = []
    fsq1_history = []
    rz_norm_history: list[float] = []
    f_norm1_history: list[float] = []
    gcr2_p_history: list[float] = []
    gcz2_p_history: list[float] = []
    gcl2_p_history: list[float] = []
    step_status_history: list[str] = []
    restart_reason_history: list[str] = []
    pre_restart_reason_history: list[str] = []
    time_step_history: list[float] = []
    res0_history: list[float] = []
    res1_history: list[float] = []
    fsq_prev_history: list[float] = []
    bad_growth_streak_history: list[int] = []
    iter1_history: list[int] = []
    include_edge_history: list[int] = []
    zero_m1_history: list[int] = []
    dt_eff_history: list[float] = []
    update_rms_history: list[float] = []
    w_curr_history: list[float] = []
    w_try_history: list[float] = []
    w_try_ratio_history: list[float] = []
    restart_path_history: list[str] = []
    min_tau_history: list[float] = []
    max_tau_history: list[float] = []
    bad_jacobian_history: list[int] = []
    grad_rms_history = []
    step_history = []
    r00_last = float("nan")
    z00_last = float("nan")
    wb_last = float("nan")
    wp_last = float("nan")
    w_vmec_last = float("nan")

    # Conjugate-gradient-like time-stepping state.
    time_step = float(step_size)
    k_ndamp = 10
    inv_tau = [0.15 / time_step] * k_ndamp
    fsq_prev = 1.0
    fsq0_prev = 1.0
    vRcc = jnp.zeros((int(state.Rcos.shape[0]), mpol, nrange), dtype=jnp.asarray(state.Rcos).dtype)
    vRss = jnp.zeros_like(vRcc)
    vZsc = jnp.zeros_like(vRcc)
    vZcs = jnp.zeros_like(vRcc)
    vLsc = jnp.zeros_like(vRcc)
    vLcs = jnp.zeros_like(vRcc)
    vRsc = jnp.zeros_like(vRcc)
    vRcs = jnp.zeros_like(vRcc)
    vZcc = jnp.zeros_like(vRcc)
    vZss = jnp.zeros_like(vRcc)
    vLcc = jnp.zeros_like(vRcc)
    vLss = jnp.zeros_like(vRcc)
    flip_sign = float(initial_flip_sign)
    max_coeff_delta_rms = 1e-5
    max_update_rms = 5e-3
    if bool(reference_mode):
        max_coeff_delta_rms = 5e-6
        max_update_rms = 1e-3
    ijacob = 0
    bad_resets = 0
    iter1 = 1
    res0 = -1.0
    k_preconditioner_update_interval = 25
    state_checkpoint = state
    bad_growth_streak = 0
    # Restart trigger factors:
    # - bad_jacobian: time_step *= 0.9
    # - bad_progress: time_step /= 1.03
    restart_badjac_factor = 0.9
    restart_badprog_factor = 1.03
    huge_force_restart_count = 0
    huge_force_restart_budget = 2
    res1 = -1.0
    vmec2000_fact = 1.0e4

    # Edge-force gating uses the *previous* iteration's residual (the first
    # iteration initializes forces to 1.0). Track that explicitly.
    prev_rz_fsq = 2.0

    scan_print_env = os.getenv("VMEC_JAX_SCAN_PRINT", "1").strip().lower()
    scan_print_mode = os.getenv("VMEC_JAX_SCAN_PRINT_MODE", "debug_print").strip().lower()
    scan_print_ordered = os.getenv("VMEC_JAX_SCAN_PRINT_ORDERED", "0").strip().lower() not in ("", "0", "false", "no")
    print_live = scan_print_env not in ("", "0", "false", "no")
    _jax_debug = None
    _io_callback = None
    if print_live:
        try:
            from jax import debug as _jax_debug  # type: ignore[assignment]
        except Exception:
            _jax_debug = None
    if scan_print_mode not in ("debug_print", "debug_callback", "io_callback"):
        scan_print_mode = "debug_print"
    if scan_print_mode == "io_callback":
        try:
            from jax.experimental import io_callback as _io_callback  # type: ignore[assignment]
        except Exception:
            scan_print_mode = "debug_print"
            _io_callback = None

    def _print_vmec2000_iter_row(
        *,
        iter_idx: int,
        fsqr: float,
        fsqz: float,
        fsql: float,
        fsqr1: float,
        fsqz1: float,
        fsql1: float,
        delt0r: float,
        r00: float,
        w_mhd: float,
        z00: float | None = None,
    ) -> None:
        if not (bool(verbose) and bool(vmec2000_control) and bool(verbose_vmec2000_table)):
            return
        if not print_live:
            return
        if bool(cfg.lasym):
            z_val = float("nan") if z00 is None else float(z00)
            if _jax_debug is not None:
                if scan_print_mode == "debug_print":
                    _jax_debug.print(
                        "{i:5d}{fsqr:10.2E}{fsqz:10.2E}{fsql:10.2E}{r00:11.3E}{z00:11.3E}{dt:10.2E}{w:12.4E}",
                        i=iter_idx,
                        fsqr=fsqr,
                        fsqz=fsqz,
                        fsql=fsql,
                        r00=r00,
                        z00=z_val,
                        dt=delt0r,
                        w=w_mhd,
                        ordered=bool(scan_print_ordered),
                    )
                elif scan_print_mode == "debug_callback":
                    def _cb(i, fsqr_v, fsqz_v, fsql_v, r00_v, z00_v, dt_v, w_v):
                        print(
                            f"{int(i):5d}"
                            f"{float(fsqr_v):10.2E}{float(fsqz_v):10.2E}{float(fsql_v):10.2E}"
                            f"{float(r00_v):11.3E}{float(z00_v):11.3E}{float(dt_v):10.2E}{float(w_v):12.4E}",
                            flush=True,
                        )
                        return None
                    _jax_debug.callback(
                        _cb,
                        iter_idx,
                        fsqr,
                        fsqz,
                        fsql,
                        r00,
                        z_val,
                        delt0r,
                        w_mhd,
                        ordered=bool(scan_print_ordered),
                    )
                else:
                    def _cb_io(i, fsqr_v, fsqz_v, fsql_v, r00_v, z00_v, dt_v, w_v):
                        print(
                            f"{int(i):5d}"
                            f"{float(fsqr_v):10.2E}{float(fsqz_v):10.2E}{float(fsql_v):10.2E}"
                            f"{float(r00_v):11.3E}{float(z00_v):11.3E}{float(dt_v):10.2E}{float(w_v):12.4E}",
                            flush=True,
                        )
                        return ()
                    _io_callback(  # type: ignore[misc]
                        _cb_io,
                        None,
                        iter_idx,
                        fsqr,
                        fsqz,
                        fsql,
                        r00,
                        z_val,
                        delt0r,
                        w_mhd,
                        ordered=bool(scan_print_ordered),
                    )
            else:
                # VMEC screen format (lasym, fixed-boundary): i5,3e10.2,2e11.3,e10.2,e12.4
                print(
                    f"{int(iter_idx):5d}"
                    f"{float(fsqr):10.2E}{float(fsqz):10.2E}{float(fsql):10.2E}"
                    f"{float(r00):11.3E}{z_val:11.3E}{float(delt0r):10.2E}{float(w_mhd):12.4E}",
                    flush=True,
                )
        else:
            if _jax_debug is not None:
                if scan_print_mode == "debug_print":
                    _jax_debug.print(
                        "{i:5d}{fsqr:10.2E}{fsqz:10.2E}{fsql:10.2E}{r00:11.3E}{dt:10.2E}{w:12.4E}",
                        i=iter_idx,
                        fsqr=fsqr,
                        fsqz=fsqz,
                        fsql=fsql,
                        r00=r00,
                        dt=delt0r,
                        w=w_mhd,
                        ordered=bool(scan_print_ordered),
                    )
                elif scan_print_mode == "debug_callback":
                    def _cb(i, fsqr_v, fsqz_v, fsql_v, r00_v, dt_v, w_v):
                        print(
                            f"{int(i):5d}"
                            f"{float(fsqr_v):10.2E}{float(fsqz_v):10.2E}{float(fsql_v):10.2E}"
                            f"{float(r00_v):11.3E}{float(dt_v):10.2E}{float(w_v):12.4E}",
                            flush=True,
                        )
                        return None
                    _jax_debug.callback(
                        _cb,
                        iter_idx,
                        fsqr,
                        fsqz,
                        fsql,
                        r00,
                        delt0r,
                        w_mhd,
                        ordered=bool(scan_print_ordered),
                    )
                else:
                    def _cb_io(i, fsqr_v, fsqz_v, fsql_v, r00_v, dt_v, w_v):
                        print(
                            f"{int(i):5d}"
                            f"{float(fsqr_v):10.2E}{float(fsqz_v):10.2E}{float(fsql_v):10.2E}"
                            f"{float(r00_v):11.3E}{float(dt_v):10.2E}{float(w_v):12.4E}",
                            flush=True,
                        )
                        return ()
                    _io_callback(  # type: ignore[misc]
                        _cb_io,
                        None,
                        iter_idx,
                        fsqr,
                        fsqz,
                        fsql,
                        r00,
                        delt0r,
                        w_mhd,
                        ordered=bool(scan_print_ordered),
                    )
            else:
                # VMEC screen format (fixed-boundary): i5,3e10.2,e11.3,e10.2,e12.4
                print(
                    f"{int(iter_idx):5d}"
                    f"{float(fsqr):10.2E}{float(fsqz):10.2E}{float(fsql):10.2E}"
                    f"{float(r00):11.3E}{float(delt0r):10.2E}{float(w_mhd):12.4E}",
                    flush=True,
                )

    nstep_screen = int(indata.get_int("NSTEP", 1)) if indata is not None else 1
    nstep_override = os.getenv("VMEC_JAX_NSTEP_OVERRIDE", "").strip()
    if nstep_override not in ("", "0"):
        try:
            nstep_screen = int(nstep_override)
        except Exception:
            pass
    if nstep_screen < 1:
        nstep_screen = 1

    def _should_print_vmec2000(iter_idx: int, max_iter: int) -> bool:
        if not (bool(verbose) and bool(vmec2000_control) and bool(verbose_vmec2000_table)):
            return False
        if iter_idx <= 1:
            return True
        if iter_idx >= max_iter:
            return True
        return (iter_idx % nstep_screen) == 0

    def _should_sample_vmec2000(iter_idx: int, max_iter: int) -> bool:
        """Sample VMEC2000 scalar diagnostics on the screen cadence."""
        if iter_idx <= 1:
            return True
        if iter_idx >= max_iter:
            return True
        return (iter_idx % nstep_screen) == 0

    # VMEC2000 caches 1D preconditioner/norm/tcon updates every `ns4` iterations
    # (vmec_params.f: ns4=25), reusing the cached values between refreshes.
    # This materially affects the nonlinear iteration trace because the
    # Garabedian time-step control depends on ratios of the *preconditioned*
    # residual scalars.
    vmec2000_cache_valid = False
    cache_precond_diag = None
    cache_tcon = None
    cache_norms = None
    cache_rz_scale = None
    cache_l_scale = None
    cache_rz_norm = None
    cache_f_norm1 = None
    cache_prec_rz_mats = None
    cache_prec_rz_jmax = None
    cache_prec_lam_prec = None
    cache_prec_faclam = None
    cache_prec_lam_debug = None
    bcovar_update_history: list[int] = []
    iter_offset = 0

    if resume_state is not None:
        iter_offset = int(resume_state.get("iter_offset", iter_offset))
        time_step = float(resume_state.get("time_step", time_step))
        inv_tau = list(resume_state.get("inv_tau", inv_tau))
        fsq_prev = float(resume_state.get("fsq_prev", fsq_prev))
        fsq0_prev = float(resume_state.get("fsq0_prev", fsq0_prev))
        flip_sign = float(resume_state.get("flip_sign", flip_sign))
        iter1 = int(resume_state.get("iter1", iter1))
        ijacob = int(resume_state.get("ijacob", ijacob))
        bad_resets = int(resume_state.get("bad_resets", bad_resets))
        res0 = float(resume_state.get("res0", res0))
        res1 = float(resume_state.get("res1", res1))
        prev_rz_fsq = float(resume_state.get("prev_rz_fsq", prev_rz_fsq))
        bad_growth_streak = int(resume_state.get("bad_growth_streak", bad_growth_streak))
        huge_force_restart_count = int(resume_state.get("huge_force_restart_count", huge_force_restart_count))

        if "vRcc" in resume_state:
            vRcc = jnp.asarray(resume_state["vRcc"])
            vRss = jnp.asarray(resume_state.get("vRss", vRss))
            vZsc = jnp.asarray(resume_state.get("vZsc", vZsc))
            vZcs = jnp.asarray(resume_state.get("vZcs", vZcs))
            vLsc = jnp.asarray(resume_state.get("vLsc", vLsc))
            vLcs = jnp.asarray(resume_state.get("vLcs", vLcs))

        state_checkpoint = resume_state.get("state_checkpoint", state)
        vmec2000_cache_valid = bool(resume_state.get("vmec2000_cache_valid", vmec2000_cache_valid))
        cache_precond_diag = resume_state.get("cache_precond_diag", cache_precond_diag)
        cache_tcon = resume_state.get("cache_tcon", cache_tcon)
        cache_norms = resume_state.get("cache_norms", cache_norms)
        cache_rz_scale = resume_state.get("cache_rz_scale", cache_rz_scale)
        cache_l_scale = resume_state.get("cache_l_scale", cache_l_scale)
        cache_rz_norm = resume_state.get("cache_rz_norm", cache_rz_norm)
        cache_f_norm1 = resume_state.get("cache_f_norm1", cache_f_norm1)
        cache_prec_rz_mats = resume_state.get("cache_prec_rz_mats", cache_prec_rz_mats)
        cache_prec_rz_jmax = resume_state.get("cache_prec_rz_jmax", cache_prec_rz_jmax)
        cache_prec_lam_prec = resume_state.get("cache_prec_lam_prec", cache_prec_lam_prec)
        cache_prec_faclam = resume_state.get("cache_prec_faclam", cache_prec_faclam)
        cache_prec_lam_debug = resume_state.get("cache_prec_lam_debug", cache_prec_lam_debug)

    def _fmt_axis_coeff(val: float) -> str:
        s = f"{float(val):.16g}"
        if "e" in s:
            s = s.replace("e", "E")
        return s

    def _print_axis_guess(raxis_cc, zaxis_cs) -> None:
        try:
            r_line = "      RAXIS_CC =    " + "   ".join(_fmt_axis_coeff(v) for v in np.ravel(raxis_cc))
            z_line = "      ZAXIS_CS =    " + "   ".join(_fmt_axis_coeff(v) for v in np.ravel(zaxis_cs))
            print("  ---- Improved AXIS Guess ----", flush=True)
            print(r_line, flush=True)
            print(z_line, flush=True)
            print("  -----------------------------", flush=True)
        except Exception:
            pass

            res0 = -1.0
            res1 = -1.0
            prev_rz_fsq = 2.0
            vmec2000_cache_valid = False
            cache_precond_diag = None
            cache_tcon = None
            cache_norms = None
            cache_rz_scale = None
            cache_l_scale = None
            cache_rz_norm = None
            cache_f_norm1 = None
            cache_prec_rz_mats = None
            cache_prec_rz_jmax = None
            cache_prec_lam_prec = None
            cache_prec_faclam = None
            cache_prec_lam_debug = None

    def _safe_dt_from_force(
        *,
        dt_nominal: float,
        frcc,
        frss,
        fzsc,
        fzcs,
        flsc,
        flcs,
        frsc=None,
        frcs=None,
        fzcc=None,
        fzss=None,
        flcc=None,
        flss=None,
    ) -> float:
        """Optional limiter for dt based on force magnitude.

        The reference iteration uses `time_step` directly (with restart-trigger
        adjustments) and does not apply a force-based dt limiter. Keep this
        behavior off by default for parity; enable only as a stability crutch
        during debugging.
        """
        frcc = jnp.asarray(frcc)
        frss = jnp.asarray(frss) if frss is not None else jnp.zeros_like(frcc)
        fzsc = jnp.asarray(fzsc)
        fzcs = jnp.asarray(fzcs) if fzcs is not None else jnp.zeros_like(fzsc)
        flsc = jnp.asarray(flsc)
        flcs = jnp.asarray(flcs) if flcs is not None else jnp.zeros_like(flsc)
        frsc = jnp.asarray(frsc) if frsc is not None else jnp.zeros_like(frcc)
        frcs = jnp.asarray(frcs) if frcs is not None else jnp.zeros_like(frcc)
        fzcc = jnp.asarray(fzcc) if fzcc is not None else jnp.zeros_like(fzsc)
        fzss = jnp.asarray(fzss) if fzss is not None else jnp.zeros_like(fzsc)
        flcc = jnp.asarray(flcc) if flcc is not None else jnp.zeros_like(flsc)
        flss = jnp.asarray(flss) if flss is not None else jnp.zeros_like(flsc)
        rms = jnp.sqrt(
            jnp.mean(
                frcc * frcc
                + frss * frss
                + frsc * frsc
                + frcs * frcs
                + fzsc * fzsc
                + fzcs * fzcs
                + fzcc * fzcc
                + fzss * fzss
                + flsc * flsc
                + flcs * flcs
                + flcc * flcc
                + flss * flss
            )
        )
        rms_f = float(np.asarray(rms))
        if not np.isfinite(rms_f) or rms_f <= 0.0:
            return max(float(dt_nominal), 1e-12)
        # With this integrator, first-step coefficient update is O(dt^2 * force).
        dt_lim = np.sqrt(max_coeff_delta_rms / max(rms_f, 1e-30))
        dt_eff = min(float(dt_nominal), float(dt_lim))
        return max(dt_eff, 1e-12)

    def _apply_vmec_scale_m1_precond_rhs(frzl_in: TomnspsRZL, mats: dict[str, Any]) -> TomnspsRZL:
        """Apply VMEC `scale_m1_par` factors before the radial preconditioner solve."""
        if (not bool(getattr(cfg, "lconm1", True))) or (int(cfg.mpol) <= 1):
            return frzl_in
        dr = jnp.asarray(mats["dr"])
        dz = jnp.asarray(mats["dz"])
        if dr.shape[0] == 0:
            return frzl_in
        sr = -dr[:, 1, 0]
        sz = -dz[:, 1, 0]
        denom = sr + sz
        fac_r = jnp.where(denom != 0.0, sr / denom, jnp.ones_like(sr))
        fac_z = jnp.where(denom != 0.0, sz / denom, jnp.ones_like(sz))

        ns_full = int(jnp.asarray(frzl_in.frcc).shape[0])
        nsolve = min(ns_full, int(sr.shape[0]))
        fac_r_full = jnp.ones((ns_full,), dtype=jnp.asarray(frzl_in.frcc).dtype).at[:nsolve].set(fac_r[:nsolve])
        fac_z_full = jnp.ones((ns_full,), dtype=jnp.asarray(frzl_in.fzsc).dtype).at[:nsolve].set(fac_z[:nsolve])

        frss = frzl_in.frss
        fzcs = frzl_in.fzcs
        frsc = getattr(frzl_in, "frsc", None)
        fzcc = getattr(frzl_in, "fzcc", None)
        if frss is not None:
            frss = jnp.asarray(frss)
            frss = frss.at[:, 1, :].set(frss[:, 1, :] * fac_r_full[:, None])
        if fzcs is not None:
            fzcs = jnp.asarray(fzcs)
            fzcs = fzcs.at[:, 1, :].set(fzcs[:, 1, :] * fac_z_full[:, None])
        if frsc is not None:
            frsc = jnp.asarray(frsc)
            frsc = frsc.at[:, 1, :].set(frsc[:, 1, :] * fac_r_full[:, None])
        if fzcc is not None:
            fzcc = jnp.asarray(fzcc)
            fzcc = fzcc.at[:, 1, :].set(fzcc[:, 1, :] * fac_z_full[:, None])

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

    def _pop_iteration_histories() -> None:
        def _pop(hist):
            if hist:
                hist.pop()

        for h in (
            include_edge_history,
            zero_m1_history,
            bcovar_update_history,
            w_history,
            fsqr2_history,
            fsqz2_history,
            fsql2_history,
            r00_history,
            z00_history,
            wb_history,
            wp_history,
            w_vmec_history,
            rz_norm_history,
            f_norm1_history,
            gcr2_p_history,
            gcz2_p_history,
            gcl2_p_history,
            fsq1_history,
            fsqr1_history,
            fsqz1_history,
            fsql1_history,
            min_tau_history,
            max_tau_history,
            bad_jacobian_history,
            step_history,
            dt_eff_history,
            update_rms_history,
            w_curr_history,
            w_try_history,
            w_try_ratio_history,
            restart_path_history,
            step_status_history,
            restart_reason_history,
            pre_restart_reason_history,
            time_step_history,
            res0_history,
            res1_history,
            fsq_prev_history,
            bad_growth_streak_history,
            iter1_history,
            grad_rms_history,
        ):
            _pop(h)

    def _maybe_dump_time_control(*, iter_idx: int, fsq: float, fsq0: float, res0: float, res1: float, time_step: float) -> None:
        if os.getenv("VMEC_JAX_DUMP_TIMECONTROL", "") in ("", "0"):
            return
        dump_dir = os.getenv("VMEC_JAX_DUMP_DIR", "")
        if not dump_dir:
            return
        try:
            path = Path(dump_dir) / "time_control.log"
            with path.open("a", encoding="utf-8") as f:
                f.write(
                    f"iter={iter_idx} fsq={fsq:.6e} fsq0={fsq0:.6e} "
                    f"res0={res0:.6e} res1={res1:.6e} time_step={time_step:.6e}\n"
                )
        except Exception:
            return

    def _dump_time_control_trace(*, stage: str, iter2: int, iter1: int, fsq: float, fsq0: float, res0: float, res1: float, time_step: float, irst: int) -> None:
        if os.getenv("VMEC_JAX_DUMP_TIMECONTROL", "") in ("", "0"):
            return
        dump_dir = os.getenv("VMEC_JAX_DUMP_DIR", "")
        if not dump_dir:
            return
        try:
            path = Path(dump_dir) / "time_control_trace.log"
            with path.open("a", encoding="utf-8") as f:
                f.write(
                    f"{int(iter2):8d} {int(iter1):8d} "
                    f"{float(fsq): .16e} {float(fsq0): .16e} "
                    f"{float(res0): .16e} {float(res1): .16e} "
                    f"{float(time_step): .16e} {int(irst):3d} {stage}\n"
                )
        except Exception:
            return

    def _maybe_dump_checkpoint(*, iter_idx: int, fsq: float, fsq0: float, res0: float, res1: float) -> None:
        if os.getenv("VMEC_JAX_DUMP_CHECKPOINT", "") in ("", "0"):
            return
        dump_dir = os.getenv("VMEC_JAX_DUMP_DIR", "")
        if not dump_dir:
            return
        try:
            path = Path(dump_dir) / "checkpoint.log"
            with path.open("a", encoding="utf-8") as f:
                f.write(
                    f"iter={iter_idx} fsq={fsq:.6e} fsq0={fsq0:.6e} res0={res0:.6e} res1={res1:.6e}\n"
                )
        except Exception:
            return

    # VMEC `eqsolve`: if the initial Jacobian changes sign, improve the axis
    # guess *before* the first iteration (no extra iter1). This aligns the
    # zero_m1 gating and time-control history with VMEC2000.
    if bool(vmec2000_control) and (not axis_reset_done) and bool(lmove_axis):
        try:
            k0, _frzl0, _gcr2_0, _gcz2_0, _gcl2_0, _rz_scale0, _l_scale0, _norms0 = _compute_forces_iter(
                state,
                include_edge=False,
                zero_m1=jnp.asarray(1.0, dtype=jnp.asarray(state.Rcos).dtype),
                constraint_precond_diag=zero_precond_diag,
                constraint_tcon=zero_tcon,
                constraint_precond_active=jnp.asarray(False),
                constraint_tcon_active=jnp.asarray(False),
                iter_idx=None,
                iter2=1,
            )
            ptau_min0, ptau_max0 = _ptau_minmax(k0)
            bad_jacobian_ptau = None
            if (ptau_min0 is not None) and (ptau_max0 is not None):
                min_tau_ptau = float(np.asarray(ptau_min0))
                max_tau_ptau = float(np.asarray(ptau_max0))
                bad_jacobian_ptau = (min_tau_ptau < 0.0) and (max_tau_ptau > 0.0)

            bad_jacobian_state = False
            min_tau_state = float("nan")
            max_tau_state = float("nan")
            if badjac_use_state or (bad_jacobian_ptau is None):
                jac0 = vmec_half_mesh_jacobian_from_state(
                    state=state,
                    modes=static.modes,
                    trig=trig,
                    s=s,
                    lconm1=bool(getattr(static.cfg, "lconm1", True)),
                    lthreed=bool(getattr(static.cfg, "lthreed", True)),
                    mask_even=getattr(static, "m_is_even", None),
                    mask_odd=getattr(static, "m_is_odd", None),
                )
                tau0 = jnp.asarray(jac0.tau)
                tau0_use = tau0[1:] if int(tau0.shape[0]) > 1 else tau0
                min_tau_state = float(np.asarray(jnp.min(tau0_use)))
                max_tau_state = float(np.asarray(jnp.max(tau0_use)))
                bad_jacobian_state = (min_tau_state < 0.0) and (max_tau_state > 0.0)

            axis_reset_debug = os.getenv("VMEC_JAX_AXIS_RESET_DEBUG", "").strip().lower() not in ("", "0", "false", "no")
            fsq_phys0_val = None
            try:
                fsqr0 = _norms0.r1 * _norms0.fnorm * _gcr2_0
                fsqz0 = _norms0.r1 * _norms0.fnorm * _gcz2_0
                fsql0 = _norms0.fnormL * _gcl2_0
                fsq_phys0_val = float(np.asarray(fsqr0 + fsqz0 + fsql0))
            except Exception:
                fsq_phys0_val = None

            if bad_jacobian_ptau is None:
                bad_jacobian0 = bad_jacobian_state
            else:
                if badjac_use_state:
                    # Require both ptau and state sign changes to avoid
                    # false-positive axis resets on benign cases.
                    bad_jacobian0 = bool(bad_jacobian_ptau) and bool(bad_jacobian_state)
                else:
                    bad_jacobian0 = bool(bad_jacobian_ptau)
            if bad_jacobian0 and axis_reset_fsq_min > 0.0:
                if (fsq_phys0_val is None) or (not np.isfinite(fsq_phys0_val)):
                    bad_jacobian0 = False
                elif fsq_phys0_val < axis_reset_fsq_min:
                    bad_jacobian0 = False
            if axis_reset_debug:
                try:
                    fsq_debug_val = float("nan") if fsq_phys0_val is None else float(fsq_phys0_val)
                    print(
                        "[axis_reset] fsq0="
                        f"{fsq_debug_val:.6e} "
                        f"axis_reset_fsq_min={axis_reset_fsq_min:.3e} "
                        f"badjac_ptau={bad_jacobian_ptau} badjac_state={bad_jacobian_state} "
                        f"badjac_used={bad_jacobian0}",
                        flush=True,
                    )
                except Exception:
                    pass

            force_axis_reset_init = bool(force_axis_reset) or (
                bool(getattr(cfg, "lthreed", True)) and axis_reset_always_3d
            )
            if bad_jacobian0 or force_axis_reset_init:
                if verbose and bool(vmec2000_control) and bool(verbose_vmec2000_table):
                    if bad_jacobian0 or force_axis_reset_init:
                        print(" INITIAL JACOBIAN CHANGED SIGN!", flush=True)
                    print(" TRYING TO IMPROVE INITIAL MAGNETIC AXIS GUESS", flush=True)
                state = _reset_axis_from_boundary(state, k_guess=k0, full_reset=False, refine_axis_guess=False)
                if verbose and bool(vmec2000_control) and bool(verbose_vmec2000_table):
                    if axis_reset_coeffs is not None:
                        raxis_cc, _raxis_cs, _zaxis_cc, zaxis_cs = axis_reset_coeffs
                        _print_axis_guess(raxis_cc, zaxis_cs)
                axis_reset_done = True
                ijacob = 1
                state_checkpoint = state
                vRcc = jnp.zeros_like(vRcc)
                vRss = jnp.zeros_like(vRcc)
                vZsc = jnp.zeros_like(vRcc)
                vZcs = jnp.zeros_like(vRcc)
                vLsc = jnp.zeros_like(vRcc)
                vLcs = jnp.zeros_like(vRcc)
                res0 = -1.0
                res1 = -1.0
                prev_rz_fsq = 2.0
                vmec2000_cache_valid = False
                cache_precond_diag = None
                cache_tcon = None
                cache_norms = None
                cache_rz_scale = None
                cache_l_scale = None
                cache_rz_norm = None
                cache_f_norm1 = None
                cache_prec_rz_mats = None
                cache_prec_rz_jmax = None
                cache_prec_lam_prec = None
                cache_prec_faclam = None
                cache_prec_lam_debug = None
        except Exception:
            pass

    last_iter2 = 0
    for it in range(max_iter):
        iter2 = it + 1 + int(iter_offset)
        last_iter2 = iter2
        converged = False
        skip_time_control = False
        force_bcovar_update = False
        time_step_report_hold: float | None = None
        while True:
            iter_since_restart = iter2 - iter1
            fsq_prev_before = fsq_prev
            fsq0_prev_before = fsq0_prev
            pre_restart_reason = "none"
            if time_step_report_hold is None:
                time_step_report_hold = float(time_step)
            time_step_report = float(time_step_report_hold)
            if vmec2000_control:
                # VMEC2000 `constrain_m1` logic (residue.f90):
                #   zero gcz(m=1) if (fsqz_prev < 1e-6) OR (iter2 < 2).
                fsqz_prev = float(fsqz2_history[-1]) if fsqz2_history else 1.0
                zero_m1_val = 1.0 if (iter2 < 2) or (fsqz_prev < 1.0e-6) else 0.0
            else:
                # A conservative heuristic early in a restart window.
                zero_m1_val = (
                    1.0
                    if (iter_since_restart < 2) or (len(fsqz2_history) and fsqz2_history[-1] < 1e-6)
                    else 0.0
                )
            zero_m1 = jnp.asarray(zero_m1_val, dtype=jnp.asarray(state.Rcos).dtype)
            if vmec2000_control:
                # VMEC2000 fixed-boundary residuals do not include the edge
                # surface in the force pipeline; keep this disabled for parity.
                include_edge = False
            else:
                include_edge = bool(iter_since_restart < 50) and (float(prev_rz_fsq) < 1e-6)
            if track_history:
                include_edge_history.append(int(bool(include_edge)))
            # `zero_m1` originates from host control flow, so keep the history
            # without forcing an unnecessary device synchronization.
            if track_history:
                zero_m1_history.append(int(zero_m1_val > 0.5))
    
            need_bcovar_update = bool(vmec2000_control) and (
                (not bool(vmec2000_cache_valid))
                or bool(force_bcovar_update)
                or ((iter2 - iter1) % k_preconditioner_update_interval == 0)
            )
            force_bcovar_update = False
            bcovar_update_history.append(int(bool(need_bcovar_update)))
    
            use_cached_precond = bool(vmec2000_control) and bool(vmec2000_cache_valid) and (not bool(need_bcovar_update))
            constraint_precond_diag = (
                cache_precond_diag if (use_cached_precond and cache_precond_diag is not None) else zero_precond_diag
            )
            # VMEC updates tcon only when refreshing the 1D preconditioner
            # blocks; between refreshes it reuses the last tcon profile.
            constraint_tcon_override = cache_tcon if (use_cached_precond and cache_tcon is not None) else zero_tcon
            constraint_precond_active = jnp.asarray(use_cached_precond, dtype=bool)
            constraint_tcon_active = jnp.asarray(use_cached_precond, dtype=bool)

            if profile_active and (not profile_started) and (profile_start_iter is not None) and (iter2 == profile_start_iter):
                if has_jax():
                    try:
                        Path(profile_dir).mkdir(parents=True, exist_ok=True)
                        jax.profiler.start_trace(profile_dir, create_perfetto_trace=profile_perfetto)
                        profile_started = True
                    except Exception:
                        profile_active = False

            t_compute_start = time.perf_counter() if timing_enabled else None
            k, frzl, gcr2, gcz2, gcl2, rz_scale, l_scale, norms_current = _compute_forces_iter(
                state,
                include_edge=bool(include_edge),
                zero_m1=zero_m1,
                constraint_precond_diag=constraint_precond_diag,
                constraint_tcon=constraint_tcon_override,
                constraint_precond_active=constraint_precond_active,
                constraint_tcon_active=constraint_tcon_active,
                iter_idx=_iter_idx_for_dump(iter2),
                iter2=iter2,
            )
            if timing_enabled:
                try:
                    if has_jax():
                        jax.block_until_ready(gcr2)
                except Exception:
                    pass
                timing_stats["compute_forces"] += time.perf_counter() - float(t_compute_start)
            norms_used = cache_norms if (bool(vmec2000_control) and bool(vmec2000_cache_valid) and (not bool(need_bcovar_update))) else norms_current
            fsqr = norms_used.r1 * norms_used.fnorm * gcr2
            fsqz = norms_used.r1 * norms_used.fnorm * gcz2
            fsql = norms_used.fnormL * gcl2
            debug_iter_env = os.getenv("VMEC_JAX_DEBUG_ITER", "").strip()
            if debug_iter_env:
                try:
                    debug_iter = int(debug_iter_env)
                except Exception:
                    debug_iter = -1
                if debug_iter > 0 and int(iter2) == debug_iter:
                    try:
                        gcr2_val = float(np.asarray(gcr2))
                        gcz2_val = float(np.asarray(gcz2))
                        gcl2_val = float(np.asarray(gcl2))
                        fn_val = float(np.asarray(norms_used.fnorm))
                        r1_val = float(np.asarray(norms_used.r1))
                        rcos_sum = float(np.sum(np.asarray(state.Rcos)))
                        zsin_sum = float(np.sum(np.asarray(state.Zsin)))
                        lsin_sum = float(np.sum(np.asarray(state.Lsin)))
                        rcos_ck = float(np.sum(np.asarray(state_checkpoint.Rcos)))
                        zsin_ck = float(np.sum(np.asarray(state_checkpoint.Zsin)))
                        lsin_ck = float(np.sum(np.asarray(state_checkpoint.Lsin)))
                        fsqr_dbg = float(np.asarray(norms_used.r1 * norms_used.fnorm * gcr2))
                        fsqz_dbg = float(np.asarray(norms_used.r1 * norms_used.fnorm * gcz2))
                        fsql_dbg = float(np.asarray(norms_used.fnormL * gcl2))
                        print(
                            f"[nonscan-state] iter={iter2} rcos_sum={rcos_sum:.6e} "
                            f"zsin_sum={zsin_sum:.6e} lsin_sum={lsin_sum:.6e} "
                            f"rcos_ck={rcos_ck:.6e} zsin_ck={zsin_ck:.6e} lsin_ck={lsin_ck:.6e} "
                            f"gcr2={gcr2_val:.6e} gcz2={gcz2_val:.6e} gcl2={gcl2_val:.6e} "
                            f"fnorm={fn_val:.6e} r1={r1_val:.6e} "
                            f"fsqr={fsqr_dbg:.6e} fsqz={fsqz_dbg:.6e} fsql={fsql_dbg:.6e}",
                            flush=True,
                        )
                    except Exception:
                        pass
            if bool(vmec2000_control) and bool(vmec2000_cache_valid) and (not bool(need_bcovar_update)):
                rz_scale = cache_rz_scale
                l_scale = cache_l_scale
            if bool(vmec2000_control) and bool(need_bcovar_update):
                if constraint_tcon0 is None or float(constraint_tcon0) == 0.0:
                    cache_precond_diag = None
                    cache_tcon = jnp.zeros((int(s.shape[0]),), dtype=jnp.asarray(state.Rcos).dtype)
                else:
                    from .vmec_constraints import precondn_diag_axd1_from_bcovar
    
                    ard1, azd1 = precondn_diag_axd1_from_bcovar(
                        trig=trig,
                        s=s,
                        bsq=k.bc.bsq,
                        r12=k.bc.jac.r12,
                        sqrtg=k.bc.jac.sqrtg,
                        ru12=k.bc.jac.ru12,
                        zu12=k.bc.jac.zu12,
                    )
                    cache_precond_diag = (ard1, azd1)
                    cache_tcon = jnp.asarray(k.tcon)
                cache_norms = norms_used
                cache_rz_scale = rz_scale
                cache_l_scale = l_scale
                cache_rz_norm = _rz_norm(state)
                cache_f_norm1 = jnp.where(
                    jnp.asarray(cache_rz_norm) != 0.0,
                    1.0 / jnp.asarray(cache_rz_norm),
                    jnp.asarray(float("inf"), dtype=jnp.asarray(cache_rz_norm).dtype),
                )
                if not bool(cfg.lasym):
                    from .preconditioner_1d_jax import rz_preconditioner_matrices
    
                    cache_prec_lam_prec = _lambda_preconditioner(k.bc)
                    mats, _jmin, jmax = rz_preconditioner_matrices(bc=k.bc, k=k, trig=trig, s=s, cfg=cfg)
                    cache_prec_rz_mats = mats
                    cache_prec_rz_jmax = int(jmax)
                vmec2000_cache_valid = True
            fsqr_f, fsqz_f, fsql_f = _device_get_floats(fsqr, fsqz, fsql)
            fsq0_curr = fsqr_f + fsqz_f + fsql_f
            prev_rz_fsq_before = prev_rz_fsq
            prev_rz_fsq = fsqr_f + fsqz_f

            w_history.append(fsq0_curr)
            fsqr2_history.append(fsqr_f)
            fsqz2_history.append(fsqz_f)
            fsql2_history.append(fsql_f)
            # VMEC printout uses r00 = r1(1,0): axis R at theta=0, zeta=0,
            # evaluated in real space after scalxc (see funct3d.f).
            # For parity diagnostics, sample these scalars on VMEC's screen cadence.
            sample_vmec = bool(vmec2000_control) and _should_sample_vmec2000(int(iter2), int(max_iter))
            need_scalar = bool(sample_vmec) or (bool(verbose) and (not bool(vmec2000_control)))
            if need_scalar:
                try:
                    r00_j = jnp.asarray(k.pr1_even)[0, 0, 0]
                    if bool(cfg.lasym):
                        z00_j = jnp.asarray(k.pz1_even)[0, 0, 0]
                    else:
                        z00_j = jnp.asarray(0.0, dtype=jnp.asarray(r00_j).dtype)
                except Exception:
                    if not np.any(m0_mask):
                        r00_j = jnp.asarray(float("nan"))
                        z00_j = jnp.asarray(float("nan"))
                    else:
                        r00_j = jnp.sum(jnp.asarray(state.Rcos)[0, m0_mask])
                        if bool(cfg.lasym):
                            z00_j = jnp.sum(jnp.asarray(state.Zcos)[0, m0_mask])
                        else:
                            z00_j = jnp.asarray(0.0, dtype=jnp.asarray(r00_j).dtype)
                # `norms_used` may be cached (VMEC2000 `ns4=25` behavior), but
                # `norms_current` already reflects the current bcovar state and
                # therefore matches VMEC's printed wb/wp without recomputing.
                wb_j = jnp.asarray(norms_current.wb)
                wp_j = jnp.asarray(norms_current.wp)
                r00_val, z00_val, wb_val, wp_val = _device_get_floats(r00_j, z00_j, wb_j, wp_j)
                if bool(vmec2000_control):
                    # Match VMEC's printed precision (E11.3) for parity checks.
                    r00_val = float(f"{float(r00_val):.3E}")
                    z00_val = float(f"{float(z00_val):.3E}")
            else:
                r00_val = r00_last
                z00_val = z00_last
                wb_val = wb_last
                wp_val = wp_last
            r00_last = float(r00_val)
            z00_last = float(z00_val)
            wb_last = float(wb_val)
            wp_last = float(wp_val)
            w_vmec_last = (wb_last + wp_last / (gamma - 1.0)) * float(TWOPI * TWOPI)
            if track_history:
                r00_history.append(r00_last)
                z00_history.append(z00_last)
                wb_history.append(wb_last)
                wp_history.append(wp_last)
                w_vmec_history.append(w_vmec_last)
    
            if verbose and (not (bool(vmec2000_control) and bool(verbose_vmec2000_table))):
                print(
                    f"[solve_fixed_boundary_residual_iter] iter={it:03d} fsqr={fsqr_f:.3e} fsqz={fsqz_f:.3e} "
                    f"fsql={fsql_f:.3e} include_edge={include_edge}",
                    flush=True,
                )
            # Terminate on invariant residuals (fsqr/fsqz/fsql), not fsq1.
            if (fsqr_f <= ftol) and (fsqz_f <= ftol) and (fsql_f <= ftol):
                if verbose and not (bool(vmec2000_control) and bool(verbose_vmec2000_table)):
                    print(
                        f"[solve_fixed_boundary_residual_iter] converged: "
                        f"fsqr={fsqr_f:.3e} fsqz={fsqz_f:.3e} fsql={fsql_f:.3e} <= ftol={ftol:.3e}",
                        flush=True,
                    )
                if bool(vmec2000_control) and bool(verbose_vmec2000_table):
                    # Always print the final (converged) iteration row.
                    _print_vmec2000_iter_row(
                        iter_idx=int(iter2),
                        fsqr=fsqr_f,
                        fsqz=fsqz_f,
                        fsql=fsql_f,
                        fsqr1=fsqr1_f,
                        fsqz1=fsqz1_f,
                        fsql1=fsql1_f,
                        delt0r=float(time_step),
                        r00=float(r00_last),
                        w_mhd=float(w_vmec_last),
                        z00=float(z00_last),
                    )
                converged = True
                break
    
            # Precondition forces.
            t_precond_start = time.perf_counter() if timing_enabled else None
            frzl_lam_pre = None
            if bool(vmec2000_control) and bool(cfg.lthreed):
                from .preconditioner_1d_jax import rz_preconditioner_apply, rz_preconditioner_matrices
    
                need_lam_prec = os.getenv("VMEC_JAX_DUMP_LAM", "") not in ("", "0")
                need_lamcal = os.getenv("VMEC_JAX_DUMP_LAMCAL", "") not in ("", "0")
                need_prec_refresh = (not bool(vmec2000_cache_valid)) or (cache_prec_lam_prec is None) or (cache_prec_rz_mats is None) or (cache_prec_rz_jmax is None) or bool(need_bcovar_update)
                if need_prec_refresh:
                    t_prec_refresh_start = time.perf_counter() if timing_enabled else None
                    if need_lamcal:
                        if need_lam_prec:
                            lam_prec, faclam_dump, lam_debug = _lambda_preconditioner(
                                k.bc, return_faclam=True, return_debug=True
                            )
                        else:
                            lam_prec, lam_debug = _lambda_preconditioner(k.bc, return_debug=True)
                            faclam_dump = None
                    else:
                        if need_lam_prec:
                            lam_prec, faclam_dump = _lambda_preconditioner(k.bc, return_faclam=True)
                        else:
                            lam_prec = _lambda_preconditioner(k.bc)
                            faclam_dump = None
                        lam_debug = None
                    mats, _jmin, jmax = rz_preconditioner_matrices(bc=k.bc, k=k, trig=trig, s=s, cfg=cfg)
                    cache_prec_lam_prec = lam_prec
                    cache_prec_faclam = faclam_dump
                    cache_prec_lam_debug = lam_debug
                    cache_prec_rz_mats = mats
                    cache_prec_rz_jmax = int(jmax)
                    if timing_enabled and t_prec_refresh_start is not None:
                        try:
                            if has_jax():
                                jax.block_until_ready(lam_prec)
                        except Exception:
                            pass
                        timing_stats["precond_refresh"] += time.perf_counter() - float(t_prec_refresh_start)
                else:
                    lam_prec = cache_prec_lam_prec
                    mats = cache_prec_rz_mats
                    jmax = int(cache_prec_rz_jmax)
                    faclam_dump = cache_prec_faclam if need_lam_prec else None
                    lam_debug = cache_prec_lam_debug if need_lamcal else None
                _maybe_dump_lam_prec(lam_prec=lam_prec, faclam=faclam_dump, static=static, iter_idx=int(iter2))
                if lam_debug is not None:
                    _maybe_dump_lamcal(lam_debug=lam_debug, static=static, iter_idx=int(iter2))
                frzl_rhs = _apply_vmec_scale_m1_precond_rhs(frzl, mats)
                frzl_rz = rz_preconditioner_apply(
                    frzl_in=frzl_rhs,
                    mats=mats,
                    jmax=jmax,
                    cfg=cfg,
                )
                frzl_lam_pre = frzl_rz
                frcc = jnp.asarray(frzl_rz.frcc)
                frss = frzl_rz.frss
                fzsc = jnp.asarray(frzl_rz.fzsc)
                fzcs = frzl_rz.fzcs
                flsc = jnp.asarray(frzl_rz.flsc) * jnp.asarray(lam_prec)
                flcs = None if frzl_rz.flcs is None else (jnp.asarray(frzl_rz.flcs) * jnp.asarray(lam_prec))
                frsc = jnp.zeros_like(frcc)
                frcs = jnp.zeros_like(frcc)
                fzcc = jnp.zeros_like(fzsc)
                fzss = jnp.zeros_like(fzsc)
                flcc = jnp.zeros_like(flsc)
                flss = jnp.zeros_like(flsc)
                if getattr(frzl_rz, "frsc", None) is not None:
                    frsc = jnp.asarray(frzl_rz.frsc)
                if getattr(frzl_rz, "frcs", None) is not None:
                    frcs = jnp.asarray(frzl_rz.frcs)
                if getattr(frzl_rz, "fzcc", None) is not None:
                    fzcc = jnp.asarray(frzl_rz.fzcc)
                if getattr(frzl_rz, "fzss", None) is not None:
                    fzss = jnp.asarray(frzl_rz.fzss)
                if getattr(frzl_rz, "flcc", None) is not None:
                    flcc = jnp.asarray(frzl_rz.flcc) * jnp.asarray(lam_prec)
                if getattr(frzl_rz, "flss", None) is not None:
                    flss = jnp.asarray(frzl_rz.flss) * jnp.asarray(lam_prec)
            elif not bool(cfg.lthreed):
                from .preconditioner_1d_jax import rz_preconditioner_apply, rz_preconditioner_matrices

                need_lam_prec = os.getenv("VMEC_JAX_DUMP_LAM", "") not in ("", "0")
                need_lamcal = os.getenv("VMEC_JAX_DUMP_LAMCAL", "") not in ("", "0")
                need_prec_refresh = (
                    (not bool(vmec2000_cache_valid))
                    or (cache_prec_lam_prec is None)
                    or (cache_prec_rz_mats is None)
                    or (cache_prec_rz_jmax is None)
                    or bool(need_bcovar_update)
                )
                if need_prec_refresh:
                    t_prec_refresh_start = time.perf_counter() if timing_enabled else None
                    if need_lamcal:
                        if need_lam_prec:
                            lam_prec, faclam_dump, lam_debug = _lambda_preconditioner(
                                k.bc, return_faclam=True, return_debug=True
                            )
                        else:
                            lam_prec, lam_debug = _lambda_preconditioner(k.bc, return_debug=True)
                            faclam_dump = None
                    else:
                        if need_lam_prec:
                            lam_prec, faclam_dump = _lambda_preconditioner(k.bc, return_faclam=True)
                        else:
                            lam_prec = _lambda_preconditioner(k.bc)
                            faclam_dump = None
                        lam_debug = None
                    mats, _jmin, jmax = rz_preconditioner_matrices(bc=k.bc, k=k, trig=trig, s=s, cfg=cfg)
                    cache_prec_lam_prec = lam_prec
                    cache_prec_faclam = faclam_dump
                    cache_prec_lam_debug = lam_debug
                    cache_prec_rz_mats = mats
                    cache_prec_rz_jmax = int(jmax)
                    if timing_enabled and t_prec_refresh_start is not None:
                        try:
                            if has_jax():
                                jax.block_until_ready(lam_prec)
                        except Exception:
                            pass
                        timing_stats["precond_refresh"] += time.perf_counter() - float(t_prec_refresh_start)
                else:
                    lam_prec = cache_prec_lam_prec
                    mats = cache_prec_rz_mats
                    jmax = int(cache_prec_rz_jmax)
                    faclam_dump = cache_prec_faclam if need_lam_prec else None
                    lam_debug = cache_prec_lam_debug if need_lamcal else None
                _maybe_dump_lam_prec(lam_prec=lam_prec, faclam=faclam_dump, static=static, iter_idx=int(iter2))
                if lam_debug is not None:
                    _maybe_dump_lamcal(lam_debug=lam_debug, static=static, iter_idx=int(iter2))
                frzl_rhs = (
                    _apply_vmec_scale_m1_precond_rhs(frzl, mats)
                    if bool(getattr(cfg, "lasym", False))
                    else frzl
                )
                frzl_rz = rz_preconditioner_apply(
                    frzl_in=frzl_rhs,
                    mats=mats,
                    jmax=jmax,
                    cfg=cfg,
                )
                frzl_lam_pre = frzl_rz
                frcc = jnp.asarray(frzl_rz.frcc)
                frss = frzl_rz.frss
                fzsc = jnp.asarray(frzl_rz.fzsc)
                fzcs = frzl_rz.fzcs
                flsc = jnp.asarray(frzl_rz.flsc) * jnp.asarray(lam_prec)
                flcs = None if frzl_rz.flcs is None else (jnp.asarray(frzl_rz.flcs) * jnp.asarray(lam_prec))
                frsc = jnp.zeros_like(frcc)
                frcs = jnp.zeros_like(frcc)
                fzcc = jnp.zeros_like(fzsc)
                fzss = jnp.zeros_like(fzsc)
                flcc = jnp.zeros_like(flsc)
                flss = jnp.zeros_like(flsc)
                if getattr(frzl_rz, "frsc", None) is not None:
                    frsc = jnp.asarray(frzl_rz.frsc)
                if getattr(frzl_rz, "frcs", None) is not None:
                    frcs = jnp.asarray(frzl_rz.frcs)
                if getattr(frzl_rz, "fzcc", None) is not None:
                    fzcc = jnp.asarray(frzl_rz.fzcc)
                if getattr(frzl_rz, "fzss", None) is not None:
                    fzss = jnp.asarray(frzl_rz.fzss)
                if getattr(frzl_rz, "flcc", None) is not None:
                    flcc = jnp.asarray(frzl_rz.flcc) * jnp.asarray(lam_prec)
                if getattr(frzl_rz, "flss", None) is not None:
                    flss = jnp.asarray(frzl_rz.flss) * jnp.asarray(lam_prec)
            else:
                frcc = _apply_radial_tridi(frzl.frcc * rz_scale[:, None, None], precond_radial_alpha)
                frss = (
                    _apply_radial_tridi(frzl.frss * rz_scale[:, None, None], precond_radial_alpha)
                    if frzl.frss is not None
                    else None
                )
                fzsc = _apply_radial_tridi(frzl.fzsc * rz_scale[:, None, None], precond_radial_alpha)
                fzcs = (
                    _apply_radial_tridi(frzl.fzcs * rz_scale[:, None, None], precond_radial_alpha)
                    if frzl.fzcs is not None
                    else None
                )
                flsc = _apply_radial_tridi(frzl.flsc * l_scale[:, None, None], precond_lambda_alpha)
                flcs = (
                    _apply_radial_tridi(frzl.flcs * l_scale[:, None, None], precond_lambda_alpha)
                    if frzl.flcs is not None
                    else None
                )
                frsc = (
                    _apply_radial_tridi(frzl.frsc * rz_scale[:, None, None], precond_radial_alpha)
                    if getattr(frzl, "frsc", None) is not None
                    else jnp.zeros_like(frcc)
                )
                frcs = (
                    _apply_radial_tridi(frzl.frcs * rz_scale[:, None, None], precond_radial_alpha)
                    if getattr(frzl, "frcs", None) is not None
                    else jnp.zeros_like(frcc)
                )
                fzcc = (
                    _apply_radial_tridi(frzl.fzcc * rz_scale[:, None, None], precond_radial_alpha)
                    if getattr(frzl, "fzcc", None) is not None
                    else jnp.zeros_like(fzsc)
                )
                fzss = (
                    _apply_radial_tridi(frzl.fzss * rz_scale[:, None, None], precond_radial_alpha)
                    if getattr(frzl, "fzss", None) is not None
                    else jnp.zeros_like(fzsc)
                )
                flcc = (
                    _apply_radial_tridi(frzl.flcc * l_scale[:, None, None], precond_lambda_alpha)
                    if getattr(frzl, "flcc", None) is not None
                    else jnp.zeros_like(flsc)
                )
                flss = (
                    _apply_radial_tridi(frzl.flss * l_scale[:, None, None], precond_lambda_alpha)
                    if getattr(frzl, "flss", None) is not None
                    else jnp.zeros_like(flsc)
                )
    
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
            if frzl_lam_pre is not None:
                _maybe_dump_lam_gcl(
                    frzl_pre=frzl_lam_pre,
                    frzl_post=frzl_pre,
                    static=static,
                    iter_idx=int(iter2),
                    delta_s=delta_s,
                )
            _maybe_dump_gc(frzl=frzl_pre, static=static, iter_idx=int(iter2), label="precond")
    
            # Mode-diagonal preconditioning in (m, n>=0) storage.
            frcc_u = frcc * w_mode_mn[None, :, :]
            frss_u = (frss if frss is not None else jnp.zeros_like(frcc_u)) * w_mode_mn[None, :, :]
            fzsc_u = fzsc * w_mode_mn[None, :, :]
            fzcs_u = (fzcs if fzcs is not None else jnp.zeros_like(fzsc_u)) * w_mode_mn[None, :, :]
            flsc_u = flsc * w_mode_mn[None, :, :]
            flcs_u = (flcs if flcs is not None else jnp.zeros_like(flsc_u)) * w_mode_mn[None, :, :]
            frsc_u = frsc * w_mode_mn[None, :, :]
            frcs_u = frcs * w_mode_mn[None, :, :]
            fzcc_u = fzcc * w_mode_mn[None, :, :]
            fzss_u = fzss * w_mode_mn[None, :, :]
            flcc_u = flcc * w_mode_mn[None, :, :]
            flss_u = flss * w_mode_mn[None, :, :]
            if timing_enabled:
                try:
                    if has_jax():
                        jax.block_until_ready(flsc_u)
                except Exception:
                    pass
                timing_stats["preconditioner"] += time.perf_counter() - float(t_precond_start)
    
            # VMEC's lambda coefficients can be expressed in multiple scaling
            # conventions (e.g. restart vs. `wout` vs. internal). Allow parity drivers
            # to apply a constant scale to the lambda residual channel before mapping
            # it into coefficient updates.
            if lambda_update_scale != 1.0:
                flsc_u = flsc_u * lambda_update_scale_j
                flcs_u = flcs_u * lambda_update_scale_j
                flcc_u = flcc_u * lambda_update_scale_j
                flss_u = flss_u * lambda_update_scale_j
    
            if auto_flip_force and it == 0:
                # Choose force direction by a tiny trial step on the VMEC residual
                # (fsqr+fsqz+fsql), not magnetic energy. Energy monotonicity is not a
                # reliable proxy for VMEC's preconditioned convergence metrics.
                w_curr = float(fsqr_f + fsqz_f + fsql_f)
                # Use a probe step that is large enough to be numerically decisive,
                # but still small relative to typical pseudo-time updates.
                dt_probe = min(1e-2, 0.1 * float(time_step))
                dR_dir = dt_probe * _mn_cos_to_signed_physical(frcc_u, frss_u)
                dZ_dir = dt_probe * _mn_sin_to_signed_physical(fzsc_u, fzcs_u)
                dL_dir = dt_probe * _mn_sin_to_signed_physical_lambda(flsc_u, flcs_u)
                dR_sin_dir = dt_probe * _mn_sin_to_signed_physical(frsc_u, frcs_u)
                dZ_cos_dir = dt_probe * _mn_cos_to_signed_physical(fzcc_u, fzss_u)
                dL_cos_dir = dt_probe * _mn_cos_to_signed_physical_lambda(flcc_u, flss_u)
                if not bool(cfg.lasym):
                    dR_sin_dir = jnp.zeros_like(dR_sin_dir)
                    dZ_cos_dir = jnp.zeros_like(dZ_cos_dir)
                    dL_cos_dir = jnp.zeros_like(dL_cos_dir)

                def _trial(sign: float) -> float:
                    st_try = VMECState(
                        layout=state.layout,
                        Rcos=jnp.asarray(state.Rcos) + sign * dR_dir,
                        Rsin=jnp.asarray(state.Rsin) + sign * dR_sin_dir,
                        Zcos=jnp.asarray(state.Zcos) + sign * dZ_cos_dir,
                        Zsin=jnp.asarray(state.Zsin) + sign * dZ_dir,
                        Lcos=jnp.asarray(state.Lcos) + sign * dL_cos_dir,
                        Lsin=jnp.asarray(state.Lsin) + sign * dL_dir,
                    )
                    _, _, gcr2_t, gcz2_t, gcl2_t, _, _, norms_t = _compute_forces_iter(
                        st_try,
                        include_edge=True,
                        zero_m1=zero_m1,
                        constraint_precond_diag=constraint_precond_diag,
                        constraint_tcon=constraint_tcon_override,
                        constraint_precond_active=constraint_precond_active,
                        constraint_tcon_active=constraint_tcon_active,
                        iter2=iter2,
                    )
                    fsqr_t, fsqz_t, fsql_t = _fsq_from_norms(
                        norms_t,
                        gcr2_in=gcr2_t,
                        gcz2_in=gcz2_t,
                        gcl2_in=gcl2_t,
                    )
                    return float(np.asarray(fsqr_t + fsqz_t + fsql_t))
    
                w_pos = _trial(+1.0)
                w_neg = _trial(-1.0)
                if np.isfinite(w_neg) and np.isfinite(w_pos) and (w_neg < w_pos):
                    flip_sign = -1.0
                    if verbose and not (bool(vmec2000_control) and bool(verbose_vmec2000_table)):
                        print(
                            "[solve_fixed_boundary_residual_iter] flipping force sign "
                            f"(w_curr={w_curr:.3e} w_pos={w_pos:.3e} w_neg={w_neg:.3e})"
                        )
    
            # Damping for the fixed-point update.
            gcr2_p, gcz2_p, gcl2_p = vmec_gcx2_from_tomnsps(
                frzl=frzl_pre,
                lconm1=bool(getattr(static.cfg, "lconm1", True)),
                apply_m1_constraints=False,
                include_edge=True,
                apply_scalxc=False,
                s=s,
            )
            if bool(vmec2000_control) and bool(vmec2000_cache_valid) and (cache_rz_norm is not None) and (cache_f_norm1 is not None):
                rz_norm = jnp.asarray(cache_rz_norm)
                f_norm1 = jnp.asarray(cache_f_norm1)
            else:
                rz_norm = _rz_norm(state)
                f_norm1 = jnp.where(rz_norm != 0.0, 1.0 / rz_norm, jnp.asarray(float("inf"), dtype=rz_norm.dtype))
            fsqr1 = gcr2_p * f_norm1
            fsqz1 = gcz2_p * f_norm1
            if bool(vmec2000_control):
                # VMEC2000 `residue.f90`: fsql1 = hs * SUM( (faclam*gcl)**2 ) over all js.
                flsc_pre = jnp.asarray(frzl_pre.flsc)
                gcl2_full = jnp.sum(flsc_pre[1:] * flsc_pre[1:])
                if frzl_pre.flcs is not None:
                    flcs_pre = jnp.asarray(frzl_pre.flcs)
                    gcl2_full = gcl2_full + jnp.sum(flcs_pre[1:] * flcs_pre[1:])
                if getattr(frzl_pre, "flcc", None) is not None:
                    flcc_pre = jnp.asarray(frzl_pre.flcc)
                    gcl2_full = gcl2_full + jnp.sum(flcc_pre[1:] * flcc_pre[1:])
                if getattr(frzl_pre, "flss", None) is not None:
                    flss_pre = jnp.asarray(frzl_pre.flss)
                    gcl2_full = gcl2_full + jnp.sum(flss_pre[1:] * flss_pre[1:])
                fsql1 = gcl2_full * delta_s
            else:
                fsql1 = gcl2_p * delta_s
            if os.getenv("VMEC_JAX_DUMP_LAM", "") not in ("", "0") and frzl_lam_pre is None:
                gcr2_raw, gcz2_raw, gcl2_raw = vmec_gcx2_from_tomnsps(
                    frzl=frzl,
                    lconm1=bool(getattr(static.cfg, "lconm1", True)),
                    apply_m1_constraints=False,
                    include_edge=True,
                    apply_scalxc=False,
                    s=s,
                )
                fsql1_pre = gcl2_raw * delta_s
                _maybe_dump_lam_fsql1(
                    fsql1_pre=fsql1_pre,
                    fsql1_post=fsql1,
                    static=static,
                    iter_idx=int(iter2),
                )
            (
                fsqr1_f,
                fsqz1_f,
                fsql1_f,
                rz_norm_f,
                f_norm1_f,
                gcr2_p_f,
                gcz2_p_f,
                gcl2_p_f,
            ) = _device_get_floats(
                fsqr1,
                fsqz1,
                fsql1,
                rz_norm,
                f_norm1,
                gcr2_p,
                gcz2_p,
                gcl2_p,
            )
            fsq1 = fsqr1_f + fsqz1_f + fsql1_f
            if track_history:
                rz_norm_history.append(rz_norm_f)
                f_norm1_history.append(f_norm1_f)
                gcr2_p_history.append(gcr2_p_f)
                gcz2_p_history.append(gcz2_p_f)
                gcl2_p_history.append(gcl2_p_f)
                fsq1_history.append(fsq1)
                fsqr1_history.append(fsqr1_f)
                fsqz1_history.append(fsqz1_f)
                fsql1_history.append(fsql1_f)

            # Jacobian sign-change check (VMEC jacobian.f sets irst=2).
            bad_jacobian = False
            if bool(reference_mode) or bool(vmec2000_control):
                ptau_min, ptau_max = _ptau_minmax(k)
                min_tau_ptau = max_tau_ptau = None
                bad_jacobian_ptau = None
                if ptau_min is not None and ptau_max is not None:
                    min_tau_ptau, max_tau_ptau = _device_get_floats(ptau_min, ptau_max)
                    if bool(vmec2000_control):
                        tau_tol = max(abs(ptau_tol), 0.0)
                        bad_jacobian_ptau = (min_tau_ptau < -tau_tol) and (max_tau_ptau > tau_tol)
                    else:
                        tau_scale = max(abs(min_tau_ptau), abs(max_tau_ptau))
                        tau_tol = max(1.0e-12, 1.0e-3 * tau_scale)
                        bad_jacobian_ptau = (min_tau_ptau < -tau_tol) and (max_tau_ptau > tau_tol)

                need_state_jac = (
                    badjac_use_state
                    or dump_ptau_state
                    or (iter2 <= 2)
                    or (bad_jacobian_ptau is None)
                    or bool(bad_jacobian_ptau)
                )
                if need_state_jac:
                    jac_state = vmec_half_mesh_jacobian_from_state(
                        state=state,
                        modes=static.modes,
                        trig=trig,
                        s=s,
                        lconm1=bool(getattr(static.cfg, "lconm1", True)),
                        lthreed=bool(getattr(static.cfg, "lthreed", True)),
                        mask_even=getattr(static, "m_is_even", None),
                        mask_odd=getattr(static, "m_is_odd", None),
                    )
                    tau = jnp.asarray(jac_state.tau)
                    if int(tau.size) > 0:
                        tau_use = tau[1:] if int(tau.shape[0]) > 1 else tau
                        tau_min = jnp.min(tau_use)
                        tau_max = jnp.max(tau_use)
                        min_tau_state, max_tau_state = _device_get_floats(tau_min, tau_max)
                    else:
                        min_tau_state = float("nan")
                        max_tau_state = float("nan")
                    if np.isfinite(min_tau_state) and np.isfinite(max_tau_state):
                        if bool(vmec2000_control):
                            tau_tol = max(abs(ptau_tol), 0.0)
                            bad_jacobian_state = (min_tau_state < -tau_tol) and (max_tau_state > tau_tol)
                        else:
                            tau_scale = max(abs(min_tau_state), abs(max_tau_state))
                            tau_tol = max(1.0e-12, 1.0e-3 * tau_scale)
                            bad_jacobian_state = (min_tau_state < -tau_tol) and (max_tau_state > tau_tol)
                    else:
                        bad_jacobian_state = False
                else:
                    min_tau_state = float("nan")
                    max_tau_state = float("nan")
                    bad_jacobian_state = False

                if badjac_use_state:
                    bad_jacobian = bad_jacobian_state
                    min_tau = min_tau_state
                    max_tau = max_tau_state
                else:
                    bad_jacobian = bool(bad_jacobian_ptau) if bad_jacobian_ptau is not None else False
                    min_tau = min_tau_ptau if min_tau_ptau is not None else float("nan")
                    max_tau = max_tau_ptau if max_tau_ptau is not None else float("nan")

                _maybe_dump_ptau(
                    iter_idx=int(iter2),
                    ptau_min=float(min_tau_ptau if min_tau_ptau is not None else float("nan")),
                    ptau_max=float(max_tau_ptau if max_tau_ptau is not None else float("nan")),
                    tau_min_state=min_tau_state if np.isfinite(min_tau_state) else None,
                    tau_max_state=max_tau_state if np.isfinite(max_tau_state) else None,
                    badjac_ptau=bad_jacobian_ptau,
                    badjac_state=bad_jacobian_state,
                    badjac_used=bool(bad_jacobian),
                    mode=badjac_mode,
                    label="iter",
                )

                if np.isfinite(min_tau) and np.isfinite(max_tau):
                    if track_history:
                        min_tau_history.append(min_tau)
                        max_tau_history.append(max_tau)
                        bad_jacobian_history.append(int(bool(bad_jacobian)))
                    if bad_jacobian and os.getenv("VMEC_JAX_DUMP_BADJAC", "") not in ("", "0"):
                        dump_dir = os.getenv("VMEC_JAX_DUMP_DIR", "")
                        if dump_dir:
                            try:
                                path = Path(dump_dir) / "bad_jacobian.log"
                                with path.open("a", encoding="utf-8") as f:
                                    f.write(
                                        f"iter={iter2} min_tau={min_tau:.6e} max_tau={max_tau:.6e}\n"
                                    )
                            except Exception:
                                pass
                else:
                    if track_history:
                        min_tau_history.append(float("nan"))
                        max_tau_history.append(float("nan"))
                        bad_jacobian_history.append(0)
            else:
                if track_history:
                    min_tau_history.append(float("nan"))
                    max_tau_history.append(float("nan"))
                    bad_jacobian_history.append(0)

            # VMEC eqsolve: after the first evolve step, if the Jacobian is bad
            # and ijacob==0, retry with an improved axis guess.
            if (
                bool(vmec2000_control)
                and (not axis_reset_done)
                and bool(lmove_axis)
                and (iter2 == 1)
            ):
                fsq_curr = fsqr_f + fsqz_f + fsql_f
                huge_initial_forces = (not np.isfinite(fsq_curr)) or (fsq_curr > 1.0e2)
                force_axis_reset_init = bool(force_axis_reset) or (
                    bool(getattr(cfg, "lthreed", True)) and axis_reset_always_3d
                )
                if (not force_axis_reset_init) and axis_reset_fsq_min > 0.0:
                    if np.isfinite(fsq_curr) and (fsq_curr < axis_reset_fsq_min):
                        bad_jacobian = False
                        huge_initial_forces = False
                if bad_jacobian or huge_initial_forces or force_axis_reset_init:
                    if verbose and bool(vmec2000_control) and bool(verbose_vmec2000_table):
                        if bad_jacobian or force_axis_reset_init:
                            print(" INITIAL JACOBIAN CHANGED SIGN!", flush=True)
                        print(" TRYING TO IMPROVE INITIAL MAGNETIC AXIS GUESS", flush=True)
                    state = _reset_axis_from_boundary(state, k_guess=k, full_reset=False, refine_axis_guess=False)
                    if verbose and bool(vmec2000_control) and bool(verbose_vmec2000_table):
                        if axis_reset_coeffs is not None:
                            raxis_cc, _raxis_cs, _zaxis_cc, zaxis_cs = axis_reset_coeffs
                            _print_axis_guess(raxis_cc, zaxis_cs)
                    state_checkpoint = state
                    vRcc = jnp.zeros_like(vRcc)
                    vRss = jnp.zeros_like(vRcc)
                    vZsc = jnp.zeros_like(vRcc)
                    vZcs = jnp.zeros_like(vRcc)
                    vLsc = jnp.zeros_like(vRcc)
                    vLcs = jnp.zeros_like(vRcc)
                    time_step = float(time_step)
                    ijacob = 1
                    axis_reset_done = True
                    iter1 = iter2
                    bad_growth_streak = 0
                    inv_tau = [0.15 / time_step] * k_ndamp
                    vmec2000_cache_valid = False
                    cache_precond_diag = None
                    cache_tcon = None
                    cache_norms = None
                    cache_rz_scale = None
                    cache_l_scale = None
                    cache_rz_norm = None
                    cache_f_norm1 = None
                    cache_prec_rz_mats = None
                    cache_prec_rz_jmax = None
                    cache_prec_lam_prec = None
                    cache_prec_faclam = None
                    cache_prec_lam_debug = None
                    _pop_iteration_histories()
                    prev_rz_fsq = prev_rz_fsq_before
                    # VMEC restarts the iteration after axis reset without
                    # advancing the iteration counter. Emulate that by
                    # repeating iter2==1 on the next loop pass.
                    if iter2 == 1:
                        iter_offset -= 1
                    continue

            # VMEC-style time-step control: VMEC2000's `TimeStepControl` + `restart_iter`.
            if bool(vmec2000_control) and (not skip_time_control):
                fsq0 = fsq0_curr  # physical residual on current state
                # VMEC's TimeStepControl uses the *previous* preconditioned
                # residual (fsq) which is updated at the end of evolve.f.
                # VMEC's TimeStepControl uses `fsq` from the *previous* evolve
                # step (initialized to 1.0). It does not switch to fsq1 when
                # iter2 == iter1 (restart window).
                fsq = fsq_prev
                irst_tc = 1
                if bool(bad_jacobian) and (iter2 > iter1):
                    # VMEC's irst=2 path: use previous physical residual when
                    # the Jacobian changes sign.
                    irst_tc = 2
                    fsq0 = fsq0_prev
                if (iter2 == iter1) or (res0 < 0.0) or (res1 < 0.0):
                    res0 = fsq
                    res1 = fsq0
                    state_checkpoint = state
                    _dump_time_control_trace(
                        stage="init",
                        iter2=int(iter2),
                        iter1=int(iter1),
                        fsq=float(fsq),
                        fsq0=float(fsq0),
                        res0=float(res0),
                        res1=float(res1),
                        time_step=float(time_step),
                        irst=int(irst_tc),
                    )
                    _maybe_dump_checkpoint(iter_idx=int(iter2), fsq=float(fsq), fsq0=float(fsq0), res0=float(res0), res1=float(res1))
                res0 = min(res0, fsq)
                res1 = min(res1, fsq0)
                _dump_time_control_trace(
                    stage="pre",
                    iter2=int(iter2),
                    iter1=int(iter1),
                    fsq=float(fsq),
                    fsq0=float(fsq0),
                    res0=float(res0),
                    res1=float(res1),
                    time_step=float(time_step),
                    irst=int(irst_tc),
                )
                if (fsq <= res0) and (fsq0 <= res1) and (irst_tc == 1):
                    _dump_time_control_trace(
                        stage="checkpoint",
                        iter2=int(iter2),
                        iter1=int(iter1),
                        fsq=float(fsq),
                        fsq0=float(fsq0),
                        res0=float(res0),
                        res1=float(res1),
                        time_step=float(time_step),
                        irst=int(irst_tc),
                    )
                    state_checkpoint = state
                    _maybe_dump_checkpoint(iter_idx=int(iter2), fsq=float(fsq), fsq0=float(fsq0), res0=float(res0), res1=float(res1))
                if (irst_tc == 1) and ((iter2 - iter1) > 10) and (
                    (fsq > vmec2000_fact * max(res0, 1e-30)) or (fsq0 > vmec2000_fact * max(res1, 1e-30))
                ):
                    irst_tc = 3
                if irst_tc != 1:
                    _maybe_dump_time_control(
                        iter_idx=int(iter2),
                        fsq=float(fsq),
                        fsq0=float(fsq0),
                        res0=float(res0),
                        res1=float(res1),
                        time_step=float(time_step),
                    )
                    pre_restart_reason = "bad_jacobian" if irst_tc == 2 else "time_control"
                    state = state_checkpoint
                    vRcc = jnp.zeros_like(vRcc)
                    vRss = jnp.zeros_like(vRss)
                    vZsc = jnp.zeros_like(vZsc)
                    vZcs = jnp.zeros_like(vZcs)
                    vLsc = jnp.zeros_like(vLsc)
                    vLcs = jnp.zeros_like(vLcs)
                    vRsc = jnp.zeros_like(vRsc)
                    vRcs = jnp.zeros_like(vRcs)
                    vZcc = jnp.zeros_like(vZcc)
                    vZss = jnp.zeros_like(vZss)
                    vLcc = jnp.zeros_like(vLcc)
                    vLss = jnp.zeros_like(vLss)
                    iter1_prev = int(iter1)
                    time_step_prev = float(time_step)
                    _dump_time_control_trace(
                        stage="restart",
                        iter2=int(iter2),
                        iter1=iter1_prev,
                        fsq=float(fsq),
                        fsq0=float(fsq0),
                        res0=float(res0),
                        res1=float(res1),
                        time_step=time_step_prev,
                        irst=int(irst_tc),
                    )
                    # VMEC2000 `restart_iter`: irst=2 (bad-jac) -> dt*0.9,
                    # irst=3 (time-control) -> dt/1.03.
                    if irst_tc == 2:
                        time_step = max(restart_badjac_factor * time_step, 1e-12)
                        ijacob += 1
                        step_status = "restart_bad_jacobian"
                        restart_reason = "bad_jacobian"
                    else:
                        time_step = max(time_step / restart_badprog_factor, 1e-12)
                        step_status = "restart_time_control"
                        restart_reason = "time_control"
                    bad_resets += 1
                    iter1 = iter2
                    bad_growth_streak = 0
                    fsq_prev = fsq_prev_before
                    fsq0_prev = fsq0_prev_before
                    inv_tau = [0.15 / time_step] * k_ndamp
                    vmec2000_cache_valid = False
                    cache_precond_diag = None
                    cache_tcon = None
                    cache_norms = None
                    cache_rz_scale = None
                    cache_l_scale = None
                    cache_rz_norm = None
                    cache_f_norm1 = None
                    cache_prec_rz_mats = None
                    cache_prec_rz_jmax = None
                    cache_prec_lam_prec = None
                    cache_prec_faclam = None
                    cache_prec_lam_debug = None
                    force_bcovar_update = True
                    if track_history:
                        step_history.append(0.0)
                        dt_eff_history.append(0.0)
                        update_rms_history.append(0.0)
                        w_curr_history.append(float(fsqr_f + fsqz_f + fsql_f))
                        w_try_history.append(float("nan"))
                        w_try_ratio_history.append(float("nan"))
                        restart_path_history.append(
                            "vmec2000_bad_jacobian" if irst_tc == 2 else "vmec2000_time_control"
                        )
                        step_status_history.append(step_status)
                        restart_reason_history.append(restart_reason)
                        pre_restart_reason_history.append(pre_restart_reason)
                        time_step_history.append(float(time_step))
                        res0_history.append(float(res0))
                        res1_history.append(float(res1))
                        fsq_prev_history.append(float(fsq_prev))
                        bad_growth_streak_history.append(int(bad_growth_streak))
                        iter1_history.append(int(iter1))
                        grad_rms_history.append(float(np.sqrt(max(fsqr_f + fsqz_f + fsql_f, 0.0))))
                    _pop_iteration_histories()
                    prev_rz_fsq = prev_rz_fsq_before
                    skip_time_control = True
                    continue
    
            # --- time-step control trackers + optional restart triggers ---
            fsq = fsqr_f + fsqz_f + fsql_f
            fsq_res = fsq if bool(reference_mode) else fsq1
            if bool(vmec2000_control):
                # VMEC2000 updates res0 only inside TimeStepControl. To keep the
                # generic tracker from drifting, only tighten res0 when the
                # preconditioned residual decreases relative to the previous
                # iteration.
                if (fsq_res <= fsq_prev) and np.isfinite(fsq_res):
                    res0 = min(res0, fsq_res)
                res0_old = res0
            else:
                if (iter2 == iter1) or (res0 < 0.0):
                    res0 = fsq_res
                res0_old = res0
                res0 = min(res0, fsq_res)
    
            # Store a "good" checkpoint once residual has improved for many
            # iterations since the last restart marker.
            if (not bool(vmec2000_control)) and (fsq1 <= res0_old) and ((iter2 - iter1) > 10):
                state_checkpoint = state

            # Restart triggers (bad progress / bad Jacobian proxy).
            # `bad_jacobian` computed above (before TimeStepControl) so that
            # VMEC's irst=2 restart takes precedence over time-control restarts.
            if stage_prev_fsq is not None and iter2 == 1 and pre_restart_reason == "none":
                try:
                    prev_stage_fsq_val = float(stage_prev_fsq)
                except Exception:
                    prev_stage_fsq_val = None
                if prev_stage_fsq_val is not None and np.isfinite(prev_stage_fsq_val):
                    if fsq > (prev_stage_fsq_val * stage_transition_factor):
                        pre_restart_reason = "stage_transition"
    
            huge_initial_forces = False
            if iter2 == 1 and lmove_axis:
                fsq_init = float(fsq)
                huge_initial_forces = (not np.isfinite(fsq_init)) or (fsq_init > 1.0e2)
            if fsq_res > 100.0 * max(res0, 1e-30):
                bad_growth_streak += 1
            else:
                bad_growth_streak = 0

            vmecpp_bad_progress = False
            if vmecpp_restart:
                vmecpp_bad_progress = (
                    (iter2 - iter1) > (k_preconditioner_update_interval // 2)
                    and (iter2 > 2 * k_preconditioner_update_interval)
                    and ((fsqr_f + fsqz_f) > 1.0e-2)
                )
    
            if bool(reference_mode):
                # Conservative restart logic used in the reference-mode trace.
                if bad_jacobian and (fsq > 1.0e1):
                    pre_restart_reason = "bad_jacobian"
                elif (iter2 > iter1) and (fsq > 100.0 * max(res0, 1e-30)):
                    pre_restart_reason = "bad_jacobian"
                elif (
                    (iter2 - iter1) > (k_preconditioner_update_interval // 2)
                    and (iter2 > 2 * k_preconditioner_update_interval)
                    and ((fsqr_f + fsqz_f) > 1.0e-2)
                ):
                    pre_restart_reason = "bad_progress"
            elif bool(vmec2000_control):
                # VMEC cadence: restart immediately on a Jacobian sign change
                # (irst=2 path in jacobian.f + TimeStepControl).
                if bad_jacobian and (iter2 > iter1):
                    pre_restart_reason = "bad_jacobian"
                elif vmecpp_bad_progress:
                    pre_restart_reason = "bad_progress_vmecpp"
            else:
                if vmecpp_bad_progress:
                    pre_restart_reason = "bad_progress_vmecpp"
                elif (iter2 > (iter1 + 8)) and (bad_growth_streak >= 2):
                    pre_restart_reason = "bad_jacobian"
                elif (
                    (iter2 - iter1) > (k_preconditioner_update_interval // 2)
                    and (iter2 > 2 * k_preconditioner_update_interval)
                    and (fsq1 > 5.0 * max(res0, 1e-30))
                    and (fsq1 > 0.95 * max(fsq_prev, 1e-30))
                ):
                    pre_restart_reason = "bad_progress"
    
            if use_restart_triggers and pre_restart_reason != "none":
                state_before_restart = state
                vRcc_before = vRcc
                vRss_before = vRss
                vZsc_before = vZsc
                vZcs_before = vZcs
                vLsc_before = vLsc
                vLcs_before = vLcs
                vRsc_before = vRsc
                vRcs_before = vRcs
                vZcc_before = vZcc
                vZss_before = vZss
                vLcc_before = vLcc
                vLss_before = vLss
                state = state_checkpoint
                vRcc = jnp.zeros_like(vRcc)
                vRss = jnp.zeros_like(vRss)
                vZsc = jnp.zeros_like(vZsc)
                vZcs = jnp.zeros_like(vZcs)
                vLsc = jnp.zeros_like(vLsc)
                vLcs = jnp.zeros_like(vLcs)
                vRsc = jnp.zeros_like(vRsc)
                vRcs = jnp.zeros_like(vRcs)
                vZcc = jnp.zeros_like(vZcc)
                vZss = jnp.zeros_like(vZss)
                vLcc = jnp.zeros_like(vLcc)
                vLss = jnp.zeros_like(vLss)
                if pre_restart_reason == "bad_jacobian":
                    time_step = max(restart_badjac_factor * time_step, 1e-12)
                    ijacob += 1
                    step_status = "restart_bad_jacobian"
                elif pre_restart_reason == "stage_transition":
                    time_step = max(time_step * stage_transition_scale, 1e-12)
                    step_status = "restart_stage_transition"
                else:
                    time_step = max(time_step / restart_badprog_factor, 1e-12)
                    step_status = "restart_bad_progress"
                if bool(huge_initial_forces) and (pre_restart_reason == "bad_jacobian"):
                    huge_force_restart_count += 1
                else:
                    huge_force_restart_count = 0
                if ijacob in (25, 50):
                    scale = 0.98 if ijacob < 50 else 0.96
                    time_step = max(scale * float(step_size), 1e-12)
                time_step_iter = float(time_step)
                bad_resets += 1
                iter1 = iter2
                bad_growth_streak = 0
                fsq_prev = fsq_prev_before
                fsq0_prev = fsq0_prev_before
                inv_tau = [0.15 / time_step] * k_ndamp
                if not bool(vmec2000_control):
                    vmec2000_cache_valid = False
                    cache_precond_diag = None
                    cache_tcon = None
                    cache_norms = None
                    cache_rz_scale = None
                    cache_l_scale = None
                    cache_rz_norm = None
                    cache_f_norm1 = None
                    cache_prec_rz_mats = None
                    cache_prec_rz_jmax = None
                    cache_prec_lam_prec = None
                    cache_prec_faclam = None
                    cache_prec_lam_debug = None
                else:
                    vmec2000_cache_valid = False
                    cache_precond_diag = None
                    cache_tcon = None
                    cache_norms = None
                    cache_rz_scale = None
                    cache_l_scale = None
                    cache_rz_norm = None
                    cache_f_norm1 = None
                    cache_prec_rz_mats = None
                    cache_prec_rz_jmax = None
                    cache_prec_lam_prec = None
                    cache_prec_faclam = None
                    cache_prec_lam_debug = None
                    force_bcovar_update = True
                if track_history:
                    step_history.append(0.0)
                    dt_eff_history.append(0.0)
                    update_rms_history.append(0.0)
                    w_curr_history.append(float(fsqr_f + fsqz_f + fsql_f))
                    w_try_history.append(float("nan"))
                    w_try_ratio_history.append(float("nan"))
                    restart_path_history.append("pre_restart_trigger")
                    step_status_history.append(step_status)
                    restart_reason_history.append(pre_restart_reason)
                    pre_restart_reason_history.append(pre_restart_reason)
                    time_step_history.append(time_step_iter)
                    res0_history.append(float(res0))
                    res1_history.append(float(res1))
                    fsq_prev_history.append(float(fsq_prev))
                    bad_growth_streak_history.append(int(bad_growth_streak))
                    iter1_history.append(int(iter1))
                    grad_rms_history.append(float(np.sqrt(max(fsqr_f + fsqz_f + fsql_f, 0.0))))
                if verbose:
                    if bool(vmec2000_control) and bool(verbose_vmec2000_table):
                        # VMEC does not print rejected restart steps.
                        pass
                    else:
                        print(
                            f"[solve_fixed_boundary_residual_iter] iter={it:03d} "
                            f"dt_eff=0.000e+00 update_rms=0.000e+00 "
                            f"fsqr1={fsqr1_f:.3e} fsqz1={fsqz1_f:.3e} fsql1={fsql1_f:.3e} "
                            f"step_status={step_status}",
                            flush=True,
                        )
                _maybe_dump_xc(
                    state=state_before_restart,
                    vRcc=vRcc_before,
                    vRss=vRss_before,
                    vZsc=vZsc_before,
                    vZcs=vZcs_before,
                    vLsc=vLsc_before,
                    vLcs=vLcs_before,
                    static=static,
                    iter_idx=int(iter2),
                )
                _pop_iteration_histories()
                prev_rz_fsq = prev_rz_fsq_before
                skip_time_control = True
                continue
    
            break
        if profile_started and (profile_start_iter is not None) and (iter2 == profile_start_iter):
            if has_jax():
                try:
                    jax.block_until_ready(state.Rcos)
                    jax.profiler.stop_trace()
                except Exception:
                    pass
            profile_started = False
            profile_active = False
        if converged:
            break
        if iter2 == iter1:
            inv_tau = [0.15 / time_step] * k_ndamp
        else:
            invtau_num = 0.0 if fsq1 == 0.0 else min(abs(np.log(fsq1 / fsq_prev)), 0.15)
            inv_tau = inv_tau[1:] + [invtau_num / time_step]
        fsq_prev = fsq1
        fsq0_prev = fsq0_curr

        otav = float(np.sum(inv_tau)) / float(k_ndamp)
        dtau = time_step * otav / 2.0
        b1 = 1.0 - dtau
        fac = 1.0 / (1.0 + dtau)

        t_update_start = time.perf_counter() if timing_enabled else None
        if bool(strict_update):
            # Strict update semantics: one preconditioned momentum update per
            # iteration in (m, n>=0) storage, no line-search accept/reject.
            w_curr = fsqr_f + fsqz_f + fsql_f
            state_backup = state
            dt_eff = float(time_step)
            if bool(limit_dt_from_force):
                dt_eff = _safe_dt_from_force(
                    dt_nominal=time_step,
                    frcc=frcc_u,
                    frss=frss_u,
                    fzsc=fzsc_u,
                    fzcs=fzcs_u,
                    flsc=flsc_u,
                    flcs=flcs_u,
                    frsc=frsc_u,
                    frcs=frcs_u,
                    fzcc=fzcc_u,
                    fzss=fzss_u,
                    flcc=flcc_u,
                    flss=flss_u,
                )

            # Momentum semantics: v <- fac*(b1*v + dt*F), x <- x + dt*v.
            # Do not drop the dt factor in the force term; otherwise updates
            # scale like O(dt) instead of O(dt^2) and can immediately blow up.
            force_scale = float(dt_eff)

            vRcc = fac * (b1 * vRcc + force_scale * (flip_sign * jnp.asarray(frcc_u)))
            vRss = fac * (b1 * vRss + force_scale * (flip_sign * jnp.asarray(frss_u)))
            vRsc = fac * (b1 * vRsc + force_scale * (flip_sign * jnp.asarray(frsc_u)))
            vRcs = fac * (b1 * vRcs + force_scale * (flip_sign * jnp.asarray(frcs_u)))
            vZsc = fac * (b1 * vZsc + force_scale * (flip_sign * jnp.asarray(fzsc_u)))
            vZcs = fac * (b1 * vZcs + force_scale * (flip_sign * jnp.asarray(fzcs_u)))
            vZcc = fac * (b1 * vZcc + force_scale * (flip_sign * jnp.asarray(fzcc_u)))
            vZss = fac * (b1 * vZss + force_scale * (flip_sign * jnp.asarray(fzss_u)))
            vLsc = fac * (b1 * vLsc + force_scale * (flip_sign * jnp.asarray(flsc_u)))
            vLcs = fac * (b1 * vLcs + force_scale * (flip_sign * jnp.asarray(flcs_u)))
            vLcc = fac * (b1 * vLcc + force_scale * (flip_sign * jnp.asarray(flcc_u)))
            vLss = fac * (b1 * vLss + force_scale * (flip_sign * jnp.asarray(flss_u)))

            update_rms = float(
                np.asarray(
                    jnp.sqrt(
                        jnp.mean(
                            (dt_eff * vRcc) ** 2
                            + (dt_eff * vRss) ** 2
                            + (dt_eff * vRsc) ** 2
                            + (dt_eff * vRcs) ** 2
                            + (dt_eff * vZsc) ** 2
                            + (dt_eff * vZcs) ** 2
                            + (dt_eff * vZcc) ** 2
                            + (dt_eff * vZss) ** 2
                            + (dt_eff * vLsc) ** 2
                            + (dt_eff * vLcs) ** 2
                            + (dt_eff * vLcc) ** 2
                            + (dt_eff * vLss) ** 2
                        )
                    )
                )
            )
            if bool(limit_update_rms) and np.isfinite(update_rms) and (update_rms > max_update_rms):
                scl = max_update_rms / max(update_rms, 1e-30)
                vRcc = vRcc * scl
                vRss = vRss * scl
                vRsc = vRsc * scl
                vRcs = vRcs * scl
                vZsc = vZsc * scl
                vZcs = vZcs * scl
                vZcc = vZcc * scl
                vZss = vZss * scl
                vLsc = vLsc * scl
                vLcs = vLcs * scl
                vLcc = vLcc * scl
                vLss = vLss * scl
                update_rms = float(
                    np.asarray(
                        jnp.sqrt(
                            jnp.mean(
                                (dt_eff * vRcc) ** 2
                                + (dt_eff * vRss) ** 2
                                + (dt_eff * vRsc) ** 2
                                + (dt_eff * vRcs) ** 2
                                + (dt_eff * vZsc) ** 2
                                + (dt_eff * vZcs) ** 2
                                + (dt_eff * vZcc) ** 2
                                + (dt_eff * vZss) ** 2
                                + (dt_eff * vLsc) ** 2
                                + (dt_eff * vLcs) ** 2
                                + (dt_eff * vLcc) ** 2
                                + (dt_eff * vLss) ** 2
                            )
                        )
                    )
                )

            dR = dt_eff * _mn_cos_to_signed_physical(vRcc, vRss)
            dZ = dt_eff * _mn_sin_to_signed_physical(vZsc, vZcs)
            dL = dt_eff * _mn_sin_to_signed_physical_lambda(vLsc, vLcs)
            dR_sin = dt_eff * _mn_sin_to_signed_physical(vRsc, vRcs)
            dZ_cos = dt_eff * _mn_cos_to_signed_physical(vZcc, vZss)
            dL_cos = dt_eff * _mn_cos_to_signed_physical_lambda(vLcc, vLss)
            if not bool(cfg.lasym):
                dR_sin = jnp.zeros_like(dR_sin)
                dZ_cos = jnp.zeros_like(dZ_cos)
                dL_cos = jnp.zeros_like(dL_cos)
            state_try = VMECState(
                layout=state.layout,
                Rcos=jnp.asarray(state.Rcos) + dR,
                Rsin=jnp.asarray(state.Rsin) + dR_sin,
                Zcos=jnp.asarray(state.Zcos) + dZ_cos,
                Zsin=jnp.asarray(state.Zsin) + dZ,
                Lcos=jnp.asarray(state.Lcos) + dL_cos,
                Lsin=jnp.asarray(state.Lsin) + dL,
            )
            state_try = _enforce_fixed_boundary_and_axis(
                state_try,
                static,
                edge_Rcos=edge_Rcos,
                edge_Rsin=edge_Rsin,
                edge_Zcos=edge_Zcos,
                edge_Zsin=edge_Zsin,
                enforce_lambda_axis=True,
                idx00=idx00,
            )
            state_try = _apply_vmec_lambda_axis_rules(state_try)
            need_trial_eval = bool(backtracking) or bool(reference_mode) or bool(use_direct_fallback)
            probe_bad_jacobian = False
            if need_trial_eval:
                _, _, gcr2_t, gcz2_t, gcl2_t, _, _, norms_t = _compute_forces_iter(
                    state_try,
                    include_edge=include_edge,
                    zero_m1=zero_m1,
                    constraint_precond_diag=constraint_precond_diag,
                    constraint_tcon=constraint_tcon_override,
                    constraint_precond_active=constraint_precond_active,
                    constraint_tcon_active=constraint_tcon_active,
                    iter2=iter2,
                )
                fsqr_t, fsqz_t, fsql_t = _fsq_from_norms(
                    norms_t,
                    gcr2_in=gcr2_t,
                    gcz2_in=gcz2_t,
                    gcl2_in=gcl2_t,
                )
                w_try = float(np.asarray(fsqr_t + fsqz_t + fsql_t))
                w_try_ratio = w_try / max(w_curr, 1e-30) if np.isfinite(w_try) else float("inf")
                if bool(reference_mode) and (float(np.asarray(zero_m1)) > 0.5):
                    _, _, gcr2_probe, gcz2_probe, gcl2_probe, _, _, norms_probe = _compute_forces_iter(
                        state_try,
                        include_edge=include_edge,
                        zero_m1=jnp.asarray(0.0, dtype=zero_m1.dtype),
                        constraint_precond_diag=constraint_precond_diag,
                        constraint_tcon=constraint_tcon_override,
                        constraint_precond_active=constraint_precond_active,
                        constraint_tcon_active=constraint_tcon_active,
                        iter2=iter2,
                    )
                    fsqr_probe, fsqz_probe, fsql_probe = _fsq_from_norms(
                        norms_probe,
                        gcr2_in=gcr2_probe,
                        gcz2_in=gcz2_probe,
                        gcl2_in=gcl2_probe,
                    )
                    w_probe = float(np.asarray(fsqr_probe + fsqz_probe + fsql_probe))
                    if (not np.isfinite(w_probe)) or (w_probe > 1.0e2 * max(w_curr, 1e-30)):
                        probe_bad_jacobian = True
                        w_try = float("inf")
                        w_try_ratio = float("inf")
            else:
                w_try = w_curr
                w_try_ratio = 1.0

            # The reference iteration is typically stable under its restart
            # triggers, but our parity-path preconditioners are still evolving.
            # Add a small,
            # bounded backtracking on the position update (not the force
            # evaluation) to prevent systematic residual growth.
            alpha = 1.0
            accept_ratio = 1.001 if backtracking else float("inf")
            if np.isfinite(w_try) and (w_try > accept_ratio * max(w_curr, 1e-30)):
                for _ in range(8):
                    alpha *= 0.5
                    state_try = VMECState(
                        layout=state.layout,
                        Rcos=jnp.asarray(state.Rcos) + alpha * dR,
                        Rsin=jnp.asarray(state.Rsin) + alpha * dR_sin,
                        Zcos=jnp.asarray(state.Zcos) + alpha * dZ_cos,
                        Zsin=jnp.asarray(state.Zsin) + alpha * dZ,
                        Lcos=jnp.asarray(state.Lcos) + alpha * dL_cos,
                        Lsin=jnp.asarray(state.Lsin) + alpha * dL,
                    )
                    state_try = _enforce_fixed_boundary_and_axis(
                        state_try,
                        static,
                        edge_Rcos=edge_Rcos,
                        edge_Rsin=edge_Rsin,
                        edge_Zcos=edge_Zcos,
                        edge_Zsin=edge_Zsin,
                        enforce_lambda_axis=True,
                        idx00=idx00,
                    )
                    state_try = _apply_vmec_lambda_axis_rules(state_try)
                    _, _, gcr2_t, gcz2_t, gcl2_t, _, _, norms_t = _compute_forces_iter(
                        state_try,
                        include_edge=include_edge,
                        zero_m1=zero_m1,
                        constraint_precond_diag=constraint_precond_diag,
                        constraint_tcon=constraint_tcon_override,
                        constraint_precond_active=constraint_precond_active,
                        constraint_tcon_active=constraint_tcon_active,
                        iter2=iter2,
                    )
                    fsqr_t, fsqz_t, fsql_t = _fsq_from_norms(
                        norms_t,
                        gcr2_in=gcr2_t,
                        gcz2_in=gcz2_t,
                        gcl2_in=gcl2_t,
                    )
                    w_try = float(np.asarray(fsqr_t + fsqz_t + fsql_t))
                    w_try_ratio = w_try / max(w_curr, 1e-30) if np.isfinite(w_try) else float("inf")
                    if np.isfinite(w_try) and (w_try <= accept_ratio * max(w_curr, 1e-30)):
                        # Keep momentum consistent with the smaller step.
                        vRcc = alpha * vRcc
                        vRss = alpha * vRss
                        vZsc = alpha * vZsc
                        vZcs = alpha * vZcs
                        vLsc = alpha * vLsc
                        vLcs = alpha * vLcs
                        update_rms *= alpha
                        dt_eff *= alpha
                        break

            # Require (near) monotone improvement; otherwise fall back to the
            # restart/timestep control path.
            if np.isfinite(w_try) and (w_try <= accept_ratio * max(w_curr, 1e-30)):
                state = state_try
                step_status = "momentum"
                restart_reason = "none"
                huge_force_restart_count = 0
                restart_path = "momentum_accept"
            else:
                if use_direct_fallback:
                    # Try a small direct-force step (no momentum memory) before
                    # a full restart. This is an experimental parity path.
                    dt_direct = max(0.1 * dt_eff, 1e-12)
                    force_rms = float(
                        np.asarray(
                            jnp.sqrt(
                                jnp.mean(
                                    frcc_u * frcc_u
                                    + frss_u * frss_u
                                    + frsc_u * frsc_u
                                    + frcs_u * frcs_u
                                    + fzsc_u * fzsc_u
                                    + fzcs_u * fzcs_u
                                    + fzcc_u * fzcc_u
                                    + fzss_u * fzss_u
                                    + flsc_u * flsc_u
                                    + flcs_u * flcs_u
                                    + flcc_u * flcc_u
                                    + flss_u * flss_u
                                )
                            )
                        )
                    )
                    if np.isfinite(force_rms) and force_rms > 0.0:
                        dt_cap = max_update_rms / max(force_rms, 1e-30)
                        dt_direct = max(min(dt_direct, float(dt_cap)), 1e-12)
                    dR_dir = dt_direct * _mn_cos_to_signed(flip_sign * frcc_u, flip_sign * frss_u)
                    dZ_dir = dt_direct * _mn_sin_to_signed(flip_sign * fzsc_u, flip_sign * fzcs_u)
                    dL_dir = dt_direct * _mn_sin_to_signed(flip_sign * flsc_u, flip_sign * flcs_u)
                    dR_sin_dir = dt_direct * _mn_sin_to_signed(flip_sign * frsc_u, flip_sign * frcs_u)
                    dZ_cos_dir = dt_direct * _mn_cos_to_signed(flip_sign * fzcc_u, flip_sign * fzss_u)
                    dL_cos_dir = dt_direct * _mn_cos_to_signed(flip_sign * flcc_u, flip_sign * flss_u)
                    if not bool(cfg.lasym):
                        dR_sin_dir = jnp.zeros_like(dR_sin_dir)
                        dZ_cos_dir = jnp.zeros_like(dZ_cos_dir)
                        dL_cos_dir = jnp.zeros_like(dL_cos_dir)
                    state_dir = VMECState(
                        layout=state.layout,
                        Rcos=jnp.asarray(state.Rcos) + dR_dir,
                        Rsin=jnp.asarray(state.Rsin) + dR_sin_dir,
                        Zcos=jnp.asarray(state.Zcos) + dZ_cos_dir,
                        Zsin=jnp.asarray(state.Zsin) + dZ_dir,
                        Lcos=jnp.asarray(state.Lcos) + dL_cos_dir,
                        Lsin=jnp.asarray(state.Lsin) + dL_dir,
                    )
                    state_dir = _enforce_fixed_boundary_and_axis(
                        state_dir,
                        static,
                        edge_Rcos=edge_Rcos,
                        edge_Rsin=edge_Rsin,
                        edge_Zcos=edge_Zcos,
                        edge_Zsin=edge_Zsin,
                        enforce_lambda_axis=True,
                        idx00=idx00,
                    )
                    state_dir = _apply_vmec_lambda_axis_rules(state_dir)
                    _, _, gcr2_d, gcz2_d, gcl2_d, _, _, norms_d = _compute_forces_iter(
                        state_dir,
                        include_edge=include_edge,
                        zero_m1=zero_m1,
                        constraint_precond_diag=constraint_precond_diag,
                        constraint_tcon=constraint_tcon_override,
                        constraint_precond_active=constraint_precond_active,
                        constraint_tcon_active=constraint_tcon_active,
                        iter2=iter2,
                    )
                    fsqr_d, fsqz_d, fsql_d = _fsq_from_norms(
                        norms_d,
                        gcr2_in=gcr2_d,
                        gcz2_in=gcz2_d,
                        gcl2_in=gcl2_d,
                    )
                    w_dir = float(np.asarray(fsqr_d + fsqz_d + fsql_d))
                    if np.isfinite(w_dir) and (w_dir <= 1.5 * max(w_curr, 1e-30)):
                        state = state_dir
                        vRcc = jnp.zeros_like(vRcc)
                        vRss = jnp.zeros_like(vRss)
                        vZsc = jnp.zeros_like(vZsc)
                        vZcs = jnp.zeros_like(vZcs)
                        vLsc = jnp.zeros_like(vLsc)
                        vLcs = jnp.zeros_like(vLcs)
                        vRsc = jnp.zeros_like(vRsc)
                        vRcs = jnp.zeros_like(vRcs)
                        vZcc = jnp.zeros_like(vZcc)
                        vZss = jnp.zeros_like(vZss)
                        vLcc = jnp.zeros_like(vLcc)
                        vLss = jnp.zeros_like(vLss)
                        step_status = "fallback_direct"
                        restart_reason = "none"
                        huge_force_restart_count = 0
                        restart_path = "fallback_direct"
                        update_rms = float(
                            np.asarray(
                                jnp.sqrt(
                                    jnp.mean(
                                        (dt_direct * frcc_u) ** 2
                                        + (dt_direct * frss_u) ** 2
                                        + (dt_direct * frsc_u) ** 2
                                        + (dt_direct * frcs_u) ** 2
                                        + (dt_direct * fzsc_u) ** 2
                                        + (dt_direct * fzcs_u) ** 2
                                        + (dt_direct * fzcc_u) ** 2
                                        + (dt_direct * fzss_u) ** 2
                                        + (dt_direct * flsc_u) ** 2
                                        + (dt_direct * flcs_u) ** 2
                                        + (dt_direct * flcc_u) ** 2
                                        + (dt_direct * flss_u) ** 2
                                    )
                                )
                            )
                        )
                    else:
                        # Roll back state and zero velocity.
                        state = state_backup
                        vRcc = jnp.zeros_like(vRcc)
                        vRss = jnp.zeros_like(vRss)
                        vZsc = jnp.zeros_like(vZsc)
                        vZcs = jnp.zeros_like(vZcs)
                        vLsc = jnp.zeros_like(vLsc)
                        vLcs = jnp.zeros_like(vLcs)
                        # Tighten displacement caps when restarting from
                        # catastrophic growth; otherwise dt_eff can remain
                        # stuck at the same limit.
                        max_coeff_delta_rms = max(0.5 * max_coeff_delta_rms, 1e-12)
                        max_update_rms = max(0.8 * max_update_rms, 1e-6)
                        if bool(probe_bad_jacobian) or (not np.isfinite(w_try)):
                            time_step = max(restart_badjac_factor * time_step, 1e-12)
                            ijacob += 1
                            restart_reason = "bad_jacobian"
                            step_status = "restart_bad_jacobian"
                            restart_path = "catastrophic_nonfinite"
                        else:
                            time_step = max(time_step / restart_badprog_factor, 1e-12)
                            restart_reason = "bad_progress"
                            step_status = "restart_bad_progress"
                            restart_path = "catastrophic_growth"
                        # Adjust time_step at reset milestones.
                        if ijacob in (25, 50):
                            scale = 0.98 if ijacob < 50 else 0.96
                            time_step = max(scale * float(step_size), 1e-12)
                        bad_resets += 1
                        iter1 = iter2
                        fsq_prev = fsq_prev_before
                        fsq0_prev = fsq0_prev_before
                        inv_tau = [0.15 / time_step] * k_ndamp
                        update_rms = 0.0
                        if bool(vmec2000_control):
                            vmec2000_cache_valid = False
                            cache_precond_diag = None
                            cache_tcon = None
                            cache_norms = None
                            cache_rz_scale = None
                            cache_l_scale = None
                            cache_rz_norm = None
                            cache_f_norm1 = None
                            cache_prec_rz_mats = None
                            cache_prec_rz_jmax = None
                            cache_prec_lam_prec = None
                            cache_prec_faclam = None
                            cache_prec_lam_debug = None
                else:
                    # Roll back state and zero velocity.
                    state = state_backup
                    vRcc = jnp.zeros_like(vRcc)
                    vRss = jnp.zeros_like(vRss)
                    vZsc = jnp.zeros_like(vZsc)
                    vZcs = jnp.zeros_like(vZcs)
                    vLsc = jnp.zeros_like(vLsc)
                    vLcs = jnp.zeros_like(vLcs)
                    # Tighten displacement caps when restarting from catastrophic
                    # growth; otherwise dt_eff can remain stuck at the same limit.
                    max_coeff_delta_rms = max(0.5 * max_coeff_delta_rms, 1e-12)
                    max_update_rms = max(0.8 * max_update_rms, 1e-6)
                    if bool(probe_bad_jacobian) or (not np.isfinite(w_try)):
                        time_step = max(restart_badjac_factor * time_step, 1e-12)
                        ijacob += 1
                        restart_reason = "bad_jacobian"
                        step_status = "restart_bad_jacobian"
                        restart_path = "catastrophic_nonfinite"
                    else:
                        time_step = max(time_step / restart_badprog_factor, 1e-12)
                        restart_reason = "bad_progress"
                        step_status = "restart_bad_progress"
                        restart_path = "catastrophic_growth"
                    # Adjust time_step at reset milestones.
                    if ijacob in (25, 50):
                        scale = 0.98 if ijacob < 50 else 0.96
                        time_step = max(scale * float(step_size), 1e-12)
                    bad_resets += 1
                    iter1 = iter2
                    fsq_prev = fsq_prev_before
                    fsq0_prev = fsq0_prev_before
                    inv_tau = [0.15 / time_step] * k_ndamp
                    update_rms = 0.0
                    if not bool(vmec2000_control):
                        vmec2000_cache_valid = False
                        cache_precond_diag = None
                        cache_tcon = None
                        cache_norms = None
                        cache_rz_scale = None
                        cache_l_scale = None
                        cache_rz_norm = None
                        cache_f_norm1 = None
                        cache_prec_rz_mats = None
                        cache_prec_rz_jmax = None
                        cache_prec_lam_prec = None
                        cache_prec_faclam = None
                        cache_prec_lam_debug = None
            if timing_enabled and t_update_start is not None:
                try:
                    if has_jax():
                        jax.block_until_ready(state.Rcos)
                except Exception:
                    pass
                timing_stats["update"] += time.perf_counter() - float(t_update_start)
            timing_stats["iterations"] += 1
            if track_history:
                step_history.append(float(dt_eff))
                w_curr_history.append(float(w_curr))
                w_try_history.append(float(w_try))
                w_try_ratio_history.append(float(w_try_ratio))
                restart_path_history.append(str(restart_path))
        else:
            accepted = False
            step_status = "rejected"
            step_factor = 1.0
            vRcc_best, vRss_best = vRcc, vRss
            vZsc_best, vZcs_best = vZsc, vZcs
            vLsc_best, vLcs_best = vLsc, vLcs
            vRsc_best, vRcs_best = vRsc, vRcs
            vZcc_best, vZss_best = vZcc, vZss
            vLcc_best, vLss_best = vLcc, vLss
            state_best = state
            dt_eff = float(time_step)
            update_rms = 0.0
            w_curr = fsqr_f + fsqz_f + fsql_f

            for _bt in range(6):
                dt_try = time_step * step_factor
                vRcc_try = fac * (b1 * vRcc + dt_try * (flip_sign * jnp.asarray(frcc_u)))
                vRss_try = fac * (b1 * vRss + dt_try * (flip_sign * jnp.asarray(frss_u)))
                vRsc_try = fac * (b1 * vRsc + dt_try * (flip_sign * jnp.asarray(frsc_u)))
                vRcs_try = fac * (b1 * vRcs + dt_try * (flip_sign * jnp.asarray(frcs_u)))
                vZsc_try = fac * (b1 * vZsc + dt_try * (flip_sign * jnp.asarray(fzsc_u)))
                vZcs_try = fac * (b1 * vZcs + dt_try * (flip_sign * jnp.asarray(fzcs_u)))
                vZcc_try = fac * (b1 * vZcc + dt_try * (flip_sign * jnp.asarray(fzcc_u)))
                vZss_try = fac * (b1 * vZss + dt_try * (flip_sign * jnp.asarray(fzss_u)))
                vLsc_try = fac * (b1 * vLsc + dt_try * (flip_sign * jnp.asarray(flsc_u)))
                vLcs_try = fac * (b1 * vLcs + dt_try * (flip_sign * jnp.asarray(flcs_u)))
                vLcc_try = fac * (b1 * vLcc + dt_try * (flip_sign * jnp.asarray(flcc_u)))
                vLss_try = fac * (b1 * vLss + dt_try * (flip_sign * jnp.asarray(flss_u)))

                dR_try = dt_try * _mn_cos_to_signed(vRcc_try, vRss_try)
                dZ_try = dt_try * _mn_sin_to_signed(vZsc_try, vZcs_try)
                dL_try = dt_try * _mn_sin_to_signed(vLsc_try, vLcs_try)
                dR_sin_try = dt_try * _mn_sin_to_signed(vRsc_try, vRcs_try)
                dZ_cos_try = dt_try * _mn_cos_to_signed(vZcc_try, vZss_try)
                dL_cos_try = dt_try * _mn_cos_to_signed(vLcc_try, vLss_try)
                if not bool(cfg.lasym):
                    dR_sin_try = jnp.zeros_like(dR_sin_try)
                    dZ_cos_try = jnp.zeros_like(dZ_cos_try)
                    dL_cos_try = jnp.zeros_like(dL_cos_try)

                state_try = VMECState(
                    layout=state.layout,
                    Rcos=jnp.asarray(state.Rcos) + dR_try,
                    Rsin=jnp.asarray(state.Rsin) + dR_sin_try,
                    Zcos=jnp.asarray(state.Zcos) + dZ_cos_try,
                    Zsin=jnp.asarray(state.Zsin) + dZ_try,
                    Lcos=jnp.asarray(state.Lcos) + dL_cos_try,
                    Lsin=jnp.asarray(state.Lsin) + dL_try,
                )
                state_try = _enforce_fixed_boundary_and_axis(
                    state_try,
                    static,
                    edge_Rcos=edge_Rcos,
                    edge_Rsin=edge_Rsin,
                    edge_Zcos=edge_Zcos,
                    edge_Zsin=edge_Zsin,
                    enforce_lambda_axis=True,
                    idx00=idx00,
                )
                state_try = _apply_vmec_lambda_axis_rules(state_try)
                _, _, gcr2_t, gcz2_t, gcl2_t, _, _, norms_t = _compute_forces_iter(
                    state_try,
                    include_edge=include_edge,
                    zero_m1=zero_m1,
                    constraint_precond_diag=constraint_precond_diag,
                    constraint_tcon=constraint_tcon_override,
                    constraint_precond_active=constraint_precond_active,
                    constraint_tcon_active=constraint_tcon_active,
                    iter2=iter2,
                )
                fsqr_t, fsqz_t, fsql_t = _fsq_from_norms(
                    norms_t,
                    gcr2_in=gcr2_t,
                    gcz2_in=gcz2_t,
                    gcl2_in=gcl2_t,
                )
                w_try = float(np.asarray(fsqr_t + fsqz_t + fsql_t))
                if np.isfinite(w_try) and (w_try <= 1.05 * w_curr):
                    accepted = True
                    step_status = "momentum"
                    state_best = state_try
                    vRcc_best, vRss_best = vRcc_try, vRss_try
                    vZsc_best, vZcs_best = vZsc_try, vZcs_try
                    vLsc_best, vLcs_best = vLsc_try, vLcs_try
                    vRsc_best, vRcs_best = vRsc_try, vRcs_try
                    vZcc_best, vZss_best = vZcc_try, vZss_try
                    vLcc_best, vLss_best = vLcc_try, vLss_try
                    dt_eff = float(dt_try)
                    update_rms = float(
                        np.asarray(
                            jnp.sqrt(
                                jnp.mean(
                                    (dt_try * vRcc_try) ** 2
                                    + (dt_try * vRss_try) ** 2
                                    + (dt_try * vRsc_try) ** 2
                                    + (dt_try * vRcs_try) ** 2
                                    + (dt_try * vZsc_try) ** 2
                                    + (dt_try * vZcs_try) ** 2
                                    + (dt_try * vZcc_try) ** 2
                                    + (dt_try * vZss_try) ** 2
                                    + (dt_try * vLsc_try) ** 2
                                    + (dt_try * vLcs_try) ** 2
                                    + (dt_try * vLcc_try) ** 2
                                    + (dt_try * vLss_try) ** 2
                                )
                            )
                        )
                    )
                    break
                step_factor *= 0.5

            state = state_best
            vRcc, vRss = vRcc_best, vRss_best
            vZsc, vZcs = vZsc_best, vZcs_best
            vLsc, vLcs = vLsc_best, vLcs_best
            vRsc, vRcs = vRsc_best, vRcs_best
            vZcc, vZss = vZcc_best, vZss_best
            vLcc, vLss = vLcc_best, vLss_best
            if not accepted:
                # No acceptable update was found; damp velocity to avoid runaway.
                vRcc = 0.5 * vRcc
                vRss = 0.5 * vRss
                vRsc = 0.5 * vRsc
                vRcs = 0.5 * vRcs
                vZsc = 0.5 * vZsc
                vZcs = 0.5 * vZcs
                vZcc = 0.5 * vZcc
                vZss = 0.5 * vZss
                vLsc = 0.5 * vLsc
                vLcs = 0.5 * vLcs
                vLcc = 0.5 * vLcc
                vLss = 0.5 * vLss
                dt_eff = float(step_size * step_factor)
                update_rms = 0.0
                step_status = "rejected"
            timing_stats["iterations"] += 1
            if track_history:
                step_history.append(dt_eff)
                restart_reason = "none"
                w_curr_history.append(float(w_curr))
                w_try_history.append(float("nan"))
                w_try_ratio_history.append(float("nan"))
                restart_path_history.append("non_strict")
        _maybe_dump_xc(
            state=state,
            vRcc=vRcc,
            vRss=vRss,
            vZsc=vZsc,
            vZcs=vZcs,
            vLsc=vLsc,
            vLcs=vLcs,
            static=static,
            iter_idx=int(iter2),
        )
        if track_history:
            dt_eff_history.append(float(dt_eff))
            update_rms_history.append(float(update_rms))
        if verbose:
            if bool(vmec2000_control) and bool(verbose_vmec2000_table):
                if _should_print_vmec2000(int(iter2), int(max_iter)):
                    _print_vmec2000_iter_row(
                        iter_idx=int(iter2),
                        fsqr=fsqr_f,
                        fsqz=fsqz_f,
                        fsql=fsql_f,
                        fsqr1=fsqr1_f,
                        fsqz1=fsqz1_f,
                        fsql1=fsql1_f,
                        delt0r=float(time_step),
                        r00=float(r00_last),
                        w_mhd=float(w_vmec_last),
                        z00=float(z00_last),
                    )
            else:
                print(
                    f"[solve_fixed_boundary_residual_iter] iter={it:03d} "
                    f"dt_eff={dt_eff:.3e} update_rms={update_rms:.3e} "
                    f"fsqr1={fsqr1_f:.3e} fsqz1={fsqz1_f:.3e} fsql1={fsql1_f:.3e} "
                    f"step_status={step_status}",
                    flush=True,
                )
        if track_history:
            step_status_history.append(step_status)
            restart_reason_history.append(restart_reason)
            pre_restart_reason_history.append(pre_restart_reason)
            time_step_history.append(float(time_step))
            res0_history.append(float(res0))
            res1_history.append(float(res1))
            fsq_prev_history.append(float(fsq_prev))
            bad_growth_streak_history.append(int(bad_growth_streak))
            iter1_history.append(int(iter1))
            grad_rms_history.append(float(np.sqrt(max(fsqr_f + fsqz_f + fsql_f, 0.0))))
        skip_time_control = False

    diag: Dict[str, Any] = {
        "ftol": ftol,
        "gamma": gamma,
        "step_size": float(step_size),
        "precond_radial_alpha": float(precond_radial_alpha),
        "precond_lambda_alpha": float(precond_lambda_alpha),
        "strict_update": bool(strict_update),
        "reference_mode": bool(reference_mode),
        "use_restart_triggers": bool(use_restart_triggers),
        "use_direct_fallback": bool(use_direct_fallback),
        "max_update_rms": float(max_update_rms),
        "converged": bool(converged),
        "badjac_use_state": bool(badjac_use_state),
        "badjac_mode": badjac_mode,
        "light_history": bool(light_history),
        "ijacob": int(ijacob),
        "bad_resets": int(bad_resets),
        "iter1_final": int(iter1),
        "res0": float(res0),
        "step_status_history": np.asarray(step_status_history, dtype=object),
        "restart_reason_history": np.asarray(restart_reason_history, dtype=object),
        "pre_restart_reason_history": np.asarray(pre_restart_reason_history, dtype=object),
        "time_step_history": np.asarray(time_step_history, dtype=float),
        "res0_history": np.asarray(res0_history, dtype=float),
        "res1_history": np.asarray(res1_history, dtype=float),
        "fsq_prev_history": np.asarray(fsq_prev_history, dtype=float),
        "bad_growth_streak_history": np.asarray(bad_growth_streak_history, dtype=int),
        "iter1_history": np.asarray(iter1_history, dtype=int),
        "bcovar_update_history": np.asarray(bcovar_update_history, dtype=int),
        "include_edge_history": np.asarray(include_edge_history, dtype=int),
        "zero_m1_history": np.asarray(zero_m1_history, dtype=int),
        "dt_eff_history": np.asarray(dt_eff_history, dtype=float),
        "update_rms_history": np.asarray(update_rms_history, dtype=float),
        "w_curr_history": np.asarray(w_curr_history, dtype=float),
        "w_try_history": np.asarray(w_try_history, dtype=float),
        "w_try_ratio_history": np.asarray(w_try_ratio_history, dtype=float),
        "restart_path_history": np.asarray(restart_path_history, dtype=object),
        "min_tau_history": np.asarray(min_tau_history, dtype=float),
        "max_tau_history": np.asarray(max_tau_history, dtype=float),
        "bad_jacobian_history": np.asarray(bad_jacobian_history, dtype=int),
        "r00_history": np.asarray(r00_history, dtype=float),
        "z00_history": np.asarray(z00_history, dtype=float),
        "wb_history": np.asarray(wb_history, dtype=float),
        "wp_history": np.asarray(wp_history, dtype=float),
        "w_vmec_history": np.asarray(w_vmec_history, dtype=float),
        "fsq1_history": np.asarray(fsq1_history, dtype=float),
        "fsqr1_history": np.asarray(fsqr1_history, dtype=float),
        "fsqz1_history": np.asarray(fsqz1_history, dtype=float),
        "fsql1_history": np.asarray(fsql1_history, dtype=float),
        "rz_norm_history": np.asarray(rz_norm_history, dtype=float),
        "f_norm1_history": np.asarray(f_norm1_history, dtype=float),
        "gcr2_p_history": np.asarray(gcr2_p_history, dtype=float),
        "gcz2_p_history": np.asarray(gcz2_p_history, dtype=float),
        "gcl2_p_history": np.asarray(gcl2_p_history, dtype=float),
    }
    if timing_enabled:
        iters = max(int(timing_stats["iterations"]), 1)
        timing_report = {
            "iterations": int(timing_stats["iterations"]),
            "compute_forces_s": float(timing_stats["compute_forces"]),
            "preconditioner_s": float(timing_stats["preconditioner"]),
            "precond_refresh_s": float(timing_stats["precond_refresh"]),
            "update_s": float(timing_stats["update"]),
            "compute_forces_per_iter_s": float(timing_stats["compute_forces"]) / iters,
            "preconditioner_per_iter_s": float(timing_stats["preconditioner"]) / iters,
            "update_per_iter_s": float(timing_stats["update"]) / iters,
        }
        diag["timing"] = timing_report
        try:
            print(
                "[vmec_jax timing] "
                f"iters={timing_report['iterations']} "
                f"compute_forces={timing_report['compute_forces_s']:.3e}s "
                f"precond={timing_report['preconditioner_s']:.3e}s "
                f"precond_refresh={timing_report['precond_refresh_s']:.3e}s "
                f"update={timing_report['update_s']:.3e}s "
                f"(per-iter: {timing_report['compute_forces_per_iter_s']:.3e}, "
                f"{timing_report['preconditioner_per_iter_s']:.3e}, "
                f"{timing_report['update_per_iter_s']:.3e})",
                flush=True,
            )
        except Exception:
            pass
    diag["resume_state"] = {
        "time_step": float(time_step),
        "inv_tau": list(inv_tau),
        "fsq_prev": float(fsq_prev),
        "fsq0_prev": float(fsq0_prev),
        "flip_sign": float(flip_sign),
        "iter1": int(iter1),
        "iter_offset": int(last_iter2),
        "ijacob": int(ijacob),
        "bad_resets": int(bad_resets),
        "res0": float(res0),
        "res1": float(res1),
        "prev_rz_fsq": float(prev_rz_fsq),
        "bad_growth_streak": int(bad_growth_streak),
        "huge_force_restart_count": int(huge_force_restart_count),
        "vRcc": np.asarray(vRcc),
        "vRss": np.asarray(vRss),
        "vZsc": np.asarray(vZsc),
        "vZcs": np.asarray(vZcs),
        "vLsc": np.asarray(vLsc),
        "vLcs": np.asarray(vLcs),
        "vRsc": np.asarray(vRsc),
        "vRcs": np.asarray(vRcs),
        "vZcc": np.asarray(vZcc),
        "vZss": np.asarray(vZss),
        "vLcc": np.asarray(vLcc),
        "vLss": np.asarray(vLss),
        "state_checkpoint": state_checkpoint,
        "vmec2000_cache_valid": bool(vmec2000_cache_valid),
        "cache_precond_diag": cache_precond_diag,
        "cache_tcon": cache_tcon,
        "cache_norms": cache_norms,
        "cache_rz_scale": cache_rz_scale,
        "cache_l_scale": cache_l_scale,
        "cache_rz_norm": cache_rz_norm,
        "cache_f_norm1": cache_f_norm1,
        "cache_prec_rz_mats": cache_prec_rz_mats,
        "cache_prec_rz_jmax": cache_prec_rz_jmax,
        "cache_prec_lam_prec": cache_prec_lam_prec,
        "cache_prec_faclam": cache_prec_faclam,
        "cache_prec_lam_debug": cache_prec_lam_debug,
    }
    return SolveVmecResidualResult(
        state=state,
        n_iter=len(w_history) - 1,
        w_history=np.asarray(w_history, dtype=float),
        fsqr2_history=np.asarray(fsqr2_history, dtype=float),
        fsqz2_history=np.asarray(fsqz2_history, dtype=float),
        fsql2_history=np.asarray(fsql2_history, dtype=float),
        grad_rms_history=np.asarray(grad_rms_history, dtype=float),
        step_history=np.asarray(step_history, dtype=float),
        diagnostics=diag,
    )


def first_step_diagnostics(
    state0: VMECState,
    static,
    *,
    indata,
    signgs: int,
    step_size: float | None = None,
    include_constraint_force: bool = True,
    apply_m1_constraints: bool = True,
    precond_radial_alpha: float = 0.5,
    precond_lambda_alpha: float = 0.5,
    mode_diag_exponent: float = 1.0,
    include_edge: bool = True,
    zero_m1: bool = True,
    use_axisymmetric_preconditioner: bool = False,
) -> Dict[str, Any]:
    """Return a first-step diagnostic bundle (single force/precondition/update eval).

    This computes the initial forces, preconditioned residuals, time-step
    scalings, and the resulting first-step coefficient updates without
    running an iterative solve.
    """
    if not has_jax():
        raise ImportError("first_step_diagnostics requires JAX (jax + jaxlib)")

    from .energy import flux_profiles_from_indata
    from .static import build_static
    from .vmec_forces import vmec_forces_rz_from_wout, vmec_residual_internal_from_kernels
    from .vmec_residue import (
        vmec_apply_m1_constraints,
        vmec_apply_scalxc_to_tomnsps,
        vmec_force_norms_from_bcovar_dynamic,
        vmec_gcx2_from_tomnsps,
        vmec_rz_norm_from_state,
        vmec_scalxc_from_s,
        vmec_wint_from_trig,
        vmec_zero_m1_zforce,
    )
    from .vmec_tomnsp import TomnspsRZL, vmec_angle_grid, vmec_trig_tables

    signgs = int(signgs)
    cfg = static.cfg
    grid_vmec = vmec_angle_grid(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(cfg.nfp),
        lasym=bool(cfg.lasym),
    )
    static_vmec = build_static(cfg, grid=grid_vmec)
    s = jnp.asarray(static_vmec.s)
    flux = flux_profiles_from_indata(indata, s, signgs=signgs)
    chipf_wout = jnp.asarray(flux.chipf)
    phips = jnp.asarray(flux.phips)
    if phips.shape[0] >= 1:
        phips = phips.at[0].set(0.0)
    from .boundary import boundary_from_indata

    boundary = boundary_from_indata(indata, static_vmec.modes)
    idx00 = _mode00_index(static_vmec.modes)
    r00 = float(np.asarray(boundary.R_cos)[int(idx00)]) if int(idx00) >= 0 else float(np.asarray(boundary.R_cos)[0])
    gamma = float(indata.get_float("GAMMA", 5.0 / 3.0))
    lrfp = bool(indata.get_bool("LRFP", False))
    chips = _half_mesh_from_full_mesh(chipf_wout) if lrfp else None
    mass = _mass_half_mesh_from_indata(
        indata=indata,
        s_full=s,
        phips=phips,
        r00=r00,
        gamma=gamma,
        lrfp=lrfp,
        chips=chips,
    )
    pres = _pressure_half_mesh_from_indata(indata=indata, s_full=s)
    ncurr = int(indata.get_int("NCURR", 0))
    icurv = _icurv_full_mesh_from_indata(indata=indata, s_full=s, signgs=signgs)

    wout_like = _WoutLikeVmecForces(
        nfp=int(cfg.nfp),
        mpol=int(cfg.mpol),
        ntor=int(cfg.ntor),
        lasym=bool(cfg.lasym),
        signgs=signgs,
        phipf=jnp.asarray(flux.phipf),
        phips=phips,
        chipf=chipf_wout,
        pres=pres,
        mass=mass,
        gamma=gamma,
        ncurr=ncurr,
        lcurrent=True,
        icurv=icurv,
    )

    trig = getattr(static_vmec, "trig_vmec", None)
    if trig is None:
        trig = vmec_trig_tables(
            ntheta=int(cfg.ntheta),
            nzeta=int(cfg.nzeta),
            nfp=int(wout_like.nfp),
            mmax=int(wout_like.mpol) - 1,
            nmax=int(wout_like.ntor),
            lasym=bool(wout_like.lasym),
            dtype=jnp.asarray(state0.Rcos).dtype,
        )
    if not bool(wout_like.lasym):
        # For lasym=False keep Z-force intact in the first-step diagnostic.
        zero_m1 = False

    constraint_tcon0: float | None = None
    if bool(include_constraint_force):
        constraint_tcon0 = float(indata.get_float("TCON0", 1.0))

    def _zero_edge_rz(a):
        a = None if a is None else jnp.asarray(a)
        if a is None or a.shape[0] < 2:
            return a
        return a.at[-1].set(jnp.zeros_like(a[-1]))

    def _apply_radial_tridi(rhs, alpha: float):
        if alpha <= 0.0:
            return rhs
        rhs = jnp.asarray(rhs)
        if rhs.ndim == 2:
            rhs2 = rhs
            orig_shape = None
        elif rhs.ndim == 3:
            ns = int(rhs.shape[0])
            rhs2 = rhs.reshape(ns, -1)
            orig_shape = rhs.shape
        else:
            raise ValueError(f"expected (ns,K) or (ns,M,N), got {rhs.shape}")
        ns = int(rhs2.shape[0])
        if ns < 3:
            return rhs
        alpha = jnp.asarray(alpha, dtype=rhs2.dtype)
        a = -alpha
        b = 1.0 + 2.0 * alpha
        c = -alpha
        x0 = rhs2[0]
        xN = rhs2[-1]
        d = rhs2[1:-1]
        d = d.at[0].add(alpha * x0)
        d = d.at[-1].add(alpha * xN)
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

            (cp_last, dp_last), (cp, dp) = jax.lax.scan(fwd, (cp0, dp0), d[1:])

            def bwd(carry, cp_dp):
                x_next = carry
                cp_i, dp_i = cp_dp
                x_i = dp_i - cp_i * x_next
                return x_i, x_i

            _, x_rev = jax.lax.scan(bwd, dp_last, (cp, dp), reverse=True)
            x_int = jnp.concatenate([x_rev, dp_last[None, :]], axis=0)
        out = jnp.concatenate([x0[None, :], x_int, xN[None, :]], axis=0)
        if orig_shape is not None:
            out = out.reshape(orig_shape)
        return out

    def _metric_surface_precond_from_bcovar(bc):
        guu = jnp.asarray(bc.guu)
        r12 = jnp.asarray(bc.jac.r12)
        bsubu = jnp.asarray(bc.bsubu)
        bsubv = jnp.asarray(bc.bsubv)
        nzeta = int(guu.shape[2])
        w_ang = vmec_wint_from_trig(trig, nzeta=nzeta).astype(guu.dtype)
        w3 = w_ang[None, :, :]
        rz_denom = jnp.sum((guu * (r12 * r12)) * w3, axis=(1, 2))
        rz_scale = jnp.where(rz_denom > 0.0, 1.0 / jnp.sqrt(rz_denom), 1.0)
        l_denom = jnp.sum(((bsubu * bsubu) + (bsubv * bsubv)) * w3, axis=(1, 2))
        l_scale = jnp.where(l_denom > 0.0, 1.0 / jnp.sqrt(l_denom), 1.0)
        rz_scale = jnp.clip(rz_scale, 1e-4, 1e2)
        l_scale = jnp.clip(l_scale, 1e-4, 1e2)
        return rz_scale, l_scale

    def _pshalf_from_s(s_arr):
        s_arr = np.asarray(s_arr, dtype=float)
        if s_arr.size < 2:
            return np.sqrt(np.maximum(s_arr, 0.0))
        sh = 0.5 * (s_arr[1:] + s_arr[:-1])
        p = np.concatenate([sh[:1], sh], axis=0)
        return np.sqrt(np.maximum(p, 0.0))

    def _sm_sp_from_s(s_arr):
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

    def _lambda_preconditioner(bc, *, return_faclam: bool = False):
        from .preconditioner_1d_jax import lambda_preconditioner

        return lambda_preconditioner(
            bc=bc,
            trig=trig,
            s=s,
            cfg=cfg,
            return_faclam=return_faclam,
        )

    def _rz_preconditioner(frzl_in: TomnspsRZL, bc, k):
        from .preconditioner_1d_jax import rz_preconditioner

        return rz_preconditioner(
            frzl_in=frzl_in,
            bc=bc,
            k=k,
            trig=trig,
            s=s,
            cfg=cfg,
        )

    mask_pack = getattr(static_vmec, "tomnsps_masks", None)

    def _compute_forces(state: VMECState):
        k = vmec_forces_rz_from_wout(
            state=state,
            static=static_vmec,
            wout=wout_like,
            indata=None,
            constraint_tcon0=constraint_tcon0,
            use_vmec_synthesis=True,
            trig=trig,
        )
        frzl = vmec_residual_internal_from_kernels(
            k,
            cfg_ntheta=int(cfg.ntheta),
            cfg_nzeta=int(cfg.nzeta),
            wout=wout_like,
            trig=trig,
            apply_lforbal=apply_lforbal,
            include_edge=False,
            masks=mask_pack,
        )
        frzl = vmec_apply_scalxc_to_tomnsps(frzl=frzl, s=s)
        frzl_raw = frzl
        if bool(apply_m1_constraints):
            frzl = vmec_apply_m1_constraints(
                frzl=frzl,
                lconm1=bool(getattr(cfg, "lconm1", True)),
            )
        frzl = vmec_zero_m1_zforce(frzl=frzl, enabled=jnp.asarray(float(bool(zero_m1))))
        frzl = TomnspsRZL(
            frcc=_zero_edge_rz(frzl.frcc),
            frss=_zero_edge_rz(frzl.frss),
            fzsc=_zero_edge_rz(frzl.fzsc),
            fzcs=_zero_edge_rz(frzl.fzcs),
            flsc=frzl.flsc,
            flcs=frzl.flcs,
            frsc=_zero_edge_rz(getattr(frzl, "frsc", None)),
            frcs=_zero_edge_rz(getattr(frzl, "frcs", None)),
            fzcc=_zero_edge_rz(getattr(frzl, "fzcc", None)),
            fzss=_zero_edge_rz(getattr(frzl, "fzss", None)),
            flcc=getattr(frzl, "flcc", None),
            flss=getattr(frzl, "flss", None),
        )
        gcr2, gcz2, gcl2 = vmec_gcx2_from_tomnsps(
            frzl=frzl,
            lconm1=bool(getattr(cfg, "lconm1", True)),
            apply_m1_constraints=False,
            include_edge=bool(include_edge),
            apply_scalxc=False,
            s=s,
        )
        gcr2_raw, gcz2_raw, gcl2_raw = vmec_gcx2_from_tomnsps(
            frzl=frzl_raw,
            lconm1=bool(getattr(cfg, "lconm1", True)),
            apply_m1_constraints=False,
            include_edge=bool(include_edge),
            apply_scalxc=False,
            s=s,
        )
        norms = vmec_force_norms_from_bcovar_dynamic(bc=k.bc, trig=trig, s=s, signgs=signgs)
        fsqr = norms.r1 * norms.fnorm * gcr2
        fsqz = norms.r1 * norms.fnorm * gcz2
        fsql = norms.fnormL * gcl2
        rz_scale, l_scale = _metric_surface_precond_from_bcovar(k.bc)
        return k, frzl, fsqr, fsqz, fsql, rz_scale, l_scale, (gcr2_raw, gcz2_raw, gcl2_raw)

    k, frzl, fsqr, fsqz, fsql, rz_scale, l_scale, g_raw = _compute_forces(state0)
    gcr2_raw, gcz2_raw, gcl2_raw = g_raw

    if bool(use_axisymmetric_preconditioner) and (not bool(cfg.lthreed)) and (not bool(cfg.lasym)):
        lam_prec = _lambda_preconditioner(k.bc)
        frzl_pre = _rz_preconditioner(frzl, k.bc, k)
        frcc = jnp.asarray(frzl_pre.frcc)
        frss = frzl_pre.frss
        fzsc = jnp.asarray(frzl_pre.fzsc)
        fzcs = frzl_pre.fzcs
        flsc = jnp.asarray(frzl_pre.flsc) * jnp.asarray(lam_prec)
        flcs = frzl_pre.flcs
        if not (jnp.all(jnp.isfinite(frcc)) and jnp.all(jnp.isfinite(fzsc)) and jnp.all(jnp.isfinite(flsc))):
            frcc = jnp.asarray(frzl.frcc)
            frss = frzl.frss
            fzsc = jnp.asarray(frzl.fzsc)
            fzcs = frzl.fzcs
            flsc = jnp.asarray(frzl.flsc)
            flcs = frzl.flcs
    else:
        frcc = _apply_radial_tridi(frzl.frcc * rz_scale[:, None, None], precond_radial_alpha)
        frss = _apply_radial_tridi(frzl.frss * rz_scale[:, None, None], precond_radial_alpha) if frzl.frss is not None else None
        fzsc = _apply_radial_tridi(frzl.fzsc * rz_scale[:, None, None], precond_radial_alpha)
        fzcs = _apply_radial_tridi(frzl.fzcs * rz_scale[:, None, None], precond_radial_alpha) if frzl.fzcs is not None else None
        flsc = _apply_radial_tridi(frzl.flsc * l_scale[:, None, None], precond_lambda_alpha)
        flcs = _apply_radial_tridi(frzl.flcs * l_scale[:, None, None], precond_lambda_alpha) if frzl.flcs is not None else None

    frzl_pre = TomnspsRZL(
        frcc=frcc,
        frss=frss,
        fzsc=fzsc,
        fzcs=fzcs,
        flsc=flsc,
        flcs=flcs,
        frsc=getattr(frzl, "frsc", None),
        frcs=getattr(frzl, "frcs", None),
        fzcc=getattr(frzl, "fzcc", None),
        fzss=getattr(frzl, "fzss", None),
        flcc=getattr(frzl, "flcc", None),
        flss=getattr(frzl, "flss", None),
    )

    mpol = int(cfg.mpol)
    ntor = int(cfg.ntor)
    nrange = ntor + 1
    nfp = float(cfg.nfp)
    w_mode_mn = (1.0 + (jnp.arange(mpol)[:, None] ** 2 + (jnp.arange(nrange)[None, :] * nfp) ** 2)) ** (
        -float(mode_diag_exponent)
    )
    frcc_u = frcc * w_mode_mn[None, :, :]
    frss_u = (frss if frss is not None else jnp.zeros_like(frcc_u)) * w_mode_mn[None, :, :]
    fzsc_u = fzsc * w_mode_mn[None, :, :]
    fzcs_u = (fzcs if fzcs is not None else jnp.zeros_like(fzsc_u)) * w_mode_mn[None, :, :]
    flsc_u = flsc * w_mode_mn[None, :, :]
    flcs_u = (flcs if flcs is not None else jnp.zeros_like(flsc_u)) * w_mode_mn[None, :, :]

    def _mode_rms(a):
        a = jnp.asarray(a)
        return jnp.sqrt(jnp.mean(a * a, axis=0))

    frcc_mode = _mode_rms(frcc_u)
    fzsc_mode = _mode_rms(fzsc_u)
    flsc_mode = _mode_rms(flsc_u)

    gcr2_p, gcz2_p, gcl2_p = vmec_gcx2_from_tomnsps(
        frzl=frzl_pre,
        lconm1=bool(getattr(static.cfg, "lconm1", True)),
        apply_m1_constraints=False,
        include_edge=True,
        apply_scalxc=False,
        s=s,
    )
    rz_norm = vmec_rz_norm_from_state(
        state=state0,
        static=static,
        s=s,
        apply_scalxc=False,
        ns_min=0,
        ns_max=int(jnp.asarray(s).shape[0]),
    )
    norms = vmec_force_norms_from_bcovar_dynamic(bc=k.bc, trig=trig, s=s, signgs=signgs)
    f_norm1 = jnp.where(rz_norm != 0.0, 1.0 / rz_norm, jnp.asarray(float("inf"), dtype=rz_norm.dtype))
    delta_s = jnp.asarray(s[1] - s[0], dtype=rz_norm.dtype)
    fsqr1 = gcr2_p * f_norm1
    fsqz1 = gcz2_p * f_norm1
    fsql1 = gcl2_p * delta_s

    if step_size is None:
        time_step = float(indata.get_float("DELT", 5e-3))
    else:
        time_step = float(step_size)
    invtau = 0.15 / time_step
    otav = invtau
    dtau = time_step * otav / 2.0
    b1 = 1.0 - dtau
    fac = 1.0 / (1.0 + dtau)

    vRcc = fac * time_step * frcc_u
    vRss = fac * time_step * frss_u
    vZsc = fac * time_step * fzsc_u
    vZcs = fac * time_step * fzcs_u
    vLsc = fac * time_step * flsc_u
    vLcs = fac * time_step * flcs_u

    dRcc = time_step * vRcc
    dRss = time_step * vRss
    dZsc = time_step * vZsc
    dZcs = time_step * vZcs
    dLsc = time_step * vLsc
    dLcs = time_step * vLcs

    return {
        "fsqr": float(np.asarray(fsqr)),
        "fsqz": float(np.asarray(fsqz)),
        "fsql": float(np.asarray(fsql)),
        "fsqr1": float(np.asarray(fsqr1)),
        "fsqz1": float(np.asarray(fsqz1)),
        "fsql1": float(np.asarray(fsql1)),
        "gcr2_raw": float(np.asarray(gcr2_raw)),
        "gcz2_raw": float(np.asarray(gcz2_raw)),
        "gcl2_raw": float(np.asarray(gcl2_raw)),
        "rz_norm": float(np.asarray(rz_norm)),
        "f_norm1": float(np.asarray(f_norm1)),
        "f_norm_rz": float(np.asarray(norms.fnorm)),
        "f_norm_l": float(np.asarray(norms.fnormL)),
        "scalxc": np.asarray(vmec_scalxc_from_s(s=s, mpol=int(cfg.mpol))),
        "time_step": float(time_step),
        "dtau": float(dtau),
        "b1": float(b1),
        "fac": float(fac),
        "rz_scale": np.asarray(rz_scale),
        "l_scale": np.asarray(l_scale),
        "frzl": frzl,
        "frzl_pre": frzl_pre,
        "frcc_u": np.asarray(frcc_u),
        "frss_u": np.asarray(frss_u),
        "fzsc_u": np.asarray(fzsc_u),
        "fzcs_u": np.asarray(fzcs_u),
        "flsc_u": np.asarray(flsc_u),
        "flcs_u": np.asarray(flcs_u),
        "frcc_mode_rms": np.asarray(frcc_mode),
        "fzsc_mode_rms": np.asarray(fzsc_mode),
        "flsc_mode_rms": np.asarray(flsc_mode),
        "dRcc": np.asarray(dRcc),
        "dRss": np.asarray(dRss),
        "dZsc": np.asarray(dZsc),
        "dZcs": np.asarray(dZcs),
        "dLsc": np.asarray(dLsc),
        "dLcs": np.asarray(dLcs),
        "bcovar": k.bc,
    }
    if use_scan and dumps_enabled:
        raise ValueError("use_scan is incompatible with debug dumps (VMEC_JAX_DUMP_*).")
