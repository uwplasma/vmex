"""Force-channel debug dump helpers for VMEC solve diagnostics."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ...._solve_runtime import _dump_env_enabled, _dump_iter_selected, _parse_iter_list


def dump_array(x):
    """Return an array suitable for optional NPZ debug dumps."""

    return np.asarray(x) if x is not None else np.zeros((0,), dtype=float)


def gc_from_frzl(*, frzl, cfg):
    """Map VMEC force channels into the legacy ``gc`` debug-dump layout."""

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


def maybe_dump_gc(*, frzl, static, iter_idx: int, label: str) -> None:
    """Optionally dump force-channel ``gc`` arrays in the legacy debug layout."""

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
    gcr, gcz, gcl = gc_from_frzl(frzl=frzl, cfg=static.cfg)
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


def maybe_dump_tomnsps(*, frzl, static, iter_idx: int, label: str = "raw") -> None:
    """Optionally dump TOMNSP force blocks for parity/debug inspection."""

    env = os.getenv("VMEC_JAX_DUMP_TOMNSPS", "")
    if not _dump_env_enabled(env):
        return
    if not _dump_iter_selected(iter_idx=iter_idx, iter_env=os.getenv("VMEC_JAX_DUMP_ITER", "")):
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns = int(static.cfg.ns)
    path = outdir / f"tomnsps_{label}_ns{ns}_iter{int(iter_idx)}.npz"

    np.savez(
        path,
        frcc=dump_array(frzl.frcc),
        frss=dump_array(getattr(frzl, "frss", None)),
        fzsc=dump_array(frzl.fzsc),
        fzcs=dump_array(getattr(frzl, "fzcs", None)),
        flsc=dump_array(frzl.flsc),
        flcs=dump_array(getattr(frzl, "flcs", None)),
        frsc=dump_array(getattr(frzl, "frsc", None)),
        frcs=dump_array(getattr(frzl, "frcs", None)),
        fzcc=dump_array(getattr(frzl, "fzcc", None)),
        fzss=dump_array(getattr(frzl, "fzss", None)),
        flcc=dump_array(getattr(frzl, "flcc", None)),
        flss=dump_array(getattr(frzl, "flss", None)),
        ns=int(static.cfg.ns),
        mpol=int(static.cfg.mpol),
        ntor=int(static.cfg.ntor),
        lasym=bool(static.cfg.lasym),
    )


def maybe_dump_force_kernels(*, k, static, iter_idx: int, label: str = "raw") -> None:
    """Optionally dump low-level force-kernel fields and BCOVAR payloads."""

    env = os.getenv("VMEC_JAX_DUMP_FORCE_KERNELS", "")
    if not _dump_env_enabled(env):
        return
    if not _dump_iter_selected(iter_idx=iter_idx, iter_env=os.getenv("VMEC_JAX_DUMP_ITER", "")):
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns = int(static.cfg.ns)
    path = outdir / f"force_kernels_{label}_ns{ns}_iter{int(iter_idx)}.npz"

    np.savez(
        path,
        armn_e=dump_array(getattr(k, "armn_e", None)),
        armn_o=dump_array(getattr(k, "armn_o", None)),
        brmn_e=dump_array(getattr(k, "brmn_e", None)),
        brmn_o=dump_array(getattr(k, "brmn_o", None)),
        crmn_e=dump_array(getattr(k, "crmn_e", None)),
        crmn_o=dump_array(getattr(k, "crmn_o", None)),
        azmn_e=dump_array(getattr(k, "azmn_e", None)),
        azmn_o=dump_array(getattr(k, "azmn_o", None)),
        bzmn_e=dump_array(getattr(k, "bzmn_e", None)),
        bzmn_o=dump_array(getattr(k, "bzmn_o", None)),
        czmn_e=dump_array(getattr(k, "czmn_e", None)),
        czmn_o=dump_array(getattr(k, "czmn_o", None)),
        arcon_e=dump_array(getattr(k, "arcon_e", None)),
        arcon_o=dump_array(getattr(k, "arcon_o", None)),
        azcon_e=dump_array(getattr(k, "azcon_e", None)),
        azcon_o=dump_array(getattr(k, "azcon_o", None)),
        gcon=dump_array(getattr(k, "gcon", None)),
        tcon=dump_array(getattr(k, "tcon", None)),
        blmn_e=dump_array(getattr(getattr(k, "bc", None), "blmn_even", None)),
        blmn_o=dump_array(getattr(getattr(k, "bc", None), "blmn_odd", None)),
        clmn_e=dump_array(getattr(getattr(k, "bc", None), "clmn_even", None)),
        clmn_o=dump_array(getattr(getattr(k, "bc", None), "clmn_odd", None)),
        bsubu_e=dump_array(getattr(getattr(k, "bc", None), "bsubu_e", None)),
        bsubv_e=dump_array(getattr(getattr(k, "bc", None), "bsubv_e", None)),
        bsubu=dump_array(getattr(getattr(k, "bc", None), "bsubu", None)),
        bsubv=dump_array(getattr(getattr(k, "bc", None), "bsubv", None)),
        bsupu=dump_array(getattr(getattr(k, "bc", None), "bsupu", None)),
        bsupv=dump_array(getattr(getattr(k, "bc", None), "bsupv", None)),
        guu_metric=dump_array(getattr(getattr(k, "bc", None), "guu", None)),
        guv_metric=dump_array(getattr(getattr(k, "bc", None), "guv", None)),
        gvv_metric=dump_array(getattr(getattr(k, "bc", None), "gvv", None)),
        sqrtg=dump_array(getattr(getattr(getattr(k, "bc", None), "jac", None), "sqrtg", None)),
        r12=dump_array(getattr(getattr(getattr(k, "bc", None), "jac", None), "r12", None)),
        tau=dump_array(getattr(getattr(getattr(k, "bc", None), "jac", None), "tau", None)),
        ru12=dump_array(getattr(getattr(getattr(k, "bc", None), "jac", None), "ru12", None)),
        zu12=dump_array(getattr(getattr(getattr(k, "bc", None), "jac", None), "zu12", None)),
        rs=dump_array(getattr(getattr(getattr(k, "bc", None), "jac", None), "rs", None)),
        zs=dump_array(getattr(getattr(getattr(k, "bc", None), "jac", None), "zs", None)),
        bsubu_e_scaled=dump_array(
            getattr(getattr(k, "bc", None), "bsubu_e_scaled", None)
            if getattr(getattr(k, "bc", None), "bsubu_e_scaled", None) is not None
            else getattr(getattr(k, "bc", None), "clmn_even", None)
        ),
        bsubv_e_scaled=dump_array(
            getattr(getattr(k, "bc", None), "bsubv_e_scaled", None)
            if getattr(getattr(k, "bc", None), "bsubv_e_scaled", None) is not None
            else getattr(getattr(k, "bc", None), "blmn_even", None)
        ),
        bsubu_tmp=dump_array(getattr(getattr(k, "bc", None), "bsubu_tmp", None)),
        bsubv_preblend=dump_array(getattr(getattr(k, "bc", None), "bsubv_preblend", None)),
        bsubv_avg=dump_array(getattr(getattr(k, "bc", None), "bsubv_avg", None)),
        lamscale=dump_array(getattr(getattr(k, "bc", None), "lamscale", None)),
        lu0_full=dump_array(getattr(getattr(k, "bc", None), "lu0_full", None)),
        lu0_force=dump_array(getattr(getattr(k, "bc", None), "lu0_force", None)),
        lu1_full=dump_array(getattr(getattr(k, "bc", None), "lu1_full", None)),
        lvv=dump_array(getattr(getattr(k, "bc", None), "lvv", None)),
        lvv_sh=dump_array(getattr(getattr(k, "bc", None), "lvv_sh", None)),
        phip_full=dump_array(getattr(getattr(k, "bc", None), "phip_full", None)),
        phip_internal=dump_array(getattr(getattr(k, "bc", None), "phip_internal", None)),
        pr1_even=dump_array(getattr(k, "pr1_even", None)),
        pr1_odd=dump_array(getattr(k, "pr1_odd", None)),
        pz1_even=dump_array(getattr(k, "pz1_even", None)),
        pz1_odd=dump_array(getattr(k, "pz1_odd", None)),
        pru_even=dump_array(getattr(k, "pru_even", None)),
        pru_odd=dump_array(getattr(k, "pru_odd", None)),
        pzu_even=dump_array(getattr(k, "pzu_even", None)),
        pzu_odd=dump_array(getattr(k, "pzu_odd", None)),
        prv_even=dump_array(getattr(k, "prv_even", None)),
        prv_odd=dump_array(getattr(k, "prv_odd", None)),
        pzv_even=dump_array(getattr(k, "pzv_even", None)),
        pzv_odd=dump_array(getattr(k, "pzv_odd", None)),
        ns=int(static.cfg.ns),
        ntheta=int(static.cfg.ntheta),
        nzeta=int(static.cfg.nzeta),
        lasym=bool(static.cfg.lasym),
    )


def maybe_dump_scalars(*, norms, iter_idx: int, ns: int) -> None:
    """Optionally dump scalar residual and energy diagnostics."""

    env = os.getenv("VMEC_JAX_DUMP_SCALARS", "")
    if not _dump_env_enabled(env):
        return
    if not _dump_iter_selected(iter_idx=iter_idx, iter_env=os.getenv("VMEC_JAX_DUMP_ITER", "")):
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


def maybe_dump_gcx2(*, gcr2, gcz2, gcl2, iter_idx: int, include_edge: bool, ns: int) -> None:
    """Optionally dump post-scaling force-channel norm diagnostics."""

    env = os.getenv("VMEC_JAX_DUMP_GCX2", "")
    if not _dump_env_enabled(env):
        return
    if not _dump_iter_selected(iter_idx=iter_idx, iter_env=os.getenv("VMEC_JAX_DUMP_ITER", "")):
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
