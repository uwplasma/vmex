"""Covariant-field debug dump helpers for VMEC solve diagnostics."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ...._solve_runtime import _parse_iter_list


def maybe_dump_bsube(*, bc, static, iter_idx: int) -> None:
    """Optionally dump scaled full-mesh covariant ``B_u/B_v`` fields."""

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
                    f.write(f"{js + 1:6d}{lt + 1:6d}{lz + 1:6d}{bsubu[js, lt, lz]:24.16e}{bsubv[js, lt, lz]:24.16e}\n")


def maybe_dump_bsube_terms(*, bc, static, iter_idx: int) -> None:
    """Optionally dump terms entering scaled full-mesh covariant fields."""

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


def maybe_dump_bsubh(*, bc, static, iter_idx: int) -> None:
    """Optionally dump half-mesh covariant ``B_u/B_v`` fields."""

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
                    f.write(f"{js + 1:6d}{lt + 1:6d}{lz + 1:6d}{bsubu[js, lt, lz]:24.16e}{bsubv[js, lt, lz]:24.16e}\n")


def maybe_dump_bsubs(*, bc, state, static, trig, iter_idx: int, kernels=None) -> None:
    """Optionally dump radial covariant ``B_s`` reconstruction diagnostics."""

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

    from ....wout import _compute_bsubs_half_mesh, _vmec_symforce_apply

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
