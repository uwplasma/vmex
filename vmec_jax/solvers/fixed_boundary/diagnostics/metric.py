"""Metric, preconditioner-input, and state-vector debug dump helpers."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ...._solve_runtime import _parse_iter_list
from ....state import VMECState


def maybe_dump_precond_inputs(*, bc, trig, static, iter_idx: int, kernels=None) -> None:
    """Optionally dump real-space inputs used by the radial preconditioner."""

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
                        f"{j + 1:6d}{lt + 1:6d}{lz + 1:6d}"
                        f"{r12[j, lt, lz]:24.16E}"
                        f"{sqrtg[j, lt, lz]:24.16E}"
                        f"{bsq[j, lt, lz]:24.16E}"
                        f"{ru12[j, lt, lz]:24.16E}"
                        f"{zu12[j, lt, lz]:24.16E}"
                        f"{wint_full[j, lt, lz]:24.16E}\n"
                    )

    if kernels is None:
        return
    try:
        hidden = {
            "tau": np.asarray(bc.jac.tau),
            "rs": np.asarray(bc.jac.rs),
            "zs": np.asarray(bc.jac.zs),
            "pru_even": np.asarray(kernels.pru_even),
            "pru_odd": np.asarray(kernels.pru_odd),
            "pzu_even": np.asarray(kernels.pzu_even),
            "pzu_odd": np.asarray(kernels.pzu_odd),
            "pr1_odd": np.asarray(kernels.pr1_odd),
            "pz1_odd": np.asarray(kernels.pz1_odd),
        }
    except Exception:
        return
    np.savez(outdir / f"precond_hidden_iter{int(iter_idx)}.npz", **hidden)


def maybe_dump_gmetric(*, bc, static, iter_idx: int) -> None:
    """Optionally dump half-mesh metric coefficients in VMEC debug layout."""

    env = os.getenv("VMEC_JAX_DUMP_GMETRIC", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"gmetric_iter{int(iter_idx)}.dat"

    try:
        guu = np.asarray(bc.guu, dtype=float)
        guv = np.asarray(bc.guv, dtype=float)
        gvv = np.asarray(bc.gvv, dtype=float)
    except Exception:
        return

    if guu.ndim != 3 or guv.shape != guu.shape or gvv.shape != guu.shape:
        return

    # VMEC dumps `gmetric_iter*.dat` before the cylindrical `R^2` term is
    # added into `pgvv`, while the live JAX half-mesh fields store the later
    # post-`R^2` metric needed by `bsubv`, `wb`, and `wout` parity.
    gmetric_guu = np.array(guu, copy=True)
    gmetric_guv = np.array(guv, copy=True)
    gmetric_gvv = np.array(gvv, copy=True)
    try:
        r12 = np.asarray(bc.jac.r12, dtype=float)
        if r12.shape == gmetric_gvv.shape:
            gmetric_gvv = gmetric_gvv - (r12 * r12)
    except Exception:
        pass
    if gmetric_guu.shape[0] >= 1:
        gmetric_guu[0, :, :] = 0.0
        gmetric_guv[0, :, :] = 0.0
        gmetric_gvv[0, :, :] = 0.0

    ns = int(guu.shape[0])
    ntheta3 = int(guu.shape[1])
    nzeta = int(guu.shape[2])

    with path.open("w", encoding="utf-8") as f:
        f.write("# bcovar metric dump (half mesh)\n")
        f.write(f"ns={ns}\n")
        f.write(f"ntheta3={ntheta3}\n")
        f.write(f"nzeta={nzeta}\n")
        f.write("columns: js lt lz pguu pguv pgvv\n")
        for lt in range(ntheta3):
            for lz in range(nzeta):
                for js in range(ns):
                    f.write(
                        f"{js + 1:6d}{lt + 1:6d}{lz + 1:6d}"
                        f"{gmetric_guu[js, lt, lz]:24.16e}"
                        f"{gmetric_guv[js, lt, lz]:24.16e}"
                        f"{gmetric_gvv[js, lt, lz]:24.16e}\n"
                    )


def maybe_dump_xc(
    *,
    state: VMECState,
    vRcc,
    vRss,
    vZsc,
    vZcs,
    vLsc,
    vLcs,
    vRsc=None,
    vRcs=None,
    vZcc=None,
    vZss=None,
    vLcc=None,
    vLss=None,
    static,
    iter_idx: int,
) -> None:
    """Optionally dump VMEC internal ``xc`` and velocity-vector payloads."""

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
    from ....diagnostics import vmec_internal_mn_from_state, vmec_xc_from_mn_blocks

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
    # Include asymmetric channels when present so LASYM xc/v dumps match VMEC's
    # internal stacking and do not report false zero-components.
    if vRsc is not None:
        xcdot_kwargs["rsc"] = np.asarray(vRsc)
    if vRcs is not None:
        xcdot_kwargs["rcs"] = np.asarray(vRcs)
    if vZcc is not None:
        xcdot_kwargs["zcc"] = np.asarray(vZcc)
    if vZss is not None:
        xcdot_kwargs["zss"] = np.asarray(vZss)
    if vLcc is not None:
        xcdot_kwargs["lcc"] = np.asarray(vLcc)
    if vLss is not None:
        xcdot_kwargs["lss"] = np.asarray(vLss)
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
