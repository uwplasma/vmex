"""Driver helpers for VMEC-style residual scalars and wout construction."""

from __future__ import annotations

from pathlib import Path
import os
from typing import Any, Callable

import numpy as np


def residual_scalars_from_state(
    *,
    state,
    static,
    indata,
    signgs: int,
    wout=None,
    use_vmec_synthesis: bool = True,
):
    """Compute VMEC-style invariant residual scalars from a solved state."""

    from ..vmec_forces import vmec_forces_rz_from_wout, vmec_residual_internal_from_kernels
    from ..vmec_residue import vmec_force_norms_from_bcovar_dynamic, vmec_fsq_from_tomnsps_dynamic
    from ..vmec_tomnsp import TomnspsRZL, vmec_trig_tables

    class _WoutLike:
        __slots__ = ("nfp", "mpol", "ntor", "lasym", "signgs")

        def __init__(self, *, nfp: int, mpol: int, ntor: int, lasym: bool, signgs: int):
            self.nfp = int(nfp)
            self.mpol = int(mpol)
            self.ntor = int(ntor)
            self.lasym = bool(lasym)
            self.signgs = int(signgs)

    wout_like = wout
    if wout_like is None:
        wout_like = _WoutLike(
            nfp=int(static.cfg.nfp),
            mpol=int(static.cfg.mpol),
            ntor=int(static.cfg.ntor),
            lasym=bool(static.cfg.lasym),
            signgs=int(signgs),
        )

    trig = vmec_trig_tables(
        ntheta=int(static.cfg.ntheta),
        nzeta=int(static.cfg.nzeta),
        nfp=int(wout_like.nfp),
        mmax=int(wout_like.mpol) - 1,
        nmax=int(wout_like.ntor),
        lasym=bool(wout_like.lasym),
    )

    k = vmec_forces_rz_from_wout(
        state=state,
        static=static,
        wout=wout_like,
        indata=indata,
        use_wout_bsup=False,
        use_vmec_synthesis=bool(use_vmec_synthesis),
        trig=trig,
    )
    rzl = vmec_residual_internal_from_kernels(
        k,
        cfg_ntheta=int(static.cfg.ntheta),
        cfg_nzeta=int(static.cfg.nzeta),
        wout=wout_like,
        trig=trig,
    )
    frzl = TomnspsRZL(
        frcc=rzl.frcc,
        frss=rzl.frss,
        fzsc=rzl.fzsc,
        fzcs=rzl.fzcs,
        flsc=rzl.flsc,
        flcs=rzl.flcs,
        frsc=rzl.frsc,
        frcs=rzl.frcs,
        fzcc=rzl.fzcc,
        fzss=rzl.fzss,
        flcc=rzl.flcc,
        flss=rzl.flss,
    )
    norms = vmec_force_norms_from_bcovar_dynamic(bc=k.bc, trig=trig, s=static.s, signgs=int(signgs))
    scal = vmec_fsq_from_tomnsps_dynamic(
        frzl=frzl, norms=norms, lconm1=bool(getattr(static.cfg, "lconm1", True))
    )
    return float(scal.fsqr), float(scal.fsqz), float(scal.fsql)


def wout_from_fixed_boundary_run(
    run: Any,
    *,
    include_fsq: bool = True,
    path: str | Path | None = None,
    fast_bcovar: bool | None = None,
    residual_scalars_func: Callable[..., tuple[float, float, float]] = residual_scalars_from_state,
):
    """Build a minimal VMEC-style ``WoutData`` from a fixed-boundary run."""

    from ..wout import wout_minimal_from_fixed_boundary

    path = Path(path) if path is not None else Path("wout_vmec_jax.nc")

    fast_bcovar_eff = fast_bcovar
    if fast_bcovar_eff is None:
        diagnostics = getattr(getattr(run, "result", None), "diagnostics", {}) or {}
        if str(diagnostics.get("solver_mode", "")).strip().lower() == "parity":
            fast_bcovar_eff = False

    prev_fast_bcovar = None
    if fast_bcovar_eff is not None:
        prev_fast_bcovar = os.getenv("VMEC_JAX_WOUT_FAST_BCOVAR")
        os.environ["VMEC_JAX_WOUT_FAST_BCOVAR"] = "1" if fast_bcovar_eff else "0"

    try:
        fsqt = None
        converged = None
        if include_fsq:
            fsqr = fsqz = fsql = None
            res = getattr(run, "result", None)
            if res is not None:
                converged = getattr(res, "diagnostics", {}).get("converged", None)
                fsqr_hist = getattr(res, "fsqr2_history", None)
                fsqz_hist = getattr(res, "fsqz2_history", None)
                fsql_hist = getattr(res, "fsql2_history", None)
                if fsqr_hist is not None and fsqz_hist is not None:
                    fsqr_hist = np.asarray(fsqr_hist, dtype=float)
                    fsqz_hist = np.asarray(fsqz_hist, dtype=float)
                    fsqt_hist = fsqr_hist + fsqz_hist
                    nstore = 100
                    niter = int(fsqt_hist.size)
                    stride = int(niter // nstore) + 1 if niter > 0 else 1
                    fsqt = np.zeros((nstore,), dtype=float)
                    count = 0
                    for iter2 in range(1, niter + 1):
                        if iter2 % stride != 0:
                            continue
                        fsqt[count] = float(fsqt_hist[iter2 - 1])
                        count += 1
                        if count >= nstore:
                            break
                if fsqr_hist is not None and fsqz_hist is not None and fsql_hist is not None:
                    try:
                        fsqr = float(np.asarray(fsqr_hist)[-1])
                        fsqz = float(np.asarray(fsqz_hist)[-1])
                        fsql = float(np.asarray(fsql_hist)[-1])
                    except Exception:
                        fsqr = fsqz = fsql = None
            if fsqr is None or fsqz is None or fsql is None:
                fsqr, fsqz, fsql = residual_scalars_func(
                    state=run.state,
                    static=run.static,
                    indata=run.indata,
                    signgs=int(run.signgs),
                    use_vmec_synthesis=True,
                )
        else:
            fsqr = fsqz = fsql = 0.0

        wout = wout_minimal_from_fixed_boundary(
            path=path,
            state=run.state,
            static=run.static,
            indata=run.indata,
            signgs=int(run.signgs),
            fsqr=float(fsqr),
            fsqz=float(fsqz),
            fsql=float(fsql),
            fsqt=fsqt,
            converged=converged,
            flux_override=getattr(run, "flux", None),
            profiles_override=getattr(run, "profiles", None),
            force_payload_override=getattr(getattr(run, "result", None), "_final_force_payload", None),
        )
    finally:
        if fast_bcovar_eff is not None:
            if prev_fast_bcovar is None:
                os.environ.pop("VMEC_JAX_WOUT_FAST_BCOVAR", None)
            else:
                os.environ["VMEC_JAX_WOUT_FAST_BCOVAR"] = prev_fast_bcovar
    return wout


def write_wout_from_fixed_boundary_run(
    path: str | Path,
    run: Any,
    *,
    include_fsq: bool = True,
    fast_bcovar: bool | None = None,
    wout_from_run_func: Callable[..., Any] = wout_from_fixed_boundary_run,
):
    """Write a minimal VMEC-style ``wout_*.nc`` from a fixed-boundary run."""

    from ..wout import write_wout

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    wout = wout_from_run_func(run, include_fsq=include_fsq, path=path, fast_bcovar=fast_bcovar)
    write_wout(path, wout, overwrite=True)
    return wout
