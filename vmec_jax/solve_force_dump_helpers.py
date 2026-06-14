"""Force-channel debug dump helpers for VMEC solve diagnostics."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ._solve_runtime import _parse_iter_list


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
