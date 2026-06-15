"""Optional debug-output helpers for driver workflows."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


def maybe_dump_xc_init(*, state: Any, static: Any, label: str) -> None:
    """Write a VMEC ``xc``/``xcdot`` initial-state dump when requested."""

    env = os.getenv("VMEC_JAX_DUMP_XC_INIT", "")
    if not env or env == "0":
        return
    outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns = int(static.cfg.ns)
    suffix = f"_{label}" if label else ""
    path = outdir / f"xc_init{suffix}_ns{ns}.dat"
    from vmec_jax.diagnostics import vmec_internal_mn_from_state, vmec_xc_from_mn_blocks

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
    xcdot = np.zeros_like(xc)
    with path.open("w") as f:
        f.write("# xc/xcdot dump (init guess)\n")
        f.write(f"neqs={xc.size}\n")
        f.write("columns: i xc xcdot\n")
        for i, (x, xd) in enumerate(zip(xc, xcdot), start=1):
            f.write(f"{i:8d}{x:24.16e}{xd:24.16e}\n")


__all__ = ["maybe_dump_xc_init"]
