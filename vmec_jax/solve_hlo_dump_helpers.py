"""HLO dump helpers for optional VMEC solver diagnostics."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from ._compat import has_jax


HLO_DUMPED_KEYS: set[tuple[str, int, int, int, int, int, bool]] = set()


def maybe_dump_hlo_kernel(
    *,
    label: str,
    fn,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    static: Any,
    wout_like: Any,
    force: bool = False,
    has_jax_func: Callable[[], bool] = has_jax,
    path_cls: type[Path] = Path,
) -> None:
    """Optionally dump lowered JAX HLO text for a solver kernel."""

    env_dir = os.getenv("VMEC_JAX_DUMP_HLO_DIR", "").strip()
    if not env_dir:
        return
    env_all = os.getenv("VMEC_JAX_DUMP_HLO", "").strip().lower()
    enabled_all = env_all not in ("", "0", "false", "no")
    env_label = os.getenv(f"VMEC_JAX_DUMP_HLO_{label.upper()}", "").strip().lower()
    enabled_label = env_label not in ("", "0", "false", "no")
    if not force and not (enabled_all or enabled_label):
        return
    if not has_jax_func():
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
    if key in HLO_DUMPED_KEYS:
        return

    try:
        import jax
    except Exception:
        return

    outdir = path_cls(env_dir).expanduser().resolve()
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
        HLO_DUMPED_KEYS.add(key)
    except Exception:
        return
