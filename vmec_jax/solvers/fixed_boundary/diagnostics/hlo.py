"""HLO dump helpers for optional VMEC solver diagnostics."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from ...._compat import has_jax


HLO_DUMPED_KEYS: set[tuple[str, int, int, int, int, int, bool]] = set()
_HLO_DUMPED_KEYS = HLO_DUMPED_KEYS


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


def _maybe_dump_hlo_kernel(**kwargs) -> None:
    """Compatibility wrapper with dynamically monkeypatchable JAX detection."""

    kwargs.setdefault("has_jax_func", has_jax)
    maybe_dump_hlo_kernel(**kwargs)


def maybe_dump_initial_residual_hlo_kernels(
    *,
    state0: Any,
    static: Any,
    wout_like: Any,
    trig: Any,
    constraint_tcon0: Any,
    apply_lforbal: bool,
    maybe_dump_kernel: Callable[..., None] = maybe_dump_hlo_kernel,
    getenv: Callable[[str, str], str] = os.getenv,
) -> None:
    """Dump optional first-call HLO probes for residual force assembly.

    This is intentionally diagnostic-only.  All failures are swallowed so that
    debug HLO extraction never changes solver behavior.
    """

    if not getenv("VMEC_JAX_DUMP_HLO_DIR", "").strip():
        return
    try:

        def _bcovar_only(st):
            from vmec_jax.kernels.bcovar import vmec_bcovar_half_mesh_from_wout

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

        maybe_dump_kernel(
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
        from vmec_jax.kernels.forces import vmec_forces_rz_from_wout
        from vmec_jax.kernels.forces import vmec_residual_internal_from_kernels

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

        maybe_dump_kernel(
            label="tomnsps",
            fn=_tomnsps_only,
            args=(k_hlo,),
            kwargs={},
            static=static,
            wout_like=wout_like,
        )
    except Exception:
        pass
