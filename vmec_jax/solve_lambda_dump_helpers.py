"""Lambda/preconditioner debug dump helpers for VMEC solve diagnostics."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ._compat import jnp
from ._solve_runtime import _parse_iter_list
from .solve_force_dump_helpers import gc_from_frzl


def maybe_dump_lam_prec(*, lam_prec, faclam, static, iter_idx: int) -> None:
    """Optionally dump lambda preconditioner arrays in VMEC ``t`` channel layout."""

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
    if lasym:
        ntmax = 4 if lthreed else 2
    else:
        ntmax = 2 if lthreed else 1
    lam_arr = np.asarray(lam_prec)
    if lam_arr.ndim != 3:
        raise ValueError(f"lam_prec expected 3D (ns,mpol,ntor+1), got {lam_arr.shape}")

    pfaclam = np.zeros((ns, lam_arr.shape[2], lam_arr.shape[1], ntmax), dtype=lam_arr.dtype)
    pfaclam[:, :, :, 0] = np.transpose(lam_arr, (0, 2, 1))
    if ntmax > 1:
        pfaclam[:, :, :, 1:ntmax] = pfaclam[:, :, :, :1]
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


def maybe_dump_precond_mats(*, mats, static, iter_idx: int, jmax: int, used_cache: bool | None = None) -> None:
    """Optionally dump radial preconditioner matrix channels."""

    env = os.getenv("VMEC_JAX_DUMP_PRECOND_MATS", "")
    if not env or env == "0":
        return
    iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_PRECOND_MATS_ITER", ""))
    if iters is None:
        iters = _parse_iter_list(os.getenv("VMEC_JAX_DUMP_ITER", ""))
    if iters is not None and int(iter_idx) not in iters:
        return
    outdir = os.getenv("VMEC_JAX_DUMP_PRECOND_MATS_DIR", "")
    if not outdir:
        outdir = os.getenv("VMEC_JAX_DUMP_DIR", ".")
    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ns = int(static.cfg.ns)
    path = outdir / f"precond_mats_ns{ns}_iter{int(iter_idx)}.npz"
    data = {
        "ns": ns,
        "mpol": int(static.cfg.mpol),
        "ntor": int(static.cfg.ntor),
        "lthreed": bool(static.cfg.lthreed),
        "lasym": bool(static.cfg.lasym),
        "jmax": int(jmax),
    }
    if used_cache is not None:
        data["used_cache"] = bool(used_cache)
    for key in ("ar", "br", "dr", "az", "bz", "dz"):
        if key in mats:
            data[key] = np.asarray(mats[key])
    np.savez(path, **data)


def maybe_dump_lam_fsql1(*, fsql1_pre, fsql1_post, static, iter_idx: int) -> None:
    """Optionally dump pre/post lambda residual scalars."""

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


def maybe_dump_lamcal(*, lam_debug: dict[str, np.ndarray], static, iter_idx: int) -> None:
    """Optionally dump lambda-calculation debug arrays."""

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


def maybe_dump_lam_gcl(
    *,
    frzl_pre,
    frzl_post,
    static,
    iter_idx: int,
    delta_s,
) -> None:
    """Optionally dump lambda force-channel residuals before and after faclam."""

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

    _gcr_pre, _gcz_pre, gcl_pre = gc_from_frzl(frzl=frzl_pre, cfg=static.cfg)
    _gcr_post, _gcz_post, gcl_post = gc_from_frzl(frzl=frzl_post, cfg=static.cfg)

    gcl_pre = np.asarray(gcl_pre)
    gcl_post = np.asarray(gcl_post)
    delta_s_f = float(np.asarray(delta_s))
    fsql1_pre = float(np.sum(gcl_pre[1:] * gcl_pre[1:]) * delta_s_f)
    fsql1_post = float(np.sum(gcl_post[1:] * gcl_post[1:]) * delta_s_f)

    maybe_dump_lam_fsql1(
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


def maybe_dump_lulv(*, bc, static, iter_idx: int, state=None, trig=None) -> None:
    """Optionally dump lambda derivative fields and odd-m synthesis pieces."""

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
