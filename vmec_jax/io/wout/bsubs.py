"""Half-mesh Bsubs construction for VMEC WOUT/JXBFORCE diagnostics."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from vmec_jax.state import VMECState
from vmec_jax.vmec_jacobian import _apply_vmec_axis_rules
from vmec_jax.vmec_parity import vmec_m1_internal_to_physical_signed
from vmec_jax.vmec_realspace import (
    vmec_realspace_synthesis,
    vmec_realspace_synthesis_dtheta,
    vmec_realspace_synthesis_dzeta_phys,
)

from .diagnostics import pshalf_from_s as _pshalf_from_s
from .jxbforce import _jxbforce_nyquist_limits
from .nyquist import vmec_wrout_nyquist_synthesis as _vmec_wrout_nyquist_synthesis
from .parity import undo_bss_scalxc_if_enabled as _undo_bss_scalxc_if_enabled

def compute_bsubs_half_mesh(
    *,
    state: VMECState,
    geom_modes,
    s: np.ndarray,
    lconm1: bool,
    lthreed: bool,
    lasym: bool,
    bsupu: np.ndarray,
    bsupv: np.ndarray,
    trig,
    geom: dict[str, Any],
    jac_half: Any | None = None,
    force_rs: np.ndarray | None = None,
    force_zs: np.ndarray | None = None,
    force_ru12: np.ndarray | None = None,
    force_zu12: np.ndarray | None = None,
    apply_m1_constraint: bool = False,
    apply_scalxc: bool = False,
) -> np.ndarray:
    """Compute bsubs on the half mesh using VMEC's bss.f conventions."""
    if bool(lasym):
        # LASYM path uses full-interval grids. When force-kernel parity is
        # supplied (via `geom["pr*"]` populated from symforce), the same
        # algebra as the VMEC bss.f half-mesh update applies.
        pass

    # Geometry fields split into even/odd-m components on the full mesh.
    # VMEC's realspace arrays are built directly from internal coefficients
    # (which already include the 1/sqrt(s) odd-m scaling), so we keep
    # apply_scalxc=False to match bss.f inputs.
    m = np.asarray(geom_modes.m, dtype=int)
    mask_even = (m % 2) == 0
    mask_m1 = m == 1
    mask_odd_rest = (m % 2 == 1) & (~mask_m1)
    Rcos = np.asarray(state.Rcos)
    Rsin = np.asarray(state.Rsin)
    Zcos = np.asarray(state.Zcos)
    Zsin = np.asarray(state.Zsin)
    if bool(lconm1) and bool(apply_m1_constraint):
        Rcos, Zsin, Rsin, Zcos = vmec_m1_internal_to_physical_signed(
            Rcos=Rcos,
            Zsin=Zsin,
            Rsin=Rsin,
            Zcos=Zcos,
            modes=geom_modes,
            lthreed=bool(lthreed),
            lasym=bool(lasym),
            lconm1=bool(lconm1),
        )
    Rcos = _apply_vmec_axis_rules(Rcos, m)
    Rsin = _apply_vmec_axis_rules(Rsin, m)
    Zcos = _apply_vmec_axis_rules(Zcos, m)
    Zsin = _apply_vmec_axis_rules(Zsin, m)

    coeff_cos_stack = np.stack([Rcos, Zcos], axis=0)
    coeff_sin_stack = np.stack([Rsin, Zsin], axis=0)

    mask_even_f = mask_even.astype(float)
    mask_m1_f = mask_m1.astype(float)
    mask_odd_rest_f = mask_odd_rest.astype(float)

    def _eval_mask(mask: np.ndarray, *, deriv: str, apply_scalxc_local: bool):
        coeff_cos = coeff_cos_stack * mask[None, None, :]
        coeff_sin = coeff_sin_stack * mask[None, None, :]
        if deriv == "base":
            return vmec_realspace_synthesis(
                coeff_cos=coeff_cos,
                coeff_sin=coeff_sin,
                modes=geom_modes,
                trig=trig,
                coeffs_internal=True,
                apply_scalxc=bool(apply_scalxc_local),
                s=s,
            )
        if deriv == "dtheta":
            return vmec_realspace_synthesis_dtheta(
                coeff_cos=coeff_cos,
                coeff_sin=coeff_sin,
                modes=geom_modes,
                trig=trig,
                coeffs_internal=True,
                apply_scalxc=bool(apply_scalxc_local),
                s=s,
            )
        if deriv == "dzeta":
            return vmec_realspace_synthesis_dzeta_phys(
                coeff_cos=coeff_cos,
                coeff_sin=coeff_sin,
                modes=geom_modes,
                trig=trig,
                coeffs_internal=True,
                apply_scalxc=bool(apply_scalxc_local),
                s=s,
            )
        raise ValueError(f"Unknown deriv {deriv}")

    if bool(lasym):
        # LASYM: VMEC's even/odd realspace fields correspond to cos/sin phase
        # components (theta parity), not m-parity splits. Build them directly
        # from the cos/sin coefficient stacks.
        zeros = np.zeros_like(coeff_cos_stack)
        even_base = np.asarray(
            vmec_realspace_synthesis(
                coeff_cos=coeff_cos_stack,
                coeff_sin=zeros,
                modes=geom_modes,
                trig=trig,
                coeffs_internal=True,
                apply_scalxc=False,
                s=s,
            )
        )
        even_t = np.asarray(
            vmec_realspace_synthesis_dtheta(
                coeff_cos=coeff_cos_stack,
                coeff_sin=zeros,
                modes=geom_modes,
                trig=trig,
                coeffs_internal=True,
                apply_scalxc=False,
                s=s,
            )
        )
        even_p = np.asarray(
            vmec_realspace_synthesis_dzeta_phys(
                coeff_cos=coeff_cos_stack,
                coeff_sin=zeros,
                modes=geom_modes,
                trig=trig,
                coeffs_internal=True,
                apply_scalxc=False,
                s=s,
            )
        )
        odd_base = np.asarray(
            vmec_realspace_synthesis(
                coeff_cos=zeros,
                coeff_sin=coeff_sin_stack,
                modes=geom_modes,
                trig=trig,
                coeffs_internal=True,
                apply_scalxc=False,
                s=s,
            )
        )
        odd_t = np.asarray(
            vmec_realspace_synthesis_dtheta(
                coeff_cos=zeros,
                coeff_sin=coeff_sin_stack,
                modes=geom_modes,
                trig=trig,
                coeffs_internal=True,
                apply_scalxc=False,
                s=s,
            )
        )
        odd_p = np.asarray(
            vmec_realspace_synthesis_dzeta_phys(
                coeff_cos=zeros,
                coeff_sin=coeff_sin_stack,
                modes=geom_modes,
                trig=trig,
                coeffs_internal=True,
                apply_scalxc=False,
                s=s,
            )
        )
    else:
        # Match VMEC/bcovar conventions:
        # - even components use physical coefficients (no scalxc),
        # - odd components use internal coefficients (apply scalxc).
        even_base = np.asarray(_eval_mask(mask_even_f, deriv="base", apply_scalxc_local=False))
        even_t = np.asarray(_eval_mask(mask_even_f, deriv="dtheta", apply_scalxc_local=False))
        even_p = np.asarray(_eval_mask(mask_even_f, deriv="dzeta", apply_scalxc_local=False))

        odd_m1_base = np.asarray(_eval_mask(mask_m1_f, deriv="base", apply_scalxc_local=bool(apply_scalxc)))
        odd_m1_t = np.asarray(_eval_mask(mask_m1_f, deriv="dtheta", apply_scalxc_local=bool(apply_scalxc)))
        odd_m1_p = np.asarray(_eval_mask(mask_m1_f, deriv="dzeta", apply_scalxc_local=bool(apply_scalxc)))

        odd_rest_base = np.asarray(_eval_mask(mask_odd_rest_f, deriv="base", apply_scalxc_local=bool(apply_scalxc)))
        odd_rest_t = np.asarray(_eval_mask(mask_odd_rest_f, deriv="dtheta", apply_scalxc_local=bool(apply_scalxc)))
        odd_rest_p = np.asarray(_eval_mask(mask_odd_rest_f, deriv="dzeta", apply_scalxc_local=bool(apply_scalxc)))

        odd_base = odd_m1_base + odd_rest_base
        odd_t = odd_m1_t + odd_rest_t
        odd_p = odd_m1_p + odd_rest_p
        if odd_base.shape[0] >= 2:
            # VMEC axis convention: copy m=1 odd field from js=2 to js=1.
            # Axis corresponds to the radial index (js=1 -> index 0).
            odd_base[0] = odd_m1_base[1]
            odd_t[0] = odd_m1_t[1]
            odd_p[0] = odd_m1_p[1]

    R_even = even_base[0]
    Z_even = even_base[1]
    Ru_even = even_t[0]
    Zu_even = even_t[1]
    Rv_even = even_p[0]
    Zv_even = even_p[1]

    R1 = odd_base[0]
    Z1 = odd_base[1]
    Ru1 = odd_t[0]
    Zu1 = odd_t[1]
    Rv1 = odd_p[0]
    Zv1 = odd_p[1]

    s = np.asarray(s, dtype=float)
    if s.shape[0] < 2:
        return np.zeros_like(np.asarray(bsupu, dtype=float))

    # VMEC's bss.f uses internal even/odd components with explicit shalf factors.
    # See jacobian.f: rs/zs use shalf scaling; rs12/zs12 include d(shalf)/ds.
    hs = float(s[1] - s[0])
    ohs = 1.0 / hs
    dphids = 0.25
    s_half = 0.5 * (s[1:] + s[:-1])
    shalf = np.zeros_like(s, dtype=float)
    shalf[1:] = np.sqrt(np.maximum(s_half, 0.0))
    sh = shalf[:, None, None]

    rv12 = np.zeros_like(R_even, dtype=float)
    zv12 = np.zeros_like(Z_even, dtype=float)
    rs12 = np.zeros_like(R_even, dtype=float)
    zs12 = np.zeros_like(Z_even, dtype=float)

    use_parity_geom_full = (
        isinstance(geom, dict)
        and ("pr1_even" in geom)
        and ("pr1_odd" in geom)
        and ("pz1_even" in geom)
        and ("pz1_odd" in geom)
        and ("pru_even" in geom)
        and ("pru_odd" in geom)
        and ("pzu_even" in geom)
        and ("pzu_odd" in geom)
    )
    # Prefer parity-geometry inputs for bss when available. VMEC's bss.f
    # uses the realspace (symforce) parity fields directly, so default to
    # that behavior unless explicitly disabled.
    use_parity_bss = use_parity_geom_full and (os.getenv("VMEC_JAX_BSS_FROM_PARITY_GEOM", "1") not in ("", "0"))
    use_force_terms = (
        force_rs is not None and force_zs is not None and force_ru12 is not None and force_zu12 is not None
    )
    if use_force_terms:
        use_parity_bss = False

    # Use force-kernel R/Z arrays (VMEC bss.f path) when supplied.
    if use_parity_bss:
        pr1_even = np.asarray(geom["pr1_even"], dtype=float)
        pr1_odd = np.asarray(geom["pr1_odd"], dtype=float)
        pz1_even = np.asarray(geom["pz1_even"], dtype=float)
        pz1_odd = np.asarray(geom["pz1_odd"], dtype=float)
        pru_even = np.asarray(geom["pru_even"], dtype=float)
        pru_odd = np.asarray(geom["pru_odd"], dtype=float)
        pzu_even = np.asarray(geom["pzu_even"], dtype=float)
        pzu_odd = np.asarray(geom["pzu_odd"], dtype=float)

        # The parity fields are built from internal coefficients with VMEC's
        # scalxc applied (odd-m scaled by 1/max(sqrt(s), sqrt(s2))). bss.f
        # expects the *internal* odd fields (before scalxc), so undo it here
        # when the compatibility flag is enabled.
        pr1_odd, pz1_odd, pru_odd, pzu_odd = _undo_bss_scalxc_if_enabled(
            s,
            pr1_odd,
            pz1_odd,
            pru_odd,
            pzu_odd,
        )

        ru12 = np.zeros_like(R_even, dtype=float)
        zu12 = np.zeros_like(Z_even, dtype=float)
        rs = np.zeros_like(R_even, dtype=float)
        zs = np.zeros_like(Z_even, dtype=float)
        ru12[1:] = 0.5 * (pru_even[1:] + pru_even[:-1] + sh[1:] * (pru_odd[1:] + pru_odd[:-1]))
        zu12[1:] = 0.5 * (pzu_even[1:] + pzu_even[:-1] + sh[1:] * (pzu_odd[1:] + pzu_odd[:-1]))
        rs[1:] = ohs * (pr1_even[1:] - pr1_even[:-1] + sh[1:] * (pr1_odd[1:] - pr1_odd[:-1]))
        zs[1:] = ohs * (pz1_even[1:] - pz1_even[:-1] + sh[1:] * (pz1_odd[1:] - pz1_odd[:-1]))
    elif use_force_terms:
        ru12 = np.array(force_ru12, dtype=float, copy=True)
        zu12 = np.array(force_zu12, dtype=float, copy=True)
        rs = np.array(force_rs, dtype=float, copy=True)
        zs = np.array(force_zs, dtype=float, copy=True)
    # Otherwise use half-mesh Jacobian from bcovar when provided to stay
    # consistent with bsupu/bsupv (computed from the same bcovar pipeline).
    elif jac_half is not None:
        ru12 = np.array(jac_half.ru12, dtype=float, copy=True)
        zu12 = np.array(jac_half.zu12, dtype=float, copy=True)
        rs = np.array(jac_half.rs, dtype=float, copy=True)
        zs = np.array(jac_half.zs, dtype=float, copy=True)
    else:
        ru12 = np.zeros_like(R_even, dtype=float)
        zu12 = np.zeros_like(Z_even, dtype=float)
        rs = np.zeros_like(R_even, dtype=float)
        zs = np.zeros_like(Z_even, dtype=float)
        ru12[1:] = 0.5 * (Ru_even[1:] + Ru_even[:-1] + sh[1:] * (Ru1[1:] + Ru1[:-1]))
        zu12[1:] = 0.5 * (Zu_even[1:] + Zu_even[:-1] + sh[1:] * (Zu1[1:] + Zu1[:-1]))
        rs[1:] = ohs * (R_even[1:] - R_even[:-1] + sh[1:] * (R1[1:] - R1[:-1]))
        zs[1:] = ohs * (Z_even[1:] - Z_even[:-1] + sh[1:] * (Z1[1:] - Z1[:-1]))

    use_parity_geom = (
        isinstance(geom, dict)
        and ("pr1_odd" in geom)
        and ("pz1_odd" in geom)
        and ("prv_even" in geom)
        and ("prv_odd" in geom)
        and ("pzv_even" in geom)
        and ("pzv_odd" in geom)
    )
    if use_parity_geom:
        pr1_odd = np.asarray(geom["pr1_odd"], dtype=float)
        pz1_odd = np.asarray(geom["pz1_odd"], dtype=float)
        prv_even = np.asarray(geom["prv_even"], dtype=float)
        prv_odd = np.asarray(geom["prv_odd"], dtype=float)
        pzv_even = np.asarray(geom["pzv_even"], dtype=float)
        pzv_odd = np.asarray(geom["pzv_odd"], dtype=float)

        pr1_odd, pz1_odd, prv_odd, pzv_odd = _undo_bss_scalxc_if_enabled(
            s,
            pr1_odd,
            pz1_odd,
            prv_odd,
            pzv_odd,
        )

        rv12[1:] = 0.5 * (prv_even[1:] + prv_even[:-1] + sh[1:] * (prv_odd[1:] + prv_odd[:-1]))
        zv12[1:] = 0.5 * (pzv_even[1:] + pzv_even[:-1] + sh[1:] * (pzv_odd[1:] + pzv_odd[:-1]))
        rs12[1:] = rs[1:] + dphids * (pr1_odd[1:] + pr1_odd[:-1]) / sh[1:]
        zs12[1:] = zs[1:] + dphids * (pz1_odd[1:] + pz1_odd[:-1]) / sh[1:]
    else:
        rv12[1:] = 0.5 * (Rv_even[1:] + Rv_even[:-1] + sh[1:] * (Rv1[1:] + Rv1[:-1]))
        zv12[1:] = 0.5 * (Zv_even[1:] + Zv_even[:-1] + sh[1:] * (Zv1[1:] + Zv1[:-1]))

        rs12[1:] = rs[1:] + dphids * (R1[1:] + R1[:-1]) / sh[1:]
        zs12[1:] = zs[1:] + dphids * (Z1[1:] + Z1[:-1]) / sh[1:]

    # Axis fill: mirror js=2 into js=1 (VMEC convention).
    if rs12.shape[0] > 1:
        rs12[0] = rs12[1]
        zs12[0] = zs12[1]
        ru12[0] = ru12[1]
        zu12[0] = zu12[1]
        rv12[0] = rv12[1]
        zv12[0] = zv12[1]

    g_su = rs12 * ru12 + zs12 * zu12
    g_sv = rs12 * rv12 + zs12 * zv12
    bsubs = np.asarray(bsupu, dtype=float) * g_su + np.asarray(bsupv, dtype=float) * g_sv

    if os.getenv("VMEC_JAX_DUMP_BSS_INPUTS", "") not in ("", "0"):
        outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        tag = os.getenv("VMEC_JAX_DUMP_TAG", "").strip()
        name = "bss_inputs_jax" + (f"_{tag}" if tag else "") + ".dat"
        path = outdir / name
        r12 = None
        if jac_half is not None:
            try:
                r12 = np.asarray(jac_half.r12, dtype=float)
            except Exception:
                r12 = None
        if r12 is None:
            r12 = np.zeros_like(R_even, dtype=float)
            r12[1:] = 0.5 * (R_even[1:] + R_even[:-1] + sh[1:] * (R1[1:] + R1[:-1]))
            r12[0] = r12[1]
        with path.open("w") as f:
            f.write("# bss inputs dump (half mesh)\n")
            f.write(f"ns={r12.shape[0]}\n")
            f.write(f"ntheta3={r12.shape[1]}\n")
            f.write(f"nzeta={r12.shape[2]}\n")
            f.write("columns: js lt lz r12 rs zs ru12 zu12 bsupu bsupv\n")
            ns, ntheta3, nzeta = r12.shape
            for lt in range(ntheta3):
                for lz in range(nzeta):
                    for js in range(ns):
                        bsupu_val = float(np.asarray(bsupu, dtype=float)[js, lt, lz])
                        bsupv_val = float(np.asarray(bsupv, dtype=float)[js, lt, lz])
                        f.write(
                            f"{js + 1:6d}{lt + 1:6d}{lz + 1:6d}"
                            f"{r12[js, lt, lz]:24.16E}{rs[js, lt, lz]:24.16E}{zs[js, lt, lz]:24.16E}"
                            f"{ru12[js, lt, lz]:24.16E}{zu12[js, lt, lz]:24.16E}"
                            f"{bsupu_val:24.16E}{bsupv_val:24.16E}\n"
                        )

    if os.getenv("VMEC_JAX_DUMP_BSS_TERMS", "") not in ("", "0"):
        outdir = Path(os.getenv("VMEC_JAX_DUMP_DIR", ".")).expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        tag = os.getenv("VMEC_JAX_DUMP_TAG", "").strip()
        name = "bss_terms_jax" + (f"_{tag}" if tag else "") + ".npz"
        pr1_even = (
            np.asarray(geom.get("pr1_even"), dtype=float)
            if isinstance(geom, dict) and "pr1_even" in geom
            else np.zeros_like(R_even, dtype=float)
        )
        pr1_odd = (
            np.asarray(geom.get("pr1_odd"), dtype=float)
            if isinstance(geom, dict) and "pr1_odd" in geom
            else np.zeros_like(R_even, dtype=float)
        )
        prv_even = (
            np.asarray(geom.get("prv_even"), dtype=float)
            if isinstance(geom, dict) and "prv_even" in geom
            else np.zeros_like(R_even, dtype=float)
        )
        prv_odd = (
            np.asarray(geom.get("prv_odd"), dtype=float)
            if isinstance(geom, dict) and "prv_odd" in geom
            else np.zeros_like(R_even, dtype=float)
        )
        pru_even = (
            np.asarray(geom.get("pru_even"), dtype=float)
            if isinstance(geom, dict) and "pru_even" in geom
            else np.zeros_like(R_even, dtype=float)
        )
        pru_odd = (
            np.asarray(geom.get("pru_odd"), dtype=float)
            if isinstance(geom, dict) and "pru_odd" in geom
            else np.zeros_like(R_even, dtype=float)
        )
        pzu_even = (
            np.asarray(geom.get("pzu_even"), dtype=float)
            if isinstance(geom, dict) and "pzu_even" in geom
            else np.zeros_like(R_even, dtype=float)
        )
        pzu_odd = (
            np.asarray(geom.get("pzu_odd"), dtype=float)
            if isinstance(geom, dict) and "pzu_odd" in geom
            else np.zeros_like(R_even, dtype=float)
        )
        np.savez(
            outdir / name,
            r1_even=np.asarray(R_even, dtype=float),
            r1_odd=np.asarray(R1, dtype=float),
            pr1_even=pr1_even,
            pr1_odd=pr1_odd,
            rv_even=np.asarray(Rv_even, dtype=float),
            rv_odd=np.asarray(Rv1, dtype=float),
            prv_even=prv_even,
            prv_odd=prv_odd,
            pru_even=pru_even,
            pru_odd=pru_odd,
            pzu_even=pzu_even,
            pzu_odd=pzu_odd,
            rs12=np.asarray(rs12, dtype=float),
            zs12=np.asarray(zs12, dtype=float),
            rv12=np.asarray(rv12, dtype=float),
            zv12=np.asarray(zv12, dtype=float),
            ru12=np.asarray(ru12, dtype=float),
            zu12=np.asarray(zu12, dtype=float),
            gsu=np.asarray(g_su, dtype=float),
            gsv=np.asarray(g_sv, dtype=float),
            bsubs=np.asarray(bsubs, dtype=float),
            bsupu=np.asarray(bsupu, dtype=float),
            bsupv=np.asarray(bsupv, dtype=float),
            s=np.asarray(s, dtype=float),
        )

    return bsubs



def bsubuv_parity_from_state(
    *,
    state: VMECState,
    geom_modes,
    trig,
    s: np.ndarray,
    lconm1: bool,
    lthreed: bool,
    lasym: bool,
    bsupu: np.ndarray,
    bsupv: np.ndarray,
    lu1_full: np.ndarray,
    lv1_full: np.ndarray,
    sqrtg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct parity-separated bsubu/bsubv using VMEC internal even/odd splitting."""
    m = np.asarray(geom_modes.m, dtype=int)
    mask_even = (m % 2) == 0
    mask_odd = ~mask_even

    Rcos = np.asarray(state.Rcos)
    Rsin = np.asarray(state.Rsin)
    Zcos = np.asarray(state.Zcos)
    Zsin = np.asarray(state.Zsin)
    if bool(lconm1):
        Rcos, Zsin, Rsin, Zcos = vmec_m1_internal_to_physical_signed(
            Rcos=Rcos,
            Zsin=Zsin,
            Rsin=Rsin,
            Zcos=Zcos,
            modes=geom_modes,
            lthreed=bool(lthreed),
            lasym=bool(lasym),
            lconm1=bool(lconm1),
        )

    Rcos = _apply_vmec_axis_rules(Rcos, m)
    Rsin = _apply_vmec_axis_rules(Rsin, m)
    Zcos = _apply_vmec_axis_rules(Zcos, m)
    Zsin = _apply_vmec_axis_rules(Zsin, m)

    coeff_cos_stack = np.stack([Rcos, Zcos], axis=0)
    coeff_sin_stack = np.stack([Rsin, Zsin], axis=0)
    mask_stack = np.stack([mask_even.astype(float), mask_odd.astype(float)], axis=0)
    coeff_cos = coeff_cos_stack[None, ...] * mask_stack[:, None, None, :]
    coeff_sin = coeff_sin_stack[None, ...] * mask_stack[:, None, None, :]

    stack = vmec_realspace_synthesis(
        coeff_cos=coeff_cos,
        coeff_sin=coeff_sin,
        modes=geom_modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=True,
        s=s,
    )
    stack_t = vmec_realspace_synthesis_dtheta(
        coeff_cos=coeff_cos,
        coeff_sin=coeff_sin,
        modes=geom_modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=True,
        s=s,
    )
    stack_p = vmec_realspace_synthesis_dzeta_phys(
        coeff_cos=coeff_cos,
        coeff_sin=coeff_sin,
        modes=geom_modes,
        trig=trig,
        coeffs_internal=True,
        apply_scalxc=True,
        s=s,
    )

    even = np.asarray(stack[0])
    odd = np.asarray(stack[1])
    even_t = np.asarray(stack_t[0])
    odd_t = np.asarray(stack_t[1])
    even_p = np.asarray(stack_p[0])
    odd_p = np.asarray(stack_p[1])

    Ru_even = even_t[0]
    Ru_odd = odd_t[0]
    Zu_even = even_t[1]
    Zu_odd = odd_t[1]
    Rv_even = even_p[0]
    Rv_odd = odd_p[0]
    Zv_even = even_p[1]
    Zv_odd = odd_p[1]

    pshalf = _pshalf_from_s(np.asarray(s, dtype=float))[:, None, None]
    # bsubu/bsubv live on the radial half mesh in VMEC. Their parity algebra
    # must therefore use s_half = pshalf^2 (not full-mesh s_j).
    s_term = pshalf * pshalf

    guu_even = Ru_even * Ru_even + Zu_even * Zu_even + s_term * (Ru_odd * Ru_odd + Zu_odd * Zu_odd)
    guu_odd = 2.0 * (Ru_even * Ru_odd + Zu_even * Zu_odd)
    guv_even = Ru_even * Rv_even + Zu_even * Zv_even + s_term * (Ru_odd * Rv_odd + Zu_odd * Zv_odd)
    guv_odd = Ru_even * Rv_odd + Ru_odd * Rv_even + Zu_even * Zv_odd + Zu_odd * Zv_even
    gvv_even = Rv_even * Rv_even + Zv_even * Zv_even + s_term * (Rv_odd * Rv_odd + Zv_odd * Zv_odd)
    gvv_odd = 2.0 * (Rv_even * Rv_odd + Zv_even * Zv_odd)

    overg = np.where(np.asarray(sqrtg) != 0.0, 1.0 / np.asarray(sqrtg), 0.0)
    bsupu_even = np.asarray(bsupu, dtype=float)
    bsupv_even = np.asarray(bsupv, dtype=float)
    bsupu_odd = np.zeros_like(bsupu_even)
    bsupv_odd = np.zeros_like(bsupv_even)
    if int(bsupu_even.shape[0]) >= 2:
        avg_lv1 = np.asarray(lv1_full[1:] + lv1_full[:-1], dtype=float)
        avg_lu1 = np.asarray(lu1_full[1:] + lu1_full[:-1], dtype=float)
        bsupu_odd[1:] = 0.5 * overg[1:] * avg_lv1
        bsupv_odd[1:] = 0.5 * overg[1:] * avg_lu1
        bsupu_even = bsupu_even - pshalf * bsupu_odd
        bsupv_even = bsupv_even - pshalf * bsupv_odd

    bsubu_even = (
        guu_even * bsupu_even + s_term * guu_odd * bsupu_odd + guv_even * bsupv_even + s_term * guv_odd * bsupv_odd
    )
    bsubu_odd = guu_even * bsupu_odd + guu_odd * bsupu_even + guv_even * bsupv_odd + guv_odd * bsupv_even
    bsubv_even = (
        guv_even * bsupu_even + s_term * guv_odd * bsupu_odd + gvv_even * bsupv_even + s_term * gvv_odd * bsupv_odd
    )
    bsubv_odd = guv_even * bsupu_odd + guv_odd * bsupu_even + gvv_even * bsupv_odd + gvv_odd * bsupv_even

    return bsubu_even, bsubu_odd, bsubv_even, bsubv_odd



def bsubuv_parity_from_coeffs(
    *,
    bsubumnc: np.ndarray,
    bsubumns: np.ndarray,
    bsubvmnc: np.ndarray,
    bsubvmns: np.ndarray,
    modes,
    trig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split bsubu/bsubv into even/odd m parity using Fourier coefficients."""
    m = np.asarray(modes.m, dtype=int)
    mask_even = (m % 2) == 0
    mask_odd = ~mask_even
    mask_even = mask_even[None, :]
    mask_odd = mask_odd[None, :]

    bsubumnc = np.asarray(bsubumnc, dtype=float)
    bsubumns = np.asarray(bsubumns, dtype=float)
    bsubvmnc = np.asarray(bsubvmnc, dtype=float)
    bsubvmns = np.asarray(bsubvmns, dtype=float)

    bsubumnc_even = bsubumnc * mask_even
    bsubumns_even = bsubumns * mask_even
    bsubumnc_odd = bsubumnc * mask_odd
    bsubumns_odd = bsubumns * mask_odd

    bsubvmnc_even = bsubvmnc * mask_even
    bsubvmns_even = bsubvmns * mask_even
    bsubvmnc_odd = bsubvmnc * mask_odd
    bsubvmns_odd = bsubvmns * mask_odd

    # Use wrout-style Nyquist synthesis instead of generic helical eval so the
    # parity split stays on VMEC's reduced-grid normalization.
    bsubu_even = _vmec_wrout_nyquist_synthesis(
        coeff_c=bsubumnc_even,
        coeff_s=bsubumns_even,
        modes=modes,
        trig=trig,
    )
    bsubu_odd = _vmec_wrout_nyquist_synthesis(
        coeff_c=bsubumnc_odd,
        coeff_s=bsubumns_odd,
        modes=modes,
        trig=trig,
    )
    bsubv_even = _vmec_wrout_nyquist_synthesis(
        coeff_c=bsubvmnc_even,
        coeff_s=bsubvmns_even,
        modes=modes,
        trig=trig,
    )
    bsubv_odd = _vmec_wrout_nyquist_synthesis(
        coeff_c=bsubvmnc_odd,
        coeff_s=bsubvmns_odd,
        modes=modes,
        trig=trig,
    )
    return bsubu_even, bsubu_odd, bsubv_even, bsubv_odd


def bsubuv_parity_from_realspace_jxbforce(
    *,
    bsubu: np.ndarray,
    bsubv: np.ndarray,
    trig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recover jxbforce parity channels directly from real-space bsubu/bsubv."""
    # jdotb can be cancellation-limited near the edge. Keep this projection in
    # long-double to reduce parity-channel roundoff before the jxbforce filter.
    acc_dtype = np.longdouble
    bsubu = np.asarray(bsubu, dtype=acc_dtype)
    bsubv = np.asarray(bsubv, dtype=acc_dtype)
    if bsubu.shape != bsubv.shape:
        raise ValueError("bsubu/bsubv shape mismatch")
    if bsubu.ndim != 3:
        raise ValueError("Expected bsubu/bsubv with shape (ns, ntheta, nzeta)")

    ns, ntheta, nzeta = bsubu.shape
    nt2 = int(trig.ntheta2)
    if ntheta < nt2:
        raise ValueError("bsubu grid smaller than ntheta2")

    mnyq, nnyq = _jxbforce_nyquist_limits(trig)
    mmax = int(max(mnyq, 0))
    nmax = int(max(nnyq, 0))

    cosmui = np.asarray(trig.cosmui, dtype=acc_dtype)[:nt2, : mmax + 1]
    sinmui = np.asarray(trig.sinmui, dtype=acc_dtype)[:nt2, : mmax + 1]
    cosmu = np.asarray(trig.cosmu, dtype=acc_dtype)[:nt2, : mmax + 1]
    sinmu = np.asarray(trig.sinmu, dtype=acc_dtype)[:nt2, : mmax + 1]
    cosnv = np.asarray(trig.cosnv, dtype=acc_dtype)[:, : nmax + 1]
    sinnv = np.asarray(trig.sinnv, dtype=acc_dtype)[:, : nmax + 1]

    r0scale = float(getattr(trig, "r0scale", 1.0))
    base_dnorm = acc_dtype(1.0) / acc_dtype(r0scale**2)

    bsubu_even = np.zeros((ns, nt2, nzeta), dtype=acc_dtype)
    bsubu_odd = np.zeros((ns, nt2, nzeta), dtype=acc_dtype)
    bsubv_even = np.zeros((ns, nt2, nzeta), dtype=acc_dtype)
    bsubv_odd = np.zeros((ns, nt2, nzeta), dtype=acc_dtype)

    for js in range(ns):
        bu = bsubu[js, :nt2, :]
        bv = bsubv[js, :nt2, :]
        for m in range(mmax + 1):
            use_odd = (m % 2) == 1
            for n in range(nmax + 1):
                dnorm1 = base_dnorm
                if mnyq > 0 and m == mnyq:
                    dnorm1 *= 0.5
                if nnyq > 0 and n == nnyq and n != 0:
                    dnorm1 *= 0.5

                bsubumn1 = acc_dtype(0.0)
                bsubumn2 = acc_dtype(0.0)
                bsubvmn1 = acc_dtype(0.0)
                bsubvmn2 = acc_dtype(0.0)
                for k in range(nzeta):
                    for j in range(nt2):
                        tcosi1 = cosmui[j, m] * cosnv[k, n] * dnorm1
                        tcosi2 = sinmui[j, m] * sinnv[k, n] * dnorm1
                        val_u = bu[j, k]
                        val_v = bv[j, k]
                        bsubumn1 += tcosi1 * val_u
                        bsubumn2 += tcosi2 * val_u
                        bsubvmn1 += tcosi1 * val_v
                        bsubvmn2 += tcosi2 * val_v

                for k in range(nzeta):
                    for j in range(nt2):
                        tcos1 = cosmu[j, m] * cosnv[k, n]
                        tcos2 = sinmu[j, m] * sinnv[k, n]
                        ucontrib = tcos1 * bsubumn1 + tcos2 * bsubumn2
                        vcontrib = tcos1 * bsubvmn1 + tcos2 * bsubvmn2
                        if use_odd:
                            bsubu_odd[js, j, k] += ucontrib
                            bsubv_odd[js, j, k] += vcontrib
                        else:
                            bsubu_even[js, j, k] += ucontrib
                            bsubv_even[js, j, k] += vcontrib

    return (
        np.asarray(bsubu_even, dtype=float),
        np.asarray(bsubu_odd, dtype=float),
        np.asarray(bsubv_even, dtype=float),
        np.asarray(bsubv_odd, dtype=float),
    )



def bsubuv_parity_from_bcovar(
    *,
    bsubu_even: np.ndarray,
    bsubv_even: np.ndarray,
    s: np.ndarray,
    iequi: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct parity-separated bsubu/bsubv from bcovar even components."""
    s_full = np.asarray(s, dtype=float)
    psqrts = np.sqrt(np.maximum(s_full, 0.0))[:, None, None]
    pshalf = _pshalf_from_s(s_full)[:, None, None]
    scale = pshalf if int(iequi) == 1 else psqrts
    bsubu_even = np.asarray(bsubu_even, dtype=float)
    bsubv_even = np.asarray(bsubv_even, dtype=float)
    bsubu_odd = scale * bsubu_even
    bsubv_odd = scale * bsubv_even
    return bsubu_even, bsubu_odd, bsubv_even, bsubv_odd



__all__ = [
    "bsubuv_parity_from_bcovar",
    "bsubuv_parity_from_coeffs",
    "bsubuv_parity_from_realspace_jxbforce",
    "bsubuv_parity_from_state",
    "compute_bsubs_half_mesh",
]
