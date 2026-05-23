"""VMEC-style half-mesh metric + B-covariant ingredients.

This module ports the *core* algebra from VMEC2000's ``bcovar`` for
fixed-boundary parity work:

- Build half-mesh metric elements ``g_uu, g_uv, g_vv`` using VMEC's even/odd-m
  decomposition and half-mesh staggering.
- Build half-mesh Jacobian-related fields via :mod:`vmec_jax.vmec_jacobian`.
- Compute VMEC contravariant field components ``(B^u, B^v)`` and the covariant
  components ``(B_u, B_v)`` on the radial half mesh.
- Provide force-kernel inputs used by VMEC's ``forces`` routine.

The implementation here is intentionally limited to what's needed for validated
parity work. Use ``use_vmec_synthesis=True`` to switch the internal
R/Z/L synthesis to VMEC's symmetry-reduced theta grid (ntheta2/ntheta3) using
the ``fixaray`` trig tables.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from ._compat import jnp, tree_util
from .field import TWOPI
from .field import lamscale_from_phips
from .field import chips_from_wout_chipf
from .vmec_jacobian import VmecHalfMeshJacobian, jacobian_half_mesh_from_parity
from .fourier import eval_fourier, eval_fourier_dtheta, eval_fourier_dzeta_phys
from .grids import AngleGrid
from .vmec_residue import vmec_pwint_from_trig
from .vmec_parity import (
    ParityRZL,
    internal_odd_from_physical_vmec_jlam,
    internal_odd_from_physical_vmec_m1,
    split_rzl_even_odd_m,
    vmec_m1_internal_to_physical_signed,
)
from .vmec_realspace import (
    vmec_realspace_synthesis_multi,
)
from .vmec_tomnsp import VmecTrigTables, vmec_trig_tables
from .nyquist import nyquist_basis_from_wout


def _vmec_bcovar_profile_enabled() -> bool:
    value = os.environ.get("VMEC_JAX_PROFILE_BCOVAR", "")
    return value.strip().lower() not in ("", "0", "false", "no")


def _vmec_bcovar_profile_log(stage: str, start: float | None = None, **extra) -> None:
    if not _vmec_bcovar_profile_enabled():
        return
    payload = {"stage": stage}
    if start is not None:
        payload["elapsed_s"] = time.perf_counter() - start
    payload.update(extra)
    print(f"[vmec_jax bcovar] {payload}", flush=True)


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class VmecHalfMeshBcovar:
    """Half-mesh quantities used downstream by VMEC force/residue kernels."""

    jac: VmecHalfMeshJacobian

    # Half-mesh metric elements in cylindrical coordinates.
    guu: Any  # (ns, ntheta, nzeta)
    guv: Any  # (ns, ntheta, nzeta)
    gvv: Any  # (ns, ntheta, nzeta)

    # Half-mesh magnetic field components.
    bsupu: Any  # (ns, ntheta, nzeta)
    bsupv: Any  # (ns, ntheta, nzeta)
    bsubu: Any  # (ns, ntheta, nzeta)
    bsubv: Any  # (ns, ntheta, nzeta)
    bsubu_parity_even: Any  # (ns, ntheta, nzeta) half-mesh even-m channel
    bsubu_parity_odd: Any  # (ns, ntheta, nzeta) half-mesh odd-m internal channel
    bsubv_parity_even: Any  # (ns, ntheta, nzeta) half-mesh even-m channel
    bsubv_parity_odd: Any  # (ns, ntheta, nzeta) half-mesh odd-m internal channel

    # VMEC lambda force kernels (Fourier-space transform inputs) on full mesh.
    # These correspond to `bsubu_e/bsubv_e` in `bcovar.f` after the `-lamscale`
    # scaling, and are used as `(CLMN, BLMN)` in `tomnsps`.
    bsubu_e: Any  # (ns, ntheta, nzeta) unscaled full-mesh covariant B_u
    bsubv_e: Any  # (ns, ntheta, nzeta) unscaled full-mesh covariant B_v
    bsubu_e_scaled: Any  # (ns, ntheta, nzeta) scaled for tomnsps (VMEC bsubu_e)
    bsubv_e_scaled: Any  # (ns, ntheta, nzeta) scaled for tomnsps (VMEC bsubv_e)
    bsubu_tmp: Any  # (ns, ntheta, nzeta) pguv*bsupu term in bsubv_e
    bsubv_preblend: Any  # (ns, ntheta, nzeta) bsubv_e before blending
    bsubv_avg: Any  # (ns, ntheta, nzeta) averaged half-mesh bsubv
    clmn_even: Any  # (ns, ntheta, nzeta)
    clmn_odd: Any  # (ns, ntheta, nzeta)
    blmn_even: Any  # (ns, ntheta, nzeta)
    blmn_odd: Any  # (ns, ntheta, nzeta)

    # Lambda-force intermediate terms (full mesh), used for parity debugging.
    lu0_full: Any  # (ns, ntheta, nzeta) full-mesh LU even (scaled + wout.phipf)
    lu0_force: Any  # (ns, ntheta, nzeta) LU even used in lambda-force block
    lu1_full: Any  # (ns, ntheta, nzeta) full-mesh LU odd (scaled)
    lv0_full: Any  # (ns, ntheta, nzeta) full-mesh LV even (scaled)
    lv1_full: Any  # (ns, ntheta, nzeta) full-mesh LV odd (scaled)
    lvv: Any  # (ns, ntheta, nzeta) phipog * gvv (full mesh)
    lvv_sh: Any  # (ns, ntheta, nzeta) lvv * pshalf
    phip_full: Any  # (ns,) full-mesh phipf used in LU_even
    phip_internal: Any  # (ns,) VMEC internal phipf = signgs*phipf/(2π)

    # bsq = |B|^2/2 + p on half mesh (VMEC convention).
    bsq: Any  # (ns, ntheta, nzeta)

    # Force-kernel inputs (VMEC `bcovar` post-processing).
    gij_b_uu: Any  # (ns, ntheta, nzeta) = (B^u B^u) * sqrt(g)
    gij_b_uv: Any  # (ns, ntheta, nzeta) = (B^u B^v) * sqrt(g)
    gij_b_vv: Any  # (ns, ntheta, nzeta) = (B^v B^v) * sqrt(g)
    lu_e: Any  # (ns, ntheta, nzeta) = R * bsq
    lv_e: Any  # (ns, ntheta, nzeta) = (sqrt(g)/R) * bsq  (tau * bsq)

    # Lambda derivatives on half mesh (scaled-lambda).
    lam_u: Any  # (ns, ntheta, nzeta)
    lam_v: Any  # (ns, ntheta, nzeta)

    # Scalar lambda scaling factor.
    lamscale: Any

    def tree_flatten(self):
        children = (
            self.jac,
            self.guu,
            self.guv,
            self.gvv,
            self.bsupu,
            self.bsupv,
            self.bsubu,
            self.bsubv,
            self.bsubu_parity_even,
            self.bsubu_parity_odd,
            self.bsubv_parity_even,
            self.bsubv_parity_odd,
            self.bsubu_e,
            self.bsubv_e,
            self.bsubu_e_scaled,
            self.bsubv_e_scaled,
            self.bsubu_tmp,
            self.bsubv_preblend,
            self.bsubv_avg,
            self.clmn_even,
            self.clmn_odd,
            self.blmn_even,
            self.blmn_odd,
            self.lu0_full,
            self.lu0_force,
            self.lu1_full,
            self.lv0_full,
            self.lv1_full,
            self.lvv,
            self.lvv_sh,
            self.phip_full,
            self.phip_internal,
            self.bsq,
            self.gij_b_uu,
            self.gij_b_uv,
            self.gij_b_vv,
            self.lu_e,
            self.lv_e,
            self.lam_u,
            self.lam_v,
            self.lamscale,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def _pshalf_from_s(s: Any) -> Any:
    s = jnp.asarray(s)
    if s.shape[0] < 2:
        return jnp.sqrt(jnp.maximum(s, 0.0))
    sh = 0.5 * (s[1:] + s[:-1])
    p = jnp.concatenate([sh[:1], sh], axis=0)
    return jnp.sqrt(jnp.maximum(p, 0.0))


def _half_mesh_from_even_odd(even, odd_int, *, s):
    """VMEC half-mesh staggering for fields of form X = X_even + sqrt(s) X_odd."""
    even = jnp.asarray(even)
    odd_int = jnp.asarray(odd_int)
    s = jnp.asarray(s)

    ns = int(s.shape[0])
    if ns < 2:
        return even

    pshalf = _pshalf_from_s(s)[:, None, None]
    inner = 0.5 * (even[1:] + even[:-1] + pshalf[1:] * (odd_int[1:] + odd_int[:-1]))
    return jnp.concatenate([inner[:1], inner], axis=0)


def _replace_axis_row(a, row):
    a = jnp.asarray(a)
    if a.shape[0] == 0:
        return a
    return jnp.concatenate([jnp.asarray(row)[None, ...], a[1:]], axis=0)


def _replace_edge_row(a, row):
    a = jnp.asarray(a)
    if a.shape[0] == 0:
        return a
    return jnp.concatenate([a[:-1], jnp.asarray(row)[None, ...]], axis=0)


def _with_axis_zero(a):
    a = jnp.asarray(a)
    if a.shape[0] == 0:
        return a
    return jnp.concatenate([jnp.zeros_like(a[:1]), a[1:]], axis=0)


def _prepend_axis_zero(body, like):
    return jnp.concatenate([jnp.zeros_like(jnp.asarray(like)[:1]), body], axis=0)


def _avg_forward_half_to_int_or_zero(a):
    """Forward-average half mesh to integer mesh, matching zero output for ns < 2."""
    a = jnp.asarray(a)
    if a.shape[0] < 2:
        return jnp.zeros_like(a)
    body = 0.5 * (a[:-1] + a[1:])
    tail = 0.5 * a[-1:]
    return jnp.concatenate([body, tail], axis=0)


def _scale_lambda_full_mesh(a, lamscale):
    """Apply VMEC's lambda-force scaling while preserving the raw axis row."""
    a = jnp.asarray(a)
    if a.shape[0] < 2:
        return a
    return jnp.concatenate([a[:1], -lamscale * a[1:]], axis=0)


def _metric_even_odd(*, a0, a1, b0, b1, s):
    """Even/odd decomposition of (a0 + sqrt(s)a1)^2 + (b0 + sqrt(s)b1)^2."""
    s = jnp.asarray(s)
    ss = s[:, None, None]
    even = a0 * a0 + b0 * b0 + ss * (a1 * a1 + b1 * b1)
    odd = 2.0 * (a0 * a1 + b0 * b1)
    return even, odd


def _metric_cross_even_odd(*, a0, a1, b0, b1, s):
    """Even/odd decomposition of (a0 + sqrt(s)a1)(b0 + sqrt(s)b1)."""
    s = jnp.asarray(s)
    ss = s[:, None, None]
    even = a0 * b0 + ss * (a1 * b1)
    odd = a0 * b1 + a1 * b0
    return even, odd


def _apply_vmec_lambda_axis_closure(
    *,
    Lsin: Any,
    m_modes: Any,
    n_modes: Any,
    axis_copy_mask: np.ndarray | None = None,
    lthreed: bool,
    ntor: int,
) -> Any:
    """Mirror VMEC ``totzsp_mod`` axis closure for symmetric 3D lambda modes.

    VMEC sets ``lmncs(1,n,m=0) = lmncs(2,n,m=0)`` for ``n>0`` in symmetric 3D
    runs. In vmec_jax signed storage this corresponds to ``Lsin[js=0,m=0,n>0]``
    being copied from ``js=1`` for force evaluation.
    """
    Lsin = jnp.asarray(Lsin)
    ns = int(Lsin.shape[0])
    if (not bool(lthreed)) or int(ntor) <= 0 or ns < 2:
        return Lsin

    if axis_copy_mask is None:
        m_modes_np = np.asarray(m_modes, dtype=int)
        n_modes_np = np.asarray(n_modes, dtype=int)
        axis_copy_mask_np = (m_modes_np == 0) & (n_modes_np > 0)
    else:
        axis_copy_mask_np = np.asarray(axis_copy_mask, dtype=bool)
    if not np.any(axis_copy_mask_np):
        return Lsin

    axis_copy_mask = jnp.asarray(axis_copy_mask_np, dtype=Lsin.dtype)
    axis_row = jnp.where(axis_copy_mask != 0, Lsin[1, :], Lsin[0, :])
    return _replace_axis_row(Lsin, axis_row)


def vmec_bcovar_half_mesh_from_wout(
    *,
    state,
    static,
    wout,
    pres: Any | None = None,
    mass: Any | None = None,
    use_wout_bsup: bool = False,
    use_wout_bsub_for_lambda: bool = False,
    use_wout_bmag_for_bsq: bool = False,
    freeb_bsqvac_edge: Any | None = None,
    use_vmec_synthesis: bool = False,
    trig: VmecTrigTables | None = None,
    return_parity_aux: bool = False,
) -> VmecHalfMeshBcovar:
    """Compute VMEC-style half-mesh metric and B components for parity tests.

    Parameters
    ----------
    state:
        :class:`~vmec_jax.state.VMECState` (typically from :func:`~vmec_jax.wout.state_from_wout`).
    static:
        Static precomputations from :func:`~vmec_jax.static.build_static`.
    wout:
        :class:`~vmec_jax.wout.WoutData` providing ``phipf``, ``chipf``, ``phips``, ``signgs``.
    pres:
        Optional pressure profile on the *half mesh* in VMEC internal units (mu0*Pa).
        If omitted, uses ``wout.pres``.
    mass:
        Optional mass profile on the half mesh (VMEC internal units) used to
        reconstruct pressure as ``pres = mass / vp**gamma``. If provided (or
        present on ``wout`` as ``mass``), this takes precedence over ``pres``.
    use_vmec_synthesis:
        If True, evaluate the R/Z/L real-space parity pieces and their derivatives
        using VMEC's ``fixaray`` trig tables (ntheta2/ntheta3 grids).
    use_wout_bsub_for_lambda:
        If True, build the lambda-force full-mesh kernels (``blmn/clmn``) from
        ``wout``-stored ``bsub*`` fields averaged to the full radial mesh. This
        is a reference-parity mode that avoids re-deriving ``bsubv_e`` from
        lambda derivatives.
    use_wout_bmag_for_bsq:
        If True, use ``wout`` Nyquist ``|B|`` to form ``bsq = |B|^2/2 + p`` in
        this parity path, instead of deriving ``|B|^2`` from
        ``bsup/bsub`` products.
    freeb_bsqvac_edge:
        Optional free-boundary vacuum boundary ``0.5*|B|^2`` on the edge
        surface `(ntheta, nzeta)` (VMEC ``bsqvac`` convention). When provided,
        the half-mesh edge ``bsq`` is overridden as
        ``freeb_bsqvac_edge + p_edge``.
    trig:
        Optional precomputed VMEC trig tables. If omitted and
        ``use_vmec_synthesis=True``, they are built internally.
    """
    bcovar_start = time.perf_counter()
    s = jnp.asarray(static.s)
    if trig is None:
        trig = getattr(static, "trig_vmec", None)
    if trig is None:
        trig = vmec_trig_tables(
            ntheta=int(static.cfg.ntheta),
            nzeta=int(static.cfg.nzeta),
            nfp=int(getattr(wout, "nfp", static.cfg.nfp)),
            mmax=int(getattr(wout, "mpol", static.cfg.mpol)) - 1,
            nmax=int(getattr(wout, "ntor", static.cfg.ntor)),
            lasym=bool(getattr(wout, "lasym", static.cfg.lasym)),
            dtype=jnp.asarray(state.Rcos).dtype,
        )
    ns = int(s.shape[0])
    # VMEC stores internal coefficients. Undo the m=1 internal constraint for
    # R/Z before real-space synthesis.
    Rcos_int, Zsin_int, Rsin_int, Zcos_int = vmec_m1_internal_to_physical_signed(
        Rcos=state.Rcos,
        Zsin=state.Zsin,
        Rsin=state.Rsin,
        Zcos=state.Zcos,
        modes=static.modes,
        lthreed=bool(getattr(static.cfg, "lthreed", True)),
        lasym=bool(getattr(static.cfg, "lasym", False)),
        lconm1=bool(getattr(static.cfg, "lconm1", True)),
    )

    Rcos_geom = jnp.asarray(Rcos_int)
    Rsin_geom = jnp.asarray(Rsin_int)
    Zcos_geom = jnp.asarray(Zcos_int)
    Zsin_geom = jnp.asarray(Zsin_int)
    Lcos_force = jnp.asarray(state.Lcos)
    Lsin_force = _apply_vmec_lambda_axis_closure(
        Lsin=state.Lsin,
        m_modes=static.modes.m,
        n_modes=static.modes.n,
        axis_copy_mask=getattr(static, "lambda_axis_copy_mask", None),
        lthreed=bool(getattr(static.cfg, "lthreed", False)),
        ntor=int(getattr(static.cfg, "ntor", 0)),
    )
    Lsin_force = jnp.asarray(Lsin_force)

    state_parity = SimpleNamespace(
        Rcos=Rcos_geom,
        Rsin=Rsin_geom,
        Zcos=Zcos_geom,
        Zsin=Zsin_geom,
        Lcos=Lcos_force,
        Lsin=Lsin_force,
    )
    _vmec_bcovar_profile_log("setup_done", bcovar_start)

    parity_start = time.perf_counter()
    if use_vmec_synthesis:
        if trig is None:
            mmax = int(np.max(static.modes.m))
            nmax = int(np.max(np.abs(static.modes.n)))
            trig = vmec_trig_tables(
                ntheta=int(static.cfg.ntheta),
                nzeta=int(static.cfg.nzeta),
                nfp=int(wout.nfp),
                mmax=mmax,
                nmax=nmax,
                lasym=bool(wout.lasym),
                dtype=jnp.asarray(state.Rcos).dtype,
            )

        coeff_cos_stack = jnp.stack([state_parity.Rcos, state_parity.Zcos, state_parity.Lcos], axis=0)
        coeff_sin_stack = jnp.stack([state_parity.Rsin, state_parity.Zsin, state_parity.Lsin], axis=0)

        dtype = jnp.asarray(state.Rcos).dtype
        if getattr(static, "m_is_even", None) is None:
            m = np.asarray(static.modes.m, dtype=int)
            mask_even = jnp.asarray((m % 2) == 0).astype(dtype)
        else:
            mask_even = jnp.asarray(static.m_is_even, dtype=dtype)
        mask_odd = (1.0 - mask_even).astype(dtype)
        mask_stack = jnp.stack([mask_even, mask_odd], axis=0)
        coeff_cos = coeff_cos_stack[None, ...] * mask_stack[:, None, None, :]
        coeff_sin = coeff_sin_stack[None, ...] * mask_stack[:, None, None, :]
        even_odd, even_odd_t, even_odd_p = vmec_realspace_synthesis_multi(
            coeff_cos=coeff_cos,
            coeff_sin=coeff_sin,
            modes=static.modes,
            trig=trig,
            coeffs_internal=True,
            apply_scalxc=False,
            s=s,
            derivs=("base", "dtheta", "dzeta"),
        )

        even = even_odd[0]
        odd = even_odd[1]
        even_t = even_odd_t[0]
        odd_t = even_odd_t[1]
        even_p = even_odd_p[0]
        odd_p = even_odd_p[1]

        parity = ParityRZL(
            R_even=even[0],
            R_odd=odd[0],
            Z_even=even[1],
            Z_odd=odd[1],
            L_even=even[2],
            L_odd=odd[2],
            Rt_even=even_t[0],
            Rt_odd=odd_t[0],
            Zt_even=even_t[1],
            Zt_odd=odd_t[1],
            Lt_even=even_t[2],
            Lt_odd=odd_t[2],
            Rp_even=even_p[0],
            Rp_odd=odd_p[0],
            Zp_even=even_p[1],
            Zp_odd=odd_p[1],
            Lp_even=even_p[2],
            Lp_odd=odd_p[2],
        )
    else:
        # Split real-space fields into parity subsets. VMEC's bcovar pipeline
        # is built around even/odd-m parity (with odd-m stored in internal
        # form), even when LASYM is enabled. Using cos/sin splits here
        # breaks bsupu/bsupv parity for asymmetric runs.
        parity = split_rzl_even_odd_m(state_parity, static.basis, static.modes.m)
    _vmec_bcovar_profile_log("parity_done", parity_start)

    # VMEC axis convention (vmec_params.f: jmin1):
    # - m=1 odd-m internal fields are extrapolated to the axis (copy js=2),
    # - odd-m with m>=3 are zero on the axis.
    dtype = jnp.asarray(state.Rcos).dtype
    if getattr(static, "m_is_m1", None) is None:
        m_modes = np.asarray(static.modes.m, dtype=int)
        mask_m1 = jnp.asarray(m_modes == 1, dtype=dtype)
        mask_odd_rest = jnp.asarray((m_modes % 2 == 1) & (m_modes != 1), dtype=dtype)
        mask_odd = jnp.asarray(m_modes % 2 == 1, dtype=dtype)
    else:
        mask_m1 = jnp.asarray(static.m_is_m1, dtype=dtype)
        mask_odd_rest = jnp.asarray(static.m_is_odd_rest, dtype=dtype)
        mask_odd = jnp.asarray(static.m_is_odd, dtype=dtype)

    def _odd_internal_vmec(*, coeff_cos, coeff_sin, eval_fn, odd_is_internal: bool):
        if use_vmec_synthesis:
            phys_m1 = eval_fn(coeff_cos * mask_m1, coeff_sin * mask_m1)
            phys_rest = eval_fn(coeff_cos * mask_odd_rest, coeff_sin * mask_odd_rest)
        else:
            phys_m1 = eval_fn(coeff_cos * mask_m1, coeff_sin * mask_m1, static.basis, coeffs_internal=True)
            phys_rest = eval_fn(coeff_cos * mask_odd_rest, coeff_sin * mask_odd_rest, static.basis, coeffs_internal=True)
        if odd_is_internal:
            out = phys_m1 + phys_rest
            if out.shape[0] >= 2:
                out = _replace_axis_row(out, phys_m1[1])
            return out
        return internal_odd_from_physical_vmec_m1(odd_m1_phys=phys_m1, odd_mge2_phys=phys_rest, s=s)

    def _odd_internal_vmec_lambda(*, coeff_cos, coeff_sin, eval_fn, odd_is_internal: bool):
        if use_vmec_synthesis:
            phys_m1 = eval_fn(coeff_cos * mask_m1, coeff_sin * mask_m1)
            phys_rest = eval_fn(coeff_cos * mask_odd_rest, coeff_sin * mask_odd_rest)
        else:
            phys_m1 = eval_fn(coeff_cos * mask_m1, coeff_sin * mask_m1, static.basis, coeffs_internal=True)
            phys_rest = eval_fn(coeff_cos * mask_odd_rest, coeff_sin * mask_odd_rest, static.basis, coeffs_internal=True)
        if odd_is_internal:
            out = phys_m1 + phys_rest
            if out.shape[0] >= 2:
                out = _replace_axis_row(out, phys_m1[1])
            return out
        return internal_odd_from_physical_vmec_jlam(odd_m1_phys=phys_m1, odd_mge2_phys=phys_rest, s=s)

    odd_start = time.perf_counter()
    if use_vmec_synthesis:
        s_grid = s
        odd_is_internal = True

        def _odd_internal_from_phys(phys_m1, phys_rest, *, lambda_field: bool = False):
            if odd_is_internal:
                out = phys_m1 + phys_rest
                if out.shape[0] >= 2:
                    out = _replace_axis_row(out, phys_m1[1])
                return out
            if lambda_field:
                return internal_odd_from_physical_vmec_jlam(odd_m1_phys=phys_m1, odd_mge2_phys=phys_rest, s=s)
            return internal_odd_from_physical_vmec_m1(odd_m1_phys=phys_m1, odd_mge2_phys=phys_rest, s=s)

        coeff_cos_stack = jnp.stack([state_parity.Rcos, state_parity.Zcos, state_parity.Lcos], axis=0)
        coeff_sin_stack = jnp.stack([state_parity.Rsin, state_parity.Zsin, state_parity.Lsin], axis=0)
        mask_stack = jnp.stack([mask_m1, mask_odd_rest], axis=0)
        coeff_cos = coeff_cos_stack[None, ...] * mask_stack[:, None, None, :]
        coeff_sin = coeff_sin_stack[None, ...] * mask_stack[:, None, None, :]
        odd_base, odd_dtheta, odd_dzeta = vmec_realspace_synthesis_multi(
            coeff_cos=coeff_cos,
            coeff_sin=coeff_sin,
            modes=static.modes,
            trig=trig,
            coeffs_internal=True,
            apply_scalxc=True,
            s=s_grid,
            derivs=("base", "dtheta", "dzeta"),
        )

        R1 = _odd_internal_from_phys(odd_base[0, 0], odd_base[1, 0])
        Z1 = _odd_internal_from_phys(odd_base[0, 1], odd_base[1, 1])
        Ru1 = _odd_internal_from_phys(odd_dtheta[0, 0], odd_dtheta[1, 0])
        Zu1 = _odd_internal_from_phys(odd_dtheta[0, 1], odd_dtheta[1, 1])
        Rv1 = _odd_internal_from_phys(odd_dzeta[0, 0], odd_dzeta[1, 0])
        Zv1 = _odd_internal_from_phys(odd_dzeta[0, 1], odd_dzeta[1, 1])
        Lu1 = _odd_internal_from_phys(odd_dtheta[0, 2], odd_dtheta[1, 2], lambda_field=True)
        Lv1 = _odd_internal_from_phys(odd_dzeta[0, 2], odd_dzeta[1, 2], lambda_field=True)
    else:
        odd_is_internal = False
        R1 = _odd_internal_vmec(
            coeff_cos=state_parity.Rcos,
            coeff_sin=state_parity.Rsin,
            eval_fn=eval_fourier,
            odd_is_internal=odd_is_internal,
        )
        Z1 = _odd_internal_vmec(
            coeff_cos=state_parity.Zcos,
            coeff_sin=state_parity.Zsin,
            eval_fn=eval_fourier,
            odd_is_internal=odd_is_internal,
        )
        Ru1 = _odd_internal_vmec(
            coeff_cos=state_parity.Rcos,
            coeff_sin=state_parity.Rsin,
            eval_fn=eval_fourier_dtheta,
            odd_is_internal=odd_is_internal,
        )
        Zu1 = _odd_internal_vmec(
            coeff_cos=state_parity.Zcos,
            coeff_sin=state_parity.Zsin,
            eval_fn=eval_fourier_dtheta,
            odd_is_internal=odd_is_internal,
        )
        Rv1 = _odd_internal_vmec(
            coeff_cos=state_parity.Rcos,
            coeff_sin=state_parity.Rsin,
            eval_fn=eval_fourier_dzeta_phys,
            odd_is_internal=odd_is_internal,
        )
        Zv1 = _odd_internal_vmec(
            coeff_cos=state_parity.Zcos,
            coeff_sin=state_parity.Zsin,
            eval_fn=eval_fourier_dzeta_phys,
            odd_is_internal=odd_is_internal,
        )

        Lu1 = _odd_internal_vmec_lambda(
            coeff_cos=state_parity.Lcos,
            coeff_sin=state_parity.Lsin,
            eval_fn=eval_fourier_dtheta,
            odd_is_internal=odd_is_internal,
        )
        Lv1 = _odd_internal_vmec_lambda(
            coeff_cos=state_parity.Lcos,
            coeff_sin=state_parity.Lsin,
            eval_fn=eval_fourier_dzeta_phys,
            odd_is_internal=odd_is_internal,
        )
    _vmec_bcovar_profile_log("odd_channels_done", odd_start)

    # Half-mesh Jacobian quantities from VMEC's discrete formula.
    metric_start = time.perf_counter()
    jac = jacobian_half_mesh_from_parity(
        pr1_even=parity.R_even,
        pr1_odd=R1,
        pz1_even=parity.Z_even,
        pz1_odd=Z1,
        pru_even=parity.Rt_even,
        pru_odd=Ru1,
        pzu_even=parity.Zt_even,
        pzu_odd=Zu1,
        s=s,
    )
    # Metric elements on full mesh split into even/odd (internal) pieces, then
    # staggered to the half mesh using VMEC's pshalf convention.
    guu_e, guu_o = _metric_even_odd(a0=parity.Rt_even, a1=Ru1, b0=parity.Zt_even, b1=Zu1, s=s)
    guv_e, guv_o = _metric_cross_even_odd(a0=parity.Rt_even, a1=Ru1, b0=parity.Rp_even, b1=Rv1, s=s)
    guv_e2, guv_o2 = _metric_cross_even_odd(a0=parity.Zt_even, a1=Zu1, b0=parity.Zp_even, b1=Zv1, s=s)
    guv_e = guv_e + guv_e2
    guv_o = guv_o + guv_o2
    gvv_e, gvv_o = _metric_even_odd(a0=parity.Rp_even, a1=Rv1, b0=parity.Zp_even, b1=Zv1, s=s)

    # R^2 term in cylindrical metric: gvv <- gvv + R^2
    ss = s[:, None, None]
    R2_e = parity.R_even * parity.R_even + ss * (R1 * R1)
    R2_o = 2.0 * parity.R_even * R1

    gvv_e_total = gvv_e + R2_e
    gvv_o_total = gvv_o + R2_o

    guu = _half_mesh_from_even_odd(guu_e, guu_o, s=s)
    guv = _half_mesh_from_even_odd(guv_e, guv_o, s=s)
    gvv = _half_mesh_from_even_odd(gvv_e_total, gvv_o_total, s=s)
    if ns >= 1:
        guv = _with_axis_zero(guv)
        gvv = _with_axis_zero(gvv)

    def _half_even_odd_full(e_full, o_full):
        if ns < 2:
            z = jnp.zeros_like(guu)
            return z, z
        e_body = 0.5 * (e_full[1:] + e_full[:-1])
        o_body = 0.5 * (o_full[1:] + o_full[:-1])
        return (
            jnp.concatenate([e_body[:1], e_body], axis=0),
            jnp.concatenate([o_body[:1], o_body], axis=0),
        )

    guu_eh, guu_oh = _half_even_odd_full(guu_e, guu_o)
    guv_eh, guv_oh = _half_even_odd_full(guv_e, guv_o)
    gvv_eh, gvv_oh = _half_even_odd_full(gvv_e_total, gvv_o_total)

    # Lambda derivatives on half mesh (scaled lambda).
    lam_u = _half_mesh_from_even_odd(parity.Lt_even, Lu1, s=s)
    lam_v = _half_mesh_from_even_odd(parity.Lp_even, Lv1, s=s)

    # NOTE: `lam_u/lam_v` are used downstream in VMEC's magnetic-field and
    # force pipeline, but the (guu,guv,gvv) metric elements used for bsub*
    # parity are those constructed above from (Ru,Zu,Rv,Zv) and the cylindrical
    # +R^2 term.
    _vmec_bcovar_profile_log("metric_done", metric_start)

    # ---------------------------------------------------------------------
    # Contravariant B components (bsupu, bsupv) on the half mesh.
    # ---------------------------------------------------------------------
    field_start = time.perf_counter()
    # VMEC does not form bsupu/bsupv from pointwise (sqrtg, lam_u, lam_v) alone.
    # Instead, it:
    #   1) builds full-mesh LU = d(lambda)/du and LV = -d(lambda)/dv (even/odd-m),
    #   2) scales LU/LV by lamscale and adds phipf to LU_even,
    #   3) averages (LU,LV) from full -> half radial mesh using pshalf,
    #   4) adds the full-mesh flux function chips(js) via `add_fluxes`.
    #
    # See `VMEC2000/Sources/General/bcovar.f` and `add_fluxes.f90`.
    lamscale = lamscale_from_phips(wout.phips, s)
    signgs = int(getattr(wout, "signgs", 1))

    # VMEC adds the **full-mesh** flux function `chips(js)` to bsupu in
    # `add_fluxes`, while `wout` commonly stores the half-mesh array `chipf`.
    chipf_out = getattr(wout, "chipf", None)
    phipf_cached = getattr(wout, "phipf_internal", None)
    chipf_cached = getattr(wout, "chipf_internal", None)
    chips_cached = getattr(wout, "chips_eff", None)
    if phipf_cached is None:
        phipf_out = jnp.asarray(getattr(wout, "phipf"))
        signgs = int(getattr(wout, "signgs", 1))
        flux_is_internal = bool(getattr(wout, "flux_is_internal", False))
        if not flux_is_internal:
            scale = jnp.asarray(TWOPI, dtype=phipf_out.dtype) * jnp.asarray(signgs, dtype=phipf_out.dtype)
            phipf_internal = phipf_out / scale
            chipf_internal = None if chipf_out is None else (jnp.asarray(chipf_out) / scale)
        else:
            phipf_internal = phipf_out
            chipf_internal = None if chipf_out is None else jnp.asarray(chipf_out)
    else:
        phipf_internal = jnp.asarray(phipf_cached)
        chipf_internal = None if chipf_cached is None else jnp.asarray(chipf_cached)
    if chips_cached is not None:
        chips_eff = jnp.asarray(chips_cached)
    elif chipf_out is not None:
        chips_eff = chips_from_wout_chipf(
            chipf=chipf_internal,
            phipf=phipf_internal,
            iotaf=getattr(wout, "iotaf", None),
            iotas=getattr(wout, "iotas", None),
            # Solver-internal wout-like objects may omit iotaf/iotas and provide
            # half-mesh chipf; keep VMEC2000-compatible behavior in that case.
            assume_half_if_unknown=True,
        )
    else:
        chips_eff = jnp.asarray(getattr(wout, "iotaf", getattr(wout, "iotas", 0.0))) * jnp.asarray(phipf_internal)
    ncurr = int(getattr(wout, "ncurr", 0))
    lcurrent = bool(getattr(wout, "lcurrent", True))
    icurv = jnp.asarray(getattr(wout, "icurv", jnp.zeros((ns,), dtype=phipf_internal.dtype)))
    if int(icurv.shape[0]) != ns:
        icurv = jnp.zeros((ns,), dtype=phipf_internal.dtype)

    # VMEC bcovar: overg = 1 / sqrtg (phipog in bcovar.f).
    denom = jac.sqrtg
    # Avoid NaNs in autodiff by preventing division-by-zero in the inactive branch.
    safe_denom = jnp.where(denom != 0, denom, jnp.asarray(1.0, dtype=denom.dtype))
    overg = jnp.where(denom != 0, 1.0 / safe_denom, 0.0)

    # Full-mesh LU = d(lambda)/du and LV = -d(lambda)/dv in VMEC conventions.
    lu0_full = jnp.asarray(parity.Lt_even)
    lu1_full = jnp.asarray(Lu1)
    lv0_full = -jnp.asarray(parity.Lp_even)
    lv1_full = -jnp.asarray(Lv1)

    # Scale by lamscale and add wout phipf to LU_even (bsupv path).
    lu0_full = (lamscale * lu0_full) + jnp.asarray(phipf_internal)[:, None, None]
    lu1_full = lamscale * lu1_full
    lv0_full = lamscale * lv0_full
    lv1_full = lamscale * lv1_full

    pshalf = _pshalf_from_s(s)[:, None, None]

    # Radial full->half average (Fortran: for l=2..ns).
    bsupu_even = jnp.zeros_like(jac.sqrtg)
    bsupu_odd = jnp.zeros_like(jac.sqrtg)
    bsupv_even = jnp.zeros_like(jac.sqrtg)
    bsupv_odd = jnp.zeros_like(jac.sqrtg)
    bsupu = jnp.zeros_like(jac.sqrtg)
    bsupv = jnp.zeros_like(jac.sqrtg)
    if ns >= 2:
        avg_lu0 = lu0_full[1:] + lu0_full[:-1]
        avg_lu1 = lu1_full[1:] + lu1_full[:-1]
        avg_lv0 = lv0_full[1:] + lv0_full[:-1]
        avg_lv1 = lv1_full[1:] + lv1_full[:-1]

        bsupv_even = _prepend_axis_zero(0.5 * overg[1:] * avg_lu0, jac.sqrtg)
        bsupv_odd = _prepend_axis_zero(0.5 * overg[1:] * avg_lu1, jac.sqrtg)
        bsupu_even = _prepend_axis_zero(0.5 * overg[1:] * avg_lv0, jac.sqrtg)
        bsupu_odd = _prepend_axis_zero(0.5 * overg[1:] * avg_lv1, jac.sqrtg)

        bsupv = bsupv_even + pshalf * bsupv_odd
        bsupu = bsupu_even + pshalf * bsupu_odd

    if (ncurr == 1) and lcurrent and (ns >= 2):
        # VMEC's add_fluxes computes chips using bsup in VMEC orientation.
        pwint = vmec_pwint_from_trig(trig, ns=int(overg.shape[0]), nzeta=int(overg.shape[2])).astype(bsupu.dtype)
        if pwint.shape[1:] != bsupu.shape[1:]:
            # The non-VMEC-synthesis path uses the full theta grid instead of
            # VMEC's reduced ntheta3 grid. Fall back to uniform surface-average
            # weights on that active grid so the current-driven chips update can
            # run on the same angular discretization as guu/bsupu/bsupv.
            dnorm3 = jnp.asarray(getattr(trig, "dnorm3", 0.0), dtype=bsupu.dtype)
            pwint = jnp.broadcast_to(dnorm3, bsupu.shape)
            pwint = _with_axis_zero(pwint)

        top = jnp.asarray(icurv, dtype=bsupu.dtype) - jnp.sum(
            pwint * ((guu * bsupu) + (guv * bsupv)),
            axis=(1, 2),
        )
        bot = jnp.sum(pwint * (overg * guu), axis=(1, 2))

        chips_dyn = jnp.asarray(chips_eff, dtype=bsupu.dtype)
        safe_bot = jnp.where(bot != 0.0, bot, jnp.asarray(1.0, dtype=bot.dtype))
        chips_new = jnp.where(bot != 0.0, top / safe_bot, chips_dyn)
        chips_dyn = jnp.concatenate([jnp.zeros_like(chips_dyn[:1]), chips_new[1:]], axis=0)
        chips_eff = chips_dyn

    # `add_fluxes`: VMEC updates `bsupu` in VMEC orientation:
    #   bsupu <- bsupu + chips*overg.
    chip_term = jnp.asarray(chips_eff)[:, None, None] * overg
    bsupu_even = bsupu_even + chip_term
    bsupu = bsupu + chip_term

    basis_nyq = None
    grid_nyq = None

    def _nyq_grid():
        nonlocal grid_nyq
        if grid_nyq is None:
            if int(getattr(static.grid, "nfp", 0)) == int(wout.nfp):
                grid_nyq = static.grid
            else:
                grid_nyq = AngleGrid(theta=static.grid.theta, zeta=static.grid.zeta, nfp=wout.nfp)
        return grid_nyq
    if bool(use_wout_bsup):
        # Replace with wout-stored Nyquist bsup (reference parity path).
        # Nyquist `wout` field coefficients (`bsup*`, `bsub*`, `bm*`) follow the
        # output transform conventions from `wrout` and are most consistent with
        # direct Fourier evaluation on the active angular grid.
        #
        # Using VMEC synthesis tables for these reference fields introduces a
        # small but systematic mismatch in the parity path for nfp>1 cases.
        basis_nyq = nyquist_basis_from_wout(wout=wout, grid=_nyq_grid())
        bsupu = jnp.asarray(eval_fourier(wout.bsupumnc, wout.bsupumns, basis_nyq))
        bsupv = jnp.asarray(eval_fourier(wout.bsupvmnc, wout.bsupvmns, basis_nyq))
        bsupu_even = bsupu
        bsupv_even = bsupv
        bsupu_odd = jnp.zeros_like(bsupu)
        bsupv_odd = jnp.zeros_like(bsupv)

    # VMEC enforces axis bsup*=0 explicitly.
    if ns >= 1:
        bsupu = _with_axis_zero(bsupu)
        bsupv = _with_axis_zero(bsupv)
        bsupu_even = _with_axis_zero(bsupu_even)
        bsupv_even = _with_axis_zero(bsupv_even)
        bsupu_odd = _with_axis_zero(bsupu_odd)
        bsupv_odd = _with_axis_zero(bsupv_odd)

    bsubu = guu * bsupu + guv * bsupv
    bsubv = guv * bsupu + gvv * bsupv
    s_half = pshalf * pshalf
    bsubu_parity_even = (
        guu_eh * bsupu_even
        + s_half * guu_oh * bsupu_odd
        + guv_eh * bsupv_even
        + s_half * guv_oh * bsupv_odd
    )
    bsubu_parity_odd = (
        guu_eh * bsupu_odd
        + guu_oh * bsupu_even
        + guv_eh * bsupv_odd
        + guv_oh * bsupv_even
    )
    bsubv_parity_even = (
        guv_eh * bsupu_even
        + s_half * guv_oh * bsupu_odd
        + gvv_eh * bsupv_even
        + s_half * gvv_oh * bsupv_odd
    )
    bsubv_parity_odd = (
        guv_eh * bsupu_odd
        + guv_oh * bsupu_even
        + gvv_eh * bsupv_odd
        + gvv_oh * bsupv_even
    )

    # Optional reference parity path for lambda-force kernels.
    bsubu_lambda = bsubu
    bsubv_lambda = bsubv
    # VMEC internal phipf corresponds to dPhi/(2π) and includes signgs.
    # `flux_profiles_from_indata` already constructs phipf in that convention,
    # so use wout.phipf directly for the lambda-force block.
    phip_internal = jnp.asarray(phipf_internal)
    lu0_force = (lamscale * jnp.asarray(parity.Lt_even)) + phip_internal[:, None, None]

    if bool(use_wout_bsub_for_lambda):
        basis_nyq = nyquist_basis_from_wout(wout=wout, grid=_nyq_grid())
        bsubu_lambda = jnp.asarray(eval_fourier(wout.bsubumnc, wout.bsubumns, basis_nyq))
        bsubv_lambda = jnp.asarray(eval_fourier(wout.bsubvmnc, wout.bsubvmns, basis_nyq))

    b2 = bsupu * bsubu + bsupv * bsubv
    if bool(use_wout_bmag_for_bsq):
        if basis_nyq is None:
            basis_nyq = nyquist_basis_from_wout(wout=wout, grid=_nyq_grid())
        bmag_ref = jnp.asarray(eval_fourier(wout.bmnc, wout.bmns, basis_nyq))
        b2 = bmag_ref * bmag_ref
    pres_h = None
    mass_in = None
    try:
        mass_in = getattr(wout, "mass", None)
    except Exception:
        mass_in = None
    if mass is not None:
        mass_in = mass
    gamma = None
    try:
        gamma = float(getattr(wout, "gamma"))
    except Exception:
        gamma = None
    if mass_in is not None and gamma is not None and trig is not None:
        try:
            sqrtg = jnp.asarray(jac.sqrtg)
            nzeta = int(sqrtg.shape[2])
            pwint = vmec_pwint_from_trig(trig, ns=int(ns), nzeta=int(nzeta))
            signgs = int(getattr(wout, "signgs", 1))
            jac_s = jnp.asarray(float(signgs), dtype=sqrtg.dtype) * sqrtg
            vp = jnp.sum(pwint * jac_s, axis=(1, 2))
            mass_in = jnp.asarray(mass_in, dtype=vp.dtype)
            # Axis value is treated as zero in VMEC (pwint masks js=1).
            safe_vp = jnp.where(vp != 0.0, vp, jnp.asarray(1.0, dtype=vp.dtype))
            pres_1d = jnp.where(
                vp != 0.0,
                mass_in / (safe_vp**gamma),
                jnp.asarray(0.0, dtype=vp.dtype),
            )
            pres_h = pres_1d[:, None, None]
        except Exception:
            pres_h = None
    if pres_h is None:
        pres_h = jnp.asarray(wout.pres if pres is None else pres)[:, None, None]
    bsq = 0.5 * b2 + pres_h
    if freeb_bsqvac_edge is not None:
        vac_edge = jnp.asarray(freeb_bsqvac_edge, dtype=bsq.dtype)
        if vac_edge.ndim != 2:
            raise ValueError(
                f"freeb_bsqvac_edge must have shape (ntheta,nzeta), got {vac_edge.shape}"
            )
        if vac_edge.shape != bsq.shape[1:]:
            raise ValueError(
                "freeb_bsqvac_edge shape mismatch: "
                f"expected {bsq.shape[1:]}, got {vac_edge.shape}"
            )
        bsq_edge = vac_edge + pres_h[-1]
        bsq = _replace_edge_row(bsq, bsq_edge)

    # Force-kernel inputs matching what `forces.f` expects after `bcovar`.
    gij_b_uu = (bsupu * bsupu) * jac.sqrtg
    gij_b_uv = (bsupu * bsupv) * jac.sqrtg
    gij_b_vv = (bsupv * bsupv) * jac.sqrtg
    lu_e = bsq * jac.r12
    lv_e = bsq * jac.tau
    _vmec_bcovar_profile_log("field_done", field_start)

    # ---------------------------------------------------------------------
    # Lambda force kernels (bcovar.f "lambda full mesh forces" block)
    # ---------------------------------------------------------------------
    lambda_start = time.perf_counter()
    # This reproduces the structure in `bcovar.f`:
    #   - compute an intermediate bsubv_e on the full radial mesh from LU and metrics
    #   - average (bsubuh,bsubvh) from the half mesh onto the full mesh
    #   - blend bsubv_e with averaged bsubvh using bdamp(s) for near-axis stability
    #
    # Inputs:
    # - LU (full mesh, parity-split): LU = phipf + lamscale*dλ/du
    # - lvv (half mesh): lvv = g_vv / sqrtg (bcovar.f phipog * gvv)
    # - bsubu/bsubv (half mesh): covariant B components

    # Full-mesh LU parity pieces (odd is VMEC-internal 1/sqrt(s) representation).
    # VMEC uses the internal phipf (= signgs*phipf/(2π)) in the LU definition.
    lu0 = (lamscale * parity.Lt_even) + jnp.asarray(phipf_internal)[:, None, None]
    lu1 = lamscale * Lu1

    # lvv on half mesh: phipog * gvv (bcovar.f uses phipog == 1/sqrtg).
    # NOTE: phipog does **not** include the 2π scaling used in `overg`.
    phipog_safe = jnp.where(
        jac.sqrtg != 0,
        jac.sqrtg,
        jnp.asarray(1.0, dtype=jac.sqrtg.dtype),
    )
    phipog = jnp.where(jac.sqrtg != 0, 1.0 / phipog_safe, 0.0)
    if ns >= 1:
        phipog = _with_axis_zero(phipog)
    lvv = phipog * gvv

    if bool(use_wout_bsub_for_lambda):
        # Reference parity mode: use averaged wout bsub* directly.
        bsubu_tmp = jnp.zeros_like(bsubu_lambda)
        bsubv_preblend = jnp.zeros_like(bsubv_lambda)
        bsubu_e = _avg_forward_half_to_int_or_zero(bsubu_lambda)
        bsubv_e = _avg_forward_half_to_int_or_zero(bsubv_lambda)
        bsubv_avg = bsubv_e
    else:
        # Intermediate full-mesh bsubv_e (before blending), following bcovar.f.
        lvv_sh = lvv * pshalf
        bsubu_tmp = guv * bsupu  # bcovar: pguv*bsupu (sigma_an=1 isotropic)
        if ns >= 2:
            bsubv_base = jnp.concatenate(
                [
                    0.5 * (lvv[:-1] + lvv[1:]) * lu0_force[:-1],
                    0.5 * lvv[-1:] * lu0_force[-1:],
                ],
                axis=0,
            )
            bsubv_extra = jnp.concatenate(
                [
                    0.5 * ((lvv_sh[:-1] + lvv_sh[1:]) * lu1[:-1] + bsubu_tmp[:-1] + bsubu_tmp[1:]),
                    0.5 * (lvv_sh[-1:] * lu1[-1:] + bsubu_tmp[-1:]),
                ],
                axis=0,
            )
            bsubv_e = bsubv_base + bsubv_extra
        else:
            bsubv_e = jnp.zeros_like(bsubv)
        bsubv_preblend = bsubv_e

        # Average lambda forces onto full radial mesh (bsubu_e from bsubu half mesh).
        bsubu_e = _avg_forward_half_to_int_or_zero(bsubu)

        # Blend bsubv_e with half-mesh bsubv average using bdamp(s) (VMEC: bdamp=2*pdamp*(1-s)).
        pdamp = 0.05
        bdamp = (2.0 * pdamp * (1.0 - s)).astype(jnp.asarray(bsubv_e).dtype)[:, None, None]
        if ns >= 2:
            bsubv_avg = _avg_forward_half_to_int_or_zero(bsubv)
            bsubv_e = bdamp * bsubv_e + (1.0 - bdamp) * bsubv_avg
        else:
            bsubv_e = bdamp * bsubv_e + (1.0 - bdamp) * bsubv_e
            bsubv_avg = bsubv_e

    # Final scaling for tomnsps:
    # VMEC applies the "-lamscale" factor only for js>=2 (1-based). The axis (js=1)
    # is excluded so the lambda-force kernels do not introduce spurious constant
    # contributions from the copied/extrapolated half-mesh axis values.
    #
    # VMEC also exposes odd-m pieces as sqrt(s)*bsub*_e.
    psqrts = jnp.sqrt(jnp.maximum(s, 0.0))[:, None, None]
    clmn_even = _scale_lambda_full_mesh(bsubu_e, lamscale)
    blmn_even = _scale_lambda_full_mesh(bsubv_e, lamscale)
    clmn_odd = psqrts * clmn_even
    blmn_odd = psqrts * blmn_even

    bsubu_e_scaled = clmn_even
    bsubv_e_scaled = blmn_even

    _vmec_bcovar_profile_log("lambda_done", lambda_start)

    result = VmecHalfMeshBcovar(
        jac=jac,
        guu=guu,
        guv=guv,
        gvv=gvv,
        bsupu=bsupu,
        bsupv=bsupv,
        bsubu=bsubu,
        bsubv=bsubv,
        bsubu_parity_even=bsubu_parity_even,
        bsubu_parity_odd=bsubu_parity_odd,
        bsubv_parity_even=bsubv_parity_even,
        bsubv_parity_odd=bsubv_parity_odd,
        bsubu_e=bsubu_e,
        bsubv_e=bsubv_e,
        bsubu_e_scaled=bsubu_e_scaled,
        bsubv_e_scaled=bsubv_e_scaled,
        bsubu_tmp=bsubu_tmp,
        bsubv_preblend=bsubv_preblend,
        bsubv_avg=bsubv_avg,
        lu0_full=lu0_full,
        lu0_force=lu0_force,
        lu1_full=lu1_full,
        lv0_full=lv0_full,
        lv1_full=lv1_full,
        lvv=lvv,
        lvv_sh=lvv * pshalf,
        phip_full=jnp.asarray(wout.phipf),
        phip_internal=phip_internal,
        bsq=bsq,
        gij_b_uu=gij_b_uu,
        gij_b_uv=gij_b_uv,
        gij_b_vv=gij_b_vv,
        lu_e=lu_e,
        lv_e=lv_e,
        lam_u=lam_u,
        lam_v=lam_v,
        lamscale=lamscale,
        clmn_even=clmn_even,
        clmn_odd=clmn_odd,
        blmn_even=blmn_even,
        blmn_odd=blmn_odd,
    )
    _vmec_bcovar_profile_log("bcovar_done", bcovar_start)
    if bool(return_parity_aux):
        aux = SimpleNamespace(
            pr1_even=parity.R_even,
            pr1_odd=R1,
            pz1_even=parity.Z_even,
            pz1_odd=Z1,
            pru_even=parity.Rt_even,
            pru_odd=Ru1,
            pzu_even=parity.Zt_even,
            pzu_odd=Zu1,
            prv_even=parity.Rp_even,
            prv_odd=Rv1,
            pzv_even=parity.Zp_even,
            pzv_odd=Zv1,
            lu_odd=Lu1,
            lv_odd=Lv1,
        )
        return result, aux
    return result
