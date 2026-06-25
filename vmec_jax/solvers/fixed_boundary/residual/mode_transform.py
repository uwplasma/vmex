"""Mode-transform setup helpers for the VMEC residual iteration loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...._compat import has_jax, jax, jnp
from .geometry import _mn_sin_to_signed_physical_batch, _rz_norm_np

__all__ = [
    "ModeTransformContext",
    "ModeTransformHostProjection",
    "build_mode_transform_context",
    "build_mode_transform_host_projection",
    "mn_cos_to_signed_host_projected",
    "mn_sin_to_signed_host_projected",
    "mode_diag_weights_mn",
    "mode_diag_weights_mn_np",
    "vmec_scalxc_from_s_np",
]


@dataclass(frozen=True)
class ModeTransformHostProjection:
    """Precomputed DGEMM projection matrices for host mode transforms."""

    ncoeff: int
    n_half: int
    A_cos: np.ndarray | None
    A_sin: np.ndarray | None
    AB_cos: np.ndarray | None
    AB_sin: np.ndarray | None


@dataclass(frozen=True)
class ModeTransformContext:
    """Signed-mode transform state shared by scan and non-scan residual loops.

    VMEC stores R/Z/lambda updates in compact ``(m, n>=0)`` blocks but applies
    updates and diagnostics in signed-mode coefficient vectors.  Keeping all
    signed-index arrays, host projections, physical ``scalxc`` factors, and
    mode-diagonal weights in one context keeps the residual driver focused on
    controller policy instead of mode bookkeeping.
    """

    mpol: int
    ntor: int
    nrange: int
    nfp: float
    ncoeff: int
    signed_maps: Any
    idx_pos: np.ndarray
    idx_neg: np.ndarray
    m_idx_np: np.ndarray
    n_idx_np: np.ndarray
    kp_idx_np: np.ndarray
    kn_idx_np: np.ndarray
    has_kn_np: np.ndarray
    m_idx: Any
    n_idx: Any
    kp_idx: Any
    kn_idx: Any
    has_kn: Any
    include_rcc_np: np.ndarray
    host_projection: ModeTransformHostProjection
    scalxc_mn: Any
    scalxc_mn_np: np.ndarray | None
    w_mode_mn: Any
    w_mode_mn_np: np.ndarray | None
    state0_dtype: Any
    m0_mask: np.ndarray
    m0: Any
    n0: Any
    lthreed: bool
    lasym: bool
    host_update_assembly: bool

    def mn_cos_to_signed_host(self, cc: Any, ss: Any) -> np.ndarray:
        """Host DGEMM projection for cosine-like VMEC mode blocks."""

        return mn_cos_to_signed_host_projected(cc, ss, self.host_projection)

    def mn_sin_to_signed_host(self, sc: Any, cs: Any) -> np.ndarray:
        """Host DGEMM projection for sine-like VMEC mode blocks."""

        return mn_sin_to_signed_host_projected(sc, cs, self.host_projection)

    def mn_cos_to_signed(self, cc: Any, ss: Any) -> Any:
        """Map cosine-like ``(m,n>=0)`` blocks to signed coefficients."""

        if self.host_update_assembly:
            return self.mn_cos_to_signed_host(cc, ss)
        from vmec_jax.vmec_parity import _mn_cos_to_signed_cached

        cc = jnp.asarray(cc)
        ss = jnp.asarray(ss) if ss is not None else jnp.zeros_like(cc)
        return _mn_cos_to_signed_cached(cc, ss, maps=self.signed_maps, ncoeff=self.ncoeff)

    def mn_sin_to_signed(self, sc: Any, cs: Any) -> Any:
        """Map sine-like ``(m,n>=0)`` blocks to signed coefficients."""

        if self.host_update_assembly:
            return self.mn_sin_to_signed_host(sc, cs)
        from vmec_jax.vmec_parity import _mn_sin_to_signed_cached

        sc = jnp.asarray(sc)
        cs = jnp.asarray(cs) if cs is not None else jnp.zeros_like(sc)
        return _mn_sin_to_signed_cached(sc, cs, maps=self.signed_maps, ncoeff=self.ncoeff)

    def mn_sin_to_signed_batch(self, sc: Any, cs: Any) -> Any:
        """Batched sine-like transform for physical lambda helper paths."""

        from vmec_jax.vmec_parity import _mn_sin_to_signed_cached

        sc = jnp.asarray(sc)
        cs = jnp.asarray(cs) if cs is not None else jnp.zeros_like(sc)
        if has_jax():
            return jax.vmap(
                lambda sc_i, cs_i: _mn_sin_to_signed_cached(
                    sc_i,
                    cs_i,
                    maps=self.signed_maps,
                    ncoeff=self.ncoeff,
                )
            )(sc, cs)
        out = [
            _mn_sin_to_signed_cached(sc[i], cs[i], maps=self.signed_maps, ncoeff=self.ncoeff)
            for i in range(int(sc.shape[0]))
        ]
        return jnp.stack(out, axis=0)

    def _scale_physical_cos(self, cc: Any, ss: Any) -> tuple[Any, Any]:
        if self.host_update_assembly:
            cc = np.asarray(cc, dtype=float) / self.scalxc_mn_np
            ss = np.asarray(ss, dtype=float) / self.scalxc_mn_np if ss is not None else None
        else:
            cc = jnp.asarray(cc) / self.scalxc_mn
            ss = jnp.asarray(ss) / self.scalxc_mn if ss is not None else None
        return cc, ss

    def _scale_physical_sin(self, sc: Any, cs: Any) -> tuple[Any, Any]:
        if self.host_update_assembly:
            sc = np.asarray(sc, dtype=float) / self.scalxc_mn_np
            cs = np.asarray(cs, dtype=float) / self.scalxc_mn_np if cs is not None else None
        else:
            sc = jnp.asarray(sc) / self.scalxc_mn
            cs = jnp.asarray(cs) / self.scalxc_mn if cs is not None else None
        return sc, cs

    def mn_cos_to_signed_physical(self, cc: Any, ss: Any) -> Any:
        """Scale cosine-like internal blocks to physical signed coefficients."""

        return self.mn_cos_to_signed(*self._scale_physical_cos(cc, ss))

    def mn_sin_to_signed_physical(self, sc: Any, cs: Any) -> Any:
        """Scale sine-like internal blocks to physical signed coefficients."""

        return self.mn_sin_to_signed(*self._scale_physical_sin(sc, cs))

    def mn_sin_to_signed_physical_lambda(self, sc: Any, cs: Any) -> Any:
        """Map lambda sine updates onto signed physical coefficients."""

        return self.mn_sin_to_signed_physical(sc, cs)

    def mn_cos_to_signed_physical_lambda(self, cc: Any, ss: Any) -> Any:
        """Map asymmetric lambda cosine updates onto signed physical coefficients."""

        return self.mn_cos_to_signed_physical(cc, ss)

    def mn_sin_to_signed_physical_batch(self, sc: Any, cs: Any) -> Any:
        """Batched VMEC physical sine transform."""

        return _mn_sin_to_signed_physical_batch(
            sc,
            cs,
            scalxc_mn=self.scalxc_mn,
            mn_sin_to_signed_batch=self.mn_sin_to_signed_batch,
        )

    def rz_norm_np(self, state: Any) -> float:
        """Pure-NumPy R/Z norm for host-update preconditioner rebuilds."""

        return _rz_norm_np(
            state,
            kp_idx_np=self.kp_idx_np,
            kn_idx_np=self.kn_idx_np,
            has_kn_np=self.has_kn_np,
            m_idx_np=self.m_idx_np,
            n_idx_np=self.n_idx_np,
            include_rcc_np=self.include_rcc_np,
            lthreed=self.lthreed,
            lasym=self.lasym,
        )

    def rz_norm(self, state: Any) -> Any:
        """JAX R/Z norm in VMEC internal signed-mode convention."""

        rpos = jnp.asarray(state.Rcos)[:, self.kp_idx]
        zpos = jnp.asarray(state.Zsin)[:, self.kp_idx]
        has_kn_mask = self.has_kn[None, :]
        kn_idx_safe = jnp.maximum(self.kn_idx, 0)
        rneg = jnp.where(has_kn_mask, jnp.asarray(state.Rcos)[:, kn_idx_safe], 0.0)
        zneg = jnp.where(has_kn_mask, jnp.asarray(state.Zsin)[:, kn_idx_safe], 0.0)
        is_m0 = (self.m_idx == 0)[None, :]
        rcc = rpos + jnp.where(has_kn_mask, rneg, 0.0)
        zsc = jnp.where(has_kn_mask, zpos + zneg, zpos)
        is_n0 = (self.n_idx == 0)[None, :]
        rss = jnp.where(is_n0 | is_m0, 0.0, jnp.where(has_kn_mask, rpos - rneg, 0.0))
        zsc = jnp.where((~is_n0) & is_m0, 0.0, zsc)
        zcs = jnp.where(is_n0, 0.0, jnp.where(has_kn_mask, zneg - zpos, -zpos))

        # VMEC `bcovar_par` accumulates fnorm1 over l=2..ns (excludes axis).
        sl = slice(1, None)
        include_rcc = ((self.m_idx > 0) | (self.n_idx > 0))[None, :].astype(rcc.dtype)
        rz_norm = jnp.sum(zsc[sl] * zsc[sl]) + jnp.sum(include_rcc * (rcc[sl] * rcc[sl]))
        if self.lthreed:
            rz_norm = rz_norm + jnp.sum(rss[sl] * rss[sl]) + jnp.sum(zcs[sl] * zcs[sl])
        if self.lasym:
            rs_pos = jnp.asarray(state.Rsin)[:, self.kp_idx]
            zc_pos = jnp.asarray(state.Zcos)[:, self.kp_idx]
            rs_neg = jnp.where(has_kn_mask, jnp.asarray(state.Rsin)[:, kn_idx_safe], 0.0)
            zc_neg = jnp.where(has_kn_mask, jnp.asarray(state.Zcos)[:, kn_idx_safe], 0.0)

            rsc = jnp.where(has_kn_mask, rs_pos + rs_neg, jnp.where(is_n0, rs_pos, jnp.where(is_m0, 0.0, rs_pos)))
            rcs = jnp.where(has_kn_mask, rs_neg - rs_pos, jnp.where(is_n0, 0.0, jnp.where(is_m0, -rs_pos, 0.0)))
            zcc = zc_pos + jnp.where(has_kn_mask, zc_neg, 0.0)
            zss = jnp.where(is_n0 | is_m0, 0.0, jnp.where(has_kn_mask, zc_pos - zc_neg, 0.0))

            rz_norm = rz_norm + jnp.sum(rsc[sl] * rsc[sl]) + jnp.sum(rcs[sl] * rcs[sl])
            rz_norm = rz_norm + jnp.sum(zcc[sl] * zcc[sl]) + jnp.sum(zss[sl] * zss[sl])
        return rz_norm


def build_mode_transform_host_projection(signed_maps: Any, *, ncoeff: int) -> ModeTransformHostProjection:
    """Build host projection matrices for VMEC ``mn`` to signed-mode maps."""

    ncoeff_i = int(ncoeff)
    if ncoeff_i <= 0:
        return ModeTransformHostProjection(
            ncoeff=ncoeff_i,
            n_half=0,
            A_cos=None,
            A_sin=None,
            AB_cos=None,
            AB_sin=None,
        )

    idx_pos_safe_np = np.asarray(signed_maps.idx_pos_safe_flat, dtype=np.int32)
    idx_neg_safe_np = np.asarray(signed_maps.idx_neg_safe_flat, dtype=np.int32)
    idx_all_np = np.concatenate([idx_pos_safe_np, idx_neg_safe_np], axis=0)
    n_flat_mn = len(idx_all_np)
    proj_mn_signed = np.zeros((n_flat_mn, ncoeff_i), dtype=np.float64)
    for j in range(n_flat_mn):
        ix = int(idx_all_np[j])
        if 0 <= ix < ncoeff_i:
            proj_mn_signed[j, ix] = 1.0

    n_half = n_flat_mn // 2  # = mpol * nrange
    m0_mask_np = np.asarray(signed_maps.m0_mask, dtype=bool)
    n0_mask_np = np.asarray(signed_maps.n0_mask, dtype=bool)
    mask_pos_flat_f64 = np.asarray(signed_maps.mask_pos_flat, dtype=np.float64)
    mask_neg_flat_f64 = np.asarray(signed_maps.mask_neg_flat, dtype=np.float64)

    # m0_mask has shape (mpol, 1), n0_mask has shape (1, nrange).
    # | broadcasts to (mpol, nrange); individual reshape would give wrong size.
    mn_bcast_shape = np.broadcast_shapes(m0_mask_np.shape, n0_mask_np.shape)
    m0n0_flat_1d = (m0_mask_np | n0_mask_np).reshape(-1)
    n0_flat_1d = np.broadcast_to(n0_mask_np, mn_bcast_shape).reshape(-1)
    m0_flat_1d = np.broadcast_to(m0_mask_np, mn_bcast_shape).reshape(-1)
    has_neg_flat = mask_neg_flat_f64 > 0.0
    mask_no_neg_flat = ~has_neg_flat & ~n0_flat_1d

    proj_pos = proj_mn_signed[:n_half]
    proj_neg = proj_mn_signed[n_half:]

    # cos: pos = 0.5*(cc+ss) unless m0|n0 -> pos = cc.
    cc_pos_fac = np.where(m0n0_flat_1d, 1.0, 0.5) * mask_pos_flat_f64
    ss_pos_fac = np.where(m0n0_flat_1d, 0.0, 0.5) * mask_pos_flat_f64
    A_cos = cc_pos_fac[:, None] * proj_pos + 0.5 * mask_neg_flat_f64[:, None] * proj_neg
    B_cos = ss_pos_fac[:, None] * proj_pos + (-0.5) * mask_neg_flat_f64[:, None] * proj_neg

    # sin: pos depends on n0/mask_no_neg/m0 category.
    sc_pos_fac = (
        np.where(
            n0_flat_1d,
            1.0,
            np.where(mask_no_neg_flat & m0_flat_1d, 0.0, np.where(mask_no_neg_flat & ~m0_flat_1d, 1.0, 0.5)),
        )
        * mask_pos_flat_f64
    )
    cs_pos_fac = (
        np.where(
            n0_flat_1d,
            0.0,
            np.where(mask_no_neg_flat & m0_flat_1d, -1.0, np.where(mask_no_neg_flat & ~m0_flat_1d, 0.0, -0.5)),
        )
        * mask_pos_flat_f64
    )
    A_sin = sc_pos_fac[:, None] * proj_pos + 0.5 * mask_neg_flat_f64[:, None] * proj_neg
    B_sin = cs_pos_fac[:, None] * proj_pos + 0.5 * mask_neg_flat_f64[:, None] * proj_neg

    return ModeTransformHostProjection(
        ncoeff=ncoeff_i,
        n_half=n_half,
        A_cos=A_cos,
        A_sin=A_sin,
        AB_cos=np.vstack([A_cos, B_cos]),
        AB_sin=np.vstack([A_sin, B_sin]),
    )


def build_mode_transform_context(
    *,
    static: Any,
    state0: Any,
    s: Any,
    host_update_assembly: bool,
    setup_host_enforce: bool,
    divide_by_scalxc_for_update: bool,
    mode_diag_exponent: float,
    tree_has_tracer: Any,
    vmec_scalxc_from_s: Any,
) -> ModeTransformContext:
    """Build all mode-transform constants for the residual iteration loop."""

    from vmec_jax.vmec_parity import signed_maps_from_modes

    mpol = int(static.cfg.mpol)
    ntor = int(static.cfg.ntor)
    nrange = ntor + 1
    nfp = float(static.cfg.nfp)
    ncoeff = int(jnp.asarray(state0.Rcos).shape[1])
    signed_maps = (
        static.signed_maps if getattr(static, "signed_maps", None) is not None else signed_maps_from_modes(static.modes)
    )
    idx_pos = np.asarray(signed_maps.idx_pos, dtype=np.int32)
    idx_neg = np.asarray(signed_maps.idx_neg, dtype=np.int32)
    host_projection = build_mode_transform_host_projection(signed_maps, ncoeff=ncoeff)

    if getattr(static, "mn_idx_m", None) is not None:
        m_idx_np = np.asarray(static.mn_idx_m, dtype=np.int32)
        n_idx_np = np.asarray(static.mn_idx_n, dtype=np.int32)
        kp_idx_np = np.asarray(static.mn_idx_kp, dtype=np.int32)
        kn_idx_np = np.asarray(static.mn_idx_kn, dtype=np.int32)
        has_kn_np = np.asarray(static.mn_has_kn, dtype=bool) if static.mn_has_kn is not None else (kn_idx_np >= 0)
    else:
        m_idx_list: list[int] = []
        n_idx_list: list[int] = []
        kp_idx_list: list[int] = []
        kn_idx_list: list[int] = []
        for m_i in range(mpol):
            for n_i in range(nrange):
                kp = int(idx_pos[m_i, n_i])
                if kp < 0:
                    continue
                m_idx_list.append(m_i)
                n_idx_list.append(n_i)
                kp_idx_list.append(kp)
                kn_idx_list.append(int(idx_neg[m_i, n_i]))

        m_idx_np = np.asarray(m_idx_list, dtype=np.int32)
        n_idx_np = np.asarray(n_idx_list, dtype=np.int32)
        kp_idx_np = np.asarray(kp_idx_list, dtype=np.int32)
        kn_idx_np = np.asarray(kn_idx_list, dtype=np.int32)
        has_kn_np = kn_idx_np >= 0

    state0_dtype = (
        np.asarray(state0.Rcos).dtype
        if (bool(host_update_assembly) or bool(setup_host_enforce)) and (not tree_has_tracer(state0.Rcos))
        else jnp.asarray(state0.Rcos).dtype
    )
    if bool(host_update_assembly) and (not tree_has_tracer(s)):
        if bool(divide_by_scalxc_for_update):
            scalxc_mn_np = vmec_scalxc_from_s_np(s, mpol=mpol, dtype=state0_dtype)[:, :, None]
        else:
            scalxc_mn_np = np.ones((int(np.asarray(s).shape[0]), mpol, 1), dtype=state0_dtype)
        scalxc_mn = scalxc_mn_np
    else:
        scalxc_mn = vmec_scalxc_from_s(s=s, mpol=mpol).astype(jnp.asarray(state0.Rcos).dtype)[:, :, None]
        if not bool(divide_by_scalxc_for_update):
            scalxc_mn = jnp.ones_like(scalxc_mn)
        scalxc_mn_np = (
            np.asarray(scalxc_mn, dtype=float)
            if bool(host_update_assembly) and (not tree_has_tracer(scalxc_mn))
            else None
        )

    if bool(host_update_assembly) and (not tree_has_tracer(state0.Rcos)):
        w_mode_mn_np = mode_diag_weights_mn_np(
            mpol=mpol,
            nrange=nrange,
            nfp=nfp,
            mode_diag_exponent=mode_diag_exponent,
            dtype=state0_dtype,
        )
        w_mode_mn = jnp.asarray(w_mode_mn_np)
    else:
        w_mode_mn = mode_diag_weights_mn(
            mpol=mpol,
            nrange=nrange,
            nfp=nfp,
            mode_diag_exponent=mode_diag_exponent,
            dtype=jnp.asarray(state0.Rcos).dtype,
        )
        w_mode_mn_np = (
            np.asarray(w_mode_mn) if bool(host_update_assembly) and (not tree_has_tracer(w_mode_mn)) else None
        )

    m0_mask = np.asarray(
        getattr(static, "m_is_m0", None)
        if getattr(static, "m_is_m0", None) is not None
        else (np.asarray(static.modes.m) == 0)
    )

    return ModeTransformContext(
        mpol=mpol,
        ntor=ntor,
        nrange=nrange,
        nfp=nfp,
        ncoeff=ncoeff,
        signed_maps=signed_maps,
        idx_pos=idx_pos,
        idx_neg=idx_neg,
        m_idx_np=m_idx_np,
        n_idx_np=n_idx_np,
        kp_idx_np=kp_idx_np,
        kn_idx_np=kn_idx_np,
        has_kn_np=has_kn_np,
        m_idx=jnp.asarray(m_idx_np),
        n_idx=jnp.asarray(n_idx_np),
        kp_idx=jnp.asarray(kp_idx_np),
        kn_idx=jnp.asarray(kn_idx_np),
        has_kn=jnp.asarray(has_kn_np),
        include_rcc_np=(m_idx_np > 0) | (n_idx_np > 0),
        host_projection=host_projection,
        scalxc_mn=scalxc_mn,
        scalxc_mn_np=scalxc_mn_np,
        w_mode_mn=w_mode_mn,
        w_mode_mn_np=w_mode_mn_np,
        state0_dtype=state0_dtype,
        m0_mask=m0_mask,
        m0=jnp.asarray((np.arange(mpol)[:, None] == 0)),
        n0=jnp.asarray((np.arange(nrange)[None, :] == 0)),
        lthreed=bool(getattr(static.cfg, "lthreed", True)),
        lasym=bool(getattr(static.cfg, "lasym", False)),
        host_update_assembly=bool(host_update_assembly),
    )


def mn_cos_to_signed_host_projected(cc: Any, ss: Any, projection: ModeTransformHostProjection) -> np.ndarray:
    """Apply a precomputed host DGEMM projection for cosine-like modes."""

    cc_np = np.asarray(cc, dtype=float)
    ns = int(cc_np.shape[0])
    if projection.ncoeff == 0:
        return np.zeros((ns, projection.ncoeff), dtype=cc_np.dtype)
    if ss is None:
        return cc_np.reshape(ns, -1) @ projection.A_cos
    cc_ss = np.concatenate([cc_np.reshape(ns, -1), np.asarray(ss, dtype=float).reshape(ns, -1)], axis=1)
    return cc_ss @ projection.AB_cos


def mn_sin_to_signed_host_projected(sc: Any, cs: Any, projection: ModeTransformHostProjection) -> np.ndarray:
    """Apply a precomputed host DGEMM projection for sine-like modes."""

    sc_np = np.asarray(sc, dtype=float)
    ns = int(sc_np.shape[0])
    if projection.ncoeff == 0:
        return np.zeros((ns, projection.ncoeff), dtype=sc_np.dtype)
    if cs is None:
        return sc_np.reshape(ns, -1) @ projection.A_sin
    sc_cs = np.concatenate([sc_np.reshape(ns, -1), np.asarray(cs, dtype=float).reshape(ns, -1)], axis=1)
    return sc_cs @ projection.AB_sin


def vmec_scalxc_from_s_np(s_in: Any, *, mpol: int, dtype: Any) -> np.ndarray:
    """NumPy VMEC ``scalxc`` for the non-differentiated CPU host-update path."""

    s_np = np.asarray(s_in, dtype=dtype)
    ns_local = int(s_np.shape[0])
    mpol_local = int(mpol)
    if ns_local == 0 or mpol_local <= 0:
        return np.zeros((ns_local, max(mpol_local, 0)), dtype=dtype)
    sqrts = np.sqrt(np.maximum(s_np, 0.0)).astype(dtype, copy=False)
    sqrts = np.array(sqrts, dtype=dtype, copy=True)
    sqrts[-1] = np.asarray(1.0, dtype=dtype)
    sq2 = sqrts[1] if ns_local >= 2 else np.asarray(1.0, dtype=dtype)
    scal_odd = 1.0 / np.maximum(sqrts, sq2)
    is_odd = (np.arange(mpol_local) % 2) == 1
    return np.where(is_odd[None, :], scal_odd[:, None], np.ones((ns_local, mpol_local), dtype=dtype))


def mode_diag_weights_mn(
    *,
    mpol: int,
    nrange: int,
    nfp: float,
    mode_diag_exponent: float,
    dtype: Any,
) -> Any:
    """Return JAX mode-diagonal weights on the VMEC ``m,n>=0`` grid."""

    m = jnp.arange(int(mpol), dtype=jnp.float64)
    n = jnp.arange(int(nrange), dtype=jnp.float64) * float(nfp)
    k2 = (m[:, None] * m[:, None]) + (n[None, :] * n[None, :])
    w = (1.0 + k2) ** (-float(mode_diag_exponent))
    return w.astype(dtype)


def mode_diag_weights_mn_np(
    *,
    mpol: int,
    nrange: int,
    nfp: float,
    mode_diag_exponent: float,
    dtype: Any,
) -> np.ndarray:
    """Return NumPy mode-diagonal weights on the VMEC ``m,n>=0`` grid."""

    m = np.arange(int(mpol), dtype=float)
    n = np.arange(int(nrange), dtype=float) * float(nfp)
    k2 = (m[:, None] * m[:, None]) + (n[None, :] * n[None, :])
    return ((1.0 + k2) ** (-float(mode_diag_exponent))).astype(dtype)
