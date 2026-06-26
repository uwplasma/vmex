"""Boundary metric and vacuum-field projection helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .types import ExternalBoundarySample, FreeBoundarySampleSetup, VacuumBoundaryFields

_FREEB_HOST_PHASE_CACHE: dict[tuple[int, int, tuple[str, ...]], np.ndarray] = {}
_FREEB_TRIG_CACHE: dict[tuple[int, int, int, int, int, bool], Any] = {}
_FREEB_BOUNDARY_SETUP_CACHE: dict[tuple[int, int], Any] = {}
_FREEB_WINT_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def freeb_host_phase_stack(*, modes: Any, trig: Any, derivs: tuple[str, ...]) -> np.ndarray:
    """Return cached NumPy phase stacks for host-side boundary synthesis."""

    key = (id(modes), id(trig), tuple(derivs))
    cached = _FREEB_HOST_PHASE_CACHE.get(key)
    if cached is not None:
        try:
            K = int(np.asarray(modes.m).shape[0])
            nzeta = int(np.asarray(trig.cosnv).shape[0])
            ntheta3 = int(np.asarray(trig.cosmu).shape[0])
            if (
                int(cached.shape[1]) != 2 * K
                or int(cached.shape[2]) != ntheta3
                or int(cached.shape[3]) != nzeta
            ):
                cached = None
        except Exception:
            cached = None
    if cached is not None:
        return cached

    m = np.asarray(modes.m, dtype=np.int32)
    n = np.asarray(modes.n, dtype=np.int32)
    n1 = np.abs(n)
    sgn = np.where(n < 0, -1.0, 1.0)

    cosmu = np.asarray(trig.cosmu, dtype=float)
    sinmu = np.asarray(trig.sinmu, dtype=float)
    cosmum = np.asarray(trig.cosmum, dtype=float)
    sinmum = np.asarray(trig.sinmum, dtype=float)
    cosnv = np.asarray(trig.cosnv, dtype=float)
    sinnv = np.asarray(trig.sinnv, dtype=float)
    cosnvn = np.asarray(trig.cosnvn, dtype=float)
    sinnvn = np.asarray(trig.sinnvn, dtype=float)

    cosmu_m = cosmu[:, m].T
    sinmu_m = sinmu[:, m].T
    cosmum_m = cosmum[:, m].T
    sinmum_m = sinmum[:, m].T
    cosnv_n = cosnv[:, n1].T
    sinnv_n = sinnv[:, n1].T
    cosnvn_n = cosnvn[:, n1].T
    sinnvn_n = sinnvn[:, n1].T

    phase_blocks: list[np.ndarray] = []
    for deriv in derivs:
        if deriv == "base":
            cos_phase = cosmu_m[:, :, None] * cosnv_n[:, None, :] + sgn[:, None, None] * sinmu_m[:, :, None] * sinnv_n[:, None, :]
            sin_phase = sinmu_m[:, :, None] * cosnv_n[:, None, :] - sgn[:, None, None] * cosmu_m[:, :, None] * sinnv_n[:, None, :]
        elif deriv == "dtheta":
            cos_phase = sinmum_m[:, :, None] * cosnv_n[:, None, :] + sgn[:, None, None] * cosmum_m[:, :, None] * sinnv_n[:, None, :]
            sin_phase = cosmum_m[:, :, None] * cosnv_n[:, None, :] - sgn[:, None, None] * sinmum_m[:, :, None] * sinnv_n[:, None, :]
        elif deriv == "dzeta":
            cos_phase = cosmu_m[:, :, None] * sinnvn_n[:, None, :] + sgn[:, None, None] * sinmu_m[:, :, None] * cosnvn_n[:, None, :]
            sin_phase = sinmu_m[:, :, None] * sinnvn_n[:, None, :] - sgn[:, None, None] * cosmu_m[:, :, None] * cosnvn_n[:, None, :]
        else:  # pragma: no cover
            raise ValueError(f"Unknown deriv={deriv!r}")
        phase_blocks.append(np.concatenate([cos_phase, sin_phase], axis=0))

    phase_all = np.stack(phase_blocks, axis=0)
    _FREEB_HOST_PHASE_CACHE[key] = phase_all
    return phase_all


def freeb_boundary_trig(*, cfg: Any, nzeta: int) -> Any:
    """Return cached trig tables for free-boundary boundary sampling."""

    from ...kernels.tomnsp import vmec_trig_tables

    key = (
        int(cfg.ntheta),
        int(nzeta),
        int(cfg.nfp),
        int(cfg.mpol) - 1,
        int(cfg.ntor),
        bool(cfg.lasym),
    )
    cached = _FREEB_TRIG_CACHE.get(key)
    if cached is not None:
        return cached
    trig = vmec_trig_tables(
        ntheta=int(cfg.ntheta),
        nzeta=int(nzeta),
        nfp=int(cfg.nfp),
        mmax=int(cfg.mpol) - 1,
        nmax=int(cfg.ntor),
        lasym=bool(cfg.lasym),
    )
    _FREEB_TRIG_CACHE[key] = trig
    return trig


def vmec_realspace_synthesis_multi_host(
    *,
    coeff_cos: np.ndarray,
    coeff_sin: np.ndarray,
    modes: Any,
    trig: Any,
    derivs: tuple[str, ...] = ("base",),
) -> tuple[np.ndarray, ...]:
    """Host-side VMEC synthesis for external boundary sampling."""

    coeff_cos = np.asarray(coeff_cos, dtype=float)
    coeff_sin = np.asarray(coeff_sin, dtype=float)
    if coeff_cos.shape != coeff_sin.shape:
        raise ValueError("coeff_cos and coeff_sin must have the same shape")
    phase_all = freeb_host_phase_stack(modes=modes, trig=trig, derivs=tuple(derivs))
    coeff = np.concatenate([coeff_cos, coeff_sin], axis=-1)
    out = np.einsum("...k,tkij->t...ij", coeff, phase_all, optimize=True)
    return tuple(np.asarray(out[i], dtype=float) for i in range(len(derivs)))


def vmec_boundary_wint(*, static: Any, ntheta: int, nzeta: int, trig: Any | None = None) -> np.ndarray:
    """Return VMEC angular weights on the free-boundary mesh."""

    key = (id(static), int(ntheta), int(nzeta))
    cached = _FREEB_WINT_CACHE.get(key)
    if cached is not None:
        return cached

    trig = getattr(static, "trig_vmec", None) if trig is None else trig
    if trig is not None:
        try:
            from ...kernels.residue import vmec_wint_from_trig

            w = np.asarray(vmec_wint_from_trig(trig, nzeta=int(nzeta)), dtype=float)
            if w.shape == (int(ntheta), int(nzeta)):
                _FREEB_WINT_CACHE[key] = w
                return w
        except Exception:
            pass
    w = np.full((int(ntheta), int(nzeta)), 1.0 / float(max(1, int(ntheta) * int(nzeta))), dtype=float)
    _FREEB_WINT_CACHE[key] = w
    return w


def freeb_boundary_sample_setup(*, static: Any, sample_nzeta: int) -> FreeBoundarySampleSetup:
    """Return cached static data used by host-side free-boundary sampling."""

    _trig_hint = getattr(static, "trig_vmec", None)
    _ntheta3_hint = int(_trig_hint.ntheta3) if _trig_hint is not None else -1
    key = (id(static), _ntheta3_hint, int(sample_nzeta))
    cached = _FREEB_BOUNDARY_SETUP_CACHE.get(key)
    if cached is not None:
        return cached

    trig = getattr(static, "trig_vmec", None)
    if trig is None or int(sample_nzeta) != int(static.cfg.nzeta):
        trig = freeb_boundary_trig(cfg=static.cfg, nzeta=int(sample_nzeta))

    m_arr = np.asarray(static.modes.m, dtype=float)
    n_arr = np.asarray(static.modes.n, dtype=float) * float(int(static.cfg.nfp))
    second_facs = np.stack(
        [
            (-(m_arr**2)).reshape((1, -1)),
            (m_arr * n_arr).reshape((1, -1)),
            (-(n_arr**2)).reshape((1, -1)),
        ],
        axis=0,
    )

    ntheta = int(np.asarray(trig.cosmu).shape[0])
    nzeta = max(1, int(sample_nzeta))
    phi = (2.0 * np.pi / float(nzeta)) * np.arange(nzeta, dtype=float)
    phi /= float(max(1, int(static.cfg.nfp)))
    phi_grid = np.broadcast_to(phi[None, :], (ntheta, nzeta))

    if getattr(static, "m_is_even", None) is not None:
        even_m_mask = np.asarray(static.m_is_even, dtype=float).reshape((1, 1, -1))
    else:
        even_m_mask = (np.asarray(static.modes.m, dtype=int) % 2 == 0).astype(float).reshape((1, 1, -1))

    setup = FreeBoundarySampleSetup(
        trig=trig,
        second_facs=second_facs,
        phi_grid=phi_grid,
        even_m_mask=even_m_mask,
        wint_vmec=vmec_boundary_wint(static=static, ntheta=ntheta, nzeta=nzeta, trig=trig),
    )
    _FREEB_BOUNDARY_SETUP_CACHE[key] = setup
    return setup


def boundary_metric_from_rz(
    *,
    R: Any,
    Ru: Any,
    Zu: Any,
    Rv: Any,
    Zv: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute 2D boundary metric terms for ``(u=theta, v=zeta_phys)``."""

    R = np.asarray(R, dtype=float)
    Ru = np.asarray(Ru, dtype=float)
    Zu = np.asarray(Zu, dtype=float)
    Rv = np.asarray(Rv, dtype=float)
    Zv = np.asarray(Zv, dtype=float)
    g_uu = Ru * Ru + Zu * Zu
    g_uv = Ru * Rv + Zu * Zv
    g_vv = R * R + Rv * Rv + Zv * Zv
    det = g_uu * g_vv - g_uv * g_uv
    return g_uu, g_uv, g_vv, det


def covariant_boundary_field_from_cylindrical(
    *,
    br: Any,
    bp: Any,
    bz: Any,
    R: Any,
    Ru: Any,
    Zu: Any,
    Rv: Any,
    Zv: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Project cylindrical field ``(Br,Bphi,Bz)`` to boundary covariant components."""

    br = np.asarray(br, dtype=float)
    bp = np.asarray(bp, dtype=float)
    bz = np.asarray(bz, dtype=float)
    R = np.asarray(R, dtype=float)
    Ru = np.asarray(Ru, dtype=float)
    Zu = np.asarray(Zu, dtype=float)
    Rv = np.asarray(Rv, dtype=float)
    Zv = np.asarray(Zv, dtype=float)
    bu = br * Ru + bz * Zu
    bv = br * Rv + bp * R + bz * Zv
    return bu, bv


def contravariant_boundary_field_from_covariant(
    *,
    bu: Any,
    bv: Any,
    g_uu: Any,
    g_uv: Any,
    g_vv: Any,
    det_floor: float = 1.0e-30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ``(B^u,B^v)`` by inverting the 2x2 surface metric."""

    bu = np.asarray(bu, dtype=float)
    bv = np.asarray(bv, dtype=float)
    g_uu = np.asarray(g_uu, dtype=float)
    g_uv = np.asarray(g_uv, dtype=float)
    g_vv = np.asarray(g_vv, dtype=float)
    det = g_uu * g_vv - g_uv * g_uv
    det_safe = np.where(np.abs(det) >= float(det_floor), det, np.sign(det + 1.0e-300) * float(det_floor))
    bsupu = (g_vv * bu - g_uv * bv) / det_safe
    bsupv = (g_uu * bv - g_uv * bu) / det_safe
    return bsupu, bsupv, det


def vacuum_boundary_fields_from_cylindrical(
    *,
    br: Any,
    bp: Any,
    bz: Any,
    R: Any,
    Ru: Any,
    Zu: Any,
    Rv: Any,
    Zv: Any,
    det_floor: float = 1.0e-30,
) -> VacuumBoundaryFields:
    """Compute VMEC-like vacuum boundary channels from cylindrical field input."""

    R = np.asarray(R, dtype=float)
    Ru = np.asarray(Ru, dtype=float)
    Zu = np.asarray(Zu, dtype=float)
    Rv = np.asarray(Rv, dtype=float)
    Zv = np.asarray(Zv, dtype=float)
    br = np.asarray(br, dtype=float)
    bp = np.asarray(bp, dtype=float)
    bz = np.asarray(bz, dtype=float)

    g_uu, g_uv, g_vv, _ = boundary_metric_from_rz(R=R, Ru=Ru, Zu=Zu, Rv=Rv, Zv=Zv)
    bu, bv = covariant_boundary_field_from_cylindrical(
        br=br,
        bp=bp,
        bz=bz,
        R=R,
        Ru=Ru,
        Zu=Zu,
        Rv=Rv,
        Zv=Zv,
    )
    bsupu, bsupv, det = contravariant_boundary_field_from_covariant(
        bu=bu,
        bv=bv,
        g_uu=g_uu,
        g_uv=g_uv,
        g_vv=g_vv,
        det_floor=det_floor,
    )
    # VMEC vacuum.f stores bsqvac as 0.5*|B|^2 on the boundary.
    bsqvac = 0.5 * (bu * bsupu + bv * bsupv)

    # Non-unit boundary normal from x_u x x_v in cylindrical components.
    n_r = -R * Zu
    n_phi = Zu * Rv - Ru * Zv
    n_z = R * Ru
    bnormal = br * n_r + bp * n_phi + bz * n_z
    n_norm = np.sqrt(n_r * n_r + n_phi * n_phi + n_z * n_z)
    n_norm_safe = np.where(n_norm > 0.0, n_norm, 1.0)
    bnormal_unit = bnormal / n_norm_safe

    return VacuumBoundaryFields(
        bu=bu,
        bv=bv,
        bsupu=bsupu,
        bsupv=bsupv,
        bsqvac=bsqvac,
        bnormal=bnormal,
        bnormal_unit=bnormal_unit,
        g_uu=g_uu,
        g_uv=g_uv,
        g_vv=g_vv,
        det_guv=det,
    )


def sample_free_boundary_external_field(
    *,
    R: Any,
    Z: Any,
    Ru: Any,
    Zu: Any,
    Rv: Any,
    Zv: Any,
    phi: Any,
    provider_kind: str,
    provider_static: Any = None,
    provider_params: Any = None,
    axis_field: tuple[Any, Any, Any] | None = None,
    axis_r: Any | None = None,
    axis_z: Any | None = None,
    label: str | None = None,
) -> ExternalBoundarySample:
    """Project a provider-sampled external field onto VMEC boundary channels."""

    from ...external_fields import sample_external_field_cylindrical

    R_arr = np.asarray(R, dtype=float)
    Z_arr = np.asarray(Z, dtype=float)
    Ru_arr = np.asarray(Ru, dtype=float)
    Zu_arr = np.asarray(Zu, dtype=float)
    Rv_arr = np.asarray(Rv, dtype=float)
    Zv_arr = np.asarray(Zv, dtype=float)
    phi_arr = np.asarray(phi, dtype=float)
    br_ext, bp_ext, bz_ext = sample_external_field_cylindrical(
        provider_kind,
        provider_static,
        provider_params,
        R_arr,
        Z_arr,
        phi_arr,
    )
    br_ext = np.asarray(br_ext, dtype=float)
    bp_ext = np.asarray(bp_ext, dtype=float)
    bz_ext = np.asarray(bz_ext, dtype=float)
    if axis_field is None:
        br_axis = np.zeros_like(br_ext)
        bp_axis = np.zeros_like(bp_ext)
        bz_axis = np.zeros_like(bz_ext)
    else:
        br_axis = np.asarray(axis_field[0], dtype=float)
        bp_axis = np.asarray(axis_field[1], dtype=float)
        bz_axis = np.asarray(axis_field[2], dtype=float)
    br = br_ext + br_axis
    bp = bp_ext + bp_axis
    bz = bz_ext + bz_axis
    vac = vacuum_boundary_fields_from_cylindrical(
        br=br,
        bp=bp,
        bz=bz,
        R=R_arr,
        Ru=Ru_arr,
        Zu=Zu_arr,
        Rv=Rv_arr,
        Zv=Zv_arr,
    )
    nzeta = int(R_arr.shape[-1]) if R_arr.ndim else 1
    axis_r_arr = np.zeros(nzeta, dtype=float) if axis_r is None else np.asarray(axis_r, dtype=float)
    axis_z_arr = np.zeros(nzeta, dtype=float) if axis_z is None else np.asarray(axis_z, dtype=float)
    return ExternalBoundarySample(
        mgrid_path=str(label or provider_kind),
        R=R_arr,
        Z=Z_arr,
        Ru=Ru_arr,
        Zu=Zu_arr,
        Rv=Rv_arr,
        Zv=Zv_arr,
        phi=phi_arr,
        br=br,
        bp=bp,
        bz=bz,
        br_mgrid=br_ext,
        bp_mgrid=bp_ext,
        bz_mgrid=bz_ext,
        br_axis=br_axis,
        bp_axis=bp_axis,
        bz_axis=bz_axis,
        axis_r=axis_r_arr,
        axis_z=axis_z_arr,
        vac_ext=vac,
    )


__all__ = [
    "boundary_metric_from_rz",
    "contravariant_boundary_field_from_covariant",
    "covariant_boundary_field_from_cylindrical",
    "freeb_boundary_sample_setup",
    "freeb_boundary_trig",
    "freeb_host_phase_stack",
    "sample_free_boundary_external_field",
    "vacuum_boundary_fields_from_cylindrical",
    "vmec_boundary_wint",
    "vmec_realspace_synthesis_multi_host",
]
