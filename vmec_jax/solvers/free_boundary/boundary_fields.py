"""Boundary metric and vacuum-field projection helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .types import ExternalBoundarySample, VacuumBoundaryFields


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
    "sample_free_boundary_external_field",
    "vacuum_boundary_fields_from_cylindrical",
]
