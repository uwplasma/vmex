"""Validation metrics for free-boundary finite-beta response.

The helpers in this module intentionally operate on converged ``wout`` data,
not transient solver internals.  They are used by diagnostics and tests to
check the physically expected finite-beta free-boundary response: pressure
should change the equilibrium profiles and the LCFS geometry, while VMEC2000
and vmec_jax should agree to much tighter tolerances than the beta-induced
change itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .plotting import bmag_from_wout_physical, surface_rz_from_wout_physical
from .wout import read_wout


@dataclass(frozen=True)
class FreeBoundaryResponseMetrics:
    """Finite-beta response metrics measured against a reference WOUT."""

    reference_beta_percent: float
    candidate_beta_percent: float
    beta_delta_percent: float
    reference_aspect: float
    candidate_aspect: float
    aspect_delta: float
    reference_mean_iota: float
    candidate_mean_iota: float
    mean_iota_delta: float
    lcfs_rms_displacement: float
    lcfs_max_displacement: float
    lcfs_max_abs_dR: float
    lcfs_max_abs_dZ: float
    lcfs_b_rel_rms_delta: float
    axis_R_shift: float
    axis_Z_shift: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON/CSV-friendly dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class VirtualCasingBoundaryDiagnostics:
    """Finite-beta boundary residuals from a virtual-casing postsolve check."""

    required_external_b: np.ndarray
    target_external_b: np.ndarray
    normal: np.ndarray
    external_bnormal_residual: np.ndarray
    external_bnormal_residual_rms: float
    external_bnormal_residual_max: float
    pressure_balance: np.ndarray
    pressure_balance_rms: float
    pressure_balance_max: float

    def to_dict(self) -> dict[str, float]:
        """Return scalar diagnostics suitable for summary tables."""
        return {
            "external_bnormal_residual_rms": self.external_bnormal_residual_rms,
            "external_bnormal_residual_max": self.external_bnormal_residual_max,
            "pressure_balance_rms": self.pressure_balance_rms,
            "pressure_balance_max": self.pressure_balance_max,
        }


def _as_wout(wout_or_path: Any) -> Any:
    if isinstance(wout_or_path, (str, Path)):
        return read_wout(wout_or_path)
    return wout_or_path


def _as_vector_surface(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f"{name} must have shape (3, ntheta, nphi)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _broadcast_surface_scalar(values: Any, shape: tuple[int, int], *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = np.full(shape, float(arr), dtype=float)
    else:
        try:
            arr = np.broadcast_to(arr, shape).astype(float, copy=False)
        except ValueError as exc:
            raise ValueError(f"{name} must be scalar or broadcastable to {shape}") from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return np.asarray(arr, dtype=float)


def _load_virtual_casing_functional():
    try:
        from virtual_casing_jax import functional as vc_functional
    except ImportError as exc:
        raise ImportError(
            "virtual_casing_jax is required for virtual-casing finite-beta boundary diagnostics"
        ) from exc
    return vc_functional


def virtual_casing_finite_beta_boundary_diagnostics(
    surface_xyz: Any,
    total_b: Any,
    *,
    target_external_b: Any | None = None,
    pressure: Any = 0.0,
    mu0: float = 4.0e-7 * np.pi,
    nfp: int = 1,
    half_period: bool = False,
    digits: int = 6,
    quad_nt: int | None = None,
    quad_np: int | None = None,
    patch_dim0: int | None = None,
    chunk_size: int | str | None = "auto",
    target_chunk_size: int | str | None = "auto",
    vc_module: Any | None = None,
) -> VirtualCasingBoundaryDiagnostics:
    """Return optional virtual-casing finite-beta boundary diagnostics.

    ``surface_xyz`` and ``total_b`` are single-field-period arrays with shape
    ``(3, ntheta, nphi)``.  The virtual-casing solve infers the external field
    required by the total field on that surface.  When ``target_external_b`` is
    supplied, it is interpreted as the actual coil/vacuum field on the same
    grid, and the diagnostic reports the normal-field mismatch between the two.

    The pressure-balance residual uses
    ``p + (|B_inside|^2 - |B_outside|^2) / (2 * mu0)`` with ``B_outside`` equal
    to ``target_external_b`` when provided, otherwise the virtual-casing
    required external field.  This makes coil-only ``B.n`` a vacuum check while
    giving finite-beta runs a postsolve total-field pressure-balance metric.
    """
    x = _as_vector_surface(surface_xyz, name="surface_xyz")
    b_total = _as_vector_surface(total_b, name="total_b")
    if b_total.shape != x.shape:
        raise ValueError(f"total_b shape {b_total.shape} does not match surface_xyz shape {x.shape}")
    ntheta = int(x.shape[1])
    nphi = int(x.shape[2])
    pressure_arr = _broadcast_surface_scalar(pressure, (ntheta, nphi), name="pressure")
    mu0 = float(mu0)
    if not np.isfinite(mu0) or mu0 <= 0.0:
        raise ValueError("mu0 must be positive and finite")

    vc = _load_virtual_casing_functional() if vc_module is None else vc_module
    quad_nt_eff = int(quad_nt) if quad_nt is not None else max(ntheta, 2 * ntheta)
    quad_np_eff = int(quad_np) if quad_np is not None else max(nphi, 2 * nphi)
    if quad_nt_eff < ntheta or quad_np_eff < nphi:
        raise ValueError("quad_nt and quad_np must be at least the target surface resolution")

    setup = vc.prepare_functional_setup(
        x,
        digits=int(digits),
        nfp=int(nfp),
        half_period=bool(half_period),
        surf_nt=ntheta,
        surf_np=nphi,
        src_nt=ntheta,
        src_np=nphi,
        trg_nt=ntheta,
        trg_np=nphi,
        quad_nt=quad_nt_eff,
        quad_np=quad_np_eff,
        patch_dim0=patch_dim0,
    )
    required_external_b = np.asarray(
        vc.compute_external_B_functional(
            x,
            b_total,
            digits=int(digits),
            nfp=setup.nfp,
            half_period=setup.half_period,
            surf_nt=setup.surf_nt,
            surf_np=setup.surf_np,
            src_nt=setup.src_nt,
            src_np=setup.src_np,
            trg_nt=setup.trg_nt,
            trg_np=setup.trg_np,
            quad_nt=setup.quad_nt,
            quad_np=setup.quad_np,
            patch_dim0=setup.patch_dim0,
            patch_idx=setup.patch_idx,
            orient=setup.orient,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
        ),
        dtype=float,
    )
    normal = np.asarray(
        vc.target_surface_normal(
            x,
            nfp=setup.nfp,
            half_period=setup.half_period,
            surf_nt=setup.surf_nt,
            surf_np=setup.surf_np,
            trg_nt=setup.trg_nt,
            trg_np=setup.trg_np,
            orient=setup.orient,
        ),
        dtype=float,
    )
    if required_external_b.shape != x.shape:
        raise ValueError(f"virtual_casing_jax returned B field shape {required_external_b.shape}, expected {x.shape}")
    if normal.shape != x.shape:
        raise ValueError(f"virtual_casing_jax returned normal shape {normal.shape}, expected {x.shape}")

    target_b = required_external_b if target_external_b is None else _as_vector_surface(
        target_external_b,
        name="target_external_b",
    )
    if target_b.shape != x.shape:
        raise ValueError(f"target_external_b shape {target_b.shape} does not match surface_xyz shape {x.shape}")

    external_bnormal_residual = np.einsum("ijk,ijk->jk", target_b - required_external_b, normal)
    pressure_balance = pressure_arr + (
        np.sum(b_total * b_total, axis=0) - np.sum(target_b * target_b, axis=0)
    ) / (2.0 * mu0)
    return VirtualCasingBoundaryDiagnostics(
        required_external_b=required_external_b,
        target_external_b=np.asarray(target_b, dtype=float),
        normal=normal,
        external_bnormal_residual=external_bnormal_residual,
        external_bnormal_residual_rms=float(np.sqrt(np.mean(external_bnormal_residual**2))),
        external_bnormal_residual_max=float(np.max(np.abs(external_bnormal_residual))),
        pressure_balance=pressure_balance,
        pressure_balance_rms=float(np.sqrt(np.mean(pressure_balance**2))),
        pressure_balance_max=float(np.max(np.abs(pressure_balance))),
    )


def wout_beta_percent(wout: Any) -> float:
    """Return total beta in percent from a VMEC WOUT-like object."""
    for name in ("betatotal", "beta_total"):
        value = getattr(wout, name, None)
        if value is not None:
            return 100.0 * float(value)
    return float("nan")


def wout_mean_iota(wout: Any) -> float:
    """Return the finite-surface mean rotational transform."""
    values = np.asarray(getattr(wout, "iotaf", getattr(wout, "iotas", [])), dtype=float)
    values = values[np.isfinite(values)]
    if values.size > 1:
        values = values[1:]
    return float(np.mean(values)) if values.size else float("nan")


def wout_fsq_total(wout: Any) -> float:
    """Return ``fsqr + fsqz + fsql`` from a WOUT-like object."""
    return float(getattr(wout, "fsqr", np.nan)) + float(getattr(wout, "fsqz", np.nan)) + float(
        getattr(wout, "fsql", np.nan)
    )


def free_boundary_response_metrics(
    reference_wout: Any,
    candidate_wout: Any,
    *,
    ntheta: int = 144,
    nphi: int = 48,
    s_index: int | None = None,
) -> FreeBoundaryResponseMetrics:
    """Measure finite-beta LCFS/profile response relative to ``reference_wout``.

    Parameters
    ----------
    reference_wout:
        Vacuum or lower-beta reference WOUT object/path.
    candidate_wout:
        Finite-beta WOUT object/path to compare against the reference.
    ntheta, nphi:
        Resolution used for LCFS geometry and ``|B|`` comparisons.
    s_index:
        Surface index.  Defaults to the last full surface in each WOUT.

    Notes
    -----
    The metric is deliberately geometric and WOUT-native.  It does not assume a
    particular solver backend, so the same function can compare VMEC2000
    ``mgrid`` output, vmec_jax ``mgrid`` output, and vmec_jax direct-coil output.
    """
    ref = _as_wout(reference_wout)
    cand = _as_wout(candidate_wout)
    ref_nfp = max(1, int(getattr(ref, "nfp", 1)))
    cand_nfp = max(1, int(getattr(cand, "nfp", ref_nfp)))
    if ref_nfp != cand_nfp:
        raise ValueError(f"nfp mismatch: reference nfp={ref_nfp}, candidate nfp={cand_nfp}")

    ref_s_index = int(getattr(ref, "ns")) - 1 if s_index is None else int(s_index)
    cand_s_index = int(getattr(cand, "ns")) - 1 if s_index is None else int(s_index)
    theta = np.linspace(0.0, 2.0 * np.pi, int(ntheta), endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi / float(ref_nfp), int(nphi), endpoint=False)

    R_ref, Z_ref = surface_rz_from_wout_physical(ref, theta=theta, phi=phi, s_index=ref_s_index)
    R_cand, Z_cand = surface_rz_from_wout_physical(cand, theta=theta, phi=phi, s_index=cand_s_index)
    displacement = np.sqrt((R_cand - R_ref) ** 2 + (Z_cand - Z_ref) ** 2)

    B_ref = bmag_from_wout_physical(ref, theta=theta, phi=phi, s_index=ref_s_index)
    B_cand = bmag_from_wout_physical(cand, theta=theta, phi=phi, s_index=cand_s_index)
    b_ref_rms = float(np.sqrt(np.nanmean(B_ref**2)))
    b_rel_rms = float(np.sqrt(np.nanmean((B_cand - B_ref) ** 2)) / (b_ref_rms + 1.0e-300))

    ref_beta = wout_beta_percent(ref)
    cand_beta = wout_beta_percent(cand)
    ref_iota = wout_mean_iota(ref)
    cand_iota = wout_mean_iota(cand)
    axis_ref_R = float(np.asarray(getattr(ref, "raxis_cc", [np.nan]), dtype=float)[0])
    axis_cand_R = float(np.asarray(getattr(cand, "raxis_cc", [np.nan]), dtype=float)[0])
    axis_ref_Z = float(np.asarray(getattr(ref, "zaxis_cs", [np.nan]), dtype=float)[0])
    axis_cand_Z = float(np.asarray(getattr(cand, "zaxis_cs", [np.nan]), dtype=float)[0])

    return FreeBoundaryResponseMetrics(
        reference_beta_percent=ref_beta,
        candidate_beta_percent=cand_beta,
        beta_delta_percent=cand_beta - ref_beta,
        reference_aspect=float(getattr(ref, "aspect", np.nan)),
        candidate_aspect=float(getattr(cand, "aspect", np.nan)),
        aspect_delta=float(getattr(cand, "aspect", np.nan)) - float(getattr(ref, "aspect", np.nan)),
        reference_mean_iota=ref_iota,
        candidate_mean_iota=cand_iota,
        mean_iota_delta=cand_iota - ref_iota,
        lcfs_rms_displacement=float(np.sqrt(np.nanmean(displacement**2))),
        lcfs_max_displacement=float(np.nanmax(displacement)),
        lcfs_max_abs_dR=float(np.nanmax(np.abs(R_cand - R_ref))),
        lcfs_max_abs_dZ=float(np.nanmax(np.abs(Z_cand - Z_ref))),
        lcfs_b_rel_rms_delta=b_rel_rms,
        axis_R_shift=axis_cand_R - axis_ref_R,
        axis_Z_shift=axis_cand_Z - axis_ref_Z,
    )
