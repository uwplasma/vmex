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
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from ...plotting import bmag_from_wout_physical, surface_rz_from_wout_physical
from ...wout import read_wout


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
    surface_ntheta: int
    surface_nphi: int
    quad_ntheta: int
    quad_nphi: int
    quad_factor_theta: float
    quad_factor_phi: float
    grid_adequacy_status: str
    nfp: int
    half_period: bool
    digits: int
    patch_dim0: int | None
    chunk_size: int | str | None
    target_chunk_size: int | str | None

    def to_dict(self) -> dict[str, float | int | str | bool | None]:
        """Return scalar diagnostics suitable for summary tables."""
        return {
            "external_bnormal_residual_rms": self.external_bnormal_residual_rms,
            "external_bnormal_residual_max": self.external_bnormal_residual_max,
            "pressure_balance_rms": self.pressure_balance_rms,
            "pressure_balance_max": self.pressure_balance_max,
            "surface_ntheta": self.surface_ntheta,
            "surface_nphi": self.surface_nphi,
            "quad_ntheta": self.quad_ntheta,
            "quad_nphi": self.quad_nphi,
            "quad_factor_theta": self.quad_factor_theta,
            "quad_factor_phi": self.quad_factor_phi,
            "grid_adequacy_status": self.grid_adequacy_status,
            "nfp": self.nfp,
            "half_period": self.half_period,
            "digits": self.digits,
            "patch_dim0": self.patch_dim0,
            "chunk_size": self.chunk_size,
            "target_chunk_size": self.target_chunk_size,
        }


@dataclass(frozen=True)
class SolvedBoundaryFieldSample:
    """Solved LCFS geometry and magnetic field on the VMEC angular grid."""

    theta: np.ndarray
    zeta: np.ndarray
    R: np.ndarray
    Z: np.ndarray
    Bmag: np.ndarray
    Bmag_near_axis: np.ndarray
    Bxyz: np.ndarray
    bsupu: np.ndarray
    bsupv: np.ndarray
    surface_xyz: np.ndarray


def free_boundary_promotion_status(
    *,
    beta_percent: Any,
    strict_components_met: Any,
    final_residual_recomputed: Any | None = None,
    virtual_casing_status: Any | None = None,
    virtual_casing_grid_adequacy_status: Any | None = None,
    direct_coil_backend: bool = True,
    require_fresh_residual: bool = True,
    beta_tol: float = 1.0e-12,
) -> dict[str, Any]:
    """Classify whether a free-boundary row is promotion-ready.

    The VMEC force residual is the hard convergence gate.  For finite-beta
    direct-coil rows, coil-only ``B.n`` is only a diagnostic because the plasma
    field also contributes at the boundary; those rows need a total-field
    boundary diagnostic such as virtual casing before being treated as
    production evidence.
    """

    def _optional_bool(value: Any) -> bool | None:
        if value is None:
            return None
        if isinstance(value, str):
            key = value.strip().lower()
            if key in {"true", "1", "yes"}:
                return True
            if key in {"false", "0", "no", "none", "null", "nan", ""}:
                return False if key in {"false", "0", "no"} else None
        try:
            scalar = float(value)
        except Exception:
            return bool(value)
        if not np.isfinite(scalar):
            return None
        return bool(scalar)

    try:
        beta_value = float(beta_percent)
    except Exception:
        beta_value = 0.0
    finite_beta = bool(abs(beta_value) > float(beta_tol))
    strict_met = _optional_bool(strict_components_met)
    strict_known = strict_met is not None
    fresh_met = _optional_bool(final_residual_recomputed)
    fresh_known = fresh_met is not None
    vc_status = None if virtual_casing_status is None else str(virtual_casing_status)
    vc_grid_status = None if virtual_casing_grid_adequacy_status is None else str(
        virtual_casing_grid_adequacy_status
    )
    vc_required = bool(finite_beta and direct_coil_backend)
    vc_available = None if not vc_required else bool(vc_status == "computed")

    blockers: list[str] = []
    if strict_met is not True:
        blockers.append("strict_force_components_unknown" if not strict_known else "strict_force_components_not_met")
    if bool(require_fresh_residual) and fresh_met is not True:
        blockers.append("fresh_final_residual_unknown" if not fresh_known else "fresh_final_residual_not_recomputed")
    if vc_required and vc_available is not True:
        if vc_status is None:
            blockers.append("virtual_casing_diagnostics_missing")
        else:
            blockers.append(f"virtual_casing_diagnostics_{vc_status}")
    if vc_required and vc_available is True and vc_grid_status not in {None, "production_ready"}:
        blockers.append(f"virtual_casing_grid_{vc_grid_status}")

    return {
        "beta_percent": beta_value,
        "finite_beta": finite_beta,
        "boundary_condition_mode": "finite_beta_total_field" if finite_beta else "vacuum_coil_normal",
        "coil_bnormal_role": "diagnostic_only" if finite_beta else "vacuum_boundary_condition",
        "strict_force_components_met": strict_met,
        "fresh_final_residual_required": bool(require_fresh_residual),
        "fresh_final_residual_met": fresh_met,
        "direct_coil_backend": bool(direct_coil_backend),
        "virtual_casing_required": vc_required,
        "virtual_casing_status": vc_status,
        "virtual_casing_grid_adequacy_status": vc_grid_status,
        "virtual_casing_available": vc_available,
        "production_candidate": not blockers,
        "promotion_blockers": blockers,
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


def virtual_casing_grid_adequacy_status(
    *,
    surface_ntheta: int,
    surface_nphi: int,
    quad_ntheta: int,
    quad_nphi: int,
    recommended_quad_factor: float = 2.0,
) -> str:
    """Classify whether a virtual-casing diagnostic grid is production-sized.

    The virtual-casing residual is only a postsolve diagnostic, but finite-beta
    free-boundary promotion depends on it.  We therefore require at least the
    default 2x source quadrature in both angular directions before treating the
    result as production evidence.
    """

    ntheta = int(surface_ntheta)
    nphi = int(surface_nphi)
    qtheta = int(quad_ntheta)
    qphi = int(quad_nphi)
    if ntheta <= 0 or nphi <= 0 or qtheta <= 0 or qphi <= 0:
        return "invalid_grid"
    if qtheta < ntheta or qphi < nphi:
        return "invalid_quad_below_surface"
    recommended = float(recommended_quad_factor)
    theta_ok = qtheta >= int(np.ceil(recommended * ntheta))
    phi_ok = qphi >= int(np.ceil(recommended * nphi))
    if theta_ok and phi_ok:
        return "production_ready"
    if not theta_ok and not phi_ok:
        return "diagnostic_only_low_quad_theta_phi"
    if not theta_ok:
        return "diagnostic_only_low_quad_theta"
    return "diagnostic_only_low_quad_phi"


def _load_virtual_casing_functional():
    extra_path = os.environ.get("VMEC_JAX_VIRTUAL_CASING_JAX_PATH")
    if extra_path:
        for raw in reversed([part for part in extra_path.split(os.pathsep) if part.strip()]):
            path = str(Path(raw).expanduser())
            if path not in sys.path:
                sys.path.insert(0, path)
    try:
        from virtual_casing_jax import functional as vc_functional
    except ImportError as exc:
        raise ImportError(
            "virtual_casing_jax is required for virtual-casing finite-beta boundary diagnostics. "
            "Install virtual-casing-jax or set VMEC_JAX_VIRTUAL_CASING_JAX_PATH to its source checkout."
        ) from exc
    return vc_functional


def surface_xyz_from_rz(R: Any, Z: Any, zeta: Any, *, nfp: int = 1) -> np.ndarray:
    """Return Cartesian surface coordinates from VMEC ``R,Z,zeta`` samples."""

    R_arr = np.asarray(R, dtype=float)
    Z_arr = np.asarray(Z, dtype=float)
    zeta_arr = np.asarray(zeta, dtype=float)
    if R_arr.shape != Z_arr.shape:
        raise ValueError(f"R and Z shapes must match, got {R_arr.shape} and {Z_arr.shape}")
    if R_arr.ndim != 2:
        raise ValueError("R and Z must have shape (ntheta, nzeta)")
    if zeta_arr.ndim != 1 or zeta_arr.shape[0] != R_arr.shape[1]:
        raise ValueError("zeta must be a 1D array with length matching the second R/Z dimension")
    phi = zeta_arr / float(max(1, int(nfp)))
    return np.stack(
        (R_arr * np.cos(phi)[None, :], R_arr * np.sin(phi)[None, :], Z_arr),
        axis=0,
    )


def edge_pressure_from_indata(indata: Any) -> float:
    """Return the VMEC power-series pressure evaluated on the LCFS."""

    scalars = getattr(indata, "scalars", {}) if indata is not None else {}
    am = np.asarray(scalars.get("AM", [0.0]), dtype=float).reshape(-1)
    pres_scale = float(scalars.get("PRES_SCALE", 1.0) or 0.0)
    if not am.size:
        return 0.0
    return float(pres_scale * np.sum(am))


def edge_pressure_from_run(run: Any) -> float:
    """Return the input edge pressure for a driver run object."""

    return edge_pressure_from_indata(getattr(run, "indata", None))


def sample_solved_boundary_field(run: Any, *, nfp: int | None = None) -> SolvedBoundaryFieldSample:
    """Sample solved LCFS geometry, field strength, and Cartesian total field."""

    from ...field import b2_from_bsup, b_cartesian_from_bsup, bsup_from_geom, lamscale_from_phips
    from ...geom import eval_geom

    nfp_eff = int(nfp if nfp is not None else getattr(getattr(run.static, "cfg", None), "nfp", 1))
    geom = eval_geom(run.state, run.static)
    lamscale = lamscale_from_phips(run.flux.phips, run.static.s)
    bsupu, bsupv = bsup_from_geom(
        geom,
        phipf=run.flux.phipf,
        chipf=run.flux.chipf,
        nfp=nfp_eff,
        signgs=int(run.signgs),
        lamscale=lamscale,
        flux_is_internal=True,
    )
    bmag = np.sqrt(np.asarray(b2_from_bsup(geom, bsupu, bsupv), dtype=float))
    bxyz = np.asarray(
        b_cartesian_from_bsup(geom, bsupu, bsupv, zeta=run.static.grid.zeta, nfp=nfp_eff),
        dtype=float,
    )
    near_axis_idx = 1 if int(bmag.shape[0]) > 1 else 0
    theta = np.asarray(run.static.grid.theta, dtype=float)
    zeta = np.asarray(run.static.grid.zeta, dtype=float)
    R = np.asarray(geom.R[-1], dtype=float)
    Z = np.asarray(geom.Z[-1], dtype=float)
    return SolvedBoundaryFieldSample(
        theta=theta,
        zeta=zeta,
        R=R,
        Z=Z,
        Bmag=np.asarray(bmag[-1], dtype=float),
        Bmag_near_axis=np.asarray(bmag[near_axis_idx], dtype=float),
        Bxyz=np.moveaxis(bxyz[-1], -1, 0),
        bsupu=np.asarray(bsupu[-1], dtype=float),
        bsupv=np.asarray(bsupv[-1], dtype=float),
        surface_xyz=surface_xyz_from_rz(R, Z, zeta, nfp=nfp_eff),
    )


def coil_external_b_on_surface(
    surface_xyz: Any,
    coil_params: Any,
    *,
    coil_geometry: Any | None = None,
    regularization_epsilon: float | None = None,
    chunk_size: int | None = None,
) -> np.ndarray:
    """Sample a direct-coil Cartesian field on a ``(3, ntheta, nzeta)`` surface."""

    from ...external_fields import build_coil_field_geometry, sample_coil_field_xyz_from_geometry

    x = _as_vector_surface(surface_xyz, name="surface_xyz")
    geometry = build_coil_field_geometry(coil_params) if coil_geometry is None else coil_geometry
    eps = (
        float(getattr(coil_params, "regularization_epsilon", 0.0))
        if regularization_epsilon is None
        else float(regularization_epsilon)
    )
    chunks = getattr(coil_params, "chunk_size", None) if chunk_size is None else chunk_size
    points_xyz = np.moveaxis(x, 0, -1)
    coil_xyz = np.asarray(
        sample_coil_field_xyz_from_geometry(
            geometry,
            points_xyz,
            regularization_epsilon=eps,
            chunk_size=chunks,
        ),
        dtype=float,
    )
    return np.moveaxis(coil_xyz, -1, 0)


def virtual_casing_diagnostics_from_run(
    run: Any,
    *,
    coil_params: Any | None = None,
    coil_geometry: Any | None = None,
    target_external_b: Any | None = None,
    pressure: Any | None = None,
    nfp: int | None = None,
    digits: int = 6,
    quad_factor: int = 2,
    chunk_size: int | str | None = "auto",
    target_chunk_size: int | str | None = "auto",
    vc_module: Any | None = None,
) -> VirtualCasingBoundaryDiagnostics:
    """Compute finite-beta virtual-casing boundary diagnostics for a solved run."""

    nfp_eff = int(nfp if nfp is not None else getattr(getattr(run.static, "cfg", None), "nfp", 1))
    sample = sample_solved_boundary_field(run, nfp=nfp_eff)
    target_b = target_external_b
    if target_b is None and coil_params is not None:
        target_b = coil_external_b_on_surface(
            sample.surface_xyz,
            coil_params,
            coil_geometry=coil_geometry,
        )
    pressure_eff = edge_pressure_from_run(run) if pressure is None else pressure
    return virtual_casing_finite_beta_boundary_diagnostics(
        sample.surface_xyz,
        sample.Bxyz,
        target_external_b=target_b,
        pressure=pressure_eff,
        nfp=nfp_eff,
        digits=int(digits),
        quad_nt=max(int(quad_factor) * int(sample.surface_xyz.shape[1]), int(sample.surface_xyz.shape[1])),
        quad_np=max(int(quad_factor) * int(sample.surface_xyz.shape[2]), int(sample.surface_xyz.shape[2])),
        chunk_size=chunk_size,
        target_chunk_size=target_chunk_size,
        vc_module=vc_module,
    )


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
    grid_status = virtual_casing_grid_adequacy_status(
        surface_ntheta=ntheta,
        surface_nphi=nphi,
        quad_ntheta=quad_nt_eff,
        quad_nphi=quad_np_eff,
    )

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
        surface_ntheta=ntheta,
        surface_nphi=nphi,
        quad_ntheta=quad_nt_eff,
        quad_nphi=quad_np_eff,
        quad_factor_theta=float(quad_nt_eff) / float(ntheta),
        quad_factor_phi=float(quad_np_eff) / float(nphi),
        grid_adequacy_status=grid_status,
        nfp=int(nfp),
        half_period=bool(half_period),
        digits=int(digits),
        patch_dim0=None if patch_dim0 is None else int(patch_dim0),
        chunk_size=chunk_size,
        target_chunk_size=target_chunk_size,
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
