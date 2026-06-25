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


def _as_wout(wout_or_path: Any) -> Any:
    if isinstance(wout_or_path, (str, Path)):
        return read_wout(wout_or_path)
    return wout_or_path


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
