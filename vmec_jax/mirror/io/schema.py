"""Schema objects for mirror-native ``mout_*.nc`` files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

MOUT_SCHEMA_VERSION = "0.1"
MOUT_ALGORITHM = "fixed_boundary_variational_chebyshev_lobatto"

MOUT_GLOBAL_ATTRIBUTES = {
    "code": "vmec_jax",
    "geometry_type": "mirror",
    "mirror_schema_version": MOUT_SCHEMA_VERSION,
    "algorithm": MOUT_ALGORITHM,
    "coordinate_order": "s,theta,xi",
    "axis": "straight",
    "fixed_boundary": "true",
    "free_boundary": "false",
    "pressure_model": "scalar_p_of_s",
}

MOUT_COORDINATE_DIMS = {
    "s": ("ns",),
    "theta": ("ntheta",),
    "xi": ("nxi",),
    "z": ("nxi",),
    "w_s": ("ns",),
    "w_theta": ("ntheta",),
    "w_xi": ("nxi",),
}

MOUT_GEOMETRY_DIMS = {
    "r": ("ns", "ntheta", "nxi"),
    "X": ("ns", "ntheta", "nxi"),
    "Y": ("ns", "ntheta", "nxi"),
    "Z": ("ns", "ntheta", "nxi"),
    "sqrtg": ("ns", "ntheta", "nxi"),
    "g_ss": ("ns", "ntheta", "nxi"),
    "g_stheta": ("ns", "ntheta", "nxi"),
    "g_sxi": ("ns", "ntheta", "nxi"),
    "g_thetatheta": ("ns", "ntheta", "nxi"),
    "g_thetaxi": ("ns", "ntheta", "nxi"),
    "g_xixi": ("ns", "ntheta", "nxi"),
    "boundary_r": ("ntheta", "nxi"),
}

MOUT_FIELD_DIMS = {
    "B_sup_s": ("ns", "ntheta", "nxi"),
    "B_sup_theta": ("ns", "ntheta", "nxi"),
    "B_sup_xi": ("ns", "ntheta", "nxi"),
    "B_cov_s": ("ns", "ntheta", "nxi"),
    "B_cov_theta": ("ns", "ntheta", "nxi"),
    "B_cov_xi": ("ns", "ntheta", "nxi"),
    "B_x": ("ns", "ntheta", "nxi"),
    "B_y": ("ns", "ntheta", "nxi"),
    "B_z": ("ns", "ntheta", "nxi"),
    "Bmag": ("ns", "ntheta", "nxi"),
    "lambda": ("ns", "ntheta", "nxi"),
}

MOUT_PROFILE_DIMS = {
    "Psi_prime": ("ns",),
    "I_prime": ("ns",),
    "pressure": ("ns",),
    "dpressure_ds": ("ns",),
    "beta": ("ns",),
}

MOUT_HISTORY_DIMS = {
    "solve_history_stage_index": ("history_steps",),
    "solve_history_iteration": ("history_steps",),
    "solve_history_pressure_scale": ("history_steps",),
    "solve_history_energy_total": ("history_steps",),
    "solve_history_residual_norm": ("history_steps",),
    "solve_history_min_sqrtg": ("history_steps",),
    "solve_history_max_sqrtg": ("history_steps",),
    "solve_history_min_Bmag": ("history_steps",),
    "solve_history_max_Bmag": ("history_steps",),
    "solve_history_mirror_ratio": ("history_steps",),
    "solve_history_step_size": ("history_steps",),
    "solve_history_accepted": ("history_steps",),
}


@dataclass(frozen=True)
class MirrorOutputGeometry:
    """Geometry arrays stored in a mirror output file."""

    r: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    sqrtg: np.ndarray
    g_ss: np.ndarray
    g_stheta: np.ndarray
    g_sxi: np.ndarray
    g_thetatheta: np.ndarray
    g_thetaxi: np.ndarray
    g_xixi: np.ndarray
    boundary_r: np.ndarray


@dataclass(frozen=True)
class MirrorOutputField:
    """Magnetic-field arrays stored in a mirror output file."""

    b_sup_s: np.ndarray
    b_sup_theta: np.ndarray
    b_sup_xi: np.ndarray
    b_cov_s: np.ndarray
    b_cov_theta: np.ndarray
    b_cov_xi: np.ndarray
    b_x: np.ndarray
    b_y: np.ndarray
    b_z: np.ndarray
    bmag: np.ndarray
    lam: np.ndarray


@dataclass(frozen=True)
class MirrorOutputProfiles:
    """Radial profile arrays stored in a mirror output file."""

    psi_prime: np.ndarray
    i_prime: np.ndarray
    pressure: np.ndarray
    dpressure_ds: np.ndarray
    beta: np.ndarray
    gamma: float


@dataclass(frozen=True)
class MirrorOutputDiagnostics:
    """Scalar diagnostics stored in a mirror output file."""

    energy_b: float
    energy_p: float
    energy_total: float
    residual_norm: float
    force_norm: float
    min_sqrtg: float
    max_sqrtg: float
    min_bmag: float
    max_bmag: float
    mirror_ratio: float


@dataclass(frozen=True)
class MirrorOutputHistory:
    """Solve-history arrays stored in a mirror output file."""

    stage_index: np.ndarray
    iteration: np.ndarray
    pressure_scale: np.ndarray
    energy_total: np.ndarray
    residual_norm: np.ndarray
    min_sqrtg: np.ndarray
    max_sqrtg: np.ndarray
    min_bmag: np.ndarray
    max_bmag: np.ndarray
    mirror_ratio: np.ndarray
    step_size: np.ndarray
    accepted: np.ndarray


@dataclass(frozen=True)
class MirrorOutput:
    """In-memory mirror-native output payload."""

    path: Path | None
    attributes: dict[str, str]
    s: np.ndarray
    theta: np.ndarray
    xi: np.ndarray
    z: np.ndarray
    w_s: np.ndarray
    w_theta: np.ndarray
    w_xi: np.ndarray
    geometry: MirrorOutputGeometry
    field: MirrorOutputField
    profiles: MirrorOutputProfiles
    diagnostics: MirrorOutputDiagnostics
    history: MirrorOutputHistory

    @property
    def ns(self) -> int:
        return int(self.s.size)

    @property
    def ntheta(self) -> int:
        return int(self.theta.size)

    @property
    def nxi(self) -> int:
        return int(self.xi.size)
