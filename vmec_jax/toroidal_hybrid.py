"""Toroidal stellarator-mirror hybrid boundary helpers.

These helpers build ordinary VMEC fixed-boundary input data.  They are not part
of the open-ended mirror coordinate system: the surface is closed and toroidal,
with weakly shaped side arcs and localized stellarator-like shaping near the
corner arcs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .boundary import BoundaryCoeffs
from .fourier import build_helical_basis, eval_fourier, project_to_modes
from .grids import AngleGrid
from .modes import vmec_mode_table
from .namelist import InData, minimal_fixed_boundary_indata
from .solvers.free_boundary.reduced_controls import ReducedControlMap, reduced_control_least_squares_step


@dataclass(frozen=True)
class ToroidalHybridBoundarySamples:
    """Real-space samples for one VMEC field period."""

    theta: np.ndarray
    zeta: np.ndarray
    R: np.ndarray
    Z: np.ndarray
    side_weight: np.ndarray
    corner_weight: np.ndarray


@dataclass(frozen=True)
class SquareAxisSplineControls:
    """Periodic spline controls for the square-axis radial envelope.

    The controls define the cylindrical major radius of the magnetic-axis
    centerline as a function of VMEC ``zeta``.  They are intentionally separate
    from ``MPOL``/``NTOR`` so geometry studies can keep one real-space target and
    project it onto different VMEC Fourier decks for convergence checks.
    """

    zeta: np.ndarray
    radius: np.ndarray

    @classmethod
    def rounded_square(
        cls,
        *,
        axis_half_width: float = 1.5,
        corner_radius_factor: float = np.sqrt(2.0),
        control_count: int = 8,
    ) -> "SquareAxisSplineControls":
        """Return uniformly spaced controls for a rounded square axis."""

        axis_half_width = float(axis_half_width)
        corner_radius_factor = float(corner_radius_factor)
        control_count = int(control_count)
        if axis_half_width <= 0.0:
            raise ValueError("axis_half_width must be positive")
        if not np.isfinite(corner_radius_factor) or corner_radius_factor <= 1.0:
            raise ValueError("corner_radius_factor must be finite and greater than one")
        if control_count < 8 or control_count % 8 != 0:
            raise ValueError("control_count must be a multiple of 8 and at least 8")
        zeta = np.linspace(0.0, 2.0 * np.pi, control_count, endpoint=False)
        corner_weight = np.sin(2.0 * zeta) ** 2
        radius = axis_half_width * (1.0 + (corner_radius_factor - 1.0) * corner_weight)
        return cls(zeta=zeta, radius=np.asarray(radius, dtype=float))

    def validate(self) -> "SquareAxisSplineControls":
        """Return normalized controls or raise for invalid input."""

        zeta_arr = np.asarray(self.zeta, dtype=float)
        radius_arr = np.asarray(self.radius, dtype=float)
        if zeta_arr.ndim != 1 or radius_arr.ndim != 1:
            raise ValueError("spline controls must be one-dimensional")
        zeta = zeta_arr.reshape(-1)
        radius = radius_arr.reshape(-1)
        if zeta.size != radius.size:
            raise ValueError("spline control zeta and radius arrays must have the same length")
        if zeta.size < 4:
            raise ValueError("at least four periodic spline controls are required")
        if not (np.all(np.isfinite(zeta)) and np.all(np.isfinite(radius))):
            raise ValueError("spline controls must be finite")
        if np.any(radius <= 0.0):
            raise ValueError("spline control radii must be positive")
        period = 2.0 * np.pi
        zeta_mod = np.mod(zeta, period)
        order = np.argsort(zeta_mod)
        zeta_sorted = zeta_mod[order]
        radius_sorted = radius[order]
        if np.any(np.diff(zeta_sorted) <= 1.0e-12):
            raise ValueError("spline control zeta nodes must be distinct modulo 2*pi")
        return SquareAxisSplineControls(zeta=zeta_sorted, radius=radius_sorted)


@dataclass(frozen=True)
class SquareAxisControlBasis:
    """Symmetry-reduced square-axis control basis.

    ``matrix @ reduced_radius`` expands a compact control vector into the full
    periodic spline-control radius vector.  This keeps production updates in a
    symmetry-preserving low-dimensional basis before any projection to VMEC
    Fourier coefficients.
    """

    controls: SquareAxisSplineControls
    symmetry: str
    labels: tuple[str, ...]
    matrix: np.ndarray

    def expand_radius(self, reduced_radius: Any) -> np.ndarray:
        """Expand reduced radii into one radius per spline control node."""

        values = np.asarray(reduced_radius, dtype=float).reshape(-1)
        if values.size != len(self.labels):
            raise ValueError("reduced_radius has the wrong length for this control basis")
        return np.asarray(self.matrix @ values, dtype=float)

    def project_radius(self, full_radius: Any) -> np.ndarray:
        """Average a full control-radius vector into the reduced basis."""

        values = np.asarray(full_radius, dtype=float).reshape(-1)
        if values.size != np.asarray(self.controls.radius).size:
            raise ValueError("full_radius has the wrong length for this control basis")
        counts = np.sum(self.matrix, axis=0)
        return np.asarray((self.matrix.T @ values) / counts, dtype=float)

    def controls_from_reduced(self, reduced_radius: Any) -> SquareAxisSplineControls:
        """Return validated spline controls from a reduced-radius vector."""

        return SquareAxisSplineControls(
            zeta=np.asarray(self.controls.zeta, dtype=float),
            radius=self.expand_radius(reduced_radius),
        ).validate()


@dataclass(frozen=True)
class SquareAxisControlProjection:
    """Least-squares fit of boundary motion to square-axis controls."""

    labels: tuple[str, ...]
    radius_delta: np.ndarray
    predicted: BoundaryCoeffs
    residual: BoundaryCoeffs
    rank: int
    singular_values: np.ndarray
    condition_number: float | None
    target_l2: float
    predicted_l2: float
    residual_l2: float
    residual_linf: float
    residual_rms: float
    residual_rel: float | None
    captured_fraction: float | None

    @property
    def radius_delta_by_label(self) -> dict[str, float]:
        """Return fitted control-radius changes keyed by label."""

        return {str(label): float(value) for label, value in zip(self.labels, self.radius_delta, strict=False)}


def _stack_boundary_coeffs(coeffs: BoundaryCoeffs) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(coeffs.R_cos, dtype=float).reshape(-1),
            np.asarray(coeffs.R_sin, dtype=float).reshape(-1),
            np.asarray(coeffs.Z_cos, dtype=float).reshape(-1),
            np.asarray(coeffs.Z_sin, dtype=float).reshape(-1),
        ]
    )


def _unstack_boundary_coeffs(values: Any, template: BoundaryCoeffs) -> BoundaryCoeffs:
    """Return boundary coefficients with the same channel shapes as ``template``."""

    vector = np.asarray(values, dtype=float).reshape(-1)
    shapes = (
        np.asarray(template.R_cos).shape,
        np.asarray(template.R_sin).shape,
        np.asarray(template.Z_cos).shape,
        np.asarray(template.Z_sin).shape,
    )
    sizes = tuple(int(np.prod(shape, dtype=int)) for shape in shapes)
    if vector.size != sum(sizes):
        raise ValueError("stacked boundary vector has incompatible size")
    stops = np.cumsum(sizes)
    starts = np.concatenate([[0], stops[:-1]])
    chunks = [vector[start:stop].reshape(shape) for start, stop, shape in zip(starts, stops, shapes, strict=True)]
    return BoundaryCoeffs(
        R_cos=chunks[0],
        R_sin=chunks[1],
        Z_cos=chunks[2],
        Z_sin=chunks[3],
    )


@dataclass(frozen=True)
class SquareAxisControlFourierMatrix:
    """Linearized VMEC boundary coefficients for square-axis control radii."""

    controls: SquareAxisSplineControls
    m: np.ndarray
    n: np.ndarray
    R_cos: np.ndarray
    R_sin: np.ndarray
    Z_cos: np.ndarray
    Z_sin: np.ndarray
    control_basis: SquareAxisControlBasis | None = None

    @property
    def control_count(self) -> int:
        """Number of spline-control variables represented by this map."""

        return int(np.asarray(self.R_cos).shape[1])

    def _control_labels(self) -> tuple[str, ...]:
        return (
            tuple(str(label) for label in self.control_basis.labels)
            if self.control_basis is not None
            else tuple(f"control_{idx}" for idx in range(self.control_count))
        )

    def boundary_delta(self, radius_delta: Any) -> BoundaryCoeffs:
        """Map a control-radius update to VMEC boundary coefficient deltas."""

        delta = np.asarray(radius_delta, dtype=float).reshape(-1)
        if delta.size != self.control_count:
            raise ValueError("radius_delta has the wrong length for this control map")
        return BoundaryCoeffs(
            R_cos=np.asarray(self.R_cos, dtype=float) @ delta,
            R_sin=np.asarray(self.R_sin, dtype=float) @ delta,
            Z_cos=np.asarray(self.Z_cos, dtype=float) @ delta,
            Z_sin=np.asarray(self.Z_sin, dtype=float) @ delta,
        )

    def stacked_jacobian(self) -> np.ndarray:
        """Return the stacked coefficient Jacobian used by reduced solvers."""

        return np.concatenate(
            [
                np.asarray(self.R_cos, dtype=float),
                np.asarray(self.R_sin, dtype=float),
                np.asarray(self.Z_cos, dtype=float),
                np.asarray(self.Z_sin, dtype=float),
            ],
            axis=0,
        )

    def reduced_control_map(
        self,
        initial_boundary: BoundaryCoeffs,
        *,
        rcond: float | None = None,
    ) -> ReducedControlMap:
        """Return the affine map from reduced controls to full boundary states."""

        initial = _stack_boundary_coeffs(initial_boundary)
        jacobian = self.stacked_jacobian()
        if jacobian.shape[0] != initial.size:
            raise ValueError("initial boundary and control map have incompatible sizes")
        return ReducedControlMap(
            initial=initial,
            jacobian=jacobian,
            labels=self._control_labels(),
            rcond=rcond,
        )

    def encode_boundary(
        self,
        boundary: BoundaryCoeffs,
        *,
        initial_boundary: BoundaryCoeffs,
        rcond: float | None = None,
        ridge: float = 0.0,
        trust_radius: float | None = None,
    ):
        """Fit a full boundary state with reduced spline-control coordinates."""

        control_map = self.reduced_control_map(initial_boundary, rcond=rcond)
        return control_map.encode(
            _stack_boundary_coeffs(boundary),
            ridge=ridge,
            trust_radius=trust_radius,
        )

    def decode_boundary(
        self,
        radius_delta: Any,
        *,
        initial_boundary: BoundaryCoeffs,
        rcond: float | None = None,
    ) -> BoundaryCoeffs:
        """Decode reduced-control coordinates into a full boundary state."""

        control_map = self.reduced_control_map(initial_boundary, rcond=rcond)
        return _unstack_boundary_coeffs(control_map.decode(radius_delta), initial_boundary)

    def project_boundary(
        self,
        boundary: BoundaryCoeffs,
        *,
        initial_boundary: BoundaryCoeffs,
        rcond: float | None = None,
        ridge: float = 0.0,
        trust_radius: float | None = None,
    ) -> BoundaryCoeffs:
        """Project a full boundary state onto the affine reduced-control map."""

        control_map = self.reduced_control_map(initial_boundary, rcond=rcond)
        projected = control_map.project(
            _stack_boundary_coeffs(boundary),
            ridge=ridge,
            trust_radius=trust_radius,
        )
        return _unstack_boundary_coeffs(projected, initial_boundary)

    def project_boundary_delta(self, delta: BoundaryCoeffs) -> SquareAxisControlProjection:
        """Fit a VMEC boundary-coefficient displacement to this control map."""

        target = _stack_boundary_coeffs(delta)
        jacobian = self.stacked_jacobian()
        if jacobian.shape[0] != target.size:
            raise ValueError("boundary delta and control map have incompatible sizes")
        if jacobian.shape[1] == 0:
            raise ValueError("control map has no control columns")

        labels = self._control_labels()
        step = reduced_control_least_squares_step(jacobian, target, labels=labels)
        predicted = self.boundary_delta(step.control_delta)
        residual = BoundaryCoeffs(
            R_cos=np.asarray(delta.R_cos, dtype=float) - np.asarray(predicted.R_cos, dtype=float),
            R_sin=np.asarray(delta.R_sin, dtype=float) - np.asarray(predicted.R_sin, dtype=float),
            Z_cos=np.asarray(delta.Z_cos, dtype=float) - np.asarray(predicted.Z_cos, dtype=float),
            Z_sin=np.asarray(delta.Z_sin, dtype=float) - np.asarray(predicted.Z_sin, dtype=float),
        )
        residual_stack = _stack_boundary_coeffs(residual)
        residual_linf = float(np.max(np.abs(residual_stack))) if residual_stack.size else 0.0
        residual_rms = float(np.sqrt(np.mean(residual_stack * residual_stack))) if residual_stack.size else 0.0
        captured_fraction = None if step.residual_rel is None else float(max(0.0, 1.0 - step.residual_rel))
        return SquareAxisControlProjection(
            labels=labels,
            radius_delta=np.asarray(step.control_delta, dtype=float),
            predicted=predicted,
            residual=residual,
            rank=int(step.rank),
            singular_values=np.asarray(step.singular_values, dtype=float),
            condition_number=step.condition_number,
            target_l2=step.target_l2,
            predicted_l2=step.predicted_l2,
            residual_l2=step.residual_l2,
            residual_linf=residual_linf,
            residual_rms=residual_rms,
            residual_rel=step.residual_rel,
            captured_fraction=captured_fraction,
        )


def _control_operator_diagnostics(jacobian: Any, *, rcond: float | None = None) -> dict[str, Any]:
    """Return dense reduced-control operator diagnostics for one Jacobian."""

    jacobian_arr = np.asarray(jacobian, dtype=float)
    if jacobian_arr.ndim != 2:
        raise ValueError("control Jacobian must be a two-dimensional array")
    singular_values = np.linalg.svd(jacobian_arr, compute_uv=False)
    finite_singular_values = singular_values[np.isfinite(singular_values)]
    min_sv = float(np.min(finite_singular_values)) if finite_singular_values.size else None
    max_sv = float(np.max(finite_singular_values)) if finite_singular_values.size else None
    if max_sv is None:
        rank_tol = np.finfo(float).eps
    elif rcond is None:
        rank_tol = max(jacobian_arr.shape) * np.finfo(float).eps * max_sv
    else:
        rank_tol = max(float(rcond) * max_sv, np.finfo(float).eps)
    rank = int(np.sum(finite_singular_values > rank_tol))
    condition = None if min_sv in (None, 0.0) or max_sv is None else float(max_sv / max(min_sv, np.finfo(float).tiny))
    gram = jacobian_arr.T @ jacobian_arr
    gram_eigenvalues = np.linalg.eigvalsh(gram) if gram.size else np.asarray([], dtype=float)
    finite_gram = gram_eigenvalues[np.isfinite(gram_eigenvalues)]
    min_gram = float(np.min(finite_gram)) if finite_gram.size else None
    max_gram = float(np.max(finite_gram)) if finite_gram.size else None
    gram_condition = (
        None
        if min_gram in (None, 0.0) or max_gram is None
        else float(max_gram / max(min_gram, np.finfo(float).tiny))
    )
    column_norms = np.linalg.norm(jacobian_arr, axis=0)
    valid = column_norms > np.finfo(float).tiny
    max_corr = None
    if int(jacobian_arr.shape[1]) > 1 and np.count_nonzero(valid) > 1:
        normalized = jacobian_arr[:, valid] / column_norms[valid]
        corr = np.abs(normalized.T @ normalized)
        max_corr = float(np.max(corr[np.triu_indices_from(corr, k=1)]))
    return {
        "rank": rank,
        "rank_tolerance": float(rank_tol),
        "rank_deficient": bool(rank < int(jacobian_arr.shape[1])),
        "singular_values": [float(value) for value in singular_values],
        "condition_number": condition,
        "gram_eigenvalues": [float(value) for value in gram_eigenvalues],
        "gram_condition_number": gram_condition,
        "column_norms": [float(value) for value in column_norms],
        "max_offdiag_column_correlation": max_corr,
        "native_reduced_solver_ready": bool(rank == int(jacobian_arr.shape[1]) and condition is not None),
    }


def square_axis_spline_control_fourier_map_status(
    *,
    controls: SquareAxisSplineControls | None = None,
    control_basis: SquareAxisControlBasis | None = None,
    symmetry: str = "square",
    nfp: int = 1,
    mpol: int = 5,
    ntor: int = 28,
    ntheta_fit: int = 64,
    nzeta_fit: int = 224,
    **sample_kwargs: Any,
) -> dict[str, Any]:
    """Return conditioning diagnostics for a spline-control boundary map.

    This summarizes the chain-rule map from square-axis spline radii to VMEC
    Fourier boundary coefficients.  It is a representation diagnostic: good
    conditioning here says the reduced control layer is numerically usable, but
    it does not by itself imply nonlinear VMEC convergence.
    """

    basis = (
        control_basis
        if control_basis is not None
        else square_axis_spline_symmetric_control_basis(controls, symmetry=symmetry)
    )
    matrix = square_axis_spline_control_fourier_matrix(
        control_basis=basis,
        nfp=int(nfp),
        mpol=int(mpol),
        ntor=int(ntor),
        ntheta_fit=int(ntheta_fit),
        nzeta_fit=int(nzeta_fit),
        **sample_kwargs,
    )
    jacobian = matrix.stacked_jacobian()
    operator = _control_operator_diagnostics(jacobian)
    return {
        "status": "available",
        "basis_symmetry": basis.symmetry,
        "labels": list(basis.labels),
        "control_count": int(matrix.control_count),
        "mode_count": int(np.asarray(matrix.m).size),
        "jacobian_shape": [int(value) for value in jacobian.shape],
        **operator,
    }


def square_axis_free_boundary_edge_control_projection_payload(
    *,
    controls: SquareAxisSplineControls | None = None,
    symmetry: str = "square",
    rcond: float = 1.0e-12,
    ridge: float = 0.0,
    trust_radius: float | None = None,
    native_force_metric: str = "pullback",
    source: str = "square_axis_free_boundary_edge_control_projection",
    nfp: int = 1,
    mpol: int = 5,
    ntor: int = 28,
    ntheta_fit: int = 64,
    nzeta_fit: int = 224,
    **sample_kwargs: Any,
) -> dict[str, Any] | None:
    """Return the solver payload for reduced square-axis LCFS edge controls.

    The free-boundary solver still stores the LCFS in VMEC Fourier
    coefficients.  This payload adds a reduced control layer by projecting edge
    updates onto the spline-control Jacobian for the requested symmetry.
    """

    symmetry_key = str(symmetry).strip().lower()
    if symmetry_key in {"", "none", "off", "false"}:
        return None
    if symmetry_key not in {"square", "stellarator", "full"}:
        raise ValueError("symmetry must be 'square', 'stellarator', 'full', or 'none'")
    rcond_value = float(rcond)
    if not np.isfinite(rcond_value) or rcond_value <= 0.0:
        raise ValueError("rcond must be positive and finite")
    ridge_value = float(ridge)
    if not np.isfinite(ridge_value) or ridge_value < 0.0:
        raise ValueError("ridge must be finite and nonnegative")
    trust_value = None if trust_radius is None else float(trust_radius)
    if trust_value is not None and (not np.isfinite(trust_value) or trust_value <= 0.0):
        raise ValueError("trust_radius must be positive and finite when supplied")
    force_metric = str(native_force_metric).strip().lower()
    if force_metric in {"", "default", "pullback", "adjoint", "gradient", "jtf", "j.t"}:
        force_metric = "pullback"
    elif force_metric in {"least_squares", "least-squares", "ls", "coordinate", "pinv", "pseudoinverse"}:
        force_metric = "least_squares"
    else:
        raise ValueError("native_force_metric must be 'pullback' or 'least_squares'")
    basis = square_axis_spline_symmetric_control_basis(controls, symmetry=symmetry_key)
    matrix = square_axis_spline_control_fourier_matrix(
        control_basis=basis,
        nfp=int(nfp),
        mpol=int(mpol),
        ntor=int(ntor),
        ntheta_fit=int(ntheta_fit),
        nzeta_fit=int(nzeta_fit),
        **sample_kwargs,
    )
    jacobian = np.asarray(matrix.stacked_jacobian(), dtype=float)
    if jacobian.ndim != 2 or jacobian.shape[1] <= 0:
        raise ValueError(f"empty edge-control Jacobian for symmetry {symmetry_key!r}")
    operator = _control_operator_diagnostics(jacobian, rcond=rcond_value)
    return {
        "enabled": True,
        "source": str(source),
        "basis_symmetry": basis.symmetry,
        "labels": list(basis.labels),
        "control_jacobian": jacobian,
        "control_count": int(jacobian.shape[1]),
        "mode_count": int(np.asarray(matrix.m).size),
        "rcond": rcond_value,
        "ridge": ridge_value,
        "trust_radius": trust_value,
        "native_force_metric": force_metric,
        "rank": operator["rank"],
        "rank_deficient": operator["rank_deficient"],
        "condition_number": operator["condition_number"],
        "gram_condition_number": operator["gram_condition_number"],
        "max_offdiag_column_correlation": operator["max_offdiag_column_correlation"],
        "native_reduced_solver_ready": operator["native_reduced_solver_ready"],
    }


def _periodic_angle_distance(a: Any, b: Any) -> np.ndarray:
    return np.abs((np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi)


def _angle_match_index(nodes: np.ndarray, target: float, *, tol: float) -> int:
    distances = _periodic_angle_distance(nodes, float(target))
    index = int(np.argmin(distances))
    if float(distances[index]) > float(tol):
        raise ValueError("spline control nodes are missing a required symmetry counterpart")
    return index


def square_axis_spline_symmetric_control_basis(
    controls: SquareAxisSplineControls | None = None,
    *,
    symmetry: str = "square",
    angle_tol: float = 1.0e-10,
) -> SquareAxisControlBasis:
    """Return a symmetry-preserving reduced basis for square-axis controls.

    ``symmetry="full"`` leaves every spline radius as an independent control.
    ``symmetry="stellarator"`` enforces the usual even-radius condition
    ``r(zeta) = r(-zeta)``.  ``symmetry="square"`` additionally groups nodes
    related by quarter-turn rotations; the eight-control low-level default
    reduces to two parameters, while the root square-coil example's 16 controls
    reduce to three.  The returned basis is a dense
    expansion matrix, so it can be used directly in chain-rule and optimization
    code without changing the current VMEC Fourier boundary interface.
    """

    validated = (
        controls
        if controls is not None
        else SquareAxisSplineControls.rounded_square(axis_half_width=1.5, corner_radius_factor=1.14)
    ).validate()
    symmetry_key = str(symmetry).strip().lower()
    if symmetry_key in {"stellsym", "stellarator_symmetry"}:
        symmetry_key = "stellarator"
    if symmetry_key in {"fourfold", "dihedral", "d4"}:
        symmetry_key = "square"
    if symmetry_key in {"identity", "unreduced", "all_controls"}:
        symmetry_key = "full"
    if symmetry_key not in {"full", "stellarator", "square"}:
        raise ValueError("symmetry must be 'full', 'stellarator', or 'square'")
    if not np.isfinite(float(angle_tol)) or float(angle_tol) <= 0.0:
        raise ValueError("angle_tol must be positive and finite")

    zeta = np.asarray(validated.zeta, dtype=float)
    n_control = int(zeta.size)
    if symmetry_key == "full":
        return SquareAxisControlBasis(
            controls=validated,
            symmetry="full",
            labels=tuple(f"control_{idx:02d}" for idx in range(n_control)),
            matrix=np.eye(n_control, dtype=float),
        )
    unused = set(range(n_control))
    orbits: list[list[int]] = []
    while unused:
        seed = min(unused)
        seed_angle = float(zeta[seed])
        if symmetry_key == "stellarator":
            targets = (seed_angle, -seed_angle)
        else:
            turns = 0.5 * np.pi * np.arange(4, dtype=float)
            targets = tuple(seed_angle + turns) + tuple(-seed_angle + turns)
        orbit = sorted({_angle_match_index(zeta, target, tol=float(angle_tol)) for target in targets})
        orbits.append(orbit)
        unused.difference_update(orbit)

    matrix = np.zeros((n_control, len(orbits)), dtype=float)
    labels: list[str] = []
    for col, orbit in enumerate(orbits):
        matrix[orbit, col] = 1.0
        representative = float(np.min(np.mod(zeta[orbit], 2.0 * np.pi)))
        if symmetry_key == "square" and np.isclose(np.mod(2.0 * representative, np.pi), 0.0, atol=angle_tol):
            label = "side"
        elif symmetry_key == "square" and np.isclose(
            np.mod(2.0 * representative - 0.5 * np.pi, np.pi), 0.0, atol=angle_tol
        ):
            label = "corner"
        else:
            label = f"{symmetry_key}_orbit_{col}"
        labels.append(label if label not in labels else f"{label}_{col}")
    return SquareAxisControlBasis(
        controls=validated,
        symmetry=symmetry_key,
        labels=tuple(labels),
        matrix=matrix,
    )


def _periodic_cubic_hermite_interpolate(x_nodes: Any, y_nodes: Any, x_eval: Any) -> np.ndarray:
    """Evaluate a periodic cubic Hermite interpolant on a full-period grid."""

    controls = SquareAxisSplineControls(
        zeta=np.asarray(x_nodes, dtype=float),
        radius=np.asarray(y_nodes, dtype=float),
    ).validate()
    x = np.asarray(x_eval, dtype=float)
    x_nodes = np.asarray(controls.zeta, dtype=float)
    y_nodes = np.asarray(controls.radius, dtype=float)
    period = 2.0 * np.pi
    n = x_nodes.size
    x_ext = np.concatenate([x_nodes[-1:] - period, x_nodes, x_nodes[:1] + period])
    y_ext = np.concatenate([y_nodes[-1:], y_nodes, y_nodes[:1]])
    slopes = np.empty(n, dtype=float)
    for idx in range(n):
        slopes[idx] = (y_ext[idx + 2] - y_ext[idx]) / (x_ext[idx + 2] - x_ext[idx])

    x_mod = np.mod(x, period)
    interval = np.searchsorted(x_nodes, x_mod, side="right") - 1
    interval = np.where(interval < 0, n - 1, interval)
    next_interval = (interval + 1) % n
    x0 = x_nodes[interval]
    x1 = x_nodes[next_interval]
    wrap = next_interval == 0
    x1 = np.where(wrap, x1 + period, x1)
    x_use = np.where(wrap & (x_mod < x0), x_mod + period, x_mod)
    h = x1 - x0
    t = (x_use - x0) / h
    y0 = y_nodes[interval]
    y1 = y_nodes[next_interval]
    m0 = slopes[interval]
    m1 = slopes[next_interval]
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    return np.asarray(h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1, dtype=float)


def _periodic_trigonometric_interpolate(
    controls: SquareAxisSplineControls,
    x_eval: Any,
) -> np.ndarray | None:
    """Evaluate uniformly spaced controls as a low-bandwidth periodic series."""

    controls = controls.validate()
    x_nodes = np.asarray(controls.zeta, dtype=float)
    y_nodes = np.asarray(controls.radius, dtype=float)
    n = int(x_nodes.size)
    period = 2.0 * np.pi
    spacing = period / float(n)
    offsets = np.mod(x_nodes - x_nodes[0], period)
    expected = spacing * np.arange(n, dtype=float)
    if not np.allclose(offsets, expected, rtol=1.0e-12, atol=1.0e-12):
        return None

    x = np.asarray(x_eval, dtype=float)
    x_rel = np.mod(x - x_nodes[0], period)
    coeffs = np.fft.rfft(y_nodes) / float(n)
    out = np.full_like(x_rel, float(np.real(coeffs[0])), dtype=float)
    for harmonic in range(1, coeffs.size):
        coeff = coeffs[harmonic]
        scale = 1.0 if n % 2 == 0 and harmonic == n // 2 else 2.0
        out += scale * (
            float(np.real(coeff)) * np.cos(float(harmonic) * x_rel)
            - float(np.imag(coeff)) * np.sin(float(harmonic) * x_rel)
        )
    return np.asarray(out, dtype=float)


def square_axis_spline_radius(zeta: Any, controls: SquareAxisSplineControls) -> np.ndarray:
    """Evaluate a periodic square-axis spline radius at VMEC ``zeta`` nodes."""

    validated = controls.validate()
    trigonometric = _periodic_trigonometric_interpolate(validated, zeta)
    if trigonometric is not None:
        return trigonometric
    return _periodic_cubic_hermite_interpolate(validated.zeta, validated.radius, zeta)


def square_axis_spline_radius_matrix(zeta: Any, controls: SquareAxisSplineControls) -> np.ndarray:
    """Return the linear map from spline control radii to sampled axis radius.

    The square-axis bridge is linear in the control radii for fixed control
    locations.  Exposing that map makes the low-dimensional controls explicit:
    ``square_axis_spline_radius(zeta, controls)`` is equivalent to
    ``matrix @ controls.radius`` after flattening ``zeta``.  This is a small
    building block for differentiable control studies without changing the
    current VMEC/VMEC2000 Fourier boundary interface.
    """

    validated = controls.validate()
    zeta_arr = np.asarray(zeta, dtype=float)
    flat_zeta = zeta_arr.reshape(-1)
    n_control = int(np.asarray(validated.radius).size)
    baseline = SquareAxisSplineControls(
        zeta=np.asarray(validated.zeta, dtype=float),
        radius=np.ones(n_control, dtype=float),
    )
    baseline_values = square_axis_spline_radius(flat_zeta, baseline)
    columns = []
    for idx in range(n_control):
        radius = np.ones(n_control, dtype=float)
        radius[idx] += 1.0
        perturbed = SquareAxisSplineControls(zeta=np.asarray(validated.zeta, dtype=float), radius=radius)
        columns.append(square_axis_spline_radius(flat_zeta, perturbed) - baseline_values)
    matrix = np.stack(columns, axis=-1)
    return matrix.reshape(zeta_arr.shape + (n_control,))


def square_axis_spline_control_fourier_matrix(
    *,
    controls: SquareAxisSplineControls | None = None,
    control_basis: SquareAxisControlBasis | None = None,
    nfp: int = 1,
    mpol: int = 5,
    ntor: int = 28,
    ntheta_fit: int = 64,
    nzeta_fit: int = 224,
    **sample_kwargs: Any,
) -> SquareAxisControlFourierMatrix:
    """Return the chain-rule map from axis controls to VMEC boundary modes.

    The active solver path still stores the boundary as VMEC Fourier
    coefficients.  This helper makes the preceding control layer explicit by
    differentiating the projected coefficients with respect to the
    low-dimensional square-axis radii.  It is linear for fixed control
    locations and fixed local cross-section shaping.
    """

    if control_basis is not None:
        controls = control_basis.controls.validate()
    else:
        controls = (
            controls
            if controls is not None
            else SquareAxisSplineControls.rounded_square(
                axis_half_width=float(sample_kwargs.get("axis_half_width", 1.5)),
                corner_radius_factor=float(sample_kwargs.get("axis_spline_corner_radius_factor", np.sqrt(2.0))),
            )
        ).validate()
    modes = vmec_mode_table(mpol=int(mpol), ntor=int(ntor))
    theta = np.linspace(0.0, 2.0 * np.pi, int(ntheta_fit), endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, int(nzeta_fit), endpoint=False)
    grid = AngleGrid(theta=theta, zeta=zeta, nfp=int(nfp))
    basis = build_helical_basis(modes, grid)

    def _project(control_radii: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        local_controls = SquareAxisSplineControls(zeta=controls.zeta, radius=np.asarray(control_radii, dtype=float))
        samples = sample_square_axis_stellarator_mirror_hybrid_boundary(
            ntheta=int(ntheta_fit),
            nzeta=int(nzeta_fit),
            axis_kind="control_spline",
            axis_spline_controls=local_controls,
            **sample_kwargs,
        )
        r_cos, r_sin = project_to_modes(samples.R, basis)
        z_cos, z_sin = project_to_modes(samples.Z, basis)
        return (
            np.asarray(r_cos, dtype=float),
            np.asarray(r_sin, dtype=float),
            np.asarray(z_cos, dtype=float),
            np.asarray(z_sin, dtype=float),
        )

    base = np.asarray(controls.radius, dtype=float)
    if control_basis is not None:
        basis_matrix = np.asarray(control_basis.matrix, dtype=float)
        reduced_base = control_basis.project_radius(base)
        base = control_basis.expand_radius(reduced_base)
        columns_in_radius_space = [basis_matrix[:, idx] for idx in range(basis_matrix.shape[1])]
    else:
        columns_in_radius_space = [np.eye(base.size, dtype=float)[idx] for idx in range(base.size)]
    base_coeffs = _project(base)
    columns = []
    for radius_delta in columns_in_radius_space:
        perturbed = base + np.asarray(radius_delta, dtype=float)
        columns.append(tuple(new - old for new, old in zip(_project(perturbed), base_coeffs, strict=True)))

    stacked = [np.stack([column[item] for column in columns], axis=-1) for item in range(4)]
    return SquareAxisControlFourierMatrix(
        controls=controls,
        m=np.asarray(modes.m, dtype=int),
        n=np.asarray(modes.n, dtype=int),
        R_cos=stacked[0],
        R_sin=stacked[1],
        Z_cos=stacked[2],
        Z_sin=stacked[3],
        control_basis=control_basis,
    )


def recommended_square_axis_nzeta(ntor: int, *, margin: int = 8, block: int = 8) -> int:
    """Return a conservative toroidal grid size for square-axis hybrids.

    The square-axis surface has localized side/corner structure, so VMEC runs
    are much less fragile when the toroidal collocation grid has room beyond
    the largest retained Fourier mode.  The result is rounded up to a small
    block size so CLI and VMEC2000 comparisons use reproducible grids.
    """

    ntor = int(ntor)
    margin = int(margin)
    block = int(block)
    if ntor < 0:
        raise ValueError("ntor must be nonnegative")
    if margin < 0:
        raise ValueError("margin must be nonnegative")
    if block <= 0:
        raise ValueError("block must be positive")
    raw = max(16, 2 * ntor + margin)
    return int(block * np.ceil(raw / block))


def recommended_square_axis_ntheta(mpol: int, *, oversample: int = 4, floor: int = 64, block: int = 8) -> int:
    """Return a conservative VMEC poloidal grid size for square-axis hybrids.

    The square-axis examples use a smooth real-space target and then project it
    onto VMEC Fourier modes. Keeping the solve grid at least as resolved as the
    projection grid avoids a common failure mode where a higher ``MPOL`` deck is
    evaluated on VMEC's small automatic poloidal grid.
    """

    mpol = int(mpol)
    oversample = int(oversample)
    floor = int(floor)
    block = int(block)
    if mpol < 0:
        raise ValueError("mpol must be nonnegative")
    if oversample <= 0:
        raise ValueError("oversample must be positive")
    if floor <= 0:
        raise ValueError("floor must be positive")
    if block <= 0:
        raise ValueError("block must be positive")
    raw = max(floor, oversample * mpol)
    return int(block * np.ceil(raw / block))


def _square_axis_mode_count(mpol: int, ntor: int) -> int:
    return int(np.asarray(vmec_mode_table(mpol=int(mpol), ntor=int(ntor)).m).size)


def recommend_square_axis_stellarator_mirror_hybrid_resolution(
    *,
    target_max_component_error: float = 5.0e-5,
    mpol: int = 5,
    ntor: int = 12,
    max_mpol: int | None = None,
    max_ntor: int | None = None,
    **sample_kwargs: Any,
) -> dict[str, Any]:
    """Recommend a finite Fourier deck for a spline-smoothed square axis.

    The square-hybrid geometry is sampled as a smooth real-space target and
    then projected to ordinary VMEC boundary Fourier coefficients. This helper
    scans a small ``MPOL``/``NTOR`` ladder and returns the lowest estimated-cost
    candidate whose projection error is below ``target_max_component_error``.
    It does not claim nonlinear convergence; it only checks that the requested
    boundary is represented well enough before the VMEC/free-boundary solve.
    """

    target = float(target_max_component_error)
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError("target_max_component_error must be positive and finite")
    mpol0 = max(3, int(mpol))
    ntor0 = max(4, int(ntor))
    max_mpol_i = max(mpol0, int(max_mpol) if max_mpol is not None else max(8, mpol0 + 2))
    max_ntor_i = max(ntor0, int(max_ntor) if max_ntor is not None else max(32, ntor0 + 8))

    candidates: list[dict[str, Any]] = []
    for mpol_i in range(mpol0, max_mpol_i + 1):
        for ntor_i in range(ntor0, max_ntor_i + 1):
            projection = square_axis_stellarator_mirror_hybrid_projection_error(
                mpol=mpol_i,
                ntor=ntor_i,
                ntheta_fit=max(64, 4 * mpol_i),
                nzeta_fit=max(128, 8 * ntor_i),
                **sample_kwargs,
            )
            max_error = float(projection["max_abs_component_error"])
            candidate = {
                "mpol": mpol_i,
                "ntor": ntor_i,
                "recommended_nzeta": recommended_square_axis_nzeta(ntor_i),
                "mode_count": _square_axis_mode_count(mpol_i, ntor_i),
                "max_abs_component_error": max_error,
                "max_abs_component_error_rel": float(projection["max_abs_component_error_rel"]),
                "meets_target": bool(max_error <= target),
            }
            candidates.append(candidate)

    if not candidates:
        raise RuntimeError("resolution scan produced no candidates")
    best_error = min(candidates, key=lambda item: float(item["max_abs_component_error"]))
    feasible = [item for item in candidates if bool(item["meets_target"])]
    recommended = (
        min(
            feasible,
            key=lambda item: (
                int(item["mode_count"]),
                int(item["recommended_nzeta"]),
                float(item["max_abs_component_error"]),
            ),
        )
        if feasible
        else best_error
    )
    return {
        "target_max_component_error": target,
        "status": "met" if feasible else "not_met",
        "candidate_count": len(candidates),
        "recommended": recommended,
        "best_error": best_error,
    }


def sample_toroidal_stellarator_mirror_hybrid_boundary(
    *,
    ntheta: int = 64,
    nzeta: int = 64,
    major_radius: float = 1.15,
    minor_radius: float = 0.18,
    axis_oval: float = 0.10,
    side_minor_modulation: float = 0.10,
    side_elongation: float = 0.28,
    corner_amplitude: float = 0.035,
    corner_helicity: int = 1,
    corner_ellipticity: float = 0.18,
    corner_rotation: float = 0.35,
    side_power: float = 1.0,
    corner_power: float = 1.0,
) -> ToroidalHybridBoundarySamples:
    """Sample a toroidal hybrid LCFS over one field period.

    The side arcs, at ``zeta = 0`` and ``pi``, are nearly axisymmetric elongated
    cross sections.  The corner arcs, at ``zeta = pi/2`` and ``3*pi/2``, carry a
    localized finite-mode rotating ellipse plus a small optional ``m=2``
    helical perturbation.  ``side_power`` and ``corner_power`` sharpen or
    broaden those two regions without moving their centers.  The formula is
    stellarator symmetric, so it can be stored with the usual VMEC ``RBC``/``ZBS``
    boundary coefficients.
    """
    ntheta = int(ntheta)
    nzeta = int(nzeta)
    if ntheta < 8 or nzeta < 8:
        raise ValueError("ntheta and nzeta must be at least 8")
    if minor_radius <= 0.0 or major_radius <= minor_radius:
        raise ValueError("major_radius must exceed positive minor_radius")
    if int(corner_helicity) < 0:
        raise ValueError("corner_helicity must be nonnegative")
    corner_ellipticity = float(corner_ellipticity)
    corner_rotation = float(corner_rotation)
    if not np.isfinite(corner_ellipticity) or not (0.0 <= corner_ellipticity < 0.95):
        raise ValueError("corner_ellipticity must be finite and satisfy 0 <= corner_ellipticity < 0.95")
    if not np.isfinite(corner_rotation):
        raise ValueError("corner_rotation must be finite")
    side_power = float(side_power)
    corner_power = float(corner_power)
    if not np.isfinite(side_power) or side_power <= 0.0:
        raise ValueError("side_power must be finite and positive")
    if not np.isfinite(corner_power) or corner_power <= 0.0:
        raise ValueError("corner_power must be finite and positive")

    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, nzeta, endpoint=False)
    theta2, zeta2 = np.meshgrid(theta, zeta, indexing="ij")

    side_weight = np.clip(np.cos(zeta2) ** 2, 0.0, 1.0) ** side_power
    corner_weight = np.clip(np.sin(zeta2) ** 2, 0.0, 1.0) ** corner_power
    axis = float(major_radius) + float(axis_oval) * np.cos(2.0 * zeta2)
    side_minor = float(minor_radius) * (1.0 + float(side_minor_modulation) * side_weight)
    elongation = 1.0 + float(side_elongation) * side_weight
    corner_shape = corner_ellipticity * corner_weight
    radial_semiaxis = side_minor * (1.0 + corner_shape)
    vertical_semiaxis = side_minor * elongation * (1.0 - corner_shape)
    rotation_harmonic = int(corner_helicity)
    corner_tilt = corner_rotation * corner_weight * np.sin(float(rotation_harmonic) * zeta2)
    corner_phase = 2.0 * theta2 - float(int(corner_helicity)) * zeta2

    R = (
        axis
        + radial_semiaxis * np.cos(theta2)
        - vertical_semiaxis * corner_tilt * np.sin(theta2)
        + float(corner_amplitude) * corner_weight * np.cos(corner_phase)
    )
    Z = (
        radial_semiaxis * corner_tilt * np.cos(theta2)
        + vertical_semiaxis * np.sin(theta2)
        + float(corner_amplitude) * corner_weight * np.sin(corner_phase)
    )

    if float(np.min(R)) <= 0.0:
        raise ValueError("boundary has nonpositive cylindrical R; reduce minor_radius or shaping amplitudes")

    return ToroidalHybridBoundarySamples(
        theta=theta,
        zeta=zeta,
        R=np.asarray(R, dtype=float),
        Z=np.asarray(Z, dtype=float),
        side_weight=np.asarray(side_weight, dtype=float),
        corner_weight=np.asarray(corner_weight, dtype=float),
    )


def sample_square_axis_stellarator_mirror_hybrid_boundary(
    *,
    ntheta: int = 64,
    nzeta: int = 128,
    axis_half_width: float = 1.5,
    axis_kind: str = "superellipse",
    axis_square_power: float = 5.0,
    axis_spline_corner_radius_factor: float = np.sqrt(2.0),
    axis_spline_controls: SquareAxisSplineControls | None = None,
    minor_radius: float = 0.10,
    side_minor_modulation: float = 0.08,
    side_elongation: float = 0.25,
    corner_amplitude: float = 0.020,
    corner_helicity: int = 1,
    corner_ellipticity: float = 0.16,
    corner_rotation: float = 0.30,
    side_power: float = 1.0,
    corner_power: float = 1.0,
) -> ToroidalHybridBoundarySamples:
    """Sample a toroidal stellarator-mirror LCFS around a square-like axis.

    The magnetic axis is represented in polar form. ``axis_kind="superellipse"``
    keeps the original smooth polar superellipse. ``axis_kind="spline"`` uses a
    lower-bandwidth rounded-square envelope through side and corner radii.
    ``axis_kind="control_spline"`` evaluates explicit periodic spline controls.
    In all cases the surface is still stored in normal VMEC cylindrical
    coordinates, so the final equilibrium can use the ordinary toroidal
    fixed/free-boundary solver path.
    """

    ntheta = int(ntheta)
    nzeta = int(nzeta)
    if ntheta < 8 or nzeta < 16:
        raise ValueError("ntheta must be >= 8 and nzeta must be >= 16")
    if axis_half_width <= 0.0:
        raise ValueError("axis_half_width must be positive")
    axis_kind = str(axis_kind).strip().lower()
    control_kinds = {"control_spline", "spline_controls", "periodic_spline"}
    if axis_kind not in {
        "superellipse",
        "spline",
        "spline_rounded_square",
        "rounded_square_spline",
        *control_kinds,
    }:
        raise ValueError("axis_kind must be 'superellipse', 'spline', or 'control_spline'")
    if axis_kind == "superellipse" and axis_square_power <= 2.0:
        raise ValueError("axis_square_power must exceed 2 for a square-like axis")
    axis_spline_corner_radius_factor = float(axis_spline_corner_radius_factor)
    if not np.isfinite(axis_spline_corner_radius_factor) or axis_spline_corner_radius_factor <= 1.0:
        raise ValueError("axis_spline_corner_radius_factor must be finite and greater than one")
    if minor_radius <= 0.0:
        raise ValueError("minor_radius must be positive")
    if int(corner_helicity) < 0:
        raise ValueError("corner_helicity must be nonnegative")

    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, nzeta, endpoint=False)
    theta2, zeta2 = np.meshgrid(theta, zeta, indexing="ij")

    if axis_kind in control_kinds:
        controls = (
            axis_spline_controls
            if axis_spline_controls is not None
            else SquareAxisSplineControls.rounded_square(
                axis_half_width=float(axis_half_width),
                corner_radius_factor=float(axis_spline_corner_radius_factor),
            )
        )
        axis_r = square_axis_spline_radius(zeta, controls)
    elif axis_kind == "superellipse":
        c = np.cos(zeta)
        s = np.sin(zeta)
        axis_r = float(axis_half_width) / np.maximum(
            np.abs(c) ** float(axis_square_power) + np.abs(s) ** float(axis_square_power),
            np.finfo(float).tiny,
        ) ** (1.0 / float(axis_square_power))
    else:
        side_radius = float(axis_half_width)
        corner_boost = axis_spline_corner_radius_factor - 1.0
        # A single smooth fourfold envelope reaches its maximum on the rounded
        # corners and its minimum at side centers.  It deliberately avoids the
        # absolute-value cusp and high-mode tail of a sharp polar square.
        corner_profile = np.sin(2.0 * zeta) ** 2
        axis_r = side_radius * (1.0 + corner_boost * corner_profile)

    side_seed = 0.5 * (1.0 + np.cos(4.0 * zeta))
    side_weight_1d = np.clip(side_seed, 0.0, 1.0) ** float(side_power)
    corner_weight_1d = np.clip(1.0 - side_seed, 0.0, 1.0) ** float(corner_power)
    side_weight = np.broadcast_to(side_weight_1d[None, :], (ntheta, nzeta))
    corner_weight = np.broadcast_to(corner_weight_1d[None, :], (ntheta, nzeta))

    minor = float(minor_radius) * (1.0 + float(side_minor_modulation) * side_weight_1d)
    radial_semiaxis = minor * (1.0 + float(corner_ellipticity) * corner_weight_1d)
    vertical_semiaxis = minor * (1.0 + float(side_elongation) * side_weight_1d) * (
        1.0 - 0.5 * float(corner_ellipticity) * corner_weight_1d
    )
    tilt = float(corner_rotation) * corner_weight_1d * np.sin(float(int(corner_helicity)) * zeta)
    phase = 2.0 * theta2 - float(int(corner_helicity)) * zeta2
    local_r = radial_semiaxis[None, :] * np.cos(theta2)
    local_z = vertical_semiaxis[None, :] * np.sin(theta2)
    local_r = local_r + float(corner_amplitude) * corner_weight_1d[None, :] * np.cos(phase)
    local_z = local_z + float(corner_amplitude) * corner_weight_1d[None, :] * np.sin(phase)
    R = axis_r[None, :] + local_r * np.cos(tilt)[None, :] - local_z * np.sin(tilt)[None, :]
    Z = local_r * np.sin(tilt)[None, :] + local_z * np.cos(tilt)[None, :]
    if float(np.min(R)) <= 0.0:
        raise ValueError("boundary has nonpositive cylindrical R; reduce minor_radius or increase axis_half_width")

    return ToroidalHybridBoundarySamples(
        theta=theta,
        zeta=zeta,
        R=np.asarray(R, dtype=float),
        Z=np.asarray(Z, dtype=float),
        side_weight=np.asarray(side_weight, dtype=float),
        corner_weight=np.asarray(corner_weight, dtype=float),
    )


def _coeff_map_from_modes(
    values: np.ndarray, modes, *, coeff_tol: float, keep_00: bool = False
) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for m_i, n_i, value in zip(
        np.asarray(modes.m, dtype=int), np.asarray(modes.n, dtype=int), np.asarray(values, dtype=float)
    ):
        if abs(float(value)) <= float(coeff_tol) and not (keep_00 and int(m_i) == 0 and int(n_i) == 0):
            continue
        out[(int(n_i), int(m_i))] = float(value)
    return out


def _indata_from_boundary_samples(
    *,
    samples: ToroidalHybridBoundarySamples,
    nfp: int,
    mpol: int,
    ntor: int,
    ns_array: int | list[int],
    niter_array: int | list[int],
    ftol_array: float | list[float],
    phiedge: float,
    coeff_tol: float,
) -> InData:
    modes = vmec_mode_table(mpol=mpol, ntor=ntor)
    grid = AngleGrid(theta=samples.theta, zeta=samples.zeta, nfp=nfp)
    basis = build_helical_basis(modes, grid)
    r_cos, r_sin = project_to_modes(samples.R, basis)
    z_cos, z_sin = project_to_modes(samples.Z, basis)
    r_cos = np.asarray(r_cos, dtype=float)
    r_sin = np.asarray(r_sin, dtype=float)
    z_cos = np.asarray(z_cos, dtype=float)
    z_sin = np.asarray(z_sin, dtype=float)

    rbs = _coeff_map_from_modes(r_sin, modes, coeff_tol=coeff_tol)
    zbc = _coeff_map_from_modes(z_cos, modes, coeff_tol=coeff_tol)
    if rbs or zbc:
        raise ValueError("sampled hybrid boundary is not stellarator symmetric at the requested tolerance")

    indata = minimal_fixed_boundary_indata(
        nfp=nfp,
        mpol=mpol,
        ntor=ntor,
        ns_array=ns_array,
        niter_array=niter_array,
        ftol_array=ftol_array,
        phiedge=phiedge,
    )
    indata.scalars.update(
        {
            "NFP": nfp,
            "MPOL": mpol,
            "NTOR": ntor,
            "LASYM": False,
            "PHIEDGE": float(phiedge),
            "NS_ARRAY": ns_array if isinstance(ns_array, list) else int(ns_array),
            "NITER_ARRAY": niter_array if isinstance(niter_array, list) else int(niter_array),
            "FTOL_ARRAY": ftol_array if isinstance(ftol_array, list) else float(ftol_array),
        }
    )
    indata.indexed = {
        "RBC": _coeff_map_from_modes(r_cos, modes, coeff_tol=coeff_tol, keep_00=True),
        "ZBS": _coeff_map_from_modes(z_sin, modes, coeff_tol=coeff_tol),
    }
    return indata


def toroidal_stellarator_mirror_hybrid_indata(
    *,
    nfp: int = 2,
    mpol: int = 5,
    ntor: int = 4,
    ntheta_fit: int = 64,
    nzeta_fit: int = 64,
    ns_array: int | list[int] = 15,
    niter_array: int | list[int] = 80,
    ftol_array: float | list[float] = 1.0e-9,
    phiedge: float = 0.05,
    coeff_tol: float = 1.0e-12,
    **sample_kwargs: Any,
) -> InData:
    """Return VMEC ``InData`` for the toroidal hybrid boundary.

    The boundary is sampled on a uniform tensor grid and projected onto the
    standard VMEC helical modes.  Defaults keep only low-order modes so the
    input remains small and useful for low-resolution solver smoke tests.
    """
    nfp = int(nfp)
    mpol = int(mpol)
    ntor = int(ntor)
    if nfp <= 0:
        raise ValueError("nfp must be positive")
    if mpol < 3:
        raise ValueError("mpol must be at least 3 so the corner m=2 shaping fits")
    corner_helicity = int(sample_kwargs.get("corner_helicity", 1))
    if ntor < corner_helicity + 2:
        raise ValueError("ntor must be at least corner_helicity + 2 to fit the localized corner shaping")

    samples = sample_toroidal_stellarator_mirror_hybrid_boundary(
        ntheta=int(ntheta_fit),
        nzeta=int(nzeta_fit),
        **sample_kwargs,
    )
    return _indata_from_boundary_samples(
        samples=samples,
        nfp=nfp,
        mpol=mpol,
        ntor=ntor,
        ns_array=ns_array,
        niter_array=niter_array,
        ftol_array=ftol_array,
        phiedge=phiedge,
        coeff_tol=coeff_tol,
    )


def square_axis_stellarator_mirror_hybrid_indata(
    *,
    nfp: int = 1,
    mpol: int = 5,
    ntor: int = 12,
    ntheta_fit: int = 64,
    nzeta_fit: int = 128,
    ns_array: int | list[int] = 9,
    niter_array: int | list[int] = 40,
    ftol_array: float | list[float] = 1.0e-8,
    phiedge: float = 0.04,
    coeff_tol: float = 1.0e-12,
    **sample_kwargs: Any,
) -> InData:
    """Return VMEC ``InData`` for the square-axis toroidal hybrid boundary."""

    nfp = int(nfp)
    mpol = int(mpol)
    ntor = int(ntor)
    if nfp <= 0:
        raise ValueError("nfp must be positive")
    if mpol < 3:
        raise ValueError("mpol must be at least 3 so the corner m=2 shaping fits")
    if ntor < 4:
        raise ValueError("ntor must be at least 4 to fit the square-like axis")
    samples = sample_square_axis_stellarator_mirror_hybrid_boundary(
        ntheta=int(ntheta_fit),
        nzeta=int(nzeta_fit),
        **sample_kwargs,
    )
    return _indata_from_boundary_samples(
        samples=samples,
        nfp=nfp,
        mpol=mpol,
        ntor=ntor,
        ns_array=ns_array,
        niter_array=niter_array,
        ftol_array=ftol_array,
        phiedge=phiedge,
        coeff_tol=coeff_tol,
    )


def square_axis_stellarator_mirror_hybrid_projection_error(
    *,
    nfp: int = 1,
    mpol: int = 5,
    ntor: int = 12,
    ntheta_fit: int = 64,
    nzeta_fit: int = 128,
    ntheta_eval: int | None = None,
    nzeta_eval: int | None = None,
    ns_array: int | list[int] = 9,
    niter_array: int | list[int] = 40,
    ftol_array: float | list[float] = 1.0e-8,
    phiedge: float = 0.04,
    coeff_tol: float = 1.0e-12,
    **sample_kwargs: Any,
) -> dict[str, float | int]:
    """Measure Fourier projection error for a square-axis hybrid boundary.

    The square-axis helper samples a smooth real-space boundary and then stores
    it as ordinary VMEC Fourier boundary coefficients.  This diagnostic reports
    how much the selected ``MPOL``/``NTOR`` truncation changes that sampled
    boundary before any equilibrium solve is attempted.
    """

    if "ntheta" in sample_kwargs or "nzeta" in sample_kwargs:
        raise ValueError(
            "Use ntheta_fit/nzeta_fit for projection sampling and "
            "ntheta_eval/nzeta_eval for error evaluation; ntheta/nzeta are "
            "reserved by the underlying boundary sampler."
        )
    ntheta_eval = int(ntheta_fit if ntheta_eval is None else ntheta_eval)
    nzeta_eval = int(nzeta_fit if nzeta_eval is None else nzeta_eval)
    target = sample_square_axis_stellarator_mirror_hybrid_boundary(
        ntheta=ntheta_eval,
        nzeta=nzeta_eval,
        **sample_kwargs,
    )
    indata = square_axis_stellarator_mirror_hybrid_indata(
        nfp=nfp,
        mpol=mpol,
        ntor=ntor,
        ntheta_fit=ntheta_fit,
        nzeta_fit=nzeta_fit,
        ns_array=ns_array,
        niter_array=niter_array,
        ftol_array=ftol_array,
        phiedge=phiedge,
        coeff_tol=coeff_tol,
        **sample_kwargs,
    )
    reconstructed = evaluate_toroidal_hybrid_indata_boundary(
        indata,
        ntheta=ntheta_eval,
        nzeta=nzeta_eval,
    )
    dR = np.asarray(reconstructed.R, dtype=float) - np.asarray(target.R, dtype=float)
    dZ = np.asarray(reconstructed.Z, dtype=float) - np.asarray(target.Z, dtype=float)
    err = np.sqrt(dR * dR + dZ * dZ)
    target_scale = max(
        float(np.ptp(np.asarray(target.R, dtype=float))),
        float(np.ptp(np.asarray(target.Z, dtype=float))),
        np.finfo(float).tiny,
    )
    rms = float(np.sqrt(np.mean(err * err)))
    max_abs = float(np.max(err))
    max_abs_R = float(np.max(np.abs(dR)))
    max_abs_Z = float(np.max(np.abs(dZ)))
    max_abs_component = max(max_abs_R, max_abs_Z)
    return {
        "nfp": int(nfp),
        "mpol": int(mpol),
        "ntor": int(ntor),
        "mode_count": _square_axis_mode_count(mpol, ntor),
        "recommended_nzeta": recommended_square_axis_nzeta(ntor),
        "ntheta_fit": int(ntheta_fit),
        "nzeta_fit": int(nzeta_fit),
        "ntheta_eval": int(ntheta_eval),
        "nzeta_eval": int(nzeta_eval),
        "max_abs_R_error": max_abs_R,
        "max_abs_Z_error": max_abs_Z,
        "max_abs_component_error": max_abs_component,
        "rms_R_error": float(np.sqrt(np.mean(dR * dR))),
        "rms_Z_error": float(np.sqrt(np.mean(dZ * dZ))),
        "max_abs_error": max_abs,
        "rms_error": rms,
        "max_abs_error_rel": float(max_abs / target_scale),
        "max_abs_component_error_rel": float(max_abs_component / target_scale),
        "rms_error_rel": float(rms / target_scale),
    }


def square_axis_resolution_deck_status(
    *,
    projection: dict[str, Any],
    mpol: int,
    ntor: int,
    nzeta: int,
    ntheta: int | None = None,
    mgrid_nphi: int | None = None,
    ns: int | None = None,
    target_max_component_error: float | None = None,
) -> dict[str, Any]:
    """Classify whether a square-axis Fourier deck is ready for a strict solve.

    This is a cheap pre-solve gate.  It checks only representation and grid
    compatibility: boundary projection error, recommended VMEC real-space grid
    sizes, and generated-mgrid plane compatibility.  It deliberately does not
    claim nonlinear VMEC convergence.
    """

    mpol_i = int(mpol)
    ntor_i = int(ntor)
    nzeta_i = int(nzeta)
    if mpol_i < 0:
        raise ValueError("mpol must be nonnegative")
    if ntor_i < 0:
        raise ValueError("ntor must be nonnegative")
    if nzeta_i <= 0:
        raise ValueError("nzeta must be positive")
    ntheta_i = None if ntheta is None else int(ntheta)
    if ntheta_i is not None and ntheta_i <= 0:
        raise ValueError("ntheta must be positive")
    mgrid_nphi_i = int(nzeta_i if mgrid_nphi is None else mgrid_nphi)
    if mgrid_nphi_i <= 0:
        raise ValueError("mgrid_nphi must be positive")

    recommended_nzeta = recommended_square_axis_nzeta(ntor_i)
    recommended_ntheta = recommended_square_axis_ntheta(mpol_i)

    def _finite_float(value: Any) -> float | None:
        try:
            out = float(value)
        except Exception:
            return None
        return out if np.isfinite(out) else None

    max_component_error = _finite_float(projection.get("max_abs_component_error"))
    rms_error = _finite_float(projection.get("rms_error"))
    nzeta_underrecommended = bool(nzeta_i < int(recommended_nzeta))
    ntheta_underrecommended = None if ntheta_i is None else bool(ntheta_i < int(recommended_ntheta))
    mgrid_nphi_multiple = bool(mgrid_nphi_i % max(1, nzeta_i) == 0)
    nzeta_margin = int(nzeta_i - int(recommended_nzeta))
    ntheta_margin = None if ntheta_i is None else int(ntheta_i - int(recommended_ntheta))
    mgrid_nphi_margin = int(mgrid_nphi_i - nzeta_i)
    points_per_toroidal_mode = float(nzeta_i / max(1, ntor_i))
    points_per_poloidal_mode = None if ntheta_i is None else float(ntheta_i / max(1, mpol_i))
    projection_meets_gate = (
        None
        if target_max_component_error is None or max_component_error is None
        else bool(max_component_error <= float(target_max_component_error))
    )

    reasons: list[str] = []
    if target_max_component_error is None:
        reasons.append("projection_gate_disabled")
    elif not bool(projection_meets_gate):
        reasons.append("projection_error_exceeds_gate")
    if nzeta_underrecommended:
        reasons.append("nzeta_below_square_axis_recommendation")
    if ntheta_underrecommended:
        reasons.append("ntheta_below_square_axis_recommendation")
    if not mgrid_nphi_multiple:
        reasons.append("mgrid_nphi_not_multiple_of_nzeta")

    if not reasons:
        status = "production_ready"
    elif reasons == ["projection_gate_disabled"]:
        status = "diagnostic_gate_disabled"
    else:
        status = "diagnostic_underresolved"

    mode_count = int(projection.get("mode_count", -1))
    return {
        "status": status,
        "reasons": reasons,
        "mpol": mpol_i,
        "ntor": ntor_i,
        "ns": None if ns is None else int(ns),
        "ntheta": None if ntheta_i is None else int(ntheta_i),
        "recommended_ntheta": int(recommended_ntheta),
        "recommended_ntheta_rule": "ceil(max(64, 4*mpol) / 8) * 8",
        "ntheta_margin": ntheta_margin,
        "ntheta_underrecommended": ntheta_underrecommended,
        "nzeta": nzeta_i,
        "recommended_nzeta": int(recommended_nzeta),
        "recommended_nzeta_rule": "ceil(max(16, 2*ntor + 8) / 8) * 8",
        "nzeta_margin": nzeta_margin,
        "nzeta_underrecommended": nzeta_underrecommended,
        "mgrid_nphi": mgrid_nphi_i,
        "mgrid_nphi_margin": mgrid_nphi_margin,
        "mgrid_nphi_multiple_of_nzeta": mgrid_nphi_multiple,
        "mode_count": mode_count,
        "fourier_boundary_channel_count": None if mode_count < 0 else 4 * mode_count,
        "points_per_toroidal_mode": points_per_toroidal_mode,
        "points_per_poloidal_mode": points_per_poloidal_mode,
        "projection_target_max_component_error": (
            None if target_max_component_error is None else float(target_max_component_error)
        ),
        "projection_max_abs_component_error": max_component_error,
        "projection_rms_error": rms_error,
        "projection_meets_gate": projection_meets_gate,
    }


def _as_int_schedule(value: int | list[int] | tuple[int, ...] | None) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


def _as_float_schedule(value: float | list[float] | tuple[float, ...] | None) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(value)]


def square_axis_strict_schedule_status(
    *,
    ns_array: int | list[int] | tuple[int, ...] | None = None,
    niter_array: int | list[int] | tuple[int, ...] | None = None,
    ftol_array: float | list[float] | tuple[float, ...] | None = None,
    target_ftol: float = 1.0e-12,
) -> dict[str, Any]:
    """Classify whether a square-axis solve schedule can support strict claims.

    This is a cheap schedule gate.  It checks the requested final force
    tolerance and simple stage-array validity before a long free-boundary run.
    It does not claim nonlinear convergence; it only says whether convergence
    to the strict component target was requested.
    """

    ns_values = _as_int_schedule(ns_array)
    niter_values = _as_int_schedule(niter_array)
    ftol_values = _as_float_schedule(ftol_array)
    target = float(target_ftol)
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError("target_ftol must be positive and finite")

    reasons: list[str] = []
    if not ftol_values:
        reasons.append("missing_ftol_array")
        final_ftol = None
    else:
        final_ftol = float(ftol_values[-1])
        if not np.isfinite(final_ftol) or final_ftol <= 0.0:
            reasons.append("invalid_final_ftol")
        elif final_ftol > target * (1.0 + 1.0e-10):
            reasons.append("final_ftol_above_strict_target")
    if any(value < 3 for value in ns_values):
        reasons.append("invalid_ns_array")
    if any(value <= 0 for value in niter_values):
        reasons.append("invalid_niter_array")
    if any((not np.isfinite(value)) or value <= 0.0 for value in ftol_values):
        reasons.append("invalid_ftol_array")
    nonempty_lengths = [len(values) for values in (ns_values, niter_values, ftol_values) if values]
    if nonempty_lengths and len(set(nonempty_lengths)) != 1:
        reasons.append("stage_array_length_mismatch")

    requested_final_ftol_meets_target = bool(final_ftol is not None and np.isfinite(final_ftol) and final_ftol <= target)
    return {
        "status": "strict_ready" if not reasons else "diagnostic_schedule",
        "reasons": list(dict.fromkeys(reasons)),
        "componentwise_target_ftol": target,
        "requested_final_ftol": final_ftol,
        "requested_final_ftol_meets_target": requested_final_ftol_meets_target,
        "ns_array": [int(value) for value in ns_values],
        "niter_array": [int(value) for value in niter_values],
        "ftol_array": [float(value) for value in ftol_values],
        "total_iteration_budget": int(sum(niter_values)),
    }


def square_axis_strict_convergence_assessment(
    *,
    resolution_deck: dict[str, Any],
    strict_schedule: dict[str, Any],
    edge_control_projection_enabled: bool = False,
    edge_control_update_mode: str = "projected_delta",
    solver_native_spline_controls: bool = False,
    target_ftol: float = 1.0e-12,
) -> dict[str, Any]:
    """Interpret strict convergence readiness for square-axis hybrid solves.

    This helper keeps three claims separate: ordinary VMEC Fourier convergence,
    reduced spline-control convergence, and VMEC2000/mgrid backend parity. It is
    diagnostic metadata only; it does not change the nonlinear update path or
    relax the requested force tolerance.
    """

    target = float(target_ftol)
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError("target_ftol must be positive and finite")
    resolution_status = str(resolution_deck.get("status", "unknown"))
    schedule_status = str(strict_schedule.get("status", "unknown"))
    strict_ftol_requested = bool(strict_schedule.get("requested_final_ftol_meets_target", False))
    representation_ready = bool(resolution_status == "production_ready")
    full_fourier_ready = bool(representation_ready and strict_ftol_requested)
    reduced_enabled = bool(edge_control_projection_enabled)
    edge_update_mode = str(edge_control_update_mode).strip().lower()
    coordinate_edge_update = bool(edge_update_mode == "coordinate")
    native_edge_controls = bool(edge_update_mode == "native_coordinate")
    native_spline = bool(solver_native_spline_controls)

    blockers: list[str] = []
    if not representation_ready:
        blockers.append(f"resolution_deck_{resolution_status}")
        blockers.extend(str(reason) for reason in resolution_deck.get("reasons", []) or [])
    if not strict_ftol_requested:
        blockers.append("final_ftol_above_strict_target")
        blockers.extend(str(reason) for reason in strict_schedule.get("reasons", []) or [])
    if not reduced_enabled and not native_spline:
        blockers.append("spline_control_updates_not_enabled")

    next_steps: list[str] = []
    if not representation_ready:
        next_steps.append("repair_fourier_projection_nzeta_or_mgrid_deck_before_long_solve")
    if not strict_ftol_requested:
        next_steps.append("request_final_component_ftol_at_or_below_1e-12")
    if reduced_enabled:
        next_steps.append("profile_reduced_edge_control_state_and_update_residuals")
        if edge_update_mode not in {"coordinate", "native_coordinate"}:
            next_steps.append("rerun_reduced_edge_control_profile_with_coordinate_update_mode")
    elif native_spline:
        next_steps.append("profile_solver_native_spline_control_state")
    else:
        next_steps.append("enable_edge_control_projection_or_promote_solver_native_spline_controls")
    next_steps.append("compare_direct_coils_against_vmec2000_generated_mgrid_reference")
    if full_fourier_ready and reduced_enabled and not native_spline:
        next_steps.append("promote_native_spline_control_state_if_full_fourier_and_vmec2000_stall_above_target")

    primary_lane = (
        "vmec_jax_full_native_spline_state"
        if native_spline
        else "vmec_jax_edge_native_spline_bridge"
        if native_edge_controls
        else "vmec_jax_fourier_strict_profile"
        if full_fourier_ready
        else "repair_preflight_before_strict_solve"
    )
    derivative_priority = [
        "implicit_or_adjoint_differentiation_of_the_converged_native_residual",
        "custom_linear_solve_for_the_vacuum_response_when_the_operator_is_reused",
        "jax_linearize_vjp_matrix_free_newton_krylov_for_solver_prototypes",
        "forward_or_reverse_autodiff_only_for_local_kernel_validation_or_small_dof_smokes",
    ]
    native_recommendation = (
        "full_native_spline_state_available"
        if native_spline
        else "promote_full_native_spline_state_after_edge_bridge_or_vmec2000_stalls"
        if native_edge_controls
        else "enable_native_coordinate_edge_bridge_then_promote_full_native_spline_state"
    )

    return {
        "schema": "square_axis_strict_convergence_assessment.v1",
        "target_component_ftol": target,
        "strict_target_policy": {
            "component_ftol": target,
            "ftol_1e_minus_8_status": "diagnostic_only_for_this_square_hybrid_lane",
            "final_stage_must_request_target": True,
            "claim_requires_componentwise_final_residual": True,
        },
        "resolution_deck_status": resolution_status,
        "strict_schedule_status": schedule_status,
        "full_fourier_strict_profile_status": "ready_to_attempt" if full_fourier_ready else "blocked_by_preflight",
        "full_fourier_strict_claim_requires": [
            "final_fsqr <= target_component_ftol",
            "final_fsqz <= target_component_ftol",
            "final_fsql <= target_component_ftol",
            "final residual recomputed on the accepted boundary",
            "finite-beta cases include virtual-casing/plasma-field boundary diagnostics",
        ],
        "edge_control_projection_enabled": reduced_enabled,
        "edge_control_update_mode": edge_update_mode,
        "reduced_control_profile_status": (
            "native_coordinate_update"
            if reduced_enabled and edge_update_mode == "native_coordinate"
            else "coordinate_update_bridge"
            if reduced_enabled and coordinate_edge_update
            else "enabled_bridge"
            if reduced_enabled
            else "not_enabled"
        ),
        "reduced_control_claim_requires": [
            "edge-control state residual measured on the accepted LCFS",
            "edge-control update-direction residual measured on the final update",
            "coordinate-update count reported when edge_control_update_mode='coordinate'",
            "native-coordinate force pullback reported when edge_control_update_mode='native_coordinate'",
            "full VMEC Fourier residual still reported separately",
        ],
        "solver_native_spline_controls": native_spline,
        "solver_native_spline_edge_controls": native_edge_controls,
        "solver_native_spline_scope": (
            "full_nonlinear_state"
            if native_spline
            else "lcfs_edge_only"
            if native_edge_controls
            else None
        ),
        "solver_native_spline_status": (
            "available"
            if native_spline
            else "edge_only_bridge"
            if native_edge_controls
            else "not_implemented"
        ),
        "full_native_spline_state_required_for_less_fourier_pressure": bool(native_edge_controls and not native_spline),
        "vmec2000_reference_role": (
            "generated-mgrid VMEC2000 is a backend/NESTOR/mgrid reference for the same Fourier deck; "
            "it cannot remove Fourier representation error from a linear-axis square target"
        ),
        "vmec2000_expected_to_fix_fourier_bottleneck": False,
        "solver_method_recommendation": {
            "primary_lane": primary_lane,
            "fast_cli_reference_lane": "vmec2000_generated_mgrid",
            "differentiable_solver_lane": "vmec_jax_full_native_spline_state_with_implicit_or_adjoint_derivatives",
            "vmec2000_robustness_assessment": (
                "use_as_fast_reference_and_mgrid_parity_backend_not_as_the_reduced_geometry_solver"
            ),
            "native_spline_recommendation": native_recommendation,
            "native_actual_force_profile_sequence": [
                "frozen_initial_vacuum_pressure_for_native_matrix_free_mechanics",
                "jax_replay_vacuum_pressure_before_claiming_free_boundary_native_residual",
                "full_native_spline_state_nonlinear_loop_before_claiming_less_fourier_pressure",
            ],
            "derivative_method_priority": derivative_priority,
        },
        "recommended_primary_solver_lane": primary_lane,
        "fast_cli_reference_lane": "vmec2000_generated_mgrid",
        "differentiable_solver_lane": "vmec_jax_full_native_spline_state_with_implicit_or_adjoint_derivatives",
        "native_spline_recommendation": native_recommendation,
        "derivative_method_priority": derivative_priority,
        "blockers": list(dict.fromkeys(blockers)),
        "recommended_next_steps": list(dict.fromkeys(next_steps)),
    }


def toroidal_stellarator_mirror_hybrid_metrics(samples: ToroidalHybridBoundarySamples) -> dict[str, float]:
    """Return lightweight geometry checks for a sampled hybrid boundary."""
    theta_reflect = (-np.arange(samples.theta.size)) % samples.theta.size
    zeta_reflect = (-np.arange(samples.zeta.size)) % samples.zeta.size
    R_reflect = samples.R[np.ix_(theta_reflect, zeta_reflect)]
    Z_reflect = samples.Z[np.ix_(theta_reflect, zeta_reflect)]
    side_cols = [0, samples.zeta.size // 2]
    corner_cols = [samples.zeta.size // 4, (3 * samples.zeta.size) // 4]
    side_r_span = float(np.mean(np.ptp(samples.R[:, side_cols], axis=0)))
    corner_r_span = float(np.mean(np.ptp(samples.R[:, corner_cols], axis=0)))
    orientation = toroidal_hybrid_cross_section_orientation(samples)
    anisotropy = toroidal_hybrid_cross_section_anisotropy(samples)
    side_weight = np.mean(samples.side_weight, axis=0)
    corner_weight = np.mean(samples.corner_weight, axis=0)
    side_region = side_weight >= 0.995
    corner_region = corner_weight >= 0.9
    anisotropy_threshold = 1.0e-14 + 1.0e-8 * float(np.max(anisotropy))
    valid_orientation = anisotropy > anisotropy_threshold
    side_valid = side_region & valid_orientation
    corner_valid = corner_region & valid_orientation
    return {
        "min_R": float(np.min(samples.R)),
        "max_R": float(np.max(samples.R)),
        "max_abs_Z": float(np.max(np.abs(samples.Z))),
        "stellsym_R_error": float(np.max(np.abs(samples.R - R_reflect))),
        "stellsym_Z_error": float(np.max(np.abs(samples.Z + Z_reflect))),
        "side_r_span_mean": side_r_span,
        "corner_r_span_mean": corner_r_span,
        "corner_weight_max": float(np.max(samples.corner_weight)),
        "side_weight_max": float(np.max(samples.side_weight)),
        "cross_section_orientation_span": float(np.ptp(orientation)),
        "side_orientation_span": float(np.ptp(orientation[side_region])) if np.any(side_region) else 0.0,
        "corner_orientation_span": float(np.ptp(orientation[corner_region])) if np.any(corner_region) else 0.0,
        "orientation_valid_fraction": float(np.mean(valid_orientation)) if valid_orientation.size else 0.0,
        "valid_cross_section_orientation_span": float(np.ptp(orientation[valid_orientation]))
        if np.any(valid_orientation)
        else 0.0,
        "valid_side_orientation_span": float(np.ptp(orientation[side_valid])) if np.any(side_valid) else 0.0,
        "valid_corner_orientation_span": float(np.ptp(orientation[corner_valid])) if np.any(corner_valid) else 0.0,
        "side_corner_weight_overlap_max": float(np.max(side_weight * corner_weight)),
        "cross_section_anisotropy_min": float(np.min(anisotropy)),
        "cross_section_anisotropy_max": float(np.max(anisotropy)),
    }


def _sample_RZ_arrays(samples: ToroidalHybridBoundarySamples) -> tuple[np.ndarray, np.ndarray]:
    R = np.asarray(samples.R, dtype=float)
    Z = np.asarray(samples.Z, dtype=float)
    if R.shape != Z.shape:
        raise ValueError("R and Z samples must have the same shape")
    if R.ndim != 2 or R.shape[0] < 3 or R.shape[1] < 1:
        raise ValueError("R and Z samples must have shape (ntheta, nzeta)")
    return R, Z


def toroidal_hybrid_cross_section_anisotropy(samples: ToroidalHybridBoundarySamples) -> np.ndarray:
    """Return the covariance anisotropy strength of each sampled cross section."""
    R, Z = _sample_RZ_arrays(samples)
    values = []
    for col in range(R.shape[1]):
        r = R[:, col] - float(np.mean(R[:, col]))
        z = Z[:, col] - float(np.mean(Z[:, col]))
        q1 = float(np.mean(r * r) - np.mean(z * z))
        q2 = float(2.0 * np.mean(r * z))
        values.append(np.hypot(q1, q2))
    return np.asarray(values, dtype=float)


def toroidal_hybrid_cross_section_orientation(samples: ToroidalHybridBoundarySamples) -> np.ndarray:
    """Return the unwrapped principal-axis angle of each sampled cross section.

    The angle is undefined where the cross-section covariance is isotropic.  Use
    `toroidal_hybrid_cross_section_anisotropy` to mask those points before
    interpreting orientation differences.
    """
    R, Z = _sample_RZ_arrays(samples)
    angles = []
    for col in range(R.shape[1]):
        r = R[:, col] - float(np.mean(R[:, col]))
        z = Z[:, col] - float(np.mean(Z[:, col]))
        rr = float(np.mean(r * r))
        zz = float(np.mean(z * z))
        rz = float(np.mean(r * z))
        angles.append(0.5 * np.arctan2(2.0 * rz, rr - zz))
    return 0.5 * np.unwrap(2.0 * np.asarray(angles, dtype=float))


def evaluate_toroidal_hybrid_indata_boundary(
    indata: InData,
    *,
    ntheta: int = 64,
    nzeta: int = 64,
) -> ToroidalHybridBoundarySamples:
    """Evaluate a generated hybrid input boundary on a uniform grid."""
    from .boundary import boundary_input_from_indata

    mpol = int(indata.get_int("MPOL", 5))
    ntor = int(indata.get_int("NTOR", 4))
    nfp = int(indata.get_int("NFP", 2))
    modes = vmec_mode_table(mpol=mpol, ntor=ntor)
    theta = np.linspace(0.0, 2.0 * np.pi, int(ntheta), endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, int(nzeta), endpoint=False)
    grid = AngleGrid(theta=theta, zeta=zeta, nfp=nfp)
    basis = build_helical_basis(modes, grid)
    boundary = boundary_input_from_indata(indata, modes)
    R = np.asarray(eval_fourier(boundary.R_cos, boundary.R_sin, basis), dtype=float)
    Z = np.asarray(eval_fourier(boundary.Z_cos, boundary.Z_sin, basis), dtype=float)
    theta2, zeta2 = np.meshgrid(theta, zeta, indexing="ij")
    return ToroidalHybridBoundarySamples(
        theta=theta,
        zeta=zeta,
        R=R,
        Z=Z,
        side_weight=np.cos(zeta2) ** 2,
        corner_weight=np.sin(zeta2) ** 2,
    )
