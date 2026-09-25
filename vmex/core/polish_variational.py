"""Tensorized fixed-pressure energy for the native high-order equilibrium.

This module is the reference/production seam for the variational polishing
candidate.  Spatial spline and Fourier derivatives are precomputed as linear
tables; evaluating the MHD energy therefore needs no nested pointwise AD.
The strong-force oracle remains independent in :mod:`vmex.core.strong_force`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .profiles import MU0
from .radial_basis import _basis_levels
from .polish import (
    HighOrderCorrection,
    NativeCorrectionLayout,
    apply_high_order_correction,
)
from .strong_force import HighOrderEquilibriumState, StrongForceSamples

Array = Any


_STABLE_TABLES = (
    "spline_value",
    "spline_first",
    "spline_second",
    "difference_first",
    "difference_second",
    "axis_factors",
)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class VariationalPlan:
    """Precomputed coefficient-to-jet tables and quadrature weights."""

    rho: Array
    theta: Array
    zeta: Array
    radial_value: Array
    radial_derivative: Array
    radial_second_derivative: Array
    profile_basis: Array
    profile_derivative: Array
    cosine: Array
    sine: Array
    cosine_theta: Array
    sine_theta: Array
    cosine_zeta: Array
    sine_zeta: Array
    cosine_theta_theta: Array
    sine_theta_theta: Array
    cosine_theta_zeta: Array
    sine_theta_zeta: Array
    cosine_zeta_zeta: Array
    sine_zeta_zeta: Array
    quadrature_weights: Array
    nfp: int
    jacobian_sign: int
    # Optional coefficient-first radial tables (see ``_stable_radial_jets``).
    # ``spline_*`` have shape (nrho, nbasis - k); ``difference_*`` hold the
    # knot factors p/(t[i+p+1]-t[i+1]); ``axis_factors`` is (6, nrho, nmode).
    spline_value: Array | None = None
    spline_first: Array | None = None
    spline_second: Array | None = None
    difference_first: Array | None = None
    difference_second: Array | None = None
    axis_factors: Array | None = None

    def tree_flatten(self):
        """Expose numerical tables as traced data and topology as metadata."""

        children = tuple(
            getattr(self, name)
            for name in (
                "rho",
                "theta",
                "zeta",
                "radial_value",
                "radial_derivative",
                "radial_second_derivative",
                "profile_basis",
                "profile_derivative",
                "cosine",
                "sine",
                "cosine_theta",
                "sine_theta",
                "cosine_zeta",
                "sine_zeta",
                "cosine_theta_theta",
                "sine_theta_theta",
                "cosine_theta_zeta",
                "sine_theta_zeta",
                "cosine_zeta_zeta",
                "sine_zeta_zeta",
                "quadrature_weights",
            )
        )
        stable = tuple(getattr(self, name) for name in _STABLE_TABLES)
        return children + stable, (int(self.nfp), int(self.jacobian_sign))

    @classmethod
    def tree_unflatten(cls, metadata, children):
        """Rebuild a plan from its JAX pytree representation."""

        nfp, jacobian_sign = metadata
        count = len(children) - len(_STABLE_TABLES)
        stable = dict(zip(_STABLE_TABLES, children[count:]))
        return cls(*children[:count], nfp=nfp, jacobian_sign=jacobian_sign, **stable)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Tensor-grid shape ``(nrho, ntheta, nzeta)``."""

        return int(self.rho.size), int(self.theta.size), int(self.zeta.size)


@dataclass(frozen=True)
class VariationalFieldSamples:
    """Geometry and fields evaluated by the tensorized first-jet kernel."""

    position: Array
    dposition_drho: Array
    dposition_dtheta: Array
    dposition_dzeta: Array
    sqrt_g: Array
    B: Array
    pressure: Array


jax.tree_util.register_dataclass(
    VariationalFieldSamples,
    data_fields=[field for field in VariationalFieldSamples.__dataclass_fields__],
    meta_fields=[],
)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class NativeGaugePlan:
    """Rows for a local normalized-tangent coordinate convention.

    The rows are mass-normalized tests in the symmetry-allowed sine space.
    They define a bordered constraint operator rather than a dense global
    nullspace, preserving the locality of the native spline coordinates.
    """

    variational: VariationalPlan
    row_modes: Array
    row_basis: Array
    row_scale: Array
    tangent_floor: float

    def tree_flatten(self):
        """Expose numerical rows as data and the floor as metadata."""

        return (
            (self.variational, self.row_modes, self.row_basis, self.row_scale),
            (float(self.tangent_floor),),
        )

    @classmethod
    def tree_unflatten(cls, metadata, children):
        """Rebuild a native gauge plan from its pytree representation."""

        (tangent_floor,) = metadata
        variational, row_modes, row_basis, row_scale = children
        return cls(
            variational=variational,
            row_modes=row_modes,
            row_basis=row_basis,
            row_scale=row_scale,
            tangent_floor=tangent_floor,
        )

    @property
    def size(self) -> int:
        """Number of independent coordinate constraints."""

        return int(self.row_modes.size)


def _span_quadrature(breakpoints: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    centers = 0.5 * (breakpoints[:-1] + breakpoints[1:])
    scales = 0.5 * np.diff(breakpoints)
    return (
        (centers[:, None] + scales[:, None] * nodes[None]).reshape(-1),
        (scales[:, None] * weights[None]).reshape(-1),
    )


def make_variational_plan(
    state: HighOrderEquilibriumState,
    *,
    radial_order: int | None = None,
    ntheta: int | None = None,
    nzeta: int | None = None,
    stable_derivatives: bool = False,
) -> VariationalPlan:
    """Build a tensor grid and analytic coefficient-to-first-jet tables.

    With ``stable_derivatives`` the force kernels differentiate spline
    coefficients by local differences before evaluating lower-degree splines
    (``_stable_radial_jets``).  The represented fields are identical; only the
    floating-point cancellation pattern changes.
    """

    degree = int(state.radial_basis.degree)
    order = degree + 3 if radial_order is None else int(radial_order)
    if order < 2:
        raise ValueError("radial_order must be at least two")
    s, weights_s = _span_quadrature(np.asarray(state.radial_basis.breakpoints, dtype=float), order)
    rho = np.sqrt(s)
    # The radial quadrature is constructed in s, while sqrt_g is the
    # Jacobian of (rho, theta, zeta): d rho = ds / (2 rho).
    weights_rho = weights_s / (2.0 * rho)
    basis = np.asarray(state.radial_basis.basis_matrix(s), dtype=float)
    basis_s = np.asarray(state.radial_basis.basis_matrix(s, derivative=1), dtype=float)
    basis_ss = np.asarray(state.radial_basis.basis_matrix(s, derivative=2), dtype=float)
    m = np.abs(np.asarray(state.m, dtype=int))
    powers = rho[None, :, None] ** m[:, None, None]
    radial_value = powers * basis[None]
    leading = np.zeros_like(radial_value)
    nonzero = m > 0
    leading[nonzero] = m[nonzero, None, None] * rho[None, :, None] ** (m[nonzero, None, None] - 1) * basis[None]
    radial_derivative = leading + (2.0 * rho[None, :, None] ** (m[:, None, None] + 1) * basis_s[None])
    leading_second = np.zeros_like(radial_value)
    second_active = m >= 2
    leading_second[second_active] = (
        m[second_active, None, None]
        * (m[second_active, None, None] - 1)
        * rho[None, :, None] ** (m[second_active, None, None] - 2)
        * basis[None]
    )
    radial_second_derivative = (
        leading_second
        + (4 * m[:, None, None] + 2) * rho[None, :, None] ** m[:, None, None] * basis_s[None]
        + 4.0 * rho[None, :, None] ** (m[:, None, None] + 2) * basis_ss[None]
    )

    signed_m = np.asarray(state.m, dtype=int)
    n = np.asarray(state.n, dtype=int)
    mmax = int(np.max(np.abs(signed_m), initial=0))
    nmax = int(np.max(np.abs(n), initial=0))
    ntheta = max(4 * mmax + 5, 8) if ntheta is None else int(ntheta)
    nzeta = (1 if nmax == 0 else 4 * nmax + 5) if nzeta is None else int(nzeta)
    if ntheta < 1 or nzeta < 1:
        raise ValueError("angular grid sizes must be positive")
    theta = 2.0 * np.pi * np.arange(ntheta) / ntheta
    zeta = 2.0 * np.pi * np.arange(nzeta) / nzeta
    tt, zz = np.meshgrid(theta, zeta, indexing="ij")
    phase = signed_m[:, None] * tt.reshape(1, -1) - n[:, None] * zz.reshape(1, -1)
    cosine = np.cos(phase)
    sine = np.sin(phase)
    cosine_theta = -signed_m[:, None] * sine
    sine_theta = signed_m[:, None] * cosine
    cosine_zeta = n[:, None] * sine
    sine_zeta = -n[:, None] * cosine
    cosine_theta_theta = -(signed_m[:, None] ** 2) * cosine
    sine_theta_theta = -(signed_m[:, None] ** 2) * sine
    cosine_theta_zeta = signed_m[:, None] * n[:, None] * cosine
    sine_theta_zeta = signed_m[:, None] * n[:, None] * sine
    cosine_zeta_zeta = -(n[:, None] ** 2) * cosine
    sine_zeta_zeta = -(n[:, None] ** 2) * sine
    angular_weight = (2.0 * np.pi / ntheta) * (2.0 * np.pi / nzeta)
    # zeta spans one field period and sqrt_g contains dphi/dzeta=1/nfp;
    # multiplying by nfp integrates the full torus.
    quadrature_weights = float(state.nfp) * weights_rho[:, None, None] * angular_weight
    stable = {}
    if stable_derivatives:
        stable = _stable_radial_tables(state.radial_basis, s, rho, m)
    return VariationalPlan(
        **stable,
        rho=jnp.asarray(rho),
        theta=jnp.asarray(theta),
        zeta=jnp.asarray(zeta),
        radial_value=jnp.asarray(radial_value),
        radial_derivative=jnp.asarray(radial_derivative),
        radial_second_derivative=jnp.asarray(radial_second_derivative),
        profile_basis=jnp.asarray(basis),
        profile_derivative=jnp.asarray(2.0 * rho[:, None] * basis_s),
        cosine=jnp.asarray(cosine),
        sine=jnp.asarray(sine),
        cosine_theta=jnp.asarray(cosine_theta),
        sine_theta=jnp.asarray(sine_theta),
        cosine_zeta=jnp.asarray(cosine_zeta),
        sine_zeta=jnp.asarray(sine_zeta),
        cosine_theta_theta=jnp.asarray(cosine_theta_theta),
        sine_theta_theta=jnp.asarray(sine_theta_theta),
        cosine_theta_zeta=jnp.asarray(cosine_theta_zeta),
        sine_theta_zeta=jnp.asarray(sine_theta_zeta),
        cosine_zeta_zeta=jnp.asarray(cosine_zeta_zeta),
        sine_zeta_zeta=jnp.asarray(sine_zeta_zeta),
        quadrature_weights=jnp.asarray(quadrature_weights),
        nfp=int(state.nfp),
        jacobian_sign=int(state.jacobian_sign),
    )


def make_native_gauge_plan(
    state: HighOrderEquilibriumState,
    variational: VariationalPlan,
    *,
    lasym: bool = False,
    tangent_floor: float = 1.0e-12,
) -> NativeGaugePlan:
    """Build bordered normalized-tangent rows without a global gauge SVD."""

    if lasym:
        raise NotImplementedError("the nonsymmetric native gauge requires paired sine/cosine rows")
    if tangent_floor <= 0.0:
        raise ValueError("tangent_floor must be positive")
    mode_zero = (np.asarray(state.m, dtype=int) == 0) & (np.asarray(state.n, dtype=int) == 0)
    row_modes, row_basis = np.nonzero(
        np.broadcast_to((~mode_zero)[:, None], (state.m.size, state.radial_basis.size))
        & (np.arange(state.radial_basis.size)[None, :] < state.radial_basis.size - 1)
    )
    if row_modes.size == 0:
        raise ValueError("native gauge has no symmetry-allowed tangent rows")
    radial = np.asarray(variational.radial_value)[row_modes, :, row_basis]
    angular = np.asarray(variational.sine)[row_modes]
    weights = np.broadcast_to(np.asarray(variational.quadrature_weights), variational.shape).reshape(
        (variational.shape[0], -1)
    )
    mass = np.einsum("ij,ki,kj->k", weights, radial**2, angular**2)
    if np.any(~np.isfinite(mass)) or np.any(mass <= 0.0):
        raise ValueError("native gauge contains an unresolved test row")
    return NativeGaugePlan(
        variational=variational,
        row_modes=jnp.asarray(row_modes, dtype=jnp.int32),
        row_basis=jnp.asarray(row_basis, dtype=jnp.int32),
        row_scale=jnp.asarray(1.0 / np.sqrt(mass)),
        tangent_floor=float(tangent_floor),
    )


def native_coordinate_scales(
    state: HighOrderEquilibriumState,
    layout: NativeCorrectionLayout,
    variational: VariationalPlan,
) -> Array:
    """Equilibrate native coordinates by physical displacement L2 norms.

    Geometry columns use their cylindrical displacement.  Lambda columns use
    the fixed-label displacement ``|x_theta delta-lambda/(1+lambda_theta)|``.
    The volume measure is the signed physical Jacobian on the variational
    grid.  This is a local-table setup operation and forms no global Jacobian.
    """

    fields = evaluate_variational_fields(state, variational)
    _, _, lambda_theta, _ = _channel(state.L_cos, state.L_sin, variational)
    weights = np.asarray(
        np.broadcast_to(variational.quadrature_weights, variational.shape)
        * (state.jacobian_sign * np.asarray(fields.sqrt_g))
    ).reshape((variational.shape[0], -1))
    if np.any(weights <= 0.0) or np.any(~np.isfinite(weights)):
        raise ValueError("native coordinate scaling requires a valid signed Jacobian")
    tangent_factor = (
        np.sum(np.asarray(fields.dposition_dtheta) ** 2, axis=-1)
        / (1.0 + np.asarray(lambda_theta).reshape(variational.shape)) ** 2
    ).reshape((variational.shape[0], -1))
    indices = np.asarray(layout.active_indices, dtype=int)
    block = layout.mnmax * layout.nbasis
    fields_index = indices // block
    remainder = indices % block
    modes = remainder // layout.nbasis
    basis_indices = remainder % layout.nbasis
    radial = np.asarray(variational.radial_value)
    cosine_squared = np.asarray(variational.cosine) ** 2
    sine_squared = np.asarray(variational.sine) ** 2

    # Contract angles before radial basis functions.  This is algebraically
    # identical to constructing one (coordinate, radius, angle) tensor, but
    # its largest temporary is (mode, radius, basis).  At the basis-191
    # checkpoint that removes an 8 GiB logical coordinate-by-grid array.
    def squared_norms(angular_squared: np.ndarray, factor: np.ndarray) -> np.ndarray:
        angular_moment = (weights * factor) @ angular_squared.T
        return np.einsum(
            "mrk,rm->mk",
            radial * radial,
            angular_moment,
            optimize=True,
        )

    geometry_cosine = squared_norms(cosine_squared, np.ones_like(weights))
    geometry_sine = squared_norms(sine_squared, np.ones_like(weights))
    lambda_cosine = squared_norms(cosine_squared, tangent_factor)
    lambda_sine = squared_norms(sine_squared, tangent_factor)
    table = np.stack((geometry_cosine, geometry_sine, lambda_cosine, lambda_sine), axis=0)
    kind = (fields_index % 2) + 2 * (fields_index >= 4)
    norms = np.sqrt(table[kind, modes, basis_indices])
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("native coordinate layout contains a zero physical column")
    return jnp.asarray(1.0 / norms)


def native_tangential_gauge_matrix(
    state: HighOrderEquilibriumState,
    layout: NativeCorrectionLayout,
    gauge: NativeGaugePlan,
    coordinate_scale: Array,
) -> Any:
    """Assemble the frozen linear gauge directly as a SciPy CSR matrix.

    This is exactly the derivative of :func:`native_tangential_gauge_residual`
    with respect to the packed, scaled native coordinates.  It contracts the
    angular tangent projection first and then couples only overlapping radial
    B-spline supports, avoiding a dense global ``jacfwd`` and dense QR setup.
    Lambda columns are structurally zero because this gauge measures frozen
    physical R/Z tangent motion.
    """

    try:
        from scipy import sparse
    except ImportError as error:  # pragma: no cover - SciPy is a VMEX dependency
        raise ImportError("native sparse gauge assembly requires SciPy") from error

    plan = gauge.variational
    scale = np.asarray(coordinate_scale, dtype=float)
    if scale.shape != (layout.size,) or np.any(~np.isfinite(scale)):
        raise ValueError("native gauge coordinate scale has the wrong shape")
    active = np.asarray(layout.active_indices, dtype=np.int64)
    block = int(layout.mnmax) * int(layout.nbasis)
    full_to_packed = np.full(6 * block, -1, dtype=np.int64)
    full_to_packed[active] = np.arange(layout.size, dtype=np.int64)

    fields = evaluate_variational_fields(state, plan)
    tangent = np.asarray(fields.dposition_dtheta).reshape((plan.shape[0], -1, 3))
    tangent_norm = np.sqrt(np.sum(tangent * tangent, axis=-1) + float(gauge.tangent_floor) ** 2)
    _, zz = np.meshgrid(np.asarray(plan.theta), np.asarray(plan.zeta), indexing="ij")
    phi = zz.reshape(-1) / float(plan.nfp)
    radial_component = (tangent[..., 0] * np.cos(phi)[None] + tangent[..., 1] * np.sin(phi)[None]) / tangent_norm
    vertical_component = tangent[..., 2] / tangent_norm
    quadrature = np.broadcast_to(np.asarray(plan.quadrature_weights), plan.shape).reshape((plan.shape[0], -1))
    radial = np.asarray(plan.radial_value)
    radial_support = radial != 0.0
    cosine = np.asarray(plan.cosine)
    sine = np.asarray(plan.sine)
    row_modes = np.asarray(gauge.row_modes, dtype=np.int64)
    row_basis = np.asarray(gauge.row_basis, dtype=np.int64)
    row_scale = np.asarray(gauge.row_scale, dtype=float)

    rows: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for test_mode in np.unique(row_modes):
        row_positions = np.flatnonzero(row_modes == test_mode)
        test_basis = row_basis[row_positions]
        test_radial = radial[test_mode]
        test_support = radial_support[test_mode]
        test_angular = sine[test_mode]
        for field, component in (
            (0, radial_component),
            (1, radial_component),
            (2, vertical_component),
            (3, vertical_component),
        ):
            angular_table = cosine if field % 2 == 0 else sine
            for mode in range(layout.mnmax):
                packed = full_to_packed[field * block + mode * int(layout.nbasis) + np.arange(layout.nbasis)]
                active_basis = np.flatnonzero(packed >= 0)
                if active_basis.size == 0:
                    continue
                angular_moment = np.sum(
                    quadrature * component * test_angular[None] * angular_table[mode][None],
                    axis=1,
                )
                coupled = test_radial.T @ (angular_moment[:, None] * radial[mode])
                overlap = test_support.T @ radial_support[mode]
                selected_overlap = overlap[np.ix_(test_basis, active_basis)]
                local_rows, local_columns = np.nonzero(selected_overlap)
                if local_rows.size == 0:
                    continue
                packed_columns = packed[active_basis[local_columns]]
                rows.append(row_positions[local_rows])
                columns.append(packed_columns)
                values.append(
                    coupled[test_basis[local_rows], active_basis[local_columns]]
                    * row_scale[row_positions[local_rows]]
                    * scale[packed_columns]
                )
    if not rows:
        raise ValueError("native sparse gauge has no structural entries")
    matrix = sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(gauge.size, layout.size),
    ).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if np.any(np.diff(matrix.indptr) == 0) or np.any(~np.isfinite(matrix.data)):
        raise ValueError("native sparse gauge contains an empty or nonfinite row")
    return matrix


def _stable_radial_tables(radial_basis, s: np.ndarray, rho: np.ndarray, m: np.ndarray) -> dict:
    """Return coefficient-first derivative tables for ``rho**m q(rho**2)``.

    For degree p and knots t, q_s = sum_i d_i B_{i,p-1}(s; t[1:-1]) with
    d_i = p (c_{i+1} - c_i) / (t_{i+p+1} - t_{i+1}); repeating once gives q_ss.
    Differences of neighbouring coefficients are formed before any large knot
    factor multiplies them, so a constant mode differentiates to exact zeros.
    """

    knots = np.asarray(radial_basis.knots, dtype=float)
    degree = int(radial_basis.degree)
    if degree < 2:
        raise ValueError("coefficient-first second derivatives need degree >= 2")
    size = knots.size - degree - 1
    first_gap = knots[degree + 1 : degree + size] - knots[1:size]
    second_gap = knots[degree + 1 : degree + size - 1] - knots[2:size]
    if np.any(first_gap <= 0.0) or np.any(second_gap <= 0.0):
        raise ValueError("knot multiplicity too high for a continuous second radial derivative")
    levels = _basis_levels(jnp.asarray(knots), jnp.asarray(s), degree)
    # On a clamped vector the first and last lower-degree functions of the full
    # knot sequence vanish identically; the interior ones are the trimmed basis.
    spline_value = np.asarray(levels[degree])
    spline_first = np.asarray(levels[degree - 1])[:, 1:-1]
    spline_second = np.asarray(levels[degree - 2])[:, 2:-2]
    m = m.astype(float)
    r = rho[:, None]
    mm = m[None, :]
    safe = np.where(mm > 0, mm - 1.0, 0.0)
    safe2 = np.where(mm > 1, mm - 2.0, 0.0)
    axis = np.stack(
        (
            r**mm,
            np.where(mm > 0, mm * r**safe, 0.0),
            np.where(mm > 1, mm * (mm - 1.0) * r**safe2, 0.0),
            2.0 * r ** (mm + 1.0),
            (4.0 * mm + 2.0) * r**mm,
            4.0 * r ** (mm + 2.0),
        )
    )
    return {
        "spline_value": jnp.asarray(spline_value),
        "spline_first": jnp.asarray(spline_first),
        "spline_second": jnp.asarray(spline_second),
        "difference_first": jnp.asarray(degree / first_gap),
        "difference_second": jnp.asarray((degree - 1) / second_gap),
        "axis_factors": jnp.asarray(axis),
    }


def _stable_radial_jets(coefficients: Array, plan: VariationalPlan) -> tuple[Array, Array, Array]:
    """Return (value, d/drho, d2/drho2) with shape (nrho, nmode) coefficient-first."""

    coefficients = jnp.asarray(coefficients)
    first = jnp.diff(coefficients, axis=1) * plan.difference_first
    second = jnp.diff(first, axis=1) * plan.difference_second
    q = jnp.einsum("mb,rb->rm", coefficients, plan.spline_value)
    q_s = jnp.einsum("mb,rb->rm", first, plan.spline_first)
    q_ss = jnp.einsum("mb,rb->rm", second, plan.spline_second)
    a = plan.axis_factors
    return a[0] * q, a[1] * q + a[3] * q_s, a[2] * q + a[4] * q_s + a[5] * q_ss


def _radial_jets(coefficients: Array, plan: VariationalPlan) -> tuple[Array, Array, Array]:
    if plan.spline_value is not None:
        return _stable_radial_jets(coefficients, plan)
    coefficients = jnp.asarray(coefficients)
    return tuple(
        jnp.einsum("mb,mrb->rm", coefficients, table)
        for table in (plan.radial_value, plan.radial_derivative, plan.radial_second_derivative)
    )


def _channel(
    cosine_coefficients: Array,
    sine_coefficients: Array,
    plan: VariationalPlan,
) -> tuple[Array, Array, Array, Array]:
    """Synthesize one scalar channel and its three coordinate derivatives."""

    if plan.spline_value is not None:
        cosine_radial, cosine_drho, _ = _stable_radial_jets(cosine_coefficients, plan)
        sine_radial, sine_drho, _ = _stable_radial_jets(sine_coefficients, plan)
    else:
        cosine_coefficients = jnp.asarray(cosine_coefficients)
        sine_coefficients = jnp.asarray(sine_coefficients)
        cosine_radial = jnp.einsum("mb,mrb->rm", cosine_coefficients, plan.radial_value)
        sine_radial = jnp.einsum("mb,mrb->rm", sine_coefficients, plan.radial_value)
        cosine_drho = jnp.einsum("mb,mrb->rm", cosine_coefficients, plan.radial_derivative)
        sine_drho = jnp.einsum("mb,mrb->rm", sine_coefficients, plan.radial_derivative)

    def synthesize(cosine_table: Array, sine_table: Array) -> Array:
        return jnp.einsum("rm,ma->ra", cosine_radial, cosine_table) + jnp.einsum("rm,ma->ra", sine_radial, sine_table)

    value = synthesize(plan.cosine, plan.sine)
    drho = jnp.einsum("rm,ma->ra", cosine_drho, plan.cosine) + jnp.einsum("rm,ma->ra", sine_drho, plan.sine)
    dtheta = synthesize(plan.cosine_theta, plan.sine_theta)
    dzeta = synthesize(plan.cosine_zeta, plan.sine_zeta)
    return value, drho, dtheta, dzeta


def _channel_second(
    cosine_coefficients: Array,
    sine_coefficients: Array,
    plan: VariationalPlan,
) -> tuple[Array, ...]:
    """Synthesize a scalar channel through its coordinate Hessian."""

    value, drho, dtheta, dzeta = _channel(cosine_coefficients, sine_coefficients, plan)
    cosine_jets = _radial_jets(cosine_coefficients, plan)
    sine_jets = _radial_jets(sine_coefficients, plan)

    def synthesize(
        radial_pair: tuple[Array, Array],
        angular_pair: tuple[Array, Array],
    ) -> Array:
        return jnp.einsum("rm,ma->ra", radial_pair[0], angular_pair[0]) + (
            jnp.einsum("rm,ma->ra", radial_pair[1], angular_pair[1])
        )

    radial_value, radial_first, radial_second = zip(cosine_jets, sine_jets)
    return (
        value,
        drho,
        dtheta,
        dzeta,
        synthesize(radial_second, (plan.cosine, plan.sine)),
        synthesize(radial_first, (plan.cosine_theta, plan.sine_theta)),
        synthesize(radial_first, (plan.cosine_zeta, plan.sine_zeta)),
        synthesize(
            radial_value,
            (plan.cosine_theta_theta, plan.sine_theta_theta),
        ),
        synthesize(
            radial_value,
            (plan.cosine_theta_zeta, plan.sine_theta_zeta),
        ),
        synthesize(
            radial_value,
            (plan.cosine_zeta_zeta, plan.sine_zeta_zeta),
        ),
    )


@jax.jit
def evaluate_variational_fields(
    state: HighOrderEquilibriumState,
    plan: VariationalPlan,
) -> VariationalFieldSamples:
    """Evaluate native first jets, magnetic field, and pressure by contractions."""

    R, R_rho, R_theta, R_zeta = _channel(state.R_cos, state.R_sin, plan)
    Z, Z_rho, Z_theta, Z_zeta = _channel(state.Z_cos, state.Z_sin, plan)
    _, _, lambda_theta, lambda_zeta = _channel(state.L_cos, state.L_sin, plan)
    tt, zz = jnp.meshgrid(plan.theta, plan.zeta, indexing="ij")
    phi = zz.reshape(-1) / float(plan.nfp)
    cosine_phi = jnp.cos(phi)[None]
    sine_phi = jnp.sin(phi)[None]

    def cylindrical(radial: Array, vertical: Array) -> Array:
        return jnp.stack((radial * cosine_phi, radial * sine_phi, vertical), axis=-1)

    position = cylindrical(R, Z)
    e_rho = cylindrical(R_rho, Z_rho)
    e_theta = cylindrical(R_theta, Z_theta)
    e_zeta = jnp.stack(
        (
            R_zeta * cosine_phi - R * sine_phi / float(plan.nfp),
            R_zeta * sine_phi + R * cosine_phi / float(plan.nfp),
            Z_zeta,
        ),
        axis=-1,
    )
    sqrt_g = jnp.sum(e_rho * jnp.cross(e_theta, e_zeta), axis=-1)
    phipf = jnp.asarray(plan.profile_basis) @ jnp.asarray(state.phipf)
    chipf = jnp.asarray(plan.profile_basis) @ jnp.asarray(state.chipf)
    pressure = jnp.asarray(plan.profile_basis) @ jnp.asarray(state.pressure)
    flux_factor = 2.0 * jnp.asarray(plan.rho)[:, None] / sqrt_g
    B_theta = flux_factor * (chipf[:, None] / float(plan.nfp) - phipf[:, None] * lambda_zeta)
    B_zeta = flux_factor * phipf[:, None] * (1.0 + lambda_theta)
    B = B_theta[..., None] * e_theta + B_zeta[..., None] * e_zeta
    shape = plan.shape
    vector_shape = shape + (3,)
    scalar_shape = shape
    return VariationalFieldSamples(
        position=position.reshape(vector_shape),
        dposition_drho=e_rho.reshape(vector_shape),
        dposition_dtheta=e_theta.reshape(vector_shape),
        dposition_dzeta=e_zeta.reshape(vector_shape),
        sqrt_g=sqrt_g.reshape(scalar_shape),
        B=B.reshape(vector_shape),
        pressure=jnp.broadcast_to(pressure[:, None, None], scalar_shape),
    )


@jax.jit
def native_tangential_gauge_residual(
    state: HighOrderEquilibriumState,
    direction: Any,
    gauge: NativeGaugePlan,
) -> Array:
    """Project frozen normalized-tangent geometry motion into gauge rows."""

    plan = gauge.variational
    delta_R, _, _, _ = _channel(direction.R_cos, direction.R_sin, plan)
    delta_Z, _, _, _ = _channel(direction.Z_cos, direction.Z_sin, plan)
    fields = evaluate_variational_fields(state, plan)
    _, zz = jnp.meshgrid(plan.theta, plan.zeta, indexing="ij")
    phi = zz.reshape(-1) / float(plan.nfp)
    delta_position = jnp.stack(
        (
            delta_R * jnp.cos(phi)[None],
            delta_R * jnp.sin(phi)[None],
            delta_Z,
        ),
        axis=-1,
    ).reshape(fields.position.shape)
    tangent = fields.dposition_dtheta
    tangent_norm = jnp.sqrt(jnp.sum(tangent * tangent, axis=-1) + float(gauge.tangent_floor) ** 2)
    tangential_motion = jnp.sum(delta_position * tangent, axis=-1) / tangent_norm
    weighted = (jnp.broadcast_to(jnp.asarray(plan.quadrature_weights), plan.shape) * tangential_motion).reshape(
        (plan.shape[0], -1)
    )
    projected = jnp.einsum(
        "ra,mrb,ma->mb",
        weighted,
        jnp.asarray(plan.radial_value),
        jnp.asarray(plan.sine),
    )
    return projected[
        jnp.asarray(gauge.row_modes),
        jnp.asarray(gauge.row_basis),
    ] * jnp.asarray(gauge.row_scale)


@jax.jit
def native_physical_force_residual(
    coordinates: Array,
    base_state: HighOrderEquilibriumState,
    layout: NativeCorrectionLayout,
    gauge: NativeGaugePlan,
    coordinate_scale: Array,
    force_scale: Array,
    volume_scale: Array,
) -> Array:
    """Return volume-weighted Cartesian force on the native state.

    Keeping all three Cartesian components preserves the physical force norm
    without assuming the radial and helical decomposition vectors are
    orthogonal.
    """

    coordinates = jnp.asarray(coordinates)
    coordinate_scale = jnp.asarray(coordinate_scale)
    if coordinates.shape != (layout.size,):
        raise ValueError(f"force coordinates have shape {coordinates.shape}; expected {(layout.size,)}")
    if coordinate_scale.shape != (layout.size,):
        raise ValueError(f"coordinate_scale has shape {coordinate_scale.shape}; expected {(layout.size,)}")
    scale = jnp.broadcast_to(jnp.asarray(force_scale), (3,))
    correction = layout.unpack(coordinate_scale * coordinates)
    if gauge.variational.spline_value is not None:
        # Accurate mode: the certified object is the (base, coordinates) pair.
        samples = evaluate_tensorized_strong_force(base_state, gauge.variational, correction)
    else:
        state = apply_high_order_correction(base_state, correction)
        samples = evaluate_tensorized_strong_force(state, gauge.variational)
    volume_weights = (
        jnp.broadcast_to(
            jnp.asarray(gauge.variational.quadrature_weights),
            gauge.variational.shape,
        )
        * float(base_state.jacobian_sign)
        * samples.sqrt_g
    )
    volume_scale = jnp.asarray(volume_scale)
    normalized_weight = jnp.sqrt(volume_weights / volume_scale)
    return (normalized_weight[..., None] * samples.force / scale).reshape(-1)


@jax.jit
def evaluate_tensorized_strong_force(
    state: HighOrderEquilibriumState,
    plan: VariationalPlan,
    correction: HighOrderCorrection | None = None,
) -> StrongForceSamples:
    """Evaluate strong force from analytic coefficient-to-second-jet tables.

    Only a local forward-mode chain rule is used to differentiate the
    covariant magnetic components.  No spatial coordinate is passed through
    nested pointwise AD, and the independent point oracle remains unchanged.

    With ``correction`` the geometry is ``state + correction`` but the two
    coefficient sets are synthesized separately and added as jets.  Because the
    jets are linear in coefficients this is the same field; it avoids rounding
    the O(1) coefficient sum, whose one-ulp error the second radial derivative
    tables amplify by ~1/ds**2.
    """

    def channel(cosine: str, sine: str):
        jets = _channel_second(getattr(state, cosine), getattr(state, sine), plan)
        if correction is None:
            return jets
        extra = _channel_second(getattr(correction, cosine), getattr(correction, sine), plan)
        return tuple(a + b for a, b in zip(jets, extra))

    R = channel("R_cos", "R_sin")
    Z = channel("Z_cos", "Z_sin")
    L = channel("L_cos", "L_sin")
    _, zz = jnp.meshgrid(plan.theta, plan.zeta, indexing="ij")
    phi = zz.reshape(-1) / float(plan.nfp)
    cosine_phi = jnp.cos(phi)[None]
    sine_phi = jnp.sin(phi)[None]
    inverse_nfp = 1.0 / float(plan.nfp)

    def cylindrical(radial: Array, vertical: Array) -> Array:
        return jnp.stack((radial * cosine_phi, radial * sine_phi, vertical), axis=-1)

    def zeta_derivative(radial: Array, radial_zeta: Array, vertical: Array) -> Array:
        return jnp.stack(
            (
                radial_zeta * cosine_phi - radial * sine_phi * inverse_nfp,
                radial_zeta * sine_phi + radial * cosine_phi * inverse_nfp,
                vertical,
            ),
            axis=-1,
        )

    e_rho = cylindrical(R[1], Z[1])
    e_theta = cylindrical(R[2], Z[2])
    e_zeta = zeta_derivative(R[0], R[3], Z[3])
    e_rho_rho = cylindrical(R[4], Z[4])
    e_rho_theta = cylindrical(R[5], Z[5])
    e_rho_zeta = zeta_derivative(R[1], R[6], Z[6])
    e_theta_theta = cylindrical(R[7], Z[7])
    e_theta_zeta = zeta_derivative(R[2], R[8], Z[8])
    e_zeta_zeta = jnp.stack(
        (
            R[9] * cosine_phi - 2.0 * R[3] * sine_phi * inverse_nfp - R[0] * cosine_phi * inverse_nfp**2,
            R[9] * sine_phi + 2.0 * R[3] * cosine_phi * inverse_nfp - R[0] * sine_phi * inverse_nfp**2,
            Z[9],
        ),
        axis=-1,
    )

    rho = jnp.broadcast_to(jnp.asarray(plan.rho)[:, None], R[0].shape)
    phipf = jnp.asarray(plan.profile_basis) @ jnp.asarray(state.phipf)
    chipf = jnp.asarray(plan.profile_basis) @ jnp.asarray(state.chipf)
    phipf_rho = jnp.asarray(plan.profile_derivative) @ jnp.asarray(state.phipf)
    chipf_rho = jnp.asarray(plan.profile_derivative) @ jnp.asarray(state.chipf)
    pressure_rho = jnp.asarray(plan.profile_derivative) @ jnp.asarray(state.pressure)
    phipf = jnp.broadcast_to(phipf[:, None], R[0].shape)
    chipf = jnp.broadcast_to(chipf[:, None], R[0].shape)
    phipf_rho = jnp.broadcast_to(phipf_rho[:, None], R[0].shape)
    chipf_rho = jnp.broadcast_to(chipf_rho[:, None], R[0].shape)

    def magnetic_covariant(
        local_rho,
        local_e_rho,
        local_e_theta,
        local_e_zeta,
        lambda_theta,
        lambda_zeta,
        local_phipf,
        local_chipf,
    ):
        sqrt_g = jnp.sum(local_e_rho * jnp.cross(local_e_theta, local_e_zeta), axis=-1)
        factor = 2.0 * local_rho / sqrt_g
        B_theta = factor * (local_chipf * inverse_nfp - local_phipf * lambda_zeta)
        B_zeta = factor * local_phipf * (1.0 + lambda_theta)
        B = B_theta[..., None] * local_e_theta + B_zeta[..., None] * local_e_zeta
        covariant = jnp.stack(
            (
                jnp.sum(B * local_e_rho, axis=-1),
                jnp.sum(B * local_e_theta, axis=-1),
                jnp.sum(B * local_e_zeta, axis=-1),
            ),
            axis=-1,
        )
        return covariant, B, sqrt_g, B_theta, B_zeta

    primals = (rho, e_rho, e_theta, e_zeta, L[2], L[3], phipf, chipf)
    values, radial_derivative = jax.jvp(
        magnetic_covariant,
        primals,
        (
            jnp.ones_like(rho),
            e_rho_rho,
            e_rho_theta,
            e_rho_zeta,
            L[5],
            L[6],
            phipf_rho,
            chipf_rho,
        ),
    )
    _, theta_derivative = jax.jvp(
        magnetic_covariant,
        primals,
        (
            jnp.zeros_like(rho),
            e_rho_theta,
            e_theta_theta,
            e_theta_zeta,
            L[7],
            L[8],
            jnp.zeros_like(phipf),
            jnp.zeros_like(chipf),
        ),
    )
    _, zeta_derivative_values = jax.jvp(
        magnetic_covariant,
        primals,
        (
            jnp.zeros_like(rho),
            e_rho_zeta,
            e_theta_zeta,
            e_zeta_zeta,
            L[8],
            L[9],
            jnp.zeros_like(phipf),
            jnp.zeros_like(chipf),
        ),
    )
    B_covariant, B, sqrt_g, B_theta, B_zeta = values
    del B_covariant
    dB_rho = radial_derivative[0]
    dB_theta = theta_derivative[0]
    dB_zeta = zeta_derivative_values[0]
    curl_numerator = jnp.stack(
        (
            dB_theta[..., 2] - dB_zeta[..., 1],
            dB_zeta[..., 0] - dB_rho[..., 2],
            dB_rho[..., 1] - dB_theta[..., 0],
        ),
        axis=-1,
    )
    J_sup = curl_numerator / (MU0 * sqrt_g[..., None])
    J = J_sup[..., 0, None] * e_rho + J_sup[..., 1, None] * e_theta + J_sup[..., 2, None] * e_zeta
    grad_rho = jnp.cross(e_theta, e_zeta) / sqrt_g[..., None]
    grad_theta = jnp.cross(e_zeta, e_rho) / sqrt_g[..., None]
    grad_zeta = jnp.cross(e_rho, e_theta) / sqrt_g[..., None]
    grad_pressure = pressure_rho[:, None, None] * grad_rho
    lorentz = jnp.cross(J, B)
    force = lorentz - grad_pressure
    force_rho = jnp.sum(force * e_rho, axis=-1)
    force_helical = curl_numerator[..., 0] / MU0
    radial_force = force_rho[..., None] * grad_rho
    helical_direction = -B_zeta[..., None] * grad_theta + B_theta[..., None] * grad_zeta
    helical_force = force_helical[..., None] * helical_direction
    shape = plan.shape
    return StrongForceSamples(
        rho=jnp.broadcast_to(jnp.asarray(plan.rho)[:, None, None], shape),
        theta=jnp.broadcast_to(jnp.asarray(plan.theta)[None, :, None], shape),
        zeta=jnp.broadcast_to(jnp.asarray(plan.zeta)[None, None, :], shape),
        sqrt_g=sqrt_g.reshape(shape),
        B=B.reshape(shape + (3,)),
        J=J.reshape(shape + (3,)),
        force=force.reshape(shape + (3,)),
        force_rho=force_rho.reshape(shape),
        force_helical=force_helical.reshape(shape),
        radial_force_density=jnp.linalg.norm(radial_force, axis=-1).reshape(shape),
        helical_force_density=jnp.linalg.norm(helical_force, axis=-1).reshape(shape),
        signed_radial_force_density=(force_rho * jnp.linalg.norm(grad_rho, axis=-1)).reshape(shape),
        signed_helical_force_density=(force_helical * jnp.linalg.norm(helical_direction, axis=-1)).reshape(shape),
        lorentz_norm=jnp.linalg.norm(lorentz, axis=-1).reshape(shape),
        grad_pressure_norm=jnp.linalg.norm(grad_pressure, axis=-1).reshape(shape),
    )


@jax.jit
def minimum_signed_jacobian(
    state: HighOrderEquilibriumState,
    plan: VariationalPlan,
) -> Array:
    """Return the minimum Jacobian after applying the declared orientation."""

    fields = evaluate_variational_fields(state, plan)
    return jnp.min(float(plan.jacobian_sign) * fields.sqrt_g)


__all__ = [
    "VariationalFieldSamples",
    "NativeGaugePlan",
    "VariationalPlan",
    "evaluate_tensorized_strong_force",
    "evaluate_variational_fields",
    "make_variational_plan",
    "make_native_gauge_plan",
    "minimum_signed_jacobian",
    "native_tangential_gauge_residual",
    "native_coordinate_scales",
    "native_tangential_gauge_matrix",
    "native_physical_force_residual",
]
