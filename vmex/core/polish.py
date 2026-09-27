"""Force-balance polish: certified continuum force balance of a VMEX solve.

The ordinary VMEX solve converges a discrete projected variational problem;
its tiny FSQR/FSQZ/FSQL do not bound the continuum force
``F = J x B - grad p``.  After the solve, :func:`polish_legacy_solution`
lifts the converged state into a native axis-regular spline representation
(``a_m(rho) = rho**|m| q(rho**2)``) and minimizes the volume-weighted
physical force::

    r_i(c) = sqrt(w_i |sqrt g_i| / V) F_i(c) / F*,   min 0.5 |r|^2   s.t.   C c = 0

over packed coefficient corrections ``c`` in a fixed metric, with ``C`` the
frozen linear tangential (relabeling) gauge and exact structural boundary and
axis constraints.  Each chart takes Gauss-Newton KKT steps whose normal matrix
is formed by partial assembly (``A = D S``, see ``_normal_system``); charts
are refined exactly (angular zero padding, then knot insertion where the force
is largest).  A final exact-Hessian Newton step on the stationarity equations
``[A^T r + C^T nu, C c] = 0`` brings the Frobenius-scaled projected gradient::

    eta = |P A^T r| / (|A|_F |r|)

below ``PolishConfig.stationarity_tolerance``.  Radial derivatives are formed
from coefficient differences and the base and correction jets are synthesized
separately, which keeps the evaluated stationarity accurate to ~1e-12.  The
result is certified with the independent oracle of
:mod:`vmex.core.strong_force` and exported to WOUT on a mesh that carries it.

Scope: fixed boundary, axisymmetric, prescribed pressure and iota
(``NCURR = 0``, ``GAMMA = 0``), stellarator-symmetric.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse
from scipy.linalg import blas
from scipy.sparse import linalg as sparse_linalg

from .errors import StrongForceCertificationError, VmecInputError
from .printing import emit_flushed, polish_certificate_summary
from .profiles import MU0
from .radial_basis import _basis_levels
from .residuals import m1_physical_to_constrained
from .solver import SpectralState
from .strong_force import (
    HighOrderEquilibriumState,
    StrongForceReport,
    StrongForceSamples,
    append_high_order_state_modes,
    certify_strong_force,
    force_error_measures,
    insert_high_order_state_knots,
    lift_high_order_state,
)
from .transforms import physical_to_internal_scale

Array = Any
_FIELDS = ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")


@dataclass(frozen=True)
class HighOrderCorrection:
    """Regularized spline coefficients for one geometry/lambda correction."""

    R_cos: Array
    R_sin: Array
    Z_cos: Array
    Z_sin: Array
    L_cos: Array
    L_sin: Array


jax.tree_util.register_dataclass(
    HighOrderCorrection,
    data_fields=list(_FIELDS),
    meta_fields=[],
)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class NativeCorrectionLayout:
    """Direct structural coordinates for native spline corrections.

    Unlike :class:`StrongRootLayout`, this layout is not defined as the image
    of a legacy restriction/prolongation pair.  It removes only exact
    structural nulls, the fixed R/Z edge coefficients, stellarator-asymmetric
    channels when requested, and the angle-independent lambda gauge.  In
    particular, independent R and Z normal displacements remain present.
    """

    mnmax: int
    nbasis: int
    active_indices: np.ndarray
    lasym: bool

    def tree_flatten(self):
        """Expose active indices as data and topology as metadata."""

        return (jnp.asarray(self.active_indices),), (
            int(self.mnmax),
            int(self.nbasis),
            bool(self.lasym),
        )

    @classmethod
    def tree_unflatten(cls, metadata, children):
        """Rebuild a native layout from its pytree representation."""

        mnmax, nbasis, lasym = metadata
        (active_indices,) = children
        return cls(
            mnmax=mnmax,
            nbasis=nbasis,
            active_indices=active_indices,
            lasym=lasym,
        )

    @property
    def size(self) -> int:
        """Number of structurally free native coefficients."""

        return int(self.active_indices.size)

    def pack(self, correction: HighOrderCorrection) -> Array:
        """Select free coefficients from a native correction."""

        flat = _flatten_high(correction)
        return flat[jnp.asarray(self.active_indices)]

    def unpack(self, vector: Array) -> HighOrderCorrection:
        """Insert a free vector into the six native coefficient tables."""

        vector = jnp.asarray(vector)
        if vector.shape != (self.size,):
            raise ValueError(
                f"free vector has shape {vector.shape}; expected {(self.size,)}"
            )
        flat = jnp.zeros(
            (len(_FIELDS) * self.mnmax * self.nbasis,), dtype=vector.dtype
        )
        flat = flat.at[jnp.asarray(self.active_indices)].set(vector)
        return _unflatten_high(flat, self.mnmax, self.nbasis)


def make_native_correction_layout(
    native: HighOrderEquilibriumState,
    *,
    lasym: bool = False,
) -> NativeCorrectionLayout:
    """Construct full native R/Z/lambda structural coordinates.

    A clamped spline's final coefficient is its value at ``rho=1``; removing
    that coefficient from R and Z corrections enforces the fixed boundary
    algebraically.  Lambda remains free at the edge, except for its complete
    ``(m,n)=(0,0)`` gauge family.
    """

    mnmax = int(np.asarray(native.m).size)
    nbasis = int(native.radial_basis.size)
    active = np.ones((len(_FIELDS), mnmax, nbasis), dtype=bool)
    if not lasym:
        for name in ("R_sin", "Z_cos", "L_cos"):
            active[_FIELDS.index(name)] = False
    mode_zero = (np.asarray(native.m, dtype=int) == 0) & (
        np.asarray(native.n, dtype=int) == 0
    )
    for name in ("R_sin", "Z_sin", "L_sin"):
        active[_FIELDS.index(name), mode_zero] = False
    for name in ("R_cos", "R_sin", "Z_cos", "Z_sin"):
        active[_FIELDS.index(name), :, -1] = False
    for name in ("L_cos", "L_sin"):
        active[_FIELDS.index(name), mode_zero] = False
    return NativeCorrectionLayout(
        mnmax=mnmax,
        nbasis=nbasis,
        active_indices=np.flatnonzero(active.reshape(-1)).astype(np.int32),
        lasym=bool(lasym),
    )


def _flatten_high(correction: HighOrderCorrection) -> Array:
    return jnp.concatenate(
        tuple(jnp.ravel(jnp.asarray(getattr(correction, name))) for name in _FIELDS)
    )


def _unflatten_high(vector: Array, mnmax: int, nbasis: int) -> HighOrderCorrection:
    vector = jnp.asarray(vector)
    block = int(mnmax) * int(nbasis)
    values = [
        vector[index * block : (index + 1) * block].reshape((mnmax, nbasis))
        for index in range(len(_FIELDS))
    ]
    return HighOrderCorrection(*values)




def _mode_table(m: np.ndarray, n: np.ndarray):
    """Construct only the mode metadata needed by the m=1 linear maps."""

    from .fourier import ModeTable

    return ModeTable(m=np.asarray(m), n=np.asarray(n))


def sample_high_order_state(
    native: HighOrderEquilibriumState,
    runtime: Any,
) -> SpectralState:
    """Evaluate a continuous native state on ``runtime``'s legacy full mesh.

    The exact inverse of the representation changes performed by
    :func:`~vmex.core.strong_force.lift_high_order_state`: clamped-spline
    evaluation with the ``rho**abs(m)`` regularity factor, VMEX Fourier
    normalization, the m=1 constrained variables, and the internal
    ``phipf/lamscale`` lambda scaling.  Unlike :meth:`HighLowTransfer.restrict`
    this samples a full equilibrium rather than a correction, so the boundary
    row and the non-evolved degrees of freedom are kept, not zeroed.  The
    target mesh is whatever ``runtime`` was prepared with — it does not have
    to match the mesh the native state was lifted from.
    """

    modes = runtime.modes
    m = np.asarray(modes.m, dtype=int)
    n = np.asarray(modes.n, dtype=int)
    if not np.array_equal(m, np.asarray(native.m)) or not np.array_equal(
        n, np.asarray(native.n)
    ):
        raise ValueError("native and legacy mode tables must match")
    setup = runtime.setup
    s = np.asarray(setup.s_full, dtype=float)
    rho = np.sqrt(np.maximum(s, 0.0))
    basis_values = np.asarray(native.radial_basis.basis_matrix(s), dtype=float)
    evaluation = jnp.asarray(
        rho[None, :, None] ** np.abs(m)[:, None, None] * basis_values[None]
    )

    def sample(coefficients: Array) -> Array:
        return jnp.einsum(
            "msk,mk->sm",
            evaluation,
            jnp.asarray(coefficients),
            precision=jax.lax.Precision.HIGHEST,
        )

    scale = jnp.asarray(physical_to_internal_scale(modes, runtime.trig))[None, :]
    R_cos, Z_sin, R_sin, Z_cos = m1_physical_to_constrained(
        sample(native.R_cos) * scale,
        sample(native.Z_sin) * scale,
        sample(native.R_sin) * scale,
        sample(native.Z_cos) * scale,
        modes=_mode_table(m, n),
        lthreed=bool(setup.lthreed),
        lasym=bool(setup.lasym),
        lconm1=bool(setup.lconm1),
    )
    lambda_scale = (
        scale * jnp.asarray(setup.phipf)[:, None] / jnp.asarray(setup.lamscale)
    )
    return SpectralState(
        R_cos=R_cos,
        R_sin=R_sin,
        Z_cos=Z_cos,
        Z_sin=Z_sin,
        L_cos=sample(native.L_cos) * lambda_scale,
        L_sin=sample(native.L_sin) * lambda_scale,
    )


def apply_high_order_correction(
    native: HighOrderEquilibriumState,
    correction: HighOrderCorrection,
) -> HighOrderEquilibriumState:
    """Add a constrained geometry correction while leaving profiles fixed."""

    return replace(
        native,
        R_cos=native.R_cos + correction.R_cos,
        R_sin=native.R_sin + correction.R_sin,
        Z_cos=native.Z_cos + correction.Z_cos,
        Z_sin=native.Z_sin + correction.Z_sin,
        L_cos=native.L_cos + correction.L_cos,
        L_sin=native.L_sin + correction.L_sin,
        source=f"{native.source}; strong-root correction",
    )




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
) -> VariationalPlan:
    """Build a tensor grid, coefficient-to-jet tables, and quadrature weights.

    Field kernels differentiate spline coefficients by local differences
    before evaluating lower-degree splines (``_stable_radial_jets``); the
    assembled derivative tables serve the linear maps (gauge, scales, and the
    Jacobian's synthesis), where their cancellation is harmless.
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
    return VariationalPlan(
        **_stable_radial_tables(state.radial_basis, s, rho, m),
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
    levels = _basis_levels(knots, s, degree)
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


def _channel(
    cosine_coefficients: Array,
    sine_coefficients: Array,
    plan: VariationalPlan,
) -> tuple[Array, Array, Array, Array]:
    """Synthesize one scalar channel and its three coordinate derivatives."""

    cosine_radial, cosine_drho, _ = _stable_radial_jets(cosine_coefficients, plan)
    sine_radial, sine_drho, _ = _stable_radial_jets(sine_coefficients, plan)

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
    cosine_jets = _stable_radial_jets(cosine_coefficients, plan)
    sine_jets = _stable_radial_jets(sine_coefficients, plan)

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
    # The certified object is the (base, coordinates) pair: base and correction
    # jets are synthesized separately (see evaluate_tensorized_strong_force).
    correction = layout.unpack(coordinate_scale * coordinates)
    samples = evaluate_tensorized_strong_force(base_state, gauge.variational, correction)
    return _weighted_force(samples, gauge.variational, base_state.jacobian_sign, force_scale, volume_scale).reshape(-1)


def _weighted_force(samples, plan: VariationalPlan, jacobian_sign, force_scale, volume_scale) -> Array:
    """``sqrt(w |sqrt g| / V) F / F*`` with shape ``plan.shape + (3,)``."""

    weights = jnp.broadcast_to(jnp.asarray(plan.quadrature_weights), plan.shape) * float(jacobian_sign) * samples.sqrt_g
    return jnp.sqrt(weights / jnp.asarray(volume_scale))[..., None] * samples.force / jnp.asarray(force_scale)


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

    return _samples_from_jets(channel("R_cos", "R_sin"), channel("Z_cos", "Z_sin"), channel("L_cos", "L_sin"), plan, state)


def _samples_from_jets(R, Z, L, plan: VariationalPlan, state: HighOrderEquilibriumState) -> StrongForceSamples:
    """Pointwise strong force from the R, Z, lambda second jets (10 arrays each).

    Each node's force depends only on that node's 30 jets; the Jacobian of the
    force residual is therefore ``A = D S`` with ``D`` pointwise and ``S`` the
    fixed synthesis tables (see ``_normal_system``).
    """

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

@dataclass(frozen=True)
class PolishConfig:
    """Tolerances and schedule of the force-balance polish.

    ``force_tolerance`` bounds the volume-RMS force over the wout scale
    ``volavgB**2 / (mu0 Aminor_p)`` and ``stationarity_tolerance`` the
    projected gradient ``eta``.  ``angular_padding`` zero-padded poloidal modes
    are added once; then up to ``max_refinements`` charts each insert
    ``insert_count`` knots at the midpoints of the largest-force spans.
    ``fail_policy="return_unpolished"`` returns the unpolished state instead of
    raising when a tolerance is missed.
    """

    force_tolerance: float = 1.0e-5
    stationarity_tolerance: float = 1.0e-8
    degree: int = 5
    angular_padding: int = 8
    insert_count: int = 16
    max_refinements: int = 6
    steps_per_chart: int = 2
    max_newton_steps: int = 4
    fail_policy: str = "raise"

    def __post_init__(self) -> None:
        if self.fail_policy not in ("raise", "return_unpolished"):
            raise ValueError("fail_policy must be 'raise' or 'return_unpolished'")
        if self.degree not in (3, 5, 7):
            raise ValueError("degree must be 3, 5 or 7")
        if min(self.force_tolerance, self.stationarity_tolerance) <= 0.0:
            raise ValueError("tolerances must be positive")


class NativePolishResult(NamedTuple):
    """Polished state, final force norm and stationarity, and work counters."""

    state: HighOrderEquilibriumState
    force_norm: float
    stationarity: float
    converged: bool
    charts: int
    gauss_newton_steps: int
    newton_steps: int
    seconds: float


class _Chart(NamedTuple):
    base: HighOrderEquilibriumState
    plan: VariationalPlan
    layout: Any
    gauge: NativeGaugePlan
    scale: jax.Array
    constraint: sparse.csr_matrix
    force: Callable[[jax.Array], jax.Array]
    pattern: dict  # per-chart CSR scatter of the span blocks, built on first use


def _chart(state, force_scale: float, volume_scale: float) -> _Chart:
    degree = int(state.radial_basis.degree)
    plan = make_variational_plan(state, radial_order=degree + 3)
    layout = make_native_correction_layout(state)
    gauge = make_native_gauge_plan(state, plan)
    scale = native_coordinate_scales(state, layout, plan)
    constraint = sparse.csr_matrix(native_tangential_gauge_matrix(state, layout, gauge, scale))

    def force(coordinates):
        return native_physical_force_residual(
            coordinates, state, layout, gauge, scale, force_scale, volume_scale
        )

    return _Chart(state, plan, layout, gauge, scale, constraint, force, {})


def _state(chart: _Chart, coordinates) -> HighOrderEquilibriumState:
    return apply_high_order_correction(
        chart.base, chart.layout.unpack(chart.scale * jnp.asarray(coordinates))
    )


# Radial-table index and angular table of each of the 10 jets, in the order of
# ``_channel_second``: value, d/drho, d/dtheta, d/dzeta, then the six second
# derivatives (rho rho, rho theta, rho zeta, theta theta, theta zeta, zeta zeta).
_JET_RADIAL = (0, 1, 0, 0, 2, 1, 1, 0, 0, 0)
_JET_ANGULAR = ("", "", "_theta", "_zeta", "", "_theta", "_zeta", "_theta_theta", "_theta_zeta", "_zeta_zeta")


@jax.jit
def _jet_jacobian(base, correction, plan, force_scale, volume_scale):
    """``D[k] = d r / d jet_k`` at every node, shape ``(30,) + plan.shape + (3,)``.

    The residual is pointwise in the jets, so one forward tangent per jet
    (30 in all) gives the full pointwise derivative.
    """

    def jets(field):
        base_jets = _channel_second(getattr(base, field + "_cos"), getattr(base, field + "_sin"), plan)
        extra = _channel_second(getattr(correction, field + "_cos"), getattr(correction, field + "_sin"), plan)
        return jnp.stack([a + b for a, b in zip(base_jets, extra)])

    stacked = jnp.stack([jets("R"), jets("Z"), jets("L")])

    def residual(values):
        samples = _samples_from_jets(tuple(values[0]), tuple(values[1]), tuple(values[2]), plan, base)
        return _weighted_force(samples, plan, base.jacobian_sign, force_scale, volume_scale)

    def tangent(k):
        direction = (jnp.arange(30) == k).astype(stacked.dtype).reshape(3, 10, 1, 1) * jnp.ones_like(stacked)
        return jax.jvp(residual, (stacked,), (direction,))[1]

    return jax.lax.map(tangent, jnp.arange(30))


def _normal_system(chart: _Chart, coordinates, residual, force_scale, volume_scale):
    """Gauss-Newton normal matrix ``A^T A`` and gradient ``A^T r`` by partial assembly.

    ``A = D S``: ``D`` is the pointwise jet derivative (``_jet_jacobian``) and
    ``S`` the tensor-product synthesis ``radial table x angular table`` of each
    coefficient.  Each radial span's dense block ``G = D S`` is formed from
    small contractions on its supported columns and accumulated as ``G^T G``;
    the global Jacobian is never formed.
    """

    plan, layout = chart.plan, chart.layout
    correction = layout.unpack(chart.scale * jnp.asarray(coordinates))
    jet = np.asarray(_jet_jacobian(chart.base, correction, plan, force_scale, volume_scale))
    nrho = int(plan.rho.size)
    jet = jet.reshape(3, 10, nrho, -1, 3)
    radial = [np.asarray(t) for t in (plan.radial_value, plan.radial_derivative, plan.radial_second_derivative)]
    angular = {
        parity: [np.asarray(getattr(plan, parity + suffix)) for suffix in _JET_ANGULAR]
        for parity in ("cosine", "sine")
    }
    active = np.asarray(layout.active_indices, dtype=np.int64)
    nbasis, mnmax = int(layout.nbasis), int(layout.mnmax)
    channel, remainder = np.divmod(active, mnmax * nbasis)
    mode, basis = np.divmod(remainder, nbasis)
    support = (radial[0] != 0.0) | (radial[1] != 0.0) | (radial[2] != 0.0)  # (mode, rho, basis)
    scale = np.asarray(chart.scale)
    spans = int(chart.base.radial_basis.breakpoints.size - 1)
    order = nrho // spans
    residual = np.asarray(residual).reshape(nrho, -1)
    cache = chart.pattern
    if not cache:
        # The span supports are fixed per chart: build the CSR pattern of their
        # union and each block's positions in its data array once.
        cache["columns"] = [np.flatnonzero(support[mode, span * order:(span + 1) * order, basis].any(axis=1))
                            for span in range(spans)]
        keys = [np.add.outer(c * layout.size, c).reshape(-1) for c in cache["columns"]]
        union = np.unique(np.concatenate(keys))
        cache["indices"] = union % layout.size
        cache["indptr"] = np.searchsorted(union, np.arange(layout.size + 1) * layout.size)
        cache["positions"] = [np.searchsorted(union, k) for k in keys]
    data = np.zeros(cache["indices"].size)
    gradient = np.zeros(layout.size)
    for span in range(spans):
        nodes = slice(span * order, (span + 1) * order)
        columns = cache["columns"][span]
        block = np.empty((order, jet.shape[3], 3, columns.size))
        # Active columns are channel-major, so each channel is a contiguous range.
        for ch in np.unique(channel[columns]):
            where = np.flatnonzero(channel[columns] == ch)
            pick = columns[where]
            field, parity = divmod(int(ch), 2)
            tables = angular["sine" if parity else "cosine"]
            synthesis = np.stack([
                radial[_JET_RADIAL[k]][mode[pick], nodes, basis[pick]][:, :, None] * tables[k][mode[pick]][:, None, :]
                for k in range(10)
            ])  # (jet, column, rho, angle)
            block[..., where[0] : where[-1] + 1] = np.einsum(
                "krac,kjra->racj", jet[field, :, nodes], synthesis, optimize=True
            )
        block = block.reshape(-1, columns.size) * scale[columns]
        gram = blas.dsyrk(1.0, block, trans=1)  # upper triangle of block^T block
        gram = np.triu(gram) + np.triu(gram, 1).T
        data[cache["positions"][span]] += gram.reshape(-1)
        gradient[columns] += block.T @ residual[nodes].reshape(-1)
    normal = sparse.csr_matrix((data, cache["indices"], cache["indptr"]), shape=(layout.size, layout.size))
    return normal, gradient


def _kkt_factor(normal, constraint):
    return sparse_linalg.splu(sparse.bmat([[normal, constraint.T], [constraint, None]], format="csc"))


class _Projector:
    """Euclidean projector onto ker C from one sparse factor per chart."""

    def __init__(self, constraint):
        n = constraint.shape[1]
        self.q = constraint.shape[0]
        self.lu = sparse_linalg.splu(
            sparse.bmat([[sparse.eye(n, format="csc"), constraint.T], [constraint, None]], format="csc")
        )

    def __call__(self, vector):
        vector = np.asarray(vector, dtype=float)
        return self.lu.solve(np.concatenate((vector, np.zeros(self.q))))[: vector.size]


def _admissible(chart: _Chart, candidate) -> bool:
    gauge = float(jnp.linalg.norm(native_tangential_gauge_residual(
        chart.base, chart.layout.unpack(chart.scale * candidate), chart.gauge)))
    return gauge < 1.0e-10 and float(minimum_signed_jacobian(_state(chart, candidate), chart.plan)) > 0.0


def _gauss_newton(chart: _Chart, coordinates, force_scale, volume_scale, steps: int):
    """Feasible Gauss-Newton KKT steps with backtracking on the force norm."""

    taken = 0
    for _ in range(steps):
        residual = np.asarray(chart.force(coordinates))
        norm = float(np.linalg.norm(residual))
        normal, gradient = _normal_system(chart, coordinates, residual, force_scale, volume_scale)
        defect = chart.constraint @ np.asarray(coordinates)
        step = _kkt_factor(normal, chart.constraint).solve(-np.concatenate((gradient, defect)))
        step = step[: chart.layout.size]
        for fraction in (1.0, 0.5, 0.25):
            candidate = coordinates + fraction * jnp.asarray(step)
            trial = float(jnp.linalg.norm(chart.force(candidate)))
            if trial < norm * (1.0 - 1.0e-8) and _admissible(chart, candidate):
                coordinates, taken = candidate, taken + 1
                break
        else:
            break
        if trial > norm * (1.0 - 1.0e-3):
            break
    return coordinates, taken


def _refine(chart: _Chart, coordinates, count: int):
    """Insert ``count`` knots at the midpoints of the largest-force spans.

    The chart residual is ``sqrt(w |sqrt g| / V) F / F*`` at every quadrature
    point, so its squared sum per radial span is the span's force score; no
    second force evaluation (and no compilation for a new plan) is needed.
    """

    breaks = np.asarray(chart.base.radial_basis.breakpoints)
    spans = breaks.size - 1
    residual = np.asarray(chart.force(coordinates))
    scores = (residual**2).reshape(spans, -1).sum(axis=1)
    selected = np.sort(np.argsort(scores)[-min(int(count), spans - 1):])
    return insert_high_order_state_knots(
        _state(chart, coordinates), 0.5 * (breaks[selected] + breaks[selected + 1])
    )


def _stationarity(chart: _Chart, coordinates, force_scale, volume_scale, config: PolishConfig):
    """Exact-Hessian constrained Newton on the stationarity equations."""

    gradient = jax.jit(jax.grad(lambda c: 0.5 * jnp.vdot(chart.force(c), chart.force(c))))
    projector = _Projector(chart.constraint)
    n, q = chart.constraint.shape[1], chart.constraint.shape[0]

    def measure(c):
        residual = np.asarray(chart.force(c))
        normal, _ = _normal_system(chart, c, residual, force_scale, volume_scale)
        frobenius = float(np.sqrt(max(normal.diagonal().sum(), 0.0)))
        g = np.asarray(gradient(c))
        eta = float(np.linalg.norm(projector(g))) / max(frobenius * float(np.linalg.norm(residual)), 1e-300)
        return eta, g, normal

    eta, g, normal = measure(coordinates)
    steps = 0
    for _ in range(config.max_newton_steps):
        if eta <= config.stationarity_tolerance:
            break
        hessian = jax.jit(lambda v, c=coordinates: jax.jvp(gradient, (c,), (v,))[1])
        factor = _kkt_factor(normal, chart.constraint)

        def apply(z):
            top = np.asarray(hessian(jnp.asarray(z[:n]))) + chart.constraint.T @ z[n:]
            return np.concatenate((top, chart.constraint @ z[:n]))

        rhs = -np.concatenate((g, chart.constraint @ np.asarray(coordinates)))
        solution, _ = sparse_linalg.gmres(
            sparse_linalg.LinearOperator((n + q, n + q), matvec=apply, dtype=float),
            rhs,
            M=sparse_linalg.LinearOperator((n + q, n + q), matvec=factor.solve, dtype=float),
            rtol=1.0e-10, atol=0.0, restart=40, maxiter=40,
        )
        candidate = coordinates + jnp.asarray(solution[:n])
        if not _admissible(chart, candidate):
            break
        trial_eta, trial_g, trial_normal = measure(candidate)
        if trial_eta >= eta:
            break
        coordinates, eta, g, normal, steps = candidate, trial_eta, trial_g, trial_normal, steps + 1
    return coordinates, eta, steps


def polish_native(
    state: HighOrderEquilibriumState,
    *,
    force_scale: float,
    volume_scale: float,
    config: PolishConfig | None = None,
    emit: Callable[[str], Any] | None = None,
) -> NativePolishResult:
    """Correct the physical force of a lifted native state (see module docstring).

    ``force_scale`` (N m^-3) and ``volume_scale`` (m^3) fix the metric of the
    residual; the returned ``force_norm`` is the quadrature volume-RMS force
    over ``force_scale``.  The returned state is the float64 sum of the final
    chart's base and correction; ``stationarity`` is evaluated on that
    (base, correction) pair, whose rounding to float64 coefficients is below
    every physical tolerance.
    """

    config = PolishConfig() if config is None else config
    started = perf_counter()
    say = (lambda _message: None) if emit is None else emit
    m_max = int(np.max(np.abs(np.asarray(state.m))))
    if config.angular_padding > 0:
        new_m = np.arange(m_max + 1, m_max + 1 + config.angular_padding)
        state = append_high_order_state_modes(state, new_m, np.zeros_like(new_m))
    charts = gauss_newton_steps = 0
    chart = coordinates = None
    force_norm = np.inf
    for stage in range(config.max_refinements + 1):
        if stage:
            state = _refine(chart, coordinates, config.insert_count)
        chart = _chart(state, force_scale, volume_scale)
        coordinates = jnp.zeros((chart.layout.size,))
        coordinates, taken = _gauss_newton(chart, coordinates, force_scale, volume_scale, config.steps_per_chart)
        charts, gauss_newton_steps = charts + 1, gauss_newton_steps + taken
        force_norm = float(jnp.linalg.norm(chart.force(coordinates)))
        say(f" native polish chart {charts}: basis {chart.base.radial_basis.size}, "
            f"{chart.layout.size} coordinates, |F|/F* = {force_norm:.3e}")
        if force_norm <= config.force_tolerance:
            break
    coordinates, eta, newton_steps = _stationarity(chart, coordinates, force_scale, volume_scale, config)
    force_norm = float(jnp.linalg.norm(chart.force(coordinates)))
    say(f" native polish stationarity: eta = {eta:.3e} after {newton_steps} Newton step(s)")
    return NativePolishResult(
        state=_state(chart, coordinates),
        force_norm=force_norm,
        stationarity=eta,
        converged=bool(force_norm <= config.force_tolerance and eta <= config.stationarity_tolerance),
        charts=charts,
        gauss_newton_steps=gauss_newton_steps,
        newton_steps=newton_steps,
        seconds=perf_counter() - started,
    )


def physical_scales(state: HighOrderEquilibriumState) -> tuple[float, float]:
    """``(F*, V)``: the wout ``volavgB**2 / (mu0 Aminor_p)`` and ``volume_p``.

    ``V = int |sqrt g|``, ``volavgB`` is the volume-RMS ``|B|`` and
    ``Aminor_p = sqrt(<area> / pi)`` with the cross-section area
    ``int |sqrt g| / R d rho d theta`` averaged over the toroidal angle,
    matching VMEC2000's definitions.
    """

    plan = make_variational_plan(state, radial_order=int(state.radial_basis.degree) + 3)
    fields = evaluate_variational_fields(state, plan)
    weights = jnp.broadcast_to(plan.quadrature_weights, plan.shape).reshape(plan.rho.size, -1)
    jacobian = jnp.abs(fields.sqrt_g).reshape(weights.shape)
    radius = jnp.linalg.norm(fields.position[..., :2], axis=-1).reshape(weights.shape)
    volume = jnp.sum(weights * jacobian)
    b_rms = jnp.sqrt(jnp.sum(weights * jacobian * jnp.sum(fields.B**2, axis=-1).reshape(weights.shape)) / volume)
    # The weights integrate the full torus; dividing by 2 pi averages the
    # (rho, theta) area over the toroidal angle.
    area = jnp.sum(weights * jacobian / radius) / (2.0 * np.pi)
    minor_radius = jnp.sqrt(area / np.pi)
    return float(b_rms**2 / (MU0 * minor_radius)), float(volume)


def native_polish_supported(source) -> bool:
    """Whether a deck is inside the native polish's qualified scope."""

    return (
        not bool(getattr(source, "lasym", False))
        and int(getattr(source, "ntor", 0)) == 0
        and int(getattr(source, "ncurr", 0)) == 0
        and float(getattr(source, "gamma", 0.0)) == 0.0
        and not bool(getattr(source, "lfreeb", False))
    )


@dataclass(frozen=True)
class PolishReport:
    """Summary of one polish, as plain host values.

    ``force_norm`` is the independent volume-RMS force over the wout scale
    ``volavgB**2 / (mu0 Aminor_p)``; ``stationarity`` the Frobenius-scaled
    projected gradient ``eta``.  ``initial_*``/``final_*`` are the certificate
    of the lifted VMEC state and of the polished state: ``*_normalized_l2`` is
    the pointwise ``eps_F`` (bounded above by 2 by construction), the other
    two the non-saturating averages over ``normalization_window``.
    """

    converged: bool
    termination_reason: str
    force_norm: float
    stationarity: float
    nonlinear_iterations: int
    charts: int
    minimum_signed_jacobian: float
    solve_seconds: float
    initial_normalized_l2: float
    final_normalized_l2: float
    initial_volume_average_force: float
    final_volume_average_force: float
    initial_magnetic_relative_force_error: float
    final_magnetic_relative_force_error: float
    normalization_window: tuple[float, float]


class PolishResult(NamedTuple):
    """Certified native state, its certificate, the report, and a solve-mesh view."""

    native_equilibrium: HighOrderEquilibriumState
    strong_force: StrongForceReport
    polish_report: PolishReport
    compatibility_state: Any


def polish_legacy_solution(
    source,
    resolution,
    legacy_state,
    *,
    config: PolishConfig | None = None,
    lconm1: bool = True,
    auto: bool = False,
    verbose: bool = False,
    emit: Any = emit_flushed,
) -> PolishResult | None:
    """Polish one converged fixed-boundary solve (see the module docstring).

    Supported decks are axisymmetric, fixed-boundary, with prescribed pressure
    and iota (``NCURR = 0``, ``GAMMA = 0``) and no ``LASYM``.  For any other
    deck ``auto=True`` returns ``None`` (the solve stays unpolished) and an
    explicit request raises :class:`~vmex.core.errors.VmecInputError`.  A
    polish that misses its tolerances raises
    :class:`~vmex.core.errors.StrongForceCertificationError`, or with
    ``config.fail_policy="return_unpolished"`` returns the lifted state with
    ``polish_report.converged`` false.
    """

    from . import implicit

    config = PolishConfig() if config is None else config
    if not native_polish_supported(source):
        if auto:
            return None
        raise VmecInputError(
            "force-balance polishing supports fixed-boundary axisymmetric decks with "
            "NCURR = 0, GAMMA = 0 and LASYM = F",
            hint="drop the polish request for this deck",
        )
    started = perf_counter()
    implicit_config = implicit.make_config(source, ns=int(resolution.ns), lconm1=bool(lconm1), multigrid=False)
    runtime = implicit.runtime_from_params(implicit.params_from_input(source), implicit_config)
    if verbose:
        emit(" native polish: lifting to the quintic spline basis...")
    lifted = lift_high_order_state(legacy_state, runtime, inp=source, degree=config.degree)
    initial = certify_strong_force(lifted)
    force_scale, volume_scale = physical_scales(lifted)
    result = polish_native(lifted, force_scale=force_scale, volume_scale=volume_scale,
                           config=config, emit=emit if verbose else None)
    final = certify_strong_force(result.state)
    force_norm = float(final.absolute_l2) / force_scale
    converged = bool(result.stationarity <= config.stationarity_tolerance
                     and force_norm <= config.force_tolerance
                     and float(final.minimum_signed_jacobian) > 0.0)
    window = initial.window_normalizations
    report = PolishReport(
        converged=converged,
        termination_reason="certified" if converged else "not-converged",
        force_norm=force_norm,
        stationarity=result.stationarity,
        nonlinear_iterations=result.gauss_newton_steps + result.newton_steps,
        charts=result.charts,
        minimum_signed_jacobian=float(final.minimum_signed_jacobian),
        solve_seconds=perf_counter() - started,
        initial_normalized_l2=float(initial.normalized_l2),
        final_normalized_l2=float(final.normalized_l2),
        initial_volume_average_force=float(window.volume_average_force),
        final_volume_average_force=float(final.window_normalizations.volume_average_force),
        initial_magnetic_relative_force_error=float(window.magnetic_relative_force_error),
        final_magnetic_relative_force_error=float(final.window_normalizations.magnetic_relative_force_error),
        normalization_window=(float(window.s_min), float(window.s_max)),
    )
    if verbose:
        emit(f" native polish: |F|_rms / F* = {force_norm:.3E} (tolerance {config.force_tolerance:.1E}),"
             f" eta = {result.stationarity:.3E} (tolerance {config.stationarity_tolerance:.1E}),"
             f" {report.solve_seconds:.1f} s")
        emit(polish_certificate_summary(
            report.initial_normalized_l2, report.final_normalized_l2, None,
            verdict="CERTIFIED" if converged else "FAILED",
            measures=force_error_measures(initial, final), window=report.normalization_window), end="")
    if not converged and config.fail_policy == "raise":
        raise StrongForceCertificationError(
            f"force-balance polish did not converge: |F|/F* = {force_norm:.3e}, "
            f"eta = {result.stationarity:.3e}",
            hint="inspect the polish report or use fail_policy='return_unpolished'",
            force_norm=force_norm,
            force_tolerance=config.force_tolerance,
            stationarity=result.stationarity,
            stationarity_tolerance=config.stationarity_tolerance,
        )
    native, certificate = (result.state, final) if converged else (lifted, initial)
    return PolishResult(native, certificate, report, _solve_mesh_state(native, source, runtime))


def _solve_mesh_state(native: HighOrderEquilibriumState, source, runtime):
    """The native state sampled on the solve mesh and the deck's modes.

    Padded poloidal modes the deck's ``MPOL`` cannot hold are dropped; this is
    the solve-grid view, not the certified object (see
    :func:`polished_wout_state` for the faithful export).
    """

    keep = np.abs(np.asarray(native.m)) < int(source.mpol)
    fields = ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")
    truncated = replace(native, m=np.asarray(native.m)[keep], n=np.asarray(native.n)[keep],
                        **{name: getattr(native, name)[keep] for name in fields})
    return sample_high_order_state(truncated, runtime)


#: Minimum radial surfaces for a polished WOUT export: four samples per span
#: of the default 32-span wout reconstruction.
_POLISHED_WOUT_MIN_NS = 129


def polished_wout_ns(native: HighOrderEquilibriumState, *, solve_ns: int) -> int:
    """Radial export mesh on which the samples determine ``native``.

    ``max(solve_ns, 129, 2 * size + 1, 2 / ds_min + 1)``: never coarser than
    the solve, four samples per default reconstruction span, two per spline
    coefficient, and two uniform-``s`` samples in the narrowest span.
    """

    determined = 2 * int(native.radial_basis.size) + 1
    narrowest = float(np.min(np.diff(np.asarray(native.radial_basis.breakpoints))))
    resolving = int(np.ceil(2.0 / narrowest)) + 1
    return max(int(solve_ns), _POLISHED_WOUT_MIN_NS, determined, resolving)


def polished_wout_input(native: HighOrderEquilibriumState, source):
    """The deck whose ``MPOL`` holds every mode of ``native`` (a wider wout)."""

    mpol = int(np.max(np.abs(np.asarray(native.m)))) + 1
    if mpol <= int(source.mpol):
        return source
    return source.change_resolution(mpol=mpol, ntor=int(source.ntor))


def polished_wout_state(native: HighOrderEquilibriumState, source, *, solve_ns: int):
    """Sample ``native`` on the :func:`polished_wout_ns` mesh for the wout writer."""

    from .solver import prepare_runtime, resolution_from_input

    source = polished_wout_input(native, source)
    ns = polished_wout_ns(native, solve_ns=solve_ns)
    return sample_high_order_state(native, prepare_runtime(source, resolution_from_input(source, ns=ns)))


__all__ = [
    "HighOrderCorrection",
    "NativeCorrectionLayout",
    "NativeGaugePlan",
    "NativePolishResult",
    "PolishConfig",
    "PolishReport",
    "PolishResult",
    "VariationalFieldSamples",
    "VariationalPlan",
    "apply_high_order_correction",
    "evaluate_tensorized_strong_force",
    "evaluate_variational_fields",
    "make_native_correction_layout",
    "make_native_gauge_plan",
    "make_variational_plan",
    "minimum_signed_jacobian",
    "native_coordinate_scales",
    "native_physical_force_residual",
    "native_polish_supported",
    "native_tangential_gauge_matrix",
    "native_tangential_gauge_residual",
    "polish_legacy_solution",
    "polish_native",
    "polished_wout_input",
    "polished_wout_ns",
    "polished_wout_state",
    "sample_high_order_state",
]
