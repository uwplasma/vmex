"""Straight-axis mirror geometry and divergence-free magnetic field.

The initial embedding is

``x = r cos(theta), y = r sin(theta), z = z(xi)``,
``r = sqrt(s) * a(s, theta, xi)``.

Using the regular scale ``a`` keeps ``r * d(r)/ds`` and the Jacobian finite at
the magnetic axis.  The contravariant field follows the VMEC construction and
is discretely divergence-free when theta and xi derivatives commute.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

Array = Any


def radial_derivative(values: Array, spacing: float) -> Array:
    """Second-order derivative on the uniform full radial mesh."""

    values = jnp.asarray(values)
    first = (-3.0 * values[0] + 4.0 * values[1] - values[2]) / (2.0 * spacing)
    interior = (values[2:] - values[:-2]) / (2.0 * spacing)
    last = (3.0 * values[-1] - 4.0 * values[-2] + values[-3]) / (2.0 * spacing)
    return jnp.concatenate([first[None], interior, last[None]], axis=0)


def _safe_divide(numerator: Array, denominator: Array) -> Array:
    denominator = jnp.asarray(denominator)
    mask = jnp.abs(denominator) > 32.0 * jnp.finfo(denominator.dtype).eps
    safe = jnp.where(mask, denominator, jnp.ones_like(denominator))
    return jnp.where(mask, numerator / safe, jnp.zeros_like(numerator))


@dataclass(frozen=True)
class MirrorGeometry:
    """Embedding, covariant metric, Jacobian, and volume on the full grid."""

    xyz: Array
    radius: Array
    d_radius_ds_regular: Array
    d_radius_dtheta: Array
    d_radius_dxi: Array
    g_ss: Array
    g_stheta: Array
    g_sxi: Array
    g_thetatheta: Array
    g_thetaxi: Array
    g_xixi: Array
    sqrt_g: Array
    volume: Array
    jacobian_sign_changed: Array


@dataclass(frozen=True)
class ContravariantField:
    """Mirror magnetic field in contravariant and flux-density form."""

    b_sup_s: Array
    b_sup_theta: Array
    b_sup_xi: Array
    jac_b_theta: Array
    jac_b_xi: Array


@dataclass(frozen=True)
class ClosedAxisGeometry:
    """Periodic centerline and a closure-corrected normal frame."""

    centerline: Array
    tangent: Array
    normal: Array
    binormal: Array
    speed: Array
    curvature: Array
    arc_length: Array
    frame_holonomy: Array
    closure_error: Array
    tangent_closure_error: Array
    frame_closure_error: Array


for _cls in (MirrorGeometry, ContravariantField, ClosedAxisGeometry):
    jax.tree_util.register_dataclass(
        _cls,
        data_fields=[field.name for field in fields(_cls)],
        meta_fields=[],
    )


def racetrack_centerline_coefficients(
    size: int,
    *,
    straight_length: float,
    return_radius: float,
) -> Array:
    """Return periodic cubic-spline controls for a planar racetrack axis.

    The two long legs are parallel to ``z`` at ``x=+/-return_radius``. The
    remaining controls follow semicircular returns. Four or more consecutive
    collinear controls make the interior of each leg exactly straight; the
    periodic cubic basis smooths the joins to C2 continuity.
    """

    size = int(size)
    straight_length = float(straight_length)
    return_radius = float(return_radius)
    if size < 16:
        raise ValueError("racetrack centerline requires at least 16 coefficients")
    if straight_length <= 0.0 or return_radius <= 0.0:
        raise ValueError("racetrack dimensions must be positive")

    leg_count = max(4, int(round(size * straight_length / (2.0 * straight_length + 2.0 * np.pi * return_radius))))
    leg_count = min(leg_count, (size - 8) // 2)
    return_count = (size - 2 * leg_count) // 2
    counts = [leg_count, return_count, leg_count, size - 2 * leg_count - return_count]
    half = 0.5 * straight_length

    right_z = np.linspace(-half, half, counts[0], endpoint=False)
    top_angle = np.linspace(0.0, np.pi, counts[1], endpoint=False)
    left_z = np.linspace(half, -half, counts[2], endpoint=False)
    bottom_angle = np.linspace(np.pi, 2.0 * np.pi, counts[3], endpoint=False)
    right = np.stack((np.full_like(right_z, return_radius), np.zeros_like(right_z), right_z), axis=-1)
    top = np.stack(
        (
            return_radius * np.cos(top_angle),
            np.zeros_like(top_angle),
            half + return_radius * np.sin(top_angle),
        ),
        axis=-1,
    )
    left = np.stack((np.full_like(left_z, -return_radius), np.zeros_like(left_z), left_z), axis=-1)
    bottom = np.stack(
        (
            return_radius * np.cos(bottom_angle),
            np.zeros_like(bottom_angle),
            -half + return_radius * np.sin(bottom_angle),
        ),
        axis=-1,
    )
    return jnp.asarray(np.concatenate((right, top, left, bottom), axis=0))


def _minimal_rotation(vector: Array, tangent_from: Array, tangent_to: Array) -> Array:
    """Parallel-transport ``vector`` through the shortest tangent rotation."""

    cross = jnp.cross(tangent_from, tangent_to)
    cosine = jnp.clip(jnp.dot(tangent_from, tangent_to), -1.0, 1.0)
    denominator = jnp.maximum(1.0 + cosine, 64.0 * jnp.finfo(cosine.dtype).eps)
    return vector + jnp.cross(cross, vector) + jnp.cross(cross, jnp.cross(cross, vector)) / denominator


def _rotate_about_axis(vector: Array, axis: Array, angle: Array) -> Array:
    cosine, sine = jnp.cos(angle), jnp.sin(angle)
    return vector * cosine + jnp.cross(axis, vector) * sine + axis * jnp.dot(axis, vector) * (1.0 - cosine)


def evaluate_closed_spline_axis(
    coefficients: Array,
    basis: Any,
    points: Array,
    *,
    initial_normal: Array | None = None,
) -> ClosedAxisGeometry:
    """Evaluate a periodic spline centerline and rotation-minimizing frame.

    ``basis`` must be a periodic :class:`CubicBSplineBasis`. The raw Bishop
    frame is parallel transported once around the curve. Its measured holonomy
    is then distributed uniformly over the period, producing a continuous
    periodic frame suitable for closed flux-surface coordinates.
    """

    if not getattr(basis, "periodic", False):
        raise ValueError("closed centerline requires a periodic spline basis")
    coefficients = jnp.asarray(coefficients)
    if coefficients.shape != (basis.size, 3):
        raise ValueError(f"centerline coefficients must have shape ({basis.size}, 3)")
    point_values = np.asarray(points, dtype=float)
    start, stop = basis.domain
    if point_values.ndim != 1 or point_values.size < 4:
        raise ValueError("closed centerline points must be a one-dimensional array of length >= 4")
    if np.any(np.diff(point_values) <= 0.0) or point_values[0] < start or point_values[-1] >= stop:
        raise ValueError("closed centerline points must increase within one fundamental period")

    period = stop - start
    extended_points = jnp.asarray(np.concatenate((point_values, [point_values[0] + period])))
    centerline = basis.evaluate(coefficients, extended_points, axis=0)
    first = basis.evaluate(coefficients, extended_points, derivative=1, axis=0)
    second = basis.evaluate(coefficients, extended_points, derivative=2, axis=0)
    speed = jnp.linalg.norm(first, axis=-1)
    if not isinstance(speed, jax.core.Tracer) and bool(jnp.any(speed <= 0.0)):
        raise ValueError("centerline derivative must not vanish")
    tangent = first / speed[:, None]

    if initial_normal is None:
        reference = jax.nn.one_hot(jnp.argmin(jnp.abs(tangent[0])), 3, dtype=tangent.dtype)
    else:
        reference = jnp.asarray(initial_normal, dtype=tangent.dtype)
        if reference.shape != (3,):
            raise ValueError("initial_normal must have shape (3,)")
    normal0 = reference - jnp.dot(reference, tangent[0]) * tangent[0]
    normal0 /= jnp.linalg.norm(normal0)

    def transport(normal, next_tangent):
        previous_tangent, previous_normal = normal
        next_normal = _minimal_rotation(previous_normal, previous_tangent, next_tangent)
        next_normal -= jnp.dot(next_normal, next_tangent) * next_tangent
        next_normal /= jnp.linalg.norm(next_normal)
        return (next_tangent, next_normal), next_normal

    (_, _), transported = jax.lax.scan(transport, (tangent[0], normal0), tangent[1:])
    raw_normal = jnp.concatenate((normal0[None], transported), axis=0)
    holonomy = jnp.arctan2(
        jnp.dot(tangent[0], jnp.cross(raw_normal[-1], normal0)),
        jnp.dot(raw_normal[-1], normal0),
    )
    fraction = (extended_points - extended_points[0]) / period
    normal = jax.vmap(_rotate_about_axis)(raw_normal, tangent, -holonomy * fraction)
    normal -= jnp.sum(normal * tangent, axis=-1)[:, None] * tangent
    normal /= jnp.linalg.norm(normal, axis=-1)[:, None]
    binormal = jnp.cross(tangent, normal)

    delta = jnp.diff(extended_points)
    arc_length = jnp.sum(0.5 * (speed[:-1] + speed[1:]) * delta)
    curvature = jnp.linalg.norm(jnp.cross(first, second), axis=-1) / speed**3
    return ClosedAxisGeometry(
        centerline=centerline[:-1],
        tangent=tangent[:-1],
        normal=normal[:-1],
        binormal=binormal[:-1],
        speed=speed[:-1],
        curvature=curvature[:-1],
        arc_length=arc_length,
        frame_holonomy=holonomy,
        closure_error=jnp.linalg.norm(centerline[-1] - centerline[0]),
        tangent_closure_error=jnp.linalg.norm(tangent[-1] - tangent[0]),
        frame_closure_error=jnp.linalg.norm(normal[-1] - normal[0]),
    )


def evaluate_geometry(state: "MirrorState", grid: "MirrorGrid") -> MirrorGeometry:
    """Evaluate axisymmetric or theta-dependent straight-axis geometry."""

    state.validate_shape(grid)
    a = jnp.asarray(state.radius_scale)
    sqrt_s = jnp.sqrt(jnp.asarray(grid.s))[:, None, None]
    radius = sqrt_s * a
    d_a_dtheta = grid.theta_basis.differentiate(a, axis=1)
    d_a_dxi = grid.axial_basis.differentiate(a, axis=2)
    d_radius_dtheta = sqrt_s * d_a_dtheta
    d_radius_dxi = sqrt_s * d_a_dxi

    ds = float(grid.s[1] - grid.s[0])
    # r * r_s is regular even though r_s itself is singular at s=0.
    r_r_s = 0.5 * radial_derivative(radius * radius, ds)
    r_s = _safe_divide(r_r_s, radius)
    r_s = r_s.at[0].set(r_s[1])

    g_ss = r_s * r_s
    g_stheta = r_s * d_radius_dtheta
    g_sxi = r_s * d_radius_dxi
    g_thetatheta = d_radius_dtheta**2 + radius**2
    g_thetaxi = d_radius_dtheta * d_radius_dxi
    g_xixi = d_radius_dxi**2 + float(grid.dz_dxi) ** 2
    sqrt_g = r_r_s * float(grid.dz_dxi)

    theta = jnp.asarray(grid.theta)[None, :, None]
    z = jnp.asarray(grid.z)[None, None, :]
    xyz = jnp.stack(
        [
            radius * jnp.cos(theta),
            radius * jnp.sin(theta),
            jnp.broadcast_to(z, radius.shape),
        ],
        axis=-1,
    )
    volume = jnp.einsum(
        "i,j,k,ijk->",
        jnp.asarray(grid.radial_weights),
        jnp.asarray(grid.theta_basis.weights),
        jnp.asarray(grid.axial_basis.weights),
        sqrt_g,
    )
    interior = sqrt_g[1:]
    sign_changed = (jnp.min(interior) <= 0.0) | (jnp.max(interior) <= 0.0)
    return MirrorGeometry(
        xyz=xyz,
        radius=radius,
        d_radius_ds_regular=r_r_s,
        d_radius_dtheta=d_radius_dtheta,
        d_radius_dxi=d_radius_dxi,
        g_ss=g_ss,
        g_stheta=g_stheta,
        g_sxi=g_sxi,
        g_thetatheta=g_thetatheta,
        g_thetaxi=g_thetaxi,
        g_xixi=g_xixi,
        sqrt_g=sqrt_g,
        volume=volume,
        jacobian_sign_changed=sign_changed,
    )


def _radial_profile(values: Array, ns: int, dtype: Any) -> Array:
    values = jnp.asarray(values, dtype=dtype)
    if values.ndim == 0:
        values = jnp.broadcast_to(values, (ns,))
    if values.shape != (ns,):
        raise ValueError(f"radial profile shape {values.shape} must be scalar or ({ns},)")
    return values[:, None, None]


def contravariant_field(
    state: "MirrorState",
    geometry: MirrorGeometry,
    grid: "MirrorGrid",
    *,
    axial_flux_derivative: Array,
    current_derivative: Array = 0.0,
) -> ContravariantField:
    """Evaluate the divergence-free mirror field representation.

    ``axial_flux_derivative`` is ``Psi'(s)`` and ``current_derivative`` is
    ``I'(s)``.  Both may be scalar or one value per radial surface.
    """

    state.validate_shape(grid)
    lam = jnp.asarray(state.lambda_stream)
    d_lambda_dtheta = grid.theta_basis.differentiate(lam, axis=1)
    d_lambda_dxi = grid.axial_basis.differentiate(lam, axis=2)
    psi_prime = _radial_profile(axial_flux_derivative, grid.ns, lam.dtype)
    current_prime = _radial_profile(current_derivative, grid.ns, lam.dtype)
    jac_b_theta = current_prime - d_lambda_dxi
    jac_b_xi = psi_prime + d_lambda_dtheta
    return ContravariantField(
        b_sup_s=jnp.zeros_like(lam),
        b_sup_theta=_safe_divide(jac_b_theta, geometry.sqrt_g),
        b_sup_xi=_safe_divide(jac_b_xi, geometry.sqrt_g),
        jac_b_theta=jac_b_theta,
        jac_b_xi=jac_b_xi,
    )


def divergence_b(field: ContravariantField, geometry: MirrorGeometry, grid: "MirrorGrid") -> Array:
    """Return ``div(B)`` from contravariant flux densities."""

    theta_term = grid.theta_basis.differentiate(field.jac_b_theta, axis=1)
    xi_term = grid.axial_basis.differentiate(field.jac_b_xi, axis=2)
    return _safe_divide(theta_term + xi_term, geometry.sqrt_g)


def normalized_divergence_rms(
    field: ContravariantField,
    geometry: MirrorGeometry,
    grid: "MirrorGrid",
) -> Array:
    """Return the volume-weighted RMS of ``div(B)`` normalized by ``B/L``.

    The magnetic axis and open end cuts are constrained coordinate boundaries,
    so this diagnostic norms the active volume. Its normalization permits
    comparisons across field strengths and mirror lengths.
    """

    divergence = divergence_b(field, geometry, grid)[1:, :, 1:-1]
    b_squared = magnetic_field_squared(field, geometry)[1:, :, 1:-1]
    weights = (
        jnp.asarray(grid.radial_weights[1:])[:, None, None]
        * jnp.asarray(grid.theta_basis.weights)[None, :, None]
        * jnp.asarray(grid.axial_basis.weights)[None, None, 1:-1]
        * geometry.sqrt_g[1:, :, 1:-1]
    )
    weight_sum = jnp.sum(weights)
    divergence_rms = jnp.sqrt(jnp.sum(weights * divergence**2) / weight_sum)
    length = float(grid.z[-1] - grid.z[0])
    field_gradient_rms = jnp.sqrt(
        jnp.sum(weights * b_squared / length**2) / weight_sum
    )
    return divergence_rms / jnp.maximum(
        field_gradient_rms, jnp.finfo(divergence_rms.dtype).tiny
    )


def magnetic_field_squared(field: ContravariantField, geometry: MirrorGeometry) -> Array:
    """Contract contravariant components with the covariant metric."""

    bs, bt, bx = field.b_sup_s, field.b_sup_theta, field.b_sup_xi
    return (
        geometry.g_ss * bs**2
        + geometry.g_thetatheta * bt**2
        + geometry.g_xixi * bx**2
        + 2.0 * geometry.g_stheta * bs * bt
        + 2.0 * geometry.g_sxi * bs * bx
        + 2.0 * geometry.g_thetaxi * bt * bx
    )


def magnetic_field_xyz(field: ContravariantField, geometry: MirrorGeometry) -> Array:
    """Convert contravariant magnetic components to Cartesian components."""

    radius = geometry.radius
    radius_s = _safe_divide(geometry.d_radius_ds_regular, radius)
    radius_s = radius_s.at[0].set(radius_s[1])
    theta = jnp.arctan2(geometry.xyz[..., 1], geometry.xyz[..., 0])
    cosine, sine = jnp.cos(theta), jnp.sin(theta)
    zeros = jnp.zeros_like(radius)
    e_s = jnp.stack([radius_s * cosine, radius_s * sine, zeros], axis=-1)
    e_theta = jnp.stack(
        [
            geometry.d_radius_dtheta * cosine - radius * sine,
            geometry.d_radius_dtheta * sine + radius * cosine,
            zeros,
        ],
        axis=-1,
    )
    e_xi = jnp.stack(
        [
            geometry.d_radius_dxi * cosine,
            geometry.d_radius_dxi * sine,
            jnp.sqrt(
                jnp.maximum(
                    geometry.g_xixi - geometry.d_radius_dxi**2,
                    0.0,
                )
            ),
        ],
        axis=-1,
    )
    return (
        field.b_sup_s[..., None] * e_s
        + field.b_sup_theta[..., None] * e_theta
        + field.b_sup_xi[..., None] * e_xi
    )


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .basis import MirrorGrid
    from .model import MirrorState
