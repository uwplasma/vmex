"""Pure-JAX Fourier coils and Biot-Savart external-field sampling.

The Fourier coefficient convention intentionally matches ESSOS:

``dofs[..., 0]``
    constant term.
``dofs[..., 2*k-1]``
    coefficient multiplying ``sin(2*pi*k*t)``.
``dofs[..., 2*k]``
    coefficient multiplying ``cos(2*pi*k*t)``.

The Biot-Savart normalization also matches ESSOS phase-1 behavior:
``1e-7 * current * mean(gamma_dash x (x - gamma) / |x - gamma|^3)``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import pi as _PI
from typing import Any

import numpy as _np

from vmec_jax._compat import jax, jnp, tree_util


_TWO_PI = 2.0 * _PI


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CoilFieldParams:
    """Differentiable direct-coil external-field parameters.

    Parameters
    ----------
    base_curve_dofs:
        Array with shape ``(n_base_coils, 3, 2 * order + 1)``.  The second axis
        is Cartesian ``x, y, z`` and the final axis follows the ESSOS Fourier
        convention documented in this module.
    base_currents:
        Array with shape ``(n_base_coils,)``.  Currents are multiplied by
        ``current_scale`` before Biot-Savart evaluation.
    n_segments:
        Number of uniform curve quadrature points per coil.
    nfp:
        Number of field periods for rotational symmetry expansion.
    stellsym:
        Whether to add stellarator-symmetry reflected coils.  Reflected coils
        carry the opposite current, matching ESSOS.
    current_scale:
        Static scalar multiplying ``base_currents``.
    regularization_epsilon:
        Optional distance smoothing in the Biot-Savart denominator.
    chunk_size:
        Optional point chunk size for memory-limited field evaluation.  If set,
        points are evaluated with ``jax.lax.map(..., batch_size=chunk_size)``.
    """

    base_curve_dofs: Any
    base_currents: Any
    n_segments: int
    nfp: int = 1
    stellsym: bool = False
    current_scale: float = 1.0
    regularization_epsilon: float = 0.0
    chunk_size: int | None = None

    def tree_flatten(self):
        children = (self.base_curve_dofs, self.base_currents)
        aux = (
            int(self.n_segments),
            int(self.nfp),
            bool(self.stellsym),
            float(self.current_scale),
            float(self.regularization_epsilon),
            None if self.chunk_size is None else int(self.chunk_size),
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        n_segments, nfp, stellsym, current_scale, regularization_epsilon, chunk_size = aux
        base_curve_dofs, base_currents = children
        return cls(
            base_curve_dofs=base_curve_dofs,
            base_currents=base_currents,
            n_segments=n_segments,
            nfp=nfp,
            stellsym=stellsym,
            current_scale=current_scale,
            regularization_epsilon=regularization_epsilon,
            chunk_size=chunk_size,
        )

    def with_arrays(self, *, base_curve_dofs: Any | None = None, base_currents: Any | None = None) -> "CoilFieldParams":
        """Return a copy with updated differentiable leaves."""

        return replace(
            self,
            base_curve_dofs=self.base_curve_dofs if base_curve_dofs is None else base_curve_dofs,
            base_currents=self.base_currents if base_currents is None else base_currents,
        )


def ellipse_coil_fourier_dofs(
    *,
    center: Any,
    normal: Any,
    major_radius: float,
    minor_radius: float | None = None,
    major_axis: Any | None = None,
) -> Any:
    """Return first-order Fourier dofs for one planar elliptical coil.

    The curve convention is
    ``gamma(t) = center + major_radius * major_axis * cos(2*pi*t)
    + minor_radius * minor_axis * sin(2*pi*t)``.  The supplied ``normal`` fixes
    the oriented coil plane through ``major_axis x minor_axis = normal``.  If no
    ``major_axis`` is supplied, a stable axis perpendicular to ``normal`` is
    chosen automatically.
    """

    center_np = _np.asarray(center, dtype=float)
    normal_np = _np.asarray(normal, dtype=float)
    if center_np.shape != (3,) or normal_np.shape != (3,):
        raise ValueError("center and normal must be length-3 vectors")
    normal_norm = float(_np.linalg.norm(normal_np))
    if normal_norm <= 0.0:
        raise ValueError("normal must be nonzero")
    normal_unit = normal_np / normal_norm

    if major_axis is None:
        trial = _np.asarray([0.0, 0.0, 1.0], dtype=float)
        if abs(float(_np.dot(trial, normal_unit))) > 0.85:
            trial = _np.asarray([1.0, 0.0, 0.0], dtype=float)
    else:
        trial = _np.asarray(major_axis, dtype=float)
        if trial.shape != (3,):
            raise ValueError("major_axis must be a length-3 vector")
    major_unit = trial - float(_np.dot(trial, normal_unit)) * normal_unit
    major_norm = float(_np.linalg.norm(major_unit))
    if major_norm <= 0.0:
        raise ValueError("major_axis must not be parallel to normal")
    major_unit = major_unit / major_norm
    minor_unit = _np.cross(normal_unit, major_unit)

    minor_radius_eff = float(major_radius if minor_radius is None else minor_radius)
    if float(major_radius) <= 0.0 or minor_radius_eff <= 0.0:
        raise ValueError("major_radius and minor_radius must be positive")

    dofs = _np.zeros((3, 3), dtype=float)
    dofs[:, 0] = center_np
    dofs[:, 1] = minor_radius_eff * minor_unit
    dofs[:, 2] = float(major_radius) * major_unit
    return jnp.asarray(dofs)


def ellipse_coil_field_params(
    *,
    centers: Any,
    normals: Any,
    currents: Any,
    major_radius: float | Any,
    minor_radius: float | Any | None = None,
    major_axes: Any | None = None,
    n_segments: int = 96,
    nfp: int = 1,
    stellsym: bool = False,
    current_scale: float = 1.0,
    regularization_epsilon: float = 0.0,
    chunk_size: int | None = None,
) -> CoilFieldParams:
    """Build direct-coil parameters from planar elliptical coils.

    This is a small SIMSOPT-style convenience layer over the Fourier-coil
    backend: each coil is specified by center, oriented normal, current, and
    semi-axis radii.  Scalar radii are broadcast to every coil.
    """

    centers_np = _np.asarray(centers, dtype=float)
    normals_np = _np.asarray(normals, dtype=float)
    if centers_np.ndim != 2 or centers_np.shape[1] != 3:
        raise ValueError("centers must have shape (n_coils, 3)")
    if normals_np.shape != centers_np.shape:
        raise ValueError("normals must have the same shape as centers")
    n_coils = int(centers_np.shape[0])
    currents_np = _np.asarray(currents, dtype=float)
    if currents_np.ndim == 0:
        currents_np = _np.full((n_coils,), float(currents_np), dtype=float)
    if currents_np.shape != (n_coils,):
        raise ValueError("currents must be scalar or have shape (n_coils,)")

    major_np = _np.asarray(major_radius, dtype=float)
    if major_np.ndim == 0:
        major_np = _np.full((n_coils,), float(major_np), dtype=float)
    if major_np.shape != (n_coils,):
        raise ValueError("major_radius must be scalar or have shape (n_coils,)")
    if minor_radius is None:
        minor_np = major_np
    else:
        minor_np = _np.asarray(minor_radius, dtype=float)
        if minor_np.ndim == 0:
            minor_np = _np.full((n_coils,), float(minor_np), dtype=float)
        if minor_np.shape != (n_coils,):
            raise ValueError("minor_radius must be scalar or have shape (n_coils,)")

    if major_axes is None:
        axes_iter = [None] * n_coils
    else:
        axes_np = _np.asarray(major_axes, dtype=float)
        if axes_np.shape == (3,):
            axes_np = _np.broadcast_to(axes_np, centers_np.shape)
        if axes_np.shape != centers_np.shape:
            raise ValueError("major_axes must be a length-3 vector or have shape (n_coils, 3)")
        axes_iter = list(axes_np)

    dofs = [
        ellipse_coil_fourier_dofs(
            center=centers_np[idx],
            normal=normals_np[idx],
            major_radius=float(major_np[idx]),
            minor_radius=float(minor_np[idx]),
            major_axis=axes_iter[idx],
        )
        for idx in range(n_coils)
    ]
    return CoilFieldParams(
        base_curve_dofs=jnp.asarray(dofs),
        base_currents=jnp.asarray(currents_np),
        n_segments=int(n_segments),
        nfp=int(nfp),
        stellsym=bool(stellsym),
        current_scale=float(current_scale),
        regularization_epsilon=float(regularization_epsilon),
        chunk_size=None if chunk_size is None else int(chunk_size),
    )


def _fourier_basis(n_segments: int, order: int) -> tuple[Any, Any, Any]:
    t = jnp.linspace(0.0, 1.0, int(n_segments), endpoint=False)
    k = jnp.arange(1, int(order) + 1, dtype=t.dtype)
    phase = _TWO_PI * t[:, None] * k[None, :]
    return t, jnp.sin(phase), jnp.cos(phase)


def fourier_curves_to_gamma(base_curve_dofs: Any, n_segments: int) -> Any:
    """Evaluate Fourier curve centerlines.

    Returns an array with shape ``(n_base_coils, n_segments, 3)``.
    """

    dofs = jnp.asarray(base_curve_dofs)
    if dofs.ndim != 3 or dofs.shape[1] != 3 or dofs.shape[2] % 2 != 1:
        raise ValueError("base_curve_dofs must have shape (n_base_coils, 3, 2 * order + 1)")
    order = (int(dofs.shape[2]) - 1) // 2
    _, sin_basis, cos_basis = _fourier_basis(n_segments, order)
    gamma = dofs[:, :, 0][:, None, :]
    if order == 0:
        return jnp.broadcast_to(gamma, (dofs.shape[0], int(n_segments), 3))
    sin_coeff = dofs[:, :, 1::2]
    cos_coeff = dofs[:, :, 2::2]
    gamma = gamma + jnp.einsum("nck,sk->nsc", sin_coeff, sin_basis)
    gamma = gamma + jnp.einsum("nck,sk->nsc", cos_coeff, cos_basis)
    return gamma


def compute_gamma_dash(base_curve_dofs: Any, n_segments: int) -> Any:
    """Evaluate ``d gamma / d t`` for normalized curve parameter ``t``."""

    dofs = jnp.asarray(base_curve_dofs)
    if dofs.ndim != 3 or dofs.shape[1] != 3 or dofs.shape[2] % 2 != 1:
        raise ValueError("base_curve_dofs must have shape (n_base_coils, 3, 2 * order + 1)")
    order = (int(dofs.shape[2]) - 1) // 2
    if order == 0:
        return jnp.zeros((dofs.shape[0], int(n_segments), 3), dtype=dofs.dtype)
    t, sin_basis, cos_basis = _fourier_basis(n_segments, order)
    k = jnp.arange(1, order + 1, dtype=t.dtype)
    factor = _TWO_PI * k
    sin_coeff = dofs[:, :, 1::2]
    cos_coeff = dofs[:, :, 2::2]
    gamma_dash = jnp.einsum("nck,sk,k->nsc", sin_coeff, cos_basis, factor)
    gamma_dash = gamma_dash - jnp.einsum("nck,sk,k->nsc", cos_coeff, sin_basis, factor)
    return gamma_dash


def compute_gamma_dashdash(base_curve_dofs: Any, n_segments: int) -> Any:
    """Evaluate ``d^2 gamma / d t^2`` for normalized curve parameter ``t``."""

    dofs = jnp.asarray(base_curve_dofs)
    if dofs.ndim != 3 or dofs.shape[1] != 3 or dofs.shape[2] % 2 != 1:
        raise ValueError("base_curve_dofs must have shape (n_base_coils, 3, 2 * order + 1)")
    order = (int(dofs.shape[2]) - 1) // 2
    if order == 0:
        return jnp.zeros((dofs.shape[0], int(n_segments), 3), dtype=dofs.dtype)
    t, sin_basis, cos_basis = _fourier_basis(n_segments, order)
    k = jnp.arange(1, order + 1, dtype=t.dtype)
    factor = (_TWO_PI * k) ** 2
    sin_coeff = dofs[:, :, 1::2]
    cos_coeff = dofs[:, :, 2::2]
    gamma_dashdash = -jnp.einsum("nck,sk,k->nsc", sin_coeff, sin_basis, factor)
    gamma_dashdash = gamma_dashdash - jnp.einsum("nck,sk,k->nsc", cos_coeff, cos_basis, factor)
    return gamma_dashdash


def _rotation_reflection_matrix(phi: Any, flip: bool) -> Any:
    c = jnp.cos(phi)
    s = jnp.sin(phi)
    rot = jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]).T
    if flip:
        rot = rot @ jnp.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    return rot


def _apply_symmetry_to_xyz_array(base: Any, *, nfp: int, stellsym: bool) -> Any:
    values = jnp.asarray(base)
    flip_list = (False, True) if bool(stellsym) else (False,)
    expanded = []
    for k in range(int(nfp)):
        angle = _TWO_PI * k / int(nfp)
        for flip in flip_list:
            mat = _rotation_reflection_matrix(angle, flip)
            expanded.append(jnp.einsum("...c,cd->...d", values, mat))
    return jnp.concatenate(expanded, axis=0)


def apply_stellarator_symmetry_to_curves(base_curve_dofs: Any, nfp: int, stellsym: bool) -> Any:
    """Expand base Fourier curve coefficients using ESSOS symmetry ordering."""

    return _apply_symmetry_to_xyz_array(base_curve_dofs, nfp=nfp, stellsym=stellsym)


def apply_stellarator_symmetry_to_currents(base_currents: Any, nfp: int, stellsym: bool) -> Any:
    """Expand base currents using ESSOS symmetry ordering."""

    currents = jnp.asarray(base_currents)
    expanded = []
    for _k in range(int(nfp)):
        expanded.append(currents)
        if bool(stellsym):
            expanded.append(-currents)
    return jnp.concatenate(expanded, axis=0)


def expanded_coil_geometry(params: CoilFieldParams) -> tuple[Any, Any, Any, Any]:
    """Return full symmetry-expanded ``gamma``, derivatives, and currents."""

    base_dofs = jnp.asarray(params.base_curve_dofs)
    base_gamma = fourier_curves_to_gamma(base_dofs, params.n_segments)
    base_gamma_dash = compute_gamma_dash(base_dofs, params.n_segments)
    base_gamma_dashdash = compute_gamma_dashdash(base_dofs, params.n_segments)
    gamma = _apply_symmetry_to_xyz_array(base_gamma, nfp=params.nfp, stellsym=params.stellsym)
    gamma_dash = _apply_symmetry_to_xyz_array(base_gamma_dash, nfp=params.nfp, stellsym=params.stellsym)
    gamma_dashdash = _apply_symmetry_to_xyz_array(base_gamma_dashdash, nfp=params.nfp, stellsym=params.stellsym)
    currents = params.current_scale * apply_stellarator_symmetry_to_currents(
        params.base_currents,
        nfp=params.nfp,
        stellsym=params.stellsym,
    )
    return gamma, gamma_dash, gamma_dashdash, currents


def build_coil_field_geometry(params: CoilFieldParams) -> tuple[Any, Any, Any]:
    """Build symmetry-expanded direct-coil geometry for Biot-Savart sampling.

    The returned tuple is ``(gamma, gamma_dash, currents)``.  It intentionally
    omits ``gamma_dashdash`` so field-only callers and benchmarks can prebuild
    just the geometry needed by Biot-Savart without changing the differentiable
    ``CoilFieldParams -> field`` path.
    """

    base_dofs = jnp.asarray(params.base_curve_dofs)
    base_gamma = fourier_curves_to_gamma(base_dofs, params.n_segments)
    base_gamma_dash = compute_gamma_dash(base_dofs, params.n_segments)
    gamma = _apply_symmetry_to_xyz_array(base_gamma, nfp=params.nfp, stellsym=params.stellsym)
    gamma_dash = _apply_symmetry_to_xyz_array(base_gamma_dash, nfp=params.nfp, stellsym=params.stellsym)
    currents = params.current_scale * apply_stellarator_symmetry_to_currents(
        params.base_currents,
        nfp=params.nfp,
        stellsym=params.stellsym,
    )
    return gamma, gamma_dash, currents


def _biot_savart_xyz_vectorized(
    points_xyz: Any,
    gamma: Any,
    gamma_dash: Any,
    currents: Any,
    regularization_epsilon: float = 0.0,
) -> Any:
    points = jnp.asarray(points_xyz)
    original_shape = points.shape[:-1]
    flat = jnp.reshape(points, (-1, 3))
    rx = flat[None, None, :, 0] - gamma[:, :, None, 0]
    ry = flat[None, None, :, 1] - gamma[:, :, None, 1]
    rz = flat[None, None, :, 2] - gamma[:, :, None, 2]
    eps = jnp.asarray(regularization_epsilon, dtype=points.dtype)
    denom2 = rx * rx + ry * ry + rz * rz + eps * eps
    if jax is None:  # pragma: no cover - dependency fallback.
        inv_r = 1.0 / jnp.sqrt(denom2)
    else:
        inv_r = jax.lax.rsqrt(denom2)
    inv_r3 = inv_r * inv_r * inv_r

    gx = gamma_dash[:, :, None, 0]
    gy = gamma_dash[:, :, None, 1]
    gz = gamma_dash[:, :, None, 2]
    weights = jnp.asarray(currents, dtype=points.dtype)[:, None, None] * inv_r3
    field_x = jnp.mean(jnp.sum(weights * (gy * rz - gz * ry), axis=0), axis=0)
    field_y = jnp.mean(jnp.sum(weights * (gz * rx - gx * rz), axis=0), axis=0)
    field_z = jnp.mean(jnp.sum(weights * (gx * ry - gy * rx), axis=0), axis=0)
    field = 1.0e-7 * jnp.stack((field_x, field_y, field_z), axis=-1)
    return jnp.reshape(field, original_shape + (3,))


def biot_savart_xyz(
    points_xyz: Any,
    gamma: Any,
    gamma_dash: Any,
    currents: Any,
    regularization_epsilon: float = 0.0,
    chunk_size: int | None = None,
) -> Any:
    """Evaluate coil Biot-Savart field at Cartesian points.

    ``chunk_size`` limits peak memory by mapping over evaluation points in
    batches.  The unchunked path is faster for small grids and is used by
    default.
    """

    points = jnp.asarray(points_xyz)
    original_shape = points.shape[:-1]
    flat = jnp.reshape(points, (-1, 3))
    if chunk_size is None:
        return _biot_savart_xyz_vectorized(
            points,
            gamma,
            gamma_dash,
            currents,
            regularization_epsilon=regularization_epsilon,
        )

    def one_point(point):
        value = _biot_savart_xyz_vectorized(
            point[None, :],
            gamma,
            gamma_dash,
            currents,
            regularization_epsilon=regularization_epsilon,
        )
        return value[0]

    if jax is None:  # pragma: no cover - dependency fallback.
        values = jnp.asarray([one_point(point) for point in flat])
    else:
        values = jax.lax.map(one_point, flat, batch_size=int(chunk_size))
    return jnp.reshape(values, original_shape + (3,))


def _cylindrical_points_to_xyz(R: Any, Z: Any, phi: Any) -> Any:
    Rb, Zb, phib = jnp.broadcast_arrays(jnp.asarray(R), jnp.asarray(Z), jnp.asarray(phi))
    return jnp.stack((Rb * jnp.cos(phib), Rb * jnp.sin(phib), Zb), axis=-1)


def _xyz_field_to_cylindrical(B_xyz: Any, phi: Any) -> tuple[Any, Any, Any]:
    phib = jnp.broadcast_to(jnp.asarray(phi), B_xyz.shape[:-1])
    bx = B_xyz[..., 0]
    by = B_xyz[..., 1]
    bz = B_xyz[..., 2]
    br = bx * jnp.cos(phib) + by * jnp.sin(phib)
    bphi = -bx * jnp.sin(phib) + by * jnp.cos(phib)
    return br, bphi, bz


def sample_coil_field_xyz_from_geometry(
    geometry: tuple[Any, Any, Any],
    points_xyz: Any,
    *,
    regularization_epsilon: float = 0.0,
    chunk_size: int | None = None,
) -> Any:
    """Sample Cartesian direct-coil field from prebuilt coil geometry."""

    gamma, gamma_dash, currents = geometry
    return biot_savart_xyz(
        points_xyz,
        gamma,
        gamma_dash,
        currents,
        regularization_epsilon=regularization_epsilon,
        chunk_size=chunk_size,
    )


def sample_coil_field_cylindrical_from_geometry(
    geometry: tuple[Any, Any, Any],
    R: Any,
    Z: Any,
    phi: Any,
    *,
    regularization_epsilon: float = 0.0,
    chunk_size: int | None = None,
) -> tuple[Any, Any, Any]:
    """Sample cylindrical direct-coil field from prebuilt coil geometry."""

    points = _cylindrical_points_to_xyz(R, Z, phi)
    field_xyz = sample_coil_field_xyz_from_geometry(
        geometry,
        points,
        regularization_epsilon=regularization_epsilon,
        chunk_size=chunk_size,
    )
    return _xyz_field_to_cylindrical(field_xyz, phi)


if jax is not None:

    @jax.jit
    def _sample_coil_field_cylindrical_from_geometry_jit(gamma, gamma_dash, currents, R, Z, phi, regularization_epsilon):
        points = _cylindrical_points_to_xyz(R, Z, phi)
        field_xyz = biot_savart_xyz(
            points,
            gamma,
            gamma_dash,
            currents,
            regularization_epsilon=regularization_epsilon,
            chunk_size=None,
        )
        return _xyz_field_to_cylindrical(field_xyz, phi)


def sample_coil_field_cylindrical_from_geometry_jit(
    geometry: tuple[Any, Any, Any],
    R: Any,
    Z: Any,
    phi: Any,
    *,
    regularization_epsilon: float = 0.0,
) -> tuple[Any, Any, Any]:
    """JIT-sample cylindrical field from prebuilt geometry.

    This helper is intended for host-forward free-boundary solves and
    benchmarks in which coil geometry is already cached and fixed.  Use the
    non-JIT ``sample_coil_field_cylindrical`` path for transformed functions
    that differentiate with respect to changing coil parameters.
    """

    if jax is None:  # pragma: no cover - dependency fallback.
        return sample_coil_field_cylindrical_from_geometry(
            geometry,
            R,
            Z,
            phi,
            regularization_epsilon=regularization_epsilon,
        )
    gamma, gamma_dash, currents = geometry
    return _sample_coil_field_cylindrical_from_geometry_jit(
        gamma,
        gamma_dash,
        currents,
        R,
        Z,
        phi,
        jnp.asarray(float(regularization_epsilon), dtype=jnp.asarray(gamma).dtype),
    )


def sample_coil_field_cylindrical(params: CoilFieldParams, R: Any, Z: Any, phi: Any) -> tuple[Any, Any, Any]:
    """Sample the direct-coil field at cylindrical coordinates."""

    geometry = build_coil_field_geometry(params)
    return sample_coil_field_cylindrical_from_geometry(
        geometry,
        R,
        Z,
        phi,
        regularization_epsilon=params.regularization_epsilon,
        chunk_size=params.chunk_size,
    )


def coil_lengths(params: CoilFieldParams) -> Any:
    """Return per-coil centerline lengths after symmetry expansion."""

    _gamma, gamma_dash, _gamma_dashdash, _currents = expanded_coil_geometry(params)
    return jnp.mean(jnp.linalg.norm(gamma_dash, axis=-1), axis=-1)


def coil_curvatures(params: CoilFieldParams) -> Any:
    """Return per-coil, per-segment curvature after symmetry expansion."""

    _gamma, gamma_dash, gamma_dashdash, _currents = expanded_coil_geometry(params)
    speed = jnp.linalg.norm(gamma_dash, axis=-1)
    numerator = jnp.linalg.norm(jnp.cross(gamma_dash, gamma_dashdash, axis=-1), axis=-1)
    return numerator / jnp.maximum(speed, 1.0e-30) ** 3


def coil_current_norm(params: CoilFieldParams) -> Any:
    """Return Euclidean norm of symmetry-expanded physical currents."""

    _gamma, _gamma_dash, _gamma_dashdash, currents = expanded_coil_geometry(params)
    return jnp.linalg.norm(currents)


def _soft_min(values: Any, alpha: float) -> Any:
    scaled = -float(alpha) * jnp.asarray(values)
    return -jax.nn.logsumexp(scaled) / float(alpha) if jax is not None else -jnp.log(jnp.sum(jnp.exp(scaled))) / float(alpha)


def coil_plasma_distance_soft(params: CoilFieldParams, boundary_xyz: Any, alpha: float = 25.0) -> Any:
    """Smooth approximation to the minimum coil-plasma distance."""

    gamma, _gamma_dash, _gamma_dashdash, _currents = expanded_coil_geometry(params)
    coil_points = jnp.reshape(gamma, (-1, 3))
    boundary_points = jnp.reshape(jnp.asarray(boundary_xyz), (-1, 3))
    diff = coil_points[:, None, :] - boundary_points[None, :, :]
    distances = jnp.linalg.norm(diff, axis=-1)
    return _soft_min(jnp.ravel(distances), alpha)


def coil_coil_distance_soft(params: CoilFieldParams, alpha: float = 25.0) -> Any:
    """Smooth approximation to the minimum distance between distinct coils."""

    gamma, _gamma_dash, _gamma_dashdash, _currents = expanded_coil_geometry(params)
    ncoils, nsegments, _ = gamma.shape
    points = jnp.reshape(gamma, (ncoils * nsegments, 3))
    coil_id = jnp.repeat(jnp.arange(ncoils), nsegments)
    diff = points[:, None, :] - points[None, :, :]
    distances = jnp.linalg.norm(diff, axis=-1)
    same_coil = coil_id[:, None] == coil_id[None, :]
    distances = jnp.where(same_coil, jnp.inf, distances)
    return _soft_min(jnp.ravel(distances), alpha)


def _smooth_hinge(x: Any, smoothness: float) -> Any:
    scaled = jnp.asarray(x) / float(smoothness)
    if jax is None:  # pragma: no cover - dependency fallback.
        return float(smoothness) * jnp.log1p(jnp.exp(scaled))
    return float(smoothness) * jax.nn.softplus(scaled)


def length_penalty(params: CoilFieldParams, maximum: float, smoothness: float = 1.0e-3) -> Any:
    """Smooth squared penalty for coil lengths above ``maximum``."""

    excess = coil_lengths(params) - float(maximum)
    return jnp.mean(_smooth_hinge(excess, smoothness) ** 2)


def curvature_penalty(params: CoilFieldParams, maximum: float, smoothness: float = 1.0e-3) -> Any:
    """Smooth squared penalty for coil curvature above ``maximum``."""

    excess = coil_curvatures(params) - float(maximum)
    return jnp.mean(_smooth_hinge(excess, smoothness) ** 2)
