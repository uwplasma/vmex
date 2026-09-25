"""Magnetic-field queries outside a VMEC plasma boundary.

The vacuum path evaluates the supplied coil or mgrid field directly.  When the
equilibrium carries plasma pressure or current, :mod:`virtual_casing_jax` adds
the field of currents inside the last closed flux surface.  The resulting
object follows the commonly used SIMSOPT magnetic-field interface while its
explicit-point methods remain JAX-transformable.
"""

from __future__ import annotations

import copy
import math
import warnings
from pathlib import Path
from typing import Any, Callable, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np

from .errors import VmecNumericalError
from .mgrid import MgridField, read_mgrid
from .profiles import MU0
from .virtual_casing import GRADED_NODES as _GRADED_NODES

Array = Any
PlasmaMode = Literal["auto", "include", "vacuum"]
AccuracyCheck = Literal["warn", "raise", "off"]
NearSurface = Literal["auto", "graded", "direct"]

__all__ = [
    "ExteriorFieldAccuracyError",
    "ExteriorFieldAccuracyWarning",
    "MagneticField",
    "VmecInteriorField",
    "VmecExtender",
]


#: Source sampling per field period when the caller does not choose one.  The
#: floor is the historical constant; the ceiling bounds the cost of the
#: geometry rule on a very high aspect ratio boundary.
_DEFAULT_SOURCE_NPHI = 32
_MAX_SOURCE_NPHI = 256

#: Resolution of the ``(rho, theta)`` sweep that seeds the unseeded interior
#: Newton inversion, and of the boundary polygon its outermost row doubles as.
#: A purely geometric first guess leaves the iteration outside the basin of
#: its root for a fifth to a quarter of the interior of a shaped boundary, at
#: any iteration count; see :func:`_invert_coordinates`.
_GUESS_NRHO = 16
_GUESS_NTHETA = 32

#: The boundary polygon is shrunk by this much towards the magnetic axis
#: before a point is called interior, so that its finite resolution can only
#: ever suppress a complaint about a stalled point, never invent one.
_INSIDE_MARGIN = 0.99


class ExteriorFieldAccuracyWarning(UserWarning):
    """Direct virtual-casing quadrature missed its requested ``digits``."""


class InteriorFieldAccuracyWarning(UserWarning):
    """A second or third derivative was taken on the fitted fallback path."""


class ExteriorFieldAccuracyError(RuntimeError):
    """Raised instead of the warning when ``accuracy_check="raise"``."""


def _check_points(points: Array, name: str = "points") -> Array:
    points = jnp.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3), got {points.shape}")
    return points


def _cyl_to_cart(points: Array) -> Array:
    r, phi, z = _check_points(points, "cylindrical points").T
    return jnp.stack((r * jnp.cos(phi), r * jnp.sin(phi), z), axis=-1)


def _cart_to_cyl(points: Array) -> Array:
    x, y, z = _check_points(points).T
    return jnp.stack(
        (jnp.hypot(x, y), jnp.mod(jnp.arctan2(y, x), 2.0 * jnp.pi), z),
        axis=-1,
    )


def _vectors_to_cyl(points_cyl: Array, vectors: Array) -> Array:
    phi = _check_points(points_cyl, "cylindrical points")[:, 1]
    bx, by, bz = _check_points(vectors, "vectors").T
    cphi, sphi = jnp.cos(phi), jnp.sin(phi)
    return jnp.stack((cphi * bx + sphi * by, -sphi * bx + cphi * by, bz), axis=-1)


def _field_cartesian(field: Any, points: Array) -> Array:
    """Evaluate a VMEX mgrid-like field or an ``xyz -> B`` callable."""
    points = _check_points(points)
    if hasattr(field, "b_cyl"):
        x, y, z = points.T
        r = jnp.hypot(x, y)
        phi = jnp.arctan2(y, x)
        br, bphi, bz = field.b_cyl(r, phi, z)
        cphi, sphi = jnp.cos(phi), jnp.sin(phi)
        return jnp.stack(
            (br * cphi - bphi * sphi, br * sphi + bphi * cphi, bz), axis=-1
        )
    if callable(field):
        value = jnp.asarray(field(points))
        if value.shape != points.shape:
            raise ValueError(
                f"external field returned shape {value.shape}, expected {points.shape}"
            )
        return value
    raise TypeError("external_field must be callable or provide b_cyl(r, phi, z)")


class MagneticField:
    """JAX magnetic field with explicit and stored-point evaluation.

    ``gradB`` has axes ``(point, B_i, x_j)``.  The SIMSOPT-compatible
    ``dB_by_dX`` swaps the last two axes to ``(point, x_j, B_i)``.

    Every method takes points as Cartesian ``xyz`` in metres with shape
    ``(n, 3)`` and returns ``B`` in tesla with the same shape; the
    cylindrical helpers convert to and from the ``(R, phi, Z)`` layout with
    ``phi`` in radians.  Points may be passed explicitly or stored once with
    :meth:`set_points`.

    The optional parameter arguments make the field differentiable with
    respect to degrees of freedom that are not spatial coordinates — coil
    currents, coil shapes, or equilibrium parameters — which is what the
    ``*_vjp`` methods pull back.  There are two mutually exclusive ways to
    supply that dependence, and either one requires ``parameters``: the
    direct path (``parameterized_B_fn``) and the factored path
    (``parameter_data_fn`` with ``B_from_data``).  Providing both, or one
    half of the factored pair, raises :exc:`ValueError`.

    Parameters
    ----------
    B_fn:
        Cartesian field callable, ``xyz (n, 3) [m] -> B (n, 3) [T]``.  This
        is the only required argument; a returned shape other than the
        input shape raises :exc:`ValueError`.
    gradB_fn:
        Optional first spatial derivative, ``xyz (n, 3) -> (n, 3, 3)``
        holding ``dB_i/dx_j`` in T/m with axes ``(point, B_i, x_j)``.  When
        ``None``, :meth:`gradB` differentiates ``B_fn`` with a per-point
        ``jax.jacfwd``, which is exact but costs one extra AD pass.
    gradgradB_fn:
        Optional second derivative, ``(n, 3, 3, 3)`` holding
        ``d2B_i/dx_j dx_k`` in T/m^2, axes ``(point, B_i, x_j, x_k)``.
        ``None`` nests two ``jax.jacfwd`` passes over ``B_fn``.
    gradgradgradB_fn:
        Optional third derivative, ``(n, 3, 3, 3, 3)`` holding
        ``d3B_i/dx_j dx_k dx_l`` in T/m^3.  ``None`` nests three
        ``jax.jacfwd`` passes over ``B_fn``.
    parameters:
        Optimizable field degrees of freedom, flattened to one dimension.
        Required by, and only meaningful with, one of the two parameterized
        paths; without it the ``*_vjp`` methods raise :exc:`RuntimeError`.
        The units are whatever the parameterization uses (A for coil
        currents, m for coil Fourier coefficients).
    parameterized_B_fn:
        Direct path: ``(parameters, xyz) -> B``, differentiated end to end
        on every parameter VJP.
    parameter_data_fn:
        Factored path, first half: ``parameters -> data``, where ``data``
        is any JAX pytree of intermediate quantities (for VMEX, the
        equilibrium spectra or the boundary surface arrays).  Its VJP is
        built once per point set and reused for every derivative order, so
        an expensive parameters-to-data map is not repeated.
    B_from_data:
        Factored path, second half: ``(data, xyz) -> B``.  Must be supplied
        together with ``parameter_data_fn``.
    dof_names:
        One name per entry of ``parameters``, in the same order, for
        labelling gradient output.  ``None`` or ``()`` leaves the names
        unset.  A mismatched length raises :exc:`ValueError`, but only when
        ``parameters`` was also given — there is nothing to compare
        against otherwise.
    """

    def __init__(
        self,
        B_fn: Callable[[Array], Array],
        gradB_fn: Callable[[Array], Array] | None = None,
        gradgradB_fn: Callable[[Array], Array] | None = None,
        gradgradgradB_fn: Callable[[Array], Array] | None = None,
        *,
        parameters: Array | None = None,
        parameterized_B_fn: Callable[[Array, Array], Array] | None = None,
        parameter_data_fn: Callable[[Array], Any] | None = None,
        B_from_data: Callable[[Any, Array], Array] | None = None,
        dof_names: tuple[str, ...] | None = None,
    ) -> None:
        self._B_fn = B_fn
        self._gradB_fn = gradB_fn
        self._gradgradB_fn = gradgradB_fn
        self._gradgradgradB_fn = gradgradgradB_fn
        self._parameters = (None if parameters is None else
                            jnp.ravel(jnp.asarray(parameters)))
        self._parameterized_B_fn = parameterized_B_fn
        self._parameter_data_fn = parameter_data_fn
        self._B_from_data = B_from_data
        self._parameter_data_vjp = None
        self._data_pullbacks: dict[Any, Callable[..., Any]] = {}
        self._spatial_fns: dict[int, Callable[[Array], Array]] = {}
        self.dof_names = tuple(dof_names or ())
        direct = parameterized_B_fn is not None
        factored = parameter_data_fn is not None or B_from_data is not None
        if factored and (parameter_data_fn is None or B_from_data is None):
            raise ValueError("parameter_data_fn and B_from_data must be provided together")
        if direct and factored:
            raise ValueError("provide parameterized_B_fn or the factored data path, not both")
        if (parameters is not None) != (direct or factored):
            raise ValueError(
                "parameters and a parameterized field path must be provided together")
        if self._parameters is not None and self.dof_names and (
                len(self.dof_names) != self._parameters.size):
            raise ValueError("dof_names must match the number of field parameters")
        self._points_cart: Array | None = None
        self._points_cyl: Array | None = None

    def set_points(self, points: Array) -> "MagneticField":
        """Store Cartesian points with shape ``(n, 3)``."""
        self._points_cart = _check_points(points)
        self._points_cyl = None
        # Spatial-derivative pullbacks close over the evaluation points.
        self._data_pullbacks.clear()
        return self

    def set_points_xyz(self, points: Array) -> "MagneticField":
        """Store Cartesian ``(x, y, z)`` points with shape ``(n, 3)``."""
        return self.set_points(points)

    set_points_cart = set_points_xyz

    def set_points_cyl(self, points: Array) -> "MagneticField":
        """Store cylindrical points ``(R, phi, Z)`` with shape ``(n, 3)``."""
        cylindrical = _check_points(points, "cylindrical points")
        self.set_points(_cyl_to_cart(cylindrical))
        self._points_cyl = cylindrical
        return self

    def get_points_cart(self) -> Array:
        """Return stored Cartesian points."""
        return self._require_points()

    def get_points_cyl(self) -> Array:
        """Return stored cylindrical points ``(R, phi, Z)``."""
        points = self._require_points()
        if self._points_cyl is None:
            self._points_cyl = _cart_to_cyl(points)
        return self._points_cyl

    def _require_points(self) -> Array:
        if self._points_cart is None:
            raise RuntimeError("call set_points() or pass points explicitly")
        return self._points_cart

    def B(self, points: Array | None = None) -> Array:
        """Return Cartesian ``B`` at explicit or stored points."""
        xyz = self._require_points() if points is None else _check_points(points)
        value = jnp.asarray(self._B_fn(xyz))
        if value.shape != xyz.shape:
            raise ValueError(f"field returned shape {value.shape}, expected {xyz.shape}")
        return value

    def B_contravariant(self, point: Array) -> Array:
        """Return Cartesian ``B`` at one point for ESSOS field-line tracing."""
        point = jnp.asarray(point)
        if point.shape != (3,):
            raise ValueError(f"point must have shape (3,), got {point.shape}")
        return self.B(point[None, :])[0]

    @staticmethod
    def to_xyz(point: Array) -> Array:
        """Return a Cartesian tracing point unchanged."""
        point = jnp.asarray(point)
        if point.shape != (3,):
            raise ValueError(f"point must have shape (3,), got {point.shape}")
        return point

    def B_cyl(self, points: Array | None = None) -> Array:
        """Return ``(B_R, B_phi, B_Z)`` at cylindrical points."""
        rphiz = self.get_points_cyl() if points is None else _check_points(
            points, "cylindrical points"
        )
        return _vectors_to_cyl(rphiz, self.B(_cyl_to_cart(rphiz)))

    def absB(self, points: Array | None = None) -> Array:
        """Return ``|B|`` with shape ``(n,)``."""
        return jnp.linalg.norm(self.B(points), axis=-1)

    def AbsB(self, points: Array | None = None) -> Array:
        """Return SIMSOPT-compatible ``|B|`` with shape ``(n, 1)``."""
        return self.absB(points)[:, None]

    def _spatial_derivative(self, order: int, xyz: Array) -> Array:
        """Return the ``order``-th Cartesian derivative of ``B_fn``, compiled once.

        Nesting ``jacfwd`` outside ``jit`` dispatches every primitive of the
        expanded graph on its own, which for the third derivative costs orders
        of magnitude more than the one compiled kernel.  The compiled callable
        is kept per order, so repeated queries pay compilation once.  ``B_fn``
        itself is never wrapped: with explicit derivative callables it may be a
        NumPy function that cannot be traced.
        """
        function = self._spatial_fns.get(order)
        if function is None:
            B_fn = self._B_fn  # bind the callable, not self: no reference cycle
            point_field = lambda point: jnp.asarray(B_fn(point[None, :]))[0]  # noqa: E731
            for _ in range(order):
                point_field = jax.jacfwd(point_field)
            function = self._spatial_fns[order] = jax.jit(jax.vmap(point_field))
        return function(xyz)

    def gradB(self, points: Array | None = None) -> Array:
        """Return ``dB_i/dx_j`` with shape ``(n, 3, 3)``."""
        xyz = self._require_points() if points is None else _check_points(points)
        if self._gradB_fn is not None:
            value = jnp.asarray(self._gradB_fn(xyz))
        else:
            value = self._spatial_derivative(1, xyz)
        expected = xyz.shape + (3,)
        if value.shape != expected:
            raise ValueError(f"field gradient returned shape {value.shape}, expected {expected}")
        return value

    def gradgradB(self, points: Array | None = None) -> Array:
        """Return ``d²B_i/dx_j dx_k`` with shape ``(n, 3, 3, 3)``."""
        xyz = self._require_points() if points is None else _check_points(points)
        if self._gradgradB_fn is not None:
            value = jnp.asarray(self._gradgradB_fn(xyz))
        else:
            value = self._spatial_derivative(2, xyz)
        expected = xyz.shape + (3, 3)
        if value.shape != expected:
            raise ValueError(
                f"second field derivative returned shape {value.shape}, expected {expected}")
        return value

    def gradgradgradB(self, points: Array | None = None) -> Array:
        """Return ``d³B_i/dx_j dx_k dx_l`` with shape ``(n, 3, 3, 3, 3)``."""
        xyz = self._require_points() if points is None else _check_points(points)
        if self._gradgradgradB_fn is not None:
            value = jnp.asarray(self._gradgradgradB_fn(xyz))
        else:
            value = self._spatial_derivative(3, xyz)
        expected = xyz.shape + (3, 3, 3)
        if value.shape != expected:
            raise ValueError(
                f"third field derivative returned shape {value.shape}, expected {expected}")
        return value

    def _data_and_pullback(self) -> tuple[Any, Callable[[Any], Any]]:
        """``parameter_data_fn`` at the parameters and its pullback, each compiled once.

        Left eager, the forward pass of ``jax.vjp`` dispatches the whole
        parameters-to-data graph one primitive at a time, and so does every
        later call of the pullback.
        """
        if self._parameter_data_vjp is None:
            data_fn = cast(Callable[[Array], Any], self._parameter_data_fn)
            data, pullback = jax.jit(lambda p: jax.vjp(data_fn, p))(self._parameters)
            self._parameter_data_vjp = (data, jax.jit(pullback))
        return cast(tuple[Any, Callable[[Any], Any]], self._parameter_data_vjp)

    def _parameter_vjp(self, order: int, cotangent: Array) -> Array:
        if self._parameters is None:
            raise RuntimeError(
                "this field was not constructed with optimizable parameters")
        points = self._require_points()

        def spatial_quantity(field_function):
            point_field = lambda point: field_function(point[None, :])[0]  # noqa: E731
            function = point_field
            for _ in range(order):
                function = jax.jacfwd(function)
            return jax.vmap(function)(points)

        if self._parameter_data_fn is not None:
            data, pullback = self._data_and_pullback()
            B_from_data = cast(Callable[[Any, Array], Array], self._B_from_data)

            def quantity_from_data(field_data):
                return spatial_quantity(lambda xyz: B_from_data(field_data, xyz))

            # The pullback below evaluates this graph anyway; take the shape
            # from an abstract trace rather than a second eager evaluation.
            expected = jax.eval_shape(quantity_from_data, data).shape
            vector = jnp.asarray(cotangent)
            if vector.shape != expected:
                raise ValueError(
                    f"cotangent has shape {vector.shape}, expected {expected}")
            if order not in self._data_pullbacks:
                self._data_pullbacks[order] = jax.jit(jax.grad(
                    lambda field_data, weight: jnp.vdot(
                        quantity_from_data(field_data), weight),
                    argnums=0, allow_int=True))
            data_bar = self._data_pullbacks[order](data, vector)
            return pullback(data_bar)[0]

        parameterized_B_fn = cast(Callable[[Array, Array], Array],
                                  self._parameterized_B_fn)

        def quantity(parameters):
            return spatial_quantity(lambda xyz: parameterized_B_fn(parameters, xyz))

        value, pullback = jax.vjp(quantity, self._parameters)
        vector = jnp.asarray(cotangent)
        if vector.shape != value.shape:
            raise ValueError(
                f"cotangent has shape {vector.shape}, expected {value.shape}")
        return pullback(vector)[0]

    def B_vjp(self, vector: Array) -> Array:
        """Return ``vector.T @ dB/dp`` for this field's parameters ``p``."""
        return self._parameter_vjp(0, vector)

    def gradB_vjp(self, vector: Array) -> Array:
        """Return a VJP of :meth:`gradB` with respect to field parameters."""
        return self._parameter_vjp(1, vector)

    def gradgradB_vjp(self, vector: Array) -> Array:
        """Return a VJP of :meth:`gradgradB` with respect to field parameters."""
        return self._parameter_vjp(2, vector)

    def gradgradgradB_vjp(self, vector: Array) -> Array:
        """Return a VJP of :meth:`gradgradgradB` with respect to parameters."""
        return self._parameter_vjp(3, vector)

    def dB_by_dX(self, points: Array | None = None) -> Array:
        """Return SIMSOPT axis order ``(point, x_j, B_i)``."""
        return jnp.swapaxes(self.gradB(points), -1, -2)

    def GradAbsB(self, points: Array | None = None) -> Array:
        """Return the Cartesian gradient of ``|B|``."""
        B = self.B(points)
        gradB = self.gradB(points)
        scale = jnp.maximum(jnp.linalg.norm(B, axis=-1), jnp.finfo(B.dtype).tiny)
        return jnp.einsum("...i,...ij->...j", B, gradB) / scale[:, None]


def _spline_moments(regular: Array) -> Array:
    """Not-a-knot cubic-spline second derivatives on the uniform ``s`` mesh.

    A piecewise-linear interpolant has zero second derivative inside every
    cell, so the second and third radial derivatives of ``B`` do not converge
    with ``ns`` at all.  A C2 spline restores them.  The system is solved once
    per field, for every Fourier coefficient at once, and its matrix depends
    only on ``ns``.
    """
    ns = regular.shape[0]
    if ns < 4:  # not-a-knot needs four nodes; fall back to the linear interpolant
        return jnp.zeros_like(regular)
    h = 1.0 / (ns - 1)
    matrix = np.zeros((ns, ns))
    rows = np.arange(1, ns - 1)
    matrix[rows, rows - 1], matrix[rows, rows], matrix[rows, rows + 1] = 1.0, 4.0, 1.0
    matrix[0, 0], matrix[0, 1], matrix[0, 2] = 1.0, -2.0, 1.0
    matrix[-1, -3], matrix[-1, -2], matrix[-1, -1] = 1.0, -2.0, 1.0
    rhs = jnp.zeros_like(regular).at[1:-1].set(
        6.0 / h**2 * (regular[:-2] - 2.0 * regular[1:-1] + regular[2:]))
    return jnp.linalg.solve(jnp.asarray(matrix, dtype=regular.dtype), rhs)


def _radial_table(coefficients: Array, modes: Array | None,
                  spline: bool = True) -> tuple[Array, Array]:
    """Axis-regularized coefficients and their spline moments.

    A regular scalar Fourier coefficient with poloidal mode ``m`` behaves as
    ``rho**|m|`` near the magnetic axis, where ``rho=sqrt(s)``.  Interpolating
    the physical coefficient directly would incorrectly make an ``m=1`` mode
    linear in ``s``; the regularized coefficient is the analytic one, and its
    radial power is restored after interpolation.
    """
    coefficients = jnp.asarray(coefficients)
    if modes is None:
        regular = coefficients
    else:
        ns = coefficients.shape[0]
        powers = jnp.abs(jnp.asarray(modes)) / 2.0
        s_mesh = jnp.arange(ns, dtype=coefficients.dtype) / (ns - 1)
        scale = s_mesh[:, None] ** powers[None, :]
        regular = coefficients / jnp.where(scale == 0.0, 1.0, scale)
        regular = regular.at[0].set(jnp.where(powers > 0, regular[1], regular[0]))
    # Zero moments reduce the evaluation below to plain linear interpolation.
    return regular, _spline_moments(regular) if spline else jnp.zeros_like(regular)


def _radial_value_and_derivative(
    coefficients: Array, s: Array, modes: Array | None = None,
    table: tuple[Array, Array] | None = None,
) -> tuple[Array, Array]:
    """Interpolate full-mesh spectra while preserving VMEC radial parity.

    ``table`` is the output of :func:`_radial_table` for these coefficients,
    built once per field rather than once per evaluated point.
    """
    regular, moments = _radial_table(coefficients, modes) if table is None else table
    ns = regular.shape[0]
    coordinate = jnp.clip(s, 0.0, 1.0) * (ns - 1)
    index = jnp.clip(jnp.floor(coordinate).astype(int), 0, ns - 2)
    step = 1.0 / (ns - 1)
    upper_weight = coordinate - index
    lower_weight = 1.0 - upper_weight
    lower, upper = regular[index], regular[index + 1]
    lower_moment, upper_moment = moments[index], moments[index + 1]
    slope = (upper - lower) * (ns - 1)
    value = (lower_weight * lower + upper_weight * upper
             + (step**2 / 6.0) * ((lower_weight**3 - lower_weight) * lower_moment
                                  + (upper_weight**3 - upper_weight) * upper_moment))
    derivative = slope + (step / 6.0) * (
        (1.0 - 3.0 * lower_weight**2) * lower_moment
        + (3.0 * upper_weight**2 - 1.0) * upper_moment)
    if modes is not None:
        safe_s = jnp.maximum(s, jnp.finfo(regular.dtype).tiny)
        powers = jnp.abs(jnp.asarray(modes)) / 2.0
        physical_scale = safe_s ** powers
        scale_derivative = jnp.where(
            powers > 0, powers * safe_s ** (powers - 1.0), 0.0)
        derivative = physical_scale * derivative + scale_derivative * value
        value = physical_scale * value
    return value, derivative


def _prepared(spectra: dict[str, Array]) -> dict[str, Any]:
    """Return ``spectra`` with the radial tables its hot paths reuse.

    Building them once per field, rather than once per evaluated point, is what
    makes the spline affordable: the solve is hoisted out of every ``vmap``.
    """
    if "_tables" in spectra:
        return spectra
    xm, xmn = spectra["xm"], spectra["xmn"]
    # The C2 interpolant is a requirement of the native form, whose Jacobian
    # carries R_s and Z_s; the fitted path neither needs it nor benefits from
    # it.  Giving it to both differentiated the half-mesh conversion's error
    # twice on a path with no native form to absorb it, which cost the fitted
    # second derivative up to ns = 82 and the third up to ns = 147 - the range
    # real wouts are written at.  So the interpolant follows the form.
    native = _has_native_form(spectra)
    tables = {
        "rmnc": _radial_table(spectra["rmnc"], xm, native),
        "zmns": _radial_table(spectra["zmns"], xm, native),
        "bsupu": _radial_table(
            _full_mesh_contravariant(spectra["bsupu"], xmn), xmn, native),
        "bsupv": _radial_table(
            _full_mesh_contravariant(spectra["bsupv"], xmn), xmn, native),
    }
    if native:
        tables["lmns"] = _radial_table(spectra["lmns"], xm)
        for name in ("phipf", "chipf"):
            tables[name] = _radial_table(
                jnp.reshape(jnp.asarray(spectra[name]), (-1, 1)), None)
    return dict(spectra, _tables=tables)


def _flux_coordinates_to_xyz(spectra: dict[str, Array], points: Array) -> Array:
    """Map VMEC ``(s, theta, phi)`` coordinates to Cartesian points."""
    spectra = _prepared(spectra)
    s, theta, phi = _check_points(points, "flux coordinates").T
    tables = spectra["_tables"]
    radial_r = jax.vmap(lambda value: _radial_value_and_derivative(
        None, value, spectra["xm"], tables["rmnc"])[0])(s)
    radial_z = jax.vmap(lambda value: _radial_value_and_derivative(
        None, value, spectra["xm"], tables["zmns"])[0])(s)
    phase = spectra["xm"][None, :] * theta[:, None] - spectra["xn"][None, :] * phi[:, None]
    radius = jnp.sum(radial_r * jnp.cos(phase), axis=1)
    z = jnp.sum(radial_z * jnp.sin(phase), axis=1)
    return jnp.stack((radius * jnp.cos(phi), radius * jnp.sin(phi), z), axis=1)


def _B_contravariant_flux(spectra: dict[str, Array], points: Array) -> Array:
    """Return ``(B^s, B^theta, B^phi)`` at VMEC flux coordinates."""
    spectra = _prepared(spectra)
    s, theta, phi = _check_points(points, "flux coordinates").T
    # The prepared tables, so every entry point shares one interpolant: these
    # helpers building their own disagreed with the field by 6e-4 once the
    # interpolant started following the form.
    tables = spectra["_tables"]
    bu_coeff = jax.vmap(lambda value: _radial_value_and_derivative(
        None, value, spectra["xmn"], tables["bsupu"])[0])(s)
    bv_coeff = jax.vmap(lambda value: _radial_value_and_derivative(
        None, value, spectra["xmn"], tables["bsupv"])[0])(s)
    phase = spectra["xmn"][None, :] * theta[:, None] - spectra["xnn"][None, :] * phi[:, None]
    return jnp.stack((jnp.zeros_like(s), jnp.sum(bu_coeff * jnp.cos(phase), axis=1),
                      jnp.sum(bv_coeff * jnp.cos(phase), axis=1)), axis=1)


class _VmecFluxCoordinateField:
    """ESSOS tracing adapter whose points and vectors use ``(s, theta, phi)``."""

    def __init__(self, spectra: dict[str, Array]) -> None:
        self.spectra = spectra

    def B_contravariant(self, point: Array) -> Array:
        point = jnp.asarray(point)
        if point.shape != (3,):
            raise ValueError(f"point must have shape (3,), got {point.shape}")
        return _B_contravariant_flux(self.spectra, point[None, :])[0]

    def to_xyz(self, point: Array) -> Array:
        point = jnp.asarray(point)
        if point.shape != (3,):
            raise ValueError(f"point must have shape (3,), got {point.shape}")
        return _flux_coordinates_to_xyz(self.spectra, point[None, :])[0]

    def to_xyz_batch(self, points: Array) -> Array:
        """Map an array of ``(s, theta, phi)`` points without nested vmaps."""
        return _flux_coordinates_to_xyz(self.spectra, points)

    @staticmethod
    def toroidal_angle_batch(points: Array) -> Array:
        """Return the continuous native VMEC toroidal coordinate."""
        return _check_points(points, "flux coordinates")[:, 2]


def _full_mesh_contravariant(coefficients: Array, modes: Array) -> Array:
    """Interpolate half-mesh VMEC spectra with a regular magnetic-axis row."""
    coefficients = jnp.asarray(coefficients)
    interior = 0.5 * (coefficients[1:-1] + coefficients[2:])
    edge = 1.5 * coefficients[-1] - 0.5 * coefficients[-2]
    axis = jnp.where(jnp.asarray(modes) == 0,
                     1.5 * coefficients[1] - 0.5 * coefficients[2], 0.0)
    return jnp.concatenate((axis[None], interior, edge[None]), axis=0)


def _half_to_full_profile(profile: Array) -> Array:
    """Move a half-mesh radial profile to the full mesh.

    A 1-D profile's midpoint-average error is smooth in ``s``, unlike the
    per-mode tables of :func:`_full_mesh_contravariant`, so a C2 interpolant
    differentiates it without amplifying node-to-node structure.
    """
    profile = jnp.asarray(profile)
    interior = 0.5 * (profile[1:-1] + profile[2:])
    edge = 1.5 * profile[-1] - 0.5 * profile[-2]
    axis = 1.5 * profile[1] - 0.5 * profile[2]
    return jnp.concatenate((axis[None], interior, edge[None]), axis=0)


def _geometry(spectra: dict[str, Array], s: Array, theta: Array, phi: Array):
    """Return ``R, Z`` and their ``s``, ``theta`` and ``phi`` derivatives at one point."""
    xm, xn = spectra["xm"], spectra["xn"]
    rc, rcs = _radial_value_and_derivative(spectra["rmnc"], s, xm)
    zs, zss = _radial_value_and_derivative(spectra["zmns"], s, xm)
    phase = xm * theta - xn * phi
    cosine, sine = jnp.cos(phase), jnp.sin(phase)
    R, Z = jnp.vdot(rc, cosine), jnp.vdot(zs, sine)
    Rs, Zs = jnp.vdot(rcs, cosine), jnp.vdot(zss, sine)
    Rt, Zt = jnp.vdot(-xm * rc, sine), jnp.vdot(xm * zs, cosine)
    Rp, Zp = jnp.vdot(xn * rc, sine), jnp.vdot(-xn * zs, cosine)
    return R, Z, Rs, Zs, Rt, Zt, Rp, Zp


def _has_native_form(spectra: dict[str, Array]) -> bool:
    """Whether the spectra carry what the native VMEC field form needs."""
    return all(spectra.get(name) is not None
               for name in ("lmns", "phipf", "chipf"))


def _contravariant_native(spectra, s, theta, phi, Rs, Zs, Rt, Zt, R):
    """``(B^theta, B^zeta)`` from the native VMEC form.

    ``B^theta = (chi' - lambda_zeta) / sqrt(g)`` and
    ``B^zeta = (phi' + lambda_theta) / sqrt(g)``, with ``sqrt(g)`` built from
    the same ``R`` and ``Z`` series that the position uses, so the field is
    divergence-free in the angles identically rather than approximately.
    ``lmns`` already carries VMEC's internal ``lamscale`` factor, and ``phipf``
    and ``chipf`` are the internal ``phip`` and ``chi'`` on the full mesh.

    The alternative, interpolating the Nyquist ``B^u``/``B^v`` tables, fits a
    rational function to a truncated series and then interpolates the fit
    radially on the half mesh.  On an exact oracle that path leaves
    ``gradgradB`` wrong by 4.5 to 72 per cent at every ``ns``; this one reaches
    2.2e-4 at ``ns = 41``.
    """
    tables = spectra["_tables"]
    xm, xn = spectra["xm"], spectra["xn"]
    lmns, _ = _radial_value_and_derivative(None, s, xm, tables["lmns"])
    phase = xm * theta - xn * phi
    cosine = jnp.cos(phase)
    lambda_theta = jnp.vdot(lmns * xm, cosine)
    lambda_phi = jnp.vdot(lmns * -xn, cosine)
    phip, _ = _radial_value_and_derivative(None, s, None, tables["phipf"])
    chip, _ = _radial_value_and_derivative(None, s, None, tables["chipf"])
    # VMEC's jacobian.f assembles tau as ru12*zs - rs*zu12, so sqrt(g) is
    # R (R_theta Z_s - R_s Z_theta); the opposite ordering flips its sign.
    jacobian = R * (Rt * Zs - Rs * Zt)
    return ((chip[0] - lambda_phi) / jacobian,
            (phip[0] + lambda_theta) / jacobian)


def _position_and_field(spectra: dict[str, Array], coordinates: Array) -> tuple[Array, Array]:
    """Cartesian position and ``B`` at one ``(rho, theta, phi)``, ``rho = sqrt(s)``."""
    rho, theta, phi = coordinates
    s = rho**2
    xmn, xnn = spectra["xmn"], spectra["xnn"]
    R, Z, Rs, Zs, Rt, Zt, Rp, Zp = _geometry(spectra, s, theta, phi)
    tables = spectra["_tables"]
    if _has_native_form(spectra):
        # d(rho)/ds carries the chain rule: _geometry differentiates in s.
        bu, bv = _contravariant_native(spectra, s, theta, phi, Rs, Zs, Rt, Zt, R)
    else:
        bu_coeff, _ = _radial_value_and_derivative(None, s, xmn, tables["bsupu"])
        bv_coeff, _ = _radial_value_and_derivative(None, s, xmn, tables["bsupv"])
        nyquist_phase = xmn * theta - xnn * phi
        bu = jnp.vdot(bu_coeff, jnp.cos(nyquist_phase))
        bv = jnp.vdot(bv_coeff, jnp.cos(nyquist_phase))
    cphi, sphi = jnp.cos(phi), jnp.sin(phi)
    e_theta = jnp.array((Rt * cphi, Rt * sphi, Zt))
    e_phi = jnp.array((Rp * cphi - R * sphi, Rp * sphi + R * cphi, Zp))
    return jnp.array((R * cphi, R * sphi, Z)), bu * e_theta + bv * e_phi


def _inverse3(matrix: Array) -> Array:
    """Adjugate inverse of one 3 x 3 matrix: a few products under nested AD."""
    (a, b, c), (d, e, f), (g, h, i) = matrix
    cofactors = jnp.array(((e * i - f * h, c * h - b * i, b * f - c * e),
                           (f * g - d * i, a * i - c * g, c * d - a * f),
                           (d * h - e * g, b * g - a * h, a * e - b * d)))
    return cofactors / (a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g))


def _cartesian_derivative(
    spectra: dict[str, Array], order: int, coordinates: Array, valid: Array,
) -> Array:
    """``order``-th Cartesian derivative of ``B`` at ``(rho, theta, phi)`` points.

    In flux coordinates ``w`` a Cartesian derivative is ``d/dw`` times the
    inverse Jacobian of the position, so every order is an explicit function of
    ``w``.  Neither the coordinate inversion nor its implicit-function rule is
    nested into the derivative graph, which roughly halves what XLA compiles.
    """
    spectra = _prepared(spectra)
    position = lambda w: _position_and_field(spectra, w)[0]  # noqa: E731
    function = lambda w: _position_and_field(spectra, w)[1]  # noqa: E731
    for _ in range(order):
        function = (lambda previous: lambda w: jax.jacfwd(previous)(w) @ _inverse3(
            jax.jacfwd(position)(w)))(function)
    value = jax.vmap(function)(coordinates)
    return jnp.where(valid.reshape(valid.shape + (1,) * (value.ndim - 1)), value, jnp.nan)


def _invert_coordinates(
    spectra: dict[str, Array], points: Array, *, newton_iterations: int,
    initial_flux: Array | None = None,
) -> tuple[Array, Array]:
    """Invert Cartesian points to ``(rho, theta, phi)`` and flag the valid ones.

    ``(rho, theta)`` is an implicit function of the point and of the spectra,
    so its derivatives come from the residual at the root through a 2 x 2
    solve (:func:`jax.lax.custom_root`), at every order.  The Newton loop
    itself is never differentiated: it runs to a tolerance, at most
    ``newton_iterations`` steps plus one polishing step.
    """
    points = _check_points(points)
    # Seeded callers already know which surface each point is on, so they need
    # neither the sweep that seeds the iteration nor the polygon that decides
    # whether a failure is worth reporting.  Branching here rather than inside
    # the traced body keeps their graph, and their results, as they were.
    seeded = initial_flux is not None
    if initial_flux is None:
        initial_flux = jnp.full_like(points, jnp.nan)
    initial_flux = _check_points(initial_flux, "initial flux coordinates")
    if initial_flux.shape != points.shape:
        raise ValueError("initial_flux and points must have the same shape")
    spectra = _prepared(spectra)
    rho_nodes = jnp.linspace(1.0 / _GUESS_NRHO, 1.0, _GUESS_NRHO)
    theta_nodes = jnp.linspace(0.0, 2.0 * jnp.pi, _GUESS_NTHETA + 1)[:-1]

    def geometry(s, theta, phi):
        return _geometry(spectra, s, theta, phi)

    def one_point(point, initial):
        x, y, z = point
        radius, phi = jnp.hypot(x, y), jnp.arctan2(y, x)
        axis_R, axis_Z, *_ = geometry(0.0, 0.0, phi)
        # VMEC's (s, theta) chart collapses at the magnetic axis although B is
        # regular there. Evaluate an infinitesimal off-axis representative;
        # stop_gradient keeps Cartesian derivatives equal to their limiting
        # off-axis values instead of differentiating the coordinate choice.
        axis_rho = jnp.asarray(1.0e-6, dtype=point.dtype)
        sample_R, sample_Z, *_ = geometry(axis_rho**2, 0.0, phi)
        sample = jnp.array((sample_R * jnp.cos(phi), sample_R * jnp.sin(phi), sample_Z))
        axis_distance2 = (radius - axis_R) ** 2 + (z - axis_Z) ** 2
        on_axis = axis_distance2 <= (16.0 * jnp.finfo(point.dtype).eps) ** 2
        point = point + jax.lax.stop_gradient(jnp.where(on_axis, sample - point, 0.0))
        x, y, z = point
        radius, phi = jnp.hypot(x, y), jnp.arctan2(y, x)
        axis_R, axis_Z, *_ = geometry(0.0, 0.0, phi)
        geometric_theta = jnp.arctan2(z - axis_Z, radius - axis_R)
        edge_R, edge_Z, *_ = geometry(1.0, geometric_theta, phi)
        edge_distance2 = (edge_R - axis_R) ** 2 + (edge_Z - axis_Z) ** 2
        geometric_rho = jnp.sqrt(((radius - axis_R) ** 2 + (z - axis_Z) ** 2)
                                 / jnp.maximum(edge_distance2, 1.0e-24))
        if seeded:
            guess_rho, guess_theta = geometric_rho, geometric_theta
        else:
            # The geometric guess equates the VMEC poloidal angle with the
            # polar angle about the axis.  On a shaped boundary the two differ
            # enough that the undamped step below leaves the basin of the
            # root, pins against the rho clip where the determinant guard
            # turns the angle step into noise, and wanders until the iteration
            # cap stops it.  Sweep the forward map at this toroidal angle and
            # start from whichever candidate actually lands nearest the query.
            node_R, node_Z = jax.vmap(lambda r: jax.vmap(
                lambda t: geometry(r**2, t, phi)[:2])(theta_nodes))(rho_nodes)
            offsets = (node_R - radius) ** 2 + (node_Z - z) ** 2
            best = jnp.unravel_index(jnp.argmin(offsets), offsets.shape)
            # Inside the innermost ring the sweep has nothing to offer, but
            # there the map is linear: (R, Z) - axis = A (rho cos t, rho sin t)
            # + O(rho^2), and antisymmetric differences about the axis cancel
            # every even-m term of A.  That candidate carries the near-axis
            # points the other two miss.
            probe = jnp.asarray(1.0e-2, dtype=point.dtype)

            def spoke(angle):
                ahead = jnp.stack(geometry(probe**2, angle, phi)[:2])
                behind = jnp.stack(geometry(probe**2, angle + jnp.pi, phi)[:2])
                return (ahead - behind) / (2.0 * probe)

            linearized = jnp.linalg.solve(
                jnp.stack((spoke(0.0), spoke(0.5 * jnp.pi)), axis=1),
                jnp.stack((radius - axis_R, z - axis_Z)))
            candidates = jnp.stack((
                jnp.stack((geometric_rho, geometric_theta)),
                jnp.stack((rho_nodes[best[0]], theta_nodes[best[1]])),
                jnp.stack((jnp.hypot(linearized[0], linearized[1]),
                           jnp.arctan2(linearized[1], linearized[0]))),
            ))

            def missed_by(candidate):
                R_try, Z_try, *_ = geometry(
                    jnp.clip(candidate[0], 1.0e-12, 1.0) ** 2, candidate[1], phi)
                offset = (R_try - radius) ** 2 + (Z_try - z) ** 2
                # A singular linearization gives a non-finite candidate; rank
                # it last rather than letting argmin select the NaN.
                return jnp.where(jnp.isfinite(offset), offset, jnp.inf)

            chosen = candidates[jnp.argmin(jax.vmap(missed_by)(candidates))]
            # The guess is not differentiated: the root carries the whole
            # dependence through custom_root below.
            guess_rho, guess_theta = jax.lax.stop_gradient((chosen[0], chosen[1]))
        rho0 = jnp.where(jnp.isfinite(initial[0]), jnp.sqrt(initial[0]), guess_rho)
        theta0 = jnp.where(jnp.isfinite(initial[1]), initial[1], guess_theta)
        # An on-axis query was displaced above to the representative at
        # ``axis_rho``; start it from that point's own coordinates.  Any other
        # start, in particular the ``s = 0`` a caller seeds the axis with, is
        # the one place where the Jacobian vanishes and the first step is set
        # by the determinant guard rather than by the geometry.
        rho0 = jnp.where(on_axis, axis_rho, rho0)
        theta0 = jnp.where(on_axis, 0.0, theta0)

        def residual(coordinates):
            R, Z, *_ = geometry(coordinates[0] ** 2, coordinates[1], phi)
            return jnp.stack((R - radius, Z - z))

        def newton_step(coordinates):
            rho, theta = coordinates
            R, Z, Rs, Zs, Rt, Zt, *_ = geometry(rho**2, theta, phi)
            Rrho, Zrho = 2.0 * rho * Rs, 2.0 * rho * Zs
            determinant = Rrho * Zt - Rt * Zrho
            safe = jnp.where(jnp.abs(determinant) > 1.0e-14, determinant, 1.0e-14)
            residual_R, residual_Z = R - radius, Z - z
            drho = (Zt * residual_R - Rt * residual_Z) / safe
            dt = (-Zrho * residual_R + Rrho * residual_Z) / safe
            return jnp.stack((jnp.clip(rho - drho, 1.0e-12, jnp.sqrt(1.05)),
                              jnp.mod(theta - dt, 2.0 * jnp.pi)))

        def solve(_, guess):
            def unconverged(carry):
                count, coordinates = carry
                return (count < int(newton_iterations)) & (
                    jnp.linalg.norm(residual(coordinates)) > 1.0e-9)

            _, coordinates = jax.lax.while_loop(
                unconverged, lambda carry: (carry[0] + 1, newton_step(carry[1])),
                (0, guess))
            return newton_step(coordinates)  # quadratic polish down to round-off

        def tangent_solve(linearized, rhs):
            (a, c), (b, d) = (linearized(jnp.array((1.0, 0.0), dtype=rhs.dtype)),
                              linearized(jnp.array((0.0, 1.0), dtype=rhs.dtype)))
            determinant = a * d - b * c
            safe = jnp.where(jnp.abs(determinant) > 1.0e-14, determinant, 1.0e-14)
            return jnp.stack((d * rhs[0] - b * rhs[1], a * rhs[1] - c * rhs[0])) / safe

        rho, theta = jax.lax.custom_root(
            residual, jnp.stack((jnp.clip(rho0, 1.0e-12, 1.0), theta0)),
            solve, tangent_solve)
        s = rho**2
        error = jnp.linalg.norm(residual(jnp.stack((rho, theta))))
        valid = (s >= -1.0e-8) & (s <= 1.0 + 1.0e-8) & (error <= 1.0e-7)
        return jnp.stack((rho, theta, phi)), valid

    return jax.vmap(one_point)(points, initial_flux)


def _inside_boundary(spectra: dict[str, Array], points: Array) -> Array:
    """Which Cartesian points lie inside the last closed surface.

    An iteration that did not converge stops wherever it happens to stop, so
    the ``s`` it reports says nothing about where the query point is: on the
    decks measured, a large fraction of points several minor radii out still
    finish at ``s <= 1``.  Deciding whether a stalled point is interior — and
    so whether its NaN is a defect or the documented answer for an exterior
    point — has to be done on the point itself.  The boundary is sampled at
    ``_GUESS_NTHETA`` poloidal nodes at the query's toroidal angle and shrunk
    by :data:`_INSIDE_MARGIN` towards the magnetic axis, so the polygon's
    finite resolution can only ever suppress a complaint, never invent one.
    """
    theta_nodes = jnp.linspace(0.0, 2.0 * jnp.pi, _GUESS_NTHETA + 1)[:-1]

    def one_point(point):
        x, y, z = point
        radius, phi = jnp.hypot(x, y), jnp.arctan2(y, x)
        axis_R, axis_Z, *_ = _geometry(spectra, 0.0, 0.0, phi)
        edge = jax.vmap(lambda t: jnp.stack(
            _geometry(spectra, 1.0, t, phi)[:2]))(theta_nodes)
        centre = jnp.stack((axis_R, axis_Z))
        vertices = centre + _INSIDE_MARGIN * (edge - centre)
        following = jnp.roll(vertices, -1, axis=0)
        rise = following[:, 1] - vertices[:, 1]
        straddles = (vertices[:, 1] > z) != (following[:, 1] > z)
        fraction = (z - vertices[:, 1]) / jnp.where(rise == 0.0, 1.0, rise)
        crossing = vertices[:, 0] + fraction * (following[:, 0] - vertices[:, 0])
        return jnp.sum(jnp.where(straddles & (crossing > radius), 1, 0)) % 2 == 1

    return jax.vmap(one_point)(_check_points(points))


def _interior_coordinates_and_B(
    spectra: dict[str, Array], points: Array, *, newton_iterations: int,
    initial_flux: Array | None = None,
) -> tuple[Array, Array]:
    """Invert VMEC coordinates and synthesize the interior Cartesian field."""
    coordinates, valid = _invert_coordinates(
        spectra, points, newton_iterations=newton_iterations, initial_flux=initial_flux)
    field = _cartesian_derivative(spectra, 0, coordinates, valid)
    return coordinates.at[:, 0].set(coordinates[:, 0] ** 2), field


class VmecInteriorField(MagneticField):
    """VMEC magnetic field at Cartesian points inside the plasma boundary.

    Cartesian points are inverted to ``(s, theta, phi)`` by a Newton solve run
    to a tolerance, differentiated implicitly at its root rather than through
    its iterations.  Spectral angular evaluation and radial interpolation then
    recover ``B``, and its Cartesian derivatives are taken explicitly in flux
    coordinates.  Points outside the last closed surface return NaNs; use
    :class:`VmecExtender` there.  An eager call whose inversion did not
    converge inside the plasma raises
    :class:`~vmex.core.errors.VmecNumericalError` instead.

    ``s`` is the normalised toroidal flux ``psi / psi_edge`` on ``[0, 1]``,
    and ``theta`` (poloidal) and ``phi`` (geometric toroidal) are in radians.
    A point is rejected — every field component set to NaN — when the
    converged ``s`` leaves ``[0, 1]`` by more than ``1e-8`` or the inverted
    ``(R, Z)`` still misses the query point by more than ``1e-7`` m, so a
    non-converged inversion cannot be mistaken for a field value.

    Parameters
    ----------
    spectra:
        VMEC Fourier tables of one equilibrium, in the layout built by
        ``vmex.core.virtual_casing._state_field_spectra`` (also what
        :meth:`from_state` produces).  The keys read here are ``rmnc`` and
        ``zmns``, full-mesh geometry coefficients of shape ``(ns, mnmax)``
        in metres; ``xm`` and ``xn``, shape ``(mnmax,)``, the poloidal mode
        number ``m`` and the *already field-period-scaled* toroidal mode
        number ``n * nfp`` of the wout convention, so the angle is
        ``m theta - n_scaled phi``; ``bsupu`` and ``bsupv``, the
        contravariant field cosine coefficients ``B^theta`` and ``B^phi``
        in T/m of shape ``(ns, mnmax_nyq)`` on the VMEC half mesh with row 0
        the unused axis row; and their Nyquist mode numbers ``xmn`` and
        ``xnn``, shape ``(mnmax_nyq,)``.  Only the stellarator-symmetric
        (``lasym = False``) families are evaluated.  Values may be JAX
        tracers, which is what makes the whole field differentiable.
    newton_iterations:
        Largest number of Newton steps taken to invert ``(R, Z) -> (s,
        theta)`` at a query point; the loop stops as soon as the position
        residual falls below ``1e-9`` m and then polishes once.  The loop is
        never differentiated, so the bound costs nothing in derivative
        graphs; raise it for strongly shaped boundaries whose geometric
        first guess is poor.
    parameters, parameterized_B_fn, parameter_data_fn, B_from_data, dof_names:
        Optimizable-parameter arguments forwarded unchanged to
        :class:`MagneticField`; see its documentation.  On the factored
        path ``parameter_data_fn`` returns a ``spectra`` mapping of the
        same shape, and after :meth:`set_points_flux` the parameter VJPs
        reuse the stored flux coordinates as Newton seeds and hold the
        mapped Cartesian points fixed.
    """

    def __init__(
        self,
        spectra: dict[str, Array],
        *,
        newton_iterations: int = 10,
        parameters: Array | None = None,
        parameterized_B_fn: Callable[[Array, Array], Array] | None = None,
        parameter_data_fn: Callable[[Array], Any] | None = None,
        B_from_data: Callable[[Any, Array], Array] | None = None,
        dof_names: tuple[str, ...] = (),
    ) -> None:
        self.spectra = spectra
        self.newton_iterations = int(newton_iterations)
        self._points_flux: Array | None = None
        self._derivative_fns: dict[tuple[int, int, int], Callable[[Array, Array], Array]] = {}
        self._warned = False

        def B_fn(points):
            return _interior_coordinates_and_B(
                spectra, points, newton_iterations=self.newton_iterations)[1]

        super().__init__(
            B_fn, parameters=parameters, parameterized_B_fn=parameterized_B_fn,
            parameter_data_fn=parameter_data_fn, B_from_data=B_from_data,
            dof_names=dof_names)

    def set_points(self, points: Array) -> "VmecInteriorField":
        """Store Cartesian points and clear any cached flux coordinates."""
        self._points_flux = None
        super().set_points(points)
        return self

    def set_points_flux(self, points: Array) -> "VmecInteriorField":
        """Map VMEC ``(s, theta, phi)`` to stored Cartesian points.

        Parameter VJPs hold these mapped physical points fixed.
        """
        self._points_flux = _check_points(points, "flux coordinates")
        super().set_points(_flux_coordinates_to_xyz(self.spectra, self._points_flux))
        return self

    def _seeds(self, points: Array | None) -> tuple[Array, Array]:
        """Cartesian points and their Newton seeds (NaN selects the geometric guess)."""
        if points is None and self._points_flux is not None:
            return self._require_points(), self._points_flux
        xyz = self._require_points() if points is None else _check_points(points)
        return xyz, jnp.full_like(xyz, jnp.nan)

    def _evaluate(self, order: int, points: Array | None) -> Array:
        """``order``-th Cartesian derivative of ``B``, compiled once per order."""
        xyz, seeds = self._seeds(points)
        spectra, iterations = self.spectra, self.newton_iterations
        if order >= 2 and not _has_native_form(spectra) and not self._warned:
            self._warned = True
            warnings.warn(
                "this field was built from the fitted B^u/B^v spectra, whose "
                "second and third radial derivatives do not converge with ns: "
                "on an exact oracle they stay near 4.5e-2 and 5.8e-1 at every "
                "resolution. B and its first derivative are unaffected. For "
                "converged high derivatives, build the field from spectra that "
                "carry lmns, phipf and chipf, which any live equilibrium "
                "supplies; those reach 2.2e-4 and 7.8e-4 at ns = 41.",
                InteriorFieldAccuracyWarning, stacklevel=3)
        key = (order, id(spectra), iterations)  # a reassigned attribute recompiles
        evaluate = self._derivative_fns.get(key)
        if evaluate is None:

            def evaluate(xyz, seeds):
                return _cartesian_derivative(spectra, order, *_invert_coordinates(
                    spectra, xyz, newton_iterations=iterations, initial_flux=seeds))

            evaluate = self._derivative_fns[key] = jax.jit(evaluate)
        return self._loud(evaluate(xyz, seeds), xyz, seeds)

    def _loud(self, value: Array, xyz: Array, seeds: Array) -> Array:
        """Raise when an eager result is NaN because the inversion stalled.

        A point outside the plasma still returns NaN, however far out it is.
        One that lies inside the boundary but whose inverted position misses
        the query point did not converge, and that must not pass for a field
        value.  Traced calls cannot raise and keep the NaN.
        """
        if isinstance(value, jax.core.Tracer) or not bool(jnp.isnan(value).any()):
            return value
        _, valid = _invert_coordinates(
            self.spectra, xyz, newton_iterations=self.newton_iterations,
            initial_flux=seeds)
        # Seeded callers were told which surface the point is on; everyone
        # else is classified against the boundary itself, because an
        # unconverged iterate's own s carries no information about where the
        # point is (see _inside_boundary).  The distinction is per point, not
        # per call: _seeds hands back a NaN row for every unseeded point rather
        # than None, and NaN <= 1 is False, so testing the array as a whole
        # silently classified every unseeded point as exterior and suppressed
        # the complaint this method exists to raise.
        label = jnp.asarray(seeds[:, 0])
        interior = jnp.where(jnp.isfinite(label), label <= 1.0 + 1.0e-8,
                             _inside_boundary(self.spectra, xyz))
        stalled = ~valid & interior
        if bool(stalled.any()):
            raise VmecNumericalError(
                f"the Cartesian-to-flux Newton inversion did not converge at "
                f"{int(stalled.sum())} of {stalled.size} interior points within "
                f"newton_iterations={self.newton_iterations}",
                hint="raise newton_iterations, or pass flux coordinates with "
                "set_points_flux()")
        return value

    def _parameter_vjp(self, order: int, cotangent: Array) -> Array:
        """Differentiate at fixed Cartesian points, the root following the spectra."""
        if self._parameter_data_fn is None:
            return super()._parameter_vjp(order, cotangent)
        if self._parameters is None:
            raise RuntimeError(
                "this field was not constructed with optimizable parameters")
        points, seeds = self._seeds(None)
        data, pullback = self._data_and_pullback()
        iterations = self.newton_iterations

        def quantity_from_data(field_data, points, seeds):
            # Only first-order implicit differentiation of the root is needed:
            # the derivative order lives in the explicit flux-coordinate graph.
            return _cartesian_derivative(field_data, order, *_invert_coordinates(
                field_data, points, newton_iterations=iterations, initial_flux=seeds))

        expected = points.shape + (3,) * order
        vector = jnp.asarray(cotangent)
        if vector.shape != expected:
            raise ValueError(
                f"cotangent has shape {vector.shape}, expected {expected}")
        cache_key = (order, iterations)  # distinct from the base class's integer keys
        if cache_key not in self._data_pullbacks:
            self._data_pullbacks[cache_key] = jax.jit(jax.grad(
                lambda field_data, weight, points, seeds: jnp.vdot(
                    quantity_from_data(field_data, points, seeds), weight),
                argnums=0, allow_int=True))
        data_bar = self._data_pullbacks[cache_key](data, vector, points, seeds)
        return pullback(data_bar)[0]

    def B(self, points: Array | None = None) -> Array:
        """Return Cartesian ``B``, seeded by stored flux coordinates when known."""
        return self._evaluate(0, points)

    def gradB(self, points: Array | None = None) -> Array:
        """Return ``dB_i/dx_j``, seeded by stored flux coordinates when known."""
        return self._evaluate(1, points)

    def gradgradB(self, points: Array | None = None) -> Array:
        """Return the second Cartesian derivative of ``B``."""
        return self._evaluate(2, points)

    def gradgradgradB(self, points: Array | None = None) -> Array:
        """Return the third Cartesian derivative of ``B``."""
        return self._evaluate(3, points)

    def get_points_flux(self) -> Array:
        """Return stored points as VMEC ``(s, theta, phi)`` coordinates."""
        return self.flux_coordinates()

    def field_in_flux_coordinates(self) -> _VmecFluxCoordinateField:
        """Return a field-line adapter in the ``(s, theta, phi)`` basis."""
        return _VmecFluxCoordinateField(self.spectra)

    def flux_coordinates(self, points: Array | None = None) -> Array:
        """Return inverted ``(s, theta, phi)`` at explicit or stored points."""
        if points is None and self._points_flux is not None:
            return self._points_flux
        xyz, seeds = self._seeds(points)
        coordinates, field = _interior_coordinates_and_B(
            self.spectra, xyz, newton_iterations=self.newton_iterations)
        self._loud(field, xyz, seeds)
        return coordinates

    @classmethod
    def from_state(cls, inp: Any, state: Any, *, runtime: Any = None,
                   newton_iterations: int = 10) -> "VmecInteriorField":
        """Construct a field from a live converged VMEX state."""
        from .virtual_casing import _state_field_spectra

        return cls(_state_field_spectra(inp, state, runtime),
                   newton_iterations=newton_iterations)

    @classmethod
    def from_parameterized_state(
        cls,
        inp: Any,
        state_runtime_fn: Callable[[Array], tuple[Any, Any]],
        parameters: Array,
        *,
        dof_names: tuple[str, ...] = (),
        newton_iterations: int = 10,
    ) -> "VmecInteriorField":
        """Construct an interior field with exact VJPs in problem parameters."""
        from .virtual_casing import _state_field_spectra

        parameters = jnp.ravel(jnp.asarray(parameters))

        def spectra_of(p):
            state, runtime = state_runtime_fn(p)
            return _state_field_spectra(inp, state, runtime)

        def B_from_spectra(spectra, points):
            return _interior_coordinates_and_B(
                spectra, points, newton_iterations=newton_iterations)[1]

        return cls(
            spectra_of(parameters), newton_iterations=newton_iterations,
            parameters=parameters, parameter_data_fn=spectra_of,
            B_from_data=B_from_spectra,
            dof_names=dof_names)


#: Below this, a normalized pressure or current is this equilibrium's own noise.
_SOURCE_FLOOR = 1.0e-8


def _scalar(wout: Any, name: str) -> float:
    """Return a finite wout scalar, or ``0.0`` when it is absent or not finite."""
    value = getattr(wout, name, None)
    if value is None:
        return 0.0
    number = float(np.asarray(value).reshape(()) if np.ndim(value) else value)
    return number if np.isfinite(number) else 0.0


def _has_plasma_sources(wout: Any) -> bool:
    """Detect pressure or current sources, including zero-net-current cases.

    The quantities are dimensional, so they are judged against this
    equilibrium's own field scale rather than an absolute floor.  A fixed
    ``1e-14`` misfires: the shipped vacuum QA wout carries ``ctor`` of 1.5e-10 A
    and up to 9.9e4 A/m^2 of axis noise in the current-density spectra, so
    ``plasma="auto"`` ran virtual casing on a vacuum equilibrium and returned
    quadrature noise where the answer is zero.  The current-density spectra are
    no longer consulted at all: near the axis they are noise with no scale of
    their own to be judged against, and a pressure or a net current is what
    actually sources an exterior plasma field.
    """
    if abs(_scalar(wout, "betatotal")) > _SOURCE_FLOOR:
        return True
    field = abs(_scalar(wout, "b0")) or abs(_scalar(wout, "volavgB"))
    pressure = getattr(wout, "presf", None)
    if pressure is not None and np.any(np.isfinite(np.asarray(pressure, dtype=float))):
        peak = float(np.nanmax(np.abs(np.asarray(pressure, dtype=float))))
        if peak > 0.0 and (field <= 0.0
                           or 2.0 * MU0 * peak / field**2 > _SOURCE_FLOOR):
            return True
    current, reference = _scalar(wout, "ctor"), abs(_scalar(wout, "rbtor"))
    if current != 0.0 and (reference <= 0.0
                           or MU0 * abs(current) / (2.0 * np.pi * reference)
                           > _SOURCE_FLOOR):
        return True
    return abs(_scalar(wout, "wp")) > 0.0 and field <= 0.0


def _mgrid_from_wout(wout: Any, base_dir: Path | None) -> MgridField | None:
    path_text = str(getattr(wout, "mgrid_file", "")).strip()
    if not path_text or path_text.upper() == "NONE":
        return None
    path = Path(path_text).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    extcur = np.asarray(getattr(wout, "extcur", ()), dtype=float).reshape(-1)
    if not np.any(np.isfinite(extcur) & (extcur != 0.0)):
        # Filling the missing currents with zeros built a coil field that is
        # identically zero, so a free-boundary wout without its currents
        # extended as if there were no coils.  VMEX's own solve_file writes
        # such a wout (nextcur = 0) for a free-boundary deck.
        raise ValueError(
            f"the wout names MGRID_FILE {path_text!r} but carries no coil "
            "currents (nextcur = 0 or every EXTCUR zero), so the coil field "
            "cannot be built from it; pass external_field (an MgridField built "
            "with the deck's EXTCUR, or a coil callable), or build the field "
            "with VmecExtender.from_equilibrium from the solved deck")
    data = read_mgrid(path)
    scaled = np.zeros((data.nextcur,), dtype=float)
    scaled[: min(extcur.size, data.nextcur)] = extcur[: data.nextcur]
    if str(data.mgrid_mode).upper().startswith(("R", "N")):
        raw = np.asarray(data.raw_coil_cur, dtype=float)
        scaled = np.divide(scaled, raw, out=scaled, where=raw != 0.0)
    return MgridField.from_mgrid_data(data, extcur=scaled)


def _source_nphi_for_digits(boundary: Any, digits: int) -> int:
    """Per-period source sampling that reaches ``digits`` one minor radius out.

    ``boundary`` is a wout (``rmnc``/``xm``) or a ``VmecInput`` (``rbc``); both
    carry ``nfp`` and enough of the boundary to get ``R0`` and ``a``.

    The off-surface quadrature error decays as ``exp(-2 pi d / h)`` with ``h``
    the finest source level's largest spacing, whose toroidal part over the
    full torus is ``2 pi R0 / n_toroidal``.  Asking for ``10**-digits`` at
    ``d = a`` gives ``n_toroidal >= digits ln(10) R0 / a``, and the default
    schedule's finest level has ``2 nfp nphi`` toroidal points, so

        nphi >= digits ln(10) R0 / (2 nfp a).

    Measured against the shipped QA wout (``R0/a = 15.9``, ``nfp = 2``) with
    ``ntheta = nphi``, the achieved error at ``d = a`` against a requested 1e-6
    is 6.4e-04 at ``nphi = 32``, 4.3e-07 at 64 and 3.5e-11 at 128; the rule
    returns 64 here.  A boundary of tokamak-like aspect ratio (``R0/a = 3``,
    ``nfp = 1``) lands on the 32 floor, so its grid is unchanged and only
    high-aspect boundaries -- where a fixed 32 missed the requested accuracy by
    four orders -- are refined.
    """
    try:
        nfp = max(int(boundary.nfp), 1)
        if hasattr(boundary, "rmnc"):
            # A wout: the last full-mesh surface, with its own mode table.
            coefficients = np.asarray(boundary.rmnc)[-1]
            poloidal = np.asarray(boundary.xm)
        else:
            # A VmecInput: rbc is indexed [n + ntor, m], and the outboard point
            # R(theta = 0, phi = 0) sums every n, so reduce the toroidal axis
            # first and keep one coefficient per poloidal mode.
            coefficients = np.asarray(boundary.rbc).sum(axis=0)
            poloidal = np.arange(coefficients.size, dtype=float)
        outboard = float(coefficients.sum())
        inboard = float((coefficients * np.cos(poloidal * np.pi)).sum())
        minor = 0.5 * (outboard - inboard)
        major = 0.5 * (outboard + inboard)
    except jax.errors.JAXTypeError as error:
        # Falling back here would silently change the source resolution.
        raise TypeError(
            "the virtual-casing source grid is a static choice and cannot be "
            "derived from a traced boundary; pass nphi and ntheta explicitly"
        ) from error
    except (AttributeError, IndexError, TypeError, ValueError):
        return _DEFAULT_SOURCE_NPHI
    if not (np.isfinite(minor) and np.isfinite(major)) or minor <= 0.0:
        return _DEFAULT_SOURCE_NPHI
    needed = math.log(10.0) * digits * major / (2.0 * nfp * minor)
    # Round up to a power of two so repeated calls share compiled kernels, and
    # keep it inside a range whose cost is measured (0.6 s to 0.8 s per call on
    # the shipped QA wout, both dominated by fixed overhead).
    power = max(_DEFAULT_SOURCE_NPHI, 1 << max(0, math.ceil(math.log2(max(needed, 1.0)))))
    return int(min(power, _MAX_SOURCE_NPHI))


def _concrete(function: Callable[..., Any], *args: Any) -> Any:
    """Evaluate ``function(*args)`` to concrete arrays, even while tracing.

    Cached quadrature data built inside a caller's ``jit`` would otherwise be
    that trace's tracers and leak into every later call.  Compiling ahead of
    time and calling the executable runs it outside any trace; the arguments
    are concrete, so it is the same computation.  (``ensure_compile_time_eval``
    would also work, but dispatches every primitive eagerly: 26 s against
    0.5 s for the source densities.)
    """
    return jax.jit(lambda: function(*args)).lower().compile()()


class VmecExtender(MagneticField):
    """Total field outside the last closed VMEC flux surface.

    Current-free vacuum equilibria use ``external_field`` directly.  For
    finite pressure or plasma current, the internal-current virtual-casing
    branch is added.  External coil currents must lie outside the query region;
    targets must not lie exactly on the source surface.

    :meth:`B` is the plain sum of whichever contributions are present, in
    tesla at Cartesian points in metres.  Prefer the classmethods
    (:meth:`from_wout`, :meth:`from_file`, :meth:`from_state`,
    :meth:`from_equilibrium`, :meth:`from_surface_data`) — this constructor
    is the low-level form that takes already-built pieces.

    **Two quadratures.**  The direct path is a periodic trapezoid rule on the
    source grid; its error decays as ``exp(-2 pi d / h)`` with ``d`` the
    distance to the surface and ``h`` the node spacing, so it is exact to
    rounding a minor radius out and useless within a few spacings.  Near the
    surface the target-graded rule of
    :func:`~vmex.core.virtual_casing.graded_plasma_field` takes over: every
    target gets nodes clustered about its nearest surface point, which keeps
    ``B`` to about 1e-12 and ``grad B`` to about 1e-10 down to 0.01 minor
    radii at about 16 ms per target.  Spatial derivatives of the direct path
    come from the closed-form layer kernels, all orders in one pass over the
    finest schedule level.

    Parameters
    ----------
    external_field:
        The coil or vacuum field, evaluated directly at the query points.
        Either an object exposing ``b_cyl(r, phi, z) -> (B_R, B_phi, B_Z)``
        such as an :class:`~vmex.core.mgrid.MgridField`, or a plain callable
        ``xyz (n, 3) [m] -> B (n, 3) [T]`` such as an ESSOS Biot-Savart coil
        field.  May be ``None`` for the plasma field alone, but then
        ``plasma_field`` must be given.  It must be traceable by JAX for the
        spatial derivatives.
    plasma_field:
        Optional virtual-casing exterior field carrying the field of the
        currents inside the last closed flux surface — a
        ``virtual_casing_jax.VirtualCasingExteriorField``, normally built by
        :meth:`from_surface_data`.  ``None`` selects the pure vacuum path,
        correct only for a current-free equilibrium.
    accuracy_check:
        What an eager call does when the quadrature it used misses the
        ``digits`` the field was built with, judged by
        :meth:`B_error_estimate`: ``"warn"`` (default) emits
        :class:`ExteriorFieldAccuracyWarning`, ``"raise"`` raises
        :class:`ExteriorFieldAccuracyError`, ``"off"`` skips the check.
        Traced calls (``jit``, ``grad``, field-line integrators) never check;
        call :meth:`B_error_estimate` there.  The returned field is the same
        in every mode.  The attribute may also be set after construction.
    near_surface:
        Which quadrature evaluates the plasma field. ``"auto"`` (default):
        eager calls use the direct path wherever its error estimate meets
        ``10**-digits`` and the graded rule at the points where it does not;
        traced calls use the direct path, since a per-point switch under a
        trace would pay for both.  ``"graded"``: every call, traced or not,
        uses the graded rule — for objectives or field-line traces that stay
        close to the surface.  ``"direct"``: every call uses the direct path.
        The attribute may also be set after construction.
    graded_nodes:
        ``(poloidal, toroidal)`` node counts of the graded rule over the full
        torus, default :data:`~vmex.core.virtual_casing.GRADED_NODES`.
    """

    def __init__(
        self, external_field: Any, plasma_field: Any | None = None, *,
        accuracy_check: AccuracyCheck = "warn",
        near_surface: NearSurface = "auto",
        graded_nodes: tuple[int, int] | None = None,
    ) -> None:
        if external_field is None and plasma_field is None:
            raise ValueError("at least one external or plasma field is required")
        self.external_field = external_field
        self.plasma_field = plasma_field
        self.accuracy_check = accuracy_check
        self._kernels: dict[tuple[Any, ...], Any] = {}
        self._graded_nodes = (_GRADED_NODES if graded_nodes is None
                              else (int(graded_nodes[0]), int(graded_nodes[1])))
        self.near_surface = near_surface

        def B_fn(points: Array) -> Array:
            return self._field(0, points)

        super().__init__(B_fn)

    # -- configuration -----------------------------------------------------

    @property
    def accuracy_check(self) -> AccuracyCheck:
        """``"warn"``, ``"raise"`` or ``"off"``; see the class documentation."""
        return self._accuracy_check

    @accuracy_check.setter
    def accuracy_check(self, mode: AccuracyCheck) -> None:
        """Set the eager accuracy-check mode; any other value raises ``ValueError``."""
        if mode not in ("warn", "raise", "off"):
            raise ValueError("accuracy_check must be 'warn', 'raise', or 'off'")
        self._accuracy_check = mode

    @property
    def near_surface(self) -> NearSurface:
        """``"auto"``, ``"graded"`` or ``"direct"``; see the class documentation."""
        return self._near_surface

    @near_surface.setter
    def near_surface(self, mode: NearSurface) -> None:
        """Set the quadrature choice; any other value raises ``ValueError``."""
        if mode not in ("auto", "graded", "direct"):
            raise ValueError("near_surface must be 'auto', 'graded', or 'direct'")
        self._near_surface = mode
        # parameter pullbacks trace the plasma field in the current mode
        self._data_pullbacks = {}

    @property
    def graded_nodes(self) -> tuple[int, int]:
        """``(poloidal, toroidal)`` node counts of the graded rule."""
        return self._graded_nodes

    @property
    def uses_virtual_casing(self) -> bool:
        """Whether plasma-current virtual casing contributes to the field."""
        return self.plasma_field is not None

    def with_graded_quadrature(self, nodes: tuple[int, int] | None = None) -> "VmecExtender":
        """A copy that evaluates the plasma field by the graded rule everywhere.

        For traced use close to the surface — field-line tracing, objectives
        on a nearby coil-clearance surface — where the per-point switch of
        ``near_surface="auto"`` is not available.  Each target costs about the
        same as a direct call on a 128 x 512 grid; fewer ``nodes`` trade
        accuracy for speed (``(64, 256)`` keeps ``B`` to about 1e-5 down to
        0.01 minor radii).  This replaces the first-order near-surface
        continuation, which carried a 1.6-2.4 % error floor next to the
        surface and took minutes and tens of GB to prepare.
        """
        if self.plasma_field is None:
            raise RuntimeError("graded quadrature requires a virtual-casing plasma field")
        other = copy.copy(self)
        other._kernels = {}
        other._spatial_fns = {}
        other._data_pullbacks = {}
        if nodes is not None:
            other._graded_nodes = (int(nodes[0]), int(nodes[1]))
        other.near_surface = "graded"
        return other

    # -- evaluation ----------------------------------------------------------

    def _quadrature_capable(self) -> bool:
        field = self.plasma_field
        return hasattr(field, "schedule_levels") and hasattr(field, "surface_data")

    def _kernel(self, key: tuple[Any, ...], build: Callable[[], Any]) -> Any:
        """A compiled callable per plasma field; replacing the field rebuilds it."""
        key = (id(self.plasma_field), id(self.external_field)) + key
        if key not in self._kernels:
            if any(other[:2] != key[:2] for other in self._kernels):
                self._kernels.clear()  # a replaced field: drop its kernels
            self._kernels[key] = build()
        return self._kernels[key]

    def _external(self, order: int, xyz: Array) -> Array:
        if self.external_field is None:
            return jnp.zeros(xyz.shape + (3,) * order, xyz.dtype)
        if order == 0:
            return _field_cartesian(self.external_field, xyz)
        external = self.external_field

        def build():
            function = lambda point: _field_cartesian(external, point[None, :])[0]  # noqa: E731
            for _ in range(order):
                function = jax.jacfwd(function)
            return jax.jit(jax.vmap(function))

        return self._kernel(("external", order), build)(xyz)

    def _direct_sources(self):
        """Nodes and weighted densities of the finest schedule level, compiled once.

        ``level_sources`` builds them eagerly, primitive by primitive, which
        took 17 s on the first call; compiled, 0.6 s.  A traced ``B_total``
        (the parameter-derivative path) is built inside the trace instead.
        """
        field = self.plasma_field

        def sources(B_total):
            view = copy.copy(field)
            view.B_total = B_total
            view._level_source_cache = {}
            return view.level_sources()

        if isinstance(field.B_total, jax.core.Tracer):
            return sources(field.B_total)
        return self._kernel(("sources",), lambda: _concrete(sources, field.B_total))

    def _direct(self, order: int, xyz: Array) -> Array:
        """The direct path: the schedule for ``B``, closed-form kernels beyond."""
        if not self._quadrature_capable():
            function = lambda point: self.plasma_field.B_plasma_xyz(point[None, :])[0]  # noqa: E731
            for _ in range(order):
                function = jax.jacfwd(function)
            return jax.vmap(function)(xyz) if order else self.plasma_field.B_plasma_xyz(xyz)
        if order == 0:
            return self._kernel(("B",), lambda: jax.jit(self.plasma_field.B_plasma_xyz))(xyz)
        from virtual_casing_jax.derivative_kernels import layer_derivatives

        return layer_derivatives(xyz, *self._direct_sources(), order=int(order))[order]

    def _series(self):
        from . import virtual_casing as vc

        surface = self.plasma_field.surface_data
        if isinstance(jnp.asarray(surface.B_total), jax.core.Tracer):
            return vc._graded_series(surface)
        return self._kernel(("series",), lambda: _concrete(vc._graded_series, surface))

    def _graded(self, order: int, xyz: Array, nodes: tuple[int, int] | None = None) -> Array:
        from . import virtual_casing as vc

        nodes = self._graded_nodes if nodes is None else nodes
        series = self._series()
        if isinstance(series["coefficients"], jax.core.Tracer) or isinstance(
                xyz, jax.core.Tracer):
            return vc._graded_field(series, xyz, order, nodes)[order]
        function = self._kernel(("graded", order, nodes), lambda: jax.jit(
            lambda s, p: vc._graded_field(s, p, order, nodes)[order]))
        return function(series, xyz)

    def _graded_estimate(self, order: int, xyz: Array) -> Array:
        """``|graded(N) - graded(3N/4)|`` over the RMS surface ``|B|``, per point.

        The rule converges geometrically in its node count, so the coarser
        evaluation's error dominates the difference and it bounds the finer
        one's from above.
        """
        n_theta, n_phi = self._graded_nodes
        coarse = (max(8, (3 * n_theta) // 4), max(8, (3 * n_phi) // 4))
        difference = self._graded(order, xyz) - self._graded(order, xyz, coarse)
        surface = self.plasma_field.surface_data
        scale = jnp.sqrt(jnp.mean(jnp.sum(jnp.asarray(surface.B_total) ** 2, axis=0)))
        return jnp.max(jnp.abs(difference.reshape(xyz.shape[0], -1)), axis=1) / scale

    def _direct_estimate(self, order: int, xyz: Array) -> Array:
        from . import virtual_casing as vc

        if order:
            return vc.offsurface_error_estimate(self.plasma_field, xyz, order=int(order))
        return self._kernel(("estimate",), lambda: jax.jit(
            lambda p: vc.offsurface_error_estimate(self.plasma_field, p)))(xyz)

    def _field(self, order: int, xyz: Array) -> Array:
        """``order``-th derivative of the total field, choosing the quadrature."""
        value = self._external(order, xyz)
        if self.plasma_field is None:
            return value
        if not self._quadrature_capable():
            return value + self._direct(order, xyz)
        mode = self._near_surface
        traced = isinstance(xyz, jax.core.Tracer) or isinstance(
            jnp.asarray(self.plasma_field.B_total), jax.core.Tracer)
        if mode == "graded":
            plasma = self._graded(order, xyz)
            if not traced and self._accuracy_check != "off":
                self._report(order, np.asarray(self._graded_estimate(order, xyz)),
                             np.ones(xyz.shape[0], bool))
            return value + plasma
        plasma = self._direct(order, xyz)
        if traced or (mode == "direct" and self._accuracy_check == "off"):
            return value + plasma
        estimate = np.array(self._direct_estimate(order, xyz), dtype=float)
        tolerance = 10.0 ** (-int(self.plasma_field.config.digits))
        graded = np.zeros(xyz.shape[0], bool)
        if mode == "auto":
            graded = ~(estimate <= tolerance)
            if graded.any():
                index = np.flatnonzero(graded)
                plasma = jnp.asarray(plasma).at[index].set(self._graded(order, xyz[index]))
                if self._accuracy_check != "off":
                    estimate[index] = np.asarray(self._graded_estimate(order, xyz[index]))
        if self._accuracy_check != "off":
            self._report(order, estimate, graded)
        return value + plasma

    def _report(self, order: int, estimate: np.ndarray, graded: np.ndarray) -> None:
        digits = int(self.plasma_field.config.digits)
        missed = ~(estimate <= 10.0 ** (-digits))
        if not np.any(missed):
            return
        worst = float(np.max(np.where(np.isfinite(estimate), estimate, np.inf)))
        quantity = "field" if not order else f"order-{order} derivative"
        if np.any(missed & graded):
            advice = ("the graded near-surface rule at "
                      f"{self._graded_nodes[0]} x {self._graded_nodes[1]} nodes missed "
                      "too; raise graded_nodes, or move the points off the surface")
        else:
            nt, npol = self.plasma_field.schedule_levels[-1]
            advice = (f"the direct quadrature on the finest source grid ({nt} toroidal x "
                      f"{npol} poloidal points over the full torus) needs targets about "
                      "two grid spacings off the surface; use near_surface='auto' or "
                      "'graded', raise nphi/ntheta or levels, or move the points out")
        message = (
            f"virtual-casing exterior {quantity}: {int(missed.sum())} of {missed.size} "
            f"points have estimated quadrature error up to {worst:.1e}, above the "
            f"requested 1e-{digits}; {advice}.")
        if self._accuracy_check == "raise":
            raise ExteriorFieldAccuracyError(message)
        warnings.warn(message, ExteriorFieldAccuracyWarning, stacklevel=4)

    def B(self, points: Array | None = None) -> Array:
        """Return Cartesian ``B``; eager calls choose and check the quadrature."""
        xyz = self._require_points() if points is None else _check_points(points)
        return self._field(0, xyz)

    def gradB(self, points: Array | None = None) -> Array:
        """Return ``dB_i/dx_j``; eager calls choose and check at first order."""
        xyz = self._require_points() if points is None else _check_points(points)
        return self._field(1, xyz)

    def gradgradB(self, points: Array | None = None) -> Array:
        """Return ``d2B_i/dx_j dx_k``; eager calls choose and check at second order."""
        xyz = self._require_points() if points is None else _check_points(points)
        return self._field(2, xyz)

    def gradgradgradB(self, points: Array | None = None) -> Array:
        """Return ``d3B_i/dx_j dx_k dx_l``; eager calls choose and check at third order."""
        xyz = self._require_points() if points is None else _check_points(points)
        return self._field(3, xyz)

    def B_error_estimate(self, points: Array | None = None, *, order: int = 0) -> Array:
        """Estimated relative error of the plasma field, shape ``(n,)``.

        It estimates the quadrature the same call of :meth:`B` (``order=0``)
        or of the ``order``-th derivative would use, relative to the RMS
        surface ``|B|``; compare it with ``10**-digits``, and ``-log10`` of it
        is the achieved digits.

        For the direct path at ``order=0``: per point, the larger of the
        finest level's double-layer quadrature error and the squared relative
        change between the last two schedule levels
        (:func:`~vmex.core.virtual_casing.offsurface_error_estimate`).  The
        error decays as ``exp(-2 pi d / h)`` with ``d`` the distance to the
        surface and ``h`` the largest spacing of the finest level.  Measured
        against a converged target-graded quadrature of the same data on the
        2.5 % beta QA deck, it is 0.76 to 1.09 times the true error wherever
        that error is above round-off, so it tracks the error rather than
        bounding it.  It is traceable.  At ``order > 0`` the direct path uses
        the a-priori estimate of af Klinteberg, Sorgentone and Tornberg, which
        is host-side and raises under a trace; on the same deck it was never
        below the true error of orders 1 to 3 and above it by 3 to 6 times at
        first order and up to 14 times at third.

        For the graded rule: the difference from the same rule at three
        quarters of its nodes, which bounds its error from above; traceable.
        Under ``near_surface="auto"`` an eager call reports the graded
        estimate at the points the direct path would have missed; a traced
        call reports the direct estimate, as a traced :meth:`B` uses it.
        """
        if self.plasma_field is None:
            raise RuntimeError("the field has no virtual-casing plasma contribution")
        xyz = self._require_points() if points is None else _check_points(points)
        order = int(order)
        if self._near_surface == "graded":
            return self._graded_estimate(order, xyz)
        estimate = self._direct_estimate(order, xyz)
        if self._near_surface == "direct" or isinstance(estimate, jax.core.Tracer):
            return estimate
        estimate = np.array(estimate, dtype=float)
        miss = ~(estimate <= 10.0 ** (-int(self.plasma_field.config.digits)))
        if miss.any():
            index = np.flatnonzero(miss)
            estimate[index] = np.asarray(self._graded_estimate(order, xyz[index]))
        return jnp.asarray(estimate)


    @classmethod
    def from_surface_data(
        cls,
        surface_data: Any,
        *,
        external_field: Any | None = None,
        digits: int = 6,
        levels: tuple[tuple[int, int], ...] | None = None,
        chunk_size: int | str = "auto",
        target_chunk_size: int | str = "auto",
        accuracy_check: AccuracyCheck = "warn",
        near_surface: NearSurface = "auto",
        graded_nodes: tuple[int, int] | None = None,
    ) -> "VmecExtender":
        """Construct the finite-beta path from traceable VMEX surface data.

        ``chunk_size`` bounds source points per virtual-casing batch;
        ``target_chunk_size`` bounds evaluation points. ``"auto"`` delegates
        both memory/performance choices to virtual-casing-jax.

        ``levels`` are full-torus ``(n_toroidal, n_poloidal)`` source grids.
        ``surface_data.gamma`` is sampled on ONE field period, so the default
        schedule carries the ``nfp`` factor:
        ``((nfp nphi, ntheta), (2 nfp nphi, 2 ntheta))``. Without it an nfp = 5
        boundary sampled at 32 points per period was resolved by a finest level
        of 64 over the whole torus -- 13 per period.

        ``near_surface`` and ``graded_nodes`` choose the quadrature near the
        surface; see the class documentation.
        """
        from . import virtual_casing as vc

        vc._require_vcj()
        nphi, ntheta = map(int, surface_data.gamma.shape[1:])
        # ``gamma`` is sampled on ONE field period, while ``levels`` counts the
        # whole torus, so the default schedule has to carry the nfp factor --
        # without it an nfp = 5 boundary sampled at 32 points per period was
        # resolved by a finest level of 64 over the torus, 13 per period.
        nfp = max(int(getattr(surface_data, "nfp", 1) or 1), 1)
        full = nphi * nfp
        schedule = levels or ((full, ntheta), (2 * full, 2 * ntheta))
        config = vc.ExteriorFieldConfig(
            digits=digits,
            src_nphi=nphi,
            src_ntheta=ntheta,
            levels=schedule,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
            branch="internal",
        )
        plasma_field = vc.VirtualCasingExteriorField(surface_data, config)
        return cls(external_field, plasma_field, accuracy_check=accuracy_check,
                   near_surface=near_surface, graded_nodes=graded_nodes)

    @classmethod
    def from_parameterized_surface_data(
        cls,
        surface_data_fn: Callable[[Array], Any],
        parameters: Array,
        *,
        external_field: Any | None = None,
        external_parameters: Array | None = None,
        external_field_from_parameters: Callable[[Array], Any] | None = None,
        external_dof_names: tuple[str, ...] = (),
        digits: int = 6,
        levels: tuple[tuple[int, int], ...] | None = None,
        chunk_size: int | str = "auto",
        target_chunk_size: int | str = "auto",
        dof_names: tuple[str, ...] = (),
        accuracy_check: AccuracyCheck = "warn",
    ) -> "VmecExtender":
        """Construct a virtual-casing field with VJPs in ``parameters``.

        ``surface_data_fn(parameters)`` must return traceable VMEX surface
        data.  Spatial derivatives and parameter VJPs then use the same JAX
        field graph; no finite-difference equilibrium solves are introduced.
        """
        parameters = jnp.ravel(jnp.asarray(parameters))
        parameterized_external = external_parameters is not None
        if parameterized_external != (external_field_from_parameters is not None):
            raise ValueError(
                "external_parameters and external_field_from_parameters "
                "must be provided together")
        if parameterized_external and external_field is not None:
            raise ValueError(
                "external_field is inferred from external_parameters; do not provide both")
        if not parameterized_external and external_dof_names:
            raise ValueError(
                "external_dof_names require external_parameters and "
                "external_field_from_parameters")
        external_parameters = (jnp.ravel(jnp.asarray(external_parameters))
                               if parameterized_external else jnp.empty(0))
        if len(external_dof_names) not in (0, int(external_parameters.size)):
            raise ValueError("external_dof_names must match external_parameters")
        if parameterized_external and not external_dof_names:
            external_dof_names = tuple(
                f"external[{index}]" for index in range(int(external_parameters.size)))
        n_plasma = int(parameters.size)
        all_parameters = jnp.concatenate((parameters, external_parameters))
        external_factory = cast(
            Callable[[Array], Any], external_field_from_parameters
        ) if parameterized_external else None

        def split(p):
            return p[:n_plasma], p[n_plasma:]

        def make_external(external_dofs):
            if external_factory is None:
                return external_field
            return external_factory(external_dofs)

        initial_external_field = make_external(external_parameters)

        initial_surface_data = surface_data_fn(parameters)

        def differentiable_surface_data(p: Array) -> tuple[Array, ...]:
            plasma_parameters, external_dofs = split(p)
            data = surface_data_fn(plasma_parameters)
            return data.gamma, data.B_total, data.normal, data.area_vector, external_dofs

        def B_from_surface_arrays(arrays: tuple[Array, ...], points: Array) -> Array:
            from dataclasses import replace

            gamma, B_total, normal, area_vector, external_dofs = arrays
            data = replace(
                initial_surface_data, gamma=gamma, B_total=B_total,
                normal=normal, area_vector=area_vector)
            live_external_field = make_external(external_dofs)
            # the pullback differentiates the quadrature the field is using
            return cls.from_surface_data(
                data, external_field=live_external_field, digits=digits,
                levels=levels, chunk_size=chunk_size,
                target_chunk_size=target_chunk_size,
                near_surface=field.near_surface,
                graded_nodes=field.graded_nodes).B(points)

        field = cls.from_surface_data(
            initial_surface_data, external_field=initial_external_field,
            digits=digits, levels=levels, chunk_size=chunk_size,
            target_chunk_size=target_chunk_size, accuracy_check=accuracy_check)
        field._parameters = all_parameters
        field._parameter_data_fn = differentiable_surface_data
        field._B_from_data = B_from_surface_arrays
        field.dof_names = tuple(dof_names) + tuple(external_dof_names)
        return field

    @classmethod
    def from_wout(
        cls,
        wout: Any,
        *,
        external_field: Any | None = None,
        plasma: PlasmaMode = "auto",
        nphi: int | None = None,
        ntheta: int | None = None,
        digits: int = 6,
        levels: tuple[tuple[int, int], ...] | None = None,
        chunk_size: int | str = "auto",
        target_chunk_size: int | str = "auto",
        base_dir: str | Path | None = None,
        accuracy_check: AccuracyCheck = "warn",
        project_current: bool = False,
    ) -> "VmecExtender":
        """Construct an exterior field from a wout-like object.

        ``nphi`` and ``ntheta`` default to the per-period source sampling that
        reaches ``digits`` one minor radius off the boundary
        (:func:`_source_nphi_for_digits`).  Both directions matter: on the
        shipped QA wout, the measured achieved error at ``d = a`` against a
        requested 1e-6 is 6.4e-04 at (32, 32), 7.1e-06 at (64, 32) and
        4.3e-07 at (64, 64) -- and (64, 64) is also the cheapest of the three
        per call, so the poloidal count follows the toroidal one.  Pass either
        explicitly to override.

        ``project_current`` keeps only the surface-gradient part of the LCFS
        field before virtual casing
        (:func:`~vmex.core.virtual_casing.surface_field_data_from_wout`), so
        the sheet current is conserved and the exterior plasma field is
        curl-free; it is off by default.
        """
        if plasma not in ("auto", "include", "vacuum"):
            raise ValueError("plasma must be 'auto', 'include', or 'vacuum'")
        chosen = _source_nphi_for_digits(wout, digits)
        if external_field is None:
            external_field = _mgrid_from_wout(
                wout, None if base_dir is None else Path(base_dir)
            )

        include_plasma = plasma == "include" or (
            plasma == "auto" and _has_plasma_sources(wout)
        )
        plasma_field = None
        if include_plasma:
            from . import virtual_casing as vc

            surface = vc.surface_field_data_from_wout(
                wout,
                nphi=chosen if nphi is None else nphi,
                ntheta=chosen if ntheta is None else ntheta,
                project_current=project_current,
            )
            return cls.from_surface_data(
                surface,
                external_field=external_field,
                digits=digits,
                levels=levels,
                chunk_size=chunk_size,
                target_chunk_size=target_chunk_size,
                accuracy_check=accuracy_check,
            )

        if external_field is None and plasma_field is None:
            raise ValueError(
                "a vacuum extension needs an mgrid file or external_field"
            )
        return cls(external_field, plasma_field, accuracy_check=accuracy_check)

    @classmethod
    def from_file(cls, path: str | Path, **kwargs: Any) -> "VmecExtender":
        """Read a wout file and resolve a relative mgrid beside it."""
        from .wout import read_wout

        path = Path(path)
        return cls.from_wout(read_wout(path), base_dir=path.parent, **kwargs)

    @classmethod
    def from_state(
        cls,
        inp: Any,
        state: Any,
        *,
        external_field: Any | None = None,
        nphi: int | None = None,
        ntheta: int | None = None,
        digits: int = 6,
        levels: tuple[tuple[int, int], ...] | None = None,
        chunk_size: int | str = "auto",
        target_chunk_size: int | str = "auto",
        accuracy_check: AccuracyCheck = "warn",
        project_current: bool = False,
    ) -> "VmecExtender":
        """Construct the differentiable finite-beta path from a live VMEX state.

        ``nphi``, ``ntheta`` and ``project_current`` behave exactly as in
        :meth:`from_wout`; pass the grid explicitly to override its default.
        """
        from . import virtual_casing as vc

        chosen = _source_nphi_for_digits(inp, digits)
        surface = vc.surface_field_data_from_state(
            inp, state,
            nphi=chosen if nphi is None else nphi,
            ntheta=chosen if ntheta is None else ntheta,
            project_current=project_current,
        )
        return cls.from_surface_data(
            surface,
            external_field=external_field,
            digits=digits,
            levels=levels,
            chunk_size=chunk_size,
            target_chunk_size=target_chunk_size,
            accuracy_check=accuracy_check,
        )

    @classmethod
    def from_equilibrium(cls, equilibrium: Any, **kwargs: Any) -> "VmecExtender":
        """Construct from an equilibrium, retaining live-state derivatives."""
        external_field = kwargs.pop("external_field", None)
        if external_field is None and bool(equilibrium.inp.lfreeb):
            from .freeboundary import _external_field_from_input

            external_field = _external_field_from_input(equilibrium.inp)

        plasma = kwargs.pop("plasma", "auto")
        if plasma not in ("auto", "include", "vacuum"):
            raise ValueError("plasma must be 'auto', 'include', or 'vacuum'")
        include_plasma = plasma == "include" or (
            plasma == "auto" and _has_plasma_sources(equilibrium.wout)
        )
        if include_plasma:
            return cls.from_state(
                equilibrium.inp,
                equilibrium.state,
                external_field=external_field,
                **kwargs,
            )
        return cls.from_wout(
            equilibrium.wout,
            external_field=external_field,
            plasma="vacuum",
            **kwargs,
        )
