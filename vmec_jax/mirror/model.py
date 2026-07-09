"""Mirror input contracts and differentiable state containers.

The supported open-end model is a finite equilibrium domain between two
fixed, flux-carrying cuts.  These cuts are not periodic and are not
plasma-vacuum interfaces.  The lateral ``s=1`` surface is the fixed or free
plasma boundary.  See ``plan.md`` Phase 5.1-5.2.

Pressure closures must return thermodynamically consistent parallel and
perpendicular moments.  In particular, production closures enforce
``p_perp = p_parallel - B * d(p_parallel)/dB`` at fixed ``s``; accepting two
unrelated pressure arrays would violate parallel force balance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp

MIRROR_INPUT_SCHEMA = "vmec_jax.mirror.input/1"
MIRROR_OUTPUT_SCHEMA = "vmec_jax.mirror.mout/1"

Array = Any


class EndCondition(str, Enum):
    """Supported axial boundary policies.

    ``FIXED_FLUX_CUT`` fixes geometry and normal flux at both axial cuts while
    allowing magnetic field lines to cross them.  End loss, sheath, sources,
    and transport are outside the equilibrium model.
    """

    FIXED_FLUX_CUT = "fixed_flux_cut"


@dataclass(frozen=True)
class MirrorResolution:
    """Static resolution for ``(s, theta, xi)`` mirror coordinates.

    ``mpol`` is the largest retained theta Fourier mode.  Axisymmetry uses
    ``mpol=0, ntheta=1``.  Three-dimensional grids require at least
    ``2*mpol+1`` points so the highest represented mode is not a Nyquist mode.
    """

    ns: int = 17
    mpol: int = 0
    ntheta: int = 1
    nxi: int = 33

    def __post_init__(self) -> None:
        if self.ns < 3:
            raise ValueError("mirror ns must be >= 3 for second-order radial differences")
        if self.mpol < 0:
            raise ValueError("mirror mpol must be >= 0")
        if self.nxi < 2:
            raise ValueError("mirror nxi must be >= 2")
        minimum_theta = 1 if self.mpol == 0 else 2 * self.mpol + 1
        if self.ntheta < minimum_theta:
            raise ValueError(
                f"ntheta={self.ntheta} cannot resolve mpol={self.mpol}; "
                f"use ntheta >= {minimum_theta}"
            )
        if self.mpol == 0 and self.ntheta != 1:
            raise ValueError("axisymmetric mirror resolution uses mpol=0 and ntheta=1")

    @property
    def axisymmetric(self) -> bool:
        """Whether theta dependence is absent."""

        return self.mpol == 0


@dataclass(frozen=True)
class MirrorConfig:
    """Numerical and boundary contract for a mirror equilibrium.

    The default nonlinear tolerance is the requested component-wise physical
    force tolerance.  It is not an optimizer objective tolerance.
    """

    resolution: MirrorResolution = MirrorResolution()
    z_min: float = -1.0
    z_max: float = 1.0
    end_condition: EndCondition = EndCondition.FIXED_FLUX_CUT
    ftol: float = 1.0e-12
    max_iterations: int = 2000

    def __post_init__(self) -> None:
        try:
            end_condition = EndCondition(self.end_condition)
        except ValueError as error:
            raise ValueError(f"unsupported mirror end condition: {self.end_condition}") from error
        object.__setattr__(self, "end_condition", end_condition)
        if not self.z_max > self.z_min:
            raise ValueError("z_max must be greater than z_min")
        if not self.ftol > 0.0:
            raise ValueError("mirror ftol must be positive")
        if self.max_iterations < 1:
            raise ValueError("mirror max_iterations must be >= 1")

    def build_grid(self) -> "MirrorGrid":
        """Build immutable collocation and quadrature data."""

        from .basis import build_mirror_grid

        return build_mirror_grid(self)


@dataclass(frozen=True)
class MirrorBoundary:
    """Lateral boundary scale ``a(theta, xi)`` in ``r=sqrt(s)*a``.

    ``radius_scale`` has shape ``(ntheta, nxi)``.  It is a differentiable JAX
    leaf so fixed-boundary shape derivatives do not require another boundary
    representation.
    """

    radius_scale: Array

    @classmethod
    def from_radius(cls, radius: Array, grid: "MirrorGrid") -> "MirrorBoundary":
        """Broadcast scalar, axial, or full theta-axial radii to the grid."""

        value = jnp.asarray(radius)
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            value = value.astype(jnp.asarray(1.0).dtype)
        if value.ndim == 0:
            value = jnp.broadcast_to(value, (grid.ntheta, grid.nxi))
        elif value.shape == (grid.nxi,):
            value = jnp.broadcast_to(value[None, :], (grid.ntheta, grid.nxi))
        elif value.shape != (grid.ntheta, grid.nxi):
            raise ValueError(
                f"boundary radius shape {value.shape} must be scalar, "
                f"({grid.nxi},), or ({grid.ntheta}, {grid.nxi})"
            )
        return cls(radius_scale=value)

    @classmethod
    def from_axis_field(
        cls,
        axial_flux_derivative: Array,
        on_axis_bz: Array,
        grid: "MirrorGrid",
        *,
        radius_floor: float = 0.0,
    ) -> "MirrorBoundary":
        """Build the leading-order flux tube ``a=sqrt(2*Psi'/|Bz|)``.

        This paraxial relation is an initializer and analytic validation
        fixture, not a replacement for a finite-radius equilibrium solve.
        """

        bz = jnp.asarray(on_axis_bz)
        if bz.shape != (grid.nxi,):
            raise ValueError(f"on_axis_bz shape {bz.shape} must be ({grid.nxi},)")
        flux = jnp.asarray(axial_flux_derivative, dtype=bz.dtype)
        if flux.ndim != 0:
            raise ValueError("flux-tube boundary requires a scalar axial_flux_derivative")
        tiny = jnp.finfo(bz.dtype).tiny
        radius = jnp.sqrt(2.0 * flux / jnp.maximum(jnp.abs(bz), tiny))
        radius = jnp.maximum(radius, jnp.asarray(radius_floor, dtype=bz.dtype))
        return cls.from_radius(radius, grid)


@dataclass(frozen=True)
class MirrorState:
    """Differentiable mirror geometry and field-line state.

    Both arrays have shape ``(ns, ntheta, nxi)``.  ``radius_scale`` defines
    ``r=sqrt(s)*radius_scale``; storing the regular scale rather than ``r``
    avoids evolving a singular radial derivative at the magnetic axis.
    ``lambda_stream`` is the divergence-free field stream function and uses a
    zero surface-average gauge in the solver lane.
    """

    radius_scale: Array
    lambda_stream: Array

    @classmethod
    def from_boundary(cls, boundary: MirrorBoundary, grid: "MirrorGrid") -> "MirrorState":
        """Construct the radial self-similar initial state for a boundary."""

        boundary_radius = jnp.asarray(boundary.radius_scale)
        expected = (grid.ntheta, grid.nxi)
        if boundary_radius.shape != expected:
            raise ValueError(f"boundary shape {boundary_radius.shape} does not match {expected}")
        shape = (grid.ns, grid.ntheta, grid.nxi)
        return cls(
            radius_scale=jnp.broadcast_to(boundary_radius[None, :, :], shape),
            lambda_stream=jnp.zeros(shape, dtype=boundary_radius.dtype),
        )

    def validate_shape(self, grid: "MirrorGrid") -> None:
        """Raise when state arrays do not match the static grid."""

        expected = (grid.ns, grid.ntheta, grid.nxi)
        if self.radius_scale.shape != expected:
            raise ValueError(f"radius_scale shape {self.radius_scale.shape} does not match {expected}")
        if self.lambda_stream.shape != expected:
            raise ValueError(f"lambda_stream shape {self.lambda_stream.shape} does not match {expected}")


def project_fixed_boundary_state(
    state: MirrorState,
    boundary: MirrorBoundary,
    grid: "MirrorGrid",
) -> MirrorState:
    """Apply fixed side/end geometry, axis regularity, and lambda gauge.

    End geometry is radial-self-similar at both fixed-flux cuts.  The lambda
    surface mean is removed with tensor-product theta/CGL quadrature; this is
    a pure gauge operation and does not change ``B``.
    """

    state.validate_shape(grid)
    boundary_radius = jnp.asarray(boundary.radius_scale)
    if boundary_radius.shape != (grid.ntheta, grid.nxi):
        raise ValueError("boundary shape does not match mirror grid")
    radius_scale = jnp.asarray(state.radius_scale)
    radius_scale = radius_scale.at[-1].set(boundary_radius)
    radius_scale = radius_scale.at[:, :, 0].set(boundary_radius[:, 0][None, :])
    radius_scale = radius_scale.at[:, :, -1].set(boundary_radius[:, -1][None, :])
    radius_scale = radius_scale.at[0].set(radius_scale[1])

    lam = jnp.asarray(state.lambda_stream)
    theta_weights = jnp.asarray(grid.theta_basis.weights)
    xi_weights = jnp.asarray(grid.axial_basis.weights)
    denominator = jnp.sum(theta_weights) * jnp.sum(xi_weights)
    surface_mean = jnp.einsum("j,k,ijk->i", theta_weights, xi_weights, lam) / denominator
    lam = lam - surface_mean[:, None, None]
    return MirrorState(radius_scale=radius_scale, lambda_stream=lam)


@dataclass(frozen=True)
class PressureMoments:
    """Closure output sampled on the equilibrium grid."""

    parallel: Array
    perpendicular: Array
    energy_density: Array


@runtime_checkable
class PressureClosure(Protocol):
    """Protocol for isotropic, bi-Maxwellian, or tabulated closures."""

    def moments(self, s: Array, magnetic_field_strength: Array) -> PressureMoments:
        """Return consistent pressure moments and generating energy density."""


jax.tree_util.register_dataclass(MirrorBoundary, data_fields=["radius_scale"], meta_fields=[])
jax.tree_util.register_dataclass(
    MirrorState,
    data_fields=["radius_scale", "lambda_stream"],
    meta_fields=[],
)
jax.tree_util.register_dataclass(
    PressureMoments,
    data_fields=["parallel", "perpendicular", "energy_density"],
    meta_fields=[],
)


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .basis import MirrorGrid
