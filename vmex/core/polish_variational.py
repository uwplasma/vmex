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
from .strong_force import HighOrderEquilibriumState

Array = Any


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class VariationalPlan:
    """Precomputed coefficient-to-jet tables and quadrature weights."""

    rho: Array
    theta: Array
    zeta: Array
    radial_value: Array
    radial_derivative: Array
    profile_basis: Array
    cosine: Array
    sine: Array
    cosine_theta: Array
    sine_theta: Array
    cosine_zeta: Array
    sine_zeta: Array
    quadrature_weights: Array
    nfp: int
    jacobian_sign: int

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
                "profile_basis",
                "cosine",
                "sine",
                "cosine_theta",
                "sine_theta",
                "cosine_zeta",
                "sine_zeta",
                "quadrature_weights",
            )
        )
        return children, (int(self.nfp), int(self.jacobian_sign))

    @classmethod
    def tree_unflatten(cls, metadata, children):
        """Rebuild a plan from its JAX pytree representation."""

        nfp, jacobian_sign = metadata
        return cls(*children, nfp=nfp, jacobian_sign=jacobian_sign)

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
    """Build a tensor grid and analytic coefficient-to-first-jet tables."""

    degree = int(state.radial_basis.degree)
    order = degree + 3 if radial_order is None else int(radial_order)
    if order < 2:
        raise ValueError("radial_order must be at least two")
    s, weights_s = _span_quadrature(
        np.asarray(state.radial_basis.breakpoints, dtype=float), order
    )
    rho = np.sqrt(s)
    # The radial quadrature is constructed in s, while sqrt_g is the
    # Jacobian of (rho, theta, zeta): d rho = ds / (2 rho).
    weights_rho = weights_s / (2.0 * rho)
    basis = np.asarray(state.radial_basis.basis_matrix(s), dtype=float)
    basis_s = np.asarray(
        state.radial_basis.basis_matrix(s, derivative=1), dtype=float
    )
    m = np.abs(np.asarray(state.m, dtype=int))
    powers = rho[None, :, None] ** m[:, None, None]
    radial_value = powers * basis[None]
    leading = np.zeros_like(radial_value)
    nonzero = m > 0
    leading[nonzero] = (
        m[nonzero, None, None]
        * rho[None, :, None] ** (m[nonzero, None, None] - 1)
        * basis[None]
    )
    radial_derivative = leading + (
        2.0 * rho[None, :, None] ** (m[:, None, None] + 1) * basis_s[None]
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
    angular_weight = (2.0 * np.pi / ntheta) * (2.0 * np.pi / nzeta)
    # zeta spans one field period and sqrt_g contains dphi/dzeta=1/nfp;
    # multiplying by nfp integrates the full torus.
    quadrature_weights = (
        float(state.nfp) * weights_rho[:, None, None] * angular_weight
    )
    return VariationalPlan(
        rho=jnp.asarray(rho),
        theta=jnp.asarray(theta),
        zeta=jnp.asarray(zeta),
        radial_value=jnp.asarray(radial_value),
        radial_derivative=jnp.asarray(radial_derivative),
        profile_basis=jnp.asarray(basis),
        cosine=jnp.asarray(cosine),
        sine=jnp.asarray(sine),
        cosine_theta=jnp.asarray(cosine_theta),
        sine_theta=jnp.asarray(sine_theta),
        cosine_zeta=jnp.asarray(cosine_zeta),
        sine_zeta=jnp.asarray(sine_zeta),
        quadrature_weights=jnp.asarray(quadrature_weights),
        nfp=int(state.nfp),
        jacobian_sign=int(state.jacobian_sign),
    )


def _channel(
    cosine_coefficients: Array,
    sine_coefficients: Array,
    plan: VariationalPlan,
) -> tuple[Array, Array, Array, Array]:
    """Synthesize one scalar channel and its three coordinate derivatives."""

    cosine_coefficients = jnp.asarray(cosine_coefficients)
    sine_coefficients = jnp.asarray(sine_coefficients)
    cosine_radial = jnp.einsum(
        "mb,mrb->rm", cosine_coefficients, plan.radial_value
    )
    sine_radial = jnp.einsum(
        "mb,mrb->rm", sine_coefficients, plan.radial_value
    )
    cosine_drho = jnp.einsum(
        "mb,mrb->rm", cosine_coefficients, plan.radial_derivative
    )
    sine_drho = jnp.einsum(
        "mb,mrb->rm", sine_coefficients, plan.radial_derivative
    )

    def synthesize(cosine_table: Array, sine_table: Array) -> Array:
        return jnp.einsum("rm,ma->ra", cosine_radial, cosine_table) + jnp.einsum(
            "rm,ma->ra", sine_radial, sine_table
        )

    value = synthesize(plan.cosine, plan.sine)
    drho = jnp.einsum("rm,ma->ra", cosine_drho, plan.cosine) + jnp.einsum(
        "rm,ma->ra", sine_drho, plan.sine
    )
    dtheta = synthesize(plan.cosine_theta, plan.sine_theta)
    dzeta = synthesize(plan.cosine_zeta, plan.sine_zeta)
    return value, drho, dtheta, dzeta


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
        return jnp.stack(
            (radial * cosine_phi, radial * sine_phi, vertical), axis=-1
        )

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
    B_theta = flux_factor * (
        chipf[:, None] / float(plan.nfp) - phipf[:, None] * lambda_zeta
    )
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
def fixed_pressure_energy(
    state: HighOrderEquilibriumState,
    plan: VariationalPlan,
    energy_scale: Array = 1.0,
) -> Array:
    """Return the dimensionless fixed-pressure functional ``W/E_ref``.

    The signed Jacobian is used inside the declared fixed-sign admissible
    domain.  Call :func:`minimum_signed_jacobian` before accepting a trial;
    clipping or taking an absolute value here would change the variation.
    """

    fields = evaluate_variational_fields(state, plan)
    signed_jacobian = float(plan.jacobian_sign) * fields.sqrt_g
    density = jnp.sum(fields.B * fields.B, axis=-1) / (2.0 * MU0) - fields.pressure
    return jnp.sum(
        jnp.asarray(plan.quadrature_weights) * signed_jacobian * density
    ) / jnp.asarray(energy_scale)


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
    "VariationalPlan",
    "evaluate_variational_fields",
    "fixed_pressure_energy",
    "make_variational_plan",
    "minimum_signed_jacobian",
]
