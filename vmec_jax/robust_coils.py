"""Pure-JAX utilities for deterministic robust-coil perturbation studies.

These helpers operate on direct-coil ``CoilFieldParams`` pytrees without
depending on ESSOS or the free-boundary solver.  Full VMEC free-boundary solves
are not yet guaranteed to be batch-transformable, so callers can use these
samples with ``jax.vmap`` for transformable objectives or a Python loop around
non-transformable solver calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vmec_jax._compat import jax, jnp, tree_util
from vmec_jax.external_fields.coils_jax import CoilFieldParams


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CoilPerturbationSample:
    """A pytree of perturbations applied to ``CoilFieldParams``.

    ``current_factors`` multiply base currents, ``centerline_dof_delta`` is
    added to Fourier centerline coefficients, ``toroidal_phase`` rotates the
    centerline around the z axis in radians, and ``displacement_xyz`` translates
    the centerline rigidly in Cartesian coordinates.
    """

    current_factors: Any
    displacement_xyz: Any
    toroidal_phase: Any
    centerline_dof_delta: Any

    def tree_flatten(self):
        return (
            self.current_factors,
            self.displacement_xyz,
            self.toroidal_phase,
            self.centerline_dof_delta,
        ), None

    @classmethod
    def tree_unflatten(cls, _aux, children):
        current_factors, displacement_xyz, toroidal_phase, centerline_dof_delta = children
        return cls(
            current_factors=current_factors,
            displacement_xyz=displacement_xyz,
            toroidal_phase=toroidal_phase,
            centerline_dof_delta=centerline_dof_delta,
        )


def identity_coil_perturbation(params: CoilFieldParams) -> CoilPerturbationSample:
    """Return a perturbation sample that leaves ``params`` unchanged."""

    currents = jnp.asarray(params.base_currents)
    dofs = jnp.asarray(params.base_curve_dofs)
    return CoilPerturbationSample(
        current_factors=jnp.ones_like(currents),
        displacement_xyz=jnp.zeros((3,), dtype=dofs.dtype),
        toroidal_phase=jnp.asarray(0.0, dtype=dofs.dtype),
        centerline_dof_delta=jnp.zeros_like(dofs),
    )


def perturb_currents(base_currents: Any, current_factors: Any) -> Any:
    """Apply deterministic multiplicative current perturbations."""

    return jnp.asarray(base_currents) * jnp.asarray(current_factors)


def displace_curve_dofs(base_curve_dofs: Any, displacement_xyz: Any) -> Any:
    """Rigidly translate Fourier centerline coefficients by ``displacement_xyz``."""

    dofs = jnp.asarray(base_curve_dofs)
    displacement = jnp.asarray(displacement_xyz, dtype=dofs.dtype)
    constant_mask = (jnp.arange(dofs.shape[-1]) == 0).astype(dofs.dtype)
    return dofs + displacement[None, :, None] * constant_mask[None, None, :]


def rotate_curve_dofs_about_z(base_curve_dofs: Any, toroidal_phase: Any) -> Any:
    """Rotate all Cartesian Fourier coefficient vectors around the z axis."""

    dofs = jnp.asarray(base_curve_dofs)
    phase = jnp.asarray(toroidal_phase, dtype=dofs.dtype)
    c = jnp.cos(phase)
    s = jnp.sin(phase)
    x = c * dofs[:, 0, :] - s * dofs[:, 1, :]
    y = s * dofs[:, 0, :] + c * dofs[:, 1, :]
    z = dofs[:, 2, :]
    return jnp.stack((x, y, z), axis=1)


def perturb_centerline_dofs(base_curve_dofs: Any, centerline_dof_delta: Any) -> Any:
    """Add a deterministic Fourier-coefficient perturbation to coil centerlines."""

    return jnp.asarray(base_curve_dofs) + jnp.asarray(centerline_dof_delta)


def perturb_coil_params(params: CoilFieldParams, sample: CoilPerturbationSample) -> CoilFieldParams:
    """Return perturbed direct-coil parameters without mutating ``params``.

    Perturbations are applied in a fixed order: additive centerline Fourier
    perturbation, toroidal rotation about z, rigid displacement, then current
    scaling.
    """

    dofs = perturb_centerline_dofs(params.base_curve_dofs, sample.centerline_dof_delta)
    dofs = rotate_curve_dofs_about_z(dofs, sample.toroidal_phase)
    dofs = displace_curve_dofs(dofs, sample.displacement_xyz)
    currents = perturb_currents(params.base_currents, sample.current_factors)
    return params.with_arrays(base_curve_dofs=dofs, base_currents=currents)


def _require_jax_random() -> None:
    if jax is None:  # pragma: no cover - vmec_jax declares JAX as a dependency.
        raise RuntimeError("JAX is required to sample stochastic coil perturbations.")


def _centerline_noise_mask(base_curve_dofs: Any, include_constant: bool) -> Any:
    dofs = jnp.asarray(base_curve_dofs)
    if include_constant:
        return jnp.ones_like(dofs)
    harmonic_mask = (jnp.arange(dofs.shape[-1]) != 0).astype(dofs.dtype)
    return jnp.ones_like(dofs) * harmonic_mask[None, None, :]


def sample_coil_perturbation(
    key: Any,
    params: CoilFieldParams,
    *,
    current_sigma: float = 0.0,
    displacement_sigma: float = 0.0,
    toroidal_phase_sigma: float = 0.0,
    centerline_sigma: float = 0.0,
    centerline_include_constant: bool = False,
) -> CoilPerturbationSample:
    """Draw one deterministic Gaussian perturbation sample from a PRNG key.

    ``current_sigma`` is fractional, so sampled current factors are
    ``1 + current_sigma * normal``.  Geometry sigmas are in the same coordinate
    units as the coil Fourier coefficients, and ``toroidal_phase_sigma`` is in
    radians.
    """

    samples = sample_coil_perturbations(
        key,
        params,
        1,
        current_sigma=current_sigma,
        displacement_sigma=displacement_sigma,
        toroidal_phase_sigma=toroidal_phase_sigma,
        centerline_sigma=centerline_sigma,
        centerline_include_constant=centerline_include_constant,
    )
    return CoilPerturbationSample(
        current_factors=samples.current_factors[0],
        displacement_xyz=samples.displacement_xyz[0],
        toroidal_phase=samples.toroidal_phase[0],
        centerline_dof_delta=samples.centerline_dof_delta[0],
    )


def sample_coil_perturbations(
    key: Any,
    params: CoilFieldParams,
    n_samples: int,
    *,
    current_sigma: float = 0.0,
    displacement_sigma: float = 0.0,
    toroidal_phase_sigma: float = 0.0,
    centerline_sigma: float = 0.0,
    centerline_include_constant: bool = False,
) -> CoilPerturbationSample:
    """Draw a batch of deterministic Gaussian coil perturbation samples."""

    _require_jax_random()
    n = int(n_samples)
    if n < 0:
        raise ValueError("n_samples must be non-negative.")

    currents = jnp.asarray(params.base_currents)
    dofs = jnp.asarray(params.base_curve_dofs)
    current_key, displacement_key, phase_key, centerline_key = jax.random.split(key, 4)

    current_factors = 1.0 + float(current_sigma) * jax.random.normal(
        current_key,
        (n,) + currents.shape,
        dtype=currents.dtype,
    )
    displacement_xyz = float(displacement_sigma) * jax.random.normal(
        displacement_key,
        (n, 3),
        dtype=dofs.dtype,
    )
    toroidal_phase = float(toroidal_phase_sigma) * jax.random.normal(
        phase_key,
        (n,),
        dtype=dofs.dtype,
    )
    centerline_dof_delta = float(centerline_sigma) * jax.random.normal(
        centerline_key,
        (n,) + dofs.shape,
        dtype=dofs.dtype,
    )
    centerline_dof_delta = centerline_dof_delta * _centerline_noise_mask(
        dofs,
        include_constant=centerline_include_constant,
    )[None, ...]

    return CoilPerturbationSample(
        current_factors=current_factors,
        displacement_xyz=displacement_xyz,
        toroidal_phase=toroidal_phase,
        centerline_dof_delta=centerline_dof_delta,
    )


def _positive_float(value: float, name: str) -> float:
    value = float(value)
    if not value > 0.0:
        raise ValueError(f"{name} must be positive.")
    return value


def aggregate_risk(
    values: Any,
    method: str = "mean",
    *,
    axis: int | tuple[int, ...] | None = None,
    std_weight: float = 1.0,
    temperature: float = 1.0,
    tail_fraction: float = 0.2,
) -> Any:
    """Aggregate scenario losses with smooth robust-risk choices.

    Supported methods are ``"mean"``, ``"mean_plus_std"``, ``"smooth_max"``,
    ``"soft_cvar"``, and ``"smooth_tail"``.  ``soft_cvar``/``smooth_tail`` use
    a stopped-gradient empirical tail threshold plus a softplus excess, giving a
    smooth upper-tail penalty with stable gradients for optimization.
    """

    x = jnp.asarray(values)
    risk = method.lower()

    if risk == "mean":
        return jnp.mean(x, axis=axis)

    if risk == "mean_plus_std":
        mean = jnp.mean(x, axis=axis, keepdims=True)
        variance = jnp.mean((x - mean) ** 2, axis=axis, keepdims=True)
        value = mean + float(std_weight) * jnp.sqrt(jnp.maximum(variance, 0.0))
        return jnp.squeeze(value, axis=axis)

    tau = _positive_float(temperature, "temperature")
    if risk == "smooth_max":
        if jax is None:  # pragma: no cover - dependency fallback.
            return tau * jnp.log(jnp.sum(jnp.exp(x / tau), axis=axis))
        return tau * jax.nn.logsumexp(x / tau, axis=axis)

    if risk in ("soft_cvar", "smooth_tail"):
        fraction = float(tail_fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("tail_fraction must be in (0, 1].")
        alpha = 1.0 - fraction
        threshold = jnp.quantile(x, alpha, axis=axis, keepdims=True)
        if jax is not None:
            threshold = jax.lax.stop_gradient(threshold)
            excess = tau * jax.nn.softplus((x - threshold) / tau)
        else:  # pragma: no cover - dependency fallback.
            excess = tau * jnp.log1p(jnp.exp((x - threshold) / tau))
        tail = threshold + jnp.mean(excess, axis=axis, keepdims=True) / fraction
        return jnp.squeeze(tail, axis=axis)

    raise ValueError(
        "method must be one of 'mean', 'mean_plus_std', 'smooth_max', "
        "'soft_cvar', or 'smooth_tail'."
    )
