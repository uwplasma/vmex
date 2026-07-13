"""Differentiable scalar and profile quantities from a spectral state.

These observables share the traceable geometry-to-field pipeline used by the
implicit solver, but they do not participate in its root or adjoint
orchestration.  Keeping them here makes that separation explicit and lets
optimization code reuse one compact implementation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import jax
import jax.numpy as jnp

from .solver import SolverRuntime, SpectralState, _physical_coefficients
from .statephysics import _field_chain as _field_chain_shared
from .transforms import physical_to_internal_scale

Array = Any

__all__ = [
    "aspect_ratio",
    "iota_axis",
    "iota_edge",
    "iota_profile",
    "mhd_energy",
    "plasma_volume",
]


# Reusing one compiled field chain keeps grad/jacrev compilation bounded.
_field_chain = jax.jit(_field_chain_shared)


def mhd_energy(state: SpectralState, rt: SolverRuntime) -> tuple[Array, Array]:
    """Return magnetic and pressure energy in WOUT normalization."""
    _, _, _, _, energies = _field_chain(state, rt)
    return energies.wb, energies.wp


def plasma_volume(state: SpectralState, rt: SolverRuntime) -> Array:
    """Return plasma volume in cubic metres."""
    _, _, _, _, energies = _field_chain(state, rt)
    return (2.0 * jnp.pi) ** 2 * jnp.abs(energies.volume)


def _edge_physical(state: SpectralState, rt: SolverRuntime):
    """Return physical WOUT-convention edge coefficients."""
    R_cos, R_sin, Z_cos, Z_sin = _physical_coefficients(
        state,
        modes=rt.modes,
        lthreed=rt.setup.lthreed,
        lasym=rt.setup.lasym,
        lconm1=rt.setup.lconm1,
    )
    scale = jnp.asarray(1.0 / physical_to_internal_scale(rt.modes, rt.trig))
    return (
        R_cos[-1] * scale,
        R_sin[-1] * scale,
        Z_cos[-1] * scale,
        Z_sin[-1] * scale,
    )


def aspect_ratio(
    state: SpectralState,
    rt: SolverRuntime,
    *,
    ntheta: int = 128,
    nzeta: int = 32,
) -> Array:
    """Return VMEC's differentiable ``Rmajor_p / Aminor_p`` aspect ratio."""
    rmnc, rmns, zmnc, zmns = _edge_physical(state, rt)
    m = jnp.asarray(np.asarray(rt.modes.m, dtype=float))
    n = jnp.asarray(np.asarray(rt.modes.n, dtype=float) * rt.resolution.nfp)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, ntheta, endpoint=False)
    zeta = jnp.linspace(
        0.0, 2.0 * jnp.pi / rt.resolution.nfp, nzeta, endpoint=False
    )
    angle = (
        m[:, None, None] * theta[None, :, None]
        - n[:, None, None] * zeta[None, None, :]
    )
    cosine, sine = jnp.cos(angle), jnp.sin(angle)
    Z = jnp.einsum("k,ktz->tz", zmns, sine) + jnp.einsum(
        "k,ktz->tz", zmnc, cosine
    )
    dR_dtheta = -jnp.einsum("k,ktz->tz", m * rmnc, sine) + jnp.einsum(
        "k,ktz->tz", m * rmns, cosine
    )
    area = jnp.abs(
        jnp.mean(jnp.sum(-Z * dR_dtheta, axis=0) * (2.0 * jnp.pi / ntheta))
    )
    aminor = jnp.sqrt(area / jnp.pi)
    rmajor = plasma_volume(state, rt) / (2.0 * jnp.pi**2 * aminor**2)
    return rmajor / aminor


def iota_profile(state: SpectralState, rt: SolverRuntime) -> Array:
    """Return differentiable full-mesh rotational transform ``iotaf``."""
    setup = rt.setup
    if int(setup.ncurr) != 1:
        return jnp.asarray(setup.iotaf)
    _, _, _, fields, _ = _field_chain(state, rt)
    phips = jnp.asarray(setup.phips)
    safe = jnp.where(phips != 0.0, phips, 1.0)
    iotas = jnp.where(phips != 0.0, fields.chips / safe, 0.0)
    iotaf = 0.5 * (iotas + jnp.roll(iotas, -1))
    iotaf = iotaf.at[0].set(1.5 * iotas[1] - 0.5 * iotas[2])
    return iotaf.at[-1].set(1.5 * iotas[-1] - 0.5 * iotas[-2])


def iota_axis(state: SpectralState, rt: SolverRuntime) -> Array:
    """Return rotational transform on the magnetic axis."""
    return iota_profile(state, rt)[0]


def iota_edge(state: SpectralState, rt: SolverRuntime) -> Array:
    """Return rotational transform on the last closed flux surface."""
    return iota_profile(state, rt)[-1]
