"""Shared state-physics primitives of the derived-quantity/objective modules.

One home (R26a consolidation) for the small private helpers that
:mod:`~vmec_jax.core.optimize`, :mod:`~vmec_jax.core.implicit`,
:mod:`~vmec_jax.core.bootstrap`, :mod:`~vmec_jax.core.stability` and
:mod:`~vmec_jax.core.omnigenity` all need — formerly byte-identical copies in
``optimize.py``/``implicit.py`` plus re-inlined recipes in ``stability.py``
(the shared module the bootstrap spec section 6.1b called ``_state_diag.py``):

- :func:`_field_chain` — geometry -> Jacobian -> metric -> fields -> energies
  of a core ``(SpectralState, SolverRuntime)`` pair, the evaluation chain
  behind every solver-native scalar target;
- :func:`_iotas_half` / :func:`_iotas_half_from_fields` — the ``ncurr``-aware
  half-mesh rotational transform (``add_fluxes.f90`` conventions);
- the half-mesh radial sampling primitives :func:`_half_grid` /
  :func:`_interp_half_grid` and the wout-table utilities :func:`_as_1d` /
  :func:`_mode_matrix`.

This module sits directly above :mod:`vmec_jax.core.solver` in the import
graph; it must not import the objective modules.  ``optimize`` re-exports
these names for backward compatibility.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import jax.numpy as jnp

from .fields import energies_and_force_norms, magnetic_fields, metric_elements
from .geometry import half_mesh_jacobian
from .solver import SolverRuntime, SpectralState, _geometry

Array = Any


# ---------------------------------------------------------------------------
# Small array / wout-table utilities
# ---------------------------------------------------------------------------


def _as_1d(values, dtype=np.float64) -> jnp.ndarray:
    try:
        seq = list(values)  # type: ignore[arg-type]
    except TypeError:
        seq = [values]
    return jnp.asarray(np.asarray(seq, dtype=dtype))


def _half_grid(ns: int, dtype) -> jnp.ndarray:
    s_full = jnp.linspace(0.0, 1.0, ns, dtype=dtype)
    return 0.5 * (s_full[:-1] + s_full[1:])


def _interp_half_grid(samples: jnp.ndarray, surfaces: jnp.ndarray, s_half: jnp.ndarray) -> jnp.ndarray:
    """Linear interpolation of half-mesh radial samples onto ``surfaces``."""
    if int(s_half.shape[0]) == 1:
        return jnp.broadcast_to(samples[:1], (surfaces.shape[0],) + samples.shape[1:])
    idx_hi = jnp.clip(jnp.searchsorted(s_half, surfaces, side="left"), 1, s_half.shape[0] - 1)
    idx_lo = idx_hi - 1
    x0, x1 = s_half[idx_lo], s_half[idx_hi]
    denom = jnp.where(x1 != x0, x1 - x0, jnp.ones_like(x1))
    t = ((surfaces - x0) / denom).reshape((surfaces.shape[0],) + (1,) * (samples.ndim - 1))
    return samples[idx_lo] + t * (samples[idx_hi] - samples[idx_lo])


def _mode_matrix(wout, name: str, *, ns: int, mn: int, optional: bool = False) -> jnp.ndarray:
    """A ``(ns, mn)`` coefficient table from a wout-like object (either layout)."""
    value = getattr(wout, name, None)
    if value is None:
        if optional:
            return jnp.zeros((ns, mn), dtype=jnp.float64)
        raise AttributeError(f"wout-like object lacks required table {name!r}")
    arr = jnp.asarray(np.ascontiguousarray(np.asarray(value, dtype=np.float64)))
    if arr.shape == (ns, mn):
        return arr
    if arr.shape == (mn, ns):
        return arr.T
    raise ValueError(f"{name}: unexpected shape {arr.shape}, expected {(ns, mn)}")


# ---------------------------------------------------------------------------
# The state -> field-state evaluation chain and its derived profiles
# ---------------------------------------------------------------------------


def _field_chain(state: SpectralState, rt: SolverRuntime):
    """Geometry -> Jacobian -> metric -> fields -> energies of a core state."""
    setup = rt.setup
    s = setup.s_full
    _, geometry = _geometry(state, rt)
    jacobian = half_mesh_jacobian(geometry, s=s)
    metrics = metric_elements(geometry, s=s)
    fields = magnetic_fields(
        geometry=geometry, jacobian=jacobian, metrics=metrics, trig=rt.trig,
        s=s, phips=setup.phips, phipf=setup.phipf, chips=setup.chips,
        signgs=setup.signgs, gamma=rt.gamma, mass=setup.mass,
        ncurr=setup.ncurr, enclosed_current=setup.icurv,
    )
    energies = energies_and_force_norms(
        jacobian=jacobian, metrics=metrics, fields=fields, trig=rt.trig,
        s=s, signgs=setup.signgs,
    )
    return geometry, jacobian, metrics, fields, energies


def _iotas_half_from_fields(setup, fields) -> jnp.ndarray:
    """Half-mesh iota from an already-evaluated field state (``add_fluxes.f90``).

    ``ncurr = 0``: the prescribed profile; ``ncurr = 1``: reconstructed from
    the current-constrained ``chips`` of the field state (differentiable),
    exactly as the solver/wout writer do.  Index 0 is the (zeroed) axis slot.
    """
    if int(setup.ncurr) != 1:
        return jnp.asarray(setup.iotas)
    phips = jnp.asarray(setup.phips)
    safe = jnp.where(phips != 0.0, phips, 1.0)
    return jnp.where(phips != 0.0, jnp.asarray(fields.chips) / safe, 0.0)


def _iotas_half(state: SpectralState, rt: SolverRuntime) -> jnp.ndarray:
    """Half-mesh rotational transform of a core state (``add_fluxes.f90``)."""
    setup = rt.setup
    if int(setup.ncurr) != 1:
        return jnp.asarray(setup.iotas)
    _, _, _, fields, _ = _field_chain(state, rt)
    return _iotas_half_from_fields(setup, fields)
