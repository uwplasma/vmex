"""Fast-ion proxies ``Gamma_c`` and ``Gamma_alpha`` of VMEX equilibria: Boozer tables to NEO_JAX.

The physics lives in :func:`neo_jax.fastion.fast_ion_proxies` (Nemov et al. 2008; Velasco et al. 2021
eqs. 14-16, 25; the bounce-averaged drifts as derivatives of the second adiabatic invariant along Boozer
field lines).  This module is the adapter: solver state or wout to the Boozer spectra of five neighbouring
half-mesh rows around each requested surface (the quartic radial fit of ``|B|``, ``iota`` and ``G + iota I``
is inside the kernel), and the least-squares terms.  Surfaces snap to the nearest half-mesh row.
The Boozer tables come from the helpers of :mod:`vmex.core.neoclassical`.
Needs ``neo-jax >= 1.2.0`` (``pip install "vmex[neoclassical]"``).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

import jax.numpy as jnp

from .neoclassical import _half_mesh, _neo_boozer_data, _state_boozer
from .omnigenity import _ResidualTerm

__all__ = ["GammaAlpha", "GammaC", "fast_ion_from_state", "fast_ion_from_wout"]


def _rows(s_half, surfaces) -> list[np.ndarray]:
    """Five consecutive half-mesh rows centred (where the mesh allows) on each surface."""
    s = np.atleast_1d(np.asarray(surfaces, float))
    if s.size == 0 or np.any((s <= 0.0) | (s >= 1.0)):
        raise ValueError("surfaces must be finite and strictly inside (0, 1)")
    centre = np.clip(np.argmin(abs(s_half[None, :] - s[:, None]), axis=1), 2, len(s_half) - 3)
    return [np.arange(c - 2, c + 3) for c in centre]


def _proxies(data, budget):
    from neo_jax.fastion import fast_ion_proxies

    return tuple(v[0] for v in fast_ion_proxies(data, budget))


def fast_ion_from_state(state, rt, surfaces: Sequence[float] = (0.35, 0.6, 0.85), *, budget=None,
                        mboz: int = 12, nboz: int = 9) -> dict:
    """``gamma_c``, ``gamma_alpha`` and ``f_trapped`` at ``surfaces`` of a solved state (traceable)."""
    from neo_jax import booz_xform_to_boozerdata_jax

    s_half = _half_mesh(rt)
    out, used = [], []
    for rows in _rows(s_half, surfaces):
        used.append(s_half[rows[2]])
        booz = _state_boozer(state, rt, s_half[rows], mboz, nboz)
        out.append(_proxies(booz_xform_to_boozerdata_jax(
            booz, mode_first=False, mode_indices=np.arange(len(booz["ixm_b"]))), budget))
    gc, ga, ft = (jnp.stack(v) for v in zip(*out))
    return {"gamma_c": gc, "gamma_alpha": ga, "f_trapped": ft, "s": np.asarray(used)}


def fast_ion_from_wout(wout, surfaces: Sequence[float] = (0.35, 0.6, 0.85), *, budget=None,
                       mboz: int = 12, nboz: int = 9) -> dict:
    """The same from a (VMEX or VMEC) ``wout``, without a solve."""
    from booz_xform_jax import Booz_xform
    from neo_jax import booz_xform_to_boozerdata_jax

    out, used = [], []
    s_half = (np.arange(int(wout.ns) - 1) + 0.5) / (int(wout.ns) - 1)
    for rows in _rows(s_half, surfaces):
        used.append(s_half[rows[2]])
        bx = Booz_xform(verbose=0, mboz=int(mboz), nboz=int(nboz))
        bx.read_wout_data(wout)
        bx.compute_surfs = [int(r) for r in rows]
        bx.run()
        out.append(_proxies(booz_xform_to_boozerdata_jax(_neo_boozer_data(bx, rows)), budget))
    gc, ga, ft = (jnp.stack(v) for v in zip(*out))
    return {"gamma_c": gc, "gamma_alpha": ga, "f_trapped": ft, "s": np.asarray(used)}


class GammaC(_ResidualTerm):
    """Least-squares term: one ``Gamma_c`` row per surface, cost the weighted sum of squares."""

    key = "gamma_c"

    def __init__(self, surfaces=(0.35, 0.6, 0.85), *, weights=None, budget=None, mboz: int = 12, nboz: int = 9):
        self.surfaces = np.atleast_1d(np.asarray(surfaces, dtype=float))
        if np.any(np.diff(self.surfaces) <= 0.0):
            raise ValueError("increasing surfaces strictly inside (0, 1) are required")
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        if self.weights is not None and self.weights.shape != self.surfaces.shape:
            raise ValueError("weights must have the same length as surfaces")
        self.budget, self.mboz, self.nboz = budget, mboz, nboz

    def compute_state(self, state, rt) -> dict:
        """All three proxies at the surfaces."""
        return fast_ion_from_state(state, rt, self.surfaces, budget=self.budget, mboz=self.mboz, nboz=self.nboz)

    def residuals_state(self, state, rt):
        """Weighted rows of the proxy."""
        value = self.compute_state(state, rt)[self.key]
        return value if self.weights is None else jnp.sqrt(jnp.asarray(self.weights)) * value

    def total_state(self, state, rt):
        """Sum of squares of the rows."""
        rows = self.residuals_state(state, rt)
        return jnp.vdot(rows, rows)


class GammaAlpha(GammaC):
    """The same term for ``Gamma_alpha`` (Velasco et al. 2021, eq. 25), the lost fraction of trapped particles."""

    key = "gamma_alpha"
