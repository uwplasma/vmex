"""Coarse -> fine radial interpolation of the spectral state (multigrid).

VMEC2000: ``Sources/TimeStep/interp.f`` — when the ``NS_ARRAY`` ladder moves
to the next radial resolution, the converged spectral coefficients ``xc`` are
interpolated linearly in radius onto the new grid with a VMEC-specific
convention (VMEC++ performs the same linear interpolation of the spectral
coefficients over the ``sqrt(s)``-scaled internal representation):

1. scale the coefficients by ``scalxc`` (``profil3d.f``) so odd-m harmonics
   enter in VMEC's internal ``1/sqrt(s)`` representation — linear in
   ``sqrt(s)`` near the axis;
2. extrapolate odd-m modes to the axis on the *scaled* array,
   ``x(js=1) = 2*x(js=2) - x(js=3)`` (Fortran 1-based);
3. interpolate linearly between the bracketing coarse surfaces using
   ``interp.f``'s ``js1/js2/xint`` uniform-grid construction;
4. divide by ``scalxc`` on the fine grid to return unscaled (internal
   physical) coefficients — the state enters and exits WITHOUT the ``scalxc``
   factor, exactly like the solver's :class:`~vmec_jax.core.solver.SpectralState`;
5. zero odd-m coefficients on the output axis row (edge convention
   ``sqrts(ns) = 1`` is built into ``scalxc``).

The interpolation acts on the m = 1-*constrained* internal coefficients that
:mod:`vmec_jax.core.solver` evolves (``interp.f`` interpolates the internal
``xc``, which is in the constrained basis): every step above is a per-mode
linear map that mixes only coefficients with the same poloidal mode number
``m``, so it commutes with the signed-(m, n) packing and with the m = 1
constraint rotation, and no basis conversion is required.

Math ported from the parity-proven legacy port
``vmec_jax/multigrid.py`` (``interp_vmec_radial_coeffs``).  Pure JAX,
jit-compatible (``ns_coarse``/``ns_fine`` are static shape information), no
host round-trips of traced values.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import jax.numpy as jnp

from .fourier import ModeTable
from .solver import SpectralState
from .transforms import odd_m_sqrt_s_scaling

__all__ = ["interpolate_coefficients", "interpolate_state"]

Array = Any


def _interp_tables(ns_coarse: int, ns_fine: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``interp.f``'s ``js1/js2/xint`` uniform-grid interpolation stencil.

    Static (host, numpy): both grid sizes are shape information, so the
    gather indices and weights are compile-time constants under ``jit``.
    """
    j = np.arange(ns_fine, dtype=np.int64)
    j1 = (j * (ns_coarse - 1)) // (ns_fine - 1)
    j2 = np.minimum(j1 + 1, ns_coarse - 1)
    xint = j.astype(np.float64) * float(ns_coarse - 1) / float(ns_fine - 1) - j1
    xint = np.clip(xint, 0.0, 1.0)
    return j1.astype(np.int32), j2.astype(np.int32), xint


def _scalxc_per_mode(ns: int, m: np.ndarray, dtype) -> Array:
    """``scalxc(js, k)`` per stored mode, shape ``(ns, mnmax)``.

    VMEC2000 ``profil3d.f``: ``1/max(sqrts(js), sqrts(2))`` for odd m, 1 for
    even m, on the uniform full mesh ``s = linspace(0, 1, ns)`` with
    ``sqrts(ns) = 1`` exactly (equals :class:`~vmec_jax.core.setup.RunSetup`
    ``.scalxc`` gathered per mode).
    """
    s = jnp.linspace(0.0, 1.0, ns, dtype=dtype)
    mpol = int(np.max(m)) + 1
    table = odd_m_sqrt_s_scaling(s, mpol)              # (ns, mpol)
    return table[:, np.asarray(m, dtype=np.int64)]     # (ns, mnmax)


def interpolate_coefficients(x_coarse: Array, *, m: np.ndarray, ns_fine: int) -> Array:
    """Interpolate one ``(ns_coarse, mnmax)`` coefficient array to ``ns_fine``.

    VMEC2000 ``interp.f`` (see the module docstring for the convention).
    ``x_coarse`` is unscaled (no ``scalxc``); the result is unscaled on the
    fine grid.  ``m`` gives the poloidal mode number of each column (static
    numpy).  ``ns_coarse == ns_fine`` returns the input unchanged (the legacy
    short-circuit); the interior-surface values are reproduced exactly in
    that case by the general path too, since ``xint`` vanishes identically.
    """
    x_coarse = jnp.asarray(x_coarse)
    ns_coarse, mnmax = int(x_coarse.shape[0]), int(x_coarse.shape[1])
    ns_fine = int(ns_fine)
    m = np.asarray(m, dtype=np.int64)
    if m.shape != (mnmax,):
        raise ValueError(f"m has shape {m.shape}, expected ({mnmax},)")
    if ns_coarse <= 0 or ns_fine <= 0:
        return jnp.zeros((max(ns_fine, 0), mnmax), dtype=x_coarse.dtype)
    if ns_coarse == ns_fine:
        return x_coarse
    if ns_fine == 1:
        return x_coarse[:1]
    if ns_coarse == 1:
        return jnp.broadcast_to(x_coarse[:1], (ns_fine, mnmax))

    dtype = x_coarse.dtype
    is_odd = jnp.asarray((m % 2) == 1)

    # 1. enter the scaled (internal odd-m 1/sqrt(s)) representation.
    scal_coarse = _scalxc_per_mode(ns_coarse, m, dtype)
    x_scaled = x_coarse * scal_coarse

    # 2. odd-m axis extrapolation on the scaled array (interp.f):
    #    x(1) = 2*x(2) - x(3)  (Fortran 1-based).
    if ns_coarse >= 3:
        axis_row = jnp.where(is_odd, 2.0 * x_scaled[1] - x_scaled[2], x_scaled[0])
        x_scaled = x_scaled.at[0].set(axis_row)

    # 3. linear interpolation between bracketing coarse surfaces.
    j1, j2, xint = _interp_tables(ns_coarse, ns_fine)
    xint = jnp.asarray(xint, dtype=dtype)
    x_fine_scaled = (1.0 - xint)[:, None] * x_scaled[j1] + xint[:, None] * x_scaled[j2]

    # 4. leave the scaled representation on the fine grid.
    scal_fine = _scalxc_per_mode(ns_fine, m, dtype)
    x_fine = x_fine_scaled / scal_fine

    # 5. zero odd-m modes on the output axis row.
    axis_row = jnp.where(is_odd, jnp.asarray(0.0, dtype=dtype), x_fine[0])
    return x_fine.at[0].set(axis_row)


def interpolate_state(
    state_coarse: SpectralState,
    *,
    ns_fine: int,
    modes: ModeTable,
    ns_coarse: int | None = None,
) -> SpectralState:
    """Interpolate a coarse solver state onto a finer radial grid.

    VMEC2000 ``interp.f``: the multigrid coarse -> fine transfer of ``xc``
    between ``NS_ARRAY`` stages.  ``state_coarse`` is the
    :class:`~vmec_jax.core.solver.SpectralState` of the converged coarse
    stage — signed-(m, n) internal packing, m = 1-constrained, odd-m WITHOUT
    the ``scalxc`` factor — and the result is in the same representation with
    ``ns_fine`` surfaces, ready for :func:`vmec_jax.core.solver.evaluate_forces`
    on the fine :class:`~vmec_jax.core.solver.SolverRuntime`.

    ``modes`` must be the ``mode_table(mpol, ntor)`` shared by both stages
    (multigrid only changes ``ns``).  ``ns_coarse`` is optional (checked
    against the array shapes when given).  Jit-compatible: ``ns_fine`` and
    ``modes`` are static, all array work is traced ``jax.numpy``.
    """
    ns_state = int(jnp.shape(state_coarse.R_cos)[0])
    if ns_coarse is not None and int(ns_coarse) != ns_state:
        raise ValueError(f"ns_coarse={ns_coarse} does not match state ns={ns_state}")
    m = np.asarray(modes.m, dtype=np.int64)
    interp = lambda x: interpolate_coefficients(x, m=m, ns_fine=int(ns_fine))  # noqa: E731
    return SpectralState(
        R_cos=interp(state_coarse.R_cos), R_sin=interp(state_coarse.R_sin),
        Z_cos=interp(state_coarse.Z_cos), Z_sin=interp(state_coarse.Z_sin),
        L_cos=interp(state_coarse.L_cos), L_sin=interp(state_coarse.L_sin),
    )
