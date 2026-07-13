"""Optimization objectives and least-squares driver for the new core (plan.md §5.1, §10).

Simsopt-style vocabulary for the QA/QH/QP/QI examples on the pure new core:

- :class:`QuasisymmetryRatioResidual` — the two-term quasisymmetry ratio
  residual of Landreman & Paul (simsopt ``QuasisymmetryRatioResidual``),
  evaluated from the wout-engine field tables of a converged core state;
  math ported verbatim from the parity-proven legacy
  ``vmec_jax/quasisymmetry.py`` (``quasisymmetry_ratio_residual_from_wout``).
- practical scalar targets — :func:`aspect_ratio`, :func:`mean_iota`,
  :func:`edge_iota`, :func:`mirror_ratio`, :func:`volume`,
  :func:`magnetic_well` — each a pure function of
  ``(SpectralState, SolverRuntime)`` on :mod:`vmec_jax.core.geometry` /
  :mod:`vmec_jax.core.fields`.
- a distilled Goodman-style QI residual (:func:`quasi_isodynamic_residual`)
  keeping exactly the four terms the legacy minimal-seed QI examples
  exercised (now ``examples/optimization/QI_optimization.py``): level-set
  bounce-width
  variance, branch trapped-well width variance, field-line profile
  consistency, and the branch-shuffle profile comparison.  The unused legacy
  knobs (``aligned_profile_*``, ``weighted_shuffle_*``,
  ``shuffle_profile_nphi_out``) were dropped.
- :func:`least_squares` — a thin :func:`scipy.optimize.least_squares` driver
  over boundary Fourier dofs (:func:`pack_boundary`/:func:`unpack_boundary`),
  taking simsopt-style ``(callable, target, weight)`` terms.

Helicity conventions (match legacy/simsopt exactly)
---------------------------------------------------
The QS residual keeps the ``|B|`` spectrum aligned with the single helicity
``chi = helicity_m * theta - helicity_n * nfp * phi`` — ``helicity_n`` is in
units of ``nfp`` (the internal target mode number is ``nn = helicity_n * nfp``):

- QA: ``(helicity_m, helicity_n) = (1, 0)``
- QH: ``(1, -1)`` (i.e. ``chi = theta + nfp*phi``; legacy/simsopt sign — the
  plan's "``n = -nfp``" written in physical toroidal mode numbers)
- QP: ``(0, 1)``

Gradient modes
--------------
:func:`least_squares` defaults to scipy finite differences (``jac=None`` ->
``"2-point"``).  ``jac="implicit"`` uses the Phase-6 implicit-gradient path
(:mod:`vmec_jax.core.implicit`): each trial boundary is solved once through
:func:`~vmec_jax.core.implicit.solve_implicit` (a ``jax.custom_vjp`` around
the host solver) and the exact residual Jacobian comes from *forward*
implicit differentiation of the fixed point — one preconditioned GMRES per
boundary dof (a few dozen residual linearizations each) instead of one full
equilibrium solve per dof.  In implicit mode every objective term must be a
traceable function of ``(SpectralState, SolverRuntime)``; vector-valued
terms exposing a ``residuals_state`` method
(:class:`QuasisymmetryRatioResidual`) contribute their full pointwise
residual vector, matching the finite-difference stacked-residual cost and
Gauss-Newton geometry (internal-grid sampling instead of the wout grid).
Wout-engine terms (:func:`d_merc`, :func:`l_grad_b`, the Boozer-based QI
residual) run on host NumPy and are finite-difference-only.  The implicit
parameter map does not implement the lasym ``readin.f`` delta rotation, so
``jac="implicit"`` requires ``lasym = False``.

Measured cost (2026-07-10, RTX A4000, nfp2 circular seed, QS + aspect +
iota objective, ``max_mode=2`` -> 24 dofs): warm implicit Jacobian 2.5 s
(~1.5 hot-restart equilibrium-solve equivalents, independent of the dof
count) vs the 2-point FD Jacobian's 24 hot solves ~ 39 s — **15.7x**; the
gap widens linearly with ``max_mode``.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, Iterable, Sequence

import numpy as np

import jax
import jax.numpy as jnp

from .input import VmecInput
from .multigrid import solve_multigrid
from .optimization_parameters import (
    _CURTOR_SCALE,  # noqa: F401 - pre-split compatibility export
    _apply_current,
    _current_dof_setup,
    _dof_modes,
    _ess_scale,
    _pack_current,
    boundary_dof_names,
    pack_boundary,
    unpack_boundary,
)
from .optimization_qi import (
    boozer_modes_from_wout,
    quasi_isodynamic_residual,
    quasi_isodynamic_residual_from_wout,
)
from .solver import (
    SolveResult,
    SolverRuntime,
    SpectralState,
    prepare_runtime,
    resolution_from_input,
)
from .fields import surface_currents
# Shared state-physics primitives (statephysics.py, R26a).  Re-exported here
# for backward compatibility: external user code and tests reach them as
# ``vmec_jax.core.optimize._as_1d`` etc.
from .statephysics import (
    _as_1d,
    _field_chain,
    _half_grid,
    _interp_half_grid,
    _iotas_half,
    _iotas_half_from_fields,
    _mode_matrix,
)
from .wout import WoutData, wout_from_state

__all__ = [
    "Equilibrium",
    "solve_equilibrium",
    "QuasisymmetryRatioResidual",
    "aspect_ratio",
    "mean_iota",
    "edge_iota",
    "mirror_ratio",
    "volume",
    "magnetic_well",
    "d_merc",
    "l_grad_b",
    "quasi_isodynamic_residual",
    "boozer_modes_from_wout",
    "quasi_isodynamic_residual_from_wout",
    "boundary_dof_names",
    "pack_boundary",
    "unpack_boundary",
    "least_squares",
    "RedlBootstrapMismatch",  # noqa: F822 - provided lazily by __getattr__ below
]

Array = Any


def __getattr__(name: str):  # PEP 562 lazy re-export
    # bootstrap.py lazily imports this module inside self_consistent_bootstrap,
    # so the f_boot objective is re-exported lazily to keep the two decoupled.
    if name == "RedlBootstrapMismatch":
        from .bootstrap import RedlBootstrapMismatch
        return RedlBootstrapMismatch
    if name == "_traceable_term":
        from .optimization_implicit import _traceable_term
        return _traceable_term
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ===========================================================================
# Converged-equilibrium bundle
# ===========================================================================


@dataclass(frozen=True)
class Equilibrium:
    """A converged fixed-boundary equilibrium plus its evaluation contexts.

    Objective callables in :func:`least_squares` receive one of these.  The
    solver-native pieces (``state``, ``runtime``) feed the differentiable
    scalar targets; ``wout`` (built lazily, host NumPy) feeds the wout-table
    objectives (QS ratio residual, Boozer-based QI residual).
    """

    inp: VmecInput
    state: SpectralState
    runtime: SolverRuntime
    result: SolveResult

    @cached_property
    def wout(self) -> WoutData:
        """Full wout dataset of this state (``vmec_jax.core.wout``, cached)."""
        r = self.result
        return wout_from_state(
            inp=self.inp, state=self.state,
            fsqr=float(r.fsqr), fsqz=float(r.fsqz), fsql=float(r.fsql),
            niter=int(r.iterations), converged=bool(r.converged),
        )


def solve_equilibrium(
    inp: VmecInput,
    *,
    initial_state: SpectralState | None = None,
    raise_on_max_iterations: bool = False,
    **solve_kwargs,
) -> Equilibrium:
    """Converge ``inp`` with the core multigrid solver -> :class:`Equilibrium`.

    ``raise_on_max_iterations=False`` by default: during optimization a
    NITER-exhausted trial state is still a usable (penalized) sample —
    VMEC2000 behaves the same way.  Extra keywords go to
    :func:`vmec_jax.core.multigrid.solve_multigrid`.
    """
    result = solve_multigrid(
        inp, verbose=False, initial_state=initial_state,
        raise_on_max_iterations=raise_on_max_iterations, **solve_kwargs,
    )
    ns = int(np.shape(result.state.R_cos)[0])
    runtime = prepare_runtime(inp, resolution_from_input(inp, ns=ns))
    return Equilibrium(inp=inp, state=result.state, runtime=runtime, result=result)


# ===========================================================================
# Quasisymmetry ratio residual (simsopt convention; legacy parity port)
# ===========================================================================


class QuasisymmetryRatioResidual:
    """Two-term quasisymmetry ratio residual (simsopt convention).

    On each requested surface the field is sampled on a uniform
    ``(theta, phi)`` grid (VMEC angles) and the pointwise residual

    ``f = [(B x grad B . grad psi)(nn - iota*m) - (B . grad B)(m*G + nn*I)] / B^3``

    (``m = helicity_m``, ``nn = helicity_n * nfp``, ``G``/``I`` the Boozer
    covariant field averages ``bvco``/``buco``) is weighted by the
    flux-surface measure ``sqrt(nfp*dtheta*dphi*|sqrt g| / V')`` so that
    ``total = sum(residuals**2)`` is simsopt's surface-averaged QS ratio.
    ``f`` vanishes identically iff ``|B|`` depends on the angles only through
    ``helicity_m*theta - nn*phi``.

    The evaluation consumes the parity-proven wout-engine tables
    (``bmnc/gmnc/bsub*/bsup*``, :mod:`vmec_jax.core.nyquist`) of a
    :class:`~vmec_jax.core.wout.WoutData` — from
    :func:`~vmec_jax.core.wout.wout_from_state` or any ``wout_*.nc`` — ported
    from legacy ``quasisymmetry_ratio_residual_from_wout`` (A/B bit-exact).
    """

    name = "qs"

    def __init__(
        self,
        surfaces,
        helicity_m: int = 1,
        helicity_n: int = 0,
        *,
        weights: Iterable[float] | None = None,
        ntheta: int = 63,
        nphi: int = 64,
    ):
        self.surfaces = np.atleast_1d(np.asarray(surfaces, dtype=float))
        self.helicity_m = int(helicity_m)
        self.helicity_n = int(helicity_n)
        self.weights = None if weights is None else np.asarray(list(weights), dtype=float)
        self.ntheta = int(ntheta)
        self.nphi = int(nphi)

    # -- wout-table evaluation ------------------------------------------------

    def compute(self, wout) -> dict[str, Array]:
        """Full diagnostics dict from a wout-like object or :class:`Equilibrium`."""
        if isinstance(wout, Equilibrium):
            wout = wout.wout
        surfaces = _as_1d(self.surfaces)
        nsurf = int(surfaces.shape[0])
        weights = jnp.ones((nsurf,)) if self.weights is None else _as_1d(self.weights)
        if int(weights.shape[0]) != nsurf:
            raise ValueError("weights must have the same length as surfaces")

        nfp = int(wout.nfp)
        iotas = _as_1d(np.asarray(wout.iotas, dtype=float))
        ns = int(iotas.shape[0])
        xm = _as_1d(np.asarray(wout.xm_nyq, dtype=float))
        xn = _as_1d(np.asarray(wout.xn_nyq, dtype=float))
        mn = int(xm.shape[0])
        s_half = _half_grid(ns, iotas.dtype)

        def half(values):
            return _interp_half_grid(values[1:], surfaces, s_half)

        iota = half(iotas)
        G = half(_as_1d(np.asarray(wout.bvco, dtype=float)))
        I = half(_as_1d(np.asarray(wout.buco, dtype=float)))  # noqa: E741 - Boozer I

        tables = {}
        for name in ("gmnc", "bmnc", "bsubumnc", "bsubvmnc", "bsupumnc", "bsupvmnc"):
            tables[name] = half(_mode_matrix(wout, name, ns=ns, mn=mn))
        for name in ("gmns", "bmns", "bsubumns", "bsubvmns", "bsupumns", "bsupvmns"):
            optional = not bool(getattr(wout, "lasym", False))
            tables[name] = half(_mode_matrix(wout, name, ns=ns, mn=mn, optional=True)
                                if not optional else jnp.zeros((ns, mn)))

        theta1d = jnp.linspace(0.0, 2.0 * jnp.pi, self.ntheta, endpoint=False)
        phi1d = jnp.linspace(0.0, 2.0 * jnp.pi / nfp, self.nphi, endpoint=False)
        dtheta, dphi = theta1d[1] - theta1d[0], phi1d[1] - phi1d[0]
        angle = (theta1d[:, None, None] * xm[None, None, :]
                 - phi1d[None, :, None] * xn[None, None, :])
        cosangle, sinangle = jnp.cos(angle), jnp.sin(angle)

        def synth(cos_tab, sin_tab, cos_w=cosangle, sin_w=sinangle):
            return (jnp.einsum("sm,tpm->stp", cos_tab, cos_w)
                    + jnp.einsum("sm,tpm->stp", sin_tab, sin_w))

        modB = synth(tables["bmnc"], tables["bmns"])
        dB_dtheta = (jnp.einsum("sm,tpm,m->stp", tables["bmnc"], -sinangle, xm)
                     + jnp.einsum("sm,tpm,m->stp", tables["bmns"], cosangle, xm))
        dB_dphi = (jnp.einsum("sm,tpm,m->stp", tables["bmnc"], sinangle, xn)
                   + jnp.einsum("sm,tpm,m->stp", tables["bmns"], -cosangle, xn))
        sqrtg = synth(tables["gmnc"], tables["gmns"])
        bsubu = synth(tables["bsubumnc"], tables["bsubumns"])
        bsubv = synth(tables["bsubvmnc"], tables["bsubvmns"])
        bsupu = synth(tables["bsupumnc"], tables["bsupumns"])
        bsupv = synth(tables["bsupvmnc"], tables["bsupvmns"])

        d_psi_d_s = -_as_1d(np.asarray(wout.phi, dtype=float))[-1] / (2.0 * jnp.pi)
        sqrtg_safe = jnp.where(sqrtg != 0.0, sqrtg, jnp.ones_like(sqrtg))
        B_dot_grad_B = bsupu * dB_dtheta + bsupv * dB_dphi
        B_cross_grad_B_dot_grad_psi = (
            d_psi_d_s * (bsubu * dB_dphi - bsubv * dB_dtheta) / sqrtg_safe)

        tiny = jnp.asarray(jnp.finfo(sqrtg.dtype).tiny, dtype=sqrtg.dtype)
        sqrtg_abs = jnp.maximum(jnp.abs(sqrtg), tiny)
        modB_safe = jnp.maximum(jnp.abs(modB), tiny)
        V_prime = nfp * dtheta * dphi * jnp.sum(sqrtg_abs, axis=(1, 2))

        nn = self.helicity_n * nfp
        prefactor = jnp.sqrt(
            weights[:, None, None] * nfp * dtheta * dphi / V_prime[:, None, None] * sqrtg_abs)
        residuals3d = prefactor * (
            B_cross_grad_B_dot_grad_psi * (nn - iota[:, None, None] * self.helicity_m)
            - B_dot_grad_B * (self.helicity_m * G[:, None, None] + nn * I[:, None, None])
        ) / (modB_safe ** 3)

        residuals1d = jnp.ravel(residuals3d)
        return {
            "surfaces": surfaces,
            "residuals1d": residuals1d,
            "residuals3d": residuals3d,
            "profile": jnp.sum(residuals3d * residuals3d, axis=(1, 2)),
            "total": jnp.sum(residuals1d * residuals1d),
            "modB": modB,
            "iota": iota,
            "G": G,
            "I": I,
            "V_prime": V_prime,
        }

    def residuals(self, wout) -> jnp.ndarray:
        """Flat least-squares residual vector (target 0, weight applied by the driver)."""
        return self.compute(wout)["residuals1d"]

    def profile(self, wout) -> jnp.ndarray:
        """Per-surface sum of squared residuals."""
        return self.compute(wout)["profile"]

    def total(self, wout) -> Array:
        """Scalar QS ratio objective ``sum(residuals**2)``."""
        return self.compute(wout)["total"]

    def J(self, eq: Equilibrium) -> jnp.ndarray:
        """Objective-term entry point for :func:`least_squares` (residual vector)."""
        return self.residuals(eq)

    __call__ = J  # the instance itself can be an objective term

    # -- traceable (state, runtime) evaluation --------------------------------

    def _pointwise_state(self, state: SpectralState, rt: SolverRuntime):
        """Weighted pointwise QS residual on the solver's internal grid.

        Traceable core of :meth:`residuals_state` / :meth:`profile_state` /
        :meth:`total_state`.  The reduced symmetric ``[0, pi]`` theta grid is
        mirrored to the full circle with the stellarator-symmetry map
        ``X(2 pi - theta, -zeta) = X(theta, zeta)`` and the ``|B|`` angular
        derivatives come from FFT spectral differentiation on that full
        periodic grid (exact at grid resolution).  Returns ``(r3d, s_half)``
        with ``r3d`` shaped ``(ns - 1, ntheta1, nzeta)`` normalized so that
        ``sum_angles r3d[i]**2`` is the surface-averaged QS ratio ``<f^2>``
        of half-mesh surface ``i`` — the same quantity as the wout-table
        :meth:`profile`, agreeing at discretization level (solver angular
        grid vs the 63x64 wout sampling), not bitwise.  Symmetric
        configurations only (``lasym = False``).
        """
        setup = rt.setup
        if bool(setup.lasym):
            raise NotImplementedError(
                "QuasisymmetryRatioResidual traceable evaluation supports "
                "lasym = False only")
        s = jnp.asarray(setup.s_full)
        nfp = int(rt.resolution.nfp)
        _, jacobian, _, fields, _ = _field_chain(state, rt)

        # Mirror the reduced [0, pi] grid to the full theta circle.
        ntheta2 = int(np.shape(fields.total_pressure)[1])
        nzeta = int(np.shape(fields.total_pressure)[2])
        ntheta1 = max(2 * (ntheta2 - 1), 1)
        i_full = np.arange(ntheta1)
        i_src = np.where(i_full < ntheta2, i_full, ntheta1 - i_full)
        k = np.arange(nzeta)
        k_src = np.where(i_full[:, None] < ntheta2, k[None, :],
                         (nzeta - k[None, :]) % nzeta)
        i_src = np.broadcast_to(i_src[:, None], (ntheta1, nzeta))

        def full(a):
            # Drop the zeroed axis row (js = 0) *before* the singular
            # divisions below: keeping it poisons reverse-mode AD with
            # 0 * inf = nan even though the row never enters the result.
            return jnp.asarray(a)[1:, i_src, k_src]

        # |B| on the half-mesh internal grid (bcovar.f: bsq = |B|^2/2 + p).
        bsq2 = 2.0 * (jnp.asarray(fields.total_pressure)
                      - jnp.asarray(fields.pressure)[:, None, None])
        tiny = jnp.asarray(jnp.finfo(bsq2.dtype).tiny, dtype=bsq2.dtype)
        bmag = jnp.sqrt(jnp.maximum(full(bsq2), tiny))

        # FFT spectral differentiation on the full periodic (theta, zeta) grid;
        # zeta spans one field period, so d/dphi carries the nfp factor.
        kt = jnp.asarray(np.fft.fftfreq(ntheta1) * ntheta1)
        kz = jnp.asarray(np.fft.fftfreq(nzeta) * nzeta * nfp)
        bhat = jnp.fft.fft2(bmag, axes=(1, 2))
        dB_dtheta = jnp.real(jnp.fft.ifft2(1j * kt[None, :, None] * bhat, axes=(1, 2)))
        dB_dphi = jnp.real(jnp.fft.ifft2(1j * kz[None, None, :] * bhat, axes=(1, 2)))

        # Profiles: iota (add_fluxes.f), Boozer covariant averages G/I (fbal.f).
        iota = _iotas_half_from_fields(setup, fields)
        cur = surface_currents(bsubu=fields.bsubu, bsubv=fields.bsubv,
                               trig=rt.trig, s=s, signgs=setup.signgs)
        G, I = jnp.asarray(cur.bvco), jnp.asarray(cur.buco)  # noqa: E741

        # d(psi)/ds = -phi_edge / (2 pi), wout sign convention.
        hs = s[1] - s[0]
        d_psi_d_s = -float(setup.signgs) * hs * jnp.sum(jnp.asarray(setup.phipf)[1:])

        gsqrt = full(jacobian.sqrt_g)
        gsqrt_safe = jnp.where(gsqrt != 0.0, gsqrt, jnp.ones_like(gsqrt))
        B_dot_grad_B = full(fields.bsupu) * dB_dtheta + full(fields.bsupv) * dB_dphi
        B_cross_grad_B_dot_grad_psi = (
            d_psi_d_s * (full(fields.bsubu) * dB_dphi - full(fields.bsubv) * dB_dtheta)
            / gsqrt_safe)
        nn = self.helicity_n * nfp
        iota_h, G_h, I_h = iota[1:], G[1:], I[1:]      # match the sliced grid
        f = (B_cross_grad_B_dot_grad_psi * (nn - iota_h[:, None, None] * self.helicity_m)
             - B_dot_grad_B * (self.helicity_m * G_h + nn * I_h)[:, None, None]) / bmag ** 3

        # Flux-surface measure weights: sum_angles r3d^2 = <f^2> per surface.
        g_abs = jnp.abs(gsqrt)
        den = jnp.maximum(jnp.sum(g_abs, axis=(1, 2), keepdims=True), tiny)
        r3d = f * jnp.sqrt(g_abs / den)                # half-mesh js = 1..ns-1
        return r3d, 0.5 * (s[:-1] + s[1:])

    def _surface_coefficients(self, s_half: jnp.ndarray) -> jnp.ndarray:
        """Nonnegative half-mesh weights ``c`` with ``sum(c * <f^2>) = total``.

        The wout-table convention interpolates per-surface totals onto the
        requested ``surfaces`` and applies ``weights``; because linear
        interpolation is linear in the profile, that is exactly a fixed
        nonnegative combination ``c`` of the half-mesh surfaces (obtained
        here as the VJP of the interpolation).
        """
        surfaces = _as_1d(self.surfaces)
        weights = (jnp.ones((int(surfaces.shape[0]),)) if self.weights is None
                   else _as_1d(self.weights))
        probe = jnp.zeros_like(s_half)
        _, vjp = jax.vjp(
            lambda p: jnp.sum(weights * jnp.interp(surfaces, s_half, p)), probe)
        return vjp(jnp.asarray(1.0, dtype=probe.dtype))[0]

    def residuals_state(self, state: SpectralState, rt: SolverRuntime) -> jnp.ndarray:
        """Traceable flat residual vector with ``sum(r**2) = total_state``.

        The internal-grid analogue of :meth:`residuals` (wout tables): the
        pointwise weighted residual of :meth:`_pointwise_state` scaled by the
        square roots of the surface coefficients — this is the residual
        vector ``jac="implicit"`` optimizes, giving the least-squares driver
        the full pointwise Gauss-Newton geometry.
        """
        r3d, s_half = self._pointwise_state(state, rt)
        c = self._surface_coefficients(s_half)
        return jnp.ravel(jnp.sqrt(c)[:, None, None] * r3d)

    def profile_state(self, state: SpectralState, rt: SolverRuntime) -> Array:
        """Traceable *weighted* per-surface QS totals at ``surfaces``.

        ``weights * interp(surfaces, <f^2> profile)`` from
        :meth:`_pointwise_state`; ``sum = total_state``.
        """
        r3d, s_half = self._pointwise_state(state, rt)
        profile = jnp.sum(r3d * r3d, axis=(1, 2))
        surfaces = _as_1d(self.surfaces)
        weights = (jnp.ones((int(surfaces.shape[0]),)) if self.weights is None
                   else _as_1d(self.weights))
        return weights * jnp.interp(surfaces, s_half, profile)

    def total_state(self, state: SpectralState, rt: SolverRuntime) -> Array:
        """Traceable scalar QS objective: ``sum(profile_state)`` (see there)."""
        return jnp.sum(self.profile_state(state, rt))


# ===========================================================================
# Practical scalar targets — pure functions of (SpectralState, SolverRuntime)
# ===========================================================================


def _aspect_scalars(state: SpectralState, rt: SolverRuntime):
    """``aspectratio.f`` scalars ``(Aminor_p, Rmajor_p, aspect, volume_p)``.

    Boundary-surface quadrature identical to the wout writer
    (:func:`vmec_jax.core.postprocess.aspect_ratio_scalars`), kept in JAX.
    """
    geometry, _, _, _, _ = _field_chain(state, rt)
    sqrts_edge = jnp.asarray(rt.setup.sqrts)[-1]
    rb = jnp.asarray(geometry.R_even)[-1] + sqrts_edge * jnp.asarray(geometry.R_odd)[-1]
    zub = (jnp.asarray(geometry.dZ_dtheta_even)[-1]
           + sqrts_edge * jnp.asarray(geometry.dZ_dtheta_odd)[-1])
    wint = jnp.asarray(rt.trig.wint)
    t1 = rb * zub * wint
    volume_p = 2.0 * jnp.pi ** 2 * jnp.abs(jnp.sum(rb * t1))
    area = 2.0 * jnp.pi * jnp.abs(jnp.sum(t1))
    area_safe = jnp.where(area != 0.0, area, 1.0)
    aminor = jnp.sqrt(area_safe / jnp.pi)
    rmajor = volume_p / (2.0 * jnp.pi * area_safe)
    return aminor, rmajor, rmajor / aminor, volume_p


def aspect_ratio(state: SpectralState, rt: SolverRuntime) -> Array:
    """VMEC aspect ratio ``Rmajor_p / Aminor_p`` (``aspectratio.f`` convention).

    ``Aminor_p = sqrt(<cross-section area> / pi)``, ``Rmajor_p =
    volume_p / (2 pi <area>)`` from the boundary surface quadrature; equals
    the wout ``aspect`` scalar of the same state.
    """
    return _aspect_scalars(state, rt)[2]


def volume(state: SpectralState, rt: SolverRuntime) -> Array:
    """Plasma volume ``volume_p`` [m^3] (wout convention, boundary quadrature)."""
    return _aspect_scalars(state, rt)[3]


def mean_iota(state: SpectralState, rt: SolverRuntime) -> Array:
    """Mean rotational transform over the half-mesh surfaces (axis excluded).

    Matches the legacy optimization ``mean_iota`` convention
    (``mean(iotas[1:])``, i.e. the mean of the wout ``iotas`` profile).
    """
    iotas = _iotas_half(state, rt)
    return jnp.mean(iotas[1:])


def edge_iota(state: SpectralState, rt: SolverRuntime) -> Array:
    """Rotational transform at the boundary (wout ``iotaf[-1]`` convention:
    linear extrapolation of the half mesh, ``1.5 iotas[-1] - 0.5 iotas[-2]``)."""
    iotas = _iotas_half(state, rt)
    return 1.5 * iotas[-1] - 0.5 * iotas[-2]


def mirror_ratio(state: SpectralState, rt: SolverRuntime, *, s_index: int = -1) -> Array:
    """Mirror ratio ``(Bmax - Bmin) / (Bmax + Bmin)`` on one half-mesh surface.

    ``|B|`` is evaluated on the solver's internal angular grid from the
    half-mesh field state (``|B|^2 = 2 (bsq - p)``, ``bcovar.f``); ``s_index``
    selects the half-mesh surface (default: outermost).  Hard max/min — smooth
    almost everywhere, adequate for finite-difference least squares (the
    legacy ``VMECMirrorRatio`` softmax knobs were an optimizer nicety only).
    """
    _, _, _, fields, _ = _field_chain(state, rt)
    bsq = jnp.asarray(fields.total_pressure) - jnp.asarray(fields.pressure)[:, None, None]
    bmag = jnp.sqrt(jnp.maximum(2.0 * bsq[s_index],
                                jnp.asarray(jnp.finfo(bsq.dtype).tiny, dtype=bsq.dtype)))
    bmax, bmin = jnp.max(bmag), jnp.min(bmag)
    return (bmax - bmin) / (bmax + bmin)


def magnetic_well(state: SpectralState, rt: SolverRuntime) -> Array:
    """VMEC/simsopt magnetic-well proxy ``(V'(0) - V'(1)) / V'(0)``.

    ``V' = dV/ds`` endpoints are linear extrapolations of the half-mesh
    differential volume ``vp`` (``bcovar.f``); positive values mean a
    favorable well (``vacuum_well`` in simsopt).  Ported from legacy
    ``vmec_jax.finite_beta.magnetic_well_from_vp``.
    """
    _, _, _, _, energies = _field_chain(state, rt)
    dvol = jnp.abs(jnp.asarray(energies.vp))[1:]
    v0 = 1.5 * dvol[0] - 0.5 * dvol[1]
    v1 = 1.5 * dvol[-1] - 0.5 * dvol[-2]
    v0_safe = jnp.where(v0 != 0.0, v0, jnp.ones_like(v0))
    return jnp.where(v0 != 0.0, (v0 - v1) / v0_safe, 0.0)


def d_merc(eq) -> jnp.ndarray:
    """Mercier stability criterion profile ``DMerc(s)`` (full mesh).

    Positive interior values indicate Mercier stability.  Evaluated through
    the parity-proven wout engine (:func:`vmec_jax.core.nyquist.mercier_and_jxb`
    via :func:`~vmec_jax.core.wout.wout_from_state`) — host NumPy, so this
    objective is finite-difference-only (not jit/AD transparent; the first
    two surfaces and the edge carry the usual near-axis noise, so practical
    targets should penalize e.g. ``min(DMerc[2:-1], 0)``).  Accepts an
    :class:`Equilibrium` or any wout-like object.
    """
    wout = eq.wout if isinstance(eq, Equilibrium) else eq
    return jnp.asarray(np.asarray(wout.DMerc, dtype=float))


def l_grad_b(eq, *, s_index: int = -1, ntheta: int = 24, nphi: int = 24) -> Array:
    """Magnetic-gradient scale length ``min L_grad_B`` on one half-mesh surface.

    ``L_grad_B = |B| sqrt(2 / (grad B : grad B))`` with ``grad B : grad B``
    the squared Frobenius norm of the Cartesian field-gradient tensor —
    the Kappel/Landreman coil-complexity / compactness proxy, and the
    ``L_grad_B`` diagnostic of the legacy QI scripts
    (``vmec_jax.quasi_isodynamic.objectives.lgradb_from_state``).  Here it is
    evaluated from the wout tables of the converged state: ``B^u``/``B^v``
    from the half-mesh ``bsupumnc/bsupvmnc`` Nyquist spectra, the coordinate
    basis vectors and their derivatives spectrally from ``rmnc/zmns``, and
    radial derivatives from the native half/full-mesh finite differences
    (one-sided at the edge) — so values agree with the legacy diagnostic at
    discretization level, not bitwise.

    Returns the (hard) minimum over a uniform ``(theta, phi)`` grid on the
    selected surface (``s_index`` indexes the ``ns``-long half-mesh arrays;
    default edge).  Larger is better; a practical least-squares term is
    ``max(1/L - 1/threshold, 0)``.  Symmetric configurations only (lasym
    sine partners are ignored).  Accepts an :class:`Equilibrium` or wout-like.
    """
    wout = eq.wout if isinstance(eq, Equilibrium) else eq
    ns = int(wout.ns)
    j = max(1, min(s_index % ns, ns - 1))
    hs = 1.0 / (ns - 1)
    xm = jnp.asarray(np.asarray(wout.xm, dtype=float))
    xn = jnp.asarray(np.asarray(wout.xn, dtype=float))
    xmn = jnp.asarray(np.asarray(wout.xm_nyq, dtype=float))
    xnn = jnp.asarray(np.asarray(wout.xn_nyq, dtype=float))
    rmnc = jnp.asarray(np.asarray(wout.rmnc, dtype=float))
    zmns = jnp.asarray(np.asarray(wout.zmns, dtype=float))
    bsupu_t = jnp.asarray(np.asarray(wout.bsupumnc, dtype=float))
    bsupv_t = jnp.asarray(np.asarray(wout.bsupvmnc, dtype=float))

    theta = jnp.linspace(0.0, 2.0 * jnp.pi, int(ntheta), endpoint=False)
    phi = jnp.linspace(0.0, 2.0 * jnp.pi / int(wout.nfp), int(nphi), endpoint=False)

    def tables(m, n):
        ang = (theta[:, None, None] * m[None, None, :]
               - phi[None, :, None] * n[None, None, :])
        return jnp.cos(ang), jnp.sin(ang)

    cosang, sinang = tables(xm, xn)

    def series(coeff, parity, second: bool = True):
        """Value + angular derivatives of a cos/sin(m theta - n phi) series."""
        base, alt = (cosang, sinang) if parity == "cos" else (sinang, cosang)
        s1 = -1.0 if parity == "cos" else 1.0
        val = jnp.einsum("m,tpm->tp", coeff, base)
        d_t = s1 * jnp.einsum("m,tpm,m->tp", coeff, alt, xm)
        d_p = -s1 * jnp.einsum("m,tpm,m->tp", coeff, alt, xn)
        if not second:
            return val, d_t, d_p, None, None, None
        d_tt = -jnp.einsum("m,tpm,m->tp", coeff, base, xm * xm)
        d_tp = jnp.einsum("m,tpm,m->tp", coeff, base, xm * xn)
        d_pp = -jnp.einsum("m,tpm,m->tp", coeff, base, xn * xn)
        return val, d_t, d_p, d_tt, d_tp, d_pp

    # Full-mesh R/Z -> half-mesh values + radial derivatives (exact on half mesh).
    R, Ru, Rv, Ruu, Ruv, Rvv = series(0.5 * (rmnc[j - 1] + rmnc[j]), "cos")
    Z, Zu, Zv, Zuu, Zuv, Zvv = series(0.5 * (zmns[j - 1] + zmns[j]), "sin")
    Rs, Rsu, Rsv, _, _, _ = series((rmnc[j] - rmnc[j - 1]) / hs, "cos", second=False)
    Zs, Zsu, Zsv, _, _, _ = series((zmns[j] - zmns[j - 1]) / hs, "sin", second=False)

    cphi, sphi = jnp.cos(phi)[None, :], jnp.sin(phi)[None, :]

    def cart(vR, vP, vZ):
        """Cylindrical (R, phi, Z) components -> Cartesian (x, y, z)."""
        return jnp.stack([vR * cphi - vP * sphi, vR * sphi + vP * cphi, vZ], axis=-1)

    zero = jnp.zeros_like(R)
    e_s, e_u, e_v = cart(Rs, zero, Zs), cart(Ru, zero, Zu), cart(Rv, R, Zv)
    # d(e_u)/du, d(e_u)/dv=d(e_v)/du, d(e_v)/dv, d(e_u)/ds, d(e_v)/ds
    deu_u = cart(Ruu, zero, Zuu)
    deu_v = cart(Ruv, Ru, Zuv)
    dev_v = cart(Rvv - R, 2.0 * Rv, Zvv)
    deu_s = cart(Rsu, zero, Zsu)
    dev_s = cart(Rsv, Rs, Zsv)

    # Half-mesh contravariant field (Nyquist modes) + radial derivative.
    cosn, sinn = tables(xmn, xnn)

    def nyq(coeff):
        return (jnp.einsum("m,tpm->tp", coeff, cosn),
                -jnp.einsum("m,tpm,m->tp", coeff, sinn, xmn),
                jnp.einsum("m,tpm,m->tp", coeff, sinn, xnn))

    bu, bu_t, bu_p = nyq(bsupu_t[j])
    bv, bv_t, bv_p = nyq(bsupv_t[j])
    lo, hi = (j - 1, j + 1) if 1 < j < ns - 1 else ((j, j + 1) if j == 1 else (j - 1, j))
    span = hs * (hi - lo)
    bu_s, _, _ = nyq((bsupu_t[hi] - bsupu_t[lo]) / span)
    bv_s, _, _ = nyq((bsupv_t[hi] - bsupv_t[lo]) / span)

    B = bu[..., None] * e_u + bv[..., None] * e_v
    dB = jnp.stack([
        bu_s[..., None] * e_u + bv_s[..., None] * e_v
        + bu[..., None] * deu_s + bv[..., None] * dev_s,
        bu_t[..., None] * e_u + bv_t[..., None] * e_v
        + bu[..., None] * deu_u + bv[..., None] * deu_v,
        bu_p[..., None] * e_u + bv_p[..., None] * e_v
        + bu[..., None] * deu_v + bv[..., None] * dev_v,
    ], axis=-2)                                        # (t, p, coord, cart)

    basis = jnp.stack([e_s, e_u, e_v], axis=-2)
    g = jnp.einsum("...ic,...jc->...ij", basis, basis)
    ginv = jnp.linalg.inv(g)
    grad_sq = jnp.einsum("...ic,...ij,...jc->...", dB, ginv, dB)
    tiny = jnp.asarray(jnp.finfo(grad_sq.dtype).tiny, dtype=grad_sq.dtype)
    bmag = jnp.sqrt(jnp.maximum(jnp.sum(B * B, axis=-1), tiny))
    return jnp.min(bmag * jnp.sqrt(2.0 / jnp.maximum(grad_sq, tiny)))


# Boundary/current parameterization and least-squares driver.
def _call_term(fun: Callable, eq: Equilibrium) -> np.ndarray:
    """Evaluate an objective callable against an :class:`Equilibrium`.

    Callables with two or more positional parameters are treated as pure
    ``(state, runtime)`` functions (the scalar targets above); single-argument
    callables receive the :class:`Equilibrium` (e.g.
    ``QuasisymmetryRatioResidual.J`` or user lambdas).
    """
    try:
        params = [p for p in inspect.signature(fun).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        two_positional = len(params) >= 2 and params[1].default is inspect.Parameter.empty
    except (TypeError, ValueError):  # builtins / partials without signature
        two_positional = False
    value = fun(eq.state, eq.runtime) if two_positional else fun(eq)
    return np.atleast_1d(np.asarray(jax.device_get(value), dtype=float)).ravel()


def least_squares(
    objective_terms: Sequence[tuple[Callable, float, float]],
    inp: VmecInput,
    *,
    max_mode: int | Sequence[int] = 1,
    x0: np.ndarray | None = None,
    initial_state: SpectralState | None = None,
    current_dofs: int | None = None,
    jac: str | None = None,
    jac_chunk_size: int | str | None = "auto",
    jac_solver: str = "block",
    recycle: bool = False,
    hot_restart: bool = True,
    warm_start: str | None = "perturbation",
    use_ess: bool = False,
    ess_alpha: float = 1.2,
    device: Any = None,
    solve_kwargs: dict | None = None,
    verbose: int = 0,
    **scipy_kwargs,
):
    """Boundary-shape least squares: simsopt's ``least_squares_serial_solve``.

    ``objective_terms`` is a list of ``(fun, target, weight)``: each ``fun``
    maps a converged :class:`Equilibrium` (or, for two-positional-argument
    callables, its ``(state, runtime)`` pair) to a scalar or residual vector,
    and contributes ``weight * (fun(eq) - target)`` rows to the stacked
    residual, i.e. ``cost = 1/2 sum_i w_i^2 (f_i - t_i)^2`` (scipy's 1/2
    convention).  The decision variables are the boundary Fourier
    coefficients up to ``max_mode`` (:func:`pack_boundary` — ``RBC(0,0)``
    fixed).  Trial boundaries whose solve fails return a large finite
    residual so the trust region backs off instead of crashing.  Objective
    swaps between stages of a staged campaign (e.g. the QP-basin-then-QI
    route) are just two calls with different ``objective_terms``, the second
    seeded with the first call's ``result.input``.

    ``current_dofs = k`` (plan R26.g, spec section 6.4) additionally frees
    the equilibrium current profile: the first ``k`` ``AC`` power-series
    coefficients (VMEC ``pcurr_type="power_series"``, i.e. ``I'(s)``) plus
    ``CURTOR``, appended to the dof vector as
    ``[..boundary.., ac_0/ac_scale, ..., ac_{k-1}/ac_scale, curtor/1e6]``
    (``ac_scale = max(|curtor|, 1)`` frozen from the seed) so the trust
    region sees O(1) numbers.  Requires ``ncurr = 1`` and an
    AC-parameterized ``pcurr_type`` (splines are rejected — re-fit them to a
    power series first, e.g. via
    :func:`vmec_jax.core.bootstrap.self_consistent_bootstrap`).  Both
    gradient modes support it (finite differences re-solve per current dof;
    ``jac="implicit"`` adds ``k + 1`` one-hot tangent rows through
    ``ImplicitParams.ac``/``curtor``, which ``runtime_from_params`` already
    traces).  Note that VMEC normalizes the AC profile by its own edge
    integral (only the *shape* of ``I'`` matters; ``CURTOR`` sets the
    amplitude), so the overall-AC-scale direction is objective-neutral; the
    trust-region solvers handle the resulting Jacobian null direction.
    This is the dof set of the self-consistent-bootstrap objective
    (:class:`vmec_jax.core.bootstrap.RedlBootstrapMismatch`).

    ``max_mode`` may be a single int or an increasing schedule (e.g.
    ``(1, 2, 3)``): each continuation stage optimizes the enlarged dof set
    starting from the previous stage's boundary (higher harmonics enter at
    their current — typically zero — values).  Repeated trial solves are
    cheap by construction: runtimes with the same
    :class:`~vmec_jax.core.fourier.Resolution` are structural pytrees, so the
    solver reuses one XLA executable across all boundary trials
    (``vmec_jax.core.solver`` plan.md Phase-2 cache; only the first solve of
    a stage compiles).  ``device`` is forwarded to the solver
    (:mod:`vmec_jax.core.device` policy applies when ``None``).

    ``initial_state`` seeds the first equilibrium solve of a single-stage
    call.  It is useful when continuing an optimization with new objective
    weights; subsequent trials still use the usual state or perturbation hot
    restart.  Pass an explicit single ``max_mode`` when using this option.

    ``use_ess`` enables Exponential Spectral Scaling of the trust region
    (:func:`_ess_scale`, ``ess_alpha``), the legacy ``use_ess`` option.

    ``jac=None`` (default) uses scipy ``"2-point"`` finite differences.
    ``jac="implicit"`` uses the Phase-6 implicit-gradient path
    (:mod:`vmec_jax.core.implicit`): one hot-restarted forward solve per
    trial boundary, and the exact residual Jacobian by forward implicit
    differentiation — one preconditioned GMRES per boundary dof instead of
    one full equilibrium solve per dof.  Requirements (see the module
    docstring): every term traceable in ``(state, runtime)`` (vector terms
    expose ``residuals_state``; wout-engine terms like :func:`d_merc` /
    :func:`l_grad_b` / the Boozer QI residual need ``jac=None``) and
    ``lasym = False`` (the implicit parameter map does not implement the
    lasym ``readin.f`` boundary rotation).
    ``jac_chunk_size`` (R17.1 memory knob, ``jac="implicit"`` only) chunks the
    per-dof Jacobian columns via :func:`solvax.chunk_map`: ``"auto"`` (default)
    lets :func:`solvax.auto_chunk_size` pick a memory-bounded width (the
    largest block that fits the device budget on GPU, a sqrt-balanced width on
    CPU) so peak Jacobian memory is ``m0 + m1*chunk`` instead of scaling with
    the full dof count; an ``int`` fixes that many boundary dofs at a time; and
    ``None`` forces one wide ``vmap`` over all dofs (the pre-R17.1 behavior,
    fastest but peak memory O(dofs)).  The column blocks are mathematically
    independent, so the assembled Jacobian is identical to float64 round-off
    (~1e-15) across chunk sizes.  It is inert for ``jac=None`` (scipy computes
    the finite-difference Jacobian itself).

    ``jac_solver`` (plan R25.2, ``jac="implicit"`` only) selects the linear
    solver behind the per-dof implicit-Jacobian columns.  ``"block"``
    (default) amortizes one block-tridiagonal factorization of the *raw*
    force Jacobian — whose radial coupling is exactly nearest-neighbor, so
    ns dense ``(3*mn, 3*mn)`` blocks assembled by 3-colored ``jax.jvp``
    probes capture it completely at a cost independent of the dof count —
    then backsolves every dof right-hand side directly
    (:func:`solvax.block_thomas_factor` / :func:`solvax.block_thomas_solve`)
    and certifies each column with a short warm-started GMRES pass against
    the preconditioned system (same ``adjoint_tol`` norm as the default
    path; columns already at tolerance cost one matvec).  ``"gmres"`` is the
    pre-R25.2 path: one independent preconditioned GMRES per dof column.
    Both produce the same Jacobian to solver tolerance; ``"gmres"`` is the
    fallback if the block path misbehaves on an exotic configuration.
    Inert for ``jac=None``; ``recycle=True`` takes precedence.

    ``recycle`` (plan R25.3, ``jac="implicit"`` only) carries a GCROT
    deflation pair across the per-dof implicit-Jacobian solves — a
    ``lax.scan`` over dof chunks (vmapped within a chunk) threads the
    :func:`solvax.gcrot` recycle space ``(C, U)`` between chunks and, via a
    Python-side holder, between successive trust-region Jacobian
    evaluations.  Recycled solves keep the exact ``adjoint_tol`` /
    ``adjoint_maxiter`` budget of the default path.  **Default False**:
    measured on the nfp2 minimal-seed max_mode-2 operator (2026-07-11), the
    solvax v0.1 recycle space (FIFO cycle corrections, not the harmonic
    Ritz vectors of GCRO-DR) *slows* warm-started columns — e.g. 140 (cold
    GMRES) -> 236/347/479 iterations at k = 2/5/10 — so columns that then
    exhaust ``adjoint_maxiter`` return larger residuals.  Enable only after
    benchmarking per-column iteration counts on your operator.
    ``recycle=False`` uses the independent per-column :func:`solvax.gmres`
    path (identical columns across chunk sizes to float64 round-off).
    Inert for ``jac=None``.

    ``hot_restart`` seeds each trial solve from the previous converged state
    (both modes; in implicit mode via the per-config host-solve cache).

    ``warm_start`` (plan R25.4, ``jac="implicit"`` only) refines what that
    seed is.  ``"perturbation"`` (default) seeds each *trial* solve with the
    DESC-style first-order prediction ``x_ref + sum_j (dx)_j dz_j``
    (arXiv:2203.15927 ``eq.perturb`` before ``eq.solve``): the per-dof state
    responses ``dz_j = -(dF/dz)^{-1} dF/dp t_j`` are exactly the columns the
    implicit Jacobian already solves, so the linearization is stashed at
    each ``jac(x_ref)`` call for free and evaluated per trial at the cost of
    one small tensor contraction.  ``"state"`` is the plain hot restart
    (seed = last converged state); ``None`` disables warm starting entirely
    (every trial re-solves from the interior guess).  All three converge to
    the same fixed points — only the inner iteration count changes — and
    anything missing or mismatched (first call, stage change, failed seed)
    falls back silently down the ladder perturbation -> state -> cold.
    ``recycle=True`` bypasses the perturbation stash (its Jacobian variant
    carries the GCROT pair instead); ``hot_restart=False`` forces
    ``warm_start=None``.  Inert for ``jac=None``.  Measured (2026-07-12,
    M-series CPU, nfp2 minimal seed, max_mode=2, QS+aspect, ``max_nfev=20``):
    total forward-solve iterations 23685 (``"state"``) -> 6364
    (``"perturbation"``), **3.7x fewer**, wall 156 s -> 145 s (this deck's
    wall is Jacobian-dominated at ns=35; the solve-phase win grows with
    ``ns``), and the seed additionally rescued a trial whose plain hot
    restart failed with a Jacobian-sign error.

    Remaining keywords go to :func:`scipy.optimize.least_squares` (e.g.
    ``max_nfev``, ``ftol``, ``xtol``, ``diff_step``).

    Returns the scipy ``OptimizeResult`` of the final stage with extra
    attributes: ``input`` (optimized :class:`VmecInput`), ``equilibrium``
    (last successfully solved :class:`Equilibrium`), ``stage_results``
    (per-``max_mode`` results for schedules) and, in implicit mode,
    ``solve_stats`` (``{"solves", "iterations"}`` totals of the stage's host
    forward solves — the R25.4 warm-start instrumentation).
    """
    import scipy.optimize

    modes_schedule = ([int(max_mode)] if np.isscalar(max_mode)
                      else [int(m) for m in max_mode])
    if len(modes_schedule) > 1:
        if x0 is not None:
            raise ValueError("x0 cannot be combined with a max_mode schedule")
        if initial_state is not None:
            raise ValueError("initial_state requires a single max_mode")
        stage_results = []
        current = inp
        result = None
        for mm in modes_schedule:
            result = least_squares(
                objective_terms, current, max_mode=mm,
                current_dofs=current_dofs, jac=jac,
                jac_chunk_size=jac_chunk_size, jac_solver=jac_solver,
                recycle=recycle, hot_restart=hot_restart,
                warm_start=warm_start, use_ess=use_ess,
                ess_alpha=ess_alpha, device=device, solve_kwargs=solve_kwargs,
                verbose=verbose, **scipy_kwargs)
            stage_results.append(result)
            current = result.input
        result.stage_results = stage_results
        return result
    max_mode = modes_schedule[0]
    k_cur, ac_scale = _current_dof_setup(inp, current_dofs)

    def _ess_scale_full() -> np.ndarray:
        scale = _ess_scale(inp, max_mode, float(ess_alpha))
        if k_cur:  # current dofs are already O(1) by construction
            scale = np.concatenate([scale, np.ones(k_cur + 1)])
        return scale

    if jac == "implicit":
        from .optimization_implicit import least_squares_implicit

        if use_ess:
            scipy_kwargs.setdefault("x_scale", _ess_scale_full())
        return least_squares_implicit(
            objective_terms, inp, max_mode=max_mode, x0=x0,
            initial_state=initial_state,
            current_dofs=current_dofs,
            jac_chunk_size=jac_chunk_size, jac_solver=jac_solver,
            recycle=recycle,
            warm_start=(warm_start if hot_restart else None),
            solve_kwargs=dict(solve_kwargs or {}),
            device=device, verbose=verbose, **scipy_kwargs)
    if jac is not None:
        raise ValueError(f"jac must be None or 'implicit', got {jac!r}")

    solve_kwargs = dict(solve_kwargs or {})
    if device is not None:
        solve_kwargs.setdefault("device", device)
    if use_ess:
        scipy_kwargs.setdefault("x_scale", _ess_scale_full())
    if x0 is None:
        x0 = pack_boundary(inp, max_mode)
        if k_cur:
            x0 = np.concatenate([x0, _pack_current(inp, k_cur, ac_scale)])
    nb = 2 * len(_dof_modes(inp, max_mode))

    def unpack_full(x: np.ndarray) -> VmecInput:
        trial = unpack_boundary(inp, np.asarray(x, dtype=float)[:nb], max_mode)
        if k_cur:
            trial = _apply_current(trial, np.asarray(x, dtype=float)[nb:],
                                   k_cur, ac_scale)
        return trial

    state_holder: dict[str, Any] = {
        "hot": initial_state, "eq": None, "nres": None}
    single_stage = int(np.asarray(inp.ns_array).size) == 1

    def fun(x: np.ndarray) -> np.ndarray:
        trial = unpack_full(x)
        try:
            seed = state_holder["hot"] if (hot_restart and single_stage) else None
            eq = solve_equilibrium(trial, initial_state=seed, **solve_kwargs)
            parts = [w * (_call_term(f, eq) - t) for (f, t, w) in objective_terms]
            residual = np.concatenate(parts)
        except Exception as exc:  # zero-crash policy: penalize, don't die
            if state_holder["nres"] is None:
                raise  # the very first evaluation must succeed (sizes scipy's residual)
            if verbose:
                print(f"[least_squares] trial solve failed: {exc}")
            return np.full((state_holder["nres"],), 1.0e6)
        if not np.all(np.isfinite(residual)):
            residual = np.where(np.isfinite(residual), residual, 1.0e6)
        state_holder["hot"] = eq.state
        state_holder["eq"] = eq
        state_holder["nres"] = residual.size
        if verbose:
            print(f"[least_squares] cost = {0.5 * float(residual @ residual):.6e}")
        return residual

    result = scipy.optimize.least_squares(fun, np.asarray(x0, dtype=float),
                                          jac="2-point", **scipy_kwargs)
    result.input = unpack_full(result.x)
    result.equilibrium = state_holder["eq"]
    return result
