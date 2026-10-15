r"""Galerkin B-spline radial discretization of the fixed-boundary MHD energy.

Every Fourier amplitude is ``rho**|m| q(s)`` with ``q`` a clamped B-spline in
``s = rho**2`` (:mod:`vmex.core.radial_basis`), so axis regularity holds by
construction.  The unknowns are the spline coefficients of ``R_mn``,
``Z_mn``, ``lambda_mn`` and, for a prescribed toroidal current, of
``chi'(s)``.  The discrete functional is

.. math::

    L = \int \Big(\frac{B^2}{2\mu_0} - p(s)\Big) dV
        - \frac{4\pi^2}{\mu_0}\int_0^1 \bar I(s)\,\chi'(s)\,ds,

with ``bar I = signgs mu0 I_tor / 2 pi`` (VMEC ``icurv``); the last term is
present only for ``NCURR = 1`` and makes stationarity in ``chi'`` the current
constraint, because ``dW/dchi'(s) = signgs 2 pi I(s)``.  Quadrature is
Gauss-Legendre in ``s`` on every knot span times a uniform angular grid.  The
Galerkin force is ``-grad L`` by JAX autodiff, so it is the exact gradient of
the discrete energy (Hindenlang, Plunk and Maj, PPCF 67 045002 (2025); Hirshman
and Whitson, Phys. Fluids 26 3553 (1983)).  :func:`solve_spline` minimizes
``L`` with a damped Newton method on the exact Hessian.

Scope: fixed boundary, ``LASYM = F``, ``GAMMA = 0``.  The coordinates and
field conventions are those of :mod:`vmex.core.strong_force`, so a result is
a :class:`~vmex.core.strong_force.HighOrderEquilibriumState` that the
independent force oracle certifies and :func:`~vmex.core.polish.sample_high_order_state`
puts on any legacy full mesh.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .profiles import MU0
from .radial_basis import BSplineBasis
from .strong_force import HighOrderEquilibriumState

Array = Any


@dataclass(frozen=True, eq=False)
class SplinePlan:
    """Static quadrature tables and the free-coefficient layout."""

    state: HighOrderEquilibriumState  # template: basis, modes, profiles, fixed values
    rho: np.ndarray  # (Nq,)
    weight: np.ndarray  # (Nq,) Gauss weights in s
    cell: float  # angular cell area times nfp
    value: np.ndarray  # (mnmax, Nq, nbasis): rho^m B_k(s)
    drho: np.ndarray  # (mnmax, Nq, nbasis): d/drho of rho^m B_k(s)
    basis: np.ndarray  # (Nq, nbasis): B_k(s) for the profiles
    cos: np.ndarray  # (mnmax, Nang)
    sin: np.ndarray  # (mnmax, Nang)
    phip: np.ndarray  # (Nq,)
    pressure: np.ndarray  # (Nq,)
    icurv: np.ndarray | None  # (Nq,) for NCURR = 1, else None
    free: np.ndarray  # flat indices into the packed coefficient vector

    @property
    def size(self) -> int:
        """Number of free coefficients."""

        return int(self.free.size)


def _pack(state: HighOrderEquilibriumState) -> Array:
    return jnp.concatenate([jnp.ravel(state.R_cos), jnp.ravel(state.Z_sin), jnp.ravel(state.L_sin), state.chipf])


def make_spline_plan(
    state: HighOrderEquilibriumState,
    *,
    icurv: Any = None,
    ntheta: int | None = None,
    nzeta: int | None = None,
    radial_order: int | None = None,
) -> SplinePlan:
    """Tabulate the basis on the quadrature grid of ``state.radial_basis``.

    ``icurv`` is a callable ``s -> signgs mu0 I(s) / 2 pi`` for a prescribed
    current; ``None`` holds ``chi'`` (the iota profile) fixed.  The default
    angular grid integrates products of four mode sets without aliasing.
    """

    basis: BSplineBasis = state.radial_basis
    m = np.asarray(state.m, dtype=int)
    n = np.asarray(state.n, dtype=int)
    ntheta = int(ntheta or 4 * int(m.max()) + 4)
    nzeta = int(nzeta or (1 if not np.any(n) else 4 * int(np.abs(n).max()) + 4))
    order = int(radial_order or basis.degree + 3)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    breaks = basis.breakpoints
    centre, half = 0.5 * (breaks[:-1] + breaks[1:]), 0.5 * np.diff(breaks)
    s = (centre[:, None] + half[:, None] * nodes).ravel()
    ws = (half[:, None] * weights).ravel()
    rho = np.sqrt(s)
    B = np.asarray(basis.basis_matrix(s))
    dB = np.asarray(basis.basis_matrix(s, derivative=1))
    rm = rho[None, :, None] ** m[:, None, None]
    value = rm * B[None]
    drho = m[:, None, None] * rho[None, :, None] ** np.maximum(m - 1, 0)[:, None, None] * B[None] * (m > 0)[:, None, None]
    drho = drho + 2.0 * rho[None, :, None] * rm * dB[None]
    theta = 2 * np.pi * np.arange(ntheta) / ntheta
    zeta = 2 * np.pi * np.arange(nzeta) / nzeta
    phase = m[:, None, None] * theta[None, :, None] - n[:, None, None] * zeta[None, None, :]
    area = (2 * np.pi) ** 2 / (ntheta * nzeta)
    nb = basis.size
    mn = m.size
    free = np.ones((3, mn, nb), dtype=bool)
    free[:2, :, -1] = False  # fixed boundary
    free[1:, (m == 0) & (n == 0)] = False  # sin(0) and the lambda gauge
    free = np.concatenate([free.ravel(), np.full(nb, icurv is not None)])
    return SplinePlan(
        state=state,
        rho=rho,
        weight=ws,
        cell=area * int(state.nfp),
        value=value,
        drho=drho,
        basis=B,
        cos=np.cos(phase).reshape(mn, -1),
        sin=np.sin(phase).reshape(mn, -1),
        phip=B @ np.asarray(state.phipf),
        pressure=B @ np.asarray(state.pressure),
        icurv=None if icurv is None else np.asarray(icurv(s), dtype=float),
        free=np.flatnonzero(free),
    )


def _components(plan: SplinePlan) -> tuple:
    """``(field, radial table, angular table)`` for R, R_t, R_z, R_rho, Z_t, Z_z, Z_rho, L_t, L_z."""

    m = np.asarray(plan.state.m, dtype=float)[:, None]
    n = np.asarray(plan.state.n, dtype=float)[:, None]
    c, s, v, d = plan.cos, plan.sin, plan.value, plan.drho
    return (
        (0, v, c), (0, v, -m * s), (0, v, n * s), (0, d, c),
        (1, v, m * c), (1, v, -n * c), (1, d, s),
        (2, v, m * c), (2, v, -n * c),
    )


def _fields(plan: SplinePlan, full: Array) -> tuple[Array, Array]:
    """The nine geometry/lambda components on the grid, and ``chi'`` per surface."""

    mn, nb = plan.value.shape[0], plan.value.shape[2]
    blocks = full[: 3 * mn * nb].reshape(3, mn, nb)
    u = [jnp.einsum("jqk,jk->qj", radial, blocks[f]) @ angular for f, radial, angular in _components(plan)]
    return jnp.stack(u), jnp.asarray(plan.basis) @ full[3 * mn * nb :]


def _density(plan: SplinePlan, u: Array, chip: Array, rho: Array, phip: Array, pressure: Array) -> Array:
    """``(B^2/2mu0 - p) |sqrt g| d rho / ds`` at points; ``u`` leads with the nine components."""

    R, R_t, R_z, R_r, Z_t, Z_z, Z_r, L_t, L_z = u
    nfp = plan.state.nfp
    jac = plan.state.jacobian_sign * (R / nfp) * (R_t * Z_r - R_r * Z_t)
    bt = chip / nfp - phip * L_z
    bz = phip * (1.0 + L_t)
    quad = (R_t**2 + Z_t**2) * bt**2 + 2 * (R_t * R_z + Z_t * Z_z) * bt * bz
    quad = quad + (R_z**2 + Z_z**2 + (R / nfp) ** 2) * bz**2
    # (B^2/2mu0) |sqrt g| drho = (2 rho / |sqrt g|) quad / (2 mu0) ds
    return rho * quad / (MU0 * jac) - pressure * jac / (2.0 * rho)


def _signed_jacobian(plan: SplinePlan, u: Array) -> Array:
    """``signgs * sqrt(g)`` in ``(rho, theta, zeta)``; positive when nested."""

    R, R_t, _, R_r, Z_t, _, Z_r = u[:7]
    return plan.state.jacobian_sign * (R / plan.state.nfp) * (R_t * Z_r - R_r * Z_t)


def _radial(plan: SplinePlan) -> tuple[Array, Array, Array]:
    return tuple(jnp.asarray(a)[:, None] for a in (plan.rho, plan.phip, plan.pressure))


def spline_energy(plan: SplinePlan, full: Array) -> Array:
    """The functional ``L`` of the module docstring for packed coefficients."""

    u, chip = _fields(plan, full)
    density = _density(plan, u, chip[:, None], *_radial(plan))
    energy = jnp.sum(jnp.asarray(plan.weight)[:, None] * density) * plan.cell
    if plan.icurv is not None:
        energy = energy - (2 * np.pi) ** 2 / MU0 * jnp.sum(jnp.asarray(plan.weight * plan.icurv) * chip)
    return energy


def spline_hessian(plan: SplinePlan, full: Array) -> Array:
    """Exact Hessian of :func:`spline_energy` over the free coefficients.

    ``L = sum_p w_p f(A_p x)`` with ``A`` the linear synthesis, so
    ``H = A^T diag(w f'') A``.  The pointwise 10x10 ``f''`` comes from JAX;
    each block is contracted angle-first, then radially, which costs
    ``O(Nq mn^2 (Nang + nb^2))`` per component pair instead of one
    Hessian-vector product per unknown.
    """

    u, chip = _fields(plan, full)
    rho, phip, pressure = (jnp.broadcast_to(a, u.shape[1:]) for a in _radial(plan))
    chip = jnp.broadcast_to(chip[:, None], u.shape[1:])
    point = jnp.concatenate([u, chip[None]]).reshape(10, -1).T

    def f(y, r, ph, pr):
        return _density(plan, y[:9], y[9], r, ph, pr)

    K = jax.vmap(jax.hessian(f))(point, rho.ravel(), phip.ravel(), pressure.ravel())
    K = K.reshape(u.shape[1:] + (10, 10)) * (jnp.asarray(plan.weight)[:, None, None, None] * plan.cell)
    mn, nq, nb = plan.value.shape
    ones = (3, plan.basis[None], np.ones((1, plan.cos.shape[1])))  # chi': one mode, angle independent
    comps = _components(plan) + (ones,)
    size = (3 * mn + 1) * nb
    H = jnp.zeros((size, size))
    for a, (fa, ra, aa) in enumerate(comps):
        for b, (fb, rb, ab) in enumerate(comps[a:], start=a):
            M = jnp.einsum("it,qt,jt->qij", aa, K[..., a, b], ab)
            block = jnp.einsum("iqk,qij,jql->ikjl", ra, M, rb).reshape(ra.shape[0] * nb, rb.shape[0] * nb)
            sa, sb = fa * mn * nb, fb * mn * nb
            H = H.at[sa : sa + block.shape[0], sb : sb + block.shape[1]].add(block)
            if b != a:
                H = H.at[sb : sb + block.shape[1], sa : sa + block.shape[0]].add(block.T)
    return H[np.ix_(plan.free, plan.free)]


def unpack_state(plan: SplinePlan, full: Array) -> HighOrderEquilibriumState:
    """Write packed coefficients back into a native state."""

    mn, nb = plan.value.shape[0], plan.value.shape[2]
    blocks = full[: 3 * mn * nb].reshape(3, mn, nb)
    return replace(plan.state, R_cos=blocks[0], Z_sin=blocks[1], L_sin=blocks[2], chipf=full[3 * mn * nb :])


def initial_spline_state(inp: Any, basis: BSplineBasis, *, reguess_axis: bool = True) -> HighOrderEquilibriumState:
    """VMEC's interior guess (``rho**m`` boundary scaling) lifted into ``basis``.

    ``reguess_axis`` applies VMEC2000's magnetic-axis guess first (the retry
    the legacy solver takes after a bad initial Jacobian).
    """

    from .solver import _initial_state, prepare_runtime, resolution_from_input, reguess_initial_axis
    from .strong_force import lift_high_order_state

    ns = max(2 * basis.size + 1, 17)
    runtime = prepare_runtime(inp, resolution_from_input(inp, ns=ns))
    state = _initial_state(runtime.setup)
    if reguess_axis:
        runtime, state, _ = reguess_initial_axis(runtime, state)
    return lift_high_order_state(state, runtime, inp=inp, radial_basis=basis)


def current_profile(inp: Any, signgs: int):
    """``s -> signgs mu0 I(s) / 2 pi`` (VMEC ``icurv``) for ``NCURR = 1``."""

    from . import profiles

    edge = float(profiles.current(inp.pcurr_type, inp.ac, inp.ac_aux_s, inp.ac_aux_f, 1.0, bloat=inp.bloat))
    scale = 0.0 if edge == 0.0 else signgs * MU0 * float(inp.curtor) / (2 * np.pi * edge)
    return lambda s: scale * np.asarray(
        profiles.current(inp.pcurr_type, inp.ac, inp.ac_aux_s, inp.ac_aux_f, s, bloat=inp.bloat)
    )


@functools.partial(jax.jit, static_argnums=0)
def _evaluate(plan: SplinePlan, full: Array, x: Array, scale: Array):
    full = full.at[plan.free].set(x)
    value, grad = jax.value_and_grad(lambda y: spline_energy(plan, full.at[plan.free].set(y)))(x)
    nested = jnp.min(_signed_jacobian(plan, _fields(plan, full)[0])) > 0.0
    return value, grad, jnp.linalg.norm(scale * grad), nested


@functools.partial(jax.jit, static_argnums=0)
def _eigensystem(plan: SplinePlan, full: Array, x: Array):
    H = spline_hessian(plan, full.at[plan.free].set(x))
    scale = 1.0 / jnp.sqrt(jnp.abs(jnp.diag(H)) + jnp.finfo(H.dtype).tiny)
    return (scale,) + tuple(jnp.linalg.eigh(scale[:, None] * H * scale[None, :]))


@jax.jit
def _newton_step(scale, eigenvalues, vectors, grad, mu, cutoff):
    size = jnp.abs(eigenvalues)
    inverse = jnp.where(size > cutoff * size.max(), 1.0 / (size + mu * size.max()), 0.0)
    return -scale * (vectors @ (inverse * (vectors.T @ (scale * grad))))


def solve_spline(
    plan: SplinePlan,
    full: Array | None = None,
    *,
    rtol: float = 1.0e-13,
    cutoff: float = 1.0e-14,
    max_iterations: int = 60,
    verbose: bool = False,
) -> tuple[Array, dict]:
    """Minimize ``L`` by Levenberg-damped Newton steps on the exact Hessian.

    The diagonally scaled Hessian (:func:`spline_hessian`) is eigendecomposed
    once per iteration; the step is ``-V (|Lambda| + mu max|Lambda|)^{-1} V^T g``
    (negative curvature reflected, eigenvalues below ``cutoff`` dropped), with
    ``mu`` raised on rejection and lowered on acceptance.  A step is accepted
    when the Jacobian stays nested and the energy or the scaled force norm
    decreases: near the minimum the energy is flat to round-off along the
    near-null poloidal relabelings while the force is not.  Convergence is
    the Newton decrement ``|g.dx| / |L|`` below ``rtol``.
    """

    full = _pack(plan.state) if full is None else jnp.asarray(full)
    x = full[plan.free]
    mu, history, started = 1.0e-4, [], perf_counter()
    for iteration in range(max_iterations):
        scale, eigenvalues, vectors = _eigensystem(plan, full, x)
        value, grad, gnorm, _ = _evaluate(plan, full, x, scale)
        while True:
            dx = _newton_step(scale, eigenvalues, vectors, grad, mu, cutoff)
            trial = _evaluate(plan, full, x + dx, scale)
            if bool(trial[3]) and (float(trial[0]) <= float(value) or float(trial[2]) < float(gnorm)):
                break
            mu *= 10.0
            if mu > 1e6:
                break
        if mu > 1e6:
            break
        decrement = abs(float(grad @ dx)) / abs(float(value))
        x, mu = x + dx, max(mu / 10.0, 1e-14)
        history.append((float(trial[0]), decrement, float(trial[2]), int(jnp.sum(eigenvalues < 0))))
        if verbose:
            print(f" spline newton {iteration:3d}  L = {float(trial[0]):.15e}  decrement = {decrement:.3e}"
                  f"  |g| = {float(trial[2]):.3e}  negative = {history[-1][3]}  mu = {mu:.1e}")
        if decrement < rtol:
            break
    info = dict(iterations=iteration + 1, seconds=perf_counter() - started, history=history)
    return full.at[plan.free].set(x), info
