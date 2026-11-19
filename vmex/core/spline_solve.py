r"""Galerkin B-spline radial discretization of the fixed-boundary MHD energy.

Every Fourier amplitude is ``rho**|m| q(s)`` with ``q`` a clamped B-spline in
``s = rho**2`` (:mod:`vmex.core.radial_basis`), so axis regularity holds by
construction.  The unknowns are the spline coefficients of ``R_mn``,
``Z_mn``, ``lambda_mn`` and, for a prescribed toroidal current, of
``chi'(s)``.  The discrete functional is

.. math::

    L = \int \Big(\frac{B^2}{2\mu_0} - p(s)\Big) dV
        + \frac{4\pi^2}{\mu_0}\int_0^1 \bar I(s)\,\chi'(s)\,ds,

with ``bar I = signgs mu0 I_tor / 2 pi`` (VMEC ``icurv``); the last term is
present only for ``NCURR = 1`` and makes stationarity in ``chi'`` the current
constraint, because ``dW/dchi'(s) = -(4 pi^2/mu0) bar I(s) = -2 pi I(s)``
in VMEC's sign conventions (checked against converged VMEX solutions).  Quadrature is
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


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=["template", "rho", "weight", "value", "drho", "basis", "cos", "sin", "phip", "pressure", "icurv", "free"],
    meta_fields=["m", "n", "nfp", "jacobian_sign", "cell"],
)
@dataclass(frozen=True)
class SplinePlan:
    """Quadrature tables and the free-coefficient layout.

    A pytree whose arrays are leaves, so jitted kernels compile once per
    shape (resolution), not once per plan.
    """

    m: tuple
    n: tuple
    nfp: int
    jacobian_sign: int
    template: Array  # packed coefficients of the initial state (fixed values included)
    rho: np.ndarray  # (Nq,)
    weight: np.ndarray  # (Nq,) Gauss weights in s
    cell: float  # angular cell area times nfp
    value: np.ndarray  # (mnmax, Nq, nbasis): rho^m B_k(s)
    drho: np.ndarray  # (mnmax, Nq, nbasis): d/drho of rho^m B_k(s)
    basis: np.ndarray  # (Nq, nchi): chi' synthesis; B_k(s), or the identity for node values
    cos: np.ndarray  # (mnmax, Nang)
    sin: np.ndarray  # (mnmax, Nang)
    phip: np.ndarray  # (Nq,)
    pressure: np.ndarray  # (Nq,)
    icurv: np.ndarray | None  # (Nq,) for NCURR = 1, else None
    free: Array  # flat indices into the packed coefficient vector

    @property
    def size(self) -> int:
        """Number of free coefficients."""

        return int(self.free.size)


def _to_rho_basis(state: HighOrderEquilibriumState, rho_basis: BSplineBasis) -> HighOrderEquilibriumState:
    """Geometry/lambda tables with q(rho) = q(s = rho^2), fitted at the collocation nodes of ``rho_basis``."""

    nodes = rho_basis.collocation_nodes
    fit = np.linalg.solve(np.asarray(rho_basis.basis_matrix(nodes)), np.asarray(state.radial_basis.basis_matrix(nodes**2)))
    tables = {name: jnp.asarray(getattr(state, name)) @ fit.T for name in ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")}
    # The tables no longer match state.radial_basis, so skip its shape check:
    # only _pack reads this copy.
    new = object.__new__(type(state))
    new.__dict__.update(state.__dict__, **tables)
    return new


def _pack(state: HighOrderEquilibriumState, chi_nodes: Array | None = None) -> Array:
    """Packed R_cos, Z_sin, L_sin and chi' (spline coefficients, or values ``chi_nodes @ chipf``)."""

    chi = state.chipf if chi_nodes is None else jnp.asarray(chi_nodes) @ state.chipf
    return jnp.concatenate([jnp.ravel(state.R_cos), jnp.ravel(state.Z_sin), jnp.ravel(state.L_sin), chi])


def make_spline_plan(
    state: HighOrderEquilibriumState,
    *,
    icurv: Any = None,
    ntheta: int | None = None,
    nzeta: int | None = None,
    radial_order: int | None = None,
    rho_basis: BSplineBasis | None = None,
) -> SplinePlan:
    """Tabulate the basis on the quadrature grid of ``state.radial_basis``.

    ``icurv`` is a callable ``s -> signgs mu0 I(s) / 2 pi`` for a prescribed
    current; ``None`` holds ``chi'`` (the iota profile) fixed.  The default
    angular grid integrates products of four mode sets without aliasing.
    ``rho_basis`` (experimental) puts the geometry and lambda on
    ``rho^|m| q(rho)``, which also admits odd powers of ``rho`` that the
    regular ``rho^|m| q(s)`` excludes; profiles stay on ``state.radial_basis``.
    """

    basis: BSplineBasis = state.radial_basis if rho_basis is None else rho_basis
    m = np.asarray(state.m, dtype=int)
    n = np.asarray(state.n, dtype=int)
    ntheta = int(ntheta or 4 * int(m.max()) + 4)
    nzeta = int(nzeta or (1 if not np.any(n) else 4 * int(np.abs(n).max()) + 4))
    order = int(radial_order or basis.degree + 3)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    breaks = basis.breakpoints
    centre, half = 0.5 * (breaks[:-1] + breaks[1:]), 0.5 * np.diff(breaks)
    x = (centre[:, None] + half[:, None] * nodes).ravel()
    wx = (half[:, None] * weights).ravel()
    s, ws = (x, wx) if rho_basis is None else (x * x, 2.0 * x * wx)
    rho = np.sqrt(s)
    G = np.asarray(basis.basis_matrix(x))
    dG = np.asarray(basis.basis_matrix(x, derivative=1)) * (2.0 * rho if rho_basis is None else np.ones_like(rho))[:, None]
    B = np.asarray(state.radial_basis.basis_matrix(s))
    rm = rho[None, :, None] ** m[:, None, None]
    value = rm * G[None]
    drho = m[:, None, None] * rho[None, :, None] ** np.maximum(m - 1, 0)[:, None, None] * G[None] * (m > 0)[:, None, None]
    drho = drho + rm * dG[None]
    theta = 2 * np.pi * np.arange(ntheta) / ntheta
    zeta = 2 * np.pi * np.arange(nzeta) / nzeta
    phase = m[:, None, None] * theta[None, :, None] - n[:, None, None] * zeta[None, None, :]
    area = (2 * np.pi) ** 2 / (ntheta * nzeta)
    nb = basis.size
    mn = m.size
    free = np.ones((3, mn, nb), dtype=bool)
    free[:2, :, -1] = False  # fixed boundary
    free[1:, (m == 0) & (n == 0)] = False  # sin(0) and the lambda gauge
    # With a prescribed current, chi' takes one value per radial node: stationarity
    # in each node value is the current constraint at that node (VMEC and DESC
    # enforce it pointwise).  A spline chi' only enforces its projection, which
    # misses the near-axis current by 15-90% inside the first span.
    chi_basis = np.eye(s.size) if icurv is not None else B
    free = np.concatenate([free.ravel(), np.full(chi_basis.shape[1], icurv is not None)])
    return SplinePlan(
        m=tuple(int(v) for v in m),
        n=tuple(int(v) for v in n),
        nfp=int(state.nfp),
        jacobian_sign=int(state.jacobian_sign),
        template=_pack(state if rho_basis is None else _to_rho_basis(state, rho_basis), None if icurv is None else B),
        rho=rho,
        weight=ws,
        cell=area * int(state.nfp),
        value=value,
        drho=drho,
        basis=chi_basis,
        cos=np.cos(phase).reshape(mn, -1),
        sin=np.sin(phase).reshape(mn, -1),
        phip=B @ np.asarray(state.phipf),
        pressure=B @ np.asarray(state.pressure),
        icurv=None if icurv is None else np.asarray(icurv(s), dtype=float),
        free=jnp.asarray(np.flatnonzero(free)),
    )


def _components(plan: SplinePlan) -> tuple:
    """``(field, radial table, angular table)`` for R, R_t, R_z, R_rho, Z_t, Z_z, Z_rho, L_t, L_z."""

    m = np.asarray(plan.m, dtype=float)[:, None]
    n = np.asarray(plan.n, dtype=float)[:, None]
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
    nfp = plan.nfp
    jac = plan.jacobian_sign * (R / nfp) * (R_t * Z_r - R_r * Z_t)
    bt = chip / nfp - phip * L_z
    bz = phip * (1.0 + L_t)
    quad = (R_t**2 + Z_t**2) * bt**2 + 2 * (R_t * R_z + Z_t * Z_z) * bt * bz
    quad = quad + (R_z**2 + Z_z**2 + (R / nfp) ** 2) * bz**2
    # (B^2/2mu0) |sqrt g| drho = (2 rho / |sqrt g|) quad / (2 mu0) ds
    return rho * quad / (MU0 * jac) - pressure * jac / (2.0 * rho)


def _signed_jacobian(plan: SplinePlan, u: Array) -> Array:
    """``signgs * sqrt(g)`` in ``(rho, theta, zeta)``; positive when nested."""

    R, R_t, _, R_r, Z_t, _, Z_r = u[:7]
    return plan.jacobian_sign * (R / plan.nfp) * (R_t * Z_r - R_r * Z_t)


def _radial(plan: SplinePlan) -> tuple[Array, Array, Array]:
    return tuple(jnp.asarray(a)[:, None] for a in (plan.rho, plan.phip, plan.pressure))


def spline_energy(plan: SplinePlan, full: Array) -> Array:
    """The functional ``L`` of the module docstring for packed coefficients."""

    u, chip = _fields(plan, full)
    density = _density(plan, u, chip[:, None], *_radial(plan))
    energy = jnp.sum(jnp.asarray(plan.weight)[:, None] * density) * plan.cell
    if plan.icurv is not None:
        energy = energy + (2 * np.pi) ** 2 / MU0 * jnp.sum(jnp.asarray(plan.weight * plan.icurv) * chip)
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
    size = 3 * mn * nb + plan.basis.shape[1]
    H = jnp.zeros((size, size))
    for a, (fa, ra, aa) in enumerate(comps):
        for b, (fb, rb, ab) in enumerate(comps[a:], start=a):
            M = jnp.einsum("it,qt,jt->qij", aa, K[..., a, b], ab)
            block = jnp.einsum("iqk,qij,jql->ikjl", ra, M, rb).reshape(ra.shape[0] * ra.shape[2], rb.shape[0] * rb.shape[2])
            sa, sb = fa * mn * nb, fb * mn * nb
            H = H.at[sa : sa + block.shape[0], sb : sb + block.shape[1]].add(block)
            if b != a:
                H = H.at[sb : sb + block.shape[1], sa : sa + block.shape[0]].add(block.T)
    return H[plan.free][:, plan.free]


def unpack_state(state: HighOrderEquilibriumState, plan: SplinePlan, full: Array) -> HighOrderEquilibriumState:
    """Write packed coefficients into ``state`` (the plan's template state)."""

    mn, nb = plan.value.shape[0], plan.value.shape[2]
    blocks = full[: 3 * mn * nb].reshape(3, mn, nb)
    chi = full[3 * mn * nb :]
    if chi.size != nb:  # node values: least-squares spline for the native state
        nodes = np.asarray(state.radial_basis.basis_matrix(np.asarray(plan.rho) ** 2))
        chi = jnp.linalg.lstsq(jnp.asarray(nodes), chi)[0]
    return replace(state, R_cos=blocks[0], Z_sin=blocks[1], L_sin=blocks[2], chipf=chi)


def initial_spline_state(inp: Any, basis: BSplineBasis, *, reguess_axis: bool = True) -> HighOrderEquilibriumState:
    """VMEC's interior guess (``rho**m`` boundary scaling) lifted into ``basis``.

    ``reguess_axis`` applies VMEC2000's magnetic-axis guess first (the retry
    the legacy solver takes after a bad initial Jacobian).
    """

    from .solver import _initial_state, prepare_runtime, resolution_from_input, reguess_initial_axis
    from .strong_force import lift_high_order_state

    ns = max(2 * basis.size + 1, 17, int(np.ceil(2.0 / np.diff(basis.breakpoints).min())) + 1)  # two samples per span
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


def angle_constraint(plan: SplinePlan, full: Array) -> Array:
    r"""VMEC's spectral-condensation condition on the poloidal angle.

    ``G_i = int ztemp eta_i ds dtheta dzeta`` with
    ``ztemp = (rcon - s rcon_edge) R_theta + (zcon - s zcon_edge) Z_theta``,
    ``rcon = sum m(m-1) R_mn cos``, ``zcon = sum m(m-1) Z_mn sin`` and
    ``eta_i = rho^m B_k(s) sin(m theta - n zeta)`` for ``m <= mpol-2``,
    ``(m, n) != (0, 0)``, away from the boundary (Hirshman and Breslau, Phys. Plasmas 5 2664
    (1998); VMEX ``forces.constraint_force``).  The energy is invariant
    under relabeling the angle only in the continuum; this condition picks
    the compact-spectrum angle, as VMEC does, and the VMEC interior guess
    satisfies it to round-off.
    """

    mn, _, nb = plan.value.shape
    m = np.asarray(plan.m)
    weight = (m * (m - 1)).astype(float)[:, None]
    blocks = full[: 3 * mn * nb].reshape(3, mn, nb)
    s = jnp.asarray(plan.rho**2)[:, None]
    ztemp = 0.0
    for block, table, tangent in ((blocks[0], plan.cos, -m[:, None] * plan.sin), (blocks[1], plan.sin, m[:, None] * plan.cos)):
        amplitude = jnp.einsum("jqk,jk->qj", plan.value, block)
        ztemp = ztemp + (amplitude @ (weight * table) - s * (block[:, -1] @ (weight * table))) * (amplitude @ tangent)
    G = jnp.einsum("jqk,q,qt,jt->jk", plan.value, jnp.asarray(plan.weight), ztemp, plan.sin) * plan.cell
    rows = np.zeros((mn, nb), dtype=bool)
    rows[(m < m.max()) & ((m > 0) | (np.asarray(plan.n) != 0)), :-1] = True
    return G.ravel()[np.flatnonzero(rows.ravel())]


@jax.jit
def _kkt_state(plan: SplinePlan, full: Array, x: Array, scale: Array, row_scale: Array):
    """Energy, gradient, constraint, its Jacobian, multipliers, scaled KKT residual, nestedness."""

    full = full.at[plan.free].set(x)

    def energy(y):
        return spline_energy(plan, full.at[plan.free].set(y))

    def constraint(y):
        return angle_constraint(plan, full.at[plan.free].set(y))

    value, grad = jax.value_and_grad(energy)(x)
    G, C = constraint(x), jax.jacrev(constraint)(x)
    Cs = C * scale
    nu = jnp.linalg.solve(Cs @ Cs.T, -Cs @ (scale * grad))
    residual = jnp.linalg.norm(scale * (grad + C.T @ nu)) + jnp.linalg.norm(row_scale * G)
    nested = jnp.min(_signed_jacobian(plan, _fields(plan, full)[0])) > 0.0
    return value, grad, G, C, nu, residual, nested


@jax.jit
def _kkt_matrix(plan: SplinePlan, full: Array, x: Array, C: Array, nu: Array):
    """Diagonally scaled Hessian of ``L`` and row-normalized constraint Jacobian.

    The constraint curvature ``nu.grad^2 G`` is left out: ``nu`` is of the
    order of the discrete relabeling asymmetry, and keeping it costs one
    Hessian-vector product per unknown for no measured gain in iterations.
    """

    full = full.at[plan.free].set(x)
    H = spline_hessian(plan, full)
    scale = 1.0 / jnp.sqrt(jnp.abs(jnp.diag(H)) + jnp.finfo(H.dtype).tiny)
    Cs = C * scale[None, :]
    row_scale = 1.0 / (jnp.linalg.norm(Cs, axis=1) + jnp.finfo(H.dtype).tiny)
    return scale[:, None] * H * scale[None, :], Cs * row_scale[:, None], scale, row_scale


@jax.jit
def _kkt_step(Hs: Array, Cs: Array, scale: Array, row_scale: Array, grad: Array, G: Array, mu: Array):
    nc = G.size
    K = jnp.block([[Hs + mu * jnp.diag(jnp.abs(jnp.diag(Hs))), Cs.T], [Cs, jnp.zeros((nc, nc))]])
    return scale * jnp.linalg.solve(K, -jnp.concatenate([scale * grad, row_scale * G]))[: grad.size]


def solve_spline(
    plan: SplinePlan,
    full: Array | None = None,
    *,
    tolerance: float = 1.0e-10,
    max_iterations: int = 40,
    verbose: bool = False,
) -> tuple[Array, dict]:
    """Minimize ``L`` subject to :func:`angle_constraint` by Newton-KKT steps.

    Each step solves ``[[H + mu D, C^T], [C, 0]] [dx, nu] = -[g, G]`` in
    diagonal scaling, with ``H`` the exact Hessian of ``L``
    (:func:`spline_hessian`), ``C = dG/dx`` and a
    Levenberg shift ``mu`` raised on rejection.  A step is accepted when the
    Jacobian stays nested and the energy or the scaled KKT residual
    ``|D (g + C^T nu)| + |G|`` decreases; convergence is that residual below
    ``tolerance``.  Without the angle condition the relabeling directions are
    near-null and Newton converges only linearly.
    """

    full = plan.template if full is None else jnp.asarray(full)
    x = full[plan.free]
    rows = jnp.ones(angle_constraint(plan, full).size)
    value, grad, G, C, nu, _, _ = _kkt_state(plan, full, x, jnp.ones(x.size), rows)
    _, _, scale, row_scale = _kkt_matrix(plan, full, x, C, nu)  # fixed scaling of the residual
    value, grad, G, C, nu, residual, _ = _kkt_state(plan, full, x, scale, row_scale)
    mu, history, started = 1.0e-4, [], perf_counter()
    for iteration in range(max_iterations):
        system = _kkt_matrix(plan, full, x, C, nu)
        while True:
            dx = _kkt_step(*system, grad, G, mu)
            trial = _kkt_state(plan, full, x + dx, scale, row_scale)
            if bool(trial[6]) and (float(trial[5]) < float(residual) or float(trial[0]) < float(value)):
                break
            mu *= 10.0
            if mu > 1e6:
                break
        if mu > 1e6:
            break
        x, mu = x + dx, max(mu / 10.0, 1e-14)
        value, grad, G, C, nu, residual, _ = trial
        history.append((float(value), float(residual)))
        if verbose:
            print(f" spline newton {iteration:3d}  L = {float(value):.15e}  KKT residual = {float(residual):.3e}  mu = {mu:.1e}")
        if float(residual) < tolerance:
            break
    info = dict(iterations=iteration + 1, seconds=perf_counter() - started, history=history, multipliers=nu)
    return full.at[plan.free].set(x), info


def boundary_indices(plan: SplinePlan) -> np.ndarray:
    """Packed indices of the fixed R/Z boundary coefficients (the last clamped one per mode)."""

    mn, _, nb = plan.value.shape
    return np.concatenate([np.arange(mn) * nb + nb - 1, mn * nb + np.arange(mn) * nb + nb - 1])


def implicit_gradient(plan: SplinePlan, full: Array, objective, parameters: np.ndarray, multipliers: Array) -> Array:
    r"""``d objective(full*) / d full[parameters]`` through the constrained equilibrium.

    The solution satisfies ``F = [grad_x (L + nu.G), G] = 0``.  With ``K`` the
    symmetric KKT matrix of :func:`solve_spline` and ``K a = [J_x, 0]``, the
    implicit-function theorem gives ``dJ/dp = J_p - d/dp (a_x.grad_x(L + nu.G) + a_nu.G)``;
    one solve serves every parameter.  ``parameters`` are packed indices
    outside ``plan.free``, such as :func:`boundary_indices`; ``multipliers``
    is ``info["multipliers"]`` from the solve.
    """

    free, parameters = jnp.asarray(plan.free), jnp.asarray(parameters)

    @jax.jit
    def gradient(full, nu):
        x, p = full[free], full[parameters]

        def pack(x, p):
            return full.at[free].set(x).at[parameters].set(p)

        def constraint(x, p):
            return angle_constraint(plan, pack(x, p))

        def lagrangian(x, p):
            return spline_energy(plan, pack(x, p)) + nu @ constraint(x, p)

        J_x, J_p = jax.grad(lambda x, p: objective(pack(x, p)), argnums=(0, 1))(x, p)
        H = spline_hessian(plan, full) + jax.hessian(lambda y: nu @ constraint(y, p))(x)
        C = jax.jacrev(constraint)(x, p)
        nc = C.shape[0]
        K = jnp.block([[H, C.T], [C, jnp.zeros((nc, nc))]])
        adjoint = jnp.linalg.solve(K, jnp.concatenate([J_x, jnp.zeros(nc)]))
        a_x, a_nu = adjoint[: x.size], adjoint[x.size :]

        def residual_dot(p):
            return jax.grad(lagrangian)(x, p) @ a_x + constraint(x, p) @ a_nu

        return J_p - jax.grad(residual_dot)(p)

    return gradient(jnp.asarray(full), jnp.asarray(multipliers))


def solve_spline_ladder(inp: Any, spans: tuple[int, ...] = (2, 4, 8), *, degree: int = 5, verbose: bool = False, **plan_options):
    """Solve on uniform knots ``spans[0]``, then refine by exact knot insertion.

    Each level starts from the previous solution prolonged by Boehm knot
    insertion (:func:`~vmex.core.strong_force.insert_high_order_state_knots`),
    so only the first level pays for the global Newton path.  Returns the
    final ``(plan, full, info)`` with per-level iteration counts in ``info``.
    """

    from .strong_force import insert_high_order_state_knots

    state = initial_spline_state(inp, BSplineBasis.clamped(np.linspace(0.0, 1.0, spans[0] + 1), degree=degree))
    icurv = current_profile(inp, state.jacobian_sign) if int(inp.ncurr) == 1 else None
    levels, full, plan = [], None, None
    for index, count in enumerate(spans):
        if index:
            previous = unpack_state(state, plan, full)
            new = np.setdiff1d(np.linspace(0.0, 1.0, count + 1), previous.radial_basis.breakpoints)
            state = insert_high_order_state_knots(previous, new)
        plan = make_spline_plan(state, icurv=icurv, **plan_options)
        full, info = solve_spline(plan, verbose=verbose)
        levels.append((count, plan.size, info["iterations"], info["seconds"]))
        if verbose:
            print(f" spline ladder: {count} spans, {plan.size} unknowns, {info['iterations']} Newton steps, {info['seconds']:.2f} s")
    info["levels"] = levels
    return plan, full, info
