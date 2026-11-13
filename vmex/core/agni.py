"""Equilibrium adapter from a VMEX ``(state, runtime)`` pair to AGNI.

:func:`make_agni_map` returns a pure, traceable map from a converged
``(SpectralState, SolverRuntime)`` pair to an ``agnimhd.EquilibriumData`` on a
fixed AGNI node set.  The map is differentiable in the state and in the live
runtime arrays (boundary, flux, pressure), so it can be handed to
``agnimhd.growth_rate(..., equilibrium_map=...)`` for gradients of the
finite-n growth rate.  It performs no equilibrium solve and no file I/O.

Conventions (SI throughout)
---------------------------
AGNI works in PEST coordinates ``(rho, v, phi)`` with ``rho = sqrt(s)``, ``phi``
the geometric toroidal angle on one field period ``[0, 2*pi/nfp)`` and ``v`` the
straight-field-line poloidal angle.  VMEX has ``signgs = -1`` (left-handed
``(s, u, zeta)``) and the lambda convention ``theta* = u + lambda``; the AGNI
chart is therefore ``v = sigma * (u + lambda)`` with ``sigma = signgs``, so that
``(rho, v, phi)`` is right-handed in the sense of the stored ``sqrtg`` sign
convention (a single global sign, checked by ``EquilibriumData.validate``).
Consequently

- ``Psi = sigma * 2*pi * psi_edge = phiedge`` is the signed toroidal flux, and
  ``psi_r = rho * Psi / pi``, ``psi_rr = Psi / pi``;
- ``iota = sigma * chi'(s) / phi'(s)``;
- ``B = (psi_r / sqrtg) (iota e_v + e_phi)`` with ``e_a = dX/da`` at fixed PEST
  coordinates;
- ``sqrtg`` and its three partials are the determinant of the AGNI-chart
  Jacobian and its derivatives.  Every derivative is an explicit contraction of
  the spline/Fourier tables over all nodes at once (no nested autodiff), and the
  PEST root ``u + lambda(rho, u, nfp*phi) = sigma * v`` is differentiated by the
  implicit function theorem: ``sqrtg = sigma*nfp*det(dX/dx)/(1 + lambda_u)``.  The
  root is a vectorized Newton whose last step starts from a stop-gradient
  iterate, so reverse mode sees only the implicit derivative and the map jits
  to a value-and-gradient in seconds;
- ``J = curl B / mu0`` comes from the high-order lift's strong-force kernel
  (:mod:`vmex.core.strong_force`); ``J^phi = J . grad(phi)``;
- the instability drive is delivered as the Cartesian vectors
  ``J x grad(rho)`` and ``(B . grad) grad(rho)``;
- ``a = sqrt(<A> / pi)`` with ``<A>`` the toroidal average of the enclosed
  cross-sectional area, ``A(phi) = int |e_rho x e_u| drho du`` at fixed ``phi``
  (full-cross-section quadrature, not the boundary Aminor of the wout).

Only the identity ``APHI`` map, ``lasym = False`` and ``lflip = False`` are
supported; anything else raises at construction.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from . import strong_force as sf

Array = Any

__all__ = ["make_agni_map"]


def _check_supported(runtime: Any, inp: Any | None) -> None:
    """Reject decks outside the validated envelope (host-side, concrete runtime)."""
    setup = runtime.setup
    if bool(setup.lasym):
        raise NotImplementedError("the AGNI adapter has not been validated for lasym=True")
    if bool(setup.lflip):
        raise NotImplementedError("the AGNI adapter does not support lflip=True decks")
    if inp is not None:
        aphi = np.asarray(inp.aphi, dtype=float).ravel()
        if aphi.size == 0 or aphi[0] != 1.0 or np.any(aphi[1:]):
            raise NotImplementedError("the AGNI adapter supports only the identity APHI map [1, 0, ...]")
    # The runtime keeps no APHI, but under the identity map the toroidal-flux
    # derivative is the constant psi_edge on both meshes.
    psi_edge = float(np.asarray(setup.psi_edge))
    phipf = np.asarray(setup.phipf, dtype=float)
    phips = np.asarray(setup.phips, dtype=float)[1:]
    scale = max(abs(psi_edge), np.finfo(float).tiny)
    if max(np.abs(phipf - psi_edge).max(), np.abs(phips - psi_edge).max()) > 1e-12 * scale:
        raise NotImplementedError("the AGNI adapter supports only the identity APHI map [1, 0, ...]")


@functools.lru_cache(maxsize=16)
def _spline_matrices(basis: Any, s: tuple[float, ...]) -> np.ndarray:
    """``d^d B_b/ds^d`` at the static points ``s`` for ``d = 0, 1, 2``, by de Boor's local recursion.

    This is the evaluator the high-order lift itself differentiates; the global
    ``basis_matrix`` recursion loses about ``1e-14`` at the right endpoint.
    """
    def row(x):
        return basis.evaluate(jnp.eye(basis.size), x, axis=-1)

    with jax.ensure_compile_time_eval():
        points = jnp.asarray(s)
        return np.stack([np.asarray(jax.vmap(f)(points)) for f in (row, jax.jacfwd(row), jax.jacfwd(jax.jacfwd(row)))])


def _radial(hs: Any, rho: np.ndarray) -> tuple[Array, Array]:
    """Radial amplitudes and spline derivatives at the static radii ``rho``.

    ``A[t, f, a, i, k] = d^a/drho^a [rho^m q(rho^2)]`` for the cos/sin tables ``t``
    and the families ``f = (R, Z, lambda)``; ``Q[d, i, b] = d^d B_b(rho^2)/drho^d``.
    """
    s = rho ** 2
    b0, b1, b2 = _spline_matrices(hs.radial_basis, tuple(s))
    Q = jnp.asarray(np.stack([b0, 2.0 * rho[:, None] * b1, 2.0 * b1 + 4.0 * s[:, None] * b2]))
    m = np.asarray(hs.m, dtype=float)
    power = [rho[:, None] ** m, m * rho[:, None] ** np.maximum(m - 1.0, 0.0),
             m * (m - 1.0) * rho[:, None] ** np.maximum(m - 2.0, 0.0)]
    tables = jnp.stack([jnp.stack([hs.R_cos, hs.Z_cos, hs.L_cos]), jnp.stack([hs.R_sin, hs.Z_sin, hs.L_sin])])
    q = jnp.einsum("tfkb,dib->dtfik", tables, Q)
    amplitude = [power[0] * q[0], power[1] * q[0] + power[0] * q[1],
                 power[2] * q[0] + 2.0 * power[1] * q[1] + power[0] * q[2]]
    return jnp.stack(amplitude, axis=2), Q


def _jets(hs: Any, A: Array, u: Array, z: Array, orders: list[tuple[int, int, int]]) -> list[Array]:
    """Spectral derivatives ``d^a_rho d^b_u d^c_zeta`` of the families in ``A`` at ``(rho_i, u_im, z_im)``."""
    m, n = np.asarray(hs.m, dtype=float), np.asarray(hs.n, dtype=float)
    theta = m * u[..., None] - n * z[..., None]
    c, s = jnp.cos(theta), jnp.sin(theta)
    trig = ((c, s), (-s, c), (-c, -s))  # derivatives of (cos, sin) by order
    out = []
    for a, b, k in orders:
        factor = m ** b * (-n) ** k
        cos_part, sin_part = trig[b + k]
        out.append(jnp.einsum("fik,imk->fim", A[0, :, a] * factor, cos_part)
                   + jnp.einsum("fik,imk->fim", A[1, :, a] * factor, sin_part))
    return out


def _unit(*axes: int) -> tuple[int, int, int]:
    index = [0, 0, 0]
    for axis in axes:
        index[axis] += 1
    return tuple(index)


def _frame(hs: Any, A: Array, u: Array, z: Array) -> tuple[Array, ...]:
    """Position ``X``, ``E = dX/d(rho, u, zeta)``, ``H = d^2X`` and the lambda jet at the native points.

    Everything is flattened to ``N = u.size`` points; ``H[n, a, j, k] = d^2 X_a/dx_j dx_k``.
    """
    orders = [(0, 0, 0)] + [_unit(j) for j in range(3)] + [_unit(j, k) for j in range(3) for k in range(3)]
    jets = dict(zip(orders, _jets(hs, A, u, z, orders)))
    val = jets[orders[0]].reshape(3, -1)
    d1 = jnp.stack([jets[_unit(j)].reshape(3, -1) for j in range(3)], axis=-1)
    d2 = jnp.stack([jnp.stack([jets[_unit(j, k)].reshape(3, -1) for k in range(3)], -1) for j in range(3)], -2)
    phi = z.reshape(-1) / float(hs.nfp)
    cs, dcs = jnp.stack([jnp.cos(phi), jnp.sin(phi)], -1), jnp.stack([-jnp.sin(phi), jnp.cos(phi)], -1) / hs.nfp
    zeta = np.eye(3)[2]  # only zeta moves the toroidal angle
    w1 = dcs[:, :, None] * zeta
    w2 = -cs[:, :, None, None] / float(hs.nfp) ** 2 * zeta[:, None] * zeta
    R, R1, R2 = val[0], d1[0], d2[0]
    xy1 = R1[:, None, :] * cs[:, :, None] + R[:, None, None] * w1
    xy2 = (R2[:, None] * cs[:, :, None, None] + R1[:, None, :, None] * w1[:, :, None, :]
           + R1[:, None, None, :] * w1[:, :, :, None] + R[:, None, None, None] * w2)
    X = jnp.concatenate([R[:, None] * cs, val[1][:, None]], axis=-1)
    E = jnp.concatenate([xy1, d1[1][:, None, :]], axis=1)
    H = jnp.concatenate([xy2, d2[1][:, None]], axis=1)
    return X, E, H, val[2], d1[2], d2[2]


def _columns(E: Array) -> tuple[Array, Array]:
    """Return ``det(E)`` and ``E^-1`` (rows are the dual vectors) from the columns of ``E``."""
    e0, e1, e2 = E[..., 0], E[..., 1], E[..., 2]
    dual = jnp.stack([jnp.cross(e1, e2), jnp.cross(e2, e0), jnp.cross(e0, e1)], axis=1)
    det = jnp.sum(e0 * dual[:, 0], axis=-1)
    return det, dual / det[:, None, None]


def _minor_radius(hs: Any, quadrature: tuple[int, int, int]) -> Array:
    """Zeta-averaged enclosed cross-section area, as an effective radius ``sqrt(<A>/pi)``."""
    n_rho, n_u, n_z = quadrature
    x, w = np.polynomial.legendre.leggauss(n_rho)
    rho, w = 0.5 * (x + 1.0), 0.5 * w
    u = np.linspace(0.0, 2.0 * np.pi, n_u, endpoint=False)
    z = np.linspace(0.0, 2.0 * np.pi, n_z, endpoint=False)  # nfp*phi, one period
    _, U, Z = np.meshgrid(rho, u, z, indexing="ij")
    E = _frame(hs, _radial(hs, rho)[0], jnp.asarray(U.reshape(n_rho, -1)), jnp.asarray(Z.reshape(n_rho, -1)))[1]
    dA = jnp.linalg.norm(jnp.cross(E[..., 0], E[..., 1]), axis=-1).reshape(n_rho, n_u, n_z)
    area = jnp.einsum("i,ijk->k", jnp.asarray(w), dA) * (2.0 * np.pi / n_u)
    return jnp.sqrt(jnp.mean(area) / np.pi)


def make_agni_map(
    runtime: Any,
    nodes: dict,
    *,
    density: Any = None,
    radius_quadrature: tuple[int, int, int] = (24, 48, 8),
    newton_iterations: int = 30,
    inp: Any | None = None,
) -> Callable[..., Any]:
    """Build ``map(pair) -> agnimhd.EquilibriumData`` on a fixed AGNI node set.

    Parameters
    ----------
    runtime
        Concrete ``SolverRuntime`` fixing the radial grid, mode table and
        boundary-condition rules; only static data (the lift projectors, the
        node tensor, the radius quadrature) is closed over.  The live arrays
        always come from the ``pair`` passed to the map.
    nodes
        ``agnimhd.Basis.nodes_and_diffmat(nfp, family)[0]``: 1-D ``rho``,
        ``theta`` (PEST ``v``) and ``zeta`` (geometric ``phi``) node arrays;
        the tensor is built rho-major, ``n = (i*n_theta + j)*n_zeta + k``.
    density
        Optional mass-density weight: an array with one entry per node, or a
        callable of the node ``rho`` array returning that array.  ``None``
        leaves the weight unset (AGNI uses 1).
    radius_quadrature
        ``(n_rho, n_u, n_zeta)`` Gauss/trapezoid points for the cross-section
        area that defines ``a``.
    newton_iterations
        Fixed Newton trip count of the PEST root.
    inp
        Optional ``VmecInput`` used to check ``APHI`` explicitly; without it the
        identity map is inferred from the runtime flux profiles.

    Returns
    -------
    ``map(pair, extras=False)``; ``pair = (state, runtime)``.  With
    ``extras=True`` it returns ``(eq, extras)`` where ``extras`` holds the
    per-node PEST-root residual, ``B``, ``J``, ``e`` (the Jacobian columns
    ``e_rho, e_v, e_phi``, shape ``(n, 3, 3)``), ``B . grad(phi)`` and the
    ``(n, 3)`` node coordinates.

    Raises
    ------
    NotImplementedError
        For ``lasym``, ``lflip`` or a non-identity ``APHI``.
    """
    from agnimhd import EquilibriumData  # optional dependency, imported lazily

    _check_supported(runtime, inp)
    nfp = int(runtime.resolution.nfp)
    rho_n, v_n, phi_n = (np.asarray(nodes[k], dtype=float) for k in ("rho", "theta", "zeta"))
    if phi_n.min() < 0.0 or phi_n.max() >= 2.0 * np.pi / nfp:
        raise ValueError("nodes['zeta'] must be the geometric angle on one field period [0, 2*pi/nfp)")
    shape = (rho_n.size, v_n.size, phi_n.size)
    R, V, P = np.meshgrid(rho_n, v_n, phi_n, indexing="ij")
    points = jnp.asarray(np.stack([R.ravel(), V.ravel(), P.ravel()], axis=1))
    weight = None
    if density is not None:
        weight = density(points[:, 0]) if callable(density) else density
        weight = jnp.asarray(weight, dtype=float).reshape(-1)
        if weight.size != points.shape[0]:
            raise ValueError("density must have one entry per node")
    lift = sf.traced_lift(runtime)
    sigma = float(runtime.setup.signgs)
    cells = points.shape[0] // rho_n.size
    v2, z2 = (jnp.asarray(points[:, k]).reshape(rho_n.size, cells) for k in (1, 2))
    z2 = nfp * z2  # native toroidal angle zeta = nfp*phi
    e_rho = jnp.asarray(np.eye(3)[0])
    rho = points[:, 0]
    dot = lambda a, b: jnp.sum(a * b, axis=-1)  # noqa: E731

    def fields(hs: Any) -> dict[str, Array]:
        A, Q = _radial(hs, rho_n)
        lam_table = A[:, 2:3]

        def residual(table, u):
            lam, lam_u = _jets(hs, table, u, z2, [(0, 0, 0), (0, 1, 0)])
            return u + lam[0] - sigma * v2, 1.0 + lam_u[0]

        # PEST root u + lambda(rho, u, nfp*phi) = sigma*v.  Newton on frozen coefficients, then one live
        # step from the stop-gradient iterate: its derivative is the implicit-function one.
        frozen = jax.lax.stop_gradient(lam_table)
        u = jax.lax.stop_gradient(jax.lax.fori_loop(
            0, int(newton_iterations), lambda _, u: u - jnp.divide(*residual(frozen, u)), sigma * v2))
        u = u - jnp.divide(*residual(lam_table, u))
        X, E, H, lam, lam1, lam2 = _frame(hs, A, u, z2)
        sg, Einv = _columns(E)
        D = 1.0 + lam1[:, 1]
        # PEST chart x = (rho, u(rho, v, phi), nfp*phi) and e = E T.
        one, zero = jnp.ones_like(D), jnp.zeros_like(D)
        T = jnp.stack([jnp.stack([one, zero, zero], -1),
                       jnp.stack([-lam1[:, 0] / D, sigma / D, -nfp * lam1[:, 2] / D], -1),
                       jnp.stack([zero, zero, nfp * one], -1)], axis=1)
        e = E @ T
        g = jnp.einsum("nai,naj->nij", e, e)
        # d det(E) by Jacobi's formula; sqrtg = sigma*nfp*det(E)/(1 + lambda_u).
        d_sg = sg[:, None] * jnp.einsum("nia,naik->nk", Einv, H)
        d_q = d_sg / D[:, None] - (sg / D ** 2)[:, None] * lam2[:, 1, :]
        d_sqrtg = sigma * nfp * jnp.einsum("nk,nkj->nj", d_q, T)
        # Profiles of rho and their rho-derivatives, broadcast to the nodes.
        phip, chip, pres = (jnp.repeat(x, cells, axis=-1) for x in jnp.einsum(
            "pb,dib->pdi", jnp.stack([hs.phipf, hs.chipf, hs.pressure]), Q[:2]))
        # Contravariant field B^j = F b_j, F = 2 rho/sqrtg_native, and its native derivatives.
        F = 2.0 * rho / sg
        b = jnp.stack([zero, chip[0] / nfp - phip[0] * lam1[:, 2], phip[0] * D], -1)
        db = jnp.stack([jnp.zeros_like(d_sg),
                        e_rho * (chip[1] / nfp - phip[1] * lam1[:, 2])[:, None] - phip[0][:, None] * lam2[:, 2, :],
                        e_rho * (phip[1] * D)[:, None] + phip[0][:, None] * lam2[:, 1, :]], axis=1)
        dF = (2.0 * e_rho - F[:, None] * d_sg) / sg[:, None]
        B_sup = F[:, None] * b
        dB_sup = dF[:, None, :] * b[:, :, None] + F[:, None, None] * db
        B = jnp.einsum("naj,nj->na", E, B_sup)
        dB = jnp.einsum("najk,nj->nak", H, B_sup) + jnp.einsum("naj,njk->nak", E, dB_sup)
        dB_cov = jnp.einsum("naik,na->nik", H, B) + jnp.einsum("nai,nak->nik", E, dB)
        curl = jnp.stack([dB_cov[:, 2, 1] - dB_cov[:, 1, 2], dB_cov[:, 0, 2] - dB_cov[:, 2, 0],
                          dB_cov[:, 1, 0] - dB_cov[:, 0, 1]], axis=-1)
        J = jnp.einsum("naj,nj->na", E, curl) / (sf.MU0 * sg)[:, None]
        gr = Einv[:, 0]  # grad(rho), the first dual vector
        grad_phi = jnp.stack([-X[:, 1], X[:, 0], jnp.zeros_like(D)], -1) / (X[:, 0] ** 2 + X[:, 1] ** 2)[:, None]
        # (B . grad) grad(rho) = -(E^-1 (B^k d_k E) E^-1)[0]
        bgg = -jnp.einsum("nb,nbc,nca->na", gr, jnp.einsum("nbck,nk->nbc", H, B_sup), Einv)
        return dict(
            g_rr=g[:, 0, 0], g_rv=g[:, 0, 1], g_rp=g[:, 0, 2], g_vv=g[:, 1, 1], g_vp=g[:, 1, 2], g_pp=g[:, 2, 2],
            g_sup_rr=dot(gr, gr), sqrtg=dot(e[..., 0], jnp.cross(e[..., 1], e[..., 2])),
            sqrtg_r=d_sqrtg[:, 0], sqrtg_v=d_sqrtg[:, 1], sqrtg_p=d_sqrtg[:, 2],
            J_sup_zeta=dot(J, grad_phi), abs_J=jnp.linalg.norm(J, axis=-1),
            iota=sigma * chip[0] / phip[0], p=pres[0], p_r=pres[1],
            J_cross_grad_rho=jnp.cross(J, gr), B_dot_grad_grad_rho=bgg,
            _B=B, _J=J, _Bphi=dot(B, grad_phi),
            _root=u.reshape(-1) + lam - sigma * points[:, 1], _e=e,
        )

    @functools.partial(jax.jit, static_argnames="extras")
    def agni_map(pair: tuple[Any, Any], extras: bool = False):
        state, live = pair
        hs = lift(state, live)
        fields_ = fields(hs)
        Psi = sigma * 2.0 * jnp.pi * live.setup.psi_edge
        public = {k: v for k, v in fields_.items() if not k.startswith("_")}
        if weight is not None:
            public["density"] = weight
        eq = EquilibriumData(
            n_rho=shape[0], n_theta=shape[1], n_zeta=shape[2], Psi=Psi,
            a=_minor_radius(hs, radius_quadrature), NFP=nfp,
            psi_r=rho * Psi / jnp.pi, psi_rr=jnp.full_like(rho, 1.0) * Psi / jnp.pi,
            validate=False, **public,
        )
        if not extras:
            return eq
        return eq, {"root_residual": fields_["_root"], "B": fields_["_B"], "J": fields_["_J"], "e": fields_["_e"],
                    "B_dot_grad_phi": fields_["_Bphi"], "points": points}

    return agni_map
