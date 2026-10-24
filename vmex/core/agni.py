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
  Jacobian and its derivatives, obtained by differentiating the PEST root
  ``u + lambda(rho, u, nfp*phi) = sigma * v`` exactly (implicit function
  theorem, not through the Newton iteration);
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


def _pest_root(iterations: int, sigma: float) -> Callable[..., Array]:
    """Return ``u(hs, rho, v, phi)`` solving ``u + lambda(rho, u, nfp*phi) = sigma*v``.

    Newton with a fixed trip count; the derivative rule is the exact implicit
    one, so nested forward differentiation (needed for the ``sqrtg`` partials)
    never differentiates through the iteration.
    """

    def residual(hs, rho, v, phi, u):
        return u + sf._RZL(hs, jnp.stack([rho, u, hs.nfp * phi]))[2] - sigma * v

    @jax.custom_jvp
    def root(hs, rho, v, phi):
        def step(_, u):
            f, fu = jax.jvp(lambda t: residual(hs, rho, v, phi, t), (u,), (jnp.ones_like(u),))
            return u - f / fu

        return jax.lax.fori_loop(0, iterations, step, sigma * v)

    @root.defjvp
    def root_jvp(primals, tangents):
        u = root(*primals)
        _, dF = jax.jvp(lambda *args: residual(*args, u), primals, tangents)
        _, Fu = jax.jvp(lambda t: residual(*primals, t), (u,), (jnp.ones_like(u),))
        return u, -dF / Fu

    return root


def _minor_radius(hs: Any, quadrature: tuple[int, int, int]) -> Array:
    """Zeta-averaged enclosed cross-section area, as an effective radius ``sqrt(<A>/pi)``."""
    n_rho, n_u, n_z = quadrature
    x, w = np.polynomial.legendre.leggauss(n_rho)
    rho, w = 0.5 * (x + 1.0), 0.5 * w
    u = np.linspace(0.0, 2.0 * np.pi, n_u, endpoint=False)
    z = np.linspace(0.0, 2.0 * np.pi, n_z, endpoint=False)  # nfp*phi, one period
    R, U, Z = np.meshgrid(rho, u, z, indexing="ij")
    points = jnp.asarray(np.stack([R.ravel(), U.ravel(), Z.ravel()], axis=1))

    def element(x):
        e = jax.jacfwd(lambda y: sf._position(hs, y))(x)
        return jnp.linalg.norm(jnp.cross(e[:, 0], e[:, 1]))

    dA = jax.vmap(element)(points).reshape(R.shape)
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
    root = _pest_root(int(newton_iterations), sigma)

    def point_fields(hs: Any, y: Array) -> dict[str, Array]:
        rho, v, phi = y

        def native(yy):
            return jnp.stack([yy[0], root(hs, yy[0], yy[1], yy[2]), hs.nfp * yy[2]])

        def position(yy):
            return sf._position(hs, native(yy))

        e = jax.jacfwd(position)(y)
        dsqrtg = jax.grad(lambda yy: jnp.linalg.det(jax.jacfwd(position)(yy)))(y)
        g = e.T @ e
        x = native(y)
        _, B, J = sf._point_force(hs, x)[:3]
        e_native = jax.jacfwd(lambda q: sf._position(hs, q))(x)
        def grad_rho(q):
            return jnp.linalg.inv(jax.jacfwd(lambda r: sf._position(hs, r))(q))[0]

        gr = grad_rho(x)
        B_sup = jnp.linalg.solve(e_native, B)
        X = sf._position(hs, x)
        grad_phi = jnp.stack([-X[1], X[0], 0.0]) / (X[0] ** 2 + X[1] ** 2)
        def pressure(r):
            return sf._profile(hs, hs.pressure, r)

        return dict(
            g_rr=g[0, 0], g_rv=g[0, 1], g_rp=g[0, 2], g_vv=g[1, 1], g_vp=g[1, 2], g_pp=g[2, 2],
            g_sup_rr=gr @ gr, sqrtg=jnp.linalg.det(e), sqrtg_r=dsqrtg[0], sqrtg_v=dsqrtg[1], sqrtg_p=dsqrtg[2],
            J_sup_zeta=J @ grad_phi, abs_J=jnp.linalg.norm(J),
            iota=sigma * sf._profile(hs, hs.chipf, rho) / sf._profile(hs, hs.phipf, rho),
            p=pressure(rho), p_r=jax.grad(pressure)(rho),
            J_cross_grad_rho=jnp.cross(J, gr),
            B_dot_grad_grad_rho=jax.jacfwd(grad_rho)(x) @ B_sup,
            _B=B, _J=J, _Bphi=B @ grad_phi, _root=x[1] + sf._RZL(hs, x)[2] - sigma * v, _e=e,
        )

    def agni_map(pair: tuple[Any, Any], extras: bool = False):
        state, live = pair
        hs = lift(state, live)
        fields = jax.vmap(lambda y: point_fields(hs, y))(points)
        Psi = sigma * 2.0 * jnp.pi * live.setup.psi_edge
        rho = points[:, 0]
        public = {k: v for k, v in fields.items() if not k.startswith("_")}
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
        return eq, {"root_residual": fields["_root"], "B": fields["_B"], "J": fields["_J"], "e": fields["_e"],
                    "B_dot_grad_phi": fields["_Bphi"], "points": points}

    return agni_map
