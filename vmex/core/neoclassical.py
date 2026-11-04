"""Neoclassical diagnostics and differentiable bounce-averaged heat bounds.

The optional NEO_JAX adapter supplies effective-ripple diagnostics. The heat
bounds retain Lorentz pitch diffusion, bounce geometry and tangential magnetic
and electric precession in a linearized single-reference-well model. They
certify its continuous pitch equation with piecewise orbit coefficients;
optional analytic Maxwell moments also include the full energy axis for the
cold-scatterer collision law. Geometry, reference-model and rounding errors
remain separate. General omnigenity need not be close to this reference.
The conserving collision-form API accepts continuous inverse-norm majorants
and retains local invariant constraints for full Coulomb collision models.
ConservingHeatBound supplies a separate full-Coulomb stellarator proxy with
passing/trapped dynamics and incompressible electric precession.
"""

from __future__ import annotations

from functools import lru_cache
from math import comb
from typing import Any, Sequence

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np
from scipy.special import gamma, gammainc, gammaincc, roots_genlaguerre

from .bounce import bounce_action
from .omnigenity import _nearest_half_mesh_rows, _ResidualTerm, boozer_spectrum_state
from .statephysics import _aspect_scalars

__all__ = [
    "diagnostic_neo_config",
    "screen_transport",
    "transport_improves",
    "epsilon_effective_from_boozer",
    "epsilon_effective_from_wout",
    "bounce_transport_bounds",
    "thermal_transport_bounds",
    "ambipolar_current_interval",
    "continuous_lorentz_heat_bounds",
    "conserving_transport_bounds",
    "conserving_goal_interval",
    "conserving_stellarator_proxy",
    "reference_pitch_grid_from_boozer",
    "bounce_geometry_from_boozer",
    "neoclassical_heat_bounds_from_geometry",
    "neoclassical_heat_bounds_from_boozer",
    "neoclassical_heat_bounds_state",
    "smooth_worst_case_bound",
    "NeoclassicalHeatBound",
    "ConservingHeatBound",
]


def _neo_imports():
    from .._compat import require_optional

    require_optional("neo-jax", "effective ripple")
    try:
        from neo_jax import NeoConfig, run_neo
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "effective ripple requires NEO_JAX; install VMEX with "
            "`pip install vmex[neoclassical]`") from exc
    return NeoConfig, run_neo


def diagnostic_neo_config():
    """Return the bounded NEO resolution used by VMEX summary figures.

    On the bundled NFP=2 QA and NFP=4 QI finite-beta decks this stays within
    4% of NEO_JAX's default ``NeoConfig``.  A 16 x 16 spline grid aliases the
    summary Boozer spectrum (``mboz=16``) and fewer than 20 steps per field
    period or 200 field periods leave 7-30% errors.  Pass a
    ``neo_jax.NeoConfig`` to :func:`epsilon_effective_from_wout` for
    publication calculations.
    """
    NeoConfig, _ = _neo_imports()
    return NeoConfig(
        theta_n=32, phi_n=32, npart=24, multra=1, no_bins=50,
        nstep_per=20, nstep_min=200, nstep_max=500, acc_req=0.02,
        max_rational_field_periods=100000,
    )


def _neo_boozer_data(bx, indices):
    """Share one mode-first Boozer transform with NEO, including sine partners."""
    booz = {
        "nfp_b": int(bx.nfp), "ns_b": len(indices), "mode_first": True,
        "asym": bool(getattr(bx, "asym", False)),
        "ixm_b": np.asarray(bx.xm_b), "ixn_b": np.asarray(bx.xn_b),
        "iota_b": np.asarray(bx.iota)[indices],
        "buco_b": np.asarray(bx.Boozer_I), "bvco_b": np.asarray(bx.Boozer_G),
        "rmnc_b": np.asarray(bx.rmnc_b), "zmns_b": np.asarray(bx.zmns_b),
        "pmns_b": -np.asarray(bx.numns_b), "bmnc_b": np.asarray(bx.bmnc_b),
        "s_b": np.asarray(bx.s_b),
    }
    if booz["asym"]:
        booz.update(rmns_b=np.asarray(bx.rmns_b), zmnc_b=np.asarray(bx.zmnc_b),
                    pmnc_b=-np.asarray(bx.numnc_b), bmns_b=np.asarray(bx.bmns_b))
    return booz


def epsilon_effective_from_boozer(booz: Any, *, config=None):
    r"""Return ``(s, epsilon_eff**(3/2))`` from Boozer-coordinate data.

    ``booz`` may be a NEO_JAX ``BoozerData`` or a booz_xform-style mapping.
    The returned NEO quantity is the effective-ripple transport measure
    :math:`\epsilon_\mathrm{eff}^{3/2}`, conventionally named ``epstot``.
    JAX arrays remain differentiable when the mapping and selected NEO path
    are JAX-native.
    """
    NeoConfig, run_neo = _neo_imports()
    cfg = NeoConfig() if config is None else config
    result = run_neo(
        booz, config=cfg, use_jax=True, progress=False, jax_surface_scan=True)
    values = getattr(result, "eps_eff", None)
    if values is None:
        values = result.epsilon_effective
    if hasattr(booz, "es"):
        surfaces = booz.es
    elif isinstance(booz, dict) and "s_b" in booz:
        surfaces = booz["s_b"]
    else:
        surfaces = np.arange(np.asarray(values).size, dtype=float)
    return surfaces, values


def epsilon_effective_from_wout(
    wout,
    *,
    surfaces: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 0.95),
    mboz: int = 16,
    nboz: int = 12,
    config=None,
    clear_jax_caches: bool = False,
):
    """Compute ``epsilon_eff**(3/2)`` directly from an in-memory VMEX wout.

    The same in-memory Boozer transform feeds NEO, including all LASYM
    partners. Caches remain available unless ``clear_jax_caches=True``.
    """
    if clear_jax_caches:
        import gc

        jax.clear_caches(); gc.collect()
    from booz_xform_jax import Booz_xform

    bx = Booz_xform(verbose=0, mboz=int(mboz), nboz=int(nboz))
    bx.read_wout_data(wout)
    s_in = np.asarray(bx.s_in, dtype=float)
    indices = sorted({int(np.argmin(np.abs(s_in - float(s)))) for s in surfaces})
    bx.compute_surfs = indices
    bx.run()
    booz = _neo_boozer_data(bx, indices)
    return epsilon_effective_from_boozer(booz, config=config)


def screen_transport(wout, surfaces, *, mboz=24, nboz=24, config=None):
    """Matched forward NEO screen; not a converged transport calculation."""
    from booz_xform_jax import Booz_xform
    NeoConfig, run_neo = _neo_imports()

    bx = Booz_xform(verbose=0, mboz=mboz, nboz=nboz)
    bx.read_wout_data(wout)
    indices = [int(np.argmin(abs(np.asarray(bx.s_in)-s))) for s in surfaces]
    if not np.allclose(np.asarray(bx.s_in)[indices], surfaces, rtol=0, atol=1e-12):
        raise ValueError("The NEO screen requires the requested half-mesh radii")
    bx.compute_surfs = indices
    bx.run(jit=True)
    config = config if config is not None else NeoConfig(theta_n=192, phi_n=192, npart=80, multra=2, nstep_per=80,
                       nstep_min=800, nstep_max=2000, acc_req=.005, ref_swi=2,
                       max_rational_field_periods=100000, rational_surface_policy="error")
    results = run_neo(_neo_boozer_data(bx, indices), config=config, use_jax=True,
                      progress=False, jax_surface_scan=False)
    values = []
    for result in results:
        diag, value = result.diagnostics, float(result.epsilon_effective)
        if (not np.isfinite(value) or value <= 0 or not diag.get("rational_correction_valid")
                or diag.get("approximation_used") or diag.get("rational_work_limit_exceeded")):
            raise ValueError(f"Invalid NEO screen at s={float(result.s):.6g}, Q={value}; keep the incumbent")
        values.append(value)
    return np.asarray(values)


def transport_improves(baseline, trial):
    """Require a resolved mean gain and no radial tradeoff in the coarse screen."""
    a, b = np.asarray(baseline), np.asarray(trial)
    return bool(a.ndim == b.ndim == 1 and a.shape == b.shape and a.size > 0 and np.isfinite(a).all()
                and np.isfinite(b).all() and np.all(a > 0) and np.all(b > 0)
                and np.all(b <= a) and b.mean() < .99*a.mean())

def _basis(edges, trial_indices):
    """Continuous linear trial functions, omitting the left Dirichlet node."""
    count = edges.shape[0] - 1
    indices = np.arange(count + 1) if trial_indices is None else np.asarray(trial_indices)
    if (indices.ndim != 1 or len(indices) < 2 or indices[0] != 0
            or indices[-1] != count or np.any(np.diff(indices) <= 0)
            or not np.issubdtype(indices.dtype, np.integer)):
        raise ValueError("trial_indices must increase from 0 to the final pitch edge")
    coarse = edges[jnp.asarray(indices)]
    interval = np.clip(np.searchsorted(indices, np.arange(count + 1), side="right") - 1,
                       0, len(indices) - 2)
    fraction = (edges - coarse[interval]) / (coarse[interval + 1] - coarse[interval])
    nodal = jnp.concatenate((jnp.zeros((1, len(indices) - 1), dtype=edges.dtype),
                             jnp.eye(len(indices) - 1, dtype=edges.dtype)))
    values = (1 - fraction[:, None]) * nodal[interval] + fraction[:, None] * nodal[interval + 1]
    return values[:-1], values[1:]


def _tail(integrals):
    """Integral over cells strictly to the right, with cells on axis zero."""
    reverse_sum = jnp.cumsum(integrals[::-1], axis=0)[::-1]
    return jnp.concatenate((reverse_sum[1:], jnp.zeros_like(integrals[:1])), axis=0)


def _pitch_forms(edges, w, p, forcing, precessions, trial_indices):
    """Exact P1 collision forms, precession tails and Fourier source integrals."""
    width = jnp.diff(edges)
    left, right = _basis(edges, trial_indices)
    change = right - left
    slope = change / width[:, None]
    nodes = jnp.asarray((.5 - np.sqrt(3 / 5) / 2, .5, .5 + np.sqrt(3 / 5) / 2), dtype=edges.dtype)
    quadrature = width[:, None] * jnp.asarray((5 / 18, 4 / 9, 5 / 18), dtype=edges.dtype)
    inverse_weight = quadrature / p[:, None]
    stiffness = jnp.einsum("c,ci,cj->ij", width * p, slope, slope)
    partial = (left[:, None, :] * (1 - nodes)[None, :, None]
               + change[:, None, :] * ((1 - nodes**2) / 2)[None, :, None])

    def precession_tail(omega):
        # int_lambda^right w omega phi_j is quadratic inside each pitch cell.
        amount = (w * omega * width)[:, None]
        return _tail(amount * (left + right) / 2)[:, None, :] + amount[:, None, :] * partial

    fourier = jnp.fft.rfft(forcing, axis=0, norm="forward")
    integrated_source = fourier.T * (w * width)[:, None]
    source_tail = (_tail(integrated_source)[:, None, :]
                   + integrated_source[:, None, :] * (1 - nodes)[None, :, None])
    load = jnp.einsum("cm,ci->mi", integrated_source, (left + right) / 2)
    return stiffness, inverse_weight, tuple(map(precession_tail, precessions)), source_tail, load


def bounce_transport_bounds(
    pitch_edges,
    weight,
    diffusion,
    precession,
    source,
    *,
    collision_frequency,
    trial_indices: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Bound one continuous-pitch orbit response at one speed.

    The equation is ``-nu/w (p g')' + omega*d_alpha g = source`` with
    ``g(left)=0`` and ``p*g'(right)=0``. Increasing ``pitch_edges`` has
    n+1 entries; positive ``weight``/``diffusion`` and finite ``precession``
    have n entries. ``source(nalpha,n)`` is a real odd-grid trigonometric
    interpolant, nalpha>=3; ``collision_frequency`` is a positive scalar.
    The norm uses ``int weight*mean_alpha(source*g) d_pitch``.

    Static ``trial_indices`` include both ends. Nested P1 trial enrichment
    tightens both bounds; three-point cell quadrature is exact in real
    arithmetic for the supplied piecewise coefficients. The zero Fourier
    mode uses the exact collision inverse. Orbit and velocity-rule errors
    are excluded, as is floating-point rounding. JIT/JVP/VJP/vmap preserve
    input derivatives. Invalid values return ``valid=False``, zero lower
    and infinite upper; malformed shapes or trial indices raise ValueError."""
    edges = jnp.asarray(pitch_edges, dtype=jnp.float64)
    w, p, omega = (jnp.asarray(a, dtype=edges.dtype) for a in (weight, diffusion, precession))
    forcing = jnp.asarray(source, dtype=edges.dtype)
    nu = jnp.asarray(collision_frequency, dtype=edges.dtype)
    if (edges.ndim != 1 or edges.shape[0] < 2 or w.shape != (edges.shape[0] - 1,)
            or p.shape != w.shape or omega.shape != w.shape or forcing.ndim != 2
            or forcing.shape[1] != w.shape[0] or nu.ndim != 0):
        raise ValueError("expected edges(n+1), coefficients(n), source(nalpha,n), and scalar nu")
    if forcing.shape[0] < 3 or forcing.shape[0] % 2 == 0:
        raise ValueError("source needs an odd number of at least three alpha samples")
    width = jnp.diff(edges)
    valid = (jnp.all(jnp.isfinite(edges)) & jnp.all(width > 0) & jnp.all(w > 0)
             & jnp.all(p > 0) & jnp.isfinite(nu) & (nu > 0)
             & jnp.all(jnp.isfinite(w)) & jnp.all(jnp.isfinite(p))
             & jnp.all(jnp.isfinite(omega)) & jnp.all(jnp.isfinite(forcing)))
    # Safe inputs keep rejected branches from contaminating reverse derivatives.
    safe_edges = jnp.where(valid, edges, jnp.arange(edges.shape[0], dtype=edges.dtype))
    w, p, omega, forcing, nu = (jnp.where(valid, a, fallback) for a, fallback in
                                ((w, 1.), (p, 1.), (omega, 0.), (forcing, 0.), (nu, 1.)))
    stiffness, inverse_weight, (q,), source_tail, load = _pitch_forms(
        safe_edges, w, p, forcing, (omega,), trial_indices)
    streaming = jnp.einsum("cq,cqi,cqj->ij", inverse_weight, q, q)
    modes = jnp.arange(load.shape[0], dtype=edges.dtype)
    dual_load = -1j * modes[:, None] / nu * jnp.einsum(
        "cq,cqi,cqm->mi", inverse_weight, q, source_tail)
    hessian = nu * stiffness[None, :, :] + modes[:, None, None]**2 / nu * streaming[None, :, :]
    rhs = jnp.stack((load, dual_load), axis=-1)
    # The alpha-independent response uses the exact collision inverse below.
    coefficients = jnp.pad(jnp.linalg.solve(hessian[1:].astype(rhs.dtype), rhs[1:]),
                           ((1, 0), (0, 0), (0, 0)))
    u, h = coefficients[..., 0], coefficients[..., 1]

    def collision_energy(coeff):
        return nu * jnp.real(jnp.einsum("mi,ij,mj->m", coeff.conj(), stiffness, coeff))

    u_tail = jnp.einsum("cqi,mi->cqm", q, u)
    lower_raw = (2 * jnp.real(jnp.sum(load.conj() * u, axis=1)) - collision_energy(u)
                 - modes**2 / nu * jnp.einsum("cq,cqm->m", inverse_weight, jnp.abs(u_tail)**2))
    residual = source_tail - 1j * modes[None, None, :] * jnp.einsum("cqi,mi->cqm", q, h)
    upper = collision_energy(h) + jnp.einsum("cq,cqm->m", inverse_weight, jnp.abs(residual)**2) / nu
    collision_only = jnp.einsum("cq,cqm->m", inverse_weight, jnp.abs(source_tail)**2) / nu
    lower = jnp.maximum(lower_raw, 0.)
    lower = lower.at[0].set(collision_only[0])
    upper = upper.at[0].set(collision_only[0])
    multiplicity = jnp.where(modes == 0, 1., 2.)
    total_lower, total_upper, total_collision = (
        jnp.sum(multiplicity * a) for a in (lower, upper, collision_only))
    finite = (jnp.all(jnp.isfinite(coefficients)) & jnp.isfinite(total_upper)
              & jnp.isfinite(total_lower) & jnp.isfinite(total_collision))
    valid = valid & finite
    return {
        "lower": jnp.where(valid, total_lower, 0.),
        "upper": jnp.where(valid, total_upper, jnp.inf),
        "collision_only": jnp.where(valid, total_collision, jnp.inf),
        "relative_gap": jnp.where(valid, (total_upper - total_lower)
                                   / jnp.where(total_upper > 0, total_upper, 1.), jnp.inf),
        "mode_lower": lower,
        "mode_upper": upper,
        "mode_collision_only": collision_only,
        "valid": valid,
    }


def thermal_transport_bounds(lower, upper, energy, weights, *, normalization=1.) -> dict[str, Any]:
    """Integrate monoenergetic intervals with a positive Maxwell energy rule.

    ``energy = m*v**2/(2*T)``. ``weights`` integrate ``sqrt(x)*exp(-x)``;
    generalized Gauss-Laguerre weights of order one half are suitable.
    ``normalization=1/(sqrt(pi)*I_B)`` converts a radial-velocity response
    to diffusivities, where ``I_B=mean_alpha(int dl/B)``. The resulting
    2-by-2 matrices order the velocity-discrete model for conjugate pressure
    and heat forces ``(1,x-5/2)``. The pressure force is
    ``X1=-d(log p)/dr+q*Er/T``. ``heat_*`` is the coefficient at zero
    pressure force; ``zero_flux_heat_*`` is its Schur complement at zero
    particle flux. A continuous-energy certificate requires a separate
    quadrature and tail estimate.
    """
    lo, hi, x, quadrature = (jnp.asarray(a) for a in (lower, upper, energy, weights))
    normalization = jnp.asarray(normalization)
    if normalization.ndim != 0:
        raise ValueError("normalization must be a scalar")
    if lo.ndim != 1 or hi.shape != lo.shape or x.shape != lo.shape or quadrature.shape != lo.shape:
        raise ValueError("lower, upper, energy and weights must be matching one-dimensional arrays")
    forces = jnp.stack((jnp.ones_like(x), x - 2.5), axis=-1)
    matrices = [normalization * jnp.einsum("v,vi,vj->ij", quadrature * a, forces, forces)
                for a in (lo, hi)]

    def schur(response):
        positive_weights = quadrature * response
        mass = jnp.sum(positive_weights)
        mean = jnp.sum(positive_weights * forces[:, 1]) / jnp.where(mass > 0, mass, 1.)
        return normalization * jnp.sum(positive_weights * (forces[:, 1] - mean)**2)

    valid = (jnp.all(jnp.isfinite(x)) & jnp.all(x > 0) & jnp.all(quadrature > 0)
             & jnp.all(jnp.isfinite(quadrature)) & jnp.isfinite(normalization)
             & (normalization > 0)
             & jnp.all(lo >= 0) & jnp.all(hi >= lo) & jnp.all(jnp.isfinite(hi)))
    return {"matrix_lower": matrices[0], "matrix_upper": matrices[1],
            "heat_lower": matrices[0][1, 1], "heat_upper": matrices[1][1, 1],
            "zero_flux_heat_lower": schur(lo),
            "zero_flux_heat_upper": schur(hi), "valid": valid}


def ambipolar_current_interval(lower, upper, signed_weights) -> tuple[Any, Any]:
    """Bound a signed sum of monoenergetic responses, including charge signs.

    The caller supplies every density, Maxwell quadrature, charge and force
    factor in ``signed_weights``. This operation bounds the supplied kinetic
    model only. Opposite strict signs at two electric fields prove existence
    of a continuous model's root; uniqueness needs an additional derivative
    estimate. An interval containing zero does not establish a root.
    """
    lo, hi, coefficients = (jnp.asarray(a) for a in (lower, upper, signed_weights))
    return (jnp.sum(jnp.where(coefficients >= 0, coefficients * lo, coefficients * hi)),
            jnp.sum(jnp.where(coefficients >= 0, coefficients * hi, coefficients * lo)))


def conserving_transport_bounds(
    collision, drift, source, source_load, inverse_bound, *, invariants=None,
    constraint_rows: tuple[int, ...] = (), constraint_columns: tuple[int, ...] = (),
) -> dict[str, Any]:
    r"""Optimize complementary bounds while retaining collision invariants.

    ``collision(n,n)`` is the trial collision Gram matrix. ``drift(r,n)``
    contains the exact images of these trials under the skew orbit operator,
    in residual generators that may be redundant. ``source(r)`` and ``source_load(n)``
    are its source coefficients and trial inner products. The positive
    semidefinite ``inverse_bound(r,r)`` must majorize the continuous collision
    inverse norm on the residual space. A feasible Coulomb pair-flux lift
    supplies such a matrix; inverting a truncated collision matrix does not.

    Columns of ``invariants(r,k)`` encode the local conserved-moment
    functionals in residual coordinates. Lower trials obey
    ``invariants.conj().T @ drift @ u = 0``; upper trials obey the same
    constraint with right-hand side ``invariants.conj().T @ source``.
    Static ``constraint_rows`` and ``constraint_columns`` select a nonsingular
    square pivot of that constraint matrix. Empty tuples require all these
    constraints to be redundant. Remove joint collision/drift gauges from
    the trial space first. All omitted constraints are checked afterwards.
    The pivot chart is differentiated, so JIT/JVP/VJP remain valid while its
    rank is fixed. No collision damping or invariant projection is added.

    The caller establishes the skew operator, exact residual representation,
    and continuum inverse majorant. ``valid`` checks finite algebra, positive
    forms, the pivot chart, and conservation constraints; it cannot establish
    those physical premises. Roundoff and geometry errors are excluded.
    Invalid algebra returns zero lower and infinite upper. Shapes and static
    pivot errors raise ValueError.
    """
    arrays = tuple(jnp.asarray(x) for x in (collision, drift, source, source_load, inverse_bound))
    dtype = jnp.result_type(*arrays, jnp.float64, 0. if invariants is None else jnp.asarray(invariants))
    c, b, s, f, g = (x.astype(dtype) for x in arrays)
    if (c.ndim != 2 or c.shape[0] < 1 or c.shape[1] != c.shape[0]
            or b.ndim != 2 or b.shape[0] < 1 or b.shape[1] != c.shape[0]
            or s.shape != (b.shape[0],) or f.shape != (c.shape[0],)
            or g.shape != (b.shape[0], b.shape[0])):
        raise ValueError("expected collision(n,n), drift(r,n), source(r), source_load(n), inverse_bound(r,r)")
    p = jnp.zeros((b.shape[0], 0), dtype=dtype) if invariants is None else jnp.asarray(invariants, dtype=dtype)
    if p.ndim != 2 or p.shape[0] != b.shape[0]:
        raise ValueError("invariants must have shape (r,k)")
    rows, columns = np.asarray(constraint_rows), np.asarray(constraint_columns)
    for indices, size in ((rows, p.shape[1]), (columns, c.shape[0])):
        if (indices.ndim != 1 or (indices.size and not np.issubdtype(indices.dtype, np.integer))
                or len(np.unique(indices)) != indices.size
                or np.any(indices < 0) or np.any(indices >= size)):
            raise ValueError("constraint pivots must be distinct integer indices in range")
    if rows.size != columns.size:
        raise ValueError("constraint rows and columns must have equal lengths")
    rows, columns = rows.astype(int), columns.astype(int)
    free = np.setdiff1d(np.arange(c.shape[0]), columns)
    eps = 256 * np.finfo(np.float64).eps * (c.shape[0] + b.shape[0] + p.shape[1])
    valid = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in (c, b, s, f, g, p)]))
    c, b, s, f, g, p = (jnp.where(valid, x, fallback) for x, fallback in
                         ((c, jnp.eye(c.shape[0])), (b, 0.), (s, 0.), (f, 0.),
                          (g, jnp.eye(g.shape[0])), (p, 0.)))
    for matrix in (c, g):
        scale = jnp.maximum(jnp.linalg.norm(matrix), 1.)
        valid &= jnp.isfinite(scale)
        valid &= (jnp.linalg.norm(matrix - matrix.conj().T) <= eps * scale)
        valid &= (jnp.linalg.eigvalsh(matrix)[0] >= -eps * scale)
    e, target = p.conj().T @ b, p.conj().T @ s
    z = jnp.eye(c.shape[0], dtype=dtype)[:, jnp.asarray(free)]
    h0 = jnp.zeros_like(f)
    if rows.size:
        er = e[jnp.asarray(rows)]
        pivot = er[:, jnp.asarray(columns)]
        singular = jnp.linalg.svd(pivot, compute_uv=False)
        chart_valid = singular[-1] > eps * singular[0]
        valid &= chart_valid
        pivot = jnp.where(chart_valid, pivot, jnp.eye(rows.size, dtype=dtype))
        rhs = jnp.concatenate((-er[:, jnp.asarray(free)], target[jnp.asarray(rows), None]), axis=1)
        solution = jnp.linalg.solve(pivot, rhs)
        z = z.at[jnp.asarray(columns)].set(solution[:, :-1])
        h0 = h0.at[jnp.asarray(columns)].set(solution[:, -1])
    chart_defect = jnp.linalg.norm(e @ z) + jnp.linalg.norm(e @ h0 - target)
    chart_scale = 1 + jnp.linalg.norm(e) * (jnp.linalg.norm(z) + jnp.linalg.norm(h0)) + jnp.linalg.norm(target)
    valid &= jnp.isfinite(chart_defect) & jnp.isfinite(chart_scale) & (chart_defect <= eps * chart_scale)
    q = c + b.conj().T @ g @ b
    rhs = jnp.stack((f, b.conj().T @ g @ s - q @ h0), axis=1)
    if free.size:
        reduced = z.conj().T @ q @ z
        positive = jnp.all(jnp.isfinite(jnp.linalg.cholesky(reduced)))
        valid &= positive
        reduced = jnp.where(positive, reduced, jnp.eye(free.size, dtype=dtype))
        coefficients = z @ jnp.linalg.solve(reduced, z.conj().T @ rhs)
    else:
        coefficients = jnp.zeros((c.shape[0], 2), dtype=dtype)
    u, h = coefficients[:, 0], h0 + coefficients[:, 1]
    residual = s - b @ h
    lower = jnp.maximum(2 * jnp.real(jnp.vdot(f, u)) - jnp.real(jnp.vdot(u, q @ u)), 0.)
    upper = jnp.real(jnp.vdot(h, c @ h) + jnp.vdot(residual, g @ residual))
    defect = jnp.linalg.norm(e @ u) + jnp.linalg.norm(e @ h - target)
    scale = 1 + jnp.linalg.norm(e) * (jnp.linalg.norm(u) + jnp.linalg.norm(h)) + jnp.linalg.norm(target)
    valid &= jnp.isfinite(defect) & jnp.isfinite(scale) & (defect <= eps * scale)
    valid &= jnp.isfinite(lower) & jnp.isfinite(upper)
    valid &= lower <= upper + eps * jnp.maximum(jnp.abs(upper), 1.)
    errors = []
    for trial, sign in ((u + h, -1), (u - h, 1)):
        remainder = s + sign * (b @ trial)
        error = jnp.real(jnp.vdot(trial, c @ trial) + jnp.vdot(remainder, g @ remainder)
                         - 2 * jnp.vdot(f, trial))
        valid &= jnp.isfinite(error) & (error >= -eps * (1 + upper + jnp.abs(jnp.vdot(f, trial))))
        errors.append(jnp.maximum(error, 0.))
    upper = jnp.maximum(upper, lower)
    return {"lower": jnp.where(valid, lower, 0.), "upper": jnp.where(valid, upper, jnp.inf),
            "relative_gap": jnp.where(valid, (upper - lower) / jnp.where(upper > 0, upper, 1.), jnp.inf),
            "lower_trial": u, "upper_trial": h, "conservation_defect": defect,
            "primal_error_squared": errors[0], "adjoint_error_squared": errors[1], "valid": valid}


def conserving_goal_interval(collision, orbit_gram, source_load, goal_load,
                             source_bounds, goal_bounds, *, orbit_error_bound=0.) -> dict[str, Any]:
    r"""Bound a signed response using primal and adjoint collision-flux errors.

    The two results must come from :func:`conserving_transport_bounds` with
    the same collision, orbit, residual metric and trial basis, and respective
    sources S and J. ``orbit_gram(i,j)=<trial_i,K trial_j>`` is skew-Hermitian.
    Return an interval for ``Re<J,(A+K)^(-1)S>``. This includes the
    antisymmetric response omitted by polarization of dissipative bounds.
    ``orbit_gram`` must describe the same orbit images as the two bound
    calls. A nonnegative ``orbit_error_bound`` may bound its Frobenius-norm
    integration error; the radius then includes this error times the primal
    and adjoint coefficient norms. It does not cover errors in other forms.
    Derivatives are valid on a fixed chart at positive interval radius;
    a collapsed radius can have a norm cusp. The caller defines the physical
    current or other observable J; this
    operation alone does not supply a bootstrap closure.
    """
    arrays = tuple(jnp.asarray(x) for x in (collision, orbit_gram, source_load, goal_load))
    dtype = jnp.result_type(*arrays, jnp.float64)
    c, w, f, j = (x.astype(dtype) for x in arrays)
    error = jnp.asarray(orbit_error_bound)
    if error.ndim or jnp.iscomplexobj(error):
        raise ValueError("orbit_error_bound must be a real scalar")
    error = error.astype(jnp.float64)
    primal = source_bounds['lower_trial'] + source_bounds['upper_trial']
    adjoint = goal_bounds['lower_trial'] - goal_bounds['upper_trial']
    if (f.ndim != 1 or j.shape != f.shape or primal.shape != f.shape or adjoint.shape != f.shape
            or c.shape != (f.size, f.size) or w.shape != c.shape):
        raise ValueError("goal forms and both bound results must use the same trial basis")
    center = jnp.real(jnp.vdot(j, primal) + jnp.vdot(adjoint, f - (c + w) @ primal))
    product = source_bounds['primal_error_squared'] * goal_bounds['adjoint_error_squared']
    radius = jnp.where(product > 0, jnp.sqrt(jnp.where(product > 0, product, 1.)), 0.)
    error_radius = jnp.where(error > 0, error * jnp.linalg.norm(primal) * jnp.linalg.norm(adjoint), 0.)
    radius += error_radius
    skew_error, orbit_norm = jnp.linalg.norm(w + w.conj().T), jnp.linalg.norm(w)
    skew = (jnp.isfinite(skew_error) & jnp.isfinite(orbit_norm)
            & (skew_error <= 2 * error + 256 * np.finfo(float).eps * f.size * (1 + orbit_norm)))
    valid = (source_bounds['valid'] & goal_bounds['valid'] & skew & jnp.isfinite(center)
             & jnp.isfinite(error) & (error >= 0)
             & jnp.isfinite(product) & (product >= 0) & jnp.isfinite(radius))
    return {"lower": jnp.where(valid, center - radius, -jnp.inf),
            "upper": jnp.where(valid, center + radius, jnp.inf),
            "center": center, "radius": radius, "orbit_error_radius": error_radius, "valid": valid}


_DEFAULT_ENERGY_EDGES = (0., .25, .5, 1., 1.5, 2., 2.5, 3., 4., 5., 7., 10., 16., 24., np.inf)
_POWERS = (3., 3.5, 4., 4.5, 5., 6., 7., 8.)


def _maxwell_moment(power, left, right):
    """Integrate x**power exp(-x), avoiding upper-tail CDF cancellation."""
    shape = power + 1
    if np.isinf(right):
        probability = gammaincc(shape, left)
    elif gammainc(shape, left) > .5:
        probability = gammaincc(shape, left) - gammaincc(shape, right)
    else:
        probability = gammainc(shape, right) - gammainc(shape, left)
    return gamma(shape) * probability


@lru_cache(maxsize=32)
def _energy_constants(energy_edges):
    """Analytic bin moments and scaled positive polynomial Gram factors."""
    edges = np.asarray(energy_edges, dtype=float)
    if (edges.ndim != 1 or len(edges) < 2 or edges[0] != 0 or not np.isposinf(edges[-1])
            or not np.all(np.isfinite(edges[:-1])) or np.any(np.diff(edges) <= 0)):
        raise ValueError("energy_edges must increase from exactly zero to positive infinity")
    moments = np.asarray([[_maxwell_moment(power, left, right) for power in _POWERS]
                          for left, right in zip(edges[:-1], edges[1:])])
    if not np.all(np.isfinite(moments)) or np.any(moments <= 0):
        raise ValueError("energy bins have nonpositive or underflowed moments; use a less extreme grid")
    scales = np.asarray([(left + right) / 2 if np.isfinite(right) else left + 5.
                         for left, right in zip(edges[:-1], edges[1:])])
    mass = moments[:, 0]
    ratios = moments / mass[:, None]
    # Scaled monomials 1,x/scale,(x/scale)^2 keep Gram entries of comparable
    # magnitude; Cholesky evaluation preserves their positive quadratic form.
    residual_factors, streaming_factors = [], []
    for row, scale in zip(ratios, scales):
        moment = dict(zip(_POWERS, row))
        residual = np.asarray([[moment[4. + i + j] / (moment[4.] * scale**(i + j))
                                for j in range(3)] for i in range(3)])
        streaming = np.asarray([[moment[6. + i + j] / (moment[6.] * scale**(i + j))
                                 for j in range(2)] for i in range(2)])
        try:
            residual_factors.append(np.linalg.cholesky(residual))
            streaming_factors.append(np.linalg.cholesky(streaming))
        except np.linalg.LinAlgError as exc:
            raise ValueError("energy moment Gram matrix lost positive definiteness; use wider bins") from exc
    return mass, ratios, scales, np.asarray(residual_factors), np.asarray(streaming_factors)


def continuous_lorentz_heat_bounds(
    geometry,
    *,
    temperature_ev,
    collision_frequency,
    radial_electric_field,
    charge_number=1.,
    energy_edges=_DEFAULT_ENERGY_EDGES,
    trial_indices: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Bound full-Maxwell heat transport for the Lorentz reference model.

    ``geometry`` comes from :func:`bounce_geometry_from_boozer`. Scalar
    temperature is in eV, signed charge in elementary-charge units, thermal
    deflection frequency in s^-1 and ``Er=-dPhi/dr`` in V/m. The coefficient
    is in m^2/s at zero pressure force ``X1=-d(log p)/dr+q*Er/T``.
    The law must be exactly ``nu_D(x)=nu_th*x**(-3/2)``; the heat source is
    ``x*(x-5/2)*S``, magnetic precession scales as x and electric precession
    is constant. This is not a general Coulomb collision model.

    Static ``energy_edges`` increase from zero to infinity. Trials are
    ``x**2`` times continuous P1 pitch functions on each bin; analytic
    Maxwell moments include the infinite tail. Nested energy/pitch trials
    tighten the interval without changing coefficients. Including 2.5 in
    the bins helps resolve the heat-source zero. Returned ``bin_lower``/
    ``bin_upper`` identify unresolved bins; the collision-only bound and
    zero Fourier mode are exact. Positive Gram factors and diagonal
    equilibration stabilize the small solves; unrepresentable bins raise.

    JIT/JVP/VJP/vmap preserve derivatives on the valid geometry branch.
    Invalid numerical data return zero lower, infinite upper and
    ``valid=False``. Geometry/reference-model and rounding errors remain
    outside the certificate."""
    static_edges = np.asarray(energy_edges, dtype=float)
    if static_edges.ndim != 1:
        raise ValueError("energy_edges must be one-dimensional and static")
    mass_np, moments_np, scale_np, residual_np, streaming_np = _energy_constants(tuple(static_edges))
    mass, moments, energy_scale, residual_factor, streaming_factor = (
        jnp.asarray(a, dtype=jnp.float64)
        for a in (mass_np, moments_np, scale_np, residual_np, streaming_np))
    edges = jnp.asarray(geometry["pitch_edges"], dtype=jnp.float64)
    w, p, source_geometry, magnetic_geometry, electric_geometry = (
        jnp.asarray(geometry[name], dtype=edges.dtype) for name in
        ("weight", "diffusion", "source_factor", "precession_factor", "electric_precession_factor"))
    temperature, nu, er, charge, normalization, geometry_valid = (
        jnp.asarray(a) for a in (temperature_ev, collision_frequency, radial_electric_field,
                                 charge_number, geometry["normalization"], geometry["valid"]))
    if any(a.ndim for a in (temperature, nu, er, charge, normalization, geometry_valid)):
        raise ValueError("temperature, collision frequency, electric field, charge and normalization must be scalars")
    if (edges.ndim != 1 or edges.size < 2 or w.shape != (edges.size - 1,) or p.shape != w.shape
            or magnetic_geometry.shape != w.shape or electric_geometry.shape not in ((), w.shape)
            or source_geometry.ndim != 2 or source_geometry.shape[1:] != w.shape):
        raise ValueError("expected edges(n+1), coefficients(n), and source_factor(nalpha,n)")
    nalpha = source_geometry.shape[0]
    if nalpha < 3 or nalpha % 2 == 0:
        raise ValueError("source_factor needs an odd number of at least three alpha samples")
    valid = (geometry_valid & jnp.all(jnp.isfinite(edges)) & jnp.all(jnp.diff(edges) > 0)
             & jnp.all(jnp.isfinite(w)) & jnp.all(w > 0)
             & jnp.all(jnp.isfinite(p)) & jnp.all(p > 0)
             & jnp.all(jnp.isfinite(source_geometry)) & jnp.all(jnp.isfinite(magnetic_geometry))
             & jnp.all(jnp.isfinite(electric_geometry)) & jnp.isfinite(temperature) & (temperature > 0)
             & jnp.isfinite(nu) & (nu > 0) & jnp.isfinite(er) & jnp.isfinite(charge) & (charge != 0)
             & jnp.isfinite(normalization) & (normalization > 0))
    edges = jnp.where(valid, edges, jnp.arange(edges.size, dtype=edges.dtype))
    w, p = jnp.where(valid, w, 1.), jnp.where(valid, p, 1.)
    gyro = 2 * jnp.where(valid, temperature, 1.) / jnp.where(valid, charge, 1.)
    nu, er, normalization = jnp.where(valid, nu, 1.), jnp.where(valid, er, 0.), jnp.where(valid, normalization, 1.)
    source = gyro * jnp.where(valid, source_geometry, 0.)
    magnetic = gyro * jnp.where(valid, magnetic_geometry, 0.)
    electric = jnp.broadcast_to(er * jnp.where(valid, electric_geometry, 0.), w.shape)
    stiffness, inverse_weight, (qe, qm), source_tail, source_load = _pitch_forms(
        edges, w, p, source, (electric, magnetic), trial_indices)
    modes = jnp.arange(source_load.shape[0], dtype=edges.dtype)
    dual_e = jnp.einsum("cq,cqi,cqm->mi", inverse_weight, qe, source_tail)
    dual_m = jnp.einsum("cq,cqi,cqm->mi", inverse_weight, qm, source_tail)
    # Every H is divided by its positive M3 mass before solving.  The energy
    # Gram factor makes magnetic/electric cancellation a sum of squares.
    q_scaled = jnp.stack((jnp.broadcast_to(qe, (mass.size,) + qe.shape),
                          energy_scale[:, None, None, None] * qm), axis=-1)
    transformed_q = jnp.einsum("bcqik,bkj->bcqij", q_scaled, streaming_factor)
    streaming = moments[:, 5, None, None] * jnp.einsum(
        "cq,bcqik,bcqjk->bij", inverse_weight, transformed_q, transformed_q)
    hessian = nu * stiffness[None, None, :, :] + modes[None, :, None, None]**2 / nu * streaming[:, None]
    lower_load = (moments[:, 3] - 2.5 * moments[:, 1])[:, None, None] * source_load[None]
    upper_load = -1j * modes[None, :, None] / nu * (
        (moments[:, 5] - 2.5 * moments[:, 4])[:, None, None] * dual_e[None]
        + (moments[:, 6] - 2.5 * moments[:, 5])[:, None, None] * dual_m[None])
    rhs = jnp.stack((lower_load, upper_load), axis=-1)
    diagonal = jnp.sqrt(jnp.diagonal(hessian, axis1=-2, axis2=-1))
    safe_diagonal = jnp.where(jnp.isfinite(diagonal) & (diagonal > 0), diagonal, 1.)
    conditioned = hessian / (safe_diagonal[..., :, None] * safe_diagonal[..., None, :])
    # No trial solve is needed for the exact alpha-independent response.
    coefficients = jnp.pad(jnp.linalg.solve(conditioned[:, 1:].astype(rhs.dtype),
                           (rhs / safe_diagonal[..., :, None])[:, 1:]),
                           ((0, 0), (1, 0), (0, 0), (0, 0)))
    coefficients /= safe_diagonal[..., :, None]
    u, h = coefficients[..., 0], coefficients[..., 1]
    lower_raw = (2 * jnp.real(jnp.sum(lower_load.conj() * u, axis=-1))
                 - jnp.real(jnp.einsum("bmi,bmij,bmj->bm", u.conj(), hessian, u)))
    collision_h = nu * jnp.real(jnp.einsum("bmi,ij,bmj->bm", h.conj(), stiffness, h))
    he, hm = (jnp.einsum("cqi,bmi->bcqm", q, h) for q in (qe, qm))
    source_tail = jnp.broadcast_to(source_tail, he.shape)
    polynomial = jnp.stack((-2.5 * source_tail,
                             (source_tail - 1j * modes * he) * energy_scale[:, None, None, None],
                             -1j * modes * hm * energy_scale[:, None, None, None]**2), axis=-1)
    transformed = jnp.einsum("bcqmk,bkj->bcqmj", polynomial, residual_factor)
    residual_energy = moments[:, 2, None] / nu * jnp.einsum(
        "cq,bcqmj->bm", inverse_weight, jnp.abs(transformed)**2)
    upper = collision_h + residual_energy
    collision_polynomial = jnp.stack((-2.5 * source_tail,
                                       source_tail * energy_scale[:, None, None, None],
                                       jnp.zeros_like(source_tail)), axis=-1)
    transformed_collision = jnp.einsum("bcqmk,bkj->bcqmj", collision_polynomial, residual_factor)
    collision_only = moments[:, 2, None] / nu * jnp.einsum(
        "cq,bcqmj->bm", inverse_weight, jnp.abs(transformed_collision)**2)
    lower = jnp.maximum(lower_raw, 0.)
    lower = lower.at[:, 0].set(collision_only[:, 0])
    upper = upper.at[:, 0].set(collision_only[:, 0])
    multiplicity = jnp.where(modes == 0, 1., 2.)
    bin_lower, bin_upper, bin_collision = (
        normalization * mass * jnp.sum(multiplicity * value, axis=-1)
        for value in (lower, upper, collision_only))
    total_lower, total_upper, total_collision = (jnp.sum(value) for value in (bin_lower, bin_upper, bin_collision))
    valid &= (jnp.all(jnp.isfinite(coefficients)) & jnp.isfinite(total_lower)
               & jnp.isfinite(total_upper) & jnp.isfinite(total_collision)
               & (total_upper >= total_lower))
    return {
        "lower": jnp.where(valid, total_lower, 0.),
        "upper": jnp.where(valid, total_upper, jnp.inf),
        "collision_only": jnp.where(valid, total_collision, jnp.inf),
        "relative_gap": jnp.where(valid, (total_upper - total_lower)
                                   / jnp.where(total_upper > 0, total_upper, 1.), jnp.inf),
        "bin_lower": jnp.where(valid, bin_lower, 0.),
        "bin_upper": jnp.where(valid, bin_upper, jnp.inf),
        "bin_collision_only": jnp.where(valid, bin_collision, jnp.inf),
        "energy_edges": jnp.asarray(energy_edges), "valid": valid,
    }

def _three_point_derivative(values, coordinates):
    """Differentiate a nonuniform three-point interpolant at its middle node."""
    h0, h1 = jnp.diff(coordinates)
    valid = jnp.isfinite(h0) & jnp.isfinite(h1) & (h0 * h1 > 0)
    h0, h1 = jnp.where(valid, h0, 1.), jnp.where(valid, h1, 1.)
    result = (-h1 / (h0 * (h0 + h1)) * values[0]
              + (h1 - h0) / (h0 * h1) * values[1]
              + h0 / (h1 * (h0 + h1)) * values[2])
    return result, valid


def _reference_traces(
    *, bmnc_b, xm_b, xn_b, iota_b, G_b, I_b, nfp, psi_b,
    bmns_b=None, nalpha=17, points_per_period=128, num_periods=2,
    helicity_m=0, helicity_n=1, well_center=np.pi,
):
    """Sample the central field and a radial stencil of its symmetry projection."""
    if helicity_m not in (0, 1):
        raise ValueError("helicity_m must be 0 (toroidal chart) or 1 (QA/QH chart)")
    if nalpha < 3 or nalpha % 2 == 0:
        raise ValueError("nalpha must be odd and at least three")
    if points_per_period < 16 or points_per_period % 2 or num_periods < 2 or nfp < 1:
        raise ValueError("need even points_per_period >= 16, num_periods >= 2, and nfp >= 1")
    cosine = jnp.asarray(bmnc_b, dtype=jnp.float64)
    sine = jnp.zeros_like(cosine) if bmns_b is None else jnp.asarray(bmns_b, dtype=cosine.dtype)
    m, n, iota, g, current, psi = (jnp.asarray(a, dtype=cosine.dtype)
                                  for a in (xm_b, xn_b, iota_b, G_b, I_b, psi_b))
    if (cosine.ndim != 2 or cosine.shape[0] != 3 or sine.shape != cosine.shape
            or m.shape != (cosine.shape[1],) or n.shape != m.shape
            or any(a.shape != (3,) for a in (iota, g, current, psi))):
        raise ValueError("three radial surfaces are required, with spectra (3,nmode) and vectors (3,)")
    alpha = 2 * jnp.pi * jnp.arange(nalpha, dtype=cosine.dtype) / nalpha
    length = float(2 * np.pi * num_periods)
    x = float(well_center) + jnp.linspace(-length / 2, length / 2,
                                         num_periods * points_per_period + 1)
    if helicity_m == 0:
        denominator = jnp.full_like(iota, float(nfp))
        chart_factor = jnp.asarray(1., dtype=cosine.dtype)
        phi = jnp.broadcast_to(x[None, :] / nfp, (nalpha, x.size))
        theta = alpha[:, None] + iota[1] * phi
        reference_mask = m == 0
        reference_phase = -x[:, None] * n[None, :] / nfp
        transverse_mode = m
    else:
        physical_n = int(helicity_n) * int(nfp)
        delta = iota - physical_n
        safe_delta = jnp.where(jnp.abs(delta) > 1.e-10, delta, 1.)
        denominator = jnp.abs(safe_delta)
        chart_factor = -nfp / safe_delta[1]
        phi = alpha[:, None] / nfp + x[None, :] / safe_delta[1]
        theta = physical_n * alpha[:, None] / nfp + iota[1] * x[None, :] / safe_delta[1]
        reference_mask = n == m * physical_n
        reference_phase = x[:, None] * m[None, :]
        transverse_mode = m * helicity_n - n / nfp
    phase = theta[..., None] * m - phi[..., None] * n
    bmag = jnp.einsum("axm,m->ax", jnp.cos(phase), cosine[1])
    bmag += jnp.einsum("axm,m->ax", jnp.sin(phase), sine[1])
    reference = jnp.einsum("xm,sm->sx", jnp.cos(reference_phase), cosine * reference_mask)
    reference += jnp.einsum("xm,sm->sx", jnp.sin(reference_phase), sine * reference_mask)
    metric = jnp.abs(g + iota * current) / denominator
    valid = (jnp.all(jnp.isfinite(cosine)) & jnp.all(jnp.isfinite(sine))
             & jnp.all(jnp.isfinite(reference)) & jnp.all(reference > 0)
             & jnp.all(jnp.isfinite(bmag)) & jnp.all(bmag > 0)
             & jnp.all(jnp.isfinite(metric)) & jnp.all(metric > 0)
             & jnp.all(jnp.isfinite(psi)))
    if helicity_m == 1:
        valid &= jnp.all(jnp.abs(iota - physical_n) > 1.e-10)
    active_modes = jnp.any((cosine != 0) | (sine != 0), axis=0)
    largest_mode = jnp.max(jnp.where(active_modes, jnp.abs(transverse_mode), 0.))
    return {"bmag": bmag, "reference": reference, "metric": metric,
            "psi": psi, "alpha": alpha, "x": x, "length": length,
            "chart_factor": chart_factor, "valid": valid,
            "alpha_alias_free": largest_mode <= (nalpha - 1) / 2,
            "largest_transverse_mode": largest_mode}


def reference_pitch_grid_from_boozer(*, ncells=24, pitch_margin=0., **boozer_options):
    """Choose full reference pitch grids and a fixed well center on the host.

    Accepts the spectrum/chart options of :func:`bounce_geometry_from_boozer`.
    ``pitch_margin`` removes that fraction at each trapping endpoint; zero
    includes ``[1/Bmax,1/Bmin]``. Cell-center sampling avoids singular edges.
    Returned ``pitch_fractions`` can stay fixed while the physical interval
    follows geometry during optimization. Frozen physical ``pitch_edges``
    instead may lose coverage; inspect the returned coverage diagnostics."""
    if ncells < 1 or not 0 <= pitch_margin < .5:
        raise ValueError("ncells must be positive and pitch_margin must be in [0,0.5)")
    options = {k: v for k, v in boozer_options.items()
               if k not in ("s_b", "psi_edge", "minor_radius", "max_wells",
                            "quadrature_order", "topology_tolerance", "allow_alpha_aliasing",
                            "radial_interpolation")}
    trace = _reference_traces(**options)
    b = np.asarray(trace["reference"][1])
    if not bool(trace["valid"]) or np.max(b) - np.min(b) <= 1.e-12 * np.max(b):
        raise ValueError("the chosen reference projection needs a finite positive magnetic well")
    interval = np.asarray((1 / np.max(b), 1 / np.min(b)))
    span = interval[1] - interval[0]
    edges = np.linspace(interval[0] + pitch_margin * span,
                        interval[1] - pitch_margin * span, int(ncells) + 1)
    center = float(np.asarray(trace["x"])[int(np.argmin(b))] % (2 * np.pi))
    return {"pitch_edges": edges, "pitch_fractions": np.linspace(pitch_margin, 1 - pitch_margin, int(ncells) + 1),
            "well_center": center, "reference_pitch_interval": interval}


def bounce_geometry_from_boozer(
    *, bmnc_b, xm_b, xn_b, iota_b, G_b, I_b, nfp, psi_b, pitch_edges=None, pitch_fractions=None,
    bmns_b=None, s_b=None, psi_edge=None, minor_radius=1., nalpha=17,
    points_per_period=128, num_periods=2, helicity_m=0, helicity_n=1,
    well_center=np.pi, max_wells=8, quadrature_order=32, topology_tolerance=1.e-6,
    allow_alpha_aliasing=False, radial_interpolation="flux",
) -> dict[str, Any]:
    """Construct differentiable coefficients for one Boozer reference well.

    Three ordered radial surfaces are required. ``psi_b`` is signed toroidal
    flux/(2*pi) in Wb and ``xn_b`` is physical toroidal mode number. ``s_b``
    or ``psi_edge`` defines ``r=minor_radius*sqrt(s)`` in metres.
    The toroidal chart (helicity_m=0) has ``x=nfp*phi, alpha=theta-iota*phi``;
    the QA/QH chart (helicity_m=1) has ``x=theta-N*phi, N=helicity_n*nfp``
    and ``dalpha/dalpha_canonical=-nfp/(iota-N)``.

    The source is the alpha derivative of the first-order action departure.
    Reference pitch diffusion and magnetic/electric precession are retained.
    ``radial_interpolation='flux'`` differentiates a quadratic in psi;
    ``'radius'`` differentiates in r and applies dr/dpsi, reproducing r+r²
    reference terms exactly when psi is proportional to s. This controls
    radial coefficient interpolation, not the error of the orbit model.

    Supply ``pitch_edges`` in inverse tesla or fixed ``pitch_fractions`` in
    [0,1], mapped to the current full trapping interval. Exactly one complete
    well nearest ``well_center`` must exist at every pitch. Marginal/ambiguous
    roots, multiple wells, invalid data and unresolved transverse modes set
    ``valid=False``. ``allow_alpha_aliasing=True`` explicitly permits the
    aliased interpolant and retains its diagnostic. Derivatives are local
    to the current root/extremum branches. Perturbation RMS/max diagnostics
    compare delta B with B0 and with the well depth; neither bounds model error."""
    if radial_interpolation not in ("flux", "radius"):
        raise ValueError("radial_interpolation must be 'flux' or 'radius'")
    trace = _reference_traces(
        bmnc_b=bmnc_b, bmns_b=bmns_b, xm_b=xm_b, xn_b=xn_b, iota_b=iota_b,
        G_b=G_b, I_b=I_b, nfp=nfp, psi_b=psi_b, nalpha=nalpha,
        points_per_period=points_per_period, num_periods=num_periods,
        helicity_m=helicity_m, helicity_n=helicity_n, well_center=well_center)
    if (pitch_edges is None) == (pitch_fractions is None):
        raise ValueError("supply exactly one of pitch_edges and pitch_fractions")
    if pitch_edges is None:
        fraction = jnp.asarray(pitch_fractions, dtype=jnp.float64)
        reference = trace["reference"][1]
        safe_reference = jnp.where(jnp.all(jnp.isfinite(reference)) & jnp.all(reference > 0), reference,
                                    1 + .1 * jnp.cos(trace["x"] - float(well_center) + jnp.pi))
        lo, hi = 1 / jnp.max(safe_reference), 1 / jnp.min(safe_reference)
        edges = lo + fraction * (hi - lo)
    else:
        edges = jnp.asarray(pitch_edges, dtype=jnp.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("pitch_edges must be one-dimensional with at least two entries")
    psi = trace["psi"]
    if s_b is None:
        if psi_edge is None:
            raise ValueError("s_b or psi_edge is required to define the physical radial coordinate")
        edge_flux = jnp.asarray(psi_edge, dtype=edges.dtype)
        s = psi / jnp.where(jnp.isfinite(edge_flux) & (edge_flux != 0), edge_flux, 1.)
        flux_valid = jnp.isfinite(edge_flux) & (edge_flux != 0)
    else:
        s = jnp.asarray(s_b, dtype=edges.dtype)
        flux_valid = jnp.asarray(True)
    if s.shape != (3,):
        raise ValueError("s_b must contain the three stencil surface labels")
    b0 = trace["reference"][1]
    coordinate, derivative_factor = psi, jnp.asarray(1., dtype=edges.dtype)
    if radial_interpolation == "radius":
        coordinate = jnp.sqrt(jnp.where(s > 0, s, 1.))
        derivative_factor = coordinate[1] / (2 * jnp.where(psi[1] != 0, psi[1], 1.))
        flux_valid &= jnp.allclose(psi * s[1], psi[1] * s, rtol=1.e-10, atol=0.)
    field_derivative, stencil_valid = _three_point_derivative(trace["reference"], coordinate)
    metric_derivative, _ = _three_point_derivative(trace["metric"], coordinate)
    field_derivative, metric_derivative = derivative_factor * field_derivative, derivative_factor * metric_derivative
    basic_valid = (trace["valid"] & stencil_valid & flux_valid
                   & jnp.all(jnp.isfinite(s)) & jnp.all(s > 0) & (psi[1] != 0)
                   & jnp.isfinite(minor_radius) & (minor_radius > 0)
                   & jnp.all(jnp.isfinite(edges)) & jnp.all(jnp.diff(edges) > 0)
                   & jnp.all(edges > 0))
    fallback = 1 + .1 * jnp.cos(trace["x"] - float(well_center) + jnp.pi)
    b0 = jnp.where(basic_valid, b0, fallback)
    metric = jnp.where(basic_valid, trace["metric"][1], 1.)
    perturbation = jnp.where(basic_valid, trace["bmag"] - b0[None, :], 0.)
    field_derivative = jnp.where(basic_valid, field_derivative, 0.)
    metric_derivative = jnp.where(basic_valid, metric_derivative, 0.)
    interval = jnp.stack((1 / jnp.max(b0), 1 / jnp.min(b0)))
    span = interval[1] - interval[0]
    pitch = (edges[1:] + edges[:-1]) / 2
    safe_pitch = jnp.where(basic_valid, pitch, jnp.mean(interval))
    sampled_b0 = jnp.broadcast_to(b0, trace["bmag"].shape)
    bounce = bounce_action(
        sampled_b0, safe_pitch, dl_dphi=metric / sampled_b0, length=trace["length"],
        periodic=False, max_wells=max_wells, quadrature_order=quadrature_order,
        topology_tolerance=topology_tolerance,
        drift_integrands={"perturbation": perturbation / b0,
                          "radial_reference": field_derivative / b0},
        parallel_integrands={"invB": 1 / sampled_b0})
    centers = (bounce["well_left"][0] + bounce["well_right"][0]) / 2
    distance = jnp.where(bounce["usable_mask"][0],
                         jnp.abs(centers - trace["length"] / 2), jnp.inf)
    selected = jnp.argmin(distance, axis=-1)
    gather = jnp.broadcast_to(selected[None, :, None], (nalpha, pitch.size, 1))

    def take(name):
        return jnp.take_along_axis(bounce[name], gather, axis=-1)[..., 0]

    usable = take("usable_mask")
    nearest = jnp.min(distance, axis=-1)
    if max_wells > 1:
        ambiguity_margin = jnp.sort(distance, axis=-1)[:, 1] - nearest
    else:
        ambiguity_margin = jnp.full_like(nearest, jnp.inf)
    one_well = jnp.sum(jnp.abs(centers - trace["length"] / 2) < np.pi, axis=-1) == 1
    # Count reference extrema independently of pitch cells: a narrow second
    # family must not escape rejection merely by falling between pitch nodes.
    start = (num_periods - 1) * points_per_period // 2
    cell = b0[start:start + points_per_period + 1]
    periodic_cell = cell[:-1]
    before, after = jnp.roll(periodic_cell, 1), jnp.roll(periodic_cell, -1)
    minima = jnp.sum((periodic_cell <= before) & (periodic_cell < after))
    maxima = jnp.sum((periodic_cell >= before) & (periodic_cell > after))
    tolerance = 32 * jnp.finfo(edges.dtype).eps * jnp.maximum(interval[1], 1.)
    band_valid = (edges[0] >= interval[0] - tolerance) & (edges[-1] <= interval[1] + tolerance)
    valid = (basic_valid & (span > 0) & band_valid & jnp.all(usable)
             & jnp.all(nearest < np.pi) & jnp.all(ambiguity_margin > 1.e-5)
             & jnp.all(one_well) & (minima == 1) & (maxima == 1)
             & (trace["alpha_alias_free"] | allow_alpha_aliasing))
    tau = jnp.where(valid, take("bounce_time")[0], 2.)
    action = jnp.where(valid, take("action")[0], 0.)
    delta_action = jnp.where(valid, -2 * take("drift_perturbation"), 0.)
    radial_action = jnp.where(valid, (metric_derivative / metric) * action
                              - 2 * take("drift_radial_reference")[0], 0.)
    fourier = jnp.fft.rfft(delta_action, axis=0)
    modes = jnp.arange(fourier.shape[0], dtype=edges.dtype)
    alpha_derivative = jnp.fft.irfft(1j * modes[:, None] * fourier, n=nalpha, axis=0)
    safe_s = jnp.where(basic_valid, s[1], 1.)
    safe_psi = jnp.where(basic_valid, psi[1], 1.)
    safe_minor = jnp.where(basic_valid, minor_radius, 1.)
    dr_dpsi = safe_minor * jnp.sqrt(safe_s) / (2 * safe_psi)
    c = jnp.where(basic_valid, trace["chart_factor"], 1.)
    # The reference is exactly periodic; one cell supplies the orbit measure.
    measure = jnp.trapezoid(metric / cell**2, dx=2 * jnp.pi / points_per_period)
    diffusion = jnp.where(valid, 2 * pitch * take("parallel_invB")[0], 1.)
    relative = perturbation / b0
    well_depth = jnp.max(b0) - jnp.min(b0)
    relative_well = perturbation / jnp.where(well_depth > 0, well_depth, 1.)
    return {
        "pitch_edges": edges, "pitch": pitch, "weight": tau / 2,
        "diffusion": diffusion, "source_factor": dr_dpsi * c * alpha_derivative / tau,
        "precession_factor": -c * radial_action / tau,
        "electric_precession_factor": -c * dr_dpsi,
        "normalization": 1 / (jnp.sqrt(jnp.pi) * measure), "orbit_measure": measure,
        "reference_action": action, "action_perturbation": delta_action,
        "radial_action_derivative": radial_action,
        "reference_pitch_interval": interval, "common_pitch_interval": interval,
        "pitch_band_fraction": (edges[-1] - edges[0]) / jnp.where(span > 0, span, 1.),
        "reference_perturbation_rms": jnp.sqrt(jnp.mean(relative**2)),
        "reference_perturbation_max": jnp.max(jnp.abs(relative)),
        "reference_well_depth": well_depth,
        "reference_perturbation_well_rms": jnp.sqrt(jnp.mean(relative_well**2)),
        "reference_perturbation_well_max": jnp.max(jnp.abs(relative_well)),
        "well_center_margin": jnp.min(ambiguity_margin),
        "reference_minima": minima, "reference_maxima": maxima,
        "alpha_alias_free": trace["alpha_alias_free"],
        "largest_transverse_mode": trace["largest_transverse_mode"],
        "surface": s[1], "radius": safe_minor * jnp.sqrt(safe_s),
        "dr_dpsi": dr_dpsi, "chart_factor": c, "valid": valid,
    }


@lru_cache(maxsize=16)
def _energy_rule(count):
    """Positive generalized Gauss-Laguerre rule for sqrt(x)*exp(-x)."""
    if count < 2:
        raise ValueError("nenergy must be at least two")
    return roots_genlaguerre(int(count), .5)


def neoclassical_heat_bounds_from_geometry(
    geometry, *, temperature_ev, collision_frequencies, radial_electric_fields,
    charge_number=1., nenergy=8, collision_factors=None, trial_indices=None,
    continuous_energy=False, energy_edges=None,
) -> dict[str, Any]:
    """Evaluate thermal intervals on a collision-frequency/electric-field grid.

    Temperature is in eV, signed charge in elementary-charge units, thermal
    frequencies in s^-1 and electric fields in V/m. The default positive
    speed rule uses ``nu(x)/nu_th=x**(-3/2)``; optional ``collision_factors``
    is a positive nenergy-vector. Heat bounds have shape (nnu,nEr), units
    m^2/s and zero pressure force ``X1=-d(log p)/dr+q*Er/T``. The conjugate
    heat polynomial is x-5/2. ``matrix_*`` order the pressure/heat responses;
    ``zero_flux_heat_*`` are Schur bounds at zero particle flux. Unnormalized
    ``mono_*`` have shape (nnu,nEr,nenergy), with energy nodes, positive
    weights and normalization returned for signed current sums.

    ``continuous_energy=True`` integrates the whole Maxwell axis for the
    exact cold-scatterer Lorentz law and requires collision_factors=None.
    Static ``energy_edges`` span zero to infinity and set the trial bins;
    nenergy is unused. This branch returns heat, collision-only and binwise
    bounds, but no monoenergetic, matrix or Schur fields. In the default
    branch speed-rule error is separate. All inputs retain derivatives;
    invalid cases return zero lower and infinite upper."""
    if continuous_energy:
        if collision_factors is not None:
            raise ValueError("continuous_energy requires the exact Lorentz law and collision_factors=None")
        bins = _DEFAULT_ENERGY_EDGES if energy_edges is None else energy_edges
        frequencies = jnp.atleast_1d(jnp.asarray(collision_frequencies, dtype=jnp.float64))
        fields = jnp.atleast_1d(jnp.asarray(radial_electric_fields, dtype=jnp.float64))
        if frequencies.ndim != 1 or fields.ndim != 1 or not frequencies.size or not fields.size:
            raise ValueError("collision frequencies and electric fields must be nonempty one-dimensional arrays")

        def continuous_case(frequency, field):
            case = continuous_lorentz_heat_bounds(
                geometry, temperature_ev=temperature_ev, collision_frequency=frequency,
                radial_electric_field=field, charge_number=charge_number,
                energy_edges=bins, trial_indices=trial_indices)
            case.pop("energy_edges")
            return case

        result = jax.vmap(lambda frequency: jax.vmap(
            lambda field: continuous_case(frequency, field))(fields))(frequencies)
        result.update(energy_edges=jnp.asarray(bins), normalization=geometry["normalization"],
                      collision_frequencies=frequencies, radial_electric_fields=fields, geometry=geometry)
        return result
    if energy_edges is not None:
        raise ValueError("energy_edges requires continuous_energy=True")
    energy_np, weights_np = _energy_rule(int(nenergy))
    energy, weights = jnp.asarray(energy_np), jnp.asarray(weights_np)
    factors = energy**-1.5 if collision_factors is None else jnp.asarray(collision_factors)
    nu = jnp.atleast_1d(jnp.asarray(collision_frequencies, dtype=energy.dtype))
    er = jnp.atleast_1d(jnp.asarray(radial_electric_fields, dtype=energy.dtype))
    temperature, charge = jnp.asarray(temperature_ev), jnp.asarray(charge_number)
    if factors.shape != energy.shape or nu.ndim != 1 or er.ndim != 1 or not nu.size or not er.size:
        raise ValueError("collision_factors needs nenergy entries; nu and Er need nonempty one-dimensional arrays")
    if temperature.ndim or charge.ndim:
        raise ValueError("temperature_ev and charge_number must be scalars")
    input_valid = (geometry["valid"] & jnp.isfinite(temperature) & (temperature > 0)
                   & jnp.isfinite(charge) & (charge != 0)
                   & jnp.all(jnp.isfinite(factors)) & jnp.all(factors > 0))
    safe_charge = jnp.where(input_valid, charge, 1.)
    gyro_factor = 2 * jnp.where(input_valid, temperature, 1.) / safe_charge
    safe_factors = jnp.where(input_valid, factors, jnp.ones_like(factors))
    normalization = geometry["normalization"]

    def scenario(nu_th, electric):
        case_valid = jnp.isfinite(nu_th) & (nu_th > 0) & jnp.isfinite(electric)
        safe_nu, safe_electric = jnp.where(case_valid, nu_th, 1.), jnp.where(case_valid, electric, 0.)

        def mono(x, factor):
            return bounce_transport_bounds(
                geometry["pitch_edges"], geometry["weight"], geometry["diffusion"],
                gyro_factor * x * geometry["precession_factor"]
                + safe_electric * geometry["electric_precession_factor"],
                gyro_factor * x * geometry["source_factor"],
                collision_frequency=safe_nu * factor, trial_indices=trial_indices)
        response = jax.vmap(mono)(energy, safe_factors)
        thermal = thermal_transport_bounds(response["lower"], response["upper"],
                                            energy, weights, normalization=normalization)
        valid = input_valid & case_valid & jnp.all(response["valid"]) & thermal["valid"]
        lo = jnp.where(valid, thermal["heat_lower"], 0.)
        hi = jnp.where(valid, thermal["heat_upper"], jnp.inf)
        collision = normalization * jnp.sum(weights * (energy - 2.5)**2 * response["collision_only"])
        return {"lower": lo, "upper": hi, "valid": valid,
                "relative_gap": jnp.where(valid, (hi - lo) / jnp.where(hi > 0, hi, 1.), jnp.inf),
                "collision_only": jnp.where(valid, collision, jnp.inf),
                "mono_lower": jnp.where(valid, response["lower"], 0.),
                "mono_upper": jnp.where(valid, response["upper"], jnp.inf),
                "matrix_lower": jnp.where(valid, thermal["matrix_lower"], 0.),
                "matrix_upper": jnp.where(valid, thermal["matrix_upper"], jnp.inf),
                "zero_flux_heat_lower": jnp.where(valid, thermal["zero_flux_heat_lower"], 0.),
                "zero_flux_heat_upper": jnp.where(valid, thermal["zero_flux_heat_upper"], jnp.inf)}

    result = jax.vmap(lambda frequency: jax.vmap(lambda field: scenario(frequency, field))(er))(nu)
    result.update(energy=energy, energy_weights=weights, normalization=normalization,
                  collision_factors=factors, collision_frequencies=nu, radial_electric_fields=er,
                  geometry=geometry)
    return result


def neoclassical_heat_bounds_from_boozer(
    *, temperature_ev, collision_frequencies, radial_electric_fields, charge_number=1.,
    nenergy=8, collision_factors=None, trial_indices=None, continuous_energy=False,
    energy_edges=None, **geometry_options,
):
    """Compose Boozer orbit geometry and the selected Maxwell integration model."""
    geometry = bounce_geometry_from_boozer(**geometry_options)
    return neoclassical_heat_bounds_from_geometry(
        geometry, temperature_ev=temperature_ev, collision_frequencies=collision_frequencies,
        radial_electric_fields=radial_electric_fields, charge_number=charge_number,
        nenergy=nenergy, collision_factors=collision_factors, trial_indices=trial_indices,
        continuous_energy=continuous_energy, energy_edges=energy_edges)


def _state_stencils(state, rt, surfaces, mboz, nboz, oversample):
    """Transform each target and its neighboring half-mesh surfaces once."""
    s_half, rows = _nearest_half_mesh_rows(int(rt.setup.s_full.shape[0]), surfaces)
    if np.any(rows <= 1) or np.any(rows >= len(s_half)):
        raise ValueError("target surfaces need a neighboring half-mesh surface on each side")
    stencil_surfaces = s_half[(rows[:, None] + np.arange(-1, 2)[None, :]) - 1]
    booz = boozer_spectrum_state(state, rt, surfaces=stencil_surfaces.ravel(),
                                 mboz=mboz, nboz=nboz, oversample=oversample)
    return booz, len(rows)


def neoclassical_heat_bounds_state(
    state, rt, *, surfaces, temperature_ev, collision_frequencies, pitch_edges=None, pitch_fractions=None,
    radial_electric_fields, mboz=8, nboz=8, oversample=2, **options,
):
    """Evaluate the proxy directly from a converged VMEX state and runtime.

    Requested target surfaces snap to the nearest half mesh; each must have
    one neighbor on either side for magnetic precession.  Supply exactly one
    of ``pitch_edges`` or ``pitch_fractions``; either can be shared
    one-dimensional edges or shape ``(nsurface,ncell+1)``.
    The physical radius uses VMEX's canonical minor radius.  A leading
    surface axis is added to every scenario and geometry output.  This
    function is JIT/JVP/VJP traceable through VMEX's Boozer transform.
    """
    booz, count = _state_stencils(state, rt, surfaces, mboz, nboz, oversample)
    if (pitch_edges is None) == (pitch_fractions is None):
        raise ValueError("supply exactly one of pitch_edges and pitch_fractions")
    grid_name = "pitch_fractions" if pitch_edges is None else "pitch_edges"
    edges = jnp.asarray(pitch_fractions if pitch_edges is None else pitch_edges, dtype=jnp.float64)
    if edges.ndim == 1:
        edges = jnp.broadcast_to(edges, (count, edges.size))
    if edges.ndim != 2 or edges.shape[0] != count:
        raise ValueError("pitch_edges must be shared one-dimensional edges or one row per target surface")
    minor_radius = _aspect_scalars(state, rt)[0]
    results = []
    for index in range(count):
        group = slice(3 * index, 3 * index + 3)
        geom = {name: booz[name][group] for name in
                ("bmnc_b", "bmns_b", "iota_b", "G_b", "I_b", "psi_b", "s_b")}
        results.append(neoclassical_heat_bounds_from_boozer(
            **geom, xm_b=booz["xm_b"], xn_b=booz["xn_b"], nfp=booz["nfp"],
            **{grid_name: edges[index]}, minor_radius=minor_radius,
            temperature_ev=temperature_ev, collision_frequencies=collision_frequencies,
            radial_electric_fields=radial_electric_fields, **options))
    return jax.tree.map(lambda *items: jnp.stack(items), *results)


def smooth_worst_case_bound(upper, *, scale=1., smoothing=.05):
    """Return a differentiable conservative maximum of normalized upper bounds.

    ``scale`` is a fixed positive reference diffusivity; ``smoothing`` is
    dimensionless.  The unnormalized log-sum-exp lies between ``max(upper/scale)``
    and that maximum plus ``smoothing*log(number_of_cases)``.  Its additive
    offset should not be subtracted if the conservative-maximum property is
    wanted.  A nonfinite or negative input returns infinity with a safe rejected
    branch.  This aggregation bounds the sampled scenarios, not a continuum
    of collision frequencies or electric fields between the samples.
    """
    values = jnp.asarray(upper)
    scale, smoothing = jnp.asarray(scale), jnp.asarray(smoothing)
    valid = (jnp.all(jnp.isfinite(values)) & jnp.all(values >= 0)
             & jnp.isfinite(scale) & (scale > 0)
             & jnp.isfinite(smoothing) & (smoothing > 0))
    safe_values = jnp.where(valid, values, 0.)
    safe_scale, safe_smoothing = jnp.where(valid, scale, 1.), jnp.where(valid, smoothing, 1.)
    total = safe_smoothing * logsumexp(safe_values / (safe_scale * safe_smoothing))
    return jnp.where(valid, total, jnp.inf)


def conserving_stellarator_proxy(
    modes, coefficients, *, iota, G_b, I_b, nfp=1, helicity=0,
    collision_frequency=1., electric_field=0., integration_order=32,
    heat_inverse_bound=4 * np.sqrt(2 / np.pi),
    stress_inverse_bound=9368 * np.sqrt(2 / np.pi) / 2205, rho_bound=None, macro_order=None,
):
    r"""Cheap continuous full-Coulomb heat bound for a finite Boozer Jacobian.

    Define ``J=Re sum(coefficients*exp(i*(m*x+n*y)))=Bhat**-2``, with
    ``x=theta-helicity*zeta``, ``y=nfp*zeta`` and integer ``modes`` of shape (N,2).
    Streaming and mirror forces retain passing and trapped particles; electric
    precession uses the incompressible DKES model. Tangential magnetic drift is
    absent. The collision operator is the full single-species linearized Landau
    operator about ``M=pi**(-3/2)*exp(-v**2)`` with pair coefficient one.
    Frequencies and fields are dimensionless and broadcast together.

    The feasible flow ``-Bhat*(D^-1 Dperp J+c)*vparallel*(v**2-5/2)`` has one
    analytically optimized constant. Defaults use explicit Maxwell-kernel flux
    lifts for its two Coulomb inverse norms. Sharper supplied constants must
    also bound the *continuous* inverses of the heat and stress polynomials.
    A nontrivial lower endpoint is available at zero electric field only.

    The periodic grid integrates polynomials through degree
    ``(integration_order+4)*max(abs(modes))`` exactly, or
    ``(integration_order+2*L+12)*max(abs(modes))`` with ``macro_order=L``;
    a positive binomial-tail
    correction bounds the remaining rational-weight quadrature error. Floating
    point rounding and the error of fitting J to another field remain separate.
    Gradients are local to a fixed nonresonant Fourier chart. A fixed
    ``rho_bound`` above the relative Fourier l1 amplitude makes the tail smooth
    even when a coefficient vanishes. Invalid charts return ``[0, infinity]``.

    ``macro_order=L`` adds a correlated conserved-moment trial, using a finite
    degree-L polynomial for J^-3, and returns its smooth affine upper bound.
    Its complete even collision-inverse block uses the Maxwell flux majorant;
    ``stress_inverse_bound`` then applies only to the ``scalar_upper`` diagnostic.
    With the default compatible inverse constants the affine trial space
    contains the scalar trial family. Sharper scalar stress constants need not
    preserve that ordering. The extended Fourier primitive chart must remain
    nonresonant, and source-free fields return zero without a singular solve.
    """
    wave = np.asarray(modes)
    if (wave.ndim != 2 or wave.shape[1] != 2 or not len(wave)
            or not np.all(np.isfinite(wave)) or not np.all(wave == wave.astype(int))
            or len(np.unique(wave, axis=0)) != len(wave)
            or np.count_nonzero(np.all(wave == 0, axis=1)) != 1):
        raise ValueError("modes must be distinct integer pairs containing exactly one zero mode")
    if (int(nfp) != nfp or nfp < 1 or int(helicity) != helicity or helicity % nfp
            or int(integration_order) != integration_order or integration_order < 0):
        raise ValueError("require positive integer nfp, helicity a multiple of nfp, and nonnegative integration_order")
    wave, order = wave.astype(int), int(integration_order)
    if macro_order is not None:
        if (np.ndim(macro_order) or np.iscomplexobj(macro_order) or not np.isfinite(macro_order)
                or int(macro_order) != macro_order or macro_order < 0):
            raise ValueError("macro_order must be a nonnegative integer or None")
        macro_order = int(macro_order)
    coefficient = jnp.asarray(coefficients) + 0.
    if coefficient.shape != (len(wave),):
        raise ValueError("coefficients must have one entry per Fourier mode")
    scalar = [jnp.asarray(x) for x in (iota, G_b, I_b, heat_inverse_bound, stress_inverse_bound)]
    nu, field = jnp.broadcast_arrays(jnp.asarray(collision_frequency), jnp.asarray(electric_field))
    if (any(x.ndim or jnp.iscomplexobj(x) for x in scalar)
            or jnp.iscomplexobj(nu) or jnp.iscomplexobj(field) or not nu.size):
        raise ValueError("geometry and inverse bounds must be real scalars; operating points must be real and nonempty")
    iota, G_b, I_b, mu1, mu2 = scalar
    zero = np.all(wave == 0, axis=1)
    center = jnp.real(coefficient[np.flatnonzero(zero)[0]])
    covariant = G_b + iota * I_b
    basic = (jnp.all(jnp.isfinite(coefficient)) & jnp.isfinite(center) & (center > 0)
             & jnp.all(jnp.isfinite(jnp.stack(scalar))) & jnp.isfinite(covariant)
             & (covariant != 0) & (mu1 > 0) & (mu2 > 0))
    center, covariant = jnp.where(basic, center, 1.), jnp.where(basic, covariant, 1.)
    relative_amplitude = jnp.sum(jnp.abs(jnp.where(zero, 0., coefficient))) / center
    rho = relative_amplitude if rho_bound is None else jnp.asarray(rho_bound)
    if rho.ndim or jnp.iscomplexobj(rho):
        raise ValueError("rho_bound must be a real scalar")
    kd = wave @ jnp.stack((iota - helicity, jnp.asarray(nfp)))
    ke = wave @ jnp.stack(((G_b + helicity * I_b) / covariant, -nfp * I_b / covariant))
    nonresonant = jnp.all(zero | (jnp.abs(kd) > 64 * jnp.finfo(center.dtype).eps * (1 + jnp.abs(ke))))
    chart = basic & nonresonant & jnp.isfinite(rho) & (rho >= relative_amplitude) & (rho < 1)
    rho, coefficient = jnp.where(chart, rho, 0.), jnp.where(chart, coefficient, zero.astype(float))
    kd, ke = jnp.where(chart, kd, 0.), jnp.where(chart, ke, 0.)
    primitive = jnp.where(zero, 0., ke / jnp.where(zero | (kd == 0), 1., kd)) * coefficient
    band = np.max(np.abs(wave), axis=0)
    shape = tuple((order + (4 if macro_order is None else 2 * macro_order + 12)) * band + 1)
    spectra = jnp.stack((coefficient, 1j * kd * coefficient, 1j * ke * coefficient,
                        primitive, 1j * ke * primitive))
    grid = jnp.zeros((5, *shape), dtype=jnp.result_type(spectra, 1j))
    grid = grid.at[:, wave[:, 0] % shape[0], wave[:, 1] % shape[1]].add(spectra)
    jacobian, dj, ej, p, ep = jnp.fft.ifft2(grid).real * np.prod(shape)
    weights, tails = [], []
    for power in (3, 4):
        tail = rho**(order + 1) * sum(comb(order + power, power - 1 - k)
            * rho**k / (1 - rho)**(k + 1) for k in range(power)) / center**power
        tails.append(tail)
        weights.append(jacobian**(-power) + 2 * tail)
    w3, w4 = weights
    even, odd = jacobian * ej - p * dj, jacobian * ep - .5 * p * ej
    valid = chart & jnp.isfinite(nu) & (nu > 0) & jnp.isfinite(field)
    nu, field = jnp.where(valid, nu, 1.), jnp.where(valid, field, 0.)
    if macro_order is not None:
        field = jnp.where(jnp.all(ej == 0), 0., field)
    a, b, d = nu * (4 * np.sqrt(2 / np.pi) / 3), 9 * mu2 / (16 * nu), (field * center)**2 * mu1 / nu
    quadratic = a + b * jnp.mean(dj**2 * w3) + .25 * d * jnp.mean(ej**2 * w4)
    linear = a * jnp.mean(p) - b * jnp.mean(even * dj * w3) - .5 * d * jnp.mean(odd * ej * w4)
    constant = -linear / quadratic
    c = constant[..., None, None]
    upper = (a * jnp.mean((p + c)**2, axis=(-2, -1))
             + b * jnp.mean((even - c * dj)**2 * w3, axis=(-2, -1))
             + d * jnp.mean((odd - .5 * c * ej)**2 * w4, axis=(-2, -1))) / center
    geometric = jnp.mean(p**2) / center
    lower = jnp.where(field == 0, 25 * nu * geometric / (16 * mu1), 0.)
    extra = {}
    if macro_order is not None:
        # Correlated moments satisfy local conservation as polynomial identities.
        # Field-independent Gram generators keep operating-point batches small.
        scalar_upper = upper
        scalar_valid = valid & jnp.isfinite(upper) & (upper >= lower)
        fx, fy = np.meshgrid(*(np.rint(np.fft.fftfreq(n) * n) for n in shape), indexing="ij")
        along = (iota - helicity) * fx + nfp * fy
        across = ((G_b + helicity * I_b) * fx - nfp * I_b * fy) / covariant
        active = ((np.abs(fx) <= (macro_order + 3) * band[0])
                  & (np.abs(fy) <= (macro_order + 3) * band[1]) & ((fx != 0) | (fy != 0)))
        nonzero = jnp.abs(along) > 64 * jnp.finfo(center.dtype).eps * (1 + jnp.abs(across))
        macro_chart = jnp.all(jnp.isfinite(along) & jnp.isfinite(across)) & jnp.all(~active | nonzero)
        def derivative(value, frequency):
            return jnp.fft.ifft2(jnp.fft.fft2(value) * (1j * jnp.where(jnp.isfinite(frequency), frequency, 0.))).real
        def primitive(value, support):
            mask = active & (np.abs(fx) <= support[0]) & (np.abs(fy) <= support[1]) & nonzero
            factor = jnp.where(mask, 1 / (1j * jnp.where(mask, along, 1.)), 0.)
            return jnp.fft.ifft2(jnp.fft.fft2(value) * factor).real
        polynomial, term = jnp.ones_like(jacobian), jnp.ones_like(jacobian)
        for k in range(1, macro_order + 1):
            term *= -(jacobian - center) / center * (k + 2) / k
            polynomial += term
        phi = polynomial * primitive(p, band) / center**3
        psi = jacobian**3 * phi
        density, flow = -derivative(psi, along), derivative(psi, across)
        r = 3 * ej * phi + jacobian * derivative(phi, across)
        force = -(jacobian * derivative(r, across) + 1.5 * ej * r)
        dj2 = jnp.mean(dj**2)
        correction = -4 * jnp.mean(force) / jnp.where(dj2 > 0, dj2, 1.)
        pressure = primitive(force + .25 * correction * dj**2, (macro_order + 3) * band)
        energy = 3 * pressure + correction * jacobian * dj
        heat_flow = -primitive(derivative(energy, across), (macro_order + 3) * band)
        a2, h0, h2 = (3 * pressure - energy) / np.sqrt(3), np.sqrt(1.5) * density, -np.sqrt(2 / 3) * energy
        b1, c1, c3 = np.sqrt(2) * flow, np.sqrt(5) * flow, -2 * heat_flow / np.sqrt(5)
        zero = jnp.zeros_like(jacobian)
        def even_residual(bv, cv, av):
            db, dc = derivative(bv, along), derivative(cv, along)
            return jnp.stack((jacobian * (np.sqrt(6) * db / 3 - 2 * np.sqrt(15) * dc / 15)
                - dj * (np.sqrt(6) * bv / 4 - np.sqrt(15) * cv / 10) + jacobian * derivative(av, across),
                -np.sqrt(6) * jacobian * dc / 3,
                np.sqrt(210) * (jacobian * dc / 15 - dj * cv / 20)), -1)
        def odd_residual(hv, av, cv):
            dh, da = derivative(hv, along), derivative(av, along)
            return jnp.stack((jacobian**2 * (np.sqrt(30) * dh / 6 - 2 * np.sqrt(15) * da / 15)
                - np.sqrt(15) * jacobian * dj * av / 10 + jacobian * derivative(cv, across) - .5 * ej * cv,
                3 * np.sqrt(10) * (jacobian**2 * da - .5 * jacobian * dj * av) / 10), -1)
        stress = jnp.asarray([np.sqrt(3) / 4, 0., -np.sqrt(42) / 8])
        even = jnp.stack((even_residual(b1, c1, zero), even_residual(zero, c3, a2),
                         dj[..., None] * stress, (jacobian * ej - p * dj)[..., None] * stress), -2)
        odd = jnp.stack((odd_residual(h0, zero, zero), odd_residual(h2, a2, c1), odd_residual(zero, zero, c3),
                        jnp.stack((-np.sqrt(5) * ej / 4, zero), -1),
                        jnp.stack((-np.sqrt(5) * (jacobian * ep - .5 * p * ej) / 2, zero), -1)), -2)
        s = np.sqrt(2 / np.pi)
        even_metric = s * jnp.asarray([[32 / 15, 0., -48 * np.sqrt(14) / 245],
            [0., 16 / 5, 0.], [-48 * np.sqrt(14) / 245, 0., 3856 / 1715]])
        odd_metric = jnp.diag(jnp.stack((4 * mu1 / 5, jnp.asarray(64 * s / 45))))
        def gram(values, metric, weight):
            values = values.reshape((-1, values.shape[-2], values.shape[-1]))
            return jnp.einsum("piv,vw,pjw,p->ij", values, metric, values, weight.ravel()) / (np.prod(shape) * center)
        even_gram, odd_gram = gram(even, even_metric, w3), gram(odd, odd_metric, w4)
        collision_modes = jnp.stack((c1, c3, np.sqrt(5) * jnp.ones_like(p) / 2, np.sqrt(5) * p / 2), -1)[..., None]
        collision_gram = 16 * s / 15 * gram(collision_modes, jnp.ones((1, 1)), jnp.ones_like(jacobian))
        stress_collision = 8 * s / 5 * jnp.mean(jacobian * a2**2) / center
        electric = field * center
        z, one = jnp.zeros_like(electric), jnp.ones_like(electric)
        even_map = jnp.stack((jnp.stack((electric, electric**3, z, z), -1),
                             jnp.stack((z, z, one, z), -1), jnp.stack((z, z, z, one), -1)), -1)
        odd_map = jnp.stack((jnp.stack((one, electric**2, electric**4, z, z), -1),
                            jnp.stack((z, z, z, electric, z), -1), jnp.stack((z, z, z, z, electric), -1)), -1)
        residual = jnp.einsum("...ki,kl,...lj->...ij", even_map, even_gram, even_map)
        residual += jnp.einsum("...ki,kl,...lj->...ij", odd_map, odd_gram, odd_map)
        collision = jnp.einsum("...ki,kl,...lj->...ij", even_map, collision_gram, even_map)
        collision = collision.at[..., 0, 0].add(electric**4 * stress_collision)
        hessian = nu[..., None, None] * collision[..., :2, :2] + residual[..., :2, :2] / nu[..., None, None]
        rhs = residual[..., :2, 2] / nu[..., None] - nu[..., None] * collision[..., :2, 2]
        diagonal = jnp.diagonal(hessian, axis1=-2, axis2=-1)
        scale = jnp.sqrt(jnp.where(diagonal > 0, diagonal, 1.))
        cross = hessian[..., 0, 1] / (scale[..., 0] * scale[..., 1])
        determinant = 1 - cross**2
        regular = (jnp.all(jnp.isfinite(hessian), axis=(-2, -1)) & jnp.all(diagonal > 0, axis=-1)
                   & (determinant > 256 * jnp.finfo(center.dtype).eps) & jnp.all(jnp.isfinite(rhs), axis=-1))
        rhs = jnp.where(regular[..., None], rhs / scale, 0.)
        cross = jnp.where(regular, cross, 0.)
        trial = (rhs - cross[..., None] * rhs[..., ::-1]) / jnp.where(regular, determinant, 1.)[..., None] / scale
        positive = jnp.concatenate((trial, one[..., None]), -1)
        negative = jnp.concatenate((-trial, one[..., None]), -1)
        upper = nu * jnp.einsum("...i,...ij,...j->...", positive, collision, positive)
        upper += jnp.einsum("...i,...ij,...j->...", negative, residual, negative) / nu
        source_free = jnp.all(ej == 0)
        upper = jnp.where(source_free, 0., upper)
        valid &= macro_chart & (regular | source_free)
        constant = trial[..., 1]
        extra["scalar_upper"] = jnp.where(scalar_valid, scalar_upper, jnp.inf)
    valid &= jnp.isfinite(upper) & (upper >= lower)
    return dict(lower=jnp.where(valid, lower, 0.), upper=jnp.where(valid, upper, jnp.inf),
                valid=valid, relative_gap=jnp.where(valid, (upper - lower) / jnp.where(upper > 0, upper, 1.), 1.),
                geometric_heat_factor=geometric, optimal_heat_flow_constant=constant,
                integration_remainders=jnp.stack(tails), relative_fourier_amplitude=relative_amplitude, **extra)


class ConservingHeatBound(_ResidualTerm):
    """Full-Coulomb stellarator upper diffusivity, in m²/s, as a VMEX loss.

    ``collision_strengths`` are Γ*n/(m²*vth³) in s⁻¹, where
    Γ=(Z*e)^4*logΛ/(8*pi*epsilon0²); they are not Lorentz deflection rates.
    ``radial_electric_fields`` are −dΦ/dr in V/m. The flux radius is
    r=minor_radius*sqrt(psi/psi_edge), with psi the signed toroidal flux/(2π).
    The source uses the pure-temperature force with zero electrochemical force X1.

    Fits a finite Fourier J=Bhat⁻² on a fixed grid and reports its sampled field
    error separately. Increase ``geometry_order``, ``fit_grid`` and Boozer
    resolution independently. ``options`` pass to conserving_stellarator_proxy;
    its collision, trajectory, resonance and rounding qualifications apply.
    """

    name = "conserving_heat_upper_bound"

    def __init__(self, surfaces, *, temperature_ev, collision_strengths, radial_electric_fields,
                 mass_kg=3.3435837724e-27, charge_number=1., geometry_order=(4, 2), fit_grid=128,
                 helicity=0, mboz=8, nboz=8, oversample=2, scale=1., smoothing=.05, **options):
        self.surfaces = np.atleast_1d(np.asarray(surfaces, dtype=float))
        self.temperature = np.broadcast_to(np.asarray(temperature_ev, dtype=float), self.surfaces.shape)
        self.frequencies, self.fields = (np.atleast_1d(np.asarray(x, dtype=float))
                                        for x in (collision_strengths, radial_electric_fields))
        if (self.surfaces.ndim != 1 or not self.surfaces.size or not np.all(np.isfinite(self.surfaces))
                or np.any(self.surfaces <= 0) or np.any(self.surfaces >= 1)
                or np.any(~np.isfinite(self.temperature)) or np.any(self.temperature <= 0)
                or any(x.ndim != 1 or not x.size or not np.all(np.isfinite(x)) for x in (self.frequencies, self.fields))
                or np.any(self.frequencies <= 0) or not np.all(np.isfinite([mass_kg, charge_number, scale, smoothing]))
                or mass_kg <= 0 or charge_number == 0 or scale <= 0 or smoothing <= 0):
            raise ValueError("require interior surfaces, positive temperatures, rates, mass and scales, and finite fields/charge")
        order = np.asarray(geometry_order)
        if (order.shape != (2,) or np.any(order < 0) or np.any(order != order.astype(int))
                or int(fit_grid) != fit_grid or fit_grid <= 2 * max(order)):
            raise ValueError("geometry_order must be two nonnegative integers; fit_grid must resolve both")
        self.wave = np.stack(np.meshgrid(*(np.arange(-k, k + 1) for k in order.astype(int)), indexing="ij"), -1).reshape(-1, 2)
        self.mass, self.charge, self.fit_grid, self.helicity = mass_kg, charge_number, int(fit_grid), helicity
        self.mboz, self.nboz, self.oversample = int(mboz), int(nboz), int(oversample)
        self.scale, self.smoothing, self.options = float(scale), float(smoothing), options

    def compute_state(self, state, rt):
        """Return surface × collision × field intervals and a composable loss."""
        booz, minor = self._boozer(state, rt), _aspect_scalars(state, rt)[0]
        band = np.array([np.asarray(booz["xm_b"]),
                         (self.helicity * np.asarray(booz["xm_b"]) - np.asarray(booz["xn_b"])) / booz["nfp"]])
        if self.fit_grid <= 2 * np.max(np.abs(band)):
            raise ValueError("fit_grid must resolve the incoming Boozer spectrum in the helical chart")
        angle = np.arange(self.fit_grid) * (2 * np.pi / self.fit_grid)
        x, y = np.meshgrid(angle, angle, indexing="ij")
        phase = (np.asarray(booz["xm_b"])[:, None, None] * (x + self.helicity * y / booz["nfp"])
                 - np.asarray(booz["xn_b"])[:, None, None] * y / booz["nfp"])
        constant = np.flatnonzero((np.asarray(booz["xm_b"]) == 0) & (np.asarray(booz["xn_b"]) == 0))[0]
        records, errors = [], []
        for k, temperature in enumerate(self.temperature):
            cosine, sine = booz["bmnc_b"][k], booz["bmns_b"][k]
            b0 = cosine[constant]
            field = (jnp.einsum("m,mij->ij", cosine, np.cos(phase))
                     + jnp.einsum("m,mij->ij", sine, np.sin(phase))) / b0
            coefficients = jnp.fft.fft2(field**-2)[self.wave[:, 0] % self.fit_grid,
                                                      self.wave[:, 1] % self.fit_grid] / self.fit_grid**2
            spectrum = jnp.zeros((self.fit_grid, self.fit_grid), dtype=coefficients.dtype)
            spectrum = spectrum.at[self.wave[:, 0] % self.fit_grid, self.wave[:, 1] % self.fit_grid].set(coefficients)
            fitted = jnp.fft.ifft2(spectrum).real * self.fit_grid**2
            errors.append(jnp.max(jnp.abs(fitted**-.5 / field - 1)))
            psi, edge = booz["psi_b"][k], booz["psi_edge"]
            dr_dpsi = minor / (2 * edge * jnp.sqrt(psi / edge))
            tau = jnp.abs(booz["G_b"][k] + booz["iota_b"][k] * booz["I_b"][k]) / (
                b0 * jnp.sqrt(2 * 1.602176634e-19 * temperature / self.mass))
            result = conserving_stellarator_proxy(
                self.wave, coefficients, iota=booz["iota_b"][k], G_b=booz["G_b"][k], I_b=booz["I_b"][k],
                nfp=booz["nfp"], helicity=self.helicity, collision_frequency=self.frequencies[:, None] * tau,
                electric_field=-self.fields[None, :] * dr_dpsi * tau, **self.options)
            factor = (2 * temperature / self.charge)**2 * tau * dr_dpsi**2
            usable = (jnp.isfinite(factor) & (factor > 0) & (psi / edge > 0) & (psi / edge < 1)
                      & (minor > 0) & (b0 > 0) & jnp.all(field > 0))
            result["valid"] &= usable
            result["lower"] = jnp.where(result["valid"], factor * result["lower"], 0.)
            result["upper"] = jnp.where(result["valid"], factor * result["upper"], jnp.inf)
            if "scalar_upper" in result:
                result["scalar_upper"] = jnp.where(usable, factor * result["scalar_upper"], jnp.inf)
            result["relative_gap"] = jnp.where(result["valid"], result["relative_gap"], 1.)
            records.append(result)
        out = jax.tree.map(lambda *values: jnp.stack(values), *records)
        total = smooth_worst_case_bound(out["upper"], scale=self.scale, smoothing=self.smoothing)
        out.update(total=total, residuals1d=jnp.atleast_1d(jnp.sqrt(jnp.where(total > 0, total, 1.)))
                   * jnp.where(total > 0, 1., 0.), geometry_fit_error_sampled=jnp.stack(errors))
        return out


class NeoclassicalHeatBound(_ResidualTerm):
    """Composable VMEX loss for the worst sampled upper heat bound.

    Use full ``pitch_fractions`` or explicitly frozen physical ``pitch_edges``
    and a fixed positive seed ``scale``. ``total_state`` plugs into
    ``VmecProblem.from_loss``; residual rows implement the least-squares
    contract. The unnormalized smooth maximum covers supplied surfaces,
    frequencies and electric fields. Geometry options pass to the adapter;
    ``continuous_energy=True`` selects full-axis Maxwell integration.
    Derivatives are branch-local. Invalid geometry returns infinity and
    requires rejection. Refine geometry and trial resolution independently;
    the certificate applies only to the declared reference model."""

    name = "neoclassical_heat_upper_bound"

    def __init__(
        self, surfaces, pitch_edges=None, *, pitch_fractions=None, temperature_ev, collision_frequencies,
        radial_electric_fields, scale=1., smoothing=.05, mboz=8, nboz=8,
        oversample=2, **options,
    ):
        self.surfaces = np.atleast_1d(np.asarray(surfaces, dtype=float))
        if (pitch_edges is None) == (pitch_fractions is None):
            raise ValueError("supply exactly one of pitch_edges and pitch_fractions")
        self.pitch_edges = None if pitch_edges is None else np.asarray(pitch_edges, dtype=float)
        self.pitch_fractions = None if pitch_fractions is None else np.asarray(pitch_fractions, dtype=float)
        grid = self.pitch_fractions if self.pitch_edges is None else self.pitch_edges
        if (self.surfaces.ndim != 1 or not self.surfaces.size
                or not np.all(np.isfinite(self.surfaces)) or np.any(self.surfaces <= 0)
                or np.any(self.surfaces >= 1)):
            raise ValueError("surfaces must be a nonempty list strictly between zero and one")
        if (grid.ndim not in (1, 2) or grid.shape[-1] < 2 or not np.all(np.isfinite(grid))
                or np.any(np.diff(grid, axis=-1) <= 0)
                or (self.pitch_edges is not None and np.any(grid <= 0))
                or (self.pitch_fractions is not None and (np.any(grid < 0) or np.any(grid > 1)))):
            raise ValueError("pitch grid must increase: positive physical edges or normalized fractions in [0,1]")
        if not np.isfinite(scale) or scale <= 0 or not np.isfinite(smoothing) or smoothing <= 0:
            raise ValueError("scale and smoothing must be fixed positive finite numbers")
        self.scale, self.smoothing = float(scale), float(smoothing)
        self.mboz, self.nboz, self.oversample = int(mboz), int(nboz), int(oversample)
        if "nalpha" not in options:
            transverse_band = (self.mboz if options.get("helicity_m", 0) == 0 else
                               self.nboz + abs(options.get("helicity_n", 1)) * self.mboz)
            options["nalpha"] = 2 * transverse_band + 1
        self.options = dict(options, temperature_ev=temperature_ev,
                            collision_frequencies=collision_frequencies,
                            radial_electric_fields=radial_electric_fields)

    def compute_state(self, state, rt):
        """Return all intervals, validity diagnostics and the scalar loss."""
        out = neoclassical_heat_bounds_state(
            state, rt, surfaces=self.surfaces, pitch_edges=self.pitch_edges, pitch_fractions=self.pitch_fractions,
            mboz=self.mboz, nboz=self.nboz, oversample=self.oversample, **self.options)
        total = smooth_worst_case_bound(out["upper"], scale=self.scale, smoothing=self.smoothing)
        positive = total > 0
        residual = jnp.where(positive, jnp.sqrt(jnp.where(positive, total, 1.)), 0.)
        out.update(total=total, residuals1d=jnp.atleast_1d(residual))
        return out
