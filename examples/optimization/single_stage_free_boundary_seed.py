#!/usr/bin/env python
r"""Seed of a free-boundary single-stage run: a rotating ellipse at the target beta and coils fitted to it.

The seed is built in four stages, each to convergence:

1. the case's rotating ellipse (``SEED = (nfp, aspect, b / a_eff)`` of
   ``single_stage_free_boundary_optimization_three_terms.py``), no hand-designed deck;
2. its fixed-boundary equilibrium, solved directly at the target pressure (no
   ramp from vacuum), PHIEDGE and the pressure calibrated to the axis |B|
   ``B0`` and the target beta; with ``--bootstrap`` the current on every
   half-mesh surface is Newton-solved to Redl's bootstrap current at the fixed
   boundary (``ThreeTermsFreeBoundaryModel(fixed_boundary=True)``) and the deck
   recalibrated;
3. coils fitted to that equilibrium: ``(B_coils + B_plasma) . n = 0`` with
   ``B_plasma`` from virtual casing (zero in vacuum), under the case's coil
   rows, by Levenberg-Marquardt with the exact Jacobian (``fit_coils_exact``),
   run to convergence and checked against the case's own coil rows;
4. the free boundary of those coils (Newton from the deck, a few steps because
   the coils hold it), written as the deck, and the deck and coils scaled so
   that its axis |B| is ``B0`` (``scaled_to_axis_field``: an exact invariance,
   the free boundary is unchanged); the three-terms example starts from this
   seed (``--seed-run``) at the root of its free-boundary equations.

Vacuum (``--beta 0``) takes the same path with no pressure and no plasma field.
The rotating ellipse is a QA start; QH and QI seeds need a starting shape matched to the configuration (future work).
Stages 1-4 run here and write ``input.run`` (the deck), ``coils.initial.json``
and ``seed_report.json`` (wall time and result of every stage) to ``--output``;
then

    COIL_CASE=qa4-beta python single_stage_free_boundary_seed.py --beta 0.025 --bootstrap --modes 8 8 \
        --output runs/seed-qa4
    COIL_CASE=qa4-beta python single_stage_free_boundary_optimization_three_terms.py --beta 0.025 --bootstrap \
        --modes 8 8 --seed-run runs/seed-qa4 --output runs/three-terms-qa4

QA nfp 2 (``qa4-beta``: A 4, b/a 0.5, 4 coils per half period), 8 x 8, ns 51,
A100: the seed (stages 1-4) takes 188 s in vacuum and 448 s at <beta> 2.5 %
with the Redl current in the solve; the three-terms example then took 3 SLSQP
steps from each without a failed trial (869 s and 1729 s end to end).
"""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import single_stage_free_boundary_optimization_three_terms as tt  # noqa: E402  (the case table and coil rows)

# ---- stage 2 -----------------------------------------------------------------------------------------------------
# Relative tolerances of the calibration: axis |B| to B0 and the volume (or axis) beta to its target.  Below the
# solves' own reproducibility (~1e-6 in axis |B| at ftol 1e-15) they would only add solves.
CALIBRATION_TOLERANCE = (1e-5, 1e-4)
CALIBRATION_SOLVES = 6             # the calibration converges in 2 (vacuum) to 5 solves (2.5 %)

# ---- stage 3 -----------------------------------------------------------------------------------------------------
SEED_COIL_CLEARANCE = 1.5          # start coils: cross sections offset by this x COIL_SURFACE_DISTANCE_LIMIT
FIT_GRID = (32, 32)                # (nphi, ntheta) per field period of the B.n rows
FIT_POINTS = 1024                  # quadrature points per coil of the fit's per-point coil rows
# (highest coil Fourier order fitted, penalty weight W on the coil rows), in turn; a larger W only while the case's
# coil rows are violated.  Order 4: orders 5-8 lower B.n further (QA vacuum seed: 1.34e-3 -> 1.0e-3 at 5, 8.8e-4 at
# 8) but converge 4-20x slower in their weakly determined high modes (527-2541 Jacobians against 130-330); a seed
# needs coils that hold its boundary, and the optimization varies every COIL_ORDER coefficient.
FIT_SCHEDULE = ((4, 1.0), (4, 1e2), (4, 1e4))
# The fit's coil rows are tightened by FIT_BUFFER (scaled units): their sampling (FIT_POINTS per coil, point
# distances) is not the case rows' (CURVATURE_POINTS, segment distances); the largest gap measured on the QA seed was
# 6.5e-5, the buffer 15x that.  The penalty's smooth width is the buffer: it turns on within FIT_BUFFER of the
# tightened rows, before any case row reaches its limit, and is ~exp(-c / FIT_BUFFER) deeper inside.
FIT_BUFFER = 1e-3
FIT_TOL = 1e-6                     # converged: a step's predicted cost reduction below this x the cost
FIT_MAX_ITERATIONS = 3000          # a cap that converged fits never reach (130-1250 iterations measured)
FIT_FEASIBILITY = 1e-6             # every scaled case coil row must end >= -this


# ---- stages 1 and 2: the fixed boundary at the target beta ------------------------------------------------------
def calibrated_equilibrium(inp, beta, device, initial_state=None):
    """Solve ``inp`` at fixed shape, rescaling PHIEDGE to the axis |B| ``B0`` and PRES_SCALE to ``beta``.

    beta ~ p / PHIEDGE^2 at fixed shape, so a PHIEDGE rescale carries the pressure along.  Every solve is
    hot-started from the previous one.  Returns the calibrated input and its equilibrium.
    """
    from vmex import optimize as opt

    axis = tt.BETA_DEFINITION == "axis"
    state = initial_state
    for solves in range(1, CALIBRATION_SOLVES + 1):
        fixed = opt.solve_equilibrium(inp, initial_state=state, device=device, raise_on_max_iterations=True,
                                      polish_force_balance=False)
        state = fixed.state
        measured = float(fixed.wout.betaxis if axis else fixed.wout.betatotal)
        b_axis = float(opt.axis_field_strength(fixed.state, fixed.runtime))
        flux = tt.B0 / b_axis
        beta_ratio = beta / measured if beta > 0 else 1.0
        if abs(flux - 1) < CALIBRATION_TOLERANCE[0] and abs(beta_ratio - 1) < CALIBRATION_TOLERANCE[1]:
            break
        inp = replace(inp, phiedge=float(inp.phiedge) * flux, pres_scale=float(inp.pres_scale) * flux**2 * beta_ratio)
    else:
        raise RuntimeError(f"the calibration did not converge in {CALIBRATION_SOLVES} solves")
    print(f"[stage 2] {solves} solves: PRES_SCALE {float(inp.pres_scale):.6e} Pa, PHIEDGE {float(inp.phiedge):.6f} Wb,"
          f" <beta> {float(fixed.wout.betatotal):.4%}, axis |B| {b_axis:.6f} T, CURTOR {float(inp.curtor):.1f} A",
          flush=True)
    return inp, fixed


def target_beta_equilibrium(beta, bootstrap, device):
    """Stages 1-2: the rotating ellipse's fixed-boundary equilibrium at ``beta``, with its Redl current if asked.

    The pressure is p ~ the case's PRESSURE_SHAPE (the kinetic profiles' (1 - s)(1 - s^5)), solved at the target
    pressure from the start.  ``bootstrap``: the current on every half-mesh surface is Newton-solved to Redl's at
    the fixed boundary, then PHIEDGE and the pressure are recalibrated at that current.  The recalibration moves
    Redl's current by a few % (QA: 6 %); the free-boundary solve carries the current as unknowns and makes it
    Redl's again, so the deck needs no second round.  Returns the deck, its equilibrium and the kinetic profiles
    (or ``None``).
    """
    from vmex.core.freeboundary_vc import ThreeTermsFreeBoundaryModel

    inp = tt.seed_input()
    shape = np.zeros_like(np.asarray(inp.am, dtype=float))
    shape[: len(tt.PRESSURE_SHAPE)] = tt.PRESSURE_SHAPE
    ac = np.zeros_like(np.asarray(inp.ac, dtype=float))  # no current: a vacuum deck is then recognised as one
    # <p> of (1 - s)(1 - s^5) is p0 (1/2 - 1/6 + 1/7) and beta = 2 mu0 <p> / B0^2: the start of the calibration
    p0 = beta * tt.B0**2 / (2 * 4e-7 * np.pi * (1 / 2 - 1 / 6 + 1 / 7))
    inp = replace(inp, am=shape, pres_scale=p0, ncurr=1, pcurr_type="power_series", ac=ac, curtor=0.0)
    inp, fixed = calibrated_equilibrium(inp, beta, device)
    if not bootstrap:
        return inp, fixed, None
    profiles = tt.redl_profiles(inp)[0]
    model = ThreeTermsFreeBoundaryModel(inp, bootstrap=profiles, bootstrap_helicity=tt.REDL_HELICITY,
                                       fixed_boundary=True)
    current = model.solve_boundary(model.params0, None)
    if not current["converged"]:
        raise RuntimeError("the fixed-boundary Redl current did not converge")
    state, _, params, _ = current["aux"]
    print(f"[stage 2] Redl current: {current['nfev']} evaluations, max relative mismatch "
          f"{model.bootstrap_residual(state, params):.1e}", flush=True)
    inp, fixed = calibrated_equilibrium(model.bootstrap.deck_with(model.fixed, current["x"]), beta, device,
                                        initial_state=state)
    return inp, fixed, tt.redl_profiles(inp)[0]


# ---- stage 3: the coils ------------------------------------------------------------------------------------------
def offset_coils(inp, n_coils, order, distance):
    """``n_coils`` planar coils per half period: the boundary's cross sections offset by ``distance``.

    Coil k lies in the plane phi_k = (k + 1/2) pi / (nfp n_coils), as ESSOS' equally spaced curves, and follows the
    boundary's cross section there moved ``distance`` along its in-plane outward normal: their field already follows
    the boundary's shaping, so the fit starts near its answer.  Fourier series of ``order`` (ESSOS layout: the
    constant, then sin and cos of each order) fitted to the offset curve; unit currents.
    """
    import jax.numpy as jnp
    from essos.coils import Coils, Curves

    nfp, ntor = int(inp.nfp), int(inp.ntor)
    rbc, zbs = np.asarray(inp.rbc, dtype=float), np.asarray(inp.zbs, dtype=float)
    ms, ns = np.arange(rbc.shape[1]), np.arange(-ntor, ntor + 1)
    theta = np.linspace(0.0, 2 * np.pi, 512, endpoint=False)
    basis = np.stack([np.ones_like(theta)] + [f(k * theta) for k in range(1, order + 1) for f in (np.sin, np.cos)],
                     axis=1)
    dofs = np.zeros((n_coils, 3, 2 * order + 1))
    for k in range(n_coils):
        phi = (k + 0.5) * np.pi / (nfp * n_coils)
        angle = ms[None, :, None] * theta[None, None, :] - ns[:, None, None] * nfp * phi
        R = np.einsum("nm,nmt->t", rbc, np.cos(angle))
        Z = np.einsum("nm,nmt->t", zbs, np.sin(angle))
        dR = np.einsum("nm,nmt->t", -rbc * ms[None, :], np.sin(angle))
        dZ = np.einsum("nm,nmt->t", zbs * ms[None, :], np.cos(angle))
        normal = np.stack([dZ, -dR]) / np.hypot(dR, dZ)
        if np.mean((np.stack([R - R.mean(), Z - Z.mean()])) * normal) < 0:
            normal = -normal  # outward
        Rc, Zc = R + distance * normal[0], Z + distance * normal[1]
        dofs[k] = np.linalg.lstsq(basis, np.stack([Rc * np.cos(phi), Rc * np.sin(phi), Zc]).T, rcond=None)[0].T
    curves = Curves(jnp.asarray(dofs), n_segments=tt.N_SEGMENTS, nfp=nfp, stellsym=True)
    return Coils(curves, jnp.ones(n_coils), currents_scale=1.0)


def seed_coils(inp, wout, n_coils):
    """Start coils (``offset_coils``), their current giving the seed's edge R B_phi with its sign.

    The offset is ``SEED_COIL_CLEARANCE`` times the coil-surface clearance, shrunk toward the plain clearance until
    every coil is within the length limit.
    """
    import jax
    import jax.numpy as jnp
    from essos.coils import Coils
    from essos.fields import BiotSavart
    from vmex.core.virtual_casing import surface_field_data_from_wout

    for factor in np.linspace(SEED_COIL_CLEARANCE, 1.05, 8):
        distance = factor * tt.COIL_SURFACE_DISTANCE_LIMIT
        coils = offset_coils(inp, n_coils, tt.COIL_ORDER, distance)
        length = float(np.max(np.asarray(coils.curves.length)))
        if length <= tt.LENGTH_LIMIT - tt.LENGTH_MARGIN:
            break
    else:
        raise ValueError(f"no start coil fits the length limit: {length:.3f} > {tt.LENGTH_LIMIT:.3f} m")
    data = surface_field_data_from_wout(wout, nphi=16, ntheta=16)
    g, b = np.asarray(data.gamma), np.asarray(data.B_total)
    phi, radius = np.arctan2(g[1], g[0]), np.hypot(g[0], g[1])
    field = np.asarray(jax.vmap(BiotSavart(coils).B)(jnp.asarray(g.reshape(3, -1).T))).T.reshape(g.shape)
    current = (np.mean(radius * (-np.sin(phi) * b[0] + np.cos(phi) * b[1]))
               / np.mean(radius * (-np.sin(phi) * field[0] + np.cos(phi) * field[1])))
    print(f"[stage 3] {n_coils} start coils per half period, {distance:.4f} m off the boundary, length "
          f"{length:.3f} m (limit {tt.LENGTH_LIMIT:.3f}), current {current:.6e} A", flush=True)
    return Coils(coils.curves, jnp.full(n_coils, current))


def fit_penalty_rows(coils, surface_points):
    """Scaled coil rows c >= 0 of the fit, one per coil or per coil point.

    Per base coil: length and MSC; per base-coil point: curvature, the distance to the nearest point of every other
    coil (symmetry copies included) and to the nearest plasma-surface point.  A single minimum over all pairs (the
    case's rows) has a gradient that jumps between pairs and stalls a Gauss-Newton fit; a row per point keeps every
    violation in the residual.  The nearest points are located without gradient and the distances to them carry
    the exact gradient of the minimum.  Same limits and margins as ``coil_inequalities``.
    """
    import jax
    import jax.numpy as jnp

    base = tt.resampled(coils, FIT_POINTS, symmetric=False)
    speed = jnp.linalg.norm(base.gamma_dash, axis=-1)
    curvature = base.curvature
    msc = jnp.sum(curvature**2 * speed, axis=1) / jnp.sum(speed, axis=1)
    every = tt.resampled(coils, FIT_POINTS).gamma  # base coils first
    points = base.gamma

    def nearest(p, cloud, allowed=None):
        d2 = jnp.sum((jax.lax.stop_gradient(p)[:, None, :] - jax.lax.stop_gradient(cloud)[None]) ** 2, axis=-1)
        if allowed is not None:
            d2 = jnp.where(allowed[None, :], d2, jnp.inf)
        return jnp.linalg.norm(p - cloud[jnp.argmin(d2, axis=1)], axis=-1)

    flat = every.reshape(-1, 3)
    owner = jnp.repeat(jnp.arange(every.shape[0]), every.shape[1])
    coil_distance = jnp.stack([nearest(points[i], flat, owner != i) for i in range(points.shape[0])])
    surface_distance = jnp.stack([nearest(points[i], surface_points) for i in range(points.shape[0])])
    return jnp.concatenate([
        ((tt.LENGTH_LIMIT - tt.LENGTH_MARGIN - base.length) / tt.LENGTH_LIMIT).ravel(),
        ((tt.MSC_LIMIT - tt.MSC_MARGIN - msc) / tt.MSC_LIMIT).ravel(),
        ((tt.CURVATURE_LIMIT - tt.CURVATURE_MARGIN - curvature) / tt.CURVATURE_LIMIT).ravel(),
        ((coil_distance - tt.COIL_DISTANCE_LIMIT - tt.DISTANCE_MARGIN) / tt.COIL_DISTANCE_LIMIT).ravel(),
        ((surface_distance - tt.COIL_SURFACE_DISTANCE_LIMIT - tt.DISTANCE_MARGIN)
         / tt.COIL_SURFACE_DISTANCE_LIMIT).ravel()])


def levenberg_marquardt(fun, jac, x0, *, tol, max_iterations):
    """Minimize 0.5 |fun(x)|^2: Levenberg-Marquardt steps through the SVD of the exact Jacobian.

    Every step solves (J^T J + mu I) p = -J^T r by the SVD of J, so the directions J leaves undetermined (a coil's
    own parametrization phase) are not moved; mu follows Nielsen's update.  Converged when a step at the
    iteration's own damping predicts a cost reduction below ``tol`` times the cost (or the cost is below 1e-30);
    a step whose damping had to grow refreshes the Jacobian instead.  Returns ``(x, converged, iterations)``.
    """
    x = np.asarray(x0, dtype=float)
    r = fun(x)
    cost, nu, mu = 0.5 * float(r @ r), 2.0, None
    for iteration in range(1, max_iterations + 1):
        U, sv, Vt = np.linalg.svd(jac(x), full_matrices=False)
        g = U.T @ r
        if mu is None:
            mu = 1e-3 * float(sv[0]) ** 2
        first = True
        while True:
            q = -sv * g / (sv**2 + mu)          # the step in the right singular basis
            predicted = -(float(g @ (sv * q)) + 0.5 * float(np.sum((sv * q) ** 2)))
            if predicted <= tol * cost or cost < 1e-30:
                if first:
                    return x, True, iteration
                break
            trial = fun(x + Vt.T @ q)
            new = 0.5 * float(trial @ trial)
            ratio = (cost - new) / predicted
            if ratio > 0:
                x, r, cost = x + Vt.T @ q, trial, new
                mu *= max(1.0 / 3.0, 1.0 - (2.0 * ratio - 1.0) ** 3)
                nu = 2.0
                break
            mu, nu, first = mu * nu, nu * 2.0, False
    return x, False, max_iterations


def fit_coils_exact(coils, wout, inp, *, vacuum):
    """Stage 3: coils with (B_coils + B_plasma) . n = 0 on the seed boundary, under the coil rows.

    Least squares on the rows sqrt(w) (B_coils + B_plasma) . n / |B| (area weights w summing to one: their norm is
    the area-weighted RMS) and sqrt(W) pen(c - FIT_BUFFER) for the scaled per-point coil rows c of
    ``fit_penalty_rows``, pen(c) = -FIT_BUFFER softplus(-c / FIT_BUFFER), by ``levenberg_marquardt`` with the exact
    Jacobian (forward mode) in the coil coefficients up to each ``FIT_SCHEDULE`` order (the currents stay those of
    ``coils``).  Why smooth: with min(c, 0) a row's Jacobian jumps where it turns on, every active row is overshot
    and the fit crawled (QA vacuum seed: 900-1500 Jacobians, B.n 2.5e-3; smooth: 130-330, B.n 1.3e-3).
    Feasibility: after every round the case's own coil rows (``coil_inequalities`` and the clearance, as the
    optimization evaluates them) are checked; a violation beyond FIT_FEASIBILITY raises W, and one left after the
    last round is an error.  In vacuum B_plasma = 0 (no virtual casing).  Returns the coils and a report.
    """
    import dataclasses
    import jax
    import jax.numpy as jnp
    from essos.surfaces import surfacerzfourier_from_boundary
    import vmex as vj

    interface = vj.PlasmaVacuumInterface.from_wout(wout, nphi=FIT_GRID[0], ntheta=FIT_GRID[1])
    if vacuum:
        interface = dataclasses.replace(interface, B_plasma=jnp.zeros_like(interface.B_plasma),
                                        Bn_plasma=jnp.zeros_like(interface.Bn_plasma))
    surface = surfacerzfourier_from_boundary(jnp.asarray(inp.rbc), jnp.asarray(inp.zbs), inp.nfp,
                                             nphi=tt.SURFACE_GRID[0], ntheta=tt.SURFACE_GRID[1])
    surface_points = jnp.asarray(surface.gamma).reshape(-1, 3)
    x0 = jnp.asarray(coils.curves.dofs).ravel()
    root_w = jnp.sqrt(interface.weights).ravel()
    mode_order = np.broadcast_to((np.arange(coils.dofs_curves.shape[2]) + 1) // 2, coils.dofs_curves.shape).ravel()

    def coils_from_u(u):
        return coils.with_dofs(jnp.concatenate((x0 + tt.COIL_STEP * u, coils.dofs_currents)))

    def normal_rows(u):
        return root_w * tt.total_normal_field(interface, tt.coil_field(coils_from_u(u))).ravel()

    def case_rows(u):
        new = coils_from_u(u)
        return jnp.concatenate([tt.coil_inequalities(new), jnp.atleast_1d(
            (tt.surface_distance(new, surface) - tt.COIL_SURFACE_DISTANCE_LIMIT - tt.DISTANCE_MARGIN)
            / tt.COIL_SURFACE_DISTANCE_LIMIT)])

    def residuals(v, u, active, root_weight):
        full = u.at[active].set(v)
        c = fit_penalty_rows(coils_from_u(full), surface_points) - FIT_BUFFER
        return jnp.concatenate([normal_rows(full), root_weight * -FIT_BUFFER * jax.nn.softplus(-c / FIT_BUFFER)])

    fun, jac, rows_at = jax.jit(residuals), jax.jit(jax.jacfwd(residuals)), jax.jit(case_rows)
    u = np.zeros(x0.size)
    before = float(jnp.linalg.norm(normal_rows(jnp.asarray(u))))
    rounds, worst = [], -np.inf
    for top, weight in FIT_SCHEDULE:
        active = jnp.asarray(np.flatnonzero(mode_order <= top))
        u_j, root, started = jnp.asarray(u), jnp.asarray(np.sqrt(weight)), time.monotonic()
        x, converged, iterations = levenberg_marquardt(
            lambda v: np.asarray(fun(jnp.asarray(v), u_j, active, root)),
            lambda v: np.asarray(jac(jnp.asarray(v), u_j, active, root)),
            u[np.asarray(active)], tol=FIT_TOL, max_iterations=FIT_MAX_ITERATIONS)
        if not converged:
            raise RuntimeError(f"the coil fit did not converge in {FIT_MAX_ITERATIONS} iterations")
        u = np.asarray(u_j.at[active].set(x))
        worst = float(np.min(np.asarray(rows_at(jnp.asarray(u)))))
        rounds.append(dict(order=int(top), weight=weight, iterations=iterations, worst_row=worst,
                           normal_rms=float(jnp.linalg.norm(normal_rows(jnp.asarray(u)))),
                           seconds=time.monotonic() - started))
        print(f"[stage 3] order {top}, W {weight:.0e}: converged in {iterations} iterations, "
              f"{rounds[-1]['seconds']:.0f} s: B.n/|B| RMS {rounds[-1]['normal_rms']:.3e}, worst scaled coil row "
              f"{worst:+.2e}", flush=True)
        if top >= max(t for t, _ in FIT_SCHEDULE) and worst >= -FIT_FEASIBILITY:
            break
    if worst < -FIT_FEASIBILITY:
        raise RuntimeError(f"the fitted coils violate the case's coil rows ({worst:+.2e})")
    return coils_from_u(jnp.asarray(u)), dict(initial_normal_rms=before, rounds=rounds)


# ---- stage 4: the free boundary, and the field scale --------------------------------------------------------------
def coil_field_of(coils):
    """The three-terms example's differentiable filament field of ``coils``."""
    import jax
    import jax.numpy as jnp

    try:  # a pytree of its arrays (as CoilChart registers it)
        jax.tree_util.register_dataclass(tt.DirectCoilField, data_fields=["gamma", "gamma_dash", "currents"],
                                         meta_fields=[])
    except ValueError:  # registered already
        pass
    return tt.DirectCoilField(jnp.asarray(coils.gamma), jnp.asarray(coils.gamma_dash), jnp.asarray(coils.currents))


def scaled_to_axis_field(inp, coils, profiles, vc_grid, trial_ftol):
    """Stage 4: the free boundary of ``coils`` as the deck, scaled with the coils so that its axis |B| is ``B0``.

    The fitted coils hold the seed's boundary only to their B.n residual, so their free boundary differs from it
    and so does its axis |B| (QA: 2-5 % below B0, QH 7 %).  Ideal-MHD equilibria are invariant under B -> lambda B,
    p -> lambda^2 p, I -> lambda I: coil currents, PHIEDGE and the plasma current scaled by lambda = B0 / |B_axis|
    and the pressure by lambda^2 leave the free boundary unchanged and put its axis |B| at B0 (beta and iota are
    unchanged; the Redl current follows, with the kinetic profiles' collisionality kept by ``redl_profiles``).
    The solve uses the three-terms example's interface grid, and the deck written is that free boundary (with the
    solved current), so the example's first solve starts at its root.  Returns the scaled deck and coils and the
    solve's report.
    """
    from essos.coils import Coils
    from vmex import optimize as opt
    from vmex.core.freeboundary_vc import solve_free_boundary_three_terms

    started = time.monotonic()
    result = solve_free_boundary_three_terms(
        replace(inp, lfreeb=False), external_field=coil_field_of(coils), nphi=vc_grid, ntheta=vc_grid,
        quadrature=(4 * int(inp.nfp) * vc_grid, 2 * vc_grid), trial_ftol=trial_ftol, bootstrap=profiles,
        bootstrap_helicity=tt.REDL_HELICITY)
    if not result.success:
        raise RuntimeError(f"the seed coils' free boundary did not converge ({result.message})")
    eq = result.equilibrium
    b_axis = float(opt.axis_field_strength(eq.state, eq.runtime))
    factor = tt.B0 / b_axis
    # The solved free boundary (and, with the Redl current in the solve, its current) becomes the deck: the
    # three-terms example then starts at its root instead of solving it again from the fixed boundary.
    inp = replace(result.input, lfreeb=False)
    scaled = replace(inp, phiedge=float(inp.phiedge) * factor, pres_scale=float(inp.pres_scale) * factor**2,
                     curtor=float(inp.curtor) * factor,
                     **({} if inp.ac_aux_f is None else {"ac_aux_f": np.asarray(inp.ac_aux_f, dtype=float) * factor}),
                     ac=np.asarray(inp.ac, dtype=float) * factor)
    coils = Coils(coils.curves, coils.dofs_currents_raw * factor, currents_scale=coils.currents_scale)
    report = dict(nfev=int(result.nfev), b_axis=b_axis, factor=factor, seconds=time.monotonic() - started,
                  normal=result.boundary_residual.normal, normal_max=result.boundary_residual.normal_max,
                  pressure=result.boundary_residual.pressure,
                  sheet_current=result.boundary_residual.sheet_current)
    print(f"[stage 4] free boundary in {report['seconds']:.0f} s ({result.nfev} evaluations): "
          f"{result.boundary_residual}, axis |B| {b_axis:.5f} T -> fields x {factor:.5f}", flush=True)
    return scaled, coils, report


def build_rotating_ellipse_seed(beta, bootstrap, device="gpu", n_coils=None, vc_grid=48, trial_ftol=1e-11):
    """Stages 1-4; returns ``(inp, fixed, profiles, coils, report)`` (``report``: every stage's wall time).

    ``fixed`` is the stage-2 equilibrium (before the stage-4 field scale)."""
    from essos.coils import Coils

    timings, started = {}, time.monotonic()
    inp, fixed, profiles = target_beta_equilibrium(beta, bootstrap, device)
    timings["fixed_boundary_s"] = time.monotonic() - started
    started = time.monotonic()
    if n_coils is None:  # the case's coil count
        n_coils = int(Coils.from_json(str(tt.COILS_FILE)).dofs_curves.shape[0])
    coils = seed_coils(inp, fixed.wout, n_coils)
    coils, fit = fit_coils_exact(coils, fixed.wout, inp, vacuum=beta == 0 and not bootstrap)
    timings["coil_fit_s"] = time.monotonic() - started
    inp, coils, free = scaled_to_axis_field(inp, coils, profiles, vc_grid, trial_ftol)
    timings["free_boundary_s"] = free["seconds"]
    return inp, fixed, profiles, coils, dict(timings=timings, fit=fit, free_boundary=free,
                                             beta=float(fixed.wout.betatotal), curtor=float(inp.curtor),
                                             phiedge=float(inp.phiedge))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="new seed directory")
    parser.add_argument("--beta", type=float, default=0.0, help="target beta (the case's BETA_DEFINITION)")
    parser.add_argument("--bootstrap", action="store_true", help="the Redl current in the solve (needs --beta > 0)")
    parser.add_argument("--modes", type=int, nargs=2, metavar=("MPOL", "NTOR"), help="default: the case's RESOLUTION")
    parser.add_argument("--ns", type=int, help="radial resolution (default: the case's RESOLUTION)")
    parser.add_argument("--ellipse", type=float, nargs=3, metavar=("NFP", "ASPECT", "RATIO"),
                        help="rotating ellipse (default: the case's SEED)")
    parser.add_argument("--coils-per-half-period", type=int, help="default: the case coil file's")
    parser.add_argument("--vc-grid", type=int, default=48, help="interface grid per field period of the stage-4 "
                        "free boundary (pass the three-terms run's --vc-grid)")
    parser.add_argument("--trial-forward-ftol", type=float, default=1e-11,
                        help="force residual of the stage-4 trial equilibria (pass the three-terms run's)")
    args = parser.parse_args(argv)
    if args.bootstrap and not args.beta > 0:
        parser.error("--bootstrap needs --beta > 0")
    if args.bootstrap and not (tt.BOOTSTRAP_IN_SOLVE and tt.BOOTSTRAP_MODEL == "redl"):
        parser.error("--bootstrap needs a case with the Redl current in the solve (BOOTSTRAP_IN_SOLVE)")
    return args


def main(argv=None):
    """Build the seed and write ``input.run``, ``coils.initial.json`` and ``seed_report.json``."""
    args = parse_args(argv)
    os.environ["JAX_ENABLE_X64"] = "1"
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    tt.RESOLUTION = (*(args.modes or tt.RESOLUTION[:2]), args.ns or tt.RESOLUTION[2])
    if args.ellipse is not None:
        tt.SEED = (int(args.ellipse[0]), float(args.ellipse[1]), float(args.ellipse[2]))
    if tt.SEED is None:
        raise SystemExit(f"case {tt.CASE} has no rotating-ellipse SEED; pass --ellipse")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inp, _, _, coils, report = build_rotating_ellipse_seed(args.beta, args.bootstrap, "gpu",
                                                           args.coils_per_half_period, args.vc_grid,
                                                           args.trial_forward_ftol)
    inp.to_indata(out / "input.run")
    coils.to_json(str(out / "coils.initial.json"))
    report.update(case=tt.CASE, seed=list(tt.SEED), resolution=list(tt.RESOLUTION))
    (out / "seed_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["timings"]), flush=True)


if __name__ == "__main__":
    main()
