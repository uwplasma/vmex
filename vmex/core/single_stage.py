"""Joint boundary/coil objectives and independent held-boundary checks.

The caller supplies a ``VmecProblem`` and an optimizer. ESSOS and virtual
casing are optional dependencies, required only by these coil workflows.
"""

from dataclasses import dataclass, replace
import gc
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

import vmex as vj
from . import optimize as opt, virtual_casing as vc
from .errors import VmecError


def coil_field(coils):
    """Batched Cartesian Biot–Savart field, preserving the point-array shape."""
    from essos.fields import BiotSavart
    field = BiotSavart(coils)
    return lambda points: jax.vmap(field.B)(points.reshape(-1, 3)).reshape(points.shape)


def _resample_coils(coils, n_segments):
    """Evaluate unchanged Fourier curves on an independent geometry grid."""
    geometry = coils.copy()
    geometry.n_segments = max(n_segments, coils.n_segments)
    return geometry


def common_current_scale(coil_B, external_target_B, weights):
    """Area-weighted least-squares common current factor for a fixed target.

    Fields use Cartesian ``(3, nphi, ntheta)`` arrays. At finite beta the
    external target is ``B_VMEC - B_plasma``; total PHIEDGE is not the coil
    flux. This calibration fits boundary field strength, not exact flux or
    pressure balance. The caller must reject a nonfinite or zero factor.
    """
    numerator = jnp.sum(weights * jnp.sum(coil_B * external_target_B, axis=0))
    denominator = jnp.sum(weights * jnp.sum(coil_B**2, axis=0))
    return numerator / denominator


def toroidal_flux(inp, coils, *, nr=24, ntheta=128):
    """Physical +phi flux through the stellarator-symmetric phi=0 section.

    Integrates ``B_phi dR dZ``; +phi is Cartesian +Y here. Increasing poloidal
    angle traverses this section with normal -Y, so Stokes' line integral of
    the vector potential has the opposite sign.
    """
    if inp.lasym:
        raise ValueError("toroidal_flux requires stellarator symmetry")
    rho, weights = np.polynomial.legendre.leggauss(nr)
    rho, weights = (rho + 1.0) / 2, weights / 2
    theta = np.arange(ntheta) * (2 * np.pi / ntheta)
    m = np.arange(inp.rbc.shape[1])
    rc, zs = inp.rbc.sum(axis=0), inp.zbs.sum(axis=0)
    phase = np.outer(theta, m)
    r, z = np.cos(phase) @ rc, np.sin(phase) @ zs
    dr, dz = -np.sin(phase) @ (m * rc), np.cos(phase) @ (m * zs)
    R, Z = rc[0] + rho[:, None] * (r - rc[0]), rho[:, None] * z
    area = rho[:, None] * ((r - rc[0]) * dz - z * dr)
    points = np.stack((R, np.zeros_like(R), Z), axis=-1)
    bphi = np.asarray(coil_field(coils)(jnp.asarray(points)))[..., 1]
    return float(np.sum(weights[:, None] * bphi * area) * (2 * np.pi / ntheta))


def hinge(constraints, weight):
    """Quadratic penalty for dimensionless constraints written as c >= 0."""
    return 0.5 * weight * jnp.sum(jnp.maximum(-constraints, 0.0)**2)


@dataclass(frozen=True)
class CoilPenalty:
    """One-sided coil geometry costs, using only unique coils for self costs."""

    length_target: float
    curvature_limit: float
    coil_distance: float
    surface_distance: float
    length_weight: float = 0.5
    curvature_weight: float = 10.0
    distance_weight: float = 1e4
    surface_weight: float = 1e4
    normalized: bool = False
    geometry_segments: int | None = None
    surface_grid: tuple[int, int] | None = None

    def __call__(self, coils, surface):
        """Return length, curvature, separation, and clearance costs."""
        from essos.objective_functions import loss_coil_separation, loss_coil_surface_distance
        n = coils.dofs_currents.size
        if self.geometry_segments is not None:
            coils = _resample_coils(coils, self.geometry_segments)
        if self.surface_grid is not None:
            from essos.surfaces import SurfaceRZFourier
            surface = SurfaceRZFourier(
                surface.rc, surface.zs, surface.nfp, surface.mpol, surface.ntor,
                nphi=self.surface_grid[0], ntheta=self.surface_grid[1], close=False,
                range_torus="full torus", rs=surface.rs, zc=surface.zc)
        if self.normalized:
            # Smooth worst-case measures: the p = 8 power mean of the sampled violations tends to their
            # maximum, is unchanged by extra nodes, and (unlike a min or max) leaves L-BFGS a gradient.
            def worst(violation, p=8):
                return (jnp.mean(violation**p) + 1e-40)**(1 / p)

            length = jnp.maximum(coils.length[:n] / self.length_target - 1, 0)
            curvature = jnp.maximum(coils.curvature[:n] / self.curvature_limit - 1, 0)
            gamma, points = coils.gamma, surface.gamma.reshape(-1, 3)

            def gap(first, second, limit):
                distance = jnp.sqrt(jnp.min(jnp.sum((first[:, None] - second[None, :])**2, axis=-1), axis=1) + 1e-30)
                return jnp.maximum(1 - distance / limit, 0)  # per node of ``first``

            pairs = jnp.triu_indices(len(coils), 1)
            coil_gap = (worst(jax.vmap(lambda i, j: gap(gamma[i], gamma[j], self.coil_distance))(*pairs))
                        if len(coils) > 1 else 0.)
            surface_gap = worst(jax.vmap(lambda curve: gap(curve, points, self.surface_distance))(gamma[:n]))
            return .5 * jnp.asarray([
                self.length_weight * jnp.mean(length**2),
                self.curvature_weight * worst(curvature)**2,
                self.distance_weight * coil_gap**2, self.surface_weight * surface_gap**2])
        length = jnp.sqrt(self.length_weight) * jnp.maximum(coils.length[:n] - self.length_target, 0)
        curvature = jnp.sqrt(self.curvature_weight) * jnp.maximum(
            coils.curvature[:n] - self.curvature_limit, 0)
        return jnp.asarray([
            0.5 * jnp.vdot(length, length), 0.5 * jnp.vdot(curvature, curvature),
            0.5 * self.distance_weight * loss_coil_separation(coils, self.coil_distance, block_size=32),
            0.5 * self.surface_weight * loss_coil_surface_distance(
                coils, surface, self.surface_distance, block_size=32)])


def _vacuum_input(inp):
    """Recognize prescribed zero pressure and current, without solving VC."""
    zero_pressure = (inp.pres_scale == 0 or
                     (inp.pmass_type == "power_series" and np.all(np.asarray(inp.am) == 0)))
    return (zero_pressure and inp.ncurr == 1 and inp.curtor == 0
            and inp.pcurr_type == "power_series" and np.all(np.asarray(inp.ac) == 0))


def total_normal_field(inp, coils, state, runtime, nphi, ntheta, precision, *, digits=4):
    """Total-field normal residual over ``|B|``, area weights, and pressure jump over ``|B|^2``."""
    data = vc.surface_field_data_from_state(inp, state, runtime=runtime, nphi=nphi, ntheta=ntheta)
    B2 = jnp.sum(data.B_total**2, axis=0)
    if _vacuum_input(inp):
        field = vc.external_B_cartesian(coil_field(coils), data.gamma, data.phi)
        area = jnp.linalg.norm(data.area_vector, axis=0)
        return (jnp.sum(field * data.normal, axis=0) / jnp.sqrt(B2), area / jnp.sum(area),
                jnp.sum(field**2, axis=0) / B2 - 1)
    interface = vc.PlasmaVacuumInterface.from_surface_data(data, digits=digits, precision=precision)
    return (interface.bnormal_residual(coil_field(coils)) / jnp.sqrt(B2), interface.weights,
            interface.pressure_balance_residual(coil_field(coils)) / B2)


def coils_around_boundary(inp, n_coils, order, offset, *, n_segments=64, stellsym=True, current=1.0):
    """Planar-section coils: the boundary's R-Z cross-section pushed ``offset`` outward, one per angle.

    The angles are those of ``CreateEquallySpacedCurves``. Unlike circles, these start outside
    boundaries with large axis excursions. The curve is the section's outward normal offset,
    truncated to ``order`` Fourier harmonics.
    """
    from essos.coils import Coils, Curves
    n, m = np.meshgrid(np.arange(-inp.ntor, inp.ntor + 1), np.arange(inp.rbc.shape[1]), indexing="ij")
    theta = np.linspace(0, 2 * np.pi, 8 * order * 2 + 1)[:-1]
    t = theta / (2 * np.pi)
    dofs = np.zeros((n_coils, 3, 2 * order + 1))
    for i, phi in enumerate((np.arange(n_coils) + .5) * 2 * np.pi / ((1 + int(stellsym)) * inp.nfp * n_coils)):
        arg = m[None] * theta[:, None, None] - n[None] * inp.nfp * phi
        dm = m[None] * np.ones_like(arg)
        R, Z = (inp.rbc[None] * np.cos(arg)).sum((1, 2)), (inp.zbs[None] * np.sin(arg)).sum((1, 2))
        dR = -(inp.rbc[None] * dm * np.sin(arg)).sum((1, 2))
        dZ = (inp.zbs[None] * dm * np.cos(arg)).sum((1, 2))
        normal = np.stack((dZ, -dR), axis=-1) / np.hypot(dR, dZ)[:, None]
        centre = np.array([R.mean(), Z.mean()])
        normal *= np.sign(np.sum(normal * (np.stack((R, Z), -1) - centre))) or 1.0
        Rc, Zc = R + offset * normal[:, 0], Z + offset * normal[:, 1]
        for axis, values in enumerate((Rc * np.cos(phi), Rc * np.sin(phi), Zc)):
            dofs[i, axis, 0] = values.mean()
            for k in range(1, order + 1):
                dofs[i, axis, 2 * k - 1] = 2 * np.mean(values * np.sin(2 * np.pi * k * t))
                dofs[i, axis, 2 * k] = 2 * np.mean(values * np.cos(2 * np.pi * k * t))
    return Coils(Curves(jnp.asarray(dofs), n_segments=n_segments, nfp=inp.nfp, stellsym=stellsym),
                 np.full(n_coils, current))


def fit_vacuum_coils(inp, equilibrium, coils, coil_penalty, *, nphi, ntheta, normal_field_limit,
                     jump_limit=.04, maxiter=200, scale=.05, bound=3.):
    """Fit coil shapes and one common current to a frozen vacuum boundary, with no equilibrium solve.

    Minimizes the coil penalty plus ``(rms(B.n/|B|)/normal_field_limit)^2 / 2`` and
    ``(rms(|B_coil|^2/|B|^2 - 1)/jump_limit)^2 / 2`` on the equilibrium's boundary, where the
    coil field is scaled by the least-squares common current factor to the VMEX boundary
    field. The second term keeps the factor from hiding a field that only looks tangent.
    Returns the coils with fitted shapes and currents multiplied by that factor.
    """
    from scipy.optimize import minimize
    from essos.surfaces import surfacerzfourier_from_boundary
    data = vc.surface_field_data_from_state(
        inp, equilibrium.solution, runtime=equilibrium.solver_context, nphi=nphi, ntheta=ntheta)
    area = jnp.linalg.norm(data.area_vector, axis=0)
    weights, Bmag = area / jnp.sum(area), jnp.linalg.norm(data.B_total, axis=0)
    surface = surfacerzfourier_from_boundary(
        jnp.asarray(inp.rbc), jnp.asarray(inp.zbs), inp.nfp, nphi=nphi, ntheta=ntheta)
    shape0 = jnp.asarray(coils.dofs_curves).ravel()

    def coils_from(u):
        return coils.with_dofs(jnp.concatenate((shape0 + scale * u, coils.dofs_currents)))

    def field_factor(trial):
        field = vc.external_B_cartesian(coil_field(trial), data.gamma, data.phi)
        factor = common_current_scale(field, data.B_total, weights)
        normal = factor * jnp.sum(field * data.normal, axis=0) / Bmag
        jump = factor**2 * jnp.sum(field**2, axis=0) / Bmag**2 - 1
        return factor, jnp.sqrt(jnp.sum(weights * normal**2)), jnp.sqrt(jnp.sum(weights * jump**2))

    def objective(u):
        trial = coils_from(u)
        _, normal, jump = field_factor(trial)
        return (jnp.sum(coil_penalty(trial, surface))
                + .5 * (normal / normal_field_limit)**2 + .5 * (jump / jump_limit)**2)

    value_and_grad = jax.jit(jax.value_and_grad(objective))
    fit = minimize(lambda u: tuple(map(np.asarray, value_and_grad(jnp.asarray(u)))),
                   np.zeros(shape0.size), jac=True, method="L-BFGS-B",
                   bounds=[(-bound, bound)] * shape0.size,
                   options={"maxiter": maxiter, "maxcor": 20, "ftol": 1e-15, "gtol": 1e-10})
    fitted = coils_from(jnp.asarray(fit.x))
    factor, normal, jump = field_factor(fitted)
    if not np.isfinite(float(factor)) or float(factor) == 0:
        raise ValueError("The boundary field must admit a finite nonzero coil current")
    print(f"[coil fit] {fit.nit} L-BFGS-B iterations ({fit.message}), no equilibrium solves: B.n/B RMS = "
          f"{100 * float(normal):.3f}%, |B|^2 jump RMS = {100 * float(jump):.3f}%, "
          f"common current factor {float(factor):.6g}", flush=True)
    return fitted.with_dofs(jnp.concatenate((fitted.dofs_curves.ravel(), fitted.dofs_currents * factor)))


class SingleStageProblem:
    """Compose any fixed-flux ``VmecProblem`` with coils and field/shape limits.

    ``value_and_grad`` takes scaled increments around ``x0``. VC precision is
    planned once on the seed and held fixed during differentiation. A failed
    equilibrium receives a finite rejection cost above the normalized seed.
    """

    def __init__(self, problem, coils, coil_penalty, *, nphi, ntheta,
                 iota_floor, aspect_limit, normal_field_limit, constraint_weight=1e3,
                 jump_limit=None, vc_digits=4, boundary_scale=0.02, coil_scale=0.05, rejected_trial_cost=10):
        self.problem, self.coils, self.coil_penalty = problem, coils, coil_penalty
        self.nphi, self.ntheta, self.vc_digits = nphi, ntheta, vc_digits
        self.n_boundary = problem.x0.size
        self.x0 = np.concatenate((problem.x0, np.asarray(coils.dofs_curves).ravel()))
        self.scales = np.concatenate((boundary_scale * problem.scales,
                                      np.full(self.x0.size - self.n_boundary, coil_scale)))
        self.seed_equilibrium = problem.equilibrium_from_x(problem.x0)
        inp = problem.input_from_x(problem.x0)
        self.nfp = inp.nfp
        precision = None if _vacuum_input(inp) else vc.plan_vc_precision(vc.surface_field_data_from_state(
            inp, self.seed_equilibrium.solution, runtime=self.seed_equilibrium.solver_context,
            nphi=nphi, ntheta=ntheta), digits=vc_digits)
        self.monitor = opt.OptimizationMonitor(problem)
        self.counts, self._cached = {"trials": 0, "rejected": 0}, {}
        self.rejected_trial_cost = rejected_trial_cost

        self.constraint_names = ["min |iota|", "aspect", "B.n/B RMS"] + (
            [] if jump_limit is None else ["|B|^2 jump RMS"])
        self.multipliers = np.zeros(len(self.constraint_names))
        self.constraint_weight, self._penalty = constraint_weight, None

        def plasma_objective(u, multipliers):
            x = jnp.asarray(self.x0) + jnp.asarray(self.scales) * u
            _, trial_coils = self.objects_from_x(x)

            def limits(state, runtime):
                normal, weights, jump = total_normal_field(
                    inp, trial_coils, state, runtime, nphi, ntheta, precision, digits=vc_digits)
                rms = jnp.sqrt(jnp.sum(weights * normal**2))
                constraints = [opt.min_abs_iota(state, runtime) / iota_floor - 1,
                               1 - opt.aspect_ratio(state, runtime) / aspect_limit,
                               1 - rms / normal_field_limit]
                if jump_limit is not None:
                    constraints.append(1 - jnp.sqrt(jnp.sum(weights * jump**2)) / jump_limit)
                slack = jnp.maximum(multipliers / constraint_weight - jnp.stack(constraints), 0)
                return .5 * constraint_weight * slack**2 - multipliers**2 / (2 * constraint_weight)

            return problem.jax_objective_from_state(
                x[:self.n_boundary], limits, n_extra_terms=len(self.constraint_names),
                return_status=True)

        def coil_objective(u):
            surface, trial_coils = self.objects_from_x(jnp.asarray(self.x0) + self.scales * u)
            costs = coil_penalty(trial_coils, surface)
            return jnp.sum(costs), costs

        self._plasma_vg = jax.jit(jax.value_and_grad(plasma_objective, has_aux=True))
        self._coil_vg = jax.jit(jax.value_and_grad(coil_objective, has_aux=True))

    def coils_from_dofs(self, dofs):
        """Update coil shapes while preserving currents."""
        return self.coils.with_dofs(jnp.concatenate((dofs, self.coils.dofs_currents)))

    def objects_from_x(self, x):
        """Return the surface and coils for an unscaled combined vector."""
        from essos.surfaces import surfacerzfourier_from_boundary
        rbc, zbs = self.problem.boundary_from_x(x[:self.n_boundary])
        surface = surfacerzfourier_from_boundary(
            rbc, zbs, self.nfp, nphi=self.nphi, ntheta=self.ntheta)
        return surface, self.coils_from_dofs(x[self.n_boundary:])

    def value_and_grad(self, u):
        """Return the normalized joint cost and gradient, caching repeated trials."""
        u = np.asarray(u, dtype=float)
        cache = self._cached
        if cache.get("key") == u.tobytes():
            return cache["value"], cache["gradient"].copy()
        self.counts["trials"] += 1
        (plasma_value, (rows, penalty, status)), plasma_gradient = self._plasma_vg(
            jnp.asarray(u), jnp.asarray(self.multipliers))
        rows = np.asarray(rows)
        if int(np.asarray(status)) != 0:
            self.counts["rejected"] += 1
            self._penalty = None
            distance, cost = float(np.linalg.norm(u)), self.rejected_trial_cost
            value = cost * (1 + distance)**2
            gradient = 2 * cost * (1 + distance) * u / max(distance, 1e-300)
            print(f"  trial {self.counts['trials']}: no certified equilibrium, rejected", flush=True)
        else:
            (coil_value, costs), coil_gradient = self._coil_vg(jnp.asarray(u))
            total = float(plasma_value + coil_value)
            cache.setdefault("scale", 1 / total if total > 0 else 1.0)
            scale = cache["scale"]
            terms = {name: 0.5 * float(rows[start:stop] @ rows[start:stop])
                     for name, start, stop in self.problem.metadata["term_slices"]}
            self._penalty = np.asarray(penalty)
            terms["iota/aspect/B.n penalty"] = float(np.sum(self._penalty))
            terms.update(zip(("coil length", "coil curvature", "coil separation",
                              "coil-surface separation"), map(float, np.asarray(costs))))
            value, gradient = self.monitor.cache_evaluation(
                u, scale * (plasma_value + coil_value), scale * (plasma_gradient + coil_gradient),
                {name: scale * term for name, term in terms.items()})
        cache.update(key=u.tobytes(), value=value, gradient=gradient)
        return value, gradient.copy()

    def update_multipliers(self, u):
        """Powell-Hestenes-Rockafellar step for the iota, aspect and field limits at ``u``.

        Returns ``{name: c}`` with ``c >= 0`` meaning satisfied; a limit that is inactive
        with no multiplier reports None. The step needs a certified equilibrium at ``u``.
        """
        self._cached.pop("key", None)
        self.value_and_grad(u)
        if self._penalty is None:
            raise ValueError("The multiplier update needs a certified equilibrium")
        rho = self.constraint_weight
        slack = np.sqrt(np.maximum(2 * self._penalty / rho + (self.multipliers / rho)**2, 0))
        values = {name: (None if gap == 0 else lam / rho - gap)
                  for name, gap, lam in zip(self.constraint_names, slack, self.multipliers)}
        self.multipliers = rho * slack
        self._cached.pop("key", None)
        return values


def _clearance(gamma, points):
    return min(float(cKDTree(points).query(c)[0].min()) for c in gamma)


def verify_design(joint, u, *, prefix, ns=101, ftol=1e-12, free_ns=(16, 51), free_ftol=1e-12,
                  free_mpol=None, geometry_segments=384, verbose=True):
    """Save the design, re-solve finely, then cold-solve the held free boundary.

    Returns ``(equilibrium, surface, metrics, free_check)``. Free-boundary
    metrics are provisional unless the final grid converged with vacuum data.
    Flux, pressure, and current profiles are unchanged in both checks. ``free_mpol`` raises the
    poloidal and toroidal mode numbers of the free-boundary check only.
    Coil engineering metrics use at least 384 points per Fourier curve.
    """
    from essos.fields import BiotSavart
    from essos.surfaces import SurfaceRZFourier, surfacerzfourier_from_boundary
    jax.clear_caches()
    gc.collect()
    x = joint.x0 + joint.scales * u
    _, coils = joint.objects_from_x(jnp.asarray(x))
    equilibrium = joint.problem.equilibrium_from_x(x[:joint.n_boundary])
    inp = replace(joint.problem.input_from_x(x[:joint.n_boundary]), ns_array=np.array([ns]),
                  ftol_array=np.array([ftol]), niter_array=np.array([8000]))
    inp.to_indata(f"input.{prefix}_optimized")
    coils.to_json(f"coils_{prefix}_optimized.json")
    print("[verify] finer fixed-boundary equilibrium and virtual casing", flush=True)
    equilibrium = opt.solve_equilibrium(inp, initial_state=equilibrium.solution, verbose=verbose)
    surface = surfacerzfourier_from_boundary(jnp.asarray(inp.rbc), jnp.asarray(inp.zbs),
                                            inp.nfp, nphi=61, ntheta=64)
    data = vc.surface_field_data_from_state(inp, equilibrium.solution,
                                           runtime=equilibrium.solver_context, nphi=61, ntheta=64)
    normal, area, jump = map(np.asarray, total_normal_field(
        inp, coils, equilibrium.solution, equilibrium.solver_context, 61, 64,
        None if _vacuum_input(inp) else vc.plan_vc_precision(data, digits=joint.vc_digits),
        digits=joint.vc_digits))
    geometry = _resample_coils(coils, max(384, geometry_segments))
    gamma, points = np.asarray(geometry.gamma), np.asarray(surface.gamma).reshape(-1, 3)
    metrics = {"B.n/B RMS": float(np.sqrt(np.sum(area * normal**2))),
               "B.n/B max": float(np.max(np.abs(normal))),
               "pressure jump RMS": float(np.sqrt(np.sum(area * jump**2))),
               "coil-surface distance": _clearance(gamma, points),
               "coil-coil distance": min(float(cKDTree(gamma[i]).query(gamma[j])[0].min())
                                         for i in range(len(gamma)) for j in range(i + 1, len(gamma))),
               "maximum curvature": float(np.max(np.asarray(geometry.curvature))),
               "min |iota|": float(opt.min_abs_iota(equilibrium.state, equilibrium.runtime)),
               "aspect": float(opt.aspect_ratio(equilibrium.state, equilibrium.runtime)),
               "beta": float(opt.volume_average_beta(equilibrium.state, equilibrium.runtime)),
               "verification solve converged": bool(np.all(np.asarray(equilibrium.result.converged)))}
    jax.clear_caches()
    gc.collect()
    print("[verify] cold NESTOR free boundary", flush=True)
    # NESTOR needs a finer angular grid and a smaller step than the fixed-boundary solve.
    free_inp = replace(inp, lfreeb=True, mgrid_file="direct ESSOS field", ns_array=np.array(free_ns),
                       ntheta=max(inp.ntheta, 32), nzeta=max(inp.nzeta, 32), delt=min(inp.delt, .3),
                       ftol_array=np.full(len(free_ns), free_ftol), niter_array=np.full(len(free_ns), 8000))
    if free_mpol is not None:  # The coil field excites boundary harmonics the optimized deck truncates.
        free_inp = free_inp.change_resolution(
            mpol=free_mpol, ntor=free_mpol, ntheta=max(2 * free_mpol + 6, free_inp.ntheta),
            nzeta=max(2 * free_mpol + 4, free_inp.nzeta))
    free = {"converged": False, "phiedge": float(inp.phiedge),
            "ns": list(free_ns), "metrics_provisional": True}
    started = time.perf_counter()
    try:
        result = vj.solve_free_boundary_multigrid(
            free_inp, external_field=BiotSavart(coils), verbose=verbose, raise_on_max_iterations=False,
            release_stage_cache=True)
        final_ns = int(result.state.R_cos.shape[0])
        converged = (bool(np.all(np.asarray(result.converged))) and result.vacuum is not None
                     and final_ns == free_ns[-1])
        wout = vj.wout_from_state(inp=free_inp, state=result.state, fsqr=result.fsqr,
                                  fsqz=result.fsqz, fsql=result.fsql, niter=result.iterations,
                                  converged=result.converged, vacuum_output=result.vacuum)
        path = vj.write_wout(f"wout_{prefix}_free_check.nc", wout)
        held = np.asarray(SurfaceRZFourier.from_wout_file(path, nphi=61, ntheta=64, close=False).gamma).reshape(-1, 3)
        free.update(converged=converged, metrics_provisional=not converged, final_ns=final_ns,
                    iterations=int(result.iterations),
                    force_errors={k: float(getattr(result, k)) for k in ("fsqr", "fsqz", "fsql")},
                    delbsq=float(result.vacuum.delbsq) if result.vacuum is not None else None,
                    aspect=float(wout.aspect), beta=float(wout.betatotal),
                    min_abs_iota=float(np.min(np.abs(wout.iotas[1:]))),
                    sampled_boundary_distance_m=float(max(cKDTree(points).query(held)[0].max(),
                                                          cKDTree(held).query(points)[0].max())),
                    coil_surface_distance_m=_clearance(gamma, held))
    except VmecError as exc:
        free["error"] = str(exc)
    free["seconds"] = time.perf_counter() - started
    print("[free boundary] " + json.dumps(free))
    return equilibrium, surface, metrics, free


def save_outputs(joint, u, equilibrium, surface, *, prefix, make_movie=False, surface_color="absB"):
    """Write equilibrium, initial/final geometry, objective history, and plots."""
    from essos.fields import BiotSavart
    from essos.surfaces import surfacerzfourier_from_boundary
    _, coils = joint.objects_from_x(jnp.asarray(joint.x0 + joint.scales * u))
    rbc, zbs = joint.problem.boundary_from_x(joint.problem.x0)
    initial = surfacerzfourier_from_boundary(rbc, zbs, equilibrium.wout.nfp, nphi=60, ntheta=60)
    for label, surf, cs in (("initial", initial, joint.coils), ("optimized", surface, coils)):
        surf.to_vtk(f"surface_{prefix}_{label}", field=BiotSavart(cs))
        cs.to_vtk(f"coils_{prefix}_{label}")
    path = vj.write_wout(f"wout_{prefix}_optimized.nc", equilibrium.wout)
    vj.plot_optimization_objects(f"{prefix}_optimization.png", ("Initial", initial, joint.coils),
                                 ("Optimized", surface, coils))
    joint.monitor.save(f"{prefix}_objectives.csv")
    joint.monitor.plot(f"{prefix}_objectives.png", title="Single-stage objective terms")
    if make_movie:
        joint.monitor.movie_surface_coils(
            f"{prefix}_optimization.gif", joint.objects_from_x, x0=joint.x0, scales=joint.scales,
            surface_color=surface_color, plasma_problem=joint.problem,
            external_field=lambda objects: coil_field(objects[1]),
            nphi=joint.nphi, ntheta=joint.ntheta, cmap="jet")
    for output in vj.plot_wout(path, ".").values():
        print(f"Wrote {output}")


def design_target_report(metrics, free, *, iota_floor, aspect_limit, normal_field_limit,
                         clearance, coil_distance, curvature, delbsq_limit, boundary_distance_limit,
                         beta_bounds=None):
    """Print one verdict and return JSON-ready targets, misses, and success.

    ``beta_bounds`` is ``(low, high)`` for finite beta and None for a vacuum design.
    Nonfinite metrics fail their bounds. A failed independent solve cannot
    pass, and provisional free-boundary metrics are never treated as checks.
    """
    limits = [("min |iota|", iota_floor, "min"), ("aspect", aspect_limit, "max"),
              ("B.n/B RMS", normal_field_limit, "max"),
              ("coil-surface distance", clearance, "min"),
              ("coil-coil distance", coil_distance, "min"),
              ("maximum curvature", curvature, "max")]
    held = [("DEL-BSQ", "delbsq", delbsq_limit, "max"),
            ("boundary distance", "sampled_boundary_distance_m", boundary_distance_limit, "max"),
            ("min |iota|", "min_abs_iota", iota_floor, "min"),
            ("aspect", "aspect", aspect_limit, "max"),
            ("coil-surface distance", "coil_surface_distance_m", clearance, "min")]
    if beta_bounds is not None:
        limits += [("beta", beta_bounds[0], "min"), ("beta", beta_bounds[1], "max")]
        held += [("beta", "beta", beta_bounds[0], "min"), ("beta", "beta", beta_bounds[1], "max")]
    rows = [(name, metrics[name], bound, kind) for name, bound, kind in limits]
    if free["converged"]:
        rows += [(f"held {name}", free[key], bound, kind) for name, key, bound, kind in held]
    unmet = [f"{name} {value:.4g} {'below' if kind == 'min' else 'above'} {bound:.4g}"
             for name, value, bound, kind in rows
             if not np.isfinite(value) or not (value >= bound if kind == "min" else value <= bound)]
    if not metrics["verification solve converged"]:
        unmet.append("the finer fixed-boundary verification did not converge")
    if not free["converged"]:
        unmet.append("the post-optimization free-boundary verification did not converge")
    targets = {f"{name} {'>=' if kind == 'min' else '<='}": bound
               for name, bound, kind in limits + [(f"held {n}", b, k) for n, _, b, k in held]}
    print("This run did NOT meet its stated targets: " + "; ".join(unmet)
          if unmet else "All stated targets met.")
    return {"targets": targets, "unmet": unmet, "met": not unmet}
