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

    def __call__(self, coils, surface):
        """Return length, curvature, separation, and clearance costs."""
        from essos.objective_functions import loss_coil_separation, loss_coil_surface_distance
        n = coils.dofs_currents.size
        length = jnp.sqrt(self.length_weight) * jnp.maximum(coils.length[:n] - self.length_target, 0)
        curvature = jnp.sqrt(self.curvature_weight) * jnp.maximum(
            coils.curvature[:n] - self.curvature_limit, 0)
        return jnp.asarray([
            0.5 * jnp.vdot(length, length), 0.5 * jnp.vdot(curvature, curvature),
            0.5 * self.distance_weight * loss_coil_separation(coils, self.coil_distance, block_size=32),
            0.5 * self.surface_weight * loss_coil_surface_distance(
                coils, surface, self.surface_distance, block_size=32)])


def total_normal_field(inp, coils, state, runtime, nphi, ntheta, precision, *, digits=4):
    """Total-field normal residual divided by equilibrium ``|B|``, and area weights."""
    data = vc.surface_field_data_from_state(inp, state, runtime=runtime, nphi=nphi, ntheta=ntheta)
    interface = vc.PlasmaVacuumInterface.from_surface_data(data, digits=digits, precision=precision)
    return interface.bnormal_residual(coil_field(coils)) / jnp.linalg.norm(data.B_total, axis=0), interface.weights


class SingleStageProblem:
    """Compose any fixed-flux ``VmecProblem`` with coils and field/shape limits.

    ``value_and_grad`` takes scaled increments around ``x0``. VC precision is
    planned once on the seed and held fixed during differentiation. A failed
    equilibrium receives a finite rejection cost above the normalized seed.
    """

    def __init__(self, problem, coils, coil_penalty, *, nphi, ntheta,
                 iota_floor, aspect_limit, normal_field_limit, constraint_weight=1e3,
                 vc_digits=4, boundary_scale=0.02, coil_scale=0.05, rejected_trial_cost=10):
        self.problem, self.coils, self.coil_penalty = problem, coils, coil_penalty
        self.nphi, self.ntheta, self.vc_digits = nphi, ntheta, vc_digits
        self.n_boundary = problem.x0.size
        self.x0 = np.concatenate((problem.x0, np.asarray(coils.dofs_curves).ravel()))
        self.scales = np.concatenate((boundary_scale * problem.scales,
                                      np.full(self.x0.size - self.n_boundary, coil_scale)))
        self.seed_equilibrium = problem.equilibrium_from_x(problem.x0)
        inp = problem.input_from_x(problem.x0)
        self.nfp = inp.nfp
        precision = vc.plan_vc_precision(vc.surface_field_data_from_state(
            inp, self.seed_equilibrium.solution, runtime=self.seed_equilibrium.solver_context,
            nphi=nphi, ntheta=ntheta), digits=vc_digits)
        self.monitor = opt.OptimizationMonitor(problem)
        self.counts, self._cached = {"trials": 0, "rejected": 0}, {}
        self.rejected_trial_cost = rejected_trial_cost

        def plasma_objective(u):
            x = jnp.asarray(self.x0) + jnp.asarray(self.scales) * u
            _, trial_coils = self.objects_from_x(x)

            def limits(state, runtime):
                normal, weights = total_normal_field(
                    inp, trial_coils, state, runtime, nphi, ntheta, precision, digits=vc_digits)
                rms = jnp.sqrt(jnp.sum(weights * normal**2))
                return hinge(jnp.stack((opt.min_abs_iota(state, runtime) / iota_floor - 1,
                                       1 - opt.aspect_ratio(state, runtime) / aspect_limit,
                                       1 - rms / normal_field_limit)), constraint_weight)

            return problem.jax_objective_from_state(
                x[:self.n_boundary], limits, n_extra_terms=1, return_status=True)

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
        (plasma_value, (rows, penalty, status)), plasma_gradient = self._plasma_vg(jnp.asarray(u))
        rows = np.asarray(rows)
        if int(np.asarray(status)) != 0:
            self.counts["rejected"] += 1
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
            terms["iota/aspect/B.n penalty"] = float(np.asarray(penalty)[0])
            terms.update(zip(("coil length", "coil curvature", "coil separation",
                              "coil-surface separation"), map(float, np.asarray(costs))))
            value, gradient = self.monitor.cache_evaluation(
                u, scale * (plasma_value + coil_value), scale * (plasma_gradient + coil_gradient),
                {name: scale * term for name, term in terms.items()})
        cache.update(key=u.tobytes(), value=value, gradient=gradient)
        return value, gradient.copy()


def _clearance(gamma, points):
    return min(float(cKDTree(points).query(c)[0].min()) for c in gamma)


def verify_design(joint, u, *, prefix, ns=101, ftol=1e-12,
                  free_ns=(16, 51), free_ftol=1e-12, verbose=True):
    """Save the design, re-solve finely, then cold-solve the held free boundary.

    Returns ``(equilibrium, surface, metrics, free_check)``. Free-boundary
    metrics are provisional unless the final grid converged with vacuum data.
    Flux, pressure, and current profiles are unchanged in both checks.
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
    normal, area = map(np.asarray, total_normal_field(
        inp, coils, equilibrium.solution, equilibrium.solver_context, 61, 64,
        vc.plan_vc_precision(data, digits=joint.vc_digits), digits=joint.vc_digits))
    gamma, points = np.asarray(coils.gamma), np.asarray(surface.gamma).reshape(-1, 3)
    metrics = {"B.n/B RMS": float(np.sqrt(np.sum(area * normal**2))),
               "B.n/B max": float(np.max(np.abs(normal))),
               "coil-surface distance": _clearance(gamma, points),
               "coil-coil distance": min(float(cKDTree(gamma[i]).query(gamma[j])[0].min())
                                         for i in range(len(gamma)) for j in range(i + 1, len(gamma))),
               "maximum curvature": float(np.max(np.asarray(coils.curvature))),
               "min |iota|": float(opt.min_abs_iota(equilibrium.state, equilibrium.runtime)),
               "aspect": float(opt.aspect_ratio(equilibrium.state, equilibrium.runtime)),
               "beta": float(opt.volume_average_beta(equilibrium.state, equilibrium.runtime)),
               "verification solve converged": bool(np.all(np.asarray(equilibrium.result.converged)))}
    jax.clear_caches()
    gc.collect()
    print("[verify] cold NESTOR free boundary", flush=True)
    free_inp = replace(inp, lfreeb=True, mgrid_file="direct ESSOS field", ns_array=np.array(free_ns),
                       ftol_array=np.full(len(free_ns), free_ftol), niter_array=np.full(len(free_ns), 8000))
    free = {"converged": False, "phiedge": float(inp.phiedge),
            "ns": list(free_ns), "metrics_provisional": True}
    started = time.perf_counter()
    try:
        result = vj.solve_free_boundary_multigrid(
            free_inp, external_field=BiotSavart(coils), verbose=verbose, raise_on_max_iterations=False)
        converged = bool(np.all(np.asarray(result.converged))) and result.vacuum is not None
        wout = vj.wout_from_state(inp=free_inp, state=result.state, fsqr=result.fsqr,
                                  fsqz=result.fsqz, fsql=result.fsql, niter=result.iterations,
                                  converged=result.converged, vacuum_output=result.vacuum)
        path = vj.write_wout(f"wout_{prefix}_free_check.nc", wout)
        held = np.asarray(SurfaceRZFourier.from_wout_file(path, nphi=61, ntheta=64, close=False).gamma).reshape(-1, 3)
        free.update(converged=converged, metrics_provisional=not converged,
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
                         clearance, coil_distance, curvature, beta_bounds,
                         delbsq_limit, boundary_distance_limit):
    """Print one verdict and return JSON-ready targets, misses, and success.

    Nonfinite metrics fail their bounds. A failed independent solve cannot
    pass, and provisional free-boundary metrics are never treated as checks.
    """
    limits = [("min |iota|", iota_floor, "min"), ("aspect", aspect_limit, "max"),
              ("B.n/B RMS", normal_field_limit, "max"),
              ("coil-surface distance", clearance, "min"),
              ("coil-coil distance", coil_distance, "min"),
              ("maximum curvature", curvature, "max"),
              ("beta", beta_bounds[0], "min"), ("beta", beta_bounds[1], "max")]
    held = [("DEL-BSQ", "delbsq", delbsq_limit, "max"),
            ("boundary distance", "sampled_boundary_distance_m", boundary_distance_limit, "max"),
            ("min |iota|", "min_abs_iota", iota_floor, "min"),
            ("aspect", "aspect", aspect_limit, "max"),
            ("coil-surface distance", "coil_surface_distance_m", clearance, "min"),
            ("beta", "beta", beta_bounds[0], "min"), ("beta", "beta", beta_bounds[1], "max")]
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
