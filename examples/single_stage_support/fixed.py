"""Fixed-boundary setup and output shared by both benchmark folders.

All equilibrium ownership uses VMEX's public accepted-state API. This module
does not run qualification or optimize when imported or when building a case.
"""
from dataclasses import replace
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from . import common

def build_problem(args, *, settings, coil_limits=None):
    import numpy as np
    from scipy.optimize import OptimizeResult, minimize

    import vmex as vj
    from vmex import optimize as opt

    import jax
    import jax.numpy as jnp
    from .diagnostics import StepHistory
    from .constraints import PhysicalConstraints
    limits = PhysicalConstraints.from_settings(settings)

    if jax.default_backend() != args.device or not jax.config.x64_enabled:
        raise RuntimeError("requested device and float64 precision are required")

    try:
        from essos.fields import BiotSavart
        from essos.objective_functions import loss_coil_separation, loss_coil_surface_distance
        from essos.surfaces import surfacerzfourier_from_boundary
    except ImportError as error:
        raise ImportError(
            "This example needs the optional coil dependencies: install vmex[coils]."
        ) from error

    OPTIONS = {"maxiter": args.accepted_steps, "gtol": settings.OPTIMIZER_GTOL}
    if settings.METHOD == "L-BFGS-B":
        OPTIONS.update(maxls=20, ftol=settings.OPTIMIZER_FTOL, maxcor=20)
    if args.constrained:
        OPTIONS = {"maxiter": args.accepted_steps, "ftol": settings.OPTIMIZER_FTOL}
    ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
    # Local copies allow bounded CI runs without changing the editable defaults.
    N_SEGMENTS, COIL_ORDER, NPHI, NTHETA = (getattr(settings, key) for key in ("N_SEGMENTS", "COIL_ORDER", "NPHI", "NTHETA"))
    fit_maxiter = args.coil_fit_maxiter
    if ci_smoke:
        N_SEGMENTS, COIL_ORDER, NPHI, NTHETA, fit_maxiter = 24, 2, 8, 8, 2
        OPTIONS["maxiter"] = 1
    DATA = args.input
    inp, seed = common.load_input(args)

    # Floor the profile minimum, not its average: a mean target is satisfiable while
    # an interior surface sits near zero transform, which is what a current-carried
    # finite-beta profile does. opt.mean_iota targets the average instead, and
    # opt.soft_min_abs_iota is the smooth-minimum variant.
    def iota_floor(equilibrium_state, solver_context):
        return jnp.maximum(
            settings.IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)


    qs = opt.QuasisymmetryRatioResidual(np.asarray(settings.QA_SURFACES), helicity_m=1, helicity_n=0)
    plasma_terms = [
        (qs.residuals_state, 0.0, 1.0),
        (opt.aspect_ratio, settings.ASPECT_TARGET, settings.ASPECT_WEIGHT),
        (iota_floor, 0.0, settings.IOTA_WEIGHT),
    ]
    def plasma_loss(state, runtime):
        rows = opt.residuals_from_tuples(state, runtime, plasma_terms)
        return 0.5 * jnp.vdot(rows, rows)


    plasma_problem = opt.VmecProblem.from_loss(
        inp, plasma_loss, max_mode=settings.MAX_MODE, vary_major_radius=settings.VARY_MAJOR_RADIUS or args.constrained,
        use_ess=True, ess_alpha=settings.ESS_ALPHA, progress=not ci_smoke, device=args.device, restart_from=seed, warm_start="state")

    # Both examples generate identical circles, or load the same ESSOS JSON.
    if coil_limits is None:
        coils0 = common.initial_coils(inp, args.coils or args.initial_coils,
            parameters=dict(vars(settings), N_SEGMENTS=N_SEGMENTS, COIL_ORDER=COIL_ORDER))
    else:
        from essos.coils import Coils
        from .common import resize_coils
        coils0 = resize_coils(Coils.from_json(str(args.coils or args.initial_coils or
            common.DATA / "coils.initial.scalar.json")), COIL_ORDER, N_SEGMENTS)
        if coils0.nfp != inp.nfp or not coils0.stellsym or coils0.dofs_curves.shape != (settings.N_COILS, 3, 2*COIL_ORDER+1):
            raise ValueError("coil count, order or symmetry differs from the case")
    curves0 = coils0.curves

    def normalized_normal_field(coils, surface):
        field = BiotSavart(coils)
        magnetic_field = jax.vmap(field.B)(surface.gamma.reshape(-1, 3)).reshape(surface.gamma.shape)
        return jnp.sum(magnetic_field * surface.unitnormal, axis=2) / jnp.linalg.norm(magnetic_field, axis=2)

    def coil_field(coils):
        field = BiotSavart(coils)
        return lambda points: jax.vmap(field.B)(points.reshape(-1, 3)).reshape(points.shape)

    def normal_field_residual(coils, surface):
        """Area-weighted rows of B.n/|B| over the target boundary.

        A flux surface requires B.n = 0 on it, so these rows are the quadrature
        of the surface-averaged square error the coils leave behind, normalized
        by |B| to make it dimensionless and by the area element so refining the
        grid does not change the objective. Driving them to zero is what makes
        the coil set reproduce the plasma boundary.
        """
        weights = surface.area_element / jnp.sum(surface.area_element)
        values = normalized_normal_field(coils, surface)
        return (jnp.sqrt(weights) * values).ravel()

    def normal_field_excess(coils, surface):
        """Hinge on the largest local B.n/|B|, zero until the limit is exceeded.

        The residual above is an average, and an average tolerates one bad patch
        by trading it against a well-matched remainder; an island forms where the
        error is locally worst, not where it is worst on average. This term reads
        the maximum instead (through a logsumexp so it stays differentiable) and
        contributes nothing until that maximum passes
        NORMAL_FIELD_OBJECTIVE_LIMIT, which keeps it from competing with the
        average term over shapes that already satisfy the bound.
        """
        values = jnp.sqrt(normalized_normal_field(coils, surface)**2 + 1.0e-12)
        smooth_maximum = jax.scipy.special.logsumexp(2000.0 * values) / 2000.0
        return jnp.maximum(smooth_maximum - settings.NORMAL_FIELD_OBJECTIVE_LIMIT, 0.0)

    def coil_lengths(coils, _surface):
        return coils.length[:settings.N_COILS]

    def coil_curvature_excess(coils, _surface):
        return jnp.maximum(coils.curvature[:settings.N_COILS] - settings.CURVATURE_OBJECTIVE_LIMIT, 0.0)

    coil_terms = [
        (normal_field_residual, 0.0, settings.NORMAL_FIELD_WEIGHT),
        (normal_field_excess, 0.0, settings.NORMAL_FIELD_LIMIT_WEIGHT),
    ]
    if coil_limits is None:
        coil_terms.extend([(coil_lengths, settings.LENGTH_TARGET, settings.LENGTH_WEIGHT),
                           (coil_curvature_excess, 0.0, settings.CURVATURE_WEIGHT)])
    coil_term_names = tuple(function.__name__ for function, _target, _weight in coil_terms) + (
        "coil separation", "coil-surface separation")
    if coil_limits is not None:
        coil_term_names = coil_term_names[:-2]

    # The public VMEX problem owns the boundary-mode convention and RBC(0,0) choice.
    x_boundary0 = plasma_problem.x0
    rbc0, zbs0 = plasma_problem.boundary_from_x(x_boundary0)

    n_curve_dofs = curves0.dofs.size
    x_coils0 = np.asarray(curves0.dofs).ravel()
    x0 = np.concatenate([x_boundary0, x_coils0])
    dof_names = plasma_problem.dof_names + curves0.dof_names

    # SciPy works in dimensionless increments u, with x = x0 + scales*u.
    scales = np.concatenate([
        settings.PARAMETER_STEP * plasma_problem.scales, np.full(n_curve_dofs, settings.COIL_STEP)])


    def objects_from_x(x):
        x_boundary, x_coils = x[:x_boundary0.size], x[x_boundary0.size:]
        rbc, zbs = plasma_problem.boundary_from_x(x_boundary)
        surface = surfacerzfourier_from_boundary(
            rbc, zbs, inp.nfp, nphi=NPHI, ntheta=NTHETA)

        coils = coils0.with_dofs(jnp.concatenate((x_coils, coils0.dofs_currents)))
        return surface, coils


    def coil_costs(x):
        surface, coils = objects_from_x(x)
        rows = [jnp.sqrt(weight) * (jnp.atleast_1d(function(coils, surface)) - target).ravel()
                for function, target, weight in coil_terms]
        if coil_limits is not None:
            return jnp.stack([0.5 * jnp.vdot(row, row) for row in rows])
        return jnp.concatenate([jnp.stack([0.5 * jnp.vdot(row, row) for row in rows]), jnp.asarray([
            0.5 * settings.COIL_DISTANCE_WEIGHT * loss_coil_separation(
                coils, settings.COIL_DISTANCE_LIMIT, block_size=32),
            0.5 * settings.COIL_SURFACE_DISTANCE_WEIGHT * loss_coil_surface_distance(
                coils, surface, settings.COIL_SURFACE_DISTANCE_LIMIT, block_size=32),
        ])])

    def coil_constraint_values(u):
        x = jnp.asarray(x0) + jnp.asarray(scales)*u
        _surface, coils = objects_from_x(x)
        rbc, zbs = plasma_problem.boundary_from_x(x[:x_boundary0.size])
        surf = surfacerzfourier_from_boundary(rbc, zbs, inp.nfp,
            nphi=coil_limits.SURFACE_GRID[0], ntheta=coil_limits.SURFACE_GRID[1])
        clearance = coil_limits.surface_distance(coils, surf)
        return jnp.r_[coil_limits.coil_inequalities(coils),
            (clearance-settings.COIL_SURFACE_DISTANCE_LIMIT-settings.DISTANCE_MARGIN)/settings.COIL_SURFACE_DISTANCE_LIMIT]

    coil_constraint_values = jax.jit(coil_constraint_values) if coil_limits else None
    coil_constraint_jacobian = jax.jit(jax.jacrev(coil_constraint_values)) if coil_limits else None
    constraint_values = constraint_jacobian = None
    plasma_problem = plasma_problem.with_accepted_state()

    if args.constrained:
        constraint_problem = opt.VmecProblem.from_tuples(inp,
            [(opt.min_abs_iota, 0.0, 1.0), (opt.major_radius, 0.0, 1.0)],
            max_mode=settings.MAX_MODE, vary_major_radius=True, use_ess=True, ess_alpha=settings.ESS_ALPHA,
            implicit_jacobian_method="reverse_adjoint", device=args.device, restart_from=seed, warm_start="state")

        def constraint_values(u):
            constraint_problem.restart_from(plasma_problem.accepted.equilibrium)
            x = x0 + scales*np.asarray(u)
            values = constraint_problem.residual(x[:x_boundary0.size])
            return np.asarray(values)

        def constraint_jacobian(u):
            constraint_values(u)
            x = x0 + scales*np.asarray(u)
            jac = constraint_problem.residual_jac(x[:x_boundary0.size])
            return np.pad(jac*scales[:x_boundary0.size], ((0,0),(0,n_curve_dofs)))


    def plasma_component(u):
        # The public stateful view seeds every trial from the accepted equilibrium.
        x = jnp.asarray(x0) + jnp.asarray(scales) * u
        x_boundary = x[:x_boundary0.size]
        value, gradient = plasma_problem.value_and_grad(np.asarray(x_boundary))
        # Keep monitoring separate from differentiation of the scalar loss.
        plasma_costs = {"plasma total": float(value)}
        scaled_gradient = gradient * jnp.asarray(scales[:x_boundary0.size])
        return (value, plasma_costs), jnp.pad(scaled_gradient, (0, x0.size - x_boundary0.size))

    def coil_objective(u):
        costs = coil_costs(jnp.asarray(x0) + jnp.asarray(scales) * u)
        return jnp.sum(costs), costs


    monitor = opt.OptimizationMonitor()
    # The host API selects certified Jacobian/fallback work without tracing both branches.
    plasma_value_and_grad = plasma_component
    coil_value_and_grad = jax.jit(jax.value_and_grad(coil_objective, has_aux=True))

    # VMEX supplies the exact equilibrium derivative; JAX differentiates the coil
    # objective. Their values and gradients add directly for any SciPy optimizer.
    def value_and_grad(u):
        (plasma_value, plasma_costs), plasma_gradient = plasma_value_and_grad(u)
        (coil_value, coil_cost_values), coil_gradient = coil_value_and_grad(u)
        terms = dict(plasma_costs)
        terms.update(zip(coil_term_names, map(float, np.asarray(coil_cost_values))))
        return monitor.cache_evaluation(
            u, plasma_value + coil_value, plasma_gradient + coil_gradient, terms)


    print("Running single_stage_optimization_scalar.py (same weighted objective; scalar adjoint)")
    print(f"Fixed-boundary VMEX + ESSOS: {x_boundary0.size} boundary and "
          f"{x_coils0.size} coil variables, one scalar plasma adjoint and JAX coil gradients")
    print(f"dof_names = {dof_names}")

    # Stage two: optimize only coil coordinates on the frozen seed boundary.
    # Currents stay fixed in objects_from_x, as in the free-boundary fit.
    # This fit calls no plasma objective, equilibrium solver, or plasma adjoint.
    print(f"\n[stage-two coil fit] Up to {fit_maxiter} L-BFGS-B iterations "
          "on the frozen seed boundary...", flush=True)
    coil_fit_monitor = opt.OptimizationMonitor(print_every=10, trace=False)

    def coil_fit_value_and_grad(u):
        full_u = jnp.pad(jnp.asarray(u), (x_boundary0.size, 0))
        (value, _costs), gradient = coil_value_and_grad(full_u)
        return coil_fit_monitor.cache_evaluation(u, value, gradient[x_boundary0.size:])

    initial_fit_cost = float(coil_fit_value_and_grad(np.zeros(n_curve_dofs))[0])
    if args.coils:
        coil_fit = OptimizeResult(x=np.zeros_like(x0), fun=initial_fit_cost, nit=0,
                                 success=True, message="reused saved fit")
    else:
        coil_fit = minimize(
            coil_fit_value_and_grad, np.zeros(n_curve_dofs), jac=True, method="L-BFGS-B",
            bounds=[(-settings.PARAMETER_BOUND, settings.PARAMETER_BOUND)] * n_curve_dofs,
            callback=coil_fit_monitor,
            options={"maxiter": fit_maxiter, "maxcor": 20, "ftol": 1.0e-15, "gtol": 1.0e-10})
        if (not np.isfinite(coil_fit.fun) or not np.all(np.isfinite(coil_fit.x))
                or coil_fit.fun > initial_fit_cost + 1e-10):
            raise RuntimeError("Stage-two coil fit returned invalid or worse coils")
        coil_fit.x = np.pad(coil_fit.x, (x_boundary0.size, 0))
    surface_seed, coils_fit = objects_from_x(jnp.asarray(x0 + scales * coil_fit.x))
    fit_rms = float(jnp.linalg.norm(normal_field_residual(coils_fit, surface_seed)))
    coil_fit_path = Path("coils.stage2.json")
    coils_fit.to_json(str(coil_fit_path))
    Path("stage_two.json").write_text(json.dumps(dict(reused=args.coils is not None,
        iterations=int(coil_fit.nit), optimizer_success=bool(coil_fit.success),
        message=str(coil_fit.message), initial_objective=initial_fit_cost,
        objective=float(coil_fit.fun)), indent=2)+"\n")
    print(f"[stage-two coil fit] {coil_fit.nit} iterations; B.n/B RMS = {100 * fit_rms:.3f}%; "
          f"status: {coil_fit.message}\nWrote {coil_fit_path}", flush=True)
    print("\n[single stage] Starting joint optimization from the fitted coils...", flush=True)
    joint_problem = vj.FunctionProblem.from_functions(
        coil_fit.x, value_and_grad=value_and_grad).with_acceptance(
            lambda u: plasma_problem.accept_x((x0 + scales*u)[:x_boundary0.size]))
    joint_problem.compile_value_and_gradient(report_interval=10.0)
    history = StepHistory(Path.cwd())


    def record_step(u):
        monitor(u)
        x = x0 + scales * u
        eq = plasma_problem.equilibrium_from_x(x[:x_boundary0.size])
        surface, coils = objects_from_x(jnp.asarray(x))
        result_state = eq.result
        diagnostics = limits.diagnostics(eq.state, eq.runtime)
        if coil_limits is not None:
            slack = float(np.min(coil_constraint_values(jnp.asarray(u))))
            diagnostics.update(coil_minimum_scaled_slack=slack)
            diagnostics["constraints_feasible"] &= int(slack >= 0)
            diagnostics["optimizer_constraints_feasible"] &= int(slack >= -1e-8)
        history.record(u, coils, surface, objective=monitor.records[-1].cost,
            qa=float(qs.total_state(eq.state, eq.runtime)),
            aspect=float(opt.aspect_ratio(eq.state, eq.runtime)),
            minimum_iota=float(opt.min_abs_iota(eq.state, eq.runtime)),
            aspect_target=settings.ASPECT_TARGET, iota_floor=settings.IOTA_FLOOR, coil_step=settings.COIL_STEP,
            rbc00_m=float(plasma_problem.boundary_from_x(x[:x_boundary0.size])[0][inp.ntor, 0]),
            **diagnostics,
            **{key: float(getattr(result_state, key)) for key in ("fsqr", "fsqz", "fsql")})
        np.savez_compressed(f"equilibrium_{len(history.rows)-1:04d}.npz",
            **{key: np.asarray(getattr(eq.state, key)) for key in
               ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")})
        monitor.save("single_stage_scalar_objectives.csv")
        if args.constrained:
            row = history.rows[-1]
            print(f"  constraints: min |iota|={row['min_abs_iota']:.7f}, R={row['major_radius_m']:.7f} m, "
                  f"physical limits met={bool(row['constraints_feasible'])}", flush=True)


    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()


    Path("provenance.json").write_text(json.dumps(dict(
        command=sys.argv, device=str(jax.devices()[0]), python=sys.version, jax=jax.__version__,
        input_sha256=sha(DATA), wout_sha256=sha(args.wout) if args.wout else None,
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        shared_interface_sha256=sha(Path(common.__file__)), shared_builder_sha256=sha(__file__),
        coil_constraint_settings=None if coil_limits is None else {k:v for k,v in vars(settings.P).items() if k.isupper()},
        coil_constraint_source_sha256=None if coil_limits is None else sha(settings.HERE/"_coil_constraints.py"),
        coils_sha256=sha(args.coils) if args.coils else sha(coil_fit_path),
        script_sha256=sha(settings.__file__), ftol=float(inp.ftol_array[-1]), method=settings.METHOD,
        dependencies={name: version(name) for name in ("numpy", "scipy", "essos", "solvax", "equinox", "diffrax", "lineax", "optimistix", "pyevtk")},
        core_sha256={str(p.relative_to(settings.HERE.parents[1])): sha(p) for p in sorted((settings.HERE.parents[1]/"vmex/core").glob("*.py"))},
        optimizer_options=OPTIONS, coil_step=settings.COIL_STEP, vary_major_radius=settings.VARY_MAJOR_RADIUS or args.constrained,
        constrained=args.constrained, constraint_settings={k:v for k,v in vars(limits).items() if k.isupper()},
        initial_u=coil_fit.x.tolist(), scales=scales.tolist()), indent=2)+"\n")
    Path("control_source.py").write_text(Path(settings.__file__).read_text())
    Path("fixed_support.py").write_text(Path(__file__).read_text())
    Path("single_stage_common.py").write_text(Path(common.__file__).read_text())
    for source in (Path(__file__).with_name("diagnostics.py"), Path(__file__).with_name("constraints.py")):
        Path(source.name).write_bytes(source.read_bytes())
    record_step(joint_problem.x0)

    if coil_limits is not None:
        for filename in ("parameters.py", "_coil_constraints.py"):
            Path(filename).write_text((settings.HERE/filename).read_text())
    constraints = []
    if args.constrained:
        quantities = opt.FunctionProblem.from_functions(coil_fit.x,
            residual=constraint_values, residual_jac=constraint_jacobian)
        constraints.append(common.physical_constraint(quantities, parameters=vars(settings)))
    if coil_limits is not None:
        quantities = opt.FunctionProblem.from_functions(coil_fit.x,
            residual=coil_constraint_values, residual_jac=coil_constraint_jacobian)
        constraints.append(quantities.nonlinear_constraint(0, np.inf))
    return SimpleNamespace(
        joint_problem=joint_problem,
        plasma_problem=plasma_problem,
        inp=inp,
        x0=x0,
        scales=scales,
        x_boundary0=x_boundary0,
        record_step=record_step,
        monitor=monitor,
        history=history,
        objects_from_x=objects_from_x,
        coil_field=coil_field,
        coils_fit=coils_fit,
        qs=qs,
        rbc0=rbc0,
        zbs0=zbs0,
        normalized_normal_field=normalized_normal_field,
        constraint_values=constraint_values,
        constraint_jacobian=constraint_jacobian,
        ci_smoke=ci_smoke,
        plasma_loss=plasma_loss,
        coil_costs=coil_costs,
        coil_constraint_values=coil_constraint_values,
        coil_constraint_jacobian=coil_constraint_jacobian,
        constraints=constraints,
        N_SEGMENTS=N_SEGMENTS,
        COIL_ORDER=COIL_ORDER,
        NPHI=NPHI,
        NTHETA=NTHETA,
        settings=settings, coil_limits=coil_limits, options=OPTIONS)


def verify_endpoint(stage, args, result):
    settings, coil_limits = stage.settings, stage.coil_limits
    import numpy as np
    import jax.numpy as jnp
    import vmex as vj
    from vmex import optimize as opt
    from essos.fields import BiotSavart
    from essos.surfaces import surfacerzfourier_from_boundary
    from .constraints import PhysicalConstraints
    limits = PhysicalConstraints.from_settings(settings)
    initial_value = stage.monitor.records[0].cost

    x_final = stage.x0 + stage.scales * result.x
    _, coils_final = stage.objects_from_x(jnp.asarray(x_final))
    equilibrium = stage.plasma_problem.equilibrium_from_x(x_final[:stage.x_boundary0.size])
    final_input = stage.plasma_problem.input_from_x(x_final[:stage.x_boundary0.size])
    final_input = replace(final_input,
        ns_array=np.array([31 if stage.ci_smoke else settings.VERIFY_NS]),
        ftol_array=np.array([1.0e-10 if stage.ci_smoke else settings.VERIFY_FTOL]),
        niter_array=np.array([settings.VERIFY_MAXITER]))
    final_equilibrium = opt.solve_equilibrium(
        final_input, initial_state=equilibrium.solution, verbose=not stage.ci_smoke,
        raise_on_max_iterations=True)

    surface_final = surfacerzfourier_from_boundary(
        jnp.asarray(final_input.rbc), jnp.asarray(final_input.zbs), stage.inp.nfp,
        nphi=coil_limits.VERIFY_SURFACE_GRID[0] if coil_limits else 61,
        ntheta=coil_limits.VERIFY_SURFACE_GRID[1] if coil_limits else 64)
    normal_field = np.asarray(stage.normalized_normal_field(coils_final, surface_final))
    area_weights = np.asarray(surface_final.area_element); area_weights = area_weights / area_weights.sum()
    normal_field_rms = float(np.sqrt(np.sum(area_weights * normal_field**2)))
    normal_field_max = float(np.max(np.abs(normal_field)))
    coil_points, surface_points = np.asarray(coils_final.gamma), np.asarray(surface_final.gamma).reshape(-1, 3)
    coil_surface_distance = min(float(np.linalg.norm(points[:, None] - surface_points[None], axis=2).min())
                                for points in coil_points)
    coil_pairs = [(i, j) for i in range(len(coil_points)) for j in range(i + 1, len(coil_points))]
    coil_distance = min(float(np.linalg.norm(coil_points[i][:, None] - coil_points[j][None], axis=2).min())
                        for i, j in coil_pairs)
    maximum_curvature = float(np.max(np.asarray(coils_final.curvature)))

    # Print results
    report = opt.EquilibriumReporter(
        ("QS total", stage.qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
        ("mean iota", opt.mean_iota, ".4f"), ("magnetic well", opt.magnetic_well, ".4f"))
    final_value = float(result.fun)
    report("final", final_equilibrium)
    print(f"\nObjective: {initial_value:.6e} -> {final_value:.6e} in {result.nit} {settings.METHOD} iterations")
    print(f"Coil lengths = {np.asarray(coils_final.length[:settings.N_COILS])}")
    print(f"B.n/B: area-weighted RMS = {100 * normal_field_rms:.3f}%, max = {100 * normal_field_max:.3f}% "
          f"(target < {100 * settings.NORMAL_FIELD_LIMIT:.1f}%)")
    print(f"Minimum coil-surface distance = {coil_surface_distance:.4f} m "
          f"(target >= {settings.COIL_SURFACE_DISTANCE_LIMIT:.4f} m)")
    print(f"Minimum coil-coil distance = {coil_distance:.4f} m (target >= {settings.COIL_DISTANCE_LIMIT:.4f} m)")
    print(f"Maximum curvature = {maximum_curvature:.4f} 1/m (target <= {settings.CURVATURE_LIMIT:.4f} 1/m)")

    # The coil metrics above are each printed against their limit; the plasma
    # targets were not, so a run that drove the coil terms down while leaving the
    # rotational transform near zero read as a success. A boundary with no
    # transform also makes the quasisymmetry residual trivially small, so report
    # both plasma targets explicitly and say plainly whether they were met.
    minimum_iota = float(opt.min_abs_iota(final_equilibrium.state, final_equilibrium.runtime))
    final_aspect = float(opt.aspect_ratio(final_equilibrium.state, final_equilibrium.runtime))
    print(f"Minimum |iota| = {minimum_iota:.4f} (target >= {settings.IOTA_FLOOR:.4f})")
    print(f"Aspect ratio = {final_aspect:.4f} (target {settings.ASPECT_TARGET:.4f})")
    unmet = []
    if minimum_iota < settings.IOTA_FLOOR:
        unmet.append(f"minimum |iota| {minimum_iota:.4f} below the {settings.IOTA_FLOOR:.4f} floor")
    if normal_field_rms > settings.NORMAL_FIELD_LIMIT:
        unmet.append(f"B.n/B RMS {100 * normal_field_rms:.3f}% above {100 * settings.NORMAL_FIELD_LIMIT:.1f}%")
    coil_verification = None
    if coil_limits is not None:
        coil_verification = coil_limits.verify(coils_final, surface_final, Path("coil_verification.json"))
        unmet.extend(name for name, passed in coil_verification["checks"].items() if not passed)
        if normal_field_max > settings.NORMAL_FIELD_LIMIT:
            unmet.append("maximum B.n/B exceeds limit")
        if not limits.diagnostics(final_equilibrium.state, final_equilibrium.runtime)["constraints_feasible"]:
            unmet.append("plasma iota/radius")
    if unmet:
        print("\nThis run did NOT meet its stated targets: " + "; ".join(unmet) + ".")
        print("A lower weighted objective with an unmet target is not a design. "
              "Raise the weight, reseed, or extend the budget before using it.")

    # Save results
    input_path = final_input.to_indata("input.single_stage_scalar_optimized")
    wout_path = vj.write_wout("wout_single_stage_scalar_optimized.nc", final_equilibrium.wout)
    coils_final.to_json("coils_single_stage_scalar_optimized.json")
    # ESSOS writes |B| and B.n/B on the surface and the coil filaments for ParaView.
    surface_initial = surfacerzfourier_from_boundary(
        stage.rbc0, stage.zbs0, stage.inp.nfp, nphi=60, ntheta=60)
    surface_initial.to_vtk("surface_single_stage_scalar_initial", field=BiotSavart(stage.coils_fit))
    stage.coils_fit.to_vtk("coils_single_stage_scalar_initial")
    field_final = BiotSavart(coils_final)
    surface_final.to_vtk("surface_single_stage_scalar_optimized", field=field_final)
    coils_final.to_vtk("coils_single_stage_scalar_optimized")
    print(f"Wrote {input_path}\nWrote {wout_path}")
    print("Wrote coils_single_stage_scalar_optimized.json")
    print("Wrote initial and optimized surface/coils VTK files")

    Path("optimization_summary.json").write_text(json.dumps(dict(
        optimizer_success=bool(result.success), message=str(result.message),
        derivative_qualified=False, coil_constraints=coil_verification, inequalities_met=not unmet, unmet=unmet,
        accepted_steps=int(result.accepted_steps), optimizer_iterations=int(result.nit), evaluations=int(result.nfev),
        best_feasible_step=min((row for row in stage.history.rows if row["constraints_feasible"]), key=lambda row:row["objective"], default={}).get("step"),
        final_qa=float(stage.qs.total(final_equilibrium)), final_aspect=final_aspect,
        **limits.diagnostics(final_equilibrium.state, final_equilibrium.runtime),
        final_min_abs_iota=minimum_iota), indent=2)+"\n")
    if args.no_plots:
        return 0 if result.success and not unmet else 2

    # Plot results
    print("Plotting results...")
    vj.plot_optimization_objects("single_stage_scalar_optimization.png",
        ("Initial (stage-two fit)", surface_initial, stage.coils_fit), ("Optimized", surface_final, coils_final))
    stage.monitor.save("single_stage_scalar_objectives.csv")
    stage.monitor.plot("single_stage_scalar_objectives.png", title="Single-stage objective terms")
    print("Wrote single_stage_scalar_optimization.png")
    print("Wrote single_stage_scalar_objectives.csv and single_stage_scalar_objectives.png")
    if args.movie:
        print("Making movie of accepted iterates...")
        stage.monitor.movie_surface_coils("single_stage_scalar_optimization.gif", stage.objects_from_x,
            x0=stage.x0, scales=stage.scales, surface_color=settings.MOVIE_SURFACE_COLOR, plasma_problem=stage.plasma_problem,
            external_field=lambda objects: stage.coil_field(objects[1]), nphi=stage.NPHI, ntheta=stage.NTHETA, cmap="jet")
    for path in vj.plot_wout(wout_path, ".").values():
        print(f"Wrote {path}")

    return 0 if result.success and not unmet else 2
