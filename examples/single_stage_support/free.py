"""Shared free-boundary logging, endpoint checks and plot exports.

The production entry supplies its physical problem and optimizer. Derivative
qualification lives in separate verification programs and is never invoked here.
"""
import csv
import os
from pathlib import Path
import sys
from . import common
from dataclasses import replace
import json
from functools import lru_cache
import signal
import time
from types import SimpleNamespace

def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=str, allow_nan=False)+"\n")

def run(args, *, case, method="SLSQP", coil_limits=None):
    """Prepare or restore a start, optimize and independently solve the endpoint."""
    if method not in ("SLSQP", "L-BFGS-B"):
        raise ValueError(f"unsupported optimizer: {method}")
    settings = {name: value for name, value in vars(case).items() if name.isupper() and name not in ("HERE", "P")}
    if args.dry_run:
        print(json.dumps(dict(optimizer=method, parameters=settings, arguments=vars(args)), indent=2, default=str))
        return 0
    out = case.setup_run(args)
    if getattr(case, "COIL_CONSTRAINTS", False):
        import _coil_constraints as coil_limits
    import numpy as np
    from vmex import optimize as opt
    qualified = case.read_qualification(args)

    def timeout(*_):
        raise TimeoutError("phase wall-time budget reached")

    previous_handler = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(case.INITIALIZATION_SECONDS)
    problem = None
    phase = "initialization"
    started = time.perf_counter()
    try:
        timings = dict(adjoint=0.0, tangent=0.0, correction=0.0, root_polish=0.0)
        trials = 0

        def event(name, **data):
            nonlocal trials
            if name in timings and 'seconds' in data:
                timings[name] += data.get('total_seconds', data['seconds'])
            if name == "proposal":
                trials += 1
                if trials > case.MAX_TRIALS:
                    raise StopIteration("trial budget reached")
            if name in ("adjoint", "tangent", "matrixfree_check", "dense_recovery", "preconditioner_refresh", "preconditioner_refresh_start", "root_polish"):
                # Keep diagnostic rows and failure reasons without serializing root arrays.
                record = {key: value for key, value in data.items() if key != "candidate"}
                with (out / "solver_events.jsonl").open("a") as stream:
                    stream.write(json.dumps({"event": name, "proposal_count": trials, **record}, default=str) + "\n")

        stage = case.build_problem(args, event=event, qualified=qualified)
        problem, inp = stage.problem, stage.inp
        qs, inequalities = stage.qs, stage.inequalities
        initial_equilibrium = problem.equilibrium_from_x(problem.x0)
        monitor = opt.OptimizationMonitor(stream=None)
        history = []
        cycle_started = time.perf_counter()

        def record_step():
            nonlocal cycle_started
            x = problem.accepted.parameters
            eq = problem.equilibrium_from_x(x)
            physical = problem.constraint_values(x)
            iota, radius = physical[:2]
            row = dict(step=problem.accepted_step, objective=problem.fun(x),
                qa=float(qs.total_state(eq.state, eq.runtime)), aspect=float(opt.aspect_ratio(eq.state, eq.runtime)),
                min_abs_iota=float(iota), major_radius_m=float(radius),
                rbc00_m=float(opt.boundary_from_state(eq.state, eq.runtime)[0][inp.ntor, 0]),
                radius_error_m=float(radius-case.RADIUS_TARGET), iota_constraint_slack=float(iota-case.IOTA_FLOOR),
                radius_constraint_slack_m=float(case.RADIUS_TOLERANCE-abs(radius-case.RADIUS_TARGET)),
                optimizer_constraints_feasible=bool(np.min(inequalities(physical)) >= -1e-8),
                constraints_feasible=bool(iota >= case.IOTA_FLOOR and abs(radius-case.RADIUS_TARGET) <= case.RADIUS_TOLERANCE),
                gradient_seconds=timings["adjoint"], predictor_seconds=timings["tangent"], solve_seconds=timings["correction"],
                polish_seconds=timings['root_polish'], root_residual=float(problem.accepted.root_residual_norm),
                step_seconds=time.perf_counter()-cycle_started, elapsed_seconds=time.perf_counter()-started,
                **{key: float(getattr(eq.result, key)) for key in ("fsqr", "fsqz", "fsql", "fedge")})
            if coil_limits is not None:
                slack = float(np.min(stage.coil_constraint.fun(x)))
                clearance = float(physical[2])
                row.update(coil_minimum_scaled_slack=slack, coil_surface_distance_m=clearance)
                row["optimizer_constraints_feasible"] &= slack >= -1e-8
                row["constraints_feasible"] &= slack >= 0 and clearance >= case.COIL_SURFACE_DISTANCE_LIMIT
            history.append(row)
            write_json(out / "accepted_steps.json", history)
            write_json(out / f"checkpoint_{problem.accepted_step:04d}.json", problem.save_checkpoint(out/f"accepted_{problem.accepted_step:04d}.npz"))
            monitor.record(x, cost=row["objective"], iteration=problem.accepted_step)
            monitor.save(out / "free_boundary_scalar_objectives.csv")
            print(f"[step {problem.accepted_step}] objective {row['objective']:.6e}; QA {row['qa']:.6e}; "
                  f"gradient {row['gradient_seconds']:.2f}s, predictor {row['predictor_seconds']:.2f}s, "
                  f"correction {row['solve_seconds']:.2f}s", flush=True)
            timings.update(dict.fromkeys(timings, 0.0))
            cycle_started = time.perf_counter()

        # Production constructs its own checked LU; parity/FD experiments live
        # exclusively in verify_free_boundary_single_stage.py.
        case.configure_solver(problem, args)
        record_step()
        phase = "optimization"
        optimization_started = time.perf_counter()
        signal.alarm(case.OPTIMIZATION_SECONDS)

        status, result = "iteration_budget_reached", None
        try:
            result = case.run_optimizer(stage, args, record_step, method=method)
            if result.stop_reason:
                status = result.stop_reason
            elif result.success:
                status = "converged"
            else:
                budget_status = 1 if method == "L-BFGS-B" else 9
                status = "iteration_or_evaluation_budget_reached" if result.status == budget_status else "optimizer_failed"
        except StopIteration as error:
            status = str(error).replace(" ", "_")
        except TimeoutError:
            status = "optimization_time_budget_reached"
        except opt.TrialRejected as error:
            # L-BFGS-B needs a gradient for every proposal. Stop cleanly at the
            # last certified accepted state when a proposal cannot supply one.
            status = "equilibrium_trial_rejected"
            write_json(out / "rejected_trial.json", dict(error=str(error)))
        summary = dict(optimizer=method, linear_solver=problem.solver_info, nonlinear_constraints=method == "SLSQP", status=status, optimizer_success=status == "converged", accepted_steps=problem.accepted_step,
            optimization_seconds=time.perf_counter()-optimization_started,
            derivative_qualified=qualified is not None,
            qualification=None if args.qualification is None else str(args.qualification.resolve()),
            seed_manifest=None if args.seed is None else str(args.seed.resolve()),
            initial=history[0], final=history[-1], verification="not run",
            optimizer_message=None if result is None else str(result.message))
        write_json(out / "optimization_summary.json", summary)

        phase = "verification"
        verified = case.verify_endpoint(stage, args, summary, history, initial_equilibrium)
        phase = "postprocessing"
        case.postprocess(stage, args, summary, monitor, history, initial_equilibrium, verified)
        print(f"{status}; numerical verification passed; physical limits met: "
              f"{summary['inequalities_met']}. Results: {out}", flush=True)
        return 0 if summary["optimizer_success"] and summary["inequalities_met"] else 1
    except Exception as error:
        write_json(out / "failure.json", dict(phase=phase, error=f"{type(error).__name__}: {error}"))
        raise
    finally:
        if problem is not None:
            problem.close()
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)

def verify_endpoint(case, coil_limits, stage, args, summary, history, initial_equilibrium):
    """Re-solve the final coils at higher resolution and report physical limits."""
    import jax
    import numpy as np
    import vmex as vj
    from vmex import optimize as opt
    from essos.fields import BiotSavart
    from essos.surfaces import SurfaceRZFourier

    out = args.output.resolve()
    problem, inp, qs = stage.problem, stage.inp, stage.qs
    method = summary["optimizer"]
    verification = None
    try:
        signal.alarm(case.VERIFICATION_SECONDS)
        accepted_eq = problem.equilibrium_from_x(problem.accepted.parameters)
        initial_wout = vj.write_wout(out / "wout_initial.nc", initial_equilibrium.wout)
        accepted_wout = vj.write_wout(out / "wout_accepted.nc", accepted_eq.wout)
        final_coils = problem.coils_from_x(problem.accepted.parameters)
        final_coils.to_json(str(out / "coils_optimized.json"))
        rbc, zbs, _, _ = opt.boundary_from_state(accepted_eq.state, accepted_eq.runtime)
        final_input = replace(inp, rbc=np.asarray(rbc), zbs=np.asarray(zbs), ns_array=np.array([case.VERIFY_NS]),
                              ftol_array=np.array([case.VERIFY_FTOL]), niter_array=np.array([case.VERIFY_MAXITER]))
        final_input.to_indata(out / "input.optimized")
        problem.close()
        verification = opt.FreeBoundaryProblem.from_tuples(final_input, [(qs.residuals_state, 0, 1)],
            coils=final_coils, coil_current_dofs=(), restart_from=accepted_wout, objective_normalization=1.0,
            solver_options=dict(device=args.device, ftol=case.VERIFY_FTOL, edge_force_tolerance=case.VERIFY_FTOL,
                                max_iterations=case.VERIFY_MAXITER))
        verified = verification.equilibrium_from_x(verification.x0)
        forces = {key: float(getattr(verified.result, key)) for key in ("fsqr", "fsqz", "fsql", "fedge")}
        if not verified.result.converged or not all(np.isfinite(v) and v <= case.VERIFY_FTOL for v in forces.values()):
            raise RuntimeError(f"independent endpoint force checks failed: {forces}")
        wout_path = vj.write_wout(out / "wout_optimized.nc", verified.wout)
        surf = SurfaceRZFourier.from_wout_file(wout_path, nphi=coil_limits.VERIFY_SURFACE_GRID[0] if coil_limits else 61,
                                              ntheta=coil_limits.VERIFY_SURFACE_GRID[1] if coil_limits else 64)
        field = np.asarray(jax.vmap(BiotSavart(final_coils).B)(surf.gamma.reshape(-1, 3))).reshape(surf.gamma.shape)
        bn = np.sum(field*np.asarray(surf.unitnormal), axis=2)/np.linalg.norm(field, axis=2)
        weights = np.asarray(surf.area_element)
        rms, maximum = float(np.sqrt(np.sum(weights*bn**2)/weights.sum())), float(np.max(np.abs(bn)))
        points, surface_points = np.asarray(final_coils.gamma), np.asarray(surf.gamma).reshape(-1, 3)
        coil_distance = min(float(np.linalg.norm(points[i][:, None]-points[j][None], axis=2).min())
                            for i in range(len(points)) for j in range(i+1, len(points)))
        clearance = min(float(np.linalg.norm(p[:, None]-surface_points[None], axis=2).min()) for p in points)
        iota, radius = float(opt.min_abs_iota(verified.state, verified.runtime)), float(opt.major_radius(verified.state, verified.runtime))
        geometry_report = None
        curvature = float(np.max(np.asarray(final_coils.curvature)))
        if coil_limits is not None:
            geometry_report = coil_limits.verify(final_coils, surf, out / "coil_verification.json")
            curvature = max(r["peak_per_m"] for r in geometry_report["coils"])
            coil_distance = geometry_report["coil_coil_lower_bound_m"]
            clearance = geometry_report["refined_coil_surface_distance_m"]
        reporter = opt.EquilibriumReporter(
            ("QS total", qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
            ("mean iota", opt.mean_iota, ".4f"), ("magnetic well", opt.magnetic_well, ".4f"))
        reported = reporter("final verification", verified)
        lengths = np.asarray(final_coils.length[:case.N_COILS])
        print(f"\nObjective: {history[0]['objective']:.6e} -> {history[-1]['objective']:.6e} "
              f"in {problem.accepted_step} accepted {method} steps")
        print(f"Coil lengths = {lengths} (length reference {case.LENGTH_TARGET:.4f} m)")
        print(f"B.n/B: RMS = {100*rms:.3f}%, max = {100*maximum:.3f}% "
              f"(target < {100*case.NORMAL_FIELD_LIMIT:.1f}%)")
        print(f"Minimum coil-surface distance = {clearance:.4f} m (target >= {case.COIL_SURFACE_DISTANCE_LIMIT:.4f} m)")
        print(f"Minimum coil-coil distance = {coil_distance:.4f} m (target >= {case.COIL_DISTANCE_LIMIT:.4f} m)")
        print(f"Maximum curvature = {curvature:.4f} 1/m (target <= {case.CURVATURE_LIMIT:.4f} 1/m)")
        print(f"Minimum |iota| = {iota:.4f} (target >= {case.IOTA_FLOOR:.4f}); "
              f"aspect = {reported['aspect']:.4f} (soft target {case.ASPECT_TARGET:.4f})")
        print(f"Major radius = {radius:.6f} m (target {case.RADIUS_TARGET:.4f} +/- {case.RADIUS_TOLERANCE:.4f} m)")
        checks = [("minimum |iota|", iota, case.IOTA_FLOOR, "below"),
                  ("radius error [m]", abs(radius-case.RADIUS_TARGET), case.RADIUS_TOLERANCE, "above"),
                  ("B.n/B RMS", rms, case.NORMAL_FIELD_LIMIT, "above"),
                  ("B.n/B maximum", maximum, case.NORMAL_FIELD_LIMIT, "above"),
                  ("coil-surface distance [m]", clearance, case.COIL_SURFACE_DISTANCE_LIMIT, "below"),
                  ("coil-coil distance [m]", coil_distance, case.COIL_DISTANCE_LIMIT, "below"),
                  ("curvature [1/m]", curvature, case.CURVATURE_LIMIT, "above")]
        unmet = [f"{name}: {value:.6g} {side} {limit:.6g}" for name, value, limit, side in checks
                 if not np.isfinite(value) or (value < limit if side == "below" else value > limit)]
        if geometry_report is not None:
            unmet += [name for name,passed in geometry_report["checks"].items() if not passed]
        feasible = not unmet
        if unmet:
            print("This run did NOT meet its stated limits: " + "; ".join(unmet))
        summary.update(verification="passed numerical checks", inequalities_met=feasible, unmet=unmet,
            coil_constraints=geometry_report,
            best_feasible_step=min((row for row in history if row["constraints_feasible"]),
                                   key=lambda row: row["objective"], default={}).get("step"),
            verified=dict(forces=forces, ns=case.VERIFY_NS, qa=reported["QS total"], aspect=reported["aspect"],
                          mean_iota=reported["mean iota"], magnetic_well=reported["magnetic well"], min_abs_iota=iota,
                          major_radius_m=radius, normal_field_rms=rms, normal_field_max=maximum,
                          coil_distance_m=coil_distance, coil_surface_distance_m=clearance, maximum_curvature=curvature,
                          coil_lengths_m=lengths.tolist(), aspect_target_error=reported["aspect"]-case.ASPECT_TARGET,
                          radius_error_m=radius-case.RADIUS_TARGET, iota_constraint_slack=iota-case.IOTA_FLOOR,
                          radius_constraint_slack_m=case.RADIUS_TOLERANCE-abs(radius-case.RADIUS_TARGET),
                          full_mesh_min_abs_iota=float(np.min(np.abs(verified.wout.iotaf)))))
        write_json(out / "optimization_summary.json", summary)

        return SimpleNamespace(initial_wout=initial_wout, wout_path=wout_path,
                               final_coils=final_coils, surface=surf)
    finally:
        if verification is not None:
            verification.close()

def postprocess(case, stage, args, summary, monitor, history, initial_equilibrium, verified):
    """Export accepted-state diagnostics, geometry, figures, movie and WOUT plots."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    import vmex as vj
    from vmex import optimize as opt
    from essos.fields import BiotSavart
    from essos.surfaces import SurfaceRZFourier

    out = args.output.resolve()
    problem, inp, chart, coils0 = stage.problem, stage.inp, stage.chart, stage.coils
    surface, coil_costs, scales = stage.surface, stage.coil_costs, stage.chart.scales
    initial_wout, wout_path = verified.initial_wout, verified.wout_path
    surf, final_coils = verified.surface, verified.final_coils
    _phase = "postprocessing"
    signal.alarm(case.POSTPROCESSING_SECONDS)
    postprocessing_started = time.perf_counter()
    seed_surface = SurfaceRZFourier.from_wout_file(initial_wout, nphi=60, ntheta=60)
    for label, export_surface, export_coils in (("initial", seed_surface, coils0), ("optimized", surf, final_coils)):
        export_surface.to_vtk(str(out / f"surface_{label}"), field=BiotSavart(export_coils))
        export_coils.to_vtk(str(out / f"coils_{label}"))

    # Replay saved accepted roots, never solve previous iterates again.
    points = monitor.x_history
    frame_steps = {tuple(x): step for step, x in enumerate(points)}

    @lru_cache(maxsize=1)
    def accepted_frame(step):
        checkpoint = out / f"accepted_{step:04d}.npz"
        identity = json.loads((out / f"checkpoint_{step:04d}.json").read_text())
        state = problem.state_from_checkpoint(checkpoint, sha256=identity["sha256"], parameters=points[step])
        return state, surface(state, initial_equilibrium.runtime), chart.coils_from_x(jnp.asarray(points[step]))

    post_monitor = opt.OptimizationMonitor(stream=None)
    coil_cost_values = jax.jit(coil_costs)
    coil_diagnostics = opt.CoilDiagnostics(scales, coefficient_step=case.COIL_STEP)
    for step, (x, row) in enumerate(zip(points, history)):
        _, step_surface, step_coils = accepted_frame(step)
        row.update(coil_diagnostics.record(x, step_coils, path=out / f"step_{step:04d}.npz"))
        coil_field = np.asarray(jax.vmap(BiotSavart(step_coils).B)(step_surface.gamma.reshape(-1, 3))).reshape(step_surface.gamma.shape)
        normal_field = np.sum(coil_field*np.asarray(step_surface.unitnormal), axis=-1)/np.linalg.norm(coil_field, axis=-1)
        area = np.asarray(step_surface.area_element)
        row.update(qa_residual_l2=float(np.sqrt(row["qa"])), aspect_error=row["aspect"]-case.ASPECT_TARGET,
            iota_violation=max(case.IOTA_FLOOR-row["min_abs_iota"], 0.0),
            normal_field_rms=float(np.sqrt(np.sum(area*normal_field**2)/area.sum())),
            normal_field_max=float(np.max(np.abs(normal_field))))
        if not all(np.isfinite(value) for value in row.values()):
            raise ValueError(f"nonfinite accepted-step diagnostics at step {step}")
        terms = dict(quasisymmetry=0.5*row["qa"], aspect=0.5*case.ASPECT_WEIGHT*row["aspect_error"]**2,
                     **{"iota floor": 0.5*case.IOTA_WEIGHT*row["iota_violation"]**2})
        terms.update(zip(("coil length", "coil curvature", "coil separation", "coil-surface separation"),
                         map(float, np.asarray(coil_cost_values(step_coils, step_surface)))))
        np.testing.assert_allclose(sum(terms.values()), row["objective"], rtol=1e-10, atol=1e-12)
        post_monitor.record(x, cost=row["objective"], iteration=step, terms=terms)
    with (out / "accepted_steps.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    write_json(out / "accepted_steps.json", history)
    post_monitor.save(out / "free_boundary_scalar_objectives.csv")

    if not args.no_plots:
        vj.plot_optimization_objects(out / "optimization.png", ("Initial", seed_surface, coils0), ("Optimized", surf, final_coils))
        post_monitor.plot(out / "objectives.png", title="Single-stage objective terms")
        if args.movie:
            def objects_from_x(x):
                return accepted_frame(frame_steps[tuple(x)])[1:]

            def movie_colors(x, objects):
                state = accepted_frame(frame_steps[tuple(x)])[0]
                data = vj.surface_field_data_from_state(inp, state, runtime=initial_equilibrium.runtime,
                                                       nphi=case.NPHI, ntheta=case.NTHETA)
                magnitude = jnp.linalg.norm(data.B_total, axis=0)
                if case.MOVIE_SURFACE_COLOR == "absB":
                    return magnitude
                interface = vj.PlasmaVacuumInterface.from_surface_data(data, digits=4)
                return interface.bnormal_residual(chart(jnp.asarray(x)))/magnitude

            post_monitor.movie(out / "optimization.gif", objects_from_x,
                color_factory=movie_colors if case.MOVIE_SURFACE_COLOR is not None else None,
                color_label=case.MOVIE_SURFACE_COLOR, cmap="jet")
        for path in vj.plot_wout(wout_path, out).values():
            print(f"Wrote {path}")
    summary.update(postprocessing="complete", postprocessing_seconds=time.perf_counter()-postprocessing_started)
    write_json(out / "optimization_summary.json", summary)


def setup_run(args):
    """Set project-local output/cache paths before importing JAX."""
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in ("TMPDIR", "XDG_CACHE_HOME", "MPLCONFIGDIR", "CUDA_CACHE_PATH"):
        directory = out / "cache" / name
        directory.mkdir(parents=True)
        os.environ[name] = str(directory)
    os.environ.update(JAX_ENABLE_X64="1", JAX_PLATFORMS="cuda,cpu" if args.device == "gpu" else "cpu",
                      VMEX_COMPILATION_CACHE="disabled", JAX_ENABLE_COMPILATION_CACHE="false", MPLBACKEND="Agg")
    import jax
    if jax.default_backend() != args.device or not jax.config.x64_enabled:
        raise RuntimeError("requested backend and float64 precision are required")
    return out

def qualification_contract(args, *, case):
    """Fingerprint both the short entry and the shared numerical implementation."""
    from vmex import optimize as opt
    excluded = {"P", "HERE", "INPUT", "COILS", "ACCEPTED_STEPS", "MAX_TRIALS", "MAKE_MOVIE", "MOVIE_SURFACE_COLOR"}
    parameters = {name: value for name, value in vars(case).items()
                  if name.isupper() and name not in excluded and not name.endswith("_SECONDS")}
    sources = [Path(__file__), Path(common.__file__)]
    if getattr(case, "COIL_CONSTRAINTS", False):
        parameters["coil_constraints"] = {name: value for name, value in vars(case.P).items()
                                          if name.isupper() and name != "ACCEPTED_STEPS"}
        sources.append(case.HERE / "_coil_constraints.py")
    return opt.OptimizationQualification.signature(parameters=parameters,
        input_path=args.input, wout_path=args.wout, resolution=args.resolution,
        grid=args.grid, ftol=args.ftol, device=args.device, sources=sources,
        functions=(case.build_problem, case.configure_solver, case.run_optimizer))

def read_qualification(args, *, case):
    """Reuse the same authenticated fitted coils and accepted seed."""
    from vmex import optimize as opt

    if args.qualification is None:
        return None
    qualified = opt.OptimizationQualification.read(args.qualification, contract=case.qualification_contract(args))
    if args.coils is not None and common.sha(args.coils) != qualified.report["artifacts"]["coils"]["sha256"]:
        raise ValueError("supplied coils differ from the qualified fitted coils")
    return qualified

def build_problem(args, *, case, event=None, qualified=None, coil_limits=None):
    """Prepare the input, stage-two coils and common scalar VMEX problem.

    Without a saved bundle, load or fit coils and solve the seed from input/WOUT.
    A bundle restores the exact seed and coils, skipping fitting and the initial
    solve. Seed manifests do not claim derivative qualification for new code.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scipy.optimize import minimize
    from vmex import optimize as opt
    from essos.coils import Coils
    from essos.fields import BiotSavart
    from essos.objective_functions import loss_coil_separation, loss_coil_surface_distance
    from essos.surfaces import surfacerzfourier_from_boundary

    out = args.output.resolve()
    contract = case.qualification_contract(args)
    if qualified is None and args.seed is not None:
        qualified = common.read_seed(args.seed, contract=contract)
        if args.coils is not None and common.sha(args.coils) != qualified[0]["artifacts"]["coils"]["sha256"]:
            raise ValueError("supplied coils differ from the saved seed")
    inp, seed = common.load_input(args, restore_state=qualified is None)
    qs = opt.QuasisymmetryRatioResidual(np.asarray(case.QA_SURFACES), 1, 0)

    def surface(state, runtime):
        rbc, zbs, _, _ = opt.boundary_from_state(state, runtime)
        return surfacerzfourier_from_boundary(rbc, zbs, inp.nfp, nphi=case.NPHI, ntheta=case.NTHETA)

    def iota_floor(state, runtime):
        return jnp.maximum(case.IOTA_FLOOR-opt.min_abs_iota(state, runtime), 0.0)

    plasma_terms = [(qs.residuals_state, 0.0, 1.0),
                    (opt.aspect_ratio, case.ASPECT_TARGET, case.ASPECT_WEIGHT),
                    (iota_floor, 0.0, case.IOTA_WEIGHT)]

    def coil_costs(coils, surf):
        if coil_limits is not None:
            return jnp.empty(0)
        return jnp.array([
            0.5*case.LENGTH_WEIGHT*jnp.sum((coils.length[:case.N_COILS]-case.LENGTH_TARGET)**2),
            0.5*case.CURVATURE_WEIGHT*jnp.sum(jnp.maximum(coils.curvature[:case.N_COILS]-case.CURVATURE_OBJECTIVE_LIMIT, 0)**2),
            0.5*case.COIL_DISTANCE_WEIGHT*loss_coil_separation(coils, case.COIL_DISTANCE_LIMIT, block_size=32),
            0.5*case.COIL_SURFACE_DISTANCE_WEIGHT*loss_coil_surface_distance(
                coils, surf, case.COIL_SURFACE_DISTANCE_LIMIT, block_size=32)])

    def loss(state, runtime, coils):
        residuals = opt.residuals_from_tuples(state, runtime, plasma_terms)
        return 0.5*jnp.vdot(residuals, residuals) + jnp.sum(coil_costs(coils, surface(state, runtime)))

    def inequalities(values):
        iota, radius = values[:2]
        width = case.RADIUS_TOLERANCE-case.RADIUS_MARGIN
        rows = [(iota-case.IOTA_FLOOR-case.IOTA_MARGIN)/case.IOTA_FLOOR,
                         (radius-case.RADIUS_TARGET+width)/case.RADIUS_TOLERANCE,
                         (case.RADIUS_TARGET+width-radius)/case.RADIUS_TOLERANCE]
        if coil_limits is not None:
            rows.append((values[2]-case.COIL_SURFACE_DISTANCE_LIMIT-case.DISTANCE_MARGIN)/case.COIL_SURFACE_DISTANCE_LIMIT)
        return np.asarray(rows)

    constraint_transform = np.array([[1/case.IOTA_FLOOR, 0], [0, 1/case.RADIUS_TOLERANCE], [0, -1/case.RADIUS_TOLERANCE]])

    def clearance(state, runtime, coils):
        rbc, zbs, _, _ = opt.boundary_from_state(state, runtime)
        surf = surfacerzfourier_from_boundary(rbc, zbs, inp.nfp,
            nphi=coil_limits.SURFACE_GRID[0], ntheta=coil_limits.SURFACE_GRID[1])
        return coil_limits.surface_distance(coils, surf)

    if coil_limits is not None:
        constraint_transform = np.pad(constraint_transform, ((0, 1), (0, 1)))
        constraint_transform[-1, -1] = 1 / case.COIL_SURFACE_DISTANCE_LIMIT

    coil_path = qualified[1]["coils"] if qualified is not None else args.coils or args.initial_coils
    coils0 = common.initial_coils(inp, coil_path, parameters=vars(case), resize=coil_limits is not None)
    fit_report = dict(reused=True, iterations=0)
    if qualified is None and args.coils is None:
        # Stage two changes only geometry on the frozen input/WOUT surface.
        # These are the scalar reference's two B.n penalties plus coil terms.
        seed_surface = surfacerzfourier_from_boundary(jnp.asarray(inp.rbc), jnp.asarray(inp.zbs),
                                                      inp.nfp, nphi=case.NPHI, ntheta=case.NTHETA)
        x0 = np.asarray(coils0.curves.dofs).ravel()

        def fit_loss(u):
            coils = coils0.with_dofs(jnp.concatenate((jnp.asarray(x0)+case.COIL_STEP*u, coils0.dofs_currents)))
            field = jax.vmap(BiotSavart(coils).B)(seed_surface.gamma.reshape(-1, 3)).reshape(seed_surface.gamma.shape)
            normal = jnp.sum(field*seed_surface.unitnormal, axis=-1)/jnp.linalg.norm(field, axis=-1)
            weights = seed_surface.area_element/jnp.sum(seed_surface.area_element)
            maximum = jax.scipy.special.logsumexp(2000*jnp.sqrt(normal**2+1e-12))/2000
            return (0.5*case.NORMAL_FIELD_WEIGHT*jnp.sum(weights*normal**2)
                    + 0.5*case.NORMAL_FIELD_LIMIT_WEIGHT*jnp.maximum(maximum-case.NORMAL_FIELD_OBJECTIVE_LIMIT, 0)**2
                    + jnp.sum(coil_costs(coils, seed_surface)))

        fit_gradient = jax.jit(jax.value_and_grad(fit_loss))
        initial_cost = float(fit_loss(jnp.zeros_like(jnp.asarray(x0))))
        fit = minimize(fit_gradient, np.zeros_like(x0), jac=True, method="L-BFGS-B",
                       bounds=[(-5., 5.) if coil_limits is not None else (-case.PARAMETER_BOUND, case.PARAMETER_BOUND)]*x0.size,
                       options=dict(maxiter=args.coil_fit_maxiter, maxcor=20, ftol=1e-15, gtol=1e-10))
        if not np.isfinite(fit.fun) or not np.all(np.isfinite(fit.x)) or fit.fun > initial_cost + 1e-10:
            raise RuntimeError("stage-two fitting returned invalid or worse coils")
        currents = np.asarray(coils0.dofs_currents_raw).copy()
        coils0 = coils0.with_dofs(jnp.concatenate((jnp.asarray(x0)+case.COIL_STEP*fit.x, coils0.dofs_currents)))
        if not np.array_equal(coils0.dofs_currents_raw, currents):
            raise RuntimeError("stage-two fitting changed fixed coil currents")
        fit_report = dict(reused=False, iterations=int(fit.nit), optimizer_success=bool(fit.success),
                          message=str(fit.message), initial_objective=initial_cost, objective=float(fit.fun))
    # Reload the saved representation once so qualification and production
    # construct identical charts, including floating-point serialization.
    fitted_path = out / "coils.stage2.json"
    coils0.to_json(str(fitted_path))
    coils0 = Coils.from_json(str(fitted_path))
    write_json(out / "stage_two.json", fit_report)
    scales = case.COIL_STEP / np.broadcast_to(np.asarray(coils0.curves.scaling), coils0.dofs_curves.shape).ravel()
    chart = opt.CoilParameters.from_coils(coils0, current_dofs=(), scales=scales)
    if qualified is None and seed is None:
        seed = opt.solve_equilibrium(inp, device=args.device, raise_on_max_iterations=True,
                                     polish_force_balance=False).state
    inp = replace(inp, lfreeb=True, mgrid_file="direct ESSOS field", ftol_array=np.array([args.ftol]))
    restart = dict(restart_from=seed) if qualified is None else dict(
        checkpoint=qualified[1]["checkpoint"], checkpoint_sha256=qualified[0]["artifacts"]["checkpoint"]["sha256"])
    write_json(out / "provenance.json", dict(contract=contract, arguments=vars(args), command=sys.argv, script_sha256=common.sha(case.__file__),
        currents_A=chart.currents.tolist(), scales=scales.tolist(), fitted_coils_sha256=common.sha(fitted_path)))
    for source in (Path(case.__file__), Path(__file__), Path(common.__file__)):
        (out / source.name).write_bytes(source.read_bytes())
    if coil_limits is not None:
        for source in (case.HERE / "parameters.py", Path(coil_limits.__file__)):
            (out / source.name).write_bytes(source.read_bytes())
    problem = opt.FreeBoundaryProblem.from_loss(inp, loss, quantities=(opt.min_abs_iota, opt.major_radius),
        coil_quantities=(clearance,) if coil_limits is not None else (),
        parameterization=chart, root_residual_atol=case.ROOT_TOLERANCE, event=event, checkpoint_identity=contract,
        solver_options=dict(device=args.device, ftol=args.ftol, edge_force_tolerance=args.ftol,
            max_iterations=int(inp.niter_array[-1]), adjoint_dense_batch_size=case.ADJOINT_BATCH_SIZE,
            adjoint_dense_max_dofs=case.ADJOINT_MAX_DOFS, adjoint_residual_rtol=case.ADJOINT_RESIDUAL_RTOL), **restart)
    if problem.accepted_step != 0:
        problem.close()
        raise ValueError("saved seed must describe an initial equilibrium at accepted step zero")
    before = problem.accepted.root_residual_norm if coil_limits is not None else None
    try:
        problem.enable_root_polishing(tolerance=case.ROOT_POLISH_TOLERANCE)
    except BaseException:
        problem.close()
        raise
    if coil_limits is not None:
        write_json(out / "root_polish_initial.json", dict(tolerance=case.ROOT_POLISH_TOLERANCE,
            before=float(before), after=float(problem.accepted.root_residual_norm), solver=problem.solver_info))
    return SimpleNamespace(problem=problem, inp=inp, chart=chart, coils=coils0, qs=qs,
                           surface=surface, coil_costs=coil_costs, inequalities=inequalities,
                           constraint_transform=constraint_transform, contract=contract,
                           coil_constraint=None if coil_limits is None else coil_limits.constraint(chart.coils_from_x))

def configure_solver(problem, args, *, case):
    """Prepare the production solver without qualification experiments."""
    problem.enable_matrix_free(rtol=case.MATRIXFREE_RTOL, restart=case.MATRIXFREE_RESTART,
        max_restarts=case.MATRIXFREE_MAX_CYCLES, rhs_batch_size=case.MATRIXFREE_RHS_BATCH_SIZE,
        refresh_horizon=case.LU_REFRESH_HORIZON, refresh_max_steps=args.accepted_steps)
