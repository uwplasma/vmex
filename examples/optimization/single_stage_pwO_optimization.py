#!/usr/bin/env python
"""Fit coils, then optimize the plasma boundary and coils for mapped pwO.

Reads the stage-1 output. TARGET_BETA selects vacuum or finite beta; the
boundary-field match includes virtual casing at finite beta. Geometry limits
apply to filaments at ARIES-CS scale. Validate the coil-driven equilibrium
separately from particle confinement and engineering tolerances.
"""

import json
import os
from dataclasses import replace
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

import vmex as vj
from vmex import optimize as opt
from vmex.core import implicit as im
from vmex.core import virtual_casing as vc
from vmex.core.pwo import MappedPwOResidual
from vmex.core.restart import restart_state
from vmex.core.scaling import aries_cs_scales
from essos.coils import Coils, Curves, fit_dofs_from_coils
from essos.fields import BiotSavart
from essos.objective_functions import loss_coil_separation, loss_coil_surface_distance
from essos.surfaces import surfacerzfourier_from_boundary

TARGET_BETA = 0.0
INPUT_FILE = Path("input.pwO_finite_beta_optimized" if TARGET_BETA else "input.pwO_optimized")
N_COILS, COIL_ORDER, N_SEGMENTS = 4, 5, 96
MAX_MODE, MAXITER, PREFIT_ITERATIONS = 2, 20, 200
NPHI, NTHETA, NS = 37, 32, 16
LENGTH_LIMIT, CURVATURE_LIMIT = 40.0, 1.5  # m, 1/m
COIL_CLEARANCE, PLASMA_CLEARANCE = 0.5, 0.5  # centreline distances, m
FIELD_RMS_LIMIT, IOTA_FLOOR, MIRROR_FLOOR = 0.02, 0.4, 0.05
SELF_LOCAL_ARC = 2 * COIL_CLEARANCE  # exclude only this physical arc neighborhood, m
PWO_OPTIONS = dict(mboz=10, nboz=10, ngamma=24, neta=24)
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAXITER, PREFIT_ITERATIONS, COIL_ORDER, N_SEGMENTS = 1, 2, 2, 24
    NPHI, NTHETA = 9, 12
    PWO_OPTIONS = dict(mboz=4, nboz=4, ngamma=12, neta=12)


def nonlocal_self_distances(gamma):
    """Pair distances outside a fixed physical arc neighborhood of each point.

    The shorter periodic arc distance uses the sampled polygon, so the
    exclusion refines with the curve instead of excluding a fixed index count.
    All symmetry copies are checked. A tiny squared-distance floor keeps the
    discarded diagonal's derivative finite; it is immaterial at metre scales.
    """
    segment = jnp.sqrt(jnp.sum((jnp.roll(gamma, -1, axis=1)-gamma)**2, axis=-1)+1e-24)
    arc = jnp.cumsum(segment, axis=1)-segment
    delta = jnp.abs(arc[:, :, None]-arc[:, None, :])
    shorter = jnp.minimum(delta, jnp.sum(segment, axis=1)[:, None, None]-delta)
    distance = jnp.sqrt(jnp.sum((gamma[:, :, None]-gamma[:, None, :])**2, axis=-1)+1e-24)
    return jnp.where(shorter > SELF_LOCAL_ARC, distance, jnp.inf)



def initial_coil_curves(inp):
    """Offset actual local plasma sections outward, then fit existing ESSOS curves."""
    boundary = surfacerzfourier_from_boundary(
        inp.rbc, inp.zbs, inp.nfp, ntheta=N_SEGMENTS, nphi=8*inp.nfp*N_COILS)
    orientation = float(np.sign(float(jnp.sum(boundary.gamma * boundary.normal))))
    if not np.isfinite(orientation) or orientation == 0:
        raise ValueError("Seed surface must enclose a finite nonzero signed volume")
    # Factory coil planes phi=(k+1/2)*pi/(nfp*N_COILS); apply symmetries once.
    sections = (boundary.gamma + 2*PLASMA_CLEARANCE*orientation*boundary.unitnormal)[2:4*N_COILS:4]
    dofs, _ = fit_dofs_from_coils(sections, COIL_ORDER, N_SEGMENTS, assume_uniform=True)
    return Curves(dofs, n_segments=N_SEGMENTS, nfp=inp.nfp, stellsym=True), orientation


def run():
    started = time.perf_counter()
    source = INPUT_FILE
    if ci_smoke and not source.exists():
        source = Path(__file__).resolve().parents[1] / "data/input.QI_nfp2_initial"
    inp = vj.VmecInput.from_file(source)
    if ci_smoke and source != INPUT_FILE:
        inp = replace(inp, ncurr=1, curtor=0.0, ac=np.zeros(21), pres_scale=0.0)
    if inp.ncurr != 1 or inp.curtor != 0:
        raise ValueError("The seed must prescribe zero enclosed toroidal current")
    inp = replace(inp, ns_array=np.array([NS]), ftol_array=np.array([1e-12]),
                  niter_array=np.array([12000]), delt=0.5)
    seed = opt.solve_equilibrium(inp, raise_on_max_iterations=True)
    if abs(float(seed.wout.betatotal) - TARGET_BETA) > max(1e-8, 0.1 * TARGET_BETA):
        raise ValueError("Seed beta does not match TARGET_BETA; run the corresponding stage-1 example")
    b_scale, r_scale = aries_cs_scales(seed.wout, target="volavgB")
    inp = vj.scale_input(inp, b_scale=b_scale, r_scale=r_scale)
    seed = opt.solve_equilibrium(inp, raise_on_max_iterations=True)
    field_ref, radius = float(seed.wout.volavgB), float(seed.wout.Rmajor_p)
    minor_ref = float(seed.wout.Aminor_p)
    aspect_ref = float(seed.wout.aspect)
    target = MappedPwOResidual([0.5], b_ref=field_ref, **PWO_OPTIONS)
    seed_sample = target.compute_state(seed.state, seed.runtime)
    local_mean_ref = float(jnp.mean(seed_sample["B_mapped"]))
    local_mirror_ref = float(jnp.min(seed_sample["mirror"]))

    def plasma_loss(state, runtime):
        sample = target.compute_state(state, runtime)
        mirror = jnp.min(sample["mirror"])
        limits = [jnp.maximum(IOTA_FLOOR - opt.min_abs_iota(state, runtime), 0) / IOTA_FLOOR,
                  (opt.aspect_ratio(state, runtime) / aspect_ref - 1) / 0.03,
                  jnp.maximum(MIRROR_FLOOR - mirror, 0) / MIRROR_FLOOR,
                  (mirror / local_mirror_ref - 1) / 0.10]
        limits.append((jnp.mean(sample["B_mapped"]) / local_mean_ref - 1) / 0.05)
        if TARGET_BETA:
            limits.append((opt.volume_average_beta(state, runtime) / TARGET_BETA - 1) / 0.1)
        return 10 * sample["total"] + jnp.sum(jnp.square(jnp.stack(limits)))

    problem = opt.VmecProblem.from_loss(inp, plasma_loss, max_mode=MAX_MODE,
                                       vary_major_radius=False, restart_from=seed,
                                       max_fsq_ratio=1.0, adjoint_tol=1e-7)
    curves, normal_orientation = initial_coil_curves(inp)
    current = field_ref * radius / (2e-7 * 2 * inp.nfp * N_COILS)
    coils0 = Coils(curves, np.full(N_COILS, current))
    data0 = vc.surface_field_data_from_state(inp, seed.state, runtime=seed.runtime,
                                            nphi=NPHI, ntheta=NTHETA, use_stellsym=False)
    precision = vc.plan_vc_precision(data0, digits=4) if TARGET_BETA else None
    plasma0 = (vc.plasma_field_on_boundary(data0, digits=4, precision=precision)
               if TARGET_BETA else jnp.zeros_like(data0.B_total))
    points0 = jnp.moveaxis(data0.gamma, 0, -1).reshape(-1, 3)
    field0 = jax.vmap(BiotSavart(coils0).B)(points0)
    desired = jnp.moveaxis(data0.B_total - plasma0, 0, -1).reshape(-1, 3)
    # Linear projection selects both the physical amplitude and winding sign.
    area0 = jnp.linalg.norm(data0.area_vector, axis=0).ravel()[:, None]
    current *= float(jnp.sum(area0 * field0 * desired) / jnp.sum(area0 * field0**2))
    if not np.isfinite(current) or current == 0:
        raise RuntimeError("Initial coils cannot initialize the seed field")
    coils0 = Coils(curves, np.full(N_COILS, current))
    nb = problem.x0.size
    x0 = jnp.concatenate((jnp.asarray(problem.x0), coils0.dofs))
    scales = jnp.concatenate((0.01 * problem.scales,
                              jnp.full(coils0.dofs_curves.size, 0.05), jnp.full(N_COILS, 0.1)))

    def objects(u, nphi=NPHI, ntheta=NTHETA):
        x = x0 + scales * u
        rbc, zbs = problem.boundary_from_x(x[:nb])
        surface = surfacerzfourier_from_boundary(rbc, zbs, inp.nfp, nphi=nphi, ntheta=ntheta)
        return surface, coils0.with_dofs(x[nb:])

    def surface_data(state, runtime, nphi=NPHI, ntheta=NTHETA):
        return vc.surface_field_data_from_state(inp, state, runtime=runtime,
                                                nphi=nphi, ntheta=ntheta, use_stellsym=False)

    def field_error(data, coils, plan, digits=4):
        plasma = (vc.plasma_field_on_boundary(data, digits=digits, precision=plan)
                  if TARGET_BETA else jnp.zeros_like(data.B_total))
        xyz = jnp.moveaxis(data.gamma, 0, -1)
        external = jax.vmap(BiotSavart(coils).B)(xyz.reshape(-1, 3)).reshape(xyz.shape)
        difference = jnp.moveaxis(external, -1, 0) + plasma - data.B_total
        area = jnp.linalg.norm(data.area_vector, axis=0)
        return jnp.sum(area * jnp.sum((difference / field_ref)**2, axis=0)) / jnp.sum(area)

    def engineering(surface, coils):
        length = jnp.maximum(coils.length / LENGTH_LIMIT - 1, 0)
        curvature = jnp.maximum(coils.curvature / CURVATURE_LIMIT - 1, 0)
        self_distance = nonlocal_self_distances(coils.gamma)
        self_penalty = jnp.maximum(1-self_distance/COIL_CLEARANCE, 0)**2
        return (jnp.mean(length**2) + jnp.mean(curvature**2) + jnp.max(self_penalty)
                + loss_coil_separation(coils, COIL_CLEARANCE, block_size=24)
                + loss_coil_surface_distance(coils, surface, PLASMA_CLEARANCE, block_size=24))

    @jax.jit
    def prefit_loss(u):
        surface, coils = objects(u)
        return 1000 * field_error(data0, coils, precision) + 100 * engineering(surface, coils)

    prefit_vg = jax.jit(jax.value_and_grad(prefit_loss))
    prefit = minimize(lambda u: tuple(map(np.asarray, prefit_vg(jnp.asarray(u)))),
                      np.zeros(x0.size), jac=True, method="L-BFGS-B",
                      bounds=[(0, 0)] * nb + [(-5, 5)] * (x0.size - nb),
                      options={"maxiter": PREFIT_ITERATIONS, "ftol": 1e-12})
    if not np.isfinite(prefit.fun) or not np.all(np.isfinite(prefit.jac)):
        raise RuntimeError("The coil prefit produced a nonfinite objective or gradient")

    def joint_loss(u):
        surface, coils = objects(u)
        def physical(state, runtime):
            return plasma_loss(state, runtime) + 1000 * field_error(surface_data(state, runtime), coils, precision)
        value, status = problem.jax_quantity_from_state((x0 + scales * u)[:nb], physical)
        return value + 100 * engineering(surface, coils), status

    joint_vg = jax.jit(jax.value_and_grad(joint_loss, has_aux=True))
    history, best = [], {"value": np.inf, "u": None}

    def value_and_gradient(u):
        (value, status), gradient = joint_vg(jnp.asarray(u))
        value, gradient = float(value), np.asarray(gradient)
        valid = int(status) == 0 and np.isfinite(value) and np.isfinite(gradient).all()
        history.append(dict(value=value if valid else None, status=int(status), valid=bool(valid)))
        if not valid:
            return np.inf, np.zeros_like(u)
        if value < best["value"]:
            best.update(value=value, u=np.array(u))
        print(f"trial {len(history)}: {value:.8e}", flush=True)
        return value, gradient

    initial = value_and_gradient(prefit.x)[0]
    if not np.isfinite(initial):
        raise RuntimeError("The initial joint equilibrium or derivative is invalid")
    result = minimize(value_and_gradient, prefit.x, jac=True, method="L-BFGS-B",
                      bounds=[(-3, 3)] * nb + [(-5, 5)] * (x0.size - nb),
                      options={"maxiter": MAXITER, "maxfun": 5 * MAXITER + 5, "ftol": 1e-12})
    u = best["u"]
    x = np.asarray(x0 + scales * u)
    coarse = problem.equilibrium_from_x(x[:nb])

    def refined(deck, source):
        modes = max(deck.mpol, 5 if ci_smoke else 7)
        deck = replace(deck, ns_array=np.array([16 if ci_smoke else 32]),
                       ftol_array=np.array([1e-12 if ci_smoke else 1e-13]),
                       niter_array=np.array([16000])).change_resolution(
                           mpol=modes, ntor=max(deck.ntor, modes),
                           ntheta=2*modes+10, nzeta=2*max(deck.ntor, modes)+8)
        return deck, opt.solve_equilibrium(deck, initial_state=restart_state(source.wout, deck),
                                           raise_on_max_iterations=True)

    deck, final = refined(problem.input_from_x(x[:nb]), coarse)
    _, reference = refined(inp, seed)
    fine = MappedPwOResidual([0.5], b_ref=field_ref, mboz=5 if ci_smoke else 14,
                             nboz=5 if ci_smoke else 14, oversample=2 if ci_smoke else 3,
                             ngamma=17 if ci_smoke else 39, neta=17 if ci_smoke else 39)
    seed_score = fine.compute_state(reference.state, reference.runtime)
    fine_mean_ref = float(jnp.mean(seed_score["B_mapped"]))

    def physics_metrics(state, runtime, wout, converged):
        score = fine.compute_state(state, runtime)
        metrics = dict(pwo=float(score["total"]), beta=float(wout.betatotal),
                       min_abs_iota=float(opt.min_abs_iota(state, runtime)),
                       aspect=float(wout.aspect), volavgB_T=float(wout.volavgB),
                       major_radius_m=float(wout.Rmajor_p), minor_radius_m=float(wout.Aminor_p),
                       local_mirror=float(jnp.min(score["mirror"])),
                       local_mean_T=float(jnp.mean(score["B_mapped"])),
                       actual_s=np.asarray(score["s_b"]).tolist())
        checks = dict(converged=bool(np.all(np.asarray(converged))),
                      valid_map=bool(np.all(np.asarray(score["valid"]))),
                      same_surface=np.array_equal(np.asarray(score["s_b"]), np.asarray(seed_score["s_b"])),
                      physical_objective_improved=metrics["pwo"] < float(seed_score["total"]),
                      iota=metrics["min_abs_iota"] >= IOTA_FLOOR,
                      local_mirror=metrics["local_mirror"] >= MIRROR_FLOOR,
                      local_mirror_held=abs(metrics["local_mirror"]/local_mirror_ref-1) <= 0.10,
                      local_field=abs(metrics["local_mean_T"]/fine_mean_ref-1) <= 0.05,
                      aspect=abs(metrics["aspect"]/aspect_ref-1) <= 0.03,
                      major_radius=abs(metrics["major_radius_m"]/radius-1) <= 0.03,
                      minor_radius=abs(metrics["minor_radius_m"]/minor_ref-1) <= 0.03,
                      field=abs(metrics["volavgB_T"]/field_ref-1) <= 0.05,
                      beta=abs(metrics["beta"]-TARGET_BETA) <= max(1e-8, 0.1*TARGET_BETA))
        return metrics, checks

    surface, coils = objects(u, nphi=61, ntheta=64)
    # Refine and shift coil quadrature before independent field/geometry checks.
    coils.curves.n_segments = 4 * N_SEGMENTS
    coils.curves.quadpoints += 0.5 / coils.curves.n_segments
    data = surface_data(final.state, final.runtime, 61, 64)
    plan = vc.plan_vc_precision(data, digits=5) if TARGET_BETA else None
    field_rms = float(jnp.sqrt(field_error(data, coils, plan, digits=5)))
    gamma, points = np.asarray(coils.gamma), np.asarray(surface.gamma).reshape(-1, 3)
    cc = min(float(np.linalg.norm(a[:, None] - b[None], axis=2).min())
             for i, a in enumerate(gamma) for b in gamma[i + 1:])
    cp = min(float(np.linalg.norm(a[:, None] - points[None], axis=2).min()) for a in gamma)
    metrics, checks = physics_metrics(final.state, final.runtime, final.wout, final.result.converged)
    metrics.update(field_vector_rms=field_rms, coil_clearance_m=cc, plasma_clearance_m=cp,
                   self_clearance_m=float(jnp.min(nonlocal_self_distances(coils.gamma))),
                   max_length_m=float(np.max(coils.length)), max_curvature_per_m=float(np.max(coils.curvature)),
                   min_curve_speed_m=float(jnp.min(jnp.linalg.norm(coils.gamma_dash, axis=-1))),
                   training_grid_pwo=float(target.total(final)), seed_fine_pwo=float(seed_score["total"]))
    checks.update(field_match=field_rms < FIELD_RMS_LIMIT, coil_clearance=cc >= COIL_CLEARANCE,
                  plasma_clearance=cp >= PLASMA_CLEARANCE, self_clearance=metrics["self_clearance_m"] >= COIL_CLEARANCE,
                  length=metrics["max_length_m"] <= LENGTH_LIMIT,
                  curvature=metrics["max_curvature_per_m"] <= CURVATURE_LIMIT,
                  regular_curves=metrics["min_curve_speed_m"] > 1e-8,
                  angular_refinement=abs(metrics["pwo"]-metrics["training_grid_pwo"])
                  <= 0.05*max(metrics["pwo"], 1e-12))
    deck.to_indata("input.single_stage_pwO")
    vj.write_wout("wout_single_stage_pwO.nc", final.wout)
    coils.to_json("coils_single_stage_pwO.json")
    summary = dict(initial=initial, final=best["value"], optimizer_status=int(result.status),
                   seconds=time.perf_counter() - started, metrics=metrics, checks=checks, trials=history,
                   scale_target="volavgB", local_mean_ref_T=local_mean_ref,
                   local_mirror_ref=local_mirror_ref, seed_fine_mirror=float(jnp.min(seed_score["mirror"])),
                   initial_coil_offset_m=2*PLASMA_CLEARANCE, initial_normal_orientation=normal_orientation,
                   self_local_arc_exclusion_m=SELF_LOCAL_ARC, coil_quadrature_shift=0.5,
                   coil_driven=dict(completed=False, reason="smoke budget" if ci_smoke else "not completed"),
                   geometric_proxies_only=True, research_validation_complete=False)
    failed_checks = ", ".join(name for name, passed in checks.items() if not passed)
    if not ci_smoke and failed_checks:
        summary["coil_driven"]["reason"] = f"Fixed-boundary validation failed: {failed_checks}"
    Path("single_stage_pwO_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if not ci_smoke:
        if failed_checks:
            raise SystemExit(summary["coil_driven"]["reason"])
        free_input = replace(deck, lfreeb=True, mgrid_file="direct ESSOS field")
        field = BiotSavart(coils)
        free = vj.solve_free_boundary_multigrid(
            free_input, external_field=field, raise_on_max_iterations=True)
        wout = vj.wout_from_state(inp=free_input, state=free.state, fsqr=free.fsqr, fsqz=free.fsqz,
                                  fsql=free.fsql, niter=free.iterations, converged=free.converged,
                                  vacuum_output=free.vacuum)
        vj.write_wout("wout_single_stage_pwO_coil_driven.nc", wout)
        context = im.runtime_from_params(im.params_from_input(free_input),
                  vj.make_free_boundary_config(free_input, field, ns=int(deck.ns_array[-1])).implicit)
        free_metrics, free_checks = physics_metrics(free.state, context, wout, free.converged)
        free_data = surface_data(free.state, context, 61, 64)
        free_points = np.moveaxis(np.asarray(free_data.gamma), 0, -1).reshape(-1, 3)
        free_metrics["plasma_clearance_m"] = min(
            float(np.linalg.norm(a[:, None]-free_points[None], axis=2).min()) for a in gamma)
        free_plan = vc.plan_vc_precision(free_data, digits=5) if TARGET_BETA else None
        free_metrics["field_vector_rms"] = float(jnp.sqrt(field_error(free_data, coils, free_plan, digits=5)))
        free_checks.update(objective_retained=free_metrics["pwo"] <= 1.05*metrics["pwo"]+1e-12,
                           plasma_clearance=free_metrics["plasma_clearance_m"] >= PLASMA_CLEARANCE,
                           field_match=free_metrics["field_vector_rms"] < FIELD_RMS_LIMIT)
        summary["coil_driven"] = dict(completed=True, metrics=free_metrics, checks=free_checks)
        Path("single_stage_pwO_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        if not all(checks.values()) or not all(free_checks.values()):
            raise SystemExit("The saved coil result misses refined plasma, geometry or coil-realization limits")
    return summary


if __name__ == "__main__":
    run()
