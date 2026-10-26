#!/usr/bin/env python
"""Fit canonical or mapped pwO targets, then rescore at higher resolution.

VMEX_PWO_SEED=near_circular runs QP conditioning before the fit; the default
uses the bundled QI seed. VMEX_PWO_FAMILY selects canonical or mapped.
Target calibration stays fixed; final validation requires local mirror within
10% of the seed. Scalar and least-squares solvers minimize the same residuals.
VMEX_PWO_METHOD=physical tries one bounded three-surface action descent step,
using raw slopes and fixed-reference action normalization.
VMEX_PWO_REFERENCE names an accepted summary to keep its grids and bounds fixed.
VMEX_PWO_FAMILY_CHARTS supplies JSON orbit charts to optimize family variance.
A dictionary selects run_centered_variance_step before seed preparation or QP
conditioning; a list retains the three-surface action-step route.
An entry's optional component=[parent,daughter1,daughter2] adds a supplied
binary junction's own-family variances and plateau-mean balance surrogates.
Neither target fitting nor sampled action descent establishes pwO or confinement.
"""

import json
import os
from dataclasses import replace
from pathlib import Path
import time

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares, linprog, minimize

import vmex as vj
from vmex import optimize as opt
from vmex.core.pwo import CanonicalPwOResidual, MappedPwOResidual, PwOActionFamilies, physical_pwo_action_from_boozer
from vmex.core.omnigenity import boozer_spectrum_state
from vmex.core.restart import restart_state

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SEED = os.environ.get("VMEX_PWO_SEED", "qi")
QS_INPUT = os.environ.get("VMEX_PWO_QS_INPUT")  # saved QP boundary skips preconditioning
SEED_PERTURBATION = 0.005  # each added Fourier coefficient is 5% of a=0.1 m
QS_MAX_MODES, QS_MAXITER = [1, 2, 3], [20, 20, 20]
QS_ASPECT_TARGET = 6.0
QS_IOTA_TARGET, QS_IOTA_TOLERANCE = 0.42, 0.02
MIRROR_TRAIN_TARGET, MIRROR_TRAIN_TOLERANCE = 0.34, 0.03  # margin for the independent gate
QS_PARAMETER_STEP, QS_PARAMETER_BOUND = 0.02, 5.0
FAMILY = os.environ.get("VMEX_PWO_FAMILY", "canonical")
METHOD = os.environ.get("VMEX_PWO_METHOD", "scalar")
PENALTY = os.environ.get("VMEX_PWO_PENALTY", "interval")
SURFACES = [0.5]
MAX_MODES, MAXITER = [1, 2, 3], [8, 8, 12]
STAGE_NS, FINAL_NS = 16, 32  # even ns puts s=0.5 on both half meshes
STAGE_FTOL, FINAL_FTOL = 1e-12, 1e-13
NITER = 12000
MPOL, FINAL_MPOL = 5, 7
PWO_OPTIONS = dict(mboz=10, nboz=10, oversample=2)
VALIDATION_OPTIONS = dict(mboz=14, nboz=14, oversample=3)
GRID, VALIDATION_GRID = 24, 39
IOTA_FLOOR, MIRROR_FLOOR, MIRROR_LIMIT, ELONGATION_LIMIT = 0.40, 0.05, 0.35, 8.0
ASPECT_TOLERANCE, SIZE_TOLERANCE, FIELD_TOLERANCE = 0.03, 0.03, 0.05
BETA_TOLERANCE = 0.10
MIRROR_RELATIVE_TOLERANCE = 0.10
PARAMETER_STEP, PARAMETER_BOUND = 0.01, 3.0
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    QS_MAX_MODES, QS_MAXITER = [1], [1]
    MAX_MODES, MAXITER = [1], [1]
    PWO_OPTIONS = dict(mboz=4, nboz=4, oversample=1)
    VALIDATION_OPTIONS = dict(mboz=5, nboz=5, oversample=2)
    GRID, VALIDATION_GRID = 12, 17
    FINAL_NS, FINAL_MPOL, FINAL_FTOL = 16, 5, 1e-12


if METHOD == "physical":
    STAGE_NS, FINAL_NS, MPOL, FINAL_MPOL = 48, 48, 7, 9
    STAGE_FTOL = FINAL_FTOL = 1e-13
    PWO_OPTIONS = dict(mboz=20, nboz=20, oversample=2)
    VALIDATION_OPTIONS = dict(mboz=20, nboz=20, oversample=3)


def make_target(surfaces, *, family, bmin, bmax, b_ref, options, grid, iota_sign=1.0, theta_center=0.0, zeta_center=0.0):
    """Build the fixed physical target shared by the stage-1 and coil examples."""
    if iota_sign not in (-1.0, 1.0):
        raise ValueError("iota_sign must be fixed at +1 or -1 from the seed")
    if family == "canonical":
        return CanonicalPwOResidual(surfaces, bmin=bmin, bmax=bmax, t1=-0.3 * iota_sign,
                                    t2=3.0 * iota_sign, theta_center=theta_center, zeta_center=zeta_center, p=2, ntheta=grid, nzeta=grid, **options)
    if family == "mapped":
        return MappedPwOResidual(surfaces, b_ref=b_ref, a=0.3, b=0.0, c=-2.0,
                                 eta_min=2.8, ngamma=grid, neta=grid, **options)
    raise ValueError("family must be canonical or mapped")


def canonical_target_bounds(unit_shape, mean, mirror):
    """Hold sampled target mean/mirror even when finite-p tails lift its minimum."""
    average = float(np.mean(unit_shape))
    low, high = float(np.min(unit_shape)), float(np.max(unit_shape))
    span = 2 * mirror * mean / (high - low - mirror * (high + low - 2 * average))
    bmin = mean - span * average
    return bmin, bmin + span


def canonical_seed_phase(field, target, nfp):
    """Choose once among the four stellarator-symmetric angular origins."""
    if any(size % 2 for size in target.shape[-2:]):
        raise ValueError("seed phase selection requires even angular grids")
    shifts = [(i * target.shape[-2] // 2, j * target.shape[-1] // 2) for i in range(2) for j in range(2)]
    candidates = [np.roll(target, shift, axis=(-2, -1)) for shift in shifts]
    costs = [float(np.mean(((field - candidate) / candidate)**2)) for candidate in candidates]
    chosen = int(np.argmin(costs))
    return float((chosen // 2) * np.pi), float((chosen % 2) * np.pi / nfp), costs


def hold_ratio(value, reference, tolerance, penalty):
    """Use the same physical tolerance for an equality or interval penalty."""
    deviation = value / reference - 1
    if penalty == "interval":
        deviation = jnp.maximum(jnp.abs(deviation) - tolerance, 0)
    return deviation / tolerance


def save_seed_checkpoint(deck, equilibrium, output_name, label):
    """Save each independently solved boundary before a later stage can fail."""
    deck.to_indata(f"input.{output_name}_{label}")
    vj.write_wout(f"wout_{output_name}_{label}.nc", equilibrium.wout)


def qp_gate_failures(metrics):
    """Check the refined QP boundary before pwO target calibration."""
    failed = []
    if not np.isfinite(metrics["qs_total"]):
        failed.append("QP objective finite")
    if (not np.isfinite(metrics["aspect"]) or
            abs(metrics["aspect"] / QS_ASPECT_TARGET - 1) > ASPECT_TOLERANCE):
        failed.append("aspect target")
    if not np.isfinite(metrics["min_abs_iota"]) or metrics["min_abs_iota"] < IOTA_FLOOR:
        failed.append("iota floor")
    if not MIRROR_FLOOR <= metrics["mirror"] <= MIRROR_LIMIT:
        failed.append("mirror interval")
    if not np.isfinite(metrics["elongation"]) or metrics["elongation"] > ELONGATION_LIMIT:
        failed.append("elongation limit")
    return failed


def component_terms(scored, component):
    """Four local surrogate terms for a supplied parent and two daughter charts."""
    families = scored["families"]
    if (not isinstance(component, (list, tuple)) or len(component) != 3 or
            any(type(i) is not int or not 0 <= i < len(families) for i in component) or
            len(set(component)) != 3):
        raise ValueError("Component requires three distinct parent/daughter chart indices")
    parent, first, second = [families[i] for i in component]
    p, a, b = [np.asarray(f["endpoints"]) for f in (parent, first, second)]
    if (p.shape != (2, 2) or a.shape != (2, 2) or b.shape != (2, 2) or
            not np.all(np.isfinite([p, a, b])) or
            max(np.max(abs(p[1]-a[0])), np.max(abs(a-b))) > 1e-8):
        raise ValueError("Component charts must share the parent/daughter event")
    rows1, rows2 = first["rows"], second["rows"]
    if any(not f["rows"] for f in (parent, first, second)) or any(
            not np.all(np.isfinite([r["J"], r["u"], r["left"], r["right"]])) or r["J"] <= 0 or
            r["right"] <= r["left"] for f in (parent, first, second) for r in f["rows"]):
        raise ValueError("Component samples require positive actions and ordered bounce pairs")
    if len(rows1) != len(rows2) or not rows1 or any(
            abs(a["u"]-b["u"]) > 1e-12 for a, b in zip(rows1, rows2)):
        raise ValueError("Component daughters require matching quadrature nodes")
    gaps = np.array([[b["left"]-a["right"], a["left"]-b["right"]]
                     for a, b in zip(rows1, rows2)])
    if not np.any(np.all(gaps > 0, axis=0)):
        raise ValueError("Component daughters must occupy distinct nonoverlapping wells")
    means = np.array([f["mean_action_m"] for f in (parent, first, second)])
    variances = np.array([f["variance_m2"] for f in (parent, first, second)])
    if not np.all(np.isfinite([means, variances])) or np.any(means <= 0) or np.any(variances < 0):
        raise ValueError("Component action statistics must be finite and physical")
    return [float((means[0]-means[1:].sum())**2), *variances.tolist()]


def family_gradient_rows(scored, baseline, entry):
    """Linearize statistics at baseline actions before differencing trial rows.

    Fixed chart weights and length retain each pitch's objective normalization.
    Finite differences of the action rows still have finite-step error.
    """
    variances = np.array([
        2*sum(r["weight"]*(r["J"]-before["mean_action_m"])*(s["J"]-r["J"])
              for r, s in zip(before["rows"], after["rows"], strict=True))
        for before, after in zip(baseline["families"], scored["families"], strict=True)])
    charts = entry["charts"]
    rows = []
    for pitch in scored["per_pitch"]:
        weights = np.array([c["weight"] if c["level_T"] == pitch["level_T"] else 0.
                            for c in charts])
        rows.append(float(weights@variances/weights.sum()/entry["length"]**2))
    if "component" in entry:
        indices = entry["component"]
        balances = [s["families"][indices[0]]["mean_action_m"]-
                    sum(s["families"][i]["mean_action_m"] for i in indices[1:])
                    for s in (baseline, scored)]
        rows.extend([2*balances[0]*(balances[1]-balances[0]), *variances[indices].tolist()])
    return rows


def run_physical_step(inp, equilibrium, refined, *, output_name, seed_route, reference=None, step_lengths=None,
                      family_charts=None):
    """Try up to three bounded steps using slopes or supplied family variance.

    family_charts is a JSON-compatible list of two independent sampling groups,
    each with three PwOActionFamilies constructor dictionaries at s=.5,.25,.75.
    Reference spectra identify the actual half-mesh surfaces. Restarts reuse
    these charts and their normalization; family discovery is caller-supplied.
    Require descent at each sampled pitch as a conservative numerical rule.
    An entry may supply component=[parent,daughter1,daughter2]. Its four local
    surrogate terms supplement every existing pitch and surface descent check.
    """
    started = time.perf_counter()
    names = ["RBC(1,1)", "ZBS(0,1)", "ZBS(-1,1)", "RBC(2,1)", "RBC(0,1)", "ZBS(1,1)", "ZBS(2,1)"]
    keys = ("raw_Jalpha2", "fixed_relative_Jalpha2")
    count = len(names)
    alpha = [np.sort(np.r_[2*np.pi*(np.arange(8)+.173)/8, 2*np.pi*(np.arange(8)+.673)/8]),
             2*np.pi*(np.arange(8)+.379)/8]
    fractions = [np.array([.25, .30, .335, .425, .49, .60, .67, .725, .775, .815, .85]),
                 np.array([.375, .55, .70, .90])]
    alpha = [[a]+[2*np.pi*(np.arange(8)+shift)/8]*2 for a, shift in zip(alpha, (.281, .487))]
    fractions = [[f]+[np.array(radial)]*2 for f, radial in
                 zip(fractions, ([.3, .5, .7, .9], [.35, .6, .8]))]
    search_calls, validation_calls = 0, 0
    summary = dict(method="physical", seed_route=seed_route, source=QS_INPUT or SEED,
                   parameters=names, accepted=False, research_validation_complete=False)

    def state(eq, options):
        spectra, bounds, surfaces, means, mirrors = [], [], [], [], []
        for surface in (.5, .25, .75):
            raw = boozer_spectrum_state(eq.state, eq.runtime, surfaces=[surface], **options)
            sp = {k: np.asarray(raw[k]) for k in
                  ("bmnc_b", "bmns_b", "xm_b", "xn_b", "iota_b", "G_b", "I_b", "nfp")}
            sp["nfp"] = int(sp["nfp"])
            t, z = np.meshgrid(2*np.pi*np.arange(193)/193,
                              2*np.pi*np.arange(193)/(193*sp["nfp"]), indexing="ij")
            B = sum(c*np.cos(m*t-n*z)+s*np.sin(m*t-n*z) for m, n, c, s in
                    zip(sp["xm_b"], sp["xn_b"], sp["bmnc_b"][0], sp["bmns_b"][0]))
            if not np.all(np.isfinite(B)) or B.min() <= 0:
                raise ValueError("Nonfinite or nonpositive field")
            spectra.append(sp); surfaces.append(float(raw["s_b"][0]))
            bounds.append((float(B.min()), float(B.max())))
            means.append(float(B.mean())); mirrors.append(float(np.ptp(B)/(B.max()+B.min())))
        metrics = dict(aspect=float(eq.wout.aspect), size=float(eq.wout.Aminor_p),
                       field=float(eq.wout.volavgB), local_field=means, local_mirror=mirrors,
                       surfaces=surfaces, iota_min=float(opt.min_abs_iota(eq.state, eq.runtime)),
                       iota=float(opt.mean_iota(eq.state, eq.runtime)),
                       mirror=float(opt.mirror_ratio(eq.state, eq.runtime)),
                       elongation=float(opt.max_elongation(eq.state, eq.runtime)))
        if not all(np.all(np.isfinite(v)) for v in metrics.values()):
            raise ValueError("Nonfinite physical metrics")
        return spectra, metrics, bounds

    def check(eq, metrics, baseline):
        limits = dict(aspect=ASPECT_TOLERANCE, size=SIZE_TOLERANCE,
                      field=FIELD_TOLERANCE, local_field=FIELD_TOLERANCE,
                      local_mirror=MIRROR_RELATIVE_TOLERANCE)
        checks = {key: np.all(abs(np.asarray(metrics[key])/baseline[key]-1) <= tol)
                  for key, tol in limits.items()}
        checks["radial_grid"] = metrics["surfaces"] == baseline["surfaces"]
        checks.update(converged=bool(np.all(eq.result.converged)),
                      iota=metrics["iota_min"] >= IOTA_FLOOR and
                      np.sign(baseline["iota"])*metrics["iota"] >= IOTA_FLOOR,
                      mirror=MIRROR_FLOOR <= metrics["mirror"] <= MIRROR_LIMIT,
                      local_mirror_floor=np.all(np.asarray(metrics["local_mirror"]) >= MIRROR_FLOOR),
                      elongation=metrics["elongation"] <= ELONGATION_LIMIT)
        if not all(checks.values()):
            raise ValueError("Physical checks failed: " + ", ".join(k for k, v in checks.items() if not v))

    def score(spectra, group, reference=None):
        scored = []
        for j, sp in enumerate(spectra):
            try:
                result = (objectives[group][j](dict(sp, s_b=[baseline["surfaces"][j]])) if objectives else
                    physical_pwo_action_from_boozer(**sp, levels=levels[group][j], alphas=alpha[group][j],
                    reference=None if reference is None else reference[j], check_roots=True))
                entry = family_charts[group][j] if objectives else {}
                if "component" in entry:
                    indices = entry["component"]
                    terms = component_terms(result, indices)
                    result.update(component_indices=indices, component_terms_m2=terms)
                scored.append(result)
            except (ValueError, RuntimeError) as error:
                raise ValueError(f"s={baseline['surfaces'][j]:.6g}, sampling group {group}: {error}") from error
        return scored

    def rows(scored):
        if objectives:
            return np.array([v for s in scored for v in [p["value"] for p in s["per_pitch"]]+s.get("component_terms_m2", [])])
        # Keep middle-surface pitches separate; each additional radius adds its own constraints.
        return np.array([p[k] for p in scored[0]["per_pitch"]+[s["components"] for s in scored[1:]] for k in keys])

    def improves(before, after):
        if objectives:
            return bool(np.all(rows(after) < rows(before)))
        def worst(s):
            return max(abs(r["dJ"]/r["J"]) for r in s["rows"])
        return all(all(c["components"][k] < b["components"][k] for k in keys)
                   and worst(c) <= worst(b) for b, c in zip(before, after))

    try:
        steps = np.asarray((.003, .001, .0003) if step_lengths is None else step_lengths, dtype=float)
        if (steps.ndim != 1 or not 1 <= len(steps) <= 3 or
                np.any(~np.isfinite(steps) | (steps <= 0)) or np.any(np.diff(steps) >= 0)):
            raise ValueError("Use one to three finite, positive, strictly decreasing step lengths")
        identity = dict(nfp=int(inp.nfp), phiedge=float(inp.phiedge), R00=float(inp.rbc[inp.ntor, 0]))
        if reference is not None and (not reference.get("accepted") or
                reference["physical_reference"]["identity"] != identity):
            raise ValueError("Reference must be an accepted step with the same NFP, flux and R00")
        if reference is not None:
            if family_charts is not None:
                raise ValueError("A restart must reuse its saved family charts")
            family_charts = reference["physical_reference"].get("family_charts")
        objectives = None
        if family_charts is not None:
            if len(family_charts) != 2 or any(len(g) != 3 for g in family_charts):
                raise ValueError("Family charts require two sampling groups on three surfaces")
            objectives = [[PwOActionFamilies(**{k: v for k, v in entry.items() if k != "component"})
                           for entry in group] for group in family_charts]
            for group, scorers in zip(family_charts, objectives):
                for entry, objective in zip(group, scorers):
                    if "component" in entry:
                        component_terms(objective.initial, entry["component"])
                        if len({entry["charts"][i]["level_T"] for i in entry["component"]}) != 1:
                            raise ValueError("Component charts must share one reflecting level")
            names[3] = "RBC(0,2)"
        summary["action_objective"] = "family_variance" if objectives else "fixed_alpha_slopes"
        if objectives and any("component" in e for g in family_charts for e in g):
            summary.update(action_objective="connected_component", component_scope=
                           "Supplied binary junction surrogates supplement all pitch aggregates. No pwO qualification.")
        problem = opt.VmecProblem.from_loss(
            inp, opt.aspect_ratio, max_mode=2, vary_major_radius=False, restart_from=equilibrium,
            max_fsq_ratio=1., refine_tol=1e-13, forward_ftol=1e-13, progress=False)
        indices = [list(problem.names).index(n) for n in names]
        x0 = problem.x0.copy()
        scale = PARAMETER_STEP*abs(float(inp.rbc[inp.ntor, 0]))
        search_calls += 1; base_eq = problem.equilibrium_from_x(x0)
        validation_calls += 1; _, reference_eq = refined(inp, base_eq)
        fine_base, fine_metrics, bounds = state(reference_eq, VALIDATION_OPTIONS)
        if reference is None:
            levels = [[lo+f*(hi-lo) for (lo, hi), f in zip(bounds, group)] for group in fractions]
            fixed = dict(identity=identity, metrics=fine_metrics,
                         levels_T=[[v.tolist() for v in g] for g in levels],
                         alpha=[[v.tolist() for v in g] for g in alpha])
            if objectives:
                fixed["family_charts"] = family_charts
        else:
            fixed = reference["physical_reference"]
            levels, alpha = fixed["levels_T"], fixed["alpha"]
            if any(len(g) != 3 for g in levels+alpha) or len(levels) != 2 or len(alpha) != 2:
                raise ValueError("Reference requires two sampling groups on three surfaces")
        baseline = fixed["metrics"]
        check(reference_eq, fine_metrics, baseline)
        before = [score(fine_base, i, None if reference is None else reference["refined_after"][i]) for i in range(2)]
        base_sp, base_metrics, _ = state(base_eq, PWO_OPTIONS)
        check(base_eq, base_metrics, baseline)
        base_scores = [score(base_sp, i, before[i]) for i in range(2)]
        normalization = np.asarray(rows(base_scores[0]) if reference is None else fixed["normalization"])
        if normalization.shape != rows(base_scores[0]).shape or np.any(~np.isfinite(normalization) | (normalization <= 0)):
            raise ValueError("Physical pitch normalization must be finite and positive")
        fixed["normalization"] = normalization.tolist()
        summary.update(physical_reference=fixed, baseline_metrics=base_metrics,
                       baseline_scores=base_scores, refined_before=before, scale_m=scale)

        def evaluate(y):
            nonlocal search_calls
            if search_calls >= 2*count+1+len(steps):
                raise RuntimeError("Search evaluation budget exhausted")
            x = x0.copy(); x[indices] += scale*y
            search_calls += 1; eq = problem.equilibrium_from_x(x)
            sp, metrics, _ = state(eq, PWO_OPTIONS); check(eq, metrics, baseline)
            return x, eq, sp, score(sp, 0, base_scores[0])

        h = 1e-4
        gradient = np.empty((len(normalization), count))
        for j in range(count):
            direction = np.eye(count)[j]
            minus, plus = [evaluate(sign*h*direction)[3] for sign in (-1, 1)]
            if objectives:
                linearized = [np.concatenate([family_gradient_rows(s, b, e) for s, b, e in
                              zip(scored, base_scores[0], family_charts[0], strict=True)])
                              for scored in (minus, plus)]
                gradient[:, j] = (linearized[1]-linearized[0])/(2*h*normalization)
            else:
                gradient[:, j] = (rows(plus)-rows(minus))/(2*h*normalization)
        norms = np.linalg.norm(gradient, axis=1)
        if np.any(~np.isfinite(norms) | (norms == 0)):
            raise ValueError("Unresolved physical gradient")
        a = gradient/norms[:, None]
        result = linprog([0.]*count+[-1.], A_ub=np.c_[a, np.ones(len(a))],
                         b_ub=np.zeros(len(a)), bounds=[(-1., 1.)]*count+[(None, None)], method="highs")
        if not result.success or result.x[-1] <= 1e-6:
            raise ValueError("No common descent in the selected boundary coefficients")
        direction = result.x[:count]/np.linalg.norm(result.x[:count])
        summary.update(direction=direction.tolist(), trials=[])
        for step_length in steps:
            trial = dict(step_length=float(step_length), accepted=False)
            summary["trials"].append(trial)
            try:
                x, candidate, sp, train = evaluate(step_length*direction)
                trial_scores = [train, score(sp, 1, base_scores[1])]
                trial["scores"] = trial_scores
                if not np.all(rows(train) < rows(base_scores[0])) or not all(
                        improves(b, c) for b, c in zip(base_scores, trial_scores)):
                    raise ValueError("Candidate failed physical action descent checks")
                validation_calls += 1; final_input, final_eq = refined(problem.input_from_x(x), candidate)
                fine_sp, final_metrics, _ = state(final_eq, VALIDATION_OPTIONS)
                check(final_eq, final_metrics, baseline)
                after = [score(fine_sp, i, before[i]) for i in range(2)]
                trial.update(refined_after=after, final_metrics=final_metrics)
                if not np.all(rows(after[0]) < rows(before[0])) or not all(
                        improves(b, c) for b, c in zip(before, after)):
                    raise ValueError("Candidate failed matched-refinement descent checks")
            except (ValueError, RuntimeError) as error:
                trial["rejection"] = str(error)
                continue
            trial["accepted"] = True
            summary.update(step_length=float(step_length), trial_scores=trial_scores,
                           refined_after=after, final_metrics=final_metrics)
            break
        else:
            raise ValueError("No step passed physical action and refinement checks")
        summary["accepted"] = True
        final_input.to_indata(f"input.{output_name}")
        vj.write_wout(f"wout_{output_name}.nc", final_eq.wout)
        vj.write_wout(f"wout_{output_name}_seed.nc", reference_eq.wout)
    except (ValueError, RuntimeError, KeyError, TypeError) as error:
        summary["rejection"] = str(error)
    summary.update(search_evaluations=search_calls, refinement_evaluations=validation_calls,
                   seconds=time.perf_counter()-started)
    summary_path = Path(f"{output_name}_summary.json")
    if not summary["accepted"] and summary_path.exists():
        summary_path = Path(f"{output_name}_rejected_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(f"Physical step accepted: {summary['accepted']}; {summary.get('rejection', '')}", flush=True)
    if not summary["accepted"] and not ci_smoke:
        raise SystemExit(1)
    return summary


def run_centered_variance_step(policy, *, output_name="pwO_optimized"):
    """Take one matched-seed, centered-action variance step on supplied families.

    ``policy`` supplies input/restart paths, positive ``fixed_B_T`` (T),
    and training/validation entries containing PwOActionFamilies constructor
    arguments (reference, charts, length), plus optional transform dictionaries.
    References may be dictionaries or NPZ paths; each needs explicit single s_b.
    Charts retain IDs, endpoints, reflecting levels, normalized weights and
    length across resolutions; validation may refine their nodes. Optional parent_families
    IDs protect those variances when guarded=True also protects aggregate Q.
    Scales=.02 m, half_step=1e-4 and trust_radius=.03 are discretionary local
    defaults. All eight max_mode=1 coefficients vary with fixed R00 and flux.
    The deck's grid/FTOL are retained; ftol_check defaults to 100*deck FTOL,
    a paired numerical sensitivity check, not an accuracy guarantee. Every
    solve restarts independently from the original restart, never a trial.
    The finite screen sums per-state observed tolerance/band/node changes,
    requiring a resolved V decrease and no resolved protected-metric increases.
    This empirical envelope is not a rigorous error bound. Only sampled local
    progress is reported; no pwO qualification follows.
    """
    started = time.perf_counter()
    inp = vj.VmecInput.from_file(Path(policy["input"]))
    seed = Path(policy["restart"])
    field = float(policy["fixed_B_T"])
    h, radius = float(policy.get("half_step", 1e-4)), float(policy.get("trust_radius", .03))
    scale = np.broadcast_to(np.asarray(policy.get("scales", .02), float), (8,)).copy()
    ftol = float(np.asarray(inp.ftol_array)[-1])
    check_ftol = float(policy.get("ftol_check", 100*ftol))
    if not np.all(np.isfinite([field, h, radius, ftol, check_ftol])) or min(field, h, radius, ftol, check_ftol) <= 0:
        raise ValueError("Positive finite fixed_B_T, FD/trust scales and FTOLs required")
    names = list(opt.boundary_dof_names(inp, 1, vary_major_radius=False))
    if len(names) != 8 or np.any(~np.isfinite(scale) | (scale <= 0)) or check_ftol <= ftol:
        raise ValueError("Require eight symmetric boundary modes, positive scales and a looser paired FTOL check")
    x0 = opt.pack_boundary(inp, 1, vary_major_radius=False)
    scorers, transforms = [], []
    for key, band in (("training", 10), ("validation", 20)):
        entry = dict(policy[key]); reference = entry.pop("reference")
        transforms.append(entry.pop("transform", dict(mboz=band, nboz=band, oversample=3)))
        if isinstance(reference, (str, Path)):
            with np.load(reference, allow_pickle=False) as archive:
                reference = {k: archive[k].copy() for k in archive.files}
        reference = {k: np.asarray(v) for k, v in reference.items()}
        reference["nfp"] = np.asarray(reference["nfp"]).item()
        if np.shape(reference.get("s_b")) != (1,) or not np.all(np.isfinite(reference["s_b"])):
            raise ValueError("Each reference requires one explicit finite half-mesh surface s_b")
        scorers.append(PwOActionFamilies(reference=reference, **entry))
    first, second = scorers
    def identities(scorer):
        return [(c["id"], c["level_T"]) for c in scorer.charts]
    if (identities(first) != identities(second) or len(set(c["id"] for c in first.charts)) != len(first.charts)
            or any(not np.array_equal(a["endpoints"], b["endpoints"])
                   for a, b in zip(first.charts, second.charts, strict=True))
            or not np.array_equal(first.weights, second.weights) or first.length != second.length
            or not np.array_equal(first.reference["s_b"], second.reference["s_b"])
            or int(first.reference["nfp"]) != int(inp.nfp) or int(second.reference["nfp"]) != int(inp.nfp)):
        raise ValueError("Training/validation require the same ordered IDs, endpoints, levels, weights, length, surface and NFP")
    ids = [c["id"] for c in first.charts]
    protected = list(policy.get("parent_families", []))
    if any(fid not in ids for fid in protected):
        raise ValueError("Unknown protected family ID")
    guarded = policy.get("guarded", True)
    if not isinstance(guarded, bool) or any(not isinstance(fid, str) or not fid for fid in ids):
        raise ValueError("guarded must be boolean and chart IDs must be nonempty strings")
    summary = dict(method="centered_family_variance", accepted=False, pwO_qualified=False,
        research_validation_complete=False, policy=policy, parameters=names, cases={}, trials=[], MHD_solves=0,
        scope="Supplied families and pitches only; bounded local progress, no global or confinement claim.")

    def save_summary():
        summary["seconds"] = time.perf_counter()-started
        Path(f"{output_name}_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False,
            default=lambda value: value.tolist() if isinstance(value, np.ndarray) else str(value))+"\n")

    def rows(score, scorer):
        v, q, blocks = [], [], []
        for family, chart, weight in zip(score["families"], scorer.charts, scorer.weights, strict=True):
            data = family["rows"]
            u, w, J, dJ = (np.asarray([r[k] for r in data]) for k in ("u", "weight", "J", "dJ"))
            if (not np.array_equal(u, [r["u"] for r in chart["rows"]]) or
                    not np.array_equal(w, [r["weight"] for r in chart["rows"]])):
                raise ValueError("Family quadrature identity changed")
            start = len(v); v.extend(np.sqrt(weight*w)*(J-w@J)/scorer.length)
            q.extend(np.sqrt(weight*w)*dJ); blocks.append(slice(start, len(v)))
        v, q = np.asarray(v), np.asarray(q)
        if (not np.all(np.isfinite([*v, *q])) or
                not np.isclose(v@v, score["value"], rtol=2e-10, atol=1e-15) or
                not np.isclose(q@q, score["raw_Jalpha2"], rtol=2e-10, atol=1e-15)):
            raise ValueError("Centered action residual does not reproduce aggregate variance")
        return v, q, blocks

    def score(eq, index, band_index=None):
        scorer = scorers[index]
        raw = boozer_spectrum_state(eq.state, eq.runtime,
            surfaces=scorer.reference["s_b"].tolist(), **transforms[index if band_index is None else band_index])
        spectrum = {k: np.asarray(raw[k]) for k in
            ("bmnc_b", "bmns_b", "xm_b", "xn_b", "iota_b", "G_b", "I_b", "nfp", "s_b")}
        spectrum["nfp"] = np.asarray(spectrum["nfp"]).item()
        if any(not np.array_equal(spectrum[k], scorer.reference[k]) for k in ("s_b", "nfp")):
            raise ValueError("Boozer surface or field period changed")
        factor = field/float(eq.wout.volavgB)
        for key in ("bmnc_b", "bmns_b", "G_b", "I_b"):
            spectrum[key] = spectrum[key]*factor
        result = scorer(spectrum); rows(result, scorer)
        return result

    def solve(displacement, label, tolerance=ftol):
        deck = opt.unpack_boundary(inp, x0+scale*displacement, 1, vary_major_radius=False)
        if tolerance != ftol:
            ladder = np.asarray(inp.ftol_array).copy(); ladder[-1] = tolerance
            deck = replace(deck, ftol_array=ladder)
        summary["MHD_solves"] += 1
        summary["cases"][label] = dict(status="solving", requested_FTOL=tolerance)
        save_summary()
        eq = opt.solve_equilibrium(deck, initial_state=restart_state(seed, deck), raise_on_max_iterations=True)
        stored = float(np.asarray(eq.inp.ftol_array)[-1])
        errors = [float(getattr(eq.result, k)) for k in ("fsqr", "fsqz", "fsql")]
        if (stored != tolerance or not bool(np.all(eq.result.converged)) or
                any(not np.isfinite(e) or not 0 <= e <= stored for e in errors)):
            raise ValueError(f"{label}: failed convergence at actual stored FTOL")
        amplitude = float(eq.wout.volavgB)
        if not np.isfinite(amplitude) or amplitude <= 0:
            raise ValueError("Invalid physical volume-average field")
        input_path, wout_path = f"input.{output_name}_{label}", f"wout_{output_name}_{label}.nc"
        # Keep converged checkpoints even when subsequent chart continuation fails.
        deck.to_indata(input_path); vj.write_wout(wout_path, eq.wout)
        summary["cases"][label].update(status="converged", stored_FTOL=stored, force_errors=errors,
            physical_volavgB_T=amplitude, normalization_factor=field/amplitude,
            input_path=input_path, wout_path=wout_path)
        save_summary()
        result = score(eq, 0); summary["cases"][label]["training"] = result; save_summary()
        print(f"Centered variance {label}: V={result['value']:.8g}", flush=True)
        return eq, result

    def resolution(eq, training):
        return {"0_0": training, **{f"{node}_{band}": score(eq, node, band)
            for node, band in ((0, 1), (1, 0), (1, 1))}}

    def assessment(before, before_check, after, after_check):
        def metrics(score):
            return dict(V=float(score["value"]), Q=float(score["raw_Jalpha2"]),
                **{fid+"_V": float(f["variance_m2"])/second.length**2
                   for fid, f in zip(ids, score["families"], strict=True)})
        grids = [{k: metrics(v) for k, v in grid.items()} for grid in
                 (before, before_check, after, after_check)]
        report = {}
        for key in grids[0]["1_1"]:
            values = [grid["1_1"][key] for grid in grids]
            state_changes = []
            for grid, check in ((grids[0], grids[1]), (grids[2], grids[3])):
                changes = dict(equilibrium_tolerance=abs(grid["1_1"][key]-check["1_1"][key]),
                    spectral_band=max(abs(g[f"{n}_1"][key]-g[f"{n}_0"][key])
                                      for g in (grid, check) for n in (0, 1)),
                    quadrature_nodes=max(abs(g[f"1_{b}"][key]-g[f"0_{b}"][key])
                                         for g in (grid, check) for b in (0, 1)))
                changes["total"] = sum(changes.values()); state_changes.append(changes)
            gain = values[0]-values[2]
            floor = 128*np.finfo(float).eps*max(1., *[abs(v[key]) for g in grids for v in g.values()])
            envelope = sum(c["total"] for c in state_changes)+floor
            classification = ("decrease_beyond_observed_changes" if gain > envelope else
                "increase_beyond_observed_changes" if gain < -envelope else "unresolved_observed_changes")
            report[key] = dict(center_finest=values[0], trial_finest=values[2], gain=gain,
                change_percent=None if values[0] == 0 else 100*(values[2]/values[0]-1),
                state_changes=state_changes, empirical_total_change=envelope, roundoff_floor=floor,
                paired_equilibrium_gain_change=abs(gain-(values[1]-values[3])),
                measured_nonincrease=gain >= 0, classification=classification)
        checks = dict(V=report["V"]["classification"] == "decrease_beyond_observed_changes")
        if guarded:
            checks.update({key: report[key]["classification"] != "increase_beyond_observed_changes"
                           for key in ["Q", *[fid+"_V" for fid in protected]]})
        return report, checks

    try:
        seed = vj.read_wout(seed)
        center_eq, center = solve(np.zeros(8), "center")
        v0, q0, blocks = rows(center, first)
        A, C, amplitude = [], [], []
        for j in range(8):
            delta = np.eye(8)[j]*h
            minus_eq, minus = solve(-delta, f"fd{j}-minus")
            plus_eq, plus = solve(delta, f"fd{j}-plus")
            vm, qm, _ = rows(minus, first); vp, qp, _ = rows(plus, first)
            A.append((vp-vm)/(2*h)); C.append((qp-qm)/(2*h))
            amplitude.append((float(plus_eq.wout.volavgB)-float(minus_eq.wout.volavgB))/(2*h*field))
        A, C, amplitude = np.asarray(A).T, np.asarray(C).T, np.asarray(amplitude)
        if (not all(np.all(np.isfinite(a)) for a in (A, C, amplitude)) or
                np.linalg.norm(amplitude) <= 1e-12 or v0@v0 <= 0):
            raise ValueError("Unresolved physical amplitude derivative or zero center variance")
        N = np.linalg.svd(amplitude[None, :], full_matrices=True)[2][1:].T
        D, E = radius*A@N, radius*C@N
        guards = [(q0, E)] + [(v0[b], D[b]) for fid, b in zip(ids, blocks) if fid in protected]
        if not guarded:
            guards = []
        def constraints(z):
            return np.asarray([1-z@z] + [float(r@r-(r+M@z)@(r+M@z))/max(float(r@r), 1e-30) for r, M in guards])
        def constraint_jac(z):
            return np.asarray([-2*z] + [-2*M.T@(r+M@z)/max(float(r@r), 1e-30) for r, M in guards])
        fit = minimize(lambda z: float((v0+D@z)@(v0+D@z)/(v0@v0)), np.zeros(7),
            jac=lambda z: 2*D.T@(v0+D@z)/(v0@v0), method="SLSQP",
            constraints=[dict(type="ineq", fun=constraints, jac=constraint_jac)], options=dict(maxiter=1000, ftol=1e-10))
        z = np.asarray(fit.x)
        if not np.all(np.isfinite(z)):
            raise ValueError("Nonfinite Gauss-Newton proposal")
        z *= min(1., (1-16*np.finfo(float).eps)/max(np.linalg.norm(z), 1e-30))
        step = radius*N@z
        summary["model"] = dict(centered_action_residual=v0.tolist(), centered_action_jacobian=A.tolist(),
            slope_jacobian=C.tolist(), amplitude_gradient=amplitude.tolist(), proposal=step.tolist(),
            predicted_V=float((v0+A@step)@(v0+A@step)), guarded=guarded, solver_success=bool(fit.success),
            solver_status=int(fit.status), solver_message=str(fit.message), solver_iterations=int(fit.nit),
            center_V=float(v0@v0), center_Q=float(q0@q0), x0=x0.tolist(), scales=scale.tolist(),
            half_step=h, trust_radius=radius, step_norm=float(np.linalg.norm(step)),
            amplitude_linear_relative_change=float(amplitude@step))
        if not fit.success or np.min(constraints(z)) < -2e-8 or summary["model"]["predicted_V"] >= float(v0@v0):
            raise ValueError("No feasible descending bounded Gauss-Newton proposal")
        center_grid = resolution(center_eq, center)
        check_eq, check_center = solve(np.zeros(8), "check-center", check_ftol)
        center_check_grid = resolution(check_eq, check_center)
        summary["center"] = dict(resolution=center_grid, paired_resolution=center_check_grid)
        for fraction in (1., .5):
            label = "full" if fraction == 1 else "half"
            trial = dict(fraction=fraction, accepted=False); summary["trials"].append(trial)
            try:
                trial_eq, after = solve(fraction*step, label)
                grid = resolution(trial_eq, after)
                paired_eq, paired = solve(fraction*step, "check-"+label, check_ftol)
                paired_grid = resolution(paired_eq, paired)
                report, checks = assessment(center_grid, center_check_grid, grid, paired_grid)
                trial.update(resolution=grid, paired_resolution=paired_grid,
                    assessment=report, checks=checks, accepted=all(checks.values()))
                if trial["accepted"]:
                    summary.update(accepted=True, accepted_fraction=fraction)
                    trial_eq.inp.to_indata(f"input.{output_name}")
                    vj.write_wout(f"wout_{output_name}.nc", trial_eq.wout)
                    break
            except (ValueError, RuntimeError) as error:
                trial["rejection"] = str(error)
    except (ValueError, RuntimeError) as error:
        summary["rejection"] = str(error)
    summary["finite_screen_pass"] = summary["accepted"]
    save_summary()
    print(f"Sampled local variance progress accepted: {summary['accepted']}; no pwO qualification", flush=True)
    return summary


def run(*, beta_targets=(0.0,), output_name="pwO_optimized", physical_steps=None):
    """Optimize with fixed flux, R00, linear edge-zero pressure and zero current.

    beta_targets is a pressure-continuation ladder in volume-average beta.
    physical_steps optionally sets up to three decreasing trial lengths in physical mode.
    Pressure is calibrated from a unit-pressure seed solve and held fixed per
    stage. Current is prescribed, not bootstrap-consistent. Derivatives hold
    the solver's released m=1 gauge fixed.
    """
    chart_path = os.environ.get("VMEX_PWO_FAMILY_CHARTS")
    family_charts = None if chart_path is None or METHOD != "physical" else json.loads(Path(chart_path).read_text())
    if METHOD == "physical" and isinstance(family_charts, dict):
        if tuple(beta_targets) != (0.0,) or physical_steps is not None:
            raise ValueError("Centered variance policy requires vacuum beta_targets and its own trust bound")
        summary = run_centered_variance_step(family_charts, output_name=output_name)
        if not summary["accepted"] and not ci_smoke:
            raise SystemExit(1)
        return summary
    if (FAMILY not in ("canonical", "mapped") or METHOD not in ("scalar", "least_squares", "physical")
            or PENALTY not in ("equality", "interval")):
        raise ValueError("require canonical|mapped family, scalar|least_squares|physical method, and equality|interval penalty")
    beta_targets = tuple(float(beta) for beta in beta_targets)
    if not beta_targets or beta_targets[0] != 0 or any(
            not np.isfinite(beta) or beta < 0 for beta in beta_targets) or any(
                right < left for left, right in zip(beta_targets, beta_targets[1:])):
        raise ValueError("beta_targets must start at vacuum and increase through finite nonnegative values")
    if METHOD == "physical" and beta_targets != (0.0,):
        raise ValueError("The bounded physical-action step currently requires a vacuum seed")
    if SEED not in ("qi", "near_circular"):
        raise ValueError("VMEX_PWO_SEED must be qi or near_circular")
    seed_route = "saved_input" if QS_INPUT else SEED
    print(f"Starting {seed_route} -> {FAMILY} pwO ({METHOD}); beta ladder {beta_targets}", flush=True)
    started = time.perf_counter()
    am = np.zeros(21); am[:2] = [1.0, -1.0]

    def prepare(deck):
        return replace(
            deck, pres_scale=0.0, pmass_type="power_series", am=am,
            ncurr=1, curtor=0.0, pcurr_type="power_series", ac=np.zeros(21), delt=0.5,
            ns_array=np.array([STAGE_NS]), ftol_array=np.array([STAGE_FTOL]),
            niter_array=np.array([NITER])).change_resolution(
                mpol=MPOL, ntor=MPOL, ntheta=2 * MPOL + 6, nzeta=2 * MPOL + 4)

    def refined(deck, source):
        deck = replace(deck, ns_array=np.array([FINAL_NS]), ftol_array=np.array([FINAL_FTOL]),
                       niter_array=np.array([NITER])).change_resolution(
                           mpol=FINAL_MPOL, ntor=FINAL_MPOL,
                           ntheta=2 * FINAL_MPOL + 10, nzeta=2 * FINAL_MPOL + 8)
        eq = opt.solve_equilibrium(deck, initial_state=restart_state(source.wout, deck),
                                  raise_on_max_iterations=True, verbose=not ci_smoke)
        return deck, eq

    qs_stages, qs_metrics = [], None
    if SEED == "qi" and not QS_INPUT:
        seed_input = prepare(vj.VmecInput.from_file(DATA_DIR / "input.QI_nfp2_initial"))
        equilibrium = opt.solve_equilibrium(seed_input, raise_on_max_iterations=True)
    else:
        near_circular_input = vj.VmecInput.from_file(DATA_DIR / "input.minimal_seed_nfp2")
        rbc, zbs = near_circular_input.rbc.copy(), near_circular_input.zbs.copy()
        rbc[near_circular_input.ntor - 1, 1] = -SEED_PERTURBATION
        zbs[near_circular_input.ntor - 1, 1] = SEED_PERTURBATION
        near_circular_input = replace(near_circular_input, rbc=rbc, zbs=zbs)
        near_circular_input = prepare(near_circular_input)
        qs = opt.QuasisymmetryRatioResidual([0.5, 0.7, 0.9], helicity_m=0, helicity_n=1)
        qs_raw_metrics = None

        def qp_metrics(eq):
            return dict(qs_total=float(qs.total_state(eq.state, eq.runtime)),
                        aspect=float(opt.aspect_ratio(eq.state, eq.runtime)),
                        min_abs_iota=float(opt.min_abs_iota(eq.state, eq.runtime)),
                        mirror=float(opt.mirror_ratio(eq.state, eq.runtime)),
                        elongation=float(opt.max_elongation(eq.state, eq.runtime)),
                        minor_radius_m=float(eq.wout.Aminor_p), volavgB_T=float(eq.wout.volavgB))

        def save_qs_summary(training_metrics=None, refined_metrics=None):
            unmet = None if refined_metrics is None else qp_gate_failures(refined_metrics)
            status = "running" if unmet is None else ("failed" if unmet else "passed")
            Path(f"{output_name}_qs_summary.json").write_text(json.dumps(
                dict(source=QS_INPUT or "near_circular", perturbation_m=SEED_PERTURBATION,
                     helicity_m=0, helicity_n=1, stages=qs_stages,
                     settings=dict(aspect_target=QS_ASPECT_TARGET, iota_target=QS_IOTA_TARGET,
                                   mirror_target=MIRROR_TRAIN_TARGET, iota_tolerance=QS_IOTA_TOLERANCE,
                                   mirror_tolerance=MIRROR_TRAIN_TOLERANCE,
                                   parameter_step=QS_PARAMETER_STEP, parameter_bound=QS_PARAMETER_BOUND),
                     raw_metrics=qs_raw_metrics, training_metrics=training_metrics,
                     metrics=refined_metrics, status=status, unmet=unmet), indent=2, allow_nan=False) + "\n")

        if QS_INPUT:
            seed_input = prepare(vj.VmecInput.from_file(Path(QS_INPUT)))
            if (seed_input.nfp != near_circular_input.nfp or
                    not np.isclose(seed_input.phiedge, near_circular_input.phiedge) or
                    not np.isclose(seed_input.rbc[seed_input.ntor, 0],
                                   near_circular_input.rbc[near_circular_input.ntor, 0])):
                raise ValueError("QP input must retain nfp, toroidal flux and major radius")
            equilibrium = opt.solve_equilibrium(seed_input, raise_on_max_iterations=True)
        else:
            seed_input = near_circular_input
            equilibrium = opt.solve_equilibrium(seed_input, raise_on_max_iterations=True)
            save_seed_checkpoint(seed_input, equilibrium, output_name, "near_circular")
            qs_raw_metrics = qp_metrics(equilibrium)
            save_qs_summary()

            def qs_constraints(state, runtime):
                mirror = opt.mirror_ratio(state, runtime)
                return jnp.stack([
                    (opt.aspect_ratio(state, runtime) / QS_ASPECT_TARGET - 1) / ASPECT_TOLERANCE,
                    jnp.maximum(QS_IOTA_TARGET - opt.min_abs_iota(state, runtime), 0) / QS_IOTA_TOLERANCE,
                    jnp.maximum(MIRROR_FLOOR - mirror, 0) / MIRROR_TRAIN_TOLERANCE,
                    jnp.maximum(mirror - MIRROR_TRAIN_TARGET, 0) / MIRROR_TRAIN_TOLERANCE,
                    jnp.maximum(opt.max_elongation(state, runtime) - ELONGATION_LIMIT, 0)
                    / ELONGATION_LIMIT])

            qs_terms = [(qs, 0.0, 1.0), (qs_constraints, 0.0, 10.0)]

            def qs_loss(state, runtime):
                rows = opt.residuals_from_tuples(state, runtime, qs_terms)
                return 0.5 * jnp.vdot(rows, rows)

            qs_problem = opt.VmecProblem.from_loss(
                seed_input, qs_loss, max_mode=max(QS_MAX_MODES), vary_major_radius=False,
                restart_from=equilibrium, max_fsq_ratio=1.0, adjoint_tol=1e-8,
                progress=not ci_smoke, evaluation_progress=not ci_smoke, jacobian_batch_size=1)
            full_x = qs_problem.x0.copy()
            for max_mode, budget in zip(QS_MAX_MODES, QS_MAXITER):
                stage = qs_problem.subproblem(max_mode=max_mode, x=full_x)
                x0, step = stage.x0, QS_PARAMETER_STEP * stage.scales

                def qs_value_and_gradient(y):
                    value, gradient = stage.value_and_grad(x0 + step * y)
                    return value, step * gradient

                def qs_record(intermediate_result):
                    print(f"QP mode {max_mode}: accepted cost {intermediate_result.fun:.6e}", flush=True)

                initial = float(qs_value_and_gradient(np.zeros_like(x0))[0])
                result = minimize(
                    qs_value_and_gradient, np.zeros_like(x0), jac=True, method="L-BFGS-B",
                    bounds=[(-QS_PARAMETER_BOUND, QS_PARAMETER_BOUND)] * x0.size, callback=qs_record,
                    options={"maxiter": budget, "maxfun": 5 * budget + 5, "maxls": 15,
                             "ftol": 1e-12, "gtol": 1e-7})
                x = x0 + step * result.x
                seed_input = stage.input_from_x(x)
                equilibrium = stage.equilibrium_from_x(x)
                full_x = qs_problem.x_from_input(seed_input)
                qs_stages.append(dict(mode=max_mode, initial_cost=initial,
                                      final_cost=float(result.fun), status=int(result.status),
                                      message=str(result.message)))
                save_seed_checkpoint(seed_input, equilibrium, output_name, "qs")
                save_qs_summary(training_metrics=qp_metrics(equilibrium))
                print(f"QP mode {max_mode}: cost {initial:.6e} -> {result.fun:.6e}", flush=True)

        qs_training_metrics = qp_metrics(equilibrium)
        qs_refined_input, qs_refined_equilibrium = refined(seed_input, equilibrium)
        save_seed_checkpoint(qs_refined_input, qs_refined_equilibrium, output_name, "qs_refined")
        qs_metrics = qp_metrics(qs_refined_equilibrium)
        save_qs_summary(qs_training_metrics, qs_metrics)
        qs_failed = qp_gate_failures(qs_metrics)
        if qs_failed:
            raise RuntimeError("QP preconditioning failed " + ", ".join(qs_failed) + "; checkpoint saved")
    if METHOD == "physical":
        reference_path = os.environ.get("VMEX_PWO_REFERENCE")
        reference = None if reference_path is None else json.loads(Path(reference_path).read_text())
        return run_physical_step(seed_input, equilibrium, refined, output_name=output_name,
                                 seed_route=seed_route, reference=reference, step_lengths=physical_steps,
                                 family_charts=family_charts)
    inp = seed_input
    seed_equilibrium = equilibrium
    aspect_ref = float(equilibrium.wout.aspect)
    radius_ref = float(equilibrium.wout.Aminor_p)
    field_ref = float(equilibrium.wout.volavgB)
    seed_iota = float(opt.mean_iota(equilibrium.state, equilibrium.runtime))
    if not np.isfinite(seed_iota) or seed_iota == 0:
        raise ValueError("the seed must have finite nonzero rotational transform")
    iota_sign = float(np.sign(seed_iota))
    # Calibrate physical target bounds once, never from an optimizer trial.
    probe = CanonicalPwOResidual(SURFACES, bmin=1.0, bmax=2.0,
                                 t1=-0.3 * iota_sign, t2=3.0 * iota_sign, zeta_center=0.0, ntheta=GRID, nzeta=GRID, **PWO_OPTIONS)
    probe_data = probe.compute_state(equilibrium.state, equilibrium.runtime)
    sampled = np.asarray(probe_data["B"])
    bmin, bmax, b_ref = float(sampled.min()), float(sampled.max()), float(sampled.mean())
    calibration, theta_center, zeta_center = None, 0.0, 0.0
    if FAMILY == "canonical":
        unit_shape = np.asarray(probe_data["B_target"]) - 1.0
        shape_mean = float(unit_shape.mean())
        mirror_ref = (bmax - bmin) / (bmax + bmin)
        raw_bounds = [bmin, bmax]
        bmin, bmax = canonical_target_bounds(unit_shape, b_ref, mirror_ref)
        calibrated = bmin + (bmax - bmin) * unit_shape
        attained_mirror = float(np.ptp(calibrated) / (calibrated.max() + calibrated.min()))
        if not (0 < bmin < bmax and np.isfinite(bmax) and
                abs(calibrated.mean() / b_ref - 1) < 1e-12 and
                abs(attained_mirror / mirror_ref - 1) < 1e-12):
            raise RuntimeError("Canonical fixed mean/mirror calibration failed")
        theta_center, zeta_center, phase_costs = canonical_seed_phase(sampled, calibrated, seed_input.nfp)
        calibration = dict(shape_mean=shape_mean, seed_mean_T=b_ref, seed_mirror=mirror_ref,
                           phase_candidates_rad=[[float(i * np.pi), float(j * np.pi / seed_input.nfp)]
                                                 for i in range(2) for j in range(2)], seed_phase_costs=phase_costs,
                           raw_bounds_T=raw_bounds, target_mean_T=float(calibrated.mean()),
                           attained_mirror=attained_mirror, unit_min=float(unit_shape.min()),
                           unit_max=float(unit_shape.max()))

    def construction(options, grid, surfaces=SURFACES):
        return make_target(surfaces, family=FAMILY, bmin=bmin, bmax=bmax,
                           b_ref=b_ref, options=options, grid=grid, iota_sign=iota_sign,
                           theta_center=theta_center, zeta_center=zeta_center)

    target = construction(PWO_OPTIONS, GRID)
    field_key = "B" if FAMILY == "canonical" else "B_mapped"
    initial_target = target.compute_state(equilibrium.state, equilibrium.runtime)
    surface_field_ref = np.asarray(initial_target[field_key]).mean(axis=(1, 2))
    surface_mirror_ref = np.asarray(initial_target["mirror"])

    def physics_rows(state, runtime):
        # A single live transform supplies the fit and local anti-collapse controls.
        out = target.compute_state(state, runtime)
        B = out[field_key]
        local_mean = jnp.mean(B, axis=(1, 2))
        local_mirror = out["mirror"]
        rows = [out["residuals1d"],
            hold_ratio(local_mean, jnp.asarray(surface_field_ref), FIELD_TOLERANCE, PENALTY),
            jnp.maximum(MIRROR_FLOOR - local_mirror, 0) / MIRROR_FLOOR]
        rows.append(hold_ratio(local_mirror, jnp.asarray(surface_mirror_ref),
                               MIRROR_RELATIVE_TOLERANCE, PENALTY))
        return jnp.concatenate(rows)

    unit_beta = None
    if beta_targets[-1] > 0:
        unit = opt.solve_equilibrium(replace(seed_input, pres_scale=1.0),
                                     initial_state=equilibrium.solution, raise_on_max_iterations=True)
        unit_beta = float(unit.wout.betatotal)
        if not np.isfinite(unit_beta) or unit_beta <= 0:
            raise RuntimeError("unit-pressure calibration did not produce positive finite beta")
    monitor = opt.OptimizationMonitor()
    stages = []
    for beta_target in beta_targets:
        print(f"Building operating point beta={beta_target:.3%}, modes={MAX_MODES}", flush=True)
        inp = replace(inp, pres_scale=0.0 if beta_target == 0 else beta_target / unit_beta)

        def constraints(state, runtime):
            mirror = opt.mirror_ratio(state, runtime)
            rows = [
                hold_ratio(opt.aspect_ratio(state, runtime), aspect_ref, ASPECT_TOLERANCE, PENALTY),
                jnp.maximum(IOTA_FLOOR - opt.min_abs_iota(state, runtime), 0) / IOTA_FLOOR,
                jnp.maximum(IOTA_FLOOR - iota_sign * opt.mean_iota(state, runtime), 0) / IOTA_FLOOR,
                jnp.maximum(MIRROR_FLOOR - mirror, 0) / MIRROR_FLOOR,
                jnp.maximum(mirror - MIRROR_TRAIN_TARGET, 0) / MIRROR_TRAIN_TOLERANCE,
                jnp.maximum(opt.max_elongation(state, runtime) - ELONGATION_LIMIT, 0)
                / ELONGATION_LIMIT]
            if beta_target > 0:
                rows.append((opt.volume_average_beta(state, runtime) / beta_target - 1)
                            / BETA_TOLERANCE)
            return jnp.stack(rows)

        terms = [(physics_rows, 0.0, 10.0), (constraints, 0.0, 10.0)]

        def loss(state, runtime):
            rows = opt.residuals_from_tuples(state, runtime, terms)
            return 0.5 * jnp.vdot(rows, rows)

        factory = opt.VmecProblem.from_loss if METHOD == "scalar" else opt.VmecProblem.from_tuples
        problem = factory(
            inp, loss if METHOD == "scalar" else terms, max_mode=max(MAX_MODES),
            vary_major_radius=False, restart_from=equilibrium, max_fsq_ratio=1.0,
            adjoint_tol=1e-8, progress=not ci_smoke, evaluation_progress=not ci_smoke,
            jacobian_batch_size=1)
        full_x = problem.x0.copy()
        for max_mode, budget in zip(MAX_MODES, MAXITER):
            stage = problem.subproblem(max_mode=max_mode, x=full_x)
            monitor.problem = stage
            x0, step = stage.x0, PARAMETER_STEP * stage.scales
            if METHOD == "scalar":
                def value_and_gradient(y):
                    value, gradient = stage.value_and_grad(x0 + step * y)
                    monitor.cache_evaluation(x0 + step * y, value, gradient)
                    return value, step * gradient

                def record(intermediate_result):
                    monitor({"x": x0 + step * intermediate_result.x, "fun": intermediate_result.fun})

                initial = float(value_and_gradient(np.zeros_like(x0))[0])
                result = minimize(
                    value_and_gradient, np.zeros_like(x0), jac=True, method="L-BFGS-B",
                    bounds=[(-PARAMETER_BOUND, PARAMETER_BOUND)] * x0.size, callback=record,
                    options={"maxiter": budget, "maxfun": 5 * budget + 5, "maxls": 15,
                             "ftol": 1e-12, "gtol": 1e-7})
                x, final = x0 + step * result.x, float(result.fun)
            else:
                bounds = (x0 - PARAMETER_BOUND * step, x0 + PARAMETER_BOUND * step)
                initial = float(stage.fun(x0))
                result = least_squares(stage.residual, x0, jac=stage.residual_jac,
                                       bounds=bounds, x_scale=stage.scales, max_nfev=budget + 1,
                                       ftol=1e-7, xtol=1e-10, callback=monitor)
                x, final = result.x, float(result.cost)
            inp = stage.input_from_x(x)
            equilibrium = stage.equilibrium_from_x(x)
            full_x = problem.x_from_input(inp)
            stages.append(dict(beta_target=beta_target, mode=max_mode, initial_cost=initial,
                               final_cost=final, status=int(result.status), message=str(result.message)))
            print(f"beta {beta_target:.3%}, mode {max_mode}: cost {initial:.6e} -> {final:.6e}", flush=True)

    final_input, final_equilibrium = refined(inp, equilibrium)
    reference_input = replace(seed_input, pres_scale=inp.pres_scale)
    _, reference_equilibrium = refined(reference_input, seed_equilibrium)
    fine = construction(VALIDATION_OPTIONS, VALIDATION_GRID)
    initial_score = fine.compute_state(reference_equilibrium.state, reference_equilibrium.runtime)
    final_score = fine.compute_state(final_equilibrium.state, final_equilibrium.runtime)
    train_score = target.compute_state(final_equilibrium.state, final_equilibrium.runtime)
    metrics = opt.EquilibriumReporter(
        ("aspect", opt.aspect_ratio, ".5f"), ("min_abs_iota", opt.min_abs_iota, ".5f"),
        ("mean_iota", opt.mean_iota, ".5f"), ("mirror", opt.mirror_ratio, ".5f"),
        ("elongation", opt.max_elongation, ".5f"),
        ("beta", opt.volume_average_beta, ".4%"))("final", final_equilibrium)
    local_B = np.asarray(final_score[field_key])
    local_mean = local_B.mean(axis=(1, 2))
    local_mirror = np.asarray(final_score["mirror"])
    metrics.update(local_mirror=local_mirror.tolist(), local_mean_T=local_mean.tolist(),
                   initial_local_mirror=np.asarray(initial_score["mirror"]).tolist(),
                   minor_radius_m=float(final_equilibrium.wout.Aminor_p),
                   volavgB_T=float(final_equilibrium.wout.volavgB),
                   initial_fine=float(initial_score["total"]), fine=float(final_score["total"]),
                   training=float(train_score["total"]))
    checks = {
        "fine-grid objective improved": metrics["fine"] < metrics["initial_fine"],
        "angular refinement within 5%": abs(metrics["fine"] - metrics["training"])
        <= 0.05 * max(abs(metrics["fine"]), 1e-12),
        "valid target/map": bool(np.all(np.asarray(final_score["valid"]))),
        "iota floor": metrics["min_abs_iota"] >= IOTA_FLOOR,
        "iota sign": iota_sign * metrics["mean_iota"] >= IOTA_FLOOR,
        "training-surface mirror floor": bool(np.all(local_mirror >= MIRROR_FLOOR)),
        "training-surface field held": bool(np.all(abs(local_mean / surface_field_ref - 1) <= FIELD_TOLERANCE)),
        "mirror interval": MIRROR_FLOOR <= metrics["mirror"] <= MIRROR_LIMIT,
        "elongation limit": metrics["elongation"] <= ELONGATION_LIMIT,
        "aspect held": abs(metrics["aspect"] / aspect_ref - 1) <= ASPECT_TOLERANCE,
        "size held": abs(metrics["minor_radius_m"] / radius_ref - 1) <= SIZE_TOLERANCE,
        "field held": abs(metrics["volavgB_T"] / field_ref - 1) <= FIELD_TOLERANCE,
        "converged": bool(np.all(np.asarray(final_equilibrium.result.converged))),
    }
    checks["training-surface mirror held"] = bool(np.all(
        abs(local_mirror / surface_mirror_ref - 1) <= MIRROR_RELATIVE_TOLERANCE))
    if beta_targets[-1] > 0:
        checks["beta held"] = abs(metrics["beta"] / beta_targets[-1] - 1) <= BETA_TOLERANCE
    unmet = [name for name, met in checks.items() if not met]
    construction_diagnostics = {
        name: np.asarray(final_score[name]).tolist()
        for name in ("eta_fraction", "mapped_area_fraction", "jacobian_lower_bound", "corner_margin", "width")
        if name in final_score}
    summary = dict(family=FAMILY, method=METHOD, smoke=ci_smoke, seconds=time.perf_counter() - started,
                   seed_route=seed_route,
                   preconditioning=(None if qs_metrics is None else
                                    dict(branch="QP", source=QS_INPUT or "near_circular",
                                         seed_perturbation_m=SEED_PERTURBATION,
                                         stages=qs_stages, raw_metrics=qs_raw_metrics,
                                         training_metrics=qs_training_metrics,
                                         metrics=qs_metrics)),
                   stages=stages, metrics=metrics, actual_s=np.asarray(final_score["s_b"]).tolist(),
                   target=dict(calibration=calibration, theta_center_rad=theta_center, zeta_center_rad=zeta_center, p=2, bmin_T=bmin, bmax_T=bmax, b_ref_T=b_ref,
                               surface_field_ref_T=surface_field_ref.tolist(), iota_sign=iota_sign,
                               surface_mirror_ref=surface_mirror_ref.tolist(),
                               t1=-0.3 * iota_sign, t2=3.0 * iota_sign,
                               a=0.3, b=0.0, c=-2.0, eta_min=2.8),
                   settings=dict(stage_ns=STAGE_NS, stage_ftol=STAGE_FTOL, mpol=MPOL,
                                 penalty=PENALTY, mirror_train_target=MIRROR_TRAIN_TARGET,
                                 mirror_train_tolerance=MIRROR_TRAIN_TOLERANCE,
                                 final_ns=FINAL_NS, final_ftol=FINAL_FTOL, final_mpol=FINAL_MPOL,
                                 niter=NITER, grid=GRID, validation_grid=VALIDATION_GRID,
                                 transform=PWO_OPTIONS, validation_transform=VALIDATION_OPTIONS,
                                 mode_ladder=MAX_MODES, iteration_budgets=MAXITER,
                                 mirror_relative_tolerance=MIRROR_RELATIVE_TOLERANCE),
                   phiedge_Wb=float(inp.phiedge), pres_scale_Pa=float(inp.pres_scale),
                   pressure="p(s)=PRES_SCALE*(1-s)", current="zero enclosed toroidal current",
                   checks=checks, unmet=unmet, construction_diagnostics=construction_diagnostics,
                   research_validation_complete=False)
    # JSON rejects nonfinite diagnostic values instead of writing a misleading result.
    Path(f"{output_name}_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    final_input.to_indata(f"input.{output_name}")
    vj.write_wout(f"wout_{output_name}.nc", final_equilibrium.wout)
    vj.write_wout(f"wout_{output_name}_seed.nc", reference_equilibrium.wout)
    monitor.save(f"{output_name}_objectives.csv")
    print(json.dumps(metrics, indent=2))
    print("Unmet checks: " + (", ".join(unmet) if unmet else "none"))
    if unmet and not ci_smoke:
        raise SystemExit(1)
    return summary


if __name__ == "__main__":
    run()
