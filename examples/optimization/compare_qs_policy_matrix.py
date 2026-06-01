#!/usr/bin/env python
# ruff: noqa: E402
"""Benchmark QA/QH optimisation policies across configured ``max_mode`` values.

This script runs a small benchmark matrix for the standalone vmec_jax exact
optimisation path.  It compares three boundary-parameterisation policies:

1. ESS only          : start from the raw input boundary, no continuation
2. Continuation only : no ESS, but stage through max_mode = 1 -> ... -> N
3. Continuation+ESS  : combine both

Each case is executed in a spawned subprocess so that a low-level crash in one
JAX/XLA run does not abort the entire matrix.  Existing ``case_result.json``
files are reused when ``SKIP_EXISTING`` is true, so interrupted matrices can be
resumed without rerunning completed expensive cases.  For successful cases, the
script stores:

- ``wout_initial.nc`` and ``wout_final.nc``
- ``history.json`` with the full optimisation trajectory
- the standard ``boundary_comparison.png``, ``bmag_surface.png``,
  ``objective_history.png`` outputs

It also generates:

- one combined objective-history figure for QA and QH
- one best-equilibrium panel for QA and QH
- a ``summary.json`` table for all cases

All configuration is in top-level variables below.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import multiprocessing as mp
from pathlib import Path
import traceback

import numpy as np

import vmec_jax as vj
from vmec_jax._compat import enable_x64, jnp
from vmec_jax.config import config_from_indata
from vmec_jax.field import signgs_from_sqrtg
from vmec_jax.geom import eval_geom
from vmec_jax.init_guess import initial_guess_from_boundary
from vmec_jax.optimization import rebuild_indata_with_resolution
from vmec_jax.quasisymmetry import quasisymmetry_ratio_residual_from_state
from vmec_jax.plotting import prepare_matplotlib_3d
from vmec_jax.wout import equilibrium_aspect_ratio_from_state, equilibrium_iota_profiles_from_state

# ─────────────────────────────────────────────────────────────────────────────
# User parameters
# ─────────────────────────────────────────────────────────────────────────────

enable_x64(True)

OUTPUT_ROOT = Path("results/qs_policy_matrix_converged")
SKIP_EXISTING = True  # Reuse completed case_result.json files when resuming an interrupted matrix.

VMEC_MPOL = 6
VMEC_NTOR = 6

QH_INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "input.nfp4_QH_warm_start"
QA_INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "input.nfp2_QA_omnigenity"

MODES = (1, 2, 3)
PROBLEMS = ("qh", "qa")

QH_METHOD = "scipy"  # Try also "auto", "auto_scalar", "gauss_newton", "scipy_matrix_free", "lbfgs_adjoint", or "scalar_trust".
QA_METHOD = "scipy"  # Same optimizer choices as QH; keep both explicit for per-case tests.

QH_MAX_NFEV = 25  # Outer least-squares budget for the final stage.
QH_CONTINUATION_NFEV = 25  # Per-stage budget when mode continuation is enabled.
QH_FTOL = 1e-3  # Relative cost-reduction tolerance for the outer optimizer.
QH_GTOL = 1e-3  # Gradient optimality tolerance for the outer optimizer.
QH_XTOL = 1e-3  # Step-size tolerance for the outer optimizer.
ESS_ALPHA = 2.5
QH_ESS_ALPHA = ESS_ALPHA
QH_TARGET_ASPECT = 6.0
QH_SURFACES = np.arange(0.0, 1.01, 0.1)

QA_MAX_NFEV = 50  # QA usually needs more steps because iota is also constrained.
QA_CONTINUATION_NFEV = 35  # Per-stage budget for max_mode continuation.
QA_FTOL = 1e-3  # Relative cost-reduction tolerance for the outer optimizer.
QA_GTOL = 1e-3  # Gradient optimality tolerance for the outer optimizer.
QA_XTOL = 1e-3  # Step-size tolerance for the outer optimizer.
QA_ESS_ALPHA = ESS_ALPHA
QA_TARGET_ASPECT = 6.0
QA_TARGET_IOTA = 0.42
QA_IOTA_WEIGHT = 10_000.0
QA_SURFACES = np.arange(0.0, 1.01, 0.1)
QH_INNER_MAX_ITER = 0  # Accepted-point VMEC iterations; 0 uses NITER from the input deck.
QH_INNER_FTOL = 0  # Accepted-point VMEC tolerance; 0 uses FTOL_ARRAY from the input deck.
QH_TRIAL_MAX_ITER = 300  # Trial-point VMEC iterations; lower this for faster diagnostics.
QH_TRIAL_FTOL = 1e-10  # Trial-point VMEC tolerance; 0 follows the accepted/input tolerance.

QA_INNER_MAX_ITER = 0  # Accepted-point VMEC iterations; 0 uses NITER from the input deck.
QA_INNER_FTOL = 0  # Accepted-point VMEC tolerance; 0 uses FTOL_ARRAY from the input deck.
QA_TRIAL_MAX_ITER = 300  # Trial-point VMEC iterations; lower this for faster diagnostics.
QA_TRIAL_FTOL = 1e-10  # Trial-point VMEC tolerance; 0 follows the accepted/input tolerance.


@dataclass(frozen=True)
class Policy:
    name: str
    use_ess: bool
    use_mode_continuation: bool


POLICIES = (
    Policy(name="ess_only", use_ess=True, use_mode_continuation=False),
    Policy(name="continuation_only", use_ess=False, use_mode_continuation=True),
    Policy(name="continuation_ess", use_ess=True, use_mode_continuation=True),
)


@dataclass
class CaseResult:
    problem: str
    max_mode: int
    policy: str
    success: bool
    crashed: bool
    message: str
    objective_final: float | None = None
    qs_final: float | None = None
    aspect_final: float | None = None
    iota_final: float | None = None
    nfev: int | None = None
    njev: int | None = None
    total_wall_time_s: float | None = None
    output_dir: str | None = None


def _problem_constants(problem: str) -> dict:
    if problem == "qh":
        return {
            "input_file": QH_INPUT_FILE,
            "method": QH_METHOD,
            "max_nfev": QH_MAX_NFEV,
            "continuation_nfev": QH_CONTINUATION_NFEV,
            "ftol": QH_FTOL,
            "gtol": QH_GTOL,
            "xtol": QH_XTOL,
            "alpha": QH_ESS_ALPHA,
            "surfaces": QH_SURFACES,
            "aspect_weight": 1.0,
            "iota_weight": 1.0,
            "qs_weight": 1.0,
            "target_aspect": QH_TARGET_ASPECT,
            "target_iota": None,
            "helicity_m": 1,
            "helicity_n": -1,
            "inner_max_iter": QH_INNER_MAX_ITER,
            "inner_ftol": QH_INNER_FTOL,
            "trial_max_iter": QH_TRIAL_MAX_ITER,
            "trial_ftol": QH_TRIAL_FTOL,
        }
    if problem == "qa":
        return {
            "input_file": QA_INPUT_FILE,
            "method": QA_METHOD,
            "max_nfev": QA_MAX_NFEV,
            "continuation_nfev": QA_CONTINUATION_NFEV,
            "ftol": QA_FTOL,
            "gtol": QA_GTOL,
            "xtol": QA_XTOL,
            "alpha": QA_ESS_ALPHA,
            "surfaces": QA_SURFACES,
            "aspect_weight": 1.0,
            "iota_weight": QA_IOTA_WEIGHT,
            "qs_weight": 1.0,
            "target_aspect": QA_TARGET_ASPECT,
            "target_iota": QA_TARGET_IOTA,
            "helicity_m": 1,
            "helicity_n": 0,
            "inner_max_iter": QA_INNER_MAX_ITER,
            "inner_ftol": QA_INNER_FTOL,
            "trial_max_iter": QA_TRIAL_MAX_ITER,
            "trial_ftol": QA_TRIAL_FTOL,
        }
    raise ValueError(f"Unknown problem '{problem}'")


def _build_stage(problem: str, cfg, indata0, max_mode: int, constants: dict):
    stage_static = vj.build_static(cfg)
    stage_boundary = vj.boundary_from_indata(indata0, stage_static.modes, apply_m1_constraint=False)
    stage_indata, stage_static, stage_boundary = vj.extend_boundary_for_max_mode(
        indata0, stage_static, stage_boundary, max_mode
    )
    stage_boundary_input = vj.boundary_input_from_indata(stage_indata, stage_static.modes)
    stage_specs = vj.boundary_param_specs(
        stage_boundary_input,
        stage_static.modes,
        max_mode=max_mode,
        min_coeff=0.0,
        include=("rc", "zs"),
        fix=("rc00",),
    )
    stage_guess = initial_guess_from_boundary(stage_static, stage_boundary, stage_indata, vmec_project=True)
    stage_geom = eval_geom(stage_guess, stage_static)
    stage_signgs = int(signgs_from_sqrtg(np.asarray(stage_geom.sqrtg), axis_index=1))
    stage_flux = vj.flux_profiles_from_indata(stage_indata, stage_static.s, signgs=stage_signgs)
    stage_pressure = jnp.zeros_like(jnp.asarray(stage_static.s))

    def stage_qs_eval(state):
        return quasisymmetry_ratio_residual_from_state(
            state=state,
            static=stage_static,
            indata=stage_indata,
            signgs=stage_signgs,
            flux_local=stage_flux,
            prof_local={"pressure": stage_pressure},
            pressure_local=stage_pressure,
            surfaces=constants["surfaces"],
            helicity_m=constants["helicity_m"],
            helicity_n=constants["helicity_n"],
        )

    def mean_iota_raw(state):
        _chips, iotas, _iotaf = equilibrium_iota_profiles_from_state(
            state=state,
            static=stage_static,
            indata=stage_indata,
            signgs=stage_signgs,
        )
        iotas = jnp.asarray(iotas, dtype=jnp.float64)
        return jnp.asarray(0.0, dtype=iotas.dtype) if int(iotas.shape[0]) <= 1 else jnp.mean(iotas[1:])

    def iota_fn(state):
        return float(mean_iota_raw(state))

    def stage_residuals_fn(state):
        parts = []
        aspect = equilibrium_aspect_ratio_from_state(state=state, static=stage_static)
        parts.append(
            jnp.sqrt(jnp.asarray(constants["aspect_weight"], dtype=jnp.float64))
            * jnp.asarray([aspect - constants["target_aspect"]], dtype=jnp.float64)
        )
        if constants["target_iota"] is not None:
            parts.append(
                jnp.sqrt(jnp.asarray(constants["iota_weight"], dtype=jnp.float64))
                * jnp.asarray([mean_iota_raw(state) - constants["target_iota"]], dtype=jnp.float64)
            )
        qs = stage_qs_eval(state)
        parts.append(
            jnp.sqrt(jnp.asarray(constants["qs_weight"], dtype=jnp.float64))
            * jnp.asarray(qs["residuals1d"], dtype=jnp.float64)
        )
        return jnp.concatenate(parts)

    stage_residuals_fn._n_non_qs = 2 if constants["target_iota"] is not None else 1
    stage_residuals_fn._qs_total_from_state = lambda state: float(stage_qs_eval(state)["total"])

    stage_opt = vj.FixedBoundaryExactOptimizer(
        stage_static,
        stage_indata,
        stage_boundary,
        stage_specs,
        stage_residuals_fn,
        boundary_input=stage_boundary_input,
        inner_max_iter=constants["inner_max_iter"],
        inner_ftol=constants["inner_ftol"],
        trial_max_iter=constants["trial_max_iter"],
        trial_ftol=constants["trial_ftol"],
    )
    return stage_specs, stage_opt, iota_fn


def _merge_stage_histories(stage_results: list[tuple[int, dict]], *, constants: dict) -> dict:
    combined_entries = []
    wall_offset = 0.0
    nfev_total = 0
    njev_total = 0
    for idx, (_mode, stage_result) in enumerate(stage_results):
        stage_hist = stage_result["_history_dump"]
        entries = stage_hist["history"] if idx == 0 else stage_hist["history"][1:]
        for entry in entries:
            entry_copy = dict(entry)
            entry_copy["wall_time_s"] = float(entry_copy["wall_time_s"]) + wall_offset
            combined_entries.append(entry_copy)
        wall_offset = combined_entries[-1]["wall_time_s"]
        nfev_total += int(stage_hist["nfev"])
        njev_total += int(stage_hist["njev"])
    final_hist = stage_results[-1][1]["_history_dump"]
    merged = {
        "label": "Optimisation",
        "max_nfev": int(sum(constants["continuation_nfev"] if m != stage_results[-1][0] else constants["max_nfev"] for m, _ in stage_results)),
        "ftol": constants["ftol"],
        "gtol": constants["gtol"],
        "xtol": constants["xtol"],
        "total_wall_time_s": float(wall_offset),
        "nfev": int(nfev_total),
        "njev": int(njev_total),
        "success": bool(final_hist["success"]),
        "message": str(final_hist["message"]),
        "objective_initial": float(stage_results[0][1]["_history_dump"]["objective_initial"]),
        "objective_final": float(final_hist["objective_final"]),
        "qs_initial": float(stage_results[0][1]["_history_dump"]["qs_initial"]),
        "qs_final": float(final_hist["qs_final"]),
        "aspect_initial": float(stage_results[0][1]["_history_dump"]["aspect_initial"]),
        "aspect_final": float(final_hist["aspect_final"]),
        "history": combined_entries,
        "target_aspect": constants["target_aspect"],
    }
    if constants["target_iota"] is not None:
        merged["target_iota"] = float(constants["target_iota"])
        if "iota" in combined_entries[0] and "iota" in combined_entries[-1]:
            merged["iota_initial"] = float(combined_entries[0]["iota"])
            merged["iota_final"] = float(combined_entries[-1]["iota"])
    return merged


def _run_case(problem: str, max_mode: int, policy: Policy, output_dir: Path) -> CaseResult:
    constants = _problem_constants(problem)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg, indata = vj.load_config(str(constants["input_file"]))
    indata = rebuild_indata_with_resolution(indata, mpol=VMEC_MPOL, ntor=VMEC_NTOR)
    cfg = config_from_indata(indata)

    stage_modes = (
        list(range(1, max_mode + 1))
        if (policy.use_mode_continuation and max_mode > 1)
        else [max_mode]
    )
    stage_results = []
    params_stage = None
    prev_specs = None

    for stage_mode in stage_modes:
        stage_specs, stage_opt, iota_fn = _build_stage(problem, cfg, indata, stage_mode, constants)
        stage_x_scale = (
            vj.create_x_scale(stage_specs, alpha=constants["alpha"])
            if policy.use_ess
            else np.ones(len(stage_specs))
        )
        params0_stage = (
            np.zeros(len(stage_specs))
            if params_stage is None
            else vj.lift_boundary_params(prev_specs, params_stage, stage_specs)
        )
        stage_budget = constants["max_nfev"] if stage_mode == max_mode else constants["continuation_nfev"]
        stage_result = stage_opt.run(
            params0_stage,
            method=constants["method"],
            max_nfev=stage_budget,
            ftol=constants["ftol"],
            gtol=constants["gtol"],
            xtol=constants["xtol"],
            x_scale=stage_x_scale,
            verbose=0,
            iota_fn=iota_fn if constants["target_iota"] is not None else None,
            target_iota=constants["target_iota"],
            target_aspect=constants["target_aspect"],
        )
        stage_results.append((stage_mode, stage_result))
        prev_specs = stage_specs
        params_stage = stage_result["x"]
        final_opt = stage_opt
        final_params0 = params0_stage
        final_result = stage_result

    if policy.use_mode_continuation and len(stage_results) > 1:
        final_result["_history_dump"] = _merge_stage_histories(stage_results, constants=constants)

    final_opt.save_wout(output_dir / "wout_initial.nc", final_params0, state=final_result.get("_state_initial"))
    final_opt.save_wout(output_dir / "wout_final.nc", final_result["x"], state=final_result.get("_state_final"))
    final_opt.save_history(output_dir / "history.json", final_result)
    vj.plot_3d_boundary_comparison(
        output_dir / "wout_initial.nc",
        output_dir / "wout_final.nc",
        outdir=output_dir,
    )
    vj.plot_bmag_contours(
        output_dir / "wout_initial.nc",
        output_dir / "wout_final.nc",
        outdir=output_dir,
    )
    vj.plot_objective_history(output_dir / "history.json", outdir=output_dir)

    hist = final_result["_history_dump"]
    return CaseResult(
        problem=problem,
        max_mode=max_mode,
        policy=policy.name,
        success=bool(hist["success"]),
        crashed=False,
        message=str(hist["message"]),
        objective_final=float(hist["objective_final"]),
        qs_final=float(hist["qs_final"]),
        aspect_final=float(hist["aspect_final"]),
        iota_final=None if constants["target_iota"] is None else float(hist["history"][-1]["iota"]),
        nfev=int(hist["nfev"]),
        njev=int(hist["njev"]),
        total_wall_time_s=float(hist["total_wall_time_s"]),
        output_dir=str(output_dir),
    )


def _worker(problem: str, max_mode: int, policy: Policy, output_dir: str, result_path: str):
    try:
        case_result = _run_case(problem, max_mode, policy, Path(output_dir))
        Path(result_path).write_text(json.dumps(asdict(case_result), indent=2))
    except Exception as exc:
        failed = CaseResult(
            problem=problem,
            max_mode=max_mode,
            policy=policy.name,
            success=False,
            crashed=True,
            message=f"{type(exc).__name__}: {exc}",
            output_dir=str(output_dir),
        )
        Path(result_path).write_text(json.dumps(asdict(failed), indent=2))
        Path(output_dir, "traceback.txt").write_text(traceback.format_exc())
        raise


def _plot_policy_matrix(results: list[CaseResult], *, problem: str, outpath: Path) -> None:
    prepare_matplotlib_3d()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(MODES), figsize=(5.5 * len(MODES), 4.5), sharey=True)
    if len(MODES) == 1:
        axes = [axes]
    for ax, mode in zip(axes, MODES, strict=False):
        ax.set_title(f"{problem.upper()} max_mode={mode}")
        for policy in POLICIES:
            rec = next((r for r in results if r.problem == problem and r.max_mode == mode and r.policy == policy.name), None)
            if rec is None or rec.output_dir is None:
                continue
            hist_path = Path(rec.output_dir) / "history.json"
            if not hist_path.exists():
                continue
            data = json.loads(hist_path.read_text())
            y = [max(float(entry["objective"]), 1e-16) for entry in data["history"]]
            x = list(range(len(y)))
            label = policy.name.replace("_", " ")
            style = "-" if rec.success else "--"
            ax.semilogy(x, y, linestyle=style, linewidth=2.0, label=label)
        ax.set_xlabel("History index")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Total objective")
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_policy_matrix_all(results: list[CaseResult], *, outpath: Path) -> None:
    prepare_matplotlib_3d()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(PROBLEMS),
        len(MODES),
        figsize=(5.5 * len(MODES), 4.4 * len(PROBLEMS)),
        sharey="row",
        squeeze=False,
    )
    for row, problem in enumerate(PROBLEMS):
        for col, mode in enumerate(MODES):
            ax = axes[row, col]
            ax.set_title(f"{problem.upper()} max_mode={mode}")
            for policy in POLICIES:
                rec = next((r for r in results if r.problem == problem and r.max_mode == mode and r.policy == policy.name), None)
                if rec is None or rec.output_dir is None:
                    continue
                hist_path = Path(rec.output_dir) / "history.json"
                if not hist_path.exists():
                    continue
                data = json.loads(hist_path.read_text())
                y = [max(float(entry["objective"]), 1e-16) for entry in data["history"]]
                x = list(range(len(y)))
                label = policy.name.replace("_", " ")
                style = "-" if not rec.crashed else "--"
                ax.semilogy(x, y, linestyle=style, linewidth=2.0, label=label)
            if row == len(PROBLEMS) - 1:
                ax.set_xlabel("History index")
            if col == 0:
                ax.set_ylabel(f"{problem.upper()} total objective")
            ax.grid(True, alpha=0.3)
    handles, labels = axes[0, -1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_best_equilibria(results: list[CaseResult], *, outpath: Path) -> None:
    prepare_matplotlib_3d()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    best = {}
    for problem in PROBLEMS:
        candidates = [
            r for r in results
            if r.problem == problem and not r.crashed and r.objective_final is not None and r.output_dir is not None
        ]
        if not candidates:
            continue
        best[problem] = min(candidates, key=lambda r: r.objective_final)

    fig, axes = plt.subplots(len(best), 2, figsize=(10, 5 * max(1, len(best))))
    if len(best) == 1:
        axes = np.asarray([axes])
    for row, problem in enumerate(best):
        rec = best[problem]
        boundary = mpimg.imread(Path(rec.output_dir) / "boundary_comparison.png")
        bmag = mpimg.imread(Path(rec.output_dir) / "bmag_surface.png")
        axes[row, 0].imshow(boundary)
        axes[row, 0].axis("off")
        axes[row, 0].set_title(f"{problem.upper()} best 3D LCFS\n{rec.policy}, mode {rec.max_mode}")
        axes[row, 1].imshow(bmag)
        axes[row, 1].axis("off")
        axes[row, 1].set_title(
            f"{problem.upper()} |B| on LCFS\nobj={rec.objective_final:.3e}"
        )
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    ctx = mp.get_context("spawn")

    for problem in PROBLEMS:
        for max_mode in MODES:
            for policy in POLICIES:
                output_dir = OUTPUT_ROOT / problem / f"mode{max_mode}" / policy.name
                result_path = output_dir / "case_result.json"
                if SKIP_EXISTING and result_path.exists():
                    result = CaseResult(**json.loads(result_path.read_text()))
                    print(
                        f"[{problem} mode={max_mode} policy={policy.name}] "
                        f"reusing existing result objective={result.objective_final}"
                    )
                else:
                    proc = ctx.Process(
                        target=_worker,
                        args=(problem, max_mode, policy, str(output_dir), str(result_path)),
                    )
                    proc.start()
                    proc.join()
                    if result_path.exists():
                        result = CaseResult(**json.loads(result_path.read_text()))
                    else:
                        result = CaseResult(
                            problem=problem,
                            max_mode=max_mode,
                            policy=policy.name,
                            success=False,
                            crashed=True,
                            message=f"worker exit code {proc.exitcode} without result file",
                            output_dir=str(output_dir),
                        )
                    if proc.exitcode not in (0, None):
                        result.crashed = True
                        if "worker exit code" not in result.message:
                            result.message = f"exit code {proc.exitcode}; {result.message}"
                results.append(result)
                print(
                    f"[{problem} mode={max_mode} policy={policy.name}] "
                    f"success={result.success} crashed={result.crashed} "
                    f"objective={result.objective_final}"
                )

    summary = [asdict(r) for r in results]
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    for problem in PROBLEMS:
        _plot_policy_matrix(results, problem=problem, outpath=OUTPUT_ROOT / f"{problem}_policy_objectives.png")
    _plot_policy_matrix_all(results, outpath=OUTPUT_ROOT / "all_policy_objectives.png")
    _plot_best_equilibria(results, outpath=OUTPUT_ROOT / "best_equilibria.png")
    print(f"Wrote {OUTPUT_ROOT / 'summary.json'}")


if __name__ == "__main__":
    main()
