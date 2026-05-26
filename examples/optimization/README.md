# Optimization Examples

These scripts are intended to be read and modified directly.  The recommended
workflow is to instantiate a VMEC object, assemble a list of objective tuples
`(objective_function, target, weight)`, call `least_squares_solve`, and then
inspect, save, or plot the returned result.

## Editable Workflow Anatomy

The QA/QH/QP scripts are intentionally linear.  Read them as four visible
blocks, not as wrappers:

| Block | Edit when you want to... | Where to look |
| --- | --- | --- |
| VMEC and stage setup | change input deck, active boundary modes, continuation, or device | top-level variables and `FixedBoundaryVMEC.from_input(...)` |
| Objective assembly | add/remove physics terms or tune targets and weights | `objective_tuples = [...]` and `LeastSquaresProblem.from_tuples(...)` |
| Solve controls | change optimizer method, tolerances, ESS, or saved stage artifacts | `least_squares_solve(...)` keyword arguments |
| Outputs and plots | change filenames, add exports, or choose figures | `result` properties, `save_optimization_result`, direct `save_input`/`save_wout`/`save_history`, and `vj.plot_*` calls |

`least_squares_solve` receives optimizer, continuation, device, and output
controls only.  Scientific targets stay in the explicit objective tuple list
or, for QI field terms, in the shared `QuasiIsodynamicOptions` and objective
objects.

The QI example follows the same pattern but has more visible physics objects:
`QuasiIsodynamicOptions`, `QuasiIsodynamicResidual`, `MirrorRatio`,
`VMECMirrorRatio`, `MaxElongation`, and optional seed-preparation helpers are all
configured as ordinary top-level variables before the objective tuple list is
built.  Use `VMECMirrorRatio` when you only need a fast mirror-ratio soft wall;
use `MirrorRatio` when you want the value evaluated from a Boozer transform or
shared with a QI Boozer field.

## Objective Tuple Pattern

Use explicit SIMSOPT-style tuples and keep the list visible:

```python
aspect = vj.AspectRatio()
iota_floor = vj.AbsMeanIotaFloor(0.41)
qs = vj.QuasisymmetryRatioResidual(helicity_m=1, helicity_n=-1, surfaces=SURFACES)

objective_tuples = [
    (aspect.J, TARGET_ASPECT, ASPECT_WEIGHT),
    (iota_floor.J, 0.0, IOTA_FLOOR_WEIGHT),
    (qs.J, 0.0, QS_WEIGHT),
]
problem = vj.LeastSquaresProblem.from_tuples(objective_tuples)
```

`weight` follows SIMSOPT semantics: the residual is
`sqrt(weight) * (objective - target)`.  Do not pre-scale callbacks.  QI terms
use the same tuple form, but encode QI thresholds and smoothing in
`QuasiIsodynamicOptions`, `MirrorRatio`, `VMECMirrorRatio`, and
`MaxElongation`; QI tuple targets
should remain `0.0`. `QuasiIsodynamicOptions` defaults to `jit_booz=True`,
which is currently faster for the Boozer/QI residual phase on both CPU and GPU;
set it to `False` only for diagnostics or parity isolation.

## Minimal Far-From-Goal Seed Inputs

The bundled files `examples/data/input.minimal_seed_nfp1`,
`examples/data/input.minimal_seed_nfp2`, `examples/data/input.minimal_seed_nfp3`,
and `examples/data/input.minimal_seed_nfp4` all use the same three-coefficient
boundary template:

```python
indata = vj.minimal_fixed_boundary_indata(nfp=2)
vj.write_indata("input.minimal_seed_nfp2", indata)
```

Only `RBC(0,0)`, `RBC(0,1)`, and `ZBS(0,1)` are nonzero.  Use these inputs
when testing whether QA, QH, QP, or QI policies can build the target
omnigenous structure from a seed that does not already contain the target
helicity.  Higher Fourier coefficients should be introduced by `max_mode`,
mode continuation, ESS, or a staged QI policy, not by the seed file itself.

For deterministic simple-seed runs, keep the raw input files exactly as above
and add only optimization-time hints.  The standalone QA/QH/QP examples call
`vj.prepare_simple_omnigenity_seed_input(...)`, which writes an `input.simple_seed`
under the run directory.  That generated input keeps only `RBC(0,0)`,
`RBC(0,1)`, and `ZBS(0,1)` from the selected seed deck, then fills active
`RBC/ZBS` modes up to `max_mode` with deterministic `1e-5` perturbations so
the Jacobian does not start on an exactly zero-transform branch.  The raw
bundled input decks are not modified.  QP is more sensitive to the
zero-transform basin in direct high-mode starts from the common minimal seed,
so `QP_optimization.py` keeps the same `SIMPLE_SEED_PERTURBATION = 1e-5` as
QA/QH but repeats intermediate continuation modes before the final high-mode
cleanup.
The lower-level QI staged policy also
has a mode-1 target-helicity preconditioner with the deterministic hint set
`RBC(1,0)`, `ZBS(1,0)`, `RBC(-1,1)`, `ZBS(-1,1)`, `RBC(1,1)`, and `ZBS(1,1)`
in VMEC input-index convention.
The QA and QP common-minimal rows also use an explicit optimization-time
reference-family preseed: QA blends active low-order RBC/ZBS terms 25% toward
`input.nfp2_QA_omnigenity`, and QP blends 10% toward `input.nfp2_QI`.  This is
recorded in `showcase_case.json`; rows that lack that provenance predate the
current seed-robustness policy.

The bounded common-seed showcase is a stress test, not a best-result table.  It
maps the configured minimal seeds to QI NFP=1/2/3/4, QA NFP=2, QH NFP=4, and QP
NFP=2, then renders the failure-revealing objective panel used by the docs.  The
QI rows dispatch through `QI_optimization.py` via `qi_staged_runner.py`, so the
common minimal seeds use the same staged/reference-family QI policy as the
standalone QI example instead of the simpler quasisymmetry sweep path.  The
checked-in panel is intentionally conservative: the renderer skips stale QI rows
and only promotes rows with current provenance, so missing QI NFP rows indicate
open validation work rather than successful hidden results.

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/generate_minimal_seed_showcase.py \
  --cases all --backend-label cpu --solver-device cpu --worker-jax-platforms cpu \
  --policy continuation --max-mode 3 --ess on \
  --max-nfev 30 --continuation-nfev 20 \
  --inner-max-iter 120 --trial-max-iter 120 \
  --inner-ftol 1e-9 --trial-ftol 1e-9 --case-timeout-s 1800 --rerun
PYTHONPATH=. python examples/optimization/render_minimal_seed_showcase.py
```

Keep `--rerun` for a fresh local reproduction.  Without it, existing
successful `showcase_case.json` rows are reused and can leave old outputs on
disk; the renderer skips known-stale rows by default, and `--include-stale`
should be reserved for debugging.  Current common-minimal QI rows use policy
case names `minimal_nfp1_qi`, `minimal_nfp2_qi`, `minimal_nfp3_qi`, and
`minimal_nfp4_qi`; old QI rows under `.../continuation/qp_preseed/...` or the
legacy case names `nfp1_qi`, `nfp2_qi`, `qi_stel_seed_3127`, and `nfp4_qi`
predate the staged dispatch.  Old QA/QP rows without `reference_preseed`
metadata also predate the current reference-family preseed policy.
When a case hits `--case-timeout-s`, the runner terminates the worker process
group, including solver or GPU descendant processes, before writing the timeout
result.

## Result Object Pattern

The standalone scripts also show how to work from the returned result object
instead of relying on hidden plotting or printing helpers:

```python
result = vj.least_squares_solve(vmec, problem, ...)

history = result.history
objective_history = result.objective_history
timing = result.timing_summary

saved_paths = vj.save_optimization_result(result, output_dir=OUTPUT_DIR)

print(history["objective_final"])
print(timing["total_wall_time_s"])
print(objective_history[-3:])

wout_final = vj.load_wout(saved_paths.final_wout)
theta, zeta, b_lcfs = vj.vmecplot2_bmag_grid(wout_final, s_index=-1)

plot_paths = {
    "boundary_comparison": vj.plot_3d_boundary_comparison(
        saved_paths.initial_wout,
        saved_paths.final_wout,
        outdir=OUTPUT_DIR,
    ),
    "boozer_lcfs_bmag_contours": vj.plot_boozer_lcfs_bmag_comparison(
        saved_paths.initial_wout,
        saved_paths.final_wout,
        outdir=OUTPUT_DIR,
    ),
    "objective_history": vj.plot_objective_history(
        saved_paths.history,
        outdir=OUTPUT_DIR,
    ),
}
```

The examples pass `save_final_outputs=False` and then call
`save_optimization_result` so users can see the result object before choosing
diagnostics and plots.  For custom filenames, call `result.initial_optimizer`
and `result.final_optimizer` directly; for convenience, omit
`save_final_outputs=False` to let `least_squares_solve` write the default final
artifacts. `result.final_params` and `result.final_state` refer to the selected exact accepted point, not an unreplayed relaxed trial point.
For continuation details, start with
`result.initial_stage` and `result.final_stage`; use
`result.stage_histories` and `result.stage_timing_summaries` for per-stage
accepted exact-replay history and timing.  The raw `result.stage_records`
remain available for custom inspection.

## Recommended Standalone Examples

- `QA_optimization.py`: recommended quasi-axisymmetric fixed-boundary optimization.
- `QH_optimization.py`: recommended quasi-helical fixed-boundary optimization.
- `QP_optimization.py`: quasi-poloidal fixed-boundary optimization from the NFP=2 QI seed.
- `QI_optimization.py`: recommended quasi-isodynamic optimization with Boozer-space QI metrics, mirror-ratio and elongation penalties, repeated same-mode continuation, and ESS. Edit `INPUT_FILE`, `OUTPUT_DIR`, seed helpers, objective weights, and optimizer controls at the top of the script, then run it directly.
- `qa_optimization_finite_beta.py`, `qh_optimization_finite_beta.py`, and `qi_optimization_finite_beta.py`:
  finite-beta stage-1 examples with pressure/current-profile terms. These intentionally use
  `FixedBoundaryExactOptimizer` directly because each continuation stage builds custom
  finite-pressure/current residual closures; the helper only standardizes stage artifacts.

`QI_optimization.py` keeps the same visible workflow as QA/QH/QP: edit
top-level controls, construct objective tuples, call `least_squares_solve`,
then save and plot from the returned result.  If a far seed needs basin
capture before local QI cleanup, enable `USE_TARGET_HELICITY_SEED` or
`USE_REFERENCE_FAMILY_SEED` in that script and point `REFERENCE_INPUT_FILE` at
a scientifically appropriate same-NFP reference.  The scientific defaults
remain visible in the driver, and physics terms stay in objective tuples
rather than being passed as shortcut arguments into `least_squares_solve`.

Run one case from the repository root:

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/QH_optimization.py
```

Set `SOLVER_DEVICE = "gpu"` inside the script, or run with
`JAX_PLATFORM_NAME=gpu`, to use a GPU-enabled JAX installation.
Optimizer and output controls are also top-level variables, including
`METHOD`, `SCIPY_TR_SOLVER`, `FTOL`, `GTOL`, `XTOL`, `INNER_MAX_ITER`,
`TRIAL_MAX_ITER`, `SAVE_STAGE_INPUTS`, `SAVE_STAGE_WOUTS`, and `MAKE_PLOTS`.
For common-minimal seed studies, also inspect `SIMPLE_SEED_PERTURBATION` and
`STAGE_MODES`; these are ordinary script-level controls, not hidden solver
settings.

For reproducible comparison artifacts, use the sweep driver rather than a
single edited script.  The current production sweep runs QI directly from its
bundled seed (`--qi-qp-preseed off`); use `--qi-qp-preseed both` only for the
focused QI preseed/no-preseed matrix.
`--backend-label` is output provenance only; it does not select a backend.
Select the process backend with `JAX_PLATFORMS=cpu`, `JAX_PLATFORM_NAME=gpu`,
or `JAX_PLATFORMS=cuda`, and pass matching `--solver-device` when the optimizer
should force a device.  Spawned workers inherit the parent JAX selection unless
`--worker-jax-platforms` is set.  Confirm the actual runtime from
`case_result.json` or summary CSV fields `jax_backend`, `jax_device_kind`,
`solver_device`, and `jax_platforms`.  Existing successful `case_result.json`
files are reused; add `--rerun` when you need a fresh local reproduction.

## QI Diagnostics

Before treating a QI result as a final candidate, audit the smooth objective
against the legacy branch diagnostics and render the constrained QI matrix.
The first-class record helpers are `vj.QIDiagnosticOptions`,
`vj.qi_diagnostics_from_boozer_output`, `vj.qi_diagnostics_from_state`, and
`vj.rank_qi_seed_records`; they return unweighted smooth/raw aliases, legacy QI,
mirror-ratio, elongation, optional `LgradB`, resolution metadata, and
diagnostic error fields.

Mirror-ratio cleanup must be guarded by a QI residual ceiling or by an
independent engineering promotion gate.  Endpoints that lower mirror ratio but
fail the independent smooth/legacy QI and engineering gates are rejected and
should not be promoted as improved QI candidates.

For far seeds, `QI_optimization.py` uses the same single script but a longer
policy.  The `input.QI_stel_seed_3127` case now starts with a deterministic
same-NFP reference-family preconditioner: it interpolates the raw seed boundary
toward the bundled NFP=3 QI reference, runs bounded fixed-boundary solves for
the candidate interpolation points, ranks them with independent QI/mirror/iota
diagnostics, prefers a non-endpoint candidate when one passes the gate, and
records that candidate as the accepted baseline before local QI cleanup.  This
is a global-to-local move; the
previous ESS-scaled local basin prefilter remains available but is not the
default for this seed because it did not enter the precise-QI basin.
The far-seed gate keeps the legacy Goodman-style metric tight at `2e-3` and
uses a `5e-3` cap for the smooth differentiable proxy on the six-surface audit.
Mirror-balanced cleanup stages are kept in diagnostic scripts because the
current all-surface mirror objective trades away the QI gate for purely local
`input.QI_stel_seed_3127` runs.
The final files in the top-level output directory come from the last promoted
stage, or from the best exact-diagnostic candidate if no stage passes the
promotion gate.  Review `basin_prefilter/top_candidates.json` and
`mirror_ramp_promotion_log.json` before using a far-seed result in figures.
Far-seed policies may use a cheaper Boozer/QI grid during optimization and a
higher-resolution final audit; both grids are written into `diagnostics.json`.

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/compare_omnigenity_qi_objective.py
PYTHONPATH=. JAX_PLATFORMS=cpu python tools/diagnostics/qi_objective_component_report.py
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/audit_qi_seed_suitability.py --quick \
  --csv results/qi_seed_audit.csv
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/audit_qi_seed_suitability.py --quick \
  --smooth-qi-max 5e-3 --legacy-qi-max 2e-3 \
  --csv results/qi_seed3127_audit.csv
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/audit_qi_seed_suitability.py --quick \
  --prefine-probes plan \
  --prefine-manifest results/qi_seed_audit/prefine_manifest.json \
  --prefine-output-dir results/qi_seed_audit/prefine_probes
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/QI_seed_robustness.py
# Edit INPUT_FILE and OUTPUT_DIR at the top of QI_optimization.py, then run:
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/QI_optimization.py
PYTHONPATH=. JAX_PLATFORMS=cpu python tools/diagnostics/qi_boundary_interpolation_scan.py \
  --seed-input examples/data/input.QI_stel_seed_3127 \
  --reference-input examples/data/input.nfp3_QI_fixed_resolution_final \
  --out-root results/diagnostics/qi_seed3127_boundary_interpolation \
  --lambdas 0.99,0.995,1.0,1.005,1.008,1.01,1.012 \
  --max-mode 4 --max-iter 80 --target-aspect 6.0 \
  --surfaces 0.1,0.28,0.46,0.64,0.82,1.0 \
  --mboz 18 --nboz 18 --nphi 151 --nalpha 31 --n-bounce 51 \
  --smooth-qi-max 5e-3 --legacy-qi-max 2e-3 \
  --max-mirror-ratio 0.35 --max-elongation 8.0
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/generate_qs_ess_sweep.py \
  --backend-label cpu --solver-device cpu --policy continuation \
  --problems qi --modes 1,2,3 --ess both --qi-qp-preseed both --rerun
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/generate_qs_ess_sweep.py \
  --backend-label cpu --solver-device cpu --policy direct \
  --problems qi --modes 1,2,3 --ess both --qi-qp-preseed both --rerun
PYTHONPATH=. python examples/optimization/render_qi_constrained_sweep.py
PYTHONPATH=. python examples/optimization/render_qi_readme_cases.py
```

To reproduce a specific reviewed NFP=1/2/3/4 QI row, edit the top-level
`INPUT_FILE`, `OUTPUT_DIR`, and optional `REFERENCE_INPUT_FILE` values in
`QI_optimization.py`, then run the command above.  The archived
`readme_qi_optimization_cases.png` panel is rendered from reviewed output
bundles under `docs/_static/qi_readme_cases`.

The constrained-QI sweep is the compact bundled-seed matrix, not the staged
far-seed runner.  If its summary reports a stale QI target aspect, rerun the
two sweep commands above with the current target-6 policy before using the
rendered matrix.
Read the docs NFP=4 QI coverage row as a minimal-seed same-NFP reference-family
proposal with an exact audit, not as a long local descent.  The generated
`docs/_static/figures/readme_qi_optimization_cases.csv` row should remain
`validation_status=case-gated` and `expected_gate_status=candidate`; passing
gate fields do not make the row an aspect-6 README best row or a common-minimal
completion.

For publication-quality QI validation, re-run the diagnostic with higher
`QI_MBOZ`, `QI_NBOZ`, `QI_NPHI`, `QI_NALPHA`, and `QI_N_BOUNCE`, then check
that the smooth-vs-legacy ranking, component totals, mirror ratio, elongation,
and LCFS `|B|` contours remain stable.

For seed-robustness experiments, first run the audit, then use
`audit_qi_seed_suitability.py --prefine-probes plan` to write a reviewed
manifest before launching expensive prefine probes.

For a new external VMEC input deck, set `INPUT_FILE = Path("/path/to/input.my_seed")`
at the top of `QI_optimization.py`.  If the seed is far from QI, first run
`audit_qi_seed_suitability.py`, then enable `USE_REFERENCE_FAMILY_SEED` with a
same-NFP reference or use the target-helicity seed helper before local
optimization.  `QI_seed_robustness.py` remains available for small diagnostic
probes, but the recommended user-facing QI path is the editable
`QI_optimization.py` workflow.

`tools/diagnostics/qi_landscape_scan.py` and `qi_basin_survey.py` are useful for
mapping rugged seed neighborhoods, but their default executed diagnostics use
trial solves for speed. Add `--exact-solve` before using scan values as
promotion evidence.

## Sweep And Rendering Tools

- `generate_qs_ess_sweep.py`: CPU/GPU QA/QH/QP/QI policy sweep over mode continuation, ESS, and maximum boundary mode.
- `render_qs_ess_publication_panel.py`: render full-sweep docs assets from sweep outputs: objective histories over all stages, initial/final 3D atlases, initial/final VMEC-angle LCFS `|B|` line-contour atlases, and wall-time/status summary tables.
- `render_readme_best_optimizations.py`: render only the compact README best-row figures and CSV table.
- `render_qi_readme_cases.py`: render the NFP=1-4 QI docs coverage figure and CSV from existing `QI_optimization.py` outputs, using Boozer `|B|` line contours only.
- `render_qi_constrained_sweep.py`: render QI-focused constrained-sweep diagnostics.

Example:

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/generate_qs_ess_sweep.py \
  --backend-label cpu --solver-device cpu --policy continuation \
  --problems qa,qh,qp,qi --modes 1,2,3 --ess both --qi-qp-preseed off --rerun
PYTHONPATH=. python examples/optimization/render_qs_ess_publication_panel.py
```

Use `--modes 1,2,3,4` only for exploratory high-mode regeneration; checked-in
docs snapshots currently contain partial/archived `max_mode<=3` rows, not a
complete reviewed CPU/GPU matrix.

Keep generated full-sweep atlases, PDFs, and bulky report panels in ignored
result directories or release assets until reviewed.  Checked-in PNG/JPEG
figures under `docs/_static/figures` are expected to stay compressed below
2 MiB each.

## Comparison And Diagnostic Scripts

- `compare_omnigenity_qi_objective.py`: compare VMEC-JAX QI metrics with the legacy omnigenity implementation.
- `compare_omnigenity_qs_mode1.py`: compare low-mode quasisymmetry optimization components.
- `compare_qs_policy_matrix.py`: compare policy choices across direct/continuation and ESS/non-ESS lanes.
- `target_iota_aspect_volume.py`, `target_iota_volume.py`, `explicit_target_iota_volume.py`, and `implicit_target_iota_volume.py`: compact historical examples for API comparison.
- `qh_fixed_resolution_exact.py`: exact fixed-resolution diagnostic retained for regression and method comparisons.

## Profiling And Test Checks

Use `tools/diagnostics/profile_exact_optimizer.py` for exact optimizer callback
profiling and `tools/diagnostics/profile_fixed_boundary.py` for raw solver
throughput. Accepted-point exact callbacks default to the tape path on both CPU
and GPU; use `VMEC_JAX_OPT_EXACT_PATH=scan` only for scan-exact diagnostics.
CPU/GPU command examples live in `docs/performance.rst`. Optimization scripts
may set `METHOD = "auto"` to let vmec_jax choose a profiled optimizer method
without changing the requested device; at present this only promotes the QA
high-mode CPU/default-backend case to the matrix-free trust-region path.

For QI-specific GPU diagnostics, isolate the Boozer/QI residual from optimizer
bookkeeping first:

```bash
JAX_PLATFORM_NAME=gpu PYTHONPATH=. python tools/diagnostics/profile_qi_boozer_gpu.py \
  --solver-device gpu --repeat 2 --jit-booz \
  --output results/diagnostics/qi_boozer_gpu.json
```

Relevant lightweight tests:

```bash
pytest -q tests/test_optimization_examples.py tests/test_qs_ess_render_smoke.py
pytest -q tests/test_quasi_isodynamic.py tests/test_qi_legacy.py tests/test_qi_diagnostics.py tests/test_booz_input.py
```
