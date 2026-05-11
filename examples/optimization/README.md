# Optimization Examples

These scripts are intended to be read and modified directly.  The recommended
workflow is to instantiate a VMEC object, assemble a list of objective tuples
`(objective_function, target, weight)`, call `least_squares_solve`, and then
inspect, save, or plot the returned result.

## Objective Tuple Pattern

Use explicit SIMSOPT-style tuples:

```python
aspect = vj.AspectRatio()
iota_floor = vj.AbsMeanIotaFloor(0.41)
qs = vj.QuasisymmetryRatioResidual(helicity_m=1, helicity_n=-1, surfaces=SURFACES)

problem = vj.LeastSquaresProblem.from_tuples(
    [
        (aspect.J, TARGET_ASPECT, ASPECT_WEIGHT),
        (iota_floor.J, 0.0, IOTA_FLOOR_WEIGHT),
        (qs.J, 0.0, QS_WEIGHT),
    ]
)
```

`weight` follows SIMSOPT semantics: the residual is
`sqrt(weight) * (objective - target)`.  Do not pre-scale callbacks.  QI terms
use the same tuple form, but encode QI thresholds and smoothing in
`QuasiIsodynamicOptions`, `MirrorRatio`, and `MaxElongation`; QI tuple targets
should remain `0.0`.

## Recommended Standalone Examples

- `QA_optimization.py`: recommended quasi-axisymmetric fixed-boundary optimization.
- `QH_optimization.py`: recommended quasi-helical fixed-boundary optimization.
- `QP_optimization.py`: quasi-poloidal fixed-boundary optimization from the NFP=2 QI seed.
- `QI_optimization.py`: recommended quasi-isodynamic optimization with Boozer-space QI metrics, mirror-ratio and elongation penalties, repeated same-mode continuation, and ESS.
- `qa_optimization_finite_beta.py`, `qh_optimization_finite_beta.py`, and `qi_optimization_finite_beta.py`: finite-beta stage-1 examples with pressure/current-profile terms.

Run one case from the repository root:

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/QH_optimization.py
```

Set `SOLVER_DEVICE = "gpu"` inside the script, or run with
`JAX_PLATFORM_NAME=gpu`, to use a GPU-enabled JAX installation.

## QI Diagnostics

Before treating a QI result as a final candidate, audit the smooth objective
against the legacy branch diagnostics and render the constrained QI matrix.
The first-class record helpers are `vj.QIDiagnosticOptions`,
`vj.qi_diagnostics_from_boozer_output`, and `vj.qi_diagnostics_from_state`.

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/compare_omnigenity_qi_objective.py
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/generate_qs_ess_sweep.py \
  --backend-label cpu --solver-device cpu --policy continuation \
  --problems qi --modes 1,2,3 --ess both --qi-qp-preseed both
PYTHONPATH=. python examples/optimization/render_qi_constrained_sweep.py
```

For publication-quality QI validation, re-run the diagnostic with higher
`QI_MBOZ`, `QI_NBOZ`, `QI_NPHI`, `QI_NALPHA`, and `QI_N_BOUNCE`, then check
that ranking, mirror ratio, elongation, and LCFS `|B|` contours remain stable.

## Sweep And Rendering Tools

- `generate_qs_ess_sweep.py`: CPU/GPU QA/QH/QP/QI policy sweep over mode continuation, ESS, and maximum boundary mode.
- `render_qs_ess_publication_panel.py`: render the large optimization atlas and summary tables from sweep outputs.
- `render_readme_best_optimizations.py`: render the compact README figures and CSV table.
- `render_qi_constrained_sweep.py`: render QI-focused constrained-sweep diagnostics.

Example:

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/generate_qs_ess_sweep.py \
  --backend-label cpu --solver-device cpu --policy continuation \
  --problems qa,qh,qp,qi --modes 1,2,3 --ess both
PYTHONPATH=. python examples/optimization/render_qs_ess_publication_panel.py
```

## Comparison And Diagnostic Scripts

- `compare_omnigenity_qi_objective.py`: compare VMEC-JAX QI metrics with the legacy omnigenity implementation.
- `compare_omnigenity_qs_mode1.py`: compare low-mode quasisymmetry optimization components.
- `compare_qs_policy_matrix.py`: compare policy choices across direct/continuation and ESS/non-ESS lanes.
- `target_iota_aspect_volume.py`, `target_iota_volume.py`, `explicit_target_iota_volume.py`, and `implicit_target_iota_volume.py`: compact historical examples for API comparison.
- `qh_fixed_resolution_exact.py`: exact fixed-resolution diagnostic retained for regression and method comparisons.

## Profiling And Test Checks

Use `tools/diagnostics/profile_exact_optimizer.py` for exact optimizer callback
profiling and `tools/diagnostics/profile_fixed_boundary.py` for raw solver
throughput.  CPU/GPU command examples live in `docs/performance.rst`.

Relevant lightweight tests:

```bash
pytest -q tests/test_optimization_examples.py tests/test_qs_ess_render_smoke.py
pytest -q tests/test_quasi_isodynamic.py tests/test_qi_legacy.py tests/test_qi_diagnostics.py tests/test_booz_input.py
```
