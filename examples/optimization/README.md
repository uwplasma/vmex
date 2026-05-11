# Optimization Examples

These scripts are intended to be read and modified directly.  The recommended
workflow is to instantiate a VMEC object, assemble a list of objective tuples
`(objective_function, target, weight)`, call `least_squares_solve`, and then
inspect, save, or plot the returned result.

## Recommended Standalone Examples

- `QA_optimization.py`: quasi-axisymmetric fixed-boundary optimization.
- `QH_optimization.py`: quasi-helical fixed-boundary optimization.
- `QP_optimization.py`: quasi-poloidal fixed-boundary optimization.
- `QI_optimization.py`: quasi-isodynamic optimization with Boozer-space QI metrics.
- `qa_optimization_finite_beta.py`, `qh_optimization_finite_beta.py`, and `qi_optimization_finite_beta.py`: finite-beta stage-1 examples with pressure/current-profile terms.

Run one case from the repository root:

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/QH_optimization.py
```

Set `SOLVER_DEVICE = "gpu"` inside the script, or run with
`JAX_PLATFORM_NAME=gpu`, to use a GPU-enabled JAX installation.

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
