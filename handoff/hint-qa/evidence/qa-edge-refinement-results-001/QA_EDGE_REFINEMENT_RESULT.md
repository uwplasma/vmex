# Matched fresh QA edge refinement

The pair tests early startup sensitivity at fixed magnetic time, not equilibrium convergence. Native HINT rejects a restart with changed grid dimensions, so both grids start fresh from their own validated vacuum/flux/limiter inputs. No field remapping is used.

The source is maintained HINT commit `bf31fc39f7179bdd91d84319c51c68e2f6fff25f`; the measured native executable SHA256 is `56faef4649083c9cb06c10764c2cd63bad78ce00d1576004f5a9fd3dcab18fb6`. Both cases use four MPI ranks, one OpenMP thread, four outer steps, 250 magnetic substeps per outer step, `dt_b=1e-5`, `eta0=.001`, no periodic pressure reset, and target plasma current −95778.34986522896 A. The grid pair is 64×64×32 and 128×128×64, NFP2, same physical domain. Exact decks and input hashes are preserved in the [coarse](../../inputs/edge-n64/hint.input) and [fine](../../inputs/edge-n128/hint.input) bundled decks.

Both runs completed successfully: 108.701s coarse and 1041.855s fine. Both reached magnetic time .01. All solver jobs in this pair are finished; no further simulations are part of this result collection.

## Reproduce diagnostics from retained native outputs

Set `PYTHON` to an environment with NumPy, SciPy, netCDF4 and matplotlib, then run from the HINT repository root. Required native run products are `hint.nc`, `vacuum.nc`, `flux.nc`, and `limiter-12.nc`; large products must be separately retained or reproduced using the hashed inputs. These commands perform postprocessing only:

```sh
for RUN in qa-edge-fresh-n64-001 qa-edge-fresh-n128-001; do
  "$PYTHON" local-support/summarize_qa_pilot.py "runs/$RUN" --target-current -95778.34986522896
  "$PYTHON" local-support/diagnose_qa_current_profile.py "runs/$RUN" --profile-csv qa-source/beta0p5/current_mapping.csv --target-current -95778.34986522896 --output "runs/$RUN/current-profile-diagnostic-001.json"
  "$PYTHON" local-support/diagnose_qa_return_current.py "runs/$RUN" --output "runs/$RUN/return-current-diagnostic-001.json" --plot "runs/$RUN/return-current-section-001.png"
done
"$PYTHON" local-support/compare_qa_edge_refinement.py . --reference-run qa-edge-fresh-n64-001 --candidate-run qa-edge-fresh-n128-001 --output local-support/qa-edge-refinement-comparison-001.json
"$PYTHON" local-support/sample_qa_cadence_targets.py . --reference-run qa-edge-fresh-n64-001 --candidate-run qa-edge-fresh-n128-001 --sample-tag edge-refinement-targets-001 --output qa-edge-refinement-comparison-001.json
```

Diagnostic helpers require fresh output names; preserve existing reports. Native target sampling additionally requires the built `build-release/MAGVAL/magval_points.exe` and `runs/qa-native-inputs/exterior-targets.npz`.

Current integrals over the whole retained box need not equal the plasma-current target. Their boundary circulation constrains the integral and may permit an opposing return-current region. Pressure-support current, current-profile agreement, force residual, and spatial current support must be interpreted together. A reconstructed snapshot parallel drive is not the exact drive from the beginning of the native magnetic substeps; full current minus parallel drive contains required pressure-driven perpendicular structure and is not a zero target.

## Completed result

| Metric at magnetic time .01 | 64×64×32 | 128×128×64 |
|---|---:|---:|
| P>0 response toroidal current [A] | −310.316 | −257.749 |
| Combined P>0 force ratio | .984952 | .971483 |
| Maximum pressure [Pa] | 133503.759 | 133128.053 |
| Main positive return-current shell | 1–2cm: +287.000A | 0–1cm: +240.799A |
| Solver runtime [s] | 108.701 | 1041.855 |

The shell-width shift tracks the mesh spacing: coarse R/Z spacings are approximately 1.46/1.82cm and fine .72/.90cm. This is evidence of grid-scale structure during startup. It does not establish that mature return current vanishes under refinement, nor establish temporal or spatial convergence. Both attained plasma currents remain below .4% of the prescribed target and both force ratios remain near unity.

At the same 192 physical targets, native sampled **total-field** differences have RMS .243516mT and maximum 1.701148mT. These include the changed vacuum-grid interpolation; the separately observed coarse vacuum interpolation error is of comparable scale. No independently matched stored vacuum target arrays were found in the vacuum-grid output tree, so this result does not isolate plasma-response error or establish plasma-response convergence. Both samples used the same target input SHA256 `c2c38f8349b9aec38d351403f90cd3746e86797d96261c3b3a78288d13b6f9f5` and sampler SHA256 `a70b83f3fb1a7ed15bd4e14e9fddc7004f4f8159c20388e801bee42afad9b252`.

Portable compact evidence is in this directory: comparison, per-run summaries, current-profile and spatial shell reports, native total-field point arrays, plots, sanitized job status and input provenance. Runner status records inherited OMP=2, but its command explicitly sets OMP=1 before launching MPI/HINT. This distinction is retained in each status note. Raw native fields are not embedded in the compact handoff.

The next scientific step requires an explicit resource decision: continue both validated native grids to a common later magnetic time and compare response-only fields at common physical targets, with pressure/current/force histories and shell widths in meters. The current startup pair alone cannot select resistivity or certify equilibrium. No continuation is launched here.
