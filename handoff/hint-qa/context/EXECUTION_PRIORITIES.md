> Archived context, 13 September 2026. The top-level handoff gives current status; older job statements below are historical. Machine locations were removed; unbundled raw data are not implied to be available.

# Execution priorities — 13 September 2026

The shortest path to the requested result is two independent work lanes: qualify HINT's 0.5% equilibrium, and qualify VMEX's own QA primal/derivative path. VMEX derivative work does not need to wait for HINT relaxation. Both ultimately feed the same comparison harness.

## Immediate integration status

SOLVAX PR104 was merged externally on 13 September at merge commit 66f97a6e0eb758af8d7f46939ffdd8ec733efbef, from the exact validated head e4b5071. All 11 CI checks and exact-source CPU/GPU contract checks passed. VMEX PR300 still requires independent approval.

VMEX PR302 is now at 14360d179af4534aa9bb638b172c8724ee289d9b. Newton refinement no longer inherits deliberately tiny adjoint Krylov dimensions. The two stale failure/underconvergence tests now exercise the intended contract. All three affected tests and 32 focused tests pass; static checks, strict docs and 98.2% changed-line coverage pass. Broader CI is running; the PR remains draft.

Both matched-restart HINT resistivity trials finished. Higher eta increases pressure-supported current but worsens force balance; these remain transient experiments, not accepted equilibria. See the resistivity assessment and raw comparison records.

The exact ns31/MPOL7/NTOR6 finite-beta QA 0.5% VMEX root passed the actual residual and geometry gate in two independent CPU processes with identical states. Matching-source GPU-environment root parity also passed (relative state difference4.18e-12). The CPU energy/boundary directional derivative screen completed: tangent/adjoint duality2.72e-12, eight certified perturbed roots, Taylor orders1.979/1.963/1.836. Small-step central differences show noise sensitivity; this is one direction, not broad gradient certification. The GPU derivative check is prepared but requires a free device.

The HINT current-profile/spatial audit locates substantial opposing current just outside pressure support. Native circulation and curl integrals agree at the retained outer contour, but inner-contour discrepancies and a band only a few grid cells wide motivate controlled spatial refinement. The prepared matched-fresh64/128 pair preserves physical constraints and common controls; native grid-changing restart is unsupported. Its t=.01 startup evidence must remain distinct from the mature t=1.08 trajectories.

## Ranked work and completion criteria

1. **Close VMEX PR302's failures.** Address the refinement/adjoint coupling and the two test-contract issues, run the affected checks, then complete required CI. This unlocks reliable QA derivative qualification and avoids optimizing against an uncertified state.

2. **Resolve HINT's current-relaxation bottleneck.** Run the two prepared eta0=0.002/0.005 trials against the existing 0.001 baseline, with identical restart, source constraints, timestep and aligned reference schedule. Measure current on every cut, reconstructed current-profile agreement, pressure support, force balance, fixed-target fields and time per useful relaxation advance. These are numerical-control experiments, not retuned physical fixtures. Any accelerated state must return to baseline resistivity and pass convergence checks.

3. **Qualify one VMEX QA root and its derivatives in parallel.** Start with the existing 0.5% case. Require repeat/cold-solve consistency and the actual-primal gate; then independently re-solved directional/Taylor and JVP/VJP checks on matching CPU/GPU source. Follow with one small feasible optimization and a cold final reevaluation. Cached snapshot parameter VJPs remain explicitly unsupported until a consistent parameterized path is implemented.

4. **Deliver one complete 0.5% comparison before expanding.** Require current/pressure constraints and time/grid/wall checks, then compare vector and plasma-response fields at the fixed targets. Check that the evaluation region is exterior to relaxed current support. Continue LHD reference resolution work within the bounded CPU budget as an independent verification lane. Expand to 2.5% using the validated procedure and preserve its known coil/WOUT mismatch explicitly.

5. **Finalize topology and uncertainty accounting.** Apply the common tracer only after field/equilibrium gates pass. Report uncertainty from spatial resolution, quadrature, interpolation, relaxation controls and wall placement separately from inter-code differences.

## Work to defer

Avoid another large unqualified relaxation campaign, repeated broad literature searches, new high-order/Krylov methods, broad solver refactors, or large optimization sweeps until a specific failing gate requires them. Reuse PR299's existing performance evidence after reconciling its state/derivative contract with PR302. Measure total cost of an accepted equilibrium or gradient, not just a faster kernel.

The cadence experiments are coupled relaxation transients, not proof of temporal order. In particular, 250 steps changes velocity-reference intervals to 100/100/50; 200/500/1000 align those intervals. None has produced the required QA equilibrium: attained pressure-supported current remains about -17.69 kA versus the -95.78 kA target.
