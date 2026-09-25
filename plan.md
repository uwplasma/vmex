# VMEX research plan

## Force-balance recovery lane (started 2026-09-24)

The polishing recovery work now follows the separate reviewed handoff
"VMEX force-balance recovery: implementation and research handoff".  It does
not replace the unrelated lanes retained below.  Work is on
`rj/force-balance-recovery` from `4632dad8` and remains unmerged pending
Rogerio's review.

### R7 continuation (2026-09-24): the stationarity floor was evaluation noise

Resumed from the R7 review handoff at PR head `131580da` (unchanged since the
review). All five checkpoint/input hashes verified. Records are in
`artifacts/r7/` and `benchmarks/polish_recovery_r7_*`; generators are
`benchmarks/r7_*.py`. Environment: Apple M4 CPU, float64, Python 3.14.6,
JAX 0.11.2, NumPy 2.5.3, SciPy 1.18.1, SOLVAX 0.26.0; `np.longdouble` has
52 mantissa bits here (no extra precision).

Diagnosis table (frozen R6 state `bac2fb88...`):

| Hypothesis | Measurement | Verdict |
|---|---|---|
| F1 replay: saved scale bypassed | saved and recomputed scales bitwise equal at this state (`frozen-audit.json`) | real latent defect, fixed + regression test; not the plateau cause |
| Gradient-path ordering | jit vs eager `||P dg||` = 3.2e-11 (gate threshold 5.56e-8) | not the cause; this check badly understates the noise |
| Input rounding of O(1) coefficients | one-ulp perturbation on half the stored coefficients moves `||Pg||` by 7-11e-7 (13-19x the threshold) | dominant floor |
| Internal arithmetic of the jet contraction | after exact first-order compensation of the ulp change, 1.3-1.9e-7 remains with the assembled derivative tables, 1.3e-12 with coefficient-first differentiation | second floor, removed by the stable kernel |
| Taylor consistency (legacy) | `T_g(h)` flat at 3-5e-7 for h in [1e-4, 4] for Newton and random feasible directions; Newton linear term 1.2e-7 is buried | Newton could not make progress |
| Taylor consistency (accurate) | `T_g` floor 1e-12; random direction shows clean h^2 scaling; full Newton step eta 9.8e-8 -> 1.9e-13, observed objective change -2.04e-22 vs model -2.38e-22 | resolved |
| F2 local/global factor 0.5 | raw steps from the R5 input: norm ratio 0.9995, cosine 0.999998; accepted fractions 1.0 vs 0.5 (`f2-raw-step-comparison.json`) | closed: a line-search artifact of noisy acceptance, not a solver difference |

Consequence for earlier records: the R6 value `2.22e-8` was a selection-biased
draw from evaluator noise (the merit search kept the lowest noisy trial). Under
the accurate evaluator that state has eta `9.8e-8`. R6 stationarity numbers are
not measurements of the true projected gradient and must not be compared with
R7 numbers.

Implemented (opt-in, default behaviour and historical replay unchanged):

- `make_variational_plan(..., stable_derivatives=True)` builds coefficient-first
  tables: q_s and q_ss from `diff(c) * p/(t[i+p+1]-t[i+1])` evaluated in lower
  degree, then the unchanged axis chain rule with explicit m=0/1 limits;
  multiplicities that break C2 are refused.
- In that mode `native_physical_force_residual` synthesizes jets of the base
  state and of `scale*c` separately and adds jets; it never forms the rounded
  coefficient sum. The certified object is the `(initial, accepted_coordinates)`
  pair; checkpoints carry `evaluation_mode = coefficient-first-split-jets`.
- Driver: same-chart continuation reuses the loader's base/plan/layout/gauge/
  stored scale (F1); `--stable-derivatives/--no-stable-derivatives`;
  `--output-steps` saves raw steps and accepted fractions (F2).

R7 result (`benchmarks/polish_recovery_r7_basis191_stationary.json`, state
`polish_recovery_r7_basis191_stationary_state.npz`): one exact-Hessian KKT
Newton step (3 GMRES iterations) gives eta `1.86e-13` (in-loop jit gradient) and
`5.88e-12` (final eager VJP path), both far below `1e-8`; independent point
epsilon_B `9.5829507294e-6` (56.687 N/m^3), gauge `1.5e-18`, minimum signed
Jacobian 23.84, radial refinement difference 1.88e-7. 142 s wall, 4.4 GB peak
RSS for 3 iterations + certificate. A fresh-process replay reproduces eta
`1.44e-13` (`replay-check.json`).

**Decision (Rogerio, 2026-09-25): option (a) accepted.** The certified object
in the accurate mode is the stored `(initial, accepted_coordinates)` pair;
driver records carry `stationarity_certified_object`. The float64 export's eta
is reported as representation-limited. Original note: the exported
ordinary float64 coefficient file (the rounded sum) has eta `5.3e-8` under the
accurate evaluator. That is its representation floor (0.5 ulp times the
measured sensitivity), not a solver defect. Force is identical to ten digits.
Options: (a) declare the certified object the stored double-word pair (schema
already carries both parts) and report the exported-file eta as
representation-limited; (b) keep eta<=1e-8 on the float64 file, which requires
a different metric scaling or state parameterization; (c) other. The status is
`stationarity_pass=true` for the pair and `representation_limited` for the
exported file until you decide.

R7.3/R7.4 progress (2026-09-25, head after `c3683426`):

- One gauge-projector LU per chart (keyed on exact CSR bytes; tested reuse and
  invalidation). The exact stationarity loop stops once eta <= 1e-8.
- `benchmarks/polish_recovery_cold.py` runs the whole polish from the input
  deck in fresh processes with a dedicated compilation cache. Stage records:
  `artifacts/r7/cold-1` (cubic), `cold-q5` (quintic, empty cache),
  `cold-q5-reload` (quintic, populated cache).

| Cold workflow (M4 CPU, float64) | cubic historical schedule | compact quintic schedule |
|---|---|---|
| stages after ordinary solve | 20 (3 dense) | 6 (all sparse) |
| final coordinates / basis | 11039 / 191 | 4369 / 76 |
| independent epsilon_B | 9.4949e-6 | 7.6771e-6 (45.41 N/m^3) |
| eta (certified pair) | 1.0e-11 | 4.0e-10 |
| gauge / min signed Jacobian | 2.2e-16 / 23.84 | 7.6e-20 / 23.84 |
| radial refinement difference | 1.8e-7 | 1.6e-5 |
| wall, empty cache | 2674 s (early stages overlapped other probes) | 352 s |
| wall, cache reload | not run | 230 s |
| max stage peak RSS | 8.1 GB (dense refine) | 5.6 GB (lift) |
| ordinary VMEX solve alone | 13.0 s | 5.9 s cold / 3.1 s reload |

The quintic path is the recommended default candidate. It is not a matched
comparison of equal accuracy (it ends more accurate). It still costs ~60x an
ordinary solve (~75x on reload), so the "few ordinary solves" target is not met.
The remaining time is assembly (local-normal JVP sweeps) and the point
certificate, not compilation. The quintic state is
`benchmarks/polish_recovery_r7_quintic_cold_stationary_state.npz`.

Next resume point: R7.3 cost reduction on the quintic path (active-local
coefficient gathers, a static CSR scatter, compiled-identity reuse), a cheaper
certificate schedule, then R7.5 (3-D, current closure, LASYM) and R7.6
implicit derivatives. Earlier text:
 (active-local assembly, one frozen
projector factor, stable jit identities), then R7.4 cold-input + quintic
comparison, R7.6 implicit derivatives on the accurate evaluator.


- P0 reproduced the exact bundled `input.shaped_tokamak_pressure` (SHA-256
  `5b2740db...fe5102`) in an isolated current dependency environment.  The
  ordinary solve converged through NS=51 in 10.79 s cold/cache-populating and
  1.80 s cache-reload; raw evidence is in
  `benchmarks/polish_recovery_p0.json`.
- The first P1 change replaces inverse-power reconstruction with a weighted,
  column-equilibrated fit of physical amplitudes.  Its tests include noisy
  high-m near-axis data, rank refusal, curvature, endpoint constraints, and
  fit diagnostics.
- A direct native correction layout now retains every symmetry-allowed R, Z,
  and lambda spline coefficient independently of the legacy transfer image.
  It eliminates fixed R/Z edge coefficients and the angle-independent lambda
  gauge structurally while leaving lambda free at the edge.  Tests preserve
  both normal directions at cross-section extrema and verify a smooth,
  boundary-fixed relabeling has zero fixed-label displacement and energy
  variation.
- The first gauge-qualified variational equation is a bordered system: a
  mass-normalized full-tangent constraint is applied matrix-free beside all
  native coordinates, rather than freezing Z or building a global nullspace.
  On the P0 state its 286 constraint rows have full rank; physical-displacement
  column scaling reduces their diagnostic condition estimate from 9.3e7 to
  61.  The 1,181-variable KKT residual and matrix action run warm in about
  0.9-1.0 ms.  Nonsymmetric/LASYM paired gauge rows remain a P5 item.
- P2 now has a tensorized native fixed-pressure energy seam.  It contracts
  coefficients into values and analytic first jets on a precomputed tensor
  grid, avoiding nested pointwise AD.  On the P0 shaped-tokamak state its
  energy, full-state gradient, and Hessian-vector product each compiled in
  under 0.3 s and ran warm in under 1 ms; independent field-oracle agreement
  and raw timings are recorded in `benchmarks/polish_recovery_p2.json`.
- The off-root virtual-work identity, including lambda, agrees with the
  independent strong-force oracle.  Tensorized analytic second-jet tables now
  produce all strong-force channels on the same native state; point values,
  JVPs, and VJPs agree with the independent nested-AD oracle.  On the P0 case
  the force compiled in 0.28 s and ran warm in 0.65 ms.
- P3 has a same-state, same-space A/B reference comparison on the user
  tokamak.  With fixed `F_star = 5.915e6 N/m^3`, Candidate A reached a KKT
  residual of `3.66e-7` but an independent force certificate of
  `epsilon_B = 1.41e-3`; all its 480-iteration GMRES solves reported
  unconverged.  Candidate B reduced the independent certificate from
  `1.948e-2` to `2.809e-4` in 5.9 s, but remains 28x above the initial `1e-5`
  gate.  Evidence and limitations are in `benchmarks/polish_recovery_p3.json`.
  The B prototype explicitly forms a Jacobian and dense gauge nullspace as a
  bounded reference; it is not eligible for production promotion.
- P3's force least-squares residual now uses the plan's fixed physical volume
  scale, `sqrt(dV / V_star)`, rather than renormalizing by each state's own
  volume.  On the baseline quadrature, integrated volume is
  `633.7993467060257 m^3` versus `V_star = 633.7993467060758 m^3` (8e-14
  relative), so this corrects the moving-scale definition without changing the
  recorded comparison at displayed precision.  Input pressure and prescribed
  iota are also sampled from their analytic input profiles at spline nodes;
  current-constrained chi remains an equilibrium unknown.
- P4 rejects unregularized radial enrichment on the same ordinary WOUT.  The
  independent force RMS grows from `1.152e5` to `2.315e5` and `1.005e6`
  N/m^3 as the cubic basis grows from 27 to 35 and 43 functions; near-axis RMS
  grows most sharply, despite a modest improvement in the window magnetic
  relative metric.  A 51-function basis is correctly refused at rank 49/50.
  Full regional data and the lift-only limitation are in
  `benchmarks/polish_recovery_p4.json`.  Next: identify a support-aware,
  regularized lift before enlarging the correction space.
- P5 prototypes optional knot-span-scaled curvature regularization in the
  lift; zero remains the default, and the observed-data rank check still runs
  before regularization.  At 43 cubic basis functions, weight `0.1` lowers
  independent force RMS from `1.005e6` to `7.36e4 N/m^3` and near-axis RMS
  from `5.13e6` to `3.86e4`, with radial quadrature sensitivity falling to
  `2.7e-11`.  It moves residual toward the edge, increases the window
  magnetic-relative metric, and still has `epsilon_B = 1.24e-2`; this is a
  lift-only result, not a recovered equilibrium.  See
  `benchmarks/polish_recovery_p5.json`.  Proceed to a bounded same-scale
  correction-step test before deciding whether the regularizer helps the
  polish objective.
- P6 runs that bounded correction-step check.  Five projected-gradient steps
  reduce fixed-scale `epsilon_B` from `1.244e-2` to `8.891e-3` in 6.8 s
  including the independent certificate; gauge residuals remain below
  `3e-22` and the solve-grid signed Jacobian stays positive.  This is only a
  28.5% improvement and is about 889x above the `1e-5` target; P3's dense
  Candidate B still outperforms it decisively.  The reproducible diagnostic
  script and complete numbers are `benchmarks/polish_recovery_p6.py` and
  `benchmarks/polish_recovery_p6.json`.  The linear step itself is fast enough
  for iteration, but the next candidate must use scalable constrained
  least-squares rather than promote projected steepest descent.
- P7 implements a symmetric bordered Gauss--Newton action and a matrix-free
  GMRES step, verified against the dense small-system action and true linear
  residual.  On the 1,439-coordinate/462-gauge P0 lift, 40 unpreconditioned
  iterations leave relative true residual `7.36e-3` against a `1e-4` request.
  A 16-probe Hutchinson diagonal is indefinite (minimum `-4.79e4`) and worsens
  the residual to `0.437`; neither step is accepted.  Full results and a
  reproduction script are `benchmarks/polish_recovery_p7.json` and `.py`.
  Next gate: derive and test a physically structured mode/radial block
  preconditioner before raising the iteration budget.
- Review found that historical P8/P9 confused full coefficient-table indices
  with packed solver positions: 935 of 1,439 indices were out of packed bounds
  and 1,397 positions were mismatched. Their original residuals (`0.6590` and
  `1.766`) are retained as invalid method-selection evidence, not conclusions
  about correctly implemented block/Schur methods. R0 repairs the partition,
  validates it for symmetric/LASYM layouts and against small dense Jacobian
  blocks, and replaces explicit inverses with Cholesky solves. On the same
  endpoint and 40-iteration budget, corrected P8/P9 residuals are `0.02899`
  and `0.02220` against `1e-4`; both steps remain rejected, but improve the
  invalid historical values by about 22.7x and 79.5x. Evidence is in
  `benchmarks/polish_recovery_r0.json`.
- Historical P10 tests at most four nonlinear damped GN iterations, with linear request
  `1e-9` and up to 1,600 Krylov iterations per solve, plus force descent,
  gauge norm `<1e-6`, and positive-J acceptance gates. The first solve's true
  linear residual is `3.04e-5`. Its full step lowers force residual norm
  `0.012436` to `0.0003245` and retains positive minimum signed Jacobian
  `1.35047`, but gauge residual is `1.711e-4`; the half/quarter/eighth steps
  improve gauge but give force norms `0.006227/0.009331/0.010884`. No trial
  satisfies all gates, so the step is rejected and the accepted state remains
  the base smoothed lift (`epsilon_B=1.244e-2`, force RMS `7.36e4 N/m^3`).
  Do not report the inadmissible full-step force number as polish progress.
  See `benchmarks/polish_recovery_p10.py` and `.json`.
- R0 also repairs P10's fail-open predicate: the script had recorded the true
  linear residual but did not gate acceptance on it. The replay still rejects
  every trial, now independently because `3.04e-5` misses its requested
  `1e-9` linear tolerance and the full-step gauge residual misses `1e-6`.
  The profile test now constructs its WOUT fixture in memory rather than
  depending on `artifacts/p0/run2`, and direct analytic profile replacement
  fails explicitly for unsupported nonzero `GAMMA`. The original P3 generator
  and complete native endpoint are absent from the branch/history/artifacts;
  reproduce them prospectively rather than inventing missing settings.
- That prospective P3 B reproduction is now complete and checkpointed by
  `benchmarks/polish_recovery_p3.py`. It reconstructs the reported 895 native
  coordinates, 286 gauge rows, 609-dimensional feasible subspace, and
  14,112-by-895 dense reference Jacobian. Four full feasible steps reach
  solve-grid residual `2.788195e-4`, original-row gauge norm `1.33e-19`, and
  independent `epsilon_B=2.825111e-4` (`1671.18 N/m^3`). This is within 0.6%
  of historical P3 B's independent `2.809002e-4`, while making no claim of
  byte-for-byte identity with the lost generator. The complete initial and
  accepted native states, knots, profiles, scale, coordinates, and gauge-row
  construction data are saved in the 39 KB checkpoint
  `benchmarks/polish_recovery_r1_p3_state.npz`; provenance and phase timings
  are in `benchmarks/polish_recovery_r1_p3.json`. The independent radial
  refinement difference is `0.0131`, so this recovered endpoint is a starting
  point for exact continuation, not a final accepted equilibrium.
- R3's first true radial refinement now exactly bisects all 24 old spans by
  Boehm knot insertion, producing 51 cubic basis functions without consulting
  the WOUT. On a third grid, native values/first jets agree at roughly
  `1e-14`; the most cancellation-sensitive spline second-derivative and force
  comparisons agree to `4.74e-8` and `3.47e-8` relative (`5.09e-4 N/m^3`
  maximum absolute force difference). Four full, rank-checked feasible dense
  steps on the 1,711-coordinate/550-gauge system reduce independent
  `epsilon_B` from `2.825e-4` to `1.669e-4` (`987.18 N/m^3`) while retaining
  original-row gauge norm `1.24e-19` and positive geometry. Progress occurs
  almost entirely on the first refined step and then plateaus; the result is
  still 16.7x above `1e-5`. The 386 MB dense Jacobian is bounded reference
  algebra only. Generator, complete checkpoint, transfer invariants, timings,
  and limitations are in `benchmarks/polish_recovery_refine.py`,
  `benchmarks/polish_recovery_r3_refinement.json`, and
  `benchmarks/polish_recovery_r3_refined_state.npz`. Next diagnose the
  feasible residual range and run the separate angular/refinement controls;
  do not reset to the WOUT or claim the radial result passes R3.
- The final 51-basis radial state is genuinely stationary in its feasible
  space: the last projected-stationarity measure is `6.09e-9` relative and
  the linearized unreachable-residual fraction is `0.99999999998`. Exact
  zero-padding from axisymmetric `m_max=11` to `15` then opens 404 additional
  feasible directions. Four full rank-checked steps lower independent
  `epsilon_B` to `1.127842e-4` (`667.17 N/m^3`) with gauge norm `1.57e-20`,
  but again reach an essentially stationary plateau. Padding four more modes
  through `m_max=19` gives only a 0.9% first-step improvement, to
  `1.117327e-4` (`660.95 N/m^3`), and its linearized unreachable fraction is
  already `0.991`. The latter bounded reference consumes 1.82 GB in explicit
  Jacobian/reduced-Jacobian arrays, close to the 2 GiB cap; do not enlarge the
  dense experiment further. Records and checkpoints are
  `benchmarks/polish_recovery_r3_angular*.json/.npz`, generated by
  `benchmarks/polish_recovery_angular.py`. This shows that adding modes through
  15 materially helps, while modes 16--19 are nearly saturated. Continue with
  targeted radial support diagnostics or matrix-free feasible projection,
  not an unbounded dense angular sweep.
- The first targeted radial-support continuation is complete. Starting from
  the `m_max=15`, 51-basis checkpoint, the generator ranks the 48 radial spans
  by their volume-weighted force-squared contribution and exactly inserts
  midpoints in the eight largest spans (indices 0--4 and 10--12). This grows
  the basis to 59 functions, with 2,683 coordinates, 870 gauge rows, and a
  1,813-dimensional feasible space. Four full feasible steps lower the
  independent certificate from `1.127835e-4` to `6.444610e-5`
  (`381.23 N/m^3`) while retaining original-row gauge norm `3.26e-20` and
  positive geometry. The last projected-stationarity measure is `9.82e-9`
  relative and its unreachable-residual fraction is effectively one, so this
  space is again converged rather than merely stopped early. Exact-transfer
  force disagreement is `3.58e-9` relative (`7.92e-6 N/m^3` maximum), and
  independent radial-grid disagreement is `1.81e-5`. This is the strongest
  result in the recovery lane, but it remains 6.44x above the `1e-5` gate and
  is not a recovered equilibrium. Runtime was 150.74 s, dominated by 119.39 s
  for the explicit Jacobians and 22.59 s for dense solves. The complete record
  and restart state are `benchmarks/polish_recovery_r3_adaptive8.json` and
  `benchmarks/polish_recovery_r3_adaptive8_state.npz`.
- A proposed second adaptive batch was stopped on request after about 30 s,
  during compilation of the gauge-constraint Jacobian and before any result or
  output artifact was written. For the anticipated 67-basis system, the full
  and reduced explicit force Jacobians alone are estimated at 2,054,307,840
  bytes (1.913 GiB), leaving too little room below the 2 GiB experiment cap for
  comfortable dense continuation. The generator now accounts for both arrays
  in its guard. On resumption, first verify dimensions and process memory, then
  prefer a matrix-free feasible/projected operator over blindly restarting the
  dense adaptive-16 run. Do not cite the interrupted attempt as a measurement.
- R4 implements the scalable continuation requested by the revised handoff.
  `native_force_jacobian_sparsity` derives the compact radial-support pattern
  and a collision-free analytic coloring for the native force Jacobian.  The
  new `polish_recovery_sparse.py` reconstructs the compressed matrix with
  SOLVAX, verifies independent random `A x` and `A.T y` products, solves the
  equality-constrained sparse normal KKT system with a true residual and
  original-row gauge certificate, and keeps nonlinear force descent, positive
  geometry, and independent certification as separate fail-closed gates.  It
  refuses unsupported non-axisymmetric or non-fixed-profile checkpoints.  The
  dense reference paths now perform their memory admission check before QR or
  duplicate reduced-Jacobian allocation, reuse the already formed reduced
  Jacobian, report complete linear certificates, and recompute final
  stationarity at the saved endpoint.  Checkpoint schema 2 records the exact
  knots, modes, scales, solve grid, gauge reference, and model scope; loaders
  validate those fields rather than inferring them silently.
- On the unchanged adaptive-8 state, R4 used 184 colors to recover an
  8,012,940-entry sparse force Jacobian. Random forward and transpose product
  errors were `6.83e-16` and `1.60e-15`; the sparse step's true KKT residual
  was about `1.20e-13`. Compressed assembly took 0.80 s versus 119.39 s for the
  earlier dense-Jacobian category, and factor/solve took 0.09 s. The step was
  correctly rejected because it changed the force objective by only
  `1.4e-10` relative. The independent certificate remained
  `6.44460965e-5`. The explicit sparse value/index estimate was 96.5 MB, but
  measured process high-water RSS was 7.19 GB; therefore the recorded 2 GiB
  guard is an explicit-array admission estimate, not an end-to-end RSS cap.
  Evidence is `benchmarks/polish_recovery_r4_sparse_current.json`.
- Exact adaptive continuation then produced independently point-certified
  values `6.291283e-5` (basis 63), `5.300104e-5` (71), `3.865855e-5` (83),
  `3.007712e-5` (95), `2.605327e-5` (107), and `2.394948e-5` (119).  A
  same-state order-6 gauge reanchoring at basis 83 changed epsilon by only
  `2.1e-11`, so the improvement is not a chart artifact.  Angular enrichment
  at basis 119 reduced epsilon to `1.891401e-5` at `m_max=17` and
  `1.840213e-5` at `m_max=19`; the small second gain supports holding
  `m_max=19` while continuing targeted radial refinement.  Further exact
  radial batches gave `1.726557e-5` (basis 131), `1.539817e-5` (143), and the
  latest fully independent result `1.393733e-5` (155).  The last run took
  349.88 s because its point-oracle certificate alone took 292.33 s, exceeding
  the five-minute incremental guideline; it is retained as valid evidence,
  not as an acceptable iteration workflow.  Peak process RSS across this
  campaign reached roughly 14.1 GB despite smaller explicit-array estimates.
- To keep subsequent increments bounded, bases 167, 179, and 191 used an
  explicitly provisional overintegrated tensorized certificate (radial orders
  6/8) plus independent point-force spot checks.  Their provisional epsilon
  values were `1.302665e-5`, `1.144273e-5`, and `9.582953e-6`; the last has
  force RMS `56.6875 N/m^3`, radial disagreement `5.02e-10`, point-force
  relative error `3.72e-6`, gauge norm `3.18e-13`, and positive geometry.
  This crosses the numerical target only provisionally.  It is **not** the
  plan's full independent point-oracle acceptance.  An isolated full
  certificate attempt ran for about five minutes and ended without producing
  a result file; its cause was not captured, and no pass is claimed.  The
  latest fully independent checkpoint is
  `benchmarks/polish_recovery_r4_adaptive96_state.npz`; the latest provisional
  restart is `benchmarks/polish_recovery_r4_adaptive132_provisional_state.npz`.
- On resumption, do not start another refinement batch first. Diagnose and
  bound the isolated full shifted point-oracle certificate, then independently
  certify the basis-191 checkpoint. If it does not pass `epsilon_B <= 1e-5`,
  take a bounded same-chart correction step before any further knot insertion
  and recertify. Preserve the full evidence hierarchy: tensorized certificates
  guide iteration, but only the independent point oracle can close the R3
  force gate. After that gate, the plan's 3-D/LASYM and prescribed-current
  closure, implicit derivatives, regressions/runtime gates, and public polish
  integration remain unimplemented and must be completed before promotion.
- R5 reviewed the certification handoff against the live source and reproduced
  all bundled synthetic checks under the project environment. The three frozen
  checkpoint hashes matched. The review's concrete allocation findings were
  confirmed: the old coordinate scaling formed an 8.07 GB logical
  coordinate-by-grid array at basis 191, the magnetic-pressure-gradient
  companion still used one all-grid `vmap`, and the sparse generator still
  differentiated and factorized a dense gauge matrix.
- The independent force and magnetic-pressure-gradient point evaluators now
  share the same fixed-working-set `lax.map` policy. Physical coordinate scaling
  contracts angular moments before radial basis functions, while same-chart
  replay uses the serialized scale exactly after parity validation. The frozen
  linear gauge is assembled directly from local radial support into CSR, and
  final projected stationarity uses a sparse augmented projection instead of a
  dense thin QR. At basis 59 the sparse gauge agrees with dense `jacfwd` to
  `2.28e-16` relative and assembles in 0.129 s versus 4.54 s for dense AD. At
  basis 191, scale validation plus gauge assembly took 13.20 s, produced
  976,638 nonzeros with no empty rows, and used 0.631 GB process high-water RSS.
- Certification is now failure-safe: `benchmarks/run_checked.py` records the
  process-group outcome, signal/timeout, stdout/stderr, expected outputs, and a
  sampled RSS trace; `polish_recovery_certify.py` writes an atomic incomplete
  record before the expensive phase and distinguishes force, quadrature,
  geometry, profile/boundary, stationarity, derivative, and product status.
  Candidate checkpoints are also saved before certification in the sparse
  recovery generator. The watchdog's success and deliberate-timeout paths were
  both exercised. `psutil 7.2.2` was added only to the isolated `.venv` for the
  benchmark runner; it is not a VMEX runtime dependency.
- The basis-191 state has now passed the full independent shifted point-oracle
  force gate. The default certificate completed all 360,960 fine and 240,640
  coarse real nodes in 46.65 s with sampled peak tree RSS 2.319 GB. It reports
  `epsilon_B=9.5829531391e-6`, force RMS `56.6874582 N/m^3`, radial-order
  difference `1.80e-7`, minimum signed Jacobian `23.8427`, and exact stored
  boundary/lambda-gauge checks. A second shifted phase changes epsilon by only
  `4.55e-13` relative. Increasing angular overintegration from multiplier 2 to
  3 covers 812,160 fine nodes in 84.85 s, changes epsilon by `2.78e-14`
  relative, and peaks at 2.530 GB. These are repository measurements in
  `artifacts/r5/basis191*-certificate*.json`; the force pass is narrow and
  applies only to the declared axisymmetric fixed-profile state.
- Same-chart nonlinear qualification remains open. From the certified state,
  one ordinary full sparse step passed all gates and improved the independent
  result to `9.5829507886e-6`; its state is
  `benchmarks/polish_recovery_r5_basis191_stationary_state.npz` (SHA-256
  `2794102b...9613`). A following step was rejected solely at the existing
  `1e-8` relative-descent floor. The saved endpoint's sparse projection has
  `2.02e-15` true residual, but projected stationarity is `1.14e-7` on the
  prior Frobenius scale, so the `1e-8` stationarity gate is not passed.
  A strict non-increase refinement improved this to `6.37e-8` before rejecting
  the next step; a more permissive numerical-floor experiment worsened it to
  `1.37e-7` and must not be promoted. The current code restores strict
  `1e-10` non-increase for the opt-in stationarity phase.
- An optional projected unsquared LSMR control was added without changing the
  default normal-KKT path. On the basis-191 endpoint, 80 and 400 iterations
  stopped at their caps with projected normal residuals `1.56e-5` and
  `1.75e-6`, respectively; both were fail-closed before nonlinear trials. This
  does not show a normal-equation accuracy defect—the normal KKT certificate is
  much tighter—but it records that unpreconditioned unsquared iteration is not
  yet a practical replacement.
- R6 resumed from the R5 strict-refinement endpoint, not the permissive-floor
  state. The canonical current-generator rerun reproduced the force result and
  `6.37084e-8` Frobenius-scaled projected stationarity; one force-qualified
  step was accepted and the next was rejected. The unchanged `1e-8` stationarity
  gate therefore remains open (`benchmarks/polish_recovery_r6_basis191_stationary.json`).
- R6 added an opt-in exact constrained Hessian correction. Its full Newton/KKT
  equation has a `4.22e-11` true relative residual, but every Newton line-search
  force candidate increased the force norm. The first quadratic-scaled gradient
  fallback was saved as a diagnostic candidate but worsened endpoint
  stationarity to `1.05468e-7`; do not use it as a restart or promote it. This
  is diagnostic evidence, not a qualified result; retain the R5 strict endpoint
  and the later R6 measured checkpoints for controlled restart only.
- A force-budgeted stationarity-merit search was then tried without relaxing
  the final independent-force (`epsilon_B <= 1e-5`), gauge, geometry, or
  `1e-8` stationarity gates. It only admitted trials that lowered the current
  Frobenius-scaled projected-gradient merit by at least `1e-4` relative with a
  sparse projection true residual below `1e-10`, while keeping the weighted
  force norm within `1e-5`. Four accepted tiny steps
  lowered the measure from `6.37090e-8` to `2.22050e-8`; independent point
  certificates remained `9.58295077e-6` with positive geometry. A further
  bounded two-step run found no improving candidate and wrote an unchanged
  checkpoint (same SHA-256 `bac2fb88...0cf44d`). This is a measured plateau,
  still more than twice the stationarity threshold; none of these endpoints is
  promoted. JSON/checkpoint records are
  `benchmarks/polish_recovery_r6_basis191_exact_stationarity_merit*.json/.npz`.
- The final code now fails closed if a current or trial projected-gradient solve
  has sparse-projection true residual above `1e-10`. A one-step basis-191
  replay exercised this guard successfully (current residual `2.07e-15`, all
  trial projections `1.98e-15`--`2.56e-15`), accepted no trial, and preserved
  the input SHA-256. See
  `benchmarks/polish_recovery_r6_basis191_exact_stationarity_merit_projection_guard.json`.
- R6 also implemented streamed span-local normal assembly and retained the
  global compressed sparse-A route as its reference. At basis 191, 188 spans
  and 232 colors produce a 4,440,725-nnz normal matrix without materializing
  the 63,535,266-entry global A. Independent full-domain checks report
  gradient/VJP mismatch `3.43e-13`, H/JVP mismatch `1.23e-15`, and symmetry
  defect `6.32e-17`; the estimated local arrays are 246.5 MB and measured
  process RSS was 3.73 GB for the one-step local run. Basis-59 dense-reference
  normal and gradient parity is now an automated test and passes. However,
  basis-191 local/global KKT steps are not yet equivalent in the ill-conditioned
  near-root problem (step cosine `0.9999985`, norm ratio about `0.5`), and the
  local assembly timing is not a matched performance win. Treat C4 as a
  promising memory-bounded prototype, not a replacement for the sparse-A
  reference. The 400/1500-iteration diagonally scaled LSMR controls both
  fail-closed before nonlinear trials (`4.84e-7`/`4.53e-7` projected residual).
- The R6 strict/merit outcomes and local-normal runs all remain within the
  earlier five-minute incremental-run guideline (about 30--121 s each; sampled
  process-tree peaks were 1.76 GB at basis 59 and about 3.0--6.1 GB at basis
  191 (one corrected failed invocation peaked at 2.16 GB). A small test's first
  invocation exposed benchmark-only absolute imports. Package-safe imports were
  added, and the focused dense-reference test then passed. No full repository
  test suite was run. Current open work is to explain local/global step
  sensitivity and the stationarity floor before another same-chart strategy;
  then finish a genuinely
  cold original-input workflow. Compact degree-5 comparison, 3-D/LASYM/current
  closure, exact implicit derivatives, and public workflow integration remain
  blocked behind these qualification gates. Parent-watchdog outcomes are
  summarized in `benchmarks/polish_recovery_r6_run_outcomes.json`; the PR is not
  product-qualified.
- Candidate B remains the selected recovery architecture and the declared
  axisymmetric candidate now passes independent force acceptance. Strict
  nonlinear stationarity, 3-D closure, implicit derivatives, and product
  promotion remain open. No end-to-end speedup or production-polish claim is
  made at this checkpoint.

## Current PR handoff (2026-09-24; R6 text below is superseded by the R7 continuation above)

The active draft is [PR #448](https://github.com/uwplasma/vmex/pull/448),
branch `rj/force-balance-recovery`, based on `main` at
`4632dad8261ca72756819c1c5fcd2e2ec022aeaa`. Read this force-balance section
first, then the PR description and the versioned JSON records. The PR body is
intended to be a self-contained continuation brief; the older lanes below
remain intact and are not superseded by this recovery experiment.

Environment: `.venv`, Python 3.14.6, JAX/JAXLIB 0.11.2, SOLVAX 0.26.0, BOOZ_XFORM_JAX 0.4.0,
VIRTUAL_CASING_JAX 0.0.8; Apple M4 CPU, float64. The required deck and WOUT
hashes, physical scales, and reproduction commands are recorded in the PR
body and benchmark JSON. Do not overwrite those baseline artifacts.

Latest R6 verification: Ruff, Python compilation, and `git diff --check` pass;
the new basis-59 local-normal-vs-dense-reference test passes (`1 passed,
63 deselected`). Canonical strict refinement reproduced `6.37e-8`; exact
Hessian/force-budgeted merit refinement reached `2.22e-8` after four accepted
steps, then plateaued with no candidate accepted. Independent point-force
certification remains below `1e-5`, but strict stationarity does not pass. Local
normal actions agree with full-domain JVP/VJP to around `1e-13`/`1e-15`; full
basis step equivalence with global sparse A remains unresolved. The full
repository suite has not been run. No recovery process is running.

The best current diagnostic restart is
`benchmarks/polish_recovery_r6_basis191_exact_stationarity_merit_second_state.npz`
(SHA-256 `bac2fb88...0cf44d`, `2.22050e-8` stationarity), but it is not
qualified. The third bounded run confirms a plateau and writes the same state.
Do not use the numerical-floor state or claim a stationary root. First resolve
why locally assembled and global-A KKT steps differ in amplitude near the flat
force minimum, and check whether the projected-stationarity floor is due to
derivative accuracy, chart/metric conditioning, or objective quadrature. Keep
the C4 local route experimental until the same frozen-step/converged-state gate
passes and its full RSS/timing is comparable. Then resume the cold
original-input-to-native workflow; compact degree-5, 3-D/LASYM/current closure,
exact implicit derivatives, public integration, and full regression gates still
remain. Do not promote or merge until Rogerio reviews complete physical,
derivative, runtime, memory, and regression evidence.

External context already reviewed (design context, not proof that VMEX has
equivalent behavior): [GVEC theory](https://gvec.readthedocs.io/latest/user/theory.html),
[GVEC current workflow](https://gvec.readthedocs.io/develop/tutorials/notebooks/040_current.html),
[DESC Part I](https://arxiv.org/abs/2203.17173),
[DESC equilibrium implementation](https://github.com/PlasmaControl/DESC/blob/master/desc/objectives/_equilibrium.py),
[VMEC++ numerics paper](https://arxiv.org/abs/2502.04374),
[experimental VMEC++ gauge PR #849](https://github.com/proximafusion/vmecpp/pull/849),
[JAX `custom_linear_solve` documentation](https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html),
[GVEC functional source](https://github.com/gvec-group/gvec/blob/main/src/functionals/mhd3d/mhd3d_evalfunc.F90),
and [SOLVAX sparse direct solver](https://github.com/uwplasma/SOLVAX/blob/main/src/solvax/sparse_direct.py).

This file is the complete handoff for the VMEX research programme: a
collaborator should be able to resume from it alone. It has two parts.

- **Part I, current state and acceptance gates (2026-09-22).** The
  authoritative operational plan: status, open pull requests, the maintainer's
  decisions, CI capacity, the six research lanes (A–F) with their
  acceptance gates, dependencies, and the continuation logbook. Checkpoint:
  main is VMEX 0.11.0 (#425, `780eb86e`) plus #427 (`equilibrium_from_x`
  returns the refined state the objective read, `084f6c0e`) and #428 (m=1
  family closure test for the mean-iota re-solve gap, `604e6a76`), checked
  2026-09-22 UTC. Recheck remote heads before use.
- **Part II, historical plan and logbook (2026-09-13 to 2026-09-20).** The
  previous plan kept in full: baseline evidence, root causes, the phased
  programme with its measurements and kill rules, force-balance decisions,
  former PR dispositions, the runbook, agent briefs and the execution logbook.
  Superseded sections are marked with what superseded them; disproved claims
  stay in place, marked disproved.

To resume: read Part I top to bottom, check the open-PR table against GitHub,
then continue at the first unmet gate of the lane you own. Consult Part II for
why a gate exists, what was already measured and what must not be repeated.
Older revisions (for example the 2026-09-06 plan at `f09288b3`) remain
readable through git history.

# Part I. Current state and acceptance gates (2026-09-22)

## Resume here (after 0.11.1, 2026-09-23)

**0.11.1 is released** (tag `v0.11.1`, on PyPI). It carries #427, #428, #413,
#429, #433, #431, #432, #435, #430, #441, #437, #434, #436, #438, #439, #411,
#442, #440 and #426; the CHANGELOG entry has the measured numbers. The last
six were landed from one green integration run (#444) whose tree is exactly
main's after the merges.

**Order of work for the next session (maintainer, 2026-09-23):**

1. **Make CI faster without losing tests or coverage.** No self-hosted
   runners or paid plan for now.
   - Evidence: the last full main run took 121 min wall for 254 job-minutes,
     with a 22.5-minute longest job. The wall time is queueing on the free
     plan's 20 concurrent jobs, shared across uwplasma, so reducing
     job-minutes is what helps; further sharding would not.
   - Cache installs across jobs (uv or pip cache keyed on
     `pyproject.toml`/lock): all 32 jobs reinstall their dependencies.
   - Reuse JAX compilation across CI runs with a read-mostly persistent
     compilation cache restored by `actions/cache`: populate it once per key,
     then open it read-only in test workers, which avoids the eviction-lock
     hang that got `VMEX_COMPILATION_CACHE` disabled.
1b. **Coherent dependency floors across the stack (maintainer, 2026-09-23).**
   A user should never have to upgrade packages by hand after
   `pip install "vmex[all]"`. Reported on 0.11.1: `pip install -e .` into an
   environment with equinox 0.11.11 installed solvax 0.26.0 and jax 0.10.2,
   and the first `vmex <input> --plot` crashed on import
   (`solvax -> equinox -> jax.interpreters.batching.NotMapped is deprecated`);
   upgrading equinox by hand to 0.13.8 fixed it. solvax 0.26.0 declares bare
   `equinox` and `jax` with no floors, so pip keeps an incompatible old
   equinox.
   - solvax: add an `equinox` floor at the first release compatible with the
     jax versions solvax supports (verify; 0.13.x worked with jax 0.10.2), and
     a `jax`/`jaxlib` floor; release solvax and raise vmex's solvax floor to
     it.
   - vmex: raise floors to versions that are known to work together (jax,
     jaxlib, solvax, booz_xform_jax, virtual-casing-jax, essos, neo-jax, gkx,
     equinox if imported), so `vmex[all]` resolves to a coherent, current
     set; do the same audit in the sibling packages (booz_xform_jax,
     virtual_casing_jax, ESSOS, neo-jax, gkx), each declaring floors for what
     it imports.
   - Guard it: a CI job that installs `vmex[all]` into an environment
     pre-seeded with old versions of the transitive dependencies (e.g.
     equinox 0.11.x, jax 0.5) and runs `vmex examples/data/input.solovev
     --plot`, so a missing floor fails CI instead of a user's first run; plus
     the existing minimum-versions job at the new floors. Keep the README
     install table's floors in sync (a test already checks them).
   - Related, same report: pip warned that other installed packages
     (desc-opt, interpax, quadax, orthax, jax-finufft) pin jax below 0.10;
     document in the installation page that DESC interoperability needs a
     jax that DESC supports, or a separate environment.
1c. **README: say that polishing is requested from the input file.** A
   `! VMEX: POLISH_FORCE_BALANCE = .TRUE.` line at the top of an INDATA deck
   is a comment to VMEC2000 (ignored there) and a directive to VMEX
   (`examples/data/input.shaped_tokamak_pressure_polished`). The README's
   polishing section should state this, with the exact line.
2. **Example timings still open from #426.** 18 optimization scripts were
   timed under five minutes with a converged NS = 71 check; scripts over
   budget kept their previous defaults. The untimed scripts are listed in
   #426's body.
3. **`vmex --trace`: fast reactor-scale alpha losses.** The flag already exists
   (0.10.0); the work is new defaults (1000 particles, 1e-2 s), scaling on by
   default with the scaling target fixed, a converged step, and speed. Full
   scope, ESSOS PR order, phases and acceptance gates: section T below.
4. **Mirrors:**
   - Verify accuracy (analytic and independent references, the Pleiades
     reference where available) and solve speed of the fixed- and
     free-boundary mirror solves; fix what is slow or wrong. On the office
     machine the QI hybrid (389 s) and any finite-beta scan exceed five
     minutes (#434).
   - Make the mirror examples as concise as the tokamak and stellarator
     examples: the API does the assembly, the user still sees parameters,
     geometry and resolution. Add `examples/mirror/mirror_fixed_boundary_axisymmetric.py`.
   - Solve mirror configurations from input files (`vmex <input>`,
     `vj.solve_file`).
5. **Later (not scheduled):** cut compilation cost on free- and
   fixed-boundary solves (cold compile is 25-60 % of example wall time; about
   120 s fixed cost on the free boundary, #439).

**Open PRs:**

| PR | State | Next step |
|---|---|---|
| #301-#304, #306, #366, #367, #302 | winding surface and HINT comparison; untouched by instruction | Keep in their PRs, unmerged. |
| #417, #418, #421, #423, #424, #419, #371, #377, #415 | superseded or failed directions | Close with pointers (left to the maintainer). |

Other follow-ups recorded in the merged PRs: `multigrid.solve_file` should
pass the free-boundary metadata to the wout writer (#430); the graded rule
under a trace needs `near_surface="graded"` and the default source grid is
sized for d = a only (#441); the exterior-field VJP differs 2-4x from
independent re-solves on boundary directions because of the m = 1 gauge
drift (#428, #430). The next weekly run is the first to test #437's core-1
cap and examples fixes.

Measurement caveat: most timings in these PRs were taken on a shared laptop
or office machine under load; ratios come from interleaved A/B runs, and
absolute wall times are upper bounds.

## T. Fast reactor-scale alpha losses: `vmex --trace`

**Priority:** after the CI-speed work and the 0.11.1 follow-ups (#443), before
the mirror work. Scoped 2026-09-23 against `origin/main` `b5f5267ef` and ESSOS
`main` `c9b41222e` (= ESSOS 0.17). Research only; nothing below has been
implemented yet.

### T.0 What already exists (do not rebuild it)

`vmex --trace` shipped in 0.10.0 (`958ffde1a`). It is wired like `--plot` and
`--booz`: it runs on a `wout_*.nc` or after a solve, and
`--scale` refuses to combine with it. The code is `vmex/core/tracing.py`
(`essos_vmec_field` and `trace_alphas`, which return `AlphaTracingResult`),
`_run_trace` in `vmex/core/cli.py`, and `plotting.plot_tracing`. It writes four
figures: `*_trace_trajectories.png`, `*_trace_vparallel.png`,
`*_trace_loss_fraction.png` and `*_trace_energy_error.png`. The tracer is
`essos.dynamics.Tracing(model="GuidingCenter")`: fixed-step Dopri8 in VMEC
`(s, θ, φ)`, `vmap` over particles, and sharding over `jax.devices()`.
Particles start on s = 0.25 with uniform θ, φ over one field period, uniform
pitch in [-1, 1) and 3.52 MeV. A particle counts as lost when a sampled s is
at least 0.99. The defaults are 200 particles, `tmax = 3e-4` s and
`dt = 5e-7` s. It does not scale the equilibrium; the docs tell users to run
`vmex --scale` first. The feature is therefore a change of defaults plus
built-in scaling, a fix to the scaling target, and a speed programme. It is not
a new command.

### T.1 Measured baseline: too slow and not converged at the default step

The runs used the ESSOS 0.17 wheel, JAX 0.10.2 and diffrax 0.7.2 on an Apple
M3 Max (10P+4E cores). The script is `scratchpad/trace-plan/bench.py`.
Particles were launched on s = 0.25 with the vmex sampling, and each wall time
includes the per-call JIT.

| wout (Nyquist modes) | particles × t | dt / tolerance | devices | wall | loss |
|---|---|---|---|---|---|
| LP QA reactorScale (128) | 20 × 1e-3 s | 5e-7 fixed | 1 | 7.7 s | 0 |
| LP QA reactorScale (128) | 100 × 1e-3 s | 5e-7 fixed | 1 | 31.3 s | 0 |
| LP QA reactorScale (128) | 100 × 1e-3 s | 5e-7 fixed | 10 | 6.8 s | 0 |
| LP QA reactorScale (128) | 200 × 1e-3 s | 5e-7 fixed | 10 | 11.6 s | 0 (3 axis) |
| ARIES-CS `n3are` (450) | 200 × 2e-3 s | 1e-6 fixed | 10 | 23.9 s | **8.0 %** |
| ARIES-CS `n3are` (450) | 200 × 2e-3 s | 5e-7 fixed (vmex default) | 10 | 47.6 s | **3.5 %** |
| ARIES-CS `n3are` (450) | 200 × 2e-3 s | 2.5e-7 fixed | 10 | 83.2 s | **2.5 %** |
| ARIES-CS `n3are` (450) | 200 × 2e-3 s | adaptive Dopri8, rtol = atol = 1e-7 | 10 | 465.7 s | **2.5 %** |

The first four rows ran at load average ~4–5 and the ARIES-CS rows at ~7–13,
with other sessions active. Treat the times as upper bounds; the ratios are
reliable.

Findings:

1. **The default step overestimates losses.** At `dt = 5e-7` a 3.5 MeV alpha
   (~1.3e7 m/s) moves ~6.5 m per step, which is 40 % of an ARIES-CS field
   period. The loss fraction converges to 2.5 % only at `dt <= 2.5e-7`, where
   the fixed and adaptive runs agree. ESSOS's VMEC field interpolates linearly
   in s, so ∂/∂s is piecewise constant. That caps the effective order of
   Dopri8.
2. **Cost is set by the right-hand side.** One GC right-hand side takes about
   8–10 µs per particle. Each evaluation performs about ten Fourier syntheses
   over all Nyquist modes plus `jacfwd`/`grad` passes, and Dopri8 uses 13
   stages per step. Scaling is linear in particles × time. The adaptive runs are
   5.6× slower than fixed steps because the `vmap`'d `while_loop` runs all
   particles in lockstep.
3. **Extrapolated cost of 1000 × 1e-2 s at the converged `dt = 2.5e-7`:**
   ~19 min for QA and ~35 min for ARIES-CS on 10 host devices. On the
   single-device default this becomes ~3–6 h. The 5-minute target needs
   **≥ 7×** for 1000 particles and **≥ 35×** for 5000.
4. **Multi-device sharding on CPU is almost free.** Setting
   `--xla_force_host_platform_device_count=10` gave 4.6× on the same run
   (31.3 → 6.8 s). vmex does not set it today.
5. **Axis handling.** ESSOS#47, merged in 0.17, already stops VMEC GC orbits at
   `s <= 1e-6` and counts them as `total_particles_unresolved`. Axis crossings
   are still not continued: 3–7 of 200 orbits within 2 ms. Over 1e-2 s they
   bias the confined count and discard statistics.

### T.2 Scaling target: fix it before tracing through it

`aries_cs_scales` (`vmex/core/scaling.py`) returns
`(5.7/|wout.b0|, 1.7/Aminor_p)`. The wout `b0` is the toroidal field at the
axis in one plane. It is neither of the literature normalisations:

| source | length | field |
|---|---|---|
| ARIES-CS design (Najmabadi et al., FST 54, 655, 2008) | R = 7.75 m, A = 4.5 | 5.7 T average on axis |
| Landreman & Paul, PRL 128, 035001 (2022), arXiv:2108.03711 | a = 1.7 m | B₀₀(s=0) = 5.7 T (Boozer (0,0) on axis) |
| Landreman, Buller & Drevlak, PoP 29, 082501 (2022), arXiv:2205.02914 | a = 1.70 m | ⟨B⟩ = 5.86 T (volume average) |
| Bader et al., NF 61, 116060 (2021); Paul et al., NF 62, 126054 (2022), arXiv:2208.02351 | V = 444 m³ | ⟨B⟩ = 5.86 T |
| reference wouts in ESSOS/SIMSOPT: `wout_n3are_R7.75B5.7.nc`, `wout_LandremanPaul2021_QA_reactorScale_lowres.nc` | `Aminor_p` = 1.7044 m | `volavgB` = 5.8646 T (b0 = 5.33 and 5.18 T) |

The current rule rescales the ARIES-CS wout itself by **1.070** in B, and the
reactor-scale LP QA wout by **1.101**, which gives ⟨B⟩ = 6.46 T. A 7–10 % field
error shrinks orbit widths and understates losses.

**Decision.** Default to `Aminor_p = 1.7044 m` and `volavgB = 5.8646 T`, the
ARIES-CS wout's own values and the de facto target of the shipped reactor-scale
files. Under that rule both reference wouts map to factors of 1.000. Keep the
LP-PRL convention (B₀₀(s=0) = 5.7 T, a = 1.7 m) as a named option, computed
from `bmnc(m=0, n=0)` extrapolated to the axis. `ScaleProbe` gains `volavgB` so
`aries_cs_input_scales` follows the same rule. Update the `--scale` help, the
docstrings and the howto.

### T.3 CLI contract

```console
vmex wout_case.nc --trace                       # 1000 alphas, 1e-2 s, scaled to ARIES-CS in memory
vmex input.case --trace                         # solve, then the same
vmex wout_case.nc --trace --trace-particles 5000 --trace-tmax 1e-2
vmex wout_case.nc --trace --trace-no-scale      # trace the equilibrium as given
```

- **Flag names.** Keep the released names (`--trace-particles`,
  `--trace-tmax`, `--trace-s`, `--trace-seed`, `--trace-timestep`,
  `--trace-times`), which shipped in 0.10.0 and 0.11.0. Do not add bare
  `--particles`/`--time`, which would collide with future flags. Change the
  defaults to `--trace-particles 1000`, `--trace-tmax 1e-2` and
  `--trace-times 1000` (a loss-time resolution of 1e-5 s). Set
  `--trace-timestep` to the converged value from T.1 until P2 replaces the
  integrator.
- **Built-in scaling.** `--trace` scales in memory with
  `scale_wout(wout, *aries_cs_scales(wout))` and prints both factors and the
  resulting `Aminor_p`/`volavgB`. `--trace-no-scale` opts out. Keep `--scale`
  itself mutually exclusive with `--trace`, because it writes a file.
- **Parallelism.** When `--trace` is present, set
  `XLA_FLAGS=--xla_force_host_platform_device_count=<performance cores>` before
  JAX is imported, unless the user already set `XLA_FLAGS`. Verify that this
  does not slow the preceding solve.
- **Outputs,** beside the input or in `--outdir`:
  - `<case>_trace_loss_fraction.png` is the primary figure: cumulative loss
    fraction against time on a log time axis. Its title carries the
    configuration, N, s₀, scaling and wall time, with a binomial 1σ band.
  - `<case>_trace.json` holds the counts, factors, versions, device count, wall
    time and `dt`.
  - `<case>_trace.npz` holds `times`, `loss_fractions`, `lost_times` and the
    initial conditions, so a run can be replotted and compared.
  - The other three figures stay. The trajectory panels plot at most 8 orbits.
- **Console.** Print loss fraction ± binomial σ, lost / axis-unresolved /
  failed counts, wall time split into compile and run, and the scaling factors.

### T.4 Dependencies (ESSOS, in order; each merge needs maintainer approval, and an ESSOS release needs manual review)

| ESSOS PR | what it does | state | action |
|---|---|---|---|
| #47 VMEC axis events | stops GC orbits at the axis or LCFS and reports axis hits separately | merged; in 0.17 | none |
| #52 one loss surface | `boundary_threshold` (default 1.0) drives both the event and the loss count, removing the s = 1 event / s = 0.99 sample mismatch | draft, **conflicting** | rebase on main, then merge (1st) |
| #53 termination metadata | per-particle `termination_times`/`termination_states` | draft, stacked on #52 | merge (2nd) |
| #54 event-based losses | counts losses from the boundary mask and termination time instead of sampled `inf` values; separates axis, custom and failure outcomes | draft, stacked | merge (3rd); vmex then reads exact loss times |
| #55 refined events | Newton-refined axis and LCFS event times; rejects out-of-range starts; +11 % cold time | draft, stacked | merge (4th) |
| #56 fill event tails | post-event samples become the last finite state | draft, stacked | merge (5th); vmex drops its non-finite bookkeeping |
| #57 batched progress | progress reported per particle batch; `particle_batch_size`; +17 % time | draft, stacked | merge only with progress off by default |
| #48 solver controls | `max_steps`/progress controls | draft, conflicting | close as superseded; 0.17 already has `max_steps = 1_000_000` and the progress meter |
| #61 `Vmec.from_arrays` + soft loss | builds the field from arrays, removing the temp-wout hop; differentiable surrogate | open, conflicting | not needed for `--trace`; rebase later for the optimisation objective |
| #25 interpolated fields | SIMSOPT-style interpolated field (coil-oriented, VMEC example) | open since 2025-10, conflicting | reuse its interpolation machinery in P2b; do not merge as-is |
| #49 near-axis convergence + fixed-step note | examples only; based on `eg/analysis` | open | none (informational) |

After #52–#56 land, release **ESSOS 0.18** and raise the vmex `coils` extra
floor to `essos>=0.18`. The fallback is to feature-detect
`termination_times` and keep the 0.17 path. The P2 items add ESSOS 0.19 or
later.

### T.5 Phases

**P1: contract and correctness (vmex only, ESSOS 0.17/0.18).**
T.2 scaling fix; T.3 defaults, in-memory scaling, JSON/NPZ output, primary
figure; CPU device sharding; converged default step. README (T.7). Tests:
the tiny-budget smoke test (8 particles) stays; add a CLI contract test for
the files, the JSON keys and the printed scaling factors; add a unit test that
the n3are and LP-QA reactor-scale wouts give factors 1 ± 1e-3.
Expected wall: ~20–35 min for 1000 × 1e-2 s on the laptop. Ship it as correct
but not fast, with the time stated in the help text.

**P2: speed (ESSOS).**
- (a) Land the #52–#56 stack.
- (b) Add a Boozer-coordinate GC field built from vmex's `booz_xform_jax`
  output. Tabulate |B|, I(s), G(s), ι(s) (and the K/B_s term at finite β) on a
  regular grid in (√s, θ_B, ζ_B) and evaluate with periodic cubic splines, as
  in SIMSOPT `InterpolatedBoozerField` / FIRM3D #73, using the #25 machinery.
  Use the GC equations in Boozer coordinates (SIMSOPT `tracing` Boozer model).
  The target is a right-hand side of ≤ 1 µs per particle, ≥ 10× cheaper.
- (c) Add a regular chart near the axis: pseudo-Cartesian (√s cos θ, √s sin θ)
  below s ≈ 0.01, as in DESC and FIRM3D #79. The target is zero axis-unresolved
  orbits.
- (d) Choose the integrator by measurement at matched loss convergence
  (Dopri5, Tsit5 or Dopri8, fixed or adaptive), and keep fixed steps under
  `vmap`.

Gate: 1000 × 1e-2 s in ≤ 5 min on the laptop.

**P3: symplectic GC integrator, only if P2 misses the gate, or for 5000
particles and 0.2 s.**
Use Albert–Kasilov–Kernbichler explicit–implicit Euler in canonicalised flux
coordinates: JCP 403, 109065 (2020), arXiv:1903.06885. It is more than 3×
faster than RK45 at equal statistical accuracy, and SIMPLE uses it. The
closest ports are FIRM3D `ODE_solver="symplectic"` (#30 merged; #79 axis fix
open) and SIMSOPT branch `symplectic` (`1ecf27f26`: 5 commits, unmerged, 2733
behind master, with no PR). Implement it in ESSOS over the P2b Boozer field,
with a fixed step per field period, a Newton implicit stage and `vmap`.
Orbit classification (Albert et al., JPP 86, 815860201, 2020: a further
2–5×) applies to 0.2–1 s runs, not 1e-2 s. Leave it out.

### T.6 Acceptance gates

- **G1 scaling.** The reference wouts map to factors 1 ± 1e-3. Scaling a wout
  and scaling-then-solving its deck commute (the existing `scale_wout`
  contract).
- **G2 cross-code.** Use the same scaled wout and identical initial conditions
  (N = 1000, s = 0.25, 1e-2 s), with SIMSOPT `trace_particles_boozer` or SIMPLE
  as the reference. The loss fraction must agree within 2 binomial σ, and
  per-particle lost/confined labels must agree at ≥ 95 %. Run it on the
  ARIES-CS `n3are` wout (lossy) and the LP QA reactor-scale wout (near zero).
- **G3 published number (manual or weekly lane, not per PR).** ARIES-CS
  `n3are` at ⟨B⟩ = 5.86 T and V = 444 m³ (the file's own scale), s = 0.3,
  0.2 s, N ≥ 1000. The reference is 2470 of 10⁴ lost (Paul et al. 2022,
  Table 2), and the result must match within 2σ (σ ≈ 1.4 % at N = 1000). The
  LP precise QA at the PRL's Protocol A loses only "a few" of 5000.
- **G4 convergence.** Halving the step changes the loss fraction by < 1σ. There
  are no failed orbits, and after P2c there are no axis-unresolved orbits.
- **G5 wall time.** 1000 × 1e-2 s in ≤ 5 min on the M3 Max laptop (10 devices,
  load < 4, compile included), and 5000 ≤ 25 min. Record it in a cited
  `benchmarks/` JSON and time it A/B/A/B on a pinned SHA.

### T.7 README

In the first "Inspect the physics" bullet and in "Solve, plot and restart",
add `vmex --scale wout_my_case.nc` beside the `--plot`/`--booz` lines, plus
one sentence: "`--scale` writes `*_scaled` at ARIES-CS size (a = 1.70 m,
⟨B⟩ = 5.86 T); two factors `B R` scale by hand." Add
`vmex wout_my_case.nc --trace` with its one-line output and the loss-fraction
figure once G2 passes. Stay within the README line cap.

### T.8 Risks

- **Convention spread.** B₀₀(axis) = 5.7 T and ⟨B⟩ = 5.86 T differ by up to
  ~10 % on real configurations. State the convention in every output and
  figure.
- **1e-2 s only captures prompt losses.** Published losses are at 0.2 s. Keep
  G3 on a slow lane and never compare 1e-2 s numbers against 0.2 s tables.
- **Merges and releases are gated.** Every ESSOS merge needs maintainer
  approval and every release needs manual review. P1 must work on 0.17.
- **`vmap` lockstep.** One slow orbit sets the wall time. Adaptive stepping
  under `vmap` is 5.6× slower; P2d must measure it and must not assume it.
- **Scope limits.** `lasym` wouts stay rejected, since ESSOS reads symmetric
  tables only. A finite-β Boozer field needs the B_s/K term, or P2b is
  vacuum-only.
- **Device count.** Forcing host devices is process-global. Check solve+trace
  in one process and GPU hosts, which must not get a forced CPU device count.

## Current status

| Area | Status | Public evidence / source | Next action and completion gate |
|---|---|---|---|
| Counters and controlled profiling | Delivered; warm measurement correction merged (#420, `17bd8469`) | #310, #393, #420; `benchmarks/optimization.py`, `benchmarks/profile_workflows.py` | Remeasure warm aggregates with schema 2, then complete a JIT-enabled checkpoint separating cold, cache reload and warm runs. |
| Newton refinement and block responses | Delivered; anchor contract active; `equilibrium_from_x` now returns the refined state (#427); mean-iota re-solve gap explained as the m=1 gauge family (#428) | #330, #335, #338, #427 (`084f6c0e`), #428 (`604e6a76`); `vmex/core/implicit.py` | Lane A: anchor the free-boundary state at its root and bound the reference restart (new PR to follow). |
| Stage compilation and Jacobian batching | Delivered | #390 (`39db0388`), #392 (`a18bc448`) | Preserve frozen stage variables, final designs and memory bounds on current integrated sources. |
| Cold setup and WOUT export | Delivered | #396 (`ddf7d3ee`), #400 | Measure remaining startup costs after these changes, not against the superseded eager setup. |
| Free-boundary root and Schur reuse | Delivered; deterministic cold recovery merged (#416, `6f1df723`); adjoint exact at the root but the returned state is off it (lane A) | #383 (`92e6e0bf`), #385, #397 (`3c965913`), #416 | Lane A (off-root state, restart cap) and lane B (vacuum islands, feasible full designs). |
| Native interior field and exterior accuracy | Validated against independent oracles (#430); near-surface graded rule and closed-form derivative kernels in the library, continuation removed (logbook 2026-09-23) | #378, #399, #403, #409, #430; `benchmarks/extender_ab_20260923.json` | Keep the curl-free projection opt-in. E2's table for long exterior traces; a traced per-point switch; grid sizing below d = a. |
| Fixed-boundary single stage | Both fixed-boundary examples (zero and 0.5 % beta) meet every target in #411's runs at `952c3160` | #368; draft #411 (`35024f37`), which supersedes #371 | Refresh the four scripts' docstring numbers from #411's run table, requalify the shipped budgets, then merge #411. |
| Free-boundary single stage (zero and 0.5 % beta) | 0.5 % beta meets every target; zero beta meets every target once an iota ceiling (max\|iota\| <= 0.44) keeps the 4/9 island chain out of the plasma | draft #411 ([run table](https://github.com/uwplasma/vmex/pull/411#issuecomment-5784049398)); #411 supersedes #377 | Lane B: land the iota ceiling and cold 16 -> 51 verification in #411; per-trial cost (lane A restart cap) is the remaining usability gap. |
| QI objective | Bounded candidate rejected; nonsmoothness remains | `vmex/core/optimize.py`, `413d7fd2` | Lane D: retain current objective and the measured limitation; no width sweep or production surrogate promotion. |
| Strong-force polishing | Active research; general 3-D promotion unmet; unresolved-lift rejection merged (#414, `311e7ddd`); same-mesh comparison merged (#412, `07d43327`) | `docs/explanation/validation.md`, existing E1/E2 records | Lane E: resolve lift/axis accuracy, then certify an affordable 3-D correction and its derivative. |
| Optimization example defaults | Owner decision: bake #426's defaults into main | draft #426 (`52d27ee5`) | Finite-beta examples set beta through PHIEDGE normalization, not a pressure-calibration loop (in progress); fix the final `INITIAL JACOBIAN CHANGED SIGN!` refinement failure #426 reports. |
| Plan, evidence and repository footprint | Active | This plan; `benchmarks/INDEX.md` | Lane F: compact public reproduction, no dead artifacts, no deletion of sole scientific evidence. |
| Winding surfaces and broad new physics | Deferred | #301/#303/#304 and independent contributor PRs | Do not restart winding work or alter contributor branches as part of this campaign. |

The latest open-PR list supersedes numbers in historical entries; the table
under "Open pull requests" below is the state at this checkpoint. Coordinate
with these contributors instead of duplicating their work. #302/#306 are design references: port only
needed contracts onto current main, not their old implementation/evidence trees.

## Open pull requests at this checkpoint

State on 2026-09-22 after main `604e6a76` (0.11.0 + #427 + #428). "Head" is
the commit reviewed here; recheck before acting. No merge is authorized
without maintainer approval.

| PR | State | Owns | Disposition |
|---|---|---|---|
| #407, #410, #416 | merged, released in 0.11.0 (#425) | c1/c2 lane split; released integration floors; deterministic free-boundary cold recovery | done; "CI capacity", "Dependencies", lane B |
| #408, #409, #412, #414, #420 | merged, released in 0.11.0 | Schur warning timings; surface-call spectra; same-mesh polishing comparison; unresolved-lift rejection; complete warm profiling | done |
| #427, #428 | merged after 0.11.0 | `equilibrium_from_x` returns the refined state; m=1 family closure test for the mean-iota gap | done; lane A |
| #411 | draft, `35024f37` | the four single-stage examples (renames, least-squares form, matched fixed/free and finite-beta pairs, iota ceiling) | finish and merge; supersedes #371 and #377 (lane B) |
| #371, #377 | open, `9b427fb0` / `e53e3b57` | earlier single-stage split; earlier finite-beta free-boundary record | close as superseded by #411 when it merges |
| #413 | this plan and documentation corrections | plan, README, doc qualifications | lane F |
| #415 | draft, `9f754112` | remove duplicated benchmark narratives | merge after a rebase on main |
| #417 → #418 → #421 → #423 → #424 | stacked drafts, heads `1459f9df`, `9e0baa28`, `e94e46c6`, `02bf33d9`, `7de9f45d` | refined-state anchor contract, direct primal validity, callback device placement, exact supplied-state measurement, certification of freshly materialized equilibria | close: the premise did not reproduce on main; the one real defect landed as #427 ([review](https://github.com/uwplasma/vmex/pull/417#issuecomment-5766740229)) |
| #419 | draft, `709a35ea` | guarded radial-factor reuse | close: the direction failed its gates; its finding is kept in lane C |
| #426 | draft, `52d27ee5` | optimization example defaults (QA/QH/QI/QP mode ladders, budgets, aspect targets, live monitoring, vector-residual finite-beta QA) | owner: bake the defaults into main; replace its finite-beta pressure calibration with PHIEDGE normalization ("Maintainer decisions") |

The #417 stack never touches the free-boundary derivative path
(`freeboundary_implicit` calls none of its admission code), so it is not on
the critical path of lane B. The critical path for the free-boundary workflow
is now lane A's off-root state and restart budget, then #411.

## Maintainer decisions

**Single-stage example scope.** The standard single-stage examples are one
matched fixed/free-boundary pair using the proven fixed-boundary vacuum case's
seed geometry, NFP, physical limits, objective definitions and coil
parameterization, plus one finite-beta pair at 0.5 percent beta with a simple
pressure profile and zero plasma current. #411 implements this. The detailed
acceptance rules are in lane B.

**Bootstrap examples are deferred.** Bootstrap current and plasma-current
optimization go to `single_stage_fixed_boundary_finite_beta_bootstrap.py` and
`single_stage_free_boundary_finite_beta_bootstrap.py`, to be added only when
those workflows are ready. Do not add them, or duplicates of them, before then.

**Optimization example defaults (#426).** The improved mode ladders,
iteration budgets, aspect-ratio targets, boundary steps, verification
resolutions and live progress output in #426 are to be baked into main as the
defaults of the QA/QH/QI/QP examples.

**Finite beta through PHIEDGE, not a pressure loop.** At zero net current,
volume-average beta depends on the pressure and field only through
`PRES_SCALE/PHIEDGE^2`. Finite-beta fixed-boundary examples therefore set beta
by normalizing PHIEDGE for the requested beta, instead of #426's
pressure-calibration/continuation loop at fixed PHIEDGE. Work in progress; it
replaces the calibration in #426 and in #411's fixed-boundary finite-beta
script (`PRES_SCALE` from one seed solve) once measured.

**Vacuum free boundary is fixed in the example, not the solver.** The
zero-beta free-boundary limit cycle is a missing equilibrium (a 4/9 island
chain at the requested flux), not a solver defect; see lane B. The example
keeps the transform below the resonance. Tiny-beta regularization is rejected.

**Examples run in at most five minutes.** Every example, including
compilation, optimization and its final verification solve, must finish in
five minutes on a laptop; longer runs scare users away. Reduce iterations and
degrees of freedom where needed, but an optimization example must still
visibly change its design (coils and surface, for single stage). The
preferred route is making VMEX faster (solves and derivatives), not only
smaller examples.

**Only cited benchmark records stay.** The earlier policy of keeping uncited
records in `benchmarks/` is reversed (#431): records and scripts not cited by
the published docs or tests, and not run by CI, are deleted; git history keeps
them, linked by permalink where the plan or code still names them.

**Documentation explains methods, goals and results** (#429). New numbers in
the docs must cite a committed record or test.

## CI capacity (c1/c2 parity lanes)

The split is implemented (#407, `be805b73`, released in 0.11.0). Five
solve-heavy modules moved from c2 to the new lane `pr-parity-c5`
(`test_wout_from_result.py`, `test_refine_staging.py`, `test_scaling.py`,
`test_solver_axis_initialization.py`, `test_vacuum_analytic_recurrence.py`);
`test_strong_force.py` and `test_mgrid.py` moved from c1 to `pr-parity-c6`;
`test_strong_force_solovev.py` and `test_force_oracle.py` moved from c1 to
`pr-parity-c7`. c1 fell from 1216–1458 s to 415 s of job wall time. `PR gate`
needs the whole parity matrix, so the new lanes are gated without a change to
the required checks. c2 still owns `plan.md` (`tests/test_test_manifest.py`
asserts it), so every plan edit runs c2. A cancelled lane at its cap on a
documentation PR is still a capacity issue, not a defect: rerun it and record
the job time.

**Lane audit (2026-09-23).** Every workflow job's last 30 runs (41 for `CI`)
were read, plus a fresh `Weekly high resolution` dispatch on `2feba0d1c`. All
`CI` jobs passed every run that was not cancelled; each `PR gate` failure was a
cancelled or superseded run, or a failure in that PR's own code. Nightly is green
(opt-qh failed 2026-09-16 to 09-22 and recovered after #427). Retired: `Trusted GPU physics`
(`gpu.yml`), which needs a self-hosted runner that is registered by hand and is
not registered now; its last success was 2026-07-31. Its `gpu-smoke` tests are
declared `local_only` in `tests/manifest.json`, with the command in
`docs/project/contributing.rst`. Fixed: the weekly examples shards (a stale
LASYM objectives filename since #398, and the QI maximum-J example's 900 s
timeout against 814–832 s runs), weekly core-1's 15-minute cap (cancelled at
it), and `Publish to PyPI` failing on every asset-bundle release. Removed the
manifest lane labels that no workflow selects (`device-rig`,
`pr-implicit-response`, `pr-physics-mirror-spline`,
`pr-physics-mirror-output`). A guard test now fails on any lane no workflow
selects unless it is declared `local_only`. Weekly hmfb failed today only at
`pip install` (PyPI briefly returned no `equinox` distribution), so it was
rerun rather than changed.

## Execution and acceptance

First qualify the integrated certification, recovery, dependency and profiler
changes. Then establish consistent equilibrium families and close the controlled
collaborator comparison on one QI and one QA workflow before expanding the
matrix. Factor reuse depends on those accuracy gates. Free-boundary checks can
proceed independently; QI surrogate sweeps and broad polishing rewrites remain
stopped. Coordinate shared source edits, use isolated worktrees and exact pins,
and inspect compute occupancy before bounded runs. Do not stop others' jobs.

Every implementation PR must include the problem, exact tested revisions,
before/after evidence, accuracy and memory limits, runnable public commands,
and remaining gates. Preserve at least 95% changed executable coverage with
meaningful physics/numerics tests; coverage alone does not certify a method.
Exercise JIT explicitly for staged behavior, since ordinary unit tests disable
it. Review CPU and GPU placement rather than inferring it from available hardware.

Obtain explicit maintainer approval before any merge; earlier blanket merge
authorization is superseded. ESSOS also requires manual release review. Require
the intended scientific gates and current checks; never bypass failed numerical
checks. (The earlier deferral of every release until all research gates
passed was superseded by 0.11.0, which shipped the integration fixes with
their limitations named.) VMEX 0.12 follows when #411 and #426's defaults are
on main and lane A's free-boundary anchoring has landed or is explicitly named
as a limitation; the release notes must carry the qualifications of lanes A
and B.

Commit authorship is the maintainer's, with no agent attribution. Inspect the
exact outgoing diff and text before publication: no private filesystem paths,
home usernames, host aliases, checkout/environment names, or private artifact
and script names. Store machine-specific recovery details privately.

## A. Derivative and equilibrium consistency

Source: `vmex/core/implicit.py`, `vmex/core/optimize.py`,
`vmex/core/freeboundary_implicit.py`; baseline `f719c4ff`, re-audited at
`780eb86e`.
Native FSQ, the refined nonlinear residual, the linear-response residual and
observable accuracy are distinct quantities. Current refinement may return an
improved state without reaching its target; host admission uses native FSQ,
and Jacobian reuse keys do not explicitly identify the refined state.

Implement the smallest state/provenance contract that ensures a residual and
its derivative use the same coefficients, parameters, mask and residual
operator. Refresh stale evidence, reject invalid responses without caching
them, and preserve unrelated programming errors. Do not introduce a universal
1e-10 primal cutoff: near-null modes impose measured attainable floors.
Expose qualification when the observable's requested accuracy is not justified.
(Superseded in part 2026-09-22: on main the fixed-boundary residual and
derivative already use the same refined state and repeat across histories;
the contract work that remains is free boundary, below.)

**Independent audit verdict (2026-09-22,
[#417 comment](https://github.com/uwplasma/vmex/pull/417#issuecomment-5784815035)).**
Measured on main `780eb86e` (0.11.0) with #411's finite-beta free-boundary
single stage (0.5 % beta, ns 31, mpol = ntor = 5, 81 coil dofs), laptop under
load; timings are labelled accordingly.

- **The free-boundary adjoint is exact at the root.** Against central FD of
  Newton-anchored roots with the same frozen complement it agrees to
  1e-9–6e-7 (6.0e-8 objective, 8.2e-9 beta, 1.8e-8 mean iota, 1.0e-7 aspect at the seed;
  1.3e-9–6.4e-7 for every term at the optimum).
- **It is fast.** A gradient adds 5.5–9 s to a value, almost independent of the
  number of coil dofs (45/81/117), 4–13x faster than forward FD near the seed
  and 68x mid-optimization (35.3 s against 2415 s at 81 dofs). Live JAX arrays
  stay at 210 (1.2 MiB) over 20 consecutive gradients: no leak. Peak RSS
  2.4–3.2 GiB for the trial loop, 4.5 GiB fixed boundary.
- **(i) The returned free-boundary state is not at that root.** A status-0
  solve at ftol 1e-12 sits 1.15e-2 (seed) and 1.26e-2 (optimum) from the root
  in coefficient norm; four Newton steps on the projected coupled residual take
  |F| from 3e-7 to 5e-14. Fixed boundary anchors this with `_refined_state`;
  free boundary has no anchor. Warm FD therefore reads dJ/h = 15.6 where cold
  solves and the adjoint give 30.0 and 29.7: the restart meets ftol along a
  soft mode before reaching the root. The gradient is right; the values the
  optimizer compares it with are not at the same point.
- **(ii) Wasted restart work.** Near the optimum about 80 % of each trial
  (about 20 s of 25 s) is a seed-reference restart that runs to its
  4000-iteration cap without converging before the cold fallback converges in
  823–1425 iterations. A stall test or budget before the fallback would cut the
  trial cost about 4x.
- **(iii) The value map depends on the start at finite beta.** From the seed
  reference, a mid-path reference and a cold start, J differs by up to 12 %
  (t = 0.25: 1.1598, 1.3257, 1.2959); every evaluation returned status 0, so no
  status flip was seen at finite beta, but the map is not a function of the
  coils alone, and it jumps where the production path switches from restart to
  fallback.
- **The ~1 % mean-iota gap is not a missing term.** It is the released m=1
  `Z_sin` gauge family: the linearization holds it at its converged value,
  each cold re-solve freezes it where its own path left it (about 0.5 per unit
  boundary change). On `li383_low_res` (ftol 1e-13, h = 5e-4) the gap is 1.0 %
  (RBC(1,1)) and 2.8 % (ZBS(-1,1)); adding the iota response to that drift
  closes it to 2.3e-6 and 1.6e-5. Over mpol 4–8 and ns 16–64 the gap spans
  0.06–2.8 % and does not fall monotonically. #428 pins the closure as a test
  (`tests/test_implicit_grad_fd.py::test_li383_mean_iota_resolve_fd_gap_is_the_m1_constrained_family`).
  Pinning the family to a canonical, parameter-differentiable value is the
  option if sub-percent agreement with re-solves is wanted (VMEC++ #849 is the
  precedent; see lane C).

Work on (i) and (ii) is in progress; a new PR will follow. Its gates: the
anchored free-boundary state repeats from different references, the value
and gradient refer to the same state (warm and cold FD agree with the adjoint
at the audit's precision), and trial cost at the optimum falls without a new
failure mode.

The #417 → #424 stack (hardening of the fixed-boundary anchor contract) is to
be closed: its premise did not reproduce on main (Jacobians bit-identical to
#424, history-independent residuals and Jacobians) and it never touches
`freeboundary_implicit.py`. Its one real defect, `equilibrium_from_x` returning
the unrefined state (objective mean iota 0.553590 against WOUT 0.554491),
landed as #427. The stack's record, as it stood on 2026-09-21, is kept in the
continuation logbook below.

Gate on repeated points after accepted/rejected trials, independent problem
instances, direct Jacobian calls, changed/missing anchors, and host/staged lanes.
Report raw/projected nonlinear and linear residuals. Verify forward/reverse
agreement and two-sided Taylor convergence; use independently reconverged
perturbations wherever their noise floor permits. A same-root linear identity
is not proof of accurate nonlinear parameter dependence. Public fixtures must
replace reliance on unavailable collaborator states. The baseline counters
are in [`benchmarks/optimization_counters_20260913.json`](https://github.com/uwplasma/vmex/blob/07a47279d5cea819bb23c5329fab7f14cced2456/benchmarks/optimization_counters_20260913.json).

## B. Free-boundary scientific workflow

Source: `vmex/core/freeboundary_implicit.py`, existing single-stage examples;
baseline #383/#397. The fixed reference and bounded cold-rebuild mechanism
already exist. Test them rather than implementing a second policy.

**Current state (2026-09-22).** #416 is merged. The four single-stage examples
live in draft #411, which supersedes #371 and #377. Its latest runs (clearance
limit 0.15 m, aspect limit 6.0 with an aspect-5.97 seed, order-4 coils on a
0.5 m circle, one-sided coil length <= 5.5 m, 50 iterations / 100 trials; four
runs concurrent on one laptop, so wall times are pessimistic):

| example | targets | wall | peak RSS | iterations / trials |
|---|---|---|---|---|
| fixed, zero beta (`952c3160`) | all met (min iota 0.4289, aspect 5.969, B.n/B RMS 0.800 %, clearance 0.261 m) | 335 s | 3.8 GB | 50 / 86 |
| fixed, 0.5 % beta (`952c3160`) | all met (beta 0.5011 %, min iota 0.4295, aspect 5.971, B.n/B RMS 0.800 %) | 566 s | 9.1 GB | 50 / 65 |
| free, 0.5 % beta (`952c3160`) | all met (beta 0.4833 %, min iota 0.4309, aspect 5.930, clearance 0.246 m) | 2399 s | 14.3 GB | 27 / 109 (trial cap) |
| free, zero beta, before the iota ceiling (`952c3160`; `08d97509` warm-verified) | misses B.n/B RMS (1.18–1.48 % vs 1 %); verification solve limit-cycles | 1536 s | 7.7 GB | 27 / 95 (9 rejected) |
| free, zero beta, with max\|iota\| <= 0.44 (`35024f37`) | all met (min iota 0.431, max iota 0.436, aspect 5.93, B.n/B RMS 0.617 %, coil-surface 0.244 m, coil-coil 0.191 m, max curvature 6.90); cold 16 -> 51 verification converges to 1e-12 | 1898 s | not recorded | 30 / 101 (2 rejected) |

Earlier #411 runs at `e0fa2174` (aspect-4 seed, order-3 coils) missed targets
in three of four examples; that table is on #411 and is superseded by the one
above. Coil terms are inactive at the end of every run (coils 4.5–4.8 m);
quasisymmetry is traded against the iota floor (seed min iota 0.404 below the
0.43 floor), not against the coils. Free-boundary wall time (26–40 min) is
still too long for a user-facing example: per-trial cost is 16–22 s against
4–9 s fixed boundary, and lane A (ii) accounts for most of it near the
optimum. The scripts' docstring numbers predate these runs and must be
refreshed from this table before #411 merges.

**The zero-beta free-boundary limit cycle is a missing equilibrium, not a
solver defect**
([#411 diagnosis](https://github.com/uwplasma/vmex/pull/411#issuecomment-5784049398)).
Field-line tracing of the optimized coils (64 lines, 1000 transits) finds
nested surfaces with iota 0.459 -> 0.449 out to 0.79 of PHIEDGE, then a
separatrix, a 9-island chain locked at 4/9 = 0.4444, and a stochastic layer
outside. The PHIEDGE boundary cuts through the islands: no nested flux surface
encloses the requested flux, so no zero-beta, zero-current nested-surface
equilibrium exists for that deck. VMEC2000 (fsq floors at 3.6e-8, bursts to
3.8e-4, DELT collapses to 7.4e-3) and VMEC++ (bursts to 4e-5, then
`JACOBIAN_75_TIMES_BAD` at 9022 iterations) fail on the same mgrid deck the
same way; VMEX floors near 4e-9 with bursts to 1e-4. Controls: the
Landreman-Paul QA coils converge cold at zero beta (ns 31 to 1e-11 in 1832
iterations; 16 -> 51 in 627 at ns 51), so zero beta alone is not degenerate.
Ruled out: tiny-beta regularization (<beta> 1e-6 to 5.4e-3, none converged in
20000 iterations; do not build it into VMEX), NESTOR cadence and DELT
(NVACSKIP 6/15, DELT 0.3), higher resolution, and any zero-pressure code path.
Inside the good surfaces a vacuum solution exists (PHIEDGE x 0.70 converges
with the 16 -> 51 ladder). While unconverged, VMEX's iota read below the
field-line iota, so the `min |iota| >= 0.43` floor pushed the real profile up
through 4/9.

Fix, in the example: add `max |iota| <= 0.44` (penalty from 0.437) next to the
floor, restore the cold 16 -> 51 verification at ftol 1e-12 (revert the
warm-started verification of `08d97509`, keeping the `[accepted]` diagnostic
line), and report max |iota|. The window [0.42, 0.44] excludes 2/5 and 4/9;
the rationals left with n a multiple of NFP are high order (3/7, 5/12).
Independent cold solves of the new coils at ns 16, 31 and 51 converge and give
B.n/B 0.617–0.618 %; field-line tracing shows no locked chain and nested
surfaces to about 1.11 times the boundary distance on the outboard midplane
(thin margin at phi = pi/4). Suggested solver follow-up (optional, not a fix):
for a zero-beta, zero-current free boundary, report the coil-field B.n/B RMS
on the boundary or DEL-BSQ and warn when it does not fall with resolution;
that catches a missing flux surface directly instead of as a limit cycle. The
history-dependent trial status seen before (status 0 during the optimization,
status 2 with fsq 1.374e-8 on re-solve) is a symptom of the missing fixed
point at the old coils.

**Record of the #416/#377 derivative studies (2026-09-21).** Superseded as the
current account by lane A's audit (exact adjoint at the root; values off the
root; m=1 gauge family). Kept for its measurements.

Draft #416 removes the exhausted global rebuild budget and passes controlled
and physical repeated-point checks. The [independent pressure study](https://github.com/uwplasma/vmex/pull/416#issuecomment-5755622514)
on #416 plus #417 now resolves one discrepancy: tightening native `ftol` from
1e-9 to 1e-10 to 1e-12 reduces the adjoint/re-solve gap from 14.24 percent to
8.54 percent to 0.0167 percent. At the tightest level, independent differences
at relative steps 1e-3 and 1e-4 agree within 1.1e-5 relative; every repeated
state coefficient agrees exactly. This qualifies the tested pressure observable
on one coarse asymmetric deck, not every free-boundary derivative. Preserve
it as a physical regression. The [actual finite-beta objective study](https://github.com/uwplasma/vmex/pull/377#issuecomment-5756092179)
now finds no finite-difference agreement window at `ftol=1e-9`, despite exact
repeated states and unchanged active sets. At the saved endpoint, AD is +1.189
but centered differences are -2.034 and -9.736 at steps 1e-3 and 1e-4.
The [exact-reference tighter study](https://github.com/uwplasma/vmex/pull/377#issuecomment-5756328738)
preserved that full production stage at `ftol=1e-12`. The admitted base's raw
norm improved from 1.64e-3 to 7.98e-5 and its coefficients repeated exactly.
However, the minus perturbation at h=1e-3 failed admission after cold recovery
(FSQ/ftol=711.87), while the plus side passed. The smaller-step sweep stopped;
this pair cannot assess derivative accuracy. The 8,000-step `[15, 31]` trial
ladder also missed convergence (FSQ/ftol=1413.12), but its saved state has
[valid direct geometry](https://github.com/uwplasma/vmex/pull/377#issuecomment-5756768575).
The initial contrary diagnosis used a fixed-boundary accessor that replaced
the evolved edge; it did not measure the supplied free-boundary state.
The [same-state recovery A/B](https://github.com/uwplasma/vmex/pull/377#issuecomment-5756840863)
rejects restoring the full production continuation payload as the fix: both
arms failed, and restoring it worsened FSQ by 273x. Completed tight pairs at
[h=1e-4](https://github.com/uwplasma/vmex/pull/377#issuecomment-5756902103)
and [h=5e-5](https://github.com/uwplasma/vmex/pull/377#issuecomment-5757012060)
preserved the same saved base, adjoint and production reference. Both pairs
passed native/geometry admission without cold recovery; the directional gap
fell from 2.1746 percent to 0.08114 percent. This improves local agreement but
does not establish an asymptotic window or quantify nonlinear root error.
The [new complete base capture](https://github.com/uwplasma/vmex/pull/377#issuecomment-5757443492)
on composite `c7807705` saved state, parameters, mask and `rcon0/zcon0`
before differentiation. It reproduced the old base exactly; the adjoint
repeated within 5.54e-9 with true relative transpose residual 2.87e-11.
The saved h=5e-5 state chord nevertheless has a 2.31 percent tangent defect
relative to the field term, using the base frozen complement and constraints.
Its nonzero frozen m=1 component also enters the full objective chord but not
the projected residual chord. Do not attribute that discrepancy solely to
finite-step curvature or root noise before separating the complement term.
Next replay the existing endpoints against the same base operator and resolve
the adjoint-weighted tangent/complement identity; no new endpoint solves are
needed for that discrimination. The public record includes controls and
checksums but lacks downloadable tapes and a standalone generator: publish
those or qualify newly generated public inputs before claiming reproducibility.
Do not substitute zero constraints or repeat the completed base capture.
Qualification on the proposed minimum stack remains separate from this
SOLVAX 0.20.0 study, as do other directions and final design feasibility.
Native stopping, measured coupled residuals and observable agreement remain
separate gates; #385's same-root identity alone was insufficient.

The standard single-stage examples now target a simpler matched comparison:
use the proven fixed-boundary vacuum case's seed geometry, NFP, physical
limits, objective definitions and coil parameterization for the fixed/free
pair. First recheck its recorded feasible result on the integrated source;
then select the shortest measured budget that still passes independent final
checks. An iteration cap alone is not a runtime bound: count evaluations,
line-search trials, equilibrium solves, compilation and output costs.
Free boundary evolves the plasma surface from the coil field, so do not copy
independent boundary DOFs or assume the circular coil seed carries the fixed
seed's transform. Reuse the fixed-boundary coil-fit initialization and verify
its free-boundary root before optimization. Report physical limits on the same
fine grids; include coil-surface clearance in both formulations.

For the standard finite-beta pair, set the target to 0.5 percent (`0.005`),
use a simple pressure shape such as `p(s)=p0*(1-s)`, and prescribe zero plasma
current (`ncurr=1`, zero current profile and `curtor=0`). Calibrate seed pressure
and report final beta; keep the same beta convention and acceptance in both
scripts. (Maintainer decision 2026-09-22: set beta through PHIEDGE
normalization rather than a pressure-calibration loop; see "Maintainer
decisions".) Remove bootstrap mismatch, kinetic-profile preparation, bootstrap
Picard iterations and plasma-current DOFs from these introductory examples.
Coil currents are separate external-field parameters. Self-consistent bootstrap
and plasma-current optimization are deferred to later
`single_stage_fixed_boundary_finite_beta_bootstrap.py` and
`single_stage_free_boundary_finite_beta_bootstrap.py`; do not add duplicate
scripts before those workflows are ready. Keep the existing 2.5-percent case
as derivative evidence, not the default example or a task to rerun unchanged.
Simpler optimization does not qualify a gradient: retain the same-root and
independent-perturbation checks before claiming convergence or speedup.

The next paragraph is the pre-#411 record; #411's table above supersedes its
design results. With #409 merged, integrate example reporting fixes and qualify these simpler
vacuum and finite-beta workflows on the memory-fixed source. Measure retained
arrays, RSS/device memory slope, fallback counts, accepted progress and final
fine-grid verification. The updated #377 vacuum record at `cf9811bf` reports
1164.1 s and 4.09 GiB,
with its stated targets met, replacing 3304.1 s and 18.89 GiB. The recovered
finite-beta run at the same source completed in 1776.2 s and 5.26 GiB: its
output solve converged, but L-BFGS-B stopped abnormally and minimum absolute
iota was 0.258577 against 0.42. The committed record is updated in #377
(`e53e3b57`); these shared-CPU timings predate #416 and are not a controlled
speedup comparison. The [optimized-coil verification](https://github.com/uwplasma/vmex/pull/377#issuecomment-5755533991)
now supports `[15, 31]`: it converged where single-grid budgets through 20,000
iterations failed. The extra 9-surface rung also converged but is not required
by this case. Reproduce the optimizer failure after recovery fixes, not an
already completed measurement.
Determine whether root accuracy, the optimizer or the seed limits progress
before adjusting weights. Require all stated constraints, not merely a lower
weighted objective. Preserve the documented experimental free-boundary AD scope
until CPU/GPU and independent derivative gates actually pass.

## C. Reuse numerical work without changing the certified operator

SOLVAX #117 merged at `1d02a140`, providing reusable scalar tridiagonal
factorization and checked factored solves; it is not in the current 0.24.0
release. Keep released-package qualification separate from VMEX adoption.
The VMEX scalar-cache experiment is rejected: complete solves did not improve
and the 151-surface carried cache grew from 277,568 to 491,264 bytes, despite
bit-identical trajectories. Do not integrate it on kernel timings alone. VMEX
retains ownership of physical bands and invalidation on updates and recovery.

**#419 is to be closed (2026-09-22): the guarded-reuse direction failed its
gates.** Its finding is kept here: independent reconvergence moves the frozen
m=1 `Z_sin` complement between families (cross-family projected residuals
0.066 and 0.092 against 1e-13 within a family), which is the same gauge family
that lane A identified as the whole mean-iota re-solve gap (#428). Any reuse
or pinning experiment must first fix that family; the record follows.

Draft #419 included the [bounded QI optimizer result](https://github.com/uwplasma/vmex/pull/419#issuecomment-5756488807):
312/312 reused response columns passed exact-current checks at 1e-11 tolerance,
but both arms exhausted their budgets. Independent reconvergence changed the
frozen m=1 Z-sine complement: each state's projected residual was about 1e-13,
while cross-family residuals were 0.066 and 0.092. Fine angular sampling also
raised mirror ratios to 0.21056 and 0.21114, above the hard 0.21 limit.
Do not promote reuse or redefine these failed gates around a saved state.
First establish a consistent residual family and the physical/parameter
dependence of the released components; distinguish coordinate gauge and
sampling sensitivity from physical equilibrium changes. Then repeat the
independent-root and fine-grid checks before timing a complete optimizer.
An explicit precedent is [VMEC++ #849](https://github.com/proximafusion/vmecpp/pull/849)
(`cec07e9e`, open): it pins the m=1 gauge to a boundary-derived radial profile
and differentiates that profile. Test this as an opt-in VMEX experiment on the
two saved states, with identical assembly in the root, objective and derivative.
The native high-force phase can evolve these components, so the pinned profile
defines a new discrete family rather than reproducing native cold-start history.
Require same-family reconvergence, parameter-direction checks including the
profile derivative, and independent physical/resolution checks before adoption;
retain existing restart semantics until explicitly qualified. The
[bounded post-solve projection](https://github.com/uwplasma/vmex/pull/419#issuecomment-5756859333)
failed both root gates. An A-only trace found an accurate first raw linear
solve (relative defect 1.17e-12), followed by a full step that increased the
projected nonlinear residual from 0.0603 to 2.29e9. The
[bounded globalized follow-up](https://github.com/uwplasma/vmex/pull/419#issuecomment-5757238643)
accepted eight finite, geometry-valid steps with independently accurate linear
directions. Raw residual fell from 8626 to 6837, but projected residual remained
0.0357 and derivative admission failed. Stop extending post-solve recovery;
inspect pinning from initialization and through multigrid transfers next.
This does not qualify the gauge policy or reject initialization-time pinning.
Reuse accepted-point factors for nearby refinement only with measured progress
and bounded refactor fallback.
Stale factors used as an exact solver already failed: do not revive that idea.
Report factorization counts, correction iterations, refactor rate, final state
and observable differences, compile cost and memory.

Use numerical tolerances for same-point derivatives and final physical quality,
not bitwise optimizer trajectory identity. Tiny linear-algebra differences can
alter trust-region steps; investigate conditioning and feasibility before
attributing different trajectories to inaccurate derivatives or claiming speedup.

Revisit broad inter-rung cache clearing after checking #397's memory behavior
at high resolution. Preserve the memory bound that motivated cache release.
Earlier descent-to-Newton handover follows lane A; if a finish fails, recover
with the original solve policy. No accepted trial may lose its accuracy contract.

## D. QI smoothness and the Boozer dependency

Inspect existing `d1/smooth-qi-well` work before implementing anything new.
Map two-sided Taylor behavior and well-topology changes on public frozen
spectra and real solved states. Separate interpolation/well selection error
from equilibrium noise. Compare one justified smooth alternative with the
current objective; do not start a second omnigenity formulation concurrently.

Use the same seed, constraints and budgets. Check final QI on an independent
fine Boozer grid, accepted progress, failed trials, full runtime and memory;
include a transport-related diagnostic when available. An improved surrogate
must preserve the physical design quality. Do not freeze well minima in
production to make a derivative test pass.

The [paired design record](https://github.com/uwplasma/vmex/pull/413#issuecomment-5755523238)
shows candidate `413d7fd2` failed the same-budget solved-design gate
against `f719c4ff`: fine-grid QI increased from 0.002767 to 0.004558 (65 percent),
although both designs met the stated constraints. It remains experimental.
The [saved-boundary cross-evaluation](https://github.com/uwplasma/vmex/pull/413#issuecomment-5755580667)
now separates score changes from design quality. At identical frozen spectra,
the definitions differ by at most 0.51 percent; the candidate boundary remains
61 percent worse on the fine grid under the current definition. All four
residual families worsen, especially branch width. This supports an inferior
optimization trajectory, not a mere change of metric. Keep the candidate out
of production and stop this surrogate experiment without a width sweep.
The hard-well differentiability limitation remains documented; a new attempt
requires evidence of a better trajectory, not another local Taylor test alone. Both arms used
JAX 0.11.1, Boozer 0.4.0 and SOLVAX 0.24.0. Different shared load and compilation
state prevent a runtime claim from the 295 s versus 205 s measurements.

The PyPI booz_xform_jax 0.4.0 wheel passed four symmetric/asymmetric dense-
reference projection tests, including magnetic-only values and bmnc JVP/VJP
([evidence](https://github.com/uwplasma/vmex/pull/410#issuecomment-5755351706)).
At the proposed Boozer 0.4.0/SOLVAX 0.21.0 floors with JAX 0.9.2,
[VMEX integration](https://github.com/uwplasma/vmex/pull/410#issuecomment-5755560071)
passed all 21 Boozer-table, nine plotting and eight omnigenity tests, with JIT
and full-marked tests enabled. The clean all-extras minimum-version stack uses
JAX 0.10.1. Its numerical checks exposed missing `tprim`/`fprim` drive fields
in GKX 1.7.1; the 1.8.0 floor
correction was added to #410, which merged (`e76930af`) and shipped in 0.11.0. Eleven packaging, four GKX,
one NEO and three NESTOR/adjoint checks pass. The new minimum-version nightly
selection passes all seven tests together; explicit optional-package imports
prevent missing integrations from silently skipping the gate. Static preflight, warning-free
Sphinx build and all six source/HTML navigation checks pass. The dependency
gate of this lane is closed; the QI smoothness gate above is not.

## E. Strong-force polishing

Keep the current axis-regular `rho^|m| q(s)` representation with splines in
`s`; the recorded knot/coordinate changes do not justify a replacement.
E1's complete virtual-work identity passes. E2 succeeds on the shaped tokamak
but misses the 3-D force-reduction gate. Existing resolution scans implicate
axis source data and the lift; increasing the spline basis can worsen the fit.

First audit `lift_high_order_state` for unsupported spans, rank and axis
regularity. Refuse an underdetermined lift rather than filling it silently.
Use manufactured/analytic fields and independent off-grid quadrature to
separate representation error from nonlinear-solver error. Then resume the
bounded E3 correction and radial/angular resolution ladder, using
[`benchmarks/e1_functional_consistency.py`](https://github.com/uwplasma/vmex/blob/07a47279d5cea819bb23c5329fab7f14cced2456/benchmarks/e1_functional_consistency.py), [`benchmarks/e2_dense_reference.py`](https://github.com/uwplasma/vmex/blob/07a47279d5cea819bb23c5329fab7f14cced2456/benchmarks/e2_dense_reference.py),
[`benchmarks/residual_vs_resolution.py`](https://github.com/uwplasma/vmex/blob/07a47279d5cea819bb23c5329fab7f14cced2456/benchmarks/residual_vs_resolution.py) and [`benchmarks/knot_grading.py`](https://github.com/uwplasma/vmex/blob/07a47279d5cea819bb23c5329fab7f14cced2456/benchmarks/knot_grading.py)
(removed from the tree; restore them from that revision).

Promotion requires a non-axisymmetric finite-beta case with positive geometry,
preserved boundary/flux/profile constraints, independently reduced strong force,
nonlinear stationarity and derivative verification. Report dimensional force,
volume-normalized force and near-axis/bulk/edge contributions. The bounded
legacy `eps_F` is insufficient. Separate native-state comparison from WOUT
export/refitting error. Stop a failed bounded attempt and record its cause;
do not repeat multi-hour W7-X attempts without new evidence of progress.

The closed-hybrid force plateau in issue #211 remains an explicit admission
question, not permission to relax a threshold. Broader mirror/anisotropy and
coordinate rewrites remain deferred. The [resolved analytic recovery study](https://github.com/uwplasma/vmex/pull/413#issuecomment-5755801536)
reduces independent force strongly but still misses stationarity after 40 steps.
It consumes 7,927 of 8,000 possible PCG iterations. A
[fixed-endpoint dense comparison](https://github.com/uwplasma/vmex/pull/413#issuecomment-5755874974)
now identifies inner PCG starvation in this case: the dense step reaches
independent `eps_F=6.62e-8`, while PCG has linear residual 0.230 and barely
improves force. The [bounded follow-up](https://github.com/uwplasma/vmex/pull/413#issuecomment-5756143841)
rejects strict inner convergence as a recovery policy (12 rejected trials)
and rank-64 Nyström as the fix (linear tolerance missed; off-grid J worsened).
[SOLVAX #119](https://github.com/uwplasma/SOLVAX/pull/119) exposes inner diagnostics
while preserving useful inexact steps; 15 focused tests pass. These are PCG's
recursive residuals, not independent certificates. The
[saved-matrix LSQR/LSMR comparison](https://github.com/uwplasma/vmex/pull/413#issuecomment-5756201847)
produced no useful physical step at 200 iterations. Even
[2,000 LSMR iterations](https://github.com/uwplasma/vmex/pull/413#issuecomment-5756210776)
missed the 1e-3 true-normal-residual gate (2.416e-3), with direction error
0.987 relative to the dense reference. Stop this solver sweep.
The [bounded production-chart QA diagnostic](https://github.com/uwplasma/vmex/pull/413#issuecomment-5756509703)
now supplies the exact experimental patch and reproduction controls. Its
18,018-by-1,336 augmented solve passed GELSD/GELSY checks below 1e-14;
stored-matrix PCG also converged in 324/600 iterations. One admissible dense
step reduced independent dimensional force L2 by 25 percent, not the required
10x, and did not establish nonlinear stationarity. Total cost was 604 seconds
and 11.13 GiB peak RSS; setup and Jacobian assembly dominated factorization.
Do not integrate a dense production path on this evidence or extend the sweep.
Stored-matrix PCG is not a matrix-free parity check; one step does not identify
a representation limit. A further correction experiment requires a specific
failing production endpoint and a measured cause, with independent physics
gates retained. General 3-D polishing remains unqualified.

## F. Public evidence, documentation and repository maintenance

Use `benchmarks/INDEX.md` to preserve the connection between claims, inputs,
generators and records. For September 19 conclusions held outside the repo,
recover and sanitize the minimal executable reproduction and compact numerical
record in existing benchmark infrastructure. Inspect every field and string;
never copy private reports wholesale. If evidence is unavailable, mark the
claim unverified and schedule only the experiment needed to decide it.

Retain one canonical record per scientific claim. Delete redundant narrative
summaries and obsolete uncited records only after checking code, tests, docs,
figure provenance and generators. Regenerate the index. Large raw profiles and
arrays belong in a versioned research artifact with public provenance, not the
source tree. Preserve historical results through immutable commits. Do not
rewrite repository history to save a small archive.

Delete merged branches only when their remote head still equals the recorded
merged PR head, no open PR depends on them, and no active worktree uses them.
Use expected-head protection against concurrent pushes. Preserve unsubmitted
contributor branches; an old timestamp does not prove abandonment.

Keep tutorials/how-to/reference/explanation organization. Shorten the README
without losing runnable simple and advanced entry points, accuracy scopes and
high-impact results. Use the current public helpers. Synchronize release and
capability statements, and close documented issues only after checking the fix.
#413 removes the unqualified exterior field-line showcase from the README
while retaining its artifact and figure record (now cited by no page). The field explanation and
example docstring match the implementation's near-surface-continuation warning;
source sampling is described as geometry-dependent. No numerical behavior
changed. Independent qualification of the supplied-surface-field variant is
still required before restoring a physical-topology claim.

Documentation claims corrected in #413 on 2026-09-22, each checked against
the code or a measurement first:

- `docs/explanation/adjoint-gradients.md` (validating the gradients): quotes
  the measured m=1 gauge gap (1.0 % / 2.8 % on `li383_low_res`, closure 2e-6,
  test from #428) instead of main's "naive FD matches `jax.grad` to
  rtol <= 1e-6" for bulk integrals, and says free-boundary checks need
  Newton-anchored re-solves.
- `docs/reference/objectives.rst` (Mercier/Glasser rows): the tests use
  frozen-path FD (`frozen_path_directional_fd`), not independently
  reconverged equilibria.
- `README.md` (single stage), `docs/explanation/nestor-vacuum.rst` (coupled
  adjoint) and `docs/explanation/validation.md` ("What is NOT validated"): the
  free-boundary off-root state and the zero-beta island-chain finding.
- `docs/explanation/validation.md` (tokamak polish): explains why the README's
  read-back WOUT numbers (2.9e3 -> 61 N m^-3 near axis) differ from the
  native-state table (9.8e2 -> 6.7e1).
- `examples/hot_restart_scan.py` docstring, `docs/howto/restart-from-previous-run.md`
  and `docs/howto/parameter-scans.md`: "a handful of iterations" and "many
  iterations, not one" replaced by measurements. The example's PHIEDGE scan at
  zero pressure with a prescribed transform takes one warm iteration per point
  (about 300 cold; rerun 2026-09-22). Boundary moves of 1e-4 to 1e-2 on the
  low-resolution QA deck took 212–391 against 806 (Part II §2).

Left to a code PR, because they are docstrings in `vmex/core`:
`vmex/core/implicit.py::frozen_path_directional_fd` still says a naive
re-solve FD of `wb`/aspect matches `jax.grad` to `rtol <= 1e-6`. Its
`li383_low_res` example (adjoint -0.773 against naive FD +0.045 for
`d(iota_edge)/d(RBC(-1,1))`) should be re-measured now that the refined-state
anchor and the m=1 family account are in place. The `freeboundary_implicit`
module docstrings should state the off-root scope once lane A's anchoring
lands.

## Dependencies and publication

#420 (merged `17bd8469`) corrects aggregate warm profiling: schema 1 repeated
only the first stage, omitting transforms or derivatives from multi-stage
workflows. Schema 2 repeats every stage in order. Nine focused JIT-enabled
tests pass, covering all 21 changed executable lines. Remeasure affected warm
aggregates before citing them; separately recorded first-call stages remain valid.

Verified release inventory at this checkpoint: VMEX 0.11.0 (#425), whose
floors are `booz_xform_jax>=0.4.0`, `solvax>=0.21.0`, `gkx>=1.8.0` (turbulence
extra) and `virtual-casing-jax>=0.0.7` (#410); latest sibling releases checked
2026-09-21 were SOLVAX 0.24.0, booz_xform_jax 0.4.0, virtual-casing-jax 0.0.7
and ESSOS 0.17. Review current
source and installed-package interoperability before proposing upgrades.
Keep generic linear algebra in SOLVAX, Boozer transforms in booz_xform_jax,
quadrature/error estimates in virtual-casing-jax, and coil physics in ESSOS.
Open focused upstream PRs for measured defects; ESSOS merges/releases require
manual review. No release solely to refresh a version number.

First close the collaborator case and representative QA/QI comparisons, then
expand the publication matrix to cold/cache-reload/warm CLI and Python runs,
QA/QH/QI/QP across declared NFP/resolution choices, fixed/free-boundary single
stage, CPU/GPU, and final physical accuracy. Use quiet paired runs and actual
device placement; never compare profiled and unprofiled times as a speedup.
The collaborator's 0.3/modified-0.7 report remains unresolved by a controlled
complete comparison. Pin available historical sources; label unavailable modifications
unreproducible rather than inventing equivalence.

Paper 1 can cover validated equilibrium/AD/optimization performance, with
public inputs, commands, dependencies, hardware and compact outputs. General
3-D strong-force improvement is a separate claim requiring lane E. Complete a
JIT-enabled CPU/GPU and optional-dependency checkpoint before either promotion;
per-PR green tests and line coverage alone are insufficient. Keep real public-API
smokes in required CI: #409 escaped fast tests that mocked the broken call.
Investigate the cross-module test-order failure recorded in #390.

Extend #390 stage reuse only to compatible unconverted ladders, checking frozen
variables and final designs. Profile plotting separately from equilibrium work;
measure Boozer/confinement diagnostics and cache-release costs before changing
policy. The GPU guide now labels automatic placement as a historical workload
heuristic; device defaults still need current workload evidence. Issue #157 is
closed after verifying the published correction and executing its resolution
example on the bundled circular tokamak (work 1632, recommendation CPU).

## Continuation logbook

Entries below record observations at the time, not pending instructions.
The current status table and lanes above supersede their earlier checkpoints.

### 2026-09-21: restart from the current repository

Reviewed main `f719c4ff`, current open PRs and sibling releases. Replaced the
obsolete operational queue while retaining historical reasoning by immutable
link. (Superseded the same day by the maintainer: the historical plan is kept
in this file as Part II, see the rework entry below.) Identified #409 as the immediate public-call integration correction;
do not duplicate it. Started isolated derivative-contract, factor-reuse and
QI investigations. Free-boundary qualification and polishing remain required,
not silently deferred by the performance work. All six lane gates above remain
open unless a later entry links the final implementation and evidence.

Removed 58 unchanged, merged branches after checking
open PR dependencies and worktree use, with expected-head leases. Tags and
commit history were retained. Another 37 candidates were already absent on recheck; no removal is attributed
to this work. Unsubmitted and active contributor branches remain protected.

Open #410 now owns dependency-floor updates, #411 the revised single-stage
example split, and #412 the same-mesh polishing comparison. Do not duplicate
them. At this checkpoint the Boozer package index still listed 0.3.0;
the later released-wheel qualification is recorded in lane D. Draft #414 (`edf779ca`) rejects unresolved radial lifts using the rank of the
existing least-squares factorization. Four analytic tests and two existing
lift/certificate integration tests pass. A Linux CPU run at the exact PR head
then passed the four analytic tests under coverage.py: all four changed
executable statements covered, none missing. This is changed-statement
coverage, not full-module or branch coverage; repository CI remains pending.
This does not close the 3-D polishing gate. #414 later merged as `311e7ddd`.

### 2026-09-21: scientific contract review

A focused controlled recovery test reproduced a remaining free-boundary
history dependence: unrelated failed rebuilds exhaust the configuration-wide
budget and change the answer at a previously successful point. The candidate
allows one cold retry per stalled call; physical validation and final
status-aware checks remain required before promotion. Derivative/state-identity
work was then unpublished; its subsequent implementation and coverage are
recorded under #417/#418 in lane A.

[SOLVAX #117](https://github.com/uwplasma/solvax/pull/117) at `a217447` retains
original coefficient bands for checked factored solves; reconstructing them
from factors would lose information through cancellation. The focused suite
reports 127 passing tests. VMEX integration and end-to-end gains remain open.

The bounded QI experiment found a well-selection transition with a one-sided
Taylor remainder floor. One subcell alternative removed that local floor,
but changed the fixture objective by 0.535 percent. No production change or
optimized-design/transport qualification follows from that experiment alone.
The [public frozen-spectrum probe and measured Taylor table](https://github.com/uwplasma/vmex/pull/413#issuecomment-5755290878)
reproduce the baseline switch without a new repository file. Historical
environment details were not captured; recheck the transition on the installed
stack. The September 19 factor-reuse reproductions required by lane F remain
outstanding.

Draft #415 removes two duplicated benchmark narratives, retaining the JSON
measurements and validation documentation. Citation and performance-documentation
guards pass (38 tests); no unique measurement or numerical code is removed.

### 2026-09-21: recovery candidate and released-wheel validation

Draft #416 (`85517771`) removes history-dependent rebuild suppression, with
one cold retry per stalled call. The eight-failure controlled regression and
a real asymmetric repeated-point test after a rejected doubled-current trial
pass. The agent reports 20 non-full module tests, 100 percent changed-line
coverage and passing preflight; required repository CI and parent final review
remain promotion gates. This restores repeatability, not free-boundary
pressure-gradient accuracy or finite-beta design feasibility.

Boozer 0.4.0 is now published on PyPI. The isolated released-wheel checks above
passed on CPU with Python 3.11.14 and JAX 0.9.2/x64; both fixtures match the
release tag. SOLVAX #117 now reports 128 passing focused tests at `9c6f1a17`.
The revised VMEX scalar-factor cache still failed: two 151-surface pairs
regressed by 0.6–1.2 percent and increased carried cache storage by 77 percent.
The implementation was withdrawn; exact-state parity does not justify a slower,
larger cache. Shared-load timings are diagnostic, not portable speed estimates.

### 2026-09-21: plan rework keeps the full history in this file

The maintainer rejected replacing the historical plan with links to older
revisions: the code stays slim, but the plan must carry enough context for a
complete handoff. The historical plan therefore returns in full as Part II,
with superseded sections marked; the current material above is Part I. The
only text removed from the historical plan is a host alias and private lock
paths (§7 and §9), which must not be published. #413's documentation changes
were reduced to factual corrections checked against the code: the adjoint
equations and the default block-transpose adjoint with its Krylov fallback,
preconditioner recomputation, the column certificate on the raw operator,
the unqualified near-surface continuation, geometry-dependent default source
sampling, the automatic GPU placement as a workload heuristic, and the scope of
the implicit and free-boundary derivative claims.

### 2026-09-21: derivative stack record (superseded 2026-09-22)

> **Superseded 2026-09-22.** The independent review found the stack's premise
> not reproduced on main; its one real defect landed as #427 and the stack is
> to be closed. Lane A carries the current verdict. Kept as the record of what
> the stack implemented and measured.

Draft #417 (`1459f9df`) is rebased onto `45f3a7ae`; its 21-test suite passes
with 98 percent changed executable coverage (319/325). Remaining uncovered
lines are direct multi-RHS guards. Contributor-owned #418 is now restacked at
`9e0baa28` and reports 18 passing focused tests on that exact stack. Review the
pair together: direct calls must retain finite-state, geometry and FSQ admission
when no absolute tolerance is supplied. Full optimization and GPU gates remain open.
Callback placement #421 (`95aed755`) now includes #418 and fresh certificate
alignment. PR #422 (`e25aa1fd`) merged into the #421 feature branch at `e94e46c6`,
with only complementary tests; it has not delivered that stack to main.
Eleven forced-two-CPU cases pass,
covering caller/runtime mismatch, explicit-device precedence and exact returned
coefficients. Instrumented measurement hooks do not certify a physical root.
The earlier GPU-facing check used a CPU host root; accelerator-resident root
and complete optimization qualification remain open on the combined stack.
Draft #423 (`c5fa14d8`) fixes a further certificate mismatch: measure the
supplied state directly and reject inconsistent fixed edges separately from
geometry, rather than silently assembling a different edge. The regression
fails on its parent; 55 focused tests and 11 forced-two-CPU cases pass, with
12/12 changed executable source lines covered. A real small JIT-enabled root
passes. Follow-up `02bf33d9` admits transform roundoff and aligns all public
measurement inputs before either check, preserving the supplied coefficients.
Eight focused cases cover ulp/material changes, geometric scaling, staged
asymmetric setup and cross-device measurement; all nine added executable source
statements are covered. The combined penalty/device suite passes all 71 tests
on JAX 0.10.1/SOLVAX 0.21.0 with two CPU devices. A subsequent
[explicit CPU/CUDA callback check](https://github.com/uwplasma/vmex/pull/423#issuecomment-5757198007)
on NS5/MPOL3 Solovev passed on JAX 0.11.1/SOLVAX 0.22.0, with coefficients
agreeing within 3.2e-15 and repeated certificates identical. Template placement
was verified; native result placement was not independently recorded before
host conversion. Asymmetric accelerator and complete optimization gates remain.
Draft #424 (`7de9f45d`, based on #423) addresses a newly reproduced public-API
regression: factory preflight caches a native-only solve, so immediate
`equilibrium_from_x(x0)` skipped certification and then failed. The draft
refreshes missing/stale evidence through the status callback and reuses valid
certificates. It also removes cache-hit-count dependence from penalty fault
injection. The original materialization failure is reproduced independently;
the candidate's numerical suite and coverage remain unverified. Its PR includes
commands and evidence for continuation. Work is handed off without further
runs or subagents at the maintainer's request; no merges are authorized.

### 2026-09-22: VMEX 0.11.0, the free-boundary audit and the vacuum finding

VMEX 0.11.0 was released (#425, `780eb86e`), carrying #403, #407, #408, #409,
#410, #412, #414, #416 and #420. #427 (`equilibrium_from_x` returns the
refined state; objective and WOUT mean iota now agree to 1e-12, where main
differed by 1.6e-3) and #428 (closure test for the mean-iota gap) merged after
it.

The independent review of the #417 stack did not reproduce its premise on
main; the stack and #419 are to be closed and #371/#377 are superseded by
#411. The follow-up audit of the free-boundary adjoint (lane A) found it
exact at the root and 4–68x faster than finite differences, and located the
remaining problems in the forward state: no Newton anchoring (about 1.2e-2 off
the root), a seed-reference restart that burns about 80 % of each late trial,
and start-dependent values at finite beta. The mean-iota gap is the m=1 gauge
family, not a missing term. Anchoring and a restart budget are in progress.

The zero-beta free-boundary limit cycle in #411 was traced to a 4/9 island
chain at the boundary of the optimized coils (lane B), reproduced by VMEC2000
and VMEC++; an iota ceiling at 0.44 in the example removes it, and all four
single-stage examples now meet their targets. Tiny-beta regularization was
tested and rejected.

The maintainer decided: bake #426's optimization defaults into main; set
finite-beta beta through PHIEDGE normalization instead of a pressure loop
(in progress); defer the bootstrap single-stage scripts. The release deferral
stated in "Execution and acceptance" was superseded by 0.11.0 and restated for
0.12. This entry also corrected stale documentation claims listed in lane F.

### 2026-09-22/23: review, fixes, docs, and the pause

- Merged: #427 (`equilibrium_from_x` returns the refined state: iota 0.551556
  vs 0.552562 on 0.11.0), #428 (m=1 family closure test; the defect-pinning
  `gap > 3e-3` assertion was dropped), #413 (this plan), #429 (docs: 54 -> 44
  pages; README/landing explain methods, goals, results).
- Free-boundary adjoint audit (#417 comment 5784815035): exact at the root
  (1e-7..1e-9 vs FD of Newton-anchored roots), +5-9 s per gradient, 4-68x
  faster than FD, no leak. Two practical defects: the returned state is off the
  root, and ~80 % of trial time near the optimum is a restart run to its cap.
  #432 found why `ftol` is not a root test here (sum of squares; the edge row
  enters `getfsq` only for 50 iterations after a restart, as in VMEC2000's
  `residue.f90`): converged states sat 3.5e-2 to 1.3e-1 from the root, mostly a
  poloidal-angle relabelling pinned only by the weak spectral-condensation
  force, and the objective there was 0.2-17 % off.
- Vacuum free boundary: the limit cycle was a 4/9 island chain at the
  requested flux (field-line tracing; VMEC2000 and VMEC++ fail the same way;
  Landreman-Paul coils converge cold at zero beta); fixed in #411 with an iota
  ceiling. Tiny-beta regularization rejected.
- PHIEDGE: beta depends only on `PRES_SCALE/PHIEDGE^2` at zero current, so
  PHIEDGE is set in closed form at the target aspect with one correction solve
  (#426). For free boundary the coils fix |B| and PHIEDGE sets plasma size, so
  the analogue is choosing `PRES_SCALE` from the coil field.
- Single-stage runs (#411, loaded laptop): fixed zero beta 335 s and fixed
  0.5 % beta 566 s meet every target; free 0.5 % beta 2399 s meets every
  target; free zero beta with the iota ceiling meets every target (~1900 s).
  All exceed the new five-minute limit; bottleneck analysis and speedups are
  in the #411 handoff comment.
- `benchmarks/`/`tools/` audit: neither ships in the wheel; `benchmarks/`
  goes from 132 to 70 files (#431); `tools/` keeps 17 of 18 (CI lane selection,
  doc guards, asset fetch). Deleting files does not shrink clones (history is
  36.6 MiB; `--depth 1` is 4.7 MiB; `docs/_static` is 43 % of history).

### 2026-09-22: the exterior and interior fields against independent oracles

A validation pass over virtual casing (VC), its derivatives and the extender
at `604e6a76`, with virtual-casing-jax 0.0.7 from PyPI (#430). The references are
independent of the code under test: an own surface-integral evaluator on a
target-graded periodic trapezoid rule (E7's substitution, two refinements
agreeing to 2e-9 in B and 2e-6 in grad B down to `d = 0.01 a`), a volume
Biot–Savart integral of `curl B` of the interior field, the coil field of
free-boundary equilibria, and `frozen_path_directional_fd`. Timings were
taken with other jobs on the laptop (load 4–90) and are upper bounds.

- **Validated.** Virtual casing of the native-form LCFS field equals the
  volume Biot–Savart field of the interior current to 3e-12 on the 2.5 % β QA
  deck (six Gauss points per radial spline cell; three leave a 1e-6 floor),
  which ties the exterior path's formula, signs, normal and nfp replication to
  the interior field; the asset-free version is now a PR test. The shipped path
  equals the own evaluator to 1e-14. At the grid `from_state` picks for that
  deck (64 × 64 per period) the plasma field is right to 1e-13 at `d = a`,
  1e-6 at 0.5 a, 2.4e-2 at 0.2 a and O(1) from 0.1 a in; the order-0 estimate
  is 0.76–1.09 × the true error, so it tracks rather than bounds it, and it
  flags every unresolved point. Nested-AD derivatives equal the closed-form
  kernels to 3e-12; the a-priori per-order estimate is never below the true
  error of orders 1–3 and 3–6 × (order 1) to 14 × (order 3) above it.
  Surface-data → field derivatives match central FD to 1e-9 (geometry) and
  1e-11 (field); state → exterior and interior field to 6e-8 and 2e-6;
  on-surface VC to 2e-8; the linearized-VC JVP columns of the functional API
  to 5e-9. On free-boundary CTH-like equilibria VC at interior points equals
  minus the MGRID coil field to 3–5e-4 (vacuum) and 4–7e-4 (β = 0.19 %), the
  interior field equals the coil field to 2e-4–3e-3 in vacuum, and coil plus
  VC just outside matches the interior field at the LCFS to 4e-4–1e-3 (vacuum)
  and 3–6e-3 (finite β) — floors set by MGRID interpolation and ns = 31, not by
  VC. Interior field on the breathing circle: B 2.5e-8 → 5e-14, grad B
  2.8e-6 → 3e-8, grad-grad B 2.2e-4 → 1.2e-5 over ns = 41 → 161, div B at
  round-off, curl B within 5e-8 of the exact current.
- **Derivatives in problem parameters are exact for the frozen map, and the
  re-solve map differs.** On `li383_low_res` the implicit reverse pass through
  equilibrium, live-state surface data and VC matches the frozen-path FD to
  4e-7–1e-6 on three boundary directions and 3e-10 on the current (interior
  field: 1e-5–3e-4 and 2e-10). Independent re-solves differ by 2–4× on the
  boundary directions (interior 6 %–110 %) and by 3e-3 on the current; on the
  QA deck by up to 10×. That is #428's m = 1 gauge drift, which the exterior
  field feels far more than iota does; the re-solve states themselves move
  with warm-start history (λ by 5e-3, R by 3e-5 at one point). A new weekly
  test pins the frozen-path agreement. Consequence for lane A: an optimizer
  that line-searches the production map on an exterior-field objective is
  not following the derivative it is given.
- **Fixed** (#430). `_mgrid_from_wout` built an identically zero
  coil field from a wout that names an MGRID but carries no currents, which
  is what `solve_file` writes for a free-boundary deck (`multigrid.py` does not
  pass the free-boundary metadata that the CLI passes; left to that file's
  owner); it now raises. The per-order estimate returned NaN with hundreds of
  RuntimeWarnings for targets tens of minor radii out, and the eager
  derivative check then warned "error up to inf" where the field is exact;
  fixed upstream (virtual_casing_jax #15: asinh Newton start, clamped root,
  unknown-near-is-inf) and released as 0.0.8, now the `freeb` floor. `project_current` is
  reachable from `from_wout`/`from_state` (default off). The continuation's
  docstring blamed it for disagreeing with a direct value on a current-free
  deck, where both numbers were errors; it now carries a finite-β record.
- **The near-surface continuation** (E2's retired Taylor plan) is approximate
  and not affordable: 1.6–2.4 % of the plasma field (about 1e-3 of |B|) at
  every distance from 0.01 a to 0.1 a — a floor from its bilinear table —
  3.9–6.6 % at 0.2 a and 18–32 % at 0.5 a, with 196–397 s and 18.5 GB to
  prepare at 32 × 32; at the 64 × 64 default the process was killed for memory
  on a 36 GB machine. It is not needed above 0.5 a. E7's graded rule is the
  replacement: the reference above is that rule, 0.2–0.4 s per target in
  unoptimised NumPy; remove the continuation once E7 lands.
- **The curl-free projection default (#381) should stay off.** The removed
  part of the covariant pair sits just above the resolved modes (m = 8, 9 at
  mpol = 7) and does not shrink with ns (QA, 6e-3 of the gradient part at
  ns = 31–201). Against a joint (mpol, ns) = (12, 128) reference on li383,
  projecting leaves the plasma field's error unchanged or raises it by up to
  1.6×, and raises grad B's by up to 1.9× (mpol 6–10, d = 0.2–1 a); only on an axisymmetric deck, where the curl
  is a radial discretisation error (3.6e-4 at ns = 51 → 7.8e-5 at 401), does it
  help, by about 5× at ns = 51–101. It makes the exterior field curl-free, which is
  a physical requirement, but it does not make it closer to the equilibrium's
  field at practical resolution; keep it opt-in.
- **Open.** E7 in the library (point queries and E2's table). E5's wiring:
  the closed-form kernels are 5–15× faster warm for all four orders, but their
  first call costs 17 s because `level_sources` builds the densities eagerly;
  fix that upstream first. The interior third derivative stalls at 6e-4 on
  the breathing circle (cubic splines have no fourth radial derivative); a
  quintic interpolant would restore convergence, below the E8 gate today.
  The order-0 estimate can sit 1.3× under the true error.

### 2026-09-23: the graded near-surface rule replaces the continuation

E7 is in the library as `virtual_casing.graded_plasma_field` and as
`VmecExtender`'s `near_surface` mode ("auto" by default: eager calls switch
each point whose direct estimate misses `10**-digits` to the graded rule;
"graded"/`with_graded_quadrature()` everywhere, traceable; "direct" as
before). The rule interpolates the one-period surface samples spectrally,
grades about the target's nearest surface point with local spacing d/8, and
takes B and its derivatives from virtual-casing-jax's closed-form layer
kernels. Measured at 128 × 512 nodes on the 2.5 % β QA deck: 1e-12 in B and
1e-10 in grad B from d = a down to 0.01 a (3e-8 at 0.003 a) against the
converged reference; on the two-source torus oracle 6e-10 of the field scale
1 mm from a 0.3 m surface, inside and outside. `with_near_surface_continuation`
and `near_surface_plan` are removed; the tracing example traces through the
graded field at 64 × 256 nodes (about 4 ms per point).

E5 is wired: direct-path derivatives use the closed-form kernels on the finest
level (same values to 1e-12). The 17 s first call was `level_sources`
dispatching eagerly; VMEX now compiles it ahead of time (0.5 s). One trap:
`jax.ensure_compile_time_eval` also dispatches eagerly (26 s), so cached data
are built with `jax.jit(...).lower().compile()()`, which runs outside any
caller's trace. A/B/A/B on the office workstation
(`benchmarks/extender_ab_20260923.json`): first `B`…`gradgradgradB` calls at
16 targets 6.0 s → 2.9 s; warm `gradgradgradB` 0.50 → 0.09 s (16 targets) and
1.98 → 0.47 s (128); near-surface `B` at 16 targets 0.05 a out costs 0.42 s
warm instead of 0.04 s, and is right where the direct value was 25× too large.

Still open: traced calls under "auto" use the direct path (a per-point switch
under a trace would pay for both); the per-order switch uses the a-priori
estimate, 3–14× conservative, so some resolved points take the slower rule;
E2's table for long exterior traces; the default source grid is still sized
for d = a only.

# Part II. Historical plan and logbook (2026-09-13 to 2026-09-20)

> **Status of Part II.** This is the plan as it stood on main at `f719c4ff`,
> kept in full so that its reasoning, measurements, disproved hypotheses and
> decisions remain readable here. Where Part I disagrees, Part I is current.
> Superseded sections carry a note naming what superseded them; everything
> else (evidence, root causes, kill rules, force-balance decisions, logbook)
> remains the record. Text is unchanged except for those notes and the removal
> of private host aliases and lock paths from §7 and §9.

Authoritative plan, revised **2026-09-13** by an independent review of the
2026-09-06 plan (merged as [#283](https://github.com/uwplasma/vmex/pull/283),
readable with its full logbook at
[`f09288b3`](https://github.com/uwplasma/vmex/blob/f09288b37bac7d7122e98220192795e770cd1f/plan.md))
and of the performance branch behind [#299](https://github.com/uwplasma/vmex/pull/299).
Base: main at `f09288b3`, 0.8.1. It is a planning change: no equilibrium
algorithm, derivative guarantee or performance claim is promoted here. The
evidence behind every new number is in
[`benchmarks/review_20260913.json`](benchmarks/review_20260913.json), produced by
`benchmarks/review_20260913_equilibrium.py` and
`benchmarks/review_20260913_exterior.py`; older numbers cite their records.

**Revised 2026-09-19** by a second step-back review the maintainer ordered on
the seven areas he named: free boundary and its derivatives (report L-FB), the
VMEC extender (L-EXT), cold runs and the single solve (L-SOLVE), the
optimization loops (P-OPT), the repository census (SLIM-A), source and tests
(SLIM-B), and one example template with its correctness pass (EX). Base: main
at `f86cd13a`, 0.9.1. It is a planning change: no equilibrium algorithm,
derivative guarantee or performance claim is promoted here. Every number added
on 2026-09-19 names the report that measured it; the reports, their scripts and
their JSON live in the review's evidence set `vmex-review-evidence-20260919`,
which is not in this repository. Where a measurement disproves a claim this
plan made, the old claim stays in place and is marked disproved rather than
deleted.

**Release hold.** No tag, version bump or publication date is scheduled. One
release follows the gates of §4 and §5.

> **Superseded 2026-09-21.** VMEX 0.9.0, 0.9.1 and 0.10.0 have since been
> released. The next release waits for the integration and research gates in
> Part I ("Execution and acceptance").

## 0. How to use this plan

> **Superseded 2026-09-21.** Resume from the Part I preamble instead. §3, the kill rules of §4, §5 and §9 remain evidence; §6 and §8 are historical.

An agent resuming this work reads §1, §3, §4 (phases and PR list), §6
(dispositions and release candidate), §7 (environment), §8 (the coordination rules and the brief it
was given) and the last entry of §9, in that order. It checks the remote PR
state and dirty worktrees, and continues at the first unmet gate. An agent
given one brief of §8 needs nothing else from the conversation that produced
it. After each merged implementation PR the coordinator appends one logbook
entry in the format of §9 and updates the affected gate in place. It does not add a work
package because another code has a feature, does not rerun the historical
inventories, and does not launch a multi-hour run before the counters of
Phase A exist.

## 1. Decision: fix what users measure

> **Superseded 2026-09-21.** The four complaints remain the motivation. The A–H priority table is replaced by Part I lanes A–F: A (counters) and C (derivative sizing, #390/#392) are delivered; B maps to Part I lanes A and C; D to lane D; E to the native interior field and exterior-accuracy rows of the status table; F to lane B; G to "Dependencies and publication"; H to lane F and "Example scope".

Users report four things: quasi-isodynamic optimization is slow; single-stage
and free-boundary single-stage optimization are slow and end without a valid
design; the field outside the last closed flux surface is inaccurate and slow;
and VMEC++ is faster. The 2026-09-06 plan put none of these on its critical
path (its next five weeks were E3, the resolution ladder and the Taylor test)
and the performance branch behind #299 spent twenty-nine commits on the
Jacobian kernel while its own record shows the equilibrium *anchor* dominating.
This revision reorders the work around the four complaints, keeps the
force-balance research line (§5) as the second product, and adds the item the
old plan lacked entirely: an accuracy contract for the exterior field.

| Priority | Deliverable | Exit decision |
|---|---|---|
| A | Counters, known-answer oracles and honest examples | every optimization record splits solve, refinement, Jacobian, adjoint and compile; the exterior field reports its achieved digits; the shipped single-stage example meets its stated targets or fails loudly |
| B | One certified equilibrium solve per trial | the objective at a repeated `x` agrees to 1e-9; a gradient call costs at most half of today's; the joint single-stage phase no longer exits with precision loss |
| C | Derivatives sized to the problem | one linear solve per degree of freedom or per gradient, never per residual row; batched by default |
| D | A QI example that converges: near-axis seed, then a smooth objective | the QI example reaches today's final metric in at most a third of the time with zero failed trials |
| E | An exterior field with a stated error | the same-data identity is below 1e-6 at every distance down to 0.005 minor radii at the default node count, the source data conserve current, and every derivative order the API exposes carries an estimate; tracing outside the plasma in seconds (restated 2026-09-19: the 1e-6 *vacuum* identity is unreachable from ns = 50 source data, whose floor is 1e-5 near the surface — L-EXT 1.6) |
| F | A free-boundary gradient that costs at most three forward solves | the free-boundary single-stage example converges to a stated design in under ten CPU minutes |
| G | The published comparison and the paper-1 package | cross-code table from committed records on named hardware; then §5's E3 and ladder |
| H | One example template, and a repository that carries no dead weight | every shipped example reads the same way and every one of them is run or exempt with a reason; each module is at or above the 95 % coverage gate; the PR test lanes stop running 1,979 tests 3,055 times |

A → B → C is the order; D, E and F are independent of each other and start
after A. §5's force-balance line resumes after G. H runs beside them: its
correctness items are prerequisites for trusting any example-derived number,
its slimming items are not. No release before every open PR is merged or
explicitly deferred and the gates of A–F hold.

## 2. Evidence at the baseline

Measured 2026-09-13 on one Apple-silicon laptop (14 cores, shared, load 3–11)
unless a record is cited; timings are diagnostic samples, not rankings. Rows
tagged **L-FB**, **L-EXT**, **L-SOLVE**, **P-OPT**, **SLIM-A**, **SLIM-B** or
**EX** were measured on 2026-09-19 at 0.9.1 by that report of
`vmex-review-evidence-20260919`, on machines other sessions were loading
(load 10–60, briefly above 100): their wall times are upper bounds, and their
counts, ratios, identities and error norms are not affected.

**Equilibrium solve against VMEC++** (`input.LandremanPaul2021_QA_lowres`, nfp = 2,
MPOL = NTOR = 8, ns = 50, FTOL 1e-11, four threads; the shipped 1e-13 is
unreachable for both codes):

| case | VMEC++ 0.7.4 | VMEX, JAX 0.11.1 | VMEX, JAX 0.9.2 |
|---|---|---|---|
| single grid, cold | 0.74 s / 806 it | 4.0 s | 7.6 s |
| single grid, warm | 0.74 s | 1.9 s / 806 it | 3.2 s |
| three-rung ladder, cold / warm | 0.95 s | 11.6 s / 2.5 s | 25 s / 7.7 s |
| boundary moved 1e-2, hot restart | 0.72 s / 813 it (no saving) | 0.98 s / 391 it | 1.8 s |
| boundary moved 1e-3, hot restart | 0.74 s / 910 it (worse than cold) | 0.63 s / 272 it | — |
| boundary moved 1e-4, hot restart | 0.39 s / 472 it | 0.51 s / 212 it | — |

VMEC++ 0.7.4 is 2.6× cheaper per iteration than VMEX warm (0.9 vs 2.4 ms) and
12× cheaper than VMEX cold on the ladder; VMEC++ 0.5.3 was 8× slower than 0.7.4
on the same deck, so comparisons against old wheels are void. VMEX's
`profil3d`-spread hot restart beats VMEC++'s interior-only restart at every
step size, so inside an optimizer loop the two solves cost about the same
(0.5–0.6 s vs 0.4–0.7 s). **The per-equilibrium gap is cold compile and the
per-iteration constant, not the loop; the per-optimization gap is what
surrounds the solve.** `mode="jit"` and `mode="cli"` cost the same warm (2.5 vs
2.1 s); the earlier 2× reading was load noise.

**Where a cold run goes at 0.9.1** (L-SOLVE 1.1, every JAX trace, lowering and
backend-compile event recorded from `jax.monitoring` with its program name):

| deck | cold process | import | trace | lower | XLA compile (programs) | exec + Python | warm-cache process | warm in-process solve |
|---|---|---|---|---|---|---|---|---|
| `input.solovev`, ns 11 | 5.42 s | 0.55 | 0.44 | 0.35 | 3.45 (150) | 0.30 | 2.14 | 0.038–0.059 |
| `input.li383_low_res`, ns 16 | 6.09 | 0.72 | 0.44 | 0.36 | 4.01 (187) | 0.32 | 2.13 | 0.075–0.080 |
| `QA_lowres`, FTOL 1e-11 ladder | 15.65 | 0.43 | 1.14 | 0.74 | 9.32 (413) | 3.78 | 7.13 | 2.47–3.2 |
| `input.cth_like_free_bdy`, ns 15 | 17.46 | 0.57 | 1.40 | 1.12 | 11.49 (245) | 2.45 | 7.21 | 1.87 |

A cold run is **60–66 % XLA compile**, 10–12 % tracing and lowering, 2–24 %
execution, and **40–50 % of the compile seconds are single-op eager programs**,
not the solver lanes: on the QA ladder 30 named-lane programs cost 5.64 s
against 383 one-op programs costing 3.68 s. Attribution by source line:
`setup.py::radial_grids` 9.2 % of compile seconds, `setup.py::flux_profiles`
9.1 %, `residuals.py::_m1_rotate_threed` 7.1 %, `setup.py::interior_guess`
4.2 %, `solver.py:2345` 1.9 %, `solver.py:914` 1.2 %. One first eager dispatch
costs 19.7 ms against 86 ms to compile sixty of the same ops as one program
(13.8×), and their warm *execution* is 50 ms of a multi-second solve, so this
is a compile cost and not a run cost. The CLI asks for 413 compilations where
229 are distinct: 118 of the difference is the inter-rung `jax.clear_caches()`
that `release_stage_cache=True` triggers between rungs, and about 65 is the
CLI-only export and summary path. **There is no measurable change in the single
solve between 0.8.1 and 0.9.1** — the 0.9 work went into the optimization path.
Against the peers, one process each (L-SOLVE 1.2, load 60–100, upper bounds):
VMEC++ 0.7.4 runs `input.solovev` in 0.003 s inside a 0.22 s process where VMEX
takes 0.04 s inside a 4.0–5.4 s process, so **VMEX's own executable beats both
peers in-process on the small decks and loses by 9–25× per process, all of it
start-up**; VMEC++'s whole process is shorter than VMEX's import alone
(0.22–0.29 s against 0.43–0.72 s). On the QA ladder VMEC++ single-threaded is
2.14 s, so VMEX warm is within 1.2–1.5× of one-thread VMEC++ and 2.5–3× behind
its four-thread run: that closes open item L1.

**The compile counter reads zero, intermittently, and the harness cannot tell
why.** Main's c1 parity lane is red on and off from a pre-existing flake:
`tests/test_profile_workflows.py::test_compile_counting_and_warm_contract`
fails `assert 0 >= 1` — the counter reads zero — alternating across recent main
runs (failure at `80707817`, success at `4d822c40` and `456b3546`, failure at
`f86cd13a`), and the weekly lane's
`test_cold_and_cache_reload_subprocess_regimes` fails `assert 0 > 0` in the same
file. The harness cannot distinguish *nothing compiled because the cache was
warm* from *the counter is broken*, so neither the red lane nor the green one
carries information; and the same counter feeds committed benchmark records, so
a silent zero corrupts data and not only tests. It is being fixed separately,
and it is recorded here because **every compile-side gate in this plan — B4,
S1, S2, S4, P1 — is stated in compile counts.**

**The per-iteration constant** (L-SOLVE 3). Thread scaling on the office Xeon,
warm QA ladder, `taskset`-pinned, three repeats: 8.63 s (1 core), 6.22 (2),
5.32 (4), 5.34 (8), 5.49 (18) — **the iteration stops scaling at four cores**,
and at 18 cores burns 63 % more CPU seconds for the same wall time, while
VMEC++ gains 2.2× from one thread to four. The HLO census of `_block_lane` at
ns = 50, mpol = ntor = 8 counts 22,474 lines, **844 fusions**, 20 dots and 14
`while` loops, of which **570 serial Thomas sweep trips per iteration** are the
radial preconditioner. Caching that factorisation — the bands refresh on the
`ns4 = 25` cadence but the forward sweep runs every iteration, twelve times —
is 2.7× cheaper per apply at ns = 16, 4.6× at ns = 50 and 4.4× at ns = 201,
with max |difference| 0.0e+00 against today's Thomas: about 8 % of the ns = 50
iteration and 12 % at ns = 201. The checked wrapper costs 30–60 % of the apply
in isolation but 1.7 % of the whole iteration (2.423 against 2.382 ms/it, final
`fsqr` identical to the last digit), so the lever is factor reuse, not the
checking. **B5's sharding arm is disproved as written**: the 2.4× per-iteration
gap is not parallelism `shard_map` could recover, because XLA already occupies
four cores and the machine gives nothing beyond them; the 844 fusions are the
rest, and nothing measured on 2026-09-19 closes that gap.

**Where an optimization step goes** (committed records):

| item | value | record |
|---|---|---|
| QA value + gradient, 48 dof | 7.63 s = 0.35 s descent + 3.86 s Newton refinement + 3.06 s adjoint | #266 |
| constructed QI, 8 dof, CPU | build 18.8 s, first derivative 45.5 s, then 22.8 s for `nfev = 5` (two failed trials); 7 host solves, 14,484 descent iterations; warm residual 0.06–0.17 s, warm Jacobian 1.2–1.4 s | `review_optimization_20260912.json` on the #299 branch |
| same, RTX A4000 | build 110 s, first derivative 187 s; 7× slower than CPU end to end | same |
| QA least-squares startup, 48 dof, 6,723 rows | build 18.2 s + compile 22.1 s; warm value + gradient 16.6 s | [`qa_optimization_startup_least_squares_m4.json`](benchmarks/qa_optimization_startup_least_squares_m4.json) |
| collaborator full run, CPU | 672 s: stage 1 82 s / 63 nfev; coils only 201 s / 1,845 nfev; joint BFGS 336 s / 188 nfev / 176 njev, exit "precision loss" | #299 record |
| warm campaign F8 | 5 evaluations in 146.6 s with 102 compiles and 1,890 traces, 11.9 GB | [`baselines/m4/F8_warm.json`](benchmarks/baselines/m4/F8_warm.json) |
| constructed QI, 8 dof, this laptop, JAX 0.9.2 | build 11.4 s, first derivative 61.0 s, contract check 80.3 s, one solve of 185 iterations; no split between descent, refinement, block assembly, per-column GMRES and compile exists in the record | [`review_20260913.json`](benchmarks/review_20260913.json) |

**The shipped optimization examples, run unmodified at 0.9.1** (P-OPT 1, fresh
process, empty directory, four threads, compilation cache disabled):

| run | wall | compile | where the warm phase goes |
|---|---|---|---|
| `QI_optimization.py`, full | 263.3 s, nfev 20 / njev 11, 0 failed trials, final cost 0.0311670 | 132 s = 50 % (1,421 XLA programs) | warm least squares 71.9 s: descent 33.3 s (17,251 iterations), refinement 16.4 s (19 factorisations), Jacobians 18.0 s; 9 of 20 objective calls were rejected trials costing 28.7 s |
| `QA_optimization.py`, full | 567.9 s, three stages at 8 / 24 / 48 dof, cost 21.45 → 1.74e-4 | 301 s = 53 % | the stages' first residual-and-Jacobian calls are 99.3 / 113.1 / 82.1 s against a ~5 s warm equivalent: **276 s = 49 % of the run is per-stage recompilation at identical array shapes** (`MINIMUM_MPOL` pins mpol = ntor = 5 for every `max_mode`) |
| `single_stage_optimization.py`, capped at 600 s | 163 joint trials, 3.3 s median | 46 s = 8 % | per trial: descent 28 %, refinement 36 %, block adjoint and objective 36 %; **the coil side is 1 %** (14 s of 600), so the ESSOS reverse pass cannot move this example |
| one warm QI point | trial + Jacobian ≈ 4.8 s | — | **≈ 97 % block assembly and factorisation**: objective rows 0.02–0.03 s, one raw block assembly 1.4–1.8 s, Newton finish 1.79 s of which 1.69 s is that factorisation, Jacobian 3.0–3.9 s at the shipped `jacobian_batch_size=1` |

Across QA, QI and single stage, **60–78 % of every trial descent runs between
FSQ 1e-9 and FTOL 1e-12**, while the shipped block Newton finish crosses that
gap in three GMRES steps costing 0.10 s.

**Objective replay on public data** (QA terms of `benchmarks/optimization.py`,
8 dof, warm excursions of 1e-3, 1e-2 and 1e-1 then return to `x0`; loaded
machine, three jobs): with the default refinement the return drift is
2.1e-7 / 1.7e-7 / 2e-8 and each evaluation takes 14–20 s; with
`refine_tol=inf` the drift is 8e-8 / 1.1e-7 / 1.1e-7 and each evaluation
takes 0.4–0.9 s; cold solves reproduce to 2–3e-8 and take 15–17 s. On this
case refinement buys no reproducibility and costs about twenty solves per
evaluation. The 1.5 % drift in the #299 record is on a QI objective whose
well locations move with the solve, so that drift is objective sensitivity,
not solver noise (Phase D), while the refinement cost is the anchor (Phase B).

**Jacobian batching probe** (QA, 48 dof, 6,722 rows, same loaded machine):
warm block Jacobians 16–40 s at `jacobian_batch_size=1`, 15–16 s at 16 and
`"auto"` (which resolves to 16), 28 s at 8; the batched Jacobians differ from
the serial one by 0.5–2 % relative, which a correct implementation cannot do.
That difference must be reproduced in isolation before C1 changes the default.
**Disproved 2026-09-19** (P-OPT 2.4): the isolation was run and there is no
batch dependence. At one solved and refined QI state the tangents at probe and
response chunk widths (1,1), (13,1), (13,24), (150,24), (150,1) and (150,150)
agree with (1,1) to **6.1e-11 … 8.9e-11 relative** with 24 of 24 columns
certified at every width; two problems built identically at width 1 give
bit-identical Jacobians; a third built at `None` agrees end to end to 4.1e-12;
recomputing is bit-identical. A separate 21 % difference that P-OPT's own
multi-section script reported also failed to reproduce, and its four candidate
causes (width, problem rebuild, the return-to-a-repeated-`x` path, a preceding
loose solve left in the caches) were each excluded at 4.6e-12 … 2.2e-10. Warm
cost at that point is **3.0–3.9 s at width 1, 0.83 s at `"auto"` (13) and
0.56 s at `None`**, so the shipped default costs 4–7× the widest setting for a
Jacobian that is the same to 4e-12. **C1's blocker is therefore dissolved.**
The same run establishes that the objective *and* the Jacobian at a repeated
`x` are functions of `x` at 0.9.1 — cost, residual, Jacobian and anchor are
bit-identical after an excursion and reproduce to 7e-12 / 2.2e-10 from a cold
re-solve — so B1c closed the path dependence of the 2026-09-13 review for both.

**Shipped single-stage example, smoke mode** (`VMEX_EXAMPLES_CI=1`, ESSOS #58):
224 s wall, 3.9 GB peak, three trials and one BFGS iteration; ends at
min |ι| 0.071 against its 0.42 floor, aspect 10.2 against 4, B·n RMS 3.1 %
against 1 %. `vmecpp.autodiff.DifferentiableVmec` in the 0.7.4 wheel raises
"no exact residual transpose, rebuild with Enzyme": pip users of VMEC++ have
no derivatives, and its backward pass re-solves the equilibrium.

Verified in code: refinement runs on every value-only trial
(`optimize.py:3122` → `implicit.py:1635`, `refine=True`); `jacobian_batch_size`
defaults to 1, so the block assembly (about 4,650 VJPs at ns = 31) and the
per-dof GMRES corrector are serial `lax.map` loops; `minimize()` with
`objective_terms` sets `jac_solver="reverse"` and runs one adjoint per
*residual row* (`optimize.py:2316, 3060`), contrary to its docstring; the jit key
holds `x0.tobytes()` (`optimize.py:2628–2644`), so every `max_mode` stage
recompiles; the implicit callback runs `mode="cli"` with a host sync every ten
iterations; `_newton_step` (`solver.py:1088`) exists but is reachable only with
`prec2d` configured, which no optimizer path sets. The #299 record shows the
same 847-component design vector returning costs that differ by 4 %, and ±1e-4
probes drifting 1.5 %: the objective is a function of the solve history.

**Four defects in the optimization loops** (P-OPT 4, each with a reproducer,
none of them a wrong published number):

1. **Refinement runs before the acceptance status is known.**
   `implicit.py:1813` calls `_refine_fixed_point` before the status decided at
   `implicit.py:1866–1869`, so a trial that will be rejected as budget-exhausted
   still pays the full Newton anchor: measured **37.1 s and 6,041 Krylov
   iterations after a 2.1 s descent**, and its rows were then replaced by the
   penalty. It does not fire on the shipped examples (no status-2 trials there)
   and is unbounded on a hard deck.
2. **Status-2 trials are not counted as failures** (`optimize.py:3204–3229`
   increments `failed_trials` only on an exception or `_LAST_STATUS_ERROR`), so
   "0 failed trials" in a committed record does not exclude budget-exhausted
   trials.
3. **Two status-free lanes hand out an unconverged equilibrium** — the
   fixed-boundary analogue of the free-boundary case. `metadata["jax_state_runtime"]`
   (`optimize.py:3588–3591`, behind `exterior_field`, `interior_field` and
   `surface_field_values`) calls `solve_implicit`, which carries no status, and
   `equilibrium_from_x` (`optimize.py:3447–3470`) checks only the params key.
   At FSQ/FTOL = 1.2e7 against `max_fsq_ratio = 1e2` both returned a state
   silently (aspect 11.65). The optimizer's own lanes are safe.
4. **`equilibrium_from_x` returns the pre-refinement state** while its docstring
   says it returns "the exact accepted state already used by the objective"; it
   differed from the anchor the objective and gradient were evaluated at by
   **1.3e-3 max-abs** at the QA seed.

There is **no leak and no per-trial recompilation in the fixed-boundary loops**:
over 14 perturbed trials the least-squares lane holds 171 → 175 live arrays then
flat, 1.28 MiB live, RSS flat at 1.88 GiB, zero new XLA compilations after the
first trial, and the scalar lane the same. The free-boundary Schur adjoint's
per-trial closure leak has no fixed-boundary counterpart, because
`_refine_step_core`, `_adjoint_block_core` and `_preconditioned_residual_lane`
are module-level `jax.jit` with `cfg` static and every per-call array a traced
argument — which is the property the free-boundary path lacks.

**Exterior field** (vacuum deck, `ctor = −4e-11`, so the exact plasma field
outside is zero; the direct path equals a full-torus trapezoid rule of the
surface data, reproduced independently to three digits). Median | maximum
relative error versus distance `d` in minor radii and per-period source grid N:

| N | d = a | 0.5 a | 0.2 a | 0.1 a | 0.05 a | 0.02 a |
|---|---|---|---|---|---|---|
| 32 (default) | 2e-5 | 5.8e-3 \| 1.5e-2 | 0.12 \| 0.25 | 0.28 \| 0.91 | 0.43 \| 2.4 | 0.53 \| 4.0 |
| 64 (after the one scheduled doubling) | 2e-6 | 3.2e-5 | 1.8e-2 | 0.13 \| 0.20 | 0.31 \| 0.66 | 0.48 \| 1.2 |
| 128 | — | 3e-6 | 2.9e-4 | 1.8e-2 | 0.13 \| 0.27 | 0.34 \| 1.8 |
| 256 | — | — | 7e-6 | 3.2e-4 | 1.9e-2 | 0.19 \| 0.34 |

The error is `exp(−2π d/h)` with `h` the full-torus toroidal spacing: 1e-4
needs `d ≳ 1.7 h`. The jitted schedule returns its last level silently when the self-test fails (`virtual_casing_jax/integrals.py:754–777`), and it keeps a coarser level whenever that double-layer self-test passes, which in #312's calibration happened for errors as large as 0.30 at a 1e-4 tolerance, so VMEX's returned field can be worse than this table (the worst target at d = a is 3.1e-3); the shipped
example evaluates 0.03 m outside a QA surface on a 12×12 grid. The on-surface
partition-of-unity path is fine (2e-4 median, 1e-3 max), so the surface current
is not the problem. The direct path carries a 0.42 s fixed latency per call
and 0.6 ms per target only at batch 1,000. The near-surface Taylor plan
(bilinear table of the on-surface B and ∇B, first-order Taylor) costs 56–96 s
to build at 32×32 (247 s at 64×64) and 0.04 ms per target; on the finite-β
QA deck against a 256×256 reference it is 0.1–0.2 % of B at 0.1–0.2 a where
the direct default is 12–31 %, and 0.7–2 % at 0.5 a where the direct path is
better. No test in the repository compares the exterior field with an oracle
or places a target within a grid spacing.

**The exterior field against a converged reference** (L-EXT; a target-graded
trapezoid reference, self-consistent between 768×384 at grading 0.985 and
1024×512 at 0.97 to 6e-13 down to `d = 0.01 a` and 7.5e-10 at 0.005 a, which is
the first converged finite-β exterior reference this project has had — earlier
reviews were reference-limited below 0.05 a). The shipped path agrees with an
independent evaluator to 4.4e-9 at `d = a`, the formula, signs, outward normal,
`nfp` replication, the on-surface ½ jump and µ0 are all verified, and the
Ampère loop closes to 7.8e-4. At the default grid the quadrature error against
that reference is 1.5e-10 median at `d = a`, 1.5e-4 at 0.2 a, 1.2e-2 at 0.1 a
and 0.33 at 0.02 a; #312's estimate tracks it within 0.6–1.2× at most targets,
with one 28× under-estimate. Four findings change Phase E:

- **The converged exterior plasma field is not curl-free, and the quadrature
  cannot see it.** With quadrature error ≤ 1e-12, |curl B| / |grad B| is 5e-3 at
  `d = a`, 2.8e-2 at 0.5 a, 9e-2 at 0.2 a and 0.14–0.20 below 0.1 a, while
  div B is 1e-14. Virtual casing of tangential data is the field of the sheet
  current K = n × B, whose curl vanishes only if `div_s K = 0`, i.e.
  `∂_θ B_φ − ∂_φ B_θ = µ0 √g J^s = 0` on the LCFS; VMEC's data satisfy that only
  to discretisation. Projecting the covariant pair onto surface gradients (a
  current potential, as NESCOIL and REGCOIL build it) takes the curl to
  **≤ 6e-10** and moves B_plasma by 0.07–1.2 % and grad B_plasma by 0.8–14 % —
  that difference is the size of the source-data error the shipped field carries
  today, invisible to any quadrature estimate.
- **E3 as written is withdrawn.** A quadratic three-point edge extrapolation
  `(15, −10, 3)/8` in place of `1.5 / −0.5` does *not* remove the curl
  (5.0e-3 against 5.2e-3 at `d = a`): the non-conservation is VMEC's discrete
  `J^s`, not the extrapolation order.
- **The Phase E gate of §1 is unreachable from ns = 50 data.** The vacuum
  identity (exact answer zero, quadrature converged) sits at 2.4e-6 … 1.3e-5
  raw, 2.3e-6 … 1.9e-5 projected and 2.0e-6 … 1.7e-5 with the quadratic
  extrapolation: the floor near the surface is ~1e-5 and neither change moves
  it, because it is the gradient part of the VMEC edge-field error. Gates are
  restated in Phase E against a converged quadrature of the *same* data, plus a
  separate source-data gate.
- **The level was chosen for B and the same level was then differentiated.**
  At the shipped default the third derivative was 170 % wrong at 0.5 a and 1.2 %
  at `d = a` where the finest level gives 8e-9, each derivative multiplying the
  trapezoid error by about `n`. **Closed 2026-09-19** by virtual_casing_jax #7:
  VMEX's third derivative at `d = a` went 1.2e-2 → 8.5e-9, a factor 1.4e6. What
  remains is order-aware grid *sizing*, which is #9's job.

Three methods were measured and ranked. **Closed-form derivative kernels**
(derivative tensors of 1/r contracted with K before the source sum) give B and
all three derivatives in one pass at 1.17 ms/point against 23.6 ms/point for
four nested-`jacfwd` calls through the shipped schedule — **20×**, first call
0.45 s against 3.1 s, identical to AD at 2.7e-15. The **KST a-priori estimate**
(af Klinteberg, Sorgentone, Tornberg, [arXiv:2012.06870](https://arxiv.org/abs/2012.06870))
is conservative by 1.5–3.6× and never below 1× wherever the rule is resolved,
across eleven orders of magnitude and all four derivative orders with no
calibration, at the cost of two complex Newton solves on the boundary series per
target; it replaces the double-layer self-test and the two-level difference, and
it inverts for the grid. A **target-graded trapezoid rule** (substitute
`θ = θ* + u − a_p sin u`, `φ = φ* + v − a_t sin v` and apply the same periodic
rule in (u, v), with `1 − a = d/4h`) reaches **≤ 4e-9 in B and ≤ 3e-6 in grad B
for every `d ∈ [0.005 a, 0.5 a]` with today's finest default node count**
(256×128), where the uniform rule gives 1.4e-6, 2.5e-2, 4, 34 and 54 at the same
distances; 192×96 nodes still give ≤ 1e-6. Its unoptimised prototype costs
40–90 ms per target, dominated by a non-separable boundary-series evaluation, so
it is a point-query and table-filling method and never an ODE right-hand side.

**Equivalent sources (E1) are killed by their own rule** (L-EXT 4). On the
shipped, unprojected data the fit cannot converge at all (residual 2e-3 … 1e-2
at every depth and mode count, floor set by the curl). On projected data, with
24 modes and 82k quadrature nodes at `s_src = 0.9`, the inward continuation
still gives 3.2e-4 at 0.2 a, 2.4e-3 at 0.1 a and 9.8e-2 at 0.02 a against E1's
gate of 1e-4 at `d = 0.02 a` with at most four times the surface grid's source
count. It is also not cheaper per target than the direct sum it would replace
(0.55 ms against 0.30 ms at 256×128), because the source surface needs its own
fine quadrature.

**The interior branch is wrong in its second and third derivatives** (L-EXT 3).
Against the exact oracle `wout_purely_toroidal_field.nc` at ns = 101, B is
1.3e-5 … 1.2e-6 and grad B 5e-5 … 3e-4, but **grad-grad B is 2–17 % wrong and
the third derivative 10–46 %**, because `_radial_value_and_derivative`
(`extender.py:401–438`) interpolates every coefficient linearly in `s`, so AD
returns a zero second radial derivative inside each cell. On a finite-β
axisymmetric deck |div B| / |grad B| is 1–3 % at ns = 11, 21, 41 and 81 with no
convergence.

**The plan's "the Taylor plan is broken" claim is corrected** (L-EXT B5): it
rests on a measurement taken on the vacuum QA wout, where the exact plasma field
is zero, so the ~1e-5 it returned was its error and the direct value it was
compared against (4e-6 T at 4h) was the source-data floor, not a certified
field. The finite-β record is 1.2e-3 … 1.7e-3 of B at 0.1–0.2 a: **approximate,
not broken**. It is retired on its merits — first order, not solenoidal, 56–416 s
to build — and not on that diagnosis.

Three smaller defects: `VmecProblem.exterior_field` (`problem.py:896–939`) and
the `equilibrium_from_x` factory (`optimize.py:3483–3515`) hard-code
`nphi = ntheta = 32` and drop `accuracy_check`, bypassing the geometry rule of
`from_wout`/`from_state`; `_has_plasma_sources` (`extender.py:822–832`) uses
absolute 1e-14 thresholds on dimensional quantities, so the vacuum QA wout
(ctor 1.5e-10 A) reports plasma sources and `plasma="auto"` runs virtual casing
on a vacuum equilibrium; and #374's per-order derivative cache keys on the
config while `B_fn` reads mutable attributes at trace time, so mutating
`external_field`, `plasma_field`, `near_surface_plan` or `newton_iterations`
after the first derivative call silently keeps the old graph.

**Free boundary.** In the host GCROT lane every transpose matvec re-runs the
primal, including NESTOR's full assembly and LU, because `_transpose_matvec`
builds its VJP inside the jitted matvec (`freeboundary_implicit.py:853–860`,
`freeboundary.py:913–927`); the traced and Schur lanes hold one VJP closure. One
gradient costs one forward solve plus `nedge` (≈100) coupled pullbacks or up to
300 GCROT matvecs; the NCSX certificate is 280 s cold and its accuracy floor is
2e-3 from root non-reproducibility. #299's analytic-term contraction removed a
20 GiB compile, not this cost. The free-boundary single-stage example solves in
59 s at FTOL 1e-9 with no predictor and an un-jitted objective.

**Free-boundary determinism, and what it costs** (FB-B, #383). Fixing the
reference seed removes a **57 % objective spread** — five calls at one point are
now bit-identical — and takes a warm trial from 21–26 s to **2.0 s**. It is a
trade, not a free win. On main every finite-difference leg restarts from the
previous call, which correlates the legs: finite differences *look* clean while
the values themselves are not reproducible. With a fixed reference, nearby
parameters can take different paths, which leaves **1–4 % roughness** — that is
measured and **not resolved**. A warm restart from the fixed reference also
stalls for large parameter moves: 2 % on every coil current burns 2,500
iterations and lands 9.2e-3 away from a cold answer that converges in 76, while
1e-3 on one dof reconverges in 106. Under `max_fsq_ratio = 1` such a step
becomes a failed trial, which is the bound on how far a line search may move
before the reference has to be rebuilt.

**NESTOR, measured** (L-FB, a manufactured exterior Neumann problem with
interior charges on an nfp = 3 rotating ellipse, so the exact potential and a
zero `bsqvac` are known). **NESTOR is second-order accurate**, and the cause is
not the singular point: the tan-periodised singular function behaves
like |δ − π| / (4√a) at the antipodal lines `du = π`, `dv = π`, a C⁰ kink whose
derivative jump makes the trapezoid rule O(h²) by Euler–Maclaurin, while the
singular point itself would give O(h³) because its degree-0 remainder is odd and
cancels on the symmetric grid. A 20-line Euler–Maclaurin line correction in a
patched copy raises the source term to order **2.9–3.0** and cuts its error 6.4×
at N = 32 and 11× at N = 64. The same `greenf`/`precal` construction is in
vmec2000, PARVMEC, VMEC++ and DESC's `nestor.py`, so every NESTOR in use has it
and no source states NESTOR's order. Malhotra et al. (PPCF 62 (2020) 024004,
[arXiv:1909.07417](https://arxiv.org/abs/1909.07417)) name NESTOR, state that
Merkel's subtraction is low order and measure O(h²) for it.

At VMEC-paired grids (`ntheta = 2 mpol + 6`, `nzeta = 2 ntor + 4`) with smooth
data, so that Galerkin truncation is negligible:

| mpol = ntor | grid/period | potential, rel. max | truncation floor | grad Φ residual rms / max |
|---|---|---|---|---|
| 6 | 18×16 | 1.2e-2 | 2.7e-3 | 3.9e-2 / 7.8e-2 |
| 8 | 22×20 | 8.3e-3 | 6.0e-4 | 2.6e-2 / 5.7e-2 |
| 10 | 26×24 | 6.3e-3 | 2.0e-4 | 1.9e-2 / 5.0e-2 |
| 12 | 30×28 | 4.4e-3 | 5.4e-5 | 1.5e-2 / 3.6e-2 |
| 12, 2× grid | 60×56 | 9.4e-4 | 6.2e-5 | 2.9e-3 / 8.6e-3 |

At mpol = ntor = 12 the quadrature error is **80× the Galerkin truncation
floor**: the unknowns are wasted, and the surface-field response is wrong by
1.5 % rms. Doubling only the vacuum grid divides the potential error by 4.7 and
the grad Φ error by 4–5.

**The cost split disproves what F0 expected to find.** Warm, VMEC-paired grid:
the kernel scan is **50–55 %** of a NESTOR call and the **LU is 1.5–2 %**, so
VMEC++'s benchmark-header claim that build and factorisation dominate is not
true of this port, and **F1a's `custom_linear_solve` is not a speed item** —
`jnp.linalg.solve` against `lu_factor`/`lu_solve` makes no measurable
difference. The dense forward response is affordable (0.02 s at mpol 3, 1.40 s
at 8, 9.0 s at 12) but it is **not low rank**: at mpol = 8 the 240×256 response
has numerical rank 222, with singular values still at 1e-3 by index 151, which
rules out a randomized low-rank shortcut.

**What breaks first is precision, not cost.** The contraction `Σ_l cmns[l,m,n]
T_l` (`vacuum.py:791–801`) is an alternating sum whose cancellation factor grows
about 30× per +2 in mpol: 2.7e3–6.6e3 at mpol 6, **7.3e4–8.0e4 at 8**, 2.1e6 at
10, **1.6e8–1.8e8 at 12**. The stability-selected `T_l` recurrence itself is
correct to 7.4e-12 against 50-digit quadrature. Differentiation amplifies the
cancellation: the JVP-against-VJP dot-product identity of the whole map is
3e-13 (mpol 3), 2e-11 (6), **3.5e-8 (8)**, 9e-7 (10), 4.5e-5 (12) and 2.9e-4
(mpol 12, nfp 5), while the non-singular kernel alone stays at 1e-13…1e-15. The
acceptance in the code is `10 · adjoint_tol · |rhs|` = 1e-9 relative
(`implicit.py:2126–2132`). The prediction that followed — that a Schur system
built from forward columns and certified against the reverse-mode operator
would miss that acceptance somewhere between mpol 7 and 8 and fall into the
GCROT fallback (`freeboundary_implicit.py:767–775`), silently restoring today's
cost — **was tested on 2026-09-19 and did not reproduce.** FB-A raised the NCSX
deck to mpol = ntor = 8 (ns = 15, 11,520 dof, five perturbed trials): **zero
certificate fallbacks** in both the `edge_response` and `coupled_gcrot` lanes,
and the response-linearized JVP still matches the exact coupled one to
**1.41e-13 relative** (1.5e-11 on the value). The cancellation is real in the
neighbouring quantity — the response's own JVP against a central difference of
the vacuum pressure degrades from 1.1e-9 at mpol 3 to **2.0e-6 at mpol 8** —
but it does not propagate, and the mechanism is the design: nothing in that
lane compares a forward-mode kernel derivative against a reverse-mode one. The
response is a dense matrix, JAX transposes it exactly, and the only
forward-versus-reverse comparison left is the certificate against the exact
coupled transpose, which passes. **So an mpol-aware acceptance is not needed
for that lane** (F4 below is narrowed accordingly). What does degrade with
resolution is the payoff, not the correctness: the response-lane speedup falls
to **2.16× at mpol 8** from 4–5× at mpol 3–7, because the response build is a
larger share of the gradient.

**The real NCSX edge Schur matrix** (`boundary_schur` lane on main, ns = 15, at
the deck's own resolution mpol = 7, ntor = 6 — not 3): nedge = 247, of which 84
columns of S − I are exactly zero; |λ| ∈ [0.11, 8.98] with **35 eigenvalues
further than 1 from unity and 88 further than 0.3**; cond(S) = 8.6e5 although
the eigenvalues span only 80×, so the matrix is strongly non-normal. S − I is
*not* low rank to solver accuracy (124 of 163 active singular values exceed
1e-3), but the clustering makes **unrestarted GMRES converge superlinearly:
≤ 2e-10 in 50 matvecs for S and Sᵀ**, against 247 probe columns today and 671
coupled GCROT matvecs — after a 30-iteration plateau that a restart length of 30
would never leave, so the restart must exceed ~60. **F2 as written is not
supported**: an inexact dense preconditioner needs ≤ 1e-4 operator error to pay
(8 iterations); at 1e-2 it gives 44 against 50 for none, and at 3e-2…1e-1 it is
2.5–4× *worse* (123–189 iterations). Carrying a recycled deflation subspace
across optimizer iterates (GCRO-DR, which SOLVAX's `gcrot` already implements)
is the replacement, and is untested.

**F3's frozen derivative is killed by measurement.** The dropped kernel-shape
term is `|A⁻¹ dA·pot| / |A⁻¹ db|` = **0.10–0.18** over four random low-mode
boundary perturbations, independent of grid and of quadrature oversampling; the
option's gate was 1e-2. Separately, the domain derivative was derived in
Hadamard form (Φ′ solves the same exterior problem with data
`N·∇Φ′ = ∂_u(ξ_N B^u) + ∂_φ(ξ_N B^φ)`) and agrees with AD of NESTOR to 1.8–3.6 %
at mpol 6 and 0.7–1.9 % at mpol 8 — the first known-answer check of the response
that is not a finite difference of the same code. It is 1–3 % inexact, so it is
a **test**, never the operator and (by the preconditioner numbers above) not a
preconditioner either.

**A different quadrature removes the cause.** A 170-line second-kind Nyström
prototype with the O(h³) diagonal zeta correction of Wu & Martinsson
([arXiv:2007.02512](https://arxiv.org/abs/2007.02512)) is **2–5× more accurate
than NESTOR at equal grid** (3.7e-3 against 8.3e-3 at 22×20; 1.4e-3 against
4.4e-3 at 30×28), at observed order 2.85, with no `T_l` recurrences, no `cmns`
cancellation, no tan tables and condition number 2.88 at every size. Their
O(h⁵) stencil is the production target. Partition-of-unity and BIEST quadrature
are **not** better at VMEX's grids: Malhotra's own W7-X table needs ~84 points
per period for 3e-4, i.e. a 3–5× geometry upsample per direction, and its
near-surface off-surface evaluation is stated as unsolved.

**The field, checked 2026-09-13.** VMEC++ 0.7.x ships an implicit adjoint whose
GMRES solve costs 40–50 s against a 2 s forward solve at ns = 25 (their PR 855),
an Enzyme build absent from the wheels, and hot restart limited to a single
radial grid. DESC's omnigenity objective (Dudt et al. 2024,
[arXiv:2305.08026](https://arxiv.org/abs/2305.08026)) keeps one Boozer
projection per surface but removes field-line tracing, wells and every
`argmin`; it co-optimizes a parametrized `|B|(ρ,η)` and runs under thirty
minutes on one A100; DESC's proximal projection warm-starts from the last
*accepted* equilibrium with rollback on rejected trust-region steps, so it is
deterministic within a run but path dependent across runs. VMEC2000's own
`PRECON_TYPE='GMRES'` mode is a finite-difference Jacobian built once at
switch-on plus GMRES(20) at 1e-3 with finite-difference matvecs; it never
certifies a root. Off-surface virtual casing within one grid spacing is an
open problem in every fusion code (Malhotra et al. 2020 is on-surface only and
says so; EXTENDER uses adaptive Newton–Cotes; DESC evaluates on-surface only,
with a volume Biot–Savart branch unmerged since 2024); the periodic-trapezoid
error `(1/h)^{1/2} e^{−2πd/h}` means no affordable grid fixes it, so the method
must change — equivalent sources (quadrature by fundamental solutions) or
hedgehog extrapolation are the two proven routes. The only
independent large-sample cost number is ConStellaration's three minutes per
DESC optimization run against about one hour per VMEC++ run driven by finite
differences ([arXiv:2506.19583](https://arxiv.org/abs/2506.19583)); no 2025–26
QI or single-stage paper publishes wall times except Dudt's.

**The repository is already small; the 969 MB was never what a user downloads**
(SLIM-A). A plain `git clone` of `uwplasma/vmex` is **40 MiB** (37 single-branch,
8.4 with `--filter=blob:none`, 5.6 at `--depth 1`); the 969 MB figure describes
the maintainer's long-lived personal checkout, and a `--mirror` clone is 834 MiB
of which ~94 % is `refs/pull/*`. The largest blob reachable from any branch or
tag today is 4.5 MB (`examples/data/mgrid_ncsx_c09r00_small.nc`). Every large
blob the brief named lives only in PR refs (PR #1's two mgrid files, the QS
sweep panels of PRs #8–#17) except two `gkx.*.nc` files, which exist **nowhere
on GitHub** and are reachable only from a local branch or worktree. Deleting
every safely-deletable branch moves the mirror by −14 MiB (−1.7 %); a history
rewrite of `main` could reclaim **~22.65 MiB** of an already-33.7 MiB pack,
almost all of it superseded figure generations, while invalidating every fork,
open PR branch, 36 tags and the PyPI/DOI provenance they carry — **not
recommended**. Of 102 non-`main` branches, **74 are delete-safe** (68 merged PRs,
6 closed-unmerged whose heads survive at `refs/pull/N/head`), 15 are keep (5 open
PRs, the six winding-surface branches, the two HINT-comparison branches,
`wip/single-stage-split`, two release branches) and 13 need a maintainer answer
(no PR, unique commits — three of them external-contributor branches). Sixty
registered local worktrees hold 6.55 GiB, **3.45 GiB of it stale** by git's own
`prunable` flag or by checking out a delete-safe branch: that is the largest
low-risk footprint win and it is on the maintainer's disk, not a cloner's. In
the tracked tree, `tools/` has zero orphans under a four-pass search (literal
path, basename, date-stripped stem, and bare module name — the pass that would
have caught the `build_qi_sheet_mgrid.py` incident), `benchmarks/` has none
beyond the three the repository's own `INDEX.md` already declares uncited, and
`tests/data` duplicates nothing in `examples/data` by name or by hash. The
confirmed orphan list is **10 files, 262 KiB** (nine `history.json` files under
`docs/_static/readme_best_cases/` and `qi_readme_cases/`, plus one derived
`.panels.h5`), with four input decks and five more flagged as needing an answer
rather than a grep.

**Coverage, tests and source** (SLIM-B, from the coverage artifacts of a green
PR run on `b6f2a750`). PR-lane coverage is **94.31 %** (22,880 / 24,260
statements); **49 of 73 measured modules are at or above 95 % and 24 are
below**, and 395 of the 564 statements needed to lift every module to 95 % are in
three free-boundary modules — `core/freeboundary_implicit.py` at **43.2 %**,
`mirror/free_boundary.py` at 65.9 %, `core/freeboundary.py` at 87.5 %.
Seventy-seven functions have zero PR coverage (461 statements), led by
`_host_boundary_schur_adjoint` (115); each is exercised only by `full`-marked
nightly or weekly tests, so the changed-lines gate cannot see a regression in
them. The coverage `omit` is a basename glob, so it hides **three** modules and
not the two it names: `vmex/mirror/turbulence.py` (107 statements, measured 94 %
locally) is invisible to both the changed-lines gate and the 90 % mirror floor,
and `core/virtual_casing.py` measures 44 %, its gap being the whole
wout/state surface-data route. **`tests/conftest.py` sets
`jax_disable_jit=True` for the entire suite**: 46 modules (661 tests) never
enable it, and **11 tests whose names claim they check jit-compatibility (29
nodes) ran with jit disabled**, where `jax.jit` is inert — all 29 pass when jit
is forced on (47.7 s against 68.7 s eager), so none hides a defect today, but
each would have stayed green through one. On CI shape: **1,979 unique PR nodes
are executed 3,055 times**, because five "representative physics" jobs (1,835 s)
re-run modules that a parity lane already runs in full, and `--dist load` rebuilds
`scope="module"` fixtures on every xdist worker — about **2,500 s per PR** that
`--dist loadfile` removes with one flag. The package has almost no dead code
(4 symbols, 23 lines), but 37 symbols and 586 lines are production code kept
alive only by tests, and the square-root homotopy lane is unreachable from every
production root: retiring it is **−1,620 source and −990 test lines** and ≥ 1,560 s
of one PR lane. Docstrings are **22 %** of the package's 64,850 lines.

**The examples** (EX, all 67 files under `examples/` read against one proposed
template of twelve rules with eight family variants; six exemplars converted and
verified byte-for-byte on stdout and artefacts). The correctness findings are
separable from the formatting, and they are what matters here: the coverage
guard in `tests/test_examples.py` matched **basenames**, so the eight
`stellarator_asymmetry/*` examples were considered covered by their symmetric
namesakes and **never ran**; two of them declare a reduced radial grid under
`if ci_smoke:` and overwrite it unconditionally three statements later, so the
smoke budget was inert — `QH_optimization.py` exited 1 in 42 s with the 1e6
sentinel and zero iterations, and `QA_optimization.py` was solving the full grid
in the smoke lane. Two examples declare `IOTA_FLOOR`/`MIRROR_LIMIT`/`ELONGATION_LIMIT`
and then hard-code different numbers in their callbacks, so editing the
documented constant changes nothing; one carries a commented objective row whose
constant is defined nowhere (`NameError` if uncommented); two write the same
three output file names, so running both in one directory silently overwrites;
and six state something their code or deck contradicts, including two that call
a deck with `betatotal = 4.262e-02` "zero pressure". Forty-one of 66 examples
pass a physics- or accuracy-affecting numeric literal to a library call below
the parameter block, the pinned `nphi = ntheta = 32` virtual-casing grids among
them.

## 3. Root causes

| complaint | cause, verified in code or record |
|---|---|
| slow QI | two nonlinear solves per trial (descent to 1e-12, then a refinement that on the seed deck exhausts 6,000 GCROT iterations and returns the state unchanged, because the raw Jacobian's edge λ modes put its condition number near 6e12); serial Jacobian with a per-dof GMRES corrector; 45–190 s recompile per `max_mode` stage; a non-smooth surrogate residual (`argmin`, `cummax`, `interp`) with 17,712 rows that fails trials; a circular-torus seed where every published QI result used a near-axis one |
| slow single-stage | path-dependent objective, so BFGS line searches fail; refinement and the full Jacobian on every trial; penalty BFGS instead of least squares; a seed at ι ≈ 0.08 against a 0.42 floor, with B·n weighted 70× the ι term |
| slow free-boundary single-stage | 59 s solves with no predictor; NESTOR inside every adjoint matvec; un-jitted objective |
| exterior field | trapezoid rule off-surface with silent non-convergence; O(N_src) per target with 0.42 s latency; no oracle test. Added 2026-09-19 (L-EXT): the LCFS source data do not conserve current, so the converged field is not curl-free (5e-3 at `d = a`, 0.2 near the LCFS) whatever the quadrature; the schedule chose its level for B and then differentiated that level; and the interior branch interpolates every coefficient linearly in `s`, so its second radial derivative is identically zero |
| "VMEC++ is faster" | cold compile and the 2.6× per-iteration constant; the loop itself is at parity and VMEX's derivative is cheaper than VMEC++'s adjoint. Split 2026-09-19 (L-SOLVE): the cold process is 60–66 % XLA compile, 40–50 % of those seconds spent on ~130 single-op eager programs per rung that no solver lane needs; the per-iteration constant is 844 fusions and 570 serial Thomas trips, not missing threads — VMEX already saturates four cores |
| slow optimization runs that are not the solver | 49 % of the QA run is per-stage recompilation at identical array shapes, because the jit key holds `x0.tobytes()`; 60–78 % of every trial descent runs below FSQ 1e-9, where the block Newton finish crosses in three GMRES steps; a gradient at a new point builds two full block factorisations; the shipped `jacobian_batch_size=1` costs 4–7× the widest setting (P-OPT 1, 2.1–2.4) |

## 4. Programme

> **Superseded 2026-09-21.** The phase tables are replaced by Part I lanes A–F as the work queue. Their measurements, gates and kill rules remain evidence; "Kill rules and things not to repeat" stays in force unless Part I records new evidence against a rule.

Principles: one nonlinear solve per trial; the objective is a function of `x`;
a derivative is sized to its number of outputs; every accuracy claim has a
known-answer oracle; every performance claim is a before/after row in
`benchmarks/optimization.py` with the counters of A1. Each item is one PR of at
most one week, opened from a worktree, verified before push, and merged only
when every lane is green. Commits carry the maintainer's authorship and no tool
attribution. One heavy local job at a time; the office box takes one.

### Phase A, week 1: instruments, oracles, honest examples

| PR | files | verification and gate |
|---|---|---|
| A1 counters | `vmex/core/monitoring.py`, `implicit.py`, `optimize.py`, `benchmarks/optimization.py` | every `OptimizationRecord` and benchmark row carries descent iterations, refinement steps and matvecs, Jacobian columns, GMRES iterations per column, adjoint matvecs and compiles; the QI, QA and single-stage rows show the split; no performance PR merges without a before/after row |
| A2 exterior oracles | `tests/test_virtual_casing_physics.py`, `vmex/core/virtual_casing.py`, `extender.py`, `docs/explanation/nestor-vacuum.rst`, `examples/vmex_get_B_outside_plasma.py` | tests for the vacuum identity outside, the interior identity inside and Malhotra on-surface parity; `B_plasma_xyz` returns achieved digits and a failed schedule raises or warns; the `d ≳ 2h` rule and the default grid documented; the example uses a grid that passes its own check |
| A3 honest single stage | `examples/optimization/single_stage_optimization.py`, `single_stage_free_boundary_optimization.py` | seed with mean ι ≥ 0.3; ι and aspect as bounds or constraints (ESSOS augmented Lagrangian or `trust-constr`); QS normalized by ι; the objective jitted; the example meets its targets or fails loudly; a committed profile row exists for both examples |
| A4 documentation truth | `README.md`, `docs/reference/performance.rst`, `docs/reference/objectives.rst`, `docs/explanation/adjoint-gradients.md`, `docs/howto/run-on-gpu.md`, `docs/howto/parameter-scans.md`, `docs/_static/figures/figures.json` | the 14.5-minute QA, 17.3-minute QI and 33× adjoint prose numbers either gain a record or go; the README qualifies the exterior field and GPU optimization; the orphaned extender figure is cited or removed; prose gate and cited-path test pass |
| A5 external-corpus robustness | fetch the 23 decks where a VMEC-family peer strictly converged and VMEX did not (`atf*`, `bean14`, `belt20`, `c82vac20`, `qas14`, `w7s20`, `cooper`, `Q2_KINK`, `HSX_QHS`, `WISTELL-A`, W7-X `d23p4_tm`, `qhs46`, ITER hybrid), run them, and classify each failure as an INDATA feature not parsed, a Jacobian reset, an iteration budget or genuine non-convergence; fix only what the classification names | VMEX's strictly-converged count on the 344-case `itpplasma/benchmark_vmec` corpus reaches vmec2000's 179, from 168; no shipped deck regresses |
| A5 robustness against the external corpus | the 23 named decks, then whatever the classification points at | `itpplasma/benchmark_vmec` runs 344 cases through twelve codes. VMEX is strictly converged on all 168 it reports — third in the VMEC family, ahead of VMEC++ at 157 — but 23 cases converge for a peer and not for VMEX (`atf*`, `bean14`, `belt20`, `c82vac20`, `qas14`, `w7s20`, `cooper`, `Q2_KINK`, `HSX_QHS`, `WISTELL-A`, W7-X `d23p4_tm`, `qhs46`, ITER hybrid). Fetch them, run them, classify each failure (INDATA feature not parsed / Jacobian reset / iteration budget / genuine non-convergence), and fix only what the classification shows. Gate: the strictly-converged count reaches vmec2000's 179, with no case regressing. Independent of B–F; after A1–A4 |
| A6 the test suite runs with jit disabled | `tests/conftest.py`, the 11 tests named "jittable", one lint test | `jax_disable_jit=True` is set for the whole suite, so `jax.jit` is inert and line coverage cannot tell a traced path from an eager one. This is one defect that has now hidden four separate problems: the blocked jitted free-boundary pullback whose test monkeypatched the function away, a jit wrapper that would have broken NumPy callables, two coverage gates that could not see tracer branches, and 11 tests (29 nodes) named "jittable" that never enabled jit (SLIM-B 3.1). Gate: those 29 nodes run with jit on (verified green, 47.7 s against 68.7 s eager) and a lint test fails when a test body contains `jax.jit` without a jit-enabling fixture; then either the suite default inverts, with the ~15 kernel A/B modules opting out, or one nightly lane reruns the PR selection with jit forced on. Kill neither arm on cost before measuring it: the lane is the cheaper of the two and would have caught all three of this week's defects |
| A7 the compile counter must fail loudly | `tests/test_profile_workflows.py`, the counting harness behind it, and the records it writes | the counter reads zero intermittently on main's c1 lane and in the weekly lane, and the harness cannot separate a warm-cache zero from a broken counter (§2). Gate: the regime is asserted explicitly rather than inferred from a positive count, so a warm-cache run asserts zero and a cold run asserts its lane count; the flaky node is deterministic over ten consecutive runs; a record written with a zero count is marked unusable rather than published. Being fixed separately — this row exists so that no compile-count gate of B4, S1, S2, S4 or P1 is read as evidence until it is |

### Phase B, weeks 1–3: one certified solve per trial

| PR | change | evidence | gate |
|---|---|---|---|
| B1 Newton finish | in `solver.py`/`implicit.py`: when `fsq` falls below a switch threshold, take matrix-free Newton–GCROT steps inside the descent (the `_newton_step` lane with exact JVPs and the true-residual check) so the returned state is a certified root; delete the separate refinement pass; value-only trials get the same root; #302's anchor contract folds in here | refinement is 51 % of a gradient call; drift 1.5 % → 6e-8 when refined | objective replay at a repeated `x` agrees to 1e-9 on the QI and single-stage cases; the joint phase exits without precision loss; gradient call ≤ 0.5× today; certificate values unchanged on the P1 matrix |
| B2 warm starts everywhere | perturbation predictor into the free-boundary cache; rung skipping for `initial_state`; hot restart for CLI and `solve_file` sequences; `mode="jit"` inside the callback | 806 → 212 iterations at a 1e-4 move | iterations per accepted trial ≤ 0.3× cold on the QI example |
| B3 exact block adjoint (measured on both decks: adjoint residual ≤ 1.2e-10 in one factorization, 300–5,000× cheaper than production with the Jacobian's factor reused, while B1's dense check puts the production Krylov error at 9e-6 (QA) to 3e-5 (QI); the 1.1e-3 QI gap between lanes is the raw-versus-preconditioned formulation at non-root anchors, which B3 aligns on the raw formulation), then Krylov recycling | on the seed deck the QI adjoint's GCROT stalls at 2.6e-5 and takes 13,228 iterations (78 s), while the raw block factorization is exact at the state, so first solve the adjoint by transposing that factorization together with the 1-D preconditioner (one direct solve; B1 is measuring it); keep GCROT as fallback and certifier; then return the adjoint's Krylov iteration count from compiled programs (A1's counters report `None` for every compiled adjoint, which covers scalar `minimize()` gradients); then warm-start λ across trials, reuse the GCROT deflation space across Newton and adjoint solves, and freeze the preconditioner in the matvec at the root | three solves of one operator per gradient today | adjoint matvecs per gradient ≤ 0.5×, measured by the returned count |
| B4 compile hygiene | `x0` out of the jit key; configs keyed by content; one compile per resolution; persistent cache where jaxlib allows; the JAX value-and-gradient lane must reuse the host lane's executables instead of compiling its own | 102 compiles in a five-evaluation warm campaign; on the A1 rows (#310) the JAX value-and-gradient lane spends 41.6 s (QA) and 71.0 s (QI) compiling after the host derivative has already compiled, and builds take 244–500 XLA compiles  A per-module census of one `input.circular_tokamak` solve (2026-09-16) counts 152 XLA compilations for a 7.6 s cold solve whose warm repeat is 0.16 s: four named lanes (`_block_lane` twice, 2.49 s) and about 130 single eager ops, each compiling its own module — `broadcast_in_dim` 22, `multiply` 19, `copy` 17 — in `setup.py`, `solver.py:2345` (leaf-by-leaf carry copy) and `solver.py:914` (`_zero_cache`). A jitted whole-tree copy plus `device_put` of NumPy zeros takes it to 129. The persistent cache already halves the QA_lowres cold start across processes (20.3 s to 10.5 s) and cuts the single-stage gradient's compile from 49.8 s to 6.0 s, so what is left to attack is the 10 s of Python-side tracing. | cold to first gradient ≤ 20 s CPU on the QI example; zero recompiles across `max_mode` stages; the second lane adds under 5 s of compile |
| B5 per-iteration constant | one bounded experiment (≤ 1 week), in this order: ms per iteration for VMEX and VMEC++ at 1, 2 and 4 threads; an HLO census of the iteration (ops, loops, loop trips; the CPU tridiagonal solve is two `lax.scan` Thomas sweeps, about 100 serial trips per iteration at ns = 50); a dispatch arm with a batched tridiagonal kernel; and only if the thread scaling shows headroom, `shard_map` with radial slabs and the tridiagonal solve split over modes, as VMEC++'s OpenMP does | 2.4 vs 0.9 ms per iteration; JAX's CPU thunk runtime has documented 2.5–14× regressions on many-small-kernel workloads and host devices share one thread pool | warm ns = 50 QA below 1.5 ms per iteration; the dispatch arm keeps iteration counts identical and the final state within 1e-12 relative; kill sharding below 1.25× on four devices or with collectives above 30 % of the iteration. **Revised 2026-09-19 (L-SOLVE 3.1): the sharding arm is dropped, not gated** — the iteration stops scaling at four cores on an 18-core Xeon and burns 63 % more CPU there for the same wall time, so there is no parallelism to recover. The dispatch arm keeps its gate and is now sized by the HLO census: 844 fusions and 570 serial Thomas trips per iteration |

**Cold start and the single solve (L-SOLVE S1–S7).** These sit underneath
Phase B: S1, S2, S4 and S5 attack the 60–66 % of a cold process that is XLA
compile, S3 and S6 the iteration, S7 the fallback adjoint. Each gate names the
measurement that justifies it.

| PR | change | evidence | gate and kill rule |
|---|---|---|---|
| S1 fold the cold path's eager ops into jitted lanes | jit the array-building *tail* of `setup.py::radial_grids`/`flux_profiles`/`interior_guess`/`blend_m0`, `residuals.py::_m1_rotate_threed`, `solver.py:2345`'s leaf-by-leaf carry copy and `solver.py:914`'s `_zero_cache` — not the head, which reads `VmecInput` as host numbers and raises "Error interpreting argument … as an abstract array" | 383 of 413 QA compile requests are single-op programs costing 3.68 s of 9.32 s; one first eager dispatch costs 19.7 ms against 86 ms for sixty of the same ops fused (13.8×); warm execution of the same code is 50 ms, so there is no run-time regression to trade. Expected 2.5–3.2 s off a 14.7 s cold QA run and 1.3–1.5 s off a 5.2 s li383 run (the residual program count is inferred) | parity suite unchanged, iteration counts identical and final `fsqr` bit-identical on the regression decks, compile requests on a cold `input.circular_tokamak` solve below 60. Jit whole existing functions rather than restructuring their arithmetic; kill the PR if any lane changes float association |
| S2 narrow the inter-rung `jax.clear_caches()` | release the lane executables (`_release_used_lane_executables`, where the megabytes are) without dropping every jit and lowering cache; `multigrid.py:583–596` and the free-boundary twin at `:974–986`, which `cli.py:812,842` triggers on every CLI run | controlled in-process A/B, same call, one flag changed: **+118 compile requests, +0.75 s compile, +0.95 s wall** on the cold QA ladder with 656 iterations and `fsqr = 9.747403887641546e-12` identical across all four arms. A single-rung deck shows nothing; with the persistent cache on the cost is 0.2–0.3 s | same iterations and bit-identical `fsqr`; **peak RSS not above today's on the W7-X ladder**, which is what the release exists to bound. **Ordering constraint: this must not land before F-hoist** — see Phase F; the cache clearing is what currently hides the free-boundary Schur lane's per-gradient closure leak, and the two changes must not both land |
| S3 cache the tridiagonal factorisation | store `c'` and `1/denominator` in `PreconditionerCache` beside the bands they are built from; **SOLVAX first** (`tridiagonal_factor` / `tridiagonal_solve_factored` plus a checked variant that validates at factor time), since the block path already has this split and the scalar path does not | micro-benchmark at VMEX's exact shapes inside one jitted `lax.scan` of 200 applications: **2.7× (ns 16), 4.6× (ns 50), 4.4× (ns 201)** per apply, max \|difference\| 0.0e+00; the HLO census shows 570 serial sweep trips per iteration. Expected 8 % of the ns = 50 iteration, ~12 % at ns = 201 (inferred from the micro-benchmark share) | byte-identical trajectories on the regression decks and unchanged parity iteration windows. The factors must be invalidated exactly when the bands are, Jacobian resets included; a missed invalidation does not corrupt the answer but changes iteration counts, so parity is the detector |
| S4 `vmex --warm`: ahead-of-time lane executables | `jax.experimental.serialize_executable` stores the compiled `_block_lane`; a later process deserializes instead of tracing, lowering, hashing and reading the cache. Keyed like a cache entry, under `_compat._cache_machine_fingerprint` and on (version, lane, treedef, avals, static meta) | prototype, three alternations at load 47–57 so only the ratio is claimed: **3.2–4× on the lane start** (solovev 1.36–1.62 s against 0.28–0.42 s; QA ns 50 1.64–2.04 s against 0.50–0.70 s; payloads 2.2 and 4.3 MB). In quiet seconds that is 1.2–1.8 s of the 6.5–7.1 s warm-cache QA process. Shape-polymorphic `jax.export` does **not** help: JAX documents that exported polymorphic functions are still recompiled per concrete shape, so AOT-per-structure is the only form that works | bit-identical solve from a deserialized executable against the jitted lane on every regression deck; a corrupted payload falls back silently, proven by a test; no change when `--warm` was never run. Payloads never ship in wheels |
| S5 compile the whole ladder's lanes up front | the `NS_ARRAY` ladder is known at start and XLA compilation releases the GIL; today `--prefetch-compile` overlaps only rung k+1 with rung k, and the coarse rungs iterate for less time (0.4 s) than a lane compiles (1.4 s) | three alternations of the existing flag on a cold QA run: 0–21 % (29.2/29.1/33.1 s against 36.9/34.0/33.3 s, loaded machine). Launching all three at once turns 4.3 s of serial lane compile into about 1.6 s (inferred) | cold ladder wall improved on a ≥ 8-core host with peak RSS within 1.2×, no change to results; stays opt-in, because it raises peak memory and contends with the solve on small CPU sets |
| S6 opt-in FIRE-style velocity projection | add the `v·F < 0` freeze-and-restart rule to `step.py`'s damped dynamics, as a third mode beside 1-D and 2-D preconditioning | Bitzek et al., Phys. Rev. Lett. 97, 170201 (2006), Table I: 2–5× fewer function calls than CG on four systems with L-BFGS comparable, for an iteration of exactly VMEX's structure. **Not verified for MHD equilibrium** — no source applies FIRE to a VMEC-type problem, so the transfer is inferred and no factor is promised | kill unless it cuts iterations ≥ 1.5× on at least two of the stiff cases of the committed 2-D preconditioner record with `wb` matching to 1e-10; the 1-D default stays byte-identical. It changes the iteration path, so it can never be the parity default |
| S7 pass a recycle space to GCROT on the fallback adjoint | `solvax.gcrot` implements FIFO recycling and `recycle_strategy="harmonic"` (GCRO-DR); every call in `implicit.py` (lines 1507, 2290, 2994) passes `recycle=None`, so `solution.recycle` is discarded and no deflation space crosses solves | read in the source of both packages. The stalling case it targets is this plan's own record: the QI seed adjoint, 13,228 iterations, stalling at 2.6e-5. After B3a/B3b this is the fallback, not the common path, so the honest expectation is no effect on the shipped decks and a possible rescue on the QI seed | fallback-path matvecs per gradient ≤ 0.5× on the QI seed deck, measured by A1's counters; no certified value changes. Cost is two keyword arguments and a carried field, so a null result closes it cheaply |

### Phase C, weeks 2–4: derivatives sized to the problem

| PR | change | gate |
|---|---|---|
| C1 batched Jacobian | `jacobian_batch_size="auto"` with the measured-memory chunk, after reproducing in isolation the 0.5–2 % batch dependence of §2; the per-column GMRES certifier already takes zero iterations on the A1 rows (#310), so the Jacobian's 27–32 s first-call cost is block assembly, factorization and their compile, which is what to reduce. **The isolation ran on 2026-09-19 and found no batch dependence at all** (P-OPT 2.4, §2 above): six probe/response chunk widths agree with (1,1) to 6.1e-11…8.9e-11 with every column certified, a rebuilt problem at `None` agrees end to end to 4.1e-12, and the 21 % difference a review script reported also failed to reproduce with four candidate causes excluded. C1's blocker is dissolved; the default change itself is P4 below and is already in execution | warm 48-dof Jacobian ≤ 0.4 s CPU; columns identical to 1e-10 across batch sizes |
| C2 `minimize()` | route `objective_terms` through the block Jacobian or a scalar adjoint; docstring true | one linear solve per dof or per gradient, never per row |
| C3 joint least squares | single stage as TRF/LM on `[r_plasma; r_coil]` with `[J_plasma; J_coil]` (coil block by `jacfwd`); Jacobian only at accepted points; move #311's augmented-Lagrangian wrapper into one small library helper so examples stay short, and improve quasisymmetry while holding its constraints (#311 met them at 0.113 against 0.101 at its seed) | stage 1 needed 63 nfev / 26 njev where joint BFGS needed 188 / 176 → joint phase ≤ 0.4× today; design meets targets; cold re-evaluation matches. **Note 2026-09-19 (P-OPT 1):** the single-stage budget says a driver change cannot pay for itself — of 541 s of warm trials the plasma program is 535 s and the coil program 5.9 s (14–32 ms per trial, 1 %), so C3 is worth doing for the constraint handling it settles, not for wall time |

**The optimization loop (P-OPT P1–P5).** Measured on the shipped examples run
unmodified; gains are percentages of that example's wall. P1 and P4 are already
in execution.

| PR | change | evidence | gate and kill rule |
|---|---|---|---|
| P1 compile once at the largest `max_mode` and mask the stages | build one problem at the largest `max_mode` and give each stage a boolean dof mask applied to the tangent stack and the dof projector, instead of a new problem per stage; the jit key at `optimize.py:2628–2644` holds `x0.tobytes()` and the dof layout, so each stage is a cache miss by construction although `MINIMUM_MPOL` keeps every array shape identical | 276 s of the 568 s QA run are stages 2 and 3's first-Jacobian calls (113.1 + 82.1 s against a ~5 s warm equivalent). A masked prototype runs the same three stages with compile *inside* the stages of 0.1 / 0.0 / 0.0 s, total trace+MLIR+XLA 41.6 s against 301 s, and reaches **the same cost to 11 significant figures with identical nfev/njev**. Expected 230–276 s off the QA run (40–49 %); nothing on one-stage examples | per-stage cost identical to 1e-10 and nfev/njev unchanged on QA and QH; XLA compile requests after the first stage below 10; the masked components of `x` provably unchanged each stage — a leak there would change the design silently, so assert it rather than inspect it |
| P2 one block factorisation per point, shared by the Newton finish and the Jacobian/adjoint | today a gradient at a new point builds two full raw block systems, one in `_refine_block_factors` at the descent iterate and one in `_raw_block_system` at the refined anchor; use the refinement's factors for the tangent and adjoint solves and let the existing certifier absorb the difference | defect correction driven by the refinement's factors takes the tangent system at the refined anchor from 1.0 to **8.8e-10…1.4e-8 relative in 3–4 passes** on all five QI trial pairs — four to six orders inside `jacobian_adjoint_tol = 1e-4`; one factorisation is 1.4–1.8 s of a 4.8 s warm QI trial-plus-Jacobian. Expected ~1.5 s per accepted point: ~16 s of the QI run's 72 s least-squares phase and ~90 s of the single-stage run's 600 s | Jacobian within 1e-9 relative of today's and adjoint residual within its existing acceptance on the QA and QI seed decks; the counters show one factorisation per accepted point. The certificate must be measured on the raw operator at the **anchor**, not at the factorisation point — `_implicit_evolved_tangent_multi_rhs` already does exactly that check |
| P3 stop trial descents at FTOL 1e-10 and let the block Newton finish close the gap | move the hand-over point (`forward_ftol` for trials) and leave `refine_tol` and the final solve unchanged | on four recorded QI optimizer steps, each seeded by the same perturbation predictor: **1.8–2.9× fewer descent iterations at 1e-10** with the finish certifying on all four and the objective within 8e-9 relative; 2.6–4.7× at 1e-9 within 4.1e-8; at 1e-8 the finish fails on two of four large steps, which is the kill line. Expected: QI descent 33.3 → 12–17 s, single-stage descent 149 → 55–80 s (12–15 % of that run), QA descent 82.9 → 30–45 s | every accepted trial reaches `refine_tol`; objective replay at a repeated `x` within 1e-9; the QI and single-stage designs unchanged within their stated target margins; a per-run counter of finishes that did not certify. `_REFINE_BLOCK_STALL` already returns the unrefined state, so the failure mode is a lost anchor, not a wrong one |
| P4 make `jacobian_batch_size="auto"` the default | the width bounds the `chunk_map` over the block system's 150-row probe basis; at 1 it is a serial `lax.map`, which is why the gain is in the probe assembly and not in the columns | widths agree to 6e-11 at a fixed state and 4.1e-12 end to end (C1 above); warm cost 3.0–3.9 s at width 1 against 0.83 s at `"auto"` and 0.56 s at `None`. Expected: QI Jacobians 18.0 → 4–9 s, QA Jacobians 77.6 → 18–25 s | Jacobian identical to 1e-9 across widths on QA, QI and a LASYM deck; peak RSS within 1.2× of today's on the largest shipped deck; first-call compile not above today's. `"auto"` and not `None`, because it is the memory-aware setting |
| P5 reuse the previous accepted point's factors as the finish's preconditioner | keep the last accepted point's `_RawBlockSystem` for the next trial's GMRES and refactor on a stall | on five consecutive QI trial pairs, for every scaled step ≤ 0.71 the stale factors carry the finish to `refine_tol` and it costs **0.35–0.84 s instead of 1.56–1.96 s (1.9–4.8×)**; at the outlier step 2.85 it fails and must refactor. **Verified negative in the same table:** the same factors used as a *solver* for the tangent system diverge by up to 11 orders (2.7e+11 at that step), so "reuse the Jacobian across nearby trials" is dead and this is a preconditioner-only result. Expected: QI refinement 16.4 → ~6 s, single-stage refinement 194 → ~70 s | iterations to `refine_tol` and the final anchor unchanged within the certificate on QA and QI; a counter reports the refactor rate; no trial is left uncertified that certifies today |

Two cheap items outside the optimizer, from the same budget: three
`EquilibriumReporter` calls cost 12.0 s on the QI run (about 10 s of it
compiling the constructed-QI total a second time, eagerly) and the two plot
calls cost 29.1 s — together 16 % of a 263 s run, spent after the design is
final. Report from the compiled residual, or from the last residual vector.

### Phase D, weeks 3–6: the QI objective

| PR | change | gate |
|---|---|---|
| D0 near-axis seed and ladder | seed the QI example from a pyQIC near-axis QI boundary (the shipped, unused `examples/data/input.QI_stel_seed_3127` or `from_paper("QI NFP2 r2")` truncated to modes ≤ 2), run a `[2, 3]` / `[20, 60]` ladder, and add Goodman's `phimin` sign rule to the residual; every published QI result started from a near-axis seed and none from a circular torus | validation-grid QI before/after; failed trials and nfev per stage recorded; time to the current final metric ≤ 1/2 |
| D1 smooth residual | replace the two `argmin`s (`optimize.py:1039, 1068`), the five `cummax` calls and `interp` in `quasi_isodynamic_residual` by softplus/logsumexp wells with a differentiable well location (the frozen well location has zero derivative today); per-well moments instead of 17,712 point rows | Taylor test passes in both directions; failed-trial rate → 0; same final QI metric within 5 % |
| D2 target-field lane | Dudt-style omnigenity: monotone-spline `|B|(ρ,η)` plus a deformation `h`, residual = Boozer-spectrum evaluation at mapped angles (30–130 smooth rows per surface, bounce points by construction), well parameters co-optimized; scalar adjoint viable; DESC's tutorial does 100 iterations in two CPU minutes | QI example reaches today's final metric in ≤ 1/3 the time; ε_eff and the Boozer spectrum of the result reported |
| D3 Boozer cost | `oversample=1` validated by the existing fine-grid check; volume physics hoisted out of the per-surface loop; `magnetic_only` when upstream booz_xform_jax #8 merges | Boozer share of residual + JVP ≤ 1/3 |

### Phase E, weeks 3–6: the exterior field

Revised 2026-09-19 from L-EXT. Two gates replace the single gate of §1, because
the measurements separate two errors that the old gate conflated:
the **quadrature** error of the rule, and the **source-data** error of the LCFS
field VMEC hands it. The vacuum identity at ns = 50 floors at ~1e-5 near the
surface and neither a better extrapolation nor the projection moves it, so
"vacuum identity ≤ 1e-6 down to 0.005 a" is unreachable from these data by any
quadrature or fit and is retired as a gate:

- **Quadrature gate.** Against a converged quadrature of the *same* source data
  (the graded reference of L-EXT 0, self-consistent to 6e-13 down to 0.01 a),
  the shipped path is within 1e-6 in B for every `d ≥ 0.005 a` at the default
  node count, and every derivative order the API exposes carries an estimate
  that is never below 1× and never above 5× the true error where the rule is
  resolved.
- **Source-data gate.** On the finite-β QA deck with a converged quadrature,
  \|curl B\| / \|grad B\| ≤ 1e-8 at `d ≥ 0.2 a` and the Ampère loop unchanged; the
  vacuum-identity floor is reported as a function of `ns` rather than asserted
  to be small (measured 1e-5 at ns = 50, unchanged by extrapolation order or by
  the projection).

| PR | change | evidence | gate and kill rule |
|---|---|---|---|
| E0 level choice and per-period schedule (**landed upstream**) | select schedule levels by #312's calibrated estimate instead of the double-layer self-test, which passed errors up to 0.30 at a 1e-4 tolerance, and scale the default levels by nfp | merged as virtual_casing_jax #7 together with a field-interpolation fix: the exterior identity went from ~1e-4 to 5–9e-6 and the path is 36 % faster. The same PR closes L-EXT's B2 for VMEX: the third derivative at `d = a` went **1.2e-2 → 8.5e-9**, a factor 1.4e6 | vacuum and interior identities on nfp = 2, 3 and 5 decks meet the requested digits wherever `d ≥ 2h`; returned fields change only where the old level was under-resolved. VMEX side is #376, blocked on the 0.0.6 publish |
| E1 equivalent-source exterior field | **dropped 2026-09-19, killed by its own kill rule** (L-EXT 4). The rule was 1e-4 at `d = 0.02 a` with at most four times the surface grid's source count; the best fit measured (projected data, `s_src = 0.9`, 24 modes, 82k quadrature nodes) gives 3.2e-4 at 0.2 a, 2.4e-3 at 0.1 a and 9.8e-2 at 0.02 a. On the shipped, unprojected data it cannot fit at all — residual 2e-3…1e-2 at every depth and mode count, floored by the curl of E4. It is also not cheaper per target than the direct sum (0.55 ms against 0.30 ms at 256×128) | — | the row stays as the record of a closed negative result; E7 delivers what E1 was for |
| E2 tabulated field | an (R, φ, Z) table through `MgridField.from_parameterized_cartesian_field` with half-period symmetry, filled by E5 far from the surface and E7 near it. **Revised:** tabulate the vector potential A of E4's divergence-free sheet current and take the curl analytically, so the table is exactly solenoidal instead of divergence-cleaned after the fact. The first-order Taylor plan is retired on its merits — first order, not solenoidal, 56–416 s to build — and **the plan's "the Taylor plan is broken" verdict is withdrawn**: it came from a vacuum deck whose exact plasma field is zero, so the ~1e-5 it returned was its own error and the direct value it was compared with was the source-data floor, not a certified field; the finite-β record is 1.2e-3…1.7e-3 of B at 0.1–0.2 a, approximate and not broken | fill cost ~1e5 points × 0.3–5 ms = 0.5–8 min once, then ~1e-6 s per stage instead of an O(N) sum (inferred, not prototyped). Tracing needs this: an ESSOS `trace_field_lines` run is ~1e5 right-hand sides, each a single-point O(N) sum, and under `vmap` the schedule's `lax.cond` becomes a `select`, so every stage pays both levels and the self-test | Poincaré section invariant under table refinement; \|div B\| ≤ 1e-12 by construction; field-line tracing outside the LCFS in seconds with a stated error taken from the fill's own estimate |
| E3 source data | **withdrawn as written** (L-EXT 1.6): the LCFS covariant field from a better extrapolation does not remove the curl — a quadratic three-point `(15, −10, 3)/8` measures 5.0e-3 against 5.2e-3 at `d = a`, because the non-conservation is VMEC's discrete `J^s` and not the extrapolation order. Replaced by E4 | — | the finite-β interior identity target moves to E4's gate |
| E4 conserve current in the source data | build K from the surface-gradient part of the covariant pair, `ν_k = −i (k_t b_t + k_p b_p)/(k_t² + k_p²)`, keeping the means (they are µ0 I_tor/2π and µ0 I_pol/2π): a current potential, as NESCOIL and REGCOIL build it. Two FFTs, ~60 lines | measured: \|curl B\| / \|grad B\| from 5e-3 (`d = a`), 2.8e-2 (0.5 a), 9.2e-2 (0.2 a) to **≤ 6e-10**; B_plasma moves 0.07–1.2 % and grad B_plasma 0.8–14 %, which is the size of the source-data error the shipped field carries today. Because the true pair is a surface gradient (`J^s = 0`), an orthogonal projection cannot increase the data error in its norm | the source-data gate above. **Open as #381 with the projection off by default; flipping the default is the maintainer's call**, since it moves published exterior fields by up to 1.2 % and their gradients by up to 14 % |
| E5 closed-form derivative kernels | derivative tensors of 1/r contracted with K before the source sum (~60 moment sums), one pass giving B through grad³B; exposed as `gradgradB_fn`/`gradgradgradB_fn`, ~150 lines upstream | measured on the same finest grid, 48 targets, warm: **1.17 ms/point for all four orders against 23.6 ms/point in four nested-`jacfwd` calls (20×)**, 11× for the third derivative alone, first call 0.45 s against 3.1 s; identical to AD at 2.7e-15. The nested path pays both schedule levels plus the self-test on every call because `lax.cond` becomes `select` under `vmap` | equality with nested `jacfwd` to 1e-12 on the two-source torus; ≥ 10× on the grad B example; parameter VJPs keep working |
| E6 KST estimate as the level selector, per derivative order | af Klinteberg–Sorgentone–Tornberg error estimate 4: the complex root `t0` of `R²(t)` nearest the real axis, the 1-D estimate integrated over the other parameter with the combined linear root model, directions swapped and added; kernel order `k` has `p = 3/2 + k`. Two complex Newton solves on the boundary series per target, O(mnmax) and not O(N_sources), and invertible for the grid | measured against the true error of single-level uniform sums on the finite-β QA deck, 12 verified targets per distance: estimate/true median **1.6–2.1 across eleven orders of magnitude and all four derivative orders**, minimum 1.5, maximum 3.6, with no calibration constant; it drops below 1× only where the true error is already O(1) (`d < h`). Upstream as #9 (0.0.7 candidate), with #10 vectorising the grid-sizing call it drives (86× at 8 targets, 19× at 64, 5.2× at 512; the dominant cost is a pointwise Fourier evaluation of the level's nodes, independent of target count, so the per-target gain falls with batch size and a 19× gap at 512 targets is still open, deferred to the wiring PR) | never below 1× and never above 5× the true error on nfp = 2, 3 and 5 decks for orders 0–3 where the rule is resolved; the third derivative at 0.5 a meets the requested digits or is refused. It replaces the double-layer self-test and the two-level difference |
| E7 target-graded trapezoid for `d < 2h` | substitute `θ = θ* + u − a_p sin u`, `φ = φ* + v − a_t sin v` (entire, periodic, monotone for `a < 1`) and apply the same periodic rule in (u, v), with `1 − a = d/4h` per direction clipped at 0.995, so one fixed node count serves every distance | measured, max relative error over six verified targets against the converged reference: with **256×128 nodes — today's finest default count — ≤ 4e-9 in B and ≤ 3e-6 in grad B for every `d ∈ [0.005 a, 0.5 a]`**, where the uniform rule at the same count gives 1.4e-6, 2.5e-2, 4, 34 and 54; 192×96 nodes give ≤ 1e-6. The unoptimised prototype costs 40–90 ms per target, dominated by a non-separable boundary-series evaluation that is separable in principle (O((n_t + n_p)·mnmax) trig plus small matmuls). No published reference was found for this exact grading of a global toroidal rule; treat it as own construction, validated numerically | the quadrature gate above on nfp = 2, 3 and 5 at the default node count, with the KST estimate in the graded variable within 5×. It is a per-target method: right for point queries, surface-adjacent objectives and filling E2's table, wrong inside an ODE right-hand side — kill it for tracing rather than optimise it there |
| E8 interior field from the native VMEC form | evaluate B from `B^u = (χ′ − Φ′ λ_φ)/√g`, `B^v = Φ′(1 + λ_θ)/√g` with `√g` from the same R, Z, and a C² radial interpolant (cubic in ρ of the regularized coefficients); keep W2's implicit inverse map | measured against the exact oracle `wout_purely_toroidal_field.nc` at ns = 101: grad-grad B is 6 %, 2 %, 4 %, 17 % wrong and grad³B 33 %, 10 %, 15 %, 46 % at s = 0.3, 0.55, 0.8, 0.95, because `_radial_value_and_derivative` (`extender.py:401–438`) interpolates linearly in `s` and AD then returns a zero second radial derivative inside each cell; \|div B\| / \|grad B\| is 1–3 % at ns = 11, 21, 41 and 81 with no convergence | oracle grad-grad B ≤ 1e-3 and grad³B ≤ 1e-2 at ns = 101; \|div B\| / \|grad B\| converging at second order in ns. The two-step differentiable Newton inversion reproduces the ten-step unrolled result to 4e-15 for orders 0–3 and is the cheap regression oracle for W2 |
| E9 the three small defects | the `exterior_field` facade (`problem.py:896–939`) and the `equilibrium_from_x` factory (`optimize.py:3483–3515`) call `_source_nphi_for_digits` and forward `accuracy_check` instead of hard-coding `nphi = ntheta = 32`; `_has_plasma_sources` (`extender.py:822–832`) uses relative thresholds on `ctor` and β instead of absolute 1e-14 on dimensional quantities; #374's per-order derivative cache keys on the bound fields, or documents them as immutable | each has a reproducer: the facades bypass the geometry rule that #312 established; the vacuum QA wout (ctor 1.5e-10 A, `currumnc` up to 9.9e4 A/m² of axis noise) reports plasma sources, so `plasma="auto"` returns quadrature noise instead of zero; mutating `external_field`, `plasma_field`, `near_surface_plan` or `newton_iterations` after the first derivative call silently keeps the old graph | a facade-built field matches a `from_wout` field to 1e-12 on the QA deck; `plasma="auto"` returns exactly zero plasma field on the vacuum wout; a mutated extender either rebuilds or raises |

### Phase F, weeks 5–8: the free-boundary derivative

Revised 2026-09-19 from L-FB. Three of this phase's rows rested on expectations
that the measurements contradict, and they are corrected in place rather than
removed. **F1b's "matvec ≤ 0.1× one NESTOR call" is unreachable for a coupled
matvec, and the reason is arithmetic:** at the NCSX deck (ns = 15,
mpol = ntor = 3) a free-boundary value-and-gradient costs 7.213 s against
0.174 s for the value, over about 671 coupled GCROT matvecs, i.e. ~10 ms per
matvec, while a whole NESTOR call at that resolution is 1.1 ms. A coupled
matvec is one bulk block solve plus the edge coupling, and a dense edge
response makes only the edge part free, so even a zero-cost edge multiply
leaves the matvec near 10× a NESTOR call — a factor 100 from the gate (the
split is measured, the per-matvec division is inferred). The two metrics that
replace it are **matvecs per gradient** (247 probe columns and ~671 coupled
matvecs today; ≤ 50 unrestarted on the real NCSX edge Schur operator, measured)
and the phase's own **value-and-gradient ≤ 3× the value** at the deck's
resolution, not at mpol = ntor = 3.

| PR | change | evidence | gate and kill rule |
|---|---|---|---|
| F-pre jit the free-boundary pullback (**merged**) | the implicit pullback converted traced arrays to NumPy (`_solve_bwd_impl` → `_projected_residual`), so no free-boundary objective could be wrapped in `jax.jit` | merged 2026-09-19 as #375. Its test had been monkeypatching the function away, which is why the suite never saw it — one of the four problems A6's jit-disabled default hid | a jitted free-boundary objective's value and gradient match the eager path to 1e-12 relative on the CTH case |
| F0 measure (**answered**) | the warm split of one NESTOR call | measured (L-FB, §2): the kernel scan is **50–55 %** and the **LU 1.5–2 %**, so VMEC++'s benchmark-header claim that build and factorisation dominate does not hold for this port, and the flop estimate that put the LU near 5 % was the right order for the wrong reason. Baselines stand: value-and-gradient 41.5× the value at ns = 15, mpol = ntor = 3, with 70.6 s of adjoint compile the first time in a process; fixed-boundary equivalents 1.91× (vacuum) and 5.87× (finite β) | closed; its output is the split above, which re-ranks F1a and F1b |
| F1a linearize once | hoist the coupled linearization out of the host GCROT lane, where `_transpose_matvec` builds its VJP inside the jitted matvec so every transpose matvec re-runs NESTOR's full assembly and LU | the hoist keeps its evidence. **The `custom_linear_solve` half is not a speed item**: `jnp.linalg.solve` against `lu_factor`/`lu_solve` makes no measurable difference at these sizes because the LU is ≤ 5 % of a call. Keep it only if it simplifies the tangent (`dpot = A⁻¹(db − dA·pot)` is exact), and claim nothing for it | transpose identity at 1e-11 and the NCSX adjoint value unchanged to 1e-10 relative; cold compile peak not above today's |
| F1b edge response matrix (**superseded**) | as written, build NESTOR's dense response once per gradient and use it as the solve. **Superseded:** the dense edge response ships as a Krylov **preconditioner** and response lane rather than as the solve (#380, measured **4.8×** at the real deck resolution), and the endpoint is unrestarted GMRES on the exact reverse-mode edge Schur operator | the response is affordable (0.02 s at mpol 3, 1.40 s at 8, 9.0 s at 12 per gradient) but **not low rank** — rank 222 of 240×256 at mpol 8, singular values still 1e-3 at index 151 — so no randomized shortcut exists. The expectation that a forward-built Schur certified against the reverse-mode operator would miss its 1e-9 acceptance between mpol 7 and 8 **was tested and did not reproduce** (§2): zero fallbacks at mpol = ntor = 8 in both lanes, response-linearized JVP against the exact coupled one at 1.41e-13 relative. The honest cost of resolution is the payoff — the lane's speedup is 2.16× at mpol 8 against 4–5× at mpol 3–7, because the response build is a larger share | response build ≤ 300 NESTOR-call equivalents at mpol ≤ 10 (measured 195 at 8, 273 at 10, 340–645 at 12, so the bound holds only to mpol ~10); JVP against finite differences of the vacuum pressure at 1e-6; coupled-residual acceptance unchanged. **The old matvec gate is retired**, replaced by the two metrics above |
| F2 Schur Krylov (**promoted from fallback to endpoint**) | unrestarted GMRES on the edge Schur system with the exact reverse-mode operator, with F1b's dense Schur LU as its **right preconditioner** when it has been built; restart length ≥ 64 | measured on the captured NCSX matrix, S and Sᵀ, three right-hand sides: residual 1e-1 at 30 matvecs, 1e-4 at 40, **≤ 2e-10 at 50**, superlinear once the ~35–40 outlying eigenvalues are captured, after a 30-iteration plateau (residual 0.97 at 10, 0.4–0.8 at 20) that a restarted method with m = 30 would never leave. With a preconditioner at 1e-6 operator error it takes 4 iterations and at 1e-4 it takes 8. **The previous-iterate Schur LU written into this row is not supported**: at 1e-2 operator error it gives 44 iterations against 50 for none, and at 3e-2…1e-1 it is 2.5–4× worse (123–189). The lane's two-sided balancing belongs to the dense solve, not to the Krylov solve (plain row scaling made it 484 iterations) | NCSX ns = 15 value-and-gradient ≤ 3× the value at the deck resolution; **zero certificate fallbacks on an mpol = ntor = 8 deck** — measured on 2026-09-19 over five perturbed NCSX trials and met; restart ≥ 64. Risk: the outlier count is unmeasured above mpol 7. Carry a recycled deflation subspace across optimizer iterates (GCRO-DR, `gcrot(recycle=(C,U))`, shared with S7) — untested, so it is an arm and not a claim |
| F3 inexact tolerance, not a frozen derivative (**frozen arm killed**) | keep the cheap-gradient option as a looser Krylov tolerance certified by the exact adjoint residual | the frozen matrix derivative drops `A⁻¹ dA·pot`; that term measures **0.10–0.18** over four random low-mode boundary perturbations, independent of grid and of quadrature oversampling, against the option's 1e-2 gate. **Killed on the measurement**, not deferred | the looser-tolerance arm keeps its gate: the free-boundary single-stage example ≤ 10 CPU minutes with the adjoint residual reported at every trial |
| F4 test the derivative-precision floor (**narrowed**) | L-FB's R2 asked for an mpol-aware acceptance. **It is not needed for the response lane**, whose design never compares a forward-mode kernel derivative against a reverse-mode one (§2), so what survives is the test and not the tolerance change: a parametrized dot-product test that pins the measured floor, and the same test on the response's own JVP against a central difference of the vacuum pressure. Reopen the tolerance half only for a lane that does make such a comparison | the `cmns` contraction's cancellation grows ~30× per +2 in mpol (7.3e4 at 8, 1.6e8 at 12) and differentiation amplifies it: the JVP-vs-VJP identity of the whole map is 2e-11 (mpol 6), 3.5e-8 (8), 9e-7 (10), 4.5e-5 (12), 2.9e-4 (12, nfp 5), while the non-singular kernel alone holds 1e-13…1e-15. The tests already do this informally (`tests/test_freeboundary_implicit.py:376–377` uses `adjoint_tol = 1e-5` at mpol 10). The cancellation does not reach the coupled certificate: at mpol = ntor = 8 it passes with zero fallbacks, while the neighbouring response-versus-finite-difference check degrades 1.1e-9 → 2.0e-6 over the same range — which is exactly why the floor is worth a test even though the tolerance does not move | the stated floor (1e-10 to mpol 6, 1e-7 at 8, 1e-5 at 10, 1e-3 at 12) is asserted by a parametrized dot-product test, so a regression *or* an improvement is visible, and the response-versus-finite-difference floor (1.1e-9 at mpol 3, 2.0e-6 at 8) is pinned beside it; **no silent GCROT fallback**: A1's counters report every one, and the count is zero at mpol 8 today |
| F5 five known-answer tests for the vacuum solve and its derivative | T1 interior-source manufactured solution (assert `potvac` against exact coefficients and `bsqvac ≈ 0`, pinning today's mpol 8, 22×20 numbers — potential 8.3e-3, grad Φ rms 2.6e-2 — as upper bounds); T2 convergence order (≥ 1.9 today, ≥ 2.8 after the kink fix, ≥ 4.5 after an O(h⁵) rule); T3 null shape-gradient (in T1 `bsqvac = 0` for every surface enclosing the sources, so the JVP along any boundary direction is O(discretisation) — this certifies a dense response against an answer that is not a finite difference of the same code); T4 Hadamard cross-check within 5 % at mpol 6 and 2.5 % at 8; T5 the dot-product floor of F4 | none of the existing vacuum tests has a known answer: all are parity, A/B or recurrence checks. Every number above is already measured. Dommaschk potentials are **not** usable here (singular on R = 0, which lies in the exterior domain), and an interior current loop gives a multivalued potential unless it is contractible inside the plasma, where it reduces to the charge sources used | each test asset-free and under 10 s except the full-lane coupled one; T1–T5 land before any quadrature change, so that change has to improve them |
| F6 NESTOR's quadrature | three steps, in order: (i) a 20-line Euler–Maclaurin correction of the antipodal kink in `_nonsingular_terms`, opt-in and parity-breaking; (ii) decouple the vacuum quadrature grid from the plasma grid (an oversampling factor, with `bsqvac` synthesised back on the plasma grid from `potvac`); (iii) replace the analytic subtraction and the `T_l` recurrences by the zeta-corrected trapezoidal rule, O(h³) first and O(h⁵) as the target | (i) measured in a patched copy: source-term order 2.9–3.0, error ÷6.4 at N = 32 and ÷11 at N = 64, potential error ÷1.3–2.4, at zero cost. (ii) measured: doubling the vacuum grid divides the potential error by 4.7 and the grad Φ error by 4–5, at 4× cost on full updates only (26 → ~100 ms at mpol 12). (iii) prototype measured 2–5× more accurate than NESTOR at equal grid at observed order 2.85, condition number 2.88, with no `T_l`, no tan tables and **no `cmns` cancellation — which is what removes F4's derivative-precision floor at its root** | (i) T2 order ≥ 2.8 for the source term, with the VMEC2000 parity lanes keeping the legacy path. (ii) T1 improves with the oversampling factor at unchanged plasma results. (iii) T1 at mpol 12, 30×28: potential ≤ 1e-4 and dot test ≤ 1e-10. **Kill (iii) if the O(h⁵) rule is not ≥ 10× NESTOR at 30×28 on the li383 and a W7-X-like boundary**; its weights need third derivatives of the surface and are unmeasured on stellarator shapes |
| F7 hoist the per-gradient closures | make `cfg` the only static key of the free-boundary Schur lane and pass every per-trial array as an argument, as the fixed-boundary lanes already do | this is the structural fix for two symptoms at once: the Schur lane's **0.29 GiB per-gradient leak** and its retrace cost. The fixed-boundary loops have neither, and the reason is exactly this property (`_refine_step_core`, `_adjoint_block_core` and `_preconditioned_residual_lane` are module-level `jax.jit` with `cfg` static). **This conflicts directly with S2**: the inter-rung `jax.clear_caches()` is the workaround that hides the leak today. They must not both land — F7 is the fix, S2's narrowing is the workaround, and the call is retired once F7 is in | live arrays and live bytes flat across 14 perturbed free-boundary trials, as the fixed-boundary lanes measure today; zero new XLA compilations after the first trial; the NCSX adjoint value unchanged to 1e-10 relative. S2 lands only after this |
| F8 free-boundary Newton finish (arm, inferred) | the factorisation the adjoint uses, `J = A + U Wᵀ` with the bulk LU and the edge Schur solve, is also an exact Newton step for the forward coupled root; B1c's block-preconditioned finish is fixed-boundary only, and adding the Woodbury edge correction gives the free-boundary analogue | inferred from the adjoint structure, not measured. It attacks the 59 s FTOL 1e-9 free-boundary solves of §2. A movement-based `nvacskip` cadence (refactor when the edge has moved by a set fraction, instead of the fsq rule capped at 10) would cut the vacuum share ~1.7× late in a solve: parity-breaking, modest, low priority | gate it like B1c: the same iterations to the switch, the final state within the certificate tolerance, wall ≤ 0.5× on CTH and NCSX |
| F9 free-boundary determinism, with its bound | keep the fixed reference seed that makes the objective a function of `x`, and **rebuild the reference on a status-2 trial** — the identified next move | measured (FB-B, #383, §2): the 57 % objective spread goes to bit-identical and a warm trial from 21–26 s to 2.0 s, at the cost of 1–4 % roughness, because main's apparent smoothness came from every finite-difference leg restarting on the previous call. The stall measurement sets the bound: 2 % on every coil current takes 2,500 iterations and lands 9.2e-3 from a cold answer that converges in 76; 1e-3 on one dof reconverges in 106 | objective bit-identical at a repeated `x` (met); a stated step size beyond which the reference is rebuilt, with the rebuild triggered by a status-2 trial; the failed-trial rate on the free-boundary single-stage example not above its pre-fix rate. **The 1–4 % roughness is reported per run, not asserted away** — if a line search cannot make progress through it, the reference-rebuild policy is what changes, not the gate |

### Phase G, weeks 7–9: the comparison and the package

The cross-code table this phase was going to build already exists, is
third-party, and includes VMEX: `itpplasma/benchmark_vmec`, 344 cases through
twelve codes, published 2026-09-14. Phase G is therefore to reproduce that row
rather than duplicate it — read it by joining `comparison_table.csv` with
`plots/quality.csv` on `residual_over_tolerance`, not by the `status` column,
which counts unconverged runs as successes for several codes — and to
contribute an adapter fix upstream if the harness invokes VMEX in a way that
misrepresents it. What that corpus does not carry, and this phase still owes,
is the differentiable comparison: VMEX against VMEC++ 0.7.4 on cold, warm, hot
and adjoint costs on named hardware, and against DESC's omnigenity
time-to-metric, from `benchmarks/optimization.py` records; the three-way
gradient comparison and the Taylor test of the 2026-09-06 plan then close on
the fixed anchor.

A local four-code run on 2026-09-16 (one machine, one process per code) sets
the scale that table has to record. `input.circular_tokamak`: VMEC2000 0.23 s,
VMEC++ 1.20 s, VMEX 0.16 s warm and 7.6 s cold, DESC 24.3 s, all four at volume
473.74101, and VMEX against VMEC++ wout by wout at 3.6e-16 on volume and
1.2e-16 on `iotaf`. `input.LandremanPaul2021_QA_lowres` at an adequate
iteration budget: VMEC2000 10.30 s, VMEC++ 11.12 s, VMEX 13.87 s warm and
31.6 s cold, DESC 47.0 s, volume agreeing to eight figures. At the deck's
shipped budget (`NITER_ARRAY = 600 1000 1000` against `FTOL 1e-13`) no
VMEC-family code converges, VMEX included — so that deck measures the budget,
not the code. Then §5 resumes.

### Phase H: one example template, and a repository that carries no dead weight

Added 2026-09-19 from EX, SLIM-A and SLIM-B. The rule for this phase is that
**correctness is separated from formatting**, and **"safe now" is separated
from "needs a maintainer decision"**: a row in the second group is not started
until the decision is recorded here.

| PR | change | evidence | gate |
|---|---|---|---|
| H1 examples, correctness only | rewrite the coverage guard in `tests/test_examples.py` as an explicit `EXECUTED_EXAMPLES` set that must partition the shipped examples with `UNTESTED_EXAMPLES`, plus an assertion inside `_run_example`, so an undeclared example fails instead of passing; run the four LASYM optimization examples; delete the unconditional re-assignment that made the `ci_smoke` budget inert in two of them; wire the dead `IOTA_FLOOR`/`MIRROR_LIMIT`/`ELONGATION_LIMIT` constants into their callbacks; remove the commented objective row whose constant does not exist; give `QA_optimization_DMerc_vacuum.py` its own output stem; correct six docstrings to measured values | the old guard matched **basenames**, so eight `stellarator_asymmetry/*` examples were masked by their symmetric namesakes and never ran; `QH_optimization.py` then exited 1 in 42 s with cost 5.0000e+11 and zero iterations (fixed: rc 0 in 62 s, cost 0.570 → 0.344) and `QA_optimization.py` was solving the full grid in the smoke lane (fixed: rc 0 in 49 s). 66 shipped examples = 47 executed + 19 exempt, each exemption carrying a reason | a parser test forbids re-assigning a constant the `ci_smoke` block set; every shipped example is executed or exempt with a stated reason; `tests/test_cited_paths.py`, `tools/check_docs_prose.py` and `ruff check examples/` pass. Three examples stay exempt (`QA_maxJ_continuation.py`, `QA_optimization_DMerc_vacuum.py`, `QA_optimization_global.py`) and say so |
| H2 one template, family by family | the twelve rules and eight family variants of the EX proposal: shebang and imperative docstring whose every number is measured or dropped; one import block; one `UPPER_CASE` parameter block with a comment per group and no packed tuples; alternatives as a commented block introduced by what they buy; one `ci_smoke` switch, read once, last in the block; the verbatim `End of input parameters.` banner; five section banners in a fixed order; straight-line code below the banner; no commented-out code; physics- and accuracy-affecting numbers as named constants, and grids the library can size left unset; one `OUTPUT_NAME` stem per file; `print(f"Wrote {path}")` | six exemplars converted and verified: stdout identical (byte-identical for three) and every artefact identical in name and byte size, the three intentional deltas being `wrote` → `Wrote`, the monitor files following `OUTPUT_NAME`, and two files announcing monitor files they always wrote silently. Conversion is close to line-neutral except the scalar family, which grows by inlining its driver. 41 of 66 examples pass a physics- or accuracy-affecting literal below the parameter block; #384 is the precedent for deleting a pinned facade grid the library sizes better | per PR: a before/after run at `VMEX_EXAMPLES_CI=1` in a clean directory, compared by stdout and by artefact name and size; the family's example tests, `tests/test_cited_paths.py`, `tests/test_test_manifest.py`, `tools/check_docs_prose.py` and `ruff check examples/`. Order: E (25 files), F (13), A+B (12), C (3), H mirror (5), then D (6) and G (5) rebased last, after their owners land |
| H3 tests and coverage, safe now | `--dist loadfile` in the xdist PR lanes; the jit fixture of A6; path-anchor the coverage omit so `mirror/turbulence.py` is gated; replace `importorskip` on hard dependencies (82 sites in 49 modules) with plain imports; delete the two `essos.mgrid` tests for an API that release does not have; fix the two stale docstrings (`core/__init__.py`'s "legacy modules" and `implicit.py:36–44`'s claim that the raw formulation is kept for tests); remove `ModeTable.m_is_even/m_is_zero/m_is_one`, the `_iotas_half` re-export and the never-passed private kwargs; consolidate duplicated test scaffolding; add the cheap PR tests that lift 22 of the 24 sub-95 % modules over the line | `--dist loadfile` alone removes ~2,500 s per PR of module fixtures rebuilt per worker (one fixture was built three times at 394 + 367 + 365 s) and also removes the cross-module jit-state leak; the `importorskip` sites can never skip legitimately and would turn a broken install into a green run of skips; the new tests are +400…+450 lines against −800…−900 removed | every module at or above 95 % except `core/freeboundary_implicit.py`, whose 205 statements belong to the agent editing it (one non-`full` host-Schur adjoint test on a minimal vacuum free-boundary case); no test lane loses a selector it uniquely owns |
| H4 repository slimming, safe now | point cloners at `--filter=blob:none` or `--depth 1` in `README.md`/`CONTRIBUTING.md`; add a CI guard rejecting a new tracked blob over ~500 KB–1 MB unless allow-listed; delete the 10 confirmed orphan files (262 KiB); prune the stale local worktrees | a default clone is 40 MiB and `--depth 1` is 5.6 MiB, so the doc line is the whole user-facing win. The blob guard is the **only** lever that controls the 834 MiB mirror figure going forward, because `refs/pull/*` is GitHub's to keep and no rewrite or branch deletion touches it. The 10 orphans survived four search passes and nothing, not even an "uncited but kept" list, names them. 3.45 GiB of the 6.55 GiB of local worktrees is stale by git's own `prunable` flag or by checking out a delete-safe branch | the guard fails on a new 2 MB fixture and passes the tree as it stands; the orphan deletion leaves `tests/test_figure_provenance.py`, `tools/pack_reference_assets.py` and the docs build unchanged; worktree pruning has no upstream effect at all |
| H5 **needs a maintainer decision** — record the answer here before starting | (a) delete the 74 delete-safe branches — the list was derived when five PRs were open and ten are open now, so it is re-derived against the current PR list before anything is pushed; (b) the 13 ASK branches, three of them external-contributor work with no PR ever opened, which means "not submitted yet" at least as often as "abandoned" — ping before deleting; (c) rewrite `main`'s history; (d) move `examples/data/mgrid_ncsx_c09r00_small.nc` to the release-asset mechanism; (e) the four input decks with zero hits and the five with a manifest entry but no loader; (f) retire the square-root homotopy lane; (g) drop the five duplicate "representative physics" PR lanes; (h) default jit ON in tests, or one nightly jit-on rerun; (i) move the NumPy reference twins out of `freeboundary.py` (public names); (j) wire `_solve_stage_traced`/`_guess_axis_traced` into B4 or remove them; (k) the docstring policy for 14,540 docstring lines; (l) the six EX decisions — the monitor-file rename, deleting `_scalar_driver.py`, a `vj.wout_from_result` helper, promoting `_sample_closed_polyline`, moving two non-examples to `tools/`, and a smoke path for `epsilon_effective.py` | (c) is **not recommended and needs no decision unless the maintainer disagrees**: it reclaims ~22.65 MiB of an already-33.7 MiB pack while invalidating every fork, open PR branch, 36 tags and the PyPI and DOI provenance that points into the history, and it cannot touch the PR refs that are 94 % of the mirror. (f) is −1,620 source and −990 test lines, one nightly row and ≥ 1,560 s of a PR lane, keeping a 40-line rank gate for the one structural check worth preserving. (g) is −1,835 s per PR: all 107 selectors of those five jobs belong to modules a parity lane runs in full in the same workflow, same Python, same extras, same marker filter | each answer is written into this row with its date; nothing in H5 starts before its answer is here |

**Open item, observation and not assertion.** The DMerc vacuum example's stage
failure was not reproduced in this review. The place to look first is the
stability weight escalating as `10**(stage-1)` against a `stability_scale`
computed once at the seed; until someone reproduces it, that is a hypothesis
and not a defect.

### Kill rules and things not to repeat

- B1 is killed if Newton-finished states differ from descent-plus-refinement
  states by more than the certificate tolerance on the P1 matrix, or if
  Jacobian resets become more frequent on the shipped decks.
- B5 is killed by its own gate; do not extend it to multi-host sharding. Its
  sharding arm is **dropped outright** as of 2026-09-19: the iteration stops
  scaling at four cores and burns 63 % more CPU at eighteen for the same wall.
- F1b is killed if building the response matrix costs more wall time or
  compile memory than today's GCROT lane on NCSX ns = 15; F2 is then the route.
  **Superseded 2026-09-19**: the response ships as a preconditioner and F2 is
  the endpoint, so the surviving kill rule is F2's — zero certificate fallbacks
  on an mpol = ntor = 8 deck and value-and-gradient ≤ 3× the value.
- D2 is killed if it cannot reach the current surrogate's final QI metric on
  the shipped cases.
- E1 is killed if the fit cannot reach 1e-4 at `d = 0.02 a` on the vacuum
  identity with at most four times the on-surface grid's source count, and the
  hedgehog fallback cannot either; E2 with the distance rule is then the product.
  **Executed 2026-09-19: E1 is killed by this rule** (best measured 9.8e-2 at
  0.02 a), and E7's graded rule delivers what E1 was for.
- Closed as negative results on 2026-09-19, each with the measurement that
  closes it, none to be re-proposed without new evidence: **CPU sharding of the
  iteration** (no scaling past four cores); **pad-and-mask one compile for the
  whole ladder** (+27 % execution against 2.9 s of cold compile, a pure loss
  in-process); **XLA flag tuning and tiered compilation on CPU** (best flag set
  saves ~1 s of a 12–15 s cold run; O0 saves 28 % of compile and loses 2.5× on
  execution; the cheap tier compiles in 70–85 % of the optimized tier's time, so
  there is nothing to overlap); **XLA CPU scheduler flags** (both concurrency
  settings double warm wall time); **Anderson, NGMRES, nonlinear CG and L-BFGS
  as descent accelerators** (theory plus VMEX's own arm at 0.935× and 2.4×
  worse; FIRE's table says damped dynamics already beats CG by 2–5×);
  **learned warm starts** (the one hard number is 28× slower than VMEC, and
  neither source feeds a prediction to VMEC at all); **mixed precision in the
  block factorization, for now** (Carson–Higham needs `κ·u_low < 1` against a
  measured conditioning near 6e12 — estimate the condition number first, do not
  run a trial); **cyclic reduction, PCR and banded LU** (no CPU parallel
  headroom, and the GPU sweep found no win); **Broyden and other secant
  Jacobian updates** (0.83 of the exact step's predicted progress at 48 dof and
  negative on 3 of 25 pairs, against at most 9 s saved on the QI run);
  **`vmap`-batched trial points** (structurally impossible behind single-slot
  host state, and bounded by 6–11 % of a run even if free); **multi-fidelity
  `ns` ladders** (both shipped examples already run every trial at the coarsest
  rung); **choosing forward tangents against adjoint rows by problem shape**
  (the block lane's cost is dof-independent: 3.1 / 3.5 / 2.3 s at 8 / 24 / 48
  dof); **the coil-side reverse pass in single stage** (1 % of the run);
  **reusing a neighbouring trial's factorization as a solver** (diverges by up
  to 11 orders); **randomized or low-rank NESTOR response** (rank 222 of 256);
  **the frozen-matrix free-boundary derivative** (10–18 % error against a 1e-2
  gate); **the Hadamard response as operator or preconditioner** (1–3 % inexact,
  and the preconditioner numbers put that class at no gain); **partition-of-unity
  and BIEST quadrature inside the equilibrium loop** (~84 points per period for
  3e-4 on W7-X, a 3–5× geometry upsample); **equivalent sources near the
  surface** (E1 above); and **a history rewrite of `main`** (~23 MiB off an
  already-34 MiB pack, against every fork, tag and provenance link).
- Do not repeat: rcon/zcon fusion (`c595c257`), the separable-versus-dense
  synthesis switch, W7-X polish runs, winding-surface work (#301/#303/#304),
  per-row kernel tuning before the anchor is fixed, and comparisons against
  VMEC++ wheels older than 0.7.

## 5. Force balance: decisions retained

These verdicts stand and are not reopened by this revision.

- **E1 passes.** The energy gradient equals complete virtual work to 7.9e-14;
  omitting lambda leaves 3.1 % ([`benchmarks/e1_functional_consistency.py`](https://github.com/uwplasma/vmex/blob/07a47279d5cea819bb23c5329fab7f14cced2456/benchmarks/e1_functional_consistency.py)).
- **E2 passes the shaped tokamak and fails 3-D.** Full R/Z reference 188.5 vs
  the structured chart's 335.3 N m⁻³; on the finite-β QA deck the best step
  reaches 2.31× against a 10× gate and the chart advantage is 1.1–1.2×
  ([`benchmarks/e2_dense_reference.py`](https://github.com/uwplasma/vmex/blob/07a47279d5cea819bb23c5329fab7f14cced2456/benchmarks/e2_dense_reference.py)). Six toroidal planes under-resolve the
  nfp = 2 deck; use twelve.
- **The axis source data and fit are the 3-D limiter.** Three independent
  lines converge on it: E2, the residual-versus-resolution scan
  ([`benchmarks/residual_vs_resolution.py`](https://github.com/uwplasma/vmex/blob/07a47279d5cea819bb23c5329fab7f14cced2456/benchmarks/residual_vs_resolution.py), near-axis residual rises with spline
  refinement) and the refuted knot-grading hypothesis
  ([`benchmarks/knot_grading.py`](https://github.com/uwplasma/vmex/blob/07a47279d5cea819bb23c5329fab7f14cced2456/benchmarks/knot_grading.py)). `lift_high_order_state` must reject bases
  with unfed spans instead of returning a minimum-norm fill.
- **Coordinates.** Keep `ρ^|m| q(s)` with B-splines in `s`; no chart
  replacement, no generalized toroidal angle, no ρ-uniform mesh.
- **E3 and the resolution ladder** are gated behind Phase G; they run on the
  office box with the budgets and kill rules of the 2026-09-06 plan.
- **Contract.** The published residual is Thun's `F_norm` with DESC's volume
  average over `s ∈ [0.1, 0.99]`; the bounded `eps_F ≤ 2` is acceptance only;
  `solve()` reports near-axis, bulk and edge separately.

## 6. Open pull requests and the release candidate

> **Superseded 2026-09-21.** The PR dispositions below are historical. Current PR state is the Part I table "Open pull requests at this checkpoint".

State on 2026-09-13, evening. "Merge" always means after explicit maintainer
approval with every real CI lane green; the unsigned-commit PR gate is the only
check that may be red.

| repository and PR | state | disposition |
|---|---|---|
| vmex #308 | this plan, documentation only | merged; agents read `plan.md` on `main` |
| vmex #300 | DESC input bridge; lanes green except a cancelled Python 3.12 fast lane, re-run | merge when that lane is green |
| vmex #309 (A4) | documentation matched to records | merged `1aa5465e` |
| vmex #312 (A2, merged `68a119e9`) | exterior-field oracles and achieved-error estimate | merge when CI is green; warn-by-default kept (a checked eager call costs 2.2–2.5×, traced calls are unchanged); follow-ups: E0, and forward `accuracy_check` through the `exterior_field` facades in `optimize.py` and `problem.py` |
| vmex #310 (A1) | optimization counters and their record [`benchmarks/optimization_counters_20260913.json`](https://github.com/uwplasma/vmex/blob/07a47279d5cea819bb23c5329fab7f14cced2456/benchmarks/optimization_counters_20260913.json) | merged `2b9d3a3e`; next #319 → #320 → B4b |
| vmex #311 (A3, merged `0c083539`) | fixed-boundary single stage meets every target on a full run (min |ι| 0.4277 ≥ 0.42, aspect 3.979 ≤ 4, B·n RMS 0.80 % ≤ 1 %, coil clearances and curvature within limits, independent ns = 101 check converged; 2,959 s, 301 trials); smoke mode 136 s against main's 195 s; record `benchmarks/single_stage_profile_m4.json` | merge when CI is green; follow-ups: quasisymmetry worsened 0.101 → 0.113 under the constraints (C3), the constraint wrapper moves into a library helper with C3, and a second full run measures run-to-run spread |
| vmex #313 and #314 (merged `373f1e83`, `746215d3`), #315, #318, #316, #317 (S1) | #299's source re-landed as six focused PRs, in merge order: Boozer λ (#313), host trial solves (#315), Thomas selection and batching (#314), linearization reuse and field-line synthesis (#318), vacuum contraction and saved pullbacks (#316), plotting and optional magnetic-only projection (#317); 12–114 net lines each, no plan, record or handoff files | merge in that order when CI is green; #316's CTH free-boundary gradient check passes locally; #318 needs its counter rows before merge; #314's c3d failure was a test defect already on `main`: `test_qi_regression_pin_and_jit` pins the QI total on the axisymmetric Solov'ev deck, whose toroidal Boozer coefficients are about 1e-16, so the well argmin ties and 1e-15 noise flips the total between 0.13626 and 0.13500; #323 (merged `b0646713`) moves the pin to the golden `wout_li383_low_res`, whose minimum is unique; raise the SOLVAX floor to 0.21.0 once it is on PyPI |
| vmex #299 | green, but source mixed with a 630-line logbook and a 1,159-line record | close once #313–#318 merge; S1 carried all of its source |
| vmex #302 | green, but two commits add about 57,000 lines of HINT handoff evidence; its 1e-10 primal certificate is unreachable on the seed deck: at the reachable |P(gc)| ≈ 1.88e-7 every trial would fail `primal_tol` and return value-only, so the optimizer would never receive an implicit gradient (B1); no finish certifies better than 6.6e-9 on that deck, and the floor comes from non-gauge soft λ modes | do not merge; its three source commits wait for B1b's answer on the near-null λ modes |
| vmex #306 | four failing lanes, based on #302 | hold for B1 |
| vmex #319 (B4a, merged `c5ee2e0d`; total compile seconds 53.1 → 48.8 on QA and 50.2 → 36.5 on QI; its extra build programs are second copies caused by the committed final carry, with build compile time 4.66 → 4.64 s on QA and 8.09 → 8.86 s on QI, not removable without restoring the across-trial duplicates it removes) | #307's seven lines re-landed on #310's branch plus a two-line reorder that removes the extra compile #307 caused (the donation copy recompiled for a partly committed carry; cth ladder compiles cold/warm/direct 243/0/0, as before #307) | merge after #310, when CI is green |
| vmex #320 (B1a, retargeted to `main`; B1's QA_lowres run shows the plain rule would discard a certification, so it gains a linear-progress guard: stop only when the unconverged inner solve gained fewer than three digits and the step made no progress) | refinement stops after an unconverged step that does not lower |F| (+14/−9 in `implicit.py`, jit-exercised test on both JAX versions); stacked on #319 as `9b77a10e`, calling the shared commitment helper at both refinement call sites with one compile per lane pinned; QI first derivative 58.9 → 40.2 s with value, gradient, residual, Jacobian and refined state bit-identical | merge after #319, when its benchmark rows show 6,000 → 2,000 GCROT iterations with bit-identical outputs and CI is green |
| vmex #321 (B4b, retargeted to `main`) | stacked on #319: concrete JAX-lane calls reuse the host lane's certified executables while traced calls keep the compiled program; JAX-lane compile 41.6 → 0.10 s (QA) and 71.0 → 0.12 s (QI), benchmark wall 107.8 → 56.1 s and 164.9 → 77.3 s with host values and gradients bit-identical; constraint baselines 4 → 1 compiles and predictor 2 → 1; +36/−6 library lines | merge after #320 when CI is green |
| vmex #324 (B4c, first commit) | the Jacobian retry keeps the caller's `use_fft`; the new test fails without the fix | merge after #321 when CI is green |
| vmex #325 (B4d) | `_block_lane` compiles once across default-device contexts; stacked on B4c's branch | merge after B4c; fold its guard and #321's into one device helper |
| vmex #307 | superseded by #319 | closed |
| vmex #301, #303, #304 | winding surface | parked |
| booz_xform_jax #8 | opt-in magnetic-only projection, checks green; magnetic-only value 2.99 → 0.99 ms (symmetric) and 5.02 → 1.50 ms (asymmetric) on an RTX A4000 | merged `8e0208ae`; tagging 0.3.0 is the maintainer's release step |
| SOLVAX #105 | release 0.21.0 of merged #100–#104: checked Thomas GPU launch overhead, halved principal inverses, nonfinite root rejection | merged `60a87b21`; tagging 0.21.0 publishes it to PyPI and is the maintainer's release step; VMEX raises its floor once 0.21.0 is on PyPI |
| virtual_casing_jax | nothing open; 0.0.5 suffices for #312 | open an upstream PR for E0 (level choice by a calibrated estimate, per-period default schedule); no release needed for 0.9.0 |

**VMEX 0.9.0, released 2026-09-15.** It ships the DESC bridge, an exterior
field that reports its achieved accuracy, per-evaluation counters,
documentation and examples that match their records, #299's component
speedups, and — beyond what this section planned for it — B1c's Newton
finish, B3a/B3b's exact block adjoint and B4a–B4d's compile hygiene, so the
QI and single-stage examples converge and meet the targets they print.
booz_xform_jax 0.3.0 and essos 0.17 were still untagged at the tag, so the
magnetic-only Boozer projection and the twelve ESSOS examples need a git
install; SOLVAX 0.22.0 is on PyPI. It does not fix cold compile (B4's
remainder, B5) or block-Jacobian assembly (C1), and the 0.9.0 notes say so.

**State on 2026-09-19.** Merged in VMEX since the last update: **#369** and
**#370** (the two weekly-lane failures), **#372**, **#373** (weekly sharding,
with per-shard measured durations), **#374** (compile the nested field
derivatives once per order) and **#375** (a jitted free-boundary pullback builds
its projected residual — Phase F's `F-pre`). Merged upstream in
virtual_casing_jax: **#7** (level by calibrated estimate plus a
field-interpolation fix; the exterior identity goes from ~1e-4 to 5–9e-6 and the
path is 36 % faster; this is what closes L-EXT's B2 for VMEX) and **#9** (the
KST a-priori estimate, the 0.0.7 candidate); **#10** vectorises the grid-sizing
call, **86× at 8 targets, 19× at 64 and 5.2× at 512** — and the cause was not the
Python loop both sides assumed but a pointwise Fourier evaluation of the level's
nodes whose cost is independent of the target count, which is why the gain
shrinks with batch size; mode truncation and subset sizing are deferred to the
wiring PR with the remaining 19× gap at 512 targets on the record. 0.0.6 is
bumped on `main` and awaiting publish. Open at the time of writing: **#376**, the VMEX side of E0,
blocked on that publish; **#377**; **#379** (the two field-query examples);
**#380** (the dense edge response as a Krylov preconditioner and response lane,
measured 4.8× at the real deck resolution — Phase F's F1b/F2); **#381** (the
curl-free source-data projection of E4, **default off**: flipping the default
moves published exterior fields by up to 1.2 % and their gradients by up to
14 %, so it is the maintainer's call); **#382**; **#383**; **#384** (deletes a
pinned `nphi = ntheta = 32` the facade sizes better — the precedent H2's rule
T10 follows); **#385**; **#386**. The winding-surface PRs (#301, #303, #304) and
the HINT-comparison stack stay parked and are out of this review's scope.

## 7. Environment and runbook

- **Machines.** Apple-silicon laptops and a 36-core workstation with two RTX
  A4000 GPUs (the office workstation). Numerical comparisons use isolated environments:
  Python 3.12 with JAX 0.11.1 and VMEC++ 0.7.4 for the head, Python 3.11 with
  JAX 0.9.2 for the floor. VMEC++ wheels older than 0.7 are not references.
- **Heavy-job locks and load.** One heavy-job lock directory per machine
  (locations kept in private notes); take a lock with `mkdir` and set the release trap
  only after acquiring it. Other sessions share both machines, so a timing or
  peak-memory row counts only when the 1-minute load at its start and end is at
  most 20 on the laptop (14 cores) or 24 on the office workstation (36 cores);
  counts, identities and targets count at any load.
  On the office workstation only A/B timing rows take the heavy-job lock.
  Whole-example runs longer than twenty minutes take one of two slots,
  two long-job lock directories, with four threads, and report
  wall time to the minute. Untimed work (seed evaluations, counts,
  identities, focused tests) takes no lock while the load is at most 24 and
  at least 16 GB is free.
- **External references.** VMEC++ 0.7.4 wheel; DESC 0.17.x in its own
  environment; simsopt for QS metrics; ESSOS for coils; virtual_casing_jax 0.0.5.
- **Reproduction** (from the checkout, float64):

```bash
PYTHONPATH=. python benchmarks/review_20260913_equilibrium.py \
  examples/data/input.LandremanPaul2021_QA_lowres /tmp/eq.json
PYTHONPATH=. python benchmarks/review_20260913_exterior.py /tmp/wout_QA_lowres.nc /tmp/vc.json
PYTHONPATH=. python benchmarks/optimization.py --case qi --max-mode 1 --optimizer none --nfev 2
python tools/preflight.py --static
```

- **Cross-version tolerance.** Solved quantities differ between JAX 0.9.2 and 0.11.1 by about 6e-8 relative on the seed deck (A1), consistent with its near-6e12 conditioning (B1): compare solved quantities across versions at that tolerance, not bit for bit.
- **Discipline.** Never benchmark a shared checkout; pin the SHA and alternate
  A/B runs. Every artifact carries a `_provenance` block. A run that ends on an
  exit code without an observation is not a result. Verified-exterior targets
  only: a bean-shaped section puts normal offsets of half a minor radius back
  inside the plasma. A vacuum deck cannot test a Taylor continuation's
  truncation term; use a finite-β self-convergence for that. Push and base
  retargets within the same minute start two CI runs; wait on the survivor.

## 8. Agent briefs

> **Superseded 2026-09-21.** The briefs below are completed or replaced by Part I lanes; they remain as the record of what each PR was asked to do and why. The coordination rules are replaced by Part I "Execution and acceptance": in particular, earlier blanket merge authorization is superseded, and every merge needs explicit maintainer approval.

Each brief is self-contained: an agent reads §0–§1, its phase table in §4,
the coordination rules below and its own brief, and starts. Line numbers are
at `f09288b3`; confirm them before editing.

### Coordination rules (every brief)

- One brief, one branch, one draft PR against `main`, opened from a fresh
  worktree, never from a shared checkout. Branches: `a1/optimization-counters`,
  `a2/exterior-field-oracles`, `a3/single-stage-honest`, `a4/documentation-truth`.
- Edit only the files your brief owns. A needed change elsewhere goes in your
  report, not in your branch. No agent edits `plan.md`; the coordinator writes
  the logbook entry when a PR merges. Prefer extending existing test files; if
  `tests/manifest.json` needs a line, edit that line as text (re-serializing
  the file reformats every record and conflicts with every other branch).
- Commits and PR bodies carry the maintainer's authorship and no tool
  attribution.
- Heavy work (over two minutes or 2 GB: example runs, benchmark rows, broad
  test selections) runs one job at a time on a shared machine: take the
  machine's lock, four threads, fresh processes, `VMEX_COMPILATION_CACHE=disabled`
  for any timing, and record commit, versions, threads and load with the
  result. Short jobs (a strict Sphinx build, focused tests) do not take the
  lock, so a long record run never blocks another agent's push. When other
  sessions load the machine, timings are diagnostic: alternate baseline and
  candidate in fresh processes and gate on counts, identities and targets.
- Before pushing: `python tools/preflight.py --static`; the focused tests of
  the brief; `python tools/test_manifest.py check` when tests are added;
  `python tools/render_benchmark_index.py --check` when a benchmark artifact is
  added; strict Sphinx (`python -m sphinx -W -j 1 -b html docs <tmp>`) when
  documentation changes. Changed executable lines are at least 95 % covered;
  CI's diff-cover is the measurement (`pytest --cov` aborts on macOS), and
  `vmex/core/virtual_casing.py` is excluded from coverage by `pyproject.toml`.
- A result is an observation with units, not an exit code; a failed or capped
  run is reported as one. Do not change a tolerance to make a test pass.
- Changes are concise and deliberate: prefer net-negative diffs, add no option
  that does not remove a special case, and give every performance PR an A1
  counter row before and after (the benchmark row, not a component timing).
  New code paths must run under `jax.jit` in a test; a path that only works
  eagerly is not done.
- PR body: problem, change, evidence (numbers, commands, environment),
  limitations, what the reviewer should check. Draft. No merge without
  explicit maintainer approval.
- Report back: PR number and head commit, verification results, deviations
  from the brief, open questions.

### A1. Optimization counters

**Why.** No record splits an optimization evaluation into descent,
refinement, Jacobian, adjoint and compilation; every gate of Phases B–D is a
before/after row of those parts. Today's record gives build 11.4 s and first
derivative 61 s with nothing in between (§2).

**Existing seams.** `_SOLVE_STATS[cfg] = {"solves", "iterations"}`
(`implicit.py:1269, 1346`) is surfaced as `result.solve_stats`
(`optimize.py:3569`) and `diagnostics["solve_stats"]` (`problem.py:1021`);
`OptimizationRecord` carries `equilibrium_solves` and `rejected_trials`
(`monitoring.py:113–172`); refinement is `_refined_state` and
`_refine_step_core` with GCROT (`implicit.py:1403–1473`) behind the memo
`_LAST_REFINED` (`1288, 1374–1390`); the adjoint solves are
`_adjoint_solve`, `_adjoint_solve_gcrot` and `_adjoint_gcrot_core`
(`2000–2095`, `~2674`); the block Jacobian is `_raw_block_system` with its
per-column certifier (`2259–2470`; note the placeholder
`iterations=jnp.ones(...)` at 2468). `tests/test_trace_budgets.py:141–190`
shows how to count compilations with `jax_log_compiles`.

**Change.** Extend the per-configuration stats into cumulative counters:
solves, descent iterations, refinement calls, steps and matvecs, Jacobian
calls and columns, certifier iterations, adjoint calls and matvecs, and host
wall time spent in solve, refinement, Jacobian and adjoint. Read them from
values the host already receives: no new device-to-host synchronization inside
compiled code and no change to any numerical path. Expose them through the
existing `solve_stats` channel and one `counters` mapping on
`OptimizationRecord` (empty when unavailable, never zero-as-unknown).
`benchmarks/optimization.py` records the counters with build time, first
derivative time, and compile count and seconds; compile counting stays in the
benchmark, not the library.

**Gate.** Residual and Jacobian at `x0` are bit-identical with and without
the change on `--case qa --max-mode 1`; the QA and QI rows of
`benchmarks/optimization.py` show parts that sum to within 10 % of the
measured wall time; tests cover the eager and staged paths and a rejected
trial. Out of scope: any change to refinement, batching or defaults.

**Owns.** `vmex/core/monitoring.py`, `vmex/core/implicit.py` (counter lines
only), `vmex/core/optimize.py` (counter lines only), `vmex/core/problem.py`
(diagnostics only), `benchmarks/optimization.py`, and the tests that exercise
`solve_stats` (`grep -rn solve_stats tests/`).

**Commands.**
`PYTHONPATH=. python benchmarks/optimization.py --case qa --max-mode 1 --optimizer none --nfev 2`,
the same with `--case qi`, and the focused tests found by the grep above.

### A2. Exterior-field oracles and loud failure

**Why.** §2's exterior table: the default grid is 12–28 % wrong at 0.1–0.2
minor radii and says nothing; no test compares the exterior field with an
oracle.

**Facts.** The direct path is `VirtualCasingExteriorField.B_plasma_xyz` →
`_call_vc_B` → `compute_internal_B_offsurf_schedule` in `virtual_casing_jax`
(`exterior_field.py:388–418`), which calls
`computeB_offsurface_adaptive_schedule` (`integrals.py`, end of the function):
it computes the self-test error `err_best` and returns only `B_best`. The
package is `uwplasma/virtual_casing_jax` 0.0.5; VMEX requires
`virtual-casing-jax>=0.0.5` in the `freeb` extra while
`docs/explanation/nestor-vacuum.rst:132` still says 0.0.4. `from_wout`
defaults to 32×32 with one doubling (`extender.py:946, 1053–1056`). The tests
are in `tests/test_virtual_casing_physics.py` (lane `pr-parity-a1`, skipped
without the package); its `_synthetic_surface` helper already builds a
circular torus carrying a purely toroidal field. The existing field test
evaluates 0.5 m from a 12×12 torus (`146–209`).

**Change.**
1. Asset-free oracles. On the synthetic torus with `B = B0 R0/R φ̂` (its source
   current lies on the z-axis, outside the surface) the internal-branch plasma
   field is zero at exterior targets and minus the applied field at interior
   targets. Assert agreement to the requested digits at `d ≥ 3h` and assert
   that the error estimate of item 2 flags targets at `d < h`. Optional
   finite-current oracle: a circular filament on the magnetic axis, checked
   against an independent Biot–Savart evaluation.
2. Achieved error. Make the attained accuracy observable on the public path:
   surface the schedule's `err` (a small change in `virtual_casing_jax`, then
   read here) or estimate it in `extender.py` from the last two schedule
   levels. The eager `VmecExtender.B` warns, or raises under a strict flag,
   when the estimate exceeds `10^-digits`; traced calls expose the estimate.
   State the choice and its cost in the PR.
3. `nestor-vacuum.rst`: the rule `d ≳ 2h` with `h` the finest level's full-torus toroidal spacing `2πR/n_t` (`n_t = 2·nphi` by default, independent of nfp), the default grid, the measured table
   (reproduced by the new test or `benchmarks/review_20260913_exterior.py`),
   the version fix, and when to use the near-surface continuation (below
   about 0.2 a) instead of the direct path (above about 0.5 a).
4. `examples/vmex_get_B_outside_plasma.py` evaluates 0.03 m outside on 12×12
   with `digits=4`: choose settings that pass the new check and print the
   estimate.

**Gate.** New tests pass in under a minute each; the example prints an
estimate below its requested tolerance; returned fields are unchanged wherever
the check passes.

**Owns.** `tests/test_virtual_casing_physics.py`, `vmex/core/extender.py`,
`vmex/core/virtual_casing.py`, `docs/explanation/nestor-vacuum.rst`,
`examples/vmex_get_B_outside_plasma.py`. README wording belongs to A4.

### A3. An honest single-stage example

**Facts.** `examples/optimization/single_stage_optimization.py` seeds
`input.minimal_seed_nfp2` with a 0.02 (1,1) perturbation (`84–89`), a mean ι
near 0.08 against `IOTA_FLOOR = 0.42` (`43`); ι is a hinge with weight 100
(`99–109`) while B·n carries 1e3 and 2e5 (`56–59`); `METHOD = "BFGS"` (`72`);
each trial makes two host callbacks (`212–213`). Smoke mode
(`VMEX_EXAMPLES_CI=1`, ESSOS with #58) took 224 s and 3.9 GB and ended at
ι 0.071, aspect 10.2 and B·n RMS 3.1 %. The free-boundary example does not jit
its objective (`single_stage_free_boundary_optimization.py:145`).
`tests/test_examples.py` asserts literal strings in both scripts (`290–300`)
and runs the free-boundary one under `full` (`383–398`). ESSOS ships an
augmented Lagrangian (`essos/augmented_lagrangian.py`: `eq`, `ineq`,
`combine`) used by its coil examples.

**Change.** A seed whose solved mean ι is at least 0.3 (measure it;
`input.minimal_seed_nfp2_target_helicity` or a larger rotating-ellipse
amplitude are candidates); the ι floor and aspect as constraints (SciPy
`trust-constr` or SLSQP with their implicit gradients, or the ESSOS augmented
Lagrangian), not hinges; one host callback per trial; the free-boundary
objective jitted; the QS residual normalized by ι only if the constrained run
still collapses ι, and then say so. Keep the final target report and make a
missed target a non-zero exit.

**Gate.** A full (non-smoke) run meets the ι, aspect and B·n targets within
a stated budget, or the PR reports the attained values and why; smoke mode is
no slower than 224 s; a committed profile record for both examples (wall
time, trials, solves, peak memory, final targets) written by a committed
script and indexed.

**Owns.** The two example scripts, their assertions in
`tests/test_examples.py`, and the new record and script. No library code:
library needs go in the report.

### A4. Documentation that matches the records

**Claims without a record, or contradicted by one.**
- `docs/reference/performance.rst:365–373` and
  `docs/reference/objectives.rst:268`: QA in 14.5 min, QI 25× in 17.3 min, 33×,
  3.7× fewer iterations; no committed record, and
  `docs/_static/figures/figures.json` says the gradient-stack numbers were
  typed into the generator.
- `docs/explanation/adjoint-gradients.md:189–197`: 20.35 s to 0.61 s, 23,685
  to 6,364 iterations.
- `docs/howto/parameter-scans.md:3–5`: warm restarts converge in about one
  iteration; §2 measures 212–391 iterations for boundary moves of 1e-4–1e-2.
- `README.md:19`: CPU or GPU execution, without the optimization caveat that
  `docs/howto/run-on-gpu.md:66–70` states.
- `README.md:142–143, 219–222`: the exterior field with no accuracy
  qualification ("away from source surfaces" is undefined).
- `README.md:145–149`: the single-stage example "optimizes the plasma boundary
  and the coils"; the shipped run misses its own targets.
- `docs/_static/figures/readme_extender_exterior_islands.webp` is cited by
  nothing.

**Change.** Every number either gains a committed record and generator or is
removed; qualitative statements replace unrecorded quantities; the README
qualifies GPU optimization and the exterior field (pointing to
`nestor-vacuum.rst`) and states that the single-stage example reports whether
it met its targets; the orphaned figure is cited with its distance limit or
retired. README stays at or under 300 lines.

**Gate.** `python tools/check_docs_prose.py`, `tests/test_cited_paths.py`,
`tests/test_performance_docs.py`, strict Sphinx.

**Owns.** `README.md`, `docs/reference/performance.rst`,
`docs/reference/objectives.rst`, `docs/explanation/adjoint-gradients.md`,
`docs/howto/parameter-scans.md`, `docs/howto/run-on-gpu.md`,
`docs/_static/figures/figures.json`.

### B1. Newton finish (starts on top of #310; rebase when it merges)

**Facts.** `_newton_step` (`solver.py:1088–1139`) is reachable only through
`prec2d`, which no optimizer path configures. `_refined_state`
(`implicit.py:1451`) runs after the descent on every trial, including
value-only trials (`optimize.py:3122` → `implicit.py:1635`). On the public QA
case refinement costs 14–20 s per evaluation against 0.4–0.9 s without it and
changes the return drift from 1e-7 to 2e-7 (§2). #302 and #306 carry the
contract to keep: derivatives only at a state with a fresh projected residual,
raw FSQ and admissible geometry, and caches keyed by state identity.

**Measured (A1 counters, [`benchmarks/optimization_counters_20260913.json`](https://github.com/uwplasma/vmex/blob/07a47279d5cea819bb23c5329fab7f14cced2456/benchmarks/optimization_counters_20260913.json); shared laptop under other sessions' load, so seconds are diagnostic).** On the 8-dof QA and QI rows of
`benchmarks/optimization.py`, one evaluation's first derivative spends 29.4 s
(QA) and 40.0 s (QI) in refinement, against 2.8–3.2 s for the 185-iteration
solve: all three refinement steps exhaust their GCROT budget (`m = 100` ×
`_REFINE_MAX_RESTARTS = 20` = 2,000 iterations each, 6,000 per evaluation)
without reaching the 1e-6 forcing term. `_refine_step_core` is best-effort by
design (`implicit.py:1403–1430`: convergence "is never enforced here") and
`_refined_state` keeps the lowest-residual iterate, so a capped step is
applied silently. Refinement today is a budget, not a solve.

**Measured premise failure (B1, 2026-09-13; record to be committed with the
B1 script).** On the seed deck behind the A1 rows (`input.minimal_seed_nfp2`,
mpol = ntor = 5, ns = 31, ftol 1e-12) the raw force Jacobian on the 4,352
active degrees of freedom has σ_max ≈ 9.0e3 and σ_min ≈ 1.4e-9 (condition
number ≈ 6e12); its near-null right singular vectors are edge-localized λ
(`L_sin`) modes, (m, n) = (1, 0), (4, 0) and (0, 1). No arm certifies
|F| ≤ 1e-10: Newton with the 1-D preconditioner stalls at 1,000 matvecs per
step; Newton preconditioned by the exact block factorization (one GMRES
iteration to 7e-11) takes a step of 6e-2 along those modes against a true gap
of 1.3e-4 and stalls; a Levenberg–Marquardt shift floors at 6.8e-8; Anderson
acceleration reaches the deck tolerance with 7–11× fewer force evaluations
from `fsq` = 1e-8 to 1e-10 but certifies nothing. Today's refinement runs 3 ×
2,000 GCROT iterations (34.6 s) and returns the descent state unchanged. The
achievable floor is |F| ≈ 1e-7, which the descent already reaches.

**Decisions.** (B1a) Stop refinement when a step's inner solve misses its
forcing term and the step does not lower |F| — an unconverged, non-improving
inner solve is not a Newton direction; results are identical where refinement
works and 6,000 → 2,000 GCROT iterations on the A1 rows. (B1b) Before any
further finish or certificate work, determine whether the near-null λ modes
are a discrete gauge of the equations: if the residual is invariant along them
to first order, deflate them from the Newton step, the adjoint and the
certificate, and test whether the projected residual then certifies 1e-10 and
whether objectives and gradients are insensitive to them. #302's and #306's
1e-10 primal certificate would reject every state on this deck until B1b
answers this.

**B1b answered on the seed deck (B1 measurement, record pending).** The
near-null modes are not a gauge. They are volume λ (`L_sin`) modes weighted
toward the edge: 4–17 % of each vector's norm is in the edge row. Along most
of them the raw residual changes as ε² — flat to first order, with finite
curvature. The spectrum has a two-mode cluster (σ = 1.4e-9 and 1.9e-9) above
a continuum starting near 2e-8. The certificate floor is intrinsic at this
resolution: 21–25 % of |P·gc| lies on the preconditioned images of those
modes, and removing it needs O(0.1) moves along them. A block Newton step on
the complement of the near-null space reaches a projected residual of 1.1e-10
in two steps at one or two matvecs per step. It moves the state 5e-4 to 5e-3,
though, changing the QI value by up to 3.6e-4 and its gradient by up to
4e-3; QA barely notices (value 3e-8, gradient 3e-4). The best full
certificate any arm reached is 6.6e-9. So QI derivatives on this deck are
meaningful to about 1e-3, and a primal certificate at 1e-10 rejects every
state. The block-tridiagonal tangent is exact at the state (solve defect
6e-15), while the QI adjoint's GCROT stalls at 2.6e-5 relative from 2,500 to
6,000 iterations. It is accepted only through a tenfold slack, after 13,228
iterations and 78 s.

**QA_lowres reverses part of the seed-deck picture (B1 measurement).** On
`input.LandremanPaul2021_QA_lowres` (MPOL = NTOR = 8, ns = 50) the raw
Jacobian's condition number is about 2e12 with no spectral gap; its near-null
modes are m = 0 λ modes with linear response, which the 1-D preconditioner
restores, carrying 4 % of the certificate residual. Main's refinement
certifies there (6.34e-7 → 5.66e-12 in 5,343 GCROT iterations, through an
unconverged first step whose linear residual still falls to 5.2e-5), and
Newton preconditioned by the block factorization, switched in at
`fsq` ≈ 1e-6, certifies 9e-13 in four full steps at a seventh of the work of
descent plus refinement. The seed deck is the pathological case: its first
step stalls at 4.2e-3 and nothing certifies. Decisions: #320 stops only when
the unconverged inner solve gained fewer than three digits (threshold 1e-3,
one named constant, two measured decks) and the step made no progress; B1c,
after B3, switches to block-preconditioned Newton at `fsq` ≈ 1e-6 and falls
back to descent when two full steps fail to lower `fsq` tenfold, judging kill
rules on full steps only; the primal certificate becomes goal-oriented,
|μᵀF_raw| ≤ 1e-8·max(|m|, m_ref) from B3's exact adjoint with a fresh raw FSQ
and geometry check, keeping 1e-10 only where no block path exists, which
supersedes #302's absolute `primal_tol`. Deflating the near-null modes is
unusable: it moves the QI gradient by 23–36 %.

**Precedent and risk.** Every code that finishes a VMEC-type descent with
Krylov (VMEC2000's `PRECON_TYPE` modes, PARVMEC, SIESTA) preconditions it with
the 2-D radial block operator; SIESTA reaches a 1e-19 residual in 6–11
Hessian builds and 348–1,123 block back-solves. `_newton_step` preconditions
GMRES only through the 1-D preconditioned force (`solver.py:1093–1132`), so
its GMRES sees the conditioning that makes the descent take ~800 iterations.
VMEX already assembles and factors the exact radial block-tridiagonal raw
force Jacobian for its implicit derivatives (`_raw_block_system` and the block
Thomas factor, `implicit.py:2259–2470`): that factorization is the natural
2-D preconditioner.

**First step, before code.** With A1's counters on the QA (MPOL = NTOR = 8,
ns = 50) and QI cases, save states at `fsq` = 1e-6, 1e-8 and 1e-10. From each,
record (a) descent iterations to the deck tolerance plus the matvecs today's
refinement spends, (b) a forced `_newton_step` with the 1-D preconditioner:
Newton steps, GMRES matvecs per step, line-search evaluations, (c) the same
with the block factorization as preconditioner, refactored at most once per
trial, and (d) a control: Anderson acceleration (window 5–10) on the descent
map. Count work as `W = force evaluations + c·JVPs + f·factorizations` with
`c` and `f` the measured warm cost ratios to one force evaluation. Implement
the arm with the lowest `W` to the same certified residual only if it beats
(a); drop (d) below 1.5× fewer evaluations.

**Kill per arm.** More than 50 matvecs per Newton step at rtol 1e-3; `fsq`
falling less than 10× in two of the first three steps; a Jacobian reset; or a
final state farther from the refined state than the certificate tolerance.
If both Newton arms die, test refining only where a derivative is requested.

**Gate.** As in §4 B1; the kill rule of §4 applies. Owns `solver.py` and
`implicit.py` (A1's counter lines stay).

### B4. One compiled path: full jit without recompilation

**Facts.** On A1's rows ([`benchmarks/optimization_counters_20260913.json`](https://github.com/uwplasma/vmex/blob/07a47279d5cea819bb23c5329fab7f14cced2456/benchmarks/optimization_counters_20260913.json)) the JAX value-and-gradient lane spends 41.6 s
(QA) and 71.0 s (QI) compiling after the host derivative has already
compiled, and builds take 244–500 XLA compiles; a five-evaluation warm
campaign recorded 102 compiles (`benchmarks/baselines/m4/F8_warm.json`). The
problem jit cache key holds `x0.tobytes()` and `id(cfg)`
(`optimize.py:2628–2646`); three staged lanes take `static_argnames=("cfg",)`
(`implicit.py:1138, 1402, 2674`), so a new configuration object recompiles.
The host solve runs `mode="cli"` behind `jax.pure_callback`
(`implicit.py:1733`), while `mode="jit"` (`lax.while_loop`) costs the same
warm (§2). #307's seven lines in `solver.py` cut `_block_lane` cache misses from 11 to 3 by normalizing array commitment flags, but sit on #299's branch. Commitment flags are part of JAX's compile key, so any lane whose first call receives eagerly computed (uncommitted) arrays and whose later calls receive its own committed outputs compiles twice: B4's census found this in the staged refinement step, where it accounts for 12.5 s of a 25.8 s recompile on a `[1, 1]` `max_mode` schedule. One helper in `vmex/core/device.py`, introduced by #319 and called by #320, normalizes commitment wherever a tree shares a single device; B4b lists every other lane of this class.

**Change.** (1) Re-land #307 on `main`. (2) Key compiled lanes by content:
`x0` traced, configurations hashed by value. (3) Share the solve, refinement,
Jacobian and adjoint executables between the host and JAX lanes. (4) Make one
documented full-jit path — `jax.jit(jax.value_and_grad(loss))` with the
equilibrium solve as a `lax.while_loop` inside the implicit rule and no host
callback — the reference the host lane is tested against; SciPy's trial loop
stays on the host.

**Gate.** On A1's benchmark rows: the second lane adds under 5 s of
compilation; zero recompiles across trials and `max_mode` stages; the
full-jit value and gradient match the host lane to 1e-10 relative; warm
evaluation time no worse; compile counts pinned in
`tests/test_trace_budgets.py`.

**Owns.** `vmex/core/optimize.py` (jit keys and lane sharing), the
configuration keys in `vmex/core/implicit.py`, #307's lines in
`vmex/core/solver.py`, `tests/test_trace_budgets.py`. B1 owns refinement in
`implicit.py`; rebase on whichever merges first.

### B4c. Trace the solver's recovery path

**Facts.** #321's full-jit reference (`jax.jit(jax.value_and_grad(loss))`,
one program, no host callback) matches the host lane to 1e-10 on a small
Solov'ev problem but fails on the QA and QI `input.minimal_seed_nfp2`
problems: the traced solve stops at iteration 1 with `BAD_JACOBIAN_FLAG`,
because axis re-guess and the bounded Jacobian retries run only in the host
loop. Two smaller items remain from #321: `_block_lane` still compiles twice
(default-device context differs between construction and trial solves in
`_run_loop`), and content-keyed lanes are not done — problem closures bake
`x0`, `params0` and the penalty scale, frozen `cfg.inp` decisions make
boundary-free keys inexact, and a `[1, 1]` `max_mode` stage recompiles
8 programs in 25.8 s.

**Change.** Move axis re-guess and the bounded retries into
`lax.cond`/`lax.while_loop` with fixed shapes, gated on the same flags, so the
traced solve follows the host solve's decisions. Then remove the
`_block_lane` context split. The content-key design note is written (B4e) and the work is deferred: a `[1, 1]` stage recompiles 8 programs in 25.8 s because `_canonical_config` hashes all of `cfg.inp`, compiled lanes bake deck values, and problem closures bake `x0`, `params0` and the penalty scale; splitting structure tokens from traced values touches about 25 call sites across three owners, which ranks below B3, B4c and C1.

**Gate.** On both seed decks the full-jit value and gradient match the host
lane to 1e-10 with no host callback, iteration counts and final states are
identical to the host lane, and compile counts are pinned in
`tests/test_trace_budgets.py`; `_block_lane` compiles once per structure.

**Owns.** The recovery path in `vmex/core/solver.py` and `multigrid.py`, the
full-jit reference test, `tests/test_trace_budgets.py`.

### S1. Split #299 into focused PRs

**Facts.** #299 (`perf/reuse-objective-linearization`, head `47de5a8b`, CI
green) is 29 commits from another session, many mixing source with plan,
logbook and record edits (2,719 lines). Its source changes and their recorded
effects: residual linearization reuse and separable field-line synthesis
(warm 48-dof QI Jacobian 1.10 → 0.82 s); checked Thomas selection, parity
broadcast and divisor batching (GPU first derivative 210 → 123 s, a CUDA
transpose failure fixed); host trial solves outside GPU callbacks (a stall
removed); the Boozer λ half-mesh correction; analytic vacuum-mode contraction
and saved per-adjoint pullbacks (an NCSX transpose compile above 20 GiB
removed); plotting startup (#305, 21.6 → 12.5 s); and the optional
magnetic-only Boozer projection.

**Change.** Re-land those on `main` as at most six PRs, by file hunk rather
than whole commit, each with the tests #299 added for it: (1) Boozer λ
interpolation (`a6b13367`); (2) linearization reuse and field-line synthesis
(`ef8a25fa`); (3) Thomas selection, parity broadcast and divisor batching
(`20762a29`, `9343b4b4`, `cd667198`); (4) host trial solves outside GPU
callbacks (`f205c995`); (5) vacuum contraction and saved pullbacks (`931be2d5`,
`9950781d`), which is F1a's first step; (6) plotting (`b5dbcc95`) and the
magnetic-only projection (`b3ef8448`), which activates only with
booz_xform_jax ≥ 0.3.0. Carry over no plan, logbook, handoff or private
evidence; cite #299's record for the numbers. Never push to #299's branch.

**Gate.** Each PR reproduces #299's recorded numerical tolerances, carries an
A1 counter row where it claims speed (or is labeled correctness-only), keeps
changed-line coverage at or above 95 %, and passes CI.

**Owns.** Branches `s1/<topic>` and the files each hunk touches; rebase order
is (1), (4), (3), (2), (5), (6).

### A4b. Documentation left behind by S1

S1 carried #299's source but not its guidance edits, which cite flags and a
record that are not on `main`: validation guidance in `implicit.py`
docstrings, `docs/explanation/adjoint-gradients.md`,
`docs/reference/optimization.rst` and `docs/tutorials/first-gradient.md`
(`43d1765a`); the QI "smooth" wording in `omnigenity.py`, `confinement.rst`
and `objectives.rst` (`7c58c2ff`); the `least_squares` warm-start docstring
(`2ebe0d50`); the `equilibrium_from_x` docstring (`cd667198`). After #310 and
#313–#318 merge, re-state what is still true against the code and records on
`main`, and drop what is not. Also: the "handful of iterations" warm-restart
claims in `examples/hot_restart_scan.py:6` and
`docs/howto/restart-from-previous-run.md:77` (§2 measures 212–391 iterations);
name the free-boundary single-stage example in the README once #311 prints its
target status; and point the README's exterior-field distance rule at the
section #312 adds to `nestor-vacuum.rst`. Owns those docstrings, pages and
lines, after #309, #311, #312 and #313–#318 merge.

### B3. Exact block adjoint, in two PRs

**Facts.** On the seed deck the production GCROT adjoint takes 1,100
iterations and 9.8 s (QA) and 17,270 iterations and 50.7 s (QI), with Krylov
error 8.7e-6 and 3.2e-5 against a dense solve on range(P). The block adjoint
matches the exact raw adjoint to 4e-13 and 6e-10 in 0.66 s warm. The raw and
preconditioned formulations differ by 1.3e-6 (QA) and 1.1e-3 (QI) at this
non-root anchor, because the 1-D preconditioner is recomputed from the state.
Least-squares problems take J^T r from the block Jacobian and never reach this
adjoint; scalar losses (`from_loss`), direct `jax.grad`,
`minimize(objective_terms)` and the single-stage example do, and the
single-stage example pays one solve and one adjoint per trial (#311: 2,959 s
for 301 trials).

**Change.** B3a (`b3a/exact-block-adjoint`): the backward rule of
`solve_implicit` solves the raw adjoint by transposing the block
factorization, with GCROT kept as fallback and certifier; the CHANGELOG entry
is under Changed and states the formulation alignment. B3b (after #328): the
reverse Jacobian and `minimize(objective_terms)` share one factorization
across rows (first probe whether XLA already hoists it out of #328's chunked
map), and the value-and-gradient fallback becomes one VJP of ½|r|².

**Gate.** B3a: adjoint residual ≤ 1e-10; the jitted gradient equals the raw
block tangent to 1e-10; the Solov'ev, li383 and LASYM gradient-versus-FD tests
pass; `from_loss` QA and QI rows and eager `jax.grad` adjoint counters before
and after; single-stage per-trial seconds and adjoint iterations before and
after (smoke mode plus about ten trials, on the office workstation). B3b: one
factorization per point, with seconds, adjoint iterations and peak RSS on seed
QA. After B3a merges, B4c's QI gate re-runs at 1e-10.

**Owns.** B3a: the backward rule in `vmex/core/implicit.py` and
[`benchmarks/adjoint_formulation.py`](https://github.com/uwplasma/vmex/blob/07a47279d5cea819bb23c5329fab7f14cced2456/benchmarks/adjoint_formulation.py). B3b: the reverse lanes in
`vmex/core/optimize.py`.

### D0. A QI example that starts near QI

**Facts.** `examples/optimization/QI_optimization.py` perturbs the
circular-like `input.minimal_seed_nfp2` and runs one stage at `max_mode = 3`
with up to 250 evaluations of `ConstructedQIResidual`, Goodman's
squash-and-shuffle residual on a traceable Boozer transform
(`quasi_isodynamic_residual`, `optimize.py:969`). No full-run record exists;
§2 has only five-evaluation rows. Two near-QI decks ship but this example does
not use them: `input.QI_stel_seed_3127` (near-axis, nfp 3, order r1,
MPOL = NTOR = 5) and `input.QI_nfp2_initial` (simsopt, nfp 2, the seed of the
jaxopt and optax variants). Every published QI optimization started from a
near-axis seed.

**Change.** First record one full run of today's example on the office
workstation with A1 counters: wall time, evaluations, failed trials, final
constructed-QI total and its fine-grid validation, and the aspect, ι, mirror
and elongation targets. Then seed from the near-QI deck that measures best,
run a `[2, 3]` / `[20, 60]` mode ladder, and add Goodman's `phimin` sign rule
only if the residual needs it. D3 joins this branch only with its own
measurement: `oversample = 1` validated by the fine-grid check (the
magnetic-only projection is already used with booz_xform_jax ≥ 0.2.0).

**Gate.** The example reaches the baseline's final validation metric or
better, with every constraint target met, in at most half the baseline's wall
time and with zero failed trials; the record is committed under `benchmarks/`
with its index row; CI smoke mode is no slower.

**Owns.** `examples/optimization/QI_optimization.py`, its record, and the QI
wall-time sentence in `docs/reference/objectives.rst`. Branch
`d0/qi-near-axis-seed`.

### D1. A differentiable QI well location

**Facts.** `quasi_isodynamic_residual` (`optimize.py:969–1153`) picks each
field line's well with `jnp.argmin` (lines 1054 and 1083), builds the monotone
branches with `jnp.maximum.accumulate` and maps them with `jnp.interp` (line
1128). `argmin` has zero derivative, so the well location is frozen in every
Jacobian, and a trial that moves the minimum to another grid point changes the
residual discontinuously; the #299 record's 1.5 % drift was this sensitivity.
The omnigenity residual in `omnigenity.py` (line 555) repeats the pattern.

**Change.** Replace the hard minimum by a soft location (a softmin-weighted
periodic phase) and the running maximum by a smooth one (logsumexp with a
width tied to the grid spacing), keeping the residual's rows and weights so
that the sharp limit recovers today's values.

**Gate.** The Taylor test's first-order remainder converges in both directions
on the minimal seed and on D0's seed; at the chosen width the residual differs
from today's by at most 1e-3 relative at those seeds; D0's run reaches its
final validation metric within 5 % with zero failed trials and no more
evaluations; the new path runs under `jax.jit` in a test.

**Owns.** The two residual kernels and their tests. Starts after D0's baseline,
so it is measured on D0's example. Branch `d1/smooth-qi-well`.

### C3. Single stage as one least-squares problem

**Facts.** The fixed-boundary single-stage example (#311) runs a
Powell–Hestenes–Rockafellar augmented Lagrangian around L-BFGS-B on
`jax.value_and_grad` through the implicit solve and meets every target in
2,959 s and 301 trials (`benchmarks/single_stage_profile_m4.json`). The
collaborator run in §2 needed 63 evaluations in its plasma stage against
188 evaluations and 176 gradients for joint BFGS, which exited on precision
loss. The plasma residual already has a certified block Jacobian lane.

**Change.** One `scipy.optimize.least_squares` (trust-region reflective, with
bounds) on `[r_plasma; r_coil; constraint rows]` with Jacobian
`[J_plasma from the block lane; J_coil by jacfwd of the ESSOS terms]`,
evaluated only at accepted points; constraints stay augmented-Lagrangian
multipliers updated between least-squares stages unless bounded residual rows
meet the targets. A library helper is added only if it shortens both
single-stage examples.

**Gate.** Meets #311's targets (min |ι| ≥ 0.42, aspect ≤ 4, B·n RMS ≤ 1 %, coil
clearance and curvature limits, the independent ns = 101 check) in at most
0.4 of #311's 2,959 s on the same machine class, with trials and Jacobians
counted; smoke mode is no slower; the example stays one file. Measure after
B3a merges or state which adjoint the rows used.

**Owns.** `examples/optimization/single_stage_optimization.py`, its profile
record and any helper. Needs ESSOS with uwplasma/ESSOS#58, merged at
`1b3210ca` (its branch is deleted). Branch `c3/single-stage-least-squares`.

### F-err. A force-balance diagnostic users can read

**Facts.** Users report that the force-balance error panel always shows
errors near 100 %. The panel plots `abs(wout.equif)`
(`plotting.py`, `_relative_force_error_panel`), VMEC2000's `calc_fbal`
normalization computed in `postprocess.py::force_balance`: the radial residual
divided by the sum of the magnitudes of its current and pressure terms. By the
triangle inequality that ratio cannot exceed 1, and with no pressure gradient
(vacuum, which most shipped optimization examples are) it tends to 1 wherever
the current terms do not cancel. #280 found the same saturation in the polish
certificate and added non-saturating measures there (⟨|F|⟩/⟨|∇p|⟩, the DESC-style
|F|/⟨|∇(B²/2μ₀)|⟩, dimensional ⟨|F|⟩); the plot and the summary's "max ε_F"
still use the saturating ratio. Whether `equif` is also wrong on converged
finite-β cases is not yet known.

**Change.** Establish with evidence whether this is a bug, a misleading
normalization, or both: `equif` against VMEC2000 golden WOUTs; converged vacuum,
converged finite-β and deliberately unconverged states; DESC's normalized
force error on the same equilibrium. Keep `wout.equif` for file compatibility.
Plot and summarize a vacuum-safe, non-saturating measure that reuses the #280
functions, with the normalization in the axis label. Fix `equif` only if the
golden comparison shows a bug.

**Gate.** The panel metric is small on converged vacuum and finite-β decks,
large on an unconverged state, and agrees with DESC's normalized force error
within a stated tolerance where DESC is available; golden `equif` parity is
unchanged; tests run in a PR lane.

**Owns.** The force panel and summary in `vmex/core/plotting.py`, a
`force_balance` fix only if the golden comparison demands it, and their tests.
Branch `ferr/force-balance-diagnostic`.

### Literature checks left open (optional)

- **L1, behind B1 and B5: done 2026-09-13.** Folded into the B1 brief (block
  preconditioner arm, work metric, kill rules) and the B5 row (thread scaling
  first, dispatch before sharding). Unverified and still open: published
  iteration savings of VMEC2000's `PRECON_TYPE='GMRES'`, VMEC++'s
  single-thread time on this deck, and whether its wheel is built with FFTX.
- **L2, behind F: done 2026-09-13.** Folded into Phase F (F0 measurement,
  F1a/F1b split, F2 as fallback, F3 as a tolerance rather than a frozen
  derivative). Unverified and still open: DESC's free-boundary cost relative to
  fixed boundary, VMEC++'s free-boundary hot-restart iteration counts, and the
  NESTOR cost split (F0 measures it).

## 9. Execution logbook

> **Superseded 2026-09-21.** Historical. New entries go in the Part I continuation logbook.

Format: date, PR (base and head), gate, command and environment, result with
units, limitation, next action. The 2026-09-05 to 2026-09-08 entries (Phase 1
and Phase 2 of the previous plan: #284–#298) remain at `f09288b3`.

**2026-09-13, review.** Independent review of `f09288b3` and #299 (`de93e5e2`),
five read-only source and literature audits, and the measurements of §2
(`benchmarks/review_20260913.json`). Findings: the anchor, not the Jacobian,
dominates optimization cost; the objective is path dependent; `minimize()` runs
one adjoint per residual row; the exterior field's off-surface path is an
unconverged trapezoid rule with no oracle test; VMEC++ 0.7.4 is 2.6× faster per
iteration and 12× faster cold, but at parity inside a hot-restarted loop. The
2026-09-06 plan is reordered around those findings; §5 keeps its force-balance
verdicts. Limitations: single-run laptop timings; the record has no stage split for the
QI example (the A1 gap). Next action: Phase A, starting with A1.

**2026-09-13, third pass and handoff.** Added to the same record: the public
objective-replay test (refinement gives no reproducibility gain on the QA case
and costs about twenty solves per evaluation; the #299 QI drift is objective
sensitivity), the Jacobian batching probe (0.5–2 % batch dependence, to be
reproduced in isolation before C1), the shipped single-stage example in smoke
mode (224 s, ends at ι 0.07 and aspect 10.2), and the VMEC++ 0.7.4 autodiff
probe (no derivatives in the wheel). Literature and code review of the QI
objective moved a near-axis seed to the front of Phase D (D0) and documented
the frozen well location in the residual; #306 and #307 were reviewed and
dispositioned in §6. Paused before two checks finished: the precedents for
Newton finishing and CPU parallelism behind B5, and the free-boundary
derivative practice behind F1–F3 (DESC's free-boundary Jacobian, VMEC++'s
NESTOR hot restart, the adjoint literature); both are literature checks, not
gates, and B5 and F keep their kill rules. CI on this revision: every lane
green except the two manifest parity lanes still running at the pause. Raw
material for the next session: the eight review reports and every script and
log under the reviewer's `vmex-review-evidence` directory, and the worktree
`vmex-review-main` on this branch. Next action unchanged: Phase A, A1 first,
then A2 with the vacuum and interior identities as the exterior-field oracles.

**2026-09-13, reconvened.** #308 green on every lane including the PR gate;
main unchanged at `f09288b3`. Agent briefs for A1–A4, B1, B4a and the two open
literature checks added as §8, so Phase A executes from this document alone.
Phase A starts with A1–A4 in parallel on disjoint files; the office
workstation is unavailable for heavy runs (disk full), so heavy local jobs are
serialized.

**2026-09-13, L1 returned.** Literature check L1 (read-only): every Krylov
finish of a VMEC-type descent uses a 2-D block preconditioner, while
`_newton_step` has only the 1-D one; B1 gains a block-factorization arm, a
work metric and per-arm kill rules. Anderson or nonlinear-GMRES acceleration
has no evidence above 1.3–2× for this problem class and stays a control arm.
For B5, doing less per iteration (loop trips, thunk count, batched kernels)
is more plausible than host-device sharding. §2 now states the timing deck's
resolution (MPOL = NTOR = 8).

**2026-09-13, L2 returned.** Literature check L2 (read-only): no other code
has a free-boundary equilibrium adjoint (VMEC++'s excludes free boundary;
DESC solves free boundary as an outer least-squares problem; Paul et al. 2020
use perturbed nonlinear equilibria at 4e-2 accuracy; SPEC, STELLOPT and
simsopt use finite differences, simsopt resetting its axis guess to fight the
same history dependence). Phase F is rewritten: measure NESTOR's cost split
first, hoist the linearization, build the edge response matrix once per
gradient, keep previous-iterate Schur preconditioning as the fallback, and
replace the frozen-derivative option, which drops an order-one term, by a
certified looser tolerance.

**2026-09-13, A1 measured.** Draft #310 (A1, head `142d6c92`) passes its
bit-identity gate on JAX 0.9.2 and 0.11.1 and attributes 96.7 % (QA, 107.8 s)
and 95.4 % (QI, 164.9 s) of benchmark wall time. Its rows re-rank Phase B: the
first derivative is refinement (29–40 s, every GCROT step capped at 2,000
iterations without converging) plus block Jacobian (27–32 s, certifier idle),
with the solve at 3 s; the JAX lane then compiles for another 42–71 s. B1, B3,
B4 and C1 are updated in place. Timings are diagnostic (load 18–31 from other
sessions); the record is to be committed with #310 before it merges.

**2026-09-13, A3 decisions.** Draft #311 (A3) seeds the example at mean ι
0.408 and aspect 4.03 and constrains ι, aspect and B·n RMS with an augmented
Lagrangian around L-BFGS-B (one solve and one scalar adjoint per trial). Two
coordinator decisions: the shipped coil seed (0.5 m, length target 3.3 m) was
geometrically inconsistent with aspect 4 and ι 0.42 under the 0.20 m clearance
limit, so the coils move to 0.65 m and 4.1 m while every physics target stays;
the free-boundary example keeps its ι hinge until the new F-pre item makes its
pullback jittable. The coordination rules now keep short jobs out of the heavy
lock.

**2026-09-13, merge list and speed work.** Phase A PRs #309–#312 are open and
in CI. §6 now lists the merge decisions across vmex, booz_xform_jax, SOLVAX
and virtual_casing_jax, and a 0.9.0 release candidate that is explicit about
what it does not fix. The speed work starts from §8: B1 (refinement), B4
(one compiled full-jit path without recompilation) and S1 (#299's source
re-landed as focused PRs). The coordination rules now require net-negative or
minimal diffs, a counter row for every performance claim, and jit coverage for
every new path.

**2026-09-13, S1 done.** #299's source is re-landed as #313–#318 against
`main` (six pieces, 12–114 net lines, preflight and focused tests passing
locally, CI queued). Speed claims cite #299's records until #310's counters are
on `main`; #315 is correctness-only; #316's NCSX gradient check was not rerun
here. #299's guidance edits that cite records not on `main` become A4b.

**2026-09-13, B1 premise fails.** B1's measurement on the seed deck: condition
number ≈ 6e12 from edge-localized λ modes, no Newton arm certifies 1e-10,
today's refinement is a 34.6 s no-op, and the achievable floor (|F| ≈ 1e-7)
is what the descent already reaches. Decisions recorded in the B1 brief: B1a
exits refinement on an unconverged, non-improving step (6,000 → 2,000 GCROT
iterations, identical results); B1b asks whether those modes are a discrete
gauge to deflate before any certificate work; #302 and #306 stay held. The
MPOL = NTOR = 8 QA deck is being measured next to see whether resolution
changes the conditioning.

**2026-09-13, A4 and B4a reported.** #309 (A4) traces every changed claim to a
test or record, retires the typed-number gradient-stack figure, and corrects
the QI anchors to what `tests/test_omnigenity.py` asserts (at least 20×, not
36× and 138×). Its three open questions go to A4b. #319 (B4a) supersedes #307:
the extra compile came from the donation copy seeing a partly committed carry,
and normalizing before the copy restores the ladder's compile counts.

**2026-09-13, A1 and A2 reported.** #310 (A1) keeps residual and Jacobian
byte-identical on both JAX versions, explains 96.7 % and 95.4 % of the QA and
QI benchmark rows, and records adjoint counts as unknown whenever the adjoint
is compiled (B3's first step). #312 (A2) adds asset-free oracles (both
identities and on-surface parity to 1e-4 at 3h) and a calibrated achieved-error
estimate with no false pass in 816 targets, where the schedule's own self-test
passed errors up to 0.30; eager calls warn by default. Its findings add E0 and
correct the spacing rule; the exterior-field facade forwarding is a small
follow-up.

**2026-09-13, B1a opened and a recompile class named.** #320 (B1a) stops
refinement after an unconverged, non-improving step. B4's census traced
refinement's duplicate compile to commitment flags: the first step's
arguments are computed eagerly and uncommitted, later steps receive committed
outputs, and commitment is part of the compile key — the mechanism #319 fixes
in the solver. One helper in `vmex/core/device.py` (#319) now serves both
call sites (#320), removing 12.5 s of a 25.8 s `max_mode` stage recompile with
bit-identical results. Merge order: #310 → #319 → #320 → B4b.

**2026-09-13, A3 reported.** #311 (A3): the fixed-boundary single-stage
example now meets every stated target on a full run (min |ι| 0.4277, aspect
3.979, B·n RMS 0.80 %, coil clearances and curvature within limits, converged
ns = 101 check), using a rotating-ellipse seed at ι 0.408 and an augmented
Lagrangian around L-BFGS-B with one solve and one adjoint per trial. Smoke mode
is faster than main (136 s against 195 s). Quasisymmetry worsened slightly
under the constraints; that and the wrapper's home go to C3. The free-boundary
example waits for F-pre.

**2026-09-13, first merges.** On maintainer authorization to admin-merge
finalized, concise PRs with every real CI lane green: booz_xform_jax #8
(`8e0208ae`), SOLVAX #105 (`60a87b21`, untagged), vmex #309 (A4, `1aa5465e`)
and #310 (A1, `2b9d3a3e`); this plan (#308) follows, verified locally because
runners are saturated by another project. #319 is retargeted to `main`. B1
confirms #302's `primal_tol` would make every trial value-only on the seed
deck. #314 fails the QI regression pin under jit and is being investigated
before any further S1 merge. Package tags and PyPI releases stay with the
maintainer.

**2026-09-13, evening: merges, B1a, B4a–B4b and a test defect.** Merged under
maintainer authorization: booz_xform_jax #8 (`8e0208ae`), SOLVAX #105
(`60a87b21`, untagged), vmex #309 (`1aa5465e`), #310 (`2b9d3a3e`) and this plan
(#308, `4a171430`). Ready behind CI: #300, #311 (refreshed), #312, #313, #319.
Measured: B1a (#320) cuts QI refinement 24.8 → 12.0 s and the first derivative
58.9 → 40.2 s with bit-identical outputs; B4b (#321) removes the JAX lane's
recompilation (41.6 → 0.10 s QA, 71.0 → 0.12 s QI) and halves benchmark wall
time with bit-identical host values. B4's census named commitment flags as a
recompile class and one helper fixes it in the solver, the predictor, the
constraint baselines and refinement. #314's CI failure is a tied QI
regression pin already on `main`, fixed by a test-only PR before the S1
pieces merge. The full-jit reference fails on the seed decks because the
solver's recovery path is host-only; B4c takes that next. CI runners are
shared with another project, so queued runs on stacked PRs were cancelled
until their bases merge.

**2026-09-13, B1b answered on the seed deck.** The near-null λ modes are
soft physical modes, not a gauge, and they set a certificate floor near 1e-8.
QI objectives respond to them at about 1e-3, QA's at about 1e-5, so QI
derivative accuracy on this deck is bounded near 1e-3 whatever the
refinement. The block tangent is exact at the state, while the QI adjoint's
Krylov solve stalls and takes 78 s; B3 now starts from an exact adjoint
through the same block factorization. The committed record, the
MPOL = NTOR = 8 deck and the adjoint arm are still being measured.

**2026-09-13, paused.** Stopped on maintainer request; no agent, watch or
heavy job is running and the heavy-job lock is released. State to resume from:

- Awaiting CI, then admin-merge when every real lane is green: #300 (a
  re-run Python 3.12 fast lane), #311, #312, #313, #319 (build-compile
  explanation accepted), #322 (this logbook) and #323 (S1's test-only fix
  for the tied QI pin: golden `wout_li383_low_res`, minimum-gap guard 2.6e-4,
  pinned 0.3579844756860454 on both JAX versions, no solve and no shared
  cache).
- After #323 and #313: refresh #314–#318 against `main` and merge them in
  the order #315, #314, #318, #316, #317.
- After #319: retarget #320 to `main`; its re-verification at `9b77a10e`
  (bit-identity and compile counts against `f0b12ed2`) was queued and not
  run. Then rebase #321 onto `main`.
- #321 (B4b) still needs `jax.Array.committed` in place of `leaf._committed`,
  a named test that runs the traced value-and-gradient program against the
  host pair, and a CHANGELOG line for the eager `jax_value_and_grad`
  behaviour change. B4c starts with passing `use_fft` in the Jacobian retry
  (`solver.py` near line 2410), then traces the recovery path; B4d and the
  B4e design note follow.
- B1: the MPOL = NTOR = 8 QA deck run was stopped before it finished and has
  no record; the adjoint-through-block-factorization arm (f) and the
  refinement stagnation-abort arm (g) were not run. The seed-deck results
  above are interim until [`benchmarks/newton_finish_arms_20260913.json`](https://github.com/uwplasma/vmex/blob/07a47279d5cea819bb23c5329fab7f14cced2456/benchmarks/newton_finish_arms_20260913.json) is
  committed from branch `b1/newton-finish`.
- Five other tests in `tests/test_optimize.py` still read the shared `/tmp`
  Solov'ev state (a test-isolation follow-up). No package was tagged.

**2026-09-14, resumed: merges and decisions.** Merged: #323 (`b0646713`),
#313 (`373f1e83`), #314 (`746215d3`), #312 (`68a119e9`), #322 (`08d92161`),
#311 (`0c083539`) and #319 (`c5ee2e0d`); #307 closed. #320, #321 and #324 are
retargeted to `main`. B1's two-deck record decides four things: the exact
block adjoint is B3's implementation and aligns scalar-loss gradients with
the Jacobian lane's raw formulation (corrected in the afternoon entry); #320 keeps refinement's certification on well-resolved decks
through a linear-progress guard; the primal certificate becomes goal-oriented
after B3; and B1c switches to block-preconditioned Newton inside the descent.
B4e's content-keyed lanes are deferred with a written design. The heavy-job
lock and CI capacity remain the pacing constraints.

**2026-09-14, paused again.** Stopped on maintainer request; no agent, watch
or heavy job runs on the laptop or the office workstation, and both heavy-job
locks are released. The office worktrees remain for
reuse. State to resume from:

- Awaiting CI, then admin-merge when every real lane is green: #300 (its
  Python 3.12 fast lane was re-run), #315, #316 (the CTH free-boundary
  gradient check passed locally), #317 and #324, plus this logbook (#327).
  #317's and #318's Python 3.12 fast lane hit the 8-minute cap under runner
  contention (main's run takes 6:18); re-run it, and trim #317's new
  parametrized plotting tests if it times out again.
- #318 does not merge on its cold 8-dof rows, which show no gain beyond
  run-to-run spread. It needs one warm 48-dof QI measurement on the office box
  (`--max-mode 3 --optimizer least_squares --nfev 3`, main vs #318,
  alternating): merge if the warm Jacobian time or peak memory improves beyond
  spread, otherwise close it with its source kept in #299's history. The job
  was killed before it produced rows.
- **Open regression to bisect first:** on the office workstation (JAX 0.9.2,
  CPU), `problem.jax_value_and_grad` on `main` (`c5ee2e0d`) requests a
  251 GB allocation in `jit(residual_value_and_gradient)` for both the QA and
  QI benchmark rows; JAX 0.11.1 on the laptop runs the same lane. #321 avoids
  it for concrete calls, but traced calls still build that program. Bisect
  `373f1e83` against `746215d3` (#314's automatic Jacobian batch width is the
  first suspect), then fix minimally in its own PR, or report it as JAX 0.9.2
  behaviour on older commits for a floor decision.
- #321 (`b8c815e5`) still needs its final benchmark rows against a same-machine
  `main` baseline; it merges after #324. B4c's traced recovery branch
  (`f6157303`, not yet pushed at that head) waits on its full-jit seed gate,
  which was killed partway through its QI case on the office workstation;
  #325 (`06b0e4b2` locally) stacks on it.
- #320 needs the linear-progress guard (stop only when the unconverged inner
  solve gained fewer than three digits and the step made no progress), then
  its two-deck re-verification against `main`.
- #326 (B1's two-deck record) needs compaction before merge: the JSON under
  about 1,000 lines and the script trimmed toward 450–500 lines.
- After that: B3 (exact block adjoint), the goal-oriented certificate PR,
  and B1c. Close #299 when the last S1 piece merges.

**2026-09-14, afternoon: merges, the 251 GB cause, and the next user-facing
work.** Merged: #317 (`39f0bead`) and #316 (`9a6b5efc`). The office
workstation is reachable again, and heavy rows move there (§7).

- **251 GB request, found and fixed (#328).** `reverse_jit = jit(jacrev(residual_rows))`
  vmapped the GCROT(m = 100) adjoint over all 6,722 residual rows, giving
  buffers of shape (6722, 101, 9300), 47 GiB each. It sits in the `lax.cond`
  fallback taken only when the block Jacobian misses its certificate, so it
  allocated on office and not on the laptop. It is neither a JAX 0.9.2 nor a
  #314 regression: both JAX versions trace the same buffers. #328 pulls rows
  back in tangent-lane batches (0 intermediates above 64 MiB, from 9,138). A
  `make_jaxpr` scan of intermediate sizes found it without executing anything;
  the runtime bisect had passed because the laptop never took the branch.
- **Correction to B3's evidence.** B1's dense check on the seed deck (rank
  4,207 on range(P), condition 6.3e12 raw and 1.5e12 preconditioned) puts the
  production Krylov error at 8.7e-6 (QA) and 3.2e-5 (QI). The 1.1e-3 QI gap
  attributed to it is the raw-versus-preconditioned formulation at a non-root
  anchor: the 1-D preconditioner is recomputed from the state, so its implicit
  derivative carries an O(|F|) term. The least-squares Jacobian lane is raw
  and the scalar adjoint lane is preconditioned, so the two disagree by up to
  1.1e-3 on QI today. B3 aligns them on the raw formulation (a Changed entry,
  not a Fixed one). Jitted and eager QI gradients on `main` also differ by
  5.2e-5, the stagnating Krylov solve's level; B3a's direct solve removes it.
- **Where QI least-squares time goes.** The QA and QI benchmark rows never
  reach the implicit adjoint: their gradient is J^T r from the certified block
  Jacobian. The 51-second QI adjoint matters for scalar losses, direct
  `jax.grad`, `minimize(objective_terms)` and the single-stage example.
  Least-squares QI time is forward solves, refinement and certification
  (#320, B1c), then the objective's convergence (D0, D1).
- **#320 (refinement stops after a non-improving step).** QA_lowres is
  bit-identical to `main`. On the seed deck, main's refined anchor moves by up
  to 2.87e-2 under a 2.2e-15 input perturbation (gradient 2.8e-4) while #320's
  moves by O(1e-15). The goal-oriented error ratio |μᵀF_raw| (#320 / main)
  over four rounding draws is 2.31, 2.19, 1.00, 1.47 (QA) and 1.20, 1.16,
  1.00, 0.69 (QI), which passes the rule set before measuring (merge unless
  main is more than 2× lower in at least 3 of 4 draws). The seed deck's QI
  accuracy floor is about 2e-3 at both anchors. Main's first refinement step
  raises |P gc| from 1.9e-7 to 6.5e-6–1.4e-5 and steps 2–3 carry its progress;
  that is B1c's target. At load under 20: refinement 18.1 → 12.9 s (QA) and
  21.0 → 11.4 s (QI); first derivative 43.1 → 38.8 s and 51.7 → 42.1 s.
- **#321 rows** (JAX 0.11.1, load ≤ 20, `main` `0a8e3068` vs `be360e47`): QA
  row 95.3 → 51.0–57.1 s, QI 117.5–121.0 → 61.8–62.3 s; JAX value and gradient
  39 → 1.5 s and 55 → 1.6 s; host values and gradients identical. It merges on
  green CI.
- **B4c gate** (one program, no callback, 0 warm compiles, decisions identical
  to the host loop): QA gradient within 3–5e-12 of the same-backward reference
  on both JAX versions; QI within 5.2e-5 and 6.3e-6, inside the Krylov error.
  The objective's state cotangent agrees eager vs jit to 4.7e-14. QI re-gates
  at 1e-10 after B3a.
- **#318:** counts match `main` except refinement Krylov iterations (16,643 vs
  16,645); its timing and memory verdict comes from a quiet office run.
- **#300:** README condensed to its 300-line cap; the DESC caveat sentences
  are unchanged.
- **Next user-facing work.** Briefs B3, D0, D1 and C3 (§8). B3a speeds every
  scalar-loss and single-stage trial; D0 and D1 target QI convergence; C3
  replaces the single-stage augmented-Lagrangian L-BFGS-B with one least-squares
  problem.

**2026-09-14, evening: merges, the QI example fix, and a force-panel report.**
Merged: #326 (`efd04f5f`), #328 (`ac379d87`), #329 (`533d6760`), #320
(`2fbe4e57`), #321 (`b208df4f`), #332 (`afee5265`), #331 (B4c, `c4329b75`) and
#300 (DESC bridge, `066c35c7`). #321 and #300 were tested on a tree merged with
current `main` first, because their CI predated later merges. #318 closed: on
office its warm Jacobian gain (4.29/5.13 s against 4.83/5.10 s) was inside
spread and peak memory fell 4–5.5 %, not the 32 % in #299's record; #299 closed.
A session limit stopped three agents mid-task; their office jobs had finished.

- **The shipped QI example failed.** On `main`'s library the unchanged example
  ran 1,456 s, then raised in its final ns = 101 solve: the circular seed drives
  the optimizer to a design with no converged equilibrium at ns = 31 or 101,
  which trials accept under `max_fsq_ratio = 1e6`. #333 seeds from
  `input.QI_nfp2_initial`, runs one `max_mode = 2` stage and penalizes the mirror
  ratio from 1 % below its limit: 612 s, 0 failed trials (5 before), constructed
  QI 3.0e-3 (0.789 before), ι, mirror and elongation limits met, final solve
  converged. Refinement is still 361 s of its 410 s least-squares phase.
- **B3a (#330)** timings at office load ≤ 24: QI eager `jax.grad` 203.9 → 55.5 s,
  QI `from_loss` first value and gradient 190.2 → 89.0 s, single-stage example
  31.2 → 26.0 s per trial. Its multi-RHS test had compared the raw block pullback
  with the preconditioned Krylov one; it now uses an independent raw reference.
  The full-marked multi-RHS test failed on `main` on an empty parameter leaf;
  #330 fixes it. XLA does not hoist the block factorization out of #328's
  chunked map (1.77 s per chunk against 1.52 s per factorization), so B3b shares
  one factorization.
- **B4c gate:** the full-jit QI gradient is bit-identical to `jit` with the
  host-callback forward; the 5e-5 gap to eager evaluation is the stalling Krylov
  adjoint (16,161 against 17,270 iterations), which B3a replaces.
- **C3 (single stage as least squares)** is not adopted yet: at trials 20/40/60
  #311's augmented-Lagrangian L-BFGS-B reached violation/objective
  0.082/2.885, 0.085/2.413, 0.189/1.499; least squares 0.208/1.899,
  0.196/1.723, 0.163/1.707, and `x_scale="jac"` 0.135/2.166 at 60. A full run
  diverged at stage 4. With the coil fit matched (2,000 trials, 0.628 % B·n) and
  `x_scale="jac"`, least squares reached 0.043/2.527, 0.063/1.916 and
  0.051/1.868: more feasible than #311 at trial 60 but with a 25 % higher
  objective, so no variant earned a full run and the example keeps #311's
  optimizer with B3a's gain. Reverse-mode coil rows cut a warm coil Jacobian
  from 4.0 s to 0.094 s. #320's guard never fires in this example: refinement
  is about 1,700 GCROT iterations and 96 % of each value-only trial on both
  libraries. B1c is therefore the next lever for single stage and QI (88 % of
  #333's least-squares phase), and least squares gets one more proxy after it.
- **Office runbook:** only A/B timing rows take the heavy-job lock; whole-example runs
  take one of two long slots; untimed work runs lock-free at load ≤ 24.
- **Force-balance panel:** users report errors near 100 %; brief F-err (§8)
  investigates the saturating `equif` normalization against DESC and VMEC2000.

**2026-09-14, night: the solver work lands.** Merged: #333 (QI example,
`f78efafe`), #336 (`395fea8b`), #337 (force panel, `5a474463`), #325
(`d52f9d81`), #338 (B1c, `8b17fe06`), #339 (`382561da`), #340 (`2caf2516`),
#330 (B3a, `b5f84b10`) and #335 (B3b, `9dfa18a2`). PRs whose CI predated a
merge that touched the same files were checked on a tree merged with `main`
first; one such check caught a stale benchmark index on #330 before it merged.

- **B1c (#338), refinement as a block-preconditioned Newton finish.** On the
  office workstation, same day, at load ≤ 24: #333's QI example 565.1 → 359.1 s
  wall, least squares 366.9 → 127.5 s, refinement 333.0 s and 63,316 Krylov
  iterations → 66.7 s, 153 GMRES iterations and 20 factorizations, with the
  same 17,251 descent iterations, 0 failed trials and final cost within
  7.2e-7. The single-stage example ran 30.8 → 22.3 s per trial (refinement
  0.34×; the rest, about 211 s, is solves, the plasma gradient, coils and
  compile), finals within 1.2e-11, peak memory 2.42 → 2.80 GiB. Objective
  replay at a repeated x holds within 1e-9 on both decks. Where the Newton
  phase stalls (the seed benchmark deck) it replays today's Krylov anchor bit
  for bit at +5.9 % warm time and a 14 s first compile. The ≤ 0.5× gate is met
  on QI least squares and not on single stage, and was accepted as measured.
- **B3a (#330) and B3b (#335).** The implicit backward rule solves the raw
  adjoint through one block factorization: QI eager `jax.grad` 203.9 → 55.5 s
  and single-stage trials 31.2 → 26.0 s before B1c. B3b shares one
  factorization across reverse-Jacobian rows: on seed QA the reverse Jacobian
  takes 238.9 s, where #330's per-chunk path had not finished after 4,111 s,
  with the value identical and the gradient within 1.8e-13 of the block lane.
  A gradient at a new point builds two factorizations, B1c's preconditioner at
  the start iterate and the adjoint at the refined anchor.
- **B4c QI gate after B3a.** The full-jit QI gradient agrees with the
  same-backward reference to 4.85e-10 (JAX 0.11.1) and 3.71e-10 (0.9.2),
  from 5.2e-5 and 6.3e-6. The staged and host adjoints agree within 6.5e-11;
  the traced forward carries the rest (state 6.7e-14 and 1.1e-13 apart on a
  deck with raw condition near 6e12). Recorded floor: QI full-jit gradient
  within 1e-9 of the same-backward reference on the seed deck; QA stays at
  1e-10. One program, no warm compiles, no callbacks.
- **Force panel (#337).** WOUT's `equif` matches all nine VMEC2000 goldens to
  5e-13 and is bounded by 1, equal to 1 on currentless vacuum; the summary now
  plots DESC's normalized force error (converged vacuum QA 2.65e-4, 0.143 after
  40 iterations; within 12 % of DESC's own value on DESC-solved equilibria).
- **CI budget.** The Python 3.12 fast lane ran past 8 minutes because pytest
  workers each recompiled the refinement-staging tests (#336 moves that module
  and `test_mgrid.py` to their parity lanes). The c2 parity lane ran past 25
  minutes (#340 moves seven slow diagnostics modules to a new c4 lane: c2
  12:46, c4 13:49).
- **Benchmark harness (#339).** The QI profile child runs in an empty
  directory, so a relative `PYTHONPATH` imported an installed VMEX; the child
  now prepends the checkout and asserts the import. #333's record used absolute
  paths and stands. On the office workstation, always use absolute paths.
- **Next.** D1 (a differentiable QI well location) is running on #333's
  example. A wall-time split of both examples on this `main` ranks the next
  lever: about 230 s of the QI example sit outside least squares and about
  211 s of a single-stage run outside refinement.

**2026-09-15, the 0.9.0 release round.** Merged since the last entry:
#342 (single-stage coil pre-fit capped at 200 iterations, `35e7170a`), #343
(compilation-cache trim keeps recent entries, `04afcf9e`), #344 (README `--plot`
summaries, `9441d403`), #345 (`0a6b1da1`), #346 (installable ESSOS commit named,
`fed67192`), #348 (per-test ESSOS skips so the module still collects,
`46149d5d`), #349 (misc parity lane split, `c59dbccf`), #350 (LASYM QH two-stage
budget restored, `6539d679`), #351 (refine uncertified block Jacobian columns
instead of mapping the reverse lane over every row, `e26b3f83`). Then today's
round: #352, #353, #354, #355 and the two defects below.

- **The independent cross-code scoreboard exists, and Phase G should use it
  rather than rebuild it.** `itpplasma/benchmark_vmec` published a 4,128-row
  result on 2026-09-14: 344 cases through twelve codes, VMEX among them. Read by
  the `status` column alone VMEX ranks last of the VMEC family (168 successes
  against vmec2000's 198); joining `comparison_table.csv` with
  `plots/quality.csv` on `residual_over_tolerance` inverts that reading. Strictly
  converged (residual at or below the requested tolerance): vmec2000 179,
  parvmec 177, **vmex 168**, vmecpp 157, jvmec 140, educational_vmec 94, gvec 40.
  VMEX and jVMEC are the only codes whose reported successes are all converged —
  VMEX's worst `residual_over_tolerance` is 0.9999, against 1e+30 for vmecpp and
  4e+22 for educational_vmec. Median wall per case, one case per process:
  educational_vmec 0.90 s, vmec2000 2.04 s, vmecpp 3.35 s, **vmex 23.11 s**,
  desc 75.28 s — the compile tax of §2, measured by a third party.
- **Of the 35 cases where VMEX fails and a VMEC-family peer succeeds, 12 were a
  file-writing policy and 23 are a robustness backlog.** 141 of VMEX's 176
  failures are corpus-wide (GVEC/SPEC/SPECTRE-native inputs, missing mgrid
  fixtures) and every VMEC-like code fails them. Of the remaining 35, twelve are
  cases where every peer that "succeeded" was itself unconverged. Four of those
  are VMEX's own shipped decks, including `input.LandremanPaul2021_QA_lowres`:
  `vmex` on that deck reaches FSQR 2.63e-13 against its deck's 1e-13, prints
  MORE ITERATIONS REQUIRED, exits 2 and writes **nothing**, discarding 29 s of
  work. Cause, `cli.py:807,832`: `raise_on_max_iterations=not lfull3d1out`.
  VMEC2000 does not do this — `vmec.f`'s `more_iter_flag` branch with
  `lmoreiter = .false.` re-enters `runvmec` with `ictrl(2) = 0`, `runvmec.f`
  calls `fileout` whenever `ier_flag /= more_iter_flag`, and `fileout.f` sets
  `lwrite = lterm .or. ier_flag == more_iter_flag` before `wrout`; `LFULL3D1OUT`
  governs threed1 fullness only. Fixed in #356: the WOUT is written, the exit
  code stays 2, and `wout.ier_flag` records 2 rather than the 0 `vmec.f`
  substitutes. The other 23 are a genuine convergence backlog on legacy 3-D
  decks (`atf*`, `bean14`, `belt20`, `c82vac20`, `qas14`, `w7s20`, `cooper`,
  `Q2_KINK`, `HSX_QHS`, `WISTELL-A`, W7-X `d23p4_tm`, `qhs46`, ITER hybrid) and
  are not on this plan; see A5 in §4.
- **A manifest-owned test ran in no workflow for six weeks (#357).**
  `test_use_fft_reaches_every_free_boundary_lane` spied on `_make_body` with a
  re-declared keyword list; `_make_body` gained `evaluation_synthesis` on
  2026-08-01 (`8c4d83bb`) and the spy has raised `TypeError` ever since, the
  moment the vacuum steady lane is traced. It hid because the node needed
  `examples/data/mgrid_cth_like.nc` from `reference-nc`, which PR lanes do not
  fetch by policy, so it skipped everywhere; Weekly's hmfb campaigns select four
  named nodes and not this one. The fix keeps the policy: the spy takes
  `**kwargs`, asserts it reached the steady lane, and runs on the generated
  modular-coil fixture. Found by the run no lane performs — the whole suite with
  every bundle installed. On this `main` that run is 1 failed, 1,835 passed,
  128 skipped, 2 xfailed in 24 min; it belongs in the release checklist.
- **The field, checked 2026-09-15.** VMEC++ is building VMEX's differentiator:
  an implicit adjoint through Enzyme (#710, #841, #854), block-tridiagonal
  `H_SS^T` for a direct adjoint solve (#857 — the same move as B3a), m = 1 gauge
  pinning so the equilibrium is a function of the boundary (#849), and an open
  proposal to ship Enzyme kernels in pip wheels (#814). §2's "pip users of
  VMEC++ have no derivatives" is true today and has a visible expiry date. DESC
  0.17.3 adds a sparse pullback giving an N-fold Jacobian speedup (#2170) and
  reuses the Jacobian QR across the Levenberg-Marquardt sweep — both are Phase C
  comparisons.
- **Floors.** solvax 0.22.0 is on PyPI, so §6's ≥ 0.21.0 gate is clear.
  `booz_xform_jax` 0.3.0 and `essos` 0.17 remain untagged, so the magnetic-only
  Boozer projection and twelve ESSOS examples are still unavailable to a plain
  `pip install`. jax/jaxlib 0.11.x require Python ≥ 3.12; 3.11 caps a user at
  JAX 0.10.2.
- **Limitation.** Every timing above is either third-party or a single-run
  laptop sample; nothing here is an A/B row. The GPU workflow has no registered
  self-hosted runner and last succeeded 2026-07-31, so 0.9.0 ships without a
  current GPU result.
- **Next action.** Phase D (D1, a differentiable QI well location) as before,
  plus A5 for the 23-deck robustness backlog and the Phase G retarget above.


**2026-09-16: 0.9.1, and where the cold seconds, the GPU failures and the
gradient doubts actually are.** Merged #362 (`8db9c825`). ESSOS 0.17,
`booz_xform_jax` 0.3.0 and solvax 0.22.0 are all on PyPI, so the coil examples
install from PyPI and every floor the plan gates on is raisable. Measurements
below are from this laptop (14 cores, JAX 0.11.1, x64) and from the office box
(2x RTX A4000, driver 580.173, JAX 0.11.1 CUDA), with
`VMEX_COMPILATION_CACHE=disabled` where a cold compile is the subject.  Both
machines carried other jobs, so wall times are upper bounds; byte counts,
error estimates and ratios are not affected.

- **Optimization gradients could not be compiled on any machine with a GPU
  (fixed).** `resolve_implicit_device` stands the implicit path down to the CPU
  on an accelerator backend, `_callback_sharding` pinned the host callback
  there, and JAX requires a pinned device to appear in the enclosing
  computation's device assignment -- which for a jit compiled on a GPU box is
  `[cuda:0]`. All three ways of asking (default placement,
  `jax.default_device(gpu)`, `vmex device_scope("gpu")`) died in JAX's lowering
  with `ValueError: tuple.index(x): x not in tuple`, no VMEX frame; the same
  program under `JAX_PLATFORMS=cpu` returned `f = 8.2959e-11`. JAX catches
  `IndexError` there where `tuple.index` raises `ValueError`, so its own
  actionable message never printed -- worth an upstream issue. The pin is now
  applied only within one platform.
- **The GPU loses on both shipped decks, including the one the policy
  recommends for it.** Office, one process per row, peak device memory from the
  allocator's own counter with preallocation off:

  | deck | CPU cold | CPU warm | GPU cold | GPU warm | peak device |
  |---|---|---|---|---|---|
  | `circular_tokamak` | 6.74 s | 0.22 s | 12.82 s | 1.19 s | 0.16 GiB |
  | `QA_lowres` | 16.99 s | 6.20 s | 35.96 s | 6.91 s | 0.16 GiB |

  `recommended_device` answers `gpu` for `QA_lowres` (work proxy 1,536,000
  against a 100,000 threshold), and on this hardware that is the wrong answer
  cold and warm. The full sweep then settled it
  (`benchmarks/gpu_a4000_2026-09-16.json`, `run_gpu_matrix.py --skip-tridiag`):
  **the GPU wins no cell**, warm gain 0.17x (solovev) to 0.83x
  (`NuhrenbergZille_1988_QHS`, the largest case at 111 s of warm CPU work and
  the very deck the July baseline claimed up to 3x for), with the synthetic
  `ns` x `mnmax` scan walking from ns 35 to 151 without crossing over. So
  `GPU_MIN_ITERATION_WORK` is not merely mis-tuned here: the crossover it
  encodes does not exist on this hardware and JAX version. Do not move the
  constant on one machine -- the July numbers were honestly measured on
  another; the how-to now carries both, and the work package is a
  hardware-labelled policy rather than a single global threshold.
- **No terabyte anywhere.** Single-stage finite-beta value-and-gradient at
  ns = 15, mpol = 5, 24 dof: peak device 0.16 GiB, peak host 3.17 GiB on
  office and 2.2-2.5 GiB on the laptop, largest single traced intermediate
  7.7 MiB out of 148,500 equations. The full example at its shipped settings
  finishes in 900 s and 3.8 GB, driving the total objective 31.66 -> 4.06 over
  eight accepted iterations. The peak is XLA's *compile* working set, as
  `implicit.py:1137` already says.
- **The single-stage compile is already an 8x once-per-machine cost.** Same
  gradient, two processes sharing a persistent cache: process 1 trace+lower
  14.1 s, compile 49.8 s, run 19.5 s; process 2 trace+lower 10.1 s, compile
  **6.0 s**, run 5.4 s (380 entries, 12 MB). So the 50-70 s compile is paid
  once per machine, not once per run, and what is left to attack is the 10 s of
  Python-side tracing, which no cache can help.
- **Cold start is compilation, and a third of it is incidental.** A per-module
  census of one `input.circular_tokamak` solve counts **152 XLA compilations**
  (7.1 s) for a 7.6 s cold solve whose warm repeat is 0.16 s. Four are named
  lanes (`_block_lane` twice, 2.49 s, one per rung); the other ~130 are single
  eager JAX ops, each compiling its own module -- `broadcast_in_dim` 22,
  `multiply` 19, `copy` 17, `where` 8 -- in `setup.py`, `solver.py:2345`
  (leaf-by-leaf carry copy) and `solver.py:914` (`_zero_cache`). A two-change
  prototype (one jitted whole-tree copy, `device_put` of NumPy zeros) takes 152
  to 129 without touching physics. `QA_lowres` compiles 230 modules. This is
  B4's target, now with line numbers.
- **Four codes, same decks, same machine, one process each.**
  `input.circular_tokamak`: VMEC2000 0.23 s, VMEC++ 1.20 s, VMEX 0.16 s warm
  and 7.6 s cold, DESC 24.3 s -- all four at volume 473.74101, and VMEX against
  VMEC++ wout by wout at 3.6e-16 on volume, 1.0e-15 on aspect, 1.2e-16 on
  `iotaf`, 4.7e-12 on `jcuru`. At `QA_lowres`'s shipped
  `NITER_ARRAY = 600 1000 1000` against `FTOL 1e-13`, **no VMEC-family code
  converges** -- VMEC2000 `ierr=2`, VMEC++ raises, VMEX `ier_flag 2` at FSQ
  2.63e-13. Raised to 2000/4000/8000 all three converge: VMEC2000 10.30 s,
  VMEC++ 11.12 s, VMEX 13.87 s warm and 31.6 s cold (`ier 11`, 1115
  iterations, FSQ 9.92e-14), DESC 47.0 s, volume agreeing to eight figures.
- **The free-boundary adjoint is right; its certification was not.** Fixed
  boundary certifies at **1.09e-07** against a central difference; free boundary
  stopped at 1.06e-02 on the same script, which reads as a 1 % gradient error.
  Sweeping the step at ftol 1e-10: 1e-2 -> 3.22e-02, 1e-3 -> 5.04e-02,
  1e-4 -> -9.84e-02, 1e-5 -> -2.18e+01. It changes sign and diverges by three
  orders as the step shrinks -- noise, not a systematic offset. The two adjoint
  solvers **agree to 9.67e-05** (coupled GCROT 2.23693204e-02, `boundary_schur`
  2.23671581e-02), which closes the arbitration the F series wanted. The
  example now certifies against the second adjoint and prints 1.57e-04 at its
  shipped settings.
- **Gradient cost, steady state** (ns = 15, mpol = ntor = 3): fixed vacuum
  **1.91x** the value, fixed finite beta **5.87x**, free boundary **41.5x**.
  First adjoint in a process costs 20.8 s, 11.5 s and 70.6 s to compile. F1b's
  3x gate is 14x away.
- **`booz_xform_jax` is not a bottleneck; the exterior field's default grid
  was.** Boozer transform of 8 surfaces, warm: 16 ms at mboz/nboz 16/12, 33 ms
  at 24/18, 58 ms at 32/24. Virtual casing costs ~0.45 s per call *independent
  of target count* to 1024 (4096 targets, 0.84 s), so batch targets and never
  loop. `from_wout` hard-coded `nphi=ntheta=32`, and `from_surface_data`
  treated a per-period sampling as a full-torus level count; on the shipped QA
  wout the achieved error one minor radius off the boundary was **2.46e-02
  against a requested 1e-6**. Sizing the grid from the geometry and carrying
  the `nfp` factor, with the poloidal count following the toroidal one, gives
  **4.31e-07** at one minor radius and 5.05e-07 at two, at **0.116 s per call
  against 0.226 s** -- more accurate and no slower (the coarse grid fails its
  own self-test and the adaptive schedule then does more work, so the finer
  default measured faster; the mechanism is inferred, the two timings are not). Calibration (finest level, source points,
  error at `d = a`): (32,32) 128x64, 8,192, 6.4e-04; (64,32) 256x64, 16,384,
  7.1e-06; (64,64) 256x128, 32,768, 4.3e-07; (96,48) 384x96, 36,864, 1.4e-08;
  (128,64) 512x128, 65,536, 3.5e-11. Poloidal beyond 64 buys nothing
  ((64,96) repeats (64,64) exactly), which is why the rule pairs the two.
- **Force-balance polishing is not the five-minute item.** The shaped tokamak:
  `solve_file` (equilibrium + polish + WOUT export) 114.2 s of which the polish
  itself is 55.8 s over 3 nonlinear iterations, certified, eps_F volume L2
  1.284e-02 -> 1.820e-03; `plot_wout` is 12.2 s per figure set. The example
  reads as 312 s because it renders two full sets and pays a cold compile --
  the same script on a warm cache is 114 s + plots, and 82.1 s on a third run.
  The stellarator is the one that is genuinely long:
  `input.finite_beta_stellarator_polished` takes **328.0 s** for solve plus
  polish, certified. So "over five minutes for a single run" is true of the
  finite-beta stellarator and not of the tokamak.
- **Repo-size audit, by reference rather than by size.** 9.9 MB tracked across
  607 files, so this is a file-count and organization question, not a byte one:
  `tests` 133, `benchmarks` 131, `examples` 123, `docs` 100, `vmex` 79,
  `tools` 17, root 12, `assets` 1. Four files were referenced by nothing and are
  gone (the QI sheet-current mgrid builder, a repo census carrying branch and PR
  counts, a stale profile record, and the pre-commit config, which duplicated
  `tools/preflight.py` with a separately pinned ruff and a 200 kB file cap the
  tracked tree already exceeds). Everything else in `tools/` is referenced --
  `fetch_assets.py` 23 times and in three workflows, `test_manifest.py` in four
  -- and `assets/manifest.json` is read by eight places including `MANIFEST.in`,
  so the one-file directory stays.
- **`benchmarks/baselines/m4/` should NOT be consolidated.** It held 54 files,
  8.7 % of the tree; since 2026-09-22 only the records the docs or tests cite
  stay there (the rest are in git history). Merging them
  into one keyed record looks like a win. It is not:
  `profile_workflows.py` writes `{workflow}_{regime}.json` one cell at a time
  and re-executes itself in a subprocess for the cold regimes, so one file per
  cell is what lets a partial run on a shared machine keep its completed cells.
  Consolidating would trade that for a cosmetic count. The file count there is a
  consequence of a sound design; leave it.
- **21 of 66 shipped examples were run by no test and nothing said so.**
  `tests/test_examples.py` now parses every example and requires each to be
  named by a test or listed in `UNTESTED_EXAMPLES` with a reason. The QH, QI and
  QP entries drive the same code as a tested QA sibling, which is the reason
  they are exempt rather than uncovered. `take_fixed_boundary_gradients.py` got
  a test instead of an exemption: 5.0e-06 at the smoke settings against 1.1e-07
  at the shipped ones.
- **Two single-stage examples now ship, and the augmented Lagrangian's cost is
  measured rather than asserted.** The example's docstring had justified the
  augmented Lagrangian with a penalty run ending at min |iota| 0.07 against a
  0.42 floor -- from the OLD seed at mean iota 0.08, not the current
  0.408/0.391/4.03, so the justification was stale. Re-measured at 301 trials
  each, coil length target 5.3 m and coil-coil limit 0.17 m, both forms reach
  every target:

  | | penalized | augmented Lagrangian | target |
  | --- | --- | --- | --- |
  | aspect | 3.9841 | 3.9801 | <= 4.0 |
  | min \|iota\| | 0.4275 | 0.4272 | >= 0.42 |
  | B.n/B RMS | 0.800 % | 0.799 % | <= 1.0 % |
  | coil-surface | 0.2217 | 0.2329 | >= 0.20 |
  | coil-coil | 0.1898 | 0.1805 | >= 0.17 |
  | max curvature | 6.903 | 6.629 | <= 7.0 |
  | objective | 6.48e-02 | 4.49e-02 | |
  | wall, iterations | 697 s, 149 | 755 s, 127 over 6 stages | |

  The penalized form is not slower and not worse at reaching targets; what it
  costs is tuning. A quadratic penalty asymptotes just INSIDE the threshold it
  is handed, so each limit must be given tightened (coil separation 0.19
  against the 0.17 limit) and the weights raised until the design lands
  outside -- three measured attempts, the first of which settled at 0.1586.
  The augmented Lagrangian reaches the same targets with the limits as stated
  and a 1.4x lower objective. Ship both: the penalized file is the one to copy
  for a new problem, the augmented Lagrangian the one to finish with.
- **`LENGTH_TARGET` 4.1 -> 5.3 on the fixed-boundary single stage.** The seed
  coils have circumference `2*pi*0.65 = 4.08 m` but the optimizer drives them to
  about 5.3 m whatever the target says, so 4.1 was a tug it always paid and
  never satisfied. At 5.3 the coils land at 5.312 m, the length term stops
  competing with the normal-field one, and every margin improves: curvature
  6.629 against 6.909 at the old 4.1, and coil-coil 0.1805 against a 0.17 limit
  where 4.1 managed 0.1714 against a looser 0.15.
- **L-BFGS-B is required, not preferred, on this problem.** At the full-budget
  solution 3 of the 99 coil dofs sit exactly on the +/-3.0 parameter bound, in
  both forms. BFGS cannot represent a bound, so it would leave the region the
  seed solve is known to converge on. A joint least-squares driver (plan C3)
  would change the optimizer, not the constraint handling: its residuals are
  still weighted, so it would inherit the penalized form's threshold offset.

**2026-09-19, the step-back review.** On the maintainer's instruction the work
stopped and seven read-only reviews ran on the areas he named: free boundary
and its derivatives (L-FB), the VMEC extender (L-EXT), cold runs and the single
solve (L-SOLVE), the optimization loops (P-OPT), the repository census
(SLIM-A), source and tests (SLIM-B), and one example template with a separate
correctness pass (EX). Base `305e9b23`, re-checked at `f86cd13a`. Every report,
script and JSON is in `vmex-review-evidence-20260919`, outside this repository;
§2 and §4 are revised from it and each new number names its report.

- **Merged while the review ran.** VMEX #369, #370, #372, #373, #374, #375;
  upstream virtual_casing_jax #7, #9 and #10, with 0.0.6 bumped on `main` and
  awaiting publish. #375 is Phase F's `F-pre`. #10 vectorises the grid-sizing
  call by 86× at 8 targets, 19× at 64 and 5.2× at 512; the dominant cost was
  not the Python loop both sides had assumed but a pointwise Fourier
  evaluation of the level's nodes, whose cost does not depend on the target
  count — mode truncation and subset sizing are deferred to the wiring PR and
  the 19× gap at 512 targets stays on the record. Ten PRs are open (§6).
- **Measured, and it changes the plan.** A cold run is 60–66 % XLA compile and
  40–50 % of those seconds are ~130 single-op eager programs per rung that no
  solver lane needs; the iteration stops scaling at four cores, so B5's
  sharding arm is dropped rather than gated, and its dispatch arm is now sized
  by an HLO census (844 fusions, 570 serial Thomas trips). 49 % of the QA
  optimization run is per-stage recompilation at identical array shapes. A warm
  QI trial plus Jacobian is ~97 % block assembly and factorisation. NESTOR is
  second-order accurate and the cause is an antipodal kink, not the singular
  point; its LU is 1.5–2 % of a call and its kernel scan 50–55 %, so F1a's
  `custom_linear_solve` is not a speed item and VMEC++'s header claim does not
  hold for this port. The real NCSX edge Schur matrix has cond 8.6e5, ~35–40
  outlying eigenvalues and unrestarted GMRES reaching 2e-10 in 50 matvecs
  against 247 probe columns and ~671 coupled matvecs today. The converged
  exterior plasma field is not curl-free (5e-3 at `d = a`, 0.2 near the LCFS)
  because the LCFS data do not conserve current, and a target-graded trapezoid
  rule reaches ≤ 4e-9 in B for every `d ≥ 0.005 a` at today's default node
  count. PR-lane coverage is 94.31 % with 24 modules below 95 %, the coverage
  omit hides three modules and not two, and 1,979 unique tests execute 3,055
  times per PR.
- **Decisions recorded.** L-EXT's B2 is closed by upstream #7 (VMEX's third
  derivative at `d = a` 1.2e-2 → 8.5e-9). E1 is dropped, killed by its own kill
  rule. E3 as written is withdrawn — a better extrapolation does not remove the
  curl — and the curl-free projection (#381, default off) replaces it; flipping
  that default is the maintainer's call. Phase E's §1 gate is restated as
  a quadrature gate against a converged quadrature of the same data plus a
  separate source-data gate, because 1e-6 is unreachable from ns = 50 data
  whose floor is 1e-5. F1b is superseded: the dense response ships as a Krylov
  preconditioner (#380, 4.8× measured at the real deck resolution) and the
  endpoint is unrestarted GMRES on the exact reverse-mode edge Schur operator;
  the "matvec ≤ 0.1× one NESTOR call" gate is retired with its arithmetic
  reason, and two measurable metrics replace it. P1 and P4 are already in
  execution, C1's blocker having dissolved when the batch dependence failed to
  reproduce. Branch deletion, a history rewrite and the projection default are
  maintainer decisions and are listed as such in H5, not actioned.
- **One conflict to hold.** FB-A's per-gradient closure hoisting (F7) and
  L-SOLVE's S2 (narrowing the inter-rung `jax.clear_caches()`) act on the same
  symptom from opposite ends: the hoisting is the structural fix for the Schur
  lane's 0.29 GiB-per-gradient leak and its retrace cost, and the cache
  clearing is the workaround that hides it. They must not both land — F7 first,
  then the call is removed rather than narrowed.
- **Closed as negative results**, each with its measurement, in the kill-rule
  list of §4: CPU sharding, pad-and-mask, XLA flag and scheduler tuning, tiered
  compilation, Anderson/NGMRES/CG/L-BFGS as descent accelerators, learned warm
  starts, mixed precision for now, PCR and banded LU, Broyden updates,
  `vmap`-batched trials, multi-fidelity ladders, forward-versus-adjoint
  selection by problem shape, the coil-side reverse pass, reusing a
  neighbouring trial's factorization as a solver, low-rank NESTOR responses,
  the frozen-matrix derivative, the Hadamard response as an operator,
  partition-of-unity quadrature inside the loop, equivalent sources, and a
  history rewrite of `main`.
- **A prediction that was tested and failed.** L-FB predicted that a
  forward-built Schur certified at 1e-9 against the reverse-mode operator
  would fall back between mpol 7 and 8. FB-A ran it: NCSX at mpol = ntor = 8,
  ns = 15, 11,520 dof, five perturbed trials, **zero certificate fallbacks** in
  both the `edge_response` and `coupled_gcrot` lanes, with the
  response-linearized JVP matching the exact coupled one to 1.41e-13 relative
  (1.5e-11 on the value). The cancellation is real in the neighbouring
  quantity — the response's own JVP against a central difference of the vacuum
  pressure goes from 1.1e-9 at mpol 3 to 2.0e-6 at 8 — but it does not
  propagate, because that design never compares a forward-mode kernel
  derivative with a reverse-mode one: the response is a dense matrix, JAX
  transposes it exactly, and the surviving forward-versus-reverse comparison
  is the certificate against the exact coupled transpose, which passes. L-FB's
  R2 (an mpol-aware acceptance) is therefore not needed for that lane and F4
  is narrowed to the test. The scaling is the honest cost: the lane's speedup
  falls to 2.16× at mpol 8 from 4–5× at mpol 3–7, as the response build takes
  a larger share.
- **Still open, and stated as such.** The DMerc vacuum example's stage failure was
  not reproduced; the stability weight escalating as `10**(stage-1)` against a
  `stability_scale` computed once at the seed is where to look first. E2's
  tabulated field and F8's free-boundary Newton finish are inferred from
  structure and were not prototyped.
- **A counter this plan gates on cannot fail loudly.** Main's c1 parity lane is
  intermittently red on a pre-existing flake:
  `test_compile_counting_and_warm_contract` fails `assert 0 >= 1` because the
  compile counter reads zero, alternating across recent main runs (failure at
  `80707817`, success at `4d822c40` and `456b3546`, failure at `f86cd13a`), and
  the weekly `test_cold_and_cache_reload_subprocess_regimes` fails `assert 0 > 0`
  in the same file. The harness cannot tell a warm-cache zero from a broken
  counter, and the same counter feeds committed records, so a silent zero
  corrupts data as well as tests. Being fixed separately; recorded as A7
  because every compile-side gate here — B4, S1, S2, S4, P1 — is stated in
  compile counts.
- **Free-boundary determinism is a measured trade, not a clean win** (FB-B,
  #383). Fixing the reference seed makes five calls at one point
  bit-identical, removing a 57 % objective spread, and takes a warm trial from
  21–26 s to 2.0 s. What it costs is smoothness: main's finite differences
  looked clean only because every leg restarted from the previous call, which
  correlated them while the values were not reproducible; with a fixed
  reference nearby parameters can take different paths and 1–4 % roughness
  remains. A warm restart from that reference also stalls on large moves — 2 %
  on every coil current burns 2,500 iterations and lands 9.2e-3 from a cold
  answer that converges in 76, while 1e-3 on one dof reconverges in 106 — so
  under `max_fsq_ratio = 1` a large step becomes a failed trial, which bounds
  how far a line search may move before the reference is rebuilt. Rebuilding
  it on a status-2 trial is the identified next move (F9). The roughness is
  not resolved.
- **Limitations.** Every wall time in the seven reports is an upper bound: both
  machines carried other sessions all day (load 10–60, briefly above 100), and
  only the rows that say so are A/B. L-FB's sections 2–7 use one synthetic
  surface and one real Schur matrix; L-EXT ran only the nfp = 2 QA family and
  did not execute the tracing examples; P-OPT's single-stage run was capped at
  600 s and covers 163 of ~300 trials; SLIM-B's coverage is from PR lanes only,
  which carry no per-test contexts. Several literature sub-reviews were lost to
  a usage limit and nothing from them is cited.
- **Next action.** Land the ten open PRs in §6, then, in this order: A6 and H1
  (the jit-disabled suite and the examples correctness pass, because they gate
  the trust of every other number) together with A7, whose counter every
  compile-side gate is written in; then F7 before S2, then S1–S3 and
  P2–P3–P5, then Phase E's E4–E7 behind the restated gates, with F9's
  reference-rebuild policy landing beside the free-boundary work.
