# VMEX research plan

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

**Release hold.** No tag, version bump or publication date is scheduled. One
release follows the gates of §4 and §5.

## 0. How to use this plan

An agent resuming this work reads §1, §3, §4 (phases and PR list), §6
(dispositions), §7 (environment) and the last entry of §8, in that order. It
checks the remote PR state and dirty worktrees, and continues at the first
unmet gate. After each implementation PR it appends one logbook entry in the
format of §8 and updates the affected gate in place. It does not add a work
package because another code has a feature, does not rerun the historical
inventories, and does not launch a multi-hour run before the counters of
Phase A exist.

## 1. Decision: fix what users measure

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
| D | A smooth QI objective | the QI example reaches today's final metric in at most a third of the time with zero failed trials |
| E | An exterior field with a stated error | vacuum identity below 1e-6 at every distance down to 0.005 minor radii on the default grid; tracing outside the plasma in seconds |
| F | A free-boundary gradient that costs at most three forward solves | the free-boundary single-stage example converges to a stated design in under ten CPU minutes |
| G | The published comparison and the paper-1 package | cross-code table from committed records on named hardware; then §5's E3 and ladder |

A → B → C is the order; D, E and F are independent of each other and start
after A. §5's force-balance line resumes after G. No release before every open
PR is merged or explicitly deferred and the gates of A–F hold.

## 2. Evidence at the baseline

Measured 2026-09-13 on one Apple-silicon laptop (14 cores, shared, load 3–11)
unless a record is cited; timings are diagnostic samples, not rankings.

**Equilibrium solve against VMEC++** (`input.LandremanPaul2021_QA_lowres`, ns = 50,
FTOL 1e-11, four threads; the shipped 1e-13 is unreachable for both codes):

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
needs `d ≳ 1.7 h`. The jitted schedule returns its last level silently when the
self-test fails (`virtual_casing_jax/integrals.py:754–777`); the shipped
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

**Free boundary.** The reverse residual re-assembles and LU-solves NESTOR
inside every transpose matvec (`freeboundary_implicit.py:224–229`); one
gradient costs one forward solve plus `nedge` (≈100) coupled pullbacks or up to
300 GCROT matvecs; the NCSX certificate is 280 s cold and its accuracy floor is
2e-3 from root non-reproducibility. #299's analytic-term contraction removed a
20 GiB compile, not this cost. The free-boundary single-stage example solves in
59 s at FTOL 1e-9 with no predictor and an un-jitted objective.

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

## 3. Root causes

| complaint | cause, verified in code or record |
|---|---|
| slow QI | two nonlinear solves per trial (descent to 1e-12, then refinement); serial Jacobian with a per-dof GMRES corrector; 45–190 s recompile per `max_mode` stage; a non-smooth surrogate residual (`argmin`, `cummax`, `interp`) with 17,712 rows that fails trials |
| slow single-stage | path-dependent objective, so BFGS line searches fail; refinement and the full Jacobian on every trial; penalty BFGS instead of least squares; a seed at ι ≈ 0.08 against a 0.42 floor, with B·n weighted 70× the ι term |
| slow free-boundary single-stage | 59 s solves with no predictor; NESTOR inside every adjoint matvec; un-jitted objective |
| exterior field | trapezoid rule off-surface with silent non-convergence; O(N_src) per target with 0.42 s latency; no oracle test |
| "VMEC++ is faster" | cold compile and the 2.6× per-iteration constant; the loop itself is at parity and VMEX's derivative is cheaper than VMEC++'s adjoint |

## 4. Programme

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

### Phase B, weeks 1–3: one certified solve per trial

| PR | change | evidence | gate |
|---|---|---|---|
| B1 Newton finish | in `solver.py`/`implicit.py`: when `fsq` falls below a switch threshold, take matrix-free Newton–GCROT steps inside the descent (the `_newton_step` lane with exact JVPs and the true-residual check) so the returned state is a certified root; delete the separate refinement pass; value-only trials get the same root; #302's anchor contract folds in here | refinement is 51 % of a gradient call; drift 1.5 % → 6e-8 when refined | objective replay at a repeated `x` agrees to 1e-9 on the QI and single-stage cases; the joint phase exits without precision loss; gradient call ≤ 0.5× today; certificate values unchanged on the P1 matrix |
| B2 warm starts everywhere | perturbation predictor into the free-boundary cache; rung skipping for `initial_state`; hot restart for CLI and `solve_file` sequences; `mode="jit"` inside the callback | 806 → 212 iterations at a 1e-4 move | iterations per accepted trial ≤ 0.3× cold on the QI example |
| B3 Krylov recycling | warm-start λ across trials; reuse the GCROT deflation space across Newton and adjoint solves; freeze the preconditioner in the matvec at the root | three solves of one operator per gradient today | adjoint matvecs per gradient ≤ 0.5× |
| B4 compile hygiene | `x0` out of the jit key; configs keyed by content; one compile per resolution; persistent cache where jaxlib allows | 102 compiles in a five-evaluation warm campaign | cold to first gradient ≤ 20 s CPU on the QI example; zero recompiles across `max_mode` stages |
| B5 per-iteration constant | one bounded experiment (≤ 1 week): `shard_map` over radial blocks on host CPU devices, mirroring VMEC++'s OpenMP partition, plus the HLO census of the block lane | 2.4 vs 0.9 ms per iteration | keep only if warm ns = 50 QA drops below 1.5 ms per iteration with bit-identical trajectories; otherwise record and stop |

### Phase C, weeks 2–4: derivatives sized to the problem

| PR | change | gate |
|---|---|---|
| C1 batched Jacobian | `jacobian_batch_size="auto"` with the measured-memory chunk; skip the GMRES corrector when the block solve already certifies (instrument first with A1) | warm 48-dof Jacobian ≤ 0.4 s CPU; columns identical to 1e-10 |
| C2 `minimize()` | route `objective_terms` through the block Jacobian or a scalar adjoint; docstring true | one linear solve per dof or per gradient, never per row |
| C3 joint least squares | single stage as TRF/LM on `[r_plasma; r_coil]` with `[J_plasma; J_coil]` (coil block by `jacfwd`); Jacobian only at accepted points | stage 1 needed 63 nfev / 26 njev where joint BFGS needed 188 / 176 → joint phase ≤ 0.4× today; design meets targets; cold re-evaluation matches |

### Phase D, weeks 3–6: the QI objective

| PR | change | gate |
|---|---|---|
| D1 smooth residual | replace `argmin`/`cummax`/`interp` in `quasi_isodynamic_residual` by softplus/logsumexp wells and a differentiable shuffle; per-well moments instead of 17,712 point rows | Taylor test passes in both directions; failed-trial rate on the QI example → 0; same final QI metric within 5 % |
| D2 target-field lane | Dudt-style omnigenity: parametrized `|B|(ρ,η)` plus a deformation `h`, residual = Boozer-spectrum evaluation at mapped angles, well parameters co-optimized; scalar adjoint viable | QI example reaches today's final metric in ≤ 1/3 the time; ε_eff and the Boozer spectrum of the result reported |
| D3 Boozer cost | `oversample=1` validated by the existing fine-grid check; volume physics hoisted out of the per-surface loop; `magnetic_only` when upstream booz_xform_jax #8 merges | Boozer share of residual + JVP ≤ 1/3 |

### Phase E, weeks 3–6: the exterior field

| PR | change | gate |
|---|---|---|
| E1 equivalent-source exterior field | replace near-surface quadrature by a fit: point sources on a deflated interior surface (or the LCFS offset inward by ~0.25 a) plus the analytic net-current filament, fitted by least squares to the accurate on-surface partition-of-unity field on an upsampled grid (Stein–Barnett quadrature by fundamental solutions, [arXiv:2109.08802](https://arxiv.org/abs/2109.08802)); the field is then a smooth sum everywhere outside, uniform in `d`, spectral in the source count, differentiable through the on-surface data and the least-squares solve; the trapezoid rule stays beyond ~0.5 a; the fallback if the fit residual stalls is hedgehog extrapolation (p ≈ 8 trapezoid evaluations at ≥ 3h along the normal, [arXiv:2002.04143](https://arxiv.org/abs/2002.04143)) | vacuum identity ≤ 1e-5 for all `d ≥ 0.005 a` on the default grid (bounded by today's 2e-4 on-surface error until its parameters are raised); on-surface limit agrees with the partition-of-unity path; cost per target ≤ the source count in kernel evaluations, no fixed latency above 10 ms per call |
| E2 tabulated field | (R,φ,Z) table through `MgridField.from_parameterized_cartesian_field`, divergence-cleaned interpolation, half-period symmetry; the Taylor plan retained below 0.2 a and the direct path above 0.5 a until E1 lands | field-line tracing outside the LCFS in seconds with a stated error; the `d ≳ 2h` rule enforced on the direct path |
| E3 source data | LCFS covariant field from the NESTOR channel or the high-order state instead of the `1.5x[−1] − 0.5x[−2]` half-mesh extrapolation, once E1 exposes it | finite-β interior identity ≤ 1e-4 |

### Phase F, weeks 5–8: the free-boundary derivative

| PR | change | gate |
|---|---|---|
| F1 NESTOR out of the Krylov loop | differentiate the assembled system with the cached LU (`dpot = A⁻¹(db − dA·pot)`) so a transpose matvec is one linearized NESTOR, not a re-assembly | adjoint matvec ≤ 0.1× a forward NESTOR call |
| F2 Schur Krylov | Krylov on the edge Schur complement preconditioned by the previous iterate's Schur matrix; warm λ | gradient ≤ 3× a forward solve at ns = 25 |
| F3 inexact-gradient option | frozen-LU skip-path derivative with a certificate tied to the `nvacskip` cadence | documented error bound; free-boundary single-stage example ≤ 10 CPU minutes |

### Phase G, weeks 7–9: the comparison and the package

One table on named hardware from `benchmarks/optimization.py` records: VMEX
against VMEC++ 0.7.4 (cold, warm, hot, adjoint) and against DESC's omnigenity
time-to-metric; the three-way gradient comparison and the Taylor test of the
2026-09-06 plan then close on the fixed anchor. Then §5 resumes.

### Kill rules and things not to repeat

- B1 is killed if Newton-finished states differ from descent-plus-refinement
  states by more than the certificate tolerance on the P1 matrix, or if
  Jacobian resets become more frequent on the shipped decks.
- B5 is killed by its own gate; do not extend it to multi-host sharding.
- D2 is killed if it cannot reach the current surrogate's final QI metric on
  the shipped cases.
- E1 is killed if the fit cannot reach 1e-4 at `d = 0.02 a` on the vacuum
  identity with at most four times the on-surface grid's source count, and the
  hedgehog fallback cannot either; E2 with the distance rule is then the product.
- Do not repeat: rcon/zcon fusion (`c595c257`), the separable-versus-dense
  synthesis switch, W7-X polish runs, winding-surface work (#301/#303/#304),
  per-row kernel tuning before the anchor is fixed, and comparisons against
  VMEC++ wheels older than 0.7.

## 5. Force balance: decisions retained

These verdicts stand and are not reopened by this revision.

- **E1 passes.** The energy gradient equals complete virtual work to 7.9e-14;
  omitting lambda leaves 3.1 % (`benchmarks/e1_functional_consistency.py`).
- **E2 passes the shaped tokamak and fails 3-D.** Full R/Z reference 188.5 vs
  the structured chart's 335.3 N m⁻³; on the finite-β QA deck the best step
  reaches 2.31× against a 10× gate and the chart advantage is 1.1–1.2×
  (`benchmarks/e2_dense_reference.py`). Six toroidal planes under-resolve the
  nfp = 2 deck; use twelve.
- **The axis source data and fit are the 3-D limiter.** Three independent
  lines converge on it: E2, the residual-versus-resolution scan
  (`benchmarks/residual_vs_resolution.py`, near-axis residual rises with spline
  refinement) and the refuted knot-grading hypothesis
  (`benchmarks/knot_grading.py`). `lift_high_order_state` must reject bases
  with unfed spans instead of returning a minimum-norm fill.
- **Coordinates.** Keep `ρ^|m| q(s)` with B-splines in `s`; no chart
  replacement, no generalized toroidal angle, no ρ-uniform mesh.
- **E3 and the resolution ladder** are gated behind Phase G; they run on the
  office box with the budgets and kill rules of the 2026-09-06 plan.
- **Contract.** The published residual is Thun's `F_norm` with DESC's volume
  average over `s ∈ [0.1, 0.99]`; the bounded `eps_F ≤ 2` is acceptance only;
  `solve()` reports near-axis, bulk and edge separately.

## 6. Open pull requests

| PR | disposition |
|---|---|
| #299 | Split. (i) the Boozer λ half-mesh correction with its tests: merge; (ii) the `jax.linearize` hoist, Thomas selection and parity-transpose fix: merge after latest-head CI; (iii) `host_evaluate`: merge; (iv) #305 plotting: merge; (v) the NESTOR contraction and saved pullbacks: own PR with the CTH/NCSX certificates; (vi) the 630-line logbook: one entry of at most forty lines citing the record. The 1,159-line JSON stays as the record of (i)–(v). |
| #300 | DESC bridge; independent; review and merge. |
| #302 | The right contract, the wrong mechanism; fold into B1 and close. |
| #301, #303, #304 | Parked; winding-surface work stays stopped. |
| #277 | Merge after B1 lands; its assertion is on the quantity B1 certifies. |

## 7. Environment and runbook

- **Machines.** Apple-silicon laptops and a 36-core workstation with two RTX
  A4000 GPUs (`ssh office`). Numerical comparisons use isolated environments:
  Python 3.12 with JAX 0.11.1 and VMEC++ 0.7.4 for the head, Python 3.11 with
  JAX 0.9.2 for the floor. VMEC++ wheels older than 0.7 are not references.
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

- **Discipline.** Never benchmark a shared checkout; pin the SHA and alternate
  A/B runs. Every artifact carries a `_provenance` block. A run that ends on an
  exit code without an observation is not a result. Verified-exterior targets
  only: a bean-shaped section puts normal offsets of half a minor radius back
  inside the plasma. A vacuum deck cannot test a Taylor continuation's
  truncation term; use a finite-β self-convergence for that. Push and base
  retargets within the same minute start two CI runs; wait on the survivor.

## 8. Execution logbook

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
verdicts. Limitations: single-run laptop timings; the QI stage split on this
machine was not completed within the review; the near-surface method choice
in E1 is left to its bounded experiment. Next action: Phase A, starting with A1.
