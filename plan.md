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
| E | An exterior field with a stated error | vacuum identity below 1e-6 at every distance down to 0.005 minor radii on the default grid; tracing outside the plasma in seconds |
| F | A free-boundary gradient that costs at most three forward solves | the free-boundary single-stage example converges to a stated design in under ten CPU minutes |
| G | The published comparison and the paper-1 package | cross-code table from committed records on named hardware; then §5's E3 and ladder |

A → B → C is the order; D, E and F are independent of each other and start
after A. §5's force-balance line resumes after G. No release before every open
PR is merged or explicitly deferred and the gates of A–F hold.

## 2. Evidence at the baseline

Measured 2026-09-13 on one Apple-silicon laptop (14 cores, shared, load 3–11)
unless a record is cited; timings are diagnostic samples, not rankings.

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

**Free boundary.** In the host GCROT lane every transpose matvec re-runs the
primal, including NESTOR's full assembly and LU, because `_transpose_matvec`
builds its VJP inside the jitted matvec (`freeboundary_implicit.py:853–860`,
`freeboundary.py:913–927`); the traced and Schur lanes hold one VJP closure. One
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
| slow QI | two nonlinear solves per trial (descent to 1e-12, then a refinement that on the seed deck exhausts 6,000 GCROT iterations and returns the state unchanged, because the raw Jacobian's edge λ modes put its condition number near 6e12); serial Jacobian with a per-dof GMRES corrector; 45–190 s recompile per `max_mode` stage; a non-smooth surrogate residual (`argmin`, `cummax`, `interp`) with 17,712 rows that fails trials; a circular-torus seed where every published QI result used a near-axis one |
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
| B3 exact block adjoint (measured on both decks: adjoint residual ≤ 1.2e-10 in one factorization, 300–5,000× cheaper than production with the Jacobian's factor reused, while B1's dense check puts the production Krylov error at 9e-6 (QA) to 3e-5 (QI); the 1.1e-3 QI gap between lanes is the raw-versus-preconditioned formulation at non-root anchors, which B3 aligns on the raw formulation), then Krylov recycling | on the seed deck the QI adjoint's GCROT stalls at 2.6e-5 and takes 13,228 iterations (78 s), while the raw block factorization is exact at the state, so first solve the adjoint by transposing that factorization together with the 1-D preconditioner (one direct solve; B1 is measuring it); keep GCROT as fallback and certifier; then return the adjoint's Krylov iteration count from compiled programs (A1's counters report `None` for every compiled adjoint, which covers scalar `minimize()` gradients); then warm-start λ across trials, reuse the GCROT deflation space across Newton and adjoint solves, and freeze the preconditioner in the matvec at the root | three solves of one operator per gradient today | adjoint matvecs per gradient ≤ 0.5×, measured by the returned count |
| B4 compile hygiene | `x0` out of the jit key; configs keyed by content; one compile per resolution; persistent cache where jaxlib allows; the JAX value-and-gradient lane must reuse the host lane's executables instead of compiling its own | 102 compiles in a five-evaluation warm campaign; on the A1 rows (#310) the JAX value-and-gradient lane spends 41.6 s (QA) and 71.0 s (QI) compiling after the host derivative has already compiled, and builds take 244–500 XLA compiles | cold to first gradient ≤ 20 s CPU on the QI example; zero recompiles across `max_mode` stages; the second lane adds under 5 s of compile |
| B5 per-iteration constant | one bounded experiment (≤ 1 week), in this order: ms per iteration for VMEX and VMEC++ at 1, 2 and 4 threads; an HLO census of the iteration (ops, loops, loop trips; the CPU tridiagonal solve is two `lax.scan` Thomas sweeps, about 100 serial trips per iteration at ns = 50); a dispatch arm with a batched tridiagonal kernel; and only if the thread scaling shows headroom, `shard_map` with radial slabs and the tridiagonal solve split over modes, as VMEC++'s OpenMP does | 2.4 vs 0.9 ms per iteration; JAX's CPU thunk runtime has documented 2.5–14× regressions on many-small-kernel workloads and host devices share one thread pool | warm ns = 50 QA below 1.5 ms per iteration; the dispatch arm keeps iteration counts identical and the final state within 1e-12 relative; kill sharding below 1.25× on four devices or with collectives above 30 % of the iteration |

### Phase C, weeks 2–4: derivatives sized to the problem

| PR | change | gate |
|---|---|---|
| C1 batched Jacobian | `jacobian_batch_size="auto"` with the measured-memory chunk, after reproducing in isolation the 0.5–2 % batch dependence of §2; the per-column GMRES certifier already takes zero iterations on the A1 rows (#310), so the Jacobian's 27–32 s first-call cost is block assembly, factorization and their compile, which is what to reduce | warm 48-dof Jacobian ≤ 0.4 s CPU; columns identical to 1e-10 across batch sizes |
| C2 `minimize()` | route `objective_terms` through the block Jacobian or a scalar adjoint; docstring true | one linear solve per dof or per gradient, never per row |
| C3 joint least squares | single stage as TRF/LM on `[r_plasma; r_coil]` with `[J_plasma; J_coil]` (coil block by `jacfwd`); Jacobian only at accepted points; move #311's augmented-Lagrangian wrapper into one small library helper so examples stay short, and improve quasisymmetry while holding its constraints (#311 met them at 0.113 against 0.101 at its seed) | stage 1 needed 63 nfev / 26 njev where joint BFGS needed 188 / 176 → joint phase ≤ 0.4× today; design meets targets; cold re-evaluation matches |

### Phase D, weeks 3–6: the QI objective

| PR | change | gate |
|---|---|---|
| D0 near-axis seed and ladder | seed the QI example from a pyQIC near-axis QI boundary (the shipped, unused `examples/data/input.QI_stel_seed_3127` or `from_paper("QI NFP2 r2")` truncated to modes ≤ 2), run a `[2, 3]` / `[20, 60]` ladder, and add Goodman's `phimin` sign rule to the residual; every published QI result started from a near-axis seed and none from a circular torus | validation-grid QI before/after; failed trials and nfev per stage recorded; time to the current final metric ≤ 1/2 |
| D1 smooth residual | replace the two `argmin`s (`optimize.py:1039, 1068`), the five `cummax` calls and `interp` in `quasi_isodynamic_residual` by softplus/logsumexp wells with a differentiable well location (the frozen well location has zero derivative today); per-well moments instead of 17,712 point rows | Taylor test passes in both directions; failed-trial rate → 0; same final QI metric within 5 % |
| D2 target-field lane | Dudt-style omnigenity: monotone-spline `|B|(ρ,η)` plus a deformation `h`, residual = Boozer-spectrum evaluation at mapped angles (30–130 smooth rows per surface, bounce points by construction), well parameters co-optimized; scalar adjoint viable; DESC's tutorial does 100 iterations in two CPU minutes | QI example reaches today's final metric in ≤ 1/3 the time; ε_eff and the Boozer spectrum of the result reported |
| D3 Boozer cost | `oversample=1` validated by the existing fine-grid check; volume physics hoisted out of the per-surface loop; `magnetic_only` when upstream booz_xform_jax #8 merges | Boozer share of residual + JVP ≤ 1/3 |

### Phase E, weeks 3–6: the exterior field

| PR | change | gate |
|---|---|---|
| E0 level choice and per-period schedule | select schedule levels by #312's calibrated estimate instead of the double-layer self-test (which passes errors up to 0.30 at a 1e-4 tolerance), and scale the default levels by nfp (today the finest default level counts `2·nphi` toroidal points over the whole torus, so nfp = 5 with nphi = 32 gets 13 points per period); upstream in virtual_casing_jax, then VMEX's defaults | vacuum and interior identities on nfp = 2, 3 and 5 decks at the default grid meet the requested digits wherever `d ≥ 2h`; returned fields change only where the old level was under-resolved; cost per target recorded |
| E1 equivalent-source exterior field | replace near-surface quadrature by a fit: point sources on a deflated interior surface (or the LCFS offset inward by ~0.25 a) plus the analytic net-current filament, fitted by least squares to the accurate on-surface partition-of-unity field on an upsampled grid (Stein–Barnett quadrature by fundamental solutions, [arXiv:2109.08802](https://arxiv.org/abs/2109.08802)); the field is then a smooth sum everywhere outside, uniform in `d`, spectral in the source count, differentiable through the on-surface data and the least-squares solve; the trapezoid rule stays beyond ~0.5 a; the fallback if the fit residual stalls is hedgehog extrapolation (p ≈ 8 trapezoid evaluations at ≥ 3h along the normal, [arXiv:2002.04143](https://arxiv.org/abs/2002.04143)) | vacuum identity ≤ 1e-5 for all `d ≥ 0.005 a` on the default grid (bounded by today's 2e-4 on-surface error until its parameters are raised); on-surface limit agrees with the partition-of-unity path; cost per target ≤ the source count in kernel evaluations, no fixed latency above 10 ms per call |
| E2 tabulated field | (R,φ,Z) table through `MgridField.from_parameterized_cartesian_field`, divergence-cleaned interpolation, half-period symmetry; the Taylor plan retained below 0.2 a and the direct path above 0.5 a until E1 lands | field-line tracing outside the LCFS in seconds with a stated error; the `d ≳ 2h` rule enforced on the direct path |
| E3 source data | LCFS covariant field from the NESTOR channel or the high-order state instead of the `1.5x[−1] − 0.5x[−2]` half-mesh extrapolation, once E1 exposes it | finite-β interior identity ≤ 1e-4 |

### Phase F, weeks 5–8: the free-boundary derivative

| PR | change | gate |
|---|---|---|
| F-pre jit the free-boundary pullback | the free-boundary implicit pullback converts traced arrays to NumPy (`_solve_bwd_impl` → `_projected_residual` in `freeboundary_implicit.py`), so no free-boundary objective can be wrapped in `jax.jit` (found by A3, #311) | a jitted free-boundary objective's value and gradient match the eager path to 1e-12 relative on the CTH case |
| F0 measure | warm split of one NESTOR call into Green's-function transform, kernel, mode-matrix assembly and LU on NCSX ns = 15 (flop counts put the LU near 5 %; VMEC++'s benchmark header says assembly and factorization dominate) | the measured split decides how much F1a can save before F1b |
| F1a linearize once | hoist the coupled linearization out of the host GCROT lane; land #299's saved-pullback commit as its own PR; replace `lu_factor`/`lu_solve` in the vacuum pressure with a `custom_linear_solve` (the cached-LU tangent `dpot = A⁻¹(db − dA·pot)` is exact) | transpose identity at 1e-11 and the NCSX adjoint value unchanged to 1e-10 relative; cold compile peak not above today's |
| F1b edge response matrix | a linearized NESTOR matvec cannot reach 0.1× a forward call (reverse sweeps cost 1–3× the forward kernel), so build NESTOR's dense response to the edge rows, the axis and `ctor` once per gradient with forward-mode columns (about 100 edge columns plus a rank-1 `ctor` term); every coupled matvec and the whole edge Schur matrix then cost a dense multiply plus the existing batched sparse solve | response build ≤ 300 NESTOR-call equivalents; matvec ≤ 0.1× one NESTOR call; JVP against finite differences of the vacuum pressure at 1e-6; coupled-residual acceptance unchanged |
| F2 Schur Krylov, fallback | only for `nedge > 512` or if F1b fails: GCROT on the edge Schur complement preconditioned by the previous optimizer iterate's Schur LU, with a persistent recycle space (`gcrotmk`'s `CU`, not passed today at `freeboundary_implicit.py:751, 894`) and a warm λ | mean preconditioned iterations ≤ `nedge`/5 on a recorded ten-iterate trajectory; with F1b, gradient ≤ 3× a forward solve at ns = 25 |
| F3 inexact tolerance, not a frozen derivative | a frozen matrix derivative drops `A⁻¹ dA·pot`, the kernel's shape sensitivity and an order-one term (a lagged value, as in `nvacskip`, is harmless at the root; a frozen derivative is not); measure that term on CTH and NCSX, and offer cheap gradients as a looser Krylov tolerance certified by the exact adjoint residual | frozen-derivative gradient error reported and the option killed above 1e-2; free-boundary single-stage example ≤ 10 CPU minutes |

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
- F1b is killed if building the response matrix costs more wall time or
  compile memory than today's GCROT lane on NCSX ns = 15; F2 is then the route.
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

## 6. Open pull requests and the release candidate

State on 2026-09-13, evening. "Merge" always means after explicit maintainer
approval with every real CI lane green; the unsigned-commit PR gate is the only
check that may be red.

| repository and PR | state | disposition |
|---|---|---|
| vmex #308 | this plan, documentation only | merged; agents read `plan.md` on `main` |
| vmex #300 | DESC input bridge; lanes green except a cancelled Python 3.12 fast lane, re-run | merge when that lane is green |
| vmex #309 (A4) | documentation matched to records | merged `1aa5465e` |
| vmex #312 (A2, merged `68a119e9`) | exterior-field oracles and achieved-error estimate | merge when CI is green; warn-by-default kept (a checked eager call costs 2.2–2.5×, traced calls are unchanged); follow-ups: E0, and forward `accuracy_check` through the `exterior_field` facades in `optimize.py` and `problem.py` |
| vmex #310 (A1) | optimization counters and their record `benchmarks/optimization_counters_20260913.json` | merged `2b9d3a3e`; next #319 → #320 → B4b |
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

**Release candidate, VMEX 0.9.0.** Ready once #308, #300, #309, #310, #311,
#312 and #313–#318 merge, with booz_xform_jax 0.3.0 and SOLVAX 0.21.0 on
PyPI. It ships the DESC bridge, documentation and examples that match their
records, an exterior field that reports its achieved accuracy, per-evaluation
counters, and #299's component speedups. It does not fix the slow
optimizations users report: refinement (29–40 s of each first derivative,
B1), recompilation in the jitted lane (42–71 s, B4) and Jacobian assembly
(C1) follow in the next release, and the 0.9.0 notes say so.

## 7. Environment and runbook

- **Machines.** Apple-silicon laptops and a 36-core workstation with two RTX
  A4000 GPUs (`ssh office`). Numerical comparisons use isolated environments:
  Python 3.12 with JAX 0.11.1 and VMEC++ 0.7.4 for the head, Python 3.11 with
  JAX 0.9.2 for the floor. VMEC++ wheels older than 0.7 are not references.
- **Heavy-job locks and load.** Laptop `~/local/.vmex-heavy.lock`, office
  `~/vmex-agents/.heavy.lock`; take a lock with `mkdir` and set the release trap
  only after acquiring it. Other sessions share both machines, so a timing or
  peak-memory row counts only when the 1-minute load at its start and end is at
  most 20 on the laptop (14 cores) or 24 on the office workstation (36 cores);
  counts, identities and targets count at any load.
  Whole-example runs longer than twenty minutes take `~/vmex-agents/.long.lock`
  on the office workstation instead, one at a time with four threads, so they
  never block a short A/B row; their wall time is reported to the minute.
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

**Measured (A1 counters, `benchmarks/optimization_counters_20260913.json`; shared laptop under other sessions' load, so seconds are diagnostic).** On the 8-dof QA and QI rows of
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

**Facts.** On A1's rows (`benchmarks/optimization_counters_20260913.json`) the JAX value-and-gradient lane spends 41.6 s
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
`benchmarks/adjoint_formulation.py`. B3b: the reverse lanes in
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
record and any helper. Needs ESSOS branch `rj/vmex-optimization-interfaces`
(uwplasma/ESSOS#58). Branch `c3/single-stage-least-squares`.

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
  above are interim until `benchmarks/newton_finish_arms_20260913.json` is
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
locks are released. The office worktrees under `~/vmex-agents` remain for
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
