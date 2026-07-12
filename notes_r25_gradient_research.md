# R25 — Making equilibrium-constrained least-squares gradients faster (fixed problem size)

Research survey, 2026-07-11. Scope: same boundary modes, same radial resolution;
keep FD-validated accuracy (<= 1e-4 relative) on the gradients scipy actually uses.

## 0. Cost model (taken as given)

Per scipy `trf` iterate in `vmec_jax.core.optimize._least_squares_implicit`
(/Users/rogerio/local/vmec_jax/vmec_jax/core/optimize.py):

- `fun(x)`: one full (hot-restarted) host equilibrium solve.
- `jac(x)`: **another** host solve at the same accepted `x`
  (`jacobian_rows` calls `imp.solve_implicit(params, cfg)` again), then
  `2*nm` = 24–120 preconditioned GMRES solves — one per boundary dof column
  `dz_j = -(dF/dz)^{-1} dF/dp t_j` — each GMRES matvec being a full force-map
  JVP. GMRES budget is `adjoint_tol=1e-6, adjoint_maxiter=30` per column, no
  sharing of Krylov information across columns or across iterates
  (`chunk_map`/vmap batches them but each column builds its own space).
- A max_mode 1→5 continuation campaign: multiple hours on a 36-core CPU box.

So the levers are: (a) fewer/cheaper full solves, (b) cheaper linear algebra for
the many-RHS implicit Jacobian, (c) fewer exact Jacobian evaluations, (d) a
different outer formulation entirely (one-shot/SAND).

Local assets that matter:

- SOLVAX (/Users/rogerio/local/SOLVAX/src/solvax/): `gmres`, `gcrot`
  (krylov.py:375 — GCROT/GCRO-DR-style with an explicit, fixed-shape recycle
  pair `(C, U)` that can be carried between calls; cites Morgan GMRES-DR),
  `block_thomas_factor`/`block_thomas_solve`/`block_thomas` (direct.py:67/112/182)
  — factor a block-tridiagonal once, apply to many RHS — plus
  `block_thomas_truncated(_fn)` and `chunk_map`.
- /Users/rogerio/local/vmec_jax/vmec_jax/core/preconditioner_2d.py already
  documents the VMEC2000 `precon2d.f` + BCYCLIC lineage: the force Jacobian of
  the 1D-preconditioned map is block-tridiagonal in radius with dense
  (3·mn × 3·mn) blocks, and the module docstring explicitly notes the assembled
  `solvax.block_thomas_truncated` route as the alternative it chose not to take.
- `implicit._host_solve` already has a per-config hot-restart cache
  (implicit.py ~line 652).

---

## 1. DESC: how the closest competitor does exactly this problem

Sources:
- Part II (perturbation/continuation): https://arxiv.org/abs/2203.15927
- Part III (QS optimization, Gauss-Newton trust region): https://arxiv.org/abs/2204.00078
- Constrained optimization (proximal formulation): https://arxiv.org/abs/2403.11033
- Optimizers doc: https://desc-docs.readthedocs.io/en/stable/optimizers.html
- Source (read directly): https://github.com/PlasmaControl/DESC/blob/master/desc/optimize/_constraint_wrappers.py

What DESC does (confirmed from `ProximalProjection` source, not just papers):

1. **Same implicit Jacobian identity** as vmec_jax:
   `dG/dc - dG/dx (dF/dx)^{-1} dF/dc` ("akin to a projection method" — their
   proximal wrapper re-solves equilibrium after each optimizer step).
2. **One dense factorization for all RHS, not one iterative solve per dof.**
   `_proximal_jvp_f_pure` forms the *full* reduced force-balance Jacobian `Fxh`
   by batched JVPs against the feasible-tangent basis, then does **one SVD**
   (`jnp.linalg.svd`, tiny Tikhonov shift `sf += sf[-1]`) and applies the
   pseudo-inverse to every boundary-dof RHS: `vtf.T @ (sfi * (uf.T @ Fc))`.
   No Krylov iteration at all in the Jacobian path. This works for them because
   their reduced state is small enough for dense linear algebra; the *pattern*
   (factor once, backsolve 24–120 times) is the transferable idea.
3. **No duplicate solve at the same x.** `_update_equilibrium` keeps
   `(self._allx, self._allxopt, self._allxeq)` lists and does
   `xopt = f_where_x(x, ...)`: if the optimizer asks for `fun` and then `jac`
   at the same point (which scipy always does), the converged equilibrium is
   returned from cache — zero extra solves.
4. **Perturbation warm start of the inner solve.** When x is new, they first
   apply `eq.perturb(deltas)` — a 1st/2nd-order Newton-like step using the same
   factorized `Fx` (Part II machinery) — and only then `eq.solve`, which
   "converges much faster than a cold start" (Part II). Part II demonstrates
   parameter scans "in a fraction of the time required for a full solution";
   Part III reports "orders of magnitude less computation time" than
   STELLOPT-style FD pipelines for high-dimensional optimization.
5. **Chunked batching** (`batched_vectorize` with `jac_chunk_size`) for memory —
   same as vmec_jax R17.1.
6. Their LM driver is their own `lsq-exact` (scipy-trf-like) so nothing exotic
   on the driver side; the wins are all in items 2–4.

**Relevance/effort/risk for vmec_jax:** item 3 is a ~20-line change (memoize the
host solve by dof vector; scipy's `jac` x is always the last accepted `fun` x,
so a one-entry cache suffices). Item 4 is a natural extension of the existing
hot restart: seed with `x_prev + dx_perturb` where `dx_perturb` reuses the
Jacobian linear-solve machinery already built for `jacobian_rows`. Item 2 maps
onto `block_thomas` (Section 4b) rather than dense SVD because the vmec_jax
state (3 fields × mn modes × ns surfaces, O(10^4–10^5)) is too big for dense,
but is *exactly* block-tridiagonal-in-radius in the force map. Accuracy risk:
none — all these produce the same Jacobian to solver tolerance.

## 2. Adjoint methods in stellarator optimization: forward vs reverse regime

Sources:
- Antonsen, Paul, Landreman, shape-gradient adjoints: https://arxiv.org/abs/1812.06154
- Paul et al., gradient-based optimization of 3D MHD equilibria: https://arxiv.org/abs/2012.10028
- Landreman & Paul, adjoint coil shapes: https://arxiv.org/abs/1801.04317
- Neoclassical adjoint: https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/abs/an-adjoint-method-for-neoclassical-stellarator-optimization/1301D99F7EBD027DC9297C61A483FC57

The adjoint literature's headline (Paul et al. 2021): the adjoint reduces the
number of equilibrium evaluations by the *dimension of the parameter space*
(~50–500) relative to FD — i.e. one adjoint solve per **scalar objective**
replaces one solve per **parameter**. The cost calculus:

- Reverse/adjoint: cost ∝ number of objective outputs (rows of J you contract).
- Forward/tangent (what vmec_jax does): cost ∝ number of parameters (24–120).

vmec_jax's least-squares residual has *thousands* of rows (pointwise QS
residuals) but only 24–120 columns, so **forward mode per dof is already the
asymptotically right choice**; a pure adjoint would be slower unless the
residual vector is first compressed (see Section 7, sketching). The one place
reverse mode wins is scalar contractions J^T r (the gradient of the cost):
one adjoint GMRES gives ∇(½‖r‖²) exactly, which is enough for a
gradient-descent/L-BFGS driver but *not* for Gauss-Newton geometry — the
per-iterate step quality of `trf` comes from the full J. So the literature's
verdict for this problem shape: keep forward-mode columns; the savings must
come from making each column cheaper (Sections 4) or from needing J less often
(Section 5). No stellarator adjoint paper reports beating "one linearized solve
per parameter" for a many-residual least-squares Jacobian; the "ALPOpt"-line
near-axis adjoints (Paul/Antonsen) are scalar-objective constructions.

## 3. One-shot / SAND vs NAND

Sources:
- Ta'asan (1991) one-shot origin; Bosse, Gauger, Griewank, "One-Shot Approaches
  to Design Optimization": https://www.researchgate.net/publication/302574112_One-Shot_Approaches_to_Design_Optimzation
- SAND dissertation (Stanford/SOL): https://web.stanford.edu/group/SOL/dissertations/Youngsoo_thesis.pdf
- Unsteady one-shot: https://www.sciencedirect.com/science/article/pii/S0377042715004057
- One-shot with inequality constraints (2026): https://arxiv.org/html/2606.02925

What it is: never fully re-converge the equilibrium per trial point. Advance
state (equilibrium iteration), adjoint, and design updates *simultaneously*,
either as a coupled fixed point ("piggyback"/one-shot) or by giving the
optimizer the constraint `F(z, c) = 0` explicitly (SAND/full-space SQP).
Reported cost for preconditioned one-shot: total optimization ≈ **~4× the cost
of a single simulation** ("bounded retardation", Bosse–Gauger–Griewank), versus
NAND's O(10–100) full solves. That is potentially the largest win of anything
in this survey — a multi-hour campaign collapsing to minutes.

Why it fits unusually well here: the vmec_jax inner solver is itself a
preconditioned quasi-Newton/descent fixed-point iteration on `gc(z, c) = 0`
in JAX (traceable), and `preconditioner_2d.newton_direction` already provides
the Newton step operator. A one-shot loop would interleave: k solver iterations
on z; one adjoint/tangent update; one small trust-region step on c using a
*frozen or Broyden-updated* reduced Jacobian; repeat — with full re-converge
only at continuation-stage boundaries for the FD validation checkpoint.

Risks/effort: this abandons `scipy.optimize.least_squares` as the outer driver
(need a custom TR loop or an augmented-Lagrangian LM), and convergence theory
requires the state iteration to be contractive near the optimum — VMEC-style
solvers can wander (multigrid ladders, m=1 constraint switches), so the
"bounded retardation" guarantee is not automatic. Accuracy is recovered by
final re-converge + exact Jacobian at the last iterate, so the 1e-4 validation
can still be enforced at checkpoints. Effort: high (weeks), payoff: order of
magnitude. Best treated as the long-term follow-up, not the first move.

## 4. Krylov techniques for many RHS and slowly-varying sequences

Sources:
- Parks, de Sturler et al., GCRO-DR: https://epubs.siam.org/doi/10.1137/040607277
- Aerostructural adjoint recycling (GCRO-DR/FGCRO-DR): https://arxiv.org/abs/2309.09925
- Block-GMRES surveys: https://etna.ricam.oeaw.ac.at/vol.33.2008-2009/pp207-220.dir/pp207-220.pdf ,
  https://inria.hal.science/hal-01334648
- Deflation for multiple RHS: https://arxiv.org/pdf/0707.0505
- BCYCLIC (Hirshman et al., JCP 2010): https://www.sciencedirect.com/science/article/pii/S0021999110002536

The implicit-Jacobian solves have two exploitable structures: (i) at one
iterate, all 24–120 RHS share the *same* operator `Fz`; (ii) across
trust-region iterates, `Fz` changes *slowly*.

### 4a. Subspace recycling (GCROT/GCRO-DR) — SOLVAX-native

Parks & de Sturler report ~50% iteration reduction on sequences of slowly
changing systems (first two RHS ~500 iterations, subsequent ~140 each);
the aerostructural adjoint study reports up to **39% fewer matvecs** with
GCRO-DR across an optimization-like sequence. SOLVAX `gcrot` already returns
and accepts the recycle pair `(C, U)` with fixed shapes, so the deflation
space can be threaded (1) across the dof loop within one `jac(x)` — replace
the parallel `chunk_map` with a `lax.scan` carrying `(C, U)`, or a hybrid:
scan over chunks, vmap within — and (2) across iterates, by stashing `(C, U)`
in the Python-side holder dict between `jac_jit` calls (shapes are static).
Since every matvec is a full force JVP, matvec count ≈ wall time.
Expected: 1.3–2× on the Jacobian phase. Accuracy risk: none (solves still hit
`adjoint_tol`; deflation only accelerates). Effort: low–medium (the main cost
is converting the embarrassingly-parallel column batch into a carried scan
without losing the vmap throughput — measure both).

### 4b. Direct block-tridiagonal factorization: factor once, 120 cheap backsolves

This is the VMEC-native answer (precon2d.f + BCYCLIC: "storage of the factored
blocks allows the application of the inverse to multiple right-hand sides")
and the structural analogue of DESC's single SVD. The 1D-preconditioned force
Jacobian is block-tridiagonal in radius (dense 3·mn × 3·mn blocks, ns block
rows — radial coupling is nearest-neighbor FD). Assemble the blocks by colored
JVPs: tangents supported on surfaces {j, j+3, j+6, …} give exact columns for
every third surface simultaneously, so the assembly costs **3 × (3·mn) JVPs**
(~450–1200 for mn ≈ 50–130), independent of dof count. Then
`block_thomas_factor` once and `block_thomas_solve` for all 2·nm RHS
(each backsolve is O(ns·(3mn)²) — microseconds-to-milliseconds, no JVPs).
Compare current cost: 2·nm × GMRES-iters ≈ (48–120) × (10–30) ≈ 500–3600 JVPs
*per jac call*, growing with dof count while assembly does not. At max_mode
4–5 (60 dofs, 120 columns) this wins outright per call; more importantly the
factorization is *reusable*: (i) as an essentially exact preconditioner that
cuts per-column GMRES to 1–3 iterations if you prefer to keep the iterative
outer layer as a correctness guard (recommended: use factored solve as
preconditioner + 1 GMRES residual check → retains the <=1e-4 validation
trivially); (ii) lagged across trust-region iterates (refactor every k
accepted steps, keep GMRES as corrector — cost of intermediate jac calls
collapses to ~2·nm × 2–4 JVPs); (iii) inside `preconditioner_2d` to speed the
*forward* solve's Newton phase too. Caveats: exactness of the tridiagonal
sparsity must be verified once against dense JVP columns (spectral
condensation/`tcon` and lambda coupling are still surface-local; the m=1
constraint and edge rows need the same masking `residual_fn` already applies);
memory for factored blocks ≈ ns × 2 × (3mn)² doubles — fine on CPU at these
sizes. Effort: medium (block assembly + wiring; SOLVAX solver already exists
and is battle-tested per its docs). Expected: 2–5× on the Jacobian phase at
current sizes, better at higher max_mode, plus forward-solve gains.

### 4c. Block/seed GMRES

Block GMRES shares one Krylov space across all RHS (BLAS3-friendly, fewer
matvecs); seed methods (Chan/Wan-style) project subsequent RHS onto the first
solve's space. The literature warns block methods spend more flops per matvec
saved and need deflation of converged directions. Given 4a (recycling) and 4b
(direct) are strictly simpler wins with the same or better matvec economics,
block GMRES is third choice; SOLVAX has no block-GMRES today, so effort is
higher. Skip unless 4a/4b underdeliver.

## 5. Quasi-Newton Jacobian recycling (Broyden secant updates)

Sources:
- Transtrum & Sethna, LM improvements: https://arxiv.org/abs/1201.5885
- Broyden's method background: https://en.wikipedia.org/wiki/Broyden%27s_method
- scipy least_squares docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

What it is: between exact Jacobian evaluations, update
`J_i = J_{i-1} + ((Δr - J_{i-1}Δθ)/‖Δθ‖²) Δθᵀ` from the residual/step pairs
the optimizer already produces; recompute the exact J every k accepted steps
or when the TR ratio degrades. Transtrum & Sethna (their improvement "C"):
Broyden updates "can dramatically speed up the algorithm, requiring many fewer
evaluations of the Jacobian matrix", but the algorithm "appears to be more
likely to get lost" — a direct accuracy-risk statement; they recommend smarter
re-evaluation triggers, and note the update degrades for larger parameter
counts. scipy's `least_squares` has **no native support** (its `jac` callable
is stateless from scipy's perspective, but nothing stops the callable from
closing over history: `fun` results can be cached by the wrapper, so a
stateful `jac` that returns either the secant-updated matrix or a fresh exact
one is a pure-Python wrapper around the existing `jac_jit`). Expected win: if
jac(x) is ~half the iterate cost, skipping 2 of every 3 exact evaluations
saves ~30–35% wall time. Risk: the *iterate-level* gradients are no longer
FD-validated to 1e-4 (only refresh iterates are); TR machinery tolerates this
but continuation-stage endpoints should force a refresh. Effort: low (a
~60-line wrapper + heuristics). Good "stacking" candidate on top of Section 4
since it multiplies whatever the per-jac cost becomes.

## 6. VMEC++ hot restart

Sources:
- https://www.proximafusion.com/press-news/introducing-vmecpp-open-source-software-for-fusion-research
- https://github.com/proximafusion/vmecpp

VMEC++'s "hot restart" passes the previous converged output as the initial
state so a perturbed-boundary run "takes very few iterations to converge";
they market it specifically for optimization loops ("dramatically decreases
runtimes when analyzing many similar equilibria") but publish no hard
speedup factor for optimization campaigns. vmec_jax already implements the
equivalent (`hot_restart=True`, `_HOT_CACHE` in implicit.py; `state_holder`
in the FD path). Two residual gaps worth closing: (i) the duplicate
`fun`/`jac` solve at the same x — a memoized one-entry cache (DESC's
`f_where_x` pattern) removes one full-solve *entry* per accepted iterate
(even hot-restarted re-solves pay convergence-check and callback overhead);
(ii) DESC-style *perturbation* warm start — first-order correction
`z_seed = z_prev - (dF/dz)^{-1} dF/dc Δc` using the Section-4b factorization,
which VMEC++ does not do and which should cut the inner iterations of every
*trial* solve (the dominant `fun` cost during TR step rejection) well below
plain hot restart. Effort: (i) trivial; (ii) small once 4b exists. Risk: none
for (i); (ii) only changes the initial guess, so converged answers unchanged.

## 7. Other promising directions

### 7a. Randomized subspace / sketched Gauss-Newton

- R-SGN: https://arxiv.org/abs/2211.05727 (TR variant with high-probability
  convergence rates matching deterministic order);
- variable-dimension sketching: https://arxiv.org/abs/2506.03965 ;
- inexact GN by sampling: https://arxiv.org/abs/2310.05501

Sketch the *column* space: per iteration draw S (k × 2nm, k ≈ nm/2 or less),
compute only J·S (k linearized solves instead of 2nm), take the TR step in the
sketched subspace. Directly reduces the dominant per-iterate cost by 2nm/k but
typically increases outer iteration count; net wins of 1.5–3× are reported on
overdetermined problems. Risk: stochastic steps interact badly with the strict
FD-validation story and with continuation reproducibility; scipy can't do this
natively (custom TR loop needed). Effort medium-high. Rank below 4a/4b/5.

### 7b. Matrix-free trust region with LSMR (`tr_solver='lsmr'` + LinearOperator jac)

scipy `least_squares` accepts a `jac` callable returning a
`scipy.sparse.linalg.LinearOperator`; with `tr_solver='lsmr'` the TR subproblem
only needs `Jv`/`Jᵀu` products — one tangent GMRES per `Jv`, one adjoint GMRES
per `Jᵀu` (the custom-VJP path already exists). Total per iterate ≈
2 × (LSMR iters) GMRES solves; beats forming J only if LSMR needs
< nm iterations, which is plausible at 60 dofs with good `x_scale` but
unproven, and scipy re-evaluates products across internal λ retries.
Zero accuracy risk (products are exact). Effort: low-medium. Worth a
half-day benchmark, but expected win is modest and could be negative.

### 7c. Parallel FD baseline (cheap insurance)

simsopt's `MPIFiniteDifference` distributes FD Jacobian columns over workers
(https://simsopt.readthedocs.io/en/latest/mpi.html). On a 36-core box, 24–60
hot-restarted FD solves parallelize to ~2 wall-solves per Jacobian. This is a
sanity baseline: any implicit path should beat *parallel* FD, not serial FD.
The implicit path's analogue is process-level parallelism over GMRES column
chunks — but 4b makes this moot.

---

## 8. Ranked shortlist — (expected wall-clock win) × (confidence) / (effort)

### 1. Amortized block-tridiagonal factorization for the implicit Jacobian (+ reuse everywhere)

Win ~2–5× on the jac phase now, growing with max_mode; confidence high
(exactly VMEC2000 precon2d/BCYCLIC and morally DESC's one-SVD-many-RHS,
both proven in production); effort medium. Sketch: add a
`assemble_blocks(F, z_star, params)` helper next to
`preconditioner_2d.newton_direction` that builds the ns × (3mn × 3mn)
lower/diag/upper blocks of `dF/dz` by 3-colored `jax.jvp` probes of the
existing `residual_fn` (tangents = within-surface unit vectors replicated on
surfaces j, j+3, …; reuse the dof mask/edge masking already applied there);
validate the sparsity once against dense columns at max_mode 1. In
`jacobian_rows`, replace the per-column `_adjoint_solve(Fz, -b)` with
`block_thomas_factor` once + `block_thomas_solve` over the stacked RHS —
and keep one preconditioned-GMRES pass (preconditioner = factored solve,
maxiter 2–3) as a residual-checked corrector so gradients stay exact to
`adjoint_tol` and the 1e-4 FD validation is preserved by construction.
Then lag the factorization across accepted iterates (refactor every 2–4
jac calls or when the corrector iteration count creeps up), and feed the same
factorization to (a) `preconditioner_2d` for the forward solve's Newton phase
and (b) a DESC-style first-order perturbation seed for trial solves.

### 2. Kill redundant solves: converged-state memoization + perturbation warm start (DESC's `f_where_x` + `eq.perturb` pattern)

Win ~1.5–2× on the fun/solve phase; confidence very high (DESC does exactly
this, source-verified; scipy's call pattern guarantees jac's x equals the last
accepted fun x); effort trivial-to-small. Sketch: in
`_least_squares_implicit`, hold `{"x": last_x, "state": z*}` in the existing
`holder` dict; make `jacobian_rows` accept an optional pre-solved frozen state
via the host-solve cache keyed on the exact dof vector (extend `_HOT_CACHE` to
a keyed one-entry memo: if `params` hash matches the last converged solve,
return it without re-iterating — the analogue of DESC's
`_allx/_allxopt/_allxeq` lists). Second step (needs shortlist item 1's
factorization or one extra GMRES): seed each *trial* solve with
`z_prev - (dF/dz)^{-1}(dF/dp)Δp` instead of plain `z_prev`, which is DESC
Part II's first-order perturbation and should cut inner iterations on every
TR trial, accepted or rejected. No accuracy risk: only initial guesses and
cache hits change; converged states are identical to solver tolerance.

### 3. GCROT deflation-space recycling across Jacobian columns and iterates

Win ~1.3–2× on the jac phase (literature: 39–50% matvec reductions on exactly
this "many RHS + slowly varying operator" pattern); confidence medium-high
(SOLVAX `gcrot` already exposes the fixed-shape recycle pair, so the mechanism
exists; the open question is how much recycling survives the vmap-vs-scan
throughput tradeoff on CPU); effort low-medium. Sketch: in `jacobian_rows`,
switch the column map from parallel `chunk_map` to `jax.lax.scan` over dof
chunks carrying `(C, U)` (vmap *within* a chunk, recycle *between* chunks —
first chunk builds the deflation space, later chunks converge in fewer
iterations); return the final `(C, U)` from `jac_jit` and stash it in the
Python-side holder so the next trust-region iterate's first chunk starts
deflated (operator drift is small between accepted steps — same regime as the
aerostructural GCRO-DR study). Solves still run to `adjoint_tol`, so accuracy
is untouched; benchmark matvec counts per column with/without recycling at
max_mode 3 and 5 before committing. Stackable with shortlist 1 (recycling then
accelerates the corrector) and with a Broyden-style "reuse J every other
iterate" wrapper (Section 5) if more is needed.

---

Runner-up: Broyden secant updates between exact Jacobians (Section 5) — lowest
effort of all, ~30% win, but the only candidate that *weakens* the per-iterate
1e-4 gradient guarantee, so it goes in only after 1–3 land. Long-term:
one-shot/SAND (Section 3) is the order-of-magnitude play (~4× cost of a single
solve for a whole optimization, per Bosse–Gauger–Griewank) and fits the
traceable JAX fixed-point solver unusually well, but requires replacing the
scipy driver and new convergence safeguards.
