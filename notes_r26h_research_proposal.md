# R26h — Literature + code deep-dive: proposal for vmec_jax

**Status: PROPOSAL — nothing here is implemented. Per plan.md R26h, this document is to be
reviewed by the user before any implementation.**

Date: 2026-07-11. Method: parallel web survey of DESC (docs + source layout + papers),
SIMSOPT, VMEC++, downstream diagnostics (BOOZ_XFORM / NEO / Γ_c / SFINCS / COBRA /
TERPSICHORE / BNORM / REGCOIL), the 2022–2026 stellarator-optimization literature, and the
3D-MHD numerics literature — cross-checked against the current vmec_jax code
(`vmec_jax/core/`: `optimize.py` 1632 L, `implicit.py`, `coils.py`, `freeboundary_diff.py`,
`boozer.py`/booz_xform_jax, `nyquist.py` mercier port, `preconditioner_2d.py`).

## 0. Where vmec_jax stands (gap analysis)

Already in-tree: fixed + free boundary (NESTOR **and** differentiable direct-coil virtual
casing with `value_and_grad_bnormal`), implicit-diff adjoints (O(1) memory), Boozer transform
(booz_xform_jax), `QuasisymmetryRatioResidual`, a distilled Goodman-style
`quasi_isodynamic_residual`, aspect/iota/mirror/magnetic-well/`l_grad_b` targets, DMerc in
wout, 1D + opt-in 2D preconditioner, spline profiles, hot restart, near-axis seeding,
least-squares driver with `jac="implicit"`. Planned (R26g): differentiable Redl bootstrap +
self-consistency loop.

The competitive landscape in one sentence each:

- **DESC** is the only other differentiable-equilibrium ecosystem and already ships AD
  versions of *every* physics gap on our list except turbulence proxies: general omnigenity,
  ballooning, Mercier/well, Redl bootstrap, ε_eff, Γ_c, free-boundary-as-objective, 13+ coil
  objectives ([objective sources](https://github.com/PlasmaControl/DESC/tree/master/desc/objectives);
  individual physics objectives are small, ~10–30 KB files, once shared infrastructure exists).
- **SIMSOPT**'s coil side is fully differentiable, but its **entire VMEC physics side is
  finite-difference** ([docs](https://simsopt.readthedocs.io/latest/)) — the exact hole
  vmec_jax fills.
- **VMEC++** ([arXiv:2502.04374](https://arxiv.org/abs/2502.04374),
  [repo](https://github.com/proximafusion/vmecpp)) modernizes robustness/speed
  (hot restart, JSON, zero-crash — all of which vmec_jax already matches) but has **no GPU and
  no differentiability on its public roadmap**; Proxima leans on ML surrogates
  (ConStellaration, [arXiv:2506.19583](https://arxiv.org/abs/2506.19583)) where vmec_jax
  offers exact gradients.

**Positioning thesis for everything below:** vmec_jax can be the first code to provide the
modern optimization-objective stack *with exact gradients on VMEC's own trusted, wout-compatible
representation* — every VMEC-based optimization published today (SIMSOPT/STELLOPT/ROSE
pipelines) uses finite differences for these quantities.

---

## 1. TOP 5 recommended (ranked by impact / effort)

### #1 — Ideal-MHD stability objectives: infinite-n ballooning + differentiable Mercier / magnetic well

- **What.** (a) Port COBRA's infinite-n ideal-ballooning solve — a 1D second-order ODE
  eigenvalue problem `d/dζ(g dX/dζ) + cX = λ fX` along field lines — as a batched symmetric
  eigenproblem (`jax.numpy.linalg.eigh` over (surface, α, θ₀) triplets), objective = soft-max
  of γ² over field lines. (b) Repackage the existing `nyquist.py` Mercier port (DMerc is
  already computed for wout) plus `magnetic_well` as first-class differentiable objective
  terms for the `least_squares` driver, with per-surface bounds à la DESC.
- **Why it matters.** MHD stability is the standard co-constraint in every modern optimized
  configuration (Landreman–Buller–Drevlak 2022; Gaur et al. 2024 "omnigenous equilibria with
  enhanced stability"). No VMEC-lineage code offers ballooning with exact gradients; SIMSOPT
  calls COBRAVMEC with FD, and industry practice (e.g. Thea Energy) still calls TERPSICHORE
  externally. This unlocks "stable by construction" optimization campaigns.
- **Prior art.** COBRA: Sanchez et al., JCP 161, 576 (2000)
  ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0021999100965148)),
  VMEC-coordinate version CPC (2001),
  [STELLOPT page](https://princetonuniversity.github.io/STELLOPT/COBRAVMEC.html).
  Adjoint/AD: Gaur et al., JPP 89 (2023), [arXiv:2302.07673](https://arxiv.org/abs/2302.07673)
  (self-adjointness ⇒ cheap dγ²/dparams; Newcomb-metric variant avoids the eigensolve).
  DESC: `BallooningStability`, `MercierStability`, `MagneticWell` in `_stability.py` (18.8 KB
  total; [tutorial](https://desc-docs.readthedocs.io/en/stable/notebooks/tutorials/ideal_ballooning_stability.html));
  applied at scale in [arXiv:2410.04576](https://arxiv.org/abs/2410.04576) (PPCF 2025).
  Mercier/well references: Landreman & Jorge, JPP 86 (2020),
  [arXiv:2006.14881](https://arxiv.org/abs/2006.14881).
- **Effort.** **S–M (~2–4 weeks).** Mercier/well objectives are days (quantities exist; needs
  routing through `_traceable_term`). Ballooning needs field-line geometry arrays from the
  interior solution (metric elements are already available in VMEC coordinates — COBRA was
  designed to consume exactly these) + a batched `eigh`; DESC's whole `_stability.py` is
  smaller than our `optimize.py`.
- **Differentiability.** Fully feasible: flux-surface averages are closed-form; the eigenvalue
  is differentiable via Hellmann–Feynman/self-adjointness, and `jnp.linalg.eigh` is
  reverse-mode differentiable in JAX (DESC differentiates it directly). Works with the
  existing implicit-diff pipeline (interior-state objective, same pattern as
  `QuasisymmetryRatioResidual.residuals_state`).

### #2 — General omnigenity objective (Dudt) + constructed-QI-target residual (Goodman)

- **What.** Replace/augment the current distilled `quasi_isodynamic_residual` with the two
  state-of-the-art formulations: (a) **Dudt-style general omnigenity** — parameterize a target
  omnigenous field by a monotonic-spline well `B(ρ,η)` and a Chebyshev–Fourier map
  `h(ρ,η,α)`, with helicity (M,N) selecting OP/OH/OT classes; residual = weighted mismatch of
  the equilibrium's Boozer |B| against the target on an (η,α) grid, with the target's
  parameters co-optimized. (b) **Goodman-style constructed QI target** — build the nearest
  exactly-QI |B|(θ,φ) per surface (poloidally closed contours, aligned maxima, matched bounce
  distances) and least-squares against it, plus flat-mirror (Velasco) and pwO variants as
  cheap algebraic add-ons.
- **Why it matters.** This directly fixes the README's own documented weakness — "QI …
  not precise — needs richer omnigenity residual" (QI plateaus at ~1e-2 vs 1e-5 for QS). QI is
  the reactor-relevant class (W7-X lineage, CIEMAT-QI4, Proxima's Stellaris); a precise,
  exact-gradient QI residual on VMEC equilibria exists nowhere today.
- **Prior art.** Dudt et al., JPP 90 (2024), [arXiv:2305.08026](https://arxiv.org/abs/2305.08026);
  DESC `Omnigenity` + `FixOmniBmax/Well/Map` (`_omnigenity.py`, 31.5 KB incl. QS objectives;
  [tutorial](https://desc-docs.readthedocs.io/en/stable/notebooks/tutorials/omnigenity.html)).
  Goodman et al., JPP 89 (2023), [arXiv:2211.09829](https://arxiv.org/abs/2211.09829); PRX
  Energy 3, 023010 (2024), [arXiv:2405.19860](https://arxiv.org/abs/2405.19860).
  Flat mirror: Velasco et al., NF 63, 126038 (2023),
  [arXiv:2306.17506](https://arxiv.org/abs/2306.17506). Piecewise omnigenity: PRL 133, 185101
  (2024), [arXiv:2405.07634](https://arxiv.org/abs/2405.07634) (+ combined omnigenity+pwO,
  arXiv:2603.12139). Goodman/CIEMAT pipelines today run on VMEC + derivative-free/FD — exact
  gradients are the novelty.
- **Effort.** **M (~3–5 weeks).** The Boozer transform is in-tree; needs JAX monotonic
  splines, the (η,α)↔Boozer coordinate map, and target-parameter dofs threaded through
  `least_squares` (a second optimizable parameter set — the main plumbing item). The Goodman
  constructed-target variant reuses the existing `_qi_grid` machinery.
- **Differentiability.** Fully feasible — both residuals are algebraic functions of Boozer
  spectra (DESC proves the pattern in JAX). Compatible with `jac="implicit"`.

### #3 — Differentiable bounce-averaging module: ε_eff, Γ_c (and available energy)

- **What.** A `core/bounce.py` implementing field-line following on a flux surface +
  AD-safe bounce-point detection + spectral/Simpson bounce quadrature, then three consumers:
  (a) **ε_eff^{3/2}** (Nemov 1999, the NEO quantity — the single most-used neoclassical
  figure of merit), (b) **Γ_c** fast-ion proxy (Nemov 2008 and Velasco 2021 forms), and
  optionally (c) **available energy of trapped electrons** (Mackenbach) as a TEM proxy.
  All are flux-surface-local: they need only |B|, ∇ψ and drift geometry on the surface —
  computable from the Boozer spectra vmec_jax already produces.
- **Why it matters.** ε_eff and Γ_c are *the* workhorse confinement objectives of modern
  stellarator optimization (Wistell-B, CIEMAT-QI, Stellaris assessments all use them); today
  they require external NEO/ROSE/KNOSOS runs with FD gradients in any VMEC workflow. This is
  the highest-research-impact item on the list — it turns vmec_jax into a self-contained
  neoclassical + fast-ion optimization code.
- **Prior art.** Blueprint paper: Unalmis, Gaur, Conlin, Panici, Kolemen, "Spectrally
  accurate, reverse-mode differentiable bounce-averaging algorithm," JPP 2026,
  [arXiv:2412.01724](https://arxiv.org/abs/2412.01724) — first direct reverse-mode ε_eff
  optimization of a finite-β stellarator. DESC `EffectiveRipple` (`_neoclassical.py`, 9.4 KB
  wrapper) and `GammaC` (`_fast_ion.py`, 10.6 KB) over the `desc.integrals` Bounce1D/Bounce2D
  machinery ([EffectiveRipple docs](https://desc-docs.readthedocs.io/en/latest/_api/objectives/desc.objectives.EffectiveRipple.html),
  [GammaC docs](https://desc-docs.readthedocs.io/en/latest/_api/objectives/desc.objectives.GammaC.html)).
  Physics: Nemov et al., PoP 6, 4622 (1999); Nemov et al., PoP 15, 052501 (2008); Velasco et
  al., NF 61, 116059 (2021), [arXiv:2106.05697](https://arxiv.org/abs/2106.05697);
  Mackenbach et al., PRL 128, 175001 (2022), [arXiv:2109.01042](https://arxiv.org/abs/2109.01042).
- **Effort.** **L (~6–10 weeks).** The objective wrappers are small but the bounce-integral
  library is real work (well detection under AD, pitch quadrature, `num_transit`/`num_well`
  batching; DESC's docs candidly flag Jacobian memory as the binding constraint — our
  `jac_chunk_size` machinery and O(1) adjoint help here). Do ε_eff first (simplest consumer),
  Γ_c second, AE third.
- **Differentiability.** Proven feasible (arXiv:2412.01724 is exactly this in JAX). Bounce
  points need care (implicit-function treatment at turning points); Nemov's Γ_c form converges
  with transits, Velasco's has a secular term — implement Nemov's as default (DESC ships both).

### #4 — Single-stage plasma–coil optimization with exact gradients

- **What.** Couple the existing differentiable pieces into a single-stage objective
  `J = J_plasma(equilibrium) + w·∫(B_coil·n − B_plasma·n)² dS + J_coil-regularizers`:
  `coils.py` (Fourier curves + Biot–Savart, differentiable),
  `freeboundary_diff.value_and_grad_bnormal` (virtual casing), and the implicit-diff
  equilibrium gradients. Add the standard coil regularizer set (length, curvature, coil–coil /
  coil–surface distance, linking number — each a few dozen lines in JAX) and a driver that
  optimizes boundary + coil dofs jointly.
- **Why it matters.** Single-stage optimization is the field's current frontier; every
  VMEC-based single-stage effort (Jorge et al. 2023) is bottlenecked by FD equilibrium
  gradients — vmec_jax removes exactly that bottleneck, and would be the **first
  exact-gradient single-stage VMEC**. It also showcases the direct-coil free-boundary lane
  that is already a unique vmec_jax feature.
- **Prior art.** Jorge, Goodman, Landreman, Rodrigues, Wechsung, PPCF 65, 074003 (2023),
  [arXiv:2302.10622](https://arxiv.org/abs/2302.10622). Giuliani et al., JCP (2022),
  [arXiv:2010.02033](https://arxiv.org/abs/2010.02033) + Boozer-surface variant
  [arXiv:2203.03753](https://arxiv.org/abs/2203.03753). Wechsung et al., PNAS 119 (2022),
  [DOI 10.1073/pnas.2202084119](https://www.pnas.org/doi/10.1073/pnas.2202084119).
  Differentiable coil proxy (quasi-single-stage): Fu et al.,
  [arXiv:2510.16243](https://arxiv.org/abs/2510.16243) (QUADCOIL,
  [arXiv:2408.08267](https://arxiv.org/abs/2408.08267)). DESC coil objectives:
  `_coils.py` (96.7 KB, 13+ objectives) + `QuadraticFlux`
  ([stage-two tutorial](https://desc-docs.readthedocs.io/en/stable/notebooks/tutorials/coil_stage_two_optimization.html)).
  SIMSOPT coil objectives (analytic derivatives): `CurveLength`, `LpCurveCurvature`,
  `CurveCurveDistance`, `SquaredFlux`, … ([geo docs](https://simsopt.readthedocs.io/v1.8.2/simsopt_user.geo.html)).
- **Effort.** **M (~4–6 weeks).** The hard 80% (differentiable Biot–Savart, virtual casing,
  implicit equilibrium gradients) already exists in-tree; the work is coil regularizers,
  the joint parameter vector, and an example campaign (e.g. reproduce a Landreman–Paul-class
  QA with coils, tying into the planned R26f README example).
- **Differentiability.** Already demonstrated piecewise in-tree (`value_and_grad_bnormal`,
  FD-validated free-boundary coil derivatives); the composition is straightforward JAX.

### #5 — Turbulence proxies: ITG critical gradient, flux compression, L∇B packaging

- **What.** A small family of cheap, differentiable turbulence objectives: (a)
  **Roberg-Clark ITG critical gradient** a/L_T,crit from coarse-grained gyrokinetics —
  closed-form in flux-tube geometry arrays (|∇ψ|, curvature drift, local shear, parallel
  connection length); (b) **flux-surface compression in bad curvature** (the
  Landreman/Buller/Drevlak geometric proxy, also a ConStellaration benchmark objective);
  (c) first-class packaging of the existing `l_grad_b` (Kappel/Landreman coil-buildability
  metric) and an `Elongation` target; (d) later, available energy via the #3 bounce module.
  Optionally add a `vmec_fieldlines`-equivalent that emits GX/GS2/stella geometry arrays
  (differentiable), positioning vmec_jax as the geometry front-end for gyrokinetics-in-the-loop
  (SPSA outer loops à la Kim et al.).
- **Why it matters.** Turbulent transport is the dominant loss channel in optimized
  stellarators, and this is the one physics area where **DESC has no native objective**
  (it couples to GX via SPSA/FD only) — a JAX-native ITG proxy would exceed every existing
  code, at low cost.
- **Prior art.** Roberg-Clark et al., PRR 5, L032030 (2023),
  [arXiv:2301.06773](https://arxiv.org/abs/2301.06773) (+
  [arXiv:2208.05727](https://arxiv.org/abs/2208.05727),
  [arXiv:2210.16030](https://arxiv.org/abs/2210.16030), QI extension
  [arXiv:2506.22166](https://arxiv.org/abs/2506.22166)). Mackenbach AE: PRL 128, 175001
  (2022). GX-in-the-loop: Kim et al., JPP 90 (2024),
  [arXiv:2310.18842](https://arxiv.org/abs/2310.18842); GX
  [arXiv:2209.06731](https://arxiv.org/abs/2209.06731). L∇B: Kappel/Landreman,
  [arXiv:2309.11342](https://arxiv.org/abs/2309.11342). ConStellaration objectives:
  [arXiv:2506.19583](https://arxiv.org/abs/2506.19583). SIMSOPT `vmec_compute_geometry` /
  `vmec_fieldlines` ([mhd docs](https://simsopt.readthedocs.io/v1.8.2/simsopt_user.mhd.html)).
- **Effort.** **S–M (~2–4 weeks)** for (a)–(c) — field-line geometry arrays from the interior
  solution + algebra; no eigensolves, no bounce integrals. AE (d) rides on #3.
- **Differentiability.** Fully feasible: all closed-form geometry algebra (the proxies were
  *designed* to avoid gyrokinetic solves).

### Note on Redl bootstrap (R26g — already committed, survey validates it)

Not ranked above because it is already planned, but the survey sharpens the approach: the
Redl formula needs only trapped fraction (flux-surface integrals of B), profiles, iota, G/I —
trivially JAX-portable; DESC's whole `_bootstrap.py` is 10.6 KB. Two consistency modes exist:
SIMSOPT-style outer fixed-point iteration
(`VmecRedlBootstrapMismatch`, [mhd docs](https://simsopt.readthedocs.io/v1.8.2/simsopt_user.mhd.html))
vs DESC-style *in-optimizer* residual (`BootstrapRedlConsistency`,
[tutorial](https://desc-docs.readthedocs.io/en/v0.15.0/notebooks/tutorials/bootstrap_current.html)).
**Recommendation:** implement the residual form (and, uniquely, collapse the outer loop into
one coupled residual via the existing implicit-diff machinery — no other code can), keeping
an outer-iteration mode for parity with arXiv:2205.02914 / the Zenodo benchmark. Kinetic-profile
inputs (n_e, T_e, T_i, Z_eff) need a small profile-plumbing extension. Refs: Redl et al., PoP
28, 022502 (2021); Landreman, Buller, Drevlak, PoP 29, 082501 (2022),
[arXiv:2205.02914](https://arxiv.org/abs/2205.02914); Landreman's
[AC-profile note](https://terpconnect.umd.edu/~mattland/assets/notes/computing_vmec_AC_profile_from_a_bootstrap_current_code.pdf).

---

## 2. Everything else considered

Effort: S ≲ 2 wk, M ≈ 2–6 wk, L ≳ 6 wk, XL ≳ 3 mo (in vmec_jax's architecture).
Diff = differentiability feasibility.

### A. New research-grade functionality

| Candidate | What / why | Prior art (URLs) | Effort | Diff | Verdict |
|---|---|---|---|---|---|
| Greene's-residue / island-quality diagnostic | Trace the (virtual-casing-extended) field with a differentiable ODE integrator, Newton-find island X/O points, return residues — a nested-surfaces-quality objective without changing the solver | simsopt+SPEC residue optimization ([arXiv:2111.15564](https://arxiv.org/abs/2111.15564)); single-stage island healing, PoP 32, 012504 (2025) ([AIP](https://pubs.aip.org/aip/pop/article/32/1/012504/3331878)) | M | Yes (diffable ODE + implicit Newton) | Strong runner-up; natural after #4 |
| Guiding-center / field-line tracing + loss fractions | ALPHA-style collisionless α-loss objective; Poincaré plots as a product feature | simsopt tracing ([docs](https://simsopt.readthedocs.io/v1.5.0/tracing.html), no gradients); Bindel-Landreman-Padidar direct loss optimization ([arXiv:2302.11369](https://arxiv.org/abs/2302.11369)); CATAPULT GPU tracer (arXiv:2604.07617) | L | Partial (tracing yes; loss fraction non-smooth — validation role) | Later; Γ_c (#3) is the differentiable stand-in |
| REGCOIL / winding-surface current potential in-loop | Tikhonov-regularized linear solve for coil complexity-aware stage-1; "quasi-single-stage" | Landreman NF 57, 046003 (2017) ([arXiv:1609.04378](https://arxiv.org/abs/1609.04378)); QUADCOIL ([arXiv:2408.08267](https://arxiv.org/abs/2408.08267), [arXiv:2510.16243](https://arxiv.org/abs/2510.16243)); DESC AD surface currents ([arXiv:2508.09321](https://arxiv.org/abs/2508.09321)) | M (linear lsq + implicit diff — SOLVAX-shaped) | Yes | Good follow-on to #4; QUADCOIL is the modern form |
| ANIMEC-style anisotropic pressure (p∥, p⊥) | Bi-Maxwellian/slowing-down hot species for NBI/ICRF studies; force balance gains σ = 1+(p∥−p⊥)/B² factors — modest delta on the moment/force kernels | Cooper et al., CPC 180, 1524 (2009) ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0010465509001179)); EPFL reports | L (touches `forces.py` core; parity testing burden) | Yes in principle | Defer — niche vs confinement objectives; revisit on demand |
| Toroidal-flow equilibria | Centrifugal-shift physics (tokamak hybrid/snake studies) | Cooper VMEC-with-flow adaptations; VEQ-R (arXiv:2602.11422). "SATIRE" could **not** be confirmed as a real code — do not cite | L | Yes in principle | Defer — weak stellarator demand |
| TERPSICHORE-class low-n global stability | Full 3D low-n eigenproblem | NCSX validation ([PPPL report](https://ncsx.pppl.gov/pvr/Physics_Validation_Report/Chapter_5.pdf)) | XL | Hard (large generalized eigenproblem) | Reject for now; Mercier+ballooning (#1) covers the optimization use case |
| Near-axis co-optimization (JAX pyQSC/pyQIC) | Differentiate the NAE→boundary map so NAE params and boundary co-optimize in one AD graph; r² diagnostics (B20 variation, L∇∇B, r_c) as objectives | Landreman NAE database JPP 88 (2022) ([arXiv:2209.11849](https://arxiv.org/abs/2209.11849)); Rodríguez-Plunk QI ([arXiv:2303.06038](https://arxiv.org/abs/2303.06038), [arXiv:2409.20328](https://arxiv.org/abs/2409.20328)); DESC `FixNearAxisR/Z/Lambda` ([tutorial](https://desc-docs.readthedocs.io/en/stable/notebooks/tutorials/nae_constraint.html)) | M (small ODEs + algebra; seeding exists) | Yes | Attractive second-wave item; nobody has it end-to-end |
| Reactor scalar objectives | `FusionPower`, ISS04 confinement scaling — reactor-relevant scalars from kinetic profiles | DESC `_power_balance.py` (10 KB) | S | Yes | Cheap add-on once kinetic profiles land with R26g |
| Fixed-point/permanent-magnet stage-2 | PM grids + greedy optimizers | Kaptanoglu GPMO ([arXiv:2208.10620](https://arxiv.org/abs/2208.10620), arXiv:2512.14997) | L | Partial (greedy is discrete) | Reject — simsopt serves this; not equilibrium-side |
| Divertor/edge targets | Island-width / non-resonant divertor metrics | NF 2025 (DOI 10.1088/1741-4326/addb5d); island-divertor shape opt (arXiv:2602.24049) | L | Partial (tracing-based) | Defer; iota-profile resonance-avoidance targets are the cheap differentiable slice |
| ML/surrogate hooks | Exact-gradient training signal for surrogates/PINNs; ConStellaration/QUASR benchmark drop-ins; physics-consistent decoder for generative design | ConStellaration ([arXiv:2506.19583](https://arxiv.org/abs/2506.19583)); QUASR ([arXiv:2409.04826](https://arxiv.org/abs/2409.04826), [browser](https://quasr.flatironinstitute.org)); diffusion design (arXiv:2511.20445); W7-X NN surrogates ([arXiv:2211.09743](https://arxiv.org/abs/2211.09743)) | S (benchmark adapters) | n/a | Do the ConStellaration adapter as a visibility play |

### B. Better implementations of existing functionality (numerics)

| Candidate | What / why | Prior art (URLs) | Effort | Diff | Verdict |
|---|---|---|---|---|---|
| Anderson acceleration / NGMRES wrap of the descent step | Windowed AA around the existing preconditioned-Richardson iteration; theory says AA on preconditioned Richardson ≈ right-preconditioned GMRES. **No published AA-on-VMEC exists — claimable novelty**, and it may beat the 2D preconditioner's wall-clock economics (R23) since it adds no HVP/GMRES cost | Walker & Ni SINUM 49 (2011) ([PDF](https://users.wpi.edu/~walker/Papers/Walker-Ni,SINUM,V49,1715-1735.pdf)); [arXiv:2007.01996](https://arxiv.org/abs/2007.01996); NGMRES-on-Richardson theory ([arXiv:2603.25983](https://arxiv.org/abs/2603.25983)) | S–M (SOLVAX-shaped; opt-in like prec2d) | Preserves implicit-diff (fixed point unchanged) | **Recommend trying first among numerics** |
| Batched/vmapped equilibrium ensembles | `vmap` the solver over boundary batches for GPU parameter scans / dataset generation — **no published batched-equilibrium paper exists**; pairs with ConStellaration | JAX vmap practice; DESC GPU numbers ([arXiv:2204.00078](https://arxiv.org/abs/2204.00078)) | M (uniform-shape solve path exists in `--mode jit`) | Yes | Recommend; a paper-able capability |
| Hirshman–Breslau explicit m=1 constraint (`lconm1`) | VMEC's default m=1 spectral-condensation constraint is "an unknown mixture" (VMEC++ paper); the 1998 explicit optimal-angle construction is cleaner and documented to affect convergence | Hirshman & Breslau, PoP 5, 2664 (1998) ([ADS](https://ui.adsabs.harvard.edu/abs/1998PhPl....5.2664H/abstract)); Hirshman & Meier PF 28, 1387 (1985); [VMEC++ numerics](https://arxiv.org/abs/2502.04374) | S (opt-in flag; keep parity default) | Yes | Recommend as an opt-in lane + docs note |
| Axis-resolution fix | VMEC's documented near-axis force-error pathology (convergence exponent ~−1 vs −2); m-dependent axis extrapolation/regularization repairs | APS-DPP 2023 "VMEC Convergence Near the Magnetic Axis" ([ADS](https://ui.adsabs.harvard.edu/abs/2023APS..DPPGP1115H/abstract)); Panici et al. Part I ([arXiv:2203.17173](https://arxiv.org/abs/2203.17173)); free-boundary Shafranov-shift verification ([OSTI](https://www.osti.gov/servlets/purl/2573180)) | M (opt-in; parity default untouched) | Yes | Recommend investigating; the main honest VMEC-vs-DESC accuracy gap |
| Henneberg boundary representation | Unique, non-degenerate boundary parameterization — removes null directions in boundary optimization | Henneberg, Helander, Drevlak, JPP 87 (2021) ([arXiv:2105.00768](https://arxiv.org/abs/2105.00768)); simsopt `SurfaceHenneberg` | S–M (dof-packing layer only) | Yes | Nice-to-have for optimization conditioning |
| DESC-style perturbation/continuation + deflation | 2nd/3rd-order Taylor warm starts, continuation in β/boundary-ratio; deflation to find multiple equilibrium branches | Conlin et al. Part II ([arXiv:2203.15927](https://arxiv.org/abs/2203.15927)); deflation ([arXiv:2602.09957](https://arxiv.org/abs/2602.09957), [DESC tutorial](https://desc-docs.readthedocs.io/en/stable/notebooks/tutorials/deflation.html)) | S–M (1st-order warm start already landed per R25; extend order + expose continuation API) | Yes | Partially done; finish as R25 follow-on |
| Mixed-precision solves (GMRES-IR) | fp32/bf16 descent + preconditioner factorization, fp64 polish; tolerates κ≈1e8 — **unexplored in MHD equilibrium** | Carson & Higham SISC 40 (2018) ([SIAM](https://epubs.siam.org/doi/10.1137/17M1122918)); survey IJHPCA 35 (2021) ([Sage](https://journals.sagepub.com/doi/10.1177/10943420211003313)) | M | Care needed (parity gates in fp64) | Worth a benchmark spike, GPU-motivated |
| NESTOR alternative: boundary-error-as-objective or high-order BIE | DESC free boundary needs one singular integral per residual (no dense exterior Neumann solve); BIEST offers spectral accuracy | Conlin et al. ([arXiv:2412.05680](https://arxiv.org/abs/2412.05680)); BIEST ([arXiv:1902.01205](https://arxiv.org/abs/1902.01205)); Merkel JCP 66, 83 (1986) | L | Yes (the point of the DESC formulation) | Defer; R26c tunes NESTOR first — revisit if it stays the bottleneck |
| GVEC-style radial B-splines / Zernike basis | Higher-order radial accuracy | GVEC PPCF 67, 045002 (2025) ([IOP](https://iopscience.iop.org/article/10.1088/1361-6587/adba11), [repo](https://github.com/gvec-group/gvec)); DESC Part I | XL — breaks iteration-for-iteration VMEC parity, the product's core promise | Yes | **Reject** (positioning: that's DESC/GVEC's lane) |
| Boozer-transform output extension | Emit `gmn`, `rmnc_b/zmns_b`, ν, I/G per surface (full boozmn schema) — unlocks NEO/SFINCS/Γ_c downstream chain incl. sfincs_jax | booz_xform ([repo](https://github.com/hiddenSymmetries/booz_xform), [docs](https://hiddensymmetries.github.io/booz_xform/)) | S (booz_xform_jax computes most already) | Yes | Recommend; prerequisite plumbing for #3 |
| Fixed-point solver alternatives (SIESTA-style Newton phase) | Physics-based block-Hessian Newton finishing phase (what prec2d approximates) | SIESTA PoP 18, 062504 (2011) ([PDF](https://hsx.wisc.edu/wp-content/uploads/sites/747/2016/04/SIESTA_PoP_2011.pdf)); Hirshman-Betancourt JCP 96, 99 (1991) | — | — | Already embodied by the in-tree 2D preconditioner; no action |
| MRxMHD/islands (SPEC/HINT-class physics) | Equilibria with islands/chaos | SPEC ([site](https://princetonuniversity.github.io/SPEC/)); MRX differentiable relaxation ([arXiv:2510.26986](https://arxiv.org/abs/2510.26986)) | XL | — | **Reject** — different physics model; the residue diagnostic (A-table) is the pragmatic slice |

---

## 3. Suggested sequencing (if the user approves)

1. **Wave 1 (parity-preserving, low risk):** #1 stability objectives; #5(b,c) cheap proxies;
   Boozer output extension; R26g bootstrap (as specified above). All are new objective terms —
   zero contact with the parity-critical solver path.
2. **Wave 2:** #2 omnigenity (fixes the documented QI gap → new README QI row);
   #4 single-stage (feeds the R26f free-boundary README example).
3. **Wave 3:** #3 bounce module (ε_eff → Γ_c → AE); then #5(a) ITG proxy campaign paper-able
   results.
4. **Numerics spikes (independent, benchmark-gated like R23):** Anderson acceleration;
   vmapped ensembles; `lconm1`; axis fix; mixed precision.

Each wave ends with an FD-validated gradient table (the R25 gate pattern) and a README/docs
example. Items rejected above (Zernike/spline basis change, SPEC-class physics, TERPSICHORE,
permanent magnets) are recorded so future sweeps don't re-litigate them.
