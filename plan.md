# vmec_jax Overhaul Plan — from research prototype to a fast, differentiable, research-grade VMEC

**Audience:** an autonomous agent (Cowork) with full local access, executing this plan end-to-end.
**Owner / git identity:** all commits and pushes authored as `rogeriojorge` (GitHub user `rogeriojorge`;
`gh` is already authenticated as `rogeriojorge` with `repo`+`workflow` scopes on this machine).
**Status:** this file is the single active plan. It replaces every file in `vmec_jax_plan/`
(`plan.md`, `plan_freeb.md`, `plan_differentiability.md`, `discrete_adjoint_2506_plan.md`, and the
1.5 MB / 32,088-line `plan_research_grade_performance_differentiability.md`). Those files are Phase-0
inputs (read once, extract anything still relevant into scratch `NOTES.md`) and are then deleted from
the working tree **and from git history** in Phase 1. Keep this plan.md under 200 KB; it is a working
document, updated with a short status line per phase as work proceeds, and moved to `docs/dev/` (or
deleted) at the v0.1.0 release.

**Mirror status (2026-07-12): ACTIVE IMPLEMENTATION PLAN.** Axisymmetric straight-axis finite-beta
free boundary is the supported high-beta target under the anisotropic `fixed_flux_cut` model in
Phase 5. The
toroidal Fourier hybrid is validated only through achieved beta 0.8333% and is deferred above that
measured limit. The nonaxisymmetric straight-mirror lane is deferred after failing its local-mode
refinement gate. The M9--M10 integration gates are complete; implementation proceeds through the
ordered core roadmap without reopening rejected solver variants.

---

## 0. Mission statement

Turn `vmec_jax` into the reference JAX implementation of the VMEC ideal-MHD equilibrium solver:

1. **End-to-end differentiable** library API (fixed and free boundary), fast on CPU and GPU, using
   implicit differentiation of the converged equilibrium — not unrolled iteration tapes and not the
   current "fingerprint-gated branch-local" machinery.
2. **A non-differentiable CLI fast path** that may use Python-side control flow, host callbacks,
   early exits, and donated buffers to beat the differentiable path in wall time.
3. **VMEC2000 parity**: iteration prints, `wout_*.nc` contents, threed1-style summaries, and
   converged physics quantities match VMEC2000 within per-quantity validation tolerances.
4. **Performance parity or better** than VMEC2000 single-thread CPU on the benchmark suite,
   including multigrid (`NS_ARRAY` ladders), which is currently slower than VMEC2000 — a named bug.
5. **A small, readable codebase**: about 60 Python files in `vmec_jax/`, ~30–34k library lines
   (revised 2026-07-12 after main added bootstrap/stability numerics and the open-field topology
   landed as focused modules; still
   a >4x reduction from **229 files / ~123k lines**), physically
   meaningful names, docstrings everywhere, ≥95% coverage without repo bloat (tests currently
   ~14k lines; target ≤ ~10k test lines without deleting distinct scientific gates).
6. **A ~10 MB repository** after a `git filter-repo` history rewrite (currently 57.4 MiB packed);
   large assets move to GitHub Releases; no Claude in the contributors panel.
7. **User-friendly docs** with full derivations (energy functional → forces → spectral condensation
   → preconditioner → time stepping → free boundary → adjoint), every equation linked to the
   implementing source.
8. **simsopt-style optimization examples** for QA / QH / QP / QI that start from a circular torus
   and converge to precise configurations in a single, short, readable script each.
9. **Free-boundary showcase**: β = 0→5% scans driven by ESSOS coils (stellarator + tokamak), run
   both through generated mgrid files and through direct Biot–Savart evaluation (no mgrid),
   demonstrating agreement — and that the direct path is the interpolation-free reference.
10. **Feature superset vs VMEC++** where VMEC++ has gaps: `lasym` (non-stellarator-symmetric),
    free boundary for tokamaks (`ntor=0`) and stellarators, fixed-boundary fallback on missing
    mgrid, spline/pedestal profile types, and a 2D preconditioner option — while borrowing VMEC++'s
    hot restart, JSON input, zero-crash policy, and validation methodology.
11. **Production mirror equilibria**: supported axisymmetric fixed/free straight mirrors at finite
    beta, with open axial field lines, isotropic and consistent anisotropic pressure closures,
    external coils, implicit derivatives, and mirror-native output. Nonaxisymmetric open mirrors
    remain a research API after failing local-mode refinement. Closed toroidal stellarator–mirror
    hybrids use the ordinary VMEC backend at their documented tolerance/beta limit.

Every decision below optimizes for: *simpler to use, fewer files, faster, more manageable*.

---

## 0.5 Completion roadmap (updated 2026-07-10) — honest status + ordered remaining work

This is the actionable index of what is DONE and what REMAINS. Sections 1-16 below are the original
phase specs (still authoritative for detail); this roadmap supersedes the scattered STATUS notes and
folds in every requirement from the user prompts and the two independent reviews.

### Done and verified on the current merged branch (2026-07-12)
- Legacy tree deleted; after merging main at ``2f75914f`` the package is **64 Python files / 33,474
  lines**, including the focused 20-file / 8,072-line open-mirror backend and the traceable
  omnigenity module. The tracked checkout is 7.3 MiB;
  no generated mirror results are tracked.
- Fixed-boundary equilibrium at **VMEC2000 machine-precision parity** across the 9 golden fixtures
  (exact iteration counts incl. lasym after the fixaray dnorm fix; wb ~1e-16, geometry ~1e-12).
- Structural executable reuse (SolverRuntime pytree; 0 recompiles on same-Resolution re-solve),
  NS_ARRAY multigrid ladder, hot restart, wout (full wrout.f set), plotting, Boozer, CLI on the core,
  VMEC++-JSON input, zero-crash typed errors, mgrid fallback.
- Free-boundary NESTOR solve (vacuum activates at the golden iteration exactly) + direct-coil field;
  CLI routes LFREEB decks + `--coils`.
- Implicit differentiation (custom_vjp + preconditioned-GMRES adjoint), FD-validated 1e-6..1e-12 for
  fixed-boundary boundary/profile/phiedge params; `jac="implicit"` wired into `least_squares`.
- GPU fixes (cuSPARSE tridiagonal, cache hardening, device policy) measured on office 2x A4000.
- README ns≥51 benchmark + vmecpp-style convergence figure; docs restructured (17 pages) with honest
  fixed-vs-free differentiability scoping; **CI green: 7 shards, wall ~9 min, 95% coverage gate**.
- Examples: 5 clean simsopt-style files (541 lines) on the core API; `jac="implicit"` used.

### Remaining work — ordered by priority (each item has an acceptance gate)

**R1. DONE — compact quasi-isodynamic optimization.**
QA and QH are precise. Exact QP is not a valid finite-aspect target (the documented near-axis
obstruction and measured basin remain explicit), so its accepted gate is reproducible descent into
the best measured max-mode-5 basin. For QI, one ESS/`jac="implicit"` call from the kicked circular
seed plus continuation from its converged state reaches QI residual `<1e-2`
while restoring compactness, iota, and a measured mirror bound. The deck, 77 KiB strict restart,
wall/RSS, objective components, compressed figures, and permanent full test are recorded. The
implicit Jacobian remains launch-bound and peaks near 8 GiB; this evidence feeds R3.

**Seed policy (user 2026-07-10; ties to the R1 saddle finding).** The seed does NOT have to be an
exact circular torus — an exact-axisymmetric/circular boundary is a *saddle* of the QS residual (the
symmetry-breaking harmonics are even → gradient vanishes there), which is exactly why FD stalls and
implicit gradients are needed. So seed from a **near-circular torus with the would-be-zero boundary
harmonics initialized to ~1e-4** (a small symmetry-/shape-breaking "kick"), with one shaping mode
seeded a bit larger to give the optimizer a defined descent direction, instead of exact zeros. This
makes even the first step well-posed, is physically honest, and matches how the QA precise result was
obtained ("kicked circular seed"). Make the kick amplitude an example parameter-at-top (default
~1e-4) and document it. The alternative, richer seed is the **near-axis (pyQSC_JAX/pyQIC) seed of
R19** — offer both: the tiny-kick circular seed (simplest, in-repo) and the near-axis seed (best
starting point) so users learn both routes.

_R1 status (2026-07-12, office 36-core CPU):_
- **QA precise:** single-stage max-mode-5 ESS gives `2.043e-1 -> 7.155e-6`, aspect 6.000,
  iota 0.420, in 868 s.
- **QH precise:** staged max-mode 1→5 gives `6.908e-1 -> 5.831e-5`, aspect 8.000,
  iota -1.218.
- **QP accepted basin:** the final max-mode-5 deck gives about `4.5e-2`, a 2.1x improvement over
  the earlier plateau, but is deliberately not called exact QP.
- **QI precise/compact:** max-mode-6 ESS plus three fixed-weight continuation calls give
  `4.515e-1 -> 9.578e-3`, aspect 8.001, `|iota|=0.120002`, mirror 0.426. Independent
  reconvergence reaches `(fsqr,fsqz,fsql)=(9.99e-14,4.41e-15,7.27e-15)` in 5,973 iterations.
  The acceptance tolerance is aspect `<=8.01`, mirror `<=0.45`; forcing mirror to the former
  arbitrary equality target 0.20 destroys the compact-QI Pareto point.
- **Known issue passed to R3:** the optimized input is not cold-start robust even with the
  converged axis coefficients. The compact restart is the supported reproduction path until the
  generic interior guess is improved. Evidence is in `benchmarks/qi_compact.json`.

**R2. Free boundary to production.** Forward parity/performance are complete: the CTH golden
converges with per-variable VMEC2000 gates, the fused NESTOR path is bit-equivalent, and the ns=201
benchmark measures vmec-jax warm `24.86 s` versus VMEC2000 `26.74 s` (the former 3x target is
passed). The remaining gate is a memory-bounded many-parameter coupled solved-boundary adjoint;
small-parameter forward implicit sensitivities and the scoped coil showcases are complete.

_R2 status (2026-07-12):_ the direct ESSOS Landreman-Paul scan reaches actual beta 3.350% at
``ftol=1e-10``; 3.3625% fails at the documented minimum continuation step, so 4--5% is not claimed.
Generated-mgrid equilibrium parity for those same LP-QA coils is rejected after grid refinement,
even though off-grid interpolation converges. Positive same-coil parity is instead established for
the reconstructed DIII-D tokamak at actual beta 0, 1.496%, and 3.009%, with direct/generated LCFS
coefficient differences ``2.25e-4--6.31e-4`` and reviewed figures. Evidence is in
``benchmarks/free_boundary_essos_beta.json``, ``free_boundary_essos_mgrid_parity.json``, and
``free_boundary_tokamak_coil_parity.json``. The forward
result now retains its final NESTOR cache/potential and CLI/library WOUT files populate
``potsin``/``potcos`` plus ``xmpot``/``xnpot`` and all covariant/contravariant ``*_sur`` tables.
The surface tables reconstruct the retained real-space fields to transform roundoff and match a
fresh VMEC2000 CTH WOUT within 1.2e-3 scale-relative; evidence is in
``benchmarks/free_boundary_surface_fields.json``. The coupled adjoint remains open. The
coupled NESTOR-MHD fixed-point residual now reconstructs the retained final constraint state,
keeps the LCFS edge active, and passes a converged CTH residual gate plus ``extcur`` JVP vs central
finite difference (2026-07-12). Next is the structural-dof projection and adjoint Krylov solve.
An initial generic reverse-mode trace through the full CTH NESTOR rebuild was rejected after
reaching 10.8 GiB RSS while still compilation-bound after three minutes. The production adjoint
must use an implicit/custom VJP for the NESTOR linear solve and reuse its factorization; enforce a
small-resolution peak-memory gate before repeating the production CTH run.
Transposing the accepted forward-linearization closure was also rejected on the production CTH
case (2026-07-12): the 3,690-dof closure itself built in 3.89 s at 2.80 GiB RSS, but XLA then spent
4 min 17 s compiling the fused ``NESTOR full`` transpose, and process RSS reached 7.75 GiB before
the controlled stop. This is not the compact custom NESTOR transpose required by the gate; do not
wrap that closure in another Krylov implementation or retry it on GPU.
The accepted small-parameter method is forward implicit sensitivity: JVP-only Krylov avoids that
reverse graph and passes solved-LCFS central differences on 3D CTH (0.33%) and axisymmetric DIII-D
(0.42%). Measured peaks are 4.7 and 3.3 GiB; records and reproduction live in
``benchmarks/free_boundary_sensitivity.json`` and ``profile_production.py``. Remaining R2 derivative
work is the many-parameter coil-shape adjoint and coupled-Krylov memory/preconditioning reduction.
A radial block-tridiagonal warm start was measured and rejected on axisymmetric DIII-D: it reduced
Krylov iterations 2,270→1,312 but increased sensitivity wall time to 159.7 s and peak RSS
3.3→7.27 GiB. NESTOR edge pressure carries global plasma-current/axis coupling, so the radial block
is not an exact inverse (direct residual 1.33%). The implementation was removed; evidence remains in
``benchmarks/free_boundary_sensitivity.json``. The next preconditioner must represent that global
coupling without assembling the full reverse graph.
The production 2D Newton preconditioner was also rejected for this scan: with
``precon_type="GMRES"`` and ``prec2d_threshold=1e-5`` it did not complete the first reported
equilibrium within the default 1D path's practical run budget. It remains opt-in and is not used by
the showcase.
An attempted direct-``CoilSet`` extension of the coupled forward sensitivity
was likewise removed after its real LP-QA gate failed. At ``ns=8``, ``ftol=1e-10``, the Krylov
derivative converged to residual ``1.30e-10``, but reconverged finite differences were not
step-stable, two positive perturbations stalled, and the surviving derivatives disagreed with the
implicit value. Evidence is retained in ``benchmarks/free_boundary_sensitivity.json``. Do not expose
coil-shape solved-LCFS derivatives until a strict-tolerance continuation yields a smooth independent
finite-difference gate.

**R3. Default memory and cache controls COMPLETE; residual cold cost OPEN.** R26e established that
``jac_chunk_size="auto"`` bounds Jacobian columns, the converged-state memo reduced the measured
optimization peak from 6.0 to 3.5 GB, and the faster block-Jacobian default peaks at 4.9 GB because
of transient XLA compilation rather than retained solver data. ``jac_solver="gmres"`` remains the
documented lower-memory fallback. No additional runtime-memory knob is justified. R26c now enables
the machine-fingerprinted persistent XLA cache by default on CPU and measures li383 at 4.48 s first
process / 1.34 s second process, so cache setup is closed. The circular-tokamak measurement remains
9.51 s with a fresh cache and 3.37 s in a second process using that cache (0.42 s import-only).
The remaining gate is therefore case-specific cached dispatch/solve cost: profile graph execution,
reduce structural variants where evidence supports it, or provide a non-JAX CLI lane. Retain the
small-deck target below roughly 2.5 s rather than treating one faster deck as universal. Evidence is
in ``benchmarks/cold_start.json``.

**R4. GPU production evidence COMPLETE.** The existing small-to-large matrix is now joined by a
fresh office A4000 production run at the merged main tip: fixed ``ns=201`` warm 6.91 s, multigrid
9.57 s, CTH free boundary 7.40 s, implicit gradient 24.3 s, and a two-evaluation optimization 85.2 s
with 8.64 GB peak host RSS. Coupled free-boundary sensitivity exceeded a five-minute case cap. GPU
is therefore allowed for large forward solves, while gradients, optimization, and coupled
sensitivity remain CPU-preferred. Cold graph construction (31--203 s) must always be reported
separately. Full evidence and CPU references are in
``benchmarks/production_gpu_2026-07-12.json``; micro/crossover data remain in
``benchmarks/gpu_baseline.json``.

**R5. Finite-beta + diagnostics parity. FORWARD COMPLETE; PROFILE AD DEFERRED.** The four-case
golden WOUT suite passes all 12 completeness/value/roundtrip tests and checks current harmonics,
``jdotb/bdotb/bdotgradv``, every Mercier component, and beta scalars against VMEC2000. Main's compact
digest gate adds DSHAPE, CTH, li383, and finite-beta QA/QH references through 4.26% beta. Evidence and
the exact command are in ``benchmarks/finite_beta_diagnostics_parity.json``. ``DMerc`` and current
profile gradients remain finite-difference-only: their parity engine is intentionally host NumPy,
and a full traceable rewrite of derivative-amplified diagnostics is deferred until a concrete
optimization requires it. This does not block equilibrium or mirror promotion.

**R6. Refactor + docstring hygiene. COMPLETE.** Public API-like docstrings are complete: the
original audit fixed all omissions in ``daadbf47``; the current AST audit finds 0 missing across
567 top-level definitions and public class members. Mirror plotting
moved intact to its owning package, reducing core ``plotting`` from 1,039 to 888 lines. The NumPy
NESTOR parity path now lives in ``freeboundary_reference`` (232 lines), leaving the production
free-boundary driver at 832 lines with nine net source lines added. Differentiable observables now
live in ``implicit_quantities`` (123 lines), leaving custom-VJP/adjoint orchestration in ``implicit``
at 958 lines. Nyquist grid conventions now live in ``nyquist_grid`` (60 lines), leaving field,
current, and Mercier output in ``nyquist`` at 972 lines. Optimization now separates the public
objective/driver (``optimize``, 995 lines), implicit Jacobian backend (583), Boozer/QI objective
(333), and boundary/current parameterization (150). The fixed-boundary solver now separates
numerical kernels (``solver``, 885 lines), state/setup (``solver_runtime``, 553), and host
orchestration/result assembly (``solver_driver``, 387), while preserving public and tested private
import paths. Every core module is now at or below 999 lines. Gate: no core file >~1000 lines;
0 public definitions without docstrings; ruff+mypy clean without blanket ignores. Ruff and mypy
are clean across all 72 source files; solver, multigrid, free-boundary, CLI, package, and
golden-digest regression gates pass.

**R7. Docs completion COMPLETE.** The VMEC2000↔vmec-jax glossary is complete, all 26 executable
examples are referenced from the tutorials, and the final equation-to-source audit links the
documented spectral, energy/force, preconditioning, stepping, multigrid, free-boundary,
diagnostic, bootstrap, and implicit-differentiation equations to their implementations. Three
reviewed mirror/hybrid result figures are retained as compressed documentation evidence. Strict
Sphinx with warnings as errors is green (2026-07-12).

**R8. Mirror geometry COMPLETE WITH DOCUMENTED RESEARCH LIMITS.** PR #22 is integrated with current main
and provides axisymmetric fixed/free straight mirrors through beta 50%, anisotropic closures,
external coils, mirror-native output/plotting, and implicit derivatives. Nonaxisymmetric straight
free boundary remains research-only after failing its local-mode refinement gate. The toroidal
Fourier hybrid is reproducible through achieved beta 0.8333%; higher beta and a native spline state
are deferred. Gates and evidence are consolidated in §8 Phase 5.5 and ``benchmarks/mirror_*.json``.

**R9. Release v0.1.0.** After R1-R5: regenerate benchmarks/README, refresh the release asset bundle,
tag, publish PyPI + conda-forge, verify `pip install vmec-jax && vmec --test` on a clean machine.

### Standing constraints (apply to all remaining work)
- CI wall ≤10 min, coverage ≥95%, no brittle absolute wall-clock asserts (use ratios / compile counts).
- Optimization runs use thousands of iterations for real convergence; CI uses reduced budgets.
- Docs/README claims stay honest: separate validated fixed-boundary from in-progress free-boundary.
- Use `ssh office` (2x RTX A4000) for GPU/heavy runs; keep the local machine responsive (watchdog).
- Every commit rogeriojorge, no AI trailer; push small, let CI verify; avoid rapid successive pushes
  that cancel in-flight runs via concurrency.

### R10-R16 — detailed resumable tasks (added 2026-07-10 from user review; specific steps)

**R10. Prove functionality completeness vs VMEC2000 + VMEC++ (the "is it all there?" question).**
*(R10.2 DONE 2026-07-10, 2980d812: 2D block preconditioner — matrix-free Newton via
jax.jvp HVP on solvax.gmres; 2.5-11x iteration reduction on stiff cases (aspect-100 97->18,
163->15; nfp4_QH finite-beta 1885->204); default 1D path byte-identical; CI green incl. 95%
gate. Wall neutral on CPU cold — GPU/warm-cache/gcrot-recycling win pending. Showcase = R20.)*
The core is small (34 files / 19.2k lines) because JAX/Python is far denser than Fortran/C++ and we
dropped VMEC2000's MPI, v3fit reconstruction, and ANIMEC boilerplate — NOT because physics is missing.
Verified present: fixed + free boundary (NESTOR), lasym, ntor=0 free-bdy, multigrid + hot restart,
**18 profile parameterizations** (power_series/two_power/gauss_trunc/pedestal/cubic+akima splines/
line_segment + _i/_ip integrated variants), ncurr=0/1, full wrout.f wout set, Mercier, jxbforce, bss,
Boozer, JSON+INDATA, zero-crash, implicit diff (fixed-bdy), direct-coil free boundary, GPU.
Steps:
  1. Write `docs/functionality_matrix.rst` (+ a README summary): a feature-by-feature table
     vmec_jax vs VMEC2000 vs VMEC++ with a "where implemented (module:function)" column, so
     completeness is provable and auditable. Include the LOC/file-count comparison row (see R11.3).
  2. Close the **genuine gaps**, in priority: (a) **2D preconditioner** — currently only accepted as
     an input key (`precon_type`/`prec2d_threshold` in input.py) but NOT implemented; only the 1D
     radial preconditioner exists. Implement the optional 2D block preconditioner via matrix-free
     `jax.jvp` Hessian-vector products + preconditioned GMRES (plan §7.5), activated below a
     threshold on the finest grid, to cut iteration counts on stiff cases. (b) document ANIMEC
     (anisotropic pressure) and RFP (`lrfp`) as explicitly out-of-scope/niche, or add if a user
     needs them. Gate: functionality_matrix has no unexplained ❌ for a mainstream VMEC2000 feature.

**R11. README overhaul (all ns≥51; optimization + Boozer; code-size comparison; better showcase).**
  **(R11 DONE 2026-07-11, commit caf6166c.)** All four subitems shipped: free-bdy row runs at ns≥51
  (cth_like_free_bdy + cth_like_free_bdy_lasym_small, both converged in baseline.json); README
  `readme_optimization.png` (QA/QH/QP/QI initial-vs-optimized + Boozer |B|); code-size table via
  pygount 3.2 (vmec-jax **36 files / 11,789 SLOC / 5,532 comment / 0.47 doc-ratio** vs VMEC2000 115f
  /24,190/8,425/0.35 and VMEC++ 117f/22,824/7,646/0.34 — the shipped numbers, slightly refined from
  the 2026-07-10 estimate below as the core grew during R15/R18); `readme_equilibrium_showcase.png`
  now has 3D |B|-on-surface + Boozer-|B| with jet cmap (plotting.py `cmap` arg + `boozer_modB_on_surface`).
  Performance section reconciled to baseline.json (warm faster than VMEC2000 on 9/13 rows) with an
  explicit shared-CPU caveat: the ratios are conservative lower bounds measured under load; **R9
  re-runs the runtime figure on a clean machine.** docs -W green, ruff clean.
  1. **All benchmark rows ns≥51.** `benchmarks/run_baseline.py` already ramps fixed-bdy to ns≥51;
     the **free-boundary row (cth_like_free_bdy) must also run at ns≥51** (currently its deck ns may
     be <51). Bump the free-bdy deck's final NS_ARRAY stage to ≥51 (regenerate its mgrid if the grid
     resolution requires), re-run the whole suite, regenerate `readme_runtime_compare.png`. Verify no
     row in the figure/table uses ns<51.
  2. **Optimization panels.** Add README figures showing QA/QH/QI/QP **initial vs optimized** boundary
     cross-sections + **Boozer |B| on the LCFS** for each, from the R1 converged results. One compact
     multi-panel `readme_optimization.png` (compressed <150 KB). Generate via a tracked
     `benchmarks/make_readme_figures.py` addition using `core.plotting` + `core.boozer`.
  3. **Code-size comparison.** Add a README table: source files + lines of code for vmec_jax vs
     VMEC2000 vs VMEC++ (count with `cloc`/`tokei` over `/Users/rogerio/local/STELLOPT/VMEC2000/Sources`
     and `/Users/rogerio/local/vmecpp/src`; state what's counted). Message: comparable/greater
     capability in a fraction of the code. **Measured 2026-07-10 (solver source only; tests/
     bindings/third-party excluded): vmec_jax 34 files / 19,237 Python lines; VMEC2000 115
     files / 36,693 Fortran lines; VMEC++ 117 files / ~39,677 (34,255 C++ + 5,422 Python) —
     vmec_jax is ~half the code of both, with a superset of capabilities.** ALSO report the
     **comment/docstring vs actual-code split** (user 2026-07-10) to show vmec_jax is better
     documented and more user-friendly. Measured (tokenize for Python; comment-line count for
     Fortran/C++): vmec_jax **11,274 actual code (SLOC)** + 5,112 comments/docstrings (27% of
     total) → **doc-to-code ratio 0.45**; VMEC2000 24,164 code + 8,451 comment → 0.35; VMEC++
     ~23,149 code + ~5,841 comment → ~0.25. Headline: **vmec_jax has <half the actual code AND
     the highest documentation density of the three.** Use `pygount`/`cloc` for the README
     table (install if needed) so the comment/code split is reproducible.
  4. **Showcase figure.** `readme_equilibrium_showcase.png`: show the **3D geometry with |B| color on
     the surface**; and change the current flat |B| plot to **|B| in Boozer coordinates with the `jet`
     colormap** (the STELLOPT/Boozer convention). Update `core.plotting`/`core.boozer` plot helpers as
     needed (add a `cmap` arg + a boozer-|B|-on-LCFS plot).

**R12. Rename `tests/core_new/` → `tests/`.**
  **(R12 DONE 2026-07-11.)** `git mv` of all 38 test modules + `data/` up one level; merged the golden-fixture
  + `_module_jit_enabled` machinery from `core_new/conftest.py` into the root `tests/conftest.py`; decremented
  the 28 `Path(__file__).resolve().parents[N]` depth anchors (27×[2]→[1], 1×[1]→[0]) while leaving
  `DATA_DIR.parents[1]` and the unmoved conftest untouched; updated every `tests/core_new` path in ci.yml
  (shard lists, ignores, the golden-prefetch import), benchmarks, docs, examples, and core docstrings.
  Verified: 513 tests collect, golden prefetch resolves via `tests/conftest.py`, golden parity test passes.
  Done standalone (decoupled from R21 per the 2026-07-11 reordering). Original note follows:
  "new" is meaningless to users and the legacy `tests/`
is gone. Steps: `git mv tests/core_new/* tests/` (handle conftest.py merge — root `tests/conftest.py`
already exists with the RUN_FULL/jit gates; merge the core_new conftest fixtures into it), update
every CI path in `.github/workflows/ci.yml` (parity shard file lists, ignores, prefetch), update the
golden-fetch import path, `pyproject.toml` pytest config, and any `tests/core_new` string in docs.
Gate: CI green with the flat `tests/` layout; no `core_new` anywhere.

**R13. Many more pedagogic examples (study STELLOPT / VMEC2000 / hiddenSymmetries simsopt / DESC /
VMEC++ example layouts).**
  **(R13 IN PROGRESS 2026-07-11.)** Shipped so far (each simsopt-style, params-at-top, CI-smoke-tested,
  indexed in examples/README.md): `plot_and_boozer.py` (every plot_wout figure + Boozer on the LCFS),
  `profiles_power_and_spline.py` (power-series vs cubic-spline profiles → identical equilibrium; NCURR
  0 vs 1), `take_gradients.py` (implicit-adjoint d(aspect)/d(RBC) and d(wb)/d(phiedge) vs central FD,
  rel ~1e-9), `run_from_json.py` (VMEC++ JSON ↔ &INDATA round-trip → one equilibrium),
  `hot_restart_scan.py` (seed each scan point from the previous state → warm converges in ~1 iter vs
  ~309 cold, no recompile). Light ones run in the PR examples shard; take_gradients is nightly (`full`).
  `finite_beta_scan.py` (pressure ramp → beta, Shafranov axis shift, Mercier DMerc; hot-restarted).
  `free_boundary_mgrid.py` (NESTOR free boundary from coil EXTCUR + mgrid; LCFS solved for, nightly),
  `free_boundary_beta_scan.py` (free-bdy pressure ramp → beta 0→2.6%, LCFS re-solved each point, nightly).
  8 examples shipped this session. DEFERRED (need real coil data / advanced, follow-up):
  free_boundary_essos_coils (direct-coil free bdy — no bundled CTH coils that reproduce mgrid_cth_like
  and converge; needs a purpose-built coil set), single_stage_free_boundary_opt (gated on R15 free-bdy
  IFT wrap). These two are the remaining R13 items; the rest of R13 is done. Each is one simsopt-style file (params at top, no `main()`, prints
initial→progress→final, teaches one feature) and is CI-smoke-tested (reduced budget) + doubles as the
docs tutorial (R14.3). Target set:
  - `run_fixed_boundary.py` (exists), `run_from_json.py` (VMEC++ JSON in/out + convert),
  - `free_boundary_mgrid.py` (mgrid path), `free_boundary_essos_coils.py` (direct Biot-Savart, no
    mgrid; needs ESSOS), `free_boundary_beta_scan.py` (β=0..5% hot-restarted; the README β-scan),
  - `profiles_power_and_spline.py` (power_series vs cubic/akima; pressure/iota/current; ncurr=0 vs 1),
  - `finite_beta_scan.py` (pressure ramp; beta, Mercier, Shafranov shift),
  - `take_gradients.py` (implicit d(aspect|iota|QS)/d(boundary|profile) vs FD; jacrev usage),
  - `plot_and_boozer.py` (all plot types + `--booz` on LCFS),
  - `hot_restart_scan.py` (reuse a converged state across a parameter scan; warm speedups),
  - `single_stage_free_boundary_opt.py` (ESSOS coils → free-bdy equilibrium → QS/aspect targets;
    advanced) — gated on R15 free-bdy differentiation.
  Keep QA/QH/QP/QI optimization examples. `examples/README.md` indexes them by feature. Gate: every
  example smoke-passes in CI; each maps to a docs tutorial.

**R14. Complete the documentation (full theory + algorithms + tutorials, not an overview).**
  **(R14 DONE 2026-07-12, commit 8681e25d.)** R14.1 theory complete: +536 lines across
  theory/equations/algorithms/architecture — energy functional + Hirshman-Whitson moment method,
  parities/lasym, metric→|B| pipeline, force kernels + spectral condensation, 1D preconditioner
  derivation + NEW 2D block-preconditioner section, NESTOR Green's-function formulation,
  virtual-casing free bdy, full IFT/adjoint math with the O(1)-memory argument, device-policy section.
  Every :mod:/:func: ref verified; sphinx -W green. R14.2 reference already substantial; R14.3 below.
  **(R14.3 DONE 2026-07-11.)** `docs/tutorials.rst` rewritten from a "coming soon" stub into a real
  gallery: every R13 example (`literalinclude` so the page stays in sync with the tested code) grouped
  by theme — getting started (fixed run, plot+Boozer, JSON), profiles & finite-beta, hot restart,
  differentiation (implicit + free-bdy), free boundary (mgrid + beta scan), optimization (QA + the
  QH/QP/QI note). docs `-W` green. R14.1 theory (theory/equations/algorithms/architecture, ~1100 lines)
  and R14.2 reference (api/input/wout/cli) already substantial from prior lanes; remaining R14 polish is
  incremental, not a gap.
  1. **Theory & numerics, exhaustive** (`docs/theory/` split into pages, each equation linked to its
     implementing `core` function): ideal-MHD energy functional + Hirshman-Whitson moment method;
     flux coordinates + λ; Fourier representation + parities + lasym; **how |B| is computed** (metric
     → B^u/B^v → covariant B → |B|, from `core.fields`); the MHD **forces** (`core.forces`) and
     spectral condensation; **preconditioners** (1D radial derivation + tridiagonal solve; the 2D
     extension from R10.2); Richardson time-stepping + restart; multigrid + hot restart; **NESTOR**
     free-boundary vacuum (Green's function); the **implicit differentiation** adjoint (custom_vjp +
     preconditioned GMRES, with the basin/saddle finding from R1); the CLI-vs-jit lanes; device policy.
  2. **Reference**: API autodoc (all core modules), input reference (INDATA + JSON), wout reference
     (Appendix A rendered), glossary (VMEC2000↔vmec_jax names), CLI reference.
  3. **Tutorials = the examples** (R13): one docs page per example with rendered figures + expected
     output; the examples ARE the tutorials (docs currently reference tutorials that don't exist).
  Gate: docs `-W` green; a reader can follow B-field→forces→preconditioner→solve→differentiate→optimize
  entirely from the docs; every example has a tutorial page.

**R15. Free boundary to production parity + performance + differentiability (make it excellent AND
*(R15.1 DONE 2026-07-11, c83bb2a1: free boundary now CONVERGES to VMEC2000 parity — fixed a
double-nfp vacuum phase bug (boundary synthesis used xn=n·nfp against per-period zeta; the
geometric angle is phi=zeta·onp). input.cth_like_free_bdy: was stalling at NITER; now 574
iters to fsqr=9.9e-11, wb parity 2.1e-7 vs VMEC2000's 476-iter golden. Remaining: iters 574
vs 476 (~20% tail); R15.2 DONE (f197e144): vacuum fused into jitted JAX, 27 host↔device
round-trips/iter → ~0, warm 9.43→3.48 s (2.7×, now 4.5× VMEC2000), convergence bit-identical.
R15.3 PROTOTYPE (4dcbbb54): a fixed-boundary virtual-casing objective gives coil/extcur
gradients without a NESTOR adjoint and matches central FD to `2.2e-13..1.2e-10`. The required
`VmecSurfaceFieldData`/exterior-field API currently exists only on `virtual_casing_jax`'s
`feature/jax-vmec-extender` branch, not its PyPI 0.0.2 release. Production differentiation remains
open until that API is released and an IFT/adjoint propagates coil changes through the solved
free-boundary state; add the cth golden and mgrid path to CI only after those gates.)*
show it).** Forward solve and performance steps 1--2 are complete; coupled solved-boundary coil
derivatives and the final showcase remain. Steps:
  1. **Converge as well as VMEC2000.** Diagnose why the free-bdy solve stalls (nvacskip cadence, ivac
     activation threshold, edge-force/preconditioner interaction at js=ns, delt policy) vs VMEC2000 on
     the same deck; raise NITER and match VMEC2000's converged fsq. Produce a **converged
     free-boundary golden fixture** (validate wout per-variable vs VMEC2000). This is the acceptance
     authority — no coil-derivative claims before it.
  2. **Fast.** Profile the NESTOR/vacuum solve (dominant cost); the dense scalar-potential solve and
     Green's-function assembly are the suspects. Target free-bdy warm within ~3× VMEC2000 (from 17 s).
  3. **Differentiable.** Extend the implicit residual to include the free-boundary/NESTOR contribution
     so coil-dof and pressure gradients flow through the converged free-boundary fixed point;
     FD-validate d(boundary|QS)/d(coil-dof) and d/d(extcur).
  4. **Show it.** README free-boundary parity + performance row (ns≥51); a
     `single_stage_free_boundary_opt.py` example (R13); the β-scan showcase (R13) with mgrid vs direct
     agreement. Gate: converged free-bdy wout parity vs VMEC2000; free-bdy warm within ~3× Fortran;
     free-bdy gradients FD-validated; examples + README updated.

**R16. Memory reduction (reason + act; the biggest quantitative gap).**
*(R16 FINDING 2026-07-10: the DFT-transform-tensor premise is REFUTED by profiling — those
are 0.017-2.1 MB, negligible. The peak (0.6 GB floor; 3.8 GB implicit gradient) is XLA COMPILE
working set, not data. remat/jax.checkpoint tested + REJECTED (3885 vs 3809 MB — nothing to
save). What worked: jit-factoring the implicit residual F + _field_chain → implicit gradient
3809→3045 MB (−20%) AND 40→31.6 s (−21%), bit-identical; jac_chunk_size='auto' default
(bounds GPU/large-dof runtime memory); donate CLI carry (neutral CPU). The ≥2× CPU gate is NOT
met because the bottleneck is the compiler; <1.5 GB needs a custom_vjp split of the monolithic
jacrev program (correctness risk) or a smaller XLA footprint. REFRAME R16: 'reduce the XLA
compile working set' — the real levers are jit-factoring + GPU chunking + persistent cache.)* Measured: solves use
0.6-1.5 GB (NuhrenbergZille 3.3 GB, free-bdy 2.6 GB) vs VMEC2000's 28-102 MB — **20-30×**; implicit
gradient 3.4 GB. This IS improvable — the causes are architectural, not fundamental:
  1. **Profile** peak device/host buffers with `jax.profiler.device_memory_profile()` +
     `memory_stats()` on a mid + large deck; attribute MB to: the batched-DFT transform matrices
     (dense `(nznt × mnmax)` per parity/derivative — the prime suspect; VMEC2000 uses O(N) DFT loops),
     the trajectory/history buffers, un-donated carry copies, and jit residual variants.
  2. **Act**, in impact order: donate solver-carry buffers in the CLI lane
     (`jax.jit(donate_argnums=...)`); free/rematerialize the large transform tensors instead of
     holding all parities/derivatives simultaneously (or use the FFT path where it wins); shrink the
     trajectory buffer (store only what prints needs); collapse redundant structural jit variants
     (padded shapes); for the implicit gradient, chunk the per-dof Jacobian (DESC's `jac_chunk_size`
     idea — see R17) so peak memory doesn't scale with dof count.
  3. Gate: **≥2× peak-memory reduction** on the benchmark median (target <~700 MB for mid decks,
     <1.5 GB for the largest), implicit-gradient peak <~1.5 GB, recorded in `benchmarks/baseline.json`
     and the README performance notes. Correctness (parity + gradient tests) unchanged.

**R17. Apply DESC ideas (deep-dive done 2026-07-10; https://github.com/PlasmaControl/DESC).** DESC is
a JAX Fourier-Zernike force-residual code — numerics don't transfer 1:1, but these architecture/UX
patterns do. Ordered by value, each cross-referenced into the lane it strengthens:

  *Memory (feeds R16):*
  1. **`jac_chunk_size` column-chunking of the optimization Jacobian** — DESC's headline memory knob:
     build the residual Jacobian in column blocks so peak memory = m0 + m1·chunk (time ≈ t0 +
     t1/chunk), `"auto"` picks the largest that fits. We chunk only coil eval today, NOT the objective
     Jacobian. Add a `jac_chunk_size` kwarg to `least_squares` (both FD and `jac="implicit"`), chunk
     the per-dof loop with `jax.lax.map(..., batch_size=chunk)`. THE fix for optimization memory.
  2. **`jax.checkpoint`/remat on the adjoint + field chain** — core has NO remat anywhere; wrapping
     `implicit._field_chain`/force-eval in `jax.checkpoint` recomputes in backward instead of storing,
     the direct lever on the 3.4 GB implicit-gradient backward.
  3. **Expose GPU knobs**: `XLA_PYTHON_CLIENT_MEM_FRACTION` (0.75→0.9), `XLA_PYTHON_CLIENT_ALLOCATOR=
     platform` for OOM debugging; surface in `doctor.py`. Verify our persistent cache is as aggressive
     as DESC (`jax_persistent_cache_min_compile_time_secs=0`) to help the cold small-deck target (R16).

  *Optimization depth (feeds R1):*
  4. **Block-solve all dof columns against ONE shared linearization** (block-GMRES / recycled Krylov
     subspace) instead of one preconditioned GMRES per dof — DESC's "factorize once, reuse" lesson;
     the most direct per-dof Jacobian-cost win, complements the CPU-pin fix (a37d0ec3).
  5. **Perturbation (analytic Newton) warm-start for trial solves** — seed each trial boundary with a
     first-order step `dx = −(∂F/∂x)^{-1}(∂F/∂c)dc` (we already have `∂F/∂x`, `∂F/∂c` VJPs in
     `implicit.py`) before iterating; cuts per-trial iterations → deeper QA/QH/QP/QI at fixed budget.
  6. **`bounds=(lo,hi)` inequality targets + generic `loss_function` (min/max/mean) on every term** —
     DESC's objective contract; removes weight-tuning guesswork (aspect∈(7,9), mirror∈(0.18,0.22))
     and unifies the `l_grad_b`/`mirror_ratio` bespoke reductions. Extend the `(fun,target,weight)`
     term to `(fun, target|bounds, weight, loss_function)`.
  7. **Richer objective library** into `core.optimize`, high-value first: `QuasisymmetryTripleProduct`
     (local `f_T`, no FSA — cheap complement to our ratio residual); **`EffectiveRipple` ε_eff**
     (1/ν neoclassical — we have the bounce primitives in `quasi_isodynamic`); `GammaC` (fast-ion);
     `Omnigenity`+`OmnigenousField` target (a cleaner QI formulation than our 4-term residual — study
     for QI depth); `BootstrapRedlConsistency`, `BallooningStability` (fuller stability vs Mercier);
     medium: elongation/curvature/BScaleLength/rotational-transform+shear profile targets.

  *Free-boundary differentiation (feeds R15):*
  8. **Virtual-casing `BoundaryError` as an ADDITIONAL differentiable free-boundary formulation** —
     DESC gets free-boundary gradients WITHOUT a NESTOR subsolve by making `B·n=0` and the pressure
     balance a differentiable objective via the virtual-casing principle. This sidesteps
     differentiating our NESTOR fixed point and is the cleanest route to coil→boundary→QS gradients.
     Keep NESTOR for the forward VMEC2000-parity solve; add virtual casing for the differentiable path.

  *UX / capability (feeds R13/R14/R1):*
  9. **Near-axis (pyQSC/pyQIC) seeding** — `from_near_axis` builds a physically-good `VmecInput`
     boundary from a QSC/QIC solution instead of a circular seed → better QA/QH/QI starts (our input
     decks even note "B0 not yet implemented"). Direct optimization-depth lever.
  10. **`Equilibrium` save/load (HDF5) + `EquilibriaFamily`** — return the staged `max_mode`/multigrid
      sequence as an inspectable family; make campaigns resumable (feeds the save/load UX).
  11. **Plot helpers to mirror**: `plot_qs_error` (QS `f_B/f_C/f_T` vs flux), `plot_comparison`
      (overlay before/after optimization surfaces — the README optimization panels of R11.2),
      `plot_boozer_surface`/`plot_boozer_modes` (LCFS Boozer, feeds R11.4), `plot_coefficients`
      (spectral-convergence diagnostic we lack). Add to `core.plotting`.
  12. **Notebook tutorials + output-analysis notebook** — DESC ships 7 rendered notebooks; convert the
      flagship examples to narrated notebooks with inline plots (feeds R13/R14.3); add a "how to read
      a wout / compute QS error / plot Boozer" analysis tutorial.
  13. **CI: split fast-unit vs slow-regression workflows** (DESC pattern) — we already shard; formalize
      the golden/regression split and add the I/O-format reference doc (every INDATA + wout var, §827).

  Gate: items 1-2 (chunk+remat) land first (biggest measurable win, feed R16's ≥2× gate); each other
  adopted idea lands as a tested change in its cross-referenced lane. Note: our matrix-free O(1)-memory
  adjoint is already BETTER than DESC's for a single scalar gradient — keep it; borrow the chunking,
  remat, warm-start, objectives, virtual-casing, and UX. **Route the solver-generic ones (chunk,
  remat, block/recycled Krylov, warm-start) through SOLVAX (R18); the physics ones (virtual casing,
  near-axis) through the uwplasma packages (R19).**

**R18. SOLVAX integration — slim vmec_jax, share solver infra with the uwplasma ecosystem.**
*(STATUS 2026-07-10: R18a + R18b DONE.* SOLVAX PR #1 merged + released v0.2.0 to PyPI
(backend-aware tridiagonal_solve + chunked-autodiff, example-per-capability + full docs).
vmec_jax imports them (d6b4c938): preconditioner tridiagonal, adjoint GMRES, jac_chunk_size
all via solvax; preconditioner + gradient tests bit-identical; CI green incl. 95% gate; core
−56 net lines now. Remaining: the big reduction with the 2D preconditioner on
solvax.block_thomas_truncated (R10.2).)*
`uwplasma/SOLVAX` (local `/Users/rogerio/local/SOLVAX`, v0.1.0, "differentiable structured linear
solvers, preconditioners and matrix-free methods in JAX", built on lineax) ALREADY ships: `banded`
LU (+periodic), `krylov` (`gmres`, `gcrot`=recycled Krylov), `implicit` (`linear_solve`,
`root_solve` = implicit-function-theorem custom_vjp), `direct` (`block_thomas`,
`block_thomas_truncated` = block-tridiagonal Schur elimination), `precond` (jacobi/block_jacobi/
coarse_operator/line_smoother/p_multigrid/mixed_precision/kronecker), `refine` (iterative refinement),
`operators` (MatrixFree/Sum/Kronecker). This overlaps heavily with vmec_jax's solver needs, so the
work is **bidirectional** and the net effect is a SLIMMER, better-integrated vmec_jax.

  *18a. Assess + migrate FROM vmec_jax TO SOLVAX (make the SOLVAX PR).* Read SOLVAX `src/solvax/*`
  and vmec_jax `core/preconditioner.py` + `core/implicit.py`; migrate ONLY genuinely-generic pieces
  SOLVAX lacks, as a PR to `uwplasma/SOLVAX` (branch, as rogeriojorge):
    - **Backend-aware batched tridiagonal solve**: our `tridiagonal_solve` (CPU vectorized Thomas +
      GPU `jax.lax.linalg.tridiagonal_solve`/cuSPARSE, `platform_dependent` selection, batched over
      RHS/columns). SOLVAX `banded` is general banded LU; add this specialized fast tridiagonal path
      if absent. (Not the VMEC-specific precondn/lamcal/scalfor — those stay in vmec_jax.)
    - **`jac_chunk_size` chunked-Jacobian utility** (R17.1): a generic `jax.lax.map`-based column-
      chunked forward/reverse Jacobian builder with `"auto"` sizing — a solver/AD utility that belongs
      in SOLVAX, reused by vmec_jax and sfincs_jax.
    - **Perturbation/Newton-predictor warm-start** (R17.5) if it generalizes cleanly on top of
      SOLVAX's recycled-Krylov continuation.
    - The PR MUST also add (user requirement): **one example per SOLVAX capability** (including the
      new ones and the pre-existing banded/krylov/gcrot/root_solve/block_thomas/precond/refine), and
      **comprehensive up-to-date docs** explaining every method, its equations, the source, the
      architecture, inputs/outputs, and use cases (mirror the vmec_jax docs bar). Gate: SOLVAX CI
      green, coverage kept, PR opened with examples + docs.
  *18b. Import FROM SOLVAX INTO vmec_jax (slim the core).* After (or alongside) 18a, refactor:
    - `core/implicit.py` adjoint → use SOLVAX `root_solve`/`linear_solve` + `krylov.gmres`/`gcrot`
      (gcrot gives the recycled/block Krylov of R17.4 for free) instead of the hand-rolled custom_vjp
      + `jax.scipy.sparse.linalg`. Keep our preconditioner as the Krylov `M`.
    - The 1D radial preconditioner tridiagonal solve in `core/preconditioner.py` → SOLVAX's
      tridiagonal/banded solve (the migrated one), deleting the duplicated Thomas kernel.
    - The **2D block preconditioner** (R10.2) → build on SOLVAX `direct.block_thomas_truncated`
      (truncated block-tridiagonal storage — exactly VMEC's BCYCLIC analogue) + `krylov` + `precond`,
      NOT from scratch.
    - Add `solvax` to vmec_jax runtime deps (unpinned). Gate: parity + gradient tests unchanged;
      vmec_jax core LOC drops (target −1 to −2k lines); one place to maintain the solver math.
  Net: vmec_jax gets slimmer and faster, SOLVAX gets battle-tested methods + examples + docs, and the
  uwplasma ecosystem (sfincs_jax, vmec_jax) shares one solver layer.

**R19. Physics-package reuse (uwplasma) — don't re-implement.**
  - **Virtual casing (R17.8 differentiable free boundary)**: reuse `uwplasma/virtual_casing_jax`
    (local; JAX virtual-casing with examples incl. `simsopt_stage_two_optimization_finite_beta.py`,
    `w7x_gradB.py`, `vmec_extender_python_api.py`) instead of re-implementing the virtual-casing
    `B·n` differentiable free-boundary formulation. Wire it as the differentiable free-boundary path
    (keep NESTOR for the VMEC2000-parity forward solve).
  - **Near-axis seeding (R17.9)**: use `uwplasma/pyQSC_JAX` (QA/QH near-axis) and
    `github.com/rogeriojorge/pyQIC` (local `pyQIC`, import `qic`, for QI) to build physically-good
    `VmecInput` boundary seeds via `from_near_axis`, replacing the circular-torus seed for deeper
    QA/QH/QI optimization. Both differentiable/JAX where possible so seeds flow into single-stage.
  Gate: free-boundary gradients via virtual_casing_jax FD-validated; near-axis-seeded QA/QH/QI reach
  deeper precision than the circular seed (feeds R1); examples added (R13).

**R20. Showcase everything new (README + docs + examples) — the differentiators.**
  **(R20 DONE 2026-07-11, commit caf6166c, with R11.)** README now carries `readme_precond.png`
  (2D-vs-1D iteration counts, 2.5–11x on stiff decks), a **DESC comparison section** (VMEC2000
  iteration-parity + standard wout, INDATA/JSON drop-in, NESTOR *and* virtual-casing free boundary,
  lasym, VMEC2000-format prints; honest about DESC's Zernike-at-low-res and objective-library edge),
  and a capability matrix enumerating every beyond-VMEC2000 feature (implicit diff, direct coils, 2D
  precond, chunked memory, virtual-casing free-bdy, near-axis seeding, SOLVAX-shared solvers). Each
  user-facing capability has an example (R13) + tutorial (R14). Residual follow-up: a couple of
  examples/tutorials still to broaden under R13/R14, tracked there — the showcase evidence itself is
  shipped.
  - **2D preconditioner advantages** (once R10.2/R18b land): README + docs figure — iteration-count
    and wall-time reduction vs the 1D preconditioner on a stiff case; explain the method (docs R14).
  - **DESC comparison where vmec_jax WINS** (README table + notes), beyond the O(1)-memory adjoint:
    exact VMEC2000 iteration-for-iteration parity + standard `wout` (DESC is a different equilibrium,
    not VMEC-parity); direct INDATA/JSON drop-in; free boundary with NESTOR *and* virtual casing;
    lasym; VMEC2000-format prints; and any measured speed/memory wins from R16. Be honest where DESC
    wins (Zernike accuracy at low resolution, mature objective library — which R17.7 narrows).
  - **Mention every new capability** (DESC-derived AND our own beyond-VMEC2000: implicit diff, direct
    coils, 2D preconditioner, chunked/remat memory, virtual-casing free-bdy, near-axis seeding, the
    SOLVAX-shared solvers) in BOTH README and docs, and for each user-facing one ship an example
    (R13) + a tutorial page (R14). Gate: README/docs enumerate the differentiators with evidence;
    each important new capability has an example.

**R21. Rename everything `vmec_jax` → VMEX (user 2026-07-10).**
  **ORDERING (user 2026-07-11): R21 is now the ABSOLUTE LAST step — do everything else in the plan first
  (R12 tests rename, R14 docs, R9 v0.1.0 release), THEN the VMEX rename as a standalone cutover.** R21 is
  decoupled from R12 (do R12 now, standalone) and no longer "right before R9" — v0.1.0 releases as
  `vmec-jax`, and VMEX becomes a later renamed version with a `vmec_jax` compatibility shim. Names: GitHub repo
`uwplasma/vmec_jax` → `uwplasma/VMEX`; Python import package `vmec_jax` → **`vmex`** (lowercase,
PEP 8; `import vmex`); PyPI distribution `vmec-jax` → **`vmex`** (verified AVAILABLE on PyPI
2026-07-10, HTTP 404); CLI command → **`vmex`** (keep `vmec` as an alias — do NOT rename the output
`wout_*.nc`/`boozmn_*.nc` files, those are the VMEC community conventions, not our package name).
Scope measured: 96 files / 444 Python occurrences. Do it as ONE ATOMIC sweep (a partial rename breaks
everything), paired with R12 (`tests/core_new/` → `tests/`):
  1. `git mv vmec_jax vmex`; global identifier replace `vmec_jax` → `vmex` across .py/.rst/.md/.toml/
     .yml (mind word boundaries: `vmec_jax` the package vs `vmec2000`/`vmec_input`/wout var names that
     must NOT change; and the display string "vmec-jax"/"vmec_jax" in prose → "VMEX"). Update
     `pyproject.toml` name=`vmex`, `[project.scripts] vmex = "vmex.core.cli:main"` (+ `vmec` alias),
     all `[project.urls]` to the VMEX repo, package-data paths. Update `.github/workflows/*.yml`
     (test paths, the golden fetch, size check), `docs/conf.py` + readthedocs slug, README badges
     (PyPI/docs/CI URLs → vmex / VMEX), `CITATION.cff`, `.readthedocs.yaml`.
  2. Ship a thin **`vmec_jax` compatibility shim** for one release: a stub package that
     `from vmex import *` and emits a `DeprecationWarning` (pre-1.0 courtesy so existing imports don't
     hard-break); document the deprecation. (Optional — a clean break is acceptable at v0.0.x, but the
     shim is user-friendly.)
  3. GitHub repo rename (auto-creates redirects); update the local `git remote`; re-point the
     readthedocs project and the conda-forge feedstock (if it exists) to `vmex`; keep the old
     `vmec-jax` PyPI project with a final `0.0.x` release whose long-description points to `vmex`.
  4. NO-BUGS GATE (the user's explicit requirement): after the sweep — `pip install -e .` resolves;
     `python -c "import vmex; print(vmex.__version__)"`; `vmex --test` + `vmex input.solovev` +
     `--plot` + `--booz` all work; FULL test suite green (rename any `tests/` import of the package);
     docs `-W` green; CI green incl. the 95% coverage gate; ruff clean; and a grep confirms **zero
     stray `vmec_jax` identifiers** remain except the intentional compat shim. Verify on a fresh clone.
  Gate: fresh clone installs as `vmex`, CLI/docs/CI all green, no stray identifiers, PyPI `vmex`
  published; then proceed to R9 release under the VMEX name.

**R24. Full production profiling (CPU + GPU) + runtime/memory/simplicity pass (user 2026-07-12).**
  Harness: `benchmarks/profile_production.py` — five production workflows (fixed ns=201, multigrid
  51/101/201, NESTOR free boundary, implicit value_and_grad, least_squares opt step), each reporting
  cold (compile) vs warm wall, iterations, ms/iter, peak RSS, and GPU device memory. Run with
  `JAX_PLATFORMS=cpu` for the CPU profile; plain on a GPU box for the GPU profile.
  **Finding #1 (2026-07-12, HIGH):** on a GPU box the QA/QH/QP optimization examples ran the hot loop
  on the GPU by default and took HOURS per stage vs MINUTES pinned to `JAX_PLATFORMS=cpu` (office 2x
  A4000: capped QH still in stage 2 after 100+ min on GPU; QA stage 1 in ~minutes on CPU). The
  forward-solve policy (`device.recommended_device`, 100k iteration-work threshold) and the implicit
  pin (`resolve_implicit_device` → CPU) exist, but something in the `least_squares(jac="implicit")`
  hot path still lands on the accelerator — diagnose with the GPU-vs-CPU profile pair and either fix
  the placement or make the optimizer pin its whole session to the recommended backend. Also: JAX had
  preallocated 13.3 GB VRAM (default 75%) — consider XLA_PYTHON_CLIENT_PREALLOCATE=false guidance.
  **Finding #2:** `opt_step` (2-nfev max_mode-1 least_squares, minimal_seed_nfp2) warm ~63 s / peak
  RSS ~6 GB on a contended local CPU — the heaviest per-call production path; profile where the time
  goes (per-dof implicit JVP solves vs forward solves vs trf overhead) and cut it.
  **Current CPU/GPU profile (2026-07-12, post-R25/R26 main integration):** local M-series CPU warm
  walls are fixed ns=201 5.5 s, multigrid 7.8 s, implicit gradient 17.3 s, and opt step 34.3 s. Office
  A4000 warm walls are 6.91, 9.57, 24.3, and 85.2 s respectively; CTH free boundary is 7.40 s
  (12.9 ms/iteration). Cold GPU walls are 31.3--203 s, optimization peaks at 8.64 GB host RSS, and
  coupled sensitivity exceeds a five-minute case cap. Thus forward solves are GPU-competitive, but a
  fast desktop CPU wins every measured production case and remains mandatory for gradients and
  optimization. The converged-state memo and block Jacobian explain the opt-step improvement from
  the former 88.8/151 s CPU/GPU baselines. Exact records are in
  ``benchmarks/production_gpu_2026-07-12.json``.

**R26. FINAL PRE-VMEX SWEEP (user 2026-07-12; the last content pass — after R24/R25 conclude, before
R9 release and the VMEX rename R21).** Ten items (+k added 2026-07-12):
  k. **Single-stage ESS examples (user, DO NEXT):** two example scripts — one QA, one QI — that do
     NOT use the max_mode continuation for-loop: a SINGLE least_squares call directly at large
     max_mode (>=5) with Exponential Spectral Scaling as the trust-region x_scale (use_ess=True /
     _ess_scale — higher-|m|,|n| dofs get exponentially smaller trust radii, which is what makes the
     ladder unnecessary). Simsopt-style, CI-smoke-tested, indexed in examples/README.md; QI version
     uses the new traceable omnigenity residual (h2) once landed. Compare achieved QS/QI and wall vs
     the laddered examples in the docstrings.
  a. **Trim + simplify the code** — one more dead-code/duplication/altitude sweep over vmec_jax/core.
  b. **Port more functionality to SOLVAX** — **(MOSTLY DONE 2026-07-12.)** Audit: the generic
     linear-algebra IS already in SOLVAX and imported by vmec_jax — block_thomas_factor/solve,
     chunk_map, auto_chunk_size, chunked_jacfwd/rev, gmres, gcrot (no local copies remain; the R25.2
     block Jacobian and R25.3 recycle both use solvax). The only vmec-specific glue kept local is
     implicit._adjoint_solve/_recycled_solve (they wrap solvax.gmres/gcrot with VMEC's residual
     operator — correctly not generic). REMAINING: the harmonic-Ritz GCRO-DR upgrade INSIDE SOLVAX
     (solvax.gcrot's FIFO recycle space poisons reuse — the R25.3 negative result); a focused SOLVAX
     numerics enhancement that would unlock recycle=True in vmec_jax. Optional (block Jacobian already
     gives 33x).
  c. **Performance:** (i) COLD STARTS — **(DONE 2026-07-12.)** The persistent XLA compile cache was
     enabled but had NO default directory on CPU (accelerator-only gate), so CPU CLI/API runs
     recompiled every process. Made it default-on for every backend (_compat._default_compilation_cache_dir);
     the machine fingerprint already hashes CPU model+flags (AVX2/AVX512; +macOS sysctl brand added) so
     heterogeneous shared-FS machines never collide. Measured: li383 CLI 4.48 s -> 1.34 s on the 2nd run
     (3.3x), zero user config; opt out with VMEC_JAX_COMPILATION_CACHE=disabled. (ii) FREE BOUNDARY **(DONE)** —
     measured: the mgrid and external_field (direct-coil) lanes are identical (both 1.32 s warm / 574
     iters on cth ns=15; mgrid file-read overhead ~0), and NESTOR was already fused into jitted JAX in
     R15.2 (9.43->3.48 s). No further tune warranted.
  d. **Faster optimizations/gradients** — continue past the R25 gate (block-tridiag amortization,
     recycling, perturbation warm starts all landed and measured together).
  e. **Memory reduction with DEFAULT controls** — good defaults, no advanced user knobs required.
     **(R26e ADDRESSED 2026-07-12.)** Defaults already do the work: jac_chunk_size="auto" caps the
     Jacobian column memory, the R25.1 converged-state memo cut opt_step peak RSS 6.0->3.5 GB, and
     peak memory is dominated by the transient XLA COMPILE working set (R16 finding), not data. The
     block-tridiagonal Jacobian (now the jac_solver default) trades ~40% more compile working set
     (opt_step 3.5->4.9 GB) for its 33x speed — the right default; a memory-constrained user sets
     jac_solver="gmres" for the lower-memory path (documented). No further default change warranted.
  f. **README example: free boundary from ESSOS coils** — Landreman-Paul QA with increasing pressure,
     vol-avg beta = 0%, 1%, 2%, 3% (needs a coil set reproducing the LP QA boundary).
  g. **README: QA (nfp 2) + QH (nfp 4) optimization with SELF-CONSISTENT BOOTSTRAP CURRENT**,
     reproducing arXiv:2205.02914 (Landreman-Buller-Drevlak) against the Zenodo data in
     /Users/rogerio/local/20220708-01-zenodo_for_QS_optimization_with_self_consistent_bootstrap_current
     (calculations/ + configurations/). Requires the **Redl (2021) bootstrap formula in vmec_jax,
     DIFFERENTIABLE**, + a loop iterating the current profile to self-consistency with the equilibrium.
  h. **Literature/code deep dive — DONE + USER-APPROVED SCOPE (2026-07-12).** Proposal in
     notes_r26h_research_proposal.md. User approved for implementation (in this order, respecting
     file-conflict windows):
       h1. **Stability objectives**: infinite-n ballooning (JAX COBRA port, batched 1D eigh; blueprint
           arXiv:2302.07673) + differentiable Mercier/magnetic-well optimization terms. NEW module
           core/stability.py + tests; no conflicts — START FIRST.
       h2. **General omnigenity residual** (Dudt arXiv:2305.08026) + constructed-QI targets (Goodman
           arXiv:2211.09829), reusing the in-tree Boozer transform — fixes the documented "QI not
           precise" weakness. Touches optimize.py — AFTER R25.3 merges.
       h3. **Single-stage plasma–coil optimization with exact gradients**, using ESSOS from
           github.com/uwplasma/ESSOS for the coil side (differentiable Biot-Savart already in-tree via
           CoilSet.from_essos) + virtual casing + implicit diff composed end-to-end.
       h4. **Turbulence proxies from spectrax-gk (github.com/uwplasma/spectrax-gk)** — its available
           LINEAR, NONLINEAR and QUASILINEAR proxies wired as optimization objectives on vmec_jax
           geometry.
     NOT selected: bounce-averaged eps_eff/Gamma_c module (deferred).
  i. **Docs upgrade** — deeper algorithms/performance/differentiability/equations explanations, more
     engaging with plots; better tutorials/examples/use cases; user-friendly.
  j. **Release hygiene:** repo <= 10 MB; coverage >= 95% with SIMPLE, CONCISE tests in a SMALL number
     of files covering every functionality; future-proof; literature-anchored; real physics + numerical
     testing; a VMEC2000 parity/accuracy check that does NOT require storing large wout files (scalar
     digests of golden quantities); then release v0.1.0 (R9) with all of it.

**R25. Optimization wall-time: multi-hour → under one hour (user 2026-07-12; DO BEFORE the VMEX
rename R21).** Same modes and resolution — the win must come from a more efficient/performant
GRADIENT, not from shrinking the problem. Ground it in the literature (papers, preprints, reports,
docs and source of DESC / SIMSOPT / VMEC++ / adjoint-stellarator work) and pick the best fast,
accurate gradient strategy. Candidate directions to evaluate against the measured R24 profile
(opt_step cost is algorithmic — one solve_implicit per fun(x) AND per jac(x), one GMRES per dof per
Jacobian, redundant final solve_equilibrium):
  1. Reuse/share solves: one frozen solve per trust-region iterate serving both fun and jac; reuse
     the last trial state for result.equilibrium; deeper hot restarts across trials.
  2. Krylov recycling across Jacobian evals (SOLVAX gcrot — nearby systems share spectrum) and/or
     block-GMRES over all dof right-hand-sides at once instead of one GMRES per dof.
  3. Quasi-Newton Jacobian recycling: Broyden secant updates between exact implicit Jacobians
     (recompute exactly only every k-th iterate or on trust-region step rejection).
  4. trf internals: LSMR with a Jacobian LinearOperator built from JVP/VJP closures (no dense J
     assembly), scipy tr_solver options.
  5. Literature scan: DESC chunked-jacfwd + Levenberg-Marquardt practice, SIMSOPT MPI-FD baselines,
     Landreman-Paul analytic shape-gradient adjoints, one-shot (SAND) simultaneous optimization,
     VMEC++ perturbation/hot-restart papers.
  **RESEARCHED (2026-07-12, notes_r25_gradient_research.md — DESC source-verified, papers cited).
  Ranked implementation order:**
  1. **Memoize the converged state + perturbation warm start** (DESC `_update_equilibrium`/`f_where_x`
     pattern): jac(x) never re-solves at the x fun(x) just converged (~20 lines, zero accuracy risk);
     seed trial solves with the first-order perturbation z-(dF/dz)^-1(dF/dp)dp (arXiv:2203.15927).
     ~1.5-2x on the solve phase. DO FIRST.
  2. **Amortized block-tridiagonal factorization of dF/dz**: assemble the radial blocks with 3-colored
     jax.jvp probes (cost independent of dof count), solvax.block_thomas factor once, backsolve all
     24-120 dof RHS, keep a 2-3-iter GMRES corrector for exactness. 2-5x on the jac phase; also reuses
     as forward Newton preconditioner + perturbation seed.
  3. **GCROT recycle-space carry — LANDED AS OPT-IN, NEGATIVE RESULT (f1fdd509).** Plumbing exact
     (lax.scan carry, vmap-shared within chunks) but solvax v0.1's FIFO cycle-correction recycle space
     POISONS later solves (1.7-3.4x more iterations, Jacobian drift 1.6e-2): needs harmonic-Ritz
     GCRO-DR in SOLVAX (→ R26b item). Default recycle=False keeps the exact path; flip when SOLVAX
     upgrades.
  Runner-up Broyden secant (only option that weakens the per-iterate exactness); long-term: one-shot/
  SAND (~4x one solve for a whole optimization, needs replacing the scipy driver).
  **GATE PASSED (2026-07-12): the full max_mode 1->5 QA campaign on the office CPU box completes in
  1532 s (25.5 min, target <3600 s) with memo + block-tridiagonal Jacobian + CPU pin, reaching QS
  3.73e-7 (440x deeper than the prior README deck; iota 0.42, aspect 6.000). QP (the stall-prone
  class) still running at its own pace. Gradient FD validation and CI gradient shard green throughout.**

**R22. README/showcase refinement round 2 (user 2026-07-11; DO these before R21/R9; VMEX rename deferred
as a longer refactor).**
  **(R22 DONE 2026-07-11 except the ns=201 figure regen, which is folded into R9.)** Commits caf…→3e8296b2:
  (1) install now reads PyPI-recommended-OR-conda; (2) all `|B|` plots are line contours (plot_modB,
  Boozer, showcase, optimization); (3) QI reworked into its own `readme_qi.png` at nfp 1-4 with the
  OMNIGENITY residual (1.3e-2→3.2e-3) + 3D `|B|`, and `readme_optimization.png` split to QA/QH/QP with
  3D `|B|` and the PRECISE R1 QH deck (QS 5.83e-5, straight diagonal Boozer contours; QA 1.6e-4; QP
  0.094 — honestly the hardest class); (4) DESC comparison rewritten as a concise where-each-wins table;
  (5) warm/cold/GPU/memory bullets cut to 1-2 sentences; (6) benchmark harness RAMP_NS=201 (figure regen
  at R9 — needs idle box + vmecpp). NOTE learned: the QH/QP re-optimization is genuinely multi-hour even
  capped (implicit-gradient per-dof GMRES); reused the saved R1 campaign decks from office instead.
  1. **Install: PyPI (recommended) OR conda-forge.** Make the README install section state clearly that
     it is *either* `pip install vmex/vmec-jax` (recommended) *or* `conda install -c conda-forge`, not
     both — one path, recommended = PyPI.
  2. **`|B|` as line contours, not filled.** All `|B|` plots (`plot_modB`, `plot_boozmn_modB`, the
     showcase Boozer `|B|`) should be line contours (`ax.contour`), not filled `contourf`. Regenerate the
     affected README/docs figures.
  3. **QI optimization row reworked.** In `readme_optimization.png`: QI gets its **own row** with
     **nfp = 1, 2, 3, 4** examples, displays the achieved **QI residual** (omnigenity), *not* the QS
     residual, at **higher `max_mode`**; add **3D geometry plots with `|B|` on the surface** for the
     optimized configs. QA/QH/QP keep the QS residual. Regenerate the figure + the example/deck set.
  4. **DESC comparison rewritten for clarity.** Replace the prose with a concise
     advantages/disadvantages presentation — the same clear style as the vmec-jax vs VMEC2000 vs VMEC++
     feature matrix: what vmec-jax does better, what DESC does better, in short bullets/table cells.
  5. **Concise performance descriptions.** The warm / cold / GPU / memory bullets are too long — tighten
     each to one or two clear sentences.
  6. **Full-solve benchmarks at ns=201, not ns=51.** The full-equilibrium wall-clock table/figure should
     run at **ns = 201** (not 51) so the JIT compile time is small relative to run time (a fairer warm
     comparison). Regenerate `readme_runtime_compare.png` + reconcile the numbers. (Watch memory/wall on
     the heaviest decks; ns=201 is much larger — keep the suite honest and machine-load-caveated per R11.)
     **(R22.6 CONFIG DONE 2026-07-11: `run_baseline.py` `RAMP_NS=201`, multigrid ladders extended to
     201, docstring updated. Probe confirms the point — cth_like ns=201 warm=3.8 s vs cold=10.5 s, so the
     solve dominates. Figure REGEN DEFERRED TO R9: a clean, complete ns=201 figure needs (a) an idle
     machine — the box is currently contended by the user's sfincs at ~300% CPU, and the heaviest deck
     (NuhrenbergZille ns=201) is minutes-long — and (b) `vmecpp`, which is not importable in the current
     venv (the figure would drop its column). R9 finalizes benchmarks on a clean machine per R11; ns=201
     is now the harness default it will use.)**

**R23. Decide whether the 2D block preconditioner should be the DEFAULT (user 2026-07-11: "if it is so
good, why isn't it the default?").** Measure 2D-vs-1D on representative decks: (a) **accuracy** — the
converged `wout` must match the 1D/VMEC2000 golden to the same tolerance (the 2D precond changes the
*path*, not the fixed point; verify it doesn't shift the answer); (b) **wall-clock** warm speed;
(c) **peak memory**; (d) **iteration counts** (already known 2.5–11× fewer on stiff decks). Decision
rule: if 2D is *at least as accurate* AND *faster in wall-clock* (not just iterations) AND *not
materially heavier in memory* across the board (not only stiff decks), make it the **default**;
otherwise keep it opt-in with a **documented rationale** (e.g. per-iteration cost or memory on easy
decks). Update the README precond section + the `prec2d`/`precon_type` default in `solver.py` accordingly.
  **(R23 DONE 2026-07-11 — DECISION: keep opt-in.)** Measured 1D vs 2D (ns=51, warm, ftol=1e-11):
  accuracy IDENTICAL (wb parity 1e-9…1e-11 — 2D changes the path, not the fixed point); iterations
  2.6–5.4× fewer everywhere; but **wall-clock 0.55–1.16×** — a wash-to-slower (solovev 1.16×, cth_like
  1.14×, circular_tokamak **0.55× = ~2× slower**, aspect-100 stiff **0.97× tie**): the per-iteration
  GMRES+HVP cost offsets the iteration savings even on the stiffest deck. Peak RSS ~30% higher
  (cth_like 761→987 MB, the GMRES/HVP compile graph). So default 1D unchanged; 2D stays opt-in for
  iteration-count-bound / stalling cases. README precond section rewritten with this honest rationale;
  `solver.py` default (`prec2d=None`) left as-is (correct).

---

## 1. Ground truth — current state (audited 2026-07-08)

Facts established by direct audit; the executor should trust these and not re-derive them.

### 1.1 Repository

- 57.38 MiB packed git history; 89 MB working tree; 955 tracked files; single `main` plus 4 remote
  branches. Top history bloat: ~30 historical revisions of a ~1.7 MB `plan_differentiability.md`
  (now a 39-line stub — the blobs live only in history), multi-MB PNGs
  (`readme_best_optimization_qh.png` 2.0 MB, `minimal_seed_showcase_state_panel.png` 1.7 MB), and
  the 1.5 MB `vmec_jax_plan/plan_research_grade_performance_differentiability.md` at tip.
- **Claude in contributors:** authorship is clean (`git shortlog -sne --all` shows only Rogerio
  Jorge ×2 emails + Matthew Feickert ×1). Claude appears **only via 70 commit-message trailers**
  `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`. Fix = history rewrite of commit
  *messages* (strip the trailer), not a mailmap.
- **Branches:** `origin/codex/differentiability-refactor-plan`,
  `origin/feature/freeb-essos-coil-single-stage`, `origin/phase2/freeb-adjoint-validation` are all
  **0 commits ahead** of main → delete after the rewrite. The pre-rewrite mirror head is preserved
  locally as `archive/mirror-geometry-pre-rewrite` at `e4a7f05d`; its remote history was deleted
  because the rewritten `main` made a merge or rebase unusable. It contains a *native state block
  preconditioner*, *native spline matrix-free loop*, and *square hybrid solver method lanes*.
  Reuse is behavioral and selective: port compact equations, tests, and plotting ideas only after
  validating them against the new core. Never merge or broadly cherry-pick the archived history.

### 1.2 Library (`vmec_jax/`, 229 files, ~123k lines, 49 root modules)

- **Core physics exists and is comprehensive**: Fourier transforms (`kernels/tomnsp.py` 1513,
  `fourier.py`), geometry/metrics (`kernels/bcovar.py` 1434, `kernels/jacobian.py`), forces
  (`kernels/forces.py` 2010, `kernels/residue.py`), 1D preconditioner (`preconditioner_1d_jax.py`
  2247), fixed-boundary loop (`solvers/fixed_boundary/residual/` — iteration.py 2957, update.py
  2350, runtime.py 1957, policy.py 1708 — plus a parallel `scan/` tree), multigrid (`multigrid.py`
  331, a port of VMEC2000 `interp.f`), JAX NESTOR (`solvers/free_boundary/jax_nestor_operator.py`
  1671), mgrid IO + JAX interpolation, direct-coil Biot–Savart (`external_fields/coils_jax.py`,
  `essos_adapter.py`, routed via `MGRID_FILE='DIRECT_COILS'`), wout writer (`io/wout_files/` tree
  incl. jxbforce, mercier, bsubs, nyquist), CLI with `--plot/--booz/--doctor/--test`, and
  booz_xform_jax as a hard dependency.
- **The bloat** is: (a) duplicated NumPy/JAX kernel pairs (`kernels/numpy_forces.py` 1102 vs
  `kernels/forces.py`; `preconditioner_1d.py` vs `preconditioner_1d_jax.py`), (b) facade/compat
  shims (`solve.py`, `_compat.py`, `_solve_runtime.py`), (c) the entire "branch-local adjoint"
  apparatus (`discrete_adjoint.py` 2002, `implicit.py` 1865, `solvers/*/adjoint/` ≈ 30 files with
  fingerprints, replay tapes, gate reports), (d) overlapping driver layers (`driver.py`,
  `drivers/{staging,policy,lifecycle,...}.py`, `solvers/fixed_boundary/{residual,scan}/`), and
  (e) an oversized optimization/QI workflow layer (`optimization.py` 1993,
  `optimization_workflow.py` 1881, `quasi_isodynamic/` ≈ 6k lines, `solvers/free_boundary/
  coil_optimization.py` 2646).
- ~15 files violate the project's own 1500-line rule. mypy is configured with most error codes
  disabled and per-module `ignore_errors` on the core physics; ruff ignores E402/F821/F841/E501.
  The refactor must make these crutches unnecessary.

### 1.3 Tests, examples, docs

- `tests/`: 296 files, ~138k lines, codecov project gate 95%. Includes obvious coverage-padding
  ("`test_solve_wave3..10_coverage`" files) and 3–4k-line monsters
  (`test_direct_coil_finite_pressure_sensitivity.py` 4478). Only one tiny committed fixture
  (1.1 KB); large assets already come via `tools/fetch_assets.py` — keep that pattern.
- `examples/`: QA/QH/QP scripts are ~220–250 lines (close to acceptable); **QI is the mess**:
  `QI_optimization.py` (499 lines, ~49 helper/stage references) + per-NFP variants (256 lines each,
  3-phase circular→QP-basin→QI pipelines) + helper modules `qi_optimization_cases.py` (1066),
  `qi_staged_runner.py` (805). Free-boundary `free_boundary_QS_coil_optimization.py` is 1284 lines.
  `examples/data/` and `examples/data/single_grid/` duplicate ~40 input decks.
- `docs/`: Sphinx+furo, ~35 rst pages; theory pages exist (`theory.rst`, `equations.rst`,
  `algorithms.rst`, `discrete_adjoint.rst`) but are interleaved with internal plan/lane pages
  (`aggressive_performance_plan.rst`, `accelerated_merge_readiness.rst`,
  `optimization_sweep_results.rst`, `piecewise_omnigenous_plan.rst`) that must go.

### 1.4 Performance (README 2026-07-06 snapshot)

37 normalized single-grid rows (`NS_ARRAY=151`, `FTOL=1e-14`): **warm** vmec_jax beats VMEC2000 on
33/37; **cold** on only 14/37 (Python/JAX/XLA setup dominates small cases). VMEC++ converges cleanly
on only 17/37 rows. Multigrid ladders are slower than VMEC2000 multigrid (primary suspects in §5).

### 1.5 Differentiability

AD-vs-central-FD agreement is demonstrated for fixed-boundary geometry/profile scalars, QS/QI
residuals, `DMerc`, `D_R`, and *branch-local* direct-coil free-boundary scalars. Arbitrary
differentiation through adaptive solver-control branches is explicitly unclaimed. The whole
"promoted lane / fingerprint" claim policy disappears in this overhaul: a feature is either
supported and tested, or it does not exist in the public API. Implicit differentiation (Phase 4)
makes the adaptive-branch problem moot: only the converged fixed point defines the derivative.

### 1.6 External pieces

- **ESSOS** is cloned at `/Users/rogerio/local/ESSOS`. The mgrid PR is
  **uwplasma/ESSOS #33 "Add VMEC mgrid export from ESSOS coils"**, branch `feature/mgrid-from-coils`
  (open, +325/−21 over 7 files): adds `essos.mgrid.MGrid` (SIMSOPT-compatible cylindrical grid
  layout), `coils_to_mgrid(...)`, and `Coils.to_mgrid(...)`, validated by round-trip and SIMSOPT
  parity tests. Use this branch; if it merges, use main; push fixes to the PR (as rogeriojorge) if
  needed and pin the commit hash in example docstrings.
- **booz_xform_jax** exists at `uwplasma/booz_xform_jax` (pure JAX, MIT, pip-installable from
  GitHub; also a local clone at `/Users/rogerio/local/booz_xform_jax`) and is already a runtime
  dependency wired to `vmec --booz`.
- **jaxopt is deprecated.** Modern implicit-diff stack: **Optax** (first-order optimizers) +
  **Optimistix** (root/fixed-point solves with implicit differentiation) + **Lineax** (linear
  solves). We hand-roll a thin `custom_vjp` (§6) so none of these becomes a hard dependency;
  `lineax` is an acceptable optional backend for the adjoint linear solve.

---

## 2. Environment, locations, ground rules

### 2.1 Local paths

| Code | Path | Notes |
|---|---|---|
| vmec_jax | `/Users/rogerio/local/vmec_jax` | the repo being rewritten |
| VMEC2000 (STELLOPT) | `/Users/rogerio/local/STELLOPT/VMEC2000` | executable `Release/xvmec2000`; INDATA module in `LIBSTELL/Sources/Modules/vmec_input.f` |
| VMEC++ | `/Users/rogerio/local/vmecpp` | C++/Python reference; key docs: root `AGENTS.md`, `vmec/vmec/AGENTS.md`, `docs/fourier_basis_implementation.md` |
| ESSOS | `/Users/rogerio/local/ESSOS` | check out `feature/mgrid-from-coils` (PR #33) |
| booz_xform_jax | `/Users/rogerio/local/booz_xform_jax` | default runtime dependency |

### 2.2 Python environments

Create separate venvs (never the system Python):

```bash
python -m venv ~/.venvs/vmecjax && ~/.venvs/vmecjax/bin/pip install -e "/Users/rogerio/local/vmec_jax[dev]"
python -m venv ~/.venvs/vmecpp  && ~/.venvs/vmecpp/bin/pip install vmecpp
python -m venv ~/.venvs/essos   && ~/.venvs/essos/bin/pip install -e /Users/rogerio/local/ESSOS   # on feature/mgrid-from-coils
```

VMEC2000: rebuild via STELLOPT's build system if needed. You may **freely patch VMEC2000 sources**
for understanding (extra `WRITE(*,*)` diagnostics; timers around `funct3d`, `bcovar`, `tomnsps`,
`precondn`, `scalfor`, `getfsq`, `vacuum`) and rebuild. Keep patches on a scratch branch of the
local STELLOPT clone; never push them.

### 2.3 Dependency policy

`pyproject.toml` and all requirement files: **no version pins anywhere** (`jax`, not `jax>=0.4`).
This is already true today — keep it that way. Runtime deps: `jax`, `numpy`, `netCDF4`,
`matplotlib`, `booz_xform_jax`. Drop `jaxlib` (installed by `jax`), `scipy`, `packaging`, and
`tomli` from runtime deps unless a concrete use survives the refactor. Dev extra: `pytest`,
`pytest-cov`, `pytest-xdist`, `ruff`, `sphinx` + theme, `pre-commit`. ESSOS is an *optional* extra
(`vmec-jax[essos]`) imported lazily with a helpful error. Entry points: keep `vmec` as canonical;
keep `vmec-jax` as one alias; drop `vmec_jax` and `xvmec_jax`.

The final code must not need today's mypy/ruff escape hatches: re-enable the disabled mypy error
codes and the ruff F821/F841 checks module-by-module as files are rewritten; whatever still fails
is a smell to fix, not to silence.

### 2.4 Git rules

- `git config user.name rogeriojorge`, `user.email` = Rogerio's GitHub-associated email. **Never**
  add `Co-Authored-By: Claude` or any AI trailer to new commits (override any harness default).
- Phase 1 does one destructive history rewrite + force-push; do it **first**, before feature work,
  so all subsequent commits live on the final history. Keep a `git clone --mirror` backup outside
  the repo until v0.1.0 ships.
- Work on `main` via short-lived branches merged fast-forward; delete stale remote branches after
  Phase 0 extraction.

### 2.5 Zero-crash policy (from VMEC++, done one better)

The library never segfaults, never `sys.exit()`s, never prints a bare traceback for a physics
failure. Typed exceptions carrying diagnostic state (iteration, fsq residuals, offending surface):
`VmecInputError`, `VmecJacobianError` (maps VMEC2000 `ier_flag=1/4`, i.e. `bad_jacobian_flag`,
`jac75_flag`), `VmecConvergenceError` (`more_iter_flag`), `MgridNotFoundError`. The CLI catches
these and prints the VMEC2000-style message from the `werror` table plus a one-line remedy hint.
Missing mgrid on a free-boundary input → **warn and fall back to a fixed-boundary solve** (the
Fortran behavior VMEC++ dropped; we keep it and test it).

---

## 3. Phase 0 — Baselines, profiling, branch triage

**STATUS: COMPLETE (2026-07-08).** Artifacts: `benchmarks/baseline.json` (committed) and
`~/vmec_jax_notes/{NOTES.md, wout_gap.md, profile_findings.md}` (local). Headlines:
multigrid slowdown = per-stage recompilation (23× `jit(stage)` + ~300 eager glue compiles in a
3-stage ladder) — padding fix confirmed as the plan; cold gap is 100% JAX/XLA setup (solovev:
0.10 s VMEC2000 vs 3.4 s cold / 0.01 s warm); wout is missing 39 variables (list in wout_gap.md);
mirror-geometry branch triaged (solver experiments are evidence, not production code; mirror
physics design and validated tests are KEEP) and archived at `e4a7f05d`; QP optimization must
default to max_mode=3.

Most of the audit is done (§1). What remains before touching code:

1. **NOTES.md** (scratch area, not committed): distill the five `vmec_jax_plan/*.md` files (the
   32k-line log only for still-open items — most of it is historical micro-optimization diary),
   and skim `archive/mirror-geometry-pre-rewrite`: record what its native block
   preconditioner, spline matrix-free loop, and square-hybrid solver actually do and whether any
   idea survives into §5.4/§7.5. The archive stays local until the mirror migration is complete.
2. **Baseline benchmark script** `benchmarks/run_baseline.py` (committed, small; results JSON
   committed as `benchmarks/baseline.json`): fixed suite — `solovev`, `DSHAPE`, `HELIOTRON`,
   `cth_like_fixed_bdy` (+ lasym variant), `cth_like_free_bdy`, a DIII-D-like tokamak free-boundary
   case, `nfp4_QH_warm_start`, `w7x`, precise QA/QH (Landreman–Paul) — recording wall time
   (cold+warm), peak RSS, iterations to each `ftol` stage, for: VMEC2000 (xvmec2000,
   single-thread), VMEC++ (where it converges), vmec_jax CLI (CPU; GPU if available), each
   single-grid and multigrid. This script regenerates the README plot in Phase 10.
3. **Profile vmec_jax now** (`jax.profiler` + `py-spy`): per-case split of trace/compile vs run;
   count XLA compilations across a multigrid ladder (each `ns` stage recompiling is the #1 suspect
   for "multigrid slower than VMEC2000"); host↔device syncs per iteration in the hot loop (target
   0); transform vs preconditioner vs residual cost shares.
4. **Profile VMEC2000** (timer patches per §2.2, or gprof) on the same cases: per-part budget of
   funct3d / bcovar / transforms / precondn / tridslv / vacuum, and its multigrid stage timings —
   so we know exactly what budget each part of vmec_jax must beat.
5. **wout gap list**: `ncdump -h` a VMEC2000 wout vs a vmec_jax wout for the same case; diff
   against the authoritative variable list in Appendix A; record missing/mismatched variables.

**Exit criteria:** NOTES.md (branch verdicts, open TODOs worth keeping), `benchmarks/baseline.json`,
a ranked list of multigrid slowdown causes with profile evidence, and the wout gap checklist.

---

## 4. Phase 1 — Repository consolidation and history rewrite

**STATUS: COMPLETE (2026-07-08).** History rewritten with git-filter-repo (57.4 → 11.8 MiB
packed; 0 Claude trailers; contributors = rogeriojorge + matthewfeickert), force-pushed; 4 stale
branches deleted; vmec_jax_plan/, validation/, tools/diagnostics, examples/data/single_grid
removed; figures pruned/compressed (4.8 → 1.8 MB); interim CI (fast tests + smoke + build + size
check) green locally; pre-commit 200 KB guard added. Mirror backup at ~/vmec_jax_backup.git
(keep until v0.1.0).

Goal: one branch, ≤10 MB fresh clone, no Claude in contributors, drastically fewer files.

1. **Working-tree consolidation first** (so the rewrite also shrinks the tip):
   - Delete `vmec_jax_plan/` entirely (this plan.md replaces it), `validation/`, obsolete docs
     pages (plan/lane rst files, §11), `examples/data/single_grid/` (fold unique decks into
     `examples/data/`), and the archived optimization showcase PNGs.
   - Recompress every kept image (`oxipng -o4 --strip all` / `pngquant`); target <150 KB each; keep
     only figures referenced by README/docs.
   - `tools/` shrinks to `fetch_assets.py`, `make_release_assets.py`, `compress_figures.py`.
   - Large fixtures (reference wouts, mgrids, golden stdout captures, benchmark provenance) → a
     versioned GitHub Release bundle (`vmec-jax-test-assets-vX.tar.gz`, sha256-checked, cached
     under `~/.cache/vmec_jax/`). Keep in-repo only text input decks and one tiny mgrid (<1 MB) so
     `pytest -m quick` works offline.
2. **History rewrite** with `git filter-repo` on a fresh clone (destructive; mirror-backup first):
   - `--strip-blobs-bigger-than 300K`, plus explicit `--path <old plan files, old figures, old
     fixtures> --invert-paths`.
   - **Strip the Claude trailers** with a message callback:
     `git filter-repo --message-callback 'return re.sub(rb"\n?Co-Authored-By: Claude[^\n]*", b"", message)'`.
     Verify with `git log --all --format=%b | grep -ci co-authored-by: claude` → 0. (A mailmap pass
     can also normalize `rogerio.jorge@ist.utl.pt` vs `@wisc.edu` if desired — optional.)
   - Verify: `git count-objects -vH` ≤ ~10 MB packed; `git shortlog -sne` clean.
   - Force-push `main` + tags; delete the 4 stale remote branches; confirm the GitHub contributors
     page no longer lists Claude (may take a cache cycle).
3. **Hygiene going forward:** `.gitignore` covers `wout_*.nc`, `boozmn_*.nc`, `mgrid_*.nc` (except
   the tiny fixture), `figures/`, `results/`; pre-commit `check-added-large-files` (200 KB); a CI
   job fails if packed size exceeds 15 MB.

**Exit criteria:** fresh clone ≤ 10 MB; single `main`; contributors clean; CI green on the pruned
tree (tests may be temporarily reduced — full restructure lands in Phase 9).

---

## 5. Phase 2 — Core library refactor (architecture, naming, fixed-boundary parity)

**STATUS (2026-07-10): PHASE COMPLETE — legacy deleted (3ce3402c).** vmec_jax/ = 33 files /
19k lines all-core; tests/ = 24 files / 5k lines; 323 tests vs golden; ruff clean; docs -W green.
Remaining project work tracked in §10 (examples), §12 (tutorials), §13 (release), plus follow-ups:
free-boundary vacuum tuning + potvac export + freeb ladder, full radial padding, coverage gate,
mirror design doc.

**Independent review (2026-07-10, ~65% overall) — binding follow-ups:**
1. **CI must run in <=10 minutes** (currently ~22-26) at equal-or-better coverage: shard
   (fast / parity / gradient / optional full-physics), re-enable JIT inside gradient tests (a
   global fixture disables JIT, making implicit tests 105-160 s each — the main CI cost), cache
   goldens, and restore the **95% coverage gate** (currently 90%; weak: profiles 31%, step 72%,
   printing 77% — add targeted tests, don't pad).
   *(2026-07-10 progress, 0f9aca65):* sharded into fast/parity/gradient/examples/coverage-gate/
   cli-smoke/build; module-scoped JIT fixture landed (solver tests 5-40x faster); the parity long
   pole was test_examples subprocess smokes (QA 145s/QH 101s/QP 81s/QI 67s) — isolated into their
   own shard with QH/QP/QI gated to nightly RUN_FULL; device.py (was 71%, untested) + mgrid/
   optimize error branches given targeted tests. **CI GREEN (run 29110749965)**: 7 parallel
   shards each under a 9-min timeout (parity-a 8.0m, parity-b 4.3m, gradient 8.8m, others <2m;
   wall ~9m, meets <=10m) and the **95% coverage gate PASSES** (8521 stmts, 414 missing = 95%).
   parity split into two balanced shards (a=8 heavy solver modules/104 tests, b=rest/370); one
   brittle warm-solve wall bound relaxed (0.1s->1.0s; the zero-recompile check remains the gate).
   PR #23 closed (already on main, pre-deletion base); orphan branch/worktrees cleaned.
2. **Docs/README honesty**: distinguish validated fixed-boundary implicit differentiation from
   NOT-yet-supported free-boundary/coil derivatives and the optimizer wiring status; fix
   optimization.rst "no special handling" claim; README examples claim must become true when the
   examples land.
3. **Free boundary to production**: a CONVERGED free-boundary fixture (raise NITER; the CTH deck
   stops at 1000 with fsq~9e-2), vacuum-solve performance tuning (warm 14.4 s vs Fortran 1.95 s),
   then freeb implicit derivatives — do not promote coil-derivative claims before this.
4. **Memory workstream**: solves use 0.7-1.5 GB vs VMEC2000's 27-43 MB; implicit gradient 3.4 GB.
   Profile buffers, donate in the CLI lane, audit temporaries; targets in §7.7.
5. **Optimization convergence budgets**: examples run as many iterations as needed (thousands)
   for genuine convergence; CI smoke uses reduced budgets via VMEC_JAX_EXAMPLES_CI.
6. Line/docstring hygiene: complete. Public API-like docstrings have no omissions, and every core
   module is at or below 999 lines after ownership-preserving splits of free boundary, implicit
   quantities, Nyquist conventions, optimization support, and solver runtime/orchestration.

*(superseded status of 2026-07-09:)* core landed, integration/perf hardening next. `vmec_jax/core/` has 20
modules (~10k lines), each A/B-proven vs the legacy kernels (420+ tests) — including the solve
loop (solovev 215/215 iterations vs VMEC2000, cth 434, machine-precision wout parity), the
complete wout writer (all 39 missing variables; found legacy lasym output bugs: buco/jcur*/ctor
x1/2, jdotb x1/16, fast-bcovar bsubsmns corruption), mgrid IO/field (ESSOS PR#33 cross-verified),
and multigrid interpolation. Golden fixtures are a GitHub release (golden-v1) with sha256 fetch;
core suite is in CI with coverage reporting.
**(1) DONE + (2) DONE (2026-07-09, per-stage variant):** `SolverRuntime` is now a registered
pytree passed to module-level jitted lanes (`_while_lane`/`_block_lane`): RunSetup arrays +
`rcon0/zcon0` as data, `Resolution` + scalar config as meta, numpy trig/mode/gather tables
derived from meta via lru-cached `_static_tables`. Proven (tests/core_new/
test_multigrid_ladder.py, JAX_LOG_COMPILES subprocess counts): second solve at same Resolution
with different boundary = 0 compiles, 0.02 s on solovev. `solve(initial_state=...)` hot restart
added (boundary delta spread with the profil3d radial profile; bare edge-row swap measured
fsqr~0.5, spread ~4e-6; cth 1% RBC(0,1) restart 298 vs 434 iters — <25% is stepper-rate-limited,
needs Phase-4 Newton/2D precond). `multigrid.solve_multigrid` (runvmec.f ladder: skip-decreasing,
per-stage banners/ftol/niter, interp.f handoff) matches xvmec2000 ladders to machine precision
(cth 5/9/15 rel 9e-15; nfp4_QH 9/17/35 identical printed wb; NOTE ladder-vs-single-grid wb
scatter 1.36e-8 on nfp4_QH is inherent to VMEC — reproduced by xvmec2000 itself — m=1 freeze).
Compile behavior: one block-lane compile per distinct stage structure per session (3 for a
3-stage ladder, ~3 s each — cold ladder 9.3 s vs cold direct 5.1 s on cth; warm ladder 0.3 s,
0 compiles; direct-after-ladder reuses the final-stage executable, 0 compiles). **Follow-up
(padding, not attempted — >2 h):** ONE executable for all stages = pad radial arrays to
max(ns_array) as pytree *data* (s grids/hs/profiles already data, so per-stage values reuse one
executable) + a static `ns_active` mask threaded through the radial reductions (energies/force
norms in fields.py `energies_and_force_norms`, getfsq sums in residuals.py, precondn/lamcal
integrals and the tridiagonal jmax in preconditioner.py, jacobian half-mesh differences at the
padded rows) with masked rows pinned to identity updates in the loop body; validate vs the
per-stage ladder to 1e-15 per stage, then flip solve_multigrid to a single lane. (3)-(4)
unchanged below.
(3) parity breadth: 3D/lasym/finite-beta/ncurr=1/high-mode across all nine golden fixtures
(known gap: legacy lasym solver drifts ~5% on asym harmonics — validate the new core against
golden directly); (4) switch one public vertical slice (CLI fixed-boundary path) to the core,
then delete the corresponding legacy modules and migrate tests — repeat until the legacy tree is
gone. Implicit diff (Phase 3) starts once the residual API is frozen by (1)-(2).

### 5.1 Target layout (~30 files; one concern per file, none over ~1000 lines)

```
vmec_jax/
  __init__.py       # public API: run, Equilibrium, VmecInput, wout io, plotting entries
  input.py          # INDATA parser + VMEC++-compatible JSON input + VmecInput pytree; convert CLI
  profiles.py       # power_series, gauss_trunc, two_power, pedestal, cubic/akima splines, line_segment
  fourier.py        # (m,n) bookkeeping, parity tables, mscale/nscale, angle grids, m=1 constraint maps
  transforms.py     # totzsps/totzspa + tomnsps/tomnspa equivalents: batched DFT matmuls + FFT path
  geometry.py       # R,Z,λ real-space fields, jacobian tau/sqrt(g), metrics guu,guv,gvv   (jacobian.f)
  fields.py         # B^u,B^v, |B|, covariant B, pressure, energies wb/wp, tcon           (bcovar.f)
  forces.py         # MHD force kernels + spectral-condensation constraint force          (forces.f, alias.f)
  residuals.py      # fsqr/fsqz/fsql via getfsq, m=1 constraint, fedge                    (residue.f90)
  preconditioner.py # precondn/lamcal 1D radial precond, vectorized tridiagonal solve, 2D option (scalfor.f, tridslv, precon2d ideas)
  step.py           # damped 2nd-order Richardson step, dtau damping (ndamp=10), irst back-off (evolve.f, restart.f)
  solver.py         # single-grid loop: lax.while_loop core + host-blocked CLI variant    (eqsolve.f)
  multigrid.py      # NS_ARRAY ladder, coarse→fine interpolation, hot restart             (runvmec.f, interp.f)
  vacuum.py         # NESTOR: Green's function, analyt/scalpot, potvac solve              (NESTOR_vacuum/)
  freeboundary.py   # free-boundary iteration, ivac/nvacskip cadence, MagneticField protocol
  mgrid.py          # mgrid netCDF read/write, interpolated MagneticField
  coils.py          # ESSOS bridge: coils -> direct Biot-Savart field, write_mgrid from coils
  implicit.py       # custom_vjp implicit differentiation of the equilibrium (Phase 4)
  wout.py           # wout writer/reader — full Appendix-A variable set incl. jxbforce, mercier, bss
  printing.py       # VMEC2000-format iteration lines, stage banners, threed1 summary     (printout.f)
  plotting.py       # vmec --plot for wout and boozmn files
  boozer.py         # thin wrapper over booz_xform_jax (--booz)
  optimize.py       # objectives: QS ratio residual, QI (Goodman-style), aspect, iota, mirror; least-squares driver
  errors.py         # typed exceptions + werror message table
  cli.py            # `vmec` entry point
```

Names follow physics with a **VMEC-canonical glossary**: community-expected names stay (`ns, mpol,
ntor, nfp, lasym, iotaf, presf, rmnc, zmns, lmns, bmnc, ...`); internal Fortran temporaries get
descriptive names (`force_R_cos` not `armn`, `dpressure_ds` not `pres1`, `sqrt_g` not `gsqrt` — with
the glossary mapping both ways). Ship `docs/glossary.rst`: VMEC2000 name ↔ vmec_jax name ↔ meaning ↔
defining equation ↔ source location. Every module header docstring names its VMEC2000 counterpart
file(s) and the equations it implements.

Deletions (absorbed or dropped): `kernels/numpy_forces.py` (single JAX implementation, used by both
lanes), `preconditioner_1d.py` (keep only the JAX one), `solve.py`/`_compat.py` facades,
`discrete_adjoint.py` + both `adjoint/` trees + `optimizers/fixed_boundary/exact_replay*` (replaced
by `implicit.py`), `drivers/` + `solvers/fixed_boundary/{residual,scan}/` (merged into
`solver.py`/`step.py`), `quasi_isodynamic/` (distilled into `optimize.py`), `robust_coils.py`,
`optimization_workflow.py`, `finite_beta.py`/`bootstrap_current.py`/`redl_bootstrap.py` (move to
`optimize.py`-adjacent helpers only if an example/test uses them; otherwise drop — record in NOTES).

### 5.2 State and purity

- `EquilibriumState`: frozen pytree dataclass — spectral coefficients (`rmnc, zmns, lmns` +
  `rmns, zmnc, lmnc` when `lasym`), velocity `xcdot`, `time_step`, damping history `otau[10]`,
  iteration counters, residual history, `irst`-equivalent restart flag. All solver functions are
  pure `state -> state`.
- One set of spectral kernels shared by everything (solver, wout, plotting, objectives) — today's
  lane duplication is the main divergence source; eliminate it.
- Static configuration (resolutions, flags) in a hashable `VmecConfig`; **mode/radial arrays padded
  to the maximum multigrid resolution** so `ns` stages share one compiled executable (§7.1).

### 5.3 Two execution lanes, one physics

- `solver.solve(...)`: `lax.while_loop` over a jitted iteration, fully traceable — the
  differentiable API's forward solver.
- `solver.solve_cli(...)`: Python `while` around the same jitted *N-iteration block* kernel
  (e.g. `nstep=10`-aligned blocks via `lax.scan`), residuals checked on host between blocks —
  enabling exact-`ftol` early exit, live VMEC2000-format prints, buffer donation
  (`jax.jit(..., donate_argnums=...)`), and zero AD bookkeeping. Both lanes call identical physics
  kernels; a regression test asserts per-block state agreement to machine precision.

### 5.4 Algorithmic parity targets (the VMEC2000 details that matter — verified from source)

These constants/behaviors must be ported exactly; they are why VMEC2000 converges in few iterations:

- **Richardson step** (`evolve.f`): `dtau = min(|log(fsq1/fsq)|, 0.15)`, averaged over the last
  `ndamp=10` steps; `b1 = 1−dtau/2·Δt·⟨otau⟩`… concretely:
  `otav = mean(otau)`, `dtau = delt*otav/2`, `xcdot = (1−dtau)/(1+dtau)·xcdot + delt·gc`,
  `xc += delt·xcdot`.
- **Back-off** (`restart.f`): on `irst=2` (Jacobian sign change) restore saved state, zero
  velocity, `delt *= 0.90`, count `ijacob`; on `irst=3` (residual grew >1e4× best) restore,
  `delt /= 1.03`. Escalation in `eqsolve`: try `guess_axis` on first bad Jacobian; reset delt at
  `ijacob=25,50`; give up at 75 (`jac75_flag`).
- **Preconditioner cadence**: `precondn`+`lamcal`+force norms+`tcon` recomputed every
  `ns4=25` iterations, not every step.
- **tcon**: `tcon(js) = min(|ard/arnorm|,|azd/aznorm|)·tcon0-scaled·(32·hs)²` per surface,
  `tcon(ns)=½·tcon(ns−1)`; constraint force spectrally filtered to `m ∈ [1, mpol−2]` (`alias.f`)
  with `faccon(m)` weights.
- **m=1 constraint** (`residue.f90`): internally rotate `(gcr,gcz)_{m=1}` to `((gcr+gcz)/√2, 0)`;
  released when `fsqz<1e-6` etc. Boundary input applies `rbss=½(rbs+zbc)`-style conversion
  (`lconm1`, `readin.f`).
- **Radial start indices**: R,Z evolved from `jmin2`, λ from `jlam` (m-dependent; `vmec_params.f`).
- **1D preconditioner matrices**: `precondn` builds `axm/axd/bxm/bxd/cx` from
  `ptau = r12²·bsq·wint/gsqrt`-type integrals; `scalfor` forms tridiagonal
  `dx = axd + bxd·m² + cx·(n·nfp)²` with `edge_pedestal=0.05` and the ZC(0,0)(ns) `fac=0.25`
  stabilization; `tridslv` = Thomas algorithm vectorized over all (m,n) columns. λ uses the
  diagonal `faclam` from `lamcal` (`1/(blam·(n·nfp)² + clam·m² ± 2mn·nfp)`-shaped, √s-damped for
  m>16).
- **Free-boundary cadence** (`funct3d.f`): vacuum activates when `fsqr+fsqz ≤ 1e-3`; full NESTOR
  solve when `mod(iter2−iter1, nvacskip)==0`, incremental otherwise; adaptive
  `nvacskip = max(nvskip0, 1/max(0.1, 1e11·(fsqr+fsqz)))`; edge force `rbsq` from
  `bsqvac + presf(ns)` enters `forces` at js=ns; `rcon0,zcon0` ramp ×0.9/step in free-boundary.
- **Stopping**: converged when `fsqr, fsqz, fsql ≤ ftolv` simultaneously (physical, not
  preconditioned, residuals).

**Exit criteria for Phase 2:** all fixed-boundary benchmark cases (sym + lasym) converge with wout
parity vs VMEC2000 per Appendix-A tolerances; file/LoC budget met; ruff+mypy clean without today's
blanket ignores; every public function documented.

---

## 6. Phase 3 (interleave with Phase 2) — Differentiability done right

Adopt implicit differentiation of the equilibrium fixed point (DESC precedent; Skene & Burns
arXiv:2506.14792 for reuse-the-forward-machinery adjoints; jaxopt paper for the IFT formulation):

- Equilibrium = root of the preconditioned force residual `F(x, p) = 0`
  (`x` = spectral state, `p` = boundary coefficients / profile params / phiedge / coil currents &
  geometry / extcur). Wrap the solve in **`jax.custom_vjp`** (implemented in `implicit.py`):
  - forward: run the fast CLI-style solver (non-traced host loop is fine — it's opaque to AD),
    return converged `x*`;
  - backward: solve the adjoint linear system `(∂F/∂x)ᵀ λ = ḡ` matrix-free — `∂F/∂x`-vector
    products via `jax.vjp(residual_fn, x*)` — with **the 1D preconditioner as the preconditioner**
    for GMRES/BiCGStab (`jax.scipy.sparse.linalg.gmres`/`bicgstab`, or lineax); then return
    `−λᵀ ∂F/∂p` via one more VJP. Cost target: a handful of residual evaluations per gradient,
    O(1) memory in iteration count.
- This **replaces** `discrete_adjoint.py`, the replay tapes, fingerprints, and branch-local gates
  entirely. Multigrid/adaptive control lives inside the opaque forward solve; only the final fixed
  point defines the derivative (coarse stages are an initializer — stop-gradient by construction).
- Free boundary: identical scheme; NESTOR is inside `F` (traceable JAX code already exists), so
  coil parameters differentiate with no special handling. This deletes the "same-branch
  fingerprint-gated" hedging from the README.
- Provide `diff_mode="implicit"` (default) and `"unrolled"` (debug-only, small cases).
- **Permanent gradient tests** (rtol ≤ 1e-6 vs central FD): boundary coefficients → aspect, iota,
  QS residual, volume, `DMerc`; pressure profile / `pres_scale` → beta, wout scalars; coil currents
  and coil Fourier dofs (ESSOS) → free-boundary boundary shape and QS residual. Plus an adjoint
  linear-solve convergence test (preconditioned GMRES residual < 1e-10 in ≤ ~50 iterations).

**Exit criteria:** gradient tests pass; an L-BFGS boundary optimization with implicit gradients
matches/beats the FD-driven result at a fraction of cost; backward memory ≤ 2× forward.

---

## 7. Phase 4 — Performance (fast everywhere; multigrid faster than VMEC2000)

Ranked workstreams — confirm ranking against Phase-0 profiles before executing:

1. **Kill recompilation and host syncs.** One compiled solver for the whole `NS_ARRAY` ladder:
   pad radial arrays to `max(ns_array)` and mask; mode arrays sized once. No `.item()`/`float()`
   in the hot loop; prints via `jax.debug.callback` (jit lane) or between blocks (CLI lane).
   Measure: exactly one XLA compile per (mpol,ntor,lasym,lfreeb) tuple per session.
2. **Hot restart, VMEC++-style but stronger.** Public API `run(input, restart_from=output)`.
   VMEC++ restores only `rmnc/zmns/lmns` at a single matching `ns` and immediately activates the
   vacuum contribution; we do the same *plus* allow resolution changes by reusing our multigrid
   interpolation (radial interp in √s of scaled coefficients, odd-m axis extrapolation
   `2x₁−x₂`, spectrum pad/truncate — VMEC++ `_continuation.py` mechanics, already half-present in
   `multigrid.py`). Reset `delt` conservatively; carry λ. Hot restart powers the β-scan example and
   every optimization loop.
3. **Cold-start cost.** The 23/37 cold-slower rows are XLA setup. Mitigations: JAX persistent
   compilation cache enabled by default in the CLI (`JAX_COMPILATION_CACHE_DIR` under
   `~/.cache/vmec_jax/xla`), smaller/fused graphs (fewer distinct jitted entry points), lazy
   imports so `vmec --help` stays <100 ms.
4. **Transforms.** Keep batched-DFT matmuls with basis-baked weights (GPU-optimal, AD-friendly;
   VMEC++'s FFTX codelets only buy 10–20%). Add a `jnp.fft.rfft` path selected at trace time when
   `ntheta·nzeta` is large enough to win on CPU (benchmark the crossover once, hard-code the rule).
   Fuse totzsp→geometry→forces→tomnsp; verify with XLA cost analysis that intermediates stay
   fused.
5. **Preconditioner.** Exact 1D port (§5.4) with the Thomas solve vectorized over (m,n) — this is
   *the* convergence-rate feature; recompute on the ns4=25 cadence. Then an optional
   **`precond="2d"`**: VMEC2000's precon2d builds the Hessian by finite-difference "jogs" and
   block-tridiagonal LU (BCYCLIC); in JAX we get exact Hessian-vector products for free via
   `jax.jvp(residual_fn, ...)`, so implement 2D as matrix-free GMRES on the Newton step
   preconditioned by the 1D operator, activated below a `prec2d_threshold` on the finest grid
   (mirroring `ictrl_prec2d`/GMRES lanes in `evolve.f`/`gmres_mod.f`). Check NOTES for anything
   worth stealing from the `codex/mirror-geometry` native block preconditioner before deleting it.
6. **CPU threading / vectorization.** VMEC++ beats Fortran via OpenMP over radial partitions; our
   analog is XLA CPU multi-threading over the big batched matmuls — ensure kernels are large
   enough to parallelize, document `XLA_FLAGS=--xla_cpu_multi_thread_eigen=...`/thread pinning in
   docs/performance.
7. **Memory.** Donate state buffers in the CLI lane; float64 mandatory
   (`jax.config.update("jax_enable_x64", True)` at solver import); audit temporaries; peak-RSS in
   the benchmark output.

**Exit criteria:** CLI ≥ VMEC2000 speed on ≥80% of suite rows (cold, CPU, single-grid);
multigrid strictly faster than our single-grid *and* faster than VMEC2000 multigrid on the suite
median; GPU runs validated; README plot regenerated from `run_baseline.py`.

---

### 7.8 GPU profiling workstream (added 2026-07-09; hardware available)

`ssh office` (pop-os, 2x RTX A4000 16GB, repo at ~/vmec_jax) is available for GPU work. Reported
symptom: vmec_jax is sometimes SLOWER on GPU than CPU — cause unknown. Plan:

1. **Environment**: venv on office with CUDA jax (`pip install -U "jax[cuda12]"`), editable
   vmec_jax at current main, golden fixtures via the conftest downloader.
2. **Benchmark matrix** (extend `benchmarks/run_baseline.py` with a `--device {cpu,gpu}` axis and
   an office-runner mode): all baseline decks x {cpu, gpu} x {legacy solver, core solver
   cli/jit lanes} x {single-grid, multigrid} x {cold, warm}, recording wall, device memory,
   compile vs run time (jax.profiler), and per-iteration step time across problem sizes
   (ns=11 -> 151, low and high mpol/ntor) — locate the GPU crossover point.
3. **Hypotheses to test** for GPU-slower-than-CPU: (a) small kernels + dispatch overhead at low
   resolution (GPU should win only at high ns*mnmax); (b) host<->device syncs per iteration in
   the legacy driver; (c) the tridiagonal Thomas solve serializes over ns on GPU (lax.scan) —
   consider cyclic reduction or a batched parallel solve, or pin the tridiagonal solve to CPU;
   (d) float64 throughput on A4000 (GA104 fp64 = 1/32 fp32) — measure; experiment with fp32
   preconditioner + fp64 physics; (e) recompiles from per-solve closures (identity-cache landed;
   structural runtime caching pending).
4. **Deliverables**: `benchmarks/gpu_baseline.json`, a docs/performance section explaining the
   crossover + tuning guidance, and implementation changes ranked by measured impact feeding
   Phase 4.

## 8. Phase 5 — Free boundary, ESSOS, mirrors

1. **mgrid path** (VMEC2000-compatible, tokamaks *and* stellarators — VMEC++ can do neither
   `ntor=0` free-boundary nor lasym; we support both): validate on `cth_like_free_bdy`
   (sym + lasym) and a DIII-D-like tokamak mgrid case against VMEC2000.
2. **Direct-coil path**: `CoilField` (ESSOS Biot–Savart, differentiable in coil dofs) evaluated on
   the NESTOR grid each vacuum update; no interpolation. `mgrid.write_mgrid(field, ...)` generates
   VMEC2000-compatible mgrids from any field (use/align with ESSOS PR #33's
   `essos.mgrid.MGrid`/`coils_to_mgrid` so the two codes interchange files).
3. **Fallback**: `lfreeb=T` + missing mgrid → `MgridNotFoundError`-grade warning + fixed-boundary
   solve (§2.5).
4. **Single-stage optimization with ESSOS** (both directions the plan's development goals name):
   fixed-boundary single-stage (coil objectives + VMEC QS objectives on one gradient tape, boundary
   from coils via a quadratic-flux surface or direct constraint) and free-boundary single-stage
   (coils → direct field → free-boundary equilibrium → QS/aspect targets; gradients via §6).
   One example each, marked advanced.
5. **Mirror physics (production scope finalized 2026-07-09).** Open mirrors are not toroidal VMEC
   with a long major radius. They use a mirror-native inverse-coordinate backend, while sharing
   numerical and software components with the toroidal core. The closed stellarator–mirror hybrid
   remains on the ordinary VMEC backend.

   **STATUS (2026-07-09): M0 contracts and M1 foundation landed.** The clean backend now has
   mirror schema/config/end-cut contracts, increasing-order CGL differentiation/quadrature and
   interpolation, FFT theta derivatives, regular-axis 2D/3D geometry, the divergence-free
   contravariant field, and differentiable analytic one/two-coil benchmarks. Scientific tests cover
   polynomial exactness, integration by parts, spectral interpolation, analytic cylinder/flared/3D
   metrics, flux conservation, `div(B)`, direct Biot–Savart parity, and shape gradients. M2 is next.

   **M2 STATUS (2026-07-09): physical reference solve in progress.** Mass-conserving isotropic
   energy now uses VMEC-style radial half cells; an independent `curl(B)/mu0 x B - grad(p)` tensor
   residual caught and prevented a nonvariational full-mesh discretization. The host reference lane
   combines L-BFGS with an exact-JAX residual-Newton polish and accepts a result only by the
   nondimensional variational force (the mirror analogue of VMEC `fsq`). The separately differenced
   continuum tensor force is always reported as a spatial-verification residual and must converge
   under refinement. A perturbed cylinder reduced variational force from `5.39e-2` to below
   `1e-12` and its continuum tensor residual to `6.09e-14`;
   forced short runs raise a typed convergence error. The analytic two-coil flux-tube fixture now
   verifies throat/center field and paraxial residual scaling. A closed-form flared-tube MMS shows
   axial spectral force convergence (`1.39e-5`, `1.67e-8`, `1.77e-11` at `nxi=9,13,17`). The
   scalable lane uses exact JAX Hessian products with damped Newton-GMRES and a tensor-product
   radial/CGL stiffness inverse; a 585-unknown solve reaches `1.69e-15` variational force and
   `5.09e-13` continuum residual. M2 continuation and broader resolution studies remain open.

   **M3 STATUS (2026-07-09): closure layer landed; equilibrium coupling next.** Isotropic,
   ANIMEC bi-Maxwellian (Suzuki et al. Eqs. 4–6), and bilinear tabulated `p_parallel(s,B)` closures
   are differentiable PyTrees. `p_perp` is always derived from
   `p_parallel - B*partial_B(p_parallel)`; independent inconsistent moments cannot enter the API.
   Firehose `sigma` and mirror ellipticity are computed and tested, including the isotropic limit,
   trapped/passing continuity, tabulated interpolation, and coefficient gradients. The remaining
   M3 half-mesh ANIMEC energy now reproduces the isotropic passing-particle cylinder and matches
   central-difference shape derivatives. A coordinate-invariant tensor divergence computes
   `J x B - div(P)` with metric connection terms; it recovers a constant-pressure cylinder at
   `1.46e-13` normalized residual and separately verifies parallel force balance in shaped 3D
   states. The lateral diagnostic now independently reports plasma/vacuum tangency and the ANIMEC
   normal-stress jump. A perturbed finite-beta bi-Maxwellian cylinder now solves to `1.50e-15`
   variational force, `1.11e-13` tensor force, and `1.03e-11` parallel residual with valid
   ellipticity. M3 functional consistency is now closed for axisymmetry. Force norms exclude
   constrained side/end-cut nodes, while pointwise values remain available. For a nonuniform
   bi-Maxwellian profile, axial refinement at `ns=9` reduces tensor/parallel residuals from
   `2.52e-3/3.26e-3` (`nxi=9`) to `1.37e-3/6.50e-5` (`nxi=25`); at `nxi=25`, radial refinement
   reduces tensor residual from `1.77e-3` (`ns=7`) to `1.01e-3` (`ns=17`). The independently
   projected continuum tensor force is >`0.999` correlated with the discrete ANIMEC gradient and
   their weak-form discrepancy falls below 3% by `(ns,nxi)=(13,25)`. Thus variational force is the
   true discrete `fsq` convergence gate and tensor force is the documented spatial-refinement gate,
   as in M2. M4 may proceed; broader combined refinement remains part of M10 promotion evidence.

   **5.1 Supported physical model**

   - Coordinates are `(s, theta, xi)`, with `s in [0,1]`, periodic `theta`, and nonperiodic
     `xi in [-1,1]`. Use the VMEC radial mesh, Fourier in `theta`, and Chebyshev–Gauss–Lobatto in
     `xi`; retain an axial-basis protocol so multi-domain Chebyshev or B-splines can replace a
     poorly conditioned global polynomial without changing physics kernels.
   - The field representation is divergence-free by construction:
     `J B^s = 0`, `J B^theta = I'(s) - d_xi(lambda)`, and
     `J B^xi = Psi'(s) + d_theta(lambda)`. The lambda surface mean is the removed gauge mode.
   - The pressure tensor is
     `P = p_perp I + (p_parallel - p_perp) b b`. A `PressureClosure` supplies a generating energy
     density, `p_parallel(s,B)`, `p_perp(s,B)`, and analytic/JAX derivatives. It must enforce the
     parallel-force consistency relation
     `p_perp = p_parallel - B (partial p_parallel / partial B)_s`; independent arbitrary pressure
     arrays are invalid inputs. Ship an isotropic flux-function closure, an ANIMEC-compatible
     bi-Maxwellian closure, and a consistency-checked tabulated `(s,B)` closure for kinetic-code
     moments. The isotropic limit must reduce exactly to the scalar-pressure kernels.
   - The equilibrium residual is the physical tensor-force balance
     `R = J x B - div(P)`, projected onto admissible geometry/lambda variations. The variational
     energy is used when the closure provides a valid generating functional; the direct tensor
     residual is always computed independently as the convergence and validation diagnostic.
   - Firehose/mirror ellipticity indicators are mandatory outputs:
     `sigma = 1/mu0 + (p_perp-p_parallel)/B^2 > 0` and
     `partial_B(sigma B) > 0`. Failing either condition is a typed invalid-equilibrium result, not
     solver convergence. These checks are necessary model-validity gates, not a full stability
     claim.

   **5.2 Boundary model**

   - The side `s=1` is the lateral plasma-vacuum interface and is fixed or varied depending on the
     solve. Both plasma and vacuum fields are tangent there. Free boundary enforces the anisotropic
     normal-stress jump
     `p_perp + B_plasma^2/(2 mu0) = B_vacuum^2/(2 mu0)` and reports plasma, vacuum, and total
     normalized `B.n`. The first production profiles vanish smoothly at `s=1`; finite edge pressure
     is added only with an explicit surface-current model and diagnostic.
   - `xi=+/-1` are fixed computational cuts crossed by open field lines, not free plasma-vacuum
     interfaces or periodic caps. The production `fixed_flux_cut` policy prescribes the end
     geometry, normal magnetic flux, and lambda gauge from the coil/vacuum reference state; equal
     reflected data give the default up-down-symmetric mirror. Variations vanish at the cuts, while
     field lines and flux pass through. This is a static equilibrium model between specified end
     planes; end loss, sheath, sources, ambipolar potential, and transport are documented non-goals.
   - The vacuum annulus uses `B_v = B_coil + grad(nu)` with `laplacian(nu)=0`, Fourier×Chebyshev
     discretization, Neumann data on the moving side interface, and prescribed external-field data
     at the end cuts and outer computational boundary. Demonstrate convergence as that outer
     boundary is expanded before claiming an unbounded-vacuum result. NESTOR algorithms and
     cadence are reused where topology permits, but its toroidal Green function is not reused
     blindly for an open surface.

   **5.3 Architecture and reuse**

   Add only `vmec_jax/mirror/{model,basis,geometry,forces,vacuum,solver,output}.py` plus a small
   `__init__.py`. Reuse `core/coils.py`, transforms, radial interpolation, state stepping,
   tridiagonal/Krylov utilities, implicit-diff helpers, typed errors, and plotting styles. Extend
   `core/plotting.py` to dispatch mirror output rather than creating a plotting package. Root
   examples contain parameters and public calls only; geometry, normals, coil construction,
   summaries, tracing, and plots belong in library code. The archived head at `e4a7f05d` is an
   evidence source only: port equations or focused tests, never its solver stacks or broad commits.

   Use one physical residual and two execution lanes, as for toroidal VMEC: a JAX-traceable solver
   and a faster host-controlled CLI solver. Adapt the exact 1D radial preconditioner first; tensor
   it with an axial Chebyshev Helmholtz/line solve, then add matrix-free Newton–GMRES with that
   separable operator as preconditioner. The archived normal-equation `JᵀJ` block-CG method is not a
   production preconditioner because it squares the condition number.

   **5.4 Finite implementation sequence**

   1. **M0 — specification and migration.** Freeze signs, units, nondimensional residual norms,
      input schema, `mout` schema, end-cut contract, and analytic fixtures. Extract only the
      two-coil formulas, CGL tests, MMS cases, and plotting requirements from the archive.
   2. **M1 — basis and geometry.** Implement CGL nodes/differentiation/quadrature, transforms,
      axis regularity, axisymmetric and 3D embeddings, metrics, and divergence-free field. Test
      polynomial exactness, integration by parts, positive Jacobian, flux conservation, and
      spectral convergence before adding a nonlinear solve.
   3. **M2 — fixed-boundary isotropic axisymmetry.** Implement energy, tensor residual, lambda
      gauge, VMEC-like stepping, separable preconditioner, continuation, and diagnostics. Validate
      cylinder, flared tube, two circular coils (`B_z` on axis and low-radius `B_r,B_z`), and MMS.
   4. **M3 — anisotropic fixed boundary.** Implement isotropic, bi-Maxwellian, and tabulated
      closures; port the ANIMEC pressure/force identities from `fbal.f`, `bcovar.f`, `forces.f`, and
      `funct3d.f` rather than translating preprocessor structure. Verify closure derivatives,
      isotropic-limit identity, energy-gradient/tensor-force agreement, and ellipticity gates.
   5. **M4 — fixed-boundary 3D mirror.** Add nonaxisymmetric/helical boundaries and finite axial
      current. Demonstrate visible pitch, nonzero lambda response, and convergence under radial,
      poloidal, and axial refinement using the same solver and residual.
      **STATUS (2026-07-10): corrected research lane, axis gate still open.** The host solver packs
      fixed-cut, surface-gauge-free lambda variables and closes the exact packed energy gradient.
      The original one-point radial midpoint rule was found to admit an alternating lambda
      hourglass mode: it produced `fsq<1e-12` but a `0.37` independently differenced force residual.
      Those earlier M4 refinement numbers are rejected. Two-point Gauss integration in every radial
      cell now resolves both endpoints; a regression test gives the checkerboard mode finite energy,
      and all 47 non-full mirror tests plus the scheduled `(5,5),(7,7),(9,9)` study pass. At
      `ns=15,nxi=15,ntheta=5`, lambda and pitch are smooth, variational force is `2.25e-13`, the
      independent all-row/axis/bulk residuals are `0.0430/0.107/0.00972`, and Krylov work falls from
      13,000 to 2,000 iterations. CPU takes 35.2 s and one A4000 44.2 s with matching physics.
      Axial refinement `nxi=15,21,25` changes bulk force only `0.009724,0.009695,0.009687`, so the
      remaining error is radial/axis. Matrix-free radial runs at `ns=17,19,23,27,31` all reach the
      component-wise `1e-12` variational contract; bulk force decreases
      `0.00934,0.00893,0.00779,0.00635,0.00628`. The `ns=31` case has 3,805 unknowns, takes 141 s,
      uses 10,500 Krylov iterations, and the full three-run process peaks at 1.92 GB RSS. A
      differentiable coarse/fine full-state interpolation helper is tested. Main now imports
      SOLVAX 0.2.0 for shared core solver utilities; the mirror host preconditioner remains separate
      until a JAX replacement deletes its SciPy path. A VMEC2000-source-guided experiment applying
      odd-mode `sqrt(s)` scaling only in the force diagnostic was rejected: together with a regular
      physical-lambda axis it changed the `ns=15` all/axis/bulk residuals to
      `0.0563/0.0860/0.0489`, improving only the axis row while making the well-resolved bulk five
      times worse. VMEC applies this scaling throughout its internal Fourier state, geometry,
      half-mesh energy, forces, and preconditioner. Any retry must therefore use that complete
      representation and prove improvement against the current benchmark. A disposable complete-
      representation prototype then restored odd-mode scaling in the full-grid field, Gauss-cell
      energy, force curl, and solver state together. It passed all 20 focused tests and at `ns=15`
      reached `8.90e-13`, reducing all/axis force to `0.0330/0.0412`, but raising bulk force to
      `0.0312` and runtime to 44.9 s. Its direct `ns=23` solve exceeded four minutes and 1.25 GB
      without finishing (accepted path: 66.5 s), proving the current preconditioner is incompatible
      with the scaled Hessian. The prototype was not promoted. M4 remains open, but the scaled-state
      rewrite is deferred until it is paired from inception with parity-aware SOLVAX block
      preconditioning; acceptance still requires decreasing axis and bulk residuals plus materially
      lower Krylov work/memory at `ns>=31`. Main's matrix-free SOLVAX 2D Newton primitive was merged
      in `e3d0e74d` and tested as a direct replacement for the mirror host SciPy GMRES. A controlled
      cold `ns=15,nxi=15,ntheta=5` A/B produced identical energy and continuum force; SOLVAX reduced
      Krylov iterations `3000 -> 2000` and the final relative linear residual `0.107 -> 0.0101`, but
      regressed wall time `14.42 -> 16.38 s` and peak RSS `0.94 -> 1.27 GB`. The adapter was removed.
      Reconsider it only as a compiled-loop/recycled solve that beats both CLI wall time and memory;
      sharing an API alone is not sufficient for promotion.
   6. **M5 — open-vacuum solver.** Implement the annular scalar-potential solve and couple direct
      coils/mgrid fields. Validate Laplace MMS, reciprocity, gauge removal, coil-only fields,
      side-interface `B.n`, end flux, outer-boundary convergence, and axisymmetric comparisons
      against direct circular-loop fields.
      **STATUS (2026-07-09): variational annulus and direct-coil tangency landed.** A moving-side,
      Fourier-CGL radial/axial map evaluates `B_external + grad(nu)` and solves the quadratic vacuum
      energy. The production boundary functional enforces zero correction flux on the outer/end
      cuts while natural variation enforces plasma-side tangency; fixed-potential MMS remains
      available. Exact annulus volume, linear harmonic/Laplace MMS, uniform-field cancellation,
      and differentiable direct `CoilSet` sampling are tested. For two circular end coils, normalized
      plasma `B.n` decreases `1.79e-2, 4.71e-3, 2.10e-3, 8.31e-4` over
      `(nrho,nxi)=(5,9),(7,13),(9,17),(13,25)`, and Hessian bilinear forms satisfy reciprocity at
      `2e-12`. The coupled free-boundary path now uses a mixed exterior truncation: zero correction
      potential on the outer cylinder, zero correction flux on axial cuts, and natural total-field
      tangency on the plasma side. At scaled radial resolution, its center field approaches the
      exterior limit from above while the finite-wall Neumann variant approaches from below; their
      gap decreases `0.01836,0.00794,0.00490,0.00356 T` as outer radius moves
      `0.50,0.65,0.75,0.82 m`. This is quantified truncation uncertainty, not yet an
      outer-boundary-independent claim. The merged clean-core `MgridField` now uses the same moving-
      annulus external-field adapter as direct coils; uniform-table conversion is exact and a
      33-by-65 two-coil mgrid agrees with direct Biot-Savart below `5e-3` relative RMS on the
      annulus. A scheduled full test now solves the same beta-zero free-boundary equilibrium from
      direct coils and a 49-by-97 mgrid: both reach component-wise `1e-12`, with the LCFS agreeing
      within `5e-3` relative and the annulus field within `8e-3`. A Dirichlet-to-Neumann/boundary-
      integral exterior operator and nonaxisymmetric coils remain. A radial-only cylindrical
      cosine/Bessel DtN prototype was rejected despite exact modal tests: with open axial ends and
      coils in the exterior, its center field still shifted `0.09090,0.09718,0.10189 T` as the
      outer radius moved `0.50,0.65,0.82 m`, while plasma-side `B.n` worsened. M5 therefore requires
      a full closed-surface boundary integral (lateral surface plus axial apertures), not another
      local or radial-only outer condition. The first full-surface tranche now closes the moving
      lateral LCFS with regular `r=sqrt(s) a(theta)` end disks and consistent outward weighted
      normals. Exact cylinder area/volume, zero net normal, all divergence-theorem first moments on
      a shaped tube, cap-ring continuity, and a JAX boundary derivative pass. The released
      `virtual_casing_jax>=0.0.2` low-level kernels now evaluate nonsingular off-surface double
      layers and single-layer gradients on this mixed wall/cap quadrature. Constant-density solid
      angle converges to one inside and zero outside; the gradient reaches its far-field monopole
      limit. Axisymmetric kernel quadrature has an angular resolution independent of the one-node
      equilibrium representation. Green's third identity converges for constant, linear, and
      quadratic harmonic polynomials under joint disk-radial, axial, and angular refinement and
      vanishes at exterior targets. Repeated polar cap centers now map to one density unknown, and
      cap rims reuse the lateral end-ring unknowns; a continuity test proves exact expansion from
      this unique collocation grid back to quadrature nodes. This removes the known duplicate-node
      rank defect before system assembly. A periodic-side/polar-cap triangular mesh over those
      nodes is watertight (every edge has two owners), outward oriented, nondegenerate, and shows
      the expected second-order cylinder area/volume convergence. Disposable boundary-limit tests
      rejected offset collocation (condition numbers `1e6--1e19`, unreliable densities) and
      equal-area-disk self terms (8.3% linear-harmonic boundary error at about 1,900 unknowns).
      Following Duffy (1982), the next accepted path is local transformed singular quadrature with
      explicit cap-rim grading; smooth-surface QBX alone does not resolve the artificial sharp rim.
      The first JAX Duffy primitive now regularizes a vertex-singular linear triangle. Orders
      `2,4,8,16` converge monotonically to the analytic right-triangle single-layer integral, with
      order 16 at `1.4e-14`; linear density and geometry/density gradient identities pass. The
      assembled interior identity uses `S(q)+K(u-u_target)=0`, preserving the constant nullspace
      and avoiding an invalid smooth-surface jump coefficient at the rim. For harmonic `u=x,z`,
      its worst normalized residual falls `3.47e-3 -> 1.78e-3` from 154 to 862 nodes. Duffy orders
      8 and 10 agree, so panel/rim refinement, not quadrature order, is now the measured limiter.
      Axisymmetric densities now reduce exactly to `nxi+2(ns-1)` ring unknowns while retaining all
      angular source panels. Evaluating one representative target per orbit cuts the 57-unknown,
      1,762-vertex Jacobian from 67.5 to 2.75 s; its `u=z` recovery is unchanged at 3.27% with
      condition number 19.1. Power grading toward the cap rim then reduces 45-unknown `u=z`
      recovery error from 4.39% ungraded to 0.222%, 0.0589%, 0.0264%, and 0.0216% at grades
      2, 2.5, 3, and 3.5 without degrading conditioning (`cond=17.0`). A differentiable weighted
      saddle solve now removes the constant gauge and reports compatibility, condition, gauge, and
      equation residual; its forward JVP is finite. The graded MMS has roundoff flux/gauge closure
      and `8.9e-9` panel-discrete equation residual. The decaying exterior equation correctly adds
      the identity, `S(q)+K(u-u_target)+u=0`, has no constant gauge, and reverses the representation
      sign. A zero-flux dipole MMS closes its algebraic residual below `3e-14` with condition below
      5; boundary error decreases `14.9% -> 5.44%`, and exterior gradient error reaches 1.12%.
      Side-boundary field reconstruction from Neumann data plus the CGL potential derivative
      converges `48.4% -> 14.5% -> 5.28% -> 3.08%`; the finest solve takes 5.7 s with condition
      number 4.02. Endpoint exclusion does not improve the rate, while spectral filtering,
      off-surface extrapolation, and Richardson correction make it worse. Linear side-panel density
      interpolation remains the accuracy limiter. The finest grid is now accepted for a guarded
      M6 coupling study, but the BIE field is not yet the default stress backend.
      The implementation is split by ownership into 355-line geometry/maps, 299-line panel/Duffy,
      and 434-line BIE/solve modules; the public `vmec_jax.mirror` API is unchanged.
      Analytic Green gradients remove axis `NaN` from differentiating safe-distance branches.
      Duffy panel field evaluation reduces the two-coil near-cap uniform-field reconstruction error
      `0.998% -> 0.573% -> 0.369%` over three meshes (the global polar rule gave
      `10.1% -> 6.84% -> 4.94%`), with roundoff compatibility and condition number below 10.
      Shaped/finite-beta data and actual replacement of the annular truncation remain promotion
      gates; this manufactured coil cancellation is not yet free-boundary coupling.
      A tested adapter now converts an axisymmetric plasma end field plus direct coils into complete
      wall/cap Neumann data and matches a uniform-cylinder construction to roundoff.
      `solve_axisymmetric_exterior_vacuum` now consolidates closure, Neumann assembly, the decaying
      solve, and lateral total-field reconstruction. Its direct-coil tangency is below `2e-12`,
      compatibility is below `2e-12`, conditioning is below 10, and a complete boundary-shape JVP
      is finite and nonzero. The M5-to-M6 differentiation seam is therefore implemented.
      The seam is now an opt-in `vacuum_backend="exterior"` in the actual coupled solve and beta
      continuation. A two-coil `(ns,nxi,ntheta_panel)=(5,7,8)` scan at beta 0 and 10% converges in
      seven evaluations per point with maximum residuals `7.93e-16/2.95e-15`, tangency below
      `6.3e-17`, stress below `1.3e-15`, and center expansion `0.252576 -> 0.255603` m. This closes
      the first nonlinear unbounded-equilibrium gate; resolution parity remains open.
      A beta-zero `(5,7,8),(7,13,12),(9,17,16)` resolution study gives center radii
      `0.2525753,0.2531506,0.2531155` m and axis fields `0.0840027,0.0835434,0.0835623` T; the last
      two agree within `1.39e-4/2.26e-4` relative. Compatibility improves
      `1.60e-8 -> 1.02e-9 -> 5.92e-10`, condition stays below 3.31, and force stays below `5.8e-15`.
      Monolithic forward AD at the 120-variable third grid was terminated at 9.67 GB RSS. An
      adaptive exact-JVP Jacobian now keeps monolithic AD through 80 variables and chunks six
      columns above it; the third grid then converges in 118.8 s at 5.48 GB RSS. Physics convergence
      is established at beta zero. The office A4000 then completes the full third-grid beta 0/10%
      scan in 119.5 s at 1.99 GB host RSS. Beta 10% gives radius/axis field
      `0.2561004 m/0.0797034 T`, agreeing with `ns=7` within `3.0e-5/3.65e-4` relative; residual is
      `6.92e-15`, compatibility `6.43e-10`, and condition 3.29. A permanent full three-grid test
      gates observables at `5e-4` and compatibility at `2e-9`. The finite-beta two-coil accuracy
      gate is closed; higher-order panels and lower CPU memory remain.
      The root beta-scan example now exposes the exterior backend and writes compatibility/condition
      diagnostics. A solved 0/10% exterior render at `ftol=1e-12` produced all three figure panels;
      the 3D LCFS, coils, cap-to-cap lines, field arrows, `|B|`, pressure, beta, and residual plots
      were visually inspected. Shared beta diagnostics now support both vacuum field layouts.
      Shaped/finite-beta exterior MMS, higher-order side density, tighter trace/near-field convergence,
      and coupling that deletes the finite outer cylinder remain the next M5 gates.
      **Spectral side-density option (2026-07-11):** global CGL/Fourier interpolation of lateral
      Dirichlet and Neumann data now replaces piecewise-linear density inside side panels when
      `spectral_side_density=True`; geometry and cap density remain linear. It reproduces resolved
      Fourier-Chebyshev data to `2e-13`, preserves axisymmetric/nonaxisymmetric shape JVPs, and
      improves the medium dipole boundary-potential error `5.44% -> 1.19%` and far-field gradient
      error `1.12% -> 0.72%`, with condition below 4. The coupled API and beta continuation expose
      the option, but the default remains linear pending the coupled findings under M7.
   7. **M6 — axisymmetric finite-beta free boundary.** Vary the lateral interface and interior
      state jointly, with beta continuation `0, 0.01, 0.03, 0.10` and hot restarts. Validate
      isotropic and anisotropic cases against an independently generated Pleiades/WHAM-style
      reference, paraxial pressure balance, outward flux-surface expansion, and the expected
      central diamagnetic trend `B0/Bvac approximately sqrt(1-beta)` in its validity regime.
      **STATUS (2026-07-10): coupled isotropic scan and beta diagnostics landed.** The production residual
      solves plasma interior force, vacuum-potential stationarity, and free-side normal stress as a
      square system; it does not vary the Neumann vacuum functional as a shape energy. A coupled
      mass-amplitude unknown and central-pressure equation make scan beta an achieved quantity,
      rather than the pressure from an uncorrected reference mass. Direct two-coil solves at beta
      `0,1%,3%,10%` all reach `<5.9e-15` residual and `<2.7e-15` active
      stress error. At `(ns,nxi,nrho)=(7,13,7)`, solved center radii increase monotonically
      `0.253077,0.253367,0.253957,0.256132` through requested beta 10%. The corresponding achieved
      central beta is 10%, volume-averaged beta is 3.37%, and `B_axis(beta)/B_axis(0)=0.95218`,
      within 0.37% relative of the paraxial `sqrt(1-beta)` prediction. Extending the diagnostic scan
      to requested beta 25% and 50% gives 3.32% and 8.03% center-radius expansion and monotonic
      diamagnetic field depression while retaining `<4.9e-15` nonlinear residual. Increasing the
      seed radius from 0.25 m to 0.50 m reduces, rather than amplifies, relative LCFS motion at fixed
      beta, so small visual displacement is not evidence of uncoupled pressure. The root example
      now separates nominal, achieved-central, and volume beta and renders geometry displacement,
      pressure, fields, coils, cap-to-cap lines, and convergence. Full beta continuation now hot
      restarts the lateral boundary, plasma interior, and vacuum potential. A formal
      `(ns,nxi,nrho)=(5,7,5),(7,13,7),(9,17,9)` test requires the last two center radius, field,
      and achieved-beta triples to agree within `1e-4` relative; normalized vacuum `B.n` decreases
      `1.60e-2,4.25e-3,1.85e-3`. Separate fixed-mass solves from +/-10% free-side radius
      perturbations agree to `2e-12` absolute in boundary and `2e-11` relative in `B^2`.
      Compact, atomic `.npz` restart files now preserve only boundary, plasma state, vacuum
      potential, and calibrated mass scale; loading validates schema, finiteness, and both grid
      shapes. Continuation now propagates the solved mass scale as well as geometry and vacuum state,
      and can resume a beta suffix from a loaded restart while retaining the original beta-zero
      pressure reference. The root beta-scan example exposes both save and resume inputs. Source
      ownership is now explicit: the 483-line `mirror/vacuum.py` contains only annulus operators,
      while the 387-line `mirror/free_boundary.py` owns the coupled nonlinear solve and result.
      The coupled solve now also accepts a consistent ANIMEC closure, solves its positive pressure
      amplitude, balances the interface with `p_perp`, calibrates requested beta from midplane
      `p_perp`, and rejects firehose/mirror-elliptic states. A genuinely bi-Maxwellian two-coil
      `0,1%` scan reaches `9.1e-16` residual and `9.9e-16` stress error with a 4.93% pressure
      anisotropy and outward LCFS motion. A scheduled 10% study at
      `(ns,nxi,nrho)=(5,7,5),(7,13,7),(9,17,9)` stays below `9.2e-15` residual/stress, reduces
      vacuum `B.n` `1.56e-2,4.25e-3,1.85e-3`, and agrees in the last two grids within `5.6e-4`
      over center radius, field ratio, volume beta, and anisotropy. The root example exposes the
      same pressure-model switch. At `ns=7`, requested beta `0,10%,25%,50%` remains elliptic with
      `<5.8e-15` residual, monotonic radius expansion, and field ratios
      `1.000,0.954,0.880,0.745`. A tabulated `p_parallel(s,B)` sampled from the same closure reaches
      `2.25e-15` at 1% and reproduces radius/field ratio `0.2533157/0.995514`; the example exposes
      this third pressure-model option.
      An independent Pleiades Green-function study at upstream commit `0161abb3` converges its
      1%,3%,10% field ratios to `0.995370,0.986049,0.952754` on a 51x101 grid; the 10% `vmec_jax`
      ratio differs by 0.061% relative. Higher-resolution vacuum tangency/exterior closure,
      independent boundary curves remain.
      **Unbounded high-beta gate (2026-07-11):** the exterior backend now continues through
      requested beta `0,10%,25%,50%` at `ftol=1e-12`. All three grids converge below `8.1e-15`.
      On `(ns,nxi,ntheta_panel)=(9,17,16)`, beta 50% gives center radius `0.2726602 m`, central
      field ratio `0.747645`, volume beta `0.219148`, exterior compatibility `2.09e-9`, and
      condition 3.23. Medium-to-fine relative changes are `7.4e-4` in radius, `4.2e-3` in field,
      and `4.7e-3` in volume beta, so the permanent gate keeps `5e-4` below 50% and uses an honest
      `5e-3` high-beta tolerance. The root example defaults to this unbounded backend and scans
      `0,1%,3%,10%,25%,50%`; annulus output remains available in a separate result directory.
   8. **M7 — nonaxisymmetric finite-beta free boundary.** Add helical coils/boundaries, then require
      3D force, interface, field-line, and resolution gates. This lane is supported only after M6;
      no axisymmetric boundary replicated in theta counts as a 3D validation.
      **STATUS (2026-07-10): first full-theta exterior seam landed; equilibrium open.** A Cartesian
      field conversion now handles all contravariant components, and a general closed-surface
      Neumann adapter samples finite-current theta-dependent lateral/end-cut data plus direct coils.
      A genuine `mpol=1,ntheta=3` shaped case matches metric and Cartesian `|B|^2` within `5e-13`,
      has lateral `B.n < 2e-15`, and closes integrated flux within `2e-3`. The full-theta exterior
      Dirichlet solve now has condition below 20 and equation residual below `2e-12`; theta/axial
      tangential reconstruction gives total lateral `B.n < 3e-15`, and its complete shape JVP is
      finite and nonzero. The shared `solve_free_boundary_cli` now couples this closure to the full
      theta-dependent plasma/interior/interface residual. Two oppositely offset end coils retain a
      genuine midplane `m=1` radius spread (`0.433 -> 0.467` mm at beta 0 -> 10%) instead of
      relaxing to replicated axisymmetry. Both points converge in 9/7 evaluations with force below
      `3.7e-15`, tangency below `3e-17`, stress below `2.1e-15`, compatibility about `1.05e-3`, and
      condition below 3.7; theta-zero center radius expands `0.201985 -> 0.204418` m. The coarse 3D
      smoke has a measured `2e-3` compatibility gate while axisymmetric production remains `1e-6`.
      The shared `solve_beta_scan_cli` now carries finite current through the reference and every
      hot-started beta point; the old axisymmetric name is an alias. Next: nonaxisymmetric resolution
      convergence and independent coil/field references.
      **Resolution blocker measured:** `(ns,ntheta,nxi)=(5,5,5),(7,5,7),(9,7,9)` all converge at
      beta 0/10% below `1.2e-14`, with compatibility improving `~1.6e-5 -> ~1.2e-5 -> ~8e-8` and
      condition below 3.65. However, beta-zero theta-zero center field is
      `0.083300,0.084187,0.083302` T and theta radius spread is `0.638,0.362,0.582` mm; beta-10
      values are similarly nonmonotonic. The last two point fields differ by about 1%. Two-point
      wall/RSS measurements are 25.6 s local, 287.7 s/2.74 GB A4000, and 896.1 s/3.65 GB A4000.
      Do not schedule a still larger brute-force grid. Next: define Fourier-mode/global observable
      convergence, improve the exterior trace order, and reduce host Jacobian cost before rerunning.
      `boundary_fourier_amplitudes` now removes odd-grid peak-to-peak bias and recovers analytic
      `m=0,1,2` amplitudes to `5e-17`; future studies must gate this modal metric plus global
      volume/energy/theta-averaged fields rather than raw theta-node extrema.
      `summarize_nonaxisymmetric_beta_scan` now makes those gates reusable: achieved and volume
      beta, theta-mean midplane radius/field, Fourier amplitudes, plasma volume, and total energy
      are computed from the solved state with the same quadrature used by the equilibrium.
      A one-primal `jax.linearize` plus batched `lax.map` Jacobian was measured and rejected: the
      axisymmetric three-grid test exceeded 7.5 GB RSS before completion at 170 s, versus 5.48 GB
      for the existing host-chunked exact JVP columns. Keep the bounded-memory path until a
      matrix-free or block-eliminated nonlinear solve removes the dense SciPy Jacobian contract.
      The coarse genuine-3D continuation now also reaches beta `25%` and `50%` without stalling:
      residual stays below `3.7e-15`, normal stress below `2.1e-15`, and vacuum tangency below
      `4.4e-17`. From beta zero to 50%, mean midplane radius grows `0.201794 -> 0.217968 m`, mean
      central field falls `0.082215 -> 0.071173 T`, and the physical `m=1` radius amplitude grows
      `0.255 -> 0.421 mm`. This closes nonlinear-continuation robustness, but not the documented
      spatial-resolution blocker; no 3D production-accuracy claim is made yet.
      The beta-0/50% endpoint study at `(7,5,7)` and `(9,7,9)` now uses global diagnostics. At
      beta 50%, medium-to-fine relative changes are `6.46e-4` mean radius, `1.09e-3` mean field,
      `4.45e-4` volume, and `5.28e-4` total energy; compatibility improves `1.18e-5 -> 2.55e-8`.
      The `m=1` amplitude changes `0.305 -> 0.131 mm` (132% relative to the fine value), so global
      high-beta response is stable to 0.2% while local 3D shape remains blocked. The exact inputs,
      endpoint values, and 519/1410 s A4000 timings live in
      `benchmarks/mirror_free_boundary_nonaxisymmetric.json`.
      Spectral side density does not close this gate. Axisymmetric beta-50 medium/fine convergence
      changes only marginally (`4.19e-3 -> 4.15e-3` for field) and the fine spectral/linear fields
      agree within `1.1e-4`. In 3D, spectral beta-50 medium/fine radius/field/volume/energy changes
      are `4.87e-4/3.09e-3/2.32e-4/5.79e-4`, but `m=1` changes `0.154 -> 0.0569 mm` (171%) and the
      fine spectral field differs from linear by 14.1%. The medium/fine endpoint pairs cost
      565/1830 s and 3.66/5.39 GiB RSS on A4000. Keep this research option opt-in. The next accuracy
      experiment must raise side geometry and cap density order together; do not add another
      brute-force grid or treat density interpolation alone as promotion evidence.
      Curved parametric side panels now evaluate exact Fourier-CGL side geometry at Duffy nodes.
      For the dipole MMS, boundary error decreases from `0.0845 -> 0.00786 -> 0.00131` over
      `(ns,nxi,ntheta)=(9,13,16),(13,21,24),(17,29,32)`, about 2.6 times below the fine planar-
      side result; axisymmetric and 3D shape JVPs remain finite. Separating residuals shows the
      remaining fine-grid error is concentrated on the planar caps/rim, so side refinement alone
      is complete. A density-only Fourier/local-cubic cap prototype improves the medium dipole
      field error `1.056% -> 0.503%`, but raises the coarse `ns=5` axisymmetric/3D condition numbers
      `3.00 -> 242.6` and `2.66 -> 53.4`; it was rejected. The accepted paired method instead uses
      exact polar/star-shaped cap geometry and selects linear, quadratic, or cubic radial density
      interpolation according to ring count, retaining Fourier interpolation in angle. Circular
      cap area and orientation are exact to roundoff; axisymmetric and 3D shape JVPs pass. At the
      medium dipole grid, boundary error improves `0.786% -> 0.725%`, off-surface field error
      `1.056% -> 0.121%`, and condition remains 4.89. Coarse axisymmetric/3D conditions are
      `3.07/2.73`. Both actual beta `0,10%,25%,50%` coupled gates pass at `ftol=1e-12` in 436 s.
      The equilibrium API exposes this as `exterior_high_order_cap_panels`. Endpoint benchmarking
      rejects it for production: axisymmetric compatibility is `4.71e-6/3.35e-5` at beta 0/50%
      on `(7,13)` and `5.74e-6/1.67e-5` on `(9,17)`, above the `1e-6` gate. The medium 3D pair costs
      2,149 s and 9.26 GiB RSS (3.8x/2.5x the spectral pair), while the fine pair did not finish in
      94 minutes at 11.9 GiB RSS and 9.25 GiB GPU memory. Keep the tested API opt-in for quadrature
      research; do not spend another grid on this formulation.
      **Final M7 refinement verdict (2026-07-12): unsupported, deferred.** The consistent
      `(5,3,5),(7,5,7),(9,7,9)` spectral-side/linear-cap ladder converges all six beta-0/50
      equilibria below `1e-14`; compatibility falls from about `1e-3` to `1e-7`. At beta 50%,
      medium-to-fine mean radius/field/volume/energy changes are
      `1.70e-3/1.01e-3/4.36e-4/5.67e-4`, but center `m=1` changes by 72% (36% at beta zero).
      Excluding only endpoint CGL nodes was itself resolution-dependent, so future diagnostics
      integrate the Fourier-CGL interpolant on fixed `|xi|<=0.75`; the completed runs retain raw
      profiles. The 3D lane is not promoted and no larger dense-Jacobian run is justified. Reopen
      only after a matrix-free/block-eliminated BIE changes the cost/accuracy formulation.
   9. **M8 — toroidal stellarator–mirror hybrid.** Model the closed square/rounded-square torus with
      straight mirror sides and stellarator corners using ordinary VMEC Fourier equilibrium.
      Piecewise splines are low-dimensional axis/boundary design controls projected to Fourier.
      Validate mode convergence and `wout` parity with VMEC2000 before considering a native spline
      equilibrium state. Then run the 16-coil free-boundary beta scan using solved boundaries.
      **STATUS (2026-07-11): coil-informed fixed-boundary Fourier equilibrium and a genuine
      low-beta NESTOR branch are accepted at `ftol=1e-8`; tolerance promotion and the 1--50%
      toroidal free-boundary scan remain open.** A compact
      clean-core module samples a superellipse square axis with four straight mirror
      regions and localized rotating corner ellipses, then least-squares projects the single
      real-space target into ordinary `RBC/ZBS` and axis coefficients. Geometry tests prove four
      corner regions, side straightness below 2 mm, aligned side sections, and corner orientation
      span above 0.2 rad. Maximum projection error decreases 13.0, 0.912, 0.262, and 0.077 mm at
      `(mpol,ntor)=(4,8),(6,16),(6,20),(8,24)`. No legacy native-spline/replay solver was restored.
      The circular-axis member converges component-wise to `1e-12` in 1,870 iterations. The axis
      superellipse exponent is now a continuous low-dimensional control: continuation reaches
      `p=4.20` on `ns=3` at `1e-8`, but lifting that state to `ns=5` stalls near `1.14e-6`; direct
      `ns=5` continuation stops near `p=3.05`. Linear circle-to-square continuation reaches 44%
      before the next step stalls, and higher Fourier bandwidth does not move that limit.
      VMEC2000 reproduces the default and unshaped-square 5,000-iteration residuals to the printed
      digits. Its 10,000-iteration DELT scan is best at `DELT=0.1`, reaching
      `2.53e-7/1.60e-7/3.62e-7` but not convergence. The current 2D preconditioner was also rejected
      (>2 minutes versus 8.2 seconds for 1D). The square target is therefore a shared equilibrium
      basin/geometry limitation, not a vmec_jax parity defect. Next: construct a coil-informed,
      curvature-bounded target family and repeat the vmec_jax/VMEC2000 parity gate. Do not add the
      root solved example or 16-coil beta scan until that gate passes.
      A 4,096-point spectral audit at the 44% continuation limit finds minimum curvature
      `0.332 m^-1` and tightest curvature radius 0.559 m versus 0.1 m minor radius. The basin is lost
      before the exact-square zero-curvature limit, so a simple curvature-floor constraint does not
      explain or fix the stall. Do not spend another run on imposed superellipse continuation;
      extract the next boundary/axis target from the 16-coil vacuum flux geometry.
      **Coil-informed target (2026-07-11):** clean-core simsopt-style
      `planar_ellipse_coils` and `square_mirror_coils` constructors now replace the deleted legacy
      example's local builders. A differentiable RK4 Biot-Savart trace extracts one closed magnetic
      axis turn from the actual 16 coils. It is planar to `2.4e-17 m`, closes within integration
      error, keeps side straightness below 1.7 mm, and spans radius `1.500--1.861 m`; `ntor=16/20`
      reconstruct it within 0.226/0.057 mm. The axis field varies by a factor 1.70, so the accepted
      boundary seed scales its cross-section as `a proportional to |B_axis|^-1/2`. This thin-tube
      seed changes the `ns=5, ftol=1e-8` solve from a 1,171-iteration near-stall to convergence in
      62 iterations. Continuing side elongation, corner ellipticity, and corner rotation in 10%
      increments reaches the full requested stellarator shaping; every stage converges. With a
      flat 3 kA toroidal-current profile, the last stage takes 509 iterations with components
      `(1.00e-8,3.39e-9,6.59e-9)` and produces `iota=-0.805...-0.807`. The Fourier fixed-boundary
      geometry blocker is closed at `1e-8`. Tightening remains open: the unshaped base reaches
      `1e-9` in 212 iterations, but `DELT=0.02,0.05,0.1,0.2` does not reach `1e-10`, and the shaped
      state destabilizes when polished to `1e-9`. With 3 kA current, the unshaped `1e-8` base takes
      1,518 iterations and does not reach `1e-9`. The production 2D preconditioner at steps
      `0.25,0.5,1.0` worsens the 300-iteration residual to `2.9e-8,2.8e-8,5.4e-8`; it is rejected
      for this lane. An independent VMEC2000 restart from the vmec_jax WOUT converges in 1,071
      iterations without Jacobian resets, with identical fixed LCFS to roundoff, magnetic energy
      within `3.73e-7`, LCFS `|B|` within `2.52e-4` relative L2, and iota within `1.09e-3`
      relative L2. This accepts basin/solver parity at the validated `1e-8` floor, not
      machine-precision parity. The reproducible deck is emitted by the example and compact data
      live in `benchmarks/mirror_hybrid_fixed_boundary.json`. The toroidal NESTOR driver now accepts
      a same-resolution free-boundary hot state, retains its evolved LCFS, and rebinds constraint
      baselines; beta scans no longer cold-start every point. Two hot-start defects are closed: a
      fixed seed can no longer terminate before NESTOR activation, and the pre-vacuum best-residual
      checkpoint is reset when vacuum pressure turns on. The coil trace now supplies signed
      `B_phi`, a toroidal-plane flux-tube scale, and a signed `PHIEDGE` estimate. With that matched
      flux, the unshaped beta-zero LCFS executes NESTOR and converges at `ftol=1e-8` in 3 iterations
      (`DELT=0.002`, volume `0.33298 m^3`, aspect 16.55). The optional `max_vacuum_skip` cap can
      force a full NESTOR update every iteration without changing default VMEC2000 cadence. Next:
      close finite-pressure stepping, design a hybrid-specific scaled block preconditioner, then
      use this continuation path for the 16-coil `0--50%` beta scan.
      **Finite-beta boundary isolated (2026-07-11):** fixed-LCFS pressure prediction followed by a
      strict fixed corrector and NESTOR release converges target beta `0.05,0.10,0.15,0.20,0.25%`;
      achieved values are `0.0511--0.2555%`, and every free correction takes 3 iterations below
      `1e-8`. The next `0.30%` fixed predictor reaches `1e-7`, but its `1e-8` corrector exhausts
      5,000 iterations. Halved beta steps and a `1e-7 -> 1e-8` restart do not move this boundary.
      `ntor=12/16` fail the beta-zero `1e-8` gate, so `ntor=20` is equilibrium resolution, not just
      design overhead; splines can reduce controls but cannot remove these modes. The generic 2D
      block over 100 iterations improves lambda `9.14e-9 -> 9.48e-10` but worsens radial force
      `9.98e-8 -> 1.90e-7`; tighter/looser step tuning is rejected. VMEC2000 is not more robust:
      a cold run remains at `(4.88e-6,1.34e-6,3.56e-6)` after 4,320 iterations and a free-metadata
      WOUT restart at `(1.42e-5,1.08e-5,1.20e-6)` after 5,200, using only ~85 MiB. Compact evidence
      is in `benchmarks/mirror_hybrid_free_boundary.json`. The next implementation is a constrained,
      R/Z/lambda-scaled fixed-boundary corrector; do not start the 1--50% production scan before it.
      The matrix-free Newton system now removes fixed edge rows, axis-null harmonics, lambda-axis
      values, and zero/gauge modes, reducing the hybrid solve from 3,390 stored coordinates to
      2,294 active unknowns; the existing stiff aspect-100 full test still reaches the same
      equilibrium in fewer iterations. A direct exact-direction audit shows a 1% Newton step is
      descending while the former 25--100% steps are not, so Jacobian-aware backtracking is the
      remaining corrector work rather than more fixed step-factor scans. **Corrector barrier
      closed (2026-07-11):** the safeguarded corrector minimizes the largest normalized physical
      force component, rejects sign-changing Jacobians, falls back to the regular VMEC update when
      no Newton candidate descends, and can attempt the expensive matrix-free solve on a static
      cadence. With step 0.25 and cadence 10, the former 0.30% blocker converges in 291 iterations
      to `(9.99e-9,2.49e-9,3.28e-9)` (299 s and 1.57 GiB peak RSS on an RTX A4000). Releasing that
      state through a full NESTOR update converges in 3 iterations to achieved beta 0.3064%, with
      volume `0.332983315 m^3` and aspect 16.5511. The old 0.30% blocker is closed; the next finite
      task is an adaptive continuation through 1, 3, 10, 25, and 50%, retaining each solved free
      LCFS as the following fixed predictor boundary. Fixed solves can now return a typed
      unconverged checkpoint instead of raising, so long correctors resume their own state in
      explicit 1,000-iteration chunks. `SolveResult.newton_history` records the accepted line-search
      step and inner GMRES residual per nonlinear iteration for convergence plots and stall audits.
      **Continuation audit:** target beta 0.50%, 0.40%, and 0.35% are rejected after 1,000
      corrector iterations at maxima `2.22e-8`, `1.49e-8`, and `1.20e-8`. Resuming the serialized
      0.35% state for a second 1,000-iteration chunk worsens the result to
      `(1.25e-8,3.31e-9,1.27e-8)`, so more blind iterations are not the next method. Next: use the
      recorded accepted-step and GMRES histories to compare R/Z/lambda block scaling and inexact
      Newton regularization on the reproducible 0.35% predictor, accept a method only if all three
      physical residuals descend, then restart adaptive continuation. No result above 0.30% is
      claimed yet. **Scaling screen:** changing GMRES `rtol` from `1e-2` to `1e-3` at restart
      80 is bit-identical because the Krylov budget stops near linear residual `4.1e-6`; tolerance
      is not the knob. The lambda inner-force norm is `4.64e-5` versus `4.97e-6/4.52e-6` for R/Z,
      although physical FSQL is already smallest. Optional left row scaling leaves the exact Newton
      root and physical line search unchanged. Lambda scales 0.1 and 0.03 reduce the 120-iteration
      maximum from `2.34e-8` to `1.76e-8`; 0.03 lowers the linear residual to about `2.0e-6` and
      accepts four full 0.25 steps. Lambda scale 0.01 closes the 0.35% corrector in 51 iterations
      at `(9.85e-9,4.71e-9,3.76e-9)`: all five scheduled Newton attempts take the full 0.25 step,
      and the inner residual is `9.1--9.2e-7`. The full NESTOR release then converges in 3
      iterations at achieved beta 0.3575%. Next: resume adaptive continuation from this solved free
      LCFS with the accepted scaling, retaining the strict component gate through 50%.
      **Balanced continuation:** twelve additional fixed-corrector/NESTOR points now reach target
      beta 0.6978125% (achieved 0.7128%). Lambda row scales decrease from 0.01 to 0.005 to 0.004 as the
      limiting component shifts; every free release converges in 3 iterations. At the endpoint the
      fixed corrector takes 315 iterations and lands at
      `(9.96e-9,1.67e-9,9.997e-9)`, so FSQR and FSQL are simultaneously tangent to the gate.
      Compact data live in `benchmarks/mirror_hybrid_balanced_continuation.json`. Next: replace
      manually staged row scales with a tested dynamic block-balancing rule (bounded and frozen
      during each GMRES solve), then continue toward 1%; do not extrapolate this branch to 50%.
      The bounded rule is now implemented: at each Newton attempt it targets scaled lambda norm
      `0.1 * max(||g_R||,||g_Z||)`, clips the scale, freezes it for GMRES, and records it in
      `newton_history`. Target-ratio screens 0.05/0.1/0.2 select 0.1; it advances one more genuine
      point in 261 corrector + 3 NESTOR iterations at
      `(9.996e-9,1.42e-9,9.976e-9)`. Cadence screening selects 20; it advances two further points,
      but the final accepted pressure increment collapses to `6.25e-5` and the endpoint takes 300
      corrector iterations at `(9.9995e-9,3.77e-10,9.9836e-9)`. Since dynamic scaling and cadence
      do not remove the tangent gate, next
      diagnose Jacobian conditioning, nested-surface quality, and a possible equilibrium/bifurcation
      limit before spending more continuation steps. The endpoint audit excludes geometric failure:
      minimum active-half-mesh `|sqrt(g)|` is 74% of its median, no sign change occurs, spectral-tail
      L2 content is `1.06e-4`, and iota is smooth over `0.07191--0.07241`. The LCFS displacement
      from target beta 0.30% is only 78 nm. The blocker is numerical conditioning, not lost nested
      surfaces. Right scaling and GCROT were screened and rejected: right scales are numerically
      indistinguishable, while GCROT misses the gate and costs 40% more. The matrix-free corrector
      was provisionally deferred for 1--50% continuation pending a true coupled radial/Fourier
      block structure or equivalent Schur preconditioner.
      A permanent root example now reconstructs the coils and mgrid, performs the genuine
      fixed-predictor/fixed-corrector/NESTOR sequence, and requests the complete ladder through
      1%, 3%, 10%, 25%, and 50%. It writes WOUT and summary data only for accepted equilibria,
      then plots 3D coils/solved LCFS/two-turn field lines, `|B|`, cross-sections, profiles, and
      predictor/corrector/free-release force histories. It stops with the typed sub-1% barrier;
      it never substitutes prescribed or interpolated high-beta boundaries. A reduced scheduled
      test runs a real beta-zero square-coil NESTOR solve and asserts the complete figure set.
      **Krylov-width result:** a controlled screen shows that more restart cycles at Arnoldi width
      80 do not cross target beta 0.7040625% after 400 iterations (`1.016e-8/2.41e-10/1.012e-8`,
      314.8 s). Width 120 with only three restart cycles converges in 121 corrector iterations
      (`9.94e-9/6.77e-10/9.98e-9`, 115.3 s), and NESTOR releases it in 3 iterations at achieved
      beta 0.7190%. At target beta 0.72%, width 120 remains GPU-bound beyond 578 s, while width 160
      converges in 171 corrector iterations and releases in 3 at achieved beta 0.7343%, with all
      components below `1e-8`. Width 160 also accepts targets 0.735% and 0.75% in 291 and 411
      corrector iterations, reaching achieved beta 0.7643%. Target 0.775% then converges in 431
      corrector iterations and releases at achieved beta 0.7896%. Target 0.8% then converges in
      565 corrector iterations and releases at achieved beta 0.8147%. A direct step to target
      0.825% exhausts 1,000 iterations at `1.075e-8/6.55e-10/1.001e-8`; bisecting to 0.8125%
      converges in 554 iterations and releases at achieved beta 0.8272%. A second direct 0.825%
      attempt from that midpoint also exhausts 1,000 iterations; bisecting again to 0.81875%
      converges in 746 iterations and releases at achieved beta 0.8333%. Accepted increments have
      collapsed to `6.25e-5`, so the matrix-free path is not viable to extrapolate to 1--50%.
      A residual-weighted radial/channel exact-JVP coarse Schur correction fails the exact 0.825%
      barrier at `1.229e-8/8.12e-10/1.219e-8` after 1,000 iterations, worse than the baseline.
      It is removed. Any future preconditioner must retain radial, channel, and Fourier coupling.
      Arnoldi widths 240 and 320 are also rejected at the same barrier: both remain unconverged
      after 1,020 s, already slower than the width-160 failure, and are terminated. More Krylov
      width is not the next method.
      An exact-JVP implementation of VMEC2000's dense Fourier/channel,
      nearest-radial block-tridiagonal Hessian also fails at
      `2.657e-8/5.85e-10/2.599e-8` after 1,000 iterations and is removed. Synthetic and stiff
      axisymmetric tests pass, isolating the problem to the hybrid's nonlocal constrained
      Jacobian. The next method must preserve that nonlocal coupling.
      A complete 2,294-unknown reduced Jacobian is then assembled and solved directly. It fits in
      5.5 GiB on the RTX A4000 but finishes the same cap at
      `7.100e-8/5.51e-10/4.499e-9`. Since exact linear coupling does not globalize the residual,
      the Fourier free-boundary hybrid lane is **deferred above achieved beta 0.8333%**. Do not
      extrapolate it to 1--50% or add more solver variants without a new equilibrium/globalization
      formulation. The straight-axis mirror lanes remain the validated 50% high-beta capability.
   10. **M9 — implicit differentiation and optimization.** Wrap the converged mirror residual in a
       `custom_vjp`; solve JVP/VJP systems matrix-free with the primal preconditioner. Validate
       boundary, pressure, current, and coil derivatives against central differences. Do not
       differentiate through iteration histories or restore fingerprint/replay machinery.
       **STATUS (2026-07-12): fixed-boundary adjoint landed.** The packed, constrained
       energy-gradient residual now has an exact reverse-AD transpose solve using the primal
       separable radial/poloidal/axial preconditioner. One solve returns total gradients with
       respect to boundary, axial flux, mass/pressure profile, and axial current. A finite-current,
       finite-pressure gauge-free-lambda case converges the adjoint in four iterations at
       `8.85e-16` relative linear residual; its combined directional derivative matches two fully
       reconverged central-difference equilibria to `1.10e-7` relative. The root example writes
       MOUT and reviewed geometry/field/residual/sensitivity plots. A 585-unknown matrix-free case
       reaches `9.23e-10` adjoint residual in 171 iterations and 3.41 s versus 8.61 s for the primal;
       the scheduled full test keeps it above the dense-reference threshold. Core's three-color
       radial block factorization was ported and reaches `2.69e-15` with zero GMRES corrections,
       but costs 4.79 s for this single RHS; unlike a many-column forward Jacobian it cannot
       amortize assembly, so GMRES remains the scalar-adjoint default. Registered closure pytrees
       now use the same path: closure coefficients plus boundary/flux/
       current match reconverged FD to `5.22e-9` through the anisotropic functional. Dedicated
       bi-Maxwellian mass/hot-fraction gradients pass at `2.99e-9` with positive ellipticity.
       Tabulated pressure-value gradients pass at `5.34e-8`; interpolation knots are correctly
       static pytree metadata. The public `solve_fixed_boundary_implicit` custom VJP now runs the
       host nonlinear solve through `pure_callback` and applies the matrix-free adjoint in reverse;
       isotropic and registered anisotropic gradients match the explicit adjoint at `2e-9` rtol.
       The supported axisymmetric exterior free boundary now has an implicit coil adjoint over the
       physical LCFS/plasma fixed point. An accepted `ns=5,nxi=7` case reaches `1.63e-15`
       primal residual and `1.07e-8` compatibility; its shared-preconditioner transpose solve
       reaches `3.68e-10` in 19 iterations, and combined circular-coil radius/current controls
       match two reconverged equilibria to `4.28e-10` relative. End cuts and the physical pressure
       profile are held fixed. M9 is complete for supported mirror lanes; 3D free-boundary
       derivatives are deferred with the failed M7 refinement gate.
   11. **M10 — performance, outputs, and promotion.** Benchmark CPU/GPU cold/warm time, memory,
       scaling, and CLI versus JAX lanes; add mirror-native `mout` output, restart, `--plot`, docs,
       and short root examples. Remove obsolete archived implementations only after parity data are
       recorded. Mark the feature supported only when every gate below passes.
       **STATUS (2026-07-11): output/plot integration landed.** A compact NetCDF `mout/1` schema
       stores the physical mirror grid, geometry, stream function, Cartesian field, both pressure
       moments, interface metrics, convergence history, closure metadata, and optional coils.
       `vmec --plot mout_*.nc` renders the horizontal 3D LCFS/coils/cap-to-cap field lines, `|B|`,
       cross-sections, pressure, and `ftol` history. The 0--50% straight-mirror example writes one
       file per accepted equilibrium and renders its endpoint through this disk-backed path.
       Before the latest main integration the complete package was 59 Python files / 31,044 lines.
       The post-refactor recount is 64 files / 33,474 lines, while the mirror backend is 20 files /
       8,072 lines and its largest module is 862 lines. Tests total 14,054 Python lines.
       Generated outputs remain ignored and the tracked tree is 7.3 MiB. The mirror structure meets
       its bound and aggregate source is inside the revised evidence-based budget; tests are 4,054
       lines above target. Keep evidence-preserving test simplification as a release cleanup lane;
       do not collapse distinct physics operators merely to satisfy a line count.
       The office-CPU CLI/custom-VJP fixed-forward comparison is
       `16.28/18.65 s` cold and `3.75/4.04 s` warm, with 1.06x RSS; the CLI remains the fastest
       forward path. The launch-bound `(15,5,15)` fixed case remains faster on CPU than A4000
       (`35.21/44.20 s`). Fixed scaling through 3,805 unknowns and the rejected M7 BIE scaling are
       consolidated in `benchmarks/mirror_performance.json`.
       Coverage instrumentation is closed. The failure was coverage `--source` pre-importing the
       package and colliding with NumPy extension loading on macOS, not nested JAX/BIE tracing.
       Running coverage without `--source`, then restricting the report with `--include`, passes
       the focused release gate: 3,269 statements, 157 missed, 95% with `--fail-under=95`.
       This combines the non-nightly shard, full fixed/free adjoints, and compact input/diagnostic
       contract tests. The ordinary core CI combine still omits mirror because those artifacts
       intentionally exclude nightly adjoints. Exact commands and size/test evidence live in
       `benchmarks/mirror_m10_audit.json`.

   **5.5 Convergence, validation, and presentation gates**

   - A requested `ftol=1e-12` means every active nondimensional physical force component is
     `<=1e-12`; optimizer status alone is never convergence. Report raw/component-normalized force,
     energy change, Jacobian minimum, step norm, linear residual, `B.n`, normal-stress jump, and
     closure consistency versus iteration. Stalling is a failed solve with a typed reason.
   - Reference cases pass `ns`, `mpol`, `nxi` (and `ntor` for 3D) studies. Observables converge at
     the expected order or spectrally until roundoff; a tighter nonlinear tolerance may not conceal
     discretization error. Research runs use `max_iter >= 1000` when needed, but CI uses small
     deterministic cases with the same equations.
   - Axisymmetric finite-beta free boundary must match independent reference curves for boundary,
     on-axis `B`, flux, and pressure to documented combined tolerances; M6 also requires monotonic
     continuation from beta zero, positive ellipticity indicators, and independence from reasonable
     initial boundaries. M7 requires the corresponding 3D manufactured/reference evidence.
   - Every beta is an actual equilibrium solve. Saved output drives field-line tracing and plots;
     no prescribed stand-in boundary is plotted as a result. Required figures are horizontal in
     mirror `z`, show coils, solved surfaces, field vectors and cap-to-cap field lines, `|B|`, cross
     sections, pressure moments, on-axis analytic/reference comparisons, residual histories,
     spectra, beta trends, current/twist, and mirror/well diagnostics. Render tests inspect artists
     and nonblank pixels; documentation commits only compressed showcase images.
   - Mirror outputs are `mout_*.nc`, never fake toroidal `wout`. They store geometry, fields,
     `p_parallel`, `p_perp`, closure metadata, force/interface histories, stability indicators,
     end-cut data, coils, and provenance. Toroidal hybrids continue to use `wout` and Boozer.
   - Supported means API/CLI documentation, restart compatibility, typed failures, >=95% focused
     coverage, CPU and office-GPU benchmarks, examples, and no known case labelled converged with a
     nonzero solver flag. Free-boundary axisymmetric and 3D mirrors are separate capability flags.

---

## 9. Phase 6 — Outputs: prints, wout completeness, JSON, Boozer

1. **Prints**: replicate VMEC2000 layout byte-for-column. The authoritative formats (from
   `printout.f`, `initialize_radial.f`, `runvmec.f`) are in Appendix B — implement `printing.py`
   directly from them (screen lane: `iter, fsqr, fsqz, fsql, RAX(v=0)[, ZAX], DELT, WMHD[, DEL-BSQ]`;
   threed1 lane adds preconditioned `fsqr1,fsqz1,fsql1`, `BETA`, `<M>`, `FEDGE`). Golden stdout
   captures of VMEC2000 per benchmark case go in the release asset bundle; tests diff structure
   exactly and values within tolerance. Also: `BEGIN FORCE ITERATIONS` banner, per-stage
   `NS = … NO. FOURIER MODES = … FTOLV = … NITER = …` banners, `VACUUM PRESSURE TURNED ON AT n
   ITERATIONS`, final timing + `EXECUTION TERMINATED NORMALLY`-style `werror` messages, and the
   threed1 summary file.
2. **wout completeness**: implement the full Appendix-A variable set (unit conventions included:
   `presf/pres/mass/jcuru/jcurv/ctor` divided by μ0 on write; `phipf/chipf` × 2π·signgs; `qfact =
   1/iotaf`; `lmns` half-mesh; `bsubsmns` full-mesh). Add VMEC++'s useful extras where free
   (`fsqt`, `wdot` already exist in VMEC2000; consider `lmns_full` as an extension attribute).
   Parity test: per-variable `CompareWOut`-style relative+absolute tolerances (global default +
   looser `currumnc/currvmnc`), following vmecpp-validation methodology. wout must load in simsopt
   and booz_xform unchanged.
3. **JSON input**: accept VMEC++'s exact schema alongside INDATA (keys in Appendix C; sparse
   `{"n":…,"m":…,"value":…}` boundary coefficients; dense axis arrays). `vmec convert input.foo
   --to json|indata` round-trips. We additionally support the profile types VMEC++ lacks
   (splines, pedestal, two_power, gauss_trunc…) — same key names as INDATA.
4. **Boozer**: `booz_xform_jax` stays a default dependency; `vmec --booz [--plot]` writes
   `boozmn_*.nc` + |B| contours + spectrum plots; one integration test.

---

## 10. Phase 7 — Examples (the public face; simsopt-simple)

Rules: each example is **one file**, < ~120 lines, top docstring stating goal/physics/expected
runtime/achieved result, public API only, smoke-tested in CI at reduced resolution (`--ci` flag).

```
examples/
  data/                          # text input decks + one tiny mgrid
  fixed_boundary_run.py          # run + plot + wout + --booz equivalent, ~30 lines
  free_boundary_mgrid.py         # cth_like via mgrid
  free_boundary_direct_coils.py  # same physics via ESSOS CoilField; compares to the mgrid run
  free_boundary_beta_scan.py     # THE README example (below)
  QA_optimization.py             # circular torus -> precise QA   (nfp=2)
  QH_optimization.py             # circular torus -> precise QH   (nfp=4)
  QP_optimization.py             # circular torus -> precise QP   (nfp=2 or 3)
  QI_optimization.py             # circular torus -> precise QI   (nfp=1)
  single_stage_fixed_boundary.py # advanced, ESSOS
  single_stage_free_boundary.py  # advanced, ESSOS
```

**Example style requirements (user directive 2026-07-09; binding):** pedagogic and user-friendly —
no `main()` functions; input parameters at the top; the user writes their own objective function
in the script (importing gradient-ready building blocks from `vmec_jax.core.optimize`); scripts
teach: reading input files AND creating a `VmecInput` from scratch, writing outputs (wout),
plotting, and printing initial conditions, per-iteration progress, and final results. Not so
minimal that users cannot generalize; no auxiliary-function mazes (anything reusable moves into
the source). Optimization examples target **precise QA/QH/QP/QI at several nfp, max_mode=5, with
ESS, from a circular torus**; for QI, try **QP-first-then-QI** as the simple route before any
seed/preconditioner machinery (compare against the legacy seed approach and report). Runs must be
fast on CPU and GPU: reuse the warm solver (structural executable cache), use implicit autodiff
with measured accuracy (test gradient accuracy in CI), and include **commented-out but fully
tested** extra objective terms (DMerc, LgradB, magnetic well, ...) that work when uncommented
(tests exercise them uncommented).

**Optimization examples** mirror simsopt's `QH_fixed_resolution.py` (66 lines: build equilibrium →
`QuasisymmetryRatioResidual(surfaces, helicity_m, helicity_n)` + aspect target → one least-squares
call). Ours: `vmec_jax.optimize.QuasisymmetryResidual(m, n)` with (QA: m=1,n=0; QH: m=1,n=−nfp;
QP: m=0,n=1) + aspect (+ iota/mirror targets), staged `max_mode` 1→2→3 continuation inside one
visible loop, gradient-based least squares using Phase-6 implicit gradients (highlight: no finite
differences, no MPI). **QI** uses a Goodman-style penalty (B-contour alignment, mirror-ratio /
target-B shaping, elongation + iota + aspect practical targets) implemented and documented inside
`optimize.py` — the example stays one file; continuation detail (grow ntor ~2× faster than mpol) is
a documented option, not example-level machinery. Today's QP-basin-then-QI trick, if still needed,
lives inside `optimize.py` with a docstring, honestly stated. Achieved objective values go in each
docstring and are loosely asserted by the CI smoke test.

**`free_boundary_beta_scan.py`** (featured in README): Landreman–Paul precise-QA coils from ESSOS
+ a simple ESSOS tokamak coil set; for β = 0,1,2,3,4,5%: free-boundary solve (hot-restarting each β
from the previous), once via generated mgrid and once via direct Biot–Savart; plot boundary
cross-sections evolving with β for both machines, overlay mgrid vs direct, report the difference
(direct = interpolation-free reference). Output: one compressed panel figure for the README.

**Measured direct-coil status (2026-07-12):** the LP-QA example now calibrates pressure with a
local secant, scales beta acceptance to the continuation step, and bisects failed steps down to
0.0125%. On the office A4000 at `ns=51,mpol=5,ntor=5`, component-wise `ftol=1e-10`, it advances
the old 3.120% endpoint to target/actual beta 3.350% in 149 minutes. The final accepted solve takes
18,161 iterations with residual sum `1.3e-10`; target 3.3625% exhausts 20,000 iterations at
`1.491e-10`. Axis shift is +16.20 cm and every plotted surface is a solved LCFS. The example writes
WOUT for requested equilibria and the measured endpoint. Targets 4% and 5%, the mgrid/direct parity
overlay, and the second tokamak machine remain open; do not substitute prescribed high-beta
surfaces. Compact accepted and rejected histories are in `benchmarks/free_boundary_essos_beta.json`.
The LP-QA generated-mgrid parity attempt is now closed as a negative result. Scaled-group and raw
single-group fields agree with direct Biot--Savart at grid nodes to `1.4e-15`, and refining from
`96x96x48` to `160x160x160` lowers off-grid field RMS error from `7.09e-3` to `6.40e-4`. Nevertheless,
an `ns=16` mgrid solve leaves an already converged direct-coil state on its first NESTOR releases
and becomes non-finite within 6,000 iterations. This is the expected uniform-trilinear limitation
for tightly fitting modular coils, not a file-layout defect. Evidence is in
`benchmarks/free_boundary_essos_mgrid_parity.json`. Do not spend a multi-GB grid on this case;
positive equilibrium-level mgrid/direct parity moves to the well-separated tokamak coil example.
That positive lane is now complete. `tokamak_coils` builds circular TF coils and arbitrary
`(radius,z,current)` PF loops without example-level Fourier assembly. The root example reconstructs
the bundled DIII-D field with 128 TF and 23 PF coils, generates a `145x225x1` raw single-group
mgrid, and solves both providers at actual beta 0, 1.496%, and 3.009%. All six equilibria converge
at `ftol=1e-8`; direct/generated LCFS coefficient differences are `2.25e-4--6.31e-4`. From beta
zero to 3.009%, direct-coil volume rises 43.4%, the axis moves outward 34.0 cm, and mean iota rises
0.494 -> 0.621. The example writes both WOUT sets, CSV parity data, a reviewed LCFS overlay, and
the standard 3D field-line, `|B|`, surface, profile, and Mercier figures. Evidence is in
`benchmarks/free_boundary_tokamak_coil_parity.json`.

---

## 11. Phase 8 — Tests and coverage (≥95%, no bloat)

- `tests/` mirrors the module layout: one file per module + `test_parity/` (wout/print goldens
  from the asset bundle) + `test_gradients/` (FD checks) + `test_examples.py` (smoke).
- Markers: `quick` (offline, <2 min, every push), `parity` (asset bundle), `slow` (nightly).
  Coverage gate ≥95% on `vmec_jax/` from quick+parity (`--cov-fail-under=95`).
- Property tests: transform round-trip (tomnsp∘totzsp = identity on band-limited data), residual
  invariance under nfp rotation, lasym-off ≡ symmetric path, CLI-lane ≡ jit-lane per block,
  JSON↔INDATA round-trip.
- Delete the wave/coverage-padding files and the 3–4k-line lane tests. Budget: ≤ ~10k lines total,
  including the mirror scientific-validation suite.
- Keep the `VMEC2000_INTEGRATION=1` opt-in gate that runs xvmec2000 side-by-side locally/nightly.

---

## 12. Phase 9 — Documentation overhaul

Sphinx with `furo` (already) or `sphinx-book-theme`, MathJax, `sphinx-copybutton`, `myst-parser`.
Landing page: what/why, 3-command quickstart, gallery. Delete the internal plan/lane pages
(`aggressive_performance_plan`, `accelerated_merge_readiness`, `optimization_sweep_results`,
`piecewise_omnigenous_plan`, `free_boundary_plan`, `discrete_adjoint` in its current form).

Structure:
1. **Getting started** — install, `vmec --test`, first run, plotting, Boozer.
2. **Tutorials** — one page per example, rendered figures, expected output.
3. **Theory & numerics** (the differentiator; every equation links to the implementing function):
   ideal-MHD energy functional and the Hirshman–Whitson steepest-descent moment method; flux
   coordinates and λ; Fourier representations, parities, lasym; force residuals and the m=1
   constraint; spectral condensation (`alias`/tcon); half/full radial meshes and jmin/jlam
   conventions; the 1D preconditioner derivation + tridiagonal solve + the 2D matrix-free
   extension; Richardson time stepping, damping, and irst back-off; multigrid + hot restart;
   NESTOR (Merkel Green's-function method) and the free-boundary cadence (ivac/nvacskip); the
   implicit-differentiation adjoint with derivation and cost analysis (cite Skene & Burns 2026,
   jaxopt, DESC); CLI lane vs differentiable lane.
4. **Reference** — API autodoc; input reference (every INDATA variable + JSON schema); wout
   variable reference (Appendix A rendered); glossary; CLI reference.
5. **Performance & validation** — benchmark methodology, plots, parity tables, GPU notes,
   profiling guide (including how VMEC2000 was instrumented).
6. **Developer guide** — architecture map (module ↔ VMEC2000 subroutine), adding an objective,
   release checklist.

---

## 13. Phase 10 — Benchmarks, README, release

1. Re-run `benchmarks/run_baseline.py`; regenerate the README benchmark figure (compressed):
   vmec_jax CPU/GPU cold+warm vs VMEC2000 vs VMEC++, single-grid and multigrid.
2. Rewrite README: short pitch; install; quickstart; β-scan figure; one optimization figure;
   feature table vs VMEC2000/VMEC++ (differentiable ✓, lasym ✓, free-boundary tokamak+stellarator
   ✓, JSON ✓, hot restart ✓, zero-crash ✓, mgrid fallback ✓, GPU ✓, Boozer built-in ✓, spline
   profiles ✓, 2D preconditioner ✓); CLI reference; docs link. Remove all "research lane" language.
3. Upload the release asset bundle; tag `v0.1.0`; publish to PyPI; update the conda-forge
   feedstock; verify `pip install vmec-jax && vmec --test` in a clean venv.

---

## 14. Acceptance checklist (definition of done)

- [ ] Fresh clone ≤ 10 MB; single branch; zero `Co-Authored-By: Claude` trailers in history; Claude
      absent from the GitHub contributors panel; all new commits authored by rogeriojorge.
- [ ] `vmec_jax/` remains within the §0.5 budget of 50–60 files / ~30–32k lines after the
      mirror backend lands; no mirror file exceeds ~900 lines; docstrings and source/equation
      cross-references are complete; ruff and mypy pass without blanket ignores.
- [ ] Fixed + free boundary (mgrid and direct-coil; tokamak and stellarator; sym and lasym)
      converge with wout + print parity vs VMEC2000 per Appendix-A tolerances; missing-mgrid
      fixed-boundary fallback works and is tested.
- [ ] Fixed-boundary axisymmetric mirror meets the component-wise `1e-12` force contract and its
      analytic field, fixed-flux end-cut, anisotropic-closure, and resolution tests;
      nonaxisymmetric mirror is supported only after its physical-residual and resolution gates.
- [ ] Straight-axis finite-beta free-boundary mirrors are supported in axisymmetry: solved lateral
      interfaces satisfy total `B·n` and anisotropic normal-stress balance, every beta scan point
      through 50% is a converged equilibrium, ellipticity gates pass, and results agree with
      independent Pleiades/WHAM-style reference data. Nonaxisymmetric free boundary remains an
      explicit research API, not a supported capability, until local Fourier modes converge.
- [ ] Toroidal stellarator–mirror hybrid has VMEC2000 parity at its documented fixed-boundary
      tolerance and a reproducible free-boundary continuation through the measured 0.8333% beta
      limit. Every published point uses a solved surface and total `B·n`; higher beta and a native
      spline equilibrium state remain explicitly deferred rather than release blockers.
- [ ] CLI ≥ VMEC2000 speed on ≥80% of suite rows (cold CPU); multigrid faster than VMEC2000
      multigrid on the suite median and faster than our own single-grid; GPU benchmarked;
      hot restart works and is used by examples.
- [ ] Implicit-diff gradients validated vs central FD (boundary, profiles, coil dofs, extcur);
      backward memory ≤2× forward; no fingerprint/replay machinery remains.
- [ ] QA/QH/QP/QI examples: single-file, <~120 lines, from circular torus to precise
      configurations with achieved values in docstrings; β-scan free-boundary example with ESSOS
      coils (mgrid + direct, agreeing) featured in README.
- [ ] VMEC++-schema JSON inputs accepted and round-trip converted; `--booz` works out of the box;
      typed zero-crash exceptions throughout.
- [ ] Coverage ≥95% with tests ≤ ~10k lines; goldens in release assets; CI green including example
      smoke tests and a repo-size check.
- [ ] Docs rebuilt per §12 with equations linked to source; README benchmark plot regenerated;
      v0.1.0 on PyPI + conda-forge.

---

## 15. Risks and mitigations

- **History rewrite is destructive** → fresh clone + `git clone --mirror` backup kept until
  v0.1.0; verify trailer count and pack size before force-pushing.
- **Multigrid slowdown may not be recompilation** → Phase-0 profiles decide; alternates to check:
  preconditioner recompute cadence, interpolation quality on restart, `delt` reset policy across
  stages (compare `irst`/`delt` handling line-by-line with `runvmec.f`/`restart.f` and VMEC++
  `_iteration.py`).
- **Deleting 100k+ lines can drop silent capabilities** → NOTES.md records every deleted module's
  purpose; parity + gradient + example tests are the safety net; the mirror-geometry branch and
  pre-rewrite mirror keep everything recoverable.
- **Adjoint linear solve may converge slowly near marginal equilibria** → 1D preconditioner as
  GMRES preconditioner; fall back to more inner iterations; document conditioning diagnostics.
- **QI from a circular torus is genuinely hard** → staged continuation + Goodman-style residual
  live in `optimize.py`, documented; if NFP-specific seeding is truly unavoidable, the example
  docstring says so honestly.
- **ESSOS PR #33 churn** → pin the commit hash in example docstrings; push fixes to the PR as
  rogeriojorge if needed.
- **float32 GPUs** → require x64 at solver import; document the performance implication.
- **Parity tolerance fights** → per-quantity rel+abs tolerances (CompareWOut methodology) with a
  looser current-density tolerance; never invent ad-hoc tolerances per test.
- **Open-end ambiguity** → support one explicit `fixed_flux_cut` model first: fixed geometry,
  prescribed normal flux, and no end-plane variations. State clearly that this is equilibrium in a
  truncated open tube, not a sheath, end-loss, source, or transport model.
- **Anisotropic closure inconsistency** → accept only closures generated by `p_parallel(s,B)` (or
  a thermodynamically consistent energy) and derive/check `p_perp`; reject independent tables and
  fail on firehose/mirror ellipticity violations.
- **Open-vacuum truncation error** → solve on expanding outer domains and require convergence;
  never reuse toroidal NESTOR kernels without open-surface MMS and flux tests.
- **High-beta bifurcation or solver stall** → beta continuation from vacuum, hot restart,
  separable preconditioning, and explicit ellipticity/conditioning diagnostics; do not return a
  best iterate as a converged equilibrium.

---

## 16. Key references (cite in docs)

- Hirshman & Whitson, Phys. Fluids 26, 3553 (1983) — steepest-descent moment method.
- Hirshman, van Rij & Merkel, Comput. Phys. Commun. 43, 143 (1986) — NESTOR.
- Merkel, J. Comput. Phys. 66, 83 (1986) — vacuum Green's-function method.
- Schilling et al., *The Numerics of VMEC++*, arXiv:2502.04374 — hot restart, JSON, zero-crash,
  validation methodology.
- Skene & Burns, *Fast automated adjoints for spectral PDE solvers*, arXiv:2506.14792 — adjoints
  reusing forward spectral machinery; template for `implicit.py`.
- Blondel et al., *Efficient and Modular Implicit Differentiation*, NeurIPS 2022 (jaxopt) — the
  IFT/custom_vjp formulation (note: jaxopt itself is deprecated; Optax/Optimistix/Lineax are the
  living successors).
- Dudt & Kolemen (2020); Conlin et al. (2023) — DESC: JAX equilibrium solver + implicit-derivative
  optimization precedent.
- Goodman et al., *Constructing precisely quasi-isodynamic magnetic fields*, JPP (2023),
  arXiv:2211.09829 — QI objective; ntor-faster-than-mpol continuation.
- simsopt `examples/2_Intermediate/QH_fixed_resolution.py` — style target for optimization examples.
- STELLOPT VMEC wiki (princetonuniversity.github.io/STELLOPT/VMEC) — INDATA semantics.
- Cooper et al., *Three-dimensional anisotropic pressure free boundary equilibria*, CPC 180,
  1524–1533 (2009), DOI 10.1016/j.cpc.2009.04.006 — ANIMEC energy, pressure closure, normal-stress
  interface condition, and anisotropic free-boundary reference.
- STELLOPT `_ANIMEC` sources `fbal.f`, `bcovar.f`, `forces.f`, `funct3d.f`, and `jxbforce.f` —
  implementation anchors for pressure moments, effective current, edge force, and diagnostics.
- Endrizzi et al., *Physics basis for the Wisconsin HTS Axisymmetric Mirror (WHAM)*, JPP 89 (2023),
  DOI 10.1017/S0022377823000806 — finite-beta anisotropic mirror validation context.
- Frank et al., *Integrated modelling of equilibrium and transport in axisymmetric magnetic mirror
  fusion devices*, JPP 91 E110 (2025), DOI 10.1017/S002237782510055X — Pleiades anisotropic force
  balance, diamagnetic expansion, paraxial check, and ellipticity criteria.
- Frank et al., *Nonlinear anisotropic equilibrium reconstruction in axisymmetric magnetic
  mirrors*, arXiv:2509.17288 — current WHAM high-beta reconstruction benchmark context.
- Pleiades (`github.com/eepeterson/pleiades`) — independent axisymmetric circular-coil, flux, and
  scalar-pressure regression reference; its Green-function algorithm is validation, not the 3D
  mirror backend.
- Trefethen, *Spectral Methods in MATLAB*; Boyd, *Chebyshev and Fourier Spectral Methods* — CGL
  differentiation, quadrature, filtering, and convergence references.

---

## Appendix A — wout variable checklist (from VMEC2000 `wrout.f`; implement all)

**Scalars:** `version_`, `input_extension`, `mgrid_file`, `pcurr_type`, `pmass_type`, `piota_type`,
`wb`, `wp`, `gamma`, `rmax_surf`, `rmin_surf`, `zmax_surf`, `nfp`, `ns`, `mpol`, `ntor`, `mnmax`,
`mnmax_nyq`, `iter2→niter`, `itfsq`, `lasym`, `lrecon`, `lfreeb`, `lrfp`, `ier_flag`, `aspect`,
`betatotal`, `betapol`, `betator`, `betaxis`, `b0`, `rbtor0`, `rbtor`, `signgs`, `IonLarmor`,
`volavgB`, `ctor` (/μ0), `Aminor_p`, `Rmajor_p`, `volume_p`, `ftolv`, `fsql`, `fsqr`, `fsqz`,
`nextcur`, `extcur(:)`, `mgrid_mode`; if lfreeb: `mnpd`, `nobser`, `nobd`, `nbsets`, `nbfld(:)`,
`curlabel(:)`.
**Mode arrays:** `xm`, `xn`, `xm_nyq`, `xn_nyq`.
**Axis:** `raxis_cc`, `zaxis_cs` (+ `raxis_cs`, `zaxis_cc` if lasym).
**Profile inputs:** `am`, `ac`, `ai`, `am_aux_s/f`, `ac_aux_s/f`, `ai_aux_s/f`.
**Radial 1D:** `iotaf`, `q_factor` (=1/iotaf), `presf` (/μ0), `phi`, `phipf` (2π·signgs·),
`chi`, `chipf` (2π·signgs·), `jcuru` (/μ0), `jcurv` (/μ0), `iotas`, `mass` (/μ0), `pres` (/μ0),
`beta_vol`, `buco`, `bvco`, `vp`, `specw`, `phips`, `over_r`, `jdotb`, `bdotb`, `bdotgradv`,
`DMerc`, `DShear`, `DWell`, `DCurr`, `DGeod`, `equif`.
**History:** `fsqt(:)`, `wdot(:)`.
**Free-boundary potential (lfreeb):** `potvac` sin (+cos if lasym), `xmpot`, `xnpot` — note VMEC++
skips these; we implement them.
**2D (mode×radius):** `rmnc`, `zmns`, `lmns` (half mesh), `gmnc` (half), `bmnc` (half),
`bsubumnc` (half), `bsubvmnc` (half), `bsubsmns` (full), `currumnc`, `currvmnc`, `bsupumnc`,
`bsupvmnc`; if lfreeb the `*_sur` surface arrays; if lasym all `*mns/*mnc` partners
(`rmns, zmnc, lmnc, gmns, bmns, bsubumns, bsubvmns, bsubsmnc, currumns, currvmns, bsupumns,
bsupvmns` + `*_sur`).
**Tolerances:** CompareWOut-style combined rel+abs per variable; global default (start 1e-10 for
geometry Fourier coefficients, 1e-8 for derived profiles) with a documented looser bound for
`currumnc/currvmnc`; calibrate against actual VMEC2000-vs-VMEC2000 run-to-run scatter in Phase 0.

## Appendix B — VMEC2000 print formats (from `printout.f` / `initialize_radial.f` / `runvmec.f`)

```
Headers:
  iter_line  = "  ITER    FSQR      FSQZ      FSQL   "
  fsq_line   = "   fsqr      fsqz      fsql      DELT    "
  raxis_line = "RAX(v=0) "        zaxis_line = "  ZAX(v=0)      "
  threed1 hdr suffixes: "WMHD      BETA      <M>   DEL-BSQ   FEDGE" (freeb)
                        "WMHD      BETA      <M>"                  (fixed)
Data lines (Fortran FORMATs):
  screen fixed sym : (i5,1p,3e10.2, e11.3, e10.2, e12.4)             ! ITER FSQR FSQZ FSQL RAX DELT WMHD
  screen freeb sym : (i5,1p,3e10.2, e11.3, e10.2, e12.4, e11.3)      ! + DEL-BSQ
  screen lasym     : adds ZAX(v=0) column (2e11.3)
  threed1 sym      : (i6,1x,1p,7e10.2, e11.3, e12.4, e11.3, 0p,f7.3, 1p,2e9.2)
Banners:
  ' FSQR, FSQZ = Normalized Physical Force Residuals' /
  ' fsqr, fsqz = Preconditioned Force Residuals' / 23('-') /
  ' BEGIN FORCE ITERATIONS' / 23('-')
  '  NS = ',i4,' NO. FOURIER MODES = ',i4,' FTOLV = ',1p,e10.3,' NITER = ',i6   (per stage)
  '  VACUUM PRESSURE TURNED ON AT ',i4,' ITERATIONS'
```

## Appendix C — VMEC++ JSON input keys (adopt verbatim; extend for our extra features)

`lasym, nfp, mpol, ntor, ntheta, nzeta, ns_array, ftol_array, niter_array, delt, tcon0, aphi,
phiedge, nstep, pmass_type, am, am_aux_s, am_aux_f, pres_scale, adiabatic_index(=gamma), spres_ped,
ncurr, pcurr_type, ac, ac_aux_s, ac_aux_f, curtor, piota_type, ai, ai_aux_s, ai_aux_f, bloat,
raxis_c, zaxis_s, raxis_s, zaxis_c, rbc, zbs, rbs, zbc, lfreeb, mgrid_file, extcur, nvacskip,
free_boundary_method, lforbal` — boundary coefficients as sparse `{"n": int, "m": int,
"value": float}` lists; axis arrays dense length `ntor+1`. Our extensions (documented, ignored by
VMEC++): `precon_type`, `prec2d_threshold`, spline profile types, mirror-geometry keys (Phase 5.5).

## Appendix D — VMEC2000 algorithm constants (parity-critical; from source)

| Item | Value / rule | Source |
|---|---|---|
| Richardson damping window | `ndamp = 10` | vmec_params.f |
| dtau cap | `bprec·0.15` (bprec=6 with 2D precond on) | evolve.f |
| Velocity update | `xcdot ← (1−dtau)/(1+dtau)·xcdot + delt·gc; xc += delt·xcdot` | evolve.f |
| Precond recompute cadence | every `ns4 = 25` iterations | bcovar.f |
| Jacobian reset (`irst=2`) | restore state, zero velocity, `delt ×= 0.90` | restart.f |
| Residual-growth back-off (`irst=3`) | growth >1e4× best after >10 steps; `delt /= 1.03` | evolve.f/restart.f |
| Escalation | guess_axis on 1st bad jac; delt reset at ijacob=25,50; abort at 75 | eqsolve.f |
| Constraint scaling | `tcon(js)=min(|ard/arnorm|,|azd/aznorm|)·tcon0-scaled·(32hs)²; tcon(ns)=½tcon(ns−1)` | bcovar.f |
| Constraint spectrum | m ∈ [1, mpol−2], weights `faccon(m)` | alias.f |
| m=1 constraint | rotate (gcr,gcz)_{m=1} → ((gcr+gcz)/√2, 0); input `rbss=½(rbs+zbc)` when lconm1 | residue.f90, readin.f |
| Edge pedestal / ZC00 stabilization | 0.05 / fac=0.25 | scalfor.f |
| λ precond | `faclam ∝ 1/(blam(n·nfp)²+clam·m²±2mn·nfp)`, √s damping m>16 | lamcal.f90 |
| Vacuum turn-on | `fsqr+fsqz ≤ 1e-3` | funct3d.f |
| Vacuum cadence | full solve when `mod(iter2−iter1,nvacskip)=0`; adaptive `nvacskip=max(nvskip0, 1/max(0.1,1e11(fsqr+fsqz)))` | funct3d.f |
| Free-bdy constraint ramp | `rcon0,zcon0 ×= 0.9` per iteration | funct3d.f |
| Convergence | `fsqr,fsqz,fsql ≤ ftolv` simultaneously | evolve.f |
| 2D precond activation | finest grid, `fsqr+fsqz+fsql < prec2d_threshold`; GMRES/CG/TFQMR lanes via precon_type | evolve.f, precon2d.f, gmres_mod.f |
