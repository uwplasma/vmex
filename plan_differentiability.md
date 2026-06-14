# Research-Grade Differentiable VMEC Plan

Status: deferred research lane, not a blocker for the current free-boundary
phase-2/phase-3 release.

Last updated: 2026-06-14.

Repository: `/Users/rogeriojorge/local/vmec_jax`.

## Executive Summary

The long-term goal is a research-grade differentiable VMEC variant that keeps
VMEC2000-compatible fixed-boundary and free-boundary physics while exposing
validated derivatives for equilibrium, boundary optimization, direct-coil
optimization, finite-beta metrics, Boozer-space objectives, and stability
objectives.

The near-term release should remain conservative:

1. Fixed-boundary derivatives and production optimization APIs are promoted only
   where AD-vs-FD, VMEC2000 parity, and physics gates pass.
2. Free-boundary direct-coil providers, mgrid providers, and branch-local
   same-fingerprint replay/controller derivatives are promoted.
3. Arbitrary differentiation through adaptive host branch changes in
   `run_free_boundary` is not yet promoted.
4. A future differentiable-controller lane can explore fully JAX-visible branch
   selection, but only if it preserves VMEC parity, improves optimization
   robustness, or enables capabilities that the branch-local seam cannot.

## Terms

- Fixed boundary: the plasma boundary is prescribed by Fourier coefficients.
- Free boundary: the boundary is solved with an external vacuum field supplied
  by mgrid interpolation, ESSOS-generated grids, or direct coils.
- Direct coil provider: pure-JAX Biot-Savart field evaluation from Fourier coil
  parameters.
- Adaptive branch: a discrete path selected by the nonlinear VMEC controller,
  including accepted/rejected steps, Jacobian resets, timestep limiters,
  restart/fallback selection, and preconditioner policies.
- Branch fingerprint: compact metadata that identifies the adaptive branch used
  by a solve or replay. Derivatives are only promoted when the plus/minus
  finite-difference perturbations keep the same fingerprint.
- Branch-local derivative: a derivative of one fixed accepted/rejected
  controller path. This can be validated against complete-solve central finite
  differences under an unchanged fingerprint.
- Arbitrary adaptive derivative: a derivative through changes in branch
  selection. This is generally nonsmooth and is not claimed by current code.

## Literature and Code Anchors

The plan is anchored in the following references and implementation patterns:

1. VMEC/STELLOPT documentation: VMEC uses Fourier-expanded geometry and a
   variational energy minimization; free-boundary VMEC uses vacuum fields from
   mgrid files and NESTOR-style exterior solves.
   URL: https://princetonuniversity.github.io/STELLOPT/VMEC.html
2. VMEC++ numerics: modern VMEC reimplementation practices, restart behavior,
   robust execution, and validation philosophy.
   URL: https://arxiv.org/abs/2502.04374
3. DESC code suite: JAX-based stellarator equilibrium and optimization,
   continuation, perturbation, free-boundary residuals, and differentiable
   optimization design.
   URL: https://desc-docs.readthedocs.io/
4. High-order free-boundary DESC work: free-boundary residual formulation and
   NESTOR/VMEC context for vacuum-boundary conditions.
   URL: https://arxiv.org/html/2412.05680v1
5. SIMSOPT: practical optimization API design, caching, coil/boundary
   parameter spaces, and mixed analytic/AD derivatives.
   URL: https://simsopt.readthedocs.io/
6. JAX control-flow docs: `lax.scan`, `lax.while_loop`, and transformed loops.
   Static loops should use `scan`; dynamic loops use `while_loop` when a
   JAX-visible controller is worth the tradeoff.
   URL: https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html
7. JAXopt implicit differentiation: solver-output differentiation by implicit
   rules rather than storing every unrolled iteration.
   URL: https://jaxopt.github.io/stable/implicit_diff.html
8. Fast automated adjoints for spectral PDE solvers: construct efficient
   discrete adjoints for sparse spectral PDE solvers rather than relying on
   naive tape retention.
   URL: https://arxiv.org/abs/2506.14792
9. Mercier and Glasser/Jorge-Landreman stability work: physics anchors for
   DMerc and resistive `D_R` gates.
   URL: https://arxiv.org/abs/2006.14881

## Current Source Map

Differentiable fixed-boundary and optimization code:

- `vmec_jax/driver.py`: production `run_fixed_boundary` and
  `run_free_boundary` entry points, host controller, WOUT generation.
- `vmec_jax/optimization.py`: fixed-boundary least-squares utilities,
  accepted-point replay, scalar-adjoint and exact callback policies.
- `vmec_jax/optimization_workflow.py`: user-facing optimization helpers and
  problem assembly.
- `vmec_jax/quasisymmetry.py`: quasisymmetry residuals and targets.
- `vmec_jax/quasi_isodynamic.py`, `vmec_jax/qi_optimization.py`,
  `vmec_jax/qi_diagnostics.py`: QI/QP objectives, diagnostics, gates, and
  minimal-seed workflows.
- `vmec_jax/wout.py`, `vmec_jax/vmec_output.py`: equilibrium output and metric
  extraction.

Differentiable free-boundary and coil code:

- `vmec_jax/external_fields/coils_jax.py`: Fourier coils, stellarator symmetry
  expansion, Biot-Savart sampling, and coil metrics.
- `vmec_jax/external_fields/mgrid_jax.py`: JAX mgrid interpolation path.
- `vmec_jax/external_fields/essos_adapter.py`: optional ESSOS adapter.
- `vmec_jax/free_boundary.py`: free-boundary provider hook, direct-coil/mgrid
  sampling, VMEC/NESTOR integration.
- `vmec_jax/free_boundary_adjoint.py`: dense vacuum solve, JAX NESTOR pieces,
  accepted-boundary replay, fixed-trace custom-VJP helpers, and branch-local
  reports.
- `vmec_jax/free_boundary_adjoint_controller.py`: JAX-visible nonlinear and
  segmented controller primitives used for same-fingerprint validation.
- `vmec_jax/free_boundary_validation.py`: validation helpers and bounded
  physical fixture checks.

Examples and docs:

- `examples/optimization/QA_optimization.py`, `QH_optimization.py`,
  `QP_optimization.py`, `QI_optimization.py`: fixed-boundary optimization
  scripts that should remain clear, Simsopt-like, and user editable.
- `examples/optimization/free_boundary_QA_finite_beta_coil_optimization.py`:
  conservative coil-only free-boundary optimization example with complete
  solves as acceptance authority and optional same-branch derivative reports.
- `docs/free_boundary_coil_optimization.rst`: current free-boundary claims and
  limitations.
- `plan_freeb.md`: current phase-1/phase-2/phase-3 execution log.

## Architecture Goals

1. Preserve VMEC semantics by default.
   Production `run_fixed_boundary` and `run_free_boundary` must stay compatible
   with VMEC2000 input/output behavior and validated finite-positive physical
   WOUTs.
2. Make differentiability explicit.
   Every promoted derivative path must state whether it differentiates an
   equilibrium residual, fixed accepted branch, same-fingerprint adaptive
   replay, or an arbitrary adaptive branch.
3. Validate derivatives with physics scalars.
   Gates must include meaningful outputs: aspect, iota, boundary displacement,
   Bnormal RMS, QS/QI proxies, DMerc, `D_R`, pressure/current response, and
   WOUT-level geometry scalars.
4. Keep optimization practical.
   The public optimization API should let users assemble objectives as tuples
   or lightweight objective objects, choose weights/targets, run SciPy or JAX
   optimizers, save inputs/WOUTs, and plot results without hiding the workflow.
5. Avoid large runtime regressions.
   Full exact derivatives should not force dense tape construction when a
   scalar-adjoint, matrix-free, projected, or implicit path is available.
6. Keep claims conservative.
   Do not claim full adaptive-loop differentiability until fingerprint-gated
   full adaptive AD-vs-central-FD gates pass on physical scalar outputs.

## 2026-06-14 Differentiability and Refactor Review

This review cross-checked the current codebase against current JAX, Python,
JAXopt, Optax, Equinox, Orthax, VMEC++, DESC, SIMSOPT, Scientific Python, and
spectral-adjoint guidance.

Key conclusions:

1. Treat the VMEC solve as a differentiable numerical program with explicit
   state objects, residual operators, solver policies, and output adapters.
   Do not keep growing monolithic VMEC2000-style translation files.
2. Use native JAX PyTrees and frozen dataclasses first.  Equinox-style modules
   are attractive for filtered transforms, but adding Equinox as a mandatory
   dependency is not justified until a concrete filtered-transform seam needs
   it.  The source layout should remain plain-Python and JAX-native.
3. Use `lax.scan` or static-trip `fori_loop` for fixed-budget iteration traces
   that need reverse-mode AD.  Use `while_loop` only for JAX-visible dynamic
   controllers whose derivative semantics are intentionally limited or supplied
   by a custom rule.
4. Use `jax.custom_vjp`, `jax.custom_jvp`, and JAXopt-style implicit
   differentiation for equilibrium roots and branch-local fixed points instead
   of relying on unbounded tape retention through every nonlinear iteration.
5. Use Optax as an optional optimizer backend only through a small adapter.
   Keep SciPy least-squares as the beginner-friendly and VMEC/SIMSOPT-familiar
   path.
6. Orthax can be useful for future orthogonal-polynomial profile bases, but it
   should not replace current VMEC-compatible polynomial and spline profile
   handling unless a validation fixture needs that basis.
7. Keep optional dependencies imported lazily at call sites.  `import vmec_jax`
   and `vmec --test` must remain beginner-friendly and fast.
8. Adopt a Scientific-Python-style separation between public API, internal
   kernels, diagnostics, optional external gates, and generated artifacts.

Current line-count hotspots from the main branch:

.. code-block:: text

   vmec_jax/solve.py                                      15438
   vmec_jax/free_boundary_adjoint.py                       6941
   vmec_jax/wout.py                                        6321
   vmec_jax/optimization.py                                5441
   tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py  5150
   vmec_jax/free_boundary.py                               4271
   vmec_jax/optimization_workflow.py                       4252
   vmec_jax/driver.py                                      4064
   vmec_jax/discrete_adjoint.py                            3557

These files are too large for sustained research development.  The refactor
must reduce local cognitive load while preserving VMEC parity, public API
compatibility, and validated derivative behavior.

## Target Package Architecture

The refactored package should expose stable public modules and keep large
implementation modules split by responsibility.  The target layout is:

.. code-block:: text

   vmec_jax/
     api.py                     stable public import surface
     config/                    parsed INDATA, profiles, grids, run options
     state/                     PyTree equilibrium, force, and output states
     kernels/
       geometry.py              real-space geometry and metric kernels
       fourier.py               Fourier transforms, mode maps, Nyquist tables
       fields.py                covariant/contravariant B, J, pressure terms
       forces.py                fixed-boundary and finite-beta force blocks
       residuals.py             residual assembly and norms
     solvers/
       fixed_boundary.py        fixed-boundary solve orchestration
       free_boundary.py         free-boundary solve orchestration
       controller.py            accepted/rejected/update policy data classes
       scan.py                  JAX-visible fixed-budget traces
       implicit.py              root/JVP/VJP/linear-solve derivative seams
       outputs.py               accepted-state, rerun, WOUT, and checkpoints
     free_boundary/
       providers.py             mgrid/direct-coil/ESSOS provider protocol
       nestor.py                source/NESTOR operators
       adjoint.py               branch-local reports and custom VJP seams
       fingerprints.py          adaptive branch metadata and promotion checks
     objectives/
       quasisymmetry.py
       quasi_isodynamic.py
       finite_beta.py
       stability.py             DMerc, Glasser D_R, magnetic well, jdotB
       coils.py
       least_squares.py         objective tuple/object assembly
     optimization/
       boundary.py              boundary DOF spaces and continuation policies
       coils.py                 coil DOF spaces and proposal/acceptance loops
       callbacks.py             exact/scalar/matrix-free callback policies
       result.py                history, provenance, saved artifacts
       scipy_backend.py
       jaxopt_backend.py        optional
       optax_backend.py         optional
     io/
       namelist.py
       wout.py
       booz.py
       assets.py
     plotting/
       geometry.py
       boozer.py
       optimization.py
       stability.py
     diagnostics/
       parity.py
       performance.py
       source_health.py

The existing module names should remain available through compatibility
re-exports until the next major release.  Tests should import from the new
module paths when validating new functionality and from old paths when
checking backward compatibility.

## Refactor Migration Waves

Wave 0: Baseline and source-health guard.

1. Add a diagnostic that reports Python source line counts and flags files over
   agreed warning/error thresholds.
2. Add CI/documentation guidance that new large helpers must be split before
   merging unless an explicit exemption is recorded.
3. Freeze the current public API surface with import and smoke tests.

Wave 1: Extract pure data/state/config modules.

1. Move parsed profile/config objects into small frozen dataclasses and PyTrees.
2. Split large solver carries into named `EquilibriumState`, `ForceState`,
   `ControllerState`, `SolveTrace`, and `OutputState` containers.
3. Add docstrings describing units, mesh location, shape conventions, and
   differentiability status for every public state object.

Wave 2: Extract pure kernels from `solve.py`.

1. Separate residual norm calculation, timestep updates, scan fallback
   planning, restart/fallback policy, and axis/Jacobian repair into focused
   modules.
2. Keep each extracted helper testable with synthetic small arrays.
3. Require one parity or numerical gate for every extracted physics kernel.

Wave 3: Solver-controller seam.

1. Introduce explicit controller policy objects for fixed boundary and free
   boundary.
2. Represent accepted/rejected steps, resets, limiter choices, and fallback
   choices in a branch fingerprint object.
3. Keep hard branch changes nondifferentiable by default; expose
   same-fingerprint derivative reports and changed-fingerprint rejection.

Wave 4: Derivative backends.

1. Make exact, scalar-adjoint, projected, matrix-free, and implicit
   derivatives pluggable through one `DerivativePolicy` interface.
2. Promote each backend only with AD-vs-central-FD gates on physical scalars.
3. Use JAX rematerialization only on localized kernels where profiling shows
   tape memory pressure; do not blanket-remat the solve.

Wave 5: Optimization API cleanup.

1. Replace high-argument helper calls with Simsopt-like objective tuples and
   lightweight objective objects.
2. Keep example scripts explicit: parameters at top, VMEC object, objectives,
   optimizer call, result inspection, saving, and plotting.
3. Add optional JAXopt/Optax backends behind adapters without making either
   mandatory for beginner installs.

Wave 6: Free-boundary production adjoint.

1. Keep complete solves as acceptance authority for coil optimization.
2. Use branch-local vector/JVP paths only when fingerprints match.
3. Add a fully JAX-visible adaptive-controller prototype only as a research
   experiment after branch-local gates are exhausted.

Wave 7: Documentation and examples.

1. Turn every public objective and derivative policy into an example-backed
   docs page.
2. Keep README short; move derivations, sweep tables, limitations, and
   validation provenance into docs.
3. Add "developer map" pages for new contributors that explain where to add
   a kernel, objective, solver policy, external-field provider, or test.

## Refactor Test and Validation Contract

Every migrated module must satisfy one or more of these gates:

1. Import/backward-compatibility gate: old public imports still work.
2. Shape/unit gate: synthetic arrays validate mesh location, sign conventions,
   and mode ordering.
3. Numerical identity gate: algebraic identities such as divergence-free field
   relations, covariant/contravariant consistency, Fourier reconstruction, and
   radial interpolation hold to tight tolerances.
4. AD-vs-central-FD gate: smooth scalar outputs agree for boundary modes,
   pressure/current profile coefficients, spline knots, coil currents, coil
   Fourier coefficients, DMerc, `D_R`, QS/QI residuals, Bnormal RMS, aspect,
   iota, and finite-beta response.
5. External parity gate: compact fixtures agree with VMEC2000, VMEC++, SIMSOPT,
   booz_xform_jax, or ESSOS where applicable and available.
6. Physics gate: outputs satisfy finite-positive geometry, monotone/expected
   profile behavior, magnetic-axis regularity, force residual convergence,
   Boozer-space symmetry expectations, Mercier/Glasser sign conventions, and
   coil-engineering constraints.
7. Artifact gate: saved `input.final`, `wout_final.nc`, Boozer files, and
   history JSON reproduce rerun results when convergence is claimed.
8. Performance gate: compact cold/warm solve and derivative timing budgets are
   recorded, with optional GPU and VMEC2000/ESSOS lanes separated from required
   CI.

The required CI suite should stay below the current runtime budget by using
small deterministic fixtures, sharded coverage, fetched assets for large WOUTs,
and optional markers for VMEC2000, SIMSOPT, ESSOS, GPU, and full-resolution
physics.  Coverage increases must come from real physics/numerics/API tests,
not scaffold-only tests.

## Documentation and Pedagogy Contract

New public code should satisfy:

1. A short docstring with inputs, outputs, mesh conventions, differentiability
   status, and failure modes.
2. One docs paragraph or example snippet for user-facing APIs.
3. One test that demonstrates the simplest expected user workflow.
4. Comments only where they explain a non-obvious numerical or physics choice.
5. No hidden environment-variable behavior in beginner examples; advanced
   toggles belong in explicit variables or CLI flags.

## Main Work Lanes

### Lane A: Fixed-Boundary Differentiable Core

Goal: production-grade fixed-boundary derivatives for boundary optimization.

Steps:

1. Audit all state carried by `run_fixed_boundary` and accepted-point replay.
2. Keep exact, scalar-adjoint, and auto-scalar paths under a single public
   policy with clear fallback rules.
3. Add AD-vs-central-FD gates for boundary modes, pressure/current profile
   coefficients, spline profile knots, finite-beta response, DMerc, `D_R`,
   volume-average field metrics, iota, and QS/QI objective terms.
4. Add regression gates for max-mode continuation and direct-start policies.
5. Keep WOUT, input.final, and stage checkpoint output identical between the
   optimization result and rerunning the saved input when the solve converges.

Success metrics:

- AD-vs-FD relative error below `1e-4` for smooth scalar objectives on compact
  fixtures, or tighter when conditioning allows.
- VMEC2000 parity on converged equilibria for fixed-boundary benchmarks.
- Cold and warm runtime budgets documented and enforced in CI smoke tests.

### Lane B: Free-Boundary Providers

Goal: direct-coil and mgrid free-boundary solves that are VMEC-compatible and
JAX-differentiable where claimed.

Steps:

1. Maintain pure-JAX direct-coil Biot-Savart provider and JAX mgrid
   interpolation provider.
2. Keep ESSOS adapter optional and skip cleanly when unavailable.
3. Expand bounded VMEC2000/mgrid/direct-coil parity only with fixtures that
   stay inside generated grid domains and produce finite positive geometry.
4. Add finite-beta free-boundary fixtures with actual `LASYM=T` when claiming
   non-stellarator-symmetric finite-beta coverage.
5. Add Bnormal RMS and boundary-displacement physics gates for provider parity.

Success metrics:

- Direct-coil and generated-mgrid WOUT geometry agree within documented
  tolerances for bounded fixtures.
- External-field provider gradients pass AD-vs-FD for currents and Fourier
  coil coefficients.
- Nonphysical forced-active diagnostics remain excluded from promotion gates.

### Lane C: Full Nonlinear Free-Boundary Adjoint

Goal: exact or validated derivatives through the free-boundary equilibrium
solve without overclaiming nonsmooth adaptive branch changes.

Current promoted state:

- Accepted-boundary replay.
- Fixed-trace custom VJP.
- Same-branch scalar/vector/JVP reports.
- Accepted/rejected controller-slot evidence under unchanged fingerprints.
- Complete-solve central-FD validation for physical scalars under fixed
  fingerprints.

Not yet promoted:

- Arbitrary derivatives through host adaptive branch changes.

Steps:

1. Keep branch-local/fingerprint-gated gates as the production-safe seam.
2. Add one narrow full adaptive branch AD-vs-central-FD test only when the
   branch fingerprint is unchanged and the test reports this explicitly.
3. Extend physical scalar coverage to aspect, mean iota, QS proxy, boundary
   moment, Bnormal RMS, and finite-beta response.
4. Add a negative gate: if plus/minus perturbations change the fingerprint, the
   derivative report must decline promotion.
5. Prototype a fully JAX-visible controller only as a deferred research path.
   Compare it against the host controller for branch decisions, residual traces,
   convergence, and VMEC2000 parity before any promotion.

Success metrics:

- Same-fingerprint AD-vs-central-FD relative error below `1e-4` for compact
  physical scalar gates.
- Branch-changing perturbations are detected and rejected, not silently
  differentiated.
- Full-loop claims in docs exactly match executable tests.

### Lane D: Differentiable Optimization APIs

Goal: clear, Simsopt-like scripts and reusable APIs for boundary and coil
optimization.

Steps:

1. Keep example scripts short and explicit: input parameters at top, objective
   tuple assembly, optimizer call, result extraction, saving, and plotting.
2. Move reusable objective terms, plotting helpers, seed builders, and staged
   QI policies into source modules.
3. Ensure objective terms are composable for QA, QH, QP, QI, finite beta,
   DMerc, `D_R`, mirror ratio, elongation, magnetic well, and current-profile
   terms.
4. For coil-only free-boundary optimization, keep complete solves as acceptance
   authority while using branch-local derivative reports only for proposals.
5. Add examples for mgrid free-boundary, direct-coil free-boundary, and
   coil-only finite-beta QA optimization.

Success metrics:

- New users can modify objective tuples without changing source code.
- Examples run under documented smoke budgets in CI.
- Production examples print provenance, save input/WOUT/history, and plot
  actual initial and final states.

### Lane E: QI Seed-Robustness and Omnigenity

Goal: robust QI recovery from simple/minimal seeds for NFP1/2/3/4 when possible,
with evidence gates that do not confuse QP with true QI.

Steps:

1. Keep QI README artifacts provenance-gated and generated from minimal seed
   inputs, not from already-QI stage outputs.
2. Use smooth QI, legacy QI, mirror ratio, elongation, iota floor, and aspect
   gates together.
3. Keep NFP4 conservative if the current method is not robust.
4. Add landscape scans and stage checkpoints so failures preserve useful
   diagnostics.
5. Compare JAX QI metrics against the legacy Goodman/omnigenity objective and
   booz_xform_jax outputs.

Success metrics:

- Public cases satisfy `smooth_qi <= 5e-3`, legacy QI gate, aspect gate,
  mirror gate, and iota floor gate.
- Boozer contours close in the expected QI pattern by visual inspection before
  README promotion.
- The README panel uses actual initial minimal seeds and final optimized WOUTs.

### Lane F: Physics Gates and Stability Metrics

Goal: differentiable physics metrics with literature-anchored tests.

Steps:

1. Maintain DMerc and `D_R` AD-vs-FD gates.
2. Add finite-beta fixtures with pressure/current polynomial and spline
   profiles.
3. Add tests for magnetic well, current density, vector B, vector J, jdotB,
   iota/shear, beta, volume-average B, and pressure/current response.
4. Compare with VMEC2000 WOUT values and analytic near-axis limits where
   available.
5. Document formulas, assumptions, sign conventions, and expected applicability.

Success metrics:

- Physics gates are actual numerical/physics tests, not smoke-only tests.
- Coverage stays at or above the release target while CI runtime remains
  bounded.
- All metric docs include formulas and validation provenance.

### Lane G: CPU/GPU Performance

Goal: competitive cold solves, warm solves, and optimization callbacks on CPU
and GPU.

Steps:

1. Continue profiling force assembly, scan trials, accepted-point replay,
   tangent/JVP construction, and first-call exact tape creation.
2. Promote matrix-free NESTOR/source response only when profiling shows a clear
   mode-count threshold where it beats dense paths.
3. Cache shape-stable trace setup and avoid recompilation across same-shape
   accepted points.
4. Keep GPU enabled when users install GPU JAX; do not force CPU except in
   explicit user-selected modes.
5. Keep benchmark panels comparing VMEC2000, vmec_jax CPU, and vmec_jax GPU on
   bounded fixtures.

Success metrics:

- No hidden CPU-forcing in production code.
- Performance regressions are caught by compact benchmarks.
- Optimization callback traces report compile, force, replay, tangent, and I/O
  timing buckets.

### Lane H: Documentation, CI, Release Hygiene

Goal: docs and release artifacts match what the code can actually do.

Steps:

1. Keep README concise: installation, quick test, core examples, and best public
   figures only.
2. Move long optimization tables, sweep results, and limitations into docs.
3. Keep CI fast by sharding tests by cost and using optional external gates for
   VMEC2000/ESSOS.
4. Keep the git repository lean: large WOUTs/mgrids/artifacts live in releases
   or downloadable assets, not tracked history.
5. Add release gates: local smoke, docs build, coverage, physics gates, optional
   external parity, and artifact-size gate.

Success metrics:

- CI remains green and finishes within the documented budget.
- README claims are backed by tests or generated artifact provenance.
- New release tags are cut only after release gate results are recorded.

## Refactoring Plan

Priority refactors:

1. Separate controller state, physics state, and output state in
   `vmec_jax/driver.py` so derivative seams are easier to test.
2. Factor free-boundary branch fingerprints into a small public/internal data
   structure shared by reports, tests, and docs.
3. Split dense, projected, scalar, and matrix-free derivative paths behind a
   single policy interface.
4. Move QI stage-policy machinery out of example scripts into source modules.
5. Consolidate plotting helpers for boundary comparison, Boozer LCFS contours,
   objective history, stability profiles, and coil geometry.
6. Add docstrings to all public objective objects and derivative policy entry
   points.

Refactoring success gates:

- No example optimization driver exceeds the agreed readability guard unless
  there is an explicit test exemption.
- Public APIs have tests for nominal behavior, failure behavior, and docs
  examples.
- Internal helpers are testable without long VMEC solves.

## Test Matrix

Required local/CI test classes:

1. Unit tests for input parsing, profile construction, Fourier mode selection,
   coil geometry, mgrid interpolation, and objective assembly.
2. AD-vs-FD tests for fixed-boundary boundary modes, pressure/current profiles,
   coil currents, coil Fourier coefficients, DMerc, `D_R`, QS/QI objectives,
   and free-boundary branch-local reports.
3. VMEC2000 parity tests for selected fixed-boundary and bounded free-boundary
   converged equilibria.
4. Physics gates for finite-positive WOUTs, aspect, iota, pressure/beta,
   magnetic well, Mercier/Glasser, Bnormal RMS, and Boozer-space symmetry.
5. Regression tests for saved input.final reruns matching optimization WOUTs.
6. Performance smoke tests for cold solve, warm solve, accepted replay, scalar
   adjoint, matrix-free response, and GPU paths when GPU is available.
7. Artifact provenance tests for README/docs figures.

Optional external gates:

- VMEC2000 executable parity.
- ESSOS direct-coil and generated-mgrid parity.
- booz_xform_jax parity and Boozer plot generation.
- GPU benchmark matrix on `ssh office`.

## Promotion Gates

A feature can be promoted in docs/README only when:

1. There is a unit or physics test covering the API.
2. There is an AD-vs-FD or external parity gate for derivative claims.
3. Failure modes are documented and tested.
4. Runtime is bounded or the test is explicitly optional.
5. The claim is scoped to the actual differentiability seam tested.

Branch-differentiation promotion rules:

1. Same branch and same fingerprint: may promote after AD-vs-FD passes.
2. Changed branch fingerprint: must reject derivative promotion.
3. Smooth surrogate controller: may be documented as a research surrogate only
   until it matches production host-controller outputs and VMEC2000 parity.
4. Arbitrary hard adaptive branch changes: do not claim classical derivatives;
   use nonsmooth optimization language, subgradient/surrogate language, or
   derivative-free/global proposal language as appropriate.

## Milestones

Milestone 1: current release closeout.

- Keep phase 2 conservative and green.
- Keep phase 3 coil-only example complete-solve-authoritative.
- Recover CI after QI example guard fix.
- Record deferred differentiability plan.

Milestone 2: fixed-boundary derivative hardening.

- Expand profile/stability/objective AD-vs-FD gates.
- Refactor public optimization objective tuple assembly.
- Keep examples and docs aligned.

Milestone 3: free-boundary branch-local production hardening.

- Add more physical scalars to same-fingerprint complete-solve FD gates.
- Expand bounded direct-coil/mgrid/VMEC2000 parity fixtures.
- Improve runtime of branch-local vector/JVP reports.

Milestone 4: optional differentiable adaptive controller research.

- Implement a fully JAX-visible controller prototype.
- Compare step-by-step branches against the host controller.
- Add changed-branch diagnostics and nonsmooth objective experiments.
- Promote only if it provides a capability that branch-local reports cannot.

Milestone 5: single-stage optimization release.

- Demonstrate coil-only QA/QH or finite-beta QA with direct coils.
- Keep complete solves as acceptance authority.
- Use exact or branch-local validated derivatives for proposals.
- Compare against mgrid and VMEC2000 bounded fixtures.

Milestone 6: research-grade differentiable VMEC variant.

- Unified fixed/free-boundary differentiability story.
- Validated physics metrics and optimization examples.
- CPU/GPU benchmarked derivative policies.
- Documentation explicitly states every promoted and non-promoted seam.

## Current Open-Lane Completion Snapshot

These percentages describe the current main-branch plan state, not the deferred
research lane:

- Direct-coil/free-boundary phase 1: 100%.
- Full nonlinear free-boundary adjoint phase 2: 99.9999998% for fixed
  same-branch/fingerprint-gated gates; arbitrary adaptive branch changes remain
  unclaimed.
- VMEC parity and physics gates: 99.8%.
- Single-stage coil-only optimization phase 3: 100% for conservative
  complete-solve-authoritative examples; future publication-grade arbitrary
  adaptive derivatives remain deferred.
- CPU/GPU performance: 99.4%.
- CI/runtime/coverage hygiene: 100% locally; latest GitHub Actions run must
  pass after the QI line-count repair.
- Docs/release hygiene: 100%.
- QI minimal-seed README artifacts: 97.5% infrastructure/provenance-ready;
  public promotion depends on current NFP evidence.
- Deferred differentiable adaptive-controller research lane: 5%.

## Decisions Needed Later

1. Should vmec_jax implement a fully JAX-visible adaptive controller as a
   research surrogate, knowing it may be slower and may diverge from VMEC2000
   branch behavior?
2. Should arbitrary hard branch changes be handled with nonsmooth optimization,
   smoothing/relaxation, or derivative-free global proposals rather than AD?
3. What runtime budget is acceptable for publication-grade free-boundary
   coil-only optimization examples?
4. Which physics scalar is the release-critical promotion target for the next
   full adaptive branch-local gate: QS, Bnormal RMS, aspect, iota, or finite
   beta response?
5. Which QI NFP cases should be public release examples versus longer-running
   research artifacts?

## Running Log Template

For future updates, append entries with:

1. Date and commit.
2. Steps taken.
3. Results obtained.
4. Tests and commands run.
5. Best next steps.
6. User decisions needed.
7. Completion percentages by lane.

## 2026-06-14 Umbrella PR and Solver Helper Extractions

Commit: `6e8a335` plus follow-up extraction on
`codex/differentiability-refactor-plan`.

Steps taken:

1. Opened draft PR #20 as the single umbrella PR for the full
   differentiability/refactor plan.
2. Updated the PR title to make clear that this is the long-lived umbrella
   branch, not a short standalone planning PR.
3. Added the source-health diagnostic and documented it in
   `docs/code_structure.rst`.
4. Performed the first low-risk extraction from `vmec_jax/solve.py`:
   force-block mode weighting, lambda full-mesh residual norm, and
   stability-guard timestep calculation now live in
   `vmec_jax/solve_force_norm_helpers.py`.
5. Performed the second low-risk extraction from `vmec_jax/solve.py`:
   dtype-aware gradient, conjugate-gradient, and Levenberg-Marquardt tolerance
   policy now live in `vmec_jax/solve_tolerance_helpers.py`.
6. Performed the third low-risk extraction from `vmec_jax/solve.py`:
   fixed-boundary edge constraints, magnetic-axis regularity, lambda-gauge
   projection, and NumPy/JAX coefficient-slice helpers now live in
   `vmec_jax/solve_constraint_helpers.py`.
7. Performed the fourth low-risk extraction from `vmec_jax/solve.py`:
   gradient-descent state updates and feasible-gradient projection now live in
   `vmec_jax/solve_gradient_helpers.py`.
8. Kept backward-compatible private aliases in `solve.py` so existing tests and
   internal imports continue to work.

Results obtained:

1. Draft PR #20 CI passed before the follow-up extraction.
2. `solve.py` decreased from roughly 15438 to 14932 lines.
3. The extracted helpers are pure and synthetic-testable, making them a safe
   pattern for the next solver-kernel split.
4. Focused Ruff, pytest, source-health, and fast docs checks passed for the
   extracted helper modules.

Best next steps:

1. Keep all refactor work on PR #20 until the full plan is finalized.
2. Continue Wave 1/Wave 2 by extracting small pure solver helpers from
   `solve.py`: preconditioner application payloads, residual-loop
   controller-state bookkeeping, and scan/restart scalar policy adapters are
   the next low-risk candidates.
3. Add compatibility tests for every extracted private alias before ratcheting
   any source-health threshold.

User decisions needed:

No immediate decision.  PR #20 remains draft until the full refactor plan is
complete.

Completion:

- Differentiability/refactor plan: 100%.
- Differentiability/refactor implementation: 7%.
- Source-health instrumentation: 100%.
- Solver monolith reduction: 4% of the large-file extraction work.
