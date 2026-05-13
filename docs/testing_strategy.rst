Testing, Coverage, and Repository Size Strategy
===============================================

This page defines the validation target for ``vmec-jax``.  The goal is not to
collect many smoke tests; the goal is a compact, physics-based test suite that
protects VMEC2000 parity, numerical correctness, differentiability, and user
workflows while keeping required CI under ten minutes.


Target State
------------

- Required CI wall time: under ten minutes for the required test, docs, and
  build jobs on GitHub-hosted CPU runners.
- Current ``main`` local CI-equivalent coverage baseline: clean at
  ``d5a1adf`` on 2026-05-13 with ``623 passed, 21 skipped,
  85 deselected``, ``67.46%`` coverage, and ``8:37`` runtime.
- Long-term required coverage: 95% line coverage for ``vmec_jax`` package
  code.  The current Python 3.11 required coverage gate is ``63%``.
- Required local command: ``pytest -q -m "not full and not vmec2000"`` remains
  the routine development gate.
- Nightly/manual coverage: larger VMEC2000, GPU, and full-resolution physics
  checks run outside the required PR gate.
- Repository checkout size: keep the tracked source tree small enough that a
  fresh clone and ``pip install .`` are not dominated by generated figures,
  optimization outputs, or bulky reference ``wout`` files.


Current Command Map
-------------------

Run these from the repository root.  They mirror the current CI split and are
the recommended local escalation path.

.. list-table::
   :header-rows: 1
   :widths: 24 52 24

   * - Scope
     - Command
     - When to run
   * - Fast local gate
     - ``JAX_ENABLE_X64=1 pytest -q -m "not full and not vmec2000"``
     - Before pushing ordinary code or docs-adjacent changes that touch tested
       APIs.
   * - Bundled wout parity gate
     - ``JAX_ENABLE_X64=1 pytest -q tests/test_residue_getfsq_parity.py tests/test_wout_profiles_currents_bundled_parity.py tests/test_vmec2000_exec_threed1.py``
     - Required no-executable physics gate: recompute VMEC2000 ``fsqr/fsqz/fsql``,
       verify flux/pressure/iota/current wout-field invariants, and cover the
       VMEC2000 trace parser against bundled fixtures.
   * - Coverage gate
     - ``JAX_ENABLE_X64=1 pytest -q -m "not full and not vmec2000" --cov=vmec_jax --cov-report=xml --cov-report=term-missing:skip-covered --cov-fail-under=63``
     - Python 3.11 required CI coverage job.  The latest local equivalent on
       ``d5a1adf`` reported ``67.46%`` coverage while the enforced gate stayed
       at ``63%``.
   * - Optimization workflow smoke
     - ``pytest -q tests/test_optimization_examples.py tests/test_qs_ess_render_smoke.py``
     - After changing objective tuple construction, examples, or sweep
       rendering docs.
   * - QI objective checks
     - ``pytest -q tests/test_quasi_isodynamic.py tests/test_qi_legacy.py tests/test_qi_diagnostics.py tests/test_booz_input.py``
     - After changing QI diagnostics, Boozer input handling (including LASYM
       geometry/magnetic channels), smooth-QI residual settings, or first-class
       QI diagnostic record fields.
   * - QI ranking/report smoke
     - ``pytest -q tests/test_qi_objective_component_report.py tests/test_qi_seed_suitability_audit.py tests/test_qs_ess_render_smoke.py``
     - After changing QI branch-ranking metrics, seed audit/prefine manifests,
       sweep summary fields, or renderer selection rules.
   * - Optional validation plan helper
     - ``python validation/qi_seed_robustness_plan.py --output results/qi_seed_audit/validation_plan.json``
     - To record the current non-required VMEC2000/SIMSOPT/QI seed-robustness
       lanes and concrete bounded parity commands before a local or scheduled
       validation run.
   * - Bounded physics smoke
     - ``RUN_FULL=1 pytest -q tests/test_wout_comprehensive_parity.py::test_wout_comprehensive_parity[circular_tokamak] tests/test_wout_comprehensive_parity.py::test_wout_comprehensive_parity[nfp4_QH_warm_start] tests/test_driver_api.py::test_run_free_boundary_smoke_on_bundled_small_case``
     - Before merging solver changes that affect fixed/free-boundary physics.
   * - Full physics tier
     - ``python tools/fetch_assets.py`` then ``RUN_FULL=1 JAX_ENABLE_X64=1 pytest -q -m "full and not vmec2000"``
     - Manual/nightly parity and high-cost physics validation.
   * - External VMEC2000 tier
     - ``VMEC2000_EXEC=/path/to/xvmec2000 VMEC2000_INTEGRATION=1 pytest -q tests/test_vmec2000_exec_fast_validation.py::test_fast_vmec2000_stage_trace_validation_cases``
     - First executable-backed parity gate; broaden to ``pytest -q -m vmec2000``
       only after the bounded smoke is green.
   * - External SIMSOPT tier
     - ``RUN_SIMSOPT_VALIDATION=1 pytest -q tests/test_simsopt_optional_validation.py::test_qh_quasisymmetry_residual_matches_simsopt_wout_formula``
     - Optional formula-level diagnostic parity when SIMSOPT is installed.
   * - Docs fast build
     - ``SPHINX_FAST=1 LC_ALL=C.UTF-8 LANG=C.UTF-8 python -m sphinx -W -j auto -b html docs docs/_build/html``
     - Required build job, minimal landing page.
   * - Docs full build
     - ``READTHEDOCS=True LC_ALL=C.UTF-8 LANG=C.UTF-8 python -m sphinx -W -j auto -b html docs docs/_build/html_full``
     - Required full guide/API docs job.


Test Tiers
----------

The suite is split by execution cost and external dependencies:

.. list-table::
   :header-rows: 1

   * - Tier
     - Marker / CI job
     - Purpose
     - Target runtime
   * - Unit kernels
     - unmarked
     - Pure array, Fourier, namelist, boundary, geometry, profile, and helper
       invariants.  These should be deterministic, small, and dependency-light.
     - 1-2 minutes total
   * - Bounded physics parity
     - unmarked or dedicated required selections
     - Low-resolution fixed/free-boundary solves with physics assertions:
       convergence, Jacobian sign, force residual, surface geometry, field
       positivity, and selected ``wout`` scalar/profile parity.
     - 5-8 minutes total
   * - Workflow and docs artifacts
     - unmarked required selections
     - CLI smoke tests, optimization objective tuple construction, example
       workflow assembly, sweep rendering smoke tests, and repository size
       audit.
     - seconds to a few minutes
   * - Full physics
     - ``full``
     - Larger reference assets, multigrid cases, high-resolution regression
       decks, and broader parity matrices.
     - nightly/manual
   * - External VMEC2000
     - ``vmec2000``
     - Direct comparison against a local Fortran executable such as
       ``~/bin/xvmec2000``.  These tests validate that stored parity fixtures
       have not drifted.
     - local/nightly/manual
   * - GPU
     - backend-specific selection
     - Accelerator parity, allocator behavior, compile/replay cost, and GPU
       optimization callback checks.
     - dedicated GPU runner


Physics Gates
-------------

Every nontrivial solver change should be covered by at least one physics gate,
not only by a smoke test that checks that a file was produced.

Core solve gates:

- ``FSQR``, ``FSQZ``, and ``FSQL`` decrease or satisfy the requested tolerance
  on representative fixed-boundary cases.
- The final equilibrium has finite positive ``|B|`` on the LCFS and no obvious
  Jacobian sign failure.
- Axisymmetric and stellarator-symmetric cases preserve the expected Fourier
  mode parity.
- ``LASYM = T`` cases exercise both symmetric and asymmetric coefficient
  blocks and verify that asymmetric modes can change when active.
- Free-boundary cases verify that the mgrid path and vacuum-field coupling are
  used, not silently bypassed.

VMEC2000 parity gates:

- Required CI includes ``tests/test_residue_getfsq_parity.py``.  This reads
  small bundled VMEC2000 ``wout`` files, reconstructs the equilibrium state,
  recomputes the scalar residual pipeline
  ``bcovar -> forces -> tomnsps -> getfsq``, and compares ``fsqr/fsqz/fsql``
  to the values stored by VMEC2000.  It is not a smoke test: it fails on force
  normalization, Fourier convention, or residual assembly drift while avoiding
  a full solve.
- Required CI also includes ``tests/test_vmec2000_exec_threed1.py`` so the
  ``threed1`` parser used by executable-backed diagnostics is covered by a
  bundled fixture even when ``xvmec2000`` is unavailable.
- Required CI includes ``tests/test_wout_profiles_currents_bundled_parity.py``.
  This no-solve gate reads converged bundled ``wout`` fixtures and verifies
  input flux profiles, half-mesh ``phi`` integration, finite-beta
  ``pres/presf`` staggering, ``iotas -> iotaf`` radial smoothing, and the
  VMEC surface-averaged Ampere relation
  ``jcuru = -d(bvco)/ds / mu0`` and ``jcurv = d(buco)/ds / mu0`` on interior
  surfaces.
- Compare converged equilibria, not arbitrary finite-step transient states,
  unless the test is explicitly a short-trace regression.
- For each small reference deck, compare key scalars and profiles:
  aspect ratio, volume, toroidal flux, ``iotaf/iotas``, pressure profiles,
  ``bmnc/bmns``, covariant and contravariant field coefficients, and force
  norms with documented tolerances.
- Keep a manifest of cases covering axisymmetry, non-axisymmetry,
  up-down asymmetry, stellarator asymmetry, free boundary, single-grid, and
  multigrid.

Numerical-method gates:

- Fourier transforms and inverse transforms are tested against analytic spectra
  and hand-built low-mode examples.
- ``m=1`` axis and constraint handling are tested with sign and indexing
  conventions matching VMEC2000.
- JAX scan and non-scan solver paths agree on the same bounded inputs.
- Fallback convergence paths are triggered early and only when needed; tests
  assert both the early trigger and the parity-preserving final equilibrium.

Differentiability gates:

- JVP/VJP and discrete-adjoint callbacks are checked against finite differences
  on small deterministic cases, with tolerances appropriate to double
  precision.
- Optimization objective gradients are nonzero for seeded active boundary
  modes that should move iota, aspect ratio, quasisymmetry, or QI residuals.
- Exact callback replay must not retain unbounded host or XLA state across a
  bounded sequence of accepted points.

Optimization gates:

- The examples should include tiny callback-budget regressions that verify
  construction, objective assembly, ESS scaling, continuation seeding, and
  artifact writing.
- ``LeastSquaresProblem.from_tuples`` should preserve SIMSOPT weight semantics:
  tuple ``weight`` means a residual multiplier of ``sqrt(weight)``.
- QI workflow tests should cover routing of ``QuasiIsodynamicResidual`` terms,
  rejection of invalid nonzero QI targets, and compatibility of smooth QI
  metrics with the legacy branch-ranking diagnostics.  The cheap synthetic
  Boozer ranking guard is
  ``pytest -q tests/test_qi_objective_component_report.py``; it must stay fast
  enough to run in ordinary PR checks.
- QI diagnostic-record tests should keep smooth/raw/legacy QI totals,
  mirror-ratio, elongation, optional ``LgradB``, resolution metadata, and
  diagnostic error fields stable enough for sweep renderers and downstream audit
  scripts.
- GPU optimization tests should assert the device-aware defaults separately:
  accepted-point exact callbacks default to tape on CPU and GPU, while relaxed
  trial residuals default to scan unless ``VMEC_JAX_OPT_TRIAL_SCAN=0`` is set.
- Full optimization sweeps are not required PR tests; they remain generated
  benchmark artifacts documented in :doc:`optimization_sweep_results`.

QI seed-robustness gates:

- Required PR tests protect the metric semantics, not global optimizer
  robustness.  The cheap QI gates use synthetic Boozer spectra and mocked
  state diagnostics to ensure smooth QI, legacy branch/shuffle QI, mirror
  ratio, elongation, optional ``LgradB``, and summary metadata stay compatible.
- The next realistic validation step is a small solved-state diagnostic fixture
  using ``qi_diagnostics_from_state``.  It should compare smooth QI, legacy QI,
  mirror ratio, elongation, iota, aspect ratio, and Boozer ``|B|`` contour
  quality on one bounded case without launching a full optimization sweep.
- Use ``examples/optimization/audit_qi_seed_suitability.py --quick`` as the
  no-optimization preflight before a multi-seed QI sweep.  It ranks existing
  solved seeds and records missing optional reference checkouts instead of
  making the default gate machine-specific.  The audit defaults to
  ``include_bounce_endpoints=True`` so smooth-QI seed ranking uses the same
  normalized level range as the legacy Goodman-style branch/shuffle diagnostic.
- Use ``--prefine-probes plan`` to write a hard-capped QI-only probe manifest
  before any actual seed-robustness run.  The manifest is a review artifact:
  it makes selected seeds, run commands, and output paths explicit before
  expensive probes start.
- Use ``validation/qi_seed_robustness_plan.py`` to record the optional
  validation lanes and acceptance criteria.  The plan includes required CI
  baseline checks, family-representative QI solved-state audit, dry-run prefine
  manifests, explicit tiny prefine runs, SIMSOPT formula parity, and VMEC2000
  executable smoke.  It is intentionally declarative and must not become a
  heavy required CI lane.
- A full seed-robust QI claim requires starting constrained QI from QI, QP, QH,
  QA, and a simple non-omnigenous seed, then auditing convergence, legacy QI
  score, engineering constraints, and Boozer contour plots.  That matrix is
  manual/nightly validation until it is cheap enough to summarize as curated
  artifacts.

The current detailed lane list and next parity gates are in
:doc:`optional_validation_plan`.


Release Checklist
-----------------

Use :doc:`release_checklist` as the command-level gate before tagging.  The
checklist keeps the release path tied to the same required lanes described
here: continuation correctness, exact accepted-point history/output selection,
VMEC residual parity, Boozer/LASYM input spectra, QI diagnostic metadata,
warning-clean docs, CI status, and artifact hygiene.


Coverage Plan to 95%
--------------------

Coverage should rise because core physics code is directly tested, not because
tests execute long workflows incidentally.

1. Split monolithic solver sections into small internal functions only where it
   improves testability without changing numerical behavior.
2. Add unit-level tests for every extracted physics kernel: geometry, Fourier
   synthesis, force assembly, residual norms, profile interpolation,
   preconditioner blocks, and wout serialization.
3. Add low-resolution parity fixtures for each physics class in the manifest.
   The fixtures should be small enough to run in required CI after
   ``tools/fetch_assets.py``.
4. Move large generated data and figures out of the tracked tree before raising
   coverage gates.  Coverage runs should not require downloading presentation
   artifacts.
5. Raise ``--cov-fail-under`` in stages after the corresponding tests are
   merged.  The current required fast-suite gate is ``63%``; the latest local
   baseline is ``67.46%``.  The next planned ratchets are 70%, 80%, 90%, then
   95%.


Repository Size Plan
--------------------

The current source tree is dominated by generated documentation figures and
reference outputs.  The target is to keep only source, small fixtures, and
current README figures in git.

Actions:

- Keep generated optimization sweeps under ``examples/optimization/results/``
  ignored and out of the repository.
- Keep only README-critical PNGs in ``docs/_static/figures``.  Keep large
  publication panels, PDFs, and historical atlases as generated local artifacts
  under ``examples/optimization/results`` or attach them to GitHub releases.
- Replace committed large ``wout`` references with small compressed fixtures
  or generated-on-demand assets where runtime permits.
- Add a size audit to every release checklist and eventually to CI with a
  documented threshold.
- Avoid committing both PNG and PDF versions unless both are directly linked
  from docs.

Run the current audit with:

.. code-block:: bash

   python tools/diagnostics/repo_size_audit.py --top 40

Required CI also runs the audit with an initial source-tree ceiling of
``60 MiB`` total and ``5 MiB`` per tracked file.  Increase those limits only
when a new small reference fixture has a documented physics-test purpose.


Refactoring Plan
----------------

Refactoring should make the code easier to test and reason about without
changing reference behavior.

- Preserve public APIs such as ``run_fixed_boundary``, ``run_free_boundary``,
  CLI behavior, ``wout`` writing, and optimization example workflows.
- Extract small pure functions from large numerical routines only when the
  extracted function has a clear mathematical contract and a targeted test.
- Prefer seam extractions that already have parity evidence: VMEC force
  helpers, residual normalization, Mercier/Redl algebra, wout schema helpers,
  and optimization tuple/routing policy.  Do not split solve orchestration,
  free-boundary coupling, or accepted-point replay as a broad cleanup without a
  narrow benchmark or parity gate.
- Add docstrings to public and semi-public functions that state conventions:
  radial mesh, VMEC vs physical toroidal angle, Fourier signs, symmetry
  assumptions, and units.
- Add comments near non-obvious VMEC2000 compatibility choices, especially
  sign conventions, mode indexing, fallback convergence logic, and scan
  controller decisions.
- Keep JAX import and backend selection lazy enough that users can choose CPU or
  GPU before importing heavy runtime code.
