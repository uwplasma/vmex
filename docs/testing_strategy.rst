Testing, Coverage, and Repository Size Strategy
===============================================

This page defines the validation target for ``vmec-jax``.  The goal is not to
collect many smoke tests; the goal is a compact, physics-based test suite that
protects VMEC2000 parity, numerical correctness, differentiability, and user
workflows while keeping required CI bounded by explicit job timeouts.


Target State
------------

- Required CI wall time: keep required test, docs, build, bounded physics, and
  coverage jobs below ten minutes on hosted CPU runners.  Required jobs use
  explicit ``timeout-minutes: 10`` guards; manual/nightly full-physics jobs are
  the only intentionally longer lane.
- Routine local gate:
  ``JAX_ENABLE_X64=1 pytest -q -m "not full and not vmec2000 and not simsopt"``.
- Local CI-equivalent coverage runs should record pass/skip/deselect counts,
  coverage percentage, runtime, Python/JAX versions, and commit SHA in the
  release notes or validation artifact for the candidate being checked.
- Current coverage target: keep the required ``95%`` actual line coverage gate
  green with meaningful fast and bounded-physics tests while preserving
  sub-ten-minute required CI runtime.  The current Python 3.11 required
  coverage gate is ``95%``; the latest green required CI run before the final
  layout cleanup had a slowest shard of about ``505 s`` and a combined coverage
  gate of about ``32 s``.
- Nightly/manual coverage: larger VMEC2000, GPU, and full-resolution physics
  checks run outside the required PR gate.
- Repository checkout size: keep the tracked source tree small enough that a
  fresh clone and ``pip install .`` are not dominated by generated figures,
  optimization outputs, or bulky reference ``wout`` files.


Testing Taxonomy
----------------

The suite uses marker- and command-level taxonomy rather than a single
``pytest`` bucket.  New tests should be assigned to the cheapest bucket that
proves the behavior under review.

.. list-table::
   :header-rows: 1
   :widths: 18 30 32 20

   * - Bucket
     - Selection
     - Owns
     - CI policy
   * - Fast required
     - Unmarked tests included by
       ``-m "not full and not vmec2000 and not simsopt"``
     - Pure kernels, parsing, small workflow assembly, CLI helpers,
       differentiability checks, repository hygiene, and deterministic
       low-cost physics guards.
     - Required on Python 3.10, 3.11, and 3.12.  Python 3.11 also owns the
       line-coverage gate.
   * - Required fetched-fixture parity
     - Explicit required test-file selections using released WOUT fixtures and no
       external executables.
     - VMEC2000 residual reconstruction, converged ``wout`` profile/current
       invariants, parser compatibility, and bounded matrix physics gates.
     - Required where runtime and fixture size are bounded.  Failures represent
       real parity drift, not missing local tools.
   * - Full physics
     - ``full`` marker plus ``RUN_FULL=1``; fetched-asset rows additionally
       require ``tools/fetch_assets.py``.
     - Larger fixed/free-boundary references, multigrid cases, convergence-only
       representatives, and broader parity matrices.
     - Manual or scheduled/nightly unless a case is reduced enough to promote
       into required fetched-fixture parity.
   * - External VMEC2000
     - ``vmec2000`` marker plus ``VMEC2000_EXEC`` and
       ``VMEC2000_INTEGRATION=1``.
     - Executable-backed convergence and final-equilibrium parity against a
       local Fortran VMEC2000 build.
     - Never required for ordinary PR CI because availability is
       machine-specific.
   * - External SIMSOPT
     - ``simsopt`` marker plus ``RUN_SIMSOPT_VALIDATION=1`` and importable
       SIMSOPT.
     - Formula-level and workflow-level diagnostic parity against SIMSOPT.
     - Optional release validation only.
   * - Documentation and packaging
     - Sphinx fast/full builds, wheel/sdist build, compileall, and source-tree
       size audit.
     - Warning-clean docs, installability, importability, and artifact hygiene.
     - Required, but separate from line coverage so docs-only changes do not
       distort package coverage.


Testing Layers
--------------

Run the cheapest layer that can catch the failure mode under review, then
escalate only when the change affects solver physics, external parity, or
generated artifacts.

.. list-table::
   :header-rows: 1
   :widths: 22 35 43

   * - Layer
     - Local command
     - Purpose
   * - Fast unit tests
     - ``JAX_ENABLE_X64=1 pytest -q -m "not full and not vmec2000 and not simsopt"``
     - Required daily gate for pure kernels, Fourier conventions, namelist
       parsing, boundary helpers, geometry/profile algebra, CLI helpers, and
       small workflow assembly.  These tests should stay deterministic and
       dependency-light.
   * - Required CI
     - Fast suite plus the coverage and fetched-fixture parity commands below.
     - Protects importability, package build, fast docs, line coverage, and
       no-executable VMEC2000 parity fixtures.  CI should fail on real numerical
       drift without requiring local VMEC2000, SIMSOPT, GPUs, or large assets.
   * - Full physics validation
     - ``RUN_FULL=1 JAX_ENABLE_X64=1 pytest -q -m "full and not vmec2000"``;
       run ``python tools/fetch_assets.py`` first for fetched-asset rows.
     - Manual or nightly validation for larger reference assets, multigrid
       cases, broad fixed/free-boundary parity, and expensive regression
       matrices that are too slow for every PR.
   * - Optional VMEC2000 parity
     - ``VMEC2000_EXEC=/path/to/xvmec2000 VMEC2000_INTEGRATION=1 pytest -q -m vmec2000``
     - Executable-backed comparison against the Fortran code.  Prefer
       converged final-equilibrium ``wout`` and diagnostic parity; use
       short-trace regressions only for timestep/stage-trace compatibility.
       Do not make the required gate depend on a machine-local executable.
   * - Optional SIMSOPT parity
     - ``RUN_SIMSOPT_VALIDATION=1 JAX_ENABLE_X64=1 pytest -q -m simsopt``
     - Formula-level and workflow-level diagnostic parity when SIMSOPT is
       installed.  These tests are useful for release validation but remain
       optional because dependency availability is machine-specific.
   * - Documentation
     - ``SPHINX_FAST=1 LC_ALL=C.UTF-8 LANG=C.UTF-8 python -m sphinx -W -j auto -b html docs docs/_build/html``
     - Required fast docs build.  Keep it focused on the landing page; full docs
       and API pages remain in the non-fast build.


Local Coverage Workflow
-----------------------

Run coverage from the repository root after installing the development extras
or at least ``pytest-cov``.  The CI coverage job installs the plain package,
which includes matplotlib-backed plotting support, so plotting tests contribute
to the required line-coverage gate instead of being skipped.

Required CI-equivalent coverage:

.. code-block:: bash

   JAX_ENABLE_X64=1 pytest -q -m "not full and not vmec2000 and not simsopt" \
     --cov=vmec_jax \
     --cov-report=xml \
     --cov-report=term:skip-covered \
     --cov-fail-under=95

Full-physics coverage add-on when assets are available:

.. code-block:: bash

   python tools/fetch_assets.py
   RUN_FULL=1 JAX_ENABLE_X64=1 pytest -q -m "full and not vmec2000" \
     --cov=vmec_jax \
     --cov-append \
     --cov-report=term-missing:skip-covered \
     --cov-report=html

Use ``--cov-report=term-missing:skip-covered`` for small follow-ups and
``htmlcov/index.html`` for module-level triage.  Coverage ratchets should follow
real tests for physics kernels and user workflows; do not raise the threshold
because optional dependency tests happened to run on one developer machine.


Opt-in Local CI Gate
--------------------

Before pushing a change that is likely to consume GitHub Actions minutes, run
the local CI gate from the repository root:

.. code-block:: bash

   python tools/diagnostics/repo_health/local_ci_gate.py

This command is opt-in; it is not installed as a Git hook and it does not run
automatically on every push.  It mirrors the required hosted CI lanes that are
safe to run on a normal developer machine:

- CLI smoke via ``vmec --help`` and a two-iteration
  ``input.circular_tokamak`` solve.
- Python compile check for package, examples, tests, tools, and validation
  helpers.
- Repository size audit using
  ``tools/diagnostics/repo_health/repo_size_audit.py --top 20 --max-total-mib 50 --max-file-mib 2``.
- Fast required pytest suite with the current ``95%`` coverage fail-under:
  ``JAX_ENABLE_X64=1 pytest -q -m "not full and not vmec2000 and not simsopt"``
  plus ``--cov=vmec_jax`` and ``--cov-fail-under=95``.
- Bounded physics smoke after ``python tools/fetch_assets.py``.
- Wheel/sdist build.
- Fast Sphinx docs with ``SPHINX_FAST=1``.
- Full Sphinx docs with warnings as errors.

Use ``python tools/diagnostics/repo_health/local_ci_gate.py --dry-run`` to inspect the
commands without running them.  Use ``--list`` to see stage names, ``--only
STAGE`` to run a single lane, and ``--skip STAGE`` to omit a lane when the
change scope or local environment makes that appropriate.  Keep VMEC2000,
SIMSOPT, GPU, and broader full-physics validation in the optional lanes below;
they remain intentionally outside this local pre-push gate.


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
     - ``JAX_ENABLE_X64=1 pytest -q -m "not full and not vmec2000 and not simsopt"``
     - Before pushing ordinary code or docs-adjacent changes that touch tested
       APIs.
   * - Released WOUT parity gate
     - ``python tools/fetch_assets.py --bundle wout-fixtures && JAX_ENABLE_X64=1 pytest -q tests/test_residue_getfsq_parity.py tests/io/wout/test_wout_profiles_currents_bundled_parity.py tests/parity/test_physics_parity_helper_gates.py tests/diagnostics/parity/test_vmec2000_exec_threed1.py``
     - Required no-executable physics gate: recompute VMEC2000 ``fsqr/fsqz/fsql``,
       verify flux/pressure/iota/current wout-field invariants, protect small
       Mercier/JXBFORCE/Boozer helper identities, and cover the VMEC2000 trace
       parser against released fixtures.
   * - Parity manifest guard
     - ``pytest -q tests/diagnostics/parity/test_parity_sweep_manifest_thresholds.py``
     - Cheap schema and fixture-path check for the fixed/free-boundary parity
       sweep manifest.  It does not launch VMEC2000; it protects the bounded
       optional sweep matrix from stale local inputs, unbounded compare-mode
       entries, and accidental removal of required physics classes.  This is
       dry-run wiring only, not evidence that the external manifest matrix has
       completed.
   * - Coverage gate
     - ``JAX_ENABLE_X64=1 pytest -q -m "not full and not vmec2000 and not simsopt" --cov=vmec_jax --cov-report=xml --cov-report=term:skip-covered --cov-fail-under=95``
     - Python 3.11 required CI coverage job.  Record the measured coverage,
       pass/skip/deselect counts, runtime, and commit SHA in the validation
       artifact for the candidate being checked; the gate itself is 95%
       while optional executable validation stays in separate opt-in lanes.
   * - Optimization workflow smoke
     - ``pytest -q tests/optimization/test_examples.py tests/diagnostics/optimization/test_qs_ess_render_smoke.py``
     - After changing objective tuple construction, examples, or sweep
       rendering docs.
   * - QI objective checks
     - ``pytest -q tests/objectives/test_quasi_isodynamic.py tests/objectives/test_qi_legacy.py tests/objectives/test_qi_diagnostics.py tests/postprocessing/test_booz_input.py``
     - After changing QI diagnostics, Boozer input handling (including LASYM
       geometry/magnetic channels), smooth-QI residual settings, or first-class
       QI diagnostic record fields.  This includes a low-resolution
       ``qi_diagnostics_from_state`` check on the bundled
       ``input.QI_stel_seed_3127`` solved seed when optional Boozer dependencies
       are installed.
   * - QI ranking/report smoke
     - ``pytest -q tests/objectives/test_qi_objective_component_report.py tests/diagnostics/qi/test_qi_seed_suitability_audit.py tests/diagnostics/optimization/test_qs_ess_render_smoke.py``
     - After changing QI branch-ranking metrics, seed audit/prefine manifests,
       sweep summary fields, or renderer selection rules.
   * - Optional validation plan helper
     - ``python validation/qi_seed_robustness_plan.py --output results/qi_seed_audit/validation_plan.json``
     - To record the current non-required VMEC2000/SIMSOPT/QI seed-robustness
       lanes and concrete bounded parity commands before a local or scheduled
       validation run.
   * - Bounded physics smoke
     - ``python tools/diagnostics/repo_health/local_ci_gate.py --only fetch-assets --only physics-smoke``
     - Before merging solver changes that affect fixed/free-boundary physics.
       Use ``--dry-run`` on the same command to print the current expanded
       fixture list instead of copying a stale hand-maintained command.
   * - Full physics tier
     - ``python tools/fetch_assets.py`` then ``RUN_FULL=1 JAX_ENABLE_X64=1 pytest -q -m "full and not vmec2000"``
     - Manual/nightly parity and high-cost physics validation.  The latest
       local coverage-appended run reached ``72.35%`` coverage with
       ``74 passed, 4 skipped`` in ``27:21`` after refreshing released assets.
   * - External VMEC2000 tier
     - ``VMEC2000_EXEC=/path/to/xvmec2000 VMEC2000_INTEGRATION=1 JAX_ENABLE_X64=1 pytest -q tests/parity/test_vmec2000_exec_fast_validation.py::test_vmec2000_converged_wout_diagnostics_validation``
     - Preferred executable-backed release gate: run low-resolution bundled
       inputs to convergence in VMEC2000 and ``vmec_jax``, then compare final
       ``wout`` geometry, flux/profile, magnetic-field, and scalar diagnostics.
       Broaden to ``pytest -q -m vmec2000`` only after the bounded smoke is
       green.  The ``basic_non_stellsym_pressure`` converged LASYM finite-beta
       representative passed locally after the LASYM covariant-field scaling
       fix in ``e0b00e7`` and remained below the nightly LASYM magnetic
       tolerance in a bounded 2026-05-19 rerun; keep it as dated optional
       evidence rather than a broad strict-LASYM parity promotion or required
       PR check. Keep
       ``tests/parity/test_vmec2000_exec_fast_validation.py::test_fast_vmec2000_stage_trace_validation_cases``
       for deliberate short-trace regressions.
   * - External SIMSOPT tier
     - ``RUN_SIMSOPT_VALIDATION=1 JAX_ENABLE_X64=1 pytest -q -m simsopt tests/test_simsopt_optional_validation.py tests/postprocessing/test_redl_bootstrap_simsopt_parity.py``
     - Optional SIMSOPT diagnostic parity on bundled converged ``wout``
       fixtures: QS formula parity, state-derived QS diagnostics, and Redl
       bootstrap mismatch normalization.
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
  used, not silently bypassed.  The physics-smoke tier also includes a bounded
  finite-pressure CTH-like response check: a zero-pressure and nonzero-pressure
  free-boundary solve must differ in LCFS geometry and LCFS ``|B|``.  The
  optional VMEC2000 tier extends this to a DIII-D mgrid finite-beta response
  comparison against the external executable.
- Direct-coil free-boundary provider tests must exercise the same physical
  contracts as an ``mgrid`` backend.  Required fast tests now generate an
  in-memory VMEC-layout ``mgrid`` by sampling the pure-JAX Biot-Savart coil
  provider and then require the JAX ``mgrid`` interpolator to reproduce the
  same field at grid nodes, including toroidal wrapping and external-current
  scaling.
- Robust-coil perturbation tests should protect geometry and field invariants,
  not only random-number plumbing.  Required fast tests check that rigid
  centerline translations/rotations preserve coil length and curvature, and
  that current perturbations scale the direct Biot-Savart field linearly.

VMEC2000 parity gates:

- Required CI includes ``tests/test_residue_getfsq_parity.py``.  This reads
  released VMEC2000 ``wout`` fixtures, reconstructs the equilibrium state,
  recomputes the scalar residual pipeline
  ``bcovar -> forces -> tomnsps -> getfsq``, and compares ``fsqr/fsqz/fsql``
  to the values stored by VMEC2000.  It is not a smoke test: it fails on force
  normalization, Fourier convention, or residual assembly drift while avoiding
  a full solve.
- Required CI also includes ``tests/diagnostics/parity/test_vmec2000_exec_threed1.py`` so the
  ``threed1`` parser used by executable-backed diagnostics is covered by a
  bundled fixture even when ``xvmec2000`` is unavailable.
- Required CI includes ``tests/io/wout/test_wout_profiles_currents_bundled_parity.py``.
  This no-solve gate reads converged released ``wout`` fixtures and verifies
  input flux profiles, half-mesh ``phi`` integration, finite-beta
  ``pres/presf`` staggering, ``iotas -> iotaf`` radial smoothing, and the
  VMEC surface-averaged Ampere relation
  ``jcuru = -d(bvco)/ds / mu0`` and ``jcurv = d(buco)/ds / mu0`` on interior
  surfaces.
- Required CI includes ``tests/parity/test_physics_parity_helper_gates.py``.  This
  small no-solve helper gate uses released converged ``wout`` fixtures to
  protect VMEC Mercier decomposition, JXBFORCE ``bdotgradv`` normalization,
  magnetic-well endpoint extrapolation, Boozer spectral handoff conventions,
  the edge-current ``ctor`` scalar, and the normalized ``equif`` radial
  force-balance identity.
- Compare converged equilibria, not arbitrary finite-step transient states,
  unless the test is explicitly a short-trace regression.
- For each small reference deck, compare key scalars and profiles:
  aspect ratio, volume, toroidal flux, ``iotaf/iotas``, pressure profiles,
  ``bmnc/bmns``, covariant and contravariant field coefficients, and force
  norms with documented tolerances.
- Keep a manifest of cases covering axisymmetry, non-axisymmetry,
  up-down asymmetry, stellarator asymmetry, free boundary, single-grid, and
  multigrid.

SIMSOPT parity gates:

- SIMSOPT checks are always optional: tests are marked ``simsopt`` and require
  ``RUN_SIMSOPT_VALIDATION=1`` in addition to the importable SIMSOPT package.
- Use released converged ``wout`` fixtures rather than launching external
  solvers.  The current gate compares the VMEC-only QS formula, the
  state-derived QS diagnostic path reconstructed from ``wout``, and Redl
  bootstrap mismatch normalization against SIMSOPT.
- These tests should skip when SIMSOPT or its optional runtime dependencies are
  absent; a required CI failure should never depend on a developer-machine
  SIMSOPT installation.

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
- For the free-boundary direct-coil lane, exact-gradient claims are staged.
  Required tests validate provider derivatives, dense implicit vacuum chains,
  JAX-visible controller primitives, and accepted-trace replay.  The strongest
  default gate currently runs base/plus/minus complete tiny free-boundary
  solves for a current-only perturbation, rejects finite-difference samples
  that leave the accepted branch, and compares the fixed-trace/controller
  custom-VJP directional derivative against the complete-solve central finite
  difference.  This is a same-branch accepted-trace validation, not yet a
  derivative of every adaptive host-controller branch.  Promoting adaptive
  branch differentiation requires a fingerprint-gated full adaptive
  AD-vs-central-FD gate through the adaptive loop, not only accepted-trace
  replay.
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
  optional ``BoozerBTarget`` homotopy terms for far seeds, rejection of invalid
  nonzero QI targets, and compatibility of smooth QI metrics with the legacy
  branch-ranking diagnostics.  The cheap synthetic Boozer ranking guard is
  ``pytest -q tests/objectives/test_qi_objective_component_report.py``; it must stay fast
  enough to run in ordinary PR checks.
- QI diagnostic-record tests should keep smooth/raw/legacy QI totals,
  mirror-ratio, elongation, optional ``LgradB``, resolution metadata, and
  diagnostic error fields stable enough for sweep renderers and downstream
  audit scripts.  ``rank_qi_seed_records`` is part of the public API and should
  remain covered through both ``vmec_jax`` and ``vmec_jax.api`` re-export
  checks.  The bundled solved-seed gate intentionally uses a tiny grid; it is a
  metadata and sign/range regression, not a replacement for
  publication-resolution Boozer contour review.
- Optimization tests should assert the device-aware defaults separately:
  accepted-point exact callbacks default to the tape path on CPU and GPU, CPU
  keeps full tapes, GPU/CUDA/ROCm uses JVP-only exact tapes with basepoint
  carries when exact-tape env vars are unset, and relaxed trial residuals keep
  the VMEC-control loop as the production baseline unless
  ``VMEC_JAX_OPT_TRIAL_SCAN`` explicitly forces a scan profiling path.
- Exact optimizer tests should preserve final-output correctness: when trial
  solves and exact accepted-point replays disagree, histories and saved outputs
  must select the best finite exact accepted point rather than an unreplayed
  trial point.
- Full optimization sweeps are not required PR tests; they remain generated
  benchmark artifacts documented in :doc:`optimization_sweep_results`.

QI seed-robustness gates:

- Required PR tests protect the metric semantics, not global optimizer
  robustness.  The cheap QI gates use synthetic Boozer spectra, mocked state
  diagnostics, and one bundled solved-state QI seed to ensure smooth QI,
  legacy branch/shuffle QI, mirror ratio, elongation, optional ``LgradB``, and
  summary metadata stay compatible.
- The solved-state diagnostic fixture uses ``qi_diagnostics_from_state`` on
  ``input.QI_stel_seed_3127``/``wout_QI_stel_seed_3127.nc`` without launching
  a solve or optimization sweep.  Higher-resolution comparison of smooth QI,
  legacy QI, mirror ratio, elongation, iota, aspect ratio, and Boozer ``|B|``
  contour quality remains part of manual/nightly seed-robustness validation.
- Use ``tools/diagnostics/qi/audit_qi_seed_suitability.py --quick`` as the
  no-optimization preflight before a multi-seed QI sweep.  It ranks existing
  solved seeds through the public ``rank_qi_seed_records`` helper and records
  missing optional reference checkouts instead of making the default gate
  machine-specific.  The audit defaults to
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

Historical local optional evidence:

- ``outputs/rerun_20260525_123334`` passed the VMEC2000 stage-trace smoke
  matrix (``6`` cases, ``0`` failures) and the selected full QH warm-start row
  (``1`` case, ``0`` failures).  Treat this as dated release-candidate evidence,
  not as a permanent substitute for rerunning the matrix on the final commit.
- The same rerun refreshed CPU/GPU profiler reports for fixed-boundary,
  exact-callback, and QI Boozer/residual isolation.  Performance regressions
  should be diagnosed from those JSON reports before broadening the matrix.

The current detailed lane list and next parity gates are in
:doc:`optional_validation_plan`.


Release Checklist
-----------------

Use :doc:`release_checklist` as the command-level gate before tagging.  The
checklist keeps the release path tied to the same required lanes described
here: continuation correctness, exact accepted-point history/output selection,
VMEC residual parity, Boozer/LASYM input spectra, QI diagnostic metadata,
warning-clean docs, CI status, and artifact hygiene.


Coverage Plan to 85%, 90%, and 95%
----------------------------------

Coverage should rise because core physics code is directly tested, not because
tests execute long workflows incidentally.

The required CI gate is staged deliberately:

.. list-table::
   :header-rows: 1
   :widths: 14 28 38 20

   * - Stage
     - Gate
     - Required evidence before changing CI
     - CI action
   * - Completed
     - ``--cov-fail-under=85``
     - The earlier 85% gate was carried by required CI and superseded after
       local CI-equivalent coverage exceeded the next gate.
     - Superseded.
   * - Superseded
     - ``--cov-fail-under=90``
     - A local CI-equivalent run reaches at least 90% actual package coverage,
       keeps acceptable runtime, and the uncovered-line triage shows remaining
       gaps are mostly full/optional physics lanes rather than untested public
       fast workflows.
     - Superseded by the 95% gate.
   * - Current
     - ``--cov-fail-under=95``
     - Solver/driver seams have focused unit or bundled-parity tests, public
       workflow APIs have fast guards, and optional-dependency coverage is not
       needed to pass the gate.
     - Keep enforced.

Do not use optional local state as ratchet evidence.  A developer machine with
SIMSOPT, VMEC2000, fetched large assets, or a GPU may produce useful appended
coverage, but that coverage does not justify raising the required PR gate unless
the same lines are covered by required fast or bundled-parity tests.

1. Split monolithic solver sections into small internal functions only where it
   improves testability without changing numerical behavior.
2. Add unit-level tests for every extracted physics kernel: geometry, Fourier
   synthesis, force assembly, residual norms, profile interpolation,
   preconditioner blocks, and wout serialization.
3. Add low-resolution parity fixtures for each physics class in the manifest.
   The fixtures should be small enough for required CI when bundled, or for the
   full tier after ``tools/fetch_assets.py`` when they require released assets.
4. Move large generated data and figures out of the tracked tree before raising
   coverage gates.  Coverage runs should not require downloading presentation
   artifacts.
5. Raise ``--cov-fail-under`` only after corresponding tests are merged and a
   clean local CI-equivalent coverage run proves the next gate is feasible.
   The current enforced ratchet is 95%; future increases should be tied to
   concrete solver/refactor coverage wins, not incidental optional coverage.


Repository Size Plan
--------------------

The current source tree is dominated by generated documentation figures and
reference outputs.  The target is to keep only source, small fixtures, and
reviewed README/docs-critical figures in git.

Actions:

- Keep generated optimization sweeps under ``examples/optimization/results/``
  ignored and out of the repository.
- Keep only README/docs-critical reviewed PNGs and compact summaries in
  ``docs/_static/figures``.  Keep exploratory sweeps, superseded publication
  panels, PDFs, historical atlases, and raw optimization outputs under
  ``examples/optimization/results`` or attach them to GitHub releases.
- Replace committed large ``wout`` references with small compressed fixtures
  or generated-on-demand assets where runtime permits.
- Keep the size audit in every release checklist and in CI with the documented
  threshold below.
- Avoid committing both PNG and PDF versions unless both are directly linked
  from docs.

Run the current audit with:

.. code-block:: bash

   python tools/diagnostics/repo_health/repo_size_audit.py --top 40 --max-total-mib 50 --max-file-mib 2

Required CI also runs the audit with a source-tree ceiling of
``50 MiB`` total and ``2 MiB`` per tracked file.  Increase those limits only
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
