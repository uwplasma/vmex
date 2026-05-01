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
- Required coverage: 95% line coverage for ``vmec_jax`` package code.
- Required local command: ``pytest -q -m "not full and not vmec2000"`` should
  be fast enough for routine development.
- Nightly/manual coverage: larger VMEC2000, GPU, and full-resolution physics
  checks run outside the required PR gate.
- Repository checkout size: keep the tracked source tree small enough that a
  fresh clone and ``pip install .`` are not dominated by generated figures,
  optimization outputs, or bulky reference ``wout`` files.


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
- Full optimization sweeps are not required PR tests; they remain generated
  benchmark artifacts documented in :doc:`optimization_sweep_results`.


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
   merged.  The current required fast-suite gate is ``58%``; the next planned
   ratchets are 60%, 70%, 80%, 90%, then 95%.


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
- Add docstrings to public and semi-public functions that state conventions:
  radial mesh, VMEC vs physical toroidal angle, Fourier signs, symmetry
  assumptions, and units.
- Add comments near non-obvious VMEC2000 compatibility choices, especially
  sign conventions, mode indexing, fallback convergence logic, and scan
  controller decisions.
- Keep JAX import and backend selection lazy enough that users can choose CPU or
  GPU before importing heavy runtime code.
