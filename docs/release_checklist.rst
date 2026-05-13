Release checklist
=================

This checklist is the short path from a clean development branch to a release
candidate.  It is intentionally command-driven so release hygiene stays
repeatable and independent of local notebooks or generated optimization
artifacts.

Required pre-push gate
----------------------

Run these checks before pushing a release-candidate commit:

.. code-block:: bash

   git status --short --branch
   ruff check vmec_jax tests examples tools validation
   python -m compileall -q vmec_jax examples tests tools validation
   pytest -q tests/test_optimization_helpers.py tests/test_continuation_exact_history.py
   pytest -q tests/test_residue_getfsq_parity.py tests/test_wout_profiles_currents_bundled_parity.py tests/test_vmec2000_exec_threed1.py
   pytest -q tests/test_booz_input.py tests/test_quasi_isodynamic.py tests/test_qi_legacy.py tests/test_qi_diagnostics.py tests/test_qi_objective_component_report.py
   JAX_ENABLE_X64=1 pytest -q -m "not full and not vmec2000" --cov=vmec_jax --cov-report=xml --cov-report=term-missing:skip-covered --cov-fail-under=63
   python -m sphinx -W -b html docs docs/_build/html_release

These tests cover the required local lanes: continuation semantics, exact
accepted-point history/output selection, no-executable VMEC residual parity,
Boozer input spectra including ``lasym=True`` channels, QI diagnostic metadata,
the required Python 3.11 coverage gate, and warning-clean documentation.  The
latest local CI-equivalent coverage baseline on 2026-05-13 is
``643 passed, 21 skipped, 85 deselected`` with ``66.98%`` coverage in ``7:50``;
the enforced gate remains ``63%`` until the next planned ratchet.

Required GitHub Actions gate
----------------------------

After pushing, verify the newest workflow run on ``main``:

.. code-block:: bash

   gh run list --repo uwplasma/vmec_jax --branch main --limit 5
   gh run view RUN_ID --repo uwplasma/vmec_jax --json status,conclusion,jobs

The required release baseline is:

- Fast tests pass on all configured Python versions.
- The wheel/sdist build succeeds.
- Full docs build with warnings as errors.
- The parity-manifest smoke job succeeds.
- The bounded physics smoke job succeeds.
- Manual/nightly physics jobs may be skipped, but must not fail.

Artifact hygiene gate
---------------------

Before tagging, keep the repository free of transient outputs:

.. code-block:: bash

   git status --short
   python tools/diagnostics/repo_size_audit.py --top 40

Do not commit optimization result trees, rerun ``wout`` files, profiler traces,
or generated PDFs unless a small artifact is explicitly referenced by README or
docs and has a documented validation purpose.

Release tagging gate
--------------------

Tag only after the local and GitHub gates are green:

.. code-block:: bash

   git tag -a vX.Y.Z -m "vmec-jax vX.Y.Z"
   git push origin vX.Y.Z
   gh release create vX.Y.Z --repo uwplasma/vmec_jax --title "vmec-jax vX.Y.Z" --notes-file RELEASE_NOTES.md

The release notes should list user-visible changes, validation coverage, known
limitations, and any optional external validation that was not run.

Optional research-grade gates
-----------------------------

These checks are not required for every PR, but they are expected before
claiming a broader physics milestone:

- ``RUN_FULL=1 pytest tests/test_wout_comprehensive_parity.py -v`` for bundled
  fixed-boundary, finite-beta, and ``lasym`` references.
- VMEC2000 executable parity with ``~/bin/xvmec2000`` for newly added input
  decks or convergence-policy changes.
- SIMSOPT comparison scripts for optimization objective and derivative parity.
- CPU/GPU profiling sweeps for any accepted-point replay, scan, or device
  default change.
- Seed-robust QI probes from QI, QP, QH, QA, and non-omnigenous seeds before
  advertising global QI robustness.
