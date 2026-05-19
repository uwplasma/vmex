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
   pytest -q tests/test_residue_getfsq_parity.py tests/test_wout_profiles_currents_bundled_parity.py tests/test_converged_wout_matrix_parity.py tests/test_vmec2000_exec_threed1.py
   pytest -q tests/test_booz_input.py tests/test_quasi_isodynamic.py tests/test_qi_legacy.py tests/test_qi_diagnostics.py tests/test_qi_objective_component_report.py tests/test_qi_seed_suitability_audit.py tests/test_qs_ess_render_smoke.py
   pytest -q tests/test_quasisymmetry.py tests/test_optimization_examples.py tests/test_implicit_helpers.py tests/test_wout_additional_helpers.py tests/test_solve_additional_helpers.py tests/test_free_boundary_additional_helpers.py tests/test_vmec_kernel_additional_helpers.py tests/test_solve_branch_coverage.py tests/test_implicit_wout_driver_branch_coverage.py
   JAX_ENABLE_X64=1 pytest -q -m "not full and not vmec2000 and not simsopt" --cov=vmec_jax --cov-report=xml --cov-report=term-missing:skip-covered --cov-fail-under=85
   python -m build
   python -m sphinx -W -b html docs docs/_build/html_release

These tests cover the required local lanes: continuation semantics, exact
accepted-point history/output selection, no-executable VMEC residual parity,
Boozer input spectra including ``lasym=True`` channels, QI diagnostic metadata
including the bundled solved-state QI seed gate and public
``rank_qi_seed_records`` export, QI example objective assembly including the
optional ``BoozerBTarget`` homotopy term, solve-free JVP/VJP routing checks,
pure driver/runtime policy helpers, implicit/wout serialization helpers,
free-boundary/solver helper branches, VMEC-kernel helper branches, packaging
hygiene, additional solve/implicit/wout/driver branch coverage, the required
Python 3.11 coverage gate, and warning-clean documentation.  Record the exact
pass/skip/deselect count and coverage percentage from the release-candidate
commit in the release notes.  The enforced local and CI coverage gate is
``85%``.

Before cutting a new release, bump ``project.version`` in ``pyproject.toml``
and choose the matching tag name:

.. code-block:: bash

   python - <<'PY'
   import tomllib
   version = tomllib.load(open("pyproject.toml", "rb"))["project"]["version"]
   print(f"current project.version = {version}")
   PY
   git tag --list 'v*' | tail -20
   git ls-remote --tags origin 'v*' | tail -20

Do not reuse an existing local or remote tag.  The release scope should be
summarized in the release notes before tagging.

Required GitHub Actions gate
----------------------------

After pushing, verify the newest workflow run on ``main``:

.. code-block:: bash

   gh run list --repo uwplasma/vmec_jax --branch main --workflow CI --limit 5
   gh run view RUN_ID --repo uwplasma/vmec_jax --json status,conclusion,jobs

The required release baseline is:

- Fast tests pass on all configured Python versions.
- The wheel/sdist build succeeds.
- Full docs build with warnings as errors.
- The parity-manifest smoke job succeeds.
- The bounded physics smoke job succeeds.
- Manual/nightly physics jobs may be skipped, but must not fail.

The current CI workflow also includes a dedicated full-docs job.  Treat both
``Build (wheel/sdist + docs)`` and ``Docs (full guide)`` as release blockers.

Artifact hygiene gate
---------------------

Before tagging, keep the repository free of transient outputs:

.. code-block:: bash

   git status --short
   rm -rf build dist vmec_jax.egg-info
   python tools/diagnostics/repo_size_audit.py --top 40
   git check-ignore -v docs/_build/html/index.html docs/api/generated/vmec_jax.solve.rst .DS_Store

Do not commit optimization result trees, rerun ``wout`` files, profiler traces,
or generated PDFs unless a small artifact is explicitly referenced by README or
docs and has a documented validation purpose.

Release tagging gate
--------------------

Tag only after the local and GitHub gates are green:

.. code-block:: bash

   git tag -a vX.Y.Z -m "vmec-jax vX.Y.Z"
   git push origin vX.Y.Z
   cat > /tmp/vmec_jax_release_notes.md <<'EOF'
   User-visible changes:
   - ...

   Validation:
   - ...

   Known limitations:
   - ...
   EOF
   gh release create vX.Y.Z --repo uwplasma/vmec_jax --title "vmec-jax vX.Y.Z" --notes-file /tmp/vmec_jax_release_notes.md

For GitHub releases, ``publish-pypi.yml`` validates that the release tag
matches ``project.version`` in ``pyproject.toml`` after stripping an optional
leading ``v``.  Do not publish a tag unless ``pyproject.toml`` has the same
version and the CI gates above are green.

Current release note
--------------------

The latest prepared release from this checklist is
`v0.0.11 <https://github.com/uwplasma/vmec_jax/releases/tag/v0.0.11>`_, built
from the matching release tag after the required CI gates pass.  Verify PyPI
with a no-dependencies wheel download after publication.

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
- Do not claim broad strict external LASYM parity until the nightly
  executable-backed ``up_down_asymmetric_tokamak`` residual gap is resolved.
  ``basic_non_stellsym_pressure`` is the promoted finite-beta LASYM executable
  check.
- SIMSOPT comparison scripts for optimization objective and derivative parity.
- CPU/GPU profiling sweeps for any accepted-point replay, scan, or device
  default change; verify final artifacts still select the best finite exact
  accepted point when trial and exact residual histories disagree.
- Seed-robust QI probes from QI, QP, QH, QA, and non-omnigenous seeds before
  advertising global QI robustness.  Far-seed probes should include the
  optional same-NFP ``BoozerBTarget`` homotopy lane when a solved QI reference
  wout is available.  Treat trial-solve landscape/basin scans as triage only;
  use exact-solve diagnostics before promotion claims.
