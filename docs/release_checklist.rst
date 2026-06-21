Release checklist
=================

This checklist is the short path from a clean development branch to a release
candidate.  It is intentionally command-driven so release hygiene stays
repeatable and independent of local notebooks or generated optimization
artifacts.

Required pre-push gate
----------------------

Run the opt-in local CI gate before pushing a release-candidate commit:

.. code-block:: bash

   python tools/diagnostics/local_ci_gate.py

This mirrors the hosted CI lanes that should be safe on a normal developer
machine: CLI smoke, compile check, repository size audit, fast pytest with the
current ``95%`` coverage fail-under, selected bounded physics smoke, wheel/sdist
build, fast docs, and full docs.  It is deliberately not a Git hook; use
``--dry-run``, ``--list``, ``--only STAGE``, and ``--skip STAGE`` when you need
to inspect or narrow the gate locally.

The authoritative expanded command form is printed by:

.. code-block:: bash

   python tools/diagnostics/local_ci_gate.py --dry-run

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
``95%``.

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
- The ``Coverage Gate (py3.11 combined)`` job succeeds and reports at least
  95% combined coverage.
- Manual/nightly physics jobs may be skipped, but must not fail.

During release preparation, record the latest completed green ``main`` CI
baseline for the exact release-candidate commit.  Do not reuse historical run
IDs from earlier audits after pushing coverage, docs, or validation changes.
Re-check GitHub Actions before cutting any release candidate.

The current CI workflow also includes a dedicated full-docs job.  Treat both
``Build (wheel/sdist + docs)`` and ``Docs (full guide)`` as release blockers.

Artifact hygiene gate
---------------------

Before tagging, keep the repository free of transient outputs:

.. code-block:: bash

   git status --short
   rm -rf build dist vmec_jax.egg-info
   python tools/diagnostics/repo_size_audit.py --top 40 --max-total-mib 50 --max-file-mib 2
   python tools/diagnostics/repo_size_audit.py --top 40 --include-ignored
   git check-ignore -v docs/_build/html/index.html docs/api/generated/vmec_jax.solve.rst .DS_Store

Do not commit optimization result trees, rerun ``wout`` files, profiler traces,
or generated PDFs unless a small artifact is explicitly referenced by README or
docs and has a documented validation purpose.
The ``--include-ignored`` report is informational: it quantifies local ignored
Sphinx builds, generated WOUT/BOOZ/mgrid files, fetched assets, and result
trees without changing the tracked-size release gate.

For one-case optimization showcase smoke checks, use the bounded renderer
instead of refreshing the full production matrix:

.. code-block:: bash

   PYTHONPATH=. python examples/optimization/render_minimal_seed_showcase.py \
     --output-root /path/to/minimal_seed_showcase \
     --figure-dir /path/to/figures \
     --cases qi_nfp1 --skip-missing

Use the production ``--publication-matrix`` render only after the full
showcase matrix has been regenerated.

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
   - For direct-coil free-boundary optimization, state whether gradients are
     fixed accepted-branch, same-branch complete-solve validated, or a promoted
     adaptive full-loop derivative. Do not imply full adaptive coil-to-Boozer
     differentiation unless the release candidate includes a fingerprint-gated
     full adaptive AD-vs-central-FD gate through the adaptive loop.
   EOF
   gh release create vX.Y.Z --repo uwplasma/vmec_jax --title "vmec-jax vX.Y.Z" --notes-file /tmp/vmec_jax_release_notes.md

For GitHub releases, ``publish-pypi.yml`` validates that the release tag
matches ``project.version`` in ``pyproject.toml`` after stripping an optional
leading ``v``.  Do not publish a tag unless ``pyproject.toml`` has the same
version and the CI gates above are green.

Current repository release note
-------------------------------

The latest repository tag prepared from this checklist is
`v0.0.15 <https://github.com/uwplasma/vmec_jax/releases/tag/v0.0.15>`_, built
from the matching release tag after the required CI gates passed.  Verify PyPI
with a no-dependencies wheel download after each publication.
Package-index descriptions are immutable for already-published files, so any
README or installation wording merged after the latest public tag will not
appear on PyPI until the next release upload.
Do not describe the repository tag as the latest PyPI or conda-forge package
unless the corresponding package index has been checked.

The release notes should list user-visible changes, validation coverage, known
limitations, and any optional external validation that was not run. For
direct-coil free-boundary milestones, the limitations must explicitly separate
JAX-native direct-coil fields, fixed accepted-branch replay/custom-VJP gates,
same-branch complete-solve finite-difference evidence, and any still-unclaimed
adaptive host-controller or Boozer/QS full-loop derivative.
The next release candidate should cite the latest completed green CI run after
re-checking the newest ``main`` workflow and the latest passing local coverage
result from the release-candidate commit at the enforced ``95%`` coverage gate.
After the release is public, verify the package index and installation state:

- PyPI should show the new release and ``pip install vmec-jax==X.Y.Z`` should
  import the same version from a clean virtual environment.
- Conda-forge may lag PyPI, and PyPI may lag repository tags; README wording
  should not imply any package index is already synchronized.
- Release notes should use post-publication wording once the tag is published,
  not "must be green before this release is published" phrasing.

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

Historical local release-hygiene snapshot
-----------------------------------------

The latest documented local rerun evidence for the post-``v0.0.14`` candidate
is the June 3 release-hygiene/refactor refresh through commit ``8c7ffcd``.
It is not a release tag by itself, but it is the current evidence bundle to
cite until a newer release-candidate run replaces it:

- Local required coverage gate:
  ``JAX_ENABLE_X64=1 pytest -q -n 4 -m 'not full and not vmec2000 and not simsopt'
  --cov=vmec_jax --cov-report=xml --cov-report=term:skip-covered
  --cov-fail-under=95`` passed with ``2423 passed, 20 skipped, 1 xfailed`` and
  ``95.16%`` coverage at the enforced ``95%`` gate.
- Focused solve/refactor checks passed:
  ``ruff check vmec_jax/solve.py tests/test_solve_dump_helpers.py``,
  ``JAX_ENABLE_X64=1 pytest -q tests/test_solve_dump_helpers.py``, and
  ``JAX_ENABLE_X64=1 pytest -q tests/test_solve_dump_helpers.py
  tests/test_solve_additional_helpers.py tests/test_solve_runtime.py``.
- GitHub Actions run ``26885868424`` passed build, full docs, CLI smoke,
  parity dry-run, physics smoke, and Python 3.10/3.11/3.12 fast-test lanes for
  ``8c7ffcd``; the manual/nightly physics lane was skipped as expected.

Before tagging, replace this snapshot with fresh local/CI evidence from the
exact release-candidate commit if any additional source, test, or docs changes
land after ``8c7ffcd``.
