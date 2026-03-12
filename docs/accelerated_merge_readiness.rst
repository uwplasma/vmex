Accelerated Mode Merge Readiness
================================

This page tracks whether branch ``codex/nonparity-performance`` is ready to
merge into ``main`` as an **experimental** feature.

The important distinction is:

- **mergeable to main**: the accelerated mode is isolated, documented, tested,
  and useful behind an explicit opt-in API,
- **ready to become default**: the accelerated mode has passed the broader
  fixed-boundary and free-boundary acceptance matrix and can replace the
  existing default controller.

Current recommendation
----------------------

Current recommendation: **merge as experimental, do not make default**.

Rationale:

- the public API split is explicit:
  ``run_fixed_boundary(..., solver_mode="accelerated")`` and
  ``vmec_jax input.name --solver-mode accelerated``,
- the parity/default path remains available and unchanged for ordinary users,
- local validation is green on the branch,
- representative fixed-boundary benchmarks now show clear wins from the
  accelerated controller's new single-grid default,
- accelerated free-boundary is still intentionally conservative and is not yet
  a new fast controller, so the branch does not overclaim readiness.

What this branch adds
---------------------

- explicit ``default`` / ``parity`` / ``accelerated`` solver policies,
- ftol-derived accelerated convergence targets instead of fixed absolute
  stopping literals,
- compact accelerated histories and resume payloads,
- a bundled accelerated-mode benchmark harness,
- an accelerated fixed-boundary controller that now defaults to a single
  final-grid solve unless the caller explicitly requests multigrid,
- a CLI-only fixed-boundary follow-up stack:

  - explicit staged inputs (``NS_ARRAY`` + ``NITER_ARRAY``) can replay their
    input-defined schedule after a missed single-grid fast pass,
  - staged inputs without ``NITER_ARRAY`` still have the reduced-budget
    multigrid fallback when accelerated mode is explicitly requested,
  - strict parity finish blocks continue from state only,
- a bundled Python example that compares the parity and optimized CLI-style
  driver tracks directly.

Representative fixed-boundary reassessment
------------------------------------------

The current fixed-boundary review set is:

- ``examples/fixed_boundary_driver_tracks.py`` for live parity-vs-optimized
  comparisons on the current branch,
- the warmed bundled fixed-boundary CPU sweep used for the README benchmark,
- the same-host CPU/GPU warmed sweep on the updated 16-case bundled matrix.

Key current results:

- the latest warmed fixed-boundary reassessment on 2026-03-11 shows the
  optimized CLI path converging on all 16 bundled cases,
- that same matrix now shows 13 cases faster than the current branch baseline
  and 3 roughly neutral, with no bundled CPU regressions left,
- targeted non-axisymmetric fixes materially improved final ``wout`` quality on
  the QA/QH reactor-scale cases, while also removing the earlier runtime
  regressions on the bundled CPU sweep.

Current CLI behavior is better captured as policy than as one stale table:

- easy fixed-boundary inputs remain on the fast single-grid route,
- explicit staged inputs now retry the input-defined stage schedule before the
  strict finisher starts,
- the bundled driver example already confirms the intended easy-case behavior
  on ``input.circular_tokamak``:
  parity ``28.863s`` vs optimized CLI-style ``3.445s``, both converged at
  ``fsq_total ~ 2e-14``.
- the freshest full warmed fixed-boundary CPU matrix is now favorable:
  all 16 cases converge, 13 of 16 improve on the current default path, and the
  remaining 3 are effectively neutral,
- same-host CPU/GPU benchmarking still confirms the GPU path can help on the
  larger 3D bundled cases, but backend choice still matters and remains a
  separate question from the fixed-boundary CPU controller decision.

These numbers justify the current controller split on the branch:
keep the optimized fixed-boundary logic available for explicit testing and
review, and promote it cautiously only after backend selection becomes
automatic on mixed CPU/GPU workstations.

Merge checklist
---------------

This branch is ready for a draft or review PR when all of the following are
true:

- ``pytest -q`` passes on the branch,
- docs build passes,
- accelerated-mode docs explain scope and limitations clearly,
- default/parity behavior remains available and tested,
- the branch includes at least one benchmark artifact demonstrating the
  accelerated fixed-boundary controller is useful on representative cases,
- no user-set environment variable is required for accelerated fixed-boundary
  correctness on the benchmarked bundled cases,
- the current GPU backend-selection limitation is explicitly documented if the
  branch is merged before automatic device choice lands.

Current PR summary
------------------

The branch is ready for an honest review PR.

- Positive signal:
  the optimized CLI-style controller now keeps convergence and removes the
  runtime-regression blocker on the full bundled fixed-boundary CPU matrix.
- Remaining caution:
  final-state quality versus VMEC2000 is still a separate topic from the
  runtime comparison, and GPU/default-library policy should not be flipped just
  because the fixed-boundary CPU CLI story improved.
  The latest useful quality fix was in ``wout`` export for ``lasym=False`` 3D
  cases, which removed symmetry-forbidden ``rmns/zmnc`` output and cut the
  bundled QA/QH quality metric by about an order of magnitude. A follow-up
  staged-controller fixes then brought the reactor-scale QA/QH cases into the
  ``1e-4`` to ``1e-3`` range and QA-lowres to about ``4e-3``. The last
  branch-specific ``basic_non_stellsym_pressure`` regression was removed by
  keeping ``lasym=True`` current-driven 3D staged runs fully on the
  conservative controller, which restores baseline-level quality there
  (about ``2.98e-02`` max relRMS) at roughly neutral runtime.

Conclusion:

- mergeable as an experimental branch if reviewers want the tooling and the
  fixed-boundary wins on ``main``,
- plausible to promote as the default **CLI fixed-boundary CPU** controller on
  ``main`` if reviewers accept the non-parity scope,
- not yet ready to become the universal default across GPU and non-CLI
  workflows.

Recommended reviewer checklist
------------------------------

1. Verify the API split and docs:

   .. code-block:: bash

      git diff main...HEAD -- vmec_jax/driver.py vmec_jax/cli.py docs/performance.rst

2. Re-run the main validation gates:

   .. code-block:: bash

      pytest -q
      SPHINX_FAST=1 LC_ALL=C LANG=C python -m sphinx -W -j auto -b html docs docs/_build/html_fastcheck

3. Re-run representative accelerated fixed-boundary benchmarks serially:

   .. code-block:: bash

      python tools/diagnostics/benchmark_accelerated_mode.py \
        --ids LandremanSenguptaPlunk_section5p3_low_res \
        --kind fixed --baseline-mode default --candidate-mode accelerated \
        --candidate-cli-fixed-boundary-mode \
        --jax-platforms cpu

      python tools/diagnostics/benchmark_accelerated_mode.py \
        --ids LandremanPaul2021_QA_lowres \
        --kind fixed --baseline-mode default --candidate-mode accelerated \
        --candidate-cli-fixed-boundary-mode \
        --jax-platforms cpu

      python tools/diagnostics/benchmark_accelerated_mode.py \
        --ids LandremanPaul2021_QA_reactorScale_lowres \
        --kind fixed --baseline-mode default --candidate-mode accelerated \
        --candidate-cli-fixed-boundary-mode \
        --jax-platforms cpu

4. Confirm the merge scope is still experimental:

- do not switch the repo-wide default to ``solver_mode="accelerated"``,
- do not advertise accelerated free-boundary as finished,
- do not remove or weaken parity-mode coverage.

Not yet ready for default
-------------------------

The branch should **not** make accelerated mode the default controller yet.

The remaining gates are broader than this PR:

- exact per-channel final-stage ``FTOL`` is now enforced on the accelerated
  fixed-boundary return path, but that still needs to be revalidated on the
  full bundled fixed-boundary matrix after any further controller changes,
- full bundled example runtime and memory matrix on CPU and GPU,
- full final-``wout`` accuracy matrix against VMEC2000 at the accelerated-mode
  target,
- accelerated free-boundary redesign and validation,
- gradient checks on representative accelerated fixed-boundary and
  free-boundary workflows,
- policy hardening for unseen inputs beyond the current representative set.
