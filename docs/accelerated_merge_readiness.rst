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

The latest serial CPU reassessment artifact is:

- ``outputs/accelerated_fixed_boundary_reassessment_20260309/summary.json``
- ``examples/fixed_boundary_driver_tracks.py`` for live parity-vs-optimized
  comparisons on the current branch
- ``outputs/accelerated_cli_fixed_boundary_no_n3are_20260310/summary.json``
  for the latest bundled fixed-boundary CLI-style sweep

Key results from that artifact:

- ``input.LandremanSenguptaPlunk_section5p3_low_res``:
  ``45.48s`` current default vs ``0.198s`` accelerated single-grid and
  ``0.232s`` accelerated explicit multigrid,
- ``input.LandremanPaul2021_QA_lowres``:
  ``8.18s`` current default vs ``7.31s`` accelerated single-grid and
  ``8.10s`` accelerated explicit multigrid,
- ``input.n3are_R7.75B5.7_lowres``:
  ``1.25s`` accelerated single-grid with final ``fsq_total ~ 1.1e-4`` in the
  plain accelerated API path before the newer CLI staged-followup controller.

Current CLI behavior is better captured as policy than as one stale table:

- easy fixed-boundary inputs remain on the fast single-grid route,
- explicit staged inputs now retry the input-defined stage schedule before the
  strict finisher starts,
- the bundled driver example already confirms the intended easy-case behavior
  on ``input.circular_tokamak``:
  parity ``28.863s`` vs optimized CLI-style ``3.445s``, both converged at
  ``fsq_total ~ 2e-14``.
- the new 15-case bundled fixed-boundary CLI-style reassessment shows
  11 of 15 cases faster on the optimized path, with the strongest wins on
  ``LandremanSenguptaPlunk_section5p3_low_res`` (``249.49x``),
  ``basic_non_stellsym_pressure`` (``12.47x``), and
  ``ITERModel`` (``1.78x``),
- that same reassessment also surfaces the current blockers clearly:
  ``li383_low_res`` (``0.0036x``),
  ``up_down_asymmetric_tokamak`` (``0.0225x``),
  ``LandremanPaul2021_QA_lowres1`` (``0.93x``), and
  ``solovev`` (``0.94x``) are all slower on the optimized CLI-style path.

These numbers justify the current controller split on the branch:
keep the optimized fixed-boundary logic available for explicit testing and
review, but do not promote it to the repo-wide default until the slow outliers
are closed.

The bundled ``n3are`` example now includes an explicit
``NITER_ARRAY = 1000 1000 5000``. The conservative staged CLI fallback remains
important for the generic ``NS_ARRAY`` without ``NITER_ARRAY`` class, but that
policy is no longer represented by the checked-in ``n3are`` input itself.

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
- the remaining staged hard-case limitation is explicitly documented if the
  branch is merged before every staged hard case is demonstrated at ``FTOL``.

Current PR summary
------------------

The branch is ready for an honest review PR, but not for a default flip.

- Positive signal:
  the optimized CLI-style controller is materially better on most of the
  bundled fixed-boundary matrix and keeps convergence on the successful cases.
- Blocking signal:
  the March 10 serial sweep still has severe slowdowns on
  ``li383_low_res`` and ``up_down_asymmetric_tokamak``, plus smaller slowdowns
  on ``LandremanPaul2021_QA_lowres1`` and ``solovev``.
- Hard-outlier signal:
  ``n3are_R7.75B5.7_lowres`` remains too expensive; a same-branch cold
  ``solver_mode="default"`` run took ``41.67s`` and stopped at
  ``fsq_total ~ 6.90e-2``, while the optimized CLI-style run exceeded 15
  minutes without finishing the cold solve during reassessment.

Conclusion:

- mergeable as an experimental branch if reviewers want the tooling and the
  partial wins on `main`,
- not ready to become the default controller,
- not ready to market as a universal fixed-boundary runtime win yet.

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
        --ids n3are_R7.75B5.7_lowres \
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

- full bundled example runtime and memory matrix on CPU and GPU,
- full final-``wout`` accuracy matrix against VMEC2000 at the accelerated-mode
  target,
- accelerated free-boundary redesign and validation,
- gradient checks on representative accelerated fixed-boundary and
  free-boundary workflows,
- policy hardening for unseen inputs beyond the current representative set.
