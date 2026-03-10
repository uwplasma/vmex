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
- a CLI-only fixed-boundary follow-up policy for staged inputs without
  ``NITER_ARRAY``: budgeted warm-start multigrid plus short parity polish.

Representative fixed-boundary reassessment
------------------------------------------

The latest serial CPU reassessment artifact is:

- ``outputs/accelerated_fixed_boundary_reassessment_20260309/summary.json``
- ``outputs/accelerated_cli_fixed_boundary_reassessment_20260309/summary.json``
- ``outputs/accelerated_cli_fixed_boundary_hybrid_20260309/summary.json``

Key results from that artifact:

- ``input.LandremanSenguptaPlunk_section5p3_low_res``:
  ``45.48s`` current default vs ``0.198s`` accelerated single-grid and
  ``0.232s`` accelerated explicit multigrid,
- ``input.LandremanPaul2021_QA_lowres``:
  ``8.18s`` current default vs ``7.31s`` accelerated single-grid and
  ``8.10s`` accelerated explicit multigrid,
- ``input.n3are_R7.75B5.7_lowres``:
  ``1.25s`` accelerated single-grid with final
  ``fsq_total ~ 1.1e-4`` in the plain accelerated API path, versus
  ``16.4s`` / ``fsq_total ~ 6.8e-6`` for the new CLI-only staged follow-up.

Latest staged-hybrid follow-up:

- the staged CLI accelerated controller now uses accelerated coarse stages,
  a strict parity final stage with the full user ``NITER`` budget, and a
  state-only strict finisher that keeps the best continuation block,
- on ``input.n3are_R7.75B5.7_lowres`` that moved the best measured residual to
  ``fsq_total ~ 1.61e-6``,
- the same artifact confirms the easy accelerated CLI cases remain closed:
  ``LandremanSenguptaPlunk_section5p3_low_res`` at ``~3.0e-14``,
  ``LandremanPaul2021_QA_lowres`` at ``~3.0e-13``,
  and ``li383_low_res`` at ``~1.24e-14``.

These numbers justify the current fixed-boundary accelerated default:
avoid staged VMEC-style multigrid unless the user explicitly asks for it in the
API, but allow the CLI to use a more robust staged fallback on difficult inputs
that lack explicit stage budgets.

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
  branch is merged before ``n3are``-class inputs can be driven to ``FTOL``.

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
        --jax-platforms cpu

      python tools/diagnostics/benchmark_accelerated_mode.py \
        --ids LandremanPaul2021_QA_lowres \
        --kind fixed --baseline-mode default --candidate-mode accelerated \
        --jax-platforms cpu

      python tools/diagnostics/benchmark_accelerated_mode.py \
        --ids n3are_R7.75B5.7_lowres \
        --kind fixed --baseline-mode default --candidate-mode accelerated \
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
