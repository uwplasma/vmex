Optional validation plan
========================

This page records the optional validation lanes used to build confidence beyond
required PR CI.  These checks are concrete and reproducible, but they are not
required for ordinary pull requests because they depend on local VMEC2000,
SIMSOPT, optional seed repositories, or expensive optimization runs.

Current CI baseline
-------------------

The latest verified ``main`` CI run checked during this update was green:

- workflow: ``CI``
- status: ``success``
- head SHA: ``5ca8216699c766621a1fe30e47db9b68befd36c2``
- completed: ``2026-05-11T17:14:59Z``
- run: https://github.com/uwplasma/vmec_jax/actions/runs/25684339586

The required CI split remains:

- fast tests on Python 3.10, 3.11, and 3.12,
- Python 3.11 coverage at the current ``63%`` gate,
- bounded physics smoke,
- parity manifest dry-run smoke,
- wheel/sdist build plus fast docs,
- full guide docs.

The manual/nightly ``physics-full`` job remains outside ordinary PR execution.

Plan helper
-----------

The declarative helper is:

.. code-block:: bash

   python validation/qi_seed_robustness_plan.py \
     --output results/qi_seed_audit/validation_plan.json

It writes a JSON manifest with required lanes, optional lanes, family
representatives, acceptance criteria, and deferred validation lanes.  For a
reviewable text form:

.. code-block:: bash

   python validation/qi_seed_robustness_plan.py \
     --format markdown \
     --output results/qi_seed_audit/validation_plan.md

The helper does not run VMEC2000, SIMSOPT, or optimization.  It only records the
commands and gates that should be executed deliberately by a local or scheduled
validation lane.

Family-representative QI workflow
---------------------------------

The seed-robustness workflow starts with solved-state diagnostics rather than a
full optimizer sweep.  The family representatives are:

.. list-table::
   :header-rows: 1
   :widths: 14 34 18 34

   * - Family
     - Label
     - Required for probe
     - Source
   * - ``QI``
     - ``qi_nfp3_fixed_resolution``
     - yes
     - bundled ``examples/data`` input+wout
   * - ``QP``
     - ``qp_from_omnigenity_nfp2_qi``
     - optional
     - ``OMNIGENITY_OPTIMIZATION_ROOT`` checkout
   * - ``QH``
     - ``qh_nfp4_warm_start``
     - yes
     - bundled ``examples/data`` input+wout
   * - ``QA``
     - ``qa_landreman_paul_lowres``
     - yes
     - bundled ``examples/data`` input+wout
   * - ``simple``
     - ``simple_circular_tokamak``
     - yes
     - bundled ``examples/data`` input+wout

Run the no-optimization audit first:

.. code-block:: bash

   PYTHONPATH=. python examples/optimization/audit_qi_seed_suitability.py \
     --quick \
     --output results/qi_seed_audit/summary.json \
     --csv results/qi_seed_audit/summary.csv

Acceptance for this lane:

- QI, QH, QA, and simple rows are present from bundled assets.
- QP is included when ``OMNIGENITY_OPTIMIZATION_ROOT`` is available; otherwise
  it is recorded in ``skipped_defaults`` rather than failing.
- Each row records smooth QI, legacy QI, mirror ratio, elongation, aspect
  ratio, mean iota, failed constraints, and ranks.
- Smooth-QI rows use ``include_bounce_endpoints=True`` by default, matching the
  normalized level endpoints used by the legacy branch-shuffle diagnostic.
- No optimization is launched.

Required no-executable CI coverage reads bundled VMEC2000 ``wout`` files for
the QI, QH, QA, simple, and finite-beta representatives and checks final
residual, flux, energy, iota-profile, and non-axisymmetric geometry quantities.
There is still no checked-in QP ``input`` + ``wout`` fixture; QP remains an
optional ``OMNIGENITY_OPTIMIZATION_ROOT`` lane until a small fixture can be
added.

Before running even tiny optimizer probes, write a dry-run manifest:

.. code-block:: bash

   PYTHONPATH=. python examples/optimization/audit_qi_seed_suitability.py \
     --quick \
     --prefine-probes plan \
     --prefine-manifest results/qi_seed_audit/prefine_manifest.json

This manifest is the review artifact.  It lists selected rows, hard caps,
expected output files, exact commands, selected ``phimin`` values, endpoint
mode, and the repeated-stage prefine plan.  The default prefine plan is capped
at ``--prefine-stage-modes 1,1,2,2,3`` with per-stage and total ``nfev`` caps
recorded for each selected seed.  Only after review should a local operator use
``--prefine-probes run --prefine-reviewed``.

Planned and executed prefine manifests include deterministic compact summaries
with status counts, completed stage modes, best seed by final objective, best
objective improvement, failed and timed-out probes, objective-history
regressions when compact histories are present, acceptance status, and a
recommended next action.  These summaries are promotion gates for the next
probe only; longer QI sweeps still require final diagnostics and plot review.

Optional external lanes
-----------------------

SIMSOPT formula parity:

.. code-block:: bash

   RUN_SIMSOPT_VALIDATION=1 pytest -q tests/test_simsopt_optional_validation.py

VMEC2000 executable smoke:

.. code-block:: bash

   VMEC2000_EXEC=/path/to/xvmec2000 \
   VMEC2000_INTEGRATION=1 \
   pytest -q tests/test_vmec2000_exec_fast_validation.py

Full VMEC2000 marker tier:

.. code-block:: bash

   VMEC2000_EXEC=/path/to/xvmec2000 \
   VMEC2000_INTEGRATION=1 \
   pytest -q -m vmec2000

These lanes must remain optional.  They skip or stay unselected unless the
operator installs the external dependency and exports the required environment
variables.

Next parity gates
-----------------

The next parity gates are:

- Add one small solved-state QI fixture around ``qi_diagnostics_from_state``
  before making optimizer seed-robustness claims.
- Run reviewed repeated-stage family-prefine probes across QI, QP, QH, QA, and
  simple seeds.
- Keep VMEC2000 executable smoke green before broadening the executable-backed
  manifest matrix.
- Keep SIMSOPT formula-level comparisons green where SIMSOPT is installed.

Deferred validation lanes
-------------------------

The deferred lanes are intentionally not required CI:

- Full multi-seed constrained-QI optimization sweep with visual Boozer ``|B|``
  contour audit.
- Full VMEC2000 parity manifest against a local executable and fetched large
  assets.
- SIMSOPT finite-difference optimization comparison beyond formula-level
  residual parity.
- GPU-specific QI prefine and optimization robustness matrix.
