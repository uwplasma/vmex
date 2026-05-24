Optional validation plan
========================

This page records the optional validation lanes used to build confidence beyond
required PR CI.  These checks are concrete and reproducible, but they are not
required for ordinary pull requests because they depend on local VMEC2000,
SIMSOPT, optional seed repositories, or expensive optimization runs.

Current CI verification
-----------------------

Do not treat a checked-in SHA or workflow URL as the current release baseline.
Before using this plan for release validation, verify the newest ``main`` CI run
directly:

.. code-block:: bash

   gh run list --repo uwplasma/vmec_jax --branch main --workflow CI --limit 5
   gh run view RUN_ID --repo uwplasma/vmec_jax --json status,conclusion,jobs

Record the verified run ID, head SHA, conclusion, and completion time in the
release notes for the candidate commit.

The required CI split remains:

- fast tests on Python 3.10, 3.11, and 3.12,
- Python 3.11 coverage at the current ``95%`` gate,
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

The helper does not run VMEC2000, SIMSOPT, optimization, or GitHub API
queries.  Its default CI baseline is an explicit ``unverified`` placeholder;
pass the verified run metadata with ``--ci-status``, ``--ci-head-sha``,
``--ci-url``, and ``--ci-completed-at-utc`` only after checking the current
``main`` workflow.  The manifest records commands and gates that should be
executed deliberately by a local or scheduled validation lane.

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
recorded for each selected seed.  Constrained mirror cleanup uses
``--prefine-mirror-surface-index all`` by default so the acceptance gate cannot
pass by improving only one Boozer surface.  Only after review should a local
operator use ``--prefine-probes run --prefine-reviewed``.

Planned and executed prefine manifests include deterministic compact summaries
with status counts, completed stage modes, best seed by final objective, best
objective improvement, failed and timed-out probes, objective-history
regressions when compact histories are present, acceptance status, and a
recommended next action.  These summaries are promotion gates for the next
probe only; longer QI sweeps still require final diagnostics and plot review.
Completed monotone probes with final QI objective at or below ``5e-2`` are
accepted as stable low-objective seeds even if the tiny smoke budget leaves
them unchanged.
Continuation probes carry the optimized VMEC input between stages.  In
particular, a mode-1 projection that zeros higher boundary modes remains the
seed for the next mode-2 stage unless the stage itself reintroduces those modes
as active zero-increment degrees of freedom.
Accepted optimization histories are exact-replay histories, not raw trial-solve
histories.  If the relaxed trial solve accepts a point that replays worse under
the exact Jacobian path, the optimizer retains the best exact point as the
final output and increments ``rejected_trial_exact_history_count``.

Optional external lanes
-----------------------

External VMEC asset intake
~~~~~~~~~~~~~~~~~~~~~~~~~~

Candidate SIMSOPT and Landreman ``vmec_equilibria`` decks are tracked in
``validation/external_vmec_asset_manifest.toml``.  This manifest is
metadata-only: it records pinned source URLs, visible license status,
physics-family tags, companion reference files, and whether an asset is a good
small fixture, an explicit fetched asset, or only a reference target.
Use the inventory helper to list or verify candidates without copying upstream
data into the repository:

.. code-block:: bash

   python tools/diagnostics/external_vmec_asset_inventory.py \
     --family stellarator --family fixed_boundary

   python tools/diagnostics/external_vmec_asset_inventory.py \
     --repository landreman_vmec_equilibria \
     --source-root landreman_vmec_equilibria=outputs/external_benchmark_sources/vmec_equilibria \
     --fail-missing

The current policy is:

- SIMSOPT assets are MIT licensed upstream.  Small text input decks may be
  bundled with attribution when they materially improve required CI; larger
  NetCDF references should stay fetched or optional.
- ``landreman/vmec_equilibria`` has no visible GitHub license metadata.  Keep
  those cases as explicit local-fetch/reference validation targets unless the
  license is clarified.
- Free-boundary Landreman inputs are useful targets, but several reference
  missing mgrid files in that repository.  Prefer the SIMSOPT W7-X generated
  mgrid workflow, or keep those Landreman rows as blocked optional lanes until
  the corresponding mgrid source is identified.

Representative candidates already recorded there include:

- SIMSOPT circular tokamak and ITERModel axisymmetric fixed-boundary decks.
- SIMSOPT ``basic_non_stellsym`` and
  ``LandremanSenguptaPlunk_section5p3`` for fixed-boundary ``LASYM=true``.
- SIMSOPT ``LandremanPaul2021_QA_lowres`` and
  ``LandremanPaul2021_QH_reactorScale_lowres`` for QA/QH non-axisymmetric
  fixed-boundary parity and performance.
- SIMSOPT ``examples/2_Intermediate/free_boundary_vmec.py`` for a W7-X
  generated-mgrid free-boundary workflow.
- Landreman ``HSX_QHS_vacuum_ns201``, Ku/Boozer QHS, NCSX, W7-X, and ITER-like
  fixed-boundary decks as optional reference/fetch-only validation targets.

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

Recent bounded parity notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

On 2026-05-19, after ``e0b00e7``, the no-executable physics gates stayed green:

.. code-block:: bash

   python -m pytest tests/test_wout_physics_gates.py tests/test_vmec_parity_physics_fast_gates.py -q

This covered 13 bundled physics/parity scalar checks in about 2.4 seconds on
the local machine.

The additional executable-backed
``up_down_asymmetric_tokamak`` nightly gate did not promote.  VMEC2000 and
``vmec_jax`` both wrote zero aspect/volume scalars for the low-residual
zero-pressure end state, so the optional gate treats aspect as unavailable for
that specific case in the saved-artifact comparison.  The leading residuals in
that dated artifact were ``lmns`` relRMS about ``1.78e-2`` against a ``1e-3``
gate, ``bsupumns`` relRMS about ``1.05e-2`` against a ``1e-2`` gate, and
``bsubvmns`` diff RMS about ``5.72e-4`` against a near-zero VMEC2000 reference.
This remains an optional/instrumented LASYM gap, not a promoted strict external
parity result.

On 2026-05-22, local external fixed-boundary probes were added to the optional
planning manifest after bounded ``xvmec2000`` stage-trace checks against
``$SIMSOPT_ROOT`` and ``$LANDREMAN_VMEC_EQUILIBRIA_ROOT``:

.. code-block:: bash

   python tools/diagnostics/external_vmec_asset_inventory.py \
     --source-root simsopt=$SIMSOPT_ROOT \
     --source-root landreman_vmec_equilibria=$LANDREMAN_VMEC_EQUILIBRIA_ROOT \
     --json-out outputs/external_vmec_assets/all_local_inventory.json

   EXTERNAL_IDS="fixed_nonaxis_lasym_false_simsopt_qh_reactor_lowres_external,fixed_nonaxis_lasym_false_landreman_w7x_standard_boundary,fixed_nonaxis_lasym_false_landreman_ncsx_fixed_boundary,fixed_nonaxis_lasym_false_simsopt_w7x_standard,fixed_nonaxis_lasym_true_simsopt_basic_non_stellsym_external,fixed_nonaxis_lasym_false_landreman_hsx_qhs_fixed"
   python tools/diagnostics/parity_sweep_manifest.py \
     --vmec-exec ~/bin/xvmec2000 \
     --ids "$EXTERNAL_IDS" \
     --output-root outputs/parity_sweeps_external_matrix

Those rows stay optional ``planning`` entries.  They are bounded single-grid,
eight-iteration checks that broaden external coverage across SIMSOPT QH,
SIMSOPT W7-X, SIMSOPT ``LASYM=true`` basic non-stellarator-symmetric,
Landreman W7-X, Landreman NCSX, and Landreman HSX-QHS fixed-boundary assets
without vendoring the external inputs or requiring free-boundary mgrid files.
The refreshed local run
``outputs/parity_sweeps_external_full6/20260522_224144/summary.json`` passed
all six rows against ``~/bin/xvmec2000`` with per-row runtimes between
``13.45 s`` and ``15.10 s``.  This is still optional evidence because the
inputs live in local external checkouts, but it is the current bounded
VMEC2000 matrix for SIMSOPT/Landreman fixed-boundary assets.
The Landreman Ku/Boozer QHS deck is now a bounded optional manifest row after
the comparator learned to patch multiline ``NS_ARRAY``/``NITER_ARRAY``/
``FTOL_ARRAY`` assignments and to ignore ``NITER=-1`` final-reference
``threed1`` records.  The local probe
``outputs/parity_sweeps_external_kuboozer_probe/n4qh_4013d3d`` passed the
eight-iteration ``single_ns=16`` stage-trace gate with zero residual-scalar
drift and matching final-grid WOUT geometry.  The pre-existing
SIMSOPT ``LandremanSenguptaPlunk_section5p3`` LASYM=true planning probe
remained a non-promoted target in the same run context because the strict
``2e-3`` stage-trace gate failed at iteration 10 with a maximum printed
``fsqz`` relative difference of about ``2.97e-2``.

The 2026-05-22 optional converged-WOUT nightly rerun, plus the follow-up LASYM
``bsubvmns`` fix, classify the current optional lanes as follows:

- ``LandremanPaul2021_QA_lowres`` converged-WOUT parity passed locally against
  ``~/bin/xvmec2000``.
- ``basic_non_stellsym_pressure`` converged ``LASYM=true`` finite-beta WOUT
  parity now passes locally after the asymmetric ``bsubvmns`` channel was
  reconstructed from VMEC's corrected half-mesh IEQUI source.
- ``cth_like_free_bdy`` converged-WOUT parity is skipped in the optional
  pytest matrix until it is reduced to a bounded gate; the promoted
  free-boundary evidence is still the stock-executable stage-trace smoke.

Next parity gates
-----------------

The next parity gates are:

- Keep the solved-state QI diagnostic fixture green while broadening
  optimizer seed-robustness claims.
- Run reviewed repeated-stage family-prefine probes across QI, QP, QH, QA, and
  simple seeds.
- Keep VMEC2000 executable smoke green before broadening the executable-backed
  manifest matrix.  The current stock-executable ``LASYM=true`` coverage is a
  vacuum-entry smoke, not strict field-by-field free-boundary parity:

  .. code-block:: bash

     VMEC2000_EXEC=/path/to/xvmec2000 \
     VMEC2000_INTEGRATION=1 \
     pytest -q tests/test_vmec2000_exec_fast_validation.py::test_vmec2000_free_boundary_lasym_true_reaches_vacuum_solve

- Keep the optional bounded free-boundary ``LASYM=true`` manifest case
  instrumented until strict external parity is demonstrated.  The
  ``freeb_scalpot`` comparator needs an instrumented VMEC2000 executable that
  honors the ``VMEC_DUMP_*`` environment variables; a stock ``xvmec2000`` run
  can still solve the case, but it will not emit the scalpot/vacuum dumps used
  for this diagnostic.

  .. code-block:: bash

     VMEC2000_EXEC=/path/to/xvmec2000 \
     VMEC2000_INTEGRATION=1 \
     PYTHONPATH=. python tools/diagnostics/parity_sweep_manifest.py \
       --ids freeb_nonaxis_lasym_true_cth_like_local \
       --output-root results/parity/freeb_lasym_true \
       --manifest tools/diagnostics/parity_manifest.toml \
       --vmec-exec "$VMEC2000_EXEC"

  This self-contained ``vmec_jax/examples`` case uses a scaled synthetic mgrid
  with current signs matched to the plasma current. It exercises
  ``lfreeb=True``, ``lasym=True``, non-axisymmetric vacuum coupling with
  explicit per-iteration runtime and ``freeb_scalpot`` accuracy thresholds.
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
