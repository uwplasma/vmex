Optimization Sweep Results
==========================

This page collects the generated optimization sweep artifacts used by the
README and the main optimization guide.  The current sweep covers QA, QH, QP,
and QI targets:

- QA: the reference omnigenity NFP=2 QA deck, aspect ratio near 5,
  signed mean iota target 0.42, and quasi-axisymmetry.
- QH: the bundled NFP=4 warm start, aspect ratio near 5, quasi-helical
  symmetry, and a smooth ``abs(mean_iota) >= 0.41`` lower bound.
- QP: aspect ratio near 5, quasi-poloidal symmetry, and a smooth
  ``abs(mean_iota) >= 0.41`` lower bound, using the same bundled NFP=2 seed as
  the QI runs.
- QI: aspect ratio near 10, a differentiable smooth Boozer-space quasi-isodynamic
  residual evaluated through ``booz_xform_jax``, maximum mirror-ratio penalty,
  maximum-LCFS-elongation penalty, and a smooth ``abs(mean_iota) >= 0.41``
  lower bound.  ``LgradB`` is available as an optional commented term in the
  example script but is not active by default.
  The QI rows use the bundled ``input.nfp2_QI`` omnigenity seed.
  The production CLI can optionally use a same-mode QP preseed; the current
  best gated QI row starts the constrained QI refinement directly from the seed
  with ``--qi-qp-preseed off``.

Individual Examples
-------------------

Each standalone example keeps all user controls as top-level Python variables:

.. code-block:: bash

   PYTHONPATH=. python examples/optimization/QA_optimization.py
   PYTHONPATH=. python examples/optimization/QH_optimization.py
   PYTHONPATH=. python examples/optimization/QP_optimization.py
   PYTHONPATH=. python examples/optimization/QI_optimization.py

The QP script is quasisymmetry with ``HELICITY_M = 0``.  The QI script is a
different objective: it builds Boozer spectra with ``booz_xform_jax``, improves
the smooth QI residual, and then adds mirror-ratio and LCFS-elongation
penalties.  A commented ``LgradB`` block is included for users who want that
extra regularization term.  The extra terms are imported from
``vmec_jax.optimization_workflow`` and assembled explicitly in the script, so
users can change weights or add terms such as magnetic-well depth by appending
another residual block in the same section.  Install the optional dependency set with
``python -m pip install ".[qi]"`` before running QI cases from a source
checkout.

Sweep Reproduction
------------------

Run the CPU production sweep:

.. code-block:: bash

   PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/generate_qs_ess_sweep.py --backend-label cpu --solver-device cpu --policy continuation --problems qa,qh,qp,qi --modes 1,2,3 --ess both --qi-qp-preseed off --rerun
   PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/generate_qs_ess_sweep.py --backend-label cpu --solver-device cpu --policy direct --problems qa,qh,qp,qi --modes 1,2,3 --ess both --qi-qp-preseed off --rerun
   PYTHONPATH=. python examples/optimization/render_qs_ess_publication_panel.py

The current QI sweep configuration uses aspect target 10.  If the generated
``qs_ess_summary_all.csv`` still contains QI rows with ``target_aspect = 5``,
those rows predate the current QI policy and the sweep summary should be
regenerated before using the QI entries.

The constrained QI study has one extra axis: whether the QI solve starts from a
same-mode QP preseed.  The production CLI default is ``--qi-qp-preseed off``;
regenerate the focused preseed/no-preseed matrix with:

.. code-block:: bash

   PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/generate_qs_ess_sweep.py --backend-label cpu --solver-device cpu --policy continuation --problems qi --modes 1,2,3 --ess both --qi-qp-preseed both --rerun
   PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/generate_qs_ess_sweep.py --backend-label cpu --solver-device cpu --policy direct --problems qi --modes 1,2,3 --ess both --qi-qp-preseed both --rerun
   PYTHONPATH=. python examples/optimization/render_qi_constrained_sweep.py

Seed-robust QI is a larger validation matrix than the tracked constrained-QI
snapshot.  The current artifacts use the bundled NFP=2 QI seed and optional
same-mode QP preseed.  Before documenting QI as robust to arbitrary starts,
run the constrained objective from QI, QP, QH, QA, and a simple
non-omnigenous boundary seed, then inspect both numerical gates and Boozer
``|B|`` contour plots.  Until those rows are generated and curated, treat the
full multi-seed matrix as deferred validation rather than a required
reproduction command.

Use the bounded seed preflight to choose and document starting points before
launching that matrix:

.. code-block:: bash

   PYTHONPATH=. python examples/optimization/audit_qi_seed_suitability.py --quick --csv results/qi_seed_audit.csv
   PYTHONPATH=. python examples/optimization/audit_qi_seed_suitability.py --quick --smooth-qi-max 5e-3 --legacy-qi-max 2e-3 --csv results/qi_seed3127_audit.csv

The preflight ranks seeds by smooth-plus-legacy QI score and leaves mirror,
elongation, aspect, and iota gates as explicit columns.  The second command
shows the relaxed smooth-gate convention used by the far-seed
``qi_stel_seed_3127`` optimization case.  This is deliberate:
for QI robustness work, a seed that is already QI-like but violates mirror or
aspect slightly is usually more informative than a non-QI seed that satisfies
the engineering constraints.

To review the tiny QI-prefine probes before executing them:

.. code-block:: bash

   PYTHONPATH=. python examples/optimization/audit_qi_seed_suitability.py --quick --prefine-probes plan --prefine-manifest results/qi_seed_audit/prefine_manifest.json --prefine-output-dir results/qi_seed_audit/prefine_probes

The dry-run manifest is the intended bridge between seed audit and full
seed-robust QI sweeps.  It is bounded by design and should be inspected before
switching to ``--prefine-probes run``.

NFP=4 QI remains a tracked stress case rather than a passing robustness lane.
Use ``VMEC_JAX_QI_RUN_CASE=nfp4_qh_warm_to_qi`` only as a non-passing stress
fixture unless its independent ``diagnostics.json`` records a smooth/legacy QI
and mirror-ratio gate pass.  Local May 2026 quick audits found no passing NFP=4
path: the best archived same-NFP QI reference was still above the ``2e-3``
smooth/legacy gates, while the bundled QH warm start and local QH-to-QI cleanup
were farther from the legacy-QI threshold.  Keep these rows in seed-audit
tables and do not mix them with promoted QI sweep results.

Run the GPU production sweep on a machine with a working JAX GPU install:

.. code-block:: bash

   PYTHONPATH=. JAX_PLATFORM_NAME=gpu python examples/optimization/generate_qs_ess_sweep.py --backend-label gpu --solver-device gpu --policy continuation --problems qa,qh,qp,qi --modes 1,2,3 --ess both --qi-qp-preseed off --rerun
   PYTHONPATH=. JAX_PLATFORM_NAME=gpu python examples/optimization/generate_qs_ess_sweep.py --backend-label gpu --solver-device gpu --policy direct --problems qa,qh,qp,qi --modes 1,2,3 --ess both --qi-qp-preseed off --rerun
   PYTHONPATH=. python examples/optimization/render_qs_ess_publication_panel.py

Render the compact README panels from the best stellarator-symmetric rows:

.. code-block:: bash

   PYTHONPATH=. python examples/optimization/render_readme_best_optimizations.py

The default per-case timeout is 1200 seconds.  The current science configs use
NFP=4 for QH, aspect targets near 5 for QA/QH/QP, aspect target near 10 for QI,
signed iota 0.42 for QA, and high-priority ``abs(mean_iota) >= 0.41``
constraints for QH/QP/QI.
They use ``inner_max_iter = trial_max_iter = 120`` and
``ftol = trial_ftol = 1e-9``; GPU production sweeps cap those values at 180
if a future problem config requests a larger replay budget.  Add
``--diagnostic-budgets`` only for bounded quick-look GPU diagnostics, and use
``--case-timeout-s 0`` only for unbounded local diagnostics.

Run the non-stellarator-symmetric sweep by adding
``--stellarator-asymmetric``.  This sets ``LASYM = T`` in memory, includes
``RBS`` and ``ZBC`` boundary degrees of freedom, seeds initially-zero
asymmetric modes with ``1e-7``, and writes separate outputs under the
``asymmetric`` backend subdirectory.

.. code-block:: bash

   PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/generate_qs_ess_sweep.py --backend-label cpu --solver-device cpu --policy continuation --problems qa,qh,qp,qi --modes 1,2,3 --ess both --qi-qp-preseed off --stellarator-asymmetric --rerun
   PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/generate_qs_ess_sweep.py --backend-label cpu --solver-device cpu --policy direct --problems qa,qh,qp,qi --modes 1,2,3 --ess both --qi-qp-preseed off --stellarator-asymmetric --rerun
   PYTHONPATH=. JAX_PLATFORM_NAME=gpu python examples/optimization/generate_qs_ess_sweep.py --backend-label gpu --solver-device gpu --policy continuation --problems qa,qh,qp,qi --modes 1,2,3 --ess both --qi-qp-preseed off --stellarator-asymmetric --rerun
   PYTHONPATH=. JAX_PLATFORM_NAME=gpu python examples/optimization/generate_qs_ess_sweep.py --backend-label gpu --solver-device gpu --policy direct --problems qa,qh,qp,qi --modes 1,2,3 --ess both --qi-qp-preseed off --stellarator-asymmetric --rerun
   PYTHONPATH=. python examples/optimization/render_qs_ess_publication_panel.py

For NVIDIA-only JAX installations, ``JAX_PLATFORMS=cuda`` is also valid.  Do
not use ``JAX_PLATFORMS=gpu``: some JAX versions interpret that as both CUDA
and ROCm and fail if ROCm is not installed.

README Best Rows
----------------

The README intentionally shows only one best ``LASYM = F`` result per target.
QA/QH/QP are selected from the CPU matrix and filtered against the common
aspect-5 target.  QI candidates are selected from the current constrained-QI
matrix when target-10 rows exist and from the dedicated staged
``QI_optimization.py`` default lane otherwise; the selected QI row must satisfy
the legacy branch diagnostic, mirror-ratio, elongation, iota, and aspect-ratio
gates.  These panels include the original deck LCFS before any ``max_mode=1``
optimization work, final LCFS, per-stage objective history, and final
outer-surface ``|B|`` in Boozer coordinates evaluated with
``booz_xform_jax``.
The source table is also available as
:download:`readme_best_optimizations.csv <_static/figures/readme_best_optimizations.csv>`.

.. image:: _static/figures/readme_best_optimization_qa.png
   :width: 100%
   :align: center
   :alt: Best README QA optimization panel

.. image:: _static/figures/readme_best_optimization_qh.png
   :width: 100%
   :align: center
   :alt: Best README QH optimization panel

.. image:: _static/figures/readme_best_optimization_qp.png
   :width: 100%
   :align: center
   :alt: Best README QP optimization panel

.. image:: _static/figures/readme_best_optimization_qi.png
   :width: 100%
   :align: center
   :alt: Best README QI optimization panel

QI_optimization Input Coverage
------------------------------

The dedicated QI README/docs renderer covers both bundled QI inputs without
rerunning optimization jobs.  It reads the existing ``QI_optimization.py``
outputs, records the final smooth QI metric, legacy QI metric, mirror ratio,
elongation, iota, aspect, and CPU wall time, and draws final Boozer ``|B|``
with line contours only.

.. list-table::
   :header-rows: 1
   :widths: 22 27 11 11 11 10 9 9 9 9

   * - Input
     - Output/provenance
     - Final J
     - Smooth QI
     - Legacy QI
     - Mirror
     - Elong.
     - Aspect
     - Iota
     - CPU min
   * - ``examples/data/input.nfp2_QI``
     - ``results/qi_opt/ess/nfp2_qi``
     - ``1.17e-2``
     - ``1.13e-3``
     - ``3.09e-4``
     - ``0.225/0.30``
     - ``6.43/8.2``
     - ``9.999/10.0``
     - ``-0.5043``
     - ``14.7``
   * - ``examples/data/input.QI_stel_seed_3127``
     - ``results/qi_opt/ess/qi_stel_seed_3127_current_public_final``
     - ``1.12e-1``
     - ``4.32e-3``
     - ``1.16e-3``
     - ``0.316/0.35``
     - ``3.91/8.0``
     - ``3.465/4.0``
     - ``-1.0366``
     - ``6.3``

.. image:: _static/figures/readme_qi_optimization_cases.png
   :width: 100%
   :align: center
   :alt: QI optimization coverage for NFP=2 QI and seed-3127 inputs

Source table:
:download:`readme_qi_optimization_cases.csv <_static/figures/readme_qi_optimization_cases.csv>`.

Regenerate these lightweight artifacts with:

.. code-block:: bash

   PYTHONPATH=. python examples/optimization/render_qi_readme_cases.py

The staged objective panel concatenates every recorded history file used by
the promoted result.  It plots the best-so-far value in each stage, normalized
to that stage's first objective, with dashed separators marking
objective-definition or weight changes.  For the seed-3127 lane, the inset is
a boundary-reference interpolation scan, not an optimizer trajectory.

Objective Histories
-------------------

The all-policy panel contains every available backend/policy row.  Solid curves
met the optimizer success criterion; dashed curves are stopped, failed, or
budgeted lanes.  The summary tables distinguish ``max_nfev`` stops from
1200 second timeouts and GPU-memory failures.  Curves are split by objective
stage and plotted as best-so-far values within that stage, so QP preseed and
full constrained QI refinement are not treated as one continuous scalar
objective.  Vertical dotted lines mark continuation stage boundaries.

Constrained QI Matrix
---------------------

The constrained QI renderer compares CPU and available GPU rows for
``max_mode = 1, 2, 3``, ESS on/off, continuation/direct, and QP-preseed
on/off using the bundled NFP=2 ``input.nfp2_QI`` seed.  The current tracked
snapshot predates the target-10 QI policy and contains target-5 rows, so it is
stale for current constrained-QI claims.  Rerun the commands above to populate
the CPU/GPU/direct/asymmetric matrix under the current objective policy.  For
each requested
``max_mode``, the input boundary is projected onto
``max(abs(m), abs(n)) <= max_mode`` before the stage is built, so the
``max_mode=1`` rows zero the mode-2 coefficients present in the warm start.
The QI objective is intentionally not ranked by scalar objective alone: rows
are also evaluated by the legacy branch-squash/stretch/shuffle diagnostic,
raw smooth QI residual, maximum mirror ratio, maximum LCFS elongation,
``abs(mean_iota) >= 0.41``, and aspect ratio near 10.  The default smooth QI
objective includes ``shuffle_profile_weight = 1.0`` so the optimizer follows
the same ranking as the legacy diagnostic on the seed and reference
omnigenity cases.  Rows that stop at ``max_nfev`` but have valid VMEC solves
and satisfy the physics gates are kept as valid stopped rows.

.. image:: _static/figures/qi_constrained_objective_panel.png
   :width: 100%
   :align: center
   :alt: Constrained QI objective matrix

Downloadable constrained-QI summaries:

- :download:`qi_constrained_summary.csv <_static/figures/qi_constrained_summary.csv>`
- :download:`qi_constrained_summary.json <_static/figures/qi_constrained_summary.json>`
- :download:`qi_constrained_best.json <_static/figures/qi_constrained_best.json>`

The stale target-5 snapshot reports a CPU repeated-stage continuation,
``max_mode=3``, ESS row without a same-mode QP preseed: legacy-ranked QI
diagnostic ``2.17e-3``, maximum mirror ratio ``0.2106`` for a target ``0.21``,
maximum elongation ``4.30`` for a target ``8.0``, aspect ratio ``5.001``, mean
iota ``-0.5494``, and total wall time ``11.3 min``.  Do not use that row as a
current target-10 constrained-QI best case.  Best-row selection uses
``vmec_jax.qi_promotion_score``: raw-fallback legacy diagnostics are rejected,
rows above the loose ``2e-2`` QI promotion ceiling cannot win solely by having
good mirror/elongation, and engineering-clean rows are preferred over lower-QI
rows only when they remain QI-like.
The constrained matrix is a bundled-QI-seeded lane, not evidence that the
optimizer is seed-robust across unrelated QA/QH/QP/simple starts.

Non-stellarator-symmetric LASYM runs use the same script with
``--stellarator-asymmetric``.  The current LASYM artifacts are intentionally
published as partial 1200 second lanes.  Timeout and OOM rows are kept because
they document the current cost envelope of the asymmetric exact/replay path.
The frozen snapshot has 13 CPU LASYM rows and 61 GPU LASYM rows.

The objective-history figures are generated artifacts.  They are not tracked in
git; regenerate them with
``PYTHONPATH=. python examples/optimization/render_qs_ess_publication_panel.py``
after producing or fetching the sweep results.  The renderer writes:

- ``objective_panel_all_policies.png/.pdf``
- ``objective_panel_cpu_policies.png/.pdf``
- ``objective_panel_gpu_policies.png/.pdf``
- ``objective_panel_asymmetric_all_policies.png/.pdf``
- legacy aliases ``objective_panel.png/.pdf``

Final-State Atlases
-------------------

The final-state atlases show the LCFS and line contours of ``|B|`` on the LCFS.
Each 3-D panel has its own colorbar because the aspect-ratio constraint changes
the absolute ``|B|`` range.

Partial LASYM atlases are rendered separately.  Missing or failed lanes are
shown as placeholders, while successful lanes include one colorbar per 3-D
surface and one colorbar per LCFS ``|B|`` contour panel.

Generated atlas filenames include:

- ``final_state_atlas_continuation.png/.pdf``
- ``final_state_atlas_direct.png/.pdf``
- ``final_state_atlas_cpu_continuation.png/.pdf``
- ``final_state_atlas_cpu_direct.png/.pdf``
- ``final_state_atlas_gpu_continuation.png/.pdf``
- ``final_state_atlas_gpu_direct.png/.pdf``
- ``final_state_atlas_asymmetric_*``
- legacy alias ``geometry_atlas.png/.pdf``

Summary Tables
--------------

The summary-table image is intended for reports and presentations.  The CSV and
JSON are better for analysis scripts.

Downloadable summaries:

- :download:`summary_all.csv <_static/figures/qs_ess_summary_all.csv>`
- :download:`summary_all.json <_static/figures/qs_ess_summary_all.json>`

Generated summary-table figures include ``summary_tables_all_policies``,
``summary_tables_cpu_policies``, ``summary_tables_gpu_policies``,
``summary_tables_asymmetric_all_policies``, and legacy alias
``summary_table``.

Publication Panel
-----------------

The full panel combines objective histories, final-state atlases, and summary
tables.  It is large by design and should be used for review, not as the only
README figure.

Generated full-panel filenames include ``publication_panel_full.png/.pdf``,
legacy alias ``publication_panel.png/.pdf``, and
``publication_panel_asymmetric_full.png/.pdf`` for the partial LASYM lanes.

Finite-beta Stage-One Examples
------------------------------

The finite-beta examples mirror the VMEC-only stage-one part of
``/Users/rogeriojorge/local/single_stage_optimization_finite_beta`` without
SIMSOPT or coils:

.. code-block:: bash

   PYTHONPATH=. python examples/optimization/qa_optimization_finite_beta.py
   PYTHONPATH=. python examples/optimization/qh_optimization_finite_beta.py
   PYTHONPATH=. python examples/optimization/qi_optimization_finite_beta.py

The input decks are bundled as:

- ``examples/data/input.nfp2_QA_finite_beta``
- ``examples/data/input.nfp4_QH_finite_beta``
- ``examples/data/input.nfp4_QI_finite_beta``

Each script builds the optimization problem explicitly: load the VMEC input,
construct ``FiniteBetaTargets``, define the global residuals for aspect ratio,
iota lower/mean/upper bounds, volume-averaged field proxy, and total beta, then
append the field-quality residual.  QA/QH use quasisymmetry residuals and QI
uses the smooth Boozer-space QI residual.  The small shared helper only keeps
the stage bookkeeping, structured stage/final summaries, and artifact writing
consistent.  After the direct ``FixedBoundaryExactOptimizer`` calls complete,
``stage1_result.stage_summaries`` and ``stage1_result.final_summary`` expose
JSON-friendly diagnostics such as objective, aspect, iota, function counts,
termination status, method, device, and wall time.  The scripts save
``input.initial``, ``input.final``, ``wout_initial.nc``, ``wout_final.nc``, and
``history.json`` for each run.

These examples deliberately remain one layer lower than
``least_squares_solve``.  The ordinary QA/QH/QP/QI scripts are the recommended
objective-tuple API examples; finite-beta stage one keeps direct
``FixedBoundaryExactOptimizer`` calls visible so users can see the custom
finite-pressure/current residual closures, stage-local profile data, and
optional bootstrap-current residual wiring.

All finite-beta controls are plain variables at the top of the scripts.  For
QI, ``QI_MBOZ``, ``QI_NBOZ``, ``QI_NPHI``, ``QI_NALPHA``, and
``QI_N_BOUNCE`` control the Boozer/QI residual grid. ``MAX_MIRROR_RATIO`` and
``MAX_ELONGATION`` control the two engineering penalties.  The default QI grid
is small enough for first-run diagnostics; increase it for final
research-quality QI runs. ``QI_OPTIONS.phimin`` controls the start of the
one-field-period well interval; keep ``0.0`` for the bundled NFP=2 seed, or set
``np.pi / nfp`` when auditing a reference field whose first well starts there.

Redl bootstrap-current mismatch is now available as ``vj.RedlBootstrapMismatch``.
The implementation ports the SIMSOPT/Redl et al. bootstrap algebra and uses a
differentiable VMEC-state geometry approximation with fixed trapped-fraction
quadrature on nearest full-mesh surfaces.  Mercier ``DMerc`` is also available
as ``vj.DMerc``, a smooth lower-bound objective backed by the differentiable
``mercier_terms_from_state`` path for stellarator-symmetric and LASYM
equilibria.  VMEC/JXBFORCE profile accessors ``vj.JDotB``, ``vj.BDotB`` and
``vj.BDotGradV`` are also available for finite-beta targeting or
regularization, together with state-derived ``vj.ToroidalCurrent`` and
``vj.ToroidalCurrentGradient`` current-profile objectives.  Advanced vector
targeting is available through ``vj.BVector`` for Cartesian magnetic field on a
single radial surface and ``vj.JVector`` for flattened VMEC-coordinate
``(J^theta, J^zeta)`` current-density components.  The JAX
``mercier_gpp_from_realspace_geometry``,
``mercier_surface_integrals_from_realspace``, and
``mercier_terms_from_profile_integrals`` helpers now cover the VMEC-style
geometry channel, surface reductions, and algebraic
``DMerc = DShear + DCurr + DWell + DGeod`` step once real-space field channels
are available.  ``mercier_realspace_geometry_channels_from_state``,
``mercier_bsubs_half_mesh_from_geometry``,
``mercier_zeta_half_mesh_from_realspace_geometry``,
``mercier_bsubs_full_mesh_from_half_mesh``,
``mercier_bsubs_derivatives_lasym_false``, and
``mercier_bdotk_from_covariant_derivatives`` also port state synthesis of the
even/odd geometry channels, half-mesh toroidal geometry, radial covariant field
assembly, jxbforce full-mesh averaging, stellarator-symmetric derivative
reconstruction, LASYM derivative reconstruction, ``itheta/izeta/bdotk``,
``jdotb/bdotb/bdotgradv`` profile blocks, and ``torcur/ip`` current-profile
blocks.  The ``RUN_FULL=1`` finite-beta test suite compares this state-level
path against the existing VMEC/wout Mercier/JXBFORCE implementation on the
bundled finite-beta QI input.  The next finite-beta validation lane is comparing
the new Redl residual against SIMSOPT's Boozer-geometry and VMEC-geometry Redl
paths on converged finite-beta equilibria.
