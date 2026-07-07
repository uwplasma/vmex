Validation and parity with VMEC2000
====================================

``vmec_jax`` uses a layered validation matrix.  Required CI fetches released
VMEC2000-produced ``wout`` fixtures before the parity gates, while a fresh git
clone keeps only the small input decks needed to generate new ``wout`` files.
The validation matrix includes no-solve profile/current gates,
convergence-only end-to-end gates, and focused physics regressions.  Direct
comparisons against a local VMEC2000 Fortran executable are opt-in
``vmec2000`` tests because they require an external executable and are not part
of the default PR gate.

Parity means: given the same input namelist and convergence settings, the
``wout_*.nc`` output of ``vmec_jax`` agrees with the relevant VMEC2000 reference
or executable run to within tolerances set by the convergence level (not by
implementation error).

Reference data
--------------

Eleven ``wout`` reference files are pre-computed with VMEC2000 and shipped as
release assets restored by ``python tools/fetch_assets.py``.  A stable subset is used for strict field-by-field
end-to-end parity in ``tests/test_wout_comprehensive_parity.py``; the remaining
references are covered by no-solve profile/current gates, convergence-only
end-to-end gates, or optional refreshed-reference lanes until their released
references are promoted.

+------------------------------------------+----------------------------------+--------------+---------+
| Input                                    | Coverage                         | lasym        | bdy     |
+==========================================+==================================+==============+=========+
| ``circular_tokamak``                     | axisymmetric, no pressure        | False        | fixed   |
+------------------------------------------+----------------------------------+--------------+---------+
| ``shaped_tokamak_pressure``              | axisymmetric, pressure profile   | False        | fixed   |
+------------------------------------------+----------------------------------+--------------+---------+
| ``DSHAPE``                               | axisymmetric D-shape (STELLOPT)  | False        | fixed   |
+------------------------------------------+----------------------------------+--------------+---------+
| ``nfp4_QH_warm_start``                   | 3D quasi-helical (nfp=4)         | False        | fixed   |
+------------------------------------------+----------------------------------+--------------+---------+
| ``LandremanPaul2021_QA_lowres``          | 3D quasi-axisymmetric (nfp=2)    | False        | fixed   |
+------------------------------------------+----------------------------------+--------------+---------+
| ``nfp3_QI_fixed_resolution_final``       | 3D quasi-isodynamic (nfp=3)      | False        | fixed   |
+------------------------------------------+----------------------------------+--------------+---------+
| ``QI_stel_seed_3127``                    | QI far-seed solved-state fixture | False        | fixed   |
+------------------------------------------+----------------------------------+--------------+---------+
| ``li383_low_res``                        | 3D SIMSOPT reference (nfp=3)     | False        | fixed   |
+------------------------------------------+----------------------------------+--------------+---------+
| ``cth_like_fixed_bdy``                   | 3D current-driven (CTH-like)     | False        | fixed   |
+------------------------------------------+----------------------------------+--------------+---------+
| ``purely_toroidal_field``                | zero-current special case        | False        | fixed   |
+------------------------------------------+----------------------------------+--------------+---------+
| ``basic_non_stellsym_simsopt``           | lasym=True SIMSOPT reference     | True         | fixed   |
+------------------------------------------+----------------------------------+--------------+---------+

Reference WOUT fixtures and large mgrid files not shipped with the git repo can
be fetched once::

  python tools/fetch_assets.py

AD-vs-finite-difference evidence
--------------------------------

The README differentiation panel is generated from central finite differences
and the same public JAX objectives used in examples and tests:

.. image:: _static/figures/readme_ad_fd_evidence.png
   :width: 100%
   :align: center
   :alt: Automatic-differentiation slopes compared with central finite differences

The rows cover fixed-boundary geometry/profile scalars, quasisymmetry and
smooth quasi-isodynamic residual diagnostics, the finite-beta Mercier
``DMerc`` and resistive-interchange ``D_R`` terms, and direct-coil
free-boundary scalar derivatives.  The fixed-boundary and stability rows use
ordinary JAX AD through the differentiable diagnostic expressions.  The
free-boundary rows are deliberately narrower: they are branch-local,
same-fingerprint replay derivatives compared with complete-solve central
finite differences for the same accepted branch.  They do **not** claim
differentiation through arbitrary adaptive branch changes in the host
free-boundary controller.

The checked-in evidence and provenance are:

- :download:`readme_ad_fd_evidence.csv <_static/figures/readme_ad_fd_evidence.csv>`
- :download:`readme_ad_fd_evidence.json <_static/figures/readme_ad_fd_evidence.json>`

For new direct-coil free-boundary derivative checks, prefer the public
``free_boundary_value_and_jvp(..., validate_fd=True)`` report's
``validation_summary`` field when building compact provenance tables.  It
summarizes same-branch compatibility, scalar AD-vs-FD pass status,
cotangent-projected VJP-vs-FD pass status, and maximum scalar errors while
leaving the full ``fd_validation`` payload available for detailed audits.

Regenerate the panel from a same-branch direct-coil report with:

.. code-block:: bash

   JAX_ENABLE_X64=1 python examples/optimization/free_boundary_QS_coil_optimization.py \
     --smoke --provider circle \
     --outdir outputs/pr20_ad_fd/qs_same_branch \
     --write-same-branch-report \
     --same-branch-report-mode vector \
     --same-branch-report-ad-mode direct \
     --same-branch-report-direction current-only \
     --same-branch-report-vector-keys aspect,qs_total,mean_iota,lcfs_boundary_moment \
     --same-branch-report-rejected-slot-gate \
     --max-evals 1 --max-iter 1 --vmec-max-iter 3

   JAX_ENABLE_X64=1 python tools/diagnostics/docs_artifacts/readme_ad_fd_evidence.py \
     --branch-local-report outputs/pr20_ad_fd/qs_same_branch/same_branch_complete_solve_report.json \
     --figure-out docs/_static/figures/readme_ad_fd_evidence.png \
     --csv-out docs/_static/figures/readme_ad_fd_evidence.csv \
     --json-out docs/_static/figures/readme_ad_fd_evidence.json

The focused derivative tests used for this evidence include:

.. code-block:: bash

   JAX_ENABLE_X64=1 pytest -q \
     tests/test_glasser_resistive_interchange.py \
     tests/test_quasisymmetry.py::test_quasisymmetry_wout_residual_gradient_matches_finite_difference \
     tests/test_quasisymmetry.py::test_quasisymmetry_wout_residual_jvp_and_vjp_match_finite_difference \
     tests/test_quasi_isodynamic.py::test_qi_weighted_shuffle_profile_residual_is_finite_and_differentiable \
     tests/test_free_boundary_qs_coil_optimization_smoke.py::test_branch_local_scalar_report_adapter_records_gate_evidence \
     tests/test_free_boundary_qs_coil_optimization_smoke.py::test_branch_local_scalar_report_adapter_records_failure_modes

Automated parity tests
----------------------

Required CI includes a no-executable residual parity gate:

.. code-block:: bash

   PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=1 pytest -q -p no:cacheprovider \
     tests/test_residue_getfsq_parity.py \
     tests/test_wout_profiles_currents_bundled_parity.py \
     tests/test_physics_parity_helper_gates.py \
     tests/test_vmec_parity_physics_fast_gates.py \
     tests/test_wout_physics_gates.py \
     tests/test_converged_wout_matrix_parity.py \
     tests/test_wout_fixture_inventory.py \
     tests/test_vmec2000_exec_threed1.py \
     tests/test_parity_sweep_manifest_thresholds.py

``tests/test_residue_getfsq_parity.py`` reads released VMEC2000 ``wout``
files, reconstructs the solved state, recomputes the
``bcovar -> forces -> tomnsps -> getfsq`` scalar-residual path, and compares
``fsqr``, ``fsqz``, and ``fsql`` to the VMEC2000-stored values.  It currently
covers ``circular_tokamak`` and ``shaped_tokamak_pressure`` without running
VMEC2000 or a full ``vmec_jax`` solve.  ``tests/test_vmec2000_exec_threed1.py``
keeps the executable trace parser covered with a bundled ``threed1`` fixture
when ``xvmec2000`` is absent from CI.

``tests/test_wout_profiles_currents_bundled_parity.py`` is a second required
no-solve wout-field gate.  It checks converged released equilibria directly:
``phipf`` and ``phi`` follow the input flux profile and VMEC half-mesh
integration, finite-beta ``pres/presf`` follow the VMEC radial stencil,
``iotaf`` follows the ``iotas`` full-to-half mesh convention, and the stored
surface-averaged current profiles ``jcuru/jcurv`` match the VMEC finite
difference of ``bvco/buco`` divided by ``mu0``.  The covered fixtures include
axisymmetric finite-beta, non-axisymmetric current-driven, 3D finite-beta, and
``lasym=True`` solved wouts.

``tests/test_converged_wout_matrix_parity.py`` keeps a CI-safe converged-wout
matrix over released VMEC2000 outputs.  The representative fixtures cover
fixed-boundary and free-boundary outputs, axisymmetric and non-axisymmetric
geometry, ``lasym=False`` and ``lasym=True`` channels, and single-grid plus
multigrid input decks.  The gate checks metadata consistency against the
``input.*`` files, final residual RSS limits, flux and iota mesh conventions,
finite stored geometry/field blocks, and the presence or absence of asymmetric
Fourier channels.  It also asserts finite-pressure cases have nonzero stored
pressure, positive pressure energy, and positive beta scalars, including the
fetched single-grid ``lasym=True`` finite-beta reference.

The full test tier runs ``vmec_jax`` end-to-end.  Promoted strict-parity cases
compare the standard ``wout`` field set against VMEC2000 references, while
known-drift or convergence-only cases use explicitly documented finite-output
and physics-gate checks instead of silent promotion.  Run with:

.. code-block:: bash

   RUN_FULL=1 pytest tests/test_wout_comprehensive_parity.py -v

PR #20 readiness parity
-----------------------

For PR #20 readiness, the release gate also refreshed four converged WOUTs
against the local VMEC2000 executable
``/Users/rogeriojorge/local/STELLOPT/VMEC2000/Release/xvmec2000``:
``LandremanPaul2021_QA_lowres``, ``nfp4_QH_warm_start``, ``solovev``, and
``ITERModel``.  All four passed.  The VMEC2000 and ``vmec_jax`` residual RSS
values matched to the reported precision, and representative relative RMS
errors were at roundoff to ``~1e-11`` for core geometry/field channels such as
``rmnc``, ``zmns``, ``iotas``, ``bmnc``, and ``gmnc``.  The worst reported
channel was ``bsubvmnc`` on ``solovev`` at ``4.37e-5``.

The run provenance is checked in as:

- :download:`pr20_wout_parity_summary.json <_static/figures/pr20_wout_parity_summary.json>`

Regenerate this gate with:

.. code-block:: bash

   python tools/diagnostics/parity/converged_wout_parity_benchmark.py \
     --nightly --vmec-exec ~/bin/xvmec2000 \
     --case nfp4_QH_warm_start \
     --case solovev \
     --case ITERModel \
     --case LandremanPaul2021_QA_lowres \
     --output-dir outputs/pr20_wout_parity

The promoted strict-parity cases pass with the following tolerances per field
category:

.. list-table:: Default parity tolerances
   :header-rows: 1
   :widths: 40 20 20

   * - Field category
     - rtol
     - atol
   * - Geometry Fourier coefficients (rmnc, zmns, lmns, gmnc)
     - 1×10⁻⁶
     - 1×10⁻⁷
   * - Magnetic-field Fourier coefficients (bmnc, bsup\*, bsub\*)
     - 5×10⁻⁵
     - 1×10⁻⁷
   * - 1-D profiles (phi, iotas, iotaf, pres, vp, phipf, chipf)
     - 1×10⁻⁶
     - 1×10⁻⁷
   * - Scalar energy/shape (wb, wp, volume_p)
     - 1×10⁻⁶
     - 1×10⁻⁷
   * - Current/field diagnostics (bvco, bdotb, bdotgradv)
     - 5×10⁻⁵
     - 1×10⁻⁷
   * - Near-zero or cancellation-limited (buco, jcuru, jcurv, jdotb)
     - 1×10⁻²
     - 1×10⁻⁸
   * - MHD stability coefficients (DMerc, D_R, Dshear, Dwell, Dcurr, Dgeod)
     - 1×10⁻³
     - 1×10⁻⁸
   * - Equilibrium force residual (equif)
     - 1×10⁻³
     - 1×10⁻⁸

Convergence is also verified (``fsqr``, ``fsqz``, ``fsql`` < 10⁻¹⁰) on every
case before the field comparisons.

Convergence-only tests
----------------------

For input files without a VMEC2000 reference wout, the test suite still
verifies that ``vmec_jax`` converges and produces finite, physically consistent
``wout`` fields.  The convergence-only cases extend coverage to:

- **Stellarator-asymmetric (lasym=True) fixed-boundary**: ``basic_non_stellsym_pressure``,
  ``LandremanSenguptaPlunk_section5p3_low_res``, ``up_down_asymmetric_tokamak``.
  ``basic_non_stellsym_simsopt``.
- **Free-boundary**: ``cth_like_free_bdy`` (requires mgrid from ``fetch_assets.py``).

These cases are exercised by:

.. code-block:: bash

   RUN_FULL=1 pytest tests/test_wout_comprehensive_parity.py -v -k "convergence_only"

QI diagnostics and case coverage
--------------------------------

The required QI tests currently validate the diagnostic definitions and
metadata contracts rather than claiming global optimizer robustness from every
possible seed.  The fast local QI gate is:

.. code-block:: bash

   pytest -q tests/test_quasi_isodynamic.py tests/test_qi_legacy.py tests/test_qi_diagnostics.py tests/test_qi_objective_component_report.py tests/test_qi_seed_suitability_audit.py tests/test_booz_input.py

This gate covers smooth Boozer-space QI residuals, the legacy branch/shuffle
diagnostic used for ranking, mirror-ratio and elongation records, Boozer input
handling, including stellarator-asymmetric geometry and magnetic channels, and
synthetic ranking consistency.  It is intentionally cheap enough for ordinary
development.

The public QI README panel and CSV are generated artifacts, not static
validation fixtures.  They may be regenerated only from reviewed
case-specific NFP=1/2/3/4 minimal-seed runs whose raw/source WOUTs match
``examples/data/input.minimal_seed_nfp*``.  The renderer rejects a row if the
initial WOUT does not match the paired input deck or if the case-gated QI,
mirror, elongation, iota, or finite-metric fields fail.  The seed-3127 and
finite-beta NFP=4 lanes remain explicit diagnostics/stress fixtures rather than
README promotion rows.  These generated rows are not global seed-robustness
claims and are not additional aspect-5 README best-row promotions.
May 2026 bounded NFP=4 reruns that reproduce this metric envelope should be
recorded as QA/provenance checks unless they provide a new reviewed
``docs/_static/qi_readme_cases/nfp4_minimal`` replacement bundle.  Do not
regenerate the README/docs NFP=4 panel from scratch solely from a scratch
result directory; keep the row documented as case-gated and
reference-preconditioned.
``examples/optimization/QI_optimization.py``
is the editable entry point for extending this to other inputs: change the
top-level ``INPUT_FILE`` and ``OUTPUT_DIR`` for a new minimal/circular-like VMEC
deck.  For archived rendered case lanes or stress tests, use
``VMEC_JAX_QI_RUN_CASE`` with ``examples/optimization/qi_optimization_cases.py``
(``nfp1_qi`` through ``nfp4_qi`` are public minimal-seed aliases; named far
seeds such as ``qi_stel_seed_3127`` are explicit diagnostics).  Use
``nfp4_qi_finite_beta`` or ``nfp4_qh_warm_to_qi`` only as NFP=4 stress lanes
until their independent diagnostics pass the QI, mirror, engineering, and
multi-seed gates.
The current
``qi_stel_seed_3127`` far-seed lane first runs a deterministic same-NFP
reference-family boundary preconditioner, records the selected candidate as an
accepted baseline when the independent gates pass, and then runs guarded
QI/iota cleanup.  Review
``boundary_reference_preconditioner/summary.json`` to see which interpolation
point was selected, and review ``mirror_ramp_promotion_log.json`` because
failed cleanup stages are not silently promoted.  The older ESS-scaled basin
prefilter remains available as an opt-in diagnostic and writes
``basin_prefilter/top_candidates.json`` when enabled.  Far-seed stages may use
lower Boozer/QI resolution during the optimization and a higher-resolution
final audit; both resolutions are written to ``diagnostics.json`` so promotion
claims can be traced.  Far seeds may use a solved same-NFP QI wout through
``boozer_target_wout``/``boozer_target_weight`` as an opt-in homotopy
experiment, but that term is not a final acceptance diagnostic.  A seed-robust
QI claim still requires the constrained objective to be run and visually
audited from QI, QP, QH, QA, and simple non-omnigenous starting boundaries.
Accept a row only when the final state satisfies all of: low legacy QI
diagnostic, closed-looking Boozer ``|B|`` contours, mirror ratio at target,
acceptable elongation, ``abs(mean_iota) >= 0.41``, and aspect ratio near the
configured target.

Before launching expensive optimization sweeps, rank available solved seeds
with:

.. code-block:: bash

   PYTHONPATH=. python tools/diagnostics/qi/audit_qi_seed_suitability.py --quick --csv results/qi_seed_audit.csv
   PYTHONPATH=. python tools/diagnostics/qi/audit_qi_seed_suitability.py --quick \
     --case qi_stel_seed_3127:qi:examples/data/input.QI_stel_seed_3127:examples/data/wout_QI_stel_seed_3127.nc \
     --output results/qi_seed_audit/qi_stel_seed_3127.json \
     --csv results/qi_seed_audit/qi_stel_seed_3127.csv

The audit performs no optimization.  It reads solved ``input``/``wout`` pairs
and reports smooth QI, legacy QI, mirror ratio, elongation, aspect ratio, and
mean iota.  Mirror ratio is evaluated over all selected Boozer surfaces by
default.  Optional reference cases from ``omnigenity_optimization`` are used
when ``OMNIGENITY_OPTIMIZATION_ROOT`` points to that checkout; missing optional
cases are recorded as skipped rather than failing the audit.
The bundled default set includes ``input.QI_stel_seed_3127`` when its matching
``wout_QI_stel_seed_3127.nc`` fixture is present.  On 2026-05-12 this seed
audited as a useful but not already-accepted QI robustness start:
smooth/legacy QI were about ``5.0e-2``/``5.0e-2`` before optimization, with
mirror ratio far above the default target and aspect, iota, and elongation
still requiring optimization.
The smooth QI diagnostic includes normalized bounce endpoints by default so the
ranked smooth metric samples the same level range as the legacy Goodman-style
branch-shuffle diagnostic; pass ``--no-include-bounce-endpoints`` only for
interior-level ablation studies.
Rows are ranked by the combined smooth-plus-legacy QI score, while engineering
constraint failures are reported separately so a QI-like seed with a fixable
mirror/aspect violation is not hidden behind a non-QI seed that merely satisfies
the engineering constraints.

Far-seed basin survey
~~~~~~~~~~~~~~~~~~~~~

For inputs that are not already close to the desired QI basin, a purely local
least-squares run can spend its budget improving the wrong basin.  The intended
workflow is therefore global-local: first use a bounded basin survey to try
larger ESS-scaled boundary perturbations, then promote the best candidates into
the differentiable local QI optimizer.  This is closer to basin-hopping than to
a replacement optimizer: random/axis-aligned jumps survey basins, while the
accepted candidates still rely on VMEC/JAX diagnostics and exact local
derivatives for refinement.

Plan a deterministic survey without running VMEC:

.. code-block:: bash

   PYTHONPATH=. python tools/diagnostics/qi/qi_basin_survey.py \
     --input examples/data/input.QI_stel_seed_3127 \
     --output-dir results/diagnostics/qi_basin_survey

Run the bounded survey after reviewing the plan:

.. code-block:: bash

   PYTHONPATH=. python tools/diagnostics/qi/qi_basin_survey.py \
     --input examples/data/input.QI_stel_seed_3127 \
     --output-dir results/diagnostics/qi_basin_survey \
     --execute --save-candidate-inputs

The survey writes ``plan.json``, ``candidates.json``, ``top_candidates.json``,
and ``candidates.csv``.  Candidates are ranked by smooth QI, legacy QI, mirror
ratio, elongation, mean-iota floor, and aspect-ratio proximity.  The top
``input.candidate`` files are not final equilibria; they are starting points for
short local constrained QI optimizations.  Bayesian optimization and
population/evolutionary optimizers are useful future lanes, but they are not the
first production default here because a mode-3 QI boundary has dozens of active
DOFs and each accepted evaluation is expensive.  A cheap deterministic basin
survey gives most of the immediate benefit while preserving the differentiable
local-refinement path.

Promote the top surveyed candidates through bounded differentiable local
refinements:

.. code-block:: bash

   PYTHONPATH=. python tools/diagnostics/qi/qi_basin_promote.py \
     --candidates results/diagnostics/qi_basin_survey/top_candidates.json \
     --out-root results/diagnostics/qi_basin_promotion

After reviewing ``promotion_plan.json``, run:

.. code-block:: bash

   PYTHONPATH=. python tools/diagnostics/qi/qi_basin_promote.py \
     --candidates results/diagnostics/qi_basin_survey/top_candidates.json \
     --out-root results/diagnostics/qi_basin_promotion \
     --execute

The promotion matrix tries direct mode-3 refinement, repeated mode
continuation, QI-then-augmented-Lagrangian cleanup, and a soft-wall
mirror/elongation cleanup.  A promoted row must pass the independent QI+iota
gate and the engineering gate; otherwise it remains diagnostic evidence about
the local basin.

When penalty-based promotion jumps between incompatible basins, use the
filter-search diagnostic.  It accepts a trial only if the already-satisfied
gates are preserved while the active failed gate improves:

.. code-block:: bash

   PYTHONPATH=. python tools/diagnostics/qi/qi_filter_search.py \
     --input results/diagnostics/qi_basin_survey/top_candidate/input.candidate \
     --output-dir results/diagnostics/qi_filter_search

After reviewing ``plan.json``, add ``--execute``.  The search phase order is
QI, then iota, then mirror/elongation.  The output history is checkpointed
after every evaluated trial, and ``--max-trials-per-iteration`` can bound an
interactive batch.  This makes it clear whether the seed has a nearby feasible
path or whether a broader global/basin method is required, even if a long
diagnostic run is interrupted.

To audit a new input deck, first run VMEC once so the audit has a matching
``wout`` file:

.. code-block:: bash

   vmec /path/to/input.my_seed
   PYTHONPATH=. python tools/diagnostics/qi/audit_qi_seed_suitability.py \
     --quick \
     --case my_seed:qi:/path/to/input.my_seed:/path/to/wout_my_seed.nc \
     --output results/qi_seed_audit/my_seed_summary.json \
     --csv results/qi_seed_audit/my_seed_summary.csv

The ``--case`` format is ``label:family:input_path:wout_path``.  The family is
one of ``qi``, ``qp``, ``qh``, ``qa``, or ``simple`` and is used only for
ranking/reporting.  This is the correct first step for arbitrary inputs such as
``examples/data/input.QI_stel_seed_3127``: audit the solved seed, inspect the
reported QI and engineering metrics, then launch a bounded QI optimization only
if the seed is plausible.

To turn the audit into a bounded seed-robustness worklist without launching a
full sweep, add a dry-run prefine manifest:

.. code-block:: bash

   PYTHONPATH=. python tools/diagnostics/qi/audit_qi_seed_suitability.py --quick --prefine-probes plan --prefine-manifest results/qi_seed_audit/prefine_manifest.json --prefine-mirror-weight 2.0 --prefine-elongation-weight 0.5 --prefine-mirror-surface-index all

The manifest records top-ranked seeds plus one best-ranked representative from
each available seed family, hard-capped constrained-QI prefine settings,
expected output files, and exact commands for running one tiny probe at a time.
The default prefine objective includes smooth QI, a QI ceiling, all-surface
mirror ratio, and elongation terms; set ``--prefine-mirror-weight 0`` and
``--prefine-elongation-weight 0`` only for a QI-only ablation.  The default
probe is a bounded repeated-stage continuation,
``--prefine-stage-modes 1,1,2,2,3``, with explicit per-stage and total
``nfev`` caps recorded in the manifest.  Each plan also records the selected
``phimin`` value, its source, the endpoint mode, and the QI options used by the
probe.
Use ``--prefine-probes run --prefine-reviewed`` only after reviewing the
manifest and deliberately executing those capped probes.  Unless explicitly
overridden, prefine probes inherit the audit endpoint setting and record the
alignment in the manifest; by default this is
``endpoint_mode=include_bounce_endpoints``.  Passing
``--no-prefine-include-bounce-endpoints`` is an explicit interior-level
ablation, not the seed-robustness default.
The prefine manifest summary is deterministic and JSON-only: it reports status
counts, completed stage modes, best finite candidate by final objective, best
objective improvement, failed and timed-out probes, objective-history
regressions when compact histories are present, automatic acceptance status,
and one recommended next probe action.  These summaries are audit aids, not a
substitute for final QI physics and plot review.
For tiny smoke probes, automatic acceptance does not require artificial
movement if the seed is already stable and has low objective: a completed,
monotone, finite probe with final QI objective at or below ``5e-2`` is marked
``accepted_stable_low_objective`` rather than rejected for having no measurable
two-evaluation improvement.
Prefine manifests also record ESS controls.  Use ``--no-prefine-use-ess`` and
``--prefine-ess-alpha VALUE`` for bounded ablations when a seed fails at a
higher continuation mode; the selected settings are written into both the
manifest and generated run command.
Mode-continuation stages are stateful: every repeated or higher-mode stage is
rebuilt from the previous stage's optimized VMEC input and starts with a zero
increment vector.  This is part of the validation contract because lower-mode
QI probes can intentionally project high-order seed modes out; later stages
must not silently reintroduce those original high modes from the deck.
Exact optimizer histories are also filtered through an exact-replay acceptance
guard.  Trial residuals may use a cheaper VMEC solve for memory/runtime
reasons, but final outputs and accepted objective histories use the best exact
accepted-point residual seen by the Jacobian path.  Any trial-accepted point
that replays worse is counted in ``rejected_trial_exact_history_count`` rather
than plotted as a monotone accepted step.  Non-finite exact residuals are
discarded before they can become the selected final point.
If SciPy later aborts on a non-finite trust-region linear algebra step after a
finite exact point has already been accepted, the optimizer returns that best
exact point with ``success=False`` and records ``optimizer_exception`` in
``history.json``.  This preserves scientifically useful QI-prefine artifacts
without hiding that the optimizer terminated abnormally.
By default the audit uses ``--phimin-policy well-phase``: each seed is scored at
both ``phimin=0`` and ``phimin=pi/nfp`` and the better QI well phase is used for
ranking and prefine planning.  Use ``--phimin-policy fixed --phimin VALUE`` when
you need a strict single-phase comparison against a legacy run.

For the current optional validation plan, including CI verification commands,
family-representative QI probe workflow, VMEC2000/SIMSOPT optional lanes, and
deferred parity gates, see :doc:`optional_validation_plan`.

Validated ``wout`` fields
--------------------------

Every run produces a NetCDF3-classic ``wout_*.nc`` compatible with VMEC2000
tools.  All of the following fields are written and tested:

- **Geometry Fourier**: ``rmnc``, ``zmns``, ``lmns`` (and ``rmns``, ``zmnc``,
  ``lmnc`` for lasym).
- **Nyquist Fourier**: ``gmnc``, ``bmnc``, ``bsupumnc``, ``bsupvmnc``,
  ``bsubumnc``, ``bsubvmnc``, ``bsubsmns`` (and ``gmns``, ``bmns``,
  ``bsupumns``, ``bsupvmns``, ``bsubumns``, ``bsubvmns`` for lasym).
- **1-D profiles**: ``phi``, ``phipf``, ``phips``, ``chipf``, ``iotas``,
  ``iotaf``, ``pres``, ``presf``, ``vp``.
- **Scalar diagnostics**: ``wb``, ``wp``, ``volume_p``, ``ctor``,
  ``signgs``, ``ns``, ``nfp``, ``mpol``, ``ntor``, ``lasym``, ``gamma``.
- **Current/field diagnostics**: ``buco``, ``bvco``, ``jcuru``, ``jcurv``,
  ``jdotb``, ``bdotb``, ``bdotgradv``, ``equif``.
- **Axis geometry**: ``raxis_cc``, ``zaxis_cs`` (and ``raxis_cs``,
  ``zaxis_cc`` for lasym).
- **MHD stability coefficients**: ``DMerc``, ``D_R``, ``HGlasser``,
  ``GlasserCorrection``, ``GlasserShearValid``, ``DShear``, ``DWell``,
  ``DCurr``, ``DGeod``.
- **Convergence scalars**: ``fsqr``, ``fsqz``, ``fsql``.

Current parity status
---------------------

**Fixed boundary**
  Strict field-by-field parity is established for the promoted comprehensive
  cases.  Other shipped or fetched references are covered by no-solve profile,
  current, b-field, converged-wout matrix, or convergence-only gates until they
  are promoted.

**Stellarator-asymmetric (lasym=True)**
  ``lasym=True`` channels are covered by bundled/fetched reference physics gates
  and convergence tests.  The ``basic_non_stellsym_pressure`` executable-backed
  finite-beta comparison passes the optional nightly converged-WOUT matrix after
  reconstructing the asymmetric ``bsubvmns`` channel from VMEC's corrected
  half-mesh IEQUI source.  Strict external LASYM parity is still not promoted
  broadly: the axisymmetric zero-pressure ``up_down_asymmetric_tokamak``
  nightly comparison remains a known residual gap, led by ``lmns`` and the
  near-zero ``bsubvmns`` sine covariant channel.
  The Boozer input adapter is required to preserve the asymmetric
  geometry/lambda channels (``rmns``, ``zmnc``, ``lmnc``) and magnetic sine
  channels through ``booz_xform_jax`` for QI and LASYM Boozer diagnostics.

**Free boundary**
  vmec_jax produces converged free-boundary equilibria for the bundled CTH-like
  and D3D cases.  Quantitative parity requires ``fetch_assets.py`` for the
  mgrid files.  The free-boundary coil-optimization validation page records the
  current high-resolution finite-beta WOUT-panel evidence: DIII-D
  VMEC2000-compatible ``mgrid`` rows through actual beta 3.33% at ``ns=101`` and
  a strict LP-QA direct-coil stellarator forward lane through actual beta 1.93%.
  The ``mgrid`` path remains the VMEC2000-compatible backend, while the
  direct-coil path provides a differentiable Biot-Savart external-field
  provider for coil currents and Fourier curve coefficients.
  The same page records the current phase-2 adjoint evidence: accepted-trace
  replay gates for current-only, Fourier-only, and mixed coil-control
  perturbations, accepted-state ``bsqvac`` derivatives with respect to the VMEC
  state, and JAX-visible masked nonlinear-controller AD-vs-FD checks. These are
  same-branch, branch-fingerprint-gated validation primitives compared against
  complete-solve finite differences; complete solves remain the acceptance
  authority. They do not promote production differentiation through adaptive
  accepted/rejected ``run_free_boundary`` branch changes.
  See :doc:`free_boundary_coil_optimization` for the artifact links,
  reproduction commands, and phase-2 adjoint limitations.

**Near-zero diagnostics**
  Quantities like ``jdotb`` and Mercier coefficients involve finite-difference
  postprocessing where relative error can be inflated near zero even when both
  codes agree in absolute terms.  See :doc:`jxbforce_mercier` for details.

Per-iteration trace parity
--------------------------

For the highest-fidelity parity (matching VMEC2000 iteration-by-iteration), use
the executable comparator tools:

.. code-block:: bash

   python tools/diagnostics/parity/vmec2000_exec_stage_trace_compare.py \
     --case circular_tokamak --max-iter 10 --single-ns 13

   python tools/diagnostics/parity/parity_sweep_manifest.py --tier smoke

   python tools/diagnostics/parity/wout_compare_axis_mask.py \
     --a /path/to/vmec2000/wout_case.nc \
     --b /path/to/vmec_jax/wout_case.nc \
     --rtol 1e-4 --atol 1e-12

Regenerate converged-wout benchmark summaries with:

.. code-block:: bash

   python tools/diagnostics/parity/converged_wout_parity_benchmark.py --all-discovered-execs
   python tools/diagnostics/parity/converged_wout_parity_benchmark.py --nightly --all-discovered-execs
   python tools/diagnostics/parity/converged_wout_parity_benchmark.py --dry-run --scan-local-execs --all-discovered-execs

The first command runs the bounded circular end-state comparison.  The nightly
variant adds the slower representative non-axisymmetric, ``lasym=True``,
multigrid, and free-boundary cases.  The runner discovers
``$VMEC2000_EXEC``, ``~/bin/xvmec2000``, ``xvmec2000`` on ``PATH``, and the
standard adjacent STELLOPT build path by default, then de-duplicates symlinks
before running.  Use ``--scan-local-execs`` when you also want to recursively
inventory older local benchmark-tree executables before deciding which ones are
safe to run.

Manifest-driven sweep (fixed + free boundary)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The canonical parity matrix is defined in ``tools/diagnostics/parity/parity_manifest.toml``:

.. code-block:: bash

   python tools/diagnostics/parity/parity_sweep_manifest.py --tier smoke
   python tools/diagnostics/parity/parity_sweep_manifest.py --tier full

The manifest covers: fixed-boundary axisymmetric and non-axisymmetric,
``lasym=False`` and ``lasym=True``, free-boundary axisymmetric and
non-axisymmetric.  Required CI only smoke-tests the manifest schema and bounded
dry-run wiring; executing the matrix against VMEC2000 remains an optional local
or scheduled lane.

Latest local executable rerun
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The 2026-05-25 local rerun under ``outputs/rerun_20260525_123334`` used
``/Users/rogeriojorge/local/STELLOPT/VMEC2000/Release/xvmec2000`` and passed
all selected stage-trace comparisons:

.. list-table::
   :header-rows: 1

   * - Matrix
     - Cases
     - Failed
     - Representative coverage
   * - ``parity_smoke``
     - ``6``
     - ``0``
     - circular tokamak, ITERModel, up/down asymmetric tokamak,
       Landreman-Paul QA low resolution, ``basic_non_stellsym_pressure``, and
       bundled CTH-like free-boundary LASYM deck
   * - ``parity_full``
     - ``1``
     - ``0``
     - ``input.nfp4_QH_warm_start`` fixed-boundary QH stage trace

These are bounded stage-trace checks, not a replacement for the optional
converged-WOUT nightly matrix.  They are useful release-candidate evidence that
the latest dirty performance/refactor work has not broken the short VMEC2000
trace path.

Optional VMEC2000 executable checks
-----------------------------------

The default required test suite does not need a local VMEC2000 build.  CI
fetches released ``wout`` references for parity rows; local runs without the
bundle skip those optional fixture tests cleanly:

.. code-block:: bash

   pytest -q -m "not full and not vmec2000"
   RUN_FULL=1 pytest tests/test_wout_comprehensive_parity.py -v

Direct executable comparisons are opt-in because they require a VMEC2000
Fortran executable, and some checks also require ``mpi4py`` and the VMEC2000
Python extension.  Prefer the bounded commands below before broadening to the
full marker suite.

The fastest executable-backed stage-trace validation is:

.. code-block:: bash

   VMEC2000_EXEC=/path/to/xvmec2000 \
   VMEC2000_INTEGRATION=1 \
   pytest -q tests/test_vmec2000_exec_fast_validation.py::test_fast_vmec2000_stage_trace_validation_cases

This command uses bundled fixed-boundary inputs, a single ``ns=13`` grid,
``max_iter=2``, lite dump output, and a 60 second VMEC2000 timeout per case.

The same optional file includes a stock-executable free-boundary
``LASYM=true`` smoke that caps the bundled synthetic CTH-like deck at 120
iterations and verifies VMEC2000 reaches the vacuum solve without an
``I_TOR`` mismatch:

.. code-block:: bash

   VMEC2000_EXEC=/path/to/xvmec2000 \
   VMEC2000_INTEGRATION=1 \
   pytest -q tests/test_vmec2000_exec_fast_validation.py::test_vmec2000_free_boundary_lasym_true_reaches_vacuum_solve

For WOUT-level generated-``mgrid`` free-boundary promotion, use the bounded
W7-X fixture instead of the older LP-QA generated-``mgrid`` diagnostic.  The
LP-QA row remains useful for direct-coil-vs-generated-``mgrid`` vmec_jax
provider parity, but it is not a VMEC2000 WOUT-promotion case.  The W7-X
fixture follows the documented SIMSOPT free-boundary workflow: generate
``mgrid.w7x.nc`` from ``simsopt.configs.get_data("w7x")``, set
``LFREEB=T``, set ``NZETA`` to the generated toroidal plane count, set
``EXTCUR=1.0``, and run raw VMEC2000:

.. code-block:: bash

   VMEC2000_EXEC=~/bin/xvmec2000 \
   python tools/diagnostics/free_boundary/vmec2000_generated_mgrid_w7x_fixture.py \
     --workdir /tmp/vmec_jax_w7x_generated_mgrid_fixture \
     --out results/vmec2000_w7x_generated_mgrid_fixture.json \
     --strict

The corresponding optional pytest gate is:

.. code-block:: bash

   VMEC2000_EXEC=~/bin/xvmec2000 \
   VMEC2000_INTEGRATION=1 \
   pytest -q tests/test_free_boundary_essos_coil_parity.py::test_vmec2000_w7x_generated_mgrid_fixture_reaches_active_vacuum_and_finite_wout

The promotion criteria are intentionally stricter than "VMEC2000 returned":
active vacuum evidence must be present, a parseable WOUT must be written,
``fsqr + fsqz + fsql`` must be finite and below the configured bound, and the
geometry scalars ``aspect``, ``volume_p``, ``Rmajor_p``, and ``Aminor_p`` must
be finite and positive.  The generated ``mgrid`` and WOUT remain in the local
work directory and are not git fixtures.

For a short CLI comparison against the executable:

.. code-block:: bash

   VMEC2000_EXEC=/path/to/xvmec2000 \
   VMEC2000_INTEGRATION=1 \
   VMEC2000_CLI_NITER=5 \
   pytest -q tests/test_cli_vmec2000_exec.py

This caps both VMEC2000 and ``vmec_jax`` CLI runs at five iterations.  To run
the whole executable-backed suite after the bounded checks are green:

.. code-block:: bash

   VMEC2000_EXEC=/path/to/xvmec2000 \
   VMEC2000_INTEGRATION=1 \
   pytest -q -m vmec2000

Converged end-state VMEC2000-vs-``vmec_jax`` comparisons are in
``tests/test_vmec2000_converged_parity.py``.  By default this runs only the
bounded fixed-boundary circular case when ``VMEC2000_INTEGRATION=1`` is set.
Set ``VMEC2000_NIGHTLY=1`` as well to include the slower non-axisymmetric,
``lasym=True``, multigrid, and free-boundary representatives:

.. code-block:: bash

   VMEC2000_EXEC=/path/to/xvmec2000 \
   VMEC2000_INTEGRATION=1 \
   VMEC2000_NIGHTLY=1 \
   pytest -q tests/test_vmec2000_converged_parity.py

On 2026-05-29 this nightly command passed locally with
``~/bin/xvmec2000``: ``4 passed, 1 skipped, 1 xfailed`` in ``15:06``.  The
skipped row is the intentionally deferred converged free-boundary WOUT parity
case, and the xfail is the documented zero-pressure axisymmetric LASYM gap.

The fetched single-grid ``lasym=True`` finite-beta fixture is a required
bundled-reference physics gate, and the executable-backed
``basic_non_stellsym_pressure`` converged-WOUT row passes locally against
``~/bin/xvmec2000`` after the asymmetric ``bsubvmns`` output channel was switched
to VMEC's corrected half-mesh IEQUI source.  The separate zero-pressure,
axisymmetric ``up_down_asymmetric_tokamak`` strict external LASYM gap remains
non-promoted: a 2026-05-19 rerun showed ``lmns=1.78e-2`` relRMS,
``bsupumns=1.05e-2`` relRMS, and ``bsubvmns`` ``diff_rms=5.72e-4`` against a
near-zero ``ref_rms=4.10e-5``.  The ``reference_state_roundtrip_rel_rms`` split
from the converged-wout benchmark keeps the remaining lambda work focused on
the ``m=1,3,4`` LASYM channels and the near-zero ``bsubvmns`` comparison on
absolute error.  The free-boundary converged-WOUT row is skipped until it is
reduced to a bounded nightly gate; use the promoted stage-trace free-boundary
smoke for routine executable parity.

Optional SIMSOPT formula parity is similarly guarded and targeted:

.. code-block:: bash

   RUN_SIMSOPT_VALIDATION=1 \
   pytest -q tests/test_simsopt_optional_validation.py::test_qh_quasisymmetry_residual_matches_simsopt_wout_formula

The machine-readable list of these bounded parity commands is emitted by:

.. code-block:: bash

   python validation/qi_seed_robustness_plan.py --output results/qi_seed_audit/validation_plan.json

The emitted plan intentionally does not embed a stale green CI run by default.
Verify the current ``main`` CI run with ``gh run list``/``gh run view`` and pass
the run metadata to the helper when preparing release-validation artifacts.

Skip behavior is intentional.  Tests marked ``vmec2000`` skip unless
``VMEC2000_INTEGRATION=1`` is set.  They also skip, rather than fail, when the
VMEC2000 executable, VMEC2000 Python extension, ``mpi4py``, ``netCDF4``, an
input deck, or a VMEC2000-produced ``wout`` is unavailable.  Required PR CI
therefore excludes ``vmec2000`` tests; optional scheduled/manual CI can enable
them after installing VMEC2000 and exporting ``VMEC2000_EXEC``.

VMECPlot2 compatibility
-----------------------

``vmec_jax`` writes **NetCDF3-classic** ``wout_*.nc`` files compatible with
``vmecPlot2.py``.  Any workflow that reads VMEC2000 output can consume
``vmec_jax`` output without modification.

The showcase scripts generate side-by-side comparison figures using the same
VMECPlot2-style grids (theta/zeta resolution, toroidal angle conventions)::

  python examples/showcase_axisym_input_to_wout.py --suite
