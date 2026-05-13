Validation and parity with VMEC2000
====================================

``vmec_jax`` achieves full numerical parity with **VMEC2000** across fixed-boundary
and free-boundary configurations, including axisymmetric, non-axisymmetric,
stellarator-symmetric (``lasym=False``) and stellarator-asymmetric
(``lasym=True``) equilibria.

Parity means: given the same input namelist and convergence settings, the
``wout_*.nc`` output of ``vmec_jax`` agrees with the output of the VMEC2000
Fortran executable to within tolerances set by the convergence level (not by
implementation error).

Reference data
--------------

Ten bundled ``wout`` reference files are pre-computed with VMEC2000 and
shipped in ``examples/data/``:

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
| ``li383_low_res``                        | 3D SIMSOPT reference (nfp=3)     | False        | fixed   |
+------------------------------------------+----------------------------------+--------------+---------+
| ``cth_like_fixed_bdy``                   | 3D current-driven (CTH-like)     | False        | fixed   |
+------------------------------------------+----------------------------------+--------------+---------+
| ``purely_toroidal_field``                | zero-current special case        | False        | fixed   |
+------------------------------------------+----------------------------------+--------------+---------+
| ``basic_non_stellsym_simsopt``           | lasym=True SIMSOPT reference     | True         | fixed   |
+------------------------------------------+----------------------------------+--------------+---------+

Large reference wouts and mgrid files not shipped with the git repo can be
fetched once::

  python tools/fetch_assets.py

Automated parity tests
----------------------

Required CI includes a no-executable residual parity gate:

.. code-block:: bash

   JAX_ENABLE_X64=1 pytest -q tests/test_residue_getfsq_parity.py tests/test_wout_profiles_currents_bundled_parity.py tests/test_vmec2000_exec_threed1.py

``tests/test_residue_getfsq_parity.py`` reads small bundled VMEC2000 ``wout``
files, reconstructs the solved state, recomputes the
``bcovar -> forces -> tomnsps -> getfsq`` scalar-residual path, and compares
``fsqr``, ``fsqz``, and ``fsql`` to the VMEC2000-stored values.  It currently
covers ``circular_tokamak`` and ``shaped_tokamak_pressure`` without running
VMEC2000 or a full ``vmec_jax`` solve.  ``tests/test_vmec2000_exec_threed1.py``
keeps the executable trace parser covered with a bundled ``threed1`` fixture
when ``xvmec2000`` is absent from CI.

``tests/test_wout_profiles_currents_bundled_parity.py`` is a second required
no-solve wout-field gate.  It checks converged bundled equilibria directly:
``phipf`` and ``phi`` follow the input flux profile and VMEC half-mesh
integration, finite-beta ``pres/presf`` follow the VMEC radial stencil,
``iotaf`` follows the ``iotas`` full-to-half mesh convention, and the stored
surface-averaged current profiles ``jcuru/jcurv`` match the VMEC finite
difference of ``bvco/buco`` divided by ``mu0``.  The covered fixtures include
axisymmetric finite-beta, non-axisymmetric current-driven, 3D finite-beta, and
``lasym=True`` solved wouts.

The test suite runs ``vmec_jax`` end-to-end and compares every standard
``wout`` field against the VMEC2000 references.  Run with:

.. code-block:: bash

   RUN_FULL=1 pytest tests/test_wout_comprehensive_parity.py -v

All ten reference cases pass with the following tolerances per field category:

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
     - 5×10⁻³
     - 1×10⁻⁸
   * - MHD stability coefficients (DMerc, Dshear, Dwell, Dcurr, Dgeod)
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
  (Note: ``basic_non_stellsym_simsopt`` now has a VMEC2000 reference and is in
  the full parity suite.)
- **Free-boundary**: ``cth_like_free_bdy`` (requires mgrid from ``fetch_assets.py``).

These cases are exercised by:

.. code-block:: bash

   RUN_FULL=1 pytest tests/test_wout_comprehensive_parity.py -v -k "convergence_only"

QI diagnostics and seed robustness
----------------------------------

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

The current constrained-QI sweep artifacts document one successful bundled
NFP=2 ``input.nfp2_QI`` lane.  A seed-robust QI claim is deferred until the
same constrained objective has been run and visually audited from QI, QP, QH,
QA, and simple non-omnigenous starting boundaries.  For that audit, accept a
row only when the final state satisfies all of: low legacy QI diagnostic,
closed-looking Boozer ``|B|`` contours, mirror ratio at target, acceptable
elongation, ``abs(mean_iota) >= 0.41``, and aspect ratio near the configured
target.

Before launching expensive optimization sweeps, rank available solved seeds
with:

.. code-block:: bash

   PYTHONPATH=. python examples/optimization/audit_qi_seed_suitability.py --quick --csv results/qi_seed_audit.csv

The audit performs no optimization.  It reads solved ``input``/``wout`` pairs
and reports smooth QI, legacy QI, mirror ratio, elongation, aspect ratio, and
mean iota.  Optional reference cases from ``omnigenity_optimization`` are used
when ``OMNIGENITY_OPTIMIZATION_ROOT`` points to that checkout; missing optional
cases are recorded as skipped rather than failing the audit.
The bundled default set includes ``input.QI_stel_seed_3127`` when its matching
``wout_QI_stel_seed_3127.nc`` fixture is present.  On 2026-05-12 this seed
audited as a useful near-axis QI start: smooth/legacy QI were about
``5.0e-2``/``5.0e-2`` before optimization, with mirror ratio already inside
the target and aspect, iota, and elongation still requiring optimization.
The smooth QI diagnostic includes normalized bounce endpoints by default so the
ranked smooth metric samples the same level range as the legacy Goodman-style
branch-shuffle diagnostic; pass ``--no-include-bounce-endpoints`` only for
interior-level ablation studies.
Rows are ranked by the combined smooth-plus-legacy QI score, while engineering
constraint failures are reported separately so a QI-like seed with a fixable
mirror/aspect violation is not hidden behind a non-QI seed that merely satisfies
the engineering constraints.

To turn the audit into a bounded seed-robustness worklist without launching a
full sweep, add a dry-run prefine manifest:

.. code-block:: bash

   PYTHONPATH=. python examples/optimization/audit_qi_seed_suitability.py --quick --prefine-probes plan --prefine-manifest results/qi_seed_audit/prefine_manifest.json

The manifest records top-ranked seeds plus one best-ranked representative from
each available seed family, hard-capped QI-only prefine settings, expected
output files, and exact commands for running one tiny probe at a time.  The
default probe is now a bounded repeated-stage continuation,
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

For the current optional validation plan, including the verified green CI
baseline, family-representative QI probe workflow, VMEC2000/SIMSOPT optional
lanes, and deferred parity gates, see :doc:`optional_validation_plan`.

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
- **MHD stability coefficients**: ``DMerc``, ``DShear``, ``DWell``, ``DCurr``,
  ``DGeod``.
- **Convergence scalars**: ``fsqr``, ``fsqz``, ``fsql``.

Current parity status
---------------------

**Fixed boundary**
  Established for all shipped reference cases.  ``rmnc/zmns`` Fourier
  coefficients agree at ``rtol=1e-6``; derived magnetic-field quantities at
  ``5×10⁻⁵``.  MHD stability coefficients (Mercier terms) agree at ``1e-3``.

**Stellarator-asymmetric (lasym=True)**
  vmec_jax converges to the same tight residuals as lasym=False cases.  No
  VMEC2000 reference files exist for the shipped lasym=True inputs, but
  cross-checks via the manifest sweep confirm per-iteration ``fsq*`` trace
  alignment.  The Boozer input adapter is required to preserve the asymmetric
  geometry/lambda channels (``rmns``, ``zmnc``, ``lmnc``) and magnetic sine
  channels through ``booz_xform_jax`` for QI and LASYM Boozer diagnostics.

**Free boundary**
  vmec_jax produces converged free-boundary equilibria for the bundled CTH-like
  and D3D cases.  Quantitative parity requires ``fetch_assets.py`` for the
  mgrid files.

**Near-zero diagnostics**
  Quantities like ``jdotb`` and Mercier coefficients involve finite-difference
  postprocessing where relative error can be inflated near zero even when both
  codes agree in absolute terms.  See :doc:`jxbforce_mercier` for details.

Per-iteration trace parity
--------------------------

For the highest-fidelity parity (matching VMEC2000 iteration-by-iteration), use
the executable comparator tools:

.. code-block:: bash

   python tools/diagnostics/vmec2000_exec_stage_trace_compare.py \
     --case circular_tokamak --max-iter 10 --single-ns 13

   python tools/diagnostics/parity_sweep_manifest.py --tier smoke

   python tools/diagnostics/wout_compare_axis_mask.py \
     --a /path/to/vmec2000/wout_case.nc \
     --b /path/to/vmec_jax/wout_case.nc \
     --rtol 1e-4 --atol 1e-12

Manifest-driven sweep (fixed + free boundary)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The canonical parity matrix is defined in ``tools/diagnostics/parity_manifest.toml``:

.. code-block:: bash

   python tools/diagnostics/parity_sweep_manifest.py --tier smoke
   python tools/diagnostics/parity_sweep_manifest.py --tier full

The manifest covers: fixed-boundary axisymmetric and non-axisymmetric,
``lasym=False`` and ``lasym=True``, free-boundary axisymmetric and
non-axisymmetric.

Optional VMEC2000 executable checks
-----------------------------------

The default required test suite does not need a local VMEC2000 build.  It uses
bundled ``wout`` references and should be run during routine development with:

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

Optional SIMSOPT formula parity is similarly guarded and targeted:

.. code-block:: bash

   RUN_SIMSOPT_VALIDATION=1 \
   pytest -q tests/test_simsopt_optional_validation.py::test_qh_quasisymmetry_residual_matches_simsopt_wout_formula

The machine-readable list of these bounded parity commands is emitted by:

.. code-block:: bash

   python validation/qi_seed_robustness_plan.py --output results/qi_seed_audit/validation_plan.json

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
