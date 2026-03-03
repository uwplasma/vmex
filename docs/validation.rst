Validation and regression testing
=================================

``vmec-jax`` is developed using a parity-first workflow: whenever possible,
intermediate quantities are validated against bundled **VMEC2000**
``wout_*.nc`` reference outputs before tightening nonlinear iteration parity.

Bundled regression cases
------------------------

The repository includes a small set of low-resolution cases under
``examples/data/`` used by tests and examples:

- Axisymmetric tokamak sanity cases (fixed boundary):

  - ``input.circular_tokamak`` + ``wout_circular_tokamak_reference.nc``
  - ``input.shaped_tokamak_pressure`` + ``wout_shaped_tokamak_pressure_reference.nc``
  - ``input.solovev`` + ``wout_solovev_reference.nc``

- 3D stellarator-symmetric fixed-boundary cases:

  - ``input.li383_low_res`` + ``wout_li383_low_res_reference.nc``
  - ``input.n3are_R7.75B5.7_lowres`` + ``wout_n3are_R7.75B5.7_lowres.nc``

Additional files may be present for future parity work; the automated test suite
is intentionally kept small to keep runtime reasonable.

What is validated today
-----------------------

The test suite in ``tests/`` focuses on:

- correct INDATA parsing and boundary evaluation,
- geometry and metric/Jacobian sanity checks,
- regression comparisons against bundled ``wout`` files for selected quantities,
- scalar residual parity (``fsqr/fsqz/fsql``) on reference ``wout`` states,
- end-to-end smoke tests for the fixed-boundary solvers on small axisymmetric cases.
- optional VMEC2000 executable parity (QA signgs1) when ``VMEC2000_INTEGRATION=1`` is set.

Recommended validation scripts
------------------------------

All of the following scripts are designed to run quickly on bundled data:

Manifest-driven sweep (fixed + free boundary)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The canonical parity matrix now lives in:

- ``tools/diagnostics/parity_manifest.toml``

The manifest includes representative cases across:

- fixed-boundary axisymmetric and non-axisymmetric,
- ``lasym=False`` and ``lasym=True``,
- free-boundary axisymmetric and non-axisymmetric.

Run the manifest sweep runner:

::

  python tools/diagnostics/parity_sweep_manifest.py --tier smoke
  python tools/diagnostics/parity_sweep_manifest.py --tier full
  python tools/diagnostics/parity_sweep_manifest.py --ids freeb_nonaxis_lasym_false_cth_like

Dry-run (print commands only):

::

  python tools/diagnostics/parity_sweep_manifest.py --tier smoke --dry-run

Outputs (logs + JSON summary) are written under:

- ``outputs/parity_sweeps/<timestamp>/``

Each case directory stores comparator logs and, for free-boundary cases,
per-iteration scalpot comparator JSON payloads.
Free-boundary cases can also define quantitative pass/fail thresholds in the
manifest via ``metric_thresholds_rel_scaled`` (for keys like
``source_sym``, ``bvec_nonsing_fouri``, ``amatrix``, ``potvac``), so the sweep
fails on metric drift even when command return code is zero. For turn-on /
restart-sensitive phases, per-iteration thresholds are available via
``metric_thresholds_rel_scaled_by_iter``.

- Pipeline parity snapshot (solver-free)::

    python tools/diagnostics/pipeline_parity_summary.py \
      --cases circular_tokamak shaped_tokamak_pressure solovev \
      n3are_R7.75B5.7_lowres LandremanPaul2021_QA_lowres li383_low_res

- Scalar residual parity (solver-free, reference states)::

    python tools/diagnostics/getfsq_parity_cases.py --solve-metric

- End-to-end solve snapshot (short solve, compares a few end-to-end outputs)::

    python tools/diagnostics/end_to_end_solve_parity_summary.py --use-input-niter --fast

- Axis-masked ``wout`` comparator (for converged runs, skip near-axis points)::

    python tools/diagnostics/wout_compare_axis_mask.py \
      --a /path/to/vmec2000/wout_case.nc \
      --b /path/to/vmec_jax/wout_case.nc \
      --axis-skip 6 --rtol 1e-4 --atol 1e-12

- Runtime + residual benchmark for a fixed iteration budget (communication-oriented)::

    python tools/diagnostics/benchmark_fixed_boundary_runtime_and_residuals.py \
      --iters 10 \
      --cases circular_tokamak shaped_tokamak_pressure solovev purely_toroidal_field \
      --run-vmec2000 --vmec2000-timeout 60

  To run at higher resolution::

    python tools/diagnostics/benchmark_fixed_boundary_runtime_and_residuals.py \
      --iters 20 \
      --cases circular_tokamak shaped_tokamak_pressure solovev purely_toroidal_field \
      --ns-override 17 \
      --run-vmec2000 --vmec2000-ns-override 17 --vmec2000-timeout 60 \
      --no-vmec2000-use-input-niter

The parity-first defaults keep runs under ~60s per case. Increase ``--iters`` and/or ``--ns-override`` for longer traces.

External VMEC2000 runs (optional)
---------------------------------

If you have the VMEC2000 Python extension installed (``vmec`` + ``mpi4py`` +
``netCDF4``), you can run VMEC2000 and compare its output to bundled references::

  python tools/diagnostics/external_vmec_driver_compare.py --case circular_tokamak

Per-iteration trace parity (VMEC2000 executable, reduced grid):

::

  python tools/diagnostics/vmec2000_exec_stage_trace_compare.py --case circular_tokamak --max-iter 10 --vmec-nstep 1 --single-ns 13
  python tools/diagnostics/vmec2000_exec_stage_trace_compare.py --case nfp4_QH_warm_start --max-iter 10 --single-ns 16 --vmec-timeout 60 --rtol 1e-4 --atol 1e-12
  python tools/diagnostics/nonaxis_parity_batch.py --max-cases 8 --single-ns 13 --max-iter 1 --vmec-timeout 60

README trace figures (axisym + QH, 100 iterations)::

  python tools/diagnostics/readme_fsq_trace.py \
    --axisym-input examples/data/input.shaped_tokamak_pressure \
    --qh-input examples/data/input.nfp4_QH_warm_start \
    --niter 100 \
    --outdir docs/_static/figures

This uses a reduced grid to stay under ~1 minute; increase ``--max-iter`` or ``--single-ns`` for deeper parity checks.
For longer traces under the timeout cap you can split the vmec_jax run::

  python tools/diagnostics/vmec2000_exec_stage_trace_compare.py --case circular_tokamak --max-iter 30 --split-iter 15 --single-ns 13 --vmec-nstep 1 --vmec-timeout 60

The comparator now consumes VMEC2000 scalar/force dumps to match full-precision
``fsq*`` values and cross-checks ``include_edge``/``zero_m1`` gating.
The trace comparator also dumps VMEC2000 ``tomnsps_kernels`` and vmec_jax
``force_kernels`` to compare ``blmn/clmn`` (lambda-force full-mesh inputs) with
per-index reporting of the first mismatch.

Internal force-block parity scan (tomnsps + gc, executable):

::

  python tools/diagnostics/vmec2000_exec_internal_scan.py --case circular_tokamak --single-ns 13 --iter-start 1 --iter-stop 5

This dumps internal force blocks per iteration and stops at the first mismatch beyond tolerance.

VMECPlot2 compatibility (wout completeness)
-------------------------------------------

``vmec_jax`` now writes **NetCDF3-classic** ``wout_*.nc`` files with the fields
required by ``vmecPlot2.py``. This enables side-by-side figure generation using
the legacy VMEC plotting script:

::

  # vmec_jax output (short run)
  python examples/showcase_axisym_input_to_wout.py \
    --case circular_tokamak --max-iter 5 --no-vmec2000-trace

  # vmecPlot2 figures
  python vmecPlot2.py examples/outputs/showcase/circular_tokamak/wout_circular_tokamak_vmec_jax.nc /tmp/vmecplot2_jax
  python vmecPlot2.py examples/data/wout_circular_tokamak_reference.nc /tmp/vmecplot2_ref

The in-repo showcase plots now use the same VMECPlot2-style grids (theta/zeta
resolution and toroidal angle conventions) so figure-to-figure comparisons are
faithful to the legacy script.

Current parity status (high-level)
---------------------------------

- Fixed-boundary parity is established for axisymmetric and non-axisymmetric
  cases, including ``lasym=False`` and ``lasym=True``, in the VMEC2000
  executable comparator workflow.
- Per-iteration scalar histories (``fsqr/fsqz/fsql`` and preconditioned
  ``fsq*1`` channels) and key end-state ``wout`` fields are aligned to the
  current project tolerance target (typically ``rtol=1e-3``; tighter in many
  channels).
- For cancellation-limited post-processing diagnostics (notably ``jdotb`` and
  Mercier terms), interpretation depends strongly on whether the underlying
  physical signal is expected to be small. For currentless / vacuum-like cases,
  relative comparisons of ``jdotb`` can be misleading.

See :doc:`jxbforce_mercier` for a detailed explanation of the VMEC2000
conventions used in ``jxbforce.f`` / ``mercier.f`` and for profile comparisons
between VMEC2000 and ``vmec_jax``.

Axis reset and bad-Jacobian parity notes
----------------------------------------

VMEC2000 resets the magnetic axis *before* iteration 1 if the half-mesh
Jacobian changes sign. To match this behavior, the vmec-jax VMEC2000 loop now:

- preflights the Jacobian sign using both VMEC-style ``ptau`` and the
  state-based Jacobian,
- triggers the axis reset before the first iteration (no duplicate ``iter=1``),
- preserves the VMEC2000 ``ijacob`` count and checkpoint state.

This eliminates spurious restarts and aligns the ``zero_m1`` gating with
VMEC2000 (including the ``fsqz_prev < 1e-6`` condition used later in long runs).

Scan vs non-scan parity notes
-----------------------------

The VMEC2000 parity loop has two implementations:

- **Scan fast path (default)**: the loop is lifted into ``jax.lax.scan`` for
  lower Python overhead.
- **Non-scan parity path**: conservative reference path (``--parity``) that
  mirrors VMEC2000 control flow step-by-step.

The runtime selects scan by default and can fall back to the non-scan path
when parity guards detect drift on difficult stages.

Latest fixed-boundary executable parity pass
--------------------------------------------

A final VMEC2000 executable sweep (axis-masked ``wout`` comparison with
``--radial-skip 6 --radial-drop-edge``) was run over representative bundled
inputs:

- axisymmetric: ``circular_tokamak``, ``shaped_tokamak_pressure``,
  ``solovev``, ``ITERModel``,
- non-axisymmetric ``lasym=False``: ``LandremanPaul2021_QA_lowres``,
  ``nfp4_QH_warm_start``,
- non-axisymmetric ``lasym=True``: ``up_down_asymmetric_tokamak``,
  ``basic_non_stellsym_pressure``.

Observed behavior:

- per-iteration VMEC scalar traces (``fsqr/fsqz/fsql``, ``fsq*1``, ``delt``)
  remain aligned in the comparator workflow,
- primary magnetic geometry channels (``rmnc/zmns/bmnc/bsubs``) are within the
  fixed-boundary project target for these runs,
- cancellation-limited post-processed channels (especially ``jdotb``) can show
  inflated relative error in currentless/vacuum-like regimes despite small
  absolute differences.

The long-standing n3are stress case is still tracked as a dedicated fixed-
boundary diagnostic outlier for some post-processed channels; this does not
change the project scope decision that the next major implementation target is
free-boundary parity.

Scope and known gaps
--------------------

The primary parity target for fixed-boundary solves is complete. The remaining
scope gap is:

- **free-boundary VMEC parity** (vacuum coupling and moving-boundary updates).

Operational caveat (known and accepted):

- For near-axis and near-zero diagnostics, relative errors can be dominated by
  denominator noise; use axis masking and absolute/physics-aware checks.
