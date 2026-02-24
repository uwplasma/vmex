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

- 3D stellarator-symmetric cases (used mainly for kernel validation; nonlinear solve parity deferred):

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

Current observed mismatches (updated parity status):

- **Single-grid axisym parity** (`--single-ns 13`) matches VMEC2000 at machine
  precision for the first **10 iterations** on the axisymmetric benchmark
  suite (``circular_tokamak``, ``shaped_tokamak_pressure``, ``solovev``).
- **Non-axisymmetric kernel parity** remains the top gap: ``bsub*`` and
  ``abs(B)`` comparisons for 3D reference states are still off by O(1).
- **Full-grid QA/QH traces** now match VMEC2000 at `rtol=1e-4`,
  `atol=1e-12` for 100-iteration runs.
- **Scan (jax.lax.scan) VMEC2000 control path** can diverge on large-`ns`
  inputs (e.g., ``input.QI_nfp2``). For parity runs, the default is now the
  non-scan loop; enable scan explicitly with ``--fast`` when you only care
  about speed.
- ``betapol``/``betator`` now match to tight tolerance on converged QA/QI runs.
- ``jdotb``, ``bsubsmns``, and Mercier terms (``DMerc``/``DGeod``) still show
  case-dependent drift away from VMEC2000, even when comparing away from the
  axis (skip first 6 radial points).
- Force-iteration traces (``fsqr/fsqz/fsql``) and global convergence behavior
  are much closer than some derived post-processing diagnostics.

QI/QA ``wout`` parity snapshot (converged, skip first 6 radial points,
``rtol=1e-4``, ``atol=1e-12``):

.. list-table::
   :header-rows: 1
   :widths: 18 10 16 16 16 16

   * - Case
     - NS
     - betapol max_rel
     - betator max_rel
     - DMerc max_rel
     - jdotb max_rel
   * - ``input.QI_nfp2``
     - 31
     - ``1.482e-06``
     - ``2.053e-06``
     - ``1.238e-02``
     - ``4.112e+01``
   * - ``input.LandremanPaul2021_QA_lowres``
     - 50
     - ``0.000e+00``
     - ``0.000e+00``
     - ``5.886e-02``
     - ``2.310e+00``

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

- **Non-scan (default)**: Python control flow + JAX kernels. This is the
  reference for per-iteration parity.
- **Scan (``--fast``)**: The loop is lifted into ``jax.lax.scan`` for speed.
  This path is still parity-sensitive and can diverge for large-`ns` inputs.

For parity-critical runs (including ``input.QI_nfp2``), keep the default
non-scan loop. Use ``--fast`` only when parity is not required.

Full-grid parity snapshot (VMEC2000 exec comparator, `--use-input-niter`, `max_iter=100`,
`rtol=1e-4`, `atol=1e-12`):

.. list-table::
   :header-rows: 1
   :widths: 16 36 14 18 16 18

   * - Case
     - Input
     - Status
     - fsq_total (VMEC/JAX)
     - runtime_s (vmec2000/jax)
     - Notes
   * - shaped_tokamak_pressure
     - ``examples/data/input.shaped_tokamak_pressure``
     - PASS
     - ``1.422e-07 / 1.422e-07``
     - ``0.213 / 5.552``
     - Axisymmetric
   * - QA signgs1
     - ``input.qa_signgs1``
     - PASS
     - ``1.412e-04 / 1.412e-04``
     - ``0.443 / 5.569``
     - 3D fixed boundary
   * - QH warm start
     - ``examples/data/input.nfp4_QH_warm_start``
     - PASS
     - ``2.888e-07 / 2.888e-07``
     - ``0.272 / 5.130``
     - 3D fixed boundary

VMEC++ bad-progress restart check (QA_lowres)
---------------------------------------------

Using ``input.LandremanPaul2021_QA_lowres`` with multigrid enabled, we compared
the fine-grid (last stage) **iteration 1** ``fsq_total`` with and without the
VMEC++ bad-progress restart flag:

.. list-table::
   :header-rows: 1
   :widths: 24 24 24

   * - Run
     - fine-grid iter-1 fsq_total
     - Notes
   * - baseline (vmecpp_restart=False)
     - ``2.412e-04``
     - stage offsets [0, 189, 893]
   * - vmecpp_restart=True
     - ``2.412e-04``
     - no change at iter-1; VMEC++ trigger occurs later in the stage

Scope and known gaps
--------------------

The primary parity target is fixed-boundary VMEC2000 parity. Items explicitly
deferred for now include:

- free boundary,
- ``lasym=True`` (up-down / non-stellarator-symmetric),
- remaining non-axisymmetric cases beyond the current QA/QH/n3are/QA-lowres
  sweep (``ntor>0`` and/or ``nfp>1``),
- parallelization and multi-device execution.

Current blockers worth tracking:

- ``lasym=True`` axisymmetric case (``input.up_down_asymmetric_tokamak``) shows large bcovar/force-kernel mismatches at iter 1.
- Lambda-path internal parity (``flsc``/``gcl`` and ``blmn``/``clmn``) matches VMEC2000
  to ~1e-10 abs on reduced grids with full dumps; remaining work is to validate the
  same at higher resolution and across more ``lasym=True`` cases.
