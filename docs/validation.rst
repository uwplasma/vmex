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

Recommended validation scripts
------------------------------

All of the following scripts are designed to run quickly on bundled data:

- Pipeline parity snapshot (solver-free)::

    python examples/validation/pipeline_parity_summary.py

- Scalar residual parity (solver-free, reference states)::

    python examples/validation/getfsq_parity_cases.py --solve-metric

- End-to-end solve snapshot (short solve, compares a few end-to-end outputs)::

    python examples/validation/end_to_end_solve_parity_summary.py --use-input-niter --fast

- Runtime + residual benchmark for a fixed iteration budget (communication-oriented)::

    python examples/validation/benchmark_fixed_boundary_runtime_and_residuals.py \
      --iters 20 \
      --cases circular_tokamak shaped_tokamak_pressure solovev purely_toroidal_field \
      --ns-override 17

  To include the VMEC2000 executable (if built)::

    python examples/validation/benchmark_fixed_boundary_runtime_and_residuals.py \
      --iters 20 \
      --cases circular_tokamak shaped_tokamak_pressure solovev purely_toroidal_field \
      --ns-override 17 \
      --run-vmec2000 --vmec2000-ns-override 17 --vmec2000-timeout 60 \
      --no-vmec2000-use-input-niter

The quick flags above keep runs under ~60s. Drop ``--disable-jit``/``--no-warmup`` and increase ``--iters``/``--cases`` for higher-fidelity traces.

External VMEC2000 runs (optional)
---------------------------------

If you have the VMEC2000 Python extension installed (``vmec`` + ``mpi4py`` +
``netCDF4``), you can run VMEC2000 and compare its output to bundled references::

  python tools/diagnostics/external_vmec_driver_compare.py --case circular_tokamak

Per-iteration trace parity (VMEC2000 executable, reduced grid):

::

  python tools/diagnostics/vmec2000_exec_stage_trace_compare.py --case circular_tokamak --max-iter 10 --vmec-nstep 1 --single-ns 17
  python tools/diagnostics/vmec2000_exec_stage_trace_compare.py --case nfp4_QH_warm_start --max-iter 10 --single-ns 16 --vmec-timeout 60 --rtol 1e-3

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

  python tools/diagnostics/vmec2000_exec_internal_scan.py --case circular_tokamak --single-ns 17 --iter-start 1 --iter-stop 5

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
  precision for the first **15 iterations** on ``circular_tokamak`` and
  ``shaped_tokamak_pressure``. When split into two 15-iter phases (warm start),
  the restart resets time-step/momentum state and the first mismatch appears at
  iteration 16. Continuous 30-iter parity at ``--single-ns 13`` remains pending
  under the 60s cap.
- **Multigrid parity** (full `NS` from input, `--use-input-niter`) now matches
  per-iteration traces on ``circular_tokamak`` and ``shaped_tokamak_pressure``
  for a 10-iteration cap; 20-iter multigrid traces are now generated for the
  benchmark figures (reduced ns for runtime).
- **Non-axisymmetric early-iteration parity** now reproduces VMEC2000 on
  ``nfp4_QH_warm_start`` for the first 10 iterations (`--single-ns 16`):
  per-iteration ``fsq*`` traces, ``xc/v`` updates, and tomnsps/gc raw blocks
  (including lambda-path ``flsc/gcl`` and ``blmn/clmn``) match within the
  configured tolerance after applying VMEC's lambda axis closure in ``bcovar``.
- ``betapol``, ``betator``, ``betaxis``, ``ctor``, and ``DMerc`` are present but
  still placeholders in ``vmec_jax`` (zeros) until the VMEC2000 diagnostics path
  is fully ported.

Scope and known gaps
--------------------

The primary parity target is fixed-boundary, axisymmetric VMEC2000. Items
explicitly deferred for now include:

- free boundary,
- ``lasym=True`` (up-down / non-stellarator-symmetric),
- non-axisymmetric full multigrid end-to-end nonlinear solve parity
  (``ntor>0`` and/or ``nfp>1``),
- parallelization and multi-device execution.

Current blockers worth tracking:

- ``lasym=True`` axisymmetric case (``input.up_down_asymmetric_tokamak``) shows large bcovar/force-kernel mismatches at iter 1.
- ``purely_toroidal_field`` multigrid trace matches early iterations but the ``r00``/``WMHD`` diagnostics become non-finite at later iterations in ``vmec_jax``.
- Lambda-path internal parity (``flsc``/``gcl`` and ``blmn``/``clmn``) matches VMEC2000
  to ~1e-10 abs on reduced grids with full dumps; remaining work is to validate the
  same at higher resolution.
