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

- Runtime + residual benchmark for a fixed iteration budget (optional, communication-oriented)::

    python examples/validation/benchmark_fixed_boundary_runtime_and_residuals.py --iters 5 --cases circular_tokamak --ns-override 9 --disable-jit --no-warmup

  To include the VMEC2000 executable (if built)::

    python examples/validation/benchmark_fixed_boundary_runtime_and_residuals.py --iters 5 --cases circular_tokamak --ns-override 9 --disable-jit --no-warmup --run-vmec2000 --vmec2000-ns-override 9

The quick flags above keep runs under ~60s. Drop ``--disable-jit``/``--no-warmup`` and increase ``--iters``/``--cases`` for higher-fidelity traces.

External VMEC2000 runs (optional)
---------------------------------

If you have the VMEC2000 Python extension installed (``vmec`` + ``mpi4py`` +
``netCDF4``), you can run VMEC2000 and compare its output to bundled references::

  python tools/diagnostics/external_vmec_driver_compare.py --case circular_tokamak

Per-iteration trace parity (VMEC2000 executable, reduced grid):

::

  python tools/diagnostics/vmec2000_exec_stage_trace_compare.py --case circular_tokamak --max-iter 20 --vmec-nstep 1 --single-ns 17

This uses a reduced grid to stay under ~1 minute; increase ``--max-iter`` or ``--single-ns`` for deeper parity checks.

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

Current observed mismatches (circular_tokamak, 5-iter snapshot):

- ``bmnc`` and LCFS ``|B|`` plots differ at the ~5e-2 relative level.
- ``buco``/``bvco`` are within a few percent; **``jcuru``/``jcurv`` scaling is now
  corrected** (remaining differences are tied to earlier-force parity).
- ``betapol``, ``betator``, ``betaxis``, ``ctor``, and ``DMerc`` are present but
  still placeholders in ``vmec_jax`` (zeros) until the VMEC2000 diagnostics path
  is fully ported.

On ``shaped_tokamak_pressure`` (20-iter snapshot), the dominant gaps are:

- ``bmnc`` ~1e-2 relative, ``buco`` ~3e-3, ``bvco`` ~6e-4 (good agreement).
- ``jcuru``/``jcurv`` scaling is corrected; residual differences track the same
  lambda/force-kernel mismatches seen in the internal scan.
- ``pres/presf`` differ at the ~0.24 relative level (profile staging mismatch).
- ``rmnc/zmns`` differ at the ~1e-2 level (geometry still drifting in the nonlinear loop).

These mismatches are now tracked explicitly so we can converge the diagnostics
in step with the force/iteration parity work.

Scope and known gaps
--------------------

The primary parity target is fixed-boundary, axisymmetric VMEC2000. Items
explicitly deferred for now include:

- free boundary,
- ``lasym=True`` (up-down / non-stellarator-symmetric),
- non-axisymmetric end-to-end nonlinear solve parity (``ntor>0`` and/or ``nfp>1``),
- parallelization and multi-device execution.

Current blockers worth tracking:

- ``lasym=True`` axisymmetric case (``input.up_down_asymmetric_tokamak``) shows large bcovar/force-kernel mismatches at iter 1.
- ``purely_toroidal_field`` multigrid trace matches early iterations but the ``r00``/``WMHD`` diagnostics become non-finite at later iterations in ``vmec_jax``.
- Axisymmetric internal scans now match VMEC2000 for R/Z force blocks, but the first mismatch appears in the lambda block at iter 1:
  ``flsc`` (~0.36 rel), ``gcl`` (~0.50 rel), and the lambda-force kernel ``blmn`` (~0.68 rel). This is the current top
  blocker for nonlinear trace parity. Recent change: ``lvv`` now uses ``phipog=1/sqrtg`` (no ``2π`` factor), which
  reduced the mismatch from ~6x; remaining scaling still under investigation.
- Axisymmetric nonlinear traces still diverge from VMEC2000 after the first few iterations on some cases (e.g. ``shaped_tokamak_pressure``); the next focus is matching the lambda-force path and VMEC2000 time-step/preconditioner updates exactly.
