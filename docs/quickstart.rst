Quickstart
==========

Run the minimal showcase (recommended)
----------------------------------------

The simplest way to get started is the axisymmetric showcase. It runs a small
suite of bundled inputs, writes a ``wout_*.nc`` for each, produces plots, and
prints a parity summary against bundled VMEC2000 reference ``wout`` files::

  python examples/showcase_axisym_input_to_wout.py --suite

Run the test suite::

  pytest -q

Kernel parity on reference states (solver-free)
-------------------------------------------------

To validate intermediate pipeline quantities on *reference* ``wout`` states (no
nonlinear solve), run::

  python examples/validation/pipeline_parity_summary.py

Scalar residual parity (``fsqr/fsqz/fsql``) on reference states
-----------------------------------------------------------------

To compare scalar residuals reconstructed from a reference state against
``wout.fsqr/fsqz/fsql``::

  python examples/validation/getfsq_parity_cases.py --solve-metric

End-to-end solve snapshot
-------------------------

To run a short fixed-boundary solve and compare a few end-to-end outputs against
bundled references::

  python examples/validation/end_to_end_solve_parity_summary.py --use-input-niter

External VMEC2000 run (optional)
--------------------------------

If you have the VMEC2000 Python extension installed (``vmec`` + ``mpi4py`` +
``netCDF4``), you can run VMEC2000 on an input and compare outputs to bundled
references::

  python tools/diagnostics/external_vmec_driver_compare.py --case circular_tokamak

ParaView export (VTK)
---------------------

Export a surface B-field (including ``Bx``, ``By``, ``Bz``, and ``|B|``) and a
field-line trace for ParaView (requires ``netCDF4``)::

  python examples/visualization/vtk_field_and_fieldlines.py examples/data/input.li383_low_res --hi-res --outdir vtk_out

A minimal API sketch (recommended)
----------------------------------

Most users should start from the small public API in ``vmec_jax.api``::

  import vmec_jax.api as vj

  run = vj.run_fixed_boundary(
      "examples/data/input.shaped_tokamak_pressure",
      solver="vmec2000_iter",
      max_iter=30,
      verbose=True,
  )
  wout = vj.write_wout_from_fixed_boundary_run(
      "wout_shaped_tokamak_pressure_vmec_jax.nc",
      run,
      include_fsq=True,
  )

  wref = vj.read_wout("examples/data/wout_shaped_tokamak_pressure_reference.nc")
  print("fsq_total(ref)=", float(wref.fsqr + wref.fsqz + wref.fsql))
  print("fsq_total(new)=", float(wout.fsqr + wout.fsqz + wout.fsql))
