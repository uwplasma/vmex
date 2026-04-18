Quickstart
==========

Run the minimal showcase (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest way to get started is the axisymmetric showcase. It runs a small
suite of bundled inputs, writes a ``wout_*.nc`` for each, produces plots, and
prints a parity summary against bundled VMEC2000 reference ``wout`` files.
If you have a fresh clone, fetch the optional large assets first::

  python tools/fetch_assets.py

By default the showcase uses a parity-first single-grid run (``--single-ns 13``)
and VMEC2000-style per-iteration **screen** output (FSQR/FSQZ/FSQL, RAX, DELT, WMHD)::

  python -m venv .venv
  source .venv/bin/activate
  python -m pip install -e .
  python examples/showcase_axisym_input_to_wout.py --suite

If you want a release-style non-editable install instead::

  python -m pip install .

Run the test suite::

  pytest -q

Run the full test suite (requires netCDF assets)::

  python tools/fetch_assets.py
  RUN_FULL=1 pytest -q

CLI (VMEC2000-style executable)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once installed (or when working from the repo), you can run vmec_jax like the
VMEC2000 executable by pointing it to a single ``input.*`` file::

  vmec_jax examples/data/input.circular_tokamak

Sanity check (verifies the console script is wired to the right interpreter)::

  vmec_jax --help

If the ``vmec_jax`` command is not found or raises ``ModuleNotFoundError``,
install and run via the module entrypoint::

  python -m pip install -e .
  python -m vmec_jax examples/data/input.circular_tokamak

This writes ``wout_circular_tokamak.nc`` next to the input file and prints the
VMEC2000-style per-iteration screen output by default. Use ``--quiet`` to
silence the iteration table, and ``--outdir`` or ``--output`` to control where
the ``wout_*.nc`` file is written. If you only want a short debug run, pass
``--max-iter`` and ``--no-multigrid`` (single grid).

If you want to compare the conservative parity track against the optimized
fixed-boundary CLI-style controller from Python, run::

  python examples/fixed_boundary_driver_tracks.py \
    examples/data/input.circular_tokamak \
    --quiet --json

That example writes two ``wout`` files (parity and optimized) unless you pass
``--no-write-wout``, and it prints a short runtime / ``fsq_total`` comparison
table at the end.

Kernel parity on reference states (solver-free)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To validate intermediate pipeline quantities on *reference* ``wout`` states (no
nonlinear solve), run::

  python tools/diagnostics/pipeline_parity_summary.py

By default this covers the 4-axisymmetric benchmark suite (``circular_tokamak``,
``purely_toroidal_field``, ``shaped_tokamak_pressure``, ``solovev``).

Scalar residual parity (``fsqr/fsqz/fsql``) on reference states
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To compare scalar residuals reconstructed from a reference state against
``wout.fsqr/fsqz/fsql``::

  python tools/diagnostics/getfsq_parity_cases.py --solve-metric

End-to-end solve snapshot
-------------------------

To run a short fixed-boundary solve and compare a few end-to-end outputs against
bundled references::

  python tools/diagnostics/end_to_end_solve_parity_summary.py --use-input-niter --fast

Drop ``--fast`` and increase ``--max-iter`` for a full parity snapshot (longer runtime).

External VMEC2000 run (optional)
--------------------------------

If you have the VMEC2000 Python extension installed (``vmec`` + ``mpi4py`` +
``netCDF4``), you can run VMEC2000 on an input and compare outputs to bundled
references::

  python tools/diagnostics/external_vmec_driver_compare.py --case circular_tokamak

A minimal API sketch (recommended)
----------------------------------

Most users should start from the small public API in ``vmec_jax.api``::

  import vmec_jax.api as vj

  run = vj.run_fixed_boundary(
      "examples/data/input.shaped_tokamak_pressure",
      max_iter=10,
      verbose=True,
  )
  wout = vj.write_wout_from_fixed_boundary_run(
      "wout_shaped_tokamak_pressure_vmec_jax.nc",
      run,
      include_fsq=True,
  )

  # If you only need an in-memory wout object (no file I/O):
  wout_mem = vj.wout_from_fixed_boundary_run(run, include_fsq=True)

  wref = vj.read_wout("examples/data/wout_shaped_tokamak_pressure_reference.nc")
  print("fsq_total(ref)=", float(wref.fsqr + wref.fsqz + wref.fsql))
  print("fsq_total(new)=", float(wout.fsqr + wout.fsqz + wout.fsql))

Simple optimization example
---------------------------

For a minimal VMEC-JAX-only optimization workflow, run::

  python examples/optimization/target_iota_aspect_volume.py --opt-steps 2

This example keeps the boundary DOF set small (``|m|, |n| <= 1``) and targets
equilibrium aspect ratio, total volume, and mean iota on the bundled
``cth_like_fixed_bdy`` fixed-boundary case.
