Quickstart
==========

Install directly from PyPI::

  pip install vmec-jax

Run the minimal showcase (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest way to get started is the axisymmetric showcase. It runs a small
suite of input decks, writes a ``wout_*.nc`` for each, and produces plots.  The
optional parity summary uses released VMEC2000 reference ``wout`` fixtures that
are intentionally not tracked in git.  Fetch them first when you want CI-style
validation rather than just generating fresh outputs from the inputs::

  python tools/fetch_assets.py --list
  python tools/fetch_assets.py

For a small free-boundary smoke test that does not require the large asset
bundle, use the bundled ``input.cth_like_free_bdy_lasym_small`` case together
with the tracked ``mgrid_cth_like_lasym_small.nc`` file in ``examples/data``.

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

Run the full test suite (requires released netCDF assets)::

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

``vmec_jax`` writes diagnostic ``wout`` files from the last available state even
when the requested residual tolerance is not reached. These files preserve the
computed geometry, profiles, field diagnostics, and residual traces, and mark
the solver status with ``ier_flag`` plus ``vmec_jax_converged__logical__`` and
``vmec_jax_status``. Treat ``vmec_jax_status = nonconverged`` as a diagnostic
checkpoint rather than a validated equilibrium.

Free-boundary CLI smoke test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a small bundled free-boundary case, run::

  vmec_jax examples/data/input.cth_like_free_bdy_lasym_small

This input references the tracked ``examples/data/mgrid_cth_like_lasym_small.nc``
fixture, so it works in a fresh clone without downloading the large asset
bundle. The resulting ``wout_cth_like_free_bdy_lasym_small.nc`` can be plotted
with::

  vmec_jax --plot examples/data/wout_cth_like_free_bdy_lasym_small.nc

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
released references, fetch the optional WOUT fixtures first::

  python tools/fetch_assets.py --bundle wout-fixtures

  python tools/diagnostics/end_to_end_solve_parity_summary.py --use-input-niter --fast

Drop ``--fast`` and increase ``--max-iter`` for a full parity snapshot (longer runtime).

External VMEC2000 run (optional)
--------------------------------

If you have the VMEC2000 Python extension installed (``vmec`` + ``mpi4py`` +
``netCDF4``), you can run VMEC2000 on an input and compare outputs to released
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

  wref = vj.read_wout("examples/data/wout_shaped_tokamak_pressure.nc")
  print("fsq_total(ref)=", float(wref.fsqr + wref.fsqz + wref.fsql))
  print("fsq_total(new)=", float(wout.fsqr + wout.fsqz + wout.fsql))

For free-boundary decks, prefer the explicit entrypoint::

  import vmec_jax.api as vj

  freeb = vj.run_free_boundary(
      "examples/data/input.cth_like_free_bdy_lasym_small",
      verbose=False,
      use_initial_guess=False,
  )
  wout_freeb = vj.wout_from_fixed_boundary_run(freeb, include_fsq=True)
  print("wb =", float(wout_freeb.wb))
  print("wp =", float(wout_freeb.wp))

Use ``run_fixed_boundary(...)`` if you deliberately want one driver that
accepts either mode. It remains backward compatible and will still dispatch to
the free-boundary path when ``LFREEB = T`` in the input deck.

Simple optimization example
---------------------------

For a VMEC-JAX-only optimization workflow with explicit SIMSOPT-style
objective construction, run::

  PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/QH_optimization.py

The script builds a ``FixedBoundaryVMEC`` object, constructs objective tuples
such as aspect ratio, iota floor, and quasisymmetry residuals, runs
``least_squares_solve``, then shows how to save and plot the resulting
equilibrium.  The companion ``examples/optimization/README.md`` file lists the
recommended standalone examples, sweep/rendering tools, and older comparison
scripts.
