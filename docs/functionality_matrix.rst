Functionality and support matrix
================================

This page is the release authority for what ``vmec-jax`` supports. A feature
is **supported** only when it has a documented public entry point, regression
tests, and a validated example or benchmark. **Research** means the API is
available, but a stated numerical or physics gate has not passed. **Deferred**
means the attempted formulation and its evidence are retained, but it is not a
supported user workflow.

The comparison was audited against VMEC2000 commit ``512375ce`` and VMEC++
commit ``378b4ff``. VMEC2000 entries refer to the solver distributed in
STELLOPT; separate programs such as ANIMEC and BOOZ_XFORM are labelled
``external`` rather than treated as built-in VMEC functionality.

Core equilibrium capabilities
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 31 15 15 15 24

   * - Capability
     - vmec-jax
     - VMEC2000
     - VMEC++
     - vmec-jax implementation
   * - Fixed-boundary toroidal equilibrium
     - Supported
     - Supported
     - Supported
     - :func:`vmec_jax.core.solver_driver.solve`
   * - Stellarator-asymmetric geometry (``LASYM``)
     - Supported
     - Supported
     - Not supported
     - :mod:`vmec_jax.core.fourier`, :mod:`vmec_jax.core.transforms`
   * - Prescribed iota and prescribed current (``NCURR=0/1``)
     - Supported
     - Supported
     - Supported
     - :mod:`vmec_jax.core.profiles`, :mod:`vmec_jax.core.fields`
   * - Finite pressure and finite toroidal current
     - Supported
     - Supported
     - Supported
     - :mod:`vmec_jax.core.profiles`, :mod:`vmec_jax.core.setup`
   * - Multigrid ``NS_ARRAY`` ladder
     - Supported
     - Supported
     - Supported
     - :func:`vmec_jax.core.multigrid.solve_multigrid`
   * - Reusable converged-state hot restart
     - Supported
     - Internal run restart
     - Supported
     - :func:`vmec_jax.core.multigrid.solve_multigrid`
   * - VMEC radial preconditioner
     - Supported, default
     - Supported, default
     - Supported
     - :mod:`vmec_jax.core.preconditioner`
   * - 2-D block/Newton preconditioner
     - Supported, opt-in
     - Supported, opt-in
     - Not supported
     - :mod:`vmec_jax.core.preconditioner_2d`
   * - CPU and GPU execution
     - Supported
     - CPU
     - CPU
     - :mod:`vmec_jax.core.device`

Free boundary and external fields
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 31 15 15 15 24

   * - Capability
     - vmec-jax
     - VMEC2000
     - VMEC++
     - vmec-jax implementation
   * - NESTOR free boundary from an ``mgrid`` file
     - Supported
     - Supported
     - Supported for ``ntor>0``
     - :func:`vmec_jax.core.freeboundary.solve_free_boundary`
   * - Axisymmetric (``ntor=0``) free boundary
     - Supported
     - Supported
     - Not supported
     - :mod:`vmec_jax.core.freeboundary`
   * - Free boundary directly from filament coils
     - Supported
     - Requires MAKEGRID
     - Requires MAKEGRID
     - :mod:`vmec_jax.core.coils`, :mod:`vmec_jax.core.freeboundary`
   * - Missing-``mgrid`` fixed-boundary fallback
     - Supported
     - Supported
     - Error
     - :meth:`vmec_jax.core.input.VmecInput.from_file`
   * - Solved-LCFS sensitivities for a few coil-current groups
     - Supported, forward
     - Not supported
     - Not supported
     - :class:`vmec_jax.core.freeboundary_implicit.CoupledFreeBoundaryProblem`
   * - Simultaneous plasma-boundary and coil virtual-casing gradient
     - Research
     - Not supported
     - Not supported
     - :mod:`vmec_jax.core.freeboundary_diff`
   * - Many-parameter reverse derivative of the NESTOR fixed point
     - Deferred
     - Not supported
     - Not supported
     - See :ref:`free-boundary-derivative-scope`

Inputs, outputs, and diagnostics
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 31 15 15 15 24

   * - Capability
     - vmec-jax
     - VMEC2000
     - VMEC++
     - vmec-jax implementation
   * - VMEC ``INDATA`` namelist
     - Supported
     - Supported
     - Supported
     - :class:`vmec_jax.core.input.VmecInput`
   * - VMEC++-schema JSON
     - Supported
     - Not supported
     - Supported
     - :class:`vmec_jax.core.input.VmecInput`
   * - Power, two-power, pedestal, line, cubic, and Akima profiles
     - Supported
     - Supported
     - Supported
     - :func:`vmec_jax.core.profiles.evaluate_profile`
   * - Standard NetCDF ``wout``
     - Supported
     - Supported
     - Supported
     - :mod:`vmec_jax.core.wout`
   * - VMEC-format iteration and summary output
     - Supported
     - Supported
     - Similar output
     - :mod:`vmec_jax.core.printing`
   * - Mercier, well, current, force, and beta diagnostics
     - Supported
     - Supported
     - Supported subset
     - :mod:`vmec_jax.core.nyquist`, :mod:`vmec_jax.core.postprocess`
   * - Built-in plotting
     - Supported
     - External tools
     - External tools
     - :mod:`vmec_jax.core.plotting`
   * - Built-in Boozer transform command
     - Supported
     - External BOOZ_XFORM
     - External BOOZ_XFORM
     - :func:`vmec_jax.core.boozer.run_booz_xform`
   * - Typed exceptions instead of process termination
     - Supported
     - Not supported
     - Supported
     - :mod:`vmec_jax.core.errors`

Differentiation, optimization, and extended physics
---------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 31 15 15 15 24

   * - Capability
     - vmec-jax
     - VMEC2000
     - VMEC++
     - vmec-jax implementation
   * - Fixed-boundary implicit differentiation
     - Supported
     - Not supported
     - Not supported
     - :mod:`vmec_jax.core.implicit`
   * - Boundary/profile/current optimization
     - Supported
     - External tools
     - External tools
     - :mod:`vmec_jax.core.optimize`
   * - Quasisymmetry and omnigenity objectives
     - Supported
     - External tools
     - External tools
     - :mod:`vmec_jax.core.optimize`, :mod:`vmec_jax.core.omnigenity`
   * - Redl bootstrap-current iteration
     - Supported
     - External tools
     - External tools
     - :mod:`vmec_jax.core.bootstrap`
   * - Infinite-n ballooning and turbulence proxy objectives
     - Supported
     - External tools
     - External tools
     - :mod:`vmec_jax.core.stability`, :mod:`vmec_jax.core.turbulence`
   * - ANIMEC anisotropic toroidal equilibrium
     - Not supported
     - External ANIMEC
     - Not supported
     - Deliberately outside the toroidal release contract
   * - Reversed-field-pinch ``LRFP`` mode
     - Not supported
     - Supported
     - Not supported
     - Deferred niche VMEC2000 mode
   * - MPI/PARVMEC domain decomposition
     - Not supported
     - Supported
     - Not supported
     - JAX device execution is the parallel path
   * - V3FIT reconstruction hooks
     - Not supported
     - Supported in STELLOPT build
     - Not supported
     - Deliberately outside the equilibrium library

Mirror and hybrid geometry
--------------------------

These rows are vmec-jax extensions and have no VMEC2000 or VMEC++ analogue.
The exact numerical contracts and failed promotion gates are in
:doc:`mirror_geometry`.

.. list-table::
   :header-rows: 1
   :widths: 35 18 47

   * - Capability
     - Status
     - Implementation and limit
   * - Axisymmetric straight mirror, fixed boundary
     - Supported
     - :func:`vmec_jax.mirror.solve_fixed_boundary_cli`; component-wise
       ``ftol=1e-12`` contract
   * - Axisymmetric straight mirror, free boundary
     - Supported
     - :func:`vmec_jax.mirror.solve_free_boundary_cli`; finite-beta scans
       through requested beta 50%
   * - Axisymmetric anisotropic mirror closures
     - Supported
     - :class:`vmec_jax.mirror.BiMaxwellianPressureClosure` and
       :class:`vmec_jax.mirror.TabulatedPressureClosure`
   * - Straight-mirror fixed/free implicit derivatives
     - Supported in axisymmetry
     - :mod:`vmec_jax.mirror.implicit` and
       :mod:`vmec_jax.mirror.free_boundary_implicit`
   * - Nonaxisymmetric straight mirror
     - Research
     - Local Fourier-mode refinement did not pass; no production claim
   * - Toroidal Fourier stellarator-mirror hybrid
     - Supported within measured envelope
     - Ordinary VMEC backend; free-boundary continuation through achieved
       beta 0.8333%
   * - Native spline-state toroidal hybrid
     - Deferred
     - Geometry initialization exists; no independent spline equilibrium
       state is claimed

Code and validation footprint
-----------------------------

The 2026-07-13 release candidate contains 72 Python files and 33,960
physical lines, including a 20-file, 8,072-line mirror backend. Every core
module is at most 999 lines and the public-definition audit reports 0 missing
docstrings across 569 definitions. The 60 tracked test modules contain 14,197
physical lines and collect 719 tests. This is an evidence-preserving test
budget, not production source bloat: the tracked checkout is about 9 MB and
generated results are ignored. Reproducible counts and validation commands
are recorded in ``benchmarks/mirror_m10_audit.json``.
