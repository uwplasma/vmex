Code structure
==============

Top-level package layout (selected):

- ``vmec_jax/api.py``: stable public import surface for solve, I/O, plotting,
  diagnostics, and documented optimization workflow objects
- ``vmec_jax/namelist.py``: minimal Fortran namelist reader (``&INDATA``)
- ``vmec_jax/config.py``: run discretization config extracted from inputs
- ``vmec_jax/static.py``: static grids + Fourier basis tensors (PyTrees)
- ``vmec_jax/state.py``: coefficient containers (PyTrees)
- ``vmec_jax/boundary.py``: boundary coefficient parsing + VMEC m=1 constraint
- ``vmec_jax/init_guess.py``: profil3d-style initial guess (axis + boundary)
- ``vmec_jax/coords.py`` / ``vmec_jax/geom.py``: geometry and metric kernels
- ``vmec_jax/field.py`` / ``vmec_jax/energy.py``: B-field and energy diagnostics
- ``vmec_jax/vmec_tomnsp.py``: VMEC ``fixaray`` tables + DFT tomnsps kernels
- ``vmec_jax/vmec_bcovar.py``: bcovar/metric assembly and half-mesh pipeline
- ``vmec_jax/preconditioner_1d.py``: VMEC-style preconditioner operators
- ``vmec_jax/solve.py``: fixed-boundary solvers + VMEC2000 iteration loop
- ``vmec_jax/driver.py``: CLI-facing fixed/free-boundary drivers, output
  policies, staged solve dispatch, and wout writing
- ``vmec_jax/free_boundary.py``: mgrid loading, NESTOR-like vacuum coupling,
  and free-boundary runtime state helpers
- ``vmec_jax/optimization.py``: exact fixed-boundary optimizer, boundary DOF
  maps, accepted-point replay, and discrete-adjoint Jacobian plumbing
- ``vmec_jax/optimization_workflow.py``: user-facing optimization problem
  objects, objective tuples, continuation stages, and example workflow helpers
- ``vmec_jax/qi_diagnostics.py``: smooth and legacy QI diagnostics, seed
  ranking metadata, mirror/elongation gates, and acceptance annotations
- ``vmec_jax/mirror/``: experimental open-ended mirror geometry domain package;
  first phases keep grids, bases, kernels, solvers, I/O, plotting, validation,
  and optimization under this package rather than adding root-level
  ``mirror_*`` helpers
- ``vmec_jax/plotting.py``: VMEC-style geometry, ``|B|`` contour, Boozer-grid,
  objective-history, and publication-panel plotting helpers
- ``vmec_jax/wout.py``: minimal ``wout_*.nc`` reader for regression

The ``examples/`` folder contains user-facing scripts and curated parity demos.
Developer-only diagnostics and research utilities live under ``tools/``:

- ``tools/diagnostics/vmec2000_exec_stage_trace_compare.py``: per-iteration
  VMEC2000 vs vmec_jax parity comparator.
- ``tools/diagnostics/qh_vmec_vs_vmecjax.py``: QH comparison figures.
- ``tools/diagnostics/readme_fsq_trace.py``: README fsq_total traces.

For most scripts, prefer ``import vmec_jax as vj`` or ``import vmec_jax.api as
vj``.  Import lower-level modules directly only when developing kernels,
validation tools, or tests that need implementation-specific behavior.
