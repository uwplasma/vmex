Code structure
==============

Top-level package layout (selected):

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
- ``vmec_jax/wout.py``: minimal ``wout_*.nc`` reader for regression

The ``examples/`` folder contains user-facing scripts and curated parity demos.
Developer-only diagnostics and research utilities live under ``tools/``:

- ``tools/diagnostics/vmec2000_exec_stage_trace_compare.py``: per-iteration
  VMEC2000 vs vmec_jax parity comparator.
- ``tools/diagnostics/qh_vmec_vs_vmecjax.py``: QH comparison figures.
- ``tools/diagnostics/readme_fsq_trace.py``: README fsq_total traces.

For most scripts, the recommended import surface is the small public API in
``vmec_jax.api``.
