vmec-jax
========

``vmec-jax`` is a clean-room, JAX-native reimplementation of the **VMEC2000**
ideal-MHD equilibrium code for stellarators and tokamaks. It solves fixed- and
free-boundary equilibria with VMEC2000-parity numerics, writes standard
``wout_*.nc`` output, and — unlike the Fortran original — is differentiable
(fixed boundary) and runs on CPUs and GPUs.

Why vmec-jax?
-------------

- **VMEC2000 parity.** The solver ports the VMEC2000 algorithms
  (steepest-descent moment method, 1D radial preconditioner, Richardson time
  stepping, spectral condensation, NESTOR vacuum solve) constant-for-constant.
  Benchmark decks converge in the *same* iteration counts as VMEC2000 and the
  ``wout`` files agree per-variable (see :doc:`performance`).
- **Differentiable.** Gradients of fixed-boundary equilibrium properties
  with respect to boundary shape and profile parameters via implicit
  differentiation of the converged fixed point
  (:mod:`vmec_jax.core.implicit`) — no finite differences, no iteration
  unrolling — validated against central finite differences (see
  :doc:`optimization`). Free-boundary and coil-parameter derivatives are
  not yet supported by the implicit residual (roadmap).
- **Drop-in workflow.** The ``vmec`` command reads VMEC2000 ``input.*``
  namelists and VMEC++-style JSON, prints VMEC2000-format iteration output,
  and writes ``wout_*.nc`` files that load unchanged in simsopt and
  booz_xform.
- **Batteries included.** Built-in plotting (``vmec --plot``), Boozer
  transform (``vmec --booz`` via ``booz_xform_jax``), spline profiles,
  multigrid with hot restart, and typed zero-crash error handling.

Quickstart
----------

.. code-block:: bash

   pip install vmec-jax
   vmec --test                       # bundled QH case: solve + wout + plots
   vmec input.circular_tokamak       # run any VMEC input deck
   vmec --plot wout_circular_tokamak.nc

See :doc:`quickstart` for a full tour, including the Python API and the
Boozer-coordinate workflow.

.. figure:: _static/figures/readme_runtime_compare.png
   :alt: Runtime comparison of vmec_jax against VMEC2000 and VMEC++
   :align: center
   :width: 95%

   Benchmark-suite runtimes: ``vmec_jax`` (cold and warm) versus VMEC2000 and
   VMEC++. Warm (compiled-cache) solves are the relevant number for
   optimization loops; see :doc:`performance` for the full table.

Documentation
-------------

.. only:: not fast

   .. toctree::
      :maxdepth: 1
      :caption: Getting started

      installation
      quickstart

   .. toctree::
      :maxdepth: 1
      :caption: Tutorials

      tutorials

   .. toctree::
      :maxdepth: 1
      :caption: Theory and numerics

      theory
      equations
      algorithms
      architecture

   .. toctree::
      :maxdepth: 1
      :caption: Reference

      api/index
      cli
      input_reference
      wout_reference

   .. toctree::
      :maxdepth: 1
      :caption: Performance and validation

      performance

   .. toctree::
      :maxdepth: 1
      :caption: Developer guide

      optimization
      contributing
      references

.. only:: fast

   Fast doc builds are enabled (``SPHINX_FAST=1``). The full user guide and API
   reference are skipped to keep CI fast.
