vmec-jax documentation
======================

``vmec-jax`` is an incremental, JAX-based rewrite of **VMEC2000**, targeting:

- VMEC2000 parity for fixed-boundary and free-boundary solves,
- axisymmetric and non-axisymmetric configurations with ``lasym=False/True``,
- end-to-end differentiability (JAX autodiff),
- laptop-friendly performance (careful JIT boundaries, minimal allocations),
- stepwise validation against VMEC2000 output (``wout_*.nc``).

.. only:: not fast

   .. toctree::
      :maxdepth: 2
      :caption: User guide

      overview
      installation
      quickstart
      theory
      equations
      vmec_wiki_primer
      algorithms
      validation
      optimization
      optimization_sweep_results
      discrete_adjoint
      simsopt_comparison
      jxbforce_mercier
      free_boundary_plan
      performance
      aggressive_performance_plan
      accelerated_merge_readiness
      code_structure
      contributing
      references

.. only:: fast

   Fast doc builds are enabled (``SPHINX_FAST=1``). The full user guide and API
   reference are skipped to keep CI fast.

.. only:: not fast

   .. toctree::
      :maxdepth: 2
      :caption: API reference

      api/index
