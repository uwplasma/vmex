vmec-jax documentation
======================

``vmec-jax`` is an incremental, JAX-based rewrite of **VMEC2000**, targeting:

- bundled-reference parity and optional executable-backed VMEC2000 validation
  for fixed-boundary and free-boundary solves,
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
      optimization
      optimization_sweep_results

   .. toctree::
      :maxdepth: 2
      :caption: Physics and algorithms

      theory
      equations
      vmec_wiki_primer
      algorithms
      discrete_adjoint
      simsopt_comparison
      jxbforce_mercier

   .. toctree::
      :maxdepth: 2
      :caption: Validation and release

      validation
      testing_strategy
      release_checklist
      optional_validation_plan
      free_boundary_plan
      performance

   .. toctree::
      :maxdepth: 2
      :caption: Development notes

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
