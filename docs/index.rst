vmec-jax documentation
======================

``vmec-jax`` is an incremental, JAX-based rewrite of **VMEC2000**, targeting:

- bundled-reference validation and optional executable-backed VMEC2000 checks
  for representative fixed-boundary and free-boundary solves, with strict field
  parity promoted case-by-case,
- axisymmetric and non-axisymmetric ``lasym=False/True`` coverage, including
  convergence/physics gates where strict parity is not yet promoted,
- end-to-end differentiability (JAX autodiff),
- performance profiling and tuned default paths, with CPU/GPU benchmark results
  documented per case rather than implied globally,
- required fast coverage enforced at the 95% gate after the latest
  CI-equivalent coverage ratchet,
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
      free_boundary_coil_optimization

   .. toctree::
      :maxdepth: 2
      :caption: Physics and algorithms

      theory
      equations
      vmec_wiki_primer
      algorithms
      discrete_adjoint
      piecewise_omnigenous_plan
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
