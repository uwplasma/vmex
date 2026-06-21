Code structure
==============

The package is organized by domain rather than by the original VMEC2000
subroutine names.  The goal is to keep the public surface small, preserve
VMEC2000-compatible numerics, and put differentiable implementation details
behind focused modules that are easy to test.

Core public surfaces:

- ``vmec_jax/api.py`` and ``vmec_jax/__init__.py`` expose the stable user API.
- ``vmec_jax/cli.py`` implements the ``vmec`` command-line entry point.
- ``vmec_jax/solve.py``, ``vmec_jax/driver.py``, ``vmec_jax/free_boundary.py``,
  and ``vmec_jax/wout.py`` are compatibility facades.  Keep them thin; new
  implementation logic should live in the domain packages below.

The current root namespace still contains historical compatibility modules.
Treat those as a migration burden, not as a pattern to copy.  New production
code should not add another root-level module unless it is a documented public
facade; put kernels, controllers, diagnostics, validation gates, and workflow
helpers in the domain packages below.

Numerical domains:

- ``vmec_jax/solvers/fixed_boundary/`` contains fixed-boundary VMEC iteration,
  scan, preconditioning, optimization, diagnostics, and result containers.
- ``vmec_jax/solvers/free_boundary/`` contains free-boundary provider plumbing,
  NESTOR/vacuum-coupling helpers, direct-coil validation seams, and
  branch-local adjoint/replay evidence.
- ``vmec_jax/external_fields/`` contains differentiable coil and ``mgrid``
  field providers.
- ``vmec_jax/io/wout/`` contains persisted-WOUT schema, netCDF I/O, flux
  conventions, JXBFORCE/Mercier reducers, ``DMerc``/Glasser ``D_R``
  diagnostics, and compatibility helpers.
- ``vmec_jax/optimizers/fixed_boundary/`` contains objective terms, exact
  replay, matrix-free/scalar-gradient paths, SciPy adapters, and workflow
  output helpers used by example-style optimizations.

Physics and geometry kernels:

- Root modules such as ``config.py``, ``static.py``, ``state.py``,
  ``boundary.py``, ``init_guess.py``, ``coords.py``, ``geom.py``,
  ``field.py``, ``energy.py``, ``vmec_tomnsp.py``, ``vmec_bcovar.py``, and
  ``preconditioner_1d.py`` hold reusable VMEC data structures and kernels.
- Objective/diagnostic modules such as ``quasisymmetry.py``,
  ``quasi_isodynamic.py``, ``qi_diagnostics.py``, ``finite_beta.py``, and
  ``plotting.py`` expose higher-level physics quantities used by examples,
  tests, and docs.

The ``examples/`` folder contains user-facing scripts and curated parity demos.
Developer-only diagnostics and research utilities live under ``tools/``:

- ``tools/diagnostics/source_health.py``: report largest Python source files
  and optionally fail above a line-count threshold for staged refactor ratchets.
- ``tools/diagnostics/vmec2000_exec_stage_trace_compare.py``: per-iteration
  VMEC2000 vs vmec_jax parity comparator.
- ``tools/diagnostics/qh_vmec_vs_vmecjax.py``: QH comparison figures.
- ``tools/diagnostics/readme_fsq_trace.py``: README fsq_total traces.

For most scripts, prefer ``import vmec_jax as vj`` or ``import vmec_jax.api as
vj``.  Import lower-level modules directly only when developing kernels,
validation tools, or tests that need implementation-specific behavior.

Where to make changes
---------------------

Use this map before adding files or changing public behavior:

- User-facing solve, plotting, and example workflows: change the public facade
  only when the stable API changes.  Put implementation details in the domain
  modules and re-export through ``vmec_jax/api.py`` or ``vmec_jax/__init__.py``
  only after the workflow is documented and tested.
- Fixed-boundary VMEC iteration semantics: edit
  ``vmec_jax/solvers/fixed_boundary/residual/`` for the VMEC2000-style
  residual controller, ``scan/`` for the JAX-visible VMEC2000 scan path, and
  ``preconditioning/`` or ``optimization/`` only for their named numerical
  domains.  Keep ``vmec_jax/solve.py`` as a compatibility facade.
- Free-boundary and direct-coil work: edit ``vmec_jax/external_fields/`` for
  coil or ``mgrid`` providers, ``vmec_jax/solvers/free_boundary/`` for NESTOR
  and provider plumbing, and ``vmec_jax/solvers/free_boundary/adjoint/`` for
  branch-local replay/JVP validation.  Do not claim differentiation through
  arbitrary adaptive host-controller branch changes unless a fingerprint-gated
  complete-solve AD-vs-FD test promotes that exact scope.
- Optimization science terms: add differentiable objectives in
  ``vmec_jax/optimization_workflow.py`` or the focused modules under
  ``vmec_jax/optimizers/fixed_boundary/``.  Keep example scripts
  SIMSOPT-like: editable top-level parameters, visible objective tuples, then
  a solve call and explicit result inspection/plotting.
- WOUT, Mercier, JXB, and profile diagnostics: use ``vmec_jax/io/wout/`` and
  ``vmec_jax/finite_beta.py``.  Preserve VMEC2000 storage conventions unless a
  documented diagnostic intentionally exposes a smoother differentiable proxy.
- Parity and physics gates: put cheap required tests under ``tests/`` with no
  local executable dependency; use ``tools/diagnostics/`` and optional markers
  for VMEC2000, SIMSOPT, GPU, or large fetched-asset validation.

Refactoring direction
---------------------

The current code intentionally preserves VMEC2000 semantics, but several
translation-era modules are now too large for long-term research development.
Use ``plan_differentiability.md`` as the active single source of truth for the
staged differentiability/refactor plan.  ``plan_freeb.md`` is now a closed
free-boundary evidence summary, not a parallel work plan; new free-boundary
follow-ups should be summarized in ``plan_differentiability.md``. ``plan.md``
and ``discrete_adjoint_2506_plan.md`` are historical/reference plans.  Before
starting a large extraction, run:

.. code-block:: bash

   python tools/diagnostics/source_health.py --top 30

The diagnostic is report-only by default on this draft PR so it can guide
large tranches without creating a brittle gate.  Ratchet it with
``--fail-lines`` only after a target module has been split and compatibility,
physics, and parity tests are green.

Current review guardrails:

- Do not add new root-level implementation modules.
- Keep compatibility facades outside hot JAX/VMEC loops.
- Put new fixed-boundary solver work under ``solvers/fixed_boundary/`` and new
  free-boundary/direct-coil work under ``solvers/free_boundary/`` or
  ``external_fields/``.
- Put new differentiable objective terms under ``optimizers/`` or the focused
  physics modules, then expose them through the public API only after tests and
  docs exist.
- Keep branch-local free-boundary derivative claims separate from arbitrary
  adaptive host-branch differentiation claims.
