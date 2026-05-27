Piecewise-Omnigenous Optimization Plan
======================================

Status
------

This is a planning and acceptance document for adding piecewise-omnigenous
optimization (pwO) to ``vmec_jax``.  The immediate seed should be the current
reviewed optimized QP configuration because its Boozer ``|B|`` contours already
resemble a piecewise-omnigenous topology more than a clean quasi-poloidally
symmetric state.  The archived path below records the original planning seed
and must be replaced by the current aspect-5, reviewed best QP row before any
pwO result is promoted:

.. code-block:: text

   examples/optimization/results/qs_ess_sweep/cpu/continuation/qp/mode3/no_ess/input.final
   examples/optimization/results/qs_ess_sweep/cpu/continuation/qp/mode3/no_ess/wout_final.nc

The plan is intentionally separate from the current QA/QH/QP/QI production
scripts until the diagnostics, gradients, and validation gates are stable.

Literature Basis
----------------

The pwO objective should be based on second-adiabatic-invariant structure, not
only visual ``|B|`` contour fitting.

- Velasco et al., `Piecewise omnigenous stellarators
  <https://arxiv.org/abs/2405.07634>`_, introduced pwO fields as a relaxation
  of full omnigenity in which transitioning particles are allowed while
  retaining tokamak-like collisional energy transport in the studied examples.
  The key implementation
  implication is that topology and well labels matter; a scalar contour
  smoothness penalty alone is insufficient.
- Velasco, Sánchez, and Calvo, `Exploration of the parameter space of piecewise
  omnigenous stellarator magnetic fields <https://arxiv.org/abs/2412.14871>`_,
  characterize pwO fields systematically.  This motivates exposing pwO
  template parameters as optimization variables or staged hyperparameters
  rather than hard-coding one fixed parallelogram.
- Velasco et al., `Combination of quasi-isodynamic and piecewise omnigenous
  magnetic fields <https://arxiv.org/abs/2603.12377>`_, combine QI behavior in
  low-field regions with pwO behavior in high-field regions.  This is the most
  plausible near-term path for ``vmec_jax`` because the existing QI and QP machinery can
  supply differentiable Boozer-space diagnostics and robust seeds.
- Fernández-Pacheco et al., `Piecewise omnigenous magnetohydrodynamic
  equilibria as fusion reactor candidates <https://arxiv.org/abs/2601.14886>`_,
  present at least one pwO MHD equilibrium with reactor-relevant physics
  metrics.  The ``vmec_jax`` implementation therefore needs VMEC parity,
  Mercier/current/profile diagnostics, and finite-beta extension hooks before
  claiming production readiness.

Objective Architecture
----------------------

The first production objective should have three separable residual families:

1. ``pwo_shape_residual``: compare normalized Boozer ``|B|`` on selected
   surfaces against a smooth pwO template.  Start with the parallelogram-style
   template used by the local ``omnigenity_optimization/class_pwO.py`` reference
   and expose center, slopes, widths, inside/outside weights, and smoothing
   width as explicit parameters.
2. ``pwo_well_invariant_residual``: compute a differentiable proxy for
   ``dJ_parallel / d alpha`` within each labeled well family.  The first version
   should use fixed well labels from a seed/reference state to avoid
   non-differentiable sorting and branch switching during early tests.
3. ``pwo_engineering_residuals``: reuse existing differentiable constraints for
   aspect ratio, ``abs(mean_iota)`` floor, mirror ratio, elongation, QI/QP
   regularization, Mercier/``jdotb`` once promoted, and optional finite-beta
   profile terms.

Implementation Phases
---------------------

Phase 1: diagnostics only
~~~~~~~~~~~~~~~~~~~~~~~~~

- Add ``vmec_jax.piecewise_omnigenous`` with pure JAX helpers for template
  construction, smooth inside/outside masks, and normalized Boozer ``|B|`` grid
  residuals.
- Add ``vmec_jax.pwo_diagnostics`` for non-optimization diagnostics and plotting
  summaries.
- Add tests against synthetic Boozer grids with exact known pwO templates,
  including gradient checks against finite differences.
- Add a diagnostic example that reads the optimized QP seed WOUT and reports pwO
  shape score, well-label consistency, aspect ratio, iota, mirror, and
  elongation.

Phase 2: differentiable local optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Add ``examples/optimization/pwO_optimization.py`` using the same visible
  workflow as QA/QH/QP/QI: top-level parameters, ``FixedBoundaryVMEC`` object,
  explicit objective tuples, ``LeastSquaresProblem.from_tuples(...)``,
  ``least_squares_solve(...)``, then explicit save/plot calls.
- Use the optimized QP seed first, then test the common minimal NFP=2 seed.
- Keep aspect target 5 and ``abs(mean_iota) >= 0.41`` for comparability with
  README QA/QH/QP/QI results.
- Use continuation over mode lists and ESS as existing examples do; only add
  global/reference-family preconditioning if local optimization cannot keep the
  pwO topology.

Phase 3: branch-robust objective
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Replace fixed well labels with a staged label update policy: labels are
  frozen inside one accepted stage, then recomputed only at stage boundaries and
  recorded in diagnostics.
- Add a soft label-assignment option for differentiability experiments, but do
  not use it as the default until it ranks known pwO references and, when
  available, QI-pwO references the same way as the fixed-label diagnostic.
- Add a ``QI-pwO`` hybrid residual that applies QI-like constraints in
  low-field regions and pwO constraints in high-field regions.

Validation Gates
----------------

Before promoting pwO to README examples, require:

- synthetic template tests with analytic residual zero and finite-difference
  gradient agreement,
- agreement with the local ``omnigenity_optimization/class_pwO.py`` diagnostic
  ranking on the QP seed, QI seed, and at least one known pwO input from
  ``/Users/rogeriojorge/local/omnigenity_optimization/inputs``,
- VMEC2000 parity for the promoted final WOUT at the same fixed-boundary input,
- Boozer-resolution convergence of the pwO score,
- no regression in QA/QH/QP/QI production figures and local CI gate.

Open Risks
----------

- Well topology can change discontinuously.  The first implementation should
  not differentiate through well relabeling.
- A visual pwO-like QP contour plot may still fail the second-invariant gate.
  Promotion should use diagnostics, not contour appearance alone.
- pwO templates add hyperparameters.  The example should show a small, explicit
  parameter block, while broader template sweeps belong in a diagnostic script.
