Optimization and differentiability
==================================

.. note::

   This page is a short orientation. Full optimization tutorials (QA/QH/QP/QI
   from a circular torus, free-boundary beta scans, single-stage runs) arrive
   with the rewritten ``examples/`` gallery — see :doc:`tutorials`.

Objectives (:mod:`vmec_jax.core.optimize`)
------------------------------------------

Simsopt-style building blocks, all pure functions of a converged core state:

- :class:`~vmec_jax.core.optimize.QuasisymmetryRatioResidual` — the
  Landreman-Paul two-term quasisymmetry ratio residual (QA: ``m=1, n=0``;
  QH: ``m=1, n=-1``; QP: ``m=0, n=1`` in ``nfp`` units);
- scalar targets: :func:`~vmec_jax.core.optimize.aspect_ratio`,
  :func:`~vmec_jax.core.optimize.mean_iota`,
  :func:`~vmec_jax.core.optimize.edge_iota`,
  :func:`~vmec_jax.core.optimize.mirror_ratio`,
  :func:`~vmec_jax.core.optimize.volume`,
  :func:`~vmec_jax.core.optimize.magnetic_well`,
  :func:`~vmec_jax.core.optimize.d_merc`,
  :func:`~vmec_jax.core.optimize.l_grad_b`;
- a Goodman-style quasi-isodynamic residual
  (:func:`~vmec_jax.core.optimize.quasi_isodynamic_residual`);
- :func:`~vmec_jax.core.optimize.least_squares` — a thin
  ``scipy.optimize.least_squares`` driver over boundary Fourier degrees of
  freedom (:func:`~vmec_jax.core.optimize.pack_boundary` /
  :func:`~vmec_jax.core.optimize.unpack_boundary`), taking simsopt-style
  ``(callable, target, weight)`` terms. Jacobians default to scipy
  finite differences (``jac=None``); ``jac="implicit"`` switches to exact
  implicit-differentiation Jacobians through
  :mod:`vmec_jax.core.implicit`. Implicit mode is restricted to
  fixed-boundary, stellarator-symmetric (``LASYM = F``) problems whose
  objective terms are implicit-differentiable (the Mercier/`L_grad_B`/QI
  diagnostics run on host NumPy and need ``jac=None``).

Gradients (:mod:`vmec_jax.core.implicit`)
-----------------------------------------

Derivatives through the equilibrium use **implicit differentiation** of the
converged fixed point: the solve is wrapped in ``jax.custom_vjp``; the
forward pass runs the fast (opaque) solver, and the backward pass solves the
adjoint linear system matrix-free with preconditioned GMRES. This replaced
the earlier discrete-adjoint / replay-tape machinery entirely — coarse
multigrid stages are just an initializer and are stop-gradient by
construction. See the *Implicit differentiation* section of
:doc:`algorithms` for the formulation and cost analysis.

Gradient accuracy is validated in CI against central finite differences for
**fixed-boundary** degrees of freedom: boundary Fourier coefficients,
``phiedge``, and profile parameters (``pres_scale``), on a 2D (solovev) and
a 3D (li383) case — the gradient table lives in
``tests/test_implicit_grad.py``, with agreement at the 1e-6
relative level (2D) and at the finite-difference noise floor (3D).

.. code-block:: python

   import jax
   from vmec_jax.core import implicit
   from vmec_jax.core.input import VmecInput

   inp = VmecInput.from_file("input.solovev")
   p0 = implicit.params_from_input(inp)

   sol = implicit.run(inp, p0)                        # ImplicitSolution pytree
   grad = jax.grad(lambda p: implicit.run(inp, p).wb)(p0)   # adjoint gradient

**Free boundary** is differentiable through a different route
(:mod:`vmec_jax.core.freeboundary_diff`): rather than differentiating the
NESTOR vacuum solve, the plasma boundary contribution to the vacuum field is
computed by **virtual casing**, which is a smooth function of the coil /
``extcur`` parameters and the plasma surface. Coil-parameter derivatives of
free-boundary outputs are obtained end-to-end this way and are
finite-difference-validated. (The two scopes are complementary: the
fixed-boundary implicit adjoint is validated to ~1e-6 relative; the
free-boundary virtual-casing path is FD-validated.)

Worked results
--------------

From a near-circular torus seed, staged ``max_mode`` continuation with ESS and
``jac="implicit"`` reaches precise quasisymmetry (measured on an office CPU):
QA (nfp 2) QS ``1.70e-4``, QH (nfp 4) QS ``5.83e-5``, QP (nfp 2) QS
``9.4e-2`` (basin-limited), and QI (nfp 1, QP→QI) ``2.14e-2``. Implicit
gradients are *essential* here, not merely faster: the exact-axisymmetric seed
is a saddle of the QS residual where finite differences stall, and for QP the
implicit path selects a better basin. The complete scripts are in
``examples/optimization/`` (``QA``/``QH``/``QP``/``QI``).

.. figure:: _static/figures/readme_optimization.png
   :alt: QA/QH/QP seed vs optimized boundary, 3-D |B|, and Boozer |B| on the LCFS
   :align: center
   :width: 100%

   Quasisymmetry (QA/QH/QP): seed (grey) vs optimized (blue) boundary cross
   sections (top), the optimized LCFS in 3-D coloured by ``|B|`` (middle), and
   ``|B|`` in Boozer coordinates on the LCFS (jet line contours, bottom), whose
   contour geometry reads off the symmetry family. The label is the QS residual
   measured on the plotted equilibrium. Reproduce with
   ``benchmarks/make_readme_figures.py --only optimization`` from the decks in
   ``benchmarks/opt_decks/``.

.. figure:: _static/figures/readme_qi.png
   :alt: QI equilibria at nfp 1-4: boundary, 3-D |B|, and Boozer |B| on the LCFS
   :align: center
   :width: 100%

   Quasi-isodynamic (QI) equilibria at nfp 1/2/3/4 (bundled decks in
   ``examples/data/``): boundary cross sections, 3-D ``|B|`` geometry, and
   ``|B|`` in Boozer coordinates on the LCFS (jet). The label is the QI
   (omnigenity) residual, not QS. Reproduce with
   ``benchmarks/make_readme_figures.py --only qi``.
