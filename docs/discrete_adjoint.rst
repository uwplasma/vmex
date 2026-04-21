Discrete-adjoint differentiation
==================================

Overview
--------

vmec_jax computes exact Jacobians of any differentiable scalar functional
(quasisymmetry residuals, aspect ratio, etc.) with respect to boundary shape
parameters using a **discrete-adjoint** technique.

In contrast to the classical finite-difference approach used by SIMSOPT +
VMEC2000, discrete-adjoint differentiation:

* is **exact** — derivatives are computed to floating-point precision,
  not truncated by step-size selection;
* requires only **one checkpoint replay** rather than one full forward solve
  per parameter;
* scales with **number of output quantities**, not number of input parameters
  — ideal when the parameter space is large.


Background: what are we differentiating?
-----------------------------------------

The VMEC iteration produces a *converged equilibrium state*
:math:`\boldsymbol{q}^* \in \mathbb{R}^N` (packed array of ``R``, ``Z``,
``λ`` Fourier coefficients on the VMEC half/full mesh) by applying the
iterative map

.. math::

   \boldsymbol{q}_{k+1} = \Phi(\boldsymbol{q}_k,\, p)

from an initial guess :math:`\boldsymbol{q}_0(p)` that depends on the boundary
parameters :math:`p \in \mathbb{R}^m`.  At convergence,
:math:`\boldsymbol{q}^*(p)` satisfies the fixed-point equation

.. math::

   \boldsymbol{q}^*(p) = \Phi(\boldsymbol{q}^*(p),\, p).

A scalar objective :math:`f = \ell(\boldsymbol{q}^*(p), p)` is then evaluated
from the converged state.  We want :math:`\partial f / \partial p_i`.

The challenge: :math:`\Phi` is a Fortran-style scan loop — a sequence of
~5 000 composite iterations — and automatic differentiation through 5 000
unrolled loop iterations would require storing the full trajectory in memory
and would produce enormous computation graphs.


The discrete-adjoint approach
-------------------------------

vmec_jax uses a **two-pass strategy**, analogous to the adjoint method in
optimal control:

**Forward pass (checkpoint tape)**

The forward solve runs normally, but at every ``N_checkpoint``-th iteration
a snapshot of the VMEC state is stored to a compact in-memory *checkpoint
tape*.  Between checkpoints, the state is not stored — it is recomputed on
demand during the backward pass.  This trades memory for recomputation.

.. code-block:: text

   Forward pass
   ────────────────────────────────────────────
   q₀(p) ──→ q₁ ──→ … ──→ q_{c₁} ┐ checkpoint
                                   └──→ … ──→ q_{c₂} ┐ checkpoint
                                                       └──→ … ──→ q*(p)

   The checkpoints {q_{c₁}, q_{c₂}, ...} are stored.
   All other iterates are discarded.

**Tangent propagation (JVP replay)**

For each boundary parameter :math:`p_i`, the tangent vector
:math:`\partial \boldsymbol{q}_0 / \partial p_i` is propagated forward
through the tape using **Jacobian-vector products (JVPs)**:

.. math::

   \frac{\partial \boldsymbol{q}_{k+1}}{\partial p_i}
   = \frac{\partial \Phi}{\partial \boldsymbol{q}_k}
     \cdot \frac{\partial \boldsymbol{q}_k}{\partial p_i}
   + \frac{\partial \Phi}{\partial p_i}

Because JAX traces the iterative map :math:`\Phi` as a JAX program, JVPs are
available via ``jax.jvp`` with no extra code.  All :math:`m` tangents are
propagated simultaneously using ``jax.vmap(jax.jvp(Φ, ...))`` — a single
batched JVP that visits each checkpoint interval exactly once.

This gives the full Jacobian column batch
:math:`\partial \boldsymbol{q}^* / \partial p_i` for all :math:`i`
in :math:`O(m)` JVPs, which is roughly equivalent to 1–2 forward solves
regardless of :math:`m`.

**Objective linearization**

Finally, the Jacobian of the objective with respect to the final state is
applied:

.. math::

   \frac{\partial f}{\partial p_i}
   = \frac{\partial \ell}{\partial \boldsymbol{q}^*}
     \cdot \frac{\partial \boldsymbol{q}^*}{\partial p_i}
   + \frac{\partial \ell}{\partial p_i}

using one more ``jax.jvp`` call on the residuals function.

The result is the exact (machine-precision) dense Jacobian matrix
:math:`J \in \mathbb{R}^{n_r \times m}` where :math:`n_r` is the number
of residuals and :math:`m` is the number of boundary DOFs.


Implementation in vmec_jax
---------------------------

The key functions live in ``vmec_jax/discrete_adjoint.py``:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Function
     - Role
   * - :func:`~vmec_jax.build_residual_checkpoint_tape`
     - Run forward solve; store checkpoints.  Returns a
       :class:`~vmec_jax.ResidualCheckpointTape`.
   * - :func:`~vmec_jax.checkpoint_tape_state_jvp_columns`
     - Given the tape and a batch of parameter tangents, propagate
       all tangents through the tape.  Returns
       :math:`\partial q^* / \partial p` (columns stacked).
   * - :func:`~vmec_jax.checkpoint_tape_param_jvp`
     - Single-parameter JVP (used internally).
   * - :func:`~vmec_jax.checkpoint_tape_state_vjp`
     - Reverse-mode (VJP) for scalar loss functions —
       cheaper than the forward-mode columns when
       :math:`n_r = 1` (e.g., single scalar objective).

The :class:`~vmec_jax.FixedBoundaryExactOptimizer` in
``vmec_jax/optimization.py`` orchestrates everything:

1. Call :func:`~vmec_jax.build_residual_checkpoint_tape` with the
   tight forward-solve settings.
2. Propagate boundary tangents via
   :func:`~vmec_jax.checkpoint_tape_state_jvp_columns`.
3. Multiply by the residuals Jacobian to form
   :math:`J = \partial r / \partial p`.
4. Solve the Gauss-Newton normal equations
   :math:`J^T J\, \Delta p = -J^T r` via LAPACK ``dgelsd``.
5. Armijo backtracking line search (relaxed forward solve at trial points).
6. Cache-hit detection: if the next call to ``residual_fun`` is at the
   same :math:`p` as the last tape build, reuse the tape.


Dynamic replay bucketing
~~~~~~~~~~~~~~~~~~~~~~~~~

The tape length :math:`K` (number of VMEC iterations to convergence) varies
slightly from one Gauss-Newton step to the next.  A different :math:`K`
would trigger XLA recompilation of the replay scan.

vmec_jax pads short tapes to the nearest multiple of
``VMEC_JAX_DYNAMIC_REPLAY_BUCKET`` (default: 1024) so that the same compiled
XLA kernel is reused across steps:

.. code-block:: bash

   export VMEC_JAX_DYNAMIC_REPLAY_BUCKET=512   # finer bucketing (more memory)
   export VMEC_JAX_DYNAMIC_REPLAY_BUCKET=2048  # coarser (fewer recompiles)


Comparison with other approaches
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 24 24 24

   * - Property
     - vmec_jax (discrete-adjoint)
     - SIMSOPT + VMEC2000 (FD)
     - Continuous adjoint (DESC)
   * - Jacobian cost
     - ≈ 1–2 × forward solve
     - m × forward solve
     - 1 × backward solve
   * - Accuracy
     - Machine precision
     - :math:`O(\sqrt{\varepsilon_\text{machine}})` FD error
     - Machine precision
   * - Memory
     - O(checkpoint\_interval × state)
     - O(1)
     - O(state)
   * - Subprocess dependency
     - None (pure Python/JAX)
     - Fortran binary required
     - None (Python/JAX)
   * - Differentiable through solver?
     - Yes (JAX autodiff)
     - No
     - Yes
   * - GPU support
     - Yes
     - No
     - Yes

**Continuous adjoint (DESC)**: DESC [Dudt et al., 2023]_ builds a continuous
PDE adjoint for the MHD equilibrium equations, solving the adjoint problem
exactly once using a Newton-Krylov solver.  The cost is one backward solve
(same order as one forward solve).  vmec_jax's discrete-adjoint replays the
iteration tape instead of solving a continuous adjoint equation, and is
therefore directly applicable to VMEC's fixed-point iteration without
reformulating the equations.

**Implicit differentiation (IFT)**: an alternative is to differentiate the
fixed-point equation implicitly via the implicit function theorem (IFT).
vmec_jax provides :func:`~vmec_jax.solve_fixed_boundary_state_implicit` for
this path.  It requires solving a linear system :math:`(I - \partial\Phi/\partial q)\,v = b`
which is approximated via CG + JVP.  The discrete-adjoint tape replay avoids
this linear solve entirely and is the default in
:class:`~vmec_jax.FixedBoundaryExactOptimizer`.


.. seealso::

   * :doc:`optimization` — practical guide to running vmec_jax optimizations
   * :doc:`simsopt_comparison` — detailed runtime and accuracy comparison with SIMSOPT
   * :func:`~vmec_jax.build_residual_checkpoint_tape`
   * :func:`~vmec_jax.checkpoint_tape_state_jvp_columns`
   * :class:`~vmec_jax.FixedBoundaryExactOptimizer`
