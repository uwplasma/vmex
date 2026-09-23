The solver: iteration, preconditioning and multigrid
====================================================

Each VMEX iteration follows VMEC2000's ``funct3d`` ordering: synthesize the
geometry from the spectral state, form the fields and forces, project the
residuals back to Fourier space, precondition them, and advance with a damped
second-order Richardson step. ``NS_ARRAY`` repeats this on a ladder of radial
grids. This page documents the discretization, the update and its controller,
the restart rules, the two preconditioners, the multigrid ladder and hot
restart, and the two execution lanes. The equations being solved are in
:doc:`variational-problem`.

Discretization summary
----------------------

Radial grid
~~~~~~~~~~~

A uniform grid in :math:`s \in [0,1]` with ``ns`` points:

.. math::

   s_j = \frac{j}{ns-1},\qquad j=0,\dots,ns-1.

Following VMEC2000, quantities live on a mix of the *full mesh* (:math:`s_j`)
and the *half mesh* (:math:`s_{j-1/2}`): geometry derivatives, the Jacobian,
and ``|B|``-type quantities are half-mesh; R/Z coefficients and ``iotaf`` are
full-mesh. Odd-m coefficients are stored internally with the axis-regular
:math:`\sqrt{s}` factor removed (``scalxc``, see
:doc:`variational-problem`), and R/Z and :math:`\lambda`
evolution starts from the m-dependent ``jmin2``/``jlam`` radial indices
(``vmec_params.f``). These conventions are implemented in
:mod:`vmex.core.geometry` and :mod:`vmex.core.setup`.

Angular grids and Fourier transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Uniform tensor-product grids in :math:`\theta` and :math:`\zeta` (one field
period) with VMEC's symmetry-reduced theta extent. The analysis transform
(``tomnsps``) is the two-stage weighted DFT of
:doc:`variational-problem`, built from precomputed trig tables
(``fixaray.f``) in :mod:`vmex.core.fourier`; the transforms
(``totzsps/totzspa`` synthesis, ``tomnsps/tomnspa`` analysis) are batched
``dot_general`` matmuls in :mod:`vmex.core.transforms` — GEMM-friendly
and XLA-fusable while matching VMEC2000 normalization exactly.

Geometry pipeline
~~~~~~~~~~~~~~~~~

Per iteration (:mod:`vmex.core.geometry`, VMEC2000 ``jacobian.f``):

1. synthesize :math:`(R,Z,\lambda)` and their angular derivatives on the
   ``(s,\theta,\zeta)`` grid from the spectral state;
2. form radial derivatives on the half mesh;
3. compute the half-mesh Jacobian :math:`\sqrt{g}` and the metric elements
   ``guu, guv, gvv``;
4. evaluate the Jacobian sign proxy :math:`\tau`; a sign change away from the
   axis flags a bad Jacobian (``irst = 2``).

The Richardson update
---------------------

VMEX evolves the stacked Fourier coefficients :math:`\mathbf{x}` of
:math:`(R,Z,\lambda)` with VMEC2000's preconditioned, damped second-order
Richardson ("momentum") iteration, a discretization of

.. math::

   \ddot{\mathbf{x}} + \frac{1}{\tau}\dot{\mathbf{x}} = P^{-1} F(\mathbf{x}),

with :math:`F` the spectral force residual and :math:`P` the preconditioner
described below. One step is

.. math::

   \mathbf{v}_{k+1} = \mathrm{fac}\,\bigl(b_1\,\mathbf{v}_k + \Delta t\,P^{-1}F(\mathbf{x}_k)\bigr),
   \qquad
   \mathbf{x}_{k+1} = \mathbf{x}_k + \Delta t\,\mathbf{v}_{k+1},

with :math:`b_1 = 1 - \Delta\tau`, :math:`\mathrm{fac} = 1/(1 + \Delta\tau)` and
the damping

.. math::

   \Delta\tau = \tfrac{1}{2}\,\Delta t\,\langle \mathrm{otau}\rangle,
   \qquad
   \mathrm{otau}_k = \min\!\left(\left|\ln\frac{\mathrm{fsq}_k}{\mathrm{fsq}_{k-1}}\right|,\; 0.15\right) \Big/ \Delta t,

averaged over the last ``ndamp = 10`` steps, where :math:`\mathrm{fsq}` is the
preconditioned residual sum (``evolve.f``). The window resets to the cap after
every restart. :func:`~vmex.core.step.damping_coefficients` advances the
window and returns ``(b1, fac)``, :func:`~vmex.core.step.momentum_update`
applies the update, and the traced controller scalars (``delt``, damping
history, best-residual trackers, ``iter1``, ``ijacob``) live in
:class:`~vmex.core.step.StepControl`. The controller tracks the minimum of the
preconditioned residual (``res0``) and of the physical residual (``res1``);
growth triggers the restart rules below.

The iteration is *not* a monotone descent on :math:`W`, and it does not test
for one. Hirshman and Whitson (1983) guarantee monotone decrease only for
first-order (:math:`\ddot{\mathbf{x}} = 0`) descent; the second-order term
trades that guarantee for a faster asymptotic rate. Every step is taken
unconditionally, with no line search and no energy comparison; the only
intervention is :func:`~vmex.core.step.restart_decision`, a residual test.

Convergence is declared when the *physical* residuals satisfy
``fsqr, fsqz, fsql <= ftolv`` simultaneously; the residual norms and the m=1
constraint rotation follow ``residue.f90`` (:mod:`vmex.core.residuals`).

Restart control (``restart.f``)
-------------------------------

The loop keeps a checkpoint of the best state and applies VMEC2000's exact
back-off rules (:func:`~vmex.core.step.restart_decision` classifies the
step as ``STEP_OK``/``RESTART_JACOBIAN``/``RESTART_GROWTH``;
:func:`~vmex.core.step.apply_restart` restores the checkpoint, zeroes
the velocity and rescales ``delt``):

- **bad Jacobian** (``irst = 2``): restore the checkpoint, zero the velocity,
  ``delt *= 0.90``; on the first bad Jacobian the axis guess is recomputed
  (``guess_axis``), and ``delt`` is reset at ``ijacob = 25, 50`` with a hard
  stop at 75 (``jac75_flag``).  VMEX then offers a bounded driver-level
  recovery (two attempts by default): restart the best finite checkpoint with
  zero velocity and half the preceding initial ``DELT`` (capped at 0.5).
  This is a continuation, not a fresh ``profil3d`` initialization: the
  first-pass ``LMOVE_AXIS`` transfer is disabled on the driver-level retry, so
  a still-large force cannot replace the checkpoint with a cold axis-derived
  state.
  The force equations and stopping tolerance do not change.  Set
  ``jacobian_retries=0`` (Python) or ``--jacobian-retries 0`` (CLI) for the
  exact VMEC2000 fatal-stop policy.  Free-boundary recovery rebuilds the
  axis-current filament and all resolution/geometry-dependent NESTOR
  structures before continuing;
- **residual blow-up** (``irst = 3``): if after more than 10 steps the
  residual exceeds :math:`10^4\times` the checkpoint value, restore and
  ``delt /= 1.03``.

Constraint strength (``tcon``)
------------------------------

The spectral-condensation constraint force of
:doc:`variational-problem` is scaled per surface by

.. math::

   \mathrm{tcon}(j) = \min\!\left(\left|\frac{a_{rd}}{a_{r,\mathrm{norm}}}\right|,
   \left|\frac{a_{zd}}{a_{z,\mathrm{norm}}}\right|\right)\cdot
   \mathrm{tcon}_0\text{-scaled}\cdot(32\,h_s)^2,
   \qquad \mathrm{tcon}(ns) = \tfrac{1}{2}\,\mathrm{tcon}(ns-1)

(``bcovar.f``). Implemented in :mod:`vmex.core.forces` (constraint force)
and :mod:`vmex.core.fields` (``tcon``). Preconditioner matrices, force
norms, and ``tcon`` are recomputed every ``ns4 = 25`` iterations and reused
in between — this cadence is parity-critical and is mirrored exactly.

Preconditioning
---------------

Two preconditioners make each step effective: the always-on 1D radial
tridiagonal preconditioner ported from ``precondn.f``/``scalfor.f``, and an
opt-in matrix-free 2D block preconditioner (``precon2d.f`` analogue) that
takes inexact Newton steps on stiff decks.

1D radial preconditioner (``precondn.f``, ``scalfor.f``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The 1D preconditioner approximates the diagonal (in ``(m,n)``) of the
linearized radial force operator: for each spectral column the R/Z force is
replaced by the solution of a radial tridiagonal system

.. math::

   \bigl[\,b_x(s),\; d_x(s),\; a_x(s)\,\bigr]\,X = F_{mn}(s), \qquad
   d_x(s) = -\bigl(a_{xd} + b_{xd}\,m^2 + c_x\,(n\,\mathrm{NFP})^2\bigr),

whose coefficients are flux-surface integrals over the half mesh
(``precondn.f``, :func:`~vmex.core.preconditioner.precondn`) of
:math:`p_\tau = -4\,r_{12}^2\,\mathrm{bsq}\,w/\sqrt{g}`-type quantities: the
poloidal-derivative couplings give :math:`a_x`, the radial-derivative
couplings :math:`b_x`, and
:math:`c_x = \langle \tfrac14 p_{\mathrm{factor}} (B^v)^2 \sqrt{g}\rangle`
the toroidal couplings, each with even-m and odd-m columns (the odd column
carries the internal :math:`\sqrt{s}` scalings). Assembly of the per-mode
system with the :math:`m^2` and :math:`(n\,\mathrm{NFP})^2` weights, the
``edge_pedestal = 0.05`` and ZC(0,0)(ns) ``fac = 0.25`` stabilizations of
``scalfor.f``, and the ``jmin`` axis-row rules is
:func:`~vmex.core.preconditioner.scalfor_matrices`; the application is
:func:`~vmex.core.preconditioner.scalfor`. The solve is a Thomas
algorithm vectorized over all spectral columns simultaneously
(:func:`vmex.core.preconditioner.tridiagonal_solve`, a thin arg-order
adapter over ``solvax.tridiagonal_solve`` — the shared SOLVAX linear-solver
package). Production application uses ``solvax.tridiagonal_solve_checked``:
the unregularized Thomas pivots must pass VMEC2000's
``abs(pivot) > 1e-8*abs(diagonal)`` condition and a backward-residual check.
Rejected columns receive an identity preconditioner action and a typed
diagnostic instead of an amplified finite or NaN/Inf update. :math:`\lambda`
uses the diagonal ``faclam`` factors from
``lamcal.f90`` (:func:`~vmex.core.preconditioner.lamcal`):

.. math::

   \mathrm{faclam} \propto
   \frac{\sqrt{s}^{\,\min(m^2/16^2,\,8)}}
        {b_\lambda\,(n\,\mathrm{NFP})^2 \pm 2mn\,\mathrm{NFP}\,d_\lambda
         + c_\lambda\,m^2},

with :math:`b_\lambda = \langle g_{uu}/\sqrt{g}\rangle`,
:math:`c_\lambda = \langle g_{vv}/\sqrt{g}\rangle`,
:math:`d_\lambda = \langle g_{uv}/\sqrt{g}\rangle` (the :math:`\sqrt{s}`
damping only bites for :math:`m > 16`).

The matrices are refreshed on the ``ns4 = 25`` cadence above.

2D block preconditioner (``precon2d.f``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For stiff cases (high beta, high aspect ratio, high mode number) VMEC2000
optionally switches to its 2D preconditioner: a **Newton step** on the
1D-preconditioned force. Let :math:`g(\mathbf{x})` be the 1D-preconditioned
spectral force map (with the ``ns4`` cache frozen, so the 1D operator is a
fixed linear map during the Newton solve). The update direction solves

.. math::

   J\,\delta = -g(\mathbf{x}), \qquad
   J = \frac{\partial g}{\partial \mathbf{x}}
   \quad\text{(block-tridiagonal in radius)}.

Because the invertible 1D operator :math:`M_{\mathrm{1D}}^{-1}` is baked
into both :math:`J` and the right-hand side, it cancels exactly:
:math:`\delta` is the same full Newton step
:math:`-(\partial F/\partial\mathbf{x})^{-1} F` on the raw force — the 1D
preconditioner only conditions the linear solve (near equilibrium
:math:`M_{\mathrm{1D}}` approximates
:math:`\partial F/\partial \mathbf{x}`, so :math:`J \approx I`).

VMEC2000 (``Sources/Hessian/precon2d.f``) builds :math:`J` explicitly by
finite-difference "jogs" of every spectral column and LU-factors it with
BCYCLIC. In :mod:`vmex.core.preconditioner_2d` the force map is
traceable, so :math:`J v` is an **exact Hessian-vector product** from one
``jax.jvp`` (:func:`~vmex.core.preconditioner_2d.flat_operator`) — no
jogs, no assembled blocks — and the system is solved with matrix-free
restarted GMRES from SOLVAX (``solvax.gmres``) in
:func:`~vmex.core.preconditioner_2d.newton_direction`. A loose GMRES
tolerance yields an inexact Newton step; peak memory stays at one force
graph. Activation mirrors the main ``evolve.f`` gates
(:class:`~vmex.core.preconditioner_2d.Prec2DConfig`): finest grid only,
``iter2 >= 10``, and ``fsqr + fsqz + fsql < prec2d_threshold``; the wiring in
:mod:`vmex.core.solver` swaps the Newton direction for the 1D force
direction under a ``lax.cond``, leaving the default 1D-only path untouched.
It does **not** reproduce VMEC2000's distinct CG/GMRESR/TFQMR evolution
algorithms or its ``PRE_NITER`` budget mutation; see
:doc:`/reference/vmec2000-compatibility`.

On three stiff cases in ``benchmarks/preconditioner_2d_stiff_cases.json``
(VMEX 0.8.1) it took 5.4x, 10.9x and 7.7x fewer iterations than the 1D path
and reached the same ``wb`` to better than ``1e-5`` relative. That record
measures iterations, not wall time or memory, which is why the 2D path is
opt-in; the table is in :doc:`/reference/performance`.

The multigrid ladder
--------------------

``NS_ARRAY`` runs the solve as a ladder of radial resolutions: each stage
converges at its own ``ns`` with its own ``FTOL_ARRAY``/``NITER_ARRAY``
entry, and the converged coefficients seed the next finer grid through
VMEC2000's ``interp.f`` transfer.

A ladder is a convergence aid, not a requirement. A smooth optimized deck or
a hot restart can use one entry such as ``NS_ARRAY = 31`` and avoid the
coarse solves and their executable shapes. Keep a coarse-to-fine ladder for a
rough boundary, a poor axis guess, or a high-resolution cold start; VMEX also
prepends an emergency ``ns=3`` stage only when the first Jacobian is invalid
(below).

Stage transfer (``interp.f``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`vmex.core.multigrid.solve_multigrid` runs the ``NS_ARRAY``
ladder: each stage solves at its ``ns`` with its own
``FTOL_ARRAY``/``NITER_ARRAY`` entry, and the converged coefficients are
interpolated in :math:`\sqrt{s}`-internal form to the next finer grid
(``interp.f``): scale by ``scalxc``, extrapolate odd-m modes to the axis on
the scaled array, interpolate linearly in :math:`s`, unscale, and zero the
odd-m axis row (:func:`~vmex.core.multigrid.interpolate_coefficients` /
:func:`~vmex.core.multigrid.interpolate_state`). In equations, the staging
interpolates *scaled* coefficients between grids,

.. math::

   x_{\mathrm{scaled}} = x \cdot \mathrm{scalxc},

with odd-m extrapolation to the axis performed **before** interpolation;
after linear interpolation on a uniform radial grid, coefficients are
unscaled:

.. math::

   x_{\mathrm{new}} = \frac{x_{\mathrm{scaled,new}}}{\mathrm{scalxc}_{\mathrm{new}}}.

VMEX implements this exact pipeline so that stage-to-stage coefficient
transfer matches VMEC2000.

Each distinct stage structure compiles once; repeated
ladders with the same structures reuse those executables. A single
maximum-resolution masked executable is future work, not current behavior.
``--prefetch-compile`` controls whether upcoming stage executables are
compiled ahead of time or on first use (the CLI defaults to the
lower-memory on-demand path).

Carried module state
~~~~~~~~~~~~~~~~~~~~

The transfer includes VMEC2000's non-geometric module state.  In particular,
``initialize_radial.f`` resets ``fsq``, ``iter1``, ``iter2``, ``ijacob``,
and the time-step controller, but it does **not** reset
``fsqr/fsqz/fsql``.  VMEX therefore passes the previous stage's three
invariant residuals into the first force evaluation on the next grid.  This
matters in free boundary: ``residue.f90`` uses the retained
``fsqr + fsqz`` to decide whether the carried edge-force row belongs in the
first fine-grid norm.  Axis re-guess transfers and bounded JAC75 retries
retain the same residual state instead of silently becoming cold starts.

Free-boundary ladders
~~~~~~~~~~~~~~~~~~~~~

:func:`vmex.core.multigrid.solve_free_boundary_multigrid` implements
``runvmec.f``'s radial ladder.  Increasing grids interpolate ``xstore`` using
the same odd-m :math:`\sqrt{s}` scaling as fixed boundary; equal grids rerun
the current state and the ladder stops at the first decreasing entry.
``ivac``, adaptive ``nvacskip``, the exact ``rbsq`` edge product, and the
three invariant residual channels are carried.  Because the free-boundary
block is guarded by ``iter2 > 1``, a new stage uses that carried edge product
on iteration 1 and performs its first full update on iteration 2.  The
resolution-specific NESTOR basis, Green-function program, axis-current
filament program, cached potential matrix, and traced cadence loop are selected
or rebuilt for the new stage.  Vacuum activates only once across the ladder
(see :doc:`nestor-vacuum`).

If a first requested grid remains invalid after ``guess_axis``, the fixed- or
free-boundary driver retries the ladder once with an ``ns=3``, ``ftol=1e-4``
stage prepended.  The coarse equilibrium is then interpolated to the user's
first grid; free-boundary vacuum activation restarts cleanly.  This narrow
recovery follows the current VMEC++ driver; it fires only for
``bad_jacobian_flag`` and does not hide convergence, non-finite, or input
failures.  ``coarse_grid_retry=False`` disables it for strict failure studies.

Hot restart
~~~~~~~~~~~

The same interpolation seam provides hot restart
(:func:`vmex.core.solver.hot_restart_state`): a previous solution (e.g.
the previous point of a parameter scan) can seed the solve directly, at the
same or a different radial resolution, via ``solve(...,
initial_state=state)`` and ``solve_multigrid(..., initial_state=state)``;
``initial_state=`` seeds the first rung and skips none.
Restarting from a ``wout_*.nc`` file — including VMEC2000-written ones —
goes through :mod:`vmex.core.restart` (``restart_from`` / ``--restart`` /
the ``RESTART_WOUT`` deck key), which inverts the wout output maps back to
the internal spectral state, resamples radial/mode-table differences, and
drops every leading ``NS_ARRAY`` rung whose resolution the seed already
meets or exceeds. The recipe with measured iteration counts is
:doc:`/howto/restart-from-previous-run`.

Two execution lanes, one physics
--------------------------------

:mod:`vmex.core.solver` exposes the same jitted iteration through two
lanes (selected by ``vmex --mode cli|jit``): the default **CLI lane**, a
Python ``while`` loop around a jitted *N-iteration block* kernel with host
residual checks between blocks (exact-``ftol`` early exit, live
VMEC2000-format printing every ``NSTEP`` iterations, buffer donation, zero
autodiff bookkeeping), and the **JIT lane**, a single ``lax.while_loop``
over the same physics — fully traceable, the forward solver inside the
differentiable API. A regression test asserts per-block state agreement
between the lanes to machine precision. Which device (CPU or GPU) a lane
runs on is decided by the measured placement policy of
:mod:`vmex.core.device` — see :doc:`/howto/run-on-gpu` and
:doc:`architecture`.
