Algorithms
==========

This page documents the numerical algorithms of the ``vmec_jax`` solver and
how they map onto their VMEC2000 counterparts. The module-by-module map is in
:doc:`architecture`; the underlying equations are derived in
:doc:`equations`.

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
:math:`\sqrt{s}` factor removed (``scalxc``), and R/Z and :math:`\lambda`
evolution starts from the m-dependent ``jmin2``/``jlam`` radial indices
(``vmec_params.f``). These conventions are implemented in
:mod:`vmec_jax.core.geometry` and :mod:`vmec_jax.core.setup`.

Angular grids and Fourier transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Uniform tensor-product grids in :math:`\theta` and :math:`\zeta` (one field
period) with VMEC's symmetry-reduced theta extent. VMEC does **not** use a
plain FFT: the analysis transform (``tomnsps``) is a two-stage weighted DFT
with endpoint half-weights and ``mscale/nscale`` normalization baked into
precomputed trig tables (``fixaray.f``):

.. math::

   \tilde{f}^{c}_{m}(\zeta_k) = \sum_i f(\theta_i,\zeta_k)\,\mathrm{cosmui}_{i,m},
   \qquad
   c_{m,n} = \sum_k \tilde{f}^{c}_m(\zeta_k)\,\mathrm{cosnv}_{k,n},

and analogously for the sine tables. Derivative terms use ``cosnvn/sinnvn``,
which already include the field-period scaling :math:`n\,\mathrm{NFP}`. The
tables are built in :mod:`vmec_jax.core.fourier` and the transforms
(``totzsps/totzspa`` synthesis, ``tomnsps/tomnspa`` analysis) are batched
``dot_general`` matmuls in :mod:`vmec_jax.core.transforms` — GEMM-friendly
and XLA-fusable while matching VMEC2000 normalization exactly.

Geometry pipeline
~~~~~~~~~~~~~~~~~

Per iteration (:mod:`vmec_jax.core.geometry`, VMEC2000 ``jacobian.f``):

1. synthesize :math:`(R,Z,\lambda)` and their angular derivatives on the
   ``(s,\theta,\zeta)`` grid from the spectral state;
2. form radial derivatives on the half mesh;
3. compute the half-mesh Jacobian :math:`\sqrt{g}` and the metric elements
   ``guu, guv, gvv``;
4. evaluate the Jacobian sign proxy :math:`\tau`; a sign change away from the
   axis flags a bad Jacobian (``irst = 2``).

The iteration loop
------------------

VMEC solves the fixed-boundary equilibrium by evolving the stacked Fourier
coefficients :math:`\mathbf{x}` of :math:`(R,Z,\lambda)` with a
preconditioned, damped second-order Richardson iteration:

.. math::

   \mathbf{v}_{k+1} = \frac{1-d_k}{1+d_k}\,\mathbf{v}_k
                      + \frac{\Delta t}{1+d_k}\,P^{-1}\mathbf{r}(\mathbf{x}_k),
   \qquad
   \mathbf{x}_{k+1} = \mathbf{x}_k + \Delta t\,\mathbf{v}_{k+1},

where :math:`\mathbf{r}` is the spectral force residual and the damping

.. math::

   d_k = \tfrac{1}{2}\,\Delta t\,\langle \mathrm{otau}\rangle,
   \qquad
   \mathrm{otau} \leftarrow \min\!\left(\left|\log\frac{\mathrm{fsq}_k}{\mathrm{fsq}_{k-1}}\right| / \Delta t,\; \frac{0.15}{\Delta t}\right)

is averaged over the last ``ndamp = 10`` steps (``evolve.f``). This is
implemented in :mod:`vmec_jax.core.step`.

Convergence is declared when the *physical* residuals satisfy
``fsqr, fsqz, fsql <= ftolv`` simultaneously; the residual norms and the m=1
constraint rotation follow ``residue.f90`` (:mod:`vmec_jax.core.residuals`).

Restart control (``restart.f``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The loop keeps a checkpoint of the best state and applies VMEC2000's exact
back-off rules (:mod:`vmec_jax.core.step`):

- **bad Jacobian** (``irst = 2``): restore the checkpoint, zero the velocity,
  ``delt *= 0.90``; on the first bad Jacobian the axis guess is recomputed
  (``guess_axis``), and ``delt`` is reset at ``ijacob = 25, 50`` with a hard
  stop at 75 (``jac75_flag``);
- **residual blow-up** (``irst = 3``): if after more than 10 steps the
  residual exceeds :math:`10^4\times` the checkpoint value, restore and
  ``delt /= 1.03``.

1D radial preconditioner (``precondn.f``, ``scalfor.f``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The R/Z forces are preconditioned per ``(m,n)`` column with a radial
tridiagonal operator

.. math::

   D_{mn}(s) = a_{xd} + b_{xd}\,m^2 + c_x\,(n\,\mathrm{NFP})^2,

whose coefficients are built from flux-surface integrals of
:math:`p_\tau = r_{12}^2\,B^2\,w/\sqrt{g}`-type quantities, with the
``edge_pedestal = 0.05`` and ZC(0,0)(ns) ``fac = 0.25`` stabilizations of
``scalfor.f``. The solve is a Thomas algorithm vectorized over all spectral
columns (:func:`vmec_jax.core.preconditioner.tridiagonal_solve`).
:math:`\lambda` uses the diagonal ``faclam`` factors from ``lamcal.f90``
(:math:`\propto 1/(b_\lambda (n\,\mathrm{NFP})^2 + c_\lambda m^2 \pm 2mn\,\mathrm{NFP})`,
:math:`\sqrt{s}`-damped for :math:`m > 16`).

Preconditioner matrices, force norms, and the constraint multiplier ``tcon``
are recomputed every ``ns4 = 25`` iterations and reused in between — this
cadence is parity-critical and is mirrored exactly.

Spectral condensation (``alias.f``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The constraint force that drives spectral condensation is assembled from

.. math::

   z_{\mathrm{temp}} = (r_{\mathrm{con}} - r_{\mathrm{con},0})\,r_{\theta,0}
                     + (z_{\mathrm{con}} - z_{\mathrm{con},0})\,z_{\theta,0},

spectrally filtered to :math:`m \in [1, \mathrm{mpol}-2]` with the
``faccon(m)`` weights, and scaled per surface by

.. math::

   \mathrm{tcon}(j) = \min\!\left(\left|\frac{a_{rd}}{a_{r,\mathrm{norm}}}\right|,
   \left|\frac{a_{zd}}{a_{z,\mathrm{norm}}}\right|\right)\cdot
   \mathrm{tcon}_0\text{-scaled}\cdot(32\,h_s)^2,
   \qquad \mathrm{tcon}(ns) = \tfrac{1}{2}\,\mathrm{tcon}(ns-1)

(``bcovar.f``). Implemented in :mod:`vmec_jax.core.forces` (constraint force)
and :mod:`vmec_jax.core.fields` (``tcon``).

Two execution lanes, one physics
--------------------------------

:mod:`vmec_jax.core.solver` exposes the same jitted iteration through two
lanes (selected by ``vmec --mode cli|jit``):

- **CLI lane** (default): a Python ``while`` loop around a jitted
  *N-iteration block* kernel, with residuals checked on the host between
  blocks. This gives exact-``ftol`` early exit, live VMEC2000-format printing
  every ``NSTEP`` iterations, and buffer donation, with zero autodiff
  bookkeeping.
- **JIT lane**: a single ``lax.while_loop`` over the same physics — fully
  traceable, used as the forward solver inside the differentiable API.

A regression test asserts per-block state agreement between the lanes to
machine precision.

Multigrid and hot restart (``runvmec.f``, ``interp.f``)
-------------------------------------------------------

:mod:`vmec_jax.core.multigrid` runs the ``NS_ARRAY`` ladder: each stage
solves at its ``ns`` with its own ``FTOL_ARRAY``/``NITER_ARRAY`` entry, and
the converged coefficients are interpolated in :math:`\sqrt{s}`-internal form
to the next finer grid (``interp.f``). Mode and radial arrays are padded to
the maximum resolution so all stages share one compiled executable — the
ladder pays a single JIT compile. The same interpolation seam provides hot
restart: a previous solution (e.g. the previous point of a parameter scan)
can seed the solve directly.

Free boundary (NESTOR)
----------------------

For ``LFREEB = T`` decks, :mod:`vmec_jax.core.vacuum` implements Merkel's
Green's-function method (NESTOR): the scalar magnetic potential on the plasma
boundary is expanded in Fourier harmonics, the singular integrals are handled
with the ``analyt.f`` analytic decomposition, and the resulting dense system
is solved for ``potvac``. :mod:`vmec_jax.core.freeboundary` drives the
coupling with the VMEC2000 cadence (``funct3d.f``):

- the vacuum solve activates once :math:`\mathrm{fsqr}+\mathrm{fsqz} \le 10^{-3}`;
- a **full** NESTOR solve runs when ``mod(iter2 - iter1, nvacskip) == 0``,
  with cheaper incremental updates in between, and the cadence adapts as

  .. math::

     \mathrm{nvacskip} \leftarrow \max\!\left(\mathrm{nvskip}_0,\;
     \frac{1}{\max(0.1,\; 10^{11}\,(\mathrm{fsqr}+\mathrm{fsqz}))}\right);

- the vacuum pressure enters the edge force through
  ``rbsq = bsqvac + presf(ns)`` at ``js = ns``, and the constraint reference
  surfaces ``rcon0, zcon0`` ramp by 0.9 per iteration.

The external field comes either from an ``mgrid`` file
(:mod:`vmec_jax.core.mgrid`, trilinear interpolation weighted by ``EXTCUR``)
or directly from a Biot-Savart coil set (:mod:`vmec_jax.core.coils`, ESSOS
layout) — the latter is interpolation-free and differentiable through the
coil parameters.

Implicit differentiation
------------------------

Gradients of equilibrium properties are computed by implicit differentiation
of the converged fixed point (:mod:`vmec_jax.core.implicit`). The equilibrium
is the root of the preconditioned force residual :math:`F(x, p) = 0` with
:math:`x` the spectral state and :math:`p` the parameters (boundary
coefficients, profiles, ``phiedge``, coil currents/geometry). The
``custom_vjp`` wrapper:

- **forward**: runs the fast CLI-lane solver (opaque to autodiff) and returns
  the converged :math:`x^\star`;
- **backward**: solves the adjoint system
  :math:`(\partial F/\partial x)^{\!\top}\lambda = \bar{g}` matrix-free with
  preconditioned GMRES (Jacobian-vector products via ``jax.vjp`` on the
  residual function, the 1D radial preconditioner as the GMRES
  preconditioner), then returns :math:`-\lambda^{\!\top}\,\partial F/\partial p`
  with one more VJP.

The cost is a handful of residual evaluations per gradient with O(1) memory
in the iteration count; multigrid stages act purely as an initializer and are
stop-gradient by construction. Gradient accuracy is validated against
central finite differences in CI. See :doc:`optimization` for usage and the
references (Skene & Burns 2026; jaxopt; DESC) in :doc:`references`.
