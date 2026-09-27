High-order strong-force balance
===============================

VMEX's legacy solver establishes stationarity of the discrete VMEC energy on
a staggered, uniform ``s`` mesh.  A small ``FSQR/FSQZ/FSQL`` certifies those
projected discrete equations; it does not bound the continuum force
``J x B - grad(p)``.  The optional polish keeps that fast solver as the
initializer, re-solves the continuum force on a native high-order
representation, and certifies the result with an independent oracle.

Representation
--------------

The continuous coordinates are ``(rho, theta, zeta)``, ``rho = sqrt(s)``, with
``zeta`` spanning one field period.  Each real Fourier amplitude is

.. math::

   X_{mn}(\rho) = \rho^{|m|} q_{mn}(s), \qquad
   q_{mn}(s) = \sum_k c_{kmn} B_k(s),

with clamped B-splines ``B_k`` (quintic in the polish).  The analytic factor
``rho**abs(m)`` gives exact magnetic-axis regularity.  The lift from the legacy
state undoes VMEX's ``m=1`` constrained variables and Fourier normalization,
keeps the magnetic axis and the fixed boundary exact, and removes the lambda
``(0,0)`` gauge coefficient structurally.

Independent certificate
-----------------------

:func:`~vmex.core.strong_force.certify_strong_force` evaluates

.. math::

   \mathbf{F} = \frac{(\nabla\times\mathbf{B})\times\mathbf{B}}{\mu_0} - \nabla p

pointwise by nested automatic differentiation of the spline-Fourier geometry,
on Gauss nodes of order ``degree + 3`` per span and angular grids offset from
every solve node, and reports the dimensional volume-RMS force, its
near-axis/bulk/edge parts, a flux-surface profile, a radial-quadrature
difference and the signed-Jacobian margin.  Its pointwise ratio

.. math::

   \varepsilon_F = \frac{2|\mathbf{F}|}
        {|\mathbf{J}\times\mathbf{B}|+|\nabla p|+F_{\mathrm{floor}}}

is bounded above by 2 by construction and saturates in vacuum, so it must
never be quoted on its own: read the dimensional force and the
``<|F|>/<|grad(B^2/2mu0)|>`` averages with it.

The polish
----------

:mod:`vmex.core.polish` minimizes the volume-weighted force

.. math::

   r_i(c) = \sqrt{\frac{w_i |\sqrt{g_i}|}{V}}\,\frac{\mathbf{F}_i(c)}{F_*},
   \qquad \min_c \tfrac12 |r|^2 \quad \text{subject to} \quad C c = 0,

over the packed spline coefficients ``c``, where ``F* = volavgB**2 / (mu0
Aminor_p)`` and ``V`` is the plasma volume (the wout definitions) and ``C`` is a
frozen linear tangential gauge that removes the relabeling freedom of the
flux-surface angles.  The steps are:

1. pad the poloidal modes (zero coefficients, an exact refinement);
2. per chart, take Gauss--Newton steps on the KKT system
   ``[[A^T A, C^T], [C, 0]]`` with backtracking on ``|r|``;
3. insert knots at the midpoints of the spans with the largest force (exact
   Boehm insertion) and start a new chart, until ``|r|`` meets its tolerance;
4. take one exact-Hessian Newton step on the stationarity equations
   ``[A^T r + C^T nu, C c] = 0`` (GMRES preconditioned by the Gauss--Newton
   factor).

The Jacobian is assembled by partial assembly, ``A = D S``: each node's force
depends on the coefficients only through 30 jets (value, first and second
derivatives of ``R``, ``Z`` and ``lambda``), so ``D`` is the pointwise jet
derivative (30 forward tangents) and ``S`` the fixed spline-by-Fourier
synthesis; each radial span contributes one dense block of ``A^T A``.  Radial
derivatives are formed from coefficient differences and the base and
correction jets are synthesized separately, which keeps the evaluated
stationarity accurate to about ``1e-12`` instead of the ``1e-7`` floor of
contracting O(1) coefficients with derivative tables.

Acceptance
----------

A polish is accepted when, together,

* the independent volume-RMS force over ``F*`` is at most
  ``PolishConfig.force_tolerance`` (``1e-5``);
* the Frobenius-scaled projected stationarity
  ``eta = |P A^T r| / (|A|_F |r|)`` is at most
  ``PolishConfig.stationarity_tolerance`` (``1e-8``), with ``P`` the projector
  onto ``ker C``; and
* the signed Jacobian stays positive.

Otherwise the driver raises
:class:`~vmex.core.errors.StrongForceCertificationError`, or returns the
unpolished state with ``fail_policy="return_unpolished"``.

Scope and cost
--------------

The polish covers fixed-boundary axisymmetric decks with prescribed pressure
and iota (``NCURR = 0``, ``GAMMA = 0``, ``LASYM = F``).  On
``input.shaped_tokamak_pressure`` it certifies ``|F|/F* = 7.1e-6`` and
``eta = 1.3e-10`` in 68 s end to end with an empty compilation cache and 32 s
with a populated one (the ordinary solve alone takes about 5 s).  The polished
WOUT samples the native state on enough radial surfaces to resolve the
narrowest inserted span and widens ``MPOL`` to hold the padded modes, so the
file carries the certified state.  Non-axisymmetric equilibria need a
matrix-free Gauss--Newton on the same ``A = D S`` structure and are future
work.
