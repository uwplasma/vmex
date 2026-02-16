Equations and derivations
=========================

This page states the **explicit equations** solved (or approximated) by
``vmec_jax`` and connects them to the VMEC2000 formulation. The goal is to make
the physics, the coordinate conventions, and the force-balance residuals fully
transparent so that parity checks can be done equation-by-equation.

Notation and conventions
------------------------

We use VMEC's curvilinear flux coordinates:

- :math:`s \in [0,1]`: normalized toroidal flux label (VMEC radial coordinate).
- :math:`\theta \in [0,2\pi)`: poloidal angle.
- :math:`\zeta \in [0,2\pi)`: *field-period* toroidal angle.
- :math:`\phi_{\mathrm{phys}} = \zeta/\mathrm{NFP}`: physical toroidal angle.

The Fourier phase convention is:

.. math::

   \mathrm{phase}(m,n;\theta,\zeta) = m\theta - n\zeta,

where :math:`n` is the field-period toroidal mode number
(VMEC stores :math:`xn = n\,\mathrm{NFP}` in ``wout``).

Ideal MHD equilibrium
---------------------

The ideal MHD equilibrium is defined by:

.. math::

   \nabla p = \mathbf{J} \times \mathbf{B},

with Maxwell's equations (in magnetostatic form):

.. math::

   \nabla \cdot \mathbf{B} = 0, \qquad
   \mathbf{J} = \frac{1}{\mu_0}\nabla \times \mathbf{B}.

The pressure is a **flux function**: :math:`p = p(s)` and is specified by the
VMEC input profiles. VMEC (and ``vmec_jax``) use pressure in units of
:math:`\mu_0\,\mathrm{Pa}` so that :math:`p` has the same units as :math:`B^2`.

Energy principle (VMEC formulation)
-----------------------------------

VMEC solves for a stationary point of the ideal-MHD energy functional in
straight-field-line coordinates:

.. math::

   W = \frac{1}{(2\pi)^2}\int \left(\frac{B^2}{2} + \frac{p}{\gamma-1}\right) dV,

where :math:`\gamma` is the ratio of specific heats (VMEC input ``GAMMA``).
The VMEC fixed-boundary update loop can be viewed as a (preconditioned)
steepest-descent method that drives the force residuals to zero.

Flux coordinates and straight-field-line angle
----------------------------------------------

VMEC introduces a scalar field :math:`\lambda(s,\theta,\zeta)` to define the
straight-field-line poloidal angle:

.. math::

   u = \theta + \lambda(s,\theta,\zeta).

Field lines are straight in :math:`(u,\zeta)`:

.. math::

   \frac{du}{d\zeta} = \iota(s),

where :math:`\iota(s)` is the rotational transform.

Magnetic field representation
-----------------------------

In VMEC's flux-coordinate representation, the magnetic field has **no radial
contravariant component**:

.. math::

   B^s = 0.

VMEC therefore stores only the contravariant components in the angular
directions:

.. math::

   \mathbf{B} = B^u \nabla u + B^v \nabla v, \qquad v \equiv \zeta.

In terms of VMEC's flux functions
:math:`\Phi(s)` (toroidal flux) and :math:`\chi(s)` (poloidal flux), we define:

.. math::

   \Phi'(s) \equiv \frac{d\Phi}{ds} \quad (\text{``phipf''}), \qquad
   \chi'(s) \equiv \frac{d\chi}{ds} \quad (\text{``chipf''}).

VMEC's **contravariant** components (``bsupu`` and ``bsupv`` in ``wout``)
are computed as:

.. math::

   B^v = \frac{\Phi'(s) + \mathrm{lamscale}\,\partial_{\theta}\lambda}
                {\mathrm{signgs}\,\sqrt{g}\,2\pi},

.. math::

   B^u = \frac{\chi'(s) - \mathrm{lamscale}\,\partial_{\zeta}\lambda}
                {\mathrm{signgs}\,\sqrt{g}\,2\pi}.

Here:

- :math:`\sqrt{g}` is the signed Jacobian,
- ``signgs`` is VMEC's sign convention such that ``signgs*sqrtg`` is positive
  away from the magnetic axis,
- ``lamscale`` is the VMEC scaling applied to :math:`\lambda` derivatives
  (stored in ``wout`` and used by ``vmec_jax`` for parity).

The **covariant** components are defined by:

.. math::

   B_i = \mathbf{B}\cdot \mathbf{e}_i
       = \sum_{j} g_{ij} B^j, \qquad i,j \in \{s,u,v\},

where :math:`g_{ij}` is the covariant metric and
:math:`\mathbf{e}_i = \partial_i \mathbf{r}`.
VMEC stores these as ``bsub*`` in ``wout``.

Current density
---------------

The current density follows directly from the curl:

.. math::

   \mathbf{J} = \frac{1}{\mu_0} \nabla \times \mathbf{B}.

VMEC reports covariant current components in ``wout`` as ``jcuru`` and
``jcurv`` (poloidal and toroidal current densities on the half mesh) and uses
these in the force kernels. The parallel and perpendicular currents satisfy:

.. math::

   \mathbf{J} = \mathbf{J}_{\parallel} + \mathbf{J}_{\perp}, \qquad
   \nabla p = \mathbf{J}_{\perp} \times \mathbf{B}.

Force balance in VMEC (residual form)
-------------------------------------

VMEC evaluates the force balance in real space, then transforms the residual
forces back to Fourier space. In VMEC2000, these residuals are packaged as
``tomnsps`` Fourier arrays and projected into three directions:

- :math:`F_R`: radial (R) force balance residual,
- :math:`F_Z`: vertical (Z) force balance residual,
- :math:`F_\lambda`: stream-function (lambda) residual.

These are combined into scalar norms that appear in the VMEC screen output:

.. math::

   \mathrm{FSQR} = r_1\,\mathrm{fnorm}\,\lVert F_R \rVert^2, \qquad
   \mathrm{FSQZ} = r_1\,\mathrm{fnorm}\,\lVert F_Z \rVert^2,

.. math::

   \mathrm{FSQL} = \mathrm{fnormL}\,\lVert F_\lambda \rVert^2,

where ``fnorm``, ``fnormL``, and ``r1`` are the VMEC normalization factors
computed from ``bcovar`` (half-mesh metrics + ``bsup``/``bsub``).
``vmec_jax`` reproduces these scalars from the same internal quantities to
match VMEC2000's per-iteration printout.

Pressure and beta
-----------------

VMEC reports thermal and magnetic energy scalars in ``wout``:

.. math::

   W_B = \frac{1}{(2\pi)^2}\int \frac{B^2}{2}\,dV, \qquad
   W_P = \frac{1}{(2\pi)^2}\int p\,dV.

The total volume-averaged beta is computed by VMEC as:

.. math::

   \beta_{\mathrm{total}} = \frac{W_P}{W_B}.

``vmec_jax`` follows the same normalization when emitting ``wout`` files.

Geometric embedding and Jacobian
--------------------------------

Surfaces are represented in cylindrical coordinates by Fourier series:

.. math::

   R(s,\theta,\zeta) = \sum_{m,n}\left[
     R_{mn}^c(s)\cos(m\theta-n\zeta) + R_{mn}^s(s)\sin(m\theta-n\zeta)
   \right],

.. math::

   Z(s,\theta,\zeta) = \sum_{m,n}\left[
     Z_{mn}^c(s)\cos(m\theta-n\zeta) + Z_{mn}^s(s)\sin(m\theta-n\zeta)
   \right].

We embed into Cartesian coordinates using the physical toroidal angle:

.. math::

   x = R\cos\phi_{\mathrm{phys}}, \qquad
   y = R\sin\phi_{\mathrm{phys}}, \qquad
   z = Z.

The Jacobian is:

.. math::

   \sqrt{g} = \mathbf{e}_s \cdot (\mathbf{e}_\theta \times \mathbf{e}_{\phi}).

VMEC's sign convention enforces ``signgs*sqrtg > 0`` away from the axis.

Implementation mapping (``vmec_jax``)
-------------------------------------

Key modules that directly implement the equations above:

- ``vmec_jax/geom.py``: geometry, metric, Jacobian.
- ``vmec_jax/vmec_bcovar.py``: contravariant/covariant field components.
- ``vmec_jax/vmec_forces.py``: real-space force kernels (``A,B,C`` blocks).
- ``vmec_jax/vmec_tomnsp.py``: VMEC-style Fourier transforms of forces.
- ``vmec_jax/vmec_residue.py``: VMEC scalar residuals (FSQR/FSQZ/FSQL).
- ``vmec_jax/solve.py``: fixed-boundary iteration loop (VMEC2000 parity path).

References (local)
------------------

- ``/Users/rogeriojorge/local/test/vmecpp/docs/the_numerics_of_vmecpp.pdf``
  (the VMEC++ numerics notes; VMEC2000 conventions and formulas).
- VMEC2000 source in ``/Users/rogeriojorge/local/test/STELLOPT/VMEC2000/Sources``.
