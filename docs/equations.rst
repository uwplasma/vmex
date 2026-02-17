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
See References [1-3] for the original VMEC formulation.

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

Internal scaling and regularity (``scalxc``)
--------------------------------------------

VMEC enforces regularity at the magnetic axis by storing **odd-m** contributions
in an *internal* form that factors out :math:`\sqrt{s}`:

.. math::

   X(s,\theta,\zeta) =
   X_{\mathrm{even}}(s,\theta,\zeta)
   + \sqrt{s}\,X_{\mathrm{odd,int}}(s,\theta,\zeta).

Equivalently,

.. math::

   X_{\mathrm{odd,int}} = \frac{X_{\mathrm{odd,phys}}}{\sqrt{s}}.

VMEC implements this via the ``scalxc`` array, which is 1 for even-m harmonics
and :math:`1/\sqrt{s}` for odd-m harmonics. ``scalxc`` is applied when
interpolating coefficients between radial grids and when assembling
preconditioned residuals (VMEC2000 ``profil3d`` / ``interp`` / ``scalxc``).

On the axis, VMEC applies *odd-m* rules:

- :math:`m=1`: extrapolate the internal odd field to the axis by copying the
  first off-axis value,
- :math:`m\ge 2`: force the internal odd field to zero on-axis.

m=1 internal constraint (``lconm1``)
-----------------------------------

When ``LCONM1`` is enabled (VMEC default for 3D runs), VMEC stores the *m=1*
boundary coefficients in a constrained internal basis:

.. math::

   R^{s}_{1n,\mathrm{int}} = \frac{1}{2}\left(R^{s}_{1n,\mathrm{phys}} + Z^{c}_{1n,\mathrm{phys}}\right),
   \qquad
   Z^{c}_{1n,\mathrm{int}} = \frac{1}{2}\left(R^{s}_{1n,\mathrm{phys}} - Z^{c}_{1n,\mathrm{phys}}\right).

This transformation is applied in VMEC2000 ``readin`` and inverted when
converting to physical coefficients for diagnostics. ``vmec_jax`` uses the same
internal basis so that boundary handling and multigrid interpolation match
VMEC2000.

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

``bcovar`` + ``add_fluxes`` (poloidal flux correction)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VMEC updates the **contravariant** :math:`B^u` using the *full-mesh* poloidal
flux function :math:`\chi'(s)` (``chips``). In VMEC2000 ``add_fluxes``,
``chips`` is computed from force balance on each surface:

.. math::

   \chi'(s) = \frac{I_\varphi(s) - \langle g^{uu} B_u + g^{uv} B_v \rangle}
                   {\langle g^{uu}/\sqrt{g} \rangle},

where the angle brackets denote the VMEC surface quadrature, and
:math:`I_\varphi(s)` is the integrated toroidal current (``icurv``).
VMEC then applies the correction

.. math::

   B^u \leftarrow B^u + \chi'(s)\,\frac{1}{\sqrt{g}}.

VMEC stores the *half-mesh* averaged ``chipf`` in ``wout``; ``vmec_jax``
follows VMEC’s averaging rules to convert between ``chipf`` and ``chips``.

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

Time-step control (Garabedian update)
-------------------------------------

VMEC’s nonlinear fixed-boundary iteration uses a Garabedian-style conjugate
gradient update with a time-step control mechanism. The update computes

.. math::

   \tau_n = \min\left(\left|\ln\frac{\mathrm{fsq}_n}{\mathrm{fsq}_{n-1}}\right|,\; 0.15\right),

and maintains a moving average :math:`\overline{\tau}` over ``ndamp`` steps.
The damping factor is

.. math::

   \Delta\tau = \frac{\Delta t\,\overline{\tau}}{2}, \qquad
   b_1 = 1-\Delta\tau, \qquad
   \mathrm{fac} = \frac{1}{1+\Delta\tau}.

The update in VMEC2000 is:

.. math::

   \dot{x} \leftarrow \mathrm{fac}\,\bigl(b_1\dot{x} + \Delta t\,F\bigr), \qquad
   x \leftarrow x + \Delta t\,\dot{x},

where :math:`F` is the preconditioned residual vector (``gc``).

VMEC’s ``TimeStepControl`` tracks the minimum of the preconditioned residual
(``res0``) and the physical residual (``res1``). If either grows by a factor of
``1e4`` after 10 steps, VMEC restores the last good state and reduces
``DELT`` by a factor of 1.03. ``vmec_jax`` mirrors this logic to reproduce the
VMEC2000 iteration trace.

Multigrid interpolation (``interp.f``)
--------------------------------------

VMEC’s multigrid staging interpolates *scaled* coefficients between grids:

.. math::

   x_{\mathrm{scaled}} = x \cdot \mathrm{scalxc},

with odd-m extrapolation to the axis performed **before** interpolation.
After linear interpolation on a uniform radial grid, coefficients are unscaled:

.. math::

   x_{\mathrm{new}} = \frac{x_{\mathrm{scaled,new}}}{\mathrm{scalxc}_{\mathrm{new}}}.

``vmec_jax`` implements this exact pipeline so that stage-to-stage coefficient
transfer matches VMEC2000.

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

Jacobian sign check (``tau``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VMEC evaluates an auxiliary Jacobian-like scalar :math:`\tau` built from
even/odd-m real-space derivatives (see VMEC2000 ``jacobian.f``). In compact
form,

.. math::

   \tau \equiv (R_u Z_s - R_s Z_u) + \text{(odd-m corrections in } \sqrt{s}\text{)}.

If :math:`\tau` changes sign away from the axis, VMEC flags a bad Jacobian and
restarts the iteration with a refined axis guess. ``vmec_jax`` reproduces the
same parity split, half-mesh averaging, and sign check so that Jacobian-reset
behavior matches VMEC2000.

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
