Confinement physics: quasisymmetry, omnigenity, stability
=========================================================

The pages :doc:`variational-problem` and :doc:`iteration` derive the *solver* physics —
the energy functional whose stationary point is an equilibrium, and the
numerics that find it. This page derives the **target functionals** that a
stellarator-design campaign puts in front of that equilibrium: the confinement
and stability metrics of :mod:`vmex.core.omnigenity`,
:mod:`vmex.core.optimize`, :mod:`vmex.core.nyquist`,
:mod:`vmex.core.stability` and :mod:`vmex.core.bootstrap`. It is the
first-principles companion to :doc:`/reference/objectives`, which catalogs the same
quantities as ready-to-use optimizer terms.

The unifying object is the field strength :math:`|B|` expressed in **Boozer
coordinates**: quasisymmetry, omnigenity, and the neoclassical transport that
they control are all statements about the angular structure of :math:`|B|` at
fixed flux label, and are cleanest in the coordinates in which field lines are
straight *and* the parallel current is a flux function.

.. contents:: On this page
   :local:
   :depth: 1


Boozer coordinates
------------------

Motivation
~~~~~~~~~~

Guiding-center orbits — and therefore neoclassical transport, fast-ion
confinement, and the bootstrap current — depend on the magnetic geometry almost
entirely through the variation of :math:`|B|` along and across field lines.
Boozer coordinates :math:`(s, \theta_B, \zeta_B)` are the flux coordinates that
make this variation maximally transparent: field lines are straight,

.. math::

   \frac{d\theta_B}{d\zeta_B} = \iota(s),

and the covariant representation of :math:`\mathbf B` collapses to the two
surface functions

.. math::

   \mathbf B = \nabla\psi\times\nabla\theta_B + \iota\,\nabla\zeta_B\times\nabla\psi
             = I(s)\,\nabla\theta_B + G(s)\,\nabla\zeta_B + \beta_*\,\nabla\psi,

where :math:`2\pi I(s)` is the toroidal current inside :math:`s`, :math:`2\pi
G(s)` the poloidal current outside it (VMEC's ``buco`` and ``bvco``,
:func:`~vmex.core.fields.surface_currents`), and :math:`\psi=\Phi/2\pi` is
the toroidal flux per radian. Because :math:`I` and :math:`G` are flux
functions, the magnetic differential equation for the coordinate shift is
linear and can be solved surface by surface.

Construction
~~~~~~~~~~~~

Starting from the straight-field-line (VMEC) angles :math:`(\theta,\zeta)` with
the renormalized poloidal angle :math:`u=\theta+\lambda`, the Boozer angles
differ by a single periodic scalar :math:`\nu(s,\theta,\zeta)`:

.. math::

   \theta_B = \theta + \lambda + \iota\,\nu, \qquad
   \zeta_B  = \zeta + \nu .

The shift :math:`\nu` follows from the covariant field. Introduce the periodic
part :math:`w` of the Boozer generating potential, fixed by

.. math::

   \frac{\partial w}{\partial\theta} = B_\theta, \qquad
   \frac{\partial w}{\partial\zeta}  = B_\zeta,

with :math:`B_\theta,B_\zeta` the covariant components on the surface
(:func:`~vmex.core.fields.magnetic_fields`). ``booz_xform_jax``'s kernel
inverts this spectrally: the non-axisymmetric (:math:`m\ne 0`) harmonics of
:math:`w` come from :math:`B_\theta` and the axisymmetric (:math:`m=0`)
harmonics from :math:`B_\zeta`, matching the mode split of the Fortran
``booz_xform``. The coordinate shift is then

.. math::

   \nu = \frac{w - I(s)\,\lambda}{G(s) + \iota(s)\,I(s)} .

Finally the Boozer :math:`|B|` spectrum is obtained by the angle-transform
quadrature, weighting :math:`|B|` (evaluated on the VMEC grid) by the Jacobian
of the angle map:

.. math::

   \hat B^{B}_{mn}(s) = \Bigl\langle\,|B|\,
      \cos\!\bigl(m\,\theta_B - n\,\zeta_B\bigr)\,
      \frac{\partial(\theta_B,\zeta_B)}{\partial(\theta,\zeta)}\Bigr\rangle,

the angle bracket being the surface average over the original
:math:`(\theta,\zeta)` grid. These are the ``bmnc_b`` coefficients (physical
mode numbers ``xm_b``, ``xn_b``) consumed by every metric below.

One transform implementation — ``booz_xform_jax``'s — executes these
equations; vmex only feeds it. The host driver
:func:`vmex.core.boozer.run_booz_xform` calls it on a ``wout_*.nc`` file and
writes a standard ``boozmn_*.nc`` (used by ``vmex --booz`` and by
:func:`~vmex.core.optimize.quasi_isodynamic_residual_from_wout`). The
traceable :func:`~vmex.core.omnigenity.boozer_spectrum_state` calls the same
jittable kernel in memory on wout-convention tables built from the solver's
internal state (:func:`~vmex.core.boozer_tables.boozer_input_tables`), so
the Boozer spectrum — and any metric built on it — carries exact implicit
gradients. The two routes differ only through the table construction (the
wout file stores half-mesh finite-difference averages of some families) and
agree on the bundled QI deck to a measured :math:`1.7\times10^{-6}\,B_{00}`
absolute and :math:`2\times10^{-5}` relative on the dominant modes.


Quasisymmetry
-------------

Confinement rationale
~~~~~~~~~~~~~~~~~~~~~~

A field is **quasisymmetric** when :math:`|B|` depends on the two Boozer angles
only through a single linear combination,

.. math::

   |B| = |B|\bigl(s,\; M\theta_B - N\zeta_B\bigr),

for a fixed integer helicity :math:`(M,N)`. The magnitude, not the vector
field, need only be symmetric: this is enough for the guiding-center Lagrangian
to acquire an ignorable angle, hence a conserved canonical momentum and confined
collisionless orbits — the same confinement a true axisymmetric tokamak enjoys,
but in a compact 3D device. The helicity fixes the family
(:class:`~vmex.core.optimize.QuasisymmetryRatioResidual`, ``helicity_n`` in
units of ``nfp``):

.. list-table::
   :header-rows: 1
   :widths: 26 20 54

   * - family
     - :math:`(M,N)`
     - :math:`|B|` contours in Boozer angles
   * - QA (quasi-axisymmetric)
     - :math:`(1,0)`
     - close toroidally (tokamak-like)
   * - QH (quasi-helical)
     - :math:`(1,\pm\mathrm{nfp})`
     - close helically
   * - QP (quasi-poloidal)
     - :math:`(0,1)`
     - close poloidally (QI-like)

The two-term residual
~~~~~~~~~~~~~~~~~~~~~~

Rather than Fourier-filter the Boozer spectrum, ``vmex`` uses the
Landreman–Paul *two-term* local residual, which is an exact pointwise
diagnostic of the same condition and needs no mode truncation. On each
requested surface, sampled on a uniform :math:`(\theta,\phi)` grid,

.. math::

   f_{\mathrm{QS}} =
   \frac{(\mathbf B\times\nabla B\cdot\nabla\psi)\,(N-\iota M)
       - (\mathbf B\cdot\nabla B)\,(M G + N I)}{B^{3}},

with :math:`M=` ``helicity_m``, :math:`N=` ``helicity_n``\ :math:`\times`\
``nfp``, and :math:`G,I` the Boozer covariant averages ``bvco``/``buco``; this
is the residual inside eq. (1) of Landreman & Paul, Phys. Rev. Lett. 128,
035001 (2022). The key algebraic fact is that :math:`f_{\mathrm{QS}}`
**vanishes identically iff** :math:`|B|` is quasisymmetric with helicity
:math:`(M,N)` (Helander, Rep. Prog. Phys. 77, 087001 (2014)); there is no
residual symmetry-breaking harmonic left to penalize. The flux-surface sum
:math:`\sum f_{\mathrm{QS}}^2`, weighted by the surface measure
:math:`\sqrt{\mathrm{nfp}\,\Delta\theta\,\Delta\phi\,|\sqrt g|/V'}`, is
simsopt's ``QuasisymmetryRatioResidual`` A/B formula on the same grid with the
same weighting. No simsopt oracle runs in this test suite, so the gated
comparison is VMEX's own traceable lane against its wout lane
(``tests/test_optimize_traceable_qs.py``). Kept in
Gauss–Newton (per-point) form, it feeds the least-squares driver as an exact
residual vector rather than a pre-summed scalar. The metric is evaluated from
the parity-proven wout tables (:mod:`vmex.core.nyquist`) and also exposes a
traceable ``residuals_state`` lane, so the *same* term optimizes under both
``jac=None`` and ``jac="implicit"`` (see :doc:`/reference/objectives`).


Omnigenity and quasi-isodynamicity
----------------------------------

Omnigenity generalizes quasisymmetry: it asks only that the **bounce-averaged
radial drift of trapped particles vanish**, without requiring a symmetry of
:math:`|B|`. Equivalently (Cary & Shasharina 1997), the second adiabatic
invariant

.. math::

   \mathcal J_\parallel(s,\alpha;B^*) = \oint v_\parallel\,d\ell,
   \qquad v_\parallel \propto \sqrt{1 - B/B^*},

is a flux function — independent of the field-line label :math:`\alpha` at every
trapping level :math:`B^*`. Every quasisymmetric field is omnigenous; the
converse is not true, which leaves omnigenity the larger (and for
poloidally-closed contours, bootstrap-suppressing) design space. A **quasi-
isodynamic** (QI) field is an omnigenous field whose :math:`|B|` contours close
**poloidally** (:math:`M=0`), so that the trapped-particle precession is purely
poloidal and the bootstrap current is small by construction — the target of the
nfp1–nfp4 decks in ``examples/data/``.

.. _confinement-mapped-pwo:

pwO contour variance
~~~~~~~~~~~~~~~~~~~~

A field is piecewise omnigenous (pwO) when it is omnigenous within each class of
magnetic wells and the classes join compatibly; `Velasco et al. (2024)
<https://arxiv.org/abs/2405.07634v2>`_ introduce the class and `Calvo et al.
(2025) <https://arxiv.org/abs/2505.02546v3>`_ use it to remove the bootstrap
current, and `Velasco et al. (2026) <https://arxiv.org/abs/2603.20125v2>`_ perturb
a quasi-axisymmetric field into one. `Liu et al. (2026)
<https://arxiv.org/abs/2603.12139v1>`_ give a squeezed coordinate map that turns
the condition into a contour fit. With :math:`N=N_{\rm fp}`, set :math:`x=\eta`
on the left branch and :math:`x=2\pi-\eta` on the right, and

.. math::

   D(x)&=(\pi-x)(1-cx),\qquad
   S(x,y)=ax\sin(y+b\sin y),\qquad \kappa=\iota D(x)/N,\\
   \theta_B&=\gamma,\\
   \zeta_{B,L}&=[\pi-S(x,\gamma+\kappa)-D(x)]/N,\\
   \zeta_{B,R}&=[\pi+S(x,-\gamma+\kappa)+D(x)]/N.

The map uses :math:`|\iota|`; a negative :math:`\iota` reads it at
:math:`-\zeta_B`, so a mirror image has the same residual. The mapped angle
:math:`\gamma` is not the field-line label :math:`\alpha=\theta_B-\iota\zeta_B`.
The parameters :math:`(a,b,c)` and the cut :math:`\eta\in[\eta_{\min},2\pi-\eta_{\min}]`
are free variables of the optimization (:ref:`objectives-mapped-pwo`). For mapped field
:math:`\mathcal B_{sij}` on uniform endpoint-free gamma nodes and midpoint eta
nodes, the residual is

.. math::

   r_{sij}=\sqrt{\frac{w_s}{N_\gamma N_\eta}}
     \frac{\mathcal B_{sij}-\overline{\mathcal B}_{sj}}{B_{00,s}},
   \qquad
   f_{\rm map}=\sum_{sij}r_{sij}^2
     =\sum_s\frac{w_s}{B_{00,s}^2}
       \left\langle\operatorname{Var}_\gamma\mathcal B_s\right\rangle_\eta,

with :math:`\sum_s w_s=1` and the live Boozer mean :math:`B_{00,s}>0`, so the
objective is invariant under field rescaling. There is no Jacobian weight: this
is neither a physical-area nor a bounce-time average.

The map is one-to-one on the torus when :math:`\zeta_B` increases along each line
of constant :math:`\theta_B` and the line spans less than one field period; the
code checks both on the nodes, and requires :math:`c<-1/\pi` and a positive
Jacobian. Equal-field points on the two branches are separated by
:math:`\Delta\zeta_B=2D(x)/N`, so a common monotone branch profile gives
field-line-independent bounce separation, the construction behind `Landreman and
Catto (2012) <https://arxiv.org/abs/1112.5725>`_. A small variance alone does not
establish common extrema, complete wells, compatible transitions or radial
action behavior, and it does not certify confinement: pair the residual with a
mirror ceiling, a transform floor and an independent neoclassical (NEO) or particle-tracing check.


Bounce action
~~~~~~~~~~~~~

:func:`vmex.core.bounce.bounce_action` evaluates the physical-pitch form

.. math::

   \mathcal J_\parallel
   = 2\int_{\ell_1}^{\ell_2}\sqrt{1-\lambda B}\,d\ell

for every complete magnetic well in a sampled field line. The factor two
closes the bounce orbit; multiply by the chosen reference speed for a
dimensional invariant. Crossings and wells retain fixed array shapes, while a
sine-mapped Gauss--Legendre rule removes the square-root singularity at each
bounce point. Invalid slots are NaN and carry explicit absent, marginal,
merged, truncated, and overflow masks.

:func:`vmex.core.bounce.bounce_action_from_boozer` synthesizes field lines from
``booz_xform``-convention harmonics. It uses
:math:`d\ell/d\zeta_B=|G+\iota I|/B`, so pitch values are shared physical
values rather than separately normalized levels. Both entry points support
JIT, forward AD, and reverse AD. Derivatives are defined within a fixed well
topology; marginal and merged masks identify topology changes that an
optimizer must exclude or resolve.

The ``--plot`` polar diagnostic instead holds the normalized trapping class
:math:`\lambda_n` fixed across radius,
:math:`1/\lambda=B_{\min}(s)+\lambda_n[B_{\max}(s)-B_{\min}(s)]`, following
Rodríguez, Helander & Goodman (2024). It plots
:math:`x=s\cos\alpha`, :math:`y=s\sin\alpha`; an omnigenous field therefore
has concentric circular contours. This display convention does not change the
physical-pitch contract of the optimization objectives.

The constructed-QI target
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because :math:`\mathcal J_\parallel` uniformity is awkward to differentiate,
:class:`~vmex.core.omnigenity.QIResidual` implements a lightweight smooth
surrogate distilled from level-set conditions used by the
*constructed-QI-target* method of Goodman *et al.* (2023). On each surface,
:math:`|B|` is sampled along Boozer field lines
:math:`\theta_B=\alpha+\iota\phi_B` over one field period, and three families of
residual — each an **exact zero of an exactly QI field** — are stacked:

* **Bounce-distance uniformity** (``well_weight``). For every trapping level
  :math:`B^*`, the distance :math:`\delta(\alpha,B^*)` between the two monotone
  branches of the magnetic well, minus its field-line average. This is the
  Cary–Shasharina condition (constant well width at fixed :math:`B^*` across
  field lines) that Goodman's "shuffle" step enforces; the branch envelopes are
  built from smooth running-maximum occupancy integrals so the term is
  differentiable.

* **Extremum alignment** (``extremum_weight``). The per-field-line
  :math:`B_{\min}` and :math:`B_{\max}` minus their field-line averages —
  poloidal closure of the extremal :math:`|B|` contours (Goodman's "align the
  maxima" step; also the flat-:math:`B_{\max}` condition of Dudt *et al.* 2024).

* **Single-well monotonicity** (``squash_weight``). The pointwise distance
  between :math:`|B|` and its monotone branch envelopes — Goodman's "squash"
  distance, which penalizes side wells (more than one magnetic well per period).

Every operation (sigmoid occupancies, running maxima, level-space quadrature) is
smooth or piecewise-smooth, so the residual is jit/grad/jvp-transparent and QI
optimization runs with the exact implicit adjoint, exactly like the QS residual.
These are necessary conditions at finite sampling, not a converse theorem: the
surrogate can be small while the fuller construction remains large.

VMEX keeps the three QI formulations separate:

* :class:`~vmex.core.omnigenity.QIResidual` is the existing compact level-set
  objective; its call signature and defaults are unchanged.
* :class:`~vmex.core.qi.ConstructedQIResidual` applies the fuller smooth
  squash-and-shuffle construction to the same traceable Boozer spectrum. Each well starts at its
  :math:`|B|` minimum; the grid argmin jumps a whole cell as the field moves,
  so the residual is discontinuous there. ``argmin_softness`` replaces it by
  the softmin centroid :math:`\sum_i i\,w_i`,
  :math:`w_i\propto e^{-(b_i-\min b)/\tau}` (circular on a periodic line),
  and reads the branches by linear interpolation, which makes the residual
  continuous; an exactly QI field is still a zero.
* :class:`~vmex.core.qi.JInvariantQIResidual` minimizes the normalized
  dispersion of :math:`\mathcal J_\parallel` pooled over all complete wells,
  field lines, and traced periods at each surface and pitch. Its pitch array
  is supplied in inverse tesla and is shared across surfaces; it is never
  renormalized independently on each field line. Wells are not matched into
  families: two different wells can have distinct actions and produce a
  positive residual even when both actions are independent of :math:`\alpha`.
  Thus this objective alone does not measure within-family omnigenity or
  establish radial drift suppression in a multiwell field.

The action residual excludes cut edge wells but retains complete wells inside
the bounded trace. A pitch block with no complete well on even one sampled field
line, a marginal or merged level, or more wells than ``max_wells`` returns NaN
with a false ``valid_pitch`` flag. This makes a topology error visible instead
of turning it into a favorable zero. The low-level Boozer-spectrum function
accepts cosine and sine harmonics, and so does the equilibrium objective:
:func:`~vmex.core.omnigenity.boozer_spectrum_state` runs symmetric and
``lasym`` states through the same booz_xform kernel and returns ``bmns_b``
alongside ``bmnc_b`` (all-zero in the symmetric case),
which every QI residual above passes through. On an up-down-asymmetric deck
the traceable cosine and sine spectra are gated against the host
booz_xform_jax reconstruction at ``2e-2`` and ``3e-2`` relative, and the QI
state derivative is checked against a finite difference of the same objective.

Maximum-J
~~~~~~~~~~~~~

A maximum-J field satisfies

.. math::

   \left.\frac{\partial\mathcal J_\parallel}{\partial s}
   \right|_{\alpha,\lambda} < 0, \qquad
   s = \frac{\psi}{\psi_{\rm edge}},

where :math:`s` increases from the magnetic axis to the boundary.
:class:`~vmex.core.maxj.MaximumJResidual` evaluates the action at the same
physical pitch and field-line label on adjacent surfaces, pairs complete wells
only when they are reciprocal nearest neighbours, and forms the physical
finite-difference derivative. The least-squares rows use the dimensionless
slope :math:`\psi_{\rm edge}(\partial J/\partial\psi)/J=(\partial J/\partial s)/J`; ``target=0``
penalizes only violations of the condition above, while a negative target
requests a finite margin.

The result also reports the maximum-J fraction of resolved trapped samples,
the fraction meeting ``target``, deep and shallow subsets split at normalized
trapping depth
:math:`(B_{\max}-1/\lambda)/(B_{\max}-B_{\min})=1/2`, and the fraction of
radial-pitch blocks excluded by topology guards. Fractions are uniform over
the supplied pitch samples by default; ``pitch_weights`` accepts user
quadrature weights. These values are not the bounce-time-weighted Maxwellian
phase-space fraction :math:`f_J` defined by Rodríguez and Plunk. That
diagnostic additionally requires radial and pitch integration weighted by the
normalized bounce time.

VMEX carries the VMEC sign convention through the intermediate derivative:
``psi_edge = signgs*phiedge/(2*pi)`` and the ``APHI`` remap sets the half-mesh
``psi_b`` values. Multiplication by ``psi_edge`` then converts ``dJ/dpsi`` to
the outward derivative ``dJ/ds``; reversing the signed-flux convention cannot
turn a central maximum into a minimum.
A nonmonotone flux map, missing or ambiguous well, topology transition, or
well displacement beyond ``match_tolerance`` returns NaN with
``valid_pitch_pair=False``.

Maximum-J remains a separate objective term. Users combine it with any QI,
aspect-ratio, iota, stability, or engineering residual through VMEX's ordinary
composite least-squares interface; no fixed QI-plus-maximum-J weighting is
built into the class. :class:`~vmex.core.maxj.ConstructedMaximumJResidual`
evaluates the same radial action condition after Goodman's smooth
squash-and-shuffle construction. It is the continuation target analogous to
the published :math:`g_J`: it establishes a favorable direction without
asking a local optimizer to cross actual-well topology changes. The final
:class:`~vmex.core.maxj.MaximumJResidual` remains the physical certificate.

``QI_maxJ_continuation.py`` starts from the bundled ``nfp = 3`` QI
equilibrium, retains a magnetic-well target, and ramps a negative slope
margin in the actual field. Its research ladder recomputes a common physical
pitch once after the weak stage and freezes it throughout the remaining
stages, and its final stage lengthens the field-line trace and increases the
number of field-line labels, preventing a short trace from aliasing a visibly
non-omnigenous result; the five-minute default runs the middle stage alone at
the seed's pitch. The script raises if the incoming wells cannot be resolved
at that pitch.
The final ``plot_wout(..., j_pitch=pitch)`` call passes that same pitch to the
polar :math:`J(\alpha,s)` panel, providing a visual diagnostic of :math:`J`
on the sampled pitch, surfaces, and field-line labels.

.. _confinement-qi-fidelity:

Metric fidelity: report the wout-based residual
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The traceable :class:`~vmex.core.omnigenity.QIResidual` is built for inexpensive
*gradient flow*, not for *labeling* a configuration. Its level-space envelopes
and finite sampling can report values well below an independent construction.
Use :class:`~vmex.core.qi.ConstructedQIResidual` for production optimization,
with reduced sampling only when its ranking has been checked against the
default resolution. When quoting "how QI is this equilibrium?", also evaluate
the wout/Boozer-based
:func:`~vmex.core.optimize.quasi_isodynamic_residual_from_wout` on the written
``wout``. Always report the surfaces and sampling because an outer-surface
total is not numerically interchangeable with a core-to-edge total.


Ideal MHD stability: Mercier and the magnetic well
--------------------------------------------------

Mercier criterion
~~~~~~~~~~~~~~~~~~

The Mercier criterion is the ideal-MHD stability condition for localized
interchange modes resonant on a rational surface — the toroidal generalization
of the Suydam criterion. Following Glasser–Greene–Johnson, the surface is
Mercier-**stable** where

.. math::

   D_{\mathrm{Merc}} = D_{\mathrm{shear}} + D_{\mathrm{well}}
                     + D_{\mathrm{curr}} + D_{\mathrm{geod}} > 0 ,

a decomposition into four flux-surface integrals with distinct physical origin.
:func:`~vmex.core.nyquist.mercier_and_jxb` ports VMEC2000 ``mercier.f`` term
for term (with :math:`{}' \equiv d/d\psi`, :math:`p` the pressure,
:math:`\iota` the transform, :math:`V'` the differential volume, and
:math:`I_\varphi` the enclosed toroidal current):

.. list-table::
   :header-rows: 1
   :widths: 24 20 56

   * - term
     - sign
     - origin
   * - :math:`D_{\mathrm{shear}} = \tfrac14(\iota')^2`
     - :math:`\ge 0`
     - magnetic shear (always stabilizing)
   * - :math:`D_{\mathrm{well}}`
     - either
     - pressure gradient :math:`\times` magnetic well :math:`V''`
   * - :math:`D_{\mathrm{curr}}`
     - either
     - parallel-current / kink drive :math:`\propto \iota'`
   * - :math:`D_{\mathrm{geod}}`
     - :math:`\le 0`
     - geodesic curvature (always destabilizing)

The well term carries the sign of :math:`p'\,(V'' - p'\langle\cdots\rangle)`:
with the pressure decreasing outward (:math:`p'<0`), a magnetic well
(:math:`V''<0`, volume that decreases toward the edge) is stabilizing. The
geodesic term is a manifestly non-positive Schwarz-inequality remainder. Because
the individual pieces involve radial derivatives of surface averages, the two
surfaces nearest the axis and the edge carry the usual numerical noise; a
practical objective penalizes ``min(DMerc[2:-1], 0)``. ``vmex`` exposes the
reporting profile as :func:`~vmex.core.optimize.d_merc`, evaluated through the
parity-proven wout engine.  The live-state counterpart
:func:`~vmex.core.stability.d_merc_state` is a pure-JAX port of the same
``jxbforce.f``/``mercier.f`` path for ``jit``/AD use and agrees with the wout
profile to floating-point round-off.  For optimization,
:func:`~vmex.core.stability.mercier_stability_residual` excludes ``[0:2]``
and the edge and returns
``smoothing * softplus((margin - DMerc) / smoothing)``; targeting it to zero
penalizes unstable (negative) ``DMerc`` with a smooth gradient.  At finite
``smoothing`` the residual is positive, rather than exactly zero, on stable
surfaces but decays exponentially with the stability margin.  Both profile
lanes retain VMEC's near-axis and edge limitations.  Both support
``lasym = True``: the asymmetric lane carries the sine-parity content through
the ``jxbforce.f`` parity-split filter and the full-theta surface integrals,
and is validated per-term against live VMEC2000 (below).

For a vacuum equilibrium, :math:`p'=0` makes :math:`D_{\rm well}` exactly
zero; VMEX does not add a pressure floor. The reported Mercier index can still
contain shear, current, and geodesic terms. In the additional current-free
limit it reduces to :math:`D_{\rm shear}=S^2/4`, so a positive vacuum value is
a mathematically defined shear result, not a finite-beta interchange margin.

Parallel current and resistive interchange
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~vmex.core.stability.jdotb_state` exposes the same full-mesh
``jdotb = <J.B>`` profile that VMEC2000 computes in ``jxbforce.f``, but as a
pure-JAX function of the converged state.  Its WOUT normalization and
extrapolated axis/edge entries are unchanged, so it can be used directly as
an implicit-differentiation current objective. This is a direct port of the
``jxbforce.f`` current reconstruction, distinct from the cheaper
``bootstrap.vmec_j_dot_B`` force-balance identity used by the Redl mismatch.
For optimization, :func:`~vmex.core.stability.jdotb_residual` selects
``jdotb[2:-1]`` to exclude the usual axis/edge entries.
This lane supports ``lasym = True``.  A converged finite-pressure,
up-down-asymmetric tokamak agrees with live VMEC2000 to ``1.52e-3`` relative
over ``[2:-1]``; its regression gate is ``2e-3``.

Assuming the ideal prerequisite :math:`D_{\rm Merc}>0`, the
Glasser--Greene--Johnson necessary condition for local resistive interchange
stability is :math:`D_R \leq 0`.  In the VMEC Mercier normalization, with
:math:`S=d\iota/d\Phi` and
:math:`D_{\rm shear}=S^2/4`, the Landreman--Jorge relation (J. Plasma
Phys. 86, 905860510 (2020), eq. (5.6); :math:`D_R` and :math:`H` are
their eqs. (5.1) and (5.4)) is

.. math::

   D_R = -D_{\rm Merc}
       + \frac{(H-S^2/2)^2}{S^2},\qquad
   H=S\left(t_{JB}-\frac{\langle\mu_0J\cdot B\rangle}
                              {\langle B^2\rangle}t_{BB}\right).

:func:`~vmex.core.stability.glasser_d_r_state` reuses the traceable Mercier
surface integrals and the VMEC-consistent ``jdotb/bdotb`` averages to evaluate
this expression.  Exact zero-shear entries are set to zero because the
criterion is undefined there; a positive ``shear_epsilon`` provides a smooth,
explicit regularization for optimization.  The
:func:`~vmex.core.stability.glasser_stability_residual` helper selects
``D_R[2:-1]`` and smoothly penalizes values above ``-margin``; it defaults to
``shear_epsilon=1e-8`` so zero-shear optimization seeds remain finite.
It must be combined with ``mercier_stability_residual`` so an
ideal-Mercier-unstable surface is never accepted based on :math:`D_R` alone.
After optimization, use ``mercier_shear_state`` to verify
``abs(S) >> shear_epsilon`` on every target surface; regularization makes the
numerics finite but cannot establish GGJ stability at zero shear.
For a pressure-free, current-free equilibrium, :math:`H=0` and
:math:`D_{\rm Merc}=S^2/4`, hence the expression above gives :math:`D_R=0`.
Small reconstructed departures from zero reflect finite-resolution
cancellation; a zero-beta force-free equilibrium with parallel current may
retain nonzero current and geodesic contributions. Vacuum :math:`D_R` is
therefore not a pressure-driven stability margin.
VMEC2000 does not write ``D_R`` itself.  A live `DCON/GPEC
<https://github.com/PrincetonUniversity/GPEC>`_ evaluation independently
reproduces the symmetric VMEC normalization at ``ns=51`` (``D_I`` maximum
absolute difference ``9.10e-4`` and ``D_R`` ``8.63e-5`` over normalized
poloidal flux ``[0.1, 1)``); ``tests/test_stability.py`` holds the two to
that reference within ``1e-3`` and ``1e-4``.

LASYM scope.  ``mercier.f`` integrates real-space fields over the full
theta interval with the uniform lasym weights and the ``jxbforce.f`` inputs
carry both parity channels, so VMEC2000 anchors the asymmetric lane.
The analysis weight is ``1/(nzeta*(ntheta2-1))`` over the half-interval
reduced grid in both symmetry modes, which is the physical-scale norm VMEX
uses.  On the bundled up-down-asymmetric deck, raised to finite pressure by
the stability fixture (the shipped deck has ``AM = 0``, so ``DWell`` vanishes
identically on it), :func:`~vmex.core.stability.d_merc_state` reproduces the
golden VMEC2000 ``DMerc`` profile to ``1.2e-4`` scale-relative
(``DGeod`` ``2.1e-3``, ``<J.B>`` ``1.0e-4``) with full interior
sign agreement — inside the ``5e-2`` tolerance class of the symmetric live
gate — and an independent NumPy reconstruction of the ``mercier.f``
integrals from the wout Fourier tables (both parities) reproduces the same
profiles at the few-percent reconstruction class for VMEX and VMEC2000
alike.  For LASYM :math:`D_R` no external oracle exists (the DCON
comparison above is symmetric-only, and its asymmetric ``H`` remained
sensitive near marginality): the asymmetric
:func:`~vmex.core.stability.glasser_d_r_state` is validated by
internal consistency — the exact GGJ identity applied to the
VMEC2000-anchored LASYM ``DMerc`` and the independently validated
``jdotb``/``bdotb`` averages — not by a DCON anchor, and near-marginal
asymmetric :math:`D_R` values inherit the corresponding sensitivity.

Magnetic well
~~~~~~~~~~~~~~

The dominant stabilizing ingredient of :math:`D_{\mathrm{well}}` — the sign of
:math:`V''(s)` — is also useful on its own as a cheap, fully traceable proxy.
The ``--plot`` summary shows this dimensional derivative directly, labeled
:math:`V''(s)` (magnetic well), and aligns its zero with the stability-index
axis. Negative values denote a well and positive values a hill.

The separate ``*_stability.png`` figure retains the full Mercier
decomposition and exposes the pressure-gradient dependence more directly.
For a trial pressure derivative :math:`p'`, with geometry and current frozen,
it evaluates

.. math::

   D_{\mathrm{well}}(p') = p'\left(V''-p'T_{pp}\right)T_{bb},\qquad
   D_{\mathrm{Merc}}(p') = D_{\mathrm{Merc},0}-D_{\mathrm{well},0}
     +D_{\mathrm{well}}(p'),

and :math:`D_R(p')=D_{R,0}+D_{\mathrm{well},0}-D_{\mathrm{well}}(p')`.
The plot reports :math:`\min_s D_{\mathrm{Merc}}` and
:math:`-\max_s D_R`, so positive values are favorable for both criteria. It
rescales the WOUT pressure shape to trial volume-average beta; because a
vacuum WOUT contains no such shape, that case is explicitly labeled and uses
:math:`p(s)\propto1-s`. This is a coefficient-level sensitivity diagnostic,
not a finite-pressure stability certificate: geometry, Shafranov shift,
transform, and current respond to pressure, so selected beta points must be
re-solved before drawing a stability conclusion.

:func:`~vmex.core.optimize.magnetic_well` returns the finite-difference
vacuum-well measure

.. math::

   W = \frac{V'(0) - V'(1)}{V'(0)},

with :math:`V'=dV/ds` extrapolated from the half-mesh differential volume
:math:`vp` (VMEC ``bcovar.f``). Positive :math:`W` means :math:`V'` decreases
outward — a magnetic well, favorable for interchange stability — matching
simsopt's ``vacuum_well``. Being a pure
``(equilibrium_state, solver_context)`` function it carries
exact implicit gradients and is a cheaper Mercier-adjacent target. Near-axis
analytic context for both measures is in Landreman–Jorge (2020) and
Kim–Jorge–Dorland (2021); see :doc:`/project/references`.

Ideal ballooning
~~~~~~~~~~~~~~~~~

Interchange stability bounds only the :math:`n\to\infty`, radially-localized
limit. The complementary *ballooning* limit — high toroidal mode number,
extended along the field line — is provided by :mod:`vmex.core.stability`
as a fully differentiable eigenvalue objective (a JAX port of the COBRA
solve in the Gaur *et al.* formulation). It solves the self-adjoint
field-line ODE eigenproblem

.. math::

   \frac{d}{d\eta}\!\Bigl(g\,\frac{dX}{d\eta}\Bigr) + c\,X = \lambda\,f\,X,
   \qquad X(\pm\eta_b)=0,

along the straight-field-line angle :math:`\eta`, where :math:`g` is the
line-bending term, :math:`c` the pressure/curvature drive, and :math:`f>0` the
inertia; :math:`\lambda = (\gamma a_N/v_A)^2 > 0` flags instability.
**Discretization.** The three-point stencil on a uniform grid of step
:math:`h` turns the ODE into a symmetric tridiagonal eigenproblem whose
largest eigenvalue has error :math:`a h^2 + O(h^4)`. VMEX solves it on the
grid and on its every-other-point subgrid and returns
:math:`(4\lambda_h - \lambda_{2h})/3`, the Richardson step COBRA also takes
(Sanchez *et al.* 2000). The coefficients oscillate along the line with the
toroidal harmonics, about :math:`n_{fp} N/\iota` times per poloidal turn,
so the extrapolation needs the grid to resolve them: on the bundled QA seed
(:math:`n_{fp}=2`, :math:`\iota\approx0.42`) 32 points per turn read
:math:`\lambda` 13 % low and 64 points 3 % low, while the extrapolation from
128 points per turn is within 0.4 % of the converged value. The default is
641 points over :math:`\eta\in[-5\pi,5\pi]`. An unstable mode is localized
and converged in domain within three turns; a stable line tends to
:math:`\lambda\to0^-` as the domain grows, so thresholds should not sit just
below zero.

**Radial derivatives.** Geometry spectra live on the full radial mesh,
:math:`\iota` and :math:`p` on the half mesh. Each is read at the requested
:math:`s` from the degree-4 polynomial through its five nearest nodes, so
any :math:`s\in(0,1)` can be evaluated and the same surface is compared
across radial grids. The second radial derivative enters :math:`\nabla|B|`,
and :math:`\lambda` is sensitive to it: with three-point differences, as in
COBRA, the eigenvalue on the QA seed at ``ns = 71`` is 2–6 % low, and the
three-point value extrapolated from stencils :math:`h_s` and
:math:`2h_s` agrees with the five-point value to 0.3 %.

**Reductions.** At fixed :math:`(\alpha,\zeta_0)`, :math:`\lambda(s)` is
rough: as :math:`\iota` changes, a long field line samples the field
periods at different phases, and on the QA seed :math:`\lambda` at
:math:`\alpha=0` changes sign twice between :math:`s=0.80` and
:math:`0.83`. The maximum over a dense set of lines and ballooning
parameters is the smooth, convergent quantity: it agrees within 5 % between
``ns = 41`` and ``101``. :func:`~vmex.core.stability.ballooning_residual`
returns one row :math:`\max(\lambda-\lambda_0,0)` per line (Gaur *et al.*
2025; DESC's ``BallooningStability``), and
:func:`~vmex.core.stability.ballooning_growth_rate` the hard maximum. The
usage recipe is in :doc:`/reference/objectives`.



.. _neoclassical-heat-bounds:

Neoclassical heat bounds
-------------------------

:class:`vmex.core.neoclassical.NeoclassicalHeatBound` minimizes an upper
bound for a bounce-averaged, single-reference-well Lorentz model, retaining
pitch diffusion and magnetic/electric tangential precession. The upper–lower gap
bounds the remaining trial-space error. See the
:ref:`optimization recipe <heat-bound-optimization>` and
``examples/optimization/QA_optimization_heat_bounds.py``.

The reference response vanishes in exact quasisymmetry, while ordinary
neoclassical transport and bootstrap current remain. This objective does
not certify total heat loss or an additive positive part of it. Passing
particles, momentum-conserving collisions, finite orbit width, surface
potential variation and multiple-well coupling require a fuller model.
Orbit/Fourier errors and rounding are outside the variational interval.

The two functionals
~~~~~~~~~~~~~~~~~~~

For periodic label :math:`\alpha` and pitch :math:`\lambda\in[a,b]`, write

.. math::

   (A+K)g=S,\qquad
   A=-\frac{\nu_D}{w}\partial_\lambda(p\partial_\lambda),\qquad
   K=\omega(\lambda)\partial_\alpha .

The measure is :math:`w\,d\lambda\,d\alpha/(2\pi)`. Positive :math:`w,p`,
:math:`\nu_D>0`, :math:`g(a)=0` and natural :math:`p g'(b)=0` make
:math:`A` positive and self-adjoint. :math:`K` is skew-adjoint because
:math:`w\omega` is independent of :math:`\alpha`. The response is
:math:`D=\operatorname{Re}\langle S,g\rangle=\|g\|_A^2`.
For admissible trials,

.. math::

   L[u]&=2\operatorname{Re}\langle S,u\rangle
          -\|u\|_A^2-\|Ku\|_{A^{-1}}^2,\\
   U[h]&=\|h\|_A^2+\|S-Kh\|_{A^{-1}}^2,\qquad L[u]\le D\le U[h].

With :math:`z=(A-K)^{-1}S`, the exact trials are
:math:`u_*=(g+z)/2` and :math:`h_*=(g-z)/2`. Completing squares gives

.. math::

   D-L[u]&=\|u-u_*\|_A^2+\|K(u-u_*)\|_{A^{-1}}^2,\\
   U[h]-D&=\|h-h_*\|_A^2+\|K(h-h_*)\|_{A^{-1}}^2.

These error identities establish the enclosure. The residual identity

.. math::

   U-L=\|S-Au-Kh\|_{A^{-1}}^2+\|Ah+Ku\|_{A^{-1}}^2

measures its width. No division by precession occurs at resonance.
This follows the complementary variational principle of
`van Rij and Hirshman (1989) <https://doi.org/10.1063/1.859116>`_.

The collision inverse is evaluated continuously:

.. math::

   \|r\|_{A^{-1}}^2=\frac1{\nu_D}\int_a^b\frac1{p(\lambda)}
       \left|\int_\lambda^b w(t)r(t)\,dt\right|^2d\lambda .

Piecewise-constant coefficients and continuous P1 trials make the cell
integrands quartic; three-point Gauss integration is exact in real
arithmetic. The bound encloses that differential equation, not just its
discrete matrix. The mean Fourier response is integrated directly without
a redundant solve. Enriching the trial space tightens both endpoints.

For an analytical limit on each retained Fourier block, set
:math:`T=A^{-1/2}KA^{-1/2}`, :math:`s=A^{-1/2}S` and
:math:`C=\|s\|^2`. Then

.. math::

   D=\langle s,(I-T^2)^{-1}s\rangle,\qquad
   C/(1+\Gamma^2)\le D\le C,\quad \Gamma\ge\|T\|.

For small :math:`\Gamma`, :math:`D=C-\|Ts\|^2+R` with
:math:`0\le R\le\Gamma^4 C`. This explains the collision-dominated limit
and precession suppression while retaining pitch coupling.

Geometry and heat normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Project the Boozer field onto a one-dimensional :math:`B_0` with
:math:`dl_0=C_B/B_0\,d\zeta`. The QA/QH chart retains :math:`n=mN`
harmonics; the toroidal chart retains :math:`m=0`. For one well,

.. math::

   J_0&=2\int\sqrt{1-\lambda B_0}\,dl_0,\qquad
   \tau_g=2\int\frac{dl_0}{\sqrt{1-\lambda B_0}},\\
   \delta J&=-2\int\frac{\delta B}{B_0}
       \frac{1-\lambda B_0/2}{\sqrt{1-\lambda B_0}}\,dl_0,\\
   J_{\psi0}&=\frac{C_{B,\psi}}{C_B}J_0
       -2\int\frac{B_{0,\psi}}{B_0}
       \frac{1-\lambda B_0/2}{\sqrt{1-\lambda B_0}}\,dl_0,\\
   w&=\tau_g/2,\qquad
   p=2\lambda\int\frac{\sqrt{1-\lambda B_0}}{B_0}\,dl_0 .

Here :math:`\psi` is signed toroidal flux divided by :math:`2\pi`.
For energy :math:`\mathcal E`, charge :math:`q` and periodic-label factor
:math:`c=d\alpha/d\alpha_{\rm canonical}`,

.. math::

   S=\frac{dr}{d\psi}\frac{2\mathcal E}{q}
          \frac{c\,\partial_\alpha\delta J}{\tau_g},\qquad
   \omega=-\frac{2\mathcal E}{q}\frac{c\,J_{\psi0}}{\tau_g}
           -c\,\frac{dr}{d\psi}E_r .

The helical chart has :math:`c=-N_{\rm fp}/(\iota-N)`; the toroidal chart
has :math:`c=1`. The first variation is valid at regular interior pitch;
it supplies no uniform remainder near moving trapping boundaries.

The conjugate forces are :math:`X_1=-d\ln(nT)/dr+qE_r/T` and
:math:`X_T=-d\ln T/dr`. With :math:`x=mv^2/(2T)` and
:math:`I_B=\int dl_0/B_0`,

.. math::

   \chi_T=\frac1{\sqrt{\pi}I_B}\int_0^\infty
       \sqrt{x}e^{-x}(x-5/2)^2D_x\,dx .

This is the pressure-force diagonal heat coefficient in m²/s. Signed
heat flux, zero-particle-flux heat transport and fixed-speed diffusion
are different quantities. The default positive speed rule returns matrix
and Schur-complement intervals. ``continuous_energy=True`` integrates
the entire Maxwell axis for exactly :math:`\nu_D(x)=\nu_{\rm th}x^{-3/2}`,
including infinity; it returns heat and binwise intervals. Its static
energy bins control trial approximation, not energy quadrature.

JIT/JVP/VJP derivatives pass through equilibrium, Boozer geometry, bounce
roots and trial solves on a fixed regular well branch. Normalized
``pitch_fractions`` follow the whole reference trapping interval.
Invalid topology, unresolved transverse modes and nonfinite data return
an infinite objective. Fix the scale and trial topology within each stage.

Near-axis limits
~~~~~~~~~~~~~~~~~

For first-order QS let :math:`\epsilon=\eta r` and

.. math::

   B/B_*=1-\epsilon\cos\chi+
       \epsilon^2[b_0+b_2\cos2\chi+\widetilde b(\alpha,\chi)]
       +O(\epsilon^3).

With :math:`\kappa=k^2`, complete elliptic integrals :math:`K,E`,
:math:`F=E-(1-\kappa)K` and :math:`\ell=|G_0|/(B_*|\iota_0-N|)`,

.. math::

   J_0=8\sqrt2\,\ell\sqrt\epsilon\,F[1+O(\epsilon)],\qquad
   \tau_g=\frac{4\sqrt2\,\ell}{\sqrt\epsilon}K[1+O(\epsilon)],\qquad
   A=\frac{\nu_D}{\epsilon}
       [-K^{-1}\partial_\kappa(F\partial_\kappa)]+O(\nu_D).

The trapped measure is :math:`O(\sqrt\epsilon)`. Second-order symmetry
breaking gives :math:`\delta J=O(\epsilon^{3/2})` and
:math:`S=O(\epsilon)`; magnetic precession is :math:`O(\epsilon^{-1})`.
At fixed positive collision frequency and drift prefactors,

.. math::

   g=O(\epsilon^2),\qquad D_{\rm ref}=O(\epsilon^{7/2}).

This is a model asymptotic. Physical bounce averaging also requires
:math:`\nu_D\ell/v\ll\epsilon^{3/2}` and small orbit width; fixed physical
parameters violate these conditions sufficiently close to the axis.
Joint limits can change the response power.

The first and second precession terms are derived by
`Rodríguez and Mackenbach (2023) <https://doi.org/10.1017/S0022377823001125>`_.
Regular :math:`\Phi(\psi)` gives :math:`E_r=O(r)` and electric precession
:math:`O(1)`. Fixed nonzero :math:`E_r` is a different annular ordering.
A common fractional power in particle fluxes cancels from ambipolarity;
it does not itself imply fractional powers of :math:`E_r`.

Two numerical choices follow:

* ``radial_interpolation="radius"`` differentiates a quadratic interpolant
  in :math:`r=a\sqrt{s}`, then applies :math:`dr/d\psi`. It reproduces
  the radial derivative of :math:`B_0+rB_1+r^2B_2` exactly. The default
  ``"flux"`` preserves earlier results.
* Well-relative diagnostics compare :math:`\delta B` with
  :math:`B_{0,\max}-B_{0,\min}`. Small :math:`\delta B/B_0` alone is
  insufficient when the well shrinks.

Generic QI has finite on-axis variation :math:`B_0(\varphi)` and different
power counting. Its second-order action contains both
:math:`\delta J[B_2]` and :math:`\tfrac12\delta^2J[B_1,B_1]`.
The linear-action adapter omits the latter; an exactly translated well
demonstrates the missing cancellation. It is not a controlled
second-order QI model.

The effective trapped fraction has :math:`f_t\sim1.4624\sqrt{\eta r}`
in the QS outer limit. A half-power bootstrap expansion requires a
separate momentum-conserving closure; this even-in-parallel-velocity
response cannot calculate bootstrap current. The source-dependent
potato-orbit region requires the nonlocal treatment discussed by
`Helander (2000) <https://doi.org/10.1063/1.874137>`_.

Validation and design use
~~~~~~~~~~~~~~~~~~~~~~~~~

The reference campaign encloses 832/832 independent exact-cell responses:
169 manufactured and 221 each for QA, QH and a toroidal QI projection.
At all 17 sampled fields from −1 to +1 kV/m, the 12-node speed-rule width
is at most 10% on sampled collision rows 10–100 s⁻¹ for QA and
0.1–100 s⁻¹ for QH/QI. Three continuous-energy widths are
5.85%, 6.66% and 5.95%. These are sampled-grid/model statements.

An eight-iteration, three-coefficient QA optimization gives continuous-energy
intervals for the maximum over six scenarios:

.. list-table::
   :header-rows: 1

   * - Shape
     - Lower (m²/s)
     - Upper (m²/s)
   * - Seed
     - 0.0100217194
     - 0.0104519299
   * - Candidate
     - 0.0008243780
     - 0.0008826379

The intervals establish a 91.19% decrease of the maximum reference response;
pointwise guaranteed decreases are 82.72–92.36%. AD differs from re-solved
finite differences by 0.08925%. The run reaches its iteration budget.
A higher Boozer bandwidth changes one candidate upper endpoint by 11%,
so a narrow variational interval does not establish geometry convergence.

Independent DKX fixed-speed Lorentz diffusion at 1 keV and zero electric
field changes by −5.8815% at 100 s⁻¹ and +0.1254% at 1000 s⁻¹; the
adverse sign survives angular refinement. A separate QA result is
4.73 million times the reference upper endpoint. These comparisons
establish no uniform reduction of physical heat transport.

Near-axis checks use second-order QA/QH equilibrium coefficients from QSC.
All 48 model intervals enclose exact-cell references. Last-three-radius
exponents 3.5033 (QA) and 3.5039 (QH) approach 7/2. Independent orbit
integrals show first-order relative errors proportional to :math:`\epsilon`
and second-order errors proportional to :math:`\epsilon^2` for action
and precession. These validate the stated reference equation, not a
full-transport closure.

Acceptance needs separate variational, orbit/radial/Fourier and fuller
kinetic checks. A physical application of the conserving theorem below must
establish its orbit measure and skew operator, include the relevant
trapped–passing coupling, and resolve geometry and finite-orbit-width effects.

.. _conserving-collision-bounds:

Conserving Coulomb collision forms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`vmex.core.neoclassical.conserving_transport_bounds` evaluates the
complementary construction for supplied collision and orbit forms, retaining
the local conserved moments. It accepts rectangular orbit images so that a
continuous residual can be represented in a larger space than the trials.
This API supplies the constrained algebra; the Boozer objective above
continues to use its stated Lorentz model.

For Maxwellians :math:`M_a` at a common temperature and common flow, the
full linearized Landau collision form is

.. math::

   a_C(f,g)=\frac12\sum_{a,b}\gamma_{ab}\int
       \frac{M_aM_b'}{|\boldsymbol u|}
       (\Pi_{\boldsymbol u}\Delta_{ab}f)^*\cdot
       (\Pi_{\boldsymbol u}\Delta_{ab}g)\,d\mu_x\,d^3v\,d^3v',

where :math:`\boldsymbol u=\boldsymbol v-\boldsymbol v'`,
:math:`\Pi_{\boldsymbol u}=I-\boldsymbol u\boldsymbol u^T/u^2`,
:math:`\Delta_{ab}g=\nabla_vg_a/m_a-\nabla_{v'}g_b/m_b`, and the symmetric
coefficients :math:`\gamma_{ab}` contain the Coulomb constants.
Both test-particle and field-particle terms are included. Write
:math:`a_C(f,g)=\langle\mathcal B f,\mathcal B g\rangle`, so
:math:`A_C=\mathcal B^*\mathcal B`. At each position the kernel contains
species densities, common momentum and common energy. Streaming couples
these local fields to the dissipative velocity sector.

Let :math:`(A_C+K)g=S` with :math:`K^*=-K` in the stated kinetic measure.
Choose collision pair fluxes satisfying
:math:`\mathcal B^*F_u=Ku` and :math:`\mathcal B^*F_h=S-Kh`. Then

.. math::

   L&=2\operatorname{Re}\langle S,u\rangle
        -\|\mathcal B u\|^2-\|F_u\|^2,\\
   U&=\|\mathcal B h\|^2+\|F_h\|^2,\\
   D-L&=\|\mathcal B g-\mathcal B u+F_u\|^2,\qquad
   U-D=\|\mathcal B h+F_h-\mathcal B g\|^2.

These identities establish the continuum bounds without a Coulomb spectral
gap. Flux feasibility requires :math:`PKu=0` and :math:`P(S-Kh)=0`, where
:math:`P` projects onto the local collision kernel. Removing all of that
kernel changes the response. For example,

.. math::

   A=\begin{pmatrix}0&0\\0&\nu\end{pmatrix},\qquad
   K=\begin{pmatrix}0&k\\-k&0\end{pmatrix}

gives :math:`D=\nu/k^2` for :math:`S=(1,0)^T`, but :math:`D=0` for
:math:`S=(0,1)^T`. An unconstrained pseudoinverse gives a false zero upper
in the first case and a false positive lower in the second.
The first balance grows with collision frequency because the source acts
on a conserved moment whose flux is set by streaming.

An explicit continuous inverse-norm majorant follows from auxiliary Landau
forms :math:`a_p` with kernel :math:`|u|^p\Pi_u`, for even
:math:`p=2,4,\ldots`. Their operators map complete degree-:math:`d`
velocity polynomials to degree at most :math:`d+p-2`. A polynomial flux

.. math::

   F=\sum_p |u|^{p+1}\mathcal B q_p,\qquad
   \sum_p A_pq_p=r

therefore satisfies :math:`\mathcal B^*F=r` throughout velocity space when
all resulting polynomial coefficients match. Its norm is
:math:`\sum_{p,q}a_{p+q+1}(q_p,q_q)`. Minimizing this finite quadratic form
gives a Gram matrix :math:`G` such that :math:`\|r\|_{A_C^\dagger}^2\le r^*Gr`.
The Maxwell kernel :math:`p=2` supplies a feasible initial lift; enriching
the flux space can tighten the bound. A truncated Coulomb matrix inverse
does not supply this continuum majorant.

The committed tests include exact low-degree brackets. For unit mass,
density and pair coefficient at :math:`T=1/2`, the normalized parallel-heat
polynomial has :math:`a_C=16\sqrt{2/\pi}/15`, :math:`a_2=8` and
:math:`a_5=1024\sqrt{2/\pi}/5`. Its one-trial lower and Maxwell-flux upper
are :math:`15/[16\sqrt{2/\pi}]` and :math:`16\sqrt{2/\pi}/5`, respectively.
The tests also check invariant coupling, nested trial and inverse-majorant
spaces, complex Fourier data and derivatives of moving constraint charts.

Signed observables require an adjoint estimate. For
:math:`(A_C-K)z=J`, feasible primal and adjoint residual fluxes with norms
:math:`\eta_g,\eta_z` give

.. math::

   \left|\operatorname{Re}\langle J,g\rangle-c\right|
       \le\eta_g\eta_z,\qquad
   c=\operatorname{Re}\{\langle J,g_0\rangle+
               \langle z_0,S-(A_C+K)g_0\rangle\}.

:func:`vmex.core.neoclassical.conserving_goal_interval` evaluates this
interval. Polarizing dissipative bounds alone omits the antisymmetric
cross response. The physical current moment and its kinetic model remain
caller inputs.

Full-Fokker–Planck variational transport predates this implementation:
see `Taguchi (1982) <https://doi.org/10.1143/JPSJ.51.1630>`_ and
`van Rij and Hirshman (1989) <https://doi.org/10.1063/1.859116>`_.
The common-temperature assumption is essential for this exact Landau
entropy form; unequal-temperature symmetry-restoring operators are
different models, as discussed by
`Sugama et al. (2019) <https://doi.org/10.1063/1.5115440>`_.
``valid`` checks the supplied finite algebra and conservation chart. The
caller must establish the exact orbit images, continuous inverse majorant,
measure and boundary conditions; rounding remains outside the theorem.

.. _full-coulomb-stellarator-proxy:

Full-Coulomb stellarator upper objective
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`vmex.core.neoclassical.ConservingHeatBound` evaluates a continuous
single-species Landau upper bound directly from the equilibrium's Boozer
spectrum. It includes the test-particle and field-particle collision terms,
parallel streaming, mirror trapping and the incompressible DKES electric
drift. Tangential magnetic drift and interspecies collisions are outside
this objective. The scalar response is the heat coefficient for the
temperature force with the other thermodynamic force set to zero.

The underlying :func:`vmex.core.neoclassical.conserving_stellarator_proxy`
takes a finite Fourier representation of :math:`J=\widehat B^{-2}` in
:math:`x=\theta-h\zeta`, :math:`y=N_{\rm fp}\zeta`. Its complex coefficients
define the real part of the Fourier sum; :math:`h` is an integer multiple
of :math:`N_{\rm fp}`. Define

.. math::

   D=(\iota-h)\partial_x+N_{\rm fp}\partial_y,\qquad
   D_\perp=\frac{G+hI}{G+\iota I}\partial_x
                  -\frac{N_{\rm fp}I}{G+\iota I}\partial_y,
   \qquad P_0=D^{-1}D_\perp J,\quad\langle P_0\rangle=0.

Angle brackets here denote a uniform angular average. Each nonconstant
Fourier coefficient of :math:`P_0` is obtained by multiplying that of
:math:`J` by the ratio of the :math:`D_\perp` and :math:`D` wave numbers.
The implementation rejects a resonant chart; it does not replace the
denominator by a fitted floor. Set :math:`P=P_0+c` and

.. math::

   F=JD_\perp J-PDJ,\qquad
   E=JD_\perp P-\tfrac12PD_\perp J.

The odd heat-flow trial :math:`-\widehat B P v_\parallel(v^2-5/2)` satisfies
the local particle, momentum and energy balances. Its residual contains
only the heat polynomial :math:`q_1=v_\parallel(v^2-5/2)` and the stress
polynomial :math:`q_2=(v_\parallel^2-v^2/3)(v^2-5/2)`. Continuous inverse
norm bounds :math:`\mu_j\ge\langle q_j,A_C^\dagger q_j\rangle` give

.. math::

   U(c)=\frac1{\langle J\rangle}\left[
      \frac{4\nu}{3}\sqrt{\frac2\pi}\langle P^2\rangle
      +\frac{9\mu_2}{16\nu}\left\langle\frac{F^2}{J^3}\right\rangle
      +\frac{(E_*\langle J\rangle)^2\mu_1}{\nu}
                       \left\langle\frac{E^2}{J^4}\right\rangle\right].

This is a positive scalar quadratic in :math:`c`; its minimum requires
one division. The defaults are the explicit Maxwell-kernel flux lifts

.. math::

   A_2^{-1}q_1=q_1/8,\quad \mu_1=4\sqrt{2/\pi},\qquad
   A_2^{-1}q_2=(v_\parallel^2-v^2/3)(v^2/14-1/6),\quad
   \mu_2=\frac{9368}{2205}\sqrt{2/\pi}.

These auxiliary fluxes bound the inverse of the Coulomb operator; they do
not replace its collision law by Maxwell-molecule collisions. Sharper
continuous inverse bounds may be supplied. A finite Coulomb matrix inverse
alone is insufficient. At :math:`E_*=0`, an isobaric trial also gives
:math:`L=25\nu\langle P_0^2\rangle/(16\mu_1\langle J\rangle)`.
At finite electric field this compact objective returns the universal
lower endpoint zero.

Rational angular weights have an explicit integration correction. Write
:math:`J=j_0(1-r_J)` and establish
:math:`\|r_J\|_\infty\le\rho<1` from the Fourier amplitudes. For integer
:math:`p`, the tail after degree :math:`M` in the binomial expansion is

.. math::

   T_{p,M}(\rho)=\rho^{M+1}\sum_{k=0}^{p-1}
      {M+p\choose p-1-k}\frac{\rho^k}{(1-\rho)^{k+1}}.

The grid resolves every term through degree :math:`M+4` in the retained
Fourier band. For the nonnegative numerators :math:`F^2,E^2`, replacing
the sampled weight :math:`J^{-p}` by
:math:`J^{-p}+2j_0^{-p}T_{p,M}` bounds the remaining quadrature error.
The scalar minimization uses these corrected weights. A fixed
``rho_bound`` above the current relative Fourier amplitude makes this
tail correction differentiable when a Fourier coefficient crosses zero.
Floating-point rounding remains separate from the analytic bound.

``ConservingHeatBound`` fits :math:`J` on the chosen grid and reports the
sampled relative magnetic-field fitting error. The interval applies to
that finite representation. Refine ``geometry_order``, ``fit_grid`` and
the Boozer transform independently; the integration correction does not
bound the change to a different magnetic field.

The state objective returns heat diffusivity in m\ :sup:`2`/s. With
:math:`v_t=\sqrt{2eT_{\rm eV}/m}` and
:math:`\tau=|G+\iota I|/(B_{00}v_t)`, its operating inputs become

.. math::

   \nu=\tau\nu_C,\qquad
   E_*=-\tau E_r\frac{dr}{d\psi},\qquad
   \chi_r=\left(\frac{2T_{\rm eV}}{Z}\right)^2
              \tau\left(\frac{dr}{d\psi}\right)^2D.

Here ``collision_strengths`` means
:math:`\nu_C=n(Ze)^4\ln\Lambda/(8\pi\epsilon_0^2m^2v_t^3)` in
s\ :sup:`-1`, and ``radial_electric_fields`` means :math:`E_r=-d\Phi/dr`
in V/m. VMEX supplies the signed toroidal flux :math:`\psi=\Phi_t/(2\pi)`.
The flux radius :math:`r=a\sqrt{\psi/\psi_{\rm edge}}` fixes its derivative
even when the computational radial mesh uses a nonlinear flux profile.
The loss is the smooth upper envelope over the selected surfaces,
collision strengths and electric fields; ``scale`` is a fixed positive
diffusivity scale in m\ :sup:`2`/s.

Bootstrap current (Redl)
------------------------

The self-consistent parallel current a stellarator generates from its own
pressure gradient — the bootstrap current — sets the achievable :math:`\beta`
and, in a QI device, must be kept small. :mod:`vmex.core.bootstrap`
provides a differentiable evaluation and a self-consistency loop (reproducing
Landreman–Buller–Drevlak, arXiv:2205.02914).

Two independent estimates of :math:`\langle\mathbf J\cdot\mathbf B\rangle` are
compared. The **equilibrium** value follows from the exact MHD identity

.. math::

   \langle\mathbf J\cdot\mathbf B\rangle(s) =
   \frac{\langle B^2\rangle\,I'(s) + \mu_0\,I(s)\,p'(s)}{2\pi\,\psi_a},
   \qquad I(s) = \mathrm{signgs}\,\frac{2\pi}{\mu_0}\,\mathrm{buco}(s),

(:func:`~vmex.core.bootstrap.vmec_j_dot_B`). The **kinetic** value is the
Redl *et al.* (2021) analytic closure
(:func:`~vmex.core.bootstrap.j_dot_B_redl`), a fit in the effective trapped
fraction :math:`f_t` and the Sauter collisionalities, with the quasisymmetry
isomorphism :math:`\iota\to\iota-\mathrm{nfp}\,\mathrm{helicity\_n}` applied as
in simsopt. The trapped fraction itself uses the singularity-removing
substitution :math:`y=\sqrt{1-\lambda B_{\max}}` in

.. math::

   f_t = 1 - \tfrac34\langle B^2\rangle
         \int_0^{1/B_{\max}}\!
         \frac{\lambda\,d\lambda}{\langle\sqrt{1-\lambda B}\rangle},

evaluated with fixed-order Gauss–Legendre quadrature so the whole chain stays
differentiable. :func:`~vmex.core.bootstrap.trapped_fraction_from_state`
evaluates the radial profile directly from a converged state; every VMEX WOUT
stores the same full-mesh result as ``vmex_trapped_fraction``. A QI axis with
finite :math:`B_0(\varphi)` mirror ratio therefore has a finite trapped
fraction rather than an imposed zero. Their normalized mismatch is the residual
:class:`~vmex.core.bootstrap.RedlBootstrapMismatch` (the exact formula and
the finite-beta profile conventions are in :doc:`/reference/objectives`); driving it to
zero, optionally with ``current_dofs`` freed, yields a current profile
consistent with the plasma the equilibrium describes.

Effective ripple
----------------

The Nemov effective ripple is the geometric coefficient that sets the
low-collisionality :math:`1/\nu` transport scale,
:math:`D_{11}\propto\epsilon_{\mathrm{eff}}^{3/2}/\nu`.  NEO conventionally
reports :math:`\epsilon_{\mathrm{eff}}^{3/2}` (``epstot``), not
:math:`\epsilon_{\mathrm{eff}}` itself.  VMEX keeps the validated NEO
algorithm in the optional NEO_JAX package rather than duplicating a second
neoclassical solver:

.. code-block:: python

   from neo_jax import NeoConfig
   import vmex as vj

   config = NeoConfig(theta_n=64, phi_n=64, npart=40)
   s, epsilon_eff_3_2 = vj.epsilon_effective_from_wout(
       equilibrium.wout, surfaces=[0.2, 0.5, 0.8, 0.95], config=config)

The in-memory adapter performs BOOZ_XFORM without an intermediate file and
then uses NEO_JAX's batched JAX surface scan.  The smaller configuration used
by ``--plot`` is a trend diagnostic; converged work must refine NEO controls
and verify radial convergence.  The current WOUT entry point is diagnostic,
not an optimization objective.  The planned objective lane will connect the
traceable VMEX state transform directly to NEO_JAX, use its supported
forward-mode sensitivities, and certify them against reconverged finite
differences and STELLOPT NEO before exposing the result in objective tuples.
LASYM equilibria pass their sine Boozer harmonics; a NEO_JAX without them is
rejected rather than allowed to drop them silently.

Fast-ion confinement proxy
--------------------------

Ripple wells let trapped energetic particles drift radially: where the
contours of the second adiabatic invariant :math:`J` cross flux surfaces,
ions born on a superbanana leave the device within a few bounce times.
Nemov's :math:`\Gamma_c` (Nemov, Kasilov, Kernbichler, Leitold, Phys.
Plasmas 15, 052501 (2008), eq. 61) measures the angle between the
bounce-averaged drift and the flux surface,

.. math::

   \gamma_c = \frac{2}{\pi}\arctan\frac{v_r}{v_p}, \qquad
   \Gamma_c = \frac{\pi}{4\sqrt 2}\left\langle
     \int_{1/B_{\max}}^{1/B}\!\mathrm d\lambda\,
     \frac{B}{\sqrt{1-\lambda B}}\,\gamma_c^{2}\right\rangle,

with :math:`v_r` and :math:`v_p` the bounce-averaged radial and
poloidal-tangential magnetic drifts (Velasco et al., Nucl. Fusion 61,
116059 (2021), eqs. 14-16, in the Nemov/DESC form with the surface-tangency
factor evaluated at the field minimum of each well; KNOSOS uses the
:math:`\gamma_c^*` variant, which drops that factor).  The residual rows
are :math:`\Gamma_c` per surface, so the least-squares cost is the sum of
:math:`w\,\Gamma_c^{2}`; the square is the least-squares form, and
:math:`\Gamma_c` itself is a proxy rather than a loss law — Velasco et al.
2021 (section 5.4) find the prompt-loss fraction follows their
:math:`|\gamma_c^*|` variant (eq. 21) more linearly, and validate
:math:`\Gamma_\alpha` as the predictor.  Bader et al. and Paul et
al. document that such proxies correlate imperfectly with measured
energetic-particle losses, which bounds what any :math:`\Gamma_c` value
claims. :class:`~vmex.core.gammac.GammaC` evaluates the drift ratio in
Nemov's own form — DESC's rewrite into single-valued periodic maps
(``desc.compute._fast_ion`` on the bounce kernel of Unalmis et al., J.
Plasma Phys. 92(3), 2026, doi:10.1017/S0022377826101652, DESC's sibling
``GammaC`` objective) — because the literal line-integral form of the
poloidal drift carries a secular shear term that drives the proxy to zero
as the sampled line lengthens. The ingredients are exact spectral point
evaluations on PEST field lines (the :mod:`vmex.core.stability` machinery)
and the drift kernels of :func:`vmex.core.bounce.bounce_action`; each was
validated against finite differences of the drift-kinetic identities
:math:`\partial_\alpha J \propto \tau_b\,\overline{\mathbf v_M\cdot\nabla s}`
and :math:`\partial_s J \propto -\tau_b\,\overline{\mathbf v_M\cdot\nabla\alpha}`
and against the equilibrium's own ``wout`` tables.

The pitch integral samples the reflecting level :math:`1/\lambda`
uniformly with an open midpoint rule (the Unalmis et al. pitch-sampling
guidance), and wells sliding across the ends of the bounded trace are
faded by a half-transit taper applied to the normalizing
:math:`\int\mathrm dl/B` as well — an unbiased window on the ergodic
limit that removes end-of-trace discontinuities. Superbanana layers —
where the tangential drift :math:`v_p` reverses and
:math:`\gamma_c \to \pm 1` — are physical and are kept; resolving them
sets the ``nalpha``/``num_pitch`` budget, they dominate the residual
resolution noise of boundary gradients, and ``excluded_fraction`` reports
any well-slot overflow so an under-provisioned evaluation is visible.
Axisymmetry sends
:math:`\Gamma_c \to 0` exactly (:math:`\partial_\alpha J = 0`), which the
tests anchor together with the QA-versus-unoptimized ordering.

**The hard value is not a gradient objective.** The discretized
:class:`~vmex.core.gammac.GammaC` sampled gradient can change sign under refinement:
hard well detection and hard branch selection make the discretized
:math:`\Gamma_c` piecewise in the boundary coefficients with
grid-dependent breakpoints, and the measured li383 refinement ladder flips
sign three times (``tests/test_gammac.py``).  Two structures carry the
noise, both identified by direct gradient decomposition: a pitch node
landing at distance :math:`d` from a barrier top inherits the
:math:`1/d` level-curvature of the logarithmically divergent bounce time,
and a well cell whose radial *and* tangential bounce-averaged drifts
cancel simultaneously (a superbanana corner) contributes
:math:`\mathrm d(\arctan(v_r/v_p))/\mathrm dp \sim 1/\|(v_r, v_p)\|` with
arbitrary sign.  :class:`~vmex.core.gammac.GammaCSmooth` regularizes
exactly these two structures on the same wells and bounce quadrature — a
kernel floor :math:`1/\sqrt{\max(1-\lambda B, 0) + \delta}` with
:math:`\delta` a field-dependent fraction of the trapped range with live derivatives
(which also removes the birth/death jump of the well-connection graph of
Ochs 2025, since a newborn well's floored bounce time starts at zero
rather than the finite oscillator period), and a per-well corner floor
that compares the drift cancellation against the well's own
absolute-value bounce integrals. Both classes freeze normalized pitch-grid
shape derivatives, so even the smooth gradient is approximate. Check actual
forward descent and sampling refinement for each configuration. The surrogate
can underestimate shallow ripple wells; anneal ``temperature`` with a finer
budget and report the independent hard value.
