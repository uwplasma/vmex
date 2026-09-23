Mirror geometry
===============

``vmex.mirror`` contains two spline-native scalar-pressure equilibrium
models. Open mirrors use coordinates ``(s, theta, xi)`` with a nonperiodic
axial coordinate and fixed-flux end cuts; they are not reinterpreted as thin
periodic tori. Closed stellarator-mirror hybrids use a periodic longitudinal
B-spline around two exactly straight mirror legs and two curved stellarator
returns.

Lane status follows the generated contract in :doc:`/reference/capabilities`
(``benchmarks/capabilities.json``); this page explains the methods and quotes
the evidence behind each row.

.. list-table::
   :header-rows: 1
   :widths: 46 20 34

   * - Lane
     - Status
     - Evidence
   * - Open, fixed boundary, axisymmetric
     - supported
     - ``benchmarks/mirror_fixed_boundary.json``
   * - Open, fixed boundary, rotating ellipse
     - release candidate
     - ``benchmarks/mirror_fixed_boundary.json``
   * - Open, free boundary, axisymmetric, 0--10% beta
     - supported
     - ``benchmarks/mirror_free_boundary_axisymmetric.json``
   * - Open, free boundary, axisymmetric, above 10% up to 80% beta
     - extended validation
     - ``benchmarks/mirror_free_boundary_axisymmetric.json``
   * - Open, free boundary, nonaxisymmetric
     - deferred
     - ``benchmarks/mirror_free_boundary_nonaxisymmetric.json``
   * - Closed hybrid, circular section
     - supported
     - ``tests/mirror/test_splines.py``, ``benchmarks/mirror_hybrid_fixed_boundary.json``
   * - Closed hybrid, rotating elliptical section
     - extended validation
     - ``benchmarks/mirror_hybrid_fixed_boundary.json``
   * - Anisotropic pressure (all topologies)
     - not implemented
     - roadmap

The boundary-condition contract -- the natural boundary terms of the mirror
energy, the mechanism enforcing each, and the exterior boundary-value problem --
is derived in :doc:`/explanation/mirror-boundary-conditions`. Run recipes are
in :doc:`/howto/mirror-machines`; the ``mout_*.nc`` output format is
:doc:`/reference/wout-file`. The present code addresses ideal-MHD nested-surface
geometry with scalar pressure only; ANIMEC-derived and open-mirror anisotropic
equilibria are roadmap work.

Open coordinates and the axis condition
---------------------------------------

The open coordinates are
:math:`(s,\theta,\xi)\in[0,1]\times[0,2\pi)\times[-1,1]`. The lateral
surface :math:`s=1` is the plasma-vacuum interface; the planes
:math:`\xi=\pm1` are prescribed computational cuts that magnetic flux passes
through (their semantics, and the exterior Green surface that temporarily closes
them, are in :doc:`/explanation/mirror-boundary-conditions`). The
divergence-free representation is

.. math::

   \sqrt{g}B^\theta = I'(s)-\partial_\xi\lambda, \qquad
   \sqrt{g}B^\xi = \Psi'(s)+\partial_\theta\lambda,

with :math:`B^s=0` and a zero-surface-mean gauge for :math:`\lambda`.
All theta samples at the magnetic axis denote one physical point. Writing
:math:`q_0=\lim_{s\to0}r\,\partial_s r`, single-valued axial field requires

.. math::

   \partial_\theta\lambda(0,\theta,\xi)
   = \Psi'(0)\left[\frac{q_0(\theta,\xi)}{\langle q_0\rangle_\theta}-1\right].

The solver eliminates the axis stream function with this condition and
includes its geometry dependence in variational and implicit derivatives.
Axis ``|B|`` nonuniformity is stored as a separate axis-uniformity diagnostic.
An earlier version lacked this condition and let ``|B|`` vary over theta at
``s=0``; the canonical fixed-open records were regenerated with the regular
axis, and shaped values from before the change are invalid.

The complete nested radial profile at each cut is prescribed by the initial
state and remains fixed during primal and implicit solves. A state built only
from the LCFS uses radially self-similar cuts (``impose_self_similar_cuts``).
For a finite-radius supplied field, initialize nested cut surfaces from its
enclosed flux; otherwise the incompatible cut data appear as a localized
strong-force error even when the variational residual is small.

Resolution contract
-------------------

``MirrorResolution(ns=..., mpol=..., nxi=...)`` has one angular-resolution
input: ``mpol`` is the largest represented Fourier mode. The plasma
collocation size is the read-only value ``ntheta = 2*mpol + 1``; therefore
``mpol=0`` has one theta node and ``mpol=4`` has nine. Users cannot request an
inconsistent node/mode pair or accidentally introduce a Nyquist mode.
Exterior angular quadrature remains an independent argument because it
integrates the vacuum boundary rather than defining plasma unknowns.

Radial location map
-------------------

The radial mesh follows the VMEC staggering convention, adapted to the
two-point Gauss energy rule:

* geometry and stream-function unknowns are stored on full surfaces
  :math:`s_i`;
* metric terms, :math:`\sqrt{g}B^\theta`, :math:`\sqrt{g}B^\xi`, and magnetic
  energy are evaluated at two Gauss points inside each radial cell;
* Gauss averages define cell-centered covariant
  :math:`B_\theta`, :math:`B_\xi`, Jacobian, and pressure at
  :math:`s_{i+1/2}`;
* current and ``J x B - grad(p)`` are reconstructed on interior full surfaces;
* the magnetic axis, fixed LCFS, and open end cuts are reported separately
  from the unconstrained physical-volume norm.

Radial curl and pressure terms use conservative cell differences,

.. math::

   \sqrt{g}J^\theta_i = \frac{1}{\mu_0}
   \left[\partial_\xi B_{s,i}
   - \frac{B_{\xi,i+1/2}-B_{\xi,i-1/2}}{\Delta s}\right],

.. math::

   \sqrt{g}J^\xi_i = \frac{1}{\mu_0}
   \left[\frac{B_{\theta,i+1/2}-B_{\theta,i-1/2}}{\Delta s}
   - \partial_\theta B_{s,i}\right], \qquad
   p'_i = \frac{p_{i+1/2}-p_{i-1/2}}{\Delta s}.

Contravariant field on a full surface is reconstructed by averaging flux
density and Jacobian separately, rather than averaging their ratio. This is
the same placement used by VMEC2000 ``jxbforce`` and avoids differentiating an
unrelated full-mesh reconstruction of the energy field.

Radial magnetic energy uses two-point Gauss integration. A midpoint rule
admits an alternating lambda hourglass mode; a dedicated regression test
(``test_radial_gauss_quadrature_controls_lambda_checkerboard_mode``) assigns
that mode finite energy. Two manufactured fixtures isolate the pointwise-force
reconstruction from the nonlinear solve: a cylindrical finite-beta state with
analytic radial ``B_z(s)`` converges at second order when ``ns`` doubles, and
a theta-dependent self-similar tube carrying a uniform Cartesian field recovers
that field. These show that radial differentiation and nonaxisymmetric
coordinates work independently; they do not by themselves validate shaped
solved states.

Strong-force gate normalization
-------------------------------

The pointwise residual ``|J x B - grad(p)|`` is a force density and needs a
reference force scale to become a dimensionless gate. The primary
normalization divides by :math:`B^2/(\mu_0 a)`, where the minor radius ``a``
is the flux-equivalent LCFS radius: the midplane cross-section for open
mirrors and the circuit-averaged section for closed hybrids
(:func:`vmex.mirror.forces.effective_minor_radius`; a caller may also pass
``minor_radius`` explicitly). Transverse pressure balance acts on the
minor-radius scale, so :math:`B^2/(\mu_0 a)` is the magnitude of the
competing equilibrium forces, and every lane has one structural ``a``.

The earlier normalization used the device length ``L`` (cap-to-cap extent for
open mirrors, axis arc length for the closed racetrack). That number is
linear in an arbitrary length, so lanes with different ``L/a`` cannot be
compared by it, and the long closed racetrack reads much worse than an open
mirror with a lower absolute force density. It remains available as the
secondary ``device_normalized_rms`` diagnostic. **The committed
``benchmarks/mirror_*.json`` records quote the device-length normalization**;
they are historical evidence and are not rewritten under the new one.

Gate evaluation reports zones rather than a single folded number:
:func:`vmex.mirror.forces.force_gate_zones` returns the all-volume, bulk,
end-collar, and near-axis/first-row norms (all minor-radius-normalized)
together with the device-length total. For open mirrors, ``bulk`` and
radial-axis diagnostics use the central 80% of the axial coordinate and
``end_collar`` the outer 20% nearest the two fixed cuts; the all-volume norm
retains both. Constrained-data regions -- the frozen end cuts and the
regularized axis -- are therefore visible next to the unconstrained bulk.
Promotion additionally requires refinement evidence:
:func:`vmex.mirror.forces.refinement_convergence` reports per-step ratios and
monotonicity for residuals from two or more resolutions, and
:func:`vmex.mirror.forces.passes_promotion_gate` combines the absolute gate
(``0.05`` in the examples and tests) on the finest rung with that
monotone-decrease requirement.

Open fixed boundary
-------------------

Spline basis and state
~~~~~~~~~~~~~~~~~~~~~~

``vmex.mirror.splines.CubicBSplineBasis`` is the compatibility name for the
shared :class:`vmex.core.radial_basis.BSplineBasis`. Open mirrors use clamped
knots, exact endpoint values, Gauss-Legendre quadrature on every nonzero span,
and exact Boehm knot insertion. Closed hybrids use folded uniform periodic
splines and exact dyadic refinement. The default remains cubic; the shared
basis also supports odd degrees 5 and 7. Evaluation, two derivatives,
coefficient transfer, and its transpose are JAX operations. Tests match SciPy,
reproduce polynomials through the active degree, preserve open and periodic
curves under refinement, verify partition of unity and periodic closure, and
check JVP/VJP actions.

``SplineMirrorState`` and ``SplineMirrorBoundary`` store geometry and stream
function coefficients rather than sampled values. ``SplineMirrorDiscretization``
evaluates them on endpoint-augmented Gauss nodes before calling the shared
geometry and energy kernels, and applies side/end constraints plus the lambda
gauge in coefficient space. For a quadratic flared tube, the spline state
matches the 41-node Chebyshev reference in volume to ``rtol=3e-14`` and in
total energy to ``rtol=2e-12``
(``test_coefficient_native_state_matches_chebyshev_polynomial_geometry_and_energy``).

Solve
~~~~~

The public ``solve_fixed_boundary`` minimizes the scalar-pressure energy
directly in the active spline coefficients. It fixes the side and end
coefficients, eliminates the weighted stream-function gauge, and uses the
shared host L-BFGS plus residual-Newton policy. The independent staggered first
variation is assembled on the quadrature grid and pulled back through the
spline evaluation matrix rather than reused from autodiff. The convenience
wrapper ``solve_fixed_boundary_from_radius(radius, config)`` builds a default
axisymmetric boundary of the requested LCFS radius and runs this solve in one
call. Public inputs are ``SplineMirrorBoundary``, ``SplineMirrorState``, and
``SplineMirrorDiscretization``; the result is a ``SplineMirrorSolveResult``.
CGL values remain available for quadrature and evaluated-state parity tests,
not as a second production state.

Open fixed-boundary systems use exact JAX Hessian-vector products and
matrix-free Newton at every size. Radius-only systems use the
radial/Fourier/axial tensor inverse. Finite-current nonaxisymmetric systems
freeze a sparse local Hessian at the start of Newton: columns are evaluated in
batches of 32, only same-field-channel terms with neighbouring radial rows and
axial coefficient distance at most four are stored, and all poloidal coupling
is retained. Sparse LU is reused for every Newton step; no dense Hessian is
stored. Tangent and adjoint systems reuse the traceable separable
preconditioner; the host sparse factor is a primal acceleration and is not
differentiated.

Changing a prescribed spline boundary uses
``SplineMirrorDiscretization.transfer_boundary``. It rescales every nested
surface at spline collocation nodes before projection, instead of replacing
only the LCFS and risking crossed surfaces. The optimizer also rejects any
trial with a changed Jacobian sign, matching the toroidal merit policy.

``initialize_from_cartesian_field`` keeps a supplied spline geometry fixed,
infers :math:`\Psi'(s)` from the surface-averaged axial flux, and obtains the
nonzero poloidal stream-function modes from the remaining contravariant field.
It accepts either Cartesian field samples or a point callable and performs no
coil construction or Biot--Savart integration.

Evidence
~~~~~~~~

All values below are from ``benchmarks/mirror_fixed_boundary.json``
(``ftol=1e-12``; strong-force values device-length normalized).

*Axisymmetric refinement* (supported). Three independent solves at
``(ns, nxi, elements)`` = ``(5, 9, 4)``, ``(7, 13, 6)``, ``(9, 17, 8)`` give
all-volume strong force ``0.0435 -> 0.0347 -> 0.0287`` and relative field RMS
error ``1.45e-3 -> 1.07e-3 -> 7.02e-4``, with variational and independent weak
residuals at or below ``1.22e-15`` on every rung.

*Rotating ellipse* (release candidate). A flux-conserving ellipse whose major
axis turns by 90 degrees between the cuts, LCFS radius ``0.12 m``, solved at
``(ns, mpol, elements) = (7, 6, 6)`` with corrected (self-similar) cuts:
variational residual ``1.72e-16``, independent weak residual ``1.694e-16``,
normalized divergence ``6.61e-15``, and strong force ``0.0418`` all-volume,
``0.0190`` bulk, ``0.0732`` end collar. Its section is an exact discrete flux
surface.

*Straight field-line mirror (SFLM)* (validation only). The Ågren--Savenko field
(`Phys. Plasmas 11, 5041 (2004) <https://doi.org/10.1063/1.1799351>`_) is an
equilibrium only to order :math:`(a/c)^2`, and its analytic cut profile is held
fixed at the two end cuts. At LCFS radius ``0.10 m`` and the same resolution,
the corrected-cut solve reaches variational residual ``3.101e-16``, weak
residual ``3.11e-16``, and divergence ``7.22e-15``, but its strong force is
``1.09`` all-volume, ``0.175`` bulk, and ``2.272`` end collar, and the
record's ``straight_field_line_independent_strong_force`` gate is ``false``.
The pre-corrected-cut SFLM refinement ladder was removed from the record
because it no longer reproduces the current boundary condition; no refinement
evidence for the corrected cuts is recorded. The capability contract lists
broader straight-field-line validation as incomplete.

.. image:: /_static/figures/mirror_fixed_boundary_3d.webp
   :alt: Solved axisymmetric and 90-degree rotating-ellipse fixed-boundary mirrors coloured by LCFS field strength
   :width: 100%

Analytic nonaxisymmetric fixtures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``vmex.mirror.analytic`` contains validation data that never call an
equilibrium solve. ``RotatingEllipseParaxial`` maps a unit circle through a
flux-conserving ellipse whose major axis turns by 90 degrees from one end to
the other. A compensating field-line-label angle makes the first-order vacuum
identity vanish while preserving

.. math::

   X_{1c}Y_{1s}-X_{1s}Y_{1c}=\bar B/B_0.

It evaluates the Rodriguez-Helander-Goodman Appendix-C Riccati equation and
the independent general and magnetic-well-minimum formulas for
``(B20,B2c,B2s)``. Tests recover those coefficients from low-radius Fourier
samples and verify that the order-``r`` ``m=1`` field strength is zero. This is
the coefficient oracle for the native-spline fixed-boundary solve; it is not
itself an equilibrium.

``StraightFieldLineMirror`` implements the SFLM of Ågren and Savenko: the
marginally stable minimum-:math:`B` vacuum field whose flux lines are straight
but nonparallel. The second-order paraxial scalar potential, the on-axis field,
the ellipticity, and the straight field lines :math:`x = x_0(1 + z/c)`,
:math:`y = y_0(1 - z/c)` are all from `Magnetic mirror minimum B field with
optimal ellipticity <https://doi.org/10.1063/1.1799351>`_ (Phys. Plasmas
**11**, 5041, 2004); the Cartesian-like Clebsch labels :math:`(x_0, y_0)` used
by ``clebsch_labels``, and the proof that the marginal minimum-:math:`B` field
is quadrupolar up to a rigid rotation, are from `Rigid rotation symmetry of a
marginally stable minimum B field and analytical expressions of the flux
coordinates <https://doi.org/10.1063/1.1870002>`_ (Phys. Plasmas **12**,
042505, 2005). VMEX truncates the potential at relative order
:math:`(a/c)^4`, where the 2004 derivation itself stops. Its tests verify
curl-free field, the expected order-``(a/c)^2`` solenoidal and field-line
truncation errors, axial flux conservation, and

.. math::

   B_0(z)=\frac{B_0(0)}{1-z^2/c^2}, \qquad
   \mathcal E(z)=\frac{1+|z/c|}{1-|z/c|}.

Both fixtures require a thin tube and ``|z|<c``. The long-thin ordering
expands in :math:`(a/L)^2` alone: :math:`\beta` is *not* a second small
parameter (see `Interpreting beta`_). They are asymptotic equilibrium
references, not finite-beta solutions or ellipticity predictions.

Periodic stellarator-mirror hybrid
----------------------------------

The implemented target follows the topology proposed for `stellarators
linking axisymmetric mirrors (SLAM)
<https://downloads.regulations.gov/DOE-HQ-2023-0038-0020/attachment_1.pdf>`_:
two long straight mirror legs are joined by two curved stellarator sections.
The same topology appears in the `APS linked-mirror abstract
<https://meetings.aps.org/Meeting/DPP22/Session/NP11.24>`_. It is related to,
but geometrically distinct from, the warm-stellarator/rectilinear-mirror
fusion--fission hybrid proposed by `Moiseenko et al.
<https://doi.org/10.1017/S0022377823000442>`_. The implementation uses de
Boor's local cubic B-spline construction and `Bishop's rotation-minimizing
frame <https://doi.org/10.1080/00029890.1975.11993807>`_. It does not model
fusion--fission blankets, minority fast ions, end losses, or kinetic
stability.

Let :math:`u\in[0,2\pi)` parameterize the closed circuit. The coordinate map is

.. math::

   \boldsymbol x(s,\theta,u) = \boldsymbol c(u)
   + \sqrt{s}\,a(s,\theta,u)
     [\cos\theta\,\boldsymbol n(u)+\sin\theta\,\boldsymbol b(u)].

``CubicBSplineBasis.periodic_uniform`` represents :math:`\boldsymbol c`.
Control points are divided between straight leg, return, opposite straight
leg, and second return. A cubic spline has local support, so every central span
whose four active controls are collinear is exactly straight; increasing the
number of controls lengthens the exact straight region rather than merely
improving a global Fourier fit. ``evaluate_closed_spline_axis`` transports a
Bishop rotation-minimizing frame :math:`(\boldsymbol n,\boldsymbol b)` around
the curve and distributes its residual holonomy over the period. This frame is
well defined where the straight-leg curvature is exactly zero, unlike the
Frenet frame.

The LCFS is an ellipse written as a polar radius,

.. math::

   a(1,\theta,u) =
   \frac{A B}{\sqrt{[B\cos(\theta-\alpha(u))]^2
                   +[A\sin(\theta-\alpha(u))]^2}}.

The section angle is :math:`\alpha(u)=\alpha_{\mathrm{ret}}(u)+N\,u`. The
return term :math:`\alpha_{\mathrm{ret}}` is constant on each straight leg and
changes smoothly by 90 degrees through each return. The ``section_turns``
integer :math:`N` superposes a genuine rotating ellipse: the major axis turns
continuously by :math:`N` full :math:`2\pi` turns per circuit. Because the polar
radius is :math:`2\pi`-periodic in :math:`\theta-\alpha` and :math:`N` is an
integer, the section closes on itself exactly, and the straight-leg axis stays
exactly straight while the ellipse it carries rotates. The radial surfaces and
stream function use the same periodic longitudinal basis. The divergence-free
field is the open expression with :math:`\xi` replaced by :math:`u`;
periodicity removes end cuts, and all longitudinal coefficients are active. A
finite :math:`I'(s)` gives nonzero rotational transform; the rotating section
amplifies that current-driven transform rather than supplying a standalone
geometric one.

``build_stellarator_mirror_hybrid`` constructs the discretization, closed
axis, LCFS, and a vacuum-field initial stream function. The ordinary
``solve_fixed_boundary`` then uses the same energy, host globalization,
matrix-free Hessian actions, and separable preconditioner as open mirrors.
``trace_closed_field_line`` integrates :math:`d\theta/du=B^\theta/B^u` with
periodic RK4 steps. Source ownership: periodic basis/refinement is in
``basis.py``; axis, Bishop frame, and embedding are in ``geometry.py``;
coefficient packing, initialization, solve dispatch, and tracing are in
``splines.py``; the shared energy and force diagnostics are in ``forces.py``;
and the figure is produced by ``output.plot_stellarator_mirror_hybrid``.

Junction freeze and the circular section
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The leg-return junction -- where an exactly straight leg (zero curvature)
meets a circular return (curvature :math:`1/R`) -- is rounded across the cubic
spline's local support. Building the axis directly in a finer solve basis
narrows that rounding and sharpens the junction curvature overshoot, so the
as-built geometry family is a different curve at every resolution. The
committed record shows the consequence: in
``benchmarks/mirror_hybrid_fixed_boundary.json`` (``same_geometry``), the
as-built circular-section ladder at 16/32/64 controls reads
``0.0528 -> 0.108 -> 0.0237`` (quadrature order 3, device normalized) --
the finest rung is below ``0.05`` but the ladder is not monotone
(``monotone_same_geometry_refinement: false``), and the record's status is
``active-validation``.

``build_stellarator_mirror_hybrid(axis_coefficient_count=...)`` freezes the
junction as an explicit design parameter of the closed B-spline axis family.
The racetrack axis and rotating section are constructed at that base control
count and then exactly refined (``refine_periodic_uniform``, dyadic and
curve-preserving to roundoff) to the solve ``coefficient_count``, so the
junction-transition width is held fixed while the equilibrium resolution
increases. The default ``axis_coefficient_count=None`` keeps the legacy
behaviour of building the geometry in the solve basis.

Under the frozen junction, the full-lane test
``test_stellarator_mirror_frozen_junction_force_ladder_converges``
(``tests/mirror/test_splines.py``, ``ns=5``, ``mpol=2``, junction frozen at
16 controls, solve basis 16/32/64) requires every rung to reach
``ftol=1e-12`` with normalized divergence below ``1e-12``, a monotone
device-normalized all-volume ladder that starts above ``0.2`` and ends below
it, and a monotone minor-radius bulk ladder that passes the promotion gate
with the finest rung below ``2e-3``. This test, not the JSON record, is the
refinement evidence for the supported circular-section row; its per-rung values
are not committed to a benchmark record.

The rotating elliptical section
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The rotating-elliptical-section hybrid is in extended validation: the
capability contract keeps its independent strong-force promotion gate open.
The recorded ladder (``h1_20260716`` in
``benchmarks/mirror_hybrid_fixed_boundary.json``, return-only rotation,
``ns=5``, ``mpol=3``, 16/32/64 controls) converges variationally
(``variational_max`` at most ``6.60e-14``) and decreases monotonically, but the
device-normalized strong force plateaus at ``0.5733 -> 0.356 -> 0.333``, above
``0.05`` (``h1_absolute_strong_force: false``). Radial/poloidal refinement to
``(ns, mpol) = (7, 4)`` lowers it to ``0.2271``; the ``(9, 5)`` rung was stopped
at the resource gate.

A beta-zero ablation at ``ns=5``, ``mpol=3``, 32 controls (same record) localizes
the floor to the racetrack geometry rather than to pressure or current: the
rotating ellipse gives ``0.430`` with current and ``0.424`` without; a fixed
ellipse gives ``0.1641``, a circular section ``0.158``, and the circular-axis,
circular-section limit ``0.00832``.

``section_turns`` adds genuine toroidal rotation of the section. The shipped
example (``examples/mirror/stellarator_mirror_hybrid.py``: ``ns=5``,
``mpol=4``, 32 controls, ``semi_major=0.45``, ``semi_minor=0.25``,
``section_turns=2``) converges to ``ftol=1e-12`` with a divergence-free field,
and ``test_stellarator_mirror_toroidal_rotation_raises_transform`` requires the
traced transform at ``s=0.75`` to exceed ``0.12`` at the example's imposed
current. The near-axis representation defect behind the force floor is not
resolved by the rotation, so finite-beta continuation and rotating-section
sensitivity claims are deferred.

.. image:: /_static/figures/stellarator_mirror_hybrid.webp
   :alt: Solved periodic B-spline stellarator-mirror hybrid with its axis, boundary magnetic field, cross-sections, transform, and residuals
   :width: 100%

QI-mirror hybrid: Fourier vs B-spline
-------------------------------------

A quasi-isodynamic (QI) stellarator has poloidally closed ``|B|`` contours and
low-curvature magnetic-axis segments at its stellarator-symmetry planes, so
the QI axis is a natural place to cut and insert a straight mirror cell.
``examples/mirror/qi_mirror_hybrid_fourier_vs_bspline.py`` makes that
construction concrete and compares the two ways of representing the resulting
axis.

*Cut locations.* The example solves ``input.nfp2_QI`` with the VMEC (Fourier)
core, reads the magnetic axis :math:`\mathbf{r}(\phi)` from the axis Fourier
arrays, and computes the 3-D curvature
:math:`\kappa = \lVert \mathbf{r}' \times \mathbf{r}'' \rVert / \lVert
\mathbf{r}' \rVert^{3}` spectrally over the torus. An nfp=2 QI axis has four
curvature minima, at the four stellarator-symmetry planes; all four are cut,
so the inserted legs are stellarator symmetric.

*Cut-and-splice.* ``splice_straight_legs`` cuts the closed axis at the minima
and inserts an exactly-straight leg at each **along the local axis tangent**.
The per-cut leg lengths are chosen so the inserted displacements cancel and
the loop closes; this splits the legs into two symmetry classes of equal
length. One stellarator-symmetric half is built and reflected about the ``x``
axis. ``tests/mirror/test_qi_hybrid.py`` requires closure below ``1e-12``,
legs aligned with the local tangent, and pairwise-equal leg lengths. The seam
is therefore tangent-continuous; what remains is a *curvature* break -- the
zero-curvature leg meeting the finite-curvature return -- which is what
separates the two representations.

*Representation accuracy.* The same closed hybrid axis is fitted with a
truncated Fourier series (global, VMEC-native) and a periodic cubic B-spline
(local, the ``vmex.mirror`` lane). On a model QI axis,
``test_bspline_reproduces_legs_better_than_fourier`` requires the 256-control
B-spline to reproduce the leg midpoint below ``1e-9``, while a 32-harmonic
Fourier least-squares fit rings above ``1e-4`` on the leg interior. The
B-spline reproduces the straight cell to machine precision once each leg is
backed by enough collinear controls (its error is confined to a few
knot-spacings around the junction, the signature of local support); the
global Fourier series rings across the whole leg. The *maximum* error of both
bases is set by the leg/return curvature break (a cubic B-spline is
:math:`C^2` and also rounds a curvature step), so that limit is shared.

*Equilibrium.* ``build_qi_mirror_hybrid`` fits the spliced axis into the closed
solve basis, wraps it in a constant circular section (rotation-invariant, so the
large frame holonomy of a fully 3-D axis does not enter the boundary), and
returns a ``StellaratorMirrorSetup`` that feeds ``solve_fixed_boundary`` like the
analytic racetrack. This B-spline equilibrium is a scalar-pressure spline model
whose transform comes from a weak axial current: it demonstrates the geometry
and the exactly-straight mirror cell, **not** a reproduction of the QI
rotational transform.

*Fourier-lane limitation.* A literal VMEC re-solve of a straight-axis QI-mirror
device is out of scope by construction: VMEC's toroidal coordinate is the
cylindrical angle :math:`\phi`, and a straight axis segment cannot be
parameterised by :math:`\phi` (its :math:`R(\phi), Z(\phi)` are degenerate).
This is why the closed-axis, arc-length-parameterised B-spline lane exists.
The VMEC solve serves as the QI *reference* (axis, ``iota``, ``|B|``), and the
Fourier side of the comparison is the axis-representation accuracy above.

.. image:: /_static/figures/qi_mirror_hybrid.webp
   :alt: QI-mirror hybrid comparison: QI axis coloured by curvature with cut locations, the spliced straight-leg hybrid axis, hybrid boundary magnetic field, and Fourier-versus-B-spline representation accuracy at the seam
   :width: 100%

Implicit gradients
------------------

Fixed boundary
~~~~~~~~~~~~~~

``spline_fixed_boundary_adjoint`` differentiates a scalar diagnostic through
the converged coefficient residual. Boundary and periodic-axis spline
coefficients, flux, conserved mass, and current remain differentiable. The
transpose Hessian action uses exact JAX reverse AD, and the nonlinear iteration
history is never differentiated or stored. The example
``examples/mirror/mirror_fixed_boundary_nonaxisymmetric.py`` differentiates
rotating-ellipse volume with respect to a native boundary coefficient; the
record gives agreement with two fully reconverged centered-difference solves
to ``5.01e-10`` relative, with transpose linear residual ``1.80e-11``
(``derivatives`` in ``benchmarks/mirror_fixed_boundary.json``). An SFLM
adjoint is not reported: its analytic cut profile is fixed input data, not a
solved discrete flux surface, so a shape derivative through it is not a
meaningful optimization sensitivity.

``spline_fixed_boundary_tangent`` solves the complementary forward system
``F_u du = -F_p dp`` with exact residual JVPs and the same preconditioner. On a
nonaxisymmetric finite-current ``solve_lambda=True`` case,
``tests/mirror/test_implicit.py`` requires radius and stream-function
tangents to agree with two fully reconverged centered differences within
``2e-4`` in relative state norm, with linear residual below ``1e-8``.

On the closed circular limit, periodic boundary and axis controls pass the
parameter JVP/VJP transpose identity to ``rtol=2e-12``, and an adjoint volume
derivative agrees with fully reconverged centered differences at steps
``4e-4``, ``2e-4``, ``1e-4`` (finest ``rtol=2e-5``)
(``differentiation_20260716`` in
``benchmarks/mirror_hybrid_fixed_boundary.json``). The supported
circular-section racetrack is a valid sensitivity target; the
rotating-elliptical-section racetrack does not carry a sensitivity claim.

Axisymmetric free boundary
~~~~~~~~~~~~~~~~~~~~~~~~~~

``free_boundary_adjoint`` differentiates the axisymmetric exterior equilibrium
with respect to a differentiable external-field callable, axial flux, conserved
mass profile, and axial current. The physical fixed point contains the lateral
LCFS and plasma-interior radii. The exterior Neumann BIE eliminates vacuum
unknowns, so its exact reverse-AD field and shape responses enter the
interface-stress rows directly. The transpose solve reuses the separable primal
plasma preconditioner and does not assemble a dense Jacobian or retain
nonlinear iterations. Pressure changes through the same conserved-mass and
solved-volume relation as the primal; end-cut radii and any central-pressure
calibration target remain fixed. Coil-design derivatives belong to the ESSOS
integration layer; VMEX differentiates only the supplied field object.

The adjoint is supported through the 10% beta ceiling. The implicit derivative
matches a reconverged finite difference to ``1.08e-10`` relative (adjoint
relative residual ``1.37e-9``; ``implicit_derivative`` in
``benchmarks/mirror_free_boundary_axisymmetric.json``). The nonaxisymmetric
free-boundary derivative is unavailable because that lane is deferred.

Linear backend
~~~~~~~~~~~~~~

The one-shot mirror adjoint uses SciPy GMRES around exact JAX JVP/VJP actions
and inherits the placement of its primal arrays. The recorded backend audit
(``linear_backend_audit_20260716`` in
``benchmarks/mirror_hybrid_fixed_boundary.json``) kept host GMRES: a SOLVAX
GPU solve on the closed adjoint is faster warm but its first compiled solve
does not amortize in a one-shot derivative API. Fixed- and free-boundary
derivatives solve the linearized converged coefficient residual and never
retain or differentiate the nonlinear iteration history.

Reported mirror ratios and lengths
----------------------------------

"Mirror ratio" and "mirror length" have no single conventional meaning, so
VMEX pins four quantities in :mod:`vmex.mirror.metrics` and every example,
test and doc page in the mirror lane reports those and nothing else. All four
are host-side diagnostics built from discrete extrema; they are not
differentiable.

:math:`R_{m,\rm axis}` (per leg)
   :math:`\max|B| / \min|B|` on the magnetic axis over one :math:`|B|` well,
   the well being the axial interval between the two :math:`|B|` maxima that
   bound its minimum. An open mirror has one well; the periodic hybrid has one
   per straight leg, which is why it is reported per leg. On an open mirror the
   two ends of the modelled grid count as bounding maxima, since the coils
   usually sit outside the grid.

:math:`R_{m,\rm LCFS}`
   :math:`\max|B| / \min|B|` over the last closed flux surface. It is a
   different number from :math:`R_{m,\rm axis}` on any shaped boundary, and is
   always reported separately rather than as "the" mirror ratio.

:math:`L_{\rm mirror,B}` (per leg)
   The arc-length distance between the two :math:`|B|` maxima bounding a well:
   the length of the mirror cell as the field measures it, not as the device
   does.

:math:`L_{\rm straight}`
   The arc length over which the axis curvature is negligible (below a
   fraction ``tolerance`` of the largest curvature on the axis, default
   ``1e-3``), reported with its individual spans. It is a geometric quantity
   and is unrelated to :math:`L_{\rm mirror,B}` unless the :math:`|B|` maxima
   happen to fall at the leg ends.

Wells shallower than 5% of the total on-axis :math:`|B|` swing (the default
``minimum_relative_depth``) are merged into their neighbours by persistence
pruning, so sampling ripple in a solved :math:`|B|` is not reported as an extra
leg. For the racetrack hybrid with two 8 m legs,
``test_closed_racetrack_hybrid_has_throatless_legs_and_two_straight_spans``
(``tests/mirror/test_metrics.py``) requires exactly two wells, leg ratios
:math:`R_{m,\rm axis} < 1.01` -- the legs have no throat, because the leg
semi-axes are constant -- an :math:`R_{m,\rm LCFS}` above 2 from the shaped
returns, and two equal straight spans with total :math:`L_{\rm straight}`
between 10 and 16 m: the spline's cubic local support rounds each leg-return
junction, so the exactly straight arc is shorter than the nominal
:math:`2 \times 8` m.

Two related quantities elsewhere in VMEX are deliberately *not* mirror ratios.
:func:`vmex.core.optimize.mirror_ratio` is the ``|B|`` modulation depth
:math:`m=(B_{\max}-B_{\min})/(B_{\max}+B_{\min})` on one surface, the QI
optimization knob, related by :math:`R_m = (1+m)/(1-m)`. The ``epsilon`` key
of the gyrokinetic flux-tube contract is that same modulation depth taken along
one field line; its definition and the exported field-line mirror ratio are in
:doc:`mirror-gyrokinetics`.

Interpreting beta
-----------------

The equilibrium uses a VMEC-style conserved mass profile. Because geometry
changes pressure at fixed mass, the beta-scan driver adds one mass-amplitude
unknown and one central-pressure equation to the coupled nonlinear system.
Requested and achieved central beta therefore agree to the solve tolerance
without an outer sequence of complete solves. For the default profile
``p(s) = p0 (1-s)``, pressure vanishes at the LCFS, so a 10% central beta does
not imply a 10% edge-pressure jump or volume beta.

``summarize_axisymmetric_beta_scan`` reports:

* requested central beta,
* achieved central beta normalized by the reference vacuum field,
* volume-averaged beta,
* local central beta normalized by the solved plasma field,
* center radius and plasma/vacuum-side field,
* diamagnetic field ratio, and
* error against the long-thin relation ``B/B_vac = sqrt(1-beta)``.

That relation is not a small-:math:`\beta` expansion. It is Eq. (30) of Ryutov
*et al.*, `Magneto-hydrodynamically stable axisymmetric mirrors
<https://doi.org/10.1063/1.3624763>`_ (Phys. Plasmas **18**, 092301, 2011): the
leading-order term of the long-thin (paraxial) expansion in
:math:`(a/L)^2`, exact in :math:`\beta` for any :math:`\beta < 1`. Its accuracy
is set by the aspect of the device, not by how large :math:`\beta` is.

Recorded case and status
~~~~~~~~~~~~~~~~~~~~~~~~

``benchmarks/mirror_free_boundary_axisymmetric.json`` records a two-loop ESSOS
coil set (loop radius ``0.9 m`` at ``z = +/-1.0 m``, ``2.0e5 A`` each, cuts at
``z = +/-0.8 m``, initial center radius ``0.25 m``) and the beta sequence
0, 1, 3, 10, 25, 50%. With :math:`a = 0.25` m and the half coil separation
:math:`L = 1.0` m, :math:`(a/L)^2 = 0.0625`. The record states three distinct
things, which should not be conflated:

* **Three-grid refinement gate (sets the status).** The ``refinement`` rows
  pass for 0, 1, 3, and 10% and fail for 25% and 50% (``passed: false``;
  ``gates.beta_25_force`` and ``gates.beta_50_force_and_observables`` are
  ``false``). The record status is ``supported-through-beta-0.10``.
* **Fine-grid convergence.** A separate run at
  ``(ns, nxi, elements, exterior_ntheta) = (13, 25, 13, 24)``
  (``fine_grid_promotion.fine_grid_50``) converges every point from 0 through
  50% (at most 44 Newton-GMRES iterations, variational residual at most
  ``8.5e-15``) with minor-radius bulk force at most ``2.41e-3``, below the
  ``0.05`` gate; the record sets ``passes_through_beta`` to ``0.5``.
* **Contract.** The capability contract has not been promoted on that run: the
  supported range is 0--10%, and above 10% up to 80% is extended validation
  (:doc:`/reference/capabilities`). The shipped beta-scan example also runs an
  80% point, which no record covers.

At 50% the recorded research observables are a center-radius change of
``+7.73%`` and a center-field change of ``-23.73%``, i.e. a field ratio of
``0.76269`` against the long-thin value :math:`\sqrt{0.5} = 0.7071`. That gap is
of the size of the neglected :math:`O((a/L)^2)` correction, not evidence that
the relation fails at large :math:`\beta`; the solve is the finite-aspect
answer. This is an equilibrium benchmark only: it does not establish flute,
firehose, mirror-mode, or kinetic stability.

The shipped example ``examples/mirror/mirror_free_boundary_beta_scan.py`` uses
a different, compact coil set sized to the plasma (loop radius ``0.5 m`` at
``z = +/-1.0 m``, ``3.72e5 A`` each), which keeps the vacuum on-axis midplane
field of the recorded geometry (about ``0.0836 T``) with a deeper mirror well.

The finite-beta mirror trend follows the WHAM/Pleiades discussion in Frank et
al., `Confinement performance predictions for a high field axisymmetric tandem
mirror <https://doi.org/10.1017/S002237782510055X>`_. An independent Pleiades
Green-function reference for the recorded two-coil case (1, 3, and 10% beta on
three grids) is committed as
``examples/data/pleiades_two_coil_beta_reference.csv`` and regenerated by
``examples/mirror/pleiades_mirror_reference.py`` from an external Pleiades
checkout; no VMEX-versus-Pleiades comparison is recorded.

.. image:: /_static/figures/mirror_free_boundary_beta_scan.webp
   :alt: Solved axisymmetric free-boundary mirror beta scan with ESSOS coils
   :width: 100%

Open exterior
-------------

The exterior boundary-value problem, its Neumann data, the cap-only
compatibility projection, and the Duffy-collocation discretization are derived
in :doc:`/explanation/mirror-boundary-conditions`; this section records
implementation choices.

:func:`vmex.mirror.build_closed_mirror_surface` closes a star-shaped lateral
LCFS with disks at both fixed-flux cuts. It stores outward ``n dA`` directly,
including quadrature weights, so the disk center is regular and no unit-normal
division enters geometric identities. For axisymmetric equilibrium grids, which
intentionally store one theta node, the adapter supplies an independent
Cartesian angular quadrature; ``axisymmetric_ntheta`` controls its resolution
without adding redundant theta unknowns. ``ClosedMirrorSurface`` provides a
unique collocation grid and an explicit map back to all quadrature nodes, so
repeated geometry never becomes duplicate BIE unknowns, and the same unique
nodes define a watertight outward-oriented triangular panel mesh. For
axisymmetric data the angular panel nodes remain fully resolved as sources but
one target is evaluated per rotational orbit. Power grading in the cap radius
clusters nodes at the rim without changing cylinder area or volume.

On a decaying-dipole manufactured solution, ``tests/mirror/test_exterior.py``
requires condition number below 5, equation residual below ``3e-14``, and
boundary and lateral-field errors that decrease monotonically under refinement.
Linear density interpolation on side triangles limits the lateral field; an
opt-in spectral side-density rule evaluates lateral Dirichlet and Neumann data
with global Fourier-Chebyshev interpolation on the same linear panel geometry
and must cut the dipole error by more than a factor of three
(``test_spectral_side_density_improves_exterior_dipole``). The beta-scan
example enables it with ``EXTERIOR_SPECTRAL_SIDE_DENSITY = True``, and the
recorded case used it (``spectral_side_density: true``).

``solve_axisymmetric_exterior_vacuum`` owns the complete adapter: it closes the
moving boundary, continues the plasma field through both end cuts, cancels the
supplied external normal field on the side, solves the exterior Neumann
problem, and returns the lateral total-field trace. It is the sole vacuum
model in the coupled axisymmetric free-boundary and beta-continuation drivers.
Restart files contain only the plasma state, boundary, and pressure scale; the
BIE potential is solved into ``result.vacuum_field.neumann_result`` rather
than hot-started. MGRID and vectorized ``xyz -> B`` callables share one
external-field adapter; MGRID interpolation tests remain in VMEX, and filament
sampling and Biot-Savart parity tests live with ESSOS. The free-space model has
no artificial outer cylinder.

The coefficient solver uses a dense ``jacfwd`` only through 32 unknowns; larger
systems expose exact repeated JVP/VJP actions through a SciPy ``LinearOperator``
with no materialized Jacobian, globalized by a bounded trust region and
polished by a Newton--GMRES step using the fixed-solver spline preconditioner.
Generic interior and full-node virtual-casing adapters are left to
``virtual_casing_jax``; VMEX retains only the operator used by mirror
equilibria.

Two cheaper boundary-limit approximations were tried and rejected: offset
collocation gave ill-conditioned density systems, and replacing each singular
single-layer panel by an equal-area disk was stable but converged only
algebraically. The implementation therefore follows local singular quadrature.
Relevant numerical foundations are Duffy's `vertex-singularity transform
<https://doi.org/10.1137/0719090>`_ and the distinction between smooth-surface
QBX error control and explicit corner treatment discussed by
`af Klinteberg and Tornberg <https://arxiv.org/abs/1603.08366>`_ and
`Helsing and Ojala <https://doi.org/10.1016/j.jcp.2008.06.022>`_.

Nonaxisymmetric free boundary (deferred)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``benchmarks/mirror_free_boundary_nonaxisymmetric.json`` keeps the negative
evidence. A three-grid study reached roundoff nonlinear residuals, but the
center ``m=1`` observable changed by ``81.3%`` (beta 0) and ``72.7%`` (beta
50%) between grids, and the beta-pair wall time grew from ``293 s`` to
``2995 s`` (NVIDIA RTX A4000). Those states also predate the corrected
magnetic-axis regularity map, so the theta-dependent exterior was removed
rather than presented as an equilibrium model. The fixed-boundary rotating
ellipse remains the release-candidate nonaxisymmetric case.
