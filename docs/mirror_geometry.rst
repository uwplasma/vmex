Mirror geometry
===============

``vmec_jax.mirror`` is the open-field-line equilibrium backend. It uses
coordinates ``(s, theta, xi)`` with a nonperiodic axial coordinate and
fixed-flux end cuts. It does not reinterpret a straight mirror as a periodic
torus. Axisymmetric and nonaxisymmetric fixed-boundary lanes are supported,
as is axisymmetric free boundary through 10% requested beta. Higher-beta and
nonaxisymmetric free-boundary states remain research lanes. Periodic hybrids
are explicitly deferred after failing their same-geometry refinement gate.

Open topology and end cuts
--------------------------

The open coordinates are
:math:`(s,\theta,\xi)\in[0,1]\times[0,2\pi)\times[-1,1]`. The lateral
surface :math:`s=1` is the plasma-vacuum interface. The planes
:math:`\xi=\pm1` are prescribed computational cuts through the flux tube;
magnetic flux passes through them, so they are neither material interfaces nor
``B.n=0`` boundaries. The divergence-free representation is

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
Axis ``|B|`` nonuniformity is stored as a separate promotion diagnostic.

An unbounded exterior Green solve requires a geometrically closed integration
surface, so disks temporarily close the two cuts. Their Neumann data continue
the nonzero plasma and applied-field through-flux across each cut. The disks do
not close the plasma or acquire an interface pressure-balance equation.
Tangency and total-pressure continuity are enforced only on the lateral LCFS.
The complete nested radial profile at each cut is prescribed by the initial
state and remains fixed during primal and implicit solves. A state built only
from the LCFS uses radially self-similar cuts. For a finite-radius supplied
field, callers should instead initialize nested cut surfaces from its enclosed
flux; otherwise the incompatible cut data appear as a localized strong-force
error even when the variational residual is small.

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

For open mirrors, ``bulk`` and radial-axis diagnostics use the central 80% of
the axial coordinate. ``end_collar`` uses the outer 20% nearest the two fixed
cuts. The all-volume norm retains both regions.

In particular, radial curl and pressure terms use conservative cell
differences,

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

Current capability
------------------

The branch currently includes:

* supported axisymmetric and nonaxisymmetric fixed-boundary finite-current
  solves,
* an isotropic VMEC-style conserved-mass pressure energy with independent
  weak and pointwise force diagnostics,
* a free-space boundary-integral vacuum model with an ``xyz -> B`` field
  callable or the shared ESSOS/MAKEGRID-compatible ``MgridField``; coil
  geometry and Biot-Savart evaluation remain in ESSOS,
* coupled axisymmetric free-boundary beta continuation with compressed
  restart files,
* a closed-surface Neumann solve on the lateral LCFS and both end disks,
* component-wise nonlinear convergence checks at a requested ``ftol=1e-12``.

The axisymmetric free-boundary path is supported through 10% requested beta
and retains 25% and 50% as explicitly labeled research scans. The
nonaxisymmetric free-boundary path is a deferred research lane because its
point observables were not monotone under spatial refinement.

Deferred closed hybrid
----------------------

The closed stellarator-mirror candidate failed its declared same-geometry
refinement gate. Its strong-force sequence was
``0.05277, 0.10756, 0.02365`` at quadrature order 3 and
``0.05280, 0.10714, 0.02353`` at order 4. The nonmonotone middle state is
reproduced at both orders, so it cannot be promoted by considering only the
finest value.

Periodic splines, moving-frame geometry, center maps, closed initializers,
and closed nonlinear/preconditioner paths were removed from the release
runtime. The compact negative evidence remains in
``benchmarks/mirror_hybrid_fixed_boundary.json``; a future hybrid must start
from the bounded design in ``plan.md`` and pass its gates before exposing an
API.

Plotting and output scope
-------------------------

Straight-axis mirror examples write mirror-native ``mout_*.nc`` files and
render horizontal 3D, coil, cap-to-cap field-line, ``|B|``, pressure,
cross-section, and residual figures. The same figures can be regenerated with
``vmec --plot mout_*.nc``. The data include geometry, the stream function,
Cartesian magnetic field, isotropic pressure, interface residuals, solver
history, normalized variational, staggered-weak, and pointwise-force residuals,
normalized ``div(B)``, and optional coil curves. The
variational residual defines ``ftol``. The staggered weak residual independently
assembles the first variation on the energy quadrature and is checked on the
same constrained solver variables. The pointwise force reconstructs
``J x B - grad(p)`` on the full mesh. It does not define nonlinear ``ftol``,
but its magnitude and refinement are independent promotion gates. Its total,
near-axis, first-radial-row, bulk, and end-collar norms are reported
separately. ``div(B)`` checks the field representation.
Straight-axis mirror data are never encoded as a toroidal WOUT file.

The saved ``|B|`` array is reconstructed from the same radial Gauss cells used
by the magnetic-energy functional. Cartesian field samples remain separate for
field-line direction. Plots resample the uniformly spaced poloidal data with
their resolved Fourier modes, so low-order ellipses are not displayed as
polygons.

The compact six-point isotropic data are recorded in
``benchmarks/mirror_free_boundary_axisymmetric.json`` with
``promotion_status=supported_through_beta_0.10_research_through_beta_0.50``.
The nested-cut GPU matrix contains independent radial, axial,
exterior-angular, exterior-order, and combined refinements. At 10%, the fine
all-volume/core forces are ``1.44e-2``/``1.62e-3`` and every independent fine
force is below ``2.47e-2``. At 25%, one independent force is ``5.70e-2`` and
fails the ``5e-2`` gate. The 50% fine force is ``6.69e-2`` and its
medium-to-fine center-field change is ``1.02%``. Small variational residuals
do not override those independent failures.

Fixed-boundary 3D solver
------------------------

The fixed-boundary host solver jointly advances geometry and a gauge-free
stream function for helical boundaries with finite axial current. Its Newton
preconditioner combines radial, Fourier-poloidal, and CGL-axial model
stiffness. The eliminated weighted-mean stream-function node is handled by a
symmetric lift and projection, so the reduced preconditioner remains positive
definite.

The formal full-physics test refines ``(ns,nxi)`` through ``(5,5)``, ``(7,7)``,
and ``(9,9)`` at ``mpol=1`` and requires component-wise residuals below
``1e-12``. Radial magnetic energy uses two-point Gauss integration. This is
essential: the former midpoint rule admitted an alternating lambda hourglass
mode and its published refinement data were rejected. The corrected lambda
and pitch profiles are smooth, and a dedicated regression test assigns the
alternating mode finite energy.

Two manufactured pointwise-force fixtures isolate the reconstruction from the
nonlinear solve. A cylindrical finite-beta state with analytic radial
``B_z(s)`` and pressure balance converges at second order when ``ns`` doubles.
A theta-dependent self-similar tube carrying an exactly uniform Cartesian
field gives normalized Lorentz force below ``1e-12``. These gates show that
radial differentiation and nonaxisymmetric coordinates work independently;
they do not promote shaped solved states whose pointwise force remains large.
The first corrected rotating-ellipse audit also exposed a missing axis
condition: the old state varied ``|B|`` by 9--20% over theta at ``s=0`` even
though those samples are one physical point. That freedom has been removed;
the canonical fixed-open records have since been regenerated with the
regular axis, while older shaped values remain invalid.

Historical shaped records with inconsistent ``mpol`` and ``ntheta`` semantics
were removed from the compact benchmark rather than labeled as current
evidence. Schema v4 retains only results generated under
``ntheta = 2*mpol + 1`` and keeps failed strong-force studies explicitly.

Independent nonaxisymmetric analytic fixtures
----------------------------------------------

``vmec_jax.mirror.analytic`` contains validation data that never call an
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

``StraightFieldLineMirror`` implements the Agren-Savenko paraxial scalar
potential, on-axis field, Clebsch labels, straight nonparallel field lines,
and analytic elliptical sections. Its tests verify curl-free field, the
expected order-``(a/c)^2`` solenoidal and field-line truncation errors, axial
flux conservation, and

.. math::

   B_0(z)=\frac{B_0(0)}{1-z^2/c^2}, \qquad
   \mathcal E(z)=\frac{1+|z/c|}{1-|z/c|}.

Both fixtures require a thin tube and ``|z|<c``. The long-thin ordering treats
``beta`` and ``lambda=(a/L)^2`` as simultaneous small parameters;
``B/B_vac=sqrt(1-beta)`` is an asymptotic pressure-balance reference, not a
finite-beta solution or ellipticity prediction.

Native spline basis status
--------------------------

``vmec_jax.mirror.splines.CubicBSplineBasis`` supplies the longitudinal
basis for the open solver. It uses clamped knots, exact endpoint values,
four-point Gauss-Legendre quadrature on every nonzero span, and exact Boehm
knot insertion. Evaluation and two derivatives are JAX operations. Tests
match SciPy, reproduce cubics, preserve curves under insertion, and verify
JVP/VJP actions.

``SplineMirrorState`` and ``SplineMirrorBoundary`` store geometry and stream
function coefficients rather than sampled values. ``SplineMirrorDiscretization``
evaluates them on endpoint-augmented Gauss nodes before calling the shared
geometry and energy kernels, and applies side/end constraints plus the lambda
gauge in coefficient space. A quadratic flared tube uses 9 coefficients and 26
evaluation nodes versus 41 Chebyshev nodes; volume agrees to roundoff, total
energy agrees to ``5.0e-13`` relative, and an energy directional derivative
agrees with finite differences to ``1.9e-8`` relative.

The public ``solve_fixed_boundary_cli`` minimizes the scalar-pressure energy
directly in the active spline coefficients. It fixes the side and end
coefficients, eliminates the weighted stream-function gauge, and uses the
shared host L-BFGS plus residual-Newton policy. The independent staggered first
variation is assembled on the quadrature grid and pulled back through the
spline evaluation matrix rather than reused from autodiff.

For an ``ns=5`` finite-pressure, finite-current flared tube, both paths converge
in 59 iterations below ``ftol=1e-12``. Seven spline coefficients replace nine
Chebyshev axial nodes and reduce active variables from 45 to 31. Relative
differences are ``5.1e-7`` in energy, ``5.9e-6`` in volume, and ``3.2e-4`` in
center radius.

Independent solves at ``(ns,nxi)=(5,9),(7,13),(9,17)`` pair 7, 9, and 11
spline coefficients with the nodal grids. Energy error decreases
``5.12e-7 -> 1.46e-7 -> 5.60e-8``, volume error decreases
``6.50e-6 -> 2.24e-6 -> 9.85e-7``, and sampled radius RMS error decreases
``2.40e-4 -> 6.49e-5 -> 2.94e-5``. On the finest grid the Cartesian field and
``|B|`` differ by ``7.02e-4`` and ``3.05e-4`` RMS. A relative coefficient
test is deliberately not applied to the near-zero gauge stream function; the
physical field is the invariant comparison. Splines use roughly half the
active radius variables and take 2.70, 3.32, and 4.14 seconds versus 5.77,
9.29, and 12.15 seconds for the retired CGL migration solves. All variational and
independent weak residuals remain below ``1.3e-15``. Compact evidence is in
the ``axisymmetric_refinement`` section of
``benchmarks/mirror_fixed_boundary.json``.

Open fixed-boundary systems use exact JAX Hessian-vector products and
matrix-free Newton at every size. Radius-only systems retain the inexpensive
radial/Fourier/axial tensor inverse. Finite-current nonaxisymmetric systems
freeze a sparse local Hessian at the start of Newton: columns are evaluated in
batches of at most 32, only same-field-channel terms with neighboring radial
rows and axial coefficient distance at most four are stored, and all poloidal
coupling is retained. Sparse LU is reused for every Newton step; no dense
Hessian is stored.

On the 591-variable analytic-seeded SFLM, the tensor baseline takes 2,000
Krylov iterations and ends at true relative residual ``7.83e-2`` in 4.53
seconds. The frozen local factor reaches ``9.18e-11`` in 660 iterations and
4.14 seconds, with the same final energy and strong force. An isolated
current-main SOLVAX right-preconditioned FGMRES trial follows the same
iteration curve, so the host CLI remains on SciPy GMRES. The same sparse builder is used by
forward tangent and reverse adjoint systems.

On the flared finite-beta case, knot refinement from 5 to 11 coefficients
reduces relative energy error against an ``nxi=17`` Chebyshev oracle from
``1.09e-6`` to ``5.14e-8`` and volume error from ``1.19e-5`` to ``2.18e-6``.
Both errors decrease at every refinement and all coefficient solves retain
variational and staggered residuals below ``9e-15``.

Nonaxisymmetric spline evidence
-------------------------------

Changing a prescribed spline boundary uses
``SplineMirrorDiscretization.transfer_boundary``. It rescales every nested
surface at spline collocation nodes before projection, instead of replacing
only the LCFS and risking crossed surfaces. The optimizer also rejects any
trial with a changed Jacobian sign, matching the regular VMEC-JAX merit policy.

The old zero-stream continuation produced a nonconvergent strong-force floor
for both shaped cases. Those runs remain historical negative evidence and are
not the default. Physical supplied-field initialization changes the result.
For the 90-degree rotating ellipse, combined refinement
``(ns,mpol,elements)=(5,4,4),(7,6,6),(9,8,8)`` reduces all-volume strong force
``6.39e-2 -> 2.41e-2 -> 9.76e-3``. The corresponding SFLM sequence
``(7,6,6),(9,8,8),(11,10,10)`` reduces it
``4.18e-2 -> 2.11e-2 -> 1.09e-2``. Zero-limit fits have observed orders
``2.69`` and ``2.63``; both are consistent with zero. Variational and
independent weak residuals remain near roundoff, Jacobians stay positive, and
the bulk strong-force gate is below ``2e-2``.

``initialize_from_cartesian_field`` now keeps a supplied spline geometry fixed,
infers :math:`\Psi'(s)` from the surface-averaged axial flux, and obtains the
nonzero poloidal stream-function modes from the remaining contravariant field.
It accepts either Cartesian field samples or a point callable and performs no
coil construction or Biot--Savart integration. On the independent
Agren--Savenko field at ``(ns,mpol,elements)=(9,8,8)``, field reconstruction
error is ``3.42e-4``, field tangency error is ``1.33e-4``, and the all-volume
strong-force norm is ``5.12e-3``. The inferred flux differs from the analytic
``B0*a^2/2 = 4.5e-4`` by less than ``6.3e-5`` relative. A JVP test confirms
that field-amplitude derivatives pass through the projection.

The parser-free root example runs both fixtures through five coefficient-space
continuation stages and writes MOUT plus horizontal 3-D, cross-section,
``|B|``, residual, symmetry, and analytic-direction figures::

   python examples/mirror_fixed_boundary_nonaxisymmetric.py

At its compact demonstration resolution, the rotating ellipse retains central
all-volume strong force above the finest benchmark value; the example is a
fast workflow demonstration, while the compact benchmark carries promotion
evidence. The SFLM continuation
now reprojects the analytic field at each shape stage and infers
``Psi'(s)`` instead of using a prescribed scalar. Its final variational and
staggered residuals are near roundoff, normalized ``div(B)`` is below
``8.0e-15``, and mean field-direction cosine exceeds ``0.9999996``. The final
compact all-volume strong-force norm is ``3.52e-2``; the refined benchmark,
not this compact run, establishes convergence.
The figures expose both residual histories and show the actual solved nested
surfaces and cap-to-cap field lines, not the analytic target alone.

Coefficient fixed-boundary gradients
------------------------------------

``spline_fixed_boundary_adjoint`` differentiates a scalar diagnostic through
the converged coefficient residual. Boundary spline coefficients, flux,
conserved mass, and current remain differentiable. The transpose Hessian
action uses exact JAX reverse AD and the nonlinear iteration history is never
differentiated or stored. The root example differentiates rotating-ellipse
volume with respect to a native boundary coefficient and checks it against two
fully reconverged equilibria::

   python examples/mirror_fixed_boundary_nonaxisymmetric.py

On the fine grid the rotating-ellipse and SFLM volume adjoints agree with
centered differences to ``1.7e-10`` and ``2.0e-10`` relative. Their transpose
linear residuals are below ``7.4e-10``. These derivatives now inherit promoted
fixed-open primal states.

``spline_fixed_boundary_tangent`` solves the complementary forward system
``F_u du = -F_p dp`` with exact residual JVPs and the same preconditioner. On a
nonaxisymmetric finite-current ``solve_lambda=True`` case, both radius and
stream-function tangents agree with two fully reconverged centered differences
within ``2e-4`` in relative state norm, with linear residual below ``1e-8``.
This establishes both open-spline derivative directions. The deferred
closed hybrid has no primal or derivative API.

The former CGL fixed solve, custom VJP, and nodal adjoint have been removed.
Public fixed-boundary inputs are
``SplineMirrorBoundary``, ``SplineMirrorState``, and
``SplineMirrorDiscretization``; public ``solve_fixed_boundary_cli`` returns a
``SplineMirrorSolveResult``. CGL values remain available for quadrature and
evaluated-state parity tests, not as a second production state.

Axisymmetric free-boundary implicit gradients
---------------------------------------------

``free_boundary_adjoint`` currently differentiates the research axisymmetric
exterior equilibrium with respect to a differentiable external-field callable,
axial flux, conserved mass profile, and axial current. The physical fixed point
contains the lateral LCFS and plasma-interior radii. The exterior Neumann BIE
eliminates vacuum unknowns, so its exact reverse-AD field and shape responses
enter the interface-stress rows directly. The transpose solve reuses the
separable primal plasma preconditioner and does not assemble a dense Jacobian
or retain nonlinear iterations.

The validation uses a differentiable curl-free paraxial mirror field and
checks both its strength/axial-curvature controls and a finite-pressure mass
direction against fully reconverged equilibria. Pressure changes through the
same conserved-mass and solved-volume relation as the primal. End-cut radii
and any central-pressure calibration target remain fixed. Coil-design
derivatives belong to the ESSOS integration layer; vmec_jax differentiates
only the supplied field object.

The adjoint consumes the primal coefficient residual, coefficient packing,
and block preconditioner. It remains a research API until the
coefficient-native free-boundary primal lane passes its memory, strong-force,
and refinement gates. The
nonaxisymmetric free-boundary derivative is deliberately unavailable because
local Fourier-mode refinement failed; it will not be presented as a supported
gradient.

``device=None`` uses the shared measured device policy. On the office host,
the corrected ``15x15`` case took 35.2 seconds on CPU and 44.2 seconds on one
RTX A4000. Energy and force diagnostics agree to numerical precision. Explicit
``device=`` arguments and JAX platform environment pins are always honored.

The host CLI remains the forward-performance reference. Fixed-boundary
derivatives solve the linearized converged coefficient residual and never
retain or differentiate the nonlinear iteration history. The free-boundary
adjoint still reconstructs the former nodal residual and is not a supported
gradient until it is migrated to the primal coefficient problem.

Release evidence
----------------

Coverage must run without ``--source=vmec_jax.mirror``; that coverage option
pre-imports the package and can trigger a duplicate NumPy extension import on
macOS before pytest collection. The equivalent report restriction is applied
after execution::

   coverage erase
   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 coverage run -m pytest \
       tests/mirror -m "not full" -q
   RUN_FULL=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 coverage run --append \
       -m pytest tests/mirror/test_implicit.py -q
   coverage report --include="*/vmec_jax/mirror/*" --fail-under=95

The release audit is rerun from the final branch rather than copied from an
intermediate repository snapshot. Generated MOUT files and figures remain
ignored. Four compact JSON files retain numerical evidence for fixed open,
free axisymmetric, deferred free nonaxisymmetric, and the negative closed
hybrid disposition; repository-shape and promotion gates are maintained in
``plan.md``.

.. image:: _static/figures/mirror_fixed_boundary_3d.png
   :alt: Fixed-boundary axisymmetric convergence, nonaxisymmetric force gates, and matrix-free solver evidence
   :width: 100%

Beta scan example
-----------------

From a developer checkout, run:

.. code-block:: bash

   python examples/mirror_free_boundary_beta_scan.py

The script has editable inputs at its top and no command-line parser. It
solves every beta point from 0 through 50% and writes CSV plus three reviewed
figures under ``results/mirror_free_boundary_beta_scan/``. The figures include horizontal
``z`` geometry, LCFS displacement, on-axis and LCFS ``|B|``, pressure balance,
coils, cap-to-cap field lines, field arrows, and coupled residual histories.
Generated results are ignored by git. CSV rows include closed-surface
compatibility and BIE condition number for every beta point. Values through
10% exercise the supported lane; 25% and 50% are research continuation points
and must not be presented as promoted merely because the nonlinear solve ends.
The CSV ``supported_lane`` column and plot labels carry that distinction.

Set ``SAVE_RESTARTS = True`` to write one compressed ``.npz`` hot-start per
beta point. :func:`vmec_jax.mirror.output.load_free_boundary_restart` checks its
schema and coefficient shapes before returning the boundary, plasma state, and
calibrated mass scale. Schema 2 migration requires the original nodal grid
explicitly; schema 3 never guesses it. The BIE potential is recomputed because the moving
boundary changes at every continuation point.
Set ``RESTART_FROM`` and trim ``BETAS`` to resume only the unfinished suffix;
the original beta-zero boundary remains the pressure-profile reference.

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
* error against the paraxial estimate ``B/B_vac = sqrt(1-beta)``.

At the fine combined grid ``(ns,nxi,ntheta_panel)=(9,17,16)``, the default 10%
request reaches 10% central beta and 3.39% volume beta. The center radius
expands by 1.21%, while the central field falls by 4.38%. Thus field depression
is the more sensitive validation observable for this zero-edge-pressure
profile.

On that grid, the 50% research point reaches center radius ``0.272554 m``,
field ratio ``0.762687``, and volume beta ``0.216984`` with nonlinear residual
``8.31e-13``. The paraxial small-beta estimate is intentionally shown but is
not an accuracy reference at 50%; the failed strong-force and refinement gates
control its status.

The finite-beta mirror trend follows the WHAM/Pleiades discussion in Frank et
al., `Confinement performance predictions for a high field axisymmetric tandem
mirror <https://doi.org/10.1017/S002237782510055X>`_. A checked Pleiades
Green-function reference at upstream commit ``0161abb3`` gives a 10% field
ratio of 0.952754 on a 51 by 101 grid. The production mixed-truncation
``vmec_jax`` solve gives 0.956197, a 0.36% relative difference. Boundary
independent-reference curves remain promotion gates rather than being replaced
by this scalar comparison.
That study reports robust Pleiades equilibria for ``beta < 1`` and the expected
outward flux-surface expansion and diamagnetic field depression. Extending the
numerical gate to 50% therefore probes a scientifically relevant nonlinear
regime, but it is an equilibrium benchmark only: it does not establish flute,
firehose, mirror-mode, or kinetic stability.

MGRID and vectorized ``xyz -> B`` callables share one external-field adapter.
MGRID interpolation tests remain in vmec_jax; filament sampling and
Biot-Savart parity tests live with ESSOS.

The free-space model has no artificial outer cylinder. Exterior trace order,
panel refinement, closed-surface compatibility, and memory cost remain
promotion gates.

Open-exterior foundation
------------------------

:func:`vmec_jax.mirror.build_closed_mirror_surface` closes a star-shaped
lateral LCFS with disks at both fixed-flux cuts. It stores outward ``n dA``
directly, including quadrature weights, so the disk center is regular and no
unit-normal division enters geometric identities. For axisymmetric equilibrium
grids, which intentionally store one theta node, the adapter supplies an
independent Cartesian angular quadrature. ``axisymmetric_ntheta`` controls its
resolution without adding redundant theta unknowns to the equilibrium solve.
Polar quadrature repeats each cap center and the cap rims coincide with lateral
end rings. ``ClosedMirrorSurface`` therefore also provides a unique
collocation grid and an explicit map that expands continuous collocation values
back to all quadrature nodes; repeated geometry never becomes duplicate BIE
unknowns.
The same unique nodes define a watertight outward-oriented triangular panel
mesh. Every undirected edge belongs to exactly two panels, no panel is
degenerate, and cylinder area/volume converge at the expected second order as
the inscribed angular polygon is refined. This topology is the input for local
Duffy quadrature at singular panels.
The local Duffy primitive maps a vertex-singular triangle to a regular unit
square, interpolates density linearly, and is differentiable in panel geometry
and density. On a right triangle, orders ``2,4,8,16`` converge monotonically;
order 16 matches the analytic constant-density single-layer integral within
``1.4e-14`` and the linear ``x+y`` density gives exactly half that value.
For axisymmetric M6 data, angular panel nodes remain fully resolved as sources
but one target is evaluated per rotational orbit. The density dimension is
``nxi + 2(ns-1)``. At 57 unknowns and 1,762 panel vertices this representative-
target Jacobian takes 2.75 seconds instead of 67.5 seconds, with unchanged
3.27% ``u=z`` boundary error and condition number 19.1. The reduction is exact
for ring-constant values and does not lower angular integration resolution.
Power grading in the cap radius resolves the edge density without changing
closed-surface area or volume. At 45 reduced unknowns, ``u=z`` recovery improves
from 4.39% ungraded to 0.222%, 0.0589%, and 0.0264% at grades 2, 2.5, and 3;
grade 3.5 gives 0.0216% with condition number 17.0. The differentiable
exterior solve reports net-flux compatibility, condition number, and its full
equation residual. The decaying exterior equation is
``S(q) + K(u-u_target) + u = 0``. It has no constant gauge freedom, and its
off-surface representation has the opposite sign. A zero-flux dipole MMS
closes this equation to ``3e-14`` with condition number below 5. Boundary
trace error decreases ``14.9% -> 5.44%`` over the regular two-grid test, while
the exterior field-gradient error at ``z=2`` decreases to 1.12%.
Reconstructing the field directly on the side from solved normal data and the
CGL derivative of boundary potential exposes the remaining coupling blocker:
the dipole lateral-field error decreases ``48.4% -> 14.5% -> 5.28% -> 3.08%``
over four meshes; the finest solve takes 5.7 seconds and has condition number
4.02. Removing endpoint bands does not change the rate. Spectral filtering,
off-surface extrapolation, and two-grid Richardson correction were measured
and rejected because they increase the error. Linear density interpolation on
side triangles are therefore the leading candidate limiter. The finest-grid
result is accurate enough for guarded M6 coupling and is the root beta-scan
example default. The option below isolates density order before any
library-wide default changes.

An opt-in spectral side-density rule now evaluates lateral Dirichlet and
Neumann data with global Fourier-Chebyshev interpolation while retaining the
same linear panel geometry and cap density. It reproduces resolved
Fourier-Chebyshev functions to ``2e-13`` and improves the medium dipole
boundary-potential error from 5.44% to 1.19% and far-field gradient error from
1.12% to 0.72%. Set ``EXTERIOR_SPECTRAL_SIDE_DENSITY = True`` in the beta-scan
example to exercise it. The default is false because the coupled 3D study
below shows that density order alone is insufficient.

An experimental curved-side and high-order-cap variant was removed after its
bounded nonaxisymmetric endpoint run failed to complete in 690 seconds. Its
medium manufactured improvement did not justify roughly 400 lines of extra
geometry, interpolation, and differentiation code. The retained production
path uses linear Duffy panels, with optional Fourier-Chebyshev interpolation
of side density. Cap disks only close the end cuts for the Green identity and
use the same linear panel rule.

Source ownership is kept narrow: ``exterior.py`` builds geometry and reduction
maps, ``exterior_mesh.py`` owns side-panel topology and Duffy assembly,
including its density interpolation, and ``exterior_bie.py`` owns Neumann
solves. These numerical kernels remain in
their owning modules rather than the flattened public namespace.
``solve_axisymmetric_exterior_vacuum`` now owns the complete M6 adapter:
it closes the moving boundary, continues the plasma field through both end
cuts, cancels the supplied external normal field on the side, solves the exterior
Neumann problem, and returns the lateral total-field trace. Tangency and a
full shape JVP pass on the coupled adapter, so the remaining gate is nonlinear
equilibrium behavior rather than a missing differentiation path.
The adapter is the sole vacuum model in the coupled axisymmetric free-boundary
and beta-continuation drivers. On the coarse
``(ns,nxi,ntheta_panel)=(5,7,8)`` two-coil gate, beta 0 and 10% both converge
in seven nonlinear evaluations. Maximum residuals are ``7.93e-16`` and
``2.95e-15``; vacuum tangency is below ``6.3e-17`` and normalized stress below
``1.3e-15``. The center radius increases from 0.252576 m to 0.255603 m. This
proves the unbounded model is an actual finite-beta equilibrium path, while
the four-grid dipole study above still limits its quantitative promotion.
Restart files contain only the plasma state, boundary, and pressure scale; the
BIE potential is solved into ``result.vacuum_field.neumann_result`` rather than
hot-started.
The exterior solve applies the standard Neumann compatibility projection only
on the artificial end caps, leaving every lateral LCFS datum unchanged. The
corrected compatibility must close near roundoff and the condition number must
remain below ``1e8``. ``raw_compatibility_error`` reports the relative cap
correction before projection; it must decrease under refinement and fall below
``1e-6`` on the finest promotion grid. Neither quantity is conflated with the
``1e-12`` force and interface-stress convergence contract.

The beta-zero exterior resolution study at ``(ns,nxi,ntheta_panel)`` equal to
``(5,7,8)``, ``(7,13,12)``, and ``(9,17,16)`` gives center radii
``0.2525753, 0.2531506, 0.2531155`` m and axis fields
``0.0840027, 0.0835434, 0.0835623`` T. The last two grids agree within
``1.39e-4`` and ``2.26e-4`` relative. Raw flux compatibility improves
``1.60e-8 -> 1.02e-9 -> 5.92e-10`` while every force solve remains below
``5.8e-15``.

The coefficient solver uses a dense ``jacfwd`` only through 32 unknowns. Larger
systems expose exact repeated JVP/VJP actions through a SciPy ``LinearOperator``;
no identity matrix or coupled Jacobian is materialized. A bounded trust-region
stage globalizes the solve and the fixed-solver spline preconditioner supplies
a Newton--GMRES polish. In the T6b CPU audit, a 48-unknown beta-zero case reached
``2.20e-14`` in 30.6 seconds and 44 nonlinear evaluations at 2.08 GiB peak RSS.
Without the polish, capped LSMR stalled at ``4.75e-6`` after 200 evaluations.
Cached ``jax.linearize`` retained more than 3 GiB and was slower than repeated
actions on the small A/B fixture, so repeated actions are the production path.
The schema-5 nested-cut matrix below was regenerated with coefficient boundary
work and the accepted strong-force reconstruction.

The office RTX A4000 study continues every grid through 50% beta. On the fine
grid, beta 50% gives center radius ``0.272554 m``, axis field ``0.063578 T``,
volume beta ``0.216984``, and all-volume/core force
``6.69e-2``/``1.50e-2``. Medium-to-fine relative changes are ``0.137%`` in
radius and ``1.02%`` in center field. The point remains research-only. The
same matrix promotes 10%, where all independent force and observable gates
pass. Higher-order panels and lower runtime remain later optimization work.
Tests require exact cylinder area and volume, zero integrated normal, the full
tensor divergence theorem on a theta-shaped flared tube, cap/side ring
continuity, and a JAX shape derivative. The reduced decaying-exterior solve is
tested against an analytic dipole at boundary and off-surface targets, including
refinement, spectral side density, compatibility, conditioning, and shape JVPs.
Generic interior and full-node virtual-casing adapters are intentionally left
to virtual-casing-jax; vmec_jax retains only the operator used by mirror
equilibria. Broader shaped and near-singular references and higher-order side
panels remain promotion work.

Nonaxisymmetric free-boundary mirrors are explicitly deferred. A historical
three-grid study reached roundoff nonlinear residuals, but local ``m=1``
observables changed by 73--81% between the first two grids. Runtime grew from
293 to 2,995 seconds and host memory from 2.74 to 7.35 GiB on an RTX A4000.
Those states also predate the corrected magnetic-axis regularity map. The
unsupported theta-dependent exterior and diagnostics have therefore been
removed rather than presented as an equilibrium model. Compact negative
evidence remains in
``benchmarks/mirror_free_boundary_nonaxisymmetric.json``. Fixed-boundary
nonaxisymmetric mirrors remain supported; free-boundary promotion requires a
structured exterior Jacobian and a new three-grid local-mode study.

``solve_beta_scan_cli`` is the axisymmetric coefficient-native hot-start driver.

Two cheaper boundary-limit approximations were tested and rejected. Inward or
outward offset collocation produced density-system condition numbers from
``1e6`` to ``1e19`` and did not converge reliably. Replacing each singular
single-layer panel by an equal-area disk was stable but only algebraic: the
hardest linear harmonic retained 8.3% boundary error at about 1,900 unknowns.
The implementation therefore follows local singular quadrature rather than
labeling either approximation research-grade. Relevant numerical foundations
are Duffy's `vertex-singularity transform
<https://doi.org/10.1137/0719090>`_ and the distinction between smooth-surface
QBX error control and explicit corner treatment discussed by
`af Klinteberg and Tornberg <https://arxiv.org/abs/1603.08366>`_ and
`Helsing and Ojala <https://doi.org/10.1016/j.jcp.2008.06.022>`_.
