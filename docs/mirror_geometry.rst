Mirror geometry research path
=============================

``vmec_jax.mirror`` is the open-field-line equilibrium backend under active
development. It uses coordinates ``(s, theta, xi)`` with a nonperiodic axial
coordinate and fixed-flux end cuts. It does not reinterpret a straight mirror
as a periodic torus.

Current capability
------------------

The branch currently includes:

* axisymmetric and nonaxisymmetric fixed-boundary finite-current solves,
* isotropic and ANIMEC-style anisotropic pressure energies and independent
  tensor-force diagnostics,
* a variational scalar-potential vacuum annulus with direct JAX Biot-Savart
  coils or the shared ESSOS/MAKEGRID-compatible ``MgridField``,
* coupled axisymmetric isotropic and anisotropic free-boundary beta
  continuation with compressed restart files,
* a free-space boundary-integral vacuum backend on the lateral LCFS and both
  end disks, alongside the finite-annulus backend,
* a genuine theta-dependent exterior free-boundary solve with finite axial
  current and nonaxisymmetric coils, and
* component-wise nonlinear convergence checks at a requested ``ftol=1e-12``.

The axisymmetric free-boundary path is a research capability with completed
annulus and unbounded-exterior resolution studies through 50% requested beta.
Higher-order exterior traces and additional independent boundary references
remain promotion gates. The nonaxisymmetric path also converges through 50%,
but its point observables are not yet monotone under spatial refinement, so it
remains a development capability. The toroidal stellarator-mirror hybrid now
has a converged coil-informed fixed-boundary path and independent VMEC2000
restart parity at ``1e-8``; stricter preconditioning and free-boundary gates
remain open.

Toroidal hybrid foundation
--------------------------

``sample_stellarator_mirror_hybrid`` builds a closed square-like toroidal axis
in the horizontal plane. Four long superellipse sides are the mirror sections;
localized rotating elliptical cross-sections provide stellarator shaping at
the four corners. ``stellarator_mirror_hybrid_input`` projects this one
real-space target into ordinary ``RBC/ZBS`` arrays and returns a standard
fixed-boundary :class:`vmec_jax.VmecInput`; there is no native spline state or
second equilibrium solver.

The axis side distance from the ideal square is below 2 mm in the designated
side regions, four corner regions are detected, side ellipses remain aligned,
and corner orientation spans more than 0.2 radians. Fourier maximum component
error decreases from 13.0 mm at ``(mpol,ntor)=(4,8)`` to 0.912 mm at ``(6,16)``,
0.262 mm at ``(6,20)``, and 0.077 mm at ``(8,24)``.

The original imposed-superellipse target did not produce a converged M8
square-axis equilibrium. The circular-axis member of that shaped family
converges component-wise to ``1e-12`` in
1,870 iterations. Continuous superellipse-exponent continuation reaches
``p=4.20`` on ``ns=3`` at ``1e-8``, but the state stalls near ``1.14e-6``
when lifted to ``ns=5``; direct ``ns=5`` continuation stops near ``p=3.05``.
Linear circle-to-square continuation stops near 44% of the requested axis.

VMEC2000 reproduces the default and unshaped-square stalled residuals to the
printed digits after 5,000 iterations, including its repeated Jacobian resets.
A 10,000-iteration VMEC2000 time-step scan finds ``DELT=0.1`` best, at
``2.53e-7/1.60e-7/3.62e-7``, but it does not converge. Increasing Fourier
bandwidth does not move the continuation limit. Direct activation of the
current 2D preconditioner exceeded two minutes versus 8.2 seconds for the 1D
run and was rejected. These parity checks identify a target/basin limitation,
not a vmec_jax implementation discrepancy. This result motivated the
coil-informed, curvature-bounded target family described below.
An additional 4,096-point spectral curvature audit rules out a simple tube
curvature violation at the 44% continuation limit: minimum curvature is
``0.332 m^-1`` and the tightest curvature radius is 0.559 m, versus 0.1 m
minor radius. The exact square target does approach zero curvature on its
sides, but the solver basin is lost well before that limit. Another imposed
superellipse continuation is therefore not the next experiment; the target
must be extracted from the 16-coil vacuum flux geometry.

That coil-informed path is now available. ``planar_ellipse_coils`` constructs
arbitrary oriented elliptical coils, while ``square_mirror_coils`` builds four
ordered groups of N mirror coils. ``trace_square_coil_vacuum_axis`` follows
one closed Biot-Savart field-line turn and reports closure, planarity, field
strength, and the thin-flux-tube scale ``sqrt(B_ref/B_axis)``. For the default
16 coils, the traced axis is planar to roundoff, has straight-side error below
1.7 mm, and is represented to 0.057 mm by ``ntor=20``.

For free-boundary initialization, the trace also reports signed ``B_phi`` and
``toroidal_flux_scale``. ``coil_informed_toroidal_flux`` estimates the signed
VMEC ``PHIEDGE`` through a constant-toroidal-angle section; this avoids using
the total-field tube area when the square axis has a radial tangent component.

Using that axis and flux-conserving cross-section, the ``ns=5`` unshaped
fixed-boundary solve converges at ``1e-8`` in 62 iterations. Ten-percent
continuation steps then reach the complete rotating-corner ellipse target;
with a flat 3 kA toroidal-current profile, the final stage converges in 509
iterations and gives ``iota=-0.805...-0.807``. This closes the former 44%
Fourier-geometry blocker. The root example writes WOUT plus 3D coils, LCFS,
pitched field lines, ``|B|``, cross-sections, profiles, and force histories.
An independent VMEC2000 restart from the vmec_jax WOUT converges in 1,071
iterations with no Jacobian resets. The fixed LCFS agrees to roundoff,
magnetic energy to ``3.73e-7`` relative, LCFS ``|B|`` to ``2.52e-4`` relative
L2, and iota to ``1.09e-3`` relative L2. This accepts fixed-boundary basin and
solver parity at the current force floor; it is not machine-precision parity.
The example also writes the exact ``input.stellarator_mirror_hybrid`` deck
used for the independent run, and the compact results are recorded in
``benchmarks/mirror_hybrid_fixed_boundary.json``. Tolerance promotion remains
open: the unshaped state
reaches ``1e-9``, but neither it nor the fully shaped state reaches ``1e-10``
under the tested ``DELT`` values. A suitable block preconditioner is required
before the free-boundary 16-coil beta scan is promoted. The current
generic 2D block preconditioner was tested at step factors 0.25, 0.5, and 1.0
on the 3 kA base and increased, rather than reduced, the residual; a
hybrid-scaled block is needed rather than further tuning of that path.

The first toroidal free-boundary gate exposed and fixed two hot-start defects:
a force-converged fixed seed could return before NESTOR activation, and its
pre-vacuum best checkpoint could reject every coupled update. With signed
coil-matched ``PHIEDGE``, the unshaped toroidal-flux seed now completes a real
beta-zero NESTOR turn-on and converges at ``ftol=1e-8`` in three iterations
using ``DELT=0.002``. Finite-pressure continuation and the full 0--50% scan
remain open.

A split fixed-predictor/fixed-corrector/NESTOR continuation then converges
target beta 0.05--0.25% (achieved 0.0511--0.2555%); each released-boundary
correction takes three iterations. The 0.30% fixed predictor reaches
``1e-7``, but its ``1e-8`` corrector exhausts 5,000 iterations. Smaller beta
steps and tolerance restarts do not move that boundary. ``ntor=12`` and 16
also fail at beta zero, while 20 converges, so spline controls cannot replace
the equilibrium Fourier bandwidth. The generic 2D block reduces lambda but
worsens radial force, and VMEC2000 cold/restart runs likewise remain above
``1e-6`` after more than 4,000 iterations. Complete values are in
``benchmarks/mirror_hybrid_free_boundary.json``. The 0--50% scan remains a
blocked research target, not a supported example.

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

At ``ns=15,nxi=15,ntheta=5``, the variational force is ``2.25e-13`` and the
independently differenced all-row/axis/bulk force residuals are
``0.0430/0.107/0.00972``. The split is intentional: the first two off-axis
rows expose the still-open mode-regular axis stencil instead of contaminating
the bulk convergence measure. At ``ns=31`` (3,805 unknowns), the matrix-free
solve reaches ``6.81e-13`` in 141 seconds and bulk force falls to ``0.00628``.
The axis residual and 10,500 Krylov iterations remain promotion blockers.
Systems through 2,048 unknowns have a bounded dense reference polish; larger
systems report matrix-free convergence honestly.

``device=None`` uses the shared measured device policy. On the office host,
the corrected ``15x15`` case took 35.2 seconds on CPU and 44.2 seconds on one
RTX A4000. Energy and force diagnostics agree to numerical precision. Explicit
``device=`` arguments and JAX platform environment pins are always honored.

.. image:: _static/figures/mirror_fixed_boundary_3d.png
   :alt: Fixed-boundary helical mirror refinement, force residuals, and CPU/GPU timing
   :width: 100%

Beta scan example
-----------------

From a developer checkout, run:

.. code-block:: bash

   python examples/mirror_free_boundary_beta_scan.py

The script has editable inputs at its top and no command-line parser. It
solves every beta point from 0 through 50% and writes CSV plus three reviewed
figures under ``results/mirror_free_boundary_beta_scan_exterior/``. The figures include horizontal
``z`` geometry, LCFS displacement, on-axis and LCFS ``|B|``, pressure balance,
coils, cap-to-cap field lines, field arrows, and coupled residual histories.
Generated results are ignored by git.
The default ``VACUUM_BACKEND = "exterior"`` removes the finite outer cylinder.
Set it to ``"annulus"`` for the bounded comparison model; each backend writes
to a separate result directory. Exterior CSV rows include closed-surface
compatibility and BIE condition number for every beta point.

``PRESSURE_MODEL`` selects the mass-conserving isotropic scan, a consistent
bi-Maxwellian ANIMEC closure, or a tabulated ``p_parallel(s,B)`` sampled from
that closure. In the anisotropic lane the solved beta target is midplane
``p_perp``, the interface stress uses ``p_perp``, and convergence also requires
the firehose/mirror ellipticity indicators to remain valid.

Set ``SAVE_RESTARTS = True`` to write one compressed ``.npz`` hot-start per
beta point. :func:`vmec_jax.mirror.load_free_boundary_restart` checks its
schema and both plasma/vacuum grid shapes before returning the boundary,
plasma state, vacuum potential, and calibrated mass scale.
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

At ``(ns,nxi,nrho)=(7,13,7)``, the default 10% request reaches 10% central
beta and 3.37% volume beta. The center radius expands by 1.21%, while the
central field falls by 4.78%; the field ratio is within 0.37% relative of the
paraxial estimate. Thus field depression is the more sensitive validation
observable for this zero-edge-pressure profile.

On the unbounded exterior grid ``(ns,nxi,ntheta_panel)=(9,17,16)``, the 50%
point reaches center radius ``0.272660 m``, field ratio ``0.747645``, and
volume beta ``0.219148`` with nonlinear residual ``7.7e-15``. The paraxial
small-beta estimate is intentionally shown but is no longer an accuracy
reference at 50%; the solved nonlinear pressure balance is the governing gate.

The finite-beta mirror trend follows the WHAM/Pleiades discussion in Frank et
al., `Confinement performance predictions for a high field axisymmetric tandem
mirror <https://doi.org/10.1017/S002237782510055X>`_. A checked Pleiades
Green-function reference at upstream commit ``0161abb3`` gives a 10% field
ratio of 0.952754 on a 51 by 101 grid. The production mixed-truncation
``vmec_jax`` solve gives 0.952176, a 0.061% relative difference. Boundary and
anisotropic independent-reference curves
remain promotion gates rather than being replaced by this scalar comparison.
That study reports robust Pleiades equilibria for ``beta < 1`` and the expected
outward flux-surface expansion and diamagnetic field depression. Extending the
numerical gate to 50% therefore probes a scientifically relevant nonlinear
regime, but it is an equilibrium benchmark only: it does not establish flute,
firehose, mirror-mode, or kinetic stability.

The direct-coil and mgrid routes share one external-field adapter. A formal
full-physics test samples the two end coils onto a 49 by 97 mgrid and solves
the same beta-zero free-boundary equilibrium through both routes. Both reach
``ftol=1e-12``; their LCFS agrees within 0.5% and the annulus field within
0.8%. This validates interpolation and coupling parity, not yet the open-
exterior truncation.

The mixed vacuum truncation fixes the correction potential on the outer
cylinder, preserves zero correction flux through the axial cuts, and obtains
total-field tangency naturally on the plasma side. Finite-wall Neumann and
mixed-Dirichlet center fields approach one another as the outer cylinder is
expanded, but their remaining gap is reported as truncation uncertainty. A
free-space boundary-integral backend now removes this truncation. The annulus
remains useful for parity and inexpensive comparison tests; exterior trace
order and memory cost remain M5 promotion gates.

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
The interior Calderon identity uses
``S(q) + K(u-u_target) = 0``. Target subtraction preserves the constant
nullspace and absorbs the local solid-angle coefficient, including at cap
rims. For the independent harmonic fields ``u=x`` and ``u=z``, the worst
normalized residual falls from ``3.47e-3`` on a 154-node mesh to ``1.78e-3``
on an 862-node mesh. Raising Duffy order from 8 to 10 does not change the
error, identifying linear-panel/rim resolution as the next limiter.
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
area-weighted saddle solve reports net-flux compatibility, gauge error,
condition number, and its full equation residual. In this case compatibility
and gauge close near roundoff, while the panel-discrete equation residual is
``8.9e-9``; it is not an exterior discretization claim.
The decaying exterior equation is distinct:
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

``exterior_high_order_cap_panels=True`` pairs exact polar/star-shaped cap
geometry with Fourier interpolation in angle and a resolution-aware radial
rule: linear below seven rings, quadratic through ten rings, and local cubic
thereafter. This avoids the ill-conditioning of cubic interpolation on a
five-ring, strongly rim-graded cap. Circular disk area and orientation are
exact to roundoff, and both axisymmetric and 3D boundary-shape JVPs are finite.
On the medium dipole MMS, the paired side-and-cap method reduces the boundary
potential error from 0.786% to 0.725% and the off-surface field error from
1.056% to 0.121%, with condition number 4.89. Coarse conditions remain 3.07
axisymmetrically and 2.73 in 3D. Coupled beta scans through 50% pass at
``ftol=1e-12``; the option remains off by default pending medium/fine cost and
local-mode convergence evidence.

Source ownership is kept narrow: ``exterior.py`` builds geometry and reduction
maps, ``exterior_mesh.py`` owns side-panel topology and Duffy assembly,
``exterior_cap_panels.py`` owns exact polar cap kernels,
``exterior_interpolation.py`` owns reusable density interpolation, and
``exterior_bie.py`` owns Neumann solves. Public functions remain exported from
``vmec_jax.mirror``.
``solve_axisymmetric_exterior_vacuum`` now owns the complete M6 adapter:
it closes the moving boundary, continues the plasma field through both end
cuts, cancels direct-coil normal field on the side, solves the exterior
Neumann problem, and returns the lateral total-field trace. Tangency and a
full shape JVP pass on the coupled adapter, so the remaining gate is nonlinear
equilibrium behavior rather than a missing differentiation path.
The adapter is also available as ``vacuum_backend="exterior"`` in the coupled
axisymmetric free-boundary and beta-continuation drivers. On the coarse
``(ns,nxi,ntheta_panel)=(5,7,8)`` two-coil gate, beta 0 and 10% both converge
in seven nonlinear evaluations. Maximum residuals are ``7.93e-16`` and
``2.95e-15``; vacuum tangency is below ``6.3e-17`` and normalized stress below
``1.3e-15``. The center radius increases from 0.252576 m to 0.255603 m. This
proves the unbounded model is an actual finite-beta equilibrium path, while
the four-grid dipole study above still limits its quantitative promotion.
The legacy ``outer_radius`` argument is ignored by this backend, and restart
files retain a zero placeholder potential because the BIE potential is solved
and stored in ``result.vacuum_field.neumann_result`` rather than hot-started.
The nonlinear validity gate requires relative closed-surface flux compatibility
below ``1e-6`` and condition number below ``1e8``. Compatibility is reported
separately and must decrease under refinement; it is not conflated with the
``1e-12`` force and interface-stress convergence contract.

The beta-zero exterior resolution study at ``(ns,nxi,ntheta_panel)`` equal to
``(5,7,8)``, ``(7,13,12)``, and ``(9,17,16)`` gives center radii
``0.2525753, 0.2531506, 0.2531155`` m and axis fields
``0.0840027, 0.0835434, 0.0835623`` T. The last two grids agree within
``1.39e-4`` and ``2.26e-4`` relative. Flux compatibility improves
``1.60e-8 -> 1.02e-9 -> 5.92e-10`` while every force solve remains below
``5.8e-15``.

The 120-variable third grid exposed a CLI memory problem in monolithic forward
AD: it was terminated at 9.67 GB RSS before producing an iteration. The solver
now keeps the fast monolithic Jacobian through 80 variables and evaluates exact
forward-mode JVP columns in chunks of six above that point. The same third grid
then converges in 118.8 seconds at 5.48 GB peak RSS. This closes the first
three-grid beta-zero physics gate and identifies CPU memory as a performance
blocker; the GPU result below closes the finite-beta follow-up.

The office RTX A4000 study now continues every grid through 50% beta. On the
third grid, beta 50% gives center radius ``0.2726602 m``, axis field
``0.0624749 T``, volume beta ``0.219148``, compatibility ``2.09e-9``, and
condition 3.23. All nonlinear residuals through the scan remain below
``8.1e-15``. Medium-to-fine relative changes at 50% are ``7.4e-4`` in radius,
``4.2e-3`` in field, and ``4.7e-3`` in volume beta. The ``full`` regression
therefore preserves the ``5e-4`` low-beta gate and uses a separate ``5e-3``
high-beta gate. Higher-order panels and lower CPU memory remain M5/M10 work.
An analytic Green-gradient evaluator avoids differentiating safe-distance
branches on the symmetry axis. Duffy panel evaluation also controls near-cap
targets in the interior validation: for two circular coils outside a
radius-0.3 m, length-1.2 m cylinder, the reconstructed uniform on-axis field
error at ``z=-0.4,0,0.4`` decreases
``0.998% -> 0.573% -> 0.369%`` across the measured meshes. The first two grids
are a regular regression and require roundoff flux compatibility, condition
number below 10, zero transverse axis field, and decreasing error.

Tests require exact cylinder area and volume, zero integrated normal, the full
tensor divergence theorem on a theta-shaped flared tube, cap/side ring
continuity, and a JAX shape derivative. The off-surface double-layer and
single-layer-gradient adapters call the released ``virtual_casing_jax>=0.0.2``
kernels; a constant double layer converges to one inside and zero outside, and
the single layer reaches its far-field monopole limit. Green's representation
converges jointly under disk-radial, axial, and angular refinement for the
harmonic manufactured fields ``1``, ``x``, ``z``, and ``x^2-y^2`` while
returning zero at exterior targets.

The off-surface functions reject malformed source and target arrays. M5 now has
separately tested interior and decaying-exterior reduced Neumann solves plus an
opt-in axisymmetric nonlinear coupling. It still needs broader shaped and
near-singular references and higher-order side panels before the finite outer
cylinder can be removed as the default backend.

The first M7 nonaxisymmetric seam is also explicit. ``magnetic_field_xyz``
converts the full contravariant mirror field without an axisymmetry shortcut,
and ``plasma_coil_neumann`` assembles lateral plus graded-cap data on a
theta-dependent closed surface. A finite-current ``mpol=1`` manufactured case
matches the metric ``|B|^2`` contraction to ``5e-13``, keeps lateral ``B.n``
below ``2e-15``, and closes integrated flux within ``2e-3`` on a small grid.
The same case now solves the full-theta exterior Dirichlet trace with condition
number below 20 and equation residual below ``2e-12``. Theta/axial surface
derivatives reconstruct total lateral tangency below ``3e-15``, and a complete
boundary-shape JVP is finite and nonzero. This is a differentiable 3D vacuum
closure. ``solve_free_boundary_cli`` now inserts it into the same nonlinear
plasma/interior/interface residual used in axisymmetry; the former
``solve_axisymmetric_free_boundary_cli`` name remains an alias.

With two oppositely offset circular end coils, an ``mpol=1, ntheta=3`` tube
retains a genuine midplane ``m=1`` radius spread of 0.433 mm after solving,
rather than relaxing to replicated axisymmetry. Beta 0 and 10% converge in
9 and 7 evaluations with force below ``3.7e-15``, tangency below ``3e-17``,
stress below ``2.1e-15``, compatibility near ``1.05e-3``, and condition below
3.7. At 10%, the theta-zero center radius expands from 0.201985 m to
0.204418 m and the ``m=1`` spread remains 0.467 mm. The coarse M7 smoke uses
a separate ``2e-3`` compatibility gate; axisymmetric production remains at
``1e-6``. Nonaxisymmetric resolution convergence and independent coil/field
references remain promotion gates.
The same genuine-3D case continues through 25% and 50% without stalling. From
zero to 50%, mean midplane radius grows ``0.201794 -> 0.217968 m``, mean
central field falls ``0.082215 -> 0.071173 T``, and Fourier ``m=1`` radius
amplitude grows ``0.255 -> 0.421 mm`` while residual remains below
``3.7e-15``. This is a nonlinear robustness result, not a spatial-accuracy
promotion.
``solve_beta_scan_cli`` is the topology-independent hot-start driver and
propagates the finite-current profile through its reference and finite-beta
solves; ``solve_axisymmetric_beta_scan_cli`` remains a compatibility alias.

The first M7 resolution audit is deliberately not promoted. For
``(ns,ntheta,nxi)=(5,5,5),(7,5,7),(9,7,9)``, the beta-zero theta-zero center
radius is ``0.201934, 0.201504, 0.202207`` m, axis field is
``0.083300, 0.084187, 0.083302`` T, and sampled theta radius spread is
``0.638, 0.362, 0.582`` mm. At 10%, the corresponding values are
``0.204334, 0.204008, 0.204625`` m, ``0.079472, 0.080224, 0.079416`` T, and
``0.685, 0.391, 0.618`` mm. Every nonlinear residual is below ``1.2e-14`` and
compatibility improves from about ``1.6e-5`` to ``8e-8``, but the observables
are nonmonotonic and the last two axis fields differ by about 1%.

Measured two-point runtimes are 25.6 seconds locally, then 287.7 and 896.1
seconds on one A4000; the GPU grids use 2.74 and 3.65 GB host RSS. The finest
run is therefore evidence of both nonlinear robustness and unresolved spatial
convergence, not a validation result. M7 needs a better refinement coordinate,
higher-order exterior trace, and lower-memory Jacobian before another larger
grid is worthwhile.

Global observables are better behaved at high beta. For the beta-0/50%
endpoint study, the ``(7,5,7)`` to ``(9,7,9)`` relative changes at 50% are
``6.46e-4`` in theta-mean radius, ``1.09e-3`` in theta-mean field,
``4.45e-4`` in volume, and ``5.28e-4`` in total energy. Compatibility improves
from ``1.18e-5`` to ``2.55e-8``. The ``m=1`` amplitude changes from 0.305 to
0.131 mm, however, so only the global response is stable to 0.2%; local 3D
shape is not promoted. Inputs and complete values are stored in
``benchmarks/mirror_free_boundary_nonaxisymmetric.json``.

Spectral side density does not close the local 3D gate. At beta 50%, its
medium-to-fine changes are ``4.87e-4`` in mean radius, ``3.09e-3`` in mean
field, ``2.32e-4`` in volume, and ``5.79e-4`` in energy, but the ``m=1``
amplitude changes from 0.154 to 0.0569 mm. The fine spectral central field
also differs from the fine linear-density result by 14.1%. The endpoint pairs
take 565 and 1,830 seconds and peak at 3.66 and 5.39 GiB RSS on an A4000.
Consequently the option remains a research diagnostic; the next discretization
study must increase side-geometry and cap-density order together rather than
adding another brute-force grid.

Exact polar cap geometry with resolution-adaptive radial density interpolation
was then tested as ``exterior_high_order_cap_panels``. It improves the medium
dipole off-surface field error from 1.056% to 0.121% without ill-conditioning,
but fails the equilibrium gates. Axisymmetric compatibility is
``4.71e-6/3.35e-5`` at beta 0/50% on ``(ns,nxi)=(7,13)`` and
``5.74e-6/1.67e-5`` on ``(9,17)``, above the production ``1e-6`` limit. The
medium 3D endpoint pair costs 2,149 s and 9.26 GiB RSS, 3.8 times and 2.5
times the spectral-side pair. Its fine run exceeded 94 minutes, 11.9 GiB RSS,
and 9.25 GiB GPU memory without completing. The API remains opt-in for
quadrature research and is rejected as the production cap discretization.
``boundary_fourier_amplitudes`` now reports theta mean and peak-normalized
positive Fourier modes without the odd-grid bias of sampled peak-to-peak
values. Its analytic ``m=0,1,2`` test closes to ``5e-17``. The next audit will
use this modal diagnostic plus volume, energy, and theta-averaged fields.

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
