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
* a differentiable closed-surface adapter for the lateral LCFS and both end
  disks, with nonsingular off-surface Laplace kernels from
  ``virtual_casing_jax``, and
* component-wise nonlinear convergence checks at a requested ``ftol=1e-12``.

The axisymmetric free-boundary path is a research capability. A first formal
resolution study, free-side initial-condition test, anisotropic closure study,
and restart-file path are complete. Higher-resolution vacuum tangency,
open-exterior closure, independent boundary references, and output gates in
``plan.md`` remain. Nonaxisymmetric free-boundary mirrors and the toroidal
stellarator-mirror hybrid are later milestones and must not be inferred from
the axisymmetric result.

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

This is not yet a converged M8 square-axis equilibrium. The circular-axis
member of the same shaped family converges component-wise to ``1e-12`` in
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
not a vmec_jax implementation discrepancy. The next M8 gate is therefore a
coil-informed, curvature-bounded target family followed by the same VMEC2000
parity test; no root example will label this geometry solved before then.

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
solves every beta point and writes CSV plus three reviewed figures under
``results/mirror_free_boundary_beta_scan/``. The figures include horizontal
``z`` geometry, LCFS displacement, on-axis and LCFS ``|B|``, pressure balance,
coils, cap-to-cap field lines, field arrows, and coupled residual histories.
Generated results are ignored by git.
Set ``VACUUM_BACKEND = "exterior"`` in the same input block to replace the
finite outer cylinder with the differentiable free-space BIE solve. The CSV
then includes closed-surface compatibility and BIE condition number for every
beta point; the default remains ``"annulus"`` until the exterior memory and
finite-beta resolution gates above close.

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

The finite-beta mirror trend follows the WHAM/Pleiades discussion in Frank et
al., `Confinement performance predictions for a high field axisymmetric tandem
mirror <https://doi.org/10.1017/S002237782510055X>`_. A checked Pleiades
Green-function reference at upstream commit ``0161abb3`` gives a 10% field
ratio of 0.952754 on a 51 by 101 grid. The production mixed-truncation
``vmec_jax`` solve gives 0.952176, a 0.061% relative difference. Boundary and
anisotropic independent-reference curves
remain promotion gates rather than being replaced by this scalar comparison.

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
true exterior Dirichlet-to-Neumann or boundary-integral operator remains an M5
promotion gate.

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
side triangles is therefore the limiter. The finest-grid result is accurate
enough for a guarded first M6 coupling study, but higher-order side density is
still required before replacing the annulus backend by default.
Source ownership is kept narrow: ``exterior.py`` builds geometry and reduction
maps, ``exterior_mesh.py`` owns panel topology and Duffy quadrature, and
``exterior_bie.py`` owns layer evaluation and Neumann solves. Public functions
remain exported from ``vmec_jax.mirror``.
``solve_axisymmetric_exterior_vacuum`` now owns the complete M6 adapter:
it closes the moving boundary, continues the plasma field through both end
cuts, cancels direct-coil normal field on the side, solves the exterior
Neumann problem, and returns the lateral total-field trace. Tangency and a
full shape JVP pass on the coupled adapter, so the remaining gate is nonlinear
equilibrium behavior rather than a missing differentiation path.
The adapter is also available as ``vacuum_backend="exterior"`` in the coupled
axisymmetric free-boundary and beta-continuation drivers. On the bounded
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

The beta-10 third grid now also converges on one office RTX A4000. Its center
radius is 0.2561004 m and axis field is 0.0797034 T, agreeing with ``ns=7``
within ``3.0e-5`` and ``3.65e-4`` relative. Beta 0 and 10% each require eight
nonlinear evaluations with residuals ``3.43e-15`` and ``6.92e-15``. The full
two-point GPU run takes 119.5 seconds and 1.99 GB host RSS, compared with
118.8 seconds and 5.48 GB for beta zero alone on the local CPU. A ``full``
three-grid regression now preserves the ``5e-4`` observable and ``2e-9``
compatibility gates. Accuracy promotion is closed for this two-coil case;
higher-order panels and lower CPU memory remain M5/M10 work.
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
separately tested interior and decaying-exterior reduced Neumann solves. It
still needs shaped-boundary and
finite-beta coil-data studies, broader near-singular targets, resolution/error
targets tight enough for interface coupling, and replacement of the finite
outer cylinder in the coupled free-boundary solve.

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
