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
The assembled boundary-limit identity uses
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
``8.9e-9``; it is not described as a ``1e-12`` exterior discretization.

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
a tested on-surface reduced Neumann solve, but still needs shaped-boundary and
coil-data studies, near-singular field evaluation, resolution/error targets
tight enough for interface coupling, and replacement of the finite outer
cylinder in the coupled free-boundary solve.

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
