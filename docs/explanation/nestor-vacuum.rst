The NESTOR vacuum solve
=======================

Free-boundary VMEX couples the plasma iteration to Merkel's Green's-function
vacuum solve (NESTOR, J. Comp. Phys. 66, 83 (1986)), ported from VMEC2000's
``vacuum.f`` pipeline with the same activation cadence. This page explains
the exterior Neumann problem, the full-vs-incremental update split, and
which parts of the free-boundary problem are differentiated; the run recipe
is :doc:`/howto/free-boundary`.

The exterior Neumann problem
----------------------------

For ``LFREEB = T`` decks, :mod:`vmex.core.vacuum` implements Merkel's
Green's-function method. In the vacuum region the field is curl-free, so it
is written as

.. math::

   \mathbf{B}_{\mathrm{vac}} = \mathbf{B}_{\mathrm{ext}} + \nabla\Phi,
   \qquad \nabla^2 \Phi = 0,

with :math:`\mathbf{B}_{\mathrm{ext}}` the field of the external coils
(mgrid or Biot–Savart) plus the net-toroidal-current filament, and the
plasma boundary acting as a flux surface:

.. math::

   \mathbf{n}\cdot(\mathbf{B}_{\mathrm{ext}} + \nabla\Phi) = 0
   \quad \text{on } \partial\Omega.

Green's second identity turns this exterior Neumann problem into a boundary
integral equation for the surface potential,

.. math::

   \frac{\Phi(\mathbf{x}')}{2}
   = \oint_{\partial\Omega} \Bigl[
     \Phi(\mathbf{x})\,\mathbf{n}\cdot\nabla G(\mathbf{x},\mathbf{x}')
     + G(\mathbf{x},\mathbf{x}')\,
       \mathbf{n}\cdot\mathbf{B}_{\mathrm{ext}}(\mathbf{x})
     \Bigr]\, dS, \qquad
   G = \frac{1}{4\pi\,|\mathbf{x}-\mathbf{x}'|},

which, after expanding :math:`\Phi` in Fourier harmonics
:math:`\sin(mu - nv)/\cos(mu - nv)` on the boundary, becomes a dense
``mnpd2 x mnpd2`` linear system for the potential coefficients ``potvac``.
The :math:`|\mathbf{x}-\mathbf{x}'| \to 0` singularity of :math:`G` is split
off and integrated analytically (``analyt.f``, the ``cmns`` coefficient
tables); the regular remainder is tabulated on the angular grid (``greenf`` /
``fourp``). Implementation: geometry-independent tables in
:func:`~vmex.core.vacuum.vacuum_basis`, the jitted full/incremental
solves in :func:`~vmex.core.vacuum.make_vacuum_solver`, and the surface
field :math:`B_u = \mathrm{bexu} + \partial_u\Phi` (etc.) with
:math:`\mathrm{bsqvac} = |B_{\mathrm{vac}}|^2/2` in
:func:`~vmex.core.vacuum.vacuum_channels`.

Coupling cadence (``funct3d.f``)
--------------------------------

:func:`vmex.core.freeboundary.solve_free_boundary` drives the coupling
with the VMEC2000 cadence:

- the vacuum solve activates once :math:`\mathrm{fsqr}+\mathrm{fsqz} \le 10^{-3}`;
- a **full** NESTOR solve runs when ``mod(iter2 - iter1, nvacskip) == 0``,
  factoring the dense potential matrix once; cheaper incremental updates
  reuse that LU factor (VMEC2000's ``DGETRF``/``DGETRS`` split) while only
  rebuilding the analytic right-hand side, and the cadence adapts as

  .. math::

     \mathrm{nvacskip} \leftarrow \max\!\left(\mathrm{nvskip}_0,\;
     \frac{1}{\max(0.1,\; 10^{11}\,(\mathrm{fsqr}+\mathrm{fsqz}))}\right);

- the vacuum pressure enters the edge force through
  ``rbsq = (bsqvac + presf_ns) * R(edge) / hs`` at ``js = ns``, and the
  constraint reference surfaces ``rcon0, zcon0`` ramp by 0.9 per iteration.

The multigrid form of this coupling — carried vacuum state, per-stage NESTOR
rebuilds, one activation across the ladder — is described in
:doc:`iteration`.

External fields
---------------

The forward NESTOR solver consumes a :class:`~vmex.core.mgrid.MgridField`.
It may be loaded from an ``mgrid`` file (trilinear interpolation weighted by
``EXTCUR``) or built once with
:meth:`~vmex.core.mgrid.MgridField.from_cartesian_field`, which tabulates an
ESSOS/SIMSOPT Biot--Savart object or any ``xyz -> B`` callable.  The resulting
table and its current scale remain JAX-differentiable; tabulation itself does
not retain coil-geometry derivatives. For a coupled solve,
:meth:`~vmex.core.mgrid.MgridField.from_parameterized_cartesian_field` instead
tabulates ``field(coil_parameters, xyz)`` entirely in JAX, retaining exact
shape/current derivatives through interpolation and NESTOR. Direct,
interpolation-free ESSOS derivatives use the virtual-casing residual below.
VMEX carries no coil code.

On a GPU free-boundary run, the plasma iteration, mgrid interpolation, cached
vacuum arrays, and final state remain on the accelerator. The dense NESTOR
assembly/factor/solve is explicitly placed on CPU and its small boundary
inputs/outputs are bridged inside the jitted cadence loop. This follows the
VMEC++ accelerator decomposition and avoids the alternate LASYM branch seen
with accelerator dense linear algebra. An explicitly requested GPU LASYM
multigrid ladder therefore seeds only its coarsest rung on CPU, then transfers
the converged branch to all finer GPU rungs.

What is (and is not) differentiated
-----------------------------------

The NESTOR iteration above is a host-driven fixed point and is not
differentiated. For coil/current optimization,
:mod:`vmex.core.virtual_casing` instead expresses the interface conditions as
smooth objectives on a prescribed boundary. At the plasma-vacuum
interface the total exterior field
:math:`\mathbf{B}_{\mathrm{out}} = \mathbf{B}_{\mathrm{coil}} +
\mathbf{B}_{\mathrm{plasma}}` must be tangent, and pressure balance holds:

.. math::

   \mathbf{B}_{\mathrm{out}}\cdot\mathbf{n} = 0, \qquad
   |\mathbf{B}_{\mathrm{in}}|^2 + 2\mu_0 p = |\mathbf{B}_{\mathrm{out}}|^2.

The plasma's own exterior field comes from the **virtual-casing principle**.
In the BIEST convention, its layer densities on :math:`\partial\Omega` are
:math:`\sigma=\mathbf{B}\cdot\mathbf{n}` and
:math:`\mathbf{J}=\mathbf{B}\times\mathbf{n}`. The exterior field of the
enclosed plasma currents is the internal branch
:math:`-\nabla G[\sigma]-\mathrm{BiotSavart}[\mathbf{J}]`, evaluated with an
accurate singular quadrature (reused from the optional
``virtual_casing_jax`` package,
required as ``virtual-casing-jax >= 0.0.8`` from the canonical
``uwplasma/virtual_casing_jax`` repository;
:func:`~vmex.core.virtual_casing.surface_field_data_from_wout`
adapts a converged boundary + field, and
:func:`~vmex.core.virtual_casing.plasma_field_on_boundary` evaluates
the integral). The key structural fact: for a *fixed* trial boundary,
:math:`\mathbf{B}_{\mathrm{plasma}}` on that boundary does not depend on the
coil degrees of freedom, so it is precomputed once and frozen. The residual
assembled by
:class:`~vmex.core.virtual_casing.PlasmaVacuumInterface` is then a
smooth JAX function of the external-field dofs alone (coil Fourier
coefficients/currents of a callable ESSOS coil field via
:func:`~vmex.core.virtual_casing.external_B_cartesian`, or
``extcur``), so ``jax.value_and_grad`` returns gradients validated against
finite differences — no NESTOR adjoint is required.

The finite-beta single-stage example uses a pressure profile that vanishes at
the LCFS. It therefore needs no prescribed physical sheet current in the jump
condition; nonzero edge pressure or an imposed sheet current requires an
additional interface model.

Despite using the interface equations, that example is a **fixed-boundary**
optimization: every trial boundary is prescribed to VMEX and reconverged, and
both boundary and coil coefficients are decision variables. Virtual casing
separates the converged total VMEX field into plasma-current and external-coil
parts; it does not run NESTOR or a free-boundary equilibrium. The preview
``single_stage_free_boundary_optimization*.py`` examples instead hold the
plasma boundary implicit and vary only coil parameters through the coupled
NESTOR derivative below. They need ESSOS
(``pip install "vmex[coils]"``).

The reported normalized total-pressure jump is

.. math::

   \left\langle\left[
   (|\mathbf B_{\rm out}|^2-|\mathbf B_{\rm in}|^2-2\mu_0p_{\rm edge})
   / B_{\rm ref}^2\right]^2\right\rangle_A^{1/2}.

It is dimensionless and vanishes when the ideal-MHD pressure-balance
condition holds. It is not an error in the prescribed volume pressure
profile. Even when :math:`p_{\rm edge}=0`, it supplies the tangential-field
magnitude condition that ``B.n/B`` alone does not constrain.

Field-query API
---------------

:class:`~vmex.core.extender.MagneticField` provides stored Cartesian points,
``B``, ``absB``, and spatial derivatives through ``gradgradgradB``. A field
constructed from :meth:`~vmex.core.problem.VmecProblem.exterior_field` also
provides ``B_vjp`` and the three spatial-derivative VJPs in the problem's
boundary/current DOFs. The virtual-casing path applies outside the LCFS;
:class:`~vmex.core.extender.VmecInteriorField` evaluates the live VMEC
spectral field inside. Direct off-surface quadrature must stay away from the
source surface and all targets must stay away from external coil filaments.
The experimental
:meth:`~vmex.core.extender.VmecExtender.with_near_surface_continuation`
prepares the singular on-surface plasma field and gradient once, then uses the
first-order continuation
:math:`\mathbf B(\mathbf x_\Gamma+\delta\mathbf x)=\mathbf B_\Gamma+
\nabla\mathbf B_\Gamma\delta\mathbf x+O(|\delta\mathbf x|^2)`. It is
approximate, carries no error estimate, and is not qualified for physics
(measured below). Use direct quadrature within its measured accuracy range; a
finite field-line trace does not validate the extrapolated field or its
topology.

Virtual casing reconstructs the field produced by currents inside the plasma
surface. It does not determine the external coil field: supply an ESSOS coil
field or MGRID field and :class:`~vmex.core.extender.VmecExtender` adds the two.
This distinction matters for finite-beta exterior tracing and coil design.

``vmex_get_B_gradB.py`` demonstrates the stable interior API. The exterior
field and tracing previews need ESSOS (``pip install "vmex[coils]"``). The
single-stage previews write initial and optimized surface/coil VTK files;
setting
``MAKE_MOVIE=True`` adds a compact animation of accepted iterates. Set the
examples' ``MOVIE_SURFACE_COLOR`` to ``None``, ``"absB"``, ``"B.n/B"``, or
a scalar-field callable to control boundary coloring without storing VTK data
for every iteration.

Accuracy outside the surface
----------------------------

The direct path (:meth:`~vmex.core.extender.VmecExtender.B` without a
continuation plan) evaluates the virtual-casing integrals with the periodic
trapezoid rule on a fixed schedule of source grids. At a target a distance
:math:`d` from the surface its error behaves as :math:`e^{-2\pi d/h}`, up to a
weak algebraic factor, where :math:`h` is the largest source spacing of the
finest schedule level. Schedule levels count points over the **full torus**, while the source grid
holds one field period, so the default ``levels`` of ``from_wout``,
``from_state`` and ``exterior_field`` carries the ``nfp`` factor:
``((nfp nphi, ntheta), (2 nfp nphi, 2 ntheta))``. The finest level therefore has
``2 nfp nphi`` toroidal points on the whole torus and a toroidal spacing
:math:`h = 2\pi R/(2\,\mathrm{nfp}\,\mathrm{nphi})`, which improves with
``nfp`` rather than ignoring it. Keep :math:`d \gtrsim 2h`: one spacing gives
about three digits, two spacings about five. Unless ``nphi``, ``ntheta`` or
``levels`` is given, the per-period sampling is sized from the boundary so
that the requested ``digits`` are reached one minor radius out,
:math:`\mathrm{nphi} \ge \mathrm{digits}\,\ln 10\,R_0/(2\,\mathrm{nfp}\,a)`
with a floor of 32 (``_source_nphi_for_digits``); on the shipped QA WOUT
(:math:`R_0/a = 15.9`, ``nfp = 2``) that gives 64, and a tokamak-like aspect
ratio stays at 32.

The requested ``digits`` does not bound the returned error. The schedule
refines, target by target, until the achieved-error estimate below meets the
tolerance, but it cannot refine past its finest level, and the estimate does
not see truncation of the source data on that grid itself. On the
vacuum deck ``input.LandremanPaul2021_QA_lowres`` (``ctor`` of order
:math:`10^{-11}` A, so the exact plasma field outside is zero) the returned
field has these median | maximum errors relative to ``volavgB``, for 40
targets along the outward normal, ``digits = 4``, versus distance in minor
radii :math:`a` and per-period source grid ``nphi = ntheta = N``. The runs
date from 2026-09-13 (``benchmarks/review_20260913.json`` records the
``N = 32`` rows), before the grid was sized from the boundary:

.. list-table::
   :header-rows: 1

   * - N
     - d = a
     - 0.5 a
     - 0.2 a
     - 0.1 a
     - 0.05 a
     - 0.02 a
   * - 32
     - 3e-5 | 3e-3
     - 6e-3 | 1.3e-2
     - 0.12 | 0.18
     - 0.31 | 0.56
     - 0.45 | 1.3
     - 0.55 | 2.7
   * - 64
     - 2e-5 | 3e-4
     - 4e-5 | 3e-4
     - 1.8e-2 | 3e-2
     - 0.13 | 0.17
     - 0.30 | 0.62
     - 0.49 | 1.9

Each doubling of the grid moves a given error level about twice as close to the
surface, so no affordable grid reaches :math:`10^{-4}` within 0.1 a. The
known-answer tests in ``tests/test_virtual_casing_physics.py`` check the
identities on a circular torus carrying an outside z-axis current and an
inside axis filament: the internal branch returns the filament field outside
and on the surface and minus the z-axis field inside, to :math:`10^{-4}` at
three finest-level spacings.

Eager :meth:`~vmex.core.extender.VmecExtender.B` calls therefore check an
estimate of the returned error,
:meth:`~vmex.core.extender.VmecExtender.B_error_estimate`
(:func:`~vmex.core.virtual_casing.offsurface_error_estimate`). Since
virtual-casing-jax 0.0.6 the same estimate also chooses the schedule's level,
so the two can no longer disagree; it reports, per point, the larger of the
returned level's double-layer error and the square of the relative change
between the last two levels (halving the spacing squares the trapezoid error
factor). Errors are relative to the RMS of :math:`|B|` on the surface. When any point exceeds :math:`10^{-\mathrm{digits}}`,
``B`` emits :class:`~vmex.core.extender.ExteriorFieldAccuracyWarning`, or raises
:class:`~vmex.core.extender.ExteriorFieldAccuracyError` with
``accuracy_check="raise"``; ``accuracy_check="off"`` skips the estimate. The
returned field is identical in every mode. Traced calls (``jit``, ``grad``,
field-line integration) never check and can call the estimate directly. The
eager check evaluates the schedule a second time to obtain it.

On 816 targets at ``digits = 4`` (the vacuum deck above at N = 32 and 64 over
six distances, and the torus oracle on two schedules from 0.25 to 4 finest
spacings) no target that passed the estimate had an error above
:math:`1.5\times10^{-4}`, no target with an error below :math:`3\times10^{-5}`
was flagged, and the error was within 1.7 times the estimate for nine targets
in ten and within 21 times for 99 in 100. The double-layer self-test that
chose the level before virtual-casing-jax 0.0.6 passed 11 of the same targets
with errors above :math:`3\times10^{-4}`, one of them at 0.30, which is why
the estimate now chooses the level. The estimate does not see truncation of the source data on the finest
grid itself: at d = a with N = 64 the error reached 26 times the estimate for
one target in ten, while staying below :math:`3\times10^{-4}`.

Use the direct path above about 0.5 a with ``N >= 64``. Against a converged
target-graded quadrature of the same surface data on the 2.5 % beta QA deck
(ns = 51, remeasured 2026-09-22), the grid that
:meth:`~vmex.core.extender.VmecExtender.from_state` chooses there
(64 x 64 per period) returns the plasma field to 1e-13 at d = a and at worst
1e-6 at 0.5 a, but 2.4 % at 0.2 a and order one at 0.1 a and closer; the
estimate tracks the error (0.76 to 1.09 times it) and flags every such point.
:meth:`~vmex.core.extender.VmecExtender.with_near_surface_continuation` is the
only closer option and is approximate: with a 32 x 32 grid it is off by
1.6--2.4 % of the plasma field (about 1e-3 of :math:`|B|`) at every distance
from 0.01 a to 0.1 a, a floor set by its bilinear on-surface table rather than
by the distance, by 3.9--6.6 % at 0.2 a, and by 18--32 % at 0.5 a. Preparing
it took 196--397 s and 18.5 GB of memory on a loaded laptop, and at the
64 x 64 grid the process was killed for memory on a 36 GB machine. It has no
error estimate. A target-graded periodic trapezoid rule (substitute
:math:`\theta=\theta^*+u-a\sin u` about the target's nearest surface point, and
likewise in :math:`\phi`) is the reference used above: its 256 x 1024 and
384 x 1536 node versions agree to 2e-9 in the field and 2e-6 in its gradient
at every distance down to 0.01 a. It is the candidate replacement (plan item
E7), not yet part of the library.

Use the direct path only where its estimate meets the requested accuracy,
and refine the source data separately: the estimate does not bound
source-data truncation. JIT-traced evaluations require an explicit estimate
check. The exterior tracing example and its saved illustration use the
continuation and remain experimental. An earlier record of the continuation
(2026-09-16, the shipped QA wout) compared it with the direct path on a
current-free deck, whose exact plasma field is zero, so both of its numbers
were errors; it is withdrawn in favour of the finite-beta measurement above.

Coupled free-boundary adjoint
-----------------------------

Let :math:`F(z,c)=0` be the converged VMEC force residual after NESTOR has
computed the vacuum pressure on the moving edge, with equilibrium state
:math:`z` and coil parameters :math:`c`. For a scalar objective :math:`J`,
VMEX solves

.. math::

   F_z^T\lambda = J_z^T, \qquad
   \frac{dJ}{dc} = J_c - \lambda^T F_c,

:func:`~vmex.core.freeboundary_implicit.solve_free_boundary_implicit` keeps
the host-driven forward iterations off the AD tape, re-evaluates the complete
VMEC--NESTOR residual at the converged state, and solves this transpose system
with one matrix-free GCROT adjoint. A direct ESSOS ``BiotSavart.b_cyl`` field
retains coil shape/current derivatives without writing an mgrid file.

The public construction is explicit: create
:func:`~vmex.core.freeboundary_implicit.make_free_boundary_config`, map the
coil vector to a field with ``field_from_parameters``, call the implicit solve,
stack physics rows with :func:`vmex.core.optimize.residuals_from_tuples`, and
apply ``jax.value_and_grad``. ``take_free_boundary_gradients.py`` certifies
one direction by comparing the coupled GCROT adjoint with the boundary-Schur
adjoint, two independent solvers of the same linear system; a central
difference of re-solves has no usable step on this free boundary. The
free-boundary single-stage
previews pass the same scalar pair to SciPy. These examples need ESSOS
(``pip install "vmex[coils]"``).

Accuracy scope. Against finite differences of re-solves anchored by Newton
steps on the same projected coupled residual, the adjoint agreed to
1e-9--6e-7 on the 0.5 % beta single-stage objective. The forward solve does
not apply that anchoring. A status-0 solve at ``ftol = 1e-12`` sat about 1.2e-2
from the root in coefficient norm, and restarts from different references
returned values differing by up to 12 %. The gradient is exact for the root;
the value belongs to a nearby unanchored state. A zero-beta, zero-current
free boundary exists only when a nested flux surface of the coil field
encloses PHIEDGE. When an island chain or stochastic layer sits at that flux
(a 4/9 chain in one optimized coil set, confirmed by field-line tracing),
VMEX, VMEC2000 and VMEC++ limit-cycle instead of converging. Small beta does
not regularize it. Keep the transform away from the resonance, or reduce
PHIEDGE inside the good surfaces.

This path is currently limited to reverse mode. Its low-memory host Krylov
lane peaks near 3--5 GB on the bundled coarse examples, but the first coupled
transpose still takes about one to two minutes to compile on the reference
CPU and is not yet a practical GPU path. Its ``device="auto"`` policy therefore
uses the CPU on an accelerator host unless the process already pins JAX
placement, while retaining an explicit per-call GPU override.
Three transpose solvers are available through ``adjoint_solver``:
``"coupled_gcrot"`` (the certified default), ``"edge_response"``, which
iterates the coupled transpose on a dense model of NESTOR's edge response
built once per gradient, and ``"boundary_schur"``, the boundary-Schur
transpose. The boundary-Schur solver
differentiates one three-surface force row at a time, retains every terminal
radial stencil coupling in the bulk, isolates the one evolved edge row that
contains NESTOR's response, and eliminates the radial bulk with a
two-sided-equilibrated, globally pivoted
sparse LU. Reverse-mode row differentiation is used because each local
Jacobian has three times more inputs than outputs. The reduced transpose is
solved and back-substituted, then checked against the original coupled
residual; a failed certificate continues with coupled Krylov from the Schur
answer. No dense full-state Jacobian is formed.

The reduced lane is not yet the default. Direct local-row assembly removes
the full radial basis sweep, the pivoted band solve removes the inaccurate
no-pivot elimination, and the exact one-row interface avoids redundant NESTOR
pullbacks. Local-force and vacuum-response compilation remain the cold-cost
targets. The next measured step is to cache accepted-state local executables
and batch the NESTOR edge pullbacks on GPU. Promotion requires lower cold time
and memory on the bundled 3-D case
while retaining the re-solve finite-difference, CPU/GPU, and fixed/free field
certificates. Timings belong in the resource harness, not committed
JSON files.
