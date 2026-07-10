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
* coupled axisymmetric isotropic free-boundary beta continuation, and
* component-wise nonlinear convergence checks at a requested ``ftol=1e-12``.

The axisymmetric free-boundary path is a research capability. A first formal
resolution study, free-side initial-condition test, and in-memory continuation
restart are complete. Higher-resolution vacuum tangency and outer-domain
convergence, independent-reference, anisotropic, restart-file, and output gates
in ``plan.md`` remain. Nonaxisymmetric free-boundary mirrors and the toroidal
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
``1e-12``. A 1,039-unknown ``15x15`` office-CPU run reaches ``7.67e-13`` in
65.0 seconds without a dense fallback. The Krylov solve is still expensive;
systems through 1,024 unknowns therefore have a bounded dense residual-Newton
fallback, while larger systems report matrix-free convergence honestly.

``device=None`` uses the shared measured device policy. On the office host,
the 747-unknown ``13x13`` case took 41.4 seconds on CPU and 99.9 seconds on one
RTX A4000. CPU and GPU energies agreed to displayed precision, and maximum
stream function and pitch agreed within ``1e-8`` relative. Explicit
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

The mixed vacuum truncation fixes the correction potential on the outer
cylinder, preserves zero correction flux through the axial cuts, and obtains
total-field tangency naturally on the plasma side. Finite-wall Neumann and
mixed-Dirichlet center fields approach one another as the outer cylinder is
expanded, but their remaining gap is reported as truncation uncertainty. A
true exterior Dirichlet-to-Neumann or boundary-integral operator remains an M5
promotion gate.
