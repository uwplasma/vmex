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
  coils,
* coupled axisymmetric isotropic free-boundary beta continuation, and
* component-wise nonlinear convergence checks at a requested ``ftol=1e-12``.

The axisymmetric free-boundary path is a research capability. A first formal
resolution study, free-side initial-condition test, and in-memory continuation
restart are complete. Higher-resolution vacuum tangency and outer-domain
convergence, independent-reference, anisotropic, restart-file, and output gates
in ``plan.md`` remain. Nonaxisymmetric free-boundary mirrors and the toroidal
stellarator-mirror hybrid are later milestones and must not be inferred from
the axisymmetric result.

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

The scan input multiplies a VMEC-style conserved mass profile. Because the
geometry changes, requested and achieved beta are not exactly equal. For the
default profile ``p(s) = p0 (1-s)``, pressure also vanishes at the LCFS, so a
10% central beta does not imply a 10% edge-pressure jump or volume beta.

``summarize_axisymmetric_beta_scan`` reports:

* requested profile-amplitude beta,
* achieved central beta normalized by the reference vacuum field,
* volume-averaged beta,
* local central beta normalized by the solved plasma field,
* center radius and plasma/vacuum-side field,
* diamagnetic field ratio, and
* error against the paraxial estimate ``B/B_vac = sqrt(1-beta)``.

At ``(ns,nxi,nrho)=(7,13,7)``, the default 10% request reaches 9.47% central
beta and 3.18% volume beta. The center radius expands by 1.02%, while the
central field falls by 4.31%; the field ratio is within 0.57% relative of the
paraxial estimate. Thus field depression is the more sensitive validation
observable for this zero-edge-pressure profile.

The finite-beta mirror trend follows the WHAM/Pleiades discussion in Frank et
al., `Confinement performance predictions for a high field axisymmetric tandem
mirror <https://doi.org/10.1017/S002237782510055X>`_. Independent reference
curves remain an explicit promotion gate rather than being replaced by the
paraxial approximation.
