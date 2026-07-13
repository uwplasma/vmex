Tutorials
=========

Every tutorial below is a runnable script in the repository's ``examples/``
directory: parameters at the top, no ``main()``, and only the public API.
Fast workflows are smoke-tested on pull requests; expensive optimization and
free-boundary workflows run in nightly CI, and external-code validation
scripts state their pinned dependency explicitly. Copy a script, edit the
parameter block, and go.

Run any of them directly::

   python examples/fixed_boundary_run.py

The light scripts finish in a few seconds once JAX has compiled; the
free-boundary and optimization scripts are heavier (see each section).

.. contents:: On this page
   :local:
   :depth: 1


Getting started
---------------

Fixed-boundary run
~~~~~~~~~~~~~~~~~~~

Read an ``&INDATA`` deck, converge the equilibrium on the multigrid ladder with
VMEC2000-format progress printing, and write and plot the ``wout``.  This is the
three-step workflow every new user needs.

.. literalinclude:: ../examples/fixed_boundary_run.py
   :language: python

All diagnostics and the Boozer transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

vmec-jax ships its plotting and its Boozer transform in the box.  This produces
every ``plot_wout`` figure (flux-surface summary, cross-sections, ``|B|``,
profiles, and a 3D ``|B|`` render with LCFS field lines) and the straight-field-line Boozer ``|B|`` spectrum on the last closed
flux surface — the view used to judge quasisymmetry.

.. literalinclude:: ../examples/plot_and_boozer.py
   :language: python

VMEC++ JSON input
~~~~~~~~~~~~~~~~~

vmec-jax reads both the classic ``&INDATA`` namelist and the VMEC++ JSON schema,
and can write the JSON form — a drop-in for either ecosystem.  This converts a
deck, reads it back, and confirms the two representations describe one
equilibrium.

.. literalinclude:: ../examples/run_from_json.py
   :language: python


Profiles and finite-beta physics
--------------------------------

Profile representations
~~~~~~~~~~~~~~~~~~~~~~~~

Pressure and rotational transform (or current) can be given as polynomial
coefficients (``power_series``) or spline knots (``cubic_spline``).  This solves
the same equilibrium both ways and shows they agree, then points to the
``NCURR=0`` vs ``NCURR=1`` (prescribed iota vs prescribed current) switch.

.. literalinclude:: ../examples/profiles_power_and_spline.py
   :language: python

Finite-beta pressure scan
~~~~~~~~~~~~~~~~~~~~~~~~~~

Ramp the pressure and read three diagnostics straight from the wout: the
volume-averaged beta, the Shafranov shift (outward motion of the magnetic axis),
and the Mercier ``DMerc`` interchange-stability profile.  Each step is
hot-restarted from the previous equilibrium.

.. literalinclude:: ../examples/finite_beta_scan.py
   :language: python


Performance: hot restart
------------------------

Seed each point of a parameter scan from the previous converged state.  Warm
restarts converge in about one iteration instead of hundreds, and because
vmec-jax caches one compiled executable per solver structure, the whole scan
recompiles nothing.

.. literalinclude:: ../examples/hot_restart_scan.py
   :language: python


Differentiation
---------------

Fixed-boundary gradients (implicit differentiation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Differentiate a *converged* equilibrium.  ``jax.grad`` returns exact derivatives
of wout scalars (aspect ratio, magnetic energy, ...) with respect to the
boundary Fourier coefficients and profile parameters, computed by the implicit
function theorem — one adjoint solve, O(1) memory, no finite-difference step to
tune.  The script checks the adjoint gradient against central differences.

.. literalinclude:: ../examples/take_gradients.py
   :language: python

Free-boundary gradients (virtual casing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The differentiable complement for free boundary: the plasma–vacuum interface
mismatch is written as a smooth objective and differentiated with respect to the
coil currents (``extcur``) and coil Fourier shape, finite-difference-validated.

.. literalinclude:: ../examples/take_free_boundary_gradients.py
   :language: python


Free boundary
-------------

From an mgrid file
~~~~~~~~~~~~~~~~~~

Prescribe the *coils* instead of the boundary: the coil currents (``EXTCUR``)
and their tabulated vacuum field (an mgrid file) drive a NESTOR vacuum solve, and
VMEC finds the plasma boundary that balances against it.  The last closed flux
surface is an output, not an input.

.. literalinclude:: ../examples/free_boundary_mgrid.py
   :language: python

Direct ESSOS coils
~~~~~~~~~~~~~~~~~~

``free_boundary_essos_coils.py`` replaces mgrid interpolation with direct,
differentiable Biot--Savart evaluation of the Landreman--Paul QA coils. It
calibrates pressure against *achieved* WOUT beta and uses bounded adaptive
continuation. The validated branch reaches 3.350% actual beta; the failed
3.3625% minimum-step trial is retained as a conditioning limit rather than
shown as a converged surface.

.. image:: _static/figures/readme_essos_beta_scan.png
   :alt: Direct-coil Landreman-Paul free-boundary beta continuation
   :width: 95%

.. literalinclude:: ../examples/free_boundary_essos_coils.py
   :language: python

Direct coils and their generated mgrid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``free_boundary_tokamak_coils.py`` constructs circular TF/PF coils, generates
a VMEC2000-compatible mgrid from those exact filaments, and solves both
free-boundary providers at actual beta 0, 1.496%, and 3.009%. The maximum LCFS
coefficient difference is ``6.31e-4``. Both WOUT sets and the parity CSV are
written alongside the standard 3D, ``|B|``, surface, profile, and Mercier plots.

.. image:: _static/figures/readme_tokamak_coil_parity.png
   :alt: Direct-coil and generated-mgrid solved tokamak boundaries
   :width: 95%

.. image:: _static/figures/tokamak_coil_3d.png
   :alt: Tokamak coils, solved LCFS, field lines, and boundary field strength
   :width: 62%

.. literalinclude:: ../examples/free_boundary_tokamak_coils.py
   :language: python

Free-boundary beta scan
~~~~~~~~~~~~~~~~~~~~~~~~

Ramp the pressure of the free-boundary case at fixed coil currents; the boundary
is re-solved by NESTOR at every step as the plasma pushes outward against the
external field. Each converged state seeds the next pressure point, including
its solved LCFS.

.. literalinclude:: ../examples/free_boundary_beta_scan.py
   :language: python

Single-stage free-boundary coil optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Optimize external-current amplitudes against the virtual-casing normal-field
residual using its exact gradient. This example requires the fetched CTH WOUT
and mgrid assets; it skips cleanly when those optional validation files are not
installed.

.. literalinclude:: ../examples/single_stage_free_boundary_opt.py
   :language: python


Straight mirrors and toroidal hybrids
-------------------------------------

Fixed-boundary mirror gradients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Solve a finite-pressure, finite-current flared mirror at ``ftol=1e-12`` and
differentiate an interior radius with the mirror implicit adjoint. Boundary,
flux, pressure, and current derivatives are checked against independently
reconverged central differences before MOUT and the standard plots are written.

.. image:: _static/figures/mirror_fixed_boundary_3d.png
   :alt: Fixed-boundary mirror geometry and magnetic field
   :width: 82%

.. literalinclude:: ../examples/mirror_fixed_boundary_gradients.py
   :language: python

Free-boundary mirror through 50% beta
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two circular end coils drive an open-field equilibrium whose lateral LCFS is
solved jointly with the exterior vacuum. Every accepted beta point is a
converged equilibrium, and the endpoint plots include the horizontal mirror,
coils, cap-to-cap field lines, ``|B|``, pressure, cross-sections, and force
history. The default exterior solve is intentionally a full/nightly workflow.

.. image:: _static/figures/mirror_free_boundary_beta50_summary.png
   :alt: Solved 50 percent beta mirror boundary, field, pressure, and convergence
   :width: 95%

.. literalinclude:: ../examples/mirror_free_boundary_beta_scan.py
   :language: python

Toroidal stellarator--mirror hybrid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Four straight mirror-like sides and four rotating-ellipse stellarator corners
remain a closed torus. The fixed-boundary script traces the 16-coil vacuum
axis, continues corner shaping, and solves the finite-current equilibrium.
The free-boundary script uses the same coils and publishes only accepted NESTOR
surfaces; its Fourier corrector currently stops at the documented 0.8333%
achieved-beta limit.

.. image:: _static/figures/hybrid_fixed_coils_fieldlines.png
   :alt: Sixteen coils, fixed-boundary hybrid LCFS, and field lines
   :width: 82%

.. literalinclude:: ../examples/toroidal_stellarator_mirror_hybrid.py
   :language: python

.. image:: _static/figures/hybrid_free_coils_fieldlines.png
   :alt: Sixteen coils and solved free-boundary hybrid LCFS with field lines
   :width: 82%

.. literalinclude:: ../examples/toroidal_stellarator_mirror_hybrid_free_boundary.py
   :language: python

Independent Pleiades reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The resolution-qualified two-coil beta trend is independently regenerated
with Pleiades at pinned commit ``0161abb3``. Set ``PLEIADES_ROOT`` at the top;
the script writes ignored review output and never silently replaces the bundled
CSV benchmark.

.. literalinclude:: ../examples/validation/pleiades_mirror_reference.py
   :language: python


Optimization
------------

The ``examples/optimization/`` gallery drives a circular torus to precise
quasisymmetric (QA, QH, QP) and quasi-isodynamic (QI) configurations with
gradient-based least squares — user-authored ``(function, target, weight)``
objective terms with implicit-differentiation gradients
(``jac="implicit"``).  See :doc:`objectives` for the full objective library
(quasisymmetry and omnigenity residuals, Redl bootstrap, ballooning
stability, turbulence proxies, scalar targets) and :doc:`optimization` for
the differentiation machinery and the measured campaign timings.

Single-call ESS optimization (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The recommended pattern is **one** ``least_squares`` call with *all* the
boundary harmonics released at once and Exponential Spectral Scaling
(``use_ess=True``) ordering them through the trust region — no
``max_mode`` continuation loop.  Measured: precise QA (QS 7.2e-6) in
14.5 minutes on a CPU.

.. literalinclude:: ../examples/optimization/QA_optimization_ess.py
   :language: python

``QI_optimization_ess.py`` is the quasi-isodynamic analogue: the traceable
Goodman constructed-QI residual (:class:`~vmec_jax.core.omnigenity.QIResidual`)
plus practical targets, one call at ``max_mode = 6`` (25x residual
reduction in 17.3 minutes).

.. literalinclude:: ../examples/optimization/QI_optimization_ess.py
   :language: python

Staged ``max_mode`` continuation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The classic ladder — one least-squares stage per ``max_mode``, each seeded
with the previous stage's boundary — remains available and is what
``QA_optimization.py``, ``QH_optimization.py``, ``QP_optimization.py``, and
``QI_optimization.py`` run (QI with a quasi-poloidal basin stage first).
It reaches the same precision class as the single-call pattern at roughly
twice the wall time; the scripts stay side by side so the comparison is
reproducible.

.. literalinclude:: ../examples/optimization/QA_optimization.py
   :language: python

.. literalinclude:: ../examples/optimization/QH_optimization.py
   :language: python

.. literalinclude:: ../examples/optimization/QP_optimization.py
   :language: python

.. literalinclude:: ../examples/optimization/QI_optimization.py
   :language: python

Self-consistent bootstrap current
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The QA and QH bootstrap examples alternate the equilibrium with the
differentiable Redl current model until the current-profile mismatch closes.
They reproduce the workflow of Landreman, Buller, and Drevlak (2022) against
the optional Zenodo archive selected by ``VMEC_JAX_ZENODO_2205_02914``.

.. image:: _static/figures/readme_bootstrap.png
   :alt: Self-consistent QA and QH bootstrap-current validation
   :width: 95%

.. literalinclude:: ../examples/optimization/QA_bootstrap_selfconsistent.py
   :language: python

.. literalinclude:: ../examples/optimization/QH_bootstrap_selfconsistent.py
   :language: python

These are the heaviest examples (hundreds to thousands of solves) and are
exercised in the nightly CI run.
