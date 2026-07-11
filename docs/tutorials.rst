Tutorials
=========

Every tutorial below is a runnable script in the repository's ``examples/``
directory: parameters at the top, no ``main()``, only the public API, and each
is smoke-tested in CI (``tests/core_new/test_examples.py``) so the code on this
page always runs.  Copy a script, edit the parameter block, and go.

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
profiles, 3D render) and the straight-field-line Boozer ``|B|`` spectrum on the last closed
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

Free-boundary beta scan
~~~~~~~~~~~~~~~~~~~~~~~~

Ramp the pressure of the free-boundary case at fixed coil currents; the boundary
is re-solved by NESTOR at every step as the plasma pushes outward against the
external field.

.. literalinclude:: ../examples/free_boundary_beta_scan.py
   :language: python


Optimization
------------

The ``examples/optimization/`` gallery drives a circular torus to precise
quasisymmetric (QA, QH, QP) and quasi-isodynamic (QI) configurations with
gradient-based least squares — user-authored ``(function, target, weight)``
objective terms and one least-squares call per ``max_mode`` continuation stage,
using implicit-differentiation gradients (``jac="implicit"``).  See
:doc:`optimization` for the objective library (quasisymmetry residual, aspect
ratio, magnetic well, Mercier ``DMerc``, ``|grad B|`` length) and the differentiation
machinery, and the QA example for the canonical pattern:

.. literalinclude:: ../examples/optimization/QA_optimization.py
   :language: python

The QH, QP, and QI scripts follow the same structure at different field periods;
QI runs a quasi-poloidal stage first, then refines to omnigenity.  These are the
heaviest examples (thousands of solves across the continuation ladder) and are
exercised in the nightly CI run.
