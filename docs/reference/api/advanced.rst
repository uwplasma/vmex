Advanced API
============

Everything below :doc:`basic`: the solver internals of :mod:`vmex.core` and
the open-field-line lane :mod:`vmex.mirror`, grouped as in
:doc:`/explanation/architecture`. Every docstring names the VMEC2000
counterpart it ports.

The package docstring of :mod:`vmex.core` is the module map this page follows:

.. automodule:: vmex.core
   :no-members:

Profiles
--------

.. automodule:: vmex.core.profiles
   :members:

Radial basis and axis regularity
--------------------------------

.. automodule:: vmex.core.radial_basis
   :members:

High-order reconstruction and force certificate
------------------------------------------------

.. automodule:: vmex.core.strong_force
   :members:

Force-balance polishing
-----------------------

.. automodule:: vmex.core.polish
   :members: PolishConfig, PolishReport, PolishResult, polish_legacy_solution, polish_native,
      native_polish_supported, physical_scales, polished_wout_ns, polished_wout_input,
      polished_wout_state, sample_high_order_state

:func:`vmex.solve`, :func:`vmex.solve_multigrid` and :func:`vmex.solve_file`
accept ``polish_force_balance`` (``polish`` on the single-grid call and
``solve_file``): ``False`` (default), ``True``, or ``"auto"``.  The polish
covers fixed-boundary axisymmetric decks with prescribed pressure and iota
(``NCURR = 0``, ``GAMMA = 0``, ``LASYM = F``); ``True`` on any other deck
raises, ``"auto"`` leaves it unpolished.  A deck can request it with comment
directives that VMEC2000 ignores::

   !@VMEX POLISH = AUTO
   !@VMEX POLISH_FAIL = ERROR

(``! VMEX: POLISH_FORCE_BALANCE = .TRUE.`` still parses).  Directives belong
to :mod:`vmex.core.run_options` and never become :class:`vmex.VmecInput`
fields; structured JSON carries them in a reserved ``_vmex`` section.
Precedence is ``CLI option > Python keyword > file directive > default``.
``polish_fail`` selects the failure behavior: ``"error"`` raises,
``"fallback"`` returns the unpolished state, ``"warn"`` does the same with a
:class:`RuntimeWarning`.

A certified result carries ``native_equilibrium`` (the certified continuous
state), ``strong_force`` (its independent certificate) and ``polish_report``;
``polished_state`` is its view on the solve mesh and the deck's modes, which
:class:`vmex.core.optimize.Equilibrium` uses.  WOUT files sample the native
state on the denser :func:`vmex.core.polish.polished_wout_ns` mesh,
with ``MPOL`` widened to hold padded modes, so the file carries the certified
state.

For boundary objectives, :func:`vmex.evaluate_high_order_surface` returns a
one-field-period array view accepted by ESSOS, and
:func:`vmex.surface_field_data_from_high_order` converts the same analytic
geometry and edge field for ``virtual_casing_jax``.  Neither path writes a
``wout`` file or finite-differences a surface tangent.  Field-aligned
objectives use :func:`vmex.boozer_spectrum_high_order`, which sends continuous
geometry and field tables to BOOZ_XFORM_JAX without reconstructing a sampled
radial mesh.

Spectral representation and physics kernels
-------------------------------------------

.. automodule:: vmex.core.fourier
   :members:

.. automodule:: vmex.core.transforms
   :members:

.. automodule:: vmex.core.geometry
   :members:

.. automodule:: vmex.core.fields
   :members:

.. automodule:: vmex.core.forces
   :members:

.. automodule:: vmex.core.residuals
   :members:

Solver
------

.. automodule:: vmex.core.setup
   :members:

.. automodule:: vmex.core.preconditioner
   :members:

.. automodule:: vmex.core.preconditioner_2d
   :members:

.. automodule:: vmex.core.step
   :members:

.. automodule:: vmex.core.solver
   :members:

.. automodule:: vmex.core.multigrid
   :members:

.. automodule:: vmex.core.restart
   :members:

.. automodule:: vmex.core.device
   :members:

Free boundary
-------------

.. automodule:: vmex.core.vacuum
   :members:

.. automodule:: vmex.core.freeboundary
   :members:

.. automodule:: vmex.core.freeboundary_implicit
   :members:

.. automodule:: vmex.core.virtual_casing
   :members:

.. automodule:: vmex.core.mgrid
   :members:

.. automodule:: vmex.core.extender
   :members:

Physics objectives
------------------

The objective catalog with usage snippets is :doc:`/reference/objectives`.
The wout-parity scalar targets (aspect ratio, volume, beta, elongation, iota)
that the objective modules share live in one place and are re-exported by
:mod:`vmex.core.optimize`:

.. automodule:: vmex.core.statephysics
   :members:

.. automodule:: vmex.core.omnigenity
   :members:

.. automodule:: vmex.core.bounce
   :members:

.. automodule:: vmex.core.qi
   :members:

.. automodule:: vmex.core.maxj
   :members:

.. automodule:: vmex.core.gammac
   :members:

.. automodule:: vmex.core.bootstrap
   :members:

.. automodule:: vmex.core.stability
   :members:

.. automodule:: vmex.core.turbulence
   :members:

Outputs
-------

.. automodule:: vmex.core.scaling
   :members:

.. automodule:: vmex.core.nyquist
   :members:

.. automodule:: vmex.core.postprocess
   :members:

.. automodule:: vmex.core.printing
   :members:

.. automodule:: vmex.core.plotting
   :members:

.. automodule:: vmex.core.boozer
   :members:

The differentiable route from a spectral state to a single-surface Boozer
transform: traceable ``wout``-convention mode tables that ``booz_xform_jax``
consumes, so ``jax.grad`` flows from boundary coefficients through to
downstream kinetic codes.

.. automodule:: vmex.core.boozer_tables
   :members:

Straight-axis mirrors
---------------------

.. automodule:: vmex.mirror
   :no-members:

Collocation bases, geometry, and force kernels first; then the spline
discretization and the solves built on them; then the exterior vacuum used by
the free-boundary lane, derivatives, gyrokinetic geometry, and MOUT output.

.. automodule:: vmex.mirror.analytic
   :members:

.. automodule:: vmex.mirror.basis
   :members:

.. automodule:: vmex.mirror.geometry
   :members:

.. automodule:: vmex.mirror.forces
   :members:

.. automodule:: vmex.mirror.splines
   :members:

.. automodule:: vmex.mirror.model
   :members:

.. automodule:: vmex.mirror.solver
   :members:

.. automodule:: vmex.mirror.exterior
   :members:

.. automodule:: vmex.mirror.free_boundary
   :members:

.. automodule:: vmex.mirror.implicit
   :members:

.. automodule:: vmex.mirror.turbulence
   :members:

.. automodule:: vmex.mirror.output
   :members:

.. automodule:: vmex.mirror.metrics
   :members:

Errors and CLI
--------------

.. automodule:: vmex.core.errors
   :members:

.. automodule:: vmex.core.cli
   :members:

``vmex --doctor`` collects and formats its installation report here.

.. automodule:: vmex.doctor
   :members:
