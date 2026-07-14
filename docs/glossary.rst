Glossary and VMEC2000 name map
==============================

This page maps the names used in VMEC2000 papers, output, and Fortran source
to the public objects and internal variables used by vmec-jax. See
:doc:`architecture` for the module-to-subroutine map and :doc:`equations` for
the defining equations.

Coordinates and meshes
----------------------

.. glossary::

   ``s``
      Normalized toroidal flux, ``s = Phi/Phi_edge``. Full-mesh values are
      ``RunSetup.s_full``; half-mesh values are ``RunSetup.s_half``.

   ``u``, ``theta``
      VMEC poloidal angle. Real-space arrays use the theta axis selected by
      :class:`~vmec_jax.core.fourier.Resolution`.

   ``v``, ``zeta``
      Geometric toroidal angle. Fourier phases use ``m*theta - n*nfp*zeta``.

   ``lambda``
      Stream-function correction to the straight-field-line angle. Its
      spectral state is ``SpectralState.L_sin`` and, for ``lasym=True``,
      ``SpectralState.L_cos``.

   full mesh
      The ``ns`` radial surfaces including axis and LCFS. Geometry
      coefficients in :class:`~vmec_jax.core.solver.SpectralState` live here.

   half mesh
      Midpoints between full-mesh surfaces. Magnetic and thermodynamic fields
      naturally live here; WOUT interpolation conventions are documented in
      :doc:`wout_reference`.

   ``mnmax`` / ``mnmax_nyq``
      Number of boundary-resolution / Nyquist-resolution Fourier modes,
      represented by :class:`~vmec_jax.core.fourier.ModeTable`.

State, fields, and forces
-------------------------

.. list-table:: VMEC2000 to vmec-jax names
   :header-rows: 1
   :widths: 18 29 53

   * - VMEC2000 name
     - vmec-jax name
     - Meaning
   * - ``rmn``, ``zmn``, ``lmn``
     - ``SpectralState.R_*``, ``Z_*``, ``L_*``
     - Fourier coefficients of the geometry and stream function.
   * - ``xc``
     - :class:`~vmec_jax.core.solver.SpectralState`
     - Evolved, internally normalized spectral state.
   * - ``xcdot``
     - solver-loop velocity state
     - Richardson momentum variable; not an equilibrium output.
   * - ``tau`` / ``sqrtg``
     - geometry Jacobian tables
     - Flux-coordinate Jacobian computed by :mod:`vmec_jax.core.geometry`.
   * - ``bsupu``, ``bsupv``
     - contravariant field tables
     - Contravariant magnetic-field components.
   * - ``bsubu``, ``bsubv``, ``bsubs``
     - covariant field tables
     - Covariant magnetic-field components, including Nyquist postprocessing.
   * - ``armn``, ``azmn``, ``blmn``
     - :class:`~vmec_jax.core.transforms.SpectralForce`
     - Spectral radial, vertical, and lambda force channels.
   * - ``fsqr``, ``fsqz``, ``fsql``
     - :class:`~vmec_jax.core.residuals.ForceResiduals`
     - Component-wise normalized physical force residuals used for convergence.
   * - ``fsqr1``, ``fsqz1``, ``fsql1``
     - :class:`~vmec_jax.core.residuals.PreconditionedResiduals`
     - Residuals after radial/lambda preconditioning; diagnostic, not ``ftol``.
   * - ``wb``, ``wp``
     - ``FunctDiagnostics.wb``, ``wp``
     - Magnetic and pressure energy contributions.
   * - ``rcon``, ``zcon``
     - ``SolverRuntime.rcon0``, ``zcon0``
     - Spectral-condensation constraint baselines.

Solver controls
---------------

.. list-table:: Iteration and restart names
   :header-rows: 1
   :widths: 18 29 53

   * - VMEC2000 name
     - vmec-jax name
     - Meaning
   * - ``FTOL_ARRAY``
     - ``VmecInput.ftol_array``
     - Component-wise force tolerance at each multigrid stage.
   * - ``NITER_ARRAY``
     - ``VmecInput.niter_array``
     - Maximum iterations at each multigrid stage.
   * - ``NS_ARRAY``
     - ``VmecInput.ns_array``
     - Coarse-to-fine radial-resolution ladder.
   * - ``delt``
     - ``StepControl.time_step``
     - Richardson time step.
   * - ``otau``
     - ``StepControl.inv_tau``
     - Ten-value damping history.
   * - ``irst``
     - restart kind in :mod:`vmec_jax.core.step`
     - Normal step, Jacobian restart, or residual-growth restart.
   * - ``ijacob``
     - ``_LoopCarry.ijacob``
     - Jacobian restart/escalation counter.
   * - ``ivac``
     - ``FreeBoundaryState.ivac``
     - Free-boundary vacuum activation state.
   * - ``nvacskip``
     - ``VmecInput.nvacskip``
     - Number of MHD iterations between NESTOR vacuum updates.
   * - ``lfreeb``
     - ``VmecInput.lfreeb`` / ``SolverRuntime.lfreeb``
     - Selects the solved free-boundary lane.
   * - ``mgrid_file``, ``extcur``
     - ``VmecInput.mgrid_file``, ``extcur``
     - External-field source and per-current-group amplitudes.

Output and API names
--------------------

``wout`` is the standard toroidal VMEC output and is represented by
:class:`~vmec_jax.core.wout.WoutData`. ``mout`` is the mirror-native open-field
output represented by :class:`~vmec_jax.mirror.output.MoutData`. A converged
toroidal API result is :class:`~vmec_jax.core.optimize.Equilibrium`; a
straight-mirror result is :class:`~vmec_jax.mirror.solver.MirrorSolveResult`
or :class:`~vmec_jax.mirror.free_boundary.FreeBoundaryMirrorResult`.
