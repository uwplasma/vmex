Architecture and execution
==========================

The implementation lives in :mod:`vmex.core` (toroidal equilibria) and
:mod:`vmex.mirror` (open mirrors and stellarator-mirror hybrids). Each module
has one concern and a header docstring naming its VMEC2000 counterpart and the
equations it implements. The physics kernels are pure JAX and shared by the
CLI, the differentiable API, plotting and the wout writer. This page maps the
modules, states the purity and placement rules, and explains how independent
work runs in parallel. The generated :doc:`/reference/api/basic` and
:doc:`/reference/api/advanced` pages document every public module.

Module map
----------

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Module
     - Role
     - VMEC2000 counterpart
   * - :mod:`~vmex.core.input`
     - ``VmecInput``: INDATA + structured JSON parsing, round-trip writers
     - ``readin.f``, ``vmec_input.f``
   * - :mod:`~vmex.core.profiles`
     - pressure / iota / current parameterizations (power series, splines,
       two-power, pedestal, ...)
     - ``profile_functions.f``, ``profil1d.f``
   * - :mod:`~vmex.core.fourier`
     - ``(m,n)`` bookkeeping, parity tables, ``mscale/nscale``, trig tables,
       m=1 constraint maps
     - ``fixaray.f``
   * - :mod:`~vmex.core.transforms`
     - spectral <-> real-space transforms as batched matmuls
     - ``totzsps.f``/``tomnsps.f`` (``totzsp.f90``/``tomnsp.f90``)
   * - :mod:`~vmex.core.geometry`
     - real-space :math:`R, Z, \lambda`, half-mesh Jacobian ``tau``/``sqrt(g)``,
       metrics ``guu, guv, gvv``
     - ``jacobian.f``
   * - :mod:`~vmex.core.fields`
     - :math:`B^u, B^v, |B|`, covariant B, pressure, energies ``wb/wp``,
       ``tcon``
     - ``bcovar.f``, ``add_fluxes.f90``
   * - :mod:`~vmex.core.forces`
     - MHD force kernels + spectral-condensation constraint force
     - ``forces.f``, ``alias.f``
   * - :mod:`~vmex.core.residuals`
     - ``fsqr/fsqz/fsql``, m=1 constraint, ``fedge``
     - ``residue.f90``, ``getfsq.f``
   * - :mod:`~vmex.core.preconditioner`
     - 1D radial preconditioner, vectorized tridiagonal (Thomas) solve
     - ``precondn.f``, ``scalfor.f``, ``lamcal.f90``, ``tridslv``
   * - :mod:`~vmex.core.preconditioner_2d`
     - 2D block preconditioner: matrix-free Newton step (``jax.jvp``
       Hessian-vector products + SOLVAX GMRES)
     - ``Hessian/precon2d.f`` (jog-free)
   * - :mod:`~vmex.core.step`
     - damped 2nd-order Richardson step, ``dtau`` damping (``ndamp=10``),
       ``irst`` back-off
     - ``evolve.f``, ``restart.f``
   * - :mod:`~vmex.core.setup`
     - radial grids, 1D profile arrays, boundary processing, initial guess
     - ``profil1d.f``, ``profil3d.f``, ``readin.f``
   * - :mod:`~vmex.core.solver`
     - single-grid solve loop: ``lax.while_loop`` core + host-blocked CLI lane
     - ``funct3d.f``, ``eqsolve.f``
   * - :mod:`~vmex.core.multigrid`
     - fixed- and free-boundary ``NS_ARRAY`` ladders, coarse-to-fine
       interpolation, hot restart, vacuum continuation/rebuild
     - ``runvmec.f``, ``interp.f``
   * - :mod:`~vmex.core.vacuum`
     - NESTOR: Green's function, ``analyt``/``scalpot``, ``potvac`` solve
     - ``NESTOR_vacuum/`` (``precal``, ``surface``, ``bextern``, ``analyt``,
       ``greenf``, ``fourp``, ``scalpot``, ``solver``, ``bsqvac``)
   * - :mod:`~vmex.core.freeboundary`
     - free-boundary iteration, ``ivac``/``nvacskip`` cadence, external-field
       protocol
     - ``funct3d.f`` (free-boundary block)
   * - :mod:`~vmex.core.mgrid`
     - mgrid netCDF read/write, differentiable interpolated field, and
       ESSOS/SIMSOPT/``xyz -> B`` host-side Biot--Savart tabulation
     - MAKEGRID file format, ``mgrid_mod.f90``
   * - :mod:`~vmex.core.nyquist`
     - Nyquist-resolution Fourier tables, ``bsubs``, jxbforce, Mercier
     - ``wrout.f``, ``bss.f``, ``jxbforce.f``, ``mercier.f``
   * - :mod:`~vmex.core.postprocess`
     - derived wout quantities (beta, currents, ``specw``, ``equif``, ...)
     - ``eqfor.f``, ``bcovar.f`` outputs
   * - :mod:`~vmex.core.wout`
     - VMEC-compatible ``wout_*.nc`` schema, writer and reader, including
       symmetric and LASYM NESTOR potential/surface tables
     - ``wrout.f``
   * - :mod:`~vmex.core.printing`
     - VMEC2000-format iteration lines, stage banners, termination summary
     - ``printout.f``, ``initialize_radial.f``, ``runvmec.f``
   * - :mod:`~vmex.core.cli`
     - the ``vmex`` command (``vmec`` is an alias)
     - ``vmec.f``/``runvmec.f`` driver

The modules without a VMEC2000 counterpart group as follows:

- **Derivatives and optimization:** :mod:`~vmex.core.implicit` (implicit
  function theorem, adjoint and tangent solves),
  :mod:`~vmex.core.freeboundary_implicit` (coupled VMEX--NESTOR derivative),
  :mod:`~vmex.core.optimize` and :mod:`~vmex.core.problem` (objectives,
  drivers and optimizer-neutral callables), :mod:`~vmex.core.monitoring`.
- **Objectives and diagnostics:** :mod:`~vmex.core.omnigenity`,
  :mod:`~vmex.core.qi`, :mod:`~vmex.core.maxj`, :mod:`~vmex.core.bounce`,
  :mod:`~vmex.core.gammac`, :mod:`~vmex.core.bootstrap`,
  :mod:`~vmex.core.stability`, :mod:`~vmex.core.turbulence`,
  :mod:`~vmex.core.neoclassical`, :mod:`~vmex.core.boozer`,
  :mod:`~vmex.core.boozer_tables`, :mod:`~vmex.core.statephysics`.
- **Fields outside the solver:** :mod:`~vmex.core.virtual_casing`,
  :mod:`~vmex.core.extender` (exterior and interior field queries),
  :mod:`~vmex.core.tracing` (ESSOS handoff and alpha tracing).
- **High-order force balance:** :mod:`~vmex.core.strong_force`,
  :mod:`~vmex.core.radial_basis` and the ``polish*`` modules
  (:doc:`high-order-force-balance`).
- **Workflow:** :mod:`~vmex.core.restart`, :mod:`~vmex.core.parallel`,
  :mod:`~vmex.core.desc`, :mod:`~vmex.core.scaling`,
  :mod:`~vmex.core.run_options`, :mod:`~vmex.core.plotting`,
  :mod:`~vmex.core.device`, :mod:`~vmex.core.errors`.

State and purity
----------------

The solver state is a frozen pytree
(:class:`~vmex.core.solver.SpectralState`): spectral coefficients of
:math:`R, Z, \lambda` (plus the asymmetric partners when ``lasym``), the
Richardson velocity, time step, damping history, iteration counters, and the
restart flag. Solver functions are pure ``state -> state`` maps, which is what
makes the same kernels usable from ``jit``, ``grad``, and ``vmap``.

Static configuration (resolutions, flags) is hashable and kept out of traced
signatures. Each distinct ``NS_ARRAY`` stage structure compiles its own
executable, and later ladders with the same structures reuse it; the two
execution lanes (a host-driven CLI loop and a traced ``lax.while_loop``) are
described in :doc:`iteration`.

Device placement
----------------

:mod:`vmex.core.device` decides where each solve runs. An explicit ``device=``
always wins; ``device=None`` follows JAX placement; the default
``device="auto"`` stands down for an active ``jax.default_device`` context or
a user-pinned JAX platform, and otherwise applies a per-stage rule
(:func:`~vmex.core.device.recommended_device`): CPU below
:data:`~vmex.core.device.GPU_MIN_ITERATION_WORK` (``100_000``) of the work
proxy :math:`w = \mathrm{ns} \times \mathrm{mnmax} \times \mathrm{nznt}`
(:func:`~vmex.core.device.iteration_work`), CPU above
:data:`~vmex.core.device.GPU_MAX_SPECTRAL_MODES` (``512``) modes, and GPU in
between. That rule was calibrated on the July 2026 record
``benchmarks/gpu_baseline.json``. The later re-measurement on two RTX A4000
GPUs (``benchmarks/gpu_a4000_2026-09-16.json``, VMEX 0.9.1) found no deck or
problem size where the GPU was faster, so the thresholds do not transfer
between machines; measure on your own hardware (:doc:`/howto/run-on-gpu`).

Three paths differ from the rule. Free-boundary accelerator runs keep the
plasma iteration on the selected device but run the dense NESTOR block on the
CPU, reusing its LU factor between full updates. High-level optimization pins
implicit-gradient work to the CPU when VMEX owns placement
(:func:`~vmex.core.device.resolve_implicit_device`), because the gradient graph
is launch-bound on an accelerator; low-level :func:`~vmex.core.implicit.run`
follows JAX placement unless given ``device="auto"``. Mirror solves, whose
SciPy control loop calls JAX repeatedly, default to the CPU
(:func:`~vmex.core.device.resolve_mirror_device`). Explicit devices and JAX
placement contexts are honored on all three paths.

Parallel execution
------------------

**Within one solve.** XLA:CPU multithreads the batched ``totzsps``/``tomnsps``
transforms and the tridiagonal preconditioner solves, so a single solve already
uses several cores. Several adjoint or tangent right-hand sides for one fixed
point share one linearization
(:func:`~vmex.core.implicit.implicit_state_pullback_multi_rhs`,
:func:`~vmex.core.implicit.implicit_state_tangent_multi_rhs`).

**Across independent solves.** Each host solve releases the Python GIL while
XLA executes, so :func:`vmex.core.parallel.solve_ensemble` runs independent
equilibria (a parameter scan, an ensemble optimization) on a plain
:class:`concurrent.futures.ThreadPoolExecutor`. Opaque finite-difference
Jacobians use the same mechanism
(:func:`~vmex.core.parallel.finite_difference_jacobian`), and
:func:`~vmex.core.parallel.evaluate_problems` takes one problem object per
member so mutable caches are never shared. The recipe is
:doc:`/howto/parallel-ensembles`.

``workers=None`` resolves to the smaller of the item count and the CPUs
available to the process (including Linux affinity and common Slurm, PBS, SGE
and LSF allocations); ``workers=N`` overrides it and ``workers=1`` gives a
serial baseline. A single implicit-gradient optimization has no independent
equilibria to distribute, and JAX reports one CPU *device* for the whole host
(an XLA backend with its own threads, not one core), so VMEX exposes no
``workers`` argument on that path.

.. list-table:: Parallel and placement controls
   :header-rows: 1
   :widths: 30 25 45

   * - workload
     - default
     - control
   * - one forward or implicit solve
     - XLA CPU threading, automatic placement
     - ``device="cpu"`` or ``device="gpu"`` (or a concrete ``jax.Device``)
   * - finite-difference Jacobian
     - ``workers=None``
     - ``workers=N``
   * - parameter scan or ensemble
     - ``workers=None``
     - ``workers=N`` and one problem object per member
   * - several GPUs
     - no automatic multi-GPU use
     - place independent members on explicit devices

**Limits.** Scaling is sub-linear: ensemble workers and the XLA threads inside
each solve share the same cores. An ensemble finishes no sooner than its
slowest member, so members of similar size and shape (a scan at fixed
resolution sharing one executable) parallelize best. The reverse pass of the
implicit adjoint dispatches many small operations that hold the GIL, so a
``value_and_grad`` ensemble overlaps its forward solves much better than its
gradients.

**Rejected mechanisms.** ``pmap`` over forced host CPU devices
(``--xla_force_host_platform_device_count``) splits the cores into artificial
devices, starving each solve's XLA threading and serializing the host
callbacks. ``vmap`` over the host callback cannot handle ensembles of
different shapes and degenerates to a vectorized host loop for equal shapes.

**Multi-GPU.** JAX's multi-device model is explicit sharding; it does not turn
a host callback or an unsharded solve into a multi-GPU program. Placing
independent ensemble members on distinct devices works today through explicit
``device=`` arguments. Sharding one large traced solve (``mode="jit"``) across
devices with ``jax.sharding`` is future work and has not been measured.

Naming conventions
------------------

Community-expected VMEC names are kept (``ns, mpol, ntor, nfp, lasym, iotaf,
presf, rmnc, zmns, lmns, bmnc, ...``). Internal Fortran temporaries get
descriptive names (``sqrt_g`` rather than ``gsqrt``); every module docstring
cross-references the VMEC2000 source it ports.
