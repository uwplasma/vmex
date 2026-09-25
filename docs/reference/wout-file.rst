Output files (wout and mout)
============================

Toroidal solves write a VMEC2000-compatible ``wout_*.nc``; open straight-axis
mirror solves write a mirror-native ``mout_*.nc``.  The two schemas are
separate: mirror data are never encoded as a toroidal WOUT.

wout files
----------

:mod:`vmex.core.wout` writes the VMEC2000 netCDF schema with the names,
dimensions, dtypes and units expected by simsopt, booz_xform and other
VMEC-ecosystem tools.  A declared variable may be fill-valued where its
producer is not implemented; see :doc:`vmec2000-compatibility`.

Build a wout from a solve in one call, then write it:

.. code-block:: python

   import vmex as vj

   result = vj.solve_multigrid(inp)
   wout = vj.wout_from_result(inp, result)
   vj.write_wout("wout_case.nc", wout)

:func:`~vmex.core.wout.wout_from_result` takes the residuals, iteration
count, convergence flag and NESTOR tables (``result.vacuum``, ``None`` for a
fixed-boundary solve) from ``result``; any
:func:`~vmex.core.wout.wout_from_state` keyword passed to it overrides the
value taken from ``result``.  It exports ``result.state``, not
``result.polished_state``; pass ``state=`` to export another state.
:func:`~vmex.core.wout.wout_from_state` is the explicit form, with every
field passed by name.  :func:`~vmex.core.wout.read_wout` reads a file back
into :class:`~vmex.core.wout.WoutData`.

Unit conventions (applied on write, as in ``wrout.f``):

- ``presf, pres, mass, jcuru, jcurv, ctor`` are divided by :math:`\mu_0`;
- ``phipf, chipf`` are multiplied by :math:`2\pi\,\mathrm{signgs}`;
- ``q_factor = 1 / iotaf``;
- ``lmns`` is on the half mesh; ``bsubsmns`` on the full mesh.

Scalars
~~~~~~~

``version_``, ``input_extension``, ``mgrid_file``, ``pcurr_type``,
``pmass_type``, ``piota_type``, ``wb``, ``wp``, ``gamma``, ``rmax_surf``,
``rmin_surf``, ``zmax_surf``, ``nfp``, ``ns``, ``mpol``, ``ntor``, ``mnmax``,
``mnmax_nyq``, ``niter``, ``itfsq``, ``lasym``, ``lrecon``, ``lfreeb``,
``lmove_axis``, ``lrfp``, ``ier_flag``, ``aspect``, ``betatotal``,
``betapol``, ``betator``, ``betaxis``, ``b0``, ``rbtor0``, ``rbtor``,
``signgs``, ``IonLarmor``, ``volavgB``, ``ctor``, ``Aminor_p``, ``Rmajor_p``,
``volume_p``, ``ftolv``, ``fsql``, ``fsqr``, ``fsqz``, ``nextcur``,
``extcur(:)``, ``mgrid_mode``.

Mode arrays, axis and profile inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``xm``, ``xn``, ``xm_nyq``, ``xn_nyq`` (with ``xn = n * nfp``);
``raxis_cc``, ``zaxis_cs`` (plus ``raxis_cs``, ``zaxis_cc`` when ``lasym``);
``am``, ``ac``, ``ai`` and the spline tables ``am_aux_s/f``, ``ac_aux_s/f``,
``ai_aux_s/f``.

Radial profiles
~~~~~~~~~~~~~~~

Full mesh: ``iotaf``, ``q_factor``, ``presf``, ``phi``, ``phipf``, ``chi``,
``chipf``, ``jcuru``, ``jcurv``, ``jdotb``, ``bdotb``, ``bdotgradv``,
``DMerc``, ``DShear``, ``DWell``, ``DCurr``, ``DGeod``, ``equif``.

Half mesh: ``iotas``, ``mass``, ``pres``, ``beta_vol``, ``buco``, ``bvco``,
``vp``, ``specw``, ``phips``, ``over_r``.

Convergence history: ``fsqt(:)``, ``wdot(:)``.

Fourier tables (mode x radius)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Full mesh: ``rmnc``, ``zmns``, ``bsubsmns``, ``currumnc``, ``currvmnc``.

Half mesh: ``lmns``, and the Nyquist-resolution tables ``gmnc``, ``bmnc``,
``bsubumnc``, ``bsubvmnc``, ``bsupumnc``, ``bsupvmnc``.

When ``lasym = T``, all asymmetric partners are written (``rmns``, ``zmnc``,
``lmnc``, ``gmns``, ``bmns``, ``bsubumns``, ``bsubvmns``, ``bsubsmnc``,
``currumns``, ``currvmns``, ``bsupumns``, ``bsupvmns``).

Free-boundary variables
~~~~~~~~~~~~~~~~~~~~~~~

When ``lfreeb = T``, ``nextcur``, ``extcur``, ``curlabel`` and
``mgrid_mode`` carry the coil-group metadata of the mgrid file; ``curlabel``
uses VMEC2000's 30-character label dimension.  The NESTOR tables
(``result.vacuum``, passed as ``vacuum_output``) add the potential modes
``potsin``/``xmpot``/``xnpot`` and the four ``*_sur`` surface-field tables
with VMEC2000's Nyquist normalization; LASYM runs also write ``potcos`` and
the four sine ``*_sur`` partners.  The CLI and
:func:`~vmex.core.wout.wout_from_result` include them automatically.

VMEX extension
~~~~~~~~~~~~~~

Two names that VMEC2000 readers may ignore:

- ``vmex_diagnostics_schema = 1`` identifies the extension.
- ``vmex_trapped_fraction`` is the effective trapped-particle fraction
  :math:`f_t` on the full normalized-toroidal-flux mesh.

:math:`f_t` is computed from the converged half-mesh :math:`|B|` and
:math:`\sqrt{g}` with 64-point pitch quadrature.  At the axis, VMEX keeps the
poloidal :math:`m=0` field and extrapolates it linearly in :math:`s`
(finite-radius poloidal modes vanish there by regularity), so a constant
on-axis field gives zero trapped fraction while a QI :math:`B_0(\varphi)`
with a finite mirror ratio stays finite.  Symmetric and LASYM equilibria use
their own full-surface angular grids.

Reading an older WOUT sets ``vmex_diagnostics_schema`` to zero and
``vmex_trapped_fraction`` to ``None``; rewriting that object does not add the
extension.

Parity with VMEC2000
~~~~~~~~~~~~~~~~~~~~

WOUT parity against representative VMEC2000 golden runs is asserted per
variable with combined relative and absolute tolerances, with a looser bound
for ``currumnc``/``currvmnc``.  This is not a parity claim for fill-valued or
untested modes.  See :doc:`performance` and :doc:`vmec2000-compatibility`.

mout files
----------

Open straight-axis mirror solves write ``mout_*.nc`` NetCDF files
(:mod:`vmex.mirror.output`) that store physical-grid arrays, so a solved
equilibrium can be plotted or inspected without rebuilding the solver
objects.  The closed stellarator-mirror hybrid has no mout schema.

- :func:`vmex.mirror.output.mout_from_result` builds a
  :class:`~vmex.mirror.output.MoutData` from a fixed- or free-boundary
  result.
- :func:`vmex.mirror.output.write_mout` writes it (schema-stamped);
  :func:`vmex.mirror.output.read_mout` reads it and rejects any schema other
  than ``vmex.mirror.model.MIRROR_OUTPUT_SCHEMA``.
- ``vmex --plot mout_*.nc`` renders the mirror figure set
  (:doc:`/howto/plot-diagnostics`).

Arrays
~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 22 54

   * - Variable
     - Shape
     - Meaning
   * - ``s``
     - ``(ns,)``
     - radial flux-label grid
   * - ``theta``
     - ``(ntheta,)``
     - poloidal angle grid
   * - ``xi``
     - ``(nxi,)``
     - nonperiodic axial coordinate grid
   * - ``z``
     - ``(nxi,)``
     - axial position of each ``xi`` node
   * - ``boundary_radius``
     - ``(ntheta, nxi)``
     - LCFS radius
   * - ``radius_scale``
     - ``(ns, ntheta, nxi)``
     - nested-surface radial scale (defines the geometry)
   * - ``lambda_stream``
     - ``(ns, ntheta, nxi)``
     - stream function
   * - ``mod_b``
     - ``(ns, ntheta, nxi)``
     - ``|B|`` on the radial Gauss cells of the magnetic-energy functional
   * - ``b_xyz``
     - ``(ns, ntheta, nxi, 3)``
     - Cartesian field samples (for field-line direction)
   * - ``pressure``
     - ``(ns, ntheta, nxi)``
     - isotropic pressure
   * - ``history``
     - ``(iterations, k)``
     - solver residual history
   * - ``coil_xyz``
     - ``(ncoil, npoint, 3)``
     - optional coil polylines for plotting

Scalars
~~~~~~~

``ftol``, ``iterations``, ``converged``, ``mass_scale``,
``variational_max`` (defines nonlinear convergence), ``normal_stress_rms``,
``b_normal_rms``, ``staggered_weak_max`` (independent staggered-weak
residual on the energy quadrature), ``pointwise_force_rms`` (reconstructed
``J x B - grad(p)`` norm; a diagnostic, not the nonlinear ``ftol``),
``normalized_divergence_rms`` (checks the field representation),
``message``, and the ``schema`` attribute.

Mirror free-boundary restart files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Axisymmetric free-boundary beta scans can write one compressed ``.npz``
restart per beta point (schema ``vmex.mirror.free_boundary_restart/3``):

- :class:`vmex.mirror.output.FreeBoundaryRestart` holds the
  coefficient-native ``boundary``, ``plasma_state`` and calibrated
  ``mass_scale``.
- :func:`vmex.mirror.output.save_free_boundary_restart` writes it atomically
  as data only (``boundary_radius_coefficients``, ``radius_coefficients``,
  ``lambda_coefficients``, ``mass_scale``).
- :func:`vmex.mirror.output.load_free_boundary_restart` rejects any other
  schema and checks the coefficient shapes against the target
  discretization.

The boundary-integral potential is not stored; it is recomputed on load
because the boundary moves at every continuation point.  The scan workflow
is in :doc:`/howto/mirror-machines`.
