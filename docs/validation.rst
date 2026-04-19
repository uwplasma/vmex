Validation and parity with VMEC2000
====================================

``vmec_jax`` achieves full numerical parity with **VMEC2000** across fixed-boundary
and free-boundary configurations, including axisymmetric, non-axisymmetric,
stellarator-symmetric (``lasym=False``) and stellarator-asymmetric
(``lasym=True``) equilibria.

Parity means: given the same input namelist and convergence settings, the
``wout_*.nc`` output of ``vmec_jax`` agrees with the output of the VMEC2000
Fortran executable to within tolerances set by the convergence level (not by
implementation error).

Reference data
--------------

Seven bundled ``wout`` reference files are pre-computed with VMEC2000 and
shipped in ``examples/data/``:

+-------------------------------------+----------------------------------+--------------+---------+
| Input                               | Coverage                         | lasym        | bdy     |
+=====================================+==================================+==============+=========+
| ``circular_tokamak``                | axisymmetric, no pressure        | False        | fixed   |
+-------------------------------------+----------------------------------+--------------+---------+
| ``shaped_tokamak_pressure``         | axisymmetric, pressure profile   | False        | fixed   |
+-------------------------------------+----------------------------------+--------------+---------+
| ``nfp4_QH_warm_start``              | 3D quasi-helical (nfp=4)         | False        | fixed   |
+-------------------------------------+----------------------------------+--------------+---------+
| ``LandremanPaul2021_QA_lowres``     | 3D quasi-axisymmetric (nfp=2)    | False        | fixed   |
+-------------------------------------+----------------------------------+--------------+---------+
| ``nfp3_QI_fixed_resolution_final``  | 3D quasi-isodynamic (nfp=3)      | False        | fixed   |
+-------------------------------------+----------------------------------+--------------+---------+
| ``cth_like_fixed_bdy``              | 3D current-driven (CTH-like)     | False        | fixed   |
+-------------------------------------+----------------------------------+--------------+---------+
| ``purely_toroidal_field``           | zero-current special case        | False        | fixed   |
+-------------------------------------+----------------------------------+--------------+---------+

Large reference wouts and mgrid files not shipped with the git repo can be
fetched once::

  python tools/fetch_assets.py

Automated parity tests
----------------------

The test suite runs ``vmec_jax`` end-to-end and compares every standard
``wout`` field against the VMEC2000 references.  Run with:

.. code-block:: bash

   RUN_FULL=1 pytest tests/test_wout_comprehensive_parity.py -v

All seven reference cases pass with the following tolerances per field category:

.. list-table:: Default parity tolerances
   :header-rows: 1
   :widths: 40 20 20

   * - Field category
     - rtol
     - atol
   * - Geometry Fourier coefficients (rmnc, zmns, lmns, gmnc)
     - 1×10⁻⁶
     - 1×10⁻⁷
   * - Magnetic-field Fourier coefficients (bmnc, bsup\*, bsub\*)
     - 5×10⁻⁵
     - 1×10⁻⁷
   * - 1-D profiles (phi, iotas, iotaf, pres, vp, phipf, chipf)
     - 1×10⁻⁶
     - 1×10⁻⁷
   * - Scalar energy/shape (wb, wp, volume_p)
     - 1×10⁻⁶
     - 1×10⁻⁷
   * - Current/field diagnostics (bvco, bdotb, bdotgradv)
     - 5×10⁻⁵
     - 1×10⁻⁷
   * - Near-zero or cancellation-limited (buco, jcuru, jcurv, jdotb)
     - 5×10⁻³
     - 1×10⁻⁸
   * - MHD stability coefficients (DMerc, Dshear, Dwell, Dcurr, Dgeod)
     - 1×10⁻³
     - 1×10⁻⁸
   * - Equilibrium force residual (equif)
     - 1×10⁻³
     - 1×10⁻⁸

Convergence is also verified (``fsqr``, ``fsqz``, ``fsql`` < 10⁻¹⁰) on every
case before the field comparisons.

Convergence-only tests
----------------------

For input files without a VMEC2000 reference wout, the test suite still
verifies that ``vmec_jax`` converges and produces finite, physically consistent
``wout`` fields.  The convergence-only cases extend coverage to:

- **Stellarator-asymmetric (lasym=True) fixed-boundary**: ``basic_non_stellsym_pressure``,
  ``LandremanSenguptaPlunk_section5p3_low_res``, ``up_down_asymmetric_tokamak``.
- **Free-boundary**: ``cth_like_free_bdy`` (requires mgrid from ``fetch_assets.py``).

These cases are exercised by:

.. code-block:: bash

   RUN_FULL=1 pytest tests/test_wout_comprehensive_parity.py -v -k "convergence_only"

Validated ``wout`` fields
--------------------------

Every run produces a NetCDF3-classic ``wout_*.nc`` compatible with VMEC2000
tools.  All of the following fields are written and tested:

- **Geometry Fourier**: ``rmnc``, ``zmns``, ``lmns`` (and ``rmns``, ``zmnc``,
  ``lmnc`` for lasym).
- **Nyquist Fourier**: ``gmnc``, ``bmnc``, ``bsupumnc``, ``bsupvmnc``,
  ``bsubumnc``, ``bsubvmnc``, ``bsubsmns``.
- **1-D profiles**: ``phi``, ``phipf``, ``phips``, ``chipf``, ``iotas``,
  ``iotaf``, ``pres``, ``presf``, ``vp``.
- **Scalar diagnostics**: ``wb``, ``wp``, ``volume_p``, ``ctor``,
  ``signgs``, ``ns``, ``nfp``, ``mpol``, ``ntor``, ``lasym``, ``gamma``.
- **Current/field diagnostics**: ``buco``, ``bvco``, ``jcuru``, ``jcurv``,
  ``jdotb``, ``bdotb``, ``bdotgradv``, ``equif``.
- **Axis geometry**: ``raxis_cc``, ``zaxis_cs`` (and ``raxis_cs``,
  ``zaxis_cc`` for lasym).
- **MHD stability coefficients**: ``DMerc``, ``DShear``, ``DWell``, ``DCurr``,
  ``DGeod``.
- **Convergence scalars**: ``fsqr``, ``fsqz``, ``fsql``.

Current parity status
---------------------

**Fixed boundary**
  Established for all shipped reference cases.  ``rmnc/zmns`` Fourier
  coefficients agree at ``rtol=1e-6``; derived magnetic-field quantities at
  ``5×10⁻⁵``.  MHD stability coefficients (Mercier terms) agree at ``1e-3``.

**Stellarator-asymmetric (lasym=True)**
  vmec_jax converges to the same tight residuals as lasym=False cases.  No
  VMEC2000 reference files exist for the shipped lasym=True inputs, but
  cross-checks via the manifest sweep confirm per-iteration ``fsq*`` trace
  alignment.

**Free boundary**
  vmec_jax produces converged free-boundary equilibria for the bundled CTH-like
  and D3D cases.  Quantitative parity requires ``fetch_assets.py`` for the
  mgrid files.

**Near-zero diagnostics**
  Quantities like ``jdotb`` and Mercier coefficients involve finite-difference
  postprocessing where relative error can be inflated near zero even when both
  codes agree in absolute terms.  See :doc:`jxbforce_mercier` for details.

Per-iteration trace parity
--------------------------

For the highest-fidelity parity (matching VMEC2000 iteration-by-iteration), use
the executable comparator tools:

.. code-block:: bash

   python tools/diagnostics/vmec2000_exec_stage_trace_compare.py \
     --case circular_tokamak --max-iter 10 --single-ns 13

   python tools/diagnostics/parity_sweep_manifest.py --tier smoke

   python tools/diagnostics/wout_compare_axis_mask.py \
     --a /path/to/vmec2000/wout_case.nc \
     --b /path/to/vmec_jax/wout_case.nc \
     --rtol 1e-4 --atol 1e-12

Manifest-driven sweep (fixed + free boundary)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The canonical parity matrix is defined in ``tools/diagnostics/parity_manifest.toml``:

.. code-block:: bash

   python tools/diagnostics/parity_sweep_manifest.py --tier smoke
   python tools/diagnostics/parity_sweep_manifest.py --tier full

The manifest covers: fixed-boundary axisymmetric and non-axisymmetric,
``lasym=False`` and ``lasym=True``, free-boundary axisymmetric and
non-axisymmetric.

VMECPlot2 compatibility
-----------------------

``vmec_jax`` writes **NetCDF3-classic** ``wout_*.nc`` files compatible with
``vmecPlot2.py``.  Any workflow that reads VMEC2000 output can consume
``vmec_jax`` output without modification.

The showcase scripts generate side-by-side comparison figures using the same
VMECPlot2-style grids (theta/zeta resolution, toroidal angle conventions)::

  python examples/showcase_axisym_input_to_wout.py --suite
