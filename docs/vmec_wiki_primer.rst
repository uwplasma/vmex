VMEC wiki primer (for vmec-jax parity)
======================================

This page condenses the core VMEC wiki material into a single, parity-focused
reference. It is meant to be **pedagogical**, linking VMEC input/output concepts
directly to the ``vmec_jax`` implementation.

Primary sources:

- VMEC overview and theory: https://princetonuniversity.github.io/STELLOPT/VMEC.html
- VMEC input namelist tutorial: https://princetonuniversity.github.io/STELLOPT/Tutorial%20VMEC%20Input%20Namelist.html
- VMEC input variable table (v8.47): https://princetonuniversity.github.io/STELLOPT/VMEC%20Input%20Namelist%20%28v8.47%29.html

Overview and theory (VMEC)
--------------------------

VMEC seeks toroidal MHD equilibria by minimizing the total plasma potential
energy under force balance and magnetic constraints. The force balance
conditions are

.. math::

   \mathbf{F} = -\mathbf{j} \times \mathbf{B} + \nabla p = 0,\qquad
   \nabla \times \mathbf{B} = \mu_0 \mathbf{j},\qquad
   \nabla \cdot \mathbf{B} = 0.

The VMEC formulation is variational: the total energy is written in flux
coordinates, Fourier-expanded in angular coordinates, and minimized by a
Richardson/steepest-descent update. The magnetic field is represented in a
contravariant form using a field-line straightening angle
:math:`\theta^*=\theta+\lambda(\rho,\theta,\zeta)`, which enforces flux
constraints by construction. See the VMEC theory page for the full derivation
and coordinate definitions.

Input file structure (INDATA)
-----------------------------

VMEC reads a Fortran namelist named ``INDATA`` from ``input.<case>``. The
namelist can omit parameters (defaults apply), but unknown names are rejected.

Runtime control parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

Key runtime inputs from the VMEC tutorial page:

- ``DELT`` controls the Richardson update blending/step size (0–1).
- ``NITER`` is the max iteration count for a given radial resolution; VMEC will
  run up to *twice* this value if convergence is not reached.
- ``NSTEP`` controls how often diagnostics are printed to screen and ``threed1``.
- ``TCON0`` sets the constraint-force weight (values >1 are treated as 1).
- ``NS_ARRAY``, ``FTOL_ARRAY``, and ``NITER_ARRAY`` define the multigrid
  sequence: number of radial surfaces per stage, the tolerance per stage, and
  the maximum iterations per stage.
- ``LWOUTTXT`` switches text vs NetCDF ``wout`` output (NetCDF is default when
  available).

Grid and symmetry parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Important grid and symmetry inputs include:

- ``LASYM`` (stellarator symmetry switch).
- ``NFP`` (toroidal field periods).
- ``MPOL``/``NTOR`` (poloidal/toroidal mode limits).
- ``PHIEDGE`` (enclosed toroidal flux; also used to scale boundary amplitude).
- ``NTHETA``/``NZETA`` control angular grid resolution (with defaults tied to
  ``MPOL``/``NTOR``; in free-boundary mode, ``NZETA`` must match the mgrid file).

Free-boundary parameters
~~~~~~~~~~~~~~~~~~~~~~~~

For free-boundary runs, VMEC uses:

- ``LFREEB`` to enable free-boundary mode.
- ``MGRID_FILE`` for the vacuum field grid (MAKEGRID output).
- ``EXTCUR`` for external coil currents.
- ``NVACSKIP`` to control how often the vacuum solve is updated.

Profiles (pressure / iota / current)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VMEC represents input profiles as low-order polynomials in the normalized flux
coordinate :math:`s` (unless advanced profile functions are enabled):

- ``AM`` coefficients define mass/pressure; with ``GAMMA=0`` the profile is
  interpreted as a pressure profile.
- ``BLOAT``, ``SPRES_PED``, and ``PRES_SCALE`` shape/scale the pressure profile.
- ``NCURR`` selects whether the rotational transform (``AI``) or toroidal
  current (``AC``) is specified; ``AC_FORM`` chooses the current profile form.

Magnetic axis and boundary
~~~~~~~~~~~~~~~~~~~~~~~~~~

VMEC requires an *initial* magnetic axis guess via Fourier coefficients
``RAXIS`` and ``ZAXIS`` (torodial harmonics). The boundary is specified by
Fourier coefficients:

- ``RBC``/``RBS`` for the R boundary coefficients,
- ``ZBC``/``ZBS`` for the Z boundary coefficients.

For stellarator-symmetric cases, only ``RBC`` and ``ZBS`` are required.
Sign conventions in the boundary coefficients affect the rotation direction:
negative ``ZBS(0,1)`` implies a clockwise rotation in cylindrical coordinates,
and a negative iota is a strong indicator that the boundary sign needs to be
flipped. VMEC indexes mode arrays with **toroidal index first**, then poloidal,
and for full-torus plots the toroidal angle is multiplied by ``NFP``.

Preconditioning and additional inputs
-------------------------------------

The VMEC input variable table (v8.47) includes preconditioning controls such as
``PRECON_TYPE`` and thresholds like ``PREC2D_THRESHOLD``, along with extensive
diagnostic and reconstruction settings. The full table is large; consult the
official VMEC input namelist page for the complete list of variables and
descriptions.

Outputs and diagnostics
-----------------------

VMEC writes diagnostic output to screen and ``threed1`` and emits core files
including ``wout``, ``jxbout``, and ``mercier``. When ``LDIAGNO`` is enabled,
``diagno_in`` is also generated. The ``wout`` file contains Fourier coefficients
for surfaces (e.g., ``rmnc``/``zmns``) and profiles, plus mode index arrays
(``xm``/``xn``). VMEC stores data over one field period; to plot a full torus,
the toroidal mode index is scaled by ``NFP``.

Mapping into vmec-jax
---------------------

``vmec_jax`` mirrors the VMEC execution model for fixed-boundary, axisymmetric
cases:

- The multigrid sequence is controlled by ``NS_ARRAY``, ``FTOL_ARRAY``, and
  ``NITER_ARRAY`` (mirrored by ``multigrid_use_input_niter`` in the Python API).
- ``DELT`` is used as the base timestep for Richardson updates.
- ``TCON0`` and constraint-force normalization follow VMEC conventions.
- Axis/boundary Fourier coefficients are initialized using the same mode
  conventions (toroidal index first, then poloidal).

For parity work, treat the VMEC wiki pages above as the canonical reference for
input semantics, coordinate conventions, and output meanings. When in doubt,
compare against ``xvmec2000`` traces (``threed1``) and ``wout`` output.
