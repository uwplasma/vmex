Input file and VMEC2000 compatibility
=====================================

This page lists every input VMEX reads, its default, and what VMEX does with
it relative to VMEC2000.  A control is called *implemented* only when it
reaches the solver path; validation evidence is stated separately.  The
mechanisms behind the controls are explained in :doc:`/explanation/iteration`,
:doc:`/explanation/variational-problem` and :doc:`/explanation/nestor-vacuum`.

The VMEC2000 reference is the STELLOPT tree: ``vmec_input.f`` (the
``&INDATA`` namelist and defaults), ``readin.f`` (post-read normalization and
preconditioner selection), ``runvmec.f``/``evolve.f``/``vmec.f`` (multigrid,
convergence and WOUT policy), ``NESTOR_vacuum`` (vacuum solve), and
``wrout.f``/``jxbforce.f`` (WOUT variables).

Input formats
-------------

:meth:`vmex.core.input.VmecInput.from_file` auto-detects two formats:

- the VMEC2000 ``&INDATA`` Fortran namelist (``input.<case>``), and
- structured JSON (``.json`` suffix or leading ``{``) with the same names.

Both round-trip: ``VmecInput.to_json`` and ``VmecInput.to_indata`` write
decks that parse back to the same input.  VMEC2000 defaults and ``readin.f``
normalizations are applied on construction (the :mod:`vmex.core.input`
docstring has the exact rules).  Unknown names and active unsupported
physics fail before setup.

``VmecInput`` has value semantics: its arrays are owned and read-only, so an
in-place edit cannot invalidate a compiled solve.  Copy the array, edit it,
and build the new deck with ``dataclasses.replace``.

Fortran namelist semantics
~~~~~~~~~~~~~~~~~~~~~~~~~~

INDATA assignments are replayed in source order into the initialized
``vmec_input.f`` arrays, as a Fortran namelist reader does.  A later short
dense assignment replaces only the elements it supplies:

.. code-block:: fortran

   FTOL_ARRAY = 1e-6, 1e-11, 1e-30, 1e-30
   FTOL_ARRAY = 1e-7, 1e-7

leaves entries 3 and 4 at ``1e-30``.  Indexed and dense writes override one
another in textual order.

- An indexed designator followed by several values is a starting-element
  assignment: ``APHI(1)=1,0`` writes elements 1 and 2.
- Array sections use inclusive Fortran bounds, first subscript fastest, so
  ``RBC(-6:6,0)=...`` equals the 13 scalar assignments.  Omitted limits
  (``RBC(:,0)``, ``RBC(-6:,0)``) resolve from the declared VMEC2000 bounds.
  Bounds, rank and value count are checked before setup.
- Repeat syntax works for values (``4*0.0``) and nulls (``3*``).  A null
  field advances the position without replacing the current value, so
  ``APHI=,0.5`` keeps the default first coefficient.
- Single- and double-quoted strings may contain spaces, commas, ``!`` or
  ``=``; a doubled delimiter is a literal quote.

Structured JSON
~~~~~~~~~~~~~~~

Keys are the lower-case :class:`~vmex.core.input.VmecInput` field names,
which match INDATA except for the axis arrays (``raxis_c``, ``zaxis_s``,
``raxis_s``, ``zaxis_c``) and the alias ``adiabatic_index`` for ``gamma``.
Boundary coefficients are sparse ``{"n", "m", "value"}`` lists; axis arrays
are dense of length ``ntor + 1``:

.. code-block:: json

   {
     "lasym": false, "nfp": 5, "mpol": 5, "ntor": 4,
     "ns_array": [31], "ftol_array": [1e-12], "niter_array": [2000],
     "phiedge": 0.5,
     "raxis_c": [1.0, 0.1, 0.0, 0.0, 0.0],
     "zaxis_s": [0.0, 0.1, 0.0, 0.0, 0.0],
     "rbc": [{"n": 0, "m": 0, "value": 1.0}, {"n": 0, "m": 1, "value": 0.3}],
     "zbs": [{"n": 0, "m": 1, "value": 0.3}]
   }

``free_boundary_method="nestor"`` is accepted; ``only_coils`` and ``biest``
are different boundary models and are rejected.  Unknown keys are errors.

DESC equilibria
~~~~~~~~~~~~~~~

The CLI also accepts a DESC equilibrium as the input argument: a DESC text
deck, an HDF5 file (``.h5``/``.hdf5``, needs ``h5py``) or a pickle
(``.pkl``/``.pickle``).  DESC itself is not imported, and pickles are read
with an unpickler that never runs their code.
:func:`vmex.core.desc.write_desc_input` converts the boundary and profiles to
an ``input.<name>`` INDATA deck next to the source (or in ``--outdir``),
which the ordinary solve then runs.

``--desc-tol`` (default ``0.01``, allowed range ``0`` to ``0.01``) bounds the
boundary position and angular derivatives discarded when the Fourier
resolution is truncated, relative to the RMS boundary shape; ``0`` keeps
every nonzero mode.  One extra poloidal (and, for ``ntor > 0``, toroidal)
harmonic is then added for the interior solve.

Status vocabulary
-----------------

``implemented``
   The VMEX solver path consumes the control.  This alone does not claim
   numerical parity for every equilibrium.

``parity-regressed``
   Tests compare the trajectory, state or WOUT quantity with VMEC2000 on at
   least one representative case.

``deliberate divergence``
   Same mathematical purpose, disclosed different algorithm or extension.

``partial``
   A documented subset is implemented; the rest fails explicitly where
   dropping it could change a result.

``accepted no-op``
   Cannot change equilibrium physics; kept so legacy decks parse.  VMEX does
   not produce the requested legacy artifact.

``accepted with warning``
   The equations are unaffected, but VMEX warns that a requested auxiliary
   artifact is not produced.

``rejected when active``
   Recognized, but raises :class:`~vmex.core.input.UnsupportedInputModeError`
   before setup.  An ordinary equilibrium is never substituted.

INDATA variables
----------------

Symmetry and resolution
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 11 16 51

   * - Variable
     - Default
     - Status
     - VMEX behavior
   * - ``LASYM``
     - ``F``
     - implemented
     - Non-stellarator-symmetric mode; enables the ``RBS/ZBC`` and
       ``*mns/*mnc`` partners.
   * - ``NFP``
     - 1
     - implemented
     - Number of field periods.
   * - ``MPOL``
     - 6
     - implemented
     - Poloidal modes ``m = 0 .. MPOL-1``.
   * - ``NTOR``
     - 0
     - implemented
     - Toroidal modes ``n = -NTOR .. NTOR``.
   * - ``NTHETA`` / ``NZETA``
     - 0
     - implemented
     - Angular grid points; 0 selects the VMEC default.  Explicit values
       below ``2*MPOL+6`` / ``2*NTOR+4`` stay legal for VMEC2000 parity but
       can alias nonlinear force products (see the diagnostic below).  With a
       tabulated free-boundary field, an ``NZETA`` that does not divide the
       mgrid's planes per period raises before iteration one
       (``mgrid_mod.f``'s ``ier_flag = 9``); ``NZETA = 0`` picks the smallest
       divisor at or above ``2*NTOR+4``, where VMEC2000 simply rejects (see
       :func:`vmex.core.freeboundary.free_boundary_resolution`).

Multigrid ladder and stepping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 11 16 51

   * - Variable
     - Default
     - Status
     - VMEX behavior
   * - ``NS_ARRAY``
     - ``[31]``
     - parity-regressed
     - Radial surfaces per stage.  The active ladder ends at the first
       nonpositive or decreasing entry; equal entries rerun.  The old
       ``NS_ARRAY(1)=0`` form expands through ``NSIN`` to ``[NSIN, 31]``.
       Fixed and free boundary use the same ladder.
   * - ``FTOL_ARRAY``
     - ``[1e-10]``
     - implemented
     - Force tolerance per stage (converged when ``FSQR``, ``FSQZ`` and
       ``FSQL`` are all below it).  An explicit zero first entry generates
       ``readin.f``'s geometric ladder from ``1e-8`` to ``FTOL``.
   * - ``FTOL``
     - 1e-10
     - implemented
     - Single-grid tolerance when ``FTOL_ARRAY(1)=0`` and the final target
       of the generated ladder.
   * - ``NITER_ARRAY`` / ``NITER``
     - ``[100]``
     - implemented
     - Iteration cap per stage.  Scalar ``NITER`` fills the array only when
       every entry is still ``-1`` (VMEC2000's fallback).  A ``-1`` left in a
       partially assigned array runs one ``eqsolve`` pass before the limit
       check, as in VMEC2000.
   * - ``DELT``
     - 1.0
     - implemented
     - Initial time step.  After VMEC2000's 75-Jacobian-reset condition, the
       driver retries the stage from its best finite checkpoint with a
       smaller ``DELT`` (``jacobian_retries=2`` by default); set
       ``jacobian_retries=0`` / ``--jacobian-retries 0`` for VMEC2000's
       immediate stop.
   * - ``TCON0``
     - 1.0
     - implemented
     - Constraint-force (spectral-condensation) multiplier; see
       `LASYM constraint scaling (tcon)`_.
   * - ``NSTEP``
     - 10
     - implemented
     - Iterations between progress prints.
   * - ``APHI``
     - ``[1, 0, ...]``
     - implemented, with a validity gate
     - Radial-flux map ``Phi(x) = sum_i APHI(i)*x**i`` up to the edge-flux
       normalization.  If ``Phi'(s)`` changes sign inside ``s`` in
       ``[0, 1]`` the ``s -> Phi`` map folds, and VMEX raises a typed input
       error naming the interval (VMEC2000 runs such decks and can write a
       WOUT with NaN residuals or negative pressure).  Tangential zeros and
       zeros at ``s = 0`` or ``s = 1`` are allowed.
   * - ``PHIEDGE``
     - 1.0
     - implemented
     - Total enclosed toroidal flux [Wb].
   * - ``TIME_SLICE``
     - 0
     - implemented
     - Printed in the run header; no effect on the equations.
   * - ``LFORBAL``
     - ``F``
     - implemented
     - Replaces the ``m=1, n=0`` R/Z forces with VMEC2000's non-variational
       flux-averaged force balance (``fbal.f``), using full-mesh ``chipf``
       reconstructed from half-mesh ``chips`` by ``add_fluxes.f90`` for both
       ``NCURR`` modes; WOUT uses the same reconstruction.  See
       `Pressureless current-free vacuum limit`_.
   * - ``LMOVE_AXIS``
     - ``T``
     - implemented
     - Allows the first-pass ``irst=4`` axis re-guess when the first force
       sum exceeds the VMEC2000 threshold.  A missing or all-zero axis is not
       pre-inferred: VMEX follows VMEC2000's zero-axis first pass and one
       ``eqsolve.f`` recovery transfer.  Recorded in WOUT.
   * - ``LFULL3D1OUT``
     - ``F``
     - accepted no-op
     - Does not gate the WOUT.  An NITER-exhausted run writes its WOUT with
       ``ier_flag = 2`` either way, as ``fileout.f`` does; the threed1 file
       it requests is not produced.
   * - ``RESTART_WOUT``
     - ``''``
     - VMEX extension
     - Hot restart from the named ``wout_*.nc`` (relative to the deck unless
       absolute); the CLI ``--restart`` flag overrides it.  See
       :doc:`/howto/restart-from-previous-run`.
   * - ``PRE_NITER``
     - —
     - rejected when active with 2-D GMRES
     - VMEC2000's post-activation iteration-budget change is not
       implemented.
   * - ``MAX_MAIN_ITERATIONS``
     - —
     - rejected above 1
     - Use an explicit hot restart instead of extra ``NITER`` blocks.
   * - ``LGIVEUP`` / ``FGIVEUP``
     - —
     - rejected when ``LGIVEUP=T``
     - VMEC2000's early stop between poorly converged stages is not
       implemented.
   * - ``OMP_NUM_THREADS``
     - —
     - accepted no-op
     - JAX/XLA owns threading; see :doc:`/explanation/architecture`.

Profiles
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 11 16 51

   * - Variable
     - Default
     - Status
     - VMEX behavior
   * - ``PMASS_TYPE``
     - ``power_series``
     - implemented
     - Pressure profile type, e.g. ``power_series``, ``two_power``,
       ``two_power_gs``, ``two_Lorentz``, ``gauss_trunc``, ``rational``,
       ``cubic_spline``, ``akima_spline``, ``line_segment``, ``pedestal``
       (full list in :mod:`vmex.core.profiles`).
   * - ``AM`` / ``AM_AUX_S`` / ``AM_AUX_F``
     - zeros / —
     - implemented
     - Pressure coefficients (dense, indices 0..20) / spline knots and values.
   * - ``PRES_SCALE``
     - 1.0
     - implemented
     - Pressure scale factor [Pa].
   * - ``GAMMA``
     - 0.0
     - implemented
     - Adiabatic index (JSON alias ``adiabatic_index``).
   * - ``SPRES_PED``
     - 1.0
     - implemented
     - Pressure pedestal location in ``s``.
   * - ``BLOAT``
     - 1.0
     - implemented
     - Profile-argument expansion factor.
   * - ``NCURR``
     - 0
     - implemented
     - 0: prescribed iota (``AI``); 1: prescribed toroidal current (``AC``).
   * - ``PCURR_TYPE``
     - ``power_series``
     - implemented
     - Current profile type; ``*_i`` forms prescribe :math:`I(s)`, ``*_ip``
       forms :math:`I'(s)`.  One of ``power_series``, ``power_series_i``,
       ``two_power``, ``two_power_gs``, ``gauss_trunc``, ``sum_atan``,
       ``rational``, ``pedestal``, ``sum_cossq_s``, ``sum_cossq_sqrts``,
       ``sum_cossq_s_free``, and the ``_i``/``_ip`` spline and
       ``line_segment`` forms.  ``sum_atan``, ``rational``, ``pedestal`` and
       ``sum_cossq_*`` prescribe :math:`I(s)`.
   * - ``AC`` / ``AC_AUX_S`` / ``AC_AUX_F``
     - zeros / —
     - implemented
     - Current coefficients / spline knots and values.
   * - ``CURTOR``
     - 0.0
     - implemented
     - Total toroidal current [A].
   * - ``PIOTA_TYPE``
     - ``power_series``
     - implemented, excluding RFP
     - Iota profile type: ``power_series``, ``sum_atan``,
       ``nice_quadratic``, ``rational``, ``cubic_spline``, ``akima_spline``,
       ``line_segment``.
   * - ``AI`` / ``AI_AUX_S`` / ``AI_AUX_F``
     - zeros / —
     - implemented
     - Iota coefficients / spline knots and values.

Axis and boundary
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 11 16 51

   * - Variable
     - Default
     - Status
     - VMEX behavior
   * - ``RAXIS_CC`` / ``ZAXIS_CS``
     - zeros
     - implemented
     - Axis initial guess, cos/sin coefficients for ``n = 0 .. NTOR``.
   * - ``RAXIS_CS`` / ``ZAXIS_CC``
     - zeros
     - implemented
     - Asymmetric axis partners (``LASYM = T``).
   * - obsolete ``RAXIS`` / ``ZAXIS``
     - —
     - implemented
     - Nonzero entries override the modern names, as in
       ``read_indata_namelist``.
   * - ``RBC(n,m)`` / ``ZBS(n,m)``
     - zeros
     - implemented
     - Boundary coefficients of
       :math:`R\cos / Z\sin(m\theta - n\,\mathrm{NFP}\,\zeta)`.
   * - ``RBS(n,m)`` / ``ZBC(n,m)``
     - zeros
     - implemented
     - Asymmetric boundary partners (``LASYM = T``).
   * - ``TVOLUME`` / ``LVOLUME_RFIX``
     - —
     - rejected when active
     - Target-volume rescaling (``RESCALE_BOUNDARY``) is not ported.

Free boundary
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 11 16 51

   * - Variable
     - Default
     - Status
     - VMEX behavior
   * - ``LFREEB``
     - ``T``
     - implemented
     - Free-boundary mode; forced ``F`` when ``MGRID_FILE = 'NONE'``.
   * - ``MGRID_FILE``
     - ``'NONE'``
     - implemented
     - MAKEGRID vacuum-field file.  An unreadable file falls back to a
       fixed-boundary solve with a warning, as VMEC2000 does, and the WOUT
       records the fixed-boundary solve.  ``'DIRECT_COILS'`` tabulates an
       ESSOS coil field into an in-memory mgrid (with ``vmex --coils``).
   * - ``EXTCUR``
     - —
     - implemented
     - External coil-group currents [A].
   * - ``NVACSKIP``
     - 1
     - implemented
     - Full vacuum-solve cadence and adaptive lower bound; ``<= 0`` falls
       back to ``NFP``.
   * - ``MFILTER_FBDY`` / ``NFILTER_FBDY``
     - -1
     - implemented
     - Suppress high boundary modes in setup and in the free-boundary
       degrees of freedom.
   * - ``TRIP3D_FILE``
     - ``'NONE'``
     - rejected when active
     - A non-``NONE`` value raises ``D00E_TRIP3D_MODE_UNSUPPORTED``.

Preconditioner
~~~~~~~~~~~~~~

``PRECON_TYPE`` defaults to ``NONE`` and ``PREC2D_THRESHOLD`` to ``1e-30``.
VMEC2000's ``readin.f`` maps four strings to four different 2-D block
algorithms (``CG``, ``GMRES``, ``GMRESR``, ``TFQMR``, types 1-4); ``NONE``,
``DEFAULT`` and unknown strings keep the 1-D radial preconditioner.  VMEX
uses this contract:

.. list-table::
   :header-rows: 1
   :widths: 24 22 54

   * - ``PRECON_TYPE``
     - Status
     - Meaning
   * - ``NONE`` or ``DEFAULT``
     - implemented
     - VMEC-parity 1-D radial tridiagonal plus lambda preconditioner.  These
       disable only the optional 2-D block preconditioner, not ``scalfor``
       or lambda scaling.
   * - ``GMRES``
     - deliberate divergence
     - Exact JAX JVP of the preconditioned force, solved matrix-free by
       restarted SOLVAX GMRES.  VMEC2000 assembles a block-tridiagonal
       operator by finite differences.
   * - ``CG``, ``GMRESR``, ``TFQMR``
     - rejected when active
     - Not aliases for VMEX GMRES.
   * - any other string
     - rejected
     - Prevents a typo from selecting a solver.

``PREC2D_THRESHOLD`` is read by VMEX GMRES on the finest radial stage after
the minimum-iteration gate; :class:`~vmex.core.preconditioner_2d.Prec2DConfig`
is the Python interface.  A preconditioner changes the update, never the
equilibrium root: convergence is judged on the physical ``FSQR``, ``FSQZ``,
``FSQL``.

The radial solve replays VMEC2000 ``serial_tridslv``'s modified-pivot test at
the same ``1e-8`` relative threshold and also checks a normwise backward
residual.  Where VMEC2000 executes ``STOP`` on a rejected pivot, VMEX applies
the identity to the rejected coefficient columns only and keeps the update
finite; ``tools/diagnose_input.py`` reports this as
``D04E_RADIAL_PRECONDITIONER_REJECTED``.

Reconstruction, anisotropy and legacy output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The reconstruction family is accepted only while reconstruction is inactive
and is rejected as a unit otherwise (``LRECON`` with ``IMSE>0`` or
``ITSE>0`` raises ``D00A_RECONSTRUCTION_MODE_UNSUPPORTED``; an inert
``LRECON=T`` stays inert, as in VMEC2000):

``LRECON, IMSE, ITSE, PSA, PFA, ISA, IFA, IMATCH_PHIEDGE, IOPT_RAXIS,
TENSI, TENSP, TENSI2, FPOLYI, MSEANGLE_OFFSET, MSEANGLE_OFFSETM, ISNODES,
IPNODES, RSTARK, DATASTARK, SIGMA_STARK, RTHOM, DATATHOM, SIGMA_THOM,
PRESFAC, PRES_OFFSET, PHIDIAM, SIGMA_DELPHID, NFLXS, INDXFLX, DSIOBT,
SIGMA_FLUX, NBFLD, INDXBFLD, BBC, SIGMA_B, SIGMA_CURRENT, LPOFR``.

The ANIMEC family ``AH, AT, BCRIT, PH_TYPE, PT_TYPE, AH_AUX_S, AH_AUX_F,
AT_AUX_S, AT_AUX_F`` is accepted at its isotropic defaults; nonzero ``AH`` or
non-default ``AT`` raises ``D00F_ANIMEC_MODE_UNSUPPORTED``.  ``LRFP=T``
raises ``D00B_RFP_MODE_UNSUPPORTED``.

.. list-table::
   :header-rows: 1
   :widths: 32 20 48

   * - Variables
     - Status
     - VMEX behavior
   * - ``LBSUBS=T``
     - rejected when active
     - Requests a different ``B_s`` diagnostic in ``jxbforce.f``.
   * - ``LNYQUIST=F``
     - rejected when active
     - VMEX writes the Nyquist WOUT only.
   * - ``LMAC, LEDGE_DUMP, LOLDOUT, LWOUTTXT, LDIAGNO``
     - accepted with warning
     - Auxiliary monitor, edge, legacy, text-WOUT or DIAGNO files are not
       produced; the netCDF WOUT is.
   * - ``LMOVIE, LSPECTRUM_DUMP, LOPTIM``
     - accepted no-op
     - Obsolete; no behavior in the audited VMEC2000 source.
   * - ``LBOOZ, MBOOZ, NBOOZ, BOOZ_SURFACES``
     - VMEX extension; ``LBOOZ=T`` rejected
     - Use ``vmex --booz --mbooz ... --nbooz ... --booz-surfaces ...``
       (:doc:`cli`).

No-silent-physics policy
------------------------

Parsing a name is not evidence that the solver uses it.  VMEX applies these
rules:

1. Unknown INDATA variables and unknown JSON keys are input errors.
2. Active controls that change the mathematical problem, the iteration
   contract or the WOUT convention are implemented or rejected before
   iteration 1.
3. Neutral spellings such as ``AH=0``, ``AT=[1,0,...]``,
   ``TRIP3D_FILE='NONE'`` and an inactive reconstruction block are accepted.
4. Active legacy output requests that do not change the equilibrium are
   accepted with a warning; obsolete controls are no-ops.
5. Symmetry-limited derived methods raise on ``LASYM=T`` instead of dropping
   Fourier partners.

From a source checkout, ``python tools/diagnose_input.py input.case`` runs
the same classification plus a first-force-pass check without a full solve.
Its default output is a PASS/FAIL checklist and one assessment code (for
example ``D00E_TRIP3D_MODE_UNSUPPORTED``), with no path, input value,
coefficient or force magnitude, so it can be shared for a confidential deck;
``--details`` adds values for local use.
``W01_ANGULAR_GRID_BELOW_VMEC_DEFAULT`` flags explicit ``NTHETA``/``NZETA``
below the VMEC2000 automatic floor; it is a convergence-risk warning, not a
rejection.

Equilibrium capability matrix
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 24 19 57

   * - Capability
     - Status
     - Scope and evidence
   * - Fixed boundary, stellarator symmetric
     - parity-regressed
     - Force iteration, multigrid, restart, profiles and WOUT variables have
       VMEC2000 golden tests.
   * - Fixed boundary, ``LASYM=T``
     - parity-regressed, one corrected diagnostic
     - Full asymmetric solve and WOUT partners.  ``currvmns`` uses the
       corrected denominator described in `LASYM currvmns`_.
   * - Free boundary, NESTOR, symmetric
     - parity-regressed on representative cases
     - Mgrid field, plasma-current filament, full/incremental vacuum cadence,
       pressure coupling and WOUT channels.
   * - Free boundary, NESTOR, ``LASYM=T``
     - live-VMEC2000-tested
     - A converged CTH-like asymmetric case, including the NESTOR potential
       and surface-field WOUT partners.
   * - Free-boundary ``NS_ARRAY``
     - implemented
     - Plasma state, ``ivac``, adaptive ``nvacskip``, boundary pressure and
       vacuum continuation are carried; resolution-specific NESTOR
       structures are rebuilt at each grid.
   * - Hot restart
     - implemented
     - ``initial_state`` / ``restart_from`` (a wout file, ``WoutData``,
       ``SolveResult`` or ``SpectralState``), CLI ``--restart``, deck
       ``RESTART_WOUT``.  Coarse rungs below the restart resolution are
       skipped.  A user free-boundary restart repeats vacuum activation;
       continuation between stages carries the vacuum state.
   * - ESSOS/SIMSOPT field callable, ``--coils`` / ``DIRECT_COILS``
     - VMEX extension
     - An ``xyz -> B`` field is tabulated once into an
       :class:`~vmex.core.mgrid.MgridField` and follows the NESTOR path.
       Table values and current scale stay differentiable; coil geometry
       does not through the table.
   * - ``only_coils`` / BIEST vacuum
     - rejected
     - Different boundary models, not aliases for NESTOR.
   * - TRIP3D, reconstruction, RFP, ANIMEC, target-volume rescaling
     - rejected when active
     - See the tables above.

WOUT contract
-------------

VMEX writes a VMEC2000-shaped netCDF WOUT; :doc:`wout-file` lists the
variables.

* ``lrecon`` and ``lrfp`` are false because active modes are rejected.
* ``lmove_axis`` records the input value.
* ``lfreeb`` records the solve actually run: a missing-mgrid fallback is not
  labeled free-boundary.
* NITER exhaustion writes the unconverged WOUT with ``ier_flag = 2`` for
  either boundary mode (VMEC2000's ``vmec.f`` records ``0`` there).  Fatal
  numerical or Jacobian failures never write a WOUT.
* NESTOR potential and surface variables are written for symmetric and
  asymmetric solves.

Differentiation and derived methods
-----------------------------------

*AD* is automatic differentiation of the stated map; *FD-validated* means a
test compares that derivative with finite differences.

.. list-table::
   :header-rows: 1
   :widths: 27 20 53

   * - Method
     - Status
     - Scope
   * - Fixed-boundary implicit equilibrium derivative
     - implemented; FD-validated
     - Boundary, profile, current and flux parameters at a converged fixed
       point.  Multigrid and iteration history are initializers, not
       differentiated.
   * - Free-boundary implicit derivative
     - experimental
     - :mod:`vmex.core.freeboundary_implicit`: reverse-mode derivative of
       the reconverged VMEC--NESTOR root with respect to profiles and
       external-field parameters (ESSOS coil shape and currents, or mgrid
       currents), checked against reconverged finite differences.
       ``device="auto"`` runs it on the CPU; see :doc:`capabilities` for the
       open promotion gates.
   * - Virtual-casing external-field residual
     - implemented; FD-validated
     - Coil/current derivatives on a prescribed plasma boundary.
   * - State-to-surface field data for virtual casing
     - partial
     - Traceable and vmex-native; ``LASYM=T`` raises.  The solver paths need
       the optional ``virtual-casing-jax >= 0.0.7``.
   * - Mgrid tabulation
     - partial
     - Differentiable in table values and current scale, not in the coil
       geometry that produced the table.
   * - Mercier WOUT diagnostic
     - implemented
     - VMEC2000-style WOUT engine and FD-validated implicit objective,
       symmetric and ``LASYM``.
   * - Traceable Boozer/QI/omnigenity
     - implemented; FD-validated
     - Symmetric and ``LASYM`` spectra and objectives.
   * - Ballooning and turbulence geometry
     - implemented
     - Symmetric and ``LASYM`` through the sine-parity ``R``/``Z``/``λ``
       field-line geometry (:mod:`vmex.core.stability`,
       :mod:`vmex.core.turbulence`).
   * - ``L_grad_B`` WOUT and state objectives
     - partial
     - ``LASYM=F`` only; both lanes raise on asymmetric input.
   * - Quasilinear/nonlinear-window turbulence proxies
     - value-level
     - Eigenvector-weighted objectives use finite-difference optimization;
       the growth-rate lane is AD-capable.

LASYM currvmns
--------------

The legacy ``read_wout_mod.f90::Compute_Currents`` asymmetric odd-``m``
branch divides the inner ``bsubumns`` coefficient by the outer half-mesh
``sqrt(s)``.  VMEX uses the inner half-mesh denominator, matching the
PARVMEC-calibrated correction in VMEC++ 0.7.1.  Only the derived asymmetric
current-density WOUT channel changes, not the force iteration.

LASYM constraint scaling (``tcon``)
-----------------------------------

Older VMEC2000 trees carry two LASYM-only factors of one half, and they are a
matched pair, not two independent corrections:

1. ``fixaray.f`` sets the Fourier *analysis* weight to
   ``dnorm = 1/(nzeta*ntheta3)`` when ``LASYM`` is on (half the symmetric
   ``1/(nzeta*(ntheta2-1))``), which halves every reduced-interval force
   projection; and
2. ``bcovar.f`` applies ``IF (lasym) tcon = p5*tcon`` to the whole
   spectral-condensation array.

Upstream retired **both** together.  STELLOPT ``v6.5.0-42-g9177f58c`` and
PARVMEC ``master`` set ``dnorm = 1/(nzeta*(ntheta2-1))`` unconditionally —
correct, because ``symforce.f`` gives the kernel a definite parity first, so
the endpoint-half-weighted reduced integral already equals the full-grid
average — and comment out the ``tcon`` halving in both ``bcovar`` routines.
VMEC++ 0.5.3 independently implements the same convention
(``intNorm = 1/(nZeta*(nThetaReduced-1))`` with no ``LASYM`` branch, and only
the ``tcon(ns) = 0.5*tcon(ns-1)`` edge rule).

VMEX follows the retired-pair convention.  ``constraint_scaling`` takes no
``lasym`` argument and ``trig_tables`` builds one ``cosmui``/``sinmui`` for
both symmetry modes; only the surface-average weight ``dnorm3``
(``wint``/``cosmui3``) stays LASYM-dependent.  Reinstating the ``tcon``
halving *alone* — the reading a 2024 source snapshot invites — reproduces
neither convention, because it changes the constraint-to-MHD force ratio by
two in a code that no longer halves the analysis weight.

Measured on ``input.up_down_asymmetric_tokamak`` (``NS = 17``, 2000
iterations, ``NSTEP = 1``), comparing the per-iteration
``(FSQR, FSQZ, FSQL)`` trajectory against both binaries, at the three printed
digits of the VMEC screen output:

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - VMEX variant
     - vs. upstream ``xvmec2000``
     - vs. 2024-snapshot ``xvmec2000``
   * - as shipped (no halving)
     - **4.9e-3** (print precision)
     - 5.5e+1
   * - ``tcon`` halved only
     - 2.8e+1
     - 5.7e+1
   * - ``dnorm`` halved only
     - 5.9e+1
     - 1.8e+0
   * - both halved
     - 7.5e+1
     - **5.0e-3** (print precision)

(max relative deviation over all 2000 iterations).  The pair also shifts the
converged state: between the two binaries the asymmetric harmonics of
``input.up_down_asymmetric_tokamak`` move by 1.8e-2 (``rmns``) and 1.4e-2
(``zmnc``) relative to their maxima and ``wb`` by 1.5e-7, and VMEX reproduces
each side of that shift to the same figures.  The shipped golden fixtures come
from the upstream binary, so the end-to-end ``tests/test_parity_breadth.py``
case already fails if the halving is reintroduced (it would move ``rmns`` by
2.1e-4 absolute against a 2e-5 tolerance).  ``tests/test_forces_residuals.py``
pins the convention directly, as an exact symmetric limit: for a
stellarator-symmetric configuration the ``LASYM`` lane must reproduce the
symmetric lane's ``tcon`` and ``gcon`` to round-off, and either half-factor
breaks that by exactly two.

Pressureless current-free vacuum limit
--------------------------------------

A pressureless, current-free, nearly axisymmetric vacuum can be harder to
iterate than its boundary suggests.  In the axisymmetric limit the interior
surfaces have a weak parameterization direction: the variational
``m=1,n=0`` force changes little while the magnetic axis drifts, and small
three-dimensional boundary modes only weakly remove that direction.

The :download:`NFP=3 example
<../../examples/data/input.near_degenerate_vacuum_nfp3>` reproduces this
limit.  With ``LFORBAL=F``, VMEX and VMEC2000 follow the same trajectory and
stop just above ``FTOL=1e-11`` after 3,500 iterations.  With ``LFORBAL=T``,
both replace that equation by VMEC2000's flux-averaged force balance and
converge in 941 iterations to
``(FSQR, FSQZ, FSQL) = (9.13e-12, 5.38e-12, 2.32e-12)``; the WOUT geometry
and field coefficients agree to better than ``5e-10`` relative.

For this class of input, try ``LFORBAL`` first.  Loosening ``FTOL`` or
removing unused high-order modes may also end the iteration, but changes the
requested accuracy or discretization.  VMEX does none of these automatically.
