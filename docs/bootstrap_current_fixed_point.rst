Bootstrap-Current Fixed-Point Plan
==================================

This page plans the next finite-beta lane for ``vmec_jax``: compute a
self-consistent VMEC current profile from the Redl bootstrap-current formula
without optimizing the plasma boundary, coils, or current coefficients as
research design variables.

The short answer is that the Redl formula can drive the VMEC input current, but
not by assigning ``AC_AUX_F = <J.B>_Redl``.  Redl gives a flux-surface-averaged
parallel-current quantity, while VMEC consumes a toroidal-current profile shape
plus ``CURTOR``.  A conversion and fixed-point loop are required.

Current Implementation Status
-----------------------------

Implemented now:

- ``vmec_jax.bootstrap_current`` contains pure JAX helpers for the
  Redl-current source term, low-beta and lagged-pressure derivative updates,
  integrating-factor current updates, damping, VMEC ``PHIEDGE`` sign
  conversion, and conversion to ``PCURR_TYPE = "cubic_spline_ip"`` /
  ``AC_AUX_S/F`` / ``CURTOR``.
- ``vj.apply_current_profile_to_indata`` and
  ``vj.bootstrap_current_update_to_indata`` write the generated current profile
  to a copy of a VMEC ``InData`` object without mutating the original input.
- Manufactured-profile tests cover the update formulas, sign convention,
  VMEC spline-current round trip, damping validation, and autodiff through the
  pure current-update helpers.

Next implementation step:

- Add the solve-in-the-loop fixed-point driver that runs VMEC, evaluates Redl
  geometry, updates the current profile, writes per-stage inputs/WOUTs, and
  stops on Redl-mismatch/current-update convergence.

Literature Anchors
------------------

The implementation should follow these sources:

- Redl et al., Physics of Plasmas 28, 022502 (2021), for the analytic
  bootstrap-current fit used by SIMSOPT and ``vmec_jax``.
- Landreman, Buller, and Drevlak, "Optimization of quasisymmetric stellarators
  with self-consistent bootstrap current and energetic particle confinement",
  for the finite-beta QS objective strategy: penalize the mismatch between
  VMEC ``<J.B>`` and the Redl prediction, and include the current profile in
  the solve.
- Landreman's note "Computing VMEC's AC current profile and CURTOR from a
  bootstrap current code", for the conversion from ``<J.B>`` to VMEC current
  input.  The key equation is independent of the angular coordinate choice.
- STELLOPT VBOOT/SFINCS documentation, for the production precedent: iterate
  between an equilibrium code and a bootstrap-current code until the current
  profile is self-consistent.
- DESC's ``BootstrapRedlConsistency`` objective and bootstrap-current tutorial,
  for two complementary approaches: current-profile optimization and iterative
  solves.  DESC also highlights two numerical details that matter here: exclude
  the magnetic axis and the edge when the kinetic profiles vanish, and enforce
  regularity of the current profile near the axis.
- JAX/JAXopt implicit-differentiation documentation, for the phase-2 path:
  expose the converged fixed point through an implicit rule instead of taping
  every Picard/Anderson iteration.

VMEC Current Convention
-----------------------

The VMEC input controls are:

``NCURR``
   ``1`` means the equilibrium is current-driven.

``CURTOR``
   Total toroidal current at the LCFS.

``PCURR_TYPE``
   For this lane, use ``"cubic_spline_ip"`` first.  The suffix ``_ip`` means
   the profile values represent :math:`I_T'(s)`, not :math:`I_T(s)`.

``AC_AUX_S`` and ``AC_AUX_F``
   Knot locations and profile values.  With ``"cubic_spline_ip"``,
   ``AC_AUX_F`` stores the current-derivative shape.  VMEC scales this shape
   so the integrated edge current matches ``CURTOR``.

The corresponding ``vmec_jax`` code path is
``profiles.eval_profiles(..., pcurr_type="cubic_spline_ip")`` followed by the
normalization in ``wout._icurv_full_mesh_from_indata``.  Any fixed-point helper
must therefore set ``NCURR = 1``, a nonzero ``CURTOR``, ``PCURR_TYPE``, and
consistent ``AC_AUX_S/F`` arrays.

Conversion Equation
-------------------

Let :math:`I(s)` be the enclosed toroidal current and :math:`p(s)` the pressure
profile.  Landreman's VMEC-current note gives the profile equation

.. math::

   \frac{d I}{d s}
   + \frac{\mu_0 I}{\langle B^2\rangle}\frac{d p}{d s}
   =
   2\pi \frac{d\psi}{d s}
   \frac{\langle J_\parallel B\rangle}{\langle B^2\rangle}.

The Redl formula provides the right-hand-side quantity
:math:`\langle J_\parallel B\rangle = \langle J.B\rangle_\mathrm{Redl}` from
the current VMEC geometry, pressure, density, temperature, collisionality, and
QS helicity.  The current update is therefore an ODE solve or derivative update
for :math:`I(s)`, not a direct copy of Redl values into VMEC input arrays.

For initial implementation, support three update policies:

``low_beta``
   Drop the pressure-gradient correction:

   .. math::

      I'_{k+1}(s) =
      2\pi \psi'(s)
      \frac{\langle J.B\rangle_{\mathrm{Redl}, k}}
           {\langle B^2\rangle_k}.

``lagged_pressure``
   Use the pressure-gradient term from the previous equilibrium:

   .. math::

      I'_{k+1}(s) =
      -\frac{\mu_0 I_k(s)}{\langle B^2\rangle_k}p'(s)
      + 2\pi \psi'(s)
      \frac{\langle J.B\rangle_{\mathrm{Redl}, k}}
           {\langle B^2\rangle_k}.

``integrating_factor``
   Solve the ODE for :math:`I(s)` with an integrating factor.  This is the most
   accurate deterministic update and should become the default once parity
   tests are green.

All three policies must enforce :math:`I(0)=0`, avoid Redl samples at the axis,
and avoid the edge when density or temperature vanishes.

Fixed-Point Algorithm
---------------------

The deterministic workflow is:

1. Build standard finite-beta profiles using
   ``vj.standard_finite_beta_profiles(beta_percent)``.
2. Write pressure into the input deck with ``vj.with_pressure_profile``.
3. Initialize a VMEC current profile, preferably
   ``PCURR_TYPE = "cubic_spline_ip"`` on half-cell interior knots.
4. Run ``vmec_jax`` to a converged finite-beta equilibrium.
5. Compute Redl geometry and ``<J.B>_Redl`` with
   ``vj.redl_bootstrap_mismatch_from_state`` or a lower-level geometry helper.
6. Compute :math:`\langle B^2\rangle`, :math:`p'(s)`, :math:`I_k(s)`, and
   :math:`\psi' = \mathrm{signgs}\,\Phi_\mathrm{edge}/(2\pi)`.
7. Update :math:`I'_{k+1}` or :math:`I_{k+1}` using one of the policies above.
8. Fit/resample the updated profile to ``AC_AUX_S/F`` and set
   ``CURTOR = signgs * I_{k+1}(1)`` with VMEC's sign convention.
9. Damped update:

   .. math::

      c_{k+1} = (1-\alpha)c_k + \alpha\Pi(I_{k+1}),

   where :math:`\Pi` is the VMEC-profile projection and ``alpha`` starts in
   ``[0.3, 0.7]``.
10. Repeat until both the Redl mismatch and current-profile update norm are
    below tolerance.

After the base Picard loop is correct, add guarded Anderson acceleration.  It
should only accept an accelerated step when it lowers the Redl mismatch and
keeps ``CURTOR``, ``AC_AUX_F``, beta, and VMEC residuals finite.

Planned API
-----------

Add a module such as ``vmec_jax/bootstrap_current.py`` with function-first,
testable helpers:

``BootstrapCurrentOptions``
   Frozen dataclass containing ``helicity_n``, profile coefficients,
   ``surfaces``, ``n_current``, update policy, damping, Anderson depth,
   convergence tolerances, and VMEC solve budgets.

``BootstrapCurrentIteration``
   Frozen dataclass or serializable dict containing iteration number,
   residual norms, ``CURTOR``, ``AC_AUX_S/F``, beta, aspect, mean iota, VMEC
   force residuals, and output paths.

``redl_current_derivative_update(...)``
   Pure JAX helper returning :math:`I'(s)` for ``low_beta`` and
   ``lagged_pressure``.

``redl_current_integrating_factor_update(...)``
   Pure JAX helper returning :math:`I(s)` and :math:`I'(s)` from the integrating
   factor formula.

``vmec_current_profile_from_bootstrap_update(...)``
   Convert updated current samples to VMEC ``PCURR_TYPE``, ``AC_AUX_S/F``, and
   ``CURTOR`` with documented sign convention.

``apply_current_profile_to_indata(...)``
   Return a copy of ``InData`` with ``NCURR``, ``CURTOR``, ``PCURR_TYPE``, and
   ``AC_AUX_S/F`` applied.

``bootstrap_current_fixed_point(...)``
   Driver that runs the VMEC/Redl iteration, writes per-stage inputs/WOUTs,
   records JSON/CSV history, and optionally uses Anderson acceleration.

The first version may be a deterministic preconditioner outside the
optimization variable vector.  A later version can wrap the converged fixed
point in an implicit derivative, using the fixed-point residual
:math:`F(c, x)=0` and a transpose linear solve.

Tests and Validation Gates
--------------------------

Fast unit tests:

- ``ProfilePressure`` and standard finite-beta profiles match SIMSOPT profile
  algebra and units.
- ``cubic_spline_ip`` evaluation integrates the supplied derivative profile.
- ``apply_current_profile_to_indata`` sets ``NCURR``, ``CURTOR``,
  ``PCURR_TYPE``, ``AC_AUX_S``, and ``AC_AUX_F`` without mutating input data.
- Low-beta, lagged-pressure, and integrating-factor updates reproduce analytic
  manufactured profiles for constant :math:`\langle B^2\rangle`, linear
  pressure, and polynomial ``<J.B>``.
- Sign-convention tests verify ``CURTOR = signgs * I(1)`` and the
  ``PHIEDGE``/``phips`` convention used by ``vmec_jax``.
- Redl samples exclude ``s=0`` and edge points where density or temperature
  vanish.
- Damping and Anderson proposal filters are deterministic and reject
  non-finite or mismatch-increasing proposals.

Physics and parity tests:

- One fixed-point update on a bundled finite-beta tokamak reduces the Redl
  mismatch relative to the initial zero/ad-hoc current.
- The full fixed-point loop reaches a documented mismatch tolerance on a tiny
  tokamak fixture within a bounded number of VMEC solves.
- Optional SIMSOPT parity compares ``<J.B>_Redl`` and normalized mismatch
  against ``VmecRedlBootstrapMismatch`` on the same WOUT and profiles.
- Optional DESC parity compares the iterative-solve current profile on a
  small analytic equilibrium when DESC is installed.
- Optional VMEC2000 parity runs the final VMEC input generated by the
  fixed-point loop and compares WOUT current/profile channels, iota, beta,
  aspect, and force residuals.

Differentiability tests:

- JAX gradients of the pure current-update helpers with respect to pressure
  coefficients and Redl ``<J.B>`` samples match central finite differences.
- If Anderson acceleration is enabled, the default differentiable path should
  be the damped Picard map; Anderson can remain a non-differentiable
  performance wrapper until an implicit fixed-point rule is added.
- Phase-2 implicit differentiation validates gradients of a converged
  fixed-point current profile with respect to pressure-profile coefficients
  against finite differences of the full fixed-point loop.

Benchmarks
----------

Add scripts under ``tools/benchmarks``:

``bench_bootstrap_current_fixed_point.py``
   Matrix over update policy, damping, Anderson depth, number of current knots,
   and backend.  Record VMEC solve time, Redl geometry time, current update
   time, mismatch reduction, and total iterations.

``bench_redl_geometry_profiles.py``
   Profile Redl geometry assembly and trapped-fraction quadrature versus
   radial and angular resolution.

``compare_bootstrap_fixed_point_simsopt.py``
   Compare vmec_jax fixed-point update against SIMSOPT current-profile
   optimization for the same QA/QH/QI input deck and standard profiles.

Publication Plots
-----------------

Reviewer-ready figures should be generated from JSON/CSV summaries, not
hand-built:

- Density, temperature, and pressure profiles for beta labels used in examples.
- ``<J.B>_VMEC`` versus ``<J.B>_Redl`` before and after the fixed-point loop.
- Normalized Redl mismatch and current-update norm versus fixed-point
  iteration.
- VMEC current derivative ``I'(s)`` and integrated current ``I(s)`` over
  iterations.
- Iota, beta, aspect, and VMEC force residuals before/after.
- Free-boundary beta-response panels with and without bootstrap-current update:
  cross sections, iota, and LCFS ``|B|`` contours.
- Comparison table: vmec_jax fixed point, SIMSOPT current-profile optimization,
  DESC iterative solve when available, and VMEC2000 final-input replay.

Implementation Order
--------------------

1. Add pure current-profile conversion/update helpers and unit tests.
2. Add ``apply_current_profile_to_indata`` and WOUT/current-profile round-trip
   tests.
3. Add a fixed-point driver for fixed-boundary finite-beta inputs.
4. Validate against a small tokamak and a small QH/QA finite-beta fixture.
5. Add optional SIMSOPT/DESC/VMEC2000 parity scripts.
6. Add generated plots and docs examples.
7. Thread the fixed-point current preconditioner into finite-beta free-boundary
   coil examples.
8. Add implicit fixed-point differentiation only after forward fixed-point
   parity is stable.

Acceptance Criteria
-------------------

- The fixed-point driver reduces normalized Redl mismatch on at least one
  bundled finite-beta fixture without changing boundary or coil variables.
- Generated final input files run directly with ``vmec_jax`` and, when
  available, VMEC2000.
- ``CURTOR`` and ``AC_AUX_F`` sign/normalization are covered by tests.
- SIMSOPT Redl mismatch parity remains green.
- Documentation states explicitly that this is a current-profile
  self-consistency preconditioner, not a replacement for later full
  finite-beta shape/coil optimization.
