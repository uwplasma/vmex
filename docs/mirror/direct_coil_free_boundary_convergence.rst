Direct-Coil Free-Boundary Convergence Plan
==========================================

This page is the active convergence plan for the square-coil
stellarator-mirror-hybrid free-boundary lane. It records what is already known,
what evidence is still missing, and the finite set of gates required before a
direct-coil finite-beta run is described as a converged production equilibrium.

Current Status
--------------

The repo-root example
``examples/toroidal_stellarator_mirror_hybrid_square_coils_free_boundary.py``
now runs actual ``vmec_jax.run_free_boundary`` solves from a direct-coil
provider. It writes solved WOUT files, solved-state field-line and LCFS plots,
per-beta CSV/JSON metrics, and checkpoints the summary after each beta row.
The branch also keeps the final live state by default; older behavior that
returned a best-scored intermediate state is now opt-in through
``VMEC_JAX_RETURN_BEST_SCORED_STATE=1``.

The latest strict fresh residual evidence at the coarse review resolution
``NS=9, MPOL=5, NTOR=12`` reaches active free-boundary convergence through
``5%`` beta. At the same resolution, ``7%`` beta is the first transition case:
it stalls just above the requested radial-force tolerance and shows many
bad-Jacobian resets. ``8%`` through ``10%`` beta stall with larger radial-force
floors. Older resolution and step-size probes show that increasing resolution
and increasing ``DELT`` from the default reduce the ``10%`` radial-force floor,
but do not yet prove strict convergence.

The current production target for this lane is stricter than that evidence:
``FTOL=1e-12`` for each final force component. The square-coil example now
therefore defaults to a VMEC-style staged solve, with explicit
``NS_ARRAY``, ``NITER_ARRAY``, and ``FTOL_ARRAY`` ending at ``1e-12``. It also
uses negative ``PHIEDGE`` for the default positive toroidal current and
square-coil field orientation. VMEC2000 rejects the opposite sign in the vacuum
subroutine for this generated-``mgrid`` deck. The default step size is
``DELT=0.05`` and the default free-boundary activation threshold is VMEC-like
``1e-3`` so the vacuum/edge coupling has enough final-grid iterations to
converge. Beta continuation still uses the previous final-grid state after the
initial staged solve; the driver disables multigrid when a live ``restart_state``
is supplied.

Latest backend profiling changes the immediate conclusion. With positive
``PHIEDGE``, raw VMEC2000 stops with ``PHIEDGE HAS WRONG SIGN IN VACUUM
SUBROUTINE`` before a useful convergence comparison. With negative ``PHIEDGE``,
VMEC2000 completes the generated-``mgrid`` square-coil cases, but the current
geometry still does not reach ``1e-12``: a low-mode ``NS=5, MPOL=3, NTOR=4``
case drops to total physical residual about ``5.95e-4`` after 5000 VMEC2000
iterations, and the higher-mode ``NS=9, MPOL=6, NTOR=23`` case remains around
``4.94e-3`` after 1000 VMEC2000 iterations. ``vmec_jax`` generated-``mgrid``
also activates the dense VMEC-like NESTOR branch on the sign-corrected deck but
is still underconverged. The current square-coil setup is therefore a
diagnostic/stability target, not yet a converged production equilibrium.

The same profiling identified an ``NZETA`` robustness rule. ``MPOL=5,
NTOR=12, NZETA=16`` fails in VMEC2000 after the initial Jacobian changes sign,
while the same generated-``mgrid`` deck with ``NZETA=32`` completes and reaches
total residual about ``6.58e-6`` after 1000 iterations. The branch now exposes
``vmec_jax.recommended_square_axis_nzeta`` and the square-coil example defaults
to ``NZETA=64`` for ``NTOR=23``. Production-style example runs fail early if
``NZETA`` is below the recommendation; diagnostic profiling can still run
underresolved grids and records ``nzeta_underrecommended`` in the JSON report.

Physics And Algorithm Findings
------------------------------

VMEC2000 remains the solve-side reference. Its free-boundary path delays vacuum
coupling until the interior ``R/Z`` force residual is already small, forces full
NESTOR updates for the first vacuum iterations, then uses ``NVACSKIP`` to
cadence expensive full vacuum updates. The vacuum magnetic pressure enters the
edge force through the dedicated ``bsqvac``/``rbsq`` path, not by enabling
generic fixed-boundary edge rows. VMEC documentation and the SIMSOPT VMEC
interface also make clear that strict production runs usually need large
final-grid iteration budgets, often thousands of iterations.

DESC separates the postsolve finite-beta boundary condition from the VMEC-style
iterative solve. Its finite-beta ``BoundaryError`` objective uses the virtual
casing principle to compute the plasma contribution to the exterior magnetic
field, then checks both normal-field and magnetic-pressure jump conditions. The
cheaper ``VacuumBoundaryError`` is valid only when pressure and plasma current
vanish. This matches the interpretation here: coil-only ``B.n`` is a vacuum
diagnostic, not a finite-beta promotion criterion.

The ``virtual_casing_jax`` package is the preferred optional postprocessor for
research-grade finite-beta diagnostics because it provides JAX-compatible
on-surface external/internal field functionals, normal-field JVP columns, and
high-order singular quadrature. The direct-coil convergence lane should use it
as an optional diagnostic dependency rather than copying DESC's singular
integral implementation into ``vmec_jax``.

VMEC2000 remains the robustness baseline for ``mgrid`` free-boundary solves.
The branch now includes a native
``vmec_jax.external_fields.write_mgrid_from_coils`` helper and a square-coil
backend profile diagnostic,
``tools/diagnostics/profile_square_coil_free_boundary.py``. That diagnostic
writes the same square-coil field to a VMEC-compatible ``mgrid.nc`` and can run
``vmec_jax`` through both direct-coil and generated-mgrid paths, plus optional
raw ``xvmec2000`` on the generated mgrid. This is the required comparison when
judging whether a stall is caused by the direct provider, mgrid interpolation,
or the VMEC-style nonlinear solve itself.

The square-axis stellarator-mirror hybrid geometry now has a lower-bandwidth
``axis_kind="spline"`` option. It is still projected into VMEC Fourier boundary
coefficients, but it replaces the sharp polar-square/superellipse content with
a smooth rounded-square envelope before projection. This is the practical
near-term way to reduce ``NTOR`` sensitivity; a true spline basis inside the
VMEC solve would be a larger solver reparameterization.

Promotion Gates
---------------

A direct-coil finite-beta equilibrium is promoted only when all gates below are
true for the current live state:

1. ``LFREEB`` is active and NESTOR/direct-coil coupling has turned on.
2. A fresh final NESTOR/direct-coil sample has been recomputed on the accepted
   state, not on a stale trial or best-scored state.
3. ``final_fsqr``, ``final_fsqz``, and ``final_fsql`` each satisfy the requested
   ``FTOL`` on the final recompute.
4. The LCFS coefficients change with beta in the solved WOUTs, proving that the
   boundary is not effectively fixed.
5. A resolution ladder in ``NS``, ``MPOL``, ``NTOR``, and boundary sampling
   shows stable convergence of the final force residuals, LCFS shape, near-axis
   ``|B|``, mirror ratio, and mean iota.
6. Finite-beta promotion uses VMEC force residuals plus a total-field
   pressure-balance diagnostic. Coil-only ``B.n`` remains a vacuum-only check.
7. The plotted LCFS, cross sections, ``|B|`` maps, field-line traces, residual
   histories, and beta trends come from solved states and are regenerated from
   ignored ``results/`` artifacts.

Execution Plan
--------------

The remaining work is deliberately narrow:

1. Run the square-coil backend profile at small and moderate grids with
   VMEC-compatible negative ``PHIEDGE``: direct-coil ``vmec_jax``,
   generated-mgrid ``vmec_jax``, and generated-mgrid VMEC2000. First use
   ``FTOL=1e-8`` to identify provider/parity issues, then repeat the
   best-performing setup at ``FTOL=1e-12``. The profiler accepts explicit
   ``--ns-array``, ``--niter-array``, and ``--ftol-array`` arguments for staged
   VMEC-style runs.
2. Re-run the square-coil beta ladder with the live-state default and per-beta
   checkpointing using the staged ``FTOL_ARRAY`` ending at ``1e-12``. Keep
   ``DELT=0.05`` and the VMEC-like ``FREE_BOUNDARY_ACTIVATE_FSQ=1e-3`` unless a
   benchmark shows a better value.
3. Run resolution closure around the first transition beta and at ``10%`` beta,
   comparing ``NS``, ``MPOL``, ``NTOR``, ``NZETA``, generated-mgrid resolution,
   LCFS shape, near-axis field, mirror ratio, mean iota, and residual histories.
4. Keep the optional virtual-casing postsolve diagnostic
   ``vmec_jax.free_boundary_validation.virtual_casing_finite_beta_boundary_diagnostics``
   attached to the square-coil example outputs. The helper accepts a solved
   surface, total surface field, and direct-coil field, then reports the
   required external-field normal mismatch and finite-beta magnetic-pressure
   jump. It is optional at import time and should be skipped when
   ``virtual_casing_jax`` is not installed.
5. Promote only rows that pass the force-residual and postsolve boundary
   diagnostics. Keep unconverged rows in the example output as explicit stall
   evidence with ``production_free_boundary_claim = false``.

Best-Practice Constraints
-------------------------

Keep generated WOUTs, raw JSON, and full-resolution figures under ignored
``results/`` paths. If a figure is committed for review, compress it and keep
only summary panels. Source changes should stay in the existing free-boundary,
external-field, or mirror-domain modules; avoid adding new root-level helper
files. Diagnostic functions should have small public entry points, clear
docstrings, and tests that exercise vacuum and finite-beta interpretation
separately.

Reviewed External Anchors
-------------------------

- DESC free-boundary tutorial:
  https://desc-docs.readthedocs.io/en/stable/notebooks/tutorials/free_boundary_equilibrium.html
- DESC free-boundary paper/preprint:
  https://arxiv.org/abs/2412.05680
- VMEC/STELLOPT documentation:
  https://princetonuniversity.github.io/STELLOPT/VMEC.html
- SIMSOPT VMEC free-boundary interface notes:
  https://simsopt.readthedocs.io/v1.10.2/example_vmec.html
- VMEC++ free-boundary numerics and active free-boundary PRs:
  https://arxiv.org/abs/2502.04374 and
  https://github.com/proximafusion/vmecpp/pulls?q=free+boundary
- WHAM finite-beta mirror physics basis:
  https://www.osti.gov/biblio/2001162
- VMEC/DESC/SPEC free-boundary Shafranov-shift verification:
  https://doi.org/10.1063/5.0253843
