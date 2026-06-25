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
For the square-coil example, unconverged diagnostic WOUTs now return the
lowest fresh free-boundary residual state by default. The solve still records
``converged = false`` and ``production_free_boundary_claim = false`` unless the
strict final residual gates pass. This keeps plots tied to the best available
scored equilibrium-like state instead of the last unscored update when
``max_iter`` is exhausted. The behavior is controlled by
``RETURN_BEST_SCORED_STATE`` in the example and by
``--return-best-scored-state`` in the backend profiler.

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
converge. The square-coil example now sets ``NVACSKIP=1`` by default. This is
intentionally conservative: the VMEC-style adaptive vacuum cadence cannot go
below the input ``NVACSKIP`` floor, so the earlier ``NVACSKIP=NZETA`` default
allowed long stale vacuum-source reuse windows just when this new geometry
needed fresh free-boundary residuals. Beta continuation still uses the previous
final-grid state after the initial staged solve; the driver disables multigrid
when a live ``restart_state`` is supplied.

Latest backend profiling changes the immediate conclusion. With positive
``PHIEDGE``, raw VMEC2000 stops with ``PHIEDGE HAS WRONG SIGN IN VACUUM
SUBROUTINE`` before a useful convergence comparison. With negative ``PHIEDGE``,
VMEC2000 completes the generated-``mgrid`` square-coil cases, but the current
geometry still does not reach ``1e-12``: a low-mode ``NS=5, MPOL=3, NTOR=4``
case drops to total physical residual about ``5.95e-4`` after 5000 VMEC2000
iterations; a well-sampled mid-mode ``NS=9, MPOL=5, NTOR=12, NZETA=32`` case
drops from about ``6.58e-6`` at 1000 VMEC2000 iterations to ``7.04e-8`` at 5000
iterations and ``3.39e-8`` at 10000 iterations; and the higher-mode
``NS=9, MPOL=6, NTOR=23`` case remains around ``4.94e-3`` after 1000 VMEC2000
iterations. ``vmec_jax`` generated-``mgrid`` also activates the dense VMEC-like
NESTOR branch on the sign-corrected deck, but at the well-sampled mid-mode
point it originally reported a final residual around ``3.31e-4`` after 1000
iterations on the same deck. The cause was not mgrid interpolation: profiling
showed that the live JAX residual tail matched VMEC2000 when ``NVACSKIP=1``,
but the final recompute was missing the same free-boundary constraint baseline
used during the iteration. That final-recompute path now reuses the matching
``rcon0/zcon0`` baseline, including for best-scored fallback states. With the
fix, the 1000-iteration ``NS=9, MPOL=5, NTOR=12, NZETA=32, NVACSKIP=1`` JAX
generated-mgrid best-scored residual recomputes to about ``1.02e-5``. The
matching VMEC2000 generated-mgrid case reaches about ``1.46e-5`` at 1000
iterations and about ``7.0e-8`` by 4200--5000 iterations, then oscillates. A
10000-iteration full-update VMEC2000 profile exceeded the local timeout. A
staged ``NS=9 -> 17`` full-update run exposed a separate vacuum-grid envelope
issue: with the older narrow generated mgrid, VMEC2000 repeatedly printed
``Plasma Boundary exceeded Vacuum Grid Size`` in the ``NS=17`` stage. A wider
``48 x 40 x 32`` mgrid with ``1.2`` fractional padding and ``0.5`` absolute
padding removed that warning, but the ``NS=17`` stage still oscillated, with
best sampled total residual about ``3.45e-7`` and final row about ``9.77e-6``
after 3000 iterations at ``DELT=0.05``. Repeating the same widened-mgrid staged
profile with ``DELT=0.02`` removed the large oscillations and reached a final
``NS=17`` total residual of about ``1.42e-7`` after 3000 iterations, with no
vacuum-grid warnings. Extending the final ``NS=17`` stage to 5000 iterations
plateaus near ``1.0e-7``. Reducing to ``DELT=0.01`` with the same iteration
schedule is worse, because the coarse stage underconverges and the final stage
remains near ``1e-5``. The current square-coil setup is therefore a
diagnostic/stability target, not yet a converged ``FTOL=1e-12`` production
equilibrium, but the evidence now points to widened mgrid envelopes plus
moderate damping around ``DELT=0.02`` as the productive continuation path.
The exact widened-mgrid, ``DELT=0.02``, ``NS=9`` comparison shows that
``vmec_jax`` and VMEC2000 now agree on the robust mgrid deck: after 5000
iterations, ``vmec_jax`` generated-mgrid recomputes to total residual about
``1.40e-6`` and VMEC2000 reaches about ``1.50e-6`` on the same mgrid. Extending
the same deck to 10000 iterations keeps that agreement: ``vmec_jax`` reaches
about ``1.30e-7`` and VMEC2000 reaches about ``1.11e-7`` with no vacuum-grid
warnings. Both still miss the requested ``FTOL=1e-12`` by several orders of
magnitude. The 10000-iteration ``vmec_jax`` run used fresh free-boundary updates
on every tail iteration, ended with the same effective step size as the VMEC2000
row (about ``0.0194``), and had only one restart over the run. The older
``~7e-8`` VMEC2000
result used the narrower ``DELT=0.05`` mgrid deck; it is useful as an
optimistic low-resolution reference, but it is not the right target for
radial-resolution studies because the corresponding ``NS=17`` run can move
outside the vacuum grid. The profiler now also records an
initial-boundary provider-parity block before running any force iterations.
On the widened ``48 x 40 x 32`` square-coil deck, generated-mgrid sampling and
the exact direct Biot-Savart provider agree on the initial boundary to about
``3.2e-4`` RMS relative field-vector error and ``1.5e-3`` RMS relative
coil-only ``B.n`` error. That rules out a simple toroidal-angle, current-scale,
or interpolation-convention mismatch as the cause of the direct-coil stall.
With the same provider-parity-checked setup, a 1000-iteration direct-coil
``vmec_jax`` run at ``FTOL=1e-12`` remains monotone near the tail but only
reaches total residual about ``4.1e-4`` and boundary ``B.n`` RMS about
``6.4e-3``. Extending the direct run to 3000 iterations improves the total
residual to about ``4.7e-6`` and boundary ``B.n`` RMS to about ``4.4e-3`` with
fresh full updates every iteration. At 5000 iterations, direct-coil
``vmec_jax`` reaches about ``1.35e-6``, essentially the same floor as
generated-mgrid ``vmec_jax`` at the same iteration budget. At 10000 iterations,
direct-coil ``vmec_jax`` reaches about ``1.88e-7``; generated-mgrid ``vmec_jax``
reaches about ``1.30e-7``; and VMEC2000 reaches about ``1.11e-7``. The direct
path is therefore slower but tracks the same residual floor. Extending the
direct-coil run to 25000 iterations did not close the gap to ``1e-12``. It found
its best fresh summed force residual, about ``1.07e-7``, near iteration 11140,
then cycled: the final residual-history tail was still near ``1.52e-7`` and the
fresh final recompute on the returned best-scored state was worse, about
``4.18e-7``. The run used ``NVACSKIP=1`` and full fresh boundary updates on
24935 of 25000 iterations, had no bad-Jacobian history flags, and took about
3435 seconds on the local CPU. The strict residual gap is therefore no longer
evidence for a JAX-specific solve-control mismatch, a simple direct-provider
convention error, or a plain iteration-budget issue at this resolution. The
current evidence points to the square-coil Fourier representation, resolution
closure, and free-boundary nonlinear cycling around a low-resolution floor. A
matching 25000-iteration VMEC2000 generated-``mgrid`` comparison is the next
robustness check.

The same profiling identified an ``NZETA`` robustness rule. ``MPOL=5,
NTOR=12, NZETA=16`` fails in VMEC2000 after the initial Jacobian changes sign,
while the same generated-``mgrid`` deck with ``NZETA=32`` completes and reaches
total residual about ``6.58e-6`` after 1000 iterations. The branch now exposes
``vmec_jax.recommended_square_axis_nzeta`` and the square-coil example defaults
to ``NZETA=64`` for ``NTOR=23``. Production-style example runs fail early if
``NZETA`` is below the recommendation; diagnostic profiling can still run
underresolved grids and records ``nzeta_underrecommended`` in the JSON report.
The profiler also rejects generated-mgrid plane counts that are not multiples
of ``NZETA`` because the VMEC-plane mgrid sampler intentionally uses the
discrete VMEC zeta planes without toroidal interpolation.
The rounded-square ``axis_kind="spline"`` option is now the default because it
reduces low-mode projection error relative to the superellipse axis. It is
still projected to VMEC Fourier coefficients, so large straight sections plus
localized stellarator corners remain a difficult Fourier representation; using
the spline envelope is a bandwidth reduction, not a replacement for resolution
closure. For the current square-coil shape parameters, the spline envelope
reduces the max component projection error from about ``3.2e-4`` to
``1.3e-4`` at ``MPOL=5, NTOR=12`` and from about ``7.1e-5`` to ``1.4e-5`` at
``MPOL=6, NTOR=23``. This supports using the spline envelope for the hybrid
square axis, while still requiring ``MPOL``/``NTOR``/``NZETA`` convergence.

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

Mirror-physics checks should stay simple and explicit. A two-coil mirror has
its on-axis field minimum near the midplane and maxima near the coils; this is
the baseline shape used by standard magnetic-mirror descriptions such as
Fitzpatrick's plasma-physics notes
(``https://farside.ph.utexas.edu/teaching/plasma/Plasma/node22.html``).
Recent axisymmetric mirror design papers also include plasma diamagnetism in
the mirror-ratio interpretation; for example, the Hammir model notes the
finite-beta factor entering the mirror ratio
(``https://arxiv.org/html/2411.06644v1``). Therefore beta-scan validation
should look for a solved near-axis ``|B|`` depression in the plasma region and
the corresponding effective mirror-ratio response, not just changes in the coil
field. For the stellarator-mirror hybrid, non-axisymmetric mirror literature
warns that bent or non-axisymmetric mirror fields add radial and longitudinal
drifts, so the hybrid lane must retain field-line pitch, iota, and cross-section
diagnostics when non-axisymmetric corner shaping is enabled
(``https://doi.org/10.1063/1.4765692``).

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
Profile reports can be compared with
``tools/diagnostics/summarize_square_coil_profiles.py``; the summarizer reads
ignored JSON artifacts and prints the final and best-scored residual totals
without adding result files to the repository. New profiler rows also include
compact convergence-control statistics for long histories, including time-step
extrema, full-vacuum-update counts, bad-Jacobian counts, and tail boundary
``B.n`` diagnostics, so future long runs can be audited without storing bulky
full history arrays in the repository. The profile JSON also records a compact
geometric residual-tail projection for the summed VMEC force residual. The
summary table exposes the tail decay factor and the estimated additional
iterations to ``1e-12``; this is a diagnostic estimate, not a convergence
claim, but it separates monotone under-budget runs from true residual floors.

The square-axis stellarator-mirror hybrid geometry now has a lower-bandwidth
``axis_kind="spline"`` option. It is still projected into VMEC Fourier boundary
coefficients, but it replaces the sharp polar-square/superellipse content with
a smooth rounded-square envelope before projection. This is the practical
near-term way to reduce ``NTOR`` sensitivity; a true spline basis inside the
VMEC solve would be a larger solver reparameterization. The public helper
``square_axis_stellarator_mirror_hybrid_projection_error`` and the square-coil
profiler's ``boundary_projection`` JSON block now report the Fourier truncation
error for the selected ``MPOL``/``NTOR``/fit-grid combination; the profile
summarizer exposes this as ``boundary_proj_max`` and ``boundary_proj_rel``.
These metrics should be reviewed whenever changing ``MPOL``, ``NTOR``, or
``NZETA``: they diagnose input-boundary underfitting before the free-boundary
nonlinear solve is interpreted.

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
   VMEC-style runs. Use ``--solver-mode parity`` and ``--nvacskip 1`` for
   convergence evidence; larger ``NVACSKIP`` values are speed experiments, not
   strict residual evidence. For radial-resolution ladders, use a widened mgrid
   envelope and record ``vacuum_grid_exceeded_count`` before interpreting the
   residual floor. Keep the provider-parity block enabled unless the run is a
   pure solver-speed benchmark; it verifies that direct-coil and generated-mgrid
   boundary fields still match after resolution or coil changes. Use the
   tail-projection columns in the summary table to choose between extending the
   iteration budget and changing the resolution/schedule; do not interpret the
   projection as proof of convergence.
2. Re-run the square-coil beta ladder with per-beta checkpointing and the
   best-scored diagnostic fallback using the staged ``FTOL_ARRAY`` ending at ``1e-12``. Keep
   ``DELT=0.05``, ``NVACSKIP=1``, ``solver_mode="parity"``, and the VMEC-like
   ``FREE_BOUNDARY_ACTIVATE_FSQ=1e-3`` unless a benchmark shows a better value.
3. Run resolution closure around the first transition beta and at ``10%`` beta,
   comparing ``NS``, ``MPOL``, ``NTOR``, ``NZETA``, generated-mgrid resolution,
   LCFS shape, near-axis field, mirror ratio, mean iota, and residual histories.
   The next numerical knob is not a smaller global step size; ``DELT=0.01`` is
   too slow for the current schedule. Since the direct-coil and JAX mgrid paths
   reproduce the VMEC2000 widened-mgrid ``DELT=0.02`` floor through 10000
   iterations, the next solve-side work is a staged iteration/runtime schedule,
   mode/mgrid refinement, and radial-resolution closure. A larger ``NS`` ladder
   should not be interpreted unless ``vacuum_grid_exceeded_count`` remains zero.
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
