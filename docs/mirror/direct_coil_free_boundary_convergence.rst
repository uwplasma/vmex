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
subroutine for this generated-``mgrid`` deck. The default free-boundary
activation threshold is VMEC-like ``1e-3`` so the vacuum/edge coupling has
enough final-grid iterations to converge. The square-coil example now sets
``NVACSKIP=1`` by default and uses ``DELT=0.02`` with staged
``NITER_ARRAY = 4000, 8000, 12000``. This is
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
matching 25000-iteration VMEC2000 generated-``mgrid`` comparison reached the
same conclusion faster: it finished successfully with no vacuum-grid overflow,
final summed residual about ``1.37e-7``, and best sampled summed residual about
``9.88e-8`` after about 843 seconds. VMEC2000 is therefore faster on this local
CPU benchmark, but not more robust in the sense of reaching the requested
``FTOL=1e-12`` on this low/mid-mode square-coil deck.
Increasing the square-axis representation from ``NTOR=12`` to ``NTOR=16`` while
keeping the spline envelope materially improves the diagnostic deck. The
configured production-shape boundary projection error drops to max component
error about ``4.76e-5``. At ``NS=9, MPOL=5, NTOR=16, NZETA=40``, direct-coil
``vmec_jax`` decreases from summed residual about ``4.55e-4`` at 1000
iterations to about ``4.04e-8`` at 5000 iterations, with the final 128 stored
residuals monotone and no bad-Jacobian events in the compact history. The
matching VMEC2000 generated-``mgrid`` run reaches summed residual about
``1.38e-8`` after 8000 iterations, with best sampled summed residual about
``1.25e-8`` and no vacuum-grid overflow. This is the first clear evidence that
``NTOR`` refinement, not just more iterations at ``NTOR=12``, lowers the
current residual floor. Extending the direct-coil run to 12000 iterations
continues improving the solved state but still does not prove
``FTOL=1e-12`` convergence: the best stored summed residual is about
``5.45e-9`` near iteration 11269, while the fresh recompute on the returned
best-scored state is about ``8.24e-9``. The tail projection slows from the
5000-iteration estimate, the final tail is no longer strictly monotone, and the
boundary ``B.n`` RMS rises from its earlier minimum. The next evidence should
therefore change resolution, staging, or representation rather than simply
extending this same ``NS=9, MPOL=5, NTOR=16`` deck locally.
A first local 16000-iteration VMEC2000 generated-``mgrid`` extension was not
usable as convergence evidence: when run concurrently with the direct-coil
profile, it timed out at 900 seconds with parsed force rows only through
iteration 1600 and summed residual about ``1.18e-5``. Since the previous
same-deck 8000-iteration VMEC2000 run finished in about 203 seconds, this row
is recorded as a runtime/profiling anomaly rather than a new residual floor.
A clean same-deck VMEC2000 extension later confirmed the cycling: at
``NS=9, MPOL=5, NTOR=16, NZETA=40`` it reached a best sampled summed residual
about ``1.15e-8`` but worsened to about ``4.96e-7`` by 16000 iterations. A
staged ``NS=9 -> 13`` version on a widened generated ``mgrid`` improved the
best sampled residual to about ``5.04e-9`` but still cycled upward to about
``7.16e-8`` by the end of the ``NS=13`` stage. This confirms that simply
continuing the ``NTOR=16`` deck is not a credible path to ``FTOL=1e-12``.

The next successful robustness step is the higher-mode spline deck. For the
current production-shape square axis, ``MPOL=6, NTOR=23`` with
``NZETA=64`` lowers the Fourier boundary projection max component error to
about ``1.44e-5``. A single-stage VMEC2000 generated-``mgrid`` run at
``NS=9`` reached summed residual about ``2.05e-8`` after 4000 iterations with
no vacuum-grid overflow. A staged ``NS=9 -> 13`` VMEC2000 generated-``mgrid``
run reached summed residual about ``2.11e-10`` at ``NS=13``. Extending the
ladder to ``NS=17`` on a wider ``72 x 56 x 64`` generated ``mgrid`` reached a
best sampled summed residual about ``2.28e-11`` and final summed residual about
``3.17e-11`` after the 12000-iteration final stage, again with no vacuum-grid
overflow. The final components were about ``1.14e-11`` radial, ``1.72e-11``
vertical, and ``3.05e-12`` lambda. This is still above a per-component
``1e-12`` production claim, but it is several orders of magnitude better than
the ``NTOR=16`` staged floor and shows that the practical route is high-mode
spline projection plus staged radial resolution, not longer low/mid-mode local
runs. VMEC2000 is therefore the best robustness reference and is much faster
than the current direct-coil JAX path for these mgrid benchmarks, but even
VMEC2000 still needs additional algorithmic or resolution work before this
deck can be promoted at ``FTOL=1e-12``.
A 24000-iteration final-stage continuation of the same ``MPOL=6, NTOR=23,
NZETA=64, NS=9 -> 13 -> 17`` VMEC2000 deck finished in about 2369 seconds with
no vacuum-grid overflow. It improved the best sampled summed physical residual
only modestly, to about ``1.86e-11``, and ended at about ``2.19e-11``. The best
physical component row was still above the strict target, with approximately
``8.42e-12`` radial, ``7.70e-12`` vertical, and ``2.45e-12`` lambda residuals.
This converts the earlier near miss into a resolution/algorithm floor for this
Fourier deck: do not spend further CPU on simply extending the same
``6,23`` schedule before testing the ``7,28`` / ``8,32`` spline ladders or a
free-boundary acceleration/numerical-kernel change.

The same profiling identified an ``NZETA`` robustness rule. ``MPOL=5,
NTOR=12, NZETA=16`` fails in VMEC2000 after the initial Jacobian changes sign,
while the same generated-``mgrid`` deck with ``NZETA=32`` completes and reaches
total residual about ``6.58e-6`` after 1000 iterations. The branch now exposes
``vmec_jax.recommended_square_axis_nzeta`` and the square-coil example defaults
to ``NZETA=64`` for ``NTOR=23``. The backend profiler now also resolves omitted
``--nzeta`` or ``--nzeta auto`` to this recommendation for the selected
``NTOR``. Production-style example runs fail early if ``NZETA`` is below the
recommendation; diagnostic profiling can still run underresolved grids and
records ``nzeta_auto`` and ``nzeta_underrecommended`` in the JSON report.
The profiler also rejects generated-mgrid plane counts that are not multiples
of ``NZETA`` because the VMEC-plane mgrid sampler intentionally uses the
discrete VMEC zeta planes without toroidal interpolation.
During long VMEC2000 runs the profiler writes
``_partial_vmec2000_payload.json`` beside the final profile report, so strict
``FTOL=1e-12`` ladders can be audited for live stage residuals and
vacuum-grid overflow before ``xvmec`` exits. The summary tool also accepts an
active VMEC2000 profile directory and will read that sidecar, or parse
``vmec2000_mgrid/threed1*`` directly when the sidecar is not present.
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
The public helper
``vmec_jax.recommend_square_axis_stellarator_mirror_hybrid_resolution`` scans a
small finite ``MPOL``/``NTOR`` ladder for the current spline-smoothed target and
returns the lowest estimated-cost candidate whose projection error satisfies
the requested threshold. This is a boundary-representation gate, not a
nonlinear-convergence claim.
A higher-mode projection spot check on the same spline shape gives max
component boundary errors of about ``1.27e-4`` for ``MPOL=5, NTOR=12``,
``4.77e-5`` for ``5,16``, ``1.44e-5`` for ``6,23``, ``8.96e-6`` for
``7,28``, and ``6.12e-6`` for ``8,32``. If the current ``6,23`` strict solves
plateau above ``1e-12``, the next finite resolution ladder should therefore be
``7,28`` and ``8,32`` with ``NZETA`` at least the corresponding
``recommended_square_axis_nzeta`` value, rather than a blind iteration-budget
extension of the same Fourier deck.

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

VMEC++ source review gives three concrete follow-up candidates for the
``vmec_jax`` direct-coil path if the high-mode staged profiles still plateau
above ``1e-12``. First, VMEC++ has an Anderson(1) accelerator for the
free-boundary vacuum-pressure coupling; that is the most directly relevant
nonlinear-cycling fix to prototype behind an opt-in flag. Second, its NESTOR
singular-integral code includes a Miller-recurrence numerical-stability update
for high ``MPOL``/``NTOR`` runs; the matching ``vmec_jax`` dense-NESTOR replay
path should be audited before pushing beyond ``NTOR=23``. Third, VMEC++ uses
sqrt-weighted interpolation for multigrid transitions, which is relevant if
the ``NS=13 -> 17`` or later ``NS=17 -> 25`` transitions create residual bumps.
These are solver-control and numerical-kernel changes, not geometry changes;
they should be tested against the existing VMEC2000 generated-mgrid reports
before being trusted for direct-coil production claims.

The Anderson(1) pressure lane is now implemented as an opt-in diagnostic path
for ``vmec_jax``. Set ``VMEC_JAX_FREEB_ANDERSON_PRESSURE=1`` or pass
``--freeb-anderson-pressure`` to
``tools/diagnostics/profile_square_coil_free_boundary.py``. The mixer operates
only on full NESTOR updates, stores the mixed ``bsqvac`` back into the NESTOR
runtime so ``ivacskip`` reuse sees the same pressure, and records
``freeb_anderson_pressure_*`` histories plus the last applied theta in the
profile JSON. This is not enabled by default and is not yet promotion evidence;
it is the next controlled A/B profile against the already stored VMEC2000 and
non-Anderson direct-coil rows.

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
(``https://arxiv.org/html/2411.06644v1``). The WHAM physics-basis paper
emphasizes self-consistent finite-beta anisotropic equilibria
(``https://doi.org/10.1017/S0022377823000806``), and recent Pleiades/WHAM
reconstruction work treats beta, stored energy, and diamagnetic magnetic
signals as equilibrium validation quantities
(``https://doi.org/10.1063/5.0306291``). Therefore beta-scan validation should
look for a solved near-axis ``|B|`` depression in the plasma region, the
corresponding effective mirror-ratio response, stored-energy/pressure trends,
and finite-beta boundary-balance diagnostics, not just changes in the coil
field. The 2025 Hammir performance model keeps the same integrated
heating-equilibrium-transport viewpoint for tandem-mirror end plugs
(``https://doi.org/10.1017/S002237782510055X``). For the
stellarator-mirror hybrid, non-axisymmetric mirror literature
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
full history arrays in the repository. The summary table now also exposes the
fresh-boundary convergence gate, fresh recheck/reject/failure counts, best
fresh-boundary state counts, final recompute status, and ``include_edge``
statistics. Those columns make VMEC++-style edge-force propagation and fresh
direct-coil recomputation auditable from the compact table. The profile JSON
also records a compact geometric residual-tail projection for the summed VMEC
force residual. The summary table exposes the tail decay factor and the
estimated additional iterations to ``1e-12``; this is a diagnostic estimate,
not a convergence claim, but it separates monotone under-budget runs from true
residual floors.

The square-axis stellarator-mirror hybrid geometry now has a lower-bandwidth
``axis_kind="spline"`` option. It is still projected into VMEC Fourier boundary
coefficients, but it replaces the sharp polar-square/superellipse content with
a smooth rounded-square envelope before projection. This is the practical
near-term way to reduce ``NTOR`` sensitivity; a true spline basis inside the
VMEC solve would be a larger solver reparameterization. The public helper
``square_axis_stellarator_mirror_hybrid_projection_error`` and the square-coil
profiler's ``boundary_projection`` JSON block now report the Fourier truncation
error, mode count, and recommended ``NZETA`` for the selected
``MPOL``/``NTOR``/fit-grid combination; the profile summarizer exposes this as
``boundary_mode_count``, ``boundary_recommended_nzeta``,
``boundary_proj_max``, and ``boundary_proj_rel``.
These metrics should be reviewed whenever changing ``MPOL``, ``NTOR``, or
``NZETA``: they diagnose input-boundary underfitting before the free-boundary
nonlinear solve is interpreted.
The root square-coil example now enforces
``MAX_BOUNDARY_PROJECTION_ERROR = 5e-5`` by default. This keeps the current
``MPOL=6, NTOR=23, NZETA=64`` production-style deck enabled, while rejecting
the older ``MPOL=5, NTOR=12`` low-mode deck unless the user explicitly sets the
threshold to ``None`` for diagnostic profiling. This guard keeps Fourier
boundary underfitting separate from nonlinear-solver or direct-coil provider
failures.

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
   projection as proof of convergence. Since the ``NTOR=12`` deck floors near
   ``1e-7`` and the improved ``NTOR=16`` deck slows near ``1e-8`` by 12000
   direct-coil iterations, do not spend more local CPU on longer single-stage
   runs at either same deck before changing resolution, schedule, or geometry
   representation. For pure direct-provider GPU profiling, use
   ``--jit-forces --coil-chunk-size 0 --skip-mgrid --skip-provider-parity`` so
   the run does not spend memory generating an unused mgrid. When a backend
   comparison still needs a generated mgrid, keep the mgrid write chunked even
   if the direct sampler is using the full-geometry JIT path.
   For the first pressure-mixing A/B run, keep the same staged deck and add
   ``--freeb-anderson-pressure`` only to the direct-coil backend run; compare
   ``free_boundary_anderson_pressure_last_theta``,
   ``freeb_anderson_pressure_applied_stats``, final component residuals, and
   the tail projection against the non-Anderson profile before changing any
   other solver knob.
2. Re-run the square-coil beta ladder with per-beta checkpointing and the
   best-scored diagnostic fallback using the staged ``FTOL_ARRAY`` ending at
   ``1e-12``. Keep ``DELT=0.02``, ``NVACSKIP=1``,
   ``solver_mode="parity"``, and the VMEC-like
   ``FREE_BOUNDARY_ACTIVATE_FSQ=1e-3`` unless a benchmark shows a better value.
   The current robust default ladder is ``NS_ARRAY = 9, 13, 17`` with
   ``NITER_ARRAY = 4000, 8000, 12000`` and
   ``FTOL_ARRAY = 1e-8, 1e-10, 1e-12``.
3. Run resolution closure around the first transition beta and at ``10%`` beta,
   comparing ``NS``, ``MPOL``, ``NTOR``, ``NZETA``, generated-mgrid resolution,
   LCFS shape, near-axis field, mirror ratio, mean iota, and residual histories.
   The next numerical knob is not a smaller global step size; ``DELT=0.01`` is
   too slow for the current schedule. Since the direct-coil, JAX mgrid, and
   VMEC2000 widened-mgrid ``DELT=0.02`` paths all sit near the same ``1e-7``
   low-resolution floor, the next solve-side work is a staged
   iteration/runtime schedule, mode/mgrid refinement, and radial-resolution
   closure. A larger ``NS`` ladder should not be interpreted unless
   ``vacuum_grid_exceeded_count`` remains zero.
4. If the ``MPOL=7, NTOR=28`` and ``MPOL=8, NTOR=32`` spline-projected
   VMEC/VMEC2000 ladders still stall above the strict ``1e-12`` component gate,
   move the square-axis hybrid into a true spline-basis lane instead of only
   increasing Fourier modes. The current code already uses a spline-smoothed
   real-space target before VMEC Fourier projection; that is a bandwidth
   reduction, not a new nonlinear-solve basis. A true spline-basis lane should
   keep the same direct-coil/free-boundary diagnostics but replace the
   boundary/control representation with a small set of periodic side/corner
   spline control points, then project to Fourier only for VMEC2000 parity or
   WOUT export. This keeps the primary ``vmec_jax`` path closer to the intended
   straight-side geometry while preserving a benchmarkable VMEC2000 comparison.
5. Keep the optional virtual-casing postsolve diagnostic
   ``vmec_jax.free_boundary_validation.virtual_casing_finite_beta_boundary_diagnostics``
   attached to the square-coil example outputs. The helper accepts a solved
   surface, total surface field, and direct-coil field, then reports the
   required external-field normal mismatch and finite-beta magnetic-pressure
   jump. It is optional at import time and should be skipped when
   ``virtual_casing_jax`` is not installed.
6. Promote only rows that pass the force-residual and postsolve boundary
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
