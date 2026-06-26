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
``NITER_ARRAY = 4000, 8000, 24000``. This is
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

The active strict reference has now moved to the projection-gated
``MPOL=5, NTOR=28, NZETA=64`` rounded-spline square-axis deck, with
``NS=9 -> 13 -> 17``, ``NITER_ARRAY=4000,8000,24000``,
``FTOL_ARRAY=1e-8,1e-10,1e-12``, ``DELT=0.02``, ``NVACSKIP=1``, and a widened
``88 x 64 x 64`` generated ``mgrid``. This active row was launched with
``axis_kind="spline"``; the follow-up commands use ``axis_kind="control_spline"``,
which gives the same low-bandwidth rounded-square projection for the default
uniform controls. The current VMEC2000 row is still
running, not converged: a live sidecar snapshot at final-stage iteration
``10307`` reports summed residual about ``2.31e-11``, max component about
``1.11e-11``, strict gap about ``11.1``, and no vacuum-grid overflow. The tail
is still decreasing slowly but is close enough to a floor that simply launching
another identical VMEC2000 row is not useful. VMEC2000 is therefore the
robustness and speed reference for generated-``mgrid`` studies, but it has not
solved the strict ``1e-12`` problem for this square-axis deck. The direct JAX
GPU rows on the matched ``control_spline`` deck are still much farther from
strict tolerance, around the ``1e-6`` force scale in their current ``1e-10``
stage, so they remain differentiable research rows rather than production
evidence.

The same profiling identified an ``NZETA`` robustness rule. ``MPOL=5,
NTOR=12, NZETA=16`` fails in VMEC2000 after the initial Jacobian changes sign,
while the same generated-``mgrid`` deck with ``NZETA=32`` completes and reaches
total residual about ``6.58e-6`` after 1000 iterations. The branch now exposes
``vmec_jax.recommended_square_axis_nzeta`` and the square-coil example leaves
``NZETA=None`` by default, resolving the VMEC zeta grid from the edited
``NTOR`` at runtime. The backend profiler also resolves omitted ``--nzeta`` or
``--nzeta auto`` to this recommendation for the selected ``NTOR``.
Production-style example runs fail early if ``NZETA`` is below the
recommendation; diagnostic profiling can still run underresolved grids and
records ``nzeta_auto`` and ``nzeta_underrecommended`` in the JSON report.
The example metrics JSON now also carries a ``resolution_deck`` block with the
same projection/``NZETA`` gate used by the profiler.
The repo-root square-coil example also writes
``square_coil_hybrid_preflight.json`` before starting heavy solves. That
preflight records the requested component-wise ``FTOL=1e-12`` target, the full
staged ``NS_ARRAY``/``NITER_ARRAY``/``FTOL_ARRAY`` budget, the
projection/``NZETA`` deck status, spline-control map conditioning, and a
``spline_bridge`` block stating that the current control-spline path is still a
projection bridge into VMEC Fourier coefficients rather than a solver-native
spline-control update.
The profiler also rejects generated-mgrid plane counts that are not multiples
of ``NZETA`` because the VMEC-plane mgrid sampler intentionally uses the
discrete VMEC zeta planes without toroidal interpolation.
During long VMEC2000 runs the profiler writes
``_partial_vmec2000_payload.json`` beside the final profile report, so strict
``FTOL=1e-12`` ladders can be audited for live stage residuals and
vacuum-grid overflow before ``xvmec`` exits. The summary tool also accepts an
active VMEC2000 profile directory and will read that sidecar, or parse
``vmec2000_mgrid/threed1*`` directly when the sidecar is not present.
Rows with no parsed force iterations now include ``progress_phase``. A value
of ``startup_or_pre_iteration_output`` means VMEC2000 has opened ``threed1``
but has not yet printed the force table; it is not a strict residual result.
The profiler exposes ``--nstep`` and the root example writes ``NSTEP`` into the
generated VMEC input. Use ``--nstep 1`` for strict VMEC2000 profiles so long
high-mode runs expose force rows as soon as VMEC prints them; larger values are
acceptable for short production runs but make live convergence diagnosis weaker.
Completed ``vmec_jax`` backend rows also report
``boundary_coeff_delta_*`` and ``boundary_sample_displacement_*`` fields, which
measure how far the accepted LCFS moved from the initial prescribed boundary.
These are evidence that a row used a moving free boundary; they complement, but
do not replace, the strict force-component and finite-beta pressure-balance
promotion gates.
For ``spline`` and ``control_spline`` square-axis rows, the same solved
boundary displacement is also projected onto the square-reduced side/corner
control map. The nested ``boundary_reduced_control_projection`` block reports
the fitted side/corner radius changes, rank, singular values, relative
projection residual, and captured fraction. The summary table exposes the
status, relative residual, captured fraction, and compact fitted deltas. This
is the bridge diagnostic for deciding whether a solver-native spline-control
update is justified: high capture means the accepted Fourier LCFS motion lies
mostly in the intended low-dimensional control subspace, while low capture
means a reduced update would need more controls or a different basis.
Backend rows now also include a compact ``free_boundary_promotion`` block. The
summary table exposes this as ``boundary_condition_mode``,
``coil_bnormal_role``, ``production_candidate``, ``promotion_blockers``,
``virtual_casing_required``, and ``virtual_casing_available``. For vacuum rows,
coil-only ``B.n`` is labelled as the vacuum boundary condition. For finite-beta
direct-coil rows, coil-only ``B.n`` is labelled ``diagnostic_only`` and the row
is blocked from production promotion unless the strict force components pass,
the final residual was freshly recomputed on the accepted state, and the
finite-beta virtual-casing boundary diagnostic was computed.
The rounded-square ``axis_kind="control_spline"`` bridge is now the default
because it keeps the intended side/corner axis controls independent of
``MPOL`` and ``NTOR`` before the VMEC Fourier projection. With the default
uniform side/corner controls it has the same low-bandwidth projection behavior
as ``axis_kind="spline"`` and reduces low-mode projection error relative to the
superellipse axis. It is still projected to VMEC Fourier coefficients, so large
straight sections plus localized stellarator corners remain a difficult
Fourier representation; using the control-spline envelope is a bandwidth
reduction, not a replacement for resolution closure. The square-axis
side/corner localization powers are also important:
the older ``side_power=corner_power=1.4`` stress shape has a high-mode tail,
while the current first-order default ``1.0`` keeps the same broad side/corner
geometry much closer to finite Fourier bandwidth.
The backend profiler records these choices as ``side_power`` and
``corner_power`` in its JSON configuration, and the summary table prints them
next to ``MPOL``, ``NTOR``, and ``NZETA`` so strict-resolution sweeps are not
mixed with shape-smoothing sweeps.
For production backend sweeps, ``--max-boundary-projection-error`` can enforce
the same kind of Fourier-closure gate used by the root example; omit it, or
pass ``none``, for diagnostic underresolved profiles.
The public helper
``vmec_jax.recommend_square_axis_stellarator_mirror_hybrid_resolution`` scans a
small finite ``MPOL``/``NTOR`` ladder for the current spline-smoothed target and
returns the lowest estimated-cost candidate whose projection error satisfies
the requested threshold. This is a boundary-representation gate, not a
nonlinear-convergence claim.
The first control-basis bridge is now available through
``vmec_jax.SquareAxisSplineControls`` and ``axis_kind="control_spline"``. It
defines the square-axis radial envelope with periodic spline controls that are
independent of ``MPOL`` and ``NTOR``, then projects the sampled target to a VMEC
Fourier boundary when building ``InData``. This is still a projection bridge,
not a solver-native spline basis, but it makes the intended control variables
explicit and testable before replacing the nonlinear solve representation.
Uniformly spaced controls are evaluated with a periodic trigonometric
interpolant so the default side/corner control set preserves the same
low-bandwidth projection behavior as the rounded ``axis_kind="spline"`` target;
irregular controls fall back to a periodic cubic Hermite interpolation.
A projection spot check on the first-order spline shape gives max component
boundary errors of about ``5.60e-6`` for ``MPOL=5, NTOR=12``, ``1.76e-9`` for
``5,20``, ``1.92e-11`` for ``5,25``, and ``3.5e-12`` for ``5,28`` or
``5,32``. Mixed higher-poloidal rows give comparable values:
``6,23`` is about ``3.33e-10``, while ``7,28`` and ``8,32`` are both around
``3.46e-12``. Therefore ``8,32`` is useful as a robustness/performance probe,
but it is not clearly improving boundary projection over ``7,28`` or
``5,28`` for the current first-order spline target. The older sharpened
``1.4`` shape, used by the already-running strict profiles before this update,
gave about
``1.27e-4``, ``4.76e-5``, ``2.41e-5``, ``1.44e-5``, ``8.98e-6``, and
``6.14e-6`` on the same ladder. Future strict profiles should therefore rerun
the first-order-weight geometry before spending more time on higher Fourier
mode counts for the sharpened stress shape.

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
The repository now exposes the same postsolve diagnostic through
``vmec_jax.free_boundary_validation.virtual_casing_diagnostics_from_run``.
It samples the solved LCFS, the total VMEC surface field, and the direct-coil
external field, then reports the virtual-casing external normal-field residual
and finite-beta pressure-balance residual. The square-coil backend profiler can
write this block for direct-coil rows with ``--virtual-casing-diagnostics``.
The flag is opt-in because the singular integral can be expensive and depends
on the optional ``virtual_casing_jax`` package; when the package is missing the
profile records a skipped status instead of failing the run.

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

DESC's source and PR history reinforce the same split. ``VacuumBoundaryError``
is the cheap vacuum normal-field objective, while finite-beta
``BoundaryError`` uses the total exterior field, including the plasma
contribution from virtual casing. DESC PRs in this area added the original
free-boundary objectives, multiple magnetic-field support, singular-integral
chunking, and zero-pressure special handling. Two open DESC efforts are also
important for our next steps: a discretization-error patch for
pressure-balance optimization and a non-singular BIE / improved
FFT-interpolation branch. The lesson for ``vmec_jax`` is to avoid treating
coil-only ``B.n`` as a finite-beta residual and to add accepted-boundary
pressure-balance/provider-parity diagnostics before claiming that a direct-coil
stall is purely an optimizer problem.

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
residual floors. Direct-coil rows produced with
``--virtual-casing-diagnostics`` also expose virtual-casing status,
external-normal residuals, pressure-balance residuals, and required/target
external-field RMS values in the same summary table.
The summary helper also recognizes direct-GPU result folders named
``square_coil_direct_gpu_*``. This keeps active cached-JIT speed probes from
collapsing to a generic ``launcher`` case name, and preserves ``MPOL``,
``NTOR``, ``NS``, ``NZETA``, and final-stage budget hints inferred from the
folder name. Completed stalled direct rows that have not already used
``freeb_jax_nestor_operator`` now recommend ``direct-gpu-jax-nestor`` as the
next profile kind; rows that already used it fall back to the ordinary
direct-GPU ``DELT``/stage-budget lane.
For staged solves, the same table now exposes ``stage_count``,
``stage_ns_array``, ``stage_niter_array``, ``stage_ftol_array``,
``stage_budget_total``, ``stage_budget_final``, ``current_stage_index``,
``current_stage_last_iter``, ``remaining_stage_budget``, and
``remaining_total_stage_budget``. Use ``stage_budget_final`` and
``remaining_stage_budget`` when judging an active final-grid row: live VMEC
iteration numbers are stage-local, so the current-stage budget is the useful
quantity for deciding whether a tail estimate can still fit in the active
``FTOL=1e-12`` stage.
Completed VMEC2000 rows with a flat tail above ``FTOL`` recommend another
``vmec2000`` follow-up row for the ``DELT``/stage-budget scan rather than
detouring through accepted-LCFS provider parity first. That keeps the reference
solver lane focused on whether the generated-``mgrid`` VMEC2000 problem can
reach the strict component gate before comparing direct and generated-field
providers on an accepted LCFS.
Completed JAX backend profiles also expose
``free_boundary_jax_nestor_operator_applied``,
``free_boundary_jax_nestor_operator_reason``,
``free_boundary_jax_nestor_operator_jitted``,
``free_boundary_jax_nestor_operator_cache_hit``, and
``free_boundary_jax_nestor_operator_time_s``. Treat a requested operator flag
as a setup condition and the ``*_applied``/``*_reason`` fields as the evidence
that the experimental operator was actually used on a vacuum update.
The profiler also records ``--virtual-casing-quad-factor``,
``--virtual-casing-chunk-size``, and
``--virtual-casing-target-chunk-size`` for finite-beta postsolve diagnostics.
Those knobs control the optional virtual-casing quadrature and memory use only;
they do not change the equilibrium solve or promote a coil-only vacuum
``B.n`` check to a finite-beta boundary condition.
Rows produced with ``--accepted-provider-parity`` also compare generated-mgrid
and direct-coil fields on the accepted JAX backend LCFS, not only on the input
boundary. The compact backend payload is ``accepted_provider_parity``; the
summary table exposes its status, sample label, field-vector RMS relative
difference, and coil-only ``B.n`` RMS relative difference. Use this diagnostic
when investigating late free-boundary stalls or pressure-balance
discretization errors, because a clean initial-boundary parity check does not
prove the two providers still agree on the moved LCFS.
For active VMEC2000 rows, the profiler and summary table now also report
``tail_plateau_*`` fields. ``flat_above_stage_ftol`` identifies tails whose
recent relative span is small while the last residual remains above the current
stage tolerance; those rows should be treated as stalled until a
``DELT``/stage-budget scan moves the floor.
The summary table also carries an explicit strict-evidence classifier:
``backend_role``, ``strict_evidence_status``, ``strict_evidence_blockers``,
``resolution_deck_status``, and ``resolution_deck_reasons``. A row is strict
production evidence only if the requested ``FTOL`` is ``1e-12`` or tighter,
all three final VMEC force components pass that tolerance, the appropriate
fresh residual/finite-beta promotion gates pass, and the cheap
projection/``NZETA``/``mgrid_nphi`` deck gate is production-ready. Rows run at
``FTOL=1e-8`` or ``1e-10`` are labelled ``non_strict_ftol`` even when they
converge to that looser target. Rows with bad Fourier projection, low
``NZETA``, or mgrid-plane mismatch are labelled ``diagnostic_underresolved``
even if their force components happen to be small.
The table also emits ``recommended_followup_profile_kind`` and
``recommended_followup_reason``. These columns translate the convergence,
resolution, grid, and accepted-LCFS provider-parity evidence into the next
profile family to run: wait for active rows, repair the resolution preflight,
widen the VMEC2000 mgrid, run accepted-provider parity, or run a direct-GPU
``DELT``/stage-budget probe. They are recommendations for the next diagnostic
row, not a replacement for inspecting the residual tail and physics plots.

Finite-beta mirror validation should check the sign of the magnetic response,
not only numerical convergence. Ideal MHD force balance gives the familiar
magnetic-pressure relation ``p + B^2/(2 mu0) ~= constant`` when field-line
tension is not the dominant transverse balance, so increasing plasma pressure
should diamagnetically reduce ``|B|`` in the high-pressure region. Recent
axisymmetric mirror modelling with Pleiades/RealTwin reports the same
qualitative behavior: self-consistent high-beta mirror equilibria show outward
flux-surface expansion and a central magnetic-field reduction from plasma
diamagnetism. The square-coil beta scan should therefore report both residual
convergence and a physical response check: higher beta should not merely move a
prescribed boundary, it should produce a solved free-boundary equilibrium with
the expected pressure-supported ``|B|`` depression and small external-normal
field residuals. Useful references for this validation target are the
Hammir/RealTwin mirror modelling paper
`arXiv:2411.06644 <https://arxiv.org/html/2411.06644v1>`__, the BEAM mirror
study in `Journal of Plasma Physics
<https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/prospects-for-a-highfield-compact-breakeven-axisymmetric-mirror-beam-and-applications/65480116EDB94296B25A52C7E5240EA8>`__,
and standard MHD magnetic-pressure balance notes such as
`Hutchinson/ANU chapter 6
<https://people.physics.anu.edu.au/~jnh112/AIIM/c17/chap06.pdf>`__.

The square-axis stellarator-mirror hybrid geometry now has a lower-bandwidth
``axis_kind="control_spline"`` bridge, or equivalently the rounded
``axis_kind="spline"`` envelope for the default uniform controls. The target is
still projected into VMEC Fourier boundary coefficients, but it replaces the
sharp polar-square/superellipse content with a smooth rounded-square envelope
before projection. This is the practical near-term way to reduce ``NTOR``
sensitivity; a true spline basis inside the VMEC solve would be a larger
solver reparameterization. The public
``square_axis_spline_radius_matrix`` helper exposes the linear map from the
few square-axis control radii to sampled radius values for fixed control
locations. This gives differentiable studies a compact control vector today,
while keeping the active VMEC/JAX and VMEC2000 comparison on the same Fourier
boundary interface. The companion
``square_axis_spline_control_fourier_matrix`` helper gives the chain-rule map
from those controls to projected VMEC boundary coefficients, which is the
needed bridge before introducing a solver-native control-basis state vector.
For square-coil production updates, pair it with
``square_axis_spline_symmetric_control_basis``. The default square basis maps
the eight side/corner control radii into two symmetry-preserving parameters,
while the stellarator-symmetric basis keeps the usual ``r(zeta)=r(-zeta)``
pairs. These reduced maps are the preferred diagnostic/optimization controls
before increasing Fourier mode count further.
The public helper
``square_axis_stellarator_mirror_hybrid_projection_error`` and the square-coil
profiler's ``boundary_projection`` JSON block now report the Fourier truncation
error, mode count, and recommended ``NZETA`` for the selected
``MPOL``/``NTOR``/fit-grid combination; the profile summarizer exposes this as
``boundary_mode_count``, ``boundary_recommended_nzeta``,
``boundary_proj_max``, and ``boundary_proj_rel``.
These metrics should be reviewed whenever changing ``MPOL``, ``NTOR``, or
``NZETA``: they diagnose input-boundary underfitting before the free-boundary
nonlinear solve is interpreted.
The profiler also has a cheap ``--resolution-diagnostics-only`` mode that
writes the ``boundary_projection`` and ``resolution_deck`` JSON blocks, then
exits before coil, mgrid, or equilibrium work. Use it as the first check after
changing mode counts or ``mgrid_nphi``; it records projection-gate status,
recommended ``NZETA``, and whether the generated mgrid toroidal plane count is
compatible with the VMEC ``NZETA`` grid. The same report includes
``control_basis`` metadata for ``spline`` and ``control_spline`` axes: the full
spline-control radii plus the square and stellarator-symmetric reduced bases.
This makes it clear whether a strict run is testing the intended compact spline
controls or only a larger Fourier projection deck.
It also includes ``control_fourier_map`` for the square-reduced spline controls:
side/corner labels, stacked ``4K x 2`` coefficient-Jacobian shape, singular
values, condition number, and column norms. This is a preflight diagnostic for
the solver-native reduced-control lane, not a nonlinear convergence claim.
Its nested ``candidate_bases`` block records the same conditioning diagnostics
for both the two-control square basis and the five-control
stellarator-symmetric basis.
The adjacent ``spline_bridge`` block states the current representation status:
``axis_kind="control_spline"`` uses periodic spline controls for the
real-space square-axis target, but the nonlinear free-boundary solve still
uses VMEC Fourier boundary coefficients. It can therefore smooth and reduce
the input geometry, but it is not yet a solver-native spline/control-basis
equilibrium.
The reusable source hook is
``SquareAxisControlFourierMatrix.project_boundary_delta(...)``. It returns a
``SquareAxisControlProjection`` with the fitted control update, reconstructed
boundary delta, residual delta, rank, singular values, relative residual, and
captured fraction.
Completed JAX backend rows add the complementary postsolve diagnostic
``boundary_reduced_control_projection``. It projects the actual accepted LCFS
coefficient displacement onto the same ``control_fourier_map`` and reports the
least-squares side/corner update, relative residual, and captured fraction.
The nested ``candidate_bases`` block also compares the two-control square basis
with the five-control stellarator-symmetric basis, so a low square capture can
be diagnosed as a basis-size issue before changing the nonlinear algorithm.
Use this after real solves to tell whether the observed free-boundary motion is
representable by the compact spline controls before moving the nonlinear
iteration into that reduced basis.
The root square-coil example now enforces
``MAX_BOUNDARY_PROJECTION_ERROR = 5e-12`` by default and uses
``MPOL=5, NTOR=28, NZETA=64`` as the production-style deck. This is strict
enough for ``FTOL=1e-12`` studies on the current spline-smoothed square target.
The latest preflight matrix gives the current deck classification:

.. list-table::
   :header-rows: 1

   * - deck
     - status
     - reason
   * - ``MPOL=5, NTOR=20, NZETA=48``
     - diagnostic-only
     - projection max ``1.763e-9`` exceeds ``5e-12``
   * - ``MPOL=5, NTOR=28, NZETA=48``
     - diagnostic-only
     - ``NZETA=48`` is below recommendation ``64``
   * - ``MPOL=5, NTOR=28, NZETA=64``
     - production-ready
     - projection max ``3.481e-12`` and mgrid-compatible
   * - ``MPOL=6, NTOR=32, NZETA=72``
     - production-ready
     - projection max ``3.468e-12`` and mgrid-compatible
   * - ``MPOL=5, NTOR=28, NZETA=64, mgrid_nphi=96``
     - diagnostic-only
     - ``mgrid_nphi`` is not a multiple of ``NZETA``

Lower-mode decks remain useful for diagnostic profiling, but they should set
the threshold to ``None`` or pass ``--max-boundary-projection-error none`` so
the report explicitly labels them as underresolved experiments. This guard
keeps Fourier boundary underfitting separate from nonlinear-solver or
direct-coil provider failures.
The same classification is now available without launching the backend
profiler:

.. code-block:: bash

   python tools/diagnostics/square_coil_resolution_matrix.py \
     --decks 5:20:48,5:28:48,5:28:64,6:32:72,7:28:auto,8:32:auto \
     --print-preflight-commands \
     --include-control-map

The matrix command uses the public
``vmec_jax.square_axis_resolution_deck_status`` helper, so scripts and
notebooks can apply the same projection/``NZETA``/``mgrid_nphi`` gate before
starting a strict ``FTOL=1e-12`` run.  Add
``--print-vmec2000-commands`` to emit generated-``mgrid`` VMEC2000 reference
commands for only the rows you want to run.  The optional
``--include-control-map`` column set reports the square two-control and
stellarator-symmetric five-control spline-map conditioning.  Use it to decide
whether a low-dimensional spline/control update is numerically credible before
changing the nonlinear solver's boundary update path.
After the strict-deck gate update, a finite
``--max-boundary-projection-error`` requires the whole ``resolution_deck`` to
be production-ready before any backend solve starts. Passing the projection
number is not enough: ``NZETA`` must also meet the square-axis recommendation
and the generated-``mgrid`` toroidal plane count must be compatible with the
VMEC ``NZETA`` grid. This makes user edits to ``MPOL``, ``NTOR``, ``NZETA``,
or ``mgrid_nphi`` fail fast unless the run is explicitly marked diagnostic by
using ``--max-boundary-projection-error none``.

For post-stall follow-up commands, use
``tools/diagnostics/square_coil_followup_commands.py``. The default
``--profile-kind vmec2000`` preserves the generated-``mgrid`` VMEC2000
reference scan. Use ``--profile-kind resolution-preflight`` when the summary
table recommends repairing the projection/``NZETA``/``mgrid_nphi`` deck before
running another equilibrium. Use ``--profile-kind provider-parity`` for the
next direct-coil/generated-mgrid comparison with both initial-boundary parity
and accepted-LCFS parity. Use ``--profile-kind full-backend`` only when
resources are available for direct, generated-mgrid, and VMEC2000 in one
profile. Use ``--profile-kind direct-gpu`` for direct-only cached-JIT GPU speed
probes; that mode intentionally skips generated mgrid and accepted-provider
parity.

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
   VMEC-style runs. Use ``--solver-mode parity``, ``--nvacskip 1``, and
   ``--nstep 1`` for convergence evidence; larger ``NVACSKIP`` values are speed
   experiments, and larger ``NSTEP`` values hide live VMEC2000 residual rows.
   For radial-resolution ladders, use a widened mgrid envelope and record
   ``vacuum_grid_exceeded_count`` before interpreting the residual floor. Keep
   the provider-parity block enabled unless the run is a pure solver-speed
   benchmark; it verifies that direct-coil and generated-mgrid boundary fields
   still match after resolution or coil changes. Use the
   tail-projection columns in the summary table to choose between extending the
   iteration budget and changing the resolution/schedule; do not interpret the
   projection as proof of convergence. Since the ``NTOR=12`` deck floors near
   ``1e-7`` and the improved ``NTOR=16`` deck slows near ``1e-8`` by 12000
   direct-coil iterations, do not spend more local CPU on longer single-stage
   runs at either same deck before changing resolution, schedule, or geometry
   representation. For pure direct-provider GPU profiling, use
   ``--jit-forces --coil-chunk-size 0 --jit-direct-sampler --skip-mgrid
   --skip-provider-parity`` so the run does not spend memory generating an
   unused mgrid and uses the cached direct-coil sampler. The profiler caches
   static direct-coil geometry by default because this is a forward CLI
   workflow; pass ``--no-direct-static-cache`` only for differentiable-provider
   parity checks. When a backend comparison still needs a generated mgrid, keep
   the mgrid write chunked even if the direct sampler uses the full-geometry
   JIT path. Add ``--accepted-provider-parity`` to backend-comparison rows
   when diagnosing a late stall; skip it for pure direct-GPU speed runs because
   it intentionally keeps the generated mgrid available and performs an
   accepted-LCFS direct/mgrid sample after the solve.
   For the first pressure-mixing A/B run, keep the same staged deck and add
   ``--freeb-anderson-pressure`` only to the direct-coil backend run; compare
   ``free_boundary_anderson_pressure_last_theta``,
   ``freeb_anderson_pressure_applied_stats``, final component residuals, and
   the tail projection against the non-Anderson profile before changing any
   other solver knob. For the next direct-GPU solver-kernel A/B, emit the
   cached-JIT direct row with ``--profile-kind direct-gpu-jax-nestor`` from
   ``tools/diagnostics/square_coil_followup_commands.py``. That adds
   ``--freeb-jax-nestor-operator`` while keeping the strict ``FTOL_ARRAY`` and
   staged ``NS_ARRAY`` deck fixed. Optional one-at-a-time switches are
   ``--freeb-include-edge``, ``--freeb-dense-solve-mode mode|grid``,
   ``--freeb-experimental-fouri-matrix``/``--no-freeb-experimental-fouri-matrix``,
   and ``--freeb-add-analytic-bvec``/``--no-freeb-add-analytic-bvec``. The
   profiler writes the selected values to ``configuration`` and each JAX
   backend's ``free_boundary_solver_overrides`` block, and it explicitly sets
   ``VMEC_JAX_FREEB_JAX_NESTOR_OPERATOR=0`` unless the flag is requested so
   ambient shell environment cannot contaminate a robustness comparison. The
   direct-GPU follow-up commands include ``--verbose-solver`` so active rows can
   be summarized from ``launcher.log`` before final JSON is written.
2. Re-run the square-coil beta ladder with per-beta checkpointing and the
   best-scored diagnostic fallback using the staged ``FTOL_ARRAY`` ending at
   ``1e-12``. Keep ``DELT=0.02``, ``NVACSKIP=1``,
   ``solver_mode="parity"``, and the VMEC-like
   ``FREE_BOUNDARY_ACTIVATE_FSQ=1e-3`` unless a benchmark shows a better value.
   The current robust default ladder is ``NS_ARRAY = 9, 13, 17`` with
   ``NITER_ARRAY = 4000, 8000, 24000`` and
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
4. Prioritize the first-order ``MPOL=5, NTOR=28`` / ``NZETA=64`` spline target
   before further increasing Fourier mode count. The projection scan shows that
   this lower-bandwidth deck is already at the same ``~3.5e-12`` representation
   floor as ``7,28`` and ``8,32`` for the current target, while the active
   ``8,32`` VMEC2000 profile has high startup cost before any force rows are
   available. If the ``5,28`` and ``7,28``/``8,32`` VMEC/VMEC2000 ladders still
   stall above the strict ``1e-12`` component gate, move the square-axis hybrid
   into a solver-native spline-basis lane instead of only increasing Fourier
   modes. The current code already supports explicit periodic side/corner spline
   controls through ``SquareAxisSplineControls`` and a symmetry-reduced
   ``SquareAxisControlBasis``, but those controls are still projected to Fourier
   coefficients before the solve. The next lane should keep the same
   direct-coil/free-boundary diagnostics and move the nonlinear update variables
   to that reduced spline basis, projecting to Fourier only for VMEC2000 parity
   or WOUT export. This keeps the primary ``vmec_jax`` path closer to the
   intended straight-side geometry while preserving a benchmarkable VMEC2000
   comparison.
5. Keep the optional virtual-casing postsolve diagnostic
   attached to the square-coil example outputs and enable
   ``--virtual-casing-diagnostics`` on direct-coil backend profiles that will be
   used as finite-beta evidence. The diagnostic accepts a solved run and
   direct-coil parameters, then reports the required external-field normal
   mismatch and finite-beta magnetic-pressure jump. It is optional at import
   time and should be skipped when ``virtual_casing_jax`` is not installed.
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
