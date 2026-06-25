Mirror Examples
===============

These examples exercise the experimental fixed-boundary mirror backend from a
source checkout.  They intentionally use low resolution and small iteration
budgets so they run quickly.

Run from the repository root:

```bash
python examples/mirror/fixed_cylinder.py --outdir results/mirror/cylinder
python examples/mirror/fixed_flared_tube.py --outdir results/mirror/flared
python examples/mirror/wham_vacuum_boundary.py --outdir results/mirror/wham
python examples/mirror/nonaxisymmetric_boundary.py --outdir results/mirror/nonaxisymmetric
python examples/mirror_two_coil_axisym.py --outdir results/mirror/two_coil_axisym
python examples/mirror_finite_current_pitch.py --outdir results/mirror/finite_current_pitch
python examples/mirror_free_boundary_circular_coils.py --outdir results/mirror/free_boundary_circular_coils
python examples/mirror_fixed_boundary_solve_diagnostic.py --outdir results/mirror/fixed_boundary_solve_diagnostic
python examples/mirror_manufactured_fixed_boundary.py --outdir results/mirror/manufactured_fixed_boundary
python examples/mirror_stellarator_hybrid_boundary.py --outdir results/mirror/stellarator_hybrid_boundary
python examples/mirror_implicit_sensitivity.py --outdir results/mirror/implicit_sensitivity
python examples/mirror_implicit_solve_benchmark.py --outdir results/mirror/implicit_solve_benchmark
python examples/toroidal_stellarator_mirror_hybrid.py --outdir results/toroidal_stellarator_mirror_hybrid
python examples/toroidal_stellarator_mirror_hybrid_convergence.py --outdir results/toroidal_stellarator_mirror_hybrid_convergence
python examples/toroidal_stellarator_mirror_hybrid_square_coils_free_boundary.py
python examples/mirror_solver_comparison.py --outdir results/mirror/solver_comparison
python examples/mirror_residual_newton_convergence_grid.py --outdir results/mirror/residual_newton_convergence_grid
```

The physical mirror examples write a mirror-native ``mout_*.nc`` file and,
unless ``--no-plots`` is passed, a set of PNG diagnostics including horizontal
``z``-axis geometry, boundary magnetic-field direction with field-line traces,
``|B|``, beta, cap-to-cap field-line pitch, magnetic-well-proxy, and
residual/step-history figures.  The manufactured validation example writes
metrics and targeted convergence/geometry/``|B|`` plots rather than a
production ``mout`` file.  These are research fixtures for the scalar-pressure
fixed-boundary mirror path, not WHAM predictive modelling tools.  For
physically axisymmetric mirrors use the cylinder, flared-tube, or WHAM
examples; the nonaxisymmetric example is a solver/plot stress test.
The two-coil and finite-current root examples also record compact
mirror-Boozer-like extrema in their metrics JSON files so no-plot benchmark
runs still retain surface-average ``|B|``, ripple, mirror-ratio, pitch, and
well-proxy summaries.

The same standard figure bundle is available from the CLI:

```bash
vmec --plot results/mirror/two_coil_axisym/mout_two_coil_axisym.nc --outdir results/mirror/two_coil_axisym/cli_figures
```

For mirror ``mout_*.nc`` files, ``vmec --plot`` writes nested ``r-z`` surfaces,
cross sections, 3-D boundary ``|B|`` with field-line overlays, boundary field
direction, ``|B|`` maps, Jacobian, pressure/beta, radial diagnostics, and
mirror-Boozer-like surface-average/pitch diagnostics, and residual/force
history plots.

The root-level ``examples/mirror_two_coil_axisym.py`` script is the first
analytic benchmark example: it builds a fixed boundary from the closed-form
on-axis field of two equal circular coils, overlays the mirror on-axis ``B_z``
against that analytic expression, draws the coils in the 3-D views, compares
low-radius off-axis ``B_r``/``B_z`` against the circular-loop Biot-Savart field,
and writes a small ``ns``/``nxi`` convergence study.

The root-level ``examples/mirror_finite_current_pitch.py`` script uses the same
two-coil fixed boundary with nonzero ``I'`` so the boundary field-line traces
have visible cap-to-cap pitch.

The root-level ``examples/mirror_stellarator_hybrid_boundary.py`` script is the
first straight-axis hybrid geometry fixture. It keeps the mirror axis straight
in ``z`` while a central elliptical cross-section rotates by one
field-period-like angle and tapers smoothly into circular mirror end sections.
The metrics JSON reports end circularity, midplane theta variation, up-down
symmetry error, residual/force diagnostics, standard plot paths, and explicit
``hybrid_fixture_kind``/``final_hybrid_target_kind`` labels. The final
stellarator-mirror hybrid target is a separate toroidal lane: mirror-like side
arcs connected through stellarator-like corner arcs using ordinary VMEC/JAX
toroidal boundary coefficients and solver paths.

The root-level ``examples/toroidal_stellarator_mirror_hybrid.py`` script starts
that toroidal lane. It writes a VMEC-compatible ``input.*`` file whose LCFS has
mirror-like side arcs and localized rotating-ellipse stellarator corners, writes
boundary metrics and plots, including a side/corner weight and principal-axis
orientation diagnostic, and can optionally run the ordinary toroidal
fixed-boundary solver with ``--run-solve`` to produce a ``wout_*.nc`` plus
standard VMEC plots.

The companion ``examples/toroidal_stellarator_mirror_hybrid_convergence.py``
script scans ``ns`` and ``mpol:ntor`` pairs for the same generated toroidal
hybrid input. By default it writes lightweight JSON/CSV boundary-fit reports;
pass ``--run-solve`` to run the ordinary fixed-boundary driver for each row and
record runtime, iteration count, final ``fsq``, residual history, convergence
status, best ``fsq`` reached, aspect, mean iota, magnetic-well proxy, and a
``wout_*.nc``. With plots enabled, solved rows also write ``fsq`` history plus
iota and Mercier ``DWell`` profile figures. The no-solve path also records
target and fitted side/corner principal-axis orientation spans, covariance
anisotropy ranges, valid-axis fractions, and an orientation-preservation plot
when plots are enabled. Pass ``--run-vmec2000`` to run the
same generated inputs through the local VMEC2000 executable and record parsed
``threed1`` residual histories beside the VMEC/JAX rows. Use
``--solver-mode parity --no-use-scan`` for the closest VMEC2000-control
comparison, or keep the default accelerated mode for the fast CLI path. The
CSV records requested ``ftol``, VMEC/JAX strict and total-``fsq`` convergence
flags, and VMEC2000 WOUT residual components when available. When plots are
enabled and both solvers ran, the example also writes a final
``fsqr``/``fsqz``/``fsql`` component comparison. CSV/JSON rows also label the
VMEC/JAX and VMEC2000 initialization policies used for the comparison, including
whether VMEC/JAX used the raw input-axis branch or inferred a missing axis from
the boundary. With ``--run-solve`` enabled, ``direct_initial_*`` fields evaluate
the VMEC/JAX residual on the pre-iteration initial state, while ``initial_*``
fields are the first stored VMEC/JAX solve-history row. VMEC2000 comparisons
use the first parsed ``threed1`` row. The direct-initial diagnostic can be
disabled with ``--no-direct-initial-residual`` for large scans. Pass
``--nstep 1`` when running ``--run-vmec2000`` to make VMEC2000 print every
iteration into ``threed1``; pass ``--full-solver-diagnostics`` to keep full
VMEC/JAX terminal step histories or scan time-step histories and write the
step-diagnostics plot. Add
``--no-cli-finish`` when the goal is raw VMEC-style trajectory parity rather
than the faster CLI finish/fallback policy. The residual-history plot then
aligns VMEC/JAX and VMEC2000 by actual iteration labels. Rows also expose CLI
finish budgets, finish residuals, finish modes, and fallback flags so a compact
fast-path result is not mistaken for a raw fixed-iteration trajectory.
``initial_fsq_ratio_direct_initial`` records how far the first stored VMEC/JAX
history row is from the pre-iteration direct residual. Treat mean-iota and
direct-initial residual agreement as useful regression signals, and use the
history fields to understand convergence after solver startup. For
toroidal-hybrid geometry refinement, pass ``--shape-cases default,sharp`` to
scan the default boundary and a sharpened side/corner preset in one run; use
``5:20`` or higher ``mpol:ntor`` pairs when asserting exact fit for the
sharpened preset. Pass ``--resolution-preset smoke``, ``promotion``, or
``target`` to use a named resolution ladder. The ``target`` preset expands to
``ns = 7,9,15`` and ``mpol:ntor = 5:20,6:24`` and labels rows as target-ladder
inputs. Office GPU runs of that ladder reached total-``fsq`` convergence at
``ftol=1e-8`` for all six rows with VMEC2000 outputs present. Rows report the
largest ``fsqr``/``fsqz``/``fsql`` component, that component divided by
requested ``ftol``, and the strict-component bottleneck when one remains. A
targeted 160-iteration office closure run then strict-converged all six target
rows in 124-134 iterations, with the largest VMEC/JAX component below
``0.98`` times requested ``ftol``. Use
``--case-filter '*ns015*'`` or another comma-separated shell pattern to run a
subset of the generated case names when splitting the target campaign across
machines. After split campaigns finish, pass ``--aggregate-json`` one or more
existing convergence JSON files to merge chunked rows, de-duplicate by case,
write a compact aggregate CSV/JSON report with strict-component blocker
counts, and optionally regenerate the residual/history plots without rerunning
VMEC/JAX or VMEC2000.

The root-level
``examples/toroidal_stellarator_mirror_hybrid_square_coils_free_boundary.py``
script keeps all user inputs in a top-of-file parameter block. It builds a
closed square array with ``N`` circular or elliptical coils per side (``N=4`` by
default, so 16 total coils), writes one direct-coil free-boundary VMEC input per
beta value, runs ``vmec_jax.run_free_boundary`` with nonzero toroidal current
and a staged ``NS_ARRAY``/``NITER_ARRAY``/``FTOL_ARRAY`` schedule ending at
``FTOL=1e-12``, and writes one ``wout_*.nc`` per beta. The default
``PHIEDGE`` sign is negative for the positive-current square-coil orientation,
matching the raw VMEC2000 generated-``mgrid`` vacuum sign check. The default
iteration ladder is ``NS_ARRAY = 9, 13, 17``,
``NITER_ARRAY = 4000, 8000, 12000``, and
``FTOL_ARRAY = 1e-8, 1e-10, 1e-12``, with ``DELT = 0.02`` and
``NVACSKIP = 1``. The default
square axis uses the low-bandwidth rounded ``axis_kind="spline"`` profile before
VMEC Fourier projection, which is less sensitive to ``NTOR`` than the sharper
polar superellipse. The default ``NZETA`` follows
``recommended_square_axis_nzeta``; underresolved production-style example runs
raise before solving because ``NTOR=12, NZETA=16`` was observed to fail while
``NZETA=32`` completed the same VMEC2000 generated-``mgrid`` case. The plots use
the solved VMEC states: 3-D coils plus solved LCFS
and field-line traces, top-view solved boundaries, side/corner cross sections,
solved-boundary ``|B|``, and residual/iota diagnostics. The metrics JSON records
convergence status, force components, free-boundary ``B.n`` diagnostics, WOUT
paths, beta scan rows, solver objective-history extrema, bad-Jacobian/reset
counts, best fresh free-boundary residuals, and a stall classification. Current
direct-coil square-hybrid runs are therefore treated as explicit convergence
diagnostics: if the final recomputed force components miss the requested
``FTOL``, the run is labelled as ``not_converged_or_max_iter`` rather than being
presented as a production equilibrium. For finite-beta cases, coil-only
``B.n`` is not the physical free-boundary target because the plasma field also
contributes at the interface. Use the final VMEC force residuals, total-pressure
balance, and eventually a virtual-casing plasma-field diagnostic for promotion
claims; coil-only ``B.n`` is a vacuum check. Long beta scans checkpoint the
summary CSV and metrics JSON after each beta by default, so a later
high-beta stall or interrupted SSH session still leaves the completed lower-beta
evidence on disk.

The active convergence plan and promotion gates for this lane are kept in
``docs/mirror/direct_coil_free_boundary_convergence.rst``. In short, a
finite-beta direct-coil row is a production candidate only when the live final
state has fresh active free-boundary coupling, every final force component
meets the requested ``FTOL``, the LCFS changes with beta, and a total-field
pressure-balance diagnostic agrees. Coil-only ``B.n`` remains a vacuum check.

The default activation threshold is now VMEC-like
(``FREE_BOUNDARY_ACTIVATE_FSQ = 1e-3``), and the solver now blocks ``LFREEB``
convergence until the free-boundary vacuum/edge coupling has actually turned
on. Direct-coil free-boundary convergence candidates are also rechecked with a
fresh external-field sample and the current plasma-current normalization before
the solve is allowed to exit; rejected candidates are reported through the
``free_boundary_fresh_convergence_*`` metrics. Older coarse review evidence at
(``NS=9, MPOL=5, NTOR=12``), beta ``0%``, ``1%``, ``3%``, and ``5%`` reach
strict ``FTOL=1e-8`` active free-boundary convergence in the default scan. A
5000-iteration office run with the fresh direct-coil gate makes beta ``7%`` the
first high-beta stall in this configuration: its final fresh ``fsqr`` remains
just above tolerance, while beta ``8%`` through ``10%`` have larger
restart-limited ``fsqr`` floors. These rows remain diagnostics rather than
strict production equilibria. The plot bundle includes
near-axis ``|B|`` and mirror-ratio trends so finite-beta scans can be compared
against the expected diamagnetic field-reduction / mirror-ratio-increase trend
from linear-trap mirror literature, instead of relying only on LCFS-averaged
``|B|``.

For direct-provider versus mgrid/VMEC2000 profiling, use::

  python tools/diagnostics/profile_square_coil_free_boundary.py \
    --ftol 1e-12 --max-iter 12000 --phiedge -0.04 \
    --solver-mode parity --nvacskip 1 --delt 0.02 \
    --mpol 6 --ntor 23 --nzeta 64 --axis-kind spline \
    --ns-array 9,13,17 --niter-array 4000,8000,12000 \
    --ftol-array 1e-8,1e-10,1e-12 \
    --mgrid-nr 72 --mgrid-nz 56 --mgrid-nphi 64 \
    --mgrid-padding-fraction 1.2 --mgrid-min-padding 0.5 \
    --run-vmec2000

The report stays under ignored ``results/`` paths and records ``vmec_jax``
direct-coil, ``vmec_jax`` generated-mgrid, and optional raw VMEC2000
generated-mgrid residuals for the same square-coil field. To profile a staged
VMEC-style ladder without editing the example, add for example
``--ns-array 9,13,17 --niter-array 4000,8000,12000 --ftol-array 1e-8,1e-10,1e-12``.
Use larger ``--nvacskip`` only as a speed experiment; for convergence review,
``--nvacskip 1`` avoids stale free-boundary residuals on this square-hybrid
Fourier deck. For ``NS`` ladders above the initial surface, use a widened mgrid
envelope and check the reported ``vacuum_grid_exceeded_count`` before
interpreting the residual floor, for example::

  --delt 0.02 --mgrid-nr 72 --mgrid-nz 56 --mgrid-nphi 64 --mgrid-padding-fraction 1.2 --mgrid-min-padding 0.5

For direct-coil-only GPU profiling, add ``--jit-forces --coil-chunk-size 0
--skip-mgrid --skip-provider-parity``. The default chunk size of ``512`` is the
conservative host-forward path; chunk size ``0`` uses the cached full-geometry
JIT direct-coil sampler and is the intended mode for direct-provider GPU
comparisons. Backend-comparison profiles that still need a generated mgrid keep
the mgrid write chunked even when the direct sampler uses chunk size ``0``.

The best completed VMEC2000 reference so far uses ``MPOL=6, NTOR=23,
NZETA=64`` with the spline square-axis projection. A staged
``NS=9 -> 13 -> 17`` run on a widened generated ``mgrid`` reaches best sampled
summed physical residual about ``1.86e-11`` and final summed physical residual
about ``2.19e-11`` after a 24000-iteration final-stage budget, with no
vacuum-grid overflow. This is still not a per-component ``1e-12`` production
solve, but it is several orders below the older ``NTOR=16`` staged floor.
``DELT=0.01`` was worse on the lower-mode schedule because the coarse stage
underconverged; the next comparison target is the direct-coil provider on the
same high-mode staged deck and then the ``MPOL=7, NTOR=28`` / ``MPOL=8,
NTOR=32`` spline ladders if the direct path shows the same floor.

Summarize one or more reports with::

  python tools/diagnostics/summarize_square_coil_profiles.py \
    results/square_coil_freeb_backend_profile_* --markdown

The root-level ``examples/mirror_free_boundary_circular_coils.py`` script is a
free-boundary planning fixture. It builds ESSOS-compatible circular-loop direct
coil parameters, samples the external field on the mirror axis and side
boundary, writes a reusable JSON setup for the default 1%, 3%, and 10%
beta-scan cases, builds the initial fixed-boundary flux-tube boundary from the
sampled on-axis field, and plots the coils, boundary, on-axis field comparison,
and boundary ``|B|``. Pass ``--run-fixed-boundary-baseline`` to write one
low-resolution fixed-boundary ``mout`` per beta case as a controlled pre-LCFS
baseline, plus side-boundary normal-field and total-pressure imbalance
diagnostics against the external coils.

With ``--run-lcfs-pilot`` and ``--run-fixed-boundary-baseline``, the example
applies low-resolution candidate LCFS updates and reports actual before/after
diagnostics. The default ``--lcfs-proposal-mode best_predicted`` scores local
pressure, shape-preserving scale, normal-field-slope, mixed
scale/normal-field, and no-op candidates using a dimensionless merit with
pressure-balance and normalized normal-field terms. Pass
``--lcfs-proposal-mode coupled`` to run short fixed-boundary trial solves for
the allowed non-noop candidates and choose the lowest realized score against a
no-op fallback. The coupled score combines the realized LCFS merit ratio with a
nonnegative ``final_fsq`` growth penalty controlled by
``--lcfs-coupled-fsq-weight``. Pass
``--lcfs-require-bnormal-nonincrease`` to enable a stricter guard: candidates
that increase exact coil-resampled ``B_ext.n`` RMS are filtered out, the mixed
scale/normal-field candidate is selected when it improves merit while satisfying
the guard, and otherwise the explicit no-op candidate records a skipped pilot
row. Metrics JSON rows include ``lcfs_update_allowed_strategies`` and
``lcfs_update_rejection_reason`` fields so downstream scripts can distinguish
accepted, rejected, and guard-limited pilot steps. Top-level metrics also
record ``workflow_status``, ``free_boundary_solve_status``,
``beta_scan_requested_percent``, ESSOS-compatible direct-coil metadata, and
aggregate LCFS pilot counts so benchmark scripts can validate that the 1%, 3%,
and 10% beta cases were actually exercised. As of schema version ``0.13``,
``free_boundary_solve_status`` can distinguish not-run, converged, and
not-converged pilot or coupled-loop workflows; convergence requires every
requested beta row to stop on ``target_merit``. Multi-step pilots can stop on an
explicit target merit with ``--lcfs-pilot-target-merit`` or on small accepted
merit improvement with ``--lcfs-pilot-stagnation-rtol``; each pilot row records
a ``stop_reason`` and each beta row records ``lcfs_pilot_stop_reason``. Use
``--lcfs-pilot-fsq-growth-limit`` to reject an otherwise merit-improving pilot
when its fixed-boundary ``final_fsq`` grows beyond a configured multiple of the
baseline row. When plots are enabled and baseline rows exist, the example also
writes a cross-beta summary figure comparing
pressure-balance RMS, external normal-field RMS, LCFS merit, and final ``fsq``
before and after pilot updates. This is still an LCFS pilot workflow, not a
converged free-boundary equilibrium solve. The public
``mirror_lcfs_residual`` helper returns the normalized pressure-balance and
external-normal-field residual vector behind the scalar LCFS merit, which is
the target vector for the next true coupled free-boundary solve lane. The
public ``mirror_free_boundary_least_squares_step`` helper now adds the first
line-searched boundary-coefficient step on top of the combined equilibrium plus
LCFS residual vector. It uses central finite differences so CLI workflows can
exercise the coupled residual contract before the fully differentiable solve
path replaces those derivatives with implicit/JAX/adjoint variants. The
package-level ``mirror_free_boundary_guarded_least_squares_loop`` helper owns
the reusable repeated-step guard policy through state and trial callbacks,
while this root example still owns the host-side fixed-boundary trial solve and
plot/report generation. For reduced residual-vector prototypes that are already
pure JAX functions of boundary parameters,
``mirror_free_boundary_residual_vector_jacobian_jax`` provides forward,
reverse, or automatic JAX Jacobian selection beside the host-side finite
difference helper. ``mirror_free_boundary_residual_vector_least_squares_step``
uses the same vector residual contract and can choose the
``finite_difference`` or ``jax`` Jacobian backend for one damped, line-searched
step.

Use finite differences for the current host-side CLI workflows that call fixed
boundary solves, write MOUT files, or invoke plotting/report callbacks. Use
``jacobian_backend="jax"`` with ``jax_mode="auto"`` for reduced residual-vector
prototypes that are already pure JAX functions of boundary parameters. The
automatic mode uses forward-mode differentiation when the number of boundary
parameters is no larger than the residual-vector length, and reverse mode for
smaller residual or scalar-like targets. The benchmark
``examples/mirror_free_boundary_vector_ls_benchmark.py`` compares the finite
difference, JAX forward, JAX reverse, and JAX automatic routes on the same
reduced free-boundary residual contract. The public
``mirror_free_boundary_residual_vector_least_squares_solve`` helper repeats
that reduced residual-vector step with target-residual, rejected-step,
stagnation, and max-step stop reasons for compact nonlinear prototypes. Step
and solve rows report the selected JAX mode, Jacobian rank, nullity,
conditioning, singular values, selected ridge candidate, and predicted versus
realized residual reduction so poorly conditioned or over-aggressive boundary
parameterizations are visible before they are used in expensive coupled
fixed-boundary trials.

The circular-coil beta-scan metrics use the compact schema
``mirror_free_boundary_circular_coil_beta_scan`` version ``0.13``. The top-level
JSON records the workflow status, direct-coil metadata, requested beta list,
setup JSON path, aggregate pilot counts, optional LS boundary-step settings,
LS ridge-candidate settings, the LS boundary polynomial degree, the optional
ordered ``--ls-boundary-polynomial-degree-candidates`` list, the
``--ls-boundary-inner-solve-steps`` setting, figure paths, and
``fixed_boundary_baseline_rows``. It also embeds
``summary_rows``, the same compact baseline/last-accepted/final-trial table
written to CSV. Each beta row records fixed-boundary residual and LCFS metrics,
the selected next LCFS update, all candidate-update summaries, per-beta pilot
summary fields, optional ``ls_boundary_step`` diagnostics from
``--run-ls-boundary-step``, and ``lcfs_pilot_rows``. The LS diagnostic fits the
baseline side boundary to an even polynomial ``[r0, a2, a4, ...]`` through
``--ls-boundary-polynomial-degree``. Degree 4 preserves the original
``[r0, a2, a4]`` path; higher degrees use a tabulated axisymmetric boundary for
the realized trial. The diagnostic evaluates one line-searched least-squares
step using the combined residual vector, and when plots are enabled writes a
residual-component/backtracking figure. Step rows report Jacobian rank,
nullity, conditioning, singular values, selected ridge, tried ridge
candidates, and predicted/actual reduction fractions. Pass
``--run-ls-boundary-coupled-trial`` with ``--run-ls-boundary-step`` to rerun the
fixed-boundary solve on the LS-selected polynomial boundary and record realized
``fsq``, normalized force, LCFS merit ratio, and optional trial plots. Pass
``--run-ls-boundary-coupled-loop`` to repeat realized LS-selected boundary
updates with target-merit, stagnation, and ``fsq`` growth guards; loop rows
record each LS step, realized trial, acceptance decision, stop reason, and
optional per-step plots. When polynomial-degree candidates are supplied, each
beta row tries them in order, stops at the first attempt that reaches
``target_merit``, and otherwise keeps the attempt with the lowest realized LCFS
merit. The JSON records the candidate list, selected degree, and compact
per-degree attempt summaries. The loop normally follows the one-step line-search
update. When ``--ls-boundary-inner-solve-steps`` is greater than 1, the example
also runs the reduced residual-vector nonlinear LS solver on the frozen
residual before each realized fixed-boundary trial. The realized selector keeps
the line-search candidate when it passes the loop guards and uses the inner
solve as a fallback when the line-search path stalls. Step rows record
``inner_solve_rows``, selected ridge diagnostics, residual history, and whether
the realized trial used the inner-solve candidate.

A low-resolution target-merit run with
``--baseline-maxiter 5``, ``--ls-boundary-max-relative-step 0.05``,
``--ls-boundary-coupled-loop-target-merit 0.1``,
``--ls-boundary-coupled-loop-fsq-growth-limit 1.5``, and
``--ls-boundary-inner-solve-steps 2`` reaches the converged
``free_boundary_solve_status`` for the default 1%, 3%, and 10% beta rows, with
final LCFS merit around ``0.064`` for 1% beta and ``0.047`` for 3% and 10%
beta in the local test grid. This is reduced circular-coil diagnostic evidence,
not a full promoted production free-boundary LCFS solver. Each
pilot row always
contains ``accepted``, ``rejection_reason``, ``stop_reason``,
``lcfs_merit_improvement_fraction``, final residual/``fsq`` diagnostics when a
trial solve ran, ``fsq_growth_ratio`` relative to the beta row baseline, and
the next candidate-update summary. Coupled-mode pilot rows also include
``coupled_trial_rows`` with one compact realized score row per tried strategy.
Each beta row also reports final/best pilot ``fsq`` growth ratios plus
``lcfs_pilot_last_accepted_*`` fields. Rejected
pilot rows are kept in JSON for audit, but the summary plot draws the last
accepted pilot state when a later trial is rejected. The same run also writes
``free_boundary_circular_coils_beta_scan_summary.csv`` with baseline,
last-accepted, and final-trial columns for quick ESSOS comparison reports. Use
``--lcfs-pilot-fsq-growth-limit 1.0`` as a strict residual-regression
diagnostic; at the current low resolution, ``1.1`` is the pragmatic tolerance
that keeps the first accepted 3% and 10% pilot updates while still rejecting the
next residual-growing trial.

The root-level ``examples/mirror_fixed_boundary_solve_diagnostic.py`` script
runs an actual L-BFGS fixed-boundary relaxation from a perturbed interior state.
Its default diagnostic uses ``ns_array=31``, ``maxiter=2000``, and explicit
``ftol=1e-12``/``gtol=1e-12`` and writes a JSON table with optimizer status,
iteration counts, residuals, ``fsq``, and plot paths. Pass
``--optimizer residual_newton`` to exercise the axisymmetric matrix-free
residual-Newton path instead of the scaled L-BFGS-B path. The residual-Newton
path defaults to a VMEC-like reduced-coordinate tridiagonal preconditioner,
with radial/lambda smoothing, an open-``xi`` radius smoother adapted to the
mirror cap constraints, and an adaptive inner ``lsmr`` iteration policy; pass
``--residual-preconditioner none`` for baseline unpreconditioned studies or
``--residual-linear-maxiter-policy fixed`` for controlled fixed-budget studies.
Pass ``--residual-linear-solver dense_lstsq`` on small grids to use the dense
reduced Hessian as a reference solve when diagnosing whether the matrix-free
Krylov correction is limiting convergence, or
``--residual-linear-solver lsqr`` to compare the alternate SciPy least-squares
Krylov iteration against the default ``lsmr`` path. Pass
``--residual-linear-solver block_dense_lstsq`` on small grids to solve the
radius and lambda dense-Hessian blocks separately as a block-correction
reference. Pass ``--residual-linear-solver block_lsmr`` to keep the same
radius/lambda split but solve each block with matrix-free LSMR; this is the
scalable diagnostic path for testing whether split corrections can approach
the dense block reference without materializing the full Hessian. On small
matrix-free runs, ``--residual-compare-dense-step`` also records the
dense-reference step norm, cosine, and relative error for the last Newton
correction.
Finite-current diagnostics can also pass
``--residual-preconditioner radial_xi_lambda_xi_tridi`` to smooth lambda
updates along the open axial coordinate when the residual decomposition is
lambda dominated. In current-carrying two-coil probes, pair that mode with
``--residual-xi-alpha 1.0`` before increasing resolution or outer iteration
budgets.

The root-level ``examples/mirror_manufactured_fixed_boundary.py`` script solves
a sourced manufactured fixed-boundary problem with a known stationary state. It
uses the same reduced-coordinate layout and geometry scaling as the mirror
solver, then applies an exact-Hessian damped residual iteration to verify that a
perturbed projected state can reach the requested projected ``gtol``.

The root-level ``examples/mirror_implicit_sensitivity.py`` script is the first
differentiability example. It manufactures an exact tiny-grid reduced root with
a linear source and small state ridge, computes the sensitivity with the
forward implicit wrapper, computes a custom-VJP source gradient with the
implicit adjoint, solves an independently perturbed source problem, and
compares the finite-difference state change against the implicit result. The
default solve method is the dense reference; pass ``--solve-method
matrix_free_cg`` to exercise the matrix-free JAX CG path on the same wrappers.
With plots enabled it writes a component comparison figure for the reduced
sensitivity vector.

The root-level ``examples/mirror_implicit_parameter_gradients.py`` script
extends that differentiability check to source, pressure-profile,
current-profile, flux-profile, and polynomial-boundary parameters. It compares
custom VJP directional derivatives against forward sensitivity contractions and
separately solved finite-difference roots, then writes JSON metrics and an
optional summary plot. The default solve method is the dense reference; pass
``--solve-method matrix_free_cg`` to exercise the same custom-VJP contract with
the matrix-free JAX CG linear solve.

The root-level ``examples/mirror_implicit_solve_benchmark.py`` script benchmarks
the same forward implicit wrapper over a small ``ns``/``nxi`` ladder. It writes
JSON/CSV rows comparing dense and matrix-free JAX CG runtime, Python-side peak
memory, linear residual, and relative error against the dense reference. With
plots enabled it writes a compact runtime/memory/error summary figure.

The root-level ``examples/mirror_solver_comparison.py`` script compares the
production gradient-descent, scaled L-BFGS-B, and residual-Newton paths on
small cylinder and two-coil fixed-boundary cases, and includes the sourced
manufactured residual-Newton gate in the same JSON/plot report. With plots
enabled it also writes the standard mirror plot bundle for the residual-Newton
physical cases, including the 3-D boundary, field-line overlays, ``|B|``,
cross sections, and residual history.

The root-level ``examples/mirror_residual_newton_convergence_grid.py`` script
runs two-coil residual-Newton convergence grids over ``ns``, ``nxi``, outer
iteration budget, inner ``lsmr`` iteration budget, and preconditioner mode. It
writes JSON metrics, residual heatmaps/budget plots, preconditioner comparison
plots, residual-component plots that split radius/lambda and cap/interior
contributions, and the standard mirror plot bundles for both the best-residual
row and the highest-resolution, highest-budget row. It can also run
``--residual-linear-solver dense_lstsq`` for small exact-Hessian reference
rows, ``--residual-linear-solver block_dense_lstsq`` for block-correction
reference rows, ``--residual-linear-solver block_lsmr`` for matrix-free split
radius/lambda block rows, or ``--residual-linear-solver lsqr`` for LSQR/LSMR
comparisons. Its default policy is
``fixed`` so the requested ``--residual-linear-maxiter-array`` values remain
literal; pass ``--residual-linear-maxiter-policy adaptive`` to exercise the
production adaptive inner budget in the same report. Pass ``--i-prime`` with a
nonzero value to run the same decomposition on a finite-current, pitched-field
case. Those finite-current runs are diagnostic artifacts for the current
axisymmetric residual-Newton path: they verify nonzero lambda residual behavior
and write field-line plots. For lambda-dominated ``block_lsmr`` studies, pass
``--residual-block-lambda-maxiter`` to give the lambda block a larger Krylov
budget than the radius block without spending the same iterations on both
blocks. JSON rows also include compact iterative linear-solve diagnostics such
as the stop code, actual iteration count, residual norm, normal-equation
residual norm, and condition estimate when a Krylov solver is used. With
``--residual-compare-dense-step``, rows also include dense-reference step
comparison metrics.
