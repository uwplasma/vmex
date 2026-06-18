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
python examples/toroidal_stellarator_mirror_hybrid.py --outdir results/toroidal_stellarator_mirror_hybrid
python examples/toroidal_stellarator_mirror_hybrid_convergence.py --outdir results/toroidal_stellarator_mirror_hybrid_convergence
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

The same standard figure bundle is available from the CLI:

```bash
vmec --plot results/mirror/two_coil_axisym/mout_two_coil_axisym.nc --outdir results/mirror/two_coil_axisym/cli_figures
```

For mirror ``mout_*.nc`` files, ``vmec --plot`` writes nested ``r-z`` surfaces,
cross sections, 3-D boundary ``|B|`` with field-line overlays, boundary field
direction, ``|B|`` maps, Jacobian, pressure/beta, radial diagnostics, and
residual/force history plots.

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
symmetry error, residual/force diagnostics, and standard plot paths. The final
stellarator-mirror hybrid target is a separate toroidal lane: mirror-like side
arcs connected through stellarator-like corner arcs using ordinary VMEC/JAX
toroidal boundary coefficients and solver paths.

The root-level ``examples/toroidal_stellarator_mirror_hybrid.py`` script starts
that toroidal lane. It writes a VMEC-compatible ``input.*`` file whose LCFS has
mirror-like side arcs and localized stellarator-like corner shaping, writes
boundary metrics and plots, and can optionally run the ordinary toroidal
fixed-boundary solver with ``--run-solve`` to produce a ``wout_*.nc`` plus
standard VMEC plots.

The companion ``examples/toroidal_stellarator_mirror_hybrid_convergence.py``
script scans ``ns`` and ``mpol:ntor`` pairs for the same generated toroidal
hybrid input. By default it writes lightweight JSON/CSV boundary-fit reports;
pass ``--run-solve`` to run the ordinary fixed-boundary driver for each row and
record runtime, iteration count, final ``fsq``, residual history, convergence
status, best ``fsq`` reached, aspect, mean iota, magnetic-well proxy, and a
``wout_*.nc``. With plots enabled, solved rows also write ``fsq`` history plus
iota and Mercier ``DWell`` profile figures. Pass ``--run-vmec2000`` to run the
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
VMEC/JAX terminal step histories and write the step-diagnostics plot. The
residual-history plot then aligns VMEC/JAX and VMEC2000 by actual iteration
labels. ``initial_fsq_ratio_direct_initial`` records how far the first stored
VMEC/JAX history row is from the pre-iteration direct residual. Treat mean-iota
and direct-initial residual agreement as useful regression signals, and use the
history fields to understand convergence after solver startup. For
toroidal-hybrid geometry refinement, pass ``--shape-cases default,sharp`` to
scan the default boundary and a sharpened side/corner preset in one run; use
``5:10`` or higher ``mpol:ntor`` pairs when asserting exact fit for the
sharpened preset.

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
``--lcfs-require-bnormal-nonincrease`` to enable a stricter guard: candidates
that increase exact coil-resampled ``B_ext.n`` RMS are filtered out, the mixed
scale/normal-field candidate is selected when it improves merit while satisfying
the guard, and otherwise the explicit no-op candidate records a skipped pilot
row. Metrics JSON rows include ``lcfs_update_allowed_strategies`` and
``lcfs_update_rejection_reason`` fields so downstream scripts can distinguish
accepted, rejected, and guard-limited pilot steps. This is still an LCFS pilot
workflow, not a converged free-boundary equilibrium solve.

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
