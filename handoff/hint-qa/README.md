# HINT–VMEX finite-beta QA comparison

Reviewed 2026-09-24 against VMEX main
[`926892ab7131a6bc0c5b61218d1f75e7b77bc401`](https://github.com/uwplasma/vmex/commit/926892ab7131a6bc0c5b61218d1f75e7b77bc401)
(release 0.11.2). The measured VMEX runs below remain pinned to their earlier
source; advancing main does not silently requalify them. The source and PR
status table below records the September 21 audit and is historical where
later status differs.
This is a research protocol and historical baseline, **not a qualified
HINT–VMEX equilibrium comparison**. The two supplied finite-beta QA cases
remain the goal: nominal volume beta 0.5% and 2.5%, first comparing fields
outside the given boundary, then a separately matched free-boundary study.

The HINT creator requested the maintained
[HINT3D `current` branch](https://github.com/yasuhiro-suzuki/HINT3D/tree/current),
which still resolves to
[`bf31fc39f7179bdd91d84319c51c68e2f6fff25f`](https://github.com/yasuhiro-suzuki/HINT3D/commit/bf31fc39f7179bdd91d84319c51c68e2f6fff25f)
at this review. `master` is not the study source. The supplied 30-page LHD
manual describes MKVAC, MKFLX, MKLIM, HINT, GPRTS and HMAG and provides an
LHD example; its commands and LHD parameters are illustrative, not instructions
to substitute that case for the finite-beta QA inputs here. Geiger's 2022
W7-X poster reports broad HINT versus VMEC/EXTENDER agreement but differences
around islands and at the edge, including underestimated island size in that
VMEX approach. Its W7-X configurations and historical EXTENDER implementation
are not the present QA case or current VMEX code. These author materials set
the questions and failure modes; they do not provide a quantitative QA answer.

Since the September 21 audit, dependency floors [#410](https://github.com/uwplasma/vmex/pull/410),
free-boundary recovery [#416](https://github.com/uwplasma/vmex/pull/416),
the research plan [#413](https://github.com/uwplasma/vmex/pull/413),
the independent exterior-field oracles [#430](https://github.com/uwplasma/vmex/pull/430),
and target-graded near-surface quadrature [#441](https://github.com/uwplasma/vmex/pull/441)
merged to main. The old #417/#418/#421/#423 stack closed; #422 merged only
into its feature branch. Dependent [#306](https://github.com/uwplasma/vmex/pull/306)
remains a draft based on this PR. Reassess its derivative guards against the
current main implementation before promoting that stack. #430/#441 materially
improve the VMEX prescribed-boundary exterior-field oracle but do not validate
the HINT run or the inter-code comparison.

## Sources and scope

| Source | Required interpretation |
|---|---|
| [HINT3D `current`](https://github.com/yasuhiro-suzuki/HINT3D/tree/current) | The creator maintains this branch; do not substitute `master`. The measured source is `bf31fc39f7179bdd91d84319c51c68e2f6fff25f` plus [the eight-file patch](hint-portability.patch). The patch is not upstream or proof of a bug-free solver. |
| [VMEX PR #302](https://github.com/uwplasma/vmex/pull/302) | Branch `fix/hint-comparison-derivative-contract` is reconciled with the pinned current-main baseline. Its old numerical source `14360d179af4534aa9bb638b172c8724ee289d9b` is historical evidence, not certification of this branch. |
| [SOLVAX #104](https://github.com/uwplasma/SOLVAX/pull/104) | Tested head `e4b507185464851ac5e8de22041f2f8e384554e9`, merged as `66f97a6e0eb758af8d7f46939ffdd8ec733efbef`; released in 0.21.0. Use at least 0.21.0 for its rejection of nonfinite nonlinear roots. |
| [VMEX #409](https://github.com/uwplasma/vmex/pull/409) | Merged as `45f3a7aea`: fixes the surface-field assembly regression introduced by #403. Its complete Git tree equals the tested candidate `aabedb8f`; execution evidence below retains its original commit identity. |
| [VMEX #410](https://github.com/uwplasma/vmex/pull/410) | Open draft at review: owns dependency-floor changes. Complete exact-floor validation there rather than duplicate it here. |
| [VMEX #416](https://github.com/uwplasma/vmex/pull/416) | Open candidate: removes history-dependent suppression of free-boundary cold recovery. Qualify repeated accepted points after rejected trials on the integrated source. |
| [VMEX #417](https://github.com/uwplasma/vmex/pull/417) | Open candidate: anchors derivative admission and reuse to measured refined coefficients. Exact-case QA and GPU evidence remain necessary. |
| [VMEX #418](https://github.com/uwplasma/vmex/pull/418) | Draft follow-up to #417: applies unconditional validity checks to direct derivative paths even with no absolute primal cutoff; restacked on #417 head `1459f9df`, with 18 focused module tests passing at #418 head `9e0baa28`. Integrated QA/GPU qualification remains open. |
| [virtual_casing_jax #14](https://github.com/uwplasma/virtual_casing_jax/pull/14) | Open focused fix: prevents outer-JIT source construction from caching tracers while preserving source-field gradients. Seventeen derivative/lifecycle tests and two leak-check tests pass on CPU; no GPU or whole-workflow performance claim. |
| [ESSOS #71](https://github.com/uwplasma/ESSOS/pull/71) | Draft for manual review: an analytic circular-loop oracle checks complete on-axis Cartesian tensors through order three and two selected field-value current/radius JVPs. All six field tests pass; this finds no defect in that scope and does not qualify off-axis fields or all parameter derivatives. |
| [VMEX #421](https://github.com/uwplasma/vmex/pull/421) | Draft at `95aed755`, stacked on #418: callback refinement and certification use the runtime device and exact returned coefficients. Integrated CPU modules pass 52 tests, five focused two-device tests pass, and a real small CPU callback/certificate succeeds. The GPU-facing smoke used a CPU root; accelerator-resident root qualification remains open. |
| [VMEX #419](https://github.com/uwplasma/vmex/pull/419) | Draft guarded block-factor reuse benchmark: exact-current residual checks and fallback are demonstrated on its QI case; a full-optimizer accuracy, memory and runtime comparison remains required. This is distinct from the rejected scalar-factor cache. |
| [VMEX #413](https://github.com/uwplasma/vmex/pull/413) | Proposed replacement product plan. Reconcile after integration; its six research lanes do not replace this study's physical comparison gates. |

[#422](https://github.com/uwplasma/vmex/pull/422) is merged into #421 as
`e94e46c6`, retaining only complementary placement tests. Independent review
found no blocker; all 11 cases passed on two forced CPU devices, including
an explicitly JIT-enabled run. The default pytest configuration disables JIT,
so that older command alone is not JIT evidence. No duplicate production
implementation remains. The consolidated #421 still needs its integration
checks/review and GPU-root qualification before main promotion. Review in
dependency order: #416, #417, #418, then #421/#423. The #423 follow-up
`02bf33d9` already admits boundary-transform roundoff and aligns public
measurement inputs; do not duplicate the earlier local correction. Its reported
71-test CPU suite does not qualify an accelerator root or full optimizer.
Forward field diagnostics can
continue on the pinned main independently of derivative-stack promotion.

Current main has raw block Newton/adjoint solves, deterministic free-boundary
reference selection (#383), structured factor reuse (#395), a retained-array
leak fix (#397), and native interior-field reconstruction (#403). The latter
postdates the 0.10.0 release commit. Old universal `primal_tol=1e-10` admission,
cache patches and refinement-budget overrides are not replayed over these
new implementations. A near-null lambda mode can make a universal raw norm
threshold inappropriate; goal-oriented residual estimates are still proposals,
not implemented gradient certificates. Audit current cache/root coherence
independently. Review also rejected the old radial-lift guard: one sample
strictly inside every elementary knot span is not a necessary spline-rank
condition. Future admission must inspect the effective constrained design
matrix after mode-dependent row filtering, with a valid graded-grid regression
as well as a deficient-fit regression. Conditioning and physical force
convergence remain separate questions. No solver changes from the older
branch are retained in this reconciliation.

The old performance PR #299 was split into #313–318. DESC conversion #300 is
merged. Phase G of [the main plan](../../plan.md) concerns a broader external
benchmark; it does not close the HINT gates below. Live PR state supersedes
these dated dispositions. Do not merge overlapping historical PRs wholesale.

## Immutable physical inputs

The legacy public deck names say LandremanPaul, but these are the finite-beta
Landreman–Buller–Drevlak family. They share an outer boundary and differ in
interior equilibrium, current and associated coils; **this is not a beta-only
scan**. Preserve the known 2.5% coil/WOUT mismatch. Do not silently refit coils,
change enclosed flux, or pressure-load the original vacuum shape.

The exact WOUTs are retained as [0.5%](data/beta0p5.nc) and
[2.5%](data/beta2p5.nc), with [SHA-256 hashes](data/SHA256.json).
Their original internal input labels are public VMEX example labels.
Decks and coil JSONs already live in
[VMEX example data at the pinned baseline](https://github.com/uwplasma/vmex/tree/f719c4ff503c526368254aebf3ba9a9119c17f87/examples/data):
`input.LandremanPaul2021_QA_beta0p5_bootstrap`,
`input.LandremanPaul2021_QA_beta2p5_bootstrap`, and the corresponding
`ESSOS_biot_savart_LandremanPaulQA_beta*_bootstrap.json` files.
Their hashes are included in the manifest and their bytes match the measured
copies. Do not duplicate or regenerate them without recording changed hashes.
The 192 fixed Cartesian [targets](data/targets.npz) are also preserved.

| Quantity | 0.5% case | 2.5% case |
|---|---:|---:|
| Source WOUT volume beta (%) | 0.5038845418 | 2.5469063197 |
| Source WOUT toroidal current (A) | -95778.349865 | -272398.496522 |

At 0.5%, input `CURTOR=-95673.867957 A` differs from the WOUT edge-extrapolated
current by 104.481908 A; retain this finite-mesh distinction. HINT pressure is
in magnetic units, `p_HINT = mu0 * p_Pa`. The candidate current table is
proportional to `<J.B>/<B^2>`, normalized on HINT's selected toroidal cut;
it is not `dI/ds`. Record the actual imposed parallel current separately from
the curl-derived total current. Their difference includes physical
perpendicular current and is not a zero-target residual.

## What has actually been achieved

These are source-specific diagnostic measurements. No numerical result below
certifies current VMEX main, HINT equilibrium convergence or island widths.

| Lane | Evidence | Remaining gate |
|---|---|---|
| Native HINT | HINT and six native tools built with debug/release configurations; small regressions exposed pressure interpolation, initialization, periodic bounds, callback and viscosity-stencil defects. | Independent review of the patch; cross-toolchain rerun and reference convergence. |
| LHD reference | Coarse continuation to normalized time 100: last pressure change 0.5773%, peak pressure 26.5165 kPa, pressure integral 279584.83 J. | Time/grid/trace convergence, quantitative reference geometry, manual's 256-grid reproduction. |
| Historical VMEX 0.5% root | Two cold CPU states identical; projected residual 4.4748e-11 at ns31/MPOL7/NTOR6. CPU/GPU-environment state difference 4.179e-12. | Rerun on current source and dependencies; GPU device placement was not a kernel profile. |
| Historical VMEX derivative | Live energy in one boundary direction: JVP/VJP duality 2.723e-12; Taylor orders 1.979, 1.963, 1.836. | Other directions/objectives, current-source CPU/GPU, external oracle and free-boundary response. Small-step FD worsened from 4.22e-7 to 1.09e-5. |
| Synthetic calibration | One accepted scalar step reduced energy error from 1.21e-4 to 6.31e-8. | Timed out before 1e-9 target and independent cold final solve; no successful optimization claim. |
| HINT later-time 0.5% | At time 1.08, pressure-supported current -17.692 kA versus -95.778 kA target; combined force ratio 0.014006. | Large current deficit, return-current support and physical relaxation remain unresolved. |
| HINT resistivity test | Matched restarts at eta .002/.005 gave -19.889/-25.521 kA, but force ratios worsened to .015232/.019714. | Do not promote higher eta because current alone improves. |
| HINT fresh grid pair | 64x64x32 and 128x128x64 completed at time .01 with identical dt, eta and cadence. Currents only -310.316/-257.749 A; opposing-current shell follows grid scale. | Startup diagnostic, not mature equilibrium or grid convergence. |

The fresh pair used four outer cycles of 250 steps, `dt=1e-5`, `eta=.001`,
four MPI ranks and one OpenMP thread per rank. Native HINT rejects
grid-changing restarts; do not relabel a remapped state as a native restart.
The later-time and fresh-start trajectories have different histories and
cannot be combined into one convergence sequence.

A current-source CPU screen of the exact 0.5% deck subsequently completed
700 iterations with finite state and reported `fsqr=9.93997e-12`,
`fsqz=1.35863e-12`, `fsql=3.95010e-12`. This verifies root execution on
`f719c4ff`, not a fresh physical-force certificate or an exterior/VJP result.
That historical main lacked #409; current main includes the fix. In a fresh environment,
the exact 0.5% outside-field test subsequently passed at reviewed #409 head
`aabedb8f2212747ed2651fb13ba81468cc5f7039`: **1 passed, 1 deselected in
132.23 s**, CPU-only, within a 240 s process-group cap. It actually exercised
the live-state surface assembly, exterior field, spatial derivatives through
third order and all four VJP outputs, with `accuracy_check='raise'`. Assertions
check finite/nonzero B and |B| and finite VJP maxima with at least one nonzero
maximum; they do not inspect every derivative or VJP component. This is not an import skip, FD/duality comparison,
GPU result, or independent physical-force certificate. The focused live-state
surface regression also passed separately. The candidate subsequently merged as `45f3a7aea`, with the identical complete
Git tree `d070d6401240f993c72612103d7bd6b4acd03226`. No new numerical run is
implied by that tree-identity check.

The isolated test environment passed `pip check` and source-placement checks:
Python 3.11.14, JAX/jaxlib 0.9.2, SOLVAX 0.21.0, booz_xform_jax 0.4.0,
virtual-casing-jax 0.0.7, ESSOS 0.17, NumPy 2.4.6, SciPy 1.17.1,
netCDF4 1.7.4, h5py 3.16.0 and pytest 9.1.1. This lists the measured core
versions rather than claiming a complete transitive environment lock.
An isolated matching-core GPU environment is prepared with the JAX 0.9.2
CUDA-12 runtime packages; dependency, source-placement and CPU callback checks
pass. Both devices were occupied at the readiness check, so no GPU run was
started. A future run must select CUDA first while retaining the CPU backend
for host callbacks, disable preallocation, enforce time/memory bounds, and
retain numerical outputs for comparison rather than report a test pass alone.

A captured-output CPU run on the same #409 candidate completed with all 15
arrays finite, nonzero field, and all four VJP vectors nonzero. The portable
runner subsequently reproduced every array byte-for-byte in a separate process
(184.2 s wall time); container ZIP hashes differ, numerical member hashes agree. The [record](vmex-cpu.json) and [numeric archive](data/vmex-cpu.npz)
retain the single exterior target, spatial derivatives through order three,
four 144-entry VJPs, all-ones cotangents and exact parameter ordering. This is
candidate-source execution evidence, not a finite-difference certificate.
The saved `B_error_estimate` is dimensionless, order-zero virtual-casing plasma
quadrature error normalized by RMS total field on the source surface. It does
not estimate combined coil-plus-plasma field error or derivative accuracy.

A subsequent bounded physical screen at the same tested source exposed a
qualification gap. The public strong-force evaluator requires a continuous
spline lift of the native state. On the same 1,280 shifted points with
`s` in `[0.1, 0.99]`, degree-five lifts with 13 and 8 spans gave dimensional
volume-weighted force L2 norms of `2.390844e6` and `3.225829e8 N/m^3`.
Their magnetic fields differed by relative L2 `0.00731069`, while their force
arrays differed by `150.8595`. This two-fit sensitivity is not a rigorous error
bound and cannot accept or reject the native equilibrium. Roundoff divergence
of either fitted flux-coordinate field is a representation identity, not an
independent native-state certificate. The next qualification step must control
reconstruction error before interpreting this diagnostic. The compact
[CPU record](vmex-cpu.json) retains the screen, normalization and limitations;
a portable reproduction command for this supplemental screen remains open.

Pressure agrees with the prescribed profile on the correct native half mesh
to `1.46e-11 Pa`; the negative full-mesh WOUT edge extrapolate is not the
prescribed boundary pressure. Flux matches the target exactly. The current
reconstruction differs from input CURTOR by `104.481908 A` (`0.109206%`), the
finite-mesh distinction already recorded above.

A subsequent native-form screen avoids the two arbitrary high-order lifts.
It uses the public interior `B` and `gradB`, the flux-coordinate geometry
Jacobian and the prescribed pressure profile on 1,280 shifted points with
`s` in `[0.1, 0.9]`. It still uses the finite-NS C2 radial interpolant and
seeded Cartesian inversion. Measured force L2 is `9.290990e6 N/m^3`, mean
`|F|/mean|grad p|` is `2.263727`, and the pointwise normalized L2 is `1.040597`.
These values do not support force-balance acceptance. The sampling window
also differs from the two-lift screen, so the norms are not a controlled
representation comparison.

On a common 32-point subset, fourth-order Cartesian stencils at
`h/a = 1e-3, 5e-4, 2.5e-4` approach the analytic force with relative L2 errors
`3.91e-5, 6.65e-8, 4.16e-9`. This checks the derivative and coordinate
implementation for this interpolant; it does not establish physical force
convergence. Analytic divergence at roundoff is likewise a representation
identity. The [record](vmex-cpu.json) and [numeric arrays](data/native-force.npz)
retain the points, weights, fields, force and stencil results. Their force
algebra and weighted norm were independently recomputed. The run completed
in 71.1 s on one CPU core. The portable [force diagnostic](diagnose-force.py)
now reproduces all 17 arrays and the compressed archive bitwise at default
NS31/angular16 and at NS121/angular32. Both runs passed exact-state guards;
existing numeric JSON fields are unchanged apart from timing and memory.
The runner adds per-radius metrics and configurable measurement angular counts,
with per-surface evaluation to bound batch memory. The largest of
the five sampled radial-band force norms occurs at `s=0.137528`
(`2.6527344e7 N/m^3`); retain these per-radius metrics under refinement
rather than relying on the aggregate norm alone.

A fixed-state angular check at 16x16 and 32x32 preserved the exact native-state
hash. Global force L2 changed by -0.0875%, and mean force by -0.5044%, but the
outer sampled radial band changed by +9.87% in force L2. The dominant inner
band remained about `2.65e7 N/m^3`. Thus the aggregate is comparatively stable,
while per-radius quadrature convergence remains uneven. The compact record
retains both grids and independently checked weighted reductions; its raw
sample archive is retained separately and is not publicly downloadable.
The follow-up at 48x48 reproduced the shared 32-grid arrays exactly. Global
force L2 changed by only +0.0000689%, and the outer-band change contracted to
+0.0155%; all other band L2 changes were below 0.00061%. Per-band mean-force
and ratio changes were below 0.391%. The 32-grid measurement is adequate for
this fixed-state force-L2 screen. This justified the controlled radial
resolution checks below at unchanged physical and angular settings.
This refines the diagnostic, not the equilibrium or its physical accuracy.

Changing only `NS_ARRAY=31` to `61` converged in 891 iterations with all three
FSQ channels below the unchanged `1e-11` tolerance. At the same flux/angle
coordinates and 32x32 measurement, force L2 fell from `9.282864e6` to
`1.094951e6 N/m^3` (88.2%); mean force over mean pressure-gradient magnitude
fell from `2.252405` to `0.450795`. The inner-band force L2 fell 92.1%, while
the middle band changed only 0.63%. This is substantial resolution dependence
of the native solve/reconstruction, not a force-balance certificate. Flux
remained exact; the output/input current difference decreased to 20.9194 A.
The two states map common flux coordinates to slightly different Cartesian
points. The NS121 rung also converged (1,800 iterations) with all other
settings retained. Its force L2 was `8.270284e5 N/m^3`, another 24.47% reduction,
and mean force/mean pressure gradient was `0.361272`. Per-band force changes
from NS61 ranged from -83.4% to +10.1%, so radial convergence is not established.
The input/output current difference decreased to 4.57070 A, with edge flux
preserved to roundoff. A separate fixed-state NS121 measurement check now
finds 32x32 to 48x48 changes of 0.0104% in global force L2 and less than
0.028% in each radial-band L2. Mean-force and pressure-ratio changes remain
below 0.79% per band; the global pointwise-normalized metric changes 1.06%,
so that metric has a weaker qualification. Shared 32-grid B/curl/coordinates
match exactly; weight, pressure-gradient and force differences are only
operation-order roundoff. Force-L2 measurement error does not explain the
24.47% radial change. Separate solver angular resolution and Fourier
truncation effects before extending the radial ladder.

At NS121, changing only solver `NTHETA`/`NZETA` from 20 to 40 changes
force L2 from `827028.445` to `811135.553 N/m^3` (-1.92%); per-band changes
range from -1.03% to -4.05%. All native FSQ channels remain below `1e-11`.
This is distinct from measurement quadrature: Fourier truncation is unchanged,
and common flux coordinates map to slightly different Cartesian positions.
Two solver grids do not establish angular convergence. The 32-point diagnostic
qualification applies to the grid-20 state; qualify it again on grid 40 before
interpreting smaller effects. Next isolate Fourier truncation and radial
reconstruction. The compact record retains hashes and metrics; supplemental
raw archives still need public hosting.

A subsequent one-control poloidal Fourier check at NS121 and solver angular40
changes only `MPOL=7` to `8`. The solve reports convergence in 1,357 iterations.
Force L2 falls from `811135.553` to `424603.079 N/m^3` (-47.65%); all five
radial bands fall by 32.1–51.9%. Mean force/mean pressure gradient is still
`0.187681`, so this is sensitivity evidence, not force-balance acceptance.
The common flux nodes move by up to 4.54 mm in Cartesian space; the unweighted
B difference is 0.0882%. This makes Fourier truncation a material confounder
of radial attribution. Qualify measurement on the new state and separate the
next poloidal/toroidal truncation check before extending the radial ladder.
The runner retained convergence and iteration count, but not final FSQ channels
or output current/flux/pressure scalars; those cannot be claimed from this run. Future
runs now record these fields explicitly; the metadata extension passed a small
Solovev full-pipeline check. `CURTOR` is an input constraint only in the
current-constrained mode (`NCURR=1`); full-mesh WOUT edge pressure is an
extrapolated output, distinct from the prescribed boundary pressure.

To reproduce the force screen, set `VMEX_SOURCE` to a clean checkout of the
pinned main revision, create `results`, and run from the HINT handoff branch:

```sh
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python handoff/hint-qa/diagnose-force.py \
  --source-root "$VMEX_SOURCE" \
  --input examples/data/input.LandremanPaul2021_QA_beta0p5_bootstrap \
  --expected-head 45f3a7aeaeedd50ee7a04d553bc0d281bb8510a4 \
  --expected-tree d070d6401240f993c72612103d7bd6b4acd03226 \
  --output results/native-force.json --samples-output results/native-force.npz
```

The command requires `GAMMA=0`, enables x64, checks source placement and refuses
existing outputs. `--angular-count 32` changes only measurement sampling;
change `NS_ARRAY`, `NTHETA`/`NZETA`, or Fourier resolution in a separately
identified input to change the solve. `--expected-state` accepts a recorded
SHA-256 and fails before field measurement if the solved coefficients differ.
Apply external time/memory limits on shared machines.

The portable [capture command](capture-vmex.py) takes explicit paths and records
source/dependency/input identities. Use the exact tested source revision above
for reproduction, or record a new baseline for an integrated revision. With
that checkout selected as `VMEX_SOURCE` and an existing output directory:

```sh
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
  VMEX_COMPILATION_CACHE=disabled OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python handoff/hint-qa/capture-vmex.py --source-root "$VMEX_SOURCE" \
  --output-prefix results/vmex-cpu
```

Use a process-group time/memory limit on shared machines. A paired GPU run must
retain the CPU backend for callbacks and record actual placement; neither a
GPU environment nor a CPU pass establishes GPU parity. All backend tolerances
are separate from the physical accuracy and finite-difference gates.

The first actual CUDA capture used the original public driver and reached a
GPU process, but exited at the first target materialization: callback
refinement combined a GPU state with CPU parameters. No numerical archive
was written. This is a software placement failure, not failed physics or a
CPU/GPU parity result. The capture command now accepts `--device gpu` for
VMEX's supported explicit placement; omitting it preserves the measured CPU
workflow. The explicit-device retry reached problem construction in 25.8 s,
then hit its 660-second guard (661.647 s elapsed), with peak GPU process
allocation 486 MiB and host RSS 1,791,552 KiB. Both reserved output files
remained empty: zero of 15 arrays were captured. Cleanup completed. This
establishes a bounded timeout, not a deadlock or CPU/GPU parity. The automatic
callback placement correction is proposed in #421; its forced-two-CPU
regression fails on the parent and passes after correction. Qualify a small
real numerical callback before another full capture. This patch has not
been shown to resolve the explicit-device timeout.

Source review of HINT's linear drive finds no normalization defect explaining
the observed deficit: its cut-zero imposed-current integral is constructed
to match `inet0`. That source enters the resistive evolution; it does not
directly assign the attained curl-derived current. The existing history
instead reports attained current inside positive pressure, averaged over
toroidal cuts. Comparing it with a cut-zero target supported on `ss<jcuts`
mixes both masks and cuts. Further, `ss` is frozen during each magnetic
substep sequence while the field and current normalization evolve.
The optional [current-drive diagnostic patch](hint-diagnostics.patch) reports
source and attained currents at a native history update on the same
`ss` support for cut zero and min/mean/max across cuts, alongside the original
pressure-mask current, step-start traced axis pressure and a source
cancellation ratio. This can distinguish source normalization, cut variation and physical
relaxation without large diagnostic arrays or changes to the dynamics.
Serial/two-rank agreement, default/off behavior and a zero-source control
have been checked; sign-changing cancellation and broader decomposition
fixtures remain. The linear branch also ignores `inet1`; the present QA deck
uses zero, so that conditional limitation does not explain this run.

On a short 64x64x32 diagnostic fixture (two outer cycles, two magnetic steps,
`dt_b=1e-4`, `eta0=1e-3`, `lc_in=0.2`, energy output every step),
the first source cut carried **-95778.349865 A**, matching its target to
summation precision; attained current was **-5.389768 A on the same support**
and **-4.592082 A inside positive pressure**. Source cut min/mean/max were
-96700.575679/-95709.272272/-94625.010911 A. This directly demonstrates cut
variation and distinct attained-current masks, not convergence of the later
trajectory. Enabling the diagnostic left the one-rank history and native
field file byte-identical to the disabled run. Two-rank diagnostic values
matched apart from timing; history differences were at most 1e-22. A final
zero-drive guard check emitted four finite, zero-source records without
reading uninitialized normalization volumes. The historical accounting-only source hash is
`bc4913186bd6de2a0118ba9f1e65a252160420cc47d62733abd0fc398c2e8fc2`.

A subsequent [later-time diagnostic](current-drive.json) restarted the retained
state at code time 1.08 and completed one outer block of 100 magnetic updates
at `dt_b=1e-4`, reaching 1.09. The shortened block is diagnostic-only; it does
not preserve the production cadence beyond those updates. Four MPI ranks and
one OpenMP thread completed in 51.2 s. The imposed source cut-zero current was
-95.778350 kA and its cut mean -95.672992 kA, with no sign cancellation.
Attained current on the same `ss<1` support averaged **-19.062808 kA (19.903%
of target)**; the positive-pressure comparator averaged -17.783723 kA and
matched native history. Thus differing masks alone do not explain the deficit.
Source normalization is working at this measured state; the dynamical cause
remains unresolved. Curl-derived response current also includes perpendicular current,
so equality to the imposed parallel-current source is not itself a valid
zero-residual condition. Raw restart assets remain necessary for independent
reproduction.

The optional patch now also decomposes the instantaneous current-rate RHS
using the production Faraday and boundary stencils, then the production
fourth-order toroidal curl, on a frozen source mask. Repeating the same
100-update sample with the extended diagnostic gives these cut-mean rates:

| Contribution | Current rate (kA per code-time) |
|---|---:|
| Ideal | +2.765848 |
| Resistive response | +23.121079 |
| Imposed drive | -35.437527 |
| Divergence cleaning | approximately zero |
| Total | -9.550599 |

The term sum agrees with a separate total-RHS evaluation within
`6.73e-11 A/code-time` (relative `5.75e-15`). The imposed term is active in
the target direction; response and ideal terms oppose it. This supports slow
relaxation/current redistribution at this state, without determining a
long-time rate or physical-second timescale. The original-length follow-up below tests this rate over a bounded interval
with unchanged physical settings and magnetic cadence.

The subsequent original-length block also completed: 1,000 updates from the
same retained time-1.08 restart to 1.18, preserving `dt_b`, resistivity,
`nenergyb=100`, `nrefb=100` and smoothing cadence. Only the one-outer-block cap
and diagnostic flag changed. Ten complete samples took 60.5 s on four MPI
ranks. The imposed mean rate remained about `-35.44 kA/code-time`; the net
mean rate remained negative (`-9.55` to `-9.72 kA/code-time`). Pressure-supported
current moved monotonically from `-17.692417` to `-18.647239 kA`; same-source-
mask attained current reached `-19.933130 kA` at the final sample.

Native history force ratio improved overall from `0.0140061` to `0.0128942`,
but reached its minimum `0.0128270` at time 1.16 and then increased. This is
nonmonotonic late behavior, not steady convergence. The native absolute
force record is a mean **squared normalized code residual**, not an SI force
density; [the record](current-drive.json) defines both force normalizations
and retains the full ten-sample means and provenance. Term closure remained
below `6.63e-15` relative. No physical-time extrapolation or equilibrium
acceptance follows from this one block. Its endpoint has not been reused.

A drive-off control from the identical time-1.08 restart changed only `inet0`
to zero (`inet1` was already zero). The first attempt failed before evolution
because field inputs were missing. The corrected attempt completed 1,000
updates and ten finite logged samples, then failed writing a read-only restart
copy. Its checkpoint remains at time 1.08: there is no saved endpoint or
independent endpoint-field audit. The [record](current-drive.json) preserves
both failures and explicitly labels the series partial observational evidence.
Within that in-memory block, imposed current/rate terms are exactly zero,
term closure is below `2e-15` relative, and pressure-mask current changes from
`-17.6924` to `-16.0019 kA`; the driven control ends at `-18.6472 kA`.
Both force normalizations decrease monotonically, with final relative residual
`0.00936965` versus the driven `0.0128942`. This supports investigating the
competition between imposed drive and the opposing response; it does not
justify changing the physical target or certify a steady equilibrium. No
third run was made. A future controlled endpoint requires writable-copy
preflight and successful checkpoint output before any exterior-field audit.

The tested extended source SHA-256 is
`21aa3e90830b8be8166cfca21f845f29cd8b7638a639b0ddc6123b46cac0e488`.
On a two-update mature-restart fixture, the previous binary with diagnostics
off, the new binary off, and the new binary on produce byte-identical field
and history files. One-rank and four-rank diagnostic values also agree
byte-for-byte apart from timing. The repeated 100-update output/history match
the prior accounting-only run exactly. The compact [record](current-drive.json)
keeps both source versions and validation hashes. These checks used the
standard Ohm-law build; the alternative `BDIFF2ND` build is not numerically
qualified here.

New native postprocessing at the same 192 targets separates the cross-grid
field differences: total RMS/max **0.243516/1.701148 mT**, vacuum
**0.232940/1.697470 mT**, plasma response **0.074009/0.321955 mT**.
Response means `native(total) - native(vacuum)` on each grid, followed by a
fine-minus-coarse vector difference. Norms do not add linearly. This removes
a major ambiguity in the earlier total-field comparison, but two startup
grids do not establish a response convergence order. A point exterior to
WOUT is not necessarily exterior to the relaxed current support.
The [compact field record](field-summary.json) retains exact metrics and
postprocessor, target, native **sampler** binary and field hashes. The sampler
hash is not the HINT solver executable hash. The compact
[sample archive](data/native-samples.npz) contains all 192 target vectors for
both total and vacuum fields on both grids. Independently recomputing the six
metrics and sampled-array hashes reproduces the record exactly. The public
[native driver](sample-points.f90) and [postprocessor](sample_fields.py) also
reproduce the original sampled-array hashes when run with the retained raw
inputs and native objects. A separate clean macOS build passed a constant-field check and now samples
all 192 real-data targets within 6.218e-15 T of the retained Linux results.
This is cross-toolchain roundoff agreement, not bitwise equality; exact
metrics and binary provenance are in the field record. The
portable [coarse](data/coarse.in) and [fine](data/fine.in) native decks retain
all measured numerical namelists, including the current-profile table; only
their header comments and limiter/history filenames changed. Original deck
and prepared vacuum/flux/limiter hashes are recorded separately. Those large
prepared fields and the wall-generation inputs are not bundled, so the
source decks/coils and WOUTs alone cannot recreate the measured fresh pair.
A 56,131,731-byte archive of both original snapshots, the six prepared
fields, portable decks and a checksum manifest is now assembled and inspected
for private metadata. It remains outside Git history and is not yet publicly
downloadable. Publish it through a suitable research-data channel, and complete
preprocessing and independent equilibrium reproduction before promotion.
VMEX's release-published workflow is for PyPI packages; a diagnostic-data
archive must not be published as an accidental package release.

The archive contains only the nominal 0.5% case at two grids, not both beta
cases. Its member checksums verify bytes but do not yet capture the complete
preparation provenance. The preparation runner must connect each output to
its source hashes, generator revision, exact command, package/compiler/MPI
versions and native executable hash. Record numerical array hashes separately
from NetCDF container hashes, which can change across library versions.

The minimal reproducible preparation sequence is: validate the named WOUT,
input deck and coil JSON; derive pressure/current metadata and the profile
using documented quadrature; verify the two boundaries are identical before
sharing a wall; generate the numerical poloidal-buffer wall and direct coil
grid; run native MKFLX and MKLIM for flux and limiter fields; then render the
HINT deck and verify matching coordinates, field-period count, finite fields,
limiter values, containment and current-cut integrals. Read field-period count
from the source instead of assuming two. Native flux/limiter preprocessing
should remain native; a Python reconstruction is an independent check.
Native `build_config.json` records the requested commit and options, but older
builds do not record the applied source patch, and incremental make can reuse
objects after compiler/flag changes. Binary hashes identify the measured
executables; the config alone does not prove source/flags provenance. A clean
rebuild with build-time source fingerprints is required for each promoted executable.
The updated [build helper](build.py) now forces recompilation and writes a
schema-2 manifest for only the selected tools, guarding source changes and
recording source/helper/compiler-wrapper/binary hashes. It invalidates stale
metadata before rebuilding and rejects output directories inside the source
tree. MKFLX/MKLIM were rebuilt twice in one isolated directory; both produced
identical binaries and the regenerated preparation arrays/decks stayed exact.
The preparation runner checks selected-tool coverage, source-set checksums and
actual binary hashes. This qualifies those preprocessors' recorded source sets. A subsequent full
HINT Release build fingerprints 30 source inputs and passes the four-update
serial zero-drive fixture in 3.07 s. Magnetic arrays match the earlier fixture
exactly; maximum pressure/velocity differences are 4.65e-16/1.90e-19 in native
stored units. The build and smoke hashes are retained in `prepare.json`.
External library/toolchain closure and production-equilibrium validation remain
separate.
Wall generation must pin Shapely/GEOS and the buffer/resampling settings.
Use explicit inputs, refuse conflicting outputs, and preserve the measured
current-table/deck contract. Snapshot extraction and deterministic data
packaging follow simulation as separate steps.

The [preparation runner](prepare.py) now performs this sequence with native
MKFLX/MKLIM and public manifest-verified inputs. It enables x64 after imports,
checks shared boundary coefficients, and refuses an existing output directory.
The [preparation record](prepare.json) retains exact environment requirements,
source/driver/binary hashes and numerical comparisons. In a VMEX environment,
install the additional preparation packages and select the separately built
native tools with `HINT_ROOT`:

```sh
python -m pip install essos==0.17 f90nml==1.4.4 shapely==2.0.7
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python handoff/hint-qa/prepare.py --inputs handoff/hint-qa/data \
  --output results/prepared --native-bin "$HINT_ROOT/build-release" \
  --source-commit bf31fc39f7179bdd91d84319c51c68e2f6fff25f --device cpu
```

The default prepares both cases at 64x64x32; `--grid 128,128,64` selects the
published fine protocol. `--prepare-only` still computes wall, vacuum and
current inputs but skips executing native flux/limiter tools. It does not run
HINT. The pinned CPU/x64 coarse run completed both preprocessors for both
cases. Limiter arrays equal the retained fields exactly; maximum vacuum and
flux differences are below `9.401e-12 T` and `3.376e-14`, respectively.
The 0.5% deck remains byte-identical to `coarse.in`, and both current tables
match the historical published tables. The final metadata-capable runner was rerun through both native tools; all
NetCDF variables and both decks match the pinned run exactly. Fine-grid numerical
preparation and complete equilibrium reproduction remain unqualified.

Before a restart control, inventory every read-side file named by the deck
and verify its reference hash, readability and dimensions. HINT follow mode
reads and later updates `hint.nc`; use a regular private copy with owner write
permission and verify that permission before MPI launch. Keep the retained
checkpoint immutable. Field-input existence alone is insufficient: a copied
read-only restart can evolve successfully and fail only at final NetCDF output.

The small [restart preflight](prepare-restart.py) stages this file set without
launching HINT. Its checksum manifest has a `files` object containing exactly
the deck basename and the four `FOPEN` basenames; each value contains `sha256`
and may also contain `bytes`. The output directory must not already exist:

```sh
python handoff/hint-qa/prepare-restart.py --deck control.input \
  --source-dir retained-assets --sha256-manifest restart-sha256.json \
  --output staged-control
```

The utility supports the qualified `run_mode='follow'`, `flx_type='file'`
path. It verifies and stages every named field, creates the restart as a fresh
owner-readable/writable regular file, confirms a no-write `r+b` open, and
rechecks that retained source hashes and modes did not change. The generated
`restart-preflight.json` records basenames, checksums and staging checks.
Successful staging does not establish that HINT can write its final checkpoint.

A subsequent identical bounded control passed this preflight and completed
with exit status zero in 61.10 s (four MPI ranks, one OpenMP thread).
The output reopened successfully with 20 snapshots instead of 19, advancing
code time from 1.08 to 1.18. All seven endpoint field arrays are finite; the
retained input checkpoint hash and mode are unchanged. The history file and
160 non-timing diagnostic lines are identical to the earlier partial run.
[The persistence record](current-drive.json) preserves endpoint/input/binary
hashes and resource measurements. This resolves the write failure, not the
physical current/force or topology gates. Native field extrema are recorded
without interpreting all-domain values as plasma-region measurements.

Native MAGVAL sampling now compares the driven and drive-off endpoints at
code time 1.18 on the same 64x64x32 grid and 192 retained physical targets.
The vector difference is **2.02591 mT RMS and 5.13595 mT maximum**. Plasma-response
RMS magnitudes are 12.91773 mT driven and 11.19831 mT drive-off. This is a
control sensitivity, not HINT–VMEX error or an equilibrium comparison. Both
controls use the same vacuum sample; subtracting it preserves their vector
difference. The [compact arrays](data/endpoint-controls.npz) and
[hashes/provenance](current-drive.json) retain this audit. Fields use native
cylindrical components at the same targets, so vector-difference norms are
rotation invariant. Target exteriority to relaxed current support, masks and
interpolation/grid convergence remain open.

A code-faithful stencil audit changes the interpretation of those targets.
MAGVAL uses an 8x8x8 tensor stencil; a literal reconstruction agrees with its
native field samples within 1.53e-14 T. Only 5/192 stencils have no positive
pressure nodes, and only 14/192 lie wholly inside the limiter. Every target's
nearest limiter node is inside, showing why nearest-cell classification would
miss this overlap. These are stencil properties, not proof that the target
points themselves lie inside the plasma. Interpolated pressure is negative at
53 targets because of high-order ringing, so its sign is not a support test.
The [per-target support counts](data/endpoint-support.npz) use native `P>0`
and `limiter>0` conventions; the separate 1e-12 weight screen is only a
sensitivity check. Driven/off pressure arrays match exactly.

The time-1.18 endpoint lacks native `ss` or `ss<jcuts`, so its 192 target samples
cannot be retrospectively classified. A subsequent bounded continuation with
the optional diagnostic advanced from time 1.18 to 1.28. Its saved
[eligibility mask](data/source-eligibility-mask.npz) has 14,899 eligible grid
nodes with `jcuts=1`, trace time 1.18 and endpoint time 1.28. The
[classification record](source-eligibility.json) and
[per-target arrays](data/source-eligibility-targets.npz) use the native 8x8x8
MAGVAL spline stencil: 5/192 targets have no eligible node in the full stencil;
7/192 have none among nodes with absolute interpolation weight above 1e-12.
At offsets 0.05 and 0.15 minor radii, every target overlaps eligible nodes;
at 0.3 and 0.5, the full-stencil zero counts are 1/48 and 4/48.
These are source-eligibility results for the **new** step-B entry, not labels
for the old time-1.18 field comparison. The mask is an amplitude-independent
superset of possible imposed-current source cells, not the attained current
support; response and return currents can extend beyond it. It excludes native
R/Z boundary layers. The raw full output and restart are still not publicly
hosted, so the compact extract does not independently reproduce the run.
Mesh, wall clearance, time relaxation and interpolation order must be varied
separately before admitting exterior targets to an inter-code field comparison.

The optional diagnostic patch now implements this output for the primary
NetCDF file. Set `lsource_support_diag=.true.` in `stepb_inp1`; it defaults
off. Variables `source_support`, `source_support_jcuts` and
`source_support_trace_time` describe the last frozen step-B eligibility,
while `t_snap` remains the endpoint time. Missing diagnostic records are fill,
not zero. HDF5, binary output and separate snapshot files are outside scope.
The fresh Release build passes off/on physical-array parity, byte-identical
one/four-rank toroidal output, legacy follow, disabled follow without stale
values, and explicit rejection of partial/wrong-type schemas. The short
zero-drive fixture validates output semantics, not mature QA support. The
production-platform build and single bounded continuation have completed for
the 0.5% case; the extract above is their available evidence. A repeated
independent read of the full raw output and current/force closure is still
required before interpreting this as a mature equilibrium. Historical runs
retain their original patch hashes.

Recompute the endpoint difference without a native build:

```sh
python - <<'PYCODE'
import numpy as np
with np.load('handoff/hint-qa/data/endpoint-controls.npz', allow_pickle=False) as a:
    delta = a['drive_off_total_cyl_T'] - a['driven_total_cyl_T']
    lengths = np.linalg.norm(delta, axis=1)
    print('RMS and maximum [T]:', np.sqrt(np.mean(lengths**2)), lengths.max())
PYCODE
```

## Execution order and research acceptance

1. **Freeze a working baseline.** The recorded runs use their pinned historical
   commits. Start a new validation on current main `926892ab7` (release 0.11.2),
   which includes #410, #416, #430 and #441. Reassess derivative admission on
   this source: the old #417/#418/#421/#423 stack closed without main promotion,
   and draft #306 still depends on this branch. Candidate evidence is not
   integrated-main evidence.
   Pin all source commits, dirty patches, input hashes,
   Python/JAX/compiler/MPI versions and device placement. Use `current` for
   HINT. Keep source-data and quadrature error distinct; `project_current`
   is an opt-in model change, not an invisible accuracy switch.
2. **Requalify current VMEX on 0.5%.** Use existing solve, force-diagnostic,
   implicit-gradient and free-boundary examples/tests. Compare independent
   cold roots, fresh raw residuals/FSQ, geometry, pressure, current and flux.
   Repeat a small matched CPU/GPU case before expensive runs. Check live
   parameter derivatives rather than differentiating cached field snapshots.
   Do not impose the removed historical universal certificate on new solvers.
3. **Resolve HINT current closure before a long campaign.** The bounded
   source/support and term-rate measurements above establish an active drive
   opposed by response and ideal terms. The original-length block confirms
   continuing current relaxation with a late force rebound. Resolve this
   nonmonotonic behavior before declaring a mature equilibrium. Save the
   actual flux label, imposed current and traced axis pressure at native updates. A snapshot reconstruction using maximum pressure as axis pressure
   is insufficient. Integrate current on multiple cuts and separately inside
   pressure support, return-current shells and wall. Near-zero all-domain
   current is not automatically a bug or a failed plasma-current target.
4. **Qualify the lower-beta equilibrium.** Extend each grid natively to the
   same later times, with fixed pressure history and cadence. Then vary grid,
   timestep, tracing length, resistivity and wall one at a time. Record final
   beta definition, pressure integral, enclosed current, displacement,
   divergence, force balance and parallel pressure variation. Retain at least
   three levels where a convergence order is claimed. Progress to 2.5% only
   after the procedure passes, preserving its original mismatch.
5. **Compare fields before topology.** Keep prescribed-boundary exterior
   reconstruction and matched free-boundary equilibrium comparisons separate.
   The former does not certify free-boundary force balance; the latter must
   match external coils, flux and pressure/current constraints. VMEX's nested
   surfaces and HINT's possible non-nested structure are distinct model scopes.
   Establish coil-only agreement with
   direct Biot–Savart; compare total, vacuum and plasma-response vectors at
   fixed targets in tesla and relative to the response. Include distance
   bins, maximum errors and target classification against WOUT, relaxed
   pressure/current support and wall. Refine HINT interpolation, VMEX surface
   data and virtual-casing quadrature independently. Check current projection
   separately. Avoid normalization by a vanishing response without a stated
   absolute floor.
6. **Use one tracer for quantitative topology.** Fix seeds, toroidal sections,
   coordinate conventions and wall; refine tracing tolerance, length and seed
   spacing independently of the field mesh. Measure transform, last closed
   surface, resonant islands and wall connection lengths. Report unresolved
   islands as unresolved; failure to reach a wall before a cap is not proof
   of a closed surface. Historical EXTENDER can supply an additional oracle
   if its required source/dependencies become available.
7. **Finish derivatives and optimization as a separate claim.** On qualified
   roots run tangent/adjoint residuals, duality and at least four perturbation
   halvings with independent nonlinear solves. Include pressure, boundary and
   coil directions and an external gradient oracle. #387 corrects the earlier
   claim that cold finite differences were intrinsically unusable: a valid
   asymptotic regime exists, but root selection matters. A final design must
   meet physical constraints and survive an independent cold reevaluation.
   Topology transitions are not ordinary smooth objectives.
8. **Publish a rerunnable result package.** Preserve negative results and all
   changed physical constraints; supply compact records, pinned commands,
   independently reviewable source changes and downloadable raw fields with
   checksums. A second machine/operator must reproduce the accepted tables.
   The present compact handoff lacks raw restarts and a current-API end-to-end
   reproduction runner; it is not yet that publication package.

Provisional study gates, to freeze before the final comparison: under 1%
change in global observables and under 5% in *resolved* island widths between
final refinements. Aim for each controlled numerical error contribution below
one fifth of the claimed inter-code difference, and report a combined error
budget without assuming independent errors. These are study targets, not
published guarantees; define absolute floors, masks and denominators before
evaluating them. Neither a small time change nor the input beta label alone
establishes equilibrium. Exact constraint matching and physical residuals are
prerequisites even if two codes agree numerically.

## Reuse and performance

The merged profiler correction [#420](https://github.com/uwplasma/vmex/pull/420)
repeats every stage of multi-stage warm workflows. Historical schema-1 warm
aggregates could omit diagnostics or derivatives; remeasure those complete
workflows before making performance claims. Its schema-2 contract is separate
from the source-specific bounded timings retained here.

Reuse VMEX's shipped `vmex_fieldline_tracing_finite_beta.py`,
`vmex_fixed_free_boundary_comparison.py`, and optimization examples, adapting
their inputs explicitly. Their successful execution is a software check, not
a HINT acceptance gate. Use the merged #409 surface-field fix.
Do not carry a second implicit solver or a duplicated historical helper suite
in this branch.

Reuse these examples as API references, preserving the benchmark physics.
`vmex_fixed_free_boundary_comparison.py` restricts a parent plasma and refits
coil currents; its result is not the unchanged QA comparison.
`vmex_fieldline_tracing_finite_beta.py` uses first-order near-surface
continuation and a distance cutoff. Quantitative island widths require an
independent continuation/field-accuracy study, not only tighter tracer tolerance.
The optional strong-force work in #412/#414 is supporting research, not a
prerequisite to every field comparison or proof of general 3-D convergence.
Use a common mesh and independent force oracle; separate native-state error
from WOUT export/refitting error.

Existing CPU qualification entry points, with dependencies installed from the
chosen pinned baseline, include:

```sh
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu VMEX_COMPILATION_CACHE=disabled RUN_FULL=1 \
  python -m pytest -q tests/test_examples.py::test_field_query_examples_run -k outside
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu VMEX_COMPILATION_CACHE=disabled RUN_FULL=1 \
  python -m pytest -q tests/test_examples.py::test_take_fixed_boundary_gradients
```

The first command passed at the exact #409 head and core versions recorded
above; cap BLAS/OpenMP/XLA thread pools when using a shared CPU host. It
exercises the exact 0.5% QA deck and finite field/VJP outputs. The second
remains a next-run command and checks a QA-family deck against central finite
differences. Neither supplies an exact 0.5% gradient-correctness certificate;
add a small exact-deck FD/duality case to the existing test infrastructure
before claiming one.

The current virtual-casing-jax floor is 0.0.7. Its
[`B_and_derivatives_xyz` path](https://github.com/uwplasma/virtual_casing_jax/pull/12)
is not yet wired into VMEX's nested spatial differentiation. This is a bounded
performance candidate after parity tests; the upstream speedup is not a VMEX
measurement. A small guarded extender override could combine fast plasma
spatial derivatives with the existing external-field derivatives, preserving
near-surface and unavailable-API fallbacks. First obtain the #14 tracer-cache
fix in a distinct released version, then test field composition, source
pullbacks and eager/JIT lifecycle. Do not change parameter derivatives or
infer a workflow speedup from pure-kernel timings. SOLVAX 0.22–0.24 add APIs
unused by current VMEX, so no upgrade is
required merely because those releases exist. SOLVAX #119 merged as
`2d96aa89`, after the current 0.24.0 release, and exposes inner-linear
convergence diagnostics; after a release, retaining these in polish
reports is a narrow useful integration. Keep strict rejection opt-in: the
current plan records unsuccessful strict-policy experiments. Boozer magnetic-only
evaluation
is a performance benefit covered by #410. ESSOS 0.17 supplies the used coil
and tracing contracts; experimental ESSOS PRs require manual review.

Measure accepted-work cost: cold compile, warm solve, refinement, derivative,
postprocessing, memory and total wall time separately on named hardware.
The #397 memory fix and #395 factor reuse invalidate older performance
extrapolations, but do not establish a speedup on these QA inputs. Keep local
CPU checks small; allow at most one GPU study and one bounded four-rank native
run on a shared compute host, after checking availability. Never stop unrelated
jobs. No larger run is justified by a timeout alone.

## Native build and literature

With Git authentication and an MPI GNU Fortran, NetCDF-Fortran and HDF5-Fortran
toolchain, from this repository's root:

```sh
git clone --branch current https://github.com/yasuhiro-suzuki/HINT3D.git HINT3D
git -C HINT3D checkout --detach bf31fc39f7179bdd91d84319c51c68e2f6fff25f
git -C HINT3D apply ../handoff/hint-qa/hint-portability.patch
python handoff/hint-qa/build.py --hint-root HINT3D --mode Debug --hdf5-prefix "$HDF5_PREFIX"
python handoff/hint-qa/build.py --hint-root HINT3D --mode Release --hdf5-prefix "$HDF5_PREFIX"
```

These commands build the upstream native suite with the source patch and
`build-release/MAGVAL/sample-points.exe` from the public driver. The sampler
uses the same compiled MAGVAL routines while replacing the upstream program
entry point. A new build is not expected to have the historical executable
hash. Rebuild and validate the toolchain on the target machine.

Recompute the published metrics without a native build or large raw fields:

```sh
python - <<'PY'
import json
from pathlib import Path
import numpy as np
root = Path('handoff/hint-qa')
samples = np.load(root / 'data/native-samples.npz', allow_pickle=False)
expected = json.loads((root / 'field-summary.json').read_text())
for label, kind in [('total_field', 'total'), ('vacuum_field', 'vacuum'),
                    ('plasma_response', 'response')]:
    def field(grid):
        if kind == 'response':
            return samples[f'{grid}_total'] - samples[f'{grid}_vacuum']
        return samples[f'{grid}_{kind}']
    delta = np.linalg.norm(field('fine') - field('coarse'), axis=1)
    measured = [np.sqrt(np.mean(delta**2)), delta.max()]
    record = expected['coarse_to_fine_metrics'][label]
    np.testing.assert_allclose(measured, [record['rms_vector_delta_T'],
                                       record['max_vector_delta_T']], rtol=1e-13)
    print(label, measured, 'T')
PY
```

To repeat native interpolation, first obtain the hash-matched raw snapshot
and vacuum files identified in `field-summary.json`, then run:

```sh
python handoff/hint-qa/sample_fields.py \
  --sampler HINT3D/build-release/MAGVAL/sample-points.exe \
  --targets handoff/hint-qa/data/targets.npz \
  --coarse-snapshot dataset/coarse.npz --coarse-vacuum dataset/coarse.nc \
  --fine-snapshot dataset/fine.npz --fine-vacuum dataset/fine.nc \
  --hint-commit bf31fc39f7179bdd91d84319c51c68e2f6fff25f \
  --output resampling.json --samples-output resampling.npz
```

The generic `dataset` names above are placeholders for the hash-matched
inputs, not a download location. Snapshot NPZs need `B_cyl`, `P_mu0_Pa`, `R`,
`phi`, `Z`; vacuum NetCDFs need the three `Bvac_*` components and native grid
metadata. The CLI rejects unequal snapshot/vacuum coordinates, uses native
cylindrical components in tesla and records canonical little-endian float64
array hashes. Install NumPy and netCDF4 in the postprocessing environment.

For current-drive diagnosis, apply the optional patch **after** the baseline
portability patch, rebuild HINT, and set `lcurrent_drive_diag=.true.` in the
existing `stepb_inp1` namelist:

```sh
git -C HINT3D apply ../handoff/hint-qa/hint-diagnostics.patch
python handoff/hint-qa/build.py --hint-root HINT3D --mode Release \
  --only HINT --hdf5-prefix "$HDF5_PREFIX"
```

It defaults to false and does no diagnostic reductions during the four
Runge–Kutta field evaluations. At energy/history updates its `CURRENT_DRIVE_DIAG`
records report context, source-support extrema, then cut-zero/min/mean/max
currents for the imposed source, attained curl(response) on that support, and
the attained pressure-mask comparator. Currents are amperes; `step_start_paxis`
is `mu0*p/B0^2`, the Bphi extrema are normalized by B0, and `timeb` is code
time. The cancellation record contains the cut-zero integral of absolute
source current and its signed/absolute ratio (both zero for zero source).
The timing is the largest per-rank wall time for the native source update,
not an end-to-end benchmark. Additional `CURRENT_DRIVE_RATE` records give
ideal, resistive-response, imposed-drive, cleaning, sum and total current rates
in A per code-time on the frozen source mask. The diagnostic saves/restores
all magnetic and velocity increments; it does not reuse a last-stage increment
as the instantaneous RHS. Its separate timing includes the repeated term
computations and reductions. The patch does not correct or reinterpret the
underlying current-drive model; use its output to choose the next controlled
experiment. Sequential application of both patches to the pinned upstream
source was verified to reproduce the tested source exactly.

To recreate the short diagnostic fixture, start from `data/coarse.in` and
the same hash-matched native inputs. Change `nstep=2`, `nstepb=2`,
`nenergyb=1`, `dt_b=1e-4`, `lc_in=0.2`, and `kprocs=1` (or 2 for the toroidal
decomposition check), retaining `iprocs=jprocs=1` and `OMP_NUM_THREADS=1`.
Toggle only `lcurrent_drive_diag` for the off/on comparison. Set both
`inet0=inet1=0` for the separate zero-source control. These deliberately short
traces and runs validate diagnostics only; they are not production settings.

The patch fixes pressure-table weights, uninitialized
values, sub-64-grid loop strides, periodic interpolation bounds, an external
callback declaration, read-only field access and a toroidal viscosity stencil.
The QA baseline's zero viscosity means the last fix does not explain its
current deficit. Native input preparation now has a public runner. Independent
end-to-end
reproduction still requires downloadable hash-matched restart assets, successful
endpoint persistence and a reviewed simulation/postprocessing sequence.

The [Geiger manuscript](https://conferences.iaea.org/event/214/contributions/17520/attachments/10058/15492/IAEA2020_JGeiger_Manuscript_8p_finalversion.pdf)
and supplied W7-X poster motivate this study: qualitative field agreement did
not establish quantitative island/edge agreement. Exact poster reproduction
needs configuration-specific inputs and final fields not supplied here.
[Historical EXTENDER](https://princetonuniversity.github.io/STELLOPT/EXTENDER.html)
offers coil/plasma/total separation and specified evaluation points; it is a
different implementation from modern VMEX.

[Landreman–Buller–Drevlak](https://doi.org/10.1063/5.0098166) supplies the
finite-beta QA context, including equilibrium/bootstrap-current consistency.
Prescribed-current agreement is not a new bootstrap-consistency result.
[Kanno et al. (1999)](https://doi.org/10.1017/S0022377898007405)
is a directly relevant current-carrying HINT precedent: the publisher abstract
reports adding net toroidal current and applying the revised code to LHD-like
nested equilibria. Only the abstract was accessible in this review. Its full
current prescription and normalization must be checked against maintained
`cal_netj` before treating it as a formula-level validation of the QA setup;
it does not explain or excuse the present attained-current deficit.
[Infinity Two, Appendices B–D](https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/magnetohydrodynamic-equilibrium-and-stability-properties-of-the-infinity-two-fusion-pilot-plant/6348ED5B1CA97BFF845C75F6284D5415)
provides accessible HINT–VMEC context with mesh, pressure-control and boundary
matching limitations; its configuration-specific thresholds do not transfer
to these QA cases. Its printed local-field denominator and the maintained
code's surface-profile current drive must not be silently equated. Read the
version-specific implementation before changing a current formula. Full-text
review of the 1989, 2006 and 2017 foundational HINT papers remains incomplete.

The June 2026 [systematic VMEC–HINT LHD comparison](https://arxiv.org/abs/2606.10490v2)
compares magnetic-axis position, on-axis transform and last-closed-surface
volume across three configurations. Its configuration-dependent edge
stochasticity supports measuring these observables alongside exterior fields;
it does not predict the QA outcome. The paper scans **axis beta**, while these
QA case labels denote **volume beta**. Do not transfer its transition thresholds
or interpret nested-surface failure as a numerical discrepancy alone. Its linear
initial pressure profile and zero-net-current context also differ from the QA
pressure/current inputs. Matching beta labels would not match the experiment.

### Immediate bounded experiments

Restart preflight, persisted readback and matched-time native field sampling
now pass. Next classify targets against pressure/source/wall support and resolve
the physical current/force evolution before extending relaxation. Persistence
and finite arrays alone do not establish field accuracy or equilibrium.

The M8/N6, NS121, solver-grid40 measurement32-to48 run reproduced the exact
native-state hash and converged in the same 1,357 iterations. Force L2 changed
by +0.00921% globally and at most 0.02630% in a radial bin, passing the
prospective 0.1%/0.5% measurement limits. The complete gate **fails**: the
s=0.284612 mean-force ratios change by -1.36978%, and the outer-bin
pointwise-normalized L2 changes by +1.01106%, exceeding the 1% limit.
These remain measurement sensitivities, not physical acceptance thresholds.
The run took 157.10 s, including 102.46 s solve, with 2.57 GB peak RSS on
one CPU core. All three native FSQ channels are below 1e-11; their sum is
1.50545e-11. NCURR=1 prescribes current: input -95673.86796 A versus exported
-95678.43865 A. The negative exported edge pressure (-36.316 Pa) is a WOUT
full-mesh extrapolation, distinct from the nearly zero prescribed edge value.

`--state-output results/state.npz` now completed end-to-end, retaining all six
native coefficient matrices with source/input/state hashes, resolution and solve
metadata. Independent non-pickle loading reproduces the exact state digest.
A paired WOUT was generated without a solve using public APIs. Re-reading it
with `read_wout`/`state_from_wout`, then replacing all six arrays from the NPZ,
restores the exact state hash and yields a finite native-field probe. WOUT alone
is not a bitwise native checkpoint. The pair is retained but not publicly
hosted. The full public replay now passes: at angular48 all 17 diagnostic
arrays are bitwise identical to the solve-produced archive. It rebuilds WOUT
scalars from the exact input/state, so unrelated same-resolution template
physics cannot alter normalization or finite-difference step sizes. Retained
FSQ/iteration metadata is labeled explicitly; replay does not re-solve or
freshly certify the nonlinear residual.

The subsequent angular48-to64 check passes all prospective measurement gates:
global force-L2 change 1.09e-6%, maximum radial-bin force-L2 change 2.06e-6%,
maximum mean/ratio change 0.2442%, and maximum pointwise-normalized change
0.2823%. The angular64 force L2 is 424642.199 N/m3. Hardened replays use about
2.25 GB peak RSS; the exact runtime/source records are in `vmex-cpu.json`.
The earlier angular32-to48 failure remains part of the record. This qualifies
measurement at the tested refinements on this state and five radial surfaces,
not radial/Fourier resolution or continuum equilibrium. Next compare M9/N6 and
M8/N7 separately, preserving NS121/grid40 and using the qualified measurement
settings. Save each state and check its own measurement sensitivity if needed.

To replay with the same source/input pins and output arguments as a fresh
force diagnostic, additionally pass:

```sh
--state-input dataset/state.npz --template-wout dataset/state.wout.nc \
  --expected-state NATIVE_STATE_SHA256 --angular-count 64
```

The `dataset` paths are placeholders for the hash-matched retained pair, not
download locations. The runner rejects source/input/state identity mismatches,
wrong shapes and nonfinite arrays, and refuses existing output files.

Keep broader QI objective, factor-reuse and 3-D polishing development off this
study's critical path. The latest #413/#419 evidence includes failed independent
root and fine-grid gates despite accurate linear responses. Optimize complete
accepted-work cost only after the underlying root and observable are qualified.


A subsequent bounded native-device check at #423 head `02bf33d9` uses JAX
0.11.1/SOLVAX 0.22.0 with x64 and JIT enabled. All six returned state arrays
reside on the explicitly selected CPU or GPU. A loose Solovev placement probe
has identical packed state hashes. A stronger LASYM/theta-flipped case
(MPOL3, NS5, FTOL1e-8) converges in 222 iterations on each device; scalar state
norms agree within 2.8e-13 relative. Full asymmetric coefficient differences
were not retained. These are native-solve placement results, not tests of the
implicit callback, certificate, derivatives, exact QA deck or free boundary.
The exact environment and limits are in `vmex-cpu.json`; cold timings do not
establish performance.


The separate M9/N6 and M8/N7 controls now complete with NS121, solver grid40
and angular64 measurement. Both meet the three native FSQ limits. M9/N6
raises force L2 by 10.18% globally and 50.84% in the innermost sampled bin;
M8/N7 changes it by -0.422% globally and at most 0.885% across bins. The
poloidal sequence is nonmonotonic. Parsed boundary coefficients are unchanged:
newly admitted boundary modes are zero. Common flux-coordinate points map to
different physical positions, so these are directional resolution sensitivities,
not fixed-position field errors or a convergence certificate. All six state
arrays and paired WOUT templates are retained. Their angular64-to80 replays
now pass all unchanged measurement gates without solving again: maximum
mean/ratio/pointwise changes are 0.104% for M9/N6 and 0.182% for M8/N7;
global and per-bin force L2 change below 1e-9%. Thus angular64 sampling does
not explain the large poloidal force change on this window. Prioritize
poloidal resolution and root-family checks over a broad toroidal sweep;
five-surface sampling still does not certify the full volume.


A separate combined #423 test now instruments the native solve before host
conversion and follows the same small asymmetric state through the status
callback and certificate. CPU/GPU both converge in 222 iterations; native
coefficients and callback returns match exactly within each run. All six
coefficient arrays are retained and independently compared: maximum CPU/GPU
difference is 2.74e-12, or 8.11e-11 relative to the corresponding coefficient
family scale. Certificate state/parameter identities and repeated measurement
match. This uses refinement disabled and `primal_tol=None`: configured
admission passes, while `strict_root_certified` is false. It does not close
the refined exact-QA, derivative, optimization or free-boundary GPU gates.

## September 24 handoff review

The review covered this branch's source changes, portable inputs, numerical
records, two HINT patches, the creator's LHD manual, Geiger's W7-X poster,
current VMEX main, and the live PR graph. This branch changes VMEX only through
research helpers, compact evidence, documentation and restart-preflight tests;
it does not replace VMEX equilibrium operators. The HINT portability patch
and optional diagnostic patch apply to the pinned `current` source, but remain
local research changes requiring independent upstream review. Short serial/MPI
regressions and the single bounded 0.5% continuation establish that those
paths run; they do not show a converged HINT QA equilibrium or certify all HINT
outputs, boundary conditions or long-term stability.

The most useful next discriminator is a **same-physical-point** field and
force audit of the saved M8/N6, M9/N6 and M8/N7 VMEX states, with root-family
and constrained-mode provenance explicit. The observed M9 force increase is
not explained by the checked angular sampling; a new solve ladder should wait
for that diagnostic. For HINT, check attained toroidal current and force
balance through later native continuations on multiple grids and independently
classify support at each saved time. Then establish a coil-only baseline and
compare vacuum, plasma-response and total vectors at common targets with
controlled HINT interpolation and VMEX quadrature. Use #441's graded rule near
the boundary with its reported estimate. Repeat at 2.5% only after the 0.5%
method and its coil/WOUT mismatch accounting are stable. Island-width claims
require a separate common-tracer convergence study.

The PR remains draft because no accepted inter-code field table, converged
HINT current/force state, common topology result, full raw restart package or
independent cold reproduction exists. Its earlier CI run passed the fast,
physics, quality and two-device checks, but the C2 manifest lane and aggregate
gate failed; that failure must be reevaluated on the current CI definition.
Do not interpret this branch's older `plan.md` diff as replacing the merged
[current research plan](https://github.com/uwplasma/vmex/blob/main/plan.md).
