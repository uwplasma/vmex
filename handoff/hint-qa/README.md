# HINT–VMEX finite-beta QA comparison

Reviewed 2026-09-21 against VMEX main
[`f719c4ff503c526368254aebf3ba9a9119c17f87`](https://github.com/uwplasma/vmex/commit/f719c4ff503c526368254aebf3ba9a9119c17f87).
This is a research protocol and historical baseline, **not a qualified
HINT–VMEX equilibrium comparison**. The two supplied finite-beta QA cases
remain the goal: nominal volume beta 0.5% and 2.5%, first comparing fields
outside the given boundary, then a separately matched free-boundary study.

## Sources and scope

| Source | Required interpretation |
|---|---|
| [HINT3D `current`](https://github.com/yasuhiro-suzuki/HINT3D/tree/current) | The creator maintains this branch; do not substitute `master`. The measured source is `bf31fc39f7179bdd91d84319c51c68e2f6fff25f` plus [the eight-file patch](hint-portability.patch). The patch is not upstream or proof of a bug-free solver. |
| [VMEX PR #302](https://github.com/uwplasma/vmex/pull/302) | Branch `fix/hint-comparison-derivative-contract` is reconciled with the pinned current-main baseline. Its old numerical source `14360d179af4534aa9bb638b172c8724ee289d9b` is historical evidence, not certification of this branch. |
| [SOLVAX #104](https://github.com/uwplasma/SOLVAX/pull/104) | Tested head `e4b507185464851ac5e8de22041f2f8e384554e9`, merged as `66f97a6e0eb758af8d7f46939ffdd8ec733efbef`; released in 0.21.0. Use at least 0.21.0 for its rejection of nonfinite nonlinear roots. |
| [VMEX #409](https://github.com/uwplasma/vmex/pull/409) | Open at review: fixes the surface-field assembly regression introduced by #403. Include its reviewed fix before current-source field qualification. |
| [VMEX #410](https://github.com/uwplasma/vmex/pull/410) | Open draft at review: owns dependency-floor changes. Complete exact-floor validation there rather than duplicate it here. |
| [VMEX #416](https://github.com/uwplasma/vmex/pull/416) | Open candidate: removes history-dependent suppression of free-boundary cold recovery. Qualify repeated accepted points after rejected trials on the integrated source. |
| [VMEX #417](https://github.com/uwplasma/vmex/pull/417) | Open candidate: anchors derivative admission and reuse to measured refined coefficients. Exact-case QA and GPU evidence remain necessary. |
| [VMEX #413](https://github.com/uwplasma/vmex/pull/413) | Proposed replacement product plan. Reconcile after integration; its six research lanes do not replace this study's physical comparison gates. |

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
The exterior path on unchanged main still needs #409. In a fresh environment,
the exact 0.5% outside-field test subsequently passed at reviewed #409 head
`aabedb8f2212747ed2651fb13ba81468cc5f7039`: **1 passed, 1 deselected in
132.23 s**, CPU-only, within a 240 s process-group cap. It actually exercised
the live-state surface assembly, exterior field, spatial derivatives through
third order and all four VJP outputs, with `accuracy_check='raise'`. Assertions
check finite/nonzero B and |B| and finite VJP maxima with at least one nonzero
maximum; they do not inspect every derivative or VJP component. This is not an import skip, FD/duality comparison,
GPU result, or independent physical-force certificate. The focused live-state
surface regression also passed separately. Required GitHub review and CI
remain prerequisites to merging #409.

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
reading uninitialized normalization volumes. The tested final source hash is
`bc4913186bd6de2a0118ba9f1e65a252160420cc47d62733abd0fc398c2e8fc2`.

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
inputs and native objects. A separate clean build passed a constant-field
check; that clean build has not been tested on the large raw fields. The
portable [coarse](data/coarse.in) and [fine](data/fine.in) native decks retain
all measured numerical namelists, including the current-profile table; only
their header comments and limiter/history filenames changed. Original deck
and prepared vacuum/flux/limiter hashes are recorded separately. Those large
prepared fields and the wall-generation inputs are not bundled, so the
source decks/coils and WOUTs alone cannot recreate the measured fresh pair.
Package the remaining materials before promoting this to an independently
reproduced equilibrium result.

## Execution order and research acceptance

1. **Freeze a working baseline.** Incorporate the reviewed #409 fix and
   validate #410's floors. Include accepted #416 recovery changes before
   free-boundary qualification and #417 state/derivative changes before
   sensitivity claims; candidate evidence is not integrated-main evidence.
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
3. **Resolve HINT current closure before extending runtime.** Save the actual
   flux label, imposed current and traced axis pressure at a native drive
   update. A snapshot reconstruction using maximum pressure as axis pressure
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

Reuse VMEX's shipped `vmex_fieldline_tracing_finite_beta.py`,
`vmex_fixed_free_boundary_comparison.py`, and optimization examples, adapting
their inputs explicitly. Their successful execution is a software check, not
a HINT acceptance gate. Fix #409 before their surface-field paths are used.
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
measurement. SOLVAX 0.22–0.24 add APIs unused by current VMEX, so no upgrade is
required merely because those releases exist. Boozer magnetic-only evaluation
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
not an end-to-end benchmark. The patch does not correct or reinterpret the
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
current deficit. Native input preparation and current-drive diagnostics still
need a compact public runner before an independent end-to-end rerun.

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
[Infinity Two, Appendices B–D](https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/magnetohydrodynamic-equilibrium-and-stability-properties-of-the-infinity-two-fusion-pilot-plant/6348ED5B1CA97BFF845C75F6284D5415)
provides accessible HINT–VMEC context with mesh, pressure-control and boundary
matching limitations; its configuration-specific thresholds do not transfer
to these QA cases. Its printed local-field denominator and the maintained
code's surface-profile current drive must not be silently equated. Read the
version-specific implementation before changing a current formula. Full-text
review of the 1989, 2006 and 2017 foundational HINT papers remains incomplete.
