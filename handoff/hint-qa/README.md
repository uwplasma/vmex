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
| [VMEX PR #302](https://github.com/uwplasma/vmex/pull/302) | Branch `fix/hint-comparison-derivative-contract` is being reconciled with current main. Its old numerical source `14360d179af4534aa9bb638b172c8724ee289d9b` is historical evidence, not certification of this branch. |
| [SOLVAX #104](https://github.com/uwplasma/SOLVAX/pull/104) | Tested head `e4b507185464851ac5e8de22041f2f8e384554e9`, merged as `66f97a6e0eb758af8d7f46939ffdd8ec733efbef`; released in 0.21.0. Use at least 0.21.0 for its rejection of nonfinite nonlinear roots. |
| [VMEX #409](https://github.com/uwplasma/vmex/pull/409) | Open at review: fixes the surface-field assembly regression introduced by #403. Include its reviewed fix before current-source field qualification. |
| [VMEX #410](https://github.com/uwplasma/vmex/pull/410) | Open draft at review: owns dependency-floor changes. Complete exact-floor validation there rather than duplicate it here. |

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
hash is not the HINT solver executable hash. Raw snapshot/sample arrays and
the custom sampling driver/postprocessor sources are not bundled, so these
numbers are retained diagnostic evidence, not independently reproducible from
this compact bundle. The exact prepared HINT decks, current table and
wall/limiter/flux/vacuum preparation inputs are also absent. The public source
decks/coils and reference WOUTs alone cannot recreate the measured fresh pair.
Package these missing materials with generic names and explicit hashes before
promoting the record to reproducible research evidence.

## Execution order and research acceptance

1. **Freeze a working baseline.** Incorporate the reviewed #409 fix and
   validate #410's floors. Pin all source commits, dirty patches, input hashes,
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
5. **Compare fields before topology.** Establish coil-only agreement with
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

Existing CPU qualification entry points, with dependencies installed from the
chosen pinned baseline, include:

```sh
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu VMEX_COMPILATION_CACHE=disabled RUN_FULL=1 \
  python -m pytest -q tests/test_examples.py::test_field_query_examples_run -k outside
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu VMEX_COMPILATION_CACHE=disabled RUN_FULL=1 \
  python -m pytest -q tests/test_examples.py::test_take_fixed_boundary_gradients
```

These are next-run commands, not results of this review. The first exercises
the exact 0.5% QA deck and finite field/VJP outputs; the second checks a
QA-family deck against central finite differences. Neither supplies an exact
0.5% gradient-correctness certificate; add a small exact-deck FD/duality case
to the existing test infrastructure before claiming one.

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

These commands build the upstream native suite with the source patch; they
do not build the custom sampler used for the retained field record. The build
helper is retained for portability; rebuild and validate the toolchain on the
target machine. The patch fixes pressure-table weights, uninitialized
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
