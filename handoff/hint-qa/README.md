# HINT–VMEX independent-agent handoff

This work tests whether maintained HINT free-boundary equilibria and magnetic fields outside a prescribed plasma boundary agree with VMEX's VMEC/virtual-casing/extender capabilities for the existing finite-beta QA cases labelled 0.5% and 2.5%. It also establishes which VMEX roots and derivatives are valid before using them in optimization. **No converged HINT QA comparison or quantitative island-width agreement has been obtained.**

The user selected the existing finite-beta cases, authorized independent parallel work and normal PR merges, and requested this portable handoff. Preserve the original pressure, current, toroidal flux, boundary and associated coil data. The legacy filenames say LandremanPaul, but these are finite-beta Landreman–Buller–Drevlak cases. Their outer boundaries coincide; their interior equilibria, current and coils differ. They are not a beta-only scan. The 2.5% coil/WOUT mismatch must remain visible; “associated coils” does not mean an exact free-boundary match.

## Source branches and commits

| Repository | Required source and disposition |
|---|---|
| [HINT3D](https://github.com/yasuhiro-suzuki/HINT3D) | The creator explicitly requested **`current`**, not the infrequently maintained `master`. Tested baseline: [`bf31fc39f7179bdd91d84319c51c68e2f6fff25f`](https://github.com/yasuhiro-suzuki/HINT3D/commit/bf31fc39f7179bdd91d84319c51c68e2f6fff25f). Apply the [bundled source patch](hint-current-validation.patch). Those eight-file corrections are not claimed to be upstream. |
| [VMEX PR302](https://github.com/uwplasma/vmex/pull/302) | Branch **`fix/hint-comparison-derivative-contract`**. Numerical evidence uses [`14360d179af4534aa9bb638b172c8724ee289d9b`](https://github.com/uwplasma/vmex/commit/14360d179af4534aa9bb638b172c8724ee289d9b). Later handoff-only commits do not retroactively change the tested source. Keep the PR draft until required checks and review pass. |
| [SOLVAX PR104](https://github.com/uwplasma/SOLVAX/pull/104) | Tested head [`e4b507185464851ac5e8de22041f2f8e384554e9`](https://github.com/uwplasma/SOLVAX/commit/e4b507185464851ac5e8de22041f2f8e384554e9), branch `fix/vmex-solver-contract`; merged externally as [`66f97a6e0eb758af8d7f46939ffdd8ec733efbef`](https://github.com/uwplasma/SOLVAX/commit/66f97a6e0eb758af8d7f46939ffdd8ec733efbef). All 11 checks and exact-source CPU/GPU contract validation passed. |
| [VMEX PR299](https://github.com/uwplasma/vmex/pull/299) | Related derivative-reuse/performance work; reconcile its state/derivative contract before combining. Head observed during handoff: `de93e5e20a3ba3c2cd9899cb0543ef33702614e4`. Its earlier performance evidence is not current QA certification. |
| [VMEX PR300](https://github.com/uwplasma/vmex/pull/300) | Separate DESC-conversion work. Head changed externally to `1a10c646ed90aca9c9ace332b372b4072cf308e8`; earlier green checks/review at `798d671e` do not certify this newer head. Inspect current checks/reviews before any merge. |

Git authentication may be needed on the next machine. No specific home directory, SSH alias, installed environment or private filesystem is required by this handoff. Do not bypass required reviews or stop unrelated compute jobs.

## What changed and what was measured

PR302 admits ordinary derivatives only at the actual returned state: finite projected residual below `primal_tol`, normalized raw FSQ, and valid geometry. Historical host stopping FSQ is separate. Disabling Newton refinement does not disable admission. Refinement Krylov dimensions no longer inherit a deliberately exhausted adjoint budget. Uncertified states remain value-only; eager errors and traced invalid derivatives are explicit.

Certified caches retain immutable coefficients and matching host diagnostics under a reentrant lock. Cached fields support spatial derivatives, but **parameter VJPs through cached snapshots are unsupported**. Use the live parameterized solve for the examples here. Radial lifting rejects knot spans lacking an interior source sample; that is not a conditioning or high-order-formulation certificate.

| Lane | Measured result | Limit / next gate |
|---|---|---|
| PR302 regression repair | All three previously failed selections pass; 32 further focused tests, static checks and strict docs pass. Changed-line coverage: 167/170 = 98.2%. | Broad CI and independent review remain separate requirements. |
| QA 0.5% CPU root | Original ns31/MPOL7/NTOR6/NFP2, `ncurr=1`. Two cold processes give identical states; residual `4.4748087e-11 < 1e-10`. Solve/certificate times 21.64/26.57 s. | One fixed-boundary resolution, not mesh/free-boundary convergence. |
| QA root CPU/GPU | All 75 VMEX and 23 SOLVAX Python source hashes, inputs and tested helper match. State relative difference `4.1790e-12`; both actual-primal gates pass. | GPU-environment result with source placement audit, not a kernel-profile or speedup claim. GPU solve/certificate: 130.30 s. |
| QA CPU directional derivative | Live magnetic-energy objective, one `Rbc(m=1,n=0)` direction; JVP/VJP duality error `2.72e-12`. Eight perturbed live solves in one process pass, with hot restart disabled. Taylor orders `1.979, 1.963, 1.836`. | Small-step central FD errors worsen from `4.22e-7` to `1.09e-5`; do not claim monotone FD convergence or a full gradient matrix. GPU derivative run is prepared, not executed because devices were occupied. |
| Synthetic calibration | Safeguarded scalar Newton accepted `q=4.99739021734e-5` toward synthetic target `q=5e-5`; energy error fell from `1.20970e-4` to `6.31091e-8`. | **Timed out** before `1e-9` tolerance and independent cold final check. Initial scalar-gradient phase took 147.37 s and showed compilation activity. No optimized-design claim. |
| HINT resistivity | Same restart and final time: eta `.001/.002/.005` yields positive-pressure current `−17.692/−19.889/−25.521 kA`. | Target is `−95.778 kA`; force ratio worsens `.014006/.015232/.019714`. Higher eta is not promoted. |
| HINT exterior sensitivity | Eta `.002/.005` differ from `.001` by `1.523/5.408 mT` RMS at 192 native-sampled fixed targets. | Transient numerical-control differences, not HINT–VMEX error. Targets exterior to WOUT may not be exterior to relaxed current. |
| HINT edge audit | Most opposing current lies 1–8 cm outside pressure support; the first 1–2 cm band is about one coarse grid step. Outer circulation independently supports near-zero total cut current. | Total-domain current need not equal plasma target. Matched fresh startup runs finished on both grids; their edge band follows grid scale, so no mature-grid convergence is established. |
| LHD reference | Maintained native workflow and coarse continuation to normalized time 100; last saved pressure change 0.5773%, peak 26.5165 kPa, pressure integral 279584.83 J. | Time/grid/reference-topology gates and manual 256-grid reproduction remain open. |

The computed QA edge-extrapolated current is `−95778.34986522896 A`, equal to WOUT. Input `CURTOR=−95673.86795711746 A`; retain the `104.481908 A` finite-mesh edge-extrapolation distinction. Pressure half-mesh relative error is `2.05e-16`. Reference volume beta is 0.5038845418%; the supplied 2.5% case is 2.5469063197%. Reference beta read from WOUT is not a newly measured relaxed beta.

## Reproduce VMEX diagnostics

Clone the PR branch to obtain this handoff, then create a separate pinned numerical checkout. Commands below run from that clone's root:

```sh
git clone --branch fix/hint-comparison-derivative-contract https://github.com/uwplasma/vmex.git vmex-handoff
cd vmex-handoff
git worktree add --detach ../vmex-qa-measured 14360d179af4534aa9bb638b172c8724ee289d9b
git clone https://github.com/uwplasma/SOLVAX.git ../solvax-qa-measured
git -C ../solvax-qa-measured checkout --detach e4b507185464851ac5e8de22041f2f8e384554e9
python3 -m venv .venv-qa
. .venv-qa/bin/activate
python -m pip install -e ../solvax-qa-measured -e ../vmex-qa-measured 'jax==0.9.2' 'jaxlib==0.9.2' netCDF4
JAX_PLATFORMS=cpu python handoff/hint-qa/helpers/tested-cpu-14360d17/qualify_finite_beta_qa_root.py \
  --vmex-root ../vmex-qa-measured --solvax-root ../solvax-qa-measured \
  --input handoff/hint-qa/inputs/beta0p5/input.LandremanPaul2021_QA_beta0p5_bootstrap \
  --reference-wout handoff/hint-qa/inputs/beta0p5/wout_LandremanPaul2021_QA_beta0p5_bootstrap.nc \
  --output results/qa-root-cpu-001 --cold-runs 2 --per-run-timeout 240
JAX_PLATFORMS=cpu python handoff/hint-qa/helpers/tested-cpu-14360d17/qualify_finite_beta_qa_derivatives.py \
  --vmex-root ../vmex-qa-measured --solvax-root ../solvax-qa-measured \
  --input handoff/hint-qa/inputs/beta0p5/input.LandremanPaul2021_QA_beta0p5_bootstrap \
  --output results/qa-derivatives-cpu-001 --step 0.0001 --per-run-timeout 240
```

These commands pin JAX and solver source; they are not a complete transitive environment lock. Use a compatible NVIDIA JAX installation for GPU, inspect occupancy first, and expose `JAX_PLATFORMS=cuda,cpu` with the selected `CUDA_VISIBLE_DEVICES`. Host callbacks need the CPU backend. Disable shared compilation caching as the helpers do. GPU derivative budget: 600 s worker / 630 s external process-group cap. The resource runner in `hint-support/` provides per-host locks and process-group cleanup when copied into HINT's `local-support/`; do not confuse reports copied between hosts with live locks.

The latest root helper additionally skips physical diagnostics for status-1 placeholder states. The exact tested pair is retained separately with its original hashes; successful status-0 behavior is unchanged. [Helper details](helpers/README.md) explain this distinction. [Calibration script](helpers/calibrate_finite_beta_qa_energy.py) accepts `--target results/qa-derivatives-cpu-001`, but its recorded attempt is incomplete. Diagnose compilation/staging first; a promising next implementation uses the already measured full-parameter gradient contracted with the fixed direction, with separate primal/gradient timings. Do not merely raise its timeout or weaken tolerances. VMEX's ordinary `custom_vjp` cannot directly satisfy a nonlinear wrapper that invokes `jax.linearize`; no generic SOLVAX nonlinear-JVP integration is claimed.

Both exact WOUTs are [bundled](inputs/SHA256.json): they are absent from the verified public `reference-nc` release bundle. Decks and associated ESSOS coil JSONs also remain [tracked in VMEX](https://github.com/uwplasma/vmex/tree/14360d179af4534aa9bb638b172c8724ee289d9b/examples/data); copies here make the study inputs explicit.

## Reproduce maintained HINT and prepare inputs

```sh
git clone --branch current https://github.com/yasuhiro-suzuki/HINT3D.git ../HINT3D
git -C ../HINT3D switch -c study/hint-qa-validation bf31fc39f7179bdd91d84319c51c68e2f6fff25f
git -C ../HINT3D apply ../vmex-handoff/handoff/hint-qa/hint-current-validation.patch
mkdir -p ../HINT3D/local-support ../HINT3D/qa-source ../HINT3D/runs
cp handoff/hint-qa/hint-support/* ../HINT3D/local-support/
python handoff/hint-qa/helpers/stage_inputs.py --hint-root ../HINT3D
python handoff/hint-qa/helpers/build_suite.py --hint-root ../HINT3D --mode Debug --hdf5-prefix "$HDF5_PREFIX"
python handoff/hint-qa/helpers/build_suite.py --hint-root ../HINT3D --mode Release --hdf5-prefix "$HDF5_PREFIX"
```

Provide GNU Fortran/MPI (`FC`), NetCDF-Fortran (`NF_CONFIG`) and HDF5-Fortran (`HDF5_PREFIX`) from the new machine's toolchain. Seven bundled postprocessor regressions pass; Python syntax, input staging (including idempotency), helper hashes and patch application against the pinned HINT baseline were verified. The portable build-helper CLI was syntax/help checked; rebuilding on a new toolchain remains required. It builds HINT, MKVAC, MKFLX, MKLIM, HMAG, GPRTS and MAGVAL in isolated directories. Run HINT from the input directory with `OMP_NUM_THREADS=1 mpiexec -n 4 .../build-release/HINT/hint.exe < hint.input`; rank count must equal the namelist decomposition. Installation can place the built executable on the user's chosen PATH; no prior installation is assumed.

The patched defects and limits are described in [source review](context/REVIEW.md). Pressure diagnostics, periodic bounds, allocation initialization, callback declarations, read-only field access and a duplicated toroidal-viscosity stencil index were corrected. These are eight source files, not proof of a bug-free solver. QA's current default viscosity is zero, so that stencil fix does not explain its current deficit.

The [native support scripts](hint-support/) expect to be copied into the HINT checkout's `local-support/`; their workspace root is its parent. The portable input convention is `qa-source/beta*/inputs/` for WOUT/deck/coil files and `qa-source/beta*/current_mapping.csv` for the current table. Use the staging helper described below to create this layout. `prepare_qa_wall.py` generates the 10/12/15 cm numerical walls, native flux/limiter decks; the staging helper installs the exact bundled fixed targets. `prepare_qa_vacuum.py --source-root qa-source --case beta0p5 --nr 64 --ntor 32 --platform cpu` samples the same associated coil field; use `128/64` for refinement. These require ESSOS and the virtual-casing extras declared by VMEX, plus netCDF4, Shapely and plotting/scientific Python dependencies. Record their versions; do not upgrade an unrelated environment.

From the VMEX handoff checkout used above, switch to HINT before preparing geometry and grids:

```sh
cd ../HINT3D
python local-support/prepare_qa_wall.py
JAX_PLATFORMS=cpu python local-support/prepare_qa_vacuum.py --source-root qa-source --case beta0p5 --nr 64 --ntor 32 --platform cpu
```

Run native MKFLX/MKLIM with generated namelists, link the corresponding vacuum grid, verify coordinates/NFP and initial-plasma containment, then use the bundled HINT decks. The [edge protocol](context/QA_EDGE_REFINEMENT_PLAN.md) specifies the fresh 64×64×32 / 128×128×64 pair: four outer steps × 250 magnetic steps, common `dt=1e-5`, `eta0=.001`, total time `.01`, four MPI ranks. Native HINT rejects grid-changing restarts; no coarse-field remap is authorized by this protocol. The 128 limiter has been generated and all 89,754 initial plasma cells are contained. Both runs completed: coarse 108.70 s, fine 1041.86 s within its 1200 s cap and memory/disk guards. At time .01 the pressure-supported currents are −310.316 A and −257.749 A, far from the −95.778 kA target. The dominant opposing-current shell shifts from 1–2 cm on the coarse grid to 0–1 cm on the fine grid, exposing grid-scale startup structure. See the [final edge result addendum](evidence/qa-edge-refinement-results-001/QA_EDGE_REFINEMENT_RESULT.md) for field decomposition and limitations.

Full historical HINT restarts are not bundled. Hashes in reports identify those artifacts but cannot resume them. Reproduce the fresh setup or obtain the original fields separately. This limitation is preferable to commands that depend on an undisclosed private directory.

## Next work, in order

1. Complete PR302 exact-head CI/review; reconcile overlapping PR299 before combining performance changes. SOLVAX104 is already merged.
2. Interpret the completed matched fresh HINT grid pair at common physical coordinates, pressure-distance shells and fixed native targets. Compare divergence, force, current and wall support. Startup evidence at `.01` does not certify the mature `1.08` state; continue each grid natively to a common later time only under a new bounded protocol. Do not choose eta from plasma current alone or drive full-vector `J − J_parallel` to zero.
3. Run the prepared exact-source GPU derivative screen when a device is free. Profile and complete the small synthetic calibration with an independently cold final solve; then expand to other directions/objectives, an external gradient oracle and a genuinely feasible optimization. Cached snapshot parameter VJPs remain unavailable.
4. Qualify the full 0.5% equilibrium and given-boundary/exterior comparison before 2.5%. Require current/pressure, time/grid/trace/wall checks and exterior-to-relaxed-current classification. Keep any separately matched free-boundary case distinct from the original inputs. Complete LHD reference convergence in parallel within the compute budget.
5. Only then use a common field-line tracer for islands/edge topology, with separate interpolation, quadrature, wall, relaxation and tracing uncertainty. Published qualitative agreement does not supply a quantitative island-width tolerance.

## Literature and evidence map

- [Literature synthesis and access limits](context/LITERATURE_GAP_REVIEW.md), [earlier context](context/LITERATURE_CONTEXT.md), [input contract](context/QA_INPUT_CONTRACT.md), [cadence audit](context/QA_CADENCE_AUDIT.md), [resistivity/spatial assessment](context/QA_RESISTIVITY_ASSESSMENT.md), [VMEX qualification](context/QA_VMEX_QUALIFICATION.md), [acceptance matrix](context/GATE_MATRIX.md).
- [Compact numerical evidence](evidence/) and [serialization limits](evidence/README.md). Original source/input hashes are preserved; machine locations are replaced by logical labels, so exported JSON bytes are not claimed to equal the private originals.
- The supplied Geiger/Suzuki W7-X poster reported qualitative HINT–VMEC/EXTENDER agreement but quantitative island/edge differences. Exact W7-X inputs are unavailable here. The [Geiger manuscript](https://conferences.iaea.org/event/214/contributions/17520/attachments/10058/15492/IAEA2020_JGeiger_Manuscript_8p_finalversion.pdf) and [EXTENDER documentation](https://princetonuniversity.github.io/STELLOPT/EXTENDER.html) provide context; historical EXTENDER is distinct from modern VMEX.
- The creator's [LHD example collection](https://u.pcloud.link/publink/show?code=kZwo5G5ZYPUNwfLwGiXRxQ4g7zFqTQG6Cl6y) contains the manual/example inputs. Preserve original downloads and hashes. The supplied script names needed explicit r360/vessel-name corrections; smoke decks and reference decks must remain distinct. Publisher full text for the 1989/2006/2017 foundations was not obtained; do not claim those methods were fully audited.
- [Landreman–Buller–Drevlak finite-beta QA](https://doi.org/10.1063/5.0098166), [Infinity Two HINT comparison](https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/magnetohydrodynamic-equilibrium-and-stability-properties-of-the-infinity-two-fusion-pilot-plant/6348ED5B1CA97BFF845C75F6284D5415), and [2026 LHD preprint](https://arxiv.org/abs/2606.10490v2) motivate constraint matching and convergence checks; their device-specific thresholds are not QA acceptance criteria.

As-of date: 13 September 2026. Scientific gates, source commits and explicit evidence take precedence over stale historical job prose in archived context files.
