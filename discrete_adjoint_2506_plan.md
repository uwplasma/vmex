# Discrete-Adjoint Recovery Plan for QH Optimization

Last updated: 2026-04-14
Repo: `vmec_jax`
Branch: `codex/discrete-adjoint-2506`
Coupled repo: `simsopt` on `codex/discrete-adjoint-2506`

## Goal

Recover a fast and correct autodiff path for the fixed-boundary QH benchmark used by
`simsopt/examples/2_Intermediate/QH_fixed_resolution_jax.py`, without finite
differences, by adapting the discrete-adjoint ideas from
`arXiv:2506.14792 / Fast automated adjoints for spectral PDE solvers`.

Target benchmark:

- objective: `aspect + quasisymmetry`
- no `mean_iota` term for QH
- `max_mode = 1`
- VMEC `mpol = ntor = 3`
- 8 free boundary DOFs
- `max_nfev = 10`

Reference:

- classic `QH_fixed_resolution.py`: final objective about `0.21378863910867005`
- recent VMEC-JAX path: materially worse objective and much worse runtime

## What the paper actually contributes

The paper is not "use AD on everything and hope it works". The practical method is:

1. represent the forward PDE problem using solver-owned operator graphs,
2. reverse the high-level solver control flow,
3. provide efficient VJPs for the solver's low-level sparse operators,
4. reuse the same sparse linear-solve machinery in the backward pass,
5. checkpoint only where the time-marching or Newton loop requires replay.

The examples confirm this split:

- EVP: use left eigenvectors / solver sensitivity routines directly.
- NLBVP: solve the nonlinear problem, then call `compute_sensitivities(...)`.
- IVP: wrap the solver loop in a direct-adjoint loop with explicit checkpoint
  management.

This is a discrete adjoint of the actual solver, not a separate continuous-adjoint
derivation and not a generic dense Jacobian build.

## Feasibility for VMEC-JAX

### What transfers well

- VMEC-JAX already has a discrete iterative forward solver:
  `solve_fixed_boundary_residual_iter(...)`.
- The optimization controls are low-dimensional boundary DOFs, which is favorable.
- The solver already assembles physically meaningful residual/update quantities.
- The current failure mode is exactly where a solver-faithful discrete adjoint helps:
  lambda/gauge/closure handling is embedded in the primal solver but only partially
  represented in the current implicit derivative layer.

### What does not transfer directly

- Dedalus has symbolic PDE/operator graphs and solver-native sensitivity hooks.
  VMEC-JAX does not.
- The current VMEC-JAX solver is an algorithmic code path with many control-path
  heuristics, masking rules, and update constraints, rather than a clean generic
  sparse-operator framework.
- There is no existing solver-owned `compute_sensitivities(...)` entry point to hook
  into.

### Bottom line

The paper's method is feasible in spirit, but not as a drop-in port.

The right adaptation for VMEC-JAX is:

- build a manual discrete adjoint for the fixed-boundary residual solver,
- replay the actual primal iteration path,
- differentiate each accepted update step against the quantities the solver already
  computes,
- avoid the current reduced-state stationarity surrogate as the primary derivative
  relation.

This is likely a better long-term direction than continuing to patch the current
implicit active-state machinery.

## Additional sources consulted

### Code sources

- Dedalus adjoint branch:
  - `dedalus/extras/adjoint.py`
  - `dedalus/core/solvers.py`
- Dedalus adjoint examples:
  - `nlbvp_lane_emden/lane_emden.py`
  - `ivp_optimal_mixing/optimal_mixing.py`
  - `evp_mathieu/mathieu_evp_adj.py`
- Existing `vmec_jax` implementation:
  - `vmec_jax/solve.py`
  - `vmec_jax/implicit.py`
  - `vmec_jax/driver.py`
  - `docs/algorithms.rst`
  - `docs/equations.rst`
  - `tests/test_implicit_helpers.py`
  - `tests/test_driver_api.py`

### Literature / equations sources

- Skene & Burns, `Fast automated adjoints for spectral PDE solvers`,
  `arXiv:2506.14792`.
- VMEC-JAX local equations docs in `docs/equations.rst`.
- VMEC-JAX local algorithm/control-law docs in `docs/algorithms.rst`.
- SIMSOPT QH objective equations in `simsopt/src/simsopt/mhd/vmec_diagnostics.py`.

Practical consequence:

- the differentiated object must be the accepted solver trajectory,
- the backward pass must reuse solver-native structure,
- local Taylor/VJP checks must exist below the full optimization level,
- runtime tracking must happen at each layer, not only in the top-level script.

## Why this is likely better than the current path

The current path tries to infer sensitivities from an auxiliary implicit relation in
`implicit.py`. That has repeatedly failed around:

- lambda branch selection,
- reduced active-state packing,
- masked constraints and gauge handling,
- mismatch between the actual primal iteration logic and the differentiated relation.

A solver-faithful discrete adjoint should instead inherit:

- the exact branch decisions used in the accepted primal trajectory,
- the actual fixed-boundary enforcement used by the solver,
- the same closure/gauge rules that make the primal solve behave,
- a backward cost proportional to a small multiple of the forward operator work,
  rather than dense or chunked Jacobian construction around the full state.

## Scope for the first implementation

This branch should target one problem only at first:

- fixed boundary,
- `lasym=False`,
- QH benchmark path,
- CPU only,
- no free-boundary support,
- no attempt to cover every solver mode.

If that path becomes correct and fast, then generalize.

## Code inventory and expected refactor surface

### Files that will definitely matter

- `vmec_jax/solve.py`
  - owns the fixed-boundary residual iteration,
  - must expose a compact accepted-step tape and replay hooks,
  - is the core file for the new discrete-adjoint path.
- `vmec_jax/implicit.py`
  - owns the current fragile derivative path,
  - should become a compatibility layer rather than the place where the new QH
    derivative logic lives.
- `vmec_jax/driver.py`
  - already controls solver mode selection and staged runs,
  - should expose the new backend cleanly for scripts and tests.
- `vmec_jax/tests/test_implicit_helpers.py`
  - good home for local operator and derivative regression tests.
- `vmec_jax/tests/test_driver_api.py`
  - good home for backend-selection and policy regressions.

### New files likely worth creating

- `vmec_jax/discrete_adjoint.py` or `vmec_jax/residual_adjoint.py`
  - explicit home for tape/replay/VJP code.
- `vmec_jax/tests/test_discrete_adjoint_qh.py`
  - exact 8-DOF QH derivative regression and short runtime gates.
- `vmec_jax/tools/diagnostics/benchmark_discrete_adjoint_qh.py`
  - function-level and short-horizon runtime harness.

### Files that should not be the first place we modify

- free-boundary modules,
- GPU-specific execution paths,
- general `lasym=True` support,
- scan-mode control-flow variants.

## Working implementation strategy

Keep the work split into three layers:

1. **Primal tape layer**
   - expose a deterministic record of accepted fixed-boundary iterations.
2. **Discrete-adjoint layer**
   - define per-step VJPs and reverse replay over that tape.
3. **Public backend layer**
   - integrate the new derivative backend into the existing user-facing API.

Do not mix these layers in the same patch unless a very small compatibility shim
forces it.

## Implementation plan

### Phase 0: preserve a stable forward baseline

- [x] Define and freeze one differentiable primal mode for the QH benchmark.
- [ ] Minimize optional control-path features during the new discrete-adjoint path:
  - no experimental solver fallbacks,
  - no scan-vs-nonscan ambiguity,
  - fixed restart policy,
  - fixed boundary enforcement path.
- [~] Record exactly which forward intermediates are needed for replay.

Acceptance:

- repeated forward solves from the same state/control produce the same accepted update
  history and final state.
- the forward run exports enough structured metadata to explain the accepted
  control-flow decisions.

Validation / runtime gates:

- Gate 0A: exact QH start-point determinism on CPU.
- Gate 0B: one-iteration runtime benchmark for the chosen primal mode.
- Gate 0C: small circular-tokamak forward runtime benchmark.

### Phase 1: add a committed regression harness

- [~] Add a small committed regression that compares AD against central FD on the exact
  8-DOF QH benchmark start point.
- [~] Check:
  - full objective gradient,
  - one or two problematic lambda-internal sensitivities,
  - at least one `R/Z` interior sensitivity.
- [x] Add a small-runtime directional-derivative test around the benchmark start.

Acceptance:

- every adjoint change is judged against the same exact benchmark probe.

Validation / runtime gates:

- Gate 1A: directional derivative test on the exact QH start point.
- Gate 1B: component tests for one lambda, one `Rcos`, and one `Zsin`
  interior sensitivity.
- Gate 1C: finite-difference probe runtime benchmark so the regression remains
  cheap enough to run frequently.

### Phase 2: expose the primal iteration tape

- [~] Refactor `solve_fixed_boundary_residual_iter(...)` so one iteration step has a
  clear functional boundary:
  - current state,
  - preconditioned residual/update quantities,
  - accepted step metadata,
  - next state.
- [~] Store or checkpoint only what the backward replay actually needs.
- [~] Separate diagnostics from adjoint-critical state.

Acceptance:

- the forward solve can return a compact tape/checkpoint object sufficient for replay.

Design requirements:

- tape entries must distinguish:
  - proposed update,
  - accepted update,
  - backtracking/time-step scaling,
  - restart/checkpoint rollback,
  - boundary/axis/lambda enforcement.
- tape data must be split into:
  - adjoint-critical state,
  - diagnostics only.

Validation / runtime gates:

- Gate 2A: replay one accepted step and recover the exact same next state.
- Gate 2B: replay a small multi-step case and recover the exact final state.
- Gate 2C: tape size and tape-creation wall-time benchmark.

### Phase 3: implement per-step VJPs

- [ ] Identify the operator blocks in one accepted update:
  - residual/force assembly,
  - preconditioner application,
  - lambda update pieces,
  - fixed-boundary/axis/gauge enforcement,
  - acceptance/backtracking scaling.
- [ ] Write VJPs for these blocks against the exact discrete operations, not a
  surrogate reduced state.
- [ ] Keep the first implementation explicit and narrow rather than overly generic.

Acceptance:

- one accepted forward step passes local Taylor or VJP consistency tests.

Implementation note:

The first per-step VJP does not need to be fully symbolic. It does need to be
solver-faithful. Freezing branch decisions from the accepted primal step is
acceptable for the first version.

Validation / runtime gates:

- Gate 3A: local VJP identity check for each operator block.
- Gate 3B: one-step Taylor test for the composed accepted update.
- Gate 3C: one-step backward runtime benchmark broken down by block.

### Phase 4: reverse the accepted iteration history

- [ ] Build a backward replay over the accepted primal tape.
- [ ] Accumulate cotangents from final state to initial state and boundary controls.
- [ ] Treat nondifferentiable control-flow decisions as frozen from the accepted primal
  trajectory for the first implementation.

Acceptance:

- full reverse-mode gradient on the exact QH start point is in the same ballpark as
  central finite differences.

Validation / runtime gates:

- Gate 4A: one-iteration reverse gradient vs FD.
- Gate 4B: two-iteration reverse gradient vs FD.
- Gate 4C: exact QH start-point gradient vs FD on the full converged primal solve.
- Gate 4D: backward/forward runtime ratio benchmark on:
  - a one-iteration small case,
  - the exact QH start-point solve.

Targets:

- no single lambda component remains an orders-of-magnitude outlier,
- backward runtime is clearly below the current dense/chunked Jacobian path.

### Phase 5: integrate into the public wrapper

- [ ] Add a dedicated derivative backend in `vmec_jax` for the discrete-adjoint
  residual path.
- [ ] Keep the current implicit path available behind an explicit option until the new
  path is validated.
- [ ] Make the new path the preferred backend only after correctness is established.

Acceptance:

- `simsopt` can call the new path without local hacks.

Validation / runtime gates:

- Gate 5A: public API can request the new backend explicitly.
- Gate 5B: existing non-adjoint solver paths remain unchanged.
- Gate 5C: backend-selection overhead is negligible relative to solve time.

### Phase 6: wire the new path into the QH example

- [ ] Draft a new `QH_fixed_resolution_jax.py` in `simsopt` that mirrors the classic
  example structurally.
- [ ] Use autodiff from the new discrete-adjoint path only.
- [ ] Benchmark against the exact classic reference after every material change.

Acceptance:

- the example materially improves QH quasisymmetry under the exact 10-evaluation
  benchmark and moves toward runtime parity.

Validation / runtime gates:

- Gate 6A: `max_nfev=1` script smoke benchmark.
- Gate 6B: `max_nfev=2` exact QH benchmark.
- Gate 6C: `max_nfev=10` full apples-to-apples benchmark.

Per-run metrics to record every time:

- total wall time,
- objective call count,
- derivative call count,
- wall time per objective call,
- wall time per derivative call,
- final total objective,
- final QS objective,
- final aspect.

## Runtime and profiling policy

Every new function added for the discrete-adjoint path must be benchmarked in
isolation before it is trusted in the end-to-end script.

### Required microbench harnesses

- accepted-step forward replay cost,
- one-step VJP cost,
- full reverse replay cost,
- tape creation cost,
- tape replay memory footprint.

### Required macrobench harnesses

- exact QH start-point forward solve,
- exact QH start-point gradient,
- `max_nfev=2` QH optimization,
- `max_nfev=10` QH optimization.

### Performance policy

- do not wait until the 10-evaluation script to discover a 10x slowdown,
- do not add dense Jacobian construction to the new path except as a temporary
  diagnostic,
- keep benchmarking on CPU first,
- treat backward/forward wall-time ratio as a first-class metric.

## Coupling plan with simsopt

Clean `simsopt` mainline does **not** yet include the `VmecJax` wrapper or the
JAX least-squares solver surface used in the previous experimental branch. That
means this branch must deliver a backend that is stable enough to justify
reintroducing that consumer layer.

Recommended sequencing:

1. finish Phases 1-4 here in `vmec_jax`,
2. only then port the minimum `VmecJax` consumer layer into clean `simsopt`,
3. only then resume end-to-end optimizer benchmarking.

## Risks

### High-risk items

- The accepted primal update path contains heuristics that may be awkward to
  differentiate exactly.
- A naive tape may be too large if we store everything.
- Some per-step operations may still require targeted analytic VJPs or custom JAX
  primitives to be efficient.

### Mitigations

- Start with fixed-boundary `lasym=False` only.
- Freeze control-flow decisions from the primal tape in the first version.
- Prefer replay/checkpointing over storing large dense intermediates.
- Validate with directional derivatives before optimizing runtime.

## Go / no-go criteria

Continue this branch if, after Phases 1 to 4:

- full QH gradient mismatch drops to a small multiple of FD,
- lambda sensitivities stop being outliers,
- backward cost is clearly better than the current dense/chunked Jacobian path.

Stop or reduce scope if:

- the per-step adjoint requires effectively re-deriving too much of VMEC by hand, or
- the replay path is not simpler than a corrected root-based implicit derivative.

## Immediate next tasks

- [~] Add the exact QH derivative regression on this clean branch.
- [~] Refactor the forward residual solver to expose a compact accepted-step tape.
- [ ] Prototype a one-step discrete VJP for the fixed-boundary residual update.
- [~] Create a small runtime harness for step/tape/VJP costs before any large
  consumer-side refactor.

## Activity log

- 2026-04-14:
  - Added an exact QH warm-start fixture and QH-specific regression file.
  - Locked Gate 1A for the current implicit path: the QH aspect directional derivative at `max_iter=1` matches central FD to machine precision.
  - Confirmed the lambda branch is still the main local derivative outlier on the exact QH start point.
  - Added `ResidualIterationTrace` plus extraction from the existing residual-solver diagnostics.
  - Added `ResidualCheckpointTape` built from repeated one-step replay using `resume_state`.
  - Validated Gate 2A and Gate 2B: one-step and two-step QH checkpoint replay recover the exact direct-solve final state.
  - Established an important constraint for the next phase: exact multi-step replay currently requires `resume_state_mode='full'`; the existing minimal checkpoint is not sufficient once cached VMEC control/preconditioner state is reused.
  - Added a QH diagnostic script that reports aspect/lambda AD-vs-FD plus direct-vs-replay runtime and final-state agreement.
  - Added an explicit `replay_residual_checkpoint_step(...)` primitive and validated it against the second direct QH step from a full resume checkpoint.
  - Implemented the first real per-step reverse target: the strict-update state-advance block used by the QH benchmark path.
  - Locked the first Phase 3 gates on exact QH one-step data:
    - block reconstruction matches the solver state exactly when evaluated on the JAX path,
    - local JVP/VJP identity passes,
    - one-step Taylor remainder passes.
  - Measured the standalone strict-update state-advance block runtime on exact QH one-step data at about `4.3e-2 s`, well below the full one-step direct solve cost.
  - Added solver-side `adjoint_trace` capture for the accepted strict-update branch so replay tape entries now include solver-faithful pre/post velocity blocks, force blocks, and step metadata from the exact QH primal step.
  - Extended `ResidualCheckpointTape` to keep per-step strict-update traces alongside replay checkpoints.
  - Implemented the strict-update velocity recurrence as a separate block and locked its first local Phase 3 gates on exact QH one-step data:
    - velocity block reconstruction matches the solver trace exactly,
    - local JVP/VJP identity passes.
  - Re-ran the exact one-step QH diagnostic harness after the trace refactor:
    - aspect AD vs FD remains at machine precision,
    - lambda scalar mismatch remains the primary local derivative defect,
    - replay/direct final state agreement remains exact,
    - checkpoint replay overhead remains negligible relative to direct one-step solve time.
  - Composed the first solver-faithful accepted-step helper over the saved strict-update QH trace and locked new local Phase 3 gates:
    - accepted-step reconstruction matches the solver state and post-update velocities exactly,
    - local JVP/VJP identity passes for the composed accepted-step map,
    - one-step Taylor remainder passes for the composed accepted-step map.
  - Tightened the accepted-step map to include the velocity RMS limiter and the VMEC lambda-gauge enforcement that happens after fixed-boundary/axis projection in the primal solver.
  - Added a direct local test for the velocity RMS limiter on an active clipped branch so that block is not only validated on the inactive QH first-step case.
  - Re-benchmarked the exact QH one-step composed accepted-step helper:
    - accepted-step block runtime is about `1.29e-1 s`,
    - accepted-step block state reconstruction remains exact,
    - lambda scalar mismatch is unchanged, confirming the remaining defect is still upstream of the accepted-step map.
  - Added the next exact upstream block for the QH path: preconditioned force channels from the stored `rz_preconditioner_apply` output plus lambda/mode scaling.
  - Locked new local gates on exact QH one-step data for that block:
    - `frzl_rz -> fr*_u` reconstruction matches the solver trace exactly,
    - local JVP/VJP identity passes,
    - composing `frzl_rz -> accepted step` still reconstructs the solver state exactly.
  - Added the next upstream block for the current QH branch assumptions (`lasym=False`): raw Fourier residuals through `rz_preconditioner_apply` into the solver force channels.
  - Locked new local gates on exact QH one-step data for that block:
    - `frzl -> fr*_u` reconstruction matches the solver trace exactly,
    - local JVP/VJP identity passes.
  - Re-benchmarked the exact QH one-step block runtimes:
    - `frzl -> fr*_u` runtime is about `2.14e-2 s`,
    - `frzl_rz -> fr*_u` runtime is about `6.0e-4 s`,
    - `frzl_rz -> accepted step` remains exact with no change to the lambda mismatch diagnostic.
  - This leaves the remaining one-step upstream seam at the raw-force assembly itself (`_compute_forces_iter` / `vmec_forces_rz_from_wout` + `vmec_residual_internal_from_kernels` + post-`tomnsps` VMEC cleanup), which is now the only part of the first-step map not yet extracted into solver-faithful discrete-adjoint blocks.
  - Extracted the raw-force assembly into `raw_force_residual_from_state(...)` and verified that the full one-step composition
    `state -> raw force -> state-dependent preconditioner -> preconditioned channels -> accepted strict update`
    reconstructs the exact first QH step on the current branch.
  - Isolated a real algebra bug in the raw preconditioner helper: the `scale_m1_par` rewrite on the RHS must be applied before
    `rz_preconditioner_apply_jit`; once fixed, `frzl -> fr*_u` and `raw force -> accepted step` both matched the traced QH first step.
  - Probed boundary derivatives through the extracted one-step map and found that the apparent preconditioner AD failure was upstream:
    traced `initial_guess_from_boundary(..., vmec_project=True)` was silently disabling missing-axis inference, so AD and FD were
    differentiating different initialization branches.
  - Added `extract_axis_override_from_state(...)` and an `axis_override=` path in `initial_guess_from_boundary(...)` so the
    initialization branch can be frozen explicitly for differentiated replay paths.
  - Locked two new exact QH gates with that frozen-axis path:
    - projected-initial-guess boundary derivative matches central FD,
    - extracted one-step boundary derivative matches central FD.
  - Added the first tape-level reverse helper, `checkpoint_tape_state_vjp(...)`, which chains VJPs backward through the extracted
    strict-update step map over a stored residual checkpoint tape.
  - Locked the corresponding two-step exact QH gate: the chained tape reverse pass matches a direct two-step JAX VJP through the
    extracted step map.
  - Added the first control-level reverse helper, `checkpoint_tape_param_vjp(...)`, which composes the frozen-axis projected initial
    guess with the tape-level reverse pass to propagate cotangents back to boundary parameters.
  - Locked the corresponding two-step exact QH control gate: the replay-based parameter VJP matches a direct two-step JAX VJP
    through the same frozen-axis extracted map for the `(m,n)=(0,1)` boundary coefficient.
  - Added the first scalar objective gate on top of that control path: a two-step QH aspect gradient propagated with
    `checkpoint_tape_param_vjp(...)` matches both direct AD through the extracted map and central finite differences.
  - The current root cause is therefore no longer in the accepted-step algebra or the raw-force assembly; it was the
    initialization branch mismatch caused by traced missing-axis handling. The next phase should thread the frozen-axis branch choice
    through the replay/tape consumer path and then start the reverse-over-history implementation on top of that consistent primal map.
  - Added the matching forward helpers, `checkpoint_tape_state_jvp(...)` and
    `checkpoint_tape_param_jvp(...)`, and locked exact-QH two-step gates showing
    they match direct JAX JVPs through the extracted replay map.
  - This established that the discrete-adjoint tape now supports both reverse
    and forward sensitivity transport over the validated replay path, which is
    the right interface for the `m >> n` QH least-squares consumer in
    `simsopt`.
  - Identified a new JIT blocker in the residual solver while exercising that
    forward path from `simsopt`: a NumPy-only `ptau` cache precompute was still
    converting traced `s` arrays with `np.asarray(...)`.
  - Began removing that blocker by making the `ptau` cache tracer-safe so the
    forward discrete-adjoint backend can participate in `jacfwd`-driven SciPy
    least-squares solves.
  - Continued hardening the traced residual path by removing additional
    unconditional host-only caches (`mn` index arrays, `scalxc_mn`, mode-diag
    weights) from the traced setup path.
  - This moved the `jit=True` failure mode forward from traced NumPy
    conversions to a real JAX issue: `_rz_norm(...)` still uses boolean
    indexing with a traced mask, which must be rewritten before the forward
    discrete-adjoint SciPy path can run under JIT on CPU.
  - Rewrote the traced `_rz_norm(...)` gathers to use clipped gathers plus
    masking instead of boolean indexing, which removed that JAX error class.
  - After that fix, the remaining `jit=True` blockers are no longer localized
    cache conversions; they are host scalar bookkeeping paths in
    `solve_fixed_boundary_residual_iter(...)` (`int(jmax)`, `_device_get_floats`,
    VMEC-style scalar histories / diagnostics) that need to be separated from
    the traced solve path more systematically.
  - Added `checkpoint_tape_state_jvp_columns(...)`, which reuses a single
    per-step linearization and propagates multiple packed-state tangents
    through the same replay tape with `vmap(...)` instead of retracing the
    same step map column by column.
  - Locked a new exact-QH slow regression showing the batched helper matches
    stacked single-column `checkpoint_tape_state_jvp(...)` results on the same
    2-step replay tape.
  - Began the next replay-runtime cut: a scan-backed replay-column path for the
    stored-preconditioner branch, so the tape can run over stacked step traces
    instead of a Python loop of per-step closures.
  - To support that path, registered two replay payload containers as pytrees:
    `VmecTrigTables` and `_WoutLikeVmecForces`.
  - Added a cheap circular-tokamak regression alongside the existing exact-QH
    one to validate the new replay-column path.
  - The scan-backed path now works for the stored-preconditioner branch, but
    the rebuilt-preconditioner branch still hits traced host-concretization
    sites (`int(jmax)` and related static-arg expectations in the 1D
    preconditioner apply path).
  - Production was kept stable on purpose:
    `rebuild_preconditioner=True` still uses the proven per-step replay loop,
    while the scan-backed implementation is currently restricted to the
    stored-preconditioner branch until those rebuilt-preconditioner
    concretization sites are cleaned up.
  - Recovered the next replay bottleneck without widening the architecture:
    the tape already records `precond_jmax`, and on the fixed-boundary replay
    path it is constant across the tape. Threaded that value back into the
    rebuilt-preconditioner replay as a static override instead of trying to
    make the whole 1D preconditioner stack dynamically indexed.
  - Updated `state_dependent_preconditioner_from_forces(...)` /
    `strict_update_one_step_from_state(...)` to accept a
    `preconditioner_jmax_override`, and threaded the recorded `precond_jmax`
    through replay JVP/VJP helpers.
  - Re-enabled the scan-backed batched replay path for
    `rebuild_preconditioner=True` whenever `precond_jmax` is constant across
    the tape, with a safe fallback to the old per-step loop only for tapes
    that genuinely vary `jmax`.
  - The first follow-on concretization on that path was
    `float(lambda_update_scale)` inside
    `preconditioned_force_channels_from_rz_output(...)`; removed that host
    conversion by keeping the lambda scaling purely in JAX arithmetic.
  - Added a new exact-QH slow regression showing the rebuilt-preconditioner
    batched replay path matches stacked single-column replay JVPs.
  - Validated that the production `simsopt` microcase still gives the same
    numerical result while dropping wall time from about `18.98 s` to
    `16.93 s` on:
    `QH_fixed_resolution_jax.py --max-mode 1 --max-nfev 2 --vmec-max-iter 1
    --method scipy --jac jax --residual-derivative-backend discrete_adjoint
    --jit`.
  - The next wrapper-side optimization exposed a real traced NumPy path in the
    Boozer/JXBFORCE low-pass helper used by QS residuals:
    `_jxbforce_nyquist_limits_from_trig(...)` was converting traced trig arrays
    with `np.asarray(...)`, and the same path also used `float(trig.r0scale)`.
  - Made that QS/Boozer helper tracer-safe by switching those scalars/arrays to
    `jnp.asarray(...)`, which unblocked JIT-compiling the residual-side
    Jacobian block in the `simsopt` discrete-adjoint wrapper.
  - Fresh-main comparison checkpoint:
    - cloned `vmec_jax` main to `/Users/rogeriojorge/local/vmec_jax_main_fresh`
      and installed it into an isolated comparison environment.
  - Direct fixed-boundary forward-solve timings show no meaningful runtime
    regression in the core `vmec_jax` branch relative to main:
    - QA `max_iter=1`: branch `2.454 s`, main `2.429 s`
    - QA `max_iter=20`: branch `0.334 s`, main `0.326 s`
    - QH `max_iter=1`: branch `1.963 s`, main `1.912 s`
    - QH `max_iter=20`: branch `0.238 s`, main `0.228 s`
  - Full QA forward solve also matches main closely:
    - branch `3.733 s`, main `3.622 s`,
    - both stop after `113` iterations with `fsqz_last ≈ 4.39e-12`.
  - Conclusion from this comparison:
    - the current major runtime cost is not a low-level forward-solver slowdown
      introduced in the `vmec_jax` branch,
    - the remaining bottleneck sits above the core solver, in the
      `simsopt`-side objective / Jacobian / reporting path.
  - Returned to the exact full-inner QH bottleneck and found the next real
    production issue: the wrapper was still building the discrete-adjoint tape
    by replaying `solve_fixed_boundary_residual_iter(max_iter=1)` in a Python
    loop, one solve call per accepted step.
  - Added `build_residual_checkpoint_tape_direct(...)`, which runs one full
    residual solve with `adjoint_trace=True` and extracts the complete
    `adjoint_step_trace` history in one pass instead of chunked replay.
  - Also added lean tape storage controls to the older replay builder:
    - optional suppression of stored `packed_states`,
    - optional suppression of scalar trace history,
    - optional suppression of saved `resume_states`,
    - explicit `final_packed_state` on the tape so consumers do not need a
      stacked checkpoint history just to recover the final state.
  - Added exact-QH slow regressions showing:
    - the replay builder can skip debug storage and still preserve the exact
      final state,
    - the new direct builder matches the replay-built 2-step QH tape on final
      state and step-trace count.
  - Most important production result:
    the exact full-inner QH SciPy path no longer dies in a compile storm before
    the first iteration. With the new direct-tape path:
    - `QH_fixed_resolution_jax.py --max-mode 1 --max-nfev 1 --timings
      --method scipy --jac jax --residual-derivative-backend discrete_adjoint
      --jit`
      now reaches and finishes iteration 0 with solve wall time about
      `16.58 s`, instead of exiting before the first SciPy iteration.
  - Profiled the exact path again and found the next concrete Jacobian
    bottleneck was not the replay math itself but rebuilding the stacked replay
    tape from Python step dictionaries on every call.
  - Refactored `ResidualCheckpointTape` to cache a NumPy-stacked replay trace
    and its static flags once at tape-construction time, then reuse that cached
    structure in `checkpoint_tape_state_jvp_columns(...)`.
  - Exact-QH component timings after that refactor:
    - `x0` exact solve about `8.58 s`, exact Jacobian about `6.55 s`
      (previously about `7.72 s` + `7.64 s`);
    - nearby `x1` exact solve about `6.35 s`, exact Jacobian about `6.12 s`
      (previously about `5.61 s` + `8.54 s`).
  - Same-process exact SciPy callback timings improved as well:
    - `x0_cold`: `15.26 s -> 13.66 s`
    - `x0_warm`: `11.39 s -> 10.11 s`
    - nearby `x1_warm`: `12.62 s -> 10.68 s`
  - End-to-end exact QH benchmark improvement on the apples-to-apples
    `max_mode=1`, `max_nfev=2` probe:
    - objective unchanged at `0.2651202937448159`
    - solve wall time improved from about `29.73 s` to about `24.51 s`.
