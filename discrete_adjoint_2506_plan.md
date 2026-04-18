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
  - Traced the next remaining compile leak in the exact Jacobian path and found
    it was no longer the helper-column functions, but the scan transport
    closure itself: the replay-column path was recompiling `_run_scan` for the
    same `786`-step tape because the scan body was being rebuilt on each call.
  - Added a dedicated scan-runner cache in `checkpoint_tape_state_jvp_columns`
    keyed by the static flags and stacked-trace signature, so the jitted
    transport closure is reused across exact Jacobian calls with the same tape
    shape.
  - Exact replay regressions remain green on the rebuilt-preconditioner path.
  - Compile logging now shows `_run_scan` compiles once for `786` steps and
    once for `792` steps across the `x0 / x0 / x1` probe, instead of compiling
    the `786`-step case twice.
  - Updated exact callback timings after the scan-runner cache:
    - `x0a`: solve about `7.95 s`, jac about `4.29 s`
    - `x0b`: solve about `5.61 s`, jac about `2.64 s`
    - `x1`: solve about `5.64 s`, jac about `3.48 s`
  - Profiled the current pushed exact path again with `cProfile` on the exact QH
    start point after the callback/runtime fixes:
    - warm forward solve and warm discrete-adjoint primal solve are both still
      dominated by the full-inner force loop, not by replay:
      about `8.9 s` total, with about `7.1 s` inside
      `compute_forces_numpy(...)`.
    - the biggest force-kernel pieces are still
      `vmec_forces_rz_from_wout(...)` and
      `vmec_bcovar_half_mesh_from_wout(...)`.
  - Broke the warm exact Jacobian into its three stages and found:
    - initial-state columns are negligible (`~4e-4 s`);
    - replay-column transport dominates (`~2.63 s`);
    - residual-side tangent propagation is negligible (`~8e-3 s`);
    - final host materialization is negligible.
  - Reduced one remaining replay-cache overhead by changing
    `_stacked_trace_signature(...)` to read leaf `shape`/`dtype` directly
    instead of materializing every stacked leaf with `np.asarray(...)`.
    This cuts the Jacobian-side `np.asarray` calls from about `152` down to
    about `6` on the warm exact QH probe.
  - Fixed the `host_update_assembly` control path in
    `solve_fixed_boundary_residual_iter(...)` so `host_update_assembly=False`
    now really disables the NumPy auto-host path instead of being overridden by
    `_auto_host`. Default behavior is unchanged when the argument is omitted.
  - Used that explicit override to compare the exact QH warm primal solve on
    the true NumPy-host path versus the true warmed JAX force path:
    - NumPy-host path: about `5.56 s`, `786` iterations,
      `fsqz_last ≈ 2.24e-14`
    - JAX force path: about `8.10 s`, `786` iterations,
      `fsqz_last ≈ 2.24e-14`
  - Conclusion from that audit:
    on this exact QH CPU path, the NumPy-host force loop is still faster than
    the warmed JAX force kernels, so the current production default should stay
    on the NumPy-host path unless later kernel work changes that result.
  - Ran a direct production-path derivative audit on the benchmark QH setup and
    closed one major ambiguity:
    the direct-tape state is not drifting away from the traced replay map.
    On the exact benchmark wrapper path, the returned state, the tape's
    `final_packed_state`, and the last `adjoint_step_trace["state_post"]`
    agree exactly.
  - The apparent AD-vs-FD mismatch comes from a different place:
    the discrete-adjoint Jacobian linearizes a frozen-axis local map, while the
    plain value callback re-seeds `axis_override` at perturbed points.
    When finite differences are taken through the frozen-axis local map, the
    first Jacobian column matches central FD closely on the benchmark QH start;
    the large mismatch only appears against the moving-axis callback.
  - A deeper full-inner audit narrowed the remaining bug further:
    once `max_iter` is large, moving-axis FD and frozen-axis FD become almost
    the same, and both disagree with the production Jacobian.
  - Frozen-axis iteration sweep on the benchmark QH setup:
    - `max_iter=1`: relative column error about `3.1e-4`
    - `max_iter=2`: about `9.1e-4`
    - `max_iter=5`: about `4.3e-3`
    - `max_iter=10`: about `1.5e-2`
    - `max_iter=20`: about `6.5e-2`
    - `max_iter=50`: about `5.7e-1`
    - `max_iter=100`: about `4.85`
  - Exact QH trace audit at `max_iter=100` shows:
    - all 100 steps are `momentum_accept`
    - no restarts occur
    - `time_step` stays fixed at `0.9`
    - but `dt_eff`, `b1`, `fac`, and `force_scale` all vary at every step
  - That shifts the main derivative hypothesis again:
    the remaining long-horizon drift is likely from frozen solver-control
    history (`dt_eff` / damping-control scalars and the residual-history state
    that updates them), not from replaying the wrong accepted branch and not
    from the initialization axis branch.
  - I prototyped a velocity-carry replay transport locally. It was not enough
    by itself to fix the long-horizon drift, so it was not kept on the branch.
  - Full dynamic-carry replay fix implemented and validated on the strict
    no-restart QH production path:
    - trace now records `inv_tau_before`, `fsq_prev_before`,
      `reset_inv_tau`, `limit_dt_from_force`, and
      `max_coeff_delta_rms_pre` for each strict-update step;
    - replay columns now carry packed state, velocity history, `inv_tau`,
      and `fsq_prev`, and recompute `fsq1`, `dt_eff`, `b1`, `fac`, and
      `force_scale` dynamically instead of freezing them from the base trace.
  - During that implementation I found and fixed another concrete mismatch:
    the rebuild path was reconstructing `w_mode_mn` with exponent `-1`,
    while the exact production residual solve uses `mode_diag_exponent=0.0`
    on this path. Reusing the traced `w_mode_mn` removed the first-step force
    mismatch and made the dynamic replay primal-faithful.
  - Exact primal replay check on the benchmark QH path at `max_iter=20` now
    matches the direct tape final state with packed-state `linf` about
    `3.8e-7`.
  - Frozen-axis derivative sweep after the full carry fix:
    - `max_iter=1`: relative column error about `3.0e-8`
    - `max_iter=10`: about `1.5e-6`
    - `max_iter=20`: about `2.6e-5`
    - `max_iter=50`: about `1.15e-3`
  - Added a slow regression locking the dynamic replay primal consistency on a
    20-step QH tape.
  - New traced moving-axis audit:
    the real remaining mismatch was not the frozen-axis replay path alone.
    `initial_guess_from_boundary(..., vmec_project=True)` was still taking a
    traced zero-axis fallback, so the differentiated moving-axis initial-state
    map did not match the primal callback. That is now fixed with a
    JAX-compatible `_recompute_axis_from_state_vmec_jax(...)` path in
    `vmec_jax/init_guess.py`.
  - After that axis fix, the moving-axis initial-state derivative on the exact
    QH start point matches central FD to machine precision again.
  - The production tangent transport now linearizes each dynamic replay step at
    the stored primal carry for that step, instead of linearizing the whole
    multi-step scan around a numerically drifting replayed trajectory.
    This preserves the solver-control carry sensitivity (`inv_tau`, `fsq_prev`,
    momentum history) while avoiding the long-horizon basepoint drift that was
    still contaminating the full-inner Jacobian.
  - Current wrapper-facing QH audits after those fixes:
    - exact start point, moving-axis, `max_iter=1`: production Jacobian now
      matches FD tightly;
    - exact start point, moving-axis, `max_iter=20`: relative column error is
      now about `2.25e-3`, down from the earlier `3.45e-2`.
  - The lean dynamic-only tape path is no longer exact enough to serve as the
    production tangent transport after the stepwise replay upgrade, so the
    production simsopt wrapper is back on full step traces for correctness.
  - Full tape-size audit on the exact QH start point showed the retained tape
    itself is not the 50 GB problem:
    - full step traces are about `94 MB`
    - stacked replay payload is about `71 MB`
    - dynamic initial carry is negligible
    - the large RSS spikes are coming from executable / transient callback
      retention, not raw tape storage alone.
  - The first lean exact scan refactor removed a large amount of compile churn
    and memory, but a later accepted-point audit exposed a new failure mode:
    the exact 10-evaluation QH benchmark ended at a point whose full-inner
    tape contained exactly one `restart_bad_progress` /
    `catastrophic_growth` step after 173 clean momentum steps.
  - Because the previous production transport only supported the all-dynamic
    case, that single restart forced the late-iteration Jacobian back onto the
    older generic long-horizon replay path, and the derivative error exploded
    again there.
  - The replay transport now supports mixed tapes:
    - contiguous momentum-accept segments use the exact stored-basepoint
      dynamic scan;
    - catastrophic restart steps use an exact carry reset map
      (`state` identity, velocity reset, `inv_tau` reset, `fsq_prev` carry).
  - Added a slow wrapper-side restart-point regression on the exact QH
    benchmark branch point.
  - Exact late-point audit on the benchmark QH final iterate after the mixed
    replay fix:
    - tested columns `0..3` now match central FD to about `1e-2` to `3e-2`
      in objective-direction error instead of the earlier `~1e19` failure;
    - the catastrophic late-iteration Jacobian blow-up is gone.
  - Stable direct comparison set generated on 2026-04-16 from the
    simsopt-side harness in
    `/Users/rogeriojorge/local/simsopt_discrete_adjoint/tools/diagnostics/qh_classic_vs_jax_compare.py`:
    - JAX exact `max_mode=1`, 90 s cap: final total `0.24902495986882436`,
      peak RSS `14.93 GB`
    - JAX exact `max_mode=2`, 90 s cap: final total `0.22622233073571194`,
      peak RSS `16.91 GB`
  - That comparison reinforces the current vmec_jax-side diagnosis:
    derivative quality is good enough to leave the critical path for now, while
    the remaining blocker is forward exact solve cost plus executable/memory
    retention on nearby boundary points.
  - Any further vmec_jax change should therefore be justified by one of two
    measurable wins on the new harness:
    1. reduced exact RSS / executable retention on repeated nearby solves, or
    2. lower exact residual solve wall time without giving back the fixed
       late-iteration Jacobian behavior.
  - The draft vmec_jax PR (`uwplasma/vmec_jax#6`) has been closed
    intentionally.
  - Do not reopen any vmec_jax PR until the exact JAX QH benchmark is
    competitive with classic both in runtime and in final quasisymmetry.
  - Consumer-side audit on 2026-04-17:
    - nearby-point continuation through the exact SciPy callback path was
      tested and rejected on the simsopt side because it regressed the QH
      nearby-point callback pair instead of reducing it;
    - that result reinforces the current vmec_jax-side direction: the next
      meaningful win must come from the forward exact solve or replay
      transport itself, not from wrapper-level warm-state plumbing.
  - Merged `origin/main` through `16b9ed0` into
    `codex/discrete-adjoint-2506` on 2026-04-17 and revalidated the
    discrete-adjoint path:
    - `tests/test_discrete_adjoint_qh.py -k 'dynamic_replay_scan_matches_primal_qh_full_inner or checkpoint_tape_state_jvp_columns_matches_single_column_qh_rebuild_preconditioner'`
      still passes;
    - the remaining exact runtime issue is now more explicit on the merged
      branch: nearby QH points still build different dynamic replay lengths
      (`786`, `792`, `794` on the exact mode-1 probe), so the replay scan sees
      new shapes and can retain/compile fresh executables;
    - the next vmec_jax implementation target should therefore be padded or
      bucketed replay traces / base carries so nearby points reuse the same
      dynamic scan executable.
  - Implemented replay bucketing on 2026-04-17:
    - dynamic replay payloads are now padded to a stable bucket size
      (default `32`) with an explicit `active` mask, so nearby exact QH points
      reuse the same scan shape;
    - added a slow regression that checks direct dynamic replay payloads for
      nearby short lengths share the same padded leading dimension;
    - post-fix exact QH nearby-point audit on the simsopt side now shows a
      stable replay bucket length `800` for `x0/x1/x2`, and the jacobian time
      at nearby points drops to about `2.50 s` / `2.37 s` instead of growing
      with tape length;
    - this moves the next blocker further toward the forward primal solve and
      outer optimization behavior, rather than replay-executable churn.
    - downstream audit also showed that `max_mode=2` still has a usable local
      least-squares descent direction from the exact Jacobian, so the current
      mode-2 failure is now more likely an outer-solver policy issue than a
      replay-derivative issue.
  - Exact QH tolerance audit on 2026-04-17:
    - the simsopt-side comparison harness was upgraded so it can run the exact
      JAX path at explicit `ftol/gtol/xtol`, stream iteration snapshots to disk
      during the run, and optionally collect JAX compile logs / profiler traces;
    - the streamed snapshots were necessary because long exact JAX children are
      still being killed by the OS before writing summaries;
    - full profiled runs (`JAX_LOG_COMPILES=1` plus `jax.profiler.start_trace`)
      are too heavy on the exact path today: both `mode=1` and `mode=2`
      children died with return code `-9`, and the trace dirs remained empty
      because the children never reached `stop_trace()`;
    - compile-log inspection from those failed profiled runs shows the replay
      shape problem is not fully solved yet:
      - `mode=1` still recompiles `_run_scan` for stacked lengths
        `192`, `672`, and `768`;
      - `mode=2` still recompiles `_run_scan` for stacked lengths
        `896` and `928`;
      - repeated `scan` / `_ptau_compute_jit` compilations are still present
        late in the run;
    - important memory audit on the vmec_jax side:
      - the stored replay tape is not the `~20 GB` culprit;
      - on the exact `mode=1` start point,
        `dynamic_base_carries_stacked` is only about `47.7 MB` and
        `stacked_step_traces` about `0.02 MB`;
      - therefore the dominant memory spike must come from the live exact
        forward solve / replay JVP / executable construction path, not from
        retaining the serialized tape itself;
    - coarse replay bucketing (`VMEC_JAX_DYNAMIC_REPLAY_BUCKET=1024`) helps but
      is not sufficient:
      - on exact `mode=1`, the accepted trajectory reaches
        `0.2406365` by `nfev_observed=11`, but the child is still killed before
        convergence;
      - the cold coarse-bucket run peaked around `19.24 GB` RSS;
      - a warmed rerun reduced the observed peak to about `17.22 GB`, which is
        strong evidence that compilation/executable construction is still a
        material component of the spike;
      - exact `mode=2` with the same coarse bucket reaches accepted iterates at
        `0.2262223` and `0.2016143`, then times out at 300 s while the outer
        solver is evaluating a bad trial point rather than a new accepted step;
    - rejected local experiment:
      - chunking Jacobian columns at the simsopt wrapper level did not produce
        a clear memory win and was reverted.
    - follow-up streamed audit after the explicit tolerance patch on the
      consumer side:
      - exact `mode=1` with `VMEC_JAX_DYNAMIC_REPLAY_BUCKET=1024` still exits
        before summary/termination, but now reaches accepted iterates
        `0.2605233`, `0.2490250`, `0.2440611`, `0.2417510`, and `0.2406365`
        by `nfev_observed=11`;
      - that latest `mode=1` run observed peak RSS of only about `9.06 GB`,
        much lower than the earlier `17-19 GB` coarse-bucket probes, which
        suggests the warmed executable path is materially cheaper than the cold
        tolerance audit implied;
      - exact `mode=2` on the same merged/warmed path reaches accepted
        iterates `0.2262223`, `0.2016143`, and `0.1987163` by
        `nfev_observed=9`, but still exits before writing a final summary;
      - the latest `mode=2` exact run still spikes to about `23.10 GB` RSS,
        so the remaining long-run memory problem is now more concentrated in
        the higher-dimensional exact path than in the small `mode=1` case;
      - updated diagnosis:
        - dynamic replay bucketing plus warmed executables are sufficient to
          get real exact descent in both `mode=1` and `mode=2`;
        - the remaining vmec_jax-side blocker is long-run executable/live-JVP
          memory retention on exact nearby points, not the earlier replay-tape
          serialization or frozen-control derivative issues.
    - consumer-side large-Jacobian chunking audit on 2026-04-17:
      - the simsopt wrapper was temporarily taught to chunk the exact
        discrete-adjoint Jacobian columns so large control spaces could be
        tested without changing the vmec_jax replay algebra;
      - this is a genuine memory/runtime tradeoff, not a derivative bug:
        derivative regressions stayed green throughout the experiment;
      - exact `mode=2` with chunk size `8` reduced observed peak RSS from
        about `23.10 GB` to about `20.77 GB` but did not extend accepted
        progress beyond the baseline trajectory;
      - exact `mode=2` with chunk size `4` lowered peak RSS further to about
        `19.35 GB`, but only two accepted iterates were reached within the
        300 s wall-clock window;
      - conclusion:
        - the high-dimensional JVP column batch is indeed part of the live
          memory spike;
        - however, coarse consumer-side chunking alone is not enough to make
          the exact `mode=2` path finish, so the default branch behavior should
          stay unchunked while this remains an opt-in debugging knob.
    - vmec_jax-side replay-column chunking audit on 2026-04-17:
      - added an opt-in environment knob,
        `VMEC_JAX_REPLAY_COLUMN_CHUNK`, to chunk the replay-column transport
        inside `checkpoint_tape_state_jvp_columns(...)` itself, so the expensive
        scan/JVP path can be chunked without redoing the outer initial-state and
        residual linearizations in the simsopt wrapper;
      - targeted vmec_jax replay regressions and the simsopt full-inner/moving-axis
        Jacobian regressions remained green after this change;
      - exact `mode=2` with `VMEC_JAX_REPLAY_COLUMN_CHUNK=8`:
        - same accepted iterates `0.2262223 -> 0.2016143`;
        - peak RSS reduced from about `23.10 GB` to about `20.32 GB`;
        - second accepted iterate arrived later (`151.57 s` vs `103.80 s`);
      - exact `mode=2` with `VMEC_JAX_REPLAY_COLUMN_CHUNK=12`:
        - same accepted iterates through `0.1987163` by `nfev_observed=9`;
        - cold run peak RSS dropped further to about `17.81 GB`, but the third
          accepted iterate arrived much later (`237.54 s`);
        - warmed rerun reached the same `0.1987163` by about `167.93 s` with
          peak RSS about `21.56 GB`;
      - conclusion:
        - replay-side chunking is a better lever than wrapper-side chunking for
          the exact `mode=2` memory spike;
        - it is still a runtime/memory tradeoff, so it should remain opt-in for
          now rather than being forced as the branch default.
    - finite-difference vmec_jax audit on 2026-04-17:
      - user asked whether vmec_jax could be used like classic vmec2000 with
        outer finite differences, taking advantage of warm in-process JIT reuse
        after the first forward solve;
      - answer from the audit: yes in principle, but only through a true
        forward-only residual callback; routing finite differences through the
        discrete-adjoint SciPy residual path would wrongly build replay tapes
        and is not a meaningful FD benchmark;
      - the new simsopt teaching script
        `QH_fixed_resolution_jaxfd.py` uses the corrected forward-only setup and
        serial SciPy `2-point` finite differences so perturbed calls can reuse
        the same warm executables;
      - measured `max_mode=1`, `max_nfev=10`, `ftol=gtol=xtol=1e-4` results are
        still poor:
        - cold timed run died before the first accepted iterate after about
          `156.38 s`, with max RSS about `22.69 GB` and peak footprint about
          `49.30 GB`;
        - warm timed rerun still died before the first accepted iterate after
          about `169.01 s`, with max RSS about `19.16 GB` and peak footprint
          about `45.68 GB`;
        - one unbuffered run did reach SciPy iteration 1 / `nfev=3` with cost
          `1.2958e-01`, so the approach can descend, but it is not stable;
      - conclusion:
        - warm-JIT forward solves alone do not make outer finite differences a
          compelling fallback on the current QH path;
        - the exact discrete-adjoint route remains the stronger path to ship.
    - MPI finite-difference vmec_jax fallback audit on 2026-04-17:
      - user asked whether the classic
        `least_squares_mpi_solve` finite-difference workflow could be reused
        with vmec_jax as a minimal swap-in replacement for vmec2000;
      - audit result:
        - not as a direct drop-in, because the classic
          `QuasisymmetryRatioResidual` / `LeastSquaresProblem.from_tuples`
          path expects `Optimizable` parents, while `VmecJax` is not one;
        - the simsopt side now has a tested teaching-script fallback,
          `QH_fixed_resolution_jaxfd_mpi.py`, that wraps vmec_jax in a minimal
          local `Optimizable` shim and uses `QuasisymmetryRatioResidualJax` to
          avoid the Fortran-style parent graph assumptions;
        - on `mpirun -n 2`, that corrected fallback still takes about `75 s`
          just to reach SciPy iteration 0 / `nfev=1`;
      - conclusion:
        - warm vmec_jax forward solves do not make the classic MPI FD workflow
          attractive on the current branch;
        - the remaining effort should stay focused on the exact
          discrete-adjoint path.
    - accelerated-MPI finite-difference follow-up on 2026-04-17:
      - user then asked whether the MPI FD fallback could at least use the
        same warmed accelerated primal path highlighted by the vmec_jax
        runtime docs;
      - the simsopt fallback script was refactored so its forward evaluations
        bypass the conservative wrapper solve and instead use
        `run_fixed_boundary(..., solver_mode="accelerated")` on a rewritten
        input for each boundary point, while caching same-`x` residual/objective
        work inside each rank;
      - measured result on `mpirun -n 2` with `max_nfev=1`:
        - cold accelerated run still did not finish the initial warm-up
          objective within `120 s`;
        - a warmed rerun again did not finish the same initial warm-up
          objective within `120 s`;
      - conclusion:
        - accelerated warm vmec_jax is not enough to rescue the current MPI FD
          QH workflow;
        - the real work should remain on the exact discrete-adjoint path.
    - exact mode-2 replay-memory pass on 2026-04-17:
      - added an automatic replay-column chunking fallback in
        `checkpoint_tape_state_jvp_columns(...)` for large exact Jacobians when
        the user has not set `VMEC_JAX_REPLAY_COLUMN_CHUNK` explicitly;
      - the auto chunk size is computed from the dynamic replay carry size and
        a target live-memory budget
        (`VMEC_JAX_REPLAY_COLUMN_TARGET_MB`, default about `384 MB`);
      - this keeps the current explicit chunk env as the highest-priority
        override, so existing manual experiments still behave the same;
      - together with the simsopt-side preallocated Jacobian assembly, the
        exact `max_mode=2` Jacobian at `x0` now measures about:
        - `30.56 s` wall time,
        - `5.43 GB` max RSS,
        - versus the earlier exact full-run memory spikes above `20 GB`;
      - targeted discrete-adjoint replay regressions remained green;
      - conclusion:
        - this pass materially reduces the live memory term of the exact
          `mode=2` Jacobian path;
        - it does not by itself fix the remaining `mode=2` zero-step behavior,
          which is now clearly an outer-solver issue rather than a replay
          memory/executable issue.
    - exact outer-solver follow-up on 2026-04-18:
      - the resumed audit confirmed that the vmec_jax-side exact Jacobian was
        already good enough for `max_mode=2`; the remaining zero-step failure
        was in the outer SciPy policy, not in replay transport or derivative
        assembly;
      - the main supporting datapoint was the direct dense least-squares step
        at `x0`, which reduced total objective from `0.3002458` to about
        `0.1127546` at `alpha=0.5` using the exact concrete `J`;
      - after the simsopt-side concrete Gauss-Newton path was enabled, the
        same vmec_jax exact residual/Jacobian backend delivered real descent:
        - `max_mode=2`, `max_nfev=2`,
          `VMEC_JAX_DYNAMIC_REPLAY_BUCKET=1024`,
          `VMEC_JAX_REPLAY_COLUMN_CHUNK=12`:
          - final estimated total objective `0.0620440`;
        - `max_mode=1`, `max_nfev=3`,
          `VMEC_JAX_DYNAMIC_REPLAY_BUCKET=1024`:
          - final estimated total objective `0.2321525`;
      - conclusion:
        - vmec_jax derivative quality is no longer the limiting factor for the
          current QH exact runs;
        - the remaining work is to reduce forward/replay memory retention so
          the better optimizer can run longer without exhausting the machine.
    - exact replay/runtime follow-up on 2026-04-18:
      - after the concrete exact Gauss-Newton path started moving, the next
        simsopt-side profile showed that vmec_jax was still paying for too many
        exact residual solves in backtracking;
      - with residual reuse and line-search scale reuse added on the consumer
        side, the vmec_jax exact backend now supports:
        - `max_mode=2`, `nfev=2`: estimated total objective `0.0620440`;
        - `max_mode=2`, `nfev=3`: estimated total objective `0.0603123`;
      - the early objective quality now beats the classic user-supplied
        `max_mode=2`, `nfev=3` reference (`~0.102316`), which strongly suggests
        the vmec_jax exact derivatives and replay transport are no longer the
        quality bottleneck for this regime;
      - the live remaining problem is pure resource behavior:
        - `max_mode=2`, `nfev=3` still reaches about `22.47 GB` RSS and about
          `59.04 GB` peak footprint;
        - so the next vmec_jax-side target remains executable/array retention
          on repeated nearby exact solves, not another derivative rewrite.
    - exact `jit_forces` follow-up on 2026-04-18:
      - the user-facing wrapper path was then checked against the earlier raw
        vmec_jax primal audit, which had suggested `jit_forces=True` could be
        materially faster on warm CPU solves;
      - the wrapper did not previously expose that knob, even though the exact
        discrete-adjoint path always flows through force-heavy residual solves;
      - after exposing `jit_forces` on the simsopt side, the exact
        `max_mode=2`, `nfev=1` Gauss-Newton benchmark showed:
        - `jit_forces=False`: about `102.87 s`
        - `jit_forces=True`: about `60.80 s`
        - `jit_forces="auto"`: about `62.67 s`
        - same objective in all three cases;
      - conclusion:
        - the earlier vmec_jax force-kernel audit does transfer to the exact
          optimization path;
        - explicit or automatic force-kernel JIT should stay enabled on the
          QH exact route while the longer-horizon memory problem is being
          reduced.
    - forward-only line-search follow-up on 2026-04-18:
      - the next simsopt-side optimization profile showed that vmec_jax was
        still paying exact replay-tape construction costs on pure line-search
        trial points, even though those points only needed forward residuals;
      - after exposing a forward-only residual callback from the objective
        stage and routing the concrete exact GN line search through the normal
        forward solve path, the vmec_jax backend delivered the same objective
        with materially less memory pressure:
        - exact `max_mode=2`, `nfev=2`:
          - same estimated total objective `0.0620440`,
          - max RSS reduced to about `13.47 GB`,
          - line-search wall time reduced substantially;
        - exact `max_mode=2`, `nfev=3`:
          - streamed descent still reached the third accepted step,
          - max RSS reduced to about `17.50 GB` before late-process failure;
      - conclusion:
        - replay/exact derivative quality is still not the bottleneck;
        - the remaining problem is long-run forward-path retention after the
          good optimizer step is already found.
