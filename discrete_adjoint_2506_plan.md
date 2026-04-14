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

## Implementation plan

### Phase 0: preserve a stable forward baseline

- [ ] Define and freeze one differentiable primal mode for the QH benchmark.
- [ ] Minimize optional control-path features during the new discrete-adjoint path:
  - no experimental solver fallbacks,
  - no scan-vs-nonscan ambiguity,
  - fixed restart policy,
  - fixed boundary enforcement path.
- [ ] Record exactly which forward intermediates are needed for replay.

Acceptance:

- repeated forward solves from the same state/control produce the same accepted update
  history and final state.

### Phase 1: add a committed regression harness

- [ ] Add a small committed regression that compares AD against central FD on the exact
  8-DOF QH benchmark start point.
- [ ] Check:
  - full objective gradient,
  - one or two problematic lambda-internal sensitivities,
  - at least one `R/Z` interior sensitivity.
- [ ] Add a small-runtime directional-derivative test around the benchmark start.

Acceptance:

- every adjoint change is judged against the same exact benchmark probe.

### Phase 2: expose the primal iteration tape

- [ ] Refactor `solve_fixed_boundary_residual_iter(...)` so one iteration step has a
  clear functional boundary:
  - current state,
  - preconditioned residual/update quantities,
  - accepted step metadata,
  - next state.
- [ ] Store or checkpoint only what the backward replay actually needs.
- [ ] Separate diagnostics from adjoint-critical state.

Acceptance:

- the forward solve can return a compact tape/checkpoint object sufficient for replay.

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

### Phase 4: reverse the accepted iteration history

- [ ] Build a backward replay over the accepted primal tape.
- [ ] Accumulate cotangents from final state to initial state and boundary controls.
- [ ] Treat nondifferentiable control-flow decisions as frozen from the accepted primal
  trajectory for the first implementation.

Acceptance:

- full reverse-mode gradient on the exact QH start point is in the same ballpark as
  central finite differences.

### Phase 5: integrate into the public wrapper

- [ ] Add a dedicated derivative backend in `vmec_jax` for the discrete-adjoint
  residual path.
- [ ] Keep the current implicit path available behind an explicit option until the new
  path is validated.
- [ ] Make the new path the preferred backend only after correctness is established.

Acceptance:

- `simsopt` can call the new path without local hacks.

### Phase 6: wire the new path into the QH example

- [ ] Draft a new `QH_fixed_resolution_jax.py` in `simsopt` that mirrors the classic
  example structurally.
- [ ] Use autodiff from the new discrete-adjoint path only.
- [ ] Benchmark against the exact classic reference after every material change.

Acceptance:

- the example materially improves QH quasisymmetry under the exact 10-evaluation
  benchmark and moves toward runtime parity.

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

- [ ] Add the exact QH derivative regression on this clean branch.
- [ ] Refactor the forward residual solver to expose a compact accepted-step tape.
- [ ] Prototype a one-step discrete VJP for the fixed-boundary residual update.
