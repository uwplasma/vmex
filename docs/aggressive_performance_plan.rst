Aggressive Performance Plan
===========================

This page defines the non-parity performance track for ``vmec-jax``.

The goal of this track is different from the VMEC2000-parity track:

- keep the solver end-to-end differentiable,
- preserve accurate final equilibria,
- allow the iteration path to differ from VMEC2000,
- optimize for runtime and memory on both CPU and GPU,
- keep one general policy that works across fixed/free boundary,
  axisymmetric/non-axisymmetric, and ``lasym=False/True`` cases.

In other words, this track accepts "reach the same good equilibrium faster"
instead of "follow VMEC2000 iteration-by-iteration".

Target product contract
-----------------------

This track should ship as an explicit solver mode, not as a silent change to
the strict-parity path.

Acceptance criteria:

- final ``fsq_total = fsqr + fsqz + fsql`` is small for every bundled example,
  with the default target ``<= max(ftol, 1e-8)`` unless the validated parity
  baseline for that case converges to a higher floor,
- final ``wout`` fields match VMEC2000 at about ``rtol=1e-2`` on
  well-conditioned channels, with the same near-axis and near-zero caveats
  already used in the parity docs,
- final states have no unacceptable Jacobian sign issues,
- gradients remain available through the accelerated solve or through an
  implicit-differentiation wrapper around the converged fixed point,
- the solver policy is data-driven and general, not case-ID based.

Current bottleneck summary
--------------------------

The current code already shows where the time goes:

- :mod:`vmec_jax.solve` still carries a host-controlled VMEC2000-style loop for
  the conservative path. That loop includes Jacobian checks, same-iteration
  restarts, screen-cadence logic, and free-boundary cadence updates.
- :mod:`vmec_jax.driver` uses conservative scan guards because the product
  contract today is VMEC2000 parity, not merely fast convergence.
- :mod:`vmec_jax.kernels.tomnsp`, :mod:`vmec_jax.kernels.bcovar`,
  :mod:`vmec_jax.kernels.forces`, and :mod:`vmec_jax.kernels.residue` still execute
  as several moderate-size float64 kernels with significant memory traffic.
- :mod:`vmec_jax.free_boundary` adds dense vacuum coupling, edge-force terms,
  and iteration cadence logic that make the current GPU path especially
  launch-latency sensitive.

The March 2026 benchmark snapshot in :doc:`performance` shows the current
outliers:

- fixed-boundary heavy 3D reactor-scale cases are now the main useful GPU wins,
  but small axisymmetric cases still favor CPU,
- fixed-boundary ``lasym=True`` cases are improved but still not consistently
  GPU-favorable,
- free-boundary ``input.DIII-D_lasym_false`` is the dominant runtime and memory
  outlier on both CPU and GPU,
- the GPU is still slower than the CPU for several moderate-size cases because
  the device is fed many short kernels with host synchronization between them.

Performance doctrine
--------------------

The performance track should follow five rules:

1. Keep a dual-mode architecture

   - ``parity`` mode: preserve VMEC2000-compatible iteration behavior.
   - ``accelerated`` mode: preserve final-equilibrium quality and
     differentiability, but allow different iteration history.

2. Optimize for device residency first

   The biggest wins, especially on GPU, come from keeping long stretches of the
   nonlinear solve on-device and shape-stable.

3. Optimize memory movement before micro-optimizing FLOPs

   The current solver is bottlenecked by transforms, synthesis, tensor
   reshaping, and host/device traffic at least as much as by raw arithmetic.

4. Keep policies general

   Choose fast paths from solver signals, residual trends, conditioning, and
   shape metadata rather than hand-maintained per-example switches.

5. Separate "differentiate through iterations" from "differentiate the final solution"

   For large production solves, implicit differentiation around the converged
   solution should be the default gradient story. Backpropagating through every
   nonlinear iteration should remain available for shorter solves and tests.

Why this is technically plausible
---------------------------------

This direction is consistent with current literature and JAX constraints:

- The VMEC++ work emphasizes modernized numerics, cache-friendly transforms,
  restart handling, and implementation structure focused on performance and
  optimizer workflows rather than strict Fortran control-flow mimicry.
- The DESC continuation/perturbation work shows that staged homotopy and warm
  starts are a practical way to reach difficult equilibria more efficiently.
- The JAX documentation explicitly notes that ``lax.scan`` lowers to a single
  loop operator, while ``lax.while_loop`` is not reverse-mode differentiable by
  default. That pushes this project toward bounded masked scans, custom VJPs,
  or implicit differentiation when we want both speed and gradients.
- JAX's async-dispatch, buffer-donation, profiling, and rematerialization tools
  align well with the type of memory- and orchestration-heavy workload in this
  solver.

Complete execution plan
-----------------------

Phase 0: Measurement and acceptance harness
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before refactoring the solver, expand the benchmark and validation harness.

Deliverables:

- Add an "accelerated-mode" benchmark matrix that records, for each example:

  - cold runtime,
  - warm runtime,
  - compile time,
  - peak RSS / device memory,
  - final ``fsq_total``,
  - convergence flag,
  - final ``wout`` deltas vs VMEC2000 at ``rtol=1e-2`` on tracked channels.

- Split metrics into:

  - fixed boundary vs free boundary,
  - axisymmetric vs non-axisymmetric,
  - ``lasym=False`` vs ``lasym=True``,
  - CPU vs GPU.

- Persist profiler artifacts for the heaviest cases:

  - XLA/JAX traces,
  - kernel timelines,
  - compile logs,
  - memory profiles.

- Add a benchmark dashboard artifact under ``outputs/accelerated_mode/...``.

Why first:

- this branch cannot be judged by parity manifests alone,
- runtime work without a stable acceptance harness will regress accuracy or
  silently shift work from runtime to compile time.

Phase 1: Introduce an explicit accelerated solver mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add a new public mode, for example ``solver_mode="accelerated"``, while keeping
the current strict path intact.

The new mode should:

- target final equilibrium quality instead of per-iteration parity,
- default to device-resident loop execution,
- allow different restart timing and step acceptance from VMEC2000,
- expose a single stable user path through the CLI and Python API.

The mode contract should define:

- convergence target based on ``fsq_total`` and Jacobian health,
- final-output validation target relative to VMEC2000,
- allowed differences in iteration count and residual trajectory.

Phase 2: Replace the host-controlled outer loop with a device-resident controller
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the highest-value refactor.

Current problem:

- the conservative loop in :mod:`vmec_jax.solve` keeps returning control to the
  host for restart decisions, Jacobian checks, and free-boundary cadence.

Accelerated-mode design:

- move the outer solve to a bounded masked ``lax.scan`` where possible,
- keep static-shape carry state for:

  - current state,
  - trust-region or line-search parameters,
  - residual history summaries,
  - preconditioner cache handles,
  - free-boundary reuse counters,
  - convergence flags.

- use masked iterations after convergence instead of Python breaks,
- move the accept/reject policy onto the device,
- update only compact scalar summaries each step instead of full history arrays.

Gradient strategy:

- prefer ``lax.scan`` for differentiable bounded loops,
- avoid relying on reverse-mode through ``while_loop`` for the main path,
- if truly adaptive loops are needed, wrap them with custom VJP or use
  implicit differentiation around the converged solve.

Phase 3: Replace VMEC-style restart logic with a modern nonlinear controller
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

VMEC2000's exact restart cadence is excellent for parity, but it is not the
best contract for accelerated mode.

Controller upgrades:

- replace Garabedian-style parity restarts with a trust-region or safeguarded
  line-search controller based on:

  - ``fsq_total`` decrease,
  - Jacobian health,
  - state step norm,
  - preconditioned residual norm.

- add Anderson acceleration and/or L-BFGS style outer updates once the solution
  enters a stable basin,
- retain a robust fallback to steepest-descent-like updates when conditioning is
  poor,
- promote VMEC++-style "bad progress" restart ideas from optional experiments to
  part of the accelerated controller if they improve general convergence.

Success criterion:

- fewer iterations and fewer rejected steps, without requiring the same
  iteration history as VMEC2000.

Phase 4: Use continuation aggressively
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The DESC continuation literature is directly relevant here.

Add staged continuation over:

- radial resolution ``ns``,
- angular resolution ``mpol``/``ntor``,
- pressure or current amplitude,
- boundary-shape amplitude,
- free-boundary coupling strength.

Policy:

- solve an easier nearby equilibrium first,
- warm-start the next stage from the previous converged state,
- escalate resolution and physics only when residual targets are met.

This should become the default path for hard cases, especially:

- large ``ns`` axisymmetric tokamak cases,
- large 3D fixed-boundary cases,
- free-boundary cases with heavy vacuum coupling,
- ``lasym=True`` cases with extra mode content.

Phase 5: Refactor transforms and synthesis around factorized operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The angular transforms and synthesis are a major runtime and memory cost.

Required changes:

- finish the transition to separable theta/zeta transforms everywhere,
- eliminate redundant materialization of full ``(ns, ntheta, nzeta)`` tensors
  when only edge or half-mesh data are needed,
- fuse transform + synthesis stages where the intermediate arrays are only used
  once,
- keep mode-major and surface-major layouts explicit and benchmark both on CPU
  and GPU,
- reduce signed/even/odd parity shuffles in the hot path.

State-of-the-art implementation options:

- XLA-friendly separable DFT/GEMM blocks,
- optional FFT-backed paths where scaling and weights still preserve acceptable
  final accuracy,
- custom Pallas kernels for the most memory-bound synthesis kernels if XLA
  fusion stalls.

Phase 6: Redesign the preconditioner for accelerated mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The current preconditioner is heavily shaped by parity requirements.

Accelerated-mode goals:

- keep the physical role of the preconditioner,
- reduce rebuild cost,
- make the application fully batched and device-resident,
- support cheaper approximate updates between full refreshes.

Concrete work:

- use fused tridiagonal solves where they improve runtime and do not damage
  final convergence quality,
- refresh the preconditioner on residual/conditioning triggers rather than on
  VMEC2000 cadence alone,
- explore low-rank or diagonal-plus-banded updates instead of full rebuilds,
- make the preconditioner compatible with implicit differentiation and
  Jacobian-vector products.

Phase 7: Free-boundary redesign for accelerated mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Free-boundary is the largest remaining cost center.

Short-term changes:

- keep free-boundary control fully on-device,
- batch several cheap reuse steps between full vacuum refreshes,
- continue pushing edge-only coupling data instead of full-volume tensors,
- move dense vacuum operator assembly/factorization out of the inner loop
  whenever geometry permits reuse.

Medium-term changes:

- replace repeated dense operator work with a matrix-free or cached-operator
  apply,
- exploit toroidal block structure and separability in the vacuum solve,
- evaluate low-rank, hierarchical, or FFT-accelerated boundary integral
  formulations for large free-boundary cases.

Validation focus:

- ``input.DIII-D_lasym_false``,
- bundled non-axisymmetric free-boundary cases,
- at least one additional larger non-axisymmetric free-boundary example with
  finite pressure/current.

Phase 8: Memory-first refactor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Runtime work will stall if peak memory stays too high.

Required memory changes:

- donate buffers into scan kernels where safe,
- use compact ring-buffer histories instead of full arrays,
- checkpoint only selected intermediates needed for gradients,
- apply rematerialization selectively around the largest synthesis blocks,
- keep stage-static trig/phase tables cached and shared,
- use chunked theta/zeta evaluation for the largest cases,
- avoid simultaneous storage of multiple equivalent parity layouts.

Acceptance target:

- reduce peak memory materially on the current large outliers without pushing
  runtime backward.

Phase 9: Gradient path split
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The project must stay differentiable, but accelerated solves do not need to
backpropagate through every iteration in the same way.

Recommended policy:

- small solves and tests: allow explicit differentiation through the bounded
  iteration loop,
- large production solves: default to implicit differentiation at the converged
  equilibrium,
- document clearly which gradients are exact-through-iterations and which are
  implicit fixed-point gradients.

This is essential for memory scalability.

Phase 10: Expand the example matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The current bundled set is useful but not yet sufficient for an accelerated
solver policy.

Add examples in three classes:

1. Small smoke cases

   - fast CPU/GPU sanity checks,
   - useful for compile and API regressions.

2. Medium realistic cases

   - representative default examples for user-facing performance claims,
   - should cover fixed/free, axisymmetric/non-axisymmetric, ``lasym=False/True``.

3. Stress cases

   - large ``ns`` axisymmetric tokamak,
   - large 3D fixed-boundary case,
   - large non-axisymmetric free-boundary case,
   - at least one free-boundary ``lasym=True`` case with finite pressure/current.

If the repo cannot bundle all assets, provide deterministic generation scripts
for the heavier cases.

Phase 11: Validation and release gates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Accelerated mode should have its own CI gates.

Required checks:

- ``pytest -q`` plus accelerated-mode regression tests,
- docs build,
- bundled example runtime/memory matrix on CPU,
- GPU benchmark tier on a CUDA runner or scheduled benchmark host,
- final ``wout`` comparison against VMEC2000 with ``rtol=1e-2`` targets,
- gradient checks on a representative fixed-boundary and free-boundary case.

Do not ship accelerated mode as default until:

- every bundled example converges to the accelerated-mode target,
- no benchmarked case requires user-set environment variables for correctness,
- the solver policy remains general for unseen inputs.

Recommended implementation order
--------------------------------

Follow this sequence:

1. measurement harness,
2. explicit accelerated-mode API split,
3. device-resident outer loop,
4. modern nonlinear controller,
5. continuation policy,
6. transform/synthesis refactor,
7. preconditioner redesign,
8. free-boundary redesign,
9. memory-first cleanup,
10. gradient-path split and documentation.

This ordering attacks the dominant architecture problem first: too much
host-driven control around moderate-size kernels.

Concrete first sprint
---------------------

The first sprint on this branch should do only the highest-value work:

1. Add accelerated-mode benchmark and acceptance harness.
2. Add a new solver mode flag and thread it through CLI/API.
3. Prototype a bounded masked-scan outer loop for fixed-boundary solves.
4. Replace parity-based scan acceptance with final-quality-based acceptance.
5. Benchmark on:

   - ``input.circular_tokamak``,
   - ``input.up_down_asymmetric_tokamak``,
   - ``input.LandremanPaul2021_QA_lowres``,
   - ``input.LandremanPaul2021_QA_reactorScale_lowres``,
   - ``input.cth_like_free_bdy``,
   - ``input.cth_like_free_bdy_lasym_small``,
   - ``input.DIII-D_lasym_false``.

6. Define the first go/no-go threshold:

   - fixed-boundary accelerated mode must beat current default runtime on at
     least four of the six representative fixed-boundary cases,
   - no representative case may lose final ``wout`` quality beyond the
     ``rtol=1e-2`` target.

References and inspiration
--------------------------

- VMEC++ repository:
  https://github.com/proximafusion/vmecpp
- The Numerics of VMEC++:
  https://arxiv.org/abs/2502.04374
- DESC Part I:
  https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/desc-stellarator-code-suite-part-1-quick-and-accurate-equilibria-computations/69611B218B412BC279BDF2A080135718
- DESC Part II (perturbation and continuation):
  https://www.cambridge.org/core/services/aop-cambridge-core/content/view/5766F6B713EC93D438A35705F2C1E861/S0022377823000399a.pdf/desc_stellarator_code_suite_part_2_perturbation_and_continuation_methods.pdf
- JAX ``lax.scan``:
  https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html
- JAX ``lax.while_loop``:
  https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html
- JAX async dispatch:
  https://docs.jax.dev/en/latest/async_dispatch.html
- JAX profiling:
  https://docs.jax.dev/en/latest/profiling.html
- JAX buffer donation:
  https://docs.jax.dev/en/latest/buffer_donation.html
- JAX gradient checkpointing:
  https://docs.jax.dev/en/latest/gradient-checkpointing.html
