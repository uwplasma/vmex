Performance notes
=================

This page collects practical advice for using ``vmec-jax`` efficiently.

Enable float64
--------------

VMEC2000 is float64-first. For parity, enable x64 in JAX::

  export JAX_ENABLE_X64=1

JIT boundaries and compile latency
----------------------------------

On CPU, compilation can dominate runtime for moderate problem sizes. ``vmec-jax`` uses:

- a jitted geometry kernel (``eval_geom``),
- non-jitted solver gradients by default (to reduce compile latency).

Solver functions accept ``jit_grad=True`` to trade longer compile time for faster
iterations.

To reduce initial compilation overhead during startup, you can disable JIT for
the **initial guess** phase by setting::

  export VMEC_JAX_DISABLE_JIT_INIT=1

This keeps the solver kernel JIT-compiled, but avoids compiling the initial
boundary->state projection path (useful for short runs or rapid profiling).

To reduce per-iteration latency spikes in multigrid runs, ``vmec-jax`` can
precompile the force kernel at the start of each stage. This is enabled by
default when ``jit_forces=True``; you can override it with::

  export VMEC_JAX_JIT_PRECOMPILE=0

If you prefer to run a few iterations without JIT before compiling, set::

  export VMEC_JAX_JIT_WARMUP_ITERS=2

Scan-mode iteration (fast path)
-------------------------------

The scan-based loop lifts the VMEC2000 iteration into ``jax.lax.scan`` to reduce
Python overhead. You can enable it with:

- ``--fast`` on the CLI,
- ``performance_mode=True`` in ``run_fixed_boundary`` (default),
- or ``VMEC_JAX_USE_SCAN=1``.

**Important**: scan parity is case-dependent on difficult large-``ns`` stages.
The runtime uses scan as the default fast path, with a fallback to the
non-scan parity path when parity guards detect drift. You can always force the
conservative path with ``--parity``.

For LASYM fixed-boundary stages in ``performance_mode=True``, the default
selector now uses:

- a timed scan/non-scan probe on CPU backends,
- a short parity-only probe on accelerator backends.

This keeps the default GPU path from paying the full warmed non-scan timing
cost while still rejecting scan when the short parity probe disagrees.

Controls:

- ``VMEC_JAX_DYNAMIC_SCAN_TIMED=1``: force a timed probe even on accelerators.
- ``VMEC_JAX_DYNAMIC_SCAN_TIMED=0``: force parity-only probing.
- ``VMEC_JAX_DYNAMIC_SCAN_ITERS=<int>``: override the probe window
  (defaults to ``10`` on CPU, ``3`` on accelerators).

For quiet accelerator scans, ``vmec-jax`` also increases the default scan chunk
target and caps each chunk to the remaining iteration budget. This reduces
host/device launch overhead without changing the in-scan hold semantics.

Controls:

- ``VMEC_JAX_SCAN_CHUNK_SIZE=<int>``: override the chunk target explicitly.

Debug dump env vars are incompatible with scan mode.

Experimental accelerated mode
-----------------------------

``vmec-jax`` now exposes an explicit experimental solver policy for the
non-parity performance track:

- Python API: ``run_fixed_boundary(..., solver_mode="accelerated")``
- CLI: ``vmec_jax input.name --solver-mode accelerated``

Current behavior of this first slice:

- fixed-boundary stages force the masked VMEC-control scan path and skip the
  parity-oriented scan-selection probes,
- when the caller does not explicitly request multigrid, accelerated
  fixed-boundary runs now default to a single final-grid stage. This avoids
  per-stage interpolation and recompilation overhead that was dominating the
  heavy bundled fixed-boundary cases,
- accelerated fixed-boundary stages still use a scalar total-residual target
  derived from the input ``ftol`` budget as a cheap *in-block* early-stop:
  ``fsq_total_target = ftol * 3`` for the three VMEC residual channels
  (``fsqr``, ``fsqz``, ``fsql``). However, the returned accelerated
  fixed-boundary run now accepts convergence only when the final-stage
  per-channel rule is satisfied, matching the requested ``FTOL`` literally on
  ``fsqr``, ``fsqz``, and ``fsql``,
- the experimental solver controls no longer rely on fixed absolute
  convergence thresholds. By default:

  - gradient-based stopping derives ``grad_tol`` from the initial gradient
    scale and machine precision,
  - the Gauss-Newton path derives its CG tolerance from the current residual
    progress against the same ``ftol`` budget,
  - the Gauss-Newton damping seed is derived from the local normal-equation
    curvature scale instead of a fixed literal damping floor,
  - residual-objective ``m=1`` release thresholds now default to ``ftol``
    instead of hardcoded residual cutoffs,
- accelerated runs now request compact histories and a minimal resume payload
  by default, so the result object does not carry the full parity-era
  momentum/preconditioner cache unless the caller explicitly asks for it,
- the CLI executable now has an extra fixed-boundary-only policy layer on top
  of accelerated mode:

  - the first attempt is the same fast final-grid solve used by the optimized
    Python API path,
  - if a staged input provides explicit ``NS_ARRAY`` / ``NITER_ARRAY`` and the
    fast final-grid attempt misses the target, the CLI replays that staged
    schedule automatically (accelerated coarse stages, strict parity final
    stage),
  - if the input is staged but does not provide ``NITER_ARRAY`` and the user
    explicitly forces accelerated mode, the CLI falls back to a reduced
    warm-start multigrid budget derived from the coarsest-to-finest ``ns``
    ratio,
  - strict parity finish blocks then continue from state only, without reusing
    the parity-era nonlinear-controller caches,
- free-boundary cases currently stay on the existing robust path; accelerated
  free-boundary control is not implemented yet,
- the mode is intended to reduce control overhead while preserving final
  residual quality, not to reproduce the VMEC2000 iteration trace.

Use the dedicated comparison harness to evaluate it against the current default
solver policy:

.. code-block:: bash

  python tools/diagnostics/benchmark_accelerated_mode.py \
    --baseline-mode default \
    --candidate-mode accelerated \
    --candidate-cli-fixed-boundary-mode \
    --kind fixed \
    --jax-platforms cpu

The harness reports:

- cold and warm runtime,
- peak process memory,
- final ``fsq_total``,
- convergence flags,
- reference-``wout`` relRMS metrics when bundled references are available.

Early March 2026 smoke results on the local CPU host:

- ``input.up_down_asymmetric_tokamak``: about ``4.1x`` warm speedup with a
  materially smaller memory footprint than the current default path,
- ``input.circular_tokamak``: approximately neutral in runtime, with good
  final quality (``~1.2e-5`` reference-``wout`` relRMS),
- ``input.LandremanPaul2021_QA_lowres``: approximately neutral with the
  current ftol-derived total target,
- free-boundary accelerated mode is currently a control-path alias for the
  robust baseline, not a new fast free-boundary controller.

Serial fixed-boundary follow-up measurements from
``outputs/accelerated_fixed_boundary_reassessment_20260309/summary.json``
show why the single-grid default is now the accelerated fixed-boundary policy:

- ``input.LandremanSenguptaPlunk_section5p3_low_res``:
  ``45.48s`` current default vs ``0.198s`` accelerated single-grid and
  ``0.232s`` accelerated explicit multigrid; the accelerated single-grid route
  converges and is dramatically faster than both,
- ``input.LandremanPaul2021_QA_lowres``:
  ``8.18s`` current default vs ``7.31s`` accelerated single-grid and
  ``8.10s`` accelerated explicit multigrid; the accelerated single-grid route
  now carries the full staged iteration budget and converges at
  ``~3.0e-13``,
- ``input.LandremanPaul2021_QA_reactorScale_lowres``:
  ``21.15s`` warmed on the optimized CLI track versus ``43.20s`` for VMEC2000
  on the current bundled CPU benchmark, showing the same controller policy
  carries over to a heavier reactor-scale 3D case.

The fixed-boundary CLI path is now best understood as a controller stack,
not a single algorithm:

- easy inputs stay on the fast final-grid optimized path,
- staged inputs can automatically escalate into their input-defined stage
  schedule before paying the cost of parity finish blocks,
- only the genuinely hard cases should reach the final strict continuation
  phase.

For an up-to-date side-by-side comparison on your machine, use the bundled
driver example:

.. code-block:: bash

  python examples/fixed_boundary_driver_tracks.py \
    examples/data/input.circular_tokamak \
    --quiet --json

On the current branch, that example produced the following local CPU smoke
result for ``input.circular_tokamak``:

- parity track: ``28.863s`` with ``fsq_total ~2.04e-14``,
- optimized CLI-style track: ``3.445s`` with ``fsq_total ~2.85e-14``.

That example uses the same public Python driver entry point as library users,
but it enables ``cli_fixed_boundary_mode=True`` on the optimized path so the
controller matches the executable behavior exactly.

Latest serial bundled fixed-boundary reassessment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current bundled fixed-boundary benchmark set uses the shipped QA/QH
reactor-scale reference inputs in place of the retired internal stress cases.

Current warmed fixed-boundary CPU reassessment on the optimized CLI track,
using the same branch baseline as the comparator, is recorded in
``outputs/accelerated_cli_fixed_boundary_full_20260311_r2/summary.json``:

- all 16 bundled fixed-boundary cases converge on both the baseline and
  optimized paths,
- the optimized path is now faster on 13 of 16 cases and roughly neutral on
  the remaining 3,
- the earlier runtime-regression blocker on the bundled CPU matrix is gone.

Final-``wout`` accuracy is a separate gate from residual convergence. The
earlier full fixed-boundary audit is recorded in
``outputs/fixed_wout_audit_20260310_r3/summary.json``, and the later staged-3D
controller fixes improved several non-axisymmetric cases materially:

- strong final-``wout`` agreement on the current shipped showcase cases:
  ``ITERModel`` (max relRMS ``6.01e-06``),
  ``shaped_tokamak_pressure`` (``1.55e-07``),
  ``circular_tokamak`` (``1.03e-05``),
- targeted staged non-axisymmetric follow-up fixes then brought the reactor
  QA/QH Fourier-channel errors down to the branch target range on direct
  comparisons:
  ``LandremanPaul2021_QA_lowres`` now reaches about
  ``rmnc 5.83e-05``, ``zmns 2.83e-04``, ``lmns 4.75e-03``;
  ``LandremanPaul2021_QA_reactorScale_lowres`` reaches about
  ``rmnc 2.49e-05``, ``zmns 1.61e-04``, ``lmns 2.86e-03``;
  ``LandremanPaul2021_QH_reactorScale_lowres`` reaches about
  ``rmnc 6.12e-05``, ``zmns 2.60e-04``, ``lmns 9.97e-03``,
- the runtime picture is now favorable on the bundled CPU matrix, but the
  branch remains experimental because the non-parity scope and GPU/default
  policy questions are broader than this one fixed-boundary CPU result.
- a later ``wout`` audit found that much of the remaining QA/QH benchmark
  error was coming from symmetry-forbidden geometry channels being exported
  for ``lasym=False``:
  zeroing ``rmns`` and ``zmnc`` in ``wout`` for symmetric runs reduced the
  bundled 3D quality metric from about ``3.37e-01`` to ``4.19e-02`` on
  ``LandremanPaul2021_QA_lowres``, from ``3.56e+00`` to ``3.14e-02`` on
  ``LandremanPaul2021_QA_reactorScale_lowres``, and from ``4.61e+00`` to
  ``2.22e-02`` on ``LandremanPaul2021_QH_reactorScale_lowres`` in
  ``outputs/fixed_wout_3d_audit_20260311_r1/summary.json``,
- the next narrowed audit then focused on the staged current-driven 3D
  continuation policy itself:
  for 3-stage ``lasym=True`` current-driven runs, the first attempt used a
  mixed controller that kept the entry/final stages conservative and
  accelerated only the interior stage,
- that change materially reduced the remaining non-axisymmetric lambda drift:
  ``basic_non_stellsym_pressure`` improved from about ``3.46e-01`` to
  ``3.46e-02`` max relRMS while still running faster than baseline
  (about ``23.69s`` baseline vs ``19.36s`` optimized in the targeted audit),
  and the QA/QH reactor-scale cases held at about ``3.14e-02`` and
  ``2.22e-02`` with large runtime wins,
- the remaining bundled 3D quality gap is therefore much narrower and now
  mostly a lambda-field accuracy question rather than a broad geometry or
  force-balance mismatch,
- a final targeted controller split closed most of that remaining gap:
  ``lasym=False`` current-driven 3D CLI runs now go straight to staged
  multigrid on the conservative non-scan residual path,
- with that split, the latest targeted audit reached:
  ``LandremanPaul2021_QA_lowres`` about ``4.20e-03`` max relRMS at
  about ``100.6s`` warmed runtime,
  ``LandremanPaul2021_QA_reactorScale_lowres`` about ``6.42e-04`` at
  about ``125.1s``,
  ``LandremanPaul2021_QH_reactorScale_lowres`` about ``6.00e-05`` at
  about ``180.2s``,
  while ``basic_non_stellsym_pressure`` remained the last branch-specific
  lambda outlier,
- the follow-on strict-``FTOL`` pass removed that branch-specific regression by
  keeping ``lasym=True`` current-driven 3D staged runs fully on the
  conservative controller:
  ``basic_non_stellsym_pressure`` now lands back at about ``2.98e-02`` max
  relRMS, matching the current baseline quality instead of worsening it, with
  essentially neutral warmed runtime (about ``22.24s`` baseline vs ``22.31s``
  optimized),
- a follow-on experiment that added a final-grid parity polish to the
  staged 3D accelerated path was rejected because it raised runtime
  substantially without improving those benchmarked quality numbers.

Representative warmed CPU baseline-vs-optimized points from the current full
matrix:

- ``ITERModel``: ``0.36s`` baseline vs ``0.20s`` optimized,
- ``LandremanPaul2021_QA_lowres``: ``8.75s`` baseline vs ``7.86s`` optimized,
- ``LandremanPaul2021_QA_reactorScale_lowres``:
  ``10.42s`` baseline vs ``10.06s`` optimized,
- ``LandremanSenguptaPlunk_section5p3_low_res``:
  ``49.56s`` baseline vs ``0.21s`` optimized,
- ``basic_non_stellsym_pressure``:
  ``12.81s`` baseline vs ``1.04s`` optimized,
- ``up_down_asymmetric_tokamak``: ``1.01s`` baseline vs ``0.45s`` optimized.

Same-host CPU/GPU reassessment on a reference GPU workstation is now complete
for the same 16-case bundled fixed-boundary set:

- both CPU and GPU converge on all 16 cases,
- GPU is already faster on the heavier 3D cases
  (``LandremanPaul2021_QA_lowres``,
  ``LandremanPaul2021_QA_lowres1``,
  ``LandremanPaul2021_QA_reactorScale_lowres``,
  ``LandremanPaul2021_QH_reactorScale_lowres``,
  ``cth_like_fixed_bdy``),
- CPU still wins on the remaining 11 smaller or more launch-latency-dominated
  cases.

Representative same-host CPU/GPU warmed comparisons:

- ``LandremanPaul2021_QA_reactorScale_lowres``:
  CPU ``23.72s`` vs GPU ``13.06s``,
- ``LandremanPaul2021_QH_reactorScale_lowres``:
  CPU ``31.39s`` vs GPU ``16.56s``,
- ``cth_like_fixed_bdy``:
  CPU ``1.45s`` vs GPU ``0.89s``,
- ``circular_tokamak``:
  CPU ``0.70s`` vs GPU ``1.97s``,
- ``solovev``:
  CPU ``0.16s`` vs GPU ``0.55s``.

That makes the current branch state clearer than the earlier stress-case
benchmark story:

- the optimized fixed-boundary CLI path is now a clean warmed CPU win across
  the shipped bundled matrix,
- the GPU path is functional and convergent on the same bundled matrix,
- automatic backend selection is now the remaining step before the GPU story
  becomes uniformly stronger than CPU on mixed-size workloads.

Additional controller finding from March 2026:

- the existing fully non-VMEC scan path was re-probed as a possible next
  accelerated controller, but it is not yet robust enough to become the
  default accelerated path: on representative fixed-boundary cases it is much
  faster, but it can diverge badly in ``fsq_total`` and final ``wout`` quality.
  The current accelerated mode therefore stays on the masked VMEC-control scan
  until a more stable device-resident controller is in place.

If you want an automatic parity probe when using scan, set::

  export VMEC_JAX_SCAN_PARITY_GUARD=1

This runs a short scan-vs-non-scan probe at the start of each stage and falls
back to the non-scan loop if a mismatch is detected. It is **off by default**
because it adds extra compilation and iteration overhead.

Scan chunking (fixed NSTEP blocks)
----------------------------------

To avoid retracing for variable tail lengths, the scan loop executes in fixed
chunks of length ``NSTEP`` (the VMEC input parameter). Iterations beyond
``NITER`` are masked by the in-scan hold condition, so the extra work is a
no-op and does not affect parity.

Controls:

- ``VMEC_JAX_VMEC2000_CHUNKED=1`` (default): enable chunked scan.
- ``VMEC_JAX_SCAN_CHUNK_SIZE=<int>``: override chunk length (defaults to
  ``NSTEP``).

This reduces compilation cache misses when the stage transition changes
``NITER`` but keeps the same ``NSTEP`` cadence.

Live NSTEP printing (debug callback)
------------------------------------

VMEC2000-style iteration rows (scan and non-scan) are printed using a JAX debug
callback by default. This keeps the output VMEC-like without inserting extra
Python-side synchronization on every step.

Defaults:

- Live printing is **enabled** when ``verbose`` and ``vmec2000_control`` are on.
- The backend uses ``jax.debug.print`` (differentiable).

Disable live printing with:

::

  export VMEC_JAX_SCAN_PRINT=0

If you want to reduce any remaining host-callback overhead, increase ``NSTEP``
in the input file. Fewer prints means fewer callbacks.

Quiet scan runs (``--quiet`` / ``verbose=False``) automatically switch to a
minimal history mode: only ``fsqr/fsqz/fsql`` (and therefore ``w_history``) are
kept. Per-iteration print scalars (``r00``, ``w_mhd``) and time-step histories
are skipped to reduce host/device traffic. Override with::

  export VMEC_JAX_SCAN_MINIMAL=0   # keep full scan diagnostics
  export VMEC_JAX_SCAN_MINIMAL=1   # force minimal histories

In fast mode (``performance_mode=True`` / ``--fast``), ``scan_minimal`` is the
default for **quiet** runs (``verbose=False``) unless explicitly overridden by
``VMEC_JAX_SCAN_MINIMAL``. When ``verbose=True``, scan keeps the extra scalar
histories needed for VMEC-style printing.

Advanced knobs (not required for normal use):

- ``VMEC_JAX_SCAN_PRINT_MODE=debug_print`` (default)
- ``VMEC_JAX_SCAN_PRINT_MODE=debug_callback`` (alternate callback)
- ``VMEC_JAX_SCAN_PRINT_ORDERED=1`` to force ordered prints (may reduce parallelism)

DFT tomnsps (GEMM path)
-----------------------

VMEC2000's ``tomnsps`` analysis transform is now implemented as a two-stage
DFT using the precomputed ``fixaray`` trig/weight tables:

- theta stage: multiply by ``cosmui/sinmui`` (endpoint-weighted + ``mscale``),
- zeta stage: multiply by ``cosnv/sinnv`` (with ``nscale`` and ``n*NFP`` in
  ``cosnvn/sinnvn`` for derivative terms).

The core contractions are done with batched ``dot_general`` calls so XLA can
lower them into GEMM kernels. This follows the VMEC++ basis approach (see
References [5-6]) while keeping VMEC2000 parity.

Recent updates to the DFT path:

- **Stacked theta contractions**: multiple force kernels are concatenated into
  a single cosine and sine projection per iteration, reducing the number of
  ``dot_general`` launches.
- **Derivative-factor fusion**: the :math:`n\,\mathrm{NFP}` factor for
  ``cosnvn/sinnvn`` is applied *after* the zeta contraction, so the same
  ``cosnv/sinnv`` basis can be reused for derivative blocks.
- **Stacked zeta contractions**: cosine- and sine-basis transforms for the
  derivative and non-derivative blocks are grouped to reduce kernel dispatches.

An FFT-based path remains available for experiments:

- ``VMEC_JAX_TOMNSPS_FFT=1`` enables the FFT implementation (not default).

Preconditioner weight caching
-----------------------------

The 1D radial preconditioner uses angular weights
:math:`w_i=\mathrm{cosmui3}_{i,0}/\mathrm{mscale}_0` on the VMEC internal grid.
These weights are now cached in the trig table as ``wint3_precond`` and reused
whenever the preconditioner diagonal is refreshed. This avoids rebuilding the
same weight tensor in every refresh call and keeps the preconditioner refresh
path purely algebraic in ``bsq``, ``r12``, ``sqrtg``, ``ru12``, and ``zu12``.

Free-boundary WP1 micro-benchmark
---------------------------------

For free-boundary staging, use the dedicated benchmark script:

.. code-block:: bash

  python tools/benchmarks/bench_free_boundary_wp1.py \
    --input examples/data/input.DIII-D \
    --interp-points 20000 \
    --interp-repeats 5

This reports:

- metadata validation/load time,
- full mgrid tensor load time,
- interpolation throughput and sampled ``|B_ext|`` stats.

Solver note: this benchmark isolates external-field staging cost. The sampling
toggle below is diagnostic-only; it does not describe overall free-boundary
solver maturity. You can disable that sampling with:

.. code-block:: bash

  export VMEC_JAX_FREEB_SAMPLE_EXTERNAL=0

WP2 free-boundary runtime controls
----------------------------------

Current free-boundary coupling uses a lightweight spectral potential solve.
To keep runtime bounded:

- mgrid field tensors are cached by path in-process (avoids per-iteration
  NetCDF reloads),
- Poisson spectral denominators are stage-static,
- ``ivacskip`` reuses prior potential (skip solve) when ``ivac != 1``.

Control flags:

.. code-block:: bash

  export VMEC_JAX_FREEB_COUPLE_EDGE=1         # default: on
  export VMEC_JAX_FREEB_SAMPLE_EXTERNAL=1     # default: on

If profiling free-boundary solver-only cost, disable sampling diagnostics:

.. code-block:: bash

  export VMEC_JAX_FREEB_SAMPLE_EXTERNAL=0

Bundled example runtime/memory matrix (March 2026)
--------------------------------------------------

For repeatable runtime/memory sweeps across the bundled inputs, use:

.. code-block:: bash

  python tools/diagnostics/example_runtime_memory_matrix.py \
    --backend both \
    --vmec-exec /path/to/xvmec2000

Recent artifacts from this tool:

- ``outputs/example_runtime_memory_matrix_cpu_20260306/summary.json``:
  all bundled examples on a reference CPU host, including VMEC2000 timings.
- ``outputs/example_runtime_memory_matrix_gpu_20260306_summary.json``:
  all bundled examples on a reference CUDA host (CUDA JAX).
- ``outputs/example_runtime_memory_matrix_gpu_freeb_20260306_rerun_summary.json``:
  corrected GPU rerun for the bundled free-boundary cases after staging bundled
  ``mgrid`` files in the benchmark clone.

Current snapshot highlights:

- Fixed-boundary scan performance on the reference GPU host improved materially
  after the accelerator-aware scan probe and quiet-scan chunking changes:

  - ``input.circular_tokamak`` now runs in about ``13.8s`` / ``1.97 GiB``.
  - ``input.LandremanPaul2021_QA_lowres`` now runs in about ``33.9s`` /
    ``2.66 GiB``.
  - ``input.up_down_asymmetric_tokamak`` now runs in about ``16.5s`` /
    ``1.60 GiB``.
  - ``input.basic_non_stellsym_pressure`` now runs in about ``141.1s`` /
    ``3.68 GiB``.
  - ``input.LandremanSenguptaPlunk_section5p3_low_res`` now runs in about
    ``77.1s`` / ``2.13 GiB``.

- Fixed-boundary ``lasym=True`` on the reference CPU host remains:

  - ``input.up_down_asymmetric_tokamak`` about ``6.7s`` / ``0.89 GiB`` versus
    VMEC2000 about ``0.74s``.
  - ``input.basic_non_stellsym_pressure`` about ``29.7s`` / ``3.22 GiB``
    versus VMEC2000 about ``2.02s``.
  - ``input.LandremanSenguptaPlunk_section5p3_low_res`` about ``46.8s`` /
    ``4.07 GiB`` versus VMEC2000 about ``0.69s``.

- Bundled free-boundary cases remain the dominant default-path outliers:

  - ``input.DIII-D_lasym_false``:
    about ``428.2s`` / ``7.36 GiB`` on the reference CPU host,
    about ``1602.3s`` / ``6.23 GiB`` on the reference GPU host,
    versus VMEC2000 about ``14.4s``.
  - ``input.cth_like_free_bdy``:
    about ``41.8s`` / ``1.64 GiB`` on the reference CPU host,
    about ``155.8s`` / ``2.30 GiB`` on the reference GPU host,
    versus VMEC2000 about ``2.48s``.
  - ``input.cth_like_free_bdy_lasym_small``:
    about ``37.6s`` / ``1.47 GiB`` on the reference CPU host,
    about ``103.5s`` / ``1.97 GiB`` on the reference GPU host,
    versus VMEC2000 about ``0.63s``.

- Recent parity-path free-boundary GPU work narrowed the large-``ns``
  force-kernel overhead:

  - deferring non-scan scalar-history materialization was effectively neutral
    on the smaller ``input.cth_like_free_bdy`` case
    (about ``111.3s`` warm on ``70fc418`` versus about ``111.4s`` warm on
    ``f35ce44``).
  - passing only the free-boundary ``bsqvac`` edge slice into the force kernel
    instead of rebuilding a mostly-zero ``(ns, ntheta, nzeta)`` array every
    iteration materially improves the heavy axisymmetric case. On the reference
    GPU host, a parity-path ``max_iter=10`` probe of
    ``input.DIII-D_lasym_false`` dropped:

    - ``compute_forces`` from about ``5.79s`` total
      (``0.579s/iter`` on ``70fc418``) to about ``2.58s`` total
      (``0.258s/iter`` on ``f35ce44``),
    - ``preconditioner`` from about ``0.675s`` to about ``0.324s``,
    - ``update`` from about ``0.914s`` to about ``0.535s``.

- The current GPU path is not yet a universal speedup:

  - same-host CPU/GPU benchmarking shows GPU is already faster on the heavier
    QA/QH reactor-scale cases, but not yet on the smallest axisymmetric cases,
  - the current mixed result is now backend-selection limited rather than
    convergence limited: all 16 bundled fixed-boundary cases converge on both
    CPU and GPU.

Why the GPU can still be slower than the CPU
--------------------------------------------

This is a consequence of the current solver architecture, not a statement that
VMEC-like equilibria are fundamentally better suited to CPUs. The short
version is:

- the **fast** path is the scan-lifted path, where JAX can keep long stretches
  of work on-device,
- the **parity** path is still a host-controlled VMEC2000-style iteration,
- many of the slowest benchmark rows are exactly those parity-path solves,
  especially free-boundary cases.

In more detail:

1. VMEC2000 parity requires a host-controlled nonlinear loop

   The conservative path preserves VMEC2000-style semantics such as:

   - Garabedian time-step control,
   - Jacobian sign checks,
   - same-iteration restarts,
   - free-boundary ``ivac/ivacskip/nvacskip`` cadence,
   - per-iteration diagnostics and VMEC-style tables,
   - stage transitions and cache refresh rules.

   In the current implementation, those decisions still happen in Python on the
   host. Each iteration therefore launches several short JAX kernels, waits for
   scalar decisions, then launches the next block. CPUs tolerate that control
   pattern much better than GPUs because the launch/synchronization cost is
   smaller.

2. The kernels are mostly moderate-size float64 kernels, not giant batched GPU kernels

   For parity we run in float64, matching VMEC2000 numerics. On many of the
   shipped examples the per-iteration grids are only moderate in size, so the
   GPU never reaches the kind of occupancy that would amortize launch overhead.
   The work is also heavy in transforms, synthesis, and tensor assembly
   (`bcovar`, `tomnsps`, force kernels), which are often memory-traffic bound
   rather than one large dense GEMM.

   The result is that the CPU can look surprisingly competitive, because it is
   executing the same float64 algebra with lower orchestration overhead and
   without paying for many small host->device transitions.

3. Free-boundary parity is the worst case for the current GPU stack

   Free-boundary adds more than just one extra kernel. It adds:

   - external/vacuum field staging,
   - extra edge-force coupling,
   - free-boundary reuse/refresh cadence,
   - more restart-sensitive control flow,
   - larger edge/state tensors on some axisymmetric cases.

   The timing probes in this repo show that on the current parity free-boundary
   GPU path, ``compute_forces`` dominates. For example, on the reference GPU
   host:

   - ``input.cth_like_free_bdy`` with ``performance_mode=False`` spends about
     ``0.278s/iter`` in ``compute_forces``, while preconditioning and update are
     much smaller.
   - ``input.DIII-D_lasym_false`` is even more sensitive to force-path data
     movement because of its large ``ns``.

4. Data movement and edge-coupling details matter a lot on large free-boundary cases

   Recent profiling made this explicit. Passing only the free-boundary
   ``bsqvac`` edge slice into the force kernel, instead of rebuilding a mostly
   zero ``(ns, ntheta, nzeta)`` array each iteration, was nearly neutral on the
   smaller ``input.cth_like_free_bdy`` case but materially improved the large
   axisymmetric case. On a parity-path ``max_iter=10`` probe of
   ``input.DIII-D_lasym_false`` on the reference GPU host:

   - ``compute_forces`` dropped from about ``0.579s/iter`` to about ``0.258s/iter``,
   - ``preconditioner`` dropped from about ``0.067s/iter`` to about ``0.032s/iter``,
   - ``update`` dropped from about ``0.091s/iter`` to about ``0.054s/iter``.

   That is a good example of the current situation: the GPU is not losing
   because of the physics model itself, but because the parity path still
   contains control-flow and data-shaping patterns that are cheap on CPU and
   expensive on GPU.

5. Compilation and warmup amplify the gap on short runs

   JAX/XLA compile cost is front-loaded. On short solves, or on runs that only
   execute a small number of iterations per stage, compile and cache warmup can
   dominate the wall time. This hurts accelerator results more than CPU results
   because the GPU path has higher startup overhead and stricter sensitivity to
   retracing/recompilation.

6. Differentiability and parity constraints limit aggressive GPU-only shortcuts

   ``vmec-jax`` is not trying to be a separate non-parity GPU solver. We are
   preserving:

   - end-to-end differentiability,
   - VMEC2000-compatible iteration behavior where parity is required,
   - VMEC-style outputs and diagnostics.

   That rules out some easy GPU wins that would change ordering, skip
   diagnostics, or replace the parity controller with a different nonlinear
   algorithm. The current performance work is therefore focused on moving more
   of the existing algorithm into longer device-resident regions without
   changing the numerical contract.

7. When the GPU already helps today

   The GPU story is already much better when the solve can remain on the fast
   scan path, or when repeated runs can amortize compile cost. That is why the
   fixed-boundary scan cases improved materially after:

   - accelerator-aware scan probing,
   - larger quiet-scan chunks,
   - reduced launch overhead in the scan path.

   The next large gains on GPU are therefore expected to come from the same
   direction on the parity/free-boundary side: keeping more of the
   force/residual/control pipeline on-device for longer stretches, and reducing
   per-iteration host orchestration.

Experimental tridiagonal solver (scan only)
-------------------------------------------

The scan preconditioner can optionally use XLA's fused tridiagonal solver with
pretransposed coefficients (``dl/d/du``) computed once per stage. This can be
faster but is **not parity-safe** in general.

Enable for experiments only:

- ``VMEC_JAX_TRIDI_SOLVE=1`` (build pretransposed coefficients)
- ``VMEC_JAX_SCAN_PRECOND_LAXTRIDI=1`` (use the fused solver in scan)

If parity diverges, leave these disabled (the default).

Boundary decomposition cache + JAX-friendly initial guess
---------------------------------------------------------

``boundary_from_indata`` now caches the boundary decomposition across runs
using the input file path + mtime (or a coefficient fingerprint when the path
is unavailable). This trims repeated host work in workflows that solve the same
input file multiple times in a single process.

The initial-guess path also supports a fully JAX-backed boundary flip and
constraint application, which reduces Python-side overhead and keeps the path
JAX-friendly for future JIT staging. Control this with:

- ``VMEC_JAX_INIT_GUESS_JAX=1`` (default): use JAX boundary flip path.
- ``VMEC_JAX_INIT_GUESS_JAX=0``: fall back to NumPy/Python boundary flips.

Implementation map (performance-critical paths)
------------------------------------------------

- ``vmec_jax/vmec_tomnsp.py``: VMEC ``fixaray`` tables + DFT-based ``tomnsps``.
- ``vmec_jax/init_guess.py``: initial guess, axis blending, JAX boundary flip.
- ``vmec_jax/boundary.py``: input boundary decomposition + cache.
- ``vmec_jax/static.py``: cached grids, phase stacks, and per-solve constants.

Recent profiling snapshot (QA, 3 iterations on CPU)
---------------------------------------------------

- Default loop: ~0.26s total wall time (post-warmup).
- Scan loop: ~0.083s total wall time (~3x faster for this short run).

Longer runs benefit more because Python control-flow overhead scales with the
iteration count in the non-scan path.

VMEC++ bad-progress restarts (optional)
-----------------------------------------------

VMEC++ introduces a "bad progress" restart policy that detects large residuals
on refined grids and restarts the time-step controller more aggressively. This
is now available in ``vmec_jax`` behind an explicit flag so the VMEC2000 parity
path remains unchanged by default.

The VMEC++-style trigger follows the VMEC++ criteria:

- ``iter2 - iter1 > k_preconditioner_update_interval / 2``
- ``iter2 > 2 * k_preconditioner_update_interval``
- ``fsqr + fsqz > 1e-2`` (physical residual on the full grid)

When triggered, the restart path reduces ``delt`` by ``1/1.03`` (the VMEC++
"bad progress" factor) and resets the cached preconditioner state.

Enable it with:

- ``run_fixed_boundary(..., vmecpp_restart=True)``

Note: the VMEC++ restart flag is currently wired to the VMEC2000-control path.
When scan is active, it takes effect on fallback segments that execute in the
non-scan parity controller.

Static precomputation
---------------------

Use ``VMECStatic`` to avoid rebuilding:

- mode tables,
- angle grids,
- Fourier basis tensors,
- radial grid.

VMEC phase-stack cache
----------------------

The VMEC real-space synthesis path builds full ``(m,n)`` phase tables from the
``fixaray`` trig tables. This is correct but expensive to repeat inside the JIT
kernel. ``VMECStatic`` now precomputes and caches stacked phase tensors for the
VMEC grid (including ``dtheta``/``dzeta`` variants) and attaches them to the
cached trig tables. The precompute uses NumPy on the host to avoid extra JAX
compilation work. This reduces both runtime and compilation work because the
kernel no longer rebuilds the phase tables from scratch every iteration.

Control this behavior with:

- ``VMEC_JAX_CACHE_VMEC_PHASE=1`` (default): precompute phase stacks in
  ``build_static`` for fastest execution.
- ``VMEC_JAX_CACHE_VMEC_PHASE=0``: skip the extra cached tensors to save memory.

Compilation cache
-----------------

JAX can persist compiled executables to disk. Enable it with
``VMEC_JAX_COMPILATION_CACHE_DIR=/path/to/cache`` (or the upstream
``JAX_COMPILATION_CACHE_DIR``) to drastically reduce *repeat* compile times
across runs with the same shapes/static arguments.

CLI profiling (pre-iteration overhead)
--------------------------------------

To capture a JAX trace for the VMEC2000-style CLI path, set
``VMEC_JAX_PROFILE_DIR`` before invoking ``vmec_jax``. By default the CLI also
emits a Perfetto-compatible trace (``perfetto_trace.json.gz``); disable that
extra file by setting ``VMEC_JAX_PROFILE_PERFETTO=0``. The trace is written in
TensorBoard/Chrome trace format::

  VMEC_JAX_PROFILE_DIR=/tmp/vmec_jax_trace \\
    vmec_jax examples/data/input.ITERModel --max-iter 3 --no-multigrid --no-use-input-niter --quiet

For tighter windows (e.g., pre-iteration or iter-1 only), set
``VMEC_JAX_PROFILE_WINDOW=pre`` (or ``iter1`` / ``iterN``) and optionally start
a profiler server for XProf inspection::

  VMEC_JAX_PROFILE_DIR=/tmp/vmec_jax_trace \\
  VMEC_JAX_PROFILE_WINDOW=pre \\
  VMEC_JAX_PROFILE_SERVER=1 VMEC_JAX_PROFILE_SERVER_PORT=9999 \\
    vmec_jax examples/data/input.ITERModel --max-iter 3 --no-multigrid --quiet

With ``VMEC_JAX_PROFILE_SERVER=1`` you can also capture a tight window using
``python -m jax.collect_profile`` from another terminal (see the JAX profiling
guide for the exact invocation).

Recent traces show that the pre-iteration time is dominated by JIT
compilation/cache misses (``pjit cache_miss`` + backend compile) rather than
the nonlinear iteration itself. This is expected for short runs on CPU.
For repeated runs, the compilation cache (``VMEC_JAX_COMPILATION_CACHE_DIR``)
can significantly reduce this overhead once the cache is warm.

Persistent compilation cache tuning
-----------------------------------

JAX's persistent cache can be made more aggressive via ``vmec_jax`` environment
variables:

- ``VMEC_JAX_CACHE_MIN_COMPILE_TIME_SECS`` (default: 0)
- ``VMEC_JAX_CACHE_MIN_ENTRY_SIZE_BYTES`` (default: -1)
- ``VMEC_JAX_COMPILATION_CACHE_MAX_SIZE`` (optional)

These map to JAX's persistent cache configuration and allow caching more (or
fewer) compiled executables to reduce repeat-start latency for stable shapes.
Enable cache-miss diagnostics by setting
``VMEC_JAX_EXPLAIN_CACHE_MISSES=1``; JAX will log a short summary whenever a
cache miss triggers a compilation.

Batched radial smoothing
------------------------

The scan path now batches the radial tridiagonal smoother across the R/Z
components (and separately for lambda) so the solver does fewer tridi solves per
iteration. This reduces kernel count and Python overhead while preserving the
VMEC update math.

Batched VMEC real-space synthesis
---------------------------------

The VMEC-grid synthesis path now batches base + derivative (dtheta/dzeta)
evaluations into a single stacked ``einsum`` call. This reduces kernel count in
the pre-iteration setup (especially the bcovar/realspace pipeline) while
preserving the original algebra and parity outputs.

Vectorized multigrid conversion
-------------------------------

Multigrid staging now uses the vectorized signed↔(m,n) conversion helpers from
``vmec_parity`` instead of Python loops. In the current path the signed→(m,n)
conversion uses precomputed dense maps (matmul) to avoid repeated gather-heavy
indexing. This trims host-side overhead during grid transitions, which shows up
prominently in short profiling traces.

Multigrid interpolation caches
------------------------------

Radial interpolation now caches the ``(j1,j2,xint)`` weights and ``scalxc``
profiles for reuse across multigrid stages. This reduces host-side setup costs
when multiple grids are visited in a single solve.

Precomputed (m,n)→signed maps
-----------------------------

The fixed-boundary update now builds dense mapping matrices once per solve to
convert ``(m,n>=0)`` force blocks into signed Fourier updates via matmul. This
reduces scatter-heavy updates inside the iteration loop and keeps the JIT graph
more regular.

Batched sin conversions
-----------------------

The scan update now batches the Z/L ``(m,n)`` sin-block conversions into a
single matmul-based mapping, reducing kernel count compared to converting each
field independently.

Scatter-free boundary/axis enforcement
--------------------------------------

The fixed-boundary/axis enforcement step now uses concatenation instead of
scatter updates for the edge and axis rows. This trims scatter-heavy kernels in
the scan loop without changing the VMEC constraints.
Axis m=0 masks are now reused from ``VMECStatic`` to avoid per-iteration mask
construction.

Lambda gauge masking
--------------------

The (m,n)=(0,0) lambda gauge constraint now uses a boolean mask instead of a
scatter update, trimming another small scatter kernel from the iteration loop.

Vectorized axis blending
------------------------

Initial-guess axis blending now updates all ``m=0`` Fourier columns in one
vectorized scatter instead of looping over toroidal modes. This reduces
index-heavy overhead during startup.

Cached mode scaling
-------------------

``VMECStatic`` now caches the per-mode internal scaling factors
``1/(mscale*nscale)`` so initial-guess construction avoids repeated gathers
from the trig tables.

Avoid Python objects in jitted functions
----------------------------------------

JAX ``jit`` requires inputs to be arrays or PyTrees. ``vmec-jax`` makes the key
containers PyTrees:

- ``VMECState``
- ``HelicalBasis``
- ``Geom``

If you build your own containers, follow the same approach.

Memory considerations
---------------------

The current Fourier implementation stores ``(K, ntheta, nzeta)`` basis tensors
for cos/sin phases. This is acceptable for low-resolution validation cases, but
will become heavy for larger ``mpol/ntor``.

Planned upgrades (post-parity):

- factorized DFTs (theta/phi separable) using precomputed trig/weight tables,
- FFT-based angular transforms only if they reproduce VMEC scaling and weights,
- chunked evaluation in ``theta``/``zeta`` to reduce peak memory.
