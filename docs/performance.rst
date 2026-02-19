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

For pure-performance runs (no VMEC2000 control logic), pass ``use_scan=True`` to
``run_fixed_boundary``. This moves the outer iteration loop into ``jax.lax.scan``
and eliminates most Python control-flow + host/device syncs.
Alternatively, use solver ``vmec2000_iter_fast`` (alias ``vmec2000_scan``), which
enables scan mode automatically.
You can also set ``VMEC_JAX_USE_SCAN=1`` to force scan mode for VMEC-style runs.
For a one-line opt-in, pass ``performance_mode=True`` to ``run_fixed_boundary``,
which forces scan mode and disables VMEC2000 control logic.

Important:

- ``use_scan=True`` disables VMEC2000 control features (restart triggers, strict
  update logic, and VMEC-style preconditioner caches). Use it for speed, not
  per-iteration parity.
- Debug dump env vars are incompatible with scan mode.

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

Note: scan mode bypasses VMEC2000-style control logic; the VMEC++ restart flag
only affects the VMEC2000-style iteration loop.

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
