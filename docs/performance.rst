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

Scan-mode iteration (fast path)
-------------------------------

For pure-performance runs (no VMEC2000 control logic), pass ``use_scan=True`` to
``run_fixed_boundary``. This moves the outer iteration loop into ``jax.lax.scan``
and eliminates most Python control-flow + host/device syncs.
Alternatively, use solver ``vmec2000_iter_fast`` (alias ``vmec2000_scan``), which
enables scan mode automatically.
You can also set ``VMEC_JAX_USE_SCAN=1`` to force scan mode for VMEC-style runs.

Important:

- ``use_scan=True`` disables VMEC2000 control features (restart triggers, strict
  update logic, and VMEC-style preconditioner caches). Use it for speed, not
  per-iteration parity.
- Debug dump env vars are incompatible with scan mode.

Recent profiling snapshot (QA, 3 iterations on CPU)
---------------------------------------------------

- Default loop: ~0.26s total wall time (post-warmup).
- Scan loop: ~0.083s total wall time (~3x faster for this short run).

Longer runs benefit more because Python control-flow overhead scales with the
iteration count in the non-scan path.

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
``VMEC_JAX_PROFILE_DIR`` before invoking ``vmec_jax``. The trace is written in
TensorBoard/Chrome trace format::

  VMEC_JAX_PROFILE_DIR=/tmp/vmec_jax_trace \\
    vmec_jax examples/data/input.ITERModel --max-iter 3 --no-multigrid --no-use-input-niter --quiet

Recent traces show that the pre-iteration time is dominated by JIT
compilation/cache misses (``pjit cache_miss`` + backend compile) rather than
the nonlinear iteration itself. This is expected for short runs on CPU.
For repeated runs, the compilation cache (``VMEC_JAX_COMPILATION_CACHE_DIR``)
can significantly reduce this overhead once the cache is warm.

Batched radial smoothing
------------------------

The scan path now batches the radial tridiagonal smoother across the R/Z
components (and separately for lambda) so the solver does fewer tridi solves per
iteration. This reduces kernel count and Python overhead while preserving the
VMEC update math.

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
