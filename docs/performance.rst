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

Important:

- ``use_scan=True`` disables VMEC2000 control features (restart triggers, strict
  update logic, and VMEC-style preconditioner caches). Use it for speed, not
  per-iteration parity.
- Debug dump env vars are incompatible with scan mode.

Static precomputation
---------------------

Use ``VMECStatic`` to avoid rebuilding:

- mode tables,
- angle grids,
- Fourier basis tensors,
- radial grid.

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
