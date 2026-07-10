Performance and validation
==========================

This page summarizes the measured performance and parity status of the core
solver. All numbers come from checked-in benchmark artifacts —
``benchmarks/baseline.json`` (CPU suite, regenerated with
``benchmarks/run_baseline.py``) and ``benchmarks/gpu_baseline.json`` (GPU
matrix, ``benchmarks/run_gpu_matrix.py``; 2x NVIDIA RTX A4000, jax 0.6.2
cuda12) — and from the end-to-end parity suite in
``tests/core_new/test_parity_breadth.py``.

Benchmark suite (CPU)
---------------------

Wall times in seconds; "cold" is a fresh process including JIT compilation,
"warm" is a second in-process solve reusing the compiled executable (the
number that matters inside optimization loops, where the structural
executable cache makes every solve after the first warm).

.. list-table::
   :header-rows: 1
   :widths: 30 8 14 14 14 14

   * - case
     - ns
     - VMEC2000
     - vmec_jax cold
     - vmec_jax warm
     - VMEC++
   * - solovev
     - 11
     - 0.10
     - 3.4
     - **0.013**
     - 0.07
   * - DSHAPE (multigrid)
     - 128
     - 1.13
     - 10.6
     - **0.42**
     - 1.50
   * - circular_tokamak
     - 201
     - 0.18
     - 5.8
     - **0.046**
     - 0.40
   * - cth_like_fixed_bdy
     - 15
     - 0.31
     - 5.3
     - **0.026**
     - failed
   * - li383_low_res
     - 16
     - 0.12
     - 6.1
     - **0.10**
     - 0.09
   * - LandremanPaul2021_QA_lowres (multigrid)
     - 50
     - 6.0
     - 19.9
     - **5.7**
     - 2.9
   * - LandremanPaul2021_QH_reactorScale_lowres
     - 75
     - 8.3
     - 19.6
     - **8.2**
     - failed
   * - nfp4_QH_warm_start
     - 35
     - 0.33
     - 7.1
     - **0.13**
     - 0.41
   * - NuhrenbergZille_1988_QHS
     - 51
     - 121.7
     - 162.9
     - 123.9
     - 48.8
   * - cth_like_free_bdy_lasym_small (free boundary)
     - 15
     - 1.9
     - 19.8
     - **7.7**
     - failed

Reading the table:

- **Cold** runs are dominated by XLA compilation (~2-9 s), not physics; the
  persistent compilation cache removes most of it on subsequent processes.
- **Warm** solves are typically 2-10x faster than VMEC2000 on small and
  medium decks and competitive on the largest ones.
- VMEC++ rows marked *failed* aborted during the first iterations on those
  decks; ``vmec_jax`` converges on the full suite (zero-crash policy).

Parity with VMEC2000
--------------------

Per-iteration algorithmic parity (same step control, preconditioner cadence,
constants) means the solver does not just reach the same answer — it takes
the *same number of iterations* as VMEC2000 on the benchmark decks:

.. list-table::
   :header-rows: 1
   :widths: 36 16 16 32

   * - case
     - VMEC2000 iters
     - vmec_jax iters
     - notes
   * - solovev
     - 215
     - 215
     - exact match
   * - DSHAPE (multigrid 16/32/64/128)
     - 908
     - 903
     -
   * - circular_tokamak (multigrid 10/17)
     - 368
     - 368
     - exact match
   * - cth_like_fixed_bdy
     - 434
     - 434
     - exact match
   * - nfp4_QH_warm_start (ns=35)
     - 450
     - 450
     - exact match
   * - LandremanPaul2021_QA_lowres
     - 1000
     - 1000
     - golden run is NITER-capped at its FTOL 1e-13
   * - LandremanPaul2021_QH_reactorScale_lowres
     - 2408
     - 2406
     -
   * - up_down_asymmetric_tokamak (lasym)
     - 2000 (capped)
     - 1951
     - both stopped at the matched residual 1.5e-13; a fully converged
       VMEC2000 rerun (fsq ~1e-16) matches the core to <= 7.3e-7 on every
       checked harmonic, in 3197 vs 3118 iterations
   * - li383_low_res (single grid, ns=16)
     - 123
     - within the ±25% parity gate
     -

.. figure:: _static/figures/readme_parity.png
   :alt: iteration parity of vmec_jax against VMEC2000 golden runs
   :align: center
   :width: 95%

   Iteration-for-iteration parity against the golden VMEC2000 fixtures
   (regenerated with ``benchmarks/make_readme_figures.py --only parity``).

Parity holds not just at the converged endpoint but along the whole
trajectory.  The trace below runs the quick-start QH case
(``nfp4_QH_warm_start``, single grid at ``ns=51``) through all three codes
and plots the total force residual ``fsqr + fsqz + fsql`` per iteration:
the vmec_jax curve lies exactly on top of VMEC2000's (both converge in 502
iterations), and VMEC++ follows a near-identical path (501 iterations).
The vmec_jax trace comes from ``SolveResult.fsq_history``, the VMEC2000
trace from its stdout iteration table run with ``NSTEP = 1``, and the
VMEC++ trace from the ``fsqt`` array of its wout payload.

.. figure:: _static/figures/readme_convergence.png
   :alt: force residual vs iteration for vmec_jax, VMEC2000, and VMEC++
   :align: center
   :width: 95%

   Force residual vs iteration on ``nfp4_QH_warm_start`` at ``ns=51``
   (``benchmarks/make_readme_figures.py --only convergence``; traces cached
   in ``benchmarks/convergence_nfp4_ns51.json``).

The parity suite additionally asserts, per case: convergence at the deck's
``ftol``; ``wb`` within 1e-7 relative of the golden wout; boundary/interior
``rmnc/zmns`` harmonics at rtol 1e-5; and ``iotaf`` at rtol 1e-5. Where the
golden VMEC2000 run is itself NITER-capped (LandremanPaul QA, the lasym
tokamak), both codes are stopped at a matched residual and the documented
absolute tolerances cover the golden run's own remaining non-convergence.
wout files are compared per-variable with CompareWOut-style combined
rel+abs tolerances.

GPU guidance
------------

Measured behavior (``benchmarks/gpu_baseline.json``):

- **Per-iteration throughput favours the GPU at every tested size** (0.83 ms
  vs 1.90 ms per iteration at ``ns=35, mpol=2, ntor=2``; up to ~3x on
  NuhrenbergZille-class decks: 90 s vs 277 s wall).
- **The GPU pays fixed per-solve overheads** (~0.2-0.4 s dispatch/transfer
  floor plus compile or cache-load in cold processes), so small decks that
  finish in well under a second of CPU work stay faster on the CPU
  (``solovev``: 0.043 s CPU vs 0.29 s CUDA warm).

Device policy
~~~~~~~~~~~~~

:mod:`vmec_jax.core.device` encodes this as a default placement rule using
the per-iteration work proxy ``ns * mnmax * nznt`` (the cost driver of the
batched-matmul transforms): below ``GPU_MIN_ITERATION_WORK = 100_000`` the
solve stays on the CPU, above it the GPU is used. The policy is a *default*
only:

- an explicit ``device=`` argument to ``solve``/``solve_multigrid`` always
  wins;
- if you pinned the platform yourself via ``JAX_PLATFORMS`` (or
  ``JAX_PLATFORM_NAME``), the automatic policy stands down entirely.

.. code-block:: bash

   JAX_PLATFORMS=cpu  vmec input.solovev      # force CPU
   JAX_PLATFORMS=cuda vmec input.big_case     # force GPU

Persistent compilation cache
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``vmec-jax`` enables JAX's persistent XLA compilation cache on accelerators,
so the multi-second compile cost is paid once per machine, not once per
process.

.. warning::

   **cwd-shadowing pitfall.** Running ``python`` with a working directory
   that contains a ``vmec_jax`` source checkout can shadow the installed
   package as a namespace package: ``vmec_jax/__init__.py`` never runs, the
   persistent compilation cache is never enabled, and every solve pays the
   full XLA recompile (measured ~7 s vs ~1.7 s warm on CUDA for solovev).
   If GPU runs are mysteriously slow, check that
   ``python -c "import vmec_jax; print(vmec_jax.__file__)"`` points where
   you expect.

Float64 is required (enforced at solver import). On GPUs this means fp64
arithmetic, but the solve is latency- rather than FLOP-bound at benchmark
sizes: the tridiagonal preconditioner solve, for instance, measures identical
fp32/fp64 GPU times (~15 us per radial row, independent of the number of
spectral columns).

Reproducing the numbers
-----------------------

.. code-block:: bash

   python benchmarks/run_baseline.py       # CPU suite -> benchmarks/baseline.json
   python benchmarks/run_gpu_matrix.py     # GPU matrix -> benchmarks/gpu_baseline.json
   pytest tests/core_new/test_parity_breadth.py   # end-to-end parity suite

The parity suite needs the golden VMEC2000 fixtures (fetched release assets);
it is skipped automatically when they are unavailable.
