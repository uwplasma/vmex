Performance records
===================

This page collects the measured wall time, iteration counts and memory of the
solver and of the force-balance polish.  Every number on it quotes a committed
record under ``benchmarks/`` and names the VMEX version, commit or date that
record was measured at; a number without a record is not stated.  Hosts differ
between records, so the quantity that carries across machines is a ratio
between codes on one host, not an absolute time.  The most current speed
record is ``benchmarks/fresh_decks_vs_vmec2000_2026-09-02.json`` (VMEX 0.8.1,
Apple M4): on six decks VMEX had never been run on, it took 0.50–1.34x the
VMEC2000 wall time with a warm persistent compilation cache and 0.60–3.13x
with the cache cleared.

Fresh decks against ``xvmec2000`` (VMEX 0.8.1, 2026-09-02)
------------------------------------------------------------

Six fixed-boundary decks VMEX had never been benchmarked on (``nfp = 1`` to
``6``, tokamak and stellarator, vacuum and finite beta with net current) were
run through ``vmex`` and the serial Fortran ``xvmec2000`` on an Apple M4 with
JAX 0.11.1 in float64, at VMEX commit ``8ef81c44``.  Cold is a fresh process
with the persistent compilation cache removed; warm is a fresh process
reusing it; each run used a fresh directory.  The record, with reference
binary and deck hashes, is
``benchmarks/fresh_decks_vs_vmec2000_2026-09-02.json``, with a per-deck
narrative in the companion ``.md``.  Iterations are listed as xvmec2000 /
vmex.

=========================  =============  ================  =============  =============  =============  ===========
deck                       nfp/mpol/ntor  ns ladder         xvmec2000 [s]  vmex cold [s]  vmex warm [s]  iterations
=========================  =============  ================  =============  =============  =============  ===========
ITER model (tokamak)       1/12/0         13→201, 6 levels  6.7            21.0           9.0            1469 / 1470
ESTELL                     2/6/5          9→65, 4 levels    23.1           24.0           14.2           2301 / 2301
ARIES-CS n3are             3/9/5          16/49             5.1            10.6           5.8            1496 / 1496
HSX QHS (vacuum)           4/10/10        11→201, 7 levels  162.3          97.2           81.7           1575 / 1575
W7-X standard              5/10/10        13/25/51          9.2            14.9           8.0            1105 / 1105
Nührenberg–Zille 1988 QHS  6/9/5          16/51             6.4            11.5           6.5            1843 / 1843
=========================  =============  ================  =============  =============  =============  ===========

On every deck VMEX follows the Fortran trajectory: the same per-level
iteration counts (the ITER model differs by one iteration at its 1e-18
floor) and the same Jacobian-reset counts on the five decks that record
them.  The largest recorded differences are ``1.4e-10`` relative in
``iotaf`` (ARIES-CS), ``2.2e-13`` relative in beta, and ``2.8e-17``
absolute in the boundary harmonics (W7-X).  Warm, VMEX took 0.50x (HSX) to
1.34x (ITER model) the Fortran wall time.  The cold gap is XLA compilation:
on the ITER model the compile census counts 12.65 s across 650 programs, six
of them ``_block_lane`` compiles (one per ``NS_ARRAY`` level), against a
12.0 s cold-minus-warm difference.  That cost is paid once per machine and
JAX version.

Numerical reproducibility
-------------------------

Everything runs in float64 (``jax_enable_x64`` is mandatory).  Two solves of
the same input on the same machine, JAX version and device are
bit-identical, and persistent-cache hits reload byte-identical executables,
so cached and freshly compiled runs agree bit for bit.  Across VMEX versions
the algorithm is fixed but the compiled graph is not: a restructured traced
lane changes XLA's fusion and hence the order of floating-point reductions,
so a trajectory can drift from an earlier release in the last places while
keeping its iteration count.  The golden, restart, step-control and parity
suites pin that tolerance class.  The batched tridiagonal solver used on
accelerators is numerically equivalent to, not bit-identical with, the CPU
Thomas sweep.

Bundled-deck baseline (VMEX 0.3.0, 2026-07-26)
------------------------------------------------

Wall times in seconds, with every deck's final ``NS_ARRAY`` stage ramped to
``ns = 201``.  "Cold" is a fresh process including JIT compilation; "warm" is
a second in-process solve reusing the compiled executable, the case inside
an optimization loop.  The table is rendered from ``benchmarks/baseline.json``
by ``tools/render_performance_docs.py``.  That record was measured on
2026-07-26 at commit ``314e5ba5`` (VMEX 0.3.0) on an Apple-Silicon host, one
fresh process per code run sequentially; its ``_provenance`` block states that
runtime package versions were not recorded, and it has not been re-measured
since.  Read it as a VMEX 0.3.0 result; the fresh-deck record above is the
current comparison.

.. begin generated-baseline-table (tools/render_performance_docs.py)

.. list-table::
   :header-rows: 1
   :widths: 34 14 14 14 14

   * - case
     - VMEC2000
     - vmex cold
     - vmex warm
     - VMEC++
   * - li383_low_res
     - 0.86
     - 3.36
     - **0.434**
     - 0.341
   * - solovev
     - 1.07
     - 3.23
     - **0.319**
     - 0.845
   * - circular_tokamak
     - 1.35
     - 4.12
     - **0.522**
     - 1.26
   * - nfp4_QH_warm_start
     - 1.42
     - 3.51
     - **0.641**
     - 0.782
   * - nfp4_QH_warm_start (multigrid)
     - 1.48
     - 11.9
     - **0.787**
     - 1.05
   * - DSHAPE
     - 1.83
     - 6.27
     - **0.812**
     - 1.87
   * - cth_like_fixed_bdy
     - 6.04
     - 6.65
     - **3.5**
     - failed
   * - cth_like_fixed_bdy (multigrid)
     - 7.08
     - 17.3
     - **4.57**
     - failed
   * - cth_like_free_bdy
     - 20.2
     - 28.9
     - **13**
     - 6.36
   * - LandremanPaul2021_QA_lowres
     - 34.7
     - 28.6
     - **22.2**
     - 12.2
   * - LandremanPaul2021_QA_lowres (multigrid)
     - 55.8
     - 45.2
     - **35.9**
     - 16.5
   * - LandremanPaul2021_QH_reactorScale_lowres
     - 61.7
     - 45.7
     - **38.4**
     - failed
   * - NuhrenbergZille_1988_QHS
     - 106
     - 98.6
     - **74.6**
     - 45.7
   * - cth_like_free_bdy_lasym_small
     - 154
     - 133*
     - **105**
     - failed

Bold marks vmex warm beating VMEC2000 (14 of 14 rows).
``*`` marks an equal-iteration-budget run whose CLI exit was nonzero
(the deliberately NITER-bounded LASYM stress row: both codes exhaust
the same budget, so the wall times compare equal work); ``failed``
marks an aborted run and ``n/a`` an unsupported configuration.

.. end generated-baseline-table

VMEC++ ran through its Python API in a fresh process on the same host; its
``failed`` rows aborted during the first iterations.  The record also carries
each run's iteration count and peak RSS.

Free-boundary multigrid parity (commit ``b0cc789e``, 2026-07-24)
------------------------------------------------------------------

``benchmarks/freeboundary_multigrid.json`` runs the public converged
CTH-like ``NS_ARRAY = 7, 15`` ladder on an Apple-Silicon CPU (measurement
commit ``b0cc789e``, VMEX 0.3.0; the record states that the VMEC2000
executable hash was not recorded).  VMEC2000 takes 239 + 340 iterations in
0.92 s; VMEX takes 250 + 340 iterations, 8.61 s cold and 1.20 s warm.  Both
turn the vacuum on exactly once and enter the fine grid at the same raw
residual, ``FSQR = 1.73``.  Against the VMEC2000 wout, the final
scale-relative maximum differences are ``6.08e-5`` (R), ``3.59e-4`` (Z),
``1.99e-6`` (iota) and ``6.07e-8`` (relative ``wb``).

``tests/test_parity_breadth.py`` gates iteration counts against the golden
VMEC2000 runs within ``+-25%`` (the count moves with the floating-point
path) and asserts ``wb`` within 1e-7 relative and ``rmnc/zmns`` and
``iotaf`` at rtol 1e-5; the physics comparisons are in
:doc:`/explanation/validation`.

The figure plots the total force residual ``fsqr + fsqz + fsql`` per
iteration for ``nfp4_QH_warm_start`` on a single grid at ``ns=51``: 502
iterations for VMEX and VMEC2000, 501 for VMEC++.  The traces are in
``benchmarks/convergence_nfp4_ns51.json``, last written on 2026-07-29 (PR
#80); host and JAX version were not recorded.

.. figure:: /_static/figures/readme_convergence.webp
   :alt: force residual vs iteration for VMEX, VMEC2000, and VMEC++
   :align: center
   :width: 95%

   Force residual vs iteration on ``nfp4_QH_warm_start`` at ``ns=51``
   (``benchmarks/make_readme_figures.py --only convergence``).

GPU (VMEX 0.9.1, 2026-09-16)
----------------------------

``benchmarks/gpu_a4000_2026-09-16.json`` re-measured the CPU-versus-GPU
sweep on two NVIDIA RTX A4000s with JAX 0.11.1 at the VMEX 0.9.1 commit.
The GPU won no cell: warm gain ran 0.17x to 0.83x across every shipped deck
and every point of the synthetic ``ns x mnmax`` scan, including
``NuhrenbergZille_1988_QHS`` at 111 s of warm CPU work.  It replaces the
2026-07-09 record ``benchmarks/gpu_baseline.json`` (jax 0.6.2), whose
throughput crossover does not transfer.  The placement policy and how to
measure on your own hardware are in :doc:`/howto/run-on-gpu` and
:doc:`/explanation/architecture`.

Persistent compilation cache (2026-09-03)
-----------------------------------------

JAX re-scans the whole cache directory on every write: ``put`` globs each
entry, stats it and reads its access-time sidecar under the directory-wide
lock, so the cost grows with the number of resident entries.
``benchmarks/cache_entry_scaling_m4_2026-09-03.json`` (commit ``2d3be2c0``,
Apple M4, JAX 0.11.1; the record notes other work was running, so its wall
times are upper bounds) measures 0.028 ms per resident entry: a write costs
5.7 ms at 250 entries and 304 ms at 10880.  On one bundled deck a solve took
7.46 s against an empty cache and 31.3 s against a 10880-entry cache,
25.623 s of it in cache writes.

JAX's own bound is on bytes and these entries are small, so it never fires.
VMEX therefore trims the directory to its least-recently-used ``1024``
entries once per process, before anything writes.  With that bound the same
solve took 11.48 s, and a warm rerun 3.14 s with 289 of 289 lookups hitting.
Entries used within the last 24 hours survive the trim up to four times the
bound, so a large workload keeps its own working set.  Set
``VMEX_CACHE_MAX_ENTRIES`` to change the bound, or to ``0`` to disable
trimming.

With jaxlib < 0.10 the cache defaults to off, because those releases crash
deserializing large cached CPU executables; ``VMEX_COMPILATION_CACHE=1``
forces it on.

High-mode FFT synthesis (commit ``ecfbe31d``, 2026-07-28)
---------------------------------------------------------

``use_fft=None`` (the default) selects the separable toroidal FFT synthesis
only above 512 modes on accelerators and ARM CPUs; smaller problems and x86
CPUs keep the dense real contraction, and the implicit-differentiation path
always uses the dense lanes.  An explicit ``use_fft=True`` or ``False``
always wins.  ``benchmarks/high_mode_fft.json`` (Apple M4, JAX 0.10.2,
measurement commit ``ecfbe31d``, VMEX 0.3.0) compares the two on the
537-mode CTH-like case: peak RSS 8.21 GB with FFT against 9.63 GB dense,
983 s against 826 s cold, and 335 s for both warm.  Both runs stop at the
2500-iteration cap, so they are compared on cost, not on a converged answer.

2D block preconditioner (VMEX 0.8.1, 2026-09-03)
------------------------------------------------

The default 1D radial preconditioner is what reproduces VMEC2000
iteration for iteration.  For stiff decks, such as very high aspect ratio or
strong finite-β coupling, the opt-in 2D block preconditioner
(:mod:`vmex.core.preconditioner_2d`) replaces the radial-only approximation
with a matrix-free Newton step: a Jacobian-vector-product Hessian applied
through GMRES.  The default 1D path is unchanged by its presence.  The rows
come from ``benchmarks/preconditioner_2d_stiff_cases.json`` (commit
``8b1c5ffe``, Apple M3 Max, JAX 0.9.2, measured 2026-09-03), and
``tests/test_figure_provenance.py`` fails when this table drifts from it.
The iteration count falls 5.4x to 10.9x on these three cases.

.. list-table::
   :header-rows: 1
   :widths: 34 16 16 16 18

   * - stiff case
     - 1D radial
     - 2D block
     - reduction
     - ``wb`` agreement
   * - aspect-100 tokamak, ns=51
     - 97
     - 18
     - 5.4x
     - 3.6e-11
   * - aspect-100 tokamak, ns=101
     - 163
     - 15
     - 10.9x
     - 3.8e-11
   * - nfp4 QH, finite beta, ns=51
     - 1885
     - 246
     - 7.7x
     - 5.7e-7

.. figure:: /_static/figures/readme_precond.webp
   :alt: 2D vs 1D preconditioner iteration counts on stiff cases
   :align: center
   :width: 90%

   Iterations to converge, 2D block vs 1D radial preconditioner
   (``benchmarks/make_readme_figures.py --only precond``).

It is opt-in on purpose.  Fewer iterations is not fewer seconds: each 2D
Newton step is a GMRES solve over Hessian-vector products and costs more than
a 1D radial sweep.  The record's wall times include compilation and its
protocol states they are not a speed claim, so this page makes none.  The
converged ``wb`` matches the 1D result to the agreement column, so the
preconditioner changes the path, not the fixed point.  Reach for it when the
1D iteration count is the bottleneck or stalls.

One such stall: on the aspect-100 case at ``ns=51``, ``FTOL=1e-11``,
``PRECON_TYPE='GMRES'`` and ``PREC2D_THRESHOLD=1e-6``, the opt-in live test
``test_live_vmec2000_exact_jvp_gmres_robustness`` asserts that VMEC2000's
finite-difference block GMRES stops between 1e-10 and 1e-8 and asks for more
``PRE_NITER``, while VMEX converges below 1e-11 in fewer iterations than a
VMEC2000 1D solve and matches its ``wb`` to 1e-8 relative.

High-order strong-force kernel (VMEX 0.7.0, 2026-08-28)
-------------------------------------------------------

``benchmarks/strong_force.py`` measures the independent continuum oracle,
pointwise ``J x B - grad(p)``, and its reverse-mode coefficient gradient.
The record ``benchmarks/strong_force_m4.json`` (commit ``9481f64a``, arm64
macOS, JAX 0.11.1) disabled the persistent compilation cache and used
float64, five Fourier modes, eight radial elements, 64 points and 20 warm
repeats:

======  ==============  ===============  =============  ==============  ================
degree  cold force [s]  warm force [ms]  cold grad [s]  warm grad [ms]  d2/drho2 L2 err
======  ==============  ===============  =============  ==============  ================
3       1.43            0.290            1.84           0.552           2.87e-3
**5**   2.12            0.458            2.63           0.853           **7.73e-7**
7       2.90            0.720            3.43           1.23            6.09e-10
======  ==============  ===============  =============  ==============  ================

The accuracy column reconstructs ``rho^2 exp(s)`` and compares its second
``rho`` derivative at 2001 points.  Degree 5, the default of
:func:`~vmex.core.strong_force.lift_high_order_state` (the polish driver
defaults to 3), keeps warm force and gradient evaluation below a
millisecond; degree 7 is available for
p-refinement and certification at a larger compile footprint (the record's
``peak_rss_increase_mib`` fields).

High-order low-physics preconditioner (VMEX 0.7.0, 2026-08-28)
--------------------------------------------------------------

``benchmarks/polish_preconditioner.py`` measures the high-to-low transfer,
one stored exact raw-force block factor, and forward and transpose
high-order applications.  The record ``benchmarks/polish_preconditioner_m4.json``
(commit ``7bb306e0``, arm64 macOS, JAX 0.11.1) disabled the persistent
compilation cache and used float64 and 20 warm repeats:

==  ====  ====  ==========  =================  =================  ===================  ===============
ns  mpol  ntor  factor [s]  cold forward [ms]  warm forward [ms]  warm transpose [ms]  factor RSS [MiB]
==  ====  ====  ==========  =================  =================  ===================  ===============
5   3     0     4.61        98.1               0.0274             0.0269               193
7   4     0     4.44        121                0.0289             0.0291               201
5   3     1     5.58        159                0.0306             0.0336               217
==  ====  ====  ==========  =================  =================  ===================  ===============

Across these cases the transfer round trip is below ``1.2e-15``,
forward/transpose duality below ``2.6e-15``, and the factored low-block
residual below ``4.8e-12``.  Factor construction includes JAX assembly and
compilation and dominates a first use, so factors are kept across Krylov
steps and continuation stages until the quality policy requests a refresh.
The table is an overhead gate at structural resolution, not a scaling claim.

Collocation-polish derivative gate (VMEX 0.7.1, 2026-08-29)
-----------------------------------------------------------

``benchmarks/polish_implicit.py`` measures matrix-free implicit-function
tangents, adjoints, and the custom VJP of the least-squares stationarity
equation the polish solves.  The record ``benchmarks/polish_implicit_m4.json``
(commit ``e176b1ac``, arm64 macOS, JAX 0.11.1, persistent cache disabled)
uses a 17-coordinate Solov'ev structural case whose primal reaches relative
optimality ``1.13e-7`` in nine steps.

- Warm medians over ten repeats: 6.44 ms tangent, 6.83 ms adjoint, 6.61 ms
  custom VJP.  Cold compile-plus-execute: 7.13 s, 7.50 s and 9.43 s.
- Incremental peak RSS, compilation included: 52.2 MiB, 156.4 MiB and
  237.9 MiB.
- Tangent and adjoint each take 17 Krylov iterations; their dot-product
  mismatch is ``1.90e-10``, and the custom VJP agrees with the explicit
  adjoint to ``8.75e-21`` relative squared error.
- For the relative field-strength variance at ``rho=0.7``, the implicit
  directional derivative agrees with two re-polished finite-difference
  endpoints to ``5.11e-5`` relative error; those two solves take 21.22 s
  against the 6.61 ms warm gradient.

Polish memory at production stellarator resolution (VMEX 0.8.1, 2026-09-03)
---------------------------------------------------------------------------

``benchmarks/polish_memory.py`` runs the polish setup three times on one
build, changing only how the independent force sweep is scheduled, and
records each arm's peak resident memory from ``os.wait4`` so an arm the OS
kills still reports one.  The record is ``benchmarks/polish_memory_w7x.json``
(commit ``529f1789``, x86_64 Linux CPU, JAX 0.9.2), on the W7-X standard
configuration at ``MPOL = NTOR = 10``, ``ns = 51``, the resolution at which
polishing was reported to run out of memory.

- ``flat``, the pre-0.8.2 sweep (one ``vmap`` over every evaluation point),
  peaks at 34.2 GiB on the first certificate and exits at the chart stage.
- ``batched`` schedules the same per-point kernel in automatically sized
  batches.  Its certificate peaks at 3.05 GiB, but without checkpointing the
  chart build still stores whole-grid linearization residuals and the arm
  exits there too.
- ``auto``, the shipped policy, also checkpoints the kernel so reverse-mode
  passes stay per batch: 3.01 GiB at the certificate, 15.4 GiB at the chart,
  and it completes.

The certificate's absolute L2 force error agrees across the three arms to 14
significant digits; only the schedule differs.  This is a memory record:
the batched arms trade time for memory, and the record's wall times include
that trade.

Polish cost prediction (VMEX 0.8.1, 2026-09-03)
-----------------------------------------------

``benchmarks/polish_cost.py`` records, per deck, what one Gauss--Newton
linear product costs and what the configured iteration limits allow in the
worst case.  These measurements are behind
``PolishConfig.auto_budget_seconds``, the ceiling ``POLISH = AUTO`` prices a
solve against before committing to it.  They are machine-specific, which is
why AUTO measures at run time.  The record is
``benchmarks/polish_cost_office.json`` (commit ``529f1789``, AMD 36-core
x86_64 Linux CPU, JAX 0.9.2) at driver defaults, 80 nonlinear iterations of
up to 600 linear products: the shaped tokamak prices at 1 126 s and the
bundled Solov'ev at 501 s, both inside the default 3 600 s budget, while the
finite-beta QA case prices at 87 848 s and is the deck AUTO turns away.

Historical and generated records
--------------------------------

``benchmarks/baselines/m4/`` holds historical schema-1 workflow records; their
aggregate warm timing repeats only the first stage, so they do not establish
complete-workflow performance.  ``benchmarks/INDEX.md`` lists every committed
record with the script that generates it.
