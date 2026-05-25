Free-Boundary Plan
==================

This document is the implementation and validation summary for
VMEC2000-parity-targeted free-boundary capability in ``vmec-jax`` while
preserving:

- fixed-boundary parity and defaults,
- selected differentiable JAX paths today, with full free-boundary/NESTOR
  adjoints as the research target,
- profiled CPU performance on the supported public paths,
- bounded memory usage.

Implementation status snapshot (March 2026)
-------------------------------------------

This page records the free-boundary implementation plan and historical
milestone status.  Use :doc:`validation` and :doc:`quickstart` for the current
supported free-boundary workflow.

WP0 is implemented:

- typed free-boundary input config parsed from ``&INDATA``:
  ``LFREEB``, ``MGRID_FILE``, ``EXTCUR``, ``NVACSKIP``,
- VMEC2000-aligned defaults:
  ``LFREEB=T`` with ``MGRID_FILE='NONE'`` disables free-boundary,
  ``NVACSKIP<=0`` falls back to ``NFP``,
- typed free-boundary runtime state scaffold in ``VMECStatic``,
- mgrid loader skeleton (metadata + optional BR/BP/BZ tensor loading),
- trilinear mgrid field interpolation utility with periodic toroidal angle,
- unit tests for parsing, normalization, and mgrid metadata loading.

WP1 is in place:

- driver now loads and validates mgrid metadata for ``LFREEB=T``,
  with strict checks for ``NFP`` agreement and ``kp % nzeta == 0``.
- solve loop now carries VMEC-style free-boundary control state
  (``ivac``, ``ivacskip``, ``nvacskip``) in diagnostics/resume state.
- free-boundary control now follows VMEC turn-on/ramp semantics:
  delayed vacuum activation (``ivac`` starts negative), forced full updates
  while ``ivac<=2``, and adaptive ``nvacskip`` updates on full solves.
- an external-field sampling hook now exists:
  mgrid trilinear interpolation on the boundary with EXTCUR weighting,
  emitted as ``free_boundary_external_field`` diagnostics.
- VMEC2000-aligned dense vacuum coupling and comparator coverage are in place.

WP2 now has an initial coupling scaffold plus a VMEC2000-like dense path:

- boundary geometry/tangent metric terms are evaluated on the edge surface,
- external cylindrical field is projected to ``Bu/Bv`` and inverted to
  ``B^u/B^v`` via the 2x2 boundary metric,
- ``bsqvac`` proxy and boundary-normal channel summaries are emitted in
  ``free_boundary_external_field`` diagnostics.
- ``vmec2000_like_dense_integral`` provides a dense Green-function-like
  boundary operator assembly plus linear solve on the boundary grid.
- ``spectral_poisson_external_only`` remains available as the previous fast
  surrogate model.
- mode selection is controlled by ``VMEC_JAX_FREEB_NESTOR_MODE`` with an
  ``auto`` default; large boundary grids fall back to the spectral path via
  ``VMEC_JAX_FREEB_VMEC_LIKE_MAX_POINTS``.
- VMEC-style ``ivac``/``ivacskip``/``nvacskip`` update vs reuse behavior is
  preserved (funct3d cadence semantics, ``ictrl_prec2d=0`` path).
- On ``ivacskip != 0``, the implementation now reuses the cached operator and
  refreshes only the RHS/solve (instead of freezing ``phi``), aligning closer
  to VMEC2000 ``scalpot`` reuse semantics.
- In VMEC-like dense mode, non-singular source assembly now defaults to the
  Green-function path (``VMEC_JAX_FREEB_USE_GREENF_SOURCE=1`` default) instead
  of direct ``bexni`` projection, improving fouri/source parity on vacuum
  turn-on iterations.
- On ``ivacskip > 0`` reuse steps, vmec-jax now carries cached
  ``gsource/source_sym/bvecNS`` channels in runtime state so reuse follows
  VMEC ``scalpot`` semantics (analytic update + cached non-singular source).
- edge ``bsq`` coupling is now threaded into the force path by overriding the
  half-mesh edge magnetic-pressure term from vacuum channels.
- free-boundary residual assembly now keeps the core VMEC interior stencil by
  default (``include_edge=False``), with edge forcing entering only through the
  dedicated ``rbsq`` terms in ``forces``. A debug-only override remains
  available via ``VMEC_JAX_FREEB_INCLUDE_EDGE=1``.
- the free-boundary comparator now parses and reports additional bextern
  geometry/operator channels (``rub/rvb/zub/zvb``, ``snr/snv/snz``, and
  ``brad_axis/bphi_axis/bz_axis``) to localize late-iteration drift.

The dense operator is the default VMEC2000-aligned vacuum path used for the
current free-boundary parity matrix. Dump-to-dump comparators and manifest
sweeps are used to maintain quantitative agreement in the coupled
``scalpot/vacuum`` channels.

Program goals and parity matrix
-------------------------------

Project-level free-boundary acceptance requires parity and performance across
all major VMEC branches:

1. axisymmetric + non-axisymmetric geometries,
2. ``lasym=False`` and ``lasym=True``,
3. fixed-boundary and free-boundary solves.

To make this auditable and repeatable, the case matrix is now codified in:

- ``tools/diagnostics/parity_manifest.toml``

and executed by:

- ``tools/diagnostics/parity_sweep_manifest.py``

Current manifest coverage:

- fixed-boundary, axisymmetric, ``lasym=False``:
  ``input.circular_tokamak``.
- fixed-boundary, axisymmetric, ``lasym=True``:
  ``input.up_down_asymmetric_tokamak``.
- fixed-boundary, non-axisymmetric, ``lasym=False``:
  ``input.LandremanPaul2021_QA_lowres``, ``input.nfp4_QH_warm_start``.
- fixed-boundary, non-axisymmetric, ``lasym=True``:
  ``input.basic_non_stellsym_pressure``,
  ``input.LandremanSenguptaPlunk_section5p3``.
- free-boundary, non-axisymmetric, ``lasym=False``:
  ``input.cth_like_free_bdy``, ``input.stellcopt``.
- free-boundary, axisymmetric, ``lasym=False``:
  ``input.DIII-D_lasym_false``.
- free-boundary, axisymmetric, ``lasym=True``:
  ``input.DIII-D``, ``input.DIII-D_reset``.

Coverage now includes a bundled free-boundary non-axisymmetric ``lasym=True``
case (``input.cth_like_free_bdy_lasym_small``), and the preserved
``input.cth_like_free_bdy`` smoke fixture now ships with its bundled
``mgrid`` file. The manifest therefore covers the fixed/free,
axisymmetric/non-axisymmetric, and ``lasym`` true/false branches with
self-contained repo fixtures, but strict executable-backed parity remains the
optional VMEC2000 tier described in :doc:`validation`.

The current implementation focus is:

``source_sym -> bvecNS -> amatrix -> potvac -> edge-force channels``

with runtime and memory optimization on the validated parity path.

Current tests and benchmark coverage
------------------------------------

Implemented coverage (``tests/test_free_boundary_wp0.py``):

- parser/default behavior parity for ``LFREEB``, ``MGRID_FILE``, ``NVACSKIP``,
  and indexed ``EXTCUR(i)``,
- mgrid metadata + full tensor load checks from synthetic NetCDF fixtures,
- strict metadata validation (``NFP`` and ``kp % nzeta``),
- trilinear interpolation exactness on synthetic affine fields,
- solver diagnostics plumbing:
  ``free_boundary`` control block + ``free_boundary_external_field`` summary,
- environment-controlled external-field sampling disable path.

Micro-benchmark coverage:

- ``tools/benchmarks/bench_free_boundary_wp1.py`` measures metadata validation,
  full field load, and interpolation throughput on random boundary-like points.
- Intended use is regression-style performance tracking during WP2/WP3
  integration; it is not a physics acceptance benchmark.

WP2 dump-to-dump alignment harness
----------------------------------

A dedicated free-boundary comparator is now available:

- ``tools/diagnostics/vmec2000_exec_freeb_scalpot_compare.py``

It runs VMEC2000 with ``VMEC_DUMP_SCALPOT=1``, ``VMEC_DUMP_BEXTERN=1``,
``VMEC_DUMP_FOURI=1`` and vmec-jax with
``VMEC_JAX_DUMP_SCALPOT=1`` on the same input, then reports deltas for:

- scalpot RHS vector (VMEC mode space vs vmec-jax projected mode space),
- scalpot matrix (VMEC LU-space matrix vs vmec-jax projected dense operator),
- vacuum boundary ``bsqvac`` channel.
- fouri non-singular source channels (``gsource``, ``source_sym``, ``bvecNS``).

When VMEC2000 includes ``VMEC_DUMP_BEXTERN`` support, the comparator also
reports upstream source-channel deltas:

- ``bexu`` / ``bexv`` (covariant external tangential channels),
- ``bexn`` / ``bexni`` (normal source channels used by ``scalpot``).
- ``bexn`` decomposition channels (``bexn_term_r``, ``bexn_term_phi``,
  ``bexn_term_z``) from ``brad*snr``, ``bphi*snv``, ``bz*snz`` to localize
  source drift to geometry vs field contributions.
- free-boundary edge-coupling channels from ``funct3d``/``forces``
  (``pgcon``, ``rbsq``, ``dbsq``, ``bsqvac``, ``p1e/p1o``,
  ``pzu0/pru0``) plus an inferred ``ohs`` check to flag multigrid stage
  misalignment in comparisons. On the vmec-jax side, ``dbsq`` is reported as a
  diagnostic proxy from ``gcon -`` extrapolated plasma ``bsq`` so VMEC2000
  ``DEL-BSQ`` failures can be localized before full WOUT promotion.

The comparator now also caps VMEC ``NITER_ARRAY`` stages to the requested
``--max-iter`` budget when multigrid is active, so VMEC and vmec-jax dumps are
generated from the same stage window during short turn-on diagnostics.
It also accepts ``--activate-fsq`` to force vmec-jax active vacuum coupling
during intentionally short dump-to-dump traces.

Updated benchmark snapshot (March 2026):

- ``input.cth_like_free_bdy`` (non-axisymmetric, ``lasym=False``), iter 53:
  ``grpmn_nonsing rel_scaled ~1e-13``, ``amatrix rel_scaled ~1e-13``,
  ``potvac rel_scaled ~6.8e-2``.
- ``input.cth_like_free_bdy_lasym_small`` (non-axisymmetric, ``lasym=True``),
  iter 78 (full-update step, ``ivacskip=0``):
  ``grpmn_nonsing rel_scaled ~1e-11``, ``amatrix rel_scaled ~1e-11``,
  ``potvac rel_scaled ~1e-5``.
- ``input.DIII-D`` / ``input.DIII-D_reset`` (axisymmetric, ``lasym=True``):
  direct turn-on-window comparator on ``input.DIII-D`` is now near
  machine precision at iter 80
  (``source_sym ~2.06e-12``, ``bvec_nonsing_fouri ~2.07e-12``,
  ``amatrix ~1.44e-13``, ``potvac ~1.83e-12``).
- 2026-03-05 manifest rerun: all fixed-boundary manifest cases passed.
- 2026-03-05 manifest rerun: ``input.DIII-D`` and ``input.DIII-D_reset`` passed
  at current tightened thresholds.
- 2026-03-06 manifest rerun: ``input.cth_like_free_bdy_lasym_small`` now passes
  the current parity thresholds at iter 80 and iter 100, but the full-tier case
  still fails global status by runtime thresholds only.
- 2026-03-05 non-axisymmetric ``lasym=True`` cadence fix:
  the 3D path now preserves the pre-turn-on residual only where needed and
  invalidates cached ``ivac/ivacskip`` controls whenever a same-iteration
  restart updates ``iter1``. With that change, JAX matches the VMEC control
  trace around the late reuse window on
  ``input.cth_like_free_bdy_lasym_small``:
  ``(94,94,3,0)``, ``(95,95,3,0)``, ``(96,96,3,0)``, ``(97,97,3,0)``,
  ``(98,97,3,1)``, ``(99,99,3,0)``, ``(100,99,3,1)``.
- 2026-03-05 direct comparator after the cadence fix:
  ``input.cth_like_free_bdy_lasym_small`` iter 99 is back to near machine
  precision on the full-update step
  (``source_sym ~2.6e-8``, ``bvec_nonsing_fouri ~2.4e-8``,
  ``amatrix ~1.3e-11``, ``potvac ~1.1e-7``, ``bsqvac ~1.3e-7``).
- 2026-03-05 direct comparator after the cadence fix:
  ``input.cth_like_free_bdy_lasym_small`` iter 100 no longer shows the old
  order-one reuse failure; cached source/matrix channels are near machine
  precision (``source_sym ~2.6e-8``, ``bvec_nonsing_fouri ~2.4e-8``,
  ``amatrix ~1.3e-11``). The remaining reuse-step drift is now confined to
  the reconstructed field/coupling channels
  (``potvac ~7.1e-3``, ``bsqvac ~1.25e-2``,
  ``freeb_coupling_pgcon ~1.25e-2``).
- 2026-03-06 turn-on ``iter1`` control split:
  all free-boundary paths still use the same-iteration soft restart at turn-on,
  but only the non-axisymmetric ``lasym=True`` path now preserves the pre-turn-on
  ``iter1`` anchor. This keeps ``input.DIII-D`` at machine precision while
  matching the late VMEC reuse cadence on
  ``input.cth_like_free_bdy_lasym_small``.
- 2026-03-06 direct comparator after the ``iter1`` control split:
  ``input.cth_like_free_bdy`` iter 60 remains tight
  (``source_sym ~5.6e-7``, ``bvec_nonsing_fouri ~5.8e-7``,
  ``amatrix ~1.1e-13``, ``potvac ~8.4e-4``).
- 2026-03-06 direct comparator after the ``iter1`` control split:
  ``input.cth_like_free_bdy_lasym_small`` iter 60 is now near machine
  precision in the source/matrix channels with much smaller field drift
  (``bvec ~6.3e-5``, ``potvac ~4.0e-5``, ``bsqvac ~2.0e-4``).
- 2026-03-06 direct comparator after the ``iter1`` control split:
  ``input.cth_like_free_bdy_lasym_small`` iter 100 still has visible
  reuse-step drift in the reconstructed field/coupling channels, but it now
  stays within the current parity thresholds
  (``potvac ~1.0e-1``, ``bsqvac ~3.1e-1``,
  ``freeb_coupling_pgcon ~3.1e-1``). The remaining failure mode for this case
  is runtime, not comparator thresholds.
- 2026-03-06 free-boundary force-kernel JIT fix:
  the non-scan free-boundary path now keeps the jitted force kernels enabled,
  and the jitted wrapper accepts the free-boundary ``bsqvac`` edge-coupling
  argument. This preserves DIII-D parity while removing the runtime-only
  failure on the heavy local ``lasym=True`` case.
- 2026-03-06 manifest rerun after the free-boundary JIT fix:
  ``freeb_nonaxis_lasym_true_cth_like_local`` now passes with
  ``failed_cases=0`` in
  ``outputs/parity_sweeps/20260306_075253/summary.json``. Per-iteration
  runtimes dropped to about ``33.5s`` at iter 80 and ``32.7s`` at iter 100.
- 2026-03-06 direct full-solve runtime after the free-boundary JIT fix:
  ``run_fixed_boundary("examples/data/input.cth_like_free_bdy_lasym_small")``
  dropped from about ``71.5s`` to about ``37.8s`` on the same local machine
  and iteration count.
- 2026-03-05 manifest cleanup rerun
  (``outputs/parity_sweeps/20260305_183853/summary.json``):
  preserved local ``input.cth_like_free_bdy`` now passes in-manifest at
  iter 53/54/60
  (``source_sym ~5.3e-7``, ``bvec_nonsing_fouri ~5.5e-7``,
  ``amatrix ~1.1e-13``, ``potvac <= 3.6e-4``).
- 2026-03-05 manifest cleanup rerun:
  ``input.DIII-D_lasym_false`` now passes in-manifest and shows the same
  turn-on envelope at iter 80
  (``source_sym ~8.4e-3``, ``bvec_nonsing_fouri ~8.4e-3``,
  ``amatrix ~1.7e-3``, ``potvac ~9.4e-3``) and returns to near machine
  precision by iter 100+.
- 2026-03-05 manifest cleanup rerun:
  ``input.stellcopt`` now compares at post-turn-on iter 80 instead of
  pre-turn-on iterations with missing dumps; it passes the current coarse
  thresholds with ``source_sym ~2.72e-1``, ``bvec_nonsing_fouri ~2.80e-1``,
  ``amatrix ~1.20e-1``, ``potvac ~3.56e-1``.
- 2026-03-05 DIII-D iter-72 preconditioner cache diagnosis:
  raw ``gc`` and hidden preconditioner inputs already matched VMEC2000 to
  machine precision, and the first persistent mismatch was in the assembled
  ``scalfor`` matrices after free-boundary turn-on.
- 2026-03-05 DIII-D comparator path fix:
  relative ``MGRID_FILE`` entries are now resolved against the input file
  directory, which restores repo-root comparator runs. A direct rerun of
  ``input.DIII-D`` at iter 80 again emits ``jax_dumps`` and returns the
  expected turn-on-window parity metrics
  (``source_sym ~8.29e-3``, ``bvec_nonsing_fouri ~8.31e-3``,
  ``amatrix ~1.51e-3``, ``potvac ~9.45e-3``).
- 2026-03-05 DIII-D preconditioner cache-reuse fix:
  JAX now caches the full parity coefficients from ``precondn`` and
  reassembles ``scalfor`` matrices for a new ``jmax`` without forcing a fresh
  ``bcovar`` refresh. This matches VMEC2000 stale-cache behavior at the
  turn-on ``jmax=15 -> 16`` transition.
- 2026-03-05 direct iter-72 matrix comparison after the cache-reuse fix:
  JAX ``scalfor`` matrices now match VMEC2000 to machine precision with
  ``used_cache=True`` and ``jmax=16``
  (``ar/dr/br/az/dz/bz rel ~1e-14``).
- 2026-03-05 direct iter-80 comparator after the cache-reuse fix:
  ``input.DIII-D`` now returns near machine-precision parity in the prior
  turn-on blocker channels
  (``source_sym ~2.06e-12``, ``bvec_nonsing_fouri ~2.07e-12``,
  ``amatrix ~1.44e-13``, ``potvac ~1.83e-12``).
- 2026-03-05 cold-start direct runtime/memory matrix vs VMEC2000:
  fixed-boundary default runs are currently about ``26x``-``50x`` slower and
  use about ``6x``-``12x`` more RSS.
- 2026-03-05 cold-start direct runtime/memory matrix vs VMEC2000:
  free-boundary default runs are currently about ``23x``-``98x`` slower and use
  about ``12x``-``16x`` more RSS.
- 2026-03-05 cold-start direct runtime/memory matrix vs VMEC2000: worst
  observed case in this matrix is the local non-axisymmetric ``lasym=True``
  free-boundary solve at about ``62s`` / ``1.74 GiB`` RSS versus VMEC2000 at
  about ``0.63s`` / ``110 MiB`` RSS.

Key implementation updates that closed the matrix-side gap:

- Fortran-equivalent ``fourp`` poloidal extent was enforced with
  ``nu2 = nu/2 + 1`` (from ``read_indata.f``), including ``lasym=True`` runs.
- The non-singular ``grpmn`` assembly now uses raw ``fourp`` scale in JAX
  (no extra post-factor), matching VMEC ``fouri`` matrix construction.
- VMEC ``eqsolve`` turn-on behavior is mirrored: after ``ivac==1`` the solver
  promotes to ``ivac=2`` for subsequent iterations.
- VMEC ``becoil`` toroidal sampling was aligned in JAX mgrid interpolation:
  with ``use_vmec_kv=True`` the code now uses direct ``kv=mod(i-1,nv)+1``
  indexing (0-based: ``k=min(k, kp-1)``), i.e. no ``kp/nzeta`` rescaling.
- VMEC free-boundary turn-on sequencing was tightened in the non-scan path:
  ``ivac`` now starts at ``0`` (not ``-1``), so turn-on iteration aligns with
  VMEC ``funct3d`` for the same trajectory.
- Initial axis reset is no longer implicitly forced by ``LFREEB``. The default
  now follows VMEC Jacobian checks; forced behavior is available only via
  ``VMEC_JAX_FORCE_AXIS_RESET_INIT=1`` for debugging.
- The turn-on ``restart_iter(irst=2)`` parity path no longer mutates the
  persistent time step in JAX. VMEC calls restart on a local ``delt0`` copy.
- Free-boundary parity defaults no longer rely on manifest-side environment
  tuning. The default mode selection keeps VMEC-like dense assembly active on
  practical grids (large ``VMEC_JAX_FREEB_VMEC_LIKE_MAX_POINTS`` default) and
  the manifest now runs these cases without ``VMEC_JAX_FREEB_USE_GREENF_SOURCE``
  overrides.
- The free-boundary scalpot comparator now infers VMEC-style multigrid staging
  from ``NS_ARRAY`` by default (``--multigrid auto``), preventing false
  "missing JAX dump" failures on staged free-boundary inputs where vacuum
  turn-on happens before the finest grid.
- MGRID interpolation now accepts axisymmetric vacuum grids with a single
  toroidal plane (``kp=1``), matching VMEC behavior for DIII-D-like files and
  avoiding silent fallback when free-boundary sampling raises.
- VMEC ``tolicu`` axis-current construction is now mirrored for ``nv=1`` by
  using the same ``precal.f`` toroidal replication rule (``nvper=64`` when
  ``nv==1``). This removes the degenerate zero-field axis-current path in
  axisymmetric free-boundary runs and aligns ``brad_axis/bz_axis`` channels.
- Free-boundary source assembly now defaults to VMEC-like Green-function
  non-singular source in all topologies (axisymmetric and non-axisymmetric),
  with ``VMEC_JAX_FREEB_USE_GREENF_SOURCE`` retained only as a diagnostic
  override.
- In Green-function source assembly, ``nv==1`` now applies the same
  ``greenf.f`` normalization as VMEC2000 (divide accumulated ``delgr/delgrp``
  by ``nvper``), removing the axisymmetric over-scaling branch and improving
  ``source_sym/bvec_nonsing_fouri/amatrix/potvac`` parity for DIII-D runs.
- VMEC2000-iter stage budgeting with ``NITER_ARRAY`` + capped ``max_iter`` now
  consumes iterations in coarse-to-fine order (VMEC-like), instead of
  prioritizing the finest stage. This restores early-iteration free-boundary
  diagnostics parity for staged inputs.
- R/Z preconditioner caching now stores full parity coefficients and reuses
  them across turn-on ``jmax`` changes, so the cached matrix path matches VMEC
  ``scalfor`` reuse instead of recomputing fresh coefficients at iter 72.

The previous axisymmetric ``lasym=True`` turn-on residual is closed in the
preconditioner path. Time-control and restart traces remain aligned
channel-by-channel (``iter1``, ``ivac``, ``ivacskip``, ``irst``,
``res0/res1``, ``delt``), and the cached-channel diagnostics
(``source_sym_cached``, ``gsource_cached``, ``bvecNS_cached``,
``source_cache_iter``) remain enabled to localize any future reuse-step drift.

Scope and acceptance target
---------------------------

Target behavior for ``LFREEB=T``:

1. Match VMEC2000 iteration control (``ivac``, ``ivacskip``, restart behavior).
2. Match vacuum pressure coupling at the plasma edge (``bsqvac`` path).
3. Match free-boundary ``wout`` channels at project tolerances.
4. Keep ``LFREEB=F`` behavior unchanged.

Initial parity target for free-boundary is the same working policy used in
fixed-boundary validation:

- core channels: ``rtol=1e-3`` (often better),
- axis masking for near-axis cancellation-limited diagnostics,
- explicit handling of near-zero denominators for relative-error reporting.

VMEC2000 source deep dive (source-of-truth)
-------------------------------------------

Primary files and responsibilities in
``STELLOPT/VMEC2000/Sources``:

1. Input and setup

   - ``Input_Output/read_indata.f``: disables free-boundary when
     ``MGRID_FILE='NONE'``.
   - ``Input_Output/readin.f``: reads mgrid via ``read_mgrid(...)`` when
     ``LFREEB=T`` and validates ``NZETA`` against the mgrid toroidal grid plus
     ``NFP`` consistency.
   - ``LIBSTELL/Sources/Modules/mgrid_mod.f``: mgrid fields, interpolation
     metadata, ``nextcur``, ``mgrid_mode``, and external field tables.

2. Iteration control and coupling

   - ``TimeStep/runvmec.f``: initializes vacuum communicator/state per stage
     and carries ``ivac``/grid transitions in multigrid.
   - ``General/funct3d.f``: computes plasma fields, computes
     ``ivacskip = mod(iter2-iter1, nvacskip)`` with early-iteration and
     preconditioner overrides, triggers ``vacuum_par`` full vs skipped updates,
     and injects edge free-boundary forcing inputs (``pgcon``, ``rbsq``).
   - ``General/forces.f``: adds free-boundary edge-force terms when
     ``ivac >= 1``.

3. Vacuum solve (NESTOR)

   - ``NESTOR_vacuum/vacuum.f``: surface geometry, external field, potential
     solve, ``bsqvac``, and ``B^R/B^phi/B^Z`` on the boundary.
   - ``NESTOR_vacuum/scalpot.f``: integral-equation assembly, full update vs
     reused matrix (``ivacskip != 0``), and LU reuse.
   - ``NESTOR_vacuum/surface.f``, ``bextern.f``, ``becoil.f``,
     ``analyt.f``, ``greenf.f``, ``fouri.f``, and ``fourp.f``.

4. Output diagnostics

   - ``Input_Output/eqfor.f``: uses ``bsqvac`` and edge terms for
     free-boundary diagnostics plus betas/shape metrics.
   - ``wrout``/fileout path: persists free-boundary outputs for ``wout``
     parity.

Core equations and numerics to replicate
----------------------------------------

The free-boundary branch is NESTOR-coupled VMEC. The key equations as used in
``vacuum.f`` are:

1. Field decomposition at boundary:

   .. math::

      \mathbf{B} = R B_\phi \nabla \zeta + I_{\mathrm{tor}} \nabla \theta - \nabla \Phi.

2. Scalar potential tangential Fourier representation
   (VMEC/NESTOR ``(m,n)`` basis on the boundary).

3. Covariant tangential components:

   .. math::

      B_u = \partial_u \Phi + B^{\mathrm{ext}}_u,\qquad
      B_v = \partial_v \Phi + B^{\mathrm{ext}}_v.

4. Contravariant components from metric inversion:

   .. math::

      \begin{bmatrix}B^u\\ B^v\end{bmatrix}
      = g^{-1}
      \begin{bmatrix}B_u\\ B_v\end{bmatrix},

   with VMEC/NESTOR scaling conventions for :math:`g_{uv}`, :math:`g_{vv}`
   involving ``NFP``.

5. Vacuum magnetic pressure on the interface:

   .. math::

      p_{\mathrm{vac}} = \frac{1}{2}\left(B_u B^u + B_v B^v\right)
      = \frac{|\mathbf{B}_{\mathrm{vac}}|^2}{2}.

6. Coupling into plasma edge force balance in ``funct3d/forces`` via edge
   terms (``pgcon``, ``rbsq``), including the soft-start/restart behavior.

Numerically critical details to match:

- full vs skipped vacuum update cadence (``ivacskip`` logic),
- LU factor reuse and source-term updates in ``scalpot``,
- VMEC internal basis/scaling conventions (``mscale/nscale``, sign conventions),
- checkpoint/restart sequencing around first vacuum activation.

VMEC++ cross-checks that inform implementation
----------------------------------------------

VMEC++ contains a modular NESTOR implementation that is useful as a second
reference for structure and performance:

- ``vmecpp/cpp/vmecpp/free_boundary/nestor/nestor.cc``
  : explicit staged update
  ``surface -> bextern -> singular integrals -> regularized integrals -> DFT -> LU solve -> bsqvac``.
- ``vmecpp/cpp/vmecpp/free_boundary/mgrid_provider/*``
  : explicit mgrid loading/interpolation and ``nextcur`` checks.
- ``vmecpp`` numerics PDF documents NESTOR derivation and vacuum-pressure path.

Important caveat from VMEC++ docs:

- VMEC++ currently does **not** support ``lasym=True`` free-boundary.
  For ``lasym=True`` free-boundary parity, VMEC2000 remains the only source of
  truth.

Free-boundary case inventory (local)
------------------------------------

Cases found across STELLOPT / VMEC++ / SIMSOPT that are useful for staged
development:

1. Lightweight primary smoke case (recommended first):

   - input:
     ``vmecpp/src/vmecpp/cpp/vmecpp/test_data/input.cth_like_free_bdy``
   - mgrid:
     ``vmecpp/src/vmecpp/cpp/vmecpp/test_data/mgrid_cth_like.nc``
   - reference wout:
     ``vmecpp/src/vmecpp/cpp/vmecpp/test_data/wout_cth_like_free_bdy.nc``
   - notes: local, self-contained, finite-pressure/current free-boundary.

2. VMEC2000 benchmark axisymmetric+``lasym=True`` free-boundary:

   - input:
     ``STELLOPT/BENCHMARKS/VMEC_TEST/input.DIII-D``
     (and ``input.DIII-D_reset``)
   - mgrid:
     ``STELLOPT/BENCHMARKS/VMEC_TEST/mgrid_d3d_ef.nc``
   - notes: strong parity target for restart/time-control + output behavior.

3. Additional STELLOPT free-boundary benchmark:

   - input:
     ``STELLOPT/BENCHMARKS/STELLOPT_TEST/STELLCOPT/input.stellcopt``
   - mgrid:
     ``STELLOPT/BENCHMARKS/STELLOPT_TEST/STELLCOPT/mgrid_RUN09.00000.nc``
   - notes: good for non-axisymmetric free-boundary stress.

4. SIMSOPT free-boundary workflow example:

   - script:
     ``simsopt/examples/2_Intermediate/free_boundary_vmec.py``
   - notes: generates mgrid from coils, then runs VMEC free-boundary.

Implementation work packages
----------------------------

WP0: input and state model parity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Extend typed config/state to carry:
   ``lfreeb``, ``mgrid_file``, ``extcur``, ``nvacskip``, ``ivac`` state.
2. Add strict validation for mgrid dimensions, ``NFP``, and ``NZETA``.
3. Define VMEC2000-compatible fallback policy:
   - explicit error vs fixed-boundary fallback (configurable, documented).

Deliverables:

- new free-boundary config dataclasses,
- parser tests for free-boundary namelist keys,
- mgrid metadata validation tests.

WP1: MGRID loader/interpolator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Implement a deterministic mgrid loader (NetCDF) for:
   - coil-response tensors ``br/bp/bz``,
   - grid bounds and indexing,
   - ``nextcur``/mode metadata.
2. Implement field interpolation equivalent to ``becoil`` path.

Deliverables:

- ``vmec_jax/free_boundary/mgrid.py`` (or equivalent),
- unit tests against known mgrid test files (VMEC++ makegrid test data).

WP2: NESTOR core (vacuum solve)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Port boundary geometry/singular-integral setup equivalent to
   ``surface`` + ``analyt``.
2. Port regularized Green-function accumulation equivalent to ``greenf`` path.
3. Build linear system and LU factorization; support ``ivacskip`` matrix reuse.
4. Compute ``B_u/B_v``, ``B^u/B^v``, ``bsqvac``, boundary cylindrical components.

Deliverables:

- pure-JAX/Numpy hybrid NESTOR kernel with stage-static caches,
- VMEC2000 dump comparator for vacuum-only channels.

WP3: coupled VMEC loop integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Integrate ``ivac`` and ``ivacskip`` state machine into solve loop.
2. Add free-boundary edge coupling exactly where VMEC2000 applies it in
   ``funct3d``/``forces``.
3. Preserve restart/checkpoint ordering and print cadence.

Deliverables:

- per-iteration parity comparator extensions for free-boundary channels,
- scan and non-scan behavior alignment tests.

WP4: output parity
~~~~~~~~~~~~~~~~~~

1. Extend ``wout`` assembly with free-boundary quantities populated from solved
   state (not placeholders).
2. Match ``eqfor`` diagnostics and output conventions.

Deliverables:

- wout parity checks for free-boundary cases,
- regression baselines for selected free-boundary inputs.

WP5: differentiability and performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Keep linear solves differentiable (default AD path; optional custom VJP).
2. Cache stage-static vacuum matrices/bases to avoid per-iteration rebuild.
3. Keep fast path default with parity fallback.

Deliverables:

- gradient sanity tests (finite-difference vs AD for selected outputs),
- runtime/memory benchmark updates.

Documentation plan (required)
-----------------------------

Add or update docs pages:

1. ``docs/free_boundary_plan.rst`` (this page) -> implementation tracker.
2. ``docs/free_boundary_theory.rst`` (new):
   - equations, NESTOR derivation summary, basis/scaling conventions.
3. ``docs/algorithms.rst``:
   - free-boundary control flow and loop state machine.
4. ``docs/validation.rst``:
   - free-boundary parity methodology and masking rules.
5. ``docs/performance.rst``:
   - free-boundary-specific performance knobs and profiling notes.
6. README:
   - required external files (mgrid), minimal free-boundary run recipe.

Testing, validation, and benchmark plan
---------------------------------------

Unit tests
~~~~~~~~~~

1. MGRID parsing:
   - dimension checks, ``nextcur`` consistency, mode flags.
2. Interpolation parity:
   - known-point interpolation against VMEC++/VMEC reference values.
3. Vacuum algebra:
   - covariant/contravariant consistency and ``bsqvac`` reconstruction.
4. ``ivacskip``:
   - matrix reuse path vs full update path consistency.

Integration/parity tests
~~~~~~~~~~~~~~~~~~~~~~~~

1. Free-boundary executable comparator (new): extend the stage comparator to
   include ``ivac``, ``ivacskip``, ``bsqvac``, and edge-force channels.
2. Case ladder: Tier 1 is the preserved local ``input.cth_like_free_bdy``
   smoke fixture, Tier 2 is ``input.DIII-D`` plus
   ``input.DIII-D_lasym_false``, and Tier 3 is ``input.stellcopt`` plus
   additional STELLOPT/SIMSOPT cases.
3. ``wout`` parity:
   - same axis masking policy where required for cancellation-limited channels.

Benchmarks
~~~~~~~~~~

1. Runtime:
   - VMEC2000 vs vmec_jax total wall time and stage breakdown.
2. Memory:
   - peak RSS/allocator tracking by stage.
3. Throughput:
   - short multigrid and single-grid free-boundary runs.
4. Optional:
   - warm vs cold JIT comparison for free-boundary path.

Definition of done for free-boundary milestone
----------------------------------------------

1. ``LFREEB=F`` fixed-boundary parity and performance are unchanged.
2. At least one nontrivial free-boundary case converges with per-iteration
   parity to VMEC2000 control traces.
3. Free-boundary ``wout`` channels match VMEC2000 at project tolerances.
4. Free-boundary docs, tests, parity scripts, and benchmarks are in repo and
   CI covers a reduced but representative subset.
