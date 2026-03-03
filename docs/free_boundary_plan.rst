Free-Boundary Plan
==================

This document is the implementation and validation plan for adding
VMEC2000-quality free-boundary capability to ``vmec-jax`` while preserving:

- fixed-boundary parity and defaults,
- end-to-end differentiability,
- high CPU performance (scan fast path + parity fallback),
- bounded memory usage.

Current Implementation Status (March 2, 2026)
---------------------------------------------

WP0 is implemented:

- typed free-boundary input config parsed from ``&INDATA``:
  ``LFREEB``, ``MGRID_FILE``, ``EXTCUR``, ``NVACSKIP``,
- VMEC2000-aligned defaults:
  ``LFREEB=T`` with ``MGRID_FILE='NONE'`` disables free-boundary,
  ``NVACSKIP<=0`` falls back to ``NFP``,
- typed free-boundary runtime state scaffold in ``VMECStatic``,
- mgrid loader skeleton (metadata + optional BR/BP/BZ tensor loading),
- unit tests for parsing, normalization, and mgrid metadata loading.

WP1 is partially in place:

- driver now loads and validates mgrid metadata for ``LFREEB=T``,
  with strict checks for ``NFP`` agreement and ``kp % nzeta == 0``.
- solve loop now carries VMEC-style free-boundary control placeholders
  (``ivac``, ``ivacskip``, ``nvacskip``) in diagnostics/resume state.
- vacuum coupling and NESTOR solve integration remain pending.

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

   - ``Input_Output/read_indata.f``
     - disables free-boundary when ``MGRID_FILE='NONE'``.
   - ``Input_Output/readin.f``
     - reads mgrid via ``read_mgrid(...)`` when ``LFREEB=T``,
     - validates ``NZETA`` against mgrid toroidal grid and ``NFP`` consistency.
   - ``LIBSTELL/Sources/Modules/mgrid_mod.f``
     - mgrid fields, interpolation metadata, ``nextcur``, ``mgrid_mode``,
       external field tables.

2. Iteration control and coupling

   - ``TimeStep/runvmec.f``
     - initializes vacuum communicator/state per stage,
     - carries ``ivac``/grid transitions in multigrid.
   - ``General/funct3d.f``
     - computes plasma fields,
     - computes ``ivacskip = mod(iter2-iter1, nvacskip)`` (with early-iteration
       and preconditioner overrides),
     - triggers ``vacuum_par`` full vs skipped updates,
     - injects edge free-boundary forcing inputs (``pgcon``, ``rbsq``).
   - ``General/forces.f``
     - adds free-boundary edge-force terms when ``ivac >= 1``.

3. Vacuum solve (NESTOR)

   - ``NESTOR_vacuum/vacuum.f``
     - surface geometry + external field + potential solve + ``bsqvac``,
       ``B^R/B^phi/B^Z`` on boundary.
   - ``NESTOR_vacuum/scalpot.f``
     - integral-equation assembly, full update vs reused matrix
       (``ivacskip != 0``), LU reuse.
   - ``NESTOR_vacuum/surface.f``, ``bextern.f``, ``becoil.f``,
     ``analyt.f``, ``greenf.f``, ``fouri.f``, ``fourp.f``.

4. Output diagnostics

   - ``Input_Output/eqfor.f``
     - uses ``bsqvac`` and edge terms for free-boundary diagnostics and
       betas/shape metrics.
   - ``wrout``/fileout path
     - persists free-boundary outputs for ``wout`` parity.

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
  - explicit staged update:
    ``surface -> bextern -> singular integrals -> regularized integrals -> DFT -> LU solve -> bsqvac``.
- ``vmecpp/cpp/vmecpp/free_boundary/mgrid_provider/*``
  - explicit mgrid loading/interpolation and ``nextcur`` checks.
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

1. Free-boundary executable comparator (new):
   - extend stage comparator to include
     ``ivac``, ``ivacskip``, ``bsqvac``, edge force channels.
2. Case ladder:
   - Tier 1: ``input.cth_like_free_bdy`` (VMEC++ test data),
   - Tier 2: ``input.DIII-D`` (VMEC2000 benchmark),
   - Tier 3: additional STELLOPT/SIMSOPT cases.
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
