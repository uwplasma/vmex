Free-Boundary Plan
==================

This page documents the free-boundary implementation plan for ``vmec-jax``,
based on the VMEC2000 source tree (``STELLOPT/VMEC2000/Sources``).

VMEC2000 free-boundary flow (source-of-truth)
----------------------------------------------

The free-boundary path in VMEC2000 is centered around ``LFREEB`` and the
vacuum-coupling modules under ``NESTOR_vacuum``:

1. Input and setup

   - ``Input_Output/read_indata.f``: enables/disables ``LFREEB`` based on
     ``MGRID_FILE``.
   - ``Input_Output/readin.f``: calls ``read_mgrid(...)`` when ``LFREEB=T`` and
     stores external-coil data.
   - ``TimeStep/runvmec.f``: initializes vacuum communicators
     (``SetVacuumCommunicator``), carries ``ivac`` state across multigrid.

2. Plasma iteration and vacuum updates

   - ``General/funct3d.f``:
     - computes plasma fields (``bcovar`` + force kernels),
     - decides when to run full vs skipped vacuum update
       (``ivacskip = mod(iter2-iter1, nvacskip)``),
     - calls ``vacuum_par(...)`` (or ``vac2_vacuum`` path in some builds),
     - injects edge pressure/boundary terms via ``pgcon``/``rbsq``.

3. Vacuum solve

   - ``NESTOR_vacuum/vacuum.f`` + helpers:
     - ``surface``: boundary geometry at the interface,
     - ``bextern``/``becoil``: external field from MGRID coils,
     - ``scalpot``/``analyt``/``greenf``: scalar-potential matrix assembly,
     - LAPACK solve (``DGETRF/DGETRS``),
     - outputs ``bsqvac`` and integrated edge quantities
       (including ``bsubuvac``/``bsubvvac`` checks).

4. Force coupling and output

   - ``General/forces.f``: adds free-boundary edge force contributions
     (``ivac >= 1`` branch, uses ``rbsq``).
   - ``Input_Output/eqfor.f`` and ``wrout`` path: consumes ``bsqvac`` and
     free-boundary edge terms for diagnostics and output fields.

Implementation strategy in vmec-jax
-----------------------------------

Phase 1: data model + parser parity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Add full ``LFREEB`` input state to JAX config:
   ``mgrid_file``, ``extcur``, ``nvacskip``, filters, coil-set metadata.
2. Build deterministic MGRID loader (host side), with JAX arrays stored in
   stage-static containers.
3. Preserve current fixed-boundary behavior when ``LFREEB=F``.

Phase 2: vacuum kernel parity (NESTOR equivalent)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Implement interface-surface geometry kernels equivalent to
   ``surface.f`` (same orientation/sign conventions).
2. Implement external-field interpolation equivalent to
   ``bextern.f``/``becoil.f`` from MGRID.
3. Implement scalar-potential matrix assembly and solve equivalent to
   ``scalpot.f`` + ``analyt*.f`` + ``greenf.f`` + LU solve.
4. Validate vacuum-only outputs first (``bsqvac``, ``bsubu/v`` integrals)
   against VMEC2000 dumps before coupling to force updates.

Phase 3: coupled iteration parity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Integrate ``ivac``/``ivacskip``/``nvacskip`` logic into VMEC2000 control
   loop (scan and non-scan), matching restart behavior.
2. Add free-boundary edge-force injection in JAX force kernels with the same
   ordering used by VMEC2000 ``funct3d``/``forces``.
3. Revalidate time-step/restart parity with vacuum active.

Phase 4: output parity
~~~~~~~~~~~~~~~~~~~~~~

1. Extend ``wout`` writing path to include free-boundary diagnostics populated
   from the coupled solve (not placeholders).
2. Match VMEC2000 ``eqfor`` free-boundary diagnostics and printout behavior.

Phase 5: differentiability and performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Keep the vacuum linear solve differentiable:
   - default autodiff through ``jax.scipy.linalg.solve`` or equivalent,
   - optional ``custom_vjp`` if needed for stability/performance.
2. Cache stage-static structures (MGRID tensors, basis tables, vacuum matrix
   sparsity/layout metadata).
3. Keep fast path default while preserving non-scan parity fallback for
   difficult stages.

Validation plan (free boundary)
-------------------------------

1. Start with low-resolution free-boundary cases from STELLOPT / simsopt test
   suites to verify iteration-level parity.
2. Add VMEC2000 executable comparator coverage for:
   - scalar traces (``fsqr/fsqz/fsql``, DELT, restart events),
   - vacuum channels (``bsqvac``, boundary terms),
   - final ``wout`` parity under axis masking conventions already used in
     fixed-boundary validation.
3. Promote to CI-gated tests once runtime is stable.

Acceptance criteria
-------------------

- ``LFREEB=F`` behavior unchanged (existing fixed-boundary parity retained).
- ``LFREEB=T`` matches VMEC2000 per-iteration traces on selected benchmarks.
- Final ``wout`` parity for free-boundary cases at project tolerances.
- End-to-end differentiation remains available for fixed-boundary and the new
  free-boundary path.
