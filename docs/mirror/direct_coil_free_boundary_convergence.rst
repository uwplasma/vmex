Direct-Coil Free-Boundary Convergence Plan
==========================================

This page is the active convergence plan for the square-coil
stellarator-mirror-hybrid free-boundary lane. It records what is already known,
what evidence is still missing, and the finite set of gates required before a
direct-coil finite-beta run is described as a converged production equilibrium.

Current Status
--------------

The repo-root example
``examples/toroidal_stellarator_mirror_hybrid_square_coils_free_boundary.py``
now runs actual ``vmec_jax.run_free_boundary`` solves from a direct-coil
provider. It writes solved WOUT files, solved-state field-line and LCFS plots,
per-beta CSV/JSON metrics, and checkpoints the summary after each beta row.
The branch also keeps the final live state by default; older behavior that
returned a best-scored intermediate state is now opt-in through
``VMEC_JAX_RETURN_BEST_SCORED_STATE=1``.

The latest strict fresh residual evidence at the coarse review resolution
``NS=9, MPOL=5, NTOR=12`` reaches active free-boundary convergence through
``5%`` beta. At the same resolution, ``7%`` beta is the first transition case:
it stalls just above the requested radial-force tolerance and shows many
bad-Jacobian resets. ``8%`` through ``10%`` beta stall with larger radial-force
floors. Older resolution and step-size probes show that increasing resolution
and increasing ``DELT`` from the default reduce the ``10%`` radial-force floor,
but do not yet prove strict convergence.

Physics And Algorithm Findings
------------------------------

VMEC2000 remains the solve-side reference. Its free-boundary path delays vacuum
coupling until the interior ``R/Z`` force residual is already small, forces full
NESTOR updates for the first vacuum iterations, then uses ``NVACSKIP`` to
cadence expensive full vacuum updates. The vacuum magnetic pressure enters the
edge force through the dedicated ``bsqvac``/``rbsq`` path, not by enabling
generic fixed-boundary edge rows. VMEC documentation and the SIMSOPT VMEC
interface also make clear that strict production runs usually need large
final-grid iteration budgets, often thousands of iterations.

DESC separates the postsolve finite-beta boundary condition from the VMEC-style
iterative solve. Its finite-beta ``BoundaryError`` objective uses the virtual
casing principle to compute the plasma contribution to the exterior magnetic
field, then checks both normal-field and magnetic-pressure jump conditions. The
cheaper ``VacuumBoundaryError`` is valid only when pressure and plasma current
vanish. This matches the interpretation here: coil-only ``B.n`` is a vacuum
diagnostic, not a finite-beta promotion criterion.

The ``virtual_casing_jax`` package is the preferred optional postprocessor for
research-grade finite-beta diagnostics because it provides JAX-compatible
on-surface external/internal field functionals, normal-field JVP columns, and
high-order singular quadrature. The direct-coil convergence lane should use it
as an optional diagnostic dependency rather than copying DESC's singular
integral implementation into ``vmec_jax``.

Promotion Gates
---------------

A direct-coil finite-beta equilibrium is promoted only when all gates below are
true for the current live state:

1. ``LFREEB`` is active and NESTOR/direct-coil coupling has turned on.
2. A fresh final NESTOR/direct-coil sample has been recomputed on the accepted
   state, not on a stale trial or best-scored state.
3. ``final_fsqr``, ``final_fsqz``, and ``final_fsql`` each satisfy the requested
   ``FTOL`` on the final recompute.
4. The LCFS coefficients change with beta in the solved WOUTs, proving that the
   boundary is not effectively fixed.
5. A resolution ladder in ``NS``, ``MPOL``, ``NTOR``, and boundary sampling
   shows stable convergence of the final force residuals, LCFS shape, near-axis
   ``|B|``, mirror ratio, and mean iota.
6. Finite-beta promotion uses VMEC force residuals plus a total-field
   pressure-balance diagnostic. Coil-only ``B.n`` remains a vacuum-only check.
7. The plotted LCFS, cross sections, ``|B|`` maps, field-line traces, residual
   histories, and beta trends come from solved states and are regenerated from
   ignored ``results/`` artifacts.

Execution Plan
--------------

The remaining work is deliberately narrow:

1. Re-run the square-coil beta ladder with the live-state default and per-beta
   checkpointing at higher resolution. The first queued office run uses
   ``NS=13, MPOL=6, NTOR=14, NZETA=40, DELT=0.05, FTOL=1e-8`` and a gradual beta
   continuation from ``0%`` to ``10%``.
2. If the transition remains near ``7%`` beta, run local continuation probes
   around that transition with ``DELT`` and ``FREE_BOUNDARY_ACTIVATE_FSQ`` scans,
   using the same final fresh-residual gate.
3. Run a higher-resolution closure attempt for ``10%`` beta starting from
   ``NS=17, MPOL=7, NTOR=16, NZETA=48`` and compare LCFS, near-axis field,
   mirror ratio, mean iota, and residual histories against the ``NS=13`` ladder.
4. Add an optional virtual-casing postsolve diagnostic that accepts a solved
   VMEC state, total surface field, and direct-coil field, then reports
   finite-beta normal-field and magnetic-pressure jump residuals. The helper
   should be optional at import time and skipped when ``virtual_casing_jax`` is
   not installed.
5. Promote only rows that pass the force-residual and postsolve boundary
   diagnostics. Keep unconverged rows in the example output as explicit stall
   evidence with ``production_free_boundary_claim = false``.

Best-Practice Constraints
-------------------------

Keep generated WOUTs, raw JSON, and full-resolution figures under ignored
``results/`` paths. If a figure is committed for review, compress it and keep
only summary panels. Source changes should stay in the existing free-boundary,
external-field, or mirror-domain modules; avoid adding new root-level helper
files. Diagnostic functions should have small public entry points, clear
docstrings, and tests that exercise vacuum and finite-beta interpretation
separately.

Reviewed External Anchors
-------------------------

- DESC free-boundary tutorial:
  https://desc-docs.readthedocs.io/en/stable/notebooks/tutorials/free_boundary_equilibrium.html
- DESC free-boundary paper/preprint:
  https://arxiv.org/abs/2412.05680
- VMEC/STELLOPT documentation:
  https://princetonuniversity.github.io/STELLOPT/VMEC.html
- SIMSOPT VMEC free-boundary interface notes:
  https://simsopt.readthedocs.io/v1.10.2/example_vmec.html
- WHAM finite-beta mirror physics basis:
  https://www.osti.gov/biblio/2001162
- VMEC/DESC/SPEC free-boundary Shafranov-shift verification:
  https://doi.org/10.1063/5.0253843
