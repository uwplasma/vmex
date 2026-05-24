Free-Boundary Coil Optimization
===============================

This page documents the research branch for true single-stage free-boundary
optimization with differentiable coils. The existing VMEC-compatible
``mgrid`` path remains the parity backend. The new direct-coil path evaluates
the external field from coil Fourier coefficients and currents in JAX, so the
coil parameters can become the independent optimization variables.

Architecture
------------

The intended single-stage loop is:

.. code-block:: text

   coil Fourier dofs/currents
      -> differentiable Biot-Savart external field
      -> vmec_jax free-boundary equilibrium
      -> wout/Boozer/QS diagnostics
      -> coil-only objective update

.. image:: _static/figures/freeb_single_stage_architecture.png
   :alt: Direct-coil free-boundary architecture
   :width: 100%

Phase 1 in this branch includes JAX-native coil-field sampling, an ESSOS coil
adapter, generated-``mgrid`` compatibility, and forward free-boundary solves
from direct coils. Phase 2 is the production custom adjoint through the full
free-boundary vacuum/NESTOR solve. Dense toy vacuum-adjoint tests are present
now, but the full-solve adjoint is not claimed as publication-ready until
finite-difference checks of complete solves pass.

Finite-pressure direct-coil support is currently a provider/coupling validation
lane: active NESTOR diagnostics respond to coil-current changes and matched
direct/generated-``mgrid`` samples agree, but accepted-equilibrium sensitivity
and high-beta convergence remain promotion gates.

Low-Resolution Beta Scan
------------------------

The first diagnostic uses ESSOS Landreman-Paul QA coils and a four-point
pressure scan. The zero-pressure endpoint is retained as a reference, but the
finite-pressure points are the meaningful provider-plumbing checks. The same
coil set is used two ways:

1. ESSOS coils are sampled onto an ``mgrid`` file and solved by the legacy
   free-boundary compatibility path.
2. The same ESSOS coils are converted to ``CoilFieldParams`` and sampled
   directly by the differentiable JAX Biot-Savart provider.

.. image:: _static/figures/freeb_single_stage_beta_scan.png
   :alt: Low-resolution direct-coil beta scan
   :width: 100%

The scalar diagnostics from the two ``vmec_jax`` providers agree exactly in
the JSON summary for this low-resolution smoke run. The scan records both the
input ``PRES_SCALE`` and the output energy ratio ``100 W_p / W_B`` so future
plots cannot accidentally validate only the vacuum case.

The example uses ``--activate-fsq 1.0`` by default. This forces early
VMEC2000-style NESTOR turn-on so the short run exercises active finite-pressure
vacuum coupling instead of stopping in the inactive ``ivac=-1`` cadence. That
is useful for provider validation. The residuals shown here are recomputed on
the accepted final state with a fresh active NESTOR sample, but this is still
not a converged high-beta result: the active residual norm remains large and
must be bounded against VMEC2000 before this becomes a promoted finite-beta
single-stage optimization claim.

.. image:: _static/figures/freeb_single_stage_provider_parity.png
   :alt: Direct-coil provider parity
   :width: 100%

The committed numerical summary is stored in
``docs/_static/figures/freeb_single_stage_beta_scan_summary.csv``.

Reproduction
------------

Run the beta scan from the repository root. Until the ESSOS ``to_mgrid`` PR is
merged and released, put the ESSOS branch checkout on ``PYTHONPATH``.

.. code-block:: bash

   PYTHONPATH=/Users/rogeriojorge/local/ESSOS_mgrid_pr:$PYTHONPATH \
     python examples/free_boundary_essos_coils_beta_scan.py \
     --outdir results/free_boundary_essos_coils_beta_scan_readme \
     --activate-fsq 1.0

The ESSOS Landreman-Paul QA fixture has relatively weak currents for the short
finite-pressure smoke. Use ``--coil-current-scale`` to run matched direct/mgrid
sensitivity studies with stronger coils:

.. code-block:: bash

   PYTHONPATH=/Users/rogeriojorge/local/ESSOS_mgrid_pr:$PYTHONPATH \
     python examples/free_boundary_essos_coils_beta_scan.py \
     --outdir results/free_boundary_essos_coils_beta_scan_scaled \
     --coil-current-scale 100 \
     --activate-fsq 1e99

Render the README/docs figures from the generated JSON summary:

.. code-block:: bash

   python tools/diagnostics/render_freeb_single_stage_readme.py \
     --summary results/free_boundary_essos_coils_beta_scan_readme/summary.json \
     --outdir docs/_static/figures

The example writes ``input.*`` decks, ``wout_*.nc`` files, a generated mgrid,
and ``summary.json`` in the output directory. Those runtime files are ignored
by git; the committed figures and CSV are generated artifacts for documentation
only.

Phase-1 Coil-Only Optimization Smoke
------------------------------------

The initial single-stage optimization example is deliberately a smoke scaffold,
not a promoted QS design. It optimizes only coil currents and selected coil
Fourier coefficients. The VMEC plasma boundary coefficients are never included
in the optimization vector; the plasma surface is recomputed by a direct-coil
free-boundary solve at every objective evaluation.

The default objective is a cheap proxy:

- accepted-state VMEC residual,
- aspect-ratio target,
- mean-iota target.

The example records ``history.json``, ``summary.json``, and the best ``wout``.
It exits with code ``77`` when optional ESSOS assets are unavailable. For a
dependency-light developer smoke, use the synthetic circular coil provider:

.. code-block:: bash

   python examples/optimization/free_boundary_QS_coil_optimization.py \
     --smoke \
     --provider circle \
     --max-evals 1 \
     --max-iter 1 \
     --vmec-max-iter 1 \
     --pressure-scale 100 \
     --activate-fsq 1e99 \
     --outdir results/free_boundary_QS_coil_optimization_circle_smoke

For the ESSOS Landreman-Paul QA coils, put ESSOS on ``PYTHONPATH`` and use:

.. code-block:: bash

   PYTHONPATH=/Users/rogeriojorge/local/ESSOS_mgrid_pr:$PYTHONPATH \
     python examples/optimization/free_boundary_QS_coil_optimization.py \
     --smoke \
     --max-evals 3 \
     --outdir results/free_boundary_QS_coil_optimization_essos_smoke

The next promotion step is replacing the cheap proxy with a Boozer/QS objective
and validating finite-difference gradients of the complete direct-coil
free-boundary loop.

Robust Coil Perturbations
-------------------------

The phase-1 direct-coil example can optionally evaluate a robust objective by
adding perturbed coil scenarios to the nominal free-boundary solve:

.. code-block:: bash

   python examples/optimization/free_boundary_QS_coil_optimization.py \
     --smoke \
     --provider circle \
     --robust-samples 2 \
     --robust-risk mean_plus_std

The default remains the deterministic nominal objective. When
``--robust-samples`` is positive, the example samples common coil perturbations
with ``vmec_jax.robust_coils`` and aggregates nominal plus perturbed scenario
losses with ``mean``, ``mean_plus_std``, or ``smooth`` risk aggregation.

``vmec_jax.robust_coils`` provides pure-JAX perturbation helpers for robust coil
objectives:

- multiplicative current perturbations,
- rigid Cartesian displacements,
- toroidal phase rotations about the z axis,
- additive Fourier-centerline perturbations,
- risk aggregation with mean, mean-plus-standard-deviation, smooth maximum, and
  smooth tail/CVaR-like penalties.

These functions operate on ``CoilFieldParams`` pytrees and do not require
ESSOS. They can be used with ``jax.vmap`` for transformable objective pieces;
the example intentionally uses a Python loop around full free-boundary solves
until the production solver path is fully JAX-transformable.

Benchmarks
----------

The branch includes lightweight, non-CI benchmark scripts:

.. code-block:: bash

   python tools/benchmarks/bench_freeb_direct_coil_matrix.py \
     --quick \
     --out results/bench_freeb_direct_coil_matrix/summary.json

   python tools/benchmarks/bench_external_field_providers.py \
     --points 48 --segments 48 \
     --out results/bench_external_field_providers.json

   python tools/benchmarks/bench_freeb_direct_coil_solve.py \
     --max-iter 2 \
     --out results/bench_freeb_direct_coil_solve.json

   python tools/benchmarks/bench_freeb_coil_gradient.py \
     --points 24 --segments 48 --matrix-size 24 \
     --out results/bench_freeb_coil_gradient.json

Each benchmark writes JSON with backend/device information, cold/compile
timing, warm timing, and the problem dimensions. Defaults are intentionally
small and CPU-safe; GPU production benchmarks should raise the grid and segment
counts explicitly.

The matrix runner is the quickest smoke command for the direct-coil benchmark
lane. It runs the provider, direct free-boundary solve, and coil-gradient
scripts with small CPU-only defaults, writes each child JSON into the output
directory, and records their paths plus compact timing/status rows in
``summary.json``. GPU rows are opt-in:

.. code-block:: bash

   python tools/benchmarks/bench_freeb_direct_coil_matrix.py \
     --quick \
     --include-gpu \
     --backend-note "local workstation smoke" \
     --out results/bench_freeb_direct_coil_matrix_gpu/summary.json

If no JAX GPU device is available, the matrix records a skipped GPU row rather
than falling back silently to CPU.

Validation Status
-----------------

Current fast tests cover:

- direct-coil Biot-Savart derivatives with respect to currents, coil Fourier
  coefficients, and evaluation coordinates;
- ESSOS adapter value parity when ESSOS is installed;
- JAX ``mgrid`` interpolation value and gradient checks;
- a direct-coil runtime hook that does not require an ``mgrid`` file and uses
  nonzero pressure;
- generated-``mgrid`` versus direct-coil ``vmec_jax`` provider parity for the
  ESSOS Landreman-Paul QA finite-pressure smoke case;
- active direct-coil NESTOR-step sensitivity to coil-current changes, including
  the expected linear normal-field/source scaling and quadratic ``bsqvac``
  scaling;
- direct-provider source refresh on reuse and trial-state vacuum-field refresh,
  so direct coils are not scored against stale pre-update source data;
- robust-coil perturbation/risk aggregation utilities;
- dense toy vacuum-adjoint tests.

The optional VMEC2000 generated-``mgrid`` comparison is present but xfailed for
now. VMEC2000 reads the generated grid and advances the trace locally, but the
current generated-``mgrid`` free-boundary parity gap is not bounded tightly
enough for a promoted gate. That is a validation task, not a reason to regress
the existing VMEC2000-parity ``mgrid`` fixtures.

Next Implementation Steps
-------------------------

- Bound active accepted-equilibrium sensitivity to direct coil parameters with
  full-solve finite-difference checks, then promote the optional xfail.
- Replace the phase-1 coil-only optimization proxy with a Boozer/QS objective
  once the complete direct-coil free-boundary loop has validated gradients.
- Promote the VMEC2000 generated-``mgrid`` comparison after the direct/mgrid
  trace discrepancy is bounded.
- Replace the dense toy vacuum-adjoint scaffold with the production
  matrix-free/custom-linear-solve NESTOR operator.
- Run the benchmark matrix on CPU and GPU and turn the JSON summaries into
  documentation plots.
