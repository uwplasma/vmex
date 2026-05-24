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

Low-Resolution Beta Scan
------------------------

The first reviewer-facing diagnostic uses ESSOS Landreman-Paul QA coils and a
four-point pressure scan. The zero-pressure endpoint is retained as a reference,
but the promoted checks are the finite-pressure points. The same coil set is
used two ways:

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
     --outdir results/free_boundary_essos_coils_beta_scan_readme

Render the README/docs figures from the generated JSON summary:

.. code-block:: bash

   python tools/diagnostics/render_freeb_single_stage_readme.py \
     --summary results/free_boundary_essos_coils_beta_scan_readme/summary.json \
     --outdir docs/_static/figures

The example writes ``input.*`` decks, ``wout_*.nc`` files, a generated mgrid,
and ``summary.json`` in the output directory. Those runtime files are ignored
by git; the committed figures and CSV are generated artifacts for documentation
only.

Validation Status
-----------------

Current fast tests cover:

- direct-coil Biot-Savart derivatives with respect to currents, coil Fourier
  coefficients, and evaluation coordinates;
- ESSOS adapter value parity when ESSOS is installed;
- JAX ``mgrid`` interpolation value and gradient checks;
- a direct-coil free-boundary runtime hook that does not require an ``mgrid``
  file;
- generated-``mgrid`` versus direct-coil ``vmec_jax`` free-boundary parity for
  the ESSOS Landreman-Paul QA smoke case;
- dense toy vacuum-adjoint tests.

The optional VMEC2000 generated-``mgrid`` comparison is present but xfailed for
now. VMEC2000 reads the generated grid and advances the trace locally, but the
current generated-``mgrid`` free-boundary parity gap is not bounded tightly
enough for a promoted gate. That is a validation task, not a reason to regress
the existing VMEC2000-parity ``mgrid`` fixtures.

Next Implementation Steps
-------------------------

- Add the first coil-only optimization example where the plasma boundary is
  never an independent optimization variable.
- Promote the VMEC2000 generated-``mgrid`` comparison after the direct/mgrid
  trace discrepancy is bounded.
- Replace the dense toy vacuum-adjoint scaffold with the production
  matrix-free/custom-linear-solve NESTOR operator.
- Add benchmark plots for direct-coil sampling and free-boundary solve timing
  on CPU and GPU.
