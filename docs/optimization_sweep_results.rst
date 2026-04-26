Optimization Sweep Results
==========================

This page collects the generated optimization sweep artifacts used by the
README and the main optimization guide.  The current sweep covers QA, QH, QP,
and QI targets:

- QA: aspect ratio, mean iota, and quasi-axisymmetry.
- QH: aspect ratio and quasi-helical symmetry.
- QP: aspect ratio, quasi-poloidal symmetry, and an absolute-iota lower bound.
- QI: aspect ratio and a differentiable smooth Boozer-space
  quasi-isodynamic residual evaluated through ``booz_xform_jax``.  The sweep
  first runs a same-mode QP preseed and then applies the QI residual so the
  final state does not remain trapped in the QH warm-start basin.

Individual Examples
-------------------

Each standalone example keeps all user controls as top-level Python variables:

.. code-block:: bash

   PYTHONPATH=. python examples/optimization/qa_fixed_resolution_jax_ess.py
   PYTHONPATH=. python examples/optimization/qh_fixed_resolution_jax.py
   PYTHONPATH=. python examples/optimization/qp_fixed_resolution_jax_ess.py
   PYTHONPATH=. python examples/optimization/qi_fixed_resolution_jax_ess.py

The QP script is quasisymmetry with ``HELICITY_M = 0``.  The QI script is a
different objective: it builds Boozer spectra with ``booz_xform_jax``, first
uses QP as a preseed, and then penalizes field-line variation in smooth
magnetic-well widths and normalized well profiles.  Install the optional
dependency set with ``python -m pip install ".[qi]"`` before running QI cases
from a source checkout.

Sweep Reproduction
------------------

Run the CPU production sweep:

.. code-block:: bash

   PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/generate_qs_ess_sweep.py --backend-label cpu --solver-device cpu --policy continuation --problems qa,qh,qp,qi --modes 1,2,3 --ess both
   PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/generate_qs_ess_sweep.py --backend-label cpu --solver-device cpu --policy direct --problems qa,qh,qp,qi --modes 1,2,3 --ess both
   PYTHONPATH=. python examples/optimization/render_qs_ess_publication_panel.py

Run the GPU diagnostic sweep on a machine with a working JAX GPU install:

.. code-block:: bash

   PYTHONPATH=. JAX_PLATFORM_NAME=gpu python examples/optimization/generate_qs_ess_sweep.py --backend-label gpu --solver-device gpu --policy continuation --problems qa,qh,qp,qi --modes 1,2,3 --ess both
   PYTHONPATH=. JAX_PLATFORM_NAME=gpu python examples/optimization/generate_qs_ess_sweep.py --backend-label gpu --solver-device gpu --policy direct --problems qa,qh,qp,qi --modes 1,2,3 --ess both
   PYTHONPATH=. python examples/optimization/render_qs_ess_publication_panel.py

The default per-case timeout is 600 seconds.  Use ``--case-timeout-s 0`` only
for unbounded local diagnostics.

Objective Histories
-------------------

The all-policy panel contains every available backend/policy row.  Solid curves
met the optimizer success criterion; dashed curves are bounded-budget stops.
Vertical dotted lines mark continuation stage boundaries.

.. image:: _static/figures/qs_ess_objective_panel_all_policies.png
   :width: 100%
   :align: center
   :alt: Full QA, QH, QP, and QI optimization policy sweep

The legacy-compatible objective panel filename is also regenerated from the
same current data, so older links no longer point at stale pre-QP/QI figures:

.. image:: _static/figures/qs_ess_objective_panel.png
   :width: 100%
   :align: center
   :alt: Legacy objective panel alias generated from current sweep data

Backend-specific objective panels:

.. image:: _static/figures/qs_ess_objective_panel_cpu_policies.png
   :width: 100%
   :align: center
   :alt: CPU optimization policy sweep

.. image:: _static/figures/qs_ess_objective_panel_gpu_policies.png
   :width: 100%
   :align: center
   :alt: GPU optimization policy sweep

Final-State Atlases
-------------------

The final-state atlases show the LCFS and line contours of ``|B|`` on the LCFS.
Each 3-D panel has its own colorbar because the aspect-ratio constraint changes
the absolute ``|B|`` range.

.. image:: _static/figures/qs_ess_final_state_atlas_continuation.png
   :width: 100%
   :align: center
   :alt: CPU continuation final-state atlas

.. image:: _static/figures/qs_ess_final_state_atlas_direct.png
   :width: 100%
   :align: center
   :alt: CPU direct final-state atlas

Backend-qualified atlases:

.. image:: _static/figures/qs_ess_final_state_atlas_cpu_continuation.png
   :width: 100%
   :align: center
   :alt: CPU continuation final-state atlas

.. image:: _static/figures/qs_ess_final_state_atlas_cpu_direct.png
   :width: 100%
   :align: center
   :alt: CPU direct final-state atlas

.. image:: _static/figures/qs_ess_final_state_atlas_gpu_continuation.png
   :width: 100%
   :align: center
   :alt: GPU continuation final-state atlas

.. image:: _static/figures/qs_ess_final_state_atlas_gpu_direct.png
   :width: 100%
   :align: center
   :alt: GPU direct final-state atlas

The legacy ``geometry_atlas`` alias is regenerated from the CPU continuation
atlas:

.. image:: _static/figures/qs_ess_geometry_atlas.png
   :width: 100%
   :align: center
   :alt: Legacy geometry atlas alias generated from current sweep data

Summary Tables
--------------

The summary-table image is intended for reports and presentations.  The CSV and
JSON are better for analysis scripts.

.. image:: _static/figures/qs_ess_summary_tables_all_policies.png
   :width: 100%
   :align: center
   :alt: Full optimization sweep summary tables

.. image:: _static/figures/qs_ess_summary_table.png
   :width: 100%
   :align: center
   :alt: Legacy summary table alias generated from current sweep data

.. image:: _static/figures/qs_ess_summary_tables_cpu_policies.png
   :width: 100%
   :align: center
   :alt: CPU optimization sweep summary tables

.. image:: _static/figures/qs_ess_summary_tables_gpu_policies.png
   :width: 100%
   :align: center
   :alt: GPU optimization sweep summary tables

Downloadable summaries:

- :download:`summary_all.csv <_static/figures/qs_ess_summary_all.csv>`
- :download:`summary_all.json <_static/figures/qs_ess_summary_all.json>`

Publication Panel
-----------------

The full panel combines objective histories, final-state atlases, and summary
tables.  It is large by design and should be used for review, not as the only
README figure.

.. image:: _static/figures/qs_ess_publication_panel_full.png
   :width: 100%
   :align: center
   :alt: Full publication panel

The legacy ``publication_panel`` alias is regenerated from the same full panel:

.. image:: _static/figures/qs_ess_publication_panel.png
   :width: 100%
   :align: center
   :alt: Legacy publication panel alias generated from current sweep data

Current QI Snapshot
-------------------

The current CPU QI bounded sweep uses ``input.nfp4_QH_warm_start`` as the
input deck, applies a QP preseed for the requested mode/policy, and then
minimizes the QI residual on five surfaces.  In this run, continuation
``max_mode=2`` without ESS reached ``J = 6.85e-4`` and continuation
``max_mode=3`` with ESS reached ``J = 1.25e-3``.  The final ``|B|`` panels are
no longer QH-like; the preseed moves them toward poloidally closed wells before
the QI refinement.

The GPU QI bounded sweep is included as a diagnostic matrix.  It completed
without timeouts on an RTX A4000 system.  The best GPU QI entries in the
current panel are continuation ``max_mode=3`` with ESS (``J = 7.35e-3``) and
direct ``max_mode=2`` with ESS (``J = 6.10e-3``).  Those GPU rows use
intentionally small diagnostic budgets; the CPU QI rows remain the
production-accuracy reference until the GPU exact-replay path is tuned further.
