Optimized Fixed-Boundary Merge Notes
====================================

This page records the final merge scope for the optimized non-autodiff solver
work.

Final decision
--------------

Merge the optimized non-autodiff fixed-boundary path as the default user
experience for ordinary CLI and Python usage.

Do not change these defaults in the same step:

- keep parity-oriented routes available,
- keep autodiff-oriented workflows available,
- keep free-boundary on the current robust default controller,
- do not market the merged result as a blanket VMEC2000 runtime win.

What changes on merge
---------------------

- ordinary non-autodiff fixed-boundary CLI runs use the optimized controller
  automatically,
- ordinary non-autodiff Python fixed-boundary runs do the same,
- accelerated finishes accept only when the requested final-stage ``FTOL`` is
  satisfied through ``fsqr``, ``fsqz``, and ``fsql``,
- staged and symmetry-sensitive fixed-boundary cases keep the more
  conservative handoff/follow-up logic that proved reliable on the bundled
  matrix,
- free-boundary performance improvements are included, but free-boundary does
  not switch to the accelerated controller by default.

Final branch-head evidence
--------------------------

Local validation on the final merge head:

- ``pytest -q`` completed successfully:
  ``186 passed, 12 skipped, 58 warnings``,
- ``SPHINX_FAST=1 LC_ALL=C LANG=C python -m sphinx -W -j auto -b html docs docs/_build/html_fastcheck``
  passed.

Fresh final-head artifacts:

- fixed-boundary optimized readiness:
  ``outputs/readiness_fixed_all_20260313/summary.json``
- fixed-boundary VMEC2000-vs-optimized warmed runtime matrix:
  ``outputs/fixed_runtime_vmec2000_accel_cpu_warm_20260313/summary.json``
- free-boundary VMEC2000-vs-default warmed runtime matrix:
  ``outputs/free_runtime_vmec2000_cpu_warm_20260313/summary.json``

Key results from those artifacts:

- fixed-boundary optimized path converged on all 16 bundled cases,
- fixed-boundary optimized path is faster on 13 of 16 cases, effectively
  neutral on 1, and slower on 2,
- all 21 shipped ``vmec_jax`` rows in the final VMEC2000 runtime matrix
  converged,
- free-boundary DIII-D remains much slower than VMEC2000 on CPU, but the
  branch still improved the robust default path materially
  (for example ``DIII-D_lasym_false`` is down to about ``113.78s`` warmed on
  the same host, versus the earlier ``~174s`` branch checkpoint).

Representative fixed-boundary before/after points
-------------------------------------------------

Before = old default fixed-boundary controller on the same final head.
After = optimized controller that becomes the ordinary non-autodiff default.

- ``ITERModel``: ``8.86s`` -> ``4.48s``
- ``LandremanPaul2021_QA_reactorScale_lowres``: ``47.17s`` -> ``45.52s``
- ``LandremanSenguptaPlunk_section5p3_low_res``: ``15.86s`` -> ``7.48s``
- ``up_down_asymmetric_tokamak``: ``24.58s`` -> ``2.97s``

Scope boundary
--------------

This merge is about the default non-autodiff fixed-boundary user path.

It is not a claim that:

- accelerated free-boundary is ready to be the default controller,
- the merged code is broadly faster than VMEC2000 on same-host CPU,
- parity-oriented or autodiff-oriented workflows should be removed or hidden.

Reviewer checklist
------------------

1. Confirm the default-path scope:

   .. code-block:: bash

      git diff main...HEAD -- vmec_jax/driver.py vmec_jax/cli.py README.md

2. Confirm the main local gates:

   .. code-block:: bash

      pytest -q
      SPHINX_FAST=1 LC_ALL=C LANG=C python -m sphinx -W -j auto -b html docs docs/_build/html_fastcheck

3. Inspect the final artifacts if needed:

   - ``outputs/readiness_fixed_all_20260313/summary.json``
   - ``outputs/fixed_runtime_vmec2000_accel_cpu_warm_20260313/summary.json``
   - ``outputs/free_runtime_vmec2000_cpu_warm_20260313/summary.json``
