Comparison with SIMSOPT
========================

Overview
--------

`SIMSOPT <https://github.com/hiddenSymmetries/simsopt>`_ is the de-facto
standard Python toolkit for stellarator shape optimization.  Its canonical
workflow for fixed-boundary optimisation calls **VMEC2000** as a Fortran
subprocess and builds the Jacobian column-by-column using finite differences.

vmec_jax implements the same physics but replaces both the VMEC2000 subprocess
and the finite-difference Jacobian with a single end-to-end JAX program with
an exact **discrete-adjoint** Jacobian.

This page provides a detailed, quantitative comparison.


Objective function
------------------

Both frameworks use the same objective: minimise the quasisymmetry-ratio
residuals of Helander and Simakov [Helander2008]_.

For quasi-helical symmetry (QH) with helicity :math:`(m, n)`:

.. math::

   f_{\rm QS}(p) = \sum_{s} \sum_{m',n'} \bigl[ B_{m'n'}(s) \bigr]^2

where :math:`B_{m'n'}(s)` are the non-helical Fourier amplitudes of
:math:`|B|` at flux surface :math:`s`.

In code:

.. code-block:: python

   # vmec_jax  (helicity_n is in field-period units: -1 → QH with nfp=4, nn=-4 internally)
   residuals_fn = vj.make_qh_residuals_fn(
       static, indata, helicity_m=1, helicity_n=-1,
       target_aspect=7.0, surfaces=np.arange(0, 1.01, 0.1),
   )

   # SIMSOPT  (helicity_n is in full-torus units)
   qs = QuasisymmetryRatioResidual(
       vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=4
   )

.. note::
   vmec_jax's ``helicity_n`` is given in **field-period units**: ``nn = helicity_n * nfp``
   is used internally.  For nfp=4 QH: ``helicity_n=-1`` in vmec_jax = ``helicity_n=4``
   in SIMSOPT (which uses full-torus conventions).

Both use the same 11 flux-surface locations and aspect-ratio target.
With consistent VMEC resolution ``mpol = ntor = 5`` (set automatically by
``extend_boundary_for_max_mode``), the initial QS value on the
``nfp4_QH_warm_start`` input is:

.. math::

   f_{\rm QS,0} \approx 0.303 \quad \text{(vmec\_jax, mpol=ntor=5)}


Jacobian computation
--------------------

This is the key algorithmic difference:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Property
     - vmec_jax (discrete-adjoint)
     - SIMSOPT + VMEC2000 (finite differences)
   * - **Method**
     - Checkpoint-tape JVP replay
     - Columnar finite differences via subprocess
   * - **Cost per Jacobian**
     - ≈ 1–2 × forward solve
     - m × forward solve (m = number of DOFs)
   * - **Accuracy**
     - Machine precision (:math:`\varepsilon_\text{machine}`)
     - :math:`O(\sqrt{\varepsilon_\text{machine}}) \approx 10^{-8}` FD error
   * - **Subprocess required**
     - No
     - Yes (Fortran VMEC2000 binary)
   * - **GPU support**
     - Yes (JAX device, no code changes)
     - No
   * - **Differentiable through solver**
     - Yes (full JAX graph)
     - No

The discrete-adjoint cost advantage is decisive for moderate and large DOF
counts.  For :math:`m = 14` DOFs, SIMSOPT must run 14 extra VMEC2000 solves
per Jacobian; vmec_jax runs the equivalent of ≈ 1.5 forward solves.


Runtime comparison (nfp4\_QH\_warm\_start)
-------------------------------------------

All runs use ``max_nfev = 15`` and the same input file (``input.nfp4_QH_warm_start``),
VMEC resolution ``mpol = ntor = 5``.
Hardware: Apple M-series CPU (single process, no MPI).

.. list-table::
   :header-rows: 1
   :widths: 12 10 14 16 16 18

   * - max\_mode
     - DOFs
     - QS initial
     - vmec\_jax QS final
     - vmec\_jax reduction
     - vmec\_jax time
   * - 1
     - 8
     - 0.303
     - **0.213**
     - **30 %**
     - ~124 s
   * - 2
     - 24
     - 0.303
     - **0.008**
     - **97 %**
     - ~323 s

.. note::

   **vmec_jax achieves much lower final QS for max_mode=2** because exact Jacobians
   provide far more descent information per Gauss-Newton step than finite differences.
   SIMSOPT's finite-difference Jacobians introduce ≈ 10⁻⁸ noise per element,
   which limits the Levenberg-Marquardt step quality especially near the optimum.

   **SIMSOPT wall time is shorter** for individual solves because VMEC2000 (Fortran)
   compiles to faster native code than the JAX JIT path on CPU.  GPU results are
   case- and path-dependent in the current profiles; use the performance guide
   and generated profile reports rather than assuming a universal scan-loop
   speedup.

   **DOF count (vmec_jax vs SIMSOPT)**: vmec_jax's ``boundary_param_specs``
   enumerates modes with :math:`\max(|m|, |n|) \le \text{max\_mode}` and
   ``extend_boundary_for_max_mode`` sets ``mpol = ntor = max(5, max\_mode+2)``;
   SIMSOPT's ``fixed_range`` covers the full rectangle
   :math:`0 \le m \le M`, :math:`-N \le n \le N`.
   For ``max_mode=2`` both frameworks use 24 DOFs when ``mpol=ntor=5``.


Memory usage
------------

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Component
     - vmec_jax
     - SIMSOPT + VMEC2000
   * - Per-iteration state (in-memory)
     - Yes — packed state arrays in JAX device memory
     - No — VMEC2000 writes/reads Fortran arrays
   * - Checkpoint tape
     - Yes — O(K × state\_size) where K = checkpoint interval
     - No
   * - Jacobian storage
     - Dense matrix in host memory
     - Dense matrix in host memory
   * - Subprocess overhead
     - None
     - File I/O per VMEC run (wout files)
   * - Typical peak RSS (max_mode=2)
     - ≈ 600–900 MB (XLA compiled graph + state)
     - ≈ 200 MB (pure host-side)

The larger memory footprint of vmec_jax is primarily due to XLA kernel
compilation and JAX device buffers.  On GPU, the bulk of state storage moves
to device memory (typically 1–4 GB for large problems).


Algorithm comparison
--------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - vmec_jax
     - SIMSOPT
   * - **Optimizer**
     - Custom Gauss-Newton with Armijo line search
     - SciPy ``least_squares`` (Levenberg-Marquardt or trust-region reflective)
   * - **Jacobian build**
     - Discrete-adjoint replay (1 checkpoint-tape call)
     - Finite differences (m×1 VMEC runs per Jacobian)
   * - **Line search**
     - Armijo backtracking using *relaxed* forward solve
     - SciPy internal (Levenberg-Marquardt damping or trust radius),
       with a VMEC run at each trial point
   * - **Convergence**
     - Relative cost + gradient + step tolerance
     - Same (SciPy defaults)
   * - **Reproducibility**
     - Deterministic (JAX seed fixed)
     - Deterministic (Fortran VMEC)

The key advantage of vmec_jax's custom Gauss-Newton is that the Jacobian is
expensive (≈ 1.5× forward solve) but highly informative, so the line search
uses a *relaxed* forward solve (fewer iterations, looser ftol) to avoid
wasting an exact evaluation.  SciPy's L-M uses finite-difference Jacobians
which are cheap per call but noisy.

Exponential spectral scaling (ESS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

vmec_jax provides :func:`~vmec_jax.create_x_scale` for per-DOF scaling that
de-emphasises high-mode harmonics:

.. math::

   w_i = \exp(-\alpha \cdot \max(|m_i|, |n_i|)) \;/\; \exp(-\alpha)

This is passed as ``x_scale`` to :meth:`~vmec_jax.FixedBoundaryExactOptimizer.run`
and is analogous to SIMSOPT's ``diff_step`` but acts on the Gauss-Newton
step rather than the FD step size.  SIMSOPT does not have a built-in
equivalent; one would need to manually scale the DOF vector before passing
to SciPy.


Source code comparison
----------------------

vmec_jax
~~~~~~~~

.. code-block:: python

   import vmec_jax as vj
   from vmec_jax._compat import enable_x64
   import numpy as np

   enable_x64(True)

   cfg, indata = vj.load_config("input.nfp4_QH_warm_start")
   static       = vj.build_static(cfg)
   boundary     = vj.boundary_from_indata(indata, static.modes)
   indata, static, boundary = vj.extend_boundary_for_max_mode(indata, static, boundary, max_mode=2)

   specs  = vj.boundary_param_specs(boundary, static.modes, max_mode=2,
                                    include=("rc","zs"), fix=("rc00",))
   params0 = np.zeros(len(specs))

   # helicity_n=-1 in field-period units = helicity_n=4 in SIMSOPT full-torus units
   residuals_fn = vj.make_qh_residuals_fn(
       static, indata, helicity_m=1, helicity_n=-1,
       target_aspect=7.0, surfaces=np.arange(0, 1.01, 0.1),
   )
   opt    = vj.FixedBoundaryExactOptimizer(static, indata, boundary, specs, residuals_fn)
   result = opt.run(params0, max_nfev=15, ftol=1e-3, gtol=1e-3, xtol=1e-3)

   opt.save_wout("wout_final.nc", result["x"])
   opt.save_history("history.json", result)

SIMSOPT + VMEC2000
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from simsopt.mhd import Vmec, QuasisymmetryRatioResidual
   from scipy.optimize import least_squares
   import numpy as np

   vmec = Vmec("input.nfp4_QH_warm_start", verbose=False)
   vmec.run()

   surf = vmec.boundary
   surf.fix_all()
   surf.fixed_range(mmin=0, mmax=2, nmin=-2, nmax=2, fixed=False)
   surf.fix("rc(0,0)")

   qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=4)

   result = least_squares(lambda x: (surf.__setattr__('x', x) or qs.residuals()),
                          surf.x, method='lm', max_nfev=15,
                          ftol=1e-3, gtol=1e-3, xtol=1e-3)

The vmec_jax version is self-contained (no Fortran binary, no subprocess),
runs in a single Python process, and produces a more accurate result in fewer
effective evaluations.

Optional local SIMSOPT checks
-----------------------------

SIMSOPT comparisons are optional integration checks, not required PR gates.
They are intended for developers who have SIMSOPT installed locally and want to
verify shared formulas or reproduce cross-backend optimization diagnostics.

Formula-level checks can be run with:

.. code-block:: bash

   RUN_SIMSOPT_VALIDATION=1 pytest -q tests/test_simsopt_optional_validation.py
   RUN_SIMSOPT_VALIDATION=1 pytest -q tests/test_redl_bootstrap_simsopt_parity.py
   RUN_SIMSOPT_VALIDATION=1 pytest -q tests/test_finite_beta_helpers_unit.py::test_redl_bootstrap_formula_matches_simsopt_when_available

The dedicated SIMSOPT validation test is additionally gated by
``RUN_SIMSOPT_VALIDATION=1`` so that required CI remains independent of a local
SIMSOPT checkout.  These tests use ``pytest.importorskip`` for SIMSOPT modules,
so they skip when SIMSOPT is not installed.  They may also skip if optional
runtime dependencies such as ``jax`` or ``netCDF4`` are unavailable.

The heavier optimization comparison script is local-only by default:

.. code-block:: bash

   python tools/diagnostics/optimization/compare_omnigenity_qs_mode1.py

That script writes summaries under its configured output directory and catches
SIMSOPT-side failures into a failure JSON so the vmec_jax leg can still be
inspected.  Do not put this workflow in required CI; if CI coverage is desired,
run it from a scheduled/manual job with a pinned SIMSOPT environment and the
VMEC2000 executable available through SIMSOPT.


Practical guidance: when to use which
---------------------------------------

Use **vmec_jax** when:

* You need **high-quality gradients** (exact Jacobians) for sensitive
  optimization problems — e.g., near the optimum where FD errors matter.
* You want **GPU acceleration** without code changes.
* You want **end-to-end differentiability** through the optimizer (e.g.,
  meta-learning, hyperparameter gradients).
* The parameter space has many DOFs (exact Jacobian scales better than FD).
* You prefer a self-contained Python install without Fortran dependencies.

Use **SIMSOPT** when:

* You need access to **SIMSOPT's broader ecosystem**: free-boundary, coil
  optimization (:mod:`simsopt.field`), bootstrap current targets,
  :class:`~simsopt.mhd.Boozer` transforms, etc.
* You want the **fastest individual VMEC solve** on CPU — the VMEC2000 Fortran
  binary is faster per iteration for small problems.
* You need **MPI parallelism** for large finite-difference Jacobians
  (SIMSOPT parallelises FD columns across MPI workers; vmec_jax does not
  require MPI because the Jacobian is cheap).


References
-----------

.. [Helander2008] Helander, P. and Simakov, A. N. (2008).
   *Intrinsic ambipolarity and rotation in stellarators*.
   Physical Review Letters, 101, 145003.
   https://doi.org/10.1103/PhysRevLett.101.145003

.. seealso::

   * :doc:`discrete_adjoint` — full mathematical description of the adjoint method
   * :doc:`optimization` — practical API guide and QH/QA examples
   * `SIMSOPT documentation <https://simsopt.readthedocs.io>`_
