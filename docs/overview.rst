Overview
========

What is VMEC?
-------------

VMEC (Variational Moments Equilibrium Code) computes ideal-MHD equilibria in
toroidal geometry by representing flux surfaces with Fourier series and solving
a fixed-boundary equilibrium problem. The canonical public reference
implementation is **VMEC2000** (Fortran).

What is vmec-jax?
-----------------

``vmec-jax`` is a from-scratch Python package that ports VMEC2000’s fixed-boundary
pipeline to JAX:

- vectorized numerical kernels (``jax.numpy`` + ``jit``),
- end-to-end differentiation through equilibrium objectives and solvers,
- parity-first development against VMEC2000 ``wout_*.nc`` reference outputs.

The recommended end-to-end entrypoint is the axisymmetric showcase script:
``examples/showcase_axisym_input_to_wout.py``. It runs bundled inputs, writes
new ``wout_*.nc`` files, produces VMEC-style plots, and prints a small parity
summary against bundled VMEC2000 reference ``wout`` files.

.. figure:: _static/figures/showcase_shaped_tokamak_pressure_surfaces.png
   :alt: Nested flux surfaces (phi=0)
   :align: center
   :width: 90%

   Example output: nested flux-surface cross-sections at ``phi=0`` for the
   bundled ``shaped_tokamak_pressure`` case.

Scope (current)
---------------

The current parity target is **fixed-boundary, axisymmetric** VMEC2000:

- ``ntor = 0`` and ``nfp = 1`` (axisymmetric), with ``lasym = False``.
- Fixed boundary only (free boundary deferred).

Many low-level kernels are also exercised on bundled 3D reference ``wout`` files,
but end-to-end nonlinear solve parity for 3D cases is intentionally deferred.

Initial guess
-------------

``vmec_jax`` initializes Fourier coefficients with VMEC-style axis regularity:
``rho = sqrt(s)`` scaling for ``m>0`` modes and a linear blend between axis and
boundary for ``m=0`` coefficients when axis inputs are provided.

When axis inputs are not provided, ``vmec_jax`` can recompute an axis guess by
searching each toroidal plane for an axis position that maximizes a minimum
Jacobian proxy on VMEC’s reduced theta grid.

Design principles
-----------------

Minimal dependencies
~~~~~~~~~~~~~~~~~~~~

Core runtime:

- ``numpy`` (required)
- ``jax`` + ``jaxlib`` (optional, required for differentiation and performance)

Optional:

- ``netCDF4`` for reading/writing ``wout_*.nc`` regression data.

Regression-first development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bundled VMEC2000 ``wout_*.nc`` files are treated as ground truth for:

- Fourier mode ordering and normalization,
- Nyquist ``sqrt(g)`` and B-field coefficients,
- scalar integrals like ``wb`` and total volume,
- scalar residual measures (``fsqr/fsqz/fsql``).

Nonlinear iteration parity is tracked separately from kernel parity on reference
states (which is solver-free and therefore isolates conventions).
