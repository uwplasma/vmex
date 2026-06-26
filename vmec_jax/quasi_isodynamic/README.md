# Quasi-Isodynamic Package

This folder owns all QI-specific logic.

- `objectives.py`: differentiable smooth-QI, mirror-ratio, elongation, and
  LgradB objectives.
- `diagnostics.py`: solved-state and Boozer-output diagnostic records, seed
  ranking, and promotion gates.
- `legacy.py`: non-differentiable NumPy/SciPy branch diagnostic used as an
  independent validation reference.
- `optimization.py`: staged QI workflow helpers for examples and sweeps.

Use `import vmec_jax.quasi_isodynamic as qi` for low-level QI work, or
`import vmec_jax as vj` for the documented public API.
