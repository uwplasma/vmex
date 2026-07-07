# Quasi-Isodynamic Package

This folder owns all QI-specific logic.

- `objectives.py`: differentiable smooth-QI, mirror-ratio, elongation, and
  LgradB objectives.
- `optimization_terms.py`: SIMSOPT-style QI objective-term wrappers used by
  fixed-boundary optimization examples.
- `diagnostics.py`: solved-state and Boozer-output diagnostic records, seed
  ranking, and promotion gates.
- `legacy.py`: non-differentiable NumPy/SciPy branch diagnostic used as an
  independent validation reference.
- `optimization.py`: staged QI workflow helpers for examples and sweeps.
- `seed_search.py`: package-owned basin-survey and local landscape-scan
  primitives used by staged QI workflows and developer diagnostic wrappers.

Use `import vmec_jax.quasi_isodynamic as qi` for low-level QI work, or
`import vmec_jax as vj` for the documented public API.
