# vmec_jax package map

This directory contains the Python package.  Keep new code in a domain folder
when possible; root-level modules are for public facades, compact physics
building blocks, or compatibility entry points.

## Public facades

- `api.py`: curated public imports.
- `driver.py`: high-level VMEC workflows used by Python callers and the CLI.
- `cli.py` and `__main__.py`: command-line entry points.
- `solve.py`, `wout.py`, and `free_boundary.py`: thin public facades for
  heavily used solver and validation APIs. Prefer adding implementation code
  under `solvers/` or `io/` and re-exporting here only when the API is public.

## Domain folders

- `solvers/`: fixed-boundary and free-boundary solver implementation.
- `optimizers/`: reusable optimization algorithms and residual builders.
- `external_fields/`: coil, mgrid, and ESSOS field providers.
- `io/`: persisted VMEC data formats, especially WOUT netCDF helpers.
- `drivers/`: implementation helpers for `driver.py`; these are not general
  input/output utilities.
- `resources/`: tiny package-bundled inputs needed after `pip install`, such
  as the `vmec --test` quick-start deck. User-facing examples live in
  `examples/data/`.

## Root physics modules

Small standalone physics modules may remain at the root when they are commonly
imported directly, for example `quasisymmetry.py`, `finite_beta.py`,
`profiles.py`, `boundary.py`, and `field.py`.  Large topic families should live
in domain folders with clear names.  QI-related implementation belongs under
`quasi_isodynamic/`, and low-level VMEC force/residue kernels belong under
`kernels/`.

## Naming rules

- Prefer descriptive domain names over generic names like `finish`, `io`, or
  `utils`.
- Avoid creating one-file folders unless they are a stable public domain.
- Keep public facades thin: implementation goes in domain folders, and
  user-facing imports are re-exported only where they simplify the documented
  API.
- Add a short README when creating a package folder that is not obvious from
  its name.
