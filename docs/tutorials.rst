Tutorials
=========

Worked, single-file tutorials are being added together with the rewritten
``examples/`` gallery. Each tutorial will be a runnable script of under ~120
lines using only the public API, with the achieved results recorded in its
docstring and smoke-tested in CI. The planned set:

- **Fixed-boundary run** — solve an input deck, plot the equilibrium, write
  the ``wout`` file, and run the Boozer transform.
- **Free-boundary with an mgrid file** — a CTH-like stellarator from a
  precomputed vacuum-field grid.
- **Free-boundary directly from coils** — the same physics via ESSOS
  Biot-Savart coil fields, no mgrid interpolation.
- **Free-boundary beta scan** — boundary evolution with increasing pressure
  for a tokamak and a stellarator coil set, mgrid vs direct-coil overlay.
- **QA / QH / QP / QI optimization** — from a circular torus to precise
  quasisymmetric and quasi-isodynamic configurations with gradient-based
  least squares (implicit differentiation, no finite differences).
- **Single-stage optimization** — simultaneous coil and plasma optimization
  with ESSOS (advanced).

Until the tutorial pages land, start from:

- :doc:`quickstart` — the CLI and minimal Python API workflow,
- the ``examples/`` directory of the repository and its input decks under
  ``examples/data/``,
- :doc:`optimization` — the objectives and differentiation building blocks
  the optimization tutorials are built on.
