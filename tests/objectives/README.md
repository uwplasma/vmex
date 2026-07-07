# Objective and Stability Tests

This folder contains tests for objective functions and scalar diagnostics used
by optimization workflows:

- quasisymmetry and quasi-isodynamic residuals,
- QI/QS objective assembly and staged QI bookkeeping,
- augmented-Lagrangian helper logic,
- Mercier/Glasser differentiable stability objectives.

These tests validate objective mathematics and differentiability. Full
optimization-script smoke tests live in `tests/optimization/`, and external
VMEC2000 parity gates live in `tests/parity/`.
