# Adjoint and Differentiation Tests

This folder contains tests for fixed-boundary discrete-adjoint machinery and
fast differentiation API seams:

- accepted-point replay and chunked Jacobian construction,
- cotangent packing/unpacking helpers,
- branch-local exact callback metadata,
- small AD/JVP/VJP helper contracts.

Production solver behavior is tested in `tests/solvers/`; physics kernels are
tested in `tests/kernels/`.
