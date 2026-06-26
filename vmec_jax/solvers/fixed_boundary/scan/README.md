# Fixed-Boundary Scan Controller

This package contains VMEC2000-style scan/controller code for fixed-boundary
solves.

Scan helpers own adaptive iteration control, fallback policy, trace formatting,
and postprocessing for VMEC-style runs. Differentiable fixed-branch replay and
optimization logic should call into these helpers through explicit interfaces
rather than duplicating controller state.
