# Fixed-Boundary Residual Iteration

This package contains the residual iteration loop and its supporting setup,
policy, runtime, and update helpers.

`iteration.py` is still the largest fixed-boundary solve seam. New code should
prefer adding focused helpers in this package instead of growing the main loop.
The long-term goal is to keep the loop readable while preserving VMEC2000
parity and differentiability hooks.
