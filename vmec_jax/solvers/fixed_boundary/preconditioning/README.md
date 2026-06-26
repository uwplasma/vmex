# Fixed-Boundary Preconditioning

This package contains preconditioner payload construction and application for
fixed-boundary solves.

Preconditioners are part of the VMEC nonlinear solve. Keep them independent from
outer optimization objectives so they can be tested against VMEC2000 parity and
used from both CLI and differentiable Python workflows.
