# Free-Boundary Adjoint Helpers

This package contains direct-coil and free-boundary derivative helpers:
NESTOR-mode operators, accepted-boundary replay, branch-local reports, and
controller replay utilities.

Current production claims are branch-local and fingerprint-gated. Values come
from complete free-boundary solves; derivatives are validated for the same
accepted/rejected controller fingerprint. Do not claim arbitrary adaptive
branch differentiation from this package until hard branch selection is itself
made JAX-visible and AD-vs-FD validated.
