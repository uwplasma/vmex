# Free-boundary solver

Free-boundary implementation is organized around the direct-coil/mgrid provider
path and the NESTOR-style boundary response:

- `boundary_fields.py`, `mgrid.py`, `axis_current.py`, and `control.py`: field
  coupling and control helpers.
- `jax_nestor_operator.py`: JAX-visible source and mode-space response pieces.
- `adjoint/`: branch-local replay, controller fingerprints, custom-VJP helpers,
  dense validation solves, and same-branch derivative evidence.
- `coil_optimization.py`: single-stage coil optimization utilities.
- `validation.py`: bounded parity and physical-gate helpers.

## Current differentiation status

Validated today:

- direct-coil and JAX mgrid external-field sampling,
- accepted-boundary replay through JAX-visible NESTOR/mode-space pieces,
- branch-local physical-scalar derivatives under fixed controller
  fingerprints,
- bounded VMEC2000/mgrid/direct-coil parity gates for physical fixtures.

Still intentionally conservative:

- arbitrary derivatives through hard accepted/rejected branch changes in the
  adaptive host controller. When the branch fingerprint changes, the map is
  nonsmooth; use fingerprint-gated branch-local reports for exact coil
  derivatives.
