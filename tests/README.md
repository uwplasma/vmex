# Test Suite Map

The test suite is organized by the code path or artifact being validated:

- `tests/solvers/`: production solver algorithms, residual iteration, implicit
  differentiation, preconditioners, and staged solver examples.
- `tests/adjoints/`: fixed-boundary discrete-adjoint replay, exact callback
  seams, and fast differentiation API contracts.
- `tests/assets/`: downloader, manifest, and optional validation-asset metadata
  gates.
- `tests/drivers/`: command-line entrypoints, Python driver APIs, run policies,
  and finish/output orchestration.
- `tests/external_fields/`: differentiable coil, mgrid, ESSOS-adapter, and
  robust-coil provider tests.
- `tests/free_boundary/`: production free-boundary, direct-coil, ESSOS, and
  branch-local derivative behavior.
- `tests/io/`: public file formats and interchange artifacts, currently WOUT.
- `tests/kernels/`: low-level VMEC physics kernels, finite-beta terms,
  real-space geometry, force/residue helpers, and small parity identities.
- `tests/parity/`: VMEC2000/VMEC++ agreement, bundled parity fixtures, and
  physics parity gates.
- `tests/postprocessing/`: Boozer, plotting, profile examples, bootstrap
  current, field-line diagnostics, and stability diagnostic examples.
- `tests/objectives/`: differentiable QS/QI objective, augmented-Lagrangian,
  and Mercier/Glasser objective tests.
- `tests/optimization/`: fixed-boundary and QI/QS optimization workflows.
- `tests/diagnostics/`: diagnostics, renderers, benchmark parsers, and repo
  health checks. These support validation but are not the solver API itself.
- `tests/fixtures/`: small reusable fixtures.

Root-level tests are being reserved for cross-cutting public API, CLI, kernel,
parity, and physics-gate coverage while domain-specific tests move into the
folders above.
