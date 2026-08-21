# Changelog

## Unreleased (0.6.0)

VMEX 0.6 extends the differentiable equilibrium contract beyond fixed-boundary
shape objectives:

- `Equilibrium` and `VmecExtender` expose interior and exterior magnetic-field
  values, spatial derivatives, and boundary-parameter VJPs. Finite-beta
  exterior fields use the separately packaged `virtual-casing-jax` backend.
- Fixed-boundary optimization interfaces and free-boundary derivative examples
  cover boundary, profile, current, direct-coil, bootstrap, stability, QS/QI,
  maximum-J, and neoclassical objectives without selecting one optimizer.
- The NESTOR free-boundary root has certified coupled and boundary-Schur
  implicit-adjoint lanes. Fixed-boundary Jacobians now enforce their numerical
  certificate: automatic mode falls back to the reverse adjoint, while a
  forced solver raises rather than returning an approximate derivative.
- LASYM reconstruction, Boozer projection, Mercier/Glasser diagnostics, APHI
  validation, restart handling, and the optimization examples gained
  regression, finite-difference, VMEC2000, and bounded campaign coverage.

Release gates are a green PR suite, Nightly and Weekly physics campaigns,
documentation link checking, wheel/sdist installation tests on Python 3.10 and
3.12, plus compatibility of the core coil/CLI adapters with released ESSOS
0.16. New ESSOS-owned helper APIs remain advisory until they are independently
reviewed and released. The `v0.6.0` GitHub release must be marked latest; asset
bundle releases are supporting data releases, not the project release.

## 0.5.0

VMEX 0.5 makes converged equilibria optimizer-neutral building blocks:

- `VmecProblem` exposes consistent SciPy and JAX residual, Jacobian, scalar,
  and value/gradient contracts for QI, QS, geometry, profile, stability, and
  user-authored objectives.
- Exact implicit derivatives, automatic Jacobian direction, hot and
  perturbation restarts, ESS scales, bounded Jacobian batches, and smooth
  failed-trial handling are production defaults.
- `forward_ftol`, `forward_max_iterations`, and `max_fsq_ratio` make forward
  convergence and derivative certification explicit. Evaluations report FSQ
  and certification diagnostics.
- J-invariant QI and maximum-J objectives can share one Boozer transform;
  free-boundary, virtual-casing, bootstrap, stability, plotting, and standard
  wout workflows remain available.
- Automatic ensemble and finite-difference worker counts respect process CPU
  limits on shared and scheduler-managed machines.

Upgrade notes:

- Tuple weights use cost semantics in `VmecProblem.from_tuples`: weight `w`
  contributes `w * (f - target)^2 / 2`. The compatibility
  `optimize.least_squares` wrapper retains its documented residual-weight
  convention.
- The experimental cross-iterate `recycle=True` Jacobian lane was removed;
  it was slower than the certified block/GMRES defaults. The production
  reverse adjoint continues to use GCROT internally.
- Decorative optimization scans and duplicate ESS examples were removed.
  Use `benchmarks/optimization.py` and the backend-specific QI examples.

Before tagging `v0.5.0`, the release PR must pass the PR gate, the bounded
Nightly physics campaigns, documentation link checking, an sdist/wheel build,
and installation/import tests of both artifacts. Publishing a GitHub release
whose tag matches `pyproject.toml` triggers the trusted PyPI workflow.

Earlier release notes live on [GitHub Releases](https://github.com/uwplasma/vmex/releases).
