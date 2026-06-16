Mirror Examples
===============

These examples exercise the experimental fixed-boundary mirror backend from a
source checkout.  They intentionally use low resolution and small iteration
budgets so they run quickly.

Run from the repository root:

```bash
python examples/mirror/fixed_cylinder.py --outdir results/mirror/cylinder
python examples/mirror/fixed_flared_tube.py --outdir results/mirror/flared
python examples/mirror/wham_vacuum_boundary.py --outdir results/mirror/wham
python examples/mirror/nonaxisymmetric_boundary.py --outdir results/mirror/nonaxisymmetric
python examples/mirror_two_coil_axisym.py --outdir results/mirror/two_coil_axisym
python examples/mirror_finite_current_pitch.py --outdir results/mirror/finite_current_pitch
python examples/mirror_fixed_boundary_solve_diagnostic.py --outdir results/mirror/fixed_boundary_solve_diagnostic
python examples/mirror_manufactured_fixed_boundary.py --outdir results/mirror/manufactured_fixed_boundary
```

The physical mirror examples write a mirror-native ``mout_*.nc`` file and,
unless ``--no-plots`` is passed, a set of PNG diagnostics including horizontal
``z``-axis geometry, boundary magnetic-field direction with field-line traces,
``|B|``, beta/twist-proxy, magnetic-well-proxy, and residual/step-history
figures.  The manufactured validation example writes metrics and targeted
convergence/geometry/``|B|`` plots rather than a production ``mout`` file.
These are research fixtures for the scalar-pressure fixed-boundary mirror path,
not WHAM predictive modelling tools.  For physically axisymmetric mirrors use
the cylinder, flared-tube, or WHAM examples; the nonaxisymmetric example is a
solver/plot stress test.

The same standard figure bundle is available from the CLI:

```bash
vmec --plot results/mirror/two_coil_axisym/mout_two_coil_axisym.nc --outdir results/mirror/two_coil_axisym/cli_figures
```

For mirror ``mout_*.nc`` files, ``vmec --plot`` writes nested ``r-z`` surfaces,
cross sections, 3-D boundary ``|B|`` with field-line overlays, boundary field
direction, ``|B|`` maps, Jacobian, pressure/beta, radial diagnostics, and
residual/force history plots.

The root-level ``examples/mirror_two_coil_axisym.py`` script is the first
analytic benchmark example: it builds a fixed boundary from the closed-form
on-axis field of two equal circular coils, overlays the mirror on-axis ``B_z``
against that analytic expression, draws the coils in the 3-D views, compares
low-radius off-axis ``B_r``/``B_z`` against the circular-loop Biot-Savart field,
and writes a small ``ns``/``nxi`` convergence study.

The root-level ``examples/mirror_finite_current_pitch.py`` script uses the same
two-coil fixed boundary with nonzero ``I'`` so the boundary field-line traces
have visible cap-to-cap pitch.

The root-level ``examples/mirror_fixed_boundary_solve_diagnostic.py`` script
runs an actual L-BFGS fixed-boundary relaxation from a perturbed interior state.
Its default diagnostic uses ``ns_array=31``, ``maxiter=2000``, and explicit
``ftol=1e-12``/``gtol=1e-12`` and writes a JSON table with optimizer status,
iteration counts, residuals, ``fsq``, and plot paths.

The root-level ``examples/mirror_manufactured_fixed_boundary.py`` script solves
a sourced manufactured fixed-boundary problem with a known stationary state. It
uses the same reduced-coordinate layout and geometry scaling as the mirror
solver, then applies an exact-Hessian damped residual iteration to verify that a
perturbed projected state can reach the requested projected ``gtol``.
