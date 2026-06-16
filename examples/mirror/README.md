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
```

Each script writes a mirror-native ``mout_*.nc`` file and, unless
``--no-plots`` is passed, a set of PNG diagnostics including horizontal
``z``-axis geometry, boundary magnetic-field direction with field-line traces,
``|B|``, beta/twist-proxy, magnetic-well-proxy, and residual/step-history
figures.  These are research fixtures for the scalar-pressure fixed-boundary
mirror path, not WHAM predictive modelling tools.  For physically axisymmetric
mirrors use the cylinder, flared-tube, or WHAM examples; the nonaxisymmetric
example is a solver/plot stress test.
