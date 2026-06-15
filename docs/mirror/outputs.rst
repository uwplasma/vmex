Mirror Output Files
===================

Mirror outputs use a mirror-native NetCDF file named ``mout_*.nc``.  These
files are not VMEC ``wout_*.nc`` files and are not intended to be consumed by
classic toroidal WOUT readers.

The current schema version is ``0.1`` and stores:

- global attributes including ``geometry_type = "mirror"``,
  ``coordinate_order = "s,theta,xi"``, and
  ``algorithm = "fixed_boundary_variational_chebyshev_lobatto"``;
- coordinate arrays ``s``, ``theta``, ``xi``, physical ``z``, and quadrature
  weights;
- geometry arrays ``r``, ``X``, ``Y``, ``Z``, ``sqrtg``, metric terms, and the
  fixed side-boundary radius;
- field arrays for contravariant, covariant, Cartesian, ``|B|``, and
  ``lambda`` data;
- radial ``Psi_prime``, ``I_prime``, pressure, pressure-gradient, and beta
  profiles;
- scalar energy, residual, Jacobian, field-strength, and mirror-ratio
  diagnostics;
- solve-history arrays for continuation stage, iteration, pressure scale,
  residual, energy, Jacobian, ``|B|``, mirror ratio, step size, and acceptance.

Use ``vmec_jax.mirror.write_mirror_output`` and
``vmec_jax.mirror.load_mirror_output`` for Python roundtrips.  Use
``vmec --plot mout_case.nc --outdir figures`` or
``vmec_jax.mirror.plot_mirror_output`` to write the standard mirror plots.
