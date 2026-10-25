# Trace a resolved equilibrium

The tracer follows the field of the WOUT, so it is only as accurate as the
equilibrium. Losses from a nearly quasi-symmetric field are set by
symmetry-breaking `|B|` components of order 1e-3, the size of the error of an
under-resolved free-boundary solve. `trace_alphas` records
`metadata["spectral_truncation"]`, the largest LCFS `R, Z` amplitude at
`m = mpol - 1` or `|n| = ntor` relative to the `m = 1, n = 0` amplitude, and
warns above `RESOLUTION_TOLERANCE = 3e-3`. `--trace-no-resolution-check`
(`check_resolution=False` in Python) silences it.

## What to do when it warns

1. Re-solve at larger `mpol` and `ntor` (6, then 7) and trace the same births.
2. Compare `|B|` on a flux surface with the coil field: the loss fraction
   follows that error, not `mpol` itself.
3. Quote the coil-field loss when the WOUT losses do not settle.

## Landreman-Paul QA coils

Free-boundary vacuum WOUTs from the ESSOS Landreman-Paul QA coils, scaled to
ARIES-CS (`volavgB`); 1,000 alphas born at `s = 0.25`, the same births in
every row, 10 ms. The error is `sqrt(f (1 - f) / N)`; `|B|` error is the
largest relative difference from the coils at `s = 0.25`.

| field | `\|B\|` error | lost % (Boozer / ESSOS VMEC / SIMPLE / FIRM3D) |
|---|---|---|
| mpol = ntor = 5, ns = 31 | 2.8e-3 | 6.9 ± 0.8 / 7.1 / 5.6 / 6.9 |
| mpol = ntor = 5, ns = 101 | 2.8e-3 | 6.4 ± 0.8 |
| mpol = ntor = 6, ns = 31 | 8.2e-4 | 0.9 ± 0.3 / 1.1 / 1.1 / 1.1 |
| mpol = ntor = 6, ns = 51 | 8.3e-4 | 0.7 ± 0.3 |
| mpol = ntor = 7, ns = 51 | 1.5e-3 | 5.2 ± 0.7 / 5.9 (a 1e-10 solve) |
| mpol = ntor = 7, ns = 101 | 1.4e-3 | 5.1 ± 0.7 |
| coils, guiding centre, tricubic grid 64 / 96 | 7.7e-8 | 0.7 ± 0.3 / 0.9 ± 0.3 |

mpol = ntor = 8 did not converge (`fsq` 1e-3 after 20,000 iterations). On a
200-particle subset, enriched in losses, direct Biot-Savart loses 3.5 %, the
tricubic grid 3.5 %, mpol = 6 3.5 %, mpol = 7 18 % and mpol = 5 34.5 %;
DESC on a VMECIO fit of the mpol = 5 WOUT loses 37 %, but that fit is off by
1.5 cm in `R, Z`, so it is not an independent check.

Every code agrees on a given WOUT, and at mpol = 5 neither the Boozer mode
cut, the Boozer resolution, the integrator nor `ns` moves the result. The
mpol = 7 WOUT is converged to `fsq` 1e-12 yet sits further from the coil field
than mpol = 6 at both ns = 51 and 101, and loses accordingly: the loss fraction
has not converged in `mpol` here, and the coil-field number is the reference.
