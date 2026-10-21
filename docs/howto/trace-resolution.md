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

1. Re-solve at larger `mpol` and `ntor`. `ntor` must resolve the coil ripple:
   the eight Landreman-Paul QA coils per period put `|B|` harmonics of 1e-4
   at `n = 9-11` per period inside `s = 0.25`, so `ntor <= 8` leaves a 5e-4
   error even on a fixed boundary traced from the coils; `ntor = 12` brings
   it to 8e-5.
2. Relax a free-boundary solve: `vmex.relax_free_boundary_constraint(inp,
   result, external_field=field)` restarts the converged state with the
   spectral constraint off (see below).
3. Compare `|B|` on a flux surface with the coil field: the loss fraction
   follows that error, not `mpol` itself.

## Why a converged free-boundary field is off

The spectral-condensation force that fixes the poloidal angle is tangential in
real space, but its `m (m - 1)` mode filter gives it a normal part. A converged
free-boundary equilibrium balances it against the vacuum pressure, so its LCFS
is not a flux surface of the coils. VMEC++ gives the same fields to 1e-5.
Largest `|B|` error against Biot-Savart at `s = 0.25`, ESSOS LP-QA coils,
`ns = 51`, `ntor = 12` (`ns` 26-201, the NESTOR grid, `mf` and the mgrid
order and spacing move none of these by more than 10%):

| `mpol` | 6 | 7 | 8 | 9 | 10 | 12 | 14 |
|---|---|---|---|---|---|---|---|
| converged | 2.1e-4 | 1.3e-3 | 2.6e-4 | 6.7e-4 | 3.0e-4 | 2.5e-4 | 1.9e-4 |
| relaxed, 5,000 iterations | | 3.3e-5 | 2.6e-5 | | | | |

The relaxed state keeps a force residual near 1e-10, because nothing fixes
the angle any more, and reports `converged = False`.

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
