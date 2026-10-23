# Solve a free-boundary field that matches the coils

A converged free-boundary equilibrium is not exactly in the coil field.
VMEC fixes the poloidal angle with a spectral-condensation force. That force
is tangential in real space, but its `m (m - 1)` mode filter gives it a
normal part, so the converged LCFS balances it as well as the vacuum
pressure and is not a flux surface of the coils. VMEC2000 and VMEC++ share
this; VMEC++ reproduces the VMEX errors below to 1e-5.
{func}`vmex.relax_free_boundary_constraint` continues the converged state
once with a weak constraint (`tcon0 = 1e-3`, 2,000 iterations) and removes
most of the bias. The default solve is unchanged and keeps VMEC2000 parity.

## When to use it

Use it on a free-boundary equilibrium whose field is compared with the
coils or traced: alpha losses, field lines, Boozer spectra of a vacuum
configuration. It only helps when `mpol` and `ntor` resolve the LCFS and the
coil ripple (see Limits). The eight Landreman-Paul QA coils per period put
`|B|` harmonics of 1e-4 at `n = 9-11` per period inside `s = 0.25`, so
`ntor <= 8` leaves a 5e-4 error even on a fixed boundary traced from the
coils, and `ntor = 12` brings it to 8e-5.

## How

```python
import vmex

field = vmex.MgridField.from_coils(coils, order=3)          # or MgridField.from_input(inp)
result = vmex.solve_free_boundary(inp, external_field=field)
relaxed = vmex.relax_free_boundary_constraint(inp, result, external_field=field)
wout = vmex.wout_from_result(inp, relaxed)
```

`python examples/free_boundary_relax_constraint.py` runs this on the ESSOS
LP-QA coils and prints the `|B|` error against Biot-Savart before and after.

## Judging the result

Compare `|B|` with the coil field on flux surfaces; do not judge by `ftol`.
With a weak constraint the force residual stays at 1e-10 to 1e-7 and the result
reports `converged = False`. Call the relaxation once: restarting it again and
again with `tcon0 = 0` let `fsq` grow to 1e-4 and the error to 7e-3.

The continuation has a slowly growing mode. On NCSX `fsq` falls to 8e-11 at
about 600 iterations, then doubles every 100: at 2,000 iterations it is 2e-7
and the error is 5.5e-4 at `s = 0.25`, worse than the 2.0e-4 before the
relaxation, while stops at 500, 1,000, 3,000, 5,000 and 10,000 iterations all
give 7e-5. The function therefore returns the lowest-`fsq` iterate: if the
final `fsq` is more than a decade above the minimum it replays the
(deterministic) continuation to that iteration, 592 extra iterations on NCSX,
and `relaxed.iterations` is the iteration it stopped at. `converged = False` can
hide such a spike, so check `relaxed.iterations` and `relaxed.fsqr`.

## Limits

- **Resolve first.** W7-X (50 coils, `R0 = 6.18 m` surface, `ftol = 1e-11`)
  needs `mpol = 12`, `ntor = 16`. The largest `|B|` error at `s = 0.25` after
  relaxing, with the converged value in brackets: `mpol = ntor = 8` 3.3e-3
  (4.1e-3), 10 / 10 8.4e-4 (1.3e-3), 10 / 12 3.1e-4 (1.0e-3), 12 / 12 3.2e-4
  (1.0e-3), 10 / 16 9.9e-5 (7.8e-4), 12 / 16 6.2e-5 (6.0e-4). The gain is
  1.2x, 1.5x, 3x, 3x, 8x and 10x; it is only 1.2-3x at `ntor <= 12`.
- **Low resolution can get worse.** CTH (LASYM, real mgrid) at `mpol = 5`,
  `ntor = 4` goes from `B.n/|B|` 2.3e-3 to 6.5e-3 over 2,000 iterations at
  `tcon0 = 1e-3`, with the LCFS 9.6 mm away; `tcon0 = 0` drifts the poloidal
  angle by 41 mm. At `mpol = ntor = 8` it is 1.4e-3 to 3.0e-3. The lowest-`fsq` iterate
  (351 and 52 iterations) is back at the converged state, no better, so a
  minimum in the first few dozen iterations means the relaxation did nothing.
  `tcon0 = 1e-2` is neutral on both.
- **The default `tcon0 = 1` solve can stall at `mpol >= 8`** on finite beta
  (matched LP-QA 0.5%, `mpol = 8`, `ntor = 12`: 8,000 iterations, `fsq` 3.0e-5,
  `B.n/|B|` 5.4e-3). Starting the solve at `tcon0 = 0.3` converges it in 2,679
  iterations (`B.n/|B|` 4.1e-4). Relaxing the stalled state does reach 2.8e-10
  but moves the LCFS by 12.8 mm, so fix the solve instead.
- **`tcon0 = 1e-2` is the conservative choice.** It reaches `ftol` on NCSX
  `mpol = 8` and LP-QA `mpol = 7`, holds `fsq` near 1e-10 on NCSX `mpol = 10-12`,
  never drifts and is neutral at low resolution, but
  improves the field 1.5-4x less than `1e-3` once the resolution is adequate.
- A tabulated field needs about 16 planes per coil spacing: a W7-X or NCSX
  table with 32-40 planes per period had 1e-3 to 3e-3 errors. Evaluate direct
  Biot-Savart with at least 8 points per coil segment; the simsopt default
  `ppp = 2` for W7-X moves `|B|` on the axis by 3%.

## Study

Largest relative `|B|` error against Biot-Savart at `s = 0.25` (at `s = 0.9`
in brackets), LCFS rms `B.n/|B|`, and the extra wall time on 2-3 CPU threads of a loaded machine.
`ftol = 1e-11`; before relaxing, `ns` 26-201, the NESTOR grid, `mf` and the
mgrid order and spacing move the error by less than 10%. The relaxed columns
are a 2,000-iteration continuation; on NCSX they are the lowest-`fsq`
iterate (the plain 2,000-iteration endpoint is the spike above).

| case | converged | `tcon0 = 1e-3` | `tcon0 = 0`, 5,000 it. | `tcon0 = 1e-2` |
|---|---|---|---|---|
| LP-QA, `mpol = 7`, `ntor = 12`, `ns = 31` | 1.4e-3 (3.9e-3), B.n 5.8e-4, 72 s | 8.8e-5 (3.2e-4), 3.9e-4, +36 s | 9.7e-5 (1.5e-4), 1.3e-4, +94 s | 3.2e-4 (8.7e-4), ftol in 794 it., +48 s |
| LP-QA, `mpol = 8`, `ntor = 12`, `ns = 51` | 2.7e-4 (1.6e-3), 3.1e-4, 59 s | 4.0e-5 (3.1e-4), 1.9e-4, +77 s | | 1.2e-4 (6.1e-4) |
| HSX coils, `mpol = 10`, `ntor = 16` | 9.7e-4 (3.7e-3), 1.2e-3, 92 s | 6.4e-4 (1.2e-3), 7.6e-4, +91 s | | 5.9e-4 (1.5e-3) |
| W7-X, `mpol = 12`, `ntor = 16`, `ns = 31` | 6.0e-4 (9.9e-4), 3.0e-4, 258 s | 6.2e-5 (1.8e-4), 2.1e-4, +270 s | 6.3e-5 (2.0e-4), 1.8e-4, +689 s | 1.4e-4 (3.8e-4), +289 s |
| W7-X, `mpol = 10`, `ntor = 12`, `ns = 31` | 1.0e-3 (2.6e-3), 6.3e-4, 104 s | 3.1e-4 (5.7e-4), 3.8e-4, +118 s | 3.0e-4 (6.1e-4), +341 s | 5.0e-4 (7.8e-4), +117 s |
| NCSX, `mpol = 10`, `ntor = 12`, `ns = 31` | 2.0e-4 (1.4e-3), 2.0e-4, 74 s | 6.8e-5 (1.6e-4), 1.4e-4, +108 s | 4.4e-5 (1.5e-4), +308 s | 8.2e-5 (3.8e-4), 1.9e-4, +140 s |
| NCSX, `mpol = 10`, `ntor = 12`, `ns = 51` | 1.7e-4 (1.3e-3), 2.1e-4, 115 s | 6.1e-5 (1.9e-4), 1.2e-4, +245 s | | 5.5e-5 (3.4e-4) |

Finite beta has no coil reference for the interior field, so the table gives
the LCFS `B.n/|B|` (coils plus virtual-casing plasma field), the LCFS move
between the converged and relaxed states, and the relative `|B|` change from
the converged state at `s = 0.25` / `0.9`. Zero toroidal current for W7-X and NCSX (`p = p0 (1 - s)`).

| case | converged | `tcon0 = 1e-3` | `tcon0 = 1e-2` |
|---|---|---|---|
| LP-QA 0.5%, `mpol = 7`, `ntor = 12` | 3.8e-4, 2,655 it. | 3.9e-4, LCFS 1.8 mm, `\|B\|` 0.8e-3 / 2.5e-3 | 3.9e-4, 1.6 mm, 0.7e-3 / 2.3e-3 |
| LP-QA 0.5%, `mpol = 7`, `ntor = 6` | 9.1e-4, 1,867 it. | 9.0e-4, 2.3 mm, 1.2e-3 / 3.7e-3 | 9.0e-4, 2.0 mm, 0.9e-3 / 3.4e-3 |
| LP-QA 0.5%, `mpol = 8`, `ntor = 12` | stalls: 5.4e-3, `fsq` 3e-5 | 5.2e-4, 12.8 mm (from the stalled state) | not converged |
| CTH LASYM, `mpol = 5`, `ntor = 4`, `ns = 15` | 2.3e-3, 513 it. | 6.5e-3, 9.6 mm; lowest `fsq` 2.2e-3 | 2.2e-3, 1.8 mm, 0.8e-3 / 2.4e-3 |
| CTH LASYM, `mpol = 8`, `ntor = 8`, `ns = 31` | 1.4e-3, `fsq` 9e-10 after 6,000 it. | 3.0e-3, 4.3 mm; lowest `fsq` (52 it.) 1.4e-3 | 1.4e-3, 0.1 mm |
| W7-X 0.95% / 2.9%, `mpol = 10`, `ntor = 12` | converged | LCFS 2.9 / 5.6 mm | 2.6 / 2.2 mm |
| NCSX 1.0%, `mpol = 10`, `ntor = 12` | converged | LCFS 3.1 mm | 2.3 mm |

- The bias does not depend on pressure: relaxing W7-X and NCSX at 1-3% beta
  moves the LCFS by 2.5-5.6 mm, the size of the vacuum shifts (2.8 and 3.2 mm),
  changes beta by less than 6e-7 and the volume by less than 2.4e-5.
- The bundled `input.cth_like_free_bdy_lasym_small` fixture uses a synthetic
  9 x 11 x 20 table that is not a vacuum field (`R B_phi` rises from -0.33 to
  -1.0 T m); use a real mgrid for any accuracy study. The LASYM deck converges
  with it in 513 iterations. The LP-QA 2.5% bootstrap deck is not an
  equilibrium of the bundled coils (axis displaced by 17-22 cm).
- Lowering `tcon0` from the start barely helps: LP-QA `mpol = 7` gives 1.3e-3,
  1.2e-3 and 9.7e-4 at 0.5, 0.3 and 0.1. Stopping the first solve at
  `ftol = 1e-8` and relaxing to the same `ftol` gives 2.2e-4.
- HSX is limited by resolution, not by the constraint: twelve coils per
  period need more than `mpol = 10`, `ntor = 16`.
- At 1% beta (LP-QA coils) the relaxation moves the LCFS by up to 1.6 mm and
  leaves beta unchanged to 1e-6. In vacuum the relaxed edge iota, 0.41557,
  matches the field-line value of the coils, 0.41582, against 0.41458
  converged. W7-X `mpol = 12`, `ntor = 16` goes from -1.1e-3 to -0.5e-3 in
  edge iota against field lines; NCSX gains nothing there.
- Peak memory grows by 0.3-1 GB (1.0 to 1.5 GB on W7-X and NCSX, 1.3 to
  1.5-1.8 GB on LP-QA, 4.1 to 5.1 GB on HSX).
- Alpha losses, 1,000 births at `s = 0.25`, 10 ms, ARIES-CS scale: `mpol = 7`
  6.7% converged and 0.4% relaxed; `mpol = 8` 1.1% and 0.7%; the coil field
  0.7-0.9% ({doc}`trace-resolution`). The error is about 0.3%.

The default stays the VMEC2000 solve. The relaxation costs 50-130% more
time, does not reach `ftol`, and the free-boundary adjoint differentiates
the converged VMEC root, so it is opt-in. Fixed-boundary solves are not
affected.
