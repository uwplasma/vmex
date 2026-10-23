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

Use it on any free-boundary equilibrium whose field is compared with the
coils or traced: alpha losses, field lines, Boozer spectra of a vacuum
configuration. The relaxation only helps once `ntor` resolves the coil
ripple. The eight Landreman-Paul QA coils per period put `|B|` harmonics of
1e-4 at `n = 9-11` per period inside `s = 0.25`, so `ntor <= 8` leaves a 5e-4
error even on a fixed boundary traced from the coils, and `ntor = 12` brings
it to 8e-5.

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
With a weak constraint the force residual stays at 1e-9 to 1e-7 and the result
reports `converged = False`. Call the relaxation once: restarting it again and
again with `tcon0 = 0` let `fsq` grow to 1e-4 and the error to 7e-3.

## Study

Largest relative `|B|` error against Biot-Savart at `s = 0.25` (at `s = 0.9`
in brackets), LCFS rms `B.n/|B|`, and the extra wall time on 3 CPU threads.
`ntor = 12`, `ftol = 1e-11`; before relaxing, `ns` 26-201, the NESTOR grid,
`mf` and the mgrid order and spacing move the error by less than 10%.

| case | converged | `tcon0 = 1e-3`, 2,000 it. | `tcon0 = 0`, 5,000 it. | `tcon0 = 1e-2` |
|---|---|---|---|---|
| LP-QA, `mpol = 7`, `ns = 31` | 1.4e-3 (3.9e-3), B.n 5.8e-4, 72 s | 8.8e-5 (3.2e-4), 3.9e-4, +36 s | 9.7e-5 (1.5e-4), 1.3e-4, +94 s | 3.2e-4 (8.7e-4), ftol in 794 it., +48 s |
| LP-QA, `mpol = 8`, `ns = 51` | 2.7e-4 (1.6e-3), 3.1e-4, 59 s | 4.0e-5 (3.1e-4), 1.9e-4, +77 s | | 1.2e-4 (6.1e-4) |
| HSX coils, `mpol = 10`, `ntor = 16` | 9.7e-4 (3.7e-3), 1.2e-3, 92 s | 6.4e-4 (1.2e-3), 7.6e-4, +91 s | | 5.9e-4 (1.5e-3) |

- Lowering `tcon0` from the start barely helps: LP-QA `mpol = 7` gives 1.3e-3,
  1.2e-3 and 9.7e-4 at 0.5, 0.3 and 0.1. Stopping the first solve at
  `ftol = 1e-8` and relaxing to the same `ftol` gives 2.2e-4.
- HSX is limited by resolution, not by the constraint: twelve coils per
  period need more than `mpol = 10`, `ntor = 16`.
- At 1% beta (LP-QA coils) the relaxation moves the LCFS by up to 1.6 mm and
  leaves beta unchanged to 1e-6. In vacuum the relaxed edge iota, 0.41557,
  matches the field-line value of the coils, 0.41582, against 0.41458
  converged. The LCFS spectral truncation moves from 2.7e-3 to 3.2e-3.
- Peak memory grows by 0.3-1 GB (1.3 to 1.5-1.8 GB on LP-QA, 4.1 to 5.1 GB on HSX).
- Alpha losses, 1,000 births at `s = 0.25`, 10 ms, ARIES-CS scale: `mpol = 7`
  6.7% converged and 0.4% relaxed; `mpol = 8` 1.1% and 0.7%; the coil field
  0.7-0.9% ({doc}`trace-resolution`). The error is about 0.3%.

The default stays the VMEC2000 solve. The relaxation costs 50-130% more
time, does not reach `ftol`, and the free-boundary adjoint differentiates
the converged VMEC root, so it is opt-in. Fixed-boundary solves are not
affected.
