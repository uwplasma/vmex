# Solve a free-boundary field that matches the coils

A converged free-boundary equilibrium is not exactly in the coil field.
VMEC fixes the poloidal angle with a spectral-condensation force. That force
is tangential in real space, but its `m (m - 1)` mode filter gives it a
normal part, so the converged LCFS balances it as well as the vacuum
pressure and is not a flux surface of the coils. VMEC2000 and VMEC++ share
this; VMEC++ reproduces the VMEX errors below to 1e-5.
{func}`vmex.relax_free_boundary_constraint` continues the converged state
with the constraint off (`tcon0 = 0`) and removes the bias.

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
relaxed = vmex.relax_free_boundary_constraint(inp, result, external_field=field,
                                              iterations=5000)
wout = vmex.wout_from_result(inp, relaxed)
```

`python examples/free_boundary_relax_constraint.py` runs this on the ESSOS
LP-QA coils and prints the `|B|` error against Biot-Savart before and after.

## Judging the result

Compare `|B|` with the coil field on flux surfaces; do not judge by `ftol`.
Nothing fixes the poloidal angle without the constraint, so the force
residual stays near 1e-10 and the result reports `converged = False`.

Largest relative `|B|` error against Biot-Savart, ESSOS LP-QA coils,
`ns = 51`, `ntor = 12`. `ns` 26-201, the NESTOR grid, `mf` and the mgrid
order and spacing move the converged values by less than 10%:

| `mpol` | `s = 0.25` converged | relaxed | `s = 0.9` converged | relaxed |
|---|---|---|---|---|
| 6 | 2.1e-4 | | 1.2e-3 | |
| 7 | 1.3e-3 | 3.3e-5 | 4.0e-3 | 1.4e-4 |
| 8 | 2.6e-4 | 2.6e-5 | 1.7e-3 | 1.9e-4 |
| 9 | 6.7e-4 | | 1.7e-3 | |
| 12 | 2.5e-4 | | 1.3e-3 | |
| 14 | 1.9e-4 | | 7.2e-4 | |

The converged error changes erratically with `mpol` and does not converge;
after relaxation it does not depend on the parity of `mpol`.
