# Solve a free-boundary equilibrium with Newton

VMEX's default solver is steepest descent on the MHD energy `W`. It cannot
settle on an unstable equilibrium, which is a saddle of `W`. The opt-in
{func}`vmex.solve_free_boundary_newton` solves the same free-boundary force
balance by Newton's method and counts the ideal-MHD unstable directions at the
result. Descent stays the default; use Newton as a polish step.

## When to use it

- Descent drifts or stalls on a current-carrying free-boundary case.
- You need `|F|` far below the descent tolerance (Newton reaches about 1e-13).
- You want to know whether the equilibrium is ideal-MHD unstable.

## How

```python
import vmex

inp = vmex.VmecInput.from_file("input.case")        # LFREEB = T deck
field = vmex.MgridField.from_coils(coils, nfp=inp.nfp, ...)  # or read_mgrid
out = vmex.solve_free_boundary_newton(inp, field)

out.converged      # residual < 1e-9
out.residual       # |F| of the projected force residual
out.n_unstable     # negative eigenvalues of the force Jacobian
out.modes          # nev most negative: (eigenvalue, [(energy fraction, m, n), ...])
out.wout           # WoutData of the Newton-converged equilibrium
```

The force Jacobian equals `K d2W` with `K > 0` (VMEC's force metric), so it
has the inertia of the energy Hessian and each negative eigenvalue is a
direction with `delta W < 0`. Eigenvalue magnitudes are in VMEC's force
metric, not growth rates. A stable equilibrium gives `n_unstable == 0`.
Pass `count_modes=False` to skip the eigen-decomposition.

## Starting state

By default the solve starts from the free-boundary descent solve of the deck
(set the deck's `FTOL` to 1e-6 or tighter), anchored by a Krylov Newton step.

- `start="fixed"` starts from the fixed-boundary solve instead, the route for
  current-carrying cases where free-boundary descent drifts. It takes the
  constraint reference from that solve, so on the test fixture it lands on a
  different (also stable) root than the descent start.
- `start=` a `WoutData`, a wout path, or a `SpectralState` on the deck's radial
  grid reuses an existing solution.

Newton is fragile from a cold start or from a state converged only to 1e-4:
start from a state converged to about 1e-6 or tighter.

## Costs

Measured in a study on stable references and current-carrying cases:

- 2-10x fewer iterations than descent.
- 14-52 s of one-off compile time, so it does not pay on small decks.
- 1.5-3x the memory of descent (the dense reduced Jacobian and its JVPs).

The count was validated on stable reference cases (0 negative eigenvalues).
The dense fallback is meant for the moderate resolutions of a polish step.
