# Solve a free-boundary equilibrium with Newton

VMEX's default solver is steepest descent on the MHD energy `W`. The opt-in
{func}`vmex.solve_free_boundary_newton` puts a converged free-boundary state
on the exact root of the coupled plasma–vacuum force balance. It uses the
Krylov Newton anchor of the free-boundary adjoint: GMRES preconditioned by the
radial block-tridiagonal factors plus NESTOR's low-rank coupling, with steps
damped by Deuflhard's natural monotonicity test (NLEQ-ERR). Descent stays the
default; use Newton as a polish step.

## How

```python
import vmex

inp = vmex.VmecInput.from_file("input.case")        # LFREEB = T deck, FTOL <= 1e-6
field = vmex.MgridField.from_input(inp)
out = vmex.solve_free_boundary_newton(inp, field)

out.converged      # the anchor certified residual <= refine_tol (1e-10)
out.residual       # |F| of the projected raw force residual
out.wout           # WoutData; fsqr/fsqz/fsql are VMEC's normalized residuals
```

## Starting state

- `start=None` (default) runs the free-boundary descent of the deck and
  anchors it.
- `start=` a `WoutData`, a wout path or a `SpectralState` on the deck's
  radial grid anchors that state.

Every start solves the same residual. The free-boundary loop damps the
constraint baseline `rcon0/zcon0` by 0.9 per iteration, so it is zero at a
converged state, and a supplied start uses that zero baseline. On the test
tokamak a wout restart reproduces the descent root's volume to 5e-10.
The anchor needs a start converged to about 1e-6 or tighter.

## What it does not do

It does not judge stability. The inertia of VMEC's force Jacobian is not
the inertia of `d2W`: the spectral-condensation constraint, the m = 1
internal basis and the constraint baseline enter the force. On a ring
current in a vertical field with decay index 2.5, which is radially
unstable, the Jacobian had no negative eigenvalue.
