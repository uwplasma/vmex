# Differentiate an equilibrium

In this lesson you compute an exact derivative of a converged equilibrium —
`d(aspect ratio)/d(boundary coefficient)` — with `jax.grad`, and check it
against finite differences.

## The three lines that matter

```python
import jax
from vmex.core import implicit
from vmex.core.input import VmecInput

inp = VmecInput.from_file("examples/data/input.solovev")
p0 = implicit.params_from_input(inp)          # differentiable parameters
grad = jax.grad(lambda p: implicit.run(inp, p).aspect)(p0)
```

{func}`vmex.core.implicit.run` is the differentiable solve entry point: it
converges the equilibrium and returns a pytree whose scalar outputs
(`aspect`, `wb`, `volume`, ...) can be wrapped in any JAX transformation.
`grad` is a pytree matching `p0`: one derivative per boundary Fourier
coefficient, profile coefficient, `phiedge`, `pres_scale`, `curtor`.

## Check it

`examples/take_fixed_boundary_gradients.py` runs the check on the bundled Solovev deck
(ns=11, ftol=1e-12), comparing the adjoint gradient against a central finite
difference through two full re-solves:

```text
d(aspect)/d(RBC(0,1))  AD=-1.5182532271e+00  FD=-1.5182532279e+00  rel=5.58e-10
d(wb)/d(phiedge)       AD=+1.2910254037e-01  FD=+1.2910254037e-01  rel=1.64e-12
```

That is one run of the script on VMEX 0.11.0; the last digits of the finite
difference move between machines, and `tests/test_examples.py` fails the
example only above `1e-4`. The agreement is at the finite-difference noise
floor, and the adjoint side has no step size at all.

```{literalinclude} ../../examples/take_fixed_boundary_gradients.py
:language: python
```

## Why there is no step size to tune

The gradient does not difference two solves and does not backpropagate
through the iteration. It applies the implicit function theorem at the
converged fixed point: one linear (adjoint) solve per scalar output, O(1)
memory no matter how many iterations the forward solve took
({doc}`/explanation/adjoint-gradients`). Two consequences you get for free:

- cost is independent of the number of parameters — the gradient above has
  one entry per boundary coefficient at the price of one adjoint solve;
- there is no finite-difference truncation/roundoff trade-off, which is
  exactly what the agreement printed above shows.

One caveat worth knowing before you rely on FD checks yourself: for
solver-sensitive outputs (iota, mirror ratio, magnetic well) a naive
re-solving finite difference is not a valid reference — the frozen-path
check in {doc}`/explanation/adjoint-gradients` is.

## Model and references

At a converged state $x$ the force residual vanishes, $F(x, p) = 0$, so for any parameter $p$
(boundary or profile coefficient)

$$\frac{dx}{dp} = -\left(\frac{\partial F}{\partial x}\right)^{-1}\frac{\partial F}{\partial p}, \qquad
\frac{df}{dp} = \frac{\partial f}{\partial p} - \lambda^{\mathsf T}\frac{\partial F}{\partial p}, \quad
\left(\frac{\partial F}{\partial x}\right)^{\mathsf T}\lambda = \frac{\partial f}{\partial x}.$$

One adjoint solve for $\lambda$ gives the gradient of a scalar $f$ with respect to every parameter, and
memory does not grow with the number of forward iterations. VMEX uses this rather than differentiating
through the iteration because the iteration count varies and its history would have to be stored.
Assumptions and checks: {doc}`/explanation/adjoint-gradients`.

- T. M. Antonsen, E. J. Paul and M. Landreman, *J. Plasma Phys.* 85, 905850207 (2019): adjoint shape gradients for 3-D equilibria.
- S. G. Krantz and H. R. Parks, *The Implicit Function Theorem* (Birkhäuser, 2002).

Next: {doc}`first-optimization` puts the gradient to work.
