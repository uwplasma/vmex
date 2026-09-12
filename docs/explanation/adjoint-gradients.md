# Adjoint gradients and SOLVAX

VMEX differentiates converged fixed-boundary equilibria by applying the
implicit function theorem at the converged fixed point — one linear solve per
scalar objective, O(1) memory in the iteration count, no unrolling, no
finite-difference step size. All of that linear algebra runs on
[SOLVAX](https://pypi.org/project/solvax/), the linear/adjoint solver layer
factored out of this code base; this page states the formulation and the six
SOLVAX solve classes exactly as the code uses them.

## The implicit function theorem on the fixed point

The equilibrium is the root of the force residual $F(x, p) = 0$ with $x$ the
spectral state and $p$ the parameters
({class}`~vmex.core.implicit.ImplicitParams`: boundary coefficients,
profiles, `phiedge`, `pres_scale`, `curtor`). If $\partial F/\partial x$ is
invertible at the root, the solution map $p \mapsto x^\star(p)$ is
differentiable with

$$
\frac{dx^\star}{dp}
= -\left(\frac{\partial F}{\partial x}\right)^{-1}
  \frac{\partial F}{\partial p},
$$

and for a scalar objective $\mathcal{J}(x^\star(p))$ with cotangent
$\bar{g} = \partial\mathcal{J}/\partial x$, the reverse-mode (adjoint) form
needs **one** linear solve regardless of the number of parameters:

$$
\left(\frac{\partial F}{\partial x}\right)^{\!\top} \lambda = \bar{g},
\qquad
\frac{d\mathcal{J}}{dp}
= -\lambda^{\!\top}\,\frac{\partial F}{\partial p}.
$$

{func}`~vmex.core.implicit.solve_implicit` wraps this in `jax.custom_vjp`:
the forward pass runs the fast CLI-lane host solver (`jax.pure_callback` —
multigrid staging, restarts, and adaptive time-step control stay invisible to
autodiff; only the fixed point defines the derivative); the backward pass
solves the adjoint system matrix-free
({func}`~vmex.core.implicit.adjoint_matvec`): one `jax.vjp` linearization of
the residual function ({func}`~vmex.core.implicit.residual_fn`) is reused as
the transposed operator, and one more VJP contracts $\lambda$ against
$\partial F/\partial p$.

The residual is the **self-consistently 1D-preconditioned force** `gc` of a
single fresh {func}`~vmex.core.solver.evaluate_forces` pass:
$F = M(x,p)\,f(x,p)$ with $f$ the raw spectral force and $M$ the invertible
1D-preconditioner map. At the root $dF = M\,df + dM\,f = M\,df$ up to
$O(\mathrm{ftol})$, so the implicit gradients equal those of the raw force to
solver accuracy — while the adjoint Krylov solve inherits VMEC's own
preconditioning for free: near equilibrium $\partial F/\partial x$ is close
to the identity, so it converges in a handful of iterations.

Why the gradient is cheap: reverse-mode through an *unrolled* iteration would
store every iterate (memory linear in the iteration count) and backpropagate
through thousands of steps. The implicit adjoint touches only the converged
state: its cost is a fixed handful of residual evaluations (one linearization
plus the Krylov matvecs) and its memory is O(1) in the iteration count —
independent of how many Richardson steps, restarts, or multigrid stages the
forward solve needed. Multigrid stages act purely as an initializer and are
stop-gradient by construction.

## Where the derivative is taken

The theorem holds at a *root* of $F$, and the host solver does not stop at
one: `ftol` gates the sum of squares of the force, so a solve it reports
converged still returns with $|F| \sim \sqrt{\mathrm{ftol}}$ — 2.7e-07 at
`ftol = 1e-12` on the non-stellarator-symmetric `basic_non_stellsym_simsopt`
deck. Where $\partial F/\partial x$ carries a small singular value (the lasym
$m = 1$ families: 1.5e-04) that residual is a 1.8e-03 displacement of the
state — enough to move a solver-sensitive metric by more than the derivative
being measured.

VMEX therefore Newton-refines the state toward the root inside the host
callback, before any lane reads it (`ImplicitConfig.refine_tol`, default
1e-10; `inf` disables it, and a refinement that fails to improve the residual
leaves the host state in place). Value, cotangent and linearization then all
sit at the same point — which is also where the frozen-path finite-difference
reference measures, its own Newton endpoints being roots as well. Both have
to move together: on $d(\sum D_\mathrm{Merc})/d(\mathrm{RBS}(1,1))$ the
gradient agrees with that reference to rel 5.4e-07 when they do, against
4.2e-03 at the host stopping point and 5.7e-03 with the linearization alone
refined.

Nearby optimization trials reuse the previous Newton displacement as a guess.
VMEX evaluates the new frozen residual before accepting that guess; if it does
not reach `refine_tol`, VMEX discards it and replays the original refinement.
This changes work, not the accepted numerical path. Public optimization
factories expose the same `refine_tol`. Refinement is best-effort within a
bounded budget, so the achieved residual can exceed this tolerance. Disabling
it with `numpy.inf` requires separate derivative and repeatability validation.

## The six SOLVAX solve classes

Every linear solve in the gradient stack goes through SOLVAX. The complete
inventory, with the call site each class serves:

1. `solvax.gmres` / `solvax.gcrot` solve the implicit-function-theorem
   systems in `vmex/core/implicit.py`: adjoint $(dF/dz)^T \lambda = b$
   (warm-started GMRES; GCROT(m,k) with subspace recycling) and tangent
   $(dF/dz)\, dz = -(dF/dp)\, t$ via the multi-RHS drivers
   {func}`~vmex.core.implicit.implicit_state_tangent_multi_rhs` /
   {func}`~vmex.core.implicit.implicit_state_pullback_multi_rhs`;
   every solve returns a {class}`~vmex.core.implicit.LinearResponseReport`
   (residual_norm, tolerance, iterations, converged).
2. `solvax.block_thomas_factor/solve` power the amortized Jacobian path in
   `vmex/core/optimize.py`: the raw scalxc-scaled residual Jacobian is
   block-tridiagonal in radius; assembled with 3-colored jvp probes,
   factored once, backsolved per boundary dof, then one warm-started GMRES
   pass per column certifies `cfg.adjoint_tol`.
3. `solvax.tridiagonal_solve(_checked)` performs the per-mode radial 1D
   preconditioner solves (`vmex/core/preconditioner.py`; the `precondn.f` /
   `scalfor.f` analogue — see {doc}`preconditioners`).
4. `solvax.gmres` solves the matrix-free 2D block-preconditioner Newton
   direction (`vmex/core/preconditioner_2d.py`; matvec
   `v -> jvp(g, state, v)`; the `precon2d.f` analogue).
5. `solvax.chunk_map` / `auto_chunk_size` bound memory for Jacobian columns
   and multi-RHS batches (`vmex/core/optimize.py`).
6. `solvax.SpluFactorization` owns the pivoted sparse factorization used to
   eliminate the radial bulk in the experimental free-boundary Schur
   transpose (`vmex/core/freeboundary_implicit.py`). VMEX supplies the
   physics-specific radial blocks and NESTOR edge response; SOLVAX owns the
   reusable factorization and transposed solves.

The mirror lane keeps its own adjoint solver (`vmex/mirror/implicit.py`).

## Certificates

Each tangent or adjoint solve reports its residual norm, tolerance, iteration
count, and convergence. VMEX then checks every block-response column against
the exact linearized operator.

- ``auto`` retries a failed block response with the reverse adjoint.
- An explicitly selected host solver raises
  {class}`~vmex.core.errors.AdjointSolveError` when the check fails.
- The JAX API selects the reverse branch inside the compiled graph, where a
  Python exception is unavailable.

## Validating the gradients: the frozen path

The adjoint differentiates the root with constrained coordinates and the dof
projector frozen at the base state. The preconditioner and `tcon` remain
state-dependent. {func}`~vmex.core.implicit.frozen_path_directional_fd`
Newton-solves those same equations at perturbed parameters and tests their
implicit derivative; `tests/test_implicit_grad.py` uses this contract.

Independent nonlinear re-solves test a second requirement: whether the
objective supplied to an optimizer follows that differentiable root. They
can disagree for iota, mirror, magnetic well, and Boozer/QI metrics because
finite force residuals and restart-dependent constrained coordinates affect
the returned state. A frozen-path pass does not dismiss that disagreement.
For example, the recorded `li383_low_res` edge-iota derivative is -0.773
from the adjoint and +0.045 from an independent central difference.

Validate optimization with identical-parameter return and cold-replay checks,
then re-solve Taylor tests across decreasing steps with achieved nonlinear
residuals below the measured error. Check final force balance and resolution
independently. The collaborator replay in `benchmarks/optimization.py` and
`benchmarks/review_optimization_20260912.json` records the unresolved QI
history dependence; neither a linear-solve certificate nor requesting
`refine_tol=1e-10` establishes repeatability.

## Forward mode for least-squares Jacobians

The adjoint is the right tool for one scalar objective and many parameters.
Vector residuals need the full Jacobian over all boundary dofs, which
`jac_solver="auto"` computes in **forward** mode: per dof tangent $t_j$, the
state response is

$$
dz_j = -\left(\frac{\partial F}{\partial z}\right)^{-1}
       \frac{\partial F}{\partial p}\, t_j,
$$

one linear solve per dof. Rather than an independent GMRES per column, the
vector-residual path (`jac_solver="block"`, SOLVAX class 2 above) exploits a
structural fact: in the **raw** force formulation the radial coupling of
$\partial F/\partial z$ is exactly nearest-neighbor, so the operator is
*exactly* block-tridiagonal — `ns` dense $(3\,mn \times 3\,mn)$ blocks.
(The preconditioned formulation used by the adjoint is dense in radius,
because the 1D preconditioner's inverse is.) Measured: the warm Jacobian
phase of the benchmark optimization step drops from 20.35 s to 0.61 s (33x;
`docs/_static/figures/gradient_stack_speedup.webp`, reproduced by
`docs/_static/figures/sources/make_optimization_docs_figures.py`). The same
per-dof responses $dz_j$ double as a first-order perturbation warm start for
the optimizer's next trial solves — the DESC-style `eq.perturb` pattern —
measured 3.7x fewer total forward iterations over 20 trials (23,685 to
6,364). How these plug into an optimization campaign is
{doc}`/howto/optimize-a-boundary`.

```{figure} /_static/figures/gradient_stack_speedup.webp
:alt: measured before/after of the three gradient-stack optimizations
:width: 100%

Measured gradient-stack speedups on the nfp2 minimal-seed deck (Jacobian
phase and trial iterations) and the full QA campaign (right); 2026-07-12,
CPU. Regenerate with
`python docs/_static/figures/sources/make_optimization_docs_figures.py`.
```

## Free-boundary root

The free-boundary path differentiates the converged coupled VMEX--NESTOR root,
including direct coil-shape/current parameters, while keeping host iterations
off the AD tape. The default whole-state transpose is exact but cold-compile
and memory limited. An advanced boundary-Schur transpose, its full-residual
certificate, and its current performance limits are documented in
{doc}`nestor-vacuum`. The supported scope of every AD path is one row of
{doc}`/reference/capabilities`.
