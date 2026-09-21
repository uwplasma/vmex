# Adjoint gradients and SOLVAX

VMEX differentiates converged fixed-boundary equilibria by applying the
implicit function theorem at the converged fixed point — one linear solve per
scalar objective, O(1) memory in the iteration count, no unrolling, no
finite-difference step size. All of that linear algebra runs on
[SOLVAX](https://pypi.org/project/solvax/), the linear/adjoint solver layer
factored out of this code base; this page states the formulation and the six
SOLVAX solve classes exactly as the code uses them.

## The implicit function theorem on the fixed point

Write the equilibrium as $F(z, p) = 0$, where $z$ contains independent evolved
coordinates and the assembled spectral state $x(z,p)$ includes the prescribed
boundary and frozen components. The parameters $p$ are
({class}`~vmex.core.implicit.ImplicitParams`: boundary coefficients,
profiles, `phiedge`, `pres_scale`, `curtor`). If $\partial F/\partial z$ is
invertible at the root, the solution map $p \mapsto z^\star(p)$ is
differentiable with

$$
\frac{dz^\star}{dp}
= -\left(\frac{\partial F}{\partial z}\right)^{-1}
  \frac{\partial F}{\partial p},
$$

For the assembled scalar objective $J(z,p)=\mathcal{J}(x(z,p),p)$, the
reverse-mode (adjoint) form needs **one** linear solve regardless of the
number of parameters:

$$
\left(\frac{\partial F}{\partial z}\right)^{\!\top} \lambda
= \frac{\partial J}{\partial z},
\qquad
\frac{dJ}{dp}
= \frac{\partial J}{\partial p}
- \lambda^{\!\top}\,\frac{\partial F}{\partial p}.
$$

{func}`~vmex.core.implicit.solve_implicit` wraps this in `jax.custom_vjp`.
The forward pass runs the host solver through `jax.pure_callback`. The
backward pass first uses the raw-force block transpose and checks its linear
residual; a failed host-eager certificate falls back to preconditioned Krylov.
A failed staged scalar adjoint returns non-finite sensitivities. VJPs of
{func}`~vmex.core.implicit.residual_fn` and the state assembly supply the
implicit and direct parameter terms. The derivative describes the selected
root family, not the host iteration history.

The matrix-free lane uses the **self-consistently 1D-preconditioned force** `gc` of a
single fresh {func}`~vmex.core.solver.evaluate_forces` pass:
$F = M f$ on the evolved subspace, with $f$ the raw spectral force.
For nonsingular $M$, raw and preconditioned implicit derivatives agree at an
exact root: $dF=M\,df+(dM)f=M\,df$. At an approximate root the $(dM)f$ term
remains; its effect depends on the residual and conditioning, not on `ftol`
alone. Preconditioning can reduce Krylov work but does not guarantee a
near-identity operator or a fixed iteration count.

Implicit differentiation avoids storing the forward iteration history. Its
memory still depends on resolution, linearization, factors and Krylov workspace;
runtime includes root refinement and the measured linear-solve work. Host
multigrid and restart decisions are outside the AD tape. They can nevertheless
select different roots or frozen components, so their effect must be checked
by independent reconvergence rather than assumed absent.

## Where the derivative is taken

The theorem holds at a *root* of $F$. The host solver instead stops on native
normalized squared-force measures. These do not directly bound the residual
norm used by every derivative formulation, state error or observable error.
Near a root, state error depends on $(\partial F/\partial z)^{-1}F$; small
singular values can amplify an apparently small residual. For example, a
solve reported converged at `ftol = 1e-12` on the non-stellarator-symmetric
`basic_non_stellsym_simsopt` deck returned with a residual of 2.7e-07; where
$\partial F/\partial z$ carries a small singular value (the lasym $m = 1$
families: 1.5e-04) that residual is a 1.8e-03 displacement of the state —
enough to move a solver-sensitive metric by more than the derivative being
measured.

VMEX therefore attempts Newton refinement inside the host callback before
derivative evaluation (`ImplicitConfig.refine_tol`, default
1e-10; `inf` disables it, and a refinement that fails to improve the residual
leaves the host state in place). When refinement succeeds, value, cotangent
and linearization use the refined root, as does a successfully reconverged
frozen-path finite-difference reference. Both have
to move together: on $d(\sum D_\mathrm{Merc})/d(\mathrm{RBS}(1,1))$ the
gradient agrees with that reference to rel 5.4e-07 when they do, against
4.2e-03 at the host stopping point and 5.7e-03 with the linearization alone
refined.

Nearby optimization trials reuse the previous Newton displacement as a guess.
VMEX evaluates the new frozen residual before accepting that guess; if it does
not reach `refine_tol`, VMEX discards it and replays the original refinement.
Public optimization factories expose the same `refine_tol`. Its default is
a refinement target, not a universal observable-accuracy certificate. Qualify
any change against nonlinear residuals and independently reconverged observable
derivatives; disabling refinement with `numpy.inf` is a legacy comparison.

## The six SOLVAX solve classes

Every linear solve in the gradient stack goes through SOLVAX. The complete
inventory, with the call site each class serves:

1. `solvax.gmres` / `solvax.gcrot` provide tangent solves, response corrections
   and the host-eager scalar-adjoint fallback in `vmex/core/implicit.py`.
   Tangents solve $(dF/dz)\, dz = -(dF/dp)\, t$ via the multi-RHS drivers
   {func}`~vmex.core.implicit.implicit_state_tangent_multi_rhs` /
   {func}`~vmex.core.implicit.implicit_state_pullback_multi_rhs`;
   every solve returns a {class}`~vmex.core.implicit.LinearResponseReport`
   (residual_norm, tolerance, iterations, converged).
2. `solvax.block_thomas_factor/solve` provide the default scalar reverse
   response and the amortized optimizer Jacobian. The raw-force radial blocks
   are assembled with 3-colored JVP probes and factored once. Transpose or
   per-parameter solves use these factors, with refinement/correction and
   certificates against the exact operator.
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

## Validating the gradients

The implicit derivative describes the root of the selected discrete residual
$F$. Its evolved-dof projector and converged m=1 force branch are fixed;
non-evolved state components remain at their anchor values. The residual's
preconditioner is evaluated from the current state and parameters. A small
linear-response defect certifies that linear equation, not the accuracy of
the nonlinear root or the observable.

Use two complementary checks:

- {func}`~vmex.core.implicit.frozen_path_directional_fd` Newton-solves the
  same frozen residual at perturbed parameters. Agreement checks the implicit
  linearization; `tests/test_implicit_grad.py` exercises this contract.
- Independently reconverge perturbed equilibria over several step sizes and
  measure the same observable. Require admitted roots, consistent physical
  constraints and reference data, and a finite-difference agreement window.
  Also check repeated parameters after unrelated trials and in a new problem
  instance. This tests the parameter dependence used by an optimizer.

Mirror ratio, iota and Boozer/QI diagnostics can amplify root, sampling or
branch-selection differences. On `li383_low_res`,
$d(\iota_{\mathrm{edge}})/d(\mathrm{RBC}(-1,1))$ is $-0.773$ from the adjoint
but $+0.045$ from a naive central difference through the full host solver. If independent differences disagree or change
sign, inspect those causes rather than dismissing the check or assuming the
adjoint is wrong. Compare raw/projected residuals, frozen state components and
independently refined diagnostics. If no agreement window is resolved, report
that observable's derivative as unqualified; frozen-path agreement alone does
not close the gap. See the [validation record](validation.md) and current
[research plan](https://github.com/uwplasma/vmex/blob/main/plan.md) for the measured scope.

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
(The preconditioned scalar fallback is dense in radius,
because the 1D preconditioner's inverse is.) The block path factors those
blocks once, back-solves every dof right-hand side, and checks each column
against the exact raw operator, applying correction when necessary.
No committed record measures its cost against
the per-column GMRES fallback (`jac_solver="gmres"`), so this page quotes no
speedup. The same per-dof responses $dz_j$ double as a first-order
perturbation warm start for the optimizer's next trial solves — the
DESC-style `eq.perturb` pattern, and the default
`warm_start="perturbation"`. This supplies an initial guess; verify that
restarts preserve the selected root family and final observables. Its saving
has no committed record either. How these
plug into an optimization campaign is {doc}`/howto/optimize-a-boundary`.

## Free-boundary root

The free-boundary path differentiates the converged coupled VMEX--NESTOR root,
including direct coil-shape/current parameters, while keeping host iterations
off the AD tape. The default whole-state transpose is exact but cold-compile
and memory limited. An advanced boundary-Schur transpose, its full-residual
certificate, and its current performance limits are documented in
{doc}`nestor-vacuum`. The supported scope of every AD path is one row of
{doc}`/reference/capabilities`.
