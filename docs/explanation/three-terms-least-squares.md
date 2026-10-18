# The three-terms free boundary by least squares

This page covers ``method="least_squares"`` of
{func}`~vmex.core.freeboundary_vc.solve_free_boundary_three_terms`
({mod}`vmex.core.freeboundary_lsq`; the default, ``method="galerkin"``, solves a
square system for the deck's normal displacement, and is faster): what it
minimizes, why the poloidal labelling must be fixed, how the
exact Hessian is built through the equilibrium, and how the design gradients
of a single-stage problem follow from it.

## What is minimized

The unknowns are the boundary Fourier coefficients $b$ (with $R_{00}$).
Every iterate is a converged fixed-boundary equilibrium on that boundary.
The rows $r(b)$ are, on an $n_\phi \times n_\theta$ grid over one field
period, each weighted by the square root of its share of the area:

- in vacuum, $B_\mathrm{coil}\cdot n / B_0$. $B_\mathrm{out}$ is the coil
  field exactly, so this is the field-line condition, and the interior field
  does not enter;
- at finite beta, the jump $(B_\mathrm{out} - B_\mathrm{in}) / B_0$ (three
  Cartesian components), with $B_\mathrm{out} = B_\mathrm{coil} +
  B_\mathrm{plasma}$ from virtual casing and $B_\mathrm{in}$ restricted to
  VMEC's own Galerkin band ($m \le$ `mpol` $-1$, $|n| \le$ `ntor`; outside it
  the edge field is not determined by the equilibrium). Since
  $B_\mathrm{in}\cdot n = 0$, the cost is the area mean of
  $(B_\mathrm{out}\cdot n)^2 + |n \times (B_\mathrm{out} - B_\mathrm{in})|^2$
  over $B_0^2$: the normal field and the sheet current in one norm, with no
  relative weights. With zero edge pressure the pressure balance
  $|B_\mathrm{out}| = |B_\mathrm{in}|$ follows from
  $B_\mathrm{out} = B_\mathrm{in}$, so it is not a row. All three terms are
  still reported.

$B_0$ is the area-RMS $|B|$ on the seed boundary. The net poloidal current
$G_\mathrm{plasma} = G_\mathrm{coil}$, a direction that $B\cdot n$ cannot
see, is an equality constraint (as are the bootstrap current's rows, below).
The solver minimizes $f(b) = \tfrac12 |r(b)|^2$ subject to them.

## Why the labelling must be fixed

A set of boundary coefficients describes a surface *and* a poloidal angle on
it. The relabelling $\theta \to \theta + \omega(\theta, \phi)$ leaves the
surface unchanged. With truncated series, though, it changes the coefficients
and the shapes they can represent, and with them the residual at the
truncation level.

The exact Hessian of $f$ makes this visible. On P30A coils at 8 × 8 modes:

| subspace | Hessian eigenvalues |
|---|---|
| relabellings | about $-10^{-8}$ to $10^{-9}$: flat and indefinite |
| shape (complement) | $2\cdot 10^{-7}$ to $5\cdot 10^{2}$ |

So a free labelling has no minimum near the start: the cost keeps falling as
the angle drifts. Anchoring the labelling to the start with weighted rows makes
every start solve a different problem. Run to full convergence from three
starts, that gives quasi-symmetry residuals of $4.5$, $3.4$ and
$3.8 \cdot 10^{-6}$, on surfaces 0.1 mm apart.

The solver therefore uses the Galerkin method's unknowns instead of the
coefficients: the displacement of the deck's boundary along its radial
coordinate lines, one coefficient per boundary mode. These displacements are
normal to the deck, so the deck fixes the labelling and no tangential
(relabelling) direction is left. The start (``initial_boundary``) is
converted to such a displacement, so every start reaches the same boundary.

A first version kept the coefficients and constrained the displacement from the
deck to have no component along the deck's relabellings. That removes the
relabellings with the boundary's own modes, but not the tangential
displacements that the truncated relabellings with higher modes produce. At
8 × 8 and β = 2.5 % the weakest reduced Gauss-Newton directions of that
formulation were 90-97 % tangential. Along them the cost grows as $t^4$
(the surface moves only at second order): ten times the quadratic model at
steps of 0.03, so trust-region steps failed and the solve took 45 models.

### Why not a reference-free labelling

A gauge defined by the boundary alone is possible: stationarity of the spectral
width $F_p = \sum m^p (R_{mn}^2 + Z_{mn}^2)$ under the $m > 0$ relabellings,
and of the twist $\sum (n N_{fp})^2 (R_{mn}^2 + Z_{mn}^2)$ under the shifts. It
is unique, but at low resolution its labelling costs the truncated series real
shape. On an optimized, strongly shaped 8-mode boundary, relabelling to the
spectral-width minimum and truncating moves the surface by 4-7 mm (maximum
normal distance, $p = 4$ or $6$, at 8 and 10 modes), against 0.05 mm for LP QA
at 8 modes with $p = 6$. For the
vacuum benchmark the field-line distance of the last closed surface at 12 modes
is 0.114 mm in that gauge, against 0.048 mm with the deck's labelling.

### Constraint rows: the net current and the bootstrap current

The net-current row and, with the bootstrap current in the solve, the current's
rows are equality constraints, not least-squares rows: any finite weight would
trade them against the interface conditions. The steps live in the null space
of their Jacobian. Each trial is restored onto them by chord Newton steps
(minimum-norm, scaled coordinates) to $\sqrt{\mathrm{FTOL}}$, the relative
precision of an equilibrium's fields. The Hessian is the Lagrangian's, with the
constraints' multipliers.

## The exact Hessian through the equilibrium

Let the equilibrium be $F(z, p) = 0$, with $z$ the evolved state and $p$ the
parameters (the boundary among them), and write $\Phi(z, p) = \tfrac12 |r|^2$.

- The multiplier solves $F_z^\top \lambda = \Phi_z$. That is one transposed
  solve with the block factorization the implicit derivatives already use.
- The Lagrangian is $L = \Phi - \lambda\cdot F$, and the reduced gradient is
  $L_p$.
- Along the state responses $F_z\,\delta z_j = -F_p\,\delta p_j$, the Hessian is

$$
H_{ij} = [\delta p_i;\ \delta z_i]^\top\, \nabla^2 L\, [\delta p_j;\ \delta z_j].
$$

Each column costs one forward-over-reverse pass of $L$. Unlike Gauss-Newton's
$J^\top J$, $H$ contains $\sum_k r_k \nabla^2 r_k$. That term is not small,
because the residual does not vanish at finite resolution.

Checked against central differences of the exact gradient, the error falls as
$\epsilon^2$. In vacuum it is $4\cdot10^{-6}$ to $4\cdot10^{-5}$ at
$\epsilon = 10^{-4}$. At $\beta = 2.5\,\%$ it is $10^{-1}$ at
$\epsilon = 10^{-3}$ and $10^{-3}$ at $10^{-4}$.

The trust-region step is solved exactly in the eigenbasis of the reduced,
ESS-scaled model, so indefinite models are handled too.  The exact Hessian
enters as products: a preconditioned block-Krylov subspace (preconditioner
the reduced $J^\top J$, Eisenstat-Walker forcing), used where a Gauss-Newton
model fails and to certify convergence.

- The Hessian is recomputed every few steps. In between, BFGS updates use exact
  adjoint gradients.
- When a step is not very successful by the radius update's own test, a
  second-order correction $-(JN)^{+} r(x_\mathrm{trial})$ follows it. At finite
  beta this takes a weak, strongly non-quadratic low-order shape mode from
  29 Hessians to 10.
- The solve stops when the Newton step's predicted gain, or the boundary
  displacement it implies, falls below its tolerance. A stall is reported as a
  stall.

## Design gradients

In single-stage optimization the boundary $b^*(x)$ minimizes $f(b, x)$ (on the current rows' manifold, when the current is in the
solve). A function $h$ of the equilibrium then has

$$
\frac{dh}{dx} = h_x - \mu^\top f_{bx}, \qquad
N^\top W N\, \mu_N = N^\top h_b, \quad \mu = N \mu_N,
$$

with $W$ the Lagrangian Hessian and $N$ a null-space basis of the
constraints' Jacobian. The current rows depend on the plasma parameters too;
their response off the null space adds one more term for those parameters.

- The coil part of $f_{bx}\,\mu$ is one forward-over-reverse pass through the
  field.
- The plasma-parameter part comes from Hessian columns along those parameters.

At $\beta = 2.5\,\%$ (6 × 6) the error against re-solved central differences is
below $2\cdot10^{-3}$ for $\iota$ and aspect ratio, and at most $2\,\%$ for the
quasi-symmetry residual. Gauss-Newton is off by up to 21 % on the same
directions.
