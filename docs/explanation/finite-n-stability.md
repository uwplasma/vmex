# Finite-n stability with AGNI

This page covers the optional coupling to [AGNI](https://github.com/rahulgaur104/AGNI)
(`agnimhd`), which computes the global ideal-MHD stability of a fixed-boundary equilibrium at
finite toroidal mode number. It states what the coupling adds to the infinite-n ballooning
screen, the data VMEX hands to AGNI, how to choose the discretization, how the growth rate is
differentiated through the equilibrium, and what has been measured. The adapter is
{func}`vmex.core.agni.make_agni_map`; the examples are `examples/finite_n_stability.py` and
`examples/optimization/QI_optimization_finite_n.py`.

Install with `pip install "vmex[agni]"`. The extra currently needs an AGNI that has
`Basis(restrict=True)`, from [AGNI#117](https://github.com/rahulgaur104/AGNI/pull/117); the
latest tag, v0.2.0, does not have it, so `pyproject.toml` declares no floor.

## What AGNI adds

The ballooning objective ({doc}`confinement`) is an infinite-n eigenproblem along one field line.
AGNI solves the energy principle of Bernstein et al. on the whole plasma at finite n, with the
poloidal and toroidal harmonics of a family coupled and the radial structure resolved.

| | VMEX ballooning | AGNI through VMEX |
|---|---|---|
| mode number | $n \to \infty$ | finite, Fourier bandwidth $(M, N)$ |
| domain | one field line per surface | annulus $\varepsilon \le \rho \le 1$, all surfaces coupled |
| boundary | $X(\pm\eta_b)=0$ | $\boldsymbol\xi\cdot\nabla\rho = 0$ at $\rho=\varepsilon$ and $\rho=1$ |
| result | largest $\lambda$ per line | largest $\gamma^2$ per toroidal family |

For a static equilibrium $\mathbf J\times\mathbf B=\nabla p$ and a displacement $\boldsymbol\xi$,
the potential energy is a Hermitian form. With $r=\rho$, adiabatic index $\Gamma$ and
$\mathbf Q=\nabla\times(\boldsymbol\xi\times\mathbf B)$, AGNI's compact form (its eqs. 16-17, written
here for $2\,\delta W$) is

$$
2\,\delta W = \operatorname{Re}\!\int\!\Big[\frac{|\mathbf C|^2}{\mu_0}
+ \Gamma p\,|\nabla\!\cdot\boldsymbol\xi|^2 - \mathscr D_r\,|\boldsymbol\xi\cdot\nabla r|^2\Big]\,dV,
\qquad
\mathbf C = \mathbf Q + \mu_0\,\frac{\mathbf J\times\nabla r}{|\nabla r|^2}\,(\boldsymbol\xi\cdot\nabla r),
$$

$$
\mathscr D_r = \frac{2\,(\mathbf J\times\nabla r)\cdot[(\mathbf B\cdot\nabla)\nabla r]}{|\nabla r|^4}.
$$

The adapter supplies $\mathbf J\times\nabla r$ and $(\mathbf B\cdot\nabla)\nabla r$; AGNI forms
$\mathscr D_r$. Discretized with a mass form $M=\int\rho_m\,\mathbf e_a^*\cdot\mathbf e_b\,dV$,
the problem is $Ku=\omega^2Mu$. AGNI whitens it to a Hermitian $H$ and returns

$$
\gamma^2 = -\lambda_{\min}(H),\qquad \lambda_{\min}(H)=\min_v \frac{v^\dagger H v}{v^\dagger v},
$$

so **positive $\gamma^2$ means unstable**. The interface calls this `gamma2`.

## Normalization and physical growth rate

AGNI normalizes lengths by the effective minor radius $a$ and fields by
$B_N=|\Psi|/(\pi a^2)$. The adapter defines $a=\sqrt{\langle A\rangle_\phi/\pi}$, with $A(\phi)$
the area of the full enclosed cross section ($0\le\rho\le1$) at fixed $\phi$, so changing the
inner cutoff $\varepsilon$ does not change $a$. It is not the wout `Aminor`.

The normalized density is $d=\rho_m/\rho_{\rm ref}$. `make_agni_map(..., density=...)` takes
one value per node (or a callable of $\rho$); without it $d=1$. A physical growth rate needs
a declared reference mass density $\rho_{\rm ref}$ (kg m$^{-3}$):

$$
\omega_A^2 = \frac{B_N^2}{\mu_0\rho_{\rm ref}a^2} = \frac{\Psi^2}{\pi^2\mu_0\rho_{\rm ref}a^6},
\qquad
\gamma_{\rm phys} = \omega_A\sqrt{\gamma^2}\ \ [\mathrm{s}^{-1}].
$$

$\gamma_{\rm phys}$ is an exponential rate: do not divide it by $2\pi$. Optimize $\gamma^2$,
not $\gamma_{\rm phys}$: the square root has a singular derivative at marginality, and the
normalized value also changes with $\Psi$ and $a$. Both examples use $d=1$ and report $\gamma^2$.

## The data contract

The adapter is a pure, traceable map from a converged `(state, runtime)` pair to an
`agnimhd.EquilibriumData` on a fixed node set. It solves no equilibrium and reads no file. It is
differentiable in the state and in the live runtime arrays (boundary, flux, pressure).

| quantity | convention |
|---|---|
| coordinates | PEST $(\rho, v, \phi)$, $\rho=\sqrt s$; $\phi$ is the physical toroidal angle on one period $[0,2\pi/N_{\rm fp})$, not $N_{\rm fp}\phi$ |
| orientation | $v=\sigma(u+\lambda)$ with $\sigma=\mathrm{signgs}=-1$, so that $\sqrt g>0$; `validate()` rejects a sign-changing or negative $\sqrt g$ (AGNI#114) |
| flux | $\Psi=\sigma\,2\pi\,\psi_{\rm edge}$ = `PHIEDGE`, the signed total toroidal flux in Wb (not multiplied by $N_{\rm fp}$) |
| reduced flux | $\psi=\Psi\rho^2/2\pi$, so $\psi_r=\rho\Psi/\pi$ and $\psi_{rr}=\Psi/\pi$ |
| rotational transform | $\iota=\sigma\,\chi'(s)/\phi'(s)$, the current-consistent value of the solved equilibrium |
| pressure | $p$ and $p_r=dp/d\rho$ in Pa |
| field | $\mathbf B=\dfrac{\psi_r}{\sqrt g}\,(\iota\,\mathbf e_v+\mathbf e_\phi)$, $\mathbf e_a=\partial\mathbf X/\partial a$ at fixed PEST coordinates |
| current | $\mathbf J=\nabla\times\mathbf B/\mu_0$ from the strong-force kernel of the high-order lift, in A m$^{-2}$; $J^\phi=\mathbf J\cdot\nabla\phi$ and $|\mathbf J|$ are supplied |
| minor radius, density | $a$ as above, in m; dimensionless $d$ |

The poloidal component $J^v$ is not supplied: AGNI infers it from force balance,
$J^v=\iota J^\phi+p_r/\psi_r$. The identity holds only as well as the equilibrium is
force balanced, so it is a gate (see the validation table).

The PEST angle is found at each node from $u+\lambda(\rho,u,N_{\rm fp}\phi)=\sigma v$ by a
vectorized Newton iteration whose last step starts from a stop-gradient iterate. Reverse mode
therefore sees the implicit derivative, $\partial u/\partial c|_{\rho,v,\phi}=-\partial_c\lambda/(1+\lambda_u)$.
All derivatives of $\sqrt g$ are explicit contractions of the spline and Fourier tables
(no nested autodiff), which compiles a value-and-gradient in 3-5 s.

The adapter raises `NotImplementedError` for `lasym`, `lflip` and any `APHI` other than the
identity map. AGNI's `incompressible=True` is not available. Use `parity=None` in
`AssemblyConfig`; the parity reduction has not been validated for the families that need a
Bloch phase.

## Toroidal families

With $N_{\rm fp}$ field periods the equilibrium couples toroidal harmonics $n=q+kN_{\rm fp}$,
$k\in\mathbb Z$. A single integer $n$ is not a closed eigenproblem; the family $q$ is. AGNI uses
$\exp[i(mv+n\phi)]$, so a family obeys
$\boldsymbol\xi(\phi+2\pi/N_{\rm fp})=e^{2\pi iq/N_{\rm fp}}\boldsymbol\xi(\phi)$ and a resonance is
$m\iota+n=0$. Wout and VMEC labels usually use $\exp[i(m\theta-n\zeta)]$, which flips the sign of $n$.

Families $q$ and $N_{\rm fp}-q$ have equal eigenvalues, so `basis.families(NFP)` lists
$q=0,\dots,\lfloor N_{\rm fp}/2\rfloor$ (0, 1, 2 for $N_{\rm fp}=4$). Family 0 and, for even
$N_{\rm fp}$, $q=N_{\rm fp}/2$ are real; the others are complex. One AGNI call evaluates one
family, so every objective must loop over all of them. A family is reported stable only when its
eigensolve converged and its eigenvalue is above the numerical floor.

## Why `Basis(restrict=True)` is required

The unknown is a nodal vector. The derivative matrices are truncated at `mpol` and `ntor`, so
a displacement above the truncation, or at the Nyquist frequency of the grid, is mapped to zero
derivative. It bends no field line and can pose as an instability. On an even-$N_{\rm fp}$ torus
some family always contains the full-torus Nyquist mode.

`restrict=True` stores an orthonormal basis $U$ of the range of each derivative matrix (plus the
constant where a derivative annihilates it) and solves the Hermitian $H=U^\dagger A U$. The
restricted dimension depends only on $(M,N)$: at $(6,5)$ with 10 radial nodes it is 4004
(family 0, real) and 3640 (family 1, complex). Gradients hold the unwhitened displacement
fixed, so the moving mass metric is differentiated. It is implemented for the separable radial
bases (`radial="lobatto"`), with `eigsh` and dense solves.

On the finite-beta QA deck ($N_{\rm fp}=2$, $8\times12\times8$ nodes) the nodal operator gives
top $\gamma^2$ of $5.14\times10^{-3}$ (family 0) and $6.92\times10^{-3}$ (family 1); the
restricted operator gives $2.85\times10^{-3}$ and $3.39\times10^{-3}$, a factor 1.8 and 2.0.
The restricted values match an independent dense `eigvalsh(U^H A U)` to nine digits.

## Choosing the collocation nodes

The nodes are collocation points for the products of equilibrium coefficients that make up
the operator (metric, $\sqrt g$, $\mathbf J$, $\mathscr D_r$). The bandwidth $(M,N)$ and the
node count $(n_\theta,n_\zeta)$ are separate choices, and the nodes must oversample:

$$
n_\theta \ge 2M+16,\qquad n_\zeta \ge 2N+14 .
$$

The QA deck ($N_{\rm fp}=2$, VMEX ns 45, mpol = ntor = 5, $n_\rho=10$, $\varepsilon=0.05$) shows
why. Refining the nodes at fixed bandwidth $(6,5)$, the largest $\gamma^2$ over both families is

| nodes $n_\theta\times n_\zeta$ | $15\times13$ | $19\times17$ | $25\times21$ | $31\times27$ |
|---|---|---|---|---|
| top $\gamma^2$ | $2.7\times10^{-3}$ | $6.1\times10^{-4}$ | $2.3\times10^{-5}$ | $1.1\times10^{-6}$ |

A joint sweep with $n_\theta=2M+1$, $n_\zeta=2N+1$ (the usual choice) gave $6.9\times10^{-3}$
at $13\times11$ (family 0) and $3.6\times10^{-3}$ at $21\times19$ and never left the aliased
regime. At adequate nodes the same deck has no resolved instability: top $\gamma^2\le2\times10^{-6}$
for bandwidths up to $(8,7)$ (both families) and $(10,9)$ (family 0). The earlier "unstable"
values were aliasing error. The same collapse occurs on a VMEX ns 91, mpol = ntor = 8 equilibrium,
so it is not an equilibrium-resolution effect.

The margin of 16 and 14 nodes is a property of the equilibrium coefficients, not of the
bandwidth, and is measured on this deck. Re-measure it when the equilibrium changes: $(4,3)$
at $21\times19$ and $29\times25$ costs about 4 minutes. Both examples assert the rule.

## Differentiating through the equilibrium

Let $z$ be the evolved equilibrium state, $c$ the controls (boundary, profiles, flux),
$F(z,c)=0$ the equilibrium residual and $\mathcal G(z,c)$ the stability objective, including the
adapter. One adjoint solve gives the total derivative of a scalar:

$$
F_z^{\top}a=\mathcal G_z^{\top},\qquad
\frac{d\mathcal G}{dc}=\mathcal G_c-a^{\top}F_c .
$$

AGNI supplies $\mathcal G_z$ and $\mathcal G_c$: for a simple eigenvalue
$d\gamma^2=-v^\dagger\,dH\,v$ (Hellmann-Feynman), with $v$ recomputed at every trial equilibrium.
VMEX supplies the root response ({doc}`adjoint-gradients`). Direct dependence of the map on
the boundary, flux and pressure enters $\mathcal G_c$.

```python
import agnimhd as ag
import vmex as vj
from vmex import optimize as opt
from vmex.core.agni import make_agni_map
from vmex.core.solver import prepare_runtime, resolution_from_input

inp = vj.VmecInput.from_file("examples/data/input.nfp4_QI_finite_beta")
basis = ag.Basis(10, 32, 28, radial="lobatto", mpol=6, ntor=5, restrict=True,
                 automorphism={"eps": 0.02, "x_0": 0.6, "m_1": 2.5, "m_2": 3.0})
nodes, dm = basis.nodes_and_diffmat(inp.nfp, 0)                      # family 0
equilibrium_map = make_agni_map(prepare_runtime(inp, resolution_from_input(inp, ns=45)), nodes, inp=inp)
assembly, solver = ag.AssemblyConfig(gamma=5 / 3, parity=None), ag.SolverConfig(eigensolver="eigsh", sigma=1e-2)

def loss(state, runtime):
    return ag.growth_rate((state, runtime), dm, assembly, solver, equilibrium_map=equilibrium_map)

problem = opt.VmecProblem.from_loss(inp, loss, max_mode=3, warm_start=None)
value, gradient = problem.value_and_grad(problem.x0)
```

`from_loss` takes one scalar. A vector of growth rates must be reduced inside `loss`.
`sigma` must lie above the largest $\gamma^2$ so that shift-invert selects the top mode; the
examples check it against a dense `eigh` of the same restricted block.

**Supported:** first-order reverse-mode gradients of a scalar. **Not supported:** forward mode
and JVPs, derivatives of eigenvectors, and Hessians. A zero eigenvector derivative does not
give a correct second derivative.

**An unresolved eigensolve must not look stable.** The optimization example returns NaN for an
eigensolve with residual above $10^{-9}$, an exception or a non-finite value. The loss is then
non-finite and VMEX rejects the trial. That example also minimizes the sum of the top two
$\gamma^2$ per family rather than one eigenvalue, because the top two modes swap under small
boundary steps (see Limitations). `growth_rate` returns only the top eigenvalue, so the example
takes the top-two eigenvectors from a host dense eigensolve of $H$ and evaluates each $\gamma^2$
as a Rayleigh quotient (`v_fixed=`), which has the Hellmann-Feynman gradient at an eigenvector.

## The examples

`python examples/finite_n_stability.py` solves the deck, evaluates the map once, and runs one
eigensolve per family with a dense cross-check. It takes about ten minutes on 16 cores.
`python examples/optimization/QI_optimization_finite_n.py` stabilizes the same seed; it takes
hours. `VMEX_EXAMPLES_CI=1` runs a short pass that only shows the scripts execute; its values
are not results. Both use `input.nfp4_QI_finite_beta` at twice its shipped pressure (beta 5.1 %,
$N_{\rm fp}=4$), bandwidth $(6,5)$, $32\times28$ nodes, $n_\rho=10$, $\varepsilon=0.02$,
$\Gamma=5/3$, $d=1$.

**Analysis example** (VMEX ns 45; seed survey values):

| family | $\gamma^2$ | verdict |
|---|---|---|
| 0 | $4.5071\times10^{-4}$ | unstable |
| 1 | $4.8772\times10^{-4}$ | unstable |
| 2 | $-5.9\times10^{-8}$ (at $28\times24$) | no resolved instability |

Family 0 is a single-helicity mode, dominant $(m,n)=(3,-4)$ (about half the energy of the
$\pm$ pair) at $\rho=0.68$-$0.78$, near but not on its rational surface ($\iota$ is 1.264-1.296,
the surface is at 4/3). Family 1 is a high-$m$ mode that peaks next to its own rational surface
($7\iota=9$ at $\rho=0.86$ for its $(-7,9)$ pair at $(8,7)$).

**Optimization example** (campaign version with the multi-bandwidth validation; the committed
script has the same objective and optimizer). Seed and optimized boundary, each re-solved at
ns 91 with dense eigensolves of the restricted block, residuals $\le10^{-10}$:

| bandwidth, nodes | family 0 seed $\to$ final | family 1 seed $\to$ final | family 2 seed $\to$ final |
|---|---|---|---|
| $(6,5)$, $32\times28$ | $4.91\to0.82\times10^{-4}$ | $5.11\to0.73\times10^{-4}$ | $-5.9\to-6.3\times10^{-8}$ |
| $(8,7)$, $36\times32$ | $6.32\to1.86\times10^{-4}$ | $14.2\to7.14\times10^{-4}$ | stable |
| $(10,9)$, $40\times36$ | $7.25\to2.36\times10^{-4}$ | $15.9\to9.53\times10^{-4}$ | $3.5\times10^{-4}\to-1.6\times10^{-8}$ |

At $(6,5)$ the final values are 0.167 (family 0) and 0.144 (family 1) of the seed. The
uncertainty from $\varepsilon=0.01$ and $38\times34$ nodes is $3.1\times10^{-6}$ and
$6.7\times10^{-6}$, so the reduction is 130 and 65 times larger. The optimizer sees only
$(6,5)$, so the gain shrinks with bandwidth. **No stability crossing was achieved**: families 0
and 1 stay unstable at every bandwidth. The seed's family 2 is unstable at $(10,9)$; the
optimized boundary is stable there although family 2 is not in the objective. All targets
(beta, aspect ratio, volume, mean and edge $\iota$, constructed QI) stay within their
tolerances; the largest deviation is 0.28 of a tolerance (aspect ratio). The run took 73 evaluations,
about 2.6 h, with a median of 126 s per value and gradient on 16 cores.

## Validation

| gate | measured |
|---|---|
| geometry, flux and $\mathbf B$ identities (QA, $N_{\rm fp}=2$, finite beta) | $10^{-14}$ |
| PEST root residual, every mapped case | $\le 9\times10^{-16}$ |
| explicit-derivative map against the previous autodiff map | $10^{-13}$ ($\sqrt g_r$ at the edge $4\times10^{-11}$); against finite differences $5\times10^{-10}$ |
| $J^v$ from $\nabla\times\mathbf B$ against $\iota J^\phi+p_r/\psi_r$ (QA) | about 12 % at mpol = ntor = 5, 3 % at 8; larger near the axis and the edge |
| restricted Hellmann-Feynman gradient against finite differences | $10^{-9}$ to $2\times10^{-6}$ |
| `eigenpair` (`eigsh`) against dense `eigh`, seed | $2.3\times10^{-11}$ (family 0), $1.3\times10^{-11}$ (family 1) |
| value of `problem.value_and_grad` against a direct solve and `growth_rate` | relative $4\times10^{-13}$ |
| total gradient against frozen-gauge finite differences ($h=10^{-4}$, three directions) | $9\times10^{-8}$ to $3\times10^{-7}$, window $h\in[10^{-4},10^{-3}]$ |
| total gradient against cold re-solved differences | 0.5 %, 5 % and 23 % apart (not an error, see below) |
| top-two sum gradient against central differences, $h=10^{-6}$ (small instance) | $1.0\times10^{-7}$ |
| seed $\gamma^2$ over three node levels, $(6,5)$ | 0.2 % (family 0), 0.9 % (family 1) |
| seed $\gamma^2$, $\varepsilon=0.01$ against 0.02 | +0.4 %, +0.1 % |
| seed $\gamma^2$, ns 91 against ns 45 | +4.9 %, +0.9 % |
| seed $\gamma^2$, mpol = ntor = 7 against 5 | $-24$ %, $-9.6$ % |

The total-gradient and eigensolver rows are for the seed deck. The regression test
`tests/test_agni.py::test_total_gradient_matches_reconverged_differences` checks the gradient
against the frozen-path difference (measured 2e-8) and the sign and size of its gap to the cold
difference (measured 5 %) on a small case. It is skipped when `agnimhd` is missing.

## Limitations

- **Model.** Fixed boundary only, so external kink and wall modes are not covered. The domain
  is the annulus $\varepsilon\le\rho\le1$; $\varepsilon=0.01$ and $0.02$ differ by 0.1-0.4 % on
  the seed, which is not a proof that every mode is insensitive to the cutoff. $\Gamma=5/3$ and
  uniform density were the only settings run.
- **Bandwidth.** Growth rates rise with the bandwidth and have not plateaued. On the seed,
  family 0 is $4.5$, $6.8$, $7.7$, $8.8\times10^{-4}$ for $M=6,8,10,12$ and family 1 is $4.9$,
  $14.5$, $16.5\times10^{-4}$ for $M=6,8,10$. Every value is a lower bound of the growth
  rate in the full space. Dewar et al. discuss why the ideal stellarator spectrum has no
  intrinsic short-wavelength scale, so the cutoff matters. A design that is stable at
  $(6,5)$ may be unstable at higher bandwidth, so validate at $(8,7)$ and $(10,9)$.
- **Stable means checked space.** "No resolved instability" refers to the bandwidth, nodes and
  families tested. Marginal modes of $10^{-6}$ to $10^{-8}$ localize on the innermost cutoff
  shell and are not interior modes. Family 0 also has a cluster of numerically null modes at
  about $10^{-12}$ that is not physical.
- **Equilibrium accuracy.** The equilibrium enters through $\mathbf J$, so its force balance
  matters. Changing mpol from 5 to 7 moves the seed by $-24$ % and VMEC `delt` from 0.9 to 0.5
  moves it by 20 %. Radial resolution and the cutoff were scanned only on the QA deck, on
  values that were still aliased: they changed by 1-2 % above $n_\rho=10$.
- **The $m=1$ gauge.** The implicit gradient is the exact derivative on the branch with the
  antisymmetric $(+n,-n)$ $m=1$ `Z_sin` combination held fixed. A cold re-solve moves it, so
  the re-solved objective is a different function whose slope differs by 0.5-23 % in the
  measured directions. Warm-started re-solves are path dependent through the same gauge: the
  same boundary re-evaluated between other solves gave a top-two sum of $7.2$, $7.7$ and
  $8.1\times10^{-4}$ on the small instance, while cold solves repeat bit for bit. The optimization example therefore uses
  `warm_start=None`, at a cost of 3-10 s per cold ns 45 solve against a 60-130 s evaluation.
  The objective still has a noise floor near $10^{-3}$ relative, which ends line searches.
- **Near-degenerate modes.** The seed's top two family-0 modes are $4.5521\times10^{-4}$ and
  $4.4132\times10^{-4}$, a 3 % gap, and exchange under a step of $10^{-2}$ in some boundary
  coefficients, where the single-eigenvalue gradient jumps. Use a sum over the active cluster, as
  the optimization example does. The sum is smooth only while the next eigenvalue stays apart;
  family 1's third mode sits in the null cluster.
- **Cost.** A value and gradient of family 0 at $(6,5)$ takes 15.7 s warm on an idle 10-core
  laptop and 130 s on a shared node. Complex $(12,11)$ was not run for family 1: it needs an
  estimated 28 GB.

## References

- R. Gaur, S. Patil, P. Gupta, D. Patch and T. Qian, "AGNI: A differentiable MHD stability
  solver & optimizer for magnetic confinement fusion devices,"
  [arXiv:2608.01750](https://arxiv.org/abs/2608.01750).
- I. B. Bernstein, E. A. Frieman, M. D. Kruskal and R. M. Kulsrud, "An energy principle for
  hydromagnetic stability problems," *Proc. R. Soc. A* 244, 17 (1958),
  [doi:10.1098/rspa.1958.0023](https://doi.org/10.1098/rspa.1958.0023).
- R. L. Dewar, C. Cuthbert and R. Ball, "Strong quantum chaos in the global ballooning mode
  spectrum of three-dimensional plasmas," *Phys. Rev. Lett.* 86, 2321 (2001),
  [doi:10.1103/PhysRevLett.86.2321](https://doi.org/10.1103/PhysRevLett.86.2321).

These are entries 52-54 of {doc}`/project/references`.
