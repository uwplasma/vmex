# Solver algorithms and other equilibrium codes

This page places the VMEX solver among the published algorithms and codes,
and records which solver and preconditioner suit which kind of run. The
equations themselves are derived in [the variational problem](variational-problem.rst),
the iteration in [the solver](iteration.rst) and the vacuum solve in
[NESTOR](nestor-vacuum.rst). Measurements live in
[the benchmark reference](../reference/performance.rst).

## What VMEX solves

VMEX minimizes the ideal-MHD energy at fixed toroidal flux, with either the
rotational transform or the enclosed current prescribed:

$$
W = \int_V \Big(\frac{B^2}{2\mu_0} + \frac{p}{\gamma - 1}\Big)\, dV,
\qquad
\sqrt g\, B^\theta = \chi' - \Phi'\,\partial_\zeta\lambda,
\quad
\sqrt g\, B^\zeta = \Phi'\,(1 + \partial_\theta\lambda).
$$

$R$, $Z$ and the stream function $\lambda$ are Fourier series in
$(\theta, \zeta)$ and finite differences in $s$: $R, Z, \lambda$ on the full
radial mesh, $\sqrt g$, $B$ and $p$ on the half mesh (Hirshman & Whitson
1983). The force residuals are the Fourier-projected gradients
$F_{X,mn} = -\partial W/\partial X_{mn}$, normalized to `fsqr`, `fsqz`,
`fsql`. They measure the energy gradient, not the local force
$|\mathbf J\times\mathbf B - \nabla p|$. Panici et al. (2023) show the two can
differ by orders of magnitude near the axis.

Free boundary adds the vacuum pressure balance
$[\![B^2/2\mu_0 + p]\!] = 0$ on the last closed surface. NESTOR obtains the
vacuum field from an exterior Neumann problem for the scalar potential
(Merkel 1986; Hirshman, van Rij & Merkel 1986). The virtual-casing principle
gives the plasma field outside the boundary from the surface current
$\mathbf n\times\mathbf B/\mu_0$ with one singular surface integral and no
linear solve (Shafranov & Zakharov 1972; Drevlak et al. 2005).

The stability and confinement diagnostics are post-processing of the same
state: the Mercier criterion $D_\mathrm{Merc} = D_S + D_W + D_I + D_G$,
ideal ballooning on field lines (Connor, Hastie & Taylor 1978; Sanchez et al.
2000), maximum-$J$ ($\partial J/\partial\psi < 0$ for trapped orbits),
quasisymmetry (Boozer spectra, or the two-term form of Helander & Simakov
2008), and quasi-isodynamicity (Cary & Shasharina 1997; Goodman et al. 2023).

## The iteration and its preconditioner

VMEX keeps VMEC's damped second-order Richardson iteration
$\ddot x + \tau^{-1}\dot x = M^{-1}F(x)$ (Hirshman & Whitson 1983). With
optimal damping it needs $O(\sqrt{\kappa(M^{-1}H)})$ iterations, against
$O(\kappa)$ for steepest descent, so the preconditioner $M$ decides the
cost.

$M$ is the radial tridiagonal approximation of the energy Hessian of
Hirshman & Betancourt (1991), one tridiagonal system per $(m, n)$ and field,
built from $\theta$- and $\zeta$-averaged metric coefficients. It ignores:

- $m$–$m'$ and $n$–$n'$ coupling from the angular variation of the metric;
- coupling between $R$, $Z$ and $\lambda$;
- the radial structure of the $\lambda$ block, which VMEC treats as diagonal.

The last point matters for $m = 0$, $n \ge 1$: $\lambda$ enters $B^\theta$ as
the half-mesh average $(\lambda_{j-1} + \lambda_j)/2$, so the radial
checkerboard $(-1)^j$ is nearly free. A diagonal preconditioner leaves it as
the slowest mode, visible as surface-to-surface sign flips in $\langle
\mathbf J\cdot\mathbf B\rangle$ and $D_\mathrm{Merc}$. The exact tridiagonal
block costs one extra solve per toroidal mode; see [the
solver](iteration.rst).

The slowest remaining mode near the axis couples $R/Z$ ($m \le 2$, $n = 0$)
to $\lambda$ ($m = 1$–$4$, $n = 0$). A field-diagonal preconditioner cannot
reach it.

## Other codes

| code | radial and angular basis | solver |
| --- | --- | --- |
| VMEC2000 / PARVMEC | finite differences × Fourier | preconditioned Richardson; optional full block-tridiagonal Hessian (`precon2d`) factored by cyclic reduction (Hirshman et al. 2010) |
| DESC (Dudt & Kolemen 2020; Panici et al. 2023) | Zernike × Fourier | collocation of the force; trust-region Newton with AD Jacobians; perturbation and continuation (Conlin et al. 2023) |
| DESC free boundary (Conlin et al. 2024) | as DESC | boundary residuals of $\mathbf B\cdot\mathbf n$ and pressure balance with high-order singular quadrature |
| GVEC (Hindenlang, Plunk & Maj 2025; Hindenlang et al. 2026) | B-splines × Fourier, flexible frame | energy minimization with a per-mode banded B-spline preconditioner |
| SPEC (Hudson et al. 2012) | multi-region relaxed MHD, Chebyshev × Fourier | Newton on interface geometry; allows islands |
| MRX (Blickhan, Stratton & Kaptanoglu 2025) | finite-element exterior calculus, no nested surfaces | relaxation and inexact Newton in JAX |

Panici et al. report a radial convergence slope of about $-1$ for VMEC's
finite differences on the D-shaped benchmark, against exponential
convergence in DESC's Zernike degree. High accuracy with VMEC's
discretization therefore needs large `ns`, where Richardson iteration counts
grow.

Library building blocks:

- **Sparse direct solvers.** MUMPS (Amestoy et al. 2001) and SuperLU_DIST
  (Li & Demmel 2003) pay off for global spectral Jacobians. An exactly
  block-tridiagonal radial Hessian is solved optimally by block Thomas
  ($O(ns\,k^3)$) or, in parallel, by cyclic reduction.
- **PETSc SNES** gives Newton–Krylov with Eisenstat–Walker forcing (Knoll &
  Keyes 2004), field-split Schur preconditioners and geometric multigrid.
- **FFTW** (Frigo & Johnson 2005) real-to-real transforms are $O(N\log N)$.
  At VMEC sizes the separable partial sum, one toroidal stage and one
  poloidal stage per parity, is as fast.

## Which solver for which run

**Fixed boundary.** Richardson with the 1-D preconditioner is the right
default for the nonlinear phase: one force evaluation per step, and restarts
absorb Jacobian sign changes. Add the $m = 0$ $\lambda$ block. In the linear
tail ($f_{sq} \lesssim 10^{-6}$) Newton–Krylov with exact JVPs converges in
far fewer force evaluations, provided its preconditioner captures the
coupling above (VMEX's optional `PRECON_TYPE = 'GMRES'`). Anderson
acceleration helps at stage entry but limit-cycles in the linear tail.

**Optimization inner solves.** Hot-start from the previous equilibrium,
finish with a Newton tail, and reuse its factorization or Krylov space for
the implicit adjoint. GMRES preconditioned only by the 1-D $M$ converges
poorly on the transposed system; an assembled block-tridiagonal Hessian
(3 colourings of JVPs, independent of `ns`) with block Thomas is the
dependable alternative.

**High resolution.** The first-order radial error, not the solver, bounds
accuracy. Options are Richardson extrapolation in `ns` and a higher-order
radial basis (B-splines, as in GVEC). Until then: `ns` continuation, then
a Newton tail with per-$n$ block LU to bound memory.

**GPU.** Matrix-free Richardson and JVP-GMRES are bandwidth-bound and map
well. Many small tridiagonal sweeps are latency-bound; dense block
factorizations (BLAS3) use the hardware better.

**Free boundary.** NESTOR is second-order accurate in grid points and costs
a dense solve per vacuum update. A second-kind boundary integral equation
with partition-of-unity singular quadrature (Malhotra et al. 2019) reaches
10th–12th order and needs only GMRES with resolution-independent iteration
counts. The growth of iteration counts with `mpol` in free boundary is a
preconditioning problem: the vacuum response of an edge displacement adds a
stiffness $\sim (\mathbf k\cdot\mathbf B)^2/|\mathbf k_\perp|$ per mode that
the edge rows of $M$ do not contain.

## Pallas kernels on GPU

Measured on one RTX A4000 with JAX 0.11.2:

- **Batched radial tridiagonal solve.** A Pallas Thomas kernel, one program
  per block of columns sweeping rows in registers, is 2–5 times faster than
  cuSPARSE and the scan Thomas from about $10^4$ columns. Ensembles and
  high-resolution decks reach that size. At a single VMEC solve's few hundred
  columns cuSPARSE wins: 0.17 against 0.24 ms at `ns = 51` in float64.
  The kernel is an opt-in SOLVAX method.
- **NESTOR Green's function assembly.** A tiled Pallas kernel loses to XLA
  on a 32 × 32 surface grid. On 64 × 64 it wins only at `mf = nf = 12`
  (18 against 30 ms) and loses at `mf = nf = 8` (22 against 14 ms).
- **Fourier synthesis and projection** are small batched matmuls that XLA
  already fuses. A Pallas fusion was not attempted.

## Host-device sharding on CPU

Splitting one equilibrium's radial axis over forced host devices
(`--xla_force_host_platform_device_count`) slows a preconditioned force
evaluation: 4.5, 9.3, 16.4 and 26.9 ms on 1, 2, 4 and 8 devices (`ns = 128`,
128 modes, Apple M-series). XLA already multithreads one device, and the
radial tridiagonal sweeps force gathers between shards. Parallelism across
independent equilibria (`vmex.core.parallel`) remains the right use of
cores.

## Candidate preconditioner

Average the metric coefficients over $\zeta$ only, not over $\theta$. The
Hessian then stays block-diagonal in $n$ but couples all $m$ and all three
fields, with $3\,\mathrm{mpol}\times 3\,\mathrm{mpol}$ blocks per surface.
Factoring it costs $O(ns\,(\mathrm{ntor}+1)(3\,\mathrm{mpol})^3)$, about
$6\times10^7$ flops at `ns = 99`, `mpol = ntor = 12`, against
$\sim 10^{11}$ for the full 2-D block. It contains the axis coupling and the
$\lambda$ checkerboard, and serves as $M$ for Richardson or as the right
preconditioner of Newton–Krylov. It is untested in VMEX.

## References

- Amestoy, Duff, L'Excellent & Koster (2001), SIAM J. Matrix Anal. Appl. 23, 15.
- Blickhan, Stratton & Kaptanoglu (2025), arXiv:2510.26986.
- Cary & Shasharina (1997), Phys. Rev. Lett. 78, 674.
- Conlin et al. (2023), J. Plasma Phys. 89, 955890305.
- Conlin et al. (2024), arXiv:2412.05680.
- Connor, Hastie & Taylor (1978), Phys. Rev. Lett. 40, 396.
- Drevlak, Monticello & Reiman (2005), Nucl. Fusion 45, 731.
- Dudt & Kolemen (2020), Phys. Plasmas 27, 102513, doi:10.1063/5.0020743.
- Frigo & Johnson (2005), Proc. IEEE 93, 216.
- Goodman et al. (2023), J. Plasma Phys. 89, 905890504.
- Helander & Simakov (2008), Phys. Rev. Lett. 101, 145003.
- Hindenlang, Plunk & Maj (2025), Plasma Phys. Control. Fusion 67, 045002.
- Hindenlang et al. (2026), J. Open Source Softw. 11, 9670, doi:10.21105/joss.09670.
- Hirshman & Whitson (1983), Phys. Fluids 26, 3553, doi:10.1063/1.864116.
- Hirshman, van Rij & Merkel (1986), Comput. Phys. Commun. 43, 143.
- Hirshman & Betancourt (1991), J. Comput. Phys. 96, 99.
- Hirshman et al. (2010), J. Comput. Phys. 229, 6392.
- Hudson et al. (2012), Phys. Plasmas 19, 112502.
- Knoll & Keyes (2004), J. Comput. Phys. 193, 357.
- Li & Demmel (2003), ACM Trans. Math. Softw. 29, 110.
- Malhotra et al. (2019), J. Comput. Phys. 397, 108791.
- Merkel (1986), J. Comput. Phys. 66, 83.
- Panici et al. (2023), J. Plasma Phys. 89, 955890303.
- Sanchez et al. (2000), J. Comput. Phys. 161, 576.
- Shafranov & Zakharov (1972), Nucl. Fusion 12, 599.
