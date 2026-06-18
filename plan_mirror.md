# Plan: fixed-boundary mirror equilibria in `vmec_jax` using Chebyshev--Gauss--Lobatto axial discretization

Status: planning document for a new `vmec_jax` mirror-geometry research lane.

Last updated: 2026-06-15.

Primary target repository: <https://github.com/uwplasma/vmec_jax>

Primary refactor context: PR #20, "[codex] Research-grade differentiability refactor umbrella", <https://github.com/uwplasma/vmec_jax/pull/20>.

Primary external design targets: VMEC, DESC, GVEC-style variational ideal-MHD equilibrium solves, not Grad--Shafranov or Green's-function equilibrium solves.

---

## 0. Executive summary

We want `vmec_jax` to solve ideal-MHD equilibria for **mirror geometries**: straight-axis, open-ended magnetic configurations whose flux surfaces are nested open flux tubes rather than closed toroidal surfaces. The first implementation should be **fixed boundary**, scalar finite-beta, differentiable, and compatible with future optimization and mirror-Boozer-like diagnostics. Free boundary and anisotropic pressure come later.

The recommended first production implementation is:

\[
(s,\theta,\xi),\qquad s\in[0,1],\quad \theta\in[0,2\pi),\quad \xi\in[-1,1],
\]

with:

- `s`: VMEC-like flux-surface label, initially a finite-difference or nodal radial grid.
- `theta`: periodic azimuthal angle around the straight mirror axis, represented with Fourier modes.
- `xi`: nonperiodic axial coordinate, represented at Chebyshev--Gauss--Lobatto nodes.
- fixed side boundary `s = 1`, prescribed by `r_b(theta, xi)` or later by `X_b(theta, xi), Y_b(theta, xi), Z_b(theta, xi)`.
- open end planes `xi = -1` and `xi = +1`, not periodically identified.
- a VMEC-like divergence-free contravariant magnetic-field representation with `B^s = 0` and a stream function `lambda`.
- variational minimization or residual solve of the ideal-MHD energy functional.

The first useful solver should be the narrowest robust case:

```text
straight-axis mirror
axisymmetric or weakly 3D fixed side boundary
Fourier(theta) x Chebyshev-Lobatto(xi)
VMEC-like radial finite-difference grid
scalar p(s)
Psi'(s) prescribed
I'(s) optional, initially zero
lambda carried in state, initially allowed to be zero for simple axisymmetric cases
mout_*.nc mirror-native output
vmec --plot mout_*.nc works
mirror-specific Boozer-like straight-field-line transform later
```

The implementation should not fake a torus with a huge aspect ratio, should not use Fourier modes in the axial coordinate, and should not replace VMEC with Grad--Shafranov or Green's functions. Axisymmetric Grad--Shafranov and Pleiades results may still be used as optional validation comparisons.

---

## 1. Scope and non-goals

### 1.1 In scope for the fixed-boundary phase

1. A new mirror backend that follows the `vmec_jax` research-grade refactor principles from PR #20.
2. Chebyshev--Gauss--Lobatto nodes and differentiation/quadrature in the axial coordinate.
3. Fourier modes in `theta`, with efficient transforms and derivative matrices.
4. Axisymmetric fixed-boundary mirror equilibria as the first working path.
5. Nonaxisymmetric fixed-boundary surfaces after the axisymmetric solver is reliable.
6. Scalar finite-beta pressure profiles `p(s)`.
7. VMEC-like divergence-free contravariant field representation.
8. Variational energy, residuals, and JAX gradients.
9. Real numerical tests, manufactured-solution tests, physics tests, code-parity tests, convergence tests, and benchmarks.
10. Mirror-native I/O, plotting, examples, and documentation.
11. Mirror optimization objectives that call the fixed-boundary mirror solver.
12. A mirror-specific straight-field-line / Boozer-like transform.

### 1.2 Deferred until fixed boundary is stable

1. Free-boundary mirror equilibria.
2. Vacuum-region PDE/spectral potential solve for free boundary.
3. External coil optimization coupled to a mirror free-boundary solve.
4. Anisotropic pressure closure and kinetic-coupled closures.
5. Full 3D transverse geometry with arbitrary non-circular axis or generalized Frenet frame.
6. Production B-spline or spectral-element axial basis.
7. Open-field-line loss models, sheath/end-plate physics, or kinetic distribution evolution.

### 1.3 Explicit non-goals

1. Do not implement the primary mirror equilibrium solver as Grad--Shafranov.
2. Do not implement the primary mirror equilibrium solver using Green's functions.
3. Do not make the mirror a closed torus by imposing artificial axial periodicity.
4. Do not store mirror results in classic toroidal `wout_*.nc` files without a mirror-native schema.
5. Do not make toroidal `booz_xform_jax` consume mirror files as fake VMEC WOUT files.

---

## 2. Source and literature anchors

This section lists the sources that should be cited in code comments, tests, documentation, and PR descriptions where relevant. Some are direct implementation anchors; others are validation or design context.

### 2.1 `vmec_jax` source and PR #20 anchors

- Repository: <https://github.com/uwplasma/vmec_jax>
- Current attached source inspected locally from `vmec_jax-main(1).zip`.
- PR #20: <https://github.com/uwplasma/vmec_jax/pull/20>
- PR #20 title: `[codex] Research-grade differentiability refactor umbrella`.
- PR #20 branch: `codex/differentiability-refactor-plan`.
- PR #20 head SHA at inspection: `f6bee2dae1b0b0aee55a1f5fa93e9b7ae671c896`.
- PR #20 base: `main`, base SHA at inspection: `807255b2d29c2c733234004024faa6bcad9df8c1`.
- PR #20 was open, draft, mergeable, with 114 commits and 126 changed files at inspection.
- PR #20 key architectural instruction: expose a small public API, keep implementation organized by scientific/numerical responsibility, stop adding flat root-level helper modules, keep compatibility shims thin, and make tests target domain packages.

Important PR #20 documents and implications:

- `plan_differentiability.md` states that long-term `vmec_jax` should keep VMEC-compatible fixed/free-boundary physics while exposing validated derivatives for equilibrium, optimization, finite-beta metrics, Boozer objectives, and stability objectives.
- `plan_differentiability.md` defines a target package architecture with `api.py`, `cli.py`, `core/`, `kernels/`, `solvers/`, `objectives/`, `optimization/`, `io/`, `plotting/`, `validation/`, and `performance/`.
- The plan explicitly says the new domain package architecture supersedes older flat `solve_*`, `driver_*`, and `free_boundary_*` helper-file proliferation.
- The plan gives naming rules: avoid new root-level `solve_*`, `driver_*`, `free_boundary_*`, and `wout_*` modules; avoid vague `_helpers`, `_utils`, `_misc`, `_common`; use nouns that describe scientific domain objects.
- It defines line-count and cognitive-load budgets: new functions usually under 80 lines, files usually under 800 lines, warning above 1500 lines, hard gate above 2000 lines once migration is active.
- It defines validation gates: numerical identities, AD-vs-central-FD, external parity, physics gates, artifact reproducibility, and performance gates.

The mirror implementation should follow these rules even if developed on `main` before or outside the PR #20 branch.

### 2.2 VMEC theory and implementation anchors

- VMEC/STELLOPT documentation: <https://princetonuniversity.github.io/STELLOPT/VMEC.html>
- VMEC uses a variational method to minimize total energy and assumes Fourier expansion in poloidal and toroidal coordinates.
- VMEC seeks ideal-MHD force balance in a toroidal domain:
  \[
  -\mathbf j\times\mathbf B + \nabla p = 0,\qquad
  \nabla\times\mathbf B = \mu_0\mathbf j,\qquad
  \nabla\cdot\mathbf B = 0.
  \]
- VMEC writes the energy as:
  \[
  W=\int\left(\frac{|\mathbf B|^2}{2\mu_0}+\frac{p}{\gamma-1}\right)d^3x.
  \]
- VMEC enforces flux conservation and `div B = 0` through a contravariant magnetic-field representation and a field-line-straightening stream function `lambda`.
- VMEC fixed boundary prescribes Fourier amplitudes at the outer flux surface. Free boundary incorporates vacuum fields and a vacuum scalar potential.

Mirror `vmec_jax` should keep the VMEC variational/divergence-free structure but replace toroidal topology with open-ended mirror topology.

### 2.3 DESC documentation and source anchors

- DESC docs: <https://desc-docs.readthedocs.io/en/stable/>
- DESC basis/grid docs: <https://desc-docs.readthedocs.io/en/stable/notebooks/basis_grid.html>
- DESC source: <https://github.com/PlasmaControl/DESC>
- DESC current `desc/basis.py` source anchor: <https://github.com/PlasmaControl/DESC/blob/master/desc/basis.py>
- DESC `basis.py` exports `ChebyshevDoubleFourierBasis` and `ChebyshevPolynomial`.
- DESC `ChebyshevDoubleFourierBasis` is a tensor product of Chebyshev polynomials and two Fourier series. It is useful as a transform implementation reference, but it is not directly the mirror basis because mirror geometry needs Chebyshev in axial `xi`, Fourier in azimuthal `theta`, and no periodic toroidal coordinate.
- DESC `ChebyshevDoubleFourierBasis.evaluate` evaluates the product of Chebyshev, poloidal Fourier, and toroidal Fourier factors and supports derivatives.
- DESC docs state that Fourier-Zernike basis represents toroidal equilibrium positions and stream function, and that Zernike polynomials are useful because they automatically satisfy magnetic-axis regularity.
- DESC docs mention Chebyshev-Gauss-Lobatto node patterns in concentric grids.

DESC active/recent development threads relevant to this plan:

1. PR #2012: <https://github.com/PlasmaControl/DESC/pull/2012>
   - Title: `Fourier-Chebyshev Fit to B for particle tracing`.
   - Branch: `yge/particle-fit`.
   - Head SHA at inspection: `7b21b40f10d95b97a5f7bbad5eca95aeee195912`.
   - Status at inspection: open draft.
   - Adds a `FourierChebyshevField` with `init`, `build`, `fit`, and `evaluate` methods compatible with optimization.
   - Explicit TODO: examine spectral convergence and consider a spline version.
   - Lesson for `vmec_jax`: use build/fit/evaluate separation, cache transforms, keep field interpolation compatible with optimization, and test spectral convergence.

2. PR #2194: <https://github.com/PlasmaControl/DESC/pull/2194>
   - Title: `correct CGL nodes in OmnigenousField.change_resolution`.
   - Branch: `dd/omni_bugfix`.
   - Head SHA: `1c616e13c8672a2facf692863853537b1aa8afde`.
   - Merge commit SHA: `6b3978fc63151e16956a33dc76cdcc0b25538879`.
   - Status at inspection: merged.
   - Body says Chebyshev-Gauss-Lobatto nodes used for resolution change were incorrect, affecting interpolation accuracy.
   - Lesson for `vmec_jax`: CGL nodes, ordering, interpolation, and resolution-change tests must be correctness-critical tests, not smoke tests.

3. PR #1508: <https://github.com/PlasmaControl/DESC/pull/1508>
   - Title: `Poloidal FFT Implementation`.
   - Branch: `rg/poloidal_fft`.
   - Head SHA at inspection: `c74b6b6af04a04591f2937af77d6044fb430c61f`.
   - Status at inspection: open draft.
   - Body says initial implementation works for Fourier, DoubleFourier, and FourierChebyshev basis, but not Zernike yet.
   - Lesson for `vmec_jax`: separable FFT/matrix transforms are preferable to dense full-node-by-mode matrices; Zernike coupling complicates transforms, so start with a simpler VMEC-like radial grid and Fourier-Chebyshev tensor product.

4. PR #1893: <https://github.com/PlasmaControl/DESC/pull/1893>
   - Title: `A variational principle based ideal MHD stability solver and optimizer`.
   - Status at inspection: open.
   - Body emphasizes discretizing an energy principle and maintaining symmetric generalized eigenvalue structure.
   - Lesson for `vmec_jax`: because the mirror solver is variational, tests should include gradient checks and Hessian symmetry checks.

### 2.4 GVEC anchors

- GVEC docs: <https://gvec.readthedocs.io/latest/>
- GVEC source mirror: <https://github.com/gvec-group/gvec>
- GVEC is an open-source flexible 3D ideal-MHD equilibrium solver inspired by VMEC.
- GVEC uses radial B-splines of arbitrary polynomial degree and Fourier series in poloidal/toroidal directions.
- GVEC supports flexible mappings not restricted to standard cylindrical coordinates.
- Lesson for `vmec_jax`: Chebyshev-Lobatto is a good first axial basis, but the mirror backend should keep an axial-basis abstraction that can later support multi-domain Chebyshev, B-splines, or spectral elements.

### 2.5 WHAM, Pleiades, and mirror-physics anchors

- WHAM physics basis article: <https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/physics-basis-for-the-wisconsin-hts-axisymmetric-mirror-wham/35CCAE07989A73709B38C15F38A5CDBE>
- DOI: <https://doi.org/10.1017/S0022377823000806>
- WHAM paper notes that WHAM is a high-field axisymmetric mirror platform and that finite-beta anisotropic ion distributions are part of equilibrium modelling.
- WHAM paper states that the 17 T, 2 kA steady-state, 5.5 cm warm-bore HTS mirror magnets are centered at `z = +/- 98 cm`.
- WHAM paper discusses mirror ratio, expansion ratio, finite-beta diamagnetism, and high-beta anisotropic equilibrium modelling.
- Pleiades docs: <https://pleiades.readthedocs.io/en/latest/>
- Pleiades computes axisymmetric magnetic fields and includes an MHD equilibrium solver used to generate magnetic mirror equilibria and GENRAY/CQL3D inputs for WHAM.
- Pleiades is not the target equilibrium algorithm here, but it is a useful optional comparison code for vacuum fields, geometry, and axisymmetric mirror equilibria.
- Uploaded `coil_model_WHAM-2.txt` script:
  - Uses `magpylib` to build a two-coil WHAM-like mirror field.
  - Uses 8 axial layers and 310 radial windings.
  - Uses coil-pack centers at `z = +/- 0.98 m`.
  - Uses `I_coil = 2000 * 17.0 / 17.51 A`.
  - Computes `|B|` contours on an `(r,z)` grid and on-axis `B_z(z)`.
  - Cites the IEEE Transactions on Applied Superconductivity paper: `Design of High Field HTS Coils for Magnetic Mirror`, Radovinsky et al., 2023.

Additional mirror-related paper anchor:

- `Nonlinear anisotropic equilibrium reconstruction in axisymmetric magnetic mirrors`, arXiv:2509.17288, <https://arxiv.org/abs/2509.17288>
- The abstract states that the work extends nonlinear equilibrium reconstruction to high-beta anisotropic-pressure plasmas and applies it to WHAM.
- Lesson for future phases: design pressure APIs now so anisotropic closures can be added later without rewriting geometry and field kernels.

### 2.6 Numerics and validation anchors

- VMEC++ numerics paper: <https://arxiv.org/abs/2502.04374>
  - Useful for modern VMEC implementation practices, parity philosophy, restart behavior, and Python-friendly APIs.
- Clenshaw--Curtis quadrature: <https://en.wikipedia.org/wiki/Clenshaw%E2%80%93Curtis_quadrature>
  - Useful reference for Chebyshev-related quadrature; code should cite a primary numerical analysis reference in docs if possible.
- Trefethen, `Spectral Methods in MATLAB`, for Chebyshev-Lobatto derivative matrices and spectral convergence tests.
- Boyd, `Chebyshev and Fourier Spectral Methods`, for Chebyshev filtering, aliasing, and endpoint behavior.
- Roache, `Verification and Validation in Computational Science and Engineering`, for method-of-manufactured-solutions methodology.

---

## 3. How this fits into the PR #20 architecture

The mirror implementation should not add new root-level `vmec_jax/mirror_*.py`, `solve_mirror_*`, or `driver_mirror_*` files. It should be a domain package with a small public surface and cohesive implementation modules.

### 3.1 Recommended mirror package tree

Use a new `vmec_jax/mirror/` package that mirrors the PR #20 domain architecture internally:

```text
vmec_jax/
  mirror/
    __init__.py                 small public mirror namespace
    api.py                      user-facing mirror workflows and compatibility surface

    core/
      __init__.py
      config.py                 MirrorConfig, MirrorSolveOptions, enums, units
      state.py                  MirrorStateAxisym, MirrorState3D, PyTree registration
      grids.py                  MirrorGrid, radial/theta/xi nodes, weights, masks
      basis.py                  ThetaFourierBasis, AxialBasis protocol, transform cache
      profiles.py               ScalarMirrorPressure, flux/current/twist profiles
      boundary.py               MirrorBoundary, parameterizations, constraints
      runtime.py                dtype/JIT/runtime settings, optional dependency flags

    kernels/
      __init__.py
      chebyshev.py              CGL nodes, D matrices, Clenshaw-Curtis weights
      fourier.py                theta FFT/evaluation/derivatives, real-mode conventions
      geometry.py               embeddings, metrics, Jacobian, volume, shape checks
      fields.py                 B contravariant/covariant/cartesian kernels
      energy.py                 magnetic/pressure energy densities and totals
      forces.py                 variational gradients/force blocks and projections
      residuals.py              residual assembly, norms, physical scalar diagnostics
      constraints.py            axis/boundary/lambda gauge constraints
      manufactured.py           MMS source-term and exact-solution helpers
      filtering.py              Chebyshev/Fourier spectral filters and mode spectra

    solvers/
      __init__.py
      fixed_boundary/
        __init__.py
        api.py                  run_mirror_fixed_boundary
        continuation.py         pressure/resolution/stage continuation
        nonlinear.py            residual/energy solve orchestration
        types.py                optimizer options, step, run, diagnostics payloads
        reduced.py              reduced-coordinate masks, packing, scaling, bounds
        preconditioners.py      residual preconditioners, adaptive inner budgets
        optimizers.py           high-level GD/L-BFGS-B/residual-Newton dispatch
        checkpoints.py          restart payloads, deterministic artifacts
        diagnostics.py          trace rows, residual histories, shape guards

      free_boundary/
        __init__.py
        api.py                  future run_mirror_free_boundary
        domains.py              plasma/vacuum/wall domain maps
        vacuum_potential.py     future PDE/spectral scalar-potential solve
        interfaces.py           plasma-vacuum continuity residuals
        wall.py                 conducting wall geometry and boundary conditions
        diagnostics.py          future Bnormal/pressure/interface diagnostics

      differentiation/
        __init__.py
        policies.py             exact/implicit/scalar-adjoint policy objects
        implicit.py             root/JVP/VJP helpers for fixed-boundary solve
        finite_difference.py    central-FD validation helpers
        linear_solvers.py       matrix-free CG/dense fallback for small tests

    objectives/
      __init__.py
      mirror_ratio.py           target mirror ratio and well-depth objectives
      bfield.py                 B-profile objectives, Bmax/Bmin, smoothness
      boundary.py               boundary smoothness, wall clearance, symmetry
      constraints.py            min-Jacobian, radius positivity, endpoint guards
      least_squares.py          objective tuple/object assembly

    optimization/
      __init__.py
      boundary.py               boundary DOF spaces and transforms
      workflow.py               Simsopt-like optimization assembly
      callbacks.py              derivative policies and accepted-solve callbacks
      result.py                 histories, artifacts, plotting hooks
      backends/
        __init__.py
        scipy.py                required backend through scipy.optimize
        jaxopt.py               optional backend; lazy import
        optax.py                optional backend; lazy import

    io/
      __init__.py
      mout.py                   mirror-native netCDF read/write
      schema.py                 mout variable definitions, units, versioning
      input.py                  mirror input parsing/writing, YAML/TOML/namelist bridge
      assets.py                 fixture metadata and example asset helpers

    plotting/
      __init__.py
      geometry.py               r-z surfaces, 3D surfaces, wall/boundary plots
      bfield.py                 |B| maps, mirror ratio, spectra
      diagnostics.py            residual, Jacobian, energy histories
      boozer.py                 mirror-Boozer-like spectra and field-line plots
      export.py                 VTK/VTU/NPZ data export

    boozer/
      __init__.py
      transform.py              mirror straight-field-line transform
      spectra.py                |B| Fourier-Chebyshev spectra on mirror coordinates
      fieldlines.py             open-field-line integration diagnostics
      io.py                     mbmn_*.nc mirror-Boozer output

    validation/
      __init__.py
      analytic.py               cylinder/flared-tube analytic checks
      wham.py                   WHAM coil fixture metadata and optional magpylib checks
      desc_parity.py            Fourier-Chebyshev transform parity with DESC
      gvec_parity.py            optional GVEC-style mapping checks
      pleiades_parity.py        optional Pleiades/axisymmetric comparison hooks
      convergence.py            resolution convergence utilities
      manufactured.py           MMS test-case definitions

    performance/
      __init__.py
      profiling.py              cold/warm/JIT timings and memory helpers
      transform_benchmarks.py   transform scaling and allocation benchmarks
```

This structure gives mirror work a clear home while respecting PR #20's rule that new code should not proliferate flat root-level helper modules. If PR #20 lands first and the top-level `core/`, `kernels/`, etc. packages are in place, generic pieces such as Chebyshev utilities may later move to `vmec_jax/kernels/orthogonal.py` or `vmec_jax/core/grids.py`, but the first mirror implementation should keep topology-specific code under `vmec_jax/mirror/`.

### 3.2 Public API additions

Expose only the minimal stable API at first:

```python
import vmec_jax as vj

cfg = vj.mirror.MirrorConfig(...)
result = vj.mirror.run_mirror_fixed_boundary(cfg)
vj.mirror.write_mout(result, "mout_case.nc")
vj.mirror.plot_mirror_output("mout_case.nc")
```

Add to `vmec_jax/api.py` only after the API is stable:

```python
from .mirror.api import (
    MirrorConfig,
    MirrorBoundary,
    MirrorFixedBoundaryResult,
    run_mirror_fixed_boundary,
    load_mirror_output,
    write_mirror_output,
    plot_mirror_output,
)
```

Avoid putting all mirror internals into `vmec_jax/__init__.py` immediately. The top-level import should remain lightweight and not import optional plotting, `magpylib`, or heavy validation packages.

### 3.3 CLI additions

Extend the existing `vmec` CLI by dispatching on input/output type:

```bash
vmec mirror examples/mirror/input.fixed_flared.toml
vmec --plot mout_fixed_flared.nc
vmec --mirror-booz mout_fixed_flared.nc
vmec mirror-optimize examples/mirror/target_ratio.toml
```

Do not add a second executable unless needed. Existing scripts `vmec`, `vmec-jax`, `vmec_jax`, `xvmec_jax` already point to `vmec_jax.cli:main`.

---

## 4. Mathematical formulation

### 4.1 Coordinates

Use mirror flux coordinates:

\[
(s,\theta,\xi),\qquad s\in[0,1],\quad \theta\in[0,2\pi),\quad \xi\in[-1,1].
\]

The coordinate `xi` maps to physical axial coordinate `z` through a monotone map:

\[
z=z(\xi),\qquad z_\xi > 0.
\]

For the first implementation use a linear map:

\[
z(\xi)=z_0+L\xi.
\]

Later allow smooth maps that cluster physical resolution near mirror throats or expanders.

### 4.2 Axisymmetric straight-axis embedding

First implementation:

\[
\mathbf x(s,\theta,\xi)=r(s,\xi)\cos\theta\,\hat{\mathbf x}
+r(s,\xi)\sin\theta\,\hat{\mathbf y}+z(\xi)\hat{\mathbf z}.
\]

Axis regularity is enforced by:

\[
r(s,\xi)=\rho\,a(s,\xi),\qquad \rho=\sqrt{s}.
\]

Boundary:

\[
r(1,\xi)=r_b(\xi),\qquad r(0,\xi)=0.
\]

Coordinate basis vectors:

\[
\mathbf e_s=r_s\hat{\mathbf e}_r,
\]

\[
\mathbf e_\theta=r\hat{\mathbf e}_\theta,
\]

\[
\mathbf e_\xi=r_\xi\hat{\mathbf e}_r+z_\xi\hat{\mathbf e}_z.
\]

Metric entries:

\[
g_{ss}=r_s^2,
\qquad g_{s\theta}=0,
\qquad g_{s\xi}=r_s r_\xi,
\]

\[
g_{\theta\theta}=r^2,
\qquad g_{\theta\xi}=0,
\qquad g_{\xi\xi}=r_\xi^2+z_\xi^2.
\]

Jacobian:

\[
J=\sqrt{g}=\mathbf e_s\cdot(\mathbf e_\theta\times\mathbf e_\xi)=r r_s z_\xi.
\]

For `r = a sqrt(s)` and `z = L xi`,

\[
J=\frac{a^2 L}{2}.
\]

### 4.3 3D straight-axis embedding

After axisymmetric cases work, allow:

\[
r(s,\theta,\xi)=\rho\,a(s,\theta,\xi)
\]

or full transverse coordinates:

\[
\mathbf x(s,\theta,\xi)=X(s,\theta,\xi)\hat{\mathbf x}
+Y(s,\theta,\xi)\hat{\mathbf y}
+Z(s,\theta,\xi)\hat{\mathbf z}.
\]

The `r(s,theta,xi)` representation is simpler and should come first. The full `X,Y,Z` representation is more general and should be designed but not required for the fixed-boundary MVP.

### 4.4 Divergence-free contravariant magnetic field

Use the mirror analogue of VMEC's contravariant form:

\[
B^s=0,
\]

\[
J B^\theta=I'(s)-\partial_\xi\lambda,
\]

\[
J B^\xi=\Psi'(s)+\partial_\theta\lambda.
\]

Then:

\[
\nabla\cdot\mathbf B
=\frac{1}{J}\left[\partial_\theta(JB^\theta)+\partial_\xi(JB^\xi)\right]
=\frac{1}{J}\left[-\lambda_{\xi\theta}+\lambda_{\theta\xi}\right]=0.
\]

`Psi'(s)` is the axial flux derivative. `I'(s)` is an optional twist/current-like flux function. `lambda` is the mirror version of the VMEC field-line-straightening stream function.

For the first axisymmetric MVP, allow:

\[
I'(s)=0,\qquad \lambda=0.
\]

But include `lambda` in state from the start so twisted and nonaxisymmetric cases do not require a redesign.

### 4.5 Magnetic energy

The magnetic energy is:

\[
W_B=\int \frac{B^2}{2\mu_0} J\,ds\,d\theta\,d\xi.
\]

With `B^s = 0`,

\[
B^2 = g_{\theta\theta}(B^\theta)^2
+2g_{\theta\xi}B^\theta B^\xi
+g_{\xi\xi}(B^\xi)^2.
\]

In axisymmetry with `lambda=0`, `I'=0`:

\[
B^\xi=\frac{\Psi'(s)}{J},
\]

\[
B^2=(B^\xi)^2(r_\xi^2+z_\xi^2).
\]

### 4.6 Pressure energy

Start with prescribed scalar pressure:

\[
p=p(s).
\]

Use:

\[
W_p=\int \frac{p(s)}{\gamma-1} J\,ds\,d\theta\,d\xi.
\]

Later add the VMEC mass-conserving pressure model:

\[
p(s)=\frac{M(s)}{\left[\int\int J\,d\theta\,d\xi\right]^\gamma}.
\]

### 4.7 Total energy and residuals

Fixed boundary solves:

\[
W = W_B+W_p.
\]

The solved state should satisfy stationarity under admissible variations:

\[
\frac{\delta W}{\delta a}=0,
\qquad
\frac{\delta W}{\delta\lambda}=0,
\]

with side boundary, axis regularity, end constraints, and lambda gauge imposed by projection.

Use JAX AD initially to form gradients for correctness. Later add analytic force kernels for performance and VMEC-style parity.

### 4.8 Lambda gauge

Since adding a flux function to `lambda` does not change the magnetic field,

\[
\lambda\rightarrow\lambda+c(s),
\]

a gauge must be fixed. For each `s`, remove the surface average:

\[
\langle \lambda\rangle_{\theta,\xi}=0.
\]

In coefficient storage this means the `(m=0,k=0)` lambda mode is removed or constrained to zero. In nodal storage, project after each update.

### 4.9 Boundary and end-plane conditions

Fixed side boundary:

\[
r(1,\theta,\xi)=r_b(\theta,\xi).
\]

Axis:

\[
r(0,\theta,\xi)=0,
\]

with `r = sqrt(s) a` and later `rho^{|m|}` regularity for nonaxisymmetric modes.

End planes are open field-line cuts, not periodic boundaries. For the first MVP choose a conservative fixed-end policy:

\[
r(s,\theta,-1)=\sqrt{s}\,r_b(\theta,-1),
\]

\[
r(s,\theta,+1)=\sqrt{s}\,r_b(\theta,+1).
\]

After the solver is stable, add natural variational end conditions as an option. The end policy must be explicit in `MirrorConfig` and in docs.

---

## 5. Discretization

### 5.1 Radial grid

Use the existing VMEC-like finite radial grid at first. Recommended initial storage:

```python
s_full = linspace(0, 1, ns)
s_half = 0.5 * (s_full[:-1] + s_full[1:])
rho_full = sqrt(s_full)
```

For axis regularity, store `a(s,xi)` and evaluate `r = sqrt(s) * a`. Avoid evaluating singular `r_s` at `s=0` directly; use analytic or one-sided regular formulas at the axis.

Later options:

1. radial B-splines inspired by GVEC;
2. Zernike-like disk basis in `(rho, theta)` inspired by DESC;
3. finite-element radial elements for free boundary.

### 5.2 Azimuthal Fourier grid

Use a uniform periodic grid:

\[
\theta_j=\frac{2\pi j}{N_\theta},\qquad j=0,\ldots,N_\theta-1.
\]

Use real Fourier modes:

\[
f(s,\theta,\xi)=f_0(s,\xi)+\sum_{m=1}^{M}\left[f^c_m(s,\xi)\cos(m\theta)+f^s_m(s,\xi)\sin(m\theta)\right].
\]

For the first axisymmetric solver, set `mpol = 0`, but implement the Fourier basis early enough that tests cover `m > 0` derivatives and transforms.

### 5.3 Chebyshev--Gauss--Lobatto axial nodes

Use:

\[
\xi_j=\cos\left(\frac{\pi j}{N_\xi}\right),\qquad j=0,\ldots,N_\xi.
\]

Public ordering should be increasing physical order:

\[
-1=\xi_0<\xi_1<\cdots<\xi_{N_\xi}=1.
\]

If an internal cosine ordering is used, store a permutation and test it. This is important because DESC PR #2194 found a CGL-node bug in a resolution-change path.

### 5.4 Chebyshev derivative matrix

For nodes in canonical cosine ordering `x_j = cos(pi*j/N)`, the standard first-derivative matrix is:

\[
D_{ij}=\frac{c_i}{c_j}\frac{(-1)^{i+j}}{x_i-x_j},\qquad i\ne j,
\]

with

\[
c_0=c_N=2,\qquad c_j=1 \; \text{otherwise},
\]

and diagonal entries set so rows sum to zero:

\[
D_{ii}=-\sum_{j\ne i}D_{ij}.
\]

If nodes are reordered into increasing physical order, reorder `D` consistently:

```python
D_inc = P @ D_cos @ P.T
```

where `P` maps cosine ordering to increasing ordering.

Second derivative may be computed as:

```python
D2_xi = D_xi @ D_xi
```

for tests and diagnostics, while production code should use first derivatives as much as possible.

### 5.5 Clenshaw--Curtis weights

Use Clenshaw--Curtis weights on `[-1,1]` for `xi`. Implement in pure NumPy/JAX with deterministic tests. Store weights in the public node ordering.

Volume quadrature:

\[
\int fJ\,ds\,d\theta\,d\xi\approx
\sum_{i,j,k} f_{ijk}J_{ijk} w^s_i w^\theta_j w^\xi_k.
\]

### 5.6 Nodal first, coefficient-space later

For the first fixed-boundary implementation, store `a(s,theta,xi)` and `lambda(s,theta,xi)` nodally in `xi`. This makes fixed endpoint constraints, boundary constraints, plotting, and continuation simpler.

Add coefficient-space transforms later for:

1. spectral filtering;
2. mode spectra and convergence diagnostics;
3. compressed output;
4. efficient interpolation to field-line or Boozer-like grids;
5. parity with DESC's Fourier-Chebyshev development.

### 5.7 Dealiasing and filtering

Nonlinear energy terms involve products of geometry and field quantities. For `theta`, support oversampled collocation:

\[
N_\theta \ge 2M+1
\]

for exact representation of linear terms, and optionally `3/2` oversampling for nonlinear products.

For `xi`, include an exponential Chebyshev filter for optional stabilization:

\[
\sigma_k=\exp[-\alpha(k/N)^p].
\]

Filtering should be disabled by default in correctness tests and enabled only as an explicit solver option. Tests should verify that filters reduce high-mode energy without changing low modes.

---

## 6. Solver design

### 6.1 First state object

Axisymmetric nodal state:

```python
@dataclass(frozen=True)
class MirrorStateAxisym:
    a: Array       # shape (ns, nxi), r = sqrt(s) * a
    lam: Array     # shape (ns, nxi), initially zero; gauge-projected
```

3D nodal state after axisymmetry:

```python
@dataclass(frozen=True)
class MirrorState3D:
    a: Array       # shape (ns, ntheta, nxi)
    lam: Array     # shape (ns, ntheta, nxi)
```

Later coefficient state:

```python
@dataclass(frozen=True)
class MirrorStateSpectral:
    a_cos: Array   # shape (ns, m_modes, k_modes)
    a_sin: Array
    lam_cos: Array
    lam_sin: Array
```

All state objects must be JAX PyTrees.

### 6.2 Static object

```python
@dataclass(frozen=True)
class MirrorStatic:
    grid: MirrorGrid
    theta_basis: ThetaFourierBasis
    axial_basis: ChebyshevLobattoBasis
    radial_ops: RadialOperators
    constraints: MirrorConstraintMasks
    dtype: Any
```

Static objects should own precomputed matrices and masks and should not be differentiated with respect to by default.

### 6.3 Fixed-boundary algorithm

Stage 1: build input and static data.

1. Parse `MirrorConfig`.
2. Build radial, theta, and CGL `xi` grids.
3. Build boundary `r_b(theta,xi)`.
4. Build initial guess `r = sqrt(s) r_b(theta,xi)` or a smoothed interior guess.
5. Build pressure and flux profiles.
6. Project fixed side boundary, axis regularity, end policy, and lambda gauge.

Stage 2: solve.

1. Evaluate geometry and field kernels.
2. Compute energy and residuals.
3. Use LBFGS or Gauss--Newton-like residual minimization for fixed-budget solves.
4. Apply constraints by projection after each update.
5. Continue in pressure and resolution.
6. Write `mout_*.nc` and a solve-history JSON/NPZ.

Stage 3: validate and plot.

1. Check finite positive `J`.
2. Check `Bmag > 0`.
3. Check side boundary exactly matches prescribed boundary.
4. Check `div B` identity.
5. Check residual/gradient norm.
6. Generate plots if requested.

### 6.4 Continuation

Use staged continuation similar in spirit to VMEC/DESC:

```text
resolution stage 0: ns small, nxi small, mpol=0, pressure fraction 0
pressure stages: 0 -> 0.1 -> 0.25 -> 0.5 -> 0.75 -> 1
resolution stages: increase ns, nxi, mpol as needed
```

Every stage should record:

```text
stage index
ns, ntheta, nxi, mpol
pressure scale
energy_B, energy_p, energy_total
residual norm
min(J), max(J)
min(B), max(B), mirror ratio
optimizer iterations
line-search failures or restarts
```

### 6.5 Optimizer choices

Required backend:

- SciPy LBFGS-B or trust-region for beginner-friendly deterministic solves.

JAX-native backend:

- plain JAX gradient descent/LBFGS implementation inside `mirror/solvers/fixed_boundary/optimizers.py`.

Optional later backends:

- JAXopt implicit differentiation backend.
- Optax for experimental optimization loops.

Do not add JAXopt or Optax as mandatory dependencies.

### 6.6 Derivatives

Initial derivative claims should be narrow:

1. Pure-kernel derivatives are JAX differentiable.
2. Fixed-boundary solve-output derivatives are experimental until AD-vs-central-FD gates pass.
3. Optimization objectives may initially differentiate through a fixed-budget solve or use finite differences.
4. Promote implicit derivatives only after root residuals and Hessian/linear-solve gates pass.

This follows PR #20's conservative derivative-policy approach.

---

## 7. I/O design

### 7.1 Mirror input file

Add a mirror-native input format. TOML is recommended because Python 3.11 includes `tomllib`; for Python 3.10 the project already uses `tomli` optionally.

Example:

```toml
[mirror]
geometry_type = "mirror"
name = "fixed_flared_tube"
mode = "fixed_boundary"
axis = "straight"
z_min = -1.2
z_max = 1.2

[resolution]
ns = 17
ntheta = 1
nxi = 33
mpol = 0

[boundary]
type = "polynomial_radius"
r0 = 0.25
a2 = -0.35
a4 = 0.05
fix_end_surfaces = true

[flux]
psi_edge = 0.05
iprofile = "zero"

[pressure]
type = "polynomial"
gamma = 0.0
coeffs = [1000.0, -1000.0]

[solver]
optimizer = "lbfgs"
ftol = 1.0e-10
maxiter = 2000
pressure_continuation = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
```

Also allow Python API construction from dataclasses. Do not force mirror users to write classic VMEC `INDATA`, since `RBC/RBS/ZBC/ZBS` toroidal Fourier boundary coefficients do not represent an open mirror boundary.

### 7.2 Mirror output file: `mout_*.nc`

Use mirror-native NetCDF files:

```text
mout_<case>.nc
```

Global attributes:

```text
code = "vmec_jax"
geometry_type = "mirror"
mirror_schema_version = "0.1"
algorithm = "fixed_boundary_variational_chebyshev_lobatto"
coordinate_order = "s,theta,xi"
axis = "straight"
fixed_boundary = true
free_boundary = false
pressure_model = "scalar_p_of_s"
```

Dimensions:

```text
ns
ntheta
nxi
m_modes
k_modes          optional, once coefficient spectra are stored
history_steps
```

Coordinate variables:

```text
s(ns)
theta(ntheta)
xi(nxi)
z(nxi)
w_s(ns)
w_theta(ntheta)
w_xi(nxi)
```

Geometry variables:

```text
r(ns,ntheta,nxi)
X(ns,ntheta,nxi)
Y(ns,ntheta,nxi)
Z(ns,ntheta,nxi)
sqrtg(ns,ntheta,nxi)
g_ss, g_stheta, g_sxi, g_thetatheta, g_thetaxi, g_xixi
boundary_r(ntheta,nxi)
```

Field variables:

```text
B_sup_s
B_sup_theta
B_sup_xi
B_cov_s
B_cov_theta
B_cov_xi
B_x
B_y
B_z
Bmag
lambda
Psi_prime(ns)
I_prime(ns)
```

Profile variables:

```text
pressure(ns)
dpressure_ds(ns)
beta(ns or scalar)
```

Diagnostics:

```text
energy_B
energy_p
energy_total
residual_norm
force_norm
min_sqrtg
max_sqrtg
min_Bmag
max_Bmag
mirror_ratio
solve_history_* arrays
```

### 7.3 Compatibility with WOUT

Do not overload `wout_*.nc`. Provide explicit conversion/export helpers only for diagnostic comparison where possible:

```python
mirror_surface_to_vtk(...)
mirror_output_to_npz(...)
mirror_axisym_slice_to_csv(...)
```

Classic VMEC `booz_xform_jax` should reject `mout` files with a clear message and suggest `vmec --mirror-booz`.

---

## 8. Plotting and visualization

### 8.1 CLI behavior

Existing behavior:

```bash
vmec --plot wout_case.nc
```

New behavior:

```bash
vmec --plot mout_case.nc
```

Dispatch by NetCDF attribute `geometry_type`.

### 8.2 Required mirror plots

`vmec --plot mout_case.nc --outdir figures` should write:

```text
mirror_surfaces_rz.png          nested surfaces in r-z for axisymmetric cases
mirror_boundary_3d.png          3D side boundary when ntheta > 1
mirror_bmag_sxi.png             |B|(s,xi), axisymmetric or theta-averaged
mirror_bmag_boundary.png        |B|(theta,xi) on s=1
mirror_jacobian.png             sqrt(g), min/max diagnostics
mirror_pressure_profile.png     p(s)
mirror_residual_history.png     residual/energy history
mirror_spectral_content.png     optional Chebyshev/Fourier spectra
```

Plotting tests should check numerical arrays behind the plots, not only file existence.

### 8.3 VTK export

Add structured-grid export:

```python
write_mirror_vtk(mout, "case.vts")
```

or, if avoiding a VTK dependency, write `.npz` plus docs for ParaView conversion. Optional VTK dependency must be lazy.

---

## 9. Mirror-Boozer-like coordinates

Standard Boozer coordinates are toroidal closed-surface coordinates. Mirrors have open field lines and a nonperiodic axial coordinate, so this must be a separate transform.

### 9.1 Straight-field-line transform

On each surface, define a transformed angle:

\[
\vartheta=\theta+\Lambda_m(s,\theta,\xi),
\]

with a condition that field lines are straight in `(vartheta, xi)`:

\[
\frac{d\vartheta}{d\xi}=\frac{B^\theta}{B^\xi} + \partial_\xi\Lambda_m + \frac{B^\theta}{B^\xi}\partial_\theta\Lambda_m
\]

or solve a surface PDE of the form:

\[
B^\xi\partial_\xi\vartheta+B^\theta\partial_\theta\vartheta=C(s),
\]

with gauge and endpoint conditions replacing toroidal periodicity in `xi`.

First tests:

1. Straight cylinder with `Btheta = 0` gives `vartheta = theta`.
2. Constant-pitch cylinder gives linear angle advance.
3. Axisymmetric mirror gives only `m=0` content in `|B|`.

### 9.2 Mirror-Boozer output

Use:

```text
mbmn_<case>.nc
```

Variables:

```text
geometry_type = "mirror_boozer"
s
theta_b
xi
Bmag(theta_b,xi,s)
Bmag_mk(s,m,k)
mirror_ratio(s)
well_depth(s)
bounce_points optional
```

---

## 10. Optimization integration

### 10.1 Mirror objectives

Initial objectives:

1. Target mirror ratio:

\[
R_m(s)=\frac{\max_\xi |B(s,\xi)|}{\min_\xi |B(s,\xi)|}.
\]

2. Target axial well shape:

\[
|B(s,\xi)|\approx B_{target}(s,\xi).
\]

3. Minimum Jacobian barrier:

\[
\min J > J_{floor}.
\]

4. Boundary smoothness:

\[
\sum_k k^p |r_{b,k}|^2.
\]

5. Endpoint radius and wall-clearance constraints.

6. Axis regularity and no-negative-radius constraints.

### 10.2 Optimization workflow

Use a Simsopt-like workflow consistent with current `optimization_workflow.py` style:

```python
boundary = MirrorBoundary.polynomial_radius(r0=..., a2=..., a4=...)
vmec = FixedBoundaryMirror(config, boundary)
problem = LeastSquaresProblem.from_terms([
    MirrorRatio(target=8.0, weight=1.0),
    MinJacobian(floor=1e-5, weight=10.0),
])
result = least_squares_solve(vmec, problem)
result.save("results/mirror_ratio_opt")
```

Keep example scripts explicit and short. Do not hide all details behind opaque one-liners.

---

## 11. Free-boundary path, deferred

Free boundary should follow the VMEC energy-principle idea, not Green's functions.

Future free-boundary model:

1. Plasma domain: `0 <= s <= 1`.
2. Vacuum domain: `1 <= sigma <= sigma_wall`.
3. Wall boundary: prescribed conducting or physical wall surface.
4. External field: coil field provider, mgrid-like field, or imported vacuum field.
5. Vacuum correction potential:

\[
\mathbf B_v=\mathbf B_{ext}+\nabla\nu.
\]

6. Vacuum energy:

\[
W_v=\int_{\Omega_v}\frac{|\mathbf B_v|^2}{2\mu_0}\,dV.
\]

7. Total energy:

\[
W=W_p+W_B+W_v.
\]

The free-boundary solve varies the plasma boundary, interior geometry, `lambda`, and vacuum potential `nu`, subject to interface and wall conditions.

This is deliberately deferred until fixed-boundary geometry, fields, residuals, I/O, plotting, and optimization are trusted.

---

## 12. Test plan overview

Tests must be real scientific/numerical tests, not scaffolds. The test suite should be layered:

```text
unit math tests
geometry and identity tests
energy/gradient tests
manufactured-solution tests
fixed-boundary solver tests
literature-anchored WHAM tests
code-to-code parity tests
plotting/CLI/I/O tests
Boozer-like transform tests
optimization tests
regression and benchmark tests
```

Suggested directory structure:

```text
tests/mirror/
  test_chebyshev_lobatto.py
  test_theta_fourier_basis.py
  test_mirror_grids.py
  test_mirror_geometry_axisym.py
  test_mirror_geometry_3d.py
  test_mirror_field_identities.py
  test_mirror_energy.py
  test_mirror_forces_gradients.py
  test_mirror_manufactured_solutions.py
  test_mirror_fixed_boundary_axisym.py
  test_mirror_fixed_boundary_3d.py
  test_mirror_io.py
  test_mirror_plotting.py
  test_mirror_boozer.py
  test_mirror_optimization.py
  test_mirror_convergence.py
  test_mirror_performance.py
  test_wham_magpylib_regression.py
  test_desc_fourier_chebyshev_parity.py
  test_optional_pleiades_parity.py

validation/mirror/
  wham_coils.json
  manufactured_cases.json
  reference_mout_cylinder.nc or compressed diagnostics
  reference_mout_flared_finite_beta.nc or compressed diagnostics
```

Add pytest markers:

```toml
"mirror: tests for open-ended mirror geometry backend"
"magpylib: optional tests requiring magpylib"
"pleiades: optional tests requiring Pleiades"
"desc: optional tests comparing against DESC"
```

Core tests should not require optional packages.

---

## 13. Unit and numerical identity tests

### 13.1 Chebyshev--Lobatto tests

1. **Node endpoint and ordering test**
   - Verify public nodes are monotone increasing.
   - Verify first and last nodes are exactly `-1` and `+1` within roundoff.
   - Verify internal and public order permutations are inverse-consistent.
   - This test is directly motivated by DESC PR #2194.

2. **First derivative exactness**
   - For `q = 0, ..., N`, test:
     \[
     D_\xi \xi^q = q\xi^{q-1}.
     \]
   - Include endpoints.

3. **Second derivative exactness**
   - Test:
     \[
     D_\xi^2 \xi^q = q(q-1)\xi^{q-2}.
     \]

4. **Clenshaw--Curtis quadrature for monomials**
   - Test:
     \[
     \int_{-1}^{1}\xi^q d\xi =
     \begin{cases}
     2/(q+1), & q\;\text{even},\\
     0, & q\;\text{odd}.
     \end{cases}
     \]

5. **Spectral convergence**
   - Test interpolation and derivative convergence for:
     \[
     e^\xi,
     \qquad \cos(3\xi),
     \qquad \frac{1}{1+0.2\xi^2}.
     \]
   - Require exponential-like convergence for smooth functions.

6. **Resolution-change test**
   - Interpolate from `Nxi=16` to `Nxi=32` and back for analytic functions.
   - Verify errors match spectral expectations.

7. **Filter test**
   - Project a mixed low/high Chebyshev signal.
   - Apply filter.
   - Verify high-mode norm decreases and low modes remain unchanged within tolerance.

### 13.2 Fourier theta tests

1. **Fourier derivative exactness**
   - For:
     \[
     f(\theta,\xi)=\cos(m\theta)T_k(\xi),
     \]
     verify `dtheta` and `dxi` exactly.

2. **Orthogonality / Parseval test**
   - Verify resolved quadrature norms:
     \[
     \int_0^{2\pi}\cos^2(m\theta)d\theta=\pi,
     \qquad m>0.
     \]

3. **FFT parity test**
   - If FFT path and direct matrix path both exist, compare transforms for random resolved coefficients.

4. **Aliasing test**
   - Show unresolved modes alias as expected and oversampling prevents contamination in nonlinear products.

---

## 14. Geometry and metric tests

### 14.1 Straight cylinder

Use:

\[
r(s,\xi)=a\sqrt{s},\qquad z=L\xi.
\]

Exact:

\[
J=\frac{a^2L}{2},\qquad
V=2\pi L a^2.
\]

Verify:

- `sqrtg`.
- metric entries.
- physical volume.
- no negative Jacobian.
- endpoint handling.
- axis treatment.

### 14.2 Flared polynomial tube

Use:

\[
r(s,\xi)=\sqrt{s}\,a(1+\epsilon\xi^2),\qquad z=L\xi.
\]

Exact quantities:

\[
r_s=\frac{a(1+\epsilon\xi^2)}{2\sqrt{s}},
\]

\[
r_\xi=2a\epsilon\xi\sqrt{s},
\]

\[
J=r r_s L.
\]

Verify metric, Jacobian, and volume:

\[
V=\pi L\int_{-1}^{1}a^2(1+\epsilon\xi^2)^2 d\xi.
\]

### 14.3 Nonaxisymmetric boundary

Use:

\[
r_b(\theta,\xi)=r_0(1+a_2\xi^2)(1+\epsilon\cos 2\theta).
\]

Verify:

- side boundary exactness;
- periodic theta closure;
- metric finite positive for small `epsilon`;
- symmetry identities.

---

## 15. Field identity and energy tests

### 15.1 Discrete divergence-free identity

For random smooth `lambda`:

\[
J B^\theta=I'(s)-\lambda_\xi,
\qquad
J B^\xi=\Psi'(s)+\lambda_\theta.
\]

Verify:

\[
\partial_\theta(JB^\theta)+\partial_\xi(JB^\xi)=0.
\]

This is a required core test.

### 15.2 Lambda gauge invariance

Add `c(s)` to `lambda`. Verify no change in:

- `B^theta`;
- `B^xi`;
- `Bmag`;
- `W_B`;
- residuals.

### 15.3 Constant axial field in a cylinder

For `r = a sqrt(s)`, `z = L xi`, `lambda = 0`, `I' = 0`, choose `Psi'(s)` such that `B_z = B0`. Verify:

\[
\mathbf B=B_0\hat{\mathbf z},
\]

\[
W_B=\frac{B_0^2}{2\mu_0}V.
\]

### 15.4 Pressure energy analytic test

For cylinder and:

\[
p(s)=p_0(1-s),
\]

verify:

\[
W_p=\frac{1}{\gamma-1}\int p(s)J\,ds\,d\theta\,d\xi.
\]

### 15.5 Slender mirror ratio test

For a slender tube with slowly varying area:

\[
A(\xi)\propto r_b(\xi)^2,
\]

check:

\[
B(\xi)\propto \frac{1}{A(\xi)}.
\]

Verify computed mirror ratio agrees with the area-ratio prediction in the small-slope limit.

### 15.6 Gradient check

For a small state, compare JAX gradients of `W` with central finite differences for selected degrees of freedom.

### 15.7 Hessian symmetry

For small states, test:

\[
u^T H v = v^T H u.
\]

This is important because the solver is variational.

---

## 16. Method of manufactured solutions (MMS)

MMS should be a first-class validation lane. The goal is to verify geometry, field, energy, residual, constraints, and solver behavior against exact smooth solutions that are not limited to trivial cylinders.

### 16.1 General MMS principle

Choose smooth exact fields:

\[
a_*(s,\theta,\xi),\qquad \lambda_*(s,\theta,\xi),
\]

that satisfy boundary and gauge constraints. Compute the residual that the unforced ideal-MHD energy would produce:

\[
R_a^*=\left.\frac{\delta W}{\delta a}\right|_{a_*,\lambda_*},
\qquad
R_\lambda^*=\left.\frac{\delta W}{\delta\lambda}\right|_{a_*,\lambda_*}.
\]

Define a manufactured objective:

\[
W_{MMS}=W-\langle R_a^*, a\rangle - \langle R_\lambda^*, \lambda\rangle.
\]

Then the exact manufactured solution is stationary:

\[
\left.\frac{\delta W_{MMS}}{\delta a}\right|_{a_*,\lambda_*}=0,
\qquad
\left.\frac{\delta W_{MMS}}{\delta\lambda}\right|_{a_*,\lambda_*}=0.
\]

This approach is natural in a variational JAX code because source terms can be generated with automatic differentiation at the exact manufactured state.

### 16.2 Axisymmetric MMS cases

Case A: polynomial flared tube.

\[
a_*(s,\xi)=a_0(1+\epsilon\xi^2)(1+\alpha s(1-s)(1-\xi^2)).
\]

\[
\lambda_*(s,\xi)=0.
\]

Case B: nonzero lambda.

\[
a_*(s,\xi)=a_0(1+\epsilon\xi^2),
\]

\[
\lambda_*(s,\xi)=\lambda_0 s(1-s)(1-\xi^2)\xi.
\]

Case C: finite pressure.

\[
p(s)=p_0(1-s)^2.
\]

### 16.3 3D MMS cases

Use:

\[
a_*(s,\theta,\xi)=a_0(1+\epsilon\xi^2)
\left[1+\delta s(1-s)(1-\xi^2)\cos(2\theta)\right].
\]

\[
\lambda_*(s,\theta,\xi)=\lambda_0s(1-s)(1-\xi^2)\sin(\theta)T_3(\xi).
\]

Tests:

1. exact state has residual equal to zero in manufactured problem;
2. perturbed initial condition converges back to exact solution;
3. convergence rate improves with `Nxi`, `Ntheta`, `ns`;
4. finite-difference and JAX-generated manufactured sources agree.

### 16.4 MMS implementation files

```text
vmec_jax/mirror/kernels/manufactured.py
vmec_jax/mirror/validation/manufactured.py
tests/mirror/test_mirror_manufactured_solutions.py
```

Public helper:

```python
make_mms_case(name: str, resolution: MirrorResolution) -> ManufacturedMirrorCase
```

MMS artifacts should store exact solution formulas, parameters, and reference residuals in JSON or small NetCDF files.

---

## 17. Fixed-boundary solver tests

### 17.1 Zero-pressure cylinder stationarity

- Boundary: cylinder.
- Pressure: zero.
- Initial state: exact cylinder plus small admissible perturbation.
- Expected: solver returns to cylinder; residual norm reaches tolerance; energy decreases.

### 17.2 Zero-pressure flared tube relaxation

- Boundary: flared polynomial tube.
- Pressure: zero.
- Initial state: perturbed nested surfaces.
- Expected: no negative Jacobian; residual decreases; solution smooth.

### 17.3 Finite-pressure continuation

- Boundary: flared tube.
- Pressure: `p0(1-s)`.
- Continuation: `[0, 0.1, 0.25, 0.5, 0.75, 1]`.
- Expected: each stage converges or improves residual; geometry remains finite-positive; final state reproducible.

### 17.4 Resolution convergence

Solve the same smooth case at increasing resolution:

```text
(ns, nxi, mpol) = (9, 17, 0), (17, 33, 0), (33, 65, 0)
```

Expected:

- energy converges;
- residual converges;
- `Bmag` profile converges;
- Chebyshev spectra decay rapidly for smooth cases.

### 17.5 Nonaxisymmetric perturbation decay

- Axisymmetric boundary.
- Initial condition with small `m=1` or `m=2` perturbation.
- Expected: perturbation decays if not supported by boundary or physics.

### 17.6 Fixed nonaxisymmetric boundary

- Boundary: `m=2` ellipticity varying in `xi`.
- Expected: boundary exactly preserved; interior surfaces smooth; residual finite and convergent.

---

## 18. WHAM and literature-anchored tests

### 18.1 WHAM coil fixture

Convert the uploaded `coil_model_WHAM-2.txt` script to a fixture without requiring plotting at test time.

Fixture metadata:

```json
{
  "source": "coil_model_WHAM-2.txt",
  "reference": "Radovinsky et al., Design of High Field HTS Coils for Magnetic Mirror, IEEE TAS 2023",
  "coil_centers_z_m": [-0.98, 0.98],
  "nz": 8,
  "nr": 310,
  "dz_HF_m": 0.1144,
  "r_in_HF_m": 0.043,
  "r_out_HF_m": 0.365,
  "I_coil_A": 2000 * 17.0 / 17.51
}
```

Tests:

1. Build the coil fixture deterministically.
2. If `magpylib` is installed, compare pure-JAX or project coil evaluator to `magpylib` for:
   - `B_z(0,z)`;
   - `B_r(r,z)`;
   - `B_z(r,z)`;
   - `|B|(r,z)`;
   - on-axis mirror ratio.
3. Save a small reference table for core CI so tests do not require `magpylib`.
4. Verify symmetry about `z=0`.
5. Verify coil centers match WHAM physics-basis article values `z = +/- 98 cm`.

### 18.2 Boundary-from-vacuum-flux initialization

Use the WHAM vacuum field only to generate initial fixed boundaries:

```python
boundary = mirror_boundary_from_vacuum_flux_tube(wham_field, psi_value, z_grid)
```

Tests:

1. boundary is positive;
2. boundary is symmetric in `xi` for symmetric coils;
3. interpolation is resolution-consistent;
4. smoothing/filtering preserves endpoints;
5. generated `r_b(xi)` produces expected approximate mirror-ratio trend.

### 18.3 WHAM finite-beta qualitative test

Using a WHAM-inspired fixed boundary and scalar pressure:

1. solve low beta;
2. solve higher beta;
3. verify diamagnetic response changes `Bmag` and mirror-ratio diagnostics smoothly;
4. do not claim WHAM experimental predictive validity in this scalar-pressure fixed-boundary phase.

### 18.4 Future anisotropic anchor

Add a skipped or xfailed design test for anisotropic closure API:

```python
closure = AnisotropicMirrorClosure(p_parallel=lambda s, B: ..., p_perp=lambda s, B: ...)
```

This protects the API shape for later WHAM-relevant anisotropy without claiming implementation now.

---

## 19. Code-to-code parity tests

### 19.1 DESC Fourier-Chebyshev scalar parity

Optional `@pytest.mark.desc` test:

1. Create scalar fields represented with Fourier in `theta` and Chebyshev in `xi`.
2. Evaluate with `vmec_jax.mirror` transforms.
3. Evaluate equivalent expressions with DESC basis utilities where applicable.
4. Compare values and derivatives.

Use DESC source/docs as the basis implementation reference, not as an equilibrium comparison.

### 19.2 DESC PR #2012 compatibility-style test

If PR #2012 or a similar feature lands in DESC, add a parity test for build/fit/evaluate semantics:

```python
field = FourierChebyshevField(...)
field.build(eq_or_grid)
fit = field.fit(...)
field.evaluate(...)
```

For `vmec_jax`, mirror field interpolation should expose similar build/evaluate separations.

### 19.3 Pleiades optional comparisons

Optional `@pytest.mark.pleiades`:

1. Compare vacuum coil fields for WHAM-like coils.
2. Compare axisymmetric field-line/flux-tube shape diagnostics.
3. Compare simple scalar-pressure or anisotropic literature cases only as validation data, not as the method to be implemented.

### 19.4 GVEC mapping-inspired tests

Optional mapping tests:

1. Verify mirror geometry kernels can be expressed as a general mapping, not only hard-coded `r,z`.
2. Use simple generalized coordinate maps to compare metric/Jacobian identities.
3. Keep full GVEC code comparison optional.

### 19.5 Toroidal-limit sanity check

Construct a long weakly varying tube and compare local metric/field pieces with a large-aspect-ratio toroidal `vmec_jax` case away from the artificial seam. This is a sanity check only and must not be described as topological equivalence.

---

## 20. Boozer-like tests

1. **Straight cylinder**
   - Constant axial field.
   - Expected mirror-Boozer transform is identity up to gauge.

2. **Constant-pitch cylinder**
   - Set `I'(s) != 0`, `lambda = 0`.
   - Expected field-line pitch is analytic.

3. **Axisymmetric mirror**
   - `|B|` in mirror-Boozer-like coordinates should have only `m=0` Fourier content.

4. **Mirror ratio consistency**
   - Compute mirror ratio from physical grid and from `mbmn` output.
   - Results must agree.

5. **Endpoint behavior**
   - Verify field-line integration reaches expected end plane and does not wrap around.

---

## 21. Plotting, CLI, and documentation tests

### 21.1 CLI tests

1. `vmec mirror examples/mirror/input.fixed_cylinder.toml` produces `mout_fixed_cylinder.nc`.
2. `vmec --plot mout_fixed_cylinder.nc` generates expected figures.
3. `vmec --plot wout_existing.nc` remains unchanged.
4. `vmec --mirror-booz mout_fixed_cylinder.nc` writes `mbmn_fixed_cylinder.nc` after that feature lands.
5. `booz_xform_jax` path rejects `mout` with a helpful error.

### 21.2 Plot content tests

Do not only check files exist. Extract arrays used for plotting and test:

1. boundary curve equals stored `r_b`;
2. symmetric input produces symmetric plots;
3. `Bmax >= Bmin > 0`;
4. no negative radius;
5. `min(sqrtg)` in plot equals stored diagnostic;
6. residual history matches solver trace.

### 21.3 Docs build tests

Update docs and run:

```bash
SPHINX_FAST=1 LC_ALL=C.UTF-8 LANG=C.UTF-8 python -m sphinx -W -j auto -b html docs docs/_build/html
```

Add normal docs build in optional docs CI.

---

## 22. Optimization tests

### 22.1 Boundary-parameter gradient test

Parameterize:

\[
r_b(\xi;p)=r_0[1+p(1-\xi^2)].
\]

Run a tiny solve and compare JAX or implicit gradient of mirror ratio with central finite differences.

### 22.2 Target mirror-ratio optimization

Use:

\[
r_b(\xi)=r_0(1+a\xi^2+b\xi^4).
\]

Optimize `a,b` to hit target `R_m`. Verify:

1. objective decreases;
2. final mirror ratio within tolerance;
3. boundary remains positive;
4. solver artifacts reproduce results.

### 22.3 Pressure sensitivity

Differentiate final diagnostics with respect to `p0`. Compare with central finite differences under fixed continuation policy.

---

## 23. Regression tests

Create deterministic compact baselines for:

1. zero-pressure cylinder;
2. zero-pressure flared tube;
3. finite-pressure flared tube;
4. WHAM-inspired fixed boundary;
5. nonaxisymmetric fixed boundary;
6. manufactured 3D case.

Store compressed reference diagnostics instead of large NetCDF when possible:

```json
{
  "volume": ...,
  "energy_B": ...,
  "energy_p": ...,
  "min_jacobian": ...,
  "max_B": ...,
  "min_B": ...,
  "mirror_ratio": ...,
  "residual_final": ...,
  "B_axis_sample": [...],
  "r_boundary_sample": [...]
}
```

NetCDF reference files should be small and only used where schema/I/O regression is being tested.

---

## 24. Performance and benchmark tests

### 24.1 Transform scaling

Benchmark:

```text
Ntheta = 1, 8, 16, 32
Nxi = 17, 33, 65, 129
ns = 9, 17, 33
```

Track:

1. CGL derivative matrix construction time;
2. geometry kernel execution time;
3. field kernel execution time;
4. energy and residual time;
5. gradient time;
6. JIT compile vs warm execution time.

### 24.2 Allocation discipline

Add tests or diagnostics to catch accidental dense tensors such as:

```text
(ns * ntheta * nxi, nmodes_theta * nmodes_xi)
```

for production-sized cases. Dense Vandermonde is okay for tiny tests, not for production paths.

### 24.3 JIT cache stability

Call the same kernel/solve twice. The second call should not rebuild static matrices or trigger unexpected recompilation for identical static shapes.

### 24.4 CI strategy

Core CI should run:

1. CGL unit tests;
2. geometry analytic tests;
3. divergence-free identity;
4. energy analytic tests;
5. tiny fixed-boundary solve;
6. I/O schema test;
7. one plot-data test.

Optional markers should cover large convergence, WHAM/magpylib, Pleiades, DESC, and GPU benchmarks.

---

## 25. Step-by-step implementation plan

### PR M0: plan and documentation skeleton

Files:

```text
plan_mirror.md
docs/mirror/index.rst
docs/mirror/overview.rst
```

Tasks:

1. Commit this plan.
2. Add docs stub linking to this plan but marking feature as planned/experimental.
3. Add no runtime behavior yet.

Validation:

```bash
python -m ruff check docs/conf.py
SPHINX_FAST=1 LC_ALL=C.UTF-8 LANG=C.UTF-8 python -m sphinx -W -j auto -b html docs docs/_build/html
```

### PR M1: mirror package skeleton and Chebyshev core

Files:

```text
vmec_jax/mirror/__init__.py
vmec_jax/mirror/api.py
vmec_jax/mirror/core/grids.py
vmec_jax/mirror/core/basis.py
vmec_jax/mirror/kernels/chebyshev.py
vmec_jax/mirror/kernels/fourier.py
tests/mirror/test_chebyshev_lobatto.py
tests/mirror/test_theta_fourier_basis.py
```

Tasks:

1. Add CGL nodes in increasing public order.
2. Add first derivative matrix.
3. Add Clenshaw--Curtis weights.
4. Add Fourier theta basis and derivatives.
5. Add cached `MirrorGrid`.
6. Add exactness, quadrature, and convergence tests.

### PR M2: axisymmetric geometry kernels

Files:

```text
vmec_jax/mirror/core/state.py
vmec_jax/mirror/core/boundary.py
vmec_jax/mirror/kernels/geometry.py
vmec_jax/mirror/kernels/constraints.py
tests/mirror/test_mirror_geometry_axisym.py
```

Tasks:

1. Add `MirrorStateAxisym`.
2. Add `MirrorBoundary` for cylinder and polynomial/flared boundaries.
3. Add geometry evaluation.
4. Add metrics and Jacobian.
5. Add fixed boundary and endpoint projection.
6. Add cylinder/flared analytic tests.

### PR M3: magnetic field and energy

Files:

```text
vmec_jax/mirror/core/profiles.py
vmec_jax/mirror/kernels/fields.py
vmec_jax/mirror/kernels/energy.py
vmec_jax/mirror/kernels/residuals.py
tests/mirror/test_mirror_field_identities.py
tests/mirror/test_mirror_energy.py
```

Tasks:

1. Add `PsiPrimeProfile`, `IPrimeProfile`, scalar `PressureProfile`.
2. Add contravariant `B` kernels.
3. Add `Bmag`, covariant/cartesian components.
4. Add energy integrals.
5. Add divergence-free, gauge-invariance, constant-field, and analytic energy tests.

### PR M4: variational gradients and MMS

Files:

```text
vmec_jax/mirror/kernels/forces.py
vmec_jax/mirror/kernels/manufactured.py
vmec_jax/mirror/validation/manufactured.py
tests/mirror/test_mirror_forces_gradients.py
tests/mirror/test_mirror_manufactured_solutions.py
```

Tasks:

1. Add AD gradient wrappers for energy.
2. Add residual projection.
3. Add MMS source-term builder.
4. Add gradient and Hessian symmetry tests.
5. Add MMS exact-solution residual and convergence tests.

### PR M5: fixed-boundary solver

Files:

```text
vmec_jax/mirror/solvers/fixed_boundary/api.py
vmec_jax/mirror/solvers/fixed_boundary/continuation.py
vmec_jax/mirror/solvers/fixed_boundary/nonlinear.py
vmec_jax/mirror/solvers/fixed_boundary/optimizers.py
vmec_jax/mirror/solvers/fixed_boundary/diagnostics.py
tests/mirror/test_mirror_fixed_boundary_axisym.py
tests/mirror/test_mirror_convergence.py
```

Tasks:

1. Add `run_mirror_fixed_boundary`.
2. Add LBFGS/GD solve path.
3. Add pressure continuation.
4. Add resolution continuation.
5. Add trace diagnostics.
6. Add cylinder, flared, finite-pressure, and convergence tests.

### PR M6: I/O and plotting

Files:

```text
vmec_jax/mirror/io/schema.py
vmec_jax/mirror/io/mout.py
vmec_jax/mirror/plotting/geometry.py
vmec_jax/mirror/plotting/bfield.py
vmec_jax/mirror/plotting/diagnostics.py
vmec_jax/mirror/plotting/export.py
tests/mirror/test_mirror_io.py
tests/mirror/test_mirror_plotting.py
```

Tasks:

1. Add `mout_*.nc` schema.
2. Add read/write roundtrip.
3. Add plot-data helpers and PNG writing.
4. Add CLI dispatch for `vmec --plot mout_*.nc`.
5. Add I/O and plot numerical-content tests.

### PR M7: WHAM fixture and examples

Files:

```text
vmec_jax/mirror/validation/wham.py
validation/mirror/wham_coils.json
examples/mirror/fixed_cylinder.py
examples/mirror/fixed_flared_tube.py
examples/mirror/wham_vacuum_boundary.py
tests/mirror/test_wham_magpylib_regression.py
```

Tasks:

1. Convert uploaded `coil_model_WHAM-2.txt` into metadata and reusable fixture.
2. Add optional `magpylib` comparison.
3. Add pure stored reference arrays for core CI.
4. Add examples.

### PR M8: nonaxisymmetric fixed-boundary surfaces

Files:

```text
vmec_jax/mirror/core/state.py
vmec_jax/mirror/kernels/geometry.py
vmec_jax/mirror/kernels/fourier.py
tests/mirror/test_mirror_geometry_3d.py
tests/mirror/test_mirror_fixed_boundary_3d.py
```

Tasks:

1. Generalize axisymmetric state to nodal `theta` dependence.
2. Enforce periodic theta and side boundary.
3. Add `m>0` tests.
4. Add nonaxisymmetric fixed-boundary solve.

### PR M9: mirror-Boozer-like transform

Files:

```text
vmec_jax/mirror/boozer/transform.py
vmec_jax/mirror/boozer/spectra.py
vmec_jax/mirror/boozer/fieldlines.py
vmec_jax/mirror/boozer/io.py
vmec_jax/mirror/plotting/boozer.py
tests/mirror/test_mirror_boozer.py
```

Tasks:

1. Implement straight-field-line transform.
2. Add `mbmn_*.nc` output.
3. Add identity and constant-pitch tests.
4. Add mirror-ratio consistency tests.

### PR M10: mirror optimization

Files:

```text
vmec_jax/mirror/objectives/*.py
vmec_jax/mirror/optimization/*.py
examples/mirror/optimize_target_mirror_ratio.py
tests/mirror/test_mirror_optimization.py
```

Tasks:

1. Add mirror-ratio and B-well objectives.
2. Add boundary parameter spaces.
3. Add least-squares workflow.
4. Add gradient and target-ratio optimization tests.

### PR M11: fixed-boundary documentation completion

Files:

```text
docs/mirror/index.rst
docs/mirror/quickstart.rst
docs/mirror/inputs.rst
docs/mirror/outputs.rst
docs/mirror/theory.rst
docs/mirror/algorithms.rst
docs/mirror/testing.rst
docs/mirror/examples.rst
docs/mirror/optimization.rst
docs/mirror/boozer.rst
docs/mirror/validation.rst
docs/index.rst
docs/references.rst
docs/code_structure.rst
docs/testing_strategy.rst
```

Tasks:

1. Fully document fixed-boundary mirror input/output.
2. Include equations and derivations.
3. Include limitations.
4. Include examples and plotting.
5. Include test philosophy and validation table.
6. Build docs with `-W`.

### PR M12: free-boundary design skeleton only

Files:

```text
vmec_jax/mirror/solvers/free_boundary/*.py
docs/mirror/free_boundary.rst
tests/mirror/test_mirror_free_boundary_interfaces.py
```

Tasks:

1. Add interfaces and configuration only.
2. No production free-boundary claim.
3. Document PDE/spectral potential path.
4. Add shape/unit tests only.

---

## 26. Documentation plan

### 26.1 Add docs pages

Add a new docs subtree:

```text
docs/mirror/index.rst
docs/mirror/overview.rst
docs/mirror/quickstart.rst
docs/mirror/inputs.rst
docs/mirror/outputs.rst
docs/mirror/theory.rst
docs/mirror/discretization.rst
docs/mirror/algorithms.rst
docs/mirror/fixed_boundary.rst
docs/mirror/free_boundary_future.rst
docs/mirror/boozer.rst
docs/mirror/optimization.rst
docs/mirror/validation.rst
docs/mirror/testing.rst
docs/mirror/examples.rst
docs/mirror/api.rst
```

Update `docs/index.rst` under Physics and algorithms and User guide:

```rst
.. toctree::
   :maxdepth: 2
   :caption: Mirror geometries

   mirror/index
```

For fast docs builds, keep mirror API autosummary optional if it slows CI.

### 26.2 `docs/mirror/overview.rst`

Explain:

1. what a mirror geometry is;
2. why toroidal Fourier boundary representations do not apply;
3. what fixed boundary means for an open flux tube;
4. what is implemented now and what is deferred.

Include diagrams:

- coordinate sketch `(s, theta, xi)`;
- side boundary and open end planes;
- nested flux tubes in `r-z`.

### 26.3 `docs/mirror/theory.rst`

Include derivations:

1. coordinate mapping;
2. metric and Jacobian;
3. divergence-free `B` representation;
4. energy functional;
5. pressure model;
6. gauge fixing;
7. fixed-boundary variational conditions;
8. relation to VMEC toroidal formulation;
9. why Green's functions are not the equilibrium method here.

### 26.4 `docs/mirror/discretization.rst`

Explain:

1. radial grid;
2. Fourier theta basis;
3. Chebyshev--Gauss--Lobatto axial grid;
4. derivative matrices;
5. Clenshaw--Curtis quadrature;
6. resolution change;
7. filtering and aliasing;
8. node ordering conventions.

Include a warning based on DESC PR #2194: CGL node ordering and resolution-change tests are critical.

### 26.5 `docs/mirror/inputs.rst`

Document TOML schema:

- `[mirror]`
- `[resolution]`
- `[boundary]`
- `[flux]`
- `[pressure]`
- `[solver]`
- `[output]`

Include full examples for cylinder, flared tube, and WHAM-inspired boundary.

### 26.6 `docs/mirror/outputs.rst`

Document `mout_*.nc` schema with:

1. dimensions;
2. variables;
3. units;
4. coordinate order;
5. field component conventions;
6. how it differs from `wout_*.nc`;
7. how to plot and inspect.

### 26.7 `docs/mirror/algorithms.rst`

Document:

1. fixed-boundary solve flow;
2. continuation;
3. optimizer choices;
4. constraints and projections;
5. diagnostics;
6. derivative policies;
7. failure modes and remedies.

### 26.8 `docs/mirror/validation.rst`

Include tables:

| Test family | Purpose | Required? | Source anchor | Tolerance |
|---|---|---:|---|---:|
| CGL exactness | derivative/quadrature correctness | yes | Trefethen/DESC PR #2194 | near roundoff |
| cylinder energy | analytic energy | yes | analytic | 1e-10 small case |
| divergence-free identity | field representation | yes | VMEC-style contravariant B | 1e-10--1e-12 |
| MMS | residual/solver correctness | yes | MMS methodology | convergence-based |
| WHAM vacuum | fixture/parity | optional/core reduced | WHAM script, WHAM paper | documented |
| DESC transform parity | transform parity | optional | DESC basis | documented |
| Pleiades parity | mirror code comparison | optional | Pleiades | documented |

### 26.9 `docs/mirror/testing.rst`

Explain how to run:

```bash
pytest tests/mirror -q
pytest tests/mirror -m "not magpylib and not pleiades and not desc" -q
pytest tests/mirror -m magpylib -q
pytest tests/mirror -m desc -q
pytest tests/mirror -m full -q
```

### 26.10 `docs/mirror/examples.rst`

Link examples:

```text
examples/mirror/fixed_cylinder.py
examples/mirror/fixed_flared_tube.py
examples/mirror/wham_vacuum_boundary.py
examples/mirror/nonaxisymmetric_boundary.py
examples/mirror/optimize_target_mirror_ratio.py
```

### 26.11 `docs/code_structure.rst`

Update the code structure page to include the mirror package and to state:

- mirror implementation lives under `vmec_jax/mirror/`;
- new mirror code follows PR #20 package boundaries;
- mirror I/O uses `mout`, not `wout`;
- mirror plotting dispatches from `--plot` by output schema.

### 26.12 `docs/references.rst`

Add references to:

- VMEC/STELLOPT docs;
- DESC docs/source;
- DESC PR #2012, #2194, #1508, #1893;
- GVEC docs/JOSS paper;
- WHAM physics-basis paper;
- WHAM HTS coil design paper from uploaded script;
- Pleiades docs;
- VMEC++ numerics;
- spectral methods references;
- MMS references.

---

## 27. Example files to add

```text
examples/mirror/README.md
examples/mirror/fixed_cylinder.py
examples/mirror/fixed_flared_tube.py
examples/mirror/wham_vacuum_boundary.py
examples/mirror/nonaxisymmetric_boundary.py
examples/mirror/optimize_target_mirror_ratio.py
examples/mirror/inputs/fixed_cylinder.toml
examples/mirror/inputs/fixed_flared_tube.toml
examples/mirror/inputs/wham_vacuum_boundary.toml
examples/mirror/inputs/nonaxisymmetric_boundary.toml
examples/mirror/inputs/optimize_target_mirror_ratio.toml
```

Each example should:

1. be runnable from a source checkout;
2. save output into a user-selected directory;
3. run at low resolution by default;
4. show how to increase resolution;
5. generate plots;
6. mention limitations.

---

## 28. Acceptance criteria for fixed-boundary mirror MVP

The feature should not be considered complete until all of these pass:

1. CGL derivative and quadrature tests pass.
2. Fourier-Chebyshev tensor derivative tests pass.
3. Cylinder/flared geometry analytic tests pass.
4. Divergence-free `B` identity passes.
5. Constant-field energy test passes.
6. Pressure-energy analytic test passes.
7. JAX gradient vs central finite difference passes for small states.
8. Hessian symmetry passes for small states.
9. MMS residual and convergence tests pass.
10. Zero-pressure cylinder fixed-boundary solve converges.
11. Finite-pressure flared tube fixed-boundary solve converges with continuation.
12. `mout_*.nc` read/write roundtrip passes.
13. `vmec --plot mout_*.nc` produces numerical-content-validated plots.
14. WHAM fixture metadata and core reference-field tests pass.
15. Documentation builds with Sphinx `-W` in fast mode.
16. No new root-level helper modules are added in violation of PR #20 naming rules.
17. `import vmec_jax` remains fast and does not import optional mirror validation dependencies.

---

## 29. Known risks and mitigations

### Risk: Chebyshev ringing near sharp expanders or walls

Mitigation:

- start with smooth fixed-boundary examples;
- include spectral filters;
- include multi-domain Chebyshev or B-spline abstraction as a planned second phase;
- inspect Chebyshev spectra in plots and tests.

### Risk: axis singularity

Mitigation:

- first axisymmetric state stores `a` with `r = sqrt(s) a`;
- do not evaluate singular `r_s` at the axis naively;
- add dedicated axis regularity tests;
- later add Zernike-like or vector-basis regularization for 3D.

### Risk: open-end boundary conditions are physically ambiguous

Mitigation:

- make end policy explicit in inputs and docs;
- start with fixed end cross-sections;
- add natural end conditions only after tests validate them;
- do not overclaim end-loss physics.

### Risk: Boozer terminology confusion

Mitigation:

- call it `mirror-Boozer-like` or `mirror straight-field-line` in docs;
- do not run toroidal `booz_xform_jax` on `mout`;
- document open-line differences clearly.

### Risk: free-boundary scope creep

Mitigation:

- do fixed boundary first;
- free-boundary files can exist as interface skeletons only after MVP;
- no Green's-function equilibrium method;
- future free boundary uses vacuum-region PDE/spectral potential and energy principle.

### Risk: derivative overclaiming

Mitigation:

- follow PR #20 derivative-claim boundaries;
- mark solve-through derivatives experimental until AD-vs-FD gates pass;
- promote only pure kernels and validated fixed-branch derivatives.

---

## 30. Final design decisions

1. Use `vmec_jax/mirror/` as a domain package, not flat root-level helper files.
2. Use Chebyshev--Gauss--Lobatto nodes in axial `xi` for the first implementation.
3. Store axial dependence nodally at first; add coefficient transforms later.
4. Use Fourier in `theta`, not in `xi`.
5. Use VMEC-like radial finite differences first.
6. Use `r = sqrt(s) a` for first axisymmetric regularity.
7. Use a divergence-free contravariant field representation with `lambda`.
8. Use variational energy/residual solve, not Grad--Shafranov or Green's functions.
9. Add scalar pressure first, anisotropic API later.
10. Add fixed boundary first, free boundary later.
11. Use `mout_*.nc` mirror-native output.
12. Make `vmec --plot` dispatch seamlessly for `mout`.
13. Add mirror-specific straight-field-line / Boozer-like transform after fixed-boundary output is stable.
14. Use WHAM/magpylib/Pleiades/DESC/GVEC as validation and design anchors, not as replacement solvers.
15. Treat tests as scientific verification gates: analytic, manufactured, parity, convergence, regression, and benchmark tests.

---

## 31. 2026-06-16 mirror-plot and boundary-condition assessment

This assessment was added after reviewing the first nonaxisymmetric fixed-boundary plots against mirror expectations, DESC fixed-boundary behavior, and magnetic-mirror literature.

### Plotting convention

- 3-D mirror plots should show the open axial coordinate horizontally.
- The plotting convention is therefore `(horizontal, vertical, depth) = (z, x, y)`.
- Boundary `|B|`, vector, field-line, and geometry plots should use this convention consistently.

### Field lines and pitch

- Boundary magnetic-field plots should overlay field-line traces from one cap to the other.
- The trace equation used for the side boundary is `d theta / d xi = B^theta / B^xi`.
- For the current zero-current examples, `I' = 0`, so no appreciable pitch is expected.
- Pitch should appear in later tests once finite `I'`, finite `lambda`, or intentionally helical boundary/input data are used.
- Open mirror field lines do not have a toroidal rotational transform; current plots should label `I'/Psi'` as an open-field twist proxy, not as true iota.

### Poloidal symmetry

- A physical axisymmetric mirror should be poloidally symmetric up to numerical tolerance.
- The observed theta variation in the first stress-test plot is not, by itself, evidence of poor convergence; it is imposed by the prescribed `epsilon*cos(m theta)` side boundary.
- DESC fixed-boundary solves likewise preserve the user-specified spectral boundary rather than symmetrizing it.
- Axisymmetric mirror examples and tests should use `epsilon = 0`, and should include explicit checks that boundary radius and `|B|` have negligible theta variation.
- Nonaxisymmetric examples should be documented as stress tests for 3-D fixed-boundary machinery, not as canonical mirror configurations.

### End-cap field strength

- Magnetic-mirror literature describes stronger field at the mirror throats/end regions and weaker field in the central well.
- In a fixed-flux, low-beta, axisymmetric approximation, `|B_z|` scales approximately like flux divided by cross-sectional area, so smaller end radius gives stronger end-cap field.
- The earlier positive-`a2` polynomial boundary made the radius larger at the caps and therefore produced the wrong qualitative trend.
- Canonical fixed-boundary examples should use smaller end radii or a tabulated WHAM/Pleiades-like flux tube so that `|B|_end > |B|_center`.
- Longer mirrors are useful for visual separation, but the sign and physical meaning of the radius profile matter more than length alone.

### DESC comparison

- DESC fixed-boundary equilibria enforce explicit surface constraints on the LCFS coefficients (`FixBoundaryR`, `FixBoundaryZ`, and related self-consistency constraints).
- DESC uses spectral surface representations in angular coordinates; the boundary condition is an input constraint, not a post-solve symmetry operation.
- DESC does not directly provide the open mirror cap policy needed here, but it gives the right design lesson: make fixed-boundary and cap constraints explicit, mode-resolved, and testable.
- The mirror branch should therefore add an explicit cap-policy layer: equal left/right caps by default, optional independent cap data, and axisymmetric mode filters for physical mirror examples.

### Residual and convergence plots

- Every example figure set should include a residual-history plot with projected residual norm over solve-history index.
- For L-BFGS examples, convergence plots should also include accepted reduced-step norms because the energy is the optimized scalar and projected residual callbacks are not guaranteed to decrease monotonically every recorded step.
- Once a VMEC-like `fsq` diagnostic is implemented for mirror solves, plots should include both `fsq` and the projected residual norm.
- Example captions and logs should report final energy, final residual, minimum Jacobian, mirror ratio, and end/center `|B|` comparison.

### Plan changes from this assessment

- Use horizontal-z 3-D plots by default.
- Overlay field lines on boundary magnetic-field vector plots.
- Correct canonical examples so cap fields are stronger than center fields.
- Add regression tests for end-stronger-than-center field strength in mirror-like fixed-boundary cases.
- Promote theta-symmetry tests for axisymmetric mirror cases into the M9/M11 validation lane.
- Treat DESC as a boundary-condition reference for explicit fixed-surface constraints, while keeping vmec_jax mirror cap conditions mirror-native.

---

## 32. 2026-06-16 analytic two-coil benchmark lane

The next step from scaffolds toward research-grade solves is to add analytic gates before adding more optimizer complexity.

### Analytic reference

- Use the closed-form on-axis magnetic field of a circular current loop:
  `B_z(z) = mu0 I R^2 / (2 (R^2 + (z - z0)^2)^(3/2))`.
- Use two equal circular coils centered at `z = +/- L/2` as the first mirror benchmark.
- The analytic field should be used to build a fixed near-axis flux-tube boundary with `r_b(z) = sqrt(2 psi / |B_z(z)|)`.
- With `I' = 0`, `lambda = 0`, and the self-similar axisymmetric state `r = sqrt(s) r_b(z)`, the mirror field on axis should reproduce the analytic `B_z(z)` to roundoff.

### What this validates now

- fixed-boundary flux normalization;
- axis regularity for the stored `a(s,z)` representation;
- mirror ratio and cap/center field trend;
- root example plotting, including coil location, flux tube, `|B|`, field lines, and residual/step history.

### What this does not validate yet

- finite-radius off-axis Biot-Savart field matching;
- free-boundary coil-plasma coupling;
- anisotropic finite-beta mirror equilibrium;
- VMEC-like `fsq` force convergence.

### Required code gates

- generic circular-loop on-axis helpers independent of the WHAM fixture;
- two-coil fixed-boundary example in the repo-root `examples/` folder;
- test comparing the helper to the full circular-loop on-axis branch;
- test comparing mirror on-axis `B_z` to the analytic two-coil `B_z`;
- plots and JSON metrics reporting analytic mirror ratio, mirror output mirror ratio, relative on-axis error, final residual, energy, and minimum Jacobian.

### Next solve-quality steps

- Add a true mirror `fsq` diagnostic rather than overinterpreting the current projected residual norm.
- Add a normalized force metric that separates boundary-constrained degrees of freedom from free interior degrees of freedom.
- Add off-axis low-radius comparisons against circular-loop Biot-Savart fields to test `B_r` and radial variation.
- Add convergence studies in `ns` and `nxi` for the analytic two-coil boundary.
- Only after those gates pass, promote higher-beta scalar-pressure and then anisotropic mirror solve benchmarks.

---

## 33. 2026-06-16 force-diagnostic, off-axis, and finite-current pitch lane

This lane moves the analytic two-coil work from plotting scaffold toward testable research diagnostics.

### Implemented in this lane

- Added mirror-native projected-force diagnostics to fixed-boundary traces and `mout` files:
  - `fsq = ||F_projected||^2 / N_active`;
  - `normalized_force = sqrt(fsq) / max(|energy_total|, tiny)`;
  - `active_force_dof`, counting only unconstrained interior shape degrees of freedom plus gauge-reduced `lambda` degrees of freedom.
- Extended residual-history plots to show projected residual norm, normalized force, mirror `fsq`, accepted step norm, and total energy.
- Added reusable circular-loop Biot-Savart helpers for off-axis axisymmetric fields:
  - `circular_loop_field_rz(r, z_rel, ...)`;
  - `two_coil_field_rz(r, z, ...)`.
- Extended `examples/mirror_two_coil_axisym.py` so the root example writes:
  - coil-overlaid 3-D flux-tube geometry;
  - coil-overlaid 3-D `|B|`;
  - analytic on-axis `B_z` comparison;
  - low-radius off-axis `B_r`/`B_z` comparison against circular-loop Biot-Savart;
  - `ns`/`nxi` convergence JSON and plot.
- Added `examples/mirror_finite_current_pitch.py`, using the same two-coil boundary with nonzero `I'` so field-line pitch is visible from cap to cap.
- Added tests for:
  - circular-loop off-axis field against independent direct Biot-Savart quadrature;
  - two-coil on-axis mirror benchmark;
  - low-radius off-axis mirror-vs-Biot-Savart agreement;
  - root two-coil CLI example;
  - root finite-current pitch CLI example and field-line theta advance;
  - `mout` round-trip of `fsq`, normalized force, and active force DOF.

### Current benchmark results

- The vacuum two-coil axis field matches the analytic on-axis `B_z` to roundoff (`axis_bz_relative_linf ~ 5.8e-16`).
- The two-coil mirror ratio is `13.940048820983622`, and the output mirror ratio agrees to plotting/roundoff tolerance.
- The low-radius off-axis comparison at `s = 0.125` has relative `B_r` and `B_z` errors of about `2.8%` and `2.7%`, respectively. This is acceptable for the near-axis flux-tube benchmark and should not be overclaimed as full finite-radius coil-plasma matching.
- The finite-current pitch example gives mean cap-to-cap theta advance of about `3.43 rad` (`0.55` turns) for the default `I'`.
- The convergence sweep currently shows machine-level on-axis errors and decreasing `fsq` with increasing `ns`/`nxi`; off-axis errors are limited by the near-axis boundary model, not by the loop-field analytic expression.

### Interpretation

- The new `fsq` is a mirror-native normalized projected-force diagnostic, not a bit-for-bit VMEC toroidal `fsq` port.
- It is now suitable for comparing mirror solves across resolutions and examples, but further scaling studies are needed before using absolute thresholds as convergence guarantees.
- The finite-current example verifies the field-line pitch visualization path. It is not yet a finite-beta or anisotropic mirror physics benchmark.

### Next research gates

- Add cap-policy objects for equal symmetric caps, independent left/right caps, and explicit axisymmetric mode filtering.
- Add manufactured equilibria with known nonzero `I'` and pressure where projected forces have analytic expectations.
- Add scalar-pressure two-coil benchmarks and compare convergence of energy, `fsq`, and magnetic well proxy over `ns`, `nxi`, and optimizer settings.
- Add higher-order off-axis benchmarks by shrinking the flux tube radius and confirming the low-radius Biot-Savart discrepancy scales down as expected.
- Add anisotropic-pressure APIs only after scalar-pressure force diagnostics converge on these analytic and manufactured cases.

---

## 34. 2026-06-16 fixed-boundary solve diagnostic and mirror CLI plotting lane

This lane diagnoses the current reduced-coordinate fixed-boundary solve behavior and promotes mirror plotting through the standard `vmec --plot` path.

### Implemented in this lane

- Added explicit `ftol` to `MirrorSolveOptions` and `OptimizerOptions`; `tolerance` remains the projected-gradient `gtol`.
- Added L-BFGS-B lower bounds for independent reduced `a` coordinates while leaving gauge-fixed `lambda` coordinates unbounded.
- Added optimizer summaries to in-memory fixed-boundary results:
  - raw optimizer success/status/message;
  - `nit`, `nfev`, `njev`;
  - whether the mirror wrapper accepted the optimizer final state.
- Added `examples/mirror_fixed_boundary_solve_diagnostic.py`.
  - Default diagnostic uses `ns_array=31`, `maxiter=2000`, `ftol=1e-12`, `gtol=1e-12`.
  - The example starts from a perturbed interior state on the two-coil fixed boundary.
  - It writes JSON metrics plus standard mirror plots for each `ns`.
- Added mirror cross-section plot data and writer.
- Added cross sections to the standard `plot_mirror_output` bundle.
- Updated 3-D boundary plots and root example coil plots so field-line overlays are visible on the rendered surface.
- Verified `vmec --plot mout_*.nc` writes:
  - nested `r-z` surfaces;
  - cross sections;
  - 3-D boundary `|B|` with field-line overlays;
  - boundary field-direction plot;
  - `|B|` maps;
  - Jacobian;
  - pressure/beta;
  - radial diagnostics;
  - residual/force history.

### Current fixed-boundary solve finding

For the two-coil perturbed-interior diagnostic with `ns=31`, `nxi=33`, `maxiter=2000`, `ftol=1e-12`, `gtol=1e-12`, and `line_search_steps=128`:

- SciPy L-BFGS-B reports `CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH`.
- It uses `nit=10`, `nfev=266`, and `njev=266`.
- The mirror wrapper rejects the optimizer final state, so no accepted energy or residual reduction is recorded.
- The accepted final projected residual remains `0.05300130515941412`.
- The accepted final mirror `fsq` remains `1.4855305915395756e-06`.
- The requested projected `gtol=1e-12` is not reached.

### Interpretation

- Reaching SciPy `ftol` is not sufficient for a valid mirror fixed-boundary solve; the candidate state must also pass mirror admissibility and accepted-energy checks.
- The current two-coil fixed-boundary residual is therefore a real open lane, not a plotting artifact.
- The next solver work should inspect why the raw L-BFGS-B final state is rejected:
  - positive-Jacobian failure versus energy increase;
  - reduced-coordinate scaling;
  - projected-gradient consistency with the reduced L-BFGS-B objective;
  - end-cap constraints and whether fixed cap nodes overconstrain the two-coil near-axis boundary.

### Next gates

- Store diagnostic raw candidate-state reason for rejection in optimizer summaries.
- Add reduced-coordinate scaling/preconditioning before increasing optimizer complexity.
- Add a manufactured fixed-boundary case that reaches projected `gtol` and use it as the first solver-convergence acceptance test.
- Add a two-coil solve-quality study over `ns`, `nxi`, `line_search_steps`, `ftol`, `gtol`, and perturbation amplitude.
- Keep `vmec --plot mout_*.nc` as a required regression path for every new mirror output quantity.

---

## 35. 2026-06-16 VMEC-like optimizer scaling and candidate rejection diagnostics lane

This lane aligns the mirror reduced-coordinate L-BFGS path with the existing
toroidal `vmec_jax.optimization` scaling convention before adding a larger
VMEC-style residual iteration.

### Implemented in this lane

- Added `reduced_coordinate_scaling` to `MirrorSolveOptions` and `OptimizerOptions`.
- Default scaling is `geometry`.
  - Independent radius DOFs scale with the local fixed-boundary radius.
  - Gauge-fixed `lambda` DOFs use the median boundary-radius scale until a
    dedicated mirror radial/lambda preconditioner is promoted.
- The mirror L-BFGS-B wrapper now optimizes `y = x / scale`, transforms
  gradients as `grad_y = grad_x * scale`, and scales positive-radius bounds in
  the same coordinate system. This follows the same `x_scale` pattern used in
  regular toroidal fixed-boundary optimization.
- Added candidate-state diagnostics to optimizer summaries:
  - raw candidate energy and residual;
  - raw candidate minimum `a` and minimum `sqrt(g)`;
  - whether energy, radius, and Jacobian acceptance gates passed;
  - a compact rejection reason such as `energy_increase` or
    `nonpositive_jacobian`.
- Added the scaling policy to mirror output metadata.
- Extended the root fixed-boundary diagnostic JSON with the raw candidate
  diagnostics.

### Solver-design interpretation

- This is a diagonal preconditioner/scaling step, not yet a full VMEC radial
  block preconditioner.
- It is deliberately close to the production toroidal optimizer API: scale the
  internal optimizer variable, leave the physical reduced vector and force
  kernels unchanged, and enforce bounds in scaled coordinates.
- The next solver upgrade should reuse the regular VMEC residual-iteration
  ingredients where applicable:
  - accepted-state monotonicity/restart logic;
  - limited-memory descent safeguards;
  - radial smoothing/tridiagonal preconditioning adapted to open-ended `xi`
    cap constraints;
  - a separate lambda preconditioner after the mirror lambda residual has a
    stronger manufactured benchmark.

### Updated fixed-boundary solve finding

With geometry scaling enabled, the two-coil perturbed-interior diagnostic with
`ns=31`, `nxi=33`, `maxiter=2000`, `ftol=1e-12`, `gtol=1e-12`, and
`line_search_steps=128` now has an accepted raw optimizer candidate:

- SciPy L-BFGS-B still stops by `CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH`.
- It uses `nit=359`, `nfev=376`, and `njev=376`.
- The mirror wrapper accepts the candidate (`optimizer_rejection_reason="accepted"`).
- The final projected residual drops from `0.05300130515941412` to
  `2.7422308936414194e-06`.
- The final mirror `fsq` drops to `3.976642133284726e-15`.
- The accepted candidate has `min(a)=0.07743070667688474` and
  `min(sqrt(g))=0.0028931215777452182`.
- The requested projected `gtol=1e-12` is still not reached, so this is an
  accepted improvement, not yet a full projected-gradient convergence proof.

### Next gates

- Add a manufactured fixed-boundary case that reaches projected `gtol`; this
  should become the first true convergence acceptance test.
- Add a mirror residual-iteration solver lane that mirrors the regular VMEC
  residual iteration more directly than scalar L-BFGS-B does.

---

## 36. 2026-06-16 manufactured fixed-boundary convergence gate

This lane adds the first true fixed-boundary mirror convergence gate with a
known stationary state. It is separate from the two-coil benchmark: the two-coil
case validates physically motivated vacuum-field geometry, while this
manufactured case validates the reduced solver machinery against a known
projected residual target.

### Implemented in this lane

- Added the named MMS case `axisym_projected_fixed_boundary`.
  - The exact state is explicitly projected through the same fixed-boundary,
    end-cap, axis, and lambda-gauge policy used by the mirror solver.
  - MMS sources are generated from the exact projected state.
- Added `solve_axisym_mms_fixed_boundary`.
  - It uses the same reduced-coordinate packing and geometry scaling as the
    mirror L-BFGS-B path.
  - It solves the reduced manufactured force vector directly with an exact
    JAX reduced Hessian and damped Newton line search.
  - It enforces positive-radius trial states.
- Added a regression test showing a perturbed projected state reaches
  projected `gtol=1e-12`.
- Added `examples/mirror_manufactured_fixed_boundary.py`.
  - It writes JSON metrics.
  - It writes residual/`fsq`/exact-error history, exact-vs-solved geometry,
    solved `|B|` map, and 3-D boundary `|B|` plots.

### Validation result

For `ns=5`, `nxi=9`, `maxiter=20`, `gtol=1e-12`, `ftol=1e-12`, and a
`0.2%` admissible perturbation:

- Damped Newton reaches `gtol` in `3` accepted iterations.
- It uses `nfev=6` and `njev=3`.
- Reduced residual drops from `0.009638224760704337` to
  `3.109010553584091e-15`.
- Manufactured `fsq` drops to `1.5845814134913533e-31`.
- Exact-state error drops from `3.790866427575665e-04` to
  `1.1114510037036217e-16`.

### Interpretation

- This is the first mirror fixed-boundary convergence proof against a known
  projected solution.
- The manufactured residual solve is a validation harness, not a replacement
  for the production physical two-coil solve path.
- The next production solver step should adapt this residual-iteration pattern
  to unsourced physical fixed-boundary solves, with VMEC-style accepted-state
  safeguards and radial/lambda preconditioners.

### Next gates

- Add a production `optimizer="residual_newton"` or similarly named mirror
  residual-iteration path that reuses the same reduced scaling and acceptance
  diagnostics.
- Add a comparison table over gradient descent, scaled L-BFGS-B, and residual
  Newton on cylinder, manufactured, and two-coil cases.
- Extend manufactured gates to finite pressure and nonzero-current cases after
  the scalar-pressure physical solver path is stable.

---

## 37. 2026-06-16 production axisymmetric residual-Newton optimizer lane

This lane promotes the residual-iteration idea from the manufactured validation
harness into the production fixed-boundary mirror solver for axisymmetric
states.

### Implemented in this lane

- Added `optimizer="residual_newton"` for axisymmetric fixed-boundary mirror
  solves.
- Added `residual_linear_maxiter` to `MirrorSolveOptions` and
  `OptimizerOptions`.
- The optimizer uses:
  - the same reduced-coordinate packing as L-BFGS-B;
  - the same geometry-based coordinate scaling;
  - a matrix-free exact JAX Hessian-vector product;
  - bounded SciPy `lsmr` inner solves;
  - damped backtracking;
  - positive-radius and positive-Jacobian admissibility checks;
  - monotone physical-energy and projected-residual acceptance;
  - the existing optimizer summary and candidate diagnostics path.
- The 3-D dispatcher now rejects `residual_newton` explicitly with a clear
  axisymmetric-only error.
- `examples/mirror_fixed_boundary_solve_diagnostic.py` now accepts
  `--optimizer lbfgs|residual_newton` and `--residual-linear-maxiter`.

### Validation result: perturbed cylinder

For a small physical cylinder with `ns=5`, `nxi=9`, `tolerance=1e-12`,
`ftol=1e-14`, and `residual_linear_maxiter=64`:

- Projected residual drops from `0.01523117825133471` to
  `5.559206409266248e-17`.
- The optimizer reaches projected `gtol`.
- It uses `5` accepted outer iterations.
- Energy decreases from `0.013971785058962185` to the accepted final state.
- The final state keeps positive radius and positive Jacobian.

### Validation result: small two-coil physical diagnostic

For the two-coil diagnostic with `ns=9`, `nxi=17`, `maxiter=15`,
`gtol=1e-8`, `ftol=1e-12`, and `residual_linear_maxiter=32`:

- The optimizer accepts all `15` outer steps but does not reach `gtol`.
- Projected residual drops from `0.06786492634022206` to
  `6.53350188839826e-05`.
- Mirror `fsq` drops to `1.7143231697069735e-11`.
- Energy drops by `0.00029052860893919244`.
- Candidate diagnostics pass:
  - `optimizer_rejection_reason="accepted"`;
  - `min(a)=0.07860225137361508`;
  - `min(sqrt(g))=0.0029972531869492825`.

### Interpretation

- The production residual-Newton path is now useful for small axisymmetric
  physical solves and reaches tight projected residuals on a cylinder.
- On the two-coil benchmark it improves residual and energy substantially but
  still does not reach `gtol=1e-8` within the tested budget.
- This confirms the next open solver problem is not just scalar optimization
  scaffolding: the two-coil physical solve needs stronger preconditioning,
  better cap-policy studies, or a refined residual model before claiming tight
  convergence.

### Next gates

- Add a comparison table/example over gradient descent, scaled L-BFGS-B, and
  residual Newton for cylinder, manufactured, and two-coil cases.
- Add radial/lambda preconditioning for the residual-Newton inner solve, using
  the regular VMEC radial preconditioner design as the reference but adapting
  it to open `xi` caps.
- Study two-coil convergence over `residual_linear_maxiter`, `maxiter`,
  `ns`, `nxi`, and cap constraints.

---

## 38. 2026-06-17 fixed-boundary solver comparison benchmark lane

This lane adds a compact benchmark/report script that compares the currently
available mirror fixed-boundary solver paths before adding the next
preconditioner layer.

### Implemented in this lane

- Added the root-level `examples/mirror_solver_comparison.py` script.
- The script runs production fixed-boundary solves on two physical cases:
  - perturbed circular cylinder;
  - perturbed analytic two-coil mirror boundary.
- For each physical case it compares:
  - projected gradient descent;
  - geometry-scaled L-BFGS-B;
  - axisymmetric residual Newton.
- The same report includes the sourced manufactured fixed-boundary convergence
  gate using `solve_axisym_mms_fixed_boundary`.
  - This is intentionally marked as `manufactured_source_validation` because
    production gradient descent and L-BFGS-B do not yet include MMS source
    terms.
- The script writes:
  - `solver_comparison_metrics.json` with per-case/per-solver metrics and full
    residual histories;
  - residual-history and final-residual comparison plots;
  - physical benchmark boundary plots with horizontal `z`;
  - standard mirror `mout` and plot bundles for the residual-Newton physical
    cases, including 3-D boundary, field-line overlays, `|B|`, cross sections,
    and residual history.
- Added a smoke regression that runs all three benchmark cases without plots at
  reduced resolution.
- Documented the new root example in `examples/mirror/README.md`.

### Interpretation

- The comparison report now makes the solver tradeoff visible in one place:
  gradient descent is a low-order baseline, scaled L-BFGS-B improves the scalar
  energy objective, and residual Newton is the first path that reaches tight
  projected residuals on the cylinder.
- The two-coil physical case remains the hard production benchmark: residual
  Newton reduces the residual substantially, but the next solver lane still
  needs a stronger radial/lambda preconditioner and cap-policy studies before
  claiming tight convergence on realistic mirror boundaries.
- The sourced MMS row is a validation gate, not a production optimizer
  comparison row. Source-aware gradient/L-BFGS wrappers can be added later if
  they become useful for diagnosing the manufactured problem.

### Default benchmark result before residual preconditioning

For `examples/mirror_solver_comparison.py --outdir results/mirror/solver_comparison`:

- Cylinder:
  - gradient descent: residual `1.523118e-02 -> 9.100286e-03`;
  - scaled L-BFGS-B: residual `1.523118e-02 -> 4.342639e-07`;
  - residual Newton: residual `1.523118e-02 -> 5.559206e-17`, reaching
    projected `gtol`.
- Two-coil:
  - gradient descent: residual `6.786493e-02 -> 1.719975e-02`;
  - scaled L-BFGS-B: residual `6.786493e-02 -> 1.006745e-04`;
  - residual Newton: residual `6.786493e-02 -> 2.433266e-05`, not yet
    reaching projected `gtol`.
- Manufactured sourced gate:
  - residual Newton: residual `9.638225e-03 -> 3.109011e-15`, reaching
    projected `gtol`.

### Next gates

- Use the comparison example as the baseline artifact when adding radial/lambda
  preconditioning to residual Newton.
- Extend the comparison to a grid over `ns`, `nxi`, `maxiter`, and
  `residual_linear_maxiter` after the preconditioner is in place.
- Add source-aware manufactured gradient/L-BFGS wrappers only if the
  manufactured comparison needs optimizer-level parity beyond the current
  residual-Newton convergence gate.

---

## 39. 2026-06-17 mirror residual preconditioning lane

This lane adds the first VMEC-like reduced-coordinate preconditioner to the
axisymmetric mirror residual-Newton path.

### Implemented in this lane

- Added `residual_preconditioner`, `residual_radial_alpha`,
  `residual_lambda_alpha`, and `residual_xi_alpha` to `MirrorSolveOptions` and
  `OptimizerOptions`.
- Added `axisym_reduced_residual_preconditioner`.
  - It respects the mirror reduced-coordinate packing:
    independent interior radius `a` nodes followed by gauge-fixed `lambda`
    nodes.
  - It applies a symmetric tridiagonal zero-Dirichlet smoother to reduced
    coordinates.
  - `radial_tridi` applies VMEC-like radial smoothing to `a` and `lambda`.
  - `radial_xi_tridi` also smooths radius updates along the open axial `xi`
    direction, using zero ghost caps to respect fixed mirror end-cap
    constraints.
- Wired the preconditioner into the residual-Newton `lsmr` solve as a right
  preconditioner.
  - The matrix-vector product solves `H_y P z = -g_y`.
  - The transpose path applies `P H_y` because the smoother is symmetric and
    the reduced Hessian is symmetric.
  - The physical trial step remains `step_y = P z`.
- Promoted the mirror residual-Newton default to
  `residual_preconditioner="radial_xi_tridi"`,
  `residual_radial_alpha=0.5`, `residual_lambda_alpha=0.5`, and
  `residual_xi_alpha=0.2`.
- Kept `--residual-preconditioner none` and `radial_tridi` available for
  controlled baseline studies.
- Wrote the preconditioner settings to mirror output metadata, fixed-boundary
  diagnostic JSON, and solver-comparison JSON.
- Added direct unit coverage for the reduced preconditioner layout, identity
  mode, high-frequency damping, size validation, and alpha validation.
- Updated the root diagnostic and solver-comparison examples to expose the new
  controls.

### Validation result

For the two-coil physical diagnostic with `ns=9`, `nxi=17`, `maxiter=12`,
`line_search_steps=32`, `optimizer=residual_newton`,
`residual_linear_maxiter=48`, `gtol=1e-12`, and `ftol=1e-12`:

- No preconditioner:
  - residual `6.78649263e-02 -> 2.43326621e-05`;
  - `fsq=2.37782508e-12`;
  - `nit=12`, `njev=576`.
- Radial/lambda tridiagonal preconditioner:
  - residual `6.78649263e-02 -> 4.33062917e-06`;
  - `fsq=7.53186709e-14`;
  - `nit=12`, `njev=576`.
- Radial/lambda plus open-`xi` radius preconditioner:
  - residual `6.78649263e-02 -> 8.18367717e-07`;
  - `fsq=2.68966153e-15`;
  - `nit=12`, `njev=576`.

For the regenerated default solver-comparison report at
`results/mirror/solver_comparison_preconditioned`:

- Cylinder residual Newton reaches `5.610232e-17`.
- Two-coil residual Newton reaches `8.183677e-07`.
- The sourced manufactured gate still reaches `3.109011e-15`.
- The generated plot bundle still renders horizontal-`z` geometry, visible
  field-line overlays, and `|B|` maps with high-field end caps and a low-field
  central well.

### Interpretation

- The preconditioner is a real improvement on the two-coil physical benchmark:
  about `5.6x` better than radial/lambda-only smoothing and about `30x` better
  than the unpreconditioned residual-Newton solve at the same outer and inner
  iteration budgets.
- This still does not prove tight two-coil convergence to `gtol=1e-12`; it
  lowers the residual floor for the next convergence-grid lane.
- The implementation is deliberately a reduced-coordinate mirror analogue of
  regular VMEC's radial tridiagonal residual preconditioning, with the open
  mirror `xi` cap adaptation kept explicit and tunable.

### Next gates

- Run convergence grids over `ns`, `nxi`, `residual_linear_maxiter`, and
  `maxiter` with the new default preconditioner.
- Study whether cap constraints or a better field-aligned axial preconditioner
  are needed to reach projected `gtol` on the two-coil physical benchmark.
- Add richer solver plots that compare preconditioner modes directly in one
  report if the grid study shows a stable benefit across resolutions.

---

## 40. 2026-06-17 residual-Newton convergence-grid lane

This lane turns the single two-coil residual-Newton/preconditioner benchmark
into an explicit convergence-grid artifact over resolution and solver budget.

### Implemented in this lane

- Added the root-level `examples/mirror_residual_newton_convergence_grid.py`
  script.
- The script runs the analytic two-coil fixed-boundary residual-Newton solve
  over:
  - `ns`;
  - `nxi`;
  - outer `maxiter`;
  - inner `residual_linear_maxiter`;
  - preconditioner mode.
- The script writes `residual_newton_convergence_grid_metrics.json` containing
  per-row residuals, `fsq`, optimizer status, iteration counts, and per-row
  residual histories.
- With plots enabled it writes:
  - a final-residual heatmap over `ns` and `nxi`;
  - a solver-budget plot at the highest `ns,nxi`;
  - a residual-history plot at the highest `ns,nxi` and largest budgets;
  - standard mirror `mout`/plot bundles for both the best-residual row and the
    highest-resolution/highest-budget row.
- Added root-example smoke coverage for the convergence-grid script.
- Documented the script in `examples/mirror/README.md`.

### Validation result

For
`examples/mirror_residual_newton_convergence_grid.py --outdir results/mirror/residual_newton_convergence_grid`
with the default preconditioner `radial_xi_tridi`:

- `ns=5`, `nxi=9`, `maxiter=6`, `linear=16`:
  residual `1.127985e-11`, not below `gtol=1e-12`.
- `ns=5`, `nxi=9`, `maxiter=6`, `linear=48`:
  residual `1.671645e-16`, reaches `gtol`.
- `ns=5`, `nxi=17`, `maxiter=6`, `linear=48`:
  residual `7.011819e-14`, reaches `gtol`.
- `ns=9`, `nxi=9`, `maxiter=12`, `linear=48`:
  residual `6.563983e-11`, still above `gtol`.
- `ns=9`, `nxi=17`, `maxiter=12`, `linear=48`:
  residual `8.183677e-07`, still above `gtol`.

The best row is `ns=5`, `nxi=9`, `maxiter=6`, `linear=48`.
The hard reference row is `ns=9`, `nxi=17`, `maxiter=12`, `linear=48`.
Both selected rows write standard mirror plot bundles. The hard reference row
still shows horizontal-`z` geometry, visible field-line overlays, and `|B|`
with high-field end caps and a low-field central well.

### Interpretation

- The convergence grid shows that the new preconditioner can reach tight
  projected residuals on smaller two-coil grids and on `ns=5,nxi=17`.
- The `ns=9,nxi=17` case remains the hard benchmark. Increasing the inner
  `lsmr` budget from `16` to `48` matters more than doubling the outer budget,
  but the residual is still `8.183677e-07` at `maxiter=12`.
- The highest-resolution residual history is still decreasing at the final
  recorded iteration, so the next study should extend the high-resolution row
  to larger outer and inner budgets before changing physics assumptions.
- If that extended row still stalls far above `gtol`, the next likely issues
  are cap-policy conditioning and the absence of a more field-aligned axial
  preconditioner.

### Next gates

- Run a targeted high-resolution budget extension for `ns=9`, `nxi=17`, with
  larger `maxiter` and `residual_linear_maxiter` values.
- Add a preconditioner-mode comparison grid for the high-resolution row only:
  `none`, `radial_tridi`, and `radial_xi_tridi`.
- Inspect cap-node and end-cap residual components separately if the extended
  high-resolution row stops decreasing before reaching projected `gtol`.

---

## 41. 2026-06-17 high-resolution residual decomposition and budget lane

This lane adds residual component diagnostics to the two-coil residual-Newton
grid and uses them to decide whether the hard `ns=9,nxi=17` row is limited by
cap boundary conditions, outer iteration count, or inner matrix-free linear
solve budget.

### Implemented in this lane

- Extended `examples/mirror_residual_newton_convergence_grid.py` so every row
  recomputes the final projected AD residual and writes component norms for:
  - radius residual;
  - lambda residual;
  - cap, cap-adjacent, and interior `xi` contributions;
  - lambda radial-axis, radial-edge, and radial-interior contributions;
  - maximum absolute projected radius/lambda residuals.
- Added a residual-component plot to the convergence-grid report.
- Added smoke coverage that checks the component norm against the solver's
  final residual norm.
- Documented the residual-component plot in `examples/mirror/README.md`.

### High-resolution preconditioner result

For `ns=9`, `nxi=17`, `maxiter=12`, `residual_linear_maxiter=48`:

- No preconditioner:
  - residual `2.433266212e-05`;
  - `fsq=2.377825084e-12`.
- Radial/lambda tridiagonal preconditioner:
  - residual `4.330629175e-06`;
  - `fsq=7.531867088e-14`.
- Radial/lambda plus open-`xi` radius preconditioner:
  - residual `8.183677170e-07`;
  - `fsq=2.689661527e-15`.

The new residual decomposition shows all projected residual in the radius
equation for this vacuum axisymmetric two-coil case:

- `radial_xi_tridi` final radius norm: `8.183677170e-07`;
- lambda norm: `0.0`;
- radius cap-adjacent norm: `1.256561431e-07`;
- radius interior-`xi` norm: `8.086632512e-07`.

This makes the cap-boundary hypothesis less likely for this benchmark. The
dominant remaining residual is in the interior axial radius equation.

### High-resolution budget result

For `ns=9`, `nxi=17`, `radial_xi_tridi`, `gtol=1e-12`, `ftol=1e-12`:

- `maxiter=12`, `residual_linear_maxiter=48`:
  - residual `8.183677170e-07`;
  - `fsq=2.689661527e-15`;
  - status: maximum iterations reached.
- `maxiter=24`, `residual_linear_maxiter=48`:
  - residual `6.009149176e-09`;
  - `fsq=1.450195736e-19`;
  - status: maximum iterations reached.
- `maxiter=12`, `residual_linear_maxiter=96`:
  - residual `5.031504692e-13`;
  - `fsq=1.016708412e-27`;
  - status: `gtol` satisfied.
- `maxiter=24`, `residual_linear_maxiter=96`:
  - residual `5.031504692e-13`;
  - `fsq=1.016708412e-27`;
  - status: `gtol` satisfied.

The high-resolution hard row therefore can reach the requested projected
`gtol=1e-12`. The primary limitation at `linear=48` is inner linear solve
budget/conditioning, not the outer residual-Newton formulation and not an
obvious cap-boundary residual.

### Generated artifacts

- `results/mirror/residual_newton_highres_preconditioners/`
  - preconditioner comparison plot;
  - residual history;
  - residual component plot;
  - standard mirror plot bundle for the best high-resolution row.
- `results/mirror/residual_newton_highres_budget_extension/`
  - budget plot;
  - residual history;
  - residual component plot;
  - standard mirror plot bundles for the best and highest-budget rows.

The rendered plots show the mirror horizontally with `z` as the long axis,
visible field-line overlays, high `|B|` near both end caps, and a low-field
central well.

### Interpretation

- The VMEC-like reduced-coordinate tridiagonal preconditioner is effective but
  the `ns=9,nxi=17` row needs a larger inner `lsmr` budget than the earlier
  default to reach tight tolerance.
- Increasing outer iterations helps at fixed `linear=48`, but increasing the
  inner budget to `96` is the decisive change.
- The next production-quality solver lane should add adaptive inner linear
  iteration/tolerance control, using the current residual norm and requested
  `gtol`, instead of requiring users to manually guess `96`.
- A cap-aware preconditioner may still be useful for larger or finite-current
  cases, but it is not the first explanation for this two-coil benchmark.

### Next gates

- Add an adaptive residual-Newton inner solve policy, likely with a default
  derived from `ns`, `nxi`, current residual norm, and requested `gtol`.
- Run the same component/budget diagnostics on finite-current/helical-pitch
  examples where lambda residuals should no longer be identically zero.
- Promote the high-resolution `linear=96` two-coil row into a benchmark gate,
  with a runtime-conscious smoke variant for CI.

---

## 42. 2026-06-17 adaptive residual-Newton inner-budget lane

This lane turns the high-resolution budget finding into production solver
behavior: residual-Newton now has an adaptive inner `lsmr` budget policy so
users do not need to manually guess `residual_linear_maxiter=96` for the hard
two-coil row.

### Implemented in this lane

- Added `residual_linear_maxiter_policy` to `MirrorSolveOptions` and
  `OptimizerOptions`.
  - `adaptive` is the production default.
  - `fixed` preserves exact user-requested budgets for convergence studies.
- Added `residual_linear_adaptive_factor`, default `6.0`.
- Added `axisym_residual_linear_maxiter`, which chooses the effective `lsmr`
  iteration cap from:
  - the user-requested base `residual_linear_maxiter`;
  - the mirror resolution `max(ns,nxi)`;
  - the reduced system size, which caps the budget.
- Added residual-linear policy/effective-budget fields to optimizer summaries,
  mirror output attributes, solve-diagnostic JSON, solver-comparison JSON, and
  convergence-grid JSON.
- Kept `examples/mirror_residual_newton_convergence_grid.py` defaulting to
  `--residual-linear-maxiter-policy fixed`, so budget sweeps remain literal.
- Updated solver diagnostics and solver comparison examples to use the
  production adaptive policy by default.
- Added tests for fixed/adaptive policy behavior and summary reporting.

### Validation result

For the high-resolution two-coil hard row:

```bash
JAX_ENABLE_X64=1 python examples/mirror_residual_newton_convergence_grid.py \
  --outdir results/mirror/residual_newton_highres_adaptive_policy \
  --ns-array 9 \
  --nxi-array 17 \
  --maxiter-array 12 \
  --residual-linear-maxiter-array 16 \
  --residual-linear-maxiter-policy adaptive \
  --preconditioners radial_xi_tridi
```

Result:

- requested base `residual_linear_maxiter=16`;
- policy `adaptive`;
- effective max `102`;
- effective last `102`;
- final residual `2.023006523381e-13`;
- final `fsq=1.643596543631e-28`;
- status: `gtol` satisfied.

The residual remains radius-only at the final tolerance scale:

- radius norm `2.023006523381e-13`;
- lambda norm `0.0`;
- radius interior-`xi` norm `1.944627215141e-13`;
- radius cap-adjacent norm `5.576561555067e-14`.

For the regenerated public solver-comparison example:

```bash
JAX_ENABLE_X64=1 python examples/mirror_solver_comparison.py \
  --outdir results/mirror/solver_comparison_adaptive
```

Key residual-Newton results:

- cylinder residual `5.610231716299e-17`, effective inner budget `54`;
- two-coil residual `2.023006523381e-13`, effective inner budget `102`;
- sourced manufactured gate residual `3.109010553584e-15`.

### Interpretation

- The previous high-resolution two-coil residual gap was an inner linear-solve
  budget issue. Adaptive policy closes it from a requested base of `16`.
- The current production residual-Newton path is now a tight-convergence claim
  for the vacuum two-coil benchmark at `ns=9,nxi=17`, `gtol=1e-12`,
  `maxiter=12`.
- Fixed-budget studies remain available and reproducible by passing
  `residual_linear_maxiter_policy="fixed"` or the CLI equivalent.

### Next gates

- Add a runtime-conscious adaptive two-coil benchmark gate for CI, likely with
  a reduced resolution and a separate heavier example artifact for `ns=9,nxi=17`.
- Run finite-current/helical-pitch residual decomposition under adaptive
  residual Newton to validate lambda residual behavior.
- Start the M9 straight-field-line / mirror-Boozer-like diagnostic lane once
  the finite-current adaptive gate is documented.

---

## 43. 2026-06-17 plan, source, method, and architecture review

This section is the current single roadmap checkpoint.  The repo-root
`plan_mirror.md` is the canonical plan; the copy in
`/Users/rogeriojorge/Downloads/plan_mirror.md` must be kept byte-for-byte in
sync after each plan edit.

### Review scope

- Branch: `codex/mirror-geometry`.
- PR: <https://github.com/uwplasma/vmec_jax/pull/21>, kept as draft.
- Current CI state after the adaptive residual-Newton commit: all checks pass
  except the combined Python 3.11 coverage gate, where exact line coverage is
  about `94.92%` against the required `95.00%`.
- Current mirror package size: about `6,885` Python lines.
- Largest simplification target:
  `vmec_jax/mirror/solvers/fixed_boundary/optimizers.py`, about `1,337`
  lines, mixing reduced-state packing, scaling, preconditioners, L-BFGS,
  residual Newton, and 3D variants.
- Documentation drift found:
  - `docs/mirror/outputs.rst` still described schema `0.1`, while code writes
    schema `0.2`.
  - `docs/mirror/overview.rst` still described the old projected-gradient path
    as the main solver and did not describe residual Newton, adaptive inner
    budgets, solver-comparison examples, or mirror `--plot` output.

### External method review

- DESC current public source describes itself as solving and optimizing 3D MHD
  equilibria with pseudo-spectral methods and automatic differentiation.  Its
  source structure separates compute kernels, objective wrappers, optimizer
  wrappers, derivative helpers, geometry, I/O, and plotting.  Relevant local
  source anchors from `/tmp/DESC_review`:
  - `desc/derivatives.py`: `Derivative` wrappers select `grad`, `fwd`, `rev`,
    Hessian, JVP, and VJP paths.
  - `desc/objectives/objective_funs.py`: objectives expose scaling,
    derivative mode, and chunked Jacobian concepts.
  - `desc/optimize/optimizer.py`: optimizer orchestration separates objective
    construction, constraints, scaling, and solver dispatch.
  - `desc/backend.py`: `jax.lax.custom_root` wrappers provide implicit
    differentiation for root solves instead of differentiating through every
    nonlinear iteration.
- The spectral-solver adjoint preprint `arXiv:2506.14792` supports the same
  direction for sparse spectral PDE solvers: build adjoint/transpose solves
  from symbolic or operator graphs so gradients keep sparse-solver speed and
  memory behavior.
- JAX `lax.custom_linear_solve` is directly relevant to mirror residual Newton:
  it gives matrix-free linear solves with gradients defined implicitly at the
  solution and requires a transpose solve for reverse mode unless the operator
  is symmetric.
- JAXopt implicit differentiation is relevant for the fixed-boundary
  equilibrium map: `custom_root`, `custom_fixed_point`, `root_jvp`, and
  `root_vjp` give a direct route to differentiating solved states with respect
  to boundary, profile, and current parameters.
- Lineax is relevant for future JAX-native linear least-squares and Krylov
  solves because it supports PyTree-valued vectors, general linear operators,
  transposes, structured operators, and stable gradients.
- Optax is useful for composable first-order optimization and staged objective
  optimization, but it is not the main equilibrium Newton/Krylov solve engine.
- Equinox is useful if mirror config/state objects grow into mixed static and
  dynamic PyTrees, but adopting it now would add a dependency before there is a
  clear need.

### Differentiability policy

Use two explicitly different execution paths:

1. Fast CLI/reference path:
   - may use NumPy, SciPy `minimize`/`lsmr`, NetCDF, and Matplotlib;
   - must be fast, memory-conscious, and well diagnosed;
   - is allowed to be non-differentiable because it is for command-line solves,
     examples, benchmark plots, and regression artifacts.
2. Research-grade differentiable path:
   - must keep the residual/energy/field kernels as JAX functions over arrays
     and PyTrees;
   - must not rely on differentiating through host-side SciPy loops;
   - should use implicit root/linear-solve differentiation or explicit
     adjoint/VJP rules around converged solves;
   - should expose forward-mode JVPs for few-parameter, many-output studies and
     reverse/adjoint VJPs for scalar objectives over many design parameters;
   - should choose Lineax or `jax.lax.custom_linear_solve` for JAX-native
     matrix-free linear solves once the residual-Newton reference path is
     stable enough to port.

This policy resolves the CLI-versus-differentiability tension: CLI speed is
kept where useful, while the differentiable API gets a separate implementation
contract and validation gates.

### File-simplification plan

Do one no-behavior-change refactor before adding more solver machinery:

1. Split `optimizers.py` into small modules while preserving the public imports:
   - `types.py`: `OptimizerOptions`, `OptimizerStep`, `OptimizerRun`, and
     `_CandidateDiagnostics`;
   - `reduced.py`: axisymmetric/3D reduced-state masks, packing, unpacking,
     bounds, gradient packing, and coordinate scaling;
   - `preconditioners.py`: residual-preconditioner key parsing, tridiagonal
     smoothers, and adaptive inner-budget policy;
   - `optimizers.py`: high-level projected-gradient, L-BFGS, residual-Newton,
     and 3D solver dispatch only.
2. Keep re-export compatibility from `optimizers.py` until downstream examples
   and tests are updated.
3. Add tests around the refactor before changing behavior.
4. Keep docstrings pedagogical: explain what each reduced vector represents,
   what is fixed by the boundary, and why lambda has a gauge constraint.
5. Avoid adding a deeper directory tree unless a module grows beyond one
   coherent responsibility.

### Finite completion roadmap

The remaining draft-PR work is finite and ordered:

1. M8l coverage/docs gate:
   - add focused tests to lift exact coverage above `95.00%`;
   - update mirror Sphinx docs to schema `0.2` and current solver status;
   - confirm `ruff`, mirror tests, packaging smoke, and docs build.
2. M8m finite-current residual gate:
   - run adaptive residual-Newton residual decomposition on finite-current or
     helical-pitch cases;
   - verify lambda residuals are nonzero where expected and converge under the
     same diagnostics used for the vacuum two-coil case.
3. M8n no-behavior-change solver-file simplification:
   - split `optimizers.py` as listed above;
   - preserve public imports and JSON/mout fields;
   - rerun tests and regenerate one solver-comparison artifact.
4. M8o adaptive benchmark gate:
   - add a CI-conscious two-coil adaptive residual-Newton benchmark;
   - keep the heavier `ns=9`, `nxi=17` run as an example artifact, not a
     default CI cost.
5. M9 mirror straight-field-line diagnostics:
   - define a mirror-Boozer-like transform and pitch/twist diagnostics;
   - compare finite-current field-line traces, iota-like twist, and `|B|`
     spectra.
6. M10 differentiable solve API:
   - prototype implicit differentiation for the fixed-boundary solved state;
   - compare unrolled autodiff, implicit root VJP/JVP, and custom linear-solve
     approaches on small manufactured cases;
   - select the default method from accuracy, memory, runtime, and API
     simplicity, not from implementation convenience.
7. M11 optimization objectives:
   - expose mirror ratio, well depth, on-axis field, beta, and geometric
     objectives using the differentiable solve path where available;
   - keep CLI optimization benchmarks separate from differentiable API tests.
8. M12 documentation/readiness:
   - document examples, output schema, CLI plotting, solver choices,
     differentiability policy, and validation limits;
   - update the draft PR body with final results and convert from draft only
     after all gates pass.

### Completion percentages after this review

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `80%`.
- Fixed-boundary axisymmetric solve: `75%`.
- Residual Newton / preconditioning: `65%`.
- Two-coil and manufactured validation: `70%`.
- Finite-current pitch validation: `45%`.
- Plotting and `vmec --plot` mirror support: `75%`.
- I/O schema and docs: `60%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `15%`.
- Free-boundary mirror lane: `5%`.
- PR merge readiness overall: `55%`.

### Logging rule

Every future work pass must append a dated plan section with:

- steps taken;
- results obtained;
- how it was tested;
- file structure and best-practice notes;
- best next steps;
- percentage of completion for open lanes;
- whether user input is needed.

This keeps progress in one plan and prevents scattered status files.

---

## 44. 2026-06-17 M8l coverage and documentation gate

This lane implements the immediate actions found in the review above: fix the
coverage-gate risk, update stale mirror documentation, and keep the plan
canonical and synchronized.

### Steps taken

- Updated `docs/mirror/index.rst` to describe the current experimental status:
  fixed-boundary scalar-pressure geometry, fields, residuals, manufactured
  checks, `mout` output, plotting, and optimizer prototypes are present;
  free-boundary mirrors, anisotropic pressure, and the final differentiable
  implicit-solve API remain planned.
- Updated `docs/mirror/overview.rst` to include:
  - residual Newton;
  - adaptive inner linear budgets;
  - VMEC-like reduced-coordinate preconditioning;
  - solver-comparison, manufactured, finite-current, and fixed-boundary
    diagnostic examples;
  - the fast CLI/reference path versus research-grade differentiable path
    policy.
- Updated `docs/mirror/outputs.rst` from schema `0.1` to schema `0.2` and added
  the new residual-Newton metadata, `fsq`, and normalized-force output fields.
- Added `tests/mirror/test_mirror_low_level_coverage.py`, a low-cost coverage
  test file for:
  - Chebyshev/Fourier convenience methods and guardrails;
  - boundary, grid, profile, and state validation;
  - field/energy/residual diagnostic guardrails;
  - reduced-state source-shape checks and pressure-continuation guards;
  - circular-coil validation guardrails;
  - in-memory mirror-output plot naming and `show=True` dispatch.
- Synced the repo-root `plan_mirror.md` to
  `/Users/rogeriojorge/Downloads/plan_mirror.md`.

### Results obtained

- Mirror-only source coverage from the mirror test suite moved from `92%` to
  `95%`.
- Covered about 89 previously missed mirror source lines.  The combined CI gate
  was short by about 38 lines, so this should clear the exact `95.00%` gate
  unless unrelated coverage changes land first.
- The documentation now matches the current code schema and solver status.
- The plan now contains one finite roadmap from M8l through M12 and a clear
  differentiation policy.

### How it was tested

Commands run:

```bash
JAX_ENABLE_X64=1 python -m pytest -q tests/mirror/test_mirror_low_level_coverage.py
JAX_ENABLE_X64=1 coverage erase
JAX_ENABLE_X64=1 coverage run --source=vmec_jax -m pytest -q tests/mirror tests/test_packaging_metadata.py
coverage report --skip-covered --include='vmec_jax/mirror/*'
JAX_ENABLE_X64=1 python -m pytest -q tests/mirror/test_mirror_low_level_coverage.py tests/mirror/test_mirror_plotting.py tests/mirror/test_mirror_io.py
LC_ALL=C.UTF-8 LANG=C.UTF-8 SPHINX_FAST=1 python -m sphinx -W -j auto -b html docs docs/_build/html
LC_ALL=C.UTF-8 LANG=C.UTF-8 python -m sphinx -W -j auto -b html docs docs/_build/html_full
ruff check .
ruff check tests/mirror/test_mirror_low_level_coverage.py
ruff format --check tests/mirror/test_mirror_low_level_coverage.py
git diff --check
JAX_ENABLE_X64=1 python -m pytest -q tests/mirror tests/test_packaging_metadata.py
```

Passing results:

- new focused test: `6 passed`;
- mirror I/O/plot/new-test slice: `14 passed`;
- full mirror/package smoke: `88 passed, 1 skipped`;
- mirror coverage run: `88 passed, 1 skipped`, mirror source coverage `95%`;
- fast Sphinx docs: passed with warnings as errors;
- full mirror docs build: passed with warnings as errors;
- `ruff check .`: passed;
- new test file formatting check: passed;
- `git diff --check`: passed.

Note: full-repo `ruff format --check .` is not a useful gate on this branch
because many pre-existing files outside this lane would be reformatted.  The
new Python test file itself is formatted.

### File structure and best-practice notes

- The new tests are in `tests/mirror/` with the rest of the mirror domain
  coverage.  They exercise public APIs where possible and only touch private
  helpers that are already part of existing mirror solver tests.
- The docs remain under `docs/mirror/` and avoid duplicating example-level
  instructions already in `examples/mirror/README.md`.
- The canonical plan is still a single file, `plan_mirror.md`, and the
  downloaded copy is synchronized.
- The test file is intentionally low resolution and avoids new expensive solve
  cases, so it should help coverage without adding meaningful runtime.

### Best next steps

1. Commit and push the M8l review/docs/coverage update to the draft PR.
2. Confirm GitHub reruns the combined coverage gate green.
3. Start M8m: finite-current/helical-pitch residual decomposition under
   adaptive residual Newton, focusing on nonzero lambda residual behavior and
   visible pitch in field-line plots.
4. Then do M8n: no-behavior-change `optimizers.py` simplification before
   adding more solver machinery.

### Completion percentages after M8l

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `80%`.
- Fixed-boundary axisymmetric solve: `75%`.
- Residual Newton / preconditioning: `65%`.
- Two-coil and manufactured validation: `72%`.
- Finite-current pitch validation: `45%`.
- Plotting and `vmec --plot` mirror support: `75%`.
- I/O schema and docs: `68%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `15%`.
- Free-boundary mirror lane: `5%`.
- PR merge readiness overall: `60%`.

### User input needed

No user input is needed for the next lane.

---

## 45. 2026-06-17 M8m finite-current residual decomposition gate

This lane moves the residual-Newton convergence example from the vacuum
two-coil benchmark into a finite-current, pitched-field diagnostic while
keeping the CI cost low.  It is a diagnostic gate, not yet a claim that the
finite-current residual-Newton solve reaches tight tolerances.

### Steps taken

- Extended `examples/mirror_residual_newton_convergence_grid.py` with:
  - `--i-prime` for finite-current runs;
  - `--case-label` for stable selected-output folders;
  - JSON fields for `i_prime_value`, a simple
    `twist_proxy_i_prime_over_psi_prime`, and a `finite_current` flag.
- Kept the default case backward compatible: with `--i-prime 0`, selected
  artifacts still use `best_two_coil_residual_newton` and
  `highest_budget_two_coil_residual_newton`.
- Added a finite-current smoke test in `tests/mirror/test_mirror_examples.py`
  that verifies:
  - nonzero current is recorded in the JSON schema;
  - the adaptive residual-linear budget is exercised;
  - lambda residuals become nonzero in the finite-current case.
- Updated `examples/mirror/README.md` to document the new finite-current
  diagnostic mode and its current convergence limitations.
- Generated a finite-current artifact at:
  `results/mirror/finite_current_residual_newton_diagnostic/`.

### Results obtained

The diagnostic command was:

```bash
JAX_ENABLE_X64=1 python examples/mirror_residual_newton_convergence_grid.py \
  --outdir results/mirror/finite_current_residual_newton_diagnostic \
  --ns-array 5 \
  --nxi-array 9 \
  --maxiter-array 12 \
  --residual-linear-maxiter-array 16 \
  --residual-linear-maxiter-policy adaptive \
  --i-prime 0.01 \
  --case-label finite_current_two_coil \
  --preconditioners radial_xi_tridi
```

Key row metrics:

- `optimizer_success`: `False`.
- `optimizer_accepted`: `True`.
- `optimizer_nit`: `12`.
- `optimizer_njev`: `648`.
- adaptive inner budget effective maximum: `54`.
- initial residual norm: `9.412907913e-02`.
- final residual norm: `1.323339293e-03`.
- final `fsq`: `2.870863745e-08`.
- final normalized force: `7.447138271e-03`.
- residual `a` norm: `3.449810942e-04`.
- residual lambda norm: `1.277581672e-03`.
- residual lambda fraction: `0.9654226085`.
- cap lambda norm: `7.111363246e-04`.
- interior-xi lambda norm: `7.425365658e-04`.
- twist proxy `i_prime / psi_prime`: `1.716805350`.
- mirror ratio: `19.88051325`.

Interpretation:

- The finite-current case reduces the residual by about `71x`, but it does
  not reach tight residual tolerance in this low-resolution, 12-iteration
  diagnostic run.
- The remaining residual is lambda dominated, split between cap and interior
  xi contributions.  This confirms that the finite-current lane is exercising
  the pitch/lambda physics path rather than silently behaving like the vacuum
  case.
- A higher-budget probe at `i_prime=1e-3`, `maxiter=24`, adaptive inner budget
  factor `6`, reduced the residual to `1.267102629e-04` but remained
  iteration-limited and was too slow to use as a routine gate.  This reinforces
  the next-step need for solver-file simplification and stronger finite-current
  preconditioning before claiming tight finite-current convergence.
- The rendered figures use horizontal `z` orientation.  Cross sections remain
  circular and poloidally symmetric in the two-coil benchmark, as expected.
  Field lines are visible in the B-direction plot and on the 3D boundary plot.

Generated plot bundle:

- `results/mirror/finite_current_residual_newton_diagnostic/residual_newton_convergence_history.png`.
- `results/mirror/finite_current_residual_newton_diagnostic/residual_newton_convergence_components.png`.
- `results/mirror/finite_current_residual_newton_diagnostic/residual_newton_convergence_budget.png`.
- `results/mirror/finite_current_residual_newton_diagnostic/residual_newton_convergence_resolution_heatmap.png`.
- `results/mirror/finite_current_residual_newton_diagnostic/best_finite_current_two_coil_residual_newton/figures/best_finite_current_two_coil_residual_newton_mirror_boundary_3d.png`.
- `results/mirror/finite_current_residual_newton_diagnostic/best_finite_current_two_coil_residual_newton/figures/best_finite_current_two_coil_residual_newton_mirror_bfield_boundary.png`.
- `results/mirror/finite_current_residual_newton_diagnostic/best_finite_current_two_coil_residual_newton/figures/best_finite_current_two_coil_residual_newton_mirror_cross_sections.png`.

### How it was tested

Commands run:

```bash
JAX_ENABLE_X64=1 python -m pytest -q \
  tests/mirror/test_mirror_examples.py::test_root_residual_newton_convergence_grid_finite_current_reports_lambda_residual
JAX_ENABLE_X64=1 python -m pytest -q \
  tests/mirror/test_mirror_examples.py::test_root_residual_newton_convergence_grid_runs_without_plots \
  tests/mirror/test_mirror_examples.py::test_root_residual_newton_convergence_grid_finite_current_reports_lambda_residual
ruff check examples/mirror_residual_newton_convergence_grid.py tests/mirror/test_mirror_examples.py
ruff format --check examples/mirror_residual_newton_convergence_grid.py tests/mirror/test_mirror_examples.py
JAX_ENABLE_X64=1 python -m pytest -q tests/mirror tests/test_packaging_metadata.py
```

Passing results:

- finite-current example smoke: `1 passed`;
- default plus finite-current convergence-grid slice: `2 passed`;
- edited-file lint and format checks: passed;
- full mirror/package smoke: `89 passed, 1 skipped`.

The finite-current diagnostic images were rendered and visually inspected for
horizontal geometry orientation, field-line visibility, and cross-section
symmetry.

### File structure and best-practice notes

- The finite-current option lives in the root example that already owns the
  residual-Newton convergence-grid report, avoiding a duplicate example file.
- The new CLI arguments are small data knobs and do not change solver APIs.
- The JSON schema additions are explicit scalar diagnostics, which keeps the
  result file easy to parse in downstream notebooks and CI checks.
- The smoke test is intentionally low resolution (`ns=5`, `nxi=9`,
  `maxiter=1`) so it verifies the finite-current residual path without turning
  CI into a solve-quality benchmark.
- Tight finite-current convergence is kept as a solver gate, not papered over
  in plotting or examples.

### Best next steps

1. Commit and push the M8m finite-current diagnostic update to the draft PR.
2. Confirm the PR checks stay green after the new example test.
3. Start M8n: split `optimizers.py` without changing behavior so residual
   Newton, preconditioners, reduced-state packing, and solver dispatch are
   easier to test and improve.
4. After M8n, improve finite-current preconditioning and lambda gauge handling
   enough to make the `i_prime != 0` residual-Newton case converge to a tight
   tolerance at low and moderate resolution.

### Completion percentages after M8m

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `82%`.
- Fixed-boundary axisymmetric solve: `76%`.
- Residual Newton / preconditioning: `68%`.
- Two-coil and manufactured validation: `72%`.
- Finite-current pitch validation: `52%`.
- Plotting and `vmec --plot` mirror support: `77%`.
- I/O schema and docs: `69%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `15%`.
- Free-boundary mirror lane: `5%`.
- PR merge readiness overall: `62%`.

### User input needed

No user input is needed for the next lane.

---

## 46. 2026-06-17 M8n fixed-boundary optimizer file simplification

This lane implements the no-behavior-change solver-file simplification planned
in section 43.  The goal is to make the fixed-boundary solver easier to audit
and extend before adding stronger finite-current preconditioners.

### Steps taken

- Split `vmec_jax/mirror/solvers/fixed_boundary/optimizers.py` into focused
  support modules:
  - `types.py`: `OptimizerOptions`, `OptimizerStep`, `OptimizerRun`, and
    `_CandidateDiagnostics`;
  - `reduced.py`: reduced-coordinate masks, packing/unpacking, bounds,
    coordinate scaling, and reduced energy/gradient helpers;
  - `preconditioners.py`: residual-preconditioner key parsing, tridiagonal
    smoothers, and adaptive residual-linear-budget policy;
  - `optimizers.py`: high-level projected-gradient, L-BFGS-B,
    residual-Newton, candidate acceptance, and run-summary dispatch.
- Preserved compatibility by re-exporting the existing optimizer helper names
  from `optimizers.py`, including private names currently used by tests.
- Updated `api.py` so `MirrorSolveOptions` imports `OptimizerOptions` directly
  from `types.py` instead of from the larger optimizer dispatch module.
- Updated `docs/code_structure.rst` and the early package-structure block in
  this plan to show the new fixed-boundary solver files.
- Regenerated a lightweight solver-comparison artifact at:
  `results/mirror/solver_comparison_refactor_m8n/`.

### Results obtained

- `optimizers.py` shrank from `1337` lines to `841` lines.
- New support modules:
  - `reduced.py`: `324` lines;
  - `preconditioners.py`: `143` lines;
  - `types.py`: `81` lines.
- The refactor keeps solver behavior unchanged in the checked cases:
  - cylinder residual Newton:
    `1.523117825e-02 -> 5.610231716e-17`;
  - manufactured residual Newton:
    `9.638224761e-03 -> 3.109010554e-15`;
  - cylinder L-BFGS-B:
    `1.523117825e-02 -> 4.342638898e-07`;
  - short cylinder gradient descent:
    `1.523117825e-02 -> 1.248964815e-02`.
- The regenerated residual-history and 3D cylinder plots render correctly, with
  horizontal `z` geometry and visible standard mirror plot content.
- Mirror source coverage remains at `95%` after adding the new modules.

### How it was tested

Commands run:

```bash
ruff format vmec_jax/mirror/solvers/fixed_boundary/optimizers.py \
  vmec_jax/mirror/solvers/fixed_boundary/reduced.py \
  vmec_jax/mirror/solvers/fixed_boundary/preconditioners.py \
  vmec_jax/mirror/solvers/fixed_boundary/types.py
ruff check vmec_jax/mirror/solvers/fixed_boundary/optimizers.py \
  vmec_jax/mirror/solvers/fixed_boundary/reduced.py \
  vmec_jax/mirror/solvers/fixed_boundary/preconditioners.py \
  vmec_jax/mirror/solvers/fixed_boundary/types.py \
  vmec_jax/mirror/solvers/fixed_boundary/api.py
ruff format --check vmec_jax/mirror/solvers/fixed_boundary/optimizers.py \
  vmec_jax/mirror/solvers/fixed_boundary/reduced.py \
  vmec_jax/mirror/solvers/fixed_boundary/preconditioners.py \
  vmec_jax/mirror/solvers/fixed_boundary/types.py \
  vmec_jax/mirror/solvers/fixed_boundary/api.py
JAX_ENABLE_X64=1 python -m pytest -q \
  tests/mirror/test_mirror_low_level_coverage.py \
  tests/mirror/test_mirror_fixed_boundary_axisym.py
JAX_ENABLE_X64=1 python -m pytest -q \
  tests/mirror/test_mirror_fixed_boundary_3d.py \
  tests/mirror/test_mirror_examples.py::test_root_residual_newton_convergence_grid_finite_current_reports_lambda_residual
JAX_ENABLE_X64=1 python examples/mirror_solver_comparison.py \
  --outdir results/mirror/solver_comparison_refactor_m8n \
  --cases cylinder,manufactured \
  --maxiter-gd 4 \
  --maxiter-lbfgs 20 \
  --maxiter-newton 8 \
  --residual-linear-maxiter 16 \
  --residual-linear-maxiter-policy adaptive
LC_ALL=C.UTF-8 LANG=C.UTF-8 SPHINX_FAST=1 python -m sphinx -W -j auto -b html docs docs/_build/html
JAX_ENABLE_X64=1 python -m pytest -q tests/mirror tests/test_packaging_metadata.py
ruff check .
JAX_ENABLE_X64=1 coverage erase
JAX_ENABLE_X64=1 coverage run --source=vmec_jax -m pytest -q tests/mirror tests/test_packaging_metadata.py
coverage report --skip-covered --include='vmec_jax/mirror/*'
coverage report --skip-covered --include='vmec_jax/mirror/solvers/fixed_boundary/*'
```

Passing results:

- low-level plus axisymmetric fixed-boundary slice: `16 passed`;
- 3D fixed-boundary plus finite-current example smoke: `5 passed`;
- full mirror/package smoke: `89 passed, 1 skipped`;
- instrumented mirror/package coverage run: `89 passed, 1 skipped`;
- mirror source coverage: `95%`;
- fixed-boundary solver-module coverage: `91%`;
- fast docs build: passed with warnings as errors;
- `ruff check .`: passed;
- focused format checks: passed.

### File structure and best-practice notes

- This is intentionally a behavior-neutral split.  The high-level optimizer
  functions still live in `optimizers.py` and call the same reduced-coordinate,
  preconditioner, and residual-Newton logic as before.
- New modules follow the existing fixed-boundary solver package instead of
  creating root-level mirror helper files.
- The reduced-coordinate module explains the packed vector in plain terms:
  interior radius nodes plus gauge-fixed lambda nodes.
- The preconditioner module is now isolated, which is the right place to add
  cap-aware and finite-current lambda preconditioning next.
- Compatibility re-exports keep existing tests and downstream imports working
  while allowing future code to import directly from the focused modules.

### Best next steps

1. Commit and push the M8n simplification to the draft PR.
2. Confirm GitHub CI and combined coverage stay green.
3. Start M8o: use the simplified preconditioner module to improve
   finite-current residual-Newton convergence, focusing on the lambda-dominated
   residual from M8m.
4. Add a tight, low-resolution finite-current convergence gate only after the
   solver actually reaches tolerance without excessive inner iterations.

### Completion percentages after M8n

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `82%`.
- Fixed-boundary axisymmetric solve: `78%`.
- Residual Newton / preconditioning: `70%`.
- Two-coil and manufactured validation: `73%`.
- Finite-current pitch validation: `52%`.
- Plotting and `vmec --plot` mirror support: `77%`.
- I/O schema and docs: `71%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `15%`.
- Free-boundary mirror lane: `5%`.
- PR merge readiness overall: `64%`.

### User input needed

No user input is needed for the next lane.

---

## 47. 2026-06-17 M8o finite-current lambda-xi preconditioner gate

This lane starts converting the finite-current residual-Newton diagnostic from
scaffolded behavior into a research-grade solve path by targeting the dominant
lambda residual identified in M8m. The default residual-Newton preconditioner is
unchanged; the new behavior is opt-in for finite-current mirror probes.

### Steps taken

- Added an opt-in residual preconditioner mode,
  `radial_xi_lambda_xi_tridi`, in
  `vmec_jax/mirror/solvers/fixed_boundary/preconditioners.py`.
- Kept the existing `radial_xi_tridi` behavior intact and extended the
  smoother only when the new mode is requested:
  - radius `a` is smoothed radially and along open `xi`;
  - lambda is smoothed radially and, in the new mode, along open `xi`.
- Exposed the new mode in the residual-Newton CLI choices for:
  - `examples/mirror_residual_newton_convergence_grid.py`;
  - `examples/mirror_fixed_boundary_solve_diagnostic.py`;
  - `examples/mirror_solver_comparison.py`.
- Updated the finite-current example test to run the new preconditioner with
  `--residual-xi-alpha 1.0` and verify the JSON schema records both settings.
- Updated `examples/mirror/README.md` with the recommended finite-current
  diagnostic invocation.
- Rendered the best finite-current plot bundle at:
  `results/mirror/m8o_lambda_xi_best_plots/`.

### Results obtained

The M8m baseline finite-current diagnostic used
`radial_xi_tridi`, `residual_xi_alpha=0.2`, `i_prime=0.01`, `ns=5`,
`nxi=9`, `maxiter=12`, adaptive inner `lsmr`, and reached:

- final residual norm: `1.323339292922e-03`;
- final `fsq`: `2.870863744577e-08`;
- final normalized force: `7.447138270591e-03`;
- residual `a` norm: `3.449810941780e-04`;
- residual lambda norm: `1.277581672087e-03`;
- lambda residual fraction: `0.9654226085`.

A short `residual_lambda_alpha` sweep confirmed that changing the existing
radial lambda smoother alone did not solve the finite-current bottleneck. The
best point in that sweep was still the existing default
`residual_lambda_alpha=0.5`.

The new lambda-xi smoother produced:

- with `residual_xi_alpha=0.2`:
  - final residual norm: `9.369679625918e-04`;
  - final `fsq`: `1.439195021186e-08`;
  - final normalized force: `5.273070166859e-03`;
  - residual `a` norm: `2.347639373661e-04`;
  - residual lambda norm: `9.070804025200e-04`.
- with `residual_xi_alpha=1.0`:
  - final residual norm: `1.037065207446e-04`;
  - final `fsq`: `1.763121712288e-10`;
  - final normalized force: `5.836719181237e-04`;
  - residual `a` norm: `7.912766682054e-05`;
  - residual lambda norm: `6.703621997469e-05`;
  - lambda residual fraction: `0.6464031335`;
  - residual reduction factor from the initial state: `1.101747958e-03`;
  - mirror ratio: `19.61011454`.

Interpretation:

- The new finite-current mode improves the final projected residual by about
  `12.8x` relative to the M8m baseline at the same outer and inner budgets.
- The final `fsq` improves by about `163x`, and the normalized force improves
  by about `12.8x`.
- The residual is no longer overwhelmingly lambda dominated, which means the
  preconditioner is addressing the bottleneck found in M8m.
- The solve is still iteration-limited (`optimizer_success=False`,
  `optimizer_message="maximum iterations reached"`), so this is a solver
  progress gate, not the final finite-current convergence claim.

Generated plot bundle:

- `results/mirror/m8o_lambda_xi_best_plots/residual_newton_convergence_history.png`.
- `results/mirror/m8o_lambda_xi_best_plots/residual_newton_convergence_components.png`.
- `results/mirror/m8o_lambda_xi_best_plots/residual_newton_convergence_budget.png`.
- `results/mirror/m8o_lambda_xi_best_plots/residual_newton_convergence_resolution_heatmap.png`.
- `results/mirror/m8o_lambda_xi_best_plots/best_finite_current_lambda_xi_residual_newton/figures/best_finite_current_lambda_xi_residual_newton_mirror_boundary_3d.png`.
- `results/mirror/m8o_lambda_xi_best_plots/best_finite_current_lambda_xi_residual_newton/figures/best_finite_current_lambda_xi_residual_newton_mirror_bfield_boundary.png`.
- `results/mirror/m8o_lambda_xi_best_plots/best_finite_current_lambda_xi_residual_newton/figures/best_finite_current_lambda_xi_residual_newton_mirror_bmag_boundary.png`.
- `results/mirror/m8o_lambda_xi_best_plots/best_finite_current_lambda_xi_residual_newton/figures/best_finite_current_lambda_xi_residual_newton_mirror_bmag_sxi.png`.
- `results/mirror/m8o_lambda_xi_best_plots/best_finite_current_lambda_xi_residual_newton/figures/best_finite_current_lambda_xi_residual_newton_mirror_cross_sections.png`.

The rendered figures were visually inspected. The mirror is horizontal with
`z` as the horizontal axis, cross sections remain circular and poloidally
symmetric, field-line traces are visible over the B-direction arrows, and
`|B|` is weakest near the center and strongest near the end caps.

### How it was tested

Commands run:

```bash
JAX_ENABLE_X64=1 python examples/mirror_residual_newton_convergence_grid.py \
  --outdir results/mirror/m8o_lambda_xi_best_plots \
  --ns-array 5 \
  --nxi-array 9 \
  --maxiter-array 12 \
  --residual-linear-maxiter-array 16 \
  --residual-linear-maxiter-policy adaptive \
  --residual-xi-alpha 1.0 \
  --i-prime 0.01 \
  --case-label finite_current_lambda_xi \
  --preconditioners radial_xi_lambda_xi_tridi
JAX_ENABLE_X64=1 python -m pytest -q \
  tests/mirror/test_mirror_fixed_boundary_axisym.py::test_reduced_residual_preconditioner_preserves_axisym_layout_and_damps_high_frequency_vector \
  tests/mirror/test_mirror_examples.py::test_root_residual_newton_convergence_grid_finite_current_reports_lambda_residual
ruff check \
  vmec_jax/mirror/solvers/fixed_boundary/preconditioners.py \
  examples/mirror_residual_newton_convergence_grid.py \
  examples/mirror_fixed_boundary_solve_diagnostic.py \
  examples/mirror_solver_comparison.py \
  tests/mirror/test_mirror_fixed_boundary_axisym.py \
  tests/mirror/test_mirror_examples.py
ruff format --check \
  vmec_jax/mirror/solvers/fixed_boundary/preconditioners.py \
  examples/mirror_residual_newton_convergence_grid.py \
  examples/mirror_fixed_boundary_solve_diagnostic.py \
  examples/mirror_solver_comparison.py \
  tests/mirror/test_mirror_fixed_boundary_axisym.py \
  tests/mirror/test_mirror_examples.py
JAX_ENABLE_X64=1 python -m pytest -q tests/mirror tests/test_packaging_metadata.py
LC_ALL=C.UTF-8 LANG=C.UTF-8 SPHINX_FAST=1 python -m sphinx -W -j auto -b html docs docs/_build/html
ruff check .
git diff --check
JAX_ENABLE_X64=1 python -m coverage run --timid -m pytest -q tests/mirror tests/test_packaging_metadata.py
python -m coverage report --include='vmec_jax/mirror/*' --show-missing
```

Passing results:

- focused preconditioner plus finite-current example tests: `2 passed`;
- full mirror/package smoke: `89 passed, 1 skipped`;
- pure-Python-tracer mirror/package coverage run: `89 passed, 1 skipped`;
- mirror source coverage: `95%`;
- fast docs build: passed with warnings as errors;
- `ruff check .`: passed;
- `git diff --check`: passed;
- edited Python-file lint: passed;
- edited Python-file format check: passed.

Note: `python -m pytest --cov=vmec_jax.mirror ...` aborted locally with exit
code `134` before pytest emitted output. Running Coverage with `--timid`
avoided that C-tracer path and completed successfully.

### File structure and best-practice notes

- The new mode lives beside the existing residual preconditioner helpers in
  `preconditioners.py`, after the M8n file split. This keeps solver dispatch,
  reduced-state packing, and preconditioner policy separated.
- The default solver path is unchanged, which keeps existing two-coil and
  manufactured benchmarks stable.
- The CLI examples share the same allowed preconditioner vocabulary, so
  diagnostic scripts do not drift.
- The example test remains a low-cost schema and path check rather than a
  high-resolution convergence benchmark.
- The implementation uses the existing symmetric tridiagonal smoother instead
  of adding a new linear algebra dependency or a second smoothing kernel.

### Best next steps

1. Commit and push the M8o preconditioner gate to the draft PR and confirm CI.
2. Start M8p: run a higher-budget convergence study for
   `radial_xi_lambda_xi_tridi` over outer iterations and inner adaptive
   budgets to determine whether the current method reaches `1e-8` to `1e-10`
   residuals or needs a stronger block preconditioner.
3. If still iteration-limited, implement a block-structured residual
   preconditioner or Schur-style lambda/radius coupling using the simplified
   M8n file structure.
4. Promote the best finite-current run into a documented benchmark only after
   the residual, `fsq`, normalized force, and cross-section diagnostics all
   converge under resolution refinement.

### Completion percentages after M8o

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `82%`.
- Fixed-boundary axisymmetric solve: `79%`.
- Residual Newton / preconditioning: `74%`.
- Two-coil and manufactured validation: `73%`.
- Finite-current pitch validation: `58%`.
- Plotting and `vmec --plot` mirror support: `78%`.
- I/O schema and docs: `72%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `15%`.
- Free-boundary mirror lane: `5%`.
- PR merge readiness overall: `66%`.

### User input needed

No user input is needed for the next lane.

---

---

## 48. 2026-06-17 stellarator-mirror hybrid and ESSOS mirror free-boundary planning lane

This section adds two longer-horizon lanes requested after M8o:

1. A stellarator-mirror hybrid fixed-boundary geometry: a straight-axis mirror
   whose cross section is a rotating ellipse over one field-period-like axial
   module, then repeated with mirror/up-down symmetry.
2. A mirror-specific free-boundary example in which circular coils generated
   through ESSOS-compatible coil objects provide the vacuum field and
   `vmec_jax` solves for the mirror LCFS over beta targets of `1%`, `3%`, and
   `10%`.

These lanes should be added after the current finite-current residual-Newton
convergence work, because they need reliable 3D fixed-boundary and
free-boundary infrastructure.  They should not block M8p/M8q unless the
finite-current solve reveals geometry limitations that the hybrid geometry
lane would naturally solve.

### Sources and code checked

- Current mirror geometry state:
  - `vmec_jax/mirror/core/boundary.py`;
  - `vmec_jax/mirror/core/state.py`;
  - `vmec_jax/mirror/kernels/geometry.py`.
- Current external-field and ESSOS bridge:
  - `vmec_jax/external_fields/coils_jax.py`;
  - `vmec_jax/external_fields/essos_adapter.py`;
  - `examples/free_boundary_direct_coils_forward.py`;
  - `examples/free_boundary_essos_coils_forward.py`;
  - `examples/free_boundary_essos_coils_beta_scan.py`;
  - `examples/free_boundary_essos_example_common.py`.
- ESSOS upstream source, inspected at commit
  `d9ca5c37dc063f98a0c6b092da6024ba256b4468`:
  - `README.md`: ESSOS is a JAX coil, particle/field-line tracing, and
    optimization package with Fourier coil curves and stellarator surfaces;
  - `essos/coils.py`: `Curves`, `Coils`, `Coils_from_json`,
    `CreateEquallySpacedCurves`, Fourier DOF convention, symmetry expansion,
    and JSON serialization;
  - `essos/surfaces.py`: `SurfaceRZFourier` and B-dot-normal helpers.

### Architecture conclusions

- The current mirror 3D representation is a straight-axis, star-shaped
  cylindrical-radius model:
  `r(s, theta, xi) = sqrt(s) * a(s, theta, xi)`.
- A centered rotating ellipse on a straight axis can be represented exactly in
  this model by a positive polar radius

  `r_b(theta, xi) = A(xi) B(xi) / sqrt((B(xi) cos(theta-alpha(xi)))^2 + (A(xi) sin(theta-alpha(xi)))^2)`.

- Therefore the first hybrid lane does not need a full Cartesian
  `X(s,theta,xi),Y(s,theta,xi),Z(s,theta,xi)` surface map if the ellipse stays
  centered on the linear mirror axis.
- A later Cartesian transverse-frame map will be needed if the hybrid should
  support off-axis cross-section centers, non-star-shaped sections, or a curved
  stellarator axis.
- The existing ESSOS adapter already consumes the attributes needed from an
  ESSOS `Coils` object: `dofs_curves`, `dofs_currents`, `currents_scale`,
  `n_segments`, `nfp`, and `stellsym`.  The mirror free-boundary lane should
  reuse this bridge rather than adding ESSOS imports to core vmec_jax modules.
- ESSOS `CreateEquallySpacedCurves` builds toroidal stellarator-style coils.
  Circular mirror end coils are a different object, so vmec_jax should add a
  small ESSOS-compatible circular-loop builder and optionally upstream it to
  ESSOS later.

### Proposed plan insertion

Keep the current plan order through M8p/M8q:

- M8p: higher-budget finite-current residual-Newton convergence study for
  `radial_xi_lambda_xi_tridi`.
- M8q: block or Schur-style residual preconditioner if M8p remains
  iteration-limited.
- M8r: promote finite-current two-coil convergence only after residual, `fsq`,
  normalized force, field-line pitch, and cross-section diagnostics converge
  under resolution refinement.

Add the hybrid and ESSOS lanes after that:

- M13: rotating-ellipse fixed-boundary geometry.
- M14: stellarator-mirror hybrid fixed-boundary example and diagnostics.
- M15: mirror free-boundary direct-coil formulation.
- M16: ESSOS circular-coil mirror beta-scan example at `1%`, `3%`, and `10%`.
- M17: documentation, plotting, and validation promotion for hybrid/free-boundary
  mirror examples.

The existing M9-M12 lanes remain valid.  If ordering pressure appears, M13-M17
can run after M12 because the hybrid/free-boundary work depends on robust
mirror plotting, output, and solver diagnostics more than on mirror-Boozer-like
coordinates.

### M13: rotating-ellipse fixed-boundary geometry

Implementation:

- Add a `MirrorBoundary.rotating_ellipse(...)` constructor.
- Add a focused data container, likely `RotatingEllipseBoundaryParams`, with:
  - axial semi-major profile `A(xi)`;
  - axial semi-minor profile `B(xi)`;
  - rotation profile `alpha(xi)`;
  - optional throat/cap scale profile for mirror-length modulation;
  - optional symmetry metadata.
- Support constant, polynomial, and tabulated/Chebyshev nodal profiles first.
- Implement the exact polar radius formula in `radius_on_grid_3d`.
- Keep the first implementation centered on the straight `z` axis.
- Validate positivity and avoid hidden self-intersections by checking
  `min(radius)`, `min(sqrtg)`, and cross-section orientation.

Tests:

- Unit tests for exact ellipse radius at `alpha=0`, `alpha=pi/2`, and a
  tabulated rotation profile.
- Geometry tests comparing numerical cross-section area against `pi*A*B`.
- Volume test against `integral pi*A(xi)*B(xi) dz`.
- Symmetry tests for the requested repeated/up-down operation.
- Plot tests that render horizontal-`z` 3D geometry with rotating cross-section
  contours.

Diagnostics and plots:

- 3D boundary colored by `|B|`.
- Cross sections at several `z` planes showing the ellipse rotation.
- Rotation-angle profile `alpha(z)`.
- `min(sqrtg)`, volume, mirror ratio, residual history, and force components.

Acceptance:

- The fixed-boundary solver accepts the rotating-ellipse boundary without
  negative radius or negative Jacobian.
- Cross-section plots visibly rotate by the requested field-period angle.
- Analytic area/volume checks pass at low and moderate resolution.

### M14: stellarator-mirror hybrid fixed-boundary example

Implementation:

- Add a root example, tentatively
  `examples/mirror_stellarator_hybrid_rotating_ellipse.py`.
- Use a straight-axis mirror length and one axial field-period-like module.
- Build a rotating ellipse connected to stronger-field mirror end-cap regions.
- Add a repeat/symmetry visualization mode:
  - solve on the fundamental module;
  - expand repeated modules only for plots and diagnostics unless the solver
    needs the full repeated domain.
- Include finite-current settings so field-line pitch is visible.

Tests:

- Low-resolution smoke test with `--no-plots`.
- Plot-render test for:
  - 3D geometry;
  - field-line overlay;
  - `|B|`;
  - cross sections;
  - residual history.
- Regression JSON schema test for rotation angle, axis length, mirror ratio,
  finite-current settings, and symmetry metadata.

Acceptance:

- The example writes a `mout_*.nc` file and a full plot bundle.
- Field lines show both axial mirror propagation and helical/rotating-ellipse
  pitch.
- The geometry remains centered on the linear axis unless the user explicitly
  asks for off-axis cross-section centers.

### M15: mirror free-boundary direct-coil formulation

Implementation:

- Add mirror-specific direct circular coil helpers:
  - `make_mirror_circular_coil_params`;
  - `make_symmetric_mirror_end_coils`;
  - optional ESSOS-compatible object or JSON writer exposing ESSOS-style
    `dofs_curves`, `dofs_currents`, `currents_scale`, `n_segments`, `nfp`, and
    `stellsym`.
- Reuse `CoilFieldParams` and `from_essos_coils` for field sampling.
- Define mirror-LCFS unknowns in the mirror solver rather than toroidal VMEC
  `RBC/ZBS` coefficients:
  - fixed seed from two-coil or rotating-ellipse boundary;
  - free-boundary update driven by normal-field residual;
  - positive-radius and positive-Jacobian guards.
- Start with direct coils only; add mgrid export later only if needed for
  VMEC2000-style replay.
- Keep differentiability through coil field sampling; treat the full
  free-boundary solved-state derivative as a later implicit/adjoint lane.

Tests:

- Circular coil field checks against the existing two-coil analytic field on
  axis.
- Direct-coil provider shape, current, and symmetry tests.
- Dry-run example test that writes inputs/JSON without solving.
- Low-resolution free-boundary smoke with tiny iteration count once the mirror
  LCFS update is implemented.

Acceptance:

- Vacuum circular-coil mirror free-boundary run moves the LCFS in the expected
  direction and reports normal-field residuals.
- The run writes mirror-native `mout`, not `wout`, unless it intentionally uses
  the toroidal compatibility path.

### M16: ESSOS circular-coil mirror beta scan

Implementation:

- Add root example, tentatively
  `examples/mirror_free_boundary_essos_circular_coils_beta_scan.py`.
- Support beta targets `1`, `3`, and `10` percent by default.
- Use pressure continuation:
  - vacuum or near-vacuum seed;
  - `1%`;
  - `3%`;
  - `10%`.
- Warm-start each beta from the previous accepted LCFS.
- Write per-beta JSON summaries and one aggregate summary.
- For each beta, write:
  - `mout_beta_001.nc`, `mout_beta_003.nc`, `mout_beta_010.nc`;
  - 3D LCFS with coils;
  - cross sections;
  - `|B|`;
  - residual history;
  - normal-field residual on the LCFS;
  - pressure and beta diagnostics.

Diagnostics:

- Requested beta percent.
- Achieved beta proxy and pressure normalization.
- Coil current scale and coil geometry scale.
- LCFS volume, mirror ratio, min radius, min Jacobian.
- Free-boundary normal-field residual RMS/max.
- Fixed-boundary force residual/`fsq` after each accepted beta.

Tests:

- Dry-run schema test for default beta list `[1, 3, 10]`.
- A single low-cost beta smoke with `--betas 1 --maxiter 1 --no-plots`.
- Optional slow test for the full `[1,3,10]` scan behind an existing slow or
  physics marker.

Acceptance:

- The `1%` and `3%` cases run reliably at low/moderate resolution.
- The `10%` case either converges with pressure continuation or reports a
  clear controlled failure with the last accepted LCFS and residual metrics.
- Plots show the circular coils, the mirror LCFS, and beta-dependent LCFS
  changes.

### M17: documentation and promotion

Implementation:

- Document the hybrid boundary formula, symmetry convention, and limitations.
- Document the free-boundary circular-coil workflow and how it differs from the
  existing LP-QA toroidal ESSOS examples.
- Add example README entries with exact commands.
- Add plan checkpoints and promotion criteria.

Acceptance:

- Docs build with warnings as errors.
- Root examples have smoke tests.
- Full examples are reproducible from a clean checkout with optional ESSOS
  installation.

### Design questions for the user

The following choices would improve the plan, but the implementation can start
with the default assumptions below if no answer is available immediately.

1. Should the rotating ellipse stay centered on the straight mirror axis, or
   should the cross-section center be allowed to move off axis?
   Default: centered on the straight axis.
2. What should one "field period" mean for the rotating ellipse: `180` degrees,
   `360` degrees, or a user-selected rotation angle over one mirror module?
   Default: user-selected, with examples using `180` degrees and `360` degrees.
3. What exact up/down symmetry should be enforced for the repeated modules?
   Default: solve one module and generate plot/output repeats by reflecting
   `z -> -z` while preserving a positive Jacobian and matching ellipse
   orientation continuously.
4. Should the hybrid mirror be a finite open mirror with physical end-cap
   constraints, or a periodic axial module repeated indefinitely for
   diagnostics?
   Default: finite open mirror solve, repeated only for visualization.
5. For the ESSOS circular-coil free-boundary example, should "circular coils
   put by ESSOS" mean only two or more end coils, or should ESSOS also place
   body/helical coils around the rotating-ellipse section?
   Default: two symmetric end coils first; optional body/helical coils later.
6. Should beta targets `1%`, `3%`, and `10%` mean achieved equilibrium beta
   (`wp/wb`-style proxy) or nominal pressure input scale?
   Default: target achieved beta proxy with pressure-scale iteration if needed.
7. Should the first free-boundary mirror example be direct-coils only, or also
   generate mgrid files for replay?
   Default: direct coils first; mgrid export as a compatibility add-on.

### Completion-percentage impact

- Geometry/grids/bases remain `90%` for the current MVP, but the new hybrid
  geometry lane starts at `10%`.
- Fixed-boundary 3D mirror solve remains `79%` for current radial boundaries,
  and starts at `20%` for rotating-ellipse hybrid validation.
- Free-boundary mirror lane remains `5%`; the ESSOS circular-coil beta scan
  starts at `0%` until M15 begins.
- PR merge readiness overall remains `66%` for the current MVP because these
  are new post-M8 scope additions.

---

## 49. 2026-06-17 M8p finite-current high-budget convergence study

This lane answers whether the M8o `radial_xi_lambda_xi_tridi` preconditioner
can reach tight finite-current residuals by budget alone.  The answer is:
more inner and outer iterations help substantially, but the solve is still
iteration-limited and the remaining residual becomes radius/interior-`xi`
dominated.  M8q should therefore improve the residual-Newton correction, not
only increase budgets.

### Steps taken

- Ran fixed-budget finite-current residual-Newton grids at `ns=5`, `nxi=9`,
  `i_prime=0.01`, `residual_xi_alpha=1.0`, and
  `radial_xi_lambda_xi_tridi`.
- Separated inner linear budget effects from adaptive policy by using fixed
  `residual_linear_maxiter` values.
- Probed:
  - outer iterations `12`, `24`, `36` with inner budgets `16` and `54`;
  - outer iterations `24`, `36` with inner budget `96`;
  - one final outer `48`, inner `96` best-row probe.
- Regenerated the best-row plot bundle with the same `48 x 96` settings.
- Visually inspected residual history, residual components, horizontal-`z`
  3D geometry, field-line overlay, `|B|`, cross sections, and Jacobian plots.

### Results obtained

Baseline comparisons:

- M8m finite-current baseline, `radial_xi_tridi`, adaptive effective inner
  budget `54`, `maxiter=12`:
  - final residual: `1.323339292922e-03`;
  - final `fsq`: `2.870863744577e-08`;
  - normalized force: `7.447138270591e-03`;
  - lambda fraction: `0.9654226085`.
- M8o finite-current lambda-xi mode, adaptive effective inner budget `54`,
  `maxiter=12`:
  - final residual: `1.037065207446e-04`;
  - final `fsq`: `1.763121712288e-10`;
  - normalized force: `5.836719181237e-04`;
  - lambda fraction: `0.6464031335`.

M8p fixed-budget rows:

| outer | inner | final residual | final fsq | normalized force | a norm | lambda norm | lambda fraction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 16 | `1.956298364328e-02` | `6.273939820120e-06` | `1.089937416914e-01` | `1.172272798243e-02` | `1.566167225034e-02` | `0.800577` |
| 24 | 16 | `1.226847725838e-02` | `2.467467774418e-06` | `6.875493381846e-02` | `8.555272081116e-03` | `8.793341450072e-03` | `0.716743` |
| 36 | 16 | `8.866479842778e-03` | `1.288761718072e-06` | `4.978481653739e-02` | `6.382820203061e-03` | `6.154191340687e-03` | `0.694096` |
| 12 | 54 | `1.037065207446e-04` | `1.763121712288e-10` | `5.836719181237e-04` | `7.912766682054e-05` | `6.703621997469e-05` | `0.646403` |
| 24 | 54 | `4.332436431347e-05` | `3.077050070764e-11` | `2.438347387145e-04` | `4.331968966621e-05` | `6.364199784527e-07` | `0.014690` |
| 36 | 54 | `2.236718040228e-05` | `8.201487854887e-12` | `1.258852649564e-04` | `2.235574239952e-05` | `7.152208849730e-07` | `0.031976` |
| 24 | 96 | `1.972965755648e-06` | `6.381301431084e-14` | `1.110409861971e-05` | `1.971918207663e-06` | `6.428417572186e-08` | `0.032583` |
| 36 | 96 | `2.099900437462e-07` | `7.228822700417e-16` | `1.181850294481e-06` | `2.099315512719e-07` | `4.956031992477e-09` | `0.023601` |
| 48 | 96 | `2.931165262408e-08` | `1.408480294352e-17` | `1.649696560286e-07` | `2.930288410965e-08` | `7.169117179445e-10` | `0.024458` |

Interpretation:

- Inner budget `16` is not viable for the finite-current case.
- Inner budget `54`, matching the earlier adaptive effective budget, removes
  the worst lambda bottleneck but stalls above `2e-5` by 36 outer iterations.
- Inner budget `96` improves the residual by two additional orders of
  magnitude, reaching `2.93e-8` at 48 outer iterations.
- Even at `48 x 96`, the solve does not reach `gtol=1e-12` or a robust
  `1e-8` research threshold.
- The remaining residual is radius dominated, especially interior in `xi`.
  Lambda smoothing is no longer the main bottleneck.
- High-budget rows are too expensive for routine CI.  Future CI should keep
  low-cost schema/path tests, while high-budget convergence remains a
  documented benchmark command.

Generated artifacts:

- `results/mirror/m8p_lambda_xi_fixed_budget_probe/residual_newton_convergence_grid_metrics.json`.
- `results/mirror/m8p_lambda_xi_linear96_probe/residual_newton_convergence_grid_metrics.json`.
- `results/mirror/m8p_lambda_xi_48x96_probe/residual_newton_convergence_grid_metrics.json`.
- `results/mirror/m8p_lambda_xi_best_plots/residual_newton_convergence_grid_metrics.json`.
- `results/mirror/m8p_lambda_xi_best_plots/residual_newton_convergence_history.png`.
- `results/mirror/m8p_lambda_xi_best_plots/residual_newton_convergence_components.png`.
- `results/mirror/m8p_lambda_xi_best_plots/residual_newton_convergence_budget.png`.
- `results/mirror/m8p_lambda_xi_best_plots/residual_newton_convergence_resolution_heatmap.png`.
- `results/mirror/m8p_lambda_xi_best_plots/best_finite_current_lambda_xi_m8p_48x96_residual_newton/figures/best_finite_current_lambda_xi_m8p_48x96_residual_newton_mirror_boundary_3d.png`.
- `results/mirror/m8p_lambda_xi_best_plots/best_finite_current_lambda_xi_m8p_48x96_residual_newton/figures/best_finite_current_lambda_xi_m8p_48x96_residual_newton_mirror_bfield_boundary.png`.
- `results/mirror/m8p_lambda_xi_best_plots/best_finite_current_lambda_xi_m8p_48x96_residual_newton/figures/best_finite_current_lambda_xi_m8p_48x96_residual_newton_mirror_bmag_boundary.png`.
- `results/mirror/m8p_lambda_xi_best_plots/best_finite_current_lambda_xi_m8p_48x96_residual_newton/figures/best_finite_current_lambda_xi_m8p_48x96_residual_newton_mirror_bmag_sxi.png`.
- `results/mirror/m8p_lambda_xi_best_plots/best_finite_current_lambda_xi_m8p_48x96_residual_newton/figures/best_finite_current_lambda_xi_m8p_48x96_residual_newton_mirror_cross_sections.png`.
- `results/mirror/m8p_lambda_xi_best_plots/best_finite_current_lambda_xi_m8p_48x96_residual_newton/figures/best_finite_current_lambda_xi_m8p_48x96_residual_newton_mirror_jacobian.png`.

### How it was tested

Commands run:

```bash
JAX_ENABLE_X64=1 python examples/mirror_residual_newton_convergence_grid.py \
  --outdir results/mirror/m8p_lambda_xi_fixed_budget_probe \
  --ns-array 5 \
  --nxi-array 9 \
  --maxiter-array 12,24,36 \
  --residual-linear-maxiter-array 16,54 \
  --residual-linear-maxiter-policy fixed \
  --residual-xi-alpha 1.0 \
  --i-prime 0.01 \
  --case-label finite_current_lambda_xi_m8p_fixed_budget \
  --preconditioners radial_xi_lambda_xi_tridi \
  --no-plots
JAX_ENABLE_X64=1 python examples/mirror_residual_newton_convergence_grid.py \
  --outdir results/mirror/m8p_lambda_xi_linear96_probe \
  --ns-array 5 \
  --nxi-array 9 \
  --maxiter-array 24,36 \
  --residual-linear-maxiter-array 96 \
  --residual-linear-maxiter-policy fixed \
  --residual-xi-alpha 1.0 \
  --i-prime 0.01 \
  --case-label finite_current_lambda_xi_m8p_linear96 \
  --preconditioners radial_xi_lambda_xi_tridi \
  --no-plots
JAX_ENABLE_X64=1 python examples/mirror_residual_newton_convergence_grid.py \
  --outdir results/mirror/m8p_lambda_xi_48x96_probe \
  --ns-array 5 \
  --nxi-array 9 \
  --maxiter-array 48 \
  --residual-linear-maxiter-array 96 \
  --residual-linear-maxiter-policy fixed \
  --residual-xi-alpha 1.0 \
  --i-prime 0.01 \
  --case-label finite_current_lambda_xi_m8p_48x96 \
  --preconditioners radial_xi_lambda_xi_tridi \
  --no-plots
JAX_ENABLE_X64=1 python examples/mirror_residual_newton_convergence_grid.py \
  --outdir results/mirror/m8p_lambda_xi_best_plots \
  --ns-array 5 \
  --nxi-array 9 \
  --maxiter-array 48 \
  --residual-linear-maxiter-array 96 \
  --residual-linear-maxiter-policy fixed \
  --residual-xi-alpha 1.0 \
  --i-prime 0.01 \
  --case-label finite_current_lambda_xi_m8p_48x96 \
  --preconditioners radial_xi_lambda_xi_tridi
```

Validation:

- The command outputs were parsed from JSON metrics.
- The best-row plots were visually inspected.
- The 3D geometry remains horizontal in `z`.
- Field-line traces render on top of the B-direction arrows.
- `|B|` remains weakest near the midplane and strongest near the mirror caps.
- Cross sections remain circular and poloidally symmetric for the two-coil
  benchmark.
- The Jacobian is positive in the plotted solution.

No code changed in this lane, so the validation gate is `git diff --check`
before commit rather than a full test rerun.

### File structure and best-practice notes

- This tranche uses the existing root-level convergence-grid example instead
  of adding another benchmark script.
- The generated metrics are benchmark artifacts under `results/`, not source
  files.
- The current CLI remains appropriate: low-cost tests should exercise path and
  schema, while high-budget finite-current convergence should stay explicit.
- The result directs M8q toward solver quality, not more plotting or CLI
  scaffolding.

### Best next steps

1. Commit and push the M8p plan/benchmark log.
2. Start M8q with a radius-focused correction:
   - first try a stronger radius/interior-`xi` preconditioner that can be
     tuned separately from lambda smoothing;
   - if that is insufficient, add a block-structured or Schur-style
     reduced-residual preconditioner for coupled `a`/lambda updates.
3. Re-run the same `ns=5`, `nxi=9`, `i_prime=0.01` benchmark and require a
   meaningful improvement over the `48 x 96` residual `2.93e-8`.
4. Only after M8q improves convergence per iteration, consider moderate
   resolution finite-current checks.

### Completion percentages after M8p

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `82%`.
- Fixed-boundary axisymmetric solve: `80%`.
- Residual Newton / preconditioning: `76%`.
- Two-coil and manufactured validation: `74%`.
- Finite-current pitch validation: `62%`.
- Plotting and `vmec --plot` mirror support: `78%`.
- I/O schema and docs: `72%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `15%`.
- Free-boundary mirror lane: `5%`.
- Stellarator-mirror hybrid lane: `10%`.
- ESSOS circular-coil mirror beta scan: `0%`.
- PR merge readiness overall: `67%`.

### User input needed

No user input is needed for M8q; defaults from section 48 remain in force until
overridden.

---

## 50. 2026-06-17 M8q dense reference residual-Newton correction

This lane moved the finite-current fixed-boundary solve from scaffolded
iteration-budget studies to a true tight residual solve on the small
two-coil benchmark.  The main result is that an exact dense reduced-Hessian
linear correction reaches the requested `gtol=1e-12` on the finite-current
case in five Newton iterations.  That identifies the current matrix-free
`lsmr` correction quality, not the nonlinear residual definition, as the
dominant blocker for tight finite-current convergence.

### Steps taken

- Tested a radius/interior-`xi` smoothing idea first, because M8p left a
  radius-dominated residual.
- Rejected that idea before committing it:
  - `radial_xi2_lambda_xi_tridi`, `xi_alpha=1.0`, `24 x 54` row:
    residual `1.337885508585e-03`, `fsq=2.934323990298e-08`;
  - `radial_xi2_lambda_xi_tridi`, `xi_alpha=0.5`, `24 x 54` row:
    residual `2.227816193816e-04`, `fsq=8.136336054802e-10`;
  - both were worse than M8p `radial_xi_lambda_xi_tridi`, `24 x 54`,
    residual `4.332436431347e-05`, `fsq=3.077050070764e-11`.
- Added a public residual-Newton option:
  - `residual_linear_solver="lsmr"` remains the scalable default;
  - `residual_linear_solver="dense_lstsq"` builds the scaled reduced Hessian
    and solves each Newton correction with `np.linalg.lstsq`;
  - dense solves are intended as small-grid reference/debug runs, not the
    production large-grid default.
- Threaded the option through:
  - `MirrorSolveOptions`;
  - `OptimizerOptions`;
  - optimizer summaries;
  - mirror NetCDF global attributes;
  - root example CLIs and JSON metrics.
- Ran the finite-current `ns=5`, `nxi=9`, `i_prime=0.01` two-coil benchmark
  with the dense solver and regenerated the full plot bundle.
- Visually inspected residual history, horizontal-`z` 3D geometry, coils,
  B-direction field-line overlays, `|B|`, cross sections, and Jacobian plots.

### Results obtained

Dense reference finite-current solve:

| solver | preconditioner | outer budget | inner budget field | final residual | final fsq | normalized force | Newton iterations |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense_lstsq` | `none` | 24 | 1 | `2.150747940722e-13` | `7.583142138561e-28` | `1.210467906862e-12` | 5 |

Comparison to the best M8p matrix-free rows:

| solver | preconditioner | outer | inner | final residual | final fsq | normalized force |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `lsmr` | `radial_xi_lambda_xi_tridi` | 24 | 54 | `4.332436431347e-05` | `3.077050070764e-11` | `2.438347387145e-04` |
| `lsmr` | `radial_xi_lambda_xi_tridi` | 36 | 96 | `2.099900437462e-07` | `7.228822700417e-16` | `1.181850294481e-06` |
| `lsmr` | `radial_xi_lambda_xi_tridi` | 48 | 96 | `2.931165262408e-08` | `1.408480294352e-17` | `1.649696560286e-07` |
| `dense_lstsq` | `none` | 24 | 1 | `2.150747940722e-13` | `7.583142138561e-28` | `1.210467906862e-12` |

Interpretation:

- The finite-current residual, force normalization, lambda block, and radius
  block are all driven below the requested tolerance on the small benchmark.
- The dense solve is fast at this small size because the reduced vector has
  only 61 degrees of freedom.
- Dense reduced Hessians are not the long-term large-grid answer, but they are
  the right reference target for the next scalable solver work.
- The next production step should approximate this dense correction with a
  sparse/block/preconditioned matrix-free method rather than adding stronger
  smoothing filters.

Generated artifacts:

- `results/mirror/m8q_dense_lstsq_probe/residual_newton_convergence_grid_metrics.json`.
- `results/mirror/m8q_dense_lstsq_plots/residual_newton_convergence_grid_metrics.json`.
- `results/mirror/m8q_dense_lstsq_plots/residual_newton_convergence_history.png`.
- `results/mirror/m8q_dense_lstsq_plots/residual_newton_convergence_components.png`.
- `results/mirror/m8q_dense_lstsq_plots/residual_newton_convergence_budget.png`.
- `results/mirror/m8q_dense_lstsq_plots/residual_newton_convergence_resolution_heatmap.png`.
- `results/mirror/m8q_dense_lstsq_plots/best_finite_current_dense_lstsq_m8q_residual_newton/figures/best_finite_current_dense_lstsq_m8q_residual_newton_mirror_boundary_3d.png`.
- `results/mirror/m8q_dense_lstsq_plots/best_finite_current_dense_lstsq_m8q_residual_newton/figures/best_finite_current_dense_lstsq_m8q_residual_newton_mirror_bfield_boundary.png`.
- `results/mirror/m8q_dense_lstsq_plots/best_finite_current_dense_lstsq_m8q_residual_newton/figures/best_finite_current_dense_lstsq_m8q_residual_newton_mirror_bmag_sxi.png`.
- `results/mirror/m8q_dense_lstsq_plots/best_finite_current_dense_lstsq_m8q_residual_newton/figures/best_finite_current_dense_lstsq_m8q_residual_newton_mirror_cross_sections.png`.

### How it was tested

Focused pytest gate:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_fixed_boundary_axisym.py::test_residual_newton_dense_lstsq_solver_improves_perturbed_cylinder \
  tests/mirror/test_mirror_fixed_boundary_axisym.py::test_residual_newton_solver_reaches_tight_residual_for_perturbed_cylinder \
  tests/mirror/test_mirror_low_level_coverage.py \
  tests/mirror/test_mirror_examples.py::test_root_fixed_boundary_solve_diagnostic_runs_without_plots \
  tests/mirror/test_mirror_examples.py::test_root_solver_comparison_example_runs_without_plots \
  tests/mirror/test_mirror_examples.py::test_root_residual_newton_convergence_grid_runs_without_plots \
  tests/mirror/test_mirror_examples.py::test_root_residual_newton_convergence_grid_finite_current_reports_lambda_residual \
  -q
```

Result: `12 passed in 40.90s`.

Dense finite-current benchmark:

```bash
JAX_ENABLE_X64=1 python examples/mirror_residual_newton_convergence_grid.py \
  --outdir results/mirror/m8q_dense_lstsq_plots \
  --ns-array 5 \
  --nxi-array 9 \
  --maxiter-array 24 \
  --residual-linear-maxiter-array 1 \
  --residual-linear-solver dense_lstsq \
  --residual-linear-maxiter-policy fixed \
  --i-prime 0.01 \
  --case-label finite_current_dense_lstsq_m8q \
  --preconditioners none
```

Visual validation:

- Residual history is monotone and reaches below `1e-12`.
- 3D geometry is horizontal in `z`.
- The two circular coils render on the end caps.
- Field-line overlays are visible in the B-direction plot.
- `|B|` is weakest near the center and strongest near both caps.
- Cross sections remain circular and poloidally symmetric.
- The minimum Jacobian remains positive.

### File structure and best-practice notes

- `vmec_jax/mirror/solvers/fixed_boundary/preconditioners.py`
  contains string normalization for residual preconditioners, budget policy,
  and now linear-solver choice.
- `vmec_jax/mirror/solvers/fixed_boundary/optimizers.py`
  keeps the residual-Newton nonlinear logic in one place and branches only at
  the linear correction solve.
- `vmec_jax/mirror/solvers/fixed_boundary/api.py` and
  `types.py` carry the public option without changing the mirror state model.
- `vmec_jax/mirror/solvers/fixed_boundary/diagnostics.py` and
  `vmec_jax/mirror/io/mout.py` preserve the solver choice in summaries and
  output metadata.
- The root example scripts expose the same option name and write it into JSON
  rows, so benchmark artifacts remain auditable.
- The dense path is deliberately explicit and opt-in; the default remains the
  scalable matrix-free LSMR path.

### Best next steps

1. Commit and push the dense-reference solver tranche.
2. Start M8r: use the dense result as the reference target for a scalable
   linear correction:
   - test `lsmr` with better Hessian scaling/column equilibration;
   - compare `lsmr`, `lsqr`, `cg`/`minres` where valid, and block
     preconditioners against the dense step on the same reduced vector;
   - record linear residuals/step quality per Newton iteration.
3. Add a guarded benchmark that compares dense and matrix-free steps at
   `ns=5`, `nxi=9`; keep it out of routine CI if runtime grows.
4. Once matrix-free correction quality approaches the dense reference, rerun
   moderate-resolution finite-current convergence and then return to the
   hybrid/free-boundary lanes.

### Completion percentages after M8q

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `84%`.
- Fixed-boundary axisymmetric solve: `84%`.
- Residual Newton / preconditioning: `82%`.
- Two-coil and manufactured validation: `78%`.
- Finite-current pitch validation: `68%`.
- Plotting and `vmec --plot` mirror support: `78%`.
- I/O schema and docs: `75%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `15%`.
- Free-boundary mirror lane: `5%`.
- Stellarator-mirror hybrid lane: `10%`.
- ESSOS circular-coil mirror beta scan: `0%`.
- PR merge readiness overall: `70%`.

### User input needed

No user input is needed.  The next lane should make the matrix-free correction
approach the dense reference while preserving the dense path as the small-grid
truth model.

---

## 51. 2026-06-17 M8r matrix-free linear-solver diagnostics and LSQR comparison

This lane explains why the scalable matrix-free residual-Newton path still
lags the dense reference from M8q.  The key result is that LSMR and LSQR both
stop by iteration budget (`istop=7`) on the finite-current row.  Removing the
LSMR condition-limit stop did not change the result.  LSQR is available for
comparison, but it is not better than LSMR on this benchmark.

### Steps taken

- Probed SciPy LSMR `conlim=0` with a runtime monkey patch before changing
  source.  The result was identical to the default LSMR row, so condition-limit
  stopping is not the active bottleneck.
- Added `residual_linear_solver="lsqr"` as an opt-in matrix-free
  least-squares solver alongside the existing `lsmr` and dense reference
  `dense_lstsq` paths.
- Added compact Krylov diagnostics to optimizer summaries, mirror NetCDF
  attributes, and root example JSON rows:
  - last stop code;
  - actual last and total Krylov iterations;
  - last linear residual norm;
  - last normal-equation residual norm;
  - last condition estimate.
- Updated the root example CLIs to accept `--residual-linear-solver lsqr`.
- Updated README and mirror output docs.
- Ran finite-current `ns=5`, `nxi=9`, `i_prime=0.01` probes for:
  - LSMR, `24 x 54`;
  - LSQR, `24 x 54`;
  - LSMR, `24 x 96`;
  - one plotted LSMR, `24 x 96`, bundle.

### Results obtained

Finite-current comparison:

| solver | inner budget | final residual | final fsq | normalized force | stop code | last Krylov iterations | condition estimate | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `lsmr` | 54 | `4.332436431347e-05` | `3.077050070764e-11` | `2.438347387145e-04` | 7 | 54 | `1.424929436737e+05` | max inner iterations |
| `lsqr` | 54 | `4.765634337653e-05` | `3.723159121348e-11` | `2.682156327295e-04` | 7 | 54 | `1.150182486564e+06` | max inner iterations |
| `lsmr` | 96 | `1.972965755648e-06` | `6.381301431084e-14` | `1.110409861971e-05` | 7 | 96 | `5.284961140154e+04` | max inner iterations |
| `dense_lstsq` | reference | `2.150747940722e-13` | `7.583142138561e-28` | `1.210467906862e-12` | n/a | n/a | n/a | reached `gtol` |

Interpretation:

- LSMR and LSQR are both iteration-budget limited on the finite-current
  two-coil row.
- LSQR is slightly worse than LSMR at the same budget and leaves a much larger
  lambda residual fraction (`0.384954` vs `0.014690`).
- LSMR at inner budget 96 improves the residual by about `22x` over inner
  budget 54, but still stops by max inner iterations and remains far from the
  dense reference.
- The dense reference did not reveal a nonlinear formulation blocker.  The
  remaining scalable-solver problem is the quality/cost of the matrix-free
  linear correction.
- The next useful work is a better preconditioned linear operator or block
  correction, not switching to LSQR alone.

Generated artifacts:

- `results/mirror/m8r_lsmr_diagnostics_probe/residual_newton_convergence_grid_metrics.json`.
- `results/mirror/m8r_lsqr_probe/residual_newton_convergence_grid_metrics.json`.
- `results/mirror/m8r_lsmr96_diagnostics_probe/residual_newton_convergence_grid_metrics.json`.
- `results/mirror/m8r_lsmr96_diagnostics_plots/residual_newton_convergence_grid_metrics.json`.
- `results/mirror/m8r_lsmr96_diagnostics_plots/residual_newton_convergence_history.png`.
- `results/mirror/m8r_lsmr96_diagnostics_plots/residual_newton_convergence_components.png`.
- `results/mirror/m8r_lsmr96_diagnostics_plots/residual_newton_convergence_budget.png`.
- `results/mirror/m8r_lsmr96_diagnostics_plots/best_finite_current_lsmr96_m8r_residual_newton/figures/best_finite_current_lsmr96_m8r_residual_newton_mirror_boundary_3d.png`.
- `results/mirror/m8r_lsmr96_diagnostics_plots/best_finite_current_lsmr96_m8r_residual_newton/figures/best_finite_current_lsmr96_m8r_residual_newton_mirror_bfield_boundary.png`.
- `results/mirror/m8r_lsmr96_diagnostics_plots/best_finite_current_lsmr96_m8r_residual_newton/figures/best_finite_current_lsmr96_m8r_residual_newton_mirror_bmag_sxi.png`.
- `results/mirror/m8r_lsmr96_diagnostics_plots/best_finite_current_lsmr96_m8r_residual_newton/figures/best_finite_current_lsmr96_m8r_residual_newton_mirror_cross_sections.png`.

### How it was tested

Focused pytest gate:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_fixed_boundary_axisym.py::test_residual_newton_solver_reaches_tight_residual_for_perturbed_cylinder \
  tests/mirror/test_mirror_fixed_boundary_axisym.py::test_residual_newton_dense_lstsq_solver_improves_perturbed_cylinder \
  tests/mirror/test_mirror_low_level_coverage.py \
  tests/mirror/test_mirror_examples.py::test_root_residual_newton_convergence_grid_runs_without_plots \
  tests/mirror/test_mirror_examples.py::test_root_residual_newton_convergence_grid_finite_current_reports_lambda_residual \
  -q
```

Result: `10 passed in 29.67s`.

Static checks:

```bash
python -m ruff format --check \
  vmec_jax/mirror/solvers/fixed_boundary/preconditioners.py \
  vmec_jax/mirror/solvers/fixed_boundary/types.py \
  vmec_jax/mirror/solvers/fixed_boundary/diagnostics.py \
  vmec_jax/mirror/solvers/fixed_boundary/nonlinear.py \
  vmec_jax/mirror/solvers/fixed_boundary/optimizers.py \
  vmec_jax/mirror/io/mout.py \
  examples/mirror_residual_newton_convergence_grid.py \
  examples/mirror_solver_comparison.py \
  examples/mirror_fixed_boundary_solve_diagnostic.py \
  tests/mirror/test_mirror_fixed_boundary_axisym.py \
  tests/mirror/test_mirror_low_level_coverage.py \
  tests/mirror/test_mirror_examples.py
python -m ruff check <same files>
git diff --check
```

Result: passed.

Benchmarks:

```bash
JAX_ENABLE_X64=1 python examples/mirror_residual_newton_convergence_grid.py \
  --outdir results/mirror/m8r_lsmr96_diagnostics_plots \
  --ns-array 5 \
  --nxi-array 9 \
  --maxiter-array 24 \
  --residual-linear-maxiter-array 96 \
  --residual-linear-maxiter-policy fixed \
  --residual-linear-solver lsmr \
  --residual-xi-alpha 1.0 \
  --i-prime 0.01 \
  --case-label finite_current_lsmr96_m8r \
  --preconditioners radial_xi_lambda_xi_tridi
```

Visual validation:

- Residual history is monotone but stalls above the dense reference.
- Field-line overlays render on the B-direction plot.
- The standard horizontal-`z` geometry, `|B|`, and cross-section plots render.

### File structure and best-practice notes

- The solver choice remains one public option, `residual_linear_solver`.
- The additional diagnostics are scalar summary metadata, not large arrays.
- LSMR stays the default.  LSQR and dense solves are opt-in comparison paths.
- The example JSON row schema now records enough linear-solve information to
  debug convergence without rerunning with SciPy verbosity.

### Best next steps

1. Commit and push the M8r diagnostics/LSQR tranche.
2. Start M8s with an actual scalable correction improvement:
   - build a small-grid dense-vs-matrix-free step comparison at each Newton
     iteration;
   - inspect whether right preconditioning is worsening the operator spectrum;
   - try a block diagonal dense-on-coarse or block-Jacobi correction for the
     reduced `a`/lambda blocks;
   - compare against the dense step direction, not only final residual.
3. Once M8s improves the matrix-free step quality, rerun the finite-current
   plotted benchmark and then move to moderate-resolution checks.

### Completion percentages after M8r

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `84%`.
- Fixed-boundary axisymmetric solve: `84%`.
- Residual Newton / preconditioning: `84%`.
- Two-coil and manufactured validation: `79%`.
- Finite-current pitch validation: `69%`.
- Plotting and `vmec --plot` mirror support: `78%`.
- I/O schema and docs: `77%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `15%`.
- Free-boundary mirror lane: `5%`.
- Stellarator-mirror hybrid lane: `10%`.
- ESSOS circular-coil mirror beta scan: `0%`.
- PR merge readiness overall: `71%`.

### User input needed

No user input is needed.  The next lane should compare matrix-free steps
against the dense reference direction and then implement a better scalable
linear correction.

---

## 52. 2026-06-17 M8s dense-step comparison diagnostics

This lane adds an opt-in small-grid diagnostic that compares each matrix-free
Newton correction against the dense reduced-Hessian correction from M8q.  The
purpose is to measure the step-direction error directly, not just infer it
from final residuals.

### Steps taken

- Added `residual_compare_dense_step=False` to `MirrorSolveOptions` and
  `OptimizerOptions`.
- When enabled for matrix-free residual-Newton solves, the solver computes the
  dense scaled reduced-Hessian step at each Krylov iteration point and records
  compact last-step comparison metrics:
  - dense-reference step norm;
  - cosine between the matrix-free and dense step in scaled physical
    coordinates;
  - relative step error.
- Threaded these metrics through:
  - optimizer summaries;
  - mirror NetCDF global attributes;
  - root example CLIs;
  - root example JSON rows.
- Added root CLI flag `--residual-compare-dense-step`.
- Added focused unit coverage for the diagnostic on a small matrix-free
  residual-Newton solve.
- Ran a finite-current two-coil diagnostic with plots:
  `ns=5`, `nxi=9`, `i_prime=0.01`, LSMR, `maxiter=6`, inner budget `54`,
  `radial_xi_lambda_xi_tridi`, and dense-step comparison enabled.

### Results obtained

Finite-current dense-step comparison row:

| metric | value |
| --- | ---: |
| final residual | `1.367152243798e-03` |
| final `fsq` | `3.064106979871e-08` |
| LSMR stop code | 7 |
| last LSMR iterations | 54 |
| last LSMR residual norm | `1.560130370207e-04` |
| dense-reference step norm | `5.135420404703e-03` |
| matrix-free/dense step cosine | `0.901356881087` |
| matrix-free/dense relative step error | `0.764570403248` |

Interpretation:

- The matrix-free LSMR step points broadly in the dense-reference direction,
  but the relative error is still large (`~0.765`) at the last recorded
  correction.
- This supports the M8r conclusion: the next improvement should target the
  correction operator/preconditioner, not the outer residual definition.
- The dense-step comparison is now a reusable small-grid diagnostic for any
  future matrix-free preconditioner changes.

Generated artifacts:

- `results/mirror/m8s_dense_step_compare_plots/residual_newton_convergence_grid_metrics.json`.
- `results/mirror/m8s_dense_step_compare_plots/residual_newton_convergence_history.png`.
- `results/mirror/m8s_dense_step_compare_plots/residual_newton_convergence_components.png`.
- `results/mirror/m8s_dense_step_compare_plots/residual_newton_convergence_budget.png`.
- `results/mirror/m8s_dense_step_compare_plots/best_finite_current_dense_step_compare_m8s_residual_newton/figures/best_finite_current_dense_step_compare_m8s_residual_newton_mirror_boundary_3d.png`.
- `results/mirror/m8s_dense_step_compare_plots/best_finite_current_dense_step_compare_m8s_residual_newton/figures/best_finite_current_dense_step_compare_m8s_residual_newton_mirror_bfield_boundary.png`.
- `results/mirror/m8s_dense_step_compare_plots/best_finite_current_dense_step_compare_m8s_residual_newton/figures/best_finite_current_dense_step_compare_m8s_residual_newton_mirror_bmag_sxi.png`.
- `results/mirror/m8s_dense_step_compare_plots/best_finite_current_dense_step_compare_m8s_residual_newton/figures/best_finite_current_dense_step_compare_m8s_residual_newton_mirror_cross_sections.png`.

### How it was tested

Focused pytest gate:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_fixed_boundary_axisym.py::test_residual_newton_records_dense_step_comparison_for_matrix_free_solver \
  tests/mirror/test_mirror_fixed_boundary_axisym.py::test_residual_newton_solver_reaches_tight_residual_for_perturbed_cylinder \
  tests/mirror/test_mirror_low_level_coverage.py \
  tests/mirror/test_mirror_examples.py::test_root_residual_newton_convergence_grid_runs_without_plots \
  tests/mirror/test_mirror_examples.py::test_root_residual_newton_convergence_grid_finite_current_reports_lambda_residual \
  -q
```

Result: `10 passed in 32.88s`.

Static checks:

```bash
python -m ruff format --check <touched Python files>
python -m ruff check <touched Python files>
git diff --check
```

Result: passed.

Benchmark/plot command:

```bash
JAX_ENABLE_X64=1 python examples/mirror_residual_newton_convergence_grid.py \
  --outdir results/mirror/m8s_dense_step_compare_plots \
  --ns-array 5 \
  --nxi-array 9 \
  --maxiter-array 6 \
  --residual-linear-maxiter-array 54 \
  --residual-linear-maxiter-policy fixed \
  --residual-linear-solver lsmr \
  --residual-compare-dense-step \
  --residual-xi-alpha 1.0 \
  --i-prime 0.01 \
  --case-label finite_current_dense_step_compare_m8s \
  --preconditioners radial_xi_lambda_xi_tridi
```

Visual validation:

- Residual history is monotone over the six recorded steps.
- Field-line overlays render on the B-direction plot.
- The standard horizontal-`z` geometry, `|B|`, and cross-section plots render.

### File structure and best-practice notes

- No new script was added.  The diagnostic lives in the residual-Newton solver
  and is exposed through existing examples.
- The option defaults to `False`, so production matrix-free runs do not pay the
  dense-Hessian cost.
- Metrics are scalar summary values, keeping output size small.

### Best next steps

1. Commit and push M8s.
2. Start M8t with an actual correction improvement:
   - compare dense and matrix-free steps with and without right
     preconditioning;
   - try a block-diagonal dense correction on the reduced radius/lambda
     blocks;
   - use the dense-step relative error as the primary small-grid metric;
   - require improvement before returning to long finite-current convergence
     runs.

### Completion percentages after M8s

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `84%`.
- Fixed-boundary axisymmetric solve: `85%`.
- Residual Newton / preconditioning: `85%`.
- Two-coil and manufactured validation: `80%`.
- Finite-current pitch validation: `70%`.
- Plotting and `vmec --plot` mirror support: `78%`.
- I/O schema and docs: `78%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `15%`.
- Free-boundary mirror lane: `5%`.
- Stellarator-mirror hybrid lane: `10%`.
- ESSOS circular-coil mirror beta scan: `0%`.
- PR merge readiness overall: `72%`.

### User input needed

No user input is needed.  The next lane should use the dense-step metric to
select and validate a better scalable preconditioned correction.

---

## 53. 2026-06-17 M8t dense-step preconditioner scan plot

This lane used the M8s dense-step metric to compare existing right
preconditioners before adding a new correction algorithm.  It also adds a
dedicated dense-step comparison plot to the convergence-grid example.

### Steps taken

- Ran a finite-current dense-step preconditioner scan at:
  - `ns=5`;
  - `nxi=9`;
  - `i_prime=0.01`;
  - LSMR;
  - `maxiter=6`;
  - inner budget `54`;
  - dense-step comparison enabled.
- Compared:
  - `none`;
  - `radial_xi_tridi`;
  - `radial_xi_lambda_xi_tridi`.
- Added `residual_newton_dense_step_comparison.png` to
  `examples/mirror_residual_newton_convergence_grid.py`.
- The new plot shows last-step relative error as bars and last-step cosine as
  a line for every preconditioner row that contains dense-step metrics.
- Regenerated and visually inspected the new plot.

### Results obtained

| preconditioner | final residual | final fsq | lambda fraction | dense-step cosine | dense-step relative error |
| --- | ---: | ---: | ---: | ---: | ---: |
| `none` | `6.999955525369e-03` | `8.032684812647e-07` | `0.971081` | `0.834141083210` | `0.935121950426` |
| `radial_xi_tridi` | `8.155451089785e-03` | `1.090350532424e-06` | `0.488111` | `0.648107171466` | `0.910060777745` |
| `radial_xi_lambda_xi_tridi` | `1.367152243798e-03` | `3.064106979871e-08` | `0.851342` | `0.901356881087` | `0.764570403248` |

Interpretation:

- `radial_xi_lambda_xi_tridi` remains best among existing matrix-free
  right-preconditioners by final residual and dense-step direction metrics.
- Radius-only open-`xi` smoothing helps the lambda fraction but worsens the
  final residual and has the weakest dense-step alignment.
- No preconditioner leaves the largest dense-step relative error.
- The next scalable improvement should be a new correction/preconditioner
  family, not simply choosing a different existing mode.

Generated artifacts:

- `results/mirror/m8t_dense_step_preconditioner_scan/residual_newton_convergence_grid_metrics.json`.
- `results/mirror/m8t_dense_step_preconditioner_scan/residual_newton_dense_step_comparison.png`.
- `results/mirror/m8t_dense_step_preconditioner_scan/residual_newton_convergence_preconditioners.png`.
- `results/mirror/m8t_dense_step_preconditioner_scan/residual_newton_convergence_history.png`.
- `results/mirror/m8t_dense_step_preconditioner_scan/residual_newton_convergence_components.png`.
- `results/mirror/m8t_dense_step_preconditioner_scan/best_finite_current_dense_step_preconditioner_m8t_residual_newton/figures/`.

### How it was tested

Focused tests:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_examples.py::test_root_residual_newton_convergence_grid_runs_without_plots \
  tests/mirror/test_mirror_fixed_boundary_axisym.py::test_residual_newton_records_dense_step_comparison_for_matrix_free_solver \
  -q
```

Result: `2 passed in 11.01s`.

Static checks:

```bash
python -m ruff format --check \
  examples/mirror_residual_newton_convergence_grid.py \
  tests/mirror/test_mirror_examples.py \
  tests/mirror/test_mirror_fixed_boundary_axisym.py \
  vmec_jax/mirror/solvers/fixed_boundary/optimizers.py
python -m ruff check <same files>
git diff --check
```

Result: passed.

Benchmark/plot command:

```bash
JAX_ENABLE_X64=1 python examples/mirror_residual_newton_convergence_grid.py \
  --outdir results/mirror/m8t_dense_step_preconditioner_scan \
  --ns-array 5 \
  --nxi-array 9 \
  --maxiter-array 6 \
  --residual-linear-maxiter-array 54 \
  --residual-linear-maxiter-policy fixed \
  --residual-linear-solver lsmr \
  --residual-compare-dense-step \
  --residual-xi-alpha 1.0 \
  --i-prime 0.01 \
  --case-label finite_current_dense_step_preconditioner_m8t \
  --preconditioners none,radial_xi_tridi,radial_xi_lambda_xi_tridi
```

Visual validation:

- The dense-step comparison plot renders with an uncluttered top legend.
- The plot makes `radial_xi_lambda_xi_tridi` visibly best by relative step
  error and cosine.

### File structure and best-practice notes

- No new script was added.
- The new plot is housed with the other convergence-grid plots.
- The plot is skipped automatically when rows do not contain dense-step
  metrics, so existing no-plot and no-comparison runs remain unchanged.

### Best next steps

1. Commit and push M8t.
2. Start M8u with a new correction family:
   - block diagonal dense reference on reduced `a` and lambda blocks;
   - compare block correction against full dense and LSMR using the dense-step
     plot;
   - only promote the mode if it improves relative step error and final
     residual over `radial_xi_lambda_xi_tridi`.

### Completion percentages after M8t

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `84%`.
- Fixed-boundary axisymmetric solve: `85%`.
- Residual Newton / preconditioning: `86%`.
- Two-coil and manufactured validation: `80%`.
- Finite-current pitch validation: `70%`.
- Plotting and `vmec --plot` mirror support: `79%`.
- I/O schema and docs: `78%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `15%`.
- Free-boundary mirror lane: `5%`.
- Stellarator-mirror hybrid lane: `10%`.
- ESSOS circular-coil mirror beta scan: `0%`.
- PR merge readiness overall: `73%`.

### User input needed

No user input is needed.  The existing preconditioners have been ranked; the
next work should implement and benchmark a new block correction.

---

## 54. 2026-06-17 M8u block-dense correction reference

This lane adds a first block-correction family.  The new
`block_dense_lstsq` linear solver builds the dense scaled reduced Hessian like
the full `dense_lstsq` reference, but solves the reduced radius block and
lambda block separately.  This is still a small-grid dense reference path, but
it verifies that an `a`/lambda block correction can reach tight finite-current
residuals and is therefore a credible target for a later scalable block
preconditioner.

### Steps taken

- Added `residual_linear_solver="block_dense_lstsq"`.
- Implemented block-diagonal dense solves in the residual-Newton dense helper:
  - split reduced variables into radius `a` and lambda blocks using the
    existing reduced-coordinate layout;
  - solve each block with `np.linalg.lstsq`;
  - apply the same optional right-preconditioner transform as the full dense
    path when requested.
- Added CLI support in the root mirror examples.
- Added focused parser and solver tests.
- Ran a finite-current two-coil benchmark and regenerated the standard plot
  bundle.

### Results obtained

Finite-current two-coil block-dense row:

| solver | preconditioner | final residual | final fsq | normalized force | Newton iterations | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `block_dense_lstsq` | `none` | `1.386654012023e-15` | `3.152146473869e-32` | `7.804262636706e-15` | 5 | reached `gtol` |

Comparison:

- Full dense M8q row:
  - residual `2.150747940722e-13`;
  - `fsq=7.583142138561e-28`;
  - `nit=5`.
- Block dense M8u row:
  - residual `1.386654012023e-15`;
  - `fsq=3.152146473869e-32`;
  - `nit=5`.

Interpretation:

- The block-diagonal `a`/lambda correction is sufficient for the small
  finite-current benchmark and reaches a tighter residual than the full dense
  reference in this run.
- This strongly supports pursuing a scalable block preconditioner/block
  correction rather than only increasing Krylov budgets.
- The current implementation is an opt-in dense reference path; the next step
  is to replace dense sub-block solves with scalable approximations.

Generated artifacts:

- `results/mirror/m8u_block_dense_probe/residual_newton_convergence_grid_metrics.json`.
- `results/mirror/m8u_block_dense_plots/residual_newton_convergence_grid_metrics.json`.
- `results/mirror/m8u_block_dense_plots/residual_newton_convergence_history.png`.
- `results/mirror/m8u_block_dense_plots/residual_newton_convergence_components.png`.
- `results/mirror/m8u_block_dense_plots/residual_newton_convergence_budget.png`.
- `results/mirror/m8u_block_dense_plots/best_finite_current_block_dense_m8u_residual_newton/figures/best_finite_current_block_dense_m8u_residual_newton_mirror_boundary_3d.png`.
- `results/mirror/m8u_block_dense_plots/best_finite_current_block_dense_m8u_residual_newton/figures/best_finite_current_block_dense_m8u_residual_newton_mirror_bfield_boundary.png`.
- `results/mirror/m8u_block_dense_plots/best_finite_current_block_dense_m8u_residual_newton/figures/best_finite_current_block_dense_m8u_residual_newton_mirror_bmag_sxi.png`.
- `results/mirror/m8u_block_dense_plots/best_finite_current_block_dense_m8u_residual_newton/figures/best_finite_current_block_dense_m8u_residual_newton_mirror_cross_sections.png`.

### How it was tested

Focused pytest gate:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_fixed_boundary_axisym.py::test_residual_newton_block_dense_lstsq_solver_improves_perturbed_cylinder \
  tests/mirror/test_mirror_fixed_boundary_axisym.py::test_residual_newton_dense_lstsq_solver_improves_perturbed_cylinder \
  tests/mirror/test_mirror_low_level_coverage.py \
  tests/mirror/test_mirror_examples.py::test_root_residual_newton_convergence_grid_runs_without_plots \
  -q
```

Result: `9 passed in 9.97s`.

Static checks:

```bash
python -m ruff format --check <touched Python files>
python -m ruff check <touched Python files>
git diff --check
```

Result: passed.

Benchmark/plot command:

```bash
JAX_ENABLE_X64=1 python examples/mirror_residual_newton_convergence_grid.py \
  --outdir results/mirror/m8u_block_dense_plots \
  --ns-array 5 \
  --nxi-array 9 \
  --maxiter-array 24 \
  --residual-linear-maxiter-array 1 \
  --residual-linear-solver block_dense_lstsq \
  --residual-linear-maxiter-policy fixed \
  --i-prime 0.01 \
  --case-label finite_current_block_dense_m8u \
  --preconditioners none
```

Visual validation:

- Residual history is monotone and reaches below `1e-12`.
- Field-line overlays render on the B-direction plot.
- Standard horizontal-`z` geometry, `|B|`, and cross-section plots render.
- The minimum Jacobian is positive.

### File structure and best-practice notes

- No new file was added.
- The block correction reuses the existing reduced-coordinate layout and dense
  helper in `optimizers.py`.
- The option is explicit and opt-in, preserving the default matrix-free LSMR
  path.

### Best next steps

1. Commit and push M8u.
2. Start M8v by making the block correction more scalable:
   - replace dense block solves with block tridiagonal/tensor-product
     approximations;
   - compare block approximation against `block_dense_lstsq`;
   - require lower dense-step relative error and tight finite-current
     convergence before making it a production default.

### Completion percentages after M8u

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `84%`.
- Fixed-boundary axisymmetric solve: `86%`.
- Residual Newton / preconditioning: `88%`.
- Two-coil and manufactured validation: `81%`.
- Finite-current pitch validation: `73%`.
- Plotting and `vmec --plot` mirror support: `79%`.
- I/O schema and docs: `79%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `15%`.
- Free-boundary mirror lane: `5%`.
- Stellarator-mirror hybrid lane: `10%`.
- ESSOS circular-coil mirror beta scan: `0%`.
- PR merge readiness overall: `75%`.

### User input needed

No user input is needed.  The next lane should convert the successful
block-dense idea into a scalable block preconditioner/correction.

---

## 55. 2026-06-17 M8v moderate-resolution block-dense benchmark

This lane checked whether the M8u `block_dense_lstsq` correction still works
at the earlier hard two-coil resolution `ns=9`, `nxi=17`.  No source changes
were needed.

### Steps taken

- Ran finite-current two-coil `block_dense_lstsq` at:
  - `ns=9`;
  - `nxi=17`;
  - `i_prime=0.01`;
  - `maxiter=12`;
  - `residual_linear_maxiter=1` placeholder, because dense/block-dense solves
    do not use Krylov iteration budgets;
  - no residual preconditioner.
- Generated a full plot bundle for the same row.
- Visually inspected residual history and field-line overlay.

### Results obtained

| quantity | value |
| --- | ---: |
| active reduced dof | 249 |
| final residual | `1.742396273103e-14` |
| final `fsq` | `1.219254928724e-30` |
| normalized force | `4.836809658876e-14` |
| Newton iterations | 5 |
| optimizer success | true |
| minimum `sqrt(g)` | `2.997263429136e-03` |
| mirror ratio | `22.144701631554` |

Interpretation:

- `block_dense_lstsq` reaches tight finite-current convergence at the
  moderate `ns=9`, `nxi=17` resolution.
- Runtime remained small for this moderate row, but dense Hessian construction
  is still not the production large-grid path.
- This reinforces the next objective: keep the block-correction idea while
  replacing dense sub-block solves with scalable approximations.

Generated artifacts:

- `results/mirror/m8v_block_dense_ns9_probe/residual_newton_convergence_grid_metrics.json`.
- `results/mirror/m8v_block_dense_ns9_plots/residual_newton_convergence_grid_metrics.json`.
- `results/mirror/m8v_block_dense_ns9_plots/residual_newton_convergence_history.png`.
- `results/mirror/m8v_block_dense_ns9_plots/residual_newton_convergence_components.png`.
- `results/mirror/m8v_block_dense_ns9_plots/best_finite_current_block_dense_ns9_m8v_residual_newton/figures/best_finite_current_block_dense_ns9_m8v_residual_newton_mirror_boundary_3d.png`.
- `results/mirror/m8v_block_dense_ns9_plots/best_finite_current_block_dense_ns9_m8v_residual_newton/figures/best_finite_current_block_dense_ns9_m8v_residual_newton_mirror_bfield_boundary.png`.
- `results/mirror/m8v_block_dense_ns9_plots/best_finite_current_block_dense_ns9_m8v_residual_newton/figures/best_finite_current_block_dense_ns9_m8v_residual_newton_mirror_bmag_sxi.png`.
- `results/mirror/m8v_block_dense_ns9_plots/best_finite_current_block_dense_ns9_m8v_residual_newton/figures/best_finite_current_block_dense_ns9_m8v_residual_newton_mirror_cross_sections.png`.

### How it was tested

Commands:

```bash
JAX_ENABLE_X64=1 python examples/mirror_residual_newton_convergence_grid.py \
  --outdir results/mirror/m8v_block_dense_ns9_probe \
  --ns-array 9 \
  --nxi-array 17 \
  --maxiter-array 12 \
  --residual-linear-maxiter-array 1 \
  --residual-linear-solver block_dense_lstsq \
  --residual-linear-maxiter-policy fixed \
  --i-prime 0.01 \
  --case-label finite_current_block_dense_ns9_m8v \
  --preconditioners none \
  --no-plots
JAX_ENABLE_X64=1 python examples/mirror_residual_newton_convergence_grid.py \
  --outdir results/mirror/m8v_block_dense_ns9_plots \
  --ns-array 9 \
  --nxi-array 17 \
  --maxiter-array 12 \
  --residual-linear-maxiter-array 1 \
  --residual-linear-solver block_dense_lstsq \
  --residual-linear-maxiter-policy fixed \
  --i-prime 0.01 \
  --case-label finite_current_block_dense_ns9_m8v \
  --preconditioners none
```

Visual validation:

- Residual history reaches below `1e-12` in five Newton iterations.
- Field-line overlays render on the B-direction plot.
- Standard horizontal-`z` geometry, `|B|`, and cross-section plots render.

### File structure and best-practice notes

- This is a benchmark/log lane only; no source files changed.
- The existing convergence-grid example was sufficient.

### Best next steps

1. Commit and push the benchmark log.
2. Move to a scalable block approximation:
   - exploit the existing `a`/lambda reduced split;
   - start with block tridiagonal/tensor-product approximations;
   - benchmark against both `block_dense_lstsq` and LSMR.

### Completion percentages after M8v

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `84%`.
- Fixed-boundary axisymmetric solve: `87%`.
- Residual Newton / preconditioning: `89%`.
- Two-coil and manufactured validation: `82%`.
- Finite-current pitch validation: `75%`.
- Plotting and `vmec --plot` mirror support: `79%`.
- I/O schema and docs: `79%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `15%`.
- Free-boundary mirror lane: `5%`.
- Stellarator-mirror hybrid lane: `10%`.
- ESSOS circular-coil mirror beta scan: `0%`.
- PR merge readiness overall: `76%`.

### User input needed

No user input is needed.

---

## 56. 2026-06-17 M8w matrix-free block LSMR correction

This lane converted the successful M8u/M8v block-dense split into a scalable
matrix-free diagnostic solver.  The new `block_lsmr` mode keeps the reduced
axisymmetric `a`/lambda split, but solves each diagonal block with SciPy LSMR
through Hessian-vector products instead of materializing the reduced Hessian.

### Steps taken

- Added residual linear-solver aliases:
  - `block_lsmr`;
  - `split_lsmr`;
  - `block_matrix_free_lsmr`.
- Exposed `block_lsmr` in the root-level residual-Newton CLIs:
  - `examples/mirror_residual_newton_convergence_grid.py`;
  - `examples/mirror_solver_comparison.py`;
  - `examples/mirror_fixed_boundary_solve_diagnostic.py`.
- Implemented the split block LSMR correction in
  `projected_residual_newton_solve`.
- Refactored the iterative linear-solve details into local helper functions so
  the main residual-Newton loop now shares bookkeeping between:
  - full LSMR/LSQR;
  - split block LSMR;
  - dense and block-dense reference solves.
- Added focused unit coverage for the alias normalization and a small
  perturbed-cylinder `block_lsmr` solve.
- Updated `examples/mirror/README.md` to describe `block_lsmr` as the scalable
  split radius/lambda diagnostic path.
- Ran finite-current two-coil comparisons against the previously logged full
  LSMR and block-dense rows.
- Generated and visually inspected the full standard plot bundle for the
  24-step finite-current `block_lsmr` row.

### Results obtained

Small finite-current two-coil rows, `ns=5`, `nxi=9`, `I'=0.01`:

| row | preconditioner | outer x inner | final residual | final `fsq` | dense-step cosine | dense-step relative error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| M8w `block_lsmr` | none | 6 x 54 | `6.238420625904e-03` | `6.379982279623e-07` | `0.880578524931` | `0.875679418150` |
| M8w `block_lsmr` | `radial_xi_lambda_xi_tridi` | 6 x 54 | `3.813022192885e-04` | `2.383465285809e-09` | `0.987200858532` | `0.349685129692` |
| M8w `block_lsmr` | `radial_xi_lambda_xi_tridi` | 24 x 54 | `7.544736216469e-08` | `9.331646651820e-17` | `0.961205122336` | `0.738477842202` |

Interpretation:

- With no preconditioner, split block LSMR is slightly better aligned with the
  dense reference step than the previous full LSMR no-preconditioner row.
- With the lambda-xi smoother, the 6-step split row improves the finite-current
  residual from the earlier full-LSMR lambda-xi value of about `1.37e-03` to
  `3.81e-04`.
- At the same 24 x 54 budget, split block LSMR reaches `7.54e-08`, compared
  with the earlier full-LSMR lambda-xi value of about `4.33e-05`.
- The split LSMR path is still not dense-quality: the M8u/M8v dense and
  block-dense rows reached tight residuals in about five Newton iterations.
- The result is therefore a useful scalable approximation and benchmark lane,
  not yet the final production residual-Newton correction.

The 24-step plotted row also reported:

| quantity | value |
| --- | ---: |
| normalized force | `4.246272137637e-07` |
| optimizer success | false, maximum iterations reached |
| Newton iterations | 24 |
| last split LSMR stop code | 7 |
| last summed split LSMR iterations | 89 |
| minimum `sqrt(g)` | `3.043082037079e-03` |
| mirror ratio | `19.532434206345` |

The positive `sqrt(g)` and plot inspection show that this row remains a valid
geometry while the residual decreases.

Generated artifacts:

- `results/mirror/m8w_block_lsmr_probe/residual_newton_convergence_grid_metrics.json`.
- `results/mirror/m8w_block_lsmr_lambda_xi_probe/residual_newton_convergence_grid_metrics.json`.
- `results/mirror/m8w_block_lsmr_lambda_xi_24_probe/residual_newton_convergence_grid_metrics.json`.
- `results/mirror/m8w_block_lsmr_lambda_xi_24_plots/residual_newton_convergence_grid_metrics.json`.
- `results/mirror/m8w_block_lsmr_lambda_xi_24_plots/residual_newton_convergence_history.png`.
- `results/mirror/m8w_block_lsmr_lambda_xi_24_plots/residual_newton_dense_step_comparison.png`.
- `results/mirror/m8w_block_lsmr_lambda_xi_24_plots/best_finite_current_block_lsmr_lambda_xi_24_m8w_residual_newton/figures/best_finite_current_block_lsmr_lambda_xi_24_m8w_residual_newton_mirror_boundary_3d.png`.
- `results/mirror/m8w_block_lsmr_lambda_xi_24_plots/best_finite_current_block_lsmr_lambda_xi_24_m8w_residual_newton/figures/best_finite_current_block_lsmr_lambda_xi_24_m8w_residual_newton_mirror_bfield_boundary.png`.
- `results/mirror/m8w_block_lsmr_lambda_xi_24_plots/best_finite_current_block_lsmr_lambda_xi_24_m8w_residual_newton/figures/best_finite_current_block_lsmr_lambda_xi_24_m8w_residual_newton_mirror_bmag_sxi.png`.
- `results/mirror/m8w_block_lsmr_lambda_xi_24_plots/best_finite_current_block_lsmr_lambda_xi_24_m8w_residual_newton/figures/best_finite_current_block_lsmr_lambda_xi_24_m8w_residual_newton_mirror_cross_sections.png`.
- `results/mirror/m8w_block_lsmr_cli_smoke/residual_newton_convergence_grid_metrics.json`.

Visual validation:

- The residual history decreases monotonically over the plotted row.
- The horizontal-`z` 3-D mirror plot shows the expected narrow end caps and
  wider central throat.
- The `|B|` map is strongest near the caps and weakest near the center.
- The B-direction plot includes visible cap-to-cap field-line traces.
- Cross sections remain circular and ordered for this axisymmetric benchmark.

### How it was tested

Focused automated tests:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_fixed_boundary_axisym.py::test_residual_newton_block_lsmr_solver_improves_perturbed_cylinder \
  tests/mirror/test_mirror_fixed_boundary_axisym.py::test_residual_newton_block_dense_lstsq_solver_improves_perturbed_cylinder \
  tests/mirror/test_mirror_low_level_coverage.py::test_low_level_field_energy_residual_and_optimizer_guards \
  -q
```

Result: `3 passed in 15.62s`.

Broader nearby tests:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_fixed_boundary_axisym.py \
  tests/mirror/test_mirror_examples.py::test_root_residual_newton_convergence_grid_runs_without_plots \
  -q
```

Result: `15 passed in 41.91s`.

Lint/format/whitespace:

```bash
python -m ruff format vmec_jax/mirror/solvers/fixed_boundary/optimizers.py \
  vmec_jax/mirror/solvers/fixed_boundary/preconditioners.py \
  examples/mirror_residual_newton_convergence_grid.py \
  examples/mirror_solver_comparison.py \
  examples/mirror_fixed_boundary_solve_diagnostic.py \
  tests/mirror/test_mirror_fixed_boundary_axisym.py \
  tests/mirror/test_mirror_low_level_coverage.py
python -m ruff check vmec_jax/mirror/solvers/fixed_boundary/optimizers.py \
  vmec_jax/mirror/solvers/fixed_boundary/preconditioners.py \
  examples/mirror_residual_newton_convergence_grid.py \
  examples/mirror_solver_comparison.py \
  examples/mirror_fixed_boundary_solve_diagnostic.py \
  tests/mirror/test_mirror_fixed_boundary_axisym.py \
  tests/mirror/test_mirror_low_level_coverage.py
git diff --check
```

Result: all checks passed.

CLI smoke:

```bash
JAX_ENABLE_X64=1 python examples/mirror_residual_newton_convergence_grid.py \
  --outdir results/mirror/m8w_block_lsmr_cli_smoke \
  --ns-array 5 \
  --nxi-array 9 \
  --maxiter-array 2 \
  --residual-linear-maxiter-array 16 \
  --residual-linear-maxiter-policy fixed \
  --residual-linear-solver block_lsmr \
  --preconditioners radial_xi_lambda_xi_tridi \
  --residual-xi-alpha 1.0 \
  --i-prime 0.01 \
  --case-label finite_current_block_lsmr_cli_smoke_m8w \
  --no-plots
```

Result: metrics JSON was written with positive `sqrt(g)` and finite residual
diagnostics.

Benchmark commands:

```bash
JAX_ENABLE_X64=1 python examples/mirror_residual_newton_convergence_grid.py \
  --outdir results/mirror/m8w_block_lsmr_probe \
  --ns-array 5 \
  --nxi-array 9 \
  --maxiter-array 6 \
  --residual-linear-maxiter-array 54 \
  --residual-linear-maxiter-policy fixed \
  --residual-linear-solver block_lsmr \
  --residual-compare-dense-step \
  --i-prime 0.01 \
  --case-label finite_current_block_lsmr_m8w \
  --preconditioners none \
  --no-plots
JAX_ENABLE_X64=1 python examples/mirror_residual_newton_convergence_grid.py \
  --outdir results/mirror/m8w_block_lsmr_lambda_xi_probe \
  --ns-array 5 \
  --nxi-array 9 \
  --maxiter-array 6 \
  --residual-linear-maxiter-array 54 \
  --residual-linear-maxiter-policy fixed \
  --residual-linear-solver block_lsmr \
  --residual-compare-dense-step \
  --residual-xi-alpha 1.0 \
  --i-prime 0.01 \
  --case-label finite_current_block_lsmr_lambda_xi_m8w \
  --preconditioners radial_xi_lambda_xi_tridi \
  --no-plots
JAX_ENABLE_X64=1 python examples/mirror_residual_newton_convergence_grid.py \
  --outdir results/mirror/m8w_block_lsmr_lambda_xi_24_plots \
  --ns-array 5 \
  --nxi-array 9 \
  --maxiter-array 24 \
  --residual-linear-maxiter-array 54 \
  --residual-linear-maxiter-policy fixed \
  --residual-linear-solver block_lsmr \
  --residual-compare-dense-step \
  --residual-xi-alpha 1.0 \
  --i-prime 0.01 \
  --case-label finite_current_block_lsmr_lambda_xi_24_m8w \
  --preconditioners radial_xi_lambda_xi_tridi
```

### File structure and best-practice notes

- `preconditioners.py` remains the single normalization point for residual
  preconditioner, solver, and linear-budget option strings.
- `optimizers.py` keeps the residual-Newton algorithm in one place but now
  separates full Krylov and split block LSMR operator construction into helper
  functions inside `projected_residual_newton_solve`.
- The root-level example CLIs expose the same solver choices, avoiding drift
  between diagnostic scripts.
- Tests remain focused on behavior and diagnostics, while benchmark-quality
  assertions stay in `plan_mirror.md` and generated result artifacts.

### Best next steps

1. Commit and push M8w.
2. Probe whether `block_lsmr` can close the remaining gap to dense/block-dense:
   - separate radius and lambda inner iteration budgets;
   - adaptive split budgets based on block residuals;
   - stronger block approximations for the lambda-dominated finite-current
     residual;
   - `ns=9`, `nxi=17` comparison against M8v `block_dense_lstsq`.
3. Start reducing residual-Newton complexity by moving stable linear-solve
   helpers into a small dedicated module once the next scalable strategy is
   selected.
4. Continue the finite-current lane with visible pitch and normalized force
   diagnostics tied to solve convergence.

### Completion percentages after M8w

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `84%`.
- Fixed-boundary axisymmetric solve: `88%`.
- Residual Newton / preconditioning: `90%`.
- Two-coil and manufactured validation: `82%`.
- Finite-current pitch validation: `77%`.
- Plotting and `vmec --plot` mirror support: `79%`.
- I/O schema and docs: `80%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `15%`.
- Free-boundary mirror lane: `5%`.
- Stellarator-mirror hybrid lane: `10%`.
- ESSOS circular-coil mirror beta scan: `0%`.
- PR merge readiness overall: `77%`.

### User input needed

No user input is needed.

---

## 57. 2026-06-17 M8x split lambda-block budget diagnostic

This lane tested whether the M8w matrix-free split block solver could scale to
the moderate finite-current row used in M8v (`ns=9`, `nxi=17`) and added a
small option to focus Krylov work on the lambda block when the residual is
lambda dominated.

### Steps taken

- Ran a moderate finite-current `block_lsmr` probe at:
  - `ns=9`;
  - `nxi=17`;
  - `I'=0.01`;
  - `maxiter=12`;
  - `residual_linear_maxiter=102`;
  - `radial_xi_lambda_xi_tridi`;
  - no plots, for timing/quality comparison.
- Ran a same-budget full LSMR comparison at the same resolution and
  preconditioner.
- Ran a higher-budget `block_lsmr` probe with `maxiter=6` and
  `residual_linear_maxiter=249` for both blocks.
- Added `residual_block_lambda_maxiter` as an optional solve option:
  - default `None` preserves current behavior;
  - only `block_lsmr` uses it;
  - radius block keeps the standard effective budget;
  - lambda block can receive a larger explicit budget.
- Exposed `--residual-block-lambda-maxiter` in
  `examples/mirror_residual_newton_convergence_grid.py`.
- Stored the override in convergence-grid JSON rows and in mirror `mout`
  metadata when used.
- Updated the block-LSMR unit test to verify that the lambda override changes
  the recorded effective max budget.
- Added README guidance for lambda-dominated block-LSMR studies.
- Ran a moderate split-budget probe with radius budget `102` and lambda budget
  `249`.
- Generated a full plotted bundle for the best M8x split-budget moderate row.

### Results obtained

Moderate finite-current two-coil rows, `ns=9`, `nxi=17`, `I'=0.01`:

| row | outer | radius budget | lambda budget | last iterations | final residual | final `fsq` | normalized force |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full LSMR | 12 | 102 | 102 | 102 | `1.158016138481e-02` | `5.385547698721e-07` | `3.171545880611e-02` |
| block LSMR | 12 | 102 | 102 | 204 | `9.110897547661e-03` | `3.333672856384e-07` | `2.502563842048e-02` |
| block LSMR | 6 | 249 | 249 | 498 | `3.272284167977e-03` | `4.300338825700e-08` | `9.062380071320e-03` |
| block LSMR split budget | 6 | 102 | 249 | 351 | `3.126838214005e-03` | `3.926553099020e-08` | `8.661032213469e-03` |
| M8v block dense reference | 12 | dense | dense | n/a | `1.742396273103e-14` | `1.219254928724e-30` | `4.836809658876e-14` |

Interpretation:

- The current matrix-free Krylov paths do not close the moderate finite-current
  gap to the block-dense reference.
- Split block LSMR is modestly better than full LSMR at the same 102 budget,
  but both are still far from tight convergence.
- Increasing block budgets helps, but the high-budget matrix-free run remains
  lambda dominated and expensive.
- The new split-budget option is useful because it slightly improves the
  249/249 row while reducing last-step Krylov work from `498` to `351`
  iterations.
- This is not enough for research-grade production convergence; the next
  scalable solver lane should improve the lambda block preconditioner/operator
  rather than only increasing Krylov budgets.

Best plotted split-budget row:

| quantity | value |
| --- | ---: |
| final residual | `3.126838214005e-03` |
| final `fsq` | `3.926553099020e-08` |
| normalized force | `8.661032213469e-03` |
| Newton iterations | 6 |
| optimizer success | false, maximum iterations reached |
| residual `a` norm | `5.474244766630e-05` |
| residual lambda norm | `3.126358981464e-03` |
| lambda residual fraction | `0.999846735742` |
| minimum `sqrt(g)` | `2.997247511021e-03` |
| mirror ratio | `22.174883659076` |

Generated artifacts:

- `results/mirror/m8x_block_lsmr_ns9_probe/residual_newton_convergence_grid_metrics.json`.
- `results/mirror/m8x_full_lsmr_ns9_probe/residual_newton_convergence_grid_metrics.json`.
- `results/mirror/m8x_block_lsmr_ns9_inner249_probe/residual_newton_convergence_grid_metrics.json`.
- `results/mirror/m8x_block_lsmr_ns9_split_budget_probe/residual_newton_convergence_grid_metrics.json`.
- `results/mirror/m8x_block_lsmr_ns9_split_budget_plots/residual_newton_convergence_grid_metrics.json`.
- `results/mirror/m8x_block_lsmr_ns9_split_budget_plots/residual_newton_convergence_history.png`.
- `results/mirror/m8x_block_lsmr_ns9_split_budget_plots/residual_newton_convergence_components.png`.
- `results/mirror/m8x_block_lsmr_ns9_split_budget_plots/best_finite_current_block_lsmr_ns9_split_budget_m8x_residual_newton/figures/best_finite_current_block_lsmr_ns9_split_budget_m8x_residual_newton_mirror_boundary_3d.png`.
- `results/mirror/m8x_block_lsmr_ns9_split_budget_plots/best_finite_current_block_lsmr_ns9_split_budget_m8x_residual_newton/figures/best_finite_current_block_lsmr_ns9_split_budget_m8x_residual_newton_mirror_bfield_boundary.png`.
- `results/mirror/m8x_block_lsmr_ns9_split_budget_plots/best_finite_current_block_lsmr_ns9_split_budget_m8x_residual_newton/figures/best_finite_current_block_lsmr_ns9_split_budget_m8x_residual_newton_mirror_bmag_sxi.png`.
- `results/mirror/m8x_block_lsmr_ns9_split_budget_plots/best_finite_current_block_lsmr_ns9_split_budget_m8x_residual_newton/figures/best_finite_current_block_lsmr_ns9_split_budget_m8x_residual_newton_mirror_cross_sections.png`.

Visual validation:

- Residual history decreases but stalls above tight convergence.
- Horizontal-`z` 3-D geometry remains well ordered.
- `|B|` remains strongest near the end caps and weakest near the central
  throat.
- B-direction plot includes visible cap-to-cap field-line traces.
- Cross sections remain circular and nested.

### How it was tested

Automated tests:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_fixed_boundary_axisym.py::test_residual_newton_block_lsmr_solver_improves_perturbed_cylinder \
  tests/mirror/test_mirror_examples.py::test_root_residual_newton_convergence_grid_runs_without_plots \
  tests/mirror/test_mirror_io.py \
  -q
```

Result: `6 passed in 19.65s`.

Lint/format/whitespace:

```bash
python -m ruff check \
  vmec_jax/mirror/solvers/fixed_boundary/types.py \
  vmec_jax/mirror/solvers/fixed_boundary/api.py \
  vmec_jax/mirror/solvers/fixed_boundary/optimizers.py \
  vmec_jax/mirror/io/mout.py \
  examples/mirror_residual_newton_convergence_grid.py \
  tests/mirror/test_mirror_fixed_boundary_axisym.py
python -m ruff format --check \
  vmec_jax/mirror/solvers/fixed_boundary/types.py \
  vmec_jax/mirror/solvers/fixed_boundary/api.py \
  vmec_jax/mirror/solvers/fixed_boundary/optimizers.py \
  vmec_jax/mirror/io/mout.py \
  examples/mirror_residual_newton_convergence_grid.py \
  tests/mirror/test_mirror_fixed_boundary_axisym.py
git diff --check
```

Result: all checks passed.

CLI smoke for the new option:

```bash
JAX_ENABLE_X64=1 python examples/mirror_residual_newton_convergence_grid.py \
  --outdir results/mirror/m8x_block_lambda_budget_cli_smoke \
  --ns-array 5 \
  --nxi-array 9 \
  --maxiter-array 2 \
  --residual-linear-maxiter-array 12 \
  --residual-block-lambda-maxiter 32 \
  --residual-linear-maxiter-policy fixed \
  --residual-linear-solver block_lsmr \
  --preconditioners radial_xi_lambda_xi_tridi \
  --residual-xi-alpha 1.0 \
  --i-prime 0.01 \
  --case-label finite_current_block_lambda_budget_cli_smoke_m8x \
  --no-plots
```

Result: metrics row recorded `residual_block_lambda_maxiter=32`,
`residual_linear_maxiter_effective_max=32`, and positive `sqrt(g)`.

Moderate benchmark commands:

```bash
JAX_ENABLE_X64=1 python examples/mirror_residual_newton_convergence_grid.py \
  --outdir results/mirror/m8x_block_lsmr_ns9_probe \
  --ns-array 9 \
  --nxi-array 17 \
  --maxiter-array 12 \
  --residual-linear-maxiter-array 102 \
  --residual-linear-maxiter-policy fixed \
  --residual-linear-solver block_lsmr \
  --residual-xi-alpha 1.0 \
  --i-prime 0.01 \
  --case-label finite_current_block_lsmr_ns9_m8x \
  --preconditioners radial_xi_lambda_xi_tridi \
  --no-plots
JAX_ENABLE_X64=1 python examples/mirror_residual_newton_convergence_grid.py \
  --outdir results/mirror/m8x_full_lsmr_ns9_probe \
  --ns-array 9 \
  --nxi-array 17 \
  --maxiter-array 12 \
  --residual-linear-maxiter-array 102 \
  --residual-linear-maxiter-policy fixed \
  --residual-linear-solver lsmr \
  --residual-xi-alpha 1.0 \
  --i-prime 0.01 \
  --case-label finite_current_full_lsmr_ns9_m8x \
  --preconditioners radial_xi_lambda_xi_tridi \
  --no-plots
JAX_ENABLE_X64=1 python examples/mirror_residual_newton_convergence_grid.py \
  --outdir results/mirror/m8x_block_lsmr_ns9_split_budget_plots \
  --ns-array 9 \
  --nxi-array 17 \
  --maxiter-array 6 \
  --residual-linear-maxiter-array 102 \
  --residual-block-lambda-maxiter 249 \
  --residual-linear-maxiter-policy fixed \
  --residual-linear-solver block_lsmr \
  --residual-xi-alpha 1.0 \
  --i-prime 0.01 \
  --case-label finite_current_block_lsmr_ns9_split_budget_m8x \
  --preconditioners radial_xi_lambda_xi_tridi
```

### File structure and best-practice notes

- `MirrorSolveOptions` and `OptimizerOptions` now carry one optional field,
  `residual_block_lambda_maxiter`.
- The field is deliberately narrow: only `block_lsmr` uses it.
- `optimizers.py` keeps old behavior when the field is `None`.
- `mout.py` writes the option only when it is set, keeping old output metadata
  stable.
- The convergence-grid example is the only CLI currently exposing the option,
  because this is a diagnostic benchmarking control rather than a general
  production knob.

### Best next steps

1. Commit and push M8x.
2. Move from budget control to a better scalable lambda-block correction:
   - inspect the lambda block spectrum/condition estimates;
   - test stronger lambda `xi` and radial smoothers only after measuring the
     block residuals;
   - consider a small structured lambda-block preconditioner instead of raw
     LSMR on the Hessian-vector product;
   - keep block-dense as the correctness reference.
3. Avoid more blind `ns=9`, `nxi=17` Krylov budget sweeps until the lambda
   preconditioner/operator is improved.

### Completion percentages after M8x

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `84%`.
- Fixed-boundary axisymmetric solve: `88%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `78%`.
- Plotting and `vmec --plot` mirror support: `79%`.
- I/O schema and docs: `80%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `15%`.
- Free-boundary mirror lane: `5%`.
- Stellarator-mirror hybrid lane: `10%`.
- ESSOS circular-coil mirror beta scan: `0%`.
- PR merge readiness overall: `78%`.

### User input needed

No user input is needed.

---

## 58. 2026-06-17 M8y lambda smoother strength diagnostic

This lane tested whether the existing lambda-xi smoother could materially
improve the moderate finite-current `block_lsmr` row before adding a new
structured lambda-block preconditioner.

### Steps taken

- Used the M8x split-budget setup:
  - `ns=9`;
  - `nxi=17`;
  - `I'=0.01`;
  - radius block budget `102`;
  - lambda block budget `249`;
  - `block_lsmr`;
  - `radial_xi_lambda_xi_tridi`.
- Compared baseline M8x smoothing (`lambda_alpha=0.5`, `xi_alpha=1.0`) with
  stronger lambda smoothing:
  - `lambda_alpha=2.0`, `xi_alpha=2.0`;
  - `lambda_alpha=4.0`, `xi_alpha=4.0`.
- Ran short three-step probes to compare early residual reduction.
- Ran six-step rows for alpha 2 and alpha 4.
- Generated the full standard plot bundle for the alpha-2 six-step row.

### Results obtained

Three-step comparison:

| row | final residual at iteration 3 | final `fsq` | last condition estimate |
| --- | ---: | ---: | ---: |
| baseline alpha 0.5 / xi 1.0 | `5.288281736435e-03` | `1.123129466824e-07` | n/a |
| alpha 2 / xi 2 | `2.924857182997e-03` | `3.435658450173e-08` | `1.313407914284e+05` |
| alpha 4 / xi 4 | `2.656280989205e-03` | `2.833666142013e-08` | `8.423944279065e+05` |

Six-step comparison:

| row | final residual | final `fsq` | normalized force | last condition estimate |
| --- | ---: | ---: | ---: | ---: |
| baseline alpha 0.5 / xi 1.0 | `3.126838214005e-03` | `3.926553099020e-08` | `8.661032213469e-03` | `1.922836515180e+06` |
| alpha 2 / xi 2 | `2.292622037387e-03` | `2.110889882053e-08` | `6.356914852152e-03` | `2.205966918221e+05` |
| alpha 4 / xi 4 | `2.243485142828e-03` | `2.021375737386e-08` | `6.221477478322e-03` | `5.204567175882e+05` |

Interpretation:

- Stronger lambda smoothing is beneficial for this finite-current moderate
  row.
- Alpha 2 gives most of the residual improvement while reducing the reported
  condition estimate by nearly an order of magnitude relative to the baseline
  split-budget row.
- Alpha 4 gives a small additional residual improvement but a worse condition
  estimate than alpha 2 and higher observed runtime.
- The residual remains almost entirely lambda dominated in all rows, so alpha
  tuning alone is not the final production answer.
- The next scalable solver lane should add or test a structured lambda-block
  preconditioner/operator, using alpha 2 as a balanced benchmark setting and
  block-dense as the correctness reference.

Plotted alpha-2 row:

| quantity | value |
| --- | ---: |
| final residual | `2.292622037387e-03` |
| final `fsq` | `2.110889882053e-08` |
| normalized force | `6.356914852152e-03` |
| residual `a` norm | `2.358844253055e-05` |
| residual lambda norm | `2.292500685211e-03` |
| lambda residual fraction | `0.999947068390` |
| minimum `sqrt(g)` | `2.997256222477e-03` |
| mirror ratio | `22.063595093559` |

Generated artifacts:

- `results/mirror/m8y_lambda_smoother_alpha2_probe/residual_newton_convergence_grid_metrics.json`.
- `results/mirror/m8y_lambda_smoother_alpha2_plots/residual_newton_convergence_grid_metrics.json`.
- `results/mirror/m8y_lambda_smoother_alpha2_plots/residual_newton_convergence_history.png`.
- `results/mirror/m8y_lambda_smoother_alpha2_plots/residual_newton_convergence_components.png`.
- `results/mirror/m8y_lambda_smoother_alpha2_plots/best_finite_current_lambda_smoother_alpha2_m8y_residual_newton/figures/best_finite_current_lambda_smoother_alpha2_m8y_residual_newton_mirror_boundary_3d.png`.
- `results/mirror/m8y_lambda_smoother_alpha2_plots/best_finite_current_lambda_smoother_alpha2_m8y_residual_newton/figures/best_finite_current_lambda_smoother_alpha2_m8y_residual_newton_mirror_bfield_boundary.png`.
- `results/mirror/m8y_lambda_smoother_alpha2_plots/best_finite_current_lambda_smoother_alpha2_m8y_residual_newton/figures/best_finite_current_lambda_smoother_alpha2_m8y_residual_newton_mirror_bmag_sxi.png`.
- `results/mirror/m8y_lambda_smoother_alpha2_plots/best_finite_current_lambda_smoother_alpha2_m8y_residual_newton/figures/best_finite_current_lambda_smoother_alpha2_m8y_residual_newton_mirror_cross_sections.png`.
- `results/mirror/m8y_lambda_smoother_alpha4_probe/residual_newton_convergence_grid_metrics.json`.
- `results/mirror/m8y_lambda_smoother_alpha4_six_probe/residual_newton_convergence_grid_metrics.json`.

Visual validation:

- Alpha-2 residual history decreases but still stalls above tight convergence.
- Horizontal-`z` geometry remains well ordered.
- `|B|` remains strongest near the end caps and weakest near the center.
- B-direction field-line traces remain visible.
- Cross sections remain nested and circular.

### How it was tested

This was a benchmark/log lane; no source changes were made after M8x.

Commands:

```bash
JAX_ENABLE_X64=1 python examples/mirror_residual_newton_convergence_grid.py \
  --outdir results/mirror/m8y_lambda_smoother_alpha2_probe \
  --ns-array 9 \
  --nxi-array 17 \
  --maxiter-array 3 \
  --residual-linear-maxiter-array 102 \
  --residual-block-lambda-maxiter 249 \
  --residual-linear-maxiter-policy fixed \
  --residual-linear-solver block_lsmr \
  --residual-lambda-alpha 2.0 \
  --residual-xi-alpha 2.0 \
  --i-prime 0.01 \
  --case-label finite_current_lambda_smoother_alpha2_m8y \
  --preconditioners radial_xi_lambda_xi_tridi \
  --no-plots
JAX_ENABLE_X64=1 python examples/mirror_residual_newton_convergence_grid.py \
  --outdir results/mirror/m8y_lambda_smoother_alpha2_plots \
  --ns-array 9 \
  --nxi-array 17 \
  --maxiter-array 6 \
  --residual-linear-maxiter-array 102 \
  --residual-block-lambda-maxiter 249 \
  --residual-linear-maxiter-policy fixed \
  --residual-linear-solver block_lsmr \
  --residual-lambda-alpha 2.0 \
  --residual-xi-alpha 2.0 \
  --i-prime 0.01 \
  --case-label finite_current_lambda_smoother_alpha2_m8y \
  --preconditioners radial_xi_lambda_xi_tridi
JAX_ENABLE_X64=1 python examples/mirror_residual_newton_convergence_grid.py \
  --outdir results/mirror/m8y_lambda_smoother_alpha4_probe \
  --ns-array 9 \
  --nxi-array 17 \
  --maxiter-array 3 \
  --residual-linear-maxiter-array 102 \
  --residual-block-lambda-maxiter 249 \
  --residual-linear-maxiter-policy fixed \
  --residual-linear-solver block_lsmr \
  --residual-lambda-alpha 4.0 \
  --residual-xi-alpha 4.0 \
  --i-prime 0.01 \
  --case-label finite_current_lambda_smoother_alpha4_m8y \
  --preconditioners radial_xi_lambda_xi_tridi \
  --no-plots
JAX_ENABLE_X64=1 python examples/mirror_residual_newton_convergence_grid.py \
  --outdir results/mirror/m8y_lambda_smoother_alpha4_six_probe \
  --ns-array 9 \
  --nxi-array 17 \
  --maxiter-array 6 \
  --residual-linear-maxiter-array 102 \
  --residual-block-lambda-maxiter 249 \
  --residual-linear-maxiter-policy fixed \
  --residual-linear-solver block_lsmr \
  --residual-lambda-alpha 4.0 \
  --residual-xi-alpha 4.0 \
  --i-prime 0.01 \
  --case-label finite_current_lambda_smoother_alpha4_six_m8y \
  --preconditioners radial_xi_lambda_xi_tridi \
  --no-plots
```

### File structure and best-practice notes

- No source files changed in this lane.
- The existing convergence-grid example and plot bundle were sufficient.
- The M8x CLI option made the alpha comparison cheaper by avoiding unnecessary
  high budgets on the radius block.

### Best next steps

1. Commit and push the M8y plan log.
2. Implement a structured lambda-block preconditioner/operator instead of more
   alpha sweeps:
   - use the existing tridiagonal smoother as a baseline;
   - consider a separable radial/xi lambda inverse with tunable diagonal shift;
   - compare against block-dense on `ns=5,nxi=9` and `ns=9,nxi=17`;
   - keep alpha 2 / xi 2 as the balanced current matrix-free benchmark.
3. Continue to plot residual histories and horizontal-`z` field-line bundles
   for each new benchmark row.

### Completion percentages after M8y

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `84%`.
- Fixed-boundary axisymmetric solve: `88%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `78%`.
- Plotting and `vmec --plot` mirror support: `79%`.
- I/O schema and docs: `80%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `15%`.
- Free-boundary mirror lane: `5%`.
- Stellarator-mirror hybrid lane: `10%`.
- ESSOS circular-coil mirror beta scan: `0%`.
- PR merge readiness overall: `78%`.

### User input needed

No user input is needed.

---

## 59. 2026-06-17 M8z linear-helper extraction and rejected Helmholtz prototype

This lane simplified the residual-Newton optimizer file and checked one
structured lambda-preconditioner idea.  The preconditioner idea was rejected
and not retained; the file-structure refactor was retained.

### Steps taken

- Prototyped a local, uncommitted separable 2-D Helmholtz smoother for the
  lambda block.
- Tested it on the same moderate finite-current row used in M8x/M8y.
- Removed the Helmholtz prototype because it was worse than the existing
  sequential tridiagonal lambda smoother.
- Extracted matrix-free residual-Newton linear correction helpers from
  `optimizers.py` into `vmec_jax/mirror/solvers/fixed_boundary/linear.py`.
- Added a small `ResidualLinearSolve` dataclass so full Krylov and block-LSMR
  corrections share one return shape and one history-bookkeeping path.
- Kept SciPy sparse least-squares imports lazy inside `linear.py`.

### Results obtained

Rejected Helmholtz prototype:

| row | final residual at iteration 3 | final `fsq` | last condition estimate |
| --- | ---: | ---: | ---: |
| existing tridi alpha 2 / xi 2 | `2.924857182997e-03` | `3.435658450173e-08` | `1.313407914284e+05` |
| local Helmholtz alpha 2 / xi 2 | `5.247110193978e-03` | `1.105709453323e-07` | `1.768718650231e+05` |
| local Helmholtz alpha 8 / xi 8 | `3.607063204332e-03` | `5.225263036162e-08` | `6.241977553168e+05` |

Interpretation:

- The tested separable Helmholtz inverse did not beat the existing sequential
  tridiagonal smoother.
- The alpha-8 Helmholtz row improved over alpha-2 Helmholtz but remained worse
  than the M8y tridiagonal alpha-2 row and had a worse condition estimate.
- The prototype was not committed.  Its temporary result files are local
  investigation artifacts, not part of the reproducible public interface.

File-size impact:

| file | before | after |
| --- | ---: | ---: |
| `optimizers.py` | 1121 lines | 992 lines |
| `linear.py` | n/a | 169 lines |

### How it was tested

Automated tests:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_fixed_boundary_axisym.py::test_residual_newton_block_lsmr_solver_improves_perturbed_cylinder \
  tests/mirror/test_mirror_fixed_boundary_axisym.py::test_residual_newton_solver_reaches_tight_residual_for_perturbed_cylinder \
  tests/mirror/test_mirror_fixed_boundary_axisym.py::test_residual_newton_dense_lstsq_solver_improves_perturbed_cylinder \
  tests/mirror/test_mirror_low_level_coverage.py::test_low_level_field_energy_residual_and_optimizer_guards \
  -q
```

Result: `4 passed in 29.94s`.

Broader nearby tests:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_fixed_boundary_axisym.py \
  tests/mirror/test_mirror_examples.py::test_root_residual_newton_convergence_grid_runs_without_plots \
  -q
```

Result: `15 passed in 41.91s`.

Lint/format/whitespace:

```bash
python -m ruff format \
  vmec_jax/mirror/solvers/fixed_boundary/linear.py \
  vmec_jax/mirror/solvers/fixed_boundary/optimizers.py
python -m ruff check \
  vmec_jax/mirror/solvers/fixed_boundary/linear.py \
  vmec_jax/mirror/solvers/fixed_boundary/optimizers.py
git diff --check
```

Result: all checks passed.

### File structure and best-practice notes

- `linear.py` now owns reduced linear correction mechanics and diagnostics.
- `optimizers.py` remains the nonlinear residual-Newton driver and no longer
  carries the full/block Krylov operator construction bodies inline.
- No public API changes were made in this lane.
- The rejected Helmholtz prototype was removed to avoid adding an unhelpful
  solver mode.

### Best next steps

1. Commit and push M8z.
2. Continue lambda-block work from measured evidence:
   - keep the sequential tridiagonal alpha-2 benchmark as the current best
     matrix-free moderate row;
   - do not reintroduce the tested Helmholtz smoother unless a different
     operator form is justified;
   - look next at block residual scaling, line-search damping, or hybrid
     dense/matrix-free reference strategies.
3. Start preparing the M9 mirror straight-field-line/Boozer-like diagnostic
   lane once fixed-boundary solver notes are current.

### Completion percentages after M8z

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `84%`.
- Fixed-boundary axisymmetric solve: `88%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `78%`.
- Plotting and `vmec --plot` mirror support: `79%`.
- I/O schema and docs: `80%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `15%`.
- Free-boundary mirror lane: `5%`.
- Stellarator-mirror hybrid lane: `10%`.
- ESSOS circular-coil mirror beta scan: `0%`.
- PR merge readiness overall: `78%`.

### User input needed

No user input is needed.

---

## 60. 2026-06-17 M9 open-field pitch diagnostics

This lane added the first mirror straight-field-line/Boozer-like diagnostic:
a radial cap-to-cap field-line pitch profile.  It is deliberately named as an
open-field pitch diagnostic, not toroidal rotational transform.

### Steps taken

- Added `MirrorFieldLinePitchProfileData`.
- Added `mirror_field_line_pitch_profile_data(output, num_lines=6)`.
- The new helper traces field lines on every radial surface using
  `dtheta/dxi = B^theta / B^xi` and reports:
  - mean cap-to-cap theta advance;
  - min/max cap-to-cap theta advance;
  - mean cap-to-cap turns.
- Extended `MirrorRadialDiagnosticsData` with:
  - `field_line_theta_advance`;
  - `field_line_turns`.
- Updated the standard radial diagnostics plot to show both:
  - `I'/Psi'` profile proxy;
  - measured cap-to-cap field-line turns.
- Exported the new helper through `vmec_jax.mirror.plotting`.
- Updated plotting and finite-current example tests.
- Regenerated the finite-current pitch example with plots.

### Results obtained

Finite-current pitch example:

| quantity | value |
| --- | ---: |
| mean boundary theta advance | `3.433610700154` |
| mean boundary turns | `0.546476115583` |
| `I'/Psi'` profile proxy mean | `1.716805350077` |
| final residual | `5.937093692157e-02` |
| final `fsq` | `6.980016140486e-06` |
| normalized force | `1.113139548331e-01` |
| minimum `sqrt(g)` | `3.228109210942e-03` |
| mirror output mirror ratio | `14.072058315583` |

Generated artifacts:

- `results/mirror/m9_field_line_pitch/mout_finite_current_pitch.nc`.
- `results/mirror/m9_field_line_pitch/finite_current_pitch_metrics.json`.
- `results/mirror/m9_field_line_pitch/figures/finite_current_pitch_mirror_radial_diagnostics.png`.
- `results/mirror/m9_field_line_pitch/figures/finite_current_pitch_mirror_bfield_boundary.png`.
- `results/mirror/m9_field_line_pitch/figures/finite_current_pitch_mirror_boundary_3d.png`.
- `results/mirror/m9_field_line_pitch/figures/finite_current_pitch_mirror_bmag_sxi.png`.
- `results/mirror/m9_field_line_pitch/figures/finite_current_pitch_mirror_cross_sections.png`.
- `results/mirror/m9_field_line_pitch/figures/finite_current_pitch_theta_advance.png`.

Visual validation:

- The radial diagnostics plot now shows cap-to-cap field-line turns alongside
  `I'/Psi'`.
- The boundary B-direction plot still shows visible cap-to-cap field-line
  traces.
- Horizontal-`z` geometry and `|B|` plots render through the standard bundle.

### How it was tested

Automated tests:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_plotting.py \
  tests/mirror/test_mirror_examples.py::test_root_finite_current_pitch_example_runs_without_plots \
  -q
```

Result: `5 passed in 6.11s`.

Focused tests before the full plotting file:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_plotting.py::test_mirror_plot_data_helpers_expose_numerical_content \
  tests/mirror/test_mirror_examples.py::test_root_finite_current_pitch_example_runs_without_plots \
  -q
```

Result: `2 passed in 2.68s`.

Lint/format/whitespace:

```bash
python -m ruff check \
  vmec_jax/mirror/plotting/diagnostics.py \
  vmec_jax/mirror/plotting/__init__.py \
  tests/mirror/test_mirror_plotting.py \
  tests/mirror/test_mirror_examples.py
python -m ruff format --check \
  vmec_jax/mirror/plotting/diagnostics.py \
  vmec_jax/mirror/plotting/__init__.py \
  tests/mirror/test_mirror_plotting.py \
  tests/mirror/test_mirror_examples.py
git diff --check
```

Result: all checks passed.

Example command:

```bash
JAX_ENABLE_X64=1 python examples/mirror_finite_current_pitch.py \
  --outdir results/mirror/m9_field_line_pitch \
  --maxiter 0
```

### File structure and best-practice notes

- The new pitch profile lives in `plotting/diagnostics.py` because it is a
  diagnostic derived from output arrays, not a solver dependency.
- `plotting/bfield.py` still owns physical boundary field-line traces used for
  3-D visualization.
- The diagnostic is open-field and cap-to-cap; docs and labels avoid calling
  it toroidal iota.

### Best next steps

1. Commit and push M9.
2. Add the pitch profile to any downstream CSV/NPZ export if needed for
   analysis scripts.
3. Start the finite M10/M11 cleanup:
   - document dense/block-dense as correctness references;
   - document matrix-free block LSMR as diagnostic;
   - ensure mirror docs mention cap-to-cap pitch rather than toroidal iota.
4. Then move to the planned free-boundary/hybrid lanes.

### Completion percentages after M9

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `84%`.
- Fixed-boundary axisymmetric solve: `88%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `81%`.
- Plotting and `vmec --plot` mirror support: `82%`.
- I/O schema and docs: `80%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `35%`.
- Free-boundary mirror lane: `5%`.
- Stellarator-mirror hybrid lane: `10%`.
- ESSOS circular-coil mirror beta scan: `0%`.
- PR merge readiness overall: `80%`.

### User input needed

No user input is needed.

---

## 61. 2026-06-17 M10/M11 mirror documentation and status cleanup

This lane synchronized the public mirror documentation with the current code
state after the dense/block-dense residual references, matrix-free block-LSMR
diagnostics, split lambda-block budgets, and cap-to-cap open-field pitch
diagnostics.

### Steps taken

- Reviewed the active branch status, recent commits, and plan ordering through
  section 60.
- Updated `docs/mirror/overview.rst` to describe:
  - dense and block-dense residual-Newton reference solves;
  - matrix-free LSMR/LSQR/block-LSMR as diagnostic scalable paths;
  - adaptive and split radius/lambda inner linear-solve budgets;
  - cap-to-cap field-line overlays and open-field pitch radial diagnostics;
  - later differentiable optimization, free-boundary, hybrid, and ESSOS lanes.
- Updated `docs/mirror/index.rst` to include cap-to-cap pitch diagnostics in
  the experimental feature warning and to name stellarator-mirror hybrid
  boundaries as planned work.
- Updated `docs/mirror/outputs.rst` to document:
  - optional split lambda-block iteration metadata;
  - the standard mirror plot bundle;
  - the distinction between open-field cap-to-cap pitch and toroidal iota;
  - the absence of toroidal Boozer coordinates in current mirror `mout` files.
- Wrapped a long line in `examples/mirror/README.md` while preserving content.

### Results obtained

- Documentation now matches the implemented solver and plotting status.
- The docs no longer imply matrix-free block-LSMR is a production tight solve
  on the moderate finite-current row.
- The pitch language is explicit: the code reports cap-to-cap field-line
  advance/turns for open mirror lines, not toroidal rotational transform.
- The single plan remains ordered after section 60 and now records this cleanup
  as a finite tranche.

### How it was tested

Whitespace:

```bash
git diff --check
```

Result: passed.

Sphinx docs with warnings as errors:

```bash
python -m sphinx -W -j auto -b html docs docs/_build/html
```

Result: build succeeded.

### File structure and best-practice notes

- Status documentation stays under `docs/mirror/`:
  - `index.rst` owns the experimental status warning;
  - `overview.rst` owns architecture and solver-status narrative;
  - `outputs.rst` owns mirror-native `mout` schema and plot-bundle behavior.
- User-facing runnable examples remain documented in `examples/mirror/README.md`.
- No solver code was changed in this lane.

### Best next steps

1. Commit and push this documentation/status cleanup.
2. Resume implementation work with the next finite lane:
   - export open-field pitch diagnostics to analysis artifacts if needed;
   - continue simplifying fixed-boundary solver files without changing
     behavior;
   - begin a small, tested free-boundary data-model/coil-field bridge that can
     support the ESSOS circular-coil beta scan lane;
   - keep the stellarator-mirror hybrid boundary plan after the current
     fixed-boundary MVP gates unless user priorities change.

### Completion percentages after M10/M11 cleanup

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `84%`.
- Fixed-boundary axisymmetric solve: `88%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `81%`.
- Plotting and `vmec --plot` mirror support: `83%`.
- I/O schema and docs: `86%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `35%`.
- Free-boundary mirror lane: `5%`.
- Stellarator-mirror hybrid lane: `10%`.
- ESSOS circular-coil mirror beta scan: `0%`.
- PR merge readiness overall: `81%`.

### User input needed

No user input is needed.

---

## 62. 2026-06-17 M11a open-field pitch analysis exports

This lane made the M9 open-field pitch diagnostics available in lightweight
analysis exports, so downstream scripts can read the same cap-to-cap pitch
signals that the standard radial plot already displays.

### Steps taken

- Reused `mirror_radial_diagnostics_data` inside
  `vmec_jax/mirror/plotting/export.py`.
- Added the following radial arrays to `mirror_output_to_npz`:
  - `beta`;
  - `iota_like_twist`;
  - `field_line_theta_advance`;
  - `field_line_turns`;
  - `mean_bmag`;
  - `magnetic_well_proxy`;
  - `fsq`;
  - `normalized_force`.
- Extended `mirror_axisym_slice_to_csv` so each `(s,xi)` row also carries:
  - beta;
  - `I'/Psi'`;
  - cap-to-cap theta advance;
  - cap-to-cap turns;
  - surface-mean `|B|`;
  - magnetic-well proxy.
- Updated `docs/mirror/outputs.rst` to document the expanded `.npz` and CSV
  analysis artifacts.
- Extended the I/O export test to pin the new keys, radial shapes, and
  zero-current pitch behavior.

### Results obtained

- `.npz` exports now include the same radial beta/twist/pitch/well quantities
  shown in the standard diagnostics plot.
- CSV exports remain a single flat table for simple plotting tools; surface
  quantities are repeated across each axial point for that surface.
- Zero-current examples export zero cap-to-cap turns, matching the pitch helper
  and radial diagnostic plot.
- No new plot type was introduced in this lane; it exports existing plotted
  quantities for analysis.

### How it was tested

Focused tests:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_io.py::test_mirror_output_exports_npz_and_axisym_csv \
  tests/mirror/test_mirror_plotting.py::test_mirror_plot_data_helpers_expose_numerical_content \
  -q
```

Result: `2 passed in 1.37s`.

Full mirror I/O file:

```bash
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_io.py -q
```

Result: `4 passed in 3.09s`.

Lint/format/docs/whitespace:

```bash
python -m ruff check \
  vmec_jax/mirror/plotting/export.py \
  tests/mirror/test_mirror_io.py
python -m ruff format --check \
  vmec_jax/mirror/plotting/export.py \
  tests/mirror/test_mirror_io.py
python -m sphinx -W -j auto -b html docs docs/_build/html
git diff --check
```

Result: all checks passed.

### File structure and best-practice notes

- No new export module was added.  The high-level export entry points remain in
  `plotting/export.py`.
- Derived quantities come from `plotting/diagnostics.py`, keeping pitch
  calculation in one place.
- I/O tests own the export-schema checks because `.npz`/CSV are output
  artifacts, while plotting tests continue to own plot-data helper behavior.

### Best next steps

1. Commit and push this export lane.
2. Continue fixed-boundary simplification without behavior changes.
3. Start the free-boundary mirror skeleton with the smallest useful data model:
   circular coil sets, external-field sampling, and a placeholder LCFS/beta
   scan driver wired to tests before ESSOS integration.

### Completion percentages after M11a

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `84%`.
- Fixed-boundary axisymmetric solve: `88%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `84%`.
- I/O schema and docs: `87%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `5%`.
- Stellarator-mirror hybrid lane: `10%`.
- ESSOS circular-coil mirror beta scan: `0%`.
- PR merge readiness overall: `82%`.

### User input needed

No user input is needed.

---

## 63. 2026-06-17 M12a mirror free-boundary circular-coil bridge

This lane started the free-boundary mirror path with a tested bridge rather
than an unvalidated LCFS solve.  The new code builds ESSOS-compatible circular
coil parameters, samples their external field on mirror grids, and records the
planned 1%, 3%, and 10% beta scan cases.

### Steps taken

- Added `vmec_jax/mirror/free_boundary.py`.
- Added `MirrorCircularCoils` for axisymmetric circular-loop mirror coil sets.
- Added conversion from circular coils to `CoilFieldParams` using the ESSOS
  Fourier convention:
  - `x = R cos(2 pi t)`;
  - `y = R sin(2 pi t)`;
  - `z = z_center`.
- Added `MirrorExternalFieldSample`.
- Added `sample_mirror_axis_external_field`.
- Added `sample_mirror_boundary_external_field`.
- Added `MirrorFreeBoundaryBetaCase` and
  `make_mirror_free_boundary_beta_cases`, defaulting to 1%, 3%, and 10%.
- Exported the new helpers through `vmec_jax.mirror`.
- Added a root-level example:
  `examples/mirror_free_boundary_circular_coils.py`.
- Updated `examples/mirror/README.md` and mirror docs to describe the bridge as
  a planning fixture, not a free-boundary LCFS solve.
- Added tests for:
  - ESSOS-compatible circular-coil coefficients;
  - direct-coil on-axis parity against the analytic two-coil field;
  - mirror boundary field sampling shapes;
  - default beta-scan metadata;
  - the new root example's no-plot smoke path.

### Results obtained

Generated example artifacts:

- `results/mirror/m12_free_boundary_circular_coils/free_boundary_circular_coils_metrics.json`.
- `results/mirror/m12_free_boundary_circular_coils/figures/free_boundary_circular_coils_axis_bz.png`.
- `results/mirror/m12_free_boundary_circular_coils/figures/free_boundary_circular_coils_boundary_bmag.png`.
- `results/mirror/m12_free_boundary_circular_coils/figures/free_boundary_circular_coils_geometry.png`.

Example metrics:

| quantity | value |
| --- | ---: |
| on-axis `B_z` relative Linf error | `9.098256159668e-16` |
| minimum axis `|B_z|` | `1.294393812392e-01` |
| maximum axis `|B_z|` | `1.804391293833e+00` |
| minimum boundary `|B|` | `1.053675512271e-01` |
| maximum boundary `|B|` | `1.879005770409e+00` |

Visual validation:

- The on-axis plot overlays the direct-coil bridge and analytic two-coil field.
- The boundary `|B|` plot is poloidally symmetric for circular coils.
- The 3-D geometry plot shows the mirror horizontally with `z` as the long axis
  and circular coils at the caps.

### How it was tested

Focused tests:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_example_runs_without_plots \
  tests/test_external_fields_essos_adapter.py \
  -q
```

Result: `13 passed in 6.30s`.

Example with plots:

```bash
JAX_ENABLE_X64=1 python examples/mirror_free_boundary_circular_coils.py \
  --outdir results/mirror/m12_free_boundary_circular_coils \
  --ntheta 24 \
  --nxi 33 \
  --n-segments 256
```

Result: metrics and three PNG figures written.

Lint/format/docs/whitespace:

```bash
python -m ruff check \
  vmec_jax/mirror/free_boundary.py \
  tests/mirror/test_mirror_free_boundary.py \
  examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_examples.py \
  vmec_jax/mirror/api.py \
  vmec_jax/mirror/__init__.py
python -m ruff format --check \
  vmec_jax/mirror/free_boundary.py \
  tests/mirror/test_mirror_free_boundary.py \
  examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_examples.py \
  vmec_jax/mirror/api.py \
  vmec_jax/mirror/__init__.py
python -m sphinx -W -j auto -b html docs docs/_build/html
git diff --check
```

Result: all checks passed.

### File structure and best-practice notes

- The bridge is one mirror-domain module, `vmec_jax/mirror/free_boundary.py`.
- Existing toroidal external-field code remains the owner of direct-coil
  Biot-Savart sampling and ESSOS-compatible Fourier conventions.
- Mirror-specific code only adapts circular mirror coils and mirror grids to
  that external-field provider interface.
- The example is root-level because it is a user-facing planning fixture, but
  it does not write a `mout` file or claim an equilibrium solve.

### Best next steps

1. Commit and push M12a.
2. Extend the bridge toward the ESSOS lane:
   - allow loading circular-coil data from an ESSOS-like object or JSON;
   - write a free-boundary beta-scan driver that records the 1%, 3%, and 10%
     case setup without claiming LCFS convergence;
   - add the first LCFS update strategy only after the boundary condition and
     force-balance target are explicitly defined and tested.
3. Keep the stellarator-mirror hybrid boundary lane after the current
   fixed-boundary and circular-coil free-boundary bridge are stable.

### Completion percentages after M12a

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `84%`.
- Fixed-boundary axisymmetric solve: `88%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `84%`.
- I/O schema and docs: `88%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `18%`.
- Stellarator-mirror hybrid lane: `10%`.
- ESSOS circular-coil mirror beta scan: `8%`.
- PR merge readiness overall: `83%`.

### User input needed

No user input is needed.

---

## 64. 2026-06-17 M12b circular-coil beta-scan setup JSON

This lane made the M12a circular-coil bridge reusable by adding a serializable
scan setup artifact.  The setup records circular coil geometry, currents,
direct-coil quadrature settings, and the 1%, 3%, and 10% beta cases without
claiming a free-boundary LCFS solve.

### Steps taken

- Added JSON-friendly `to_dict`/`from_dict` methods for `MirrorCircularCoils`.
- Added JSON-friendly `to_dict`/`from_dict` methods for
  `MirrorFreeBoundaryBetaCase`.
- Added `MirrorFreeBoundaryCircularCoilScan`.
- Added `make_mirror_free_boundary_circular_coil_scan`.
- Added `write_mirror_free_boundary_circular_coil_scan`.
- Added `load_mirror_free_boundary_circular_coil_scan`.
- Exported the scan setup helpers through `vmec_jax.mirror`.
- Updated `examples/mirror_free_boundary_circular_coils.py` to write
  `free_boundary_circular_coils_setup.json`.
- Updated the example README and mirror overview docs.
- Added JSON roundtrip tests and extended the root example smoke test to load
  the generated setup JSON.

### Results obtained

Generated example artifacts:

- `results/mirror/m12b_free_boundary_circular_coil_setup/free_boundary_circular_coils_metrics.json`.
- `results/mirror/m12b_free_boundary_circular_coil_setup/free_boundary_circular_coils_setup.json`.
- `results/mirror/m12b_free_boundary_circular_coil_setup/figures/free_boundary_circular_coils_axis_bz.png`.
- `results/mirror/m12b_free_boundary_circular_coil_setup/figures/free_boundary_circular_coils_boundary_bmag.png`.
- `results/mirror/m12b_free_boundary_circular_coil_setup/figures/free_boundary_circular_coils_geometry.png`.

The setup JSON contains:

- two circular coils at `z = -1.0` and `z = 1.0`;
- radius `0.35`;
- currents `1.0e6`;
- `n_segments = 256`;
- beta cases at 1%, 3%, and 10%;
- `status = "setup_only_no_lcfs_solve"`.

Example metrics remained unchanged from the bridge validation:

| quantity | value |
| --- | ---: |
| on-axis `B_z` relative Linf error | `9.098256159668e-16` |
| minimum boundary `|B|` | `1.053675512271e-01` |
| maximum boundary `|B|` | `1.879005770409e+00` |

### How it was tested

Focused tests:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_example_runs_without_plots \
  -q
```

Result: `6 passed in 3.61s`.

Example with plots and setup JSON:

```bash
JAX_ENABLE_X64=1 python examples/mirror_free_boundary_circular_coils.py \
  --outdir results/mirror/m12b_free_boundary_circular_coil_setup \
  --ntheta 24 \
  --nxi 33 \
  --n-segments 256
```

Result: setup JSON, metrics JSON, and three PNG figures written.

Lint/format/docs/whitespace:

```bash
python -m ruff format \
  vmec_jax/mirror/free_boundary.py \
  tests/mirror/test_mirror_free_boundary.py \
  examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_examples.py \
  vmec_jax/mirror/api.py \
  vmec_jax/mirror/__init__.py
python -m ruff check \
  vmec_jax/mirror/free_boundary.py \
  tests/mirror/test_mirror_free_boundary.py \
  examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_examples.py \
  vmec_jax/mirror/api.py \
  vmec_jax/mirror/__init__.py
python -m sphinx -W -j auto -b html docs docs/_build/html
git diff --check
```

Result: all checks passed.

### File structure and best-practice notes

- The setup serializer stays beside the bridge in `vmec_jax/mirror/free_boundary.py`.
- The JSON artifact is intentionally plain and small so ESSOS or future scripts
  can generate it without importing the full solve stack.
- The status flag makes the current limitation explicit: setup only, no LCFS
  solve yet.

### Best next steps

1. Commit and push M12b.
2. Add a setup-to-fixed-boundary initializer:
   - use the sampled on-axis vacuum field to build an initial flux-tube
     boundary;
   - attach one beta case at a time;
   - run the existing fixed-boundary solve as the controlled pre-LCFS baseline.
3. Only then add a free-boundary LCFS update target.

### Completion percentages after M12b

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `84%`.
- Fixed-boundary axisymmetric solve: `88%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `84%`.
- I/O schema and docs: `88%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `22%`.
- Stellarator-mirror hybrid lane: `10%`.
- ESSOS circular-coil mirror beta scan: `12%`.
- PR merge readiness overall: `84%`.

### User input needed

No user input is needed.

---

## 65. 2026-06-17 M12c setup-to-fixed-boundary initializer

This lane factored the circular-coil setup-to-boundary step out of the example
and into tested library helpers.  A scan setup can now produce the initial
fixed-boundary flux-tube surface from sampled on-axis external field data.

### Steps taken

- Added `mirror_boundary_from_external_axis_field`.
- Added `initial_mirror_boundary_from_circular_coil_scan`.
- Exported both helpers through `vmec_jax.mirror`.
- Refactored `examples/mirror_free_boundary_circular_coils.py` so it builds
  its initial boundary through the scan setup helper instead of duplicating the
  flux-tube calculation inline.
- Added a test that compares the scan initializer against the analytic
  two-coil flux-tube boundary.
- Updated the example README and mirror overview docs.

### Results obtained

- The setup JSON is now enough to reconstruct the circular-coil scan and build
  the fixed-boundary initial surface.
- The initializer reproduces the analytic two-coil flux-tube boundary to test
  tolerance on the sampled mirror grid.
- The example still writes the same three plots and metrics, now through the
  shared initializer path.

Generated example artifacts:

- `results/mirror/m12c_setup_to_fixed_boundary_initializer/free_boundary_circular_coils_metrics.json`.
- `results/mirror/m12c_setup_to_fixed_boundary_initializer/free_boundary_circular_coils_setup.json`.
- `results/mirror/m12c_setup_to_fixed_boundary_initializer/figures/free_boundary_circular_coils_axis_bz.png`.
- `results/mirror/m12c_setup_to_fixed_boundary_initializer/figures/free_boundary_circular_coils_boundary_bmag.png`.
- `results/mirror/m12c_setup_to_fixed_boundary_initializer/figures/free_boundary_circular_coils_geometry.png`.

Example metrics:

| quantity | value |
| --- | ---: |
| on-axis `B_z` relative Linf error | `9.098256159668e-16` |
| minimum boundary `|B|` | `1.053675512271e-01` |
| maximum boundary `|B|` | `1.879005770409e+00` |

### How it was tested

Focused tests:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_example_runs_without_plots \
  -q
```

Result: `7 passed in 4.00s`.

Example with plots:

```bash
JAX_ENABLE_X64=1 python examples/mirror_free_boundary_circular_coils.py \
  --outdir results/mirror/m12c_setup_to_fixed_boundary_initializer \
  --ntheta 24 \
  --nxi 33 \
  --n-segments 256
```

Result: setup JSON, metrics JSON, and three PNG figures written.

Lint/format/docs/whitespace:

```bash
python -m ruff format \
  vmec_jax/mirror/free_boundary.py \
  tests/mirror/test_mirror_free_boundary.py \
  examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_examples.py \
  vmec_jax/mirror/api.py \
  vmec_jax/mirror/__init__.py
python -m ruff check \
  vmec_jax/mirror/free_boundary.py \
  tests/mirror/test_mirror_free_boundary.py \
  examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_examples.py \
  vmec_jax/mirror/api.py \
  vmec_jax/mirror/__init__.py
python -m sphinx -W -j auto -b html docs docs/_build/html
git diff --check
```

Result: all checks passed.

### File structure and best-practice notes

- The initializer lives with other circular-coil bridge helpers in
  `vmec_jax/mirror/free_boundary.py`.
- The helper accepts a generic axial `B_z` array, so later ESSOS/mgrid fields
  can reuse the same flux-tube initialization.
- The root example remains a planning/diagnostic fixture and still does not
  claim a free-boundary LCFS solve.

### Best next steps

1. Commit and push M12c.
2. Add a controlled beta-case fixed-boundary baseline driver:
   - load scan setup JSON;
   - build the initial boundary;
   - run existing fixed-boundary solves for each beta case at low resolution;
   - plot residual history, cross sections, boundary `|B|`, and beta profiles.
3. Use that baseline to define the free-boundary LCFS update target.

### Completion percentages after M12c

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `84%`.
- Fixed-boundary axisymmetric solve: `88%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `84%`.
- I/O schema and docs: `88%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `25%`.
- Stellarator-mirror hybrid lane: `10%`.
- ESSOS circular-coil mirror beta scan: `15%`.
- PR merge readiness overall: `84%`.

### User input needed

No user input is needed.

---

## 66. 2026-06-17 M12d beta-case fixed-boundary baseline driver

This lane added a controlled pre-LCFS baseline for the circular-coil beta scan.
The root example can now write one low-resolution fixed-boundary `mout` file
per beta case after building the initial flux-tube boundary from the scan
setup.  This is still not a free-boundary LCFS solve.

### Steps taken

- Added `--run-fixed-boundary-baseline` to
  `examples/mirror_free_boundary_circular_coils.py`.
- Added `--baseline-maxiter` and `--baseline-psi-prime` controls.
- For each beta case, the example now:
  - builds the initial boundary from the circular-coil scan;
  - runs `run_mirror_fixed_boundary` with the existing fixed-boundary solver;
  - writes a beta-labeled `mout` file;
  - optionally writes the standard mirror plot bundle for that beta row.
- Made the baseline summary robust for `maxiter=0` by falling back to
  `result.final_trace` when no optimizer summary is present.
- Extended the root example smoke test to exercise the baseline path and check
  that all beta-row `mout` files are written.
- Updated the example README and mirror overview docs.

### Results obtained

Generated example artifacts:

- `results/mirror/m12d_beta_fixed_boundary_baseline/free_boundary_circular_coils_metrics.json`.
- `results/mirror/m12d_beta_fixed_boundary_baseline/free_boundary_circular_coils_setup.json`.
- `results/mirror/m12d_beta_fixed_boundary_baseline/mout_free_boundary_circular_coils_beta_1.nc`.
- `results/mirror/m12d_beta_fixed_boundary_baseline/mout_free_boundary_circular_coils_beta_3.nc`.
- `results/mirror/m12d_beta_fixed_boundary_baseline/mout_free_boundary_circular_coils_beta_10.nc`.
- Standard mirror plot bundles under:
  - `results/mirror/m12d_beta_fixed_boundary_baseline/figures/fixed_boundary_beta_1/`;
  - `results/mirror/m12d_beta_fixed_boundary_baseline/figures/fixed_boundary_beta_3/`;
  - `results/mirror/m12d_beta_fixed_boundary_baseline/figures/fixed_boundary_beta_10/`.

Baseline rows with `baseline_maxiter=0`:

| beta percent | residual | `fsq` | min `sqrt(g)` | mirror ratio |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `1.884728113343e-01` | `9.372559528304e-05` | `3.228109210942e-03` | `13.940049439983` |
| 3 | `4.011165408143e-01` | `4.245236921236e-04` | `3.228109210942e-03` | `13.940049439983` |
| 10 | `1.237976350530e+00` | `4.043761067210e-03` | `3.228109210942e-03` | `13.940049439983` |

Visual validation:

- Representative beta-10 pressure/beta plot rendered.
- Representative beta-10 3-D boundary plot rendered horizontally with `z` as
  the long axis.
- All first 12 inspected baseline PNGs were nonblank.

### How it was tested

Focused smoke test:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_example_runs_without_plots \
  -q
```

Result: `1 passed in 3.02s`.

Focused free-boundary and example tests:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_example_runs_without_plots \
  -q
```

Result: `7 passed in 4.83s`.

Example with plots and baseline outputs:

```bash
JAX_ENABLE_X64=1 python examples/mirror_free_boundary_circular_coils.py \
  --outdir results/mirror/m12d_beta_fixed_boundary_baseline \
  --ntheta 24 \
  --nxi 33 \
  --n-segments 256 \
  --run-fixed-boundary-baseline \
  --baseline-maxiter 0
```

Result: setup JSON, metrics JSON, three `mout` files, and standard plot bundles
written.

Lint/format/docs/whitespace:

```bash
python -m ruff check \
  examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_examples.py
python -m ruff format --check \
  examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_examples.py
python -m sphinx -W -j auto -b html docs docs/_build/html
git diff --check
```

Result: all checks passed.

### File structure and best-practice notes

- The beta baseline remains in the root example because it is a workflow
  fixture, not a new solver API.
- It reuses the existing fixed-boundary solver and mirror `mout` writer instead
  of creating a parallel output path.
- The metrics label this as fixed-boundary baseline data; it is not a
  free-boundary LCFS result.

### Best next steps

1. Commit and push M12d.
2. Promote the baseline driver from example-only to a small reusable function
   only if another script needs it.
3. Define and test the actual free-boundary LCFS update target:
   - boundary normal-field/pressure-balance diagnostic;
   - cap boundary conditions;
   - convergence metric and plots for the beta scan.

### Completion percentages after M12d

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `84%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `85%`.
- I/O schema and docs: `88%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `30%`.
- Stellarator-mirror hybrid lane: `10%`.
- ESSOS circular-coil mirror beta scan: `20%`.
- PR merge readiness overall: `85%`.

### User input needed

No user input is needed.

## 67. 2026-06-18 M12e LCFS target diagnostic for circular-coil mirror baselines

This tranche turned the fixed-boundary beta-scan baseline into a measurable
pre-LCFS target.  It still does not update the LCFS, but each beta row now
reports side-boundary normal-field error and total-pressure imbalance against
the external circular-coil field.

### Steps taken

- Added `MirrorLCFSDiagnostic` and `mirror_lcfs_diagnostic`.
- Exported the diagnostic through the public `vmec_jax.mirror` API.
- Extended the root `examples/mirror_free_boundary_circular_coils.py` example
  so each optional fixed-boundary beta baseline:
  - reloads the written `mout`;
  - samples the external circular-coil field on the same axisymmetric boundary
    grid;
  - computes external `B . n` and total-pressure imbalance on the side
    boundary;
  - writes scalar diagnostic metrics to the JSON row;
  - writes a `*_lcfs_diagnostic.png` panel when plots are enabled.
- Added a direct unit test for the LCFS diagnostic on a cylindrical side
  boundary with axial external field.
- Extended the root example smoke test so the baseline rows must include the
  LCFS diagnostic metrics.
- Updated mirror docs and the mirror examples README.

### Results obtained

Generated artifacts:

- `results/mirror/m12e_lcfs_diagnostic/free_boundary_circular_coils_metrics.json`.
- `results/mirror/m12e_lcfs_diagnostic/free_boundary_circular_coils_setup.json`.
- Three beta-row `mout` files under `results/mirror/m12e_lcfs_diagnostic/`.
- Thirty-six PNGs, including one `*_lcfs_diagnostic.png` panel per beta case.

Representative beta-baseline rows with `baseline_maxiter=0`:

| beta percent | residual | `fsq` | `B_ext.n` RMS | pressure-balance RMS |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `1.884728113343e-01` | `9.372559528304e-05` | `7.657346104349e-03` | `1.803515365850` |
| 3 | `4.011165408143e-01` | `4.245236921236e-04` | `7.657346104349e-03` | `1.803515365850` |
| 10 | `1.237976350530e+00` | `4.043761067210e-03` | `7.657346104349e-03` | `1.803515365850` |

The LCFS imbalance is intentionally nonzero for these rows because the
boundary is still the analytic fixed flux-tube initializer.  These scalars are
the target for the next boundary-update lane.

Visual validation:

- Inspected the beta-10 `*_lcfs_diagnostic.png` panel; it is nonblank and
  shows signed side-boundary `B_ext.n` plus total-pressure imbalance versus
  horizontal `z`.
- Inspected the beta-10 3-D boundary plot; it keeps the mirror horizontal with
  `z` as the long axis.
- A pixel-stat smoke check reported all first inspected PNGs nonblank.

### How it was tested

Focused free-boundary and root example tests:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_example_runs_without_plots \
  -q
```

Result: `8 passed in 5.12s`.

Example with plots and baseline outputs:

```bash
JAX_ENABLE_X64=1 python examples/mirror_free_boundary_circular_coils.py \
  --outdir results/mirror/m12e_lcfs_diagnostic \
  --ntheta 24 \
  --nxi 33 \
  --n-segments 256 \
  --run-fixed-boundary-baseline \
  --baseline-maxiter 0
```

Result: metrics JSON, setup JSON, three `mout` files, and 36 PNGs written.

Lint/format/docs/whitespace:

```bash
python -m ruff check \
  vmec_jax/mirror/free_boundary.py \
  vmec_jax/mirror/api.py \
  vmec_jax/mirror/__init__.py \
  examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py
python -m ruff format --check \
  vmec_jax/mirror/free_boundary.py \
  vmec_jax/mirror/api.py \
  vmec_jax/mirror/__init__.py \
  examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py
python -m sphinx -W -j auto -b html docs docs/_build/html
git diff --check
```

Result: all checks passed.

### File structure and best-practice notes

- The diagnostic lives in `vmec_jax/mirror/free_boundary.py` with the other
  circular-coil bridge helpers because it compares a mirror output to external
  coil samples.
- It is intentionally NumPy-side: this is a fast CLI/example diagnostic, not
  yet the differentiable LCFS update kernel.
- The root example owns the diagnostic plot because the plot is workflow
  specific and uses the beta-scan row labels.
- The public API exports the dataclass and function so later free-boundary
  drivers can reuse the same measured target.

### Best next steps

1. Commit and push M12e.
2. Add the first conservative axisymmetric LCFS update proposal:
   - use the sign of the pressure-balance residual to propose a radius update;
   - preserve positive radius and cap constraints;
   - keep a small damping/line-search parameter;
   - report before/after `B_ext.n` and pressure-balance diagnostics.
3. Add a small standalone test that one damped update reduces the pressure
   imbalance for a synthetic axisymmetric target.
4. Only after that, connect the update loop to beta-scan rows and decide
   whether the LCFS updater remains CLI-fast NumPy or gets a differentiable
   JAX variant.

### Completion percentages after M12e

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `84%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `85%`.
- I/O schema and docs: `89%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `35%`.
- Stellarator-mirror hybrid lane: `10%`.
- ESSOS circular-coil mirror beta scan: `25%`.
- PR merge readiness overall: `86%`.

### User input needed

No user input is needed.

---

## 68. 2026-06-18 M12f conservative axisymmetric LCFS update proposal

This tranche added the first tested LCFS radius proposal.  It is not a full
free-boundary solve: the fixed-boundary equilibrium is kept frozen, the
external magnetic-pressure response is estimated by radial finite differences
of the circular-coil field, and a damped/clipped axisymmetric radius update is
proposed from the local pressure-balance residual.

### Steps taken

- Added `MirrorLCFSUpdateProposal`.
- Added `mirror_external_pressure_balance_response` to estimate
  `d(pressure_balance)/dr` from external-coil magnetic pressure.
- Added `propose_axisymmetric_mirror_lcfs_update`:
  - theta-averages the diagnostic residual and response;
  - applies a damped Newton-like radius step;
  - clips the update by maximum relative radius movement;
  - preserves cap radii by default;
  - returns a tabulated `MirrorBoundary` proposal.
- Exported the new dataclass and helpers through the public mirror API.
- Extended `examples/mirror_free_boundary_circular_coils.py` so each beta row
  records:
  - pressure-response min/max;
  - predicted post-update pressure-balance RMS;
  - predicted reduction fraction;
  - max absolute and relative radius movement.
- Updated the LCFS diagnostic plot to overlay the predicted pressure-balance
  curve from the damped update.
- Added a synthetic unit test showing that a known pressure response produces
  a reduced pressure-balance residual while preserving cap radii.
- Updated docs and the mirror examples README.

### Results obtained

Generated artifacts:

- `results/mirror/m12f_lcfs_update_proposal/free_boundary_circular_coils_metrics.json`.
- `results/mirror/m12f_lcfs_update_proposal/free_boundary_circular_coils_setup.json`.
- Three beta-row `mout` files under `results/mirror/m12f_lcfs_update_proposal/`.
- Thirty-six PNGs, including one LCFS diagnostic panel per beta case with the
  predicted-update overlay.

Representative beta-baseline rows with `baseline_maxiter=0` and a 5% radius
move cap:

| beta percent | pressure-balance RMS | predicted RMS | reduction fraction | max relative radius step |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `1.803515365850` | `1.797531697147` | `3.317780827395e-03` | `5.000000000000e-02` |
| 3 | `1.803515365850` | `1.797531697147` | `3.317780827395e-03` | `5.000000000000e-02` |
| 10 | `1.803515365850` | `1.797531697147` | `3.317780827395e-03` | `5.000000000000e-02` |

The predicted improvement is small because this is a conservative one-step
linearized proposal with cap radii held fixed.  It is a useful monotonicity
check, not a convergence claim.

Visual validation:

- Inspected the beta-10 LCFS panel and confirmed that the predicted-update
  curve overlays the before curve without plotting errors.
- Pixel-stat checks reported all LCFS diagnostic PNGs nonblank.

### How it was tested

Focused free-boundary and root example tests:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_example_runs_without_plots \
  -q
```

Result: `9 passed in 5.04s`.

Example with plots and update proposal outputs:

```bash
JAX_ENABLE_X64=1 python examples/mirror_free_boundary_circular_coils.py \
  --outdir results/mirror/m12f_lcfs_update_proposal \
  --ntheta 24 \
  --nxi 33 \
  --n-segments 256 \
  --run-fixed-boundary-baseline \
  --baseline-maxiter 0
```

Result: metrics JSON, setup JSON, three `mout` files, and 36 PNGs written.

Lint/format/docs/whitespace:

```bash
python -m ruff check \
  vmec_jax/mirror/free_boundary.py \
  vmec_jax/mirror/api.py \
  vmec_jax/mirror/__init__.py \
  examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py
python -m ruff format --check \
  vmec_jax/mirror/free_boundary.py \
  vmec_jax/mirror/api.py \
  vmec_jax/mirror/__init__.py \
  examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py
python -m sphinx -W -j auto -b html docs docs/_build/html
git diff --check
```

Result: all checks passed.

### File structure and best-practice notes

- The response and proposal helpers stay in `vmec_jax/mirror/free_boundary.py`
  next to the circular-coil bridge; they need both the mirror side-boundary
  diagnostic and the external-field provider.
- The helper returns a normal `MirrorBoundary`, so later fixed-boundary
  warm-start or LCFS loops can reuse the existing solver entrypoint.
- The update is explicit about its linearized assumptions and keeps the cap
  constraint simple until the cap-boundary-condition lane is implemented.
- The tests avoid an expensive solve for the monotonicity invariant and use
  the root example smoke test for end-to-end circular-coil wiring.

### Best next steps

1. Commit and push M12f.
2. Add a one- or two-step beta-row LCFS pilot loop:
   - apply the proposal boundary;
   - rerun the fixed-boundary solve from that new boundary at low resolution;
   - resample the external field;
   - report actual, not only predicted, diagnostic changes.
3. Add cap-condition diagnostics:
   - fixed equal cap radii;
   - optional equal cap `B_ext.n`/pressure-balance reporting;
   - explicit warning if caps dominate the imbalance.
4. Only after the pilot shows actual diagnostic reduction, promote the loop
   into an example option and decide on JAX differentiable equivalents.

### Completion percentages after M12f

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `84%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `85%`.
- I/O schema and docs: `89%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `40%`.
- Stellarator-mirror hybrid lane: `10%`.
- ESSOS circular-coil mirror beta scan: `30%`.
- PR merge readiness overall: `87%`.

### User input needed

No user input is needed.

---

## 69. 2026-06-18 M12g low-resolution LCFS pilot with actual diagnostics

This tranche added an optional pilot loop to the circular-coil example.  The
pilot applies the proposed axisymmetric radius boundary, reruns the
fixed-boundary solve at low resolution, recomputes the LCFS diagnostics, and
records whether the actual pressure-balance RMS improved.  Non-improving
trials are rejected and the pilot does not advance them to the next step.

### Steps taken

- Added `--run-lcfs-pilot` and `--lcfs-pilot-steps` to
  `examples/mirror_free_boundary_circular_coils.py`.
- Added `--lcfs-update-damping` and
  `--lcfs-update-max-relative-step` controls.
- For each beta row, the optional pilot now:
  - applies the proposal boundary from M12f;
  - reruns the existing fixed-boundary solver on that proposed boundary;
  - writes a pilot-step `mout`;
  - resamples the external field on the proposed boundary;
  - recomputes side-boundary `B_ext.n` and total-pressure imbalance;
  - writes a standard plot bundle and pilot LCFS diagnostic plot;
  - records an `accepted` flag and stops advancing if actual pressure-balance
    RMS does not improve.
- Extended the root example smoke test to exercise one pilot step at
  `maxiter=0` and verify that pilot `mout` and acceptance metadata are written.
- Updated docs and the mirror examples README.

### Results obtained

Generated artifacts:

- `results/mirror/m12g_lcfs_pilot_actual/free_boundary_circular_coils_metrics.json`.
- `results/mirror/m12g_lcfs_pilot_actual/free_boundary_circular_coils_setup.json`.
- Three baseline beta-row `mout` files and three pilot-step `mout` files.
- Sixty-nine PNGs, including baseline and pilot LCFS diagnostic panels.

Representative beta rows with `baseline_maxiter=0`, one pilot step, and a 5%
relative radius cap:

| beta percent | baseline pressure RMS | predicted RMS | actual pilot RMS | actual change fraction | accepted |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | `1.803515365850` | `1.797531697147` | `3.109631442303` | `-7.242056825156e-01` | `false` |
| 3 | `1.803515365850` | `1.797531697147` | `3.109631442303` | `-7.242056825156e-01` | `false` |
| 10 | `1.803515365850` | `1.797531697147` | `3.109631442303` | `-7.242056825156e-01` | `false` |

Normal-field diagnostic:

| beta percent | baseline `B_ext.n` RMS | pilot `B_ext.n` RMS |
| ---: | ---: | ---: |
| 1 | `7.657346104349e-03` | `4.164859674186e-01` |
| 3 | `7.657346104349e-03` | `4.164859674186e-01` |
| 10 | `7.657346104349e-03` | `4.164859674186e-01` |

The pilot result is intentionally logged as a negative result: the pressure-only
linearized proposal predicts a small improvement, but the actual fixed-boundary
rerun worsens the pressure-balance diagnostic and creates large `B_ext.n`
near the caps.  The next update must be slope/cap aware, not just local
external magnetic-pressure aware.

Visual validation:

- Inspected the beta-10 pilot-step LCFS panel.  The plot shows strong
  normal-field spikes near the caps after the proposed boundary is applied.
- Pixel-stat checks reported all pilot LCFS diagnostic PNGs nonblank.

### How it was tested

Focused free-boundary and root example tests:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_example_runs_without_plots \
  -q
```

Result: `9 passed in 5.07s`.

Example with plots and pilot outputs:

```bash
JAX_ENABLE_X64=1 python examples/mirror_free_boundary_circular_coils.py \
  --outdir results/mirror/m12g_lcfs_pilot_actual \
  --ntheta 24 \
  --nxi 33 \
  --n-segments 256 \
  --run-fixed-boundary-baseline \
  --run-lcfs-pilot \
  --lcfs-pilot-steps 1 \
  --baseline-maxiter 0
```

Result: metrics JSON, setup JSON, six `mout` files, and 69 PNGs written.

Lint/format/docs/whitespace:

```bash
python -m ruff check \
  examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_examples.py
python -m ruff format --check \
  examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_examples.py
python -m sphinx -W -j auto -b html docs docs/_build/html
git diff --check
```

Result: all checks passed.

### File structure and best-practice notes

- The pilot remains in the root example because it is a workflow diagnostic,
  not yet a reusable free-boundary solver.
- It reuses the existing fixed-boundary solver, `mout` writer, plot bundle,
  and LCFS diagnostic helper rather than creating a parallel output path.
- The rejection flag prevents a known-worse radius candidate from being
  silently advanced in multi-step pilot runs.
- The negative result is retained in the metrics and plan because it identifies
  the missing slope/cap term needed for a robust LCFS updater.

### Best next steps

1. Commit and push M12g.
2. Add slope/cap-aware update controls:
   - smooth or solve for radius updates in a Chebyshev basis;
   - constrain `dr/dz` near caps;
   - penalize predicted `B_ext.n`, not only pressure balance;
   - preserve equal cap conditions explicitly.
3. Add a synthetic test where a slope-aware update reduces both pressure
   balance and normal-field residual.
4. Re-run the M12g pilot with the slope-aware proposal and require actual
   acceptance before increasing pilot steps or beta resolution.

### Completion percentages after M12g

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `84%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `85%`.
- I/O schema and docs: `89%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `42%`.
- Stellarator-mirror hybrid lane: `10%`.
- ESSOS circular-coil mirror beta scan: `32%`.
- PR merge readiness overall: `87%`.

### User input needed

No user input is needed.

---

## 70. 2026-06-18 M12h slope/cap-aware LCFS update proposal

This tranche fixed the M12g failure mode by making the LCFS radius proposal
cap-aware.  The proposal now tapers smoothly to zero at the mirror caps and
can apply a small axial smoothing pass before the candidate boundary is
constructed.  The default pilot path now accepts the first low-resolution
candidate on the circular-coil baseline.

### Steps taken

- Extended `MirrorLCFSUpdateProposal` with:
  - `cap_taper_power`;
  - `smoothing_passes`.
- Added `cap_taper_power` and `smoothing_passes` controls to
  `propose_axisymmetric_mirror_lcfs_update`.
- Defaulted the proposal to a smooth `sin(pi z_norm)^2` taper and one axial
  smoothing pass.
- Exposed the controls in the root circular-coil example:
  - `--lcfs-update-cap-taper-power`;
  - `--lcfs-update-smoothing-passes`.
- Added update metadata to the baseline and pilot JSON rows.
- Added a synthetic unit test confirming that the cap-tapered update reduces
  near-cap motion relative to the untapered update while preserving cap radii.
- Tightened the root example smoke test so the default pilot must be accepted
  and reduce actual pressure-balance RMS.
- Updated docs and the mirror examples README.

### Results obtained

Generated artifacts:

- `results/mirror/m12h_cap_tapered_lcfs_pilot/free_boundary_circular_coils_metrics.json`.
- `results/mirror/m12h_cap_tapered_lcfs_pilot/free_boundary_circular_coils_setup.json`.
- Three baseline beta-row `mout` files and three pilot-step `mout` files.
- Standard plot bundles and pilot LCFS diagnostic panels.

Representative beta rows with `baseline_maxiter=0`, one pilot step, 5% radius
cap, taper power 2, and one smoothing pass:

| beta percent | baseline pressure RMS | predicted RMS | actual pilot RMS | actual change fraction | accepted |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | `1.803515365850` | `1.803438138182` | `1.802062059437` | `8.058187031492e-04` | `true` |
| 3 | `1.803515365850` | `1.803438138182` | `1.802062059437` | `8.058187031492e-04` | `true` |
| 10 | `1.803515365850` | `1.803438138182` | `1.802062059437` | `8.058187031492e-04` | `true` |

Normal-field diagnostic:

| beta percent | baseline `B_ext.n` RMS | pilot `B_ext.n` RMS |
| ---: | ---: | ---: |
| 1 | `7.657346104349e-03` | `1.111535841912e-02` |
| 3 | `7.657346104349e-03` | `1.111535841912e-02` |
| 10 | `7.657346104349e-03` | `1.111535841912e-02` |

The cap-aware proposal removes the large M12g cap spikes and produces an
accepted actual pilot step.  The pressure-balance improvement is small, and
`B_ext.n` still rises modestly, so the next step should add a combined
pressure/normal-field merit function instead of accepting only pressure RMS.

Visual validation:

- Inspected the beta-10 pilot-step LCFS panel.  The previous cap spikes are
  gone, and the pressure-balance curve is smooth through the caps.

### How it was tested

Focused free-boundary and root example tests:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_example_runs_without_plots \
  -q
```

Result: `10 passed in 5.77s`.

Example with plots and accepted pilot outputs:

```bash
JAX_ENABLE_X64=1 python examples/mirror_free_boundary_circular_coils.py \
  --outdir results/mirror/m12h_cap_tapered_lcfs_pilot \
  --ntheta 24 \
  --nxi 33 \
  --n-segments 256 \
  --run-fixed-boundary-baseline \
  --run-lcfs-pilot \
  --lcfs-pilot-steps 1 \
  --baseline-maxiter 0
```

Result: metrics JSON, setup JSON, six `mout` files, and plot bundles written.

Lint/format/docs/whitespace:

```bash
python -m ruff check \
  vmec_jax/mirror/free_boundary.py \
  examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py
python -m ruff format --check \
  vmec_jax/mirror/free_boundary.py \
  examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py
python -m sphinx -W -j auto -b html docs docs/_build/html
git diff --check
```

Result: all checks passed.

### File structure and best-practice notes

- The cap-tapering stays inside the proposal helper because it is part of the
  candidate boundary construction, not plotting or example-only bookkeeping.
- The exact untapered update remains available for controlled tests and future
  analytic comparisons.
- The pilot acceptance test is now tied to the example default, so future
  changes cannot silently regress the first accepted LCFS trial.

### Best next steps

1. Commit and push M12h.
2. Add a combined LCFS merit metric:
   - pressure-balance RMS;
   - normalized `B_ext.n` RMS;
   - optional cap-weighted terms.
3. Make pilot acceptance use the combined merit instead of pressure alone.
4. Run two pilot steps with the combined merit and inspect whether both
   pressure and normal-field diagnostics trend down.

### Completion percentages after M12h

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `84%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `85%`.
- I/O schema and docs: `89%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `46%`.
- Stellarator-mirror hybrid lane: `10%`.
- ESSOS circular-coil mirror beta scan: `35%`.
- PR merge readiness overall: `88%`.

### User input needed

No user input is needed.

---

## 71. 2026-06-18 M12i combined LCFS merit and two-step pilot

This tranche added a dimensionless LCFS merit for pilot-step acceptance.  The
merit combines pressure-balance RMS and normalized side-boundary `B_ext.n` RMS,
using the baseline pressure RMS and baseline external-field RMS as scales.  The
pilot now accepts candidates by combined merit rather than pressure RMS alone.

### Steps taken

- Added `MirrorLCFSMerit` and `mirror_lcfs_merit`.
- Exported the merit helper through the public mirror API.
- Added `--lcfs-merit-bnormal-weight` to the root circular-coil example.
- Added baseline and pilot JSON fields for:
  - combined merit;
  - pressure scale;
  - normal-field scale;
  - normal-field weight;
  - merit change fraction.
- Switched pilot acceptance from pressure-only to combined merit.
- Added a direct unit test for merit normalization.
- Tightened the root example smoke test so accepted pilots must reduce the
  combined merit.
- Updated docs and the mirror examples README.

### Results obtained

Generated artifacts:

- `results/mirror/m12i_combined_merit_pilot/free_boundary_circular_coils_metrics.json`.
- `results/mirror/m12i_combined_merit_pilot/free_boundary_circular_coils_setup.json`.
- Three baseline beta-row `mout` files and six pilot-step `mout` files.
- Nine LCFS diagnostic panels: three baseline, three step-1, three step-2.

Representative two-step rows with `baseline_maxiter=0`, taper power 2, one
smoothing pass, and normal-field weight 1:

| beta percent | row | merit | pressure RMS | `B_ext.n` RMS | accepted |
| ---: | :--- | ---: | ---: | ---: | :---: |
| 1 | baseline | `1.000020371650` | `1.803515365850` | `7.657346104349e-03` | n/a |
| 1 | step 1 | `0.999237141051` | `1.802062059437` | `1.111535841912e-02` | `true` |
| 1 | step 2 | `0.998686017484` | `1.801007306558` | `1.484484737831e-02` | `true` |

The 3% and 10% rows have the same geometry diagnostics for this `maxiter=0`
fixed-boundary pilot because the boundary and external field are shared.

Interpretation:

- The combined merit and pressure-balance RMS decrease over two accepted pilot
  steps.
- `B_ext.n` still increases smoothly, so this is not yet a balanced LCFS
  update.  The next lane should add a normal-field descent term or candidate
  line search that can reduce both components.

### How it was tested

Focused free-boundary and root example tests:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_example_runs_without_plots \
  -q
```

Result: `11 passed in 4.91s`.

Example with plots and two accepted pilot steps:

```bash
JAX_ENABLE_X64=1 python examples/mirror_free_boundary_circular_coils.py \
  --outdir results/mirror/m12i_combined_merit_pilot \
  --ntheta 24 \
  --nxi 33 \
  --n-segments 256 \
  --run-fixed-boundary-baseline \
  --run-lcfs-pilot \
  --lcfs-pilot-steps 2 \
  --baseline-maxiter 0
```

Result: metrics JSON, setup JSON, nine `mout` files, and plot bundles written.

Lint/format/docs/whitespace:

```bash
python -m ruff check \
  vmec_jax/mirror/free_boundary.py \
  vmec_jax/mirror/api.py \
  vmec_jax/mirror/__init__.py \
  examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py
python -m ruff format --check \
  vmec_jax/mirror/free_boundary.py \
  vmec_jax/mirror/api.py \
  vmec_jax/mirror/__init__.py \
  examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py
python -m sphinx -W -j auto -b html docs docs/_build/html
git diff --check
```

Result: all checks passed.

### File structure and best-practice notes

- The merit helper lives beside LCFS diagnostics in
  `vmec_jax/mirror/free_boundary.py` because it is a physics acceptance metric,
  not example-only presentation.
- The example owns the pilot loop and row bookkeeping because it is still a
  workflow diagnostic, not a reusable free-boundary solver.
- Normalization by external `|B|` keeps the normal-field term dimensionless and
  prevents small smooth changes from dominating pressure-balance progress.

### Best next steps

1. Commit and push M12i.
2. Add a normal-field-aware proposal component:
   - estimate how `B_ext.n` changes under a smooth radius perturbation;
   - include that response in the candidate direction or line search;
   - require both pressure RMS and `B_ext.n` RMS not to increase for the
     default pilot.
3. Promote the pilot loop into a small reusable helper only if another example
   or test needs it.
4. Start the stellarator-mirror hybrid boundary lane after the LCFS pilot can
   reduce both components on the circular-coil baseline.

### Completion percentages after M12i

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `84%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `85%`.
- I/O schema and docs: `89%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `49%`.
- Stellarator-mirror hybrid lane: `10%`.
- ESSOS circular-coil mirror beta scan: `38%`.
- PR merge readiness overall: `88%`.

### User input needed

No user input is needed.

---

## 72. 2026-06-18 M12j normal-field-aware LCFS proposal direction

This tranche added a normal-field-aware candidate selection path for the
circular-coil LCFS pilot.  The pilot now compares a local pressure update with
a shape-preserving scale update before running the fixed-boundary trial, using
predicted combined merit and predicted side-boundary `B_ext.n` to select the
candidate.

### Steps taken

- Added `mirror_external_bnormal` as a shared helper for side-boundary normal
  field evaluation.
- Added `propose_axisymmetric_mirror_lcfs_scale_update`, a shape-preserving
  global radius-scale candidate.
- Extended `MirrorLCFSUpdateProposal` with a `strategy` field.
- Exported the new helpers through the public mirror API.
- Added `--lcfs-proposal-mode` to the root circular-coil example:
  - `best_predicted` compares candidates and selects the lower predicted
    combined merit;
  - `local` forces the cap-tapered local pressure update;
  - `scale` forces the shape-preserving scale update.
- Added candidate summaries to baseline and pilot JSON rows:
  - strategy;
  - predicted merit;
  - predicted pressure-balance RMS;
  - predicted `B_ext.n` RMS;
  - max relative radius step.
- Added tests for:
  - zero external normal field on a cylindrical boundary with axial field;
  - synthetic shape-preserving scale update;
  - root example selection of the scale candidate when it has lower predicted
    normal-field cost than the local candidate.
- Updated docs and the mirror examples README.

### Results obtained

Generated artifacts:

- `results/mirror/m12j_normal_field_aware_pilot/free_boundary_circular_coils_metrics.json`.
- `results/mirror/m12j_normal_field_aware_pilot/free_boundary_circular_coils_setup.json`.
- Three baseline beta-row `mout` files and six pilot-step `mout` files.
- Nine LCFS diagnostic panels.

Candidate selection for the baseline rows:

| candidate | predicted merit | predicted pressure RMS | predicted `B_ext.n` RMS |
| :--- | ---: | ---: | ---: |
| local pressure | `1.000000106` | `1.803438138` | `1.1115358e-02` |
| scale pressure | `0.996004406` | `1.796260719` | `8.783073e-03` |

The default `best_predicted` mode selected `scale_pressure`.

Representative two-step actual pilot rows:

| beta percent | row | merit | pressure RMS | `B_ext.n` RMS | accepted |
| ---: | :--- | ---: | ---: | ---: | :---: |
| 1 | baseline | `1.000020371650` | `1.803515365850` | `7.657346104349e-03` | n/a |
| 1 | step 1 | `0.718910218442` | `1.296498386277` | `8.783072862553e-03` | `true` |
| 1 | step 2 | `0.486525139721` | `0.877324628786` | `1.008254026282e-02` | `true` |

The 3% and 10% rows have the same geometry diagnostics for this `maxiter=0`
fixed-boundary pilot because the boundary and external field are shared.

Interpretation:

- The normal-field-aware selector chooses a smoother scale update that reduces
  pressure and combined merit much more strongly than the previous local-only
  update.
- `B_ext.n` still increases modestly, but it is much smaller than the M12g cap
  spike and smaller than the local candidate prediction.  The next step should
  add a true normal-field descent/guard if we require both terms to decrease.

### How it was tested

Focused free-boundary and root example tests:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_example_runs_without_plots \
  -q
```

Result: `13 passed in 5.03s`.

Example with plots and two accepted pilot steps:

```bash
JAX_ENABLE_X64=1 python examples/mirror_free_boundary_circular_coils.py \
  --outdir results/mirror/m12j_normal_field_aware_pilot \
  --ntheta 24 \
  --nxi 33 \
  --n-segments 256 \
  --run-fixed-boundary-baseline \
  --run-lcfs-pilot \
  --lcfs-pilot-steps 2 \
  --baseline-maxiter 0
```

Result: metrics JSON, setup JSON, nine `mout` files, and plot bundles written.

Lint/format/docs/whitespace:

```bash
python -m ruff check \
  vmec_jax/mirror/free_boundary.py \
  vmec_jax/mirror/api.py \
  vmec_jax/mirror/__init__.py \
  examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py
python -m ruff format --check \
  vmec_jax/mirror/free_boundary.py \
  vmec_jax/mirror/api.py \
  vmec_jax/mirror/__init__.py \
  examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py
python -m sphinx -W -j auto -b html docs docs/_build/html
git diff --check
```

Result: all checks passed.

### File structure and best-practice notes

- `mirror_external_bnormal` lives in `vmec_jax/mirror/free_boundary.py` so the
  same geometry/external-field normal calculation is reused by diagnostics,
  candidate prediction, and future drivers.
- The shape-preserving scale proposal is a reusable API helper because it is a
  physically distinct candidate direction, not merely example bookkeeping.
- Candidate selection remains in the root example while it is still a pilot
  workflow rather than a production LCFS solver.

### Best next steps

1. Commit and push M12j.
2. Add a stricter pilot mode that requires `B_ext.n` RMS not to increase:
   - use a small candidate line search over scale factor;
   - include a no-op candidate so rejection is explicit;
   - report whether progress is pressure-limited or normal-field-limited.
3. After the candidate guard is stable, promote the pilot row logic into a
   reusable helper only if it is needed by another example or test.
4. Begin the stellarator-mirror hybrid boundary lane once the circular-coil
   LCFS pilot has a reliable pressure/normal-field acceptance gate.

### Completion percentages after M12j

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `85%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `85%`.
- I/O schema and docs: `90%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `52%`.
- Stellarator-mirror hybrid lane: `10%`.
- ESSOS circular-coil mirror beta scan: `40%`.
- PR merge readiness overall: `89%`.

### User input needed

No user input is needed.

---

## 73. 2026-06-18 M12k strict normal-field guard for LCFS pilots

This tranche added an optional strict normal-field guard for LCFS pilot
candidates.  In guarded mode, the example includes an explicit no-op candidate
and refuses to run a pilot solve when every nonzero candidate would increase
predicted side-boundary `B_ext.n` RMS.

### Steps taken

- Added `propose_axisymmetric_mirror_lcfs_noop_update`.
- Exported the no-op helper through the public mirror API.
- Added `--lcfs-require-bnormal-nonincrease` to the root circular-coil example.
- Extended candidate selection so guarded mode filters out nonzero candidates
  whose predicted `B_ext.n` RMS is above the current accepted value.
- Added skipped-pilot row reporting with:
  - `skipped=true`;
  - `accepted=false`;
  - `rejection_reason="normal_field_guard_no_candidate"`;
  - no pilot `mout` path.
- Added tests for:
  - no-op proposal invariants;
  - strict-guard example behavior.
- Updated docs and the mirror examples README.

### Results obtained

Generated artifacts:

- `results/mirror/m12k_strict_bnormal_guard/free_boundary_circular_coils_metrics.json`.
- `results/mirror/m12k_strict_bnormal_guard/free_boundary_circular_coils_setup.json`.
- Three baseline beta-row `mout` files.
- Thirty-six baseline PNGs, including LCFS diagnostic panels.

Strict-guard candidate summary for each beta row:

| candidate | predicted merit | predicted `B_ext.n` RMS |
| :--- | ---: | ---: |
| local pressure | `1.000000106` | `1.1115358e-02` |
| scale pressure | `0.996004406` | `8.783073e-03` |
| no-op | `1.000020372` | `7.657346e-03` |

Because both nonzero candidates increase predicted normal-field RMS, guarded
mode selects `noop` and skips the pilot solve.  This explicitly distinguishes
pressure-limited progress from normal-field-limited progress.

### How it was tested

Focused free-boundary and root example tests:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_example_runs_without_plots \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_strict_bnormal_guard_can_skip_pilot \
  -q
```

Result: `15 passed in 8.45s`.

Example with plots and strict guard:

```bash
JAX_ENABLE_X64=1 python examples/mirror_free_boundary_circular_coils.py \
  --outdir results/mirror/m12k_strict_bnormal_guard \
  --ntheta 24 \
  --nxi 33 \
  --n-segments 256 \
  --run-fixed-boundary-baseline \
  --run-lcfs-pilot \
  --lcfs-pilot-steps 1 \
  --baseline-maxiter 0 \
  --lcfs-require-bnormal-nonincrease
```

Result: metrics JSON, setup JSON, three baseline `mout` files, and 36 PNGs
written.

Lint/format/docs/whitespace:

```bash
python -m ruff check \
  vmec_jax/mirror/free_boundary.py \
  vmec_jax/mirror/api.py \
  vmec_jax/mirror/__init__.py \
  examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py
python -m ruff format --check \
  vmec_jax/mirror/free_boundary.py \
  vmec_jax/mirror/api.py \
  vmec_jax/mirror/__init__.py \
  examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py
python -m sphinx -W -j auto -b html docs docs/_build/html
git diff --check
```

Result: all checks passed.

### File structure and best-practice notes

- The no-op proposal is a first-class helper because rejection and no-op
  behavior are part of the LCFS candidate model, not just test scaffolding.
- Guarded no-op rows are explicit in the JSON, so downstream scripts can
  distinguish skipped candidates from failed solves.
- The default example still uses best-predicted merit without the strict guard;
  guarded mode is opt-in because the current pressure and normal-field
  objectives are in tension for the circular-coil baseline.

### Best next steps

1. Commit and push M12k.
2. Add a normal-field descent term rather than only a guard:
   - estimate `B_ext.n` response to smooth radius perturbations;
   - search over a small basis of scale/taper/local modes;
   - require accepted candidates to reduce pressure merit and not increase
     normal-field RMS.
3. Once pressure and normal-field can both improve, move the pilot loop into a
   reusable helper and begin the stellarator-mirror hybrid boundary lane.

### Completion percentages after M12k

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `85%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `85%`.
- I/O schema and docs: `90%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `54%`.
- Stellarator-mirror hybrid lane: `10%`.
- ESSOS circular-coil mirror beta scan: `42%`.
- PR merge readiness overall: `89%`.

### User input needed

No user input is needed.

---

## 74. 2026-06-18 M12l normal-field-slope LCFS proposal basis

This tranche added a real normal-field descent candidate for circular-coil LCFS
pilots.  The new candidate estimates the boundary slope needed for
`B_ext.n ~= 0`, integrates that slope from the current midplane radius, and
takes a clipped step toward the resulting smooth field-line-following shape.

### Steps taken

- Added `propose_axisymmetric_mirror_lcfs_bnormal_update`.
- Exported the helper through the public mirror API.
- Added `bnormal_slope` to the root example candidate set.
- Added explicit `--lcfs-proposal-mode bnormal` for diagnostics.
- Updated default `best_predicted` mode so it compares:
  - local pressure update;
  - shape-preserving scale update;
  - normal-field-slope update;
  - no-op.
- Added tests for:
  - synthetic normal-field descent;
  - candidate summary expectations in the root example.
- Updated docs and the mirror examples README.

### Results obtained

Generated artifacts:

- `results/mirror/m12l_bnormal_descent_pilot/free_boundary_circular_coils_metrics.json`.
- `results/mirror/m12l_bnormal_descent_strict_guard/free_boundary_circular_coils_metrics.json`.
- `results/mirror/m12l_forced_bnormal_pilot/free_boundary_circular_coils_metrics.json`.

Baseline candidate summary:

| candidate | predicted merit | predicted pressure RMS | predicted `B_ext.n` RMS |
| :--- | ---: | ---: | ---: |
| local pressure | `1.000000106` | `1.803438138` | `1.1115358e-02` |
| scale pressure | `0.996004406` | `1.796260719` | `8.783073e-03` |
| normal-field slope | `1.004048383` | `1.810804122` | `4.486880e-03` |
| no-op | `1.000020372` | `1.803515366` | `7.657346e-03` |

Default `best_predicted` still selects `scale_pressure` because it gives the
best combined merit.  Strict normal-field-guard mode selects `noop`, because
the normal-field-slope candidate reduces `B_ext.n` but worsens combined merit.

Forced normal-field-slope pilot:

| row | merit | pressure RMS | `B_ext.n` RMS | accepted |
| :--- | ---: | ---: | ---: | :---: |
| baseline | `1.000020371650` | `1.803515365850` | `7.657346104349e-03` | n/a |
| forced bnormal step | `1.358618842647` | `2.450280673959` | `4.486879987461e-03` | `false` |

Interpretation:

- The new candidate is a genuine normal-field descent direction.
- The current pressure and normal-field objectives remain in tension on the
  circular-coil fixed-boundary baseline.
- The next useful step is a small multi-mode line search or least-squares
  candidate that mixes scale and normal-field-slope modes, rather than choosing
  one pure mode.

### How it was tested

Focused free-boundary and root example tests:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_example_runs_without_plots \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_strict_bnormal_guard_can_skip_pilot \
  -q
```

Result: `16 passed in 7.97s`.

Examples with plots:

```bash
JAX_ENABLE_X64=1 python examples/mirror_free_boundary_circular_coils.py \
  --outdir results/mirror/m12l_bnormal_descent_pilot \
  --ntheta 24 \
  --nxi 33 \
  --n-segments 256 \
  --run-fixed-boundary-baseline \
  --run-lcfs-pilot \
  --lcfs-pilot-steps 2 \
  --baseline-maxiter 0

JAX_ENABLE_X64=1 python examples/mirror_free_boundary_circular_coils.py \
  --outdir results/mirror/m12l_forced_bnormal_pilot \
  --ntheta 24 \
  --nxi 33 \
  --n-segments 256 \
  --run-fixed-boundary-baseline \
  --run-lcfs-pilot \
  --lcfs-pilot-steps 1 \
  --baseline-maxiter 0 \
  --lcfs-proposal-mode bnormal
```

Result: metrics JSON files, `mout` outputs, and plot bundles written.

Lint/format/docs/whitespace:

```bash
python -m ruff check \
  vmec_jax/mirror/free_boundary.py \
  vmec_jax/mirror/api.py \
  vmec_jax/mirror/__init__.py \
  examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py
python -m ruff format --check \
  vmec_jax/mirror/free_boundary.py \
  vmec_jax/mirror/api.py \
  vmec_jax/mirror/__init__.py \
  examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py
python -m sphinx -W -j auto -b html docs docs/_build/html
git diff --check
```

Result: all checks passed.

### File structure and best-practice notes

- The normal-field-slope proposal is a public helper because it is a reusable
  physical candidate direction.
- The example still owns candidate selection and pilot bookkeeping while this
  remains a workflow diagnostic rather than a solver API.
- The forced mode is intentionally exposed so users can inspect the pressure
  versus normal-field tradeoff.

### Best next steps

1. Commit and push M12l.
2. Add a two-mode candidate search over scale and normal-field-slope
   amplitudes:
   - small grid search or least-squares fit in normalized pressure/normal-field
     residual space;
   - include no-op fallback;
   - accept only candidates that improve combined merit and do not increase
     `B_ext.n` beyond a configurable tolerance.
3. Once the candidate search gives a useful accepted step, promote the pilot
   loop into a reusable helper.

### Completion percentages after M12l

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `86%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `85%`.
- I/O schema and docs: `90%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `57%`.
- Stellarator-mirror hybrid lane: `10%`.
- ESSOS circular-coil mirror beta scan: `45%`.
- PR merge readiness overall: `90%`.

### User input needed

No user input is needed.

---

## 75. 2026-06-18 M12m mixed scale/normal-field LCFS candidate

This tranche added a two-direction LCFS proposal that combines the smooth
pressure scale direction with the field-line-slope direction.  The goal is to
move beyond pure pressure or pure normal-field candidates and find small
updates that improve combined merit while satisfying a strict normal-field
nonincrease guard.

### Steps taken

- Added `propose_axisymmetric_mirror_lcfs_mixed_update`.
- Exported the helper through the public mirror API.
- Added `--lcfs-proposal-mode mixed` to the root circular-coil example.
- Updated `best_predicted` candidate selection so it now compares:
  - local pressure update;
  - shape-preserving scale update;
  - normal-field-slope update;
  - mixed scale/normal-field update;
  - no-op.
- Tightened the LCFS diagnostic plot label so the two-panel plot no longer has
  overlapping y-axis text.
- Added a synthetic unit test that verifies the mixed update improves pressure
  balance without increasing side-boundary normal field.
- Updated docs and the mirror examples README.

### Results obtained

Generated artifacts:

- `results/mirror/m12m_mixed_lcfs_strict_pilot/free_boundary_circular_coils_metrics.json`.
- `results/mirror/m12m_mixed_lcfs_strict_pilot/figures/fixed_boundary_beta_10_lcfs_step_1/free_boundary_circular_coils_beta_10_lcfs_step_1_lcfs_diagnostic.png`.
- `results/mirror/m12m_mixed_lcfs_strict_pilot/figures/fixed_boundary_beta_10/free_boundary_circular_coils_beta_10_mirror_boundary_3d.png`.

Strict-guard beta-10 candidate summary:

| candidate | predicted merit | predicted pressure RMS | predicted `B_ext.n` RMS |
| :--- | ---: | ---: | ---: |
| local pressure | `1.000000106` | `1.803438138` | `1.1115358e-02` |
| scale pressure | `0.996004406` | `1.796260719` | `8.783073e-03` |
| normal-field slope | `1.004048383` | `1.810804122` | `4.486880e-03` |
| mixed scale/normal-field | `0.997007531` | `1.798081565` | `7.655748e-03` |
| no-op | `1.000020372` | `1.803515366` | `7.657346e-03` |

With `--lcfs-require-bnormal-nonincrease`, the strict guard now selects
`mixed_scale_bnormal` instead of `noop`.  The actual low-resolution pilot solve
is accepted:

| row | merit | pressure RMS | `B_ext.n` RMS | accepted |
| :--- | ---: | ---: | ---: | :---: |
| baseline | `1.000020371650` | `1.803515365850` | `7.657346104349e-03` | n/a |
| mixed pilot step 1 | `0.782835620661` | `1.411809156436` | `7.655747922838e-03` | `true` |

Interpretation:

- A small mixed direction resolves the pressure/normal-field tension seen in
  M12k-M12l for this beta-10 circular-coil baseline.
- The pure scale candidate still has lower unconstrained predicted merit, but
  it violates strict normal-field nonincrease.
- The pure normal-field-slope candidate reduces `B_ext.n` more strongly but is
  pressure-limited.
- The next useful step is to move the repeated pilot bookkeeping out of the
  root example into a reusable LCFS pilot-step helper.

### How it was tested

Focused free-boundary and root example tests:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_example_runs_without_plots \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_strict_bnormal_guard_can_skip_pilot \
  -q
```

Result: `17 passed in 9.14s`.

Strict-guard plotted example:

```bash
JAX_ENABLE_X64=1 python examples/mirror_free_boundary_circular_coils.py \
  --outdir results/mirror/m12m_mixed_lcfs_strict_pilot \
  --betas 10 \
  --ntheta 24 \
  --nxi 33 \
  --n-segments 256 \
  --run-fixed-boundary-baseline \
  --run-lcfs-pilot \
  --lcfs-pilot-steps 1 \
  --baseline-maxiter 0 \
  --lcfs-require-bnormal-nonincrease
```

Result: metrics JSON, setup JSON, baseline and pilot `mout` files, and plot
bundles written.  The LCFS diagnostic and 3D boundary plots were opened and
visually checked.

Lint/format/docs/whitespace:

```bash
python -m ruff check \
  vmec_jax/mirror/free_boundary.py \
  vmec_jax/mirror/api.py \
  vmec_jax/mirror/__init__.py \
  examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py
python -m ruff format --check \
  vmec_jax/mirror/free_boundary.py \
  vmec_jax/mirror/api.py \
  vmec_jax/mirror/__init__.py \
  examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py
python -m sphinx -W -j auto -b html docs docs/_build/html
git diff --check
```

Result: all checks passed.

### File structure and best-practice notes

- The mixed direction lives in `vmec_jax/mirror/free_boundary.py` because it is
  reusable proposal logic, not just example bookkeeping.
- The root example still owns exact coil-resampled candidate scoring and pilot
  rows while the LCFS loop is still a diagnostic workflow.
- The mixed candidate reuses the existing `MirrorLCFSUpdateProposal` structure,
  so plotting, JSON summaries, and pilot execution need no special-case
  downstream code.
- Plot cleanup stayed local to the root example LCFS diagnostic plot; the
  broader mirror plotting module remains unchanged.

### Best next steps

1. Commit and push M12m.
2. Extract the repeated LCFS pilot bookkeeping from
   `examples/mirror_free_boundary_circular_coils.py` into a small reusable
   helper:
   - candidate construction;
   - exact candidate scoring;
   - strict guard/no-op handling;
   - accepted/rejected pilot-row metadata.
3. Add a lightweight convergence row over `lcfs_pilot_steps=1,2,3` for the
   strict mixed beta-10 case to confirm monotonic merit behavior before
   promoting the workflow beyond the example.
4. Resume the stellarator-mirror hybrid boundary lane after the free-boundary
   pilot loop is factored and tested.

### Completion percentages after M12m

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `86%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `86%`.
- I/O schema and docs: `90%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `61%`.
- Stellarator-mirror hybrid lane: `10%`.
- ESSOS circular-coil mirror beta scan: `48%`.
- PR merge readiness overall: `90%`.

### User input needed

No user input is needed.

---

## 76. 2026-06-18 M12n LCFS candidate-set helper and strict mixed probe

This tranche started the pilot-loop simplification work by extracting the
standard LCFS proposal construction into one reusable helper.  The root
free-boundary example still owns exact coil-resampled scoring and actual
fixed-boundary pilot solves, but it no longer repeats the local/scale/bnormal/
mixed/no-op proposal construction at each pilot step.

### Steps taken

- Added `propose_axisymmetric_mirror_lcfs_candidate_set`.
- Exported the helper through the public mirror API.
- Simplified `_select_lcfs_proposal` in the root circular-coil example to:
  - build the standard candidate tuple once;
  - score every candidate with exact coil-resampled `B_ext.n`;
  - map explicit modes to strategy names;
  - apply the strict normal-field guard over the scored candidates.
- Added unit coverage for the standard candidate order.
- Ran a three-step strict mixed beta-10 probe to check whether repeated pilot
  steps remain monotonic before moving more loop logic into helpers.

### Results obtained

Generated artifacts:

- `results/mirror/m12n_mixed_lcfs_three_step_probe/free_boundary_circular_coils_metrics.json`.

Strict mixed beta-10 three-step probe:

| row | merit | pressure RMS | `B_ext.n` RMS | accepted |
| :--- | ---: | ---: | ---: | :---: |
| baseline | `1.000020371650` | `1.803515365850` | `7.657346104349e-03` | n/a |
| step 1 | `0.782835620661` | `1.411809156436` | `7.655747922838e-03` | `true` |
| step 2 | `0.594625153775` | `1.072354482550` | `7.615656217692e-03` | `true` |
| step 3 | `0.431268555842` | `0.777718985048` | `7.442445773682e-03` | `true` |

The repeated strict mixed pilot stays monotonic in combined merit and keeps the
side-boundary normal field nonincreasing across all accepted steps.

### How it was tested

Focused free-boundary and root example tests:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_example_runs_without_plots \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_strict_bnormal_guard_can_skip_pilot \
  -q
```

Result: `18 passed in 8.94s`.

Three-step strict mixed probe:

```bash
JAX_ENABLE_X64=1 python examples/mirror_free_boundary_circular_coils.py \
  --outdir results/mirror/m12n_mixed_lcfs_three_step_probe \
  --betas 10 \
  --ntheta 24 \
  --nxi 33 \
  --n-segments 256 \
  --run-fixed-boundary-baseline \
  --run-lcfs-pilot \
  --lcfs-pilot-steps 3 \
  --baseline-maxiter 0 \
  --lcfs-require-bnormal-nonincrease \
  --no-plots
```

Result: metrics JSON written with three accepted pilot rows.

Lint/format/whitespace:

```bash
python -m ruff check \
  vmec_jax/mirror/free_boundary.py \
  vmec_jax/mirror/api.py \
  vmec_jax/mirror/__init__.py \
  examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py
python -m ruff format --check \
  vmec_jax/mirror/free_boundary.py \
  vmec_jax/mirror/api.py \
  vmec_jax/mirror/__init__.py \
  examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py
git diff --check
```

Result: all checks passed.

### File structure and best-practice notes

- Candidate construction now has one public entry point in
  `vmec_jax/mirror/free_boundary.py`.
- Exact candidate scoring stays in the root example because it depends on the
  chosen coil provider and grid used by that workflow.
- The helper returns existing `MirrorLCFSUpdateProposal` objects, preserving the
  downstream JSON and plotting schema.

### Best next steps

1. Commit and push M12n.
2. Extract exact candidate scoring into a small helper that returns the chosen
   proposal, candidate summaries, and guard status.
3. Then extract pilot row creation so skipped, rejected, and accepted rows are
   generated through one tested path.
4. Add a plotted three-step strict mixed run after the helper extraction so the
   multi-step convergence row has figures, not only JSON.

### Completion percentages after M12n

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `86%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `86%`.
- I/O schema and docs: `90%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `63%`.
- Stellarator-mirror hybrid lane: `10%`.
- ESSOS circular-coil mirror beta scan: `50%`.
- PR merge readiness overall: `90%`.

### User input needed

No user input is needed.

---

## 77. 2026-06-18 M12o LCFS selection and pilot-row helpers

This tranche finished the next simplification slice for the circular-coil
free-boundary example.  Candidate selection now returns an explicit selection
object with exact coil-resampled scoring metadata, and skipped/completed pilot
rows are generated through shared helper functions instead of hand-built JSON
blocks in the main solve loop.

### Steps taken

- Added the private `_LCFSProposalSelection` dataclass in the root
  `examples/mirror_free_boundary_circular_coils.py` workflow.
- Updated `_select_lcfs_proposal` to return:
  - chosen proposal;
  - exact candidate summaries;
  - guard-allowed strategy list;
  - guard rejection reason when the strict normal-field guard leaves only
    no-op.
- Added shared pilot-row helpers:
  - `_next_proposal_fields`;
  - `_skipped_lcfs_pilot_row`;
  - `_completed_lcfs_pilot_row`.
- Added root example test assertions for the new guard metadata fields.
- Ran a plotted three-step strict mixed beta-10 case after the helper
  extraction.

### Results obtained

Generated artifacts:

- `results/mirror/m12o_mixed_lcfs_three_step_plots/free_boundary_circular_coils_metrics.json`.
- `results/mirror/m12o_mixed_lcfs_three_step_plots/figures/fixed_boundary_beta_10_lcfs_step_3/free_boundary_circular_coils_beta_10_lcfs_step_3_lcfs_diagnostic.png`.
- `results/mirror/m12o_mixed_lcfs_three_step_plots/figures/fixed_boundary_beta_10_lcfs_step_3/free_boundary_circular_coils_beta_10_lcfs_step_3_mirror_boundary_3d.png`.

Strict mixed beta-10 plotted three-step probe:

| row | merit | pressure RMS | `B_ext.n` RMS | accepted | figures |
| :--- | ---: | ---: | ---: | :---: | ---: |
| baseline | `1.000020371650` | `1.803515365850` | `7.657346104349e-03` | n/a | `11` |
| step 1 | `0.782835620661` | `1.411809156436` | `7.655747922838e-03` | `true` | `11` |
| step 2 | `0.594625153775` | `1.072354482550` | `7.615656217692e-03` | `true` | `11` |
| step 3 | `0.431268555842` | `0.777718985048` | `7.442445773682e-03` | `true` | `11` |

The baseline strict guard allowed `bnormal_slope`, `mixed_scale_bnormal`, and
`noop`; it selected `mixed_scale_bnormal`.  The three pilot rows remain
accepted and monotonic after the row-helper extraction.

### How it was tested

Focused root example tests:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_example_runs_without_plots \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_strict_bnormal_guard_can_skip_pilot \
  -q
```

Result: `2 passed in 7.01s`.

Plotted three-step strict mixed probe:

```bash
JAX_ENABLE_X64=1 python examples/mirror_free_boundary_circular_coils.py \
  --outdir results/mirror/m12o_mixed_lcfs_three_step_plots \
  --betas 10 \
  --ntheta 24 \
  --nxi 33 \
  --n-segments 256 \
  --run-fixed-boundary-baseline \
  --run-lcfs-pilot \
  --lcfs-pilot-steps 3 \
  --baseline-maxiter 0 \
  --lcfs-require-bnormal-nonincrease
```

Result: metrics JSON, setup JSON, baseline `mout`, three pilot `mout` files,
and plotted bundles written.  The step-3 LCFS diagnostic and 3D boundary plots
were opened and visually checked.

Lint/format/whitespace:

```bash
python -m ruff check \
  examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_examples.py
python -m ruff format --check \
  examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_examples.py
git diff --check
```

Result: all checks passed.

### File structure and best-practice notes

- Exact candidate scoring remains private to the root example because it is
  tied to the example's circular-coil provider and plotting grid.
- The selection dataclass makes guard behavior explicit without adding a new
  public API object too early.
- Pilot row helpers centralize skipped and completed row schemas, reducing
  drift risk as more pilot modes are added.
- The main fixed-boundary baseline loop is still long, but the largest JSON
  assembly blocks are now factored and can be moved into a reusable pilot-step
  helper in the next tranche.

### Best next steps

1. Commit and push M12o.
2. Extract a reusable one-step LCFS pilot helper from the remaining loop body:
   - run fixed-boundary candidate solve;
   - write/reload `mout`;
   - sample external field;
   - compute LCFS diagnostic and merit;
   - build next candidate selection;
   - optionally write plots.
3. After that extraction, update docs with the current strict mixed pilot
   workflow and proceed to the stellarator-mirror hybrid boundary lane.

### Completion percentages after M12o

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `86%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `87%`.
- I/O schema and docs: `90%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `65%`.
- Stellarator-mirror hybrid lane: `10%`.
- ESSOS circular-coil mirror beta scan: `52%`.
- PR merge readiness overall: `90%`.

### User input needed

No user input is needed.

---

## 78. 2026-06-18 M12p one-step LCFS pilot helper

This tranche extracted the actual one-step LCFS pilot execution from the
root circular-coil example loop.  The main beta-case loop now delegates the
candidate fixed-boundary solve, `mout` write/reload, external-field resampling,
LCFS diagnostic, merit evaluation, next-candidate selection, optional plot
generation, and completed row construction to one helper.

### Steps taken

- Added `_LCFSPilotStepResult`.
- Added `_run_lcfs_pilot_step`.
- Replaced the in-loop pilot solve/write/reload/sample/diagnose/select/plot
  block with one helper call.
- Kept skipped-row handling separate so no-op/guarded cases remain explicit.
- Ran a no-plot three-step strict mixed beta-10 probe after the extraction.

### Results obtained

Generated artifact:

- `results/mirror/m12p_one_step_helper_probe/free_boundary_circular_coils_metrics.json`.

Strict mixed beta-10 three-step helper probe:

| row | merit | pressure RMS | `B_ext.n` RMS | accepted |
| :--- | ---: | ---: | ---: | :---: |
| baseline | `1.000020371650` | `1.803515365850` | `7.657346104349e-03` | n/a |
| step 1 | `0.782835620661` | `1.411809156436` | `7.655747922838e-03` | `true` |
| step 2 | `0.594625153775` | `1.072354482550` | `7.615656217692e-03` | `true` |
| step 3 | `0.431268555842` | `0.777718985048` | `7.442445773682e-03` | `true` |

The state handoff through `_LCFSPilotStepResult` preserves the monotonic
three-step strict mixed behavior from M12n/M12o.

### How it was tested

Focused root example tests:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_example_runs_without_plots \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_strict_bnormal_guard_can_skip_pilot \
  -q
```

Result: `2 passed in 7.10s`.

Three-step strict mixed helper probe:

```bash
JAX_ENABLE_X64=1 python examples/mirror_free_boundary_circular_coils.py \
  --outdir results/mirror/m12p_one_step_helper_probe \
  --betas 10 \
  --ntheta 24 \
  --nxi 33 \
  --n-segments 256 \
  --run-fixed-boundary-baseline \
  --run-lcfs-pilot \
  --lcfs-pilot-steps 3 \
  --baseline-maxiter 0 \
  --lcfs-require-bnormal-nonincrease \
  --no-plots
```

Result: metrics JSON written with three accepted pilot rows.

Lint/format:

```bash
python -m ruff check \
  examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_examples.py
python -m ruff format --check \
  examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_examples.py
```

Result: both checks passed.

### File structure and best-practice notes

- The helper is still private to the example because it performs file I/O,
  plotting, and provider-specific external-field sampling.
- The main loop now owns orchestration while `_run_lcfs_pilot_step` owns the
  repeated one-step workflow.
- The next clean boundary is to move the baseline row assembly into a helper
  or promote the pilot-step pieces into a small internal module once a second
  free-boundary example needs them.

### Best next steps

1. Commit and push M12p.
2. Update the mirror example README and docs with the current strict mixed
   pilot workflow and new guard metadata fields.
3. Start the stellarator-mirror hybrid boundary lane:
   - rotating ellipse segment over one field period;
   - linear mirror-axis segment;
   - smooth connection and up-down symmetric repetition;
   - root example and geometry plots.

### Completion percentages after M12p

- Geometry/grids/bases: `90%`.
- Field/energy/residual kernels: `86%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `87%`.
- I/O schema and docs: `90%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `67%`.
- Stellarator-mirror hybrid lane: `10%`.
- ESSOS circular-coil mirror beta scan: `53%`.
- PR merge readiness overall: `90%`.

### User input needed

No user input is needed.

---

## 79. 2026-06-18 M13a strict mixed docs and first hybrid boundary fixture

This tranche updated the user-facing free-boundary documentation for the
current strict mixed LCFS pilot workflow and started the stellarator-mirror
hybrid lane with a tested straight-axis rotating-ellipse boundary fixture.

### Steps taken

- Updated the mirror examples README with:
  - strict mixed pilot behavior;
  - exact `B_ext.n` guard semantics;
  - `lcfs_update_allowed_strategies`;
  - `lcfs_update_rejection_reason`.
- Updated mirror docs/index wording so the current first hybrid fixture is no
  longer described as entirely planned work.
- Added `MirrorBoundary.rotating_ellipse_mirror_hybrid`.
- Added a root example:
  - `examples/mirror_stellarator_hybrid_boundary.py`.
- Added tests for:
  - hybrid boundary positivity and up-down symmetry;
  - circular mirror end sections;
  - positive-Jacobian 3D geometry;
  - root example smoke coverage and metrics JSON.
- Ran the plotted hybrid example and visually checked 3D/cross-section plots.

### Results obtained

Generated artifacts:

- `results/mirror/m13a_stellarator_hybrid_boundary/mout_stellarator_hybrid_boundary.nc`.
- `results/mirror/m13a_stellarator_hybrid_boundary/stellarator_hybrid_boundary_metrics.json`.
- `results/mirror/m13a_stellarator_hybrid_boundary/figures/stellarator_hybrid_boundary_mirror_boundary_3d.png`.
- `results/mirror/m13a_stellarator_hybrid_boundary/figures/stellarator_hybrid_boundary_mirror_cross_sections.png`.

Hybrid plotted example metrics:

| quantity | value |
| :--- | ---: |
| radius min | `0.264199369638` |
| radius max | `0.336000000000` |
| mirror-end theta variation max | `5.551115123126e-17` |
| midplane theta variation | `7.180063036192e-02` |
| up-down symmetry error | `0.0` |
| min `sqrt(g)` | `4.188078415030e-02` |
| mirror ratio | `1.617536892278` |
| final `fsq` | `1.084111294800e-08` |

Interpretation:

- The first hybrid fixture keeps a straight mirror axis and implements the
  requested hybrid shape at the side-boundary level.
- The central stellarator-like segment has a rotating elliptical cross-section.
- The deformation tapers smoothly to circular mirror end sections.
- The boundary satisfies `r(theta, xi) = r(-theta, -xi)` on the check grid,
  providing the intended up-down symmetry for this straight-axis first cut.
- Higher-fidelity hybrid coordinates with a curved/rotating magnetic axis
  remain a later lane.

### How it was tested

Focused boundary/example tests:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_geometry_3d.py::test_rotating_ellipse_mirror_hybrid_boundary_is_symmetric_and_circular_at_ends \
  tests/mirror/test_mirror_geometry_3d.py::test_rotating_ellipse_mirror_hybrid_geometry_has_positive_jacobian \
  tests/mirror/test_mirror_examples.py::test_root_stellarator_hybrid_boundary_example_runs_without_plots \
  -q
```

Result: `3 passed in 1.83s`.

Plotted hybrid example:

```bash
JAX_ENABLE_X64=1 python examples/mirror_stellarator_hybrid_boundary.py \
  --outdir results/mirror/m13a_stellarator_hybrid_boundary \
  --ns 7 \
  --ntheta 25 \
  --nxi 33 \
  --mpol 6 \
  --maxiter 0
```

Result: `mout`, metrics JSON, and 10 standard mirror figures written.  The
3D boundary and cross-section plots were opened and visually checked.

Lint/format/docs/whitespace:

```bash
python -m ruff check \
  vmec_jax/mirror/core/boundary.py \
  examples/mirror_stellarator_hybrid_boundary.py \
  tests/mirror/test_mirror_geometry_3d.py \
  tests/mirror/test_mirror_examples.py
python -m ruff format --check \
  vmec_jax/mirror/core/boundary.py \
  examples/mirror_stellarator_hybrid_boundary.py \
  tests/mirror/test_mirror_geometry_3d.py \
  tests/mirror/test_mirror_examples.py
python -m sphinx -W -j auto -b html docs docs/_build/html
git diff --check
```

Result: all checks passed.

### File structure and best-practice notes

- The hybrid shape is a `MirrorBoundary` classmethod because it is a reusable
  fixed-boundary parameterization.
- The root example owns workflow-specific metrics and plotting, matching the
  existing root fixed-boundary examples.
- The first hybrid cut stays within the existing straight-axis radial-boundary
  representation, avoiding premature curved-axis coordinate changes.

### Best next steps

1. Commit and push M13a.
2. Add a hybrid example README/doc note with the precise straight-axis
   limitation and the planned curved-axis follow-up.
3. Add a small parameter scan over `epsilon`/`rotation_angle` to verify
   positive Jacobian and visual quality before using this boundary in solver
   convergence studies.
4. Then continue toward the higher-fidelity stellarator-mirror hybrid lane:
   smooth axis connection, repeated field-period construction, and fixed/free
   boundary examples.

### Completion percentages after M13a

- Geometry/grids/bases: `92%`.
- Field/energy/residual kernels: `86%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `88%`.
- I/O schema and docs: `91%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `67%`.
- Stellarator-mirror hybrid lane: `22%`.
- ESSOS circular-coil mirror beta scan: `53%`.
- PR merge readiness overall: `91%`.

### User input needed

No user input is needed.

---

## 80. 2026-06-18 CI fix, stopped-work audit, and toroidal hybrid correction

This tranche resumed the stopped work by checking the draft PR state, fixing
the only failing CI check, and correcting the hybrid lane after the user
clarified that the final stellarator-mirror hybrid should remain toroidal:
mirror-like geometry on either side, with stellarator-like geometry through the
corner regions.

### Steps taken

- Checked the local branch and draft PR:
  - branch: `codex/mirror-geometry`;
  - draft PR: `https://github.com/uwplasma/vmec_jax/pull/21`;
  - head before this tranche: `54056f12`.
- Inspected CI with `gh` and the GitHub PR metadata tool.
- Found one failing check:
  - `Fast Tests (py3.11 core coverage: freeb-external)`;
  - run `27760685902`, job `82134006425`.
- Read the failing log and traced both failures to
  `test_jax_vmec_mode_matrix_gradient_wrt_grpmn_matches_finite_difference`.
- Diagnosed the failure as finite-difference cancellation in a linear
  `grpmn -> mode_matrix` objective, not a production derivative defect.
- Changed only the central-difference step in that test from `1e-6` to `1e-3`;
  the tight `rtol=3e-9`/`atol=1e-11` assertion remains in place.
- Reconciled the hybrid plan:
  - M13a remains a straight-axis rotating-ellipse fixture and test case;
  - the final hybrid target is now a toroidal fixed-boundary lane with
    mirror-like side sections and stellarator-like corner sections;
  - the toroidal lane should reuse ordinary VMEC/JAX boundary, solver,
    preconditioner, and plotting conventions instead of extending the
    open-ended mirror coordinate model beyond its topology.

### Results obtained

- CI root cause:
  - `lasym=True`: CI reported max relative difference
    `5.71346392e-09`, just above `3e-9`;
  - `lasym=False`: CI reported max relative difference
    `3.30379762e-08`, also from finite-difference cancellation.
- Local step-size probe confirmed the objective is better conditioned with a
  larger step:
  - for `lasym=False`, `eps=1e-3` gave relative mismatch about `4.16e-12`;
  - for `lasym=True`, `eps=1e-3` gave relative mismatch about `1.35e-11`.
- No generated result files or figures were added to the repository.

### How it was tested

Targeted failing test:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/test_free_boundary_vacuum_adjoint.py::test_jax_vmec_mode_matrix_gradient_wrt_grpmn_matches_finite_difference \
  -q
```

Result: `2 passed in 1.22s`.

Local replica of the failing CI bucket:

```bash
python tools/diagnostics/ci_core_bucket_args.py freeb-external > /tmp/py311-core-freeb-external.txt
JAX_ENABLE_X64=1 VMEC_JAX_SKIP_PY311_COVERAGE_ONLY=1 \
  xargs pytest -q -n 4 -m "not full and not vmec2000 and not simsopt" \
  --durations=50 --cov=vmec_jax --cov-report= \
  < /tmp/py311-core-freeb-external.txt
```

Result: `269 passed, 42 skipped, 1 xfailed, 75 warnings in 42.81s`.

### File structure and best-practice notes

- The CI fix stays in the owning test file because it is a test-discretization
  issue, not a source-code derivative change.
- The comment explains why the larger finite-difference step is valid for this
  linear objective.
- The hybrid correction keeps the straight-axis mirror package scoped to open
  mirrors.  The new toroidal hybrid work should be implemented through the
  existing toroidal VMEC boundary/input path, with helper constructors and
  examples layered around that path.
- Repository weight policy remains unchanged: examples may generate plots and
  `results/` locally, but generated artifacts should not be committed unless a
  compressed documentation figure is intentionally promoted.

### Revised toroidal hybrid milestones

M13b: toroidal hybrid source audit and parameterization design.

- Inspect current VMEC/JAX boundary coefficient conventions, examples, tests,
  and plotting paths for fixed-boundary toroidal equilibria.
- Define a compact toroidal boundary generator with:
  - mirror-like side sections, meaning weakly helical or nearly axisymmetric
    elongated side arcs;
  - stellarator-like corner sections, meaning rotating ellipse/nonaxisymmetric
    shaping localized near the turns;
  - up-down symmetry by default;
  - field-period repeat metadata compatible with existing `nfp` handling.
- Keep the first generator in a small helper module or example-local helper
  until at least two examples/tests need it.

M13c: toroidal fixed-boundary hybrid fixture and example.

- Add a repo-root example that writes a normal VMEC input/solution path for the
  toroidal hybrid, plus plots of:
  - 3D LCFS;
  - cross sections at side and corner stations;
  - `|B|`;
  - iota and magnetic-well profiles when the ordinary toroidal solve provides
    them;
  - residual/`fsq` history.
- Reuse the ordinary toroidal solver, preconditioning, continuation, and plot
  functions before adding hybrid-specific logic.

M13d: toroidal hybrid convergence and parity checks.

- Run `ns`/mode-resolution convergence at low and moderate resolution.
- Compare fixed-boundary runtime/memory and convergence against a nearby
  conventional toroidal VMEC/JAX case.
- Use local VMEC2000 for parity checks where the input representation is
  directly VMEC-compatible.

M16 update: ESSOS circular-coil free-boundary scan.

- Keep the open-ended mirror circular-coil LCFS pilot lane separate from the
  toroidal hybrid lane.
- For the requested circular-coil ESSOS beta scan, keep using the mirror-native
  circular-loop bridge for the open mirror LCFS study unless the user later
  asks for a toroidal coil set.
- The default beta targets remain `1%`, `3%`, and `10%`, with low-resolution
  dry-run and one-step pilot tests before any longer free-boundary run.

### Best next steps

1. Commit and push the CI fix and plan/doc clarification.
2. Let CI run in the background and do not block on it unless a new failure is
   available.
3. Inspect the existing toroidal fixed-boundary boundary/input code, examples,
   and tests.
4. Implement M13b/M13c as a minimal toroidal hybrid boundary generator plus a
   root example that uses ordinary VMEC/JAX solver and plotting paths.
5. Add focused tests for coefficient construction, symmetry, example smoke, and
   plot-generation paths.
6. Continue the mirror-native free-boundary LCFS pilot lane toward a finite
   `1%`, `3%`, `10%` beta scan without committing generated outputs.

### Completion percentages after M80

- Geometry/grids/bases: `92%`.
- Field/energy/residual kernels: `86%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `88%`.
- I/O schema and docs: `91%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `67%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `5%`.
- ESSOS circular-coil mirror beta scan: `53%`.
- PR merge readiness overall: `91%`.

### User input needed

No user input is needed for the next step.  The default toroidal hybrid
interpretation is mirror-like side arcs and stellarator-like corner arcs with
up-down symmetry and ordinary VMEC-compatible toroidal boundary coefficients.

---

## 81. 2026-06-18 M13b/M13c first toroidal hybrid VMEC input fixture

This tranche implemented the first toroidal stellarator-mirror hybrid fixture
after the user clarified that the final hybrid target remains toroidal.  It
does not extend the open-ended mirror coordinate system.  Instead, it writes
ordinary VMEC fixed-boundary coefficients and enters the existing toroidal
`run_fixed_boundary`/`wout`/`plot_wout` path.

### Steps taken

- Added `vmec_jax/toroidal_hybrid.py` with:
  - `ToroidalHybridBoundarySamples`;
  - `sample_toroidal_stellarator_mirror_hybrid_boundary`;
  - `toroidal_stellarator_mirror_hybrid_indata`;
  - `toroidal_stellarator_mirror_hybrid_metrics`;
  - `evaluate_toroidal_hybrid_indata_boundary`.
- Added a root example:
  - `examples/toroidal_stellarator_mirror_hybrid.py`.
- The example writes:
  - a VMEC-compatible `input.toroidal_stellarator_mirror_hybrid`;
  - a metrics JSON file;
  - boundary-only 3D/top-view/cross-section plots;
  - optionally a standard `wout_*.nc` and `plot_wout` figures through
    `--run-solve`.
- Added tests in `tests/test_toroidal_hybrid.py` for:
  - stellarator symmetry;
  - side/corner localization;
  - exact low-mode reconstruction from written VMEC coefficients;
  - root-example smoke output without plots.
- Added public exports through `vmec_jax.api` and lazy top-level `vmec_jax`
  attributes.
- Updated mirror docs and example README to distinguish:
  - the straight-axis open-ended hybrid fixture;
  - the new toroidal VMEC-compatible hybrid fixture.

### Results obtained

Generated local artifacts, not committed:

- `results/toroidal_stellarator_mirror_hybrid_m13b/input.toroidal_stellarator_mirror_hybrid`.
- `results/toroidal_stellarator_mirror_hybrid_m13b/toroidal_stellarator_mirror_hybrid_metrics.json`.
- `results/toroidal_stellarator_mirror_hybrid_m13b/figures/toroidal_hybrid_lcfs_3d.png`.
- `results/toroidal_stellarator_mirror_hybrid_m13b/figures/toroidal_hybrid_top_view.png`.
- `results/toroidal_stellarator_mirror_hybrid_m13b/figures/toroidal_hybrid_cross_sections.png`.

Plotted example metrics:

| quantity | value |
| :--- | ---: |
| min `R` | `0.857516280079` |
| max `R` | `1.448000000000` |
| max `|Z|` | `0.253440000000` |
| stellarator-symmetry `R` error | `6.661338147751e-16` |
| stellarator-symmetry `Z` error | `1.804112415016e-16` |
| `RBC` coefficient count | `8` |
| `ZBS` coefficient count | `8` |

The 3D LCFS plot was visually checked.  The corner-weight coloring localizes
the stellarator-like shaping at the corner arcs, and the cross-section plot now
draws closed side/corner curves.

A low-iteration optional solve smoke also succeeded:

- command used `--run-solve --max-iter 1 --ns 9 --niter 3 --no-plots`;
- wrote `wout_toroidal_stellarator_mirror_hybrid.nc`;
- this verifies the generated input enters the ordinary toroidal fixed-boundary
  driver.

### How it was tested

New toroidal hybrid tests:

```bash
JAX_ENABLE_X64=1 pytest tests/test_toroidal_hybrid.py -q
```

Result: `3 passed in 1.16s`.

Plotted example:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid.py \
  --outdir results/toroidal_stellarator_mirror_hybrid_m13b \
  --ntheta-fit 64 \
  --nzeta-fit 64
```

Result: input, metrics JSON, and three PNG figures written.

Optional solve smoke:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid.py \
  --outdir results/toroidal_stellarator_mirror_hybrid_m13b_solve_smoke \
  --ntheta-fit 32 \
  --nzeta-fit 32 \
  --ns 9 \
  --niter 3 \
  --run-solve \
  --max-iter 1 \
  --no-plots
```

Result: metrics JSON and `wout_toroidal_stellarator_mirror_hybrid.nc` written.

### File structure and best-practice notes

- The new helper lives at top level, not under `vmec_jax.mirror`, because this
  is a closed toroidal VMEC boundary.
- The first generator is still compact: one source module, one root example,
  one focused test file.
- The boundary is sampled and projected to the existing VMEC helical Fourier
  convention using `project_to_modes`, avoiding a separate coefficient system.
- The example keeps solver execution optional so CI and quick docs runs remain
  light; the same input can still be solved and plotted through the ordinary
  toroidal CLI/API path.

### Best next steps

1. Run lint, docs, and focused tests for the new module/example/docs.
2. Commit and push M13b/M13c.
3. Add a low-cost toroidal hybrid `--run-solve` smoke test if runtime remains
   stable in CI.
4. Add a convergence script over `ns`, `mpol`, and `ntor` for this toroidal
   hybrid input, tracking `fsq`, iota, magnetic well, runtime, and memory.
5. Use local VMEC2000 for parity on the generated input once the low-resolution
   vmec_jax solve is stable.

### Completion percentages after M81

- Geometry/grids/bases: `93%`.
- Field/energy/residual kernels: `86%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `88%`.
- I/O schema and docs: `92%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `67%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `18%`.
- ESSOS circular-coil mirror beta scan: `53%`.
- PR merge readiness overall: `91%`.

### User input needed

No user input is needed.

---

## 82. 2026-06-18 M13d toroidal hybrid convergence runner

This tranche added the first repeatable convergence harness for the toroidal
stellarator-mirror hybrid fixture.  The harness stays on the ordinary
toroidal VMEC/JAX input, solver, WOUT, and profile-diagnostic path; it is not a
mirror-native open-ended solve.

### Steps taken

- Checked PR #21 CI with the GitHub Actions helper:
  - no failing checks were detected for the latest pushed commit at the time of
    this audit;
  - the branch remained draft and mergeable.
- Added `examples/toroidal_stellarator_mirror_hybrid_convergence.py`.
- Added a focused smoke test in `tests/test_toroidal_hybrid.py` that runs the
  convergence example without solves or plots.
- Updated `examples/mirror/README.md` with the new root example command and
  diagnostic outputs.
- Extended the run-solve rows to record:
  - runtime;
  - iteration count;
  - final `fsq`;
  - convergence flag;
  - aspect ratio from the ordinary WOUT helper;
  - mean iota from the ordinary VMEC/JAX iota-profile helper;
  - magnetic-well proxy from the ordinary finite-beta helper;
  - the generated `wout_toroidal_stellarator_mirror_hybrid.nc` path.

### Results obtained

No generated results were committed.  Local ignored artifacts were written
under `results/` only for validation.

No-solve convergence plot run:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_stellarator_mirror_hybrid_convergence_m13d_plot \
  --ns-array 7,9 \
  --mode-pairs 5:4 \
  --ntheta-fit 32 \
  --nzeta-fit 32
```

Result:

- wrote a JSON summary and
  `figures/toroidal_hybrid_convergence.png`;
- the plotted fit errors were at machine precision for the two rows.

One-iteration solve smoke:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_stellarator_mirror_hybrid_convergence_m13d_solve_smoke2 \
  --ns-array 7 \
  --mode-pairs 5:4 \
  --ntheta-fit 32 \
  --nzeta-fit 32 \
  --niter 3 \
  --run-solve \
  --max-iter 1 \
  --no-plots
```

Result for `ns007_mpol05_ntor04`:

| quantity | value |
| :--- | ---: |
| max boundary fit error | `6.661338147751e-16` |
| runtime | `6.688981666 s` |
| VMEC/JAX iterations | `1` |
| final `fsq` | `1.222006074808e-02` |
| converged | `false` |
| aspect | `5.661074062196` |
| mean iota | `2.768637859473e-02` |
| magnetic-well proxy | `-1.686348206555e-02` |

The false convergence flag is expected for a deliberate one-iteration smoke
row.

### How it was tested

Focused tests:

```bash
JAX_ENABLE_X64=1 pytest tests/test_toroidal_hybrid.py -q
```

Result: `4 passed in 1.93s`.

Lint and format checks:

```bash
python -m ruff check \
  examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  tests/test_toroidal_hybrid.py
python -m ruff format --check \
  examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  tests/test_toroidal_hybrid.py
```

Result: all checks passed; both Python files were already formatted.

Documentation:

```bash
python -m sphinx -W -j auto -b html docs docs/_build/html
```

Result: build succeeded.

Whitespace:

```bash
git diff --check
```

Result: no whitespace errors.

### File structure and best-practice notes

- The convergence runner is a separate root example because it is a workflow
  script, not a reusable physics kernel.
- It reuses `vmec_jax.toroidal_hybrid` for boundary construction and the
  ordinary `run_fixed_boundary`/`write_wout_from_fixed_boundary_run` path for
  solves.
- It writes generated files below user-selected output directories, with the
  documented default under ignored `results/`.
- The CI-facing test exercises the cheap no-solve path, while real solve rows
  remain opt-in through `--run-solve`.

### Finite plan from here

M13e: toroidal hybrid solve convergence.

- Run a low-resolution matrix over `ns`, `mpol`, and `ntor` with
  `--run-solve`, initially with modest iteration budgets.
- Add optional CSV export for convergence rows so longer local/GPU runs can be
  compared without hand-parsing JSON.
- Add iota and magnetic-well plots when solve rows are present.
- Keep the committed code light and keep generated plots/results out of git.

M13f: VMEC2000 parity for the toroidal hybrid input.

- Run local VMEC2000 on at least one generated low-resolution input.
- Compare convergence status, `fsq`, boundary coefficients, aspect, iota, and
  available WOUT scalar/profile diagnostics against VMEC/JAX.
- Log runtime and memory where practical.

M13g: toroidal hybrid geometry refinement.

- Replace the first analytic side/corner weighting with a more physically
  controlled parameterization:
  - mirror-like toroidal side arcs;
  - stellarator-like corner arcs;
  - smooth periodic connection;
  - up-down symmetry by default;
  - optional helicity and corner-width controls.
- Keep the parameterization VMEC-compatible and differentiable.

M10/M13h: differentiable solved-state path.

- Keep fast CLI examples on NumPy/SciPy/host-side loops when that is the
  better runtime path.
- For research-grade derivatives, add a solved-state API based on implicit
  differentiation or adjoint/custom linear-solve differentiation instead of
  differentiating through long nonlinear iteration histories.
- Reuse the regular toroidal solver's preconditioning, residual normalization,
  line-search, and profile helpers wherever possible.

M16: ESSOS circular-coil free-boundary beta scan.

- Keep the open-ended mirror LCFS pilot separate from the toroidal hybrid lane.
- Build the default `1%`, `3%`, and `10%` beta scan through the existing
  ESSOS-compatible circular-loop bridge.
- Promote only validated setup scripts, summaries, and compressed illustrative
  figures to docs; keep run directories ignored.

Documentation cleanup:

- Update the PR body after each major section so GitHub does not lag the local
  plan.
- Keep `docs/mirror/` and `examples/mirror/README.md` synchronized with source
  status and known limitations.
- Prefer compact source modules and pedagogical docstrings over one-off helper
  proliferation.

### Best next steps

1. Commit and push M13d.
2. Refresh the PR body so it reflects sections 80-82.
3. Add CSV export and solved-row profile plots to the toroidal hybrid
   convergence runner.
4. Run the first modest `--run-solve` grid locally, then move longer rows to
   the office GPU/VMEC2000 environment.
5. Start VMEC2000 parity on the generated toroidal-hybrid input.

### Completion percentages after M82

- Geometry/grids/bases: `93%`.
- Field/energy/residual kernels: `86%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `88%`.
- I/O schema and docs: `92%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `67%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `24%`.
- ESSOS circular-coil mirror beta scan: `53%`.
- PR merge readiness overall: `92%`.

### User input needed

No user input is needed.  The current default toroidal hybrid interpretation is
mirror-like side arcs connected by stellarator-like corner arcs in a closed
toroidal VMEC geometry.

---

## 83. 2026-06-18 M13e toroidal hybrid convergence CSV and profile plots

This tranche turned the toroidal hybrid convergence runner from a JSON-only
smoke harness into a more useful diagnostic script for actual solve rows.

### Steps taken

- Added a compact CSV export:
  - `toroidal_stellarator_mirror_hybrid_convergence.csv`.
- Added `fsq_history` to solved JSON rows.
- Added solved-row plotting:
  - convergence summary;
  - residual/`fsq` history;
  - WOUT profile figure with iota and Mercier `DWell`.
- Fixed the WOUT reader path in the example by importing `read_wout`
  explicitly from `vmec_jax.wout`.
- Updated the toroidal hybrid smoke test to assert that CSV output exists and
  that no-solve rows keep an empty `fsq_history`.
- Updated `examples/mirror/README.md` to document JSON/CSV exports and the new
  solved-row plots.

### Results obtained

Solved plotting smoke:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_stellarator_mirror_hybrid_convergence_m13e_plot_smoke2 \
  --ns-array 7 \
  --mode-pairs 5:4 \
  --ntheta-fit 32 \
  --nzeta-fit 32 \
  --niter 3 \
  --run-solve \
  --max-iter 1
```

Result for `ns007_mpol05_ntor04`:

| quantity | value |
| :--- | ---: |
| max boundary fit error | `6.661338147751e-16` |
| runtime | `7.087745458 s` |
| VMEC/JAX iterations | `1` |
| final `fsq` | `1.222006074808e-02` |
| converged | `false` |
| aspect | `5.661074062196` |
| mean iota | `2.768637859473e-02` |
| magnetic-well proxy | `-1.686348206555e-02` |

Generated local ignored figures:

- `results/toroidal_stellarator_mirror_hybrid_convergence_m13e_plot_smoke2/figures/toroidal_hybrid_convergence.png`;
- `results/toroidal_stellarator_mirror_hybrid_convergence_m13e_plot_smoke2/figures/toroidal_hybrid_fsq_history.png`;
- `results/toroidal_stellarator_mirror_hybrid_convergence_m13e_plot_smoke2/figures/toroidal_hybrid_profiles.png`.

The profile plot was visually checked.  The iota profile rendered correctly.
The `DWell` profile is zero for this one-iteration vacuum-style smoke, which
is expected; it should become more informative for finite-beta or longer
solved rows.

### How it was tested

Focused tests:

```bash
JAX_ENABLE_X64=1 pytest tests/test_toroidal_hybrid.py -q
```

Result: `4 passed in 2.16s`.

Lint and format:

```bash
python -m ruff check \
  examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  tests/test_toroidal_hybrid.py
python -m ruff format --check \
  examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  tests/test_toroidal_hybrid.py
```

Result: all checks passed; both files were formatted.

Docs:

```bash
python -m sphinx -W -j auto -b html docs docs/_build/html
```

Result: build succeeded.

Whitespace:

```bash
git diff --check
```

Result: no whitespace errors.

### File structure and best-practice notes

- The plotting helpers remain local to the root example because they are
  workflow outputs, not reusable core geometry or solver logic.
- The example now has a table-first output (`CSV`) for convergence studies and
  JSON for richer row metadata.
- Solved-row profile plotting uses `vmec_jax.wout.read_wout`, preserving the
  ordinary WOUT schema and avoiding a duplicate profile parser.
- Generated results stay under ignored `results/`.

### Best next steps

1. Commit and push M13e.
2. Run a small multi-row `--run-solve` convergence grid with
   `ns=7,9` and one or two mode pairs.
3. Use the grid output to decide whether the first toroidal hybrid fixture is
   solver-friendly enough for VMEC2000 parity.
4. Start VMEC2000 parity on a low-resolution generated input and compare WOUT
   scalar/profile diagnostics.
5. Return to source simplification only after the convergence/parity bottleneck
   is visible.

### Completion percentages after M83

- Geometry/grids/bases: `93%`.
- Field/energy/residual kernels: `86%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `88%`.
- I/O schema and docs: `92%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `67%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `28%`.
- ESSOS circular-coil mirror beta scan: `53%`.
- PR merge readiness overall: `92%`.

### User input needed

No user input is needed.

---

## 84. 2026-06-18 M13e.1 best-residual convergence metrics

The first local multi-row convergence grid showed that the ordinary accelerated
VMEC/JAX iteration can be nonmonotone for the first toroidal hybrid fixture.
That made a single final-`fsq` grid summary too weak for diagnosing solver
behavior, so this tranche added best-history metrics.

### Steps taken

- Added solved-row fields:
  - `initial_fsq`;
  - `best_fsq`;
  - `best_iter`;
  - `fsq_reduction`;
  - `final_fsq`.
- Updated the convergence summary plot to use best `fsq` for solved rows.
- Kept full `fsq_history` in JSON for residual-history plots and downstream
  analysis.
- Updated the CSV export columns and example README.

### Results obtained

Four-row local grid:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_stellarator_mirror_hybrid_convergence_m13e_grid \
  --ns-array 7,9 \
  --mode-pairs 5:4,6:5 \
  --ntheta-fit 32 \
  --nzeta-fit 32 \
  --niter 12 \
  --run-solve \
  --max-iter 3
```

Result:

| case | final `fsq` | status |
| :--- | ---: | :--- |
| `ns007_mpol05_ntor04` | `1.831175339596e-02` | not converged |
| `ns007_mpol06_ntor05` | `1.807228867411e-02` | not converged |
| `ns009_mpol05_ntor04` | `1.933402066353e-02` | not converged |
| `ns009_mpol06_ntor05` | `1.924948523463e-02` | not converged |

The residual history dropped sharply on the first iteration and rose on the
third for all rows, indicating that the early accelerated path is nonmonotone
for this fixture.  The generated `fsq` history and WOUT profile plots were
visually checked.

Longer low-resolution diagnostic:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_stellarator_mirror_hybrid_convergence_m13e_ns7_long \
  --ns-array 7 \
  --mode-pairs 5:4 \
  --ntheta-fit 32 \
  --nzeta-fit 32 \
  --niter 100 \
  --ftol 1e-12 \
  --run-solve \
  --max-iter 20 \
  --no-plots
```

Result:

- `n_iter=20`;
- `converged=false`;
- final `fsq=1.883052951600e-05`;
- best `fsq=1.423987393420e-05` at iteration 18;
- the requested `ftol=1e-12` was not reached.

Best-metric smoke:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_stellarator_mirror_hybrid_convergence_m13f_best_smoke \
  --ns-array 7 \
  --mode-pairs 5:4 \
  --ntheta-fit 32 \
  --nzeta-fit 32 \
  --niter 20 \
  --ftol 1e-12 \
  --run-solve \
  --max-iter 5 \
  --no-plots
```

Result:

- `initial_fsq=1.225087897344e-02`;
- `best_fsq=7.077559614612e-03`;
- `best_iter=3`;
- `fsq_reduction=1.730946772690`;
- `final_fsq=9.744035513153e-03`;
- `converged=false`.

### How it was tested

Focused tests:

```bash
JAX_ENABLE_X64=1 pytest tests/test_toroidal_hybrid.py -q
```

Result: `4 passed in 1.98s`.

Lint and format:

```bash
python -m ruff format examples/toroidal_stellarator_mirror_hybrid_convergence.py
python -m ruff check \
  examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  tests/test_toroidal_hybrid.py
python -m ruff format --check \
  examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  tests/test_toroidal_hybrid.py
```

Result: one file was formatted, then lint and format checks passed.

Documentation and whitespace:

```bash
python -m sphinx -W -j auto -b html docs docs/_build/html
git diff --check
```

Result: docs built successfully and no whitespace errors were found.

### File structure and best-practice notes

- The best-residual metrics live in the example runner because they describe a
  convergence-study workflow, not a new solver API.
- Keeping initial/best/final values in CSV makes long local or GPU runs easy
  to compare without loading JSON.
- Full residual history remains in JSON for plotting and deeper inspection.

### Best next steps

1. Commit and push this best-residual metrics tranche.
2. Inspect the ordinary toroidal `vmec2000_iter` controls for a monotone or
   less aggressive first convergence study mode before VMEC2000 parity.
3. Run VMEC2000 on the same low-resolution generated input and compare whether
   its early `fsq` history is similarly nonmonotone.
4. If VMEC2000 converges more robustly, narrow the difference to solver
   controls, damping, restart, or normalization before changing the hybrid
   geometry fixture.

### Completion percentages after M84

- Geometry/grids/bases: `93%`.
- Field/energy/residual kernels: `86%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `88%`.
- I/O schema and docs: `92%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `67%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `31%`.
- ESSOS circular-coil mirror beta scan: `53%`.
- PR merge readiness overall: `92%`.

### User input needed

No user input is needed.

---

## 85. 2026-06-18 M13f VMEC2000 parity hook for toroidal hybrid runner

This tranche added an opt-in VMEC2000 parity path to the toroidal hybrid
convergence runner and used it on the low-resolution generated hybrid input.

### Steps taken

- Added `--run-vmec2000`, `--vmec2000-exec`, and
  `--vmec2000-timeout-s` options to
  `examples/toroidal_stellarator_mirror_hybrid_convergence.py`.
- Reused the existing `vmec_jax.vmec2000_exec` wrapper:
  - copy generated `input.*` to a case-local VMEC2000 work directory;
  - run `xvmec2000`;
  - parse `threed1.*`;
  - preserve local ignored VMEC2000 work products under `results/`.
- Added VMEC/JAX component histories:
  - `fsqr_history`;
  - `fsqz_history`;
  - `fsql_history`;
  - final and best component values.
- Added VMEC2000 row metrics:
  - return code and runtime;
  - parsed `threed1` path;
  - WOUT path;
  - iteration and `fsq` histories;
  - initial/best/final `fsq` and best iteration.
- Updated the residual-history plot to overlay VMEC/JAX and VMEC2000 histories.
- Updated `examples/mirror/README.md`.

### Results obtained

VMEC2000 smoke command:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_stellarator_mirror_hybrid_vmec2000_smoke_m13f \
  --ns-array 7 \
  --mode-pairs 5:4 \
  --ntheta-fit 32 \
  --nzeta-fit 32 \
  --niter 20 \
  --ftol 1e-12 \
  --run-solve \
  --max-iter 5 \
  --run-vmec2000 \
  --vmec2000-exec /Users/rogeriojorge/bin/xvmec2000
```

Result:

| diagnostic | VMEC/JAX | VMEC2000 |
| :--- | ---: | ---: |
| runtime | `6.880363625 s` | `0.208919458 s` |
| recorded rows | `5` | `2` |
| initial `fsq` | `1.225087897344e-02` | `7.091000000000e-02` |
| best `fsq` | `7.077559614612e-03` | `7.770000000000e-03` |
| best iteration | `3` | `20` |
| final `fsq` | `9.744035513153e-03` | `7.770000000000e-03` |
| final `fsqr` | `4.402392805632e-03` | `4.432781063259e-03` |
| final `fsqz` | `5.132793049570e-03` | `2.310638399457e-03` |
| final `fsql` | `2.088496579511e-04` | `1.028225492522e-03` |

The VMEC2000 run returned code `0` and wrote:

- `results/toroidal_stellarator_mirror_hybrid_vmec2000_smoke_m13f/ns007_mpol05_ntor04/vmec2000/threed1.toroidal_stellarator_mirror_hybrid`;
- `results/toroidal_stellarator_mirror_hybrid_vmec2000_smoke_m13f/ns007_mpol05_ntor04/vmec2000/wout_toroidal_stellarator_mirror_hybrid.nc`.

The residual-history overlay was visually checked.  VMEC/JAX reaches a
comparable best residual earlier in its recorded history, but the final row is
slightly worse than the best row.  VMEC2000 records only the first and final
rows in this short `threed1` output, so finer per-iteration parity requires
either fuller VMEC2000 diagnostics or comparing final WOUT components.

### How it was tested

Focused tests:

```bash
JAX_ENABLE_X64=1 pytest tests/test_toroidal_hybrid.py -q
```

Result: `4 passed in 1.99s`.

Lint and format:

```bash
python -m ruff format examples/toroidal_stellarator_mirror_hybrid_convergence.py
python -m ruff check \
  examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  tests/test_toroidal_hybrid.py
python -m ruff format --check \
  examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  tests/test_toroidal_hybrid.py
```

Result: one file was formatted, then lint and format checks passed.

Docs and whitespace:

```bash
python -m sphinx -W -j auto -b html docs docs/_build/html
git diff --check
```

Result: docs built successfully and no whitespace errors were found.

### File structure and best-practice notes

- VMEC2000 parity remains opt-in and local; CI and quick examples do not depend
  on a system VMEC2000 executable.
- The new code reuses the repository's existing VMEC2000 wrapper instead of
  creating a second subprocess/parser path.
- Parsed VMEC2000 fields are stored beside VMEC/JAX fields in the same JSON/CSV
  row, which keeps parity studies table-driven.
- Generated `threed1`, WOUT, and plots remain under ignored `results/`.

### Best next steps

1. Commit and push M13f parity-hook support.
2. Add a small test around `_summarize_fsq_history` and VMEC2000 parser usage
   without requiring the executable.
3. Decide whether VMEC/JAX should expose a "best state" or best-row diagnostics
   for this solver path, since final residual can be worse than best residual.
4. Run a longer VMEC/JAX row with solver controls closer to VMEC2000 and compare
   final WOUT component residuals.
5. Use the parity hook for `ns=9` and the next mode pair before changing the
   hybrid geometry parameterization.

### Completion percentages after M85

- Geometry/grids/bases: `93%`.
- Field/energy/residual kernels: `86%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `88%`.
- I/O schema and docs: `93%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `67%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `35%`.
- ESSOS circular-coil mirror beta scan: `53%`.
- PR merge readiness overall: `92%`.

### User input needed

No user input is needed.

---

## 86. 2026-06-18 M13f.1 VMEC/JAX solver-mode controls in convergence runner

This tranche exposed solver controls in the toroidal hybrid convergence runner
so the same generated input can be tested with the fast CLI path or a closer
VMEC2000-control path.

### Steps taken

- Added `--solver-mode {default,parity,accelerated}` to
  `examples/toroidal_stellarator_mirror_hybrid_convergence.py`.
- Added `--use-scan` / `--no-use-scan` to the same runner.
- Added `solver_mode` and `use_scan` columns to the CSV/JSON row outputs.
- Updated `examples/mirror/README.md` to recommend
  `--solver-mode parity --no-use-scan` for VMEC2000-control comparisons.

### Results obtained

Parity-mode, non-scan, `NITER_ARRAY=20`, VMEC/JAX capped at 20:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_stellarator_mirror_hybrid_parity_smoke_m13g \
  --ns-array 7 \
  --mode-pairs 5:4 \
  --ntheta-fit 32 \
  --nzeta-fit 32 \
  --niter 20 \
  --ftol 1e-12 \
  --run-solve \
  --max-iter 20 \
  --solver-mode parity \
  --no-use-scan \
  --run-vmec2000 \
  --vmec2000-exec /Users/rogeriojorge/bin/xvmec2000 \
  --no-plots
```

Result:

- VMEC/JAX final `fsq=1.839147093094e-05`;
- VMEC/JAX best `fsq=1.471854259816e-05` at iteration 18;
- VMEC2000 final `fsq=7.770000000000e-03` at iteration 20.

Parity-mode, non-scan, `NITER_ARRAY=100`, VMEC/JAX capped at 20:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_stellarator_mirror_hybrid_parity_niter100_m13g \
  --ns-array 7 \
  --mode-pairs 5:4 \
  --ntheta-fit 32 \
  --nzeta-fit 32 \
  --niter 100 \
  --ftol 1e-12 \
  --run-solve \
  --max-iter 20 \
  --solver-mode parity \
  --no-use-scan \
  --run-vmec2000 \
  --vmec2000-exec /Users/rogeriojorge/bin/xvmec2000 \
  --no-plots
```

Result:

- VMEC/JAX final `fsq=1.839147093094e-05`;
- VMEC2000 final `fsq=5.546000000000e-07` at iteration 100.

Parity-mode, non-scan, equal `100` iteration budget:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_stellarator_mirror_hybrid_parity_max100_m13g \
  --ns-array 7 \
  --mode-pairs 5:4 \
  --ntheta-fit 32 \
  --nzeta-fit 32 \
  --niter 100 \
  --ftol 1e-12 \
  --run-solve \
  --max-iter 100 \
  --solver-mode parity \
  --no-use-scan \
  --run-vmec2000 \
  --vmec2000-exec /Users/rogeriojorge/bin/xvmec2000 \
  --no-plots
```

Result:

| diagnostic | VMEC/JAX | VMEC2000 |
| :--- | ---: | ---: |
| runtime | `10.943271542 s` | `0.198557916 s` |
| final recorded iteration | `99` | `100` |
| best `fsq` | `1.094466932561e-10` | `5.546000000000e-07` |
| final `fsq` | `1.094466932561e-10` | `5.546000000000e-07` |
| final `fsqr` | `5.496583583656e-11` | not parsed as final component in row |
| final `fsqz` | `1.881559404507e-11` | not parsed as final component in row |
| final `fsql` | `3.566526337451e-11` | not parsed as final component in row |

VMEC/JAX reduced the residual well below the VMEC2000 `threed1` final value for
the same nominal iteration budget, but still did not mark strict convergence to
the requested `ftol=1e-12`.  This means the next parity question is no longer
whether the hybrid input is runnable; it is whether the VMEC/JAX strict
convergence flag, total-`fsq` target, and VMEC2000 `FTOL` stopping conventions
are being compared on exactly the same quantity.

### How it was tested

Focused tests:

```bash
JAX_ENABLE_X64=1 pytest tests/test_toroidal_hybrid.py -q
```

Result: `5 passed in 1.92s`.

Lint and format:

```bash
python -m ruff check \
  examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  tests/test_toroidal_hybrid.py
python -m ruff format --check \
  examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  tests/test_toroidal_hybrid.py
```

Result: all checks passed and both files were formatted.

Docs and whitespace:

```bash
python -m sphinx -W -j auto -b html docs docs/_build/html
git diff --check
```

Result: docs built successfully and no whitespace errors were found.

### File structure and best-practice notes

- Solver-control flags live in the convergence example, not the core
  `vmec_jax.toroidal_hybrid` boundary helper.
- The runner defaults remain fast (`accelerated`) while parity controls are
  explicit.
- The VMEC2000 path remains optional and does not affect tests or docs builds.

### Best next steps

1. Commit and push solver-control support.
2. Add a row field for the requested `ftol`, the VMEC/JAX strict-convergence
   target, and `converged_by_total_fsq` so parity tables explain why
   `fsq=1e-10` can still be flagged as not strictly converged.
3. Parse VMEC2000 final component values from WOUT into the VMEC2000 row fields
   to make component comparisons first-class.
4. Run `ns=9` parity once the row schema includes those convergence target
   fields.

### Completion percentages after M86

- Geometry/grids/bases: `93%`.
- Field/energy/residual kernels: `86%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `88%`.
- I/O schema and docs: `93%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `67%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `38%`.
- ESSOS circular-coil mirror beta scan: `53%`.
- PR merge readiness overall: `92%`.

### User input needed

No user input is needed.

---

## 87. 2026-06-18 M13f.2 convergence-target and VMEC2000 component fields

This tranche made the toroidal hybrid parity rows more self-explanatory by
recording the convergence target fields and VMEC2000 WOUT component residuals
directly in the CSV/JSON output.

### Steps taken

- Added row fields:
  - `requested_ftol`;
  - `fsq_total_target`;
  - `converged_strict`;
  - `converged_by_total_fsq`.
- Added VMEC2000 WOUT-derived fields:
  - `vmec2000_final_fsqr`;
  - `vmec2000_final_fsqz`;
  - `vmec2000_final_fsql`;
  - `vmec2000_aspect`;
  - `vmec2000_mean_iota`.
- Updated `examples/mirror/README.md`.

### Results obtained

Schema smoke:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_stellarator_mirror_hybrid_schema_smoke_m13h \
  --ns-array 7 \
  --mode-pairs 5:4 \
  --ntheta-fit 32 \
  --nzeta-fit 32 \
  --niter 20 \
  --ftol 1e-12 \
  --run-solve \
  --max-iter 5 \
  --solver-mode parity \
  --no-use-scan \
  --run-vmec2000 \
  --vmec2000-exec /Users/rogeriojorge/bin/xvmec2000 \
  --no-plots
```

Result:

| field | value |
| :--- | ---: |
| `requested_ftol` | `1.0e-12` |
| `fsq_total_target` | `None` |
| `converged` | `false` |
| `converged_strict` | `false` |
| `converged_by_total_fsq` | `false` |
| VMEC/JAX final `fsq` | `8.301471170622e-03` |
| VMEC2000 final `fsq` | `7.770000000000e-03` |
| VMEC2000 final `fsqr` | `4.432781063259e-03` |
| VMEC2000 final `fsqz` | `2.310638399457e-03` |
| VMEC2000 final `fsql` | `1.028225492522e-03` |
| VMEC2000 mean iota | `1.000535474965e-02` |

The `fsq_total_target` is intentionally `None` in parity mode; it is an
accelerated-mode stopping field.  The strict convergence flag therefore
continues to track the requested `ftol` convention.

### How it was tested

Focused tests:

```bash
JAX_ENABLE_X64=1 pytest tests/test_toroidal_hybrid.py -q
```

Result: `5 passed in 1.89s`.

Lint and format:

```bash
python -m ruff check \
  examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  tests/test_toroidal_hybrid.py
python -m ruff format --check \
  examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  tests/test_toroidal_hybrid.py
```

Result: all checks passed and both files were formatted.

Docs and whitespace:

```bash
python -m sphinx -W -j auto -b html docs docs/_build/html
git diff --check
```

Result: docs built successfully and no whitespace errors were found.

### File structure and best-practice notes

- The runner now carries enough schema to compare stopping rules without
  re-opening WOUT files by hand.
- VMEC2000 WOUT parsing reuses `vmec_jax.wout.read_wout`.
- The added fields are optional and empty unless the relevant solve path runs.

### Best next steps

1. Commit and push the schema update.
2. Run the `ns=9` parity row now that the row schema is complete.
3. Add a compact parity-summary plot or table for VMEC/JAX versus VMEC2000
   final residual components when both are present.
4. Return to the toroidal hybrid geometry parameterization after the parity
   summary identifies whether residual differences are solver-control or
   geometry-resolution limited.

### Completion percentages after M87

- Geometry/grids/bases: `93%`.
- Field/energy/residual kernels: `86%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `88%`.
- I/O schema and docs: `93%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `67%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `40%`.
- ESSOS circular-coil mirror beta scan: `53%`.
- PR merge readiness overall: `92%`.

### User input needed

No user input is needed.

---

## 88. 2026-06-18 M13f.3 parity component plot and ns=9 parity row

This tranche added a side-by-side final residual component plot for VMEC/JAX
and VMEC2000 parity rows, then ran the next radial-resolution parity case.

### Steps taken

- Added `toroidal_hybrid_parity_components.png` to the convergence runner when
  both VMEC/JAX and VMEC2000 final residual components are available.
- Updated `examples/mirror/README.md`.
- Ran the `ns=9`, `mpol=5`, `ntor=4` parity row with:
  - VMEC/JAX parity mode;
  - non-scan update loop;
  - `NITER_ARRAY=100`;
  - VMEC/JAX `max_iter=100`;
  - local `/Users/rogeriojorge/bin/xvmec2000`.

### Results obtained

Component-plot smoke:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_stellarator_mirror_hybrid_parity_components_smoke_m13i \
  --ns-array 7 \
  --mode-pairs 5:4 \
  --ntheta-fit 32 \
  --nzeta-fit 32 \
  --niter 20 \
  --ftol 1e-12 \
  --run-solve \
  --max-iter 5 \
  --solver-mode parity \
  --no-use-scan \
  --run-vmec2000 \
  --vmec2000-exec /Users/rogeriojorge/bin/xvmec2000
```

Generated and visually checked:

- `results/toroidal_stellarator_mirror_hybrid_parity_components_smoke_m13i/figures/toroidal_hybrid_parity_components.png`.

For the short `ns=7` smoke, the component plot shows VMEC/JAX with the larger
`fsqz` component and VMEC2000 with the larger `fsql` component.

`ns=9` parity row:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_stellarator_mirror_hybrid_parity_ns9_m13h \
  --ns-array 9 \
  --mode-pairs 5:4 \
  --ntheta-fit 32 \
  --nzeta-fit 32 \
  --niter 100 \
  --ftol 1e-12 \
  --run-solve \
  --max-iter 100 \
  --solver-mode parity \
  --no-use-scan \
  --run-vmec2000 \
  --vmec2000-exec /Users/rogeriojorge/bin/xvmec2000
```

Result:

| diagnostic | VMEC/JAX | VMEC2000 |
| :--- | ---: | ---: |
| runtime | `11.049476125 s` | `0.217696958 s` |
| best/final `fsq` | `2.055168581245e-10` | `9.160000000000e-07` |
| best iteration | `99` | `100` |
| final `fsqr` | `1.079914959190e-10` | `4.158481175634e-07` |
| final `fsqz` | `4.381988014471e-11` | `3.905312682467e-07` |
| final `fsql` | `5.370548206077e-11` | `1.088439080074e-07` |
| mean iota | `7.329089073972e-03` | `7.370706838615e-03` |

The residual-history plot was visually checked.  VMEC/JAX starts below the
VMEC2000 printed residual level and remains lower through the recorded history,
which indicates the remaining comparison is about initialization and stopping
conventions as much as force-kernel parity.

### How it was tested

Focused tests:

```bash
JAX_ENABLE_X64=1 pytest tests/test_toroidal_hybrid.py -q
```

Result: `5 passed in 1.95s`.

Lint and format:

```bash
python -m ruff check \
  examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  tests/test_toroidal_hybrid.py
python -m ruff format --check \
  examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  tests/test_toroidal_hybrid.py
```

Result: all checks passed and both files were formatted.

Docs and whitespace:

```bash
python -m sphinx -W -j auto -b html docs docs/_build/html
git diff --check
```

Result: docs built successfully and no whitespace errors were found.

### File structure and best-practice notes

- The parity component plot remains in the root convergence example, because
  it is workflow reporting rather than core solver logic.
- The plot uses existing row fields and does not re-run either solver.
- Generated plots remain under ignored `results/`.

### Best next steps

1. Commit and push the component plot and plan update.
2. Add an initialization-parity diagnostic: compare VMEC/JAX initial state
   residuals against VMEC2000's first `threed1` row and document the
   initialization difference.
3. Run the next mode-pair parity row (`mpol=6,ntor=5`) only after the
   initialization comparison is explicit.
4. Then return to M13g geometry refinement.

### Completion percentages after M88

- Geometry/grids/bases: `93%`.
- Field/energy/residual kernels: `86%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `88%`.
- I/O schema and docs: `93%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `67%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `42%`.
- ESSOS circular-coil mirror beta scan: `53%`.
- PR merge readiness overall: `92%`.

### User input needed

No user input is needed.

---

## 89. 2026-06-18 M13f.4 initial residual component fields

This tranche made the initialization difference explicit in the toroidal hybrid
parity rows by recording initial residual components for VMEC/JAX and VMEC2000.

### Steps taken

- Added VMEC/JAX row fields:
  - `initial_fsqr`;
  - `initial_fsqz`;
  - `initial_fsql`.
- Added VMEC2000 `threed1` row fields:
  - `vmec2000_initial_fsqr`;
  - `vmec2000_initial_fsqz`;
  - `vmec2000_initial_fsql`.
- Kept these fields in the CSV export so initialization comparisons are table
  driven.

### Results obtained

Schema smoke:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_stellarator_mirror_hybrid_initial_components_m13j \
  --ns-array 7 \
  --mode-pairs 5:4 \
  --ntheta-fit 32 \
  --nzeta-fit 32 \
  --niter 20 \
  --ftol 1e-12 \
  --run-solve \
  --max-iter 5 \
  --solver-mode parity \
  --no-use-scan \
  --run-vmec2000 \
  --vmec2000-exec /Users/rogeriojorge/bin/xvmec2000 \
  --no-plots
```

Result:

| component | VMEC/JAX initial | VMEC2000 initial |
| :--- | ---: | ---: |
| total `fsq` | `1.101895164557e-02` | `7.091000000000e-02` |
| `fsqr` | `5.710626825885e-03` | `4.280000000000e-02` |
| `fsqz` | `3.739018993131e-03` | `5.310000000000e-03` |
| `fsql` | `1.569305826559e-03` | `2.280000000000e-02` |

For this generated toroidal hybrid input, VMEC/JAX starts from a lower residual
than VMEC2000's first printed `threed1` row.  Most of the initial gap is in the
radial and lambda components.  This supports treating future parity differences
as an initialization/stopping-convention issue before changing the hybrid
boundary model.

### How it was tested

Focused tests:

```bash
JAX_ENABLE_X64=1 pytest tests/test_toroidal_hybrid.py -q
```

Result: `5 passed in 1.95s`.

Lint and format:

```bash
python -m ruff check \
  examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  tests/test_toroidal_hybrid.py
python -m ruff format --check \
  examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  tests/test_toroidal_hybrid.py
```

Result: all checks passed and both files were formatted.

Docs and whitespace:

```bash
python -m sphinx -W -j auto -b html docs docs/_build/html
git diff --check
```

Result: docs built successfully and no whitespace errors were found.

### File structure and best-practice notes

- The fields are schema additions to the convergence workflow only.
- No core solver behavior changed.
- Generated validation output remains ignored under `results/`.

### Best next steps

1. Commit and push this initialization-component schema update.
2. Add one concise initialization-parity note to the root example or docs after
   running the next mode-pair parity row.
3. Proceed to M13g geometry refinement once parity behavior is documented
   across `mpol=5,ntor=4` and `mpol=6,ntor=5`.

### Completion percentages after M89

- Geometry/grids/bases: `93%`.
- Field/energy/residual kernels: `86%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `88%`.
- I/O schema and docs: `93%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `67%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `43%`.
- ESSOS circular-coil mirror beta scan: `53%`.
- PR merge readiness overall: `92%`.

### User input needed

No user input is needed.

---

## 90. 2026-06-18 M13f.5 mode-pair parity evidence

This tranche ran the next toroidal hybrid parity mode pair after the row schema
was complete.  No source behavior changed; this is evidence for the plan and
for deciding when to move from parity instrumentation to geometry refinement.

### Steps taken

- Ran `mpol=6`, `ntor=5` with `ns=7,9`.
- Used:
  - VMEC/JAX `solver_mode=parity`;
  - VMEC/JAX non-scan update loop;
  - `NITER_ARRAY=100`;
  - VMEC/JAX `max_iter=100`;
  - local VMEC2000 through `/Users/rogeriojorge/bin/xvmec2000`.
- Visually checked the residual component comparison plot.

### Results obtained

Command:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_stellarator_mirror_hybrid_parity_mode65_m13j \
  --ns-array 7,9 \
  --mode-pairs 6:5 \
  --ntheta-fit 32 \
  --nzeta-fit 32 \
  --niter 100 \
  --ftol 1e-12 \
  --run-solve \
  --max-iter 100 \
  --solver-mode parity \
  --no-use-scan \
  --run-vmec2000 \
  --vmec2000-exec /Users/rogeriojorge/bin/xvmec2000
```

Result:

| case | VMEC/JAX final `fsq` | VMEC2000 final `fsq` | VMEC/JAX mean iota | VMEC2000 mean iota |
| :--- | ---: | ---: | ---: | ---: |
| `ns007_mpol06_ntor05` | `2.528886614099e-11` | `1.883000000000e-06` | `7.611228280938e-03` | `7.632971966475e-03` |
| `ns009_mpol06_ntor05` | `2.496867145599e-11` | `2.655000000000e-06` | `7.428331079955e-03` | `7.460177714933e-03` |

Initial residuals:

| case | VMEC/JAX initial `fsq` | VMEC2000 initial `fsq` |
| :--- | ---: | ---: |
| `ns007_mpol06_ntor05` | `1.465570385579e-10` | `7.007000000000e-02` |
| `ns009_mpol06_ntor05` | `1.089057628325e-10` | `7.271000000000e-02` |

The parity component plot was visually checked:

- `results/toroidal_stellarator_mirror_hybrid_parity_mode65_m13j/figures/toroidal_hybrid_parity_components.png`.

The result reinforces the section 89 conclusion: VMEC/JAX and VMEC2000 are not
starting from equivalent residual states for these generated inputs.  Mean iota
is close between codes, while final force residuals differ by orders of
magnitude after the same nominal iteration budget.  The next useful work is to
document or control initialization parity before changing physics conclusions.

### How it was tested

The command above ran both solvers and produced JSON, CSV, WOUT files, and
plots.  No committed source files changed for this evidence run.

### File structure and best-practice notes

- Results stayed under ignored `results/`.
- No generated artifacts were staged.
- The plan records only the concise metrics needed to guide the next
  implementation step.

### Best next steps

1. Add a short docs note explaining that current toroidal hybrid parity rows
   compare solved outcomes, not identical raw initial states.
2. Begin M13g: refine the toroidal side/corner parameterization while retaining
   the current parity runner as the regression harness.
3. Before claiming VMEC2000 numerical parity, add an option or fixture for a
   VMEC/JAX initial state closer to VMEC2000's raw initialization.

### Completion percentages after M90

- Geometry/grids/bases: `93%`.
- Field/energy/residual kernels: `86%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `88%`.
- I/O schema and docs: `93%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `67%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `44%`.
- ESSOS circular-coil mirror beta scan: `53%`.
- PR merge readiness overall: `92%`.

### User input needed

No user input is needed.

---
