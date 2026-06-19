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

## 91. 2026-06-18 CI coverage fix and toroidal-hybrid parity docs note

This tranche fixed the PR's immediate CI blocker and moved the section 90
parity caveat into user-facing docs.

### Steps taken

- Checked draft PR #21 and confirmed a single failing check:
  `Coverage Gate (py3.11 combined)`.
- Downloaded the CI coverage artifacts and inspected exact missed source lines.
- Added focused validation tests for:
  - toroidal-hybrid boundary geometry guards;
  - toroidal-hybrid mode extent guards;
  - non-stellarator-symmetric sampled-boundary rejection;
  - mirror circular-coil input guards;
  - mirror beta-case pressure-scale guards;
  - mirror LCFS merit and finite-difference response guards.
- Added concise parity caveats to:
  - `docs/mirror/overview.rst`;
  - `examples/mirror/README.md`.

### Results obtained

- The old CI gate failed because exact coverage was `94.96%` against the
  `95.00%` threshold.
- The new tests cover the visible toroidal-hybrid misses plus additional
  mirror free-boundary guard branches, enough to clear the small exact-coverage
  deficit without changing solver behavior or CI policy.
- The docs now explicitly say current toroidal hybrid VMEC/JAX-vs-VMEC2000
  rows compare solved outcomes from the same generated input, not identical
  raw initialized states.

### How it was tested

Focused tests:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/test_toroidal_hybrid.py \
  tests/mirror/test_mirror_free_boundary.py -q
```

Result: `46 passed in 4.05s`.

Lint and format:

```bash
python -m ruff check \
  tests/test_toroidal_hybrid.py \
  tests/mirror/test_mirror_free_boundary.py
python -m ruff format --check \
  tests/test_toroidal_hybrid.py \
  tests/mirror/test_mirror_free_boundary.py
```

Result: all checks passed.

Whitespace:

```bash
git diff --check
```

Result: no whitespace errors.

### File structure and best-practice notes

- The coverage fix is test-only and keeps behavior untouched.
- Documentation edits are limited to source docs, not generated `docs/_build`
  output.
- Generated caches and `results/` remain ignored and unstaged.

### Best next steps

1. Validate the docs build after the parity note.
2. Commit and push this docs tranche.
3. Start M13g geometry refinement:
   - expose controlled side/corner-shaping parameters;
   - preserve stellarator symmetry and positive cylindrical `R`;
   - run the existing convergence/parity runner as the regression harness;
   - keep generated plots under ignored `results/`.
4. Recheck CI after it has had time to run, without blocking active
   implementation on Actions latency.

### Completion percentages after M91

- Geometry/grids/bases: `93%`.
- Field/energy/residual kernels: `86%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `88%`.
- I/O schema and docs: `94%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `68%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `45%`.
- ESSOS circular-coil mirror beta scan: `53%`.
- PR merge readiness overall: `92%`.

### User input needed

No user input is needed.

---

## 92. 2026-06-18 M13g controlled toroidal-hybrid side/corner shaping

This tranche started M13g by making the toroidal hybrid boundary generator more
controllable without changing the ordinary toroidal VMEC solve path.

### Steps taken

- Added `side_power` and `corner_power` localization controls to
  `sample_toroidal_stellarator_mirror_hybrid_boundary`.
- Kept the localization weights nonnegative before applying fractional powers,
  so exploratory noninteger powers are numerically well defined.
- Exposed and recorded the existing/new shaping controls in the root examples:
  - `side_minor_modulation`;
  - `side_elongation`;
  - `side_power`;
  - `corner_amplitude`;
  - `corner_helicity`;
  - `corner_power`.
- Added the same controls to the convergence runner and CSV/JSON output.
- Extended tests so exact boundary roundtrips use integer powers with adequate
  `ntor`, while broader exploratory shaping remains allowed.

### Results obtained

Sharpened side/corner smoke command:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid.py \
  --outdir results/toroidal_hybrid_m13g_shape_smoke \
  --nfp 2 \
  --mpol 5 \
  --ntor 10 \
  --ns 7 \
  --niter 20 \
  --ftol 1e-9 \
  --major-radius 1.15 \
  --minor-radius 0.18 \
  --axis-oval 0.10 \
  --side-minor-modulation 0.16 \
  --side-elongation 0.35 \
  --side-power 2.0 \
  --corner-amplitude 0.025 \
  --corner-helicity 1 \
  --corner-power 2.0 \
  --ntheta-fit 32 \
  --nzeta-fit 32 \
  --run-solve \
  --max-iter 2
```

Geometry metrics:

| metric | value |
| :--- | ---: |
| `min_R` | `8.638915637183e-01` |
| `max_R` | `1.458800000000e+00` |
| `max_abs_Z` | `2.818800000000e-01` |
| `stellsym_R_error` | `6.661338147751e-16` |
| `stellsym_Z_error` | `1.942890293094e-16` |
| `side_r_span_mean` | `4.176000000000e-01` |
| `corner_r_span_mean` | `3.722168725634e-01` |
| `RBC` count | `12` |
| `ZBS` count | `14` |

Generated ignored plots checked visually:

- `results/toroidal_hybrid_m13g_shape_smoke/figures/toroidal_hybrid_lcfs_3d.png`;
- `results/toroidal_hybrid_m13g_shape_smoke/figures/toroidal_hybrid_cross_sections.png`;
- `results/toroidal_hybrid_m13g_shape_smoke/figures/wout/toroidal_stellarator_mirror_hybrid_VMEC_3Dplot.png`;
- `results/toroidal_hybrid_m13g_shape_smoke/figures/wout/toroidal_stellarator_mirror_hybrid_poloidal_plot.png`.

The LCFS plot is nonblank and shows sharpened corner localization.  The cross
section plot cleanly separates mirror-like side arcs from stellarator-like
corner arcs.

### How it was tested

Focused tests:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/test_toroidal_hybrid.py \
  tests/mirror/test_mirror_free_boundary.py -q
```

Result: `49 passed in 4.19s`.

Lint and format:

```bash
python -m ruff check \
  vmec_jax/toroidal_hybrid.py \
  examples/toroidal_stellarator_mirror_hybrid.py \
  examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  tests/test_toroidal_hybrid.py \
  tests/mirror/test_mirror_free_boundary.py
python -m ruff format --check \
  vmec_jax/toroidal_hybrid.py \
  examples/toroidal_stellarator_mirror_hybrid.py \
  examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  tests/test_toroidal_hybrid.py \
  tests/mirror/test_mirror_free_boundary.py
```

Result: all checks passed.

### File structure and best-practice notes

- Geometry controls stay in `vmec_jax/toroidal_hybrid.py`, the single source
  module for VMEC-compatible toroidal hybrid boundary generation.
- Root examples expose parameters through CLI flags and record them in
  JSON/CSV, so validation runs are reproducible.
- No generated figures, WOUT files, or caches were staged; outputs remain under
  ignored `results/`.
- The change remains compatible with the ordinary toroidal VMEC/JAX
  fixed-boundary solver and its existing solver modes.

### Best next steps

1. Commit and push M13g shaping controls.
2. Run a low-cost convergence grid for the sharpened `side_power=2`,
   `corner_power=2`, `ntor=10` case and compare residual history with the
   default shape.
3. Add a parameter-scan helper for side/corner powers and amplitudes that
   scores positivity, fit error, best `fsq`, and mean iota.
4. Recheck the PR CI state after the coverage-fix commits have had time to
   start and finish.

### Completion percentages after M92

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `86%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `88%`.
- I/O schema and docs: `94%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `68%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `49%`.
- ESSOS circular-coil mirror beta scan: `53%`.
- PR merge readiness overall: `92%`.

### User input needed

No user input is needed.

---

## 93. 2026-06-18 M13g low-cost sharpened-shape convergence comparison

This evidence run compared the default toroidal hybrid shape against the first
sharpened side/corner shape from section 92 using the same short fixed-boundary
solver budget.

### Steps taken

- Ran the default `mpol=5`, `ntor=4` toroidal hybrid row.
- Ran the sharpened `mpol=5`, `ntor=10` row with:
  - `side_minor_modulation=0.16`;
  - `side_elongation=0.35`;
  - `side_power=2.0`;
  - `corner_amplitude=0.025`;
  - `corner_power=2.0`.
- Used the ordinary toroidal VMEC/JAX fixed-boundary path with:
  - `solver_mode=parity`;
  - `use_scan=False`;
  - `ns=7`;
  - `NITER_ARRAY=30`;
  - `max_iter=20`;
  - `ftol=1e-9`.
- Wrote ignored convergence, residual-history, and profile plots for both rows.

### Results obtained

Commands:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_hybrid_m13g_default_compare \
  --ns-array 7 \
  --mode-pairs 5:4 \
  --ntheta-fit 32 \
  --nzeta-fit 32 \
  --niter 30 \
  --ftol 1e-9 \
  --run-solve \
  --max-iter 20 \
  --solver-mode parity \
  --no-use-scan
```

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_hybrid_m13g_sharp_compare \
  --ns-array 7 \
  --mode-pairs 5:10 \
  --ntheta-fit 32 \
  --nzeta-fit 32 \
  --niter 30 \
  --ftol 1e-9 \
  --run-solve \
  --max-iter 20 \
  --solver-mode parity \
  --no-use-scan \
  --side-minor-modulation 0.16 \
  --side-elongation 0.35 \
  --side-power 2.0 \
  --corner-amplitude 0.025 \
  --corner-power 2.0
```

Comparison:

| case | best `fsq` | final `fsq` | best iter | mean iota | magnetic well | runtime |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| default `5:4` | `1.471854259816e-05` | `1.839147093094e-05` | `18` | `7.629462473006e-03` | `-4.182573686373e-02` | `8.81 s` |
| sharpened `5:10` | `1.180571456258e-05` | `1.558342244087e-05` | `18` | `3.085941451021e-03` | `-5.392764202769e-02` | `8.71 s` |

The sharpened shape has a slightly lower best and final `fsq` over the same
short budget, but it changes the physics profile noticeably: mean iota is lower
and the magnetic-well proxy is more negative.  This is useful as a controlled
geometry knob, not yet an optimized target.

Generated ignored plot paths:

- `results/toroidal_hybrid_m13g_default_compare/figures/toroidal_hybrid_fsq_history.png`;
- `results/toroidal_hybrid_m13g_default_compare/figures/toroidal_hybrid_profiles.png`;
- `results/toroidal_hybrid_m13g_sharp_compare/figures/toroidal_hybrid_fsq_history.png`;
- `results/toroidal_hybrid_m13g_sharp_compare/figures/toroidal_hybrid_profiles.png`.

The sharpened residual plot was visually checked and rendered correctly.

### How it was tested

The commands above ran the solver and plot paths end to end.  No source files
changed for this evidence run.

### File structure and best-practice notes

- Evidence stays in `plan_mirror.md`; generated JSON/CSV/WOUT/PNG files remain
  ignored under `results/`.
- The comparison uses the existing convergence runner rather than adding a new
  workflow prematurely.

### Best next steps

1. Commit and push this evidence log.
2. Add a small parameter-scan helper around the existing convergence runner so
   M13g can compare side/corner powers and amplitudes without manual command
   duplication.
3. Keep scan outputs ignored and summarize only compact metrics in the plan.
4. Recheck PR CI state after the coverage-fix/shaping commits have had time to
   complete.

### Completion percentages after M93

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `86%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `88%`.
- I/O schema and docs: `94%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `68%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `51%`.
- ESSOS circular-coil mirror beta scan: `53%`.
- PR merge readiness overall: `92%`.

### User input needed

No user input is needed.

---

## 94. 2026-06-18 M13g shape-case scan helper

This tranche made the M13g side/corner parameter scan reproducible from one
command instead of repeated manual convergence-runner invocations.

### Steps taken

- Added `--shape-cases` to
  `examples/toroidal_stellarator_mirror_hybrid_convergence.py`.
- Added two initial shape presets:
  - `default`: the current baseline shape;
  - `sharp`: the section 92 sharpened side/corner shape.
- Added `shape_case` to row names and CSV output so multi-shape runs do not
  overwrite case directories.
- Added tests for:
  - preset parsing;
  - unknown preset rejection;
  - no-solve `default,sharp` scan output;
  - exact low-mode fit for both presets with `mpol=5`, `ntor=10`.
- Updated `examples/mirror/README.md` with the new option and the `ntor`
  caveat for exact sharpened-preset fits.

### Results obtained

Shape-case smoke command:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_hybrid_m13g_shape_cases_smoke \
  --ns-array 7 \
  --mode-pairs 5:10 \
  --ntheta-fit 32 \
  --nzeta-fit 32 \
  --shape-cases default,sharp
```

Rows:

| case | shape | fit error | side power | corner power | `RBC` | `ZBS` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `default_ns007_mpol05_ntor10` | `default` | `6.661338147751e-16` | `1.0` | `1.0` | `8` | `8` |
| `sharp_ns007_mpol05_ntor10` | `sharp` | `8.881784197001e-16` | `2.0` | `2.0` | `12` | `14` |

Generated ignored plot:

- `results/toroidal_hybrid_m13g_shape_cases_smoke/figures/toroidal_hybrid_convergence.png`.

### How it was tested

Focused tests:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/test_toroidal_hybrid.py \
  tests/mirror/test_mirror_free_boundary.py -q
```

Result: `50 passed in 4.89s`.

Lint, format, and whitespace:

```bash
python -m ruff check \
  examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  tests/test_toroidal_hybrid.py
python -m ruff format --check \
  examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  tests/test_toroidal_hybrid.py
git diff --check
```

Result: all checks passed.

### File structure and best-practice notes

- The scan helper extends the existing convergence runner instead of adding a
  separate script.
- Case directories include the shape preset name, avoiding accidental
  overwrites in multi-shape runs.
- Shape presets are small dictionaries next to the parser helpers, keeping the
  source file simple and easy to extend.
- Generated scan products remain ignored under `results/`.

### Best next steps

1. Commit and push the shape-case scan helper.
2. Run a short solved `--shape-cases default,sharp` scan with plots to produce
   one table and one residual plot for both presets together.
3. If the sharpened row remains slightly better, add a VMEC2000 parity row for
   the sharpened preset at `ns=7`, `mpol=5`, `ntor=10`.
4. Recheck PR CI once GitHub starts reporting checks for the latest head.

### Completion percentages after M94

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `86%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `88%`.
- I/O schema and docs: `94%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `68%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `54%`.
- ESSOS circular-coil mirror beta scan: `53%`.
- PR merge readiness overall: `92%`.

### User input needed

No user input is needed.

---

## 95. 2026-06-18 M13g solved shape-case scan and sharpened VMEC2000 parity

This tranche used the new `--shape-cases` helper for a solved default-vs-sharp
comparison, then ran a VMEC2000 parity smoke for the sharpened preset.

### Steps taken

- Ran a short solved scan with `--shape-cases default,sharp`.
- Used the same `ns=7`, `mpol=5`, `ntor=10`, `max_iter=20` parity-mode
  controls for both rows.
- Visually checked the combined residual-history plot.
- Ran a sharpened-preset VMEC2000 parity smoke with local
  `/Users/rogeriojorge/bin/xvmec2000`.
- Visually checked the residual-component parity plot.
- Rechecked PR CI status after the last push.

### Results obtained

Solved shape-case scan command:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_hybrid_m13g_shape_cases_solved \
  --ns-array 7 \
  --mode-pairs 5:10 \
  --ntheta-fit 32 \
  --nzeta-fit 32 \
  --shape-cases default,sharp \
  --niter 30 \
  --ftol 1e-9 \
  --run-solve \
  --max-iter 20 \
  --solver-mode parity \
  --no-use-scan
```

Solved comparison:

| case | initial `fsq` | best `fsq` | final `fsq` | mean iota | magnetic well |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `default_ns007_mpol05_ntor10` | `4.322385032595e-04` | `1.892679128454e-05` | `1.892679128454e-05` | `7.696236376067e-03` | `-3.944608070785e-02` |
| `sharp_ns007_mpol05_ntor10` | `3.030663121330e-04` | `1.180571456258e-05` | `1.558342244087e-05` | `3.085941451021e-03` | `-5.392764202769e-02` |

The combined residual plot rendered correctly and showed the sharp trace below
the default trace over the short run:

- `results/toroidal_hybrid_m13g_shape_cases_solved/figures/toroidal_hybrid_fsq_history.png`.

Sharpened VMEC2000 parity command:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_hybrid_m13g_sharp_vmec2000_parity \
  --ns-array 7 \
  --mode-pairs 5:10 \
  --ntheta-fit 32 \
  --nzeta-fit 32 \
  --shape-cases sharp \
  --niter 30 \
  --ftol 1e-9 \
  --run-solve \
  --max-iter 20 \
  --solver-mode parity \
  --no-use-scan \
  --run-vmec2000 \
  --vmec2000-exec /Users/rogeriojorge/bin/xvmec2000 \
  --vmec2000-timeout-s 120
```

VMEC2000 returned code `0`.

| metric | VMEC/JAX | VMEC2000 |
| :--- | ---: | ---: |
| initial `fsq` | `3.030663121330e-04` | `5.736000000000e-02` |
| final/best `fsq` | `1.558342244087e-05` | `3.743000000000e-03` |
| final `fsqr` | `7.523614811853e-06` | `1.827575564773e-03` |
| final `fsqz` | `4.952314560279e-06` | `1.299063949455e-03` |
| final `fsql` | `3.107493068743e-06` | `6.132958046633e-04` |
| mean iota | `3.085941451021e-03` | `2.705787605846e-03` |

The parity component plot rendered correctly:

- `results/toroidal_hybrid_m13g_sharp_vmec2000_parity/figures/toroidal_hybrid_parity_components.png`.

Interpretation is consistent with earlier sections: VMEC/JAX and VMEC2000 are
still not starting from equivalent residual states, but the sharpened input is
valid for both codes and gives close low-iota behavior over this short smoke.

CI status at this checkpoint:

- Latest head: `239e1430cbdcf64ed60aa88453a1053ea62ca2fa`.
- PR remains draft.
- `Parity Manifest Smoke (dry-run)` passed.
- Most remaining CI jobs were queued/pending, so no new failure was available
  to fix yet.

### How it was tested

Source/test checks before these evidence runs:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/test_toroidal_hybrid.py \
  tests/mirror/test_mirror_free_boundary.py -q
python -m ruff check \
  examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  tests/test_toroidal_hybrid.py
python -m ruff format --check \
  examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  tests/test_toroidal_hybrid.py
git diff --check
```

Result: `50 passed`; lint, format, and whitespace checks passed.

The solved scan and VMEC2000 command above exercised the example, plot, WOUT,
and VMEC2000-wrapper paths end to end.

### File structure and best-practice notes

- Evidence remains in the plan; generated products remain under ignored
  `results/`.
- The parity rows use the same convergence runner, CSV schema, and plot helpers
  as the default shape, so future shape scans are table-driven.

### Best next steps

1. Commit and push this evidence log.
2. When CI finishes, fix any concrete failures.
3. Continue M13g with a small scan over `corner_amplitude` and `side_power`
   using the shape-case framework.
4. Start designing the initialization-matched parity fixture before claiming
   strict VMEC2000 residual parity.

### Completion percentages after M95

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `86%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `88%`.
- I/O schema and docs: `94%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `68%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `57%`.
- ESSOS circular-coil mirror beta scan: `53%`.
- PR merge readiness overall: `92%`.

### User input needed

No user input is needed.

---

## 96. 2026-06-18 M13g corner-amplitude scan for sharpened side powers

This evidence run scanned corner amplitude at fixed sharpened side/corner
localization to understand the residual/iota tradeoff before adding more shape
presets.

### Steps taken

- Ran three short solved rows with:
  - `side_minor_modulation=0.16`;
  - `side_elongation=0.35`;
  - `side_power=2.0`;
  - `corner_power=2.0`;
  - `corner_amplitude` in `{0.015, 0.025, 0.035}`.
- Used `ns=7`, `mpol=5`, `ntor=10`, `max_iter=15`, `NITER_ARRAY=25`,
  `solver_mode=parity`, and `use_scan=False`.
- Checked the residual-history plot for the strongest `corner_amplitude=0.035`
  row.
- Rechecked PR CI; latest jobs were still pending, so no failure was available
  to fix.

### Results obtained

Commands used the same pattern, varying only `--corner-amplitude` and `--outdir`:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_hybrid_m13g_corner_amp_00XX \
  --ns-array 7 \
  --mode-pairs 5:10 \
  --ntheta-fit 32 \
  --nzeta-fit 32 \
  --niter 25 \
  --ftol 1e-9 \
  --run-solve \
  --max-iter 15 \
  --solver-mode parity \
  --no-use-scan \
  --side-minor-modulation 0.16 \
  --side-elongation 0.35 \
  --side-power 2.0 \
  --corner-amplitude 0.0XX \
  --corner-power 2.0
```

Scan table:

| corner amplitude | initial `fsq` | best `fsq` | final `fsq` | best iter | mean iota | magnetic well |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.015` | `5.940877991923e-04` | `4.821015678261e-05` | `5.749739682032e-05` | `13` | `1.124256826212e-03` | `-5.829802922677e-02` |
| `0.025` | `8.288639198192e-04` | `7.023760294194e-05` | `7.955666513867e-05` | `13` | `3.155372064713e-03` | `-5.640921636285e-02` |
| `0.035` | `1.607742077392e-03` | `1.548779377703e-04` | `1.548779377703e-04` | `14` | `6.020835412209e-03` | `-5.802764977093e-02` |

The stronger corner amplitude raises mean iota, but also increases the initial
and best residual in this short budget.  The `0.015` row is easiest for the
solver, while `0.025` remains a reasonable compromise for visible corner
stellarator shaping and low residual.  The `0.035` row is still decreasing but
is less attractive as a default sharpened preset.

Generated ignored plot checked visually:

- `results/toroidal_hybrid_m13g_corner_amp_0035/figures/toroidal_hybrid_fsq_history.png`.

### How it was tested

The commands above exercised the convergence runner, WOUT writing, and plot
generation for all three rows.  No source files changed for this evidence run.

### File structure and best-practice notes

- Results stay under ignored `results/`.
- The plan stores only compact numerical evidence and plot references.
- No new preset was added yet; evidence suggests keeping `corner_amplitude=0.025`
  as the current sharp preset until a target iota/residual tradeoff is chosen.

### Best next steps

1. Commit and push this evidence log.
2. When CI finishes, fix any concrete failures.
3. Continue M13g with one side-power scan at `corner_amplitude=0.025`.
4. Start an initialization-matched VMEC2000 parity design note after the next
   scan, since current residual parity remains initialization limited.

### Completion percentages after M96

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `86%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `88%`.
- I/O schema and docs: `94%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `68%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `58%`.
- ESSOS circular-coil mirror beta scan: `53%`.
- PR merge readiness overall: `92%`.

### User input needed

No user input is needed.

---

## 97. 2026-06-18 M13g side-power scan at fixed corner amplitude

This evidence run scanned integer side-localization powers at fixed corner
amplitude to decide whether sharper mirror-side localization is helping the
toroidal hybrid fixture.

### Steps taken

- Ran three short solved rows with:
  - `side_minor_modulation=0.16`;
  - `side_elongation=0.35`;
  - `corner_amplitude=0.025`;
  - `corner_power=2.0`;
  - `side_power` in `{1.0, 2.0, 3.0}`.
- Used `ns=7`, `mpol=5`, `ntor=14`, `max_iter=15`, `NITER_ARRAY=25`,
  `solver_mode=parity`, and `use_scan=False`.
- Checked the residual-history plot for the sharpest `side_power=3` row.
- Rechecked PR CI; early jobs had started passing, with the main test/coverage
  jobs still pending.

### Results obtained

Commands used the same pattern, varying only `--side-power` and `--outdir`:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_hybrid_m13g_side_power_X \
  --ns-array 7 \
  --mode-pairs 5:14 \
  --ntheta-fit 32 \
  --nzeta-fit 32 \
  --niter 25 \
  --ftol 1e-9 \
  --run-solve \
  --max-iter 15 \
  --solver-mode parity \
  --no-use-scan \
  --side-minor-modulation 0.16 \
  --side-elongation 0.35 \
  --side-power X \
  --corner-amplitude 0.025 \
  --corner-power 2.0
```

Scan table:

| side power | fit error | initial `fsq` | best `fsq` | final `fsq` | best iter | mean iota | magnetic well |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1.0` | `4.440892098501e-16` | `8.340589921272e-04` | `7.503096893488e-05` | `8.147623070621e-05` | `13` | `3.181644631676e-03` | `-5.008247661877e-02` |
| `2.0` | `8.881784197001e-16` | `9.246691962930e-04` | `8.749615941543e-05` | `9.300821628263e-05` | `13` | `3.166542338166e-03` | `-5.670267292936e-02` |
| `3.0` | `8.881784197001e-16` | `1.085636359630e-03` | `1.091580926128e-04` | `1.091580926128e-04` | `14` | `3.083387738702e-03` | `-6.178935866879e-02` |

Sharper side localization makes the magnetic-well proxy more negative but
increases the residual and does not improve mean iota in this short scan.
`side_power=1` is easiest for the solver, while `side_power=2` remains useful
for visibly sharper side/corner separation.  `side_power=3` is not attractive
as a default without a stronger physics target.

Generated ignored plot checked visually:

- `results/toroidal_hybrid_m13g_side_power_3/figures/toroidal_hybrid_fsq_history.png`.

CI status at this checkpoint:

- `Console Script Smoke`: passed.
- `Parity Manifest Smoke (dry-run)`: passed.
- `Build (wheel/sdist + docs)`: passed.
- Main fast-test and coverage jobs: pending.

### How it was tested

The commands above exercised the convergence runner, WOUT writing, and plot
generation for all three rows.  No source files changed for this evidence run.

### File structure and best-practice notes

- Results remain under ignored `results/`.
- The plan records compact metrics only.
- The current `sharp` preset remains `side_power=2`, `corner_amplitude=0.025`
  because it gives visible localization without the larger residual penalty of
  `side_power=3` or `corner_amplitude=0.035`.

### Best next steps

1. Commit and push this side-power scan evidence.
2. Wait for CI only if a concrete failure appears; otherwise continue work.
3. Start an initialization-matched parity design note for toroidal hybrids.
4. Then return to implementation lanes outside M13g: differentiable
   solved-state API design and ESSOS beta-scan completion.

### Completion percentages after M97

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `86%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `88%`.
- I/O schema and docs: `94%`.
- Differentiable solved-state API: `20%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `68%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `60%`.
- ESSOS circular-coil mirror beta scan: `53%`.
- PR merge readiness overall: `92%`.

### User input needed

No user input is needed.

---

## 98. 2026-06-18 stopped-work audit, CI checkpoint, and finite completion plan

This tranche restarted from the stopped midpoint, checked the PR state, and
converted the next work from open-ended scans into a finite milestone order.
It also made the toroidal-hybrid parity rows explicitly label their
initialization policies, so current VMEC/JAX-versus-VMEC2000 comparisons are
less likely to be read as matched-initial-state residual parity.

### Steps taken

- Checked the local tree and PR state:
  - branch: `codex/mirror-geometry`;
  - draft PR: `#21`, `[codex] Add mirror geometry infrastructure`;
  - local tree clean except ignored caches, docs build output, and `results/`.
- Checked CI with `gh pr checks 21 --watch=false`.
  - The latest run had no actionable failure at this checkpoint.
  - `Docs (full guide)` had passed.
  - Most fast-test, coverage, build, console-smoke, and parity-smoke jobs were
    still pending.
- Reviewed the changed-file footprint against `origin/main`.
  - Mirror code is isolated under `vmec_jax/mirror/`.
  - Repo-root mirror and toroidal-hybrid examples live under `examples/`.
  - Mirror docs live under `docs/mirror/` plus the root example README.
  - Tests are concentrated in `tests/mirror/` plus focused public API and
    toroidal-hybrid tests.
- Reviewed existing differentiability and solver infrastructure:
  - `vmec_jax/implicit.py` already provides custom-VJP fixed-boundary solved
    state wrappers, including a VMEC-residual path.
  - `vmec_jax/discrete_adjoint.py` and `solve.py` already support the
    discrete-adjoint trace/replay lane.
  - `run_fixed_boundary` already has a fast CLI/API path with host-update and
    performance-mode controls.
  - The mirror fixed-boundary API is separate and currently host-oriented; it
    should reuse the same differentiability policy instead of adding a third
    adjoint architecture.
- Added explicit toroidal-hybrid convergence row fields:
  - `initialization_policy`;
  - `vmec2000_initialization_policy`.
- Updated tests and docs so JSON/CSV output and user-facing documentation name
  the policy labels.

### Results obtained

The runner now writes:

- `initialization_policy = vmec_jax_default_input_boundary`;
- `vmec2000_initialization_policy = vmec2000_default_input_boundary`.

These labels are deliberately modest.  They document the current state:
VMEC/JAX and VMEC2000 are run from the same generated input file, but not yet
from a proven identical raw residual state.  Therefore:

- mean-iota and solved-profile agreement remain useful regression signals;
- strict `fsq` component parity remains an initialization-matched future gate.

### Finite completion plan from here

1. CI gate and PR hygiene:
   - check CI periodically, not continuously;
   - fix concrete failures immediately when they appear;
   - keep ignored `results/`, `docs/_build/`, and caches out of commits.
2. M13h toroidal-hybrid matched-parity lane:
   - keep the current policy labels;
   - add a raw-initialization audit fixture that writes enough initial-state
     metadata to compare VMEC/JAX and VMEC2000 residuals honestly;
   - only then add matched-initial-state parity rows if the source paths expose
     a defensible shared initialization.
3. M10 differentiable solved-state lane:
   - reuse `vmec_jax/implicit.py` and `discrete_adjoint.py`;
   - expose only stable public helpers through `vmec_jax/api.py`;
   - benchmark unrolled, implicit/custom-VJP, and discrete-adjoint modes on
     small fixed-boundary cases before promoting a mirror differentiable API;
   - keep fast CLI solves host/performance-oriented and separate from
     differentiable research APIs.
4. M8/M9 mirror fixed-boundary production lane:
   - continue reducing residual-Newton complexity only where it removes real
     duplication;
   - finish finite-current benchmark rows with residual decomposition,
     field-line pitch, `|B|`, iota-like open-field diagnostics, and magnetic
     well proxy plots;
   - retain dense/block-dense references as low-resolution truth checks.
5. M12 ESSOS/free-boundary circular-coil lane:
   - keep the example lightweight in the repo;
   - run ignored beta-scan outputs for 1%, 3%, and 10%;
   - compare LCFS pilot updates against external-coil normal-field and pressure
     merit before claiming a free-boundary solve.
6. M14 toroidal stellarator-mirror hybrid lane:
   - remain toroidal, with mirror-like side arcs and stellarator-like corners;
   - improve visualization and convergence studies with standard VMEC plots,
     cross sections, residual histories, iota, magnetic well, and VMEC2000
     comparison rows;
   - use GPU/office runs only for heavier convergence studies that would slow
     the local loop.
7. Documentation and simplification gate:
   - keep the public file structure shallow and domain-based;
   - move only stable user workflows into docs;
   - preserve pedagogical docstrings/comments and remove stale scaffolds before
     the final review.
8. Final merge-readiness gate:
   - run focused local tests after each tranche;
   - run full relevant test suites before marking the draft PR ready;
   - review generated figures manually from ignored result folders, not as
     committed artifacts unless a compressed reference is truly needed.

### How it was tested

```bash
JAX_ENABLE_X64=1 pytest tests/test_toroidal_hybrid.py -q
python -m ruff check examples/toroidal_stellarator_mirror_hybrid_convergence.py tests/test_toroidal_hybrid.py
python -m ruff format --check examples/toroidal_stellarator_mirror_hybrid_convergence.py tests/test_toroidal_hybrid.py
git diff --check
```

The section-ordering check is also part of this tranche because the plan is the
single active log.

### File structure and best-practice notes

- The source change stays in the existing convergence runner rather than
  creating a new reporting module.
- The test change extends the existing toroidal-hybrid smoke coverage.
- Documentation changes are limited to the existing mirror overview and example
  README.
- No result files or figures are committed.
- The differentiability plan explicitly reuses existing VMEC/JAX implicit and
  discrete-adjoint infrastructure instead of introducing another solver
  differentiation stack.

### Best next steps

1. Commit and push this schema/docs tranche.
2. Recheck CI once for concrete failures.
3. Start M13h by inspecting VMEC/JAX and VMEC2000 initial-state construction
   paths, then add the smallest raw-initialization audit artifact that can be
   tested cheaply.

### Completion percentages after M98

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `86%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `88%`.
- I/O schema and docs: `94%`.
- Differentiable solved-state API: `22%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `68%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `62%`.
- ESSOS circular-coil mirror beta scan: `53%`.
- PR merge readiness overall: `92%`.

### User input needed

No user input is needed.

---

## 99. 2026-06-18 M13h toroidal-hybrid axis-initialization audit

This tranche made the first M13h audit concrete by recording the VMEC/JAX
axis-initialization branch used by the toroidal-hybrid convergence runner.
This is a small but important parity field: VMEC/JAX can either start from the
raw input-axis/zero-axis state or infer a missing axis from the boundary, while
VMEC2000 parity claims require knowing which branch was used.

### Steps taken

- Inspected VMEC/JAX initialization paths in:
  - `vmec_jax/driver.py`;
  - `vmec_jax/init_guess.py`;
  - `vmec_jax/solve.py`;
  - `vmec_jax/vmec2000_exec.py`.
- Confirmed that:
  - `vmec2000_iter` parity mode defaults to the raw input-axis/zero-axis
    branch unless environment variables override it;
  - performance/default/accelerated modes may infer a missing axis from the
    boundary before the first solve iteration;
  - VMEC2000 parity data currently comes from the parsed first `threed1` row,
    not from a shared serialized initial state.
- Added `vmec_jax_axis_initialization_policy` to toroidal-hybrid convergence
  JSON/CSV rows.
- Added a small helper in the convergence runner that mirrors the driver branch
  for this specific `vmec2000_iter` call:
  - `raw_input_axis_or_zero`;
  - `boundary_inferred_missing_axis`.
- Updated tests to assert:
  - row JSON contains the field;
  - CSV contains the field;
  - parity mode maps to the raw branch;
  - accelerated/default behavior maps to boundary-inferred axis;
  - environment overrides are reflected.
- Updated mirror overview docs and the root mirror example README.

### Results obtained

The toroidal-hybrid convergence schema now records both:

- broad initialization policy labels;
- the concrete VMEC/JAX axis branch used for a row.

This narrows the parity ambiguity without adding heavy files or a new module.
It also preserves the fast CLI path: the audit is a string label computed from
runner/driver policy, not an extra solve.

### How it was tested

```bash
JAX_ENABLE_X64=1 pytest tests/test_toroidal_hybrid.py -q
python -m ruff check examples/toroidal_stellarator_mirror_hybrid_convergence.py tests/test_toroidal_hybrid.py
python -m ruff format --check examples/toroidal_stellarator_mirror_hybrid_convergence.py tests/test_toroidal_hybrid.py
git diff --check
```

Results:

- `20 passed` in `tests/test_toroidal_hybrid.py`.
- Ruff check passed.
- Ruff format check passed.
- `git diff --check` passed.
- Plan section-ordering check passed.

### File structure and best-practice notes

- The audit stays in the existing toroidal-hybrid convergence runner.
- No result files or figures are committed.
- The helper is intentionally narrow because it describes this runner's
  `run_fixed_boundary(..., solver="vmec2000_iter")` policy, not every possible
  driver call.
- Documentation remains concise and user-facing.

### Best next steps

1. Commit and push this M13h schema/audit tranche.
2. Recheck CI for any concrete failures.
3. Add a cheap raw residual audit next:
   - record whether VMEC/JAX initial residuals come from the solve history or
     an explicit first-step diagnostic;
   - compare those values to VMEC2000's first parsed `threed1` row in one
     low-resolution parity row;
   - avoid claiming matched-initial-state parity until the state itself is
     serialized or reconstructed on both sides.

### Completion percentages after M99

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `86%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `88%`.
- I/O schema and docs: `94%`.
- Differentiable solved-state API: `22%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `68%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `64%`.
- ESSOS circular-coil mirror beta scan: `53%`.
- PR merge readiness overall: `92%`.

### User input needed

No user input is needed.

---

## 100. 2026-06-18 M13h first-row residual source and ratio audit

This tranche finished the cheap raw-residual audit started in M99.  The
toroidal-hybrid convergence runner now labels where each initial residual came
from and, when both VMEC/JAX and VMEC2000 histories exist, reports first-row
VMEC/JAX-to-VMEC2000 residual ratios.

### Steps taken

- Added residual-source fields to the convergence JSON/CSV schema:
  - `initial_residual_source`;
  - `vmec2000_initial_residual_source`.
- Added first-row residual-ratio fields:
  - `initial_fsq_ratio_vmec2000`;
  - `initial_fsqr_ratio_vmec2000`;
  - `initial_fsqz_ratio_vmec2000`;
  - `initial_fsql_ratio_vmec2000`.
- Added a small `_safe_ratio` helper that returns `None` for missing,
  nonfinite, or zero-denominator values.
- Added `_attach_initial_residual_comparison(row)` and call it after the
  optional VMEC/JAX and VMEC2000 branches.
- Set the VMEC/JAX source to `vmec_jax_solve_history_first_row` when a solve
  history exists.
- Set the VMEC2000 source to `vmec2000_threed1_first_row` when parsed `threed1`
  rows exist.
- Updated tests for:
  - no-solve rows leaving source/ratio fields empty;
  - CSV serialization of empty source/ratio fields;
  - ratio computation and zero-denominator handling.
- Updated the mirror overview and root mirror example README.

### Results obtained

The current parity rows are now self-describing:

- VMEC/JAX first residuals are clearly identified as solve-history first rows.
- VMEC2000 first residuals are clearly identified as parsed `threed1` first
  rows.
- Ratios are produced only when both sides exist, so dry boundary-fit scans
  stay lightweight and unambiguous.

This still is not matched-initial-state parity.  It is a stronger audit trail
for explaining why strict residual parity is not yet claimed.

### How it was tested

```bash
JAX_ENABLE_X64=1 pytest tests/test_toroidal_hybrid.py -q
python -m ruff check examples/toroidal_stellarator_mirror_hybrid_convergence.py tests/test_toroidal_hybrid.py
python -m ruff format --check examples/toroidal_stellarator_mirror_hybrid_convergence.py tests/test_toroidal_hybrid.py
git diff --check
```

Results:

- `21 passed` in `tests/test_toroidal_hybrid.py`.
- Ruff check passed.
- Ruff format check passed.
- `git diff --check` passed.

### File structure and best-practice notes

- The change is schema-only plus helper tests inside the existing convergence
  runner/test file.
- No figures or result outputs are committed.
- The helper avoids adding a new module because the logic is runner-specific
  reporting, not shared solver infrastructure.

### Best next steps

1. Commit and push this M13h residual audit tranche.
2. Recheck CI for concrete failures.
3. Run one ignored low-resolution parity row with the new fields and log the
   resulting ratios in the plan.
4. Then move from audit fields to either:
   - a true matched-initialization fixture, if source paths expose enough state;
   - or the M10 differentiable solved-state public API cleanup if matched
     initialization would require intrusive VMEC2000 changes.

### Completion percentages after M100

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `86%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `88%`.
- I/O schema and docs: `95%`.
- Differentiable solved-state API: `22%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `68%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `65%`.
- ESSOS circular-coil mirror beta scan: `53%`.
- PR merge readiness overall: `92%`.

### User input needed

No user input is needed.

---

## 101. 2026-06-18 M13h low-resolution initial-residual audit row

This evidence run exercised the M99/M100 audit fields on one low-resolution
toroidal-hybrid VMEC/JAX-versus-VMEC2000 parity row.

### Steps taken

- Ran the toroidal-hybrid convergence runner with:
  - `ns=7`;
  - `mpol=5`, `ntor=10`;
  - `ntheta_fit=32`, `nzeta_fit=32`;
  - `NITER_ARRAY=25`;
  - VMEC/JAX `max_iter=12`;
  - `solver_mode=parity`;
  - `use_scan=False`;
  - local VMEC2000 executable `/Users/rogeriojorge/bin/xvmec2000`.
- Extracted the JSON/CSV audit fields.
- Visually checked:
  - residual-history plot;
  - final residual component parity plot;
  - iota / Mercier-well profile plot.

Command:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_hybrid_m13h_initial_residual_audit \
  --ns-array 7 \
  --mode-pairs 5:10 \
  --ntheta-fit 32 \
  --nzeta-fit 32 \
  --niter 25 \
  --ftol 1e-9 \
  --run-solve \
  --max-iter 12 \
  --solver-mode parity \
  --no-use-scan \
  --run-vmec2000 \
  --vmec2000-exec /Users/rogeriojorge/bin/xvmec2000 \
  --vmec2000-timeout-s 120
```

### Results obtained

The audit row reports:

| quantity | VMEC/JAX | VMEC2000 | ratio |
| --- | ---: | ---: | ---: |
| initial total `fsq` | `4.110992387738e-03` | `8.274000000000e-02` | `4.968567062773e-02` |
| initial `fsqr` | `2.237248194160e-03` | `5.290000000000e-02` | `4.229202635463e-02` |
| initial `fsqz` | `1.334014202350e-03` | `4.640000000000e-03` | `2.875030608512e-01` |
| initial `fsql` | `5.397299912286e-04` | `2.520000000000e-02` | `2.141785679479e-02` |

Other row metrics:

- `vmec_jax_axis_initialization_policy = raw_input_axis_or_zero`.
- `initial_residual_source = vmec_jax_solve_history_first_row`.
- `vmec2000_initial_residual_source = vmec2000_threed1_first_row`.
- VMEC/JAX best/final `fsq = 6.601249629690e-04`.
- VMEC2000 best/final `fsq = 7.770000000000e-03`.
- VMEC/JAX mean iota `7.435406701687e-03`.
- VMEC2000 mean iota `7.622815625446e-03`.
- Magnetic-well scalar from VMEC/JAX `-5.222353091036e-02`.

Interpretation:

- The raw-axis parity branch still does not produce matched initial residuals.
- VMEC/JAX starts at about `5%` of VMEC2000's parsed first-row total residual.
- Mean-iota agreement remains good for this short row, but strict force
  residual parity is still initialization-limited.

Generated ignored files checked:

- `results/toroidal_hybrid_m13h_initial_residual_audit/figures/toroidal_hybrid_fsq_history.png`.
- `results/toroidal_hybrid_m13h_initial_residual_audit/figures/toroidal_hybrid_parity_components.png`.
- `results/toroidal_hybrid_m13h_initial_residual_audit/figures/toroidal_hybrid_profiles.png`.

### How it was tested

- The command completed successfully.
- JSON and CSV both contain the new source/ratio fields.
- The generated plots rendered without blank or missing axes.
- No source files changed for this evidence run.

### File structure and best-practice notes

- Results remain under ignored `results/`.
- The plan records compact numerical evidence only.
- The runner schema now provides enough information to distinguish
  source-of-residual comparisons from true matched-state parity.

### Best next steps

1. Commit and push this evidence log.
2. Recheck CI for concrete failures.
3. Start the next M13h design step:
   - inspect whether VMEC2000 can ingest a restart/initial-state artifact that
     VMEC/JAX can also write;
   - if not, document that strict VMEC2000 residual parity is a non-goal for
     this PR and pivot to validated VMEC/JAX-internal initialization studies.
4. Continue M10 differentiable solved-state cleanup once the parity limitation
   is documented.

### Completion percentages after M101

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `86%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `88%`.
- I/O schema and docs: `95%`.
- Differentiable solved-state API: `22%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `68%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `66%`.
- ESSOS circular-coil mirror beta scan: `53%`.
- PR merge readiness overall: `92%`.

### User input needed

No user input is needed.

---

## 102. 2026-06-18 M13i direct-initial residual audit and VMEC2000 reset probe

This tranche corrected the toroidal-hybrid residual-parity interpretation from
"solve-history first row" to a true pre-iteration direct residual diagnostic.

### Steps taken

- Rechecked CI for PR #21:
  - build and console smoke were passing;
  - fast-test and docs jobs were still pending, with no concrete failure yet.
- Inspected VMEC2000 reset support in the local STELLOPT source:
  - `xvmec2000 input.ext reset=wout_file.nc` is parsed by
    `Sources/TimeStep/vmec.f`;
  - `runvmec.f` skips to the final radial grid when `reset_file_name` is
    present;
  - `initialize_radial.f` calls `load_xc_from_wout`;
  - `load_xc_from_wout.f` maps WOUT `rmnc`, `zmns`, and `lmns` into the
    internal `xc` arrays.
- Ran ignored reset probes:
  - VMEC/JAX raw-axis initial WOUT reset;
  - VMEC/JAX boundary-inferred initial WOUT reset;
  - deliberately modified reset WOUT;
  - VMEC2000 final-WOUT reset.
- Found that VMEC2000's reset mechanism is real:
  - a deliberately modified reset WOUT changes the run and can crash;
  - a VMEC2000 final-WOUT reset changes the early residual trace.
- Found that VMEC/JAX initial WOUT resets reproduce VMEC2000's default trace
  because the fixed-boundary initialization collapses to the same radial
  profile after VMEC2000's setup, not because the reset argument is ignored.
- Computed VMEC/JAX residual scalars directly on initial states:
  - raw-axis parity initial residual is enormous and is not the VMEC2000-like
    branch;
  - boundary-inferred default/accelerated initial residual matches VMEC2000's
    first `threed1` row.
- Added new convergence-runner fields:
  - `direct_initial_residual_requested`;
  - `direct_initial_residual_source`;
  - `direct_initial_axis_initialization_policy`;
  - `direct_initial_fsq`;
  - `direct_initial_fsqr`;
  - `direct_initial_fsqz`;
  - `direct_initial_fsql`;
  - `direct_initial_*_ratio_vmec2000`;
  - `direct_initial_error`.
- Added `--direct-initial-residual` / `--no-direct-initial-residual`.
- Renamed the solve-history source string to
  `vmec_jax_solve_history_first_stored_row`.
- Updated the residual-history plot to show the VMEC/JAX direct initial value
  as a pre-iteration marker at iteration `-1`.
- Updated mirror docs and example README to distinguish:
  - direct pre-iteration VMEC/JAX residuals;
  - first stored VMEC/JAX solve-history rows;
  - VMEC2000 first parsed `threed1` rows.

### Results obtained

Direct initial residual parity for the low-resolution accelerated row:

| quantity | VMEC/JAX direct initial | VMEC2000 first `threed1` | ratio |
| --- | ---: | ---: | ---: |
| total `fsq` | `8.285359474768e-02` | `8.274000000000e-02` | `1.001372912106` |
| `fsqr` | `5.296232415964e-02` | `5.290000000000e-02` | `1.001178150466` |
| `fsqz` | `4.703146208488e-03` | `4.640000000000e-03` | `1.013609096657` |
| `fsql` | `2.518812437955e-02` | `2.520000000000e-02` | `0.999528745220` |

The same run reports:

- `direct_initial_residual_source =
  vmec_jax_initial_guess_residual_scalars`;
- `direct_initial_axis_initialization_policy =
  boundary_inferred_missing_axis`;
- `initial_residual_source = vmec_jax_solve_history_first_stored_row`;
- first stored VMEC/JAX solve-history `fsq = 7.503003199772e-02`;
- VMEC/JAX best `fsq = 1.116468167248e-02`;
- VMEC2000 best/final `fsq = 7.770000000000e-03`;
- VMEC/JAX mean iota `1.280634031113e-02`;
- VMEC2000 mean iota `7.622815625446e-03`.

Interpretation:

- The earlier M101 ratio of about `0.05` compared different quantities:
  VMEC/JAX's first stored solve-history row against VMEC2000's first
  `threed1` row.
- For the normal boundary-inferred initialization, VMEC/JAX direct residual
  scalars now match VMEC2000's first row to about `0.14%` in total `fsq`.
- Strict residual parity should therefore use `direct_initial_*` fields for
  initialization checks and `*_history` fields for iteration behavior.
- The remaining mismatch is in solver trajectory and final convergence, not in
  the direct initial force diagnostic.

Generated ignored artifacts checked:

- `results/toroidal_hybrid_m13i_direct_initial_audit/toroidal_stellarator_mirror_hybrid_convergence.json`.
- `results/toroidal_hybrid_m13i_direct_initial_audit/toroidal_stellarator_mirror_hybrid_convergence.csv`.
- `results/toroidal_hybrid_m13i_direct_initial_audit/figures/toroidal_hybrid_fsq_history.png`.
- `results/toroidal_hybrid_m13i_direct_initial_audit/figures/toroidal_hybrid_convergence.png`.
- `results/toroidal_hybrid_m13i_direct_initial_audit/figures/toroidal_hybrid_parity_components.png`.
- `results/toroidal_hybrid_m13i_direct_initial_audit/figures/toroidal_hybrid_profiles.png`.

### How it was tested

Commands run:

```bash
JAX_ENABLE_X64=1 pytest tests/test_toroidal_hybrid.py -q
python -m ruff check examples/toroidal_stellarator_mirror_hybrid_convergence.py tests/test_toroidal_hybrid.py
python -m ruff format --check examples/toroidal_stellarator_mirror_hybrid_convergence.py tests/test_toroidal_hybrid.py
git diff --check
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_hybrid_m13i_direct_initial_audit \
  --ns-array 7 \
  --mode-pairs 5:10 \
  --ntheta-fit 32 \
  --nzeta-fit 32 \
  --niter 25 \
  --ftol 1e-9 \
  --run-solve \
  --max-iter 3 \
  --solver-mode accelerated \
  --no-use-scan \
  --run-vmec2000 \
  --vmec2000-exec /Users/rogeriojorge/bin/xvmec2000 \
  --vmec2000-timeout-s 120
```

Results:

- `22 passed` in `tests/test_toroidal_hybrid.py`.
- Ruff check passed.
- Ruff format check passed.
- `git diff --check` passed.
- The audit command completed successfully.
- The residual-history, convergence, final-component, and profile plots rendered
  correctly; the residual-history plot now includes the direct-initial marker.

### File structure and best-practice notes

- The source change stays in the root-level toroidal-hybrid convergence runner
  because these are reporting semantics, not reusable solver kernels.
- Tests remain in `tests/test_toroidal_hybrid.py`, including a mocked helper
  test so CI does not need a real VMEC2000 executable for this schema path.
- Documentation updates live in `docs/mirror/overview.rst` and
  `examples/mirror/README.md`.
- All generated benchmark artifacts remain ignored under `results/`.
- No committed figures or bulky outputs were added.

### Best next steps

1. Commit and push M13i.
2. Recheck PR #21 CI later for concrete failures, but do not block on pending
   jobs.
3. Move to solver-trajectory parity:
   - compare VMEC/JAX and VMEC2000 step/update controls after the matched
     direct initial residual;
   - focus on why accelerated VMEC/JAX reaches a different final state over the
     first few iterations.
4. Then continue the finite completion plan:
   - M10 differentiable solved-state API cleanup;
   - M11 mirror-Boozer-like diagnostics;
   - M12 free-boundary LCFS path;
   - M13 toroidal stellarator-mirror hybrid convergence;
   - M16 ESSOS circular-coil beta scan.

### Completion percentages after M102

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `87%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `88%`.
- I/O schema and docs: `96%`.
- Differentiable solved-state API: `22%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `68%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `70%`.
- ESSOS circular-coil mirror beta scan: `53%`.
- PR merge readiness overall: `93%`.

### User input needed

No user input is needed.

---

## 103. 2026-06-18 M13j full VMEC2000 residual-trajectory labels

### Steps taken

- Checked PR #21 CI before continuing:
  - completed jobs were passing;
  - remaining jobs were pending;
  - no failing CI log needed a fix at this checkpoint.
- Added an explicit ``--nstep`` option to
  ``examples/toroidal_stellarator_mirror_hybrid_convergence.py``.
- Wrote ``NSTEP`` into each generated toroidal-hybrid input so VMEC2000
  ``threed1`` traces can print every iteration with ``--nstep 1``.
- Added ``nstep`` to the CSV/JSON schema.
- Added ``iter_history`` to the JSON schema for VMEC/JAX residual histories.
- Made the residual-history plot show:
  - direct VMEC/JAX initial residual at iteration ``0``;
  - VMEC/JAX solve-history samples at their stored iteration labels, or at
    the physical one-based fallback ``1..N`` when the lightweight diagnostic
    path omits labels;
  - VMEC2000 ``threed1`` rows at their parsed iteration labels.
- Updated the mirror docs and example README to recommend ``--nstep 1`` for
  full VMEC/JAX versus VMEC2000 trajectory comparisons.

### Results obtained

Evidence run:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_hybrid_m13j_nstep1_trajectory_audit \
  --ns-array 7 \
  --mode-pairs 5:10 \
  --ntheta-fit 32 \
  --nzeta-fit 32 \
  --niter 25 \
  --nstep 1 \
  --ftol 1e-9 \
  --run-solve \
  --max-iter 8 \
  --solver-mode accelerated \
  --no-use-scan \
  --run-vmec2000 \
  --vmec2000-exec /Users/rogeriojorge/bin/xvmec2000 \
  --vmec2000-timeout-s 120
```

Key results:

- direct VMEC/JAX initial ``fsq = 8.285359474768e-02``;
- VMEC/JAX direct-initial / VMEC2000 first-row ``fsq`` ratio
  ``= 1.001372912106``;
- VMEC/JAX iteration labels ``[1, 2, 3, 4, 5, 6, 7, 8]``;
- VMEC/JAX ``fsq`` history:
  ``[4.602563219108e-03, 3.502378151427e-03, 1.497604222532e-03,
  3.563874272729e-03, 2.573380868047e-03, 1.700582286179e-03,
  2.035545202556e-03, 1.245978256565e-03]``;
- VMEC2000 first 10 ``fsq`` rows:
  ``[8.274000000000e-02, 1.514000000000e-02, 3.742000000000e-02,
  7.387000000000e-02, 2.497000000000e-02, 3.015000000000e-02,
  3.551000000000e-02, 2.897000000000e-02, 2.868000000000e-02,
  3.542000000000e-02]``;
- aligned VMEC/JAX / VMEC2000 ``fsq`` ratios for iterations 1 to 8:
  ``[5.562682159909e-02, 2.313327709000e-01, 4.002149178332e-02,
  4.824521825814e-02, 1.030589054084e-01, 5.640405592635e-02,
  5.732315411309e-02, 4.300925980548e-02]``;
- VMEC/JAX best/final ``fsq = 1.245978256565e-03`` in the 8-iteration run;
- VMEC2000 best/final ``fsq = 6.790000000000e-03`` /
  ``7.770000000000e-03`` in the 25-iteration printed trace.

Interpretation:

- The direct initial residual now agrees with VMEC2000 at the expected level
  for this low-resolution toroidal-hybrid audit.
- The remaining discrepancy is the solver trajectory after startup:
  accelerated VMEC/JAX immediately moves to much lower force residual than
  VMEC2000's printed first iterations.
- Next parity work should compare update controls, preconditioner usage,
  step acceptance, and lightweight-history timing, not boundary fitting.

Generated ignored artifacts checked:

- ``results/toroidal_hybrid_m13j_nstep1_trajectory_audit/toroidal_stellarator_mirror_hybrid_convergence.json``.
- ``results/toroidal_hybrid_m13j_nstep1_trajectory_audit/toroidal_stellarator_mirror_hybrid_convergence.csv``.
- ``results/toroidal_hybrid_m13j_nstep1_trajectory_audit/figures/toroidal_hybrid_fsq_history.png``.
- ``results/toroidal_hybrid_m13j_nstep1_trajectory_audit/figures/toroidal_hybrid_convergence.png``.
- ``results/toroidal_hybrid_m13j_nstep1_trajectory_audit/figures/toroidal_hybrid_parity_components.png``.
- ``results/toroidal_hybrid_m13j_nstep1_trajectory_audit/figures/toroidal_hybrid_profiles.png``.

The figure files are small ignored PNGs, not committed output.

### How it was tested

Commands run:

```bash
JAX_ENABLE_X64=1 pytest tests/test_toroidal_hybrid.py -q
python -m ruff check examples/toroidal_stellarator_mirror_hybrid_convergence.py tests/test_toroidal_hybrid.py
python -m ruff format --check examples/toroidal_stellarator_mirror_hybrid_convergence.py tests/test_toroidal_hybrid.py
git diff --check
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_hybrid_m13j_nstep1_trajectory_audit \
  --ns-array 7 \
  --mode-pairs 5:10 \
  --ntheta-fit 32 \
  --nzeta-fit 32 \
  --niter 25 \
  --nstep 1 \
  --ftol 1e-9 \
  --run-solve \
  --max-iter 8 \
  --solver-mode accelerated \
  --no-use-scan \
  --run-vmec2000 \
  --vmec2000-exec /Users/rogeriojorge/bin/xvmec2000 \
  --vmec2000-timeout-s 120
```

Results:

- `22 passed` in ``tests/test_toroidal_hybrid.py``.
- Ruff check passed.
- Ruff format check passed.
- ``git diff --check`` passed.
- The ``NSTEP=1`` audit run completed successfully.
- The refreshed residual-history plot rendered with aligned iteration labels.

### File structure and best-practice notes

- The ``--nstep`` and history-label changes remain inside the root-level
  convergence example because they are benchmark/reporting controls.
- The schema assertions stay in ``tests/test_toroidal_hybrid.py`` so the
  example contract is checked without committing generated output.
- Documentation updates stay in ``docs/mirror/overview.rst`` and
  ``examples/mirror/README.md``.
- All evidence files remain under ignored ``results/`` directories; no bulky
  benchmark outputs were added to the repository.

### Finite completion plan from here

1. Commit and push M13j.
2. Recheck PR #21 CI only for concrete failures; do not idle on pending jobs.
3. M13k: instrument the toroidal-hybrid VMEC/JAX run with solver-step
   diagnostics that explain the post-initial trajectory gap:
   - stage mode;
   - preconditioner/update path;
   - step acceptance;
   - restart/axis-reset reason;
   - history timing.
4. M13l: add a parity-mode trajectory audit beside the accelerated audit and
   decide which mismatch is a true regression versus an intended accelerated
   CLI improvement.
5. M10: finish a small differentiable solved-state API:
   - keep CLI performance paths non-differentiable when useful;
   - expose differentiable residual/objective hooks for JAX workflows;
   - document when to use implicit/adjoint/custom-VJP approaches instead of
     tracing the whole CLI solve.
6. M11: finish mirror-Boozer-like diagnostics and plots for ``|B|``, pitch,
   iota-like quantities, magnetic well, and force residual trends.
7. M12/M16: finish the free-boundary circular-coil/ESSOS lane:
   - keep coil/input generation light;
   - scan beta ``1%``, ``3%``, and ``10%``;
   - compare LCFS/on-axis/off-axis fields against analytic circular-loop
     checks where applicable.
8. M13: finish the toroidal stellarator-mirror hybrid lane:
   - mirror sections on the sides;
   - stellarator shaping in the toroidal corners;
   - up-down symmetry and one-field-period repetition controls;
   - fixed-boundary convergence benchmarks and plots.
9. Final PR-readiness pass:
   - simplify source files where recent scaffolding can be folded down;
   - remove duplicated helpers;
   - tighten docstrings/comments;
   - run focused tests plus one broader fast-test slice;
   - keep PR #21 draft until all lanes are ready for review.

### Best next steps

1. Commit and push the current M13j tranche.
2. Inspect solver diagnostics available from ``run_fixed_boundary`` and
   ``solve.py`` for the low-resolution toroidal hybrid case.
3. Add the smallest reporting hook needed to distinguish VMEC/JAX history
   timing from true update/preconditioner differences.

### Completion percentages after M103

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `87%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `91%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `88%`.
- I/O schema and docs: `96%`.
- Differentiable solved-state API: `22%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `68%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `73%`.
- ESSOS circular-coil mirror beta scan: `53%`.
- PR merge readiness overall: `93%`.

### User input needed

No user input is needed.

---

## 104. 2026-06-18 M13k full VMEC/JAX step diagnostics for toroidal hybrid audits

### Steps taken

- Added a public ``run_fixed_boundary(..., light_history=...)`` override:
  - ``None`` keeps the existing fast/light policy;
  - ``True`` forces compact histories;
  - ``False`` forces full per-step solver histories.
- Propagated the override through device rerouting, CLI staged helpers,
  parity fallbacks, finish paths, and the scan WOUT corrector.
- Added ``--full-solver-diagnostics`` to the root-level toroidal-hybrid
  convergence example.
- Added compact JSON/CSV fields for:
  - light-history mode;
  - resume-state mode;
  - multigrid stage modes, budgets, and offsets;
  - terminal step iteration labels;
  - step status and restart reason counts;
  - effective time step;
  - update RMS;
  - trial/current residual ratio;
  - bcovar update flags.
- Added ``toroidal_hybrid_step_diagnostics.png`` for the full-history audit
  path.
- Corrected component best-residual bookkeeping so component ``best_*`` values
  use the best residual array index rather than the one-based/VMEC iteration
  label.
- Updated docs to pair ``--nstep 1`` with ``--full-solver-diagnostics`` for
  trajectory audits.

### Results obtained

Evidence run:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_hybrid_m13k_full_step_diagnostics \
  --ns-array 7 \
  --mode-pairs 5:10 \
  --ntheta-fit 32 \
  --nzeta-fit 32 \
  --niter 25 \
  --nstep 1 \
  --ftol 1e-9 \
  --run-solve \
  --max-iter 8 \
  --solver-mode accelerated \
  --no-use-scan \
  --full-solver-diagnostics \
  --run-vmec2000 \
  --vmec2000-exec /Users/rogeriojorge/bin/xvmec2000 \
  --vmec2000-timeout-s 120
```

Key results:

- ``full_solver_diagnostics = True``.
- ``diagnostic_light_history = False``.
- ``diagnostic_resume_state_mode = minimal``.
- ``diagnostic_stage_modes = ["accelerated"]``.
- ``diagnostic_stage_niter = [8]``.
- ``diagnostic_step_history_size = 8``.
- ``diagnostic_step_iter_history = [1, 2, 3, 4, 5, 6, 7, 8]``.
- ``diagnostic_step_status_counts = {"momentum": 8}``.
- ``diagnostic_restart_reason_counts = {"none": 8}``.
- ``diagnostic_bcovar_updates = 1`` and the first step updated bcovar.
- ``diagnostic_final_dt_eff = 9.000000000000e-01``.
- ``diagnostic_max_update_rms = 2.738820461439e-04``.
- ``diagnostic_final_update_rms = 1.189436400498e-04``.
- ``diagnostic_w_try_ratio_history`` remained all ``1.0`` in this momentum
  path.

Interpretation:

- The accelerated low-resolution toroidal-hybrid run is not diverging because
  of restarts, backtracking, or unstable large updates.
- The trajectory difference after the matched direct-initial residual is a
  solver-policy difference: the accelerated path takes small momentum updates
  from a boundary-inferred, minimal-resume policy and rapidly reaches lower
  residuals than the VMEC2000 printed trajectory.
- The next parity audit should run the same full-diagnostic export in
  ``solver_mode=parity`` and compare raw-axis initialization plus conservative
  update behavior.

Generated ignored artifacts checked:

- ``results/toroidal_hybrid_m13k_full_step_diagnostics/toroidal_stellarator_mirror_hybrid_convergence.json``.
- ``results/toroidal_hybrid_m13k_full_step_diagnostics/toroidal_stellarator_mirror_hybrid_convergence.csv``.
- ``results/toroidal_hybrid_m13k_full_step_diagnostics/figures/toroidal_hybrid_fsq_history.png``.
- ``results/toroidal_hybrid_m13k_full_step_diagnostics/figures/toroidal_hybrid_step_diagnostics.png``.
- ``results/toroidal_hybrid_m13k_full_step_diagnostics/figures/toroidal_hybrid_convergence.png``.
- ``results/toroidal_hybrid_m13k_full_step_diagnostics/figures/toroidal_hybrid_profiles.png``.
- ``results/toroidal_hybrid_m13k_full_step_diagnostics/figures/toroidal_hybrid_parity_components.png``.

The new step-diagnostics PNG rendered correctly and is about 89 KB. No result
files are committed.

### How it was tested

Commands run:

```bash
python -m ruff check vmec_jax/driver.py examples/toroidal_stellarator_mirror_hybrid_convergence.py tests/test_toroidal_hybrid.py tests/test_driver_run_wave8_coverage.py
python -m ruff format vmec_jax/driver.py examples/toroidal_stellarator_mirror_hybrid_convergence.py tests/test_toroidal_hybrid.py tests/test_driver_run_wave8_coverage.py
python -m ruff format --check vmec_jax/driver.py examples/toroidal_stellarator_mirror_hybrid_convergence.py tests/test_toroidal_hybrid.py tests/test_driver_run_wave8_coverage.py
JAX_ENABLE_X64=1 pytest tests/test_toroidal_hybrid.py tests/test_driver_run_wave8_coverage.py::test_direct_coil_free_boundary_quiet_performance_path_uses_light_history -q
git diff --check
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_hybrid_m13k_full_step_diagnostics \
  --ns-array 7 \
  --mode-pairs 5:10 \
  --ntheta-fit 32 \
  --nzeta-fit 32 \
  --niter 25 \
  --nstep 1 \
  --ftol 1e-9 \
  --run-solve \
  --max-iter 8 \
  --solver-mode accelerated \
  --no-use-scan \
  --full-solver-diagnostics \
  --run-vmec2000 \
  --vmec2000-exec /Users/rogeriojorge/bin/xvmec2000 \
  --vmec2000-timeout-s 120
```

Results:

- `23 passed` in the focused pytest run.
- Ruff check passed.
- Ruff format check passed after formatting.
- ``git diff --check`` passed.
- The full-diagnostics audit completed successfully.
- The residual-history, step-diagnostics, profile, convergence, and parity
  component plots rendered.

### File structure and best-practice notes

- The driver change is an optional keyword routed to an existing solver option;
  it does not change default behavior or solver mathematics.
- Full histories are opt-in from the example, so normal CLI/API fast paths keep
  their current memory and runtime profile.
- The example exports scalar CSV summaries and JSON arrays only in ignored
  result files.
- Tests cover the public driver override and the example diagnostic schema.
- Documentation lives with the existing mirror overview and mirror examples
  README.

### Best next steps

1. Commit and push M13k.
2. Run the same ``--full-solver-diagnostics --nstep 1`` audit in
   ``solver_mode=parity`` to distinguish raw-axis parity behavior from
   accelerated policy behavior.
3. If parity mode still starts from an unexpectedly low solve-history row,
   inspect the initial bad-Jacobian/reset path and compare it against VMEC2000
   ``guess_axis``/``restart`` behavior.
4. Then continue to the differentiable solved-state API and mirror diagnostics
   lanes from the finite plan.

### Completion percentages after M104

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `87%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `88%`.
- I/O schema and docs: `96%`.
- Differentiable solved-state API: `22%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `68%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `76%`.
- ESSOS circular-coil mirror beta scan: `53%`.
- PR merge readiness overall: `93%`.

### User input needed

No user input is needed.

---

## 105. 2026-06-18 M13l parity trajectory audit and direct-vs-history ratios

### Steps taken

- Ran the full-diagnostics toroidal-hybrid trajectory audit in
  ``solver_mode=parity``.
- Added total residual ratio fields to the convergence example:
  - ``initial_fsq_ratio_direct_initial``;
  - ``vmec2000_initial_fsq_ratio_direct_initial``.
- Added tests for the new ratio fields and CSV serialization.
- Updated the mirror examples README to explain that these fields separate
  pre-iteration direct residuals from first stored solve-history rows.

### Results obtained

Parity evidence run:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_hybrid_m13l_parity_full_step_diagnostics \
  --ns-array 7 \
  --mode-pairs 5:10 \
  --ntheta-fit 32 \
  --nzeta-fit 32 \
  --niter 25 \
  --nstep 1 \
  --ftol 1e-9 \
  --run-solve \
  --max-iter 8 \
  --solver-mode parity \
  --no-use-scan \
  --full-solver-diagnostics \
  --run-vmec2000 \
  --vmec2000-exec /Users/rogeriojorge/bin/xvmec2000 \
  --vmec2000-timeout-s 120
```

Comparison to the refreshed accelerated audit:

- accelerated:
  - direct VMEC/JAX initial ``fsq = 8.285359474768e-02``;
  - VMEC2000 first-row ``fsq = 8.274000000000e-02``;
  - direct/VMEC2000 initial ratio ``= 1.001372912106``;
  - first stored VMEC/JAX history ``fsq = 4.602563219108e-03``;
  - first-history/direct-initial ratio ``= 5.555055556883e-02``;
  - first-history/VMEC2000-first-row ratio ``= 5.562682159909e-02``.
- parity:
  - raw-axis direct VMEC/JAX initial ``fsq = 7.655972553630e+08``;
  - VMEC2000 first-row ``fsq = 8.274000000000e-02``;
  - direct/VMEC2000 initial ratio ``= 9.253048771610e+09``;
  - first stored VMEC/JAX history ``fsq = 4.429242451231e-03``;
  - first-history/direct-initial ratio ``= 5.785342646155e-12``;
  - first-history/VMEC2000-first-row ratio ``= 5.353205766535e-02``.

Step diagnostics:

- accelerated:
  - ``diagnostic_step_status_counts = {"momentum": 8}``;
  - ``diagnostic_restart_reason_counts = {"none": 8}``;
  - max update RMS ``= 2.738820461439e-04``.
- parity:
  - ``diagnostic_step_status_counts = {"momentum": 8}``;
  - ``diagnostic_restart_reason_counts = {"none": 8}``;
  - max update RMS ``= 9.332066713808e-04``.

Interpretation:

- Accelerated direct-initial residual parity with VMEC2000 is good for this
  low-resolution audit.
- The first stored VMEC/JAX solve-history row is not the same diagnostic as
  VMEC2000's first ``threed1`` row; it is after solver startup/update.
- Parity mode with the raw input axis exposes an enormous pre-iteration
  residual, then quickly reaches a small first stored history row without
  terminal restart reasons in the exported history.
- The next numerical audit should inspect the very first bad-Jacobian/axis
  handling path inside ``solve_fixed_boundary_residual_iter`` and compare that
  against VMEC2000 ``guess_axis`` behavior, because the mismatch is before the
  terminal step-control histories now exported by the example.

Generated ignored artifacts checked:

- ``results/toroidal_hybrid_m13l_parity_full_step_diagnostics/toroidal_stellarator_mirror_hybrid_convergence.json``.
- ``results/toroidal_hybrid_m13l_parity_full_step_diagnostics/toroidal_stellarator_mirror_hybrid_convergence.csv``.
- ``results/toroidal_hybrid_m13l_parity_full_step_diagnostics/figures/toroidal_hybrid_fsq_history.png``.
- ``results/toroidal_hybrid_m13l_parity_full_step_diagnostics/figures/toroidal_hybrid_step_diagnostics.png``.

### How it was tested

Commands run:

```bash
python -m ruff check examples/toroidal_stellarator_mirror_hybrid_convergence.py tests/test_toroidal_hybrid.py
python -m ruff format --check examples/toroidal_stellarator_mirror_hybrid_convergence.py tests/test_toroidal_hybrid.py
JAX_ENABLE_X64=1 pytest tests/test_toroidal_hybrid.py -q
git diff --check
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_hybrid_m13k_full_step_diagnostics \
  --ns-array 7 \
  --mode-pairs 5:10 \
  --ntheta-fit 32 \
  --nzeta-fit 32 \
  --niter 25 \
  --nstep 1 \
  --ftol 1e-9 \
  --run-solve \
  --max-iter 8 \
  --solver-mode accelerated \
  --no-use-scan \
  --full-solver-diagnostics \
  --run-vmec2000 \
  --vmec2000-exec /Users/rogeriojorge/bin/xvmec2000 \
  --vmec2000-timeout-s 120
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_hybrid_m13l_parity_full_step_diagnostics \
  --ns-array 7 \
  --mode-pairs 5:10 \
  --ntheta-fit 32 \
  --nzeta-fit 32 \
  --niter 25 \
  --nstep 1 \
  --ftol 1e-9 \
  --run-solve \
  --max-iter 8 \
  --solver-mode parity \
  --no-use-scan \
  --full-solver-diagnostics \
  --run-vmec2000 \
  --vmec2000-exec /Users/rogeriojorge/bin/xvmec2000 \
  --vmec2000-timeout-s 120
```

Results:

- `22 passed` in ``tests/test_toroidal_hybrid.py``.
- Ruff check passed.
- Ruff format check passed.
- ``git diff --check`` passed.
- Both accelerated and parity evidence runs completed successfully.
- The residual-history and step-diagnostics figures rendered.

### File structure and best-practice notes

- The new ratio fields are scalar reporting fields in the root-level
  convergence example.
- Tests stay in ``tests/test_toroidal_hybrid.py`` and do not require VMEC2000.
- Evidence artifacts remain ignored under ``results/``.
- No source-level solver behavior changed in this tranche.

### Best next steps

1. Commit and push M13l.
2. Inspect the first-iteration bad-Jacobian/axis reset path in
   ``solve_fixed_boundary_residual_iter`` to identify where raw-axis parity
   moves from the huge direct residual to the small first stored history row.
3. If the transition is intentional and VMEC-like, document the precise
   diagnostic timing difference; if it is not, add a targeted regression test
   and align the history/print sampling.
4. Continue with differentiable solved-state API cleanup after this
   toroidal-hybrid trajectory explanation is pinned down.

### Completion percentages after M105

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `87%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `88%`.
- I/O schema and docs: `96%`.
- Differentiable solved-state API: `22%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `68%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `78%`.
- ESSOS circular-coil mirror beta scan: `53%`.
- PR merge readiness overall: `94%`.

### User input needed

No user input is needed.

---

## 106. 2026-06-18 M13m raw VMEC-style trajectory diagnostics and CI check

### Steps taken

- Checked draft PR #21 on GitHub: all reported CI jobs on the pushed head
  passed; the manual/nightly full-physics job was skipped by design.
- Trimmed the interrupted local solver diff so ``vmec_jax/solve.py`` only
  carries setup-axis-reset diagnostics, not broad formatter churn.
- Added exported setup-axis-reset fields to the toroidal hybrid convergence
  example:
  - attempted/reset/bad-Jacobian/force-reset flags;
  - pre-reset fsq;
  - ``ptau`` and state-tau ranges;
  - setup-axis-reset exception text, when present.
- Added ``--cli-finish/--no-cli-finish`` to the root-level toroidal hybrid
  convergence example.  The default remains the faster VMEC/JAX CLI
  finish/fallback policy, while ``--no-cli-finish`` keeps the raw VMEC-style
  trajectory for VMEC2000 comparisons.
- Updated mirror docs and the mirror examples README to explain when
  ``--nstep 1 --full-solver-diagnostics --no-cli-finish`` is the correct
  parity mode.
- Render-checked the accelerated and parity no-finish residual-history plots.

### Results obtained

- CI state before this commit: clean on the pushed PR head.
- The previous low first-history residual mismatch was explained as a
  diagnostic-timing issue:
  - with CLI finish enabled, ``run.result.w_history`` can refer to the
    finish/fallback attempt, not the initial VMEC-style stage;
  - with ``--no-cli-finish``, accelerated VMEC/JAX matches VMEC2000 over the
    first eight printed iterations to about ``0.1%`` for this low-resolution
    audit;
  - parity mode exposes the raw-axis direct residual near ``7.66e8``, then the
    setup-axis reset brings the stored VMEC-style trajectory close to VMEC2000.
- The no-finish evidence plots rendered:
  - ``results/toroidal_hybrid_m13n_no_cli_finish_accelerated/figures/toroidal_hybrid_fsq_history.png``;
  - ``results/toroidal_hybrid_m13n_no_cli_finish_parity/figures/toroidal_hybrid_fsq_history.png``.
- Generated artifacts remain ignored under ``results/``; no output files or
  figures are tracked.

### How it was tested

Commands run:

```bash
gh pr view --json number,url,title,isDraft,headRefName,baseRefName,mergeStateStatus,statusCheckRollup
python -m ruff check vmec_jax/solve.py examples/toroidal_stellarator_mirror_hybrid_convergence.py tests/test_toroidal_hybrid.py
python -m ruff format --check examples/toroidal_stellarator_mirror_hybrid_convergence.py tests/test_toroidal_hybrid.py
JAX_ENABLE_X64=1 pytest tests/test_toroidal_hybrid.py -q
git diff --check
```

Results:

- GitHub CI reported success on all non-skipped jobs.
- Ruff lint passed for solver, example, and test files.
- Ruff format check passed for the changed example and test files.
- ``22 passed`` in ``tests/test_toroidal_hybrid.py``.
- ``git diff --check`` passed.
- ``ruff format --check vmec_jax/solve.py`` was intentionally not used as a
  gating check for this tranche because the existing large solver file is not
  formatter-clean without unrelated whole-file style movement; the source diff
  was kept narrow instead.

### File structure and best-practice notes

- Solver-only diagnostics live in ``vmec_jax/solve.py`` beside the setup-axis
  reset path that produces them.
- Public reporting and plotting remain in the root-level convergence example,
  keeping evidence generation outside the library API.
- Tests stay focused in ``tests/test_toroidal_hybrid.py`` and do not require a
  local VMEC2000 executable.
- Docs were updated in ``docs/mirror/overview.rst`` and
  ``examples/mirror/README.md``.
- The tracked repository stays light; evidence plots and CSV/JSON outputs are
  ignored under ``results/``.

### Best next steps

1. Commit and push M13m.
2. Continue M10 differentiable solved-state API cleanup with a small,
   documented return object for fixed-boundary solved states and residual
   diagnostics.
3. Continue M12/M16 free-boundary circular-coil/ESSOS lane by pinning the
   lightweight circular-coil LCFS example and beta-scan result schema.
4. Continue M14 toroidal stellarator-mirror hybrid work by adding the toroidal
   corner-stellarator / side-mirror geometry fixture to the finite plan after
   the current mirror lanes are stable.

### Completion percentages after M106

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `87%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `88%`.
- I/O schema and docs: `96%`.
- Differentiable solved-state API: `22%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `68%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `82%`.
- ESSOS circular-coil mirror beta scan: `53%`.
- PR merge readiness overall: `95%`.

### User input needed

No user input is needed.

---

## 107. 2026-06-18 M10 public solved-state summary and residual implicit export

### Steps taken

- Added ``FixedBoundarySolvedState`` and ``fixed_boundary_solved_state(run)``
  to the public driver layer.
- Added ``FixedBoundaryRun.solved_state`` as a compact scalar summary for
  completed fixed-boundary solves.
- Re-exported the residual-based implicit fixed-boundary solver through both
  public import surfaces:
  - ``vmec_jax.solve_fixed_boundary_state_implicit_vmec_residual``;
  - ``vmec_jax.api.solve_fixed_boundary_state_implicit_vmec_residual``.
- Updated the public API docs and quickstart to describe:
  - ``run.solved_state`` for lightweight convergence metadata;
  - the residual implicit path as the solver-consistent differentiable route;
  - the energy-objective implicit path as the smaller fixed-geometry route;
  - the CLI path as free to remain faster and non-differentiable.
- Added focused tests for the solved-state summary and public exports.

### Results obtained

- Optimization and analysis scripts now have a stable, small object containing:
  - final state;
  - total and component ``fsq`` values;
  - convergence flags;
  - ``ftol``;
  - solver mode and scan policy when reported;
  - ``signgs``.
- Large histories remain on ``run.result`` and are not copied into the summary.
- The research-grade residual implicit wrapper is now discoverable from the
  same public import surfaces as the rest of the user-facing API.
- No generated outputs or figures were added.

### How it was tested

Commands run:

```bash
python -m ruff check vmec_jax/driver.py vmec_jax/api.py vmec_jax/__init__.py tests/test_driver_implicit_wave11_coverage.py
python -m ruff format --check vmec_jax/driver.py vmec_jax/api.py vmec_jax/__init__.py tests/test_driver_implicit_wave11_coverage.py
JAX_ENABLE_X64=1 pytest tests/test_driver_implicit_wave11_coverage.py -q
python - <<'PY'
import vmec_jax as vj
import vmec_jax.api as api
print(vj.FixedBoundarySolvedState.__name__)
print(api.solve_fixed_boundary_state_implicit_vmec_residual.__name__)
PY
git diff --check
```

Results:

- Ruff lint passed.
- Ruff format check passed after formatting the changed test file.
- ``5 passed`` in ``tests/test_driver_implicit_wave11_coverage.py``.
- Public import smoke printed ``FixedBoundarySolvedState`` and
  ``solve_fixed_boundary_state_implicit_vmec_residual``.
- ``git diff --check`` passed.

### File structure and best-practice notes

- The solved-state view lives in ``vmec_jax/driver.py`` beside
  ``FixedBoundaryRun`` and existing residual helper functions.
- Public exports are limited to ``vmec_jax/__init__.py`` and
  ``vmec_jax/api.py``; no new module was introduced.
- Tests reuse the existing lightweight driver fixture and avoid full physics
  solves.
- Documentation changes are concise and live in the public API page and
  quickstart.

### Best next steps

1. Commit and push M10 public solved-state API cleanup.
2. Continue M12/M16 free-boundary circular-coil/ESSOS lane by pinning a
   lightweight beta-scan schema and example contract.
3. Continue M14 toroidal stellarator-mirror hybrid geometry with the toroidal
   side-mirror / corner-stellarator fixture after the current fixed-boundary
   and diagnostics lanes are stable.
4. Return to the residual implicit path with gradient finite-difference parity
   tests on a tiny fixed-boundary case before promoting it as production-grade.

### Completion percentages after M107

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `87%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `88%`.
- I/O schema and docs: `96%`.
- Differentiable solved-state API: `30%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `68%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `82%`.
- ESSOS circular-coil mirror beta scan: `53%`.
- PR merge readiness overall: `95%`.

### User input needed

No user input is needed.

---

## 108. 2026-06-18 M12/M16 circular-coil beta-scan metrics contract

### Steps taken

- Added explicit top-level status fields to the root-level
  ``examples/mirror_free_boundary_circular_coils.py`` metrics JSON:
  - ``workflow_status``;
  - ``free_boundary_solve_status``;
  - ``external_field_provider_kind``;
  - ``coil_format``;
  - ``beta_scan_requested_percent``;
  - fixed-boundary baseline counts;
  - LCFS pilot requested/row/accepted/skipped counts.
- Added per-beta-row LCFS pilot summaries:
  - ``lcfs_pilot_status``;
  - row/accepted/skipped counts;
  - final/best pilot merit;
  - final pilot pressure-balance RMS.
- Kept the setup JSON honest: it remains a setup-only scan file and tests now
  assert ``status == "setup_only_no_lcfs_solve"``.
- Updated mirror docs to describe the workflow/status fields and to keep the
  wording clear that the current path is an LCFS pilot, not a converged
  free-boundary equilibrium solve.
- Ran a plotted low-resolution evidence case under ignored ``results/``.

### Results obtained

- The default 1%, 3%, and 10% beta case contract is now explicit in metrics
  JSON and tested.
- Downstream benchmark scripts can distinguish:
  - setup-only runs;
  - fixed-boundary baseline runs;
  - LCFS pilot runs;
  - accepted, rejected, and skipped pilot steps.
- Evidence run:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/mirror_free_boundary_circular_coils.py \
  --outdir results/mirror/free_boundary_circular_coils_m108_plotted \
  --betas 1 \
  --ntheta 8 \
  --nxi 11 \
  --n-segments 64 \
  --run-fixed-boundary-baseline \
  --baseline-maxiter 0 \
  --run-lcfs-pilot \
  --lcfs-pilot-steps 1
```

Evidence metrics:

- ``workflow_status = "lcfs_pilot"``;
- ``free_boundary_solve_status = "lcfs_pilot_not_converged_free_boundary"``;
- ``beta_scan_requested_percent = [1.0]``;
- one baseline row and one accepted pilot row.

Rendered ignored plots included:

- ``free_boundary_circular_coils_geometry.png``;
- ``free_boundary_circular_coils_axis_bz.png``;
- ``free_boundary_circular_coils_boundary_bmag.png``;
- fixed-boundary and pilot-step ``|B|``, field-boundary, cross-section,
  Jacobian, residual, radial-diagnostic, and LCFS-diagnostic figures.

### How it was tested

Commands run:

```bash
python -m ruff check examples/mirror_free_boundary_circular_coils.py tests/mirror/test_mirror_examples.py tests/mirror/test_mirror_free_boundary.py
python -m ruff format --check examples/mirror_free_boundary_circular_coils.py tests/mirror/test_mirror_examples.py tests/mirror/test_mirror_free_boundary.py
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_free_boundary.py tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_example_runs_without_plots tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_strict_bnormal_guard_can_skip_pilot -q
git diff --check
```

Results:

- Ruff lint passed.
- Ruff format check passed after formatting the changed example.
- ``33 passed`` in the focused mirror free-boundary/example tests.
- ``git diff --check`` passed.
- Plotted evidence run completed and figures rendered.

### File structure and best-practice notes

- Metrics summarization stays inside the root example because these are
  example-run bookkeeping fields, not solver kernels.
- Free-boundary helper tests stay in ``tests/mirror/test_mirror_free_boundary.py``.
- Example subprocess/schema tests stay in ``tests/mirror/test_mirror_examples.py``.
- Docs were updated in ``examples/mirror/README.md`` and
  ``docs/mirror/overview.rst``.
- Results and figures remain ignored under ``results/`` and are not tracked.

### Best next steps

1. Commit and push M108.
2. Continue M12/M16 by adding a true circular-coil LCFS iteration loop that can
   run multiple accepted pilot steps per beta until merit stagnation or a user
   tolerance is reached.
3. Add a compact cross-beta plot comparing pressure-balance RMS, normal-field
   RMS, and merit for the 1%, 3%, and 10% cases.
4. Continue M14 toroidal side-mirror / corner-stellarator fixture after the
   free-boundary metrics contract remains stable under CI.

### Completion percentages after M108

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `87%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `88%`.
- I/O schema and docs: `97%`.
- Differentiable solved-state API: `30%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `72%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `82%`.
- ESSOS circular-coil mirror beta scan: `62%`.
- PR merge readiness overall: `95%`.

### User input needed

No user input is needed.

---

## 109. 2026-06-18 M16 circular-coil cross-beta LCFS summary plot

### Steps taken

- Added a cross-beta summary plot to
  ``examples/mirror_free_boundary_circular_coils.py``.
- The plot compares, by requested nominal beta:
  - side-boundary pressure-balance RMS;
  - external normal-field RMS;
  - LCFS merit;
  - baseline versus final accepted pilot values when pilot rows exist.
- Wired the plot into top-level ``metrics["figures"]["beta_scan_summary"]``
  only when plots are enabled and baseline rows exist.
- Updated mirror docs to mention the optional cross-beta LCFS metrics plot.
- Ran a plotted low-resolution 1%, 3%, and 10% beta evidence scan under
  ignored ``results/``.

### Results obtained

Evidence command:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/mirror_free_boundary_circular_coils.py \
  --outdir results/mirror/free_boundary_circular_coils_m109_beta_summary \
  --ntheta 8 \
  --nxi 11 \
  --n-segments 64 \
  --run-fixed-boundary-baseline \
  --baseline-maxiter 0 \
  --run-lcfs-pilot \
  --lcfs-pilot-steps 1
```

Evidence metrics:

- requested beta points: ``[1.0, 3.0, 10.0]``;
- three baseline rows;
- three accepted pilot rows;
- top-level ``beta_scan_summary`` figure path recorded.

The low-resolution zero-iteration evidence plot is intentionally flat across
beta because this pilot holds the baseline shape and edge pressure fixed.  That
visible flatness is useful: it shows the next physics step should run actual
finite-beta fixed-boundary iterations or a multi-step LCFS loop before treating
the beta trend as physical.

Rendered ignored plot:

- ``results/mirror/free_boundary_circular_coils_m109_beta_summary/figures/free_boundary_circular_coils_beta_scan_summary.png``.

### How it was tested

Commands run:

```bash
python -m ruff check examples/mirror_free_boundary_circular_coils.py tests/mirror/test_mirror_examples.py tests/mirror/test_mirror_free_boundary.py
python -m ruff format --check examples/mirror_free_boundary_circular_coils.py tests/mirror/test_mirror_examples.py tests/mirror/test_mirror_free_boundary.py
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_example_runs_without_plots tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_strict_bnormal_guard_can_skip_pilot -q
git diff --check
```

Results:

- Ruff lint passed.
- Ruff format check passed.
- ``2 passed`` in focused example tests.
- ``git diff --check`` passed.
- Plotted 1/3/10 beta evidence run completed and the summary figure rendered.

### File structure and best-practice notes

- The plot helper stays inside the root example because it summarizes example
  metrics, not reusable solver state.
- Generated plots remain ignored under ``results/``.
- Docs remain in ``examples/mirror/README.md`` and ``docs/mirror/overview.rst``.

### Best next steps

1. Commit and push M109.
2. Run finite-beta fixed-boundary baseline iterations for the 1%, 3%, and 10%
   cases and check whether the cross-beta plot develops a physical trend.
3. Add a multi-step LCFS pilot loop tolerance/stagnation criterion instead of
   a fixed number of requested pilot steps.
4. Continue M14 toroidal side-mirror / corner-stellarator fixture.

### Completion percentages after M109

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `87%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `89%`.
- I/O schema and docs: `97%`.
- Differentiable solved-state API: `30%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `74%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `82%`.
- ESSOS circular-coil mirror beta scan: `66%`.
- PR merge readiness overall: `95%`.

### User input needed

No user input is needed.

---

## 110. 2026-06-18 M14 toroidal hybrid orientation diagnostics

### Steps taken

The toroidal stellarator-mirror hybrid lane needed a sharper diagnostic for the
user's clarified geometry target: toroidal sides should remain mirror-like while
corner regions carry the stellarator rotation.  I added a cross-section
principal-axis orientation diagnostic and threaded it through the public API,
metrics, example plots, tests, and docs.

Concretely:

- added `toroidal_hybrid_cross_section_orientation(samples)` in
  `vmec_jax/toroidal_hybrid.py`;
- added metric fields for total, side-region, and corner-region orientation
  spans plus side/corner weight overlap;
- exported the helper through both `vmec_jax` lazy exports and
  `vmec_jax.api`;
- added `toroidal_hybrid_region_orientation.png` to the repo-root toroidal
  hybrid example;
- extended the focused toroidal hybrid test to require flat mirror-side
  orientation and nonzero corner orientation;
- updated the mirror README and docs overview.

### Results obtained

The evidence run used:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid.py \
  --outdir results/toroidal_hybrid_m110_orientation \
  --ntheta-fit 64 \
  --nzeta-fit 64 \
  --ntor 10
```

Observed diagnostic values:

- `cross_section_orientation_span`: `0.2127248186850288`;
- `side_orientation_span`: `1.5543122344752192e-15`;
- `corner_orientation_span`: `0.2127248186850288`;
- `side_corner_weight_overlap_max`: `0.25`.

This confirms the current fixture has essentially constant side orientation and
localized corner rotation, matching the clarified toroidal hybrid intent.

Rendered ignored plot:

- `results/toroidal_hybrid_m110_orientation/figures/toroidal_hybrid_region_orientation.png`.

### How it was tested

Commands run:

```bash
python -m ruff check vmec_jax/toroidal_hybrid.py vmec_jax/api.py vmec_jax/__init__.py examples/toroidal_stellarator_mirror_hybrid.py tests/test_toroidal_hybrid.py
python -m ruff format --check vmec_jax/toroidal_hybrid.py vmec_jax/api.py vmec_jax/__init__.py examples/toroidal_stellarator_mirror_hybrid.py tests/test_toroidal_hybrid.py
JAX_ENABLE_X64=1 pytest tests/test_toroidal_hybrid.py -q
git diff --check
```

Results:

- Ruff lint passed.
- Ruff format check passed.
- `22 passed` in `tests/test_toroidal_hybrid.py`.
- `git diff --check` passed.
- The plotted example run completed and the orientation figure rendered.

### File structure and best-practice notes

- The reusable geometry diagnostic stays in `vmec_jax/toroidal_hybrid.py`, next
  to the sampler and metrics it evaluates.
- Public API plumbing is limited to `vmec_jax/api.py` and `vmec_jax/__init__.py`.
- The plot remains example-local because it is an evidence/diagnostic plot, not
  a solver primitive.
- Tests remain focused in `tests/test_toroidal_hybrid.py`.
- Generated evidence remains under ignored `results/`.

### Best next steps

1. Commit and push M110.
2. Add the same orientation diagnostics to the toroidal hybrid convergence
   example so solver traces can be interpreted by region.
3. Run a finite fixed-boundary toroidal hybrid solve at low resolution and
   record residual, solved-state metrics, and orientation preservation.
4. Continue the free-boundary circular-coil lane with finite-beta baseline
   iterations and a tolerance/stagnation LCFS pilot loop.
5. Keep moving toward the differentiable solved-state/implicit-derivative lane
   once the residual solve is sufficiently benchmarked.

### Completion percentages after M110

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `87%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `89%`.
- I/O schema and docs: `97%`.
- Differentiable solved-state API: `30%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `74%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `86%`.
- ESSOS circular-coil mirror beta scan: `66%`.
- PR merge readiness overall: `95%`.

### User input needed

No user input is needed.

---

## 111. 2026-06-18 M14 toroidal hybrid valid-axis orientation audit

### Steps taken

The M110 orientation plot exposed a branch-sensitivity problem in the principal
axis diagnostic: at some corner-centered cross sections the covariance is nearly
isotropic, so the ellipse-axis angle is mathematically undefined.  I added
anisotropy diagnostics and changed the convergence runner to compare
principal-axis angles only where the axis is well-defined.

Concretely:

- added `toroidal_hybrid_cross_section_anisotropy(samples)` in
  `vmec_jax/toroidal_hybrid.py`;
- added orientation valid fractions, valid orientation spans, and anisotropy
  min/max metrics;
- exported the anisotropy helper through `vmec_jax` and `vmec_jax.api`;
- changed the toroidal hybrid convergence CSV/JSON to include target and fitted
  anisotropy/valid-axis fields;
- replaced the misleading raw corner-span convergence plot with a valid-axis
  max orientation-fit-error plot and valid-fraction overlay;
- updated the root toroidal hybrid orientation plot to mark undefined
  covariance-axis samples explicitly;
- updated tests and docs to prefer valid-axis fields over raw orientation spans.

### Results obtained

The no-solve convergence evidence run:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_hybrid_m111_orientation_convergence \
  --ns-array 7,9 \
  --mode-pairs 5:10 \
  --ntheta-fit 32 \
  --nzeta-fit 32 \
  --side-power 2.0 \
  --corner-power 2.0
```

Observed values for both rows:

- `max_boundary_fit_error`: `2.220446049250313e-15`;
- `max_orientation_fit_error`: `3.498198326819346e-13`;
- `orientation_fit_valid_fraction`: `0.9375`;
- `valid_corner_orientation_span`: about `2.26e-13`;
- `fitted_valid_corner_orientation_span`: about `3.83e-13`;
- `cross_section_anisotropy_min`: about `4.23e-18`;
- `cross_section_anisotropy_max`: about `1.25e-2`.

This corrects the M110 interpretation: the current toroidal fixture preserves a
flat, well-defined side orientation, but the apparent corner orientation span is
coming from nearly isotropic/undefined covariance-axis samples.  The next hybrid
geometry step should therefore implement a true rotating-ellipse corner fixture
rather than treating the current localized `m=2` perturbation as sufficient.

Rendered ignored plots:

- `results/toroidal_hybrid_m111_orientation_convergence/figures/toroidal_hybrid_orientation_preservation.png`;
- `results/toroidal_hybrid_m111_boundary_plot/figures/toroidal_hybrid_region_orientation.png`.

### How it was tested

Commands run:

```bash
python -m ruff check vmec_jax/toroidal_hybrid.py vmec_jax/api.py vmec_jax/__init__.py examples/toroidal_stellarator_mirror_hybrid.py examples/toroidal_stellarator_mirror_hybrid_convergence.py tests/test_toroidal_hybrid.py
python -m ruff format --check vmec_jax/toroidal_hybrid.py vmec_jax/api.py vmec_jax/__init__.py examples/toroidal_stellarator_mirror_hybrid.py examples/toroidal_stellarator_mirror_hybrid_convergence.py tests/test_toroidal_hybrid.py
JAX_ENABLE_X64=1 pytest tests/test_toroidal_hybrid.py -q
git diff --check
```

Results:

- Ruff lint passed.
- Ruff format check passed after formatting the convergence example.
- `22 passed` in `tests/test_toroidal_hybrid.py`.
- `git diff --check` passed.
- Both plotted evidence runs completed and the images rendered.

### File structure and best-practice notes

- Reusable orientation/anisotropy diagnostics stay in
  `vmec_jax/toroidal_hybrid.py`.
- Example CSV/JSON fields and plots stay in the root convergence example because
  they are run-report artifacts.
- Public exports are explicit in `vmec_jax/api.py` and lazy in
  `vmec_jax/__init__.py`.
- The tests focus on scientific semantics: valid-axis comparison must be tight,
  and undefined-axis fractions must be visible.
- Generated plots remain ignored under `results/`.

### Best next steps

1. Commit and push M111.
2. Implement the true toroidal rotating-ellipse corner fixture so valid-axis
   corner orientation changes are physical rather than branch artifacts.
3. Re-run the toroidal hybrid convergence example and require nonzero
   valid-corner orientation span for the upgraded fixture.
4. Then run a finite fixed-boundary toroidal hybrid solve at low resolution and
   record residual, solved-state metrics, and geometry preservation.
5. Continue the circular-coil free-boundary lane with finite-beta fixed-boundary
   baselines and an LCFS pilot tolerance/stagnation criterion.

### Completion percentages after M111

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `87%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `89%`.
- I/O schema and docs: `97%`.
- Differentiable solved-state API: `30%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `74%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `84%`.
- ESSOS circular-coil mirror beta scan: `66%`.
- PR merge readiness overall: `95%`.

### User input needed

No user input is needed.

---

## 112. 2026-06-18 M14 true toroidal rotating-ellipse corner fixture

### Steps taken

After M111 showed that the earlier corner orientation was branch-driven at
nearly isotropic sections, I upgraded the toroidal hybrid boundary itself.  The
fixture now has a finite-mode rotating ellipse in the stellarator corner
regions, while the mirror-side cores remain orientation-flat.

Concretely:

- added `corner_ellipticity` and `corner_rotation` parameters to
  `sample_toroidal_stellarator_mirror_hybrid_boundary`;
- implemented the corner as a localized elliptic axis split plus odd-in-`zeta`
  tilt, preserving stellarator symmetry and VMEC `RBC`/`ZBS` storage;
- kept `corner_amplitude` as an optional small `m=2` helical perturbation;
- tightened orientation metrics so “side” spans are measured in true side cores,
  not transition zones;
- exposed the new parameters in the root toroidal hybrid example and convergence
  runner;
- updated the `sharp` shape preset to use stronger corner ellipticity/rotation;
- updated tests to require nonzero valid-corner orientation span and exact
  roundtrip at `mpol:ntor = 5:20` on a `64 x 64` fit grid;
- updated docs to describe rotating-ellipse stellarator corners.

### Results obtained

The no-solve convergence evidence run:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_hybrid_m112_rotating_ellipse \
  --ns-array 7,9 \
  --mode-pairs 5:20 \
  --ntheta-fit 64 \
  --nzeta-fit 64 \
  --shape-cases default,sharp
```

Observed values:

- default rows:
  - `max_boundary_fit_error`: `3.9968028886505635e-15`;
  - `max_orientation_fit_error`: `6.461498003318411e-14`;
  - `orientation_fit_valid_fraction`: `1.0`;
  - `valid_side_orientation_span`: `0.0`;
  - `valid_corner_orientation_span`: `2.5463968425417773`;
  - `corner_ellipticity`: `0.18`;
  - `corner_rotation`: `0.35`;
- sharp rows:
  - `max_boundary_fit_error`: `3.3306690738754696e-15`;
  - `max_orientation_fit_error`: `1.6997854654217115e-14`;
  - `orientation_fit_valid_fraction`: `1.0`;
  - `valid_side_orientation_span`: `0.0`;
  - `valid_corner_orientation_span`: `2.4132542674384743`;
  - `corner_ellipticity`: `0.22`;
  - `corner_rotation`: `0.42`.

The root boundary plot now shows flat mirror-side cores and nonzero valid
corner orientation.  The previous undefined-axis markers are gone for the
default rotating-ellipse fixture because the covariance anisotropy is nonzero
across the sampled field period.

Rendered ignored plots:

- `results/toroidal_hybrid_m112_rotating_ellipse/figures/toroidal_hybrid_orientation_preservation.png`;
- `results/toroidal_hybrid_m112_boundary_plot/figures/toroidal_hybrid_region_orientation.png`.

### How it was tested

Commands run:

```bash
python -m ruff check vmec_jax/toroidal_hybrid.py examples/toroidal_stellarator_mirror_hybrid.py examples/toroidal_stellarator_mirror_hybrid_convergence.py tests/test_toroidal_hybrid.py
python -m ruff format --check vmec_jax/toroidal_hybrid.py examples/toroidal_stellarator_mirror_hybrid.py examples/toroidal_stellarator_mirror_hybrid_convergence.py tests/test_toroidal_hybrid.py
JAX_ENABLE_X64=1 pytest tests/test_toroidal_hybrid.py -q
git diff --check
```

Results:

- Ruff lint passed.
- Ruff format check passed.
- `25 passed` in `tests/test_toroidal_hybrid.py`.
- `git diff --check` passed.
- Plotted convergence and boundary evidence runs completed and rendered.

### File structure and best-practice notes

- The rotating-ellipse formula remains in `vmec_jax/toroidal_hybrid.py` because
  it is the reusable geometry fixture.
- Example parameters and CSV fields stay in the two root examples.
- Tests continue to use the public example path, not private fixtures, for the
  root/convergence output contracts.
- Generated evidence remains under ignored `results/`.

### Best next steps

1. Commit and push M112.
2. Run a low-resolution finite fixed-boundary solve on the upgraded toroidal
   hybrid fixture and record residual histories, profiles, and orientation
   preservation.
3. Compare the same input against local VMEC2000 with `--nstep 1`,
   `--full-solver-diagnostics`, and `--no-cli-finish`.
4. Continue the circular-coil free-boundary lane with finite-beta baselines and
   LCFS pilot tolerance/stagnation logic.
5. Resume differentiable solved-state work after the residual/geometry evidence
   is stable.

### Completion percentages after M112

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `87%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `89%`.
- I/O schema and docs: `97%`.
- Differentiable solved-state API: `30%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `74%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `90%`.
- ESSOS circular-coil mirror beta scan: `66%`.
- PR merge readiness overall: `95%`.

### User input needed

No user input is needed.

---

## 113. 2026-06-18 M14 toroidal hybrid fixed-boundary and VMEC2000 evidence

### Steps taken

With the rotating-ellipse corner fixture in place, I ran the first finite
fixed-boundary evidence cases and a local VMEC2000 comparison.  The residual
history plot also needed a small robustness fix: the raw-axis direct-initial
diagnostic can be intentionally enormous in parity mode, so the plot now marks
that value as off-scale instead of letting it compress the useful residual
history.

Concretely:

- ran a default rotating-ellipse toroidal hybrid VMEC/JAX solve with `ns=7`,
  `mpol:ntor=5:20`, `max_iter=25`, and `nstep=1`;
- ran the same low-resolution input in parity mode with `--no-use-scan`,
  `--no-cli-finish`, and `--run-vmec2000`;
- updated `_write_fsq_history_plot` so direct-initial outliers are plotted near
  the visible history and labeled with their true off-scale value;
- added a regression test for the off-scale direct-initial plot path.

### Results obtained

VMEC/JAX accelerated CLI-style 25-iteration evidence:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_hybrid_m113_lowres_solve25 \
  --ns-array 7 \
  --mode-pairs 5:20 \
  --ntheta-fit 64 \
  --nzeta-fit 64 \
  --shape-cases default \
  --run-solve \
  --max-iter 25 \
  --nstep 1
```

Observed:

- first stored `fsq`: `2.161189183188116e-4`;
- best/final `fsq`: `7.00103198199691e-6` at iteration `24`;
- reduction from first stored row: `30.87x`;
- not converged to `ftol=1e-9`;
- aspect: `5.565962975574549`;
- mean iota: `0.014328435636658836`;
- magnetic well proxy: `-0.10440717436433497`;
- valid side orientation span: `0.0`;
- valid corner orientation span: `2.5463968425417773`.

VMEC/JAX parity plus VMEC2000 comparison evidence:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_hybrid_m114_vmec2000_compare \
  --ns-array 7 \
  --mode-pairs 5:20 \
  --ntheta-fit 64 \
  --nzeta-fit 64 \
  --shape-cases default \
  --run-solve \
  --max-iter 25 \
  --nstep 1 \
  --solver-mode parity \
  --no-use-scan \
  --no-cli-finish \
  --run-vmec2000 \
  --vmec2000-timeout-s 120
```

Observed:

- VMEC/JAX first stored `fsq`: `0.10777007753805128`;
- VMEC2000 first `threed1` `fsq`: `0.1078`;
- first-row ratio VMEC/JAX / VMEC2000: `0.9997224261414774`;
- VMEC/JAX best/final `fsq` after 25 rows: `0.007967987810592103`;
- VMEC2000 best/final `fsq` after 80 rows: `6.116e-6`;
- VMEC2000 residual reduction: `17625.9x`;
- raw-axis direct-initial diagnostic: `1.0015553931025441e11`, intentionally
  off-scale relative to the first stored residual;
- VMEC2000 final component residuals:
  - `fsqr = 2.5839013890856746e-6`;
  - `fsqz = 2.6858120494979713e-6`;
  - `fsql = 8.458938582274209e-7`.

The comparison shows that the first stored VMEC/JAX parity residual matches the
VMEC2000 first row closely, but the current 25-row VMEC/JAX parity trajectory is
not yet matching VMEC2000's later convergence depth.  That makes the next
solver step concrete: inspect parity trajectory controls and/or run a longer
parity case before claiming fixed-boundary convergence parity for this hybrid.

Rendered ignored plots:

- `results/toroidal_hybrid_m113_lowres_solve25/figures/toroidal_hybrid_fsq_history.png`;
- `results/toroidal_hybrid_m113_lowres_solve25/figures/toroidal_hybrid_profiles.png`;
- `results/toroidal_hybrid_m114_vmec2000_compare/figures/toroidal_hybrid_fsq_history.png`;
- `results/toroidal_hybrid_m114_vmec2000_compare/figures/toroidal_hybrid_parity_components.png`.

### How it was tested

Commands run:

```bash
python -m ruff check examples/toroidal_stellarator_mirror_hybrid_convergence.py tests/test_toroidal_hybrid.py
python -m ruff format --check examples/toroidal_stellarator_mirror_hybrid_convergence.py tests/test_toroidal_hybrid.py
JAX_ENABLE_X64=1 pytest tests/test_toroidal_hybrid.py -q
git diff --check
```

Results:

- Ruff lint passed.
- Ruff format check passed.
- `26 passed` in `tests/test_toroidal_hybrid.py`.
- `git diff --check` passed.
- VMEC/JAX and VMEC2000 evidence runs completed and rendered.

### File structure and best-practice notes

- The plotting fix stays in the convergence example because it is report
  presentation logic.
- The new regression test uses a temporary plot path and leaves the repository
  clean.
- Solver code was not changed in this tranche; the evidence points to the next
  solver-parity investigation.
- Generated evidence remains under ignored `results/`.

### Best next steps

1. Commit and push M113.
2. Run a longer VMEC/JAX parity case on the same input and compare the full
   residual trajectory with VMEC2000.
3. If VMEC/JAX still stalls above VMEC2000, inspect the parity branch update
   controls, restart behavior, and bcovar update cadence for this high-`ntor`
   hybrid.
4. After parity behavior is understood, run the accelerated CLI path with a
   tighter convergence target and record the cost/residual tradeoff.
5. Continue circular-coil finite-beta/free-boundary work after the fixed-boundary
   hybrid evidence is stable.

### Completion percentages after M113

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `87%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `90%`.
- I/O schema and docs: `97%`.
- Differentiable solved-state API: `30%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `74%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `92%`.
- ESSOS circular-coil mirror beta scan: `66%`.
- PR merge readiness overall: `95%`.

### User input needed

No user input is needed.

---

## 114. 2026-06-18 M14 toroidal hybrid 80-row VMEC2000 parity check

### Steps taken

M113 showed that the 25-row VMEC/JAX parity run stopped above VMEC2000's
80-row residual depth.  I reran the exact same upgraded rotating-ellipse
toroidal hybrid input with an 80-row VMEC/JAX parity budget, still using
`--no-use-scan`, `--no-cli-finish`, `--nstep 1`, and `--run-vmec2000`.

### Results obtained

Command run:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_hybrid_m114_parity80 \
  --ns-array 7 \
  --mode-pairs 5:20 \
  --ntheta-fit 64 \
  --nzeta-fit 64 \
  --shape-cases default \
  --run-solve \
  --max-iter 80 \
  --nstep 1 \
  --solver-mode parity \
  --no-use-scan \
  --no-cli-finish \
  --run-vmec2000 \
  --vmec2000-timeout-s 120
```

Observed:

- VMEC/JAX first stored `fsq`: `0.10777007753805128`;
- VMEC2000 first `threed1` `fsq`: `0.1078`;
- first-row ratio VMEC/JAX / VMEC2000: `0.9997224261414774`;
- VMEC/JAX best/final `fsq`: `6.115607296842458e-6` at stored row `79`;
- VMEC2000 best/final `fsq`: `6.116e-6` at row `80`;
- VMEC/JAX residual reduction: `17622.14x`;
- VMEC2000 residual reduction: `17625.90x`;
- overlapping `fsq` history ratio range: about `0.99787` to `1.00217`;
- VMEC/JAX final residual components:
  - `fsqr = 2.583901389127433e-6`;
  - `fsqz = 2.685812049486272e-6`;
  - `fsql = 8.458938582287532e-7`;
- VMEC2000 final residual components:
  - `fsqr = 2.5839013890856746e-6`;
  - `fsqz = 2.6858120494979713e-6`;
  - `fsql = 8.458938582274209e-7`;
- VMEC/JAX mean iota: `0.014891808649244371`;
- VMEC2000 mean iota: `0.01489180864924554`.

The conclusion is positive: the upgraded toroidal hybrid fixture follows the
VMEC2000 residual trajectory and final residual components when given the same
80-row parity budget.  The M113 25-row gap was an iteration-budget difference,
not a solver divergence.

Rendered ignored plots:

- `results/toroidal_hybrid_m114_parity80/figures/toroidal_hybrid_fsq_history.png`;
- `results/toroidal_hybrid_m114_parity80/figures/toroidal_hybrid_parity_components.png`;
- `results/toroidal_hybrid_m114_parity80/figures/toroidal_hybrid_profiles.png`.

### How it was tested

This was a no-code evidence tranche.  The code at the previous commit had
already passed:

```bash
python -m ruff check examples/toroidal_stellarator_mirror_hybrid_convergence.py tests/test_toroidal_hybrid.py
python -m ruff format --check examples/toroidal_stellarator_mirror_hybrid_convergence.py tests/test_toroidal_hybrid.py
JAX_ENABLE_X64=1 pytest tests/test_toroidal_hybrid.py -q
git diff --check
```

The new evidence run completed with VMEC2000 return code `0`, parsed `80`
`threed1` rows, and produced the parity figures listed above.

### File structure and best-practice notes

- No source changes were needed.
- Evidence remains under ignored `results/`.
- The plan entry records the exact command and metrics so the parity result is
  reproducible without committing large output files.

### Best next steps

1. Commit and push the M114 plan evidence.
2. Run the accelerated CLI path on the same fixture with a tighter convergence
   target and compare runtime/residual against the parity path.
3. Add a compact CI-safe test or example assertion around the upgraded fixture's
   exact geometry/diagnostic contracts only, not the VMEC2000 executable run.
4. Continue the circular-coil finite-beta/free-boundary lane with finite-beta
   baselines and LCFS pilot tolerance/stagnation logic.
5. Resume differentiable solved-state work once the fixed-boundary evidence
   matrix is stable.

### Completion percentages after M114

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `87%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `90%`.
- I/O schema and docs: `97%`.
- Differentiable solved-state API: `30%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `74%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `94%`.
- ESSOS circular-coil mirror beta scan: `66%`.
- PR merge readiness overall: `95%`.

### User input needed

No user input is needed.

---

## 115. 2026-06-18 M14 toroidal hybrid accelerated solve evidence

### Steps taken

After validating VMEC/JAX parity against VMEC2000 for the upgraded rotating
ellipse fixture, I ran the accelerated CLI-style path on the same input to check
whether the production/default path reaches a deeper residual at similar cost.

### Results obtained

Command run:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir results/toroidal_hybrid_m115_accelerated80 \
  --ns-array 7 \
  --mode-pairs 5:20 \
  --ntheta-fit 64 \
  --nzeta-fit 64 \
  --shape-cases default \
  --run-solve \
  --max-iter 80 \
  --nstep 1 \
  --solver-mode accelerated \
  --use-scan
```

Observed:

- solver mode: `accelerated`;
- scan path: `True`;
- CLI finish/fallback policy: `True`;
- runtime recorded by the example: `6.118939124979079 s`;
- stored rows: `66`;
- first stored `fsq`: `4.9979943890699955e-6`;
- best/final `fsq`: `2.7719123357593158e-9` at row `65`;
- reduction from first stored row: `1803.09x`;
- strict per-component convergence: `False`;
- total-`fsq` convergence: `True`;
- final residual components:
  - `fsqr = 1.2743019453850022e-9`;
  - `fsqz = 8.910607448490252e-10`;
  - `fsql = 6.065496455252883e-10`;
- mean iota: `0.014897818141733415`;
- magnetic well proxy: `-0.06343611260202262`.

Compared with the M114 parity/VMEC2000 evidence, the accelerated path uses a
different boundary-inferred initialization and reaches the total-`fsq` target
within the same small-run wall-time range.  This supports keeping the fast CLI
path distinct from the raw parity path: parity is for VMEC2000 control-flow
audits, while accelerated mode is the practical solve mode for examples.

Rendered ignored plots:

- `results/toroidal_hybrid_m115_accelerated80/figures/toroidal_hybrid_fsq_history.png`;
- `results/toroidal_hybrid_m115_accelerated80/figures/toroidal_hybrid_profiles.png`;
- `results/toroidal_hybrid_m115_accelerated80/figures/toroidal_hybrid_orientation_preservation.png`.

### How it was tested

This was an evidence-only tranche using code already tested in M113/M114.  The
accelerated solve completed, wrote a WOUT, and rendered the plots above.  PR
checks were queried before the run and GitHub reported no checks for the branch
at that moment.

### File structure and best-practice notes

- No source changes were needed.
- Evidence stays under ignored `results/`.
- The plan records the command and metrics so the result can be reproduced
  without storing large artifacts in git.

### Best next steps

1. Commit and push the M115 plan evidence.
2. Resume the circular-coil finite-beta/free-boundary lane:
   - run finite-beta fixed-boundary baselines for 1%, 3%, and 10%;
   - add LCFS pilot tolerance/stagnation termination;
   - compare cross-beta LCFS metrics after actual iterations.
3. After the free-boundary lane is stable, return to differentiable
   solved-state/implicit derivative promotion.
4. Keep PR checks non-blocking; inspect only failing logs when they appear.

### Completion percentages after M115

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `87%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `90%`.
- I/O schema and docs: `97%`.
- Differentiable solved-state API: `30%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `74%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `66%`.
- PR merge readiness overall: `95%`.

### User input needed

No user input is needed.

---

## 116. 2026-06-18 M16 circular-coil LCFS pilot stop criteria

### Steps taken

I resumed the ESSOS-compatible circular-coil/free-boundary lane by making the
LCFS pilot loop finite and auditable.  The example can now stop multi-step pilot
updates by target merit, accepted-step stagnation, rejected merit increase,
explicit no-op selection, or the requested step cap.

Concretely:

- added `--lcfs-pilot-target-merit`;
- added `--lcfs-pilot-stagnation-rtol`;
- added top-level `lcfs_pilot_target_merit`,
  `lcfs_pilot_stagnation_rtol`, and `lcfs_pilot_stop_reason_counts`;
- added per-beta `lcfs_pilot_stop_reason`;
- added per-pilot-row `stop_reason`;
- added accepted-row `lcfs_merit_improvement_fraction`;
- preserved current defaults, so the existing one-step pilot still stops by
  `max_steps`;
- documented the new controls in the mirror README and overview;
- added tests for accepted/max-step, strict no-op skip, and stagnation stop
  behavior.

### Results obtained

Plotted finite-beta evidence run:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/mirror_free_boundary_circular_coils.py \
  --outdir results/mirror/free_boundary_circular_coils_m116_stop_criteria \
  --ntheta 8 \
  --nxi 11 \
  --n-segments 64 \
  --run-fixed-boundary-baseline \
  --baseline-maxiter 2 \
  --run-lcfs-pilot \
  --lcfs-pilot-steps 3 \
  --lcfs-pilot-stagnation-rtol 1.0
```

Observed:

- workflow: `lcfs_pilot`;
- baseline beta rows: `3`;
- requested pilot steps: `3`;
- actual pilot rows: `3`;
- accepted pilot rows: `2`;
- skipped pilot rows: `0`;
- stop reasons: `{"rejected_merit_increase": 1, "merit_stagnation": 2}`;
- beta `1%`:
  - baseline optimizer iterations: `2`;
  - baseline `final_fsq`: `2.4933734830599714e-4`;
  - baseline LCFS merit: `1.0000568856893208`;
  - pilot stop: `rejected_merit_increase`;
  - pilot final merit: `1.027986964110209`;
- beta `3%`:
  - baseline optimizer iterations: `2`;
  - baseline `final_fsq`: `0.004468220052785157`;
  - baseline LCFS merit: `1.0000568856893208`;
  - pilot stop: `merit_stagnation`;
  - pilot final merit: `0.7182791913011406`;
  - merit improvement: `0.2817616661815753`;
- beta `10%`:
  - baseline optimizer iterations: `2`;
  - baseline `final_fsq`: `0.04263091618797516`;
  - baseline LCFS merit: `1.0000568856893208`;
  - pilot stop: `merit_stagnation`;
  - pilot final merit: `0.7182791913011406`;
  - merit improvement: `0.2817616661815753`.

This is still a low-resolution pilot, not a converged free-boundary solve, but
the beta scan now has explicit, machine-readable termination reasons and actual
finite-beta baseline iterations.

Rendered ignored plot:

- `results/mirror/free_boundary_circular_coils_m116_stop_criteria/figures/free_boundary_circular_coils_beta_scan_summary.png`.

### How it was tested

Commands run:

```bash
python -m ruff check examples/mirror_free_boundary_circular_coils.py tests/mirror/test_mirror_examples.py
python -m ruff format --check examples/mirror_free_boundary_circular_coils.py tests/mirror/test_mirror_examples.py
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_example_runs_without_plots tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_strict_bnormal_guard_can_skip_pilot tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_pilot_stagnation_stops_early -q
git diff --check
```

Results:

- Ruff lint passed.
- Ruff format check passed after formatting.
- `3 passed` in the focused circular-coil example tests.
- `git diff --check` passed.
- The plotted finite-beta run completed and rendered the beta summary figure.

### File structure and best-practice notes

- Stop criteria live in the root circular-coil example because this is still an
  example-level LCFS pilot workflow, not a core free-boundary solver API.
- The low-level mirror update proposal helpers were not changed.
- Tests exercise the CLI output contract through the example, matching how users
  will run the beta scan.
- Generated evidence remains under ignored `results/`.

### Best next steps

1. Commit and push M116.
2. Run a less artificial pilot with a realistic stagnation tolerance, e.g.
   `1e-3`, and enough steps to see whether beta trends separate.
3. Add finite-beta baseline trend fields to the beta summary plot if needed
   (for example final `fsq` or optimizer iterations).
4. Promote a compact JSON contract for the beta scan that downstream ESSOS work
   can consume.
5. Then return to differentiable solved-state/implicit derivative promotion.

### Completion percentages after M116

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `87%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `90%`.
- I/O schema and docs: `97%`.
- Differentiable solved-state API: `30%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `77%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `72%`.
- PR merge readiness overall: `95%`.

### User input needed

No user input is needed.

---

## 117. 2026-06-18 M16 circular-coil realistic pilot and final-fsq trend

### Steps taken

I ran the circular-coil beta scan with a more realistic pilot stagnation
tolerance and added fixed-boundary residual information to the beta summary.
This makes the free-boundary pilot more honest: LCFS merit can improve while the
underlying fixed-boundary residual worsens, and the plot now shows both.

Concretely:

- added `lcfs_pilot_final_fsq`, `lcfs_pilot_best_fsq`, and
  `lcfs_pilot_final_normalized_force` to per-beta summaries;
- added a fourth `final fsq` panel to
  `free_boundary_circular_coils_beta_scan_summary.png`;
- updated tests to require the new pilot `fsq` summary fields;
- updated mirror docs/README to name the final-`fsq` summary panel.

### Results obtained

Realistic low-resolution pilot run:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/mirror_free_boundary_circular_coils.py \
  --outdir results/mirror/free_boundary_circular_coils_m117_pilot_rtol \
  --ntheta 8 \
  --nxi 11 \
  --n-segments 64 \
  --run-fixed-boundary-baseline \
  --baseline-maxiter 5 \
  --run-lcfs-pilot \
  --lcfs-pilot-steps 5 \
  --lcfs-pilot-stagnation-rtol 1e-3
```

Observed:

- workflow: `lcfs_pilot`;
- baseline beta rows: `3`;
- requested pilot steps: `5`;
- actual pilot rows: `15`;
- accepted pilot rows: `15`;
- skipped pilot rows: `0`;
- stop reasons: `{"None": 12, "max_steps": 3}`;
- beta `1%`:
  - baseline `final_fsq`: `9.927119074706592e-4`;
  - pilot final `fsq`: `8.507334465997071e-4`;
  - pilot best `fsq`: `8.285450166810463e-4`;
  - pilot final merit: `0.04135630008705599`;
  - pilot final pressure RMS: `0.060082322237203306`;
- beta `3%`:
  - baseline `final_fsq`: `0.004468220052785157`;
  - pilot final `fsq`: `0.006353407728010569`;
  - pilot best `fsq`: `0.004709131909718425`;
  - pilot final merit: `0.03604071048265542`;
  - pilot final pressure RMS: `0.060082322237203306`;
- beta `10%`:
  - baseline `final_fsq`: `0.04263091618797516`;
  - pilot final `fsq`: `0.06789198818929472`;
  - pilot best `fsq`: `0.046665979566329834`;
  - pilot final merit: `0.03604071048265542`;
  - pilot final pressure RMS: `0.060082322237203306`.

Interpretation: the LCFS pilot strongly reduces pressure-balance/merit metrics,
but at this low resolution and with only five baseline iterations it can worsen
the fixed-boundary residual for the higher-beta cases.  That is exactly why the
plot now includes final `fsq`; the next free-boundary step should couple LCFS
updates to a residual-quality gate or rerun fixed-boundary baselines longer.

Rendered ignored plot:

- `results/mirror/free_boundary_circular_coils_m117_pilot_rtol/figures/free_boundary_circular_coils_beta_scan_summary.png`.

### How it was tested

Commands run:

```bash
python -m ruff check examples/mirror_free_boundary_circular_coils.py tests/mirror/test_mirror_examples.py
python -m ruff format --check examples/mirror_free_boundary_circular_coils.py tests/mirror/test_mirror_examples.py
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_example_runs_without_plots tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_strict_bnormal_guard_can_skip_pilot tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_pilot_stagnation_stops_early -q
git diff --check
```

Results:

- Ruff lint passed.
- Ruff format check passed.
- `3 passed` in focused circular-coil tests.
- `git diff --check` passed.
- The plotted realistic pilot run completed and rendered.

### File structure and best-practice notes

- Plot expansion stays inside the root example; no core solver APIs changed.
- Summary fields are plain JSON scalars so ESSOS/downstream scripts can consume
  them without loading `mout` files.
- Generated evidence remains ignored under `results/`.

### Best next steps

1. Commit and push M117.
2. Add an optional LCFS acceptance guard on fixed-boundary residual quality, for
   example requiring pilot `final_fsq <= baseline final_fsq * factor`.
3. Re-run the 1%, 3%, and 10% beta scan with that residual-quality gate and a
   longer baseline solve budget.
4. Promote the circular-coil beta scan JSON contract into docs for ESSOS
   integration.
5. Return to differentiable solved-state/implicit derivatives once free-boundary
   pilot gates are explicit.

### Completion percentages after M117

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `87%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `90%`.
- I/O schema and docs: `97%`.
- Differentiable solved-state API: `30%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `79%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `75%`.
- PR merge readiness overall: `95%`.

### User input needed

No user input is needed.

---

## 118. 2026-06-18 M16 circular-coil fsq growth guard

### Steps taken

M117 showed that LCFS merit can improve while fixed-boundary `fsq` worsens for
higher-beta pilot updates.  I added an optional residual-quality guard so a
pilot update can be rejected when its fixed-boundary `final_fsq` exceeds a
configured multiple of the baseline row.

Concretely:

- added `--lcfs-pilot-fsq-growth-limit`;
- added top-level `lcfs_pilot_fsq_growth_limit`;
- added `rejection_reason` to completed pilot rows;
- made the pilot loop reject accepted-merit candidates with
  `stop_reason = "fsq_growth_guard"` when the guard is enabled and violated;
- changed the beta summary plot to draw only accepted pilot-final values, so
  rejected trials are not presented as accepted outcomes;
- documented the guard in the mirror README and overview;
- added a focused CLI test for the fsq-growth rejection path.

### Results obtained

Guarded low-resolution pilot run:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/mirror_free_boundary_circular_coils.py \
  --outdir results/mirror/free_boundary_circular_coils_m118_fsq_guard \
  --ntheta 8 \
  --nxi 11 \
  --n-segments 64 \
  --run-fixed-boundary-baseline \
  --baseline-maxiter 5 \
  --run-lcfs-pilot \
  --lcfs-pilot-steps 5 \
  --lcfs-pilot-stagnation-rtol 1e-3 \
  --lcfs-pilot-fsq-growth-limit 1.0
```

Observed:

- requested pilot steps: `5`;
- actual pilot rows: `7`;
- accepted pilot rows: `5`;
- stop reasons: `{"None": 4, "max_steps": 1, "fsq_growth_guard": 2}`;
- beta `1%`:
  - baseline `final_fsq`: `9.927119074706592e-4`;
  - accepted pilot final `fsq`: `8.507334465997071e-4`;
  - stop reason: `max_steps`;
- beta `3%`:
  - baseline `final_fsq`: `0.004468220052785157`;
  - rejected trial `final_fsq`: `0.004709131909718425`;
  - stop reason: `fsq_growth_guard`;
- beta `10%`:
  - baseline `final_fsq`: `0.04263091618797516`;
  - rejected trial `final_fsq`: `0.046665979566329834`;
  - stop reason: `fsq_growth_guard`.

The accepted-only beta summary plot now shows the 1% accepted pilot outcome and
leaves the rejected 3%/10% trials out of the pilot-final curves, while the JSON
retains their rejected trial metrics for audit.

Rendered ignored plot:

- `results/mirror/free_boundary_circular_coils_m118_fsq_guard/figures/free_boundary_circular_coils_beta_scan_summary.png`.

### How it was tested

Commands run:

```bash
python -m ruff check examples/mirror_free_boundary_circular_coils.py tests/mirror/test_mirror_examples.py
python -m ruff format --check examples/mirror_free_boundary_circular_coils.py tests/mirror/test_mirror_examples.py
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_example_runs_without_plots tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_strict_bnormal_guard_can_skip_pilot tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_pilot_stagnation_stops_early tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_fsq_growth_guard_rejects_pilot -q
git diff --check
```

Results:

- Ruff lint passed.
- Ruff format check passed.
- `4 passed` in focused circular-coil tests.
- `git diff --check` passed.
- The guarded plotted beta run completed and rendered.

### File structure and best-practice notes

- The guard remains in the example-level pilot workflow; core solver APIs are
  unchanged.
- JSON fields remain scalar and explicit so ESSOS/downstream scripts can inspect
  accepted/rejected decisions.
- Plot semantics now distinguish accepted pilot outcomes from rejected trial
  diagnostics.
- Generated evidence remains ignored under `results/`.

### Best next steps

1. Commit and push M118.
2. Add a compact documented JSON schema section for the circular-coil beta scan
   contract.
3. Run a higher baseline budget for 1%, 3%, and 10% with the fsq guard enabled
   to see whether higher beta accepts after better fixed-boundary solves.
4. Then resume differentiable solved-state/implicit derivative promotion.

### Completion percentages after M118

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `87%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `90%`.
- I/O schema and docs: `97%`.
- Differentiable solved-state API: `30%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `81%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `78%`.
- PR merge readiness overall: `95%`.

### User input needed

No user input is needed.

---
## 119. Circular-Coil Beta-Scan JSON Contract

### Steps taken

- Added an explicit compact metrics contract to
  `examples/mirror_free_boundary_circular_coils.py`:
  - schema name: `mirror_free_boundary_circular_coil_beta_scan`;
  - schema version: `0.1`;
  - required top-level, beta-row, and pilot-row field lists;
  - allowed pilot status, stop-reason, and rejection-reason values.
- Added `circular_coil_beta_scan_schema()` and
  `validate_circular_coil_beta_scan_metrics(metrics)` so the example validates
  its JSON before writing it.
- Stabilized pilot-row shape by always emitting `stop_reason`,
  `rejection_reason`, and `lcfs_merit_improvement_fraction`, using `null` when
  a row is intermediate or a field does not apply.
- Updated the focused circular-coil example test to call the schema helper and
  validator, and to assert that the required field sets are present in the
  emitted JSON.
- Updated `examples/mirror/README.md` and `docs/mirror/overview.rst` with the
  schema name/version and the accepted/rejected pilot-row semantics.

### Results obtained

- The circular-coil beta-scan metrics are now easier for ESSOS/downstream tools
  to consume without defensive missing-key checks.
- Rejected pilot rows remain in JSON for audit, while plots continue to show
  accepted pilot-final values only.
- The contract remains example-scoped; core mirror solver APIs are unchanged.

### How it was tested

Commands run:

```bash
python -m ruff check examples/mirror_free_boundary_circular_coils.py tests/mirror/test_mirror_examples.py
python -m ruff format --check examples/mirror_free_boundary_circular_coils.py tests/mirror/test_mirror_examples.py
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_example_runs_without_plots tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_strict_bnormal_guard_can_skip_pilot tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_pilot_stagnation_stops_early tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_fsq_growth_guard_rejects_pilot -q
python - <<'PY'
import runpy
m = runpy.run_path('examples/mirror_free_boundary_circular_coils.py')
s = m['circular_coil_beta_scan_schema']()
print(s['metrics_schema'], s['metrics_schema_version'])
print(len(s['top_level_required_fields']), len(s['beta_row_required_fields']), len(s['pilot_row_required_fields']))
PY
git diff --check
```

Results:

- Ruff lint passed.
- Ruff format check passed after formatting the example file.
- `4 passed` in focused circular-coil tests.
- Schema helper reported version `0.1` and required-field counts
  `22`, `24`, and `17`.
- `git diff --check` passed.

### File structure and best-practice notes

- The schema constants live beside the example that writes the JSON, keeping the
  contract near the workflow that owns it.
- Validation is simple dictionary/key checking rather than a new dependency or
  heavyweight schema framework.
- The emitted JSON stays compact: it carries schema name/version, not a full
  embedded schema dump.
- Generated outputs remain ignored under `results/`.

### Best next steps

1. Commit and push M119.
2. Run a higher-budget circular-coil beta scan with the fsq growth guard
   enabled to determine whether the 3% and 10% rejected pilot rows become
   acceptable after better fixed-boundary baseline solves.
3. If high-beta rows remain rejected, separate LCFS-merit improvement from
   fixed-boundary residual degradation in the next proposal-selection criterion.
4. Resume differentiable solved-state/implicit-derivative promotion after the
   ESSOS-facing beta-scan contract is stable.

### Completion percentages after M119

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `87%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `90%`.
- I/O schema and docs: `98%`.
- Differentiable solved-state API: `30%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `82%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `80%`.
- PR merge readiness overall: `95%`.

### User input needed

No user input is needed.

---
## 120. Circular-Coil Higher-Budget Fsq-Guard Evidence

### Steps taken

- Checked PR CI once after pushing M119:
  - `gh pr checks 21 --watch=false` reported no checks yet for the new branch
    head, so no failure was available to fix.
- Ran the circular-coil free-boundary planning fixture with a higher
  fixed-boundary baseline budget and the strict fsq-growth guard enabled:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 python examples/mirror_free_boundary_circular_coils.py \
  --outdir results/mirror/free_boundary_circular_coils_m120_baseline20_fsq_guard \
  --ntheta 8 --nxi 11 --n-segments 64 \
  --run-fixed-boundary-baseline --baseline-maxiter 20 \
  --run-lcfs-pilot --lcfs-pilot-steps 5 \
  --lcfs-pilot-stagnation-rtol 1e-3 \
  --lcfs-pilot-fsq-growth-limit 1.0
```

- Parsed the resulting metrics JSON and rendered the beta-summary, geometry,
  and 1% residual-history plots for inspection.

### Results obtained

The run completed and wrote:

- `results/mirror/free_boundary_circular_coils_m120_baseline20_fsq_guard/free_boundary_circular_coils_metrics.json`;
- `70` ignored PNG diagnostics under
  `results/mirror/free_boundary_circular_coils_m120_baseline20_fsq_guard/figures`.

Schema/status:

- schema: `mirror_free_boundary_circular_coil_beta_scan` version `0.1`;
- workflow: `lcfs_pilot`;
- pilot rows: `3`;
- accepted pilot rows: `0`;
- skipped pilot rows: `0`;
- stop-reason counts:
  `{"rejected_merit_increase": 1, "fsq_growth_guard": 2}`.

Per-beta outcome:

- beta `1%`:
  - baseline `final_fsq`: `7.181025259848037e-8`;
  - trial `final_fsq`: `8.975298976238274e-4`;
  - baseline LCFS merit: `1.0000568856893208`;
  - trial LCFS merit: `1.0207974121441186`;
  - status: rejected by `rejected_merit_increase`.
- beta `3%`:
  - baseline `final_fsq`: `0.004468220052785157`;
  - trial `final_fsq`: `0.004709131909718425`;
  - baseline LCFS merit: `1.0000568856893208`;
  - trial LCFS merit: `0.7182791913011406`;
  - status: rejected by `fsq_growth_guard`.
- beta `10%`:
  - baseline `final_fsq`: `0.04263091618797516`;
  - trial `final_fsq`: `0.046665979566329834`;
  - baseline LCFS merit: `1.0000568856893208`;
  - trial LCFS merit: `0.7182791913011406`;
  - status: rejected by `fsq_growth_guard`.

Interpretation:

- Increasing `baseline-maxiter` from `5` to `20` made the 1% baseline much
  tighter, and the first LCFS pilot proposal no longer improves actual LCFS
  merit.
- The 3% and 10% proposals still improve actual LCFS merit, but the strict
  `final_fsq <= baseline_fsq` guard rejects them because the trial residual
  grows by about `5.4%` and `9.5%`, respectively.
- The accepted-only beta summary plot correctly shows only baseline curves,
  because all pilot trial rows were rejected.

Rendered ignored plots inspected:

- `results/mirror/free_boundary_circular_coils_m120_baseline20_fsq_guard/figures/free_boundary_circular_coils_beta_scan_summary.png`;
- `results/mirror/free_boundary_circular_coils_m120_baseline20_fsq_guard/figures/free_boundary_circular_coils_geometry.png`;
- `results/mirror/free_boundary_circular_coils_m120_baseline20_fsq_guard/figures/fixed_boundary_beta_1/free_boundary_circular_coils_beta_1_mirror_residual_history.png`.

### How it was tested

Commands run:

```bash
gh pr checks 21 --watch=false
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 python examples/mirror_free_boundary_circular_coils.py --outdir results/mirror/free_boundary_circular_coils_m120_baseline20_fsq_guard --ntheta 8 --nxi 11 --n-segments 64 --run-fixed-boundary-baseline --baseline-maxiter 20 --run-lcfs-pilot --lcfs-pilot-steps 5 --lcfs-pilot-stagnation-rtol 1e-3 --lcfs-pilot-fsq-growth-limit 1.0
python - <<'PY'
import json
from pathlib import Path
path=Path('results/mirror/free_boundary_circular_coils_m120_baseline20_fsq_guard/free_boundary_circular_coils_metrics.json')
metrics=json.loads(path.read_text())
print(metrics['metrics_schema'], metrics['metrics_schema_version'])
print(metrics['lcfs_pilot_stop_reason_counts'])
for row in metrics['fixed_boundary_baseline_rows']:
    print(row['beta_percent'], row['lcfs_pilot_status'], row['final_fsq'], row['lcfs_pilot_final_fsq'])
PY
find results/mirror/free_boundary_circular_coils_m120_baseline20_fsq_guard/figures -maxdepth 3 -type f -name '*.png' -print | wc -l
git status --short --ignored=matching results/mirror/free_boundary_circular_coils_m120_baseline20_fsq_guard
```

Results:

- CI had no reported checks for the newest head at the time of inspection.
- The example completed successfully.
- Metrics parsing succeeded.
- Figure count was `70`.
- The generated `results/` tree remains ignored by git.

### File structure and best-practice notes

- This was an evidence/logging milestone; no source code changes were needed.
- The run used the root example and the newly documented schema contract.
- Output files stayed under ignored `results/` and were not staged.
- The strict fsq guard is now doing useful work by preventing residual-worse
  LCFS updates from being reported as accepted free-boundary progress.

### Best next steps

1. Commit and push this evidence log.
2. Add an explicit `lcfs_fsq_growth_ratio`/`lcfs_pilot_fsq_growth_ratio`
   diagnostic to beta and pilot rows so downstream tools can distinguish
   slight residual growth from severe fixed-boundary degradation without
   recomputing ratios.
3. Add an optional acceptance mode or proposal-selection diagnostic that
   separates LCFS-merit improvement from fixed-boundary residual degradation,
   because the current strict guard rejects high-beta merit improvements that
   only slightly increase `fsq`.
4. Then rerun the higher-budget scan with a small tolerated fsq-growth limit
   such as `1.1` to determine whether 3% and 10% are stable under a pragmatic
   guard while 1% remains rejected by actual LCFS merit.

### Completion percentages after M120

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `87%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `91%`.
- I/O schema and docs: `98%`.
- Differentiable solved-state API: `30%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `83%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `81%`.
- PR merge readiness overall: `95%`.

### User input needed

No user input is needed.

---

## 121. Circular-Coil Fsq-Growth Ratio Diagnostics

### Steps taken

- Added explicit residual-growth diagnostics to
  `examples/mirror_free_boundary_circular_coils.py`:
  - pilot rows now include `fsq_growth_ratio`;
  - beta rows now include `lcfs_pilot_final_fsq_growth_ratio`;
  - beta rows now include `lcfs_pilot_best_fsq_growth_ratio`.
- Reused the same `fsq_growth_ratio` value for the fsq-growth guard decision so
  the logged diagnostic and acceptance decision cannot drift.
- Extended the compact schema contract to require the new beta-row and
  pilot-row fields.
- Updated the focused circular-coil tests for accepted, skipped, stagnation,
  and fsq-guard paths.
- Updated `examples/mirror/README.md` and `docs/mirror/overview.rst` to document
  the new growth-ratio fields.

### Results obtained

- Downstream ESSOS or analysis scripts can now read residual growth directly
  from the JSON rather than recomputing it from baseline and pilot rows.
- Skipped pilot rows report `fsq_growth_ratio: null`.
- Trial pilot rows report finite ratios whether accepted or rejected, including
  rejected merit-increase and fsq-growth-guard cases.
- The schema helper now reports `26` beta-row required fields and `18`
  pilot-row required fields.

### How it was tested

Commands run:

```bash
python -m ruff check examples/mirror_free_boundary_circular_coils.py tests/mirror/test_mirror_examples.py
python -m ruff format --check examples/mirror_free_boundary_circular_coils.py tests/mirror/test_mirror_examples.py
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_example_runs_without_plots tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_strict_bnormal_guard_can_skip_pilot tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_pilot_stagnation_stops_early tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_fsq_growth_guard_rejects_pilot -q
python - <<'PY'
import runpy
m=runpy.run_path('examples/mirror_free_boundary_circular_coils.py')
s=m['circular_coil_beta_scan_schema']()
print(s['metrics_schema'], s['metrics_schema_version'])
print('row fields', len(s['beta_row_required_fields']), 'pilot fields', len(s['pilot_row_required_fields']))
print('has ratio', 'lcfs_pilot_final_fsq_growth_ratio' in s['beta_row_required_fields'], 'fsq_growth_ratio' in s['pilot_row_required_fields'])
PY
python -m ruff format examples/mirror_free_boundary_circular_coils.py tests/mirror/test_mirror_examples.py
git diff --check
```

Results:

- Ruff lint passed.
- Ruff format check passed after formatting two files.
- `4 passed` in focused circular-coil tests.
- Schema helper confirmed the ratio fields are in the contract.
- `git diff --check` passed.

### File structure and best-practice notes

- The diagnostic remains example-level and schema-level; no core solver API was
  changed.
- The field names are scalar and explicit, matching the rest of the beta-scan
  JSON.
- The guard uses the same ratio that is written to JSON, reducing duplicated
  arithmetic and audit ambiguity.

### Best next steps

1. Commit and push M121.
2. Rerun the higher-budget circular-coil scan with
   `--lcfs-pilot-fsq-growth-limit 1.1` to test whether the 3% and 10% rows are
   acceptable under a small residual-growth tolerance while 1% remains rejected
   by actual LCFS merit.
3. If 3% and 10% accept at `1.1`, document strict versus tolerant guard
   behavior and make the recommended guard explicit in the example docs.
4. If they still reject, add a proposal-selection diagnostic that accounts for
   fixed-boundary residual growth before selecting the LCFS update.

### Completion percentages after M121

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `87%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `91%`.
- I/O schema and docs: `98%`.
- Differentiable solved-state API: `30%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `84%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `82%`.
- PR merge readiness overall: `95%`.

### User input needed

No user input is needed.

---

## 122. Circular-Coil Last-Accepted Pilot State

### Steps taken

- Ran the higher-budget circular-coil beta scan with a tolerant fsq-growth
  guard:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 python examples/mirror_free_boundary_circular_coils.py \
  --outdir results/mirror/free_boundary_circular_coils_m122_baseline20_fsq_guard1p1 \
  --ntheta 8 --nxi 11 --n-segments 64 \
  --run-fixed-boundary-baseline --baseline-maxiter 20 \
  --run-lcfs-pilot --lcfs-pilot-steps 5 \
  --lcfs-pilot-stagnation-rtol 1e-3 \
  --lcfs-pilot-fsq-growth-limit 1.1
```

- The run showed that beta `3%` and `10%` accepted pilot step 1, then rejected
  step 2 by the fsq-growth guard; however, the summary plot only drew rows whose
  final pilot status was accepted, hiding the accepted step-1 progress.
- Added explicit last-accepted pilot-state fields to each beta row:
  - `lcfs_pilot_last_accepted_step`;
  - `lcfs_pilot_last_accepted_merit`;
  - `lcfs_pilot_last_accepted_pressure_balance_rms`;
  - `lcfs_pilot_last_accepted_fsq`;
  - `lcfs_pilot_last_accepted_fsq_growth_ratio`;
  - `lcfs_pilot_last_accepted_normalized_force`.
- Updated the beta-scan summary plot to draw the last accepted pilot state, not
  the final rejected trial.
- Extended the compact schema, tests, README, and mirror overview docs.
- Reran the tolerant-guard evidence into a fresh ignored results directory:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 python examples/mirror_free_boundary_circular_coils.py \
  --outdir results/mirror/free_boundary_circular_coils_m122_guard1p1_last_accepted \
  --ntheta 8 --nxi 11 --n-segments 64 \
  --run-fixed-boundary-baseline --baseline-maxiter 20 \
  --run-lcfs-pilot --lcfs-pilot-steps 5 \
  --lcfs-pilot-stagnation-rtol 1e-3 \
  --lcfs-pilot-fsq-growth-limit 1.1
```

### Results obtained

Refreshed tolerant-guard run:

- output JSON:
  `results/mirror/free_boundary_circular_coils_m122_guard1p1_last_accepted/free_boundary_circular_coils_metrics.json`;
- figure count: `92` ignored PNGs;
- summary plot:
  `results/mirror/free_boundary_circular_coils_m122_guard1p1_last_accepted/figures/free_boundary_circular_coils_beta_scan_summary.png`;
- pilot rows: `5`;
- accepted pilot rows: `2`;
- stop counts: `{"rejected_merit_increase": 1, "None": 2, "fsq_growth_guard": 2}`.

Per-beta outcome:

- beta `1%`:
  - final status: `rejected`;
  - accepted rows: `0`;
  - last accepted step: `null`;
  - final rejected trial `fsq_growth_ratio`: `12498.631673700875`;
  - reason: actual LCFS merit worsened.
- beta `3%`:
  - final status: `rejected`;
  - accepted rows: `1`;
  - last accepted step: `1`;
  - last accepted `fsq_growth_ratio`: `1.0539167395712978`;
  - final rejected trial `fsq_growth_ratio`: `1.1236410160141437`;
  - last accepted LCFS merit: `0.7182791913011406`;
  - final rejected trial LCFS merit: `0.48532561064534907`.
- beta `10%`:
  - final status: `rejected`;
  - accepted rows: `1`;
  - last accepted step: `1`;
  - last accepted `fsq_growth_ratio`: `1.0946511062666966`;
  - final rejected trial `fsq_growth_ratio`: `1.2003050324215168`;
  - last accepted LCFS merit: `0.7182791913011406`;
  - final rejected trial LCFS merit: `0.48532561064534907`.

The refreshed summary plot now displays orange last-accepted pilot markers for
the `3%` and `10%` cases, while leaving the `1%` case without a pilot marker.
This matches the actual acceptance decisions and keeps rejected trial rows
available in JSON for audit.

### How it was tested

Commands run:

```bash
gh pr checks 21 --watch=false
python -m ruff check examples/mirror_free_boundary_circular_coils.py tests/mirror/test_mirror_examples.py
python -m ruff format --check examples/mirror_free_boundary_circular_coils.py tests/mirror/test_mirror_examples.py
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_example_runs_without_plots tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_strict_bnormal_guard_can_skip_pilot tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_pilot_stagnation_stops_early tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_fsq_growth_guard_rejects_pilot tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_tolerant_fsq_guard_keeps_last_accepted -q
python -m ruff format examples/mirror_free_boundary_circular_coils.py tests/mirror/test_mirror_examples.py
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 python examples/mirror_free_boundary_circular_coils.py --outdir results/mirror/free_boundary_circular_coils_m122_guard1p1_last_accepted --ntheta 8 --nxi 11 --n-segments 64 --run-fixed-boundary-baseline --baseline-maxiter 20 --run-lcfs-pilot --lcfs-pilot-steps 5 --lcfs-pilot-stagnation-rtol 1e-3 --lcfs-pilot-fsq-growth-limit 1.1
git diff --check
```

Results:

- CI had no reported checks for the newest head when inspected.
- Ruff lint passed.
- Ruff format check passed after formatting.
- Focused tests: `5 passed`.
- The tolerant plotted evidence run completed.
- The generated `results/` tree remains ignored by git.

### File structure and best-practice notes

- Final rejected-trial fields and last-accepted fields are intentionally
  separate. This avoids overloading `final_*` with accepted-state semantics.
- Plotting now uses last accepted pilot data, while JSON retains every rejected
  trial row for audit and method development.
- The schema remains compact and example-scoped.
- No core solver API was changed.

### Best next steps

1. Commit and push M122.
2. Update the README recommendation to describe strict guard `1.0` as a
   diagnostic mode and `1.1` as the current low-resolution pragmatic pilot
   tolerance for the 3%/10% circular-coil rows.
3. Add a lightweight JSON postprocessor or table helper that extracts baseline,
   last-accepted, and final rejected trial columns for ESSOS comparison reports.
4. Resume differentiable solved-state/implicit-derivative API work after the
   ESSOS beta-scan reporting contract is stable.

### Completion percentages after M122

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `87%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `92%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `30%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `85%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `84%`.
- PR merge readiness overall: `95%`.

### User input needed

No user input is needed.

---

## 123. Circular-Coil Beta-Scan CSV Report

### Steps taken

- Added a lightweight CSV report helper to
  `examples/mirror_free_boundary_circular_coils.py`:
  - `circular_coil_beta_scan_report_rows(metrics)`;
  - `_write_beta_scan_report_csv(path, rows)`.
- Added `summary_csv` to the top-level metrics contract.
- Added `report_fields` to `circular_coil_beta_scan_schema()`.
- The example now writes
  `free_boundary_circular_coils_beta_scan_summary.csv` next to the metrics JSON.
- Updated focused tests to read the CSV and check that it mirrors the JSON beta
  rows.
- Updated `examples/mirror/README.md` and `docs/mirror/overview.rst` to
  describe the CSV report and the current strict/tolerant fsq-guard usage.

### Results obtained

- The JSON remains the authoritative audit record with nested pilot rows.
- The CSV gives ESSOS/report scripts a compact table with one row per beta and
  columns for:
  - baseline residual/LCFS values;
  - pilot status and accepted-row count;
  - last accepted pilot state;
  - final trial state, which may be rejected.
- This makes strict-vs-tolerant fsq-guard comparisons easier without flattening
  JSON ad hoc in external scripts.

Lightweight evidence command:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 python examples/mirror_free_boundary_circular_coils.py \
  --outdir results/mirror/free_boundary_circular_coils_m123_csv_report \
  --ntheta 8 --nxi 11 --n-segments 64 \
  --run-fixed-boundary-baseline --baseline-maxiter 5 \
  --run-lcfs-pilot --lcfs-pilot-steps 2 \
  --lcfs-pilot-fsq-growth-limit 1.1 --no-plots
```

The CSV had `3` rows and `20` fields:

- beta `1%`: `pilot_status=accepted`, `pilot_accepted_rows=2`,
  `last_accepted_step=2`, `last_accepted_fsq_growth_ratio=0.8554750612791036`;
- beta `3%`: `pilot_status=rejected`, `pilot_accepted_rows=1`,
  `last_accepted_step=1`, `last_accepted_fsq_growth_ratio=1.0539167395712978`,
  final trial `fsq_growth_ratio=1.1236410160141437`;
- beta `10%`: `pilot_status=rejected`, `pilot_accepted_rows=1`,
  `last_accepted_step=1`, `last_accepted_fsq_growth_ratio=1.0946511062666966`,
  final trial `fsq_growth_ratio=1.2003050324215168`.

### How it was tested

Commands run:

```bash
python -m ruff check examples/mirror_free_boundary_circular_coils.py tests/mirror/test_mirror_examples.py
python -m ruff format --check examples/mirror_free_boundary_circular_coils.py tests/mirror/test_mirror_examples.py
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_example_runs_without_plots tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_tolerant_fsq_guard_keeps_last_accepted -q
python - <<'PY'
import runpy
m=runpy.run_path('examples/mirror_free_boundary_circular_coils.py')
s=m['circular_coil_beta_scan_schema']()
print('summary_csv required', 'summary_csv' in s['top_level_required_fields'])
print('report fields', len(s['report_fields']))
PY
python -m ruff format examples/mirror_free_boundary_circular_coils.py
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 python examples/mirror_free_boundary_circular_coils.py --outdir results/mirror/free_boundary_circular_coils_m123_csv_report --ntheta 8 --nxi 11 --n-segments 64 --run-fixed-boundary-baseline --baseline-maxiter 5 --run-lcfs-pilot --lcfs-pilot-steps 2 --lcfs-pilot-fsq-growth-limit 1.1 --no-plots
git diff --check
```

Results:

- Ruff lint passed.
- Ruff format check passed after formatting the example file.
- Targeted tests: `2 passed`.
- Schema helper confirmed `summary_csv` is required and the report has `20`
  fields.
- The no-plot CSV evidence run completed and wrote the expected report.
- `git diff --check` passed.

### File structure and best-practice notes

- The CSV helper is kept beside the example and schema constants because it is
  a reporting view of that example's metrics, not a core solver data model.
- The report is standard-library `csv` only; no new dependency was added.
- Generated CSV/JSON evidence remains ignored under `results/`.
- The added test covers the user-facing file emitted by the script rather than
  only testing the helper in isolation.

### Best next steps

1. Commit and push M123.
2. Revisit the differentiable solved-state lane now that the circular-coil
   ESSOS reporting contract is stable enough for handoff.
3. Define the smallest implicit-solve API that can return a converged mirror
   state and a custom derivative rule without differentiating through long CLI
   loops.
4. Keep the CLI examples on fast NumPy/SciPy paths for runtime/memory, with
   differentiable JAX APIs exposed separately for optimization.

### Completion percentages after M123

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `87%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `92%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `30%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `85%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `85%`.
- PR merge readiness overall: `95%`.

### User input needed

No user input is needed.

---

## 124. Axisymmetric Reduced Residual/Jacobian API

### Steps taken

- Added differentiable reduced-coordinate fixed-boundary utilities in
  `vmec_jax/mirror/solvers/fixed_boundary/reduced.py`:
  - `axisym_reduced_residual_jax(...)`;
  - `axisym_reduced_residual_jacobian_jax(...)`.
- The residual is the JAX gradient of the reduced axisymmetric mirror energy
  with respect to independent fixed-boundary coordinates.
- The Jacobian helper supports:
  - `derivative="hessian"` for the current energy-gradient residual;
  - `derivative="forward"` for `jax.jacfwd`;
  - `derivative="reverse"` for `jax.jacrev`.
- Exported the functions through `vmec_jax.mirror.api` and
  `vmec_jax.mirror`.
- Added focused tests that:
  - compare the JAX residual against the existing NumPy/JAX reduced-gradient
    path;
  - verify Hessian symmetry;
  - compare `hessian` and `jacfwd` modes;
  - compare the dense Jacobian action against `jax.jvp`.
- Updated `docs/mirror/overview.rst` to describe this as the first
  implicit-differentiation building block.

### Results obtained

- We now have an explicit residual map `F(x, p)` and dense linearization
  `dF/dx` for small axisymmetric fixed-boundary mirror states.
- This is the correct foundation for implicit differentiation:

```text
dF/dx * dx/dp = -dF/dp
```

or, in reverse mode,

```text
(dF/dx)^T * adjoint = dL/dx
```

- The implementation stays in JAX for differentiable kernels while the existing
  CLI/optimizer path remains free to use NumPy/SciPy for speed and memory.
- No host-side optimizer loop is being differentiated through.

### How it was tested

Commands run:

```bash
python -m ruff check vmec_jax/mirror/solvers/fixed_boundary/reduced.py vmec_jax/mirror/api.py vmec_jax/mirror/__init__.py tests/mirror/test_mirror_fixed_boundary_axisym.py
python -m ruff format --check vmec_jax/mirror/solvers/fixed_boundary/reduced.py vmec_jax/mirror/api.py vmec_jax/mirror/__init__.py tests/mirror/test_mirror_fixed_boundary_axisym.py
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_fixed_boundary_axisym.py::test_reduced_jax_residual_and_jacobian_match_gradient_and_jvp tests/mirror/test_mirror_fixed_boundary_axisym.py::test_reduced_lbfgs_gradient_matches_central_difference -q
python - <<'PY'
from vmec_jax.mirror import axisym_reduced_residual_jacobian_jax, axisym_reduced_residual_jax
print(axisym_reduced_residual_jax.__name__, axisym_reduced_residual_jacobian_jax.__name__)
PY
```

Results:

- Ruff lint passed.
- Ruff format check passed.
- Focused tests: `2 passed`.
- Public API import returned the expected function names.

### File structure and best-practice notes

- The new functions live in `solvers/fixed_boundary/reduced.py`, beside the
  reduced-coordinate packing and energy-gradient utilities they linearize.
- Public exports are explicit through `api.py` and `__init__.py`.
- The implementation is intentionally small: residual and linearization first,
  custom implicit solve/VJP later.
- The API names make the axisymmetric limitation explicit.

### Best next steps

1. Commit and push M124.
2. Add a small implicit sensitivity helper that solves dense
   `(dF/dx) dx = rhs` and `(dF/dx)^T adjoint = rhs` for tiny validation grids.
3. Validate the dense implicit sensitivity against finite differences of a
   manufactured or low-dimensional solved state.
4. Only after that, promote to a custom JAX implicit-differentiation rule or
   lineax-backed solve path for larger grids.

### Completion percentages after M124

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `88%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `92%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `37%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `85%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `85%`.
- PR merge readiness overall: `95%`.

### User input needed

No user input is needed.

---
## 125. Dense Reduced Implicit Linear Solve Helper

### Steps taken

- Added `axisym_reduced_residual_linear_solve_jax(...)` in
  `vmec_jax/mirror/solvers/fixed_boundary/reduced.py`.
- The helper forms the dense reduced residual Jacobian and solves either:
  - `J dx = rhs` for forward sensitivities; or
  - `J.T adjoint = rhs` for reverse/adjoint sensitivities.
- Added an optional scalar `ridge` regularization for tiny validation grids.
- Exported the helper through `vmec_jax.mirror.api` and `vmec_jax.mirror`.
- Extended the reduced-coordinate test to verify both forward and transpose
  solves against the dense matrix equations.
- Updated the mirror overview docs.

### Results obtained

- The differentiable lane now has three explicit small-grid reference pieces:
  - reduced residual `F(x, p)`;
  - residual Jacobian `dF/dx`;
  - dense primal/transpose linear solves.
- This is enough to validate implicit-differentiation formulas before adding a
  scalable lineax/matrix-free implementation.
- The helper is intentionally labeled and documented as a validation-grid path,
  not a production large-grid solver.

### How it was tested

Commands run:

```bash
python -m ruff check vmec_jax/mirror/solvers/fixed_boundary/reduced.py vmec_jax/mirror/api.py vmec_jax/mirror/__init__.py tests/mirror/test_mirror_fixed_boundary_axisym.py
python -m ruff format --check vmec_jax/mirror/solvers/fixed_boundary/reduced.py vmec_jax/mirror/api.py vmec_jax/mirror/__init__.py tests/mirror/test_mirror_fixed_boundary_axisym.py
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_fixed_boundary_axisym.py::test_reduced_jax_residual_and_jacobian_match_gradient_and_jvp -q
python - <<'PY'
from vmec_jax.mirror import axisym_reduced_residual_linear_solve_jax
print(axisym_reduced_residual_linear_solve_jax.__name__)
PY
```

Results:

- Ruff lint passed.
- Ruff format check passed.
- Focused reduced-coordinate test: `1 passed`.
- Public import returned `axisym_reduced_residual_linear_solve_jax`.

### File structure and best-practice notes

- The dense solve helper lives beside the residual/Jacobian it uses.
- The public API surface remains narrow and axisymmetric-specific.
- No CLI solver behavior changed.
- This keeps fast host-side examples separate from differentiable JAX building
  blocks, matching the plan's performance/differentiability split.

### Best next steps

1. Commit and push M125.
2. Add a tiny manufactured implicit sensitivity test:
   compare dense implicit `dx/dp` against finite differences of a solved
   low-dimensional state.
3. Add a short docs note explaining when to use dense validation solves versus
   future lineax/matrix-free adjoints.
4. Then implement the first custom implicit derivative wrapper around a
   converged small-grid solved state.

### Completion percentages after M125

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `88%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `83%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `92%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `42%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `85%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `85%`.
- PR merge readiness overall: `95%`.

### User input needed

No user input is needed.

---
## 126. Manufactured Reduced Implicit Sensitivity Gate

### Steps taken

- Extended the reduced JAX objective/residual with optional validation terms:
  - `source_vector`, a reduced-coordinate linear source;
  - `state_ridge`, a small quadratic state regularization;
  - `reference_vector`, the ridge reference state.
- These terms let us manufacture an exact tiny-grid root at a chosen state:

```text
F(x0, source0, reference=x0) = 0
```

where `source0` is the unforced reduced residual at `x0`.
- Added a focused test that:
  - manufactures an exact reduced root;
  - computes `dx/dp` with the dense implicit solve;
  - solves a perturbed source problem independently with SciPy root;
  - compares the finite-difference state change against the implicit
    sensitivity.
- Documented that this is a derivative-method validation gate, not yet a
  physical production differentiable equilibrium API.

### Results obtained

- The manufactured root residual is zero to numerical precision.
- The perturbed root solve reaches residual below `1e-10`.
- The finite-difference sensitivity matches the dense implicit sensitivity
  within the focused test tolerance.
- The differentiable lane now has a complete tiny-grid validation chain:
  residual -> Jacobian -> forward/adjoint dense solve -> manufactured
  sensitivity finite-difference check.

### How it was tested

Commands run:

```bash
python -m ruff check vmec_jax/mirror/solvers/fixed_boundary/reduced.py tests/mirror/test_mirror_fixed_boundary_axisym.py
python -m ruff format --check vmec_jax/mirror/solvers/fixed_boundary/reduced.py tests/mirror/test_mirror_fixed_boundary_axisym.py
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_fixed_boundary_axisym.py::test_reduced_jax_residual_and_jacobian_match_gradient_and_jvp tests/mirror/test_mirror_fixed_boundary_axisym.py::test_reduced_implicit_sensitivity_matches_manufactured_source_finite_difference -q
```

Results:

- Ruff lint passed.
- Ruff format check passed.
- Focused differentiability tests: `2 passed`.

### File structure and best-practice notes

- The source/ridge terms are optional and live in the reduced JAX objective only.
- Existing CLI solvers and examples are unchanged.
- The manufactured sensitivity test uses `scipy.optimize.root` only as a small
  independent validation solve.
- The test uses a tiny grid so dense Jacobians and dense solves remain cheap.

### Best next steps

1. Commit and push M126.
2. Add a user-facing example or developer doc snippet showing the tiny-grid
   implicit sensitivity workflow.
3. Start a lineax or matrix-free variant of the residual linear solve so the
   same API shape can scale beyond dense validation grids.
4. Then wrap a small converged fixed-boundary solve with a custom implicit VJP
   once the linear solve path is selected.

### Completion percentages after M126

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `89%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `84%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `92%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `48%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `85%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `85%`.
- PR merge readiness overall: `95%`.

### User input needed

No user input is needed.

---
## 127. Mirror Implicit Sensitivity Example

### Steps taken

- Added root-level example `examples/mirror_implicit_sensitivity.py`.
- The example:
  - builds a tiny axisymmetric reduced-coordinate mirror state;
  - manufactures an exact reduced root using a linear source and small state
    ridge;
  - computes dense implicit sensitivity with
    `axisym_reduced_residual_linear_solve_jax`;
  - independently solves a perturbed source problem with SciPy root;
  - compares finite-difference and implicit sensitivities;
  - writes JSON metrics and, unless `--no-plots` is passed, a component
    comparison plot.
- Added the example to `examples/mirror/README.md`.
- Added a smoke test in `tests/mirror/test_mirror_examples.py`.
- Updated the mirror overview to list the implicit-sensitivity example.
- Split root reporting into:
  - `perturbed_root_solver_success`: raw SciPy progress flag;
  - `perturbed_root_success`: residual-based acceptance.

### Results obtained

Plotted evidence command:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 python examples/mirror_implicit_sensitivity.py \
  --outdir results/mirror/implicit_sensitivity_m127
```

Metrics:

- vector size: `45`;
- epsilon: `1e-5`;
- state ridge: `1e-3`;
- exact manufactured-root residual norm: `0.0`;
- perturbed residual norm: `1.4596461408271606e-15`;
- relative sensitivity error: `1.30713025011797e-6`;
- max absolute sensitivity error: `0.0011775265450486572`;
- residual-based acceptance: `true`;
- raw SciPy solver success: `false`, due to progress-stall detection at machine
  precision despite tiny residual.

Rendered ignored plot:

- `results/mirror/implicit_sensitivity_m127/figures/mirror_implicit_sensitivity_components.png`.

The plot overlays implicit and finite-difference sensitivity components and
shows the small difference vector beneath them.

### How it was tested

Commands run:

```bash
python -m ruff check examples/mirror_implicit_sensitivity.py tests/mirror/test_mirror_examples.py vmec_jax/mirror/solvers/fixed_boundary/reduced.py
python -m ruff format --check examples/mirror_implicit_sensitivity.py tests/mirror/test_mirror_examples.py vmec_jax/mirror/solvers/fixed_boundary/reduced.py
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_examples.py::test_root_implicit_sensitivity_example_runs_without_plots tests/mirror/test_mirror_fixed_boundary_axisym.py::test_reduced_implicit_sensitivity_matches_manufactured_source_finite_difference -q
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 python examples/mirror_implicit_sensitivity.py --outdir results/mirror/implicit_sensitivity_m127
```

Results:

- Ruff lint passed.
- Ruff format check passed.
- Focused tests: `2 passed`.
- The plotted example completed and wrote accepted metrics.

### File structure and best-practice notes

- The example is root-level, matching the other research fixtures.
- The example output remains under ignored `results/`.
- The example is explicit that this is a tiny reduced validation problem, not a
  production equilibrium solve.
- It provides a user-facing bridge from the low-level residual/Jacobian helpers
  toward future custom implicit solved-state APIs.

### Best next steps

1. Commit and push M127.
2. Add a short differentiability section to the docs that explains the intended
   progression: dense validation -> matrix-free/lineax solve -> custom implicit
   VJP.
3. Start the scalable linear-solve abstraction behind the current dense helper,
   preserving the same forward/transpose solve semantics.
4. Keep checking CI periodically for concrete failures, but do not block on it.

### Completion percentages after M127

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `89%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `85%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `92%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `52%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `85%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `85%`.
- PR merge readiness overall: `95%`.

### User input needed

No user input is needed.

---
## 128. Mirror Differentiability Documentation

### Steps taken

- Added the mirror differentiability page to the mirror documentation toctree.
- Documented the intended split between:
  - fast CLI/example workflows, which may use NumPy/SciPy/Matplotlib for low
    runtime and memory;
  - research differentiable APIs, which should keep residuals,
    linearizations, and derivative rules in JAX and avoid differentiating
    through long host-side optimizer loops.
- Recorded the current reduced-coordinate API:
  - `axisym_reduced_residual_jax`;
  - `axisym_reduced_residual_jacobian_jax`;
  - `axisym_reduced_residual_linear_solve_jax`.
- Documented the validation ladder from dense tiny-grid reference solves to a
  scalable matrix-free/lineax solve and then to a custom implicit derivative
  around a converged solved state.

### Results obtained

- The differentiability lane is now discoverable from `docs/mirror/index.rst`.
- The docs state clearly that the current dense implicit-sensitivity machinery
  is a correctness gate, not yet a production differentiable equilibrium solve.
- The plan now has a finite next step for the differentiability lane: add a
  scalable linear-solve abstraction that preserves the existing forward and
  transpose solve semantics.

### How it was tested

Commands run:

```bash
python -m sphinx -W -b html docs docs/_build/html
git diff --check
python - <<'PY'
import re
from pathlib import Path
text = Path("plan_mirror.md").read_text()
nums = [int(m.group(1)) for m in re.finditer(r"^## (\\d+)\\.", text, flags=re.M)]
print("milestones", len(nums), "last", nums[-1], "monotonic", nums == sorted(nums))
PY
```

Results:

- Sphinx docs build passed with warnings treated as errors.
- Whitespace check passed.
- Plan milestone numbering remained monotonic.
- I also tried `python -m ruff check docs/mirror/index.rst
  docs/mirror/differentiability.rst`; this is not an applicable docs check in
  this repo because `ruff` parses `.rst` files as Python source.

### File structure and best-practice notes

- The docs page lives under `docs/mirror/` with the rest of the mirror user and
  developer documentation.
- The root examples remain in `examples/`; generated metrics and figures remain
  under ignored `results/`.
- The docs keep the production claim narrow: reduced, axisymmetric, tiny-grid
  differentiability validation is promoted; full solved-state differentiability
  remains a planned lane.

### Best next steps

1. Commit and push M128.
2. Add a scalable linear-solve abstraction behind
   `axisym_reduced_residual_linear_solve_jax`.
3. Validate the scalable solve against the dense reference in both forward and
   transpose modes.
4. Use that abstraction as the bridge toward custom implicit solved-state
   derivatives.

### Completion percentages after M128

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `89%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `85%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `92%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `54%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `85%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `85%`.
- PR merge readiness overall: `95%`.

### User input needed

No user input is needed.

---
## 129. Matrix-Free Reduced Implicit Linear Solve Gate

### Steps taken

- Added `axisym_reduced_residual_matvec_jax`, a JAX Hessian-vector product for
  the reduced axisymmetric mirror residual.
- Extended `axisym_reduced_residual_linear_solve_jax` with a method dispatch:
  - `method="dense"` keeps the existing dense Jacobian reference solve;
  - `method="matrix_free_cg"` uses `jax.scipy.sparse.linalg.cg` with the
    matrix-free Hessian-vector product.
- Kept the forward and transpose call shape the same as the dense helper. The
  current reduced residual is an energy gradient, so the Hessian gate is
  symmetric and the transpose operator is identical at this stage.
- Added explicit shape checks for the right-hand side, matvec direction, and
  optional CG initial guess.
- Exported the new matvec helper through the mirror public API.
- Updated the differentiability docs to describe the dense reference and
  matrix-free CG validation path.
- Extended the reduced fixed-boundary test to compare:
  - matrix-free Hessian-vector products against dense Jacobian products;
  - matrix-free CG solves against dense solves in both forward and transpose
    modes on a ridge-stabilized tiny problem.

### Results obtained

- The differentiability lane now has the first scalable linear-operator path
  without giving up the dense tiny-grid reference.
- The CG path is deliberately introduced as a validation gate, not a production
  solved-state derivative claim.
- The public API now exposes the operator needed for future lineax, custom VJP,
  or adjoint wrappers while keeping the implementation inside
  `vmec_jax.mirror.solvers.fixed_boundary.reduced`.

### How it was tested

Commands run:

```bash
python -m ruff check vmec_jax/mirror/solvers/fixed_boundary/reduced.py vmec_jax/mirror/api.py vmec_jax/mirror/__init__.py tests/mirror/test_mirror_fixed_boundary_axisym.py
python -m ruff format --check vmec_jax/mirror/solvers/fixed_boundary/reduced.py vmec_jax/mirror/api.py vmec_jax/mirror/__init__.py tests/mirror/test_mirror_fixed_boundary_axisym.py
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_fixed_boundary_axisym.py::test_reduced_jax_residual_and_jacobian_match_gradient_and_jvp tests/mirror/test_mirror_fixed_boundary_axisym.py::test_reduced_implicit_sensitivity_matches_manufactured_source_finite_difference -q
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_fixed_boundary_axisym.py -q
python -m sphinx -W -b html docs docs/_build/html
git diff --check
python - <<'PY'
import re
from pathlib import Path
text = Path("plan_mirror.md").read_text()
nums = [int(m.group(1)) for m in re.finditer(r"^## (\\d+)\\.", text, flags=re.M)]
print("milestones", len(nums), "last", nums[-1], "monotonic", nums == sorted(nums))
PY
```

Results:

- Ruff lint passed.
- Ruff format check passed.
- Focused reduced differentiability tests: `2 passed`.
- Full axisymmetric fixed-boundary test file: `16 passed`.
- Sphinx docs build passed with warnings treated as errors.
- Whitespace check passed.
- Plan milestone numbering remained monotonic.

### File structure and best-practice notes

- The dense solve, matrix-free matvec, and CG solve stay in the reduced
  fixed-boundary solver module because they operate on the same reduced state
  layout.
- The implementation keeps dense and matrix-free methods under one public solve
  function instead of creating a parallel helper stack.
- The matrix-free path uses JAX primitives, so it is compatible with future
  implicit-differentiation wrappers and avoids differentiating through long
  nonlinear host-side solve loops.
- The test uses a ridge-stabilized tiny problem, making the CG gate stable and
  cheap while preserving the dense reference comparison.

### Best next steps

1. Commit and push M129.
2. Add a small solved-state implicit wrapper around the reduced fixed-boundary
   residual that can call either dense or matrix-free linear solves.
3. Validate that wrapper with the manufactured source example and then a tiny
   converged fixed-boundary solve.
4. Begin benchmarking the matrix-free path on larger `ns`/`nxi` reduced grids
   before promoting it beyond validation status.

### Completion percentages after M129

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `90%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `86%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `92%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `59%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `85%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `85%`.
- PR merge readiness overall: `95%`.

### User input needed

No user input is needed.

---
## 130. Reduced Implicit Forward and Adjoint Wrappers

### Steps taken

- Added `axisym_reduced_implicit_state_sensitivity_jax`, a small forward
  implicit wrapper for

```text
F_x dx/dp = -F_p
```

- Added `axisym_reduced_implicit_adjoint_jax`, an adjoint wrapper for

```text
F_x^T adjoint = dL/dx
```

- Both wrappers delegate to `axisym_reduced_residual_linear_solve_jax`, so they
  can use either the dense reference solve or the matrix-free CG method.
- Updated the manufactured reduced sensitivity test to use the forward wrapper
  instead of calling the raw linear solve directly.
- Added wrapper parity checks against the dense reference for both forward
  sensitivity and adjoint solves on the ridge-stabilized matrix-free test
  problem.
- Updated `examples/mirror_implicit_sensitivity.py` to use the forward wrapper
  and added `--solve-method dense|matrix_free_cg`.
- Updated mirror differentiability docs and example README text.

### Results obtained

- The differentiability lane now has explicit forward and adjoint implicit
  helper APIs matching the equations documented in the plan.
- The root-level implicit-sensitivity example exercises the intended wrapper
  API, while still preserving the dense tiny-grid reference as the default.
- The same example now also completes with `--solve-method matrix_free_cg`.

Dense example metrics:

- vector size: `45`;
- solve method: `dense`;
- root residual norm: `0.0`;
- perturbed residual norm: `1.4596461408271606e-15`;
- relative sensitivity error: `1.30713025011797e-06`;
- max absolute sensitivity error: `0.0011775265450486572`;
- accepted: `true`.

Matrix-free example metrics:

- vector size: `45`;
- solve method: `matrix_free_cg`;
- root residual norm: `0.0`;
- perturbed residual norm: `9.751144411077966e-16`;
- relative sensitivity error: `1.3071302524058876e-06`;
- max absolute sensitivity error: `0.0011775265453781714`;
- accepted: `true`.

### How it was tested

Commands run:

```bash
python -m ruff check vmec_jax/mirror/solvers/fixed_boundary/reduced.py vmec_jax/mirror/api.py vmec_jax/mirror/__init__.py tests/mirror/test_mirror_fixed_boundary_axisym.py examples/mirror_implicit_sensitivity.py
python -m ruff format --check vmec_jax/mirror/solvers/fixed_boundary/reduced.py vmec_jax/mirror/api.py vmec_jax/mirror/__init__.py tests/mirror/test_mirror_fixed_boundary_axisym.py examples/mirror_implicit_sensitivity.py
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_fixed_boundary_axisym.py::test_reduced_jax_residual_and_jacobian_match_gradient_and_jvp tests/mirror/test_mirror_fixed_boundary_axisym.py::test_reduced_implicit_sensitivity_matches_manufactured_source_finite_difference tests/mirror/test_mirror_examples.py::test_root_implicit_sensitivity_example_runs_without_plots -q
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 python examples/mirror_implicit_sensitivity.py --no-plots --outdir results/mirror/implicit_sensitivity_m130_dense
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 python examples/mirror_implicit_sensitivity.py --solve-method matrix_free_cg --no-plots --outdir results/mirror/implicit_sensitivity_m130_matrix_free
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_fixed_boundary_axisym.py -q
python -m sphinx -W -b html docs docs/_build/html
git diff --check
python - <<'PY'
import re
from pathlib import Path
text = Path("plan_mirror.md").read_text()
nums = [int(m.group(1)) for m in re.finditer(r"^## (\\d+)\\.", text, flags=re.M)]
print("milestones", len(nums), "last", nums[-1], "monotonic", nums == sorted(nums))
PY
```

Results:

- Ruff lint passed.
- Ruff format check passed after formatting the reduced solver module.
- Focused wrapper and example tests: `3 passed`.
- Dense no-plot example completed and wrote accepted metrics.
- Matrix-free no-plot example completed and wrote accepted metrics.
- Full axisymmetric fixed-boundary test file: `16 passed`.
- Sphinx docs build passed with warnings treated as errors.
- Whitespace check passed.
- Plan milestone numbering remained monotonic.

### File structure and best-practice notes

- The wrappers stay in the reduced fixed-boundary module because they are thin
  equation-level helpers around the reduced residual and linear solve.
- The example now depends on the wrapper API, not a low-level solve primitive.
- The wrappers do not claim a production differentiable equilibrium solve; they
  expose the equations needed for the next custom implicit derivative gate.
- Generated example outputs remain under ignored `results/`.

### Best next steps

1. Commit and push M130.
2. Add a tiny converged fixed-boundary solved-state validation that uses the
   wrapper on an actual solver result instead of a manufactured exact root.
3. Benchmark dense versus matrix-free wrapper calls over a small `ns`/`nxi`
   grid ladder and record runtime/memory trends under ignored `results/`.
4. Then decide whether to add a custom implicit derivative wrapper around the
   solved-state API or keep the explicit helper route until the physical
   fixed-boundary solve is tighter.

### Completion percentages after M130

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `90%`.
- Fixed-boundary axisymmetric solve: `89%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `87%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `92%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `64%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `85%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `85%`.
- PR merge readiness overall: `95%`.

### User input needed

No user input is needed.

---
## 131. Tiny Solved-State Implicit Sensitivity Gate

### Steps taken

- Added a solved-state finite-difference validation for the reduced implicit
  wrapper.
- The test first runs an actual tiny fixed-boundary residual-Newton solve for a
  perturbed constant-radius cylinder.
- It then packs the converged solver result into reduced coordinates and checks
  the JAX reduced residual at that state.
- A source perturbation direction `q` is applied through the residual source
  term.
- The forward implicit wrapper computes the predicted sensitivity using

```text
F_x dx/dp = q
```

  because the source perturbation enters the residual as `F - p q`.
- The validation then solves the perturbed sourced nonlinear problem with
  SciPy root and compares the finite-difference state change against the
  wrapper sensitivity.
- A small local `state_ridge=1e-3` about the solved state is used in both the
  wrapper and the perturbed nonlinear residual. The unregularized tiny
  zero-pressure cylinder Hessian is singular enough to produce non-finite dense
  solves, so the ridge is an explicit validation regularization, not a hidden
  production claim.
- Updated the differentiability docs to state that the wrapper is now tested on
  both manufactured roots and a tiny converged fixed-boundary state.

### Results obtained

- The tiny residual-Newton solve reached final residual
  `2.8151371433923005e-17` in the prototype run.
- The reduced residual at the packed solved state was below `1e-12` in the
  test.
- The perturbed sourced root solve converged with residual below `1e-10`.
- The finite-difference sensitivity agrees with the implicit wrapper within
  the focused tolerance (`rtol=5e-4`, `atol=5e-3`).
- The differentiability lane now has evidence on an actual solver result, not
  only a manufactured exact root.

### How it was tested

Commands run:

```bash
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_fixed_boundary_axisym.py::test_reduced_implicit_sensitivity_matches_tiny_solved_state_source_finite_difference -q
python -m ruff check tests/mirror/test_mirror_fixed_boundary_axisym.py
python -m ruff format --check tests/mirror/test_mirror_fixed_boundary_axisym.py
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_fixed_boundary_axisym.py -q
python -m sphinx -W -b html docs docs/_build/html
git diff --check
python - <<'PY'
import re
from pathlib import Path
text = Path("plan_mirror.md").read_text()
nums = [int(m.group(1)) for m in re.finditer(r"^## (\\d+)\\.", text, flags=re.M)]
print("milestones", len(nums), "last", nums[-1], "monotonic", nums == sorted(nums))
PY
```

Results:

- New solved-state implicit sensitivity test: `1 passed`.
- Ruff lint passed for the Python test file.
- Ruff format check passed for the Python test file.
- Full axisymmetric fixed-boundary test file: `17 passed`.
- Sphinx docs build passed with warnings treated as errors.
- Whitespace check passed.
- Plan milestone numbering remained monotonic.
- I also tried `python -m ruff check ... docs/mirror/differentiability.rst`;
  this is not an applicable docs check in this repo because `ruff` parses `.rst`
  files as Python source.

### File structure and best-practice notes

- The new validation lives beside the other reduced axisymmetric solver tests,
  since it exercises the same reduced state layout and residual API.
- The state ridge is visible in the test and docs so the validation does not
  overstate the conditioning of the unregularized tiny physical Hessian.
- No generated artifacts are added to git.

### Best next steps

1. Commit and push M131.
2. Add a lightweight dense-versus-matrix-free wrapper benchmark over a small
   `ns`/`nxi` ladder, writing JSON/CSV under ignored `results/`.
3. Use that benchmark to decide the default method for future differentiable
   solved-state experiments.
4. Continue toward the remaining open lanes after the differentiability gates:
   toroidal hybrid VMEC2000 parity/refinement and circular-coil beta-scan
   hardening.

### Completion percentages after M131

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `90%`.
- Fixed-boundary axisymmetric solve: `90%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `88%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `92%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `68%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `85%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `85%`.
- PR merge readiness overall: `95%`.

### User input needed

No user input is needed.

---
## 132. Dense vs Matrix-Free Implicit Solve Benchmark Example

### Steps taken

- Added root-level example `examples/mirror_implicit_solve_benchmark.py`.
- The example benchmarks the forward implicit wrapper over a small
  `ns`/`nxi` ladder and compares:
  - dense reference solves;
  - matrix-free JAX CG solves;
  - relative solution error against dense;
  - relative linear residual;
  - runtime;
  - Python-side peak memory from `tracemalloc`.
- The example writes:
  - `mirror_implicit_solve_benchmark_metrics.json`;
  - `mirror_implicit_solve_benchmark.csv`;
  - optional `figures/mirror_implicit_solve_benchmark.png`.
- Added the benchmark to `examples/mirror/README.md`.
- Added a no-plot smoke test that runs the smallest `ns=5`, `nxi=7` row and
  validates the JSON/CSV output.
- Updated the differentiability docs to identify this example as the
  runtime/memory comparison fixture.

### Results obtained

Plotted benchmark command:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 python examples/mirror_implicit_solve_benchmark.py \
  --ns-array 5,7 \
  --nxi-array 7 \
  --repeat 1 \
  --outdir results/mirror/implicit_solve_benchmark_m132
```

Rendered ignored plot:

- `results/mirror/implicit_solve_benchmark_m132/figures/mirror_implicit_solve_benchmark.png`.

Metrics from the plotted run:

- `ns=5`, `nxi=7`, vector size `45`:
  - dense runtime mean: `0.2960055838339031` s;
  - matrix-free runtime mean: `0.872910083970055` s;
  - matrix-free relative error vs dense:
    `8.851680334241526e-09`;
  - matrix-free relative linear residual:
    `5.872656576175309e-09`.
- `ns=7`, `nxi=7`, vector size `67`:
  - dense runtime mean: `0.30951612489297986` s;
  - matrix-free runtime mean: `0.855826040962711` s;
  - matrix-free relative error vs dense:
    `5.009464861946238e-09`;
  - matrix-free relative linear residual:
    `4.134320052901546e-09`.

At these tiny sizes, dense is faster and has lower Python-side peak memory.
That is expected and useful: dense remains the correctness and small-grid
reference, while matrix-free is validated for accuracy before larger-grid
benchmarking.

### How it was tested

Commands run:

```bash
python -m ruff check examples/mirror_implicit_solve_benchmark.py tests/mirror/test_mirror_examples.py
python -m ruff format --check examples/mirror_implicit_solve_benchmark.py tests/mirror/test_mirror_examples.py
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_examples.py::test_root_implicit_solve_benchmark_runs_without_plots -q
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 python examples/mirror_implicit_solve_benchmark.py --ns-array 5,7 --nxi-array 7 --repeat 1 --outdir results/mirror/implicit_solve_benchmark_m132
python -m sphinx -W -b html docs docs/_build/html
git diff --check
python - <<'PY'
import re
from pathlib import Path
text = Path("plan_mirror.md").read_text()
nums = [int(m.group(1)) for m in re.finditer(r"^## (\\d+)\\.", text, flags=re.M)]
print("milestones", len(nums), "last", nums[-1], "monotonic", nums == sorted(nums))
PY
```

Results:

- Ruff lint passed.
- Ruff format check passed.
- Benchmark smoke test: `1 passed`.
- Plotted benchmark completed and wrote accepted metrics.
- Sphinx docs build passed with warnings treated as errors.
- Whitespace check passed.
- Plan milestone numbering remained monotonic.
- Visual check: the benchmark plot renders with runtime, Python peak memory,
  and relative-error panels.

### File structure and best-practice notes

- The benchmark is root-level because it is a user-facing research fixture, like
  the other mirror benchmark examples.
- The benchmark uses the public wrapper API rather than internal dense helper
  calls.
- Generated JSON/CSV/PNG outputs remain under ignored `results/`.
- The example keeps memory reporting honest by labeling it as Python-side peak
  memory; it is not a complete device-memory profiler.

### Best next steps

1. Commit and push M132.
2. Extend this benchmark only after conditioning/preconditioning is good enough
   to make larger low-ridge matrix-free rows meaningful.
3. Return to the non-differentiability lanes with the largest remaining gaps:
   toroidal hybrid VMEC2000 parity/refinement and circular-coil beta-scan
   hardening.
4. Keep the differentiability lane open for a later custom implicit derivative
   wrapper once the physical solved-state residual conditioning is improved.

### Completion percentages after M132

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `90%`.
- Fixed-boundary axisymmetric solve: `90%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `92%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `72%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `85%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `85%`.
- PR merge readiness overall: `95%`.

### User input needed

No user input is needed.

---
## 133. Circular-Coil Beta-Scan Schema 0.2 Hardening

### Steps taken

- Bumped the circular-coil beta-scan metrics schema from `0.1` to `0.2`.
- Added aggregate workflow fields to the required top-level JSON contract:
  - `fixed_boundary_baseline_count`;
  - `lcfs_pilot_requested`;
  - `lcfs_pilot_steps_requested`;
  - `lcfs_pilot_target_merit`;
  - `lcfs_pilot_stagnation_rtol`;
  - `lcfs_pilot_fsq_growth_limit`;
  - `lcfs_pilot_rows_total`;
  - `lcfs_pilot_accepted_rows_total`;
  - `lcfs_pilot_skipped_rows_total`;
  - `lcfs_pilot_stop_reason_counts`.
- Added enumerated workflow/free-boundary status values to the schema helper.
- Hardened `validate_circular_coil_beta_scan_metrics` so it checks:
  - workflow status is known;
  - free-boundary status is known;
  - requested beta list and beta-case list have the same length;
  - `fixed_boundary_baseline_count` matches the baseline row count;
  - aggregate pilot row totals match nested pilot rows;
  - aggregate stop-reason counts match nested pilot rows.
- Updated `examples/mirror/README.md` to document schema version `0.2`.
- Extended the circular-coil example test to assert schema `0.2` and verify
  that count mismatches are rejected.

### Results obtained

- The beta-scan JSON contract now covers the aggregate fields downstream ESSOS
  comparison scripts need for quick validation.
- The validator catches mismatches between top-level pilot summaries and nested
  pilot rows.
- Existing circular-coil example paths still pass with the stricter validator.

### How it was tested

Commands run:

```bash
python -m ruff check examples/mirror_free_boundary_circular_coils.py tests/mirror/test_mirror_examples.py
python -m ruff format --check examples/mirror_free_boundary_circular_coils.py tests/mirror/test_mirror_examples.py
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_example_runs_without_plots -q
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_examples.py -k free_boundary_circular_coils -q
python -m sphinx -W -b html docs docs/_build/html
git diff --check
python - <<'PY'
import re
from pathlib import Path
text = Path("plan_mirror.md").read_text()
nums = [int(m.group(1)) for m in re.finditer(r"^## (\\d+)\\.", text, flags=re.M)]
print("milestones", len(nums), "last", nums[-1], "monotonic", nums == sorted(nums))
PY
```

Results:

- Ruff lint passed.
- Ruff format check passed.
- Main circular-coil example smoke test: `1 passed`.
- Circular-coil example subset: `5 passed, 12 deselected`.
- Sphinx docs build passed with warnings treated as errors.
- Whitespace check passed.
- Plan milestone numbering remained monotonic.

### File structure and best-practice notes

- The schema remains local to the root example because it describes that
  example's planning-fixture metrics, not the mirror `mout` format.
- The stricter validator is still lightweight and avoids JSON-schema
  dependencies.
- No generated artifacts are added to git.

### Best next steps

1. Commit and push M133.
2. Add a compact top-level beta-scan status plot/table field for final trial vs
   last accepted state if downstream ESSOS reporting needs a single field.
3. Re-run a higher-budget 1%, 3%, and 10% pilot scan when compute time is
   available and record the accepted/rejected pilot behavior under schema
   `0.2`.
4. Continue toroidal-hybrid refinement only where it adds new evidence beyond
   the existing VMEC2000 80-row parity run.

### Completion percentages after M133

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `90%`.
- Fixed-boundary axisymmetric solve: `90%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `92%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `72%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `86%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `88%`.
- PR merge readiness overall: `95%`.

### User input needed

No user input is needed.

---
## 134. Circular-Coil Beta-Scan Embedded Summary Rows

### Steps taken

- Bumped the circular-coil beta-scan metrics schema from `0.2` to `0.3`.
- Added required top-level `summary_rows`.
- `summary_rows` contains the same compact baseline / last-accepted /
  final-trial report table that is written to
  `free_boundary_circular_coils_beta_scan_summary.csv`.
- Updated `validate_circular_coil_beta_scan_metrics` so it checks:
  - `summary_rows` has one row per fixed-boundary baseline row;
  - every summary row contains all `report_fields`.
- Updated the CSV writer path to write the already embedded `summary_rows`,
  keeping JSON and CSV reports in sync.
- Updated `examples/mirror/README.md` to document schema version `0.3` and the
  embedded summary table.
- Extended the circular-coil tests to validate `summary_rows`, count mismatch
  rejection, and JSON/CSV consistency.

### Results obtained

- Downstream ESSOS comparison tooling can now consume one JSON metrics file
  without needing to open the sidecar CSV for the compact summary table.
- The CSV remains available for spreadsheets and quick reports.
- Existing circular-coil example tests pass with the stricter schema.

### How it was tested

Commands run:

```bash
python -m ruff check examples/mirror_free_boundary_circular_coils.py tests/mirror/test_mirror_examples.py
python -m ruff format --check examples/mirror_free_boundary_circular_coils.py tests/mirror/test_mirror_examples.py
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_examples.py -k free_boundary_circular_coils -q
python -m sphinx -W -b html docs docs/_build/html
git diff --check
python - <<'PY'
import re
from pathlib import Path
text = Path("plan_mirror.md").read_text()
nums = [int(m.group(1)) for m in re.finditer(r"^## (\\d+)\\.", text, flags=re.M)]
print("milestones", len(nums), "last", nums[-1], "monotonic", nums == sorted(nums))
PY
```

Results:

- Ruff lint passed.
- Ruff format check passed.
- Circular-coil example subset: `5 passed, 12 deselected`.
- Sphinx docs build passed with warnings treated as errors.
- Whitespace check passed.
- Plan milestone numbering remained monotonic.

### File structure and best-practice notes

- The embedded summary is produced by the existing
  `circular_coil_beta_scan_report_rows` helper, avoiding a second table
  construction path.
- The JSON remains self-contained while preserving the CSV convenience output.
- The schema stays local to the planning fixture and avoids external
  validation dependencies.

### Best next steps

1. Commit and push M134.
2. Re-run a higher-budget schema `0.3` beta scan when compute time is available
   to capture accepted/rejected behavior in the embedded summary table.
3. Add docs for interpreting `summary_rows` if external ESSOS scripts start
   consuming it directly.
4. Continue final lane cleanup and CI-failure checks as PR #21 updates run.

### Completion percentages after M134

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `90%`.
- Fixed-boundary axisymmetric solve: `90%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `92%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `72%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `87%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `90%`.
- PR merge readiness overall: `95%`.

### User input needed

No user input is needed.

---
## 135. Draft PR Description Synchronization

### Steps taken

- Updated draft PR #21 body on GitHub.
- Replaced the stale statement that detailed logs only ran through section 83
  with the current section 134 state.
- Updated the PR contents list to include:
  - reduced implicit differentiability gates;
  - matrix-free CG and dense-vs-CG benchmark example;
  - circular-coil beta-scan schema `0.3` with embedded `summary_rows`;
  - VMEC2000 parity evidence for the toroidal hybrid fixture.
- Updated the PR checklist so partially complete lanes are explicit:
  - reduced differentiability gates are complete as method gates;
  - production differentiable solved-state API remains open;
  - circular-coil LCFS pilot workflow is complete as a planning fixture;
  - converged free-boundary mirror solve remains open.
- Kept the PR in draft state.

### Results obtained

- PR #21 now matches the current repository and plan status.
- Remote verification confirmed:
  - PR number `21`;
  - draft state `true`;
  - body contains `section 134`;
  - body contains schema `0.3`;
  - body contains `M13f`;
  - body contains the latest local validation section.

### How it was tested

Commands run:

```bash
gh pr edit 21 --body-file /tmp/vmec_mirror_pr_body.md
gh pr view 21 --json number,isDraft,body,url | python -c '...'
git diff --check
python - <<'PY'
import re
from pathlib import Path
text = Path("plan_mirror.md").read_text()
nums = [int(m.group(1)) for m in re.finditer(r"^## (\\d+)\\.", text, flags=re.M)]
print("milestones", len(nums), "last", nums[-1], "monotonic", nums == sorted(nums))
PY
```

Results:

- PR edit succeeded.
- PR body verification passed.
- Whitespace check passed.
- Plan milestone numbering remained monotonic.

### File structure and best-practice notes

- No source code changed in this tranche.
- The PR body remains a concise index; detailed evidence stays in
  `plan_mirror.md`.
- The PR remains draft as requested.

### Best next steps

1. Commit and push M135 plan log.
2. Continue with final open-lane cleanup and inspect CI only when checks appear
   or fail.
3. Prioritize any CI failures over new feature work if GitHub starts reporting
   failing checks.

### Completion percentages after M135

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `90%`.
- Fixed-boundary axisymmetric solve: `90%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `92%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `72%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `87%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `90%`.
- PR merge readiness overall: `96%`.

### User input needed

No user input is needed.

---
## 136. Reduced Source Custom-VJP Solved-State Wrapper

### Steps taken

- Added `axisym_reduced_implicit_source_state_jax`.
- The new API treats the supplied reduced vector as a cached converged
  solution of `F(x, source) = 0`.
- Its primal call returns that vector unchanged.
- Its reverse-mode derivative with respect to `source_vector` solves the same
  implicit adjoint equation used by
  `axisym_reduced_implicit_adjoint_jax`.
- Gradients with respect to the cached solved vector are intentionally zero so
  downstream users do not accidentally differentiate through a host-side
  optimizer loop.
- Exported the wrapper through `vmec_jax.mirror.api` and
  `vmec_jax.mirror`.
- Added a focused unit test that checks:
  - `jax.grad` through the custom VJP matches the explicit adjoint;
  - the directional source gradient matches the explicit forward sensitivity;
  - the same directional derivative matches a separately solved perturbed root.
- Extended `examples/mirror_implicit_sensitivity.py` so the root-level example
  reports:
  - custom-VJP adjoint relative error;
  - custom-VJP directional derivative;
  - finite-difference directional derivative;
  - custom-VJP directional relative error.
- Updated `docs/mirror/differentiability.rst` and
  `examples/mirror/README.md` to describe the cached-state contract.

### Results obtained

- The differentiability lane now has the first custom reverse-mode solved-state
  contract.
- This is still reduced-coordinate, axisymmetric, and source-parameter only;
  physical parameter derivatives remain the next differentiability promotion.
- Dense and matrix-free example runs both accepted the new custom-VJP checks.
- Dense plotted example metrics:
  - `accepted`: `true`;
  - `solve_method`: `dense`;
  - root residual norm: `0.0`;
  - perturbed residual norm: `1.4596461408271606e-15`;
  - forward sensitivity relative error: `1.30713025011797e-06`;
  - custom-VJP adjoint relative error: `0.0`;
  - custom-VJP directional relative error: `1.989642776759042e-06`.
- Matrix-free example metrics:
  - `accepted`: `true`;
  - `solve_method`: `matrix_free_cg`;
  - root residual norm: `0.0`;
  - perturbed residual norm: `9.751144411077966e-16`;
  - forward sensitivity relative error: `1.3071302524058876e-06`;
  - custom-VJP adjoint relative error: `0.0`;
  - custom-VJP directional relative error: `1.9896427732953227e-06`.
- The dense example plot rendered correctly at
  `results/mirror/implicit_sensitivity_m136/figures/mirror_implicit_sensitivity_components.png`
  and remains ignored.

### How it was tested

Commands run:

```bash
python -m ruff format --check vmec_jax/mirror/solvers/fixed_boundary/reduced.py \
  vmec_jax/mirror/api.py vmec_jax/mirror/__init__.py \
  examples/mirror_implicit_sensitivity.py \
  tests/mirror/test_mirror_fixed_boundary_axisym.py \
  tests/mirror/test_mirror_examples.py
python -m ruff check vmec_jax/mirror/solvers/fixed_boundary/reduced.py \
  vmec_jax/mirror/api.py vmec_jax/mirror/__init__.py \
  examples/mirror_implicit_sensitivity.py \
  tests/mirror/test_mirror_fixed_boundary_axisym.py \
  tests/mirror/test_mirror_examples.py
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_fixed_boundary_axisym.py::test_reduced_custom_vjp_source_state_matches_adjoint_and_perturbed_root -q
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_examples.py::test_root_implicit_sensitivity_example_runs_without_plots -q
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_fixed_boundary_axisym.py -q
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_examples.py -k 'implicit_sensitivity or implicit_solve_benchmark' -q
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 python \
  examples/mirror_implicit_sensitivity.py \
  --outdir results/mirror/implicit_sensitivity_m136 --solve-method dense
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 python \
  examples/mirror_implicit_sensitivity.py \
  --outdir results/mirror/implicit_sensitivity_m136_matrix_free \
  --solve-method matrix_free_cg --no-plots
python -m sphinx -W -b html docs docs/_build/html
git diff --check
```

Results:

- Ruff format check passed.
- Ruff lint passed.
- New custom-VJP unit test: `1 passed`.
- Root implicit sensitivity example smoke test: `1 passed`.
- Full fixed-boundary axisymmetric mirror test file: `18 passed`.
- Implicit sensitivity / solve benchmark example subset:
  `2 passed, 15 deselected`.
- Dense plotted example run succeeded.
- Matrix-free no-plot example run succeeded.
- Sphinx docs build passed with warnings treated as errors.
- Whitespace check passed.

### File structure and best-practice notes

- The custom VJP lives next to the existing reduced residual, linear solve,
  forward sensitivity, and adjoint wrappers in
  `vmec_jax/mirror/solvers/fixed_boundary/reduced.py`.
- Public exports stay centralized through `vmec_jax/mirror/api.py` and
  `vmec_jax/mirror/__init__.py`.
- The root example remains in `examples/` because it is a user-facing
  validation workflow.
- Tests stay in the existing mirror fixed-boundary and example test files; no
  reference data files were added.
- Generated metrics and figures stay under ignored `results/`.
- The API is intentionally explicit about cached solved states so the research
  path does not trace long SciPy/CLI optimizer loops.

### Best next steps

1. Commit and push M136.
2. Extend implicit differentiation from the reduced linear source to one
   physical parameter family, starting with pressure-profile coefficients or
   boundary-radius coefficients on tiny grids.
3. Add a small benchmark comparing custom-VJP reverse gradients against forward
   sensitivity contractions for several source directions.
4. Continue free-boundary and ESSOS beta-scan convergence work after the
   differentiability source wrapper is committed.

### Completion percentages after M136

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `90%`.
- Fixed-boundary axisymmetric solve: `90%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `92%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `79%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `87%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `90%`.
- PR merge readiness overall: `96%`.

### User input needed

No user input is needed.

---
## 137. Pressure-Coefficient Implicit Differentiation Gate

### Steps taken

- Added `axisym_reduced_residual_pressure_jacobian_jax`.
- Added `axisym_reduced_implicit_pressure_sensitivity_jax`.
- Added `axisym_reduced_implicit_pressure_state_jax`.
- The pressure state wrapper follows the same cached-solved-state contract as
  the source wrapper:
  - the primal call returns the supplied converged reduced vector;
  - the reverse pass solves the implicit adjoint equation;
  - the pressure-coefficient VJP is `-adjoint.T @ dF/dp_coeffs`.
- Exported the new pressure differentiability functions through
  `vmec_jax.mirror.api` and `vmec_jax.mirror`.
- Added a focused pressure-coefficient regression test that checks:
  - forward and reverse residual pressure Jacobians agree;
  - `jax.grad` through the custom pressure VJP matches `-F_p.T @ adjoint`;
  - the custom VJP directional derivative matches the forward sensitivity
    contraction;
  - the dense and matrix-free pressure sensitivity paths agree;
  - the same directional derivative matches a separately solved perturbed
    pressure-root finite difference.
- Updated `docs/mirror/differentiability.rst` to mark pressure coefficients as
  the first physical-parameter differentiability gate.

### Results obtained

- The differentiability lane now covers both:
  - reduced linear source perturbations;
  - scalar pressure-profile polynomial coefficients.
- The pressure path uses the existing dense and matrix-free linear solve
  machinery rather than adding a separate solver path.
- The new test reaches a perturbed-pressure residual below `1e-10`.
- Full axisymmetric mirror tests pass with the added pressure gate:
  `19 passed`.

### How it was tested

Commands run:

```bash
python -m ruff check vmec_jax/mirror/solvers/fixed_boundary/reduced.py \
  vmec_jax/mirror/api.py vmec_jax/mirror/__init__.py \
  tests/mirror/test_mirror_fixed_boundary_axisym.py
python -m ruff format --check vmec_jax/mirror/solvers/fixed_boundary/reduced.py \
  vmec_jax/mirror/api.py vmec_jax/mirror/__init__.py \
  tests/mirror/test_mirror_fixed_boundary_axisym.py
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_fixed_boundary_axisym.py::test_reduced_pressure_custom_vjp_matches_adjoint_and_perturbed_root -q
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_fixed_boundary_axisym.py -q
python -m sphinx -W -b html docs docs/_build/html
git diff --check
```

Results:

- Ruff lint passed.
- Ruff Python format check passed.
- New pressure custom-VJP test: `1 passed`.
- Full fixed-boundary axisymmetric mirror test file: `19 passed`.
- Sphinx docs build passed with warnings treated as errors.
- Whitespace check passed.
- A transient Ruff format command accidentally included
  `docs/mirror/differentiability.rst`; Ruff cannot parse reStructuredText, so
  that command is not a valid docs check. Sphinx is the docs validation for this
  tranche.

### File structure and best-practice notes

- Pressure differentiability stays in
  `vmec_jax/mirror/solvers/fixed_boundary/reduced.py`, alongside the residual,
  Hessian, linear solve, source sensitivity, and adjoint utilities it reuses.
- Public exports remain centralized in `vmec_jax/mirror/api.py` and
  `vmec_jax/mirror/__init__.py`.
- No generated artifacts or reference files were added.
- The implementation keeps the CLI/host-solver path separate from the
  differentiable research API: users differentiate cached solved states, not
  long optimizer traces.

### Best next steps

1. Commit and push M137.
2. Add one more physical-parameter gate for current or flux-profile
   coefficients, reusing the same pressure-coefficient pattern.
3. Add boundary-parameter implicit differentiation only after deciding whether
   the boundary object should become a JAX pytree or remain a lightweight
   dataclass with explicit coefficient wrappers.
4. Continue final free-boundary and ESSOS beta-scan convergence cleanup after
   the differentiability gates are committed.

### Completion percentages after M137

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `90%`.
- Fixed-boundary axisymmetric solve: `90%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `92%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `83%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `87%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `90%`.
- PR merge readiness overall: `96%`.

### User input needed

No user input is needed.

---
## 138. Generic Profile-Coefficient Differentiability Gate

### Steps taken

- Refactored the pressure-coefficient differentiability implementation into a
  generic profile-coefficient path.
- Added:
  - `axisym_reduced_residual_profile_jacobian_jax`;
  - `axisym_reduced_implicit_profile_sensitivity_jax`;
  - `axisym_reduced_implicit_profile_state_jax`.
- The generic path accepts one selected profile coefficient vector:
  - `pressure`;
  - `i_prime`;
  - `psi_prime`.
- Kept the pressure-specific public functions as thin wrappers around the
  generic implementation so existing callers remain stable.
- Exported the generic profile functions through `vmec_jax.mirror.api` and
  `vmec_jax.mirror`.
- Added a current-profile coefficient regression test using the generic API.
- The current-profile test checks:
  - forward and reverse `dF/dI'_coeffs` Jacobians agree;
  - `jax.grad` through the custom profile VJP matches `-F_p.T @ adjoint`;
  - the custom VJP directional derivative matches the forward sensitivity
    contraction;
  - dense and matrix-free current-profile sensitivities agree;
  - the same directional derivative matches a separately solved perturbed-root
    finite difference.
- Updated `docs/mirror/differentiability.rst` to document the generic profile
  API and current-profile coverage.

### Results obtained

- The profile-coefficient differentiability path is now shared instead of
  duplicated for each profile family.
- Pressure wrappers still pass their focused regression after the refactor.
- Current-profile coefficients now have the same dense, matrix-free, forward,
  and reverse validation pattern as pressure coefficients.
- Full fixed-boundary axisymmetric mirror tests pass with the added generic
  profile gate: `20 passed`.

### How it was tested

Commands run:

```bash
python -m ruff format --check vmec_jax/mirror/solvers/fixed_boundary/reduced.py \
  vmec_jax/mirror/api.py vmec_jax/mirror/__init__.py \
  tests/mirror/test_mirror_fixed_boundary_axisym.py
python -m ruff check vmec_jax/mirror/solvers/fixed_boundary/reduced.py \
  vmec_jax/mirror/api.py vmec_jax/mirror/__init__.py \
  tests/mirror/test_mirror_fixed_boundary_axisym.py
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_fixed_boundary_axisym.py::test_reduced_pressure_custom_vjp_matches_adjoint_and_perturbed_root \
  tests/mirror/test_mirror_fixed_boundary_axisym.py::test_reduced_current_profile_custom_vjp_matches_adjoint_and_perturbed_root -q
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_fixed_boundary_axisym.py -q
python -m sphinx -W -b html docs docs/_build/html
git diff --check
```

Results:

- Ruff Python format check passed.
- Ruff lint passed.
- Focused pressure/current profile differentiability tests: `2 passed`.
- Full fixed-boundary axisymmetric mirror test file: `20 passed`.
- Sphinx docs build passed with warnings treated as errors.
- Whitespace check passed.

### File structure and best-practice notes

- The reusable profile machinery stays in
  `vmec_jax/mirror/solvers/fixed_boundary/reduced.py`, where it can share the
  existing residual, adjoint, and dense/matrix-free linear solve helpers.
- Pressure-specific wrappers are retained for discoverability while avoiding a
  second implementation path.
- The generic public API keeps current, flux, and pressure coefficient
  derivatives under one conceptual entry point.
- No generated files or large artifacts were added.

### Best next steps

1. Commit and push M138.
2. Add a small `psi_prime` profile derivative assertion or include it in the
   current-profile test matrix if runtime remains acceptable.
3. Decide whether boundary-parameter implicit differentiation should use a
   similar explicit-coefficient wrapper or a pytree boundary object.
4. Resume final free-boundary/ESSOS beta-scan cleanup after the remaining
   differentiability profile gate is committed.

### Completion percentages after M138

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `90%`.
- Fixed-boundary axisymmetric solve: `90%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `92%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `86%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `87%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `90%`.
- PR merge readiness overall: `97%`.

### User input needed

No user input is needed.

---
## 139. Flux-Profile Differentiability Coverage and Test Simplification

### Steps taken

- Replaced the current-profile custom-VJP regression with a parameterized
  profile-coefficient test.
- The parameterized test now covers:
  - `i_prime` / current coefficients;
  - `psi_prime` / flux coefficients.
- Kept the same validation gates for each profile:
  - forward and reverse residual profile Jacobians agree;
  - custom VJP gradient matches `-F_p.T @ adjoint`;
  - custom VJP directional derivative matches the forward sensitivity
    contraction;
  - dense and matrix-free profile sensitivities agree;
  - directional derivative matches a separately solved perturbed-root finite
    difference.
- Fixed the new `psi_prime` finite-difference closure to use the perturbed flux
  profile in both residual and Jacobian callbacks.
- Updated `docs/mirror/differentiability.rst` to say pressure, current, and
  flux profile coefficients are now covered.

### Results obtained

- The generic profile API now has explicit regression coverage for all three
  supported profile families:
  - pressure;
  - current / `i_prime`;
  - flux / `psi_prime`.
- The test file avoids a third copied test body by parameterizing the current
  and flux cases.
- Full fixed-boundary axisymmetric mirror tests pass with the added flux gate:
  `21 passed`.

### How it was tested

Commands run:

```bash
python -m ruff check tests/mirror/test_mirror_fixed_boundary_axisym.py
python -m ruff format --check tests/mirror/test_mirror_fixed_boundary_axisym.py
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_fixed_boundary_axisym.py::test_reduced_pressure_custom_vjp_matches_adjoint_and_perturbed_root \
  tests/mirror/test_mirror_fixed_boundary_axisym.py::test_reduced_profile_custom_vjp_matches_adjoint_and_perturbed_root -q
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_fixed_boundary_axisym.py -q
python -m sphinx -W -b html docs docs/_build/html
git diff --check
```

Results:

- Ruff lint passed.
- Ruff Python format check passed.
- Focused pressure/profile differentiability tests: `3 passed`.
- Full fixed-boundary axisymmetric mirror test file: `21 passed`.
- Sphinx docs build passed with warnings treated as errors.
- Whitespace check passed.

### File structure and best-practice notes

- No new source API was needed; this tranche validates the generic profile API
  introduced in M138.
- The profile test is parameterized instead of duplicated, which keeps the test
  body easier to maintain as more profile families are added.
- No generated artifacts were added.

### Best next steps

1. Commit and push M139.
2. Reassess boundary-parameter differentiation: the remaining differentiability
   gap is boundary coefficients, which may need an explicit coefficient wrapper
   rather than making every boundary object a JAX pytree.
3. Resume free-boundary/ESSOS beta-scan convergence cleanup if boundary
   differentiation is deferred to a final polishing tranche.
4. Update the PR description after the next substantial tranche so the
   differentiability status is no longer stale.

### Completion percentages after M139

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `90%`.
- Fixed-boundary axisymmetric solve: `90%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `92%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `87%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `87%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `90%`.
- PR merge readiness overall: `97%`.

### User input needed

No user input is needed.

---
## 140. Polynomial-Boundary Implicit Differentiation Gate

### Steps taken

- Added an optional `boundary_radius` override through the reduced residual,
  Jacobian, Hessian-vector product, linear solve, and implicit adjoint paths.
- Added `axisym_reduced_polynomial_boundary_radius_jax`.
- Added `axisym_reduced_residual_polynomial_boundary_jacobian_jax`.
- Added `axisym_reduced_implicit_polynomial_boundary_sensitivity_jax`.
- Added `axisym_reduced_implicit_polynomial_boundary_state_jax`.
- Exported the polynomial-boundary differentiability functions through
  `vmec_jax.mirror.api` and `vmec_jax.mirror`.
- Added a regression test for polynomial boundary coefficients `[r0, a2, a4]`.
- The test checks:
  - JAX polynomial-radius evaluation matches `MirrorBoundary.polynomial_radius`;
  - forward and reverse boundary residual Jacobians agree;
  - custom boundary VJP gradient matches `-F_b.T @ adjoint`;
  - custom VJP directional derivative matches the forward sensitivity
    contraction;
  - dense and matrix-free boundary sensitivities agree;
  - directional derivative matches a separately solved perturbed-boundary root
    finite difference.
- Updated `docs/mirror/differentiability.rst` to document the boundary
  coefficient API and validation status.

### Results obtained

- The differentiable solved-state lane now covers:
  - reduced source perturbations;
  - pressure coefficients;
  - current coefficients;
  - flux coefficients;
  - axisymmetric polynomial-boundary coefficients.
- Boundary derivatives do not require turning `MirrorBoundary` into a pytree;
  the tested path uses an explicit JAX boundary-radius override.
- The existing static boundary API remains unchanged for CLI and non-AD usage.
- Full fixed-boundary axisymmetric mirror tests pass with the added boundary
  gate: `22 passed`.

### How it was tested

Commands run:

```bash
python -m ruff format --check vmec_jax/mirror/solvers/fixed_boundary/reduced.py \
  vmec_jax/mirror/api.py vmec_jax/mirror/__init__.py \
  tests/mirror/test_mirror_fixed_boundary_axisym.py
python -m ruff check vmec_jax/mirror/solvers/fixed_boundary/reduced.py \
  vmec_jax/mirror/api.py vmec_jax/mirror/__init__.py \
  tests/mirror/test_mirror_fixed_boundary_axisym.py
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_fixed_boundary_axisym.py::test_reduced_polynomial_boundary_custom_vjp_matches_adjoint_and_perturbed_root -q
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_fixed_boundary_axisym.py -q
python -m sphinx -W -b html docs docs/_build/html
git diff --check
```

Results:

- Ruff Python format check passed.
- Ruff lint passed.
- Focused polynomial-boundary differentiability test: `1 passed`.
- Full fixed-boundary axisymmetric mirror test file: `22 passed`.
- Sphinx docs build passed with warnings treated as errors.
- Whitespace check passed.

### File structure and best-practice notes

- Boundary coefficient differentiation stays in
  `vmec_jax/mirror/solvers/fixed_boundary/reduced.py`, because it reuses the
  same reduced residual and implicit linear solve machinery as source/profile
  derivatives.
- The boundary path is explicit and narrow: polynomial axisymmetric boundary
  coefficients only.
- The ordinary `MirrorBoundary` dataclass remains simple and NumPy-oriented.
- No generated artifacts or large files were added.

### Best next steps

1. Commit and push M140.
2. Add a compact benchmark/example comparing profile and boundary custom VJPs
   against forward sensitivity contractions over a tiny grid ladder.
3. Update the draft PR body so it reflects the new differentiability status.
4. Return to the free-boundary/ESSOS beta-scan convergence lane after the PR
   description is synchronized.

### Completion percentages after M140

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `91%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `92%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `90%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `87%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `90%`.
- PR merge readiness overall: `97%`.

### User input needed

No user input is needed.

---
## 141. Implicit Parameter-Gradient Example and Plot

### Steps taken

- Added root-level example `examples/mirror_implicit_parameter_gradients.py`.
- The example manufactures a tiny exact reduced root with:
  - reduced source vector;
  - pressure-profile coefficients;
  - current-profile coefficients;
  - flux-profile coefficients;
  - polynomial-boundary coefficients `[r0, a2, a4]`.
- For each selected parameter family it compares:
  - custom VJP directional derivative;
  - forward implicit sensitivity contraction;
  - separately solved finite-difference perturbed root.
- Added `--families` to run all families or a selected subset.
- Added `--solve-method dense|matrix_free_cg`.
- Added optional plotting of directional gradients and finite-difference
  relative errors.
- Updated `examples/mirror/README.md`.
- Added a root example smoke test in `tests/mirror/test_mirror_examples.py`.
- Rendered the all-family plot and verified it visually.

### Results obtained

- All-family dense example metrics under ignored
  `results/mirror/implicit_parameter_gradients_m141` were accepted.
- Reported families:
  - `source`;
  - `pressure`;
  - `current`;
  - `flux`;
  - `boundary`.
- Root residual norm: `0.0`.
- Custom-VJP vs finite-difference relative errors:
  - source: `1.193623465135548e-06`;
  - pressure: `1.0634644962054567e-05`;
  - current: `4.181993919608112e-12`;
  - flux: `2.6374914201292236e-05`;
  - boundary: `2.0710886826395606e-07`.
- The rendered plot
  `results/mirror/implicit_parameter_gradients_m141/figures/mirror_implicit_parameter_gradients.png`
  is readable after switching the directional-gradient panel to a symlog scale.
- A three-family smoke run under
  `results/mirror/implicit_parameter_gradients_m141_smoke` was also accepted.

### How it was tested

Commands run:

```bash
python -m ruff check examples/mirror_implicit_parameter_gradients.py
python -m ruff format --check examples/mirror_implicit_parameter_gradients.py
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 python \
  examples/mirror_implicit_parameter_gradients.py \
  --outdir results/mirror/implicit_parameter_gradients_m141_smoke \
  --families source,pressure,boundary --no-plots
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 python \
  examples/mirror_implicit_parameter_gradients.py \
  --outdir results/mirror/implicit_parameter_gradients_m141 \
  --solve-method dense
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_examples.py::test_root_implicit_parameter_gradients_example_runs_without_plots -q
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_examples.py \
  -k 'implicit_parameter_gradients or implicit_sensitivity or implicit_solve_benchmark' -q
python -m ruff format --check examples/mirror_implicit_parameter_gradients.py \
  tests/mirror/test_mirror_examples.py
python -m ruff check examples/mirror_implicit_parameter_gradients.py \
  tests/mirror/test_mirror_examples.py
python -m sphinx -W -b html docs docs/_build/html
git diff --check
```

Results:

- Ruff lint passed.
- Ruff Python format check passed.
- Three-family no-plot smoke run accepted.
- All-family plotted example run accepted.
- New root example smoke test: `1 passed`.
- Related implicit example subset: `3 passed, 15 deselected`.
- Sphinx docs build passed with warnings treated as errors.
- Whitespace check passed.
- A transient Ruff format command accidentally included
  `examples/mirror/README.md`; Ruff cannot parse Markdown, so that command is
  not a valid README check.

### File structure and best-practice notes

- The example is in the repository root `examples/` directory, matching the
  existing mirror root examples.
- Generated JSON and figures remain under ignored `results/`.
- The example reuses public mirror APIs and does not introduce new source
  abstractions.
- The plot uses symlog scaling so large source gradients and smaller
  profile/boundary gradients are visible in one figure.

### Best next steps

1. Commit and push M141.
2. Update the draft PR body with the completed source/profile/boundary
   differentiability gates and new example.
3. Check CI once after the push; fix failures if any are reported.
4. Resume free-boundary/ESSOS beta-scan convergence cleanup.

### Completion percentages after M141

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `91%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `92%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `92%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `87%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `90%`.
- PR merge readiness overall: `97%`.

### User input needed

No user input is needed.

---
## 142. Draft PR Description Synchronization After Differentiability Gates

### Steps taken

- Updated draft PR #21 body on GitHub.
- Changed the detailed-log pointer from section 134 to section 141.
- Added the new reduced differentiability status:
  - custom VJP cached-state gates for source parameters;
  - pressure-profile coefficients;
  - current-profile coefficients;
  - flux-profile coefficients;
  - polynomial-boundary coefficients.
- Added the new root-level example
  `examples/mirror_implicit_parameter_gradients.py`.
- Updated the current solver status so it no longer says the reduced
  differentiability path is only planned.
- Kept the PR in draft state.
- Checked PR status after the update.

### Results obtained

- Remote PR body verification confirmed:
  - PR number `21`;
  - draft state `true`;
  - body contains `section 141`;
  - body contains `mirror_implicit_parameter_gradients.py`;
  - body contains `source/profile/polynomial-boundary`;
  - body contains `22 passed`.
- CI snapshot after the push showed checks running/queued and no reported
  failures at that moment.

### How it was tested

Commands run:

```bash
gh pr edit 21 --body-file /tmp/vmec_mirror_pr_body_m141.md
gh pr view 21 --json number,isDraft,body,url | python -c '...'
gh pr view 21 --json number,isDraft,headRefName,statusCheckRollup,url
```

Results:

- PR edit succeeded.
- PR body verification passed.
- PR remained draft.
- CI/status rollup was visible and in progress, with no failure conclusion
  reported in the snapshot.

### File structure and best-practice notes

- No source files changed in this tranche.
- Detailed evidence remains in `plan_mirror.md`; the PR body stays a concise
  review index.
- The PR remains draft as requested.

### Best next steps

1. Commit and push M142 plan log.
2. Continue implementation work without waiting on CI unless a failure appears.
3. Resume free-boundary/ESSOS beta-scan convergence cleanup.
4. Later, refresh the PR body again after the next substantial tranche.

### Completion percentages after M142

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `91%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `92%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `92%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `87%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `90%`.
- PR merge readiness overall: `97%`.

### User input needed

No user input is needed.

---
## 143. Guarded Circular-Coil Free-Boundary Beta-Scan Evidence Run

### Steps taken

- Checked local branch state and draft PR #21 status.
- Confirmed GitHub CLI authentication and checked the latest PR status rollup.
- Ran the circular-coil mirror free-boundary example with:
  - `ntheta=8`;
  - `nxi=11`;
  - `n_segments=64`;
  - fixed-boundary baseline enabled;
  - LCFS pilot enabled for up to five guarded steps;
  - stagnation tolerance `1e-3`;
  - fsq-growth guard `1.1`.
- Validated the emitted metrics with the example's schema validator.
- Inspected the rendered beta-scan summary, horizontal 3D coil/boundary plot,
  LCFS diagnostic, and residual-history plot.

### Results obtained

- Output metrics:
  `results/mirror/free_boundary_circular_coils_m143_schema03_guard1p1/free_boundary_circular_coils_metrics.json`.
- Metrics schema version: `0.3`.
- Workflow status: `lcfs_pilot`.
- Free-boundary status:
  `lcfs_pilot_not_converged_free_boundary`.
- Pilot totals:
  - rows attempted: `5`;
  - accepted rows: `2`;
  - skipped rows: `0`;
  - stop reasons: one `rejected_merit_increase`, two
    `fsq_growth_guard`, and two active rows with no stop reason.
- Per-beta summary:
  - `1%`: fixed-boundary fsq `7.181025259848037e-08`; first LCFS
    trial rejected because the merit increased; final trial fsq-growth ratio
    `12498.631673700875`.
  - `3%`: fixed-boundary fsq `0.004468220052785157`; one accepted LCFS
    update; last accepted fsq `0.004709131909718425`; fsq-growth ratio
    `1.0539167395712978`; next trial stopped by the fsq-growth guard.
  - `10%`: fixed-boundary fsq `0.04263091618797516`; one accepted LCFS
    update; last accepted fsq `0.046665979566329834`; fsq-growth ratio
    `1.0946511062666966`; next trial stopped by the fsq-growth guard.
- The beta-scan summary plot shows that the LCFS pilot improves pressure
  balance and LCFS merit for `3%` and `10%`, but the coupled fixed-boundary
  residual grows enough that the guard correctly rejects further progress.
- The horizontal 3D plot shows the mirror boundary with `z` horizontal and the
  circular coils at both end caps.
- The LCFS diagnostic shows that the current pilot remains mostly a
  scalar-radius pressure-balance update; this is useful but not yet a fully
  coupled free-boundary solve.

### How it was tested

Commands run:

```bash
PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/mirror_free_boundary_circular_coils.py \
    --outdir results/mirror/free_boundary_circular_coils_m143_schema03_guard1p1 \
    --ntheta 8 --nxi 11 --n-segments 64 \
    --run-fixed-boundary-baseline --baseline-maxiter 20 \
    --run-lcfs-pilot --lcfs-pilot-steps 5 \
    --lcfs-pilot-stagnation-rtol 1e-3 \
    --lcfs-pilot-fsq-growth-limit 1.1

python - <<'PY'
import json, runpy
from pathlib import Path
m = json.loads(Path(
    "results/mirror/free_boundary_circular_coils_m143_schema03_guard1p1/"
    "free_boundary_circular_coils_metrics.json"
).read_text())
mod = runpy.run_path("examples/mirror_free_boundary_circular_coils.py")
mod["validate_circular_coil_beta_scan_metrics"](m)
print("validated", m["metrics_schema_version"], len(m["summary_rows"]))
PY

find results/mirror/free_boundary_circular_coils_m143_schema03_guard1p1/figures \
  -maxdepth 3 -type f -name '*.png' -print | sort

gh auth status
gh pr view 21 --json number,isDraft,headRefName,headRefOid,url,statusCheckRollup
```

Results:

- The example completed and wrote the metrics JSON.
- The metrics validator passed with `validated 0.3 3`.
- The expected figure tree was present, including:
  - beta-scan summary;
  - axis `B_z`;
  - boundary `|B|`;
  - horizontal coil/boundary geometry;
  - per-beta fixed-boundary diagnostics;
  - per-step LCFS diagnostics.
- The PR remains draft.
- The CI snapshot showed completed checks green so far, with other checks still
  running and no failure conclusion reported in that snapshot.

### File structure and best-practice notes

- No generated metrics or figures were added to git; they remain under ignored
  `results/`.
- The root example remains the single entry point for the ESSOS-style circular
  coil beta scan.
- The schema validator remains inside the example so downstream scripts and
  tests can reuse the same contract.
- The plot set remains organized by run and by beta/LCFS pilot step, which keeps
  root-level examples clean while still making diagnostics easy to inspect.

### Best next steps

1. Commit and push this M143 plan log.
2. Add or improve the coupled free-boundary update strategy so that pressure
   balance improvement and reduced-equilibrium residual reduction are solved
   together instead of competing under a guard.
3. Add a local regression that exercises a small guarded LCFS pilot and asserts
   the schema, status fields, and at least one accepted/rejected update path.
4. Keep the final production beta-scan evidence ignored in `results/`, with
   compressed figures only if any image must be included in docs.

### Completion percentages after M143

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `91%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `92%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `92%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `88%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `91%`.
- PR merge readiness overall: `97%`.

### User input needed

No user input is needed.

---
## 144. Circular-Coil Beta-Scan Schema Consistency Hardening

### Steps taken

- Hardened `validate_circular_coil_beta_scan_metrics` so it now checks:
  - each baseline row's `lcfs_pilot_rows` value is a JSON array;
  - per-beta LCFS pilot scalar fields match the nested pilot rows;
  - compact `summary_rows` match the fixed-boundary baseline rows;
  - corrupted generated summary values are rejected instead of only checking
    field presence.
- Added regression assertions that deliberately corrupt:
  - a summary-row value;
  - a per-beta `lcfs_pilot_rows_count` aggregate.
- Tightened the accepted-then-rejected fsq-guard test so the compact summary
  row must preserve both:
  - the last accepted pilot state;
  - the final rejected trial state.
- Revalidated the M143 metrics JSON with the hardened validator.
- Ran targeted and full mirror example test suites.

### Results obtained

- The schema validator now catches stale or hand-edited ESSOS comparison rows
  that no longer match the detailed nested baseline/pilot rows.
- The tolerant fsq-guard path is now explicitly covered as:
  - one accepted pilot update;
  - one rejected follow-up trial;
  - `summary_rows[0].last_accepted_*` pointing to the accepted step;
  - `summary_rows[0].final_trial_*` pointing to the rejected trial.
- The existing M143 metrics file still validates under the hardened schema:
  `validated current M143 metrics with hardened schema 0.3 3`.

### How it was tested

Commands run:

```bash
python -m ruff format examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_examples.py
python -m ruff check examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_examples.py

JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_example_runs_without_plots \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_fsq_growth_guard_rejects_pilot \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_tolerant_fsq_guard_keeps_last_accepted \
  -q

python - <<'PY'
import json, runpy
from pathlib import Path
m = json.loads(Path(
    "results/mirror/free_boundary_circular_coils_m143_schema03_guard1p1/"
    "free_boundary_circular_coils_metrics.json"
).read_text())
mod = runpy.run_path("examples/mirror_free_boundary_circular_coils.py")
mod["validate_circular_coil_beta_scan_metrics"](m)
print("validated current M143 metrics with hardened schema",
      m["metrics_schema_version"], len(m["summary_rows"]))
PY

JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_examples.py -q
```

Results:

- Ruff format left both touched files unchanged.
- Ruff lint passed.
- Focused free-boundary example tests passed: `3 passed`.
- Hardened validator accepted the M143 metrics file.
- Full mirror examples suite passed: `18 passed in 76.91s`.

### File structure and best-practice notes

- The schema contract remains in the root example, which is the single
  human-facing circular-coil beta-scan entry point.
- The tests remain in `tests/mirror/test_mirror_examples.py`, next to the
  existing root-example smoke and guard-path tests.
- No generated result files were added to git.
- This is a schema-contract hardening change; it does not alter the physical
  solve or plotting algorithms.

### Best next steps

1. Commit and push M144.
2. Add the coupled free-boundary update strategy so LCFS pressure-balance
   improvement and reduced-equilibrium residual control are handled together.
3. Run a compact plotted beta-scan after the coupled update is available and
   compare against the M143 guarded-pilot evidence.
4. Refresh the draft PR body after the next substantive solver tranche.

### Completion percentages after M144

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `91%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `92%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `92%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `89%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `92%`.
- PR merge readiness overall: `97%`.

### User input needed

No user input is needed.

---
## 145. Realized Coupled LCFS Pilot Candidate Selection

### Steps taken

- Added `--lcfs-proposal-mode coupled` to the root circular-coil
  free-boundary example.
- Added `--lcfs-coupled-fsq-weight` to control how strongly realized
  fixed-boundary residual growth penalizes an LCFS trial.
- Kept the existing `best_predicted`, `local`, `scale`, `bnormal`, and `mixed`
  modes unchanged.
- Extended the internal proposal selection record so the pilot loop can reuse
  the same candidate set and allowed-strategy filtering.
- Added coupled-mode trial evaluation:
  - no-op is retained as a zero-cost fallback;
  - allowed non-noop candidates are each evaluated by a short fixed-boundary
    trial solve;
  - each trial gets a realized score equal to merit ratio plus
    `lcfs_coupled_fsq_weight * max(fsq_growth_ratio - 1, 0)`;
  - the selected trial row records `coupled_trial_rows`, `coupled_score`,
    `coupled_merit_ratio`, `coupled_fsq_penalty`, and
    `coupled_fsq_weight`.
- Fixed the no-op shortcut so coupled mode still evaluates non-noop candidates
  when the predicted selector alone would pick no-op.
- Added a regression test showing that the compact 3% beta case selects the
  realized minimum-score trial.
- Updated the mirror examples README with the coupled-mode workflow and JSON
  fields.
- Ran a plotted coupled beta scan for evidence.

### Results obtained

- The compact coupled regression selects `bnormal_slope` for the 3% beta case
  with `lcfs_coupled_fsq_weight=1.0`, because it gives nearly the same LCFS
  merit improvement as the scale candidate with much lower realized fsq growth.
- Plotted evidence run:
  `results/mirror/free_boundary_circular_coils_m145_coupled/free_boundary_circular_coils_metrics.json`.
- Evidence run settings:
  - `ntheta=8`;
  - `nxi=11`;
  - `n_segments=64`;
  - fixed-boundary baseline `maxiter=20`;
  - coupled LCFS pilot up to three steps;
  - stagnation tolerance `1e-3`;
  - fsq-growth guard `1.1`;
  - coupled fsq weight `1.0`.
- Evidence run validation:
  - metrics schema version `0.3`;
  - workflow status `lcfs_pilot`;
  - free-boundary status `lcfs_pilot_not_converged_free_boundary`;
  - pilot rows total `6`;
  - accepted rows `3`;
  - skipped rows `1`;
  - stop reasons: one `noop_candidate`, two `fsq_growth_guard`, and three
    active rows with no stop reason.
- Per-beta evidence:
  - `1%`: coupled mode selected no-op instead of taking the residual-exploding
    first trial seen in M143.
  - `3%`: accepted two pilot updates before the fsq guard; last accepted fsq
    growth ratio `1.0698819230159147`; last accepted LCFS merit
    `0.4854722189173122`.
  - `10%`: accepted one pilot update before the fsq guard; last accepted fsq
    growth ratio `1.0377588770446344`; last accepted LCFS merit
    `0.7184197888333153`.
- Compared with M143, the coupled selector makes the early free-boundary pilot
  more conservative at `1%`, allows one additional accepted update at `3%`,
  and reduces first-step fsq growth at `3%` and `10%`.
- The rendered beta-scan summary plot and selected-step LCFS diagnostic were
  inspected. The summary plot clearly shows no accepted pilot at `1%` and
  improved `3%`/`10%` LCFS merit after accepted coupled steps.

### How it was tested

Commands run:

```bash
python -m ruff format examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_examples.py
python -m ruff check examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_examples.py

JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_coupled_mode_scores_realized_trials \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_tolerant_fsq_guard_keeps_last_accepted \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_example_runs_without_plots \
  -q

PYTHONPATH=.:$PYTHONPATH JAX_ENABLE_X64=1 \
  python examples/mirror_free_boundary_circular_coils.py \
    --outdir results/mirror/free_boundary_circular_coils_m145_coupled \
    --ntheta 8 --nxi 11 --n-segments 64 \
    --run-fixed-boundary-baseline --baseline-maxiter 20 \
    --run-lcfs-pilot --lcfs-pilot-steps 3 \
    --lcfs-proposal-mode coupled \
    --lcfs-coupled-fsq-weight 1.0 \
    --lcfs-pilot-stagnation-rtol 1e-3 \
    --lcfs-pilot-fsq-growth-limit 1.1

python - <<'PY'
import json, runpy
from pathlib import Path
m = json.loads(Path(
    "results/mirror/free_boundary_circular_coils_m145_coupled/"
    "free_boundary_circular_coils_metrics.json"
).read_text())
mod = runpy.run_path("examples/mirror_free_boundary_circular_coils.py")
mod["validate_circular_coil_beta_scan_metrics"](m)
print("validated", m.get("metrics_schema_version"),
      len(m.get("summary_rows", [])), "weight",
      m.get("lcfs_coupled_fsq_weight"))
PY

JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_examples.py \
  tests/mirror/test_mirror_free_boundary.py -q
python -m sphinx -W -b html docs docs/_build/html
git diff --check
```

Results:

- Ruff format/check passed.
- Focused coupled/guard tests passed: `3 passed`.
- The plotted coupled evidence run completed and validated.
- Full examples plus free-boundary tests passed: `50 passed in 78.27s`.
- Sphinx docs build passed with warnings treated as errors.
- Whitespace check passed.

### File structure and best-practice notes

- The new coupled pilot mode stays in the root example because it is still an
  ESSOS-style planning workflow, not a reusable package-level free-boundary
  solver API.
- Public mirror package imports did not change.
- The generated evidence remains under ignored `results/`.
- The JSON keeps the compact schema version at `0.3`; coupled trial rows are
  additive optional fields on pilot rows, so existing consumers remain
  compatible.
- The implementation is intentionally explicit rather than hidden behind a new
  abstraction because the next research step is to replace this pilot with a
  true coupled free-boundary solve.

### Best next steps

1. Commit and push M145.
2. Refresh the draft PR body with the coupled-mode free-boundary progress.
3. Decide whether the coupled pilot's realized trial-score table should get a
   dedicated plot panel in the example output.
4. Move from pilot candidate selection toward a true coupled free-boundary
   solve that includes boundary parameters in the nonlinear residual rather
   than selecting among externally proposed boundaries.

### Completion percentages after M145

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `91%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `92%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `92%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `91%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `94%`.
- PR merge readiness overall: `98%`.

### User input needed

No user input is needed.

---
## 146. Draft PR Description Synchronization After Coupled LCFS Pilot

### Steps taken

- Rewrote draft PR #21 body with the current M145 status.
- Updated the detailed-log pointer from section 141 to section 145.
- Added the realized coupled LCFS pilot candidate scoring status.
- Added the latest local validation gate:
  - `50 passed in 78.27s`;
  - Sphinx docs build with warnings as errors;
  - Ruff check/format-check on touched Python files;
  - whitespace check;
  - plotted coupled beta-scan evidence path.
- Kept the PR in draft state.
- Verified the remote PR body after editing.

### Results obtained

- Remote PR body verification confirmed:
  - PR number `21`;
  - draft state `true`;
  - body contains `section 145`;
  - body contains `Realized coupled LCFS`;
  - body contains `50 passed in 78.27s`;
  - body contains `free_boundary_circular_coils_m145_coupled`.

### How it was tested

Commands run:

```bash
gh pr edit 21 --body-file /tmp/vmec_mirror_pr_body_m145.md
gh pr view 21 --json number,isDraft,body,url | python -c '...'
```

Results:

- PR edit succeeded.
- PR body verification passed.
- PR remained draft.

### File structure and best-practice notes

- No source files changed in this tranche.
- Detailed evidence remains in `plan_mirror.md`; the PR body stays a concise
  review index.
- The PR remains draft as requested.

### Best next steps

1. Commit and push M146 plan log.
2. Continue toward a true coupled mirror free-boundary solve beyond LCFS pilot
   candidate selection.
3. Add a coupled trial-score plot panel only if it helps diagnose the next
   free-boundary solve tranche.
4. Check CI later and fix any failures without waiting on the full matrix now.

### Completion percentages after M146

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `91%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `92%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `92%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `91%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `94%`.
- PR merge readiness overall: `98%`.

### User input needed

No user input is needed.

---
## 147. Public LCFS Residual Vector for Coupled Free-Boundary Solves

### Steps taken

- Added `MirrorLCFSResidual` to `vmec_jax.mirror.free_boundary`.
- Added `mirror_lcfs_residual`, which returns:
  - normalized pressure-balance residual components;
  - normalized external-normal-field residual components;
  - one concatenated residual vector;
  - the same scalar value used by `mirror_lcfs_merit`;
  - the scales and RMS diagnostics used for normalization.
- Reimplemented `mirror_lcfs_merit` as a scalar wrapper around the residual
  helper while preserving compatibility with diagnostics that only expose RMS
  values.
- Exported `MirrorLCFSResidual` and `mirror_lcfs_residual` through
  `vmec_jax.mirror.api` and `vmec_jax.mirror`.
- Added a component-level test comparing residual vector entries against the
  expected pressure and normal-field components.
- Updated `docs/mirror/overview.rst` and `examples/mirror/README.md` so the
  residual-vector path is discoverable.

### Results obtained

- The free-boundary lane now has a package-level vector residual target for the
  next true coupled solve, rather than only scalar merit values and pilot
  candidate reports.
- `mirror_lcfs_merit` still returns the same scalar value and continues to work
  for the existing minimal RMS-only diagnostic tests.
- Public import check passed:
  `from vmec_jax.mirror import MirrorLCFSResidual, mirror_lcfs_residual`.

### How it was tested

Commands run:

```bash
python -m ruff format vmec_jax/mirror/free_boundary.py \
  vmec_jax/mirror/api.py vmec_jax/mirror/__init__.py \
  tests/mirror/test_mirror_free_boundary.py
python -m ruff check vmec_jax/mirror/free_boundary.py \
  vmec_jax/mirror/api.py vmec_jax/mirror/__init__.py \
  tests/mirror/test_mirror_free_boundary.py

JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_free_boundary.py::test_mirror_lcfs_merit_combines_pressure_and_normal_field \
  tests/mirror/test_mirror_free_boundary.py::test_mirror_lcfs_residual_vector_matches_merit_components \
  tests/mirror/test_mirror_free_boundary.py::test_mirror_lcfs_merit_rejects_invalid_scales \
  -q

JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_free_boundary.py -q
python -m sphinx -W -b html docs docs/_build/html
python - <<'PY'
from vmec_jax.mirror import MirrorLCFSResidual, mirror_lcfs_residual
print(MirrorLCFSResidual.__name__, callable(mirror_lcfs_residual))
PY
git diff --check
```

Results:

- Ruff format/check passed.
- Focused residual/merit tests passed: `5 passed`.
- Full free-boundary tests passed: `32 passed in 1.85s`.
- Public import check printed `MirrorLCFSResidual True`.
- Sphinx docs build passed with warnings treated as errors.
- Whitespace check passed.

### File structure and best-practice notes

- The residual helper lives beside the existing LCFS diagnostic/merit helpers in
  `vmec_jax/mirror/free_boundary.py`.
- Public exports are centralized through the existing mirror API files.
- The test is in `tests/mirror/test_mirror_free_boundary.py`, next to the LCFS
  merit and candidate-update tests.
- No generated output files were added.

### Best next steps

1. Commit and push M147.
2. Use `mirror_lcfs_residual` as the boundary-condition block in the first true
   coupled free-boundary solve prototype.
3. Add boundary-parameter finite-difference or implicit Jacobian checks for the
   LCFS residual block before coupling it to the reduced equilibrium residual.
4. Re-run the circular-coil beta scan once a coupled residual solve can update
   boundary coefficients directly.

### Completion percentages after M147

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `91%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `92%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `92%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `92%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `94%`.
- PR merge readiness overall: `98%`.

### User input needed

No user input is needed.

---
## 148. Draft PR Description Synchronization After LCFS Residual Vector

### Steps taken

- Updated draft PR #21 body after M147.
- Changed the detailed-log pointer from section 145 to section 147.
- Added the public `mirror_lcfs_residual` / `MirrorLCFSResidual` status.
- Added the latest free-boundary validation result:
  - `32 passed in 1.85s`;
  - Sphinx docs build;
  - Ruff check/format-check;
  - whitespace check.
- Kept the PR in draft state.
- Verified the remote PR body after editing.

### Results obtained

- Remote PR body verification confirmed:
  - PR number `21`;
  - draft state `true`;
  - body contains `section 147`;
  - body contains `mirror_lcfs_residual`;
  - body contains `32 passed in 1.85s`;
  - body contains `Public normalized LCFS residual vector`.

### How it was tested

Commands run:

```bash
gh pr edit 21 --body-file /tmp/vmec_mirror_pr_body_m147.md
gh pr view 21 --json number,isDraft,body,url | python -c '...'
```

Results:

- PR edit succeeded.
- PR body verification passed.
- PR remained draft.

### File structure and best-practice notes

- No source files changed in this tranche.
- Detailed evidence remains in `plan_mirror.md`; the PR body stays a concise
  review index.
- The PR remains draft as requested.

### Best next steps

1. Commit and push M148 plan log.
2. Start LCFS residual boundary-parameter Jacobian checks.
3. Check CI later and fix reported failures without waiting on the full matrix.

### Completion percentages after M148

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `91%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `92%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `92%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `92%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `94%`.
- PR merge readiness overall: `98%`.

### User input needed

No user input is needed.

---
## 149. Array LCFS Diagnostic and Boundary-Coefficient Residual Jacobian Check

### Steps taken

- Factored `mirror_lcfs_diagnostic` through a new
  `mirror_lcfs_diagnostic_from_arrays` helper.
- Kept the existing output-object diagnostic API as a wrapper, so examples and
  existing callers continue to work.
- Exported `mirror_lcfs_diagnostic_from_arrays` through `vmec_jax.mirror.api`
  and `vmec_jax.mirror`.
- Added a parity assertion showing the array helper matches the output-object
  diagnostic for the same boundary, field, and pressure inputs.
- Added a finite-difference LCFS residual Jacobian check with respect to
  polynomial boundary coefficients `(r0, a2, a4)`.

### Results obtained

- The free-boundary residual block can now be evaluated directly from arrays,
  which makes boundary-parameter Jacobian checks and future coupled residual
  assembly simpler.
- The finite-difference test verifies that the residual vector has a finite,
  nonzero boundary-coefficient Jacobian and that the column-wise Jacobian
  agrees with an independent directional finite difference to finite-difference
  precision.
- Public import check for `mirror_lcfs_diagnostic_from_arrays` passed.

### How it was tested

Commands run:

```bash
python -m ruff format vmec_jax/mirror/free_boundary.py \
  vmec_jax/mirror/api.py vmec_jax/mirror/__init__.py \
  tests/mirror/test_mirror_free_boundary.py
python -m ruff check vmec_jax/mirror/free_boundary.py \
  vmec_jax/mirror/api.py vmec_jax/mirror/__init__.py \
  tests/mirror/test_mirror_free_boundary.py

JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_free_boundary.py::test_mirror_lcfs_diagnostic_reports_side_boundary_targets \
  tests/mirror/test_mirror_free_boundary.py::test_mirror_lcfs_residual_has_boundary_coefficient_finite_difference_jacobian \
  -q

python - <<'PY'
from vmec_jax.mirror import mirror_lcfs_diagnostic_from_arrays
print(callable(mirror_lcfs_diagnostic_from_arrays))
PY

JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_free_boundary.py -q
python -m sphinx -W -b html docs docs/_build/html
git diff --check
```

Results:

- Ruff format/check passed.
- Focused diagnostic/Jacobian tests passed: `2 passed`.
- Public import check printed `True`.
- Full free-boundary tests passed: `33 passed in 1.96s`.
- Sphinx docs build passed with warnings treated as errors.
- Whitespace check passed.

### File structure and best-practice notes

- The array diagnostic helper lives beside the output-object LCFS diagnostic in
  `vmec_jax/mirror/free_boundary.py`.
- Public exports remain centralized through the existing mirror API files.
- The finite-difference Jacobian test lives in the free-boundary test module,
  next to the residual-vector and LCFS candidate tests.
- No generated result files were added.

### Best next steps

1. Commit and push M149.
2. Assemble the first prototype residual vector that combines reduced
   equilibrium residual components with `mirror_lcfs_residual` boundary
   components.
3. Add a small least-squares solve over boundary coefficients after the
   residual assembly has finite-difference coverage.
4. Check CI later and fix failures if reported.

### Completion percentages after M149

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `91%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `92%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `92%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `93%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `94%`.
- PR merge readiness overall: `98%`.

### User input needed

No user input is needed.

---
## 150. Combined Equilibrium and LCFS Residual Assembly Helper

### Steps taken

- Added `MirrorFreeBoundaryResidual` to `vmec_jax.mirror.free_boundary`.
- Added `mirror_free_boundary_residual`, which combines:
  - a reduced-equilibrium residual vector;
  - a normalized `MirrorLCFSResidual` boundary-condition vector;
  - independent equilibrium and LCFS block weights;
  - an explicit equilibrium residual scale.
- Exported `MirrorFreeBoundaryResidual` and
  `mirror_free_boundary_residual` through `vmec_jax.mirror.api` and
  `vmec_jax.mirror`.
- Added tests for:
  - equilibrium and LCFS block scaling;
  - concatenated vector layout;
  - scalar combined residual value;
  - invalid empty residuals, scales, and negative weights.
- Updated the mirror overview to mention the combined residual assembly helper.

### Results obtained

- The free-boundary lane now has the first package-level residual assembly
  helper for true coupled solve prototypes.
- The helper does not solve the coupled nonlinear problem yet; it provides the
  vector layout and normalization contract needed by a future least-squares or
  Newton solve.
- Public import check passed:
  `from vmec_jax.mirror import MirrorFreeBoundaryResidual, mirror_free_boundary_residual`.

### How it was tested

Commands run:

```bash
python -m ruff format vmec_jax/mirror/free_boundary.py \
  vmec_jax/mirror/api.py vmec_jax/mirror/__init__.py \
  tests/mirror/test_mirror_free_boundary.py
python -m ruff check vmec_jax/mirror/free_boundary.py \
  vmec_jax/mirror/api.py vmec_jax/mirror/__init__.py \
  tests/mirror/test_mirror_free_boundary.py

JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_free_boundary.py::test_mirror_free_boundary_residual_combines_equilibrium_and_lcfs_blocks \
  tests/mirror/test_mirror_free_boundary.py::test_mirror_free_boundary_residual_rejects_invalid_inputs \
  -q

python - <<'PY'
from vmec_jax.mirror import MirrorFreeBoundaryResidual, mirror_free_boundary_residual
print(MirrorFreeBoundaryResidual.__name__, callable(mirror_free_boundary_residual))
PY

JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_free_boundary.py -q
python -m sphinx -W -b html docs docs/_build/html
git diff --check
```

Results:

- Ruff format/check passed.
- Focused combined-residual tests passed: `5 passed`.
- Public import check printed `MirrorFreeBoundaryResidual True`.
- Full free-boundary tests passed: `38 passed in 2.04s`.
- Sphinx docs build passed with warnings treated as errors.
- Whitespace check passed.

### File structure and best-practice notes

- The combined residual helper lives in `vmec_jax/mirror/free_boundary.py`
  beside the LCFS residual and diagnostic helpers.
- Public exports stay centralized in the mirror API files.
- Tests remain in `tests/mirror/test_mirror_free_boundary.py` with the other
  free-boundary residual and candidate checks.
- No generated result files were added.

### Best next steps

1. Commit and push M150.
2. Build a small least-squares prototype that uses
   `mirror_free_boundary_residual` with polynomial boundary coefficients.
3. Add a plotted diagnostic for combined residual components once the
   least-squares prototype exists.
4. Refresh the PR body after the next solver-prototype tranche.

### Completion percentages after M150

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `92%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `92%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `92%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `94%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `94%`.
- PR merge readiness overall: `98%`.

### User input needed

No user input is needed.

---
## 151. Draft PR Description Synchronization After Combined Residual Assembly

### Steps taken

- Updated draft PR #21 body after M150.
- Changed the detailed-log pointer from section 147 to section 150.
- Added `mirror_free_boundary_residual` and
  `MirrorFreeBoundaryResidual` to the PR status.
- Added the latest free-boundary validation result:
  - `38 passed in 2.04s`;
  - Sphinx docs build;
  - Ruff check/format-check;
  - whitespace check.
- Kept the PR in draft state.
- Verified the remote PR body after editing.

### Results obtained

- Remote PR body verification confirmed:
  - PR number `21`;
  - draft state `true`;
  - body contains `section 150`;
  - body contains `mirror_free_boundary_residual`;
  - body contains `38 passed in 2.04s`;
  - body contains `Combined equilibrium-plus-LCFS residual assembly`.

### How it was tested

Commands run:

```bash
gh pr edit 21 --body-file /tmp/vmec_mirror_pr_body_m150.md
gh pr view 21 --json number,isDraft,body,url | python -c '...'
```

Results:

- PR edit succeeded.
- PR body verification passed.
- PR remained draft.

### File structure and best-practice notes

- No source files changed in this tranche.
- Detailed evidence remains in `plan_mirror.md`; the PR body stays a concise
  review index.
- The PR remains draft as requested.

### Best next steps

1. Commit and push M151 plan log.
2. Build the least-squares prototype that uses the combined residual helper.
3. Check CI later and fix reported failures without waiting on the full matrix.

### Completion percentages after M151

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `92%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `92%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `92%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `94%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `94%`.
- PR merge readiness overall: `98%`.

### User input needed

No user input is needed.

---
## 152. Line-Searched Free-Boundary Least-Squares Step Prototype

### Steps taken

- Added `MirrorFreeBoundaryLeastSquaresStep` to
  `vmec_jax.mirror.free_boundary`.
- Added `mirror_free_boundary_residual_jacobian_finite_difference`, a central
  finite-difference Jacobian helper for residual builders that return the
  existing `MirrorFreeBoundaryResidual`.
- Added `mirror_free_boundary_least_squares_step`, which:
  - builds the combined residual Jacobian;
  - solves a regularized linear least-squares boundary-coefficient update;
  - applies damping and a per-coefficient step cap;
  - evaluates a short backtracking list;
  - returns the best non-increasing trial, or marks the step unaccepted and
    keeps the original coefficients if all trials increase the residual.
- Exported the new dataclass and helpers through `vmec_jax.mirror.api` and
  `vmec_jax.mirror`.
- Added synthetic tests that still construct residuals through
  `mirror_lcfs_residual` and `mirror_free_boundary_residual`, so the prototype
  exercises the same vector contract as the coupled mirror solve lane.
- Updated the mirror overview and example README to describe the finite-
  difference LS step as a CLI/prototype derivative path to be replaced by
  implicit/JAX/adjoint derivatives once the full residual path is
  differentiable.
- Checked draft PR #21 once after the previous push; the PR was still draft,
  several jobs were green, and no CI failures were present in the rollup.

### Results obtained

- The free-boundary lane now has a package-level boundary-coefficient LS step
  that directly consumes the combined equilibrium-plus-LCFS residual vector.
- Linear synthetic residual tests recover the exact target coefficients and
  finite-difference Jacobian.
- A nonlinear synthetic residual test confirms the backtracking path rejects a
  worse full linear step and accepts a smaller residual-reducing step.
- Invalid input tests cover empty/nonfinite coefficients, invalid finite-
  difference step, damping, step cap, ridge, line-search factors, and callbacks
  that do not return `MirrorFreeBoundaryResidual`.
- The helper is still a prototype derivative path for CLI workflows; it is not
  yet the full coupled mirror free-boundary solve.

### How it was tested

Commands run:

```bash
python -m ruff format vmec_jax/mirror/free_boundary.py \
  vmec_jax/mirror/api.py vmec_jax/mirror/__init__.py \
  tests/mirror/test_mirror_free_boundary.py
python -m ruff check vmec_jax/mirror/free_boundary.py \
  vmec_jax/mirror/api.py vmec_jax/mirror/__init__.py \
  tests/mirror/test_mirror_free_boundary.py

JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_free_boundary.py::test_mirror_free_boundary_residual_jacobian_finite_difference_matches_linear_model \
  tests/mirror/test_mirror_free_boundary.py::test_mirror_free_boundary_least_squares_step_reduces_linear_combined_residual \
  tests/mirror/test_mirror_free_boundary.py::test_mirror_free_boundary_least_squares_step_backtracks_nonlinear_residual \
  tests/mirror/test_mirror_free_boundary.py::test_mirror_free_boundary_least_squares_step_rejects_invalid_inputs \
  tests/mirror/test_mirror_free_boundary.py::test_mirror_free_boundary_residual_jacobian_rejects_non_residual_return \
  -q

JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_free_boundary.py -q
python -m sphinx -W -b html docs docs/_build/html
git diff --check
```

Results:

- Ruff format/check passed.
- Focused LS-step tests passed: `12 passed in 0.25s`.
- Full free-boundary tests passed: `50 passed in 2.14s`.
- Sphinx docs build passed with warnings treated as errors.
- Whitespace check passed.
- Public import check printed:
  `MirrorFreeBoundaryLeastSquaresStep True True`.

### File structure and best-practice notes

- The LS-step dataclass and helpers live in `vmec_jax/mirror/free_boundary.py`
  beside the residual assembly helpers they consume.
- Public exports remain centralized through `vmec_jax/mirror/api.py` and
  `vmec_jax/mirror/__init__.py`.
- Tests stay in `tests/mirror/test_mirror_free_boundary.py`, close to the LCFS
  residual and combined residual checks.
- Documentation updates are limited to `docs/mirror/overview.rst` and
  `examples/mirror/README.md`.
- No generated result files or figures were added in this source-level
  tranche.

### Best next steps

1. Commit and push M152.
2. Wire `mirror_free_boundary_least_squares_step` into the circular-coil mirror
   beta-scan example as an optional diagnostic mode over a compact polynomial
   boundary basis.
3. Add a lightweight plot for LS residual components versus trial/backtracking
   step once the example path exists.
4. Check CI later and fix failures if any appear.

### Completion percentages after M152

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `92%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `92%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `95%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `94%`.
- PR merge readiness overall: `98%`.

### User input needed

No user input is needed.

---
## 153. Circular-Coil Example LS Boundary-Step Diagnostic and Plot

### Steps taken

- Wired `mirror_free_boundary_least_squares_step` into the root-level
  `examples/mirror_free_boundary_circular_coils.py` example behind
  `--run-ls-boundary-step`.
- Added CLI controls for the diagnostic derivative/step settings:
  - `--ls-boundary-finite-difference-step`;
  - `--ls-boundary-damping`;
  - `--ls-boundary-max-relative-step`;
  - `--ls-boundary-ridge`.
- Added a compact polynomial boundary basis for the diagnostic:
  `[r0, a2, a4]`.
- Added a frozen-interior residual builder for the example mode:
  - fixed-boundary normalized-force scalar as the equilibrium block;
  - external coil field resampled on trial polynomial boundaries;
  - LCFS pressure-balance and external-normal-field residuals normalized
    through `mirror_lcfs_residual`;
  - combined vector assembled with `mirror_free_boundary_residual`.
- Added per-beta `ls_boundary_step` JSON output with:
  - initial/new coefficients;
  - raw and limited LS steps;
  - finite-difference steps;
  - Jacobian shape;
  - residual values before/after;
  - equilibrium and LCFS component values;
  - tried backtracking factors;
  - selected factor and acceptance flag;
  - optional plot path.
- Bumped the circular-coil metrics schema to version `0.4`.
- Added schema validation for LS step counts and LS step fields.
- Added `_write_ls_boundary_step_plot`, which plots combined residual and
  equilibrium/LCFS component values versus line-search factor.
- Updated `examples/mirror/README.md` for schema `0.4` and the LS diagnostic.
- Added an example regression test for `--run-ls-boundary-step`.
- Regenerated and inspected a one-beta plotted example to verify the LS
  residual/backtracking plot renders correctly.

### Results obtained

- The one-beta LS smoke run completed:
  `results/mirror/free_boundary_circular_coils_m153_lsq_smoke/free_boundary_circular_coils_metrics.json`.
- The plotted one-beta LS run completed:
  `results/mirror/free_boundary_circular_coils_m153_lsq_plots_sorted/free_boundary_circular_coils_metrics.json`.
- The rendered LS plot was visually inspected:
  `results/mirror/free_boundary_circular_coils_m153_lsq_plots_sorted/figures/fixed_boundary_beta_1/free_boundary_circular_coils_beta_1_ls_boundary_step.png`.
- The smoke metrics reported:
  - schema version `0.4`;
  - `ls_boundary_step_requested == true`;
  - `ls_boundary_step_rows_total == 1`;
  - Jacobian shape `[23, 3]`;
  - accepted step with line-search factor `1.0`;
  - combined residual value decreased from `1.2264846366589521` to
    `1.1947548991708283`;
  - frozen equilibrium RMS stayed at `0.16117780636984214`;
  - LCFS value decreased from `1.004255509280822` to
    `0.9245964190853178`.
- The plot shows the residual components versus line-search factor with the
  selected full step marked by a vertical line.

### How it was tested

Commands run:

```bash
python examples/mirror_free_boundary_circular_coils.py \
  --outdir results/mirror/free_boundary_circular_coils_m153_lsq_smoke \
  --betas 1 --ntheta 8 --nxi 11 --n-segments 64 \
  --run-fixed-boundary-baseline --baseline-maxiter 0 \
  --run-ls-boundary-step --no-plots

python examples/mirror_free_boundary_circular_coils.py \
  --outdir results/mirror/free_boundary_circular_coils_m153_lsq_plots_sorted \
  --betas 1 --ntheta 8 --nxi 11 --n-segments 64 \
  --run-fixed-boundary-baseline --baseline-maxiter 0 \
  --run-ls-boundary-step

python -m ruff format examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_examples.py
python -m ruff check examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_examples.py

JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_example_runs_without_plots \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_ls_boundary_step_reports_reduction \
  -q

JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_examples.py -q
python -m sphinx -W -b html docs docs/_build/html
git diff --check
```

Results:

- Example LS smoke run passed.
- Example LS plotted run passed and the PNG was visually inspected.
- Ruff format/check passed on touched Python files.
- Focused example tests passed: `2 passed in 7.24s`.
- Full mirror example tests passed: `20 passed in 90.63s`.
- Sphinx docs build passed with warnings treated as errors.
- Whitespace check passed.
- A mistaken Ruff format command that included `examples/mirror/README.md`
  failed because Ruff tried to parse Markdown as Python; it was rerun correctly
  on Python files only.

### File structure and best-practice notes

- The package-level LS step remains in `vmec_jax/mirror/free_boundary.py`.
- The circular-coil example keeps example-specific polynomial fitting,
  frozen-interior residual construction, schema fields, and plotting private
  to `examples/mirror_free_boundary_circular_coils.py`.
- Tests remain in `tests/mirror/test_mirror_examples.py`, close to the other
  example schema and workflow checks.
- Documentation for the example lives in `examples/mirror/README.md`.
- Generated smoke/plot outputs remain under ignored `results/` paths and are
  not added to git.

### Best next steps

1. Commit and push M153.
2. Refresh the draft PR body to mention schema `0.4` and the LS boundary-step
   diagnostic.
3. Use the LS diagnostic as the bridge to a true coupled solve mode that reruns
   the fixed-boundary state after each boundary coefficient update instead of
   freezing the interior.
4. Check CI later and fix failures if any appear.

### Completion percentages after M153

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `93%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `92%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `96%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `95%`.
- PR merge readiness overall: `98%`.

### User input needed

No user input is needed.

---
## 154. Draft PR Description Synchronization After LS Example Diagnostic

### Steps taken

- Updated draft PR #21 body after M153.
- Changed the detailed-log pointer from section 150 to section 153.
- Added schema `mirror_free_boundary_circular_coil_beta_scan` version `0.4`.
- Added `mirror_free_boundary_least_squares_step` and the
  `--run-ls-boundary-step` circular-coil example diagnostic to the PR status.
- Added the latest validation evidence:
  - `50 passed in 2.14s` for free-boundary tests;
  - `20 passed in 90.63s` for mirror example tests;
  - Sphinx docs build;
  - Ruff check/format-check;
  - whitespace check;
  - plotted LS evidence under
    `results/mirror/free_boundary_circular_coils_m153_lsq_plots_sorted/`.
- Kept the PR in draft state.
- Verified the remote PR body after editing.

### Results obtained

- Remote PR body verification confirmed:
  - PR number `21`;
  - draft state `true`;
  - body contains `section 153`;
  - body contains schema version `0.4`;
  - body contains `mirror_free_boundary_least_squares_step`;
  - body contains `20 passed in 90.63s`;
  - body contains `free_boundary_circular_coils_m153_lsq_plots_sorted`.

### How it was tested

Commands run:

```bash
gh pr edit 21 --body-file /tmp/vmec_mirror_pr_body_m153.md
gh pr view 21 --json number,isDraft,body,url | python -c '...'
```

Results:

- PR edit succeeded.
- PR body verification passed.
- PR remained draft.
- The latest CI run for commit `52852139` was queued/in progress at the
  status check and had no reported failures in the rollup.

### File structure and best-practice notes

- No source files changed in this tranche.
- Detailed evidence remains in `plan_mirror.md`; the PR body stays a concise
  review index.
- The PR remains draft as requested.

### Best next steps

1. Commit and push M154 plan log.
2. Build the next free-boundary tranche: a true coupled diagnostic mode that
   evaluates LS-selected boundary coefficient updates by rerunning the
   fixed-boundary solve, not only by freezing the interior state.
3. Check CI later and fix failures if any appear.

### Completion percentages after M154

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `93%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `92%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `96%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `95%`.
- PR merge readiness overall: `98%`.

### User input needed

No user input is needed.

---
## 155. Realized Coupled Trial for LS-Selected Boundary Updates

### Steps taken

- Added `--run-ls-boundary-coupled-trial` to
  `examples/mirror_free_boundary_circular_coils.py`.
- Bumped the circular-coil beta-scan schema to version `0.5`.
- Added top-level metrics:
  - `ls_boundary_coupled_trial_requested`;
  - `ls_boundary_coupled_trial_rows_total`.
- Added nested `ls_boundary_step.coupled_trial` metrics with:
  - trial status;
  - trial MOUT path;
  - realized residual norm, `fsq`, and normalized force;
  - `fsq` growth ratio relative to the baseline fixed-boundary row;
  - realized LCFS pressure-balance RMS, external-normal-field RMS, LCFS merit,
    and LCFS merit ratio;
  - merit acceptance and rejection reason;
  - optional trial plot paths.
- Added schema validation for nested coupled trial rows and trial row counts.
- Added a focused regression test that runs one beta with
  `--run-ls-boundary-step --run-ls-boundary-coupled-trial`.
- Updated the mirror example README for schema `0.5` and the realized coupled
  LS trial option.
- Generated and inspected a plotted one-beta realized LS trial.

### Results obtained

- The one-beta realized LS smoke run completed:
  `results/mirror/free_boundary_circular_coils_m155_lsq_coupled_smoke/free_boundary_circular_coils_metrics.json`.
- The plotted one-beta realized LS run completed:
  `results/mirror/free_boundary_circular_coils_m155_lsq_coupled_plots/free_boundary_circular_coils_metrics.json`.
- The smoke metrics reported:
  - schema version `0.5`;
  - `ls_boundary_step_requested == true`;
  - `ls_boundary_coupled_trial_requested == true`;
  - `ls_boundary_step_rows_total == 1`;
  - `ls_boundary_coupled_trial_rows_total == 1`;
  - LS selected line-search factor `1.0`;
  - coupled trial status `accepted`;
  - `final_fsq = 0.0008633410961533115`;
  - `fsq_growth_ratio = 0.8804859565965772`;
  - `lcfs_merit = 0.48021241738262876`;
  - `lcfs_merit_ratio = 0.4801851017221157`.
- The plotted run produced:
  - the LS residual/backtracking figure;
  - the realized trial standard mirror plots;
  - the realized trial LCFS diagnostic plot.
- Visual inspection confirmed the LS plot is sorted by line-search factor and
  that the realized trial LCFS diagnostic renders useful normal-field and
  pressure-balance profiles.

### How it was tested

Commands run:

```bash
python examples/mirror_free_boundary_circular_coils.py \
  --outdir results/mirror/free_boundary_circular_coils_m155_lsq_coupled_smoke \
  --betas 1 --ntheta 8 --nxi 11 --n-segments 64 \
  --run-fixed-boundary-baseline --baseline-maxiter 0 \
  --run-ls-boundary-step --run-ls-boundary-coupled-trial --no-plots

python examples/mirror_free_boundary_circular_coils.py \
  --outdir results/mirror/free_boundary_circular_coils_m155_lsq_coupled_plots \
  --betas 1 --ntheta 8 --nxi 11 --n-segments 64 \
  --run-fixed-boundary-baseline --baseline-maxiter 0 \
  --run-ls-boundary-step --run-ls-boundary-coupled-trial

python -m ruff format examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_examples.py
python -m ruff check examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_examples.py

JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_example_runs_without_plots \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_ls_boundary_step_reports_reduction \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_ls_boundary_coupled_trial_reports_realized_solve \
  -q

JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_examples.py -q
python -m sphinx -W -b html docs docs/_build/html
git diff --check
```

Results:

- Realized coupled LS smoke run passed.
- Realized coupled LS plotted run passed and the PNG outputs were visually
  inspected.
- Ruff format/check passed on touched Python files.
- Focused example tests passed: `3 passed in 10.07s`.
- Full mirror example tests passed: `21 passed in 94.85s`.
- Sphinx docs build passed with warnings treated as errors.
- Whitespace check passed.

### File structure and best-practice notes

- The realized coupled trial remains private to the circular-coil example,
  where the fixed-boundary trial solve, MOUT output, and plots are already
  managed.
- Package-level solver helpers were not expanded in this tranche.
- The example metrics schema is versioned to `0.5` because the JSON contract
  gained required top-level fields and nested `coupled_trial` fields.
- Generated smoke/plot outputs remain under ignored `results/` paths and are
  not added to git.

### Best next steps

1. Commit and push M155.
2. Refresh the draft PR body after the schema `0.5` coupled-trial tranche.
3. Extend from one realized LS trial to a guarded multi-step coupled LS loop
   with `fsq` growth and merit guards.
4. Check CI later and fix failures if any appear.

### Completion percentages after M155

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `93%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `92%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `97%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `96%`.
- PR merge readiness overall: `98%`.

### User input needed

No user input is needed.

---
## 156. Draft PR Description Synchronization After Realized LS Coupled Trial

### Steps taken

- Updated draft PR #21 body after M155.
- Changed the detailed-log pointer from section 153 to section 155.
- Added schema `mirror_free_boundary_circular_coil_beta_scan` version `0.5`.
- Added the `--run-ls-boundary-coupled-trial` realized fixed-boundary trial
  status to the PR body.
- Added the latest validation evidence:
  - `50 passed in 2.14s` for free-boundary tests;
  - `21 passed in 94.85s` for mirror example tests;
  - Sphinx docs build;
  - Ruff check/format-check;
  - whitespace check;
  - plotted realized LS trial evidence under
    `results/mirror/free_boundary_circular_coils_m155_lsq_coupled_plots/`.
- Kept the PR in draft state.
- Verified the remote PR body after editing.

### Results obtained

- Remote PR body verification confirmed:
  - PR number `21`;
  - draft state `true`;
  - body contains `section 155`;
  - body contains schema version `0.5`;
  - body contains `--run-ls-boundary-coupled-trial`;
  - body contains `21 passed in 94.85s`;
  - body contains `free_boundary_circular_coils_m155_lsq_coupled_plots`.

### How it was tested

Commands run:

```bash
gh pr edit 21 --body-file /tmp/vmec_mirror_pr_body_m155.md
gh pr view 21 --json number,isDraft,body,url | python -c '...'
```

Results:

- PR edit succeeded.
- PR body verification passed.
- PR remained draft.
- The status check rollup for the newest push was not populated yet at the
  quick check immediately after pushing.

### File structure and best-practice notes

- No source files changed in this tranche.
- Detailed evidence remains in `plan_mirror.md`; the PR body stays a concise
  review index.
- The PR remains draft as requested.

### Best next steps

1. Commit and push M156 plan log.
2. Extend from one realized LS trial to a guarded multi-step coupled LS loop
   with `fsq` growth and merit guards.
3. Check CI later and fix failures if any appear.

### Completion percentages after M156

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `93%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `92%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `97%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `96%`.
- PR merge readiness overall: `98%`.

### User input needed

No user input is needed.

---
## 157. Guarded Multi-Step Coupled LS Loop and CI Coverage-Gate Repair

### Steps taken

- Added `--run-ls-boundary-coupled-loop` to
  `examples/mirror_free_boundary_circular_coils.py`.
- Bumped the circular-coil beta-scan schema to version `0.6`.
- Added guarded loop controls:
  - `--ls-boundary-coupled-loop-steps`;
  - `--ls-boundary-coupled-loop-target-merit`;
  - `--ls-boundary-coupled-loop-stagnation-rtol`;
  - `--ls-boundary-coupled-loop-fsq-growth-limit`.
- Added loop-level metrics:
  - requested flag and requested guard values;
  - total loop rows;
  - accepted loop rows;
  - stop-reason counts.
- Added per-beta loop summary metrics:
  - loop status;
  - row count and accepted-row count;
  - final stop reason;
  - final merit and `fsq` growth ratio;
  - last accepted step, merit, and `fsq` growth ratio.
- Added nested loop rows that record each LS boundary step, realized
  fixed-boundary trial, acceptance decision, rejection reason, stop reason,
  realized MOUT path, and optional plot paths.
- Added schema validation for nested loop rows and their nested
  `ls_boundary_step` / `coupled_trial` records.
- Added a focused root-example regression test for a two-step guarded loop.
- Updated `examples/mirror/README.md` and `docs/mirror/overview.rst` for schema
  `0.6` and the guarded realized loop.
- Investigated the current PR #21 CI failure:
  - failing check: `Coverage Gate (py3.11 combined)`;
  - root cause: exact line coverage was `94.94%`, below the `95.00%` gate.
- Downloaded the CI coverage shards, remapped runner paths to the local checkout,
  and used them to identify real mirror free-boundary coverage gaps.
- Added focused `tests/mirror/test_mirror_free_boundary.py` coverage for public
  mirror free-boundary guard paths, finite-difference residual validation,
  line-search acceptance/rejection, direct-coil pressure-response sampling, and
  LCFS proposal validation branches.

### Results obtained

- The guarded loop now performs repeated cycles of:
  1. finite-difference LS boundary step on `[r0, a2, a4]`;
  2. realized fixed-boundary solve on the selected polynomial boundary;
  3. LCFS merit / `fsq` guard evaluation;
  4. update of the current boundary and output when accepted.
- The one-beta two-step smoke run reported:
  - schema version `0.6`;
  - workflow `ls_boundary_coupled_loop`;
  - free-boundary status `ls_boundary_coupled_loop_not_converged_free_boundary`;
  - `ls_boundary_coupled_loop_rows_total == 2`;
  - `ls_boundary_coupled_loop_accepted_rows_total == 1`;
  - stop counts `{"None": 1, "ls_step_not_accepted": 1}`;
  - first loop row accepted with `fsq_growth_ratio = 0.8804859565965772`;
  - first loop row accepted with `lcfs_merit_ratio = 0.4801851017221157`;
  - second loop row stopped because the nested LS step selected no improving
    boundary step.
- The plotted loop run produced and was visually inspected:
  - step-1 LS residual/backtracking plot:
    `results/mirror/free_boundary_circular_coils_m157_lsq_loop_plots/figures/fixed_boundary_beta_1_ls_loop_step_1/free_boundary_circular_coils_beta_1_ls_loop_step_1_ls_boundary_step.png`;
  - step-2 LS residual/backtracking plot:
    `results/mirror/free_boundary_circular_coils_m157_lsq_loop_plots/figures/fixed_boundary_beta_1_ls_loop_step_2/free_boundary_circular_coils_beta_1_ls_loop_step_2_ls_boundary_step.png`;
  - step-1 realized trial LCFS diagnostic plot:
    `results/mirror/free_boundary_circular_coils_m157_lsq_loop_plots/figures/fixed_boundary_beta_1_ls_loop_step_1_ls_boundary_trial/free_boundary_circular_coils_beta_1_ls_loop_step_1_ls_boundary_trial_lcfs_diagnostic.png`.
- The updated local coverage estimate combines the failed CI shards with local
  coverage from the updated mirror free-boundary and example tests and reports:
  `Exact line coverage estimate: 95.09%`.
- The same estimate raises `vmec_jax/mirror/free_boundary.py` coverage to `98%`.

### How it was tested

Commands run:

```bash
python examples/mirror_free_boundary_circular_coils.py \
  --outdir results/mirror/free_boundary_circular_coils_m157_lsq_loop_smoke \
  --betas 1 --ntheta 8 --nxi 11 --n-segments 64 \
  --run-fixed-boundary-baseline --baseline-maxiter 0 \
  --run-ls-boundary-coupled-loop --ls-boundary-coupled-loop-steps 2 \
  --no-plots

python examples/mirror_free_boundary_circular_coils.py \
  --outdir results/mirror/free_boundary_circular_coils_m157_lsq_loop_plots \
  --betas 1 --ntheta 8 --nxi 11 --n-segments 64 \
  --run-fixed-boundary-baseline --baseline-maxiter 0 \
  --run-ls-boundary-coupled-loop --ls-boundary-coupled-loop-steps 2

python -m ruff format examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_examples.py tests/mirror/test_mirror_free_boundary.py
python -m ruff check examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_examples.py tests/mirror/test_mirror_free_boundary.py

JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_free_boundary.py -q
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_examples.py -q

JAX_ENABLE_X64=1 COVERAGE_FILE=/tmp/vmec_jax_m157.coverage \
  pytest tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py -q --cov=vmec_jax --cov-report=

JAX_ENABLE_X64=1 COVERAGE_FILE=/tmp/vmec_jax_m157_free_boundary.coverage \
  pytest tests/mirror/test_mirror_free_boundary.py -q \
  --cov=vmec_jax --cov-report=

python -m sphinx -W -b html docs docs/_build/html
git diff --check
```

Results:

- Guarded LS loop smoke run passed.
- Guarded LS loop plotted run passed and PNG outputs were visually inspected.
- Ruff format/check passed on touched Python files.
- Mirror free-boundary tests passed: `86 passed in 2.97s`.
- Mirror example tests passed during the combined coverage run; combined
  free-boundary plus example coverage run passed: `93 passed in 101.07s`.
- Updated free-boundary coverage run passed: `86 passed in 3.18s`.
- Estimated combined coverage gate rose from failed CI `94.94%` to local
  estimate `95.09%`.
- Sphinx docs build passed with warnings treated as errors.
- Whitespace check passed.

### File structure and best-practice notes

- The guarded loop remains in the root circular-coil example because it is still
  an example/diagnostic driver around host-side finite differences and realized
  fixed-boundary solves, not the final differentiable production solver API.
- Core mirror free-boundary package code was not changed for the coverage-gate
  repair; the additional tests cover existing public guard behavior and direct
  circular-coil sampling paths.
- Schema `0.6` is explicit because the JSON output contract gained new
  loop-level fields and nested loop-row records.
- Generated smoke, plot, and coverage artifacts remain under ignored `results/`
  or `/tmp` paths and are not added to git.
- The file structure remains aligned with current practice:
  - reusable physics kernels stay under `vmec_jax/mirror/`;
  - root examples own CLI workflow/report orchestration;
  - mirror example behavior is tested from `tests/mirror/test_mirror_examples.py`;
  - package API guard behavior is tested from
    `tests/mirror/test_mirror_free_boundary.py`;
  - user-facing documentation stays in `examples/mirror/README.md` and
    `docs/mirror/overview.rst`.

### Best next steps

1. Commit and push M157.
2. Refresh the draft PR body for schema `0.6`, the guarded loop, and the
   coverage-gate repair.
3. Let CI run without waiting on every check; inspect failures later if any
   remain.
4. Continue the finite-step plan by promoting the guarded loop toward a true
   coupled nonlinear free-boundary solve with reusable solver structure, while
   keeping CLI paths fast and differentiable APIs separate.

### Completion percentages after M157

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `93%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `92%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `98%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- PR merge readiness overall: `98%`.

### User input needed

No user input is needed.

---
## 158. Draft PR Description Synchronization After Guarded LS Loop

### Steps taken

- Updated draft PR #21 body after M157.
- Changed the detailed implementation-log pointer from section 155 to
  section 157.
- Updated the free-boundary circular-coil schema reference from version `0.5`
  to version `0.6`.
- Marked the guarded multi-step coupled LS loop checklist item complete.
- Added the latest local validation and coverage-gate evidence:
  - `86 passed in 2.97s` for mirror free-boundary tests;
  - `93 passed in 101.07s` for the combined free-boundary plus mirror example
    coverage run;
  - estimated combined coverage gate `95.09%` versus the `95.00%` threshold;
  - Sphinx docs build;
  - Ruff check/format-check;
  - whitespace check;
  - plotted guarded loop evidence under
    `results/mirror/free_boundary_circular_coils_m157_lsq_loop_plots/`.
- Kept the PR in draft state.
- Verified the remote PR body after editing.

### Results obtained

- Remote PR body verification confirmed:
  - PR number `21`;
  - draft state `true`;
  - head SHA `33fd0de13e0b40734e064251d03e16140a890020`;
  - body contains `section 157`;
  - body contains schema version `0.6`;
  - body contains `Guarded multi-step coupled LS loop`;
  - body contains `95.09%`;
  - body contains `86 passed in 2.97s`.

### How it was tested

Commands run:

```bash
gh pr edit 21 --body-file /tmp/vmec_mirror_pr_body_m157.md
gh pr view 21 --json number,isDraft,headRefOid,body,url | python -c '...'
```

Results:

- PR edit succeeded.
- PR body verification passed.
- PR remained draft.

### File structure and best-practice notes

- No source files changed in this administrative tranche.
- Detailed implementation evidence remains in `plan_mirror.md`; the PR body
  stays a concise review index.
- The PR remains draft as requested.

### Best next steps

1. Commit and push M158 plan log.
2. Let CI run without actively waiting on every check.
3. Inspect CI after useful elapsed time and fix any remaining failures.
4. Continue from the guarded host-side loop toward the true coupled nonlinear
   mirror free-boundary solve lane.

### Completion percentages after M158

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `93%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `92%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `98%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- PR merge readiness overall: `98%`.

### User input needed

No user input is needed.

---
## 159. Reusable Guarded Free-Boundary Least-Squares Loop Controller

### Steps taken

- Added package-level guarded loop primitives to
  `vmec_jax/mirror/free_boundary.py`:
  - `MirrorFreeBoundaryLoopState`;
  - `MirrorFreeBoundaryLoopRow`;
  - `MirrorFreeBoundaryLoopResult`;
  - `mirror_free_boundary_guarded_least_squares_loop`.
- Exposed the new loop controller and dataclasses through
  `vmec_jax.mirror.api` and `vmec_jax.mirror`.
- Designed the loop API around callbacks:
  - a residual callback builds the linearized combined residual around the
    current loop state;
  - an optional trial callback can run a realized fixed-boundary solve and
    return the next state;
  - without a trial callback, the loop uses the LS trial residual directly for
    reduced/synthetic prototypes.
- Added guards for:
  - LS step not accepted;
  - trial merit increase;
  - equilibrium-value growth;
  - target merit;
  - merit stagnation;
  - maximum step count.
- Added synthetic regression tests for:
  - single-step convergence of a linear residual;
  - merit-increase rejection while preserving the previous state;
  - equilibrium-growth rejection;
  - unaccepted LS-step stopping.
- Updated `docs/mirror/overview.rst` and `examples/mirror/README.md` to
  distinguish reusable loop policy from the root example's host-side
  fixed-boundary trial solve and plot/report generation.

### Results obtained

- The free-boundary loop policy is no longer only embedded in the root
  circular-coil example. A reusable state/callback controller now exists in the
  package for reduced prototypes and future coupled solver integration.
- The controller keeps host-side trial solves outside differentiable residual
  APIs, matching the plan's separation between fast CLI orchestration and
  differentiable research kernels.
- The current root example still owns MOUT writing, realized trial plotting,
  and schema-specific JSON rows. Wiring that example onto the reusable
  controller is the next simplification step.
- No new plot artifact was produced in this tranche because the added behavior
  is package-level loop policy tested with synthetic residuals. The latest
  plotted circular-coil loop evidence remains the M157 output under
  `results/mirror/free_boundary_circular_coils_m157_lsq_loop_plots/`.

### How it was tested

Commands run:

```bash
python -m ruff format vmec_jax/mirror/free_boundary.py \
  vmec_jax/mirror/api.py vmec_jax/mirror/__init__.py \
  tests/mirror/test_mirror_free_boundary.py
python -m ruff check vmec_jax/mirror/free_boundary.py \
  vmec_jax/mirror/api.py vmec_jax/mirror/__init__.py \
  tests/mirror/test_mirror_free_boundary.py

JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_free_boundary.py -q
python -m sphinx -W -b html docs docs/_build/html
git diff --check
```

Results:

- Ruff format/check passed on touched Python files.
- Mirror free-boundary tests passed: `90 passed in 2.85s`.
- Sphinx docs build passed with warnings treated as errors.
- Whitespace check passed.

### File structure and best-practice notes

- Reusable loop policy now lives with the other free-boundary residual and LS
  helpers in `vmec_jax/mirror/free_boundary.py`.
- The public mirror API exports the loop types from the existing
  `vmec_jax/mirror/api.py` and `vmec_jax/mirror/__init__.py` surfaces.
- Tests remain in `tests/mirror/test_mirror_free_boundary.py`, beside the
  existing residual/Jacobian/LS-step coverage.
- The root circular-coil example is not expanded further in this tranche; the
  next simplification should adapt `_run_ls_boundary_coupled_loop` to this
  package controller while preserving its schema and plots.

### Best next steps

1. Commit and push M159.
2. Refactor the root circular-coil guarded loop to use
   `mirror_free_boundary_guarded_least_squares_loop` for stop/guard policy.
3. Re-run the schema `0.6` example tests and one plotted loop smoke after that
   wiring change.
4. Then continue toward a true coupled nonlinear free-boundary solve API with a
   clear differentiability boundary.

### Completion percentages after M159

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `93%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `92%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `98%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- PR merge readiness overall: `98%`.

### User input needed

No user input is needed.

---
## 160. Circular-Coil Example Guarded Loop Uses Reusable Controller

### Steps taken

- Refactored `examples/mirror_free_boundary_circular_coils.py` so the
  schema-`0.6` guarded circular-coil LS loop uses
  `mirror_free_boundary_guarded_least_squares_loop` for repeated-step and
  stop/guard policy.
- Extracted the frozen-output LS residual builder into
  `_build_ls_boundary_residual_function`.
- Extracted LS step JSON/plot formatting into
  `_ls_boundary_step_summary_from_step`.
- Kept the root example responsible for physics-specific work:
  - realized fixed-boundary trial solves;
  - MOUT writing/loading;
  - LCFS diagnostic plotting;
  - schema-specific JSON rows and figure paths.
- Preserved the public schema fields and row values expected by the existing
  schema `0.6` tests.
- Ran and visually inspected a plotted one-beta two-step controller-loop smoke.

### Results obtained

- The focused schema regression for the guarded loop still passes after the
  refactor.
- Full mirror example tests still pass after the refactor.
- The plotted controller-loop smoke reports:
  - schema version `0.6`;
  - workflow `ls_boundary_coupled_loop`;
  - free-boundary status `ls_boundary_coupled_loop_not_converged_free_boundary`;
  - `ls_boundary_coupled_loop_rows_total == 2`;
  - `ls_boundary_coupled_loop_accepted_rows_total == 1`;
  - stop counts `{"None": 1, "ls_step_not_accepted": 1}`;
  - row 1 accepted with `fsq_growth_ratio = 0.8804859565965772`;
  - row 1 accepted with `lcfs_merit_ratio = 0.4801851017221157`;
  - row 2 skipped with `ls_step_not_accepted`.
- The plotted smoke rendered:
  - step-1 LS plot:
    `results/mirror/free_boundary_circular_coils_m160_controller_loop_plots/figures/fixed_boundary_beta_1_ls_loop_step_1/free_boundary_circular_coils_beta_1_ls_loop_step_1_ls_boundary_step.png`;
  - step-2 LS plot:
    `results/mirror/free_boundary_circular_coils_m160_controller_loop_plots/figures/fixed_boundary_beta_1_ls_loop_step_2/free_boundary_circular_coils_beta_1_ls_loop_step_2_ls_boundary_step.png`;
  - realized trial LCFS diagnostic:
    `results/mirror/free_boundary_circular_coils_m160_controller_loop_plots/figures/fixed_boundary_beta_1_ls_loop_step_1_ls_boundary_trial/free_boundary_circular_coils_beta_1_ls_loop_step_1_ls_boundary_trial_lcfs_diagnostic.png`;
  - realized trial boundary B-direction and field-line plot:
    `results/mirror/free_boundary_circular_coils_m160_controller_loop_plots/figures/fixed_boundary_beta_1_ls_loop_step_1_ls_boundary_trial/free_boundary_circular_coils_beta_1_ls_loop_step_1_ls_boundary_trial_mirror_bfield_boundary.png`.
- Visual inspection confirmed:
  - the step-1 LS plot selected the full step;
  - the step-2 LS plot selected the no-op factor because every trial increased
    the combined residual;
  - the LCFS diagnostic plot rendered the pressure-balance and normal-field
    profiles;
  - the boundary B-direction plot rendered field arrows and field lines.

### How it was tested

Commands run:

```bash
python -m ruff format examples/mirror_free_boundary_circular_coils.py
python -m ruff check examples/mirror_free_boundary_circular_coils.py \
  tests/mirror/test_mirror_examples.py vmec_jax/mirror/free_boundary.py

JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_ls_boundary_coupled_loop_reports_guarded_steps \
  -q
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_free_boundary.py -q
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_examples.py -q

python examples/mirror_free_boundary_circular_coils.py \
  --outdir results/mirror/free_boundary_circular_coils_m160_controller_loop_plots \
  --betas 1 --ntheta 8 --nxi 11 --n-segments 64 \
  --run-fixed-boundary-baseline --baseline-maxiter 0 \
  --run-ls-boundary-coupled-loop --ls-boundary-coupled-loop-steps 2

git diff --check
```

Results:

- Focused guarded-loop example test passed: `1 passed in 3.37s`.
- Mirror free-boundary tests passed: `90 passed in 2.82s`.
- Full mirror example tests passed: `22 passed in 96.84s`.
- Plotted controller-loop smoke passed and generated the expected PNG outputs.
- Whitespace check passed.

### File structure and best-practice notes

- Reusable stop/guard policy now lives in package code under
  `vmec_jax/mirror/free_boundary.py`.
- The root example keeps only workflow-specific concerns: fixed-boundary trial
  execution, NetCDF output, plotting, and JSON schema projection.
- The refactor avoids changing the schema `0.6` contract and keeps generated
  result trees ignored under `results/`.
- The code is simpler to extend because future true coupled-solve APIs can
  reuse the same controller semantics without depending on the root example.

### Best next steps

1. Commit and push M160.
2. Refresh the draft PR body to mention the reusable controller wiring.
3. Continue toward a true coupled nonlinear solve by replacing finite
   differences with the best available derivative path for the selected
   residual block.
4. Check CI after enough time has elapsed for the latest pushes to produce
   useful results.

### Completion percentages after M160

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `93%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `92%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `98%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- PR merge readiness overall: `98%`.

### User input needed

No user input is needed.

---
## 161. Draft PR Description Synchronization After Reusable Loop Wiring

### Steps taken

- Updated draft PR #21 body after M159/M160.
- Changed the detailed implementation-log pointer from section 157 to
  section 160.
- Added PR-body language that the guarded circular-coil loop is now backed by
  reusable state/callback loop policy from `vmec_jax.mirror.free_boundary`.
- Updated latest validation evidence:
  - `90 passed in 2.85s` for mirror free-boundary tests;
  - `22 passed in 96.84s` for full mirror example tests;
  - retained the prior combined coverage estimate `95.09%` versus the `95.00%`
    coverage gate;
  - plotted controller-loop evidence under
    `results/mirror/free_boundary_circular_coils_m160_controller_loop_plots/`.
- Kept the PR in draft state.
- Verified the remote PR body after editing.

### Results obtained

- Remote PR body verification confirmed:
  - PR number `21`;
  - draft state `true`;
  - head SHA `98e393be779000e62df215f24a217905527dae06`;
  - body contains `section 160`;
  - body contains reusable loop policy wording;
  - body contains `90 passed in 2.85s`;
  - body contains `22 passed in 96.84s`;
  - body contains `free_boundary_circular_coils_m160_controller_loop_plots`.

### How it was tested

Commands run:

```bash
gh pr view 21 --json body -q .body > /tmp/vmec_mirror_pr_body_current.md
gh pr edit 21 --body-file /tmp/vmec_mirror_pr_body_current.md
gh pr view 21 --json number,isDraft,headRefOid,body,url | python -c '...'
```

Results:

- PR edit succeeded.
- PR body verification passed.
- PR remained draft.

### File structure and best-practice notes

- No source files changed in this administrative tranche.
- The PR body remains a concise review index, with detailed implementation
  evidence in `plan_mirror.md`.

### Best next steps

1. Commit and push M161 plan log.
2. Check CI after enough elapsed time for useful status.
3. Continue toward the true coupled nonlinear free-boundary solve API and
   derivative strategy.

### Completion percentages after M161

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `93%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `92%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `98%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- PR merge readiness overall: `98%`.

### User input needed

No user input is needed.

---
## 162. JAX Residual-Vector Jacobian Helper for Free-Boundary Prototypes

### Steps taken

- Added `mirror_free_boundary_residual_vector_jacobian_jax` to
  `vmec_jax/mirror/free_boundary.py`.
- Exported it through `vmec_jax.mirror.api` and the flat `vmec_jax.mirror`
  namespace.
- The helper accepts a pure JAX residual-vector callback and returns:
  - the residual vector at the supplied coefficients;
  - the dense Jacobian with respect to those coefficients.
- Added derivative-mode selection:
  - `mode="forward"` uses `jax.jacfwd`;
  - `mode="reverse"` uses `jax.jacrev`;
  - `mode="auto"` chooses forward mode when parameter count is no larger than
    residual-vector length, otherwise reverse mode.
- Added validation for empty/nonfinite coefficients, empty/nonfinite residual
  vectors, nonfinite Jacobians, and invalid modes.
- Added tests comparing forward, reverse, and automatic JAX Jacobians against
  an analytic nonlinear residual-vector model.
- Updated `docs/mirror/overview.rst` and `examples/mirror/README.md` to note
  that pure-JAX reduced residual prototypes can use the JAX Jacobian helper
  beside the host-side finite-difference helper.

### Results obtained

- The free-boundary lane now has both:
  - host-side finite-difference Jacobians for CLI/example workflows with file
    I/O or realized fixed-boundary trials;
  - JAX forward/reverse Jacobians for differentiable reduced residual-vector
    prototypes.
- This is a direct step toward the derivative-strategy plan without forcing
  the current host-side circular-coil example to be differentiable end to end.
- No new plot artifact was produced in this tranche because the added behavior
  is a derivative helper with analytic unit-test comparisons.

### How it was tested

Commands run:

```bash
python -m ruff format vmec_jax/mirror/free_boundary.py \
  vmec_jax/mirror/api.py vmec_jax/mirror/__init__.py \
  tests/mirror/test_mirror_free_boundary.py
python -m ruff check vmec_jax/mirror/free_boundary.py \
  vmec_jax/mirror/api.py vmec_jax/mirror/__init__.py \
  tests/mirror/test_mirror_free_boundary.py

JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_free_boundary.py -q
python -m sphinx -W -b html docs docs/_build/html
git diff --check
```

Results:

- Ruff format/check passed on touched Python files.
- Mirror free-boundary tests passed: `94 passed in 3.53s`.
- Sphinx docs build passed with warnings treated as errors.
- Whitespace check passed.

### File structure and best-practice notes

- The derivative helper lives beside the existing finite-difference residual
  Jacobian helper in `vmec_jax/mirror/free_boundary.py`.
- The API is deliberately vector-function based, so it does not require
  differentiating through `MirrorFreeBoundaryResidual` dataclass construction or
  host-side trial solves.
- The helper is small and explicit about mode selection, which makes it easier
  to compare against future implicit/adjoint implementations.

### Best next steps

1. Commit and push M162.
2. Add a reduced free-boundary least-squares step path that can use either the
   finite-difference Jacobian or the new JAX Jacobian helper.
3. Benchmark forward vs reverse mode on a small reduced mirror residual and
   document when each path is appropriate.
4. Re-check CI once the latest pushed checks are available.

### Completion percentages after M162

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `93%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `93%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `98%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- PR merge readiness overall: `98%`.

### User input needed

No user input is needed.

---
## 163. Reduced Vector Least-Squares Step With Finite-Difference or JAX Jacobian Backend

### Steps taken

- Added `MirrorFreeBoundaryVectorLeastSquaresStep` to
  `vmec_jax/mirror/free_boundary.py`.
- Added `mirror_free_boundary_residual_vector_jacobian_finite_difference` for
  central-difference residual-vector Jacobians.
- Added `mirror_free_boundary_residual_vector_least_squares_step`, a damped
  line-searched reduced free-boundary LS step that can use:
  - `jacobian_backend="finite_difference"`;
  - `jacobian_backend="jax"` with `jax_mode="auto"`, `"forward"`, or
    `"reverse"`.
- Exported the new dataclass and helpers through `vmec_jax.mirror.api` and the
  flat `vmec_jax.mirror` namespace.
- Added analytic tests for:
  - finite-difference vector-Jacobian correctness on a linear model;
  - finite-difference and JAX backend agreement for one LS step;
  - rejected worse-trial line search behavior;
  - invalid coefficient, backend, line-search, and residual-vector outputs.
- Updated `docs/mirror/overview.rst` and `examples/mirror/README.md` to mention
  the selectable reduced LS backend.

### Results obtained

- Reduced free-boundary prototypes can now use the same damped, line-searched
  LS semantics with either host-side finite differences or JAX autodiff.
- The host-side circular-coil example remains on its tested combined-residual
  route, while pure JAX residual-vector prototypes now have a first-class
  update path that is easier to benchmark and replace with implicit/adjoint
  derivatives later.
- No new plot artifact was produced in this tranche because the added behavior
  is a numerical derivative/update helper with analytic unit-test comparisons.

### How it was tested

Commands run:

```bash
python -m ruff format vmec_jax/mirror/free_boundary.py \
  vmec_jax/mirror/api.py vmec_jax/mirror/__init__.py \
  tests/mirror/test_mirror_free_boundary.py
python -m ruff check vmec_jax/mirror/free_boundary.py \
  vmec_jax/mirror/api.py vmec_jax/mirror/__init__.py \
  tests/mirror/test_mirror_free_boundary.py

JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_free_boundary.py -q
python -m sphinx -W -b html docs docs/_build/html
git diff --check
```

Results:

- Ruff format/check passed on touched Python files.
- Mirror free-boundary tests passed: `108 passed in 3.61s`.
- Sphinx docs build passed with warnings treated as errors.
- Whitespace check passed.

### File structure and best-practice notes

- The reduced vector LS helper lives beside the existing combined-residual LS
  helper in `vmec_jax/mirror/free_boundary.py`.
- The function is intentionally vector-function based, keeping it independent
  of host-side MOUT writing, example plotting, and dataclass residual assembly.
- The API exposes the derivative backend explicitly, which supports future
  benchmarking of finite differences, forward/reverse autodiff,
  implicit-differentiation, and adjoint routes without changing example
  orchestration.

### Best next steps

1. Commit and push M163.
2. Add a small benchmark/example comparing finite-difference, forward-JAX, and
   reverse-JAX reduced LS steps on a low-dimensional mirror free-boundary
   residual.
3. Use that benchmark to decide the default derivative route for reduced
   differentiable free-boundary prototypes.
4. Check CI once the current run has useful completed jobs.

### Completion percentages after M163

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `93%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `93%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `98%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- PR merge readiness overall: `98%`.

### User input needed

No user input is needed.

---
## 164. Reduced Vector LS Backend Benchmark Example and Plots

### Steps taken

- Added root example
  `examples/mirror_free_boundary_vector_ls_benchmark.py`.
- The example compares one reduced mirror free-boundary residual-vector LS step
  across four derivative routes:
  - `finite_difference`;
  - `jax_forward`;
  - `jax_reverse`;
  - `jax_auto`.
- The reduced residual uses polynomial side-boundary coefficients
  `[r0, a2, a4]`, a target radius profile, and a slope residual term on a
  compact `xi` grid.
- Added compact metrics schema
  `mirror_free_boundary_vector_ls_benchmark` version `0.1`.
- Added `validate_vector_ls_benchmark_metrics` for downstream and test-side
  schema validation.
- Added optional plots:
  - backend residual/error/runtime summary;
  - target/initial/updated radius profiles.
- Added a root example regression test that runs the benchmark with
  `--no-plots`, validates the metrics schema, and checks finite-difference/JAX
  backend agreement.
- Updated the mirror example README and docs in M163 for the reduced LS
  backend path.

### Results obtained

- The no-plot smoke run completed:
  `results/mirror/free_boundary_vector_ls_benchmark_m164_smoke/mirror_free_boundary_vector_ls_benchmark_metrics.json`.
- The plotted run completed:
  `results/mirror/free_boundary_vector_ls_benchmark_m164_plots/mirror_free_boundary_vector_ls_benchmark_metrics.json`.
- Plotted run metrics reported:
  - schema version `0.1`;
  - all four derivative routes accepted the full line-search factor `1.0`;
  - residual RMS decreased from `0.23213479614173216` to about
    `0.0410754553836`;
  - coefficient error decreased to about `0.06472162613`;
  - finite-difference, JAX forward, JAX reverse, and JAX auto produced matching
    updated coefficients to test tolerance.
- Generated and visually inspected:
  - `results/mirror/free_boundary_vector_ls_benchmark_m164_plots/figures/mirror_free_boundary_vector_ls_backend_summary.png`;
  - `results/mirror/free_boundary_vector_ls_benchmark_m164_plots/figures/mirror_free_boundary_vector_ls_radius_profiles.png`.
- The summary plot shows residual/error agreement across backends and runtime
  differences. The radius-profile plot shows the initial profile moving toward
  the target profile after one reduced LS step.

### How it was tested

Commands run:

```bash
python -m ruff format examples/mirror_free_boundary_vector_ls_benchmark.py \
  tests/mirror/test_mirror_examples.py
python -m ruff check examples/mirror_free_boundary_vector_ls_benchmark.py \
  tests/mirror/test_mirror_examples.py

python examples/mirror_free_boundary_vector_ls_benchmark.py \
  --outdir results/mirror/free_boundary_vector_ls_benchmark_m164_smoke \
  --nxi 9 --no-plots

JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_vector_ls_benchmark_runs_without_plots \
  -q

python examples/mirror_free_boundary_vector_ls_benchmark.py \
  --outdir results/mirror/free_boundary_vector_ls_benchmark_m164_plots \
  --nxi 17

JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_free_boundary.py -q
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_examples.py -q
python -m sphinx -W -b html docs docs/_build/html
git diff --check
```

Results:

- New benchmark example no-plot smoke passed.
- New benchmark example plotted smoke passed and generated PNG outputs.
- Focused benchmark example test passed: `1 passed in 2.09s`.
- Mirror free-boundary tests passed: `108 passed in 3.53s`.
- Full mirror example tests passed: `23 passed in 98.10s`.
- Sphinx docs build passed with warnings treated as errors.
- Whitespace check passed.

### File structure and best-practice notes

- The benchmark is a root `examples/` script because it is a user-facing
  derivative-route comparison for the free-boundary lane.
- Generated plots/metrics remain under ignored `results/` paths and are not
  committed.
- The example is deliberately reduced and pure-JAX-compatible, so it can guide
  derivative-route selection without conflating host-side MOUT writing or
  realized fixed-boundary trial solves.

### Best next steps

1. Commit and push M164.
2. Refresh the draft PR body for the new reduced vector LS backend benchmark.
3. Use the benchmark evidence to set a documented default derivative route for
   reduced free-boundary prototypes.
4. Check CI after the latest push has useful completed jobs.

### Completion percentages after M164

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `93%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `93%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `99%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- PR merge readiness overall: `98%`.

### User input needed

No user input is needed.

---
## 165. Draft PR Description Synchronization After Reduced Vector LS Benchmark

### Steps taken

- Updated draft PR #21 body after M162-M164.
- Changed the detailed implementation-log pointer from section 160 to
  section 164.
- Added PR-body language for the reduced residual-vector
  finite-difference/JAX least-squares helpers and benchmark example.
- Updated latest validation evidence:
  - `108 passed in 3.53s` for mirror free-boundary tests;
  - `23 passed in 98.10s` for full mirror example tests;
  - retained the prior combined coverage estimate `95.09%` versus the `95.00%`
    coverage gate;
  - plotted reduced vector LS backend benchmark evidence under
    `results/mirror/free_boundary_vector_ls_benchmark_m164_plots/`.
- Kept the PR in draft state.
- Verified the remote PR body after editing.

### Results obtained

- Remote PR body verification confirmed:
  - PR number `21`;
  - draft state `true`;
  - head SHA `bc11a9cfa0886f5cd565aad9465356cd978cfad6`;
  - body contains `section 164`;
  - body contains reduced residual-vector finite-difference/JAX LS wording;
  - body contains `108 passed in 3.53s`;
  - body contains `23 passed in 98.10s`;
  - body contains `free_boundary_vector_ls_benchmark_m164_plots`.

### How it was tested

Commands run:

```bash
gh pr view 21 --json body -q .body > /tmp/vmec_mirror_pr_body_current.md
gh pr edit 21 --body-file /tmp/vmec_mirror_pr_body_current.md
gh pr view 21 --json number,isDraft,headRefOid,body,url | python -c '...'
```

Results:

- PR edit succeeded.
- PR body verification passed.
- PR remained draft.

### File structure and best-practice notes

- No source files changed in this administrative tranche.
- The PR body remains a concise review index; detailed evidence stays in
  `plan_mirror.md`.

### Best next steps

1. Commit and push M165 plan log.
2. Check CI once completed jobs are available.
3. Use M164 benchmark evidence to document the default derivative route for
   reduced differentiable free-boundary prototypes.

### Completion percentages after M165

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `93%`.
- I/O schema and docs: `99%`.
- Differentiable solved-state API: `93%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `99%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- PR merge readiness overall: `98%`.

### User input needed

No user input is needed.
---
## 166. Documented Free-Boundary Derivative-Backend Selection

### Steps taken

- Re-checked the active repository and draft PR state after resuming:
  - branch `codex/mirror-geometry`;
  - PR #21 still draft;
  - head SHA `3372029339498762d82c639d63fb685a1438cd39`;
  - GitHub reported all current checks passing, including the combined coverage
    gate, docs, fast tests, physics smoke, build, and Codecov statuses.
- Confirmed that `/Users/rogeriojorge/local/vmec_mirror/plan_mirror.md` is the
  active single plan; the original Downloads attachment is stale.
- Added user-facing derivative-route guidance to the mirror example README.
- Added matching Sphinx overview guidance for free-boundary derivative backend
  selection.

### Results obtained

- The documented default is now explicit:
  - use finite-difference Jacobians for current host-side CLI loops that call
    fixed-boundary trial solves, write MOUT files, or use plotting/report
    callbacks;
  - use `jacobian_backend="jax"` with `jax_mode="auto"` for reduced residual
    vector prototypes that are already pure JAX functions of boundary
    parameters;
  - automatic JAX mode uses forward differentiation when the number of boundary
    parameters is no larger than the residual-vector length and reverse
    differentiation for smaller residual or scalar-like targets.
- The docs now point to `examples/mirror_free_boundary_vector_ls_benchmark.py`
  as the benchmark that compares finite-difference, JAX forward, JAX reverse,
  and JAX automatic derivative routes.

### How it was tested

Commands run:

```bash
gh pr view 21 --repo uwplasma/vmec_jax --json number,title,isDraft,headRefName,headRefOid,baseRefName,mergeStateStatus,url
gh pr checks 21 --repo uwplasma/vmec_jax --json name,state,bucket,startedAt,completedAt,link
python -m sphinx -W -b html docs docs/_build/html
git diff --check
```

Results:

- PR state check confirmed draft PR #21 on the expected branch and SHA.
- GitHub checks were all passing on the current head; the manual/nightly full
  physics job was skipped as configured.
- Sphinx docs build passed with warnings treated as errors.
- Whitespace check passed.

### File structure and best-practice notes

- `examples/mirror/README.md` now carries practical example-level guidance for
  derivative backend choice.
- `docs/mirror/overview.rst` carries the same policy at package-doc level.
- No generated benchmark outputs or figures were committed.
- The documentation separates fast CLI runtime concerns from the differentiable
  pure-JAX residual-vector path, matching the source structure in
  `vmec_jax/mirror/free_boundary.py`.

### Best next steps

1. Commit and push M166.
2. Refresh the PR body to mention the derivative-backend policy if the body
   does not already point reviewers to section 166.
3. Continue the next open implementation lane, prioritizing the
   Mirror-Boozer-like diagnostics lane because it remains the lowest-completion
   source lane.

### Completion percentages after M166

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `82%`.
- Plotting and `vmec --plot` mirror support: `93%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `94%`.
- Mirror-Boozer-like diagnostics: `36%`.
- Free-boundary mirror lane: `99%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- PR merge readiness overall: `98%`.

### User input needed

No user input is needed.

---
## 167. Mirror-Boozer-Like Surface Diagnostics in Standard Plot and Export Paths

### Steps taken

- Added `MirrorBoozerLikeDiagnosticsData` and
  `mirror_boozer_like_diagnostics_data` in
  `vmec_jax/mirror/plotting/diagnostics.py`.
- The new helper computes mirror-native, Jacobian-weighted surface diagnostics:
  - surface measure;
  - flux-surface averaged, minimum, and maximum `|B|`;
  - per-surface mirror ratio;
  - normalized `|B|` ripple RMS;
  - `I'/Psi'` twist proxy;
  - cap-to-cap field-line turns;
  - contravariant pitch mean and RMS;
  - covariant pitch ratio;
  - magnetic-well proxy from the Jacobian-weighted `|B|` average.
- Added `write_mirror_boozer_like_diagnostics` and wired it into the standard
  `plot_mirror_output` bundle under the key `boozer_like_diagnostics`.
- Extended lightweight `.npz` and axisymmetric CSV exports with the
  mirror-Boozer-like profile fields.
- Exported the new plotting dataclass/helper/writer from
  `vmec_jax/mirror/plotting/__init__.py`.
- Updated `docs/mirror/overview.rst`, `docs/mirror/outputs.rst`, and
  `examples/mirror/README.md` so the standard plot/export documentation names
  the new diagnostics and explicitly states that these are not toroidal Boozer
  coordinates.

### Results obtained

- `vmec --plot` and `plot_mirror_output` now produce a six-panel
  `*_mirror_boozer_like_diagnostics.png` plot for mirror `mout_*.nc` files.
- The plot shows:
  - `|B|` surface average/min/max;
  - mirror ratio and normalized ripple;
  - open-field pitch from `I'/Psi'` and measured cap-to-cap turns;
  - contravariant pitch mean/RMS;
  - covariant pitch proxy;
  - magnetic-well proxy.
- Zero-current two-coil smoke shows zero pitch panels, as expected.
- Finite-current smoke shows visible pitch:
  - cap-to-cap turns from `0.5416456906746907` to `0.5464761155828083`;
  - contravariant pitch mean from `1.7111044930933443` to
    `1.7168053500772373`;
  - maximum pitch RMS `0.14859708176458764`.
- Visual plots inspected:
  - `results/mirror/boozer_like_m167_smoke/figures/two_coil_axisym_mirror_boozer_like_diagnostics.png`;
  - `results/mirror/boozer_like_m167_finite_current_smoke/figures/finite_current_pitch_mirror_boozer_like_diagnostics.png`.

### How it was tested

Commands run:

```bash
python -m ruff format vmec_jax/mirror/plotting/diagnostics.py \
  vmec_jax/mirror/plotting/export.py \
  vmec_jax/mirror/plotting/__init__.py \
  tests/mirror/test_mirror_plotting.py \
  tests/mirror/test_mirror_io.py
python -m ruff check vmec_jax/mirror/plotting/diagnostics.py \
  vmec_jax/mirror/plotting/export.py \
  vmec_jax/mirror/plotting/__init__.py \
  tests/mirror/test_mirror_plotting.py \
  tests/mirror/test_mirror_io.py
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_plotting.py \
  tests/mirror/test_mirror_io.py -q
python -m sphinx -W -b html docs docs/_build/html
JAX_ENABLE_X64=1 python examples/mirror_two_coil_axisym.py \
  --outdir results/mirror/boozer_like_m167_smoke --ns 5 --nxi 9 --maxiter 2
JAX_ENABLE_X64=1 python examples/mirror_finite_current_pitch.py \
  --outdir results/mirror/boozer_like_m167_finite_current_smoke \
  --ns 5 --nxi 9 --maxiter 2
python - <<'PY'
from pathlib import Path
import numpy as np
from vmec_jax.mirror.io.mout import load_mirror_output
from vmec_jax.mirror.plotting.diagnostics import mirror_boozer_like_diagnostics_data
output = load_mirror_output(Path(
    "results/mirror/boozer_like_m167_finite_current_smoke/mout_finite_current_pitch.nc"
))
data = mirror_boozer_like_diagnostics_data(output)
print(float(np.min(data.field_line_turns)), float(np.max(data.field_line_turns)))
print(float(np.min(data.contravariant_pitch_mean)), float(np.max(data.contravariant_pitch_mean)))
print(float(np.max(data.contravariant_pitch_rms)))
PY
git diff --check
```

Results:

- Ruff format/check passed.
- Focused plotting and I/O tests passed: `8 passed in 6.30s`.
- Sphinx docs build passed with warnings treated as errors.
- Two-coil plotted smoke completed and wrote the new Boozer-like plot.
- Finite-current plotted smoke completed and wrote the new Boozer-like plot.
- Direct helper readback from the finite-current MOUT confirmed nonzero pitch
  profiles.
- Whitespace check passed.

### File structure and best-practice notes

- Numerical profile construction stays in
  `vmec_jax/mirror/plotting/diagnostics.py` beside the existing radial,
  pressure, Jacobian, and residual-history diagnostics.
- High-level plot/NPZ/CSV wiring stays in `vmec_jax/mirror/plotting/export.py`.
- Public plotting imports stay centralized in
  `vmec_jax/mirror/plotting/__init__.py`.
- Tests are focused in the existing plotting and I/O test modules; no new test
  file was needed.
- Generated figures and MOUT files remain under ignored `results/` paths and
  are not committed.
- The docs avoid claiming real toroidal Boozer coordinates for open mirrors,
  which keeps the terminology physically honest while still making the lane
  useful for profile comparisons.

### Best next steps

1. Commit and push M167.
2. Re-check PR checks once the queued docs commit has run long enough to expose
   failures, but do not block on CI if it is still queued.
3. Continue the Mirror-Boozer-like lane with a compact example/JSON summary
   that records these profile extrema for two-coil and finite-current cases.
4. Then move to final merge-readiness cleanup: plan audit, public API audit,
   docs index audit, and a targeted all-mirror test pass.

### Completion percentages after M167

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `89%`.
- Finite-current pitch validation: `84%`.
- Plotting and `vmec --plot` mirror support: `96%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `94%`.
- Mirror-Boozer-like diagnostics: `64%`.
- Free-boundary mirror lane: `99%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- PR merge readiness overall: `98%`.

### User input needed

No user input is needed.

---
## 168. Boozer-Like Summary Metrics in Two-Coil and Finite-Current Examples

### Steps taken

- Added `mirror_boozer_like_summary_metrics` beside the profile-data helper in
  `vmec_jax/mirror/plotting/diagnostics.py`.
- Exported the summary helper from `vmec_jax/mirror/plotting/__init__.py`.
- Wired the summary metrics into:
  - `examples/mirror_two_coil_axisym.py`;
  - `examples/mirror_finite_current_pitch.py`.
- Updated example tests so the no-plot smoke paths assert the new JSON fields.
- Updated the example README to note that no-plot benchmark metrics retain the
  mirror-Boozer-like surface-average, ripple, mirror-ratio, pitch, and
  well-proxy summaries.

### Results obtained

- The two-coil and finite-current root example JSON files now include compact
  scalar summaries:
  - surface-averaged `|B|` min/max;
  - global `|B|` min/max;
  - surface mirror-ratio min/max;
  - maximum normalized `|B|` ripple RMS;
  - `I'/Psi'` mean/min/max;
  - field-line turns mean/min/max;
  - contravariant pitch mean range and pitch-RMS maximum;
  - covariant pitch-ratio range;
  - magnetic-well-proxy range.
- Low-resolution plotted smoke values:
  - two-coil vacuum:
    - `boozer_like_surface_mirror_ratio_max = 15.450964869176072`;
    - `boozer_like_field_line_turns_mean = 0.0`;
    - `boozer_like_contravariant_pitch_rms_max = 0.0`;
    - `boozer_like_bmag_ripple_rms_max = 1.0785093792282217`;
  - finite current:
    - `boozer_like_surface_mirror_ratio_max = 13.940071600691715`;
    - `boozer_like_field_line_turns_mean = 0.5430244027554318`;
    - `boozer_like_contravariant_pitch_rms_max = 0.14859708176458764`;
    - `boozer_like_bmag_ripple_rms_max = 1.0645886648294252`.
- Generated plotted smoke figures:
  - `results/mirror/boozer_like_m168_two_coil_smoke/figures/two_coil_axisym_mirror_boozer_like_diagnostics.png`;
  - `results/mirror/boozer_like_m168_finite_current_smoke/figures/finite_current_pitch_mirror_boozer_like_diagnostics.png`.

### How it was tested

Commands run:

```bash
python -m ruff format vmec_jax/mirror/plotting/diagnostics.py \
  vmec_jax/mirror/plotting/__init__.py \
  examples/mirror_two_coil_axisym.py \
  examples/mirror_finite_current_pitch.py \
  tests/mirror/test_mirror_plotting.py \
  tests/mirror/test_mirror_examples.py
python -m ruff check vmec_jax/mirror/plotting/diagnostics.py \
  vmec_jax/mirror/plotting/__init__.py \
  examples/mirror_two_coil_axisym.py \
  examples/mirror_finite_current_pitch.py \
  tests/mirror/test_mirror_plotting.py \
  tests/mirror/test_mirror_examples.py
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_plotting.py \
  tests/mirror/test_mirror_examples.py::test_root_two_coil_axisym_example_runs_without_plots \
  tests/mirror/test_mirror_examples.py::test_root_finite_current_pitch_example_runs_without_plots \
  -q
python -m sphinx -W -b html docs docs/_build/html
JAX_ENABLE_X64=1 python examples/mirror_two_coil_axisym.py \
  --outdir results/mirror/boozer_like_m168_two_coil_smoke \
  --ns 5 --nxi 9 --maxiter 2
JAX_ENABLE_X64=1 python examples/mirror_finite_current_pitch.py \
  --outdir results/mirror/boozer_like_m168_finite_current_smoke \
  --ns 5 --nxi 9 --maxiter 2
```

Results:

- Ruff format/check passed.
- Focused tests passed: `6 passed in 7.78s`.
- Sphinx docs build passed with warnings treated as errors.
- Both plotted smokes completed and wrote the new diagnostic figures.
- JSON metric readback confirmed the expected zero-pitch and finite-pitch
  behavior.

### File structure and best-practice notes

- The scalar summary helper reuses `mirror_boozer_like_diagnostics_data` rather
  than duplicating profile logic in examples.
- Existing example metrics JSON files were extended in place; no new root
  artifact type was introduced.
- Tests live in existing plotting/example suites and check both low-level
  helper output and user-facing JSON fields.
- Generated MOUT/PNG/JSON smoke outputs remain under ignored `results/` paths.

### Best next steps

1. Commit and push M168.
2. Re-check PR checks later, after GitHub attaches jobs for the latest head.
3. Continue final open lanes with a merge-readiness audit:
   - public mirror API/export audit;
   - docs index and README audit;
   - targeted full `tests/mirror` pass;
   - PR body refresh to point reviewers to M168 and the remaining known gaps.

### Completion percentages after M168

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `90%`.
- Finite-current pitch validation: `86%`.
- Plotting and `vmec --plot` mirror support: `96%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `94%`.
- Mirror-Boozer-like diagnostics: `74%`.
- Free-boundary mirror lane: `99%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- PR merge readiness overall: `98%`.

### User input needed

No user input is needed.

---
## 169. Public API and Docs Index Merge-Readiness Audit

### Steps taken

- Ran a full `tests/mirror` pass at the start of the merge-readiness audit.
- Audited `vmec_jax/mirror/api.py`, `vmec_jax/mirror/__init__.py`, and
  `vmec_jax/mirror/plotting/__init__.py` for the new mirror-Boozer-like
  helpers.
- Promoted the new Boozer-like diagnostics through the main mirror public API:
  - `MirrorBoozerLikeDiagnosticsData`;
  - `mirror_boozer_like_diagnostics_data`;
  - `mirror_boozer_like_summary_metrics`;
  - `write_mirror_boozer_like_diagnostics`.
- Added a public-import assertion in `tests/mirror/test_mirror_plotting.py`.
- Updated `docs/mirror/index.rst` so the mirror documentation index names the
  cap-to-cap pitch and mirror-Boozer-like diagnostic lane and fixes stale
  wording in the experimental warning.
- Ran a second full `tests/mirror` pass after the API/doc/test patch so the
  final evidence matches the current worktree.

### Results obtained

- `vmec_jax.mirror` and `vmec_jax.mirror.api` now expose the same
  mirror-Boozer-like public helper names.
- A lightweight import audit confirmed every `vmec_jax.mirror.__all__` and
  `vmec_jax.mirror.api.__all__` name exists and the new top-level helpers are
  the same objects as their `api` exports.
- The mirror docs index now matches the current diagnostic surface.
- Current-state full mirror suite passed.

### How it was tested

Commands run:

```bash
JAX_ENABLE_X64=1 pytest tests/mirror -q
python -m ruff format --check vmec_jax/mirror/api.py \
  vmec_jax/mirror/__init__.py \
  tests/mirror/test_mirror_plotting.py
python -m ruff check vmec_jax/mirror/api.py \
  vmec_jax/mirror/__init__.py \
  tests/mirror/test_mirror_plotting.py
python -m sphinx -W -b html docs docs/_build/html
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_plotting.py -q
python - <<'PY'
import vmec_jax.mirror as mirror
import vmec_jax.mirror.api as api
for module in (mirror, api):
    missing = [name for name in module.__all__ if not hasattr(module, name)]
    if missing:
        raise SystemExit(f"{module.__name__} missing {missing}")
for name in [
    "MirrorBoozerLikeDiagnosticsData",
    "mirror_boozer_like_diagnostics_data",
    "mirror_boozer_like_summary_metrics",
    "write_mirror_boozer_like_diagnostics",
]:
    assert getattr(mirror, name) is getattr(api, name)
print("mirror_public_api_ok")
PY
JAX_ENABLE_X64=1 pytest tests/mirror -q
```

Results:

- First broad mirror suite passed: `218 passed, 1 skipped in 194.09s`.
- Ruff format/check passed on the API/test patch.
- Sphinx docs build passed with warnings treated as errors.
- Focused plotting/API test passed: `4 passed in 5.17s`.
- Public API consistency check printed `mirror_public_api_ok`.
- Current-state broad mirror suite passed:
  `218 passed, 1 skipped in 191.77s`.

### File structure and best-practice notes

- The numerical implementation remains in the plotting diagnostics module.
- The top-level mirror API now exposes the user-facing data/helper/writer names
  without moving implementation code.
- The docs index remains concise and points readers to the existing overview,
  outputs, and differentiability pages rather than adding another near-empty
  page.
- No generated results were committed.

### Best next steps

1. Commit and push M169.
2. Refresh the draft PR body so reviewers see the M167-M169 diagnostics/API
   work and current full mirror-suite result.
3. Re-check GitHub checks after the latest head has jobs attached.
4. Continue final plan closure with a concise remaining-gap audit across:
   straight-axis hybrid fixture, final free-boundary solve status, and PR
   readiness.

### Completion percentages after M169

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `90%`.
- Finite-current pitch validation: `86%`.
- Plotting and `vmec --plot` mirror support: `97%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `94%`.
- Mirror-Boozer-like diagnostics: `82%`.
- Free-boundary mirror lane: `99%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- PR merge readiness overall: `99%`.

### User input needed

No user input is needed.

---
## 170. Draft PR Body Synchronization After Boozer-Like Diagnostics Audit

### Steps taken

- Updated draft PR #21 body after M167-M169.
- Changed the detailed implementation-log pointer from section 164 to
  section 169.
- Updated the current-content summary to include mirror-Boozer-like
  surface-average/pitch diagnostic plots and compact JSON summary metrics.
- Marked the Mirror-Boozer-like diagnostics checklist item complete in the PR
  body for the plot, export, public API, and example JSON summary paths.
- Replaced the latest-validation command block with the current full
  `tests/mirror` pass, Sphinx, ruff, public API consistency, and whitespace
  checks.
- Added the current full mirror-suite result:
  `218 passed, 1 skipped in 191.77s`.
- Added the public API consistency check result `mirror_public_api_ok`.
- Added plotted Boozer-like smoke evidence paths from M168.
- Re-checked PR state and CI status after the latest push.

### Results obtained

- PR body verification confirmed:
  - PR number `21`;
  - draft state `true`;
  - head SHA `a15ad6f9719f8285d527c33ef0a4f3ecde4f88eb`;
  - body contains `section 169`;
  - body contains the checked Mirror-Boozer-like checklist item;
  - body contains `218 passed, 1 skipped in 191.77s`;
  - body contains `mirror_public_api_ok`;
  - body contains the M168 Boozer-like finite-current plot path.
- GitHub checks are attached to the latest head and running. The nonblocking
  snapshot showed `Parity Manifest Smoke (dry-run)` passing, `Physics Full
  (manual/nightly)` skipped as configured, and the remaining standard jobs in
  progress.

### How it was tested

Commands run:

```bash
gh pr view 21 --repo uwplasma/vmec_jax --json body -q .body > /tmp/vmec_mirror_pr_body_current.md
python - <<'OUTERPY'
# edit PR body text in /tmp
OUTERPY
gh pr edit 21 --repo uwplasma/vmec_jax --body-file /tmp/vmec_mirror_pr_body_current.md
gh pr view 21 --repo uwplasma/vmec_jax --json number,isDraft,headRefOid,mergeStateStatus,body,url > /tmp/vmec_mirror_pr_view.json
python - <<'PY'
# verify required body markers
PY
gh pr checks 21 --repo uwplasma/vmec_jax --json name,state,bucket,startedAt,completedAt,link
git status -sb
```

Results:

- PR edit succeeded.
- PR body verification passed.
- PR remained draft.
- Worktree was clean before this plan-only M170 log.

### File structure and best-practice notes

- No source files changed in this administrative tranche.
- The PR body remains a concise review index while detailed evidence stays in
  `plan_mirror.md`.
- CI was sampled once after jobs attached; the workflow is intentionally not
  being polled continuously.

### Best next steps

1. Commit and push M170 plan log.
2. Continue final plan closure with the remaining-gap audit:
   - straight-axis hybrid fixture lane;
   - final free-boundary nonlinear solve status;
   - final PR readiness and known limitations.
3. Re-check CI after additional implementation or after enough time has passed
   to make a failure actionable.

### Completion percentages after M170

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `90%`.
- Finite-current pitch validation: `86%`.
- Plotting and `vmec --plot` mirror support: `97%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `94%`.
- Mirror-Boozer-like diagnostics: `83%`.
- Free-boundary mirror lane: `99%`.
- Straight-axis hybrid fixture lane: `25%`.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- PR merge readiness overall: `99%`.

### User input needed

No user input is needed.

---
## 171. Straight-Axis Hybrid Fixture Closure as Support Scope

### Steps taken

- Audited the straight-axis hybrid fixture against the corrected user target
  that the final stellarator-mirror hybrid remains toroidal.
- Kept `examples/mirror_stellarator_hybrid_boundary.py` as a straight-axis,
  open-ended support fixture rather than a production hybrid target.
- Added explicit machine-readable labels to the straight-axis example metrics:
  - `hybrid_fixture_kind = "straight_axis_open_mirror_support_fixture"`;
  - `final_hybrid_target_kind = "toroidal_stellarator_mirror_hybrid"`;
  - `production_hybrid_claim = false`;
  - a short `hybrid_scope_note`.
- Added matching target labels to
  `examples/toroidal_stellarator_mirror_hybrid.py` metrics.
- Updated tests so both straight-axis and toroidal example metrics assert the
  new scope labels.
- Updated the mirror example README and mirror overview docs to state that the
  straight-axis fixture is a support fixture and that the final hybrid target
  is the toroidal lane.

### Results obtained

- The straight-axis hybrid lane is no longer an ambiguous open production lane.
  It is closed as a support fixture for boundary, solver, and plotting stress
  tests.
- The toroidal hybrid lane remains the authoritative final hybrid target.
- Low-resolution plotted smoke outputs confirmed both metrics labels and plots:
  - straight-axis fixture:
    `results/mirror/hybrid_label_m171_straight_axis_smoke/stellarator_hybrid_boundary_metrics.json`;
  - toroidal fixture:
    `results/toroidal_stellarator_mirror_hybrid_m171_label_smoke/toroidal_stellarator_mirror_hybrid_metrics.json`.
- Visual plots inspected:
  - `results/mirror/hybrid_label_m171_straight_axis_smoke/figures/stellarator_hybrid_boundary_mirror_boundary_3d.png`;
  - `results/toroidal_stellarator_mirror_hybrid_m171_label_smoke/figures/toroidal_hybrid_lcfs_3d.png`.

### How it was tested

Commands run:

```bash
python -m ruff format --check examples/mirror_stellarator_hybrid_boundary.py \
  examples/toroidal_stellarator_mirror_hybrid.py \
  tests/mirror/test_mirror_examples.py \
  tests/test_toroidal_hybrid.py
python -m ruff check examples/mirror_stellarator_hybrid_boundary.py \
  examples/toroidal_stellarator_mirror_hybrid.py \
  tests/mirror/test_mirror_examples.py \
  tests/test_toroidal_hybrid.py
python -m sphinx -W -b html docs docs/_build/html
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_examples.py::test_root_stellarator_hybrid_boundary_example_runs_without_plots \
  -q
JAX_ENABLE_X64=1 pytest \
  tests/test_toroidal_hybrid.py::test_toroidal_hybrid_example_runs_without_plots \
  -q
JAX_ENABLE_X64=1 python examples/mirror_stellarator_hybrid_boundary.py \
  --outdir results/mirror/hybrid_label_m171_straight_axis_smoke \
  --ns 5 --ntheta 13 --nxi 17 --mpol 4 --maxiter 0
PYTHONPATH=/Users/rogeriojorge/local/vmec_mirror \
  python examples/toroidal_stellarator_mirror_hybrid.py \
  --outdir results/toroidal_stellarator_mirror_hybrid_m171_label_smoke \
  --ntheta-fit 32 --nzeta-fit 32 --ntor 6
JAX_ENABLE_X64=1 pytest tests/test_toroidal_hybrid.py -q
```

Results:

- Ruff format/check passed.
- Sphinx docs build passed with warnings treated as errors.
- Straight-axis hybrid example test passed: `1 passed in 1.98s`.
- Toroidal hybrid example test passed: `1 passed in 1.07s`.
- Plotted straight-axis and toroidal hybrid smokes completed.
- Full toroidal hybrid tests passed: `26 passed in 3.72s`.

### File structure and best-practice notes

- The straight-axis support fixture remains in the mirror examples because it
  exercises open-ended mirror plotting and 3D boundary handling.
- The toroidal target remains in the root toroidal hybrid example and
  `vmec_jax.toroidal_hybrid`, reusing ordinary VMEC/JAX toroidal boundary and
  solver paths.
- The new scope labels are data fields, not prose-only documentation, so
  downstream scripts can distinguish support fixtures from final target lanes.
- Generated smoke outputs remain under ignored `results/` paths.

### Best next steps

1. Commit and push M171.
2. Refresh the PR body if needed to mention that the straight-axis hybrid lane
   is closed as a support fixture.
3. Continue final plan closure with the final free-boundary nonlinear solve
   status and remaining known limitations.
4. Re-check CI after enough time has passed for the latest workflow to expose
   actionable failures.

### Completion percentages after M171

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `90%`.
- Finite-current pitch validation: `86%`.
- Plotting and `vmec --plot` mirror support: `97%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `94%`.
- Mirror-Boozer-like diagnostics: `83%`.
- Free-boundary mirror lane: `99%`.
- Straight-axis hybrid support fixture lane: `100%` for support-fixture scope.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- PR merge readiness overall: `99%`.

### User input needed

No user input is needed.

---
## 172. Reduced Free-Boundary Residual-Vector Nonlinear Solve Loop

### Steps taken

- Added a reusable reduced residual-vector nonlinear least-squares solve loop:
  `mirror_free_boundary_residual_vector_least_squares_solve`.
- Added result dataclasses:
  - `MirrorFreeBoundaryVectorLeastSquaresSolveRow`;
  - `MirrorFreeBoundaryVectorLeastSquaresSolveResult`.
- The loop reuses `mirror_free_boundary_residual_vector_least_squares_step` and
  records each accepted or rejected step.
- The loop now has explicit stop reasons:
  - `target_residual`;
  - `ls_step_not_accepted`;
  - `stagnation`;
  - `max_steps`.
- Exported the new solve-loop API through `vmec_jax.mirror.api` and
  `vmec_jax.mirror`.
- Extended `examples/mirror_free_boundary_vector_ls_benchmark.py`:
  - schema version `0.2`;
  - per-backend `solve_rows`;
  - solve convergence validation;
  - new plotted solve residual-history figure.
- Updated docs in `docs/mirror/overview.rst` and `examples/mirror/README.md`.
- Added focused tests for convergence, rejected steps, stagnation, invalid
  inputs, public API export, and benchmark schema `0.2`.

### Results obtained

- Reduced residual-vector prototypes can now run an actual nonlinear LS solve
  loop instead of only a one-step diagnostic.
- Plotted benchmark run:
  `results/mirror/free_boundary_vector_ls_benchmark_m172_solve_loop_plots/mirror_free_boundary_vector_ls_benchmark_metrics.json`.
- Benchmark schema `0.2` rows showed all derivative backends converged:
  - finite difference: `target_residual`, `2` accepted steps,
    final residual `1.415619462786164e-12`;
  - JAX forward: `target_residual`, `2` accepted steps,
    final residual `3.003608781755996e-17`;
  - JAX reverse: `target_residual`, `2` accepted steps,
    final residual `3.003608781755996e-17`;
  - JAX auto: `target_residual`, `2` accepted steps,
    final residual `3.003608781755996e-17`.
- New plot visually inspected:
  `results/mirror/free_boundary_vector_ls_benchmark_m172_solve_loop_plots/figures/mirror_free_boundary_vector_ls_solve_residual_history.png`.

### How it was tested

Commands run:

```bash
python -m ruff format vmec_jax/mirror/free_boundary.py \
  vmec_jax/mirror/api.py \
  vmec_jax/mirror/__init__.py \
  examples/mirror_free_boundary_vector_ls_benchmark.py \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py
python -m ruff check vmec_jax/mirror/free_boundary.py \
  vmec_jax/mirror/api.py \
  vmec_jax/mirror/__init__.py \
  examples/mirror_free_boundary_vector_ls_benchmark.py \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_free_boundary.py -q
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_vector_ls_benchmark_runs_without_plots \
  -q
python -m sphinx -W -b html docs docs/_build/html
JAX_ENABLE_X64=1 python examples/mirror_free_boundary_vector_ls_benchmark.py \
  --outdir results/mirror/free_boundary_vector_ls_benchmark_m172_solve_loop_plots \
  --nxi 17
python - <<'PY'
import vmec_jax.mirror as mirror
import vmec_jax.mirror.api as api
for module in (mirror, api):
    missing = [name for name in module.__all__ if not hasattr(module, name)]
    if missing:
        raise SystemExit(f"{module.__name__} missing {missing}")
for name in [
    "MirrorFreeBoundaryVectorLeastSquaresSolveResult",
    "MirrorFreeBoundaryVectorLeastSquaresSolveRow",
    "mirror_free_boundary_residual_vector_least_squares_solve",
]:
    assert getattr(mirror, name) is getattr(api, name)
print("mirror_public_api_ok")
PY
git diff --check
```

Results:

- Ruff format/check passed.
- Full mirror free-boundary test module passed: `116 passed in 3.61s`.
- Focused benchmark example test passed: `1 passed in 2.17s`.
- Sphinx docs build passed with warnings treated as errors.
- Plotted benchmark completed and wrote the solve residual-history figure.
- Public API consistency check printed `mirror_public_api_ok`.
- Whitespace check passed.

### File structure and best-practice notes

- The solve loop lives in `vmec_jax/mirror/free_boundary.py` beside the
  existing one-step reduced vector LS helper.
- The root benchmark example remains the user-facing derivative-backend and
  solve-loop comparison artifact.
- The loop is deliberately reduced-residual-vector scoped. Host-side realized
  fixed-boundary circular-coil trials still use the guarded callback loop until
  the full coupled free-boundary residual becomes a promoted differentiable
  solver path.
- Generated plots/metrics remain under ignored `results/` paths.

### Best next steps

1. Commit and push M172.
2. Refresh the draft PR body after this solve-loop promotion.
3. Re-check CI after the latest head has jobs attached.
4. Continue final known-limitations audit and decide whether the PR can move
   from draft after one final full local/CI pass.

### Completion percentages after M172

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `90%`.
- Finite-current pitch validation: `86%`.
- Plotting and `vmec --plot` mirror support: `97%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `95%`.
- Mirror-Boozer-like diagnostics: `83%`.
- Free-boundary mirror lane: `99%` overall, with reduced residual-vector
  nonlinear solve scope complete.
- Straight-axis hybrid support fixture lane: `100%` for support-fixture scope.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- PR merge readiness overall: `99%`.

### User input needed

No user input is needed.

---
## 173. Draft PR Body Synchronization After Reduced Solve Loop

### Steps taken

- Updated draft PR #21 body after M172.
- Changed the implementation-log pointer from section 169 to section 172.
- Added PR-body wording for the reduced residual-vector nonlinear solve loop
  and the benchmark solve residual-history plot.
- Added a checked checklist item for the reduced residual-vector nonlinear LS
  solve loop.
- Added latest validation evidence:
  - `116 passed in 3.61s` for the mirror free-boundary test module;
  - M172 plotted solve-loop evidence under
    `results/mirror/free_boundary_vector_ls_benchmark_m172_solve_loop_plots/`.
- Verified the PR body markers.
- Sampled GitHub checks once after jobs attached to the latest head.

### Results obtained

- PR body verification confirmed:
  - PR number `21`;
  - draft state `true`;
  - head SHA `d664e0e0557066eb5e9e338943f94f66dd9ffd9a`;
  - body contains `section 172`;
  - body contains the reduced solve-loop checklist item;
  - body contains `116 passed in 3.61s`;
  - body contains the M172 solve-loop plot path.
- GitHub checks are attached and running. The sampled status showed:
  - `Parity Manifest Smoke (dry-run)` passed;
  - `Physics Full (manual/nightly)` skipped as configured;
  - standard docs/build/fast/physics-smoke jobs in progress or queued.

### How it was tested

Commands run:

```bash
gh pr view 21 --repo uwplasma/vmec_jax --json body -q .body > /tmp/vmec_mirror_pr_body_current.md
python - <<'PY'
# update PR body text in /tmp
PY
gh pr edit 21 --repo uwplasma/vmec_jax --body-file /tmp/vmec_mirror_pr_body_current.md
gh pr view 21 --repo uwplasma/vmec_jax --json number,isDraft,headRefOid,mergeStateStatus,body,url > /tmp/vmec_mirror_pr_view.json
python - <<'PY'
# verify required PR body markers
PY
gh pr checks 21 --repo uwplasma/vmec_jax --json name,state,bucket,startedAt,completedAt,link
git status -sb
```

Results:

- PR edit succeeded.
- PR body verification passed.
- PR remained draft.
- Worktree was clean before this plan-only M173 log.

### File structure and best-practice notes

- No source files changed in this administrative tranche.
- The PR body remains a review index; detailed evidence stays in
  `plan_mirror.md`.
- CI was sampled once and not continuously polled.

### Best next steps

1. Commit and push M173 plan log.
2. Continue final known-limitations audit and remaining PR-readiness cleanup.
3. Re-check CI after enough time has elapsed for failures to be actionable.

### Completion percentages after M173

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `90%`.
- Finite-current pitch validation: `86%`.
- Plotting and `vmec --plot` mirror support: `97%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `95%`.
- Mirror-Boozer-like diagnostics: `83%`.
- Free-boundary mirror lane: `99%` overall, with reduced residual-vector
  nonlinear solve scope complete.
- Straight-axis hybrid support fixture lane: `100%` for support-fixture scope.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- PR merge readiness overall: `99%`.

### User input needed

No user input is needed.

---
## 174. Mirror Readiness Matrix Documentation

### Steps taken

- Repaired the committed plan ordering before this tranche so numbered
  sections now run monotonically from `0` through `173`.
- Committed and pushed the repaired plan-sync log as
  `3733c43d Sync mirror PR plan after vector solve loop`.
- Sampled PR #21 after the push:
  - draft state remained `true`;
  - latest head was `3733c43d0a9f649420b911761fbc5d45fcb09810`;
  - no failing checks were detected;
  - the latest run had parity smoke passed, full physics skipped as configured,
    and the standard jobs still running.
- Added `docs/mirror/readiness.rst` as a single review-facing scope matrix.
- Linked the new page from `docs/mirror/index.rst`.

### Results obtained

- The mirror docs now distinguish:
  - supported fixed-boundary/grid/output/plotting/two-coil paths;
  - validated prototypes for theta-dependent surfaces, residual Newton,
    reduced differentiable APIs, and toroidal hybrid fixtures;
  - diagnostic free-boundary, ESSOS beta-scan, finite-current pitch, and
    support-fixture paths;
  - deferred anisotropic/kinetic/open-end physics.
- The readiness page states the derivative-backend policy explicitly:
  finite differences for host-side CLI loops, JAX forward/reverse for pure JAX
  residual-vector prototypes according to parameter/residual shape, and dense
  solves as tiny-grid correctness references.
- The page also records the final undrafting gate: full mirror tests, warning-
  clean docs, current PR body, lightweight artifacts, and no failing GitHub
  checks at the latest pushed head.

### How it was tested

Commands run:

```bash
python -m sphinx -W -b html docs docs/_build/html
git diff --check
gh auth status
gh pr view 21 --repo uwplasma/vmec_jax --json number,isDraft,headRefOid,mergeStateStatus,url
python /Users/rogeriojorge/.codex/plugins/cache/openai-curated-remote/github/0.1.5/skills/gh-fix-ci/scripts/inspect_pr_checks.py --repo . --pr 21 --json
gh pr checks 21 --repo uwplasma/vmec_jax --json name,state,bucket,startedAt,completedAt,link
```

Results:

- Sphinx completed successfully with warnings treated as errors.
- Whitespace check passed.
- GitHub CLI authentication was valid with `repo` and `workflow` scopes.
- PR #21 remained draft and had no detected failing checks at the sampled head.

### File structure and best-practice notes

- The scope matrix is in `docs/mirror/readiness.rst`, alongside the existing
  mirror overview, output, and differentiability pages.
- `docs/mirror/index.rst` owns the toctree link, so the page is included in
  full docs without adding another root-level document.
- No generated output files or figures were added to the repository.

### Best next steps

1. Commit and push M174.
2. Re-check PR #21 after CI has had enough time to finish and inspect only
   failed jobs.
3. Update the draft PR body to point reviewers at the readiness matrix and the
   latest plan section if the branch remains otherwise clean.
4. Run a final full `tests/mirror` pass before deciding whether any lane is
   mature enough to undraft.

### Completion percentages after M174

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `90%`.
- Finite-current pitch validation: `86%`.
- Plotting and `vmec --plot` mirror support: `97%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `95%`.
- Mirror-Boozer-like diagnostics: `83%`.
- Free-boundary mirror lane: `99%` overall, with reduced residual-vector
  nonlinear solve scope complete.
- Straight-axis hybrid support fixture lane: `100%` for support-fixture scope.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- PR merge readiness overall: `99%`.

### User input needed

No user input is needed.

---
## 175. Draft PR Body Synchronization After Readiness Matrix

### Steps taken

- Updated draft PR #21 after M174.
- Changed the implementation-log pointer from section 172 to section 174.
- Added the new readiness matrix to the PR summary, checklist, validation
  notes, and reviewer notes.
- Verified PR body markers after editing.
- Sampled GitHub checks once after the latest pushed head.

### Results obtained

- PR body verification confirmed:
  - PR number `21`;
  - draft state `true`;
  - head SHA `bd9fe5d869c71f33b1f969d1a41da3409ede0a2c`;
  - body contains `section 174`;
  - body contains `docs/mirror/readiness.rst`;
  - body contains the readiness-matrix checklist item;
  - body contains the Sphinx readiness validation note.
- CI snapshot found no failing checks.
- The sampled check list showed:
  - `Parity Manifest Smoke (dry-run)` passed;
  - `Physics Full (manual/nightly)` skipped as configured;
  - standard docs/build/fast/physics-smoke jobs in progress.

### How it was tested

Commands run:

```bash
gh pr view 21 --repo uwplasma/vmec_jax --json body,number,isDraft,headRefOid,url
python - <<'PY'
# update PR body text in /tmp/vmec_mirror_pr_body_m174.md
PY
gh pr edit 21 --repo uwplasma/vmec_jax --body-file /tmp/vmec_mirror_pr_body_m174.md
python - <<'PY'
# verify required PR body markers
PY
python /Users/rogeriojorge/.codex/plugins/cache/openai-curated-remote/github/0.1.5/skills/gh-fix-ci/scripts/inspect_pr_checks.py --repo . --pr 21 --json
gh pr checks 21 --repo uwplasma/vmec_jax --json name,state,bucket,startedAt,completedAt,link
git status -sb
```

Results:

- PR edit succeeded.
- PR body verification printed `pr_body_ok`.
- CI failure inspection printed `PR #21: no failing checks detected.`
- Worktree was clean before this plan-only log.

### File structure and best-practice notes

- No source or docs files changed in this administrative tranche.
- The PR body now points reviewers to the readiness matrix instead of forcing
  them to reconstruct current support status from the long plan log.
- CI was sampled once and not continuously polled.

### Best next steps

1. Commit and push M175.
2. Run the final full local `tests/mirror` pass.
3. Re-check PR #21 only after the current CI batch has had time to finish.
4. If local tests and CI are clean, perform the final requirement-by-
   requirement completion audit before deciding whether any lane can be
   declared complete enough to undraft.

### Completion percentages after M175

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `90%`.
- Finite-current pitch validation: `86%`.
- Plotting and `vmec --plot` mirror support: `97%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `95%`.
- Mirror-Boozer-like diagnostics: `83%`.
- Free-boundary mirror lane: `99%` overall, with reduced residual-vector
  nonlinear solve scope complete.
- Straight-axis hybrid support fixture lane: `100%` for support-fixture scope.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- PR merge readiness overall: `99%`.

### User input needed

No user input is needed.

---
## 176. Final Local Mirror Test Pass After Readiness Sync

### Steps taken

- Ran the full local mirror test suite after the readiness matrix and PR-body
  synchronization commits.
- Re-checked the worktree after the test run.
- Sampled PR #21 checks once after the latest pushed head.

### Results obtained

- The full local mirror suite passed:
  - `226 passed, 1 skipped in 191.51s`.
- The worktree was clean after the test run.
- CI failure inspection found no failing checks at the sampled PR head.
- The sampled check list showed:
  - `Parity Manifest Smoke (dry-run)` passed;
  - `Console Script Smoke` passed;
  - `Fast Tests (py3.12)` passed;
  - `Physics Full (manual/nightly)` skipped as configured;
  - standard docs/build/remaining fast/physics-smoke jobs still in progress or
    queued.

### How it was tested

Commands run:

```bash
JAX_ENABLE_X64=1 pytest tests/mirror -q
git status -sb
python /Users/rogeriojorge/.codex/plugins/cache/openai-curated-remote/github/0.1.5/skills/gh-fix-ci/scripts/inspect_pr_checks.py --repo . --pr 21 --json
gh pr checks 21 --repo uwplasma/vmec_jax --json name,state,bucket,startedAt,completedAt,link
```

Results:

- Pytest completed successfully with `226 passed, 1 skipped in 191.51s`.
- The worktree was clean.
- CI failure inspection printed `PR #21: no failing checks detected.`

### File structure and best-practice notes

- No files changed during the test run before this plan-only log.
- The test result is recorded in the plan rather than adding generated output
  artifacts to the repository.
- CI was sampled once and not continuously polled.

### Best next steps

1. Commit and push M176.
2. Update the draft PR body validation note to include the `226 passed,
   1 skipped` mirror-suite result.
3. Perform a requirement-by-requirement completion audit against the plan,
   source, docs, examples, PR body, and CI state.
4. Fix any audit or CI failures that are concrete and actionable.

### Completion percentages after M176

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `90%`.
- Finite-current pitch validation: `86%`.
- Plotting and `vmec --plot` mirror support: `97%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `95%`.
- Mirror-Boozer-like diagnostics: `83%`.
- Free-boundary mirror lane: `99%` overall, with reduced residual-vector
  nonlinear solve scope complete.
- Straight-axis hybrid support fixture lane: `100%` for support-fixture scope.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- PR merge readiness overall: `99%`.

### User input needed

No user input is needed.

---
## 177. Finite-Current Coverage for Mirror-Boozer-Like Diagnostics

### Steps taken

- Added a focused finite-current plotting-test fixture in
  `tests/mirror/test_mirror_plotting.py`.
- Added a direct test that mirror-Boozer-like diagnostics capture nonzero
  cap-to-cap field-line turns, nonzero contravariant pitch, and nonzero pitch
  variation when `I'` is finite.
- Kept this as test coverage only; no production source behavior changed.

### Results obtained

- The Boozer-like diagnostics lane is now covered by both:
  - root finite-current example smoke coverage with JSON summary metrics;
  - direct plotting/diagnostic unit-style coverage for nonzero pitch proxies.
- The new test checks that summary extrema match the underlying diagnostic
  arrays, reducing the chance that exported scalar metrics drift from plotted
  data.

### How it was tested

Commands run:

```bash
python -m ruff format tests/mirror/test_mirror_plotting.py
python -m ruff check tests/mirror/test_mirror_plotting.py
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_plotting.py -q
JAX_ENABLE_X64=1 pytest tests/mirror -q
git diff --check
```

Results:

- Ruff format left the file unchanged.
- Ruff check passed.
- Focused plotting tests passed: `5 passed in 5.16s`.
- Full mirror suite passed on the updated head: `227 passed, 1 skipped in
  191.50s`.
- Whitespace check passed.

### File structure and best-practice notes

- The new helper stays local to `tests/mirror/test_mirror_plotting.py` because
  it is a tiny fixture used only by plotting/diagnostic tests.
- The test reuses the public fixed-boundary run and MOUT write/read path, so it
  verifies the same data shape that examples and `vmec --plot` consume.
- No generated result files or figures were added to the repository.

### Best next steps

1. Commit and push M177.
2. Update the draft PR body validation note to include the `227 passed,
   1 skipped` full mirror-suite result.
3. Re-check CI after the latest push has had time to finish, inspecting only
   failed jobs.
4. Continue the requirement-by-requirement audit and close the next concrete
   low-completion lane.

### Completion percentages after M177

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `90%`.
- Finite-current pitch validation: `87%`.
- Plotting and `vmec --plot` mirror support: `97%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `95%`.
- Mirror-Boozer-like diagnostics: `88%`.
- Free-boundary mirror lane: `99%` overall, with reduced residual-vector
  nonlinear solve scope complete.
- Straight-axis hybrid support fixture lane: `100%` for support-fixture scope.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- PR merge readiness overall: `99%`.

### User input needed

No user input is needed.

---
## 178. Two-Coil Flux-Tube Helper Equivalence Coverage

### Steps taken

- Added a focused test for `mirror_boundary_from_two_coil_flux_tube` in
  `tests/mirror/test_two_coil_axisym_benchmark.py`.
- Verified that the convenience helper produces the same boundary radii as the
  explicit analytic two-coil on-axis field plus
  `mirror_boundary_from_on_axis_bz` construction.
- Kept this as test coverage only; no source behavior changed.

### Results obtained

- The two-coil benchmark lane now directly protects the public convenience
  constructor used by examples and downstream scripts.
- Existing on-axis, off-axis Biot-Savart, and fixed-boundary axis-field tests
  remain unchanged; this adds the missing wrapper-equivalence gate.

### How it was tested

Commands run:

```bash
python -m ruff format tests/mirror/test_two_coil_axisym_benchmark.py
python -m ruff check tests/mirror/test_two_coil_axisym_benchmark.py
JAX_ENABLE_X64=1 pytest tests/mirror/test_two_coil_axisym_benchmark.py -q
JAX_ENABLE_X64=1 pytest tests/mirror -q
git diff --check
```

Results:

- Ruff format left the file unchanged.
- Ruff check passed.
- Focused two-coil benchmark tests passed: `4 passed in 1.10s`.
- Full mirror suite passed on the updated head: `228 passed, 1 skipped in
  195.09s`.
- Whitespace check passed.

### File structure and best-practice notes

- The new test lives in the existing two-coil benchmark module, beside the
  analytic circular-loop, off-axis Biot-Savart, and fixed-boundary checks.
- The test compares public helper output through the normal `MirrorBoundary`
  grid-evaluation API instead of inspecting internal boundary fields.
- No generated result files or figures were added to the repository.

### Best next steps

1. Commit and push M178.
2. Update the draft PR body validation note to include the `228 passed,
   1 skipped` full mirror-suite result.
3. Re-check CI after the latest push has had time to finish, inspecting only
   failed jobs.
4. Continue the completion audit with the fixed-boundary solver and
   residual-Newton/preconditioning lanes.

### Completion percentages after M178

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `91%`.
- Residual Newton / preconditioning: `92%`.
- Two-coil and manufactured validation: `92%`.
- Finite-current pitch validation: `87%`.
- Plotting and `vmec --plot` mirror support: `97%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `95%`.
- Mirror-Boozer-like diagnostics: `88%`.
- Free-boundary mirror lane: `99%` overall, with reduced residual-vector
  nonlinear solve scope complete.
- Straight-axis hybrid support fixture lane: `100%` for support-fixture scope.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- PR merge readiness overall: `99%`.

### User input needed

No user input is needed.

---
## 179. Residual-Newton Diagnostic Example Coverage

### Steps taken

- Added a focused smoke test for the root
  `examples/mirror_fixed_boundary_solve_diagnostic.py` residual-Newton path.
- The test runs the diagnostic example with matrix-free `lsmr`, fixed inner
  iteration budget, the VMEC-like `radial_xi_tridi` preconditioner, and dense
  step comparison enabled.
- Verified that the emitted JSON records the expected Krylov/preconditioner
  fields, dense-step comparison fields, accepted optimizer state, final `fsq`,
  and a written MOUT file.

### Results obtained

- The fixed-boundary diagnostic example now has direct test coverage for both
  default L-BFGS and residual-Newton/Krylov modes.
- The residual-Newton/preconditioning lane has a stronger review-facing example
  contract: metadata used in convergence plots and PR diagnostics is checked by
  the test suite, not only by manual plotted runs.

### How it was tested

Commands run:

```bash
python -m ruff format tests/mirror/test_mirror_examples.py
python -m ruff check tests/mirror/test_mirror_examples.py
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_examples.py::test_root_fixed_boundary_solve_diagnostic_residual_newton_reports_krylov_fields -q
JAX_ENABLE_X64=1 pytest tests/mirror -q
git diff --check
```

Results:

- Ruff format left the file unchanged.
- Ruff check passed.
- Focused residual-Newton diagnostic example test passed: `1 passed in 5.27s`.
- Full mirror suite passed on the updated head: `229 passed, 1 skipped in
  202.80s`.
- Whitespace check passed.

### File structure and best-practice notes

- The test stays in `tests/mirror/test_mirror_examples.py`, beside the existing
  root-example smoke tests.
- It uses a tiny `ns=5`, `nxi=7`, `maxiter=1` diagnostic run so coverage is
  meaningful without adding heavy runtime or repository artifacts.
- No generated output files or figures were added to the repository.

### Best next steps

1. Commit and push M179.
2. Update the draft PR body validation note to include the `229 passed,
   1 skipped` full mirror-suite result and residual-Newton diagnostic coverage.
3. Re-check CI after the latest push has had time to finish, inspecting only
   failed jobs.
4. Continue the requirement-by-requirement completion audit, focusing next on
   source/API simplification and any remaining finite-current validation gaps.

### Completion percentages after M179

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `93%`.
- Residual Newton / preconditioning: `94%`.
- Two-coil and manufactured validation: `92%`.
- Finite-current pitch validation: `87%`.
- Plotting and `vmec --plot` mirror support: `97%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `95%`.
- Mirror-Boozer-like diagnostics: `88%`.
- Free-boundary mirror lane: `99%` overall, with reduced residual-vector
  nonlinear solve scope complete.
- Straight-axis hybrid support fixture lane: `100%` for support-fixture scope.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- PR merge readiness overall: `99%`.

### User input needed

No user input is needed.

---
## 180. Mirror Public Namespace Single-Source Simplification

### Steps taken

- Simplified `vmec_jax/mirror/__init__.py` so it re-exports
  `vmec_jax.mirror.api` directly and imports `api.__all__` as the package
  `__all__`.
- Removed the duplicated 100-item public export list from the package
  initializer.
- Added a focused test that `vmec_jax.mirror.__all__` is exactly
  `vmec_jax.mirror.api.__all__` and that every exported object is the same
  object in both namespaces.

### Results obtained

- The mirror public namespace now has a single source of truth for exported
  symbols.
- Future public API additions only need to update `api.py`, reducing file
  churn and drift risk in `__init__.py`.
- The initializer is now four lines instead of a long duplicated import and
  `__all__` block.

### How it was tested

Commands run:

```bash
python -m ruff format vmec_jax/mirror/__init__.py tests/mirror/test_mirror_low_level_coverage.py
python -m ruff check vmec_jax/mirror/__init__.py tests/mirror/test_mirror_low_level_coverage.py
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_low_level_coverage.py -q
JAX_ENABLE_X64=1 pytest tests/mirror -q
git diff --check
```

Results:

- Ruff format left both files unchanged.
- Ruff check passed.
- Focused low-level mirror coverage passed: `7 passed in 2.32s`.
- Full mirror suite passed on the updated head: `230 passed, 1 skipped in
  203.66s`.
- Whitespace check passed.

### File structure and best-practice notes

- This is a source simplification, not a public API change: all names still
  come from `vmec_jax.mirror.api` and are re-exported at `vmec_jax.mirror`.
- The new test is in `tests/mirror/test_mirror_low_level_coverage.py`, where
  other low-level guard and key-normalization tests already live.
- No generated output files or figures were added to the repository.

### Best next steps

1. Commit and push M180.
2. Update the draft PR body validation note to include the `230 passed,
   1 skipped` full mirror-suite result and namespace simplification.
3. Re-check CI after the latest push has had time to finish, inspecting only
   failed jobs.
4. Continue the completion audit with finite-current validation and final PR
   readiness checks.

### Completion percentages after M180

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `93%`.
- Residual Newton / preconditioning: `94%`.
- Two-coil and manufactured validation: `92%`.
- Finite-current pitch validation: `87%`.
- Plotting and `vmec --plot` mirror support: `97%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `95%`.
- Mirror-Boozer-like diagnostics: `88%`.
- Free-boundary mirror lane: `99%` overall, with reduced residual-vector
  nonlinear solve scope complete.
- Straight-axis hybrid support fixture lane: `100%` for support-fixture scope.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- Public API/source simplification: `100%` for the mirror package initializer.
- PR merge readiness overall: `99%`.

### User input needed

No user input is needed.

---
## 181. Finite-Current Field-Line Plot Coverage

### Steps taken

- Added a plotted smoke test for `examples/mirror_finite_current_pitch.py`.
- The test runs a low-resolution finite-current case with plots enabled and
  checks the custom theta-advance plot, geometry/coils/field-line plot, and
  mirror-Boozer-like diagnostic plot.
- The test verifies each PNG exists, has nontrivial file size, can be read by
  Matplotlib, and is nonblank.

### Results obtained

- The finite-current pitch lane now has direct test coverage for both no-plot
  numerical metrics and plotted field-line artifacts.
- The field-line visibility requirement is protected by a cheap automated test
  instead of relying only on manual plot inspection.

### How it was tested

Commands run:

```bash
python -m ruff format tests/mirror/test_mirror_examples.py
python -m ruff check tests/mirror/test_mirror_examples.py
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_examples.py::test_root_finite_current_pitch_example_writes_nonblank_field_line_plots -q
JAX_ENABLE_X64=1 pytest tests/mirror -q
git diff --check
```

Results:

- Ruff format left the file unchanged.
- Ruff check passed.
- Focused finite-current plot smoke test passed: `1 passed in 3.46s`.
- Full mirror suite passed on the updated head: `231 passed, 1 skipped in
  205.23s`.
- Whitespace check passed.

### File structure and best-practice notes

- The new test stays in `tests/mirror/test_mirror_examples.py` with the other
  root-example smoke tests.
- It uses the existing ignored/temp output flow and does not add any generated
  figures to the repository.
- The test asserts image content generically rather than encoding brittle pixel
  snapshots.

### Best next steps

1. Commit and push M181.
2. Update the draft PR body validation note to include the `231 passed,
   1 skipped` full mirror-suite result and finite-current plot coverage.
3. Re-check CI after the latest push has had time to finish, inspecting only
   failed jobs.
4. Continue final completion audit and decide whether the remaining non-100%
   lanes are explicit future work or require another closure tranche.

### Completion percentages after M181

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `93%`.
- Residual Newton / preconditioning: `94%`.
- Two-coil and manufactured validation: `92%`.
- Finite-current pitch validation: `92%`.
- Plotting and `vmec --plot` mirror support: `98%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `95%`.
- Mirror-Boozer-like diagnostics: `89%`.
- Free-boundary mirror lane: `99%` overall, with reduced residual-vector
  nonlinear solve scope complete.
- Straight-axis hybrid support fixture lane: `100%` for support-fixture scope.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- Public API/source simplification: `100%` for the mirror package initializer.
- PR merge readiness overall: `99%`.

### User input needed

No user input is needed.


---
## 182. Matrix-Free Implicit-Parameter Gradient Example Coverage

### Steps taken

- Audited the differentiable fixed-boundary API, the differentiability docs,
  and the root implicit-gradient examples.
- Confirmed existing unit tests already compare profile and polynomial-boundary
  custom VJPs against explicit adjoints, forward sensitivities, matrix-free
  sensitivity solves, and independently solved finite-difference perturbed
  roots.
- Added a root-example test that runs
  `examples/mirror_implicit_parameter_gradients.py` through the
  `matrix_free_cg` path for pressure-profile and polynomial-boundary
  parameters with plots enabled.
- The new test verifies accepted metrics, finite-difference agreement, forward
  sensitivity agreement, and that the plotted directional-gradient PNG is
  readable and nonblank.
- Updated `docs/mirror/differentiability.rst` and `examples/mirror/README.md`
  so the documented validation surface matches the tested dense and
  matrix-free custom-VJP paths.

### Results obtained

- The documented implicit-parameter example now has direct automated coverage
  for the matrix-free pressure/boundary custom-VJP path.
- The differentiability example's plotted validation artifact is protected by a
  content check rather than only by manual inspection.
- The differentiable fixed-boundary lane remains a validated prototype, but the
  pressure/profile/boundary method gates now have stronger end-to-end example
  coverage.

### How it was tested

Commands run:

```bash
JAX_ENABLE_X64=1 python examples/mirror_implicit_parameter_gradients.py \
  --outdir /tmp/mirror_implicit_parameter_gradients_matrix_free_probe \
  --solve-method matrix_free_cg \
  --families pressure,boundary \
  --no-plots
JAX_ENABLE_X64=1 python examples/mirror_implicit_parameter_gradients.py \
  --outdir /tmp/mirror_implicit_parameter_gradients_plot_probe \
  --families pressure,boundary
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_examples.py::test_root_implicit_parameter_gradients_example_matrix_free_writes_plot -q
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_examples.py::test_root_implicit_parameter_gradients_example_runs_without_plots \
  tests/mirror/test_mirror_examples.py::test_root_implicit_parameter_gradients_example_matrix_free_writes_plot \
  tests/mirror/test_mirror_examples.py::test_root_implicit_solve_benchmark_runs_without_plots -q
python -m ruff check tests/mirror/test_mirror_examples.py
python -m ruff format --check tests/mirror/test_mirror_examples.py
python -m sphinx -W -b html docs docs/_build/html
JAX_ENABLE_X64=1 pytest tests/mirror -q
```

Results:

- Matrix-free pressure/boundary probe accepted both rows.
- Plot probe wrote a nonblank `mirror_implicit_parameter_gradients.png`.
- Focused matrix-free plotted example test passed: `1 passed in 14.75s`.
- Adjacent implicit-example tests passed: `3 passed in 37.98s`.
- Ruff check and format check passed for the touched Python test file.
- Sphinx docs build passed with warnings as errors.
- Full mirror suite passed: `232 passed, 1 skipped in 218.85s`.

### File structure and best-practice notes

- The new coverage stays in `tests/mirror/test_mirror_examples.py` beside the
  other root-example smoke tests.
- Documentation updates are limited to the mirror differentiability page and
  mirror example README, keeping the validation contract near the public entry
  points users will run.
- Generated figures remain under temporary or ignored output directories; no
  binary artifacts were added to the repository.
- The matrix-free test uses a small pressure/boundary family subset to keep CI
  runtime controlled while covering both profile and geometry-parameter VJPs.

### Best next steps

1. Commit and push M182.
2. Update the draft PR body to include the `232 passed, 1 skipped` full
   mirror-suite result and matrix-free implicit-gradient coverage.
3. Check the latest PR head for failing CI jobs after the push, without
   spending time waiting on queued jobs.
4. Continue the completion audit with the next remaining non-100% lane,
   likely toroidal hybrid convergence readiness or the final free-boundary
   documentation boundary.

### Completion percentages after M182

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `93%`.
- Residual Newton / preconditioning: `94%`.
- Two-coil and manufactured validation: `92%`.
- Finite-current pitch validation: `92%`.
- Plotting and `vmec --plot` mirror support: `98%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `96%`.
- Mirror-Boozer-like diagnostics: `89%`.
- Free-boundary mirror lane: `99%` overall, with reduced residual-vector
  nonlinear solve scope complete.
- Straight-axis hybrid support fixture lane: `100%` for support-fixture scope.
- Toroidal stellarator-mirror hybrid lane: `95%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- Public API/source simplification: `100%` for the mirror package initializer.
- PR merge readiness overall: `99%`.

### User input needed

No user input is needed.


---
## 183. Toroidal Hybrid Standalone Plot Coverage

### Steps taken

- Audited the toroidal stellarator-mirror hybrid lane after M182.
- Found that the root-level toroidal hybrid scripts failed when run exactly as
  README commands unless `PYTHONPATH` was pre-set:
  - `examples/toroidal_stellarator_mirror_hybrid.py`;
  - `examples/toroidal_stellarator_mirror_hybrid_convergence.py`.
- Added the same repository-root `sys.path` bootstrap used by the other
  root-level mirror examples so the scripts work as standalone commands from
  the repository root.
- Removed the test-only `PYTHONPATH` shim from toroidal hybrid example tests so
  CI covers the real user command path.
- Added plotted smoke tests for:
  - the toroidal hybrid boundary example's LCFS, top-view, cross-section, and
    side/corner orientation plots;
  - the no-solve toroidal hybrid convergence example's convergence and
    orientation-preservation plots.
- The new tests verify that each PNG exists, is readable by Matplotlib, and is
  nonblank.

### Results obtained

- The README-style toroidal hybrid commands now run without a manually prepared
  `PYTHONPATH`.
- The toroidal hybrid plotting lane now has end-to-end coverage for both the
  boundary example and the no-solve convergence report.
- The no-solve convergence report remains lightweight, while still protecting
  the orientation-preservation artifact needed before higher-budget solved
  convergence studies.

### How it was tested

Commands run:

```bash
python examples/toroidal_stellarator_mirror_hybrid.py \
  --outdir /tmp/toroidal_hybrid_plot_probe \
  --ntheta-fit 64 --nzeta-fit 64 --ntor 10 \
  --side-minor-modulation 0.16 --side-elongation 0.35 --side-power 2.0 \
  --corner-amplitude 0.025 --corner-ellipticity 0.22 \
  --corner-rotation 0.42 --corner-power 2.0
python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir /tmp/toroidal_hybrid_convergence_plot_probe \
  --ns-array 7,9 --mode-pairs 5:20 \
  --ntheta-fit 64 --nzeta-fit 64 \
  --side-power 2.0 --corner-power 2.0
python -m ruff check \
  examples/toroidal_stellarator_mirror_hybrid.py \
  examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  tests/test_toroidal_hybrid.py
python -m ruff format --check \
  examples/toroidal_stellarator_mirror_hybrid.py \
  examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  tests/test_toroidal_hybrid.py
pytest \
  tests/test_toroidal_hybrid.py::test_toroidal_hybrid_example_runs_without_plots \
  tests/test_toroidal_hybrid.py::test_toroidal_hybrid_example_writes_nonblank_plots \
  tests/test_toroidal_hybrid.py::test_toroidal_hybrid_convergence_example_runs_without_solve \
  tests/test_toroidal_hybrid.py::test_toroidal_hybrid_convergence_example_writes_nonblank_no_solve_plots \
  tests/test_toroidal_hybrid.py::test_toroidal_hybrid_convergence_example_scans_shape_cases_without_solve -q
pytest tests/test_toroidal_hybrid.py -q
JAX_ENABLE_X64=1 pytest tests/mirror tests/test_toroidal_hybrid.py -q
```

Results:

- Standalone plotted toroidal hybrid probe wrote nonblank `lcfs_3d`,
  `top_view`, `cross_sections`, and `region_orientation` PNGs.
- Standalone plotted convergence probe wrote nonblank `convergence` and
  `orientation` PNGs.
- Ruff check and format check passed.
- Focused toroidal hybrid example/plot tests passed: `5 passed in 5.96s`.
- Full toroidal hybrid test file passed: `28 passed in 6.65s`.
- Combined mirror plus toroidal hybrid validation passed: `260 passed,
  1 skipped in 223.64s`.

### File structure and best-practice notes

- The standalone import bootstrap is local to the two root-level example
  scripts, matching the pattern already used by other root mirror examples.
- The plot tests stay in `tests/test_toroidal_hybrid.py`, next to the toroidal
  hybrid geometry, indata, convergence, and helper tests.
- No generated images are tracked; probes used `/tmp`, while tests use pytest
  temporary directories.
- The convergence plot coverage stays on the no-solve path, so it protects the
  geometry/report artifacts without increasing CI runtime with toroidal fixed
  boundary solves.

### Best next steps

1. Commit and push M183.
2. Update the draft PR body to include the `260 passed, 1 skipped` combined
   mirror plus toroidal-hybrid validation result and the standalone hybrid plot
   coverage.
3. Inspect only failed CI jobs after the latest push; queued jobs can finish in
   the background.
4. Continue the completion audit, with likely next targets being final
   free-boundary documentation boundaries, Boozer-like diagnostic promotion
   wording, or a higher-budget toroidal hybrid solved convergence study if time
   and runtime budget allow.

### Completion percentages after M183

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `93%`.
- Residual Newton / preconditioning: `94%`.
- Two-coil and manufactured validation: `92%`.
- Finite-current pitch validation: `92%`.
- Plotting and `vmec --plot` mirror support: `99%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `96%`.
- Mirror-Boozer-like diagnostics: `89%`.
- Free-boundary mirror lane: `99%` overall, with reduced residual-vector
  nonlinear solve scope complete.
- Straight-axis hybrid support fixture lane: `100%` for support-fixture scope.
- Toroidal stellarator-mirror hybrid lane: `96%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- Public API/source simplification: `100%` for the mirror package initializer.
- PR merge readiness overall: `99%`.

### User input needed

No user input is needed.


---
## 184. Mirror-Boozer-Like Diagnostic Contract Coverage

### Steps taken

- Audited the mirror-Boozer-like diagnostic implementation, docs, public API,
  plotting bundle, IO exports, and existing tests.
- Confirmed existing coverage already checks surface-average bounds, mirror
  ratio, finite-current field-line turns, contravariant pitch, public summary
  metrics, and plotted Boozer-like diagnostics.
- Added direct tests for the remaining exported diagnostic fields:
  - positive Jacobian-weighted surface measure;
  - zero-current covariant pitch proxy;
  - finite-current covariant pitch proxy;
  - magnetic-well-proxy summary extrema;
  - covariant pitch summary extrema.

### Results obtained

- The mirror-Boozer-like diagnostic lane now has direct coverage for every
  scalar family exported in the summary and CSV/NPZ radial diagnostic contract.
- The finite-current diagnostic test now checks both contravariant and
  covariant pitch proxies, so pitch regressions are less likely to hide behind
  only field-line-turn checks.
- The implementation remains explicitly mirror-native and does not claim
  toroidal Boozer coordinates or toroidal rotational transform.

### How it was tested

Commands run:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_plotting.py::test_mirror_plot_data_helpers_expose_numerical_content \
  tests/mirror/test_mirror_plotting.py::test_mirror_boozer_like_diagnostics_capture_finite_current_pitch -q
python -m ruff check tests/mirror/test_mirror_plotting.py
python -m ruff format --check tests/mirror/test_mirror_plotting.py
JAX_ENABLE_X64=1 pytest tests/mirror/test_mirror_plotting.py -q
JAX_ENABLE_X64=1 pytest tests/mirror -q
```

Results:

- Focused Boozer-like diagnostic tests passed: `2 passed in 1.25s`.
- Ruff check and format check passed.
- Full mirror plotting test file passed: `5 passed in 5.00s`.
- Full mirror suite passed: `232 passed, 1 skipped in 218.77s`.

### File structure and best-practice notes

- The new checks stay in `tests/mirror/test_mirror_plotting.py`, next to the
  plotting-data helper tests that already construct the relevant small mirror
  outputs.
- No new fixtures or generated files were added.
- The added assertions strengthen the public diagnostic data contract without
  changing implementation behavior or introducing brittle image snapshots.

### Best next steps

1. Commit and push M184.
2. Update the draft PR body with section 184 and the latest mirror-suite
   validation result.
3. Inspect only failed CI jobs after the latest push.
4. Continue the final audit across remaining non-100% lanes, especially the
   boundary between diagnostic free-boundary evidence and future production
   free-boundary claims.

### Completion percentages after M184

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `93%`.
- Residual Newton / preconditioning: `94%`.
- Two-coil and manufactured validation: `92%`.
- Finite-current pitch validation: `92%`.
- Plotting and `vmec --plot` mirror support: `99%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `96%`.
- Mirror-Boozer-like diagnostics: `93%`.
- Free-boundary mirror lane: `99%` overall, with reduced residual-vector
  nonlinear solve scope complete.
- Straight-axis hybrid support fixture lane: `100%` for support-fixture scope.
- Toroidal stellarator-mirror hybrid lane: `96%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- Public API/source simplification: `100%` for the mirror package initializer.
- PR merge readiness overall: `99%`.

### User input needed

No user input is needed.


---
## 185. Free-Boundary Vector-LS Plot Coverage

### Steps taken

- Audited the free-boundary lane docs, readiness matrix, reduced residual-vector
  benchmark, and existing tests.
- Confirmed the diagnostic-vs-production claim boundary is explicit: the
  circular-coil bridge is documented as an LCFS pilot/reduced residual-vector
  prototype, not a converged production free-boundary equilibrium solve.
- Identified one concrete artifact gap: the reduced vector-LS benchmark was
  tested only with `--no-plots`, so its backend summary, radius-profile, and
  solve-history plots could regress unnoticed.
- Added a plotted root-example smoke test for
  `examples/mirror_free_boundary_vector_ls_benchmark.py`.
- The new test reuses the existing metrics validator and checks that all three
  generated PNGs are readable and nonblank.

### Results obtained

- The reduced free-boundary residual-vector benchmark now has both no-plot
  numerical/schema coverage and plotted artifact coverage.
- The free-boundary lane keeps its current diagnostic scope while strengthening
  the benchmark artifacts used to compare finite-difference, JAX forward, JAX
  reverse, and automatic Jacobian backends.
- No generated figures were added to the repository.

### How it was tested

Commands run:

```bash
JAX_ENABLE_X64=1 python examples/mirror_free_boundary_vector_ls_benchmark.py \
  --outdir /tmp/mirror_vector_ls_plot_probe \
  --nxi 9
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_vector_ls_benchmark_runs_without_plots \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_vector_ls_benchmark_writes_nonblank_plots -q
python -m ruff check tests/mirror/test_mirror_examples.py
python -m ruff format --check tests/mirror/test_mirror_examples.py
JAX_ENABLE_X64=1 pytest tests/mirror -q
```

Results:

- Plot probe wrote nonblank backend-summary, radius-profile, and solve-history
  figures.
- Focused vector-LS example tests passed: `2 passed in 4.76s`.
- Ruff check and format check passed.
- Full mirror suite passed: `233 passed, 1 skipped in 221.88s`.

### File structure and best-practice notes

- The new test stays in `tests/mirror/test_mirror_examples.py` beside the
  existing root-example smoke tests.
- It uses the existing example-level JSON validator, then checks only generic
  image readability and nonblank content to avoid brittle pixel snapshots.
- The benchmark remains a lightweight reduced residual-vector diagnostic; it
  does not run coupled fixed-boundary pilot solves.

### Best next steps

1. Commit and push M185.
2. Update the draft PR body with section 185 and the `233 passed, 1 skipped`
   full mirror-suite result.
3. Inspect only failed CI jobs after the latest push.
4. Continue the completion audit across remaining validation percentages,
   especially two-coil/finite-current convergence and fixed-boundary solve
   promotion language.

### Completion percentages after M185

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `93%`.
- Residual Newton / preconditioning: `94%`.
- Two-coil and manufactured validation: `92%`.
- Finite-current pitch validation: `92%`.
- Plotting and `vmec --plot` mirror support: `99%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `96%`.
- Mirror-Boozer-like diagnostics: `93%`.
- Free-boundary mirror lane: `99%` overall, with reduced residual-vector
  nonlinear solve scope complete and benchmark plots covered.
- Straight-axis hybrid support fixture lane: `100%` for support-fixture scope.
- Toroidal stellarator-mirror hybrid lane: `96%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- Public API/source simplification: `100%` for the mirror package initializer.
- PR merge readiness overall: `99%`.

### User input needed

No user input is needed.


---
## 186. Residual-Newton Convergence-Grid Plot Coverage

### Steps taken

- Audited the fixed-boundary residual-Newton convergence-grid lane and its
  existing tests.
- Confirmed the example already has no-plot numerical/schema coverage for the
  small two-coil row and finite-current lambda-residual diagnostics.
- Identified a remaining artifact gap: the convergence-grid plots and selected
  best-row mirror plot bundle were not exercised by automated tests.
- Added a plotted root-example smoke test for
  `examples/mirror_residual_newton_convergence_grid.py` at the same small
  `ns=5`, `nxi=9`, `maxiter=2`, `residual_linear_maxiter=8` settings used by
  the existing no-plot test.
- Added a small shared `_assert_nonblank_image` helper in
  `tests/mirror/test_mirror_examples.py` and reused it for existing plotted
  example checks.

### Results obtained

- The residual-Newton convergence-grid example now has automated coverage for:
  - resolution heatmap;
  - budget plot;
  - residual-history plot;
  - residual-component plot;
  - selected best-row `mout` artifact;
  - representative best-row 3D boundary and mirror-Boozer-like diagnostic
    plots.
- This closes a plot/reporting gap in the fixed-boundary convergence lane
  without changing the solver implementation or raising CI budgets.

### How it was tested

Commands run:

```bash
JAX_ENABLE_X64=1 python examples/mirror_residual_newton_convergence_grid.py \
  --outdir /tmp/mirror_residual_newton_convergence_plot_probe \
  --ns-array 5 \
  --nxi-array 9 \
  --maxiter-array 2 \
  --residual-linear-maxiter-array 8 \
  --preconditioners radial_xi_tridi
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_examples.py::test_root_finite_current_pitch_example_writes_nonblank_field_line_plots \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_vector_ls_benchmark_writes_nonblank_plots \
  tests/mirror/test_mirror_examples.py::test_root_implicit_parameter_gradients_example_matrix_free_writes_plot \
  tests/mirror/test_mirror_examples.py::test_root_residual_newton_convergence_grid_runs_without_plots \
  tests/mirror/test_mirror_examples.py::test_root_residual_newton_convergence_grid_writes_nonblank_plots -q
python -m ruff check tests/mirror/test_mirror_examples.py
python -m ruff format --check tests/mirror/test_mirror_examples.py
JAX_ENABLE_X64=1 pytest tests/mirror -q
```

Results:

- Plot probe wrote nonblank convergence-grid figures and selected best-row
  mirror plots.
- Focused plotted-example group passed: `5 passed in 30.05s`.
- Ruff check and format check passed.
- Full mirror suite passed: `234 passed, 1 skipped in 225.50s`.

### File structure and best-practice notes

- The new coverage stays in `tests/mirror/test_mirror_examples.py` with other
  root-example smoke tests.
- The helper removes repeated image-readability assertions without adding new
  testing dependencies.
- The test protects report artifacts using generic nonblank checks rather than
  brittle pixel snapshots.
- Generated convergence figures remain in pytest temporary directories or
  ignored result paths.

### Best next steps

1. Commit and push M186.
2. Update the draft PR body with section 186 and the `234 passed, 1 skipped`
   full mirror-suite result.
3. Inspect only failed CI jobs after the latest push.
4. Continue the final audit, focusing next on whether two-coil and
   finite-current convergence percentages should be promoted, explicitly
   deferred, or backed by another benchmark gate.

### Completion percentages after M186

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `94%`.
- Residual Newton / preconditioning: `95%`.
- Two-coil and manufactured validation: `93%`.
- Finite-current pitch validation: `92%`.
- Plotting and `vmec --plot` mirror support: `99%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `96%`.
- Mirror-Boozer-like diagnostics: `93%`.
- Free-boundary mirror lane: `99%` overall, with reduced residual-vector
  nonlinear solve scope complete and benchmark plots covered.
- Straight-axis hybrid support fixture lane: `100%` for support-fixture scope.
- Toroidal stellarator-mirror hybrid lane: `96%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- Public API/source simplification: `100%` for the mirror package initializer.
- PR merge readiness overall: `99%`.

### User input needed

No user input is needed.


---
## 187. Two-Coil Benchmark Plot Coverage

### Steps taken

- Audited `examples/mirror_two_coil_axisym.py` and confirmed it already writes
  the requested analytic two-coil plot bundle:
  - on-axis `B_z` comparison;
  - 3D mirror geometry with coils and field lines;
  - 3D boundary `|B|` with coils and field lines;
  - off-axis low-radius Biot-Savart `B_r` and `B_z` comparison;
  - convergence study over the fixed `ns/nxi` benchmark grid;
  - the standard mirror plot bundle from `plot_mirror_output`.
- Ran a temporary plotted probe to inspect the actual artifact names, image
  sizes, nonblank image statistics, and benchmark JSON metrics.
- Added a focused root-example smoke test that runs the two-coil example with
  plotting enabled at small `ns=5`, `nxi=9`, `maxiter=0` settings.
- Checked that the plotted run writes the benchmark convergence JSON and
  representative nonblank analytic/3D/diagnostic PNGs.

### Results obtained

- The two-coil analytic benchmark now has automated coverage for the figures
  the user reviews most directly:
  - `two_coil_axisym_axis_bz_comparison.png`;
  - `two_coil_axisym_geometry_with_coils.png`;
  - `two_coil_axisym_bmag_with_coils.png`;
  - `two_coil_axisym_off_axis_biot_savart_comparison.png`;
  - `two_coil_axisym_convergence.png`;
  - `two_coil_axisym_mirror_boundary_3d.png`;
  - `two_coil_axisym_mirror_boozer_like_diagnostics.png`.
- The test asserts the on-axis analytic agreement remains near machine
  precision and that the convergence study still reports the intended
  `(ns, nxi)` rows `(7, 17)`, `(9, 25)`, and `(11, 33)`.
- No source behavior changed; this tranche closes a reporting/plot coverage
  gap around an already implemented research benchmark.

### How it was tested

Commands run:

```bash
JAX_ENABLE_X64=1 python examples/mirror_two_coil_axisym.py \
  --outdir /tmp/mirror_two_coil_axisym_plot_probe \
  --ns 5 \
  --nxi 9 \
  --maxiter 0
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_examples.py::test_root_two_coil_axisym_example_runs_without_plots \
  tests/mirror/test_mirror_examples.py::test_root_two_coil_axisym_example_writes_nonblank_benchmark_plots -q
python -m ruff check tests/mirror/test_mirror_examples.py
python -m ruff format --check tests/mirror/test_mirror_examples.py
JAX_ENABLE_X64=1 pytest tests/mirror -q
```

Results:

- The temporary probe wrote 16 PNG files in the two-coil figure directory; all
  inspected files had nonzero image variance and expected file sizes.
- Focused two-coil tests passed: `2 passed in 7.38s`.
- Ruff check and format check passed.
- Full mirror suite passed: `235 passed, 1 skipped in 227.87s`.

### File structure and best-practice notes

- The new coverage is localized to
  `tests/mirror/test_mirror_examples.py`, alongside other root example smoke
  tests.
- The test reuses the shared nonblank-image helper introduced in M186, keeping
  image checks generic and avoiding brittle pixel snapshots.
- Generated figures remain in pytest temporary directories or ignored result
  paths; no binary output was added to the repository.
- The two-coil example remains in the repository-root `examples/` folder as the
  user requested, while detailed mirror package functionality remains under
  `vmec_jax/mirror/`.

### Best next steps

1. Commit and push M187.
2. Update the draft PR body with section 187 and the `235 passed, 1 skipped`
   full mirror-suite result.
3. Inspect only failed CI jobs after the push.
4. Continue the final audit by checking whether finite-current pitch and
   Boozer-like diagnostics need one last numerical gate or can be promoted as
   complete for this draft-PR scope.

### Completion percentages after M187

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `94%`.
- Residual Newton / preconditioning: `95%`.
- Two-coil and manufactured validation: `94%`.
- Finite-current pitch validation: `92%`.
- Plotting and `vmec --plot` mirror support: `99%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `96%`.
- Mirror-Boozer-like diagnostics: `93%`.
- Free-boundary mirror lane: `99%` overall, with reduced residual-vector
  nonlinear solve scope complete and benchmark plots covered.
- Straight-axis hybrid support fixture lane: `100%` for support-fixture scope.
- Toroidal stellarator-mirror hybrid lane: `96%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- Public API/source simplification: `100%` for the mirror package initializer.
- PR merge readiness overall: `99%`.

### User input needed

No user input is needed.

---
## 188. Finite-Current Convergence-Grid Plot Coverage

### Steps taken

- Audited the finite-current residual-Newton convergence-grid path after M187
  promoted the two-coil analytic plot coverage.
- Ran a temporary plotted finite-current convergence-grid probe with
  `i_prime=0.01`, the `radial_xi_lambda_xi_tridi` preconditioner, adaptive
  inner iteration budgets, and `residual_xi_alpha=1.0`.
- Confirmed the example writes the same four convergence/report plots as the
  vacuum two-coil grid plus a selected best-row finite-current mirror plot
  bundle.
- Added a focused plotted smoke test for the finite-current convergence-grid
  path.  The test checks the finite-current/lambda-residual JSON contract,
  nonblank convergence figures, positive selected-output cap-to-cap pitch, and
  representative nonblank 3D, field-direction, radial, and mirror-Boozer-like
  selected-row plots.

### Results obtained

- The finite-current pitch lane now has automated coverage for the plotted
  residual-Newton convergence-grid artifacts, not only the no-plot lambda
  residual row.
- The selected best-row finite-current output is loaded back from MOUT and
  checked through `mirror_field_line_pitch_profile_data`, so the plot test is
  tied to a physical positive-pitch diagnostic rather than only to file
  existence.
- This strengthens the “supported diagnostic” claim for finite-current pitch
  and mirror-Boozer-like reporting without changing the solver or promoting the
  lambda-dominated finite-current residual-Newton row to a production
  tight-convergence claim.

### How it was tested

Commands run:

```bash
JAX_ENABLE_X64=1 python examples/mirror_residual_newton_convergence_grid.py \
  --outdir /tmp/mirror_finite_current_convergence_plot_probe \
  --ns-array 5 \
  --nxi-array 9 \
  --maxiter-array 1 \
  --residual-linear-maxiter-array 8 \
  --residual-linear-maxiter-policy adaptive \
  --i-prime 0.01 \
  --preconditioners radial_xi_lambda_xi_tridi \
  --residual-xi-alpha 1.0
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_examples.py::test_root_residual_newton_convergence_grid_finite_current_reports_lambda_residual \
  tests/mirror/test_mirror_examples.py::test_root_residual_newton_convergence_grid_finite_current_writes_nonblank_plots -q
python -m ruff check tests/mirror/test_mirror_examples.py
python -m ruff format --check tests/mirror/test_mirror_examples.py
JAX_ENABLE_X64=1 pytest tests/mirror -q
```

Results:

- The temporary probe wrote four convergence-grid plots and eleven selected
  best-row finite-current mirror plots; inspected PNGs were nonblank.
- Focused finite-current convergence-grid tests passed: `2 passed in 13.94s`.
- Ruff check and format check passed.
- Full mirror suite passed: `236 passed, 1 skipped in 234.89s`.

### File structure and best-practice notes

- The new test remains in `tests/mirror/test_mirror_examples.py`, next to the
  existing finite-current no-plot convergence-grid test.
- It reuses the shared nonblank-image helper and existing public pitch-data
  loader, avoiding new image-baseline files or binary repository artifacts.
- Generated figures remain in pytest temporary directories or ignored result
  paths.
- The finite-current example and convergence-grid runner remain in the
  repository-root `examples/` folder, while plotting and pitch calculations
  stay in `vmec_jax/mirror/plotting/`.

### Best next steps

1. Commit and push M188.
2. Update the draft PR body with section 188 and the `236 passed, 1 skipped`
   full mirror-suite result.
3. Inspect only failed CI jobs after the push.
4. Continue the final audit by checking whether the readiness matrix and docs
   now accurately state the finite-current/Boozer-like diagnostic scope, then
   move to the next remaining non-100% lane.

### Completion percentages after M188

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `93%`.
- Fixed-boundary axisymmetric solve: `94%`.
- Residual Newton / preconditioning: `95%`.
- Two-coil and manufactured validation: `94%`.
- Finite-current pitch validation: `94%`.
- Plotting and `vmec --plot` mirror support: `99%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `96%`.
- Mirror-Boozer-like diagnostics: `94%`.
- Free-boundary mirror lane: `99%` overall, with reduced residual-vector
  nonlinear solve scope complete and benchmark plots covered.
- Straight-axis hybrid support fixture lane: `100%` for support-fixture scope.
- Toroidal stellarator-mirror hybrid lane: `96%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- Public API/source simplification: `100%` for the mirror package initializer.
- PR merge readiness overall: `99%`.

### User input needed

No user input is needed.

---
## 189. Solver-Comparison Plot Coverage

### Steps taken

- Audited `examples/mirror_solver_comparison.py` after M188 to identify the
  next unguarded review-facing plot path in the fixed-boundary and
  residual-Newton lanes.
- Ran a plotted solver-comparison probe with the same tiny settings as the
  existing no-plot test: cylinder, two-coil, and manufactured cases; one
  gradient-descent step; two L-BFGS and residual-Newton steps; and a small
  two-coil grid.
- Confirmed the example writes summary residual-history, final-residual, and
  physical-boundary plots plus selected residual-Newton mirror plot bundles for
  the cylinder and two-coil physical cases.
- Added a plotted smoke test that checks the summary plot set, residual-Newton
  row metadata, selected MOUT readability, and representative nonblank selected
  mirror plots.

### Results obtained

- The solver-comparison example now has automated coverage for the plots used
  to compare gradient descent, scaled L-BFGS-B, residual Newton, and the
  manufactured residual-Newton gate.
- The test ties the selected physical plot bundles back to readable mirror
  outputs with positive Jacobian diagnostics, not only to PNG existence.
- This strengthens the fixed-boundary, residual-Newton/preconditioning, and
  two-coil/manufactured validation lanes without changing solver behavior or
  adding binary artifacts.

### How it was tested

Commands run:

```bash
JAX_ENABLE_X64=1 python examples/mirror_solver_comparison.py \
  --outdir /tmp/mirror_solver_comparison_plot_probe \
  --cases cylinder,two_coil,manufactured \
  --maxiter-gd 1 \
  --maxiter-lbfgs 2 \
  --maxiter-newton 2 \
  --two-coil-ns 5 \
  --two-coil-nxi 9 \
  --residual-linear-maxiter 12
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_examples.py::test_root_solver_comparison_example_runs_without_plots \
  tests/mirror/test_mirror_examples.py::test_root_solver_comparison_example_writes_nonblank_plots -q
python -m ruff format tests/mirror/test_mirror_examples.py
python -m ruff check tests/mirror/test_mirror_examples.py
python -m ruff format --check tests/mirror/test_mirror_examples.py
JAX_ENABLE_X64=1 pytest tests/mirror -q
```

Results:

- The temporary probe wrote three summary plots, two selected residual-Newton
  MOUT files, and twenty-two selected mirror PNGs; inspected PNGs were
  nonblank.
- Focused solver-comparison tests passed: `2 passed in 23.52s` after
  formatting.
- Ruff check and format check passed.
- Full mirror suite passed: `237 passed, 1 skipped in 249.33s`.

### File structure and best-practice notes

- The new coverage stays in `tests/mirror/test_mirror_examples.py`, next to the
  existing solver-comparison root-example smoke test.
- The test reuses the common nonblank-image helper and public MOUT loader.
- Generated figures remain in pytest temporary directories or ignored result
  paths; no reference images were committed.
- Solver-comparison plotting remains owned by the root example, while the
  standard selected mirror bundles continue to flow through
  `vmec_jax.mirror.plot_mirror_output`.

### Best next steps

1. Commit and push M189.
2. Update the draft PR body with section 189 and the `237 passed, 1 skipped`
   full mirror-suite result.
3. Inspect only failed CI jobs after the push.
4. Continue the final audit with the fixed-boundary solve diagnostic example
   and documentation/readiness wording for the remaining non-100% lanes.

### Completion percentages after M189

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `94%`.
- Fixed-boundary axisymmetric solve: `95%`.
- Residual Newton / preconditioning: `96%`.
- Two-coil and manufactured validation: `95%`.
- Finite-current pitch validation: `94%`.
- Plotting and `vmec --plot` mirror support: `99%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `96%`.
- Mirror-Boozer-like diagnostics: `94%`.
- Free-boundary mirror lane: `99%` overall, with reduced residual-vector
  nonlinear solve scope complete and benchmark plots covered.
- Straight-axis hybrid support fixture lane: `100%` for support-fixture scope.
- Toroidal stellarator-mirror hybrid lane: `96%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- Public API/source simplification: `100%` for the mirror package initializer.
- PR merge readiness overall: `99%`.

### User input needed

No user input is needed.

---
## 190. Fixed-Boundary Solve Diagnostic Plot Coverage

### Steps taken

- Audited `examples/mirror_fixed_boundary_solve_diagnostic.py`, which is the
  documented fixed-boundary diagnostic runner for the requested `ns_array`,
  `maxiter`, `ftol`, and cross-section plotting lane.
- Ran a plotted probe at the existing small CI settings: `ns=7`, `nxi=13`, and
  `maxiter=2`.
- Confirmed the JSON row records the MOUT path and solver diagnostics while the
  standard mirror plot bundle is written beside that MOUT under a `figures/`
  directory.
- Added a plotted smoke test that verifies the JSON contract, reloads the MOUT,
  and checks representative nonblank 3D, cross-section, field-direction,
  residual-history, and mirror-Boozer-like figures.

### Results obtained

- The fixed-boundary solve diagnostic example now has automated coverage for
  the plot bundle users inspect when diagnosing convergence and geometry.
- The test verifies positive Jacobian diagnostics from the reloaded MOUT and
  nonnegative `fsq`/normalized-force reporting from the JSON row.
- This strengthens the fixed-boundary axisymmetric solve, field/residual
  diagnostic, and plotting lanes without changing solver behavior or tracking
  generated images.

### How it was tested

Commands run:

```bash
JAX_ENABLE_X64=1 python examples/mirror_fixed_boundary_solve_diagnostic.py \
  --outdir /tmp/mirror_fixed_boundary_solve_plot_probe \
  --ns-array 7 \
  --nxi 13 \
  --maxiter 2
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_examples.py::test_root_fixed_boundary_solve_diagnostic_runs_without_plots \
  tests/mirror/test_mirror_examples.py::test_root_fixed_boundary_solve_diagnostic_writes_nonblank_plots \
  tests/mirror/test_mirror_examples.py::test_root_fixed_boundary_solve_diagnostic_residual_newton_reports_krylov_fields -q
python -m ruff check tests/mirror/test_mirror_examples.py
python -m ruff format --check tests/mirror/test_mirror_examples.py
JAX_ENABLE_X64=1 pytest tests/mirror -q
```

Results:

- The temporary probe wrote one diagnostic JSON row, one MOUT file, and eleven
  standard mirror PNGs; inspected PNGs were nonblank.
- Focused fixed-boundary diagnostic tests passed: `3 passed in 9.69s`.
- Ruff check and format check passed.
- Full mirror suite passed: `238 passed, 1 skipped in 250.30s`.

### File structure and best-practice notes

- The new coverage stays in `tests/mirror/test_mirror_examples.py`, next to the
  existing fixed-boundary diagnostic root-example tests.
- The test infers the figure directory from the JSON `mout` path because the
  example intentionally keeps the JSON row compact.
- Plot generation remains in `plot_mirror_output`; no example-specific image
  assertions or committed binary artifacts were added.
- Generated probe and pytest outputs stay under `/tmp` or pytest temporary
  directories.

### Best next steps

1. Commit and push M190.
2. Update the draft PR body with section 190 and the `238 passed, 1 skipped`
   full mirror-suite result.
3. Inspect only failed CI jobs after the push.
4. Continue the final audit with docs/readiness synchronization and remaining
   differentiability, toroidal-hybrid, and ESSOS/free-boundary percentages.

### Completion percentages after M190

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `95%`.
- Fixed-boundary axisymmetric solve: `96%`.
- Residual Newton / preconditioning: `96%`.
- Two-coil and manufactured validation: `95%`.
- Finite-current pitch validation: `94%`.
- Plotting and `vmec --plot` mirror support: `99%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `96%`.
- Mirror-Boozer-like diagnostics: `94%`.
- Free-boundary mirror lane: `99%` overall, with reduced residual-vector
  nonlinear solve scope complete and benchmark plots covered.
- Straight-axis hybrid support fixture lane: `100%` for support-fixture scope.
- Toroidal stellarator-mirror hybrid lane: `96%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- Public API/source simplification: `100%` for the mirror package initializer.
- PR merge readiness overall: `99%`.

### User input needed

No user input is needed.

---
## 191. Implicit Solve Benchmark Plot Coverage

### Steps taken

- Audited the differentiable solved-state lane after M190 and identified that
  `examples/mirror_implicit_solve_benchmark.py` had no automated plot coverage.
- Rechecked the current external differentiability guidance for custom JAX
  derivative rules, JAXopt implicit differentiation, and Lineax-style linear
  solver abstractions to keep the lane aligned with implicit/linear-solve
  differentiation rather than unrolled CLI solves.
- Ran a plotted implicit-solve benchmark probe at the existing small CI
  settings: `ns=5`, `nxi=7`, and `repeat=1`.
- Added a plotted smoke test that verifies dense and matrix-free CG benchmark
  rows, CSV output, matrix-free error/residual tolerances, and a nonblank
  runtime/memory/error summary plot.

### Results obtained

- The implicit solve benchmark now has automated coverage for its review-facing
  summary plot as well as the dense-vs-matrix-free CG JSON/CSV contract.
- The test keeps dense solves as the tiny-grid correctness reference and checks
  the matrix-free CG row against the existing `1e-5` relative-error tolerance.
- This strengthens the differentiable solved-state API lane without changing
  the public API or differentiating through host-side benchmark/reporting code.

### How it was tested

Commands run:

```bash
JAX_ENABLE_X64=1 python examples/mirror_implicit_solve_benchmark.py \
  --outdir /tmp/mirror_implicit_solve_benchmark_plot_probe \
  --ns-array 5 \
  --nxi-array 7 \
  --repeat 1
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_examples.py::test_root_implicit_solve_benchmark_runs_without_plots \
  tests/mirror/test_mirror_examples.py::test_root_implicit_solve_benchmark_writes_nonblank_plot -q
python -m ruff check tests/mirror/test_mirror_examples.py
python -m ruff format --check tests/mirror/test_mirror_examples.py
JAX_ENABLE_X64=1 pytest tests/mirror -q
```

Results:

- The temporary probe wrote the benchmark metrics JSON, CSV, and nonblank
  `mirror_implicit_solve_benchmark.png` figure.
- Focused implicit benchmark tests passed: `2 passed in 16.46s`.
- Ruff check and format check passed.
- Full mirror suite passed: `239 passed, 1 skipped in 258.71s`.

### File structure and best-practice notes

- The new coverage stays in `tests/mirror/test_mirror_examples.py`, beside the
  no-plot implicit solve benchmark test.
- The benchmark example keeps host-side timing, memory tracing, CSV writing,
  and plotting out of the differentiable API; the differentiable path remains
  the reduced JAX residual and implicit linear-solve wrappers.
- No generated benchmark artifacts were committed.

### Best next steps

1. Commit and push M191.
2. Update the draft PR body with section 191 and the `239 passed, 1 skipped`
   full mirror-suite result.
3. Inspect only failed CI jobs after the push.
4. Continue the final audit with toroidal-hybrid convergence/readiness and the
   ESSOS/free-boundary diagnostic evidence that remains below 100%.

### Completion percentages after M191

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `95%`.
- Fixed-boundary axisymmetric solve: `96%`.
- Residual Newton / preconditioning: `96%`.
- Two-coil and manufactured validation: `95%`.
- Finite-current pitch validation: `94%`.
- Plotting and `vmec --plot` mirror support: `99%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `97%`.
- Mirror-Boozer-like diagnostics: `94%`.
- Free-boundary mirror lane: `99%` overall, with reduced residual-vector
  nonlinear solve scope complete and benchmark plots covered.
- Straight-axis hybrid support fixture lane: `100%` for support-fixture scope.
- Toroidal stellarator-mirror hybrid lane: `96%`.
- ESSOS circular-coil mirror beta scan: `97%`.
- Public API/source simplification: `100%` for the mirror package initializer.
- PR merge readiness overall: `99%`.

### User input needed

No user input is needed.

---
## 192. Circular-Coil Free-Boundary Plot Coverage

### Steps taken

- Audited `examples/mirror_free_boundary_circular_coils.py` and the existing
  free-boundary root-example tests after M191.
- Identified that the circular-coil beta-scan workflow had extensive no-plot
  schema and guard coverage, but no automated check for the plotted top-level
  coil/axis/boundary/beta-summary figures or per-beta baseline/pilot plot
  bundles.
- Ran a plotted low-resolution 1%, 3%, and 10% beta-scan probe with fixed
  boundary baselines, one LCFS pilot step per beta, and direct circular coils.
- Added a plotted smoke test that validates the metrics schema, checks the
  top-level figure set, and verifies representative nonblank baseline and pilot
  mirror/LCFS diagnostic plots for the 1% beta row.

### Results obtained

- The ESSOS-compatible circular-coil free-boundary diagnostic lane now has
  automated coverage for its main plotted beta-scan report artifacts.
- The test still uses the lightweight diagnostic/pilot settings already used by
  the no-plot schema test, so it does not promote the lane to a converged
  production free-boundary solver claim.
- The coverage directly checks the 1%, 3%, and 10% beta request, the LCFS pilot
  row count, top-level plots, and representative baseline/pilot plot bundles.

### How it was tested

Commands run:

```bash
JAX_ENABLE_X64=1 python examples/mirror_free_boundary_circular_coils.py \
  --outdir /tmp/mirror_free_boundary_circular_coils_plot_probe \
  --ntheta 8 \
  --nxi 11 \
  --n-segments 64 \
  --run-fixed-boundary-baseline \
  --baseline-maxiter 0 \
  --run-lcfs-pilot \
  --lcfs-pilot-steps 1
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_example_runs_without_plots \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_circular_coils_example_writes_nonblank_plots -q
python -m ruff check tests/mirror/test_mirror_examples.py
python -m ruff format --check tests/mirror/test_mirror_examples.py
JAX_ENABLE_X64=1 pytest tests/mirror -q
```

Results:

- The temporary probe wrote top-level axis, boundary, geometry, and beta-summary
  figures plus baseline and LCFS-pilot mirror plot bundles for 1%, 3%, and 10%
  beta cases; inspected PNGs were nonblank.
- Focused circular-coil beta-scan tests passed: `2 passed in 14.27s`.
- Ruff check and format check passed.
- Full mirror suite passed: `240 passed, 1 skipped in 270.90s`.

### File structure and best-practice notes

- The new coverage stays in `tests/mirror/test_mirror_examples.py`, beside the
  existing circular-coil beta-scan schema test.
- It reuses the existing metrics validator and nonblank-image helper rather
  than adding image baselines.
- Generated plots remain in pytest temporary directories or ignored result
  paths, keeping the repository light.
- The example keeps ESSOS-compatible coil metadata and diagnostic LCFS pilot
  reporting separate from the future production free-boundary nonlinear solver.

### Best next steps

1. Commit and push M192.
2. Update the draft PR body with section 192 and the `240 passed, 1 skipped`
   full mirror-suite result.
3. Inspect only failed CI jobs after the push.
4. Continue the final audit with toroidal-hybrid convergence/readiness and a
   full docs/readiness synchronization pass before considering undrafting.

### Completion percentages after M192

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `95%`.
- Fixed-boundary axisymmetric solve: `96%`.
- Residual Newton / preconditioning: `96%`.
- Two-coil and manufactured validation: `95%`.
- Finite-current pitch validation: `94%`.
- Plotting and `vmec --plot` mirror support: `99%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `97%`.
- Mirror-Boozer-like diagnostics: `94%`.
- Free-boundary mirror lane: `99%` overall, with circular-coil beta-scan plots
  and reduced residual-vector benchmark plots covered.
- Straight-axis hybrid support fixture lane: `100%` for support-fixture scope.
- Toroidal stellarator-mirror hybrid lane: `96%`.
- ESSOS circular-coil mirror beta scan: `99%`.
- Public API/source simplification: `100%` for the mirror package initializer.
- PR merge readiness overall: `99%`.

### User input needed

No user input is needed.

---
## 193. Readiness Documentation Synchronization

### Steps taken

- Audited `docs/mirror/readiness.rst` and `docs/mirror/overview.rst` after the
  plotted fixed-boundary, differentiability, circular-coil, finite-current, and
  solver-comparison coverage tranches.
- Updated the readiness matrix so the current claims mention fixed-boundary
  diagnostic examples, solver-comparison reports, two-coil plot evidence,
  dense-vs-matrix-free implicit benchmark plots, circular-coil beta-scan plots,
  and low-resolution toroidal-hybrid parity scope.
- Updated the overview to stop describing the ESSOS-compatible beta-scan example
  and toroidal-hybrid parity rows as future work.  The docs now label them as
  current diagnostic/prototype evidence and keep only production LCFS solves,
  production differentiable optimization APIs, and target-resolution toroidal
  hybrid convergence studies as later work.
- Ran the docs build with warnings treated as errors and reran the toroidal
  hybrid test file.

### Results obtained

- The documentation now matches the implemented status more closely: broad
  production claims remain explicitly out of scope, while current diagnostic and
  validated-prototype evidence is no longer described as missing.
- The toroidal hybrid lane remains a validated prototype: low-resolution
  VMEC2000 parity and no-solve convergence/plot tests are covered, while the
  final target-resolution convergence ladder remains a promotion gate.
- The free-boundary circular-coil lane remains diagnostic/pilot evidence, but
  the implemented 1%, 3%, and 10% beta scan plots and schema are now reflected
  in the readiness page.

### How it was tested

Commands run:

```bash
python -m sphinx -W -b html docs docs/_build/html
JAX_ENABLE_X64=1 pytest tests/test_toroidal_hybrid.py -q
git diff --check
```

Results:

- Sphinx docs build passed with warnings as errors.
- Toroidal hybrid tests passed: `28 passed in 6.41s`.
- Whitespace check passed.

### File structure and best-practice notes

- The readiness matrix remains the review-facing scope summary in
  `docs/mirror/readiness.rst`.
- The broader narrative remains in `docs/mirror/overview.rst`.
- No generated docs output was committed; the Sphinx build wrote under the
  ignored `docs/_build/html` tree.

### Best next steps

1. Commit and push M193.
2. Update the draft PR body with section 193 and the docs/toroidal validation
   result.
3. Inspect only failed CI jobs after the push.
4. Run a final local audit pass over changed files, public docs, PR body, and
   current CI before deciding whether any remaining lanes need code changes or
   only explicit deferral in the readiness matrix.

### Completion percentages after M193

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `95%`.
- Fixed-boundary axisymmetric solve: `96%`.
- Residual Newton / preconditioning: `96%`.
- Two-coil and manufactured validation: `95%`.
- Finite-current pitch validation: `94%`.
- Plotting and `vmec --plot` mirror support: `99%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `97%`.
- Mirror-Boozer-like diagnostics: `94%`.
- Free-boundary mirror lane: `99%` overall, with diagnostic scope documented.
- Straight-axis hybrid support fixture lane: `100%` for support-fixture scope.
- Toroidal stellarator-mirror hybrid lane: `97%`.
- ESSOS circular-coil mirror beta scan: `99%`.
- Public API/source simplification: `100%` for the mirror package initializer.
- PR merge readiness overall: `99%`.

### User input needed

No user input is needed.

---
## 194. Final Local Audit Checkpoint

### Steps taken

- Ran a final local audit pass after the M187-M193 plot-coverage and
  documentation synchronization tranches.
- Verified the git worktree was clean at the latest pushed head before the
  audit plan update.
- Checked that `plan_mirror.md` remains strictly ordered through M193.
- Checked the largest tracked files to confirm the recent work did not add
  generated result blobs or large binary plot outputs to the repository.
- Ran the combined mirror plus toroidal-hybrid local validation suite.

### Results obtained

- The combined local validation covers the mirror package tests and toroidal
  hybrid tests together after the added plotted smoke coverage.
- Repository-light audit showed the largest tracked files are pre-existing docs
  assets/plans and source files; the recent generated plots stayed in `/tmp`,
  pytest temporary directories, ignored `results/`, or ignored docs build
  output.
- The plan remains ordered and the branch is still a draft PR branch.

### How it was tested

Commands run:

```bash
git status -sb
git log --oneline -8
git ls-files -z | xargs -0 du -k | sort -nr | head -20
python - <<'PY'
from pathlib import Path
import re
text = Path('plan_mirror.md').read_text()
nums = [int(m.group(1)) for m in re.finditer(r'^## (\\d+)\\. ', text, flags=re.M)]
print('sections', len(nums), 'last', nums[-8:])
print('ordered', all(b == a + 1 for a, b in zip(nums, nums[1:])))
PY
JAX_ENABLE_X64=1 pytest tests/mirror tests/test_toroidal_hybrid.py -q
```

Results:

- Worktree was clean before this plan-only audit log.
- Plan ordering check reported sections ordered through M193.
- Largest tracked files were existing docs figures/plans/source files; no new
  generated result artifacts were tracked.
- Combined mirror plus toroidal-hybrid validation passed:
  `268 passed, 1 skipped in 277.69s`.

### File structure and best-practice notes

- This checkpoint is a plan-only audit entry.
- It does not change source, docs, tests, or tracked artifacts.
- The recent plot tests continue to use temporary directories and nonblank
  image checks instead of committed reference images.

### Best next steps

1. Commit and push M194.
2. Update the draft PR body with section 194 and the combined validation result.
3. Inspect only failed CI jobs after the push.
4. Leave the PR draft until GitHub checks finish cleanly and the user decides
   whether the explicitly deferred production lanes should remain deferred or
   be pursued in this branch.

### Completion percentages after M194

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `95%`.
- Fixed-boundary axisymmetric solve: `96%`.
- Residual Newton / preconditioning: `96%`.
- Two-coil and manufactured validation: `95%`.
- Finite-current pitch validation: `94%`.
- Plotting and `vmec --plot` mirror support: `99%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `97%`.
- Mirror-Boozer-like diagnostics: `94%`.
- Free-boundary mirror lane: `99%` overall, with diagnostic scope documented.
- Straight-axis hybrid support fixture lane: `100%` for support-fixture scope.
- Toroidal stellarator-mirror hybrid lane: `97%`.
- ESSOS circular-coil mirror beta scan: `99%`.
- Public API/source simplification: `100%` for the mirror package initializer.
- PR merge readiness overall: `99%`, pending GitHub checks and explicit review
  decision on deferred production lanes.

### User input needed

No user input is needed for the current draft scope.  A review decision is
needed before undrafting: keep the production free-boundary and target-resolution
hybrid convergence lanes deferred, or continue them in this PR.

---
## 195. Free-Boundary Reduced-LS Conditioning Diagnostics

### Steps taken

- Audited the remaining open production/free-boundary lane after the M194
  checkpoint and confirmed that the current circular-coil path is intentionally
  diagnostic, while the reduced residual-vector least-squares path is the right
  place to add solver-grade method evidence without overclaiming a production
  LCFS equilibrium solver.
- Added explicit least-squares diagnostics to
  `vmec_jax/mirror/free_boundary.py`:
  - Jacobian rank;
  - Jacobian nullity;
  - Jacobian condition number;
  - Jacobian singular values;
  - selected JAX differentiation mode after `jax_mode="auto"` resolution;
  - finite-difference step sizes;
  - predicted and actual residual-reduction fractions.
- Applied the same rank/nullity/conditioning diagnostics to the combined
  `MirrorFreeBoundaryLeastSquaresStep` used by the host-side coupled
  circular-coil bridge.
- Updated the repo-root vector-LS benchmark schema from
  `mirror_free_boundary_vector_ls_benchmark` version `0.2` to `0.3`, and
  exported the new diagnostics in both one-step rows and nonlinear solve rows.
- Added tests for full-rank finite-difference/JAX behavior, rank-deficient
  boundary parameterizations, and JAX automatic reverse-mode selection for a
  scalar-like residual.
- Updated `docs/mirror/readiness.rst`, `docs/mirror/overview.rst`, and
  `examples/mirror/README.md` so the documented derivative policy and
  free-boundary bridge status mention the new conditioning and backend-mode
  diagnostics.

### Results obtained

- Reduced free-boundary residual-vector solves now expose enough matrix
  diagnostics to catch rank-deficient or poorly conditioned boundary
  parameterizations before they are coupled to expensive fixed-boundary trial
  solves.
- The JAX `auto` derivative path now reports the concrete selected mode:
  forward for parameter counts no larger than residual length, and reverse for
  scalar-like or shorter residual vectors.
- The benchmark JSON now carries the same diagnostics that the code records,
  keeping plotted and non-plotted example outputs useful for method audits.
- The free-boundary circular-coil lane remains correctly labeled as diagnostic:
  this tranche improves method visibility and validation evidence, but does not
  claim a converged production free-boundary equilibrium solve.

### How it was tested

Commands run:

```bash
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_free_boundary.py::test_mirror_free_boundary_least_squares_step_reduces_linear_combined_residual \
  tests/mirror/test_mirror_free_boundary.py::test_mirror_free_boundary_residual_vector_least_squares_step_supports_fd_and_jax_backends \
  tests/mirror/test_mirror_free_boundary.py::test_mirror_free_boundary_residual_vector_least_squares_step_reports_rank_deficiency \
  tests/mirror/test_mirror_free_boundary.py::test_mirror_free_boundary_residual_vector_least_squares_step_auto_uses_reverse_for_small_residual \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_vector_ls_benchmark_runs_without_plots -q
python -m ruff format \
  vmec_jax/mirror/free_boundary.py \
  examples/mirror_free_boundary_vector_ls_benchmark.py \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py
python -m ruff check \
  vmec_jax/mirror/free_boundary.py \
  examples/mirror_free_boundary_vector_ls_benchmark.py \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_vector_ls_benchmark_runs_without_plots \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_vector_ls_benchmark_writes_nonblank_plots -q
python -m sphinx -W -b html docs docs/_build/html
git diff --check
JAX_ENABLE_X64=1 pytest tests/mirror tests/test_toroidal_hybrid.py -q
```

Results:

- Focused new diagnostic tests passed: `5 passed in 2.42s`.
- Ruff formatting made no changes.
- Ruff lint passed.
- Free-boundary plus vector-LS example coverage passed:
  `120 passed in 8.02s`.
- Sphinx docs build passed with warnings as errors.
- Whitespace check passed.
- Combined mirror plus toroidal-hybrid validation passed:
  `270 passed, 1 skipped in 280.30s`.

### File structure and best-practice notes

- The implementation stays in the existing compact free-boundary bridge module
  instead of introducing a new solver package before the coupled LCFS residual
  path is promoted.
- The public dataclasses are extended append-only, preserving existing call
  sites while making solver diagnostics available to examples and downstream
  scripts.
- The root benchmark remains the executable audit artifact for reduced
  free-boundary derivative backends; generated plots remain in temporary or
  ignored output directories.
- Documentation keeps the line between research-grade fixed-boundary pieces,
  validated reduced free-boundary methods, and deferred production
  free-boundary equilibrium solves explicit.

### Best next steps

1. Commit and push M195.
2. Update the draft PR body with section 195 and the `270 passed, 1 skipped`
   validation result.
3. Inspect only failed CI jobs after the push.
4. Continue the remaining finite lanes from the readiness matrix:
   production free-boundary LCFS promotion, target-resolution toroidal hybrid
   convergence, and final review decision on explicitly deferred production
   scopes.

### Completion percentages after M195

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `95%`.
- Fixed-boundary axisymmetric solve: `96%`.
- Residual Newton / preconditioning: `96%`.
- Two-coil and manufactured validation: `95%`.
- Finite-current pitch validation: `94%`.
- Plotting and `vmec --plot` mirror support: `99%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `97%`.
- Mirror-Boozer-like diagnostics: `94%`.
- Free-boundary mirror lane: `99.2%` overall for the current diagnostic/reduced
  solver scope, with production LCFS convergence still explicitly deferred.
- Straight-axis hybrid support fixture lane: `100%` for support-fixture scope.
- Toroidal stellarator-mirror hybrid lane: `97%`, pending target-resolution
  convergence promotion.
- ESSOS circular-coil mirror beta scan: `99%`.
- Public API/source simplification: `100%` for the current mirror package
  structure.
- PR merge readiness overall: `99.2%`, pending GitHub checks and explicit
  review decision on deferred production lanes.

### User input needed

No user input is needed for the next implementation lane.  A review decision is
still needed before undrafting: keep the production free-boundary and
target-resolution hybrid convergence lanes deferred, or continue them in this
PR.

---
## 196. Free-Boundary Adaptive Ridge Candidate Selection

### Steps taken

- Continued the free-boundary production-readiness lane after M195 by turning
  the new least-squares conditioning diagnostics into a small robustness
  improvement.
- Added optional `ridge_candidates` support to the reduced residual-vector
  least-squares step and solve loop.
- Added the same optional `ridge_candidates` support to the combined
  equilibrium-plus-LCFS least-squares step and the guarded free-boundary loop.
- Kept the existing scalar `ridge` API as the default path.  When
  `ridge_candidates` is provided, each candidate solves a Tikhonov-augmented
  linearized least-squares system, applies the usual damping and line search,
  and keeps the best accepted trial.  Ties prefer the smaller ridge.
- Extended step diagnostics and the repo-root vector-LS benchmark schema to
  record the selected ridge and tried ridge candidates.
- Added tests showing that an unregularized nonlinear LS update can be rejected
  while an adaptive nonzero ridge candidate accepts a smaller residual-reducing
  step.
- Updated the readiness docs, mirror overview, and examples README to describe
  adaptive ridge candidates as part of the reduced/diagnostic free-boundary
  solver path.

### Results obtained

- Reduced and combined free-boundary LS helpers can now recover from an
  over-aggressive unregularized Newton-like step by selecting a regularized
  ridge candidate.
- The selected ridge is visible in the dataclass result, nonlinear solve rows,
  and vector-LS benchmark JSON.
- The benchmark schema is now `mirror_free_boundary_vector_ls_benchmark`
  version `0.4`.
- The circular-coil free-boundary lane remains correctly scoped as diagnostic:
  this improves reduced solver robustness and method auditability, but does not
  claim a production converged LCFS free-boundary equilibrium solve.

### How it was tested

Commands run:

```bash
python -m ruff format \
  vmec_jax/mirror/free_boundary.py \
  examples/mirror_free_boundary_vector_ls_benchmark.py \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py
python -m py_compile \
  vmec_jax/mirror/free_boundary.py \
  examples/mirror_free_boundary_vector_ls_benchmark.py
python -m ruff check \
  vmec_jax/mirror/free_boundary.py \
  examples/mirror_free_boundary_vector_ls_benchmark.py \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_free_boundary.py::test_mirror_free_boundary_least_squares_step_selects_adaptive_ridge_candidate \
  tests/mirror/test_mirror_free_boundary.py::test_mirror_free_boundary_residual_vector_least_squares_step_selects_adaptive_ridge_candidate \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_vector_ls_benchmark_runs_without_plots -q
JAX_ENABLE_X64=1 pytest \
  tests/mirror/test_mirror_free_boundary.py \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_vector_ls_benchmark_runs_without_plots \
  tests/mirror/test_mirror_examples.py::test_root_free_boundary_vector_ls_benchmark_writes_nonblank_plots -q
python -m sphinx -W -b html docs docs/_build/html
git diff --check
JAX_ENABLE_X64=1 pytest tests/mirror tests/test_toroidal_hybrid.py -q
```

Results:

- Ruff formatting reformatted `vmec_jax/mirror/free_boundary.py`.
- Syntax compilation passed.
- Ruff lint passed.
- Focused adaptive-ridge tests passed: `3 passed in 2.25s`.
- Free-boundary plus vector-LS example coverage passed:
  `126 passed in 7.92s`.
- Sphinx docs build passed with warnings as errors.
- Whitespace check passed.
- Combined mirror plus toroidal-hybrid validation passed:
  `276 passed, 1 skipped in 279.23s`.

### File structure and best-practice notes

- The adaptive ridge logic stays inside the existing free-boundary bridge
  module because it is shared by the reduced vector path, the combined
  residual path, and the guarded loop.  A package split should wait until the
  coupled LCFS residual is promoted beyond diagnostic scope.
- The public dataclasses are extended append-only with `ridge_candidates`,
  preserving existing call sites while recording the selected regularization
  policy.
- The benchmark schema carries method diagnostics instead of committed result
  images; generated plots still live in temporary or ignored output trees.

### Best next steps

1. Commit and push M196.
2. Update the draft PR body with section 196 and the `276 passed, 1 skipped`
   validation result.
3. Inspect only failed CI jobs after the push.
4. Continue with the remaining finite lanes: either promote the circular-coil
   bridge toward a true coupled LCFS residual solve, or move to the
   target-resolution toroidal hybrid convergence ladder.

### Completion percentages after M196

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `95%`.
- Fixed-boundary axisymmetric solve: `96%`.
- Residual Newton / preconditioning: `96%`.
- Two-coil and manufactured validation: `95%`.
- Finite-current pitch validation: `94%`.
- Plotting and `vmec --plot` mirror support: `99%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `97%`.
- Mirror-Boozer-like diagnostics: `94%`.
- Free-boundary mirror lane: `99.3%` overall for the current diagnostic/reduced
  solver scope, with production LCFS convergence still explicitly deferred.
- Straight-axis hybrid support fixture lane: `100%` for support-fixture scope.
- Toroidal stellarator-mirror hybrid lane: `97%`, pending target-resolution
  convergence promotion.
- ESSOS circular-coil mirror beta scan: `99%`.
- Public API/source simplification: `100%` for the current mirror package
  structure.
- PR merge readiness overall: `99.3%`, pending GitHub checks and explicit
  review decision on deferred production lanes.

### User input needed

No user input is needed for the next implementation lane.  A review decision is
still needed before undrafting: keep the production free-boundary and
target-resolution hybrid convergence lanes deferred, or continue them in this
PR.
---
## 197. Toroidal Hybrid Target Resolution Presets

### Steps taken

- Audited the toroidal stellarator-mirror hybrid convergence runner after the
  adaptive free-boundary LS tranche.
- Added named `--resolution-preset` choices to
  `examples/toroidal_stellarator_mirror_hybrid_convergence.py`:
  - `manual`: preserve the existing `--ns-array` and `--mode-pairs` behavior;
  - `smoke`: low-cost no-solve ladder for geometry/plotting checks;
  - `promotion`: moderate no-solve ladder before expensive solved rows;
  - `target`: target no-solve ladder for the final solved/parity convergence
    campaign.
- Defined the current target ladder as
  `ns = 7,9,15` and `mpol:ntor = 5:20,6:24`.
- Added row and summary metadata:
  - `resolution_preset`;
  - `target_resolution_ladder`;
  - `target_resolution_promotion_claim`.
- Kept `target_resolution_promotion_claim` false.  The preset creates the
  finite target ladder, but production promotion still requires solved/parity
  evidence over that ladder.
- Added a no-solve target-preset test that verifies row count, `ns` values,
  mode pairs, CSV metadata, exact boundary fit, and no production claim.
- Updated `examples/mirror/README.md`, `docs/mirror/readiness.rst`, and
  `docs/mirror/overview.rst` to point to the named target ladder.

### Results obtained

- The toroidal hybrid target-resolution lane now has an explicit, runnable,
  finite ladder rather than only a prose TODO.
- Downstream scripts can distinguish ordinary manual scans from target-ladder
  inputs and can see that target-resolution promotion remains unclaimed.
- The existing manual/default behavior remains covered and unchanged for
  current no-solve tests.

### How it was tested

Commands run:

```bash
python -m ruff format \
  examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  tests/test_toroidal_hybrid.py
python -m py_compile examples/toroidal_stellarator_mirror_hybrid_convergence.py
python -m ruff check \
  examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  tests/test_toroidal_hybrid.py
JAX_ENABLE_X64=1 pytest \
  tests/test_toroidal_hybrid.py::test_toroidal_hybrid_convergence_example_runs_without_solve \
  tests/test_toroidal_hybrid.py::test_toroidal_hybrid_convergence_example_target_preset_without_solve \
  tests/test_toroidal_hybrid.py::test_toroidal_hybrid_convergence_example_writes_nonblank_no_solve_plots -q
JAX_ENABLE_X64=1 pytest tests/test_toroidal_hybrid.py -q
python -m sphinx -W -b html docs docs/_build/html
git diff --check
```

Results:

- Ruff formatting made no changes.
- Syntax compilation passed.
- Ruff lint passed.
- Focused preset tests passed: `3 passed in 3.38s`.
- Full toroidal-hybrid tests passed: `29 passed in 7.45s`.
- Sphinx docs build passed with warnings as errors.
- Whitespace check passed.

### File structure and best-practice notes

- The preset metadata lives beside the convergence runner because it is runner
  configuration, not a new physics model.
- The CSV/JSON schema gets explicit target-ladder booleans instead of relying
  on file names or comments.
- The target preset is no-solve by default, so it adds no heavy generated
  artifacts and keeps final solved/parity runs opt-in.

### Best next steps

1. Commit and push M197.
2. Update the draft PR body with section 197 and the toroidal-hybrid validation
   result.
3. Inspect only failed CI jobs after the push.
4. Use the `target` preset for the next solved/parity campaign, ideally on the
   office GPU/VMEC2000 environment, and record convergence/residual trends
   without committing generated result trees.

### Completion percentages after M197

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `95%`.
- Fixed-boundary axisymmetric solve: `96%`.
- Residual Newton / preconditioning: `96%`.
- Two-coil and manufactured validation: `95%`.
- Finite-current pitch validation: `94%`.
- Plotting and `vmec --plot` mirror support: `99%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `97%`.
- Mirror-Boozer-like diagnostics: `94%`.
- Free-boundary mirror lane: `99.3%` overall for the current diagnostic/reduced
  solver scope, with production LCFS convergence still explicitly deferred.
- Straight-axis hybrid support fixture lane: `100%` for support-fixture scope.
- Toroidal stellarator-mirror hybrid lane: `97.5%`, with a finite target
  ladder now explicit and solved/parity target-ladder evidence still pending.
- ESSOS circular-coil mirror beta scan: `99%`.
- Public API/source simplification: `100%` for the current mirror package
  structure.
- PR merge readiness overall: `99.3%`, pending GitHub checks and explicit
  review decision on deferred production lanes.

### User input needed

No user input is needed for local implementation.  Access to the office
GPU/VMEC2000 environment will be useful for the target-preset solved/parity
campaign.

---
## 198. Toroidal Hybrid Target Campaign Case Filtering

### Steps taken

- Continued the toroidal hybrid target-resolution lane after adding the named
  target preset in M197.
- Added `--case-filter` to
  `examples/toroidal_stellarator_mirror_hybrid_convergence.py`.
- The filter accepts comma-separated shell patterns matched against generated
  case names, for example `*ns015*`.
- Added helper functions for parsing and matching case filters.
- Added a guard that raises an error when filters select no rows, avoiding
  empty target-campaign reports.
- Added top-level summary metadata `case_filters` so filtered target runs are
  auditable.
- Added a filtered target-preset test that verifies `*ns015*` selects the two
  target rows with `mpol:ntor = 5:20,6:24`.
- Updated the examples README and mirror overview to document case filtering
  for splitting the target campaign across machines.

### Results obtained

- The target-resolution campaign can now be partitioned without editing the
  runner or committing generated result trees.
- The no-solve target ladder remains explicit, and filtered reports preserve
  enough metadata to reconstruct which subset was run.
- The target-resolution production claim remains false until solved/parity
  evidence is produced over the target ladder.

### How it was tested

Commands run:

```bash
python -m ruff format \
  examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  tests/test_toroidal_hybrid.py
python -m py_compile examples/toroidal_stellarator_mirror_hybrid_convergence.py
python -m ruff check \
  examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  tests/test_toroidal_hybrid.py
JAX_ENABLE_X64=1 pytest \
  tests/test_toroidal_hybrid.py::test_toroidal_hybrid_convergence_example_target_preset_without_solve \
  tests/test_toroidal_hybrid.py::test_toroidal_hybrid_convergence_example_filters_target_preset_cases \
  tests/test_toroidal_hybrid.py::test_toroidal_hybrid_convergence_history_summary_uses_iteration_labels -q
JAX_ENABLE_X64=1 pytest tests/test_toroidal_hybrid.py -q
python -m sphinx -W -b html docs docs/_build/html
git diff --check
```

Results:

- Ruff formatting made no changes.
- Syntax compilation passed.
- Ruff lint passed.
- Focused target/filter tests passed: `3 passed in 2.40s`.
- Full toroidal-hybrid tests passed: `30 passed in 8.71s`.
- Sphinx docs build passed with warnings as errors.
- Whitespace check passed.

### File structure and best-practice notes

- Case filtering stays in the root convergence runner because it is campaign
  orchestration, not core physics code.
- The filter uses standard shell-pattern matching on generated case names,
  keeping scripts simple and easy to reproduce in terminal logs.
- The runner still writes results only under user-selected output directories;
  no generated outputs or figures are tracked.

### Best next steps

1. Commit and push M198.
2. Update the draft PR body with section 198 and the `30 passed` toroidal
   validation result.
3. Inspect only failed CI jobs after the push.
4. Use `--resolution-preset target --case-filter ... --run-solve` on the
   office VMEC2000/GPU environment to start collecting solved/parity rows.

### Completion percentages after M198

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `95%`.
- Fixed-boundary axisymmetric solve: `96%`.
- Residual Newton / preconditioning: `96%`.
- Two-coil and manufactured validation: `95%`.
- Finite-current pitch validation: `94%`.
- Plotting and `vmec --plot` mirror support: `99%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `97%`.
- Mirror-Boozer-like diagnostics: `94%`.
- Free-boundary mirror lane: `99.3%` overall for the current diagnostic/reduced
  solver scope, with production LCFS convergence still explicitly deferred.
- Straight-axis hybrid support fixture lane: `100%` for support-fixture scope.
- Toroidal stellarator-mirror hybrid lane: `98%`, with target-ladder and
  target-campaign partitioning now explicit and solved/parity target evidence
  still pending.
- ESSOS circular-coil mirror beta scan: `99%`.
- Public API/source simplification: `100%` for the current mirror package
  structure.
- PR merge readiness overall: `99.4%`, pending GitHub checks and explicit
  review decision on deferred production lanes.

### User input needed

No user input is needed for local implementation.  Office VMEC2000/GPU access
is the next useful external resource for target-ladder solved/parity runs.

---
## 199. Office GPU Toroidal Hybrid Target Probe

### Steps taken

- Probed the `office` SSH environment non-interactively.
- Confirmed the host is reachable and has two NVIDIA RTX A4000 GPUs visible to
  JAX through `/home/rjorge/venvs/vmec_jax_gpu/bin/python`.
- Confirmed the VMEC2000 executable path:
  `/home/rjorge/vmec2000/_skbuild/linux-x86_64-3.11/cmake-install/bin/xvmec`.
- Created a fresh shallow clone of the draft PR branch at
  `/home/rjorge/local/vmec_mirror` to avoid touching dirty existing remote
  worktrees.
- Ran the target preset in no-solve filtered mode for `*ns015*`.
- Ran a one-row target-ladder VMEC/JAX solved smoke for
  `*ns007_mpol05_ntor20` with `max_iter=1`, `NITER_ARRAY=5`, `NSTEP=1`,
  and no plots.
- Ran the same one-row target-ladder probe with VMEC2000 enabled and a
  60-second VMEC2000 timeout.

### Results obtained

- Remote branch checkout is clean at PR head `eca789b`.
- JAX sees two CUDA devices on office.
- The no-solve target filtered command wrote:
  `/home/rjorge/local/vmec_mirror/results/toroidal_hybrid_target_probe_no_solve/toroidal_stellarator_mirror_hybrid_convergence.json`.
- The one-row VMEC/JAX solved smoke wrote:
  `/home/rjorge/local/vmec_mirror/results/toroidal_hybrid_target_probe_solve/toroidal_stellarator_mirror_hybrid_convergence.json`.
- Compact VMEC/JAX solved-smoke metrics:
  - case: `ns007_mpol05_ntor20`;
  - `ran_solve=True`;
  - `target_resolution_ladder=True`;
  - `target_resolution_promotion_claim=False`;
  - solve seconds: `26.041028084233403`;
  - `n_iter=1`;
  - `initial_fsq=0.01730447046657486`;
  - `final_fsq=0.01730447046657486`;
  - `direct_initial_fsq=0.10843912042391801`;
  - `initial_fsq_ratio_direct_initial=0.15957774647126405`;
  - converged flags all false.
- The one-row VMEC/JAX plus VMEC2000 parity probe wrote:
  `/home/rjorge/local/vmec_mirror/results/toroidal_hybrid_target_probe_vmec2000/toroidal_stellarator_mirror_hybrid_convergence.json`.
- Compact parity-probe metrics:
  - case: `ns007_mpol05_ntor20`;
  - `ran_solve=True`;
  - `ran_vmec2000=True`;
  - VMEC/JAX solve seconds: `4.816137593239546`;
  - VMEC/JAX `n_iter=1`;
  - VMEC/JAX `initial_fsq=0.01730447046657486`;
  - VMEC/JAX `final_fsq=0.01730447046657486`;
  - VMEC2000 return code: `0`;
  - VMEC2000 runtime seconds: `1.5839644763618708`;
  - VMEC2000 parsed rows: `5`;
  - VMEC2000 initial total `fsq=0.1078`;
  - VMEC2000 final total `fsq=0.041170000000000005`;
  - `initial_fsq_ratio_vmec2000=0.16052384477342171`;
  - `direct_initial_fsq_ratio_vmec2000=1.0059287608897773`;
  - `vmec2000_error=None`.

### Interpretation

- The office target-ladder workflow is viable: the branch runs from a fresh
  clone, JAX uses GPU devices, and VMEC2000 runs successfully through the
  convergence runner.
- The direct-initial VMEC/JAX residual agrees with the VMEC2000 first row to
  about `0.6%` on this target-ladder probe, matching the intended parity-audit
  signal.
- This is not a target-resolution convergence claim.  It is a one-row,
  one-iteration feasibility probe that verifies campaign plumbing and records
  initial parity behavior.

### How it was tested

Commands run on office through SSH:

```bash
ssh office 'hostname; nvidia-smi --query-gpu=name --format=csv,noheader'
ssh office 'git clone --depth=1 --branch codex/mirror-geometry https://github.com/uwplasma/vmec_jax.git /home/rjorge/local/vmec_mirror'
ssh office 'cd /home/rjorge/local/vmec_mirror && PYTHONPATH=$PWD /home/rjorge/venvs/vmec_jax_gpu/bin/python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir /home/rjorge/local/vmec_mirror/results/toroidal_hybrid_target_probe_no_solve \
  --resolution-preset target \
  --case-filter "*ns015*" \
  --ntheta-fit 64 \
  --nzeta-fit 64 \
  --no-plots'
ssh office 'cd /home/rjorge/local/vmec_mirror && PYTHONPATH=$PWD /home/rjorge/venvs/vmec_jax_gpu/bin/python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir /home/rjorge/local/vmec_mirror/results/toroidal_hybrid_target_probe_solve \
  --resolution-preset target \
  --case-filter "*ns007_mpol05_ntor20" \
  --ntheta-fit 64 \
  --nzeta-fit 64 \
  --run-solve \
  --max-iter 1 \
  --niter 5 \
  --nstep 1 \
  --ftol 1e-6 \
  --no-plots'
ssh office 'cd /home/rjorge/local/vmec_mirror && PYTHONPATH=$PWD /home/rjorge/venvs/vmec_jax_gpu/bin/python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir /home/rjorge/local/vmec_mirror/results/toroidal_hybrid_target_probe_vmec2000 \
  --resolution-preset target \
  --case-filter "*ns007_mpol05_ntor20" \
  --ntheta-fit 64 \
  --nzeta-fit 64 \
  --run-solve \
  --max-iter 1 \
  --niter 5 \
  --nstep 1 \
  --ftol 1e-6 \
  --run-vmec2000 \
  --vmec2000-exec /home/rjorge/vmec2000/_skbuild/linux-x86_64-3.11/cmake-install/bin/xvmec \
  --vmec2000-timeout-s 60 \
  --no-plots'
```

All three runner commands completed successfully.  Generated results stayed on
the office host under ignored `results/` directories and were not copied into
the repository.

### File structure and best-practice notes

- This is a plan-only evidence checkpoint.
- The remote clone is separate from existing dirty worktrees.
- The local repository remains light: no remote result JSON, WOUT, threed1, or
  plot outputs were committed.

### Best next steps

1. Commit and push M199.
2. Update the draft PR body with section 199 and the one-row office parity
   probe result.
3. Inspect only failed CI jobs after the push.
4. Run the full target preset on office in filtered chunks, starting with:
   `--resolution-preset target --case-filter '*ns007*' --run-solve
   --run-vmec2000 --nstep 1 --full-solver-diagnostics`, then proceed through
   `*ns009*` and `*ns015*` if runtime and disk remain acceptable.

### Completion percentages after M199

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `95%`.
- Fixed-boundary axisymmetric solve: `96%`.
- Residual Newton / preconditioning: `96%`.
- Two-coil and manufactured validation: `95%`.
- Finite-current pitch validation: `94%`.
- Plotting and `vmec --plot` mirror support: `99%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `97%`.
- Mirror-Boozer-like diagnostics: `94%`.
- Free-boundary mirror lane: `99.3%` overall for the current diagnostic/reduced
  solver scope, with production LCFS convergence still explicitly deferred.
- Straight-axis hybrid support fixture lane: `100%` for support-fixture scope.
- Toroidal stellarator-mirror hybrid lane: `98.2%`, with target-ladder
  workflow, filtering, and one-row solved/parity feasibility verified; full
  target-ladder convergence evidence remains pending.
- ESSOS circular-coil mirror beta scan: `99%`.
- Public API/source simplification: `100%` for the current mirror package
  structure.
- PR merge readiness overall: `99.4%`, pending GitHub checks and explicit
  review decision on deferred production lanes.

### User input needed

No user input is needed for the next remote target-ladder chunks, but disk
space on office is tight (`/home` was 99% used during the probe), so large
plotted or full-history campaigns should be run in filtered chunks and cleaned
up after extracting compact metrics.

---
## 200. Office GPU Target-Ladder Six-Row Parity Audit

### Steps taken

- Continued the M199 office target-ladder probe into the full named target
  ladder, still using filtered chunks to control disk/runtime.
- Ran three filtered target chunks on office:
  - `--case-filter "*ns007*"`;
  - `--case-filter "*ns009*"`;
  - `--case-filter "*ns015*"`.
- Each chunk used:
  - `--resolution-preset target`;
  - `--run-solve`;
  - `--run-vmec2000`;
  - `--max-iter 3`;
  - `--niter 25`;
  - `--nstep 1`;
  - `--ftol 1e-6`;
  - `--no-plots`.
- Extracted compact JSON metrics over SSH and left all generated outputs on
  office under ignored `results/` directories.
- Checked the remote result-tree sizes and available disk after the runs.

### Results obtained

All six target-ladder rows completed with VMEC2000 return code `0`.

| case | VMEC/JAX final fsq | VMEC/JAX best fsq | VMEC2000 initial fsq | VMEC2000 final fsq | direct-initial / VMEC2000 initial |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ns007_mpol05_ntor20` | `0.03769366816533995` | `0.03325359464186601` | `0.1078` | `0.00797` | `1.0059287608897773` |
| `ns007_mpol06_ntor24` | `0.06834814572298939` | `0.03997196508976027` | `0.4521` | `0.01985` | `1.0035103011877164` |
| `ns009_mpol05_ntor20` | `0.04212371799593974` | `0.0373280603814346` | `0.11149999999999999` | `0.008560000000000002` | `1.006208103367928` |
| `ns009_mpol06_ntor24` | `0.07288955131418487` | `0.043329639696754574` | `0.46709999999999996` | `0.020710000000000003` | `1.002122976587009` |
| `ns015_mpol05_ntor20` | `0.056153955478579064` | `0.023020570375674353` | `0.1162` | `0.010620000000000001` | `1.006374048260763` |
| `ns015_mpol06_ntor24` | `0.12092504641171972` | `0.06708267139943355` | `0.48469999999999996` | `0.02212` | `1.0035506240728607` |

Additional observations:

- VMEC/JAX ran exactly `3` iterations for each row and did not converge under
  these intentionally short settings.
- VMEC2000 parsed `25` rows for every row because `NSTEP=1` and `NITER_ARRAY=25`.
- VMEC2000 final fsq values were lower than the three-iteration VMEC/JAX final
  fsq values in this short campaign.
- Direct-initial VMEC/JAX residuals agree with the VMEC2000 initial row to
  roughly `0.2%` to `0.6%` across the target ladder, which is a strong
  initialization/parity signal.
- Result directories on office are compact:
  - `ns007` chunk: `1.6M`;
  - `ns009` chunk: `1.9M`;
  - `ns015` chunk: `3.1M`.
- Office `/home` remained tight but usable: about `15G` free after the runs.

### Interpretation

- The full target-ladder campaign plumbing is now verified: every target row
  runs through VMEC/JAX and VMEC2000 from the draft branch on office.
- The direct-initial parity signal remains consistent across both target mode
  pairs and all three `ns` levels.
- This is still not a target-resolution convergence claim.  The VMEC/JAX side
  was deliberately limited to three iterations, so the next campaign needs
  larger iteration budgets and should inspect convergence trajectories rather
  than only final scalars.

### How it was tested

Representative command pattern run on office:

```bash
ssh office 'cd /home/rjorge/local/vmec_mirror && PYTHONPATH=$PWD /home/rjorge/venvs/vmec_jax_gpu/bin/python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir /home/rjorge/local/vmec_mirror/results/toroidal_hybrid_target_ns007_chunk_vmec2000 \
  --resolution-preset target \
  --case-filter "*ns007*" \
  --ntheta-fit 64 \
  --nzeta-fit 64 \
  --run-solve \
  --max-iter 3 \
  --niter 25 \
  --nstep 1 \
  --ftol 1e-6 \
  --run-vmec2000 \
  --vmec2000-exec /home/rjorge/vmec2000/_skbuild/linux-x86_64-3.11/cmake-install/bin/xvmec \
  --vmec2000-timeout-s 120 \
  --no-plots'
```

The same command was repeated with `*ns009*` and `*ns015*` filters and
matching output directories.  Compact metrics were extracted from the resulting
JSON files.  No generated result files were copied into the git repository.

### File structure and best-practice notes

- This is a plan-only evidence checkpoint.
- Remote generated outputs stay under:
  `/home/rjorge/local/vmec_mirror/results/toroidal_hybrid_target_*_chunk_vmec2000`.
- No local source, test, docs, or tracked artifact changes were required for
  this evidence run.

### Best next steps

1. Commit and push M200.
2. Update the draft PR body with section 200 and the six-row target-ladder
   audit result.
3. Inspect only failed CI jobs after the push.
4. Run a second target campaign with larger VMEC/JAX iteration budgets and
   `--full-solver-diagnostics` for at least the two `mpol:ntor=5:20` rows, then
   decide whether solver settings or initialization policy need adjustment
   before running all six rows to a strict convergence target.

### Completion percentages after M200

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `95%`.
- Fixed-boundary axisymmetric solve: `96%`.
- Residual Newton / preconditioning: `96%`.
- Two-coil and manufactured validation: `95%`.
- Finite-current pitch validation: `94%`.
- Plotting and `vmec --plot` mirror support: `99%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `97%`.
- Mirror-Boozer-like diagnostics: `94%`.
- Free-boundary mirror lane: `99.3%` overall for the current diagnostic/reduced
  solver scope, with production LCFS convergence still explicitly deferred.
- Straight-axis hybrid support fixture lane: `100%` for support-fixture scope.
- Toroidal stellarator-mirror hybrid lane: `98.5%`, with target-ladder
  VMEC/JAX and VMEC2000 execution verified for all six rows; full
  target-ladder convergence evidence remains pending.
- ESSOS circular-coil mirror beta scan: `99%`.
- Public API/source simplification: `100%` for the current mirror package
  structure.
- PR merge readiness overall: `99.5%`, pending GitHub checks and explicit
  review decision on deferred production lanes.

### User input needed

No user input is needed for the next remote campaign.  Disk space on office is
still tight, so the next runs should keep `--no-plots` unless plots are needed
for a specific diagnostic row.

---
## 201. Compact Aggregation for Split Toroidal-Hybrid Target Campaigns

### Steps taken

- Added a compact aggregation mode to the root convergence runner:
  `examples/toroidal_stellarator_mirror_hybrid_convergence.py --aggregate-json`.
- Reused the existing row CSV writer and plot writers instead of introducing a
  separate postprocessing script or new public package surface.
- Added aggregate metadata:
  - schema label `toroidal_stellarator_mirror_hybrid_convergence_aggregate.v1`;
  - source JSON list and per-source row summaries;
  - duplicate-case replacement tracking;
  - aggregate row counts, VMEC2000 return-code counts, convergence counts, and
    residual-range metrics;
  - `aggregate_source_json` row provenance, including in the compact CSV.
- Updated `examples/mirror/README.md` and `docs/mirror/overview.rst` with the
  new split-campaign aggregation workflow.
- Added a regression test that builds two synthetic target chunks, replaces one
  duplicate case from the later chunk, verifies sorted aggregate rows, verifies
  compact metrics, and checks CSV provenance.
- Pulled the pushed branch on `office` and aggregated the existing three
  six-row target chunks without plots.

### Results obtained

Local implementation result:

- The convergence example can now merge existing chunk JSONs without rerunning
  VMEC/JAX or VMEC2000.
- The same code path can regenerate plots from stored row histories when plots
  are requested, but target campaign aggregation can stay very small with
  `--no-plots`.

Office target aggregate result:

- Aggregate JSON:
  `/home/rjorge/local/vmec_mirror/results/toroidal_hybrid_target_aggregate_m201/toroidal_stellarator_mirror_hybrid_convergence_aggregate.json`
- Aggregate CSV:
  `/home/rjorge/local/vmec_mirror/results/toroidal_hybrid_target_aggregate_m201/toroidal_stellarator_mirror_hybrid_convergence.csv`
- Result-tree size: `92K`.
- Source JSON count: `3`.
- Case count: `6`.
- Duplicate cases replaced: none.
- VMEC/JAX solved rows: `6`.
- VMEC2000 rows: `6`.
- VMEC2000 return-code-zero rows: `6`.
- VMEC/JAX converged rows: `0`.
- VMEC/JAX strict-converged rows: `0`.
- Direct-initial VMEC/JAX / VMEC2000 initial `fsq` ratio range:
  `1.002122976587009` to `1.006374048260763`.
- VMEC/JAX best `fsq` range:
  `0.023020570375674353` to `0.06708267139943355`.
- VMEC/JAX final `fsq` range:
  `0.03769366816533995` to `0.12092504641171972`.
- VMEC2000 final `fsq` range:
  `0.00797` to `0.02212`.
- Office `/home` remained tight at about `15G` free, so the compact aggregate
  path is useful for continued evidence gathering.

### How it was tested

Local checks:

```bash
python -m ruff check examples/toroidal_stellarator_mirror_hybrid_convergence.py tests/test_toroidal_hybrid.py
JAX_ENABLE_X64=1 pytest tests/test_toroidal_hybrid.py::test_toroidal_hybrid_convergence_example_aggregates_chunk_jsons -q
JAX_ENABLE_X64=1 pytest tests/test_toroidal_hybrid.py -q
python -m sphinx -W -b html docs docs/_build/html
git diff --check
```

Results:

- Ruff passed.
- Focused aggregate regression passed: `1 passed in 0.79s`.
- Full toroidal-hybrid tests passed: `31 passed in 9.36s`.
- Sphinx docs build passed with warnings as errors.
- Whitespace check passed.

Remote aggregation command:

```bash
ssh office 'cd /home/rjorge/local/vmec_mirror && PYTHONPATH=$PWD /home/rjorge/venvs/vmec_jax_gpu/bin/python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir /home/rjorge/local/vmec_mirror/results/toroidal_hybrid_target_aggregate_m201 \
  --aggregate-json \
    /home/rjorge/local/vmec_mirror/results/toroidal_hybrid_target_ns007_chunk_vmec2000/toroidal_stellarator_mirror_hybrid_convergence.json \
    /home/rjorge/local/vmec_mirror/results/toroidal_hybrid_target_ns009_chunk_vmec2000/toroidal_stellarator_mirror_hybrid_convergence.json \
    /home/rjorge/local/vmec_mirror/results/toroidal_hybrid_target_ns015_chunk_vmec2000/toroidal_stellarator_mirror_hybrid_convergence.json \
  --no-plots'
```

### File structure and best-practice notes

- The aggregation mode lives in the existing convergence example because it is
  an example/reporting workflow, not a new equilibrium algorithm.
- Existing CSV/plot helpers are reused, reducing duplicate plotting and row
  serialization code.
- The source package API remains unchanged; no new public import surface was
  added for a campaign-specific postprocessor.
- Tests live with the rest of the toroidal-hybrid tests in
  `tests/test_toroidal_hybrid.py`.
- Documentation updates are in the existing mirror overview and examples README
  where users already learn the target-ladder workflow.
- Generated remote outputs stay under ignored `results/` directories and were
  not copied into the repository.

### Best next steps

1. Commit and push this M201 plan entry and update the draft PR body.
2. Inspect only failed CI jobs after the push.
3. Run a larger-iteration target diagnostic campaign for the `mpol:ntor=5:20`
   target rows with `--full-solver-diagnostics`, `--nstep 1`, and `--no-plots`.
4. Aggregate that larger campaign with the new `--aggregate-json` path and use
   the step-status, restart-reason, and `dt_eff` histories to decide whether
   the remaining gap to VMEC2000 is iteration budget, initialization, line-search
   policy, or solver/preconditioner behavior.

### Completion percentages after M201

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `95%`.
- Fixed-boundary axisymmetric solve: `96%`.
- Residual Newton / preconditioning: `96%`.
- Two-coil and manufactured validation: `95%`.
- Finite-current pitch validation: `94%`.
- Plotting and `vmec --plot` mirror support: `99%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `97%`.
- Mirror-Boozer-like diagnostics: `94%`.
- Free-boundary mirror lane: `99.3%` overall for the current diagnostic/reduced
  solver scope, with production LCFS convergence still explicitly deferred.
- Straight-axis hybrid support fixture lane: `100%` for support-fixture scope.
- Toroidal stellarator-mirror hybrid lane: `98.7%`, with split target-campaign
  aggregation now repeatable and office six-row target evidence summarized in a
  compact artifact; full target-ladder convergence evidence remains pending.
- ESSOS circular-coil mirror beta scan: `99%`.
- Public API/source simplification: `100%` for the current mirror package
  structure.
- PR merge readiness overall: `99.5%`, pending GitHub checks and explicit
  review decision on deferred production lanes.

### User input needed

No user input is needed.  The next campaign should remain plot-free until a
specific diagnostic row needs figures, because office disk space is still tight.

---
## 202. Target-Ladder 80-Iteration Convergence and Solver Reporting Audit

### Steps taken

- Ran the next target-ladder diagnostic campaign on office for the
  `mpol:ntor=5:20` rows with:
  - `--resolution-preset target`;
  - `--case-filter "*mpol05_ntor20"`;
  - `--run-solve`;
  - `--max-iter 20`;
  - `--niter 80`;
  - `--nstep 1`;
  - `--ftol 1e-8`;
  - `--full-solver-diagnostics`;
  - `--run-vmec2000`;
  - `--no-plots`.
- Found that the accelerated VMEC/JAX path was using the scan backend, where
  terminal `step_status_history` is absent and scan histories are omitted when
  `scan_minimal=True`.
- Updated the convergence runner to export scan-backend diagnostics:
  `diagnostic_scan_path`, scan minimal/light flags, scan preconditioner flags,
  scan time-step histories, and compact time-step scalar summaries.
- Ran a one-row `ns007_mpol05_ntor20` full-scan diagnostic with
  `VMEC_JAX_SCAN_MINIMAL=0` to verify actual scan time-step reporting.
- Found that the 80-iteration CLI path can collapse the stored solve history to
  the best final CLI-finished state, so the row needed explicit CLI finish
  metadata.
- Updated the convergence runner to export CLI finish diagnostics:
  finish attempt counts, budgets, modes, finish residuals, best finish residual,
  budget caps, budget-exhaustion flags, and parity/staged-fallback flags.
- Added `vmec_jax_total_fsq_converged_rows` to aggregate metrics so compact
  split-campaign reports distinguish total-`fsq` convergence from strict
  component convergence.
- Ran the full target ladder at the named target resolution in two chunks:
  - `*mpol05_ntor20` for `ns=7,9,15`;
  - `*mpol06_ntor24` for `ns=7,9,15`;
  both with `--max-iter 80`, `--niter 80`, `--nstep 1`, `--ftol 1e-8`,
  `--run-vmec2000`, and `--no-plots`.
- Aggregated the two 80-iteration chunks into a six-row compact target report.

### Results obtained

20-iteration diagnostic result for the `5:20` target rows:

- VMEC/JAX final `fsq` range after 20 iterations:
  `5.158289885665352e-05` to `8.469592699488549e-05`.
- VMEC2000 final `fsq` range after 80 rows:
  `6.116e-06` to `1.2370000000000002e-05`.
- VMEC/JAX did not reach the total target in this 20-iteration diagnostic
  campaign.
- The full-scan one-row diagnostic with `VMEC_JAX_SCAN_MINIMAL=0` captured
  `20` scan time-step entries for `ns007_mpol05_ntor20`; all were `0.9`, so
  the accelerated scan path was not reducing the scan time step for that row.

80-iteration target-ladder result:

- Aggregate JSON:
  `/home/rjorge/local/vmec_mirror/results/toroidal_hybrid_target_iter80_aggregate_m210/toroidal_stellarator_mirror_hybrid_convergence_aggregate.json`
- Aggregate CSV:
  `/home/rjorge/local/vmec_mirror/results/toroidal_hybrid_target_iter80_aggregate_m210/toroidal_stellarator_mirror_hybrid_convergence.csv`
- Aggregate result-tree size: `124K`.
- Case count: `6`.
- Duplicate cases replaced: none.
- VMEC/JAX solved rows: `6`.
- VMEC/JAX rows converged by total `fsq`: `6`.
- VMEC/JAX strict-converged rows: `0`.
- VMEC2000 rows: `6`.
- VMEC2000 return-code-zero rows: `6`.
- VMEC/JAX final `fsq` range:
  `2.182503700046272e-08` to `2.5155529421439975e-08`.
- Requested total target:
  `3.0000000000000004e-08`.
- VMEC2000 final `fsq` range:
  `6.116e-06` to `2.491e-05`.
- Direct-initial VMEC/JAX / VMEC2000 initial `fsq` ratio range:
  `1.002122976587009` to `1.006374048260763`.
- Every row used two accelerated CLI finish attempts with budgets `[80, 80]`,
  finish budget cap `160`, and `cli_fixed_boundary_full_parity_fallback=False`.
- The finish-budget-exhausted flag is `True` for these rows because both
  accelerated finish attempts consumed the allowed budget, even though the best
  finish residual is below the total-`fsq` target.
- Representative row results:
  - `ns007_mpol05_ntor20`: final `fsq=2.2295870282622656e-08`;
  - `ns007_mpol06_ntor24`: final `fsq=2.5155529421439975e-08`;
  - `ns009_mpol05_ntor20`: final `fsq=2.2241775184847447e-08`;
  - `ns009_mpol06_ntor24`: final `fsq=2.246120598788413e-08`;
  - `ns015_mpol05_ntor20`: final `fsq=2.182503700046272e-08`;
  - `ns015_mpol06_ntor24`: final `fsq=2.4705205119777944e-08`.

### Interpretation

- The full named target ladder now has office GPU/VMEC2000 evidence for
  VMEC/JAX total-`fsq` convergence at `ftol=1e-8`.
- This is materially stronger than the M200 three-iteration short audit: the
  target rows are no longer only plumbing/parity checks.
- The strict component convergence flags remain false, so this is a
  total-`fsq` convergence result using the fast CLI policy, not a claim that all
  strict VMEC2000-style component stopping criteria are identical.
- VMEC/JAX reaches a much smaller total `fsq` than VMEC2000 at row 80 for these
  generated target cases, while direct-initial residual parity remains within
  about `0.2%` to `0.6%`.
- The new CLI finish fields are necessary for interpreting these rows because
  the final compact VMEC/JAX history is the best CLI-finished state, not a raw
  80-point trajectory.

### How it was tested

Local checks after reporting changes:

```bash
python -m ruff check examples/toroidal_stellarator_mirror_hybrid_convergence.py tests/test_toroidal_hybrid.py
JAX_ENABLE_X64=1 pytest tests/test_toroidal_hybrid.py::test_toroidal_hybrid_convergence_history_summary_uses_iteration_labels -q
JAX_ENABLE_X64=1 pytest tests/test_toroidal_hybrid.py::test_toroidal_hybrid_convergence_example_aggregates_chunk_jsons -q
JAX_ENABLE_X64=1 pytest tests/test_toroidal_hybrid.py -q
python -m sphinx -W -b html docs docs/_build/html
git diff --check
```

Results:

- Ruff passed.
- Focused solver/CLI diagnostic helper test passed: `1 passed in 0.22s`.
- Focused aggregate metric test passed: `1 passed in 0.79s`.
- Full toroidal-hybrid tests passed: `31 passed in 9.17s`.
- Sphinx docs build passed with warnings as errors.
- Whitespace check passed.

Representative 80-iteration office command pattern:

```bash
ssh office 'cd /home/rjorge/local/vmec_mirror && PYTHONPATH=$PWD JAX_ENABLE_X64=1 CUDA_VISIBLE_DEVICES=0 /home/rjorge/venvs/vmec_jax_gpu/bin/python examples/toroidal_stellarator_mirror_hybrid_convergence.py \
  --outdir /home/rjorge/local/vmec_mirror/results/toroidal_hybrid_target_mpol05_iter80_m207 \
  --resolution-preset target \
  --case-filter "*mpol05_ntor20" \
  --ntheta-fit 64 \
  --nzeta-fit 64 \
  --run-solve \
  --max-iter 80 \
  --niter 80 \
  --nstep 1 \
  --ftol 1e-8 \
  --run-vmec2000 \
  --vmec2000-exec /home/rjorge/vmec2000/_skbuild/linux-x86_64-3.11/cmake-install/bin/xvmec \
  --vmec2000-timeout-s 240 \
  --no-plots'
```

The same pattern was run for `*mpol06_ntor24`, then both JSONs were aggregated
with `--aggregate-json`.  Generated outputs remain on office under ignored
`results/` directories and were not copied into the git repository.

### File structure and best-practice notes

- The new reporting fields stay in the existing target-convergence example
  because they describe run evidence rather than new solver APIs.
- The source package API remains unchanged.
- The code reuses the same compact row/CSV/aggregate structure, so split
  campaigns remain easy to inspect without copying WOUT or `threed1` trees.
- Tests are colocated with existing toroidal-hybrid example tests.
- Docs were updated in `examples/mirror/README.md` and `docs/mirror/overview.rst`
  to clarify scan diagnostics and CLI finish interpretation.

### Best next steps

1. Commit and push this M202 plan entry.
2. Update the draft PR body with the six-row 80-iteration target result.
3. Inspect only failed PR checks after the push.
4. Run a final review audit focused on deferred lanes and docs/readiness wording:
   the toroidal target total-`fsq` lane is now substantially complete, while
   strict component convergence and production free-boundary LCFS convergence
   should remain explicitly labeled.
5. Optionally generate a small representative plot bundle for one converged
   target row if review needs visual evidence, but keep it out of git unless a
   compressed artifact is explicitly wanted.

### Completion percentages after M202

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `95%`.
- Fixed-boundary axisymmetric solve: `96%`.
- Residual Newton / preconditioning: `96%`.
- Two-coil and manufactured validation: `95%`.
- Finite-current pitch validation: `94%`.
- Plotting and `vmec --plot` mirror support: `99%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `97%`.
- Mirror-Boozer-like diagnostics: `94%`.
- Free-boundary mirror lane: `99.3%` overall for the current diagnostic/reduced
  solver scope, with production LCFS convergence still explicitly deferred.
- Straight-axis hybrid support fixture lane: `100%` for support-fixture scope.
- Toroidal stellarator-mirror hybrid lane: `99.4%`, with all six named target
  rows converged by total `fsq` at `ftol=1e-8` on office and VMEC2000 parity
  outputs present; strict component convergence remains a caveat.
- ESSOS circular-coil mirror beta scan: `99%`.
- Public API/source simplification: `100%` for the current mirror package
  structure.
- PR merge readiness overall: `99.6%`, pending GitHub checks, final audit, and
  explicit review decision on deferred production free-boundary/strict-component
  lanes.

### User input needed

No user input is needed for the final audit and cleanup pass.

---
## 203. Target-Ladder Readiness Wording Audit

### Steps taken

- Audited docs and examples for stale target-ladder language after the M202
  six-row 80-iteration convergence evidence.
- Updated `docs/mirror/readiness.rst` so the toroidal stellarator-mirror hybrid
  lane now states the current target-ladder total-`fsq` evidence and the strict
  component caveat.
- Updated `docs/mirror/overview.rst` to replace the old target-evidence-pending
  language with the office GPU total-`fsq` convergence result.
- Updated `examples/mirror/README.md` to describe the target preset as a named
  resolution ladder rather than a no-solve-only ladder.
- Updated the target/promotion preset descriptions in
  `examples/toroidal_stellarator_mirror_hybrid_convergence.py` to avoid stale
  no-solve wording.

### Results obtained

- Focused search no longer finds stale phrases such as target convergence
  pending, no-solve target ladder, or solved/parity evidence still needing to be
  added in the current docs/example text.
- Readiness documentation now matches the M202 evidence: target-ladder
  total-`fsq` convergence is present for all six rows, while strict component
  convergence and production free-boundary LCFS convergence remain explicit
  caveats.

### How it was tested

```bash
rg -n "target-resolution production claim false|until solved/parity evidence|target-ladder convergence evidence remains pending|full target-ladder convergence evidence remains pending|target-resolution production claim false" examples docs vmec_jax tests --glob '!docs/_build/**'
python -m ruff check examples/toroidal_stellarator_mirror_hybrid_convergence.py tests/test_toroidal_hybrid.py
JAX_ENABLE_X64=1 pytest tests/test_toroidal_hybrid.py::test_toroidal_hybrid_convergence_example_target_preset_without_solve -q
JAX_ENABLE_X64=1 pytest tests/test_toroidal_hybrid.py -q
python -m sphinx -W -b html docs docs/_build/html
git diff --check
```

Results:

- Stale target-ladder wording search returned no remaining matches after the
  final wording patch.
- Ruff passed.
- Focused target-preset test passed: `1 passed in 1.25s`.
- Full toroidal-hybrid tests passed: `31 passed in 9.19s`.
- Sphinx docs build passed with warnings as errors.
- Whitespace check passed.

### File structure and best-practice notes

- Only docs/example wording and preset descriptions changed.
- No generated outputs were tracked.
- The code behavior and public APIs remain unchanged.

### Best next steps

1. Commit and push this M203 plan entry.
2. Update the draft PR body section count to 203.
3. Snapshot PR checks.
4. Continue the final audit on remaining deferred lanes, especially production
   free-boundary LCFS and strict component convergence wording.

### Completion percentages after M203

- Geometry/grids/bases: `94%`.
- Field/energy/residual kernels: `95%`.
- Fixed-boundary axisymmetric solve: `96%`.
- Residual Newton / preconditioning: `96%`.
- Two-coil and manufactured validation: `95%`.
- Finite-current pitch validation: `94%`.
- Plotting and `vmec --plot` mirror support: `99%`.
- I/O schema and docs: `100%`.
- Differentiable solved-state API: `97%`.
- Mirror-Boozer-like diagnostics: `94%`.
- Free-boundary mirror lane: `99.3%` overall for the current diagnostic/reduced
  solver scope, with production LCFS convergence still explicitly deferred.
- Straight-axis hybrid support fixture lane: `100%` for support-fixture scope.
- Toroidal stellarator-mirror hybrid lane: `99.5%`, with current docs/readiness
  wording aligned to the six-row target total-`fsq` evidence.
- ESSOS circular-coil mirror beta scan: `99%`.
- Public API/source simplification: `100%` for the current mirror package
  structure.
- PR merge readiness overall: `99.65%`, pending GitHub checks, final audit, and
  explicit review decision on deferred production free-boundary/strict-component
  lanes.

### User input needed

No user input is needed for the next audit pass.
