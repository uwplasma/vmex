# Piecewise-omnigenous optimization

**Development status:** the array objectives and optimization drivers are
implemented. Full equilibrium derivatives, resolved optimization results,
transport, alpha retention and coil realization require separate qualification.
No low-loss configuration is certified by the target fit alone.

## Physical target

Ordinary omnigenity makes the bounce action independent of the field-line label.
Piecewise omnigenity (pwO) permits distinct trapped-orbit families to have
separate constant actions, with the appropriate relations at their junctures.
For field-line label `alpha = theta - iota*zeta` and physical pitch
`lambda = mu/E`, VMEX's normalized action is

```text
J_w = 2 integral_w sqrt(1 - lambda*B) dl
    = 2 |G + iota*I| integral_w sqrt(1 - lambda*B)/B dzeta.
```

The derivative with respect to `alpha` must vanish within a family. A variance
that pools every well is not a pwO residual: it penalizes permitted differences
between families. Well indices also need physical matching across field lines;
array position alone is not a family label.

The initial scope is a smooth canonical pwO target and a QI/PO-pwO target.
QA-pwO is a later extension, not a claim made by implementing those two families.
The ideal construction, its finite-bandwidth approximation, and a fitted MHD
equilibrium must remain separate objects in both tests and reporting.

## Implementation boundary

### Canonical target and periodic completion

For signed rotational transform `iota`, define

```text
w1 = pi*iota*(1-t1*t2)/(NFP*(iota+t2)), w2 = pi
u = (zeta-zeta_c+t1*(theta-theta_c))/w1
v = (theta-theta_c+t2*(zeta-zeta_c))/w2
g = exp(-u^(2p)-v^(2p)).
```

`canonical_pwo_field` completes the fundamental patch by a bounded lattice
union, `Bmin + (Bmax-Bmin)*(1-product(1-g_kl))`, over angular images.
This is an explicit extension, not an exact reproduction of the uncompleted
fundamental-domain formula in the CIEMAT-pw1 DESC source. The implementation
reports overlap and corner clearance. It requires the sharp parallelogram to
fit inside its centered angular cell, which bounds the omitted image tails.
For `p >= 2`, the default two image shells make the field seams negligible at
float64 precision. Derivative tests include the wrapped seams and patch center.
The union preserves the upper and lower bounds; its finite tails generally
make the attained minimum greater than `Bmin`.
Finite `p` rounds the corners and can introduce additional accessible winding
families. It does not preserve exact sharp-limit action invariance. Check
within-family actions, including corner-near wells, before interpreting a fit.

`CanonicalPwOResidual` fits the signed relative difference `(B-Btarget)/Btarget`
on a uniform Boozer grid. Surface and angular weights sum to one. The width
depends on the live signed transform. Reflect both tilt signs when reflecting
the transform; taking its absolute value would change the construction.
Target strengths and tilts are fixed in the initial drivers. The pure-array
kernel accepts array parameters for explicit JVP/VJP calculations.

### Squeezed mapping

`MappedPwOResidual` evaluates the LC mapping from Liu et al. (2026), using
`S(x,y)=a*x*sin(y+b*sin(y))` and `D(x)=c*x*x-(1+c*pi)*x+pi`.
The mapped coordinate `gamma` is distinct from the field-line label.
It computes gamma variance at each eta, divided by a fixed Tesla `b_ref`.
Discrete Parseval in gamma makes this equivalent to the nonconstant gamma
Fourier power with the same quadrature, including sine content.

Eta midpoint quadrature covers the fixed interval
`[eta_min, 2*pi-eta_min]`; no periodic extension in this cut direction is
assumed. The published optimization's `eta_min=2.8` differs from the
root `pi+1/c` at `c=-2`. A conservative analytic lower bound on the mapping
Jacobian certifies the entire interior, including between sampled nodes.
The diagnostics include that bound and the mapped area fraction.
This residual constrains the mapped region only. A pure-QI limit can also
score well; the unmapped high-field pwO families need independent checks.

### Interfaces and examples

Both classes implement `compute_state`, `residuals_state`, `total_state`,
`residuals(eq)` and `total(eq)`. `compute_state` returns the actual Boozer
half-mesh radii `s_b`, both Fourier parities, sampled fields and validity.
Eager invalid inputs raise; invalid traced states produce NaNs, never a
favorable zero. Pure kernels support JIT, JVP and VJP. The equilibrium chain
uses the solver's supported first-order implicit derivative; this does not
promise arbitrary nested or forward-mode differentiation through a VJP solver.
The implicit linearization holds the released m=1 gauge at its converged value.
Use `frozen_path_directional_fd` to check that derivative. Cold re-solves can
follow a different gauge at finite resolution; report that discrepancy and
resolution dependence separately from agreement on the frozen path.

```bash
VMEX_PWO_FAMILY=canonical python examples/optimization/pwO_optimization.py
VMEX_PWO_FAMILY=mapped python examples/optimization/pwO_optimization_finite_beta.py
python examples/optimization/single_stage_pwO_optimization.py
```

Stage 1 fixes flux and major radius, constrains transform and nonzero mirror
content, and checks physical size and field after a refined solve. The
finite-beta wrapper calls the same side-effect-free driver with a pressure
continuation ladder and zero prescribed enclosed current. Set
`VMEX_PWO_METHOD=least_squares` to compare its residual-Jacobian route with
the default scalar adjoint using the same cost. Smoke budgets exercise wiring
and report unmet checks; they do not establish the research criteria below.

The single-stage driver reads the saved stage-1 input, scales it with the
existing `volavgB` ARIES-CS convention, prefits coils, and varies boundary and
coil coefficients jointly. Full boundary-vector matching fixes amplitude and
direction; at finite beta the existing virtual-casing plasma field is included
once. Rejected solve statuses cannot enter its best-valid checkpoint. Its
geometric limits describe filaments, not conductor packs or a mechanical design.

Reuse the live `boozer_spectrum_state` transform, `bounce_action`, and
`VmecProblem` objective interface. Do not add another equilibrium solver, Boozer
transform, or particle integrator. Preserve both Fourier parities and physical
`toroidal mode = n*NFP` conventions. Report the actual half-mesh radii returned
by the transform.

Start with a bounded, periodic smooth-field target at one intermediate surface.
Check target geometry, periodic completion, and derivatives before optimizing
multiple radii. Use the equilibrium's live signed transform wherever the target
construction depends on it. Keep finite-pitch well topology diagnostics separate
from the differentiable target fit.

Follow the existing optimization examples: a short constants block, explicit
objective terms, spectral continuation, progress reporting, and a final solve
and rescore at independently refined resolution. The examples cover vacuum
stage 1, finite-beta stage 1, and simultaneous plasma-coil optimization. A
short example run and a research validation campaign have different budgets.

At finite beta, fixed-boundary coil matching uses the normal component of the
**total** field, including the virtual-casing plasma contribution. The later
coil-driven free-boundary check must recover the intended confinement region.
An unconverged equilibrium is a rejected trial, never an inexpensive optimum.

## Validation contract

- Check the target and its derivatives against independent formulas and
  quadrature, then check the full boundary-to-equilibrium-to-Boozer derivative.
- Refine equilibrium, transform, angular, pitch, and field-line resolutions
  separately. Resolve well populations, junctures, and excluded measure.
- Reproduce a published configuration at matched conventions before claiming a
  new configuration. A low target residual is not evidence of low transport.
- Use `GammaC` as a hard diagnostic; its discretized gradient need not converge.
  Assess the existing `GammaCSmooth` before introducing another regularizer.
- Validate with `vmex --trace` at the declared ARIES-CS scaling and particle
  energy. The existing 3.52 MeV default is not exactly a 3.5 MeV calculation.
  Coordinate explicit-energy support with PR #540 rather than duplicate it.
- Compare full-spectrum tracing with any mode cut or angular table. The fast
  K=0 Boozer model needs an independent field/model comparison at finite beta.
- Report numerical failures separately from confined particles. Quote a
  binomial upper confidence bound, including for zero observed losses.
- Distinguish short-time collisionless particle losses from collisional escaped
  energy and deposited heating. Match births, profiles, and duration when
  comparing with published results.

The README will show only measured results, with reproducible input decks,
source revisions, numerical controls, and one or two generated figures. Detailed
convergence, transport, and engineering evidence belongs in the documentation
and existing benchmark records. No low-loss or manufacturability claim is made
until the corresponding checks pass.

## Literature

- [Velasco et al. (2024), Piecewise Omnigenous Stellarators](https://arxiv.org/abs/2405.07634).
- [Velasco, Sanchez and Calvo (2025), parameter-space exploration](https://arxiv.org/abs/2412.14871).
- [Calvo et al. (2025), zero-bootstrap-current subset](https://arxiv.org/abs/2505.02546).
- [Fernandez-Pacheco et al. (2026), CIEMAT-pw1](https://arxiv.org/abs/2601.14886).
- [Velasco et al. (2026), QI-pwO construction](https://arxiv.org/abs/2603.12377).
- [Liu et al. (2026), mapped-coordinate optimization](https://arxiv.org/abs/2603.12139).
- [Yu et al. (2026), quasi-single-stage coil optimization](https://arxiv.org/abs/2608.03122).

CIEMAT-pw1 includes a coil-feasibility proxy, while the quasi-single-stage study
constructs QI-pwO modular coils. The research objective is therefore validated
confinement with explicit engineering constraints, not an unsupported claim to
be the first pwO coil design.
