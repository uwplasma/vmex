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
reports overlap, centered-cell clearance and separation from periodic copies.
A parallelogram may cross a coordinate cut if its periodic copies remain
separated and the omitted image bound is below `1e-30`. The residual requires
`minimum_lattice_margin=0.05`; explicit `minimum_corner_margin` restores the
older containment restriction. Derivative tests cover seams and patch centers.
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
Use `frozen_path_directional_fd` to check that derivative. Independent host re-solves can
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

At finite beta, fixed-boundary coil matching uses the full boundary vector of
the **total** field, including the virtual-casing plasma contribution once. The later
coil-driven free-boundary check must recover the intended confinement region.
An unconverged equilibrium is a rejected trial, never an inexpensive optimum.

## Measured development runs

The [development record](../../benchmarks/pwo_development.json) retains the
initial and controlled runs, their final input decks and both Fourier
parities of the controlled vacuum spectra. Each used mode budgets 8/8/12 per pressure stage,
then independently solved the
candidate and seed at NS=32, MPOL=7, force tolerance `1e-13`, and rescored with
Boozer band 14 and a 39-by-39 target grid. The optimizer exhausted its budgets;
these are improved candidates, not optimality certificates.

| Target | Actual beta (%) | Refined seed residual | Refined final residual | Final local mirror |
| --- | ---: | ---: | ---: | ---: |
| Canonical | 0 | 0.149449 | 0.112917 | 0.200623 |
| Mapped | 0 | 0.0193582 | 0.00192533 | 0.0797320 |
| Canonical | 0.497447 | 0.149439 | 0.133604 | 0.228884 |
| Mapped | 0.500153 | 0.0193501 | 0.00231872 | 0.115951 |

The two objectives have different definitions, so their magnitudes cannot rank
the constructions. The mapped residual falls 90.1%, but gamma variance divided
by the squared local field range falls only 9.6%; gamma variance divided by
total mapped-field variance increases from 0.726 to 0.981. Most of this raw
improvement follows weaker modulation.
The initial canonical calibration also leaves its target mean near 6.76 T while
holding the equilibrium near 8.2 T, preventing exact matching under the mean-field
constraint.

The current canonical driver calibrates positive target bounds to the seed-grid
mean and mirror, then holds those bounds and the selected phase (zero) fixed.
The mapped driver adds a local-mirror equality residual with a fixed training-seed
denominator and a final 10% tolerance. Controlled vacuum runs give:

| Target | Refined seed residual | Refined final residual | Improvement | Final local mirror |
| --- | ---: | ---: | ---: | ---: |
| Canonical, calibrated, phase zero | 0.0279069 | 0.0213671 | 23.4% | 0.185481 |
| Mapped, fixed mirror | 0.0193582 | 0.0176004 | 9.1% | 0.230596 |

The canonical mean conflict is removed, but its mirror falls 19.4%; centered
field error divided by target variance improves 15.9%, whereas division by the
actual field range squared worsens 32.5%. The mapped mirror changes only 0.066%;
range-normalized gamma variance improves 9.2%, while gamma variance divided by
total variance rises slightly (0.7263 to 0.7353). Angular grids 39, 57 and 96
retain these conclusions. These are modest target-fit improvements with explicit
amplitude diagnostics, not evidence of family-resolved action invariance.

The exact public driver also reproduces the controlled canonical vacuum
trajectory within the finite-beta continuation. Its beta=0.50014% endpoint
reduces the independently refined residual from 0.0279032 to 0.0246551 (11.6%),
with local mirror 0.265748. The corrected mapped beta=0.49969% endpoint
reduces its refined residual from 0.0193501 to 0.0178247 (7.9%), with mirror
0.230680 versus seed 0.230449. Both pass the driver checks.
This pressure sequence prescribes zero enclosed
current; it is not a bootstrap-consistent equilibrium sequence.

An independent NEO_JAX evaluation of the corrected mapped vacuum pair gives
an unfavorable result. At the finest settings, the candidate's
`epsilon_eff^(3/2)` is 2.84–3.35 times
its seed value across five matched half-mesh surfaces. The negative-iota
period-count correction in [NEO_JAX PR #17](https://github.com/uwplasma/NEO_JAX/pull/17)
is applied without changing signed field-line winding. The finer pitch and
field-line controls change individual values by at most 2.02% and 1.42%; all
period counts remain nonnegative and no rational approximation is used.
The target-fit improvement therefore does not establish improved effective
ripple, and this proxy alone does not certify transport or bootstrap current.

The corrected canonical vacuum pair evaluated at that same finest NEO setting
has a lower proxy: candidate/seed ratios span 0.218–0.759 across the five surfaces.
This is one matched setting, without a canonical convergence sweep. Its 19.4%
mirror reduction also prevents attributing the change solely to improved pwO
structure. Neither this result nor the mapped comparison establishes alpha retention.

The measured plots use unfilled contours in a centered periodic cell so that
closed extrema and wrapping contours remain visible. The current development
fits retain predominantly poloidally wrapping structure rather than the target
parallelogram. Reduced fit error alone must not be labeled a distinct pwO result.

Full derivative and resolution qualification, family-resolved actions,
transport and particle retention remain separate gates. The shared-host wall
times in the record are observations, not a controlled speed comparison.

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
