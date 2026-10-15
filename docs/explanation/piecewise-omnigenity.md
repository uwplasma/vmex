# Piecewise-omnigenous optimization

**Development status:** this page defines the implementation and validation
contract. It does not yet provide a pwO objective or report an optimized result.

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
and rescore at independently refined resolution. Planned examples cover vacuum
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
