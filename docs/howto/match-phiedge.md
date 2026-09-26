# Solve for the PHIEDGE that meets a geometric target

A free-boundary run takes the enclosed toroidal flux `PHIEDGE` as input and
returns the last closed flux surface (LCFS). VMEC and DESC both take the
flux as an input and do not solve for it
([VMEC namelist](https://princetonuniversity.github.io/STELLOPT/Tutorial%20VMEC%20Input%20Namelist.html),
[DESC free-boundary tutorial](https://desc-docs.readthedocs.io/en/latest/notebooks/tutorials/free_boundary_equilibrium.html)). Some codes have no flux input:
HINT, for example, is pinned by a point the plasma boundary must pass
through. Comparing such a code with VMEX on the same coils needs the inverse
map, the `PHIEDGE` whose LCFS meets a geometric target.
{func}`vmex.solve_phiedge` computes it.

```python
solved_inp, result = vmex.solve_phiedge(
    inp, external_field, target=0.45, metric="volume",
    phiedge0=-0.025, rtol=1e-4, max_iter=12)
```

`solved_inp` is the deck with the solved `phiedge`, and `result` is its
converged free-boundary solve.

## Metrics

`metric` selects the scalar that must equal `target`:

- `"volume"`: plasma volume [m^3]. It is an integral over the plasma, grows
  monotonically with `|PHIEDGE|` on the decks tested here, and does not depend
  on how the solve was started. Use it by default.
- `"r_outboard"`: LCFS major radius at `theta = phi = 0` [m], the outboard
  midplane of a stellarator-symmetric deck. Use it when the other code fixes
  a point, as HINT does, and the target lies well inside the reachable
  window. It follows the weakly restored `m = 1, n = 0` shift of the plasma,
  so near the window edge it can depend on warm versus cold starts (see
  Limitations).
- any callable `(inp, result) -> float`, for example the edge rotational
  transform:

```python
def iota_edge(inp, result):
    return float(vmex.wout_from_result(inp, result).iotaf[-1])

solved_inp, result = vmex.solve_phiedge(inp, field, 0.42, metric=iota_edge)
```

## Algorithm

Every residual `metric - target` costs one {func}`vmex.solve_free_boundary`
call. The host loop is a secant method:

1. The first step is a 5% probe of `phiedge0`. The secant then sets the
   direction, so the metric does not have to grow with `|PHIEDGE|`.
2. Before the target is bracketed, a step extrapolates at most twice the
   previous step. Once it is bracketed, a secant step that leaves the bracket
   is replaced by bisection. The Illinois, Brent and ITP
   ([Oliveira and Takahashi 2020](https://doi.org/10.1145/3423597))
   bracketing methods improve on this only after a bracket exists. In
   the runs here the secant converges from one side in 4 to 6 solves, before
   any bracket forms, so the simpler safeguard is kept.
3. Each solve is warm-started from the converged state nearest in `PHIEDGE`
   and is capped at twice the iterations of the first (cold) solve. A solve
   that fails or hits the cap is halved back toward the last converged
   `PHIEDGE`.

Changing `PHIEDGE` does not recompile the solver: after the first solve, each
step costs only its iterations. On the DIII-D test deck (`ns = 16`), warm
starts cut the iterations to tolerance from 2507 to 1302. On the
Landreman-Paul QA example below, the extrapolation cap and the iteration cap
avoid one diverging trial that otherwise took 49 s of a 63 s run.

The figure below comes from `examples/free_boundary_phiedge.py`
(Landreman-Paul QA coils optimized in ESSOS, vacuum, `ns = 31`, target
volume 0.45 m^3). The left panel shows each iterate's volume against
`PHIEDGE`. The right panel shows the LCFS at `phi = 0`: the converged one in
bold, the earlier iterates in grey. The example then re-solves cold at the
returned `PHIEDGE`, which gives the target volume to 7e-5 relative.

```{image} /_static/figures/free_boundary_phiedge.webp
:alt: PHIEDGE iterates converging to the target volume, and their LCFS cross-sections
```

## Differentiating through the matched PHIEDGE

When the coils change, the matched `PHIEDGE` changes too. By the implicit
function theorem on `g(phiedge, p) = metric(x*(phiedge, p)) - target = 0`,

```text
dphiedge/dp = -(dg/dp) / (dg/dphiedge).
```

{func}`vmex.phiedge_root` returns `params.phiedge` with this derivative
attached. Both partials come from one gradient of `g`, which is one coupled
plasma-vacuum adjoint solve (see [adjoint gradients](../explanation/adjoint-gradients.md)).
This is the construction behind `custom_root` in
[Blondel et al. 2022](https://arxiv.org/abs/2105.15183), applied to a
scalar root on top of the equilibrium's own implicit adjoint.

```python
from vmex.core import implicit as im

cfg = vmex.make_free_boundary_config(solved_inp, field, ns=16)

def residual(params, field):
    state = vmex.solve_free_boundary_implicit(params, field, cfg)
    return observable(state) - target             # a differentiable edge scalar

def objective(field):
    params = im.params_from_input(solved_inp)
    params = dataclasses.replace(params, phiedge=vmex.phiedge_root(residual, params, field))
    return some_function_of(vmex.solve_free_boundary_implicit(params, field, cfg))

jax.grad(objective)(field)   # includes the PHIEDGE response to extcur or coil dofs
```

`params.phiedge` must already be the root of `residual`. On the DIII-D test
deck, the derivative of the matched `PHIEDGE` with respect to `extcur`
(15.8 Wb per unit current scale) agrees with a central difference of two
re-solved roots to `1e-4` relative.

The host loop keeps secant steps rather than Newton steps with this
derivative: the secant reaches `rtol = 1e-4` in 4 to 6 solves of 1 to 3 s each
after the first, while one adjoint gradient costs about 40 s, mostly
compilation.

## Limitations

- **Reachable window.** A given coil set supports closed surfaces only over a
  limited range of `PHIEDGE`. On the Landreman-Paul QA coils at `ns = 16`,
  the reachable outboard radius is about 1.277 to 1.308 m. Beyond that, the
  LCFS runs into the island chains or the chaotic edge of the vacuum field,
  the solves stop converging, and `solve_phiedge` raises
  {class}`~vmex.core.errors.VmecConvergenceError` after `max_iter` solves.
- **Path dependence.** Near the edge of the window the converged state
  depends on where a solve starts. `r_outboard` follows the weakly restored
  `m = 1, n = 0` shift of a free-boundary plasma: on the Landreman-Paul QA
  coils, a warm and a cold solve at `PHIEDGE = -0.017737` Wb give the same
  volume to 5 digits but outboard radii 1.4 mm apart. A target of
  `r_outboard = 1.285` m returned `PHIEDGE = -0.018472` Wb, which is reached
  warm while a cold start there does not converge in 20000 iterations. Use
  `"volume"` when it fits your comparison, and re-solve cold at the returned
  `PHIEDGE`, as the example does, before relying on it.
- **Noise floor.** `rtol` must stay above the metric noise set by `ftol`.
  On the DIII-D deck, `ftol = 1e-11` leaves about 2e-5 m of noise in
  `r_outboard`, so `rtol = 1e-9` needs `ftol = 1e-13`.
