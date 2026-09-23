# All of VMEX

One page, the whole mental model: an input deck goes in, a solver iterates
Fourier moments to force balance, a `wout_*.nc` file comes out — and because
the solve is a JAX fixed point, you can differentiate through it and optimize
the boundary. Everything else in these docs is detail on one of those steps.

## The pipeline: input → solve → wout

```{list-table}
:header-rows: 1
:widths: 18 40 42

* - stage
  - object
  - notes
* - input
  - {class}`~vmex.core.input.VmecInput`
  - frozen dataclass with VMEC2000 semantics; reads `&INDATA` namelists and
    structured JSON ({doc}`reference/vmec2000-compatibility`), builds from keyword
    arguments, round-trips to either format; DESC inputs are converted
* - solve
  - {func}`~vmex.core.multigrid.solve_multigrid` →
    {class}`~vmex.core.solver.SolveResult`
  - the `NS_ARRAY` multigrid ladder over the VMEC2000 iteration
    ({doc}`explanation/iteration`); free boundary via
    {func}`~vmex.core.multigrid.solve_free_boundary_multigrid`
* - output
  - {func}`~vmex.core.wout.wout_from_result` /
    {func}`~vmex.core.wout.write_wout` → `wout_*.nc`
  - the VMEC2000 variable set ({doc}`reference/wout-file`); loads unchanged
    in simsopt and booz_xform
```

The `vmex` command wraps the pipeline end to end and adds plotting, the
Boozer transform, scaling, and diagnostics ({doc}`reference/cli`).

## Choosing an entry point

Four solve entry points share the same numerics; pick by what you need back:

```{list-table}
:header-rows: 1
:widths: 30 24 46

* - entry point
  - returns
  - use when
* - {func}`vmex.core.optimize.solve_equilibrium`
  - `Equilibrium` (spectral coefficients + solver context + lazy `.wout`)
  - **Default for Python work**: analysis, objectives, anything that reads
    wout tables or the `(equilibrium_state, solver_context)` scalar targets
* - {func}`vmex.core.multigrid.solve_multigrid`
  - `SolveResult` (state + convergence data)
  - you only need the converged state / iteration diagnostics — the engine
    behind the CLI and `solve_equilibrium`
* - {func}`vmex.core.implicit.run`
  - `ImplicitSolution` (differentiable pytree, carries `.runtime`)
  - gradients: wrap it in `jax.grad`/`jax.value_and_grad` — the
    implicit-adjoint path of {doc}`explanation/adjoint-gradients`
* - {func}`vmex.core.solver.solve`
  - `SolveResult` (one grid stage)
  - low-level single-`ns` building block (no `NS_ARRAY` ladder); mainly for
    solver development and tests
```

## The two lanes

The same jitted physics runs through two lanes (`vmex --mode cli|jit`): the
default host-driven **CLI lane**, with exact-`ftol` early exit and
VMEC2000-format printing, and the traced **JIT lane**, the forward solver
inside the differentiable API ({doc}`explanation/iteration`). Device placement
is described in {doc}`explanation/architecture` and {doc}`howto/run-on-gpu`.

## The multigrid ladder

`NS_ARRAY = 5 17 51` solves at ns=5, interpolates to 17, re-solves, then 51 —
VMEC2000's exact `interp.f` transfer ({doc}`explanation/iteration`). The same
seam gives hot restart: seed any solve from a previous state
(`initial_state=`) or from any wout file (`restart_from=` / `--restart`),
skipping rungs the seed already covers
({doc}`howto/restart-from-previous-run`).

## What is differentiable

Fixed-boundary equilibria are differentiable in boundary Fourier
coefficients, profiles, `phiedge`, `pres_scale`, and `curtor` through the
implicit function theorem on the converged fixed point — checked against
central finite differences on the bundled Solovev case
(`examples/take_fixed_boundary_gradients.py`, which prints the relative
agreement it reaches;
`tests/test_examples.py::test_take_fixed_boundary_gradients` fails above
`1e-4`). Coil/`extcur` derivatives on a specified
boundary go through the virtual-casing residual — the mature single-stage
lane. VMEX also differentiates the reconverged
VMEC--NESTOR free-boundary root itself:
{func}`vmex.core.freeboundary_implicit.solve_free_boundary_implicit` takes the
reverse-mode derivative of the coupled fixed point with respect to plasma
profiles and direct coil shape/current dofs. The default transpose is
`coupled_gcrot`; `boundary_schur` is opt-in. Its example,
`examples/take_free_boundary_gradients.py`, needs ESSOS
(`pip install "vmex[coils]"`). This path remains experimental because its
cold compile, memory use, and failed-trial recovery are not yet bounded. See
{doc}`reference/capabilities` for its validation grade and
{doc}`explanation/adjoint-gradients` for the method.

## Where objectives plug in

Objectives are plain functions of `(equilibrium_state, solver_context)` —
the solved spectral coefficients and the grids/profile data used to evaluate
them — including quasisymmetry,
omnigenity/QI, aspect ratio, iota, Mercier, bootstrap, turbulence proxies
({doc}`reference/objectives`). The driver
{func}`vmex.core.optimize.least_squares` takes simsopt-style
`(function, target, weight)` terms over the boundary dofs with exact implicit
Jacobians (`jac="implicit"`); {doc}`howto/optimize-a-boundary` is the
campaign recipe.

## Where to go next

Each stage has a tutorial that runs a script CI executes:
{doc}`tutorials/first-equilibrium` (solve, write, plot and Boozer-transform),
{doc}`tutorials/first-gradient` (differentiate), and
{doc}`tutorials/first-optimization` (optimize a boundary).

## The other lane: mirrors

`vmex.mirror` is a separate spline-native lane for open-mirror and
stellarator-mirror-hybrid equilibria (`mout_*.nc` output, its own implicit
adjoint): {doc}`howto/mirror-machines` and
{doc}`explanation/mirror-geometry`.
