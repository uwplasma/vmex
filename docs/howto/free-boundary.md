# Run a free-boundary equilibrium

Set `LFREEB = T` and supply the external field — a MAKEGRID `mgrid` file
with `EXTCUR` currents, or ESSOS coils directly — and VMEC finds the plasma
boundary that balances against it: the last closed flux surface is an
output, not an input.

## From an mgrid file

```text
&INDATA
  LFREEB = T
  MGRID_FILE = 'mgrid_case.nc'
  EXTCUR = 1.0e5  1.0e5  ...     ! one current per coil group
  NVACSKIP = 6                   ! full NESTOR solve cadence
  ...
```

```console
vmex input.case
```

The console shows the VMEC2000 free-boundary output (`In VACUUM` block,
`VACUUM PRESSURE TURNED ON` banner), and the wout gains the free-boundary
metadata (`nextcur`/`extcur`/`curlabel`/`mgrid_mode`) plus the NESTOR
potential and surface fields (`potsin`/`xmpot`/`xnpot`/`*_sur`). A missing
mgrid file falls back to a fixed-boundary solve with a warning — retained
VMEC2000 behavior, so check the banner if a "free-boundary" run looks
suspiciously fixed. The end-to-end example below reads the CTH-like
`mgrid_cth_like.nc`, which is not stored in git; from a source checkout,
fetch it first with `python tools/fetch_assets.py --bundle reference-nc`:

```{literalinclude} ../../examples/free_boundary_mgrid.py
:language: python
```

## From ESSOS coils (no mgrid file)

VMEX is coil-agnostic: the solver consumes only a magnetic field. Set
`MGRID_FILE = 'DIRECT_COILS'` and pass an ESSOS coils file
(requires `pip install essos`):

```console
vmex input.case --coils coils.json
```

The coils' Biot-Savart field is tabulated once into an in-memory
{class}`~vmex.core.mgrid.MgridField` via
{meth}`~vmex.core.mgrid.MgridField.from_coils` — no temporary mgrid file.
In Python the same call takes an `essos.coils.Coils` object directly
({doc}`use-essos-fields-and-coils`). The underlying
{meth}`~vmex.core.mgrid.MgridField.from_cartesian_field` adapter also accepts
ESSOS' `B(points)` protocol, SIMSOPT's `set_points(points); B()` protocol, or
any plain Cartesian callable, and the tabulated field is reused across every
radial stage and hot restart.
`examples/free_boundary_essos_coils.py` runs a beta scan against the
Landreman-Paul precise-QA coil set this way:

```{literalinclude} ../../examples/free_boundary_essos_coils.py
:language: python
```

## Mgrid interpolation order

{class}`~vmex.core.mgrid.MgridField` interpolates the tabulated field with
one of two kernels, selected by `order`:

| `order` | Kernel | Smoothness | Use |
|---|---|---|---|
| `1` (default) | trilinear | C0 value, piecewise-constant gradient | VMEC2000 / MAKEGRID parity; reproduces reference wouts |
| `3` | tricubic Catmull-Rom | C1, third-order accurate | accuracy on a given table |

Measured on the Landreman-Paul precise-QA coils tabulated on a
181 x 201 x 64 grid, the error against direct Biot-Savart is rms 1.8e-5 /
max 1.1e-4 (relative) for trilinear and about 8e-8 for tricubic. On the
CTH-like deck one tricubic field evaluation costs 4.2x a trilinear one, and a
full free-boundary solve takes 12% longer (1.63 s to 1.83 s). The trilinear
kink matters on coarse tables: on a 1 cm table two converged tokamak
equilibria differed by 7 mm in axis position; the tricubic kernel or a finer
table removes the discrepancy.

The CLI and `solve_file` always use `order=1`. To solve a deck with the
tricubic kernel, build its field and pass it as `external_field`:

```python
import vmex as vj

inp = vj.VmecInput.from_file("input.case")
field = vj.MgridField.from_input(inp, order=3)   # MGRID_FILE scaled by EXTCUR
result = vj.solve_free_boundary(inp, external_field=field)
```

{meth}`~vmex.core.mgrid.MgridField.from_file` and
{meth}`~vmex.core.mgrid.MgridField.from_mgrid_data` take the same `order`
keyword. {meth}`~vmex.core.mgrid.MgridField.from_coils` and
{meth}`~vmex.core.mgrid.MgridField.from_cartesian_field` return an
`order=1` field; switch the kernel with
`dataclasses.replace(field, order=3)`.

## Interpolation order of the tabulated field

Every tabulated field ({meth}`~vmex.core.mgrid.MgridField.from_file`,
{meth}`~vmex.core.mgrid.MgridField.from_coils`,
{meth}`~vmex.core.mgrid.MgridField.from_cartesian_field` and
{meth}`~vmex.core.mgrid.MgridField.from_input`) takes `order=1` (trilinear,
the default) or `order=3` (tricubic, C1). The table itself is the same.
`MgridField.from_input(inp, mgrid_path=None, *, order=1)` builds a deck's
field with the solver's `EXTCUR` / `raw_coil_cur` scaling, so it is the way
to get the tricubic version of the field a deck would use.

Trilinear stays the default because it is the VMEC2000 parity kernel that
every parity test and reference wout is pinned to. Tricubic is the better
choice for coarse tables, coil optimization, and orbit tracing that needs a
continuous grad B:

- Landreman-Paul QA coils, 181x201x64 table, inside the LCFS against
  Biot-Savart: trilinear rms 1.8e-5, max 1.1e-4; tricubic 8e-8.
- Same coils, 96x96x32 table, 64 points 2 cm outside a free-boundary LCFS:
  max |B| error 2.1e-3 (trilinear) vs 1.9e-4 (tricubic); max grad|B| error
  3.1e-2 vs 7.2e-3.
- CTH-like free-boundary deck (ftol 1e-12, CPU): `b_cyl` at 20k points
  1.46 ms vs 6.17 ms (4.2x); full solve 1.63 s vs 1.83 s (+12%), with
  volume and edge R00 agreeing to about 1e-5.

```python
field = vj.MgridField.from_input(inp, order=3)          # deck + mgrid file
coil_field = vj.MgridField.from_coils(coils, order=3)   # ESSOS coils
```

## Three-term free boundary (no sheet current)

NESTOR makes the boundary a flux surface of the vacuum field and balances only
`|B|` across it, so a VMEC/NESTOR equilibrium may carry an edge sheet current
`K` (a jump in the direction of the tangential field). Different such states
can solve the same deck, and the solve can leave the one whose coils were fitted
to it. `boundary_condition="three_term"` instead solves for the boundary on
which all three interface conditions hold, with the plasma's own field from
virtual casing (Conlin et al. 2024, arXiv:2412.05680):
`B_out . n = 0`, `|B_out|^2 = |B_in|^2 + 2 mu0 p`, and `n x (B_out - B_in) = 0`.
Each trial boundary is a fixed-boundary equilibrium, and Gauss--Newton with
exact implicit derivatives drives the three residuals to zero:

```python
import vmex as vj

result = vj.solve_free_boundary_multigrid(
    inp, external_field=coil_field, boundary_condition="three_term",
    three_term_options=dict(max_nfev=30))
print(result.boundary_residual)   # RMS of B.n, pressure jump and mu0 K, over |B|
```

or `vmex input.case --boundary-condition three-term` on the command line,
or a directive in the deck itself, which VMEC2000 reads as a comment (the CLI
flag and the `solve_file` keyword override it):

```fortran
!@VMEX BOUNDARY_CONDITION = THREE_TERM
&INDATA
  LFREEB = T
  MGRID_FILE = 'mgrid_case.nc'
  ...
/
```

(`"_vmex": {"boundary_condition": "three_term"}` in a JSON input).
The deck boundary is the initial guess, so a fixed-boundary design and the
coils fitted to it are a natural start. The edge pressure must vanish (a
pressure jump needs a sheet current), and `virtual-casing-jax` must be
installed. Virtual casing dominates the cost and runs on the default JAX device,
so use a GPU; expect several times the cost of a NESTOR solve.
{func}`vmex.core.freeboundary_vc.solve_free_boundary_three_term`
exposes the grid, weights and least-squares controls, and its `previous=`
restarts from an earlier result (after a coil change, say) in a few
equilibrium solves without compiling anything:

```python
from vmex.core.freeboundary_vc import solve_free_boundary_three_term

fit = solve_free_boundary_three_term(inp, external_field=coil_field)
fit.boundary_residual, fit.equilibrium.wout          # the conditions, the free boundary
moved = solve_free_boundary_three_term(inp, external_field=new_coil_field, previous=fit)
```

For a free-boundary single-stage optimization, pass
`boundary_condition="three_term"` to `FreeBoundaryProblem.from_loss` (the same
loss, parameter maps and quantities as with NESTOR): every trial is a
three-term free boundary, warm-started from the last linearized one, and the
design gradients follow from the implicit function theorem of the boundary fit
({class}`vmex.core.freeboundary_vc.ThreeTermFreeBoundaryProblem`).

```python
problem = opt.FreeBoundaryProblem.from_loss(
    inp, loss, x0, field_from_parameters=coils_from_x, plasma_from_parameters=plasma_from_x,
    quantities=(opt.major_radius,), boundary_condition="three_term",
    three_term_options=dict(quadrature=(4 * inp.nfp * 48, 96)))
result = opt.minimize(problem, method="SLSQP", constraints=[problem.nonlinear_constraint(lower, upper)])
```

For other uses of many fields and plasma parameters,
{class}`vmex.core.freeboundary_vc.ThreeTermFreeBoundaryModel` gives the
pieces directly: `solve_boundary` (cold or warm), `linearize` (the interface
Jacobian and the state responses to the boundary and any plasma parameters),
and `pullback` (reverse-mode gradients of other functions of the equilibrium,
e.g. a quasi-symmetry residual), with the coils an argument of the compiled
code.

To check a NESTOR result instead, pass `report_boundary_residual=True`: the
three conditions are evaluated on the converged boundary and returned as
`result.boundary_residual`. VMEC + NESTOR (VMEX and VMEC2000 alike) leaves a
floor of a few 1e-4 even for an exact external field, and the virtual-casing
evaluation itself is accurate to about 2e-4 on its default 48 x 48 grid, so a
`sheet_current` well above 1e-3 marks a genuine edge sheet current.

How close each method gets to a known answer at mpol = ntor = 4 to 12, and at
what cost, is recorded in {doc}`/explanation/validation`
("Three-term free boundary against VMEC + NESTOR and DESC").

## Key knobs

- `EXTCUR` — coil-group currents scaling the mgrid field.
- `NVACSKIP` — iterations between full NESTOR solves; between them, cheap
  incremental updates reuse the factored potential matrix, and the cadence
  adapts toward convergence ({doc}`/explanation/nestor-vacuum`).
- `--jacobian-retries` — free-boundary recovery after the 75-reset condition
  rebuilds the axis filament and NESTOR structures before continuing
  ({doc}`troubleshoot`).

## Convergence differences from fixed boundary

The vacuum solve activates only once `fsqr + fsqz <= 1e-3`, so early
iterations run effectively fixed-boundary; expect the residual trace to
change character at the `VACUUM PRESSURE TURNED ON` banner. Free-boundary
ladders carry the active-vacuum state and adaptive `NVACSKIP` across
`NS_ARRAY` stages ({doc}`/explanation/iteration`). On GPUs, the dense NESTOR
factor runs on CPU by design ({doc}`run-on-gpu`).

## Differentiability scope

Coil/`extcur` gradients on a specified boundary use the virtual-casing
residual. The coupled NESTOR fixed point is differentiated by
{func}`vmex.core.freeboundary_implicit.solve_free_boundary_implicit`, which
reverse-differentiates the reconverged plasma--vacuum root against plasma
profiles and direct coil variables. This path is experimental; with the
default `device="auto"` it runs on the CPU even on a GPU host.
Its coil examples need ESSOS (`pip install "vmex[coils]"`). Scope is in
{doc}`/reference/capabilities` and the mechanism in
{doc}`/explanation/nestor-vacuum`.
