# Exterior field

{class}`~vmex.core.extender.VmecExtender` returns the total magnetic field at points
outside the last closed flux surface, in tesla at Cartesian points in metres: the field
of the coils or any other external source, plus the field of the currents inside the
plasma. The plasma part is the virtual-casing field of the boundary data
(`virtual_casing_jax`, installed with `pip install "vmex[coils]"`). This page lists every
constructor, parameter, attribute and method. {doc}`/tutorials/exterior-field` walks
through them with runnable code, and {doc}`/explanation/nestor-vacuum` derives the
quadratures.

## Constructors

| Constructor | Source of the plasma field | Differentiable in |
|---|---|---|
| `VmecExtender.from_wout(wout, ...)` | a {class}`~vmex.core.wout.WoutData` | the points only |
| `VmecExtender.from_file(path, ...)` | a `wout_*.nc` file; a relative mgrid is resolved beside it | the points only |
| `VmecExtender.from_state(inp, state, ...)` | a live VMEX state | the points, and the state through `jax` |
| `VmecExtender.from_equilibrium(eq, ...)` | an {class}`~vmex.core.optimize.Equilibrium`, as `from_state` or `from_wout` | as `from_state` |
| `VmecExtender.from_surface_data(surface_data, ...)` | traceable `VmecSurfaceFieldData` | the points, and the surface data |
| `VmecExtender.from_parameterized_surface_data(fn, parameters, ...)` | `fn(parameters)` returning surface data | the points, and `parameters` through `B_vjp` and its siblings |
| `problem.exterior_field(x, ...)`, `equilibrium.exterior_field(...)` | the problem's differentiable state | the problem's degrees of freedom |
| `VmecExtender(external_field, plasma_field)` | prebuilt pieces (low level) | the points |

`plasma="auto"` (the default of `from_wout`, `from_file` and `from_equilibrium`) adds virtual
casing only when the equilibrium has pressure or a net current; `"include"` and `"vacuum"` force
either choice. A vacuum extension needs `external_field` or an mgrid file.

## Parameters shared by the constructors

| Parameter | Default | Meaning |
|---|---|---|
| `external_field` | mgrid beside the wout, else `None` | The coil or vacuum field: an object with `b_cyl(r, phi, z) -> (B_R, B_phi, B_Z)` such as an {class}`~vmex.core.mgrid.MgridField`, or a callable `xyz (n, 3) -> B (n, 3)` such as an ESSOS `BiotSavart`. Must be traceable by JAX for the derivatives. |
| `digits` | `6` | Requested accuracy of the plasma field, `10**-digits` relative to the RMS of `|B|` on the boundary. It sizes the default source grid and is the threshold of `accuracy_check` and of the `"auto"` switch. |
| `nphi`, `ntheta` | from `digits` | Source samples per field period and in poloidal angle. The default is the smallest power of two, at least 32 and at most 256, for which the direct quadrature reaches `digits` one minor radius out: `nphi >= digits ln(10) R0 / (2 nfp a)`. Pass them to override. Not available on `from_surface_data`, which takes the data as sampled. |
| `levels` | `((nfp nphi, ntheta), (2 nfp nphi, 2 ntheta))` | The direct quadrature's schedule of full-torus `(n_toroidal, n_poloidal)` source grids, coarse to fine. The finest sets the node spacing `h` of the direct path and the base of the graded rule's default node count. |
| `chunk_size` | `"auto"` | Source points per virtual-casing batch. A cap on memory; the default is tuned. |
| `target_chunk_size` | `"auto"` | Evaluation points per batch. |
| `accuracy_check` | `"warn"` | What an eager call does when its error estimate misses `10**-digits`: `"warn"` emits {class}`~vmex.core.extender.ExteriorFieldAccuracyWarning`, `"raise"` raises {class}`~vmex.core.extender.ExteriorFieldAccuracyError`, `"off"` skips the check. Traced calls never check. The returned field is the same in every mode. |
| `project_current` | `False` | `from_wout`, `from_state`: keep only the surface-gradient part of the boundary field before virtual casing, so the sheet current is conserved and the plasma field is exactly curl-free. |
| `plasma` | `"auto"` | `from_wout`, `from_file`, `from_equilibrium`: `"auto"`, `"include"` or `"vacuum"`, as above. |
| `base_dir` | `None` | `from_wout`: directory in which a relative mgrid path is resolved. |
| `runtime` | `None` | `from_state`: the solver runtime that belongs to `state`. Pass it when the profile parameters (`curtor`, `pres_scale`, ...) are differentiated; without it the profiles are read from `inp` and their direct dependence leaves the graph. |

`near_surface` and `graded_nodes` are accepted by `from_surface_data`,
`from_parameterized_surface_data` and the constructor itself. The other constructors return the
default `"auto"` field; change it afterwards with the attribute `field.near_surface = "graded"`
or with `with_graded_quadrature`.

## Near-surface quadrature

The direct quadrature is a periodic trapezoid rule on the source grid. Its error decays as
`exp(-2 pi d / h)` with `d` the distance to the boundary, so it is exact to rounding a minor
radius out and unusable within a few node spacings. `virtual_casing_jax` supplies a second rule
that gives each target its own nodes, clustered about the nearest boundary point.

| `near_surface` | Behaviour |
|---|---|
| `"auto"` (default) | The direct rule at each point where its order-0 error estimate meets `10**-digits`, the graded rule at every other point; one map for `B` and for every derivative order, eager or traced, so the jets and the parameter pullbacks differentiate the same function. |
| `"graded"` | The graded rule at every point. For field-line traces and objectives that stay close to the boundary, and under `jit`. |
| `"direct"` | The direct rule alone, as in releases before the graded rule. For comparisons and for targets known to be far away. |

| Attribute or method | Meaning |
|---|---|
| `near_surface` | The mode above. A setter; any other value raises `ValueError`. Setting it discards cached parameter pullbacks. |
| `graded_nodes` | `(poloidal, toroidal)` node counts of the graded rule over the full torus, read only. `None` is the default of `virtual_casing_jax`: twice the finest schedule level, at least 256 x 512. More nodes cost time and buy accuracy only where the surface data resolve it. |
| `with_graded_quadrature(nodes=None)` | A copy that uses the graded rule everywhere, optionally with `nodes=(poloidal, toroidal)`. The copy owns its quadrature settings and its parameter pullbacks. Raises `RuntimeError` on a vacuum extension. |
| `accuracy_check` | The mode of the parameter of the same name; a setter. |
| `uses_virtual_casing` | Whether a plasma field contributes. |

The graded rule is accurate on either side of the boundary and on it (where it returns the
outside limit). It costs a few milliseconds per target on a CPU, against microseconds for the
direct rule, and it cannot be more accurate than the boundary samples: the plasma field it
returns has the spectral content of `surface_data`, so raise `nphi` and `ntheta` to improve a
point it flags.

## Evaluation and error estimate

| Method | Returns |
|---|---|
| `B(points=None)` | Cartesian `B`, shape `(n, 3)` [T]. |
| `gradB`, `gradgradB`, `gradgradgradB` `(points=None)` | `dB_i/dx_j`, `d2B_i/dx_j dx_k`, `d3B_i/dx_j dx_k dx_l`, shapes `(n, 3, 3)`, `(n, 3, 3, 3)`, `(n, 3, 3, 3, 3)`. |
| `B_error_estimate(points=None, *, order=0)` | Estimated relative error of the plasma field, shape `(n,)`, for the quadrature the same call of `B` (or of the `order`-th derivative) would use. `-log10` of it is the achieved digits. Traceable at `order=0`; the a-priori estimate of the direct path at `order > 0` is host side and raises under a trace. Raises `RuntimeError` on a vacuum extension. |
| `absB`, `GradAbsB` `(points=None)` | `|B|` and its gradient. |
| `set_points_xyz(points)`, `set_points_cyl(points)` | Store points so the methods above can be called without arguments; return the field. |
| `B_vjp(v)`, `gradB_vjp(v)`, `gradgradB_vjp(v)`, `gradgradgradB_vjp(v)` | Vector-Jacobian product with cotangent `v` (the shape of the stored output), taken at the stored points, with respect to the degrees of freedom named in `dof_names`. Available when the field was built by `from_parameterized_surface_data`, `problem.exterior_field` or `equilibrium.exterior_field`. |
| `dof_names` | Names of the degrees of freedom, boundary modes first and then the external ones. |

Passing `points` to a derivative method is a pure function of the points and works under
`jit`, `vmap` and `grad`. The error estimate is the larger of the finest level's double-layer
quadrature error and the squared relative change between the last two schedule levels for the
direct rule; for the graded rule it is the larger of the change from every second node, the last
Taylor term next to the boundary and the surface data's own spectrum at the edge of what the
samples resolve. Measured against a converged graded quadrature it is 0.76 to 1.09 times the true
error on the 2.5% beta QA deck, so it tracks the error rather than bounding it.

## Parameterized fields and gradients

`from_parameterized_surface_data(surface_data_fn, parameters, ...)` builds a field whose
`*_vjp` methods differentiate through `surface_data_fn`.

| Parameter | Meaning |
|---|---|
| `surface_data_fn` | `parameters -> VmecSurfaceFieldData`, traceable. |
| `parameters` | The plasma degrees of freedom, flattened. |
| `dof_names` | Names of those degrees of freedom. |
| `external_field` | A fixed external field. Not allowed together with `external_parameters`. |
| `external_parameters`, `external_field_from_parameters`, `external_dof_names` | Degrees of freedom of the external field (for example ESSOS coil modes) and the function `external_parameters -> field` that builds it. The first two must be given together; the VJP returns plasma entries followed by external entries. |
| `digits`, `levels`, `chunk_size`, `target_chunk_size`, `accuracy_check`, `near_surface`, `graded_nodes` | As above. The pullback differentiates the quadrature the field currently uses. |

The VJPs and the spatial derivatives come from one JAX graph; there are no finite-difference
equilibrium solves inside. The grading centre of the graded rule is held fixed under
differentiation, which the verification in {doc}`/tutorials/exterior-field` checks against
re-solved equilibria. A copy made by `copy.copy` or `with_graded_quadrature` binds its
pullback to itself, so changing the copy's `near_surface` does not change the original.

## Examples

| Script | Shows |
|---|---|
| `examples/vmex_get_B_outside_plasma.py` | `B`, `|B|`, the error estimate, three spatial derivatives and their VJPs at a point outside a finite-beta QA boundary, with coil degrees of freedom. |
| `examples/vmex_fieldline_tracing_finite_beta.py` | Field lines of coils plus plasma through the boundary. |
| `examples/vmex_fieldline_tracing_vacuum.py` | The vacuum extension and its field lines. |
| `examples/vmex_fixed_free_boundary_comparison.py` | The exterior field of a fixed-boundary equilibrium against the free-boundary solve of the same coils. |
| `examples/vmex_essos_tracing_to_wall.py` | The exterior field, tabulated by ESSOS, to a wall. |
| `examples/mirror/mirror_exterior_field.py` | The field of a mirror plasma outside its boundary (`virtual_casing_jax.mirror`), and the coil field inside it. |
