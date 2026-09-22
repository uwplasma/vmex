# Use ESSOS fields, coils and alpha tracing

VMEX and [ESSOS](https://github.com/uwplasma/ESSOS) meet at two Python seams,
and each one runs in a single direction. Neither package imports the other:
the equilibrium crosses as a wout, the coil field crosses as a tabulated
grid. Requires ESSOS (`pip install essos`).

| Direction | Call | Crosses as |
| --- | --- | --- |
| VMEX equilibrium to ESSOS | {func}`~vmex.core.tracing.essos_vmec_field` | wout tables |
| ESSOS coils to VMEX | {meth}`~vmex.core.mgrid.MgridField.from_coils` | cylindrical field grid |

There is no third seam. VMEX owns fixed and free-boundary equilibrium
physics and its diagnostics; ESSOS owns coil geometry, Biot-Savart fields,
particle and field-line tracing and loss diagnostics.

## Hand an equilibrium to ESSOS

```python
import vmex as vj

field = vj.essos_vmec_field("wout_case.nc")   # essos.fields.Vmec
print(field.nfp, field.AbsB([0.5, 0.0, 0.0]), field.surface.gamma.shape)
```

`essos_vmec_field` takes either an in-memory
{class}`~vmex.core.wout.WoutData` or a path to a `wout_*.nc`. Released ESSOS
reads a wout *file*, so an in-memory equilibrium is written to a temporary
wout; ESSOS loads every table eagerly in its constructor, so the file is
gone by the time the field is returned. Keyword arguments pass through to
`essos.fields.Vmec` (`ntheta`, `nphi`, `close`, `range_torus`), which set the
resolution of the `field.surface` ESSOS builds alongside the field.

Three consequences worth knowing before you build on this:

- **The write severs the gradient.** This seam is a diagnostic route, not a
  differentiable one. A differentiable alpha-loss objective needs the ESSOS
  array constructor (uwplasma/ESSOS#61) and is not available here.
- **Stellarator symmetry only.** Released ESSOS reads the symmetric wout
  tables, so an `lasym` equilibrium is rejected rather than silently
  half-transferred.
- **Radial resolution is yours to choose.** ESSOS interpolates the half-mesh
  tables linearly in `s`, so its two independent `|B|` channels (`AbsB` from
  `bmnc`, and `norm(B)` built from `bsub*`, `gmnc` and the geometry) agree
  better on finer radial grids. Solve on the grid your diagnostic needs.

The tables themselves cross unchanged: `tests/test_tracing.py` requires the
file and in-memory routes to give identical fields.

## Bring an ESSOS coil field back

VMEX keeps no coil code, and its free-boundary solver consumes only a
magnetic field, so an ESSOS coil set enters through one tabulation:

```python
from essos.coils import Coils

coils = Coils.from_json("coils.json")
coil_field = vj.MgridField.from_coils(coils)          # or pass a BiotSavart
res = vj.solve_free_boundary(inp, external_field=coil_field)
```

Unset bounds default to the coil bounding box grown by 10%, and `nfp`
defaults to the coil set's own period count; the CLI equivalent is
`vmex input.case --coils coils.json` ({doc}`free-boundary`). Pass `rmin`/`rmax`/`zmin`/`zmax`
to bracket the plasma more tightly than the coils do, and `ir`/`jz`/`kp` to
set the grid (96, 96, 32 by default). The result is the same in-memory
{class}`~vmex.core.mgrid.MgridField` the mgrid-file lane produces — no
temporary file — and it is reused across every radial stage and hot restart.
`NZETA` must divide `kp`, which is VMEC2000's `mgrid_mod` pairing rule.

Judge the grid where NESTOR uses it, on the plasma boundary:
`examples/vmex_essos_workflow.py` prints the tabulated field's median and
maximum relative error against direct Biot-Savart there. Trilinear
interpolation degrades close to a filament, so a bounding box taken from the
coils is a sampling region, not an accuracy claim.

Tabulation is host-side and keeps no derivative with respect to coil shape.
For coil-shape gradients use
{meth}`~vmex.core.mgrid.MgridField.from_parameterized_cartesian_field`, which
stays inside JAX, or the virtual-casing residual
({doc}`free-boundary`). The reverse of this seam — recovering coils from a
field grid — is a coil-design inverse problem and belongs in ESSOS, not
here.

## Walk both seams

`examples/vmex_essos_workflow.py` runs the round trip end to end: solve a
fixed boundary, hand it to ESSOS, compare the rotational transform from an
ESSOS field-line trace with the wout's `iotaf`, tabulate the ESSOS coil set,
solve a vacuum free boundary with it, and hand that equilibrium back across
the same seam.

`examples/free_boundary_essos_coils.py` is the dedicated pressure scan on
the coil-held plasma, and `examples/vmex_get_B_outside_plasma.py` queries the
exterior field with coils and virtual casing.

## Trace alpha particles

`vmex --trace` follows an ensemble of fusion-born alpha particles
(guiding-centre model, ESSOS tracer) through a converged equilibrium and
reports the exact loss fraction; the same trace is one call away in Python
via {func}`~vmex.core.tracing.trace_alphas`. It is built on the first seam
above.

### From the CLI

```console
vmex --trace wout_case.nc                  # trace, print, four figures
vmex input.case --trace                    # solve first, then trace
vmex --trace wout_case.nc --outdir figs/ \
     --trace-particles 400 --trace-tmax 1e-3 --trace-s 0.3
```

The console output gives the loss fraction, the lost / axis-termination /
solver-failure counts, and the tracing wall time. Four figures are written
next to the input (or into `--outdir`): `*_trace_trajectories.png` (sampled
orbits in 3-D over a translucent LCFS), `*_trace_vparallel.png`
(normalized parallel velocity), `*_trace_loss_fraction.png` (cumulative
loss fraction against time), and `*_trace_energy_error.png` (relative
energy error of the integrator).

Particles start on one flux surface `s` (uniform in poloidal angle, one
field period in toroidal angle, uniform pitch), at the fusion-alpha birth
energy of 3.52 MeV. An orbit counts as lost when it reaches `s >= 0.99`.
Loss fractions are physically meaningful at reactor scale — run
`vmex --scale wout_case.nc` first to put the equilibrium at ARIES-CS field
and size.

### From Python

```python
import vmex as vj

result = vj.trace_alphas("wout_case.nc", nparticles=400, tmax=1e-3)
print(result.loss_fraction, result.particles_lost)
vj.plot_tracing("wout_case.nc", result, outdir="figs")
```

`trace_alphas` accepts a path or an in-memory
{class}`~vmex.core.wout.WoutData` (written through a temporary wout file —
the route released ESSOS reads) and returns an
{class}`~vmex.core.tracing.AlphaTracingResult` with the loss-fraction time
series, per-particle loss times, trajectories in flux and Cartesian
coordinates, energies, and the counts.

For anything else ESSOS does with an equilibrium (field lines, surfaces,
`|B|` queries), use the bare `essos.fields.Vmec` from
{func}`~vmex.core.tracing.essos_vmec_field` above.

### Scope

This is the exact loss-fraction *diagnostic*. The differentiable alpha-loss
*objective* (a smooth surrogate a boundary optimization can descend) is a
separate feature that waits on the ESSOS array-based field constructor
(uwplasma/ESSOS#61) and vmex's traceable field tables. The exact loss
fraction is piecewise constant in the boundary — use it to certify, not to
optimize.
