# Solve and plot your first equilibrium

In this lesson you verify the install, solve a circular tokamak, read the
solver's output, and plot the result in real space and in Boozer coordinates.

## Check the install

```console
vmex --test
```

This solves the bundled quasi-helically-symmetric stellarator case
(`input.nfp4_QH_warm_start`) end to end, writes `wout_nfp4_QH_warm_start.nc`
and diagnostic figures under `./vmex_test/`, and prints the equivalent
manual commands. If it finishes with figures on disk, everything works.

## Run an input file

`vmex` behaves like the `xvmec2000` executable: point it at a VMEC input
file. The circular tokamak deck is in a clone of the repository (any VMEC
input you have works the same way):

```console
vmex examples/data/input.circular_tokamak
```

The deck's `NS_ARRAY` has two grids, `ns = 10` and `ns = 17`, and each prints
its own iteration table. On the first grid the initial guess has a
sign-changing Jacobian, so VMEX improves the magnetic-axis guess before
iterating, as VMEC2000 does. The finer grid starts from the interpolated
coarse solution:

```text
  NS =   17 NO. FOURIER MODES =    8 FTOLV =  1.000E-14 NITER =   1000

  ITER    FSQR      FSQZ      FSQL    RAX(v=0)    DELT       WMHD
    1  3.94E-03  1.55E-03  1.22E-06  6.132E+00  9.00E-01  6.8059E+03
  200  1.65E-11  1.30E-11  1.27E-12  6.132E+00  9.00E-01  6.8059E+03
  368  8.72E-15  8.18E-15  3.53E-15  6.132E+00  9.00E-01  6.8059E+03
```

`FSQR/FSQZ/FSQL` are the normalized force-balance residuals in the radial,
vertical, and stream-function directions; a grid stops when all three fall
below its `FTOL_ARRAY` tolerance (here 1e-14). `RAX` is the magnetic axis
position, `DELT` the time step and `WMHD` the MHD energy
({doc}`/explanation/variational-problem`, {doc}`/explanation/iteration`).
The first run of each grid also compiles it with XLA; the compiled programs
are cached, so later runs start faster.

Then the summary block:

```text
 Aspect Ratio          =       3.000000
 Plasma Volume         =     473.741011 [M**3]
 Major Radius          =       6.000000 [M]
 Minor Radius          =       2.000000 [M]
 |B| on Axis (b0)      =       5.241674 [T]
 ...
 NUMBER OF JACOBIAN RESETS =    0

 Wrote WOUT file: wout_circular_tokamak.nc
```

Zero Jacobian resets means the solver never had to back off from a
self-intersecting surface during the iterations.

## What you produced

`wout_circular_tokamak.nc` is a standard VMEC2000 output file: geometry as
Fourier tables, profiles, and scalars ({doc}`/reference/wout-file`). It loads
in simsopt, booz_xform, and any other VMEC-ecosystem tool, and in VMEX:

```python
import vmex as vj

wout = vj.read_wout("wout_circular_tokamak.nc")
print("aspect ratio:", float(wout.aspect))
print("edge iota:   ", float(wout.iotaf[-1]))
```

From Python, the same solve and file are

```python
inp = vj.VmecInput.from_file("examples/data/input.circular_tokamak")
result = vj.solve_multigrid(inp)
vj.write_wout("wout_circular_tokamak.nc", vj.wout_from_result(inp, result))
```

## Plot it

Every `wout_*.nc` (from VMEX or from VMEC2000) plots directly:

```console
vmex --plot wout_circular_tokamak.nc
vmex examples/data/input.circular_tokamak --plot     # solve, then plot
```

Five PNG files appear next to the file (or in `--outdir`):

| file | contents |
|------|----------|
| `*_summary.png` | the diagnostic summary panel |
| `*_surfaces.png` | flux-surface cross-sections at several toroidal angles |
| `*_modB.png` | `\|B\|` contours in (zeta, theta) at mid radius and boundary |
| `*_stability.png` | Mercier decomposition + frozen-equilibrium pressure scan |
| `*_boundary3d.png` | 3-D plasma boundary colored by `\|B\|` |

Look at the summary panel first. What each panel shows, and how to plot from
Python or select figures, is in {doc}`/howto/plot-diagnostics`.

## Boozer coordinates

The plain install includes the differentiable `booz_xform_jax` transform. On
the stellarator that `vmex --test` wrote:

```console
vmex vmex_test/wout_nfp4_QH_warm_start.nc --booz      # write boozmn_*.nc
vmex --plot vmex_test/boozmn_nfp4_QH_warm_start.nc    # Boozer |B| contours + spectra
```

`--plot` already performs the in-process transform needed for its Boozer
`|B|` panels; `--booz` is only needed to write a standard `boozmn_*.nc` file.
In Boozer coordinates the `|B|` contours show the symmetry class directly: for
this quasi-helically-symmetric case they run diagonally. The transform
resolution and surfaces are configurable:

```console
vmex vmex_test/wout_nfp4_QH_warm_start.nc --booz --mbooz 48 --nbooz 48 \
     --booz-surfaces "0.25, 0.5, 1.0"
```

## The same steps as scripts

`examples/fixed_boundary_run.py` reads a deck (the LI383 stellarator at low
resolution), solves it on its multigrid ladder, and writes and plots the wout:

```{literalinclude} ../../examples/fixed_boundary_run.py
:language: python
```

`examples/plot_and_boozer.py` produces every `plot_wout` figure and the
Boozer `|B|` spectrum on the last closed flux surface:

```{literalinclude} ../../examples/plot_and_boozer.py
:language: python
```

Next: {doc}`first-gradient` differentiates an equilibrium.
