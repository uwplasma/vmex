# Plot results and use them downstream

`vmex --plot` renders the standard figure set from any `wout_*.nc`,
`boozmn_*.nc`, or mirror `mout_*.nc` file; the same figures are one call
away in Python via {func}`~vmex.core.plotting.plot_wout` /
{func}`~vmex.core.plotting.plot_boozmn`. VMEX wout files follow the VMEC2000
netCDF schema (names, dimensions, dtypes, units), so simsopt, booz_xform and
other VMEC-ecosystem tools read them unchanged.

## From the CLI

```console
vmex --plot wout_case.nc               # six wout figures
vmex input.case --plot                 # solve, then plot
vmex --plot boozmn_case.nc             # Boozer contours + spectra
vmex --plot mout_case.nc               # straight-axis mirror figures
vmex --plot wout_case.nc --outdir figs/
```

The wout set always includes `*_surfaces.png` (cross-sections at several
zeta over one field period, axis marked), `*_modB.png` (`|B|` contours at
mid radius and boundary), `*_profiles.png` (iota/pressure/current plus the
`fsqt` force-residual trace), `*_boundary3d.png` (3-D boundary colored by
`|B|`), `*_stability.png` (Mercier terms and a pressure scan), and
`*_summary.png`. Both symmetric and `lasym` equilibria are supported — the
sine/cosine partner tables are included whenever present. All figures use
the Agg backend at dpi >= 200, so plotting works on headless machines.

## From Python

```python
import vmex as vj

paths = vj.plot_wout("wout_case.nc", outdir="figs")        # dict[str, Path]
paths = vj.plot_wout(wout_data, outdir="figs",
                     which=("summary", "profiles"))        # select figures
```

`plot_wout` accepts a path or an in-memory
{class}`~vmex.core.wout.WoutData`, and `which=` selects a subset of
`("summary", "surfaces", "modB", "profiles", "stability", "3d")`.
Per-figure helpers
({func}`~vmex.core.plotting.plot_summary`,
{func}`~vmex.core.plotting.plot_stability`, ...) return single figures for
embedding in your own scripts; `examples/plot_and_boozer.py` is the worked
version.

## The summary figure

`*_summary.png` is a publication-style diagnostic set: rotational transform
(full mesh) with the parallel bootstrap current
$\langle \mathbf{J}\cdot\mathbf{B} \rangle$ on its right axis, a combined
confinement panel — pressure on the left axis, with the effective ripple
$\epsilon_{\rm eff}^{3/2}$ (NEO_JAX at the bounded
{func}`~vmex.core.neoclassical.diagnostic_neo_config` resolution) and the
fast-ion proxy $\Gamma_c$
({func}`~vmex.core.gammac.gamma_c_from_wout` at a compact radial-trend
sampling) sharing one dimensionless right axis — the force error
$\langle|\mathbf J\times\mathbf B-\nabla p|\rangle_s/\langle|\nabla(B^2/2\mu_0)|\rangle_V$,
Mercier `DMerc` and the Glasser
resistive-interchange $D_R$ with $V''(s)$ on a color-matched right axis,
a 3-D LCFS, and the second adiabatic invariant in the polar disk
$x=s\cos\alpha$, $y=s\sin\alpha$. Concentric $J$ contours diagnose
alpha-independence. `|B|` in Boozer coordinates appears at mid radius and on
the LCFS as unfilled jet contours with a field line of slope iota. The Boozer
transform runs in-process, so `--plot` needs no separate `--booz` pass;
`--booz` is for writing a reusable `boozmn_*.nc`. The plotted $D_R$ comes from
a host-side reconstruction of the WOUT file, which carries the sine-parity
partner tables and so covers both symmetry classes. The reconstruction checks
itself by reproducing the stored `DMerc` profile from the same integrals; on
mismatch the curve is omitted with a panel note rather than drawn
unvalidated. The force error is rebuilt from the WOUT tables on the interior
surfaces and divided by the volume average of $|\nabla(B^2/2\mu_0)|$ over
$V$: $0.1\le s\le 0.99$, the normalization DESC and the polish certificate
report; the scalar card gives the volume average of the ratio. It does not
saturate and stays defined in vacuum. A converged low-resolution deck (for
example `mpol = ntor = 2`) can still read high: that is spectral truncation
error, which `equif` could not show, not a solver failure. WOUT's `equif`
({func}`~vmex.core.postprocess.force_balance`) is not plotted: it is bounded
by 1 and equals 1 on every surface of a currentless vacuum, however well
converged.
The confinement panel's two right-axis profiles are radial *trends*, not
transport numbers: both diagnostics run at bounded summary resolution, they
share the summary's one in-process Boozer transform where the mathematics is
common (NEO consumes it directly; $\Gamma_c$ keeps its validated real-space
field-line route), and the computed profiles are cached per in-memory WOUT so
repeated summary generation does not recompute or recompile them.  A
diagnostic that is unavailable (NEO_JAX not installed, `lasym` Boozer
tables, a failed evaluation) is dropped from the panel with the reason
recorded — never drawn as zero.  Use
{func}`vmex.core.neoclassical.epsilon_effective_from_wout` with an explicit
`neo_jax.NeoConfig` when a publication effective-ripple calculation is
wanted; the library default no longer clears process-wide JAX caches
(`clear_jax_caches=False`) — the CLI releases the plot-only executables
itself after all requested diagnostics.
The two stability indices and $V''(s)$ use separate scales whose zero levels
are aligned; $V''(s)<0$ denotes a magnetic well. Their legend sits below the
panel so it cannot hide a curve.

`*_stability.png` first separates `DMerc` into shear, well, current, and
geodesic terms. Its second panel rescales the WOUT pressure profile and plots
the worst frozen-equilibrium margins, $\min_s D_{Merc}$ and $-\max_s D_R$,
against trial volume-average beta; positive is favorable. A vacuum WOUT has no
pressure shape, so this panel states that it uses $p(s)\propto1-s$. This scan
isolates the explicit pressure-gradient drive at fixed geometry and current;
finite-pressure stability must still be certified by re-solving each point.

## Boozer figures

`vmex --plot boozmn_case.nc` renders `|B|` contours on the transformed
surfaces, the Boozer spectrum, and mode-amplitude profiles
({func}`~vmex.core.plotting.plot_boozmn_modB`,
{func}`~vmex.core.plotting.plot_boozmn_spectrum`,
{func}`~vmex.core.plotting.plot_boozmn_mode_profiles`). Producing the
`boozmn_*.nc` file in the first place is {doc}`/tutorials/first-equilibrium`.

## Mirror figures

`vmex --plot mout_case.nc` renders the open-mirror set: horizontal 3D, coil
curves, cap-to-cap field lines, `|B|`, pressure, cross-sections, and
residual histories ({doc}`/reference/wout-file`).

## Read a wout in Python

```python
import vmex as vj

wout = vj.read_wout("wout_case.nc")
print("aspect ratio:", float(wout.aspect))
print("edge iota:   ", float(wout.iotaf[-1]))
print("beta total:  ", float(wout.betatotal))
```

{class}`~vmex.core.wout.WoutData` exposes every schema variable as an
attribute: scalars (`aspect`, `b0`, `volume_p`, ...), radial profiles
(`iotaf`, `presf`, `jcurv`, `DMerc`, ...), and the Fourier tables (`rmnc`,
`zmns`, `lmns`, `bmnc`, ...), with the wout mesh conventions (`lmns` is
half-mesh, `bsubsmns` full-mesh, `presf` in Pa). Fields on a grid follow from
the tables, for example `|B|` from `bmnc` with phase `xm*theta - xn*zeta`
(`xn` already includes `nfp`). A solve result becomes a wout in one call,
`vj.wout_from_result(inp, result)`. The variable list with mesh and unit notes
is {doc}`/reference/wout-file`.

## Load in simsopt

```python
from simsopt.mhd import Vmec

vmec = Vmec("wout_case.nc")            # wout mode: no Fortran VMEC needed
print(vmec.aspect(), vmec.mean_iota())
```

simsopt's `Vmec` wout-file mode, `SurfaceRZFourier.from_wout`, and its
Boozer and quasisymmetry diagnostics read VMEX files; the per-variable parity
against VMEC2000 output is in {doc}`/explanation/validation`.

## Run the Boozer transform from Python

```python
import vmex as vj

boozmn_path = vj.run_booz_xform("wout_case.nc", mbooz=48, nbooz=48)
vj.plot_boozmn(boozmn_path, outdir=".")
```

The resulting `boozmn_*.nc` is the standard format that the C++
`booz_xform` tooling also reads.

## What downstream readers should know

- **VMEX extension variables.** `vmex_diagnostics_schema = 1` and
  `vmex_trapped_fraction` (the effective trapped-particle fraction on the
  full mesh) are extra names VMEC2000 readers may ignore.
- **Declared but fill-valued variables.** A schema variable may be
  fill-valued where its producer is not implemented; the disclosed cases are
  in {doc}`/reference/wout-file` and
  {doc}`/reference/vmec2000-compatibility`.
