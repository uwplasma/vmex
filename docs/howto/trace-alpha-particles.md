# Trace alpha particles

`vmex --trace` follows fusion-born 3.52 MeV alphas through an equilibrium and
reports losses through the last closed flux surface. Install `vmex[coils]` (ESSOS ≥0.19.2).

## Run it

```console
vmex wout_case.nc --trace          # 1,000 alphas, 10 ms, ARIES-CS size
vmex input.case --trace            # solve first, then trace
```

The CLI reports progress, scaling, timestep, mode count, timing and the
binomial sampling error:

```text
 Loss fraction: 12.30% ± 1.04% (123 of 1000 particles lost)
```

## Production runs: particles, time and other flags

| flag | default | what it sets | cost |
|---|---|---|---|
| `--trace-particles N` | 1000 | ensemble size; the error is `sqrt(f (1 - f) / N)` | linear in `N` |
| `--trace-tmax T` | `1e-2` | horizon in seconds | linear in `T` |
| `--trace-timestep DT` | size-scaled step | RK4 step in seconds | `1 / DT` |
| `--trace-birth surface\|volume` | `surface` | births on `--trace-s` or through the volume at the D-T fusion rate | none |
| `--trace-s S` | 0.25 | birth surface `s = psi / psi_b` | none |
| `--collisional` | off | Monte Carlo collisions on electrons, D and T | about none |
| `--trace-ne0 N0`, `--trace-te0 T0` | `4e20`, `12` | on-axis electron density [m^-3] and temperature [keV] | none |
| `--trace-no-scale` | off | trace the equilibrium at its own size and field | none |
| `--scale-target axis` | `volavgB` | ARIES-CS convention of the in-memory scaling | none |
| `--mbooz M`, `--nbooz N` | 32, 32 | Boozer resolution of the traced field | small |
| `--trace-seed K` | 42 | births and collision noise | none |
| `--trace-times K` | 1000 | samples of the loss-fraction curve | none |
| `--trace-mode-cut C` | `1e-4` | drop Boozer `|B|` modes below `C B00` | about `1 / C` in modes |

Cost grows with `particles × tmax / timestep`; use a GPU for large ensembles.

```console
vmex wout_case.nc --trace --trace-particles 5000 --trace-tmax 0.1
vmex wout_case.nc --trace --trace-birth volume --collisional --trace-tmax 0.2
vmex wout_case.nc --trace --trace-particles 200 --trace-tmax 1e-3   # a quick look
```

Ten-millisecond traces measure prompt losses. For slowing-down losses use
`--collisional --trace-tmax 0.2`; compare matching horizons and birth distributions.

## What is traced

- **Scale:** ARIES-CS `<B> = 5.8646 T`, `a = 1.7044 m`, applied in memory
  (see {doc}`scale-a-configuration`).
- **Field:** Boozer `|B|` is splined in `sqrt(s)`; `iota`, `G` and `I` in `s`.
- **Orbits:** collisionless `K=0` guiding centres, fixed-step RK4 in the regular
  chart `sqrt(s) (cos theta, sin theta)`; crossing `s=1` counts as lost.
- **Births:** uniform pitch `v_par/v` in `[-1, 1)` and angles weighted by
  `|G + iota I| / B^2`. Volume births also use the Bosch-Hale D-T reaction rate.
- **Profiles:** `n_e = n_e0 (1 - s^5)`, `n_D = n_T = n_e/2`,
  `T_e = T_i = T_0 (1 - s)` (Landreman, Buller & Drevlak, PoP 29, 082501, 2022).
- **Collisions:** Ito Euler-Maruyama pitch scattering, slowing down and energy
  diffusion on electrons, D and T after each orbit step. Below 1.5 times the
  local temperature, an alpha counts as thermalised and confined.

The tracer rejects failed paths and numerical orbit-energy drift above `1e-3`.
Reduce `--trace-timestep` when this check fails; low energy drift alone does not
establish convergence of loss labels.

## Outputs

Next to the input (or in `--outdir`):

- `*_trace.json` holds the counts, loss fraction and error, scaling factors,
  step, Boozer modes, devices, versions and timing.
- `*_trace.npz` holds the loss-fraction curve, the loss and thermalisation
  times, the births `(s, theta_B, zeta_B, v_par/v)` and the final states.
  This is enough to replot or compare runs.
- `*_trace.png` shows cumulative losses, boundary loss locations, birth pitch,
  loss time against pitch, radial losses (or a loss-time histogram), and `iota(s)`.
- `*_trace_3d.png` shows boundary loss locations coloured by loss time.

The shaded loss-curve band is computed separately at each time as
`f(t) ± sqrt(f(t) [1 - f(t)] / N)` for `N` independent births. This is a
pointwise one-standard-error sampling band, not a confidence band for the
whole curve or an estimate of timestep, field or mode-cut error. It shrinks
to zero when no particle has been lost, so use a binomial confidence interval
and more births to assess rare losses.
At the 1,000-birth default, 10 losses mean 1% with a 95% Wilson interval of
about 0.54–1.83%; zero losses still allow up to about 0.38% at that level.
Use several thousand births for small fractions, and compare candidate fields
using the same birth sample.

## Cost and mode-cut accuracy

Cost scales with particles, orbit steps and retained harmonics. The `1e-4`
cut is a starting point: tighten both timestep and spectrum on the intended equilibrium.

The [34-equilibrium audit](../explanation/validation.md#alpha-tracing-accuracy)
checks field derivatives and individual losses. Small energy drift and similar
loss counts do not establish trajectory convergence; compare the same birth sample.

(w7-x-convergence)=

## W7-X convergence

Standard and high-mirror W7-X use 256 common births, `s=0.25`, 5 ms, and
RK4 steps of `3.125e-8 s`. The [measurement record](../../benchmarks/trace_accuracy.json)
contains source, input and birth hashes. Runs use alpha mass
`6.69509884346e-27 kg`; `1e-5` is a tighter reference, not an exact solution.

| cut | standard lost | different labels | cold / warm [s] | high-mirror lost | different labels | cold / warm [s] |
|---|---:|---:|---:|---:|---:|---:|
| `1e-5` | 37 | 0 | 62.86 / 56.87 | 53 | 0 | 62.55 / 55.85 |
| `6e-5` | 40 | 11 | 43.90 / 37.61 | 52 | 5 | 45.56 / 39.14 |
| `8e-5` | 41 | 10 | 41.55 / 35.30 | 53 | 8 | 42.37 / 35.29 |
| `1e-4` | 36 | 15 | 40.21 / 34.17 | 54 | 7 | 40.99 / 34.08 |
| `2e-4` | 39 | 10 | 37.11 / 31.03 | 52 | 5 | 37.26 / 30.90 |
| `3e-4` | 37 | 12 | 37.23 / 30.80 | 49 | 8 | 37.69 / 30.70 |
| `5e-4` | 32 | 11 | 34.59 / 28.37 | 54 | 7 | 35.00 / 28.40 |
| `6e-4` | 33 | 12 | 34.31 / 28.23 | 50 | 7 | 34.91 / 28.24 |
| `8e-4` | 30 | 11 | 33.00 / 26.90 | 49 | 8 | 33.85 / 27.14 |
| `1e-3` | 27 | 14 | 33.23 / 27.06 | 50 | 5 | 33.78 / 26.94 |

All measured energy errors are below `6.1e-5`. Tightening to `6e-5` or `8e-5`
does not consistently improve label agreement; similar total counts can hide different losses.
The shaded band is the sampling standard error defined above.

Births are the first 256 of 4,096 sampled at cut `1e-6`, with seed 42.
For each cold measurement, use a fresh process with compilation caches disabled and one cut:

```console
JAX_ENABLE_COMPILATION_CACHE=0 python benchmarks/trace_mode_cut.py WOUT --particles 256 --birth-samples 4096 --s 0.25 --birth-cut 1e-6 --tmax 0.005 --step-factor 0.25 --orbit-cuts 1e-4 --repeats 1 --out cuts.json
```

With 1,024 common standard-W7-X births over 10 ms, timestep refinement gives:

| method | timestep [s] | lost / 1024 | maximum relative energy drift |
|---|---:|---:|---:|
| ESSOS RK4 | `1.25e-7` | 250 | `6.64e-2` |
| ESSOS RK4 | `6.25e-8` | 221 | `3.47e-3` |
| ESSOS RK4 | `3.125e-8` | 218 | `1.21e-4` |
| SIMPLE midpoint, 256 steps/transit | adaptive macrosteps | 220 | `8.37e-4`, 401 saved states |

The first two RK4 settings fail the `1e-3` energy check. The refined RK4 and
SIMPLE totals are close, but 46 particle labels differ; field interpolation,
spectrum and timestep convergence remain necessary. SIMPLE's birth speed differs
by `1.39e-5` relatively because of its legacy constants. Its repeated eight-thread
CPU traces take 765.89/762.94 s; these are different hardware from the GPU cutoff study.

## Cross-code orbit checks

SIMPLE, ESSOS, FIRM3D, CATAPULT and DESC agree on 55 losses among 64 seed-field births;
SIMSOPT resolves 58 paths with 50 losses and six unresolved axis stops.
The [validation record](../explanation/validation.md) gives field discrepancies,
energy diagnostics, output policies and CPU/GPU timings.

[`trace_cross_code.py`](../../benchmarks/trace_cross_code.py) checks common births,
field agreement, exits and energy drift. The seed uses `ns=31`, `mpol=ntor=5`,
3.52 MeV alphas at `s=0.283333`, and a 2 ms horizon.

The comparison WOUT comes from the same unoptimized seed as the direct-loss
optimization example. Recreate it without running the optimizer:

```python
from dataclasses import replace
import vmex as vj
from vmex import optimize as opt
from vmex.core.scaling import aries_cs_scales, scale_wout

inp = vj.VmecInput.from_file("examples/data/input.minimal_seed_nfp2")
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor, 1] = zbs[inp.ntor, 1] = 0.17
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -0.03, 0.03
seed = replace(inp, rbc=rbc, zbs=zbs, delt=0.5).change_resolution(
    mpol=5, ntor=5, ntheta=16, nzeta=14)
equilibrium = opt.solve_equilibrium(seed)
b, r = aries_cs_scales(equilibrium.wout)
vj.write_wout("wout_alpha_seed_reactor.nc", scale_wout(equilibrium.wout, b_scale=b, r_scale=r))
```

This yields `Aminor_p=1.7044 m`, `volavgB=5.8646 T`; the benchmark WOUT's
SHA-256 is `5caaaacd2809302cf9a703a9c31fff433a51a1cfc5b543b6c45fcd024964abd2`.
With the corrected SIMSOPT field and a built SIMPLE executable, run
`python benchmarks/trace_cross_code.py wout_alpha_seed_reactor.nc --simple
PATH_TO_SIMPLE/simple.x --output cross_code.json` for the 64-birth CPU check.

The [cross-code record](../explanation/validation.md#cross-code-alpha-tracing)
reports loss agreement, energy diagnostics and CPU/GPU timings. Check the source
revisions and settings in `benchmarks/trace_accuracy.json` when reproducing it.

## From Python

```python
import vmex as vj

result = vj.trace_alphas("wout_case.nc", nparticles=400, tmax=1e-3,
                         birth="volume", collisions=True)
print(result.loss_fraction, result.loss_fraction_sigma)
vj.plot_tracing(result, "figs", name="case")
```

`trace_alphas` accepts a path or {class}`~vmex.core.wout.WoutData` and returns
{class}`~vmex.core.tracing.AlphaTracingResult`. For differentiable optimization,
see [`alpha_particle_optimization.py` (PR #515)](https://github.com/uwplasma/vmex/pull/515):
a smooth loss surrogate supplies derivatives; independent hard-loss traces validate the result.
