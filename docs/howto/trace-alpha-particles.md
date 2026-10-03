# Trace alpha particles

`vmex --trace` follows fusion-born 3.52 MeV alphas through an equilibrium and
reports the fraction lost through the last closed flux surface. It needs the
`coils` extra (`pip install "vmex[coils]"`, ESSOS 0.19.4 or later).

## Run it

```console
vmex wout_case.nc --trace          # 500 alphas, 10 ms, ARIES-CS size
vmex input.case --trace            # solve first, then trace
```

The default run takes under a minute on a 10-core laptop (see Cost below). While it runs it reports, on
stderr, the share of `tmax` traced, the elapsed time and an estimate of the time left
(ESSOS 0.19.4 and later). It then prints the scaling factors, the step, the number of Boozer
modes, the wall time split into compile and run, and the loss fraction with
its binomial error:

```text
 Loss fraction: 12.30% ± 1.04% (123 of 1000 particles lost)
```

## Production runs: particles, time and other flags

| flag | default | what it sets | cost |
|---|---|---|---|
| `--trace-particles N` | 500 | ensemble size; the error is `sqrt(f (1 - f) / N)` | linear in `N` |
| `--trace-tmax T` | `1e-2` | horizon in seconds | linear in `T` |
| `--trace-method rk4\|dopri5\|dopri8` | `rk4` | fixed-step integrator; alternatives require ESSOS method support | field dependent |
| `--trace-timestep DT` | size-scaled step | orbit step in seconds | `1 / DT` |
| `--trace-compact`, `--no-trace-compact` | on when supported | compact early losses | loss dependent |
| `--trace-birth surface\|volume` | `surface` | births on `--trace-s` or through the volume at the D-T fusion rate | none |
| `--trace-s S` | 0.25 | birth surface `s = psi / psi_b` | none |
| `--collisional` | off | Monte Carlo collisions on electrons, D and T | about none |
| `--trace-ne0 N0`, `--trace-te0 T0` | `4e20`, `12` | on-axis electron density [m^-3] and temperature [keV] | none |
| `--trace-no-scale` | off | trace the equilibrium at its own size and field | none |
| `--scale-target axis` | `volavgB` | ARIES-CS convention of the in-memory scaling | none |
| `--mbooz M`, `--nbooz N` | 32, 32 | Boozer resolution of the traced field | small |
| `--trace-seed K` | 42 | births and collision noise | none |
| `--trace-times K` | 1000 | samples of the loss-fraction curve | none |
| `--trace-mode-cut C` | `1e-4` | drop modes below `C` times the largest cosine/sine amplitude | about `1 / C` in modes |

The wall time is `particles × tmax / timestep` times a per-step cost. For
example, going from the default to 5000 alphas over 0.1 s costs 50 times the
default, about 25 min on the same laptop. Run that on a workstation or a GPU.

```console
vmex wout_case.nc --trace --trace-particles 5000 --trace-tmax 0.1
vmex wout_case.nc --trace --trace-birth volume --collisional --trace-tmax 0.2
vmex wout_case.nc --trace --trace-particles 200 --trace-tmax 1e-3   # a quick look
```

Losses from 10 ms are prompt losses only. Published reactor numbers are
slowing-down losses over about 0.2 s: `--collisional --trace-tmax 0.2`. Do not
compare a 10 ms number against a 0.2 s table. For reference, Paul et al.
(NF 62, 126054, 2022, Table 2) report 2470 of 10^4 ARIES-CS alphas born on
s = 0.3 lost within 0.2 s.

## What is traced

- **Scale.** Loss fractions are physical only at reactor size, so the
  equilibrium is first scaled in memory to ARIES-CS size: `<B> = 5.8646 T`
  and `a = 1.7044 m` (the `--scale` rule, see {doc}`scale-a-configuration`).
- **Field.** `booz_xform_jax` transforms every surface to Boozer coordinates.
  The cosine and sine `|B|` spectra are cut below `1e-4` of the largest
  combined amplitude and splined in `sqrt(s)`; `iota`, `G` and `I` are splined in `s`.
- **Orbits.** The guiding-centre equations in Boozer coordinates (White; the
  `K = 0` form of SIMSOPT) are integrated with fixed-step RK4 in the chart
  `sqrt(s) (cos theta, sin theta)`, which is regular on the magnetic axis, so
  no orbit stops there. An alpha is lost when it reaches `s = 1`.
- **Births.** Pitch `v_par / v` is uniform in `[-1, 1)`. The angles follow
  the Boozer Jacobian `(G + iota I) / B^2`. With `--trace-birth volume`, `s`
  follows the D-T rate `n_D n_T <sigma v>(T)`, weighted by the volume
  element (Bosch-Hale reactivity).
- **Profiles** (volume births and `--collisional`; Landreman, Buller &
  Drevlak, PoP 29, 082501, 2022): `n_e = n_e0 (1 - s^5)`,
  `n_D = n_T = n_e / 2`, and `T_e = T_i = T_0 (1 - s)`.
- **Collisions.** After every step, a Monte Carlo operator applies
  pitch-angle scattering, slowing down and energy diffusion on electrons, D
  and T (Ito Euler-Maruyama; Boozer & Kuo-Petravic 1981; ESSOS collision
  rates). An alpha below 1.5 times the local temperature is thermalised and
  counts as confined. The ESSOS tests check the slowing down against the
  Stix time and the pitch-angle decay against `nu_D`.

## Outputs

Next to the input (or in `--outdir`):

- `*_trace.json` holds the counts, loss fraction and error, scaling factors,
  step, Boozer modes, devices, versions and timing.
- `*_trace.npz` holds the loss-fraction curve, the loss and thermalisation
  times, the births `(s, theta_B, zeta_B, v_par/v)` and the final states.
  This is enough to replot or compare runs.
- `*_trace.png` has six panels:
  - the cumulative loss fraction against log time, with its 1σ band;
  - a heatmap of the loss locations on the boundary in `(zeta_B, theta_B)`;
  - the birth pitch of lost and confined alphas;
  - loss time against birth pitch;
  - loss fraction against birth `s` (volume births), or a loss-time
    histogram (surface births);
  - `iota(s)` with the low-order rationals `n N_fp / m`, where orbit
    resonances sit.
- `*_trace_3d.png` shows the loss locations on the 3-D boundary, coloured by
  loss time.

## Cost

On an Apple M3 Max laptop (10 performance cores, load average 6-9), the
default ARIES-CS run (`wout_n3are_R7.75B5.7.nc`, 1000 alphas, 10 ms, 41
Boozer modes) takes 28 s: 25 s of tracing, of which 1.9 s is compilation.
It loses 12.3 % ± 1.0 %. The defaults are now 500 alphas (± 1.5 %) and a
mode cut of 1e-4 (see the convergence section). `--trace` gives JAX one CPU
device per usable core (on Linux, the cores the process may run on). On Apple
silicon it uses only the performance cores unless there are at least as many
efficiency cores. On an M4 (4 + 6) all 10 cores trace 1.7x faster than the 4
performance cores. A device count in `XLA_FLAGS` or `JAX_NUM_CPU_DEVICES`
wins. The Boozer transform takes about 2 s.

**CPU or GPU.** At the default size a laptop CPU is faster than a GPU. The
CPU cost grows linearly with the number of alphas. An RTX A4000 took 70-130 s
at any count from 250 to 8000 alphas (1e-3 cut): the 80 000 RK4 steps run one
after another, and each step is too little work to fill the GPU. It beat the
M4 only from about 4000 alphas (89 s against 193 s). Use a GPU for
`--trace-particles 5000` and up.

## Convergence of the defaults

These runs use ARIES-CS (`wout_n3are_R7.75B5.7.nc`) at reactor scale, with
the same alphas launched from s = 0.25 and traced for 10 ms.

| alphas | step [s] | mode cut | modes | lost | vs reference (σ) |
|---|---|---|---|---|---|
| 4000 | 6.25e-8 | 1e-4 | 135 | 501 (12.5 %) | reference |
| 4000 | 1.25e-7 | 1e-3 | 41 | 489 (12.2 %) | -0.6 |
| 4000 | 2.5e-7 | 1e-3 | 41 | 501 | 0.0 |
| 1000 | 6.25e-8 | 1e-4 | 135 | 121 | reference |
| 1000 | 6.25e-8 | 1e-5 | 382 | 131 | +1.0 |
| 1000 | 1.25e-7 | 1e-4 | 135 | 115 | -0.6 |
| 1000 | 1.25e-7 | 1e-3 | 41 | 125 | +0.4 |
| 1000 | 2.5e-7 | 1e-3 | 41 | 132 | +1.1 |

Halving the default step (and refining the mode cut tenfold) changes the
loss fraction by 0.6σ at 4000 alphas. That is within the 1σ gate G4 of plan
section T. So is refining the cut a further hundredfold at 1000 alphas. At
2.5e-7 s the loss fraction still agrees, but the RK4 energy error grows from
1e-3 to 2e-2, so the default keeps 1.25e-7 s. The per-particle
lost/confined labels agree only 90-92 % between any two of these runs.
Over 10 ms these orbits are chaotic, so the fraction converges while
individual orbits do not. Over 2 ms, 2000 alphas give 54 and 55 losses at
6.25e-8 s and 3.125e-8 s, 2.7 %. The earlier VMEC-coordinate tracer gave
2.5 % ± 1.1 % at its converged step.

### The mode cut across geometries

ARIES-CS alone does not settle the cut. These runs trace 1000 alphas for
10 ms through six equilibria at cuts of 1e-3 and 1e-4, with the same births
at both. The cut is relative to the largest `|B|` amplitude, which is `B00` in
every case.

| equilibrium | modes at 1e-3 / 1e-4 | lost at 1e-3 / 1e-4 | difference | same label |
|---|---|---|---|---|
| ARIES-CS | 41 / 135 | 12.3 / 12.3 % | 0.0σ | 91 % |
| Landreman-Paul QA | 3 / 16 | **0.0 / 0.7 %** | **-2.7σ** | 99 % |
| Landreman-Paul QH | 4 / 14 | 0.0 / 0.0 % | | 100 % |
| HSX | 40 / 159 | **9.6 / 12.7 %** | **-2.2σ** | 86 % |
| W7-X (d23p4_tm, beta 5 %) | 29 / 78 | 1.9 / 2.1 % | -0.3σ | 99 % |
| QI, 2 field periods | 30 / 172 | 3.2 / 2.9 % | +0.4σ | 98 % |

A cut of 1e-3 misses the losses in the precise QA and in HSX. A good
quasisymmetric field has all of its symmetry-breaking modes below `1e-3 B00`,
and those are the modes that lose alphas. At 1e-5, Landreman-Paul QA still
loses 0.7 % (96 modes) and HSX 12.0 % (342 modes, within 1σ of 1e-4). The
default is therefore 1e-4, which takes about 3 times as long as 1e-3 on
ARIES-CS. `--trace-mode-cut 1e-3` is a quick look for configurations far from
quasisymmetry.

## Against SIMPLE and SIMSOPT

`benchmarks/trace_cross_code.py` traces the same 1000 alphas with three
codes. The equilibrium is ARIES-CS (`wout_n3are_R7.75B5.7.nc`, unscaled).
The alphas are born on s = 0.247 with the `--trace` births, and all three
codes get the same positions, pitches and 3.52 MeV energy. Each code runs
for 10 ms on the same 8 cores (`taskset`) of a shared 36-core x86_64
workstation, under a load average of 28-50 from other jobs. The runtime
leaves out JAX compilation (35 s), the field set-up of SIMPLE and the
interpolation tables of SIMSOPT.

| code | integrator | lost | loss fraction | runtime |
|---|---|---|---|---|
| VMEX `--trace` (ESSOS Boozer) | RK4, 1.25e-7 s | 128 | 12.8 % ± 1.1 % | 146 s |
| SIMPLE | symplectic Euler, defaults, all orbits traced | 124 | 12.4 % ± 1.0 % | 556 s |
| SIMSOPT `trace_particles_boozer` | RK45, tol 1e-9, `gc_noK` | 119 | 11.9 % ± 1.0 % | 1079 s |

The three loss fractions agree within 0.6σ. SIMSOPT uses a `booz_xform`
field with the same 32 × 32 resolution. SIMPLE reads its starts in VMEC
angles. The Boozer births are mapped with `nu` and `lambda`, and the
mapping agrees to 0.13 mm in `R, Z` and 0.2 % in SIMPLE's own `|B|`.
`docs/_static/figures/sources/make_trace_figures.py` plots the record
(`benchmarks/trace_cross_code.json`).

## From Python

```python
import vmex as vj

result = vj.trace_alphas("wout_case.nc", nparticles=400, tmax=1e-3,
                         birth="volume", collisions=True)
print(result.loss_fraction, result.loss_fraction_sigma)
vj.plot_tracing(result, "figs", name="case")
```

`trace_alphas` accepts a path or an in-memory
{class}`~vmex.core.wout.WoutData` and returns an
{class}`~vmex.core.tracing.AlphaTracingResult`. The exact loss fraction is
piecewise constant in the boundary, so use it to certify a design, not as an
optimization objective.
