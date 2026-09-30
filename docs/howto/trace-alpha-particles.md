# Trace alpha particles

`vmex --trace` follows fusion-born 3.52 MeV alphas through an equilibrium and
reports the fraction lost through the last closed flux surface. It needs the
`coils` extra (`pip install "vmex[coils]"`, ESSOS 0.19.2 or later).

## Run it

```console
vmex wout_case.nc --trace          # 500 alphas, 10 ms, ARIES-CS size
vmex input.case --trace            # solve first, then trace
```

The default run takes under a minute on a 10-core laptop (see Cost below). While it runs it reports, on
stderr, the share of `tmax` traced, the elapsed time and an estimate of the time left
(ESSOS 0.19.2 and later). It then prints the scaling factors, the step, the number of Boozer
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
| `--trace-timestep DT` | converged step (below) | RK4 step in seconds | `1 / DT` |
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
  The `|B|` spectrum, cut at modes below `1e-4` of the largest amplitude (`B00`), is
  splined in `sqrt(s)`, and `iota`, `G` and `I` are splined in `s`.
- **Orbits.** The guiding-centre equations in Boozer coordinates (White; the
  `K = 0` form of SIMSOPT) are integrated with fixed-step RK4 in the chart
  `sqrt(s) (cos theta, sin theta)`, which is regular on the magnetic axis, so
  no orbit stops there. An alpha is lost when it reaches `s = 1`.
- **Births.** Pitch `v_par / v` is uniform in `[-1, 1)`. The angles follow
  the positive Boozer volume measure `|G + iota I| / B^2`. With `--trace-birth volume`, `s`
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

## Cost and mode-cut accuracy

Tracing cost grows with the number of particles, integration steps and retained
Boozer modes. CPU/GPU crossover depends on hardware, device count and the
number of saved orbit states; benchmark the intended workload. The ESSOS
compiled-kernel change in [PR #95](https://github.com/uwplasma/ESSOS/pull/95)
removes recompilation on repeated calls and improves the matched GTX TITAN X
GPU workload by about 17%. The independent radial lookup change in
[ESSOS PR #98](https://github.com/uwplasma/ESSOS/pull/98) improves its warmed
tracing kernel a further 3.86-fold on the 30-knot seed case, with bitwise
identical orbit output.

The VMEC toroidal-flux sign was corrected in [VMEX PR #517](https://github.com/uwplasma/vmex/pull/517).
Older loss tables and figures made with the opposite sign are withdrawn:
conserved energy and a plausible total loss fraction did not reveal the
reversed radial drift. The current default cut is `1e-4`. The
[34-equilibrium spectral audit](../explanation/validation.md) compares all
seven requested cuts on three surfaces per equilibrium; angular and radial
field derivatives deteriorate much faster than `|B|` itself.

On the corrected Landreman-Paul QA field, 1,000 common births at `s = 0.3`
were traced for 10 ms on eight Apple M2 CPU devices. Times exclude transform
and compilation and are medians of three warmed calls:

| mode cut | modes | lost / 1000 | baseline losses recovered | labels matching `1e-4` | trace [s] |
|---|---:|---:|---:|---:|---:|
| `1e-4` | 16 | 16 | 16/16 | 100% | 34.74 |
| `2e-4` | 7 | 14 | 13/16 | 99.60% | 23.45 |
| `3e-4` | 3 | 6 | 6/16 | 99.00% | 9.13 |
| `5e-4` | 3 | 6 | 6/16 | 99.00% | 9.08 |
| `6e-4` | 3 | 6 | 6/16 | 99.00% | 10.08 |
| `8e-4` | 3 | 6 | 6/16 | 99.00% | 10.24 |
| `1e-3` | 3 | 6 | 6/16 | 99.00% | 9.34 |

The last five cuts retain the same three modes; their time differences are
measurement noise. The 99% overall label agreement at `3e-4` hides ten
missing losses out of 16. Halving the QA timestep at `1e-4` preserves all
1,000 labels, including the 16 losses. On ARIES-CS, `2e-4` changes 31 of 512 labels over
2 ms relative to `1e-4`. On HSX, a refined 0.5 ms, 500-birth run at `2e-4`
has the same 23 total losses as `1e-4` but recovers only 16 of those 23;
warm CPU time falls from 39.97 to 31.01 s. The [full seven-cut HSX table](../explanation/validation.md)
reports the remaining times and labels with its timestep uncertainty.
Moreover, `1e-4` and a tighter `1e-5` HSX spectrum each lose 23 particles,
but share only 15 loss IDs at the refined step. Even the current default is
not cutoff-converged for HSX losses. Thus there is no geometry-independent faster cut:
converge the timestep, inspect energy, and check individual labels on the
intended equilibrium before relaxing `--trace-mode-cut`.

## Cross-code orbit checks

The earlier 1,000-particle ARIES-CS comparison with SIMPLE and SIMSOPT used
the wrong sign for VMEC toroidal flux in the Boozer guiding-centre equations.
Its ESSOS loss counts, speed ratios, JSON record and plots are withdrawn.
Current SIMSOPT upstream also used the opposite sign in its VMEC-to-Boozer
field; [SIMSOPT PR #664](https://github.com/hiddenSymmetries/simsopt/pull/664)
corrects it. Its unpatched field matched only 44 of 64 corrected ESSOS labels;
the patched field gave the same 64 terminal labels (55 losses), but six
SIMSOPT paths approached its Boozer-axis singularity. With an inner-flux stop
at `s=0.001`, it resolves 58 paths (50 lost) and matches ESSOS on all 58;
the six axis stops are unresolved. SIMPLE with direct Boozer births matches
all 64 ESSOS labels. [`benchmarks/trace_cross_code.py`](../../benchmarks/trace_cross_code.py)
repeats the guarded check and fails on bad exits, field mismatch or excessive
full-path energy drift in resolved trajectories.

The current check uses a nonoptimized NFP=2 vacuum VMEX seed at reactor scale,
`ns=31`, `mpol=5`, `ntor=5`. The 1,024 births are the same in each tracer:
64 Boozer births from RNG seed 42 and 960 from seed 43, all at
`s=0.283333`, with the same pitch and 3.52 MeV energy. They are traced for
2 ms. On the 64-birth subset, ESSOS, FIRM3D CPU, CATAPULT GPU and DESC
classify the same 55 particles as lost. Boozer-to-VMEC birth mapping agrees
within 0.5 mm in `R` and `Z`; the DESC equilibrium fit differs by up to
0.4% in `|B|` at those births. DESC (vacuum guiding centre, adaptive, `1e-6`
tolerance) takes 32.46 s warmed for those 64 births on an Apple M2 with
endpoint output. ESSOS takes about 1.55 s on that M2 with 101 saved states;
these output policies differ. DESC's maximum survivor endpoint energy error
in this run is 2.59e-4, while ESSOS checks its maximum error at every step.

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

The [cross-code validation record](../explanation/validation.md#cross-code-alpha-trace-comparisons) reports the same-host CPU and GPU timings, full-path energy checks, and near-axis orbit discrepancies. On 1,024 common births, axis-regularized CATAPULT and ESSOS agree on all loss labels (795 lost); released CATAPULT adds seven near-axis losses. The axis correction is proposed in [FIRM3D PR #90](https://github.com/ColumbiaStellaratorTheory/firm3d/pull/90).

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
piecewise constant in the boundary. Derivative-free optimization can minimize
it using a fixed particle ensemble; a new ensemble checks the result.
