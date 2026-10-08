# Run neoclassical transport

`vmex --neoclassical` runs [DKX](https://github.com/uwplasma/DKX), the JAX
drift-kinetic solver, on a solved equilibrium. It needs the `neoclassical`
extra (`pip install "vmex[neoclassical]"`, DKX 2.8.0 or later; an older DKX runs
without `--nc-profiles` and `--nc-er` and prints the upgrade command).

## Run it

```console
vmex wout_case.nc --neoclassical                    # default preset
vmex input.case --neoclassical --nc-preset quick    # solve first, smoke run
vmex wout_case.nc --neoclassical --nc-profiles profiles.json --nc-er -10
```

It writes `case_neoclassical.png` and `case_neoclassical.h5` beside the input,
or in `--outdir`, and prints each stage with its wall time.

## What the figure shows

| panel | quantity | reference |
|---|---|---|
| top row | monoenergetic `D11`, `D31`, `D33` against `ν'` at three `E*` | Beidler et al., Nucl. Fusion 51, 076001 (2011) |
| middle left | `|B|` on the `r/a = 0.5` surface | |
| middle centre | ambipolar `E_r` per surface; bootstrap `<j·B>` at that root against the equilibrium's and, with `--nc-profiles`, VMEX's Redl formula on the same profiles | Redl et al., Phys. Plasmas 28, 022502 (2021); Landreman, Buller and Drevlak, Phys. Plasmas 29, 082501 (2022) |
| middle right | particle and heat fluxes per species at the root | Beidler et al., Nature 596, 221 (2021) |
| bottom | radial current `J_r(E_r)` on each surface, with each root labelled ion, unstable or electron | Hastings et al., Nucl. Fusion 25, 445 (1985) |

The HDF5 file holds the numbers behind the figure: the monoenergetic scan, `|B|`,
and per surface `E_r`, the three `<j·B>` and the species fluxes. The README
figure is redrawn from it by
`docs/_static/figures/sources/make_neoclassical_figure.py`.

## Presets and cost

| `--nc-preset` | surfaces | use | wall time |
|---|---|---|---|
| `quick` | 2 | smoke run; the numbers are not reportable | 5 min |
| `default` | 5 | the README figure | 6.5 min |
| `full` | 5 | wider monoenergetic grid, finer profile solves | longer |

Times are for the bundled `input.LandremanPaul2021_QA_beta2p5_bootstrap` WOUT
with `--nc-profiles` on 4 CPU cores with one BLAS thread, compilation included.

## Profiles

VMEC input decks carry pressure, not density and temperature. Pass them with
`--nc-profiles`, a JSON file in the {class}`~vmex.core.bootstrap.KineticProfiles`
convention (polynomials in `s`, lowest order first):

```json
{"ne_coeffs": [2.38e20, 0, 0, 0, 0, -2.38e20],
 "Te_coeffs": [9450.0, -9450.0], "Ti_coeffs": [9450.0, -9450.0],
 "helicity_n": 0}
```

This is `examples/data/kinetic_profiles.LandremanPaul2021_QA_beta2p5_bootstrap.json`,
the published profiles of that deck. `helicity_n` (0 for QA, +1 or -1 for QH)
is used only by the Redl formula. `T_i` may differ from `T_e`.

Without `--nc-profiles`, DKX splits the WOUT pressure into density and
temperature with `T_i = T_e`, scaling the on-axis density from the reactor
profiles of Landreman, Buller and Drevlak (2022). That split is an
assumption, stated in the figure caption, and the kinetic current is only as
good as it is. A vacuum WOUT uses DKX's generic reference plasma.
