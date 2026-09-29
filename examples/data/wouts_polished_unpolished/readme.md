# Shaped tokamak: before and after force-balance polishing

## Draft email

Subject: Matched-resolution VMEX tokamak WOUTs for force-balance comparison

Hi,

Attached are unpolished and polished WOUTs from **one** shaped, finite-pressure, fixed-boundary tokamak solve. Both have `ns=370`, `mpol=20`, `ntor=0`, and identical Fourier mode tables, pressure, iota, and boundary geometry. The only change is the interior force-balance polish; the unpolished state was sampled on the polished grid with new Fourier coefficients set to zero. The independently evaluated whole-volume RMS `|J × B − ∇p|` is `1.975e5` versus `4.180e1 N/m³` (near axis: `9.799e5` versus `1.859e2 N/m³`). Please compare the force balance from the two WOUTs with your method. The included `input.*` files reconstruct their profile and boundary data, and `input.shaped_tokamak_pressure_source` is the exact solve deck.

To reproduce, run `vmex input.shaped_tokamak_pressure_source --polish` for the polished equilibrium. `python examples/force_balance_polishing.py` writes both states on the same resolution and prints the independent force certificate; the commands below make it use this exact deck. I would be interested in your residuals, especially near the axis.

Best,
Rogerio

## Creation and reproduction notes

The source deck is the single-stage `ns=51`, `mpol=12`, `ntor=0` input in this folder (`AM=1,-1`, `AI=1.05,-0.35`, `AC=0`, `PRES_SCALE=10000`, `NITER_ARRAY=2000`). It converged in 471 iterations. `--polish` enriched the native representation; VMEX exported its state at `ns=370`, `mpol=20`, `ntor=0`. The example script sampled the **same original solve** on that grid, padding its eight new angular modes with zero coefficients. The interior `R` and `Z` coefficients differ between files while the boundary coefficients agree exactly. Thus the residual decrease is not an effect of comparing different resolutions, and the polished WOUT is not the unpolished state.

From the VMEX repository root, with this folder at `examples/data/wouts_polished_unpolished`:

```sh
bundle=examples/data/wouts_polished_unpolished
vmex "$bundle/input.shaped_tokamak_pressure_source" --no-polish --outdir /tmp/vmex-tokamak-plain
vmex "$bundle/input.shaped_tokamak_pressure_source" --polish --outdir /tmp/vmex-tokamak-polished
VMEX_POLISH_INPUT="$bundle/input.shaped_tokamak_pressure_source" \
VMEX_POLISH_OUTPUT_DIR=/tmp/vmex-tokamak-matched VMEX_EXAMPLES_CI=1 \
  python examples/force_balance_polishing.py
vmex "$bundle/wout_shaped_tokamak_pressure_unpolished.nc" --to-input --outdir /tmp/vmex-tokamak-roundtrip
vmex "$bundle/wout_shaped_tokamak_pressure_polished.nc" --to-input --outdir /tmp/vmex-tokamak-roundtrip
```

The first two commands show the standard CLI workflows. The example script creates the directly comparable pair from one solve; its `wout_shaped_tokamak_before_polish.nc` is the unpolished WOUT here and its polished WOUT is named after the source input. The last two commands recreate the two included reconstructed inputs.

`--to-input` recovers the physical profiles, `PRES_SCALE=10000`, flux, and boundary from both WOUTs. A WOUT does not retain the original multigrid controls, `NITER_ARRAY`, or the pre-export `ns/mpol`: the reconstructed inputs therefore use the stored `370/20/0` resolution and an inferred `NITER_ARRAY=1000`. Use the included source deck for an exact repeat of the original 51/12 solve. Also, WOUT `FSQR/FSQZ/FSQL` report the original discrete solve residual, so use an independent continuum force evaluation for this comparison.
