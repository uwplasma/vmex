# HINT build and finite-beta QA validation helpers

These helpers accompany the HINT–VMEX comparison handoff. Paths below are
relative examples; adapt them to your checkout layout. No remote host or
machine-specific installation is assumed. Run one numerical job at a time
unless resource ownership has been coordinated.

Use the [complete pinned and patched workflow](../README.md) as the entry point; commands here illustrate individual helper interfaces.

## Build maintained HINT

Requirements: GNU Fortran, an MPI Fortran wrapper using that compiler, HDF5
with its Fortran libraries, NetCDF C and Fortran with `nf-config`, Make,
Git and Python 3. These are requirements, not a tested compatibility matrix.
Use mutually compatible compiler/MPI/library installations.

```sh
git clone --branch current --single-branch https://github.com/yasuhiro-suzuki/HINT3D.git HINT3D
export HDF5_PREFIX="<your HDF5 Fortran installation prefix>"
python build_suite.py --hint-root HINT3D --mode Debug
python build_suite.py --hint-root HINT3D --mode Release
```

`--hint-root` defaults to the current directory. `FC` defaults to `mpif90`,
`NF_CONFIG` to `nf-config`; both may be overridden in the environment.
`--hdf5-prefix` can replace `HDF5_PREFIX` and is required if that variable is
unset. The helper expects HDF5 headers under `include` and libraries under
`lib` at that prefix. A distribution using another layout needs its matching
installation prefix or an explicit build adjustment; no layout autodetection
is claimed. GNU-specific debug flags enable bounds and floating-point checks.

The helper retains upstream Makefile object ordering, builds serially in
`build-debug` or `build-release`, and writes per-program logs plus
`build_config.json`. `--only HINT` restricts the build. The maintained `current`
branch is intentional; a reproducible study should also record/pin its exact
commit. These portability changes were syntax/CLI checked, not rebuilt on a
new platform.

## QA root and derivative diagnostics

Install VMEX and SOLVAX dependencies using their project instructions. The recorded VMEX revision for this bundle is
`14360d179af4534aa9bb638b172c8724ee289d9b`.
Use the provided finite-beta 0.5% QA input and reference WOUT; the input must
retain ns31, MPOL7, NTOR6, NFP2, ncurr=1 and its original pressure/current/flux.
The scripts are deliberately narrow validation examples, not generic input
converters. SOLVAX is imported from the explicitly selected source tree.

```sh
python qualify_finite_beta_qa_root.py \
  --vmex-root vmex --solvax-root solvax \
  --input data/input.qa_beta0p5 --reference-wout data/wout.qa_beta0p5.nc \
  --cold-runs 2 --per-run-timeout 240 --output results/qa-root
python qualify_finite_beta_qa_derivatives.py \
  --vmex-root vmex --solvax-root solvax \
  --input data/input.qa_beta0p5 --per-run-timeout 240 \
  --output results/qa-derivatives
```

Source archives without `.git` require `--source-commit` for root/derivative
VMEX provenance. Full source hashes are recorded as well. Output directories
must not already exist. GPU runs use the same scripts under a deliberately
selected device environment; pure callbacks require the CPU backend alongside
CUDA. Do not claim individual kernel placement from a device-environment
label alone.

The derivative script uses the live parameterized equilibrium custom VJP,
one boundary direction, certified matrix-free tangent/adjoint duality and
four halving central/Taylor checks. Every perturbed root requires a matching
actual-state primal certificate. It does not differentiate cached snapshot
fields. CPU remainder orders were approximately 1.98, 1.96 and 1.84;
small-step central-difference errors increased. Completion does not imply
all-direction, exterior-field, free-boundary or optimization qualification.
Read `report.json`, not just the process exit code.

## Synthetic calibration (incomplete measured attempt)

```sh
python calibrate_finite_beta_qa_energy.py \
  --vmex-root vmex --solvax-root solvax \
  --input data/input.qa_beta0p5 --target results/qa-derivatives \
  --output results/qa-calibration
```

The example targets energy at the known interior amplitude q=5e-5. Host
safeguarded scalar Newton holds pressure/current/flux fixed, limits q to
+/-1e-4 and uses ordinary VMEX derivatives. SOLVAX supplies VMEX's refinement
and adjoint; no direct nonlinear JVP wrapper is claimed. The script reserves
a separate fresh-process final certification within its total 240-second
budget. A completed run must include a passing `cold-final.json`.

The measured attempt reached its 180-second optimizer-worker limit after one
accepted trial: energy residual 1.21e-4 decreased to 6.31e-8, but the 1e-9
criterion was not met and the independent final check did not run. The next
change should separate primal/gradient timing and evaluate the already tested
full-parameter gradient contracted with the direction, then avoid a new
gradient when a certified candidate value already meets tolerance. Those
algorithm changes are **not** applied to this copied, measured script.

## Exact tested helper bytes

Keep the three top-level QA scripts together: they import provenance utilities
from their sibling `qualify_finite_beta_qa_root.py`. The latest top-level root
helper skips physical diagnostics on a failed callback's possible finite
placeholder. That guard was added after the successful CPU/GPU root and CPU
derivative observations; their recorded hashes therefore identify the earlier
helper. `tested-cpu-14360d17/` preserves the exact root/derivative pair matching
the CPU derivative manifest. The derivative script itself is unchanged.
`script-sha256.json` hashes every bundled script. No numerical run was performed
while preparing this portable copy.
