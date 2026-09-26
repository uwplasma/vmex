# Coil-constrained single-stage benchmarks

Edit shared physics and optimization settings in `parameters.py`, then run one
of the two scalar scripts. Both use `build_problem` → `run_optimizer` →
`verify_endpoint` and the public VMEX optimizer API. Their implementation is
shared in `../single_stage_support/`; no qualification runs before production.

| File | Purpose |
|---|---|
| `parameters.py` | Common resolution, optimizer settings and physical coil limits |
| `single_stage_optimization_scalar.py` | Fixed-boundary and coil optimization |
| `free_boundary_single_stage_optimization_scalar.py` | Free-boundary coil optimization |
| `_coil_constraints.py` | Geometry inequalities and independent endpoint checks |
| `verify_single_stage.py` | Explicit fixed-boundary derivative qualification |
| `verify_free_boundary_single_stage.py` | Explicit free-boundary derivative qualification |

The scripts retain their distinct physics: fixed boundary varies the boundary
and coils with a normal-field penalty; free boundary varies coils and solves
for their supported equilibrium. Both use SLSQP and explicit coil inequalities.
The separate QA-only experiment is not the default in this example.

## Run

With this checkout and `vmex[coils]` dependencies, run from this folder:

```sh
python single_stage_optimization_scalar.py --device gpu --accepted-steps 100 --output runs/fixed
python free_boundary_single_stage_optimization_scalar.py --device gpu --accepted-steps 100 --output runs/free
```

The defaults share the rotating-ellipse input and saved fitted coils in
`../single_stage_support/data/`. The initial/fitted coil bytes and currents are
unchanged; both folders now use the same files. `coils.fitted.json` replaces the
previous timestamp-named coil file. High Fourier modes are zero-padded when
increasing coil order; lowering order is rejected.

Use `--coils` to reuse a different fit, `--initial-coils` to fit a different
seed, or `--input` and optionally `--wout` to change the plasma input. Both
support `--resolution`, `--grid`, `--ftol`, `--accepted-steps`, `--dry-run`,
`--no-plots` and `--no-movie`. Only vacuum, stellarator-symmetric cases are
supported. See [the shared interface](../single-stage-benchmarks/README.md).

## Geometry and accuracy

Defaults: MPOL=8, NTOR=8, NS=51, equilibrium grid 64×64, FTOL=1e-15;
three order-16 coils and 256 Biot–Savart quadrature points. Field quadrature,
coil geometry sampling and the equilibrium grid serve different purposes.

Physical limits are length ≤5 m, peak curvature ≤5 m⁻¹, arc-length-weighted
mean-squared curvature ≤5 m⁻², intercoil distance ≥0.15 m and coil–plasma
clearance ≥0.20 m. `parameters.py` also defines the optimizer's interior
margins. The starting fit may be infeasible; a lower QA alone is not success.

Geometry constraints use 1024 curvature samples and 256 polygon segments.
Final checks refine curvature peaks and mean-squared curvature on 4096/8192
points, and refine plasma clearance beyond the production grid. A conservative
intercoil lower bound below the limit means clearance is **not certified**;
it does not by itself prove the actual spacing violates the limit. These are
filament checks, not finite-winding-pack or globally certified surface-clearance
checks. Free-boundary endpoints are independently solved at NS=201.

## Separate checks

```sh
python verify_single_stage.py --device gpu --output runs/fixed-qualification
python verify_free_boundary_single_stage.py --device gpu --output runs/free-qualification
# From the repository root:
python -m pytest tests/test_single_stage_coil_constraints.py
```

These verifiers call the production builders. Production may use a matching
free-boundary `--qualification` report or an authenticated `--seed`; neither
repeats finite differences. Shared implementation changes invalidate old
qualification fingerprints. Existing running jobs retain their frozen sources.

The geometry tests live in `tests/`, not in this production folder. Historical
source hashes, seed measurements and validation records are preserved unchanged
under `benchmarks/single_stage_provenance/`; they describe the original code
and do not certify this refactor. A budget stop, numerical verification and
physical feasibility are separate results in the output summary.
