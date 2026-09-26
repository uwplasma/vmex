# Single-stage benchmarks

The fixed- and free-boundary scalar entries follow the same public workflow:
`build_problem` → `run_optimizer` → `verify_endpoint`. Edit the physics and
optimizer settings at the beginning of each script. File handling, fitting,
checkpoint checks and output are shared in `../single_stage_support/`.

| Entry | Purpose |
|---|---|
| `single_stage_optimization_scalar.py` | Fixed boundary and coils, SLSQP |
| `free_boundary_single_stage_optimization_scalar.py` | Coil-supported free boundary, SLSQP |
| `free_boundary_single_stage_optimization.py` | Free-boundary L-BFGS-B variant |
| `qa_optimization.py` | Boundary-only QA optimization |
| `solve_free_boundary_initial_coils.py` | Free equilibrium from the saved initial coils |
| `compare_scalar_steps.py` | Compare saved optimization histories |

The formulations differ physically. Fixed boundary varies boundary coefficients
and coils and penalizes the normal field on the boundary. Free boundary varies
coils, with the equilibrium determining the boundary. These baseline entries use
coil penalties; explicit coil inequalities are in `../coil-constraints-benchmarks/`.
Matching the API alone does not make the objectives or feasible sets identical.

## Production

Use this checkout with `vmex[coils]` dependencies. From this folder:

```sh
python single_stage_optimization_scalar.py --device gpu --accepted-steps 100 --output runs/fixed
python free_boundary_single_stage_optimization_scalar.py --device gpu --accepted-steps 100 --output runs/free
```

Both scalar defaults are M8/N8/NS51, a 64×64 equilibrium grid, FTOL=1e-15,
and final NS=201 verification. The baseline fixed boundary has `MAX_MODE=3`
optimization modes, independently of equilibrium resolution. The free solver
retains coupled-root polishing, checked matrix-free adjoints, adaptive LU
refresh and bounded dense recovery through public VMEX APIs.

Both accept:

- `--input input.case`; explicit decks retain their resolution unless overridden.
- `--wout wout.nc --input input.case`; remap a saved boundary/state with its profiles.
- `--initial-coils initial.json`; fit these coils on the frozen starting boundary.
- `--coils fitted.json`; reuse fitted coils and their currents without fitting.
- `--resolution MPOL NTOR NS`, `--grid NTHETA NZETA`, `--ftol`, `--accepted-steps`
  (`--maxiter` alias), `--coil-fit-maxiter`, `--device`, and `--output`.
- `--dry-run`, `--no-plots`, and `--no-movie`.

Without a coil argument these baseline examples generate circles and fit them.
For a controlled comparison pass the same saved coil file to both, for example
`--coils ../single_stage_support/data/coils.fitted.json`.

The shared data folder contains the unchanged rotating-ellipse input, initial
coils and their metadata, and the fitted coils. `coils.fitted.json` replaces the
previous timestamp-named file; the duplicate copies in both benchmark folders
have been consolidated. The input deck itself is M3/N3/NS31 with a 48×40 grid;
script defaults and explicit-deck behavior are distinct. The workflows require
stellarator symmetry, zero pressure and zero plasma current. Finite beta needs
plasma-aware fitting and interface diagnostics beyond this example.

## Qualification and verification

Production does **not** run finite-difference qualification. Run it explicitly:

```sh
python verify_single_stage.py --device gpu --output runs/fixed-qualification
python verify_free_boundary_single_stage.py --device gpu --output runs/free-qualification
python verify_single_stage_constraints.py --help
```

Free production can reuse `--qualification runs/free-qualification/qualification.json`
or an authenticated `--seed seed.json`. A seed is not a derivative qualification.
The input, numerical settings, shared implementation and artifact hashes must
match. This refactor changes those fingerprints; old reports belong to the old
source snapshot and cannot silently certify the reorganized code.

Each production run independently verifies its endpoint. Inspect the saved
summary, accepted history, root/force diagnostics and constraint checks. A step
budget, lower QA, or a numerical verification pass does not establish optimizer
convergence or complete physical feasibility.
