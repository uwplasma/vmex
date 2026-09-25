# Restart from a previous run and scan parameters

Seed a solve from an earlier result instead of starting cold: from an
in-memory state (`initial_state=`), or from any `wout_*.nc` file — VMEX-,
VMEC2000-, or PARVMEC-written — via `restart_from=` / `vmex --restart`.
A converged same-deck restart re-converges in 1 iteration instead of 435
(cth `ns=15` at `FTOL 1e-14`, including from the VMEC2000-written golden
wout).

## Restart from a wout file (CLI)

```console
vmex input.x --restart wout_y.nc
```

or put it in the deck with the VMEX extension key (resolved relative to the
input file; the CLI flag wins):

```text
&INDATA
  RESTART_WOUT = 'wout_y.nc'
  ...
```

Both work for fixed- and free-boundary decks. Multigrid rungs whose
resolution the seed already meets or exceeds are skipped: the free-boundary
cth ladder restart takes 99 final-stage iterations instead of 340 and skips
its coarse rung.

## Restart from a wout file (Python)

Every solve entry point accepts `restart_from` — a `wout_*.nc` path, a
{class}`~vmex.core.wout.WoutData`, a previous
{class}`~vmex.core.solver.SolveResult`, or a bare
{class}`~vmex.core.solver.SpectralState`:

```python
import vmex as vj

inp = vj.VmecInput.from_file("input.x")
result = vj.solve_multigrid(inp, restart_from="wout_y.nc")
```

The reconstruction ({mod}`vmex.core.restart`) inverts the `wrout.f` output
maps exactly — `rmnc/zmns` (and LASYM partners) remapped by `(m, n)` onto
the target `MPOL/NTOR` table, rescaled to internal `mscale*nscale`
normalization, rotated into the evolved m=1-constrained basis, and the
half-mesh `lmns` inverted surface-by-surface to the full-mesh internal
lambda. R/Z are exact at machine precision and lambda on every interior
surface, so a converged wout restarts at its converged residual. Radial differences pass through the same
`interp.f` transfer as the multigrid ladder (up- and down-sampling), and a
seed finer than the whole ladder runs only the final rung.

Fixed-boundary restarts adapt the seed to the deck's (possibly perturbed)
boundary through {func}`~vmex.core.solver.hot_restart_state` and rebind the
`rcon0/zcon0` baselines; free-boundary restarts keep the wout's evolved free
edge and repeat vacuum activation (reset-file semantics).

## Restart from an in-memory state

Inside one process, skip the file entirely:

```python
result0 = vj.solve_multigrid(inp)
inp2 = ...                                   # perturbed deck
result1 = vj.solve_multigrid(inp2, initial_state=result0.state)
```

{func}`~vmex.core.multigrid.interpolate_state` moves a state between radial
resolutions explicitly when you need control over the transfer.

## Scan a parameter with hot restarts

Solve the first point cold, then seed every successive point from the
previous converged state. A warm restart takes fewer iterations than a cold
solve of the same point, by a margin that shrinks as the step between points
grows. Boundary moves of 1e-4 to 1e-2 on the low-resolution QA deck took
212–391 warm iterations against 806 cold. The example below needs only one
per point, because at zero pressure with a prescribed transform a PHIEDGE
scan does not move the geometry. Because VMEX caches one
compiled executable per solver structure, points after the first compile
nothing at fixed resolution: the new-parameter warm run in
`benchmarks/baselines/m4/F1_warm_newparams.json` records zero compiles.

### The hot-restart scan pattern

```python
import dataclasses
import vmex as vj

base = vj.VmecInput.from_file("input.case")
result = None
for phiedge in phiedge_values:
    inp = dataclasses.replace(base, phiedge=phiedge)
    result = vj.solve_multigrid(inp, restart_from=result)   # seed from the previous point
```

`examples/hot_restart_scan.py` is the complete measured version — it prints
per-point iteration counts so the cold-vs-warm difference is visible:

```{literalinclude} ../../examples/hot_restart_scan.py
:language: python
```

### Rules that keep a scan fast

- **One executable per structure.** A compiled executable is keyed by the
  solver structure (`ns`, mode tables, angular grid, lane). Keep the
  resolution fixed across the scan and only point 1 compiles; change `ns`
  or `mpol/ntor` mid-scan and you pay a fresh compile.
- **Warm-start from the neighbor.** `restart_from=` accepts the previous
  `SolveResult`, a {class}`~vmex.core.solver.SpectralState`, or, across
  processes, the previous point's wout file (above).
- **Skip the ladder on warm points.** A converged neighbor already has the
  final resolution, so `restart_from=` drops every `NS_ARRAY` rung the seed
  covers. `initial_state=` only seeds the first rung and skips none; the
  bundled example uses it on a single-grid deck, where the two coincide.

### Finite-beta scan example

`examples/finite_beta_scan.py` ramps the pressure and reads three
diagnostics straight from each wout — volume-averaged beta, the Shafranov
shift, and the Mercier `DMerc` profile — hot-restarting each step:

```{literalinclude} ../../examples/finite_beta_scan.py
:language: python
```

### Independent points: thread them

When scan points do not build on each other, solve them concurrently
instead: {doc}`parallel-ensembles`.

## When each pays off

- **`initial_state=` or `restart_from=<SolveResult>`** — parameter scans in
  one process: no I/O, and successive points take fewer iterations than a
  cold solve. How many depends on how far the equilibrium moves: 212–391
  against 806 cold for boundary moves of 1e-4 to 1e-2 on the low-resolution
  QA deck (above).
- **`restart_from=` / `--restart`** — resume across processes or machines,
  refine a converged run at higher resolution, or seed from a VMEC2000
  archive: anything that starts from a file.
- **Cold start** — a structurally different problem (changed `NFP`,
  topology, or a far-away boundary); the multigrid ladder is the designed
  cold-start path.

The mechanism (exact output-map inversion, rung skipping, carried module
state) is {doc}`/explanation/iteration`.
