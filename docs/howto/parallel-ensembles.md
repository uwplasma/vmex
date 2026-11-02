# Solve many equilibria at once

`vj.parallel.solve_ensemble` threads independent solves over a
`ThreadPoolExecutor`; each host solve releases the GIL while XLA executes,
so the solves overlap for real wall-clock speedup.

## Run an ensemble

```python
import vmex as vj

inputs = [vj.VmecInput.from_file(f) for f in deck_files]   # N independent decks
results = vj.parallel.solve_ensemble(inputs, workers=4)    # list[SolveResult]
```

`solve_ensemble` threads {func}`vmex.core.multigrid.solve_multigrid`
(default) or {func}`vmex.core.solver.solve` (`multigrid=False`) over the
ensemble and returns the results in input order. `workers=None` uses
{func}`vmex.core.parallel.available_cpus`, clamped to the ensemble size.

For anything that is not a plain solve, use the general primitive
{func}`vmex.core.parallel.map_ensemble`, which threads any independent
per-item function — e.g. a `jax.value_and_grad` of
{func}`vmex.core.implicit.run` for an ensemble of differentiable objectives.

The full pattern (a `phiedge` scan solved with 1, 2 and 4 workers, with
timing output) is `examples/parallel_ensemble_scan.py`.

## What to expect

Every ensemble result is *byte-identical* to solving that input alone: the
solves share no mutable state, and the concurrency only overlaps their
GIL-releasing XLA windows. `tests/test_parallel.py` asserts exactly zero
state difference (and identical iteration counts) against the serial solve
on a solovev / circular-tokamak / li383 ensemble and on a `phiedge` scan.

How much wall-clock you get back depends on your host — core count, how
many of them XLA is already using inside a single solve, and what else is
running — so this page quotes no speedup table. Measure your own with
`examples/parallel_ensemble_scan.py`, which runs a balanced `nfp2_QA`
`phiedge` scan at 1, 2 and 4 workers and prints the wall times. Expect
efficiency to fall as workers rise: a single solve already uses XLA's
internal threading, so workers and intra-solve threads compete for the same
cores.

## When it does not help

- **Unbalanced ensembles.** The ensemble finishes no sooner than its slowest
  member, so a balanced scan at fixed resolution (one shared executable,
  similar iteration counts) gains most.
- **Gradient ensembles.** The implicit adjoint's backward pass holds the GIL
  while it dispatches, so a threaded `value_and_grad` ensemble overlaps its
  forward solves much better than its reverse passes. Values and gradients
  stay bit-identical.
- **GPUs.** The ensemble helper is a CPU thread pool; place members on
  explicit devices yourself.

The mechanism, and why `pmap` and `vmap` were rejected, is in
{doc}`/explanation/architecture`.

## Optimizations on a full node

With no settings, `derivative_method="finite_difference"` evaluates its probes
on threads over every available CPU, and an exact-Jacobian optimization runs as
one process. No more is switched on automatically, because measurement does not
justify it. On `QA_optimization.py` (24 degrees of freedom, a shared 36-core
host), the wall time stays flat from 1 to 36 cores, with the final cost equal to
about 1e-11, and most of the run is the forward solves:

| Cores for one run | 1 | 8 | 18 | 36 |
|---|---|---|---|---|
| Wall, two repeats (s) | 621, 369 | 409, 457 | 412, 446 | 462, 502 |

Finite differences do not pay here either. One central-difference Jacobian
takes 32 s on 36 processes and 73 s on 36 threads, against 3-10 s for the exact
one, and it differs from the exact Jacobian by 0.44%.

### Advanced: independent starts

The cores do pay for independent runs. Each one below had a single core, and the
member with the shipped seed reproduced the single-core run's cost to 1e-12:

| Processes | 1 | 8 | 18 |
|---|---|---|---|
| Wall for all members (s) | 369-621 | 390 | 438 |
| Lowest cost of all members | 7.63e-4 | 5.93e-4 | 5.43e-4 |

This returns a different, better optimum rather than the same result faster,
and it needs a picklable builder, so you ask for it with
{class}`vmex.core.parallel.ProcessWorkers`:

```python
def build():          # module level; runs once in each worker
    return run_from_seed

if __name__ == "__main__":
    with vj.parallel.ProcessWorkers(build) as pool:   # one worker per core
        results = pool.map(seeds)
```

Change `threads` only if one run shows a gain from more cores, and lower
`workers` when memory is short: a worker needs up to about 3 GB at this
resolution. `cache_dir` shares compiled programs, but startup stays at 40-110 s
per worker, since in one split only 22 s of 123 s was loading programs and 6 s
importing; most of the rest is tracing. `gpus=[0, 1]` gives each worker its own
device. Two workers on two GPUs took 322 s for that Jacobian against 120 s on 8
CPU processes, so GPUs are not chosen automatically.
