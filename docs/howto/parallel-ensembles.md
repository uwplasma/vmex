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

## Threads and devices on an allocation

VMEX sizes itself to the allocation the first time {mod}`vmex.core` is used.
No setting is needed for a CPU or GPU job. The CPU count is the smallest of the
affinity mask, the cgroup quota, the Slurm, PBS, SGE or LSF allocation, and the
host size.

| Resource | Default | Set your own with |
| --- | --- | --- |
| XLA threads in one solve | CPUs, at most 16 | `NPROC` |
| Probe threads in one optimization factorization | CPUs, at most 8 | `probe_workers=` of `VmecProblem.from_tuples` and `least_squares` |
| BLAS threads in one solve | CPUs, at most 16 | `OPENBLAS_NUM_THREADS` |
| ESSOS CPU devices | CPUs | `ESSOS_CPU_DEVICES` |
| Ensemble workers | CPUs, at most one per item | `workers=` |
| GPU | every GPU JAX sees, placed by `device="auto"` | `CUDA_VISIBLE_DEVICES`, `device=` |

A value you set always wins. Set environment variables before starting Python;
they have no effect once the JAX backend or SciPy's BLAS has loaded.

Advanced settings, for when the default is not the best choice:

- Many equilibria on one node: use one process or worker per few cores and
  divide the threads, for example `NPROC=4 OPENBLAS_NUM_THREADS=4`. Independent
  solves scale much better than threads inside one solve.
- Well above 8 poloidal and toroidal modes on a large exclusive node: compare
  `NPROC=36` against the default on your case. It can gain a little wall time
  for a large increase in CPU time.
- Small cases: `NPROC=8 OPENBLAS_NUM_THREADS=8` uses less CPU for about the same
  wall time.
