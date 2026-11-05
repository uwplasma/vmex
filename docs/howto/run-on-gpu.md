# Run on GPU

Pass `--device gpu` (CLI) or `device="gpu"` (Python) to place a solve on an
accelerator. The default `auto` uses a workload heuristic from an earlier
benchmark campaign; it does not measure your hardware or guarantee a speedup.
Compare CPU and GPU on your workload before choosing a device.

## Running on CPU, GPU and whole nodes

The defaults should cover most runs. They do the following:

- **Precision and cache.** float64, and a compilation cache under `~/.cache/vmex/jax_cache`, split per machine.
- **CPU threads.** One solve uses XLA's thread pool over the CPUs the process may use.
- **Device.** `auto` chooses per stage from problem size (below). Mirror solves and the dense vacuum
  factor stay on CPU.
- **GPU memory.** Allocated on demand, so several processes can share a GPU.
- **Several GPUs.** A solve uses the first GPU unless you pass a `jax.Device`. VMEX does not shard one
  solve across GPUs, and no multi-GPU speedup of one solve has been measured.
- **Ensembles and finite differences.** `workers=None` runs up to one thread per available CPU,
  honouring affinity and the scheduler's allocation.
- **Jacobian chunks.** `jac_chunk_size="auto"` bounds memory and avoids compiling a second graph.

| Setting | Default | Change it when |
| --- | --- | --- |
| `device=` / `--device` | `auto` | Your timing shows the other device wins: the crossover varies by machine. `auto` runs eager free-boundary gradients on the GPU when the problem is large, and falls back to the CPU if GPU memory runs out |
| `device=jax.devices("gpu")[i]` | first GPU | You assign independent cases to separate GPUs |
| `workers=` | available CPUs, at most one per case | Sharing a node, or the case count exceeds what the cores can run well: each worker also uses XLA threads |
| `adjoint_solver=` (free boundary) | `boundary_schur` | Use `coupled_gcrot` only to cross-check a gradient, or when the banded factor does not fit host memory |
| `jac_chunk_size=` | `auto` | Memory runs out (smaller), or memory is ample and Jacobians are slow (larger) |
| `NPROC`, `OPENBLAS_NUM_THREADS` | not set | Several processes share a node: give each its share before starting Python |
| `VMEX_COMPILATION_CACHE_DIR` | under `~/.cache` | The home filesystem is slow or small, or a cluster needs a shared path; `disabled` turns the cache off |
| `XLA_PYTHON_CLIENT_PREALLOCATE=true` | `false` | A dedicated GPU, one process, and you want JAX's preallocation |
| `VMEX_FAST_COMPILE=1` | off | Short CPU runs dominated by compile time; warm kernels get slightly slower |

Measured on one Xeon W-2295 (36 logical CPUs, JAX 0.12.0.1), warm value plus gradient of a
free-boundary problem at `mpol = ntor = 5`, `ns = 16`, `boundary_schur`, other jobs on the node:

- Threads for one process, one run each at load 5-7: 1, 2, 4, 8 and 18 threads took 10.8, 6.4, 5.7,
  5.0 and 6.3 s, using 10.8, 11.6, 16.4, 22.1 and 52.9 CPU-seconds. More cores soon stop helping.
- `coupled_gcrot` took 301 s against 5-8 s for `boundary_schur`, with gradient norms agreeing to about
  9 digits (one run, load about 30).
- On 8 pinned cores at load 64-76, a 4-worker thread ensemble gave 0.14-0.17 evaluations/s against
  0.07-0.08 for serial evaluations (two rounds), and the gradients matched.

These numbers are from one busy node; time your own deck before changing a default.

## Select the device

```console
vmex input.case --device gpu     # explicit: always wins
vmex input.case --device cpu
vmex input.case --device auto    # default: workload heuristic
vmex input.case --device none    # leave placement to JAX
```

```python
import vmex as vj

result = vj.solve_multigrid(inp, device="gpu")
```

Explicit `device=` always wins. `auto` stands down when you pinned a JAX
default device or platform yourself (`jax.config.update("jax_default_device", ...)`,
`JAX_PLATFORMS`, `jax.default_device(...)`), so VMEX never fights your
placement. Install notes for GPU wheels: {doc}`/installation`.

## When the GPU pays off

The policy in {mod}`vmex.core.device` uses the historical measurements in
`benchmarks/gpu_baseline.json` (regenerate with `tools/run_gpu_matrix.py`;
`tools/device_parity.py` checks CPU/GPU agreement). These motivated the thresholds below; the later
A4000 results show why they are not a portable performance guarantee:

- Per-iteration throughput favors the GPU — up to 3x wall on
  NuhrenbergZille-class decks — but the GPU pays fixed per-solve overheads
  (~0.2-0.4 s dispatch/transfer floor plus compile/cache-load on cold
  processes), so small decks that converge in under a second of CPU work
  finish faster on the CPU.
- The work proxy is `ns * mnmax * nznt` (radial surfaces x spectral modes x
  angular grid — the cost driver of the batched `totzsps/tomnsps` matmuls).
  Measured decks split into two clusters: proxies up to ~24e3 where the CPU
  wins (and misclassification costs < 0.5 s either way), and >= ~490e3 where
  the GPU wins 2-3x. `GPU_MIN_ITERATION_WORK = 100_000` sits between them
  (geometric mean ~109e3). The range between the clusters is not calibrated.
- Mode count is an independent guard: the measured GPU winners have at most
  162 active modes, while a high-resolution HSX deck (`mnmax=858`) ran ~3.4x
  *slower* on the GPU even warm despite a large work proxy.
  `GPU_MAX_SPECTRAL_MODES = 512` sits between the largest measured GPU
  winner (288 modes) and that high-mode CPU winner; the cutoff is not
  claimed as a hardware-independent crossover.

Ask the policy directly:

```python
import vmex as vj
from vmex.core.device import GPU_MIN_ITERATION_WORK, iteration_work, recommended_device
from vmex.core.solver import resolution_from_input

inp = vj.VmecInput.from_file("input.my_case")
# The finest multigrid stage is the one that dominates the run.
resolution = resolution_from_input(inp, ns=int(inp.ns_array[-1]))

print(iteration_work(resolution), GPU_MIN_ITERATION_WORK)
print(recommended_device(resolution))    # "cpu" or "gpu"
```

On the two shipped decks that prints `1632 100000` / `cpu` for
`input.circular_tokamak` and `1536000 100000` / `gpu` for
`input.LandremanPaul2021_QA_lowres`.

```{warning}
**The crossover these thresholds encode does not transfer between machines.**
They come from `benchmarks/gpu_baseline.json`, measured 2026-07-09. The same
sweep re-run on 2026-09-16 on two RTX A4000s (JAX 0.11.1, driver 580.173,
one process per cell, VMEX 0.9.1) found **no cell
where the GPU wins**, warm wall in seconds:

| case | CPU warm | GPU warm | GPU gain |
| --- | --- | --- | --- |
| `solovev` | 0.07 | 0.31 | 0.21x |
| `cth_like_fixed_bdy` | 0.42 | 0.74 | 0.57x |
| `nfp4_QH_warm_start` | 0.32 | 1.45 | 0.22x |
| `LandremanPaul2021_QA_lowres` | 0.29 | 0.43 | 0.67x |
| `NuhrenbergZille_1988_QHS` | 110.91 | 133.64 | 0.83x |
| synthetic nfp4 QH, ns 35 to 151 | 0.12–0.71 | 0.47–1.41 | 0.22–0.51x |

`NuhrenbergZille_1988_QHS` is the largest case in the sweep at 111 s of warm CPU
work, and the synthetic scan walks `ns` and `mnmax` up without crossing over, so
this is not a threshold that is merely set too low here. Cold wall was about 2x
the CPU's throughout, and the iteration counts match exactly between each deck's
CPU and GPU cell (434, 2829, 125, 189), so the two ran the same physics. Peak
device memory across every GPU cell was 4.8 to 113.5 MB.

So treat `recommended_device` as a starting guess and time the deck you actually
run: `python tools/run_gpu_matrix.py --skip-tridiag --out my_gpu_matrix.json`
reproduces the table above on your own hardware (without `--out` it overwrites
the committed `benchmarks/gpu_baseline.json`). The implicit-gradient path is the exception that
does pay off — the single-stage finite-beta value-and-gradient is 1.3–1.6 s warm
on the GPU against 2.15 s on that machine's CPU.
```

## CUDA graphs

A solver iteration is many small kernels, so on a GPU it is launch-bound
(about 2 ms per iteration at `ns = 16` on an RTX A4000, rising only to 8 ms at
`ns = 101`). Unless you set `XLA_FLAGS` yourself, VMEX passes
`--xla_gpu_graph_min_graph_size=1`, so XLA captures every fused run of kernels
into a CUDA graph rather than only runs of five or more: 8-11 % faster per
iteration on an A4000 from `ns = 16` to `101`. Turning command buffers off
(`--xla_gpu_enable_command_buffer=`) makes the same iterations about twice as
slow. Capturing `while` loops crashed and capturing conditionals was slower,
so VMEX leaves both out.

## CPU placement and optimization defaults

- **Ensembles.** The ensemble helpers are CPU thread pools and assign no GPUs
  ({doc}`parallel-ensembles`); give each case its own `device=` to use several.
- **Implicit Jacobians in optimization.** High-level optimization defaults
  its implicit-gradient path to CPU because it is launch-bound on the tested
  GPUs; low-level {func}`vmex.core.implicit.run` follows JAX placement when
  `device` is omitted, and accepts `device="gpu"` plus
  {func}`~vmex.core.device.device_scope` for explicit accelerator gradients.
- **The dense NESTOR factor.** On a GPU free-boundary run the plasma
  iteration stays on the accelerator while the dense vacuum
  assembly/factor/solve is explicitly placed on CPU
  ({doc}`/explanation/nestor-vacuum`).

## Verify what you got

```console
vmex --doctor
```

prints the JAX backend, visible devices, the active default device, and
VMEX's forward/implicit placement policies. It also executes a small float64
JIT calculation on the selected JAX device; on WSL2 it reports the NVIDIA GPU
and Windows driver visible through `nvidia-smi`. See {doc}`/installation` for
the upstream fixes to the warnings seen with JAX 0.9.2 and the required
JAX/jaxlib upgrade. Per-deck CPU-vs-GPU timings and the decision sweep for a
new machine are in {doc}`/reference/performance`.

## Separate cold compile, cache reload, and warm execution

For a new GPU or a slow installation, run the cache-reload audit from a VMEX
source checkout:

```console
python tools/device_cache_reload.py --devices cpu,gpu \
    --output vmex-device-cache-reload.json
```

The default bounded case records, independently for CPU and GPU:

- a fresh process with an empty compilation cache;
- a second fresh process reloading exactly that cache;
- an in-process warm repeat in each process;
- one implicit MHD-energy gradient;
- actual state/value/gradient placement;
- peak host RSS and device memory.

Each device gets a separate temporary cache, which is removed after the
measurements. `--cache-dir PATH` preserves entries for inspection but accepts
only an empty directory; the benchmark never deletes or mixes with a user's
normal VMEX cache. Use `--full` for the larger parity case and `--devices gpu`
for the shortest WSL2 report. Compare `forward_cache_reload_speedup` and
`gradient_cache_reload_speedup` with the two `reload_*_warm_speedup` values:
the former isolates persistent-cache value across processes, while the latter
shows tracing/cache-load/dispatch overhead still paid above a true warm call.
The committed M4 CPU control is
`benchmarks/device_cache_reload_m4.json`; retain the generated JSON from a
WSL2 GPU run as the hardware-specific comparison artifact.
