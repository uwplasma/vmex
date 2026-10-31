#!/usr/bin/env python3
"""Stage and workflow timings of the raw-force block solver.

``--stages {qa,qi,none}`` times the compiled pieces of
:func:`vmex.core.implicit._raw_block_system` at the seed root of the QA
(``max_mode=2``, ``mpol=ntor=5``, 31 surfaces, block width 150) or QI
(``max_mode=3``) optimization problem: colored assembly plus scaling, the
SOLVAX factorization of the scaled bands, one-vector, 8-column and transposed
applications, the linearized-residual matvec, the complete
``_refine_block_factor_core`` executable and one ``_refine_block_step_core``
(Krylov matvecs plus applications plus the VMEX certificate). Every numeric
array and factor is a runtime argument of the executable; whole output pytrees
are synchronized. Per stage: compile, first execution, persistent-cache reload
(``jax.clear_caches`` then recompile), ``--warmups`` untimed calls and
``--repeats`` warm samples, retained output bytes, compiler temporaries, and the
device and resolved factor backend the stage ran on. Both the primal and the
transposed solves report their normwise backward error on the ORIGINAL
(unscaled) operator.

``--workflows`` times any of ``qa_jac`` (``residual_and_jac``), ``qa_grad``
(scalar ``value_and_grad``), ``qi_jac`` and ``fb_grad`` (free-boundary
``value_and_grad`` through the coupled root): one cold pass (compile or cache
reload included) and ``--passes`` warm passes over ``--points`` decision
vectors, with the per-pass deltas of the implicit solver's effort counters.

A fresh persistent-cache directory is used unless ``--cache-dir`` names one:
run twice with the same ``--cache-dir`` for the cache-reload workflow time.
Process peak memory (and the device peak where JAX reports it) closes the
report; run stages and workflows in separate processes for clean cold passes.

    python benchmarks/block_solver.py --stages qa --workflows qa_jac,qa_grad --output out.json
    python benchmarks/block_solver.py --block-factor cyclic_reduction --device gpu
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import solvax  # noqa: E402

import vmex  # noqa: E402
from vmex import optimize as opt  # noqa: E402
from vmex.core import implicit as im  # noqa: E402
from vmex.core.qi import ConstructedQIResidual  # noqa: E402

try:  # the block-factor backends exist from the factor-reuse/cyclic branches
    from vmex.core.device import resolve_block_factor
except ImportError:
    resolve_block_factor = None
SELECTABLE = "block_factor" in inspect.signature(im.make_config).parameters
NORMALIZED = "normalized=True" in inspect.getsource(im)  # factor-reuse branch onward
THOMAS_VARIANT = "thomas_normalized" if NORMALIZED else "thomas_stored_band"
QA_ASPECT, QI_OPTIONS = 6.0, dict(mboz=12, nboz=12, nphi=61, nalpha=18, n_bounce=21)


def problem_input(kind: str, smoke: bool):
    """Seed input and objective terms of the ``qa`` or ``qi`` problem."""
    mpol = 3 if smoke else 5
    if kind == "qa":
        inp = vmex.VmecInput.from_file(REPO / "examples" / "data" / "input.minimal_seed_nfp2")
        rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
        rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -0.05, 0.05
        inp = replace(inp, rbc=rbc, zbs=zbs)
        qs = opt.QuasisymmetryRatioResidual(np.linspace(0.1, 1.0, 10), helicity_m=1, helicity_n=0)
        terms = [(qs, 0.0, 1.0), (opt.aspect_ratio, QA_ASPECT, 1.0),
                 (opt.magnetic_well, 0.01, 1.0)]
    else:
        inp = vmex.VmecInput.from_file(REPO / "examples" / "data" / "input.QI_nfp2_initial")
        inp = replace(inp, ns_array=np.array([31]), ftol_array=np.array([1e-12]),
                      niter_array=np.array([5500]))
        options = dict(mboz=6, nboz=6, nphi=15, nalpha=5, n_bounce=5) if smoke else QI_OPTIONS
        terms = [(ConstructedQIResidual(np.linspace(0.1, 1.0, 6), **options), 0.0, 10.0),
                 (opt.aspect_ratio, 8.0, 0.01)]
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
    if smoke:
        inp = replace(inp, ns_array=np.asarray([9]))
    return inp, terms


def platform_of(tree) -> str | None:
    """``platform:id`` of the first device array in ``tree`` (None for host values)."""
    for leaf in jax.tree.leaves(tree):
        if isinstance(leaf, jax.Array):
            device = next(iter(leaf.devices()))
            return f"{device.platform}:{device.id}"
    return None


def nbytes(tree) -> int:
    return int(sum(leaf.nbytes for leaf in jax.tree.leaves(tree) if hasattr(leaf, "nbytes")))


ACTIVE_DEVICE = []  # the --device placement, for the device peak


def peaks() -> dict:
    """Process peak RSS and, where reported, the peak device allocation."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    stats = (ACTIVE_DEVICE or jax.devices())[0].memory_stats() or {}
    return {"process_peak_rss_bytes": int(rss) * (1 if sys.platform == "darwin" else 1024),
            "device_peak_bytes": stats.get("peak_bytes_in_use")}


def timed(fn, *args, repeats, warmups, **extra):
    """Compile, cache-reload, first, warm timings of ``jit(fn)(*args)``."""
    jax.block_until_ready(args)
    t0 = time.perf_counter()
    exe = jax.jit(fn).lower(*args).compile()
    compile_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = jax.block_until_ready(exe(*args))
    first_s = time.perf_counter() - t0
    jax.clear_caches()
    t0 = time.perf_counter()
    reload_exe = jax.jit(fn).lower(*args).compile()
    reload_s = time.perf_counter() - t0
    del reload_exe
    for _ in range(warmups):
        out = jax.block_until_ready(exe(*args))
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = jax.block_until_ready(exe(*args))
        samples.append(time.perf_counter() - t0)
    memory = exe.memory_analysis()
    return out, {
        "compile_seconds": compile_s, "first_seconds": first_s,
        "cache_reload_compile_seconds": reload_s,
        "median_seconds": float(np.median(samples)), "samples_seconds": samples,
        "output_bytes": nbytes(out), "device_array": platform_of(out),
        "temp_bytes": None if memory is None else int(memory.temp_size_in_bytes),
        **extra, **peaks()}


def band_matvec(lower, diagonal, upper, x, transpose=False):
    """``A x`` or ``A^T x`` for block bands where ``lower[i]`` couples to ``x[i-1]``."""
    def mul(blocks, y, t=False):
        return jnp.einsum("nji,njk->nik" if t else "nij,njk->nik", blocks, y)
    zero = jnp.zeros_like(x[:1])
    previous, following = jnp.concatenate([zero, x[:-1]]), jnp.concatenate([x[1:], zero])
    if not transpose:
        return mul(lower, previous) + mul(diagonal, x) + mul(upper, following)
    upper_t = jnp.concatenate([zero, mul(upper[:-1], x[:-1], True)])
    lower_t = jnp.concatenate([mul(lower[1:], x[1:], True), zero])
    return upper_t + mul(diagonal, x, True) + lower_t


def backward_error(operator, solution, rhs, transpose):
    """Per-column normwise backward error of ``A x = b`` (or ``A^T``) on the original bands."""
    residual = band_matvec(*operator, solution, transpose) - rhs
    norm_a = jnp.sqrt(sum(jnp.sum(b * b) for b in operator))
    column = lambda v: jnp.sqrt(jnp.sum(v * v, axis=(0, 1)))  # noqa: E731
    return float(jnp.max(column(residual) / (norm_a * column(solution) + column(rhs))))


def stage_timings(inp, repeats, warmups, save_bands, block_factor, device):
    kwargs = dict(block_factor=block_factor) if SELECTABLE else {}
    cfg = im.make_config(inp, device=device, **kwargs)
    # the seed root is solved on the CPU (physics is device independent; a GPU
    # solve costs minutes of compilation) and moved to the stage device
    cpu = jax.devices("cpu")[0]
    cpu_cfg = im.make_config(inp, device=cpu, **kwargs)
    with jax.default_device(cpu):
        state, mask = im.solve_implicit_with_aux(im.params_from_input(inp, device=cpu), cpu_cfg)
    place = (lambda t: jax.device_put(t, device)) if device is not None else (lambda t: t)
    params = im.params_from_input(inp, device=device)
    state, mask = place(state), place(mask)
    im._template_runtime(cfg)  # host-built; must not be first built while tracing
    fields = im._active_state_fields(cfg)
    probe = int(np.ceil(np.sqrt(len(fields) * int(mask.R_cos.shape[1]))))
    factor = (lambda *b: im._block_factor(*b, cfg)) if SELECTABLE else (
        lambda *b: solvax.block_thomas_factor(*b, normalized=NORMALIZED))
    resolved = resolve_block_factor(block_factor, device) if SELECTABLE else THOMAS_VARIANT

    def system(z, p, frozen, dof_mask, with_factors=False):
        return im._raw_block_system(p, cfg, frozen, dof_mask, fields, probe, z_star=z,
                                    factor=with_factors)

    def scaled(z, p, frozen, dof_mask):
        s = system(z, p, frozen, dof_mask)
        rs, cs = s.row_scale, s.column_scale
        previous = jnp.maximum(jnp.arange(rs.shape[0]) - 1, 0)
        following = jnp.minimum(jnp.arange(rs.shape[0]) + 1, rs.shape[0] - 1)
        return (rs[:, :, None] * s.lower * cs[previous][:, None, :],
                rs[:, :, None] * s.diagonal * cs[:, None, :],
                rs[:, :, None] * s.upper * cs[following][:, None, :])

    def original(z, p, frozen, dof_mask):
        s = system(z, p, frozen, dof_mask)
        return (s.lower, s.diagonal, s.upper), s.row_scale, s.column_scale

    args = (state, params, state, mask)
    context = {"block_factor_requested": block_factor, "block_factor_resolved": resolved}
    run = dict(repeats=repeats, warmups=warmups, **context)
    report = {}
    bands, report["assembly_and_scaling"] = timed(scaled, *args, **run)
    operator, rs, cs = jax.block_until_ready(jax.jit(original)(*args))
    factors, report["factor"] = timed(factor, *bands, **run)
    report["factor"]["retained_factor_bytes"] = nbytes(factors)
    rng = np.random.default_rng(0)
    n, m = bands[1].shape[:2]
    vector = jnp.asarray(rng.standard_normal((n, m, 1)))
    matrix = jnp.asarray(rng.standard_normal((n, m, 8)))
    for label, rhs, transpose in (("apply_vector", vector, False),
                                  ("apply_matrix8", matrix, False),
                                  ("apply_matrix8_transpose", matrix, True)):
        solution, report[label] = timed(
            lambda f, b, t=transpose: solvax.block_thomas_solve(f, b, transpose=t),
            factors, rhs, **run)
        # S = Rs A Cs: A x = b -> x = Cs S^-1 Rs b ; A^T x = b -> x = Rs S^-T Cs b
        left, right = (cs, rs) if transpose else (rs, cs)
        scaled_rhs = left[..., None] * rhs
        scaled_solution = solvax.block_thomas_solve(factors, scaled_rhs, transpose=transpose)
        report[label]["original_operator_backward_error"] = backward_error(
            operator, right[..., None] * scaled_solution, rhs, transpose)
        if not transpose:
            report[label]["solvax_scaled_relative_residual"] = float(jnp.max(
                solvax.block_tridiag_relative_residual(*bands, scaled_solution, scaled_rhs)))
    raw = lambda z, p, f, d: im.residual_fn(cfg, f, d, formulation="raw")(z, p)  # noqa: E731
    tangent = jax.tree.map(lambda a: jnp.asarray(rng.standard_normal(a.shape)), state)
    _, report["linearized_matvec"] = timed(
        lambda z, p, f, d, v: jax.jvp(lambda t: raw(t, p, f, d), (z,), (v,))[1],
        *args, tangent, **run)
    core, report["refine_block_factor_core"] = timed(
        lambda *a: im._refine_block_factor_core(*a, cfg), *args, **run)
    report["refine_block_factor_core"]["retained_factor_bytes"] = nbytes(core[0])
    step, report["refine_block_step"] = timed(
        lambda z, p, f, d, fac: im._refine_block_step_core(z, p, f, d, fac, cfg),
        *args, core[:3], **run)
    report["refine_block_step"]["certificate"] = {
        "residual_norm": float(step[2]), "krylov_iterations": int(step[3]),
        "linear_relative_residual": float(step[4])}
    if save_bands is not None:
        np.savez(save_bands, lower=bands[0], diag=bands[1], upper=bands[2])
    return {"n_blocks": int(n), "block_size": int(m), "active_fields": list(fields),
            "device": platform_of(bands), **report}


def workflows(names, inp_by_kind, n_points, passes, block_factor, device):
    out, rng = {}, np.random.default_rng(1)
    kwargs = dict(block_factor=block_factor) if SELECTABLE else {}
    for name in names:
        if name == "fb_grad":
            out[name] = free_boundary_workflow(n_points, passes, block_factor)
            continue
        kind, mode = name.split("_")
        inp, terms = inp_by_kind[kind]
        if mode == "jac":
            prob = opt.VmecProblem.from_tuples(inp, terms, max_mode=2 if kind == "qa" else 3,
                                               device=device, **kwargs)
            fn = prob.residual_and_jac
        else:
            prob = opt.VmecProblem.from_loss(
                inp, lambda s, r: opt.aspect_ratio(s, r) ** 2 + 1e3 * opt.magnetic_well(s, r) ** 2,
                max_mode=2, device=device, **kwargs)
            fn = prob.value_and_grad
        cfg = prob.metadata["config"]
        steps = [np.zeros_like(prob.x0)] + [
            0.01 * prob.scales * rng.standard_normal(prob.x0.shape) for _ in range(n_points - 1)]
        out[name] = run_passes(
            fn, prob.x0, steps, passes, cfg, context={
                "n_dofs": int(prob.x0.size), "config_device": str(cfg.device),
                "block_factor_requested": block_factor,
                "block_factor_resolved": resolve_block_factor(block_factor, cfg.device)
                if SELECTABLE else THOMAS_VARIANT})
    return out


@contextlib.contextmanager
def profiled(calls: dict):
    """Synchronized call count and wall time of the host-dispatched executables.

    The fixed-boundary block factorization and Newton step, and the
    free-boundary host adjoint with its transpose matvecs.
    """
    from vmex.core import freeboundary_implicit as fbi
    targets = [(im, "_refine_block_factor_core"), (im, "_refine_block_step_core"),
               (fbi, "_host_adjoint"), (fbi, "_transpose_matvec")]
    originals = [(module, name, getattr(module, name)) for module, name in targets
                 if hasattr(module, name)]
    for module, name, original in originals:
        def wrapper(*a, _f=original, _n=name, **k):
            t0 = time.perf_counter()
            out = jax.block_until_ready(_f(*a, **k))
            record = calls.setdefault(_n, {"calls": 0, "seconds": 0.0})
            record["calls"] += 1
            record["seconds"] += time.perf_counter() - t0
            return out
        setattr(module, name, wrapper)
    try:
        yield
    finally:
        for module, name, original in originals:
            setattr(module, name, original)


def run_passes(fn, x0, steps, passes, cfg, context):
    """Cold pass then ``passes`` warm passes; counter deltas and results per pass."""
    records, value = [], None
    for index in range(passes + 1):
        before = dict(im._SOLVE_STATS.get(cfg, {}))
        times, calls = [], {}
        with profiled(calls):
            for step in steps:
                t0 = time.perf_counter()
                value = jax.block_until_ready(fn(x0 + step))
                times.append(time.perf_counter() - t0)
        after = im._SOLVE_STATS.get(cfg, {})
        records.append({"pass": "cold" if index == 0 else "warm", "seconds": times,
                        "host_executables": calls,
                        "counters": {k: None if v is None else float(v) - float(before.get(k) or 0)
                                     for k, v in after.items()}})
    flat = np.concatenate([np.ravel(np.asarray(leaf)) for leaf in jax.tree.leaves(value)])
    return {**context, "passes": records, "checksum": float(np.sum(np.abs(flat))),
            "result_device": platform_of(value), **peaks()}


def free_boundary_workflow(n_points, passes, block_factor):
    """``value_and_grad`` of the aspect ratio through the coupled free-boundary root."""
    from essos.coils import Coils
    from essos.fields import BiotSavart

    from vmex.core.freeboundary_implicit import (
        make_free_boundary_config,
        solve_free_boundary_implicit,
    )
    data = REPO / "examples" / "data"
    ns, mpol, niter, ftol = 25, 5, 12000, 1e-10
    coils = Coils.from_json(str(data / "ESSOS_biot_savart_LandremanPaulQA.json"))
    base, n_curve = coils.dofs, coils.dofs_curves.size
    scales = jnp.concatenate((jnp.full(n_curve, 0.02), 0.2 * jnp.abs(base[n_curve:])))
    inp = vmex.VmecInput.from_file(data / "input.LandremanPaul2021_QA_lowres").change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=16)
    inp = replace(inp, lfreeb=True, mgrid_file="direct ESSOS field", phiedge=-0.025,
                  ns_array=np.array([ns]), niter_array=np.array([niter]),
                  ftol_array=np.array([ftol]))
    params = im.params_from_input(inp)
    config = make_free_boundary_config(
        inp, BiotSavart(coils), ns=ns, ftol=ftol, max_iterations=niter, adjoint_tol=1e-9,
        field_from_parameters=lambda p: BiotSavart(coils.with_dofs(base + scales * p)),
        **(dict(block_factor=block_factor) if SELECTABLE else {}))
    context = im.runtime_from_params(params, config.implicit)
    forward = lambda p: im.aspect_ratio(  # noqa: E731
        solve_free_boundary_implicit(params, p, config), context)
    fn = jax.value_and_grad(forward)
    rng = np.random.default_rng(1)
    steps = [np.zeros(base.size)] + [0.01 * rng.standard_normal(base.size)
                                     for _ in range(n_points - 1)]
    resolved = (resolve_block_factor(block_factor, config.implicit.device)
                if SELECTABLE else THOMAS_VARIANT)
    result = run_passes(fn, jnp.zeros(base.size), steps, passes, config.implicit,
                        {"n_dofs": int(base.size), "config_device": str(config.implicit.device),
                         "block_factor_resolved": resolved})
    # the value alone (root plus anchor, no adjoint), warm, at the same points
    jax.block_until_ready(forward(jnp.asarray(steps[0])))  # compile
    forward_times = []
    for step in steps * passes:
        t0 = time.perf_counter()
        jax.block_until_ready(forward(jnp.asarray(step)))
        forward_times.append(time.perf_counter() - t0)
    result["forward_only_seconds"] = forward_times
    return result


def metadata(cache_dir, cache_files_before):
    def sha(path):
        try:
            return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                           text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"vmex_sha": sha(REPO), "solvax_version": solvax.__version__,
            "vmex_version": vmex.__version__, "jax_version": jax.__version__,
            "default_device": str(jax.devices()[0]), "devices": [str(d) for d in jax.devices()],
            "x64": bool(jax.config.read("jax_enable_x64")),
            "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "load_average_start": list(os.getloadavg()), "argv": sys.argv,
            "xla_flags": os.environ.get("XLA_FLAGS", ""), "cache_dir_files_before": cache_files_before,
            "cache_dir_state": "warm" if cache_files_before else "cold"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", choices=("qa", "qi", "none"), default="qa")
    parser.add_argument("--workflows", default="qa_jac,qa_grad",
                        help="comma list of qa_jac, qa_grad, qi_jac, fb_grad, or none")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--points", type=int, default=3)
    parser.add_argument("--passes", type=int, default=2, help="warm workflow passes")
    parser.add_argument("--block-factor", default="auto",
                        choices=("auto", "thomas", "cyclic_reduction"))
    parser.add_argument("--device", default="default", choices=("default", "cpu", "gpu"),
                        help="placement of stages and workflows (default: JAX's default device)")
    parser.add_argument("--cache-dir", type=Path, help="persistent cache (default: a fresh one)")
    parser.add_argument("--smoke", action="store_true", help="mpol=3, 9 surfaces")
    parser.add_argument("--save-bands", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cache = args.cache_dir or Path(tempfile.mkdtemp(prefix="vmex_bench_cache_"))
    cache.mkdir(parents=True, exist_ok=True)
    before = sum(1 for p in cache.rglob("*") if p.is_file())
    jax.config.update("jax_enable_compilation_cache", True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    device = None if args.device == "default" else jax.devices(args.device)[0]
    ACTIVE_DEVICE.extend([device] if device is not None else [])
    names = [] if args.workflows == "none" else args.workflows.split(",")
    inputs = {k: problem_input(k, args.smoke) for k in {n.split("_")[0] for n in names} & {"qa", "qi"}}
    result = {"benchmark": "vmex_block_solver", **metadata(cache, before),
              "requested_device": args.device, "uptime": os.popen("uptime").read().strip()}
    if args.stages != "none":
        inp = problem_input(args.stages, args.smoke)[0]
        result["stages"] = stage_timings(inp, args.repeats, args.warmups, args.save_bands,
                                         args.block_factor, device)
    if names:
        result["workflows"] = workflows(names, inputs, args.points, args.passes,
                                        args.block_factor, device)
    result["cache_dir_files_after"] = sum(1 for p in cache.rglob("*") if p.is_file())
    result["load_average_end"] = list(os.getloadavg())
    result.update(peaks())
    rendered = json.dumps(result, indent=1, default=str) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
