#!/usr/bin/env python
"""Solve the shared initial coils' free boundary with direct Biot-Savart.

No coil fitting or optimization. Both scalar single-stage examples load the
same coils.initial.scalar.json. This script saves WOUT, coils and plots.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

DEVICE = "cpu"  # change to "gpu" on the CUDA server, or pass --device gpu
WALL_SECONDS = 600
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from single_stage_support import common
INPUT = common.DATA / "input.rotating_ellipse"
COILS = common.DATA / "coils.initial.scalar.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default=DEVICE)
    parser.add_argument("--output", type=Path, default=HERE / "runs" / time.strftime("initial-free-%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # Keep caches local; select float64 and the device before importing JAX.
    for key in ("TMPDIR", "XDG_CACHE_HOME", "MPLCONFIGDIR", "CUDA_CACHE_PATH"):
        cache = out / "cache" / key
        cache.mkdir(parents=True)
        os.environ[key] = str(cache)
    os.environ.update(JAX_ENABLE_X64="1", JAX_PLATFORMS="cuda" if args.device == "gpu" else "cpu",
        VMEX_COMPILATION_CACHE="disabled", JAX_ENABLE_COMPILATION_CACHE="false", MPLBACKEND="Agg")
    sys.path.insert(0, str(HERE.parents[1]))  # this VMEX checkout
    import jax
    import numpy as np
    import vmex as vj
    from vmex import optimize as opt
    from essos.coils import Coils
    from essos.fields import BiotSavart
    from essos.surfaces import SurfaceRZFourier, surfacerzfourier_from_boundary

    def timeout(_signum, _frame):
        raise TimeoutError("manual solve reached its wall-time budget")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(WALL_SECONDS)
    metadata = json.loads(COILS.with_suffix(".metadata.json").read_text())
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    if sha(COILS) != metadata["coils_sha256"] or sha(INPUT) != metadata["input_sha256"]:
        raise ValueError("shared coil/input hashes differ from the fitted case")
    shutil.copyfile(COILS, out / COILS.name)
    (out / "provenance.json").write_text(json.dumps(dict(input_sha256=sha(INPUT), coils_sha256=sha(COILS),
        script_sha256=sha(Path(__file__)), vmex=vj.__file__, jax=jax.__version__, device=str(jax.devices()[0])), indent=2)+"\n")

    inp = vj.VmecInput.from_file(INPUT)
    coils = Coils.from_json(str(COILS))
    field = BiotSavart(coils)  # evaluated directly; no mgrid interpolation
    print(f"Shared initial coils: {COILS}\nDevice: {jax.devices()[0]}", flush=True)
    started = time.perf_counter()
    seed = opt.solve_equilibrium(inp, device=args.device, raise_on_max_iterations=True,
                                 polish_force_balance=False)
    inp = replace(inp, lfreeb=True, mgrid_file="direct ESSOS field")
    result = vj.solve_free_boundary(
        inp, external_field=field, initial_state=seed.state, device=args.device,
        verbose=True, include_edge_in_convergence=True,
        edge_force_tolerance=float(inp.ftol_array[-1]), use_fft=False, jacobian_retries=0)
    solve_seconds = time.perf_counter()-started

    # Save the actual free equilibrium; the input remains its starting guess.
    inp.to_indata(out / "input.initial_free")
    wout = vj.wout_from_state(inp=inp, state=result.state, fsqr=result.fsqr, fsqz=result.fsqz,
        fsql=result.fsql, niter=result.iterations, converged=result.converged, vacuum_output=result.vacuum,
        nextcur=len(coils.dofs_currents_raw), extcur=np.asarray(coils.dofs_currents_raw))
    wout_path = vj.write_wout(out / "wout_initial_free.nc", wout)
    summary = dict(converged=bool(result.converged), iterations=int(result.iterations),
        solve_seconds=solve_seconds, aspect=float(wout.aspect), B0_T=float(wout.b0),
        **{name: float(getattr(result, name)) for name in ("fsqr", "fsqz", "fsql", "fedge")})
    (out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)

    initial_surface = surfacerzfourier_from_boundary(inp.rbc, inp.zbs, inp.nfp, nphi=61, ntheta=64)
    free_surface = SurfaceRZFourier.from_wout_file(wout_path, nphi=61, ntheta=64)
    vj.plot_optimization_objects(out / "fixed_start_vs_free_boundary.png",
        ("Fixed-boundary start", initial_surface, coils), ("Free-boundary solution", free_surface, coils))
    coils.to_vtk(str(out / "initial_coils"))
    free_surface.to_vtk(str(out / "initial_free_surface"), field=field)
    for path in vj.plot_wout(wout_path, out).values():
        print(f"Wrote {path}")
    signal.alarm(0)
    print(f"Results: {out}", flush=True)


if __name__ == "__main__":
    main()
