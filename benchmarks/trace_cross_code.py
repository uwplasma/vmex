#!/usr/bin/env python3
"""Compare common-birth alpha losses in VMEX, SIMPLE and SIMSOPT.

Use a reactor-scale WOUT and ``--cores`` matched CPU workers. Axis-stop or
energy-failed orbits are unresolved, not confined. Timings exclude field setup; first VMEX calls include compilation. See the alpha-tracing guide for controls.

Usage::

    python benchmarks/trace_cross_code.py WOUT --simple PATH/simple.x \
        [--particles 64] [--tmax 2e-3] [--cores 8] [--output result.json]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

MBOOZ = 32
N_TIMES = 101


def run_vmex(wout, n, tmax, s):
    from vmex.core.tracing import trace_alphas

    kw = dict(tmax=tmax, nparticles=n, s=s, scale=None, times_to_trace=N_TIMES)
    cold = trace_alphas(wout, **kw)
    res = trace_alphas(wout, **kw)
    if res.particles_failed or not np.isfinite(res.energy_error).all():
        raise ValueError(f"VMEX had {res.particles_failed} failed or nonfinite trajectories")
    if (not np.array_equal(cold.initial_conditions, res.initial_conditions)
            or not np.array_equal(cold.lost_times, res.lost_times)):
        raise ValueError("VMEX births or losses changed between cold and warm calls")
    t_loss = np.where(res.lost_times >= 0, res.lost_times, np.inf)
    run = res.wall_time_s
    return (
        res.initial_conditions,
        t_loss,
        run,
        dict(
            method=f"fixed-step RK4, dt = {res.metadata['timestep']:.3g} s",
            boozer_modes=res.metadata["boozer_modes"],
            cold_wall_s=cold.wall_time_s,
            compile_s=cold.metadata["compile_time_s"],
            warm_compile_s=res.metadata["compile_time_s"],
            wall_s=res.wall_time_s,
            max_energy_error=float(np.max(res.energy_error)),
        ),
    )


_FIELD = {}


def _simsopt_chunk(args):
    from simsopt.field import (MaxToroidalFluxStoppingCriterion,
                               MinToroidalFluxStoppingCriterion, trace_particles_boozer)

    stz, vpar, tmax, mass, charge, energy, axis_floor = args
    start = time.perf_counter()
    paths, hits = trace_particles_boozer(
        _FIELD["field"],
        stz,
        vpar,
        tmax=tmax,
        mass=mass,
        charge=charge,
        Ekin=energy,
        tol=1e-9,
        mode="gc_noK",
        forget_exact_path=False,
        stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0),
                           MinToroidalFluxStoppingCriterion(axis_floor)],
    )
    trace_s = time.perf_counter() - start
    t_loss = np.full(len(stz), np.inf)
    unresolved = np.zeros(len(stz), dtype=bool)
    errors = []
    field = _FIELD["field"]
    for i, (path, hit) in enumerate(zip(paths, hits)):
        if not np.isfinite(path).all():
            raise ValueError(f"SIMSOPT trajectory {i} failed or terminated early")
        if len(hit):
            code = int(hit[0, 1])
            if code == -1:
                t_loss[i] = hit[0, 0]
            elif code == -2:
                unresolved[i] = True
            else:
                raise ValueError(f"SIMSOPT trajectory {i} hit unexpected stop {code}")
        elif path[-1, 0] < tmax - 1e-15:
            raise ValueError(f"SIMSOPT trajectory {i} terminated early")
        if unresolved[i]:
            continue
        if np.min(path[:, 1]) < axis_floor:
            raise ValueError(f"SIMSOPT trajectory {i} passed its inner-flux stop")
        field.set_points(np.ascontiguousarray(path[:, 1:4]))
        B = np.asarray(field.modB()).ravel()
        mu = energy * (1 - (vpar[i] / np.sqrt(2 * energy / mass)) ** 2) / B[0]
        H = 0.5 * mass * path[:, 4] ** 2 + mu * B
        errors.append(np.max(np.abs(H / energy - 1)))
    return t_loss, trace_s, float(max(errors, default=0)), unresolved


def _simsopt_field(wout):
    from types import SimpleNamespace

    import booz_xform
    import netCDF4
    import simsopt.field.boozermagneticfield as bmf
    from simsopt.field import BoozerRadialInterpolant, InterpolatedBoozerField

    bx = booz_xform.Booz_xform()
    bx.verbose = 0
    bx.read_wout(str(wout))
    bx.mboz, bx.nboz = MBOOZ, MBOOZ
    bx.run()
    with netCDF4.Dataset(wout) as ds:
        flux = SimpleNamespace(phi=ds["phi"][:].data, chi=ds["chi"][:].data)
    # simsopt wraps booz_xform in simsopt.mhd.Boozer, which needs MPI and the
    # VMEC Python wrapper; the interpolant reads only these attributes of it.
    boozer = type("Boozer", (), {})()
    boozer.__dict__.update(
        bx=bx, mpi=None, need_to_run_code=False, equil=SimpleNamespace(s_half_grid=np.asarray(bx.s_in), wout=flux)
    )
    bmf.Vmec, bmf.Boozer = type("Vmec", (), {}), type(boozer)
    bri = BoozerRadialInterpolant(boozer, order=3, no_K=True, mpol=MBOOZ, ntor=MBOOZ)
    nfp = int(bx.nfp)
    _FIELD["field"] = InterpolatedBoozerField(
        bri,
        degree=3,
        srange=(0, 1, 15),
        thetarange=(0, np.pi, 15),
        zetarange=(0, 2 * np.pi / nfp, 15),
        extrapolate=True,
        nfp=nfp,
        stellsym=True,
    )
    psi0 = -float(flux.phi[-1]) / (2 * np.pi)
    if not np.isclose(float(_FIELD["field"].psi0), psi0, rtol=1e-12, atol=0):
        raise ValueError("SIMSOPT Boozer psi0 has the wrong sign for VMEC toroidal flux")
    _FIELD["interpolant"] = bri  # the interpolated field calls back into it


def run_simsopt(wout, births, tmax, cores, axis_floor):
    import multiprocessing as mp

    from essos import constants as c

    speed = np.sqrt(2 * c.FUSION_ALPHA_PARTICLE_ENERGY / c.ALPHA_PARTICLE_MASS)
    _simsopt_field(wout)
    _FIELD["field"].set_points(np.ascontiguousarray(births[:, :3]))
    B_start = np.asarray(_FIELD["field"].modB()).ravel().copy()
    consts = (c.ALPHA_PARTICLE_MASS, c.ALPHA_PARTICLE_CHARGE, c.FUSION_ALPHA_PARTICLE_ENERGY)
    # The interpolation tables are built lazily on the first orbit: build them
    # here, once, so the forked workers inherit them and only trace.
    _simsopt_chunk((births[:1, :3], births[:1, 3] * speed, 1e-7, *consts, axis_floor))
    chunks = [i for i in np.array_split(np.arange(len(births)), cores) if len(i)]
    jobs = [(births[i, :3], births[i, 3] * speed, tmax, *consts, axis_floor) for i in chunks]
    with mp.get_context("fork").Pool(len(chunks)) as pool:
        out = pool.map(_simsopt_chunk, jobs, chunksize=1)
    t_loss = np.concatenate([o[0] for o in out])
    unresolved = np.concatenate([o[3] for o in out])
    t_loss[unresolved] = np.nan
    return (
        t_loss,
        max(o[1] for o in out),
        dict(method="adaptive RK45, tol = 1e-9 (gc_noK)", core_seconds=float(sum(o[1] for o in out)),
             max_resolved_path_energy_error=float(max(o[2] for o in out)),
             unresolved_ids=np.flatnonzero(unresolved).tolist(), axis_floor=axis_floor,
             B_start_T=B_start),
    )


def run_simple(simple_x, wout, births, tmax, cores, npoiper2, integmode):
    import netCDF4

    from essos import constants as c

    n = len(births)
    with tempfile.TemporaryDirectory(prefix="simple_") as tmp:
        tmp = Path(tmp)
        (tmp / "wout.nc").symlink_to(Path(wout).resolve())
        np.savetxt(
            tmp / "start.dat", np.column_stack([births[:, :3], np.ones(n), births[:, 3]]), fmt="%.16e"
        )
        # SIMPLE util.F90 (9269e86) uses rounded cgs proton mass and energy units.
        (tmp / "simple.in").write_text(
            "&config\n"
            f"trace_time = {tmax:.16e}\nntestpart = {n}\nnetcdffile = 'wout.nc'\n"
            f"isw_field_type = 2\nstartmode = 6\nintegmode = {integmode}\n"
            f"npoiper2 = {npoiper2}\nntimstep = 401\noutput_orbits_macrostep = .True.\n"
            "contr_pp = -1d10\nnotrace_passing = 0\ndeterministic = .True.\n"
            "fast_class = .False.\ntcut = -1d0\n"
            f"n_d = {c.ALPHA_PARTICLE_MASS / 1.6726e-27:.16e}\nn_e = 2\n"
            f"facE_al = {3.5e6 * 1.6022e-19 / c.FUSION_ALPHA_PARTICLE_ENERGY:.16e}\n/\n"
        )
        env = dict(os.environ, OMP_NUM_THREADS=str(cores))
        log = subprocess.run(
            [str(Path(simple_x).resolve()), "simple.in"], cwd=tmp, env=env, check=True, capture_output=True, text=True
        ).stdout
        codes = np.atleast_2d(np.loadtxt(tmp / "orbit_exit_code.dat"))
        lost_rows = np.atleast_2d(np.loadtxt(tmp / "times_lost.dat"))
        with netCDF4.Dataset(tmp / "orbits.nc") as ds:
            p = np.ma.filled(ds["p_abs"][:], np.nan)
            s_path = np.ma.filled(ds["s"][:], np.nan)
            t_path = np.ma.filled(ds["time"][:], np.nan)
    run = next(
        float(line.split("completed")[1].split()[0])
        for line in log.splitlines()
        if "Particle tracing completed" in line
    )
    order = np.argsort(codes[:, 0])
    if not np.array_equal(codes[order, 0], np.arange(1, n + 1)):
        raise ValueError("SIMPLE exit rows do not match the input particles")
    code, t = codes[order, 1].astype(int), codes[order, 2]
    if not np.isin(code, (0, 1)).all():
        raise ValueError(f"SIMPLE returned numerical or unsupported exit codes: {np.unique(code).tolist()}")
    lost_rows = lost_rows[np.argsort(lost_rows[:, 0])]
    if not np.array_equal(lost_rows[:, 0], np.arange(1, n + 1)):
        raise ValueError("SIMPLE diagnostic rows do not match the input particles")
    B_start = (1 - births[:, 3] ** 2) / lost_rows[:, 4] * 1e-4
    energy_error = float(np.max(np.abs(lost_rows[:, 8] ** 2 - 1)))
    if (p.shape != (401, n) or s_path.shape != p.shape or t_path.shape != p.shape
            or not np.isfinite(p[0]).all() or not np.isfinite(s_path[0]).all()
            or not np.isfinite(t_path[0]).all()):
        raise ValueError("SIMPLE saved orbits are incomplete")
    path_energy_error = float(np.nanmax(np.abs(p * p - 1)))
    if not np.isfinite(B_start).all() or not np.isfinite([energy_error, path_energy_error]).all():
        raise ValueError("SIMPLE reported nonfinite field or energy")
    return (
        np.where(code == 1, t, np.inf),
        run,
        dict(
            method=f"symplectic {'midpoint' if integmode == 3 else 'Euler'} "
                   f"(integmode {integmode}, npoiper2={npoiper2}, 401 macrostep samples)",
            mass_kg=float(c.ALPHA_PARTICLE_MASS),
            energy_J=float(c.FUSION_ALPHA_PARTICLE_ENERGY),
            constants_source="SIMPLE util.F90, 9269e86",
            max_endpoint_energy_error=energy_error,
            max_saved_path_energy_error=path_energy_error,
            B_start_T=B_start,
        ),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("wout")
    ap.add_argument("--simple", required=True, help="path to simple.x")
    ap.add_argument("--particles", type=int, default=64)
    ap.add_argument("--tmax", type=float, default=2e-3)
    ap.add_argument("--s", type=float, default=0.283333, help="snapped to the nearest half-grid surface")
    ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--simple-npoiper2", type=int, default=256)
    ap.add_argument("--simple-integmode", type=int, choices=(1, 3), default=3,
                    help="SIMPLE symplectic Euler (1) or midpoint (3)")
    ap.add_argument("--energy-tol", type=float, default=1e-3, help="maximum relative energy drift")
    ap.add_argument("--simsopt-axis-floor", type=float, default=1e-3,
                    help="stop SIMSOPT when s reaches this inner flux surface")
    ap.add_argument("--output", type=Path, help="write the benchmark JSON to this path")
    args = ap.parse_args()
    if (args.particles < 1 or args.cores < 1 or args.simple_npoiper2 < 1
            or not 0 < args.s < 1 or not np.isfinite(args.tmax) or args.tmax <= 0
            or not np.isfinite(args.energy_tol) or args.energy_tol <= 0
            or not np.isfinite(args.simsopt_axis_floor) or not 0 < args.simsopt_axis_floor < args.s):
        ap.error("positive particles, cores, simple-npoiper2, tmax, energy-tol and "
                 "0 < simsopt-axis-floor < s < 1 required")
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={args.cores}"
    os.environ["OMP_NUM_THREADS"] = "1"  # SIMSOPT forks one process per core
    import netCDF4

    with netCDF4.Dataset(args.wout) as ds:
        ns = int(ds["ns"][:])
    j = int(np.clip(round(args.s * (ns - 1) - 0.5), 0, ns - 2))
    s = (j + 0.5) / (ns - 1)

    births, t_vmex, run_v, meta_v = run_vmex(args.wout, args.particles, args.tmax, s)
    from vmex.core.tracing import boozer_field
    from vmex.core.wout import read_wout

    field, _ = boozer_field(read_wout(args.wout))
    import jax

    B_vmex = np.asarray(jax.vmap(field.modB)(*(births[:, :3].T)))
    t_simple, run_s, meta_s = run_simple(
        args.simple, args.wout, births, args.tmax, args.cores,
        args.simple_npoiper2, args.simple_integmode)
    b_err = float(np.abs(meta_s.pop("B_start_T") / B_vmex - 1).max())
    t_sims, run_o, meta_o = run_simsopt(
        args.wout, births, args.tmax, args.cores, args.simsopt_axis_floor)
    sims_b_err = float(np.abs(meta_o.pop("B_start_T") / B_vmex - 1).max())
    errors = (meta_v["max_energy_error"], meta_s["max_saved_path_energy_error"],
              meta_o["max_resolved_path_energy_error"])
    if (not np.isfinite([*errors, b_err, sims_b_err]).all()
            or max(errors) > args.energy_tol or max(b_err, sims_b_err) > args.energy_tol):
        raise ValueError(f"energy drifts {errors} or start-field errors {(b_err, sims_b_err)} "
                         f"exceed {args.energy_tol:g}")

    times = np.linspace(0, args.tmax, N_TIMES)
    codes = {}
    for name, t_loss, run, meta in (
        ("VMEX", t_vmex, run_v, meta_v),
        ("SIMPLE", t_simple, run_s, meta_s),
        ("SIMSOPT", t_sims, run_o, meta_o),
    ):
        valid = ~np.isnan(t_loss)
        n_valid = int(valid.sum())
        if not n_valid:
            raise ValueError(f"{name} has no resolved trajectories")
        lost = np.isfinite(t_loss) & valid
        f = float(lost.sum() / n_valid)
        codes[name] = dict(
            meta,
            resolved=n_valid,
            lost=int(lost.sum()),
            loss_fraction=f,
            sigma=float(np.sqrt(f * (1 - f) / n_valid)),
            run_s=float(run),
            curve=[float((t_loss[valid] <= t).mean()) for t in times],
            label_agreement_with_vmex=int(np.count_nonzero(
                lost[valid] == np.isfinite(t_vmex[valid]))),
        )
        print(f"{name:8s} {100 * f:6.2f}% ± {100 * codes[name]['sigma']:.2f}% "
              f" ({n_valid}/{len(t_loss)} resolved)  run {run:8.2f} s")
    from importlib.metadata import version

    record = dict(
        schema="vmex.trace-cross-code/2",
        wout=Path(args.wout).name,
        s=s,
        particles=args.particles,
        tmax=args.tmax,
        cores=args.cores,
        times=times.tolist(),
        codes=codes,
        start_field=dict(max_simple_modB_mismatch_rel=b_err,
                         max_simsopt_modB_mismatch_rel=sims_b_err),
        energy_tolerance=args.energy_tol,
        host=dict(
            machine=platform.machine(),
            processor=platform.processor() or None,
            system=platform.system(),
            date=time.strftime("%Y-%m-%d"),
        ),
        versions={k: version(k) for k in ("vmex", "essos", "jax", "simsopt", "booz_xform", "booz_xform_jax")},
    )
    target = json.dumps(record, indent=1) + "\n"
    if args.output:
        args.output.write_text(target)
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(target)


if __name__ == "__main__":
    main()
