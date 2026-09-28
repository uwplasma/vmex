#!/usr/bin/env python3
"""Alpha losses from ``vmex --trace``, SIMPLE and SIMSOPT on the same particles.

One equilibrium, one set of births and one horizon for three guiding-centre
codes:

* VMEX: :func:`vmex.core.tracing.trace_alphas` (ESSOS Boozer tracer, fixed-step RK4);
* SIMSOPT: ``trace_particles_boozer`` in ``gc_noK`` mode (adaptive RK45) on a
  ``booz_xform`` field with the same Boozer resolution;
* SIMPLE (github.com/itpplasma/SIMPLE): ``simple.x`` with its default
  symplectic integrator in Boozer coordinates built from the VMEC file.

The births are those of ``vmex --trace`` (``s`` on a half-grid surface, Boozer
angles weighted by the Jacobian, pitch uniform).  SIMPLE reads starts in VMEC
angles, so the Boozer angles are mapped to VMEC ``(theta, phi)`` with the
``booz_xform`` ``nu`` and the VMEC ``lambda`` on that surface; the mapping is
checked on ``R`` and ``Z`` and SIMPLE's own ``|B|`` at the start.

Every code runs on the same ``--cores`` CPU cores (``taskset``): JAX with one
device per core, SIMPLE with OpenMP, SIMSOPT with one process per core.  The
runtime excludes compilation and set-up: JAX compilation (JAX monitoring
events), the SIMPLE field initialisation (its phase timer), and the SIMSOPT
field interpolation.  The record goes to ``benchmarks/trace_cross_code.json``;
``docs/_static/figures/sources/make_trace_benchmark_figure.py`` plots it.

Usage::

    python benchmarks/trace_cross_code.py WOUT --simple PATH/simple.x \
        [--particles 1000] [--tmax 1e-2] [--cores 8]
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

REPO = Path(__file__).resolve().parents[1]
RECORD = REPO / "benchmarks" / "trace_cross_code.json"
MBOOZ = 32
N_TIMES = 400


def vmec_angles(wout_path, bx, j, s, theta_b, zeta_b):
    """VMEC ``(theta, phi)`` of Boozer ``(theta_b, zeta_b)`` on half-grid surface ``j``.

    ``zeta_b = phi + nu`` and ``theta_b - iota zeta_b = theta* - iota phi`` with
    ``theta* = theta + lambda``; ``theta`` follows by Newton.  Returns the
    angles and the largest ``R, Z`` mismatch [m] between the two descriptions.
    """
    import netCDF4

    with netCDF4.Dataset(wout_path) as ds:
        xm, xn = ds["xm"][:], ds["xn"][:]
        lmns = ds["lmns"][:][j + 1]
        rmnc, zmns = (0.5 * (ds[k][:][j] + ds[k][:][j + 1]) for k in ("rmnc", "zmns"))
        ns = int(ds["ns"][:])
    assert abs(s - (j + 0.5) / (ns - 1)) < 1e-12
    mb, nb = np.asarray(bx.xm_b), np.asarray(bx.xn_b)
    ang_b = np.outer(theta_b, mb) - np.outer(zeta_b, nb)
    nu = np.sin(ang_b) @ np.asarray(bx.numns_b)[:, j]
    iota = float(np.asarray(bx.iota)[j])
    phi = zeta_b - nu
    theta_star = theta_b - iota * nu
    theta = theta_star.copy()
    for _ in range(50):
        ang = np.outer(theta, xm) - np.outer(phi, xn)
        f = theta + np.sin(ang) @ lmns - theta_star
        theta -= f / (1 + np.cos(ang) @ (xm * lmns))
    ang = np.outer(theta, xm) - np.outer(phi, xn)
    r_v, z_v = np.cos(ang) @ rmnc, np.sin(ang) @ zmns
    r_b = np.cos(ang_b) @ np.asarray(bx.rmnc_b)[:, j]
    z_b = np.sin(ang_b) @ np.asarray(bx.zmns_b)[:, j]
    return theta, phi, float(max(np.abs(r_v - r_b).max(), np.abs(z_v - z_b).max()))


def run_vmex(wout, n, tmax, s):
    from vmex.core.tracing import trace_alphas

    res = trace_alphas(wout, tmax=tmax, nparticles=n, s=s, scale=None, times_to_trace=N_TIMES)
    t_loss = np.where(res.lost_times >= 0, res.lost_times, np.inf)
    run = res.wall_time_s - res.metadata["compile_time_s"]
    return (
        res.initial_conditions,
        t_loss,
        run,
        dict(
            method=f"fixed-step RK4, dt = {res.metadata['timestep']:.3g} s",
            boozer_modes=res.metadata["boozer_modes"],
            compile_s=res.metadata["compile_time_s"],
            wall_s=res.wall_time_s,
        ),
    )


_FIELD = {}


def _simsopt_chunk(args):
    from simsopt.field import MaxToroidalFluxStoppingCriterion, trace_particles_boozer

    stz, vpar, tmax, mass, charge, energy = args
    start = time.perf_counter()
    _, hits = trace_particles_boozer(
        _FIELD["field"],
        stz,
        vpar,
        tmax=tmax,
        mass=mass,
        charge=charge,
        Ekin=energy,
        tol=1e-9,
        mode="gc_noK",
        forget_exact_path=True,
        stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
    )
    t_loss = [h[0, 0] if len(h) and h[0, 1] < 0 else np.inf for h in hits]
    return np.array(t_loss), time.perf_counter() - start


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
    _FIELD["interpolant"] = bri  # the interpolated field calls back into it


def run_simsopt(wout, births, tmax, cores):
    import multiprocessing as mp

    from essos import constants as c

    speed = np.sqrt(2 * c.FUSION_ALPHA_PARTICLE_ENERGY / c.ALPHA_PARTICLE_MASS)
    _simsopt_field(wout)
    consts = (c.ALPHA_PARTICLE_MASS, c.ALPHA_PARTICLE_CHARGE, c.FUSION_ALPHA_PARTICLE_ENERGY)
    # The interpolation tables are built lazily on the first orbit: build them
    # here, once, so the forked workers inherit them and only trace.
    _simsopt_chunk((births[:1, :3], births[:1, 3] * speed, 1e-7, *consts))
    chunks = np.array_split(np.arange(len(births)), cores)
    jobs = [(births[i, :3], births[i, 3] * speed, tmax, *consts) for i in chunks]
    with mp.get_context("fork").Pool(cores) as pool:
        out = pool.map(_simsopt_chunk, jobs, chunksize=1)
    return (
        np.concatenate([o[0] for o in out]),
        max(o[1] for o in out),
        dict(method="adaptive RK45, tol = 1e-9 (gc_noK)", core_seconds=float(sum(o[1] for o in out))),
    )


def run_simple(simple_x, wout, births, theta_v, phi_v, tmax, cores):
    from essos import constants as c

    n = len(births)
    with tempfile.TemporaryDirectory(prefix="simple_") as tmp:
        tmp = Path(tmp)
        (tmp / "wout.nc").symlink_to(Path(wout).resolve())
        np.savetxt(
            tmp / "start.dat", np.column_stack([births[:, 0], theta_v, phi_v, np.ones(n), births[:, 3]]), fmt="%.16e"
        )
        # 3.5 MeV / facE_al is the energy and n_d the mass in proton masses.
        (tmp / "simple.in").write_text(
            "&config\n"
            f"trace_time = {tmax:.6e}\nntestpart = {n}\nnetcdffile = 'wout.nc'\n"
            "isw_field_type = 2\nstartmode = 2\ncontr_pp = -1d10\ndeterministic = .True.\n"
            f"n_d = {c.ALPHA_PARTICLE_MASS / c.PROTON_MASS:.8f}\nn_e = 2\n"
            f"facE_al = {3.5e6 * c.ELEMENTARY_CHARGE / c.FUSION_ALPHA_PARTICLE_ENERGY:.8f}\n/\n"
        )
        env = dict(os.environ, OMP_NUM_THREADS=str(cores))
        log = subprocess.run(
            [str(Path(simple_x).resolve()), "simple.in"], cwd=tmp, env=env, check=True, capture_output=True, text=True
        ).stdout
        codes = np.loadtxt(tmp / "orbit_exit_code.dat")
        lost_rows = np.loadtxt(tmp / "times_lost.dat")
    run = next(
        float(line.split("completed")[1].split()[0])
        for line in log.splitlines()
        if "Particle tracing completed" in line
    )
    order = np.argsort(codes[:, 0])
    code, t = codes[order, 1].astype(int), codes[order, 2]
    B_start = (1 - births[:, 3] ** 2) / lost_rows[np.argsort(lost_rows[:, 0]), 4] * 1e-4
    return (
        np.where(code == 1, t, np.inf),
        run,
        dict(
            method="symplectic Euler (integmode 1, SIMPLE defaults)",
            unresolved=int((code >= 100).sum()),
            B_start_T=B_start,
        ),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("wout")
    ap.add_argument("--simple", required=True, help="path to simple.x")
    ap.add_argument("--particles", type=int, default=1000)
    ap.add_argument("--tmax", type=float, default=1e-2)
    ap.add_argument("--s", type=float, default=0.25, help="snapped to the nearest half-grid surface")
    ap.add_argument("--cores", type=int, default=8)
    args = ap.parse_args()
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={args.cores}"
    os.environ["OMP_NUM_THREADS"] = "1"  # SIMSOPT forks one process per core
    import netCDF4

    with netCDF4.Dataset(args.wout) as ds:
        ns = int(ds["ns"][:])
    j = int(round(args.s * (ns - 1) - 0.5))
    s = (j + 0.5) / (ns - 1)

    births, t_vmex, run_v, meta_v = run_vmex(args.wout, args.particles, args.tmax, s)
    from vmex.core.tracing import boozer_field
    from vmex.core.wout import read_wout

    field, bx = boozer_field(read_wout(args.wout))
    theta_v, phi_v, rz_err = vmec_angles(args.wout, bx, j, s, births[:, 1], births[:, 2])
    import jax

    B_vmex = np.asarray(jax.vmap(field.modB)(*(births[:, :3].T)))
    t_simple, run_s, meta_s = run_simple(args.simple, args.wout, births, theta_v, phi_v, args.tmax, args.cores)
    b_err = float(np.abs(meta_s.pop("B_start_T") / B_vmex - 1).max())
    t_sims, run_o, meta_o = run_simsopt(args.wout, births, args.tmax, args.cores)

    times = np.linspace(0, args.tmax, N_TIMES)
    codes = {}
    for name, t_loss, run, meta in (
        ("VMEX", t_vmex, run_v, meta_v),
        ("SIMPLE", t_simple, run_s, meta_s),
        ("SIMSOPT", t_sims, run_o, meta_o),
    ):
        f = float(np.isfinite(t_loss).mean())
        codes[name] = dict(
            meta,
            lost=int(np.isfinite(t_loss).sum()),
            loss_fraction=f,
            sigma=float(np.sqrt(f * (1 - f) / len(t_loss))),
            run_s=float(run),
            curve=[float((t_loss <= t).mean()) for t in times],
        )
        print(f"{name:8s} {100 * f:6.2f}% ± {100 * codes[name]['sigma']:.2f}%  run {run:8.2f} s")
    from importlib.metadata import version

    record = dict(
        schema="vmex.trace-cross-code/1",
        wout=Path(args.wout).name,
        s=s,
        particles=args.particles,
        tmax=args.tmax,
        cores=args.cores,
        times=times.tolist(),
        codes=codes,
        start_mapping=dict(max_RZ_mismatch_m=rz_err, max_modB_mismatch_rel=b_err),
        host=dict(
            machine=platform.machine(),
            processor=platform.processor() or None,
            system=platform.system(),
            date=time.strftime("%Y-%m-%d"),
        ),
        versions={k: version(k) for k in ("vmex", "essos", "jax", "simsopt", "booz_xform", "booz_xform_jax")},
    )
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {RECORD.relative_to(REPO)}", file=sys.stderr)


if __name__ == "__main__":
    main()
