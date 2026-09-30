#!/usr/bin/env python3
"""Audit Boozer mode cuts across WOUTs, with optional common-birth alpha traces.

Usage: python benchmarks/trace_mode_cut.py PATH [PATH ...] --out cuts.json
PATH may be a WOUT or a directory (searched recursively). Boozer transforms
all half-grid surfaces; field errors use s near 0.25, 0.5 and 0.9.
``--particles 128`` adds common-birth orbit comparisons; use more particles
and a longer horizon for loss-fraction conclusions.
"""

import argparse
import hashlib
import json
import platform
import time
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

CUTS = (0.0, 1e-5, 6e-5, 8e-5, 1e-4, 2e-4, 3e-4, 5e-4, 6e-4, 8e-4, 1e-3)
TRANSFORM_MODES = 32


def installed_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def spectrum(bx):
    bm = np.asarray(bx.bmnc_b)
    m, n = np.asarray(bx.xm_b), np.asarray(bx.xn_b)
    amplitude = np.abs(bm).max(axis=1)
    base = amplitude.max()
    r = np.sqrt(np.asarray(bx.s_b))
    # Match the ESSOS spline in r=sqrt(s); report Boozer-angle derivatives.
    spline = CubicSpline(r, np.where(m[:, None] > 0, bm / r, bm).T, axis=0)
    weight = np.where((m == 0) & (n == 0), 1.0, 0.5)
    rows = []
    for s in (0.25, 0.5, 0.9):
        radius = np.sqrt(np.clip(s, r[0] ** 2, r[-1] ** 2))
        a, da = spline(radius), spline(radius, 1)
        coeff = np.where(m > 0, radius * a, a)
        radial = np.where(m > 0, a + radius * da, da)
        angular = (m * m + n * n) * coeff**2
        norm = [max(float(np.sqrt(np.sum(weight * values))), 1e-30)
                for values in (coeff**2, radial**2, angular)]
        row = {"s": float(radius**2), "cuts": {}}
        for cut in CUTS:
            keep = amplitude > cut * base
            omit = ~keep
            row["cuts"][str(cut)] = {
                "modes": int(keep.sum()),
                "B_rms_rel": float(np.sqrt(np.sum(weight[omit] * coeff[omit] ** 2)) / norm[0]),
                "radial_derivative_rms_rel": float(np.sqrt(np.sum(weight[omit] * radial[omit] ** 2)) / norm[1]),
                "angle_gradient_rms_rel": float(np.sqrt(np.sum(weight[omit] * angular[omit])) / norm[2]),
            }
        rows.append(row)
    return rows


def scaled_field(bx, wout, b, r, cut):
    from essos.boozer import BoozerField

    # Ideal-MHD scaling leaves Boozer angles and iota unchanged; transform
    # the tables directly so older VMEC WOUTs need no rewriting/re-transform.
    return BoozerField.from_booz(
        bx.s_b, b * np.asarray(bx.bmnc_b), bx.xm_b, bx.xn_b, bx.iota,
        b * r * np.asarray(bx.Boozer_G), b * r * np.asarray(bx.Boozer_I),
        -float(np.asarray(wout.phi)[-1]) * b * r**2 / (2 * np.pi), int(bx.nfp), cut)


def orbits(path, bx, cuts, particles, tmax, repeats, save_times, birth_cut, step_factor):
    import jax
    from essos import constants as c
    from essos.boozer import trace_boozer
    from vmex.core.scaling import aries_cs_scales
    from vmex.core.tracing import TIMESTEP, sample_births
    from vmex.core.wout import read_wout

    wout = read_wout(path)
    b, r = aries_cs_scales(wout)
    fields = {cut: scaled_field(bx, wout, b, r, cut) for cut in cuts}
    birth_field = fields[birth_cut] if birth_cut in fields else scaled_field(bx, wout, b, r, birth_cut)
    births = sample_births(birth_field, particles, s=0.3, seed=42)
    speed = float(np.sqrt(2 * c.FUSION_ALPHA_PARTICLE_ENERGY / c.ALPHA_PARTICLE_MASS))
    dt = TIMESTEP * float(wout.Aminor_p) * r * step_factor / 1.7044
    results = {}
    labels = {}
    for cut, field in fields.items():
        kw = dict(speed=speed, mass=c.ALPHA_PARTICLE_MASS, charge=c.ALPHA_PARTICLE_CHARGE,
                  tmax=tmax, timestep=dt, n_save=save_times)
        start = time.perf_counter()
        out = trace_boozer(field, *births.T, **kw)
        cold_s = time.perf_counter() - start
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            out = trace_boozer(field, *births.T, **kw)
            timings.append(time.perf_counter() - start)
        failed = (np.asarray(getattr(out, "failed", np.zeros(particles, dtype=bool))) |
                  ~np.isfinite(out.states).all(axis=(1, 2)) |
                  ~np.isfinite(out.energy_error))
        if failed.any():
            raise ValueError(f"{failed.sum()} alpha trajectories failed at cut {cut:g}; reduce timestep")
        lost = out.loss_times >= 0
        labels[cut] = lost
        results[str(cut)] = {"modes": int(field.xm.size), "lost": int(lost.sum()),
                             "lost_indices": np.flatnonzero(lost).tolist(),
                             "loss_times": np.asarray(out.loss_times).tolist(),
                             "fraction": float(lost.mean()), "failed": 0,
                             "max_energy_error": float(np.max(out.energy_error)),
                             "trace_s": float(np.median(timings)), "cold_s": cold_s,
                             "timings_s": timings}
        print(f"  cut {cut:g}: {field.xm.size} modes, {lost.sum()} lost, "
              f"{np.median(timings):.2f} s warm", flush=True)
    reference = labels[min(cuts)]
    for cut in cuts:
        results[str(cut)]["label_agreement"] = float(np.mean(labels[cut] == reference))
        results[str(cut)]["baseline_loss_recall"] = (
            float(np.mean(labels[cut][reference])) if reference.any() else None)
    return {"particles": particles, "tmax": tmax, "timestep": dt, "save_times": save_times,
            "reference_cut": float(min(cuts)),
            "birth_sha256": hashlib.sha256(births.astype("<f8", copy=False).tobytes()).hexdigest(),
            "birth_cut": birth_cut, "step_factor": step_factor,
            "devices": len(jax.devices()), "cuts": results}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--particles", type=int, default=0)
    ap.add_argument("--tmax", type=float, default=1e-3)
    ap.add_argument("--repeats", type=int, default=3, help="timed calls after one compilation call")
    ap.add_argument("--save-times", type=int, default=101, help="orbit samples, including t=0; matches vmex --trace")
    ap.add_argument("--birth-cut", type=float, choices=CUTS[1:], default=1e-4,
                    help="field used to sample common births; default 1e-4")
    ap.add_argument("--orbit-cuts", nargs="+", type=float, choices=CUTS[1:], default=CUTS[2:])
    ap.add_argument("--step-factor", type=float, default=1.0, help="multiply the default RK4 step")
    ap.add_argument("--devices", type=int, help="CPU devices; default matches vmex --trace")
    args = ap.parse_args()
    if (args.repeats < 1 or args.save_times < 2 or args.particles < 0
            or not np.isfinite(args.tmax) or args.tmax <= 0
            or not np.isfinite(args.step_factor) or args.step_factor <= 0
            or (args.devices is not None and args.devices < 1)):
        ap.error("positive tmax and step-factor, repeats >= 1, save-times >= 2, particles >= 0 and devices >= 1 are required")
    if args.particles:
        import jax
        from vmex.core.cli import _split_host_devices

        if args.devices:
            jax.config.update("jax_num_cpu_devices", args.devices)
        else:
            _split_host_devices()
    from booz_xform_jax import Booz_xform

    paths = sorted({q for p in args.paths for q in (p.rglob("wout*.nc") if p.is_dir() else [p])})
    record = {"schema": "vmex.trace-mode-cut/1", "cuts": CUTS,
              "spectral_error": f"RMS relative to the uncut mboz=nboz={TRANSFORM_MODES} spectrum on the ESSOS r=sqrt(s) spline",
              "host": platform.uname()._asdict(),
              "versions": {name: installed_version(name) for name in
                           (("vmex", "jax", "booz_xform_jax") + (("essos",) if args.particles else ()))},
              "cases": []}
    for path in paths:
        row = {"path": str(path)}
        start = time.perf_counter()
        try:
            bx = Booz_xform(verbose=0, mboz=TRANSFORM_MODES, nboz=TRANSFORM_MODES)
            bx.read_wout(str(path), flux=False)
            if bool(bx.asym):
                raise ValueError("lasym: Boozer tracer is symmetric only")
            bx.run()
            row.update(nfp=int(bx.nfp), surfaces=int(np.asarray(bx.s_b).size),
                       spectrum=spectrum(bx))
            if args.particles:
                row["orbits"] = orbits(path, bx, args.orbit_cuts, args.particles, args.tmax,
                                       args.repeats, args.save_times, args.birth_cut, args.step_factor)
            print(f"{path.name}: {len(np.asarray(bx.xm_b))} modes, {time.perf_counter()-start:.1f} s", flush=True)
        except (ValueError, RuntimeError, KeyError, OSError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            if not row["error"].startswith("ValueError: lasym:"):
                row["traceback"] = traceback.format_exc(limit=6)
            print(f"{path.name}: {row['error']}", flush=True)
        record["cases"].append(row)
    target = json.dumps(record, indent=2) + "\n"
    if args.out:
        args.out.write_text(target)
    else:
        print(target)
    if any("traceback" in case for case in record["cases"]):
        raise SystemExit("one or more WOUT audits failed; inspect the JSON tracebacks")


if __name__ == "__main__":
    main()
