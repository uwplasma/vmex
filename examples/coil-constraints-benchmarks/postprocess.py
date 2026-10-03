#!/usr/bin/env python
"""Post-process a finished run of either coil-constraint benchmark.

    python postprocess.py runs/free                   # free-boundary arm
    python postprocess.py runs/fixed --match-flux     # fixed-boundary arm

Reads what both optimization scripts write: ``metrics.jsonl``, the initial
coils, the ``coils.stepN.json`` / ``wout.stepN.nc`` checkpoints and the final
``coils.json`` / ``wout.nc``. Writes into ``<run>/postprocess``:

- ``history.csv``, ``loss.png`` (QA and objective) and ``constraints.png``
  (every constrained quantity against its limits, from ``parameters.py``);
- ``evolution.gif``: coils and LCFS at every checkpoint, coloured by |B| of
  the coil field, and ``initial_final.png``;
- with ``--poincare N``, ``poincare.png`` / ``poincare.json``: field lines of
  the final coils traced for N toroidal transits from 5 points on each of the
  flux surfaces ``POINCARE_SURFACES`` of the run's own WOUT (free boundary for
  the free arm, the prescribed boundary for the fixed arm), with each line's
  distance from its surface; islands and stochastic layers show up here even
  though the nested-surface equilibrium cannot represent them;
- ``wout_dense.nc``, ``dense.json`` and the ``vmex.plot_wout`` figures: the
  final (or latest checkpoint) coils solved free-boundary at ``--ns`` (default NS201), the like-for-
  like QA of both arms. A fixed-arm run need not enclose PHIEDGE; with
  ``--match-flux`` its currents are rescaled by the logged coil-flux ratio,
  which in vacuum changes the field strength only.
- with ``--trace``, ``trace/``: ``vmex --trace`` alpha losses of that dense
  equilibrium, scaled in memory to ARIES-CS size (3.52 MeV alphas; 10 ms
  prompt losses by default, ``--trace-args "--collisional --trace-tmax 0.2"``
  for slowing-down losses).
"""

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import parameters as P  # noqa: E402

POINCARE_SURFACES = (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
POINCARE_POINTS_PER_SURFACE = 5


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="run directory written by either benchmark")
    parser.add_argument("--ns", type=int, default=201, help="radial resolution of the dense free-boundary solve")
    parser.add_argument("--no-dense", action="store_true", help="skip the dense free-boundary solve")
    parser.add_argument("--match-flux", action="store_true", help="rescale fixed-arm currents to enclose PHIEDGE")
    parser.add_argument("--seed-wout", type=Path, help="initial state of the dense solve (default: the run's wout.nc)")
    parser.add_argument("--max-iterations", type=int, default=12000, help="iteration cap of the dense solve")
    parser.add_argument("--flux-tolerance", type=float, help="flux band the fixed-arm run used, for the plot")
    parser.add_argument("--poincare", type=int, default=0, metavar="N",
                        help="trace the final coils' field lines for N toroidal transits (0: skip)")
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--trace", action="store_true", help="vmex --trace alpha losses of the dense equilibrium")
    parser.add_argument("--trace-args", default="", help='extra vmex --trace flags, e.g. "--collisional"')
    return parser.parse_args(argv)


def read_history(run):
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    keys = [k for k in rows[0] if all(isinstance(r.get(k), (int, float)) for r in rows)]
    return rows, keys


def limits(flux_tolerance=None):
    """Constrained quantities and their bounds, as the two benchmarks impose them."""
    import single_stage_optimization as fixed
    width = P.RADIUS_TOLERANCE - P.RADIUS_MARGIN
    band = (None, None) if flux_tolerance is None else (1 - flux_tolerance, 1 + flux_tolerance)
    return {
        "min_abs_iota": ("min |iota|", P.IOTA_FLOOR, None),
        "aspect": ("aspect ratio", *P.ASPECT_RANGE),
        "mirror_ratio": ("mirror ratio", None, P.MIRROR_LIMIT),
        "major_radius_m": ("major radius [m]", P.RADIUS_TARGET - width, P.RADIUS_TARGET + width),
        "coil_surface_distance_m": ("coil-plasma distance [m]", P.COIL_SURFACE_DISTANCE_LIMIT, None),
        "coil_minimum_scaled_slack": ("min coil slack", 0.0, None),
        "normal_field_rms": ("rms B.n/|B|", None, fixed.NORMAL_FIELD_CONSTRAINT),
        "flux_ratio": ("coil flux / PHIEDGE", *band),
        "current_factor": ("coil current factor", None, None),
    }


def plot_history(rows, keys, out, flux_tolerance=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (out / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    step = [r["step"] for r in rows]
    figure, axis = plt.subplots(figsize=(7, 4))
    for key, label in (("qa", P.TARGET_NAME), ("objective", "objective")):
        if key in keys:
            axis.semilogy(step, [max(r[key], 1e-16) for r in rows], label=label)
    axis.set_xlabel("accepted step"), axis.grid(True, alpha=0.3), axis.legend()
    figure.tight_layout(), figure.savefig(out / "loss.png", dpi=200), plt.close(figure)

    panels = [(k, *spec) for k, spec in limits(flux_tolerance).items() if k in keys]
    columns = 3
    figure, axes = plt.subplots((len(panels) + columns - 1) // columns, columns,
                                figsize=(4 * columns, 3 * ((len(panels) + columns - 1) // columns)), squeeze=False)
    for axis, (key, label, lower, upper) in zip(axes.flat, panels):
        axis.plot(step, [r[key] for r in rows], lw=1)
        for bound in (lower, upper):
            if bound is not None:
                axis.axhline(bound, color="r", ls="--", lw=0.8)
        axis.set_title(label, fontsize=9), axis.set_xlabel("accepted step", fontsize=8), axis.grid(True, alpha=0.3)
    for axis in list(axes.flat)[len(panels):]:
        axis.set_visible(False)
    figure.tight_layout(), figure.savefig(out / "constraints.png", dpi=200), plt.close(figure)


def checkpoints(run):
    """(label, coils, wout) for every saved step, then the final state."""
    steps = sorted(int(m.group(1)) for p in run.glob("coils.step*.json")
                   if (m := re.fullmatch(r"coils\.step(\d+)\.json", p.name)) and (run / f"wout.step{m.group(1)}.nc").exists())
    frames = [(f"step {s}", run / f"coils.step{s}.json", run / f"wout.step{s}.nc") for s in steps]
    if (run / "coils.json").exists() and (run / "wout.nc").exists():
        frames.append(("final", run / "coils.json", run / "wout.nc"))
    return frames


def plot_evolution(frames, out, final_step=None):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import vmex as vj
    from essos.coils import Coils
    from essos.fields import BiotSavart
    from essos.surfaces import SurfaceRZFourier

    objects = [(SurfaceRZFourier.from_wout_file(str(w), nphi=64, ntheta=32), Coils.from_json(str(c)))
               for _, c, w in frames]

    def modB(x, pair):
        surface, coils = pair
        points = jnp.asarray(surface.gamma).reshape(-1, 3)
        return jnp.linalg.norm(jax.vmap(BiotSavart(coils).B)(points), axis=-1).reshape(surface.gamma.shape[:-1])

    # Title each frame with the optimizer step it was saved at, not its frame index.
    labels = [f"step {final_step}" if label == "final" and final_step is not None else label for label, _, _ in frames]
    vj.plot_optimization_movie(out / "evolution.gif", [np.array([i]) for i in range(len(objects))],
                               lambda x: objects[int(x[0])], color_factory=modB, color_label="|B| [T]",
                               frame_labels=labels)
    vj.plot_optimization_objects(out / "initial_final.png", (labels[0], *objects[0]), (labels[-1], *objects[-1]))


def poincare(frame, transits, out):
    """Poincare section at phi = 0 of the coils' field, seeded on the run's own flux surfaces."""
    from fractions import Fraction

    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import vmex
    from essos.coils import Coils
    from essos.dynamics import Tracing
    from essos.fields import BiotSavart

    label, coil_path, wout_path = frame
    wout = vmex.read_wout(wout_path)
    ns, nfp, xm = int(wout.ns), int(wout.nfp), np.asarray(wout.xm)
    s_full = np.linspace(0.0, 1.0, ns)
    rmnc = np.stack([np.interp(POINCARE_SURFACES, s_full, c) for c in np.asarray(wout.rmnc).T], axis=1)
    zmns = np.stack([np.interp(POINCARE_SURFACES, s_full, c) for c in np.asarray(wout.zmns).T], axis=1)
    theta = np.linspace(0.0, 2.0 * np.pi, 720)
    curves = [(np.cos(np.outer(theta, xm)) @ r, np.sin(np.outer(theta, xm)) @ z) for r, z in zip(rmnc, zmns)]
    seeds = 2.0 * np.pi * (np.arange(POINCARE_POINTS_PER_SURFACE) + 0.5) / POINCARE_POINTS_PER_SURFACE
    points = np.array([(np.cos(xm * t) @ r, 0.0, np.sin(xm * t) @ z)
                       for r, z in zip(rmnc, zmns) for t in seeds])
    field = BiotSavart(Coils.from_json(str(coil_path)))
    r_axis = float(np.asarray(wout.rmnc)[0].sum())
    b_axis = float(jnp.linalg.norm(field.B(jnp.array([r_axis, 0.0, 0.0]))))
    started = time.monotonic()
    # dx/dt = B: a transit is about 2 pi R0 / |B| long, and field lines are longer than 2 pi R0.
    tracing = Tracing(field=field, model="FieldLineAdaptative", initial_conditions=jnp.asarray(points),
                      maxtime=1.6 * transits * 2.0 * np.pi * r_axis / b_axis, times_to_trace=transits * 96,
                      atol=1e-10, rtol=1e-10)
    trajectories = np.asarray(tracing.trajectories)
    seconds = time.monotonic() - started

    iota = np.abs(np.asarray(wout.iotas)[1:])
    s_half = 0.5 * (s_full[1:] + s_full[:-1])
    rationals = sorted({Fraction(nfp * k, m) for m in range(1, 9) for k in range(1, 4)
                        if iota.min() <= nfp * k / m <= iota.max()})
    resonances = {str(q): np.round(s_half[np.flatnonzero(np.diff(np.sign(iota - float(q))))], 3).tolist()
                  for q in rationals}
    lcfs = np.stack(curves[-1], axis=1) if POINCARE_SURFACES[-1] == 1.0 else None
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(POINCARE_SURFACES)))
    figure, axis = plt.subplots(figsize=(7, 7))
    lines, sections = [], {}
    for k, trajectory in enumerate(trajectories):
        j = k // POINCARE_POINTS_PER_SURFACE
        x, y, z = trajectory[:, :3].T
        phase = np.unwrap(np.arctan2(y, x))
        cross = np.flatnonzero(np.diff(np.floor(phase / (2.0 * np.pi))) != 0)
        f = (2.0 * np.pi * np.round(phase[cross + 1] / (2.0 * np.pi)) - phase[cross]) / np.diff(phase)[cross]
        R = np.hypot(x, y)[cross] + f * np.diff(np.hypot(x, y))[cross]
        Z = z[cross] + f * np.diff(z)[cross]
        axis.scatter(R, Z, s=0.4, color=colors[j], zorder=2)
        sections[f"line{k}_R"], sections[f"line{k}_Z"] = R, Z
        surface = np.stack(curves[j], axis=1)
        d = np.min(np.hypot(R[:, None] - surface[None, :, 0], Z[:, None] - surface[None, :, 1]), axis=1) if len(R) else np.array([np.nan])
        outside = bool(lcfs is not None and len(R) and (R.max() > lcfs[:, 0].max() + 0.01 or R.min() < lcfs[:, 0].min() - 0.01
                                                        or np.abs(Z).max() > np.abs(lcfs[:, 1]).max() + 0.01))
        lines.append(dict(s=POINCARE_SURFACES[j], theta=float(seeds[k % POINCARE_POINTS_PER_SURFACE]), crossings=int(len(R)),
                          rms_distance_m=float(np.sqrt(np.nanmean(d**2))), max_distance_m=float(np.nanmax(d)),
                          leaves_lcfs=outside))
    # The run's own flux surfaces on top of the crossings: dashed, the LCFS solid.
    for j, (R, Z) in enumerate(curves):
        edge = POINCARE_SURFACES[j] == 1.0
        axis.plot(R, Z, color="black", lw=1.0 if edge else 0.8, ls="-" if edge else "--", zorder=4)
    axis.scatter([], [], s=12, color=colors[len(colors) // 2], label="field-line crossings, coloured by seed surface")
    axis.plot([], [], color="black", ls="--", lw=0.8,
              label="WOUT flux surfaces s = " + ", ".join(f"{s:g}" for s in POINCARE_SURFACES[:-1]))
    axis.plot([], [], color="black", lw=1.0, label="WOUT LCFS")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), fontsize=7, frameon=False)
    axis.set_title(f"{P.CASE} {label}: coil-field Poincaré section, φ = 0, ≥{transits} transits")
    axis.set_xlabel("R [m]"), axis.set_ylabel("Z [m]"), axis.set_aspect("equal")
    figure.tight_layout(), figure.savefig(out / "poincare.png", dpi=200), plt.close(figure)
    per_surface = {str(s): dict(max_distance_m=max(line["max_distance_m"] for line in lines if line["s"] == s),
                                leaving_lcfs=sum(line["leaves_lcfs"] for line in lines if line["s"] == s))
                   for s in POINCARE_SURFACES}
    report = dict(frame=label, coils=str(coil_path), wout=str(wout_path), transits=transits, seconds=seconds,
                  iota_axis=float(iota[0]), iota_edge=float(iota[-1]), resonances=resonances,
                  surfaces=per_surface, lines=lines)
    (out / "poincare.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(out / "poincare.npz", surfaces=np.asarray(POINCARE_SURFACES), **sections)
    return dict(frame=label, transits=transits, seconds=round(seconds), resonances=resonances, surfaces=per_surface)


def dense_solve(frame, args, rows, out):
    import numpy as np
    import vmex
    from vmex import optimize as opt
    from vmex.core.freeboundary import _solve_free_boundary_stage, free_boundary_resolution
    from vmex.core.solver import prepare_runtime
    from vmex.core.statephysics import major_radius
    from vmex.core.wout import wout_from_state
    from essos.coils import Coils
    from essos.fields import BiotSavart
    from free_boundary_single_stage_optimization import resize_coils, seed_input, target_residual

    inp = replace(seed_input(vmex), lfreeb=True, mgrid_file="direct ESSOS field", ns_array=np.array([args.ns]),
                  ftol_array=np.array([P.EQUILIBRIUM_FTOL]), niter_array=np.array([args.max_iterations]))
    label, coil_path, wout_path = frame
    coils = resize_coils(Coils.from_json(str(coil_path)), P.COIL_ORDER, P.N_SEGMENTS)
    row = rows[-1] if label == "final" else next(r for r in rows if f"step {r['step']}" == label)
    scale = 1.0 / row["flux_ratio"] if args.match_flux else 1.0
    if scale != 1.0:
        coils = Coils(coils.curves, coils.dofs_currents_raw * scale, currents_scale=coils.currents_scale)
    field = BiotSavart(coils)
    seed = args.seed_wout or wout_path
    state = vmex.state_from_wout(vmex.read_wout(seed), inp=inp, ns=args.ns)
    resolution = free_boundary_resolution(inp, field, ns=args.ns)
    started = time.monotonic()
    stage = _solve_free_boundary_stage(
        inp, external_field=field, resolution=resolution, initial_state=state, ftol=P.EQUILIBRIUM_FTOL,
        max_iterations=args.max_iterations, include_edge_in_convergence=True, edge_force_tolerance=P.EQUILIBRIUM_FTOL,
        use_fft=False, error_on_no_convergence=False, jacobian_retries=0, allow_initial_axis_reguess=False)
    result = stage.result
    forces = {k: float(getattr(result, k)) for k in ("fsqr", "fsqz", "fsql", "fedge")}
    converged = bool(result.converged and result.vacuum is not None
                     and all(np.isfinite(v) and 0 <= v <= P.EQUILIBRIUM_FTOL for v in forces.values()))
    report = dict(frame=label, coils=str(coil_path), ns=args.ns, seed=str(seed), current_scale=scale, currents_A=np.asarray(coils.currents).tolist(),
                  converged=converged, iterations=int(result.iterations), seconds=time.monotonic() - started, **forces)
    if converged:
        wout = wout_from_state(inp=inp, state=result.state, niter=int(result.iterations), converged=True,
                               vacuum_output=result.vacuum, **{k: forces[k] for k in ("fsqr", "fsqz", "fsql")})
        path = vmex.write_wout(str(out / "wout_dense.nc"), wout)
        rt = prepare_runtime(inp, resolution)
        residuals = target_residual().residuals_state(result.state, rt)
        report.update(qa=float(np.vdot(residuals, residuals)), min_abs_iota=float(opt.min_abs_iota(result.state, rt)),
                      aspect=float(opt.aspect_ratio(result.state, rt)), major_radius_m=float(major_radius(result.state, rt)))
        report["figures"] = [str(p) for p in vmex.plot_wout(path, out, name="dense").values()]
    else:
        print("dense solve did not converge; seed it closer with --seed-wout", flush=True)
    (out / "dense.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def trace(wout, out, extra):
    """``vmex <wout> --trace`` in a fresh process, so it can give JAX one CPU device per core."""
    command = [sys.executable, "-m", "vmex", str(wout), "--trace", "--outdir", str(out / "trace"), *shlex.split(extra)]
    subprocess.run(command, check=True, env=dict(os.environ, JAX_PLATFORMS="cpu"))


def main(argv=None):
    args = parse_args(argv)
    os.environ["JAX_ENABLE_X64"] = "1"
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu" if args.device == "gpu" else "cpu")
    run = args.run.resolve()
    out = run / "postprocess"
    out.mkdir(exist_ok=True)
    rows, keys = read_history(run)
    if args.match_flux and "flux_ratio" not in keys:
        raise SystemExit("--match-flux needs a fixed-arm run that logs flux_ratio")
    plot_history(rows, keys, out, args.flux_tolerance)
    frames = checkpoints(run)
    if frames:
        plot_evolution(frames, out, final_step=rows[-1]["step"])
    if args.poincare and frames:
        print(json.dumps(poincare(frames[-1], args.poincare, out)), flush=True)
    if not args.no_dense and frames:  # the final state, or the latest checkpoint of a running job
        report = dense_solve(frames[-1], args, rows, out)
        print(json.dumps(report), flush=True)
        if args.trace and report["converged"]:
            trace(out / "wout_dense.nc", out, args.trace_args)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
