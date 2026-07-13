#!/usr/bin/env python
"""Free-boundary tokamak beta scan through direct coils and generated mgrid.

The explicit source has 128 circular toroidal-field coils and 23 circular
poloidal-field loops reconstructed from the bundled DIII-D mgrid field. Each
equilibrium is solved twice from the same predictor: direct JAX Biot-Savart and
a VMEC2000-compatible mgrid generated from those exact coils. No boundary is
prescribed after the free-boundary release.
"""

import csv
import dataclasses
import json
import os
from pathlib import Path

import numpy as np

import vmec_jax as vj
from vmec_jax.core.coils import to_mgrid_data
from vmec_jax.core.plotting import surface_rz

# --------------------------- parameters ------------------------------------
DATA = Path(__file__).resolve().parent / "data"
INPUT_FILE = DATA / "input.DIII-D_lasym_false"
REFERENCE_MGRID = DATA / "mgrid_d3d_ef.nc"  # predictor only
PF_FILE = DATA / "tokamak_reconstructed_pf_coils.json"
OUT_DIR = Path("output_free_boundary_tokamak_coils")
TARGET_BETAS = [0.0, 1.5, 3.0]  # actual volume-average beta [%]
N_TF_COILS, TF_RADIUS, TF_CURRENT = 128, 1.8, -116307.8458
NS, MPOL, NITER, FTOL = 16, 8, 10000, 1e-8
MGRID_SHAPE = (145, 225, 1)  # R, Z, phi; axisymmetric single raw group
CI = os.environ.get("VMEC_JAX_EXAMPLES_CI") == "1"
if CI:
    TARGET_BETAS = [1.5]
    NS, MPOL, NITER = 12, 6, 4000
    MGRID_SHAPE = (65, 97, 1)

# --------------------------- coils and mgrid -------------------------------
pf_coils = np.asarray(json.loads(PF_FILE.read_text())["pf_coils"], dtype=float)
coils = vj.tokamak_coils(
    major_radius=1.566429,
    tf_coil_radius=TF_RADIUS,
    tf_current=TF_CURRENT,
    pf_coils=pf_coils,
    n_tf_coils=N_TF_COILS,
    n_segments=128,
    chunk_size=256,
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
mgrid_path = OUT_DIR / "mgrid_tokamak_from_explicit_coils.nc"
ir, jz, kp = MGRID_SHAPE
grid = to_mgrid_data(
    coils, 0.7, 2.9, -1.75, 1.75, ir=ir, jz=jz, kp=kp,
    nfp=1, mgrid_mode="N", single_group=True,
)
vj.write_mgrid(mgrid_path, grid)

# --------------------------- shared predictor -------------------------------
inp = vj.VmecInput.from_file(INPUT_FILE)
base = dataclasses.replace(
    inp,
    mpol=MPOL,
    rbc=inp.rbc[:, :MPOL], zbs=inp.zbs[:, :MPOL],
    rbs=inp.rbs[:, :MPOL], zbc=inp.zbc[:, :MPOL],
    ns_array=[NS], niter_array=[NITER], ftol_array=[FTOL],
)
predictor = vj.solve_free_boundary(
    base, mgrid_path=REFERENCE_MGRID, error_on_no_convergence=False)
if not predictor.converged:
    raise RuntimeError("bundled DIII-D mgrid predictor did not converge")

# --------------------------- calibrated beta scan ---------------------------
rows, solutions = [], []
for target in TARGET_BETAS:
    pressure_scale = 0.0 if target == 0.0 else target / 1.5
    state = predictor.state
    for _attempt in range(3):
        direct_inp = dataclasses.replace(
            base, mgrid_file="explicit_tokamak_coils", extcur=[],
            pres_scale=pressure_scale,
        )
        direct = vj.solve_free_boundary(
            direct_inp, external_field=coils, initial_state=state,
            error_on_no_convergence=False,
        )
        if not direct.converged:
            raise RuntimeError(f"direct-coil solve failed at target beta {target:.2f}%")
        state = direct.state
        wd = vj.wout_from_state(
            inp=direct_inp, state=state, fsqr=float(direct.fsqr),
            fsqz=float(direct.fsqz), fsql=float(direct.fsql),
            niter=int(direct.iterations), converged=True,
            vacuum_state=direct.vacuum_state,
        )
        actual = 100.0 * float(wd.betatotal)
        if target == 0.0 or abs(actual - target) <= 0.03:
            break
        pressure_scale *= target / actual

    mgrid_inp = dataclasses.replace(
        direct_inp, mgrid_file=str(mgrid_path), extcur=[1.0])
    sampled = vj.solve_free_boundary(
        mgrid_inp, mgrid_path=mgrid_path, initial_state=predictor.state,
        error_on_no_convergence=False,
    )
    if not sampled.converged:
        raise RuntimeError(f"generated-mgrid solve failed at target beta {target:.2f}%")
    wm = vj.wout_from_state(
        inp=mgrid_inp, state=sampled.state, fsqr=float(sampled.fsqr),
        fsqz=float(sampled.fsqz), fsql=float(sampled.fsql),
        niter=int(sampled.iterations), converged=True,
        vacuum_state=sampled.vacuum_state,
    )
    edge_scale = np.linalg.norm(np.r_[direct.state.R_cos[-1], direct.state.Z_sin[-1]])
    edge_delta = np.linalg.norm(np.r_[
        sampled.state.R_cos[-1] - direct.state.R_cos[-1],
        sampled.state.Z_sin[-1] - direct.state.Z_sin[-1],
    ]) / edge_scale
    rows.append((target, actual, pressure_scale, direct.iterations,
                 sampled.iterations, edge_delta, wd.aspect, wm.aspect))
    solutions.append((wd, wm))
    label = str(target).replace(".", "p")
    vj.write_wout(OUT_DIR / f"wout_direct_beta_{label}.nc", wd)
    vj.write_wout(OUT_DIR / f"wout_mgrid_beta_{label}.nc", wm)

print(f"\n{'target':>8s} {'actual':>8s} {'PRES':>9s} {'direct':>8s} "
      f"{'mgrid':>8s} {'edge rel':>10s}")
for row in rows:
    print(f"{row[0]:7.2f}% {row[1]:7.3f}% {row[2]:9.5f} {row[3]:8d} "
          f"{row[4]:8d} {row[5]:10.3e}")

with (OUT_DIR / "tokamak_beta_parity.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(("target_beta_percent", "actual_beta_percent", "pres_scale",
                     "direct_iterations", "mgrid_iterations", "edge_relative_delta",
                     "direct_aspect", "mgrid_aspect"))
    writer.writerows(rows)

# --------------------------- reviewed figures -------------------------------
if not CI:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    fig, axes = plt.subplots(1, len(rows), figsize=(3.2 * len(rows), 3.7), dpi=130)
    axes = np.atleast_1d(axes)
    for ax, row, (wd, wm) in zip(axes, rows, solutions):
        rd, zd = surface_rz(wd, s_index=-1, theta=theta, phi=np.array([0.0]))
        rm, zm = surface_rz(wm, s_index=-1, theta=theta, phi=np.array([0.0]))
        ax.plot(rd[:, 0], zd[:, 0], lw=2.0, label="direct coils")
        ax.plot(rm[:, 0], zm[:, 0], "--", lw=1.7, label="generated mgrid")
        ax.set(title=f"actual beta {row[1]:.2f}%", xlabel="R [m]", ylabel="Z [m]")
        ax.set_aspect("equal"); ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Solved tokamak LCFS: direct Biot-Savart vs generated mgrid")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tokamak_beta_parity.png", facecolor="white")
    for target, (wd, _wm) in zip(TARGET_BETAS, solutions):
        vj.plot_wout(wd, OUT_DIR / f"direct_beta_{str(target).replace('.', 'p')}")
