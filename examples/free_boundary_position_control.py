#!/usr/bin/env python
"""Hold a free-boundary tokamak in place with feedback on the vertical field.

A tokamak plasma needs a vertical field of the right size: too little and it
slides inboard, too much and it is pushed outboard, and nothing in the
free-boundary iteration restores it.  Real machines close that loop with a
feedback-driven vertical-field coil.  ``vj.PositionControl`` does the same for
VMEX: a small curl-free uniform ``B_Z`` is added to the external field and
follows a PID law on the error between the measured magnetic axis and a target
radius (see docs/explanation/position-control.md).

The field is a DIII-D-like axisymmetric coil field (a degree-eight Chebyshev
compression of the 21-coil-group DIII-D table, ``data/d3d_chebyshev_field.npz``)
with a small asymmetric ``B_R``.  A mis-set vertical field is a uniform ``B_Z``
offset added to it.  Each offset is solved without control (a zero-gain
control that only records the trajectory) and with control.  Without control
the axis moves by tens of centimetres and a deficit never converges; with
control the axis stays on target and ``result.position_control.vertical_field``
reports the field the coils were missing.

The script also writes the 40 mT deficit as an mgrid next to a copy of
``data/input.DIII-D_position_control``, so the deck runs from the command line:
``vmex output_position_control/input.DIII-D_position_control``.
"""

import dataclasses
import shutil
import time
from pathlib import Path

import numpy as np
from numpy.polynomial.chebyshev import chebval2d

import vmex as vj
from vmex.core.fourier import mode_table
from vmex.core.mgrid import tabulate_cartesian_field, write_mgrid
from vmex.core.position_control import measure_position

DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_FILE = DATA_DIR / "input.DIII-D_lasym_false"
FIELD_FILE = DATA_DIR / "d3d_chebyshev_field.npz"
DECK_FILE = DATA_DIR / "input.DIII-D_position_control"

# Uniform B_Z offsets added to the coil field [T]; negative = vertical field
# missing (plasma slides inboard), positive = excess (pushed outboard):
OFFSETS = [-0.04, -0.02, 0.0, 0.02, 0.04]

# Radial grid, iteration budget and tolerance of every solve, and the control
# update interval (the PID gains keep their defaults; the target is the
# nominal axis):
NS, NITER, FTOL = 16, 8000, 1e-10
INTERVAL = 20

# Output directory (deck, mgrid) and the two figures (32-color lossless WebP):
OUTPUT_DIR = Path("output_position_control")
FIGURE = Path("readme_position_control.webp")
DECK_FIGURE = Path("readme_position_control_deck.webp")

###############################################################################
# End of input parameters.
###############################################################################

### Coil field and deck ########################################################

coef = np.load(FIELD_FILE)
br_c, bp_c, bz_c = (np.zeros((9, 9)) for _ in range(3))
br_c[1::2], bp_c[0], bz_c[::2] = coef["br_odd"], coef["bp_even"], coef["bz_even"]


def field_table(dbz):
    """Tabulate the compressed DIII-D field plus a uniform B_Z offset [T]."""
    def b_xyz(points):
        x, y, z = np.moveaxis(np.asarray(points), -1, 0)
        radius, phi = np.hypot(x, y), np.arctan2(y, x)
        rn, zn = radius - 1.75, z / 1.6
        br = chebval2d(zn, rn, br_c) + 3.0e-4 / radius   # asymmetric forcing
        bp, bz = chebval2d(zn, rn, bp_c), chebval2d(zn, rn, bz_c) + dbz
        return np.stack((br * np.cos(phi) - bp * np.sin(phi),
                         br * np.sin(phi) + bp * np.cos(phi), bz), axis=-1)
    return tabulate_cartesian_field(b_xyz, rmin=0.75, rmax=2.75, zmin=-1.6, zmax=1.6,
                                    ir=64, jz=81, kp=1, nfp=1, label="d3d_offset")


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy(DECK_FILE, OUTPUT_DIR / DECK_FILE.name)
write_mgrid(OUTPUT_DIR / "mgrid_d3d_vertical_deficit.nc", field_table(-0.040))
print(f"Wrote {OUTPUT_DIR / DECK_FILE.name} and its mgrid (a 40 mT vertical-field deficit)")

inp = vj.VmecInput.from_file(INPUT_FILE)
rbs, zbc = np.asarray(inp.rbs).copy(), np.asarray(inp.zbc).copy()
rbs[0, 1], zbc[0, 1] = 1.0e-4, -1.0e-4
inp = dataclasses.replace(
    inp, lasym=True, rbs=rbs, zbc=zbc, extcur=np.ones(1), mgrid_file="d3d_offset(direct)",
    ns_array=np.asarray([NS]), ftol_array=np.asarray([FTOL]), niter_array=np.asarray([NITER]))
modes = mode_table(int(inp.mpol), int(inp.ntor))

### Solve every offset with and without control ################################


def solve(dbz, control):
    t0 = time.perf_counter()
    res = vj.solve_free_boundary(
        inp, external_field=vj.MgridField.from_mgrid_data(field_table(dbz), extcur=np.ones(1)),
        max_iterations=NITER, error_on_no_convergence=False, position_control=control)
    out = dict(dbz=dbz, conv=bool(res.converged), iters=int(res.iterations),
               fsq=float(res.fsqr) + float(res.fsqz) + float(res.fsql),
               axis=float(measure_position(res.state, modes, vj.PositionControl())[0]),
               bz=res.position_control.vertical_field, history=res.position_control.history)
    print(f"  {1e3 * dbz:+6.1f} mT, {'controlled  ' if control.gain else 'uncontrolled'}: "
          f"{'converged' if out['conv'] else 'NOT converged'} after {out['iters']} iterations, "
          f"axis R = {out['axis']:.4f} m, {time.perf_counter() - t0:.0f} s")
    return out


start = time.perf_counter()
off = vj.PositionControl(gain=0.0, integral_gain=0.0, derivative_gain=0.0, interval=INTERVAL)
target = solve(0.0, off)["axis"]
print(f"nominal coils: axis R = {target:.4f} m, the target of every controlled run")
rows = [(solve(dbz, off), solve(dbz, vj.PositionControl(target=target, interval=INTERVAL)))
        for dbz in OFFSETS]

### Report ####################################################################

print(f"\n{'B_Z offset':>10s} | {'uncontrolled':^30s} | {'controlled':^44s}")
print(f"{'[mT]':>10s} | {'conv':>4s} {'iters':>5s} {'fsq':>8s} {'dR[cm]':>8s} | "
      f"{'conv':>4s} {'iters':>5s} {'fsq':>8s} {'dR[cm]':>8s} {'B_Z ctrl [mT]':>13s}")
for free, held in rows:
    print(f"{1e3 * free['dbz']:10.1f} | {'yes' if free['conv'] else 'NO':>4s} {free['iters']:5d} "
          f"{free['fsq']:8.1e} {100 * (free['axis'] - target):8.1f} | "
          f"{'yes' if held['conv'] else 'NO':>4s} {held['iters']:5d} {held['fsq']:8.1e} "
          f"{100 * (held['axis'] - target):8.2f} {1e3 * held['bz']:13.2f}")
worst = max(abs(h["axis"] - target) for _, h in rows)
recovered = max(abs(h["bz"] + h["dbz"]) for _, h in rows)
print(f"\nuncontrolled: {sum(not f['conv'] for f, _ in rows)}/{len(rows)} fail to converge, "
      f"axis moves up to {100 * max(abs(f['axis'] - target) for f, _ in rows):.0f} cm")
print(f"controlled:   {sum(h['conv'] for _, h in rows)}/{len(rows)} converge, axis within "
      f"{1e3 * worst:.1f} mm of target, B_Z^ctrl matches the offset to {1e3 * recovered:.2f} mT")
print(f"total {time.perf_counter() - start:.0f} s")

### Plot ######################################################################

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

mt = np.array([1e3 * f["dbz"] for f, _ in rows])
fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.8), dpi=110)
for key, ax in (("axis", axs[0]), ("fsq", axs[1])):
    ax.plot(mt, [f[key] for f, _ in rows], "o-", color="#b5503c", label="no control")
    ax.plot(mt, [h[key] for _, h in rows], "s-", color="#2e6da4", label="position control")
axs[0].axhline(target, color="0.6", lw=0.8, ls="--")
axs[0].set(xlabel="vertical-field offset [mT]", ylabel="magnetic axis R [m]", title="Axis position")
axs[0].legend(frameon=False)
axs[1].set_yscale("log")
axs[1].axhline(FTOL, color="0.6", lw=0.8, ls="--")
axs[1].set(xlabel="vertical-field offset [mT]", ylabel="final force residual", title="Convergence")
axs[2].plot(mt, mt, "-", color="0.6", lw=0.8, label="offset")
axs[2].plot(mt, [-1e3 * h["bz"] for _, h in rows], "s", color="#2e6da4", label="$-B_Z^{ctrl}$")
axs[2].set(xlabel="vertical-field offset [mT]", ylabel="[mT]", title="Missing field recovered")
axs[2].legend(frameon=False)
for ax in axs:
    ax.grid(alpha=0.25, lw=0.5)
fig.suptitle("Free-boundary DIII-D-like tokamak with a mis-set vertical field")
fig.tight_layout()
fig.canvas.draw()  # 32-color lossless WebP keeps the docs copy small
Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB").quantize(32).save(
    FIGURE, lossless=True, method=6)
print(f"Wrote {FIGURE}")

# The 40 mT deficit (the deck's case) against iteration:
free, held = rows[OFFSETS.index(-0.04)]
hs, hh = free["history"], held["history"]
fig, axs = plt.subplots(1, 3, figsize=(10.0, 3.2), dpi=80)
axs[0].plot(hs[:, 0], hs[:, -1], color="#b5503c", label="no control")
axs[0].plot(hh[:, 0], hh[:, -1], color="#2e6da4", label="position control")
axs[0].axhline(target, color="0.6", lw=0.8, ls="--")
axs[0].set(xlabel="iteration", ylabel="magnetic axis R [m]", title="Axis position")
axs[0].legend(frameon=False, loc="center right")
axs[1].plot(hh[:, 0], 1e3 * hh[:, 2], color="#2e6da4")
axs[1].axhline(40.0, color="0.6", lw=0.8, ls="--")
axs[1].set(xlabel="iteration", ylabel="$B_Z^{ctrl}$ [mT]", title="Feedback vertical field")
axs[2].semilogy(hs[:, 0], hs[:, 1], color="#b5503c", label="no control")
axs[2].semilogy(hh[:, 0], hh[:, 1], color="#2e6da4", label="position control")
axs[2].set(xlabel="iteration", ylabel="force residual", title="Convergence")
for ax in axs:
    ax.grid(alpha=0.25, lw=0.5)
fig.suptitle("DIII-D-like tokamak with a 40 mT vertical-field deficit")
fig.tight_layout()
fig.canvas.draw()
Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB").quantize(32).save(
    DECK_FIGURE, lossless=True, method=6)
print(f"Wrote {DECK_FIGURE}")
