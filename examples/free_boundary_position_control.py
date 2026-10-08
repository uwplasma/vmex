#!/usr/bin/env python
"""Hold a free-boundary tokamak in place with feedback on the vertical field.

A tokamak plasma needs a vertical field of the right size: too little and it
slides inboard, too much and it is pushed outboard, and nothing in the
free-boundary iteration restores it.  Real machines close that loop with a
feedback-driven vertical-field coil.  ``vj.PositionControl`` does the same for
VMEX: a small curl-free uniform ``B_Z`` is added to the external field and
follows a PID law on the error between the measured magnetic axis and a target
radius.

The field is a DIII-D-like axisymmetric coil field (a degree-eight Chebyshev
compression of the 21-coil-group DIII-D table, stored in
``data/d3d_chebyshev_field.npz``) with a small asymmetric ``B_R`` so the case
is up-down asymmetric.  A *mis-set vertical field* is then modelled by adding
a uniform ``B_Z`` offset to the tabulated field.  Each offset is solved twice:

* without control the plasma lands wherever force balance puts it (the axis
  moves by tens of centimetres) and, for a deficit of vertical field, the
  iteration never converges;
* with control the axis is held at the intended radius, every solve converges
  to ``FTOL``, and ``result.position_control.vertical_field`` reports the
  vertical field the coils were missing (the "vertical-field coil current" a
  real machine would need).

The figure is written to the working directory as lossless WebP.
"""

import dataclasses
import os
import time
from pathlib import Path

import numpy as np

import vmex as vj
from vmex.core.fourier import mode_table
from vmex.core.mgrid import tabulate_cartesian_field
from vmex.core.position_control import measure_position

DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_FILE = DATA_DIR / "input.DIII-D_lasym_false"
FIELD_FILE = DATA_DIR / "d3d_chebyshev_field.npz"

# Uniform B_Z offsets added to the coil field [T]; negative = vertical field
# missing (plasma slides inboard), positive = excess (pushed outboard):
OFFSETS = [-0.04, -0.02, 0.0, 0.02, 0.04]

NS = 16
NITER, FTOL = 8000, 1e-10
CONTROL_GAINS = dict(interval=20)   # default PID gains, target = the nominal axis

FIGURE_PATH = Path("readme_position_control.webp")
DECK_FIGURE_PATH = Path("readme_position_control_deck.webp")
DECK_FILE = DATA_DIR / "input.DIII-D_position_control"   # the same -40 mT case as an input deck
MAKE_FIGURE = True

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs: the single
# vertical-field deficit that defeats the uncontrolled solve, and no figure:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    OFFSETS = [-0.04]
    MAKE_FIGURE = False

###############################################################################
# End of input parameters.
###############################################################################

### Set up the field and the equilibrium ######################################

from numpy.polynomial.chebyshev import chebval2d  # noqa: E402

_coef = np.load(FIELD_FILE)


def coil_field_xyz(points, dbz=0.0):
    """Cartesian B [T] of the compressed DIII-D table plus a uniform B_Z offset."""
    x, y, z = np.moveaxis(np.asarray(points), -1, 0)
    radius, phi = np.hypot(x, y), np.arctan2(y, x)
    rn, zn = radius - 1.75, z / 1.6
    br_c, bp_c, bz_c = (np.zeros((9, 9)) for _ in range(3))
    br_c[1::2], bp_c[0], bz_c[::2] = _coef["br_odd"], _coef["bp_even"], _coef["bz_even"]
    br = chebval2d(zn, rn, br_c) + 3.0e-4 / radius   # asymmetric forcing
    bp, bz = chebval2d(zn, rn, bp_c), chebval2d(zn, rn, bz_c) + dbz
    return np.stack((br * np.cos(phi) - bp * np.sin(phi),
                     br * np.sin(phi) + bp * np.cos(phi), bz), axis=-1)


def make_field(dbz):
    data = tabulate_cartesian_field(
        lambda p: coil_field_xyz(p, dbz), rmin=0.75, rmax=2.75, zmin=-1.6, zmax=1.6,
        ir=64, jz=81, kp=1, nfp=1, label="d3d_offset")
    return vj.MgridField.from_mgrid_data(data, extcur=np.ones(1))


inp = vj.VmecInput.from_file(INPUT_FILE)
rbs, zbc = np.asarray(inp.rbs).copy(), np.asarray(inp.zbc).copy()
rbs[0, 1], zbc[0, 1] = 1.0e-4, -1.0e-4
inp = dataclasses.replace(
    inp, lasym=True, rbs=rbs, zbc=zbc, extcur=np.ones(1), mgrid_file="d3d_offset(direct)",
    ns_array=np.asarray([NS]), ftol_array=np.asarray([FTOL]), niter_array=np.asarray([NITER]))
modes = mode_table(int(inp.mpol), int(inp.ntor))
axis_of = lambda res: float(measure_position(res.state, modes, vj.PositionControl())[0])  # noqa: E731


def solve(dbz, target=None):
    kw = {}
    if target is not None:
        kw["position_control"] = vj.PositionControl(target=target, **CONTROL_GAINS)
    t0 = time.time()
    res = vj.solve_free_boundary(inp, external_field=make_field(dbz), max_iterations=NITER,
                                 error_on_no_convergence=False, **kw)
    fsq = float(res.fsqr) + float(res.fsqz) + float(res.fsql)
    bz = res.position_control.vertical_field if target is not None else 0.0
    return dict(dbz=dbz, conv=bool(res.converged), iters=int(res.iterations), fsq=fsq,
                axis=axis_of(res), bz=bz, seconds=time.time() - t0)


### Reference: nominal coils, no control ######################################

nominal = solve(0.0)
target = nominal["axis"]
print(f"nominal coils: axis R = {target:.4f} m, fsq {nominal['fsq']:.1e}, {nominal['iters']} iterations")
print("target for the controlled runs = the nominal axis radius\n")

### Mis-set vertical field, with and without control ##########################

print(f"{'B_Z offset':>10s} | {'-- uncontrolled --':^34s} | {'---------- controlled ----------':^52s}")
print(f"{'[mT]':>10s} | {'conv':>4s} {'iters':>5s} {'fsq':>8s} {'axis R':>8s} {'dR[cm]':>7s} | "
      f"{'conv':>4s} {'iters':>5s} {'fsq':>8s} {'axis R':>8s} {'dR[cm]':>7s} {'B_Z ctrl [mT]':>13s}")
rows = []
for dbz in OFFSETS:
    free, held = solve(dbz), solve(dbz, target)
    rows.append((free, held))
    print(f"{1e3 * dbz:10.1f} | {'yes' if free['conv'] else 'NO':>4s} {free['iters']:5d} {free['fsq']:8.1e} "
          f"{free['axis']:8.4f} {100 * (free['axis'] - target):7.1f} | {'yes' if held['conv'] else 'NO':>4s} "
          f"{held['iters']:5d} {held['fsq']:8.1e} {held['axis']:8.4f} "
          f"{100 * (held['axis'] - target):7.2f} {1e3 * held['bz']:13.2f}")

n_fail = sum(not f["conv"] for f, _ in rows)
n_held = sum(h["conv"] for _, h in rows)
worst = max(abs(h["axis"] - target) for _, h in rows)
recov = max(abs(h["bz"] + h["dbz"]) for _, h in rows)
print(f"\nuncontrolled: {n_fail}/{len(rows)} solves fail to converge; the axis wanders "
      f"{100 * max(abs(f['axis'] - target) for f, _ in rows):.0f} cm")
print(f"controlled:   {n_held}/{len(rows)} converge, axis within {1e3 * worst:.1f} mm of the "
      f"target; B_Z^ctrl cancels the offset to {1e3 * recov:.2f} mT")

if MAKE_FIGURE:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    off = np.array([1e3 * f["dbz"] for f, _ in rows])
    fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.8), dpi=110)
    ax = axs[0]
    ax.plot(off, [f["axis"] for f, _ in rows], "o-", color="#b5503c", label="no control")
    ax.plot(off, [h["axis"] for _, h in rows], "s-", color="#2e6da4", label="position control")
    ax.axhline(target, color="0.6", lw=0.8, ls="--")
    ax.set(xlabel="vertical-field offset [mT]", ylabel="magnetic axis R [m]", title="Axis position")
    ax.legend(frameon=False)
    ax = axs[1]
    ax.semilogy(off, [f["fsq"] for f, _ in rows], "o-", color="#b5503c", label="no control")
    ax.semilogy(off, [h["fsq"] for _, h in rows], "s-", color="#2e6da4", label="position control")
    ax.axhline(FTOL, color="0.6", lw=0.8, ls="--")
    ax.set(xlabel="vertical-field offset [mT]", ylabel="final force residual", title="Convergence")
    ax = axs[2]
    ax.plot(off, off, "-", color="0.6", lw=0.8, label="offset")
    ax.plot(off, [-1e3 * h["bz"] for _, h in rows], "s", color="#2e6da4",
            label="$-B_Z^{ctrl}$")
    ax.set(xlabel="vertical-field offset [mT]", ylabel="[mT]", title="Missing field recovered")
    ax.legend(frameon=False)
    for a in axs:
        a.grid(alpha=0.25, lw=0.5)
    fig.suptitle("Free-boundary DIII-D-like tokamak with a mis-set vertical field")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, pil_kwargs={"lossless": True})
    print(f"Wrote {FIGURE_PATH}")

### The same case as an input deck: `vmex examples/data/input.DIII-D_position_control` ###

if MAKE_FIGURE:
    deck = vj.VmecInput.from_file(DECK_FILE)           # LPOSITION_CONTROL = T in the deck
    mgrid = DATA_DIR / deck.mgrid_file
    # PositionControl with zero gains records the uncontrolled trajectory without acting
    # (the run is bitwise identical to position_control=False):
    still = vj.solve_free_boundary_multigrid(
        deck, mgrid_path=mgrid, raise_on_max_iterations=False,
        position_control=vj.PositionControl(gain=0.0, integral_gain=0.0, derivative_gain=0.0,
                                            target=deck.position_target))
    held = vj.solve_free_boundary_multigrid(deck, mgrid_path=mgrid)   # the deck's own control
    hs, hh = still.position_control.history, held.position_control.history
    print(f"deck: uncontrolled converged={still.converged}; controlled converged={held.converged}, "
          f"B_Z^ctrl = {1e3 * held.position_control.vertical_field:.2f} mT")
    fig, axs = plt.subplots(1, 3, figsize=(10.0, 3.2), dpi=80)
    axs[0].plot(hs[:, 0], hs[:, -1], color="#b5503c", label="no control")
    axs[0].plot(hh[:, 0], hh[:, -1], color="#2e6da4", label="position control")
    axs[0].axhline(deck.position_target, color="0.6", lw=0.8, ls="--")
    axs[0].set(xlabel="iteration", ylabel="magnetic axis R [m]", title="Axis position")
    axs[0].legend(frameon=False)
    axs[1].plot(hh[:, 0], 1e3 * hh[:, 2], color="#2e6da4")
    axs[1].axhline(40.0, color="0.6", lw=0.8, ls="--")
    axs[1].set(xlabel="iteration", ylabel="$B_Z^{ctrl}$ [mT]", title="Feedback vertical field")
    axs[2].semilogy(hs[:, 0], hs[:, 1], color="#b5503c", label="no control")
    axs[2].semilogy(hh[:, 0], hh[:, 1], color="#2e6da4", label="position control")
    axs[2].set(xlabel="iteration", ylabel="force residual", title="Convergence")
    for a in axs:
        a.grid(alpha=0.25, lw=0.5)
    fig.suptitle("DIII-D-like tokamak with a 40 mT vertical-field deficit (input.DIII-D_position_control)")
    fig.tight_layout()
    fig.savefig(DECK_FIGURE_PATH, pil_kwargs={"lossless": True})
    print(f"Wrote {DECK_FIGURE_PATH}")
