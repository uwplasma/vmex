#!/usr/bin/env python
"""Compare a finite-beta VMEX equilibrium with its MRX relaxation.

VMEX assumes nested flux surfaces; MRX (https://github.com/ToBlick/mrx)
does not. MRX represents B as a divergence-free finite-element 2-form on the
VMEX geometry and relaxes it with Newton steps until J x B - grad p vanishes,
so magnetic surfaces and the rotational transform are outputs, read off
traced field lines. Agreement between the two is a check on both.

The deck is the 2.5 % beta Landreman-Paul QA with a bootstrap current. Near
the magnetic axis VMEX needs many radial surfaces at this beta: iota at
rho = 0.1 moves 0.200 -> 0.188 -> 0.176 -> 0.166 for ns = 33, 65, 129, 257,
where DESC gives 0.166. To keep the run short this example stops at ns = 65
and MRX starts from that field, inheriting its near-axis iota; extend
NS_ARRAY to 129 or 257 to see iota near the axis move toward DESC.

The figure shows (a-c) VMEX flux surfaces and MRX Poincare points at three
toroidal angles, (d) iota, (e) the enclosed toroidal current, and (f) the
pressure. MRX normalizes its field to unit energy; its amplitude is fixed by
matching the VMEX toroidal flux, which sets every other quantity in SI units.

MRX is optional: ``pip install mrx`` (Python >= 3.11). Without it the script
prints the install command and exits. A full run takes about 4 minutes on a
laptop CPU from a cold cache; the solve and the MRX field are kept in
OUTPUT_DIR so a second run only redraws the figure.
"""

import importlib.util
import os
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import vmex as vj

INPUT_FILE = Path(__file__).resolve().parent / "data" / "input.LandremanPaul2021_QA_beta2p5_bootstrap"

# VMEX radial grids, the last of which MRX starts from:
NS_ARRAY, FTOL_ARRAY, NITER_ARRAY = [17, 33, 65], [1e-13] * 3, [20000] * 3
# MRX splines per (r, theta, zeta), spline degree, and Newton steps:
MRX_RESOLUTION, MRX_DEGREE, NEWTON_STEPS = (12, 16, 16), 3, 10
# Field lines and field periods traced for the Poincare sections:
POINCARE_LINES, POINCARE_PERIODS = 20, 150
# Reuse the wout and the relaxed MRX field found in OUTPUT_DIR:
REUSE_OUTPUTS = True
OUTPUT_DIR = Path("output_vmex_mrx_comparison")

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    NS_ARRAY, FTOL_ARRAY, NITER_ARRAY = [9, 17], [1e-8, 1e-10], [2000, 4000]
    MRX_RESOLUTION, MRX_DEGREE, NEWTON_STEPS = (5, 6, 6), 2, 1
    POINCARE_LINES, POINCARE_PERIODS = 4, 20
    REUSE_OUTPUTS = False

###############################################################################
# End of input parameters.
###############################################################################

if importlib.util.find_spec("mrx") is None:
    print("MRX is not installed; pip install mrx (Python >= 3.11) to run the comparison.")
    sys.exit(0)

MU0 = 4e-7 * np.pi
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
wout_path = OUTPUT_DIR / f"wout_QA_beta2p5_ns{NS_ARRAY[-1]}.nc"
mrx_path = OUTPUT_DIR / f"mrx_field_ns{NS_ARRAY[-1]}.npz"

### VMEX ######################################################################

if REUSE_OUTPUTS and wout_path.exists():
    print(f"Reusing {wout_path}")
else:
    inp = replace(vj.VmecInput.from_file(INPUT_FILE), ns_array=np.array(NS_ARRAY),
                  ftol_array=np.array(FTOL_ARRAY), niter_array=np.array(NITER_ARRAY))
    result = vj.solve_multigrid(inp, verbose=not ci_smoke)
    print(f"VMEX ns={NS_ARRAY[-1]}: converged = {result.converged}, fsqr = {float(result.fsqr):.2e}")
    vj.write_wout(wout_path, vj.wout_from_result(inp, result))
wout = vj.read_wout(wout_path)
ns, nfp = int(wout.ns), int(wout.nfp)
s_full = np.linspace(0.0, 1.0, ns)
s_half = 0.5 * (s_full[1:] + s_full[:-1])


def vmex_RZ(s_index, theta, phi):
    """R, Z of the full-mesh surface ``s_index`` at the angles theta, phi."""
    angle = np.multiply.outer(theta, wout.xm) - np.multiply.outer(phi, wout.xn)
    return np.cos(angle) @ wout.rmnc[s_index], np.sin(angle) @ wout.zmns[s_index]


def vmex_modB(s, theta, phi):
    """|B| at VMEC coordinates (s, theta, phi), interpolating the half mesh linearly in s."""
    angle = np.multiply.outer(theta, wout.xm_nyq) - np.multiply.outer(phi, wout.xn_nyq)
    bmnc = np.stack([np.interp(s, s_half, wout.bmnc[1:, k]) for k in range(wout.bmnc.shape[1])], -1)
    return np.sum(bmnc * np.cos(angle), axis=-1)


vmex_I = 2 * np.pi * np.abs(wout.bsubumnc[1:, 0]) / MU0     # Ampere: I(s) = 2 pi <B_u> / mu0
print(f"VMEX: iota axis/edge = {abs(wout.iotaf[0]):.4f}/{abs(wout.iotaf[-1]):.4f}, "
      f"beta = {float(wout.betatotal):.4f}, I(edge) = {vmex_I[-1] / 1e3:.1f} kA")

### MRX #######################################################################

os.environ.setdefault("MRX_DTYPE", "float64")
os.environ.setdefault("MRX_RESIDUAL_DTYPE", "float64")
try:
    import jax
    import jax.numpy as jnp
    from mrx.diagnostics.poincare import logical_field, poincare
    from mrx.differential_forms import DiscreteFunction
    from mrx.nullspace import compute_nullspaces
    from mrx.relaxation.config import Budget, Geometry, Newton, Precision, RelaxConfig
    from mrx.relaxation.initial_conditions import initial_field
    from mrx.relaxation.loop import initial_state, relax
    from mrx.relaxation.physics import beta_vol, compute_force, weak_pressure
except ImportError as error:
    sys.exit(f"MRX is installed but could not be imported: {error}")

cfg = RelaxConfig(
    geometry=Geometry(path=str(wout_path), resolution=MRX_RESOLUTION, spline_degree=MRX_DEGREE,
                      precision=Precision("float64")),
    newton=Newton(), budget=Budget(steps=NEWTON_STEPS, chunk=min(5, NEWTON_STEPS), floor_tol=0.0))
seq, _ = cfg.geometry.build()
compute_nullspaces(seq)
B0, _ = initial_field(seq)
if REUSE_OUTPUTS and mrx_path.exists():
    print(f"Reusing {mrx_path}")
    saved = np.load(mrx_path)
    B, summary = jnp.asarray(saved["B"]), {k: float(saved[k]) for k in saved.files if k != "B"}
else:
    stepper = cfg.stepper(seq)
    res = relax(initial_state(B0, stepper), stepper, **cfg.relax_kwargs())
    B = res.state.best.B
    helicity = np.asarray(res.qoi["helicity"])
    summary = dict(resid=float(res.state.best.resid), div=float(res.trace["div"][-1]),
                   dH=float((helicity[-1] - helicity[0]) / helicity[0]),
                   beta0=float(res.qoi["beta_vol"][0]), wall=float(res.wall))
    np.savez(mrx_path, B=np.asarray(B), **summary)

field = logical_field(seq)
_, _, J, _ = compute_force(B, seq)
p_w = weak_pressure(J, B, seq)
summary["beta"] = float(beta_vol(B, p_w, seq))
print(f"MRX: force residual {summary['resid']:.1e}, |div B| {summary['div']:.0e}, "
      f"helicity drift {summary['dH']:+.0e}, beta_vol {summary['beta0']:.4f} -> {summary['beta']:.4f}")

# The amplitude: the toroidal flux through zeta = 0 is the integral of the
# logical component B^zeta over (r, theta) in [0, 1]^2.
nodes, weights = np.polynomial.legendre.leggauss(64)
nodes, weights = 0.5 * (nodes + 1), 0.5 * weights
r_q, t_q = np.meshgrid(nodes, nodes, indexing="ij")
x_q = jnp.stack([r_q.ravel(), t_q.ravel(), 0 * r_q.ravel()], 1)
flux = float(jnp.sum(jax.vmap(field, (0, None))(x_q, B)[:, 2] * np.outer(weights, weights).ravel()))
scale = abs(float(wout.phi[-1]) / flux)


@jax.jit
def mrx_frame(x):
    """|B| [T] and the logical covariant theta component B_theta [T m] at logical points x."""
    def one(x):
        DPhi = jax.jacfwd(seq.map)(x)
        b = DPhi @ field(x, B) / jnp.linalg.det(DPhi)
        return jnp.linalg.norm(b), (DPhi.T @ b)[1]
    return jax.vmap(one)(x)


pressure_0form = DiscreteFunction(p_w, seq.basis_0, seq.even.E(0))
theta_l = (np.arange(128) + 0.5) / 128
rho = np.linspace(0.02, 0.98, 49)
grid = jnp.stack(np.broadcast_arrays(rho[:, None], theta_l[None, :], 0.0), -1).reshape(-1, 3)
modB_mrx, B_theta = (scale * np.asarray(a).reshape(rho.size, -1) for a in mrx_frame(grid))
mrx_I = np.abs(B_theta.mean(axis=1)) / MU0                   # Ampere around r = const
mrx_p = scale**2 / MU0 * np.asarray(jax.vmap(pressure_0form)(grid))[:, 0].reshape(rho.size, -1)

# The MRX map is a spline fit of the VMEX surfaces, so logical (r, theta, zeta)
# is VMEC (sqrt(s), 2 pi theta, 2 pi zeta / nfp) up to that fit (theta flips
# for a left-handed file): |B| can be compared at the same coordinates.
sign = -1.0 if seq.equilibrium["theta_reversed"] else 1.0
modB_vmex = vmex_modB(np.repeat(rho**2, theta_l.size), sign * 2 * np.pi * np.tile(theta_l, rho.size),
                      np.zeros(grid.shape[0])).reshape(rho.size, -1)
dB = np.abs(modB_mrx - modB_vmex) / modB_vmex

planes = (0.0, 0.25, 0.5)                                   # phi = 0, pi / (2 nfp), pi / nfp
sections = poincare(seq, B, lines=POINCARE_LINES, periods=POINCARE_PERIODS, planes=planes)
shown = sections["shown"]
iota_mrx, rho_mrx = sections["iota"][shown], sections["seed_r"][shown]
inner = rho_mrx.argmin() if shown.any() else None

### Figure ####################################################################

fig = plt.figure(figsize=(13.5, 8.6), layout="constrained")
grid_spec = fig.add_gridspec(2, 3)
for k, plane in enumerate(planes):
    ax = fig.add_subplot(grid_spec[0, k])
    phi = 2 * np.pi * plane / nfp
    theta = np.linspace(0, 2 * np.pi, 200)
    for j in np.unique(np.linspace(1, ns - 1, 9).round().astype(int)):
        R, Z = vmex_RZ(j, theta, np.full_like(theta, phi))
        ax.plot(R, Z, color="0.25", lw=0.8, label="VMEX surfaces" if j == ns - 1 else None)
    section = sections["sections"][plane]
    for line in np.flatnonzero(shown):
        ax.plot(section["R"][line], section["Z"][line], ".", color="C1", ms=1.6, mec="none",
                label="MRX field lines" if line == np.flatnonzero(shown)[0] else None)
    ax.plot(section["axisR"][::20], section["axisZ"][::20], "+", color="C3", ms=8, label="MRX axis")
    ax.plot(*vmex_RZ(0, np.zeros(1), np.full(1, phi)), "x", color="k", ms=7, label="VMEX axis")
    ax.set_aspect("equal")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(f"({'abc'[k]}) " + [r"$\phi = 0$", r"$\phi = \pi/(2 n_{fp})$", r"$\phi = \pi/n_{fp}$"][k])
    if k == 0:
        handles, labels = ax.get_legend_handles_labels()
        handles[1] = plt.Line2D([], [], ls="none", marker=".", color="C1", ms=8)
        fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=10, frameon=False)

ax = fig.add_subplot(grid_spec[1, 0])
ax.plot(np.sqrt(s_full), np.abs(wout.iotaf), "-", color="0.25", label=f"VMEX ns={ns}")
ax.plot(rho_mrx, iota_mrx, "o", color="C1", ms=4, label="MRX, traced")
ax.set_xlabel(r"$\rho$ (VMEX) or seed $r$ (MRX)")
ax.set_ylabel(r"$\iota$")
ax.set_title(r"(d) rotational transform")
ax.legend(fontsize=9)

ax = fig.add_subplot(grid_spec[1, 1])
ax.plot(np.sqrt(s_half), vmex_I / 1e3, "-", color="0.25", label=r"VMEX $2\pi\langle B_\theta\rangle/\mu_0$")
ax.plot(rho, mrx_I / 1e3, "o", color="C1", ms=3.5, label=r"MRX $\oint B\cdot dl/\mu_0$, $\phi=0$")
ax.set_xlabel(r"$\rho$")
ax.set_ylabel(r"enclosed toroidal current $|I|$ [kA]")
ax.set_title("(e) toroidal current")
ax.legend(fontsize=9)

ax = fig.add_subplot(grid_spec[1, 2])
ax.plot(np.sqrt(s_full), wout.presf / 1e3, "-", color="0.25", label="VMEX $p(s)$")
ax.fill_between(rho, mrx_p.min(axis=1) / 1e3, mrx_p.max(axis=1) / 1e3, color="C1", alpha=0.3,
                lw=0, label=r"MRX $p_w$, range over $\theta$")
ax.plot(rho, mrx_p.mean(axis=1) / 1e3, "--", color="C1", label=r"MRX $p_w$, mean over $\theta$")
ax.set_xlabel(r"$\rho$ (VMEX) or $r$ (MRX), $\phi = 0$")
ax.set_ylabel("pressure [kPa]")
ax.set_title("(f) pressure")
ax.legend(fontsize=9)

fig.suptitle(f"QA, 2.5% beta with bootstrap current: VMEX (ns={ns}) and MRX "
             f"({MRX_RESOLUTION}, p={MRX_DEGREE}, {NEWTON_STEPS} Newton steps)\n"
             "VMEX near-axis iota converges slowly with ns: 0.188, 0.176, 0.166 at rho = 0.1 "
             "for ns = 65, 129, 257 (DESC 0.166)", fontsize=11)
figure_path = OUTPUT_DIR / "vmex_mrx_comparison.png"
fig.savefig(figure_path, dpi=150)
print(f"Wrote {figure_path}")
if not ci_smoke:
    # The docs copy: docs/_static/figures/readme_vmex_mrx_comparison.webp.
    fig.savefig(OUTPUT_DIR / "readme_vmex_mrx_comparison.webp", dpi=90,
                pil_kwargs=dict(quality=70, method=6))

core = rho >= 0.3
print(f"MRX: iota at seed r = {rho_mrx.min():.2f}/{rho_mrx.max():.2f}: "
      f"{iota_mrx[inner]:.4f}/{iota_mrx[rho_mrx.argmax()]:.4f}" if inner is not None
      else "MRX: no regular field line")
print(f"MRX: I(r = {rho[-1]:.2f}) = {mrx_I[-1] / 1e3:.1f} kA, "
      f"VMEX: I(rho = {rho[-1]:.2f}) = {np.interp(rho[-1]**2, s_half, vmex_I) / 1e3:.1f} kA")
print(f"|B| difference MRX vs VMEX at phi = 0: rms {np.sqrt(np.mean(dB[core]**2)):.1e}, "
      f"max {dB[core].max():.1e} for rho >= 0.3; max {dB[~core].max():.1e} for rho < 0.3")
