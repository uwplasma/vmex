#!/usr/bin/env python
"""Polish a shaped tokamak to a certified, stationary continuum force balance.

The ordinary VMEC solve converges its discrete equations, which does not bound
the continuum force J x B - grad p.  ``polish=True`` re-solves that force on a
native quintic-spline representation and certifies it with an independent
oracle.  This script prints the certificate before and after, checks that the
written WOUT carries the polished state, and draws the README figure.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import vmex as vj
from vmex.core.polish_driver import polished_wout_input, polished_wout_state

# Input deck: fixed-boundary, axisymmetric, prescribed pressure and iota.
INPUT_FILE = Path(__file__).resolve().parent / "data" / "input.shaped_tokamak_pressure"

# Directory that receives every output file:
OUTPUT_DIR = Path("output_force_balance_polishing")
BEFORE_NAME = "shaped_tokamak_before_polish"
README_FIGURE = (Path(__file__).resolve().parents[1] / "docs" / "_static" / "figures"
                 / "readme_polish_before_after.webp")
if os.environ.get("VMEX_EXAMPLES_CI") == "1":
    README_FIGURE = OUTPUT_DIR / "polish_before_after.webp"

###############################################################################
# End of input parameters.
###############################################################################

### Solve and polish ##########################################################

inp = vj.VmecInput.from_file(INPUT_FILE)
result = vj.solve_file(INPUT_FILE, polish=True, write_wout=True, outdir=OUTPUT_DIR, verbose=True)
if result.polished_state is None or result.polish_report is None:
    raise RuntimeError("the solve did not run force-balance polishing")
report = result.polish_report
if not report.converged:
    raise RuntimeError(f"the polish did not certify: {report.termination_reason}")

### The certificate ###########################################################

# The same independent oracle on the lifted VMEC state and on the polished
# state.  eps_F is bounded above by 2 by construction; the other two rows are
# the dimensional and volume-normalized force over the stated flux window.
window = report.normalization_window
print(
    "\nindependent strong-force certificate over "
    f"s in [{window[0]:.2f}, {window[1]:.2f}]:"
    f"\n  eps_F volume L2 (<= 2 by construction) "
    f"{report.initial_normalized_l2:.3e} -> {report.final_normalized_l2:.3e}"
    f"\n  <|F|> [N m^-3]                        "
    f"{report.initial_volume_average_force:.3e} -> {report.final_volume_average_force:.3e}"
    f"\n  <|F|>/<|grad(B^2/2mu0)|>              "
    f"{report.initial_magnetic_relative_force_error:.3e} -> "
    f"{report.final_magnetic_relative_force_error:.3e}"
)
print(f"polish work: {report.nonlinear_iterations} nonlinear iterations, "
      f"{report.solve_seconds:.1f} s, projected stationarity "
      f"{report.least_squares_relative_optimality:.1e}")

### Both WOUT files on the same mesh ##########################################

# solve_file wrote the polished WOUT.  Put the ordinary VMEC state through the
# same export and read both back on the polished spline basis, so the two
# files differ only in the polish.
native = result.native_equilibrium
run = dict(fsqr=float(result.fsqr), fsqz=float(result.fsqz), fsql=float(result.fsql),
           niter=int(result.iterations), converged=bool(result.converged))
legacy = vj.high_order_state_from_wout(vj.wout_from_state(inp=inp, state=result.state, **run), inp=inp)
polished_path = OUTPUT_DIR / f"wout_{INPUT_FILE.name.removeprefix('input.')}.nc"
ns = int(vj.read_wout(polished_path).ns)
legacy_path = vj.write_wout(
    OUTPUT_DIR / f"wout_{BEFORE_NAME}.nc",
    vj.wout_from_state(inp=inp, state=polished_wout_state(legacy, inp, solve_ns=ns), **run),
)
files = {"VMEC solve": (legacy_path, inp), "polished": (polished_path, polished_wout_input(native, inp))}
certificates = {
    label: vj.certify_strong_force(
        vj.high_order_state_from_wout(path, inp=deck, radial_basis=native.radial_basis))
    for label, (path, deck) in files.items()
}
before, after = certificates.values()
print(f"\nboth WOUT files on ns = {ns}, read back and certified the same way:")
for label, field in (("RMS |F|, whole volume  [N m^-3]", "absolute_l2"),
                     ("RMS |F|, rho < 0.2     [N m^-3]", "near_axis_l2"),
                     ("RMS |F|, 0.2..0.8      [N m^-3]", "bulk_l2"),
                     ("RMS |F|, rho > 0.8     [N m^-3]", "edge_l2")):
    print(f"  {label} {float(getattr(before, field)):.3e} => {float(getattr(after, field)):.3e}")
print(f"  the written file reproduces the native state: RMS |F| "
      f"{float(vj.certify_strong_force(native).absolute_l2):.4e} native, "
      f"{float(after.absolute_l2):.4e} read back")

### Plot and save #############################################################

INK2, GRID, BASELINE = "#52514e", "#e1e0d9", "#c3c2b7"
colors = {"VMEC solve": "#eda100", "polished": "#2a78d6"}
plt.rcParams.update({"axes.edgecolor": BASELINE, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False})
figure, (profile, regions) = plt.subplots(1, 2, figsize=(9.0, 3.6),
                                          gridspec_kw={"width_ratios": [1.6, 1.0]},
                                          layout="constrained")
for label, certificate in certificates.items():
    profile.semilogy(np.asarray(certificate.radial_nodes), np.asarray(certificate.flux_surface_average),
                     color=colors[label], linewidth=2.0, label=label)
profile.set(xlim=(0.0, 1.0), xlabel=r"$\rho=\sqrt{s}$",
            ylabel=r"$\langle|\mathbf{J}\times\mathbf{B}-\nabla p|\rangle_s$  [N m$^{-3}$]")
profile.grid(True, which="major", color=GRID)
profile.legend(loc="upper center")
names = ("near axis\n" r"$\rho<0.2$", "bulk\n" r"$0.2\leq\rho\leq0.8$", "edge\n" r"$\rho>0.8$")
x, width = np.arange(3), 0.38
for i, (label, certificate) in enumerate(certificates.items()):
    values = [float(certificate.near_axis_l2), float(certificate.bulk_l2), float(certificate.edge_l2)]
    bars = regions.bar(x + (i - 0.5) * (width + 0.02), values, width, color=colors[label], label=label)
    regions.bar_label(bars, fmt="%.0f", fontsize=8, color=INK2, padding=2)
regions.set_yscale("log")
regions.set_xticks(x, names)
regions.set_ylabel(r"RMS $|\mathbf{J}\times\mathbf{B}-\nabla p|$  [N m$^{-3}$]")
regions.grid(True, axis="y", which="major", color=GRID)
regions.legend(loc="upper right")
figure.savefig(README_FIGURE, dpi=160, pil_kwargs={"lossless": True})
plt.close(figure)
print(f"Wrote {legacy_path}\nWrote {polished_path}\nWrote {README_FIGURE}")
