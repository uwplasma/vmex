#!/usr/bin/env python
"""Polish a shaped tokamak to a certified, stationary continuum force balance.

The ordinary VMEC solve converges its discrete equations, which does not bound
the continuum force J x B - grad p.  ``polish=True`` re-solves that force on a
native quintic-spline representation and certifies it with an independent
oracle.  This script prints the certificate before and after, checks that the
written WOUT carries the polished state, and draws the two README figures:
the certified force profiles, and ``vmex --plot`` of both WOUT files side by
side.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import vmex as vj
from _polish_figure import polish_figure
from vmex.core.polish import polished_wout_input, polished_wout_state

# Input deck: fixed-boundary, axisymmetric, prescribed pressure and iota.
INPUT_FILE = Path(__file__).resolve().parent / "data" / "input.shaped_tokamak_pressure"

# Directory that receives every output file:
OUTPUT_DIR = Path("output_force_balance_polishing")
BEFORE_NAME = "shaped_tokamak_before_polish"
FIGURES = Path(__file__).resolve().parents[1] / "docs" / "_static" / "figures"
if os.environ.get("VMEX_EXAMPLES_CI") == "1":
    FIGURES = OUTPUT_DIR
README_FIGURE = FIGURES / ("polish_before_after.webp" if FIGURES == OUTPUT_DIR else "readme_polish_before_after.webp")
PLOT_FIGURE = FIGURES / ("polish_plot.webp" if FIGURES == OUTPUT_DIR else "readme_polish_plot.webp")

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
      f"{report.stationarity:.1e}")

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

INK2 = "#52514e"
names = ("near axis\n" r"$\rho<0.2$", "bulk\n" r"$0.2\leq\rho\leq0.8$", "edge\n" r"$\rho>0.8$")
polish_figure(
    README_FIGURE,
    {label: (np.asarray(c.radial_nodes), np.asarray(c.flux_surface_average))
     for label, c in certificates.items()},
    {label: [float(c.near_axis_l2), float(c.bulk_l2), float(c.edge_l2)]
     for label, c in certificates.items()},
    names, r"$\langle|\mathbf{J}\times\mathbf{B}-\nabla p|\rangle_s$  [N m$^{-3}$]",
    r"RMS $|\mathbf{J}\times\mathbf{B}-\nabla p|$  [N m$^{-3}$]")

# `vmex --plot` of both files, side by side.  Its force panel is a
# finite-difference reconstruction from the WOUT, not the certificate above.
summaries = [vj.plot_wout(path, OUTPUT_DIR / stage, which=("summary",))["summary"]
             for stage, path in (("before", legacy_path), ("after", polished_path))]
figure, axes = plt.subplots(1, 2, figsize=(16.0, 6.4), layout="constrained")
for axis, image, title in zip(axes, summaries, ("VMEC solve", "polished")):
    axis.imshow(plt.imread(image))
    axis.set_title(title, color=INK2, fontsize=14)
    axis.axis("off")
figure.savefig(PLOT_FIGURE, dpi=110, bbox_inches="tight", pad_inches=0.05, pil_kwargs={"lossless": True})
plt.close(figure)
print(f"Wrote {legacy_path}\nWrote {polished_path}\nWrote {README_FIGURE}\nWrote {PLOT_FIGURE}")
