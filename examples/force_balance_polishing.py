#!/usr/bin/env python
"""Polish a shaped tokamak's force balance and show what it does and does not change.

VMEX first converges the ordinary VMEC discretization. The optional polish then
solves the higher-order strong force-balance residual and certifies the result
with an independent force oracle. A certified equilibrium is exported by
sampling the native state on a denser radial mesh, where the WOUT
reconstruction can carry the polish gain; ``solve_file`` writes that file.

The polish is requested by a VMEX-only directive in the deck, so there is no
Python flag to set here. What this script adds is the evidence:

* the certificate: the same independent oracle on the state before and after
  the polish;
* a fair comparison of the two WOUT files. The unpolished state goes through
  the very same export, onto the same radial mesh, and both files are read
  back through the same importer and oracle, so they differ only in the
  polish. (Comparing the solve-mesh WOUT with the denser polished one
  compares two meshes as well as the polish.)
* what the polish does not do. It removes the near-axis force error and
  lowers the edge error, but it does not lower the error at every radius, and
  the volume-averaged ``<|F|>/<|grad(B^2/2mu0)|>`` falls by much less. The
  force-balance panel of the WOUT summary plot cannot see the gain: its
  finite-difference reconstruction reads about 6e-3 on both files.

The two-panel figure ``polish_before_after.webp`` is the README figure. Outside
the test suite's run it is written straight into ``docs/_static/figures``.

This example has no smoke path: the polish is the demonstration, and the deck
asks for it.
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import vmex as vj
from vmex.core.plotting import _relative_force_error_profile
from vmex.core.polish_driver import polished_wout_state

# Input deck.  Its POLISH_FORCE_BALANCE directive is what requests the polish:
INPUT_FILE = (
    Path(__file__).resolve().parent / "data" / "input.shaped_tokamak_pressure_polished"
)

# Directory that receives every output file.  The before and after summary
# figures go into subdirectories of it:
OUTPUT_DIR = Path("output_force_balance_polishing")
BEFORE_NAME = "shaped_tokamak_before_polish"
README_FIGURE = (Path(__file__).resolve().parents[1] / "docs" / "_static" / "figures"
                 / "readme_polish_before_after.webp")
if os.environ.get("VMEX_EXAMPLES_CI") == "1":
    README_FIGURE = OUTPUT_DIR / "polish_before_after.webp"

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

inp = vj.VmecInput.from_file(INPUT_FILE)

### Solve and polish ##########################################################

# solve_file reads the VMEX-only directive in the deck; because the polish
# certifies, the WOUT it writes samples the native state on the export mesh.
result = vj.solve_file(INPUT_FILE, write_wout=True, outdir=OUTPUT_DIR, verbose=True)
if result.polished_state is None or result.polish_report is None:
    raise RuntimeError("the input deck did not request force-balance polishing")

### Check the result ##########################################################

report = result.polish_report
if not report.converged:
    raise RuntimeError(
        f"the polish did not certify: {report.termination_reason}"
    )
# The same independent force oracle evaluated the legacy state and the
# certified polished state; these are the certificate numbers.  eps_F is the
# acceptance threshold and is bounded above by 2 by construction, so the
# quantities that can actually move are printed with it.
window = report.normalization_window
print(
    "\nindependent strong-force certificate over "
    f"s in [{window[0]:.2f}, {window[1]:.2f}]:"
    f"\n  eps_F volume L2 (<= 2 by construction) "
    f"{report.initial_normalized_l2:.3e} -> {report.final_normalized_l2:.3e}"
    f"\n  <|F|> [N m^-3]                        "
    f"{report.initial_volume_average_force:.3e} -> "
    f"{report.final_volume_average_force:.3e}"
    f"\n  <|F|>/<|grad(B^2/2mu0)|>              "
    f"{report.initial_magnetic_relative_force_error:.3e} -> "
    f"{report.final_magnetic_relative_force_error:.3e}"
)
print(
    f"polish work: {report.nonlinear_iterations} nonlinear iterations, "
    f"{report.solve_seconds:.2f} s"
)

### Write both WOUT files on the same mesh ####################################

# solve_file already wrote the certified polished WOUT.  Put the ordinary VMEC
# state through the same export, so the two files differ only in the polish:
# read it as a continuous state and sample that on the polished file's mesh.
solve_ns = int(np.shape(np.asarray(result.state.R_cos))[0])
run = dict(fsqr=float(result.fsqr), fsqz=float(result.fsqz), fsql=float(result.fsql),
           niter=int(result.iterations), converged=bool(result.converged))
legacy = vj.high_order_state_from_wout(
    vj.wout_from_state(inp=inp, state=result.state, **run), inp=inp)
legacy_path = vj.write_wout(
    OUTPUT_DIR / f"wout_{BEFORE_NAME}.nc",
    vj.wout_from_state(inp=inp, state=polished_wout_state(legacy, inp, solve_ns=solve_ns),
                       **run),
)
case = INPUT_FILE.name.removeprefix("input.")
polished_path = OUTPUT_DIR / f"wout_{case}.nc"
print(f"Wrote {legacy_path}\nusing {polished_path}")

### Compare the two files #####################################################

# Read each file back through the same importer and certify it with the same
# oracle.  This is what a user of the WOUT gets.
files = {"VMEC solve": legacy_path, "polished": polished_path}
certificates = {
    label: vj.certify_strong_force(vj.high_order_state_from_wout(path, inp=inp))
    for label, path in files.items()
}
before, after = certificates.values()
print(f"\nboth WOUT files on ns = {int(vj.read_wout(polished_path).ns)}, "
      "read back and certified the same way (before -> after):")
for label, field in (("RMS |F|, whole volume  [N m^-3]", "absolute_l2"),
                     ("RMS |F|, rho < 0.2     [N m^-3]", "near_axis_l2"),
                     ("RMS |F|, 0.2..0.8      [N m^-3]", "bulk_l2"),
                     ("RMS |F|, rho > 0.8     [N m^-3]", "edge_l2")):
    print(f"  {label} {float(getattr(before, field)):.3e} => "
          f"{float(getattr(after, field)):.3e}")
print(f"  window <|F|>/<|grad(B^2/2mu0)|>  "
      f"{float(before.window_normalizations.magnetic_relative_force_error):.3e} => "
      f"{float(after.window_normalizations.magnetic_relative_force_error):.3e}")
print("  the written polished file reproduces the certificate: eps_F "
      f"{report.final_normalized_l2:.3e} native, {float(after.normalized_l2):.3e} "
      "read back")
worse = (np.asarray(after.flux_surface_average) > np.asarray(before.flux_surface_average))
if worse.any():
    rho = np.asarray(after.radial_nodes)[worse]
    print(f"  the polish does not lower the error everywhere: it is higher on "
          f"{worse.mean():.0%} of the surfaces, between rho = {rho.min():.2f} "
          f"and {rho.max():.2f}")
summary = [_relative_force_error_profile(vj.read_wout(path))[2] for path in files.values()]
print(f"  summary-plot force panel: {summary[0]:.3e} => {summary[1]:.3e}; this "
      "WOUT finite-difference reconstruction does not resolve the change")

### Plot and save #############################################################

INK2, GRID, BASELINE = "#52514e", "#e1e0d9", "#c3c2b7"
colors = {"VMEC solve": "#eda100", "polished": "#2a78d6"}
plt.rcParams.update({"axes.edgecolor": BASELINE, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False})
figure, (profile, regions) = plt.subplots(1, 2, figsize=(9.0, 3.6),
                                          gridspec_kw={"width_ratios": [1.6, 1.0]},
                                          layout="constrained")
for label, certificate in certificates.items():
    rho = np.asarray(certificate.radial_nodes)
    force = np.asarray(certificate.flux_surface_average)
    profile.semilogy(rho, force, color=colors[label], linewidth=2.0, label=label)
profile.set(xlim=(0.0, 1.0), xlabel=r"$\rho=\sqrt{s}$",
            ylabel=r"$\langle|\mathbf{J}\times\mathbf{B}-\nabla p|\rangle_s$  [N m$^{-3}$]")
profile.grid(True, which="major", color=GRID)
profile.legend(loc="upper center")
names = ("near axis\n" r"$\rho<0.2$", "bulk\n" r"$0.2\leq\rho\leq0.8$", "edge\n" r"$\rho>0.8$")
x, width = np.arange(3), 0.38
for i, (label, certificate) in enumerate(certificates.items()):
    values = [float(certificate.near_axis_l2), float(certificate.bulk_l2),
              float(certificate.edge_l2)]
    bars = regions.bar(x + (i - 0.5) * (width + 0.02), values, width,
                       color=colors[label], label=label)
    regions.bar_label(bars, fmt="%.0f", fontsize=8, color=INK2, padding=2)
regions.set_yscale("log")
regions.set_xticks(x, names)
regions.set_ylabel(r"RMS $|\mathbf{J}\times\mathbf{B}-\nabla p|$  [N m$^{-3}$]")
regions.grid(True, axis="y", which="major", color=GRID)
regions.legend(loc="upper right")
figure.savefig(README_FIGURE, dpi=160, pil_kwargs={"lossless": True})
plt.close(figure)
print(f"Wrote {README_FIGURE}")

# The full WOUT summaries, now on the same mesh.  Their force-balance panel is
# the finite-difference reconstruction printed above, not the certificate.
for stage, path in (("before", legacy_path), ("after", polished_path)):
    stage_dir = OUTPUT_DIR / stage
    for figure_path in vj.plot_wout(path, stage_dir).values():
        print(f"Wrote {figure_path}")
