#!/usr/bin/env python
"""Polish an equilibrium's force balance and compare the exported WOUT files.

VMEX first converges the ordinary VMEC discretization. The optional polish then
solves the higher-order strong force-balance residual and certifies the result
with independent force, radial-refinement and positive-Jacobian checks. A
certified equilibrium is exported by sampling the native state on a denser
radial mesh, where the WOUT reconstruction can carry the polish gain;
``solve_file`` writes that file directly.

The polish is requested by a VMEX-only directive in the deck, so there is no
Python flag to set here. What this script adds is the evidence: it refuses to
continue unless the polish certified, prints the independent strong-force
certificate, and saves and plots the WOUT before and after so the two can be
compared.

This example has no smoke path: the polish is the demonstration, and the deck
asks for it.
"""

from pathlib import Path

import vmex as vj

# Input deck.  Its POLISH_FORCE_BALANCE directive is what requests the polish:
INPUT_FILE = (
    Path(__file__).resolve().parent / "data" / "input.shaped_tokamak_pressure_polished"
)

# Directory that receives every output file.  The before and after figures go
# into subdirectories of it:
OUTPUT_DIR = Path("output_force_balance_polishing")
BEFORE_NAME = "shaped_tokamak_before_polish"

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

# solve_file reads the VMEX-only directive in the deck; because the polish
# certifies, the WOUT it writes samples the native state on the dense export
# mesh.  VmecInput itself contains physics only, so solve_multigrid would
# require an explicit Python flag instead.
inp = vj.VmecInput.from_file(INPUT_FILE)

### Solve and polish ##########################################################

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
# quantities that can actually move are printed with it: the dimensional
# volume-averaged force error and the vacuum-safe normalization.
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

### Print, plot and save ######################################################

# solve_file already wrote the certified polished WOUT; export the ordinary
# VMEC state alongside it for the before/after comparison.
legacy_path = vj.write_wout(
    OUTPUT_DIR / f"wout_{BEFORE_NAME}.nc",
    vj.wout_from_state(
        inp=inp,
        state=result.state,
        fsqr=float(result.fsqr),
        fsqz=float(result.fsqz),
        fsql=float(result.fsql),
        niter=int(result.iterations),
        converged=bool(result.converged),
    ),
)
case = INPUT_FILE.name.removeprefix("input.")
polished_path = OUTPUT_DIR / f"wout_{case}.nc"
print(f"Wrote {legacy_path}\nusing {polished_path}")

# The printed certificate is the polish evidence; the summary's radial
# force-balance panel shows VMEC's discrete flux-surface-averaged residual
# (wout equif), which the ordinary solve minimizes by construction.
for stage, path in (("before", legacy_path), ("after", polished_path)):
    stage_dir = OUTPUT_DIR / stage
    for figure_path in vj.plot_wout(path, stage_dir).values():
        print(f"Wrote {figure_path}")
