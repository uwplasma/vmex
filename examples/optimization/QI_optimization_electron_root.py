#!/usr/bin/env python
"""Optimize a QI boundary for a positive ambipolar radial electric field (an electron root).

The seed is the nfp = 2 vacuum QI boundary of QI_optimization.py. The
ambipolarity condition Gamma_i(E_r) = Gamma_e(E_r) of an electron-ion plasma
(Te >> Ti, low density) has one or three roots per surface. Its largest root is
the electron root when it is positive, the regime of Lascas Neto et al.,
arXiv:2405.12058. NTX (``pip install ntx``) supplies the monoenergetic D11
tables from the Boozer spectrum of the VMEX equilibrium and convolves them with
Maxwellians (ntx.ambipolar.ElectronRoot). The objective term is a hinge
asking the residual Gamma_i - Gamma_e to be negative at E_TARGET, which holds
exactly when a root above E_TARGET exists. SciPy least squares then varies the
boundary against that hinge, the constructed-QI residual, the aspect ratio and
hinges on the rotational transform and the mirror ratio. The roots, with
implicit-function derivatives, are reported before and after.
"""

import os
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import matplotlib
import numpy as np
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import vmex as vj
from vmex import optimize as opt
from vmex.core.qi import ConstructedQIResidual
from ntx.ambipolar import AmbipolarPlasma, ElectronRoot

# The seed deck (constructed QI 5e-3):
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "input.QI_nfp2_initial"

STAGE_NS = 31
STAGE_FTOL = 1.0e-12
STAGE_NITER = 5500

# Surfaces of the QI residual, and of the electron-root hinge (normalized
# radius rho = sqrt(s)); the report and plots use ROOT_PLOT_RHO:
SURFACES = np.linspace(0.1, 1.0, 6)
ROOT_RHO = (0.6,)
ROOT_PLOT_RHO = (0.3, 0.45, 0.6, 0.75, 0.9)

# Plasma, functions of rho: electron density in m^-3, temperatures in eV.
# Hot electrons, cold ions, low density:
N0, TE0, TI0 = 1.0e18, 8.0e3, 1.0e3
plasma = AmbipolarPlasma(
    density=lambda r: N0 * (1.0 - r**2) + 1.0e17,
    te=lambda r: TE0 * (1.0 - r**2) ** 1.5 + 100.0,
    ti=lambda r: TI0 * (1.0 - r**2) + 100.0)

# The root target is -dPhi/ds in volts (about 4.8 V/m of E_r per volt at
# rho = 0.6 and a = 0.25 m); the hinge holds the margin below the target flux
# imbalance, and the field axis of the D11 tables ends at E_MAX:
E_TARGET = 3000.0
E_MARGIN = 0.25
E_MAX = 2.0e4

MAX_MODES = [1, 2]
MAX_NFEV = [10, 12]

QI_OPTIONS = dict(mboz=10, nboz=10, nphi=41, nalpha=12, n_bounce=13)

ASPECT_LIMIT = 10.0
IOTA_FLOOR = 0.51
MIRROR_LIMIT = 0.21
MIRROR_HINGE_FRACTION = 0.99
ROOT_WEIGHT = 30.0

ESS_ALPHA = 1.2
MINIMUM_MPOL = 5

FINAL_NS = 31
FINAL_FTOL = 1.0e-10
FINAL_NITER = 20000

OUTPUT_NAME = "QI_electron_root_optimized"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    QI_OPTIONS = dict(mboz=8, nboz=8, nphi=31, nalpha=7, n_bounce=7)
    MAX_MODES, MAX_NFEV = [1], [3]
    ROOT_PLOT_RHO = (0.6,)

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

inp = replace(vj.VmecInput.from_file(INPUT_FILE),
              ns_array=np.array([STAGE_NS]), ftol_array=np.array([STAGE_FTOL]),
              niter_array=np.array([STAGE_NITER]), delt=0.5)
mpol = max(max(MAX_MODES) + 2, MINIMUM_MPOL)
inp = inp.change_resolution(mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
seed = opt.solve_equilibrium(inp)
phiedge = float(seed.wout.phi[-1])


### Set up the objective ######################################################

def iota_floor(equilibrium_state, solver_context):
    """Hinge on the profile minimum of |iota|."""
    return jnp.maximum(
        IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)


def aspect_excess(equilibrium_state, solver_context):
    """Hinge on the aspect ratio above its limit."""
    return jnp.maximum(opt.aspect_ratio(equilibrium_state, solver_context) - ASPECT_LIMIT, 0.0)


def mirror_excess(equilibrium_state, solver_context):
    """Hinge on the mirror ratio, pulling from MIRROR_HINGE_FRACTION of the limit."""
    return jnp.maximum(
        opt.mirror_ratio(equilibrium_state, solver_context)
        - MIRROR_HINGE_FRACTION * MIRROR_LIMIT, 0.0)


qi = ConstructedQIResidual(SURFACES, **QI_OPTIONS)
electron_root = ElectronRoot(ROOT_RHO, plasma, phiedge, e_target=E_TARGET,
                             margin=E_MARGIN, e_max=E_MAX)
objective_function_terms = [
    (qi, 0.0, 10.0),
    (electron_root, 0.0, ROOT_WEIGHT),
    (aspect_excess, 0.0, 10.0),
    (iota_floor, 0.0, 10.0),
    (mirror_excess, 0.0, 1000.0),
]


def top_root_kv(equilibrium):
    """Largest stable root of each ROOT_RHO surface in kV (-dPhi/ds), by NTX."""
    return np.asarray(electron_root.top_root(
        electron_root.tables(equilibrium.state, equilibrium.runtime))) / 1.0e3


report = opt.EquilibriumReporter(
    ("constructed QI", qi.total, ".4e"),
    ("root hinge", lambda s, r: jnp.sum(electron_root(s, r)), ".4e"),
    ("aspect", opt.aspect_ratio, ".3f"), ("min |iota|", opt.min_abs_iota, ".3f"),
    ("mirror", opt.mirror_ratio, ".3f"))
monitor = opt.OptimizationMonitor()

### Run the optimization ######################################################

report("seed", seed)
print(f"seed top root at rho = {ROOT_RHO}: {top_root_kv(seed)} kV")
problem = opt.VmecProblem.from_tuples(
    inp, objective_function_terms, max_mode=max(MAX_MODES), use_ess=True,
    ess_alpha=ESS_ALPHA, progress=not ci_smoke, evaluation_progress=not ci_smoke)
monitor.problem = problem
x = problem.x0
costs = []
for max_mode, max_nfev in zip(MAX_MODES, MAX_NFEV):
    print(f"\n===== electron-root stage, max_mode = {max_mode} =====")
    stage = problem.subproblem(max_mode=max_mode, x=x)
    print(f"dof_names = {stage.dof_names}")
    cost_start = float(np.sum(stage.residual(stage.x0) ** 2))
    result = least_squares(
        stage.residual, stage.x0, jac=stage.residual_jac, x_scale=stage.scales,
        max_nfev=max_nfev, ftol=1e-6, xtol=1e-10, verbose=2, callback=monitor)
    costs += [cost_start, 2 * result.cost]
    x = stage.embed(result.x)
    inp = problem.input_from_x(x)
    equilibrium = problem.equilibrium_from_x(x)
    report(f"mode {max_mode}", equilibrium)
    print(f"top root at rho = {ROOT_RHO}: {top_root_kv(equilibrium)} kV")

### Check the result ##########################################################

final_input = replace(inp, ns_array=np.array([FINAL_NS]),
    ftol_array=np.array([FINAL_FTOL]), niter_array=np.array([FINAL_NITER]))
final_equilibrium = opt.solve_equilibrium(final_input, initial_state=equilibrium.solution,
    verbose=not ci_smoke, raise_on_max_iterations=True)

print(f"cost: initial {costs[0]:.6e} -> final {costs[-1]:.6e}")
report("final", final_equilibrium)

# Roots on the plotted surfaces, from D11 tables at a finer field axis:
diagnostic = ElectronRoot(ROOT_PLOT_RHO, plasma, phiedge, e_max=E_MAX, n_er=401)


def roots_by_surface(equilibrium):
    """Roots, validity, stability, the table and the residual curve per surface."""
    tables = diagnostic.tables(equilibrium.state, equilibrium.runtime)
    roots, valid, stable = (np.asarray(a) for a in diagnostic.roots(tables))
    curves = np.stack([np.asarray(diagnostic.residual(diagnostic.er_grid, tables, i))
                       for i in range(len(ROOT_PLOT_RHO))])
    return roots, valid, stable, curves


before, after = roots_by_surface(seed), roots_by_surface(final_equilibrium)
for name, (roots, valid, stable, _) in (("seed", before), ("final", after)):
    for i, rho in enumerate(ROOT_PLOT_RHO):
        found = [(round(float(r) / 1e3, 3), bool(s)) for r, v, s in
                 zip(roots[i], valid[i], stable[i]) if v]
        print(f"[{name}] rho = {rho:.2f}: roots (kV, stable) {found}")
assert costs[-1] < costs[0], "the optimization did not lower the objective"

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {input_path}\nWrote {wout_path}")
print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")
print(f"Wrote {monitor.plot(f'{OUTPUT_NAME}_objectives.png')}")

# Electric-field profile and the root diagram at ROOT_RHO[0]:
fig, (ax_p, ax_d) = plt.subplots(1, 2, figsize=(10, 4))
for (label, (roots, valid, stable, _)), color in zip(
        (("seed", before), ("optimized", after)), ("tab:gray", "tab:red")):
    for i, rho in enumerate(ROOT_PLOT_RHO):
        for r, v, s in zip(roots[i], valid[i], stable[i], strict=True):
            if v:
                ax_p.plot(rho, r / 1e3, "o" if s else "x", color=color,
                          label=label if label not in ax_p.get_legend_handles_labels()[1] else None)
ax_p.axhline(0.0, color="k", lw=0.5)
ax_p.set(xlabel=r"$\rho$", ylabel=r"ambipolar $-d\Phi/ds$ (kV)",
         title="roots by surface (o stable, x unstable)")
ax_p.legend()
j = int(np.argmin(np.abs(np.asarray(ROOT_PLOT_RHO) - ROOT_RHO[0])))
grid = np.asarray(diagnostic.er_grid) / 1e3
for (label, data), color in zip((("seed", before), ("optimized", after)), ("tab:gray", "tab:red")):
    curve = data[3][j]
    ax_d.plot(grid, curve / np.max(np.abs(curve)), color=color, label=label)
ax_d.axhline(0.0, color="k", lw=0.5)
ax_d.set(xlabel=r"$-d\Phi/ds$ (kV)", ylabel=r"$(\Gamma_i-\Gamma_e)/\max|\cdot|$",
         title=rf"root diagram, $\rho$ = {ROOT_PLOT_RHO[j]:.2f}")
ax_d.legend()
fig.tight_layout()
fig.savefig(f"{OUTPUT_NAME}_roots.png", dpi=150)
print(f"Wrote {OUTPUT_NAME}_roots.png")
for path in vj.plot_wout(wout_path, ".").values():
    print(f"Wrote {path}")
