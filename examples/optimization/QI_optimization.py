#!/usr/bin/env python
"""Quasi-isodynamic (QI) optimization from a circular torus, nfp=1.

Two-stage campaign, one terms-list swap — the "QP first, then QI" route:

1. **QP basin** (implicit gradients): drive the quasisymmetry ratio residual
   with helicity (m, n) = (0, 1) plus aspect / iota-floor / mirror targets.
   This forms poloidally closed ``|B|`` contours — the topological
   prerequisite of omnigenity — from the crude circular seed.
2. **QI refinement** (finite differences): swap the QP term for the
   Goodman-style quasi-isodynamic residual
   (:func:`vmec_jax.optimize.quasi_isodynamic_residual_from_wout` — bounce
   width variance, branch widths, profile consistency, branch-shuffle), which
   aligns the bounce distances of the trapped-particle wells across field
   lines.  The Boozer transform runs on host (booz_xform_jax), so this stage
   uses ``jac=None``.

Honest comparison with the legacy seed machinery: the old
``QI_optimization_nfp1.py`` prepared its start with a bespoke seed
preconditioner (``prepare_simple_omnigenity_seed_input`` + target-helicity
sign seeding + a staged runner, ~1.5k lines of support code, since removed
from the tree, so a side-by-side rerun is no longer possible).  The staged
route below replaces all of it: stage 1 alone takes the raw circular torus
into a QP basin, and stage 2 refines omnigenity from there — the achieved
values quoted above come from exactly this script with no seed machinery.

Expected runtime: multi-hour on the office 36-core CPU at the default
budget (the stage-2 finite differences dominate; the implicit QP stages
are CPU-pinned).  Achieved (default budget, office 36-core CPU,
2026-07-11, this script as-is): the implicit QP stages take the QI residual
total 2.430 (circular nfp1 seed) -> ~0.64, and the Boozer QI stage then
refines it to QI total 2.139e-02 (QP total 1.38e-02, |mean iota| 0.15) --
a > 2-order improvement.  This is a strong QP->QI result but *not* precise
QI: the aspect ratio drifts to ~10.9 at the low aspect weight, and the
omnigenity residual plateaus well above the ~1e-3 precise bar.  Requires
``pip install booz_xform_jax``.

Honest caveat (2026-07-11): QI is the class most dependent on the quality
of the QP basin and on the omnigenity residual.  Reaching a *good* QP
basin is the prerequisite and it is basin-sensitive -- stage 1 must use
the implicit path (finite differences land in a much worse basin; cf.
``QP_optimization.py``), and different CPU runs land in different basins
(QI total 2.1e-2 to 6.6e-1 observed across runs).  Precise QI is *not*
reached by the current 4-term Goodman residual (bounce-width variance,
branch widths, profile consistency, branch shuffle); it would need a
richer omnigenity formulation -- a proper Omnigenity/OmnigenousField
target plus an EffectiveRipple (eps_eff) neoclassical metric (plan.md
R17.7) -- which is not yet implemented here.  Do not expect precise QI
from this script as-is.
"""

import os
from pathlib import Path

import numpy as np
import jax.numpy as jnp

import vmec_jax as vj
from vmec_jax import optimize as opt

# --------------------------- parameters ------------------------------------
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "input.minimal_seed_nfp1"
OUT_DIR = Path("output_QI_optimization")
SURFACES = np.linspace(0.1, 1.0, 6)        # QP and QI surfaces
ASPECT_TARGET = 6.0
IOTA_FLOOR = 0.15
MIRROR_TARGET = 0.20
QP_SCHEDULE = (1, 2, 3)                    # stage 1 (implicit gradients)
QI_SCHEDULE = (3,)                         # stage 2 (finite differences)
QP_NFEV, QI_NFEV = 2000, 1000              # trial budgets per stage
FTOL = 1e-6                                # per-stage convergence tolerance
if os.environ.get("VMEC_JAX_EXAMPLES_CI") == "1":  # smoke-test budget
    QP_SCHEDULE, QI_SCHEDULE, QP_NFEV, QI_NFEV = (1,), (1,), 6, 3
    FTOL = 1e-4

# --------------------------- seed equilibrium -------------------------------
inp = vj.VmecInput.from_file(INPUT_FILE)
eq = opt.solve_equilibrium(inp)
qp = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=0, helicity_n=1)


def qi_residuals(eq):
    """Goodman-style QI residual vector of a converged equilibrium."""
    out = opt.quasi_isodynamic_residual_from_wout(eq, surfaces=SURFACES)
    return out["residuals1d"]


def iota_shortfall(state, rt):
    return jnp.maximum(IOTA_FLOOR - jnp.abs(opt.mean_iota(state, rt)), 0.0)


def report(tag, eq):
    qi_total = float(np.sum(np.asarray(qi_residuals(eq)) ** 2))
    print(f"[{tag}] QI total = {qi_total:.6e}, QP total = {float(qp.total(eq)):.6e}, "
          f"aspect = {float(opt.aspect_ratio(eq.state, eq.runtime)):.4f}, "
          f"mean iota = {float(opt.mean_iota(eq.state, eq.runtime)):.4f}")
    return qi_total


qi_seed = report("seed", eq)

# ------------------- objectives: one terms-list swap ------------------------
practical_terms = [
    (opt.aspect_ratio, ASPECT_TARGET, 0.25),
    (iota_shortfall, 0.0, 100.0),
    (opt.mirror_ratio, MIRROR_TARGET, 10.0),
]
qp_terms = [(qp, 0.0, 1.0)] + practical_terms             # stage 1
qi_terms = [(qi_residuals, 0.0, 10.0)] + practical_terms  # stage 2

# --------------------------- stage 1: QP basin ------------------------------
for max_mode in QP_SCHEDULE:
    print(f"\n===== QP stage, max_mode = {max_mode} =====")
    result = opt.least_squares(
        qp_terms, inp, max_mode=max_mode, jac="implicit",
        use_ess=True, verbose=1, max_nfev=QP_NFEV, ftol=FTOL, xtol=1e-10,
    )
    inp = result.input
    if result.equilibrium is not None:
        report(f"QP stage {max_mode}", result.equilibrium)

# --------------------------- stage 2: QI refinement -------------------------
# The Boozer-based QI residual is a host (wout-engine) objective, so this
# stage differentiates by finite differences (jac=None, hot-restarted trials).
for max_mode in QI_SCHEDULE:
    print(f"\n===== QI stage, max_mode = {max_mode} =====")
    result = opt.least_squares(
        qi_terms, inp, max_mode=max_mode, jac=None,
        use_ess=True, verbose=1, max_nfev=QI_NFEV, ftol=FTOL, xtol=1e-10, diff_step=1e-4,
    )
    inp = result.input
    if result.equilibrium is not None:
        report(f"QI stage {max_mode}", result.equilibrium)

# --------------------------- final results ---------------------------------
eq = result.equilibrium or opt.solve_equilibrium(inp)
qi_final = report("final", eq)
print(f"\nQI total: seed {qi_seed:.3e} -> final {qi_final:.3e}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
inp.to_indata(OUT_DIR / "input.QI_optimized")
wout_path = vj.write_wout(OUT_DIR / "wout_QI_optimized.nc", eq.wout)
print(f"wrote {OUT_DIR / 'input.QI_optimized'}\nwrote {wout_path}")
for key, path in vj.plot_wout(wout_path, OUT_DIR).items():
    print(f"wrote {path}")
