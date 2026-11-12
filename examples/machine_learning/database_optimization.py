#!/usr/bin/env python
"""Learn a local vacuum-equilibrium surrogate and verify its proposed boundary.

Run from any directory. The tiny database demonstrates plumbing and held-out
error, not a generally useful stellarator model. Every label is dimensionless.
"""

import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

import jax
import vmex as vj
from vmex import optimize as opt
from vmex.core.qi import ConstructedQIResidual
from vmex.core.pwo import MappedPwOResidual

# All user choices live here. Increase sampling and resolve the equilibrium
# and diagnostics before interpreting small improvements scientifically.
INPUT_FILE = Path(__file__).resolve().parents[1] / "data/input.LandremanPaul2021_QA_lowres"
OUTPUT_DIR = Path("results/machine_learning")
SOBOL_POWER = 6                       # 65 solves; 49 training rows for 28 features
RANDOM_SEED = 42
STEP_METRES = 0.003                   # x in [-1,1], coefficient = seed + step*x
DOF_NAMES = ["RBC(0,1)", "ZBS(0,1)", "RBC(1,0)", "ZBS(1,0)", "RBC(1,1)", "ZBS(1,1)"]
MAX_MODE = 1                         # raise when adding higher-order coefficients
NS, MPOL, NTOR = 21, 5, 5
FTOL, MAX_ITERATIONS = 1e-12, 5000
SURFACES = (0.25, 0.5, 0.75)           # normalized toroidal flux s, not radius
RIDGE = 1e-3
N_CANDIDATES = 4096
IOTA_TARGET, IOTA_SCALE = 0.45, 0.05
ASPECT_TARGET, ASPECT_SCALE = 6.0, 0.5

# All eight requested quantities are exposed. Optional labels add compilation
# and per-sample cost: QI/pwO use Boozer transforms, Gamma_c bounce quadrature,
# epsilon_eff^(3/2) uses optional NEO_JAX (pip install 'vmex[neoclassical]').
COMPUTE_QI = False
COMPUTE_GAMMA_C = False
COMPUTE_PWO = False
COMPUTE_EPSILON = False
BOOZER = dict(mboz=8, nboz=8, oversample=1)
NEO_BOOZER = dict(mboz=16, nboz=12)
QI_GRID = dict(nphi=41, nalpha=12, n_bounce=13)
GC_GRID = dict(nalpha=5, num_transit=3, points_per_transit=64,
               num_pitch=24, quadrature_order=16, max_wells=128)
PWO_GRID = dict(ngamma=16, neta=16, a=0.3, b=0.0, c=-2.0, eta_min=2.8)
NEO_GRID = dict(theta_n=32, phi_n=32, npart=24, multra=1, no_bins=50,
                nstep_per=40, nstep_min=200, nstep_max=500, acc_req=0.02,
                max_rational_field_periods=100000)
LABEL_NAMES = ("mean_iota", "aspect_ratio", "qs_qa_total", "magnetic_well",
               "constructed_qi_total", "gamma_c_mean", "mapped_pwo_total",
               "epsilon_eff_3_2_mean")

###############################################################################
# One parameterization, one resolution, zero pressure and zero toroidal current.
###############################################################################

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
source = vj.VmecInput.from_file(INPUT_FILE).change_resolution(mpol=MPOL, ntor=NTOR, ntheta=2*MPOL+6, nzeta=2*NTOR+4)
seed = replace(source, ns_array=np.array([NS]), delt=0.5,
               ftol_array=np.array([FTOL]), niter_array=np.array([MAX_ITERATIONS]),
               lfreeb=False, pres_scale=0.0, ncurr=1, curtor=0.0, ac=np.zeros_like(source.ac))
seed_path = seed.to_indata(OUTPUT_DIR / "input.database_seed")
actual_surfaces = {}
qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
quantities = [opt.mean_iota, opt.aspect_ratio, qs.total_state, opt.magnetic_well]
# Explicit constructors make each optional diagnostic easy to find and change.
optional = {
    4: ConstructedQIResidual(SURFACES, **BOOZER, **QI_GRID) if COMPUTE_QI else None,
    5: opt.GammaC(SURFACES, **GC_GRID) if COMPUTE_GAMMA_C else None,
    6: MappedPwOResidual(SURFACES, **BOOZER, **PWO_GRID) if COMPUTE_PWO else None,
}
packed_seed = opt.pack_boundary(seed, MAX_MODE)
names = opt.boundary_dof_names(seed, MAX_MODE)
active = [names.index(name) for name in DOF_NAMES]


def input_from_x(x):
    """Use VMEX's public coefficient mapping, holding all other modes fixed."""
    packed = packed_seed.copy()
    packed[active] += STEP_METRES*np.asarray(x)
    return opt.unpack_boundary(seed, packed, MAX_MODE)


def measure(x):
    """Return labels, per-label errors, force residual and elapsed seconds."""
    values, errors = np.full(len(LABEL_NAMES), np.nan), ["disabled"] * len(LABEL_NAMES)
    started = perf_counter()
    try:
        eq = opt.solve_equilibrium(input_from_x(x), raise_on_max_iterations=True)
        fsq = float(eq.result.fsqr + eq.result.fsqz + eq.result.fsql)
        if not eq.result.converged or not np.isfinite(fsq):
            raise RuntimeError(f"unconverged equilibrium, FSQ={fsq}")
    except (vj.VmecError, RuntimeError, ValueError, FloatingPointError) as exc:
        return values, [f"solve: {type(exc).__name__}: {exc}"] * len(values), np.nan, perf_counter()-started
    for j, quantity in enumerate(quantities):
        try:
            values[j] = float(quantity(eq.solution, eq.solver_context))
            if not np.isfinite(values[j]):
                raise ValueError("non-finite diagnostic")
            errors[j] = ""
        except (vj.VmecError, RuntimeError, ValueError, FloatingPointError) as exc:
            values[j], errors[j] = np.nan, f"{type(exc).__name__}: {exc}"
    for j, term in optional.items():
        if term is None:
            continue
        try:
            out = term.compute_state(eq.state, eq.runtime)
            if j == 5 and np.any(np.asarray(out["overflow_fraction"]) != 0):
                raise ValueError("Gamma_c well-slot overflow; increase max_wells")
            values[j] = float(np.mean(out["gamma_c"]) if j == 5 else out["total"])
            actual_surfaces[LABEL_NAMES[j]] = np.asarray(out["s" if j == 5 else "s_b"]).tolist()
            if not np.isfinite(values[j]):
                raise ValueError("non-finite diagnostic")
            errors[j] = ""
        except (vj.VmecError, RuntimeError, ValueError, FloatingPointError) as exc:
            values[j], errors[j] = np.nan, f"{type(exc).__name__}: {exc}"
    if COMPUTE_EPSILON:
        try:
            from vmex.core.neoclassical import diagnostic_neo_config
            actual_s, ripple = vj.epsilon_effective_from_wout(
                eq.wout, surfaces=SURFACES, **NEO_BOOZER,
                config=replace(diagnostic_neo_config(), **NEO_GRID))
            actual_surfaces[LABEL_NAMES[-1]] = np.asarray(actual_s).tolist()
            ripple = np.asarray(ripple)
            if (not ripple.size or ripple.shape != np.shape(actual_s)
                    or not np.all(np.isfinite(ripple) & (ripple >= 0))):
                raise ValueError("NEO diagnostic must be finite, nonnegative, and aligned with s")
            values[-1], errors[-1] = float(np.mean(ripple)), ""
        except (vj.VmecError, ImportError, RuntimeError, ValueError, FloatingPointError) as exc:
            errors[-1] = f"{type(exc).__name__}: {exc}"
    return values, errors, fsq, perf_counter()-started


def plot_database(labels, predictions, seed_loss, proposal_loss, path):
    """Plot held-out predictions and the independently verified proposal."""
    test = np.isfinite(predictions[:, 1])
    prediction = predictions[test, 1]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    axes[0].scatter(labels[test, 1], prediction, color="#2563eb")
    limits = [min(labels[test, 1].min(), prediction.min()), max(labels[test, 1].max(), prediction.max())]
    axes[0].plot(limits, limits, color="0.6", ls="--")
    axes[0].set(xlabel="True held-out aspect ratio", ylabel="Surrogate prediction", title="Untouched test split")
    axes[1].bar(["Seed", "Verified proposal"], [seed_loss, proposal_loss], color=["#94a3b8", "#2563eb"])
    axes[1].set(ylabel="True target loss", title="One proposed design, re-solved")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)

X = np.vstack((np.zeros(len(DOF_NAMES)), 2*qmc.Sobol(len(DOF_NAMES), scramble=True,
               seed=RANDOM_SEED).random_base2(SOBOL_POWER)-1))
measured = []
for i, x in enumerate(X):
    measured.append(measure(x))
    print(f"sample {i+1}/{len(X)}: {np.isfinite(measured[-1][0]).sum()} labels, "
          f"FSQ={measured[-1][2]:.2e}, {measured[-1][3]:.2f}s", flush=True)
Y = np.array([row[0] for row in measured])
errors = np.array([row[1] for row in measured])
valid = np.isfinite(Y)
split = np.full(len(X), "train", dtype="U5")
split[1+np.random.default_rng(RANDOM_SEED).permutation(len(X)-1)[:max(1, (len(X)-1)//4)]] = "test"
train = split == "train"
provenance = dict(seed_input=seed_path.read_text(), source_input=str(INPUT_FILE),
                  vmex_version=vj.__version__, jax_version=jax.__version__,
                  devices=[str(device) for device in jax.devices()],
                  dof_names=DOF_NAMES, max_mode=MAX_MODE, step_metres=STEP_METRES,
                  surfaces_requested=SURFACES, surfaces_actual=actual_surfaces,
                  ns=NS, mpol=MPOL, ntor=NTOR, ftol=FTOL, max_iterations=MAX_ITERATIONS,
                  random_seed=RANDOM_SEED, sobol_power=SOBOL_POWER, ridge=RIDGE, n_candidates=N_CANDIDATES,
                  boozer=BOOZER, qi=QI_GRID, gamma_c=GC_GRID, pwo=PWO_GRID,
                  neo_boozer=NEO_BOOZER, neo=NEO_GRID,
                  target_iota=IOTA_TARGET, iota_scale=IOTA_SCALE,
                  target_aspect=ASPECT_TARGET, aspect_scale=ASPECT_SCALE,
                  optional_labels=[COMPUTE_QI, COMPUTE_GAMMA_C, COMPUTE_PWO, COMPUTE_EPSILON],
                  label_convention="QS/QI/pwO squared residual totals; hard Gamma_c mean; NEO epsilon_eff**1.5 mean",
                  radial_convention="QI/pwO/NEO nearest half mesh; Gamma_c nearest interior full mesh; QS interpolated half mesh")


def save_database(**extra):
    """Save numeric arrays and JSON metadata without requiring pickle to reload."""
    np.savez_compressed(OUTPUT_DIR / "database.npz", X=X, Y=Y, valid=valid, errors=errors,
                        label_names=np.array(LABEL_NAMES), split=split, provenance=json.dumps(provenance),
                        fsq=np.array([r[2] for r in measured]), seconds=np.array([r[3] for r in measured]), **extra)


save_database()                        # Preserve failures even if fitting cannot start.


def features(x):
    """Quadratic response surface: constant, linear, and unique products."""
    z = (np.atleast_2d(x)-x_mean)/x_scale
    return np.column_stack((np.ones(len(z)), z,
                            *[z[:, i]*z[:, j] for i in range(z.shape[1]) for j in range(i, z.shape[1])]))


# Both feature scaling and target scaling use the training split exclusively.
x_mean, x_scale = X[train].mean(axis=0), X[train].std(axis=0)
models, rmse, baseline_rmse = {}, np.full(Y.shape[1], np.nan), np.full(Y.shape[1], np.nan)
test_predictions = np.full_like(Y, np.nan)
for j, name in enumerate(LABEL_NAMES):
    keep, test = train & valid[:, j], ~train & valid[:, j]
    if keep.sum() < 3 or not test.any():
        continue
    center, scale = Y[keep, j].mean(), max(Y[keep, j].std(), 1e-12)
    A = features(X[keep])
    penalty = RIDGE*np.eye(A.shape[1]); penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(A.T@A+penalty, A.T@((Y[keep, j]-center)/scale))
    models[j] = (center, scale, coefficients)
    prediction = center + scale*(features(X[test])@coefficients)
    test_predictions[test, j] = prediction
    rmse[j] = np.sqrt(np.mean((prediction-Y[test, j])**2))
    baseline_rmse[j] = np.sqrt(np.mean((center-Y[test, j])**2))
    print(f"{name}: held-out RMSE {rmse[j]:.3g}; training-mean baseline {baseline_rmse[j]:.3g}")

if not {0, 1}.issubset(models) or not valid[0, :2].all():
    raise RuntimeError("Seed/core labels failed; inspect the solve and diagnostic errors")
candidates = np.vstack((np.zeros(len(DOF_NAMES)), 2*qmc.Sobol(len(DOF_NAMES), scramble=True,
                        seed=RANDOM_SEED+1).random_base2(int(np.ceil(np.log2(N_CANDIDATES))))[:N_CANDIDATES]-1))
predicted = {j: center+scale*(features(candidates)@coef) for j, (center, scale, coef) in models.items()}


def design_loss(iota, aspect):
    """Score the same physical targets as the differentiable example."""
    return 0.5*((iota-IOTA_TARGET)/IOTA_SCALE)**2 + 0.5*((aspect-ASPECT_TARGET)/ASPECT_SCALE)**2


predicted_loss = design_loss(predicted[0], predicted[1])
candidate_index = int(np.argmin(predicted_loss))
candidate = candidates[candidate_index]
proposal, proposal_errors, proposal_fsq, _ = measure(candidate)
seed_loss, proposal_loss = design_loss(Y[0, 0], Y[0, 1]), design_loss(proposal[0], proposal[1])
improved = np.isfinite(proposal_loss) and proposal_loss < seed_loss
best_x = candidate if improved else np.zeros(len(DOF_NAMES))
input_from_x(best_x).to_indata(OUTPUT_DIR / "input.database_selected")
print(f"True-solve target loss: seed {seed_loss:.6g}, proposal {proposal_loss:.6g}; "
      f"selected {'proposal' if improved else 'seed'}. Improvement is not guaranteed.")

save_database(rmse=rmse, baseline_rmse=baseline_rmse, test_predictions=test_predictions, proposal_x=candidate,
                    proposal_y=proposal, proposal_errors=np.array(proposal_errors), proposal_fsq=proposal_fsq,
                    selected_x=best_x, seed_loss=seed_loss, proposal_loss=proposal_loss,
                    predicted_proposal_loss=predicted_loss[candidate_index])
plot_database(Y, test_predictions, seed_loss, proposal_loss, OUTPUT_DIR / "database.png")
print(f"Saved {OUTPUT_DIR.resolve()}")
