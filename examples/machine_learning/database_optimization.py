"""Surrogate-assisted design: learn from a small database of equilibria, then verify.

Samples the six boundary coefficients of a vacuum QA seed on a Sobol grid, solves
each equilibrium with VMEX, and fits a ridge-regularized quadratic surrogate to
four labels (mean iota, aspect ratio, QA residual, magnetic well) with a held-out
test split. Each active step minimizes the surrogate inside the sampling box,
re-solves that boundary with VMEX, and adds it to the training set. The reported
design is the best verified boundary, never a surrogate prediction.

The database is deliberately tiny: it teaches the workflow, not a general model.
Only the two target labels drive the search; the other two show what it costs.
"""

import os
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

import vmex as vj
from vmex import optimize as opt
from vmex.core.plotting import surface_rz

INPUT_FILE = Path(__file__).resolve().parents[1] / "data/input.LandremanPaul2021_QA_lowres"
OUTPUT_DIR = Path("results/machine_learning")
DOF_NAMES = ["RBC(0,1)", "ZBS(0,1)", "RBC(1,0)", "ZBS(1,0)", "RBC(1,1)", "ZBS(1,1)"]
MAX_MODE = 1                    # boundary modes |m|,|n| <= MAX_MODE are packed
STEP = 0.05                     # coefficient = seed + STEP * x, with x in [-1, 1]
NS, MPOL, NTOR = 21, 5, 5
SOBOL_POWER = 6                 # 2**6 samples: 49 training rows for 28 features
ACTIVE_STEPS = 3                # surrogate proposals, each verified by a true solve
RIDGE, SEED = 1e-3, 42
SURFACES = (0.25, 0.5, 0.75)    # normalized toroidal flux for the QA residual
IOTA_TARGET, IOTA_SCALE = 0.45, 0.05
ASPECT_TARGET, ASPECT_SCALE = 5.0, 0.5
LABELS = ("mean_iota", "aspect_ratio", "qa_residual", "magnetic_well")

if os.environ.get("VMEX_EXAMPLES_CI") == "1":     # smoke pass: fewer, coarser solves
    SOBOL_POWER, ACTIVE_STEPS, MPOL, NTOR = 4, 1, 3, 3

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
source = vj.VmecInput.from_file(INPUT_FILE)
seed = replace(
    source, lfreeb=False, pres_scale=0.0, curtor=0.0, ncurr=1, ac=np.zeros_like(source.ac),
    ns_array=np.array([NS]), ftol_array=np.array([1e-12]), niter_array=np.array([5000]), delt=0.5,
).change_resolution(mpol=MPOL, ntor=NTOR, ntheta=2 * MPOL + 6, nzeta=2 * NTOR + 4)
packed_seed = opt.pack_boundary(seed, MAX_MODE)
active = [opt.boundary_dof_names(seed, MAX_MODE).index(name) for name in DOF_NAMES]
qa = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
quantities = (opt.mean_iota, opt.aspect_ratio, qa.total_state, opt.magnetic_well)


def input_from_x(x):
    """Boundary input with the named coefficients moved; all other modes stay fixed."""
    packed = packed_seed.copy()
    packed[active] += STEP * np.asarray(x)
    return opt.unpack_boundary(seed, packed, MAX_MODE)


def measure(x):
    """Solve the equilibrium at x and return the four labels (NaN if the solve fails)."""
    try:
        eq = opt.solve_equilibrium(input_from_x(x), raise_on_max_iterations=True)
        return np.array([float(q(eq.solution, eq.solver_context)) for q in quantities])
    except (vj.VmecError, RuntimeError, ValueError, FloatingPointError):
        return np.full(len(quantities), np.nan)


def design_loss(iota, aspect):
    """Dimensionless distance to the iota and aspect-ratio targets."""
    return 0.5 * ((iota - IOTA_TARGET) / IOTA_SCALE) ** 2 + 0.5 * ((aspect - ASPECT_TARGET) / ASPECT_SCALE) ** 2


def features(x):
    """Quadratic response surface: constant, linear and unique pairwise products."""
    x = np.atleast_2d(x)
    pairs = [x[:, i] * x[:, j] for i in range(x.shape[1]) for j in range(i, x.shape[1])]
    return np.column_stack((np.ones(len(x)), x, *pairs))


def fit():
    """Ridge fit of all labels on the finite training rows (the intercept is unpenalized)."""
    ok = train & np.isfinite(Y).all(axis=1)
    A = features(X[ok])
    penalty = RIDGE * np.eye(A.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.solve(A.T @ A + penalty, A.T @ Y[ok])


def surrogate_loss(x):
    iota, aspect = (features(x) @ weights)[:, :2].T
    return design_loss(iota, aspect)


# Database: the seed (x = 0) plus a scrambled Sobol sample; a quarter of it is held out.
X = np.vstack((np.zeros(len(DOF_NAMES)),
               2 * qmc.Sobol(len(DOF_NAMES), seed=SEED).random_base2(SOBOL_POWER) - 1))
Y = np.array([measure(x) for x in X])
train = np.ones(len(X), bool)
train[1 + np.random.default_rng(SEED).permutation(len(X) - 1)[:(len(X) - 1) // 4]] = False
print(f"{len(X)} samples, {np.isfinite(Y).all(axis=1).sum()} solved")
weights = fit()
test = np.flatnonzero(~train & np.isfinite(Y).all(axis=1))
rmse = np.sqrt(np.mean((features(X[test]) @ weights - Y[test]) ** 2, axis=0))
baseline = np.sqrt(np.mean((Y[train & np.isfinite(Y).all(axis=1)].mean(axis=0) - Y[test]) ** 2, axis=0))
for name, error, reference in zip(LABELS, rmse, baseline):
    print(f"{name}: held-out RMSE {error:.3g} (training-mean baseline {reference:.3g})")
predictions = features(X[test]) @ weights

# Active steps: screen a Sobol pool, polish the best points inside the box, verify the winner.
for step in range(ACTIVE_STEPS):
    pool = 2 * qmc.Sobol(len(DOF_NAMES), seed=SEED + 1 + step).random_base2(11) - 1
    starts = pool[np.argsort(surrogate_loss(pool))[:8]]
    polished = [minimize(lambda x: surrogate_loss(x)[0], s, method="L-BFGS-B",
                         bounds=[(-1, 1)] * len(DOF_NAMES)).x for s in starts]
    x = min(polished, key=lambda s: surrogate_loss(s)[0])
    y = measure(x)
    print(f"active {step + 1}/{ACTIVE_STEPS}: predicted loss {surrogate_loss(x)[0]:.4g}, "
          f"verified loss {design_loss(*y[:2]):.4g}")
    X, Y, train = np.vstack((X, x)), np.vstack((Y, y)), np.append(train, True)
    weights = fit()

losses = design_loss(Y[:, 0], Y[:, 1])               # NaN where a solve failed
proposals = losses[len(X) - ACTIVE_STEPS:]
best = len(X) - ACTIVE_STEPS + int(np.argmin(np.where(np.isfinite(proposals), proposals, np.inf)))
selected = X[best] if losses[best] < losses[0] else X[0]
print(f"Verified loss: seed {losses[0]:.4g}, selected {min(losses[best], losses[0]):.4g}")
input_from_x(selected).to_indata(OUTPUT_DIR / "input.database_selected")
np.savez_compressed(OUTPUT_DIR / "database.npz", X=X, Y=Y, train=train, labels=LABELS,
                    weights=weights, rmse=rmse, baseline=baseline, selected=selected)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
axes[0].scatter(Y[test, 1], predictions[:, 1])
limits = [Y[test, 1].min(), Y[test, 1].max()]
axes[0].plot(limits, limits, "--", color="0.6")
axes[0].set(xlabel="Held-out aspect ratio", ylabel="Surrogate", title="Held-out test split")
axes[1].semilogy(np.minimum.accumulate(proposals), "o-")
axes[1].set(xlabel="Active step", ylabel="Best verified loss", title="Verified proposals")
for x, style, label in ((X[0], "-", "Seed"), (selected, "--", "Selected")):
    wout = opt.solve_equilibrium(input_from_x(x), raise_on_max_iterations=True).wout
    r, z = surface_rz(wout, s_index=wout.ns - 1, theta=np.linspace(0, 2 * np.pi, 200),
                      phi=np.array([0.18 * 2 * np.pi / wout.nfp]))
    axes[2].plot(r[:, 0], z[:, 0], style, label=label)
axes[2].set(xlabel="R [m]", ylabel="Z [m]", aspect="equal", title="Boundary at 0.18 field periods")
axes[2].legend(frameon=False)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "database.png", dpi=150)
print(f"Wrote {OUTPUT_DIR.resolve()}")
