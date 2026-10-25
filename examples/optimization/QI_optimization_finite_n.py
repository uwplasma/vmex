#!/usr/bin/env python
"""Stabilize a finite-n ideal-MHD instability at fixed physical targets, with AGNI.

The seed is the NFP = 4 quasi-isodynamic deck at twice its shipped pressure (beta 5.1 %,
``examples/finite_n_stability.py``): toroidal families 0 and 1 are unstable, family 2 is not.
The scalar that is minimized is

    sum_f softplus(S_f / GAMMA_SCALE) + 0.5 TARGET_WEIGHT sum_i ((c_i - c_i,seed) / tol_i)**2,

with ``S_f`` the sum of the TOP_K largest ``gamma2`` of family ``f`` (a cluster sum: the top two
modes of a family are within a few per cent and swap under small boundary steps) and ``c_i`` the
beta, aspect ratio, plasma volume (with the aspect ratio it fixes the major and minor radius),
mean and edge rotational transform, and a hinge on the constructed quasi-isodynamic residual.
PHIEDGE and the pressure profile are fixed by construction.

``agnimhd.growth_rate`` returns only the top eigenvalue, so the top-TOP_K eigenvectors come from
a host dense eigensolve of the restricted block that ``eigenpair`` also diagonalizes, and each
``gamma2`` is the Rayleigh quotient of its eigenvector (``growth_rate(..., v_fixed=...)``): the
Hellmann-Feynman value and gradient, exact at an eigenvector. An unresolved eigensolve returns
NaN, so VMEX rejects the trial instead of scoring it low.

Needs ``agnimhd`` (``pip install "vmex[agni]"``, with ``Basis(restrict=True)``); the run takes
hours. The growth rates are lower bounds that rise with the perturbation bandwidth, which the
optimization does not see.
"""

import os
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize

import agnimhd as ag
import vmex as vj
from agnimhd.assemble import matfree_operator, operator_dtype, represented_basis
from vmex import optimize as opt
from vmex.core.agni import make_agni_map
from vmex.core.qi import ConstructedQIResidual
from vmex.core.solver import prepare_runtime, resolution_from_input

# Seed deck with the pressure doubled, and the VMEC ladder (the last ns is the optimizer's):
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "input.nfp4_QI_finite_beta"
INPUT_OVERRIDES = dict(pres_scale=2.0)
NS_ARRAY, FTOL_ARRAY, NITER_ARRAY = [13, 25, 45], [1e-10, 1e-11, 1e-12], [5000, 10000, 40000]

# AGNI basis: nodes per direction (at least 2*MPOL+16 and 2*NTOR+14) and perturbation bandwidth:
N_RHO, N_THETA, N_ZETA, MPOL, NTOR = 10, 32, 28, 6, 5
AUTOMORPHISM = {"eps": 0.02, "x_0": 0.6, "m_1": 2.5, "m_2": 3.0}  # eps is the annular axis cutoff
GAMMA = 5 / 3
OPT_FAMILIES, CHECK_FAMILIES, TOP_K = [0, 1], [0, 1, 2], 2
RESIDUAL_MAX = 1e-9       # ||H y - lambda y|| above this makes an eigensolve unresolved
EIGSH_SIGMA = 1e-2        # shift of the final agnimhd.eigenpair checks; it must lie above gamma2

# Stability term softplus(S_f / GAMMA_SCALE), width SOFTNESS in those units. A tolerance of
# deviation costs 0.5 TARGET_WEIGHT; the stability term is about 1.4 at the seed. Weight 20
# made a 0.3 % shape step cost several times that and L-BFGS-B barely moved.
GAMMA_SCALE, SOFTNESS, TARGET_WEIGHT = 1e-3, 0.05, 2.0
TOLERANCES = dict(beta=0.01, aspect=0.01, volume=0.01, mean_iota=0.01, edge_iota=0.02)
QI_SURFACES = np.linspace(0.1, 0.9, 6)
QI_OPTIONS = dict(mboz=12, nboz=12, nphi=61, nalpha=18, n_bounce=21)
QI_TOLERANCE = 0.25       # a constructed-QI total above (1 + QI_TOLERANCE) x seed is penalized

# Boundary stages. The cold-solve objective is only piecewise smooth, so a late line search
# can spend a dozen solves for 1e-3: MAXFUN caps the evaluations per stage.
MAX_MODES, MAXITER, MAXFUN = [1, 2], [10, 20], [25, 60]
PARAMETER_STEP, MAX_PARAMETER_CHANGE, ESS_ALPHA, MINIMUM_MPOL = 0.01, 3.0, 1.2, 5

# Independent re-solve of the seed and the optimum at a finer radial grid:
FINAL_NS_ARRAY, FINAL_FTOL_ARRAY, FINAL_NITER_ARRAY = (
    [13, 25, 45, 91], [1e-10, 1e-11, 1e-12, 1e-13], [5000, 10000, 40000, 40000])
OUTPUT_DIR, OUTPUT_NAME = Path("output_QI_optimization_finite_n"), "QI_finite_n_optimized"

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs; the tiny basis
# does not resolve the physics.
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    NS_ARRAY, FTOL_ARRAY, NITER_ARRAY = [13], [1e-8], [2000]
    N_RHO, N_THETA, N_ZETA, MPOL, NTOR = 6, 14, 12, 2, 1
    OPT_FAMILIES, CHECK_FAMILIES = [0], [0]
    MAX_MODES, MAXITER, MAXFUN = [1], [2], [6]
    QI_SURFACES = np.array([0.25, 0.6, 0.9])
    QI_OPTIONS = dict(mboz=8, nboz=8, nphi=31, nalpha=7, n_bounce=7)
    FINAL_NS_ARRAY, FINAL_FTOL_ARRAY, FINAL_NITER_ARRAY = [13, 25], [1e-8, 1e-9], [2000, 4000]

###############################################################################
# End of input parameters.
###############################################################################

assert jnp.zeros(()).dtype == jnp.float64, "x64 is off; AGNI needs float64"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def with_ladder(source, ns, ftol, niter):
    return replace(source, ns_array=np.array(ns), ftol_array=np.array(ftol), niter_array=np.array(niter))


def scalars(state, runtime):
    return dict(beta=opt.volume_average_beta(state, runtime), aspect=opt.aspect_ratio(state, runtime),
                volume=opt.volume(state, runtime), mean_iota=opt.mean_iota(state, runtime),
                edge_iota=opt.edge_iota(state, runtime))


# Seed: its physical values are the targets
mpol = max(max(MAX_MODES) + 2, MINIMUM_MPOL)
seed_input = with_ladder(replace(vj.VmecInput.from_file(INPUT_FILE), **INPUT_OVERRIDES), NS_ARRAY, FTOL_ARRAY,
                         NITER_ARRAY).change_resolution(mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
NFP = int(seed_input.nfp)
equilibrium = opt.solve_equilibrium(seed_input, raise_on_max_iterations=True)
seed_targets = {k: float(v) for k, v in scalars(equilibrium.state, equilibrium.runtime).items()}
qi = ConstructedQIResidual(QI_SURFACES, **QI_OPTIONS)
seed_qi = float(qi.total_state(equilibrium.state, equilibrium.runtime))
print(f"seed beta {seed_targets['beta']:.4%}; targets " + ", ".join(f"{k} {v:.6g}" for k, v in seed_targets.items())
      + f", QI total {seed_qi:.4e}")

# Top-TOP_K growth rates: eigenvectors on the host, Rayleigh quotients in JAX
basis = ag.Basis(N_RHO, N_THETA, N_ZETA, radial="lobatto", automorphism=AUTOMORPHISM, mpol=MPOL, ntor=NTOR,
                 restrict=True)
assembly = ag.AssemblyConfig(gamma=GAMMA, parity=None)
nodes = basis.nodes_and_diffmat(NFP, 0)[0]
diffmat = {f: basis.nodes_and_diffmat(NFP, f)[1] for f in CHECK_FAMILIES}
runtime0 = prepare_runtime(seed_input, resolution_from_input(seed_input, ns=NS_ARRAY[-1]))
equilibrium_map = make_agni_map(runtime0, nodes, inp=seed_input)
agni_eq0 = equilibrium_map((equilibrium.solution, runtime0))
eq_treedef = jax.tree_util.tree_structure(agni_eq0)
n_keep = {f: int(matfree_operator(agni_eq0, diffmat[f], assembly)["n_keep"]) for f in OPT_FAMILIES}
dtype = {f: operator_dtype(assembly, diffmat[f]) for f in OPT_FAMILIES}


@jax.jit
def restricted_block(eq, dm):
    """The basis ``U`` of the represented space and the Hermitian block ``H = U^H A U``."""
    op = matfree_operator(eq, dm, assembly)
    U = represented_basis(op, dm)
    H = jnp.conj(U).T @ jax.lax.map(op["Ax"], U.T, batch_size=32).T
    return U, 0.5 * (H + jnp.conj(H).T)


def top_modes_host(family, *leaves):
    """gamma2 and AGNI whitened eigenvectors of the TOP_K most unstable modes; NaN when unresolved."""
    nan = (np.full(TOP_K, np.nan), np.full((TOP_K, n_keep[family]), np.nan, dtype=dtype[family]))
    try:
        U, H = (np.asarray(a) for a in restricted_block(
            jax.tree_util.tree_unflatten(eq_treedef, [jnp.asarray(a) for a in leaves]), diffmat[family]))
        lam, y = eigh(H, subset_by_index=[0, TOP_K - 1])
        residual = float(np.max(np.linalg.norm(H @ y - y * lam, axis=0)))
        out = (-lam, (U @ y).T.astype(dtype[family])) if residual <= RESIDUAL_MAX else nan
    except Exception as error:  # never treated as stable
        print(f"  eigensolve failed: {str(error)[:100]}")
        out = nan
    print(f"  family {family}: gamma2 {', '.join(f'{g:.4e}' for g in out[0])}"
          + ("" if np.all(np.isfinite(out[0])) else "  UNRESOLVED"), flush=True)
    return out


def cluster_growth(state, runtime):
    """Sum of the TOP_K largest gamma2 of each optimized family."""
    agni_eq = equilibrium_map((state, runtime))
    leaves = jax.tree_util.tree_leaves(jax.lax.stop_gradient(agni_eq))
    sums = []
    for family in OPT_FAMILIES:
        shapes = (jax.ShapeDtypeStruct((TOP_K,), jnp.float64), jax.ShapeDtypeStruct((TOP_K, n_keep[family]), dtype[family]))
        vectors = jax.pure_callback(lambda *a, f=family: top_modes_host(f, *a), shapes, *leaves)[1]
        gamma2 = [ag.growth_rate((agni_eq,), diffmat[family], assembly, v_fixed=vectors[j], equilibrium_map=lambda p: p[0])
                  for j in range(TOP_K)]
        sums.append(jnp.sum(jnp.stack(gamma2)))
    return jnp.stack(sums)


def loss(state, runtime):
    stability = SOFTNESS * jnp.logaddexp(0.0, cluster_growth(state, runtime) / GAMMA_SCALE / SOFTNESS)
    now = scalars(state, runtime)
    rows = [(now[k] - seed_targets[k]) / (TOLERANCES[k] * abs(seed_targets[k])) for k in TOLERANCES]
    qi_rise = jnp.maximum(qi.total_state(state, runtime) / seed_qi - 1.0, 0.0) / QI_TOLERANCE
    return jnp.sum(stability) + 0.5 * TARGET_WEIGHT * (jnp.sum(jnp.stack(rows) ** 2) + qi_rise**2)


# Optimization
monitor = opt.OptimizationMonitor()
# warm_start=None: warm-started re-solves are path dependent through the m = 1 gauge.
problem = opt.VmecProblem.from_loss(
    seed_input, loss, max_mode=max(MAX_MODES), use_ess=True, ess_alpha=ESS_ALPHA, warm_start=None,
    restart_from=equilibrium, progress=False, evaluation_progress=False)
monitor.problem = problem
x = problem.x0
for max_mode, maxiter, maxfun in zip(MAX_MODES, MAXITER, MAXFUN):
    print(f"\n===== finite-n stage, max_mode = {max_mode} =====")
    stage = problem.subproblem(max_mode=max_mode, x=x)
    x0, step = stage.x0, PARAMETER_STEP * stage.scales

    def value_and_gradient(y):
        value, gradient = stage.value_and_grad(x0 + step * y)
        return monitor.cache_evaluation(x0 + step * y, value, step * gradient)

    def record(intermediate_result):  # SciPy passes an OptimizeResult only for this parameter name
        monitor({"x": x0 + step * intermediate_result.x, "fun": intermediate_result.fun})

    initial_value = float(value_and_gradient(np.zeros_like(x0))[0])
    result = minimize(
        value_and_gradient, np.zeros_like(x0), jac=True, method="L-BFGS-B",
        bounds=[(-MAX_PARAMETER_CHANGE, MAX_PARAMETER_CHANGE)] * x0.size, callback=record,
        options={"maxiter": maxiter, "maxfun": maxfun, "gtol": 1e-6, "ftol": 1e-12, "maxls": 20, "maxcor": 20})
    print(f"optimizer scalar cost: {initial_value:.6e} -> {float(result.fun):.6e}")
    x = stage.embed(x0 + step * result.x)
print(f"rejected trials: {problem.metadata['holder']['failed_trials']}")

# Check: cold re-solve at the finer grid, fresh eigensolves of every family
final_ladder = dict(ns=FINAL_NS_ARRAY, ftol=FINAL_FTOL_ARRAY, niter=FINAL_NITER_ARRAY)
growth, checked = {}, {}
for label, source in (("seed", seed_input), ("final", problem.input_from_x(x))):
    source = with_ladder(source, **final_ladder)
    result = vj.solve_multigrid(source, verbose=False)
    assert result.converged, f"{label}: force balance did not converge"
    runtime = prepare_runtime(source, resolution_from_input(source, ns=FINAL_NS_ARRAY[-1]))
    checked[label] = {k: float(v) for k, v in scalars(result.state, runtime).items()}
    checked[label]["QI total"] = float(qi.total_state(result.state, runtime))
    agni_eq = make_agni_map(runtime, nodes, inp=source)((result.state, runtime))
    growth[label] = {}
    for family in CHECK_FAMILIES:
        try:  # fresh, no warm vector; a failed or unconverged solve is NaN, never stable
            g2, _, rel = ag.eigenpair(agni_eq, diffmat[family], assembly,
                                      ag.SolverConfig(eigensolver="eigsh", sigma=EIGSH_SIGMA, eigsh_tol=1e-12))
            growth[label][family] = float(g2) if float(rel) * abs(float(g2)) <= 1e-7 else np.nan
        except Exception as error:
            print(f"family {family}: eigensolve failed: {str(error)[:100]}")
            growth[label][family] = np.nan
    final_source = source

print(f"\nns {FINAL_NS_ARRAY[-1]} re-solve, gamma2 at ({MPOL},{NTOR}) {N_THETA}x{N_ZETA} (NaN: unresolved):")
for family in CHECK_FAMILIES:
    print(f"  family {family}: seed {growth['seed'][family]:.4e}  final {growth['final'][family]:.4e}")
print("targets (deviation from the seed, tolerance):")
for k in [*TOLERANCES, "QI total"]:
    change = checked["final"][k] / checked["seed"][k] - 1.0
    print(f"  {k:10s} {checked['seed'][k]:.6g} -> {checked['final'][k]:.6g}  {change:+.3%}  "
          f"({TOLERANCES.get(k, QI_TOLERANCE):.1%}{', hinge' if k == 'QI total' else ''})")

print(f"Wrote {final_source.to_indata(OUTPUT_DIR / f'input.{OUTPUT_NAME}')}")
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for k, (label, color) in enumerate((("seed", "tab:red"), ("final", "tab:blue"))):
    axes[0].bar(np.arange(len(CHECK_FAMILIES)) + (k - 0.5) * 0.4, [growth[label][f] for f in CHECK_FAMILIES], 0.4,
                color=color, label=label)
axes[0].set(xticks=range(len(CHECK_FAMILIES)), xticklabels=CHECK_FAMILIES, xlabel="toroidal family",
            ylabel=f"gamma2 at ({MPOL},{NTOR})")
axes[0].set_yscale("symlog", linthresh=1e-6)
axes[0].legend()
for name, values in monitor.history.items():
    axes[1].semilogy([r.iteration for r in monitor.records], np.maximum(values, 1e-12), "o-", ms=3, label=name)
axes[1].set(xlabel="accepted iteration", ylabel="objective")
axes[1].legend()
fig.savefig(OUTPUT_DIR / f"{OUTPUT_NAME}.png", dpi=150)
print(f"Wrote {OUTPUT_DIR / f'{OUTPUT_NAME}.png'}")
