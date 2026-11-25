#!/usr/bin/env python
"""Global finite-n ideal-MHD stability of a VMEX equilibrium, computed with AGNI.

Solve a fixed-boundary equilibrium, evaluate it once on the AGNI node set
(``vmex.core.agni.make_agni_map``), then find the most unstable displacement of
every independent toroidal-mode family. ``gamma2`` is AGNI's normalized squared
growth rate; positive means unstable. The deck is the NFP = 4 quasi-isodynamic
equilibrium at twice its shipped pressure (beta 5.1 %), whose families 0 and 1
are unstable and family 2 is not. A family is never reported stable unless its
eigensolve converged and its eigenvalue is above the numerical floor.

Needs the optional ``agnimhd`` package (``pip install "vmex[agni]"``), in a version
that has ``Basis(restrict=True)``. The full run takes about ten minutes on
a 16-core machine, most of it in the dense cross-check of the eigensolver.
"""

import json
import os
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import agnimhd as ag
import vmex as vj
from agnimhd.assemble import matfree_operator, represented_basis
from agnimhd.plotting import mode_displacement
from vmex.core.agni import make_agni_map
from vmex.core.solver import prepare_runtime, resolution_from_input

# Input deck, with the pressure doubled (beta 5.1 %):
INPUT_FILE = Path(__file__).resolve().parent / "data" / "input.nfp4_QI_finite_beta"
INPUT_OVERRIDES = dict(pres_scale=2.0)

# VMEX radial multigrid ladder (the last entry is the AGNI map's ns) and tolerances:
NS_ARRAY, FTOL_ARRAY, NITER_ARRAY = [13, 25, 45], [1e-10, 1e-11, 1e-12], [5000, 10000, 40000]

# AGNI basis: nodes per direction (at least 2*MPOL+16 poloidal and 2*NTOR+14
# toroidal) and the Fourier bandwidth the perturbation is restricted to:
N_RHO, N_THETA, N_ZETA, MPOL, NTOR = 10, 32, 28, 6, 5
AUTOMORPHISM = {"eps": 0.02, "x_0": 0.6, "m_1": 2.5, "m_2": 3.0}  # eps is the annular axis cutoff
FAMILIES = None  # None: every independent family, basis.families(NFP)
GAMMA = 5 / 3  # adiabatic index; the mass density is uniform (weight one)

# Eigensolver: shift-invert ARPACK. The shift must lie above the largest gamma2
# (here about 5e-4), so 1e-2 is conservative; a dense eigh of the same restricted
# matrix then checks that the shift picked the top mode.
SIGMA, EIGSH_TOL = 1e-2, 1e-12
DENSE_CHECK_MAX_DIM = 5000  # skip the dense check above this restricted dimension

# A family is "unresolved" unless ||H y - lambda y|| <= RESIDUAL_MAX, |gamma2| > NOISE_FLOOR,
# gamma2 < SIGMA / 2 and, when checked, the dense top eigenvalue agrees within RESIDUAL_MAX:
RESIDUAL_MAX, NOISE_FLOOR = 1e-9, 1e-9

OUTPUT_DIR = Path("output_finite_n_stability")

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs. It only shows
# that the script executes; the physics needs the default settings.
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    NS_ARRAY, FTOL_ARRAY, NITER_ARRAY = [13], [1e-8], [2000]
    N_RHO, N_THETA, N_ZETA, MPOL, NTOR = 6, 14, 12, 2, 1
    FAMILIES = [0]

###############################################################################
# End of input parameters.
###############################################################################

assert ci_smoke or (N_THETA >= 2 * MPOL + 16 and N_ZETA >= 2 * NTOR + 14), "too few angular nodes"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Global finite-n ideal-MHD stability (AGNI). Model limits: fixed boundary; annular cutoff "
      f"eps = {AUTOMORPHISM['eps']}; perturbation Fourier bandwidth ({MPOL},{NTOR}) restricted, so gamma2 "
      "is a lower bound that keeps rising with bandwidth.")

### Equilibrium, then the AGNI map evaluated once on common nodes ##############

t0 = time.perf_counter()
inp = replace(vj.VmecInput.from_file(INPUT_FILE), **INPUT_OVERRIDES,
              ns_array=NS_ARRAY, ftol_array=FTOL_ARRAY, niter_array=NITER_ARRAY)
result = vj.solve_multigrid(inp, verbose=False)
if not result.converged:
    raise RuntimeError(f"force balance did not converge: fsqr={float(result.fsqr):.2e}, "
                       f"fsqz={float(result.fsqz):.2e}, fsql={float(result.fsql):.2e}")
beta = float(vj.wout_from_result(inp, result).betatotal)
print(f"VMEX: {INPUT_FILE.name}, pres_scale = {float(inp.pres_scale)}, converged in "
      f"{int(result.iterations)} iterations, fsqr/fsqz/fsql = {float(result.fsqr):.1e}/"
      f"{float(result.fsqz):.1e}/{float(result.fsql):.1e}  ({time.perf_counter() - t0:.0f} s)")

NFP = int(inp.nfp)
basis = ag.Basis(N_RHO, N_THETA, N_ZETA, radial="lobatto", automorphism=AUTOMORPHISM,
                 mpol=MPOL, ntor=NTOR, restrict=True)
families = list(FAMILIES if FAMILIES is not None else basis.families(NFP))
nodes, _ = basis.nodes_and_diffmat(NFP, families[0])
runtime = prepare_runtime(inp, resolution_from_input(inp, ns=NS_ARRAY[-1]))
t0 = time.perf_counter()
equilibrium_map = make_agni_map(runtime, nodes, inp=inp)
eq, extras = jax.jit(lambda pair: equilibrium_map(pair, extras=True))((result.state, runtime))
jax.block_until_ready(eq.sqrtg)

# Concrete checks: float64, the AGNI data contract, the PEST root, one sign of sqrtg.
sqrtg, iota = np.asarray(eq.sqrtg), np.asarray(eq.iota)
root_residual = float(np.abs(np.asarray(extras["root_residual"])).max())
assert sqrtg.dtype == np.float64, "x64 is off; AGNI needs float64"
eq.validate()
assert root_residual < 1e-9 and (np.all(sqrtg > 0) or np.all(sqrtg < 0)), (root_residual, sqrtg.min())
print(f"AGNI map on {N_RHO}x{N_THETA}x{N_ZETA} nodes ({time.perf_counter() - t0:.0f} s): "
      f"Psi = {float(eq.Psi):.4f} Wb, a = {float(eq.a):.4f} m, beta = {100 * beta:.2f} %, "
      f"iota in [{iota.min():.4f}, {iota.max():.4f}], min |sqrtg| = {np.abs(sqrtg).min():.3e}, "
      f"PEST root residual {root_residual:.1e}")

### Per-family eigensolve ######################################################

assembly = ag.AssemblyConfig(gamma=GAMMA, parity=None)
solver = ag.SolverConfig(eigensolver="eigsh", sigma=SIGMA, eigsh_tol=EIGSH_TOL)
rows, vectors = [], {}
print(f"\n{'family':>6s} {'gamma2':>12s} {'|residual|':>11s} {'dense gamma2':>13s}  verdict")
for family in families:
    t0 = time.perf_counter()
    _, dm = basis.nodes_and_diffmat(NFP, family)
    try:
        g2, v, rel = ag.eigenpair(eq, dm, assembly, solver)
        g2, v = float(g2), np.asarray(v)
        resid = float(rel) * abs(g2) * float(np.linalg.norm(v))
    except Exception as error:  # an eigensolve that fails is "unresolved", never stable
        g2, v, resid = np.nan, None, np.inf
        print(f"family {family}: eigensolve failed: {str(error)[:120]}")
    dense = None
    if v is not None:
        op = matfree_operator(eq, dm, assembly)
        U = represented_basis(op, dm)
        if U.shape[1] <= DENSE_CHECK_MAX_DIM:
            Ax = jax.jit(lambda X: jax.lax.map(op["Ax"], X, batch_size=32))
            AU = np.concatenate([np.asarray(Ax(U[:, s:s + 256].T)) for s in range(0, U.shape[1], 256)])
            H = np.asarray(jnp.conj(U).T) @ AU.T  # H = U^H A U, the restricted block
            dense = -float(np.linalg.eigvalsh(0.5 * (H + H.conj().T))[0])
    resolved = (np.isfinite(g2) and resid <= RESIDUAL_MAX and NOISE_FLOOR < abs(g2) < SIGMA / 2
                and (dense is None or abs(dense - g2) <= RESIDUAL_MAX))
    verdict = ("unresolved" if not resolved else "unstable" if g2 > 0
               else "no resolved instability in the checked space")
    rows.append(dict(family=family, gamma2=g2, residual=resid, gamma2_dense=dense, verdict=verdict,
                     seconds=time.perf_counter() - t0))
    vectors[family] = v
    print(f"{family:>6d} {g2:12.4e} {resid:11.1e} {'-' if dense is None else f'{dense:13.4e}':>13s}  "
          f"{verdict}  ({rows[-1]['seconds']:.0f} s)")
if any(r["gamma2_dense"] is None for r in rows):
    print("caveat: no dense cross-check, so the shift is not confirmed to select the top mode")

### Dominant mode: radial and (m, n) structure of xi^rho ########################

top = max((r for r in rows if r["verdict"] == "unstable"), key=lambda r: r["gamma2"], default=None)
if top is not None:
    family = top["family"]
    _, dm = basis.nodes_and_diffmat(NFP, family)
    xi = mode_displacement(eq, matfree_operator(eq, dm, assembly), vectors[family])[..., 0]
    radial = (np.abs(xi) ** 2).sum(axis=(1, 2))
    radial /= radial.sum()
    # The family's toroidal modes are exp(i (family + k NFP) phi); remove the family phase first.
    xi = xi * np.exp(-1j * family * np.asarray(nodes["zeta"]))
    power = np.abs(np.fft.fft2(xi, axes=(1, 2))) ** 2
    power = power.sum(axis=0) / power.sum()
    m_axis, k_axis = np.arange(-MPOL, MPOL + 1), np.arange(-NTOR, NTOR + 1)
    spectrum = power[np.ix_(m_axis % N_THETA, k_axis % N_ZETA)]
    i, j = np.unravel_index(np.argmax(spectrum), spectrum.shape)
    n_axis = family + k_axis * NFP
    top.update(dominant_m=int(m_axis[i]), dominant_n=int(n_axis[j]), dominant_power=float(spectrum[i, j]),
               radial_peak_rho=float(np.asarray(nodes["rho"])[np.argmax(radial)]))
    print(f"\ndominant mode: family {family}, (m, n) = ({top['dominant_m']}, {top['dominant_n']}) with "
          f"{top['dominant_power']:.2f} of the xi^rho power, radial peak at rho = {top['radial_peak_rho']:.2f}")

### Figure and JSON ############################################################

fig, axes = plt.subplots(1, 3 if top else 1, figsize=(14 if top else 5, 4), constrained_layout=True)
axes = np.atleast_1d(axes)
colors = {"unstable": "tab:red", "unresolved": "tab:gray"}
axes[0].bar([str(r["family"]) for r in rows], [np.nan_to_num(r["gamma2"]) for r in rows],
            color=[colors.get(r["verdict"], "tab:blue") for r in rows])
axes[0].axhspan(-NOISE_FLOOR, NOISE_FLOOR, color="k", alpha=0.2)
axes[0].set(xlabel="toroidal family", ylabel="gamma2 (positive: unstable)",
            title="red unstable, blue none resolved, gray unresolved")
axes[0].set_yscale("symlog", linthresh=1e-7)
axes[0].axhline(0, color="k", lw=0.5)
if top is not None:
    axes[1].plot(nodes["rho"], radial, "o-")
    axes[1].set(xlabel="rho", ylabel="xi^rho power", title=f"family {family}, radial structure")
    image = axes[2].pcolormesh(n_axis, m_axis, spectrum, shading="nearest")
    axes[2].set(xlabel="n", ylabel="m", title="(m, n) power of xi^rho")
    fig.colorbar(image, ax=axes[2])
figure_path = OUTPUT_DIR / "finite_n_stability.png"
fig.savefig(figure_path, dpi=150)

json_path = OUTPUT_DIR / "finite_n_stability.json"
json_path.write_text(json.dumps(dict(
    input=INPUT_FILE.name, pres_scale=float(inp.pres_scale), ns=NS_ARRAY[-1], beta=beta, Psi=float(eq.Psi),
    a=float(eq.a), iota_range=[float(iota.min()), float(iota.max())], min_abs_sqrtg=float(np.abs(sqrtg).min()),
    basis=dict(n_rho=N_RHO, n_theta=N_THETA, n_zeta=N_ZETA, mpol=MPOL, ntor=NTOR, **AUTOMORPHISM),
    solver=dict(sigma=SIGMA, eigsh_tol=EIGSH_TOL, residual_max=RESIDUAL_MAX, noise_floor=NOISE_FLOOR),
    families=[{k: v if k in ("family", "verdict") or v is None or np.isfinite(v) else None
               for k, v in row.items()} for row in rows]), indent=1))
print(f"Wrote {figure_path}\nWrote {json_path}")
if ci_smoke:
    print("smoke mode: execution only, the tiny basis does not resolve the physics")
