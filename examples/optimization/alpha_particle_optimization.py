#!/usr/bin/env python
"""Optimize alpha confinement directly from a shaped NFP=2 seed.

The two design terms are an ESSOS orbit-loss surrogate and aspect ratio.
There is no quasisymmetry objective or warm stage. Hard losses select the
accepted gradient step; fresh particles test it at a longer time. Direct
losses are noisy even with thousands of births (Bindel et al., 2023,
https://arxiv.org/abs/2302.11369); this is a research example.
"""

import os
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import vmex as vj
from essos import constants
from essos.boozer import BoozerField, guiding_center_rhs
from scipy.linalg import null_space
from scipy.optimize import minimize
from vmex import optimize as opt
from vmex.core.omnigenity import boozer_spectrum_state
from vmex.core.scaling import SCALE_TARGETS
from vmex.core.statephysics import _aspect_scalars, _field_chain
from vmex.core.tracing import sample_births

matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)
CI = os.environ.get("VMEX_EXAMPLES_CI") == "1"
N, T, STEPS, MAXITER = (16, 1e-5, 40, 2) if CI else (64, 2.5e-4, 2000, 3)
HARD_N, HARD_T = (16, 1e-4) if CI else (256, 2e-3)
CHECK_N, CHECK_T, CHECK_SEEDS = (16, 1e-4, (2,)) if CI else (512, 5e-3, (2, 3))
SURFACES = (np.arange(10) + 0.5) / 10
ASPECT_WEIGHT, MAX_ASPECT_DRIFT = 5.0, 0.1
NAME = "alpha_particle_optimized"
DATA = Path(__file__).resolve().parents[1] / "data" / "input.minimal_seed_nfp2"


def cubic(x, y):
    """JAX Hermite coefficients in ESSOS BoozerField layout."""
    h = jnp.diff(x)
    sec = jnp.diff(y, axis=0) / h[:, None]
    slope = jnp.concatenate((sec[:1], (sec[:-1] + sec[1:]) / 2, sec[-1:]))
    a, b, da, db = y[:-1], y[1:], slope[:-1], slope[1:]
    return jnp.stack(
        (
            (2 * (a - b) / h[:, None] + da + db) / h[:, None] ** 2,
            (3 * (b - a) / h[:, None] - 2 * da - db) / h[:, None],
            da,
            a,
        ),
        axis=-1,
    )


def scales(state, rt):
    """Differentiable ARIES-CS factors, equal to aries_cs_scales(wout)."""
    aminor, _, aspect, volume = _aspect_scalars(state, rt)
    wb = _field_chain(state, rt)[-1].wb
    bavg = jnp.sqrt(jnp.abs(2 * wb * (2 * jnp.pi) ** 2 / volume))
    bref, aref = SCALE_TARGETS["volavgB"]
    return bref / bavg, aref / aminor, aspect


def field_from_state(state, rt, bs, rs):
    """Live VMEX Boozer spectrum into ESSOS without a WOUT file or NumPy break."""
    a = boozer_spectrum_state(state, rt, surfaces=SURFACES, mboz=5, nboz=5, oversample=1)
    s, xm = a["s_b"], jnp.asarray(a["xm_b"], int)
    r = jnp.sqrt(s)
    b = jnp.where(xm[None, :] > 0, a["bmnc_b"] * bs / r[:, None], a["bmnc_b"] * bs)
    p = jnp.stack((a["iota_b"], a["G_b"] * bs * rs, a["I_b"] * bs * rs), axis=1)
    axis = (p[0] - s[0] * (p[1] - p[0]) / (s[1] - s[0])).at[2].set(0.0)
    ps = jnp.concatenate((jnp.array([0.0]), s))
    return BoozerField(
        r,
        cubic(r, b),
        ps,
        cubic(ps, jnp.concatenate((axis[None], p))),
        xm,
        jnp.asarray(a["xn_b"], int),
        a["psi_edge"] * bs * rs**2,
        a["nfp"],
    )


def birth_weight(field, births):
    """Physical Boozer birth density at fixed angular sample points."""
    s, theta, zeta, _ = births.T
    b = jax.vmap(field.modB)(s, theta, zeta)
    iota, G, current_i = field.profiles(s)[0].T
    return jnp.abs(G + iota * current_i) / b**2


def orbit_risk(field, births, reference_weight):
    """ESSOS guiding centres, weighted soft time peak of normalized flux."""
    s, theta, zeta, pitch = births.T
    speed = jnp.sqrt(2 * constants.FUSION_ALPHA_PARTICLE_ENERGY / constants.ALPHA_PARTICLE_MASS)
    b0 = jax.vmap(field.modB)(s, theta, zeta)
    r = jnp.sqrt(s)
    y = jnp.stack((r * jnp.cos(theta), r * jnp.sin(theta), zeta, pitch * speed), axis=1)
    mu = speed**2 * (1 - pitch**2) / (2 * b0)
    dt = T / STEPS
    rhs = jax.vmap(
        lambda yy, mm: guiding_center_rhs(field, yy, mm, constants.ALPHA_PARTICLE_MASS, constants.ALPHA_PARTICLE_CHARGE)
    )

    def advance(carry, _):
        y, lost, weight, weighted_s = carry
        k1 = rhs(y, mu)
        k2 = rhs(y + 0.5 * dt * k1, mu)
        k3 = rhs(y + 0.5 * dt * k2, mu)
        k4 = rhs(y + dt * k3, mu)
        yn = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        sn = yn[:, 0] ** 2 + yn[:, 1] ** 2
        sn = jnp.where(jnp.isfinite(sn), sn, 1.0)
        lost = lost | (sn >= 1.0)
        capped = jnp.where(lost, 0.95, jnp.minimum(sn, 0.95))
        w = jnp.exp((capped - 0.8) / 0.06)
        return (jnp.where(lost[:, None], y, yn), lost, weight + w, weighted_s + w * capped), None

    (_, _, weight, weighted_s), _ = jax.lax.scan(
        advance, (y, jnp.zeros_like(s, dtype=bool), jnp.zeros_like(s), jnp.zeros_like(s)), None, length=STEPS
    )
    per_particle = jax.nn.sigmoid((weighted_s / weight - 0.8) / 0.06)
    # Fixed births for AD, but candidate-dependent physical Boozer birth measure.
    iota, G, current_i = field.profiles(s)[0].T
    ratio = (jnp.abs(G + iota * current_i) / b0**2) / reference_weight
    return jnp.sum(ratio * per_particle) / jnp.sum(ratio)


inp = vj.VmecInput.from_file(DATA)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor, 1] = zbs[inp.ntor, 1] = 0.17
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -0.03, 0.03
seed_input = replace(inp, rbc=rbc, zbs=zbs, delt=0.5).change_resolution(mpol=5, ntor=5, ntheta=16, nzeta=14)
seed_eq = opt.solve_equilibrium(seed_input)
bs, rs, aspect_target = scales(seed_eq.state, seed_eq.runtime)
aspect_target = float(aspect_target)
seed_field = field_from_state(seed_eq.state, seed_eq.runtime, bs, rs)
assert np.isclose(float(seed_field.psi0), -float(np.asarray(seed_eq.wout.phi)[-1]) * float(bs * rs**2) / (2 * np.pi)), (
    "VMEX/ESSOS toroidal-flux signs disagree"
)
births = jnp.asarray(sample_births(seed_field, N, s=0.3, seed=1))
reference_weight = birth_weight(seed_field, births)


def loss(state, rt):
    """Candidate-weighted orbit risk plus aspect-ratio control."""
    bscale, rscale, aspect = scales(state, rt)
    return (
        orbit_risk(field_from_state(state, rt, bscale, rscale), births, reference_weight)
        + ASPECT_WEIGHT * (aspect - aspect_target) ** 2
    )


problem = opt.VmecProblem.from_loss(
    seed_input,
    loss,
    max_mode=1,
    use_ess=True,
    ess_alpha=1.2,
    restart_from=seed_eq,
    progress=False,
    evaluation_progress=False,
    adjoint_tol=1e-10,
    refine_tol=1e-12,
)
assert problem.names == (
    "RBC(1,0)",
    "RBC(-1,1)",
    "RBC(0,1)",
    "RBC(1,1)",
    "ZBS(1,0)",
    "ZBS(-1,1)",
    "ZBS(0,1)",
    "ZBS(1,1)",
)
# Freeze the redundant alternating m=1 gauge direction of the implicit solve.
basis = null_space(np.array([[0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0]]))
x0, step = np.asarray(problem.x0), 0.02


def hard_check(eq, n, tmax, seed):
    """Check real ESSOS exits using VMEX's released tracing wrapper."""
    trace = vj.trace_alphas(
        eq.wout, nparticles=n, tmax=tmax, s=0.3, seed=seed, times_to_trace=101, mboz=12, nboz=12, mode_tolerance=1e-5
    )
    drift = float(np.max(np.abs(trace.energy_error)))
    if trace.particles_failed or not np.isfinite(drift) or drift > 1e-3:
        raise ValueError("unreliable alpha trace")
    cost = trace.loss_fraction + ASPECT_WEIGHT * (float(eq.wout.aspect) - aspect_target) ** 2
    return cost, trace


best_cost, _ = hard_check(seed_eq, HARD_N, HARD_T, 1)
best_y = np.zeros(basis.shape[1])
accepted = 0


def value_grad(y):
    """Differentiate through equilibria, Boozer spectra, and ESSOS orbits."""
    tic = time.perf_counter()
    value, grad = problem.value_and_grad(x0 + step * basis @ y)
    print(f"smooth cost {value:.5f}, gradient {time.perf_counter() - tic:.1f} s", flush=True)
    return value, step * basis.T @ grad


def check(y):
    """Keep the best accepted step under real short-orbit losses."""
    global accepted, best_cost, best_y
    accepted += 1
    eq = problem.equilibrium_from_x(x0 + step * basis @ y)
    cost, trace = hard_check(eq, HARD_N, HARD_T, 1)
    aspect = float(eq.wout.aspect)
    print(
        f"step {accepted}: {trace.loss_fraction:.1%} hard losses, aspect {aspect:.3f}, "
        f"minor radius {eq.wout.Aminor_p:.4f} m, "
        f"max energy drift {np.max(np.abs(trace.energy_error)):.2e}",
        flush=True,
    )
    if cost < best_cost and abs(aspect - aspect_target) <= MAX_ASPECT_DRIFT:
        best_cost, best_y = cost, np.asarray(y).copy()


print("Direct alpha-orbit autodiff optimization (no symmetry stage)", flush=True)
minimize(
    value_grad,
    np.zeros(basis.shape[1]),
    jac=True,
    method="L-BFGS-B",
    bounds=[(-5.0, 5.0)] * basis.shape[1],
    callback=check,
    options={"maxiter": MAXITER, "maxfun": MAXITER + 10, "ftol": 1e-9},
)
final_x = x0 + step * basis @ best_y
final_input = problem.input_from_x(final_x)
final_eq = seed_eq if not np.any(best_y) else problem.equilibrium_from_x(final_x)
checks = {"seed": [], "optimized": []}
for holdout_seed in CHECK_SEEDS:
    for label, eq in (("seed", seed_eq), ("optimized", final_eq)):
        _, trace = hard_check(eq, CHECK_N, CHECK_T, holdout_seed)
        checks[label].append(trace)
        print(
            f"{label}: {CHECK_N} alphas for {1e3 * CHECK_T:g} ms lose "
            f"{100 * trace.loss_fraction:.1f}% ± {100 * trace.loss_fraction_sigma:.1f}%; "
            f"seed {holdout_seed}, aspect {eq.wout.aspect:.3f}, "
            f"minor radius {eq.wout.Aminor_p:.4f} m, "
            f"max energy drift {np.max(np.abs(trace.energy_error)):.2e}",
            flush=True,
        )

for label, eq in (("seed", seed_eq), ("optimized", final_eq)):
    bb, rr, _ = scales(eq.state, eq.runtime)
    field = field_from_state(eq.state, eq.runtime, bb, rr)
    th, ze = np.meshgrid(
        np.linspace(0, 2 * np.pi, 32, endpoint=False), np.linspace(0, np.pi, 32, endpoint=False), indexing="ij"
    )
    b = np.asarray(jax.vmap(field.modB)(jnp.full(th.size, 0.3), th.ravel(), ze.ravel()))
    print(
        f"{label}: B(s=0.3) {b.min():.3f}–{b.max():.3f} T, "
        f"mirror ratio {b.max() / b.min():.3f}; "
        f"force residuals {eq.wout.fsqr:.1e}, {eq.wout.fsqz:.1e}, "
        f"{eq.wout.fsql:.1e}",
        flush=True,
    )

wout_path = vj.write_wout(f"wout_{NAME}.nc", final_eq.wout)
print(f"Wrote {final_input.to_indata(f'input.{NAME}')} and {wout_path}")
fig, ax = plt.subplots(figsize=(5, 3), layout="constrained")
for label, traces in checks.items():
    p = np.mean([trace.loss_fractions for trace in traces], axis=0)[1:]
    n, z = CHECK_N * len(traces), 1.96
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    (line,) = ax.plot(traces[0].times[1:], 100 * p, label=label)
    ax.fill_between(
        traces[0].times[1:], 100 * (center - half), 100 * (center + half), color=line.get_color(), alpha=0.15
    )
ax.set(xscale="log", xlabel="time [s]", ylabel="alphas lost [%]")
ax.legend(frameon=False, title="shading: pointwise 95% Wilson interval")
fig.savefig(f"{NAME}_losses.png", dpi=150)
print(f"Wrote {NAME}_losses.png")
