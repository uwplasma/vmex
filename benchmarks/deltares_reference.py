"""DeltaRes reduction contracts, not an equilibrium or orbit implementation.

Labbate, Paul and Chambliss, arXiv:2609.01890v1, equations (2.8),
(3.5)-(3.10). Inputs must already use matched, complete magnetic wells.
The shear-cancelled spectrum is evaluated without a singular division.
Timings exclude field geometry, bounce integration and equilibrium adjoints.
"""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def stabilized_power(delta_s, q_max: int):
    """Return 256 |f_q|^2 / (pi q)^2, with eta on the last axis.

    Samples use eta=2*pi*j/n without the duplicate endpoint. q_max is static.
    No square root, shear division or amplitude floor occurs. Resonance
    weights, phase-space integration and the outer loss are separate.
    """
    delta_s = jnp.asarray(delta_s)
    if delta_s.ndim < 1 or q_max < 1 or delta_s.shape[-1] <= 2 * q_max:
        raise ValueError("Require a last eta axis with n_eta > 2*q_max >= 2")
    f = jnp.fft.rfft(delta_s, axis=-1)[..., 1:q_max + 1] / delta_s.shape[-1]
    q = jnp.arange(1, q_max + 1, dtype=delta_s.dtype)
    return (256.0 / jnp.pi**2) * (f.real**2 + f.imag**2) / q**2


def resonance_pairs(p_max: int, q_max: int, lower=-4.0, upper=4.0):
    """Static signed (p,q) list; retain distinct harmonics at equal p/q."""
    if p_max < 0 or q_max < 1 or not lower <= upper:
        raise ValueError("Invalid rational limits")
    pairs = [(p, q) for q in range(1, q_max + 1)
             for p in range(-p_max, p_max + 1) if lower <= p / q <= upper]
    return np.asarray(pairs, dtype=np.int32).reshape(-1, 2)


def normalized_bump(distance, half_width):
    """Appendix B density in Omega, not a radial deposition weight.

    Inactive exponential branches are finite. A nonpositive or nonfinite
    width returns NaN, not a favorable zero. A fixed-functional FD test
    must hold width fixed, or include its derivative in both evaluations.
    """
    distance, half_width = jnp.asarray(distance), jnp.asarray(half_width)
    valid = jnp.isfinite(half_width) & (half_width > 0)
    width = jnp.where(valid, half_width, 1.0)
    x = distance / width
    inside = jnp.isfinite(x) & (jnp.abs(x) < 1)
    x_safe = jnp.where(inside, x, 0.0)
    value = 71.12518788738504 / width * jnp.exp(-4.0 / (1 - x_safe**2))
    return jnp.where(valid & jnp.isfinite(distance), jnp.where(inside, value, 0.0), jnp.nan)


def paper_average(score, v_tau, bcrit, bcrit_weight, line_length):
    """Equation (3.10), preserving the paper's factor two.

    score=(surface,pitch,well), v_tau=(surface,alpha,pitch,well),
    shared bcrit and dBcrit weights=(pitch,), line_length=(surface,).
    line_length is mean_alpha integral dl/B over the matching domain.
    Finite zero padding means known absence, never unresolved topology.
    This is not automatically a probability-normalized phase-space average.
    """
    score, v_tau, bcrit, bcrit_weight, line_length = map(
        jnp.asarray, (score, v_tau, bcrit, bcrit_weight, line_length))
    if (score.ndim != 3 or v_tau.ndim != 4
            or score.shape != (v_tau.shape[0], v_tau.shape[2], v_tau.shape[3])
            or bcrit.shape != (score.shape[1],) or bcrit_weight.shape != bcrit.shape
            or line_length.shape != (score.shape[0],) or v_tau.shape[1] < 1):
        raise ValueError("Inconsistent phase-space array shapes")
    valid = (jnp.all(jnp.isfinite(score)) & jnp.all(score >= 0)
             & jnp.all(jnp.isfinite(v_tau)) & jnp.all(v_tau >= 0)
             & jnp.all(jnp.isfinite(bcrit)) & jnp.all(bcrit > 0)
             & jnp.all(jnp.isfinite(bcrit_weight)) & jnp.all(bcrit_weight >= 0)
             & jnp.any(bcrit_weight > 0)
             & jnp.all(jnp.isfinite(line_length)) & jnp.all(line_length > 0))
    safe_b = jnp.where(bcrit > 0, bcrit, 1.0)
    safe_length = jnp.where(line_length > 0, line_length, 1.0)
    weights = bcrit_weight / safe_b**2
    numerator = jnp.sum(v_tau * score[:, None] * weights[None, None, :, None], axis=(1, 2, 3))
    return jnp.where(valid, numerator / (2 * v_tau.shape[1] * safe_length), jnp.nan)


def helical_angles(chi, eta, iota, nfp: int, M: int, N: int):
    """Invert chi=M*theta-N*zeta, eta=nfp*(theta-iota*zeta)/(N-M*iota).

    N is physical, not divided by nfp. Periodic eta does not imply a
    periodic chi trace or a correctly matched well family.
    """
    D = M * jnp.asarray(iota) - N
    safe = jnp.where(D != 0, D, 1.0)
    theta = iota * chi / safe + N * eta / nfp
    zeta = chi / safe + M * eta / nfp
    return jnp.where(D != 0, theta, jnp.nan), jnp.where(D != 0, zeta, jnp.nan)


def benchmark(repeats=9):
    """Synchronized synthetic-reduction timings; input map remains dynamic."""
    if not jax.config.jax_enable_x64:
        raise RuntimeError("Run the reference in x64")
    rng = np.random.default_rng(421)
    inputs = jnp.asarray(rng.normal(size=(8, 9, 1, 40)))
    scalar = lambda data: jnp.sum(stabilized_power(data, 8))
    timings = {}
    for name, function in (("value", scalar), ("value_and_grad", jax.value_and_grad(scalar))):
        compiled = jax.jit(function)
        start = time.perf_counter()
        jax.block_until_ready(compiled(inputs))
        first = time.perf_counter() - start
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            jax.block_until_ready(compiled(inputs))
            samples.append(time.perf_counter() - start)
        timings[name] = {"first_call_s": first, "warm_median_s": float(np.median(samples))}
    return {"scope": "synthetic reduction only; no equilibrium, bounce or orbit solve",
            "python": platform.python_version(), "jax": jax.__version__,
            "devices": [str(d) for d in jax.devices()], "x64": True,
            "xla_flags": os.environ.get("XLA_FLAGS", ""), "numpy": np.__version__,
            "map_shape": list(inputs.shape), "q_max": 8, "repeats": repeats, "timings": timings}


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    result = benchmark()
    Path("deltares_reference_timing.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
