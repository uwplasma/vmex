"""Shear-cancelled DeltaRes reductions for supplied complete, matched bounce maps.

Equations (2.8), (3.5)-(3.10) of Labbate, Paul and Chambliss,
arXiv:2609.01890v1. These kernels do not construct or certify a physical
bounce map. The published phase-space convention and squared radial rows
are retained; the result is neither an island width nor a loss probability.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

def stabilized_power(delta_s, q_max: int):
    """Return ``256 |f_q|^2 / (pi q)^2``,  with eta on the last axis.

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


def linear_crossing_weights(omega, resonances, valid=None):
    """Deposit every resolved crossing onto its two radial neighbours.

    ``omega`` has leading radial axis and arbitrary trailing family axes;
    ``resonances`` is one-dimensional. Strict interval interiors and a separate
    node contribution count node crossings once, including domain endpoints.
    Known-absent nodes may be marked False in ``valid``; gaps are never bridged.
    A rational plateau, isolated rational node or resolved tangent invalidates
    that family/resonance with NaN instead of a misleading zero penalty.
    The deposition is piecewise differentiable, not differentiable at nodes.
    """
    omega, resonances = jnp.asarray(omega), jnp.asarray(resonances)
    if omega.ndim < 1 or omega.shape[0] < 2 or resonances.ndim != 1:
        raise ValueError("Require at least two radial nodes and a resonance vector")
    valid = jnp.ones(omega.shape, dtype=bool) if valid is None else jnp.asarray(valid, dtype=bool)
    if valid.shape != omega.shape:
        raise ValueError("valid must have omega's shape")
    finite = jnp.isfinite(omega)
    safe = jnp.where(valid & finite, omega, 0.0)[..., None]
    difference = safe - resonances
    paired = (valid[:-1] & valid[1:] & finite[:-1] & finite[1:])[..., None]
    step = safe[1:] - safe[:-1]
    crossing = paired & (difference[:-1] * difference[1:] < 0)
    t = -difference[:-1] / jnp.where(step != 0, step, 1.0)
    weights = jnp.zeros_like(difference)
    weights = weights.at[:-1].add(jnp.where(crossing, 1 - t, 0.0))
    weights = weights.at[1:].add(jnp.where(crossing, t, 0.0))
    node = valid[..., None] & finite[..., None] & (difference == 0)
    connected = jnp.zeros_like(node).at[:-1].set(paired).at[1:].max(paired)
    plateau = jnp.any(paired & (step == 0) & node[:-1], axis=0)
    tangent = jnp.any(node[1:-1] & paired[:-1] & paired[1:]
                      & (difference[:-2] * difference[2:] > 0), axis=0)
    resolved = ~(plateau | tangent | jnp.any(node & ~connected, axis=0)
                 | jnp.any(valid & ~finite, axis=0)[..., None]) & jnp.isfinite(resonances)
    return jnp.where(resolved[None], weights + node, jnp.nan)


def deltares_from_maps(delta_s, delta_eta, v_tau, bcrit, bcrit_weight,
                      line_length, *, p_max=8, q_max=8, rotation="frequency_ratio"):
    """Reduce certified full-bounce maps with axes (surface, eta, pitch, well).

    All cells must contain a complete, consistently matched family. Partial
    coverage is deliberately unsupported here. ``v_tau`` is speed times period
    in metres, ``delta_s`` is normalized physical flux displacement, and
    ``delta_eta`` is radians per full bounce. ``line_length`` is mean integral
    dl/B over the *same* spatial domain. Its counting must be certified by the
    caller. For independent alpha sampling, use :func:`paper_average` directly.

    ``frequency_ratio`` preserves the archive's ratio of mean frequencies,
    including genuine zero precession. ``mean_map`` averages the map rotation;
    these differ when periods vary. ``loss`` is half the sum of squared reduced
    rows, with zero target and unit row weights, without companion constraints.
    Invalid data returns NaN and ``valid=False``. This is a map reduction, not
    an equilibrium objective or a confinement certificate.
    """
    ds, de, tau = map(jnp.asarray, (delta_s, delta_eta, v_tau))
    if ds.ndim != 4 or ds.shape != de.shape or ds.shape != tau.shape or min(ds.shape) < 1:
        raise ValueError("Maps must share nonempty (surface, eta, pitch, well) axes")
    if rotation not in ("frequency_ratio", "mean_map"):
        raise ValueError("rotation must be frequency_ratio or mean_map")
    valid = (jnp.all(jnp.isfinite(ds)) & jnp.all(jnp.isfinite(de))
             & jnp.all(jnp.isfinite(tau)) & jnp.all(tau > 0))
    safe_tau = jnp.where(jnp.isfinite(tau) & (tau > 0), tau, 1.0)
    omega = (jnp.mean(de / safe_tau, axis=1) / jnp.mean(2 * jnp.pi / safe_tau, axis=1)
             if rotation == "frequency_ratio" else jnp.mean(de, axis=1) / (2 * jnp.pi))
    pairs = resonance_pairs(p_max, q_max)
    power = stabilized_power(jnp.moveaxis(ds, 1, -1), q_max)
    weights = linear_crossing_weights(omega, pairs[:, 0] / pairs[:, 1])
    score = jnp.sum(power[..., pairs[:, 1] - 1] * weights, axis=-1)
    rows = paper_average(score, tau, bcrit, bcrit_weight, line_length)
    valid &= jnp.all(jnp.isfinite(rows))
    rows = jnp.where(valid, rows, jnp.nan)
    return {"delta_res": rows, "loss": 0.5 * jnp.sum(rows**2), "omega": omega,
            "harmonic_power": power, "weights": weights, "valid": valid}
