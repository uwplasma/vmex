"""Closed-transit action bounds for passing particles near rational surfaces.

These pure kernels implement finite-moment bounds, not an orbit certificate.
The caller supplies a fixed B_star, a resolved Boozer spectrum, and separately
bounds angular, radial and field errors. No VMEC2000 equation is replaced.
"""
from math import comb, gcd

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp


def transit_basis(m, n, numerator, denominator, *, nphase=64, ntransit=512):
    """Return cosine/sine tables on full closed transits, shape (phase, transit, mode).

    Static integer ``n`` is physical (already includes nfp). Labels are
    chi=M*eta in [0, 2*pi); zeta spans [0, 2*pi*M). Precompute once per stage.
    """
    if (not isinstance(numerator, (int, np.integer))
            or not isinstance(denominator, (int, np.integer))
            or denominator <= 0 or gcd(abs(numerator), denominator) != 1):
        raise ValueError("rational indices must be reduced integers with denominator > 0")
    m, n = np.asarray(m), np.asarray(n)
    if (m.ndim != 1 or m.size == 0 or m.shape != n.shape
            or not np.all(np.isfinite(m)) or not np.all(np.isfinite(n))
            or np.any(m != np.round(m)) or np.any(n != np.round(n))):
        raise ValueError("m and physical n must be equal nonempty integer vectors")
    if any(not isinstance(k, (int, np.integer)) or k < 2 for k in (nphase, ntransit)):
        raise ValueError("angular counts must be integers >= 2")
    chi = 2 * np.pi * np.arange(nphase) / nphase
    zeta = 2 * np.pi * denominator * np.arange(ntransit) / ntransit
    angle = (chi[:, None, None] * m / denominator
             + zeta[None, :, None] * (m * numerator / denominator - n))
    return jnp.asarray(np.cos(angle)), jnp.asarray(np.sin(angle))


def _field(b):
    b = jnp.asarray(b, dtype=jnp.float64)
    if b.ndim != 2 or min(b.shape) < 1:
        raise ValueError("b must have nonempty shape (phase, transit)")
    valid = jnp.all(jnp.isfinite(b) & (b > 0) & (b <= 1))
    # Eager calls fail explicitly; transformed calls return NaN on invalid
    # trials so the caller must reject them, never clip the physical domain.
    if not isinstance(valid, jax.core.Tracer) and not bool(valid):
        raise ValueError("b must be finite and in (0, 1]")
    return b, valid


def transit_action(b, pitch_parameter):
    """Sampled action mean(sqrt(1-lambda*b)/b), shape (phase, pitch).

    ``pitch_parameter`` is lambda=mu*B_star/E, not v_parallel/v. The endpoint
    lambda=1 is action-only; dynamical bounds require a positive passing margin.
    Invalid transformed inputs return NaN; eager calls raise ValueError.
    """
    b, valid = _field(b)
    lam = jnp.atleast_1d(jnp.asarray(pitch_parameter, dtype=b.dtype))
    if lam.ndim != 1 or lam.size == 0:
        raise ValueError("pitch_parameter must be a scalar or nonempty vector")
    valid_pitch = jnp.all(jnp.isfinite(lam) & (lam >= 0) & (lam <= 1))
    if not isinstance(valid_pitch, jax.core.Tracer) and not bool(valid_pitch):
        raise ValueError("pitch_parameter must be finite and in [0, 1]")
    value = jnp.mean(jnp.sqrt(1 - b[..., None] * lam) / b[..., None], axis=1)
    return jnp.where(valid & valid_pitch, value, jnp.nan)


def moment_bound(b, *, order=32, lambda_max=0.95, softness=None, oscillation_error=None):
    """Upper envelope of action oscillation for every lambda in [0, lambda_max].

    ``order``, ``lambda_max`` and ``softness`` are static stage parameters.
    Optional nonnegative ``oscillation_error`` has ``order`` entries for powers
    (-1, 1, ..., order-1). Without it this bounds sampled measures only.
    Soft log-sum-exp adds at most 2*softness*log(nphase) per moment; it is
    deliberately not normalized by nphase. The truncation tail is one-sided.
    Powers accumulate by scan, without a (pitch, order, phase, transit) tensor.
    Invalid transformed fields/errors return NaN rather than a false bound.
    """
    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("order must be a positive integer")
    if not np.isfinite(lambda_max) or not 0 <= lambda_max <= 1:
        raise ValueError("lambda_max must be finite and in [0, 1]")
    if softness is not None and (not np.isfinite(softness) or softness <= 0):
        raise ValueError("softness must be finite and positive")
    b, valid = _field(b)

    def oscillation(x):
        if softness is None:
            return jnp.max(x) - jnp.min(x)
        return softness * (logsumexp(x / softness) + logsumexp(-x / softness))

    def step(power, _):
        return power * b, oscillation(jnp.mean(power, axis=-1))

    _, positive = jax.lax.scan(step, b, None, length=order - 1)
    oscillations = jnp.concatenate((oscillation(jnp.mean(1 / b, axis=-1))[None], positive))
    if oscillation_error is not None:
        error = jnp.asarray(oscillation_error, dtype=b.dtype)
        if error.shape != (order,):
            raise ValueError("oscillation_error must have one entry per moment")
        valid_error = jnp.all(jnp.isfinite(error) & (error >= 0))
        if not isinstance(valid_error, jax.core.Tracer) and not bool(valid_error):
            raise ValueError("oscillation_error must be finite and nonnegative")
        valid = valid & valid_error
        oscillations = oscillations + error
    weights = jnp.asarray([1.] + [comb(2*k, k) / (4**k * (2*k-1)) * lambda_max**k
                                  for k in range(2, order + 1)])
    tail = lambda_max**(order + 1) * comb(2*order, order) / 4**order
    if lambda_max < 1:
        tail = min(tail, comb(2*order+2, order+1) / (4**(order+1) * (2*order+1))
                   * lambda_max**(order+1) / (1-lambda_max))
    return jnp.where(valid, weights @ oscillations + tail, jnp.nan)
