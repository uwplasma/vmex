"""Shear-cancelled DeltaRes reductions for supplied complete, matched bounce maps.

Equations (2.8), (3.5)-(3.10) of Labbate, Paul and Chambliss,
arXiv:2609.01890v1. Supplied-map reductions require caller certification;
the restricted action primitive covers one finite-cell pitch band only.
The published phase-space convention and squared radial rows
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
    result = numerator / (2 * v_tau.shape[1] * safe_length)
    return jnp.where(valid & jnp.isfinite(result), result, jnp.nan)


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


def pitch_band_phase_space_mass(bmag, spatial_weight, bcrit_lower, bcrit_upper):
    """Analytically integrate the unit-score pitch measure before space.

    The last axis contains spatial quadrature nodes and ``spatial_weight``
    contains their nonnegative ``dl/B`` weights on the same domain. Positive
    scalar physical bounds integrate ``dBcrit/Bcrit**2``. The result preserves
    :func:`paper_average`'s factor two and is not a probability.

    Fubini's theorem gives ``2 <sqrt((1-B/U)+)-sqrt((1-B/L)+)>``.
    This counts *all accessible segments*, including cut wells and passing
    segments in a bounded trace; it does not certify trapped-family coverage.
    Complete wells alone give this result only when the spatial domain and
    pitch band have no cut or passing components. This independent measure
    avoids bounce roots and the integrable pitch-separatrix singularity.
    AD is finite away from a quadrature node coinciding with a band endpoint;
    that coincidence is nondifferentiable (the inactive branch selects zero).
    Spatial quadrature convergence and physical-field positivity still need
    independent certification. Invalid data returns NaN, never zero mass.
    """
    b, weight, lower, upper = map(jnp.asarray,
                                (bmag, spatial_weight, bcrit_lower, bcrit_upper))
    if b.ndim < 1 or b.shape[-1] < 1 or lower.ndim or upper.ndim:
        raise ValueError("Require a spatial node axis and scalar physical pitch bounds")
    weight = jnp.broadcast_to(weight, b.shape)
    valid = (jnp.all(jnp.isfinite(b) & (b > 0), axis=-1)
             & jnp.all(jnp.isfinite(weight) & (weight >= 0), axis=-1)
             & (jnp.sum(weight, axis=-1) > 0)
             & jnp.isfinite(lower) & jnp.isfinite(upper)
             & (lower > 0) & (upper > lower))
    lo, hi = jnp.where(lower > 0, lower, 1.), jnp.where(upper > 0, upper, 1.)
    def positive_sqrt(x):
        active = x > 0
        return jnp.where(active, jnp.sqrt(jnp.where(active, x, 1.)), 0.)
    density = 2 * (positive_sqrt(1 - b / hi) - positive_sqrt(1 - b / lo))
    scale = jnp.max(weight, axis=-1, keepdims=True)
    normalized = weight / jnp.where(jnp.isfinite(scale) & (scale > 0), scale, 1.)
    denominator = jnp.sum(normalized, axis=-1)
    mass = jnp.sum(normalized * density, axis=-1) / jnp.where(denominator > 0, denominator, 1.)
    return jnp.where(valid & jnp.isfinite(denominator) & (denominator > 0)
                     & jnp.isfinite(mass), mass, jnp.nan)


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
    loss = 0.5 * jnp.sum(rows**2)
    valid &= jnp.all(jnp.isfinite(rows)) & jnp.isfinite(loss)
    rows = jnp.where(valid, rows, jnp.nan)
    return {"delta_res": rows, "loss": jnp.where(valid, loss, jnp.nan), "omega": omega,
            "harmonic_power": power, "weights": weights, "valid": valid}


def _physical_flux_derivative(values, psi, psi_edge):
    """Differentiate local quadratics in actual ``s=psi/psi_edge``.

    Three neighbouring nodes are fixed by index; signed fluxes stay live.
    Interior stencils are centred, with one-sided stencils at the endpoints.
    """
    x = jnp.asarray(psi) / psi_edge
    if x.ndim != 1 or x.size < 3 or values.shape[0] != x.size:
        raise ValueError("Require at least three matching physical flux nodes")
    rows = []
    for i in range(x.size):
        lo = min(max(i - 1, 0), x.size - 3)
        indices = [lo, lo + 1, lo + 2]
        weights = []
        for j in indices:
            a, b = [k for k in indices if k != j]
            weights.append((2 * x[i] - x[a] - x[b]) / ((x[j] - x[a]) * (x[j] - x[b])))
        rows.append(jnp.einsum("j,j...->...", jnp.stack(weights), values[jnp.asarray(indices)] - values[i]))
    return jnp.stack(rows)


def restricted_action_deltares_from_boozer(
    *, bmnc_b, xm_b, xn_b, iota_b, G_b, I_b, nfp, psi_b, psi_edge,
    bcrit, bcrit_weight, mass, speed, charge, bmns_b=None, M=1, N=0,
    canonical_flux_sign=-1, n_eta=65, n_t=513, t_bounds=(-5.5, 5.5),
    quadrature_order=96, p_max=2, q_max=4, rotation="mean_map",
    endpoint_limit=1.5, log_action_limit=.35, log_period_limit=.35,
    chart_floor=1e-4, root_residual_limit=.005,
    boundary_forbidden_floor=.001, root_slope_floor=.005,
):
    """Restricted single-family action diagnostic, **not** full-population DeltaRes.

    ``bmnc_b``/``bmns_b`` are surface-by-mode spectra in tesla; ``xn_b`` and
    helicity ``N`` are physical toroidal indices. At least three distinct,
    outward ordered physical toroidal-flux-per-radian rows ``psi_b`` are
    required, with ``0 <= psi_b/psi_edge <= 1``. ``psi_edge`` has the same convention and units. The default
    canonical sign follows VMEX's ESSOS adapter, ``psi0=-Phi/(2*pi)``; callers
    with already canonical spectra must explicitly choose ``+1``.

    This evaluates exactly one complete well at every eta and shared physical
    ``bcrit`` node on the fixed bounded cell ``t=zeta-M*eta/nfp``. Both parities,
    roots, iota, line elements and actual fluxes remain live. Only grid, slot
    and local three-point stencil indices are fixed. Adjacent endpoints,
    action and period must satisfy the declared continuity limits and close
    after one eta cycle. Extra, cut, marginal or unresolved wells invalidate
    the entire result; no failed population is discarded or zero padded.

    With ``A=2*integral sqrt(1-B/bcrit) dl`` in metres and
    ``c=mass*speed/(charge*canonical_psi_edge)``, the leading local-bounce map
    is ``Delta_eta=-c*C*A_s|eta``, ``C=nfp/(N-M*iota)``. The chain terms from
    changing alpha to eta cancel. ``A_s`` differentiates a local quadratic
    interpolant on actual ``s=psi/psi_edge``; it is not a continuum derivative
    certificate. Action Fourier coefficients give the shear-cancelled power
    directly, retaining bounce cancellation before squaring. Particle constants
    use SI units; ``v_tau`` is speed times the full-bounce period in metres.

    ``bcrit_weight`` integrates dBcrit; the reduction includes its inverse-square
    pitch Jacobian. Its denominator is mean_eta integral_cell dl/B over this
    same finite cell, preserving the paper's factor two. Uniform eta's
    fixed-surface alpha Jacobian cancels in this restricted ratio.
    ``paper_pitch_band_mass`` is its unit-score integral, not full coverage or
    a probability. ``full_population`` is always False. Invalid certificates
    give NaN rows/loss and ``valid=False``. Grid refinement, production-state
    derivatives and full-domain normalization require independent checks.

    Physical mode labels must have integer ``xm_b`` and ``xn_b/nfp``. A necessary
    sampling check requires every supplied mode, including zero coefficients,
    to satisfy ``max(abs(m*iota-n))*dt < pi`` and
    ``max(abs((m*N-n*M)/nfp)) < n_eta/2``. A curvature bound additionally
    certifies continuous-t well counts at the sampled eta values: every trace
    interval must exclude a root by its interpolation-error bound or have a
    provably monotone derivative. The same bound certifies continuous-t field
    positivity at sampled eta. Equality and unresolved intervals invalidate
    the result. This does not certify eta-family resolution or integral
    accuracy; independent refinement remains required.

    Static settings include nfp, helicity, grid sizes, bounds and certificate
    limits. ``loss`` retains half the sum of squared deposited radial rows;
    its radial grid-phase dependence is not cured by action convergence.
    When using ``boozer_spectrum_state``, omit its computational mesh label
    ``s_b`` and pass the named spectrum and physical flux entries above.
    """
    from math import isfinite
    from .bounce import bounce_action

    integers = (M, N, nfp, n_eta, n_t, quadrature_order, p_max, q_max)
    limits = (endpoint_limit, log_action_limit, log_period_limit, chart_floor,
              root_residual_limit, root_slope_floor)
    if (not all(isfinite(v) and int(v) == v for v in integers)
            or not all(isfinite(v) and v > 0 for v in limits)
            or len(t_bounds) != 2 or not all(isfinite(v) for v in t_bounds)
            or not isfinite(boundary_forbidden_floor) or boundary_forbidden_floor < 0
            or nfp < 1 or n_t < 4 or n_eta <= 2 * q_max or q_max < 1 or p_max < 0
            or quadrature_order < 2 or t_bounds[1] <= t_bounds[0]):
        raise ValueError("Invalid static grid, chart, or certificate settings")
    if (M, N) not in ((1, 0), (1, nfp), (1, -nfp), (0, nfp), (0, -nfp)):
        raise ValueError("Require a primitive QA, QH, or simple QI helicity")
    if canonical_flux_sign not in (-1, 1) or rotation not in ("mean_map", "frequency_ratio"):
        raise ValueError("Require an explicit flux sign and declared rotation")
    c = jnp.asarray(bmnc_b, dtype=jnp.float64)
    sn = jnp.zeros_like(c) if bmns_b is None else jnp.asarray(bmns_b, dtype=c.dtype)
    m, n, iota, G, current, psi, edge, bc, wb = map(
        jnp.asarray, (xm_b, xn_b, iota_b, G_b, I_b, psi_b, psi_edge, bcrit, bcrit_weight))
    mass, speed, charge = map(jnp.asarray, (mass, speed, charge))
    if (any(x.ndim != 0 for x in (mass, speed, charge))
            or c.ndim != 2 or c.shape[0] < 3 or c.shape[1] < 1 or sn.shape != c.shape
            or m.shape != (c.shape[1],) or n.shape != m.shape
            or any(x.shape != (c.shape[0],) for x in (iota, G, current, psi))
            or edge.ndim != 0 or bc.ndim != 1 or bc.size < 1 or wb.shape != bc.shape):
        raise ValueError("Inconsistent surface, spectrum, flux, or physical pitch arrays")
    psi, edge = canonical_flux_sign * psi, canonical_flux_sign * edge
    t = jnp.linspace(*t_bounds, n_t)
    eta = jnp.linspace(0, 2 * jnp.pi, n_eta + 1)
    h, k = (m * N - n * M) / nfp, iota[:, None] * m[None] - n[None]
    # Separable contractions avoid a surface*eta*mode*t phase tensor.
    kt, he = k[:, :, None] * t, eta[:, None] * h
    ct, st = jnp.cos(kt), jnp.sin(kt)
    B = (jnp.einsum("smt,em->set", c[:, :, None] * ct + sn[:, :, None] * st, jnp.cos(he))
         + jnp.einsum("smt,em->set", -c[:, :, None] * st + sn[:, :, None] * ct, jnp.sin(he)))
    K = jnp.abs(G + iota * current)
    dl = K[:, None, None] / B
    length = t_bounds[1] - t_bounds[0]
    out = bounce_action(B, 1 / bc, dl_dphi=dl, length=length, periodic=False,
                        max_wells=1, quadrature_order=quadrature_order, drift_integrands={})
    A, tau, left, right = (out[key] for key in ("action", "bounce_time", "well_left", "well_right"))
    descriptors = jnp.stack((left, right, jnp.log(A), jnp.log(tau)), axis=-1)
    continuity = jnp.array([endpoint_limit, endpoint_limit, log_action_limit, log_period_limit])
    family_eta_valid = jnp.all(jnp.abs(jnp.diff(descriptors, axis=1)) <= continuity)
    family_radial_valid = jnp.all(jnp.abs(jnp.diff(descriptors, axis=0)) <= continuity)
    closure_valid = jnp.all(jnp.abs(descriptors[:, -1] - descriptors[:, 0]) < 1e-9)
    # The single complete slot has a unique reciprocal neighbour. Certify the
    # spectral roots independently of their defining linear interpolation.
    roots = jnp.concatenate((left, right), axis=-1) + t_bounds[0]
    phase = (k[:, None, None, :, None] * roots[:, :, :, None, :]
             + h[None, None, None, :, None] * eta[None, :, None, None, None])
    Br = jnp.sum(c[:, None, None, :, None] * jnp.cos(phase)
                 + sn[:, None, None, :, None] * jnp.sin(phase), axis=3)
    Bt = jnp.sum(k[:, None, None, :, None] * (-c[:, None, None, :, None] * jnp.sin(phase)
                 + sn[:, None, None, :, None] * jnp.cos(phase)), axis=3)
    root_residual = jnp.max(jnp.abs(Br / bc[None, None, :, None] - 1))
    minimum_root_slope = jnp.min(jnp.abs(Bt) / bc[None, None, :, None])
    boundary_margin = jnp.min(B[:, :, (0, -1)][:, :, None, :] / bc[None, None, :, None] - 1)
    root_boundary_distance = jnp.min(jnp.minimum(roots - t_bounds[0], t_bounds[1] - roots))
    hard_status = jnp.any(jnp.stack([out[key] for key in
                                   ("absent", "marginal", "merged", "truncated", "overflow")]))
    flux_valid = (jnp.all(jnp.isfinite(psi)) & jnp.isfinite(edge) & (jnp.abs(edge) > 1e-12)
                  & jnp.all(jnp.diff(psi) * edge > 0) & jnp.all((psi / edge >= 0) & (psi / edge <= 1)))
    chart_distance = M * iota - N
    chart_valid = (jnp.all(jnp.abs(chart_distance) > chart_floor)
                   & jnp.all(chart_distance[:-1] * chart_distance[1:] > 0))
    periodic_valid = (jnp.all(jnp.abs(m - jnp.round(m)) < 1e-12)
                      & jnp.all(jnp.abs(n / nfp - jnp.round(n / nfp)) < 1e-12)
                      & jnp.all(jnp.abs(h - jnp.round(h)) < 1e-12))
    max_trace_phase_step = jnp.max(jnp.abs(k)) * length / (n_t - 1)
    max_eta_harmonic = jnp.max(jnp.abs(h))
    sampling_valid = (max_trace_phase_step < jnp.pi) & (max_eta_harmonic < n_eta / 2)
    # |B_tt| <= M2. Linear interpolation error is <= M2*dt**2/8;
    # |B_t-secant| <= M2*dt/2. Strict bounds exclude hidden root pairs.
    dt = length / (n_t - 1)
    M2 = jnp.sum(jnp.hypot(c, sn) * k**2, axis=1)[:, None, None, None]
    reflecting = B[:, :, None, :] - bc[None, None, :, None]
    a, b = reflecting[..., :-1], reflecting[..., 1:]
    no_root = (a * b > 0) & (jnp.minimum(jnp.abs(a), jnp.abs(b)) > M2 * dt**2 / 8)
    monotone = jnp.abs(jnp.diff(B, axis=-1))[:, :, None, :] / dt > M2 * dt / 2
    unresolved_interval_count = jnp.sum(~(no_root | monotone))
    well_count_valid = unresolved_interval_count == 0
    minimum_endpoint_B = jnp.minimum(B[..., :-1], B[..., 1:])[:, :, None, :]
    field_positive_valid = jnp.all((minimum_endpoint_B > M2 * dt**2 / 8)
                                   | ((minimum_endpoint_B > 0) & monotone))
    particle_valid = (jnp.isfinite(mass) & (mass > 0) & jnp.isfinite(speed) & (speed > 0)
                      & jnp.isfinite(charge) & (charge != 0))
    valid = (flux_valid & chart_valid & periodic_valid & sampling_valid & well_count_valid & particle_valid
             & field_positive_valid & jnp.all(K > 0) & ~hard_status & jnp.all(out["usable_mask"])
             & family_eta_valid & family_radial_valid & closure_valid
             & (root_residual < root_residual_limit) & (minimum_root_slope > root_slope_floor)
             & (boundary_margin > boundary_forbidden_floor) & (root_boundary_distance > length / (n_t - 1)))
    C, coefficient = nfp / (N - M * iota), mass * speed / (charge * edge)
    A_s = _physical_flux_derivative(A, psi, edge)
    delta_eta = -coefficient * C[:, None, None, None] * A_s
    Au, tu, de = A[:, :-1], tau[:, :-1], delta_eta[:, :-1]
    aq = jnp.fft.rfft(jnp.moveaxis(Au, 1, -1), axis=-1)[..., 1:q_max + 1] / n_eta
    power = (256 / jnp.pi**2) * (coefficient * C[:, None, None, None])**2 * (aq.real**2 + aq.imag**2)
    omega = (jnp.mean(de, axis=1) / (2 * jnp.pi) if rotation == "mean_map"
             else jnp.mean(de / tu, axis=1) / jnp.mean(2 * jnp.pi / tu, axis=1))
    pairs = resonance_pairs(p_max, q_max)
    weights = linear_crossing_weights(omega, pairs[:, 0] / pairs[:, 1])
    score = jnp.sum(power[..., pairs[:, 1] - 1] * weights, axis=-1)
    line_length = jnp.mean(jnp.trapezoid(dl[:, :-1] / B[:, :-1], t, axis=-1), axis=1)
    rows = paper_average(score, tu, bc, wb, line_length)
    pitch_mass = paper_average(jnp.ones_like(score), tu, bc, wb, line_length)
    loss = .5 * jnp.sum(rows**2)
    valid &= jnp.all(jnp.isfinite(rows)) & jnp.all(jnp.isfinite(pitch_mass)) & jnp.isfinite(loss)
    rows = jnp.where(valid, rows, jnp.nan)
    return {"delta_res": rows, "loss": jnp.where(valid, loss, jnp.nan), "valid": valid, "full_population": False,
            "action": A, "action_s": A_s, "v_tau": tau, "delta_eta": delta_eta, "omega": omega,
            "harmonic_power": power, "weights": weights, "paper_pitch_band_mass": pitch_mass,
            "t": t, "eta": eta, "well_left": roots[..., :1], "well_right": roots[..., 1:],
            "root_residual": root_residual, "minimum_root_slope": minimum_root_slope,
            "boundary_forbidden_margin": boundary_margin, "root_boundary_distance": root_boundary_distance,
            "flux_valid": flux_valid, "particle_valid": particle_valid, "chart_valid": chart_valid,
            "periodic_valid": periodic_valid, "sampling_valid": sampling_valid,
            "well_count_valid": well_count_valid, "unresolved_interval_count": unresolved_interval_count,
            "field_positive_valid": field_positive_valid,
            "max_trace_phase_step": max_trace_phase_step, "max_eta_harmonic": max_eta_harmonic,
            "family_eta_valid": family_eta_valid,
            "family_radial_valid": family_radial_valid, "closure_valid": closure_valid, "hard_status": hard_status}


def streamed_restricted_action_deltares_from_boozer(*, bcrit, bcrit_weight, **kwargs):
    """Compose the restricted diagnostic with bounded-memory pitch execution.

    Arguments and certificates retain the restricted primitive's semantics.
    Physical pitch nodes and weights are live arrays, shared by every surface
    and eta. A checkpointed one-pitch ``lax.scan`` sums radial rows *before*
    the outer square. Any failed pitch invalidates the whole band; no failed
    population is discarded. Rematerializing the scan body avoids retaining
    the large bounce intermediates for all pitches in a reverse-mode solve.
    This changes execution only, not coverage, normalization or the functional.
    """
    import jax
    bc, wb = jnp.asarray(bcrit), jnp.asarray(bcrit_weight)
    if bc.ndim != 1 or bc.size < 1 or wb.shape != bc.shape:
        raise ValueError("Require matching nonempty physical pitch nodes and weights")
    # Match the primitive's explicit promotion, including mixed-precision
    # coefficients/pitch. A scan carry must not change dtype on its first step.
    coefficients = jnp.asarray(kwargs["bmnc_b"], dtype=jnp.float64)
    def body(carry, pitch):
        rows, mass, valid = carry
        # Zero quadrature weight is legal padding, not absent geometry. Check
        # that pitch's complete well with a unit weight, then apply its live
        # physical quadrature weight outside the primitive's scalar guard.
        out = restricted_action_deltares_from_boozer(
            bcrit=pitch[0][None], bcrit_weight=jnp.ones((1,), dtype=wb.dtype), **kwargs)
        return (rows + pitch[1] * out["delta_res"], mass + pitch[1] * out["paper_pitch_band_mass"],
                valid & out["valid"]), out["valid"]
    zero = jnp.zeros(coefficients.shape[0], dtype=coefficients.dtype)
    weights_valid = jnp.all(jnp.isfinite(wb) & (wb >= 0)) & jnp.any(wb > 0)
    (rows, mass, valid), pitch_valid = jax.lax.scan(
        jax.checkpoint(body),
        (zero, zero, weights_valid), (bc, wb))
    loss = .5 * jnp.sum(rows**2)
    valid &= jnp.all(jnp.isfinite(rows)) & jnp.all(jnp.isfinite(mass)) & jnp.isfinite(loss)
    rows = jnp.where(valid, rows, jnp.nan)
    return {"delta_res": rows, "loss": jnp.where(valid, loss, jnp.nan), "valid": valid,
            "pitch_valid": pitch_valid, "paper_pitch_band_mass": jnp.where(valid, mass, jnp.nan),
            "full_population": False}
