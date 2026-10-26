"""Smooth pwO targets in Boozer radians; toroidal modes include NFP.

The canonical target combines the super-Gaussian of Velasco et al. (2024,
PRL 133, 185101) with the width of Calvo et al. (2025, PRE 112, L023201).
Its bounded periodic union, 1-product(1-exp(-u**(2*p)-v**(2*p))), differs
from the isolated fundamental-domain formula in Fernandez-Pacheco et al.,
arXiv:2601.14886v2 (DESC reference: Zenodo 20390611, CC-BY-4.0). Translated
interiors must be disjoint and omitted image tails below 1e-30. A finite-p fit
does not establish piecewise action invariance.

The mapped target uses Liu et al., arXiv:2603.12139v1, equations (2), (9),
and (10); gamma is mapped alpha, not theta-iota*zeta. Gamma variance is
integrated over a fixed eta cut and divided by B_ref squared. It constrains
only the mapped region. Kernel JIT/JVP/VJP support does not extend the
equilibrium solver's differentiation contract.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from .omnigenity import boozer_spectrum_state

__all__ = ["canonical_pwo_field", "squeezed_pwo_coordinates",
           "canonical_pwo_residual_from_boozer", "mapped_pwo_residual_from_boozer",
           "physical_pwo_action_from_boozer", "PwOActionFamilies", "CanonicalPwOResidual", "MappedPwOResidual"]


def _require(condition, message):
    if not isinstance(condition, jax.core.Tracer) and not bool(np.all(np.asarray(condition))):
        raise ValueError(message)


def _integer(value, minimum, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _canonical_images(theta, zeta, *, iota, nfp, bmin, bmax, t1, t2, p,
                      theta_center, zeta_center, images):
    for name, value, minimum in (("nfp", nfp, 1), ("p", p, 2), ("images", images, 1)):
        _integer(value, minimum, name)
    iota, bmin, bmax, t1, t2 = map(jnp.asarray, (iota, bmin, bmax, t1, t2))
    determinant = 1.0 - t1 * t2
    denominator = iota + t2
    width = jnp.pi * iota * determinant / (nfp * denominator)
    extent_theta = (jnp.pi + jnp.abs(t2 * width)) / jnp.abs(determinant)
    extent_zeta = (jnp.abs(width) + jnp.pi * jnp.abs(t1)) / jnp.abs(determinant)
    margin = 1.0 - jnp.maximum(extent_theta / jnp.pi, extent_zeta * nfp / jnp.pi)
    shifts = jnp.arange(-images, images+1)
    poloidal_shift, toroidal_shift = jnp.meshgrid(shifts, shifts, indexing="ij")
    lattice_shape = (-1,) + (1,)*width.ndim
    j = poloidal_shift.reshape(lattice_shape)
    k = toroidal_shift.reshape(lattice_shape)
    separation = jnp.maximum(jnp.abs(2*jnp.pi*(t1*j+k/nfp)/width),
                             jnp.abs(2*j+2*t2*k/nfp))
    separation = jnp.min(jnp.where((j != 0) | (k != 0), separation, jnp.inf), axis=0)
    extent = 1-margin
    # If ||A*l||inf <= separation, |l_i| <= separation*rowsum(|A^-1|)_i.
    # Thus a missing index >= images+1 cannot improve the enumerated minimum.
    lattice_certified = separation*extent < 2*(images+1)
    lattice_margin = 1-2/separation
    q = 2*p
    tail_exponent = jnp.minimum((2*images+1)/extent, 750.0**(1/q))**q
    slope = 2*q*tail_exponent/(2*images+1)
    decay = jnp.exp(-slope)
    denominator_tail = -jnp.expm1(-slope)
    # Shell k contributes <= 8*k*exp(-((2*k-1)/extent)**q).
    # Convexity bounds the exponent by a tangent; sum the geometric tail.
    log_tail_bound = (jnp.log(8.0)-tail_exponent
                      + jnp.log((images+1)/denominator_tail+decay/denominator_tail**2))
    valid = (jnp.isfinite(width) & (jnp.abs(width) > 1e-12)
             & (jnp.abs(determinant) > 1e-12) & (jnp.abs(iota) > 1e-12)
             & jnp.isfinite(bmin) & jnp.isfinite(bmax) & (bmin > 0) & (bmax > bmin)
             & jnp.isfinite(margin) & lattice_certified & (lattice_margin > 0)
             & (tail_exponent >= 80) & jnp.isfinite(log_tail_bound)
             & (log_tail_bound < jnp.log(1e-30)))
    zeta_center = jnp.pi / nfp if zeta_center is None else zeta_center
    theta = jnp.asarray(theta, dtype=jnp.float64)
    zeta = jnp.asarray(zeta, dtype=jnp.float64)
    valid = (valid & jnp.isfinite(theta) & jnp.isfinite(zeta)
             & jnp.isfinite(theta_center) & jnp.isfinite(zeta_center))
    _require(valid, "canonical target needs finite coordinates, positive distinct B bounds, disjoint torus parallelograms and a certified image stencil")
    # Wrapping chooses a nearby stencil, not a single discontinuous patch.
    dt = jnp.mod(theta - theta_center + jnp.pi, 2*jnp.pi) - jnp.pi
    dz = jnp.mod(zeta - zeta_center + jnp.pi/nfp, 2*jnp.pi/nfp) - jnp.pi/nfp
    dt, dz, _ = jnp.broadcast_arrays(dt, dz, width)
    shape = (-1,) + (1,)*dt.ndim
    t = dt + 2*jnp.pi*poloidal_shift.reshape(shape)
    z = dz + 2*jnp.pi*toroidal_shift.reshape(shape)/nfp
    # Saturating below floating-point underflow also keeps high-p tangents
    # finite when a remote image's raw even power would overflow.
    limit = 750.0**(1.0/(2*p))
    u = jnp.minimum(jnp.abs((z+t1*t)/width), limit)
    v = jnp.minimum(jnp.abs((t+t2*z)/jnp.pi), limit)
    return jnp.exp(-u**(2*p)-v**(2*p)), width, margin, lattice_margin, log_tail_bound, valid


def canonical_pwo_field(theta, zeta, *, iota, nfp, bmin, bmax, t1, t2, p=2,
                        theta_center=0.0, zeta_center=None, images=2):
    """Return a bounded periodic super-Gaussian union and overlap diagnostics.

    All continuous arguments broadcast. For surface arrays use shape
    ``(nsurface,1,1)`` against angular meshes. ``nfp``, ``p``, and ``images``
    are static integers. Invalid eager data raise; traced invalid data return
    ``valid=False`` and NaN field values. ``bmin`` is a lower bound: finite-p
    tails generally keep the actual minimum above it. ``corner_margin`` measures
    centered-cell containment; ``lattice_margin`` measures torus separation.
    """
    patches, width, margin, lattice_margin, log_tail_bound, valid = _canonical_images(
        theta, zeta, iota=iota, nfp=nfp, bmin=bmin, bmax=bmax, t1=t1, t2=t2,
        p=p, theta_center=theta_center, zeta_center=zeta_center, images=images)
    union = 1.0 - jnp.prod(1.0 - patches, axis=0)
    field = jnp.asarray(bmin) + (jnp.asarray(bmax)-jnp.asarray(bmin))*union
    return {"B_target": jnp.where(valid, field, jnp.nan), "width": width,
            "corner_margin": margin, "valid": valid,
            "lattice_margin": lattice_margin, "omitted_image_log_bound": log_tail_bound,
            "overlap": jnp.sum(patches, axis=0)-jnp.max(patches, axis=0)}


def squeezed_pwo_coordinates(gamma, eta, *, iota, nfp, a=0.3, b=0.0, c=-2.0):
    """Evaluate the signed-iota LC mapping and its one-sided eta Jacobian.

    Broadcast inputs; the returned ``jacobian`` is d(zeta_B)/d(eta).
    The mapping joins continuously at eta=pi but can have unequal one-sided
    derivatives there. The caller chooses a fixed nonfolding cut domain.
    """
    _integer(nfp, 1, "nfp")
    gamma, eta = jnp.asarray(gamma), jnp.asarray(eta)
    _require(jnp.isfinite(gamma) & jnp.isfinite(eta) & (eta >= 0) & (eta <= 2*jnp.pi)
             & jnp.isfinite(iota) & jnp.isfinite(a) & jnp.isfinite(b) & jnp.isfinite(c),
             "mapping inputs must be finite with 0 <= eta <= 2*pi")
    left = eta <= jnp.pi
    x = jnp.where(left, eta, 2*jnp.pi-eta)
    distance = c*x*x - (1+c*jnp.pi)*x + jnp.pi
    derivative = 2*c*x - (1+c*jnp.pi)
    y = jnp.where(left, gamma, -gamma) + iota*distance/nfp
    h = y+b*jnp.sin(y)
    shape = a*x*jnp.sin(h)
    shape_derivative = a*jnp.sin(h) + a*x*jnp.cos(h)*(1+b*jnp.cos(y))*iota*derivative/nfp
    zeta = (jnp.pi+jnp.where(left, -1.0, 1.0)*(shape+distance))/nfp
    return {"theta": jnp.broadcast_to(gamma, zeta.shape), "zeta": zeta,
            "jacobian": -(shape_derivative+derivative)/nfp}


def _spectrum(bmnc_b, bmns_b, xm_b, xn_b, iota_b, nfp, weights):
    _integer(nfp, 1, "nfp")
    cosine = jnp.asarray(bmnc_b, dtype=jnp.float64)
    sine = jnp.zeros_like(cosine) if bmns_b is None else jnp.asarray(bmns_b, dtype=jnp.float64)
    m, n = jnp.asarray(xm_b), jnp.asarray(xn_b)
    iota = jnp.atleast_1d(jnp.asarray(iota_b, dtype=jnp.float64))
    if (cosine.ndim != 2 or sine.shape != cosine.shape or m.shape != (cosine.shape[1],)
            or n.shape != m.shape or iota.shape != (cosine.shape[0],) or cosine.shape[0] == 0):
        raise ValueError("spectra need (nsurface,nmode), modes (nmode,), and iota (nsurface,)")
    weight = jnp.ones(cosine.shape[0]) if weights is None else jnp.asarray(weights, dtype=jnp.float64)
    if weight.shape != (cosine.shape[0],):
        raise ValueError("weights must have one entry per surface")
    valid = (jnp.all(jnp.isfinite(cosine), axis=1) & jnp.all(jnp.isfinite(sine), axis=1)
             & jnp.isfinite(iota) & jnp.isfinite(weight) & (weight >= 0)
             & jnp.all(jnp.isfinite(m)) & jnp.all(jnp.isfinite(n))
             & jnp.all(m == jnp.round(m)) & jnp.all(n/nfp == jnp.round(n/nfp)))
    scale = jnp.max(weight)
    valid = valid & (scale > 0) & jnp.isfinite(jnp.sum(weight))
    _require(valid, "spectra, modes, iota and nonnegative nonzero weights must be finite and periodic")
    weight = weight/jnp.where(scale > 0, scale, 1.0)
    total_weight = jnp.sum(weight)
    valid = valid & jnp.isfinite(total_weight) & (total_weight > 0)
    _require(valid, "surface weights must admit finite positive normalization")
    return cosine, sine, m, n, iota, weight/jnp.where(total_weight > 0, total_weight, 1.0), valid


def _evaluate(cosine, sine, m, n, theta, zeta):
    phase = theta[..., None]*m - zeta[..., None]*n
    return jnp.sum(cosine[:, None, None, :]*jnp.cos(phase)
                   + sine[:, None, None, :]*jnp.sin(phase), axis=-1)


def _surface(value, nsurface, name):
    value = jnp.asarray(value, dtype=jnp.float64)
    if value.ndim > 1 or (value.ndim == 1 and value.shape != (nsurface,)):
        raise ValueError(f"{name} must be scalar or have one entry per surface")
    return jnp.broadcast_to(value, (nsurface,))[:, None, None]


def _finish(residual, B, valid, weight):
    valid = (valid & jnp.isfinite(weight) & (jnp.sum(weight) > 0)
             & jnp.all(jnp.isfinite(B) & (B > 0) & jnp.isfinite(residual), axis=(1, 2)))
    _require(valid, "field or target/map is invalid")
    scaled = jnp.where(valid[:, None, None], residual*jnp.sqrt(weight[:, None, None]/residual.shape[1]/residual.shape[2]), jnp.nan)
    flat = scaled.ravel()
    low, high = jnp.min(B, axis=(1, 2)), jnp.max(B, axis=(1, 2))
    return {"residuals1d": flat, "total": jnp.sum(flat*flat), "valid": valid,
            "mirror": (high-low)/(high+low), "B_min": low, "B_max": high}


def canonical_pwo_residual_from_boozer(*, bmnc_b, xm_b, xn_b, iota_b, nfp,
        bmin, bmax, bmns_b=None, t1=-0.3, t2=3.0, p=2, theta_center=0.0,
        zeta_center=None, images=2, minimum_corner_margin=None, minimum_lattice_margin=0.05, ntheta=32,
        nzeta=32, weights=None):
    """Normalized pointwise target fit with live signed iota and both parities.

    Surface weights sum to one; angular quadrature is uniform and normalized.
    Returns sampled ``B``, ``B_target``, overlap, margins, validity and extrema.
    ``minimum_lattice_margin`` prevents touching translated parallelograms.
    ``minimum_corner_margin`` optionally restores the older centered-cell
    containment restriction; by default a cut-crossing target is allowed.
    """
    _integer(ntheta, 4, "ntheta"); _integer(nzeta, 4, "nzeta")
    if minimum_corner_margin is not None and not 0 <= minimum_corner_margin < 1:
        raise ValueError("minimum_corner_margin must lie in [0,1)")
    if not 0 <= minimum_lattice_margin < 1:
        raise ValueError("minimum_lattice_margin must lie in [0,1)")
    cosine, sine, m, n, iota, weight, valid = _spectrum(bmnc_b, bmns_b, xm_b, xn_b, iota_b, nfp, weights)
    theta = 2*jnp.pi*jnp.arange(ntheta)[None, :, None]/ntheta
    zeta = 2*jnp.pi*jnp.arange(nzeta)[None, None, :]/(nfp*nzeta)
    B = _evaluate(cosine, sine, m, n, theta, zeta)
    params = {key: _surface(value, cosine.shape[0], key) for key, value in
              (("bmin", bmin), ("bmax", bmax), ("t1", t1), ("t2", t2),
               ("theta_center", theta_center))}
    if zeta_center is not None:
        params["zeta_center"] = _surface(zeta_center, cosine.shape[0], "zeta_center")
    target = canonical_pwo_field(theta, zeta, iota=iota[:, None, None], nfp=nfp,
                                p=p, images=images, **params)
    valid = valid & jnp.all(target["valid"], axis=(1, 2)) & (target["lattice_margin"][:, 0, 0] >= minimum_lattice_margin)
    if minimum_corner_margin is not None:
        valid = valid & (target["corner_margin"][:, 0, 0] >= minimum_corner_margin)
    out = _finish((B-target["B_target"])/target["B_target"], B, valid, weight)
    out.update({**target, "valid": out["valid"], "B": B, "theta": theta[0, :, 0], "zeta": zeta[0, 0, :]})
    return out


def mapped_pwo_residual_from_boozer(*, bmnc_b, xm_b, xn_b, iota_b, nfp,
        b_ref, bmns_b=None, a=0.3, b=0.0, c=-2.0, eta_min=2.8,
        ngamma=32, neta=32, weights=None):
    """Mapped-gamma variance over a fixed eta interval, divided by B_ref squared.

    Eta midpoint quadrature covers [eta_min,2*pi-eta_min]. Gamma is uniform
    and endpoint-free. Discrete Parseval in gamma makes this the power of all
    nonconstant gamma harmonics, averaged over eta, divided by B_ref squared.
    No Fourier periodicity in the cut eta direction is assumed.
    """
    _integer(ngamma, 4, "ngamma"); _integer(neta, 4, "neta")
    if not np.isfinite(eta_min) or not 0 < eta_min < np.pi:
        raise ValueError("eta_min must be fixed strictly between zero and pi")
    cosine, sine, m, n, iota, weight, valid = _spectrum(bmnc_b, bmns_b, xm_b, xn_b, iota_b, nfp, weights)
    ns = cosine.shape[0]
    a, b, c, reference = (_surface(x, ns, name) for name, x in
                           (("a", a), ("b", b), ("c", c), ("b_ref", b_ref)))
    gamma = 2*jnp.pi*jnp.arange(ngamma)[None, :, None]/ngamma
    eta = eta_min+(2*jnp.pi-2*eta_min)*(jnp.arange(neta)[None, None, :]+0.5)/neta
    coordinates = squeezed_pwo_coordinates(gamma, eta, iota=iota[:, None, None], nfp=nfp, a=a, b=b, c=c)
    B = _evaluate(cosine, sine, m, n, coordinates["theta"], coordinates["zeta"])
    # Sufficient global bound over the entire cut domain, including between nodes.
    bound = ((1-jnp.abs(a)*jnp.pi*jnp.abs(iota[:, None, None]/nfp)*(1+jnp.abs(b)))
             * (1+c*jnp.pi-2*c*eta_min)-jnp.abs(a))/nfp
    distance_at_cut = c*eta_min**2-(1+c*jnp.pi)*eta_min+jnp.pi
    # Positive local Jacobian is insufficient on a torus: a lifted interval
    # longer than one field period would cover physical points twice.
    span_upper = 2*(distance_at_cut+jnp.abs(a)*eta_min*jnp.minimum(
        1.0, jnp.abs(iota[:, None, None]*distance_at_cut/nfp)*(1+jnp.abs(b))))/nfp
    valid_map = (jnp.isfinite(a) & jnp.isfinite(b) & jnp.isfinite(c)
                 & (c < -1/jnp.pi) & (eta_min >= jnp.pi+1/c)
                 & jnp.isfinite(reference) & (reference > 0) & (bound > 0)
                 & (distance_at_cut > 0) & (span_upper < 2*jnp.pi/nfp))
    valid = valid & valid_map[:, 0, 0] & jnp.all(coordinates["jacobian"] > 0, axis=(1, 2))
    mean = jnp.mean(B, axis=1, keepdims=True)
    out = _finish((B-mean)/reference, B, valid, weight)
    out.update({"B_mapped": B, "B_mean_gamma": mean, "gamma": gamma[0, :, 0],
                "eta": eta[0, 0, :], "jacobian": coordinates["jacobian"],
                "jacobian_lower_bound": bound[:, 0, 0],
                "toroidal_span_upper": span_upper[:, 0, 0],
                "eta_fraction": 1-eta_min/jnp.pi,
                "mapped_area_fraction": jnp.mean(coordinates["jacobian"], axis=(1, 2))*(2*jnp.pi-2*eta_min)*nfp/(2*jnp.pi)})
    return out


def _pwo_action_kernel(*, bmnc_b, xm_b, xn_b, iota_b, G_b, I_b, nfp, bmns_b=None):
    from types import SimpleNamespace
    from scipy.integrate import quad

    _integer(nfp, 1, "nfp")
    c, m, n = (np.asarray(v, dtype=float) for v in (bmnc_b, xm_b, xn_b))
    if c.ndim == 2 and c.shape[0] == 1:
        c = c[0]
    s = np.zeros_like(c) if bmns_b is None else np.asarray(bmns_b, dtype=float)
    if s.ndim == 2 and s.shape[0] == 1:
        s = s[0]
    if c.ndim != 1 or not c.size or any(v.shape != c.shape for v in (s, m, n)):
        raise ValueError("one surface needs coefficient vectors or rows and matching mode vectors")
    scalars = [np.asarray(v, dtype=float) for v in (iota_b, G_b, I_b)]
    if any(v.shape not in ((), (1,)) for v in scalars):
        raise ValueError("iota_b, G_b and I_b must each contain one surface value")
    iota, current_G, current_I = (float(v.reshape(-1)[0]) for v in scalars)
    factor = abs(current_G+iota*current_I); freq = m*iota-n; period = 2*np.pi/nfp
    if (not all(np.all(np.isfinite(v)) for v in (c, s, m, n, *scalars, freq))
            or np.any(m != np.round(m)) or np.any(n/nfp != np.round(n/nfp))
            or not np.isfinite(factor) or factor <= 0):
        raise ValueError("finite periodic spectra and nonzero |G+iota*I| are required")
    def field(z, alpha):
        phase = m*alpha+freq*z; co, si = np.cos(phase), np.sin(phase)
        return float(c@co+s@si), float((-m*c)@si+(m*s)@co), float((-freq*c)@si+(freq*s)@co)
    def integral(alpha, left, right, level, derivative, extrema=()):
        width = .5*(right-left)
        phase = m*alpha+np.array([left, right])[:, None]*freq
        co, si = np.cos(phase), np.sin(phase)
        A, D = c*co+s*si, -c*si+s*co
        residual = level-A.sum(axis=1)
        mA, mD = m*A, m*D
        def fun(t):
            side = int(t > 0)
            distance = (2*width*np.sin((t+np.pi/2)/2)**2 if side == 0
                        else -2*width*np.sin((np.pi/2-t)/2)**2)
            sh, ch = np.sin(freq*distance/2), np.cos(freq*distance/2)
            # Retain the root residual; avoid subtracting nearly equal field values.
            sn, cm1 = 2*sh*ch, -2*sh*sh
            gap = residual[side]-A[side]@cm1-D[side]@sn
            B = level-gap; par = gap/level
            Ba = -mA[side]@sn+mD[side]@(1+cm1) if derivative else 0.
            if B <= 0 or par <= 0:
                raise ValueError("Nonpositive interior field or parallel energy")
            value = -Ba*(1-B/(2*level))/(B*B*np.sqrt(par)) if derivative else np.sqrt(par)/B
            return 2*factor*value*width*np.cos(t)
        points = [np.arcsin((z-.5*(left+right))/width) for z in extrema if left < z < right]
        out = quad(fun, -np.pi/2, np.pi/2, points=points, epsabs=1e-9, epsrel=1e-8, limit=150, full_output=1)
        if len(out) > 3:
            raise ValueError(out[3])
        return out[:2]
    return SimpleNamespace(field=field, integral=integral, coefficients=(c, s, m, n),
                           iota=iota, frequency=freq, period=period)


def physical_pwo_action_from_boozer(*, bmnc_b, xm_b, xn_b, iota_b, G_b, I_b,
        nfp, levels, alphas, bmns_b=None, reference=None, points_per_period=1024,
        check_roots=False, match_critical_census=False, max_periods=32):
    """Host NumPy/SciPy fixed-alpha physical action and action-slope diagnostic.

    Accept one surface, coefficient vectors or rows, scalar iota/G/I, physical
    Tesla reflecting levels, and alpha=theta-iota*zeta in radians. Toroidal
    modes include NFP. J=2|G+iota*I| integral sqrt(1-B/level)/B dzeta is in m;
    dJ/dalpha is in m/rad. Components average squared slopes per well, then
    equally across pitches. ``relative_Jalpha2`` uses each current J;
    ``fixed_relative_Jalpha2`` uses the initial ``reference_J`` carried through
    well matches (legacy references initialize it from J). Compare both to
    distinguish slope reduction from a changing action denominator.
    Select left bounces in [0,1) field period. Search from -4 to 8 periods,
    doubling the right extent up to max_periods to complete long wells.
    Stationary points precede bounce isolation to retain narrow barriers.
    Finite sign-change scans (optionally checked at half resolution) do not
    prove exhaustive root isolation or within-family pwO invariance. This
    forward diagnostic is not JAX differentiable. Reference matching requires
    identical grids/counts and overlapping wells with endpoint shifts <20%.
    With a reference, follow its wells across the sampling seam; reject
    splitting or merging that changes the number of overlapping wells.
    """
    from scipy.optimize import brentq, linear_sum_assignment

    _integer(nfp, 1, "nfp"); _integer(points_per_period, 8, "points_per_period")
    _integer(max_periods, 8, "max_periods")
    kernel = _pwo_action_kernel(bmnc_b=bmnc_b, bmns_b=bmns_b, xm_b=xm_b, xn_b=xn_b,
                               iota_b=iota_b, G_b=G_b, I_b=I_b, nfp=nfp)
    c, s, m, n = kernel.coefficients
    freq, period = kernel.frequency, kernel.period
    field, integral = kernel.field, kernel.integral
    levels, alphas = np.asarray(levels, dtype=float), np.asarray(alphas, dtype=float)
    if any(v.ndim != 1 or not v.size or not np.all(np.isfinite(v)) for v in (levels, alphas)) or np.any(levels <= 0):
        raise ValueError("levels and alphas must be finite nonempty vectors with positive levels")
    def root_sets(alpha, ppp, end, samples=None):
        if samples is None:
            z = np.linspace(-4*period, end*period, (end+4)*ppp+1); sample = np.zeros_like(z)
            for k in range(0, len(c), 64):
                phase = m[k:k+64, None]*alpha+freq[k:k+64, None]*z
                sample += (-freq[k:k+64]*c[k:k+64])@np.sin(phase)+(freq[k:k+64]*s[k:k+64])@np.cos(phase)
        else:
            z, sample = samples
        idx = np.flatnonzero(sample[:-1]*sample[1:] < 0)
        critical = np.unique([brentq(lambda x: field(x, alpha)[2], z[j], z[j+1], xtol=1e-13) for j in idx]+z[sample == 0].tolist())
        boundaries = np.unique(np.r_[z[0], critical, z[-1]])
        B = np.asarray([field(x, alpha)[0] for x in boundaries])
        if not np.all(np.isfinite(B)) or np.any(B <= 0):
            raise ValueError("B must remain finite and positive along the searched field line")
        result = []
        for level in levels:
            if np.any(abs(B[1:-1]-level) < 1e-11):
                raise ValueError("Reflecting level near a stationary-point tangency")
            idx = np.flatnonzero((B[:-1]-level)*(B[1:]-level) < 0)
            result.append(np.unique([brentq(lambda x: field(x, alpha)[0]-level, boundaries[j], boundaries[j+1], xtol=1e-13) for j in idx]+boundaries[B == level].tolist()))
        return result, critical, (z, sample)
    rows, checks, critical_census = [], [], []
    for ai, alpha in enumerate(alphas):
        tracked = [[r for r in reference["rows"] if (r["alpha_index"], r["pitch_index"]) == (ai, ip)]
                   for ip in range(len(levels))] if reference is not None else None
        def selected(left, right, ip):
            if tracked is None or not tracked[ip]:
                return 0 <= left < period
            return any(min(right, r["right"]) > max(left, r["left"]) for r in tracked[ip])
        end = 8
        while True:
            fine, critical, samples = root_sets(alpha, points_per_period, end)
            incomplete = any(len(r) and selected(r[-1], np.inf, ip) and field(r[-1], alpha)[2] < 0
                             for ip, r in enumerate(fine))
            if not incomplete:
                break
            if end == max_periods:
                raise ValueError(f"Incomplete well at alpha {ai} after {end} field periods")
            end = min(2*end, max_periods)
        critical_census.append(dict(alpha_index=ai, count=len(critical), positions_rad=critical.tolist(),
                                    right_extent_periods=end))
        if check_roots:
            # Nested grids share field evaluations, but isolate their roots independently.
            coarse_samples = tuple(v[::2] for v in samples) if points_per_period % 2 == 0 else None
            coarse, oldcritical, _ = root_sets(alpha, points_per_period//2, end, coarse_samples)
            if len(oldcritical) != len(critical) or np.max(abs(oldcritical-critical), initial=0) > 1e-9:
                raise ValueError(f"Critical-point refinement disagreement at {ai}")
            critical_census[-1]["coarse_fine_maximum_discrepancy_rad"] = float(np.max(abs(oldcritical-critical), initial=0))
            for ip, (a, b) in enumerate(zip(coarse, fine)):
                if len(a) != len(b) or np.max(abs(a-b), initial=0) > 1e-9:
                    raise ValueError(f"Root refinement disagreement at {(ai, ip)}")
                checks.append(dict(alpha_index=ai, pitch_index=ip, roots=len(b), maximum_discrepancy_rad=float(np.max(abs(a-b), initial=0))))
        for ip, (level, roots) in enumerate(zip(levels, fine)):
            for j, left in enumerate(roots):
                if j+1 == len(roots) or field(left, alpha)[2] > 0:
                    continue
                right = roots[j+1]
                if not selected(left, right, ip) or field(.5*(left+right), alpha)[0] >= level:
                    continue
                if min(abs(field(z, alpha)[2]) for z in (left, right)) <= 1e-8:
                    raise ValueError("Marginal bounce root")
                J, Je = integral(alpha, left, right, level, False, extrema=critical)
                dJ, de = integral(alpha, left, right, level, True, extrema=critical)
                if not np.isfinite(J+dJ+Je+de) or J <= 0:
                    raise ValueError("Invalid action or derivative")
                rows.append(dict(alpha_index=ai, pitch_index=ip, left=left, right=right, J=J, dJ=dJ, J_error=Je, dJ_error=de, reference_J=J))
    counts = [[sum(r["alpha_index"] == ai and r["pitch_index"] == ip for r in rows) for ip in range(len(levels))] for ai in range(len(alphas))]
    matches = []
    if reference is not None:
        if reference["levels"] != levels.tolist() or reference["alphas"] != alphas.tolist():
            raise ValueError("Reference grid differs")
        if match_critical_census and [r["count"] for r in reference["critical_census"]] != [r["count"] for r in critical_census]:
            raise ValueError("Candidate stationary-point counts differ")
        if reference["counts"] != counts:
            raise ValueError("Candidate alpha/pitch well counts differ")
        for ai in range(len(alphas)):
            for ip in range(len(levels)):
                aa, bb = ([r for r in rr if (r["alpha_index"], r["pitch_index"]) == (ai, ip)] for rr in (reference["rows"], rows))
                if len(aa) != len(bb):
                    raise ValueError("Reference rows disagree with well counts")
                if not aa:
                    continue
                cost = np.array([[max(abs(x[k]-y[k]) for k in ("left", "right"))/min(x["right"]-x["left"], y["right"]-y["left"]) if min(x["right"], y["right"]) > max(x["left"], y["left"]) else 1e6 for y in bb] for x in aa])
                for i, j in zip(*linear_sum_assignment(cost)):
                    if cost[i, j] >= .2:
                        raise ValueError(f"Unmatched candidate well at {(ai, ip)}")
                    anchor = aa[i].get("reference_J", aa[i]["J"])
                    if not np.isfinite(anchor) or anchor <= 0:
                        raise ValueError("Reference actions must be finite and positive")
                    bb[j]["reference_J"] = float(anchor)
                    matches.append(dict(alpha_index=ai, pitch_index=ip, maximum_relative_endpoint_shift=float(cost[i, j])))
    per_pitch = []
    for ip in range(len(levels)):
        rr = [r for r in rows if r["pitch_index"] == ip]
        if not rr:
            raise ValueError(f"No complete wells at pitch {ip}")
        per_pitch.append(dict(pitch_index=ip, wells=len(rr), raw_Jalpha2=float(np.mean([r["dJ"]**2 for r in rr])), relative_Jalpha2=float(np.mean([(r["dJ"]/r["J"])**2 for r in rr])),
                              fixed_relative_Jalpha2=float(np.mean([(r["dJ"]/r["reference_J"])**2 for r in rr]))))
    return dict(levels=levels.tolist(), alphas=alphas.tolist(), rows=rows, counts=counts, per_pitch=per_pitch,
                components={k: float(np.mean([p[k] for p in per_pitch])) for k in ("raw_Jalpha2", "relative_Jalpha2", "fixed_relative_Jalpha2")},
                root_checks=checks, critical_census=critical_census, matches=matches, family_invariance_qualified=False)


class PwOActionFamilies:
    """Local host-side action variance on explicitly supplied orbit charts.

    Each chart supplies ``level_T``, two unwrapped ``endpoints`` [theta, zeta]
    at nondegenerate tangencies, a positive fixed ``weight``, and ``rows``
    containing u, weight, left and right. Nodes satisfy 0<u<1; their positive
    weights sum to one. Chart weights include the reference alpha span and
    pitch weight. Periodic copies must be resolved by the caller.

    Event shifts are limited to 0.2 rad, span ratios to (0.25, 4), and
    root brackets to 20% of each reference well width. These are local
    numerical trust limits, not universal pwO constraints.

    Track those tangencies and bounce roots on trial spectra. Reference chart
    weights, action means and length remain fixed. Returned variance and raw
    slopes describe these charts only: no automatic family discovery, global
    topology proof, JAX differentiation or pwO qualification is provided.
    """

    def __init__(self, reference, charts, *, length):
        from copy import deepcopy
        self.reference, self.charts = deepcopy(reference), deepcopy(charts)
        self.length = float(length)
        self.base = self._kernel(self.reference)
        if not np.isfinite(self.length) or self.length <= 0 or not self.charts:
            raise ValueError("Positive reference length and nonempty orbit charts required")
        for chart in self.charts:
            q = np.asarray(chart["endpoints"], dtype=float)
            level, weight = chart["level_T"], chart["weight"]
            if q.shape != (2, 2) or not np.all(np.isfinite(q)) or not np.isfinite(level+weight) or min(level, weight) <= 0:
                raise ValueError("Invalid chart endpoints, level or weight")
            domain = q[:, 0]-self.base.iota*q[:, 1]
            if domain[1] <= domain[0]:
                raise ValueError("Chart endpoints must bound an increasing alpha interval")
            chart["endpoints"], chart["reference_domain"] = q, domain
            determinants = []
            for point in q:
                value, jac = self._event(self.base, level, point)
                determinant = float(np.linalg.det(jac))
                if np.linalg.norm(value) > 1e-8 or not np.isfinite(determinant) or abs(determinant) <= 1e-10:
                    raise ValueError("Chart endpoint is not a regular tangency")
                determinants.append(determinant)
            chart["determinants"] = determinants
            nodes = np.asarray([[r[k] for k in ("u", "weight", "left", "right")] for r in chart["rows"]], dtype=float)
            if (nodes.ndim != 2 or nodes.shape[1:] != (4,) or len(nodes) < 3 or not np.all(np.isfinite(nodes))
                    or np.any(nodes[:, :2] <= 0) or np.any(nodes[:, 0] >= 1)
                    or np.any(np.diff(nodes[:, 0]) <= 0) or abs(nodes[:, 1].sum()-1) > 1e-12
                    or np.any(nodes[:, 3] <= nodes[:, 2])):
                raise ValueError("Invalid positive quadrature or reference wells")
            chart["nodes"] = nodes
        self.weights = np.array([c["weight"] for c in self.charts], dtype=float)
        total = self.weights.sum()
        if not np.isfinite(total):
            raise ValueError("Nonfinite total family weight")
        self.weights /= total
        self.initial = self(self.reference)

    @staticmethod
    def _kernel(spectrum):
        keys = ("bmnc_b", "bmns_b", "xm_b", "xn_b", "iota_b", "G_b", "I_b", "nfp")
        return _pwo_action_kernel(**{k: spectrum[k] for k in keys if k in spectrum})

    @staticmethod
    def _event(kernel, level, q):
        c, s, m, n = kernel.coefficients
        phase = m*q[0]-n*q[1]; co, si = np.cos(phase), np.sin(phase)
        derivative, curvature = -c*si+s*co, -c*co-s*si
        f = kernel.frequency
        return (np.array([c@co+s@si-level, f@derivative]),
                np.array([[m@derivative, -n@derivative], [(f*m)@curvature, -(f*n)@curvature]]))

    def __call__(self, spectrum):
        """Rescore the same families, or raise when local continuation is unresolved."""
        from scipy.optimize import brentq, root
        kernel = self._kernel(spectrum)
        if any(not np.array_equal(spectrum.get(k), self.reference.get(k)) for k in ("nfp", "xm_b", "xn_b", "s_b")):
            raise ValueError("Family continuation requires the same surface and Fourier basis")
        field, output, ordering = kernel.field, [], {}
        for index, chart in enumerate(self.charts):
            level = chart["level_T"]; endpoints = []
            for old, determinant in zip(chart["endpoints"], chart["determinants"]):
                fit = root(lambda q: self._event(kernel, level, q), old, jac=True, tol=1e-11)
                residual, jac = self._event(kernel, level, fit.x)
                newdet = float(np.linalg.det(jac))
                if (not np.all(np.isfinite(fit.x)) or not np.isfinite(newdet) or np.linalg.norm(residual) > 1e-8
                        or newdet*determinant <= 0 or abs(newdet) < .01*abs(determinant)
                        or np.linalg.norm(fit.x-old) > .2):
                    raise ValueError("Lost tangency or exceeded local event trust range")
                endpoints.append(fit.x)
            endpoints = np.asarray(endpoints)
            a, b = endpoints[:, 0]-kernel.iota*endpoints[:, 1]
            old_a, old_b = chart["reference_domain"]
            if not .25 < (b-a)/(old_b-old_a) < 4:
                raise ValueError("Family span left local trust range")
            ordering.setdefault(level, []).extend(zip((old_a, old_b), (a, b)))
            shifts = endpoints[:, 1]-chart["endpoints"][:, 1]
            rows = []
            for u, weight, left, right in chart["nodes"]:
                alpha = a+(b-a)*u; old_alpha = old_a+(old_b-old_a)*u
                width = right-left; predictor = (1-u)*shifts[0]+u*shifts[1]; roots = []
                for z in (left, right):
                    slope = self.base.field(z, old_alpha)[2]
                    if abs(self.base.field(z, old_alpha)[0]-level) > 1e-9 or abs(slope) <= 1e-8:
                        raise ValueError("Reference bounce is not regular")
                    center = z+predictor; mismatch = field(center, alpha)[0]-level
                    radius = min(.2*width, max(width*1e-5, 4*abs(mismatch/slope)))
                    while True:
                        grid = np.linspace(center-radius, center+radius, 33)
                        values = np.array([field(v, alpha)[0]-level for v in grid])
                        cross = np.flatnonzero(values[:-1]*values[1:] < 0)
                        exact = grid[values == 0]
                        if len(cross)+len(exact) == 1:
                            z = float(exact[0]) if len(exact) else brentq(lambda v: field(v, alpha)[0]-level, grid[cross[0]], grid[cross[0]+1], xtol=1e-13)
                            break
                        if len(cross)+len(exact) > 1 or radius >= .2*width:
                            raise ValueError("Ambiguous or lost family bounce root")
                        radius = min(.2*width, 2*radius)
                    roots.append(z)
                left, right = roots
                if right <= left or field(left, alpha)[2] >= -1e-8 or field(right, alpha)[2] <= 1e-8:
                    raise ValueError("Invalid continued well orientation")
                grid = np.linspace(left, right, max(65, int(np.ceil((right-left)*max(abs(kernel.frequency))*16/np.pi))))
                slopes = np.array([field(z, alpha)[2] for z in grid])
                cross = np.flatnonzero(slopes[:-1]*slopes[1:] < 0)
                extrema = [brentq(lambda z: field(z, alpha)[2], grid[k], grid[k+1]) for k in cross]
                if max(field(z, alpha)[0] for z in [(left+right)/2, *extrema]) >= level:
                    raise ValueError("Interior reflecting barrier in continued family")
                J, error = kernel.integral(alpha, left, right, level, False, extrema=extrema)
                dJ, derivative_error = kernel.integral(alpha, left, right, level, True, extrema=extrema)
                if not np.all(np.isfinite([J, dJ, error, derivative_error])) or J <= 0:
                    raise ValueError("Invalid family action quadrature")
                rows.append(dict(u=u, weight=weight, alpha=alpha, left=left, right=right, J=J, dJ=dJ))
            weights = np.array([r["weight"] for r in rows]); actions = np.array([r["J"] for r in rows])
            mean = float(weights@actions); variance = float(weights@((actions-mean)**2))
            anchor = self.initial["families"][index]["mean_action_m"] if hasattr(self, "initial") else mean
            output.append(dict(mean_action_m=mean, variance_m2=variance, fixed_relative_variance=variance/anchor**2,
                               raw_Jalpha2=float(weights@np.array([r["dJ"]**2 for r in rows])),
                               alpha_domain=[a, b], endpoints=endpoints.tolist(), rows=rows))
        for pairs in ordering.values():
            pairs.sort()
            if any(y[0]-x[0] > 1e-6 and y[1] <= x[1] for x, y in zip(pairs, pairs[1:])):
                raise ValueError("Tracked tangency ordering changed")
        weights = self.weights
        return dict(value=float(weights@np.array([f["variance_m2"] for f in output]))/self.length**2,
                    raw_Jalpha2=float(weights@np.array([f["raw_Jalpha2"] for f in output])),
                    fixed_relative_variance=float(weights@np.array([f["fixed_relative_variance"] for f in output])),
                    families=output, family_invariance_qualified=False, global_topology_proof=False)


class _PwOResidual:
    def __init__(self, surfaces, *, weights=None, mboz=16, nboz=16, oversample=2, **options):
        self.surfaces = np.atleast_1d(np.asarray(surfaces, dtype=float))
        if self.surfaces.ndim != 1 or self.surfaces.size == 0 or not np.all(np.isfinite(self.surfaces)) or np.any((self.surfaces <= 0) | (self.surfaces > 1)):
            raise ValueError("surfaces must be nonempty with 0 < s <= 1")
        for name, value, minimum in (("mboz", mboz, 1), ("nboz", nboz, 0), ("oversample", oversample, 1)):
            _integer(value, minimum, name)
        self.mboz, self.nboz, self.oversample = mboz, nboz, oversample
        self.weights, self.options = weights, options

    def compute_state(self, state, rt):
        """Return one live Boozer transform, residual and construction diagnostics."""
        booz = boozer_spectrum_state(state, rt, surfaces=self.surfaces,
                                    mboz=self.mboz, nboz=self.nboz, oversample=self.oversample)
        out = self.kernel(bmnc_b=booz["bmnc_b"], bmns_b=booz.get("bmns_b"),
                          xm_b=booz["xm_b"], xn_b=booz["xn_b"], iota_b=booz["iota_b"],
                          nfp=booz["nfp"], weights=self.weights, **self.options)
        out.update(booz)
        return out

    def residuals_state(self, state, rt):
        """Return the flat traceable least-squares residual."""
        return self.compute_state(state, rt)["residuals1d"]

    def total_state(self, state, rt):
        """Return the sum of squared residuals on a live state."""
        return self.compute_state(state, rt)["total"]

    def residuals(self, eq):
        """Evaluate on a converged equilibrium."""
        return self.residuals_state(eq.state, eq.runtime)

    __call__ = residuals
    J = residuals

    def total(self, eq):
        """Return the scalar objective on a converged equilibrium."""
        return self.total_state(eq.state, eq.runtime)


class CanonicalPwOResidual(_PwOResidual):
    """Periodic canonical field fit; options match its pure Boozer kernel.

    ``bmin`` and ``bmax`` are fixed positive physical Tesla bounds. Widths
    depend on the live signed iota; use the reflected t1/t2 signs when
    reflecting iota. Construct a new term to change fixed target parameters,
    or call the pure kernel with explicit array parameters for joint AD.
    """
    name = "canonical_pwo"
    kernel = staticmethod(canonical_pwo_residual_from_boozer)


class MappedPwOResidual(_PwOResidual):
    """Squeezed LC mapped-field variance with fixed domain and Tesla B_ref.

    Options match :func:`mapped_pwo_residual_from_boozer`. A conservative
    positive Jacobian bound rejects maps whose nonfolding is not certified.
    Geometry, field normalization, mirror floors and high-field orbit-family
    diagnostics remain separate physical requirements.
    """
    name = "mapped_pwo"
    kernel = staticmethod(mapped_pwo_residual_from_boozer)
