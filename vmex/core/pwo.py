"""Smooth piecewise-omnigenity targets and mapped-field residuals.

No VMEC2000 counterpart: these are optimization objectives, not MHD equations.
Angles are Boozer radians and ``xn_b`` contains physical toroidal mode numbers.
The canonical construction uses the super-Gaussian of Velasco et al., PRL 133,
185101 (2024), with the zero-bootstrap width of Calvo et al., PRE 112, L023201
(2025). Its finite-p periodic completion is explicitly a bounded lattice union,
``1 - product(1 - exp(-u**(2*p)-v**(2*p)))``. This differs from the uncompleted
fundamental-domain expression in Fernandez-Pacheco et al., arXiv:2601.14886v2
and their DESC reference implementation (Zenodo 20390611, CC-BY-4.0).

For a parallelogram contained in its centered cell, omitted lattice shells
beyond ``images`` contribute at most ``sum(8*k*exp(-(2*k-1)**(2*p)))``. With
p>=2 and images>=1 this is below 1e-33; the default images=2 gives <1e-269.
Thus field values are periodic to float64 precision; parameter derivatives
require the usual conditioning checks near a degenerate width.
Overlap diagnostics quantify the finite-p difference from an isolated patch;
the sharp limit is the union of the published parallelograms. No exact finite-p
pwO or exact reproduction of the published equilibrium is asserted.

The mapped route implements Liu et al., arXiv:2603.12139v1, equations (2),
(9), and (10), for poloidal omnigenity. ``gamma`` is their mapped alpha,
not the field-line label theta-iota*zeta. A fixed cut eta interval is integrated
with midpoint quadrature, without treating its endpoints as periodic. The
residual is mapped gamma variance divided by a fixed physical B_ref. It tests
the mapped region only; the unmapped high-field structure requires independent
orbit-family validation. JIT/JVP/VJP are supported for the pure-array kernels;
full-equilibrium differentiation follows the existing state's solver contract.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from .omnigenity import boozer_spectrum_state

__all__ = ["canonical_pwo_field", "squeezed_pwo_coordinates",
           "canonical_pwo_residual_from_boozer", "mapped_pwo_residual_from_boozer",
           "CanonicalPwOResidual", "MappedPwOResidual"]


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
    valid = (jnp.isfinite(width) & (jnp.abs(width) > 1e-12)
             & (jnp.abs(determinant) > 1e-12) & (jnp.abs(iota) > 1e-12)
             & jnp.isfinite(bmin) & jnp.isfinite(bmax) & (bmin > 0) & (bmax > bmin)
             & jnp.isfinite(margin) & (margin >= 0))
    zeta_center = jnp.pi / nfp if zeta_center is None else zeta_center
    theta = jnp.asarray(theta, dtype=jnp.float64)
    zeta = jnp.asarray(zeta, dtype=jnp.float64)
    valid = (valid & jnp.isfinite(theta) & jnp.isfinite(zeta)
             & jnp.isfinite(theta_center) & jnp.isfinite(zeta_center))
    _require(valid, "canonical target needs finite coordinates, positive distinct B bounds and a nondegenerate parallelogram inside its cell")
    # Wrapping chooses a nearby stencil, not a single discontinuous patch.
    dt = jnp.mod(theta - theta_center + jnp.pi, 2*jnp.pi) - jnp.pi
    dz = jnp.mod(zeta - zeta_center + jnp.pi/nfp, 2*jnp.pi/nfp) - jnp.pi/nfp
    dt, dz, _ = jnp.broadcast_arrays(dt, dz, width)
    shifts = jnp.arange(-images, images+1)
    poloidal_shift, toroidal_shift = jnp.meshgrid(shifts, shifts, indexing="ij")
    shape = (-1,) + (1,)*dt.ndim
    t = dt + 2*jnp.pi*poloidal_shift.reshape(shape)
    z = dz + 2*jnp.pi*toroidal_shift.reshape(shape)/nfp
    # Saturating below floating-point underflow also keeps high-p tangents
    # finite when a remote image's raw even power would overflow.
    limit = 750.0**(1.0/(2*p))
    u = jnp.minimum(jnp.abs((z+t1*t)/width), limit)
    v = jnp.minimum(jnp.abs((t+t2*z)/jnp.pi), limit)
    return jnp.exp(-u**(2*p)-v**(2*p)), width, margin, valid


def canonical_pwo_field(theta, zeta, *, iota, nfp, bmin, bmax, t1, t2, p=2,
                        theta_center=0.0, zeta_center=None, images=2):
    """Return a bounded periodic super-Gaussian union and overlap diagnostics.

    All continuous arguments broadcast. For surface arrays use shape
    ``(nsurface,1,1)`` against angular meshes. ``nfp``, ``p``, and ``images``
    are static integers. Invalid eager data raise; traced invalid data return
    ``valid=False`` and NaN field values. ``bmin`` is a lower bound: finite-p
    tails generally keep the actual minimum above it.
    """
    patches, width, margin, valid = _canonical_images(
        theta, zeta, iota=iota, nfp=nfp, bmin=bmin, bmax=bmax, t1=t1, t2=t2,
        p=p, theta_center=theta_center, zeta_center=zeta_center, images=images)
    union = 1.0 - jnp.prod(1.0 - patches, axis=0)
    field = jnp.asarray(bmin) + (jnp.asarray(bmax)-jnp.asarray(bmin))*union
    return {"B_target": jnp.where(valid, field, jnp.nan), "width": width,
            "corner_margin": margin, "valid": valid,
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
        zeta_center=None, images=2, minimum_corner_margin=0.05, ntheta=32,
        nzeta=32, weights=None):
    """Normalized pointwise target fit with live signed iota and both parities.

    Surface weights sum to one; angular quadrature is uniform and normalized.
    Returns sampled ``B``, ``B_target``, overlap, margins, validity and extrema.
    """
    _integer(ntheta, 4, "ntheta"); _integer(nzeta, 4, "nzeta")
    if not 0 <= minimum_corner_margin < 1:
        raise ValueError("minimum_corner_margin must lie in [0,1)")
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
    valid = valid & jnp.all(target["valid"], axis=(1, 2)) & (target["corner_margin"][:, 0, 0] >= minimum_corner_margin)
    out = _finish((B-target["B_target"])/target["B_target"], B, valid, weight)
    out.update({**target, "valid": out["valid"], "B": B, "theta": theta[0, :, 0], "zeta": zeta[0, 0, :]})
    return out


def mapped_pwo_residual_from_boozer(*, bmnc_b, xm_b, xn_b, iota_b, nfp,
        b_ref, bmns_b=None, a=0.3, b=0.0, c=-2.0, eta_min=2.8,
        ngamma=32, neta=32, weights=None):
    """Fixed-domain mapped-gamma variance, not a global pwO certificate.

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
