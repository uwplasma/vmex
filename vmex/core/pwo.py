"""Smooth mapped pwO construction in Boozer radians; toroidal modes include NFP.

The squeezed LC mapping follows Liu et al., arXiv:2603.12139v1, equations
(2), (9), and (10). Gamma is the mapped alpha, not theta-iota*zeta.
The normalized gamma variance constrains only a fixed eta cut: it does not
establish piecewise action invariance or transport confinement. Kernel
JIT/JVP/VJP support does not extend the equilibrium solver's differentiation
contract. Field contrast and full-surface geometry require separate guards.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from .omnigenity import boozer_spectrum_state

__all__ = ["squeezed_pwo_coordinates", "mapped_pwo_residual_from_boozer", "MappedPwOResidual"]


def _require(condition, message):
    if not isinstance(condition, jax.core.Tracer) and not bool(np.all(np.asarray(condition))):
        raise ValueError(message)


def _integer(value, minimum, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


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


class MappedPwOResidual:
    """Squeezed LC mapped-field variance with a fixed domain and Tesla B_ref.

    Options match :func:`mapped_pwo_residual_from_boozer`. A conservative
    positive Jacobian bound rejects maps whose nonfolding is not certified.
    Field normalization, mirror floors and full-surface diagnostics remain
    separate physical requirements.
    """

    name = "mapped_pwo"
    kernel = staticmethod(mapped_pwo_residual_from_boozer)

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

    def diagnostics_state(self, state, rt, *, ntheta=64, nzeta=64):
        """Mapped residual and full-surface field extrema from one Boozer transform."""
        mapped = self.compute_state(state, rt)
        theta = 2*jnp.pi*jnp.arange(ntheta)[None, :, None]/ntheta
        zeta = 2*jnp.pi*jnp.arange(nzeta)[None, None, :]/(mapped["nfp"]*nzeta)
        field = _evaluate(mapped["bmnc_b"], mapped["bmns_b"], mapped["xm_b"],
                          mapped["xn_b"], theta, zeta)
        low, high = field.min(axis=(1, 2)), field.max(axis=(1, 2))
        return mapped, low, high, (high-low)/(high+low)

    def physical_metrics(self, eq, *, bmax_ref, aspect_bounds, iota_floor,
                         force_tol=1e-12, mirror_floor=.05, fields=None):
        """Check force balance, oriented volume, positive B² and full-surface limits.

        ``bmax_ref`` is a fixed ceiling, in Tesla, on each sampled surface.
        The iota floor is signed. These checks do not certify orbit confinement.
        """
        from .statephysics import _field_chain

        mapped, low, high, mirror = (self.diagnostics_state(eq.state, eq.runtime)
                                    if fields is None else fields)
        _, jacobian, _, field, _ = _field_chain(eq.state, eq.runtime)
        signed_jacobian = np.asarray(jacobian.sqrt_g)[1:]*float(eq.runtime.setup.signgs)
        pressure = np.asarray(field.pressure)
        pressure = pressure[:, None, None] if pressure.ndim == 1 else pressure
        B2 = np.asarray(2*(field.total_pressure-pressure))[1:]
        forces = [float(getattr(eq.result, k)) for k in ("fsqr", "fsqz", "fsql")]
        row = dict(aspect=float(eq.wout.aspect), iota_min=float(np.min(eq.wout.iotaf)),
                   force_errors=forces, raw=float(mapped["total"]), beta=float(eq.wout.betatotal),
                   Bmin=np.asarray(low).tolist(), Bmax=np.asarray(high).tolist(),
                   mirror=np.asarray(mirror).tolist())
        row["feasible"] = bool(
            np.isfinite(forces).all() and max(forces) <= force_tol
            and aspect_bounds[0] <= row["aspect"] <= aspect_bounds[1]
            and row["iota_min"] > iota_floor
            and np.isfinite(B2).all() and B2.min() > 0
            and np.isfinite(signed_jacobian).all() and signed_jacobian.min() > 0
            and np.asarray(mapped["valid"]).all()
            and np.isfinite(low).all() and np.all(np.asarray(low) > 0)
            and np.isfinite(high).all() and np.isfinite(mirror).all()
            and np.all(np.asarray(high) <= np.asarray(bmax_ref))
            and np.all(np.asarray(mirror) >= mirror_floor))
        return row

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
