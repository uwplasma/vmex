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

__all__ = ["squeezed_pwo_coordinates", "mapped_pwo_residual_from_boozer", "MappedPwOResidual",
           "FreeMapProblem", "one_field_period", "qa_precursor", "transport_backtrack"]


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


def _limited_map(a, b, c, eta_min, gamma, iota, nfp, neta, span_margin=.95, jacobian_margin=.1):
    """Clip (a, b, c, eta_min) per surface to a one-to-one map: zeta(eta) and its span are linear in a."""
    b, c = jnp.clip(b, -1., 1.), jnp.minimum(c, -.5)
    # eta_min at which the distance alone takes 85% of the half period.
    lin, gap = 1+c*jnp.pi, .15*jnp.pi
    eta_min = jnp.clip(eta_min, (lin-jnp.sqrt(lin*lin-4*c*gap))/(2*c), jnp.pi-1e-3)
    eta = eta_min+(2*jnp.pi-2*eta_min)*(jnp.arange(neta)[None, None, :]+.5)/neta

    def zeta(a_value, eta_value):
        return squeezed_pwo_coordinates(gamma, eta_value, iota=iota, nfp=nfp, a=a_value, b=b, c=c)

    ends = jnp.concatenate((eta_min, 2*jnp.pi-eta_min), axis=2)
    span0, span1 = ((q["zeta"][:, :, 1:]-q["zeta"][:, :, :1]) for q in (zeta(0., ends), zeta(1., ends)))
    a_span = jnp.min((span_margin*2*jnp.pi/nfp-span0)/jnp.maximum(span1-span0, 1e-12), axis=1, keepdims=True)
    jac0, jac1 = zeta(0., eta)["jacobian"], zeta(1., eta)["jacobian"]
    drop = jnp.maximum(jac0-jac1, 1e-12)
    a_jac = jnp.min((jac0-jacobian_margin/nfp)/drop, axis=(1, 2), keepdims=True)
    return jnp.clip(a, 0., jnp.minimum(a_span, a_jac)), b, c, eta_min


def mapped_pwo_residual_from_boozer(*, bmnc_b, xm_b, xn_b, iota_b, nfp,
        b_ref=None, bmns_b=None, a=0.3, b=0.0, c=-2.0, eta_min=2.8,
        ngamma=32, neta=32, weights=None, limit_map=False):
    """Mapped-gamma variance divided by the live Boozer mean field squared.

    Eta midpoint quadrature covers [eta_min,2*pi-eta_min]. Gamma is uniform
    and endpoint-free. Discrete Parseval in gamma makes this the power of all
    nonconstant gamma harmonics, averaged over eta. The default normalization
    is invariant under field rescaling and differentiated with the spectrum.
    An explicit ``b_ref`` instead uses a fixed positive reference in Tesla.
    ``limit_map=True`` clips the map on each surface to the closest one-to-one
    map for the live iota, so every parameter value yields a valid map.
    No Fourier periodicity in the cut eta direction is assumed.
    """
    _integer(ngamma, 4, "ngamma"); _integer(neta, 4, "neta")
    _require(jnp.isfinite(eta_min) & (eta_min > 0) & (eta_min < jnp.pi),
             "eta_min must be strictly between zero and pi")
    cosine, sine, m, n, iota, weight, valid = _spectrum(bmnc_b, bmns_b, xm_b, xn_b, iota_b, nfp, weights)
    ns = cosine.shape[0]
    constant = (m == 0) & (n == 0)
    mean_field = jnp.sum(jnp.where(constant, cosine, 0), axis=1)
    variance = .5*jnp.sum(jnp.where(constant, 0, cosine*cosine+sine*sine), axis=1)
    if b_ref is None:
        valid = valid & (jnp.sum(constant) == 1) & (mean_field > 0)
        b_ref = mean_field
    a, b, c, reference = (_surface(x, ns, name) for name, x in
                           (("a", a), ("b", b), ("c", c), ("b_ref", b_ref)))
    gamma = 2*jnp.pi*jnp.arange(ngamma)[None, :, None]/ngamma
    eta_min = _surface(eta_min, ns, "eta_min")
    if limit_map:
        a, b, c, eta_min = _limited_map(a, b, c, eta_min, gamma, jnp.abs(iota[:, None, None]), nfp, neta)
    eta = eta_min+(2*jnp.pi-2*eta_min)*(jnp.arange(neta)[None, None, :]+0.5)/neta
    # The map is built for positive iota. A mirror image (zeta -> -zeta, iota -> -iota)
    # is the same device, so negative iota reads the same map at -zeta.
    coordinates = squeezed_pwo_coordinates(gamma, eta, iota=jnp.abs(iota[:, None, None]), nfp=nfp, a=a, b=b, c=c)
    B = _evaluate(cosine, sine, m, n, coordinates["theta"],
                  jnp.where(iota[:, None, None] < 0, -1.0, 1.0)*coordinates["zeta"])
    # Theta equals gamma, so each line of constant gamma maps to one line of constant
    # theta. The map is one-to-one on the torus exactly when zeta increases along each
    # line and the line spans less than one field period; both are checked here on the nodes.
    ends = squeezed_pwo_coordinates(gamma, jnp.concatenate((eta_min, 2*jnp.pi-eta_min), axis=2),
                                    iota=jnp.abs(iota[:, None, None]), nfp=nfp, a=a, b=b, c=c)["zeta"]
    span = jnp.max(ends[:, :, 1]-ends[:, :, 0], axis=1)
    eta_cut = eta_min[:, 0, 0]
    jacobian_min = jnp.min(coordinates["jacobian"], axis=(1, 2))
    distance_at_cut = c*eta_min**2-(1+c*jnp.pi)*eta_min+jnp.pi
    valid_map = (jnp.isfinite(a) & jnp.isfinite(b) & jnp.isfinite(c)
                 & (c < -1/jnp.pi) & jnp.isfinite(reference) & (reference > 0)
                 & (distance_at_cut > 0) & (span[:, None, None] < 2*jnp.pi/nfp))
    valid = valid & valid_map[:, 0, 0] & (jacobian_min > 0)
    mean = jnp.mean(B, axis=1, keepdims=True)
    out = _finish((B-mean)/reference, B, valid, weight)
    out.update({"B_mapped": B, "B_mean_gamma": mean, "gamma": gamma[0, :, 0],
                "B00": mean_field, "field_variance": variance,
                "mapped_variance": jnp.mean((B-mean)**2, axis=(1, 2)),
                "eta": eta[0, 0, :], "jacobian": coordinates["jacobian"],
                "jacobian_min": jacobian_min, "toroidal_span": span,
                "eta_fraction": 1-eta_cut/jnp.pi,
                "mapped_area_fraction": jnp.mean(coordinates["jacobian"], axis=(1, 2))*(2*jnp.pi-2*eta_cut)*nfp/(2*jnp.pi)})
    return out


class MappedPwOResidual:
    """Squeezed LC mapped-field variance with a fixed domain and live B00 scale.

    Options match :func:`mapped_pwo_residual_from_boozer`. Maps that fold or
    cover a point twice on the evaluated nodes are rejected.
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

    def spectrum_state(self, state, rt):
        """Return the flat Boozer spectrum and iota, for a term whose map parameters vary."""
        booz = boozer_spectrum_state(state, rt, surfaces=self.surfaces, mboz=self.mboz,
                                     nboz=self.nboz, oversample=self.oversample)
        return jnp.concatenate((booz["bmnc_b"].ravel(), booz["iota_b"]))

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

    def physical_metrics(self, eq, *, aspect_bounds, iota_floor, bmax_ref=None,
                         force_tol=1e-12, mirror_floor=.05, mirror_ceiling=1., fields=None):
        """Check force balance, oriented volume, positive B² and full-surface limits.

        Optional ``bmax_ref`` sets a fixed ceiling in Tesla on each surface.
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
            and (bmax_ref is None or np.all(np.asarray(high) <= np.asarray(bmax_ref)))
            and np.all(np.asarray(mirror) >= mirror_floor)
            and np.all(np.asarray(mirror) <= mirror_ceiling))
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


class FreeMapProblem:
    """A boundary stage with free map parameters in one least-squares problem.

    The stage's first residual rows must be ``mapped.spectrum_state``. The map
    does not change the equilibrium, so its Jacobian columns come directly from
    the kernel. Parameters move through a sigmoid inside ``MAP_BOUNDS``, and
    ``limit_map`` keeps every value valid on every surface.
    """

    MAP_BOUNDS = dict(a=(0., .8), b=(-1., 1.), c=(-4., -.5), eta_min=(2.2, 3.1))

    def __init__(self, stage, mapped, equilibrium, values, free, *, weight=1e3, area_floor=.6):
        booz = boozer_spectrum_state(equilibrium.state, equilibrium.runtime, surfaces=mapped.surfaces,
                                     mboz=mapped.mboz, nboz=mapped.nboz, oversample=mapped.oversample)
        ns, nm = booz["bmnc_b"].shape
        self.stage, self.nx, self.nspec = stage, stage.x0.size, ns*nm+ns
        names = tuple(values)
        low, high = (np.array([self.MAP_BOUNDS[k][i] for k in names]) for i in (0, 1))
        fixed = np.array([values[k] for k in names], dtype=float)
        index = np.array([i for i, k in enumerate(names) if k in free], dtype=int)
        self.names, self.index = names, index
        xm, xn, nfp = booz["xm_b"], booz["xn_b"], booz["nfp"]

        def parameters(u):
            return jnp.asarray(fixed).at[index].set(low[index]+(high[index]-low[index])*jax.nn.sigmoid(u))
        self._parameters = parameters

        def rows(spectrum, u):
            a, b, c, eta_min = parameters(u)
            out = mapped_pwo_residual_from_boozer(
                bmnc_b=spectrum[:ns*nm].reshape(ns, nm), xm_b=xm, xn_b=xn, iota_b=spectrum[ns*nm:],
                nfp=nfp, a=a, b=b, c=c, eta_min=eta_min, weights=mapped.weights, limit_map=True)
            area = jnp.sqrt(100.)*jnp.maximum(area_floor-out["mapped_area_fraction"], 0.)
            return jnp.concatenate((jnp.sqrt(weight)*out["residuals1d"], area))
        self._rows = jax.jit(rows)
        self._columns = jax.jit(lambda s, u, v: (
            jax.vmap(lambda d: jax.jvp(lambda q: rows(q, u), (s,), (d,))[1], in_axes=1, out_axes=1)(v),
            jax.jacfwd(lambda q: rows(s, q))(u)))
        clipped = np.clip(fixed[index], low[index]+1e-3*(high[index]-low[index]), high[index]-1e-3*(high[index]-low[index]))
        fraction = (clipped-low[index])/(high[index]-low[index])
        self.x0 = np.concatenate((np.asarray(stage.x0), np.log(fraction/(1-fraction))))

    def residual(self, z):
        """Return the stage residual with the spectrum rows replaced by the mapped pwO rows."""
        raw = np.asarray(self.stage.residual(z[:self.nx]))
        return np.concatenate((np.asarray(self._rows(jnp.asarray(raw[:self.nspec]), jnp.asarray(z[self.nx:]))),
                               raw[self.nspec:]))

    def residual_jac(self, z):
        """Return the joint Jacobian over the boundary and the free map variables."""
        raw = np.asarray(self.stage.residual(z[:self.nx]))
        full = np.asarray(self.stage.residual_jac(z[:self.nx]))
        boundary, free = (np.asarray(q) for q in self._columns(
            jnp.asarray(raw[:self.nspec]), jnp.asarray(z[self.nx:]), jnp.asarray(full[:self.nspec])))
        rest = np.hstack((full[self.nspec:], np.zeros((full.shape[0]-self.nspec, free.shape[1]))))
        return np.vstack((np.hstack((boundary, free)), rest))

    def values(self, z):
        """Return the map parameters at ``z`` as a dictionary."""
        return dict(zip(self.names, map(float, self._parameters(jnp.asarray(z[self.nx:])))))


def one_field_period(inp):
    """Represent the same boundary/axis with NFP=1, retaining physical toroidal n.

    The input resolution must accommodate every nonzero physical harmonic;
    in particular an NFP=2 max-mode-2 precursor needs physical n=4.
    """
    from dataclasses import replace

    coefficients = {}
    for name in ("rbc", "zbs", "rbs", "zbc", "raxis_c", "zaxis_s", "raxis_s", "zaxis_c"):
        old = np.asarray(getattr(inp, name))
        new = np.zeros_like(old)
        boundary = old.ndim == 2
        for row in range(old.shape[0]):
            n = row-inp.ntor if boundary else row
            physical_n = n*inp.nfp
            if abs(physical_n) > inp.ntor:
                if np.any(old[row] != 0):
                    raise ValueError("NTOR cannot retain all physical toroidal harmonics")
            else:
                new[physical_n+inp.ntor if boundary else physical_n] = old[row]
        coefficients[name] = new
    return replace(inp, nfp=1, **coefficients)


def qa_precursor(inp, seed, mapped, geometry, outdir, *, maxiter, optimizer_ftol,
                 mirror_limit, mirror_weight, physical_options, **problem_options):
    """Checkpoint a native QA precursor through boundary modes 1 and 2.

    NFP2 uses the unit-weight native QA residual and geometry guards. NFP3
    uses automatic initial native-cost normalization and the mirror penalty.
    Return the best feasible native-QA trial, including intermediate trials.
    The precursor is separate from the later pure mapped-pwO optimization.
    """
    from .. import optimize as opt, write_wout
    from scipy.optimize import least_squares
    from .monitoring import FeasibleCandidateMonitor

    native = opt.QuasisymmetryRatioResidual(mapped.surfaces, helicity_m=1, helicity_n=0)
    scale = (float(native.total_state(seed.state, seed.runtime)) or 1.) if inp.nfp == 3 else 1.

    def objective(state, runtime):
        residuals = [native.residuals_state(state, runtime)/jnp.sqrt(scale), geometry(state, runtime)]
        if inp.nfp == 3:
            mirror = mapped.diagnostics_state(state, runtime)[3]
            residuals.append(jnp.sqrt(mirror_weight)*jnp.maximum(mirror/(.9*mirror_limit)-1, 0))
        return jnp.sqrt(2)*jnp.concatenate(residuals)

    diagnostics = jax.jit(lambda state, rt: (objective(state, rt), native.total_state(state, rt),
                                           mapped.diagnostics_state(state, rt)))

    best_equilibrium = (np.inf, None)

    def inspect(eq):
        nonlocal best_equilibrium
        residual, raw, fields = diagnostics(eq.state, eq.runtime)
        metrics = mapped.physical_metrics(eq, **physical_options,
            mirror_ceiling=mirror_limit if inp.nfp == 3 else 1., fields=fields)
        metrics["mapped_raw"], metrics["raw"] = metrics["raw"], float(raw)
        if metrics["feasible"] and metrics["raw"] < best_equilibrium[0]:
            best_equilibrium = (metrics["raw"], eq)
        return np.asarray(residual), metrics

    problem = opt.VmecProblem.from_tuples(inp, [(objective, 0., 1.)], max_mode=2,
        vary_major_radius=False, restart_from=seed, **problem_options)
    monitor = FeasibleCandidateMonitor(problem, inspect, outdir, maxiter=maxiter,
                                       family="QA precursor", nfp=inp.nfp, initial_scale=scale)
    with monitor:
        x = problem.x0.copy()
        monitor(x)
        inp.to_indata(monitor.outdir / "input.seed")
        write_wout(monitor.outdir / "wout_seed.nc", seed.wout)
        for mode in (1, 2):
            print(f"[QA precursor] NFP={inp.nfp}, max_mode={mode}", flush=True)
            stage = problem.subproblem(max_mode=mode, x=x)
            step = .02*np.asarray(stage.scales)
            result = least_squares(lambda y: monitor(stage.embed(y)), stage.x0,
                jac=stage.residual_jac, x_scale=step,
                bounds=(stage.x0-5*step, stage.x0+5*step), max_nfev=5*maxiter+5,
                ftol=optimizer_ftol, xtol=1e-10, gtol=1e-7,
                callback=monitor.callback(mode), verbose=2)
            x = stage.embed(result.x)
            monitor(x)
            problem.input_from_x(x).to_indata(monitor.outdir / f"input.mode{mode}")
            write_wout(monitor.outdir / f"wout_mode{mode}.nc", problem.equilibrium_from_x(x).wout)
        monitor.save(status="completed", stopping_reason=str(result.message))
        if monitor.best is None:
            raise RuntimeError("No physically feasible QA precursor; keep the original circle")
        x = monitor.best[1]
        return problem.input_from_x(x), best_equilibrium[1]


def transport_backtrack(monitor, start, proposal, baseline_Q, surfaces, *, seed):
    """Accept the first dyadic pwO step passing physics and matched NEO guards.

    ``start`` is the saved precursor; ``proposal`` is the full pwO optimizer
    direction. A smaller mapped variance than the precursor and strict NEO
    improvement on every radius versus ``baseline_Q`` are required.
    ``start`` must be the first monitored vector. Independent forward solves
    restart from its saved ``seed`` equilibrium, avoiding optimizer-cache
    revisits after a long optimization. Checkpoints retain the full direction.
    """
    from .. import optimize as opt, write_wout
    from .errors import VmecConvergenceError, VmecJacobianError, VmecNumericalError
    from .neoclassical import screen_transport, transport_improves
    from .restart import restart_state

    problem = monitor.problem
    initial = monitor.report["evaluations"][0]["raw"]

    def equilibrium(x):
        inp = problem.input_from_x(x)
        try:
            return opt.solve_equilibrium(inp, initial_state=restart_state(seed.wout, inp),
                                         raise_on_max_iterations=True), None
        except (VmecConvergenceError, VmecJacobianError, VmecNumericalError) as error:
            return None, str(error)

    problem.input_from_x(proposal).to_indata(monitor.outdir / "input.direction")
    trials = []
    for power in range(1, 7):
        alpha = 2.**-power
        x = start+alpha*(proposal-start)
        eq, error = equilibrium(x)
        if eq is None:
            print(f"[NEO backtrack] alpha={alpha:g}: equilibrium rejected", flush=True)
            trials.append(dict(alpha=alpha, error=error, accepted=False))
            monitor.save(backtracking=trials, transport_screen_passed=False)
            continue
        residual, metrics = monitor.inspect(eq)
        residual = np.asarray(residual)
        if not np.isfinite(residual).all() or not np.isfinite(residual@residual):
            raise ValueError("Backtracking residual and cost must be finite")
        history = monitor.report["evaluations"]
        row = dict(evaluation=len(history), cost=.5*float(residual@residual), **metrics)
        history.append(row)
        admissible = metrics["feasible"] and metrics["raw"] < initial
        q = screen_transport(eq.wout, surfaces) if admissible else None
        accepted = bool(admissible and transport_improves(baseline_Q, q))
        mean_q = float(np.mean(q)) if q is not None else float("nan")
        print(f"[NEO backtrack] alpha={alpha:g}, mean Q={mean_q:.6g}, accepted={accepted}", flush=True)
        trials.append(dict(alpha=alpha, physics=metrics,
                           Q=None if q is None else q.tolist(), accepted=accepted))
        monitor.save(backtracking=trials, transport_screen_passed=accepted)
        if accepted:
            problem.input_from_x(x).to_indata(monitor.outdir / "input.trial")
            write_wout(monitor.outdir / "wout_trial.nc", eq.wout)
            monitor.best = (metrics["raw"], x.copy())
            monitor.save(trial_Q=q.tolist(), accepted_alpha=alpha,
                selected_evaluation=row["evaluation"],
                initial_pwO=initial, pwO_reduction=1-metrics["raw"]/initial)
            return True
    return False
