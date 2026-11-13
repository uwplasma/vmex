"""VMEX on the exact nonsymmetric vacuum fields of Sengupta et al., arXiv:2610.28466.

Eqs (1)-(4) and (A1)-(A7) give the field, a surface label and both fluxes in closed
form. The paper's paraboloidal member has iota = 1/sqrt(5) on every surface; its
ellipsoidal member, sheared and without stellarator symmetry, has iota0 = sqrt(2)/7.
"""
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import ellipj, ellipk, ellipkinc

import vmex as vj


def sn_complex(x, y, m):
    """sn(x+iy|m) from real Jacobi functions (DLMF 22.8.1 with Jacobi's imaginary transformation)."""
    s, c, d, _ = ellipj(x, m)
    s1, c1, d1, _ = ellipj(y, 1 - m)
    den = c1**2 + m * s**2 * s1**2
    return (s * d1 + 1j * c * d * s1 * c1) / den


class Member:
    """Paraboloidal (`family="p"`, iota=1/sqrt5) or ellipsoidal (`family="e"`, iota0=sqrt2/7) member."""

    def __init__(self, family):
        self.family = family
        if family == "p":
            self.m = brentq(lambda m: ellipk(1 - m) / ellipk(m) - 2.5, 1e-6, 0.9, xtol=1e-15)
            self.K, self.Kp = ellipk(self.m), ellipk(1 - self.m)
            self.kappa = 2 * self.Kp / np.pi
            self.u0, self.v0 = np.arccosh(np.sqrt(5) / np.cosh(2 * np.pi / 5)), 0.0
            self.L = 2 * np.pi
        else:
            self.k_ = 3 - 2 * np.sqrt(2)
            self.m = self.mg = self.k_**2
            self.K, self.Kp = ellipk(self.m), ellipk(1 - self.m)
            self.kappa = 0.5
            self.Kg, self.Kt = self.K, self.Kp
            self.L = 8 * self.Kt
            lam0 = 21 * np.sqrt(2) - 28
            self.u0 = self.v0 = 2 * ellipkinc(np.arcsin(1 / np.sqrt(lam0)), self.mg)
        self.k = np.sqrt(self.m)
        self.nfp = 2  # sn has imaginary period 2K', so zeta -> zeta+L/2 is a rotation by pi
        self.psi0 = 1 / self.k - 1

    def label(self, u, v, zeta):
        """Eq. (2)."""
        m, k = self.m, self.k
        x, y, tau = self.kappa * (u - self.u0), self.kappa * (v - self.v0), self.kappa * zeta
        def cd(q):
            s, c, d, _ = ellipj(q, m)
            return c / d
        U, V = cd(x), 1 / (k * cd(y))
        sf, cf, df, _ = ellipj(tau, 1 - m)
        return np.sqrt(df**2 * (U - V)**2 / ((cf**2 + m * U**2 * sf**2) * (cf**2 + m * V**2 * sf**2)))

    def map(self, u, v, zeta):
        """Position map and metric functions P, E; Eqs (A3), (A4)."""
        if self.family == "p":
            su, sv, cu, cv = np.sinh(u), np.sinh(v), np.cosh(u), np.cosh(v)
            X = np.stack((su * np.cos(zeta) - sv * np.sin(zeta), su * np.sin(zeta) - sv * np.cos(zeta),
                          su * sv + 0.5 * np.sin(2 * zeta)))
            return X, cu * cv, su * sv - np.sin(2 * zeta)
        mg = self.mg
        sR, cR, dR, _ = ellipj((u + v) / 4, mg)
        sS, cS, dS, _ = ellipj((u - v) / 4, mg)
        st, ct, dt, _ = ellipj(zeta / 2 + self.Kt / 2, 1 - mg)
        X = np.stack((cR * st * dS, dR * ct * cS, dt * sS)) / sR
        lam, nu, mu = sR**-2, mg * sS**2, 1 - (1 - mg) * st**2
        return X, (lam - nu) / 4, (lam + nu - 2 * mu) / 4

    def profiles(self, u, v, zeta):
        x, y, tau = self.kappa * (u - self.u0), self.kappa * (v - self.v0), self.kappa * zeta
        p1, p2 = sn_complex(self.K + x, tau, self.m), sn_complex(self.K + y, self.Kp + tau, self.m)
        return p1.imag, p2.imag, (p1 + p2).real

    def ray(self, psi, theta, zeta):
        """(u,v) on the level psi along the outward coordinate ray at polar angle theta."""
        c, s = np.cos(theta), np.sin(theta)
        def f(r):
            return self.label(self.u0 + r * c, self.v0 + r * s, zeta) - psi
        hi = 1e-3
        while f(hi) < 0:
            hi *= 1.1
            if hi > 5 * self.K / self.kappa:
                raise ValueError("level does not close")
        r = brentq(f, hi / 1.1 if hi > 1e-3 else 1e-9, hi, xtol=1e-13)
        return self.u0 + r * c, self.v0 + r * s

    def polar(self, theta, zeta, h=1e-3):
        """Polar angle in (u,v) for the elliptic angle theta: level curves near the axis are u^2/a^2+v^2/b^2."""
        def d2(du, dv):
            return self.label(self.u0 + du, self.v0 + dv, zeta) - self.psi0
        return np.arctan2(np.sqrt(d2(h, 0.0) / d2(0.0, h)) * np.sin(theta), np.cos(theta))

    def flux(self, psi, n=96):
        """Toroidal and poloidal flux, 2*pi*I(psi) and 2*pi*J(psi), from the integrals of Appendix A.4."""
        g, w = np.polynomial.legendre.leggauss(n)
        th = 2 * np.pi * (np.arange(2 * n) + 0.5) / (2 * n)
        toroidal = 0.0
        for t in th:
            ue, ve = self.ray(psi, t, 0.0)
            re = np.hypot(ue - self.u0, ve - self.v0)
            r = 0.5 * re * (g + 1)
            u, v = self.u0 + r * np.cos(t), self.v0 + r * np.sin(t)
            _, P, _ = self.map(u, v, 0.0 * u)
            _, _, A = self.profiles(u, v, 0.0 * u)
            toroidal += np.sum(w * 0.5 * re * r * P * A) * (np.pi / n)
        zetas = self.L * (np.arange(2 * n) + 0.5) / (2 * n)
        J = 0.0
        for z in zetas:
            up, _ = self.ray(psi, 0.0, z)
            u = self.u0 + 0.5 * (up - self.u0) * (g + 1)
            b, _, _ = self.profiles(u, 0 * u + self.v0, 0 * u + z)
            _, _, E = self.map(u, 0 * u + self.v0, 0 * u + z)
            J += np.sum(w * 0.5 * (up - self.u0) * E * b) * (self.L / (2 * n))
        return toroidal, J

    def boundary(self, psi, mpol, ntor, nth=48, nph=32):
        """R(theta,phi), Z(theta,phi) on one field period at equal cylindrical azimuth, as Fourier data.

        The 48 x 32 grid reproduces the 256 x 64 coefficients to 1e-6 (spectral convergence of
        an analytic boundary), far below the 2e-4 iota tolerance of the test.
        """
        th = 2 * np.pi * np.arange(nth) / nth
        ph = 2 * np.pi * np.arange(nph) / (nph * self.nfp)
        R, Z = np.empty((nth, nph)), np.empty((nth, nph))
        sgn = 1.0 if self.family == "p" else -1.0
        for i, t in enumerate(th):
            zs = sgn * ph
            for _ in range(8):  # Newton on azimuth(zeta) = phi at this ray
                pts = [self.ray(psi, self.polar(t, z), z) + (z,) for z in zs]
                pts2 = [self.ray(psi, self.polar(t, z + 1e-5), z + 1e-5) + (z + 1e-5,) for z in zs]
                phi = np.array([np.arctan2(*self.map(*map(np.float64, q))[0][[1, 0]]) for q in pts])
                phi2 = np.array([np.arctan2(*self.map(*map(np.float64, q))[0][[1, 0]]) for q in pts2])
                d = (phi2 - phi + np.pi) % (2 * np.pi) - np.pi
                res = (phi - ph + np.pi) % (2 * np.pi) - np.pi
                zs = zs - res / (d / 1e-5)
                if np.max(np.abs(res)) < 1e-11:
                    break
            X = np.array([self.map(*map(np.float64, q))[0] for q in pts])
            R[i], Z[i] = np.hypot(X[:, 0], X[:, 1]), X[:, 2]
        T, P = np.meshgrid(th, ph, indexing="ij")
        rbc, zbs, rbs, zbc = {}, {}, {}, {}
        for m in range(mpol):
            for n in range(-ntor, ntor + 1):
                if m == 0 and n < 0:
                    continue
                ang = m * T - n * self.nfp * P
                f = (1.0 if m == 0 and n == 0 else 2.0) / (nth * nph)
                rbc[(n, m)], zbc[(n, m)] = f * np.sum(R * np.cos(ang)), f * np.sum(Z * np.cos(ang))
                if (m, n) != (0, 0):
                    zbs[(n, m)], rbs[(n, m)] = f * np.sum(Z * np.sin(ang)), f * np.sum(R * np.sin(ang))
        return rbc, zbs, rbs, zbc


@pytest.mark.parametrize("family", ["p", "e"])
def test_label_is_conserved_along_the_closed_form_field(family):
    q = Member(family)
    u, v, zeta, h = q.u0 + 0.05, q.v0 + 0.03, 0.4, 1e-5
    b, e, A = q.profiles(u, v, zeta)
    _, P, E = q.map(u, v, zeta)
    D, ratio = P**2 - E**2, E / P
    def d(du, dv, dz):
        return (q.label(u + du * h, v + dv * h, zeta + dz * h) - q.label(u - du * h, v - dv * h, zeta - dz * h)) / (2 * h)
    assert abs((b - ratio * e) / D * d(1, 0, 0) + (e - ratio * b) / D * d(0, 1, 0) + A / D * d(0, 0, 1)) < 1e-7


@pytest.mark.parametrize("family,iota", [("p", 1 / np.sqrt(5)), ("e", np.sqrt(2) / 7)])
def test_flux_ratio_reproduces_the_exact_transform(family, iota):
    q = Member(family)
    flux, poloidal = q.flux(q.psi0 * 1.00001, 32)
    assert abs(abs(poloidal / flux) - iota) < 1e-4


@pytest.mark.parametrize("family", ["p", "e"])
def test_vmex_transform_converges_to_the_exact_value(family, tmp_path):
    mpol, ntor, ns, q = 7, 14, 65, Member(family)
    psi = q.psi0 * 1.005
    rbc, zbs, rbs, zbc = q.boundary(psi, mpol, ntor)
    asym = family == "e"
    lines = ["&INDATA", f"NFP = {q.nfp}", f"MPOL = {mpol}", f"NTOR = {ntor}", f"LASYM = {'T' if asym else 'F'}",
             f"NS_ARRAY = 17 33 {ns}", "FTOL_ARRAY = 1e-11 1e-11 1e-13", "NITER_ARRAY = 60000 60000 60000", "DELT = 0.9",
             f"PHIEDGE = {q.flux(psi)[0]:.16e}", "NCURR = 1", "CURTOR = 0", "AC = 0", "AM = 0", "PRES_SCALE = 0"]
    groups = [("RBC", rbc), ("ZBS", zbs)] + ([("RBS", rbs), ("ZBC", zbc)] if asym else [])
    lines += [f"{name}({n},{m}) = {value:.16e}" for name, table in groups for (n, m), value in table.items()]
    (tmp_path / "input.exact").write_text("\n".join(lines) + "\n/\n")
    inp = vj.VmecInput.from_file(tmp_path / "input.exact")
    result = vj.solve_multigrid(inp, raise_on_max_iterations=True)
    iota = np.abs(np.asarray(vj.wout_from_result(inp, result).iotas))[1:]
    s = (np.arange(1, ns) - 0.5) / (ns - 1)
    middle = (s > 0.2) & (s < 0.9)
    if family == "p":
        exact = np.full_like(s, 1 / np.sqrt(5))
    else:
        psis = q.psi0 * (1 + 0.005 * np.linspace(0.02, 1.0, 12)**2)
        table = np.array([q.flux(x, 48) for x in psis]) / q.flux(psi)[0]
        exact = np.abs(np.polynomial.chebyshev.Chebyshev.fit(table[:, 0], table[:, 1], 8, domain=[0, 1]).deriv()(s))
    assert np.max(np.abs(iota - exact)[middle]) < 2e-4
