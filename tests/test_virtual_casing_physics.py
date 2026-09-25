"""Physics and derivative certificates for prescribed-interface virtual casing.

:mod:`vmex.core.virtual_casing` writes ``B_out . n = 0`` as a smooth objective
with ``B_plasma`` from the virtual-casing principle. The NESTOR free-boundary
solve is intentionally separate.
Lanes: the wout->surface-data adapter reproduces ``B_total . n / |B|`` ~
1e-16 on a converged equilibrium; asset-free synthetic-torus ``jax.grad``
vs central FD; and the real cth-like ``extcur``/coil-dof gradients vs FD
(``full``).  Skipped explicitly when ``virtual_casing_jax`` is unavailable.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from vmex.core import virtual_casing as VC  # noqa: E402
from vmex.core.extender import (  # noqa: E402
    ExteriorFieldAccuracyError,
    ExteriorFieldAccuracyWarning,
    VmecExtender,
)
from vmex.core.mgrid import MgridField, read_mgrid  # noqa: E402
from vmex.core.wout import read_wout  # noqa: E402

# jit-enable the whole module: virtual casing is far too slow interpreted.
pytestmark = [
    pytest.mark.usefixtures("_module_jit_enabled"),
    pytest.mark.skipif(
        not VC.have_virtual_casing_jax(),
        reason="requires virtual_casing_jax",
    ),
]
VmecSurfaceFieldData = VC.VmecSurfaceFieldData

REPO = Path(__file__).resolve().parents[1]
WOUT = REPO / "examples" / "data" / "single_grid" / "wout_cth_like_free_bdy.nc"
MGRID = REPO / "examples" / "data" / "mgrid_cth_like.nc"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _circular_coil_dofs(ncoils=3, order=1, R0=0.75, a=0.35, nfp=5):
    """Circular coil Fourier dofs (the ESSOS ``Curves`` Fourier convention)."""
    dofs = np.zeros((ncoils, 3, 2 * order + 1))
    for i in range(ncoils):
        p0 = (i + 0.5) * (2 * np.pi / nfp) / (2 * ncoils)
        dofs[i, 0, 0], dofs[i, 0, 2] = R0 * np.cos(p0), a * np.cos(p0)
        dofs[i, 1, 0], dofs[i, 1, 2] = R0 * np.sin(p0), a * np.sin(p0)
        dofs[i, 2, 1] = a
    return jnp.asarray(dofs), jnp.full(ncoils, 1.0e5)


def _essos_coil_field(dofs, currents, *, nfp=5, n_segments=64, stellsym=True):
    """A generic ``xyz(...,3) -> B(...,3)`` callable from ESSOS coils.

    vmex carries no coil code; the differentiable free-boundary residual
    consumes coils through the plain-callable interface, differentiating in the
    ESSOS coils' Fourier dofs (rebuilt inside the closure so ``jax.grad`` threads
    through ``essos.coils.Coils`` -> ``essos.fields.BiotSavart``).
    """
    from essos.coils import Coils, Curves
    from essos.fields import BiotSavart

    bs = BiotSavart(Coils(Curves(dofs, n_segments, nfp, stellsym), currents))

    def field(pts):
        return jax.vmap(bs.B)(pts.reshape(-1, 3)).reshape(pts.shape)

    return field


def _synthetic_surface(nphi=12, ntheta=12, nfp=3, R0=1.0, a=0.3, B0=1.0):
    """A circular torus with a purely-toroidal (tangent) field: B_total . n = 0."""
    theta = jnp.linspace(0.0, 2 * jnp.pi, ntheta, endpoint=False)
    phi = jnp.linspace(0.0, 2 * jnp.pi / nfp, nphi, endpoint=False)
    ph, th = jnp.meshgrid(phi, theta, indexing="ij")  # both (nphi, ntheta)
    R = R0 + a * jnp.cos(th)
    Z = a * jnp.sin(th)
    cph, sph = jnp.cos(ph), jnp.sin(ph)
    gamma = jnp.stack([R * cph, R * sph, Z], axis=0)
    e_th = jnp.stack([-a * jnp.sin(th) * cph, -a * jnp.sin(th) * sph, a * jnp.cos(th)], axis=0)
    e_ph = jnp.stack([-R * sph, R * cph, jnp.zeros_like(R)], axis=0)
    area = jnp.cross(e_th, e_ph, axis=0)
    normal = area / jnp.linalg.norm(area, axis=0)
    Btor = B0 * R0 / R
    B_total = jnp.stack([-Btor * sph, Btor * cph, jnp.zeros_like(Btor)], axis=0)  # toroidal, tangent
    return VmecSurfaceFieldData(gamma=gamma, B_total=B_total, normal=normal, area_vector=area,
                                theta=theta, phi=phi, nfp=nfp, stellsym=False, signgs=1,
                                source_convention="synthetic")


def _ring_field(points, *, R0=1.0, current=3.0e5, segments=1024):
    """Independent Biot-Savart field of a circular filament of radius ``R0`` at z = 0.

    The trapezoid rule on a closed smooth loop converges geometrically; at the
    distances used here (at least 0.04 m) 1024 segments are exact to rounding.
    """
    angle = 2.0 * np.pi * np.arange(segments) / segments
    source = np.stack([R0 * np.cos(angle), R0 * np.sin(angle), np.zeros(segments)], axis=1)
    tangent = np.stack([-np.sin(angle), np.cos(angle), np.zeros(segments)], axis=1)
    separation = np.asarray(points)[:, None, :] - source[None]
    kernel = np.cross(tangent[None], separation) / np.linalg.norm(
        separation, axis=-1, keepdims=True) ** 3
    return VC.MU0 * current * R0 / (2.0 * segments) * kernel.sum(axis=1)


def _two_source_torus(nphi, ntheta, *, R0=1.0, a=0.3, current=3.0e5):
    """Torus carrying the field of an outside and an inside current.

    ``_synthetic_surface`` already holds ``B0 R0 / R`` along phi, the field of a
    straight current on the z-axis (outside the surface).  A circular filament
    on the magnetic axis (inside) adds a field with ``B . n != 0``, so both
    virtual-casing layer densities are exercised.  The internal branch must
    return the filament field outside the surface and on it, and minus the
    z-axis field inside.
    """
    surface = _synthetic_surface(nphi=nphi, ntheta=ntheta, nfp=1, R0=R0, a=a)
    gamma = np.asarray(surface.gamma)
    ring = _ring_field(gamma.reshape(3, -1).T, R0=R0, current=current)
    return replace(surface, B_total=surface.B_total + jnp.asarray(ring.T.reshape(gamma.shape)))


def _torus_points(distance, *, R0=1.0, a=0.3, count=8, seed=0):
    """Points at a signed ``distance`` along the outward normal of the circular torus."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, count)
    phi = rng.uniform(0.0, 2.0 * np.pi, count)
    radius = R0 + (a + distance) * np.cos(theta)
    return np.stack([radius * np.cos(phi), radius * np.sin(phi),
                     (a + distance) * np.sin(theta)], axis=1)


def _z_axis_field(points, *, B0R0=1.0):
    x, y = points[:, 0], points[:, 1]
    radius2 = x * x + y * y
    return B0R0 * np.stack([-y / radius2, x / radius2, np.zeros_like(x)], axis=1)


def _finest_spacing(field, *, R0=1.0, a=0.3):
    """Largest source spacing of the finest schedule level (full-torus counts)."""
    n_toroidal, n_poloidal = field.plasma_field.schedule_levels[-1]
    return max(2.0 * np.pi * (R0 + a) / n_toroidal, 2.0 * np.pi * a / n_poloidal)


def _directional_fd(fun, x0, v, h):
    return (float(fun(x0 + h * v)) - float(fun(x0 - h * v))) / (2.0 * h)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_surface_data_reproduces_equilibrium_bnormal():
    """wout->surface-data adapter reproduces the VMEC free-boundary condition."""
    if not WOUT.exists():
        pytest.skip(f"wout fixture unavailable: {WOUT}")
    wout = read_wout(WOUT)
    sd = VC.surface_field_data_from_wout(wout, nphi=24, ntheta=24)
    assert sd.gamma.shape == (3, 24, 24)
    Bn = jnp.sum(sd.B_total * sd.normal, axis=0)
    absB = jnp.linalg.norm(sd.B_total, axis=0)
    rms = float(jnp.sqrt(jnp.mean(Bn**2)) / jnp.sqrt(jnp.mean(absB**2)))
    assert rms < 1e-10, f"B_total . n / |B| = {rms:.2e} (expected ~machine epsilon)"


def test_synthetic_surface_gradient_fd_validates():
    """Asset-free: grad of the free-boundary residual w.r.t. coil dofs vs central FD."""
    sd = _synthetic_surface()
    # The synthetic toroidal field is tangent to the torus by construction.
    Bn = jnp.sum(sd.B_total * sd.normal, axis=0)
    assert float(jnp.max(jnp.abs(Bn))) < 1e-12

    prob = VC.PlasmaVacuumInterface.from_surface_data(sd, digits=3)
    assert prob.Bn_plasma.shape == (12, 12)
    assert bool(jnp.all(jnp.isfinite(prob.Bn_plasma)))

    pytest.importorskip("essos")
    d0, currents = _circular_coil_dofs(nfp=3, R0=1.0, a=0.5)

    def J(dofs):
        return prob.bnormal_objective(_essos_coil_field(dofs, currents, nfp=3))

    g = jax.grad(J)(d0)
    assert bool(jnp.all(jnp.isfinite(g)))
    v = jnp.asarray(np.random.default_rng(0).standard_normal(d0.shape))
    dir_ad = float(jnp.sum(g * v))
    dir_fd = _directional_fd(J, d0, v, 1e-6)
    assert abs(dir_ad - dir_fd) <= 1e-5 * abs(dir_fd) + 1e-9, f"AD {dir_ad:.6e} vs FD {dir_fd:.6e}"


def test_finite_beta_extender_field_and_gradient_outside_lcfs(monkeypatch):
    """The VMEX field composes coil and plasma fields beyond the LCFS."""
    surface = _synthetic_surface(nphi=12, ntheta=12, nfp=1)

    def coil_field(points):
        return jnp.broadcast_to(jnp.array([0.0, 0.0, 0.4]), points.shape)

    field = VmecExtender.from_surface_data(
        surface,
        external_field=coil_field,
        digits=3,
        levels=((13, 13), (26, 26)),
        chunk_size=17,
        target_chunk_size=1,
        near_surface="direct",  # compare the direct path with AD of itself
    )
    points = jnp.array([[1.8, 0.0, 0.1], [0.0, 1.9, -0.1]])
    assert field.uses_virtual_casing

    expected = field.plasma_field.B_plasma_xyz(points) + coil_field(points)
    np.testing.assert_allclose(field.B(points), expected, rtol=2e-11, atol=2e-11)
    jacobian = jax.vmap(
        jax.jacfwd(lambda point: field.B(point[None, :])[0])
    )(points)
    np.testing.assert_allclose(field.gradB(points), jacobian, rtol=2e-5, atol=2e-6)

    direction = jnp.array([0.3, -0.4, 0.2])
    eps = 2.0e-5
    finite_difference = (
        field.B(points[:1] + eps * direction)
        - field.B(points[:1] - eps * direction)
    )[0] / (2.0 * eps)
    autodiff = field.gradB(points[:1])[0] @ direction
    np.testing.assert_allclose(autodiff, finite_difference, rtol=3e-4, atol=3e-6)

    inp = type("Input", (), {"lfreeb": False})()
    wout = type("Wout", (), {"betatotal": 0.01, "mgrid_file": ""})()
    equilibrium = type("Equilibrium", (), {"inp": inp, "state": object(), "wout": wout})()
    monkeypatch.setattr(VC, "surface_field_data_from_state", lambda *_args, **_kwargs: surface)
    live = VmecExtender.from_equilibrium(
        equilibrium,
        external_field=coil_field,
        digits=3,
        levels=((13, 13), (26, 26)),
    )
    assert live.uses_virtual_casing
    live.near_surface = "direct"
    np.testing.assert_allclose(live.B(points), field.B(points), rtol=2e-11, atol=2e-11)

    if hasattr(field.plasma_field, "plan_surface_precision"):
        plan = field.plasma_field.plan_surface_precision(digits=3)
        expected_surface_field = field.plasma_field.B_plasma_on_surface(
            digits=3, precision=plan)
    else:  # released virtual-casing-jax; public plan reuse arrives in PR #5
        plan = VC.plan_vc_precision(surface, digits=3)
        expected_surface_field = field.plasma_field._vc.compute_internal_B(
            field.plasma_field.B_total, digits=3, chunk_size=64, precision=plan)
    interface = VC.PlasmaVacuumInterface.from_surface_data(
        surface, digits=3, precision=plan,
        virtual_casing_field=field.plasma_field,
    )
    np.testing.assert_allclose(
        interface.B_plasma,
        expected_surface_field,
        rtol=2e-11, atol=2e-11,
    )


def test_parameterized_extender_vjp_matches_rebuilt_surface_fd():
    """Exterior B VJP differentiates the moving virtual-casing surface."""
    parameters = jnp.array([1.0])
    def surface(p):
        return _synthetic_surface(nphi=10, ntheta=10, nfp=1, R0=p[0])
    points = jnp.array([[1.8, 0.2, 0.1]])
    field = VmecExtender.from_parameterized_surface_data(
        surface, parameters, digits=3, levels=((11, 11), (22, 22)),
        dof_names=("R0",)).set_points(points)
    cotangent = jnp.ones_like(field.B())
    autodiff = float(field.B_vjp(cotangent)[0])

    def scalar(p):
        rebuilt = VmecExtender.from_surface_data(
            surface(jnp.array([p])), digits=3, levels=((11, 11), (22, 22)))
        return float(jnp.vdot(rebuilt.B(points), cotangent))

    step = 2.0e-5
    finite_difference = (scalar(1.0 + step) - scalar(1.0 - step)) / (2.0 * step)
    np.testing.assert_allclose(autodiff, finite_difference, rtol=3e-4, atol=3e-6)
    assert field.dof_names == ("R0",)


def test_two_source_torus_exterior_interior_and_on_surface_identities():
    """Known answers of the internal branch, away from and on the source surface.

    Targets sit three finest-level spacings off the surface.  Outside, the
    plasma field must be the inside filament's field; inside, minus the z-axis
    field; on the surface the singular (Malhotra et al.) quadrature must also
    give the filament field.  The strict check must not fire.
    """
    digits = 4
    surface = _two_source_torus(24, 24)
    field = VmecExtender.from_surface_data(
        surface, digits=digits, levels=((48, 24), (96, 48)), accuracy_check="raise")
    h = _finest_spacing(field)
    scale = float(np.sqrt(np.mean(np.sum(np.asarray(surface.B_total) ** 2, axis=0))))

    outside = _torus_points(3.0 * h)
    value = np.asarray(field.B(jnp.asarray(outside)))  # raises if the estimate misses
    error = np.linalg.norm(value - _ring_field(outside), axis=1) / scale
    assert error.max() <= 10.0 ** -digits, error
    assert float(np.max(field.B_error_estimate(jnp.asarray(outside)))) <= 10.0 ** -digits

    inside = _torus_points(-3.0 * h)
    value = np.asarray(field.plasma_field.B_plasma_xyz(jnp.asarray(inside)))
    error = np.linalg.norm(value + _z_axis_field(inside), axis=1) / scale
    assert error.max() <= 10.0 ** -digits, error

    on_surface = np.asarray(VC.plasma_field_on_boundary(
        surface, digits=digits, virtual_casing_field=field.plasma_field))
    ring = _ring_field(np.asarray(surface.gamma).reshape(3, -1).T).T.reshape(on_surface.shape)
    error = np.linalg.norm(on_surface - ring, axis=0) / scale
    assert error.max() <= 10.0 ** -digits, error.max()


def _native_helical_spectra(ns=11, nfp=2, major=1.0, minor=0.3, helical=0.05):
    """Native-form spectra of a divergence-free field on a helically shaped torus.

    ``lambda = 0`` with ``phi'`` constant and ``chi'`` varying in ``s``, so
    :class:`~vmex.core.extender.VmecInteriorField` evaluates it exactly in
    its native form: tangent to every surface and divergence-free by
    construction, carrying a volume current and no force balance.
    """
    s = np.linspace(0.0, 1.0, ns)
    rho = np.sqrt(s)
    shape = np.stack([np.full(ns, major), minor * rho * (1.0 + 0.2 * s), helical * rho], axis=1)
    return {
        "nfp": nfp, "ns": ns,
        "xm": jnp.array([0.0, 1.0, 1.0]), "xn": jnp.array([0.0, 0.0, 2.0]),
        "xmn": jnp.zeros(1), "xnn": jnp.zeros(1),
        "rmnc": jnp.asarray(shape), "zmns": jnp.asarray(np.c_[np.zeros(ns), shape[:, 1:]]),
        "rmns": None, "zmnc": None,
        "bsupu": jnp.zeros((ns, 1)), "bsupv": jnp.zeros((ns, 1)),
        "bsupu_s": None, "bsupv_s": None, "lasym": False, "signgs": -1,
        "lmns": jnp.zeros((ns, 3)), "phipf": jnp.full(ns, 0.4),
        "chipf": jnp.asarray(0.25 * s * (1.0 - 0.3 * s)),
    }


def test_exterior_field_is_the_biot_savart_field_of_the_interior_current():
    """Virtual casing of the LCFS field equals the volume Biot-Savart of ``curl B``.

    An oracle independent of any surface quadrature.  For a divergence-free
    field ``B`` tangent to the boundary, the internal-branch virtual-casing
    integral outside the surface is exactly the Biot-Savart field of the
    volume current ``curl B / mu0`` inside it.  The field here is the
    interior field's native form on a helically shaped nfp = 2 torus; the
    surface data are that same field on ``rho = 1``, and the volume integral
    takes ``curl B`` from the covariant components in flux coordinates
    (``sqrt(g) J^i = eps^ijk d_j B_k``, so the Jacobian cancels) with six
    Gauss points per radial spline cell, where the interpolant is smooth.
    Agreement to 1e-9 on the outboard and far targets ties the exterior path
    (virtual_casing_jax, its sign, normal and nfp conventions) to the
    interior one without either side reusing the other's quadrature.  B and
    grad B are both checked.
    """
    from vmex.core import extender as ext

    ns, nfp = 11, 2
    spectra = ext._prepared(_native_helical_spectra(ns, nfp))
    position = lambda w: ext._position_and_field(spectra, w)[0]  # noqa: E731

    nphi = ntheta = 24
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, ntheta, endpoint=False)
    phi = jnp.linspace(0.0, 2.0 * jnp.pi / nfp, nphi, endpoint=False)
    grid_phi, grid_theta = jnp.meshgrid(phi, theta, indexing="ij")
    edge = jnp.stack([jnp.ones(grid_phi.size), grid_theta.ravel(), grid_phi.ravel()], axis=1)

    def edge_point(w):
        x, B = ext._position_and_field(spectra, w)
        tangents = jax.jacfwd(position)(w)
        return x, B, jnp.cross(tangents[:, 1], tangents[:, 2]), tangents[:, 0]

    x, B, area, radial = jax.vmap(edge_point)(edge)
    area = area * jnp.sign(jnp.mean(jnp.sum(area * radial, axis=1)))  # outward
    soa = lambda v: jnp.moveaxis(v.reshape(nphi, ntheta, 3), -1, 0)  # noqa: E731
    surface = VmecSurfaceFieldData(
        gamma=soa(x), B_total=soa(B),
        normal=soa(area / jnp.linalg.norm(area, axis=1, keepdims=True)),
        area_vector=soa(area), theta=theta, phi=phi, nfp=nfp, stellsym=False,
        signgs=-1, source_convention="synthetic")
    assert float(jnp.max(jnp.abs(jnp.sum(B * area, axis=1)))) < 1e-12  # tangent

    nodes, weights = np.polynomial.legendre.leggauss(6)
    cells = np.linspace(0.0, 1.0, ns)
    s = np.concatenate([a + (b - a) * (nodes + 1.0) / 2.0 for a, b in zip(cells[:-1], cells[1:])])
    ds = np.concatenate([(b - a) / 2.0 * weights for a, b in zip(cells[:-1], cells[1:])])
    n_theta, n_phi = 48, 144
    rho, grid_t, grid_p = np.meshgrid(
        np.sqrt(s), 2.0 * np.pi * np.arange(n_theta) / n_theta,
        2.0 * np.pi * np.arange(n_phi) / n_phi, indexing="ij")
    volume = jnp.stack([rho.ravel(), grid_t.ravel(), grid_p.ravel()], axis=1)

    def covariant(w):
        x, B = ext._position_and_field(spectra, w)
        return jax.jacfwd(position)(w).T @ B

    def current_element(w):
        tangents = jax.jacfwd(position)(w)
        d = jax.jacfwd(covariant)(w)  # d[k, j] = d_j B_k
        curl = jnp.array([d[2, 1] - d[1, 2], d[0, 2] - d[2, 0], d[1, 0] - d[0, 1]])
        return position(w), jnp.sign(jnp.linalg.det(tangents)) * (tangents @ curl)

    sources, current = jax.jit(jax.vmap(current_element))(volume)
    # d(rho) = ds / (2 rho); angles by the periodic trapezoid rule
    weight = np.repeat(ds / (2.0 * np.sqrt(s)), n_theta * n_phi) * (
        2.0 * np.pi / n_theta) * (2.0 * np.pi / n_phi)
    current = current * jnp.asarray(weight)[:, None]

    def biot_savart(point):
        r = point[None, :] - sources
        return jnp.sum(jnp.cross(current, r) / (4.0 * jnp.pi * jnp.linalg.norm(
            r, axis=1, keepdims=True) ** 3), axis=0)

    outer = float(np.asarray(spectra["rmnc"])[-1].sum())
    points = jnp.asarray([[outer + 0.3, 0.0, 0.0],
                          [0.0, outer + 0.2, 0.05],
                          [0.3, -1.2, 0.55]])
    field = VmecExtender.from_surface_data(
        surface, digits=12, levels=((96, 48), (192, 96)), accuracy_check="off")
    B_vc, gradB_vc = np.asarray(field.B(points)), np.asarray(field.gradB(points))
    B_bs = np.asarray(jax.vmap(biot_savart)(points))
    gradB_bs = np.asarray(jax.vmap(jax.jacfwd(biot_savart))(points))
    error = np.linalg.norm(B_vc - B_bs, axis=1) / np.linalg.norm(B_bs, axis=1)
    assert error.max() <= 1e-9, error
    error = (np.linalg.norm((gradB_vc - gradB_bs).reshape(3, -1), axis=1)
             / np.linalg.norm(gradB_bs.reshape(3, -1), axis=1))
    assert error.max() <= 1e-9, error


def test_exterior_error_estimate_flags_unresolved_targets_and_fails_loudly():
    """Targets inside one grid spacing are flagged, warned about or refused; values unchanged."""
    digits = 4
    surface = _synthetic_surface(nphi=12, ntheta=12, nfp=1)
    # the direct path alone: near_surface="auto" would switch these points
    field = VmecExtender.from_surface_data(
        surface, digits=digits, accuracy_check="off", near_surface="direct")
    near = jnp.asarray(_torus_points(0.5 * _finest_spacing(field)))
    scale = float(np.sqrt(np.mean(np.sum(np.asarray(surface.B_total) ** 2, axis=0))))

    quiet = np.asarray(field.B(near))
    # the exact plasma field outside is zero: the flag reports a real error
    error = np.linalg.norm(quiet, axis=1) / scale
    estimate = np.asarray(field.B_error_estimate(near))
    assert estimate.shape == (near.shape[0],)
    assert np.all(error > 10.0 ** -digits) and np.all(estimate > 10.0 ** -digits)

    field.accuracy_check = "warn"
    with pytest.warns(ExteriorFieldAccuracyWarning, match="above the requested 1e-4"):
        loud = np.asarray(field.B(near))
    np.testing.assert_array_equal(loud, quiet)
    field.accuracy_check = "raise"
    with pytest.raises(ExteriorFieldAccuracyError, match="8 of 8 points"):
        field.B(near)

    # traced calls never check; they expose the estimate instead
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        traced = np.asarray(jax.jit(field.B)(near))
    np.testing.assert_allclose(traced, quiet, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        jax.jit(field.B_error_estimate)(near), estimate, rtol=1e-10, atol=1e-14)

    with pytest.raises(ValueError, match="accuracy_check"):
        field.accuracy_check = "loud"
    with pytest.raises(ValueError, match="near_surface"):
        field.near_surface = "taylor"
    with pytest.raises(RuntimeError, match="no virtual-casing"):
        VmecExtender(lambda xyz: jnp.zeros_like(xyz)).B_error_estimate(near)


@pytest.mark.full
def test_cth_gradient_fd_validates():
    """cth-like case: free-boundary residual gradients (extcur + coil dofs) vs central FD."""
    if not WOUT.exists():
        pytest.skip(f"wout fixture unavailable: {WOUT}")
    wout = read_wout(WOUT)
    prob = VC.PlasmaVacuumInterface.from_wout(wout, nphi=24, ntheta=24, digits=4)

    # (a) extcur via the cth mgrid (2 coil-group currents) — exact full gradient.
    if MGRID.exists():
        base = MgridField.from_mgrid_data(read_mgrid(MGRID), extcur=jnp.array([4700.0, 1000.0]))

        def J_extcur(extcur):
            mf = MgridField(br=base.br, bp=base.bp, bz=base.bz, extcur=extcur,
                            rmin=base.rmin, rmax=base.rmax, zmin=base.zmin, zmax=base.zmax, nfp=base.nfp)
            return prob.bnormal_objective(mf)

        x0 = jnp.array([4700.0, 1000.0])
        g_ad = np.asarray(jax.grad(J_extcur)(x0))
        g_fd = np.array([_directional_fd(J_extcur, x0, jnp.asarray(np.eye(2)[i]), 1.0) for i in range(2)])
        rel = np.abs(g_ad - g_fd) / (np.abs(g_fd) + 1e-30)
        assert np.all(rel < 1e-4), f"extcur grad rel err {rel} (AD {g_ad}, FD {g_fd})"

    # (b) coil Fourier dofs via Biot-Savart (ESSOS coils, callable interface) —
    # directional derivative.
    pytest.importorskip("essos")
    d0, currents = _circular_coil_dofs(nfp=int(wout.nfp))
    nfp = int(wout.nfp)

    def J_dofs(dofs):
        return prob.bnormal_objective(_essos_coil_field(dofs, currents, nfp=nfp))

    g = jax.grad(J_dofs)(d0)
    v = jnp.asarray(np.random.default_rng(1).standard_normal(d0.shape))
    dir_ad = float(jnp.sum(g * v))
    dir_fd = _directional_fd(J_dofs, d0, v, 1e-6)
    assert abs(dir_ad - dir_fd) <= 1e-5 * abs(dir_fd) + 1e-9, f"coil-dof AD {dir_ad:.6e} vs FD {dir_fd:.6e}"

    # (c) pressure-balance residual is finite and differentiable too.
    jp = jax.grad(lambda d: prob.pressure_balance_objective(_essos_coil_field(d, currents, nfp=nfp)))(d0)
    assert bool(jnp.all(jnp.isfinite(jp)))


# ---------------------------------------------------------------------------
# curl-free (current-conserving) projection of the LCFS source data
# ---------------------------------------------------------------------------


def _covariant_pair(nphi, ntheta, nfp, coefficients):
    """Build ``(B_theta, B_phi)`` from named Fourier content."""
    theta = jnp.linspace(0.0, 2 * jnp.pi, ntheta, endpoint=False)
    phi = jnp.linspace(0.0, 2 * jnp.pi / nfp, nphi, endpoint=False)
    ph, th = jnp.meshgrid(phi, theta, indexing="ij")
    b_theta = jnp.zeros_like(th)
    b_phi = jnp.zeros_like(th)
    for (m, n), (gradient, curl) in coefficients.items():
        angle = m * th - n * nfp * ph
        # grad_s of cos(m theta - n nfp phi) / 1, plus a divergence-free part
        b_theta = b_theta - gradient * m * jnp.sin(angle) - curl * n * nfp * jnp.sin(angle)
        b_phi = b_phi + gradient * n * nfp * jnp.sin(angle) - curl * m * jnp.sin(angle)
    return b_theta, b_phi


def _surface_current_divergence(b_theta, b_phi, nfp):
    """``d_theta B_phi - d_phi B_theta`` spectrally; zero iff the pair is a gradient."""
    nphi, ntheta = b_theta.shape
    k_theta = jnp.fft.fftfreq(ntheta, 1.0 / ntheta)[None, :]
    k_phi = (jnp.fft.fftfreq(nphi, 1.0 / nphi) * nfp)[:, None]
    residual = 1j * k_theta * jnp.fft.fft2(b_phi) - 1j * k_phi * jnp.fft.fft2(b_theta)
    return float(jnp.max(jnp.abs(jnp.fft.ifft2(residual))))


def test_projection_keeps_a_surface_gradient_and_removes_the_rest():
    """The projection is the identity on gradients and kills the divergence-free part."""
    nfp = 3
    gradient_only = _covariant_pair(16, 16, nfp, {(1, 0): (0.7, 0.0), (2, 1): (0.3, 0.0)})
    assert _surface_current_divergence(*gradient_only, nfp) < 1e-12
    kept = VC._project_covariant_to_surface_gradient(*gradient_only, nfp)
    np.testing.assert_allclose(np.asarray(kept[0]), np.asarray(gradient_only[0]),
                               rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(kept[1]), np.asarray(gradient_only[1]),
                               rtol=0.0, atol=1e-12)

    mixed = _covariant_pair(16, 16, nfp, {(1, 0): (0.7, 0.0), (2, 1): (0.3, 0.5)})
    assert _surface_current_divergence(*mixed, nfp) > 1e-3
    projected = VC._project_covariant_to_surface_gradient(*mixed, nfp)
    assert _surface_current_divergence(*projected, nfp) < 1e-12
    # what survives is exactly the gradient part that was put in
    np.testing.assert_allclose(np.asarray(projected[0]), np.asarray(gradient_only[0]),
                               rtol=0.0, atol=1e-12)


def test_projection_preserves_the_net_currents():
    """The (0, 0) mode is the net poloidal and toroidal current, and is not a gradient."""
    nfp = 2
    b_theta, b_phi = _covariant_pair(12, 12, nfp, {(1, 1): (0.4, 0.6)})
    b_theta, b_phi = b_theta + 3.0, b_phi - 5.0
    projected = VC._project_covariant_to_surface_gradient(b_theta, b_phi, nfp)
    assert abs(float(jnp.mean(projected[0])) - 3.0) < 1e-12
    assert abs(float(jnp.mean(projected[1])) + 5.0) < 1e-12


def test_projection_is_off_by_default_and_differentiable_when_on():
    """The keyword changes the field; it must not change differentiability."""
    surface = _synthetic_surface(nphi=12, ntheta=12, nfp=2)

    def assemble(scale, project):
        return VC._assemble_surface_field_data(
            nfp=2, ns=3, j=2,
            xm=jnp.array([0.0, 1.0]), xn=jnp.array([0.0, 2.0]),
            xmn=jnp.array([0.0, 1.0]), xnn=jnp.array([0.0, 2.0]),
            rmnc=jnp.array([[1.0, 0.0], [1.0, 0.1], [1.0, 0.3 * scale]]),
            zmns=jnp.array([[0.0, 0.0], [0.0, 0.1], [0.0, 0.3 * scale]]),
            rmns=None, zmnc=None,
            bsupu=jnp.array([[0.0, 0.0], [0.5, 0.05], [1.0, 0.1]]),
            bsupv=jnp.array([[0.0, 0.0], [0.5, 0.05], [1.0, 0.2]]),
            bsupu_s=None, bsupv_s=None, lasym=False, use_stellsym=True, signgs=1,
            nphi=12, ntheta=12, source_convention="synthetic",
            project_current=project).B_total

    default = assemble(1.0, False)
    np.testing.assert_array_equal(np.asarray(default),
                                  np.asarray(assemble(1.0, False)))
    assert not np.allclose(np.asarray(assemble(1.0, True)), np.asarray(default))

    for project in (False, True):
        cost = lambda x, p=project: jnp.sum(assemble(x, p) ** 2)  # noqa: E731
        ad = float(jax.grad(cost)(1.0))
        step = 1e-6
        fd = float((cost(1.0 + step) - cost(1.0 - step)) / (2.0 * step))
        assert abs(ad - fd) <= 1e-6 * abs(fd), (project, ad, fd)
    assert surface.nfp == 2


# ---------------------------------------------------------------------------
# per-order accuracy: B has been checked since the estimate landed, its
# derivatives never were, and they are the ones that lose accuracy fastest
# ---------------------------------------------------------------------------


def _per_order_available(field):
    """Whether the installed virtual-casing-jax carries the a-priori estimate.

    Probed at a realistic stand-off.  Far targets are covered by
    :func:`test_the_derivative_check_stays_quiet_many_surface_sizes_away`.
    """
    probe = _torus_points(3.0 * _finest_spacing(field), count=1)
    try:
        field.B_error_estimate(jnp.asarray(probe), order=1)
    except NotImplementedError:
        return False
    return True


def test_the_estimate_grows_with_derivative_order():
    """A grid that gives the field its digits need not give its curvature them."""
    surface = _synthetic_surface(nphi=16, ntheta=16, nfp=2)
    field = VmecExtender.from_surface_data(
        surface, digits=4, accuracy_check="off", near_surface="direct")
    if not _per_order_available(field):
        pytest.skip("per-order estimate needs virtual-casing-jax >= 0.0.7")
    points = jnp.asarray(_torus_points(2.0 * _finest_spacing(field)))

    by_order = [float(np.max(np.asarray(field.B_error_estimate(points, order=k))))
                for k in range(4)]
    assert by_order == sorted(by_order), by_order
    assert by_order[3] > by_order[0]


def test_a_traced_per_order_estimate_says_so_instead_of_failing_obscurely():
    """The host-side path must name the alternative, not raise from inside NumPy."""
    surface = _synthetic_surface(nphi=12, ntheta=12, nfp=1)
    field = VmecExtender.from_surface_data(surface, digits=4, accuracy_check="off")
    if not _per_order_available(field):
        pytest.skip("per-order estimate needs virtual-casing-jax >= 0.0.7")
    points = jnp.asarray(_torus_points(3.0 * _finest_spacing(field)))

    # order 0 stays traceable, as it always was
    np.testing.assert_allclose(
        np.asarray(jax.jit(field.B_error_estimate)(points)),
        np.asarray(field.B_error_estimate(points)), rtol=1e-10, atol=1e-14)

    with pytest.raises(NotImplementedError, match="cannot be traced"):
        jax.jit(lambda p: field.B_error_estimate(p, order=2))(points)


def test_eager_derivatives_check_at_their_own_order():
    """gradgradgradB warns where B does not, on the same points and grid."""
    surface = _synthetic_surface(nphi=12, ntheta=12, nfp=1)
    field = VmecExtender.from_surface_data(surface, digits=4, accuracy_check="off")
    if not _per_order_available(field):
        pytest.skip("per-order estimate needs virtual-casing-jax >= 0.0.7")
    field.near_surface = "direct"  # the check of the direct path itself
    points = jnp.asarray(_torus_points(1.2 * _finest_spacing(field)))

    quiet_B = np.asarray(field.B(points))
    quiet_third = np.asarray(field.gradgradgradB(points))
    field.accuracy_check = "warn"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loud_third = field.gradgradgradB(points)
    assert any(issubclass(w.category, ExteriorFieldAccuracyWarning) for w in caught)
    assert any("order-3 derivative" in str(w.message) for w in caught)
    # the check reports; it does not change the value
    np.testing.assert_array_equal(np.asarray(loud_third), quiet_third)
    assert np.all(np.isfinite(quiet_B))


def test_the_derivative_check_does_not_fire_far_from_the_surface():
    """Otherwise it would be noise rather than a signal."""
    surface = _synthetic_surface(nphi=16, ntheta=16, nfp=1)
    field = VmecExtender.from_surface_data(surface, digits=3, accuracy_check="warn")
    if not _per_order_available(field):
        pytest.skip("per-order estimate needs virtual-casing-jax >= 0.0.7")
    points = jnp.asarray(_torus_points(6.0 * _finest_spacing(field)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        field.gradB(points)
        field.gradgradB(points)


def test_the_derivative_check_stays_quiet_many_surface_sizes_away():
    """Ten and a hundred minor radii out, the a-priori estimate is tiny, not NaN.

    virtual-casing-jax 0.0.7 started its complex Newton so far off the real
    axis for such a target that the boundary series overflowed: hundreds of
    NumPy RuntimeWarnings per call and a NaN estimate, which the eager
    derivative check read as a missed tolerance and reported as an error "up
    to inf" at points where the field is exact to rounding.  0.0.8, the floor,
    fixes it upstream.
    """
    surface = _synthetic_surface(nphi=16, ntheta=16, nfp=1)
    field = VmecExtender.from_surface_data(surface, digits=3, accuracy_check="warn")
    if not _per_order_available(field):
        pytest.skip("per-order estimate needs virtual-casing-jax >= 0.0.7")
    points = jnp.asarray(np.concatenate([_torus_points(3.0, count=2),
                                         _torus_points(30.0, count=2)]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for order in (1, 2, 3):
            estimate = np.asarray(field.B_error_estimate(points, order=order))
            assert np.all(np.isfinite(estimate)) and estimate.max() <= 1e-20, estimate
        field.gradgradgradB(points)


@pytest.mark.full  # weekly: one reverse pass through the implicit li383 equilibrium
def test_exterior_field_parameter_derivative_is_exact_on_the_frozen_path():
    """The exterior field's derivative in boundary and current parameters is exact.

    Measured against :func:`~vmex.core.implicit.frozen_path_directional_fd`,
    which re-roots the frozen residual at the perturbed parameters: the
    implicit reverse pass through the equilibrium, the live-state surface data
    and virtual casing agree with it to about 1e-6 on ``RBC(-1,1)`` and to
    about 1e-10 on the net current.  An independent re-solve differs by a
    factor of four on the boundary direction and by 3e-3 on the current: each
    re-solve freezes the released m = 1 ``Z_sin`` family wherever its own path
    left it (#428), and the exterior field feels that gauge drift through the
    edge data far more than iota does.  That is a property of the discrete
    re-solve map, recorded here so that nobody reads the re-solve difference
    as a defect of the derivative.
    """
    from vmex.core import implicit as im
    from vmex.core.input import VmecInput

    inp = VmecInput.from_file(str(REPO / "examples" / "data" / "input.li383_low_res"))
    config = im.make_config(inp, ftol=1e-13, max_iterations=6000)
    base = im.params_from_input(inp)
    state = im.solve_implicit(base, config)
    runtime = im.runtime_from_params(base, config)
    surface = VC.surface_field_data_from_state(inp, state, runtime=runtime, nphi=32, ntheta=32)
    gamma = np.asarray(surface.gamma).reshape(3, -1).T
    normal = np.asarray(surface.normal).reshape(3, -1).T
    minor = 0.5 * float(np.ptp(np.hypot(gamma[:32, 0], gamma[:32, 1])))
    chosen = np.random.default_rng(1).choice(len(gamma), size=4, replace=False)
    points = jnp.asarray(gamma[chosen] + minor * np.array([1.0, 1.0, 0.6, 0.6])[:, None]
                         * normal[chosen])
    weight = jnp.asarray(np.random.default_rng(2).normal(size=(4, 3)))

    def exterior(state, runtime):
        data = VC.surface_field_data_from_state(inp, state, runtime=runtime, nphi=32, ntheta=32)
        field = VmecExtender.from_surface_data(data, accuracy_check="off")
        return jnp.vdot(field.B(points), weight)

    gradient = jax.grad(lambda p: exterior(
        im.solve_implicit(p, config), im.runtime_from_params(p, config)))(base)
    zero = jax.tree.map(jnp.zeros_like, base)
    ntor = int(inp.ntor)
    for tangent, step, tolerance in (
            (dataclasses.replace(zero, rbc=zero.rbc.at[ntor - 1, 1].set(1.0)), 1e-4, 1e-5),
            (dataclasses.replace(zero, curtor=jnp.asarray(1.0e4)), 1e-3, 1e-8)):
        implicit = float(sum(jnp.vdot(g, t) for g, t in zip(
            jax.tree.leaves(gradient), jax.tree.leaves(tangent))))
        frozen, info = im.frozen_path_directional_fd(base, config, exterior, tangent, h=step)
        assert max(info["newton_res"]) < 1e-10
        assert abs(implicit - frozen) <= tolerance * abs(frozen), (implicit, frozen)

