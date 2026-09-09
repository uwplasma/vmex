#!/usr/bin/env python
"""Quasi-axisymmetric boundary optimization with a magnetic well."""

from dataclasses import replace
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import vmex as vj
from vmex import optimize as opt

nfp = 2  # number of field periods
SURFACES = np.linspace(0.1, 1.0, 10)
MAX_MODES, MAX_NFEV = [1,2,3], [10, 10, 15]
MAGNETIC_WELL_TARGET = 0.01
ASPECT_TARGET = 5.0
# MAX_MODES, MAX_NFEV = [1,2,3,4,5,6,7,8,9], [10, 10, 15, 20, 25, 40, 50, 60, 60]
# MAGNETIC_WELL_TARGET = 0.07
# ASPECT_TARGET = 3.5
IOTA_FLOOR = 0.42
PARAMETER_STEP, MAX_PARAMETER_CHANGE = 0.02, 5.0
ESS_ALPHA = 1.2  # smaller values let high Fourier modes move more
MINIMUM_MPOL = 5
VARY_MAJOR_RADIUS = False  # set True to optimize RBC(0,0) instead of fixing it
SEED_PERTURBATION = 0.05

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODES, MAX_NFEV = [1], [4]

DATA = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{nfp}"
inp = vj.VmecInput.from_file(DATA)
# The exactly circular torus has zero first-order iota sensitivity. This
# explicit rotating-ellipse perturbation gives the local optimizer a QA basin.
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
inp = replace(inp, rbc=rbc, zbs=zbs)

# Floor the profile minimum, not its average: a mean target is satisfiable while
# an interior surface sits near zero transform, which is what a current-carried
# finite-beta profile does. opt.mean_iota targets the average instead, and
# opt.soft_min_abs_iota is the smooth-minimum variant.
def iota_floor(equilibrium_state, solver_context):
    return jnp.maximum(
        IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)

import jax
from simsopt.geo import SurfaceRZFourier
from vmex.core.statephysics import _aspect_scalars
from vmex.core.solver import _geometry
from jaxopt import ScipyMinimize as minimize
MINIMUM_DISTANCE = 0.05
DISTANCE_WALL_SCALE = 0.01
DISTANCE_WALL_WEIGHT = 10.0
PCA_WEIGHT = 1.0
VOLUME_WEIGHT = 0.02
SPECTRAL_WEIGHT = 0.02
WINDING_NTHETA = 24
WINDING_NPHI = 24

def r_surface(rsurfacecc, nfp, tor_angle, pol_angle, rsurfacecs = None, n_min = None, m_min = 0):
    num_m_modes = len(rsurfacecc)
    num_n_modes = len(rsurfacecc[0])
    if n_min is None:
        n_min = -(num_n_modes - 1)/2
    r = 0
    for m in range(num_m_modes):
        for n in range(num_n_modes):
            r += rsurfacecc[m][n]*jnp.cos((m_min+m)*pol_angle - (n+n_min)*nfp*tor_angle)
    if rsurfacecs is not None:
        assert len(rsurfacecc) == len(rsurfacecs), 'If rsurfacecs is specified, it must be the same length as rsurfacecc.'
        for m in range(num_m_modes):
            for n in range(num_n_modes):
                r += rsurfacecs[m][n]*jnp.sin((m_min+m)*pol_angle - (n+n_min)*nfp*tor_angle)
    return r

def z_surface(zsurfacecs, nfp, tor_angle, pol_angle, zsurfacecc = None, n_min = None, m_min=0):
    num_m_modes = len(zsurfacecs)
    num_n_modes = len(zsurfacecs[0])
    if n_min is None:
        n_min = -(num_n_modes - 1)/2
    z = 0
    for m in range(num_m_modes):
        for n in range(num_n_modes):
            z += zsurfacecs[m][n]*jnp.sin((m_min+m)*pol_angle - (n+n_min)*nfp*tor_angle)
    if zsurfacecc is not None:
        assert len(zsurfacecs) == len(zsurfacecc), 'If zsurfacecc is specified, it must be the same length as zsurfacecs'
        for m in range(num_m_modes):
            for n in range(num_n_modes):
                z += zsurfacecc[m][n]*jnp.cos((m_min+m)*pol_angle - (n+n_min)*nfp*tor_angle)
    return z

def r_surface_prime_phi(rsurfacecc, nfp, tor_angle, pol_angle, rsurfacecs = None, n_min = None, m_min = 0):
    num_m_modes = len(rsurfacecc)
    num_n_modes = len(rsurfacecc[0])
    if n_min is None:
        n_min = -(num_n_modes - 1)/2
    r_phi = 0
    for m in range(num_m_modes):
        for n in range(num_n_modes):
            r_phi += rsurfacecc[m][n]*(n+n_min)*nfp*jnp.sin((m_min+m)*pol_angle - (n+n_min)*nfp*tor_angle)
    if rsurfacecs is not None:
        assert len(rsurfacecc) == len(rsurfacecs), 'If rsurfacecs is specified, it must be the same length as rsurfacecc.'
        for m in range(num_m_modes):
            for n in range(num_n_modes):
                r_phi += -rsurfacecs[m][n]*(n+n_min)*nfp*jnp.cos((m_min+m)*pol_angle - (n+n_min)*nfp*tor_angle)
    return r_phi

def z_surface_prime_phi(zsurfacecs, nfp, tor_angle, pol_angle, zsurfacecc = None, n_min = None, m_min=0):
    num_m_modes = len(zsurfacecs)
    num_n_modes = len(zsurfacecs[0])
    if n_min is None:
        n_min = -(num_n_modes - 1)/2
    z_phi = 0
    for m in range(num_m_modes):
        for n in range(num_n_modes):
            z_phi += -zsurfacecs[m][n]*(n+n_min)*nfp*jnp.cos((m_min+m)*pol_angle - (n+n_min)*nfp*tor_angle)
    if zsurfacecc is not None:
        assert len(zsurfacecs) == len(zsurfacecc), 'If zsurfacecc is specified, it must be the same length as zsurfacecs'
        for m in range(num_m_modes):
            for n in range(num_n_modes):
                z_phi += zsurfacecc[m][n]*(n+n_min)*nfp*jnp.sin((m_min+m)*pol_angle - (n+n_min)*nfp*tor_angle)
    return z_phi

def surface_del_phi(rsurfacecc, zsurfacecs, nfp, tor_angle, pol_angle, rsurfacecs = None, zsurfacecc = None, n_min = None, m_min = 0):
    r = r_surface(rsurfacecc, nfp, tor_angle, pol_angle, rsurfacecs=rsurfacecs, n_min=n_min, m_min=m_min)
    r_phi = r_surface_prime_phi(rsurfacecc, nfp, tor_angle, pol_angle, rsurfacecs=rsurfacecs, n_min=n_min, m_min=m_min)
    z_phi = z_surface_prime_phi(zsurfacecs, nfp, tor_angle, pol_angle, zsurfacecc=zsurfacecc, n_min=n_min, m_min=m_min)
    costerm = jnp.cos(tor_angle)
    sinterm = jnp.sin(tor_angle)
    return [r_phi*costerm - r*sinterm, r_phi*sinterm + r*costerm, z_phi]

def r_surface_prime_theta(rsurfacecc, nfp, tor_angle, pol_angle, rsurfacecs = None, n_min = None, m_min = 0):
    num_m_modes = len(rsurfacecc)
    num_n_modes = len(rsurfacecc[0])
    if n_min is None:
        n_min = -(num_n_modes - 1)/2
    r_theta = 0
    for m in range(num_m_modes):
        for n in range(num_n_modes):
            r_theta += -rsurfacecc[m][n]*(m+m_min)*jnp.sin((m_min+m)*pol_angle - (n+n_min)*nfp*tor_angle)
    if rsurfacecs is not None:
        assert len(rsurfacecc) == len(rsurfacecs), 'If rsurfacecs is specified, it must be the same length as rsurfacecc.'
        for m in range(num_m_modes):
            for n in range(num_n_modes):
                r_theta += rsurfacecs[m][n]*(m+m_min)*jnp.cos((m_min+m)*pol_angle - (n+n_min)*nfp*tor_angle)
    return r_theta

def z_surface_prime_theta(zsurfacecs, nfp, tor_angle, pol_angle, zsurfacecc = None, n_min = None, m_min=0):
    num_m_modes = len(zsurfacecs)
    num_n_modes = len(zsurfacecs[0])
    if n_min is None:
        n_min = -(num_n_modes - 1)/2
    z_theta = 0
    for m in range(num_m_modes):
        for n in range(num_n_modes):
            z_theta += zsurfacecs[m][n]*(m+m_min)*jnp.cos((m_min+m)*pol_angle - (n+n_min)*nfp*tor_angle)
    if zsurfacecc is not None:
        assert len(zsurfacecs) == len(zsurfacecc), 'If zsurfacecc is specified, it must be the same length as zsurfacecs'
        for m in range(num_m_modes):
            for n in range(num_n_modes):
                z_theta += -zsurfacecc[m][n]*(m+m_min)*jnp.sin((m_min+m)*pol_angle - (n+n_min)*nfp*tor_angle)
    return z_theta

def surface_del_theta(rsurfacecc, zsurfacecs, nfp, tor_angle, pol_angle, rsurfacecs = None, zsurfacecc = None, n_min = None, m_min = 0):
    r_theta = r_surface_prime_theta(rsurfacecc, nfp, tor_angle, pol_angle, rsurfacecs=rsurfacecs, n_min=n_min, m_min=m_min)
    z_theta = z_surface_prime_theta(zsurfacecs, nfp, tor_angle, pol_angle, zsurfacecc=zsurfacecc, n_min=n_min, m_min=m_min)
    return [r_theta*jnp.cos(tor_angle), r_theta*jnp.sin(tor_angle), z_theta]

def surface_normal(rsurfacecc, zsurfacecs, nfp, tor_angle, pol_angle, rsurfacecs = None, zsurfacecc = None, n_min = None, m_min = 0):
    del_phi = surface_del_phi(rsurfacecc, zsurfacecs, nfp, tor_angle, pol_angle, rsurfacecs = rsurfacecs, zsurfacecc = zsurfacecc, n_min = n_min, m_min = m_min)
    del_theta = surface_del_theta(rsurfacecc, zsurfacecs, nfp, tor_angle, pol_angle, rsurfacecs = rsurfacecs, zsurfacecc = zsurfacecc, n_min = n_min, m_min = m_min)
    return [del_phi[1]*del_theta[2]-del_phi[2]*del_theta[1],
            del_phi[2]*del_theta[0]-del_phi[0]*del_theta[2],
            del_phi[0]*del_theta[1]-del_phi[1]*del_theta[0]]

def surface_unitnormal(rsurfacecc, zsurfacecs, nfp, tor_angle, pol_angle, rsurfacecs = None, zsurfacecc = None, n_min = None, m_min = 0):
    normal = jnp.array(surface_normal(rsurfacecc, zsurfacecs, nfp, tor_angle, pol_angle, rsurfacecs = rsurfacecs, zsurfacecc = zsurfacecc, n_min = n_min, m_min = m_min))
    return normal / jnp.sqrt(normal.dot(normal))






def _ntor_from_coefficients(coefficients):
    return (coefficients.shape[1] - 1) // 2


def _symmetric_rc_modes(mpol, ntor):
    modes = [(0, n) for n in range(0, ntor + 1)]
    modes.extend((m, n) for m in range(1, mpol + 1)
                 for n in range(-ntor, ntor + 1))
    return modes


def _symmetric_zs_modes(mpol, ntor):
    modes = [(0, n) for n in range(1, ntor + 1)]
    modes.extend((m, n) for m in range(1, mpol + 1)
                 for n in range(-ntor, ntor + 1))
    return modes


def _basis_matrix(phi_grid, theta_grid, modes, nfp, kind):
    phi = phi_grid.reshape(-1)
    theta = theta_grid.reshape(-1)
    columns = []
    for m, n in modes:
        angle = m * theta - n * nfp * phi
        if kind == "cos":
            columns.append(jnp.cos(angle))
        elif kind == "sin":
            columns.append(jnp.sin(angle))
        else:
            raise ValueError(f"Unknown Fourier basis kind: {kind}")
    return jnp.stack(columns, axis=1)


def _fit_coefficients(values, phi_grid, theta_grid, modes, mpol, ntor, nfp, kind):
    basis = _basis_matrix(phi_grid, theta_grid, modes, nfp, kind)
    fitted = jnp.linalg.lstsq(basis, values.reshape(-1), rcond=None)[0]
    coefficients = jnp.zeros((mpol + 1, 2 * ntor + 1), dtype=values.dtype)
    for index, (m, n) in enumerate(modes):
        coefficients = coefficients.at[m, n + ntor].set(fitted[index])
    return coefficients


def _evaluate_offset_grid(rc, zs, offset, nfp, rs, zc, ntheta, nphi,
                          normal_sign):
    phi = jnp.linspace(0.0, 2.0 * jnp.pi / nfp, nphi, endpoint=False)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, ntheta, endpoint=False)
    phi_grid, theta_grid = jnp.meshgrid(phi, theta, indexing="ij")

    def evaluate(tor_angle, pol_angle):
        radius = r_surface(rc, nfp, tor_angle, pol_angle, rsurfacecs=rs)
        height = z_surface(zs, nfp, tor_angle, pol_angle, zsurfacecc=zc)
        normal = jnp.asarray(surface_normal(
            rc, zs, nfp, tor_angle, pol_angle, rsurfacecs=rs,
            zsurfacecc=zc))
        unit_normal = normal / jnp.linalg.norm(normal)

        x = radius * jnp.cos(tor_angle)
        y = radius * jnp.sin(tor_angle)
        z = height

        x_offset = x + normal_sign * offset * unit_normal[0]
        y_offset = y + normal_sign * offset * unit_normal[1]
        z_offset = z + normal_sign * offset * unit_normal[2]

        return (jnp.sqrt(x_offset ** 2 + y_offset ** 2),
                z_offset,
                jnp.arctan2(y_offset, x_offset))

    evaluate_theta = jax.vmap(evaluate, in_axes=(None, 0))
    offset_radius, offset_height, offset_phi = jax.vmap(
        evaluate_theta, in_axes=(0, None))(phi, theta)
    return offset_radius, offset_height, offset_phi, theta_grid


def extend_via_normal_jax(rc, zs, offset, nfp, *, rs=None, zc=None,
                          stellsym=True, mpol_out=None, ntor_out=None,
                          ntheta=64, nphi=64, normal_sign=1.0):
    """Offset an RZ Fourier surface along its normal and refit coefficients.

    Parameters
    ----------
    rc, zs : array-like
        R-cosine and Z-sine Fourier coefficient arrays with shape
        ``(mpol + 1, 2 * ntor + 1)``.
    offset : float
        Distance to move along the unit normal.
    nfp : int
        Number of field periods.
    rs, zc : array-like, optional
        R-sine and Z-cosine coefficient arrays. These are used when evaluating
        the input surface if supplied, but this initial helper projects the
        output onto the stellarator-symmetric ``rc``/``zs`` basis.
    stellsym : bool, optional
        Present for call-site clarity. The current implementation always
        returns correctly shaped zero arrays for output ``rs`` and ``zc``.
    mpol_out, ntor_out : int, optional
        Output Fourier resolution. Defaults to the input resolution.
    ntheta, nphi : int, optional
        Real-space fitting grid size.
    normal_sign : float, optional
        Multiplier for the normal direction. Use ``-1`` if comparing against a
        convention with the opposite normal orientation.

    Returns
    -------
    tuple
        ``(rc_offset, zs_offset, rs_offset, zc_offset)``.
    """
    rc = jnp.asarray(rc)
    zs = jnp.asarray(zs)
    if rc.shape != zs.shape:
        raise ValueError(f"rc and zs must have the same shape, got "
                         f"{rc.shape} and {zs.shape}.")
    if rc.ndim != 2 or rc.shape[1] % 2 != 1:
        raise ValueError("Coefficient arrays must have shape "
                         "(mpol + 1, 2 * ntor + 1).")

    mpol_in = rc.shape[0] - 1
    ntor_in = _ntor_from_coefficients(rc)
    mpol_out = mpol_in if mpol_out is None else int(mpol_out)
    ntor_out = ntor_in if ntor_out is None else int(ntor_out)

    if rs is not None:
        rs = jnp.asarray(rs)
        if rs.shape != rc.shape:
            raise ValueError(f"rs shape {rs.shape} does not match rc shape "
                             f"{rc.shape}.")
    if zc is not None:
        zc = jnp.asarray(zc)
        if zc.shape != rc.shape:
            raise ValueError(f"zc shape {zc.shape} does not match rc shape "
                             f"{rc.shape}.")

    radius, height, offset_phi_grid, theta_grid = _evaluate_offset_grid(
        rc, zs, offset, nfp, rs, zc, ntheta, nphi, normal_sign)

    rc_modes = _symmetric_rc_modes(mpol_out, ntor_out)
    zs_modes = _symmetric_zs_modes(mpol_out, ntor_out)
    rc_offset = _fit_coefficients(
        radius, offset_phi_grid, theta_grid, rc_modes, mpol_out, ntor_out, nfp,
        "cos")
    zs_offset = _fit_coefficients(
        height, offset_phi_grid, theta_grid, zs_modes, mpol_out, ntor_out, nfp,
        "sin")

    rs_offset = jnp.zeros_like(rc_offset)
    zc_offset = jnp.zeros_like(zs_offset)
    return rc_offset, zs_offset, rs_offset, zc_offset



def surface_coefficients_from_dofs(dofs, mpol, ntor):
    num_n_modes = 2 * ntor + 1
    num_r_dofs = ntor + 1 + mpol * num_n_modes
    expected = num_n_modes * (2 * mpol + 1)
    if len(dofs) != expected:
        raise ValueError(f"Expected {expected} dofs for mpol={mpol}, ntor={ntor}.")
    rc = jnp.concatenate((jnp.zeros(ntor, dtype=dofs.dtype), dofs[:num_r_dofs]))
    zs = jnp.concatenate((jnp.zeros(ntor + 1, dtype=dofs.dtype), dofs[num_r_dofs:]))
    return rc.reshape((mpol + 1, num_n_modes)), zs.reshape((mpol + 1, num_n_modes))


def points_normals_normal_lengths(dofs, mpol, ntor, nfp, tor_num, pol_num):
    rc, zs = surface_coefficients_from_dofs(dofs, mpol, ntor)
    tor_angles = jnp.linspace(0, 2 * jnp.pi, tor_num, endpoint=False)
    pol_angles = jnp.linspace(0, 2 * jnp.pi, pol_num, endpoint=False)

    def evaluate(tor_angle, pol_angle):
        r = r_surface(rc, nfp, tor_angle, pol_angle)
        z = z_surface(zs, nfp, tor_angle, pol_angle)
        point = jnp.array([r * jnp.cos(tor_angle), r * jnp.sin(tor_angle), z])
        normal = jnp.asarray(surface_normal(rc, zs, nfp, tor_angle, pol_angle))
        lengths = jnp.linalg.norm(normal)
        return point, normal / lengths, lengths, normal

    evaluate_pol = jax.vmap(evaluate, in_axes=(None, 0))
    return jax.vmap(evaluate_pol, in_axes=(0, None))(tor_angles, pol_angles)





def _validate_matching_shape(name, coefficients, reference):
    if coefficients is not None:
        coefficients = jnp.asarray(coefficients)
        if coefficients.shape != reference.shape:
            raise ValueError(f"{name} shape {coefficients.shape} does not "
                             f"match rc shape {reference.shape}.")
    return coefficients


def coefficients_to_dofs(rc, zs, rs=None, zc=None):
    """Convert raw RZ Fourier coefficient arrays to optimizer DOFs.

    This is the inverse layout of
    ``pca_surface_objective.surface_coefficients_from_dofs``. The returned
    vector contains the independent stellarator-symmetric modes:

    - ``rc`` entries for ``m=0, n>=0`` and all ``m>0`` modes.
    - ``zs`` entries for ``m=0, n>0`` and all ``m>0`` modes.

    ``rs`` and ``zc`` are accepted so call sites can pass the four-array tuple
    returned by ``extend_via_normal_jax`` directly, but they are not included in
    the current optimizer DOF convention.
    """
    rc = jnp.asarray(rc)
    zs = jnp.asarray(zs)
    if rc.shape != zs.shape:
        raise ValueError(f"rc and zs must have the same shape, got "
                         f"{rc.shape} and {zs.shape}.")
    if rc.ndim != 2 or rc.shape[1] % 2 != 1:
        raise ValueError("Coefficient arrays must have shape "
                         "(mpol + 1, 2 * ntor + 1).")
    _validate_matching_shape("rs", rs, rc)
    _validate_matching_shape("zc", zc, rc)

    ntor = _ntor_from_coefficients(rc)
    rc_dofs = rc.reshape(-1)[ntor:]
    zs_dofs = zs.reshape(-1)[ntor + 1:]
    return jnp.concatenate((rc_dofs, zs_dofs))


def vmex_boundary_to_dense(R_cos, R_sin, Z_cos, Z_sin, solver_context):
    """Convert VMEX boundary coefficients from (ns, mnmax) to dense (m, n)."""
    modes = solver_context.modes
    m = jnp.asarray(modes.m, dtype=int)
    n = jnp.asarray(modes.n, dtype=int)
    ntor = int(solver_context.resolution.ntor)
    mpol = int(jnp.max(m)) + 1
    shape = (mpol, 2 * ntor + 1)

    scale = (jnp.asarray(solver_context.trig.mscale)[m]
             * jnp.asarray(solver_context.trig.nscale)[jnp.abs(n)])
    rc = jnp.zeros(shape, dtype=R_cos.dtype).at[m, n + ntor].set(R_cos[-1] * scale)
    rs = jnp.zeros(shape, dtype=R_sin.dtype).at[m, n + ntor].set(R_sin[-1] * scale)
    zc = jnp.zeros(shape, dtype=Z_cos.dtype).at[m, n + ntor].set(Z_cos[-1] * scale)
    zs = jnp.zeros(shape, dtype=Z_sin.dtype).at[m, n + ntor].set(Z_sin[-1] * scale)
    return rc, rs, zc, zs




MU0 = 4 * jnp.pi * 1e-7

def reduced_memory_induction_matrix(winding_points, plasma_points,
                                    dipole_normals, plasma_normals):
    difference = winding_points[None, :, :] - plasma_points[:, None, :]
    distance_squared = jnp.sum(difference ** 2, axis=2)
    diff_dot_dipole = jnp.einsum("ijk,jk->ij", difference, dipole_normals)
    diff_dot_plasma = jnp.einsum("ijk,ik->ij", difference, plasma_normals)
    dipole_dot_plasma = jnp.einsum("jk,ik->ij", dipole_normals, plasma_normals)
    return (MU0 / (4 * jnp.pi)) * (
        3 * diff_dot_dipole * diff_dot_plasma
        - distance_squared * dipole_dot_plasma) / distance_squared ** 2.5



def spectral_width(dofs, mode_weights):
    return jnp.sum(mode_weights * dofs ** 2)


def smooth_minimum_distance(winding_points, plasma_points, sharpness):
    distance = jnp.linalg.norm(
        winding_points[:, None, :] - plasma_points[None, :, :], axis=2)
    weights = jax.nn.softmax(-sharpness * distance.reshape(-1))
    return jnp.sum(weights * distance.reshape(-1))


def enclosed_volume(points, normals, tor_num, pol_num):
    integral = jnp.sum(jnp.einsum("ijk,ijk->ij", points, normals))
    return jnp.abs(integral * (2 * jnp.pi / tor_num) * (2 * jnp.pi / pol_num) / 3)






def calc_objectives(dofs, plasma_points, plasma_normals, weights, winding_R_cos):
    tor_num, pol_num = WINDING_NPHI, WINDING_NTHETA
    winding_points, winding_normals, normal_lengths, raw_normals = points_normals_normal_lengths(dofs, winding_R_cos.shape[0] - 1, _ntor_from_coefficients(winding_R_cos), nfp, tor_num, pol_num)
    
    flat_points = winding_points.reshape((-1, 3))
    flat_unitnormals = winding_normals.reshape((-1, 3))
    induction = reduced_memory_induction_matrix(
        flat_points, plasma_points, flat_unitnormals, plasma_normals)
    singular_values = jnp.linalg.svd(induction, compute_uv=False)
    singular_probabilities = singular_values / jnp.sum(singular_values)
    singular_entropy = -jnp.sum(
        singular_probabilities * jnp.log(
            jnp.maximum(singular_probabilities, 1e-300)))

    pca = 1 / jnp.maximum(singular_entropy, 1e-16)
    volume = enclosed_volume(winding_points, raw_normals, tor_num, pol_num)
    spectral = spectral_width(dofs, weights)
    distance = smooth_minimum_distance(flat_points, plasma_points, 300)
    minimum_normal_length = jnp.min(normal_lengths)
    return pca, volume, spectral, distance, minimum_normal_length





def mode_weights(surface):
    weights = []
    for name in surface.dof_names:
        mode = int(name.split("(")[1].split(",")[0])
        weights.append(mode ** 2)
    return jnp.asarray(weights)


def mode_weights_from_coefficients(rc):
    mpol = rc.shape[0] - 1
    ntor = _ntor_from_coefficients(rc)
    mode_matrix = jnp.repeat(
        jnp.arange(mpol + 1, dtype=rc.dtype)[:, None] ** 2,
        2 * ntor + 1,
        axis=1)
    return jnp.concatenate((
        mode_matrix.reshape(-1)[ntor:],
        mode_matrix.reshape(-1)[ntor + 1:]))


surface = SurfaceRZFourier(nfp=2, mpol=inp.mpol, ntor=inp.ntor)
weights = mode_weights(surface)



def winding_surface_objective(equilibrium_state, solver_context):
    # Initial from vmex
    aminor = _aspect_scalars(equilibrium_state, solver_context)[0]
    R_cos, R_sin, Z_cos, Z_sin = _geometry(equilibrium_state, solver_context)[0]
    R_cos, R_sin, Z_cos, Z_sin = vmex_boundary_to_dense(
        R_cos, R_sin, Z_cos, Z_sin, solver_context)

    # Make plamsa points and normals for the winding surface objective
    plasma_dofs = coefficients_to_dofs(R_cos, Z_sin)
    plasma_points, plasma_normals, _, _ = points_normals_normal_lengths(
        plasma_dofs, R_cos.shape[0] - 1, _ntor_from_coefficients(R_cos), nfp,
        WINDING_NPHI, WINDING_NTHETA)
    plasma_points = plasma_points.reshape((-1, 3))
    plasma_normals = plasma_normals.reshape((-1, 3))

    # Make winding surface points and normals for the winding surface objective after extending the plasma surface along its normal
    winding_R_cos, winding_Z_sin, winding_R_sin, winding_Z_cos = extend_via_normal_jax(
        R_cos, Z_sin, aminor, nfp, ntheta=WINDING_NTHETA, nphi=WINDING_NPHI)
    winding_dofs = coefficients_to_dofs(winding_R_cos, winding_Z_sin)
    local_weights = mode_weights_from_coefficients(winding_R_cos)

    scales = calc_objectives(winding_dofs, plasma_points, plasma_normals, local_weights, winding_R_cos)

    def objective(dofs):
        pca, volume, spectral, distance, minimum_normal_length = calc_objectives(
            dofs, plasma_points, plasma_normals, local_weights, winding_R_cos)
        wall = 1 + jnp.tanh((MINIMUM_DISTANCE - distance) / DISTANCE_WALL_SCALE) # Penalize surfaces that are too close to the plasma
        invalid = jnp.square(jnp.maximum(1e-6 - minimum_normal_length, 0)) * 1e12 # Avoid degenerate winding surfaces
        return (PCA_WEIGHT * pca / scales[0] # pca based objective
                - VOLUME_WEIGHT * volume / scales[1] # increase winding surface volume
                + SPECTRAL_WEIGHT * spectral / jnp.maximum(scales[2], 1e-16) # penalize high poloidal spectral content
                + DISTANCE_WALL_WEIGHT * wall + invalid)

    optimizer = minimize(method="L-BFGS-B", fun=objective, maxiter=100)
    result = optimizer.run(winding_dofs)

    _, _, _, opt_distance, _ = calc_objectives(result.params, plasma_points, plasma_normals, local_weights, winding_R_cos)

    return 1 + jnp.tanh(-opt_distance+1)

# Objective function terms
qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
objective_function_terms = [
         (qs, 0.0, 1.0),
         (opt.aspect_ratio, ASPECT_TARGET, 1.0),
         (iota_floor, 0.0, 10.0),
         (opt.magnetic_well, MAGNETIC_WELL_TARGET, 1.0),
         (winding_surface_objective, 0.0, 1.0)
         ]

report = opt.EquilibriumReporter(
    ("QS total", qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("mean iota", opt.mean_iota, ".4f"), ("magnetic well", opt.magnetic_well, ".4f"))
monitor = opt.OptimizationMonitor(stream=None)

# Optimize for QA first, then add the pressure-stability proxy locally.
equilibrium = opt.solve_equilibrium(inp)
# If a RuntimeWarning reports uncertified Jacobian columns, it is expected
# once the optimizer leaves the seed and needs no action: the shipped
# jacobian_adjoint_tol=1e-4 and jacobian_adjoint_maxiter=10 are the measured
# optimum, since ten times that budget moved the Jacobian by 2e-8 and
# certified no extra column. Both are from_tuples arguments; pass
# evaluation_progress=False to drop the per-evaluation timing lines.
for stage, (max_mode, max_nfev) in enumerate(zip(MAX_MODES, MAX_NFEV)):
    print(f"\n===== QA stage, max_mode = {max_mode} =====")
    mpol = max(max_mode + 2, MINIMUM_MPOL)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4)
    stage_terms = objective_function_terms
    problem = opt.VmecProblem.from_tuples(inp, stage_terms, max_mode=max_mode,
        vary_major_radius=VARY_MAJOR_RADIUS, use_ess=True, ess_alpha=ESS_ALPHA,
        restart_from=equilibrium)
    print(f"dof_names = {problem.dof_names}")
    monitor.problem = problem
    if not ci_smoke:
        problem.compile_residual_and_jacobian()
    step = PARAMETER_STEP * problem.scales
    result = least_squares(
        problem.residual, problem.x0, jac=problem.residual_jac,
        x_scale=step,max_nfev=max_nfev, bounds=(
                problem.x0 - MAX_PARAMETER_CHANGE * step,
                problem.x0 + MAX_PARAMETER_CHANGE * step),
        ftol=1e-6, xtol=1e-10, verbose=2, callback=monitor
    )
    inp = problem.input_from_x(result.x)
    equilibrium = problem.equilibrium_from_x(result.x)
    report(f"mode {max_mode}", equilibrium)
    # inp.to_indata(f"input.QA_max_mode_{max_mode:03d}")

# Print results
final_input = replace(inp,
    ns_array=np.array([31 if ci_smoke else 101]),
    ftol_array=np.array([1.0e-10 if ci_smoke else 1.0e-14]),
    niter_array=np.array([8000]))
final_equilibrium = opt.solve_equilibrium(
    final_input, initial_state=equilibrium.solution,
    verbose=not ci_smoke, raise_on_max_iterations=True)
final_total = report("final", final_equilibrium)["QS total"]
print(f"\nQS total {final_total:.3e}")

vacuum_name = "QA_optimized"
vacuum_input_path = final_input.to_indata(f"input.{vacuum_name}")
vacuum_wout_path = vj.write_wout(f"wout_{vacuum_name}.nc", final_equilibrium.wout)
print(f"wrote {vacuum_input_path}\nwrote {vacuum_wout_path}")

# Plot results
monitor.save("QA_optimization_objectives.csv")
monitor.plot("QA_optimization_objectives.png")
for path in vj.plot_wout(vacuum_wout_path, ".").values():
    print(f"wrote {path}")
