#!/usr/bin/env python
"""Quasi-axisymmetric optimization with a winding-surface complexity proxy.

Performance and derivative-fidelity evidence for this nested objective lives in
``benchmarks/winding_surface_optimization.md``.
"""

import functools
import os
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jaxopt import LBFGSB
from jaxopt.implicit_diff import root_jvp
from scipy.optimize import least_squares
from solvax import gcrot

import vmex as vj
from vmex import optimize as opt
from vmex.core.solver import _geometry
from vmex.core.statephysics import _aspect_scalars

#os.environ["XLA_FLAGS"] = "--xla_cpu_parallel_codegen_split_count=2"  # default is 32; cap
                                                                        # concurrent LLVM codegen
                                                                        # workers to bound peak
                                                                        # compile-time memory
#jax.config.update("jax_enable_compilation_cache", False)

# --- run-control / mode-ladder ---
MAX_MODES, MAX_NFEV = [1,], [10,]
# MAX_MODES, MAX_NFEV = [1,2,3,4,5,6,7,8,9], [10, 10, 15, 20, 25, 40, 50, 60, 60]
MINIMUM_MPOL = 3
VARY_MAJOR_RADIUS = False  # set True to optimize RBC(0,0) instead of fixing it
PARAMETER_STEP, MAX_PARAMETER_CHANGE = 0.02, 5.0
ESS_ALPHA = 1.2  # smaller values let high Fourier modes move more

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODES, MAX_NFEV = [1], [4]

FINAL_NS = 31 if ci_smoke else 51
FINAL_FTOL = 1.0e-10 if ci_smoke else 1.0e-12
FINAL_NITER = 8000

# --- equilibrium seed setup ---
nfp = 2  # number of field periods
SEED_PERTURBATION = 0.05

DATA = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{nfp}"
inp = vj.VmecInput.from_file(DATA)
# The exactly circular torus has zero first-order iota sensitivity. This
# explicit rotating-ellipse perturbation gives the local optimizer a QA basin.
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
inp = replace(inp, rbc=rbc, zbs=zbs)

# --- dormant QA/magnetic-well objective constants (their terms are
# commented out in objective_function_terms below; kept live so any of the
# four legacy terms -- qs, aspect_ratio, iota_floor, magnetic_well -- can be
# re-enabled by uncommenting a single line there) ---
SURFACES = np.linspace(0.1, 1.0, 10)
MAGNETIC_WELL_TARGET = 0.01
ASPECT_TARGET = 5.0
# MAGNETIC_WELL_TARGET = 0.07
# ASPECT_TARGET = 3.5
IOTA_FLOOR = 0.42

# --- physical constant ---
MU0 = 4 * jnp.pi * 1e-7

# --- winding-surface objective weights ---
PCA_WEIGHT = 1.0
SINGULAR_STRENGTH_WEIGHT = 0.0  # inactive (as in the ESSOS reference); kept live for completeness
VOLUME_WEIGHT = 0.02
SPECTRAL_WEIGHT = 0.02
DISTANCE_WALL_WEIGHT = 10.0
SELF_INTERSECTION_WEIGHT = 100.0

# --- winding-surface guardrail thresholds ---
MINIMUM_DISTANCE = 0.05
DISTANCE_WALL_SCALE = 0.01
MINIMUM_SELF_RADIUS = 0.05
SELF_RADIUS_WALL_SCALE = 0.01
SHARPNESS = 300.0  # shared smooth-min sharpness for the distance and self-intersection walls
SELF_NEIGHBOR_RADIUS = 2
MINIMUM_NORMAL_LENGTH = 1e-6  # degenerate-surface guard threshold (see _winding_objective_value)
INVALID_PENALTY_SCALE = 1e12  # degenerate-surface guard penalty scale

# --- winding-surface structural/solver constants ---
ACTIVE_MPOL = 2
ACTIVE_NTOR = 2
COEFFICIENT_STEP_BOUND = 1.0
WINDING_NTHETA = 16
WINDING_NPHI = 16
WINDING_MAXITER = 100
WINDING_OPTIMALITY_TOL = 1e-6
WINDING_LINEAR_RTOL = 1e-8
WINDING_LINEAR_MAX_RESTARTS = 10

# --- numerical-safety epsilon floor ---
DENOMINATOR_EPS = 1e-16  # shared division-by-zero guard for calc_objectives' scale-normalized terms


def _mode_angle(num_m_modes, num_n_modes, nfp, tor_angle, pol_angle, n_min, m_min):
    """(m, n)-grid of ``(m_min+m)*pol_angle - (n+n_min)*nfp*tor_angle``.

    Shared by every Fourier-sum helper below so each one is a single
    vectorized contraction against this basis instead of an unrolled Python
    double loop that indexes the coefficient array one (m, n) pair at a
    time -- the loop form is differentiated (jaxopt's LBFGS gradient step,
    plus this script's own root_jvp) many times over in nested contexts,
    and reverse-mode AD through dozens of chained single-element gathers is
    what actually drove the resolution-independent compile blowup, not the
    grid size or LBFGS maxiter.
    """
    m_idx = jnp.arange(num_m_modes) + m_min
    n_idx = jnp.arange(num_n_modes) + n_min
    return m_idx[:, None] * pol_angle - n_idx[None, :] * nfp * tor_angle, m_idx, n_idx


def r_surface(rsurfacecc, nfp, tor_angle, pol_angle, rsurfacecs = None, n_min = None, m_min = 0):
    rsurfacecc = jnp.asarray(rsurfacecc)
    num_m_modes, num_n_modes = rsurfacecc.shape
    if n_min is None:
        n_min = -(num_n_modes - 1)/2
    angle, _, _ = _mode_angle(num_m_modes, num_n_modes, nfp, tor_angle, pol_angle, n_min, m_min)
    r = jnp.sum(rsurfacecc * jnp.cos(angle))
    if rsurfacecs is not None:
        rsurfacecs = jnp.asarray(rsurfacecs)
        assert rsurfacecc.shape == rsurfacecs.shape, 'If rsurfacecs is specified, it must be the same length as rsurfacecc.'
        r = r + jnp.sum(rsurfacecs * jnp.sin(angle))
    return r

def z_surface(zsurfacecs, nfp, tor_angle, pol_angle, zsurfacecc = None, n_min = None, m_min=0):
    zsurfacecs = jnp.asarray(zsurfacecs)
    num_m_modes, num_n_modes = zsurfacecs.shape
    if n_min is None:
        n_min = -(num_n_modes - 1)/2
    angle, _, _ = _mode_angle(num_m_modes, num_n_modes, nfp, tor_angle, pol_angle, n_min, m_min)
    z = jnp.sum(zsurfacecs * jnp.sin(angle))
    if zsurfacecc is not None:
        zsurfacecc = jnp.asarray(zsurfacecc)
        assert zsurfacecs.shape == zsurfacecc.shape, 'If zsurfacecc is specified, it must be the same length as zsurfacecs'
        z = z + jnp.sum(zsurfacecc * jnp.cos(angle))
    return z

def r_surface_prime_phi(rsurfacecc, nfp, tor_angle, pol_angle, rsurfacecs = None, n_min = None, m_min = 0):
    rsurfacecc = jnp.asarray(rsurfacecc)
    num_m_modes, num_n_modes = rsurfacecc.shape
    if n_min is None:
        n_min = -(num_n_modes - 1)/2
    angle, _, n_idx = _mode_angle(num_m_modes, num_n_modes, nfp, tor_angle, pol_angle, n_min, m_min)
    coef_n = n_idx[None, :] * nfp
    r_phi = jnp.sum(rsurfacecc * coef_n * jnp.sin(angle))
    if rsurfacecs is not None:
        rsurfacecs = jnp.asarray(rsurfacecs)
        assert rsurfacecc.shape == rsurfacecs.shape, 'If rsurfacecs is specified, it must be the same length as rsurfacecc.'
        r_phi = r_phi + jnp.sum(-rsurfacecs * coef_n * jnp.cos(angle))
    return r_phi

def z_surface_prime_phi(zsurfacecs, nfp, tor_angle, pol_angle, zsurfacecc = None, n_min = None, m_min=0):
    zsurfacecs = jnp.asarray(zsurfacecs)
    num_m_modes, num_n_modes = zsurfacecs.shape
    if n_min is None:
        n_min = -(num_n_modes - 1)/2
    angle, _, n_idx = _mode_angle(num_m_modes, num_n_modes, nfp, tor_angle, pol_angle, n_min, m_min)
    coef_n = n_idx[None, :] * nfp
    z_phi = jnp.sum(-zsurfacecs * coef_n * jnp.cos(angle))
    if zsurfacecc is not None:
        zsurfacecc = jnp.asarray(zsurfacecc)
        assert zsurfacecs.shape == zsurfacecc.shape, 'If zsurfacecc is specified, it must be the same length as zsurfacecs'
        z_phi = z_phi + jnp.sum(zsurfacecc * coef_n * jnp.sin(angle))
    return z_phi

def surface_del_phi(rsurfacecc, zsurfacecs, nfp, tor_angle, pol_angle, rsurfacecs = None, zsurfacecc = None, n_min = None, m_min = 0):
    r = r_surface(rsurfacecc, nfp, tor_angle, pol_angle, rsurfacecs=rsurfacecs, n_min=n_min, m_min=m_min)
    r_phi = r_surface_prime_phi(rsurfacecc, nfp, tor_angle, pol_angle, rsurfacecs=rsurfacecs, n_min=n_min, m_min=m_min)
    z_phi = z_surface_prime_phi(zsurfacecs, nfp, tor_angle, pol_angle, zsurfacecc=zsurfacecc, n_min=n_min, m_min=m_min)
    costerm = jnp.cos(tor_angle)
    sinterm = jnp.sin(tor_angle)
    return [r_phi*costerm - r*sinterm, r_phi*sinterm + r*costerm, z_phi]

def r_surface_prime_theta(rsurfacecc, nfp, tor_angle, pol_angle, rsurfacecs = None, n_min = None, m_min = 0):
    rsurfacecc = jnp.asarray(rsurfacecc)
    num_m_modes, num_n_modes = rsurfacecc.shape
    if n_min is None:
        n_min = -(num_n_modes - 1)/2
    angle, m_idx, _ = _mode_angle(num_m_modes, num_n_modes, nfp, tor_angle, pol_angle, n_min, m_min)
    coef_m = m_idx[:, None]
    r_theta = jnp.sum(-rsurfacecc * coef_m * jnp.sin(angle))
    if rsurfacecs is not None:
        rsurfacecs = jnp.asarray(rsurfacecs)
        assert rsurfacecc.shape == rsurfacecs.shape, 'If rsurfacecs is specified, it must be the same length as rsurfacecc.'
        r_theta = r_theta + jnp.sum(rsurfacecs * coef_m * jnp.cos(angle))
    return r_theta

def z_surface_prime_theta(zsurfacecs, nfp, tor_angle, pol_angle, zsurfacecc = None, n_min = None, m_min=0):
    zsurfacecs = jnp.asarray(zsurfacecs)
    num_m_modes, num_n_modes = zsurfacecs.shape
    if n_min is None:
        n_min = -(num_n_modes - 1)/2
    angle, m_idx, _ = _mode_angle(num_m_modes, num_n_modes, nfp, tor_angle, pol_angle, n_min, m_min)
    coef_m = m_idx[:, None]
    z_theta = jnp.sum(zsurfacecs * coef_m * jnp.cos(angle))
    if zsurfacecc is not None:
        zsurfacecc = jnp.asarray(zsurfacecc)
        assert zsurfacecs.shape == zsurfacecc.shape, 'If zsurfacecc is specified, it must be the same length as zsurfacecs'
        z_theta = z_theta + jnp.sum(-zsurfacecc * coef_m * jnp.sin(angle))
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


def _active_dof_indices(mpol, ntor, active_mpol, active_ntor):
    """Positions in a ``coefficients_to_dofs``-packed vector with ``m <=
    active_mpol`` and ``|n| <= active_ntor``.

    Reuses ``_symmetric_rc_modes``/``_symmetric_zs_modes`` since both were
    built from the same (m, n) ordering ``coefficients_to_dofs`` packs its
    output in -- no new mode-layout logic needed. Returns a plain Python
    tuple (static, hashable) so it can be a ``jax.custom_jvp``
    ``nondiff_argnums`` value.
    """
    def active(m, n):
        return m <= active_mpol and abs(n) <= active_ntor
    rc_active = [active(m, n) for m, n in _symmetric_rc_modes(mpol, ntor)]
    zs_active = [active(m, n) for m, n in _symmetric_zs_modes(mpol, ntor)]
    return tuple(i for i, is_active in enumerate(rc_active + zs_active) if is_active)


def surface_coefficients_from_dofs(dofs, mpol, ntor):
    num_n_modes = 2 * ntor + 1
    num_r_dofs = ntor + 1 + mpol * num_n_modes
    expected = num_n_modes * (2 * mpol + 1)
    if len(dofs) != expected:
        raise ValueError(f"Expected {expected} dofs for mpol={mpol}, ntor={ntor}.")
    rc = jnp.concatenate((jnp.zeros(ntor, dtype=dofs.dtype), dofs[:num_r_dofs]))
    zs = jnp.concatenate((jnp.zeros(ntor + 1, dtype=dofs.dtype), dofs[num_r_dofs:]))
    return rc.reshape((mpol + 1, num_n_modes)), zs.reshape((mpol + 1, num_n_modes))


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


def vmex_boundary_to_dense(R_cos, R_sin, Z_cos, Z_sin, solver_context):
    """Convert VMEX boundary coefficients from (ns, mnmax) to dense (m, n)."""
    modes = solver_context.modes
    m = jnp.asarray(modes.m, dtype=int)
    n = jnp.asarray(modes.n, dtype=int)
    ntor = int(solver_context.resolution.ntor)
    mpol = int(solver_context.resolution.mpol)
    shape = (mpol, 2 * ntor + 1)

    scale = (jnp.asarray(solver_context.trig.mscale)[m]
             * jnp.asarray(solver_context.trig.nscale)[jnp.abs(n)])
    rc = jnp.zeros(shape, dtype=R_cos.dtype).at[m, n + ntor].set(R_cos[-1] * scale)
    rs = jnp.zeros(shape, dtype=R_sin.dtype).at[m, n + ntor].set(R_sin[-1] * scale)
    zc = jnp.zeros(shape, dtype=Z_cos.dtype).at[m, n + ntor].set(Z_cos[-1] * scale)
    zs = jnp.zeros(shape, dtype=Z_sin.dtype).at[m, n + ntor].set(Z_sin[-1] * scale)
    return rc, rs, zc, zs


def surface_quadrature_weights(normal_lengths, nphi, ntheta):
    """Periodic midpoint-rule weights, including the surface Jacobian.

    Vmex's plasma/winding grids use ``linspace(..., endpoint=False)`` -- no
    duplicated closing point -- so this reduces to a uniform angular cell
    size times the local Jacobian magnitude (``normal_lengths``, the
    un-normalized surface normal's length from
    ``points_normals_normal_lengths``).
    """
    return normal_lengths.reshape(-1) * (2 * jnp.pi / ntheta) * (2 * jnp.pi / nphi)


def reduced_memory_induction_matrix(winding_points, plasma_points,
                                    dipole_normals, plasma_normals,
                                    winding_weights, plasma_weights):
    difference = winding_points[None, :, :] - plasma_points[:, None, :]
    distance_squared = jnp.sum(difference ** 2, axis=2)
    diff_dot_dipole = jnp.einsum("ijk,jk->ij", difference, dipole_normals)
    diff_dot_plasma = jnp.einsum("ijk,ik->ij", difference, plasma_normals)
    dipole_dot_plasma = jnp.einsum("jk,ik->ij", dipole_normals, plasma_normals)
    kernel = (MU0 / (4 * jnp.pi)) * (
        3 * diff_dot_dipole * diff_dot_plasma
        - distance_squared * dipole_dot_plasma) / distance_squared ** 2.5
    return jnp.sqrt(plasma_weights[:, None]) * kernel * jnp.sqrt(winding_weights[None, :])


def _periodic_induction_singular_values(
        winding_points, plasma_points, dipole_normals, plasma_normals,
        winding_weights, plasma_weights, field_periods, tor_num, pol_num):
    """Return the exact full-torus spectrum from one block row.

    Uniform full-torus grids of field-periodic surfaces make the weighted
    induction matrix block circulant.  A block discrete Fourier transform
    therefore splits an ``(field_periods * block_size)`` SVD into
    ``field_periods`` independent ``block_size`` SVDs without discarding any
    Fourier sector.  Only one target-period block row is assembled, which also
    reduces kernel construction by ``field_periods``.

    If the toroidal grid cannot be divided evenly into field periods, this
    helper falls back to the full matrix.  The result is consequently exact
    for every resolution, with symmetry reduction used whenever the sampling
    admits it.
    """
    if field_periods <= 1 or tor_num % field_periods:
        induction = reduced_memory_induction_matrix(
            winding_points, plasma_points, dipole_normals, plasma_normals,
            winding_weights, plasma_weights)
        return jnp.linalg.svd(induction, compute_uv=False)

    block_size = (tor_num // field_periods) * pol_num
    first_block_row = reduced_memory_induction_matrix(
        winding_points, plasma_points[:block_size], dipole_normals,
        plasma_normals[:block_size], winding_weights,
        plasma_weights[:block_size])
    blocks = jnp.transpose(
        first_block_row.reshape((block_size, field_periods, block_size)),
        (1, 0, 2))

    if field_periods == 2:
        # Keep the two NFP=2 sectors real; a complex FFT is unnecessary here.
        sectors = jnp.stack((blocks[0] + blocks[1],
                             blocks[0] - blocks[1]))
    else:
        sectors = jnp.fft.fft(blocks, axis=0)
    spectra = jax.vmap(
        lambda sector: jnp.linalg.svd(sector, compute_uv=False))(sectors)
    return spectra.reshape(-1)


def spectral_width(dofs, mode_weights):
    return jnp.sum(mode_weights * dofs ** 2)


def smooth_minimum_distance(
        winding_points, plasma_points, sharpness, *, source_count=None):
    if source_count is not None:
        winding_points = winding_points[:source_count]
    distance = jnp.linalg.norm(
        winding_points[:, None, :] - plasma_points[None, :, :], axis=2)
    weights = jax.nn.softmax(-sharpness * distance.reshape(-1))
    return jnp.sum(weights * distance.reshape(-1))


def smooth_minimum_tangent_radius(
        points, normals, nphi, ntheta, sharpness, neighbor_radius,
        *, source_phi_count=None):
    """Nonlocal surface thickness; nearby parameter-grid points are excluded.

    A smooth-min "tangent radius" between all pairs of winding-surface grid
    points that are far apart in (theta, phi) index space -- guards against
    the surface folding back close to itself at a distance, distinct from
    the local degenerate-normal/area-element guard below.
    """
    if source_phi_count is None:
        source_phi_count = nphi
    source_count = source_phi_count * ntheta
    difference = points[None, :, :] - points[:source_count, None, :]
    distance_squared = jnp.sum(difference ** 2, axis=2)
    tangent_radius = distance_squared / (
        2 * jnp.abs(jnp.einsum(
            "ijk,ik->ij", difference, normals[:source_count])) + 1e-14)

    iphi, itheta = jnp.meshgrid(jnp.arange(nphi), jnp.arange(ntheta), indexing="ij")
    dphi = jnp.abs(
        iphi.reshape(-1, 1)[:source_count] - iphi.reshape(1, -1))
    dtheta = jnp.abs(
        itheta.reshape(-1, 1)[:source_count] - itheta.reshape(1, -1))
    dphi = jnp.minimum(dphi, nphi - dphi)
    dtheta = jnp.minimum(dtheta, ntheta - dtheta)
    nonlocal_pair = (dphi > neighbor_radius) | (dtheta > neighbor_radius)
    tangent_radius = jnp.where(nonlocal_pair, tangent_radius, 1e6)
    weights = jax.nn.softmax(-sharpness * tangent_radius.reshape(-1))
    return jnp.sum(weights * tangent_radius.reshape(-1))


def enclosed_volume(points, normals, tor_num, pol_num):
    integral = jnp.sum(jnp.einsum("ijk,ijk->ij", points, normals))
    return jnp.abs(integral * (2 * jnp.pi / tor_num) * (2 * jnp.pi / pol_num) / 3)


def _calc_objectives(dofs, plasma_points, plasma_normals, weights,
                     winding_R_cos, plasma_weights, *, periodic=False):
    tor_num, pol_num = WINDING_NPHI, WINDING_NTHETA
    winding_points, winding_normals, normal_lengths, raw_normals = points_normals_normal_lengths(dofs, winding_R_cos.shape[0] - 1, _ntor_from_coefficients(winding_R_cos), nfp, tor_num, pol_num)

    flat_points = winding_points.reshape((-1, 3))
    flat_unitnormals = winding_normals.reshape((-1, 3))
    winding_weights = surface_quadrature_weights(normal_lengths, tor_num, pol_num)
    if periodic:
        singular_values = _periodic_induction_singular_values(
            flat_points, plasma_points, flat_unitnormals, plasma_normals,
            winding_weights, plasma_weights, nfp, tor_num, pol_num)
    else:
        induction = reduced_memory_induction_matrix(
            flat_points, plasma_points, flat_unitnormals, plasma_normals,
            winding_weights, plasma_weights)
        singular_values = jnp.linalg.svd(induction, compute_uv=False)
    singular_probabilities = singular_values / jnp.sum(singular_values)
    singular_entropy = -jnp.sum(
        singular_probabilities * jnp.log(
            jnp.maximum(singular_probabilities, 1e-300)))

    pca = 1 / jnp.maximum(singular_entropy, DENOMINATOR_EPS)
    singular_strength = jnp.sum(singular_values)
    periodic_source_phi_count = None
    periodic_source_count = None
    if periodic and nfp > 1 and tor_num % nfp == 0:
        # Pairwise scalar reductions contain ``nfp`` symmetry-equivalent
        # source periods.  Retaining one source period preserves their exact
        # weighted mean while reducing both pairwise tensors by ``nfp``.
        periodic_source_phi_count = tor_num // nfp
        periodic_source_count = periodic_source_phi_count * pol_num
    volume = enclosed_volume(winding_points, raw_normals, tor_num, pol_num)
    spectral = spectral_width(dofs, weights)
    distance = smooth_minimum_distance(
        flat_points, plasma_points, SHARPNESS,
        source_count=periodic_source_count)
    self_radius = smooth_minimum_tangent_radius(
        flat_points, flat_unitnormals, tor_num, pol_num, SHARPNESS,
        SELF_NEIGHBOR_RADIUS,
        source_phi_count=periodic_source_phi_count)
    minimum_normal_length = jnp.min(normal_lengths)
    return (pca, volume, spectral, distance, minimum_normal_length,
            singular_strength, self_radius)


def calc_objectives(dofs, plasma_points, plasma_normals, weights,
                    winding_R_cos, plasma_weights):
    """Evaluate objectives for unrestricted sampled plasma geometry."""
    return _calc_objectives(
        dofs, plasma_points, plasma_normals, weights, winding_R_cos,
        plasma_weights)


def _periodic_objectives(dofs, plasma_dofs, weights, winding_R_cos):
    """Evaluate objectives on the field-periodic coefficient domain.

    Constructing the plasma geometry here ensures that every differentiated
    perturbation preserves field-period symmetry.  That restriction is what
    makes the block-DFT spectrum exact for gradients and Hessian-vector
    products as well as values.
    """
    tor_num, pol_num = WINDING_NPHI, WINDING_NTHETA
    plasma_points, plasma_normals, plasma_normal_lengths, _ = (
        points_normals_normal_lengths(
            plasma_dofs, winding_R_cos.shape[0] - 1,
            _ntor_from_coefficients(winding_R_cos), nfp, tor_num, pol_num))
    plasma_points = plasma_points.reshape((-1, 3))
    plasma_normals = plasma_normals.reshape((-1, 3))
    plasma_weights = surface_quadrature_weights(
        plasma_normal_lengths, tor_num, pol_num)
    return _calc_objectives(
        dofs, plasma_points, plasma_normals, weights, winding_R_cos,
        plasma_weights, periodic=True)


def _winding_objective_value(dofs, plasma_points, plasma_normals, plasma_weights,
                             local_weights, winding_R_cos, scales):
    objectives = calc_objectives(
        dofs, plasma_points, plasma_normals, local_weights, winding_R_cos,
        plasma_weights)
    return _combine_winding_objectives(objectives, scales)


def _periodic_winding_objective_value(
        dofs, plasma_dofs, local_weights, winding_R_cos, scales):
    """Winding objective restricted to physical field-periodic geometry."""
    objectives = _periodic_objectives(
        dofs, plasma_dofs, local_weights, winding_R_cos)
    return _combine_winding_objectives(objectives, scales)


def _combine_winding_objectives(objectives, scales):
    """Apply normalized weights and guardrails to objective components."""
    (pca, volume, spectral, distance, minimum_normal_length,
     singular_strength, self_radius) = objectives
    # The reference scales are recomputed once per outer call, so their
    # sensitivity is part of the total derivative of the executable objective.
    # Reverse mode evaluates that correction in one cotangent pass rather than
    # once per boundary degree of freedom.
    wall = 1 + jnp.tanh((MINIMUM_DISTANCE - distance) / DISTANCE_WALL_SCALE) # Penalize surfaces that are too close to the plasma
    self_intersection_wall = 1 + jnp.tanh(
        (MINIMUM_SELF_RADIUS - self_radius) / SELF_RADIUS_WALL_SCALE) # Penalize nonlocal self-approach
    invalid = jnp.square(jnp.maximum(MINIMUM_NORMAL_LENGTH - minimum_normal_length, 0)) * INVALID_PENALTY_SCALE # Avoid degenerate winding surfaces
    return (PCA_WEIGHT * pca / scales[0] # pca based objective
            + SINGULAR_STRENGTH_WEIGHT * jnp.square(
                jnp.maximum(1 - singular_strength / jnp.maximum(scales[5], DENOMINATOR_EPS), 0)) # prevent global operator weakening (inactive: weight 0)
            - VOLUME_WEIGHT * volume / scales[1] # increase winding surface volume
            + SPECTRAL_WEIGHT * spectral / jnp.maximum(scales[2], DENOMINATOR_EPS) # penalize high poloidal spectral content
            + DISTANCE_WALL_WEIGHT * wall + invalid
            + SELF_INTERSECTION_WEIGHT * self_intersection_wall) # prevent nonlocal self-intersection


def _active_winding_objective_value(active_dofs, full_dofs, active_indices,
                                    plasma_dofs, local_weights, winding_R_cos,
                                    scales):
    """Periodic winding objective restricted to the active-mode subset.

    ``full_dofs`` is the frozen (analytic offset-surface) baseline for every
    mode; only the modes at ``active_indices`` (``m <= ACTIVE_MPOL``,
    ``|n| <= ACTIVE_NTOR``) are actually varied by the inner solve, matching
    the ESSOS reference. ``active_indices`` is a plain Python tuple (static,
    hashable) rather than a JAX array precisely so it can be a
    ``nondiff_argnums`` value below -- it selects positions, it is never
    itself differentiated.
    """
    dofs = full_dofs.at[jnp.array(active_indices)].set(active_dofs)
    return _periodic_winding_objective_value(
        dofs, plasma_dofs, local_weights, winding_R_cos, scales)


def _active_dof_bounds(init_active_dofs):
    """Box bounds for the active DOFs: initial value +/- COEFFICIENT_STEP_BOUND.

    Centered on a stop_gradient'd copy of the initial active DOFs: the box is
    an engineering guardrail against runaway coefficient changes during the
    inner solve, not a physically tracked quantity, so its own sensitivity to
    the outer equilibrium is deliberately zero (validated against finite
    differences in a standalone toy problem before this was wired in, in both
    the bound-inactive and bound-active regimes).
    """
    center = jax.lax.stop_gradient(init_active_dofs)
    return center - COEFFICIENT_STEP_BOUND, center + COEFFICIENT_STEP_BOUND


@functools.partial(jax.custom_jvp, nondiff_argnums=(2,))
def _solve_winding_surface(init_active_dofs, full_dofs, active_indices,
                          plasma_dofs, local_weights, winding_R_cos, scales):
    # implicit_diff defaults to True here (unlike the earlier implicit_diff=False
    # attempt): this keeps jaxopt's compact jax.lax.while_loop forward solve
    # (compiled size independent of maxiter) instead of unrolling 100 steps into
    # every traced residual/Jacobian evaluation. The custom_jvp below supplies
    # the (forward-mode-compatible) sensitivity in place of jaxopt's own
    # reverse-mode-only custom_vjp rule, which VMEX's forward Jacobian paths
    # cannot differentiate through.  The custom linear solve used by the JVP
    # is explicitly transposable, so JAX can derive the scalar reverse/adjoint
    # path from the same rule without unrolling optimizer iterations.
    #
    # LBFGSB (not plain LBFGS) box-constrains each active dof to its initial
    # value +/- COEFFICIENT_STEP_BOUND, matching the ESSOS reference's SciPy
    # L-BFGS-B bounds. active_indices stays a closure variable here (not
    # threaded through jaxopt's own *args) since it is static and jaxopt's
    # machinery only needs to know about genuinely differentiated arguments.
    optimizer = LBFGSB(
        fun=lambda a, f, *rest: _active_winding_objective_value(
            a, f, active_indices, *rest),
        maxiter=WINDING_MAXITER, tol=WINDING_OPTIMALITY_TOL)
    bounds = _active_dof_bounds(init_active_dofs)
    return optimizer.run(
        init_active_dofs, bounds, full_dofs, plasma_dofs, local_weights,
        winding_R_cos, scales).params


def _winding_linear_solve(
        matvec, rhs, *, rtol=WINDING_LINEAR_RTOL,
        max_restarts=WINDING_LINEAR_MAX_RESTARTS):
    """Solve and certify the original optimality-Jacobian system.

    JAXopt's default ``root_jvp`` solver applies CG to normal equations,
    squaring the condition number and requiring both forward and transpose
    Hessian-vector products in every iteration.  GCROT instead solves the
    original system.  ``custom_linear_solve`` supplies its transpose solve to
    reverse-mode AD without differentiating through the Krylov iterations.
    """
    _, operator = jax.linearize(matvec, jnp.zeros_like(rhs))

    def solve(action, value):
        result = gcrot(
            action, value, rtol=rtol, max_restarts=max_restarts)
        tolerance = rtol * jnp.linalg.norm(value)
        # Solvax recomputes the true residual before returning ``gcrot``.
        # Reuse that certified norm instead of applying this (expensive
        # Hessian) operator once more solely for the same check.
        certified = result.converged & (
            result.residual_norm <= tolerance)
        return jnp.where(certified, result.x, jnp.nan)

    return jax.lax.custom_linear_solve(
        operator, rhs, solve=solve, transpose_solve=solve)


@_solve_winding_surface.defjvp
def _solve_winding_surface_jvp(active_indices, primals, tangents):
    init_active_dofs, full_dofs = primals[0], primals[1]
    rest_primals, rest_tangents = primals[2:], tangents[2:]
    sol = _solve_winding_surface(init_active_dofs, full_dofs, active_indices, *rest_primals)

    optimizer = LBFGSB(
        fun=lambda a, f, *rest: _active_winding_objective_value(
            a, f, active_indices, *rest),
        maxiter=WINDING_MAXITER, tol=WINDING_OPTIMALITY_TOL)
    bounds = _active_dof_bounds(init_active_dofs)
    zero_bounds_tangent = jax.tree.map(jnp.zeros_like, bounds)

    sol_tangent = root_jvp(
        optimality_fun=optimizer.optimality_fun, sol=sol,
        args=(bounds, full_dofs, *rest_primals),
        tangents=(zero_bounds_tangent, tangents[1], *rest_tangents),
        solve=_winding_linear_solve)
    # Match LBFGSB's infinity-norm stopping criterion.  An L2 check would grow
    # with the number of active modes and could reject a solve that the
    # optimizer correctly certified after a future resolution increase.
    stationary = jnp.max(jnp.abs(optimizer.optimality_fun(
        sol, bounds, full_dofs, *rest_primals))) <= WINDING_OPTIMALITY_TOL
    return sol, jnp.where(stationary, sol_tangent, jnp.nan)


@jax.jit
def winding_surface_objective(equilibrium_state, solver_context):
    # Initial from vmex
    aminor = _aspect_scalars(equilibrium_state, solver_context)[0]
    R_cos, R_sin, Z_cos, Z_sin = _geometry(equilibrium_state, solver_context)[0]
    R_cos, R_sin, Z_cos, Z_sin = vmex_boundary_to_dense(
        R_cos, R_sin, Z_cos, Z_sin, solver_context)

    # Keep plasma geometry in its field-periodic Fourier coefficient space so
    # the exact block-DFT spectrum remains valid under differentiation.
    plasma_dofs = coefficients_to_dofs(R_cos, Z_sin)

    # Make winding surface points and normals for the winding surface objective after extending the plasma surface along its normal
    winding_R_cos, winding_Z_sin, _, _ = extend_via_normal_jax(
        R_cos, Z_sin, aminor, nfp, ntheta=WINDING_NTHETA, nphi=WINDING_NPHI)
    winding_dofs = coefficients_to_dofs(winding_R_cos, winding_Z_sin)
    local_weights = mode_weights_from_coefficients(winding_R_cos)

    scales = _periodic_objectives(
        winding_dofs, plasma_dofs, local_weights, winding_R_cos)

    active_indices = _active_dof_indices(
        winding_R_cos.shape[0] - 1, _ntor_from_coefficients(winding_R_cos),
        ACTIVE_MPOL, ACTIVE_NTOR)
    active_index_array = jnp.array(active_indices)
    init_active_dofs = winding_dofs[active_index_array]

    active_solution = _solve_winding_surface(
        init_active_dofs, winding_dofs, active_indices, plasma_dofs,
        local_weights, winding_R_cos, scales)
    winding_solution = winding_dofs.at[active_index_array].set(active_solution)

    _, _, _, opt_distance, _, _, _ = _periodic_objectives(
        winding_solution, plasma_dofs, local_weights, winding_R_cos)

    return 1 / opt_distance


# Floor the profile minimum, not its average: a mean target is satisfiable while
# an interior surface sits near zero transform, which is what a current-carried
# finite-beta profile does. opt.mean_iota targets the average instead, and
# opt.soft_min_abs_iota is the smooth-minimum variant. Dormant: only referenced
# by a commented-out objective_function_terms entry below (see IOTA_FLOOR).
def iota_floor(equilibrium_state, solver_context):
    return jnp.maximum(
        IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)


# Objective function terms
qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
objective_function_terms = [
         #(qs, 0.0, 1.0),
         #(opt.aspect_ratio, ASPECT_TARGET, 1.0),
         #(iota_floor, 0.0, 10.0),
         #(opt.magnetic_well, MAGNETIC_WELL_TARGET, 1.0),
         (winding_surface_objective, 0.0, 1.0),
         ]

report = opt.EquilibriumReporter(
    ("QS total", qs.total, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
    ("mean iota", opt.mean_iota, ".4f"), ("magnetic well", opt.magnetic_well, ".4f"),
    ("winding objective", winding_surface_objective, ".6e"))
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
        restart_from=equilibrium, implicit_jacobian_method="auto")
    # With one residual row, "auto" uses one reverse equilibrium adjoint.  The
    # winding custom JVP is transposable through _winding_linear_solve, giving
    # a nested adjoint rather than one forward winding response per plasma DOF.
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
    inp.to_indata(f"input.QA_max_mode_{max_mode:03d}")

# Print results
final_input = replace(inp,
    ns_array=np.array([FINAL_NS]),
    ftol_array=np.array([FINAL_FTOL]),
    niter_array=np.array([FINAL_NITER]))
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
